from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_affect_relation import (
    AffectRelationConfig,
    CausalAffectRelation,
    fit_train_vad_auxiliary_loss,
    label_vad_table,
)


EMOTIONTALK_LABELS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "disgusted",
    "fearful",
)
MELD_LABELS = (
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
)


def inputs(*, d_model: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260808)
    return {
        "projected_modalities": torch.randn(2, 5, 3, d_model, generator=generator),
        "valid_mask": torch.ones(2, 5, dtype=torch.bool),
        "history_mask": torch.ones(2, 5, dtype=torch.bool),
        "turn_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 1, 2, 3]], dtype=torch.long),
        "query_indices": torch.tensor([3, 3], dtype=torch.long),
        "modality_mask": torch.ones(2, 5, 3, dtype=torch.bool),
    }


def module(mode: str = "primary_history_relation") -> CausalAffectRelation:
    torch.manual_seed(17)
    value = CausalAffectRelation(
        AffectRelationConfig(
            d_model=16,
            hidden_dim=24,
            dropout=0.0,
            mode=mode,  # type: ignore[arg-type]
        )
    )
    value.eval()
    return value


def test_forward_api_is_outcome_free_and_strictly_causal() -> None:
    assert not {
        "labels",
        "targets",
        "gold",
        "selection_labels",
    } & set(inspect.signature(CausalAffectRelation.forward).parameters)
    model = module()
    original = inputs()
    first = model(**original)
    changed = {name: value.clone() for name, value in original.items()}
    # Row 4 is future to both query rows.  It is maliciously requested in the
    # supplied mask, but the relation module must remove it itself.
    changed["projected_modalities"][:, 4] += 10_000.0
    second = model(**changed)
    assert torch.equal(first.effective_history_mask[:, 4], torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(first.relation_features, second.relation_features)
    assert torch.allclose(first.relation_residual, second.relation_residual)


def test_same_turn_lexicographic_past_is_allowed_but_query_and_future_are_removed() -> None:
    model = module()
    value = inputs()
    output = model(**value)
    assert output.effective_history_mask[1].tolist() == [True, True, True, False, False]
    assert output.effective_history_mask[0].tolist() == [True, True, True, False, False]


def test_primary_uses_history_but_capacity_control_contains_no_history_content() -> None:
    primary = module("primary_history_relation")
    control = module("history_presence_capacity_control")
    # Give both modes the same learned parameters.
    control.load_state_dict(primary.state_dict())
    base = inputs()
    changed = {name: value.clone() for name, value in base.items()}
    changed["projected_modalities"][:, 1, :, 0] += 7.0
    primary_before = primary(**base).relation_features
    primary_after = primary(**changed).relation_features
    control_before = control(**base).relation_features
    control_after = control(**changed).relation_features
    assert not torch.allclose(primary_before, primary_after)
    assert torch.allclose(control_before, control_after)
    assert primary.parameter_count() == control.parameter_count()


def test_empty_history_has_exact_zero_residual_in_both_capacity_matched_modes() -> None:
    value = inputs()
    value["history_mask"].zero_()
    for mode in (
        "primary_history_relation",
        "vad_history_only_no_history_3x3",
        "history_presence_capacity_control",
    ):
        output = module(mode)(**value)
        assert torch.count_nonzero(output.effective_history_mask) == 0
        assert torch.count_nonzero(output.relation_residual) == 0


def test_history_presence_control_uses_one_activation_bit_but_no_history_content() -> None:
    model = module("history_presence_capacity_control")
    empty = inputs()
    empty["history_mask"].zero_()
    present = {name: value.clone() for name, value in empty.items()}
    present["history_mask"][:, 0] = True
    changed_content = {name: value.clone() for name, value in present.items()}
    changed_content["projected_modalities"][:, 0] += 1000.0

    empty_output = model(**empty)
    present_output = model(**present)
    changed_output = model(**changed_content)
    assert torch.count_nonzero(empty_output.relation_residual) == 0
    assert torch.count_nonzero(present_output.relation_residual) > 0
    assert torch.allclose(present_output.relation_features, changed_output.relation_features)
    assert torch.allclose(present_output.relation_residual, changed_output.relation_residual)


def test_no_history_3x3_retains_only_capacity_coordinates_and_history_vad() -> None:
    model = module("vad_history_only_no_history_3x3")
    base = inputs()
    changed = {name: value.clone() for name, value in base.items()}
    changed["projected_modalities"][:, 1, :, 0] += 7.0
    before = model(**base).relation_features
    after = model(**changed).relation_features
    assert torch.allclose(before[:, :27], after[:, :27])
    assert not torch.allclose(before[:, 27:], after[:, 27:])


def test_missing_modality_values_are_not_consumed() -> None:
    model = module()
    base = inputs()
    base["modality_mask"][:, :, 1] = False
    changed = {name: value.clone() for name, value in base.items()}
    changed["projected_modalities"][:, :, 1] += 100_000.0
    before = model(**base)
    after = model(**changed)
    assert torch.allclose(before.relation_features, after.relation_features)
    assert torch.allclose(before.relation_residual, after.relation_residual)


def test_vad_aliases_cover_both_dataset_orders_and_auxiliary_loss_is_exact() -> None:
    emotiontalk = label_vad_table(EMOTIONTALK_LABELS)
    meld = label_vad_table(MELD_LABELS)
    by_name = {
        "neutral": emotiontalk[0],
        "happy": emotiontalk[1],
        "sad": emotiontalk[2],
        "angry": emotiontalk[3],
        "surprised": emotiontalk[4],
        "disgusted": emotiontalk[5],
        "fearful": emotiontalk[6],
    }
    expected_meld = torch.stack(
        [
            by_name["neutral"],
            by_name["surprised"],
            by_name["fearful"],
            by_name["sad"],
            by_name["happy"],
            by_name["disgusted"],
            by_name["angry"],
        ]
    )
    assert torch.equal(meld, expected_meld)
    labels = torch.arange(7, dtype=torch.long)
    assert fit_train_vad_auxiliary_loss(
        emotiontalk.clone(), labels, label_order=EMOTIONTALK_LABELS
    ).item() == pytest.approx(0.0)


def test_vad_supervision_and_configuration_fail_closed() -> None:
    with pytest.raises(ValueError, match="no frozen VAD alias"):
        label_vad_table(("neutral", "a", "b", "c", "d", "e", "f"))
    with pytest.raises(TypeError, match="torch.long"):
        fit_train_vad_auxiliary_loss(
            torch.zeros(2, 3),
            torch.zeros(2),
            label_order=EMOTIONTALK_LABELS,
        )
    with pytest.raises(ValueError, match="unknown affect-relation mode"):
        AffectRelationConfig(mode="unknown").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive fit-train auxiliary"):
        AffectRelationConfig(auxiliary_vad_weight=0.0).validate()
    AffectRelationConfig(
        auxiliary_vad_weight=0.0,
        use_vad_features=False,
    ).validate()
    with pytest.raises(ValueError, match="only the primary history branch"):
        AffectRelationConfig(
            auxiliary_vad_weight=0.0,
            use_vad_features=False,
            mode="vad_history_only_no_history_3x3",
        ).validate()


def test_relation_branch_is_small_enough_for_the_sub_two_million_backbone_budget() -> None:
    value = CausalAffectRelation(AffectRelationConfig(d_model=128, hidden_dim=128))
    assert value.parameter_count() < 25_000
