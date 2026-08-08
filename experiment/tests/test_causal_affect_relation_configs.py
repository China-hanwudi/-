from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", reason="affect-relation config tests require PyTorch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_multimodal_backbone import (  # noqa: E402
    CausalBackboneConfig,
    CausalMultimodalBackbone,
)
from hva_affect.causal_backbone_strategy_staged_pipeline import (  # noqa: E402
    REGISTERED_VARIANTS,
    StrategyStagedPipelineError,
    complete_strategy_selection,
    derive_registered_variant,
)
from hva_affect.data_contract import ContractError  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    validate_open_role_backbone_payload,
)


DATASETS = {
    "emotiontalk": (
        "neutral",
        "happy",
        "sad",
        "angry",
        "surprised",
        "disgusted",
        "fearful",
    ),
    "meld": (
        "neutral",
        "surprise",
        "fear",
        "sadness",
        "joy",
        "disgust",
        "anger",
    ),
}
VARIANTS = {
    "full": ("primary_history_relation", True, 0.1),
    "capacity_control": ("history_presence_capacity_control", True, 0.1),
    "no_vad": ("primary_history_relation", False, 0.0),
    "no_history_3x3": ("vad_history_only_no_history_3x3", True, 0.1),
}


def config_path(dataset: str, variant: str) -> Path:
    return ROOT / "configs" / f"carma_affect_relation_{dataset}_{variant}_v1.json"


def test_eight_frozen_configs_are_executable_capacity_matched_and_label_safe() -> None:
    parameter_counts: dict[str, set[int]] = {dataset: set() for dataset in DATASETS}
    training_contracts: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    schema_versions: set[str] = set()

    for dataset, label_order in DATASETS.items():
        for variant, (mode, use_vad, weight) in VARIANTS.items():
            path = config_path(dataset, variant)
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = CausalBackboneConfig.from_mapping(payload)
            model = CausalMultimodalBackbone(config)

            assert payload["status"] == (
                "frozen_open_role_production_contract_not_performance_evidence"
            )
            assert payload["experimental_contract"]["variant"] == variant
            assert payload["runtime_contract"]["staged_execution_required"] is True
            assert payload["experimental_contract"]["model_selection_may_choose_variant"] is False
            assert payload["experimental_contract"][
                "selection_labels_available_only_in_independent_evaluate_stage"
            ] is True
            assert config.affect_relation_mode == mode
            assert config.affect_relation_use_vad_features is use_vad
            assert config.auxiliary_vad_weight == pytest.approx(weight)
            assert config.emotion_label_order == (label_order if use_vad else ())
            assert model.parameter_count() < config.parameter_limit
            with pytest.raises(ContractError, match="staged-only"):
                validate_open_role_backbone_payload(payload)
            parameter_counts[dataset].add(model.parameter_count())
            training_contracts[dataset].add(
                json.dumps(payload["training_runner"], sort_keys=True)
            )
            schema_versions.add(payload["schema_version"])

    assert all(len(values) == 1 for values in parameter_counts.values())
    assert all(len(values) == 1 for values in training_contracts.values())
    assert len(schema_versions) == len(DATASETS) * len(VARIANTS)


def test_eight_frozen_configs_define_the_only_strategy_variant_names() -> None:
    """Prevent a frozen-config name from drifting into a strategy alias."""

    assert REGISTERED_VARIANTS == (
        "full",
        "no_vad",
        "no_history_3x3",
        "capacity_control",
    )
    assert set(REGISTERED_VARIANTS) == set(VARIANTS)
    for dataset in DATASETS:
        for expected_variant in REGISTERED_VARIANTS:
            payload = json.loads(
                config_path(dataset, expected_variant).read_text(encoding="utf-8")
            )
            model_config = CausalBackboneConfig.from_mapping(payload)
            assert payload["experimental_contract"]["variant"] == expected_variant
            assert derive_registered_variant(model_config) == expected_variant

            wrong_variant = next(
                value for value in REGISTERED_VARIANTS if value != expected_variant
            )
            with pytest.raises(
                StrategyStagedPipelineError,
                match="variant label differs from the validated model contract",
            ):
                complete_strategy_selection(
                    upstream=object(),
                    registered_variant=wrong_variant,
                    model_config=model_config,
                    run_config=object(),
                    private_output_root=ROOT / "must-not-be-created",
                    config_paths={},
                    code_paths={},
                    environment={},
                    device=torch.device("cpu"),
                )
    assert not (ROOT / "must-not-be-created").exists()


def test_no_vad_ablation_emits_no_auxiliary_target_but_keeps_exact_capacity() -> None:
    full = CausalBackboneConfig(
        text_dim=8,
        audio_dim=10,
        video_dim=12,
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=24,
        num_speakers=4,
        max_turns=32,
        max_relative_turn=8,
        dropout=0.0,
        affect_relation_mode="primary_history_relation",
        affect_relation_hidden_dim=24,
        affect_relation_use_vad_features=True,
        auxiliary_vad_weight=0.1,
        emotion_label_order=DATASETS["emotiontalk"],
    )
    no_vad = replace(
        full,
        affect_relation_use_vad_features=False,
        auxiliary_vad_weight=0.0,
        emotion_label_order=(),
    )
    full_model = CausalMultimodalBackbone(full).eval()
    no_vad_model = CausalMultimodalBackbone(no_vad).eval()
    assert full_model.parameter_count() == no_vad_model.parameter_count()
    assert tuple(no_vad_model.vad_label_table.shape) == (0, 3)

    generator = torch.Generator().manual_seed(20260808)
    inputs = {
        "text_features": torch.randn(2, 4, 8, generator=generator),
        "audio_features": torch.randn(2, 4, 10, generator=generator),
        "video_features": torch.randn(2, 4, 12, generator=generator),
        "speaker_ids": torch.tensor([[0, 1, 0, 1]] * 2),
        "turn_ids": torch.tensor([[0, 1, 2, 3]] * 2),
        "valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "history_mask": torch.ones(2, 4, dtype=torch.bool),
        "query_indices": torch.tensor([2, 3]),
    }
    with torch.no_grad():
        assert full_model(**inputs).query_vad is not None
        assert no_vad_model(**inputs).query_vad is None


def test_backbone_relation_configuration_rejects_ambiguous_vad_contracts() -> None:
    base = CausalBackboneConfig(
        affect_relation_mode="primary_history_relation",
        affect_relation_use_vad_features=True,
        auxiliary_vad_weight=0.1,
        emotion_label_order=DATASETS["emotiontalk"],
    )
    base.validate()
    with pytest.raises(ValueError, match="no-VAD ablation"):
        replace(base, affect_relation_use_vad_features=False).validate()
    with pytest.raises(ValueError, match="positive fit-train auxiliary"):
        replace(base, auxiliary_vad_weight=0.0).validate()
    with pytest.raises(ValueError, match="must retain the frozen VAD branch"):
        replace(
            base,
            affect_relation_mode="history_presence_capacity_control",
            affect_relation_use_vad_features=False,
            auxiliary_vad_weight=0.0,
            emotion_label_order=(),
        ).validate()
    with pytest.raises(ValueError, match="verified dataset order"):
        base.validate_dataset_label_order(DATASETS["meld"])
    base.validate_dataset_label_order(DATASETS["emotiontalk"])
