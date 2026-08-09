from __future__ import annotations

from dataclasses import fields, replace
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hva_affect.harmbench_erc_contexts as context_module  # noqa: E402
from hva_affect.harmbench_erc_contexts import (  # noqa: E402
    CONTEXT_ROSTER_SCHEMA,
    CURRENT_ONLY_STRATEGY_ID,
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    FIT_TRAIN_CONTEXT_ROLE,
    HarmBenchContextError,
    SELECTION_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    StrictPastContextRoster,
    build_strict_past_context_roster,
    validate_strict_past_context_roster,
)
from hva_affect.harmbench_erc_contract import EXPECTED_TRAINING_SEEDS  # noqa: E402
from hva_affect.harmbench_erc_crossfit import (  # noqa: E402
    make_shared_group_crossfit_plan,
    resolve_shared_group_crossfit_indices,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    make_outcome_free_role_features,
    make_synthetic_fit_feature_capability,
    make_synthetic_selection_feature_capability,
)
from hva_affect.harmbench_erc_processors import (  # noqa: E402
    fit_shared_processor,
    transform_role_features,
)
from hva_affect.harmbench_erc_protocol_v2 import (  # noqa: E402
    STRATEGY_RULE_VERSION,
    get_context_strategy_contract,
    strategy_rule_sha256,
)


FIT_ROLE = "base_and_utility_fit"
SELECTION_ROLE = "model_selection"
SEED = int(EXPECTED_TRAINING_SEEDS[0])
FOLD = 0
ROSTER_SHA = "c" * 64


def _role_features(*, role: str, group_names: list[str], protocol_start: int):
    groups: list[str] = []
    speakers: list[str] = []
    turns: list[int] = []
    keys: list[str] = []
    texts: list[str] = []
    audio: list[list[float]] = []
    video: list[list[float]] = []
    # Duplicate turn 1 deliberately exercises the same-turn exclusion rule.
    turn_pattern = (0, 1, 1, 2, 3)
    speaker_pattern = ("alice", "bob", "alice", "bob", "alice")
    for group_index, group in enumerate(group_names):
        for local_index, (turn, speaker) in enumerate(
            zip(turn_pattern, speaker_pattern, strict=True)
        ):
            row = len(groups)
            groups.append(group)
            speakers.append(f"{group}:{speaker}")
            turns.append(turn)
            keys.append(f"{role}:{group}:{local_index}")
            texts.append(
                f"synthetic {role} dialogue {group_index} utterance {local_index} "
                f"emotion token {group_index}_{local_index}"
            )
            audio.append(
                [
                    float(group_index + 1),
                    float(local_index + 1),
                    float((group_index + 1) * (local_index + 1)),
                    float((-1) ** local_index),
                ]
            )
            video.append(
                [
                    float(local_index + 2),
                    float(group_index + 2),
                    float((local_index + 1) ** 2),
                    float(row + 1),
                ]
            )
    rows = len(groups)
    return make_outcome_free_role_features(
        dataset_id="synthetic",
        role=role,
        keys=np.asarray(keys),
        texts=texts,
        audio=np.asarray(audio, dtype=np.float32),
        video=np.asarray(video, dtype=np.float32),
        groups=np.asarray(groups),
        speaker_identity=np.asarray(speakers),
        turn_ids=np.asarray(turns, dtype=np.int64),
        protocol_row_ids=np.arange(
            protocol_start, protocol_start + rows, dtype=np.int64
        ),
        row_alignment_sha256=("a" if role == FIT_ROLE else "b") * 64,
        feature_sha256=("d" if role == FIT_ROLE else "e") * 64,
    )


@pytest.fixture(scope="module")
def synthetic_contract() -> SimpleNamespace:
    fit_features = _role_features(
        role=FIT_ROLE,
        group_names=[f"fit_group_{index}" for index in range(5)],
        protocol_start=100,
    )
    selection_features = _role_features(
        role=SELECTION_ROLE,
        group_names=["selection_group_0", "selection_group_1"],
        protocol_start=1_000,
    )
    fit_capability = make_synthetic_fit_feature_capability(
        fit_features=fit_features,
        feature_manifest_sha256="1" * 64,
        synthetic_feature_projection_sha256=ROSTER_SHA,
    )
    selection_capability = make_synthetic_selection_feature_capability(
        selection_features=selection_features,
        manifest_sha256="2" * 64,
        synthetic_feature_projection_sha256=ROSTER_SHA,
    )
    plan = make_shared_group_crossfit_plan(fit_capability)
    processor = fit_shared_processor(
        fit_capability,
        plan,
        seed=SEED,
        fold=FOLD,
    )
    processor_sha = processor.receipt.processor_receipt_sha256
    fit_processed = transform_role_features(
        processor,
        fit_capability,
        expected_processor_receipt_sha256=processor_sha,
        expected_fit_feature_capability_sha256=fit_capability.capability_sha256,
        expected_transform_source_capability_sha256=(
            fit_capability.capability_sha256
        ),
        expected_crossfit_plan_sha256=plan.plan_sha256,
        expected_seed=SEED,
        expected_fold=FOLD,
    )
    selection_processed = transform_role_features(
        processor,
        selection_capability,
        expected_processor_receipt_sha256=processor_sha,
        expected_fit_feature_capability_sha256=fit_capability.capability_sha256,
        expected_transform_source_capability_sha256=(
            selection_capability.capability_sha256
        ),
        expected_crossfit_plan_sha256=plan.plan_sha256,
        expected_seed=SEED,
        expected_fold=FOLD,
    )
    return SimpleNamespace(
        fit=fit_capability,
        selection=selection_capability,
        plan=plan,
        processor=processor,
        receipt=processor.receipt,
        fit_processed=fit_processed,
        selection_processed=selection_processed,
    )


def _build(
    contract: SimpleNamespace,
    *,
    context_role: str = FIT_HELDOUT_OOF_CONTEXT_ROLE,
    strategy_id: str = "dialogue_all_past",
    source: object | None = None,
    processed: object | None = None,
    receipt: object | None = None,
    plan: object | None = None,
    **expected_overrides: object,
) -> StrictPastContextRoster:
    if source is None:
        source = (
            contract.selection
            if context_role == SELECTION_CONTEXT_ROLE
            else contract.fit
        )
    if processed is None:
        processed = (
            contract.selection_processed
            if context_role == SELECTION_CONTEXT_ROLE
            else contract.fit_processed
        )
    receipt = contract.receipt if receipt is None else receipt
    plan = contract.plan if plan is None else plan
    expected = {
        "expected_fit_plan_capability_sha256": contract.fit.capability_sha256,
        "expected_source_capability_sha256": source.capability_sha256,
        "expected_processor_receipt_sha256": contract.receipt.processor_receipt_sha256,
        "expected_processed_output_receipt_sha256": processed.output_receipt_sha256,
        "expected_crossfit_plan_sha256": contract.plan.plan_sha256,
    }
    expected.update(expected_overrides)
    return build_strict_past_context_roster(
        contract.fit,
        source,
        processed,
        receipt,
        plan,
        training_seed=SEED,
        fold=FOLD,
        context_role=context_role,
        strategy_id=strategy_id,
        **expected,
    )


def _validate(
    roster: StrictPastContextRoster,
    contract: SimpleNamespace,
    *,
    context_role: str | None = None,
    strategy_id: str | None = None,
    source: object | None = None,
    processed: object | None = None,
    receipt: object | None = None,
    plan: object | None = None,
    expected_roster_sha: str | None = None,
) -> StrictPastContextRoster:
    context_role = context_role or roster.context_role
    strategy_id = strategy_id or roster.strategy_id
    if source is None:
        source = (
            contract.selection
            if context_role == SELECTION_CONTEXT_ROLE
            else contract.fit
        )
    if processed is None:
        processed = (
            contract.selection_processed
            if context_role == SELECTION_CONTEXT_ROLE
            else contract.fit_processed
        )
    return validate_strict_past_context_roster(
        roster,
        contract.fit,
        source,
        processed,
        contract.receipt if receipt is None else receipt,
        contract.plan if plan is None else plan,
        training_seed=SEED,
        fold=FOLD,
        context_role=context_role,
        strategy_id=strategy_id,
        expected_fit_plan_capability_sha256=contract.fit.capability_sha256,
        expected_source_capability_sha256=source.capability_sha256,
        expected_processor_receipt_sha256=contract.receipt.processor_receipt_sha256,
        expected_processed_output_receipt_sha256=processed.output_receipt_sha256,
        expected_crossfit_plan_sha256=contract.plan.plan_sha256,
        expected_context_roster_sha256=(
            expected_roster_sha or roster.roster_sha256
        ),
    )


def _mutated_roster(
    roster: StrictPastContextRoster,
    *,
    query_position: int,
    context_ids: tuple[int, ...],
) -> StrictPastContextRoster:
    mutated = replace(roster)
    contexts = list(mutated.context_protocol_row_ids)
    contexts[query_position] = context_ids
    object.__setattr__(mutated, "context_protocol_row_ids", tuple(contexts))
    return mutated


def test_public_builder_has_no_raw_rows_histories_embeddings_or_outcomes() -> None:
    parameters = set(inspect.signature(build_strict_past_context_roster).parameters)
    forbidden = {
        "groups",
        "roles",
        "histories",
        "row_embedding",
        "modality_embeddings",
        "query_indices",
        "allowed_indices",
        "labels",
        "outcomes",
        "candidate_scope",
        "top_k",
        "ranking",
        "ranking_tie",
        "modality_order",
        "empty_fallback",
    }
    assert parameters.isdisjoint(forbidden)
    assert not hasattr(context_module, "StrictPastContextInputs")
    roster_fields = {item.name for item in fields(StrictPastContextRoster)}
    assert not any("label" in name or "outcome" in name for name in roster_fields)


def test_fit_train_and_heldout_queries_are_derived_from_the_live_plan(
    synthetic_contract: SimpleNamespace,
) -> None:
    train, heldout = resolve_shared_group_crossfit_indices(
        synthetic_contract.plan,
        synthetic_contract.fit,
        training_seed=SEED,
        fold=FOLD,
    )
    train_roster = _build(
        synthetic_contract,
        context_role=FIT_TRAIN_CONTEXT_ROLE,
    )
    heldout_roster = _build(
        synthetic_contract,
        context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
    )
    fit_ids = synthetic_contract.fit.fit.protocol_row_ids
    assert train_roster.query_protocol_row_ids == tuple(
        sorted(int(fit_ids[index]) for index in train)
    )
    assert heldout_roster.query_protocol_row_ids == tuple(
        sorted(int(fit_ids[index]) for index in heldout)
    )
    train_ids = set(train_roster.query_protocol_row_ids)
    heldout_ids = set(heldout_roster.query_protocol_row_ids)
    assert train_ids.isdisjoint(heldout_ids)
    assert all(
        set(context).issubset(train_ids)
        for context in train_roster.context_protocol_row_ids
    )
    assert all(
        set(context).issubset(heldout_ids)
        for context in heldout_roster.context_protocol_row_ids
    )
    assert _validate(train_roster, synthetic_contract) is train_roster
    assert _validate(heldout_roster, synthetic_contract) is heldout_roster


def test_selection_uses_feature_only_capability_and_all_selection_rows(
    synthetic_contract: SimpleNamespace,
) -> None:
    roster = _build(
        synthetic_contract,
        context_role=SELECTION_CONTEXT_ROLE,
        strategy_id="same_speaker_all_past",
    )
    assert roster.query_protocol_row_ids == tuple(
        sorted(int(value) for value in synthetic_contract.selection.selection.protocol_row_ids)
    )
    assert roster.source_capability_sha256 == synthetic_contract.selection.capability_sha256
    assert roster.fit_plan_capability_sha256 == synthetic_contract.fit.capability_sha256
    assert roster.processor_receipt_sha256 == synthetic_contract.receipt.processor_receipt_sha256
    assert roster.processed_output_receipt_sha256 == (
        synthetic_contract.selection_processed.output_receipt_sha256
    )
    assert _validate(roster, synthetic_contract) is roster


@pytest.mark.parametrize("strategy_id", STRICT_PAST_STRATEGY_IDS)
def test_every_frozen_history_strategy_is_strict_past_and_partition_local(
    synthetic_contract: SimpleNamespace,
    strategy_id: str,
) -> None:
    roster = _build(
        synthetic_contract,
        context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
        strategy_id=strategy_id,
    )
    features = synthetic_contract.fit.fit
    by_id = {
        int(protocol_id): index
        for index, protocol_id in enumerate(features.protocol_row_ids)
    }
    for query_id, context in zip(
        roster.query_protocol_row_ids,
        roster.context_protocol_row_ids,
        strict=True,
    ):
        query = by_id[query_id]
        indices = [by_id[value] for value in context]
        assert len(indices) == len(set(indices))
        assert all(features.groups[index] == features.groups[query] for index in indices)
        assert all(
            int(features.turn_ids[index]) < int(features.turn_ids[query])
            for index in indices
        )
        assert indices == sorted(
            indices,
            key=lambda index: (
                int(features.turn_ids[index]),
                int(features.protocol_row_ids[index]),
            ),
        )
        if strategy_id == "same_speaker_all_past":
            assert all(
                features.speaker_identity[index]
                == features.speaker_identity[query]
                for index in indices
            )
        if strategy_id in {
            "recent_k3",
            "similarity_top3",
            "modality_balanced_top3",
        }:
            assert len(indices) <= 3
    assert _validate(roster, synthetic_contract) is roster


def test_current_only_mechanically_forces_zero_context_and_history_consumption(
    synthetic_contract: SimpleNamespace,
) -> None:
    roster = _build(
        synthetic_contract,
        context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
        strategy_id=CURRENT_ONLY_STRATEGY_ID,
    )
    assert all(context == () for context in roster.context_protocol_row_ids)
    assert all(count == 0 for count in roster.context_counts)
    assert roster.total_context_count == 0
    assert roster.history_consumption_count == 0
    assert _validate(roster, synthetic_contract) is roster

    # Post-construction mutation simulates a stale/malicious in-memory artifact.
    nonfirst = next(
        index
        for index, query_id in enumerate(roster.query_protocol_row_ids)
        if int(synthetic_contract.fit.fit.turn_ids[
            np.flatnonzero(synthetic_contract.fit.fit.protocol_row_ids == query_id)[0]
        ])
        > 0
    )
    earlier = roster.query_protocol_row_ids[nonfirst - 1]
    forged = _mutated_roster(
        roster,
        query_position=nonfirst,
        context_ids=(earlier,),
    )
    with pytest.raises(HarmBenchContextError, match="current-only must have zero"):
        _validate(forged, synthetic_contract)


@pytest.mark.parametrize(
    "expected_name",
    [
        "expected_fit_plan_capability_sha256",
        "expected_source_capability_sha256",
        "expected_processor_receipt_sha256",
        "expected_processed_output_receipt_sha256",
        "expected_crossfit_plan_sha256",
    ],
)
def test_every_external_upstream_binding_is_required(
    synthetic_contract: SimpleNamespace,
    expected_name: str,
) -> None:
    with pytest.raises(
        HarmBenchContextError,
        match="external binding|live fold binding|differs from expected",
    ):
        _build(synthetic_contract, **{expected_name: "f" * 64})


def test_mutated_source_capability_fails_live_revalidation(
    synthetic_contract: SimpleNamespace,
) -> None:
    bad_features = replace(synthetic_contract.fit.fit)
    changed_groups = np.asarray(bad_features.groups).copy()
    changed_groups[0] = "forged_group"
    changed_groups.setflags(write=False)
    object.__setattr__(bad_features, "groups", changed_groups)
    bad_capability = replace(synthetic_contract.fit, fit=bad_features)
    with pytest.raises(HarmBenchContextError, match="capability changed"):
        _build(
            synthetic_contract,
            source=bad_capability,
            expected_source_capability_sha256=bad_capability.capability_sha256,
        )


def test_mutated_processed_output_fails_live_revalidation(
    synthetic_contract: SimpleNamespace,
) -> None:
    bad_processed = replace(synthetic_contract.fit_processed)
    changed_text = np.asarray(bad_processed.text).copy()
    changed_text[0, 0] += np.float32(0.25)
    changed_text.setflags(write=False)
    object.__setattr__(bad_processed, "text", changed_text)
    with pytest.raises(HarmBenchContextError, match="processed feature output changed"):
        _build(
            synthetic_contract,
            processed=bad_processed,
            expected_processed_output_receipt_sha256=(
                bad_processed.output_receipt_sha256
            ),
        )


def test_stale_processor_receipt_and_wrong_plan_fail_closed(
    synthetic_contract: SimpleNamespace,
) -> None:
    bad_receipt = replace(synthetic_contract.receipt)
    object.__setattr__(
        bad_receipt,
        "train_protocol_row_ids",
        tuple(reversed(bad_receipt.train_protocol_row_ids)),
    )
    with pytest.raises(HarmBenchContextError, match="processor receipt changed"):
        _build(synthetic_contract, receipt=bad_receipt)

    bad_plan = replace(synthetic_contract.plan)
    object.__setattr__(bad_plan, "plan_sha256", "f" * 64)
    with pytest.raises(HarmBenchContextError, match="crossfit plan changed"):
        _build(synthetic_contract, plan=bad_plan)


def test_role_capability_mismatch_is_rejected_without_label_surface(
    synthetic_contract: SimpleNamespace,
) -> None:
    with pytest.raises(HarmBenchContextError, match="selection context requires"):
        _build(
            synthetic_contract,
            context_role=SELECTION_CONTEXT_ROLE,
            source=synthetic_contract.fit,
            processed=synthetic_contract.fit_processed,
        )
    with pytest.raises(HarmBenchContextError, match="fit context requires"):
        _build(
            synthetic_contract,
            context_role=FIT_TRAIN_CONTEXT_ROLE,
            source=synthetic_contract.selection,
            processed=synthetic_contract.selection_processed,
        )


def test_forged_future_same_turn_and_cross_group_contexts_fail_closed(
    synthetic_contract: SimpleNamespace,
) -> None:
    heldout = _build(
        synthetic_contract,
        context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
        strategy_id="dialogue_all_past",
    )
    features = synthetic_contract.fit.fit
    by_id = {
        int(protocol_id): index
        for index, protocol_id in enumerate(features.protocol_row_ids)
    }

    future = _mutated_roster(
        heldout,
        query_position=0,
        context_ids=(heldout.query_protocol_row_ids[-1],),
    )
    with pytest.raises(HarmBenchContextError, match="current or future turn"):
        _validate(future, synthetic_contract)

    same_turn_pair: tuple[int, int] | None = None
    for left_position, left_id in enumerate(heldout.query_protocol_row_ids):
        left = by_id[left_id]
        for right_id in heldout.query_protocol_row_ids[left_position + 1 :]:
            right = by_id[right_id]
            if (
                features.groups[left] == features.groups[right]
                and int(features.turn_ids[left]) == int(features.turn_ids[right])
            ):
                same_turn_pair = (left_position, right_id)
                break
        if same_turn_pair is not None:
            break
    assert same_turn_pair is not None
    same_turn = _mutated_roster(
        heldout,
        query_position=same_turn_pair[0],
        context_ids=(same_turn_pair[1],),
    )
    with pytest.raises(HarmBenchContextError, match="current or future turn"):
        _validate(same_turn, synthetic_contract)

    train = _build(
        synthetic_contract,
        context_role=FIT_TRAIN_CONTEXT_ROLE,
        strategy_id="dialogue_all_past",
    )
    query_position = next(
        position
        for position, query_id in enumerate(train.query_protocol_row_ids)
        if int(features.turn_ids[by_id[query_id]]) > 0
    )
    query = by_id[train.query_protocol_row_ids[query_position]]
    other_group_id = next(
        protocol_id
        for protocol_id in train.query_protocol_row_ids
        if features.groups[by_id[protocol_id]] != features.groups[query]
    )
    cross_group = _mutated_roster(
        train,
        query_position=query_position,
        context_ids=(other_group_id,),
    )
    with pytest.raises(HarmBenchContextError, match="independent group"):
        _validate(cross_group, synthetic_contract)


def test_stale_context_receipt_and_wrong_requested_strategy_fail_closed(
    synthetic_contract: SimpleNamespace,
) -> None:
    roster = _build(synthetic_contract, strategy_id="recent_k3")
    stale = replace(roster)
    object.__setattr__(stale, "roster_sha256", "f" * 64)
    with pytest.raises(HarmBenchContextError, match="external binding"):
        _validate(
            stale,
            synthetic_contract,
            expected_roster_sha=roster.roster_sha256,
        )
    with pytest.raises(HarmBenchContextError, match="live derivation: strategy_id"):
        _validate(
            roster,
            synthetic_contract,
            strategy_id="similarity_top3",
        )


def test_receipt_binds_schema_plan_processor_output_and_ordered_row_hashes(
    synthetic_contract: SimpleNamespace,
) -> None:
    roster = _build(
        synthetic_contract,
        context_role=SELECTION_CONTEXT_ROLE,
        strategy_id="modality_balanced_top3",
    )
    assert roster.schema_version == CONTEXT_ROSTER_SCHEMA
    rule = get_context_strategy_contract("modality_balanced_top3")
    assert roster.strategy_rule_version == STRATEGY_RULE_VERSION
    assert roster.strategy_rule_sha256 == strategy_rule_sha256(rule)
    assert roster.candidate_scope == rule["candidate_scope"]
    assert roster.strict_past_required is True
    assert roster.top_k == 3
    assert roster.ranking == rule["ranking"]
    assert roster.ranking_tie == "ascending_protocol_row_id"
    assert roster.zero_vector_policy == rule["zero_vector"]
    assert roster.modality_order == ("text", "audio", "video")
    assert roster.duplicate_skip_policy == rule["duplicate_skip"]
    assert roster.emission_order == "ascending_turn_id_then_protocol_row_id"
    assert roster.empty_fallback == "empty_tuple"
    assert roster.crossfit_plan_sha256 == synthetic_contract.plan.plan_sha256
    assert roster.processor_receipt_sha256 == synthetic_contract.receipt.processor_receipt_sha256
    assert roster.processed_output_receipt_sha256 == (
        synthetic_contract.selection_processed.output_receipt_sha256
    )
    assert roster.query_count == len(roster.query_protocol_row_ids)
    assert roster.context_counts == tuple(
        len(context) for context in roster.context_protocol_row_ids
    )
    assert roster.total_context_count == sum(roster.context_counts)
    assert len(roster.roster_sha256) == 64
    assert _validate(roster, synthetic_contract) is roster

    with pytest.raises(HarmBenchContextError, match="immutable tuples"):
        replace(
            roster,
            query_protocol_row_ids=list(roster.query_protocol_row_ids),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("strategy_rule_version", "harmbench_erc_context_strategy_rules_v3"),
        ("strategy_rule_sha256", "f" * 64),
        ("candidate_scope", "all_dialogues"),
        ("strict_past_required", False),
        ("top_k", 4),
        ("ranking", "physical_row_order"),
        ("ranking_tie", "physical_row_order"),
        ("zero_vector_policy", "drop_zero_vectors"),
        ("modality_order", ("video", "audio", "text")),
        ("duplicate_skip_policy", "allow_duplicates"),
        ("emission_order", "rank_order"),
        ("empty_fallback", "current_vector"),
    ],
)
def test_live_validator_rejects_rule_drift_even_when_context_rows_are_identical(
    synthetic_contract: SimpleNamespace,
    field_name: str,
    changed_value: object,
) -> None:
    roster = _build(synthetic_contract, strategy_id="modality_balanced_top3")
    object.__setattr__(roster, field_name, changed_value)
    with pytest.raises(HarmBenchContextError, match="context strategy rule changed"):
        _validate(roster, synthetic_contract)


def _exact_algorithm_contract() -> SimpleNamespace:
    # Physical order and protocol order intentionally disagree.  Rows 1 and 2
    # share a turn; row 4 shares the query turn; row 6 is cross-dialogue.
    features = SimpleNamespace(
        groups=np.asarray(["d", "d", "d", "d", "d", "d", "other"]),
        speaker_identity=np.asarray(["a", "b", "a", "b", "a", "a", "a"]),
        turn_ids=np.asarray([0, 1, 1, 2, 3, 3, 0], dtype=np.int64),
        protocol_row_ids=np.asarray([50, 40, 20, 10, 5, 60, 1], dtype=np.int64),
    )
    fusion = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    text = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    audio = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.6, 0.4],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    video = np.asarray(
        [
            [1.0, 0.0],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.6, 0.4],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        features=features,
        processed=SimpleNamespace(
            fusion=fusion,
            text=text,
            audio=audio,
            video=video,
        ),
    )


def test_duplicate_turns_same_turn_and_cross_dialogue_have_exact_semantics() -> None:
    resolved = _exact_algorithm_contract()
    allowed = set(range(7))
    selected = context_module._select_context_indices(
        resolved,
        query=5,
        strategy_id="dialogue_all_past",
        allowed_indices=allowed,
    )
    # Same-turn row 4 and cross-dialogue row 6 are excluded.  Duplicate turn 1
    # is emitted by ascending protocol row (20 before 40), not physical order.
    assert selected == (0, 2, 1, 3)


def test_similarity_tie_and_zero_vector_policy_are_exact() -> None:
    resolved = _exact_algorithm_contract()
    assert context_module._cosine_score(resolved.processed.fusion, 5, 0) == 0.0
    assert context_module._cosine_score(resolved.processed.fusion, 0, 5) == 0.0
    selected = context_module._select_context_indices(
        resolved,
        query=5,
        strategy_id="similarity_top3",
        allowed_indices=set(range(7)),
    )
    # Every score ties at zero, so protocol IDs 10, 20, 40 win ranking; output
    # is then canonically emitted by turn and protocol ID.
    assert selected == (2, 1, 3)


def test_modality_overlap_skips_duplicates_in_text_audio_video_order() -> None:
    resolved = _exact_algorithm_contract()
    selected = context_module._select_context_indices(
        resolved,
        query=5,
        strategy_id="modality_balanced_top3",
        allowed_indices=set(range(7)),
    )
    # Candidate 0 tops all modalities.  It is selected once; at depth 1 text
    # contributes row 1 and audio contributes row 2, then canonical emission.
    assert selected == (0, 2, 1)
    assert len(selected) == len(set(selected)) == 3


@pytest.mark.parametrize(
    "strategy_id",
    ("recent_k3", "similarity_top3", "modality_balanced_top3"),
)
def test_top3_strategies_use_exact_empty_fallback_and_n_less_than_three(
    strategy_id: str,
) -> None:
    resolved = _exact_algorithm_contract()
    allowed = set(range(7))
    assert context_module._select_context_indices(
        resolved,
        query=0,
        strategy_id=strategy_id,
        allowed_indices=allowed,
    ) == ()
    assert context_module._select_context_indices(
        resolved,
        query=1,
        strategy_id=strategy_id,
        allowed_indices=allowed,
    ) == (0,)
