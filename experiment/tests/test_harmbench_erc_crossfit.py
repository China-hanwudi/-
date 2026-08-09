from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import inspect
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from hva_affect.emotiontalk_role_sidecar import FIT_ROLE  # noqa: E402
from hva_affect.harmbench_erc_contexts import (  # noqa: E402
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    FIT_TRAIN_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    build_strict_past_context_roster,
)
from hva_affect.harmbench_erc_crossfit import (  # noqa: E402
    HarmBenchCrossfitError,
    make_context_training_examples,
    make_shared_group_crossfit_plan,
    validate_context_training_examples,
    validate_shared_group_crossfit_plan,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    make_outcome_free_role_features,
    make_synthetic_fit_feature_capability,
)
from hva_affect.harmbench_erc_processors import (  # noqa: E402
    fit_shared_processor,
    transform_role_features,
)


def _features(groups: list[str] | None = None):
    groups = groups or [f"g{index // 2}" for index in range(20)]
    rows = len(groups)
    speakers = [f"s{index % 2}" for index in range(rows)]
    turns = [index % 2 for index in range(rows)]
    return make_outcome_free_role_features(
        dataset_id="synthetic",
        role=FIT_ROLE,
        keys=[f"r{index}" for index in range(rows)],
        texts=[f"text {index}" for index in range(rows)],
        audio=np.zeros((rows, 2), dtype=np.float32),
        video=np.zeros((rows, 3), dtype=np.float32),
        groups=groups,
        speaker_identity=speakers,
        turn_ids=turns,
        protocol_row_ids=np.arange(100, 100 + rows),
        row_alignment_sha256="a" * 64,
        feature_sha256="b" * 64,
    )


def _fit_capability(groups: list[str] | None = None):
    return make_synthetic_fit_feature_capability(
        fit_features=_features(groups),
        feature_manifest_sha256="c" * 64,
        synthetic_feature_projection_sha256="d" * 64,
    )


@pytest.fixture()
def context_training_contract():
    fit = _fit_capability()
    plan = make_shared_group_crossfit_plan(fit)
    seed = plan.training_seed_ids[0]
    fold = 0
    processor = fit_shared_processor(fit, plan, seed=seed, fold=fold)
    receipt = processor.receipt
    processed = transform_role_features(
        processor,
        fit,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_fit_feature_capability_sha256=fit.capability_sha256,
        expected_transform_source_capability_sha256=fit.capability_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
        expected_seed=seed,
        expected_fold=fold,
    )
    rosters = {}
    for strategy in STRICT_PAST_STRATEGY_IDS:
        rosters[strategy] = build_strict_past_context_roster(
            fit,
            fit,
            processed,
            receipt,
            plan,
            training_seed=seed,
            fold=fold,
            context_role=FIT_TRAIN_CONTEXT_ROLE,
            strategy_id=strategy,
            expected_fit_plan_capability_sha256=fit.capability_sha256,
            expected_source_capability_sha256=fit.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
        )
    expected_roster_shas = {
        strategy: roster.roster_sha256 for strategy, roster in rosters.items()
    }
    return fit, plan, seed, fold, receipt, processed, rosters, expected_roster_shas


def _make_examples(contract):
    fit, plan, seed, fold, receipt, processed, rosters, expected = contract
    return make_context_training_examples(
        rosters,
        fit,
        processed,
        receipt,
        plan,
        training_seed=seed,
        fold=fold,
        expected_fit_feature_capability_sha256=fit.capability_sha256,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_processed_output_receipt_sha256=processed.output_receipt_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
        expected_context_roster_sha256_by_strategy=expected,
    )


def test_plan_is_deterministic_group_whole_balanced_and_read_only() -> None:
    fit = _fit_capability()
    features = fit.fit
    first = make_shared_group_crossfit_plan(fit)
    second = make_shared_group_crossfit_plan(fit)
    assert first.plan_sha256 == second.plan_sha256
    np.testing.assert_array_equal(first.fold_assignment, second.fold_assignment)
    assert first.fold_assignment.shape == (5, features.rows)
    for seed_index in range(5):
        for group in set(features.groups.tolist()):
            assert len(set(first.fold_assignment[seed_index][features.groups == group])) == 1
        assert set(first.fold_assignment[seed_index].tolist()) == set(range(5))
    with pytest.raises(ValueError, match="read-only"):
        first.fold_assignment[0, 0] = 4


def test_train_heldout_are_disjoint_complete_and_group_isolated() -> None:
    fit = _fit_capability()
    features = fit.fit
    plan = make_shared_group_crossfit_plan(fit)
    for seed in plan.training_seed_ids:
        for fold in range(5):
            train = plan.train_indices(seed, fold, fit_capability=fit)
            heldout = plan.heldout_indices(seed, fold, fit_capability=fit)
            assert not set(train).intersection(heldout)
            assert set(train).union(heldout) == set(range(features.rows))
            assert not set(features.groups[train]).intersection(features.groups[heldout])


def test_fewer_than_five_groups_fails_closed() -> None:
    with pytest.raises(HarmBenchCrossfitError, match="fewer independent groups"):
        make_shared_group_crossfit_plan(
            _fit_capability(["g0", "g0", "g1", "g1", "g2", "g3"]),
        )


def test_plan_api_has_no_label_or_outcome_surface() -> None:
    names = set(inspect.signature(make_shared_group_crossfit_plan).parameters)
    names |= {field.name for field in fields(type(make_shared_group_crossfit_plan(
        _fit_capability()
    )))}
    assert not any("label" in name or "outcome" in name for name in names)


def test_context_augmentation_is_plan_derived_deduplicated_and_receipt_bound(
    context_training_contract,
) -> None:
    fit, plan, seed, fold, receipt, processed, rosters, expected = (
        context_training_contract
    )
    examples = _make_examples(context_training_contract)
    train = plan.train_indices(seed, fold, fit_capability=fit)
    train_protocol_ids = set(int(value) for value in fit.fit.protocol_row_ids[train])
    assert examples.example_count == len(examples.query_protocol_row_ids)
    assert set(examples.query_protocol_row_ids) == train_protocol_ids
    assert all(
        set(context).issubset(train_protocol_ids)
        for context in examples.context_protocol_row_ids
    )
    assert any(len(sources) > 1 for sources in examples.source_strategies)
    assert examples.context_roster_sha256_by_strategy == tuple(expected.items())
    assert examples.crossfit_plan_sha256 == plan.plan_sha256
    assert len(examples.example_sha256) == 64
    assert (
        validate_context_training_examples(
            examples,
            rosters,
            fit,
            processed,
            receipt,
            plan,
            training_seed=seed,
            fold=fold,
            expected_fit_feature_capability_sha256=fit.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
            expected_context_roster_sha256_by_strategy=expected,
            expected_context_training_examples_sha256=examples.example_sha256,
        )
        is examples
    )


def test_context_example_api_has_no_caller_selected_row_surface() -> None:
    parameters = set(inspect.signature(make_context_training_examples).parameters)
    assert parameters.isdisjoint(
        {"contexts", "query_indices", "allowed_indices", "groups", "labels", "outcomes"}
    )


def test_context_augmentation_rejects_wrong_role_subset_receipt_and_mutation(
    context_training_contract,
) -> None:
    fit, plan, seed, fold, receipt, processed, rosters, expected = (
        context_training_contract
    )
    changed = dict(rosters)
    changed["extra"] = rosters[STRICT_PAST_STRATEGY_IDS[0]]
    with pytest.raises(HarmBenchCrossfitError, match="roster/order"):
        make_context_training_examples(
            changed,
            fit,
            processed,
            receipt,
            plan,
            training_seed=seed,
            fold=fold,
            expected_fit_feature_capability_sha256=fit.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
            expected_context_roster_sha256_by_strategy=expected,
        )

    wrong_receipts = dict(expected)
    wrong_receipts[STRICT_PAST_STRATEGY_IDS[0]] = "f" * 64
    with pytest.raises(HarmBenchCrossfitError, match="external binding"):
        make_context_training_examples(
            rosters,
            fit,
            processed,
            receipt,
            plan,
            training_seed=seed,
            fold=fold,
            expected_fit_feature_capability_sha256=fit.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
            expected_context_roster_sha256_by_strategy=wrong_receipts,
        )

    first = STRICT_PAST_STRATEGY_IDS[0]
    mutated_roster = replace(rosters[first])
    contexts = list(mutated_roster.context_protocol_row_ids)
    contexts[0] = (int(fit.fit.protocol_row_ids[-1]),)
    object.__setattr__(mutated_roster, "context_protocol_row_ids", tuple(contexts))
    mutated_rosters = dict(rosters)
    mutated_rosters[first] = mutated_roster
    with pytest.raises(HarmBenchCrossfitError, match="invalid fit-train context roster"):
        _make_examples(
            (fit, plan, seed, fold, receipt, processed, mutated_rosters, expected)
        )

    heldout_roster = build_strict_past_context_roster(
        fit,
        fit,
        processed,
        receipt,
        plan,
        training_seed=seed,
        fold=fold,
        context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
        strategy_id=first,
        expected_fit_plan_capability_sha256=fit.capability_sha256,
        expected_source_capability_sha256=fit.capability_sha256,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_processed_output_receipt_sha256=processed.output_receipt_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
    )
    wrong_role = dict(rosters)
    wrong_role[first] = heldout_roster
    wrong_role_expected = dict(expected)
    wrong_role_expected[first] = heldout_roster.roster_sha256
    with pytest.raises(HarmBenchCrossfitError, match="invalid fit-train context roster"):
        _make_examples(
            (fit, plan, seed, fold, receipt, processed, wrong_role, wrong_role_expected)
        )


def test_context_example_live_validation_rejects_coherently_replaced_payload(
    context_training_contract,
) -> None:
    fit, plan, seed, fold, receipt, processed, rosters, expected = (
        context_training_contract
    )
    examples = _make_examples(context_training_contract)
    mutated = replace(examples)
    object.__setattr__(mutated, "query_protocol_row_ids", (999_999,))
    object.__setattr__(mutated, "example_count", 1)
    with pytest.raises(HarmBenchCrossfitError, match="live derivation"):
        validate_context_training_examples(
            mutated,
            rosters,
            fit,
            processed,
            receipt,
            plan,
            training_seed=seed,
            fold=fold,
            expected_fit_feature_capability_sha256=fit.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
            expected_context_roster_sha256_by_strategy=expected,
            expected_context_training_examples_sha256=examples.example_sha256,
        )


def test_validate_rejects_mutated_group_split() -> None:
    fit = _fit_capability()
    plan = make_shared_group_crossfit_plan(fit)
    mutated = np.array(plan.fold_assignment, copy=True)
    mutated[0, 1] = (mutated[0, 1] + 1) % 5
    mutated.setflags(write=False)
    object.__setattr__(plan, "fold_assignment", mutated)
    with pytest.raises(HarmBenchCrossfitError, match="split an independent group"):
        validate_shared_group_crossfit_plan(plan, fit)


def test_validate_rejects_plan_sha_mutation() -> None:
    fit = _fit_capability()
    plan = make_shared_group_crossfit_plan(fit)
    object.__setattr__(plan, "plan_sha256", "d" * 64)
    with pytest.raises(HarmBenchCrossfitError, match="plan SHA"):
        validate_shared_group_crossfit_plan(plan, fit)


def test_index_resolution_live_validates_plan_and_exact_seed_fold() -> None:
    fit = _fit_capability()
    plan = make_shared_group_crossfit_plan(fit)
    for seed, fold in ((999, 0), (17, 5), (17.0, 0), (True, 0)):
        with pytest.raises(HarmBenchCrossfitError):
            plan.train_indices(seed, fold, fit_capability=fit)
    plan.fold_assignment.setflags(write=True)
    plan.fold_assignment[0, 0] = (int(plan.fold_assignment[0, 0]) + 1) % 5
    plan.fold_assignment.setflags(write=False)
    with pytest.raises(HarmBenchCrossfitError):
        plan.train_indices(17, 0, fit_capability=fit)


def test_creation_rejects_seed_and_fold_coercion() -> None:
    fit = _fit_capability()
    with pytest.raises(HarmBenchCrossfitError, match="exact integers"):
        make_shared_group_crossfit_plan(fit, training_seed_ids=(17.9, 29, 43, 71, 101))
    with pytest.raises(HarmBenchCrossfitError, match="outer fold"):
        make_shared_group_crossfit_plan(fit, outer_folds="5")  # type: ignore[arg-type]
