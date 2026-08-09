from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_utility_model import (  # noqa: E402
    UtilitySplit,
    trainable_parameter_count,
)
from hva_affect.class_balanced_utility_repair import (  # noqa: E402
    MODEL_NAMES,
    MODEL_MODES,
    CapacityMatchedUtilitySpec,
    ClassBalanceSpec,
    ClassBalancedRepairContractError,
    _capacity_matched_targets,
    _nll_identity_audit,
    _per_utility_seed_aggregate_records,
    _true_reference_diagnostics,
    build_class_balance_profile,
    canonical_manifest_sha256,
    default_capacity_matched_specs,
    deterministic_weighted_resample_indices,
    fit_class_balanced_seed_scores,
    fit_class_balanced_utility_model,
    group_oof_class_balanced_predictions,
    load_class_balanced_repair_config,
    run_open_role_class_balanced_query_policy,
)


@dataclass(frozen=True)
class _Task:
    query_index: int
    candidate_index: int


def test_class_counts_use_unique_queries_and_draws_do_not_multiply_query_mass() -> None:
    tasks = (
        _Task(10, 0),
        _Task(10, 1),
        _Task(10, 2),
        _Task(11, 0),
        _Task(12, 0),
        _Task(13, 0),
        _Task(14, 0),
        _Task(15, 0),
    )
    # Unique-query counts are [3, 2, 1], although query 10 has three draws.
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
    profile = build_class_balance_profile(
        tasks,
        labels,
        n_classes=3,
        spec=ClassBalanceSpec(scheme="effective_number", beta=0.9),
    )
    np.testing.assert_array_equal(profile.class_counts, [3, 2, 1])
    assert profile.class_weights[2] > profile.class_weights[1] > profile.class_weights[0]
    # Each query contributes its class weight in total, independent of draws.
    for query, expected_class in ((10, 0), (11, 0), (12, 0), (13, 1), (14, 1), (15, 2)):
        mask = np.asarray([task.query_index == query for task in tasks])
        assert profile.task_weights[mask].sum() == pytest.approx(
            profile.class_weights[expected_class]
        )
    assert profile.task_weights.sum() == pytest.approx(6.0)

    inconsistent = labels.copy()
    inconsistent[1] = 1
    with pytest.raises(ClassBalancedRepairContractError, match="inconsistent labels"):
        build_class_balance_profile(
            tasks,
            inconsistent,
            n_classes=3,
            spec=ClassBalanceSpec(scheme="inverse_frequency"),
        )


def test_systematic_weighted_resampling_is_exact_deterministic_and_rare_focused() -> None:
    weights = np.asarray([1.0, 1.0, 8.0, 8.0], dtype=np.float64)
    first = deterministic_weighted_resample_indices(weights, size=180, seed=17)
    second = deterministic_weighted_resample_indices(weights, size=180, seed=17)
    other = deterministic_weighted_resample_indices(weights, size=180, seed=29)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 180
    assert np.sum(first >= 2) > np.sum(first < 2)
    assert not np.array_equal(first, other)


def _synthetic_split() -> tuple[UtilitySplit, tuple[_Task, ...], np.ndarray]:
    rng = np.random.default_rng(20260808)
    # Four whole groups; every group contains all seven classes, so every OOF
    # training fold can derive its own seven-class profile without held labels.
    groups = np.repeat(np.arange(4, dtype=np.int64), 7)
    labels = np.tile(np.arange(7, dtype=np.int64), 4)
    x = rng.normal(size=(28, 5))
    forward = 0.2 * x[:, 0] - 0.1 * x[:, 1] + labels / 20.0
    backward = -0.1 * x[:, 2] + 0.15 * x[:, 3] - labels / 30.0
    split = UtilitySplit.validated(
        x,
        forward,
        backward,
        groups,
        label="synthetic class-balanced split",
    )
    tasks = tuple(_Task(index, max(0, index - 1)) for index in range(len(x)))
    return split, tasks, labels


def _small_spec(mode: str) -> CapacityMatchedUtilitySpec:
    return CapacityMatchedUtilitySpec(
        name=f"small_{mode}",
        mode=mode,
        hidden_layer_sizes=(4,),
        max_iter=5,
        batch_size=8,
        early_stopping=False,
    )


def test_all_four_controls_have_identical_capacity_and_correct_target_semantics() -> None:
    split, tasks, labels = _synthetic_split()
    specs = tuple(_small_spec(mode) for mode in MODEL_MODES)
    parameters = {
        trainable_parameter_count(split.x.shape[1], spec.hidden_layer_sizes, 2)
        for spec in specs
    }
    assert len(parameters) == 1
    forward = _capacity_matched_targets(split, "forward_only_capacity_matched")
    backward = _capacity_matched_targets(split, "backward_only_capacity_matched")
    pseudo = _capacity_matched_targets(
        split, "pseudo_bidirectional_capacity_matched"
    )
    true = _capacity_matched_targets(split, "true_bidirectional_capacity_matched")
    np.testing.assert_allclose(forward[:, 0], split.forward)
    np.testing.assert_allclose(forward[:, 1], split.forward)
    np.testing.assert_allclose(backward[:, 0], split.backward)
    np.testing.assert_allclose(backward[:, 1], split.backward)
    np.testing.assert_allclose(pseudo, forward)
    np.testing.assert_allclose(true[:, 0], split.forward)
    np.testing.assert_allclose(true[:, 1], split.backward)

    fitted = fit_class_balanced_utility_model(
        split,
        tasks,
        labels,
        specs[-1],
        ClassBalanceSpec(beta=0.9),
        seed=17,
    )
    prediction = fitted.predict(split.x[:3])
    assert prediction.decision_score.shape == (3,)
    assert fitted.parameter_count == next(iter(parameters))
    # Gold labels and task objects cannot enter the fitted inference path.
    assert tuple(inspect.signature(fitted.predict).parameters) == ("x",)


def test_oof_balance_profiles_are_fold_train_only_and_clusters_stay_whole() -> None:
    split, tasks, labels = _synthetic_split()
    result = group_oof_class_balanced_predictions(
        split,
        tasks,
        labels,
        _small_spec("true_bidirectional_capacity_matched"),
        ClassBalanceSpec(beta=0.9),
        seed=17,
        maximum_splits=4,
    )
    assert result.predictions.decision_score.shape == (28,)
    assert len(result.training_profiles) == 4
    assert all(profile.unique_queries == 21 for profile in result.training_profiles)
    assert all(profile.task_rows == 21 for profile in result.training_profiles)
    for group in np.unique(split.cluster_codes):
        held_folds = np.unique(result.fold_by_row[split.cluster_codes == group])
        assert len(held_folds) == 1


def test_registered_config_and_reusable_seed_score_interface_are_frozen() -> None:
    path = ROOT / "configs" / "emotiontalk_class_balanced_utility_repair_v1.json"
    config, balance, specs = load_class_balanced_repair_config(path)
    assert config["analysis_role"] == "open_role_repair_2_of_3"
    assert balance.frequency_unit == "unique_query"
    assert balance.oof_frequency_scope == "training_fold_only"
    assert tuple(spec.name for spec in specs) == MODEL_NAMES
    assert tuple(spec.mode for spec in specs) == MODEL_MODES
    assert len(
        {
            trainable_parameter_count(59, spec.hidden_layer_sizes, 2)
            for spec in specs
        }
    ) == 1
    assert tuple(inspect.signature(fit_class_balanced_seed_scores).parameters) == (
        "cache",
        "fit_tasks",
        "fit_task_labels",
        "balance_spec",
        "specs",
        "maximum_splits",
    )


def test_runner_and_cli_expose_no_restricted_role_inputs() -> None:
    expected = (
        "data_dir",
        "feature_path",
        "base_config_path",
        "utility_config_path",
        "repair_config_path",
        "private_cache_path",
        "checkpoint_dir",
        "output_path",
    )
    assert tuple(
        inspect.signature(run_open_role_class_balanced_query_policy).parameters
    ) == expected
    forbidden = {"calibration", "holdout", "sealed", "validation", "test"}
    assert all(not (set(name.split("_")) & forbidden) for name in expected)

    path = ROOT / "scripts" / "run_emotiontalk_class_balanced_utility_repair.py"
    module_spec = importlib.util.spec_from_file_location("class_balanced_cli", path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    destinations = {action.dest for action in module.build_parser()._actions}
    assert destinations == {
        "help",
        "data_dir",
        "feature",
        "base_config",
        "utility_config",
        "repair_config",
        "cache",
        "checkpoint_dir",
        "output",
    }
    assert all(not (set(name.split("_")) & forbidden) for name in destinations)


def test_existing_output_fails_before_any_input_or_label_container_is_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_open_role_class_balanced_query_policy(
            tmp_path / "data",
            tmp_path / "features.npz",
            tmp_path / "base.json",
            tmp_path / "utility.json",
            tmp_path / "repair.json",
            tmp_path / "cache.npz",
            tmp_path / "checkpoints",
            output,
        )
    assert output.read_text(encoding="utf-8") == "keep"


def test_default_model_names_are_stable_and_no_direct_label_feature_exists() -> None:
    specs = default_capacity_matched_specs()
    assert tuple(spec.name for spec in specs) == MODEL_NAMES
    source = (ROOT / "src" / "hva_affect" / "class_balanced_utility_repair.py").read_text(
        encoding="utf-8"
    )
    assert "gold_label_appended_to_features\": False" in source
    assert "strict_epistemic_non_open_label_deserialization_seal_satisfied" in source


def _metric(
    *,
    macro_f1: float,
    accuracy: float,
    nll: float,
    excess: float,
    coverage: float,
) -> dict[str, float | int]:
    return {
        "queries": 20,
        "pooled_macro_f1": macro_f1,
        "pooled_accuracy": accuracy,
        "pooled_weighted_f1": macro_f1,
        "pooled_nll": nll,
        "pooled_brier": 0.5,
        "mean_excess_nll_vs_current": excess,
        "harm_rate_vs_current": 0.2,
        "p90_excess_nll_vs_current": 0.1,
        "cvar90_excess_nll_vs_current": 0.2,
        "actual_history_coverage": coverage,
        "mean_selected_history_count": coverage,
        "mean_candidate_fraction_selected": coverage,
    }


def test_report_contract_exposes_each_utility_seed_and_reference_diagnostics() -> None:
    current = _metric(macro_f1=0.50, accuracy=0.60, nll=1.0, excess=0.0, coverage=0.0)
    all_history = _metric(
        macro_f1=0.52, accuracy=0.61, nll=0.95, excess=-0.05, coverage=0.9
    )
    true = _metric(
        macro_f1=0.53, accuracy=0.62, nll=0.90, excess=-0.10, coverage=0.25
    )
    recency = _metric(
        macro_f1=0.51, accuracy=0.60, nll=0.93, excess=-0.07, coverage=0.25
    )
    grid = [[dict(true) for _ in range(5)] for _ in range(5)]
    recency_grid = [[dict(recency) for _ in range(5)] for _ in range(5)]
    per_seed = _per_utility_seed_aggregate_records(grid)
    assert [row["utility_seed"] for row in per_seed] == [17, 29, 43, 71, 101]
    assert all(
        row["metrics"]["pooled_macro_f1"] == pytest.approx(0.53)
        for row in per_seed
    )
    diagnostic = _true_reference_diagnostics(
        grid,
        recency_grid,
        [dict(current) for _ in range(5)],
        [dict(all_history) for _ in range(5)],
        minimum_macro_f1_gain=0.002,
        minimum_history_coverage=0.1,
    )
    assert diagnostic["registered_current_joint_gate"]["meets_four_of_five"] is True
    assert diagnostic["all_three_reference_diagnostic"]["meets_four_of_five"] is True
    assert len(diagnostic["per_utility_seed"]) == 5


def test_nll_identity_is_enforced_at_one_e_minus_twelve() -> None:
    current = _metric(macro_f1=0.5, accuracy=0.6, nll=1.0, excess=0.0, coverage=0.0)
    strategy = _metric(
        macro_f1=0.51, accuracy=0.61, nll=0.9, excess=-0.1, coverage=0.25
    )
    audit = _nll_identity_audit(
        {"strategy": [[dict(strategy) for _ in range(5)]]},
        {"strategy": [dict(strategy)]},
        [dict(current) for _ in range(5)],
        current,
        [dict(strategy) for _ in range(5)],
        strategy,
    )
    assert audit["passed"] is True
    assert audit["maximum_absolute_error"] <= 1e-12

    bad = dict(strategy)
    bad["mean_excess_nll_vs_current"] = -0.1 + 2e-12
    with pytest.raises(ClassBalancedRepairContractError, match="exceeded 1e-12"):
        _nll_identity_audit(
            {"strategy": [[bad, *[dict(strategy) for _ in range(4)]]]},
            {"strategy": [dict(strategy)]},
            [dict(current) for _ in range(5)],
            current,
            [dict(strategy) for _ in range(5)],
            strategy,
        )


def test_manifest_hash_is_canonical_and_order_independent() -> None:
    left = {"protocol": "v1", "environment": {"python": "3.11.9", "numpy": "2.3.1"}}
    right = {"environment": {"numpy": "2.3.1", "python": "3.11.9"}, "protocol": "v1"}
    first = canonical_manifest_sha256(left)
    second = canonical_manifest_sha256(right)
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
