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

from hva_affect.distributional_utility_repair import ModelPrediction  # noqa: E402
from hva_affect.emotiontalk_distributional_query_policy_runner import (  # noqa: E402
    MODEL_NAMES,
    RECENCY_STRATEGY,
    REPORT_SCHEMA_VERSION,
    STRATEGY_NAMES,
    DistributionalQueryPolicyContractError,
    audit_joint_seed_nll_identity,
    assert_aggregate_query_report,
    build_distributional_seed_contexts,
    run_open_role_distributional_query_policy,
    summarize_fold_regeneration_sensitivity,
    validate_cluster_pure_fold_positions,
)


@dataclass(frozen=True)
class _Task:
    query_index: int
    candidate_index: int


def _model_predictions(score: np.ndarray) -> dict[str, ModelPrediction]:
    return {
        name: ModelPrediction(
            decision=np.asarray(score, dtype=np.float64).copy(),
            heads=(),
            head_targets=(),
        )
        for name in MODEL_NAMES
    }


def test_distributional_threshold_is_frozen_after_query_candidate_aggregation() -> None:
    fit_tasks = (
        _Task(4, 0),
        _Task(4, 0),
        _Task(4, 1),
        _Task(4, 2),
        _Task(7, 3),
        _Task(7, 4),
        _Task(7, 5),
        _Task(7, 6),
        _Task(7, 2),
    )
    # Pair (4, 0) averages to 5.0; the eight pair scores are therefore
    # [5, 4, 3, 2, 1, 0, -1, -2] and exact top-quarter coverage freezes at 3.5.
    fit_score = np.asarray([10.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    selection_tasks = (
        _Task(4, 0),
        _Task(4, 1),
        _Task(7, 3),
        _Task(7, 3),
    )
    selection_score = np.asarray([3.6, 3.4, 4.0, 4.0])
    histories = ((), (), (), (), (0, 1, 2), (), (), (3, 4, 5, 6))

    states = build_distributional_seed_contexts(
        fit_tasks,
        selection_tasks,
        _model_predictions(fit_score),
        _model_predictions(selection_score),
        (4, 7),
        histories,
    )

    assert tuple(states) == MODEL_NAMES
    for state in states.values():
        assert state.fit_query_candidate_pairs == 8
        assert state.realized_fit_query_candidate_coverage == pytest.approx(0.25)
        assert state.threshold == pytest.approx(3.5)
        assert state.contexts == ((0,), (3,))


def test_distributional_context_builder_rejects_changed_model_order() -> None:
    predictions = _model_predictions(np.asarray([0.2, 0.1, -0.1, -0.2]))
    reversed_predictions = dict(reversed(tuple(predictions.items())))
    tasks = tuple(_Task(2, candidate) for candidate in range(4))
    with pytest.raises(
        DistributionalQueryPolicyContractError,
        match="registered model order",
    ):
        build_distributional_seed_contexts(
            tasks,
            tasks,
            reversed_predictions,
            predictions,
            (2,),
            ((), (), (0, 1, 2, 3)),
        )


def test_frozen_checkpoint_positions_require_exact_cluster_pure_cover() -> None:
    clusters = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    valid = {
        "fold_1.npz": np.asarray([0, 1, 4, 5], dtype=np.int64),
        "fold_2.npz": np.asarray([2, 3], dtype=np.int64),
    }
    audit = validate_cluster_pure_fold_positions(
        valid,
        task_count=6,
        cluster_codes=clusters,
    )
    assert audit["fit_checkpoint_exact_nonoverlapping_cover"] is True
    assert audit["fit_checkpoint_positions_unique_within_fold"] is True
    assert audit["fit_checkpoint_cluster_pure_partition"] is True
    assert audit["fit_checkpoint_task_count"] == 6
    assert audit["frozen_position_count_by_fold"] == {
        "fold_1.npz": 4,
        "fold_2.npz": 2,
    }
    assert all(
        len(value) == 64
        for value in audit["frozen_position_sha256_by_fold"].values()
    )
    with pytest.raises(
        DistributionalQueryPolicyContractError,
        match="split across frozen checkpoints",
    ):
        validate_cluster_pure_fold_positions(
            {
                "fold_1.npz": np.asarray([0, 2, 4]),
                "fold_2.npz": np.asarray([1, 3, 5]),
            },
            task_count=6,
            cluster_codes=clusters,
        )
    with pytest.raises(
        DistributionalQueryPolicyContractError,
        match="exactly cover",
    ):
        validate_cluster_pure_fold_positions(
            {
                "fold_1.npz": np.asarray([0, 1]),
                "fold_2.npz": np.asarray([2, 3]),
            },
            task_count=6,
            cluster_codes=clusters,
        )


def test_runtime_fold_regeneration_is_sensitivity_only_and_can_mismatch() -> None:
    sensitivity = summarize_fold_regeneration_sensitivity(
        {
            "fold_1.npz": np.asarray([0, 1], dtype=np.int64),
            "fold_2.npz": np.asarray([2, 3], dtype=np.int64),
        },
        {
            "fold_1.npz": np.asarray([0, 2], dtype=np.int64),
            "fold_2.npz": np.asarray([1, 3], dtype=np.int64),
        },
        task_count=4,
    )

    assert sensitivity["runtime_regenerated_fold_assignment_role"].startswith(
        "sensitivity only"
    )
    assert sensitivity["runtime_regenerated_fold_assignment_matches_frozen"] is False
    assert sensitivity["runtime_regenerated_fold_assignment_mismatched_tasks"] == 2
    assert sensitivity["environment_drift_is_not_interpreted_as_performance_difference"]


def test_joint_seed_nll_identity_is_checked_for_every_matching_base_cell() -> None:
    current = tuple(
        {"pooled_nll": 1.0 + 0.1 * base_index}
        for base_index in range(5)
    )
    strategies = (*STRATEGY_NAMES.values(), RECENCY_STRATEGY)
    grid: dict[str, list[list[dict[str, float]]]] = {}
    for strategy_index, strategy_name in enumerate(strategies):
        utility_rows: list[list[dict[str, float]]] = []
        for utility_index in range(5):
            base_rows: list[dict[str, float]] = []
            for base_index in range(5):
                excess = 0.01 * (strategy_index + 1) - 0.001 * utility_index
                base_rows.append(
                    {
                        "pooled_nll": current[base_index]["pooled_nll"] + excess,
                        "mean_excess_nll_vs_current": excess,
                    }
                )
            utility_rows.append(base_rows)
        grid[strategy_name] = utility_rows

    audit = audit_joint_seed_nll_identity(grid, current)
    assert audit["nll_identity_joint_cells_checked"] == 125
    assert audit["nll_identity_satisfied"] is True
    assert audit["nll_identity_maximum_absolute_error"] <= 1e-12

    grid[strategies[0]][0][0]["mean_excess_nll_vs_current"] += 1e-6
    with pytest.raises(
        DistributionalQueryPolicyContractError,
        match="pooled NLL and mean excess NLL are misaligned",
    ):
        audit_joint_seed_nll_identity(grid, current)


def test_distributional_query_report_fails_closed_on_row_material() -> None:
    safe = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "summary": {"seeds": [17, 29, 43, 71, 101], "mean": 0.1},
    }
    assert_aggregate_query_report(safe)
    with pytest.raises(DistributionalQueryPolicyContractError, match="forbidden field"):
        assert_aggregate_query_report(
            {"schema_version": REPORT_SCHEMA_VERSION, "row_ids": [1, 2]}
        )
    with pytest.raises(DistributionalQueryPolicyContractError, match="ndarray"):
        assert_aggregate_query_report(
            {"schema_version": REPORT_SCHEMA_VERSION, "summary": np.zeros(2)}
        )
    with pytest.raises(DistributionalQueryPolicyContractError, match="overlong list"):
        assert_aggregate_query_report(
            {"schema_version": REPORT_SCHEMA_VERSION, "summary": list(range(21))}
        )


def test_distributional_query_runner_and_cli_expose_no_restricted_role_inputs() -> None:
    expected = (
        "data_dir",
        "feature_path",
        "base_config_path",
        "utility_config_path",
        "private_cache_path",
        "checkpoint_dir",
        "lineage_report_path",
        "repair_config_path",
        "output_path",
    )
    assert tuple(
        inspect.signature(run_open_role_distributional_query_policy).parameters
    ) == expected
    forbidden = {"calibration", "holdout", "sealed", "validation", "test"}
    assert all(not (set(name.split("_")) & forbidden) for name in expected)

    path = ROOT / "scripts" / "run_emotiontalk_distributional_query_policy.py"
    spec = importlib.util.spec_from_file_location("distributional_query_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    destinations = {action.dest for action in module.build_parser()._actions}
    assert destinations == {
        "help",
        "data_dir",
        "feature",
        "base_config",
        "utility_config",
        "cache",
        "checkpoint_dir",
        "lineage_report",
        "repair_config",
        "output",
    }
    assert all(not (set(name.split("_")) & forbidden) for name in destinations)


def test_distributional_query_schema_is_explicitly_post_nll_fix_v3() -> None:
    assert REPORT_SCHEMA_VERSION.endswith("_v3")
