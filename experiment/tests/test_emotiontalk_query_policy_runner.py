from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_query_policy_runner import (  # noqa: E402
    QueryPolicyContractError,
    aggregate_candidate_draw_scores,
    build_reversible_selected_contexts,
    coverage_matched_recency_contexts,
    fit_query_candidate_coverage_threshold,
    predict_query_context_probabilities,
    predict_query_context_probabilities_by_model,
    query_strategy_metrics,
    run_open_role_query_policy,
    summarize_joint_seed_strategy,
    summarize_utility_seed_strategy,
    validate_strict_past_contexts,
)


@dataclass(frozen=True)
class _Task:
    query_index: int
    candidate_index: int


def test_candidate_scores_are_averaged_across_draws_by_query_candidate() -> None:
    tasks = [_Task(4, 1), _Task(4, 1), _Task(4, 2), _Task(7, 3)]
    result = aggregate_candidate_draw_scores(tasks, np.asarray([0.2, 0.6, -0.1, 0.8]))
    assert result == {4: {1: 0.4, 2: -0.1}, 7: {3: 0.8}}
    with pytest.raises(QueryPolicyContractError, match="task-aligned"):
        aggregate_candidate_draw_scores(tasks, np.asarray([0.2, 0.6]))


def test_fit_threshold_is_frozen_after_query_candidate_draw_aggregation() -> None:
    tasks = [
        _Task(4, 1),
        _Task(4, 1),
        _Task(4, 1),
        _Task(4, 1),
        _Task(4, 2),
        _Task(7, 3),
        _Task(7, 4),
    ]
    # Pair means are [2.5, 2.0, 1.0, -1.0], so a strict score threshold
    # can realize exactly one of four pairs (25%).  A task-row threshold would
    # instead exploit the repeated 10.0 draw and target a different unit.
    threshold, pairs, realized = fit_query_candidate_coverage_threshold(
        tasks,
        np.asarray([10.0, 0.0, 0.0, 0.0, 2.0, 1.0, -1.0]),
        target_coverage=0.25,
    )
    assert pairs == 4
    assert threshold == pytest.approx(2.25)
    assert realized == pytest.approx(0.25)


def test_contexts_are_rebuilt_independently_without_permanent_deletion() -> None:
    histories = ((), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
    original = tuple(tuple(value) for value in histories)
    first = build_reversible_selected_contexts(
        [3, 4],
        histories,
        {3: {0: 0.9, 1: -0.2, 2: 0.8}, 4: {1: 0.7, 3: -0.5}},
        threshold=0.0,
    )
    second = build_reversible_selected_contexts(
        [3, 4],
        histories,
        {3: {1: 0.9}, 4: {0: 0.9, 3: 0.9}},
        threshold=0.0,
    )
    assert first == ((0, 2), (1,))
    assert second == ((1,), (0, 3))
    assert histories == original


def test_empty_or_threshold_equal_selection_falls_back_to_current_only() -> None:
    histories = ((), (0,), (0, 1))
    contexts = build_reversible_selected_contexts(
        [1, 2],
        histories,
        {2: {0: 0.5, 1: 0.49}},
        threshold=0.5,
    )
    assert contexts == ((), ())


def test_strict_past_and_unique_query_contracts_fail_closed() -> None:
    histories = ((), (0,), (0, 1), (0, 1, 2))
    with pytest.raises(QueryPolicyContractError, match="non-past"):
        build_reversible_selected_contexts(
            [3], histories, {3: {99: 1.0}}, threshold=0.0
        )
    with pytest.raises(QueryPolicyContractError, match="unique query"):
        build_reversible_selected_contexts(
            [3, 3], histories, {3: {0: 1.0}}, threshold=0.0
        )
    with pytest.raises(QueryPolicyContractError, match="non-past"):
        validate_strict_past_contexts([3], [(0, 99)], histories)


def test_coverage_matched_recency_preserves_per_query_cardinality() -> None:
    histories = ((), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
    reference = ((0, 2), (1,))
    recency = coverage_matched_recency_contexts([3, 4], histories, reference)
    assert recency == ((1, 2), (3,))
    assert [len(value) for value in recency] == [len(value) for value in reference]


class _FakeModel:
    classes_ = np.arange(7, dtype=np.int64)

    def __init__(self, preferred: int) -> None:
        self.preferred = preferred

    def predict_proba(self, features) -> np.ndarray:
        probability = np.full((features.shape[0], 7), 0.05, dtype=np.float64)
        probability[:, self.preferred] = 0.70
        return probability


def test_set_inference_emits_one_float64_probability_row_per_query() -> None:
    dense = np.arange(15, dtype=np.float64).reshape(5, 3)
    current = {
        "text": sparse.csr_matrix(dense),
        "audio": dense + 10,
        "video": dense + 20,
    }
    quality = np.ones((5, 2), dtype=np.float64)
    histories = ((), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
    probability = predict_query_context_probabilities(
        [_FakeModel(1), _FakeModel(2)],
        current,
        quality,
        ("q0", "q1"),
        [3, 4],
        [(0, 2), ()],
        histories,
        n_classes=7,
    )
    assert probability.shape == (2, 7)
    assert probability.dtype == np.float64
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    per_model = predict_query_context_probabilities_by_model(
        [_FakeModel(1), _FakeModel(2)],
        current,
        quality,
        ("q0", "q1"),
        [3, 4],
        [(0, 2), ()],
        histories,
        n_classes=7,
    )
    assert per_model.shape == (2, 2, 7)
    np.testing.assert_allclose(probability, per_model.mean(axis=0))
    with pytest.raises(QueryPolicyContractError, match="exactly one row per query"):
        predict_query_context_probabilities(
            [_FakeModel(1)],
            current,
            quality,
            ("q0", "q1"),
            [3, 3],
            [(0,), (1,)],
            histories,
            n_classes=7,
        )


def test_five_seed_summary_reports_joint_success_count_and_four_of_five() -> None:
    current = {
        "queries": 10,
        "pooled_macro_f1": 0.50,
        "pooled_accuracy": 0.50,
        "pooled_weighted_f1": 0.50,
        "pooled_nll": 1.0,
        "pooled_brier": 0.8,
        "mean_excess_nll_vs_current": 0.0,
        "harm_rate_vs_current": 0.0,
        "p90_excess_nll_vs_current": 0.0,
        "cvar90_excess_nll_vs_current": 0.0,
        "actual_history_coverage": 0.0,
        "mean_selected_history_count": 0.0,
        "mean_candidate_fraction_selected": 0.0,
    }
    records = []
    for index in range(5):
        record = dict(current)
        record["pooled_macro_f1"] = 0.51 if index < 4 else 0.49
        record["mean_excess_nll_vs_current"] = -0.1 if index < 4 else 0.1
        record["actual_history_coverage"] = 0.2
        records.append(record)
    summary = summarize_utility_seed_strategy(
        records,
        current,
        minimum_macro_f1_gain=0.002,
        minimum_history_coverage=0.1,
    )
    assert summary["successful_utility_seeds_out_of_five"] == 4
    assert summary["meets_four_of_five"] is True
    assert "mean" in summary["metrics"]["pooled_macro_f1"]
    assert "std" in summary["metrics"]["pooled_macro_f1"]


def test_joint_estimand_uses_all_five_by_five_seed_cells() -> None:
    current = {
        "queries": 10,
        "pooled_macro_f1": 0.50,
        "pooled_accuracy": 0.50,
        "pooled_weighted_f1": 0.50,
        "pooled_nll": 1.0,
        "pooled_brier": 0.8,
        "mean_excess_nll_vs_current": 0.0,
        "harm_rate_vs_current": 0.0,
        "p90_excess_nll_vs_current": 0.0,
        "cvar90_excess_nll_vs_current": 0.0,
        "actual_history_coverage": 0.0,
        "mean_selected_history_count": 0.0,
        "mean_candidate_fraction_selected": 0.0,
    }
    current_records = [dict(current) for _ in range(5)]
    grid = []
    for utility_index in range(5):
        row = []
        for base_index in range(5):
            record = dict(current)
            record["pooled_macro_f1"] = 0.51 if utility_index < 4 else 0.49
            record["mean_excess_nll_vs_current"] = (
                -0.1 if utility_index < 4 else 0.1
            )
            record["actual_history_coverage"] = 0.2
            record["pooled_accuracy"] += base_index / 100.0
            row.append(record)
        grid.append(row)
    summary = summarize_joint_seed_strategy(
        grid,
        current_records,
        minimum_macro_f1_gain=0.002,
        minimum_history_coverage=0.1,
    )
    assert summary["joint_seed_grid_count"] == 25
    assert summary["successful_utility_seeds_out_of_five"] == 4
    assert summary["meets_four_of_five"] is True
    assert summary["metrics_across_25_seed_combinations"]["pooled_accuracy"][
        "mean"
    ] == pytest.approx(0.52)


def test_query_nll_and_excess_share_the_same_extreme_probability_floor() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    current = np.empty((2, 7), dtype=np.float64)
    selected = np.empty((2, 7), dtype=np.float64)
    current[0] = np.asarray([1e-30, *([1.0 / 6.0] * 6)])
    selected[0] = np.asarray([1e-40, *([1.0 / 6.0] * 6)])
    current[1] = np.asarray([0.8 / 6.0, 0.2, *([0.8 / 6.0] * 5)])
    selected[1] = np.asarray([0.6 / 6.0, 0.4, *([0.6 / 6.0] * 5)])
    histories = ((), (0,), (), (2,))
    queries = (1, 3)
    clusters = np.asarray(["dialogue-a", "dialogue-b"], dtype=object)

    current_metrics = query_strategy_metrics(
        labels,
        current,
        current,
        ((), ()),
        histories,
        queries,
        clusters,
        ece_bins=5,
    )
    selected_metrics = query_strategy_metrics(
        labels,
        selected,
        current,
        ((0,), (2,)),
        histories,
        queries,
        clusters,
        ece_bins=5,
    )

    assert selected_metrics["pooled_nll"] - current_metrics["pooled_nll"] == (
        pytest.approx(selected_metrics["mean_excess_nll_vs_current"], abs=1e-15)
    )


def test_runner_and_cli_expose_no_leakage_role_parameters() -> None:
    expected = (
        "data_dir",
        "feature_path",
        "base_config_path",
        "utility_config_path",
        "private_cache_path",
        "checkpoint_dir",
        "output_path",
    )
    assert tuple(inspect.signature(run_open_role_query_policy).parameters) == expected
    forbidden = {"calibration", "holdout", "sealed", "validation", "test"}
    assert all(not (set(name.split("_")) & forbidden) for name in expected)

    path = ROOT / "scripts" / "run_emotiontalk_query_policy.py"
    spec = importlib.util.spec_from_file_location("query_policy_cli", path)
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
        "output",
    }
    assert all(not (set(name.split("_")) & forbidden) for name in destinations)


def test_existing_output_fails_before_any_real_input_is_read(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_open_role_query_policy(
            tmp_path / "data",
            tmp_path / "features.npz",
            tmp_path / "base.json",
            tmp_path / "utility.json",
            tmp_path / "cache.npz",
            tmp_path / "checkpoints",
            output,
        )
    assert output.read_text(encoding="utf-8") == "keep"
