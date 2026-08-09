from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import itertools
import math
import time

import numpy as np
import pytest

from hva_affect import harmbench_erc_selection_statistics as statistics
from hva_affect import harmbench_erc_selection_prelabel as prelabel
from hva_affect.harmbench_erc_metrics import (
    classification_metrics,
    paired_true_class_regret,
)
from hva_affect.harmbench_erc_protocol_v2 import (
    EXPECTED_HISTORY_STRATEGY_ORDER,
    EXPECTED_MODEL_ORDER,
    EXPECTED_SELECTION_DATASETS,
    PROTOCOL_V2_CANONICAL_SHA256,
)


def _probability_for_prediction(prediction: int, classes: int, strength: float) -> np.ndarray:
    result = np.full(classes, (1.0 - strength) / (classes - 1), dtype=np.float64)
    result[prediction] = strength
    return result


def _dataset_source(
    dataset_index: int,
    *,
    queries: int,
    groups: int,
    all_eligible: bool = False,
) -> dict[str, object]:
    classes = 3
    row_ids = np.arange(1000 * (dataset_index + 1), 1000 * (dataset_index + 1) + queries, dtype=np.int64)
    group_tokens = np.asarray(
        [f"d{dataset_index}_g{min(groups - 1, index * groups // queries)}" for index in range(queries)],
        dtype=np.str_,
    )
    labels = np.asarray([(index + dataset_index) % classes for index in range(queries)], dtype=np.int64)
    if all_eligible:
        eligible = np.ones(queries, dtype=np.bool_)
        depth = np.asarray([1 + (index % 9) for index in range(queries)], dtype=np.int64)
    else:
        eligible = np.ones(queries, dtype=np.bool_)
        eligible[::5] = False
        if not np.any(eligible):
            eligible[-1] = True
        depth = np.zeros(queries, dtype=np.int64)
        depth_values = (1, 2, 3, 4, 8, 1, 2, 5, 9)
        depth[eligible] = np.asarray(
            [depth_values[index % len(depth_values)] for index in np.flatnonzero(eligible)],
            dtype=np.int64,
        )

    model_values: dict[str, object] = {}
    for model_index, model_id in enumerate(EXPECTED_MODEL_ORDER):
        current = np.empty((5, queries, classes), dtype=np.float64)
        for seed in range(5):
            for query in range(queries):
                # Seed-specific decisions make probability-then-seed-mean observably wrong.
                if (seed + query + dataset_index) % 4 == 0:
                    prediction = (int(labels[query]) + 1) % classes
                else:
                    prediction = int(labels[query])
                current[seed, query] = _probability_for_prediction(
                    prediction, classes, 0.64
                )

        strategy_values: dict[str, object] = {}
        for strategy_index, strategy_id in enumerate(EXPECTED_HISTORY_STRATEGY_ORDER):
            if strategy_id == "dialogue_all_past":
                query_mask = eligible.copy()
            elif strategy_id == "same_speaker_all_past":
                query_mask = eligible & ((np.arange(queries) + dataset_index) % 3 != 0)
            elif strategy_id == "recent_k3":
                query_mask = eligible & (np.arange(queries) % 2 == 0)
            elif strategy_id == "similarity_top3":
                query_mask = eligible & (np.arange(queries) % 2 == 1)
            else:
                query_mask = eligible & (np.arange(queries) % 4 != 1)
            mask = np.broadcast_to(query_mask[None, :], (5, queries)).copy()
            history = current.copy()
            for seed in range(5):
                for query in np.flatnonzero(query_mask):
                    # Mix benefits and harms without any post-outcome selection.
                    if (seed + query + strategy_index + dataset_index) % 3:
                        prediction = int(labels[query])
                    else:
                        prediction = (int(labels[query]) + 1) % classes
                    history[seed, query] = _probability_for_prediction(
                        prediction, classes, 0.72
                    )
            strategy_values[strategy_id] = {
                "probability": history,
                "use_history_mask": mask,
            }
        model_values[model_id] = {
            "current_probability": current,
            "strategies": strategy_values,
        }
    return {
        "labels": labels,
        "protocol_row_ids": row_ids,
        "group_tokens": group_tokens,
        "class_tokens": np.asarray(["neutral", "positive", "negative"], dtype=np.str_),
        "dialogue_history_eligible": eligible,
        "dialogue_depth": depth,
        "models": model_values,
    }


def _raw_fixture(*, monte_carlo: bool = False) -> dict[str, object]:
    if monte_carlo:
        return {
            "EmotionTalk": _dataset_source(0, queries=11, groups=11, all_eligible=True),
            "MELD": _dataset_source(1, queries=11, groups=11, all_eligible=True),
        }
    # Unequal Q checks equal-dataset weighting; four groups each keeps exact swaps small.
    return {
        "EmotionTalk": _dataset_source(0, queries=8, groups=4),
        "MELD": _dataset_source(1, queries=12, groups=4),
    }


@pytest.fixture(scope="module")
def exact_result() -> tuple[statistics.JointSelectionEvaluationInputs, dict[str, object], float]:
    inputs = statistics._make_trusted_synthetic_joint_selection_evaluation_inputs(
        _raw_fixture()
    )
    started = time.perf_counter()
    report = statistics._evaluate_trusted_synthetic_selection_statistics(inputs)
    return inputs, report, time.perf_counter() - started


def _dataset_strategy(report: dict[str, object], dataset_index: int, model_index: int, strategy_id: str) -> dict[str, object]:
    strategies = report["datasets"][dataset_index]["models"][model_index]["strategies"]
    return next(row for row in strategies if row["strategy_id"] == strategy_id)


def test_exact_report_is_frozen_aggregate_only_and_runtime_is_recorded(exact_result) -> None:
    _inputs, report, runtime = exact_result
    statistics.validate_selection_statistics_report(report)
    assert report["schema_version"] == statistics.SELECTION_STATISTICS_SCHEMA
    assert report["protocol_canonical_sha256"] == PROTOCOL_V2_CANONICAL_SHA256
    assert report["selection_result_status"] == statistics.EXPLORATORY_STATUS
    assert report["confirmatory_claim"] is False
    assert report["analysis_contract"]["bootstrap"]["replicates"] == 10_000
    assert report["analysis_contract"]["bootstrap"]["random_seed"] == 20_260_810
    assert report["analysis_contract"]["randomization"]["method"] == "exact"
    assert report["analysis_contract"]["randomization"]["combined_typed_clusters"] == 8
    assert report["analysis_contract"]["randomization"]["assignments"] == 256
    assert len(report["joint_cells"]) == 15
    assert all(
        row["randomization"]["status"]
        in {
            "evaluated_primary_family",
            "evaluated_secondary_not_in_primary_family",
        }
        and isinstance(row["randomization"]["macro_f1_raw_p_value"], float)
        and isinstance(row["randomization"]["mean_regret_raw_p_value"], float)
        for row in report["joint_cells"]
    )
    assert len(report["primary_holm_family"]) == 6
    assert len(report["no_harm_gate"]["cells"]) == 6
    assert runtime < 45.0


def test_primary_points_are_per_seed_E_dialogue_and_equal_dataset_weight(exact_result) -> None:
    inputs, report, _runtime = exact_result
    dataset_points: list[tuple[float, float]] = []
    for dataset in inputs._datasets:
        model = dataset.models[0]
        strategy = model.strategies[1]  # same_speaker_all_past
        eligible = dataset.dialogue_history_eligible
        labels = dataset.labels[eligible]
        macro = []
        regret = []
        for seed in range(5):
            current = model.current_probability[seed, eligible]
            history = strategy.probability[seed, eligible]
            macro.append(
                classification_metrics(labels, history)["macro_f1"]
                - classification_metrics(labels, current)["macro_f1"]
            )
            regret.append(float(np.mean(paired_true_class_regret(labels, current, history))))
        dataset_points.append((float(np.mean(macro)), float(np.mean(regret))))
    for index, expected in enumerate(dataset_points):
        observed = _dataset_strategy(report, index, 0, "same_speaker_all_past")
        assert observed["macro_f1_difference"]["point"] == pytest.approx(expected[0])
        assert observed["mean_regret"]["point"] == pytest.approx(expected[1])
    joint = next(
        row
        for row in report["joint_cells"]
        if row["model_id"] == EXPECTED_MODEL_ORDER[0]
        and row["strategy_id"] == "same_speaker_all_past"
    )
    assert joint["macro_f1_difference"]["point"] == pytest.approx(
        0.5 * (dataset_points[0][0] + dataset_points[1][0])
    )
    assert joint["mean_regret"]["point"] == pytest.approx(
        0.5 * (dataset_points[0][1] + dataset_points[1][1])
    )


def test_fallback_rows_remain_in_E_dialogue_and_are_exact_zero(exact_result) -> None:
    inputs, report, _runtime = exact_result
    dataset = inputs._datasets[0]
    model = dataset.models[0]
    strategy = model.strategies[1]
    eligible = dataset.dialogue_history_eligible
    fallback = eligible & ~strategy.use_history_mask[0]
    assert np.any(fallback)
    assert np.array_equal(
        strategy.probability[:, fallback], model.current_probability[:, fallback]
    )
    observed = _dataset_strategy(report, 0, 0, "same_speaker_all_past")["secondary"]
    assert observed["coverage"]["eligible_queries"] == int(eligible.sum())
    assert observed["coverage"]["history_context_nonempty_queries"] == int(
        strategy.use_history_mask[0, eligible].sum()
    )
    assert observed["sign_severity"]["counts"]["exact_zero_including_fallback"] >= int(
        fallback.sum() * 5
    )


def test_shared_bootstrap_and_randomization_are_bit_reproducible(exact_result) -> None:
    inputs, first, _runtime = exact_result
    second = statistics._evaluate_trusted_synthetic_selection_statistics(inputs)
    assert second == first
    assert first["analysis_contract"]["bootstrap"]["shared_plan_across_all_cells_metrics_strategies"] is True
    raw_p = [row["raw_p_value"] for row in first["primary_holm_family"]]
    # Model fixtures are deliberately identical enough to expose a reset-per-cell RNG bug.
    assert raw_p[0:2] == raw_p[2:4] == raw_p[4:6]


def _brute_force_macro_randomization(inputs: statistics.JointSelectionEvaluationInputs) -> float:
    cluster_keys: list[tuple[str, str]] = []
    for dataset in inputs._datasets:
        for token in dataset.group_tokens.tolist():
            key = (dataset.dataset_id, str(token))
            if key not in cluster_keys:
                cluster_keys.append(key)
    observed_dataset: list[float] = []
    for dataset in inputs._datasets:
        model = dataset.models[0]
        strategy = model.strategies[1]
        eligible = dataset.dialogue_history_eligible
        labels = dataset.labels[eligible]
        observed_dataset.append(
            float(
                np.mean(
                    [
                        classification_metrics(labels, strategy.probability[seed, eligible])["macro_f1"]
                        - classification_metrics(labels, model.current_probability[seed, eligible])["macro_f1"]
                        for seed in range(5)
                    ]
                )
            )
        )
    observed = abs(0.5 * sum(observed_dataset))
    exceed = 0
    for bits in itertools.product((False, True), repeat=len(cluster_keys)):
        bit_by_key = dict(zip(cluster_keys, bits))
        joint = 0.0
        for dataset in inputs._datasets:
            model = dataset.models[0]
            strategy = model.strategies[1]
            eligible = dataset.dialogue_history_eligible
            labels = dataset.labels[eligible]
            swap = np.asarray(
                [bit_by_key[(dataset.dataset_id, str(token))] for token in dataset.group_tokens],
                dtype=np.bool_,
            )[eligible]
            seed_values = []
            for seed in range(5):
                history = strategy.probability[seed, eligible]
                current = model.current_probability[seed, eligible]
                candidate = np.where(swap[:, None], current, history)
                anchor = np.where(swap[:, None], history, current)
                seed_values.append(
                    classification_metrics(labels, candidate)["macro_f1"]
                    - classification_metrics(labels, anchor)["macro_f1"]
                )
            joint += 0.5 * float(np.mean(seed_values))
        exceed += int(abs(joint) >= observed)
    return exceed / (2 ** len(cluster_keys))


def test_exact_randomization_recomputes_nonadditive_macro_f1_by_whole_cluster(exact_result) -> None:
    inputs, report, _runtime = exact_result
    expected = _brute_force_macro_randomization(inputs)
    observed = next(
        row["raw_p_value"]
        for row in report["primary_holm_family"]
        if row["model_id"] == EXPECTED_MODEL_ORDER[0] and row["metric_id"] == "Macro-F1"
    )
    assert observed == pytest.approx(expected, abs=0.0)


def test_depth_strata_fail_closed_without_merging(exact_result) -> None:
    _inputs, report, _runtime = exact_result
    secondary = _dataset_strategy(report, 0, 0, "same_speaker_all_past")["secondary"]
    assert [row["stratum_id"] for row in secondary["depth_strata"]] == [
        "depth_1",
        "depth_2_3",
        "depth_4_7",
        "depth_ge_8",
    ]
    for row in secondary["depth_strata"]:
        if row["independent_clusters"] < 2:
            assert row["status"] == "not_estimable"
            assert row["mean_regret"] is None
    assert secondary["worst_depth_stratum"]["stratum_id"] in {
        row["stratum_id"]
        for row in secondary["depth_strata"]
        if row["status"] == "estimable"
    }


def test_fake_seal_mutation_duplicate_rows_nan_and_fallback_attacks_fail(exact_result) -> None:
    inputs, _report, _runtime = exact_result
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.JointSelectionEvaluationInputs(
            schema_version=statistics.JOINT_INPUT_SCHEMA,
            protocol_canonical_sha256=PROTOCOL_V2_CANONICAL_SHA256,
            dataset_order=tuple(EXPECTED_SELECTION_DATASETS),
            model_order=tuple(EXPECTED_MODEL_ORDER),
            strategy_order=("x",),
            training_seed_order=(17, 29, 43, 71, 101),
            source_kind="attempt_bound_activated_labels",
            _input_sha256="0" * 64,
            _datasets=(),
            _seal=object(),
        )
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics._evaluate_trusted_synthetic_selection_statistics(
            replace(inputs, _input_sha256="0" * 64)
        )
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.evaluate_selection_statistics(inputs)

    raw = _raw_fixture()
    raw["EmotionTalk"]["protocol_row_ids"][1] = raw["EmotionTalk"]["protocol_row_ids"][0]
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError, match="unique"):
        statistics._make_trusted_synthetic_joint_selection_evaluation_inputs(raw)
    raw = _raw_fixture()
    raw["EmotionTalk"]["models"][EXPECTED_MODEL_ORDER[0]]["current_probability"][0, 0, 0] = np.nan
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError, match="invalid probability"):
        statistics._make_trusted_synthetic_joint_selection_evaluation_inputs(raw)
    raw = _raw_fixture()
    model = raw["EmotionTalk"]["models"][EXPECTED_MODEL_ORDER[0]]
    strategy = model["strategies"]["same_speaker_all_past"]
    fallback = ~strategy["use_history_mask"]
    seed, query = np.argwhere(fallback)[0]
    strategy["probability"][seed, query] = np.asarray([0.8, 0.1, 0.1])
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError, match="fallback"):
        statistics._make_trusted_synthetic_joint_selection_evaluation_inputs(raw)


def test_exact_schema_privacy_nonfinite_holm_and_no_harm_attacks_fail(exact_result) -> None:
    _inputs, report, _runtime = exact_result
    attack = deepcopy(report)
    attack["datasets"][0]["labels"] = [0]
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.validate_selection_statistics_report(attack)
    attack = deepcopy(report)
    attack["joint_cells"][0]["mean_regret"]["point"] = float("nan")
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.validate_selection_statistics_report(attack)
    attack = deepcopy(report)
    attack["primary_holm_family"].pop()
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.validate_selection_statistics_report(attack)
    attack = deepcopy(report)
    attack["no_harm_gate"]["cells"][0]["cell_pass"] = not attack["no_harm_gate"]["cells"][0]["cell_pass"]
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.validate_selection_statistics_report(attack)
    attack = deepcopy(report)
    attack["confirmatory_claim"] = True
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics.validate_selection_statistics_report(attack)


def test_holm_is_exact_six_stable_step_down_and_substantive_is_separate() -> None:
    adjusted, rejected = statistics._holm_step_down([0.01, 0.01, 0.03, 0.2, 0.5, 1.0])
    assert adjusted == pytest.approx([0.06, 0.06, 0.12, 0.6, 1.0, 1.0])
    assert rejected == [False, False, False, False, False, False]
    with pytest.raises(statistics.HarmBenchSelectionStatisticsError):
        statistics._holm_step_down([0.01] * 5)


def test_module_statistics_core_has_no_file_io_calls() -> None:
    tree = ast.parse(inspect.getsource(statistics))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "open",
            "read_text",
            "read_bytes",
            "write_text",
            "write_bytes",
        }:
            forbidden.append(node.func.id)
    assert forbidden == []


def test_loader_revalidates_marker_after_label_activation_boundary(monkeypatch) -> None:
    attempt = object.__new__(prelabel.AttemptStartedCapability)
    bundle = object.__new__(prelabel.LoadedSelectionPrelabelBundle)
    object.__setattr__(bundle, "_seal", prelabel._BUNDLE_SEAL)
    object.__setattr__(attempt, "_seal", prelabel._ATTEMPT_SEAL)
    object.__setattr__(attempt, "_prelabel", bundle)

    def changed_marker(_capability):
        raise prelabel.HarmBenchSelectionPrelabelError("marker changed after activation")

    monkeypatch.setattr(
        prelabel, "_revalidate_attempt_started_capability", changed_marker
    )
    with pytest.raises(
        statistics.HarmBenchSelectionStatisticsError,
        match="marker/prelabel live revalidation failed",
    ):
        statistics.load_joint_selection_evaluation_inputs(attempt, [])


def test_monte_carlo_100k_fixed_seed_plus_one_formula_and_runtime() -> None:
    inputs = statistics._make_trusted_synthetic_joint_selection_evaluation_inputs(
        _raw_fixture(monte_carlo=True)
    )
    started = time.perf_counter()
    report = statistics._evaluate_trusted_synthetic_selection_statistics(inputs)
    runtime = time.perf_counter() - started
    contract = report["analysis_contract"]["randomization"]
    assert contract["method"] == "monte_carlo"
    assert contract["combined_typed_clusters"] == 22
    assert contract["assignments"] == 100_000
    assert contract["random_seed"] == 20_260_811
    for row in report["primary_holm_family"]:
        scaled = row["raw_p_value"] * 100_001
        assert scaled >= 1.0
        assert scaled == pytest.approx(round(scaled), abs=1e-9)
    assert runtime < 90.0
