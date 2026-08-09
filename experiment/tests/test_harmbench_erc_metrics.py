from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.harmbench_erc_metrics import (  # noqa: E402
    HarmBenchMetricError,
    empirical_upper_cvar,
    ensure_finite_public_tree,
    evaluate_frozen_policy,
    evaluate_frozen_thresholds,
    hybrid_probability,
    paired_true_class_regret,
    validated_probability,
)


def toy_probabilities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    current = np.asarray(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]], dtype=np.float64
    )
    strategy = np.asarray(
        [[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.3, 0.7]], dtype=np.float64
    )
    return labels, current, strategy


def test_paired_true_class_regret_is_strategy_minus_current_nll() -> None:
    labels, current, strategy = toy_probabilities()
    observed = paired_true_class_regret(labels, current, strategy)
    expected = -np.log(strategy[np.arange(4), labels]) + np.log(
        current[np.arange(4), labels]
    )
    assert np.allclose(observed, expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(
        paired_true_class_regret(labels, current, current), np.zeros(4, dtype=np.float64)
    )


def test_empirical_cvar_handles_zero_mass_at_quantile_exactly() -> None:
    values = np.asarray([0.0] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0])
    assert empirical_upper_cvar(values, alpha=0.90) == pytest.approx(1.5)
    assert empirical_upper_cvar(np.asarray([1.0, 2.0, 3.0]), alpha=0.90) == 3.0


def test_hybrid_probability_uses_candidate_only_where_selected() -> None:
    _, current, strategy = toy_probabilities()
    selected = np.asarray([True, False, False, True])
    hybrid = hybrid_probability(current, strategy, selected)
    assert np.array_equal(hybrid[[0, 3]], strategy[[0, 3]])
    assert np.array_equal(hybrid[[1, 2]], current[[1, 2]])


def test_frozen_policy_distinguishes_population_and_conditional_risk() -> None:
    labels, current, strategy = toy_probabilities()
    eligible = np.asarray([False, True, True, True])
    selected = np.asarray([False, True, False, True])
    report = evaluate_frozen_policy(labels, current, strategy, eligible, selected)
    regret = paired_true_class_regret(labels, current, strategy)[eligible]
    used = regret[np.asarray([True, False, True])]
    assert report["nll_regret"]["coverage"] == pytest.approx(2.0 / 3.0)
    assert report["nll_regret"]["population"]["mean_regret"] == pytest.approx(
        used.sum() / 3.0
    )
    assert report["nll_regret"]["conditional_on_used"]["mean_regret"] == pytest.approx(
        used.mean()
    )
    assert report["hybrid_minus_current"]["macro_f1"] == pytest.approx(0.0)
    all_queries = report["classification_transitions"]["all_queries"]
    eligible_queries = report["classification_transitions"]["history_eligible_queries"]
    selected_queries = report["classification_transitions"]["history_selected_queries"]
    assert all_queries["history_breaks_correct_current"] == 0.25
    assert all_queries["history_rescues_wrong_current"] == 0.25
    assert eligible_queries["history_breaks_correct_current"] == pytest.approx(1.0 / 3.0)
    assert eligible_queries["history_rescues_wrong_current"] == pytest.approx(1.0 / 3.0)
    assert selected_queries["history_breaks_correct_current"] == 0.5
    assert selected_queries["history_rescues_wrong_current"] == 0.5


def test_frozen_thresholds_apply_threshold_without_test_top_k_reselection() -> None:
    labels, current, strategy = toy_probabilities()
    eligible = np.asarray([False, True, True, True])
    score = np.asarray([-100.0, -0.2, 0.1, 0.5])
    report = evaluate_frozen_thresholds(
        labels, current, strategy, eligible, score, thresholds=[0.0, 0.2]
    )
    assert report["official_test_top_k_reselection_permitted"] is False
    first = report["thresholds"]["0"]["evaluation"]
    second = report["thresholds"]["0.20000000000000001"]["evaluation"]
    assert first["nll_regret"]["used_queries"] == 1
    assert second["nll_regret"]["used_queries"] == 2


@pytest.mark.parametrize(
    "probability",
    [
        [[0.5, 0.6], [0.3, 0.7]],
        [[-0.1, 1.1], [0.3, 0.7]],
        [[math.nan, math.nan], [0.3, 0.7]],
    ],
)
def test_invalid_probability_fails_closed(probability: object) -> None:
    with pytest.raises(HarmBenchMetricError):
        validated_probability(probability, name="candidate")


def test_selection_without_history_fails_closed() -> None:
    labels, current, strategy = toy_probabilities()
    with pytest.raises(HarmBenchMetricError, match="without strictly past history"):
        evaluate_frozen_policy(
            labels,
            current,
            strategy,
            np.asarray([False, True, True, True]),
            np.asarray([True, False, False, False]),
        )


def test_zero_coverage_emits_json_safe_null_conditionals() -> None:
    labels, current, strategy = toy_probabilities()
    report = evaluate_frozen_policy(
        labels,
        current,
        strategy,
        np.asarray([False, True, True, True]),
        np.zeros(4, dtype=bool),
    )
    conditional = report["nll_regret"]["conditional_on_used"]
    assert conditional["mean_regret"] is None
    assert conditional["cvar90_regret"] is None
    ensure_finite_public_tree(report)


def test_public_tree_rejects_nan() -> None:
    with pytest.raises(HarmBenchMetricError, match="non-finite"):
        ensure_finite_public_tree({"metric": float("nan")})


def test_frozen_tail_and_harm_threshold_contract_cannot_change() -> None:
    labels, current, strategy = toy_probabilities()
    eligible = np.asarray([False, True, True, True])
    with pytest.raises(HarmBenchMetricError, match="tail alpha is frozen"):
        evaluate_frozen_policy(
            labels,
            current,
            strategy,
            eligible,
            eligible,
            tail_alpha=0.80,
        )
    with pytest.raises(HarmBenchMetricError, match="harm thresholds are frozen"):
        evaluate_frozen_policy(
            labels,
            current,
            strategy,
            eligible,
            eligible,
            harm_thresholds=(0.0,),
        )


def test_integral_float_labels_and_public_private_content_are_rejected() -> None:
    _, current, _ = toy_probabilities()
    with pytest.raises(HarmBenchMetricError, match="integer dtype"):
        paired_true_class_regret(np.asarray([0.0, 1.0, 0.0, 1.0]), current, current)
    with pytest.raises(HarmBenchMetricError, match="local path"):
        ensure_finite_public_tree({"note": "C:\\private\\labels.npy"})
    with pytest.raises(HarmBenchMetricError, match="forbidden key"):
        ensure_finite_public_tree({"row_ids": [1, 2]})
    with pytest.raises(HarmBenchMetricError, match="unexpectedly long"):
        ensure_finite_public_tree({"curve": list(range(300))})
