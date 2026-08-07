from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.negative_transfer_benchmark import (  # noqa: E402
    BenchmarkColumns,
    cluster_bootstrap_excess,
    evaluate_benchmark,
    policy_at_coverage,
    strict_upper_policy,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "history_count": [0, 1, 2, 3, 5, 10, 1],
            "excess": [9.0, -2.0, -1.0, 0.5, 1.0, 4.0, 2.0],
            "group": ["a", "a", "a", "a", "a", "b", "b"],
            "dialogue": [0, 0, 0, 1, 1, 2, 2],
            "risk_mean": [0.0, -1.5, -0.5, 0.3, 0.8, 3.0, 1.4],
            "risk_harm": [0.5, 0.1, 0.2, 0.6, 0.7, 0.95, 0.8],
            "risk_upper": [1.0, -0.2, 0.1, 0.7, 1.2, 4.0, 2.5],
        }
    )


def test_policy_at_coverage_selects_lowest_risk() -> None:
    excess = np.asarray([-2.0, -1.0, 0.5, 1.0])
    score = np.asarray([-3.0, -2.0, 0.0, 1.0])
    result = policy_at_coverage(excess, score, 0.50)
    assert result["coverage"] == 0.50
    assert result["used_queries"] == 2
    assert result["mean_policy_regret"] == -0.75
    assert result["harm_rate_among_used"] == 0.0


def test_strict_upper_policy_uses_only_negative_bounds() -> None:
    result = strict_upper_policy(np.asarray([-2.0, 1.0, -1.0]), np.asarray([-0.1, 0.0, 0.2]))
    assert result["used_queries"] == 1
    assert result["mean_policy_regret"] == -2.0 / 3.0


def test_cluster_bootstrap_is_seed_deterministic() -> None:
    excess = np.asarray([-2.0, -1.0, 1.0, 2.0])
    clusters = np.asarray(["a", "a", "b", "b"])
    first = cluster_bootstrap_excess(excess, clusters, replicates=100, seed=7)
    second = cluster_bootstrap_excess(excess, clusters, replicates=100, seed=7)
    assert first == second


def test_evaluate_benchmark_emits_only_aggregate_contract() -> None:
    result = evaluate_benchmark(
        sample_frame(),
        BenchmarkColumns("history_count", "excess", "dialogue"),
        {
            "demo": {
                "mean": "risk_mean",
                "harm_probability": "risk_harm",
                "upper": "risk_upper",
            }
        },
        [0.50],
        bootstrap_replicates=100,
        bootstrap_seed=11,
    )
    assert result["data"]["eligible_history_rows"] == 6
    assert result["reference_policies"]["current_only"]["mean_policy_regret"] == 0.0
    assert result["reference_policies"]["oracle_use_history_if_beneficial"]["used_queries"] == 2
    assert result["privacy_contract"]["row_identifiers_emitted"] is False
    assert "dialogue" not in str(result)


def test_missing_required_column_fails_closed() -> None:
    frame = sample_frame().drop(columns=["excess"])
    try:
        evaluate_benchmark(
            frame,
            BenchmarkColumns("history_count", "excess", "dialogue"),
            {"demo": {"mean": "risk_mean"}},
            [0.50],
            bootstrap_replicates=100,
            bootstrap_seed=11,
        )
    except ValueError as error:
        assert "missing required column" in str(error)
    else:
        raise AssertionError("missing column must fail closed")


def test_composite_cluster_does_not_merge_reused_dialogue_ids() -> None:
    frame = sample_frame().copy()
    frame.loc[5:, "dialogue"] = 0
    result = evaluate_benchmark(
        frame,
        BenchmarkColumns("history_count", "excess", ("group", "dialogue")),
        {"demo": {"mean": "risk_mean"}},
        [0.50],
        bootstrap_replicates=100,
        bootstrap_seed=11,
    )
    assert result["data"]["clusters"] == 3
