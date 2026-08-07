from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class BenchmarkColumns:
    history_count: str
    excess_loss: str
    cluster: str | tuple[str, ...]


def _as_finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"missing required column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"column contains missing or non-finite values: {column}")
    return values


def _cluster_values(frame: pd.DataFrame, columns: str | tuple[str, ...]) -> np.ndarray:
    names = (columns,) if isinstance(columns, str) else tuple(columns)
    if not names:
        raise ValueError("at least one cluster column is required")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required cluster columns: {missing}")
    if frame.loc[:, names].isna().any().any():
        raise ValueError("cluster columns contain missing values")
    if len(names) == 1:
        return frame[names[0]].astype(str).to_numpy()
    return frame.loc[:, names].astype(str).agg("/".join, axis=1).to_numpy()


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else float("nan")


def _expected_shortfall(values: np.ndarray, probability: float = 0.90) -> float:
    if len(values) == 0:
        return float("nan")
    threshold = np.quantile(values, probability)
    tail = values[values >= threshold]
    return float(np.mean(tail))


def _cluster_tail_concentration(excess: np.ndarray, clusters: np.ndarray) -> float:
    positive = np.maximum(excess, 0.0)
    total = float(positive.sum())
    if total <= 0:
        return 0.0
    cluster_totals = pd.Series(positive).groupby(pd.Series(clusters), sort=False).sum().to_numpy()
    keep = max(1, int(np.ceil(0.10 * len(cluster_totals))))
    return float(np.sort(cluster_totals)[-keep:].sum() / total)


def describe_excess(excess: np.ndarray, clusters: np.ndarray) -> dict:
    excess = np.asarray(excess, dtype=float)
    clusters = np.asarray(clusters)
    if len(excess) == 0:
        raise ValueError("no eligible history queries")
    if len(excess) != len(clusters):
        raise ValueError("excess and cluster lengths differ")
    cluster_means = pd.Series(excess).groupby(pd.Series(clusters), sort=False).mean().to_numpy()
    return {
        "queries": int(len(excess)),
        "clusters": int(len(np.unique(clusters))),
        "mean_excess_loss": float(np.mean(excess)),
        "median_excess_loss": float(np.median(excess)),
        "harm_rate": float(np.mean(excess > 0)),
        "benefit_rate": float(np.mean(excess < 0)),
        "zero_rate": float(np.mean(excess == 0)),
        "p90_excess_loss": _quantile(excess, 0.90),
        "p95_excess_loss": _quantile(excess, 0.95),
        "p99_excess_loss": _quantile(excess, 0.99),
        "cvar90_excess_loss": _expected_shortfall(excess, 0.90),
        "cluster_mean_p90": _quantile(cluster_means, 0.90),
        "cluster_mean_max": float(np.max(cluster_means)),
        "positive_regret_share_top_10pct_clusters": _cluster_tail_concentration(excess, clusters),
    }


def _policy_metrics(excess: np.ndarray, selected: np.ndarray) -> dict:
    selected = np.asarray(selected, dtype=bool)
    if len(excess) != len(selected):
        raise ValueError("excess and selected lengths differ")
    policy_regret = np.where(selected, excess, 0.0)
    used = excess[selected]
    return {
        "coverage": float(np.mean(selected)),
        "used_queries": int(selected.sum()),
        "mean_policy_regret": float(np.mean(policy_regret)),
        "p90_policy_regret": _quantile(policy_regret, 0.90),
        "cvar90_policy_regret": _expected_shortfall(policy_regret, 0.90),
        "harm_rate_among_used": float(np.mean(used > 0)) if len(used) else None,
        "mean_excess_among_used": float(np.mean(used)) if len(used) else None,
        "p90_excess_among_used": _quantile(used, 0.90) if len(used) else None,
    }


def policy_at_coverage(excess: np.ndarray, score: np.ndarray, coverage: float) -> dict:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if len(excess) != len(score):
        raise ValueError("excess and score lengths differ")
    keep = max(1, int(np.floor(coverage * len(excess))))
    order = np.lexsort((np.arange(len(score)), np.asarray(score, dtype=float)))
    selected = np.zeros(len(excess), dtype=bool)
    selected[order[:keep]] = True
    return _policy_metrics(np.asarray(excess, dtype=float), selected)


def strict_upper_policy(excess: np.ndarray, upper: np.ndarray) -> dict:
    return _policy_metrics(np.asarray(excess, dtype=float), np.asarray(upper, dtype=float) < 0)


def _ranking_metrics(excess: np.ndarray, score: np.ndarray, *, probability: bool) -> dict:
    harm = excess > 0
    rho = float(spearmanr(excess, score).statistic)
    auc = float(roc_auc_score(harm, score)) if len(np.unique(harm)) == 2 else None
    output = {"harm_auc": auc, "spearman_excess": rho}
    if probability:
        clipped = np.clip(score, 0.0, 1.0)
        output["brier_harm"] = float(brier_score_loss(harm.astype(int), clipped))
    return output


def _history_strata(counts: np.ndarray, excess: np.ndarray) -> dict:
    definitions = (
        ("1", counts == 1),
        ("2", counts == 2),
        ("3-4", (counts >= 3) & (counts <= 4)),
        ("5-9", (counts >= 5) & (counts <= 9)),
        ("10+", counts >= 10),
    )
    result: dict[str, dict] = {}
    for name, mask in definitions:
        if not np.any(mask):
            result[name] = {"queries": 0}
            continue
        values = excess[mask]
        result[name] = {
            "queries": int(mask.sum()),
            "mean_excess_loss": float(np.mean(values)),
            "harm_rate": float(np.mean(values > 0)),
            "p90_excess_loss": _quantile(values, 0.90),
        }
    return result


def _cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique = pd.unique(pd.Series(clusters))
    indices = [np.flatnonzero(clusters == cluster) for cluster in unique]
    return np.asarray(unique), indices


def _bootstrap_indices(indices: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    sampled = rng.integers(0, len(indices), size=len(indices))
    return np.concatenate([indices[index] for index in sampled])


def _ci(samples: Sequence[float]) -> dict:
    values = np.asarray(samples, dtype=float)
    return {
        "estimate_mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def cluster_bootstrap_excess(
    excess: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    _, indices = _cluster_index(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    means: list[float] = []
    harms: list[float] = []
    p90s: list[float] = []
    for _ in range(replicates):
        sampled = _bootstrap_indices(indices, rng)
        values = excess[sampled]
        means.append(float(np.mean(values)))
        harms.append(float(np.mean(values > 0)))
        p90s.append(_quantile(values, 0.90))
    return {
        "unit": "cluster",
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_excess_loss": _ci(means),
        "harm_rate": _ci(harms),
        "p90_excess_loss": _ci(p90s),
    }


def cluster_bootstrap_policy(
    excess: np.ndarray,
    score: np.ndarray,
    clusters: np.ndarray,
    coverage: float,
    *,
    replicates: int,
    seed: int,
) -> dict:
    _, indices = _cluster_index(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    regret: list[float] = []
    harm: list[float] = []
    actual_coverage: list[float] = []
    for _ in range(replicates):
        sampled = _bootstrap_indices(indices, rng)
        metrics = policy_at_coverage(excess[sampled], score[sampled], coverage)
        regret.append(metrics["mean_policy_regret"])
        harm.append(metrics["harm_rate_among_used"])
        actual_coverage.append(metrics["coverage"])
    return {
        "mean_policy_regret": _ci(regret),
        "harm_rate_among_used": _ci(harm),
        "actual_coverage": _ci(actual_coverage),
    }


def evaluate_benchmark(
    frame: pd.DataFrame,
    columns: BenchmarkColumns,
    selectors: Mapping[str, Mapping[str, str]],
    coverages: Sequence[float],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    counts_all = _as_finite_numeric(frame, columns.history_count).astype(int)
    excess_all = _as_finite_numeric(frame, columns.excess_loss)
    eligible = counts_all > 0
    if not np.any(eligible):
        raise ValueError("no rows have strictly past history")
    counts = counts_all[eligible]
    excess = excess_all[eligible]
    clusters = _cluster_values(frame.loc[eligible], columns.cluster)

    always = _policy_metrics(excess, np.ones(len(excess), dtype=bool))
    current = _policy_metrics(excess, np.zeros(len(excess), dtype=bool))
    oracle = _policy_metrics(excess, excess < 0)
    selector_results: dict[str, dict] = {}

    for selector_index, (name, spec) in enumerate(selectors.items()):
        observed: dict[str, np.ndarray] = {}
        for kind in ("mean", "harm_probability", "upper"):
            if kind in spec:
                observed[kind] = _as_finite_numeric(frame, spec[kind])[eligible]
        if not observed:
            raise ValueError(f"selector has no supported score columns: {name}")
        selector_output: dict[str, object] = {"scores": {}}
        for score_index, (kind, score) in enumerate(observed.items()):
            score_output: dict[str, object] = {
                "ranking": _ranking_metrics(excess, score, probability=kind == "harm_probability"),
                "coverage_policies": {},
            }
            for coverage_index, coverage in enumerate(coverages):
                key = f"{float(coverage):.2f}"
                score_output["coverage_policies"][key] = {
                    "point": policy_at_coverage(excess, score, float(coverage)),
                    "cluster_bootstrap": cluster_bootstrap_policy(
                        excess,
                        score,
                        clusters,
                        float(coverage),
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 10000 * selector_index + 1000 * score_index + coverage_index,
                    ),
                }
            selector_output["scores"][kind] = score_output
        if "upper" in observed:
            selector_output["strict_upper_below_zero"] = strict_upper_policy(excess, observed["upper"])
        selector_results[name] = selector_output

    return {
        "data": {
            "total_rows": int(len(frame)),
            "eligible_history_rows": int(eligible.sum()),
            "clusters": int(len(np.unique(clusters))),
        },
        "natural_negative_transfer": {
            "point": describe_excess(excess, clusters),
            "cluster_bootstrap": cluster_bootstrap_excess(
                excess,
                clusters,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
            "history_depth": _history_strata(counts, excess),
        },
        "reference_policies": {
            "current_only": current,
            "always_history": always,
            "oracle_use_history_if_beneficial": oracle,
        },
        "selectors": selector_results,
        "privacy_contract": {
            "aggregate_only": True,
            "row_identifiers_emitted": False,
            "cluster_identifiers_emitted": False,
        },
    }
