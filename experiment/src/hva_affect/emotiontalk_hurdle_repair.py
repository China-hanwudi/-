from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data_contract import ContractError, write_json_atomic
from .emotiontalk_endpoint_diagnostic import objective_reversal_summary
from .emotiontalk_multimodal_external import _risk_classifier, _risk_regressor
from .meld_text_pilot import sha256_file
from .negative_transfer_benchmark import BenchmarkColumns, evaluate_benchmark


EXPECTED_CACHE_FIELDS = {
    "schema_version",
    "fit_x",
    "fit_target",
    "fit_seed_targets",
    "selection_x",
    "selection_target",
    "selection_seed_targets",
    "selection_counts",
    "selection_cluster_codes",
    "feature_names",
    "base_config_sha256",
    "diagnostic_config_sha256",
}


def compose_expected_regret(
    harm_probability: np.ndarray,
    harm_severity: np.ndarray,
    benefit_magnitude: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(harm_probability, dtype=float)
    harm = np.asarray(harm_severity, dtype=float)
    benefit = np.asarray(benefit_magnitude, dtype=float)
    if not (probability.shape == harm.shape == benefit.shape):
        raise ValueError("hurdle component predictions are not aligned")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("harm probability outside [0, 1]")
    if np.any(harm < 0) or np.any(benefit < 0):
        raise ValueError("severity magnitudes must be nonnegative")
    return probability * harm - (1.0 - probability) * benefit


def _read_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_CACHE_FIELDS:
            raise ContractError(f"endpoint cache schema changed: {archive.files}")
        data = {name: archive[name].copy() for name in archive.files}
    if data["schema_version"].tolist() != ["emotiontalk_endpoint_cache_v1"]:
        raise ContractError("unexpected endpoint cache version")
    for name in ("fit_x", "selection_x", "fit_target", "selection_target"):
        if not np.isfinite(data[name]).all():
            raise ContractError(f"non-finite values in cache field {name}")
    if len(data["fit_x"]) != len(data["fit_target"]):
        raise ContractError("fit cache rows are misaligned")
    if len(data["selection_x"]) != len(data["selection_target"]):
        raise ContractError("selection cache rows are misaligned")
    if len(data["selection_counts"]) != len(data["selection_target"]):
        raise ContractError("selection counts are misaligned")
    if np.any(data["selection_counts"] <= 0):
        raise ContractError("private cache contains ineligible no-history rows")
    return data


def _validate_hashes(cache: Mapping[str, np.ndarray], base_config_path: Path, diagnostic_config_path: Path) -> None:
    expected_base = str(cache["base_config_sha256"][0])
    expected_diagnostic = str(cache["diagnostic_config_sha256"][0])
    if sha256_file(base_config_path) != expected_base:
        raise ContractError("base config hash differs from endpoint cache")
    if sha256_file(diagnostic_config_path) != expected_diagnostic:
        raise ContractError("diagnostic config hash differs from endpoint cache")


def _safe_spearman(truth: np.ndarray, score: np.ndarray) -> float | None:
    if len(truth) < 3 or np.all(score == score[0]) or np.all(truth == truth[0]):
        return None
    return float(spearmanr(truth, score).statistic)


def _fit_hurdle_models(
    base_config: dict,
    repair_config: dict,
    fit_x: np.ndarray,
    fit_target: np.ndarray,
    selection_x: np.ndarray,
) -> dict[str, np.ndarray | float | list[float]]:
    positive = fit_target > 0
    negative = fit_target < 0
    if positive.sum() < 100 or negative.sum() < 100:
        raise ContractError("insufficient positive or negative fit targets for hurdle repair")
    clip_probability = float(repair_config["severity_prediction_clip_quantile"])
    harm_cap = float(np.quantile(fit_target[positive], clip_probability))
    benefit_cap = float(np.quantile(-fit_target[negative], clip_probability))
    if harm_cap <= 0 or benefit_cap <= 0:
        raise ContractError("invalid hurdle severity cap")

    per_seed_expected, per_seed_harm_probability = [], []
    per_seed_harm_severity, per_seed_benefit_magnitude, per_seed_direct = [], [], []
    for seed in repair_config["risk_seeds"]:
        harm_classifier = _risk_classifier(base_config, int(seed))
        harm_regressor = _risk_regressor(base_config, int(seed), loss="squared_error")
        benefit_regressor = _risk_regressor(base_config, int(seed) + 10000, loss="squared_error")
        direct_regressor = _risk_regressor(base_config, int(seed) + 20000, loss="squared_error")
        harm_classifier.fit(fit_x, positive.astype(int))
        harm_regressor.fit(fit_x[positive], np.log1p(fit_target[positive]))
        benefit_regressor.fit(fit_x[negative], np.log1p(-fit_target[negative]))
        direct_regressor.fit(fit_x, fit_target)
        probability = harm_classifier.predict_proba(selection_x)[:, 1]
        harm_severity = np.clip(np.expm1(harm_regressor.predict(selection_x)), 0.0, harm_cap)
        benefit_magnitude = np.clip(np.expm1(benefit_regressor.predict(selection_x)), 0.0, benefit_cap)
        expected = compose_expected_regret(probability, harm_severity, benefit_magnitude)
        per_seed_harm_probability.append(probability)
        per_seed_harm_severity.append(harm_severity)
        per_seed_benefit_magnitude.append(benefit_magnitude)
        per_seed_expected.append(expected)
        per_seed_direct.append(direct_regressor.predict(selection_x))
        print(f"hurdle repair risk seed {seed} complete", flush=True)

    expected_array = np.asarray(per_seed_expected)
    pairwise = [
        float(spearmanr(expected_array[left], expected_array[right]).statistic)
        for left, right in combinations(range(len(expected_array)), 2)
    ]
    return {
        "harm_probability": np.mean(per_seed_harm_probability, axis=0),
        "harm_severity": np.mean(per_seed_harm_severity, axis=0),
        "benefit_magnitude": np.mean(per_seed_benefit_magnitude, axis=0),
        "expected_regret": np.mean(per_seed_expected, axis=0),
        "direct_mean": np.mean(per_seed_direct, axis=0),
        "expected_regret_seed_spearman_median": float(np.median(pairwise)),
        "expected_regret_seed_spearman_min": float(np.min(pairwise)),
        "harm_severity_cap": harm_cap,
        "benefit_magnitude_cap": benefit_cap,
    }


def run_hurdle_repair(
    cache_path: Path,
    base_config_path: Path,
    diagnostic_config_path: Path,
    repair_config_path: Path,
    output_path: Path,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"hurdle repair output already exists: {output_path}")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    repair_config = json.loads(repair_config_path.read_text(encoding="utf-8"))
    if repair_config.get("input_schema") != "emotiontalk_endpoint_cache_v1":
        raise ContractError("repair input schema changed")
    if repair_config.get("status") != "repair_1_of_3_frozen_before_repair_result":
        raise ContractError("repair status is not frozen before repair evaluation")
    cache = _read_cache(cache_path)
    _validate_hashes(cache, base_config_path, diagnostic_config_path)
    prediction = _fit_hurdle_models(
        base_config,
        repair_config,
        cache["fit_x"],
        cache["fit_target"],
        cache["selection_x"],
    )
    target = cache["selection_target"].astype(float)
    frame = pd.DataFrame({
        "history_count": cache["selection_counts"].astype(int),
        "excess_loss": target,
        "cluster": cache["selection_cluster_codes"].astype(int),
        "direct_mean": prediction["direct_mean"],
        "expected_regret": prediction["expected_regret"],
        "harm_probability": prediction["harm_probability"],
    })
    benchmark = evaluate_benchmark(
        frame,
        BenchmarkColumns("history_count", "excess_loss", "cluster"),
        {
            "direct_mean_baseline": {"mean": "direct_mean"},
            "two_part_hurdle": {
                "mean": "expected_regret",
                "harm_probability": "harm_probability",
            },
        },
        repair_config["coverages"],
        bootstrap_replicates=int(repair_config["bootstrap_replicates"]),
        bootstrap_seed=int(repair_config["bootstrap_seed"]),
    )
    positive = target > 0
    negative = target < 0
    component_metrics = {
        "harm_probability_auc": float(roc_auc_score(positive.astype(int), prediction["harm_probability"])),
        "harm_severity_spearman_on_harmful_rows": _safe_spearman(target[positive], prediction["harm_severity"][positive]),
        "benefit_magnitude_spearman_on_beneficial_rows": _safe_spearman(-target[negative], prediction["benefit_magnitude"][negative]),
        "expected_regret_spearman": _safe_spearman(target, prediction["expected_regret"]),
        "direct_mean_spearman": _safe_spearman(target, prediction["direct_mean"]),
        "expected_regret_seed_spearman_median": prediction["expected_regret_seed_spearman_median"],
        "expected_regret_seed_spearman_min": prediction["expected_regret_seed_spearman_min"],
        "harm_severity_cap": prediction["harm_severity_cap"],
        "benefit_magnitude_cap": prediction["benefit_magnitude_cap"],
    }
    gates = repair_config["gates"]
    hurdle_scores = benchmark["selectors"]["two_part_hurdle"]["scores"]["mean"]["coverage_policies"]
    safe_coverages = [
        coverage
        for coverage, row in hurdle_scores.items()
        if row["cluster_bootstrap"]["mean_policy_regret"]["ci95_high"]
        <= float(gates["maximum_safe_mean_regret_ci95_high"])
    ]
    gate_checks = {
        "expected_regret_signal": bool(
            component_metrics["expected_regret_spearman"] is not None
            and component_metrics["expected_regret_spearman"] >= float(gates["minimum_expected_regret_spearman"])
        ),
        "gain_vs_direct_mean": bool(
            component_metrics["expected_regret_spearman"] is not None
            and component_metrics["direct_mean_spearman"] is not None
            and component_metrics["expected_regret_spearman"] - component_metrics["direct_mean_spearman"]
            >= float(gates["minimum_spearman_gain_vs_direct_mean"])
        ),
        "harm_signal": bool(component_metrics["harm_probability_auc"] >= float(gates["minimum_harm_probability_auc"])),
        "safe_mean_regret_coverage": bool(len(safe_coverages) >= int(gates["minimum_safe_mean_regret_coverages"])),
    }
    reversal = objective_reversal_summary(
        target,
        np.asarray(prediction["expected_regret"]),
        np.asarray(prediction["harm_probability"]),
        repair_config["coverages"],
        {
            "maximum_selected_set_jaccard_for_reversal": 0.80,
            "minimum_mean_regret_advantage_for_reversal": 0.01,
            "minimum_harm_rate_advantage_for_reversal": 0.03,
            "minimum_reversal_coverages": 1,
        },
    )
    result = {
        "protocol": repair_config["protocol"],
        "status": "repair_1_model_selection_exploratory; calibration_and_internal_holdout_unread",
        "claim_boundary": repair_config["claim_boundary"],
        "hashes": {
            "private_cache_sha256": sha256_file(cache_path),
            "base_config_sha256": sha256_file(base_config_path),
            "diagnostic_config_sha256": sha256_file(diagnostic_config_path),
            "repair_config_sha256": sha256_file(repair_config_path),
        },
        "data": {
            "fit_history_rows": int(len(cache["fit_target"])),
            "model_selection_history_rows": int(len(target)),
            "model_selection_clusters": int(len(np.unique(cache["selection_cluster_codes"]))),
        },
        "component_metrics": component_metrics,
        "benchmark": benchmark,
        "objective_reversal": reversal,
        "safe_mean_regret_coverages": safe_coverages,
        "gate_checks": gate_checks,
        "proceed_to_stochastic_subset_augmentation": bool(all(gate_checks.values())),
        "privacy_contract": {
            "aggregate_only": True,
            "row_identifiers_emitted": False,
            "cluster_identifiers_emitted": False,
            "private_cache_not_published": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, output_path.resolve())
    return result
