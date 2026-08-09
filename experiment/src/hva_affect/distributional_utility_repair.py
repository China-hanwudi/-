"""Distributional sign-by-severity repair for bidirectional history utility.

The only data input accepted by the file runner is the private, open-role
``emotiontalk_bidirectional_oof_cache_v1`` archive.  Thresholds are learned
from group-held-out predictions on the fit role and transferred unchanged to
the disjoint model-selection role.  The public artifact contains aggregate
statistics only.

Utility is benefit-positive.  Each directional head estimates

``P(u > 0) * E[u | u > 0] - (1 - P(u > 0)) * E[-u | u < 0]``.

The true bidirectional decision is the minimum of independently supervised
forward and backward heads.  The parameter/capacity-matched pseudo control
uses two stochastic heads with the same forward target.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


CACHE_SCHEMA_VERSION = "emotiontalk_bidirectional_oof_cache_v1"
REPORT_SCHEMA_VERSION = "emotiontalk_distributional_utility_repair_report_v1"
PROTOCOL = "emotiontalk_distributional_utility_repair_v1"
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_808
DEFAULT_OOF_FOLDS = 5
PRIMARY_OPERATING_POINT = "primary_25pct"
OPERATING_POINTS = (
    (PRIMARY_OPERATING_POINT, 0.25),
    ("safety_exploration_10pct", 0.10),
)
ZERO_OPERATING_POINT = "zero_threshold_diagnostic"

MODEL_MODES = (
    ("distributional_forward_only", "forward_only"),
    ("distributional_backward_only", "backward_only"),
    ("distributional_pseudo_bidirectional", "pseudo_bidirectional"),
    ("distributional_true_bidirectional", "true_bidirectional"),
)
TRUE_MODEL_NAME = "distributional_true_bidirectional"

_CORE_CACHE_KEYS = {
    "schema_version",
    "fit_x",
    "fit_forward",
    "fit_backward",
    "fit_cluster_codes",
    "selection_x",
    "selection_forward",
    "selection_backward",
    "selection_cluster_codes",
    "base_config_sha256",
    "utility_config_sha256",
}
_OPTIONAL_CACHE_KEYS = {
    "fit_forward_seed",
    "fit_backward_seed",
    "selection_forward_seed",
    "selection_backward_seed",
    "feature_names",
}
_RESTRICTED_TOKENS = {"calibration", "holdout", "validation", "test", "sealed"}


class DistributionalRepairContractError(ValueError):
    """Raised when data lineage, modelling, or aggregate output breaks contract."""


@dataclass(frozen=True)
class UtilitySplit:
    x: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    cluster_codes: np.ndarray

    @classmethod
    def validated(
        cls,
        x: np.ndarray,
        forward: np.ndarray,
        backward: np.ndarray,
        cluster_codes: np.ndarray,
        *,
        label: str,
        require_multiple_clusters: bool = True,
    ) -> "UtilitySplit":
        features = np.asarray(x, dtype=np.float64)
        forward_array = np.asarray(forward, dtype=np.float64)
        backward_array = np.asarray(backward, dtype=np.float64)
        clusters = np.asarray(cluster_codes)
        if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
            raise DistributionalRepairContractError(
                f"{label} features must have shape (rows>=2, columns>=1)"
            )
        rows = len(features)
        if forward_array.shape != (rows,) or backward_array.shape != (rows,):
            raise DistributionalRepairContractError(f"{label} utility targets are misaligned")
        if clusters.shape != (rows,) or not np.issubdtype(clusters.dtype, np.integer):
            raise DistributionalRepairContractError(
                f"{label} cluster codes must be a row-aligned integer vector"
            )
        if not np.isfinite(features).all():
            raise DistributionalRepairContractError(f"{label} features are non-finite")
        if not np.isfinite(forward_array).all() or not np.isfinite(backward_array).all():
            raise DistributionalRepairContractError(f"{label} utilities are non-finite")
        if np.any(clusters < 0):
            raise DistributionalRepairContractError(f"{label} cluster codes must be nonnegative")
        if require_multiple_clusters and len(np.unique(clusters)) < 2:
            raise DistributionalRepairContractError(f"{label} requires multiple clusters")
        return cls(
            features,
            forward_array,
            backward_array,
            clusters.astype(np.int64, copy=False),
        )

    @property
    def strict_utility(self) -> np.ndarray:
        return np.minimum(self.forward, self.backward)


@dataclass(frozen=True)
class DistributionalCache:
    fit: UtilitySplit
    selection: UtilitySplit
    feature_names: tuple[str, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class ComponentSpec:
    learning_rate: float = 0.06
    max_iter: int = 60
    max_leaf_nodes: int = 15
    max_depth: int | None = None
    min_samples_leaf: int = 50
    l2_regularization: float = 0.1
    max_bins: int = 127
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 6
    tolerance: float = 1e-5
    minimum_conditional_rows: int = 100

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComponentSpec":
        if value.get("estimator") != "sklearn_hist_gradient_boosting":
            raise DistributionalRepairContractError("component estimator changed")
        spec = cls(
            learning_rate=float(value["learning_rate"]),
            max_iter=int(value["max_iter"]),
            max_leaf_nodes=int(value["max_leaf_nodes"]),
            max_depth=None if value.get("max_depth") is None else int(value["max_depth"]),
            min_samples_leaf=int(value["min_samples_leaf"]),
            l2_regularization=float(value["l2_regularization"]),
            max_bins=int(value["max_bins"]),
            early_stopping=bool(value["early_stopping"]),
            validation_fraction=float(value["validation_fraction"]),
            n_iter_no_change=int(value["n_iter_no_change"]),
            tolerance=float(value["tolerance"]),
            minimum_conditional_rows=int(value["minimum_conditional_rows"]),
        )
        if not 0 < spec.learning_rate <= 1 or spec.max_iter < 1:
            raise DistributionalRepairContractError("invalid boosting rate or iteration count")
        if spec.max_leaf_nodes < 2 or spec.min_samples_leaf < 1 or spec.max_bins < 2:
            raise DistributionalRepairContractError("invalid tree capacity")
        if spec.l2_regularization < 0 or spec.tolerance <= 0:
            raise DistributionalRepairContractError("invalid component regularization")
        if not 0 < spec.validation_fraction < 1 or spec.n_iter_no_change < 1:
            raise DistributionalRepairContractError("invalid early-stopping contract")
        if spec.minimum_conditional_rows < 2:
            raise DistributionalRepairContractError("conditional sample minimum is too small")
        return spec

    def public_dict(self) -> dict[str, Any]:
        return {
            "estimator": "sklearn_hist_gradient_boosting",
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "max_bins": self.max_bins,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "tolerance": self.tolerance,
            "minimum_conditional_rows": self.minimum_conditional_rows,
        }


@dataclass(frozen=True)
class HeadPrediction:
    positive_probability: np.ndarray
    positive_magnitude: np.ndarray
    negative_magnitude: np.ndarray
    expected_utility: np.ndarray


@dataclass
class FittedDistributionalHead:
    classifier: HistGradientBoostingClassifier
    positive_regressor: HistGradientBoostingRegressor
    negative_regressor: HistGradientBoostingRegressor
    positive_cap: float
    negative_cap: float

    def predict(self, x: np.ndarray) -> HeadPrediction:
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise DistributionalRepairContractError("head prediction features are invalid")
        probability = np.asarray(self.classifier.predict_proba(features)[:, 1], dtype=np.float64)
        positive = np.clip(
            np.asarray(self.positive_regressor.predict(features), dtype=np.float64),
            0.0,
            self.positive_cap,
        )
        negative = np.clip(
            np.asarray(self.negative_regressor.predict(features), dtype=np.float64),
            0.0,
            self.negative_cap,
        )
        expected = compose_expected_utility(probability, positive, negative)
        return HeadPrediction(probability, positive, negative, expected)

    @property
    def fitted_iterations(self) -> dict[str, int]:
        return {
            "sign_classifier": int(self.classifier.n_iter_),
            "positive_severity_regressor": int(self.positive_regressor.n_iter_),
            "negative_severity_regressor": int(self.negative_regressor.n_iter_),
        }


@dataclass(frozen=True)
class ModelPrediction:
    decision: np.ndarray
    heads: tuple[HeadPrediction, ...]
    head_targets: tuple[str, ...]


def compose_expected_utility(
    positive_probability: np.ndarray,
    positive_magnitude: np.ndarray,
    negative_magnitude: np.ndarray,
) -> np.ndarray:
    """Compose benefit-positive expectation from sign and conditional severities."""

    probability = np.asarray(positive_probability, dtype=np.float64)
    positive = np.asarray(positive_magnitude, dtype=np.float64)
    negative = np.asarray(negative_magnitude, dtype=np.float64)
    if probability.shape != positive.shape or probability.shape != negative.shape:
        raise ValueError("distributional components must be aligned")
    if not all(np.isfinite(value).all() for value in (probability, positive, negative)):
        raise ValueError("distributional components must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("positive probability must lie in [0, 1]")
    if np.any(positive < 0.0) or np.any(negative < 0.0):
        raise ValueError("severity magnitudes must be nonnegative")
    return probability * positive - (1.0 - probability) * negative


def compose_registered_models(
    forward_first: HeadPrediction,
    forward_second: HeadPrediction,
    backward_second: HeadPrediction,
) -> dict[str, ModelPrediction]:
    """Create matched ablations from three fitted head slots.

    The first forward and second backward slots are reused by the single-head
    baselines and the true model.  The pseudo control replaces only the second
    target with forward utility, preserving two heads and six components.
    """

    lengths = {
        len(forward_first.expected_utility),
        len(forward_second.expected_utility),
        len(backward_second.expected_utility),
    }
    if len(lengths) != 1:
        raise DistributionalRepairContractError("registered head predictions are misaligned")
    return {
        "distributional_forward_only": ModelPrediction(
            forward_first.expected_utility.copy(), (forward_first,), ("forward",)
        ),
        "distributional_backward_only": ModelPrediction(
            backward_second.expected_utility.copy(), (backward_second,), ("backward",)
        ),
        "distributional_pseudo_bidirectional": ModelPrediction(
            np.minimum(forward_first.expected_utility, forward_second.expected_utility),
            (forward_first, forward_second),
            ("forward", "forward"),
        ),
        "distributional_true_bidirectional": ModelPrediction(
            np.minimum(forward_first.expected_utility, backward_second.expected_utility),
            (forward_first, backward_second),
            ("forward", "backward"),
        ),
    }


def _restricted_field(name: str) -> bool:
    tokens = str(name).lower().replace("-", "_").split("_")
    return any(token in _RESTRICTED_TOKENS for token in tokens)


def _single_string(value: np.ndarray, *, key: str) -> str:
    array = np.asarray(value)
    if array.shape != (1,) or array.dtype.kind not in {"U", "S"}:
        raise DistributionalRepairContractError(f"{key} must be a one-string array")
    return str(array[0])


def _validated_sha256(value: str, *, key: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DistributionalRepairContractError(f"{key} must be a SHA-256 digest")
    return digest


def cache_from_mapping(values: Mapping[str, np.ndarray]) -> DistributionalCache:
    keys = set(values)
    restricted = sorted(key for key in keys if _restricted_field(key))
    if restricted:
        raise DistributionalRepairContractError(f"restricted-role fields are forbidden: {restricted}")
    missing = sorted(_CORE_CACHE_KEYS - keys)
    unknown = sorted(keys - _CORE_CACHE_KEYS - _OPTIONAL_CACHE_KEYS)
    if missing or unknown:
        raise DistributionalRepairContractError(
            f"private cache schema mismatch: missing={missing}, unknown={unknown}"
        )
    if _single_string(values["schema_version"], key="schema_version") != CACHE_SCHEMA_VERSION:
        raise DistributionalRepairContractError("private cache schema version changed")
    fit = UtilitySplit.validated(
        values["fit_x"],
        values["fit_forward"],
        values["fit_backward"],
        values["fit_cluster_codes"],
        label="fit",
    )
    selection = UtilitySplit.validated(
        values["selection_x"],
        values["selection_forward"],
        values["selection_backward"],
        values["selection_cluster_codes"],
        label="model_selection",
    )
    if fit.x.shape[1] != selection.x.shape[1]:
        raise DistributionalRepairContractError("fit and model-selection feature counts differ")
    if "feature_names" in values:
        feature_names = tuple(str(value) for value in np.asarray(values["feature_names"]).reshape(-1))
        if len(feature_names) != fit.x.shape[1] or len(set(feature_names)) != len(feature_names):
            raise DistributionalRepairContractError("feature names are not unique and aligned")
    else:
        feature_names = tuple(f"feature_{index}" for index in range(fit.x.shape[1]))
    for prefix, rows in (("fit", len(fit.x)), ("selection", len(selection.x))):
        for direction in ("forward", "backward"):
            key = f"{prefix}_{direction}_seed"
            if key not in values:
                continue
            array = np.asarray(values[key], dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != rows or not np.isfinite(array).all():
                raise DistributionalRepairContractError(f"{key} is malformed")
    hashes = {
        key: _validated_sha256(_single_string(values[key], key=key), key=key)
        for key in ("base_config_sha256", "utility_config_sha256")
    }
    return DistributionalCache(fit, selection, feature_names, hashes)


def load_private_cache(path: str | Path) -> DistributionalCache:
    cache_path = Path(path)
    if cache_path.suffix.lower() != ".npz":
        raise DistributionalRepairContractError("private cache must be an .npz archive")
    with np.load(cache_path, allow_pickle=False) as archive:
        restricted = sorted(key for key in archive.files if _restricted_field(key))
        if restricted:
            raise DistributionalRepairContractError(
                f"restricted-role fields are forbidden: {restricted}"
            )
        values = {key: np.asarray(archive[key]) for key in archive.files}
    return cache_from_mapping(values)


def validate_lineage_report(path: str | Path, cache: DistributionalCache) -> dict[str, Any]:
    lineage_path = Path(path)
    try:
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DistributionalRepairContractError(f"cannot read OOF lineage report: {error}") from error
    if not isinstance(payload, Mapping):
        raise DistributionalRepairContractError("OOF lineage report root is not a mapping")
    if payload.get("protocol") != "bidirectional_emotion_utility_v1" or payload.get("status") != (
        "train_only_different_set_oof_supervision_complete; utility_model_not_yet_selected"
    ):
        raise DistributionalRepairContractError("OOF lineage is not the frozen open-role run")
    cache_contract = payload.get("cache_contract")
    counts = payload.get("task_counts")
    hashes = payload.get("hashes")
    role_audit = payload.get("sealed_audit")
    if not all(isinstance(value, Mapping) for value in (cache_contract, counts, hashes, role_audit)):
        raise DistributionalRepairContractError("OOF lineage lacks required audit mappings")
    expected_contract = {
        "schema": CACHE_SCHEMA_VERSION,
        "numeric_dtype": "float64",
        "contains_gold_labels": False,
        "contains_row_identifiers": False,
        "private_not_for_publication": True,
    }
    if any(cache_contract.get(key) != value for key, value in expected_contract.items()):
        raise DistributionalRepairContractError("OOF lineage cache contract changed")
    expected_counts = {
        "fit_oof": len(cache.fit.x),
        "model_selection": len(cache.selection.x),
        "fit_groups": len(np.unique(cache.fit.cluster_codes)),
        "model_selection_groups": len(np.unique(cache.selection.cluster_codes)),
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected_counts.items()):
        raise DistributionalRepairContractError("OOF lineage counts differ from private cache")
    for key, expected in cache.source_hashes.items():
        if _validated_sha256(str(hashes.get(key, "")), key=key) != expected:
            raise DistributionalRepairContractError(f"OOF lineage hash differs for {key}")
    expected_role_audit = {
        "calibration_rows_used_for_training_or_metrics": 0,
        "internal_holdout_rows_used_for_training_or_metrics": 0,
        "row_level_output_emitted": False,
        "test_rows_used": 0,
        "validation_rows_used": 0,
    }
    if any(role_audit.get(key) != value for key, value in expected_role_audit.items()):
        raise DistributionalRepairContractError("OOF lineage reports restricted-role use")
    return {
        "lineage_report_sha256": hashlib.sha256(lineage_path.read_bytes()).hexdigest(),
        "upstream_role_exclusion_verified": True,
        "upstream_row_level_output_emitted": False,
        "upstream_protocol": str(payload["protocol"]),
    }


def load_repair_config(
    path: str | Path,
    *,
    enforce_registered_contract: bool = True,
) -> tuple[dict[str, Any], ComponentSpec]:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DistributionalRepairContractError(f"cannot read repair config: {error}") from error
    if not isinstance(payload, Mapping):
        raise DistributionalRepairContractError("repair config root is not a mapping")
    if payload.get("protocol") != PROTOCOL:
        raise DistributionalRepairContractError("repair protocol changed")
    if payload.get("status") != "frozen_before_distributional_model_selection_result":
        raise DistributionalRepairContractError("repair config was not frozen before scoring")
    if payload.get("input_schema") != CACHE_SCHEMA_VERSION:
        raise DistributionalRepairContractError("repair input schema changed")
    component_spec = ComponentSpec.from_mapping(payload["component_model"])
    operating_points = tuple(
        (str(value["id"]), float(value["target_fit_oof_coverage"]))
        for value in payload["operating_points"]
    )
    registered_models = tuple(
        (str(value["name"]), str(value["mode"])) for value in payload["registered_models"]
    )
    if enforce_registered_contract:
        if tuple(int(seed) for seed in payload.get("model_seeds", ())) != DEFAULT_SEEDS:
            raise DistributionalRepairContractError("registered five-seed contract changed")
        if int(payload.get("group_oof_folds", -1)) != DEFAULT_OOF_FOLDS:
            raise DistributionalRepairContractError("registered OOF-fold contract changed")
        if operating_points != OPERATING_POINTS:
            raise DistributionalRepairContractError("registered operating points changed")
        if registered_models != MODEL_MODES:
            raise DistributionalRepairContractError("registered model controls changed")
        bootstrap = payload.get("crossed_bootstrap", {})
        if int(bootstrap.get("replicates", -1)) != DEFAULT_BOOTSTRAP_REPLICATES:
            raise DistributionalRepairContractError("registered bootstrap replicate count changed")
        if int(bootstrap.get("seed", -1)) != DEFAULT_BOOTSTRAP_SEED:
            raise DistributionalRepairContractError("registered bootstrap seed changed")
    return dict(payload), component_spec


def _classifier(spec: ComponentSpec, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_bins=spec.max_bins,
        early_stopping=spec.early_stopping,
        validation_fraction=spec.validation_fraction,
        n_iter_no_change=spec.n_iter_no_change,
        tol=spec.tolerance,
        random_state=int(seed),
    )


def _regressor(spec: ComponentSpec, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_bins=spec.max_bins,
        early_stopping=spec.early_stopping,
        validation_fraction=spec.validation_fraction,
        n_iter_no_change=spec.n_iter_no_change,
        tol=spec.tolerance,
        random_state=int(seed),
    )


def fit_distributional_head(
    x: np.ndarray,
    utility: np.ndarray,
    *,
    spec: ComponentSpec,
    seed: int,
) -> FittedDistributionalHead:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(utility, dtype=np.float64)
    if features.ndim != 2 or target.shape != (len(features),):
        raise DistributionalRepairContractError("head training arrays are misaligned")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise DistributionalRepairContractError("head training arrays contain non-finite values")
    positive = target > 0.0
    negative = target < 0.0
    if int(positive.sum()) < spec.minimum_conditional_rows:
        raise DistributionalRepairContractError("too few positive rows for conditional severity")
    if int(negative.sum()) < spec.minimum_conditional_rows:
        raise DistributionalRepairContractError("too few negative rows for conditional severity")
    classifier = _classifier(spec, seed)
    positive_regressor = _regressor(spec, seed + 1_009)
    negative_regressor = _regressor(spec, seed + 2_017)
    classifier.fit(features, positive.astype(np.int8))
    positive_regressor.fit(features[positive], target[positive])
    negative_regressor.fit(features[negative], -target[negative])
    positive_cap = float(np.max(target[positive]))
    negative_cap = float(np.max(-target[negative]))
    if positive_cap <= 0.0 or negative_cap <= 0.0:
        raise DistributionalRepairContractError("conditional severity support is invalid")
    return FittedDistributionalHead(
        classifier,
        positive_regressor,
        negative_regressor,
        positive_cap,
        negative_cap,
    )


def fit_oof_coverage_threshold(decision: np.ndarray, coverage: float) -> float:
    values = np.asarray(decision, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise DistributionalRepairContractError("OOF decision values are invalid")
    if not 0.0 < float(coverage) < 1.0:
        raise DistributionalRepairContractError("coverage must lie strictly between zero and one")
    target_count = min(max(int(np.rint(float(coverage) * len(values))), 1), len(values) - 1)
    ordered = np.sort(values)[::-1]
    selected_boundary = float(ordered[target_count - 1])
    excluded_boundary = float(ordered[target_count])
    if selected_boundary > excluded_boundary:
        return selected_boundary + (excluded_boundary - selected_boundary) / 2.0
    return selected_boundary


def _group_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    cluster = np.asarray(groups)
    if array.shape != cluster.shape or array.ndim != 1:
        raise DistributionalRepairContractError("grouped values are not row-aligned")
    _, inverse = np.unique(cluster, return_inverse=True)
    counts = np.bincount(inverse)
    return np.bincount(inverse, weights=array) / counts


def _safe_correlation(kind: str, truth: np.ndarray, estimate: np.ndarray) -> float | None:
    if len(truth) < 3 or np.all(truth == truth[0]) or np.all(estimate == estimate[0]):
        return None
    result = pearsonr(truth, estimate) if kind == "pearson" else spearmanr(truth, estimate)
    statistic = float(result.statistic)
    return statistic if math.isfinite(statistic) else None


def regression_metrics(
    strict_utility: np.ndarray,
    decision: np.ndarray,
    clusters: np.ndarray,
) -> tuple[dict[str, float | None], np.ndarray]:
    truth = np.asarray(strict_utility, dtype=np.float64)
    estimate = np.asarray(decision, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 1:
        raise DistributionalRepairContractError("strict regression arrays are misaligned")
    squared_error = np.square(truth - estimate)
    cluster_rmse = np.sqrt(_group_means(squared_error, clusters))
    metrics: dict[str, float | None] = {
        "row_rmse": float(np.sqrt(np.mean(squared_error))),
        "cluster_macro_rmse": float(cluster_rmse.mean()),
        "row_mae": float(np.mean(np.abs(truth - estimate))),
        "pearson_correlation": _safe_correlation("pearson", truth, estimate),
        "spearman_correlation": _safe_correlation("spearman", truth, estimate),
        "mean_strict_utility": float(truth.mean()),
        "mean_decision_value": float(estimate.mean()),
    }
    return metrics, cluster_rmse


def policy_metrics(
    strict_utility: np.ndarray,
    decision: np.ndarray,
    clusters: np.ndarray,
    *,
    threshold: float,
) -> tuple[dict[str, float | int | None], np.ndarray]:
    truth = np.asarray(strict_utility, dtype=np.float64)
    estimate = np.asarray(decision, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 1:
        raise DistributionalRepairContractError("policy arrays are misaligned")
    selected = estimate > float(threshold)
    selected_count = int(selected.sum())
    harmful = truth < 0.0
    beneficial = truth > 0.0
    policy_utility = np.where(selected, truth, 0.0)
    excess_nll = -policy_utility
    cluster_excess = _group_means(excess_nll, clusters)
    selected_truth = truth[selected]
    result: dict[str, float | int | None] = {
        "threshold": float(threshold),
        "selected_rows": selected_count,
        "coverage": float(selected.mean()),
        "selected_harm_rate": (
            float(np.mean(selected_truth < 0.0)) if selected_count else None
        ),
        "selected_positive_precision": (
            float(np.mean(selected_truth > 0.0)) if selected_count else None
        ),
        "harm_incidence_all_rows": float(np.mean(selected & harmful)),
        "benefit_incidence_all_rows": float(np.mean(selected & beneficial)),
        "absolute_row_mean_excess_nll_vs_fallback": float(excess_nll.mean()),
        "absolute_cluster_macro_excess_nll_vs_fallback": float(cluster_excess.mean()),
        "selected_only_mean_excess_nll_vs_fallback": (
            float((-selected_truth).mean()) if selected_count else None
        ),
        "mean_policy_utility": float(policy_utility.mean()),
    }
    return result, cluster_excess


def directional_head_metrics(actual: np.ndarray, head: HeadPrediction) -> dict[str, float | None]:
    target = np.asarray(actual, dtype=np.float64)
    positive = target > 0.0
    negative = target < 0.0
    probability = head.positive_probability
    sign_auc = float(roc_auc_score(positive.astype(np.int8), probability))
    expected_error = target - head.expected_utility
    return {
        "positive_sign_auc": sign_auc,
        "positive_sign_brier": float(np.mean(np.square(positive.astype(float) - probability))),
        "expected_utility_rmse": float(np.sqrt(np.mean(np.square(expected_error)))),
        "expected_utility_pearson": _safe_correlation(
            "pearson", target, head.expected_utility
        ),
        "expected_utility_spearman": _safe_correlation(
            "spearman", target, head.expected_utility
        ),
        "positive_severity_rmse_on_positive_rows": float(
            np.sqrt(np.mean(np.square(target[positive] - head.positive_magnitude[positive])))
        ),
        "negative_severity_rmse_on_negative_rows": float(
            np.sqrt(np.mean(np.square(-target[negative] - head.negative_magnitude[negative])))
        ),
    }


def crossed_seed_shared_cluster_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int | str]:
    """Paired crossed bootstrap using one shared cluster draw per replicate."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape or candidate_array.ndim != 2:
        raise DistributionalRepairContractError(
            "bootstrap inputs must be aligned seed-by-cluster matrices"
        )
    n_seeds, n_clusters = candidate_array.shape
    if n_seeds < 2 or n_clusters < 2 or int(replicates) < 100:
        raise DistributionalRepairContractError("bootstrap dimensions or replicates are insufficient")
    if not np.isfinite(candidate_array).all() or not np.isfinite(reference_array).all():
        raise DistributionalRepairContractError("bootstrap inputs contain non-finite values")
    difference = candidate_array - reference_array
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    batch_size = 256
    for start in range(0, int(replicates), batch_size):
        stop = min(start + batch_size, int(replicates))
        batch = stop - start
        seed_draw = rng.integers(0, n_seeds, size=(batch, n_seeds))
        cluster_draw = rng.integers(0, n_clusters, size=(batch, n_clusters))
        sampled = difference[
            seed_draw[:, :, None],
            cluster_draw[:, None, :],
        ]
        bootstrap[start:stop] = sampled.mean(axis=(1, 2))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "design": "crossed_seed_with_one_shared_whole_cluster_draw_per_replicate",
        "difference_definition": "candidate_minus_reference",
        "point_difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_probability_difference_below_zero": float(np.mean(bootstrap < 0.0)),
        "replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _head_seed(training_seed: int, fold: int | None, slot: int) -> int:
    fold_offset = 0 if fold is None else (int(fold) + 1) * 10_000_019
    return int(training_seed) + fold_offset + int(slot) * 100_003


def _fit_three_head_slots(
    train: UtilitySplit,
    predict_x: np.ndarray,
    *,
    spec: ComponentSpec,
    training_seed: int,
    fold: int | None,
) -> tuple[dict[str, HeadPrediction], dict[str, dict[str, int]]]:
    plans = (
        ("forward_first", train.forward, 0),
        ("forward_second", train.forward, 1),
        ("backward_second", train.backward, 1),
    )
    predictions: dict[str, HeadPrediction] = {}
    iterations: dict[str, dict[str, int]] = {}
    for name, target, slot in plans:
        head = fit_distributional_head(
            train.x,
            target,
            spec=spec,
            seed=_head_seed(training_seed, fold, slot),
        )
        predictions[name] = head.predict(predict_x)
        iterations[name] = head.fitted_iterations
    return predictions, iterations


def generate_seed_predictions(
    cache: DistributionalCache,
    *,
    spec: ComponentSpec,
    training_seed: int,
    oof_folds: int,
) -> tuple[dict[str, ModelPrediction], dict[str, ModelPrediction], dict[str, Any]]:
    fit = cache.fit
    unique_clusters = np.unique(fit.cluster_codes)
    n_splits = min(int(oof_folds), len(unique_clusters))
    if n_splits < 2:
        raise DistributionalRepairContractError("group OOF requires at least two folds")
    head_names = ("forward_first", "forward_second", "backward_second")
    oof_components: dict[str, dict[str, np.ndarray]] = {
        name: {
            "positive_probability": np.full(len(fit.x), np.nan),
            "positive_magnitude": np.full(len(fit.x), np.nan),
            "negative_magnitude": np.full(len(fit.x), np.nan),
            "expected_utility": np.full(len(fit.x), np.nan),
        }
        for name in head_names
    }
    fold_by_row = np.full(len(fit.x), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, held_index) in enumerate(
        splitter.split(fit.x, groups=fit.cluster_codes)
    ):
        if set(fit.cluster_codes[train_index]) & set(fit.cluster_codes[held_index]):
            raise AssertionError("GroupKFold leaked a cluster")
        train = UtilitySplit.validated(
            fit.x[train_index],
            fit.forward[train_index],
            fit.backward[train_index],
            fit.cluster_codes[train_index],
            label=f"fit OOF fold {fold}",
            require_multiple_clusters=False,
        )
        heads, _ = _fit_three_head_slots(
            train,
            fit.x[held_index],
            spec=spec,
            training_seed=int(training_seed),
            fold=fold,
        )
        fold_by_row[held_index] = fold
        for name, head in heads.items():
            for field in oof_components[name]:
                oof_components[name][field][held_index] = getattr(head, field)
        print(
            f"distributional repair seed={training_seed} fit-OOF fold={fold + 1}/{n_splits}",
            flush=True,
        )
    if np.any(fold_by_row < 0):
        raise AssertionError("fit OOF did not cover every row")
    if any(
        not np.isfinite(array).all()
        for component in oof_components.values()
        for array in component.values()
    ):
        raise AssertionError("fit OOF produced non-finite components")
    oof_heads = {
        name: HeadPrediction(**component) for name, component in oof_components.items()
    }
    selection_heads, final_iterations = _fit_three_head_slots(
        fit,
        cache.selection.x,
        spec=spec,
        training_seed=int(training_seed),
        fold=None,
    )
    print(f"distributional repair seed={training_seed} full-fit complete", flush=True)
    oof_models = compose_registered_models(
        oof_heads["forward_first"],
        oof_heads["forward_second"],
        oof_heads["backward_second"],
    )
    selection_models = compose_registered_models(
        selection_heads["forward_first"],
        selection_heads["forward_second"],
        selection_heads["backward_second"],
    )
    diagnostics = {
        "oof_folds": n_splits,
        "final_fit_component_iterations": final_iterations,
    }
    return oof_models, selection_models, diagnostics


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _summarize_seed_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regression_keys = (
        "row_rmse",
        "cluster_macro_rmse",
        "row_mae",
        "pearson_correlation",
        "spearman_correlation",
    )
    policy_keys = (
        "coverage",
        "selected_harm_rate",
        "selected_positive_precision",
        "harm_incidence_all_rows",
        "absolute_row_mean_excess_nll_vs_fallback",
        "absolute_cluster_macro_excess_nll_vs_fallback",
        "selected_only_mean_excess_nll_vs_fallback",
    )
    summary: dict[str, Any] = {
        "model_selection_strict_regression": {},
        "operating_points": {},
    }
    for key in regression_keys:
        values = [record["model_selection_strict_regression"][key] for record in records]
        finite = [float(value) for value in values if value is not None]
        summary["model_selection_strict_regression"][key] = (
            _mean_std(finite) if len(finite) == len(values) else None
        )
    for operating_point in (*[name for name, _ in OPERATING_POINTS], ZERO_OPERATING_POINT):
        operating_summary: dict[str, Any] = {}
        for key in policy_keys:
            values = [
                record["operating_points"][operating_point]["model_selection"][key]
                for record in records
            ]
            finite = [float(value) for value in values if value is not None]
            operating_summary[key] = _mean_std(finite) if len(finite) == len(values) else None
        summary["operating_points"][operating_point] = operating_summary
    return summary


def run_distributional_model_selection(
    cache: DistributionalCache,
    *,
    component_spec: ComponentSpec,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    oof_folds: int = DEFAULT_OOF_FOLDS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    enforce_registered_contract: bool = True,
) -> dict[str, Any]:
    model_seeds = tuple(int(seed) for seed in seeds)
    if len(model_seeds) < 2 or len(set(model_seeds)) != len(model_seeds):
        raise DistributionalRepairContractError("model seeds must be distinct")
    if enforce_registered_contract:
        if model_seeds != DEFAULT_SEEDS:
            raise DistributionalRepairContractError("registered run requires the frozen five seeds")
        if int(oof_folds) != DEFAULT_OOF_FOLDS:
            raise DistributionalRepairContractError("registered run requires five group OOF folds")
        if int(bootstrap_replicates) != DEFAULT_BOOTSTRAP_REPLICATES:
            raise DistributionalRepairContractError("registered run requires 10000 bootstrap replicates")
        if int(bootstrap_seed) != DEFAULT_BOOTSTRAP_SEED:
            raise DistributionalRepairContractError("registered bootstrap seed changed")

    public_records: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in MODEL_MODES}
    private_excess: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            **{operating_point: [] for operating_point, _ in OPERATING_POINTS},
            ZERO_OPERATING_POINT: [],
        }
        for name, _ in MODEL_MODES
    }
    private_rmse: dict[str, list[np.ndarray]] = {name: [] for name, _ in MODEL_MODES}

    for seed in model_seeds:
        oof_models, selection_models, training_diagnostics = generate_seed_predictions(
            cache,
            spec=component_spec,
            training_seed=seed,
            oof_folds=oof_folds,
        )
        for model_name, _ in MODEL_MODES:
            oof_prediction = oof_models[model_name]
            selection_prediction = selection_models[model_name]
            fit_regression, _ = regression_metrics(
                cache.fit.strict_utility,
                oof_prediction.decision,
                cache.fit.cluster_codes,
            )
            selection_regression, selection_cluster_rmse = regression_metrics(
                cache.selection.strict_utility,
                selection_prediction.decision,
                cache.selection.cluster_codes,
            )
            private_rmse[model_name].append(selection_cluster_rmse)
            operating_records: dict[str, Any] = {}
            for operating_point, target_coverage in OPERATING_POINTS:
                threshold = fit_oof_coverage_threshold(
                    oof_prediction.decision, target_coverage
                )
                fit_policy, _ = policy_metrics(
                    cache.fit.strict_utility,
                    oof_prediction.decision,
                    cache.fit.cluster_codes,
                    threshold=threshold,
                )
                selection_policy, selection_cluster_excess = policy_metrics(
                    cache.selection.strict_utility,
                    selection_prediction.decision,
                    cache.selection.cluster_codes,
                    threshold=threshold,
                )
                private_excess[model_name][operating_point].append(selection_cluster_excess)
                operating_records[operating_point] = {
                    "role": (
                        "prespecified_primary_development_operating_point_not_confirmatory"
                        if operating_point == PRIMARY_OPERATING_POINT
                        else "exploratory_safety_operating_point"
                    ),
                    "target_fit_oof_coverage": target_coverage,
                    "fit_oof_frozen_threshold": threshold,
                    "fit_oof": fit_policy,
                    "model_selection": selection_policy,
                }
            fit_zero, _ = policy_metrics(
                cache.fit.strict_utility,
                oof_prediction.decision,
                cache.fit.cluster_codes,
                threshold=0.0,
            )
            selection_zero, selection_cluster_zero = policy_metrics(
                cache.selection.strict_utility,
                selection_prediction.decision,
                cache.selection.cluster_codes,
                threshold=0.0,
            )
            private_excess[model_name][ZERO_OPERATING_POINT].append(selection_cluster_zero)
            operating_records[ZERO_OPERATING_POINT] = {
                "role": "exploratory_diagnostic_only",
                "threshold": 0.0,
                "fit_oof": fit_zero,
                "model_selection": selection_zero,
            }
            target_arrays = {
                "forward": cache.selection.forward,
                "backward": cache.selection.backward,
            }
            head_diagnostics = {
                f"head_{index + 1}_{target_name}": directional_head_metrics(
                    target_arrays[target_name], head
                )
                for index, (target_name, head) in enumerate(
                    zip(
                        selection_prediction.head_targets,
                        selection_prediction.heads,
                        strict=True,
                    )
                )
            }
            public_records[model_name].append(
                {
                    "seed": seed,
                    "oof_folds": int(training_diagnostics["oof_folds"]),
                    "final_fit_component_iterations": {
                        key: training_diagnostics["final_fit_component_iterations"][key]
                        for key in (
                            ("forward_first",)
                            if model_name == "distributional_forward_only"
                            else ("backward_second",)
                            if model_name == "distributional_backward_only"
                            else ("forward_first", "forward_second")
                            if model_name == "distributional_pseudo_bidirectional"
                            else ("forward_first", "backward_second")
                        )
                    },
                    "fit_oof_strict_regression": fit_regression,
                    "model_selection_strict_regression": selection_regression,
                    "model_selection_head_diagnostics": head_diagnostics,
                    "operating_points": operating_records,
                }
            )

    model_reports: list[dict[str, Any]] = []
    private_excess_matrix = {
        model: {
            operating_point: np.stack(values)
            for operating_point, values in by_point.items()
        }
        for model, by_point in private_excess.items()
    }
    private_rmse_matrix = {
        model: np.stack(values) for model, values in private_rmse.items()
    }
    for model_name, mode in MODEL_MODES:
        head_count = 2 if mode in {"pseudo_bidirectional", "true_bidirectional"} else 1
        target_semantics = {
            "forward_only": "one_forward_distributional_head",
            "backward_only": "one_backward_distributional_head",
            "pseudo_bidirectional": "two_capacity_matched_heads_both_supervised_by_forward_utility",
            "true_bidirectional": "two_capacity_matched_heads_supervised_by_forward_and_backward_utility",
        }[mode]
        model_reports.append(
            {
                "name": model_name,
                "mode": mode,
                "target_semantics": target_semantics,
                "head_count": head_count,
                "component_estimators_per_head": 3,
                "total_component_estimators": 3 * head_count,
                "decision_rule": (
                    "minimum_of_two_expected_directional_utilities"
                    if head_count == 2
                    else "single_expected_directional_utility"
                ),
                "per_seed": public_records[model_name],
                "five_seed_summary": _summarize_seed_records(public_records[model_name]),
            }
        )

    ranked = sorted(
        model_reports,
        key=lambda model: (
            model["five_seed_summary"]["operating_points"][PRIMARY_OPERATING_POINT][
                "absolute_cluster_macro_excess_nll_vs_fallback"
            ]["mean"],
            model["five_seed_summary"]["model_selection_strict_regression"][
                "cluster_macro_rmse"
            ]["mean"],
            model["name"],
        ),
    )
    ranking = [
        {
            "rank": index + 1,
            "name": model["name"],
            "five_seed_mean_primary_absolute_cluster_macro_excess_nll_vs_fallback": model[
                "five_seed_summary"
            ]["operating_points"][PRIMARY_OPERATING_POINT][
                "absolute_cluster_macro_excess_nll_vs_fallback"
            ]["mean"],
            "five_seed_mean_cluster_macro_strict_utility_rmse": model[
                "five_seed_summary"
            ]["model_selection_strict_regression"]["cluster_macro_rmse"]["mean"],
        }
        for index, model in enumerate(ranked)
    ]

    true_report = next(model for model in model_reports if model["name"] == TRUE_MODEL_NAME)
    paired_contrasts: list[dict[str, Any]] = []
    for reference_index, (reference_name, _) in enumerate(MODEL_MODES[:-1]):
        reference_report = next(model for model in model_reports if model["name"] == reference_name)
        operating_contrasts: dict[str, Any] = {}
        for point_index, operating_point in enumerate(
            (*[name for name, _ in OPERATING_POINTS], ZERO_OPERATING_POINT)
        ):
            candidate_matrix = private_excess_matrix[TRUE_MODEL_NAME][operating_point]
            reference_matrix = private_excess_matrix[reference_name][operating_point]
            candidate_seed_means = candidate_matrix.mean(axis=1)
            reference_seed_means = reference_matrix.mean(axis=1)
            candidate_policy = true_report["five_seed_summary"]["operating_points"][
                operating_point
            ]
            reference_policy = reference_report["five_seed_summary"]["operating_points"][
                operating_point
            ]
            point_differences: dict[str, float | None] = {}
            for key in (
                "coverage",
                "selected_harm_rate",
                "selected_positive_precision",
                "harm_incidence_all_rows",
                "absolute_row_mean_excess_nll_vs_fallback",
                "absolute_cluster_macro_excess_nll_vs_fallback",
            ):
                candidate_metric = candidate_policy[key]
                reference_metric = reference_policy[key]
                point_differences[key] = (
                    float(candidate_metric["mean"] - reference_metric["mean"])
                    if candidate_metric is not None and reference_metric is not None
                    else None
                )
            operating_contrasts[operating_point] = {
                "role": (
                    "primary_open_role_development_contrast_not_confirmatory"
                    if operating_point == PRIMARY_OPERATING_POINT
                    else "exploratory_contrast"
                ),
                "five_seed_mean_point_differences": point_differences,
                "seed_wins_lower_absolute_cluster_macro_excess_nll_out_of_five": int(
                    np.sum(candidate_seed_means < reference_seed_means)
                ),
                "absolute_cluster_macro_excess_nll_crossed_bootstrap": (
                    crossed_seed_shared_cluster_bootstrap(
                        candidate_matrix,
                        reference_matrix,
                        replicates=bootstrap_replicates,
                        seed=int(bootstrap_seed) + reference_index * 101 + point_index,
                    )
                ),
            }
        candidate_rmse = private_rmse_matrix[TRUE_MODEL_NAME]
        reference_rmse = private_rmse_matrix[reference_name]
        paired_contrasts.append(
            {
                "candidate": TRUE_MODEL_NAME,
                "reference": reference_name,
                "difference_definition": "candidate_minus_reference",
                "operating_points": operating_contrasts,
                "strict_utility_regression": {
                    "seed_wins_lower_cluster_macro_rmse_out_of_five": int(
                        np.sum(candidate_rmse.mean(axis=1) < reference_rmse.mean(axis=1))
                    ),
                    "cluster_macro_rmse_crossed_bootstrap": (
                        crossed_seed_shared_cluster_bootstrap(
                            candidate_rmse,
                            reference_rmse,
                            replicates=bootstrap_replicates,
                            seed=int(bootstrap_seed) + 10_000 + reference_index,
                        )
                    ),
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "open_role_distributional_repair_model_selection_complete_restricted_roles_unread",
        "polarity": "benefit_positive",
        "analysis_stage": {
            "repair": "repair_1_of_3_distributional_sign_by_severity",
            "model_selection_results_are_confirmatory": False,
            "primary_25pct_role": "prespecified_development_analogue_only",
            "safety_10pct_role": "exploratory_only",
            "zero_threshold_role": "exploratory_diagnostic_only",
        },
        "claim_boundary": (
            "Open-role fit OOF and model-selection evidence only. It cannot establish "
            "confirmatory classification or safety improvement; calibration, internal "
            "holdout, validation, and test remain unread."
        ),
        "data_contract": {
            "input_schema": CACHE_SCHEMA_VERSION,
            "fit_rows": int(len(cache.fit.x)),
            "fit_clusters": int(len(np.unique(cache.fit.cluster_codes))),
            "model_selection_rows": int(len(cache.selection.x)),
            "model_selection_clusters": int(len(np.unique(cache.selection.cluster_codes))),
            "feature_count": int(cache.fit.x.shape[1]),
            "restricted_roles_read": False,
            "row_level_output": False,
            "cluster_identifiers_emitted": False,
            "source_hashes": dict(sorted(cache.source_hashes.items())),
        },
        "design_contract": {
            "model_seeds": list(model_seeds),
            "group_oof_folds": int(oof_folds),
            "component_model": component_spec.public_dict(),
            "directional_estimand": (
                "P(u>0)*E[u|u>0]-(1-P(u>0))*E[-u|u<0]"
            ),
            "severity_scale": "raw_nonnegative_magnitude_squared_error",
            "true_bidirectional_decision": "minimum_forward_backward_expected_utility",
            "pseudo_control": (
                "two stochastic forward-supervised heads; same head count, six estimator "
                "slots, capacity hyperparameters, seeds, and folds as true bidirectional"
            ),
            "fit_oof_thresholds": {
                "primary_25pct": 0.25,
                "safety_exploration_10pct": 0.10,
                "source": "fit_group_oof_only",
                "transfer": "frozen_before_model_selection_scoring",
                "tie_rule": "strict_greater_than_boundary_ties_fallback",
            },
            "zero_threshold": "diagnostic_only",
            "model_selection_hyperparameter_or_threshold_tuning": False,
            "ranking_metric": (
                "five_seed_mean_model_selection_absolute_cluster_macro_excess_nll_"
                "vs_fallback_at_primary_25pct"
            ),
        },
        "uncertainty_contract": {
            "replicates": int(bootstrap_replicates),
            "base_seed": int(bootstrap_seed),
            "pairing": "same_training_seed_and_same_model_selection_cluster",
            "resampling": (
                "training_seed_with_replacement_and_one_shared_whole_cluster_draw_"
                "per_replicate"
            ),
            "interval": "percentile_95pct",
            "difference_definition": "candidate_minus_reference",
        },
        "models": model_reports,
        "ranking": ranking,
        "paired_true_bidirectional_contrasts": paired_contrasts,
        "selected_open_role_model": ranking[0]["name"],
        "privacy_contract": {
            "aggregate_only": True,
            "row_identifiers_emitted": False,
            "cluster_identifiers_emitted": False,
            "private_cache_not_published": True,
        },
    }
    assert_aggregate_report(report)
    return report


def assert_aggregate_report(report: Mapping[str, Any]) -> None:
    forbidden_exact = {
        "predictions",
        "probabilities",
        "decision_values",
        "utilities",
        "labels",
        "cluster_codes",
        "cluster_ids",
        "row_ids",
        "query_ids",
    }

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, np.ndarray):
            raise DistributionalRepairContractError(
                f"aggregate report contains an ndarray at {'.'.join(path)}"
            )
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                if name.lower() in forbidden_exact:
                    raise DistributionalRepairContractError(
                        f"aggregate report contains forbidden field {'.'.join((*path, name))}"
                    )
                visit(child, (*path, name))
            return
        if isinstance(value, (list, tuple)):
            if len(value) > 20:
                raise DistributionalRepairContractError(
                    f"aggregate report contains an overlong list at {'.'.join(path)}"
                )
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise DistributionalRepairContractError(
                f"aggregate report contains non-finite value at {'.'.join(path)}"
            )
        if not isinstance(value, (str, int, float, bool, type(None), np.integer, np.floating)):
            raise DistributionalRepairContractError(
                f"aggregate report contains a non-JSON value at {'.'.join(path)}"
            )

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise DistributionalRepairContractError("aggregate report schema changed")
    models = report.get("models")
    ranking = report.get("ranking")
    contrasts = report.get("paired_true_bidirectional_contrasts")
    if not isinstance(models, Sequence) or len(models) != 4:
        raise DistributionalRepairContractError("aggregate report must contain four models")
    if not isinstance(ranking, Sequence) or len(ranking) != 4:
        raise DistributionalRepairContractError("aggregate report must contain four ranks")
    if not isinstance(contrasts, Sequence) or len(contrasts) != 3:
        raise DistributionalRepairContractError("aggregate report must contain three contrasts")
    visit(report, ())


def write_aggregate_report(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    assert_aggregate_report(report)
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise DistributionalRepairContractError("aggregate output must use a .json suffix")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"aggregate report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_private_distributional_repair(
    private_cache_path: str | Path,
    lineage_report_path: str | Path,
    repair_config_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the registered open-role repair; there is no restricted-role input."""

    cache_path = Path(private_cache_path)
    lineage_path = Path(lineage_report_path)
    config_path = Path(repair_config_path)
    config, component_spec = load_repair_config(config_path)
    cache = load_private_cache(cache_path)
    lineage = validate_lineage_report(lineage_path, cache)
    report = run_distributional_model_selection(
        cache,
        component_spec=component_spec,
        seeds=tuple(int(seed) for seed in config["model_seeds"]),
        oof_folds=int(config["group_oof_folds"]),
        bootstrap_replicates=int(config["crossed_bootstrap"]["replicates"]),
        bootstrap_seed=int(config["crossed_bootstrap"]["seed"]),
        enforce_registered_contract=True,
    )
    module_path = Path(__file__)
    script_path = module_path.parents[2] / "scripts" / "run_distributional_utility_repair.py"
    report["data_contract"].update(lineage)
    report["data_contract"]["hashes"] = {
        "private_cache_sha256": _sha256(cache_path),
        "lineage_report_sha256": _sha256(lineage_path),
        "repair_config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(module_path),
        "runner_sha256": _sha256(script_path) if script_path.exists() else None,
    }
    report["data_contract"]["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }
    manifest_material = json.dumps(
        {
            "hashes": report["data_contract"]["hashes"],
            "environment": report["data_contract"]["environment"],
            "seeds": list(DEFAULT_SEEDS),
        },
        sort_keys=True,
    ).encode("utf-8")
    report["data_contract"]["reproducibility_manifest_sha256"] = hashlib.sha256(
        manifest_material
    ).hexdigest()
    assert_aggregate_report(report)
    write_aggregate_report(report, output_path, overwrite=overwrite)
    return report
