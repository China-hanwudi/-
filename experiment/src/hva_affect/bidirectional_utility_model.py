"""Train-only models for benefit-positive bidirectional history utility.

This module deliberately has no knowledge of EmotionTalk raw data or any
sealed validation/test split.  Its only file input is the private
``emotiontalk_bidirectional_oof_cache_v1`` contract, which contains an OOF
fit split and a disjoint model-selection split.  Public output is aggregate
JSON; row predictions and cluster identifiers never enter that report.

Polarity is fixed throughout: a positive utility means that retaining the
candidate history item helps prediction.  Zero remains the semantic fallback
threshold for diagnostics.  The registered primary comparison transfers a
single 25% coverage threshold fitted on group-OOF scores to model selection;
the shared model scores candidates with the minimum of its two utility heads.
V3 keeps that deployment-style point estimate, adds an explicitly transductive
exact-25% score-ranking diagnostic, and reports a crossed training-seed/shared-
cluster paired bootstrap as the primary open-role sensitivity alongside the
legacy seed-then-independent-cluster bootstrap.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


CACHE_SCHEMA_VERSION = "emotiontalk_bidirectional_oof_cache_v1"
REPORT_SCHEMA_VERSION = "bidirectional_utility_model_report_v3"
# Must match experiment/configs/carma_confirmatory_analysis_v1.json.
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
MAX_TRAINABLE_PARAMETERS = 2_000_000
PRIMARY_HISTORY_COVERAGE = 0.25
PAIRED_BOOTSTRAP_REPLICATES = 10_000
PAIRED_BOOTSTRAP_SEED = 20_260_808

MODEL_MODES = (
    "forward_only",
    "backward_only",
    "pseudo_bidirectional_shared",
    "bidirectional_shared",
)
TWO_HEAD_MODES = {"pseudo_bidirectional_shared", "bidirectional_shared"}
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
_SEALED_KEY_TOKENS = {"test", "validation", "holdout", "calibration", "sealed"}


class UtilityModelContractError(ValueError):
    """Raised when a cache, split, model, or aggregate report breaks contract."""


@dataclass(frozen=True)
class UtilitySplit:
    """One row-aligned, identifier-free utility modelling split."""

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
            raise UtilityModelContractError(
                f"{label} x must have shape (rows>=2, features>=1)"
            )
        rows = features.shape[0]
        if forward_array.shape != (rows,) or backward_array.shape != (rows,):
            raise UtilityModelContractError(
                f"{label} utility targets must be one-dimensional"
            )
        if clusters.shape != (rows,) or not np.issubdtype(clusters.dtype, np.integer):
            raise UtilityModelContractError(
                f"{label} cluster codes must be a one-dimensional integer array"
            )
        if not np.isfinite(features).all():
            raise UtilityModelContractError(
                f"{label} features contain non-finite values"
            )
        if (
            not np.isfinite(forward_array).all()
            or not np.isfinite(backward_array).all()
        ):
            raise UtilityModelContractError(
                f"{label} utilities contain non-finite values"
            )
        if np.any(clusters < 0):
            raise UtilityModelContractError(
                f"{label} cluster codes must be non-negative"
            )
        if require_multiple_clusters and len(np.unique(clusters)) < 2:
            raise UtilityModelContractError(f"{label} requires at least two clusters")
        return cls(
            features,
            forward_array,
            backward_array,
            clusters.astype(np.int64, copy=False),
        )

    @property
    def strict_bidirectional_utility(self) -> np.ndarray:
        """Benefit that is positive only when both directional targets are positive."""

        return np.minimum(self.forward, self.backward)


@dataclass(frozen=True)
class BidirectionalUtilityCache:
    fit: UtilitySplit
    selection: UtilitySplit
    feature_names: tuple[str, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class UtilityModelSpec:
    """Architecture shared by the three directional ablations."""

    name: str
    mode: str
    hidden_layer_sizes: tuple[int, ...] = (32, 16)
    alpha: float = 1e-3
    max_iter: int = 80
    tolerance: float = 1e-4
    activation: str = "tanh"
    solver: str = "adam"
    batch_size: int = 1024
    learning_rate_init: float = 2e-3
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 8

    def __post_init__(self) -> None:
        if self.mode not in MODEL_MODES:
            raise UtilityModelContractError(f"unknown utility model mode: {self.mode}")
        if not self.name:
            raise UtilityModelContractError("model name must be non-empty")
        if not self.hidden_layer_sizes or any(
            int(width) < 1 for width in self.hidden_layer_sizes
        ):
            raise UtilityModelContractError(
                "hidden layers must contain positive widths"
            )
        if self.alpha < 0 or self.max_iter < 1 or self.tolerance <= 0:
            raise UtilityModelContractError("invalid MLP optimization settings")
        if self.batch_size < 1 or self.learning_rate_init <= 0:
            raise UtilityModelContractError(
                "invalid MLP batch or learning-rate settings"
            )
        if not 0 < self.validation_fraction < 1 or self.n_iter_no_change < 1:
            raise UtilityModelContractError("invalid MLP early-stopping settings")
        if self.activation not in {"identity", "logistic", "tanh", "relu"}:
            raise UtilityModelContractError(
                f"unsupported MLP activation: {self.activation}"
            )
        if self.solver not in {"lbfgs", "sgd", "adam"}:
            raise UtilityModelContractError(f"unsupported MLP solver: {self.solver}")


@dataclass(frozen=True)
class UtilityPredictions:
    """In-memory predictions; never serialized by the aggregate report writer."""

    forward: np.ndarray | None
    backward: np.ndarray | None
    decision_score: np.ndarray

    def selected(self, threshold: float = 0.0) -> np.ndarray:
        """Return True only for benefit-positive candidates; zero falls back."""

        return np.asarray(self.decision_score, dtype=np.float64) > float(threshold)


@dataclass
class FittedUtilityModel:
    spec: UtilityModelSpec
    seed: int
    x_scaler: StandardScaler
    target_mean: np.ndarray
    target_scale: np.ndarray
    estimator: MLPRegressor
    parameter_count: int

    def predict(self, x: np.ndarray) -> UtilityPredictions:
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.x_scaler.n_features_in_:
            raise UtilityModelContractError(
                "prediction features do not match fitted feature schema"
            )
        if not np.isfinite(features).all():
            raise UtilityModelContractError(
                "prediction features contain non-finite values"
            )
        standardized = np.asarray(
            self.estimator.predict(self.x_scaler.transform(features))
        )
        if standardized.ndim == 1:
            standardized = standardized[:, None]
        raw = standardized * self.target_scale[None, :] + self.target_mean[None, :]
        if self.spec.mode == "forward_only":
            forward = raw[:, 0]
            return UtilityPredictions(forward, None, forward.copy())
        if self.spec.mode == "backward_only":
            backward = raw[:, 0]
            return UtilityPredictions(None, backward, backward.copy())
        if raw.shape[1] != 2:
            raise AssertionError(
                "shared two-head model must emit exactly two utilities"
            )
        forward, backward = raw[:, 0], raw[:, 1]
        return UtilityPredictions(forward, backward, np.minimum(forward, backward))


@dataclass(frozen=True)
class OOFPredictions:
    predictions: UtilityPredictions
    fold_by_row: np.ndarray
    n_splits: int


def default_model_specs() -> tuple[UtilityModelSpec, ...]:
    """Return two directional, one degenerate, and one true bidirectional model."""

    return (
        UtilityModelSpec(name="forward_only_mlp", mode="forward_only"),
        UtilityModelSpec(name="backward_only_mlp", mode="backward_only"),
        UtilityModelSpec(
            name="pseudo_bidirectional_same_set_mlp",
            mode="pseudo_bidirectional_shared",
        ),
        UtilityModelSpec(name="bidirectional_shared_mlp", mode="bidirectional_shared"),
    )


def _contains_sealed_token(key: str) -> bool:
    tokens = key.lower().replace("-", "_").split("_")
    return any(token in _SEALED_KEY_TOKENS for token in tokens)


def _single_string(value: np.ndarray, *, key: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise UtilityModelContractError(f"{key} must contain exactly one string")
    return str(array.reshape(-1)[0])


def _validated_sha256(value: str, *, key: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise UtilityModelContractError(f"{key} must be a lowercase SHA-256 digest")
    return digest


def cache_from_mapping(values: Mapping[str, np.ndarray]) -> BidirectionalUtilityCache:
    """Validate a cache mapping without touching raw or sealed data."""

    keys = set(values)
    sealed = sorted(key for key in keys if _contains_sealed_token(key))
    if sealed:
        raise UtilityModelContractError(f"sealed split fields are forbidden: {sealed}")
    missing = sorted(_CORE_CACHE_KEYS - keys)
    unknown = sorted(keys - _CORE_CACHE_KEYS - _OPTIONAL_CACHE_KEYS)
    if missing or unknown:
        raise UtilityModelContractError(
            f"cache schema mismatch: missing={missing}, unknown={unknown}"
        )
    if (
        _single_string(values["schema_version"], key="schema_version")
        != CACHE_SCHEMA_VERSION
    ):
        raise UtilityModelContractError("cache schema version changed")

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
        label="selection",
    )
    if fit.x.shape[1] != selection.x.shape[1]:
        raise UtilityModelContractError("fit and selection feature dimensions differ")

    if "feature_names" in values:
        names = tuple(
            str(name) for name in np.asarray(values["feature_names"]).reshape(-1)
        )
        if len(names) != fit.x.shape[1] or len(set(names)) != len(names):
            raise UtilityModelContractError(
                "feature_names must be unique and feature-aligned"
            )
    else:
        names = tuple(f"feature_{index}" for index in range(fit.x.shape[1]))

    for prefix, rows in (("fit", len(fit.x)), ("selection", len(selection.x))):
        for direction in ("forward", "backward"):
            key = f"{prefix}_{direction}_seed"
            if key not in values:
                continue
            seed_targets = np.asarray(values[key], dtype=np.float64)
            if seed_targets.ndim != 2 or seed_targets.shape[1] != rows:
                raise UtilityModelContractError(
                    f"{key} must have shape (seeds, {rows})"
                )
            if not np.isfinite(seed_targets).all():
                raise UtilityModelContractError(f"{key} contains non-finite values")

    hashes = {
        key: _validated_sha256(_single_string(values[key], key=key), key=key)
        for key in ("base_config_sha256", "utility_config_sha256")
    }
    return BidirectionalUtilityCache(fit, selection, names, hashes)


def load_private_oof_cache(path: str | Path) -> BidirectionalUtilityCache:
    """Load only the whitelisted train-only cache fields with pickle disabled."""

    cache_path = Path(path)
    if cache_path.suffix.lower() != ".npz":
        raise UtilityModelContractError("private utility cache must be an .npz archive")
    with np.load(cache_path, allow_pickle=False) as archive:
        # Inspect names before reading any value so a sealed split fails closed.
        keys = set(archive.files)
        sealed = sorted(key for key in keys if _contains_sealed_token(key))
        if sealed:
            raise UtilityModelContractError(
                f"sealed split fields are forbidden: {sealed}"
            )
        values = {key: np.asarray(archive[key]) for key in keys}
    return cache_from_mapping(values)


def validate_oof_lineage_report(
    path: str | Path,
    cache: BidirectionalUtilityCache,
) -> dict[str, Any]:
    """Verify that the private cache is linked to an aggregate open-role OOF audit."""

    lineage_path = Path(path)
    if lineage_path.suffix.lower() != ".json":
        raise UtilityModelContractError("OOF lineage report must be a JSON file")
    try:
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UtilityModelContractError(
            f"cannot read OOF lineage report: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise UtilityModelContractError("OOF lineage report root must be a mapping")
    if payload.get("protocol") != "bidirectional_emotion_utility_v1":
        raise UtilityModelContractError("OOF lineage protocol changed")
    if payload.get("status") != (
        "train_only_different_set_oof_supervision_complete; utility_model_not_yet_selected"
    ):
        raise UtilityModelContractError(
            "OOF lineage status is not the frozen open-role run"
        )

    cache_contract = payload.get("cache_contract")
    task_counts = payload.get("task_counts")
    hashes = payload.get("hashes")
    role_audit = payload.get("sealed_audit")
    if not all(
        isinstance(value, Mapping)
        for value in (cache_contract, task_counts, hashes, role_audit)
    ):
        raise UtilityModelContractError(
            "OOF lineage report lacks required audit mappings"
        )
    if cache_contract.get("schema") != CACHE_SCHEMA_VERSION:
        raise UtilityModelContractError("OOF lineage cache schema differs")
    if cache_contract.get("numeric_dtype") != "float64":
        raise UtilityModelContractError("OOF lineage cache dtype differs")
    if cache_contract.get("contains_gold_labels") is not False:
        raise UtilityModelContractError("OOF lineage cache may contain gold labels")
    if cache_contract.get("contains_row_identifiers") is not False:
        raise UtilityModelContractError("OOF lineage cache may contain row identifiers")

    expected_counts = {
        "fit_oof": len(cache.fit.x),
        "model_selection": len(cache.selection.x),
        "fit_groups": len(np.unique(cache.fit.cluster_codes)),
        "model_selection_groups": len(np.unique(cache.selection.cluster_codes)),
    }
    if any(
        int(task_counts.get(key, -1)) != value for key, value in expected_counts.items()
    ):
        raise UtilityModelContractError(
            "OOF lineage task or cluster counts differ from cache"
        )
    for key, expected in cache.source_hashes.items():
        if _validated_sha256(str(hashes.get(key, "")), key=key) != expected:
            raise UtilityModelContractError(f"OOF lineage {key} differs from cache")

    expected_role_audit = {
        "calibration_rows_used_for_training_or_metrics": 0,
        "internal_holdout_rows_used_for_training_or_metrics": 0,
        "row_level_output_emitted": False,
        "test_rows_used": 0,
        "validation_rows_used": 0,
    }
    if any(role_audit.get(key) != value for key, value in expected_role_audit.items()):
        raise UtilityModelContractError("OOF lineage reports restricted-role access")
    lineage_sha256 = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    return {
        "lineage_report_sha256": lineage_sha256,
        "upstream_role_exclusion_verified": True,
        "upstream_row_level_output_emitted": False,
        "upstream_protocol": str(payload["protocol"]),
    }


def trainable_parameter_count(
    input_dimension: int,
    hidden_layer_sizes: Sequence[int],
    output_dimension: int,
) -> int:
    dimensions = [
        int(input_dimension),
        *(int(width) for width in hidden_layer_sizes),
        int(output_dimension),
    ]
    if any(width < 1 for width in dimensions):
        raise UtilityModelContractError("all network dimensions must be positive")
    return int(
        sum(
            left * right + right for left, right in zip(dimensions[:-1], dimensions[1:])
        )
    )


def _training_targets(split: UtilitySplit, mode: str) -> np.ndarray:
    if mode == "forward_only":
        return split.forward[:, None]
    if mode == "backward_only":
        return split.backward[:, None]
    if mode == "pseudo_bidirectional_shared":
        # Degenerate same-set negative control: if deletion is evaluated on
        # exactly S union {h_i}, its algebraic target duplicates u_plus.  The
        # two heads and parameter count match the true bidirectional model,
        # but no independent reverse-set information is available.
        return np.column_stack([split.forward, split.forward])
    if mode == "bidirectional_shared":
        return np.column_stack([split.forward, split.backward])
    raise UtilityModelContractError(f"unknown model mode: {mode}")


def fit_utility_model(
    split: UtilitySplit,
    spec: UtilityModelSpec,
    *,
    seed: int,
) -> FittedUtilityModel:
    """Fit a reproducible MLP; the two-output variant shares every hidden layer."""

    output_dimension = 2 if spec.mode in TWO_HEAD_MODES else 1
    parameters = trainable_parameter_count(
        split.x.shape[1], spec.hidden_layer_sizes, output_dimension
    )
    if parameters >= MAX_TRAINABLE_PARAMETERS:
        raise UtilityModelContractError(
            f"model has {parameters:,} parameters; limit is <{MAX_TRAINABLE_PARAMETERS:,}"
        )
    x_scaler = StandardScaler().fit(split.x)
    target = _training_targets(split, spec.mode)
    target_mean = target.mean(axis=0)
    target_scale = target.std(axis=0)
    target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
    standardized_target = (target - target_mean[None, :]) / target_scale[None, :]
    early_stopping = bool(spec.early_stopping and spec.solver != "lbfgs")
    optimizer_rows = (
        max(1, int(np.floor(len(split.x) * (1.0 - spec.validation_fraction))))
        if early_stopping
        else len(split.x)
    )
    estimator = MLPRegressor(
        hidden_layer_sizes=tuple(int(width) for width in spec.hidden_layer_sizes),
        activation=spec.activation,
        solver=spec.solver,
        alpha=float(spec.alpha),
        max_iter=int(spec.max_iter),
        tol=float(spec.tolerance),
        random_state=int(seed),
        shuffle=True,
        batch_size=min(int(spec.batch_size), optimizer_rows),
        learning_rate_init=float(spec.learning_rate_init),
        early_stopping=early_stopping,
        validation_fraction=float(spec.validation_fraction),
        n_iter_no_change=int(spec.n_iter_no_change),
    )
    fit_target: np.ndarray = (
        standardized_target[:, 0] if output_dimension == 1 else standardized_target
    )
    estimator.fit(x_scaler.transform(split.x), fit_target)
    actual_parameters = int(
        sum(array.size for array in estimator.coefs_)
        + sum(array.size for array in estimator.intercepts_)
    )
    if actual_parameters != parameters:
        raise AssertionError("MLP parameter count differs from declared architecture")
    if spec.mode in TWO_HEAD_MODES and int(estimator.n_outputs_) != 2:
        raise AssertionError("shared MLP did not fit a two-output representation")
    return FittedUtilityModel(
        spec=spec,
        seed=int(seed),
        x_scaler=x_scaler,
        target_mean=target_mean,
        target_scale=target_scale,
        estimator=estimator,
        parameter_count=parameters,
    )


def group_oof_predictions(
    split: UtilitySplit,
    spec: UtilityModelSpec,
    *,
    seed: int,
    maximum_splits: int = 5,
) -> OOFPredictions:
    """Generate leakage-closed OOF predictions with whole clusters held out."""

    unique_clusters = np.unique(split.cluster_codes)
    n_splits = min(int(maximum_splits), len(unique_clusters))
    if n_splits < 2:
        raise UtilityModelContractError("group OOF requires at least two folds")
    fold_by_row = np.full(len(split.x), -1, dtype=np.int32)
    forward = None if spec.mode == "backward_only" else np.full(len(split.x), np.nan)
    backward = None if spec.mode == "forward_only" else np.full(len(split.x), np.nan)
    decision = np.full(len(split.x), np.nan)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, held_index) in enumerate(
        splitter.split(split.x, groups=split.cluster_codes)
    ):
        train_clusters = set(int(value) for value in split.cluster_codes[train_index])
        held_clusters = set(int(value) for value in split.cluster_codes[held_index])
        if train_clusters & held_clusters:
            raise AssertionError(
                "GroupKFold leaked a cluster across train and held rows"
            )
        train_split = UtilitySplit.validated(
            split.x[train_index],
            split.forward[train_index],
            split.backward[train_index],
            split.cluster_codes[train_index],
            label=f"OOF fold {fold} train",
            require_multiple_clusters=False,
        )
        model = fit_utility_model(train_split, spec, seed=int(seed) + fold * 10_007)
        held = model.predict(split.x[held_index])
        fold_by_row[held_index] = fold
        decision[held_index] = held.decision_score
        if forward is not None and held.forward is not None:
            forward[held_index] = held.forward
        if backward is not None and held.backward is not None:
            backward[held_index] = held.backward
    arrays = [decision]
    arrays.extend(value for value in (forward, backward) if value is not None)
    if np.any(fold_by_row < 0) or any(not np.isfinite(value).all() for value in arrays):
        raise AssertionError("OOF prediction coverage is incomplete")
    # Every cluster must map to exactly one held fold.
    for cluster in unique_clusters:
        if len(np.unique(fold_by_row[split.cluster_codes == cluster])) != 1:
            raise AssertionError("a cluster was split across OOF folds")
    return OOFPredictions(
        UtilityPredictions(forward, backward, decision), fold_by_row, n_splits
    )


def _group_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.mean(values[groups == group]) for group in np.unique(groups)],
        dtype=np.float64,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | None]:
    error = np.asarray(prediction) - np.asarray(target)
    group_mae = _group_means(np.abs(error), groups)
    group_mse = _group_means(np.square(error), groups)
    return {
        "row_mae": float(np.mean(np.abs(error))),
        "row_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "cluster_macro_mae": float(np.mean(group_mae)),
        "cluster_macro_rmse": float(np.mean(np.sqrt(group_mse))),
        "pearson": _safe_correlation(np.asarray(target), np.asarray(prediction)),
    }


def benefit_positive_policy_metrics(
    strict_utility: np.ndarray,
    decision_score: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float = 0.0,
) -> dict[str, float | int | None]:
    """Evaluate select-vs-fallback with a zero-utility current-only fallback."""

    actual = np.asarray(strict_utility, dtype=np.float64)
    score = np.asarray(decision_score, dtype=np.float64)
    cluster_codes = np.asarray(groups)
    if actual.shape != score.shape or actual.shape != cluster_codes.shape:
        raise UtilityModelContractError("policy arrays must be row-aligned")
    if not np.isfinite(actual).all() or not np.isfinite(score).all():
        raise UtilityModelContractError("policy arrays contain non-finite values")
    selected = score > float(threshold)
    return _benefit_positive_policy_metrics_from_selection(
        actual,
        selected,
        cluster_codes,
        threshold=float(threshold),
    )


def _benefit_positive_policy_metrics_from_selection(
    strict_utility: np.ndarray,
    selected: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float | None,
) -> dict[str, float | int | None]:
    """Aggregate a fixed selection without serializing its row-level mask."""

    actual = np.asarray(strict_utility, dtype=np.float64)
    selected_array = np.asarray(selected)
    cluster_codes = np.asarray(groups)
    if (
        actual.ndim != 1
        or actual.shape != selected_array.shape
        or actual.shape != cluster_codes.shape
        or selected_array.dtype != np.bool_
    ):
        raise UtilityModelContractError(
            "fixed policy selection must be a row-aligned boolean array"
        )
    if not np.isfinite(actual).all():
        raise UtilityModelContractError("policy utilities contain non-finite values")
    selected = selected_array
    positive = actual > 0.0
    negative = actual < 0.0
    policy_utility = np.where(selected, actual, 0.0)
    excess_nll_vs_fallback = -policy_utility
    oracle_utility = np.maximum(actual, 0.0)
    oracle_opportunity_regret = oracle_utility - policy_utility

    selected_count = int(selected.sum())
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    return {
        "threshold": None if threshold is None else float(threshold),
        "selected_rows": selected_count,
        "coverage": float(selected.mean()),
        "positive_utility_selection_rate": (
            float(selected[positive].mean()) if positive_count else None
        ),
        "negative_utility_fallback_rate": (
            float((~selected[negative]).mean()) if negative_count else None
        ),
        "selected_positive_precision": (
            float(positive[selected].mean()) if selected_count else None
        ),
        "mean_policy_utility": float(policy_utility.mean()),
        "cluster_macro_policy_utility": float(
            _group_means(policy_utility, cluster_codes).mean()
        ),
        "mean_excess_nll_vs_fallback": float(excess_nll_vs_fallback.mean()),
        "cluster_macro_excess_nll_vs_fallback": float(
            _group_means(excess_nll_vs_fallback, cluster_codes).mean()
        ),
        "mean_oracle_opportunity_regret": float(oracle_opportunity_regret.mean()),
        "cluster_macro_oracle_opportunity_regret": float(
            _group_means(oracle_opportunity_regret, cluster_codes).mean()
        ),
    }


def exact_rank_coverage_selection(
    decision_score: np.ndarray,
    coverage: float = PRIMARY_HISTORY_COVERAGE,
) -> np.ndarray:
    """Select an exact score-ranked fraction without consulting utility labels.

    This helper is intentionally transductive: the complete open-role
    model-selection score vector fixes the top-k operating point.  It is a
    diagnostic for coverage drift, never a deployable threshold.  Stable input
    order breaks exact score ties, using neither labels, utilities, cluster
    membership, nor row identifiers.
    """

    score = np.asarray(decision_score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise UtilityModelContractError(
            "exact-rank coverage scores must be finite and one-dimensional"
        )
    if not 0.0 < float(coverage) < 1.0:
        raise UtilityModelContractError(
            "exact-rank coverage must lie strictly between zero and one"
        )
    requested_count = float(coverage) * len(score)
    target_count = int(np.rint(requested_count))
    if not np.isclose(requested_count, target_count, rtol=0.0, atol=1e-12):
        raise UtilityModelContractError(
            "exact-rank coverage requires an integral selected-row count"
        )
    if target_count < 1 or target_count >= len(score):
        raise UtilityModelContractError(
            "exact-rank coverage must select between one and rows-1 observations"
        )
    ranked = np.argsort(-score, kind="stable")
    selected = np.zeros(len(score), dtype=bool)
    selected[ranked[:target_count]] = True
    return selected


def exact_rank_coverage_diagnostic(
    split: UtilitySplit,
    predictions: UtilityPredictions,
    *,
    coverage: float = PRIMARY_HISTORY_COVERAGE,
) -> dict[str, Any]:
    """Evaluate a label-blind exact-coverage rank rule on an open-role split."""

    score = np.asarray(predictions.decision_score, dtype=np.float64)
    if score.shape != split.strict_bidirectional_utility.shape:
        raise UtilityModelContractError(
            "exact-rank diagnostic scores must align with the utility split"
        )
    selected = exact_rank_coverage_selection(score, coverage)
    ranked_scores = np.sort(score)[::-1]
    target_count = int(selected.sum())
    boundary_score = float(ranked_scores[target_count - 1])
    return {
        "role": "transductive_diagnostic_only_not_deployable_or_used_for_ranking",
        "selection_scope": "complete_open_role_model_selection_score_vector",
        "selection_inputs": "decision_score_only_no_labels_utilities_clusters_or_identifiers",
        "tie_rule": "stable_input_order_only_within_exact_score_ties",
        "selection_uses_labels_or_utilities": False,
        "evaluation_uses_open_role_utility_after_selection": True,
        "target_coverage": float(coverage),
        "selected_rows": target_count,
        "realized_coverage": float(selected.mean()),
        "boundary_score": boundary_score,
        "boundary_tie_count": int(np.sum(score == boundary_score)),
        "policy": _benefit_positive_policy_metrics_from_selection(
            split.strict_bidirectional_utility,
            selected,
            split.cluster_codes,
            threshold=None,
        ),
    }


def fit_oof_coverage_threshold(decision_score: np.ndarray, coverage: float) -> float:
    """Freeze a deterministic threshold on fit-OOF scores only."""

    score = np.asarray(decision_score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise UtilityModelContractError(
            "coverage threshold scores must be finite and one-dimensional"
        )
    if not 0.0 < float(coverage) < 1.0:
        raise UtilityModelContractError(
            "primary coverage must lie strictly between zero and one"
        )
    target_count = int(np.rint(float(coverage) * len(score)))
    target_count = min(max(target_count, 1), len(score) - 1)
    ordered = np.sort(score)[::-1]
    selected_boundary = float(ordered[target_count - 1])
    excluded_boundary = float(ordered[target_count])
    if selected_boundary > excluded_boundary:
        return selected_boundary + (excluded_boundary - selected_boundary) / 2.0
    # A score-only threshold cannot split an exact tie without row identity.
    # Strict > makes all boundary ties fall back, which is the conservative and
    # deterministic rule; realized coverage and deviation are always reported.
    return selected_boundary


def _cluster_excess_nll_vs_fallback(
    split: UtilitySplit,
    predictions: UtilityPredictions,
    threshold: float,
) -> np.ndarray:
    """Return current-only/fallback excess NLL per cluster in code order."""

    selected = predictions.selected(threshold)
    return _cluster_excess_nll_for_selection(split, selected)


def _cluster_excess_nll_for_selection(
    split: UtilitySplit,
    selected: np.ndarray,
) -> np.ndarray:
    """Return per-cluster excess NLL for a fixed, private row selection."""

    selected_array = np.asarray(selected)
    if selected_array.shape != split.strict_bidirectional_utility.shape or (
        selected_array.dtype != np.bool_
    ):
        raise UtilityModelContractError(
            "cluster excess-NLL selection must be a row-aligned boolean array"
        )
    policy_utility = np.where(
        selected_array,
        split.strict_bidirectional_utility,
        0.0,
    )
    return _group_means(-policy_utility, split.cluster_codes)


def _cluster_strict_rmse(
    split: UtilitySplit,
    predictions: UtilityPredictions,
) -> np.ndarray:
    """Return strict-utility RMSE per cluster in deterministic code order."""

    squared_error = np.square(
        split.strict_bidirectional_utility
        - np.asarray(predictions.decision_score, dtype=np.float64)
    )
    return np.sqrt(_group_means(squared_error, split.cluster_codes))


def paired_seed_cluster_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Legacy paired nested bootstrap over seeds and independent clusters.

    Inputs must be ``(training_seeds, clusters)`` metric matrices aligned on
    both axes.  A negative candidate-minus-reference difference is favorable.
    Each resampled seed slot receives a separate cluster resample.  This design
    is retained only as the v2 legacy sensitivity; v3 primary open-role
    sensitivity uses :func:`paired_seed_shared_cluster_bootstrap`.
    """

    difference = _validated_paired_difference(candidate, reference, replicates)
    rng = np.random.default_rng(int(seed))
    seed_count, cluster_count = difference.shape
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    # Chunking bounds peak memory for larger future datasets while preserving
    # the exact deterministic RNG stream and paired resampling contract.
    for start in range(0, int(replicates), 1_000):
        stop = min(start + 1_000, int(replicates))
        size = stop - start
        seed_index = rng.integers(0, seed_count, size=(size, seed_count))
        # Each resampled training-seed slot receives its own whole-cluster
        # resample, matching the frozen seed -> cluster hierarchy.
        cluster_index = rng.integers(
            0,
            cluster_count,
            size=(size, seed_count, cluster_count),
        )
        sampled = difference[
            seed_index[:, :, None],
            cluster_index,
        ]
        bootstrap[start:stop] = sampled.mean(axis=(1, 2))
    return _paired_bootstrap_summary(
        difference,
        bootstrap,
        replicates=int(replicates),
        seed=int(seed),
        design="nested_seed_then_independent_cluster_resampling",
        inferential_role="legacy_v2_sensitivity_only",
        cluster_resampling=(
            "independent_whole_cluster_index_vector_for_each_resampled_seed_slot"
        ),
    )


def paired_seed_shared_cluster_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Crossed paired bootstrap with one shared cluster draw per replicate.

    Inputs are aligned ``(training_seeds, clusters)`` metric matrices.  Each
    replicate first resamples training seeds and then draws exactly one vector
    of whole-cluster indices.  That same cluster vector is applied to every
    resampled seed slot, preserving the crossed/shared-cluster pairing.
    """

    difference = _validated_paired_difference(candidate, reference, replicates)
    rng = np.random.default_rng(int(seed))
    seed_count, cluster_count = difference.shape
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 1_000):
        stop = min(start + 1_000, int(replicates))
        size = stop - start
        seed_index = rng.integers(0, seed_count, size=(size, seed_count))
        # One cluster resample is shared by every sampled seed slot.  Keeping
        # the axes separate makes the crossed design explicit and auditable.
        cluster_index = rng.integers(0, cluster_count, size=(size, cluster_count))
        sampled = difference[
            seed_index[:, :, None],
            cluster_index[:, None, :],
        ]
        bootstrap[start:stop] = sampled.mean(axis=(1, 2))
    return _paired_bootstrap_summary(
        difference,
        bootstrap,
        replicates=int(replicates),
        seed=int(seed),
        design="crossed_seed_with_shared_cluster_resampling",
        inferential_role="primary_open_role_sensitivity",
        cluster_resampling=(
            "one_shared_whole_cluster_index_vector_across_all_resampled_seed_slots"
        ),
    )


def _validated_paired_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    replicates: int,
) -> np.ndarray:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        candidate_array.ndim != 2
        or candidate_array.shape != reference_array.shape
        or candidate_array.shape[0] < 2
        or candidate_array.shape[1] < 2
    ):
        raise UtilityModelContractError(
            "paired bootstrap inputs must be aligned (seeds>=2, clusters>=2) matrices"
        )
    if not np.isfinite(candidate_array).all() or not np.isfinite(reference_array).all():
        raise UtilityModelContractError(
            "paired bootstrap inputs contain non-finite values"
        )
    if int(replicates) < 100:
        raise UtilityModelContractError(
            "paired bootstrap requires at least 100 replicates"
        )
    return candidate_array - reference_array


def _paired_bootstrap_summary(
    difference: np.ndarray,
    bootstrap: np.ndarray,
    *,
    replicates: int,
    seed: int,
    design: str,
    inferential_role: str,
    cluster_resampling: str,
) -> dict[str, Any]:
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    seed_count, cluster_count = difference.shape
    return {
        "bootstrap_design": design,
        "inferential_role": inferential_role,
        "axis_contract": "aligned_training_seed_by_cluster_matrices",
        "pairing": "candidate_and_reference_paired_on_both_axes",
        "training_seed_resampling": "sample_training_seed_indices_with_replacement",
        "cluster_resampling": cluster_resampling,
        "difference_definition": "candidate_minus_reference",
        "direction": "lower_is_better",
        "five_seed_cluster_macro_difference": float(difference.mean()),
        "ci95_percentile": [float(low), float(high)],
        "ci95_upper_below_zero": bool(high < 0.0),
        "bootstrap_probability_difference_below_zero": float(np.mean(bootstrap < 0.0)),
        "replicates": replicates,
        "bootstrap_seed": seed,
        "training_seed_count": int(seed_count),
        "cluster_count": int(cluster_count),
    }


def _paired_bootstrap_designs(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Return clearly separated primary crossed and legacy nested designs."""

    return {
        "primary_crossed_seed_shared_cluster": paired_seed_shared_cluster_bootstrap(
            candidate,
            reference,
            replicates=replicates,
            seed=seed,
        ),
        "legacy_nested_seed_independent_cluster_sensitivity": (
            paired_seed_cluster_bootstrap(
                candidate,
                reference,
                replicates=replicates,
                seed=seed,
            )
        ),
    }


def evaluate_predictions(
    split: UtilitySplit,
    predictions: UtilityPredictions,
    *,
    policy_threshold: float = 0.0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "strict_bidirectional": _regression_metrics(
            split.strict_bidirectional_utility,
            predictions.decision_score,
            split.cluster_codes,
        ),
        "policy": benefit_positive_policy_metrics(
            split.strict_bidirectional_utility,
            predictions.decision_score,
            split.cluster_codes,
            threshold=float(policy_threshold),
        ),
    }
    if predictions.forward is not None:
        metrics["forward"] = _regression_metrics(
            split.forward, predictions.forward, split.cluster_codes
        )
    if predictions.backward is not None:
        metrics["backward"] = _regression_metrics(
            split.backward, predictions.backward, split.cluster_codes
        )
    return metrics


def _ensemble_predictions(
    predictions: Sequence[UtilityPredictions],
) -> UtilityPredictions:
    if not predictions:
        raise UtilityModelContractError("cannot ensemble zero predictions")
    forward_values = [value.forward for value in predictions]
    backward_values = [value.backward for value in predictions]
    forward = (
        None
        if any(value is None for value in forward_values)
        else np.mean(np.stack(forward_values), axis=0)
    )
    backward = (
        None
        if any(value is None for value in backward_values)
        else np.mean(np.stack(backward_values), axis=0)
    )
    if forward is not None and backward is not None:
        # Aggregate directional heads first, then apply the registered strict rule.
        decision = np.minimum(forward, backward)
    elif forward is not None:
        decision = forward.copy()
    elif backward is not None:
        decision = backward.copy()
    else:
        raise AssertionError("utility ensemble has no output")
    return UtilityPredictions(forward, backward, decision)


def _seed_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    excess_nll = np.asarray(
        [
            record["model_selection"]["policy"]["cluster_macro_excess_nll_vs_fallback"]
            for record in records
        ],
        dtype=np.float64,
    )
    rmse = np.asarray(
        [
            record["model_selection"]["strict_bidirectional"]["cluster_macro_rmse"]
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "selection_cluster_macro_excess_nll_mean": float(excess_nll.mean()),
        "selection_cluster_macro_excess_nll_std": float(excess_nll.std(ddof=1)),
        "selection_cluster_macro_rmse_mean": float(rmse.mean()),
        "selection_cluster_macro_rmse_std": float(rmse.std(ddof=1)),
    }


def _validate_five_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(seed) for seed in seeds)
    if len(result) != 5 or len(set(result)) != 5:
        raise UtilityModelContractError(
            "exactly five distinct model seeds are required"
        )
    return result


def run_five_seed_model_selection(
    cache: BidirectionalUtilityCache,
    *,
    specs: Sequence[UtilityModelSpec] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    maximum_oof_splits: int = 5,
    paired_bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    enforce_registered_contract: bool = True,
) -> dict[str, Any]:
    """Compare four utility models using fit OOF and disjoint selection metrics."""

    model_specs = tuple(default_model_specs() if specs is None else specs)
    if not model_specs or len({spec.name for spec in model_specs}) != len(model_specs):
        raise UtilityModelContractError(
            "model specs must be non-empty with unique names"
        )
    model_seeds = _validate_five_seeds(seeds)
    if enforce_registered_contract:
        registered = tuple((spec.name, spec.mode) for spec in default_model_specs())
        received = tuple((spec.name, spec.mode) for spec in model_specs)
        if received != registered:
            raise UtilityModelContractError(
                "registered run requires all four prespecified model names and modes"
            )
        if model_seeds != DEFAULT_SEEDS:
            raise UtilityModelContractError(
                f"registered run seeds must equal {DEFAULT_SEEDS}"
            )
        if int(paired_bootstrap_replicates) != PAIRED_BOOTSTRAP_REPLICATES:
            raise UtilityModelContractError(
                "registered run requires exactly 10000 paired bootstrap replicates"
            )
    reports: list[dict[str, Any]] = []
    private_selection: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        oof_by_seed: list[UtilityPredictions] = []
        selection_by_seed: list[UtilityPredictions] = []
        seed_records: list[dict[str, Any]] = []
        parameter_count = trainable_parameter_count(
            cache.fit.x.shape[1],
            spec.hidden_layer_sizes,
            2 if spec.mode in TWO_HEAD_MODES else 1,
        )
        if parameter_count >= MAX_TRAINABLE_PARAMETERS:
            raise UtilityModelContractError(
                f"model {spec.name} exceeds the <{MAX_TRAINABLE_PARAMETERS:,} parameter contract"
            )
        for seed in model_seeds:
            oof = group_oof_predictions(
                cache.fit,
                spec,
                seed=seed,
                maximum_splits=maximum_oof_splits,
            )
            fitted = fit_utility_model(cache.fit, spec, seed=seed)
            selection_prediction = fitted.predict(cache.selection.x)
            policy_threshold = fit_oof_coverage_threshold(
                oof.predictions.decision_score, PRIMARY_HISTORY_COVERAGE
            )
            oof_by_seed.append(oof.predictions)
            selection_by_seed.append(selection_prediction)
            seed_records.append(
                {
                    "seed": seed,
                    "oof_folds": oof.n_splits,
                    "final_fit_iterations": int(fitted.estimator.n_iter_),
                    "final_fit_loss": float(fitted.estimator.loss_),
                    "final_fit_stopped_before_limit": bool(
                        fitted.estimator.n_iter_ < spec.max_iter
                    ),
                    "fit_oof_threshold": float(policy_threshold),
                    "fit_oof": evaluate_predictions(
                        cache.fit,
                        oof.predictions,
                        policy_threshold=policy_threshold,
                    ),
                    "model_selection": evaluate_predictions(
                        cache.selection,
                        selection_prediction,
                        policy_threshold=policy_threshold,
                    ),
                    "model_selection_exact_25pct_transductive_diagnostic": (
                        exact_rank_coverage_diagnostic(
                            cache.selection,
                            selection_prediction,
                            coverage=PRIMARY_HISTORY_COVERAGE,
                        )
                    ),
                    "zero_threshold_diagnostic": {
                        "fit_oof": evaluate_predictions(cache.fit, oof.predictions),
                        "model_selection": evaluate_predictions(
                            cache.selection, selection_prediction
                        ),
                    },
                }
            )
        oof_ensemble = _ensemble_predictions(oof_by_seed)
        selection_ensemble = _ensemble_predictions(selection_by_seed)
        ensemble_threshold = fit_oof_coverage_threshold(
            oof_ensemble.decision_score, PRIMARY_HISTORY_COVERAGE
        )
        private_selection[spec.name] = {
            "ensemble": selection_ensemble,
            "ensemble_threshold": float(ensemble_threshold),
            "per_seed": tuple(selection_by_seed),
            "per_seed_thresholds": tuple(
                float(record["fit_oof_threshold"]) for record in seed_records
            ),
            "per_seed_exact_25pct_selections": tuple(
                exact_rank_coverage_selection(
                    prediction.decision_score,
                    PRIMARY_HISTORY_COVERAGE,
                )
                for prediction in selection_by_seed
            ),
        }
        target_semantics = {
            "forward_only": "different_set_forward_addition_only",
            "backward_only": "different_set_backward_deletion_only",
            "pseudo_bidirectional_shared": "degenerate_same_set_duplicated_forward_target",
            "bidirectional_shared": "independent_different_set_forward_and_backward_targets",
        }[spec.mode]
        reports.append(
            {
                "name": spec.name,
                "mode": spec.mode,
                "target_semantics": target_semantics,
                "shared_hidden_representation": spec.mode in TWO_HEAD_MODES,
                "output_heads": 2 if spec.mode in TWO_HEAD_MODES else 1,
                "decision_rule": (
                    "top_25pct_fit_oof_min(predicted_forward,predicted_backward)"
                    if spec.mode == "bidirectional_shared"
                    else (
                        "top_25pct_fit_oof_min(duplicated_forward_head_1,duplicated_forward_head_2)"
                        if spec.mode == "pseudo_bidirectional_shared"
                        else f"top_25pct_fit_oof_predicted_{spec.mode.removesuffix('_only')}"
                    )
                ),
                "architecture": {
                    "hidden_layer_sizes": list(spec.hidden_layer_sizes),
                    "activation": spec.activation,
                    "solver": spec.solver,
                    "alpha": float(spec.alpha),
                    "max_iter": int(spec.max_iter),
                    "tolerance": float(spec.tolerance),
                    "batch_size": int(spec.batch_size),
                    "learning_rate_init": float(spec.learning_rate_init),
                    "early_stopping": bool(spec.early_stopping),
                    "early_stop_fraction": float(spec.validation_fraction),
                    "n_iter_no_change": int(spec.n_iter_no_change),
                    "parameter_count": parameter_count,
                    "parameter_limit_exclusive": MAX_TRAINABLE_PARAMETERS,
                },
                "per_seed": seed_records,
                "across_seed_summary": _seed_summary(seed_records),
                "five_seed_ensemble": {
                    "fit_oof_threshold": float(ensemble_threshold),
                    "target_fit_oof_coverage": PRIMARY_HISTORY_COVERAGE,
                    "realized_fit_oof_coverage": float(
                        oof_ensemble.selected(ensemble_threshold).mean()
                    ),
                    "fit_oof_coverage_deviation": float(
                        oof_ensemble.selected(ensemble_threshold).mean()
                        - PRIMARY_HISTORY_COVERAGE
                    ),
                    "fit_oof": evaluate_predictions(
                        cache.fit,
                        oof_ensemble,
                        policy_threshold=ensemble_threshold,
                    ),
                    "model_selection": evaluate_predictions(
                        cache.selection,
                        selection_ensemble,
                        policy_threshold=ensemble_threshold,
                    ),
                    "model_selection_exact_25pct_transductive_diagnostic": (
                        exact_rank_coverage_diagnostic(
                            cache.selection,
                            selection_ensemble,
                            coverage=PRIMARY_HISTORY_COVERAGE,
                        )
                    ),
                    "zero_threshold_diagnostic": {
                        "fit_oof": evaluate_predictions(cache.fit, oof_ensemble),
                        "model_selection": evaluate_predictions(
                            cache.selection, selection_ensemble
                        ),
                    },
                },
            }
        )

    ranked = sorted(
        reports,
        key=lambda report: (
            report["across_seed_summary"]["selection_cluster_macro_excess_nll_mean"],
            report["across_seed_summary"]["selection_cluster_macro_rmse_mean"],
            report["name"],
        ),
    )
    ranking = [
        {
            "rank": index + 1,
            "name": report["name"],
            "five_seed_mean_cluster_macro_excess_nll": report["across_seed_summary"][
                "selection_cluster_macro_excess_nll_mean"
            ],
            "five_seed_mean_cluster_macro_strict_utility_rmse": report[
                "across_seed_summary"
            ]["selection_cluster_macro_rmse_mean"],
        }
        for index, report in enumerate(ranked)
    ]
    paired_contrasts: list[dict[str, Any]] = []
    true_name = "bidirectional_shared_mlp"
    if true_name in private_selection:
        candidate_private = private_selection[true_name]
        candidate_report = next(
            report for report in reports if report["name"] == true_name
        )
        candidate_excess_nll = np.stack(
            [
                _cluster_excess_nll_vs_fallback(cache.selection, prediction, threshold)
                for prediction, threshold in zip(
                    candidate_private["per_seed"],
                    candidate_private["per_seed_thresholds"],
                    strict=True,
                )
            ]
        )
        candidate_exact_excess_nll = np.stack(
            [
                _cluster_excess_nll_for_selection(cache.selection, selected)
                for selected in candidate_private["per_seed_exact_25pct_selections"]
            ]
        )
        candidate_rmse = np.stack(
            [
                _cluster_strict_rmse(cache.selection, prediction)
                for prediction in candidate_private["per_seed"]
            ]
        )
        for reference_name in (
            "forward_only_mlp",
            "backward_only_mlp",
            "pseudo_bidirectional_same_set_mlp",
        ):
            if reference_name not in private_selection:
                continue
            reference_private = private_selection[reference_name]
            reference_report = next(
                report for report in reports if report["name"] == reference_name
            )
            reference_excess_nll = np.stack(
                [
                    _cluster_excess_nll_vs_fallback(
                        cache.selection, prediction, threshold
                    )
                    for prediction, threshold in zip(
                        reference_private["per_seed"],
                        reference_private["per_seed_thresholds"],
                        strict=True,
                    )
                ]
            )
            reference_exact_excess_nll = np.stack(
                [
                    _cluster_excess_nll_for_selection(cache.selection, selected)
                    for selected in reference_private["per_seed_exact_25pct_selections"]
                ]
            )
            reference_rmse = np.stack(
                [
                    _cluster_strict_rmse(cache.selection, prediction)
                    for prediction in reference_private["per_seed"]
                ]
            )
            candidate_seed_regret = np.asarray(
                [
                    record["model_selection"]["policy"][
                        "cluster_macro_excess_nll_vs_fallback"
                    ]
                    for record in candidate_report["per_seed"]
                ],
                dtype=np.float64,
            )
            reference_seed_regret = np.asarray(
                [
                    record["model_selection"]["policy"][
                        "cluster_macro_excess_nll_vs_fallback"
                    ]
                    for record in reference_report["per_seed"]
                ],
                dtype=np.float64,
            )
            candidate_exact_seed_regret = np.asarray(
                [
                    record["model_selection_exact_25pct_transductive_diagnostic"][
                        "policy"
                    ]["cluster_macro_excess_nll_vs_fallback"]
                    for record in candidate_report["per_seed"]
                ],
                dtype=np.float64,
            )
            reference_exact_seed_regret = np.asarray(
                [
                    record["model_selection_exact_25pct_transductive_diagnostic"][
                        "policy"
                    ]["cluster_macro_excess_nll_vs_fallback"]
                    for record in reference_report["per_seed"]
                ],
                dtype=np.float64,
            )
            candidate_ensemble_policy = candidate_report["five_seed_ensemble"][
                "model_selection"
            ]["policy"]
            reference_ensemble_policy = reference_report["five_seed_ensemble"][
                "model_selection"
            ]["policy"]
            candidate_exact_ensemble = candidate_report["five_seed_ensemble"][
                "model_selection_exact_25pct_transductive_diagnostic"
            ]
            reference_exact_ensemble = reference_report["five_seed_ensemble"][
                "model_selection_exact_25pct_transductive_diagnostic"
            ]
            candidate_exact_ensemble_policy = candidate_exact_ensemble["policy"]
            reference_exact_ensemble_policy = reference_exact_ensemble["policy"]
            paired_contrasts.append(
                {
                    "candidate": true_name,
                    "reference": reference_name,
                    "deployment_operating_point": {
                        "role": "primary_deployment_style_point_estimate",
                        "operating_point": (
                            "fit_oof_frozen_top_25pct_threshold_transferred_per_model"
                        ),
                        "selection_scope": "fit_group_oof_only_before_model_selection",
                        "ensemble_excess_nll_difference_diagnostic": float(
                            candidate_ensemble_policy[
                                "cluster_macro_excess_nll_vs_fallback"
                            ]
                            - reference_ensemble_policy[
                                "cluster_macro_excess_nll_vs_fallback"
                            ]
                        ),
                        "ensemble_realized_coverage": {
                            "candidate": float(candidate_ensemble_policy["coverage"]),
                            "reference": float(reference_ensemble_policy["coverage"]),
                            "candidate_minus_reference": float(
                                candidate_ensemble_policy["coverage"]
                                - reference_ensemble_policy["coverage"]
                            ),
                        },
                        "seed_wins_out_of_five": int(
                            np.sum(candidate_seed_regret < reference_seed_regret)
                        ),
                        "paired_cluster_excess_nll_uncertainty": (
                            _paired_bootstrap_designs(
                                candidate_excess_nll,
                                reference_excess_nll,
                                replicates=paired_bootstrap_replicates,
                                seed=PAIRED_BOOTSTRAP_SEED,
                            )
                        ),
                    },
                    "coverage_matched_transductive_diagnostic": {
                        "role": "diagnostic_only_not_deployable_or_used_for_ranking",
                        "purpose": "exclude_realized_coverage_drift_as_contrast_driver",
                        "operating_point": (
                            "model_selection_score_ranked_exact_common_25pct_per_model"
                        ),
                        "selection_scope": (
                            "complete_open_role_model_selection_score_vector"
                        ),
                        "selection_inputs": (
                            "decision_score_only_no_labels_utilities_clusters_or_identifiers"
                        ),
                        "selection_uses_labels_or_utilities": False,
                        "target_coverage": PRIMARY_HISTORY_COVERAGE,
                        "ensemble_selected_rows": {
                            "candidate": int(candidate_exact_ensemble["selected_rows"]),
                            "reference": int(reference_exact_ensemble["selected_rows"]),
                        },
                        "ensemble_realized_coverage": {
                            "candidate": float(
                                candidate_exact_ensemble["realized_coverage"]
                            ),
                            "reference": float(
                                reference_exact_ensemble["realized_coverage"]
                            ),
                            "candidate_minus_reference": float(
                                candidate_exact_ensemble["realized_coverage"]
                                - reference_exact_ensemble["realized_coverage"]
                            ),
                        },
                        "ensemble_excess_nll_difference_diagnostic": float(
                            candidate_exact_ensemble_policy[
                                "cluster_macro_excess_nll_vs_fallback"
                            ]
                            - reference_exact_ensemble_policy[
                                "cluster_macro_excess_nll_vs_fallback"
                            ]
                        ),
                        "seed_wins_out_of_five": int(
                            np.sum(
                                candidate_exact_seed_regret
                                < reference_exact_seed_regret
                            )
                        ),
                        "paired_cluster_excess_nll_uncertainty": (
                            _paired_bootstrap_designs(
                                candidate_exact_excess_nll,
                                reference_exact_excess_nll,
                                replicates=paired_bootstrap_replicates,
                                seed=PAIRED_BOOTSTRAP_SEED + 2,
                            )
                        ),
                    },
                    "strict_utility_regression_diagnostic": {
                        "role": "coverage_independent_diagnostic",
                        "paired_cluster_rmse_uncertainty": (
                            _paired_bootstrap_designs(
                                candidate_rmse,
                                reference_rmse,
                                replicates=paired_bootstrap_replicates,
                                seed=PAIRED_BOOTSTRAP_SEED + 1,
                            )
                        ),
                    },
                }
            )
    output: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "train_only_model_selection_complete; sealed_evaluation_not_run",
        "polarity": "benefit_positive",
        "claim_boundary": (
            "Fit OOF and model-selection metrics do not establish validation/test improvement "
            "or top-conference readiness. This in-memory function verifies only the cache "
            "field schema; the file runner must also validate an upstream role-lineage report."
        ),
        "data_contract": {
            "input_schema": CACHE_SCHEMA_VERSION,
            "fit_rows": int(len(cache.fit.x)),
            "fit_clusters": int(len(np.unique(cache.fit.cluster_codes))),
            "model_selection_rows": int(len(cache.selection.x)),
            "model_selection_clusters": int(
                len(np.unique(cache.selection.cluster_codes))
            ),
            "feature_count": int(cache.fit.x.shape[1]),
            "restricted_role_named_fields_present": False,
            "row_level_output": False,
            "cluster_identifiers_emitted": False,
            "source_hashes": dict(sorted(cache.source_hashes.items())),
        },
        "seed_contract": {
            "count": len(model_seeds),
            "distinct": True,
            "seeds": list(model_seeds),
        },
        "selection_contract": {
            "primary_metric": (
                "five_seed_mean_model_selection.cluster_macro_excess_nll_vs_fallback"
            ),
            "primary_operating_point_role": (
                "deployment_style_fit_oof_frozen_threshold_transfer"
            ),
            "direction": "lower_is_better",
            "tie_breaker": (
                "five_seed_mean_model_selection.cluster_macro_strict_utility_rmse"
            ),
            "fallback_utility": 0.0,
            "target_fit_oof_history_coverage": PRIMARY_HISTORY_COVERAGE,
            "threshold_source": "fit_group_oof_only",
            "threshold_transfer": "frozen_before_model_selection_scoring",
            "threshold_tie_rule": "strict_greater_than; boundary_ties_fall_back",
            "realized_selection_coverage_note": (
                "May differ across models after frozen-threshold transfer; this is not an "
                "exact common 25% model-selection operating point."
            ),
            "coverage_matched_diagnostic": {
                "role": "transductive_diagnostic_only_not_deployable_or_used_for_ranking",
                "purpose": "exclude_realized_coverage_drift_as_contrast_driver",
                "selection_scope": "complete_open_role_model_selection_score_vector",
                "selection_inputs": (
                    "decision_score_only_no_labels_utilities_clusters_or_identifiers"
                ),
                "target_exact_common_coverage": PRIMARY_HISTORY_COVERAGE,
                "ranking_rule": "descending_decision_score_top_k",
                "tie_rule": "stable_input_order_only_within_exact_score_ties",
            },
            "zero_threshold_role": "diagnostic_only_not_used_for_ranking",
        },
        "uncertainty_contract": {
            "input_axis_contract": "aligned_training_seed_by_cluster_matrices",
            "difference_definition": "candidate_minus_reference",
            "direction": "lower_is_better",
            "replicates": int(paired_bootstrap_replicates),
            "primary_open_role_sensitivity": (
                "crossed_seed_with_shared_cluster_resampling"
            ),
            "primary_cluster_draw": (
                "one_shared_whole_cluster_index_vector_across_all_resampled_seed_slots"
            ),
            "legacy_sensitivity_only": (
                "nested_seed_then_independent_cluster_resampling"
            ),
            "legacy_cluster_draw": (
                "independent_whole_cluster_index_vector_for_each_resampled_seed_slot"
            ),
        },
        "models": reports,
        "ranking": ranking,
        "paired_model_contrasts": paired_contrasts,
        "selected_model": ranking[0]["name"],
    }
    assert_aggregate_report(output)
    return output


def assert_aggregate_report(report: Mapping[str, Any]) -> None:
    """Fail closed if a purported public report contains row-level material."""

    forbidden_exact = {
        "predictions",
        "probabilities",
        "scores",
        "utilities",
        "labels",
        "forward_predictions",
        "backward_predictions",
        "decision_scores",
        "selected",
        "mask",
        "selection_mask",
        "selected_indices",
        "ranked_indices",
        "row_order",
        "cluster_codes",
        "cluster_ids",
        "row_ids",
        "query_ids",
    }

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, np.ndarray):
            raise UtilityModelContractError(
                f"aggregate report contains ndarray at {'.'.join(path)}"
            )
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                safe_sealed_audit = name == "sealed_data_read" and child is False
                if name.lower() in forbidden_exact or (
                    _contains_sealed_token(name) and not safe_sealed_audit
                ):
                    raise UtilityModelContractError(
                        f"aggregate report contains forbidden field {'.'.join((*path, name))}"
                    )
                visit(child, (*path, name))
            return
        if isinstance(value, (list, tuple)):
            if len(value) > 20:
                raise UtilityModelContractError(
                    f"aggregate report contains an overlong list at {'.'.join(path)}"
                )
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise UtilityModelContractError(
                f"aggregate report contains non-finite value at {'.'.join(path)}"
            )
        if not isinstance(
            value, (str, int, float, bool, type(None), np.integer, np.floating)
        ):
            raise UtilityModelContractError(
                f"aggregate report contains non-JSON value at {'.'.join(path)}"
            )

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise UtilityModelContractError(
            f"utility report schema must equal {REPORT_SCHEMA_VERSION}"
        )
    if report.get("schema_version") == REPORT_SCHEMA_VERSION:
        expected_top = {
            "schema_version",
            "status",
            "polarity",
            "claim_boundary",
            "data_contract",
            "seed_contract",
            "selection_contract",
            "uncertainty_contract",
            "models",
            "ranking",
            "paired_model_contrasts",
            "selected_model",
        }
        if set(report) != expected_top:
            raise UtilityModelContractError("utility report top-level schema changed")
        models = report.get("models")
        ranking = report.get("ranking")
        contrasts = report.get("paired_model_contrasts")
        if not isinstance(models, Sequence) or len(models) != 4:
            raise UtilityModelContractError(
                "utility report must contain exactly four models"
            )
        if not isinstance(ranking, Sequence) or len(ranking) != 4:
            raise UtilityModelContractError(
                "utility report ranking must contain four models"
            )
        if not isinstance(contrasts, Sequence) or len(contrasts) != 3:
            raise UtilityModelContractError(
                "utility report must contain three paired contrasts"
            )
        if {
            str(model.get("name")) for model in models if isinstance(model, Mapping)
        } != {spec.name for spec in default_model_specs()}:
            raise UtilityModelContractError("utility report model identities changed")
        for key, digest in (
            report.get("data_contract", {}).get("source_hashes", {}).items()
        ):
            _validated_sha256(str(digest), key=str(key))
    visit(report, ())


def write_aggregate_report(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write aggregate-only JSON with strict standard JSON floats."""

    assert_aggregate_report(report)
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise UtilityModelContractError("aggregate output must use a .json suffix")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"aggregate report already exists: {destination}")
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UtilityModelContractError(
                "refusing to overwrite an unreadable existing aggregate report"
            ) from error
        existing_schema = (
            existing.get("schema_version") if isinstance(existing, Mapping) else None
        )
        if existing_schema != REPORT_SCHEMA_VERSION:
            raise UtilityModelContractError(
                "refusing to overwrite an aggregate report from a different schema version"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def run_private_cache_model_selection(
    private_cache_path: str | Path,
    output_path: str | Path,
    *,
    lineage_report_path: str | Path,
    specs: Sequence[UtilityModelSpec] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    maximum_oof_splits: int = 5,
    paired_bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """End-to-end aggregate interface; it has no argument for sealed data."""

    cache = load_private_oof_cache(private_cache_path)
    lineage = validate_oof_lineage_report(lineage_report_path, cache)
    report = run_five_seed_model_selection(
        cache,
        specs=specs,
        seeds=seeds,
        maximum_oof_splits=maximum_oof_splits,
        paired_bootstrap_replicates=paired_bootstrap_replicates,
        enforce_registered_contract=True,
    )
    cache_bytes = Path(private_cache_path).read_bytes()
    implementation_bytes = Path(__file__).read_bytes()
    report["data_contract"]["input_cache_sha256"] = hashlib.sha256(
        cache_bytes
    ).hexdigest()
    report["data_contract"]["implementation_sha256"] = hashlib.sha256(
        implementation_bytes
    ).hexdigest()
    report["data_contract"].update(lineage)
    experiment_root = Path(__file__).resolve().parents[2]
    reproducibility_files = {
        "utility_module_sha256": Path(__file__),
        "runner_sha256": experiment_root
        / "scripts"
        / "run_bidirectional_utility_models.py",
        "confirmatory_config_sha256": (
            experiment_root / "configs" / "carma_confirmatory_analysis_v1.json"
        ),
        "split_manifest_sha256": (
            experiment_root / "configs" / "carma_split_manifest_v1.json"
        ),
    }
    file_hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in reproducibility_files.items()
        if path.exists()
    }
    reproducibility_manifest: dict[str, Any] = {
        "file_hashes": file_hashes,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    reproducibility_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(reproducibility_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report["data_contract"]["reproducibility_manifest"] = reproducibility_manifest
    report["claim_boundary"] = (
        "Open-role fit OOF and model-selection metrics only; the validated upstream lineage "
        "reports zero calibration, internal-holdout, validation, or test rows used. These "
        "metrics do not establish confirmatory classification/safety improvement."
    )
    write_aggregate_report(report, output_path, overwrite=overwrite)
    return report
