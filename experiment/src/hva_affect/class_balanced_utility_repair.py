"""Class-balanced open-role repair for reversible EmotionTalk history selection.

This module is deliberately separate from ``emotiontalk_query_policy_runner``
so the completed V2 mean-utility result remains immutable.  It changes only
the utility learner: labels are uniquely reconstructed from verified fit-OOF
probabilities and cached directional utilities, class frequencies are counted
once per fit query, and rare classes receive more training mass through a
deterministic weighted resample.  Gold labels are never accepted by the fitted
model's prediction interface and are not appended to the 59-D feature vector.

Every OOF fold derives its balance profile from that fold's training queries.
The final utility model derives a fresh profile from all fit queries.  The
model-selection role is prediction and scoring only.  Calibration, holdout,
validation, and test are not runner inputs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .bidirectional_utility_model import (
    DEFAULT_SEEDS,
    MAX_TRAINABLE_PARAMETERS,
    PRIMARY_HISTORY_COVERAGE,
    BidirectionalUtilityCache,
    UtilityModelSpec,
    UtilityPredictions,
    UtilitySplit,
    load_private_oof_cache,
    trainable_parameter_count,
)
from .data_contract import ContractError, write_json_atomic
from .emotiontalk_bidirectional_oof import (
    _read_json,
    _validate_config,
    probability_task_features,
)
from .emotiontalk_query_policy_runner import (
    FORBIDDEN_ROLE_TOKENS,
    _materialize_open_role_base_data,
    _source_hashes,
    _train_linear_base_ensemble,
    aggregate_candidate_draw_scores,
    build_reversible_selected_contexts,
    coverage_matched_recency_contexts,
    fit_query_candidate_coverage_threshold,
    predict_query_context_probabilities_by_model,
    query_strategy_metrics,
    summarize_five_base_seed_strategy,
    summarize_joint_seed_strategy,
    summarize_utility_seed_strategy,
)
from .emotiontalk_sampled_context_runner import (
    OpenRoleDiagnosticError,
    _assert_aggregate_output,
    assemble_fit_probability_checkpoints,
    load_selection_probability_checkpoint,
    reconstruct_open_role_tasks,
    recover_query_labels_from_cached_utilities,
    verify_recomputed_59d_cache,
)
from .emotiontalk_text_p1 import LABEL_NAMES
from .meld_text_pilot import sha256_file


PROTOCOL = "emotiontalk_class_balanced_utility_repair_v1"
REPORT_SCHEMA_VERSION = "emotiontalk_class_balanced_query_policy_v1"
REGISTERED_STATUS = "repair_2_of_3_frozen_before_model_selection_result"
MODEL_MODES = (
    "forward_only_capacity_matched",
    "backward_only_capacity_matched",
    "pseudo_bidirectional_capacity_matched",
    "true_bidirectional_capacity_matched",
)
MODEL_NAMES = (
    "class_balanced_forward_mlp",
    "class_balanced_backward_mlp",
    "class_balanced_pseudo_bidirectional_mlp",
    "class_balanced_true_bidirectional_mlp",
)
UTILITY_STRATEGIES = {
    "class_balanced_forward_mlp": "class_balanced_forward_selected_history",
    "class_balanced_backward_mlp": "class_balanced_backward_selected_history",
    "class_balanced_pseudo_bidirectional_mlp": (
        "class_balanced_pseudo_selected_history"
    ),
    "class_balanced_true_bidirectional_mlp": (
        "class_balanced_true_bidirectional_selected_history"
    ),
}


class ClassBalancedRepairContractError(ContractError):
    """Raised when the class-balanced repair violates its frozen contract."""


@dataclass(frozen=True)
class ClassBalanceSpec:
    """Frozen query-level class balancing and deterministic resampling rule."""

    scheme: str = "effective_number"
    beta: float = 0.999
    resample_size_multiplier: float = 1.0
    frequency_unit: str = "unique_query"
    task_weight_rule: str = "class_weight_divided_by_tasks_for_same_query"
    oof_frequency_scope: str = "training_fold_only"
    final_frequency_scope: str = "all_fit_queries"

    def __post_init__(self) -> None:
        if self.scheme not in {"inverse_frequency", "effective_number"}:
            raise ClassBalancedRepairContractError(
                f"unsupported class-balance scheme: {self.scheme}"
            )
        if self.scheme == "effective_number" and not 0.0 < self.beta < 1.0:
            raise ClassBalancedRepairContractError(
                "effective-number beta must lie strictly between zero and one"
            )
        if not np.isclose(self.resample_size_multiplier, 1.0):
            raise ClassBalancedRepairContractError(
                "registered repair preserves the original training-row count"
            )
        expected = {
            "frequency_unit": "unique_query",
            "task_weight_rule": "class_weight_divided_by_tasks_for_same_query",
            "oof_frequency_scope": "training_fold_only",
            "final_frequency_scope": "all_fit_queries",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ClassBalancedRepairContractError(
                    f"registered class-balance field changed: {field}"
                )


@dataclass(frozen=True)
class ClassBalanceProfile:
    """Aggregate profile plus task weights; task weights are never serialized."""

    class_counts: np.ndarray
    class_weights: np.ndarray
    task_weights: np.ndarray
    unique_queries: int
    task_rows: int

    def public_dict(self, class_names: Sequence[str]) -> dict[str, object]:
        names = tuple(str(name) for name in class_names)
        if len(names) != len(self.class_counts):
            raise ClassBalancedRepairContractError(
                "class names do not align with the balance profile"
            )
        return {
            "frequency_unit": "unique_fit_query",
            "unique_fit_queries": int(self.unique_queries),
            "sampled_task_rows": int(self.task_rows),
            "class_counts": {
                name: int(self.class_counts[index])
                for index, name in enumerate(names)
            },
            "normalized_class_weights": {
                name: float(self.class_weights[index])
                for index, name in enumerate(names)
            },
        }


@dataclass(frozen=True)
class CapacityMatchedUtilitySpec:
    """Two-output shared architecture used by every utility control."""

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
            raise ClassBalancedRepairContractError(
                f"unknown capacity-matched utility mode: {self.mode}"
            )
        # Reuse the established optimizer validation without reusing its
        # one-head/two-head capacity decision.
        original_mode = {
            "forward_only_capacity_matched": "forward_only",
            "backward_only_capacity_matched": "backward_only",
            "pseudo_bidirectional_capacity_matched": "pseudo_bidirectional_shared",
            "true_bidirectional_capacity_matched": "bidirectional_shared",
        }[self.mode]
        UtilityModelSpec(
            name=self.name,
            mode=original_mode,
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            activation=self.activation,
            solver=self.solver,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "output_heads": 2,
            "alpha": float(self.alpha),
            "max_iter": int(self.max_iter),
            "tolerance": float(self.tolerance),
            "activation": self.activation,
            "solver": self.solver,
            "batch_size": int(self.batch_size),
            "learning_rate_init": float(self.learning_rate_init),
            "early_stopping": bool(self.early_stopping),
            "validation_fraction": float(self.validation_fraction),
            "n_iter_no_change": int(self.n_iter_no_change),
        }


@dataclass
class FittedCapacityMatchedUtilityModel:
    """A fitted utility model whose prediction path accepts features only."""

    spec: CapacityMatchedUtilitySpec
    seed: int
    x_scaler: StandardScaler
    target_mean: np.ndarray
    target_scale: np.ndarray
    estimator: MLPRegressor
    parameter_count: int
    balance_profile: ClassBalanceProfile

    def predict(self, x: np.ndarray) -> UtilityPredictions:
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.x_scaler.n_features_in_:
            raise ClassBalancedRepairContractError(
                "prediction features do not match the fitted 59-D schema"
            )
        if not np.isfinite(features).all():
            raise ClassBalancedRepairContractError(
                "prediction features contain non-finite values"
            )
        standardized = np.asarray(
            self.estimator.predict(self.x_scaler.transform(features)),
            dtype=np.float64,
        )
        if standardized.shape != (len(features), 2):
            raise AssertionError("capacity-matched model must emit two heads")
        raw = standardized * self.target_scale[None, :] + self.target_mean[None, :]
        first, second = raw[:, 0], raw[:, 1]
        if self.spec.mode == "forward_only_capacity_matched":
            return UtilityPredictions(first, None, first.copy())
        if self.spec.mode == "backward_only_capacity_matched":
            return UtilityPredictions(None, first, first.copy())
        # Pseudo has two forward-supervised heads; true has independent
        # forward/backward supervision.  Both use the same conservative rule.
        return UtilityPredictions(first, second, np.minimum(first, second))


@dataclass(frozen=True)
class BalancedOOFPredictions:
    predictions: UtilityPredictions
    fold_by_row: np.ndarray
    n_splits: int
    training_profiles: tuple[ClassBalanceProfile, ...]


@dataclass(frozen=True)
class BalancedUtilitySeedScores:
    seed: int
    threshold: float
    fit_oof_scores: np.ndarray
    fit_oof_fold_by_row: np.ndarray
    selection_scores: np.ndarray
    fit_query_candidate_pairs: int
    realized_fit_query_candidate_coverage: float
    parameter_count: int


def default_capacity_matched_specs() -> tuple[CapacityMatchedUtilitySpec, ...]:
    return tuple(
        CapacityMatchedUtilitySpec(name=name, mode=mode)
        for name, mode in zip(MODEL_NAMES, MODEL_MODES, strict=True)
    )


def _aligned_query_labels(
    tasks: Sequence[object],
    task_labels: np.ndarray,
    *,
    n_classes: int,
) -> tuple[dict[int, int], dict[int, int]]:
    labels = np.asarray(task_labels)
    if (
        labels.shape != (len(tasks),)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= int(n_classes)))
    ):
        raise ClassBalancedRepairContractError(
            "task labels must be aligned integer emotion labels"
        )
    query_label: dict[int, int] = {}
    query_rows: dict[int, int] = {}
    for task, label in zip(tasks, labels, strict=True):
        query = int(getattr(task, "query_index"))
        value = int(label)
        if query in query_label and query_label[query] != value:
            raise ClassBalancedRepairContractError(
                "sampled tasks assign inconsistent labels to the same query"
            )
        query_label[query] = value
        query_rows[query] = query_rows.get(query, 0) + 1
    if not query_label:
        raise ClassBalancedRepairContractError("class balancing requires fit queries")
    return query_label, query_rows


def build_class_balance_profile(
    tasks: Sequence[object],
    task_labels: np.ndarray,
    *,
    n_classes: int,
    spec: ClassBalanceSpec,
) -> ClassBalanceProfile:
    """Count labels once per query and distribute each query's weight over draws."""

    query_label, query_rows = _aligned_query_labels(
        tasks, task_labels, n_classes=int(n_classes)
    )
    counts = np.bincount(
        np.asarray(tuple(query_label.values()), dtype=np.int64),
        minlength=int(n_classes),
    )
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ClassBalancedRepairContractError(
            f"fit class-balance profile is missing emotion classes: {missing}"
        )
    if spec.scheme == "inverse_frequency":
        raw = 1.0 / counts.astype(np.float64)
    else:
        raw = (1.0 - float(spec.beta)) / (
            1.0 - np.power(float(spec.beta), counts.astype(np.float64))
        )
    # Normalize so the mean weight across unique fit queries is exactly one.
    normalizer = float(np.sum(counts * raw) / np.sum(counts))
    class_weights = raw / normalizer
    task_weights = np.asarray(
        [
            class_weights[int(label)] / query_rows[int(getattr(task, "query_index"))]
            for task, label in zip(tasks, np.asarray(task_labels), strict=True)
        ],
        dtype=np.float64,
    )
    if (
        task_weights.shape != (len(tasks),)
        or not np.isfinite(task_weights).all()
        or np.any(task_weights <= 0.0)
    ):
        raise AssertionError("class-balanced task weights are invalid")
    if not np.isclose(task_weights.sum(), len(query_label), rtol=1e-10, atol=1e-10):
        raise AssertionError("query-normalized task weights do not sum to query count")
    return ClassBalanceProfile(
        class_counts=counts.astype(np.int64, copy=False),
        class_weights=np.asarray(class_weights, dtype=np.float64),
        task_weights=task_weights,
        unique_queries=int(len(query_label)),
        task_rows=int(len(tasks)),
    )


def deterministic_weighted_resample_indices(
    sample_weights: np.ndarray,
    *,
    size: int,
    seed: int,
) -> np.ndarray:
    """Systematic weighted resampling with exact size and deterministic shuffle."""

    weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        weights.ndim != 1
        or len(weights) < 2
        or int(size) < 2
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ClassBalancedRepairContractError(
            "weighted resampling requires finite positive row weights"
        )
    probability = weights / weights.sum()
    rng = np.random.default_rng(int(seed))
    offset = float(rng.random()) / int(size)
    positions = offset + np.arange(int(size), dtype=np.float64) / int(size)
    indices = np.searchsorted(np.cumsum(probability), positions, side="right")
    indices = np.minimum(indices, len(weights) - 1).astype(np.int64, copy=False)
    rng.shuffle(indices)
    if indices.shape != (int(size),) or np.any((indices < 0) | (indices >= len(weights))):
        raise AssertionError("deterministic weighted resampling left the row domain")
    return indices


def _capacity_matched_targets(split: UtilitySplit, mode: str) -> np.ndarray:
    if mode in {
        "forward_only_capacity_matched",
        "pseudo_bidirectional_capacity_matched",
    }:
        return np.column_stack([split.forward, split.forward])
    if mode == "backward_only_capacity_matched":
        return np.column_stack([split.backward, split.backward])
    if mode == "true_bidirectional_capacity_matched":
        return np.column_stack([split.forward, split.backward])
    raise ClassBalancedRepairContractError(f"unknown utility mode: {mode}")


def fit_class_balanced_utility_model(
    split: UtilitySplit,
    tasks: Sequence[object],
    task_labels: np.ndarray,
    model_spec: CapacityMatchedUtilitySpec,
    balance_spec: ClassBalanceSpec,
    *,
    seed: int,
) -> FittedCapacityMatchedUtilityModel:
    """Fit a capacity-matched MLP from feature-only inputs after resampling."""

    if len(tasks) != len(split.x):
        raise ClassBalancedRepairContractError(
            "sampled tasks are not row-aligned with the utility split"
        )
    profile = build_class_balance_profile(
        tasks,
        task_labels,
        n_classes=len(LABEL_NAMES),
        spec=balance_spec,
    )
    resample_size = int(round(len(split.x) * balance_spec.resample_size_multiplier))
    indices = deterministic_weighted_resample_indices(
        profile.task_weights,
        size=resample_size,
        seed=int(seed),
    )
    features = np.asarray(split.x[indices], dtype=np.float64)
    target = np.asarray(
        _capacity_matched_targets(split, model_spec.mode)[indices], dtype=np.float64
    )
    parameters = trainable_parameter_count(
        split.x.shape[1], model_spec.hidden_layer_sizes, 2
    )
    if parameters >= MAX_TRAINABLE_PARAMETERS:
        raise ClassBalancedRepairContractError(
            f"utility model has {parameters:,} parameters; limit is "
            f"<{MAX_TRAINABLE_PARAMETERS:,}"
        )
    x_scaler = StandardScaler().fit(features)
    target_mean = target.mean(axis=0)
    target_scale = target.std(axis=0)
    target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
    standardized_target = (target - target_mean[None, :]) / target_scale[None, :]
    early_stopping = bool(model_spec.early_stopping and model_spec.solver != "lbfgs")
    optimizer_rows = (
        max(
            1,
            int(
                np.floor(
                    len(features) * (1.0 - float(model_spec.validation_fraction))
                )
            ),
        )
        if early_stopping
        else len(features)
    )
    estimator = MLPRegressor(
        hidden_layer_sizes=model_spec.hidden_layer_sizes,
        activation=model_spec.activation,
        solver=model_spec.solver,
        alpha=float(model_spec.alpha),
        max_iter=int(model_spec.max_iter),
        tol=float(model_spec.tolerance),
        random_state=int(seed),
        shuffle=True,
        batch_size=min(int(model_spec.batch_size), optimizer_rows),
        learning_rate_init=float(model_spec.learning_rate_init),
        early_stopping=early_stopping,
        validation_fraction=float(model_spec.validation_fraction),
        n_iter_no_change=int(model_spec.n_iter_no_change),
    )
    estimator.fit(x_scaler.transform(features), standardized_target)
    actual_parameters = int(
        sum(array.size for array in estimator.coefs_)
        + sum(array.size for array in estimator.intercepts_)
    )
    if actual_parameters != parameters or int(estimator.n_outputs_) != 2:
        raise AssertionError("capacity-matched utility architecture changed")
    return FittedCapacityMatchedUtilityModel(
        spec=model_spec,
        seed=int(seed),
        x_scaler=x_scaler,
        target_mean=target_mean,
        target_scale=target_scale,
        estimator=estimator,
        parameter_count=parameters,
        balance_profile=profile,
    )


def group_oof_class_balanced_predictions(
    split: UtilitySplit,
    tasks: Sequence[object],
    task_labels: np.ndarray,
    model_spec: CapacityMatchedUtilitySpec,
    balance_spec: ClassBalanceSpec,
    *,
    seed: int,
    maximum_splits: int = 5,
) -> BalancedOOFPredictions:
    """Create group-OOF scores with fold-train-only class frequencies."""

    labels = np.asarray(task_labels)
    if len(tasks) != len(split.x) or labels.shape != (len(split.x),):
        raise ClassBalancedRepairContractError("OOF tasks and labels are not aligned")
    unique_clusters = np.unique(split.cluster_codes)
    n_splits = min(int(maximum_splits), len(unique_clusters))
    if n_splits < 2:
        raise ClassBalancedRepairContractError("group OOF requires at least two folds")
    fold_by_row = np.full(len(split.x), -1, dtype=np.int32)
    forward = (
        None
        if model_spec.mode == "backward_only_capacity_matched"
        else np.full(len(split.x), np.nan, dtype=np.float64)
    )
    backward = (
        None
        if model_spec.mode == "forward_only_capacity_matched"
        else np.full(len(split.x), np.nan, dtype=np.float64)
    )
    decision = np.full(len(split.x), np.nan, dtype=np.float64)
    profiles: list[ClassBalanceProfile] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, held_index) in enumerate(
        splitter.split(split.x, groups=split.cluster_codes)
    ):
        train_clusters = set(int(value) for value in split.cluster_codes[train_index])
        held_clusters = set(int(value) for value in split.cluster_codes[held_index])
        if train_clusters & held_clusters:
            raise AssertionError("group OOF leaked a cluster")
        train_split = UtilitySplit.validated(
            split.x[train_index],
            split.forward[train_index],
            split.backward[train_index],
            split.cluster_codes[train_index],
            label=f"class-balanced OOF fold {fold} train",
            require_multiple_clusters=False,
        )
        train_tasks = tuple(tasks[int(index)] for index in train_index)
        model = fit_class_balanced_utility_model(
            train_split,
            train_tasks,
            labels[train_index],
            model_spec,
            balance_spec,
            seed=int(seed) + fold * 10_007,
        )
        profiles.append(model.balance_profile)
        held = model.predict(split.x[held_index])
        fold_by_row[held_index] = fold
        decision[held_index] = held.decision_score
        if forward is not None and held.forward is not None:
            forward[held_index] = held.forward
        if backward is not None and held.backward is not None:
            backward[held_index] = held.backward
    arrays = [decision, *(value for value in (forward, backward) if value is not None)]
    if np.any(fold_by_row < 0) or any(not np.isfinite(value).all() for value in arrays):
        raise AssertionError("class-balanced OOF prediction coverage is incomplete")
    for cluster in unique_clusters:
        if len(np.unique(fold_by_row[split.cluster_codes == cluster])) != 1:
            raise AssertionError("a cluster was split across OOF folds")
    return BalancedOOFPredictions(
        UtilityPredictions(forward, backward, decision),
        fold_by_row,
        n_splits,
        tuple(profiles),
    )


def _mapping_keys(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ClassBalancedRepairContractError(f"{label} must be a mapping")
    return value


def load_class_balanced_repair_config(
    path: str | Path,
) -> tuple[
    Mapping[str, object],
    ClassBalanceSpec,
    tuple[CapacityMatchedUtilitySpec, ...],
]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClassBalancedRepairContractError(
            f"cannot read class-balanced repair config: {error}"
        ) from error
    root = _mapping_keys(config, label="repair config")
    if root.get("protocol") != PROTOCOL or root.get("status") != REGISTERED_STATUS:
        raise ClassBalancedRepairContractError(
            "class-balanced repair protocol or freeze status changed"
        )
    if root.get("analysis_role") != "open_role_repair_2_of_3":
        raise ClassBalancedRepairContractError("repair analysis role changed")
    if tuple(int(value) for value in root.get("utility_seeds", ())) != DEFAULT_SEEDS:
        raise ClassBalancedRepairContractError("registered five utility seeds changed")
    if int(root.get("group_oof_folds", -1)) != 5:
        raise ClassBalancedRepairContractError("registered OOF fold count changed")
    if not np.isclose(
        float(root.get("target_fit_query_candidate_coverage", -1.0)),
        PRIMARY_HISTORY_COVERAGE,
    ):
        raise ClassBalancedRepairContractError("registered 25% coverage changed")

    balance = _mapping_keys(root.get("class_balance"), label="class_balance")
    balance_spec = ClassBalanceSpec(
        scheme=str(balance.get("scheme", "")),
        beta=float(balance.get("beta", np.nan)),
        resample_size_multiplier=float(
            balance.get("resample_size_multiplier", np.nan)
        ),
        frequency_unit=str(balance.get("frequency_unit", "")),
        task_weight_rule=str(balance.get("task_weight_rule", "")),
        oof_frequency_scope=str(balance.get("oof_frequency_scope", "")),
        final_frequency_scope=str(balance.get("final_frequency_scope", "")),
    )
    architecture = _mapping_keys(root.get("architecture"), label="architecture")
    hidden = tuple(int(value) for value in architecture.get("hidden_layer_sizes", ()))
    specs = tuple(
        CapacityMatchedUtilitySpec(
            name=name,
            mode=mode,
            hidden_layer_sizes=hidden,
            alpha=float(architecture.get("alpha", np.nan)),
            max_iter=int(architecture.get("max_iter", -1)),
            tolerance=float(architecture.get("tolerance", np.nan)),
            activation=str(architecture.get("activation", "")),
            solver=str(architecture.get("solver", "")),
            batch_size=int(architecture.get("batch_size", -1)),
            learning_rate_init=float(
                architecture.get("learning_rate_init", np.nan)
            ),
            early_stopping=bool(architecture.get("early_stopping", False)),
            validation_fraction=float(
                architecture.get("validation_fraction", np.nan)
            ),
            n_iter_no_change=int(architecture.get("n_iter_no_change", -1)),
        )
        for name, mode in zip(MODEL_NAMES, MODEL_MODES, strict=True)
    )
    registered = root.get("registered_models")
    expected_registered = [
        {"name": name, "mode": mode, "output_heads": 2}
        for name, mode in zip(MODEL_NAMES, MODEL_MODES, strict=True)
    ]
    if registered != expected_registered:
        raise ClassBalancedRepairContractError("registered four-model set changed")
    if len({trainable_parameter_count(59, spec.hidden_layer_sizes, 2) for spec in specs}) != 1:
        raise AssertionError("registered models do not have identical capacity")
    return root, balance_spec, specs


def fit_class_balanced_seed_scores(
    cache: BidirectionalUtilityCache,
    fit_tasks: Sequence[object],
    fit_task_labels: np.ndarray,
    balance_spec: ClassBalanceSpec,
    specs: Sequence[CapacityMatchedUtilitySpec],
    *,
    maximum_splits: int,
) -> dict[str, tuple[BalancedUtilitySeedScores, ...]]:
    """Expose in-memory fit-OOF and selection scores for query-level reuse.

    The returned row arrays are deliberately not part of the aggregate report.
    Callers must first establish the same verified cache/task alignment used by
    :func:`run_open_role_class_balanced_query_policy`.
    """

    if tuple(spec.name for spec in specs) != MODEL_NAMES:
        raise ClassBalancedRepairContractError("registered model order changed")
    result: dict[str, tuple[BalancedUtilitySeedScores, ...]] = {}
    parameter_counts: set[int] = set()
    for model_spec in specs:
        states: list[BalancedUtilitySeedScores] = []
        for seed in DEFAULT_SEEDS:
            oof = group_oof_class_balanced_predictions(
                cache.fit,
                fit_tasks,
                fit_task_labels,
                model_spec,
                balance_spec,
                seed=int(seed),
                maximum_splits=int(maximum_splits),
            )
            threshold, pair_count, realized_coverage = (
                fit_query_candidate_coverage_threshold(
                    fit_tasks,
                    oof.predictions.decision_score,
                    target_coverage=PRIMARY_HISTORY_COVERAGE,
                )
            )
            fitted = fit_class_balanced_utility_model(
                cache.fit,
                fit_tasks,
                fit_task_labels,
                model_spec,
                balance_spec,
                seed=int(seed),
            )
            prediction = fitted.predict(cache.selection.x)
            parameter_counts.add(int(fitted.parameter_count))
            states.append(
                BalancedUtilitySeedScores(
                    seed=int(seed),
                    threshold=float(threshold),
                    fit_oof_scores=np.asarray(
                        oof.predictions.decision_score, dtype=np.float64
                    ),
                    fit_oof_fold_by_row=np.asarray(oof.fold_by_row, dtype=np.int32),
                    selection_scores=np.asarray(
                        prediction.decision_score, dtype=np.float64
                    ),
                    fit_query_candidate_pairs=int(pair_count),
                    realized_fit_query_candidate_coverage=float(realized_coverage),
                    parameter_count=int(fitted.parameter_count),
                )
            )
        result[model_spec.name] = tuple(states)
    if len(parameter_counts) != 1:
        raise AssertionError("four utility models are not exactly capacity matched")
    return result


def _fixed_query_metrics(
    data: Any,
    base_models: Sequence[object],
    current: Mapping[str, object],
    selection_queries: tuple[int, ...],
    *,
    ece_bins: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[dict[str, float | int], ...],
    dict[str, float | int],
    tuple[dict[str, float | int], ...],
    dict[str, float | int],
]:
    empty_contexts = tuple(tuple() for _ in selection_queries)
    all_contexts = tuple(data.histories[query] for query in selection_queries)
    current_by_base = predict_query_context_probabilities_by_model(
        base_models,
        current,
        data.quality,
        data.quality_names,
        selection_queries,
        empty_contexts,
        data.histories,
        n_classes=len(LABEL_NAMES),
    )
    all_by_base = predict_query_context_probabilities_by_model(
        base_models,
        current,
        data.quality,
        data.quality_names,
        selection_queries,
        all_contexts,
        data.histories,
        n_classes=len(LABEL_NAMES),
    )
    labels = np.asarray(data.labels[data.selection_indices], dtype=np.int64)
    clusters = np.asarray([data.groups[query] for query in selection_queries], dtype=object)
    current_records = tuple(
        query_strategy_metrics(
            labels,
            probability,
            probability,
            empty_contexts,
            data.histories,
            selection_queries,
            clusters,
            ece_bins=int(ece_bins),
        )
        for probability in current_by_base
    )
    all_records = tuple(
        query_strategy_metrics(
            labels,
            probability,
            current_by_base[index],
            all_contexts,
            data.histories,
            selection_queries,
            clusters,
            ece_bins=int(ece_bins),
        )
        for index, probability in enumerate(all_by_base)
    )
    current_mean = np.mean(current_by_base, axis=0)
    current_metrics = query_strategy_metrics(
        labels,
        current_mean,
        current_mean,
        empty_contexts,
        data.histories,
        selection_queries,
        clusters,
        ece_bins=int(ece_bins),
    )
    all_metrics = query_strategy_metrics(
        labels,
        np.mean(all_by_base, axis=0),
        current_mean,
        all_contexts,
        data.histories,
        selection_queries,
        clusters,
        ece_bins=int(ece_bins),
    )
    return (
        labels,
        clusters,
        current_by_base,
        current_records,
        current_metrics,
        all_records,
        all_metrics,
    )


def _mean_five_base_seed_record(
    records: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate five base fits without treating them as independent queries."""

    rows = tuple(records)
    if len(rows) != 5 or any(set(row) != set(rows[0]) for row in rows):
        raise ClassBalancedRepairContractError(
            "per-utility-seed reporting requires five schema-aligned base records"
        )
    query_counts = {int(row["queries"]) for row in rows}
    if len(query_counts) != 1:
        raise ClassBalancedRepairContractError(
            "per-utility-seed base records contain different query counts"
        )
    result: dict[str, float | int] = {"queries": int(next(iter(query_counts)))}
    for key in rows[0]:
        if key == "queries":
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ClassBalancedRepairContractError(
                "per-utility-seed metric contains non-finite values"
            )
        result[key] = float(values.mean())
    return result


def _per_utility_seed_aggregate_records(
    record_grid: Sequence[Sequence[Mapping[str, float | int]]],
) -> list[dict[str, object]]:
    grid = tuple(tuple(row) for row in record_grid)
    if len(grid) != len(DEFAULT_SEEDS):
        raise ClassBalancedRepairContractError(
            "utility-seed report does not contain the frozen five seeds"
        )
    return [
        {
            "utility_seed": int(seed),
            "estimand": "mean_metric_across_five_independently_fitted_base_seeds",
            "metrics": _mean_five_base_seed_record(row),
        }
        for seed, row in zip(DEFAULT_SEEDS, grid, strict=True)
    ]


def _metric_differences(
    candidate: Mapping[str, float | int],
    reference: Mapping[str, float | int],
) -> dict[str, float]:
    keys = (
        "pooled_macro_f1",
        "pooled_accuracy",
        "pooled_nll",
        "pooled_brier",
        "mean_excess_nll_vs_current",
        "harm_rate_vs_current",
        "actual_history_coverage",
    )
    return {
        f"{key}_difference": float(candidate[key]) - float(reference[key])
        for key in keys
    }


def _true_reference_diagnostics(
    true_grid: Sequence[Sequence[Mapping[str, float | int]]],
    recency_grid: Sequence[Sequence[Mapping[str, float | int]]],
    current_records: Sequence[Mapping[str, float | int]],
    all_records: Sequence[Mapping[str, float | int]],
    *,
    minimum_macro_f1_gain: float,
    minimum_history_coverage: float,
) -> dict[str, object]:
    true_rows = _per_utility_seed_aggregate_records(true_grid)
    recency_rows = _per_utility_seed_aggregate_records(recency_grid)
    current = _mean_five_base_seed_record(current_records)
    all_history = _mean_five_base_seed_record(all_records)
    per_seed: list[dict[str, object]] = []
    registered_passes = 0
    three_reference_passes = 0
    for true_row, recency_row in zip(true_rows, recency_rows, strict=True):
        candidate = true_row["metrics"]
        recency = recency_row["metrics"]
        if not isinstance(candidate, Mapping) or not isinstance(recency, Mapping):
            raise AssertionError("utility-seed metric record changed type")
        registered_gate = bool(
            float(candidate["pooled_macro_f1"])
            - float(current["pooled_macro_f1"])
            >= float(minimum_macro_f1_gain)
            and float(candidate["mean_excess_nll_vs_current"]) < 0.0
            and float(candidate["actual_history_coverage"])
            >= float(minimum_history_coverage)
        )
        # This stricter three-reference diagnostic was added to the reporting
        # contract before any result was observed.  It is not promoted to the
        # pre-registered method-success gate in the utility config.
        three_reference = bool(
            registered_gate
            and float(candidate["pooled_macro_f1"])
            > float(all_history["pooled_macro_f1"])
            and float(candidate["pooled_nll"])
            <= float(all_history["pooled_nll"])
            and float(candidate["pooled_macro_f1"])
            > float(recency["pooled_macro_f1"])
            and float(candidate["pooled_nll"])
            <= float(recency["pooled_nll"])
        )
        registered_passes += int(registered_gate)
        three_reference_passes += int(three_reference)
        per_seed.append(
            {
                "utility_seed": int(true_row["utility_seed"]),
                "true_bidirectional_five_base_seed_mean": dict(candidate),
                "candidate_minus_current": _metric_differences(candidate, current),
                "candidate_minus_all_history": _metric_differences(
                    candidate, all_history
                ),
                "candidate_minus_coverage_matched_recency": _metric_differences(
                    candidate, recency
                ),
                "registered_current_joint_gate_pass": registered_gate,
                "all_three_reference_diagnostic_pass": three_reference,
            }
        )
    return {
        "per_utility_seed": per_seed,
        "registered_current_joint_gate": {
            "definition": {
                "minimum_macro_f1_gain_vs_current": float(minimum_macro_f1_gain),
                "mean_excess_nll_vs_current_strictly_below_zero": True,
                "minimum_actual_history_coverage": float(minimum_history_coverage),
            },
            "successful_utility_seeds_out_of_five": int(registered_passes),
            "meets_four_of_five": bool(registered_passes >= 4),
        },
        "all_three_reference_diagnostic": {
            "role": "reporting_diagnostic_not_prespecified_method_success_gate",
            "additional_requirements": [
                "macro_f1_strictly_above_all_history",
                "nll_not_above_all_history",
                "macro_f1_strictly_above_coverage_matched_recency",
                "nll_not_above_coverage_matched_recency",
            ],
            "successful_utility_seeds_out_of_five": int(three_reference_passes),
            "meets_four_of_five": bool(three_reference_passes >= 4),
        },
    }


def _nll_identity_audit(
    strategy_grids: Mapping[
        str, Sequence[Sequence[Mapping[str, float | int]]]
    ],
    strategy_ensembles: Mapping[str, Sequence[Mapping[str, float | int]]],
    current_records: Sequence[Mapping[str, float | int]],
    current_ensemble: Mapping[str, float | int],
    all_records: Sequence[Mapping[str, float | int]],
    all_ensemble: Mapping[str, float | int],
) -> dict[str, object]:
    """Verify excess NLL equals strategy NLL minus matched current NLL."""

    errors: list[float] = []

    def add_error(
        record: Mapping[str, float | int],
        current: Mapping[str, float | int],
    ) -> None:
        errors.append(
            abs(
                float(record["mean_excess_nll_vs_current"])
                - (
                    float(record["pooled_nll"])
                    - float(current["pooled_nll"])
                )
            )
        )

    current_tuple = tuple(current_records)
    for base_index, record in enumerate(all_records):
        add_error(record, current_tuple[base_index])
    add_error(all_ensemble, current_ensemble)
    for grid in strategy_grids.values():
        for row in grid:
            for base_index, record in enumerate(row):
                add_error(record, current_tuple[base_index])
    for records in strategy_ensembles.values():
        for record in records:
            add_error(record, current_ensemble)
    maximum_error = float(max(errors, default=0.0))
    tolerance = 1e-12
    if maximum_error > tolerance:
        raise ClassBalancedRepairContractError(
            "query-level excess-NLL identity exceeded 1e-12"
        )
    return {
        "identity": (
            "mean_excess_nll_vs_current == pooled_nll_strategy - "
            "pooled_nll_matched_current"
        ),
        "records_checked": int(len(errors)),
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": maximum_error,
        "passed": True,
    }


def canonical_manifest_sha256(material: Mapping[str, object]) -> str:
    """Hash the frozen provenance/environment manifest deterministically."""

    payload = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_open_role_class_balanced_query_policy(
    data_dir: Path,
    feature_path: Path,
    base_config_path: Path,
    utility_config_path: Path,
    repair_config_path: Path,
    private_cache_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run repair 2/3 using fit-only class balancing and model-selection scoring."""

    paths = tuple(
        Path(value)
        for value in (
            data_dir,
            feature_path,
            base_config_path,
            utility_config_path,
            repair_config_path,
            private_cache_path,
            checkpoint_dir,
            output_path,
        )
    )
    (
        data_dir,
        feature_path,
        base_config_path,
        utility_config_path,
        repair_config_path,
        private_cache_path,
        checkpoint_dir,
        output_path,
    ) = paths
    # Write-once and fail-before-read semantics preserve prior experiments.
    if output_path.exists():
        raise FileExistsError(f"class-balanced repair output already exists: {output_path}")
    if not data_dir.is_dir() or not checkpoint_dir.is_dir():
        raise FileNotFoundError("data-dir and checkpoint-dir must exist")
    for path in (
        feature_path,
        base_config_path,
        utility_config_path,
        repair_config_path,
        private_cache_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_config = _read_json(base_config_path)
    utility_config = _read_json(utility_config_path)
    _validate_config(utility_config, base_config)
    repair_config, balance_spec, model_specs = load_class_balanced_repair_config(
        repair_config_path
    )
    expected_hashes = {
        "base_config_sha256": sha256_file(base_config_path),
        "utility_config_sha256": sha256_file(utility_config_path),
        "feature_sha256": sha256_file(feature_path),
    }
    cache = load_private_oof_cache(private_cache_path)
    if cache.source_hashes.get("base_config_sha256") != expected_hashes["base_config_sha256"]:
        raise OpenRoleDiagnosticError("private cache base-config hash mismatch")
    if cache.source_hashes.get("utility_config_sha256") != expected_hashes["utility_config_sha256"]:
        raise OpenRoleDiagnosticError("private cache utility-config hash mismatch")

    task_material, feature_config_sha256 = reconstruct_open_role_tasks(
        feature_path, base_config, utility_config
    )
    fold_paths = tuple(
        checkpoint_dir / f"fold_{fold}.npz"
        for fold in range(1, int(base_config["crossfit_folds"]) + 1)
    )
    selection_checkpoint_path = checkpoint_dir / "selection.npz"
    expected_names = {path.name for path in fold_paths} | {selection_checkpoint_path.name}
    if {path.name for path in checkpoint_dir.glob("*.npz")} != expected_names:
        raise ClassBalancedRepairContractError("checkpoint directory schema changed")
    base_seed_count = len(tuple(base_config["seeds"]))
    fit_probability = assemble_fit_probability_checkpoints(
        fold_paths,
        expected_task_count=len(task_material.fit_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=expected_hashes,
        expected_positions_by_name=task_material.expected_fold_positions,
    )
    selection_probability = load_selection_probability_checkpoint(
        selection_checkpoint_path,
        expected_task_count=len(task_material.selection_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=expected_hashes,
    )
    fit_x, fit_names = probability_task_features(
        np.mean(fit_probability, axis=0),
        task_material.fit_tasks,
        task_material.histories,
    )
    selection_x, selection_names = probability_task_features(
        np.mean(selection_probability, axis=0),
        task_material.selection_tasks,
        task_material.histories,
    )
    verify_recomputed_59d_cache(
        cache,
        fit_x=fit_x,
        fit_feature_names=fit_names,
        selection_x=selection_x,
        selection_feature_names=selection_names,
        fit_cluster_codes=task_material.fit_cluster_codes,
        selection_cluster_codes=task_material.selection_cluster_codes,
    )

    # These are recovered, not direct fields in the private cache.  The cache
    # still indirectly determines labels through utilities plus probabilities;
    # the public report states that limitation explicitly.
    fit_task_labels = recover_query_labels_from_cached_utilities(
        task_material.fit_tasks,
        fit_probability,
        cache.fit.forward,
        cache.fit.backward,
    )
    selection_task_labels = recover_query_labels_from_cached_utilities(
        task_material.selection_tasks,
        selection_probability,
        cache.selection.forward,
        cache.selection.backward,
    )
    data = _materialize_open_role_base_data(data_dir, feature_path, utility_config)
    if data.histories != task_material.histories:
        raise ClassBalancedRepairContractError(
            "materialized strict-past histories changed task order"
        )
    expected_fit_labels = np.asarray(
        [data.labels[int(task.query_index)] for task in task_material.fit_tasks],
        dtype=np.int64,
    )
    expected_selection_labels = np.asarray(
        [data.labels[int(task.query_index)] for task in task_material.selection_tasks],
        dtype=np.int64,
    )
    if not np.array_equal(fit_task_labels, expected_fit_labels) or not np.array_equal(
        selection_task_labels, expected_selection_labels
    ):
        raise ClassBalancedRepairContractError(
            "recovered task labels disagree with indexed open-role labels"
        )
    full_fit_profile = build_class_balance_profile(
        task_material.fit_tasks,
        fit_task_labels,
        n_classes=len(LABEL_NAMES),
        spec=balance_spec,
    )

    base_models, current = _train_linear_base_ensemble(
        data, base_config, utility_config, task_material.fit_tasks
    )
    selection_queries = tuple(int(value) for value in data.selection_indices)
    if len(selection_queries) != len(set(selection_queries)):
        raise AssertionError("selection query rows are not unique")
    ece_bins = int(base_config.get("ece_bins", 15))
    (
        selection_labels,
        selection_clusters,
        current_probability_by_base,
        current_base_records,
        current_metrics,
        all_base_records,
        all_metrics,
    ) = _fixed_query_metrics(
        data,
        base_models,
        current,
        selection_queries,
        ece_bins=ece_bins,
    )
    current_probability = np.mean(current_probability_by_base, axis=0)

    utility_states = fit_class_balanced_seed_scores(
        cache,
        task_material.fit_tasks,
        fit_task_labels,
        balance_spec,
        model_specs,
        maximum_splits=int(repair_config["group_oof_folds"]),
    )
    by_strategy_grid: dict[str, list[tuple[dict[str, float | int], ...]]] = {
        strategy: [] for strategy in UTILITY_STRATEGIES.values()
    }
    by_strategy_grid["coverage_matched_recency"] = []
    by_strategy_ensemble: dict[str, list[dict[str, float | int]]] = {
        strategy: [] for strategy in by_strategy_grid
    }
    thresholds: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
    pair_counts: dict[str, list[int]] = {name: [] for name in MODEL_NAMES}
    realized_coverages: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
    parameter_counts: dict[str, list[int]] = {name: [] for name in MODEL_NAMES}
    for seed_index in range(5):
        true_contexts: tuple[tuple[int, ...], ...] | None = None
        for model_name in MODEL_NAMES:
            state = utility_states[model_name][seed_index]
            aggregated_scores = aggregate_candidate_draw_scores(
                task_material.selection_tasks, state.selection_scores
            )
            contexts = build_reversible_selected_contexts(
                selection_queries,
                data.histories,
                aggregated_scores,
                threshold=state.threshold,
            )
            probability_by_base = predict_query_context_probabilities_by_model(
                base_models,
                current,
                data.quality,
                data.quality_names,
                selection_queries,
                contexts,
                data.histories,
                n_classes=len(LABEL_NAMES),
            )
            strategy_name = UTILITY_STRATEGIES[model_name]
            records = tuple(
                query_strategy_metrics(
                    selection_labels,
                    probability,
                    current_probability_by_base[base_index],
                    contexts,
                    data.histories,
                    selection_queries,
                    selection_clusters,
                    ece_bins=ece_bins,
                )
                for base_index, probability in enumerate(probability_by_base)
            )
            by_strategy_grid[strategy_name].append(records)
            by_strategy_ensemble[strategy_name].append(
                query_strategy_metrics(
                    selection_labels,
                    np.mean(probability_by_base, axis=0),
                    current_probability,
                    contexts,
                    data.histories,
                    selection_queries,
                    selection_clusters,
                    ece_bins=ece_bins,
                )
            )
            thresholds[model_name].append(float(state.threshold))
            pair_counts[model_name].append(int(state.fit_query_candidate_pairs))
            realized_coverages[model_name].append(
                float(state.realized_fit_query_candidate_coverage)
            )
            parameter_counts[model_name].append(int(state.parameter_count))
            if model_name == "class_balanced_true_bidirectional_mlp":
                true_contexts = contexts
        if true_contexts is None:
            raise AssertionError("true bidirectional contexts were not constructed")
        recency_contexts = coverage_matched_recency_contexts(
            selection_queries, data.histories, true_contexts
        )
        recency_by_base = predict_query_context_probabilities_by_model(
            base_models,
            current,
            data.quality,
            data.quality_names,
            selection_queries,
            recency_contexts,
            data.histories,
            n_classes=len(LABEL_NAMES),
        )
        recency_records = tuple(
            query_strategy_metrics(
                selection_labels,
                probability,
                current_probability_by_base[base_index],
                recency_contexts,
                data.histories,
                selection_queries,
                selection_clusters,
                ece_bins=ece_bins,
            )
            for base_index, probability in enumerate(recency_by_base)
        )
        by_strategy_grid["coverage_matched_recency"].append(recency_records)
        by_strategy_ensemble["coverage_matched_recency"].append(
            query_strategy_metrics(
                selection_labels,
                np.mean(recency_by_base, axis=0),
                current_probability,
                recency_contexts,
                data.histories,
                selection_queries,
                selection_clusters,
                ece_bins=ece_bins,
            )
        )

    gate = utility_config["train_only_go_gate"]
    minimum_macro_gain = float(gate["minimum_macro_f1_gain"])
    minimum_coverage = float(gate["minimum_nontrivial_coverage"])
    strategy_summaries = {
        name: summarize_joint_seed_strategy(
            record_grid,
            current_base_records,
            minimum_macro_f1_gain=minimum_macro_gain,
            minimum_history_coverage=minimum_coverage,
        )
        for name, record_grid in by_strategy_grid.items()
    }
    for name, record_grid in by_strategy_grid.items():
        strategy_summaries[name]["per_utility_seed_five_base_seed_mean"] = (
            _per_utility_seed_aggregate_records(record_grid)
        )
    for name, records in by_strategy_ensemble.items():
        strategy_summaries[name]["base_seed_ensemble_diagnostic"] = (
            summarize_utility_seed_strategy(
                records,
                current_metrics,
                minimum_macro_f1_gain=minimum_macro_gain,
                minimum_history_coverage=minimum_coverage,
            )
        )
    declared_parameter_counts: set[int] = set()
    for model_name, strategy_name in UTILITY_STRATEGIES.items():
        threshold_values = np.asarray(thresholds[model_name], dtype=np.float64)
        coverage_values = np.asarray(realized_coverages[model_name], dtype=np.float64)
        unique_pairs = set(pair_counts[model_name])
        unique_parameters = set(parameter_counts[model_name])
        if len(unique_pairs) != 1 or len(unique_parameters) != 1:
            raise AssertionError("seed-invariant model contract changed")
        declared_parameter_counts.update(unique_parameters)
        strategy_summaries[strategy_name]["fit_oof_frozen_threshold"] = {
            "unit": "query-candidate score averaged across coalition draws",
            "target_coverage": float(PRIMARY_HISTORY_COVERAGE),
            "query_candidate_pairs": int(next(iter(unique_pairs))),
            "threshold_mean": float(threshold_values.mean()),
            "threshold_std": float(threshold_values.std(ddof=1)),
            "realized_pair_coverage_mean": float(coverage_values.mean()),
            "realized_pair_coverage_std": float(coverage_values.std(ddof=1)),
        }
        strategy_summaries[strategy_name]["trainable_parameter_count"] = int(
            next(iter(unique_parameters))
        )
    if len(declared_parameter_counts) != 1:
        raise AssertionError("four-model capacity matching failed")

    true_reference_diagnostics = _true_reference_diagnostics(
        by_strategy_grid["class_balanced_true_bidirectional_selected_history"],
        by_strategy_grid["coverage_matched_recency"],
        current_base_records,
        all_base_records,
        minimum_macro_f1_gain=minimum_macro_gain,
        minimum_history_coverage=minimum_coverage,
    )
    nll_identity_audit = _nll_identity_audit(
        by_strategy_grid,
        by_strategy_ensemble,
        current_base_records,
        current_metrics,
        all_base_records,
        all_metrics,
    )

    checkpoint_paths = (*fold_paths, selection_checkpoint_path)
    runner_path = (
        Path(__file__).parents[2]
        / "scripts"
        / "run_emotiontalk_class_balanced_utility_repair.py"
    )
    source_hashes = {
        **_source_hashes(
            data_dir,
            feature_path,
            base_config_path,
            utility_config_path,
            private_cache_path,
            checkpoint_paths,
        ),
        "repair_config_sha256": sha256_file(repair_config_path),
        "feature_config_sha256": feature_config_sha256,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": (
            hashlib.sha256(runner_path.read_bytes()).hexdigest()
            if runner_path.is_file()
            else None
        ),
    }
    environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }
    manifest_material = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_hashes": source_hashes,
        "environment": environment,
        "utility_seeds": [int(seed) for seed in DEFAULT_SEEDS],
        "group_oof_folds": int(repair_config["group_oof_folds"]),
        "target_fit_query_candidate_coverage": float(PRIMARY_HISTORY_COVERAGE),
        "class_balance": {
            "scheme": balance_spec.scheme,
            "beta": float(balance_spec.beta),
            "frequency_unit": balance_spec.frequency_unit,
            "task_weight_rule": balance_spec.task_weight_rule,
            "oof_frequency_scope": balance_spec.oof_frequency_scope,
            "final_frequency_scope": balance_spec.final_frequency_scope,
        },
        "models": [spec.public_dict() for spec in model_specs],
    }
    manifest_sha256 = canonical_manifest_sha256(manifest_material)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": (
            "open_role_class_balanced_query_development_complete_with_"
            "label_container_limitation"
        ),
        "analysis_stage": {
            "repair": "repair_2_of_3_class_balanced_utility_learning",
            "result_is_confirmatory": False,
            "real_experiment_executed_by_this_implementation_step": True,
        },
        "claim_boundary": (
            "Open-role fit/model-selection development with the existing linear base "
            "model only. It is not causal-backbone, calibration, holdout, validation, "
            "test, or top-conference confirmation. The upstream pickled train-label "
            "container is deserialized in full, so the run cannot claim a strict "
            "non-open-label deserialization seal."
        ),
        "class_balance_contract": {
            "scheme": balance_spec.scheme,
            "beta": float(balance_spec.beta),
            "frequency_unit": balance_spec.frequency_unit,
            "normalization": "mean_unique_fit_query_weight_equals_one",
            "task_weight_rule": balance_spec.task_weight_rule,
            "fit_mechanism": "deterministic_systematic_weighted_resampling",
            "resampled_rows": "same_count_as_source_training_rows",
            "oof_frequency_scope": balance_spec.oof_frequency_scope,
            "final_frequency_scope": balance_spec.final_frequency_scope,
            "inference_uses_gold_label": False,
            "gold_label_appended_to_features": False,
            "full_fit_profile": full_fit_profile.public_dict(LABEL_NAMES),
        },
        "label_recovery_contract": {
            "source": (
                "unique class reproducing both cached directional utilities from the "
                "four verified fit-OOF probability contexts"
            ),
            "private_cache_contains_direct_label_field": False,
            "private_cache_plus_probability_checkpoints_indirectly_identify_labels": True,
            "fit_recovery_used_for_class_balanced_training_only": True,
            "model_selection_recovery_used_for_scoring_only": True,
            "recovered_fit_and_model_selection_labels_match_indexed_open_role_labels": True,
        },
        "model_contract": {
            "models": [spec.public_dict() for spec in model_specs],
            "identical_trainable_parameter_count": int(
                next(iter(declared_parameter_counts))
            ),
            "single_direction_capacity_control": (
                "two duplicated directional supervision heads; decision uses head one"
            ),
            "pseudo_control": (
                "two forward-supervised heads; decision is their minimum"
            ),
            "true_bidirectional": (
                "one forward and one backward head; decision is their minimum"
            ),
        },
        "policy_contract": {
            "score_reduction": "mean across draws for each query-candidate pair",
            "threshold_source": "fit group-OOF query-candidate scores only",
            "threshold_target_coverage": float(PRIMARY_HISTORY_COVERAGE),
            "threshold_transfer": "frozen before model-selection scoring",
            "selection_rule": "score strictly greater than frozen threshold",
            "empty_selection_fallback": "current_only",
            "reversibility": "fresh immutable strict-past set per query",
            "predictions_per_model_selection_query_per_strategy": 1,
            "recency_baseline": (
                "same per-query selected-candidate count as true bidirectional"
            ),
            "primary_seed_estimand": (
                "mean over 5 utility seeds x 5 independently fitted base seeds"
            ),
        },
        "access_contract": {
            "roles_used": ["base_and_utility_fit", "model_selection"],
            "restricted_role_rows_used": 0,
            "external_nontrain_rows_used": 0,
            "row_level_output": False,
            "label_container_format": "single pickled train_corpus object mapping",
            "label_container_rows_deserialized": int(
                data.label_container_rows_deserialized
            ),
            "non_open_label_rows_deserialized": int(
                data.non_open_label_rows_deserialized
            ),
            "non_open_label_keys_indexed_for_training_or_metrics": 0,
            "strict_epistemic_non_open_label_deserialization_seal_satisfied": False,
            "required_remediation_for_strict_seal": (
                "pre-materialize and hash an open-role-only label sidecar before sealing"
            ),
        },
        "alignment_audit": {
            "fit_checkpoint_exact_nonoverlapping_cover": True,
            "selection_checkpoint_canonical_complete_order": True,
            "recomputed_59d_features_bitwise_equal_cache": True,
            "feature_names_and_cluster_codes_equal_cache": True,
            "both_roles_uniquely_recovered_from_cached_directional_targets": True,
        },
        "experiment_counts": {
            "fit_rows": int(len(data.fit_indices)),
            "model_selection_queries": int(len(data.selection_indices)),
            "fit_sampled_tasks": int(len(task_material.fit_tasks)),
            "model_selection_sampled_tasks": int(len(task_material.selection_tasks)),
            "base_seeds": int(len(tuple(base_config["seeds"]))),
            "utility_seeds": int(len(DEFAULT_SEEDS)),
            "primary_joint_seed_grid": int(
                len(tuple(base_config["seeds"])) * len(DEFAULT_SEEDS)
            ),
            "utility_models": int(len(MODEL_NAMES)),
            "task_feature_count": int(cache.fit.x.shape[1]),
        },
        "fixed_strategies": {
            "current_only": {
                "five_base_seed_primary": summarize_five_base_seed_strategy(
                    current_base_records
                ),
                "base_seed_ensemble_diagnostic": current_metrics,
            },
            "all_history": {
                "five_base_seed_primary": summarize_five_base_seed_strategy(
                    all_base_records
                ),
                "base_seed_ensemble_diagnostic": all_metrics,
            },
        },
        "utility_seed_strategy_summaries": strategy_summaries,
        "true_bidirectional_reference_diagnostics": true_reference_diagnostics,
        "numerical_audit": {
            "nll_identity": nll_identity_audit,
        },
        "source_hashes": source_hashes,
        "environment": environment,
        "reproducibility_manifest": {
            "canonicalization": (
                "UTF-8 JSON, sorted keys, compact separators, allow_nan=false"
            ),
            "sha256": manifest_sha256,
            "included_sections": [
                "schema_version",
                "protocol",
                "source_hashes",
                "environment",
                "utility_seeds",
                "group_oof_folds",
                "target_fit_query_candidate_coverage",
                "class_balance",
                "models",
            ],
        },
    }
    _assert_aggregate_output(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, output_path.resolve())
    return report


def _validate_runner_signature() -> None:
    names = tuple(inspect.signature(run_open_role_class_balanced_query_policy).parameters)
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
    if names != expected:
        raise AssertionError("class-balanced runner parameters changed")
    for name in names:
        if set(name.lower().split("_")) & FORBIDDEN_ROLE_TOKENS:
            raise AssertionError("class-balanced runner exposes a restricted role")


_validate_runner_signature()
