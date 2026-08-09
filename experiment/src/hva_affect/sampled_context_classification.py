"""Aggregate-only classification diagnostics for sampled utility contexts.

This module evaluates what a frozen candidate-selection score would do inside
the sampled S / S-plus-candidate / T / T-minus-candidate tasks used to train a
bidirectional utility model.  It deliberately does *not* implement the final
query-level history policy: several sampled candidate tasks may belong to one
query, so probabilities are first averaged within query, metrics are then
computed within cluster, and cluster metrics are finally macro-averaged.

Only in-memory arrays are accepted.  Probability provenance must be train-fold
OOF or train-fit-only, and the decision threshold must carry the exact
``fit_only_frozen`` provenance token.  No row prediction, query code, cluster
code, or label vector is returned by the public reports.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


CONTEXTS = ("s", "s_plus_candidate", "t", "t_minus_candidate")
PROBABILITY_MODES = ("train_fold_oof", "train_fit_only")
THRESHOLD_PROVENANCE = "fit_only_frozen"
EMOTION_CLASS_COUNT = 7
REPORT_SCHEMA_VERSION = "sampled_context_classification_diagnostic_v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SEALED_TOKENS = frozenset({"test", "validation", "holdout", "calibration", "sealed"})


class SampledContextDiagnosticError(ValueError):
    """Raised when diagnostic inputs break alignment or leakage contracts."""


def _sha256(value: str, *, field: str) -> str:
    digest = str(value).lower()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise SampledContextDiagnosticError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token}


def _reject_sealed_provenance(value: str, *, field: str) -> None:
    sealed = _tokens(value) & _SEALED_TOKENS
    if sealed:
        raise SampledContextDiagnosticError(
            f"{field} contains forbidden sealed-role token(s): {sorted(sealed)}"
        )


@dataclass(frozen=True)
class DiagnosticProvenance:
    """Train-only provenance shared by labels/codes and context probabilities."""

    probability_mode: str
    fit_scope_sha256: str
    task_order_sha256: str

    def __post_init__(self) -> None:
        _reject_sealed_provenance(self.probability_mode, field="probability_mode")
        if self.probability_mode not in PROBABILITY_MODES:
            raise SampledContextDiagnosticError(
                f"probability_mode must be one of {PROBABILITY_MODES}"
            )
        object.__setattr__(
            self,
            "fit_scope_sha256",
            _sha256(self.fit_scope_sha256, field="fit_scope_sha256"),
        )
        object.__setattr__(
            self,
            "task_order_sha256",
            _sha256(self.task_order_sha256, field="task_order_sha256"),
        )


@dataclass(frozen=True)
class SampledContextInputs:
    """Strictly row-aligned task labels, anonymized codes, and probabilities.

    Each context array may be ``(tasks, 7)`` or ``(base_seeds, tasks, 7)``.
    Internally every array is normalized to the latter form and stored as
    finite float64.  Labels and codes must be non-negative integer arrays.
    A query code must map to exactly one label and exactly one cluster code.
    """

    query_labels: np.ndarray
    query_codes: np.ndarray
    cluster_codes: np.ndarray
    context_probabilities: Mapping[str, np.ndarray]
    provenance: DiagnosticProvenance

    def __post_init__(self) -> None:
        labels = np.asarray(self.query_labels)
        queries = np.asarray(self.query_codes)
        clusters = np.asarray(self.cluster_codes)
        for name, values in (
            ("query_labels", labels),
            ("query_codes", queries),
            ("cluster_codes", clusters),
        ):
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise SampledContextDiagnosticError(f"{name} must be a one-dimensional integer array")
        rows = len(labels)
        if rows < 1 or len(queries) != rows or len(clusters) != rows:
            raise SampledContextDiagnosticError("labels, query codes, and cluster codes must align")
        if np.any((labels < 0) | (labels >= EMOTION_CLASS_COUNT)):
            raise SampledContextDiagnosticError(
                f"query_labels must lie in [0, {EMOTION_CLASS_COUNT - 1}]"
            )
        if np.any(queries < 0) or np.any(clusters < 0):
            raise SampledContextDiagnosticError("query and cluster codes must be non-negative")

        context_keys = set(self.context_probabilities)
        expected = set(CONTEXTS)
        if context_keys != expected:
            raise SampledContextDiagnosticError(
                "context probability schema mismatch: "
                f"missing={sorted(expected - context_keys)}, extra={sorted(context_keys - expected)}"
            )
        normalized: dict[str, np.ndarray] = {}
        shapes: set[tuple[int, int, int]] = set()
        for context in CONTEXTS:
            probability = np.asarray(self.context_probabilities[context], dtype=np.float64)
            if probability.ndim == 2:
                probability = probability[None, :, :]
            if probability.ndim != 3 or probability.shape[2] != EMOTION_CLASS_COUNT:
                raise SampledContextDiagnosticError(
                    f"{context} probabilities must have shape (tasks, 7) or (seeds, tasks, 7)"
                )
            if probability.shape[0] < 1 or probability.shape[1] != rows:
                raise SampledContextDiagnosticError(f"{context} probabilities are not task-aligned")
            if not np.isfinite(probability).all():
                raise SampledContextDiagnosticError(f"{context} probabilities contain non-finite values")
            if np.any(probability < -1e-12) or np.any(probability > 1.0 + 1e-12):
                raise SampledContextDiagnosticError(f"{context} probabilities must lie in [0, 1]")
            if not np.allclose(probability.sum(axis=2), 1.0, rtol=1e-7, atol=1e-9):
                raise SampledContextDiagnosticError(f"{context} probability rows must sum to one")
            copied = np.array(probability, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            normalized[context] = copied
            shapes.add(copied.shape)
        if len(shapes) != 1:
            raise SampledContextDiagnosticError(
                "all four context probability arrays must share seed/task/class shape"
            )

        labels = labels.astype(np.int64, copy=True)
        queries = queries.astype(np.int64, copy=True)
        clusters = clusters.astype(np.int64, copy=True)
        for query in np.unique(queries):
            mask = queries == query
            if len(np.unique(labels[mask])) != 1:
                raise SampledContextDiagnosticError("one query code maps to multiple labels")
            if len(np.unique(clusters[mask])) != 1:
                raise SampledContextDiagnosticError("one query code maps to multiple clusters")
        for array in (labels, queries, clusters):
            array.setflags(write=False)
        object.__setattr__(self, "query_labels", labels)
        object.__setattr__(self, "query_codes", queries)
        object.__setattr__(self, "cluster_codes", clusters)
        object.__setattr__(self, "context_probabilities", MappingProxyType(normalized))

    @property
    def rows(self) -> int:
        return len(self.query_labels)

    @property
    def seed_count(self) -> int:
        return self.context_probabilities[CONTEXTS[0]].shape[0]


@dataclass(frozen=True)
class FrozenUtilityDecision:
    """Candidate scores paired with a threshold learned only on fit data."""

    decision_scores: np.ndarray
    frozen_threshold: float
    task_order_sha256: str
    threshold_provenance: str = THRESHOLD_PROVENANCE

    def __post_init__(self) -> None:
        _reject_sealed_provenance(self.threshold_provenance, field="threshold_provenance")
        if self.threshold_provenance != THRESHOLD_PROVENANCE:
            raise SampledContextDiagnosticError(
                f"threshold_provenance must equal {THRESHOLD_PROVENANCE!r}"
            )
        scores = np.asarray(self.decision_scores, dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise SampledContextDiagnosticError("decision_scores must be one-dimensional and finite")
        threshold = float(self.frozen_threshold)
        if not np.isfinite(threshold):
            raise SampledContextDiagnosticError("frozen_threshold must be finite")
        copied = np.array(scores, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "decision_scores", copied)
        object.__setattr__(self, "frozen_threshold", threshold)
        object.__setattr__(
            self,
            "task_order_sha256",
            _sha256(self.task_order_sha256, field="task_order_sha256"),
        )

    def selected(self) -> np.ndarray:
        """Use a strict threshold: equality falls back to the safer endpoint."""

        return self.decision_scores > self.frozen_threshold


def _aligned_decision(inputs: SampledContextInputs, decision: FrozenUtilityDecision) -> np.ndarray:
    if len(decision.decision_scores) != inputs.rows:
        raise SampledContextDiagnosticError("decision scores and sampled tasks are not aligned")
    if decision.task_order_sha256 != inputs.provenance.task_order_sha256:
        raise SampledContextDiagnosticError(
            "decision scores and context probabilities have different task-order hashes"
        )
    return decision.selected()


def _ensemble_probabilities(inputs: SampledContextInputs) -> dict[str, np.ndarray]:
    return {
        context: np.mean(inputs.context_probabilities[context], axis=0, dtype=np.float64)
        for context in CONTEXTS
    }


def _policy_probabilities(
    probability: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    addition = np.where(
        selected[:, None],
        probability["s_plus_candidate"],
        probability["s"],
    )
    deletion = np.where(
        selected[:, None],
        probability["t"],
        probability["t_minus_candidate"],
    )
    return addition, deletion


@dataclass(frozen=True)
class _QueryAggregate:
    labels: np.ndarray
    clusters: np.ndarray
    probability: np.ndarray


def _aggregate_queries(inputs: SampledContextInputs, probability: np.ndarray) -> _QueryAggregate:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (inputs.rows, EMOTION_CLASS_COUNT):
        raise SampledContextDiagnosticError("diagnostic probability is not task-aligned")
    unique_queries = np.unique(inputs.query_codes)
    labels = np.empty(len(unique_queries), dtype=np.int64)
    clusters = np.empty(len(unique_queries), dtype=np.int64)
    aggregated = np.empty((len(unique_queries), EMOTION_CLASS_COUNT), dtype=np.float64)
    for index, query in enumerate(unique_queries):
        mask = inputs.query_codes == query
        labels[index] = inputs.query_labels[mask][0]
        clusters[index] = inputs.cluster_codes[mask][0]
        aggregated[index] = probability[mask].mean(axis=0)
    if not np.allclose(aggregated.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise AssertionError("query aggregation left the probability simplex")
    return _QueryAggregate(labels, clusters, aggregated)


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    # Match conventional macro-F1: average over classes observed in truth or
    # predictions; a class with zero precision/recall receives zero.
    classes = np.union1d(labels, predictions)
    scores: list[float] = []
    for class_index in classes:
        true_positive = np.sum((labels == class_index) & (predictions == class_index))
        false_positive = np.sum((labels != class_index) & (predictions == class_index))
        false_negative = np.sum((labels == class_index) & (predictions != class_index))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
    return float(np.mean(scores))


def _per_query_nll(aggregate: _QueryAggregate) -> np.ndarray:
    rows = np.arange(len(aggregate.labels))
    return -np.log(np.clip(aggregate.probability[rows, aggregate.labels], 1e-12, 1.0))


def _per_query_brier(aggregate: _QueryAggregate) -> np.ndarray:
    one_hot = np.eye(EMOTION_CLASS_COUNT, dtype=np.float64)[aggregate.labels]
    return np.sum((aggregate.probability - one_hot) ** 2, axis=1)


def _cluster_macro_mean(values: np.ndarray, clusters: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != clusters.shape:
        raise AssertionError("cluster aggregation inputs are misaligned")
    return float(np.mean([values[clusters == cluster].mean() for cluster in np.unique(clusters)]))


def _hierarchical_metrics(aggregate: _QueryAggregate) -> dict[str, float]:
    predictions = np.argmax(aggregate.probability, axis=1)
    cluster_metrics: list[dict[str, float]] = []
    nll = _per_query_nll(aggregate)
    brier = _per_query_brier(aggregate)
    for cluster in np.unique(aggregate.clusters):
        mask = aggregate.clusters == cluster
        cluster_metrics.append(
            {
                "macro_f1": _macro_f1(aggregate.labels[mask], predictions[mask]),
                "accuracy": float(np.mean(predictions[mask] == aggregate.labels[mask])),
                "nll": float(np.mean(nll[mask])),
                "brier": float(np.mean(brier[mask])),
            }
        )
    # The registered classification metric is computed once over the pooled
    # out-of-query predictions.  Dialogue clusters are the resampling unit, not
    # the definition of Macro-F1; averaging per-dialogue F1 would overweight
    # tiny, single-class dialogues and would not match the confirmatory contract.
    return {
        "macro_f1": _macro_f1(aggregate.labels, predictions),
        "accuracy": float(np.mean(predictions == aggregate.labels)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "cluster_macro_macro_f1": float(
            np.mean([values["macro_f1"] for values in cluster_metrics])
        ),
        "cluster_macro_accuracy": float(
            np.mean([values["accuracy"] for values in cluster_metrics])
        ),
        "cluster_macro_nll": float(np.mean([values["nll"] for values in cluster_metrics])),
        "cluster_macro_brier": float(
            np.mean([values["brier"] for values in cluster_metrics])
        ),
    }


def _endpoint_comparison(
    policy: _QueryAggregate,
    endpoint: _QueryAggregate,
) -> dict[str, float]:
    if not np.array_equal(policy.labels, endpoint.labels) or not np.array_equal(
        policy.clusters, endpoint.clusters
    ):
        raise AssertionError("paired endpoint aggregates are misaligned")
    delta = _per_query_nll(policy) - _per_query_nll(endpoint)
    return {
        "nll_regret": _cluster_macro_mean(delta, policy.clusters),
        "nll_harm_rate": _cluster_macro_mean((delta > 0.0).astype(float), policy.clusters),
    }


def _branch_report(
    inputs: SampledContextInputs,
    policy_probability: np.ndarray,
    endpoints: Mapping[str, np.ndarray],
) -> dict[str, object]:
    policy = _aggregate_queries(inputs, policy_probability)
    endpoint_aggregates = {
        name: _aggregate_queries(inputs, probability) for name, probability in endpoints.items()
    }
    return {
        "policy_metrics": _hierarchical_metrics(policy),
        "fixed_endpoint_metrics": {
            name: _hierarchical_metrics(aggregate)
            for name, aggregate in endpoint_aggregates.items()
        },
        "relative_to_fixed_endpoints": {
            name: _endpoint_comparison(policy, aggregate)
            for name, aggregate in endpoint_aggregates.items()
        },
    }


def _selected_rate_by_query_cluster(
    inputs: SampledContextInputs,
    selected: np.ndarray,
) -> float:
    query_rates: list[float] = []
    query_clusters: list[int] = []
    for query in np.unique(inputs.query_codes):
        mask = inputs.query_codes == query
        query_rates.append(float(np.mean(selected[mask])))
        query_clusters.append(int(inputs.cluster_codes[mask][0]))
    return _cluster_macro_mean(
        np.asarray(query_rates, dtype=np.float64),
        np.asarray(query_clusters, dtype=np.int64),
    )


def _assert_aggregate_only(value: object) -> None:
    if isinstance(value, np.ndarray):
        raise AssertionError("public diagnostic reports must not contain row arrays")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_aggregate_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_aggregate_only(item)


def sampled_context_classification_diagnostic(
    inputs: SampledContextInputs,
    decision: FrozenUtilityDecision,
) -> dict[str, object]:
    """Evaluate addition/deletion choices and return aggregate diagnostics only."""

    selected = _aligned_decision(inputs, decision)
    probability = _ensemble_probabilities(inputs)
    addition, deletion = _policy_probabilities(probability, selected)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "sampled_context_diagnostic_only",
        "claim_boundary": (
            "This is a sampled-context task diagnostic, not a final query-level history policy "
            "and not evidence of sealed-test improvement."
        ),
        "aggregation": {
            "order": "mean probabilities within query; pooled query classification metrics plus cluster-macro diagnostics",
            "macro_f1_classes": "classes observed in pooled query truth or prediction",
            "cluster_role": "whole-dialogue resampling unit and secondary cluster-macro diagnostic",
            "brier_definition": "multiclass sum of squared probability error",
            "base_seed_reduction": "mean probability before policy selection",
        },
        "counts": {
            "sampled_tasks": int(inputs.rows),
            "unique_queries": int(len(np.unique(inputs.query_codes))),
            "clusters": int(len(np.unique(inputs.cluster_codes))),
            "base_seeds": int(inputs.seed_count),
            "emotion_classes": EMOTION_CLASS_COUNT,
        },
        "decision": {
            "frozen_threshold": float(decision.frozen_threshold),
            "threshold_provenance": decision.threshold_provenance,
            "selected_tasks": int(np.sum(selected)),
            "query_cluster_macro_selected_rate": _selected_rate_by_query_cluster(
                inputs, selected
            ),
        },
        "addition": {
            "rule": "score > frozen_threshold selects S+h; otherwise S",
            **_branch_report(
                inputs,
                addition,
                {"always_s": probability["s"], "always_s_plus_candidate": probability["s_plus_candidate"]},
            ),
        },
        "deletion": {
            "rule": "score > frozen_threshold retains T; otherwise uses T-h",
            **_branch_report(
                inputs,
                deletion,
                {"always_t": probability["t"], "always_t_minus_candidate": probability["t_minus_candidate"]},
            ),
        },
        "provenance": {
            "probability_mode": inputs.provenance.probability_mode,
            "fit_scope_sha256": inputs.provenance.fit_scope_sha256,
            "task_order_sha256": inputs.provenance.task_order_sha256,
            "contains_row_identifiers": False,
            "contains_row_predictions": False,
        },
    }
    _assert_aggregate_only(report)
    return report


def _paired_branch(
    inputs: SampledContextInputs,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
) -> dict[str, object]:
    aggregate_a = _aggregate_queries(inputs, probability_a)
    aggregate_b = _aggregate_queries(inputs, probability_b)
    if not np.array_equal(aggregate_a.labels, aggregate_b.labels) or not np.array_equal(
        aggregate_a.clusters, aggregate_b.clusters
    ):
        raise AssertionError("paired model query aggregates are misaligned")
    metrics_a = _hierarchical_metrics(aggregate_a)
    metrics_b = _hierarchical_metrics(aggregate_b)
    nll_delta = _per_query_nll(aggregate_a) - _per_query_nll(aggregate_b)
    return {
        "model_a_metrics": metrics_a,
        "model_b_metrics": metrics_b,
        "metric_delta_a_minus_b": {
            name: float(metrics_a[name] - metrics_b[name]) for name in metrics_a
        },
        "paired_nll": {
            "a_minus_b": _cluster_macro_mean(nll_delta, aggregate_a.clusters),
            "a_harm_rate_vs_b": _cluster_macro_mean(
                (nll_delta > 0.0).astype(float), aggregate_a.clusters
            ),
            "a_win_rate_vs_b": _cluster_macro_mean(
                (nll_delta < 0.0).astype(float), aggregate_a.clusters
            ),
            "tie_rate": _cluster_macro_mean(
                np.isclose(nll_delta, 0.0, rtol=0.0, atol=1e-12).astype(float),
                aggregate_a.clusters,
            ),
        },
    }


def paired_sampled_context_model_contrast(
    inputs: SampledContextInputs,
    model_a: FrozenUtilityDecision,
    model_b: FrozenUtilityDecision,
) -> dict[str, object]:
    """Paired, aggregate-only contrast of two in-memory utility decisions."""

    selected_a = _aligned_decision(inputs, model_a)
    selected_b = _aligned_decision(inputs, model_b)
    probability = _ensemble_probabilities(inputs)
    addition_a, deletion_a = _policy_probabilities(probability, selected_a)
    addition_b, deletion_b = _policy_probabilities(probability, selected_b)
    disagreement = np.not_equal(selected_a, selected_b)
    report: dict[str, object] = {
        "schema_version": "sampled_context_paired_model_contrast_v1",
        "status": "sampled_context_diagnostic_only",
        "claim_boundary": (
            "This paired contrast uses sampled contexts only; it is not a final query-level "
            "policy comparison and accepts only in-memory open-role arrays, never sealed data."
        ),
        "decision_contrast": {
            "model_a_threshold": float(model_a.frozen_threshold),
            "model_b_threshold": float(model_b.frozen_threshold),
            "query_cluster_macro_disagreement_rate": _selected_rate_by_query_cluster(
                inputs, disagreement
            ),
        },
        "addition": _paired_branch(inputs, addition_a, addition_b),
        "deletion": _paired_branch(inputs, deletion_a, deletion_b),
        "counts": {
            "sampled_tasks": int(inputs.rows),
            "unique_queries": int(len(np.unique(inputs.query_codes))),
            "clusters": int(len(np.unique(inputs.cluster_codes))),
            "base_seeds": int(inputs.seed_count),
        },
        "provenance": {
            "probability_mode": inputs.provenance.probability_mode,
            "threshold_provenance": THRESHOLD_PROVENANCE,
            "task_order_sha256": inputs.provenance.task_order_sha256,
            "contains_row_predictions": False,
        },
    }
    _assert_aggregate_only(report)
    return report


def array_order_sha256(*arrays: np.ndarray) -> str:
    """Create an alignment digest for synthetic/in-memory task arrays.

    Production callers should prefer the coalition-task digest used when the
    59-D cache is constructed.  This helper is convenient when no task objects
    exist, and hashes dtype, shape, and C-order bytes without retaining values.
    """

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
