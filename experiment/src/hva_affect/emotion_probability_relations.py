"""Lineage-verified emotion-probability relations for CARMA-Affect.

This module consumes seven-class *predicted* probabilities and never accepts a
target or gold-emotion array.  That API boundary is useful but is not, by
itself, proof against supervision leakage: an upstream producer could disguise
gold labels as one-hot probabilities.  Safety therefore rests on a complete,
frozen lineage shared by the probability producer and the verified 59-D base
cache.  Fit rows must be group out-of-fold and model-selection rows must be
predicted by a model fit only on the frozen fit role.

The 3x3 grid measures disagreement between current-modality and
history-modality predictive distributions.  It is an interpretable engineering
feature and ablation target; it is **not** evidence of an emotion theory,
psychological mechanism, causal interaction, or causal effect.  Such claims
require separate theory variables and an appropriate identification design.

No function reads a path or downloads a model.  ``verify_base_59d_cache``
accepts cache bytes so callers can keep file access in their audited boundary
and tests can remain fully synthetic.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "video")
HISTORY_CONTEXTS = ("candidate", "s", "t", "t_minus_candidate")
BASE_PROBABILITY_CONTEXTS = ("s", "s_plus_candidate", "t", "t_minus_candidate")
PROVENANCE_MODES = ("train_fold_oof", "train_fit_only")
MODE_TO_ROLE = {
    "train_fold_oof": "base_and_utility_fit",
    "train_fit_only": "model_selection",
}
ROLE_TO_CACHE_FIELD = {
    "base_and_utility_fit": "fit_x",
    "model_selection": "selection_x",
}
EMOTION_CLASS_COUNT = 7
RELATION_METRICS = ("cosine", "l2", "mean_absolute_delta")
BASE_CACHE_FEATURE_COUNT = 59
BASE_CACHE_SCHEMA_VERSION = "emotiontalk_bidirectional_oof_cache_v1"
PROBABILITY_SUM_TOLERANCE = 1.0e-5
PROBABILITY_BOUND_TOLERANCE = 1.0e-6

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_NAME_TOKENS = frozenset(
    {"gold", "label", "labels", "target", "targets", "ytrue", "groundtruth"}
)
_VERIFIED_CACHE_TOKEN = object()


def _base_cache_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for context in BASE_PROBABILITY_CONTEXTS:
        names.extend((f"{context}_confidence", f"{context}_entropy"))
        names.extend(f"{context}_probability_{index}" for index in range(EMOTION_CLASS_COUNT))
    for direction in ("forward", "backward"):
        names.extend((f"{direction}_probability_l1", f"{direction}_probability_l2"))
        names.extend(
            f"{direction}_probability_delta_{index}" for index in range(EMOTION_CLASS_COUNT)
        )
    names.extend(
        (
            "log_addition_context_count",
            "log_deletion_context_count",
            "addition_deletion_jaccard",
            "log_full_history_count",
            "candidate_recency_fraction",
        )
    )
    if len(names) != BASE_CACHE_FEATURE_COUNT or len(set(names)) != len(names):
        raise AssertionError("canonical 59-D feature schema changed")
    return tuple(names)


BASE_CACHE_FEATURE_NAMES = _base_cache_feature_names()


class EmotionRelationContractError(ValueError):
    """Raised when probability, provenance, or row alignment is unsafe."""


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, *, field: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise EmotionRelationContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def dataset_identity_sha256(dataset: str) -> str:
    """Hash a non-empty, versioned dataset identifier."""

    identifier = str(dataset).strip()
    if not identifier:
        raise EmotionRelationContractError("dataset must be a non-empty versioned identifier")
    return _canonical_json_sha256({"dataset": identifier})


def ordered_source_sha256(dataset: str, ordered_source_ids: Sequence[str]) -> str:
    """Hash stable source identifiers in the exact integer-index order."""

    values = tuple(str(value) for value in ordered_source_ids)
    if not values or any(not value for value in values):
        raise EmotionRelationContractError("ordered source identifiers must be non-empty strings")
    if len(values) != len(set(values)):
        raise EmotionRelationContractError("ordered source identifiers must be unique")
    return _canonical_json_sha256(
        {"dataset": str(dataset).strip(), "ordered_source_ids": values}
    )


def emotion_class_order_sha256(class_order: Sequence[str]) -> str:
    values = tuple(str(value) for value in class_order)
    if len(values) != EMOTION_CLASS_COUNT or len(set(values)) != len(values):
        raise EmotionRelationContractError(
            f"class_order must contain {EMOTION_CLASS_COUNT} unique emotion classes"
        )
    return _canonical_json_sha256({"ordered_emotion_classes": values})


def emotion_context_schema_sha256() -> str:
    """Return the frozen joint context semantics shared with the 59-D cache."""

    descriptor = {
        "schema": "carma_emotion_probability_relations_v2",
        "base_probability_contexts": BASE_PROBABILITY_CONTEXTS,
        "emotion_state_contexts": ("current", *HISTORY_CONTEXTS),
        "modalities": MODALITIES,
        "relation_metrics": RELATION_METRICS,
        "candidate_semantics": "strict-past_candidate",
        "addition_semantics": "S_and_S_union_candidate",
        "deletion_semantics": "T_and_T_minus_candidate",
    }
    return _canonical_json_sha256(descriptor)


def feature_names_content_sha256(names: Sequence[str]) -> str:
    values = tuple(str(value) for value in names)
    return _canonical_json_sha256({"ordered_feature_names": values})


def numeric_matrix_content_sha256(matrix: np.ndarray) -> str:
    """Hash shape, dtype, and C-order bytes of an exact numeric matrix."""

    values = np.asarray(matrix)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise EmotionRelationContractError("matrix content hash requires a numeric 2-D array")
    contiguous = np.ascontiguousarray(values)
    header = json.dumps(
        {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _name_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(value).lower())
        if token
    }


def _reject_supervision_names(names: Sequence[str], *, field: str) -> None:
    unsafe = sorted(
        str(name)
        for name in names
        if _name_tokens(str(name)) & _FORBIDDEN_NAME_TOKENS
    )
    if unsafe:
        raise EmotionRelationContractError(
            f"{field} contains forbidden supervision fields: {unsafe}"
        )


@dataclass(frozen=True)
class TrainOnlyProvenance:
    """Complete role and artifact lineage for one probability feature block."""

    mode: str
    dataset: str
    role: str
    dataset_sha256: str
    source_order_sha256: str
    split_manifest_sha256: str
    fold_assignment_sha256: str
    task_order_sha256: str
    context_schema_sha256: str
    class_order_sha256: str
    producer_config_sha256: str

    def __post_init__(self) -> None:
        dataset = str(self.dataset).strip()
        if not dataset:
            raise EmotionRelationContractError("dataset must be a non-empty versioned identifier")
        if self.mode not in PROVENANCE_MODES:
            raise EmotionRelationContractError(
                f"provenance mode must be one of {PROVENANCE_MODES}, got {self.mode!r}"
            )
        expected_role = MODE_TO_ROLE[self.mode]
        if self.role != expected_role:
            raise EmotionRelationContractError(
                f"{self.mode} provenance requires role {expected_role!r}, got {self.role!r}"
            )
        object.__setattr__(self, "dataset", dataset)
        for name in (
            "dataset_sha256",
            "source_order_sha256",
            "split_manifest_sha256",
            "fold_assignment_sha256",
            "task_order_sha256",
            "context_schema_sha256",
            "class_order_sha256",
            "producer_config_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), field=name),
            )
        if self.dataset_sha256 != dataset_identity_sha256(dataset):
            raise EmotionRelationContractError("dataset identifier and dataset_sha256 disagree")
        if self.context_schema_sha256 != emotion_context_schema_sha256():
            raise EmotionRelationContractError("context_schema_sha256 is not the frozen v2 schema")


def _task_field(task: object, name: str) -> object:
    if isinstance(task, Mapping):
        if name not in task:
            raise EmotionRelationContractError(f"task mapping lacks {name}")
        return task[name]
    if not hasattr(task, name):
        raise EmotionRelationContractError(f"task object lacks {name}")
    return getattr(task, name)


def bidirectional_task_order_sha256(
    tasks: Sequence[object],
    *,
    dataset: str,
    role: str,
    source_order_sha256: str,
    split_manifest_sha256: str,
    fold_assignment_sha256: str,
    context_schema_sha256: str,
    class_order_sha256: str,
    producer_config_sha256: str,
) -> str:
    """Hash ordered tasks plus every lineage dimension needed for alignment.

    Task row order remains significant.  Coalition members are canonicalized as
    sorted sets because S and T are sets and their aggregation is permutation
    invariant.  ``source_order_sha256`` binds integer row indices to stable
    source identities without retaining those identities in the feature file.
    """

    dataset_value = str(dataset).strip()
    if not dataset_value:
        raise EmotionRelationContractError("dataset must be non-empty")
    if role not in ROLE_TO_CACHE_FIELD:
        raise EmotionRelationContractError(f"unsupported task role {role!r}")
    bindings = {
        "dataset": dataset_value,
        "dataset_sha256": dataset_identity_sha256(dataset_value),
        "role": role,
        "source_order_sha256": _validate_sha256(
            source_order_sha256, field="source_order_sha256"
        ),
        "split_manifest_sha256": _validate_sha256(
            split_manifest_sha256, field="split_manifest_sha256"
        ),
        "fold_assignment_sha256": _validate_sha256(
            fold_assignment_sha256, field="fold_assignment_sha256"
        ),
        "context_schema_sha256": _validate_sha256(
            context_schema_sha256, field="context_schema_sha256"
        ),
        "class_order_sha256": _validate_sha256(
            class_order_sha256, field="class_order_sha256"
        ),
        "producer_config_sha256": _validate_sha256(
            producer_config_sha256, field="producer_config_sha256"
        ),
    }
    if bindings["context_schema_sha256"] != emotion_context_schema_sha256():
        raise EmotionRelationContractError("task digest uses the wrong context schema")

    canonical_tasks: list[list[object]] = []
    for task in tasks:
        addition = sorted({int(value) for value in _task_field(task, "addition_context")})
        deletion = sorted({int(value) for value in _task_field(task, "deletion_context")})
        canonical_tasks.append(
            [
                int(_task_field(task, "query_index")),
                addition,
                deletion,
                int(_task_field(task, "candidate_index")),
            ]
        )
    return _canonical_json_sha256(
        {
            "schema": "carma_bidirectional_task_order_v2",
            "bindings": bindings,
            "ordered_tasks": canonical_tasks,
        }
    )


@dataclass(frozen=True)
class EmotionProbabilityBlock:
    """Three probability matrices with explicit source-column declarations."""

    probabilities: Mapping[str, np.ndarray]
    provenance: TrainOnlyProvenance
    class_order: tuple[str, ...]
    modality_class_orders: Mapping[str, tuple[str, ...]]
    normalization_max_correction: Mapping[str, float] = field(init=False)
    max_normalization_correction: float = field(init=False)

    def __post_init__(self) -> None:
        expected_modalities = set(MODALITIES)
        probability_keys = set(self.probabilities)
        if probability_keys != expected_modalities:
            raise EmotionRelationContractError(
                "probability modalities mismatch: "
                f"missing={sorted(expected_modalities - probability_keys)}, "
                f"extra={sorted(probability_keys - expected_modalities)}"
            )
        order_keys = set(self.modality_class_orders)
        if order_keys != expected_modalities:
            raise EmotionRelationContractError(
                "modality class-order declarations mismatch: "
                f"missing={sorted(expected_modalities - order_keys)}, "
                f"extra={sorted(order_keys - expected_modalities)}"
            )

        class_order = tuple(str(value) for value in self.class_order)
        class_hash = emotion_class_order_sha256(class_order)
        _reject_supervision_names(class_order, field="class_order")
        if class_hash != self.provenance.class_order_sha256:
            raise EmotionRelationContractError(
                "class_order and provenance class_order_sha256 disagree"
            )
        validated_orders: dict[str, tuple[str, ...]] = {}
        for modality in MODALITIES:
            actual_order = tuple(str(value) for value in self.modality_class_orders[modality])
            if actual_order != class_order:
                raise EmotionRelationContractError(
                    f"{modality} probability columns do not match canonical class_order"
                )
            validated_orders[modality] = actual_order

        validated: dict[str, np.ndarray] = {}
        corrections: dict[str, float] = {}
        row_counts: set[int] = set()
        for modality in MODALITIES:
            raw = np.asarray(self.probabilities[modality])
            if raw.ndim != 2 or raw.shape[1] != EMOTION_CLASS_COUNT:
                raise EmotionRelationContractError(
                    f"{modality} probabilities must have shape (rows, {EMOTION_CLASS_COUNT})"
                )
            if raw.shape[0] == 0:
                raise EmotionRelationContractError("probability blocks must contain at least one row")
            values = np.asarray(raw, dtype=np.float64)
            if not np.isfinite(values).all():
                raise EmotionRelationContractError(
                    f"{modality} probabilities contain non-finite values"
                )
            if np.any(values < -PROBABILITY_BOUND_TOLERANCE) or np.any(
                values > 1.0 + PROBABILITY_BOUND_TOLERANCE
            ):
                raise EmotionRelationContractError(
                    f"{modality} probabilities are materially outside [0, 1]"
                )
            clipped = np.clip(values, 0.0, 1.0)
            row_sums = clipped.sum(axis=1)
            if np.any(np.abs(row_sums - 1.0) > PROBABILITY_SUM_TOLERANCE):
                raise EmotionRelationContractError(
                    f"{modality} probability sums exceed tolerance "
                    f"{PROBABILITY_SUM_TOLERANCE:g}"
                )
            normalized = clipped / row_sums[:, None]
            correction = float(np.max(np.abs(normalized - values)))
            copied = np.array(normalized, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            validated[modality] = copied
            corrections[modality] = correction
            row_counts.add(len(copied))
        if len(row_counts) != 1:
            raise EmotionRelationContractError("modality probability rows are not aligned")

        object.__setattr__(self, "probabilities", MappingProxyType(validated))
        object.__setattr__(self, "class_order", class_order)
        object.__setattr__(self, "modality_class_orders", MappingProxyType(validated_orders))
        object.__setattr__(
            self,
            "normalization_max_correction",
            MappingProxyType(corrections),
        )
        object.__setattr__(self, "max_normalization_correction", max(corrections.values()))

    @property
    def rows(self) -> int:
        return len(self.probabilities[MODALITIES[0]])


@dataclass(frozen=True)
class EmotionRelationFeatureBundle:
    """Master feature matrix, ablation groups, and unabridged provenance."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    column_groups: Mapping[str, tuple[int, ...]]
    provenance: TrainOnlyProvenance
    class_order: tuple[str, ...]
    max_normalization_correction: float

    def select(self, group: str) -> tuple[np.ndarray, tuple[str, ...]]:
        if group not in self.column_groups:
            raise EmotionRelationContractError(
                f"unknown emotion feature group {group!r}; "
                f"available={sorted(self.column_groups)}"
            )
        columns = self.column_groups[group]
        selected = np.asarray(self.matrix[:, columns], dtype=np.float64)
        names = tuple(self.feature_names[index] for index in columns)
        return selected, names


@dataclass(frozen=True)
class BaseCacheLineage:
    """Frozen sidecar describing one role-specific matrix in a cache payload."""

    schema_version: str
    row_count: int
    cache_sha256: str
    matrix_content_sha256: str
    feature_names_content_sha256: str
    provenance: TrainOnlyProvenance
    class_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != BASE_CACHE_SCHEMA_VERSION:
            raise EmotionRelationContractError(
                f"base cache schema must be {BASE_CACHE_SCHEMA_VERSION!r}"
            )
        if int(self.row_count) <= 0:
            raise EmotionRelationContractError("base cache row_count must be positive")
        object.__setattr__(self, "row_count", int(self.row_count))
        for name in (
            "cache_sha256",
            "matrix_content_sha256",
            "feature_names_content_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), field=name),
            )
        class_order = tuple(str(value) for value in self.class_order)
        if emotion_class_order_sha256(class_order) != self.provenance.class_order_sha256:
            raise EmotionRelationContractError(
                "base cache class_order and provenance class-order hash disagree"
            )
        object.__setattr__(self, "class_order", class_order)
        canonical_names_hash = feature_names_content_sha256(BASE_CACHE_FEATURE_NAMES)
        if self.feature_names_content_sha256 != canonical_names_hash:
            raise EmotionRelationContractError(
                "base cache feature-name hash is not the canonical 59-D schema"
            )


def base_cache_lineage_sha256(lineage: BaseCacheLineage) -> str:
    """Digest a lineage sidecar for pinning before cache verification."""

    provenance = lineage.provenance
    return _canonical_json_sha256(
        {
            "schema_version": lineage.schema_version,
            "row_count": lineage.row_count,
            "cache_sha256": lineage.cache_sha256,
            "matrix_content_sha256": lineage.matrix_content_sha256,
            "feature_names_content_sha256": lineage.feature_names_content_sha256,
            "class_order": lineage.class_order,
            "provenance": {
                field_name: getattr(provenance, field_name)
                for field_name in provenance.__dataclass_fields__
            },
        }
    )


@dataclass(frozen=True)
class VerifiedBaseTaskCache:
    """A role-specific 59-D matrix constructible only by cache verification."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    lineage: BaseCacheLineage
    lineage_sha256: str
    _verification_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification_token is not _VERIFIED_CACHE_TOKEN:
            raise EmotionRelationContractError(
                "VerifiedBaseTaskCache must be created by verify_base_59d_cache"
            )


def _npz_scalar_text(archive: Mapping[str, np.ndarray], field_name: str) -> str:
    if field_name not in archive:
        raise EmotionRelationContractError(f"base cache lacks {field_name}")
    values = np.asarray(archive[field_name])
    if values.size != 1:
        raise EmotionRelationContractError(f"base cache {field_name} must be scalar")
    return str(values.reshape(-1)[0])


def verify_base_59d_cache(
    cache_payload: bytes,
    *,
    lineage: BaseCacheLineage,
    expected_lineage_sha256: str,
) -> VerifiedBaseTaskCache:
    """Verify raw NPZ bytes against a separately pinned lineage manifest."""

    expected_lineage = _validate_sha256(
        expected_lineage_sha256,
        field="expected_lineage_sha256",
    )
    actual_lineage = base_cache_lineage_sha256(lineage)
    if actual_lineage != expected_lineage:
        raise EmotionRelationContractError("base cache lineage does not match the frozen digest")
    if not isinstance(cache_payload, bytes):
        raise TypeError("cache_payload must be immutable bytes")
    if hashlib.sha256(cache_payload).hexdigest() != lineage.cache_sha256:
        raise EmotionRelationContractError("base cache payload SHA-256 mismatch")

    try:
        with np.load(io.BytesIO(cache_payload), allow_pickle=False) as archive:
            schema_version = _npz_scalar_text(archive, "schema_version")
            if schema_version != lineage.schema_version:
                raise EmotionRelationContractError("base cache schema version mismatch")
            matrix_field = ROLE_TO_CACHE_FIELD[lineage.provenance.role]
            if matrix_field not in archive or "feature_names" not in archive:
                raise EmotionRelationContractError(
                    f"base cache lacks {matrix_field} or feature_names"
                )
            matrix = np.asarray(archive[matrix_field])
            names = tuple(str(value) for value in np.asarray(archive["feature_names"]).tolist())
    except EmotionRelationContractError:
        raise
    except Exception as error:
        raise EmotionRelationContractError(f"invalid base cache NPZ payload: {error}") from error

    if matrix.dtype != np.float64:
        raise EmotionRelationContractError("base task cache must retain exact float64 dtype")
    if matrix.shape != (lineage.row_count, BASE_CACHE_FEATURE_COUNT):
        raise EmotionRelationContractError(
            f"base task cache must have shape ({lineage.row_count}, {BASE_CACHE_FEATURE_COUNT})"
        )
    if not np.isfinite(matrix).all():
        raise EmotionRelationContractError("base task cache contains non-finite values")
    if names != BASE_CACHE_FEATURE_NAMES:
        raise EmotionRelationContractError(
            "base feature names are not the canonical 59-D task schema"
        )
    if feature_names_content_sha256(names) != lineage.feature_names_content_sha256:
        raise EmotionRelationContractError("base feature-name content hash mismatch")
    if numeric_matrix_content_sha256(matrix) != lineage.matrix_content_sha256:
        raise EmotionRelationContractError("base matrix content hash mismatch")

    copied = np.array(matrix, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return VerifiedBaseTaskCache(
        matrix=copied,
        feature_names=names,
        lineage=lineage,
        lineage_sha256=actual_lineage,
        _verification_token=_VERIFIED_CACHE_TOKEN,
    )


@dataclass(frozen=True)
class CacheAlignedFeatureMatrix:
    """Verified base features joined to one emotion ablation group."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    provenance: TrainOnlyProvenance
    base_cache_lineage_sha256: str
    emotion_group: str


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-15, None)


def _validate_blocks(
    current: EmotionProbabilityBlock,
    history: Mapping[str, EmotionProbabilityBlock],
) -> tuple[EmotionProbabilityBlock, ...]:
    contexts = set(history)
    expected = set(HISTORY_CONTEXTS)
    if contexts != expected:
        raise EmotionRelationContractError(
            "history contexts mismatch: "
            f"missing={sorted(expected - contexts)}, extra={sorted(contexts - expected)}"
        )
    blocks = (current, *(history[context] for context in HISTORY_CONTEXTS))
    if len({block.rows for block in blocks}) != 1:
        raise EmotionRelationContractError("current and history rows are not aligned")
    if len({block.class_order for block in blocks}) != 1:
        raise EmotionRelationContractError(
            "all modalities and contexts must use the same ordered seven-class emotion space"
        )
    if any(block.provenance != current.provenance for block in blocks[1:]):
        raise EmotionRelationContractError(
            "probability blocks have inconsistent dataset/role/source/split/fold/task/"
            "context/class/producer provenance"
        )
    return blocks


def build_emotion_probability_relations(
    current: EmotionProbabilityBlock,
    history: Mapping[str, EmotionProbabilityBlock],
) -> EmotionRelationFeatureBundle:
    """Build raw concatenation plus candidate/S/T/T-h 3x3 discrepancies.

    ``mean_absolute_delta`` is L1 distance divided by seven.  A signed mean
    difference would be identically zero because both vectors lie on the same
    probability simplex.
    """

    blocks = _validate_blocks(current, history)
    values: list[np.ndarray] = []
    names: list[str] = []

    for modality in MODALITIES:
        probability = current.probabilities[modality]
        for class_index in range(EMOTION_CLASS_COUNT):
            values.append(probability[:, class_index])
            names.append(f"emotion_concat__current_{modality}__probability_{class_index}")
    for context in HISTORY_CONTEXTS:
        for modality in MODALITIES:
            probability = history[context].probabilities[modality]
            for class_index in range(EMOTION_CLASS_COUNT):
                values.append(probability[:, class_index])
                names.append(
                    f"emotion_concat__{context}_history_{modality}__probability_{class_index}"
                )
    simple_concat = tuple(range(len(values)))

    relation_start = len(values)
    same_modality: list[int] = []
    full_relations: list[int] = []
    context_groups: dict[str, list[int]] = {context: [] for context in HISTORY_CONTEXTS}
    for context in HISTORY_CONTEXTS:
        for current_modality in MODALITIES:
            current_probability = current.probabilities[current_modality]
            for history_modality in MODALITIES:
                history_probability = history[context].probabilities[history_modality]
                relation_values = (
                    _row_cosine(current_probability, history_probability),
                    np.linalg.norm(current_probability - history_probability, axis=1),
                    np.mean(np.abs(current_probability - history_probability), axis=1),
                )
                for metric, feature in zip(RELATION_METRICS, relation_values):
                    index = len(values)
                    values.append(feature)
                    names.append(
                        f"emotion_relation__{context}__current_{current_modality}"
                        f"__history_{history_modality}__{metric}"
                    )
                    full_relations.append(index)
                    context_groups[context].append(index)
                    if current_modality == history_modality:
                        same_modality.append(index)

    matrix = np.column_stack(values).astype(np.float64, copy=False)
    expected_concat = (1 + len(HISTORY_CONTEXTS)) * len(MODALITIES) * EMOTION_CLASS_COUNT
    expected_relation = (
        len(HISTORY_CONTEXTS) * len(MODALITIES) * len(MODALITIES) * len(RELATION_METRICS)
    )
    if len(simple_concat) != expected_concat or len(full_relations) != expected_relation:
        raise AssertionError("emotion relation feature geometry changed")
    if relation_start != expected_concat or matrix.shape[1] != expected_concat + expected_relation:
        raise AssertionError("emotion relation master matrix width changed")
    if matrix.dtype != np.float64 or not np.isfinite(matrix).all():
        raise EmotionRelationContractError("emotion relation features must be finite float64")
    if len(names) != len(set(names)):
        raise AssertionError("emotion relation feature names must be unique")
    _reject_supervision_names(names, field="generated feature names")
    matrix.setflags(write=False)

    groups: dict[str, tuple[int, ...]] = {
        "simple_concat": simple_concat,
        "same_modality_3cell": tuple(same_modality),
        "full_9cell": tuple(full_relations),
        "simple_concat_plus_same_modality": simple_concat + tuple(same_modality),
        "simple_concat_plus_full_9cell": simple_concat + tuple(full_relations),
        "all": tuple(range(matrix.shape[1])),
    }
    for context, indices in context_groups.items():
        groups[f"{context}_full_9cell"] = tuple(indices)

    return EmotionRelationFeatureBundle(
        matrix=matrix,
        feature_names=tuple(names),
        column_groups=MappingProxyType(groups),
        provenance=current.provenance,
        class_order=current.class_order,
        max_normalization_correction=max(
            block.max_normalization_correction for block in blocks
        ),
    )


def align_with_59d_task_cache(
    base_cache: VerifiedBaseTaskCache,
    *,
    emotion_features: EmotionRelationFeatureBundle,
    emotion_group: str,
) -> CacheAlignedFeatureMatrix:
    """Join only a verified canonical cache with exactly matching lineage."""

    if not isinstance(base_cache, VerifiedBaseTaskCache):
        raise TypeError("base_cache must be a VerifiedBaseTaskCache")
    if base_cache._verification_token is not _VERIFIED_CACHE_TOKEN:
        raise EmotionRelationContractError("base cache verification token is invalid")
    if base_cache.lineage.provenance != emotion_features.provenance:
        raise EmotionRelationContractError(
            "emotion features and base cache differ in dataset/role/source/split/fold/task/"
            "context/class/producer provenance"
        )
    if base_cache.lineage.class_order != emotion_features.class_order:
        raise EmotionRelationContractError("emotion and base cache class orders differ")
    if base_cache.feature_names != BASE_CACHE_FEATURE_NAMES:
        raise EmotionRelationContractError("verified cache feature schema changed after loading")
    if numeric_matrix_content_sha256(base_cache.matrix) != base_cache.lineage.matrix_content_sha256:
        raise EmotionRelationContractError("verified base matrix was altered after loading")

    emotion_x, emotion_names = emotion_features.select(emotion_group)
    if len(base_cache.matrix) != len(emotion_x):
        raise EmotionRelationContractError(
            "emotion features and 59-D cache have different row counts"
        )
    if set(base_cache.feature_names) & set(emotion_names):
        raise EmotionRelationContractError("base and emotion feature names overlap")
    combined = np.column_stack([base_cache.matrix, emotion_x]).astype(np.float64, copy=False)
    combined.setflags(write=False)
    return CacheAlignedFeatureMatrix(
        matrix=combined,
        feature_names=base_cache.feature_names + emotion_names,
        provenance=emotion_features.provenance,
        base_cache_lineage_sha256=base_cache.lineage_sha256,
        emotion_group=emotion_group,
    )
