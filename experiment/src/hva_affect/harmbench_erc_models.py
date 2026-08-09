"""Synthetic-testable model families for the HarmBench-ERC open roles.

This module is deliberately downstream of feature processing and upstream of
evaluation.  It accepts only three row-aligned float32 modality matrices and,
for history models, precomputed ordered context-index tuples.  It has no
group, speaker, time, role, outcome-selection, or filesystem surface.

The independent current-only path is physically separate: its trainers,
checkpoints, neural modules, method signatures, and model identities are not
shared with the history path.  In particular, no current-only ``fit`` or
``predict_proba`` method accepts a context/history argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
from numbers import Integral, Real
from types import MappingProxyType
from typing import ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.linear_model import SGDClassifier
import torch
from torch import nn
from torch.nn import functional as torch_functional

from .harmbench_erc_contract import EXPECTED_TRAINING_SEEDS
from .harmbench_erc_contexts import (
    CURRENT_ONLY_STRATEGY_ID,
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    FIT_TRAIN_CONTEXT_ROLE,
    SELECTION_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    StrictPastContextRoster,
    build_strict_past_context_roster,
    validate_strict_past_context_roster,
)
from .harmbench_erc_crossfit import (
    ContextTrainingExamples,
    EXPECTED_OUTER_FOLDS,
    SharedGroupCrossfitPlan,
    resolve_shared_group_crossfit_indices,
    validate_context_training_examples,
    validate_shared_group_crossfit_plan,
)
from .harmbench_erc_open_roles import (
    FitFeatureCapability,
    FitRoleCapability,
    SelectionFeatureCapability,
    validate_fit_feature_capability,
    validate_fit_role_capability,
    validate_selection_feature_capability,
)
from .harmbench_erc_processors import (
    ProcessedRoleEmbeddings,
    ProcessorReceipt,
    validate_processed_role_embeddings,
)


LINEAR_POOL_ID = "hb_linear_pool_v1"
DEEPSETS_POOL_ID = "hb_deepsets_pool_v1"
CAUSAL_GRU_ID = "hb_causal_gru_v1"
FROZEN_MODEL_IDS = (LINEAR_POOL_ID, DEEPSETS_POOL_ID, CAUSAL_GRU_ID)

CURRENT_ONLY_NAMESPACE = "harmbench_erc.current_only"
HISTORY_NAMESPACE = "harmbench_erc.history"

TEXT_DIMENSION = 256
AUDIO_DIMENSION = 128
VIDEO_DIMENSION = 128
MODALITY_DIMENSIONS: Mapping[str, int] = MappingProxyType(
    {
        "text": TEXT_DIMENSION,
        "audio": AUDIO_DIMENSION,
        "video": VIDEO_DIMENSION,
    }
)
QUERY_DIMENSION = sum(MODALITY_DIMENSIONS.values())
LINEAR_HISTORY_SUMMARY_DIMENSION = 4 * QUERY_DIMENSION + 1

DEEPSETS_PARAMETER_LIMIT = 1_000_000
CAUSAL_GRU_PARAMETER_LIMIT = 2_000_000
DEFAULT_NEURAL_EPOCHS = 12
DEFAULT_LINEAR_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-2


class HarmBenchModelError(ValueError):
    """Raised when model inputs or outputs violate the frozen contract."""


@runtime_checkable
class ProcessedRoleLike(Protocol):
    """Minimal adapter protocol expected from the future feature processor."""

    text: np.ndarray
    audio: np.ndarray
    video: np.ndarray


def _modality_matrix(values: object, *, name: str, dimension: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.dtype(np.float32):
        raise HarmBenchModelError(f"{name} must have exact float32 dtype")
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != dimension:
        raise HarmBenchModelError(
            f"{name} must have shape (positive_rows, {dimension})"
        )
    if not np.isfinite(raw).all():
        raise HarmBenchModelError(f"{name} contains non-finite values")
    result = np.array(raw, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProcessedRole:
    """Immutable, outcome-free, row-aligned processor output snapshot."""

    text: np.ndarray
    audio: np.ndarray
    video: np.ndarray

    def __post_init__(self) -> None:
        text = _modality_matrix(
            self.text, name="text", dimension=TEXT_DIMENSION
        )
        audio = _modality_matrix(
            self.audio, name="audio", dimension=AUDIO_DIMENSION
        )
        video = _modality_matrix(
            self.video, name="video", dimension=VIDEO_DIMENSION
        )
        if not (len(text) == len(audio) == len(video)):
            raise HarmBenchModelError("modality matrices are not row-aligned")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "audio", audio)
        object.__setattr__(self, "video", video)

    @property
    def rows(self) -> int:
        return len(self.text)


def _snapshot_features(features: ProcessedRoleLike) -> ProcessedRole:
    if not isinstance(features, ProcessedRoleLike):
        raise HarmBenchModelError("features do not satisfy ProcessedRoleLike")
    return ProcessedRole(
        text=features.text,
        audio=features.audio,
        video=features.video,
    )


def _exact_positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise HarmBenchModelError(f"{name} must be an exact integer")
    normalized = int(value)
    if normalized < minimum:
        raise HarmBenchModelError(f"{name} must be at least {minimum}")
    return normalized


def _validate_seed(seed: object) -> int:
    normalized = _exact_positive_integer(seed, name="seed", minimum=0)
    if normalized > np.iinfo(np.uint32).max:
        raise HarmBenchModelError("seed must fit in uint32")
    return normalized


def _validate_training_hyperparameters(
    *, num_classes: object, seed: object, epochs: object, learning_rate: object
) -> tuple[int, int, int, float]:
    classes = _exact_positive_integer(num_classes, name="num_classes", minimum=2)
    normalized_seed = _validate_seed(seed)
    normalized_epochs = _exact_positive_integer(epochs, name="epochs")
    if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(
        learning_rate, Real
    ):
        raise HarmBenchModelError("learning_rate must be finite and positive")
    rate = float(learning_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise HarmBenchModelError("learning_rate must be finite and positive")
    return classes, normalized_seed, normalized_epochs, rate


def _labels(
    values: object, *, expected_rows: int, num_classes: int
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != expected_rows:
        raise HarmBenchModelError("labels must be a row-aligned vector")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise HarmBenchModelError("labels must have an integer dtype")
    result = np.array(raw, dtype=np.int64, copy=True)
    if np.any(result < 0) or np.any(result >= num_classes):
        raise HarmBenchModelError("label is outside the fixed class roster")
    result.setflags(write=False)
    return result


OrderedContexts = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ExpandedHistoryExamples:
    """Pre-expanded query/context pairs without strategy or other metadata.

    Query indices may repeat because one fit query can contribute distinct
    contexts from multiple frozen strategies.  Labels supplied to a trainer
    are aligned to these examples, not to the underlying feature bank.
    """

    query_indices: tuple[int, ...]
    contexts: OrderedContexts
    feature_rows: int

    @property
    def rows(self) -> int:
        return len(self.query_indices)


def normalize_expanded_history_examples(
    query_indices: Sequence[object],
    contexts: Sequence[Sequence[object]],
    *,
    feature_rows: int,
) -> ExpandedHistoryExamples:
    """Validate and copy pre-expanded query/context pairs.

    This function deliberately receives no strategy ids, groups, time values,
    roles, or labels.  Upstream code owns strict-past and fold-membership
    validation; this boundary only guarantees safe feature-bank indexing.
    """

    available_rows = _exact_positive_integer(feature_rows, name="feature_rows")
    try:
        raw_queries = tuple(query_indices)
        raw_contexts = tuple(contexts)
    except TypeError as error:
        raise HarmBenchModelError(
            "expanded queries and contexts must be sequences"
        ) from error
    if not raw_queries or len(raw_queries) != len(raw_contexts):
        raise HarmBenchModelError(
            "expanded query indices and contexts must be non-empty and aligned"
        )
    normalized_queries: list[int] = []
    normalized_contexts: list[tuple[int, ...]] = []
    for example, (raw_query, raw_context) in enumerate(
        zip(raw_queries, raw_contexts, strict=True)
    ):
        if isinstance(raw_query, (bool, np.bool_)) or not isinstance(
            raw_query, Integral
        ):
            raise HarmBenchModelError(
                f"query_indices[{example}] must be an exact integer"
            )
        query = int(raw_query)
        if not 0 <= query < available_rows:
            raise HarmBenchModelError(
                f"query_indices[{example}] is outside the feature bank"
            )
        try:
            candidates = tuple(raw_context)
        except TypeError as error:
            raise HarmBenchModelError(
                f"contexts[{example}] must be an iterable of row indices"
            ) from error
        context: list[int] = []
        for value in candidates:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise HarmBenchModelError(
                    f"contexts[{example}] must contain exact integer indices"
                )
            candidate = int(value)
            if not 0 <= candidate < available_rows:
                raise HarmBenchModelError(
                    f"contexts[{example}] contains an out-of-range row"
                )
            if candidate == query:
                raise HarmBenchModelError(
                    f"contexts[{example}] contains its current query row"
                )
            context.append(candidate)
        if len(context) != len(set(context)):
            raise HarmBenchModelError(f"contexts[{example}] contains duplicate rows")
        normalized_queries.append(query)
        normalized_contexts.append(tuple(context))
    return ExpandedHistoryExamples(
        query_indices=tuple(normalized_queries),
        contexts=tuple(normalized_contexts),
        feature_rows=available_rows,
    )


def normalize_ordered_contexts(
    contexts: Sequence[Sequence[object]], *, rows: int
) -> OrderedContexts:
    """Copy and validate one ordered, duplicate-free context tuple per query.

    Temporal validity is intentionally an upstream responsibility because this
    module has no access to time or identity metadata.  GRU models consume the
    supplied order exactly; DeepSets models intentionally ignore it.
    """

    expected_rows = _exact_positive_integer(rows, name="rows")
    try:
        raw_contexts = tuple(contexts)
    except TypeError as error:
        raise HarmBenchModelError("contexts must be a row-aligned sequence") from error
    if len(raw_contexts) != expected_rows:
        raise HarmBenchModelError("contexts must contain one tuple per query")
    return normalize_expanded_history_examples(
        tuple(range(expected_rows)),
        raw_contexts,
        feature_rows=expected_rows,
    ).contexts


def _history_examples(
    contexts: Sequence[Sequence[object]],
    *,
    feature_rows: int,
    query_indices: Sequence[object] | None,
) -> ExpandedHistoryExamples:
    if query_indices is None:
        normalized = normalize_ordered_contexts(contexts, rows=feature_rows)
        return ExpandedHistoryExamples(
            query_indices=tuple(range(feature_rows)),
            contexts=normalized,
            feature_rows=feature_rows,
        )
    return normalize_expanded_history_examples(
        query_indices, contexts, feature_rows=feature_rows
    )


def _query_matrix(features: ProcessedRole) -> np.ndarray:
    result = np.concatenate(
        (features.text, features.audio, features.video), axis=1, dtype=np.float32
    )
    result.setflags(write=False)
    return result


def linear_current_summary(features: ProcessedRoleLike) -> np.ndarray:
    """Return only the current-query modalities for the independent baseline."""

    snapshot = _snapshot_features(features)
    result = np.array(_query_matrix(snapshot), dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def linear_history_summary(
    features: ProcessedRoleLike,
    contexts: Sequence[Sequence[object]],
    *,
    query_indices: Sequence[object] | None = None,
) -> np.ndarray:
    """Return query + context mean/last/delta/log1p-count summaries."""

    snapshot = _snapshot_features(features)
    examples = _history_examples(
        contexts, feature_rows=snapshot.rows, query_indices=query_indices
    )
    feature_bank = np.asarray(_query_matrix(snapshot), dtype=np.float64)
    query = feature_bank[np.asarray(examples.query_indices, dtype=np.int64)]
    mean = np.zeros_like(query)
    last = np.zeros_like(query)
    counts = np.zeros((examples.rows, 1), dtype=np.float64)
    for row, context in enumerate(examples.contexts):
        if context:
            selected = feature_bank[np.asarray(context, dtype=np.int64)]
            mean[row] = selected.mean(axis=0, dtype=np.float64)
            last[row] = selected[-1]
        counts[row, 0] = np.log1p(len(context))
    delta = query - last
    result = np.concatenate((query, mean, last, delta, counts), axis=1)
    if result.shape != (examples.rows, LINEAR_HISTORY_SUMMARY_DIMENSION):
        raise HarmBenchModelError("internal linear summary dimension changed")
    result.setflags(write=False)
    return result


def validate_probability_matrix(
    values: object, *, expected_rows: int, num_classes: int
) -> np.ndarray:
    """Fail closed on malformed probabilities and return immutable float64."""

    rows = _exact_positive_integer(expected_rows, name="expected_rows")
    classes = _exact_positive_integer(num_classes, name="num_classes", minimum=2)
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HarmBenchModelError("probabilities must be numeric") from error
    if probabilities.shape != (rows, classes):
        raise HarmBenchModelError(
            f"probabilities must have shape ({rows}, {classes})"
        )
    if not np.isfinite(probabilities).all():
        raise HarmBenchModelError("probabilities contain non-finite values")
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1.0 + 1e-12):
        raise HarmBenchModelError("probabilities fall outside [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
    ):
        raise HarmBenchModelError("probability rows do not sum to one")
    result = np.array(probabilities, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _softmax_float64(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _fit_linear_estimator(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    epochs: int,
) -> SGDClassifier:
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        fit_intercept=True,
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=seed,
        learning_rate="optimal",
        average=False,
    )
    class_roster = np.arange(num_classes, dtype=np.int64)
    for epoch in range(epochs):
        estimator.partial_fit(
            design,
            labels,
            classes=class_roster if epoch == 0 else None,
        )
    if not np.array_equal(estimator.classes_, class_roster):
        raise HarmBenchModelError("linear estimator class roster changed")
    return estimator


@dataclass(frozen=True)
class LinearHistoryCheckpoint:
    """Fitted history-aware linear model; not a current-only checkpoint."""

    num_classes: int
    seed: int
    _estimator: SGDClassifier = field(repr=False, compare=False)

    family_id: ClassVar[str] = LINEAR_POOL_ID
    model_namespace: ClassVar[str] = HISTORY_NAMESPACE
    model_identity: ClassVar[str] = f"{HISTORY_NAMESPACE}/{LINEAR_POOL_ID}"

    @property
    def parameter_count(self) -> int:
        return int(self._estimator.coef_.size + self._estimator.intercept_.size)

    def predict_proba(
        self,
        features: ProcessedRoleLike,
        contexts: Sequence[Sequence[object]],
        *,
        query_indices: Sequence[object] | None = None,
    ) -> np.ndarray:
        design = linear_history_summary(
            features, contexts, query_indices=query_indices
        )
        values = self._estimator.predict_proba(design)
        return validate_probability_matrix(
            values, expected_rows=len(design), num_classes=self.num_classes
        )


@dataclass(frozen=True)
class LinearCurrentOnlyCheckpoint:
    """Fitted physically independent current-query-only linear model."""

    num_classes: int
    seed: int
    _estimator: SGDClassifier = field(repr=False, compare=False)

    family_id: ClassVar[str] = LINEAR_POOL_ID
    model_namespace: ClassVar[str] = CURRENT_ONLY_NAMESPACE
    model_identity: ClassVar[str] = f"{CURRENT_ONLY_NAMESPACE}/{LINEAR_POOL_ID}"

    @property
    def parameter_count(self) -> int:
        return int(self._estimator.coef_.size + self._estimator.intercept_.size)

    def predict_proba(self, features: ProcessedRoleLike) -> np.ndarray:
        design = linear_current_summary(features)
        values = self._estimator.predict_proba(design)
        return validate_probability_matrix(
            values, expected_rows=len(design), num_classes=self.num_classes
        )


@dataclass(frozen=True)
class LinearHistoryTrainer:
    num_classes: int
    seed: int
    epochs: int = DEFAULT_LINEAR_EPOCHS

    family_id: ClassVar[str] = LINEAR_POOL_ID
    model_namespace: ClassVar[str] = HISTORY_NAMESPACE

    def __post_init__(self) -> None:
        values = _validate_training_hyperparameters(
            num_classes=self.num_classes,
            seed=self.seed,
            epochs=self.epochs,
            learning_rate=1.0,
        )
        object.__setattr__(self, "num_classes", values[0])
        object.__setattr__(self, "seed", values[1])
        object.__setattr__(self, "epochs", values[2])

    def fit(
        self,
        features: ProcessedRoleLike,
        labels: object,
        contexts: Sequence[Sequence[object]],
        *,
        query_indices: Sequence[object] | None = None,
    ) -> LinearHistoryCheckpoint:
        design = linear_history_summary(
            features, contexts, query_indices=query_indices
        )
        target = _labels(
            labels, expected_rows=len(design), num_classes=self.num_classes
        )
        estimator = _fit_linear_estimator(
            design,
            target,
            num_classes=self.num_classes,
            seed=self.seed,
            epochs=self.epochs,
        )
        return LinearHistoryCheckpoint(self.num_classes, self.seed, estimator)


@dataclass(frozen=True)
class LinearCurrentOnlyTrainer:
    num_classes: int
    seed: int
    epochs: int = DEFAULT_LINEAR_EPOCHS

    family_id: ClassVar[str] = LINEAR_POOL_ID
    model_namespace: ClassVar[str] = CURRENT_ONLY_NAMESPACE

    def __post_init__(self) -> None:
        values = _validate_training_hyperparameters(
            num_classes=self.num_classes,
            seed=self.seed,
            epochs=self.epochs,
            learning_rate=1.0,
        )
        object.__setattr__(self, "num_classes", values[0])
        object.__setattr__(self, "seed", values[1])
        object.__setattr__(self, "epochs", values[2])

    def fit(
        self, features: ProcessedRoleLike, labels: object
    ) -> LinearCurrentOnlyCheckpoint:
        design = linear_current_summary(features)
        target = _labels(
            labels, expected_rows=len(design), num_classes=self.num_classes
        )
        estimator = _fit_linear_estimator(
            design,
            target,
            num_classes=self.num_classes,
            seed=self.seed,
            epochs=self.epochs,
        )
        return LinearCurrentOnlyCheckpoint(self.num_classes, self.seed, estimator)


PROJECTION_DIMENSION = 32
ITEM_HIDDEN_DIMENSION = 64
GRU_HIDDEN_DIMENSION = 64


class _DeepSetsHistoryNetwork(nn.Module):
    """History DeepSets network; context membership is a set, not a sequence."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(TEXT_DIMENSION, PROJECTION_DIMENSION)
        self.audio_projection = nn.Linear(AUDIO_DIMENSION, PROJECTION_DIMENSION)
        self.video_projection = nn.Linear(VIDEO_DIMENSION, PROJECTION_DIMENSION)
        item_input = 3 * PROJECTION_DIMENSION
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input, ITEM_HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(ITEM_HIDDEN_DIMENSION, ITEM_HIDDEN_DIMENSION),
            nn.ReLU(),
        )
        head_input = 3 * ITEM_HIDDEN_DIMENSION + 1
        self.classifier = nn.Linear(head_input, num_classes)

    def _items(
        self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor
    ) -> torch.Tensor:
        projected = torch.cat(
            (
                torch.relu(self.text_projection(text)),
                torch.relu(self.audio_projection(audio)),
                torch.relu(self.video_projection(video)),
            ),
            dim=1,
        )
        return self.item_mlp(projected)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        query_indices: tuple[int, ...],
        contexts: OrderedContexts,
    ) -> torch.Tensor:
        items = self._items(text, audio, video)
        pooled_rows: list[torch.Tensor] = []
        for query, context in zip(query_indices, contexts, strict=True):
            if context:
                # Sorting is deliberate: this family is a set model, and the
                # canonical reduction order makes permutation invariance exact.
                indices = torch.tensor(sorted(context), dtype=torch.long)
                selected = items.index_select(0, indices)
                mean = selected.mean(dim=0)
                maximum = selected.max(dim=0).values
            else:
                mean = torch.zeros_like(items[query])
                maximum = torch.zeros_like(items[query])
            count = items.new_tensor([np.log1p(len(context))])
            pooled_rows.append(torch.cat((items[query], mean, maximum, count)))
        return self.classifier(torch.stack(pooled_rows, dim=0))


class _DeepSetsCurrentOnlyNetwork(nn.Module):
    """Separate current-only network with no pooling or context argument."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(TEXT_DIMENSION, PROJECTION_DIMENSION)
        self.audio_projection = nn.Linear(AUDIO_DIMENSION, PROJECTION_DIMENSION)
        self.video_projection = nn.Linear(VIDEO_DIMENSION, PROJECTION_DIMENSION)
        item_input = 3 * PROJECTION_DIMENSION
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input, ITEM_HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(ITEM_HIDDEN_DIMENSION, ITEM_HIDDEN_DIMENSION),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(ITEM_HIDDEN_DIMENSION, num_classes)

    def forward(
        self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor
    ) -> torch.Tensor:
        projected = torch.cat(
            (
                torch.relu(self.text_projection(text)),
                torch.relu(self.audio_projection(audio)),
                torch.relu(self.video_projection(video)),
            ),
            dim=1,
        )
        return self.classifier(self.item_mlp(projected))


class _CausalGRUHistoryNetwork(nn.Module):
    """One-layer GRU consuming context in supplied strict-past order + query."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(TEXT_DIMENSION, PROJECTION_DIMENSION)
        self.audio_projection = nn.Linear(AUDIO_DIMENSION, PROJECTION_DIMENSION)
        self.video_projection = nn.Linear(VIDEO_DIMENSION, PROJECTION_DIMENSION)
        self.gru = nn.GRU(
            input_size=3 * PROJECTION_DIMENSION,
            hidden_size=GRU_HIDDEN_DIMENSION,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(GRU_HIDDEN_DIMENSION, num_classes)

    def _items(
        self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (
                torch.relu(self.text_projection(text)),
                torch.relu(self.audio_projection(audio)),
                torch.relu(self.video_projection(video)),
            ),
            dim=1,
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        query_indices: tuple[int, ...],
        contexts: OrderedContexts,
    ) -> torch.Tensor:
        items = self._items(text, audio, video)
        rows: list[torch.Tensor] = []
        for query, context in zip(query_indices, contexts, strict=True):
            sequence_indices = (*context, query)
            indices = torch.tensor(sequence_indices, dtype=torch.long)
            sequence = items.index_select(0, indices).unsqueeze(0)
            _, hidden = self.gru(sequence)
            rows.append(hidden[-1, 0])
        return self.classifier(torch.stack(rows, dim=0))


class _CausalGRUCurrentOnlyNetwork(nn.Module):
    """Separate length-one GRU current-only baseline."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(TEXT_DIMENSION, PROJECTION_DIMENSION)
        self.audio_projection = nn.Linear(AUDIO_DIMENSION, PROJECTION_DIMENSION)
        self.video_projection = nn.Linear(VIDEO_DIMENSION, PROJECTION_DIMENSION)
        self.gru = nn.GRU(
            input_size=3 * PROJECTION_DIMENSION,
            hidden_size=GRU_HIDDEN_DIMENSION,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(GRU_HIDDEN_DIMENSION, num_classes)

    def forward(
        self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor
    ) -> torch.Tensor:
        items = torch.cat(
            (
                torch.relu(self.text_projection(text)),
                torch.relu(self.audio_projection(audio)),
                torch.relu(self.video_projection(video)),
            ),
            dim=1,
        )
        _, hidden = self.gru(items.unsqueeze(1))
        return self.classifier(hidden[-1])


def _parameter_count(network: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in network.parameters()))


def _torch_inputs(
    features: ProcessedRole,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.array(features.text, copy=True)),
        torch.from_numpy(np.array(features.audio, copy=True)),
        torch.from_numpy(np.array(features.video, copy=True)),
    )


def _initialized_network(
    network_type: type[nn.Module], *, num_classes: int, seed: int
) -> nn.Module:
    # fork_rng prevents model construction from changing the caller's global
    # torch RNG state while still making initialization seed-deterministic.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return network_type(num_classes)


def _fit_history_network(
    network: nn.Module,
    features: ProcessedRole,
    labels: np.ndarray,
    examples: ExpandedHistoryExamples,
    *,
    epochs: int,
    learning_rate: float,
) -> nn.Module:
    text, audio, video = _torch_inputs(features)
    target = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    network.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = network(
            text, audio, video, examples.query_indices, examples.contexts
        )
        loss = torch_functional.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise HarmBenchModelError("neural training produced a non-finite loss")
        loss.backward()
        optimizer.step()
    network.eval()
    network.requires_grad_(False)
    return network


def _fit_current_network(
    network: nn.Module,
    features: ProcessedRole,
    labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
) -> nn.Module:
    text, audio, video = _torch_inputs(features)
    target = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    network.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = network(text, audio, video)
        loss = torch_functional.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise HarmBenchModelError("neural training produced a non-finite loss")
        loss.backward()
        optimizer.step()
    network.eval()
    network.requires_grad_(False)
    return network


def _history_logits(
    network: nn.Module,
    features: ProcessedRole,
    examples: ExpandedHistoryExamples,
) -> np.ndarray:
    text, audio, video = _torch_inputs(features)
    network.eval()
    with torch.no_grad():
        logits = network(
            text, audio, video, examples.query_indices, examples.contexts
        )
    return logits.detach().cpu().numpy().astype(np.float64, copy=True)


def _current_logits(network: nn.Module, features: ProcessedRole) -> np.ndarray:
    text, audio, video = _torch_inputs(features)
    network.eval()
    with torch.no_grad():
        logits = network(text, audio, video)
    return logits.detach().cpu().numpy().astype(np.float64, copy=True)


@dataclass(frozen=True)
class DeepSetsHistoryCheckpoint:
    num_classes: int
    seed: int
    _network: _DeepSetsHistoryNetwork = field(repr=False, compare=False)

    family_id: ClassVar[str] = DEEPSETS_POOL_ID
    model_namespace: ClassVar[str] = HISTORY_NAMESPACE
    model_identity: ClassVar[str] = f"{HISTORY_NAMESPACE}/{DEEPSETS_POOL_ID}"

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self._network)

    def predict_proba(
        self,
        features: ProcessedRoleLike,
        contexts: Sequence[Sequence[object]],
        *,
        query_indices: Sequence[object] | None = None,
    ) -> np.ndarray:
        snapshot = _snapshot_features(features)
        examples = _history_examples(
            contexts, feature_rows=snapshot.rows, query_indices=query_indices
        )
        probabilities = _softmax_float64(
            _history_logits(self._network, snapshot, examples)
        )
        return validate_probability_matrix(
            probabilities,
            expected_rows=examples.rows,
            num_classes=self.num_classes,
        )


@dataclass(frozen=True)
class DeepSetsCurrentOnlyCheckpoint:
    num_classes: int
    seed: int
    _network: _DeepSetsCurrentOnlyNetwork = field(repr=False, compare=False)

    family_id: ClassVar[str] = DEEPSETS_POOL_ID
    model_namespace: ClassVar[str] = CURRENT_ONLY_NAMESPACE
    model_identity: ClassVar[str] = f"{CURRENT_ONLY_NAMESPACE}/{DEEPSETS_POOL_ID}"

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self._network)

    def predict_proba(self, features: ProcessedRoleLike) -> np.ndarray:
        snapshot = _snapshot_features(features)
        probabilities = _softmax_float64(_current_logits(self._network, snapshot))
        return validate_probability_matrix(
            probabilities,
            expected_rows=snapshot.rows,
            num_classes=self.num_classes,
        )


@dataclass(frozen=True)
class CausalGRUHistoryCheckpoint:
    num_classes: int
    seed: int
    _network: _CausalGRUHistoryNetwork = field(repr=False, compare=False)

    family_id: ClassVar[str] = CAUSAL_GRU_ID
    model_namespace: ClassVar[str] = HISTORY_NAMESPACE
    model_identity: ClassVar[str] = f"{HISTORY_NAMESPACE}/{CAUSAL_GRU_ID}"

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self._network)

    def predict_proba(
        self,
        features: ProcessedRoleLike,
        contexts: Sequence[Sequence[object]],
        *,
        query_indices: Sequence[object] | None = None,
    ) -> np.ndarray:
        snapshot = _snapshot_features(features)
        examples = _history_examples(
            contexts, feature_rows=snapshot.rows, query_indices=query_indices
        )
        probabilities = _softmax_float64(
            _history_logits(self._network, snapshot, examples)
        )
        return validate_probability_matrix(
            probabilities,
            expected_rows=examples.rows,
            num_classes=self.num_classes,
        )


@dataclass(frozen=True)
class CausalGRUCurrentOnlyCheckpoint:
    num_classes: int
    seed: int
    _network: _CausalGRUCurrentOnlyNetwork = field(repr=False, compare=False)

    family_id: ClassVar[str] = CAUSAL_GRU_ID
    model_namespace: ClassVar[str] = CURRENT_ONLY_NAMESPACE
    model_identity: ClassVar[str] = f"{CURRENT_ONLY_NAMESPACE}/{CAUSAL_GRU_ID}"

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self._network)

    def predict_proba(self, features: ProcessedRoleLike) -> np.ndarray:
        snapshot = _snapshot_features(features)
        probabilities = _softmax_float64(_current_logits(self._network, snapshot))
        return validate_probability_matrix(
            probabilities,
            expected_rows=snapshot.rows,
            num_classes=self.num_classes,
        )


@dataclass(frozen=True)
class _NeuralTrainerSettings:
    num_classes: int
    seed: int
    epochs: int = DEFAULT_NEURAL_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE

    def __post_init__(self) -> None:
        values = _validate_training_hyperparameters(
            num_classes=self.num_classes,
            seed=self.seed,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        object.__setattr__(self, "num_classes", values[0])
        object.__setattr__(self, "seed", values[1])
        object.__setattr__(self, "epochs", values[2])
        object.__setattr__(self, "learning_rate", values[3])


@dataclass(frozen=True)
class DeepSetsHistoryTrainer(_NeuralTrainerSettings):
    family_id: ClassVar[str] = DEEPSETS_POOL_ID
    model_namespace: ClassVar[str] = HISTORY_NAMESPACE

    def fit(
        self,
        features: ProcessedRoleLike,
        labels: object,
        contexts: Sequence[Sequence[object]],
        *,
        query_indices: Sequence[object] | None = None,
    ) -> DeepSetsHistoryCheckpoint:
        snapshot = _snapshot_features(features)
        examples = _history_examples(
            contexts, feature_rows=snapshot.rows, query_indices=query_indices
        )
        target = _labels(
            labels, expected_rows=examples.rows, num_classes=self.num_classes
        )
        network = _initialized_network(
            _DeepSetsHistoryNetwork,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        if _parameter_count(network) >= DEEPSETS_PARAMETER_LIMIT:
            raise HarmBenchModelError("DeepSets parameter budget was exceeded")
        fitted = _fit_history_network(
            network,
            snapshot,
            target,
            examples,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        return DeepSetsHistoryCheckpoint(
            self.num_classes, self.seed, fitted  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DeepSetsCurrentOnlyTrainer(_NeuralTrainerSettings):
    family_id: ClassVar[str] = DEEPSETS_POOL_ID
    model_namespace: ClassVar[str] = CURRENT_ONLY_NAMESPACE

    def fit(
        self, features: ProcessedRoleLike, labels: object
    ) -> DeepSetsCurrentOnlyCheckpoint:
        snapshot = _snapshot_features(features)
        target = _labels(
            labels, expected_rows=snapshot.rows, num_classes=self.num_classes
        )
        network = _initialized_network(
            _DeepSetsCurrentOnlyNetwork,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        if _parameter_count(network) >= DEEPSETS_PARAMETER_LIMIT:
            raise HarmBenchModelError("DeepSets parameter budget was exceeded")
        fitted = _fit_current_network(
            network,
            snapshot,
            target,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        return DeepSetsCurrentOnlyCheckpoint(
            self.num_classes, self.seed, fitted  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CausalGRUHistoryTrainer(_NeuralTrainerSettings):
    family_id: ClassVar[str] = CAUSAL_GRU_ID
    model_namespace: ClassVar[str] = HISTORY_NAMESPACE

    def fit(
        self,
        features: ProcessedRoleLike,
        labels: object,
        contexts: Sequence[Sequence[object]],
        *,
        query_indices: Sequence[object] | None = None,
    ) -> CausalGRUHistoryCheckpoint:
        snapshot = _snapshot_features(features)
        examples = _history_examples(
            contexts, feature_rows=snapshot.rows, query_indices=query_indices
        )
        target = _labels(
            labels, expected_rows=examples.rows, num_classes=self.num_classes
        )
        network = _initialized_network(
            _CausalGRUHistoryNetwork,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        if _parameter_count(network) >= CAUSAL_GRU_PARAMETER_LIMIT:
            raise HarmBenchModelError("causal GRU parameter budget was exceeded")
        fitted = _fit_history_network(
            network,
            snapshot,
            target,
            examples,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        return CausalGRUHistoryCheckpoint(
            self.num_classes, self.seed, fitted  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CausalGRUCurrentOnlyTrainer(_NeuralTrainerSettings):
    family_id: ClassVar[str] = CAUSAL_GRU_ID
    model_namespace: ClassVar[str] = CURRENT_ONLY_NAMESPACE

    def fit(
        self, features: ProcessedRoleLike, labels: object
    ) -> CausalGRUCurrentOnlyCheckpoint:
        snapshot = _snapshot_features(features)
        target = _labels(
            labels, expected_rows=snapshot.rows, num_classes=self.num_classes
        )
        network = _initialized_network(
            _CausalGRUCurrentOnlyNetwork,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        if _parameter_count(network) >= CAUSAL_GRU_PARAMETER_LIMIT:
            raise HarmBenchModelError("causal GRU parameter budget was exceeded")
        fitted = _fit_current_network(
            network,
            snapshot,
            target,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        return CausalGRUCurrentOnlyCheckpoint(
            self.num_classes, self.seed, fitted  # type: ignore[arg-type]
        )


HistoryCheckpoint = (
    LinearHistoryCheckpoint
    | DeepSetsHistoryCheckpoint
    | CausalGRUHistoryCheckpoint
)
CurrentOnlyCheckpoint = (
    LinearCurrentOnlyCheckpoint
    | DeepSetsCurrentOnlyCheckpoint
    | CausalGRUCurrentOnlyCheckpoint
)


def _make_history_trainer(
    model_id: str,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> LinearHistoryTrainer | DeepSetsHistoryTrainer | CausalGRUHistoryTrainer:
    """Construct a frozen-family history trainer by exact model id."""

    if model_id == LINEAR_POOL_ID:
        return LinearHistoryTrainer(
            num_classes=num_classes,
            seed=seed,
            epochs=DEFAULT_LINEAR_EPOCHS if epochs is None else epochs,
        )
    neural_epochs = DEFAULT_NEURAL_EPOCHS if epochs is None else epochs
    if model_id == DEEPSETS_POOL_ID:
        return DeepSetsHistoryTrainer(
            num_classes=num_classes,
            seed=seed,
            epochs=neural_epochs,
            learning_rate=learning_rate,
        )
    if model_id == CAUSAL_GRU_ID:
        return CausalGRUHistoryTrainer(
            num_classes=num_classes,
            seed=seed,
            epochs=neural_epochs,
            learning_rate=learning_rate,
        )
    raise HarmBenchModelError("unknown history model id")


def make_history_trainer(
    model_id: str,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> LinearHistoryTrainer | DeepSetsHistoryTrainer | CausalGRUHistoryTrainer:
    """Synthetic-test factory; production code must use the private fit core."""

    return _make_history_trainer(
        model_id,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def _make_current_only_trainer(
    model_id: str,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> (
    LinearCurrentOnlyTrainer
    | DeepSetsCurrentOnlyTrainer
    | CausalGRUCurrentOnlyTrainer
):
    """Construct a physically independent current-only trainer by exact id."""

    if model_id == LINEAR_POOL_ID:
        return LinearCurrentOnlyTrainer(
            num_classes=num_classes,
            seed=seed,
            epochs=DEFAULT_LINEAR_EPOCHS if epochs is None else epochs,
        )
    neural_epochs = DEFAULT_NEURAL_EPOCHS if epochs is None else epochs
    if model_id == DEEPSETS_POOL_ID:
        return DeepSetsCurrentOnlyTrainer(
            num_classes=num_classes,
            seed=seed,
            epochs=neural_epochs,
            learning_rate=learning_rate,
        )
    if model_id == CAUSAL_GRU_ID:
        return CausalGRUCurrentOnlyTrainer(
            num_classes=num_classes,
            seed=seed,
            epochs=neural_epochs,
            learning_rate=learning_rate,
        )
    raise HarmBenchModelError("unknown current-only model id")


def make_current_only_trainer(
    model_id: str,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> (
    LinearCurrentOnlyTrainer
    | DeepSetsCurrentOnlyTrainer
    | CausalGRUCurrentOnlyTrainer
):
    """Synthetic-test factory; production code must use the private fit core."""

    return _make_current_only_trainer(
        model_id,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def _fit_history_array_core(
    model_id: str,
    features: ProcessedRoleLike,
    labels: object,
    contexts: Sequence[Sequence[object]],
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
    query_indices: Sequence[object] | None = None,
) -> HistoryCheckpoint:
    """Private array core shared by capability-safe and synthetic entry points."""

    trainer = _make_history_trainer(
        model_id,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
    )
    return trainer.fit(
        features, labels, contexts, query_indices=query_indices
    )


def fit_synthetic_history_model(
    model_id: str,
    features: ProcessedRoleLike,
    labels: object,
    contexts: Sequence[Sequence[object]],
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
    query_indices: Sequence[object] | None = None,
) -> HistoryCheckpoint:
    """Array-level synthetic-test helper; production code must not call this."""

    return _fit_history_array_core(
        model_id,
        features,
        labels,
        contexts,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
        query_indices=query_indices,
    )


def predict_history_model(
    checkpoint: HistoryCheckpoint,
    features: ProcessedRoleLike,
    contexts: Sequence[Sequence[object]],
    *,
    query_indices: Sequence[object] | None = None,
) -> np.ndarray:
    """Unified history-family probability entry point."""

    if not isinstance(
        checkpoint,
        (
            LinearHistoryCheckpoint,
            DeepSetsHistoryCheckpoint,
            CausalGRUHistoryCheckpoint,
        ),
    ):
        raise HarmBenchModelError("checkpoint is not a history model")
    return checkpoint.predict_proba(
        features, contexts, query_indices=query_indices
    )


def _fit_current_only_array_core(
    model_id: str,
    features: ProcessedRoleLike,
    labels: object,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
) -> CurrentOnlyCheckpoint:
    """Private array core shared by capability-safe and synthetic entry points."""

    trainer = _make_current_only_trainer(
        model_id,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
    )
    return trainer.fit(features, labels)


def fit_synthetic_current_only_model(
    model_id: str,
    features: ProcessedRoleLike,
    labels: object,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
) -> CurrentOnlyCheckpoint:
    """Array-level synthetic-test helper; production code must not call this."""

    return _fit_current_only_array_core(
        model_id,
        features,
        labels,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
    )


def _sha256(value: object, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise HarmBenchModelError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def class_order_sha256(
    class_order: Sequence[str],
    *,
    dataset_id: str,
    fit_training_capability_sha256: str,
) -> str:
    """Hash ordered class semantics in the checkpoint-manifest namespace."""

    try:
        order = tuple(class_order)
    except TypeError as error:
        raise HarmBenchModelError("class order must be a sequence") from error
    if (
        len(order) < 2
        or any(not isinstance(value, str) or not value for value in order)
        or len(set(order)) != len(order)
    ):
        raise HarmBenchModelError(
            "class order must contain at least two unique non-empty strings"
        )
    if not isinstance(dataset_id, str) or not dataset_id:
        raise HarmBenchModelError("class order dataset identity is empty")
    fit_sha = _sha256(
        fit_training_capability_sha256,
        name="fit_training_capability_sha256",
    )
    return _canonical_json_sha256(
        {
            "schema_version": "harmbench_erc_checkpoint_class_order_v1",
            "dataset_id": dataset_id,
            "fit_training_capability_sha256": fit_sha,
            "ordered_class_tokens": list(order),
        }
    )


def aggregate_context_roster_sha256(
    roster_sha256_by_strategy: Sequence[tuple[str, str]],
) -> str:
    """Bind the exact ordered multi-strategy fit context roster."""

    try:
        items = tuple(roster_sha256_by_strategy)
    except TypeError as error:
        raise HarmBenchModelError(
            "context roster receipt set must be an ordered sequence"
        ) from error
    if not items or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        for item in items
    ):
        raise HarmBenchModelError("context roster receipt set is malformed")
    strategies = tuple(item[0] for item in items)
    if len(set(strategies)) != len(strategies):
        raise HarmBenchModelError("context roster strategies are duplicated")
    receipts = tuple((strategy, _sha256(digest, name=strategy)) for strategy, digest in items)
    return _canonical_json_sha256(
        {
            "schema_version": "harmbench_erc_context_roster_manifest_v1",
            "strategy_receipts": [list(item) for item in receipts],
        }
    )


@dataclass(frozen=True)
class ProductionHistoryCheckpoint:
    dataset_id: str
    model_id: str
    model_namespace: str
    training_seed: int
    fold: int
    class_order: tuple[str, ...]
    class_order_sha256: str
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    crossfit_plan_sha256: str
    context_training_examples_sha256: str
    context_roster_manifest_sha256: str
    fit_train_protocol_row_ids_sha256: str
    fit_heldout_protocol_row_ids_sha256: str
    checkpoint: HistoryCheckpoint = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model_id not in FROZEN_MODEL_IDS:
            raise HarmBenchModelError("production history model id changed")
        if self.model_namespace != HISTORY_NAMESPACE:
            raise HarmBenchModelError("production history namespace changed")
        if self.checkpoint.model_namespace != HISTORY_NAMESPACE:
            raise HarmBenchModelError("history checkpoint namespace changed")
        if self.checkpoint.family_id != self.model_id:
            raise HarmBenchModelError("history checkpoint family changed")
        if (
            isinstance(self.training_seed, (bool, np.bool_))
            or not isinstance(self.training_seed, Integral)
            or int(self.training_seed) not in EXPECTED_TRAINING_SEEDS
            or isinstance(self.fold, (bool, np.bool_))
            or not isinstance(self.fold, Integral)
            or int(self.fold) not in range(EXPECTED_OUTER_FOLDS)
        ):
            raise HarmBenchModelError("history checkpoint seed/fold changed")
        if self.checkpoint.seed != int(self.training_seed):
            raise HarmBenchModelError("history checkpoint seed differs from lineage")
        if self.checkpoint.num_classes != len(self.class_order):
            raise HarmBenchModelError("history checkpoint class count changed")
        if class_order_sha256(
            self.class_order,
            dataset_id=self.dataset_id,
            fit_training_capability_sha256=self.fit_training_capability_sha256,
        ) != _sha256(
            self.class_order_sha256, name="class_order_sha256"
        ):
            raise HarmBenchModelError("history checkpoint class order changed")
        for name in (
            "fit_training_capability_sha256",
            "fit_feature_capability_sha256",
            "processor_receipt_sha256",
            "processed_output_receipt_sha256",
            "crossfit_plan_sha256",
            "context_training_examples_sha256",
            "context_roster_manifest_sha256",
            "fit_train_protocol_row_ids_sha256",
            "fit_heldout_protocol_row_ids_sha256",
        ):
            _sha256(getattr(self, name), name=name)


@dataclass(frozen=True)
class ProductionCurrentOnlyCheckpoint:
    dataset_id: str
    model_id: str
    model_namespace: str
    training_seed: int
    fold: int
    class_order: tuple[str, ...]
    class_order_sha256: str
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    crossfit_plan_sha256: str
    independence_roster_sha256: str
    fit_train_protocol_row_ids_sha256: str
    fit_heldout_protocol_row_ids_sha256: str
    context_count: int
    history_consumption_count: int
    checkpoint: CurrentOnlyCheckpoint = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model_id not in FROZEN_MODEL_IDS:
            raise HarmBenchModelError("production current-only model id changed")
        if self.model_namespace != CURRENT_ONLY_NAMESPACE:
            raise HarmBenchModelError("production current-only namespace changed")
        if self.checkpoint.model_namespace != CURRENT_ONLY_NAMESPACE:
            raise HarmBenchModelError("current-only checkpoint namespace changed")
        if self.checkpoint.family_id != self.model_id:
            raise HarmBenchModelError("current-only checkpoint family changed")
        if (
            isinstance(self.training_seed, (bool, np.bool_))
            or not isinstance(self.training_seed, Integral)
            or int(self.training_seed) not in EXPECTED_TRAINING_SEEDS
            or isinstance(self.fold, (bool, np.bool_))
            or not isinstance(self.fold, Integral)
            or int(self.fold) not in range(EXPECTED_OUTER_FOLDS)
        ):
            raise HarmBenchModelError("current-only checkpoint seed/fold changed")
        if self.checkpoint.seed != int(self.training_seed):
            raise HarmBenchModelError(
                "current-only checkpoint seed differs from lineage"
            )
        if self.checkpoint.num_classes != len(self.class_order):
            raise HarmBenchModelError("current-only checkpoint class count changed")
        if class_order_sha256(
            self.class_order,
            dataset_id=self.dataset_id,
            fit_training_capability_sha256=self.fit_training_capability_sha256,
        ) != _sha256(
            self.class_order_sha256, name="class_order_sha256"
        ):
            raise HarmBenchModelError("current-only checkpoint class order changed")
        if (
            isinstance(self.context_count, (bool, np.bool_))
            or not isinstance(self.context_count, Integral)
            or int(self.context_count) != 0
            or isinstance(self.history_consumption_count, (bool, np.bool_))
            or not isinstance(self.history_consumption_count, Integral)
            or int(self.history_consumption_count) != 0
        ):
            raise HarmBenchModelError(
                "current-only checkpoint must prove zero context/history consumption"
            )
        for name in (
            "fit_training_capability_sha256",
            "fit_feature_capability_sha256",
            "processor_receipt_sha256",
            "processed_output_receipt_sha256",
            "crossfit_plan_sha256",
            "independence_roster_sha256",
            "fit_train_protocol_row_ids_sha256",
            "fit_heldout_protocol_row_ids_sha256",
        ):
            _sha256(getattr(self, name), name=name)


def _validate_production_fit_inputs(
    fit_capability: FitRoleCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    training_seed: object,
    fold: object,
    expected_fit_training_capability_sha256: str,
    expected_fit_feature_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> tuple[
    FitRoleCapability,
    ProcessedRoleEmbeddings,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    try:
        fit_capability = validate_fit_role_capability(fit_capability)
    except ValueError as error:
        raise HarmBenchModelError(f"fit training capability changed: {error}") from error
    expected_fit_training_sha = _sha256(
        expected_fit_training_capability_sha256,
        name="expected_fit_training_capability_sha256",
    )
    if fit_capability.capability_sha256 != expected_fit_training_sha:
        raise HarmBenchModelError(
            "fit training capability differs from the external binding"
        )
    feature_capability = fit_capability.fit.feature_capability
    expected_fit_feature_sha = _sha256(
        expected_fit_feature_capability_sha256,
        name="expected_fit_feature_capability_sha256",
    )
    if feature_capability.capability_sha256 != expected_fit_feature_sha:
        raise HarmBenchModelError(
            "fit feature capability differs from the external binding"
        )
    try:
        validate_shared_group_crossfit_plan(crossfit_plan, feature_capability)
        train_indices, heldout_indices = resolve_shared_group_crossfit_indices(
            crossfit_plan,
            feature_capability,
            training_seed=training_seed,
            fold=fold,
        )
    except ValueError as error:
        raise HarmBenchModelError(f"crossfit plan changed: {error}") from error
    expected_plan_sha = _sha256(
        expected_crossfit_plan_sha256, name="expected_crossfit_plan_sha256"
    )
    if crossfit_plan.plan_sha256 != expected_plan_sha:
        raise HarmBenchModelError("crossfit plan differs from the external binding")
    seed_value = int(training_seed)
    fold_value = int(fold)
    if not isinstance(processor_receipt, ProcessorReceipt):
        raise HarmBenchModelError("processor receipt type changed")
    expected_processor_sha = _sha256(
        expected_processor_receipt_sha256,
        name="expected_processor_receipt_sha256",
    )
    expected_train_protocol_ids = tuple(
        int(value)
        for value in feature_capability.fit.protocol_row_ids[train_indices]
    )
    if (
        processor_receipt.processor_receipt_sha256 != expected_processor_sha
        or processor_receipt.source_capability_sha256 != expected_fit_feature_sha
        or processor_receipt.crossfit_plan_sha256 != expected_plan_sha
        or processor_receipt.seed != seed_value
        or processor_receipt.fold != fold_value
        or processor_receipt.train_protocol_row_ids != expected_train_protocol_ids
    ):
        raise HarmBenchModelError("processor receipt differs from the live fit fold")
    try:
        processed = validate_processed_role_embeddings(
            processed_features,
            expected_source_capability_sha256=expected_fit_feature_sha,
            expected_processor_receipt_sha256=expected_processor_sha,
            expected_output_receipt_sha256=_sha256(
                expected_processed_output_receipt_sha256,
                name="expected_processed_output_receipt_sha256",
            ),
        )
    except ValueError as error:
        raise HarmBenchModelError(f"processed fit features changed: {error}") from error
    source = feature_capability.fit
    if (
        processed.dataset_id != source.dataset_id
        or processed.role != source.role
        or processed.source_content_sha256 != source.content_sha256
        or processed.source_row_alignment_sha256 != source.row_alignment_sha256
        or not np.array_equal(processed.protocol_row_ids, source.protocol_row_ids)
    ):
        raise HarmBenchModelError("processed fit features differ from the capability")
    return (
        fit_capability,
        processed,
        train_indices,
        heldout_indices,
        seed_value,
        fold_value,
    )


def _physical_examples(
    processed: ProcessedRoleEmbeddings,
    examples: ContextTrainingExamples,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    by_protocol = {
        int(protocol_row_id): index
        for index, protocol_row_id in enumerate(processed.protocol_row_ids)
    }
    try:
        queries = tuple(by_protocol[value] for value in examples.query_protocol_row_ids)
        contexts = tuple(
            tuple(by_protocol[value] for value in context)
            for context in examples.context_protocol_row_ids
        )
    except KeyError as error:
        raise HarmBenchModelError(
            "context training example references an unknown protocol row"
        ) from error
    return queries, contexts


def fit_history_model(
    model_id: str,
    fit_capability: FitRoleCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    context_rosters: Mapping[str, StrictPastContextRoster],
    context_training_examples: ContextTrainingExamples,
    *,
    training_seed: object,
    fold: object,
    epochs: int | None = None,
    expected_fit_training_capability_sha256: str,
    expected_fit_feature_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_context_roster_sha256_by_strategy: Mapping[str, str],
    expected_context_training_examples_sha256: str,
) -> ProductionHistoryCheckpoint:
    """Production history fit; labels and class order are capability-derived."""

    (
        fit_capability,
        processed,
        train_indices,
        heldout_indices,
        seed_value,
        fold_value,
    ) = _validate_production_fit_inputs(
        fit_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        training_seed=training_seed,
        fold=fold,
        expected_fit_training_capability_sha256=(
            expected_fit_training_capability_sha256
        ),
        expected_fit_feature_capability_sha256=(
            expected_fit_feature_capability_sha256
        ),
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
        expected_processed_output_receipt_sha256=(
            expected_processed_output_receipt_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    try:
        examples = validate_context_training_examples(
            context_training_examples,
            context_rosters,
            fit_capability.fit.feature_capability,
            processed,
            processor_receipt,
            crossfit_plan,
            training_seed=seed_value,
            fold=fold_value,
            expected_fit_feature_capability_sha256=(
                expected_fit_feature_capability_sha256
            ),
            expected_processor_receipt_sha256=expected_processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                expected_processed_output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
            expected_context_roster_sha256_by_strategy=(
                expected_context_roster_sha256_by_strategy
            ),
            expected_context_training_examples_sha256=(
                expected_context_training_examples_sha256
            ),
        )
    except ValueError as error:
        raise HarmBenchModelError(
            f"context training examples changed: {error}"
        ) from error
    query_indices, contexts = _physical_examples(processed, examples)
    allowed = set(int(value) for value in train_indices)
    if not set(query_indices).issubset(allowed) or any(
        not set(context).issubset(allowed) for context in contexts
    ):
        raise HarmBenchModelError("history training rows cross the live fit partition")
    # The low-level model core receives a physically train-only feature bank.
    # This keeps heldout embeddings outside every projection/encoder call, even
    # if a future family adds batch-dependent computation.
    train_rows = tuple(int(value) for value in train_indices)
    full_to_train = {full_row: local_row for local_row, full_row in enumerate(train_rows)}
    remapped_queries = tuple(full_to_train[value] for value in query_indices)
    remapped_contexts = tuple(
        tuple(full_to_train[value] for value in context) for context in contexts
    )
    train_features = ProcessedRole(
        text=processed.text[np.asarray(train_rows, dtype=np.int64)],
        audio=processed.audio[np.asarray(train_rows, dtype=np.int64)],
        video=processed.video[np.asarray(train_rows, dtype=np.int64)],
    )
    labels = np.asarray(fit_capability.fit.labels)[np.asarray(query_indices)]
    class_order = tuple(fit_capability.fit.label_order)
    checkpoint = _fit_history_array_core(
        model_id,
        train_features,
        labels,
        remapped_contexts,
        num_classes=len(class_order),
        seed=seed_value,
        epochs=epochs,
        query_indices=remapped_queries,
    )
    first_roster = context_rosters[next(iter(context_rosters))]
    return ProductionHistoryCheckpoint(
        dataset_id=fit_capability.dataset_id,
        model_id=model_id,
        model_namespace=HISTORY_NAMESPACE,
        training_seed=seed_value,
        fold=fold_value,
        class_order=class_order,
        class_order_sha256=class_order_sha256(
            class_order,
            dataset_id=fit_capability.dataset_id,
            fit_training_capability_sha256=fit_capability.capability_sha256,
        ),
        fit_training_capability_sha256=fit_capability.capability_sha256,
        fit_feature_capability_sha256=(
            fit_capability.fit.feature_capability.capability_sha256
        ),
        processor_receipt_sha256=processor_receipt.processor_receipt_sha256,
        processed_output_receipt_sha256=processed.output_receipt_sha256,
        crossfit_plan_sha256=crossfit_plan.plan_sha256,
        context_training_examples_sha256=examples.example_sha256,
        context_roster_manifest_sha256=aggregate_context_roster_sha256(
            examples.context_roster_sha256_by_strategy
        ),
        fit_train_protocol_row_ids_sha256=(
            first_roster.fit_train_protocol_row_ids_sha256
        ),
        fit_heldout_protocol_row_ids_sha256=(
            first_roster.fit_heldout_protocol_row_ids_sha256
        ),
        checkpoint=checkpoint,
    )


def fit_current_only_model(
    model_id: str,
    fit_capability: FitRoleCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    independence_roster: StrictPastContextRoster,
    *,
    training_seed: object,
    fold: object,
    epochs: int | None = None,
    expected_fit_training_capability_sha256: str,
    expected_fit_feature_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_independence_roster_sha256: str,
) -> ProductionCurrentOnlyCheckpoint:
    """Production current-only fit with a live zero-consumption proof."""

    (
        fit_capability,
        processed,
        train_indices,
        heldout_indices,
        seed_value,
        fold_value,
    ) = _validate_production_fit_inputs(
        fit_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        training_seed=training_seed,
        fold=fold,
        expected_fit_training_capability_sha256=(
            expected_fit_training_capability_sha256
        ),
        expected_fit_feature_capability_sha256=(
            expected_fit_feature_capability_sha256
        ),
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
        expected_processed_output_receipt_sha256=(
            expected_processed_output_receipt_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    try:
        independence = validate_strict_past_context_roster(
            independence_roster,
            fit_capability.fit.feature_capability,
            fit_capability.fit.feature_capability,
            processed,
            processor_receipt,
            crossfit_plan,
            training_seed=seed_value,
            fold=fold_value,
            context_role=FIT_TRAIN_CONTEXT_ROLE,
            strategy_id=CURRENT_ONLY_STRATEGY_ID,
            expected_fit_plan_capability_sha256=(
                expected_fit_feature_capability_sha256
            ),
            expected_source_capability_sha256=(
                expected_fit_feature_capability_sha256
            ),
            expected_processor_receipt_sha256=expected_processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                expected_processed_output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
            expected_context_roster_sha256=(
                expected_independence_roster_sha256
            ),
        )
    except ValueError as error:
        raise HarmBenchModelError(
            f"current-only independence proof changed: {error}"
        ) from error
    if (
        any(independence.context_protocol_row_ids)
        or independence.total_context_count != 0
        or independence.history_consumption_count != 0
    ):
        raise HarmBenchModelError("current-only fit consumed context or history")
    by_protocol = {
        int(protocol_row_id): index
        for index, protocol_row_id in enumerate(processed.protocol_row_ids)
    }
    try:
        ordered_train_indices = np.asarray(
            [by_protocol[value] for value in independence.query_protocol_row_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise HarmBenchModelError(
            "current-only proof references an unknown protocol row"
        ) from error
    if set(ordered_train_indices.tolist()) != set(train_indices.tolist()):
        raise HarmBenchModelError("current-only proof differs from the live fit fold")
    feature_subset = ProcessedRole(
        text=processed.text[ordered_train_indices],
        audio=processed.audio[ordered_train_indices],
        video=processed.video[ordered_train_indices],
    )
    labels = np.asarray(fit_capability.fit.labels)[ordered_train_indices]
    class_order = tuple(fit_capability.fit.label_order)
    checkpoint = _fit_current_only_array_core(
        model_id,
        feature_subset,
        labels,
        num_classes=len(class_order),
        seed=seed_value,
        epochs=epochs,
    )
    return ProductionCurrentOnlyCheckpoint(
        dataset_id=fit_capability.dataset_id,
        model_id=model_id,
        model_namespace=CURRENT_ONLY_NAMESPACE,
        training_seed=seed_value,
        fold=fold_value,
        class_order=class_order,
        class_order_sha256=class_order_sha256(
            class_order,
            dataset_id=fit_capability.dataset_id,
            fit_training_capability_sha256=fit_capability.capability_sha256,
        ),
        fit_training_capability_sha256=fit_capability.capability_sha256,
        fit_feature_capability_sha256=(
            fit_capability.fit.feature_capability.capability_sha256
        ),
        processor_receipt_sha256=processor_receipt.processor_receipt_sha256,
        processed_output_receipt_sha256=processed.output_receipt_sha256,
        crossfit_plan_sha256=crossfit_plan.plan_sha256,
        independence_roster_sha256=independence.roster_sha256,
        fit_train_protocol_row_ids_sha256=(
            independence.fit_train_protocol_row_ids_sha256
        ),
        fit_heldout_protocol_row_ids_sha256=(
            independence.fit_heldout_protocol_row_ids_sha256
        ),
        context_count=0,
        history_consumption_count=0,
        checkpoint=checkpoint,
    )


def predict_current_only_model(
    checkpoint: CurrentOnlyCheckpoint, features: ProcessedRoleLike
) -> np.ndarray:
    """Unified current-only probability entry point with no context argument."""

    if not isinstance(
        checkpoint,
        (
            LinearCurrentOnlyCheckpoint,
            DeepSetsCurrentOnlyCheckpoint,
            CausalGRUCurrentOnlyCheckpoint,
        ),
    ):
        raise HarmBenchModelError("checkpoint is not a current-only model")
    return checkpoint.predict_proba(features)


def _validated_production_prediction_checkpoint(
    checkpoint: object, *, history: bool
) -> ProductionHistoryCheckpoint | ProductionCurrentOnlyCheckpoint:
    expected_wrapper = (
        ProductionHistoryCheckpoint if history else ProductionCurrentOnlyCheckpoint
    )
    if not isinstance(checkpoint, expected_wrapper):
        raise HarmBenchModelError("production prediction checkpoint type changed")
    try:
        rebuilt = expected_wrapper(
            **{
                item.name: getattr(checkpoint, item.name)
                for item in fields(expected_wrapper)
            }
        )
    except (TypeError, ValueError) as error:
        raise HarmBenchModelError(
            f"production prediction checkpoint changed: {error}"
        ) from error
    expected_low_level: Mapping[str, type[object]]
    if history:
        expected_low_level = {
            LINEAR_POOL_ID: LinearHistoryCheckpoint,
            DEEPSETS_POOL_ID: DeepSetsHistoryCheckpoint,
            CAUSAL_GRU_ID: CausalGRUHistoryCheckpoint,
        }
    else:
        expected_low_level = {
            LINEAR_POOL_ID: LinearCurrentOnlyCheckpoint,
            DEEPSETS_POOL_ID: DeepSetsCurrentOnlyCheckpoint,
            CAUSAL_GRU_ID: CausalGRUCurrentOnlyCheckpoint,
        }
    if type(rebuilt.checkpoint) is not expected_low_level[rebuilt.model_id]:
        raise HarmBenchModelError(
            "production checkpoint family/namespace implementation changed"
        )
    return rebuilt


def _validate_production_prediction_target(
    checkpoint: ProductionHistoryCheckpoint | ProductionCurrentOnlyCheckpoint,
    fit_plan_capability: FitFeatureCapability,
    source_capability: FitFeatureCapability | SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
) -> tuple[ProcessedRoleEmbeddings, tuple[int, ...], str]:
    """Re-derive target rows and all lineage without caller-supplied digests."""

    try:
        fit_plan = validate_fit_feature_capability(fit_plan_capability)
        validate_shared_group_crossfit_plan(crossfit_plan, fit_plan)
        train_indices, heldout_indices = resolve_shared_group_crossfit_indices(
            crossfit_plan,
            fit_plan,
            training_seed=checkpoint.training_seed,
            fold=checkpoint.fold,
        )
    except (TypeError, ValueError) as error:
        raise HarmBenchModelError(
            f"production prediction fit plan changed: {error}"
        ) from error
    if (
        checkpoint.dataset_id != fit_plan.dataset_id
        or checkpoint.fit_feature_capability_sha256
        != fit_plan.capability_sha256
        or checkpoint.crossfit_plan_sha256 != crossfit_plan.plan_sha256
    ):
        raise HarmBenchModelError(
            "production checkpoint differs from the live prediction fit plan"
        )
    try:
        if isinstance(source_capability, FitFeatureCapability):
            source = validate_fit_feature_capability(source_capability)
            if source.capability_sha256 != fit_plan.capability_sha256:
                raise HarmBenchModelError(
                    "OOF prediction source differs from the fit-plan capability"
                )
            features = source.fit
            context_role = FIT_HELDOUT_OOF_CONTEXT_ROLE
            query_indices = tuple(
                sorted(
                    (int(value) for value in heldout_indices),
                    key=lambda index: int(features.protocol_row_ids[index]),
                )
            )
        elif isinstance(source_capability, SelectionFeatureCapability):
            source = validate_selection_feature_capability(source_capability)
            features = source.selection
            context_role = SELECTION_CONTEXT_ROLE
            query_indices = tuple(
                sorted(
                    range(features.rows),
                    key=lambda index: int(features.protocol_row_ids[index]),
                )
            )
            fit_features = fit_plan.fit
            if (
                set(features.groups.tolist()).intersection(
                    fit_features.groups.tolist()
                )
                or set(features.protocol_row_ids.tolist()).intersection(
                    fit_features.protocol_row_ids.tolist()
                )
                or set(features.keys.tolist()).intersection(
                    fit_features.keys.tolist()
                )
            ):
                raise HarmBenchModelError(
                    "selection prediction source overlaps the fit role"
                )
        else:
            raise HarmBenchModelError(
                "prediction source must be a typed fit or selection capability"
            )
    except HarmBenchModelError:
        raise
    except (TypeError, ValueError) as error:
        raise HarmBenchModelError(
            f"production prediction source capability changed: {error}"
        ) from error
    if (
        source.dataset_id != fit_plan.dataset_id
        or source.cross_role_feature_roster_sha256
        != fit_plan.cross_role_feature_roster_sha256
    ):
        raise HarmBenchModelError(
            "prediction source differs from the fit dataset/feature roster"
        )
    if not isinstance(processor_receipt, ProcessorReceipt):
        raise HarmBenchModelError("processor receipt type changed")
    try:
        rebuilt_receipt = ProcessorReceipt(
            **{
                item.name: getattr(processor_receipt, item.name)
                for item in fields(ProcessorReceipt)
            }
        )
    except (TypeError, ValueError) as error:
        raise HarmBenchModelError(
            f"processor receipt changed before prediction: {error}"
        ) from error
    expected_train_protocol_ids = tuple(
        int(value) for value in fit_plan.fit.protocol_row_ids[train_indices]
    )
    if (
        rebuilt_receipt.dataset_id != fit_plan.dataset_id
        or rebuilt_receipt.source_capability_sha256
        != fit_plan.capability_sha256
        or rebuilt_receipt.cross_role_feature_roster_sha256
        != fit_plan.cross_role_feature_roster_sha256
        or rebuilt_receipt.crossfit_plan_sha256 != crossfit_plan.plan_sha256
        or rebuilt_receipt.seed != checkpoint.training_seed
        or rebuilt_receipt.fold != checkpoint.fold
        or rebuilt_receipt.train_protocol_row_ids
        != expected_train_protocol_ids
        or rebuilt_receipt.processor_receipt_sha256
        != checkpoint.processor_receipt_sha256
    ):
        raise HarmBenchModelError(
            "processor receipt differs from the live prediction fold"
        )
    try:
        processed = validate_processed_role_embeddings(
            processed_features,
            expected_source_capability_sha256=source.capability_sha256,
            expected_processor_receipt_sha256=(
                rebuilt_receipt.processor_receipt_sha256
            ),
            expected_output_receipt_sha256=(
                processed_features.output_receipt_sha256
            ),
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise HarmBenchModelError(
            f"processed prediction target changed: {error}"
        ) from error
    if (
        processed.dataset_id != features.dataset_id
        or processed.role != features.role
        or processed.source_content_sha256 != features.content_sha256
        or processed.source_row_alignment_sha256 != features.row_alignment_sha256
        or processed.cross_role_feature_roster_sha256
        != source.cross_role_feature_roster_sha256
        or not np.array_equal(processed.protocol_row_ids, features.protocol_row_ids)
    ):
        raise HarmBenchModelError(
            "processed prediction target differs from the typed source"
        )
    if not query_indices:
        raise HarmBenchModelError("production prediction target is empty")
    return processed, query_indices, context_role


def predict_production_history(
    checkpoint: ProductionHistoryCheckpoint,
    fit_plan_capability: FitFeatureCapability,
    source_capability: FitFeatureCapability | SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    context_roster: StrictPastContextRoster,
) -> np.ndarray:
    """Predict OOF/selection rows from one live-validated strict-past roster.

    Query and context indices are derived solely from protocol-row IDs in the
    revalidated roster; the production surface has no raw index argument.
    """

    validated = _validated_production_prediction_checkpoint(
        checkpoint, history=True
    )
    if not isinstance(validated, ProductionHistoryCheckpoint):  # pragma: no cover
        raise HarmBenchModelError("history production checkpoint type changed")
    processed, derived_queries, context_role = (
        _validate_production_prediction_target(
            validated,
            fit_plan_capability,
            source_capability,
            processed_features,
            processor_receipt,
            crossfit_plan,
        )
    )
    if (
        not isinstance(context_roster, StrictPastContextRoster)
        or context_roster.strategy_id not in STRICT_PAST_STRATEGY_IDS
    ):
        raise HarmBenchModelError(
            "history prediction requires one frozen strict-past strategy roster"
        )
    try:
        roster = validate_strict_past_context_roster(
            context_roster,
            fit_plan_capability,
            source_capability,
            processed,
            processor_receipt,
            crossfit_plan,
            training_seed=validated.training_seed,
            fold=validated.fold,
            context_role=context_role,
            strategy_id=context_roster.strategy_id,
            expected_fit_plan_capability_sha256=(
                fit_plan_capability.capability_sha256
            ),
            expected_source_capability_sha256=source_capability.capability_sha256,
            expected_processor_receipt_sha256=(
                processor_receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=crossfit_plan.plan_sha256,
            expected_context_roster_sha256=context_roster.roster_sha256,
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise HarmBenchModelError(
            f"history prediction roster changed: {error}"
        ) from error
    by_protocol = {
        int(protocol_row_id): index
        for index, protocol_row_id in enumerate(processed.protocol_row_ids)
    }
    try:
        query_indices = tuple(
            by_protocol[value] for value in roster.query_protocol_row_ids
        )
        contexts = tuple(
            tuple(by_protocol[value] for value in context)
            for context in roster.context_protocol_row_ids
        )
    except KeyError as error:  # live roster validation should make this unreachable.
        raise HarmBenchModelError(
            "history prediction roster references an unknown protocol row"
        ) from error
    if query_indices != derived_queries:
        raise HarmBenchModelError(
            "history prediction queries differ from the derived target partition"
        )
    return validated.checkpoint.predict_proba(
        processed, contexts, query_indices=query_indices
    )


def predict_production_current_only(
    checkpoint: ProductionCurrentOnlyCheckpoint,
    fit_plan_capability: FitFeatureCapability,
    source_capability: FitFeatureCapability | SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
) -> np.ndarray:
    """Predict derived OOF/selection rows with no context/history surface."""

    validated = _validated_production_prediction_checkpoint(
        checkpoint, history=False
    )
    if not isinstance(validated, ProductionCurrentOnlyCheckpoint):  # pragma: no cover
        raise HarmBenchModelError("current-only production checkpoint type changed")
    if (
        validated.context_count != 0
        or validated.history_consumption_count != 0
    ):
        raise HarmBenchModelError(
            "current-only prediction lost its zero-history proof"
        )
    processed, query_indices, context_role = _validate_production_prediction_target(
        validated,
        fit_plan_capability,
        source_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
    )
    # The independent target proof is derived internally, so this public
    # surface still exposes no context/history argument.  It proves that the
    # exact live OOF/selection partition has an all-empty context roster.
    try:
        independence = build_strict_past_context_roster(
            fit_plan_capability,
            source_capability,
            processed,
            processor_receipt,
            crossfit_plan,
            training_seed=validated.training_seed,
            fold=validated.fold,
            context_role=context_role,
            strategy_id=CURRENT_ONLY_STRATEGY_ID,
            expected_fit_plan_capability_sha256=(
                fit_plan_capability.capability_sha256
            ),
            expected_source_capability_sha256=source_capability.capability_sha256,
            expected_processor_receipt_sha256=(
                processor_receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=crossfit_plan.plan_sha256,
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise HarmBenchModelError(
            f"current-only live independence proof changed: {error}"
        ) from error
    if (
        any(independence.context_protocol_row_ids)
        or independence.total_context_count != 0
        or independence.history_consumption_count != 0
        or tuple(independence.query_protocol_row_ids)
        != tuple(int(processed.protocol_row_ids[index]) for index in query_indices)
    ):
        raise HarmBenchModelError(
            "current-only target consumed context or history"
        )
    indices = np.asarray(query_indices, dtype=np.int64)
    target = ProcessedRole(
        text=processed.text[indices],
        audio=processed.audio[indices],
        video=processed.video[indices],
    )
    return validated.checkpoint.predict_proba(target)


__all__ = [
    "AUDIO_DIMENSION",
    "CAUSAL_GRU_ID",
    "CAUSAL_GRU_PARAMETER_LIMIT",
    "CURRENT_ONLY_NAMESPACE",
    "DEEPSETS_PARAMETER_LIMIT",
    "DEEPSETS_POOL_ID",
    "FROZEN_MODEL_IDS",
    "HISTORY_NAMESPACE",
    "HarmBenchModelError",
    "LINEAR_POOL_ID",
    "MODALITY_DIMENSIONS",
    "ProductionCurrentOnlyCheckpoint",
    "ProductionHistoryCheckpoint",
    "TEXT_DIMENSION",
    "VIDEO_DIMENSION",
    "aggregate_context_roster_sha256",
    "class_order_sha256",
    "fit_current_only_model",
    "fit_history_model",
    "predict_production_current_only",
    "predict_production_history",
    "validate_probability_matrix",
]
