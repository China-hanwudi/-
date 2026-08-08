from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "video")


@dataclass(frozen=True)
class BidirectionalCoalitionTask:
    """One candidate evaluated under two genuinely different set contexts."""

    query_index: int
    addition_context: tuple[int, ...]
    deletion_context: tuple[int, ...]
    candidate_index: int

    def __post_init__(self) -> None:
        addition = tuple(int(value) for value in self.addition_context)
        deletion = tuple(int(value) for value in self.deletion_context)
        if len(addition) != len(set(addition)):
            raise ValueError("addition context contains duplicate history indices")
        if len(deletion) != len(set(deletion)):
            raise ValueError("deletion context contains duplicate history indices")
        if self.candidate_index in addition:
            raise ValueError("candidate must not already be in the addition context")
        if self.candidate_index not in deletion:
            raise ValueError("candidate must be present in the deletion context")
        trivial_deletion = set(addition) | {int(self.candidate_index)}
        if set(deletion) == trivial_deletion:
            raise ValueError(
                "deletion context must differ from addition_context union candidate; "
                "otherwise forward and backward utilities are algebraically identical"
            )


@dataclass(frozen=True)
class BidirectionalUtilityTargets:
    forward_addition: np.ndarray
    backward_deletion: np.ndarray
    asymmetry: np.ndarray
    sign_agreement: np.ndarray


def _capped_history(
    history: Sequence[int],
    maximum_candidates: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    ordered = tuple(dict.fromkeys(int(value) for value in history))
    if len(ordered) <= maximum_candidates:
        return ordered
    recent_count = max(1, maximum_candidates // 2)
    recent = ordered[-recent_count:]
    older = np.asarray(ordered[:-recent_count], dtype=int)
    sampled = tuple(
        sorted(
            int(value)
            for value in rng.choice(
                older,
                size=maximum_candidates - recent_count,
                replace=False,
            )
        )
    )
    return sampled + recent


def _random_subset(values: Sequence[int], rng: np.random.Generator) -> tuple[int, ...]:
    values_array = np.asarray(tuple(values), dtype=int)
    cardinality = int(rng.integers(0, len(values_array) + 1))
    if cardinality == 0:
        return ()
    return tuple(
        sorted(int(value) for value in rng.choice(values_array, size=cardinality, replace=False))
    )


def sample_bidirectional_coalition_tasks(
    histories: Sequence[Sequence[int]],
    *,
    draws_per_query: int,
    maximum_candidates: int,
    seed: int,
) -> list[BidirectionalCoalitionTask]:
    """Sample different-set addition/deletion contexts without using labels."""

    if draws_per_query < 1:
        raise ValueError("draws_per_query must be positive")
    if maximum_candidates < 2:
        raise ValueError("maximum_candidates must be at least two for non-trivial tasks")

    tasks: list[BidirectionalCoalitionTask] = []
    for query_index, history in enumerate(histories):
        query_rng = np.random.default_rng(np.random.SeedSequence([seed, query_index]))
        candidates = _capped_history(history, maximum_candidates, query_rng)
        if len(candidates) < 2:
            continue
        seen: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()
        attempts = 0
        while len(seen) < draws_per_query and attempts < 100 * draws_per_query:
            attempts += 1
            candidate = int(query_rng.choice(np.asarray(candidates, dtype=int)))
            remainder = tuple(value for value in candidates if value != candidate)
            addition = _random_subset(remainder, query_rng)
            deletion_companions = _random_subset(remainder, query_rng)
            if set(deletion_companions) == set(addition):
                continue
            deletion = tuple(sorted(deletion_companions + (candidate,)))
            key = (addition, deletion, candidate)
            if key in seen:
                continue
            task = BidirectionalCoalitionTask(
                query_index=query_index,
                addition_context=addition,
                deletion_context=deletion,
                candidate_index=candidate,
            )
            seen.add(key)
            tasks.append(task)
    return tasks


def _negative_log_likelihood(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or len(labels) != len(probability):
        raise ValueError("labels and probability rows must align")
    if np.any((labels < 0) | (labels >= probability.shape[1])):
        raise ValueError("label outside probability class range")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    rows = np.arange(len(labels))
    return -np.log(np.clip(probability[rows, labels], 1e-12, 1.0))


def bidirectional_utility_targets(
    labels: np.ndarray,
    probability_s: np.ndarray,
    probability_s_plus_candidate: np.ndarray,
    probability_t: np.ndarray,
    probability_t_minus_candidate: np.ndarray,
) -> BidirectionalUtilityTargets:
    """Return benefit-positive forward and backward utility targets.

    Forward addition: L(S) - L(S union {h_i}).
    Backward deletion: L(T without {h_i}) - L(T).
    Positive values mean that retaining the candidate improves prediction.
    The caller must construct T under the non-trivial coalition contract.
    """

    arrays = [
        np.asarray(probability_s, dtype=float),
        np.asarray(probability_s_plus_candidate, dtype=float),
        np.asarray(probability_t, dtype=float),
        np.asarray(probability_t_minus_candidate, dtype=float),
    ]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("all probability arrays must have the same shape")
    loss_s, loss_s_plus, loss_t, loss_t_minus = (
        _negative_log_likelihood(labels, value) for value in arrays
    )
    forward = loss_s - loss_s_plus
    backward = loss_t_minus - loss_t
    return BidirectionalUtilityTargets(
        forward_addition=forward,
        backward_deletion=backward,
        asymmetry=forward - backward,
        sign_agreement=np.sign(forward) == np.sign(backward),
    )


def _aligned_modalities(
    values: Mapping[str, np.ndarray],
    *,
    label: str,
) -> tuple[list[np.ndarray], int, int]:
    missing = set(MODALITIES) - set(values)
    extra = set(values) - set(MODALITIES)
    if missing or extra:
        raise ValueError(f"{label} modalities mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    arrays = [np.asarray(values[name], dtype=float) for name in MODALITIES]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError(f"{label} modality embeddings must be two-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"{label} modality embeddings must share row count and aligned dimension")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{label} modality embeddings contain non-finite values")
    rows, dimension = arrays[0].shape
    return arrays, rows, dimension


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def build_three_by_three_relation_features(
    current_aligned: Mapping[str, np.ndarray],
    history_aligned: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build label-free features for all 3 current x 3 history relations.

    Inputs are post-projection embeddings in a shared aligned space. Raw WavLM,
    DINOv2, and text embeddings must not be passed before their train-only
    projection layers have produced a common dimension.
    """

    current, rows, dimension = _aligned_modalities(current_aligned, label="current")
    history, history_rows, history_dimension = _aligned_modalities(history_aligned, label="history")
    if (rows, dimension) != (history_rows, history_dimension):
        raise ValueError("current and history embeddings must share row count and aligned dimension")

    features: list[np.ndarray] = []
    names: list[str] = []
    for current_name, current_value in zip(MODALITIES, current):
        for history_name, history_value in zip(MODALITIES, history):
            cosine = _row_cosine(current_value, history_value)
            l2 = np.linalg.norm(current_value - history_value, axis=1)
            signed_mean_delta = (current_value - history_value).mean(axis=1)
            features.extend([cosine, l2, signed_mean_delta])
            prefix = f"current_{current_name}__history_{history_name}"
            names.extend([f"{prefix}__cosine", f"{prefix}__l2", f"{prefix}__mean_delta"])
    matrix = np.column_stack(features).astype(np.float32, copy=False)
    if matrix.shape != (rows, 27):
        raise AssertionError("3x3 relation schema must contain 27 features")
    return matrix, tuple(names)


def build_emotion_state_features(
    current_vad: np.ndarray,
    history_vad: np.ndarray,
    *,
    current_shift_probability: np.ndarray | None = None,
    history_shift_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create explicit VAD/appraisal-state change features without gold labels."""

    current = np.asarray(current_vad, dtype=float)
    history = np.asarray(history_vad, dtype=float)
    if current.ndim != 2 or current.shape[1] != 3 or current.shape != history.shape:
        raise ValueError("current_vad and history_vad must have aligned shape (rows, 3)")
    if not np.isfinite(current).all() or not np.isfinite(history).all():
        raise ValueError("VAD features contain non-finite values")
    rows = len(current)
    current_shift = (
        np.zeros(rows, dtype=float)
        if current_shift_probability is None
        else np.asarray(current_shift_probability, dtype=float)
    )
    history_shift = (
        np.zeros(rows, dtype=float)
        if history_shift_probability is None
        else np.asarray(history_shift_probability, dtype=float)
    )
    if current_shift.shape != (rows,) or history_shift.shape != (rows,):
        raise ValueError("shift probabilities must be one-dimensional and row-aligned")
    if np.any((current_shift < 0) | (current_shift > 1)) or np.any(
        (history_shift < 0) | (history_shift > 1)
    ):
        raise ValueError("shift probabilities must lie in [0, 1]")

    delta = current - history
    values = [
        current[:, 0],
        current[:, 1],
        current[:, 2],
        history[:, 0],
        history[:, 1],
        history[:, 2],
        delta[:, 0],
        delta[:, 1],
        delta[:, 2],
        np.abs(delta[:, 0]),
        np.abs(delta[:, 1]),
        np.abs(delta[:, 2]),
        np.linalg.norm(delta, axis=1),
        _row_cosine(current, history),
        current_shift,
        history_shift,
    ]
    names = (
        "current_valence",
        "current_arousal",
        "current_dominance",
        "history_valence",
        "history_arousal",
        "history_dominance",
        "vad_delta_valence",
        "vad_delta_arousal",
        "vad_delta_dominance",
        "vad_abs_delta_valence",
        "vad_abs_delta_arousal",
        "vad_abs_delta_dominance",
        "vad_transition_l2",
        "vad_state_cosine",
        "current_shift_probability",
        "history_shift_probability",
    )
    matrix = np.column_stack(values).astype(np.float32, copy=False)
    return matrix, names


def deterministic_protocol_bucket(dataset: str, group: str, protocol: str) -> int:
    payload = f"{dataset}\x1f{group}\x1f{protocol}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
