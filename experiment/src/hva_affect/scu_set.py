from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CounterfactualTask:
    query_index: int
    subset_indices: tuple[int, ...]
    candidate_index: int

    def __post_init__(self) -> None:
        if self.candidate_index in self.subset_indices:
            raise ValueError("candidate must not already be in the subset")
        if len(self.subset_indices) != len(set(self.subset_indices)):
            raise ValueError("subset contains duplicate history indices")


def assign_group_role(
    dataset: str,
    group: str,
    protocol: str,
    role_ranges: Mapping[str, Sequence[int]],
) -> tuple[str, int]:
    payload = f"{dataset}\x1f{group}\x1f{protocol}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    matched = [name for name, bounds in role_ranges.items() if int(bounds[0]) <= bucket <= int(bounds[1])]
    if len(matched) != 1:
        raise ValueError(f"group bucket {bucket} maps to {len(matched)} roles")
    return matched[0], bucket


def _capped_candidates(
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
    sample_count = maximum_candidates - recent_count
    sampled = tuple(sorted(int(value) for value in rng.choice(older, size=sample_count, replace=False)))
    return sampled + recent


def sample_counterfactual_tasks(
    histories: Sequence[Sequence[int]],
    *,
    subset_draws_per_query: int,
    maximum_candidates: int,
    seed: int,
) -> list[CounterfactualTask]:
    if subset_draws_per_query < 1:
        raise ValueError("subset_draws_per_query must be positive")
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    tasks: list[CounterfactualTask] = []
    for query_index, history in enumerate(histories):
        query_rng = np.random.default_rng(np.random.SeedSequence([seed, query_index]))
        candidates = _capped_candidates(history, maximum_candidates, query_rng)
        if not candidates:
            continue
        seen: set[tuple[tuple[int, ...], int]] = set()
        attempts = 0
        target = min(subset_draws_per_query, max(1, len(candidates) * (2 ** min(len(candidates) - 1, 8))))
        while len(seen) < target and attempts < 20 * target:
            attempts += 1
            candidate = int(query_rng.choice(np.asarray(candidates, dtype=int)))
            remainder = np.asarray([value for value in candidates if value != candidate], dtype=int)
            cardinality = int(query_rng.integers(0, len(remainder) + 1))
            subset = tuple(sorted(int(value) for value in query_rng.choice(remainder, size=cardinality, replace=False)))
            key = (subset, candidate)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(CounterfactualTask(query_index, subset, candidate))
    return tasks


def conditional_utility_targets(
    labels: np.ndarray,
    probability_without: np.ndarray,
    probability_with: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    without = np.asarray(probability_without, dtype=float)
    with_candidate = np.asarray(probability_with, dtype=float)
    if without.shape != with_candidate.shape or without.ndim != 2:
        raise ValueError("probability arrays must have the same two-dimensional shape")
    if len(labels) != len(without):
        raise ValueError("label and probability row counts differ")
    if np.any((labels < 0) | (labels >= without.shape[1])):
        raise ValueError("label outside probability class range")
    rows = np.arange(len(labels))
    without_loss = -np.log(np.clip(without[rows, labels], 1e-12, 1.0))
    with_loss = -np.log(np.clip(with_candidate[rows, labels], 1e-12, 1.0))
    return with_loss - without_loss


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def build_pair_features(
    current_probability: np.ndarray,
    subset_probability: np.ndarray,
    with_candidate_probability: np.ndarray,
    current_embedding: np.ndarray,
    subset_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    history_count: np.ndarray,
    subset_count: np.ndarray,
    recency_rank: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    probabilities = [
        np.asarray(current_probability, dtype=float),
        np.asarray(subset_probability, dtype=float),
        np.asarray(with_candidate_probability, dtype=float),
    ]
    embeddings = [
        np.asarray(current_embedding, dtype=float),
        np.asarray(subset_embedding, dtype=float),
        np.asarray(candidate_embedding, dtype=float),
    ]
    rows = len(probabilities[0])
    if any(value.ndim != 2 or len(value) != rows for value in probabilities + embeddings):
        raise ValueError("all probability and embedding inputs must be aligned two-dimensional arrays")
    if len({value.shape[1] for value in probabilities}) != 1:
        raise ValueError("probability class dimensions differ")
    if len({value.shape[1] for value in embeddings}) != 1:
        raise ValueError("embedding dimensions differ")

    current_p, subset_p, with_p = probabilities
    current_e, subset_e, candidate_e = embeddings
    values: list[np.ndarray] = []
    names: list[str] = []
    for prefix, probability in (
        ("current", current_p),
        ("subset", subset_p),
        ("with_candidate", with_p),
    ):
        for class_index in range(probability.shape[1]):
            values.append(probability[:, class_index])
            names.append(f"{prefix}_probability_{class_index}")
        values.extend([probability.max(axis=1), _entropy(probability)])
        names.extend([f"{prefix}_confidence", f"{prefix}_entropy"])

    for prefix, left, right in (
        ("with_minus_subset", with_p, subset_p),
        ("subset_minus_current", subset_p, current_p),
        ("with_minus_current", with_p, current_p),
    ):
        delta = left - right
        values.extend([np.abs(delta).sum(axis=1), np.linalg.norm(delta, axis=1)])
        names.extend([f"{prefix}_l1", f"{prefix}_l2"])

    for prefix, left, right in (
        ("current_candidate", current_e, candidate_e),
        ("subset_candidate", subset_e, candidate_e),
        ("current_subset", current_e, subset_e),
    ):
        values.extend([_row_cosine(left, right), np.linalg.norm(left - right, axis=1)])
        names.extend([f"{prefix}_cosine", f"{prefix}_l2"])

    history_count = np.asarray(history_count, dtype=float)
    subset_count = np.asarray(subset_count, dtype=float)
    recency_rank = np.asarray(recency_rank, dtype=float)
    if any(len(value) != rows for value in (history_count, subset_count, recency_rank)):
        raise ValueError("count and recency inputs must align with feature rows")
    values.extend(
        [
            np.log1p(history_count),
            np.log1p(subset_count),
            recency_rank / np.clip(history_count, 1.0, None),
        ]
    )
    names.extend(["log1p_history_count", "log1p_subset_count", "relative_recency_rank"])
    matrix = np.column_stack(values).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("pair feature matrix contains non-finite values")
    return matrix, tuple(names)


def sequential_reversible_selection(
    candidates: Sequence[int],
    upper_bound: Callable[[int, tuple[int, ...]], float],
    *,
    maximum_selected: int,
) -> tuple[int, ...]:
    if maximum_selected < 0:
        raise ValueError("maximum_selected must be non-negative")
    remaining = list(dict.fromkeys(int(value) for value in candidates))
    selected: list[int] = []
    while remaining and len(selected) < maximum_selected:
        scored = [(float(upper_bound(candidate, tuple(selected))), candidate) for candidate in remaining]
        score, candidate = min(scored, key=lambda item: (item[0], item[1]))
        if not np.isfinite(score):
            raise ValueError("upper bound callback returned a non-finite value")
        if score >= 0:
            break
        selected.append(candidate)
        remaining.remove(candidate)
    return tuple(selected)
