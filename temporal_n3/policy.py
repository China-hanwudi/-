"""Auditable temporal candidate selection policy for Temporal N3 v4.

Production training must persist the realized policy, seed, band boundaries,
candidate identifiers, and hashes in a frozen manifest. These helpers do not
read labels, scores, checkpoints, or test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TemporalPolicy:
    """Reference recent-heavy candidate-pool policy.

    ``max_candidates`` is an explicit development ceiling, not a substitute
    for a production manifest's percentage/realized quota record.
    """
    max_candidates: int = 6
    older_fraction: float = 0.17
    middle_fraction: float = 0.33
    recent_fraction: float = 0.50

    def validate(self) -> None:
        values = (self.older_fraction, self.middle_fraction, self.recent_fraction)
        if self.max_candidates < 1 or min(values) < 0 or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("invalid temporal policy")


def strata(n_history: int) -> tuple[range, range, range]:
    """Return chronological older 50%, middle 30%, recent 20% bands."""
    if n_history < 0:
        raise ValueError("n_history must be non-negative")
    older_end = (n_history * 50) // 100
    middle_end = older_end + (n_history * 30) // 100
    return range(0, older_end), range(older_end, middle_end), range(middle_end, n_history)


def _quota(total: int, fractions: Sequence[float]) -> list[int]:
    raw = [total * fraction for fraction in fractions]
    result = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: (raw[i] - result[i], i), reverse=True):
        if sum(result) == total:
            break
        result[index] += 1
    return result


def initial_candidates(n_history: int, policy: TemporalPolicy = TemporalPolicy()) -> list[int]:
    """Choose a deterministic reference pool, newest first within each band."""
    policy.validate()
    if n_history <= 0:
        return []
    bands = strata(n_history)
    quotas = _quota(min(n_history, policy.max_candidates), (
        policy.older_fraction, policy.middle_fraction, policy.recent_fraction,
    ))
    selected: list[int] = []
    # Reassign unavailable band quota to the remaining newest available turns.
    for band, count in zip(bands, quotas):
        selected.extend(sorted(band, reverse=True)[:count])
    for index in range(n_history - 1, -1, -1):
        if len(selected) >= min(n_history, policy.max_candidates):
            break
        if index not in selected:
            selected.append(index)
    return selected[: min(n_history, policy.max_candidates)]


def refill_after_filter(n_history: int, initial: Sequence[int], retained: Sequence[int], policy: TemporalPolicy = TemporalPolicy()) -> list[int]:
    """Refill from never-selected history, prioritizing highest retention band."""
    policy.validate()
    initial_set = set(initial)
    keep = initial_set.intersection(retained)
    target = min(len(initial_set), policy.max_candidates)
    bands = strata(n_history)
    rates = []
    for band in bands:
        band_set = set(band)
        selected = band_set & initial_set
        rates.append(len(band_set & keep) / len(selected) if selected else 0.0)
    for index in sorted(range(3), key=lambda i: (rates[i], i), reverse=True):
        for candidate in sorted(set(bands[index]) - initial_set, reverse=True):
            if len(keep) >= target:
                return sorted(keep, reverse=True)
            keep.add(candidate)
    return sorted(keep, reverse=True)
