"""Feature-only batching and conservative, metadata-grounded past history."""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from typing import Mapping, Sequence

import torch
from torch import Tensor


def select_verified_past_indices(records: Sequence[Mapping], slots: int = 3) -> tuple[list[list[int]], dict]:
    """Return newest-first indices only with source_id/split/start_time/end_time.

    Missing or invalid temporal metadata means all -1; file order is never
    treated as chronology. Candidate end <= current start and candidate start
    < current start are both required. No label/emotion field is consulted.
    """
    if slots < 1:
        raise ValueError("slots must be positive")
    identifiers = [record["sample_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_id values must be unique")
    groups = defaultdict(list)
    metadata = {}
    for index, record in enumerate(records):
        source, split = record.get("source_id"), record.get("split")
        try:
            start, end = float(record["start_time"]), float(record["end_time"])
        except (KeyError, ValueError, TypeError):
            continue
        if source is None or not str(source).strip() or split is None or not str(split).strip():
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            continue
        key = (str(split), str(source))
        metadata[index] = (key, start, end)
        groups[key].append((end, start, index))
    sorted_groups = {key: sorted(group) for key, group in groups.items()}
    end_times = {key: [item[0] for item in group] for key, group in sorted_groups.items()}
    history = [[-1] * slots for _ in records]
    for index, (key, start, _) in metadata.items():
        candidates = sorted_groups[key]
        cutoff = bisect.bisect_right(end_times[key], start)
        selected = []
        for _, other_start, other_index in reversed(candidates[:cutoff]):
            if other_index != index and other_start < start:
                selected.append(other_index)
                if len(selected) == slots:
                    break
        history[index][:len(selected)] = selected
    available = sum(any(index >= 0 for index in row) for row in history)
    return history, {"n": len(records), "verified_temporal_metadata_n": len(metadata),
                     "missing_or_invalid_metadata_n": len(records) - len(metadata),
                     "samples_with_past_history": available, "history_slots": slots,
                     "no_synthetic_history_ids": True,
                     "temporal_mechanism_evaluable": available > 0,
                     "rule": "same source and split; candidate end <= current start and candidate start < current start"}


def masked_mean_pool(sequence: Tensor, valid_steps: Tensor) -> tuple[Tensor, Tensor]:
    """Pool explicit [N,L,D] sequences with supplied [N,L] validity masks.

    This helper never infers missingness from zero-valued features. The caller
    must establish true lengths/masks from the feature data contract.
    """
    if sequence.ndim != 3 or valid_steps.shape != sequence.shape[:2]:
        raise ValueError("Expected sequence [N,L,D] and explicit valid_steps [N,L]")
    if not torch.all((valid_steps == 0) | (valid_steps == 1)):
        raise ValueError("valid_steps must be binary")
    weights = valid_steps.to(sequence.dtype)
    count = weights.sum(dim=1, keepdim=True)
    values = sequence.masked_fill(~valid_steps.bool().unsqueeze(-1), 0.0)
    pooled = values.sum(dim=1) / count.clamp_min(1)
    return pooled, (count[:, 0] > 0).to(sequence.dtype)


def make_six_way_batch(pooled: Mapping[str, Tensor], indices: Tensor, history_index: Tensor,
                       modality_mask: Tensor, speaker_same: Tensor | None = None) -> dict[str, Tensor]:
    """Build the model dictionary from pooled features; never accept targets.

    history_index is produced/audited separately using verified metadata. This
    function checks index bounds but cannot recover missing source timestamps.
    """
    if set(pooled) != {"T", "A", "V"}:
        raise ValueError("pooled must contain exactly T, A, V feature tensors")
    n = pooled["T"].size(0)
    if any(value.ndim != 2 or value.size(0) != n for value in pooled.values()):
        raise ValueError("Pooled features must each have shape [N,D]")
    if history_index.ndim != 2 or history_index.size(0) != n or history_index.size(1) < 1:
        raise ValueError("history_index must be [N,K] with padded slots")
    if history_index.dtype not in (torch.int32, torch.int64) or torch.any((history_index < -1) | (history_index >= n)):
        raise ValueError("history_index values must be integer -1 or valid indices")
    if modality_mask.shape != (n, 3):
        raise ValueError("modality_mask must be [N,3]")
    histories = history_index[indices]
    valid = histories >= 0
    safe = histories.clamp_min(0)
    result = {f"{name}_t": values[indices] for name, values in pooled.items()}
    result.update({f"{name}_h": values[safe] * valid.unsqueeze(-1) for name, values in pooled.items()})
    result.update(history_mask=valid.to(modality_mask.dtype), modality_mask=modality_mask[indices],
                  history_modality_mask=modality_mask[safe] * valid.unsqueeze(-1),
                  speaker_same=torch.zeros_like(valid, dtype=modality_mask.dtype) if speaker_same is None else speaker_same[indices] * valid)
    return result
