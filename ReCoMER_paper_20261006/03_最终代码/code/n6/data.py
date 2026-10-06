"""Packed-dataset adapter for model6 with the split guard.

The pack contract (verified against the M3ED/MOSEI/CH-SIMS packs):

* M3ED / CMU-MOSEI: keys ``T``, ``A``, ``V``, ``ids``, ``label``,
  ``modality_mask [N,3]``, ``history_index [N,3]`` (MOSEI has it too),
  ``speaker_same [N,3]``; M3ED additionally ships ``multi_label [N,7]``
  (ignored here -- that direction is closed).
* CH-SIMS_v2: keys ``ids``, ``label`` (float), ``unimodal_labels [N,3]``
  (ignored), ``T``, ``A``, ``V``; NO ``history_index`` and NO
  ``modality_mask`` (treated as all-ones, K = 0 history).

History-index semantics (from the audited reference adapter): raw slots are
NEWEST-first; only in-range indices ``0 <= v < n`` are kept; they are reversed
to canonical oldest-to-newest order and RIGHT-aligned in K slots; invalid
slots stay masked (value 0) and are never replaced by future samples.

Batches are plain python-list indexing (no DataLoader workers), exactly like
the reference trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

# The only two split files the trainer/evaluator is allowed to open.  Kept as
# a module constant so the guard is greppable and RUN_METADATA can quote it.
TRAINABLE_SPLITS = ("train.pt", "valid.pt")

MODALITIES = ("T", "A", "V")


def open_split(directory, name, task="cls"):
    """Open ``directory/name`` refusing any basename that is not train/valid.

    The sealed test split is off-limits to every training, selection and
    evaluation path; this guard makes the refusal explicit and auditable.
    """
    if name not in TRAINABLE_SPLITS:
        raise SystemExit(
            "refusing to open %s: only %s may be read (the sealed test split "
            "is off-limits to every training and model-selection path)"
            % (name, " / ".join(TRAINABLE_SPLITS)))
    return PackedDataset(Path(directory) / name, task=task)


def open_final_test(directory, task="cls"):
    """Open the sealed test split only for an explicit final evaluation.

    Training and validation continue to use :func:`open_split`, whose guard
    rejects ``test.pt``.  This separate entry point makes the final-test
    ceremony auditable and prevents accidental test use during selection.
    """
    return PackedDataset(Path(directory) / "test.pt", task=task,
                         allow_test=True)


class PackedDataset:
    """Adapter around one packed ``train.pt`` / ``valid.pt`` tensor dict."""

    def __init__(self, path, task="cls", allow_test=False) -> None:
        self.path = Path(path)
        allowed = TRAINABLE_SPLITS + (("test.pt",) if allow_test else ())
        if self.path.name not in allowed:
            raise SystemExit(
                "refusing to open %s: only %s may be read" %
                (self.path, " / ".join(allowed)))
        raw = torch.load(self.path, map_location="cpu", weights_only=True)
        for k in ("T", "A", "V", "label"):
            if k not in raw:
                raise ValueError("%s: missing packed key %r" % (self.path, k))
        self.raw = raw
        self.task = task
        self.n = int(raw["label"].shape[0])
        for m in MODALITIES:
            if int(raw[m].shape[0]) != self.n:
                raise ValueError("%s: inconsistent sample counts" % self.path)
        if task == "cls":
            self.raw["label"] = self.raw["label"].long()
        elif task == "reg":
            self.raw["label"] = self.raw["label"].float()
        else:
            raise ValueError("unknown task %r" % (task,))

        if "modality_mask" in raw:
            self.modality_mask = raw["modality_mask"].float()
        elif "raw_present" in raw:
            # CH-SIMS full pack: the builder's presence flags for the
            # zero-filled rows.  raw_present is [N, 2] bool for (A, V) --
            # text is always present -- so expand to [N, 3] with T = 1.
            rp = raw["raw_present"].bool()
            if rp.shape != (self.n, 2):
                raise ValueError("raw_present must have shape [N,2], got %r"
                                 % (tuple(rp.shape),))
            self.modality_mask = torch.cat(
                [torch.ones(self.n, 1, dtype=torch.float32), rp.float()],
                dim=1)
        else:
            # CH-SIMS_v2: no mask key -> every modality present.
            self.modality_mask = torch.ones(self.n, 3, dtype=torch.float32)

        has_hi = "history_index" in raw and raw["history_index"] is not None
        if has_hi:
            hi = raw["history_index"]
            if hi.dim() != 2 or int(hi.shape[0]) != self.n:
                raise ValueError("%s: bad history_index shape %r" % (self.path, hi.shape))
            self.k = int(hi.shape[1])
        else:
            self.k = 0
        self.has_history = self.k > 0
        self.dims = {m: int(raw[m].shape[-1]) for m in MODALITIES}

    def __len__(self) -> int:
        return self.n

    def _history(self, ix: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = int(ix.numel())
        d = self.raw
        out = {
            "T_h": torch.zeros(b, self.k, d["T"].shape[-1], dtype=torch.float32),
            "A_h": torch.zeros(b, self.k, d["A"].shape[-1], dtype=torch.float32),
            "V_h": torch.zeros(b, self.k, d["V"].shape[-1], dtype=torch.float32),
        }
        hm = torch.zeros(b, self.k, dtype=torch.float32)
        hl = torch.full((b, self.k), -100, dtype=torch.long)
        hmm = torch.zeros(b, self.k, 3, dtype=torch.float32)
        # Slot-level metadata REORDERED with the same permutation as the
        # history features (bugfix 2, audit 2026-09-21): the raw pack stores
        # speaker flags in newest-first slot order, but the features are
        # canonicalised to oldest-first, right-aligned -- so the flags must
        # follow the preserved source-position mapping.
        same = torch.zeros(b, self.k, dtype=torch.float32)
        if not self.has_history:
            return {"history_mask": hm, "history_modality_mask": hmm,
                    "speaker_same": same, "history_label": hl, **out}
        all_idx = d["history_index"]
        raw_same = d.get("speaker_same")
        k = self.k
        for row, sample_i in enumerate(ix.tolist()):
            raw_slots = [int(v) for v in all_idx[sample_i].tolist()]
            kept = [(v, pos) for pos, v in enumerate(raw_slots) if 0 <= v < self.n]
            kept.reverse()               # newest-first -> oldest-first
            kept = kept[-k:]             # keep the most recent, right-align
            start = k - len(kept)
            for j, (hidx, src_pos) in enumerate(kept, start=start):
                hm[row, j] = 1.0
                hl[row, j] = int(d["label"][hidx])
                if raw_same is not None:
                    same[row, j] = float(raw_same[sample_i, src_pos] > 0)
                for mi, m in enumerate(MODALITIES):
                    out[m + "_h"][row, j] = d[m][hidx]
                    hmm[row, j, mi] = float(self.modality_mask[hidx, mi] > 0)
        return {"history_mask": hm, "history_modality_mask": hmm,
                "speaker_same": same, "history_label": hl, **out}

    def batch(self, indices: Iterable[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ix = torch.as_tensor(list(indices), dtype=torch.long)
        d = self.raw
        b = {
            "T_t": d["T"][ix], "A_t": d["A"][ix], "V_t": d["V"][ix],
            "modality_mask": self.modality_mask[ix],
        }
        b.update(self._history(ix))
        if "speaker_same" not in d:
            # no speaker metadata in the pack: drop the zero placeholder so
            # consumers treat every memory slot as readable
            del b["speaker_same"]
        if "Vf" in d:
            # frame-level video features [N, K, 512] (W9 V-path); carried only
            # when the pack provides them
            b["Vf"] = d["Vf"][ix].float()
        return b, self.raw["label"][ix]

    def class_counts(self, num_classes: int) -> torch.Tensor:
        """A01/P0-02: requires the resolved class count; out-of-range
        labels raise (no silent clamp)."""
        if self.task != "cls":
            raise ValueError("class_counts only defined for cls")
        y = self.raw["label"].long()
        if (y < 0).any() or (y >= num_classes).any():
            raise ValueError(
                "labels out of range for num_classes=%d" % num_classes)
        return torch.bincount(y, minlength=num_classes)


class IndexedPackedDataset:
    """A non-copying dataset view that keeps history indices globally valid.

    `PackedDataset._history` interprets history_index against the complete
    packed split.  Slicing raw tensors would silently make those indices point
    at different utterances.  This view maps local fit/dev positions back to
    the full train split before batching, while exposing subset labels for
    class weights and diagnostics.
    """

    def __init__(self, base: PackedDataset, indices: Iterable[int], name: str) -> None:
        self.base = base
        self.indices = torch.as_tensor(list(indices), dtype=torch.long)
        if self.indices.dim() != 1 or self.indices.numel() == 0:
            raise ValueError("%s indices must be a non-empty vector" % name)
        if (self.indices < 0).any() or (self.indices >= base.n).any():
            raise ValueError("%s indices out of range" % name)
        if int(self.indices.unique().numel()) != int(self.indices.numel()):
            raise ValueError("%s indices contain duplicates" % name)
        self.name = name
        self.path = base.path
        self.task = base.task
        self.n = int(self.indices.numel())
        self.k = base.k
        self.has_history = base.has_history
        self.dims = base.dims
        # Trainers only consume label and ids from raw.  Avoid cloning the
        # feature tensors; `batch` below resolves all feature reads in base.
        self.raw = {"label": base.raw["label"][self.indices]}
        if "ids" in base.raw:
            self.raw["ids"] = [base.raw["ids"][int(i)] for i in self.indices.tolist()]

    def __len__(self) -> int:
        return self.n

    def batch(self, indices: Iterable[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        local = torch.as_tensor(list(indices), dtype=torch.long)
        if (local < 0).any() or (local >= self.n).any():
            raise ValueError("%s batch indices out of range" % self.name)
        return self.base.batch(self.indices[local].tolist())

    def class_counts(self, num_classes: int) -> torch.Tensor:
        y = self.raw["label"].long()
        if (y < 0).any() or (y >= num_classes).any():
            raise ValueError("labels out of range for num_classes=%d" % num_classes)
        return torch.bincount(y, minlength=num_classes)


def split_sha256(ds: PackedDataset) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(ds.path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
