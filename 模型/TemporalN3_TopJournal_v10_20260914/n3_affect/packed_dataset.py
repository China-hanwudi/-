"""Adapters for the audited packed CH-SIMS_v2, CMU-MOSEI and M3ED tensors.

History indices are canonicalised to oldest-to-newest, right aligned in K=3
slots.  Invalid indices are masked and never replaced by future samples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import torch


class PackedDataset:
    def __init__(self, path: str | Path, task: str) -> None:
        self.path = Path(path)
        self.task = task
        self.raw = torch.load(self.path, map_location="cpu", weights_only=True)
        required = {"T", "A", "V", "label", "modality_mask"}
        missing = required - set(self.raw)
        if missing:
            raise ValueError(f"{self.path}: missing packed keys {sorted(missing)}")
        self.n = int(self.raw["label"].shape[0])
        if any(int(self.raw[m].shape[0]) != self.n for m in ("T", "A", "V", "modality_mask")):
            raise ValueError(f"{self.path}: inconsistent sample counts")
        if task == "classification":
            self.raw["label"] = self.raw["label"].long()
        elif task == "regression":
            self.raw["label"] = self.raw["label"].float()
        else:
            raise ValueError(task)
        self.has_history = "history_index" in self.raw
        if self.has_history and tuple(self.raw["history_index"].shape) != (self.n, 3):
            raise ValueError("history_index must have shape [N,3]")

    def __len__(self) -> int:
        return self.n

    def _history(self, ix: torch.Tensor) -> dict[str, torch.Tensor]:
        b = int(ix.numel())
        d = self.raw
        dims = {"T": d["T"].shape[-1], "A": d["A"].shape[-1], "V": d["V"].shape[-1]}
        out = {f"{m}_h": torch.zeros(b, 3, dims[m], dtype=d[m].dtype) for m in dims}
        hm = torch.zeros(b, 3, dtype=torch.float32)
        hmm = torch.zeros(b, 3, 3, dtype=torch.float32)
        same = torch.zeros(b, 3, dtype=torch.float32)
        if not self.has_history:
            return {**out, "history_mask": hm, "history_modality_mask": hmm, "speaker_same": same}
        all_idx = d["history_index"]
        all_same = d.get("speaker_same")
        slots = int(all_idx.shape[1])
        for row, sample_i in enumerate(ix.tolist()):
            # The audited packers emit history slots NEWEST-FIRST
            # (M3ED data_audit.json: "previous 3 turns ... newest first";
            #  MOSEI data_audit.json: "newest completed segments").
            #
            # Two earlier assumptions were measured and REJECTED
            # (probe_history_order.py, 2026-09-14):
            #   * "packed row order is chronological by source" - false on MOSEI
            #     where 1738 of 2249 source groups are ordered non-chronologically;
            #   * "drop candidate indices >= the current row index" - this
            #     discarded 28.2% of MOSEI history slots and 100% of the discarded
            #     slots were verified to be genuinely EARLIER segments.
            # So we trust the packer's slot order, keep only in-range ids, and
            # reverse to obtain the canonical oldest-to-newest order, which is
            # then right-aligned.  M3ED is unaffected: its row order is already
            # chronological (only 3 of 76 groups unordered, 0 slots dropped).
            raw_slots = [int(v) for v in all_idx[sample_i].tolist()]
            kept = [(v, pos) for pos, v in enumerate(raw_slots) if 0 <= v < self.n]
            kept.reverse()                      # newest-first -> oldest-first
            kept = kept[-slots:]
            start = slots - len(kept)
            for j, (hidx, src_pos) in enumerate(kept, start=start):
                hm[row, j] = 1.0
                for mi, m in enumerate(("T", "A", "V")):
                    out[f"{m}_h"][row, j] = d[m][hidx]
                    hmm[row, j, mi] = float(d["modality_mask"][hidx, mi] > 0)
                if all_same is not None:
                    # Reorder the speaker metadata with the same canonical slot.
                    same[row, j] = float(all_same[sample_i, src_pos] > 0)
        return {**out, "history_mask": hm, "history_modality_mask": hmm, "speaker_same": same}

    def batch(self, indices: Iterable[int]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        ix = torch.as_tensor(list(indices), dtype=torch.long)
        d = self.raw
        b = {"T_t": d["T"][ix], "A_t": d["A"][ix], "V_t": d["V"][ix],
             "modality_mask": d["modality_mask"][ix].float()}
        b.update(self._history(ix))
        if "unimodal_labels" in d:
            b["unimodal_labels"] = d["unimodal_labels"][ix].float()
        return b, d["label"][ix]


def audit_packed(paths: Iterable[str | Path]) -> dict:
    report = {}
    for p in paths:
        raw = torch.load(Path(p), map_location="cpu", weights_only=True)
        report[str(p)] = {"samples": int(raw["label"].shape[0]),
                          "keys": sorted(raw.keys()),
                          "dims": {k: list(raw[k].shape) for k in ("T", "A", "V")},
                          "label_dtype": str(raw["label"].dtype)}
    return report
