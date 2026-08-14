"""MELD manifest dataset for N3 (official splits only)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class MELDFeatureDataset(Dataset):
    """Reads precomputed .pt features + labels from a manifest CSV.

    Yields the six-way tensors T_t/A_t/V_t and T_h/A_h/V_h, plus
    history_mask (3 binary slot validity), modality_mask (T/A/V present),
    label, and vad.
    """

    def __init__(self, manifest_csv: str | Path, feature_dir: str | Path, *, skip_missing_video: bool = True) -> None:
        self.feature_dir = Path(feature_dir)
        self.rows: list[dict[str, Any]] = []
        with Path(manifest_csv).open(encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                if skip_missing_video and str(r.get("video_missing", "")).lower() in {"1", "true", "True"}:
                    continue
                sid = r["sample_id"]
                fp = self.feature_dir / f"{sid}.pt"
                if fp.is_file():
                    self.rows.append({**r, "feature_path": str(fp)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        feat = torch.load(r["feature_path"], map_location="cpu", weights_only=True)
        out = {k: feat[k] for k in ("T_t", "A_t", "V_t", "T_h", "A_h", "V_h")}
        out["history_mask"] = feat.get("history_mask", torch.ones(3, dtype=torch.float32))
        out["modality_mask"] = feat.get("modality_mask", torch.ones(3, dtype=torch.float32))
        out["history_modality_mask"] = feat.get(
            "history_modality_mask",
            torch.ones(3, dtype=torch.float32) * (out["history_mask"].sum() > 0).to(torch.float32),
        )
        out["label"] = torch.tensor(int(r["label_id"]), dtype=torch.long)
        out["vad"] = feat.get("vad", torch.zeros(3))
        return out

    def subset(self, indices: list[int]) -> "MELDFeatureDataset":
        other = MELDFeatureDataset.__new__(MELDFeatureDataset)
        other.feature_dir = self.feature_dir
        other.rows = [self.rows[i] for i in indices]
        return other
