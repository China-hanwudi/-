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
        history_mask = out["history_mask"].to(dtype=torch.float32)
        modality_mask = out["modality_mask"].to(dtype=torch.float32)
        raw_history_modality = feat.get("history_modality_mask")
        if raw_history_modality is None:
            # Legacy features only stored one validity bit per history slot.
            # Expand it to an explicit K x 3 mask without inventing evidence.
            raw_history_modality = history_mask[:, None] * modality_mask[None, :]
        elif raw_history_modality.ndim == 1:
            raw_history_modality = raw_history_modality[:, None] * modality_mask[None, :]
        out["history_modality_mask"] = raw_history_modality.to(dtype=torch.float32)
        # Speaker-conditioned history signal: same-speaker slots are more
        # likely to carry a continuous affective state.  Keep it as metadata
        # so the relation encoder can use it without changing checkpoint shape.
        current_speaker = str(r.get("speaker", "")).strip()
        same = []
        for key in ("h0_speaker", "h1_speaker", "h2_speaker"):
            hist_speaker = str(r.get(key, "")).strip()
            same.append(float(bool(current_speaker) and hist_speaker == current_speaker))
        out["speaker_same"] = torch.tensor(same, dtype=torch.float32)
        if "oof_risk_targets" in feat:
            out["oof_risk_targets"] = feat["oof_risk_targets"].to(dtype=torch.float32)
        out["label"] = torch.tensor(int(r["label_id"]), dtype=torch.long)
        out["vad"] = feat.get("vad", torch.zeros(3))
        return out

    def subset(self, indices: list[int]) -> "MELDFeatureDataset":
        other = MELDFeatureDataset.__new__(MELDFeatureDataset)
        other.feature_dir = self.feature_dir
        other.rows = [self.rows[i] for i in indices]
        return other
