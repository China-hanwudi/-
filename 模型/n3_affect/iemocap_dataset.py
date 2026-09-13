"""IEMOCAP JSONL + Xiao ComposerN3 six-stream feature dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


FEATURE_KEYS = (
    "T_t", "A_t", "V_t", "T_h", "A_h", "V_h",
    "history_mask", "modality_mask", "history_modality_mask",
    "history_slot_modality_mask", "vad",
)


class IEMOCAPFeatureDataset(Dataset):
    def __init__(self, manifest_jsonl: str | Path, feature_dir: str | Path, *, strict: bool = True) -> None:
        self.manifest_path = Path(manifest_jsonl)
        if "test" in self.manifest_path.name.lower() or "sealed" in self.manifest_path.name.lower():
            raise RuntimeError(f"test/sealed manifest forbidden: {self.manifest_path}")
        self.feature_dir = Path(feature_dir)
        self.rows: list[dict[str, Any]] = []
        missing: list[str] = []
        seen: set[str] = set()
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["utterance_id"])
                if sample_id in seen:
                    raise RuntimeError(f"duplicate sample ID in manifest: {sample_id}")
                seen.add(sample_id)
                if int(row.get("label", -1)) not in {0, 1, 2, 3}:
                    raise RuntimeError(f"invalid four-class label for {sample_id}: {row.get('label')}")
                feature_path = self.feature_dir / f"{sample_id}.pt"
                if feature_path.is_file():
                    self.rows.append({**row, "sample_id": sample_id, "feature_path": str(feature_path)})
                else:
                    missing.append(sample_id)
        if strict and missing:
            raise RuntimeError(f"missing {len(missing)} features, first={missing[:10]}")
        if not self.rows:
            raise RuntimeError(f"no usable features for {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        feature = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        missing = [key for key in FEATURE_KEYS if key not in feature]
        if missing:
            raise RuntimeError(f"feature {row['sample_id']} missing keys: {missing}")
        output = {key: feature[key].to(torch.float32) for key in FEATURE_KEYS}
        # CMER consumes the canonical name; older IEMOCAP features used the
        # more explicit history_slot_modality_mask spelling.
        if output["history_modality_mask"].ndim == 1 and output["history_slot_modality_mask"].ndim >= 2:
            output["history_modality_mask"] = output["history_slot_modality_mask"]
        if output["history_modality_mask"].ndim == 2:
            output["history_modality_mask"] = output["history_modality_mask"] * output["modality_mask"].view(1, -1)
        output["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return output

    @property
    def sample_ids(self) -> set[str]:
        return {str(row["sample_id"]) for row in self.rows}

    @property
    def sessions(self) -> set[str]:
        return {str(row["session_id"]) for row in self.rows}
