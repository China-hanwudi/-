"""Read the visual_feature_v1 schema (analysis/visual_audit/FEATURE_SCHEMA.json).

Only train/valid packs are reachable; an unexpected split is refused.
Face features default to absent (face_valid=False) until an audited face
detector exists (VISUAL_AUDIT.md §3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ALLOWED_SPLITS = ("train", "valid")
REQUIRED_KEYS = ("uid", "frame_index", "timestamp_sec", "scene_feat",
                 "face_feat", "scene_valid", "face_valid", "unique_frame")


class VisualPack:
    def __init__(self, path, split: str):
        if split not in ALLOWED_SPLITS:
            raise ValueError("split %r refused (train/valid only)" % (split,))
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        for k in REQUIRED_KEYS:
            if k not in d:
                raise ValueError("%s: missing schema key %r" % (self.path, k))
        self.d = {k: d[k] for k in d.files}
        self.n = int(self.d["scene_feat"].shape[0])
        if self.d["scene_feat"].shape[1] != 16:
            raise ValueError("schema expects 16 frame slots")

    def batch(self, idx):
        ix = np.asarray(list(idx), dtype=np.int64)
        out = {
            "scene_feat": torch.tensor(self.d["scene_feat"][ix]).float(),
            "face_feat": torch.tensor(self.d["face_feat"][ix]).float(),
            "scene_valid": torch.tensor(self.d["scene_valid"][ix]).float(),
            "face_valid": torch.tensor(self.d["face_valid"][ix]).float(),
            "unique_frame": torch.tensor(self.d["unique_frame"][ix]).float(),
        }
        return out
