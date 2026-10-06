"""4-window temporal adapter (plan 00 S4.1/S4.2).

[16, D] scene/face frames + validity -> [B, 4, 192] window tokens.
First version: masked MEAN pooling per window (the temporal-conv variant is
a CONTROL, not stacked on top).  Duplicated/padded frames
(unique_frame=False) never count as independent observations: they are
excluded from the pooling mask.  A window with no valid frames emits a
zero token AND a window_valid flag of 0 (the downstream corrector masks
it out of attention).
"""
from __future__ import annotations

import torch
import torch.nn as nn

WINDOWS = 4
POINTS = 4
D_MODEL = 192


class TemporalAdapter(nn.Module):
    def __init__(self, frame_dim: int = 512, d_model: int = D_MODEL):
        super().__init__()
        self.proj_scene = nn.Linear(frame_dim, d_model)
        self.proj_face = nn.Linear(frame_dim, d_model)
        self.register_buffer("mean", torch.zeros(frame_dim))
        self.register_buffer("std", torch.ones(frame_dim))

    def fit_stats(self, scene_feat: torch.Tensor, mask: torch.Tensor) -> None:
        """Train-partition-only stats over observed (valid, unique) frames."""
        sel = mask > 0
        x = scene_feat[sel]
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, scene_feat, face_feat, scene_valid, face_valid,
                unique_frame):
        """scene/face_feat [B,16,D]; masks [B,16] -> tokens [B,4,192],
        window_valid [B,4]."""
        B = scene_feat.shape[0]
        z = (scene_feat - self.mean) / self.std
        ps = self.proj_scene(z)
        pf = self.proj_face(face_feat)
        obs = (scene_valid * unique_frame).view(B, WINDOWS, POINTS)
        fv = (face_valid * unique_frame).view(B, WINDOWS, POINTS)
        tokens = torch.zeros(B, WINDOWS, ps.shape[-1],
                             device=ps.device, dtype=ps.dtype)
        wvalid = torch.zeros(B, WINDOWS, device=ps.device, dtype=ps.dtype)
        for w in range(WINDOWS):
            m = obs[:, w].unsqueeze(-1)                     # [B,4,1]
            denom = m.sum(1).clamp_min(1.0)
            scene_tok = (ps.view(B, WINDOWS, POINTS, -1)[:, w] * m).sum(1) / denom
            fm = fv[:, w].unsqueeze(-1)
            fdenom = fm.sum(1).clamp_min(1.0)
            face_tok = (pf.view(B, WINDOWS, POINTS, -1)[:, w] * fm).sum(1) / fdenom
            tokens[:, w] = scene_tok + face_tok
            wvalid[:, w] = (m.sum(1).squeeze(-1) > 0).to(ps.dtype)
        return tokens, wvalid
