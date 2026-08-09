"""Bidirectional marginal utility heads and U_cross residual."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BidirectionalUtilityHeads(nn.Module):
    """Predict modality-level and joint utilities from relation features."""

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        # forward / backward for T,A,V and joint
        self.heads = nn.ModuleDict(
            {
                "T": nn.Linear(hidden, 2),
                "A": nn.Linear(hidden, 2),
                "V": nn.Linear(hidden, 2),
                "joint": nn.Linear(hidden, 2),
            }
        )

    def forward(self, current_pool: Tensor, relation: Tensor) -> dict[str, Tensor]:
        h = self.backbone(torch.cat([current_pool, relation], dim=-1))
        u_t = self.heads["T"](h)
        u_a = self.heads["A"](h)
        u_v = self.heads["V"](h)
        u_joint = self.heads["joint"](h)
        # U_cross = joint_forward - sum(modality_forward)
        u_cross = u_joint[:, :1] - (u_t[:, :1] + u_a[:, :1] + u_v[:, :1])
        return {
            "U_T": u_t,
            "U_A": u_a,
            "U_V": u_v,
            "U_joint": u_joint,
            "U_cross": u_cross,
        }
