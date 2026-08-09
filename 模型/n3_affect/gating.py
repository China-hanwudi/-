"""Two-level risk gates with independent current-only fallback."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TwoLevelGate(nn.Module):
    """Modality keep/drop then joint risk gate."""

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.modality = nn.Sequential(
            nn.Linear(d_model + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.joint = nn.Sequential(
            nn.Linear(d_model + 2 + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        current_pool: Tensor,
        history_streams: dict[str, Tensor],
        utilities: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        kept: list[Tensor] = []
        keep_probs: list[Tensor] = []
        for name in ("T", "A", "V"):
            hist = history_streams[f"{name}_h"]
            util = utilities[f"U_{name}"]
            gate_in = torch.cat([hist, util], dim=-1)
            logit = self.modality(gate_in)
            prob = torch.sigmoid(logit)
            kept.append(hist * prob)
            keep_probs.append(prob)
        modality_kept = torch.stack(kept, dim=1).sum(dim=1)
        keep_stack = torch.cat(keep_probs, dim=-1)
        joint_in = torch.cat([modality_kept, utilities["U_joint"], keep_stack], dim=-1)
        joint_prob = torch.sigmoid(self.joint(joint_in))
        gated_history = modality_kept * joint_prob
        # soft mix with current-only (independent fallback target)
        fused = current_pool + gated_history
        fallback = current_pool
        use_history = (joint_prob > 0.5).float()
        representation = use_history * fused + (1.0 - use_history) * fallback
        return {
            "representation": representation,
            "modality_keep_prob": keep_stack,
            "joint_keep_prob": joint_prob,
            "use_history": use_history,
        }
