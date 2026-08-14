"""Two-level risk gates with independent current-only fallback."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TwoLevelGate(nn.Module):
    """Modality keep/drop then joint risk gate.

    History mixing is a learned soft weight, not a hard 0.5 threshold.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.modality = nn.Sequential(
            nn.Linear(d_model + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # current pool + mixed utility + 3 keep probs
        self.joint = nn.Sequential(
            nn.Linear(d_model + 1 + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.hist_logit = nn.Parameter(torch.zeros(()))

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
        u_mix = utilities.get("U_mix")
        if u_mix is None:
            u_mix = utilities["U_joint"][:, :1]
        joint_in = torch.cat([modality_kept, u_mix, keep_stack], dim=-1)
        joint_prob = torch.sigmoid(self.joint(joint_in))
        hist_scale = torch.sigmoid(self.hist_logit) * joint_prob
        gated_history = modality_kept * hist_scale
        fused = current_pool + gated_history
        return {
            "representation": fused,
            "modality_keep_prob": keep_stack,
            "joint_keep_prob": joint_prob,
            "use_history": hist_scale,
            "hist_mix_weight": torch.sigmoid(self.hist_logit).expand(current_pool.size(0), 1),
        }
