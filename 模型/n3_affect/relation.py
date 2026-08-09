"""Shared low-rank 3x3 current-history relation grid."""

from __future__ import annotations

import torch
from torch import Tensor, nn


MODALITIES = ("T", "A", "V")


class SharedThreeByThree(nn.Module):
    """Nine current×history pairs with shared bilinear + type adapters."""

    def __init__(self, d_model: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.type_embed = nn.Embedding(9, d_model)
        self.out = nn.Sequential(
            nn.Linear(rank + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, streams: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return fused relation vector and 3x3 score grid.

        streams keys: T_t,A_t,V_t,T_h,A_h,V_h each [B, D]
        """
        batch = streams["T_t"].shape[0]
        device = streams["T_t"].device
        pair_feats: list[Tensor] = []
        scores: list[Tensor] = []
        pair_id = 0
        for i, cur_m in enumerate(MODALITIES):
            for j, hist_m in enumerate(MODALITIES):
                cur = streams[f"{cur_m}_t"]
                hist = streams[f"{hist_m}_h"]
                q = self.query(cur)
                k = self.key(hist)
                score = (q * k).sum(dim=-1, keepdim=True) / (q.shape[-1] ** 0.5)
                bilinear = q * k
                typ = self.type_embed(
                    torch.full((batch,), pair_id, device=device, dtype=torch.long)
                )
                pair_feats.append(self.out(torch.cat([bilinear, typ], dim=-1)))
                scores.append(score)
                pair_id += 1
        stacked = torch.stack(pair_feats, dim=1)  # [B, 9, D]
        grid = torch.cat(scores, dim=-1).view(batch, 3, 3)
        fused = stacked.mean(dim=1)
        return fused, grid
