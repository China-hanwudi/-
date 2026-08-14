"""Shared low-rank 3x3 current-history relation grid with masking."""
from __future__ import annotations

import torch
from torch import Tensor, nn


MODALITIES = ("T", "A", "V")


class SharedThreeByThree(nn.Module):
    """Nine current×history pairs with shared bilinear + type adapters.

    Pair (i, j) is valid only when the current modality i is present, the
    pooled history modality j is present, and the sample has at least one
    real previous turn. Time-slot history_mask is applied before pooling
    and must not be reused as the 3 history-modality columns.
    """

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

    def forward(
        self,
        streams: dict[str, Tensor],
        has_history: Tensor | None = None,
        modality_mask: Tensor | None = None,
        history_modality_mask: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return fused relation vector and 3x3 score grid.

        streams keys: T_t,A_t,V_t,T_h,A_h,V_h each [B, D] (history already pooled)
        has_history: [B] binary
        modality_mask: [B, 3] current T/A/V
        history_modality_mask: [B, 3] pooled history T/A/V
        history_mask: ignored if has_history is provided (kept for call-site compat)
        """
        batch = streams["T_t"].shape[0]
        device = streams["T_t"].device
        if has_history is None:
            if history_mask is not None:
                has_history = (history_mask.sum(dim=1) > 0).to(dtype=torch.float32)
            else:
                has_history = torch.ones(batch, device=device, dtype=torch.float32)
        else:
            has_history = has_history.to(dtype=torch.float32).view(batch)
        if modality_mask is None:
            modality_mask = torch.ones(batch, 3, device=device, dtype=torch.float32)
        if history_modality_mask is None:
            history_modality_mask = has_history.unsqueeze(1).expand(batch, 3)
        pair_mask = (
            modality_mask.unsqueeze(2)
            * history_modality_mask.unsqueeze(1)
            * has_history.view(batch, 1, 1)
        )
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
                feat = self.out(torch.cat([bilinear, typ], dim=-1))
                feat = feat * pair_mask[:, i, j].unsqueeze(-1)
                pair_feats.append(feat)
                scores.append(score * pair_mask[:, i, j].unsqueeze(-1))
                pair_id += 1
        stacked = torch.stack(pair_feats, dim=1)
        grid = torch.cat(scores, dim=-1).view(batch, 3, 3)
        valid_count = pair_mask.view(batch, 9).sum(dim=1, keepdim=True).clamp(min=1.0)
        fused = stacked.sum(dim=1) / valid_count
        return fused, grid
