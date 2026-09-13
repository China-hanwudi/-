"""Shared low-rank 3x3 current-history evidence grid.

Each pair keeps a scalar alignment score for auditability, while the learned
pair representation jointly encodes alignment, complementarity and conflict.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODALITIES = ("T", "A", "V")


class SharedThreeByThree(nn.Module):
    """Encode an explicit K x 3 x 3 current/history relation grid."""

    def __init__(self, d_model: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.type_embed = nn.Embedding(9, d_model)
        self.pair_norm = nn.LayerNorm(rank * 3)
        self.out = nn.Sequential(
            nn.Linear(rank * 3 + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.pair_score = nn.Parameter(torch.zeros(9))
        self.last_evidence: dict[str, Tensor] = {}

    def forward(
        self,
        streams: dict[str, Tensor],
        has_history: Tensor | None = None,
        modality_mask: Tensor | None = None,
        history_modality_mask: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return candidate representations, relation scores and feature grid.

        History streams are ``[B, K, D]``.  A two-dimensional history tensor is
        accepted as a single candidate for small callers and smoke tests.
        """
        batch = streams["T_t"].shape[0]
        device = streams["T_t"].device
        history = streams["T_h"]
        if history.ndim == 2:
            history = history.unsqueeze(1)
            for name in MODALITIES:
                streams[f"{name}_h"] = streams[f"{name}_h"].unsqueeze(1)
        candidates = history.shape[1]
        if history_mask is None:
            history_mask = torch.ones(batch, candidates, device=device)
        else:
            history_mask = history_mask.to(device=device, dtype=torch.float32)
            if history_mask.ndim == 1:
                history_mask = history_mask.unsqueeze(1)
        if modality_mask is None:
            modality_mask = torch.ones(batch, 3, device=device, dtype=torch.float32)
        else:
            modality_mask = modality_mask.to(device=device, dtype=torch.float32)
        if history_modality_mask is None:
            history_modality_mask = history_mask.unsqueeze(-1).expand(batch, candidates, 3)
        elif history_modality_mask.ndim == 2:
                history_modality_mask = history_modality_mask.unsqueeze(1).expand(batch, candidates, 3)
        history_modality_mask = history_modality_mask.to(device=device, dtype=torch.float32)
        speaker_same = streams.get("speaker_same")
        if speaker_same is None:
            speaker_same = torch.zeros(batch, candidates, device=device)
        elif speaker_same.ndim == 1:
            speaker_same = speaker_same.unsqueeze(1)
        speaker_same = speaker_same.to(device=device, dtype=torch.float32)
        pair_mask = (
            modality_mask[:, None, :, None]
            * history_modality_mask[:, :, None, :]
            * history_mask[:, :, None, None]
        )
        pair_feats: list[Tensor] = []
        scores: list[Tensor] = []
        complements: list[Tensor] = []
        conflicts: list[Tensor] = []
        pair_id = 0
        for i, cur_m in enumerate(MODALITIES):
            for j, hist_m in enumerate(MODALITIES):
                cur = streams[f"{cur_m}_t"]
                hist = streams[f"{hist_m}_h"]
                q = self.query(cur).unsqueeze(1)
                k = self.key(hist)
                qn = F.normalize(q, dim=-1)
                kn = F.normalize(k, dim=-1)
                # Alignment, complementarity and conflict are retained as
                # separate channels before being fused by the pair MLP.
                alignment = (qn * kn).sum(dim=-1, keepdim=True)
                complement = (q - k).abs().mean(dim=-1, keepdim=True)
                conflict = (1.0 - alignment).clamp_min(0.0)
                score = alignment
                pair_signal = torch.cat([q * k, (q - k).abs(), q + k], dim=-1)
                pair_signal = self.pair_norm(pair_signal)
                typ = self.type_embed(
                    torch.full((batch, candidates), pair_id, device=device, dtype=torch.long)
                )
                feat = self.out(torch.cat([pair_signal, typ], dim=-1))
                # Make complementarity/conflict visible to downstream risk
                # routing without adding a second selector network.
                feat = feat + (complement - conflict)
                feat = feat * torch.sigmoid(self.pair_score[pair_id])
                feat = feat * pair_mask[:, :, i, j].unsqueeze(-1)
                pair_feats.append(feat)
                scores.append(score * pair_mask[:, :, i, j].unsqueeze(-1))
                complements.append(complement * pair_mask[:, :, i, j].unsqueeze(-1))
                conflicts.append(conflict * pair_mask[:, :, i, j].unsqueeze(-1))
                pair_id += 1
        stacked = torch.stack(pair_feats, dim=2).view(batch, candidates, 9, -1)
        grid = torch.cat(scores, dim=-1).view(batch, candidates, 3, 3)
        complement_grid = torch.cat(complements, dim=-1).view(batch, candidates, 3, 3)
        conflict_grid = torch.cat(conflicts, dim=-1).view(batch, candidates, 3, 3)
        valid_count = pair_mask.sum(dim=(2, 3), keepdim=False).clamp(min=1.0)
        fused = stacked.sum(dim=2) / valid_count.unsqueeze(-1)
        # Speaker-conditioned state continuity, parameter-free to preserve
        # compatibility with existing checkpoints. Same-speaker evidence is
        # retained slightly more strongly; cross-speaker history is unchanged.
        fused = fused * (1.0 + 0.10 * speaker_same.unsqueeze(-1))
        fused = fused * history_mask.unsqueeze(-1)
        self.last_evidence = {
            "alignment": grid,
            "complementarity": complement_grid,
            "conflict": conflict_grid,
        }
        return fused, grid, stacked.view(batch, candidates, 3, 3, -1)
