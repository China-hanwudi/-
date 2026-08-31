"""Candidate-wise Temporal N3 modules; all routing inputs are OOF/development-only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


class BatchedCandidateThreeByThree(nn.Module):
    """Shared low-rank, masked `B x K x 3 x 3` current-history relation grid."""

    def __init__(self, d_model: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.type_embed = nn.Parameter(torch.empty(3, 3, d_model))
        nn.init.normal_(self.type_embed, std=0.02)
        self.out = nn.Sequential(nn.Linear(rank + d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))

    def forward(self, streams: Mapping[str, Tensor], history_mask: Tensor, modality_mask: Tensor | None = None, history_slot_modality_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        current = torch.stack([streams[f"{name}_t"] for name in "TAV"], dim=1)
        history = torch.stack([streams[f"{name}_h"] for name in "TAV"], dim=2)
        batch, candidates, _, _ = history.shape
        pair = self.query(current)[:, None, :, None, :] * self.key(history)[:, :, None, :, :]
        scores = pair.sum(-1) / (pair.size(-1) ** 0.5)
        if modality_mask is None:
            modality_mask = torch.ones(batch, 3, device=pair.device, dtype=pair.dtype)
        if history_slot_modality_mask is None:
            history_slot_modality_mask = history_mask[..., None].expand(-1, -1, 3)
        valid = history_mask[:, :, None, None] * modality_mask[:, None, :, None] * history_slot_modality_mask[:, :, None, :]
        typed = self.type_embed[None, None].expand(batch, candidates, -1, -1, -1)
        relation = self.out(torch.cat([pair, typed], dim=-1)) * valid[..., None]
        return relation.sum(dim=(2, 3)) / valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0).squeeze(-1), scores * valid


class UtilityRiskBottleneck(nn.Module):
    """Compress relation, bidirectional utility/risk and temporal evidence."""

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model + 10, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.LayerNorm(d_model))
        self.risk_logit = nn.Linear(d_model, 1)

    def forward(self, candidate_repr: Tensor, utilities: Mapping[str, Tensor], temporal_features: Tensor) -> dict[str, Tensor]:
        if temporal_features.shape[:-1] != candidate_repr.shape[:-1] or temporal_features.size(-1) != 2:
            raise ValueError("temporal_features must be [B,K,2]")
        evidence = torch.cat([utilities["U_T"], utilities["U_A"], utilities["U_V"], utilities["U_joint"], temporal_features], dim=-1)
        state = self.net(torch.cat([candidate_repr, evidence], dim=-1))
        return {"state": state, "risk": torch.sigmoid(self.risk_logit(state)), "evidence": evidence}


class CandidateTwoLevelGate(nn.Module):
    """Modality gate -> Utility-Risk Bottleneck -> candidate gate."""

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.modality = nn.Sequential(nn.Linear(d_model + 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.candidate = nn.Sequential(nn.Linear(d_model * 2 + 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, current_pool: Tensor, history: Mapping[str, Tensor], utilities: Mapping[str, Tensor], bottleneck: Mapping[str, Tensor], history_mask: Tensor) -> dict[str, Tensor]:
        kept, probabilities = [], []
        for name in "TAV":
            probability = torch.sigmoid(self.modality(torch.cat([history[f"{name}_h"], utilities[f"U_{name}"]], dim=-1)))
            kept.append(history[f"{name}_h"] * probability)
            probabilities.append(probability)
        candidate_repr = torch.stack(kept, dim=2).sum(dim=2)
        candidate_input = torch.cat([candidate_repr, bottleneck["state"], bottleneck["risk"], utilities["U_mix"]], dim=-1)
        candidate_prob = torch.sigmoid(self.candidate(candidate_input)) * history_mask[..., None]
        weighted = (candidate_repr * candidate_prob).sum(dim=1) / candidate_prob.sum(dim=1).clamp_min(1e-6)
        return {"representation": current_pool + weighted, "modality_keep_prob": torch.cat(probabilities, dim=-1), "candidate_keep_prob": candidate_prob, "candidate_repr": candidate_repr}


@dataclass(frozen=True)
class RoundDecision:
    action: str
    round_index: int


def authorize_next_round(round_index: int, final_gate_passed: bool, macro_gain_pp: float, weighted_gain_pp: float, nll_improved: bool, harm_reduction_pp: float, improving_seed_fraction: float) -> RoundDecision:
    """Authorize no more than two resampling fallbacks under frozen joint gates."""
    if final_gate_passed:
        return RoundDecision("accept_history", round_index)
    if round_index >= 3:
        return RoundDecision("fallback_current_only", round_index)
    allowed = macro_gain_pp >= 0.8 and weighted_gain_pp >= 0.5 and nll_improved and harm_reduction_pp >= 2.0 and improving_seed_fraction >= 0.8
    return RoundDecision("authorize_resample" if allowed else "fallback_current_only", round_index)
