"""Full Temporal N3 v4 architecture with a self-contained feature interface.

The model consumes an already-frozen candidate manifest. It deliberately does
not load a previous-round checkpoint; a production runner must instantiate a
fresh model and optimizer for every authorized resampling round.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .config import TemporalN3Config
from .encoders import TemporalN3Encoders
from .modules import BatchedCandidateThreeByThree, CandidateTwoLevelGate, UtilityRiskBottleneck


class CandidateBidirectionalUtility(nn.Module):
    """Per-candidate add-benefit and deletion-risk evidence heads."""

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(d_model * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2), nn.Tanh())
            for name in "TAV"
        })
        self.joint = nn.Sequential(nn.Linear(d_model + 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2), nn.Tanh())

    def forward(self, streams: Mapping[str, Tensor], relation: Tensor) -> dict[str, Tensor]:
        scores: dict[str, Tensor] = {}
        for name in "TAV":
            current = streams[f"{name}_t"][:, None, :].expand_as(streams[f"{name}_h"])
            scores[f"U_{name}"] = self.heads[name](torch.cat([current, streams[f"{name}_h"], relation], dim=-1))
        scores["U_joint"] = self.joint(torch.cat([relation, scores["U_T"] + scores["U_A"] + scores["U_V"]], dim=-1))
        scores["U_mix"] = scores["U_joint"][..., :1] - scores["U_joint"][..., 1:2]
        return scores


class TemporalResamplingN3(nn.Module):
    """Variable-candidate N3 with a hard current-only output route."""

    def __init__(self, cfg: TemporalN3Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TemporalN3Config()
        self.cfg.validate()
        d_model = self.cfg.d_model
        self.encoders = TemporalN3Encoders(self.cfg)
        self.current_fuse = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        # Keep a separate current-only adapter/classifier so the fallback does
        # not reuse the history branch's fusion parameters. A production run
        # must train and freeze this route as its independent A0 checkpoint.
        self.current_only_fuse = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.relation = BatchedCandidateThreeByThree(d_model, self.cfg.relation_rank, self.cfg.dropout)
        self.utility = CandidateBidirectionalUtility(d_model, self.cfg.gate_hidden, self.cfg.dropout)
        self.bottleneck = UtilityRiskBottleneck(d_model, self.cfg.gate_hidden, self.cfg.dropout)
        self.gate = CandidateTwoLevelGate(d_model, self.cfg.gate_hidden, self.cfg.dropout)
        self.history_classifier = nn.Linear(d_model, self.cfg.num_classes)
        self.current_only_classifier = nn.Linear(d_model, self.cfg.num_classes)

    def forward(self, batch: Mapping[str, Tensor], temporal_features: Tensor, history_authorized: Tensor | None = None) -> dict[str, Tensor]:
        streams = self.encoders(batch)
        history_mask = batch["history_mask"].to(streams["T_t"].dtype)
        if history_mask.ndim != 2:
            raise ValueError("history_mask must be [B,K]")
        current_input = torch.cat([streams["T_t"], streams["A_t"], streams["V_t"]], dim=-1)
        current = self.current_fuse(current_input)
        current_only = self.current_only_fuse(current_input)
        relation, grid = self.relation(streams, history_mask, batch.get("modality_mask"), batch.get("history_slot_modality_mask"))
        utilities = self.utility(streams, relation)
        bottleneck = self.bottleneck(relation, utilities, temporal_features)
        gated = self.gate(current, {f"{name}_h": streams[f"{name}_h"] for name in "TAV"}, utilities, bottleneck, history_mask)
        current_logits = self.current_only_classifier(current_only)
        history_logits = self.history_classifier(gated["representation"])
        if history_authorized is None:
            # Safety default: no external authorization means current-only.
            history_authorized = torch.zeros(current.size(0), 1, device=current.device, dtype=torch.bool)
        history_authorized = history_authorized.to(dtype=torch.bool).view(-1, 1)
        if history_authorized.size(0) != current.size(0):
            raise ValueError("history_authorized must have one value per batch item")
        # An authorized history route still cannot consume an empty candidate
        # set. This makes the fallback exact for missing/invalid history.
        has_history = history_mask.any(dim=1, keepdim=True)
        effective_authorization = history_authorized & has_history
        logits = torch.where(effective_authorization, history_logits, current_logits)
        return {"logits": logits, "current_only_logits": current_logits, "history_logits": history_logits, "history_used": effective_authorization, "relation_grid": grid, "utility_risk_state": bottleneck["state"], "utility_risk": bottleneck["risk"], **utilities, **gated}

    @staticmethod
    def fresh_round_required(previous_checkpoint: str | None, previous_optimizer_state: object | None) -> None:
        if previous_checkpoint is not None or previous_optimizer_state is not None:
            raise RuntimeError("resampling rounds require fresh parameters and optimizer state")
