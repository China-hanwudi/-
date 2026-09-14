"""Candidate-level risk filtering with a label-free hard current-only fallback."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .risk_budget import AdaptiveRiskBudget, risk_collapse_penalty


class CandidateRiskFallback(nn.Module):
    """Predict candidate risk, retain safe candidates, and hard-switch if unsafe."""

    def __init__(self, d_model: int, feature_dim: int = 4, hidden: int = 128,
                 risk_threshold: float = 0.5,
                 use_adaptive_budget: bool = True,
                 accept_budget_init: float = 0.35,
                 accept_warmup_epochs: int = 3) -> None:
        super().__init__()
        self.risk_threshold = float(risk_threshold)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model + feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # v6 "solve point": replace the constant 0.5 comparison, which was shown
        # to collapse into an all-reject policy on CMU-MOSEI, with an adaptive
        # accept budget.  v7 fix: these settings now arrive through the config
        # instead of being hard-coded, so changing the config actually changes
        # the behaviour (previously `use_adaptive_budget` was ignored).
        self.use_adaptive_budget = bool(use_adaptive_budget)
        self.budget = AdaptiveRiskBudget(
            init_budget=accept_budget_init,
            warmup_epochs=accept_warmup_epochs,
        )

    def forward(
        self,
        candidate_repr: Tensor,
        candidate_features: Tensor,
        current_logits: Tensor,
        history_logits: Tensor,
        history_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if candidate_repr.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("candidate inputs must have shape [batch, candidates, dim]")
        risk_logits = self.risk_head(
            torch.cat((candidate_repr, candidate_features), dim=-1)
        ).squeeze(-1)
        risk = torch.sigmoid(risk_logits)
        if history_mask is not None:
            risk = risk.masked_fill(history_mask <= 0, 1.0)
        safe = risk < self.risk_threshold
        safe_count = safe.sum(dim=1)
        use_history = safe_count > 0
        filtered_risk = risk.masked_fill(~safe, 0.0)
        global_risk = filtered_risk.amax(dim=1)
        # Hard decision: no interpolation between current-only and history logits.
        logits = torch.where(use_history[:, None], history_logits, current_logits)
        return {
            "logits": logits,
            "candidate_risk_logits": risk_logits,
            "candidate_risk": risk,
            "safe_candidates": safe,
            "safe_count": safe_count,
            "global_risk": global_risk,
            "use_history": use_history,
        }

    def filter_candidates(
        self,
        candidate_repr: Tensor,
        candidate_features: Tensor,
        current_logits: Tensor,
        candidate_logits: Tensor,
        history_mask: Tensor | None = None,
        uncertainty_coef: float = 0.20,
        cf_utility: Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, Tensor]:
        """Filter candidate logits and make an exact current-only decision.

        ``candidate_logits`` is ``[B, K, C]``. Safe candidates are combined
        only after the discrete risk decision; when none remain the returned
        logits are bit-for-bit ``current_logits``.
        """
        if candidate_repr.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("candidate inputs must have shape [B, K, D/F]")
        if candidate_logits.ndim != 3:
            raise ValueError("candidate_logits must have shape [B, K, C]")
        risk_logits = self.risk_head(
            torch.cat((candidate_repr, candidate_features), dim=-1)
        ).squeeze(-1)
        risk = torch.sigmoid(risk_logits)
        # A conservative uncertainty-aware upper bound prevents a candidate
        # near the learned threshold from being accepted merely due to an
        # over-confident risk head.  The coefficient is fixed across seeds and
        # datasets; it is not tuned on the test split.
        uncertainty = torch.sqrt((risk * (1.0 - risk)).clamp_min(1e-6))
        # The coefficient is pinned by the experiment configuration.  Keeping
        # it explicit makes regression calibration auditable while preserving
        # the legacy 0.20 behaviour for the classification model.
        uncertainty_coef = float(uncertainty_coef)
        if uncertainty_coef < 0:
            raise ValueError("uncertainty_coef must be non-negative")
        risk_upper = (risk + uncertainty_coef * uncertainty).clamp(max=1.0)
        if history_mask is None:
            valid = torch.ones_like(risk, dtype=torch.bool)
        else:
            valid = history_mask.to(dtype=torch.bool)
            if valid.ndim == 1:
                valid = valid.unsqueeze(1)
        risk = risk.masked_fill(~valid, 1.0)
        risk_upper = risk_upper.masked_fill(~valid, 1.0)
        if self.use_adaptive_budget:
            # v6: the accept decision is driven by an adaptive budget that is
            # discounted by the measured counterfactual utility, instead of a
            # fixed constant.  This is the repair for the observed all-reject
            # collapse on CMU-MOSEI (history use rate 0.0064).
            decision = self.budget(risk_upper, cf_utility, valid, epoch=epoch)
            safe = decision["accept"] & valid
        else:
            safe = (risk_upper < self.risk_threshold) & valid
            decision = {"accept_threshold": risk_upper.new_tensor(self.risk_threshold),
                        "accept_budget": risk_upper.new_tensor(1.0)}
        safe_count = safe.sum(dim=1)
        use_history = safe_count > 0
        weights = torch.where(safe, (1.0 - risk_upper).clamp_min(0.0), torch.zeros_like(risk_upper))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        history_logits = (candidate_logits * weights.unsqueeze(-1)).sum(dim=1)
        logits = torch.where(use_history.unsqueeze(-1), history_logits, current_logits)
        accept_rate = (safe.float().sum() / valid.float().sum().clamp_min(1.0))
        return {
            "logits": logits,
            "candidate_risk_logits": risk_logits,
            "candidate_risk": risk,
            "candidate_risk_upper": risk_upper,
            "safe_candidates": safe,
            "safe_count": safe_count,
            "global_risk": risk_upper.masked_fill(~valid, 0.0).amax(dim=1),
            "use_history": use_history,
            "candidate_weights": weights,
            "history_logits": history_logits,
            "accept_threshold": decision["accept_threshold"],
            "accept_budget": decision["accept_budget"],
            "accept_rate": accept_rate,
            "risk_collapse_penalty": risk_collapse_penalty(accept_rate),
        }
