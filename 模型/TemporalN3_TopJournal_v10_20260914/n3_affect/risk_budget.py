"""Risk-collapse repair with causal-utility adaptive gating (v6, "solve point").

Why this module exists
----------------------
The audited MOSEI run showed a pathological routing behaviour:

    mosei hard-safe   history_use_rate = 0.0064   MAE 0.6077
    mosei current-only history_use_rate = 0.0000   MAE 0.5991

The hard-safe router accepted historical evidence on 0.64% of samples and was
*worse* than the anchor it claims to be a safe superset of.  In other words the
risk head collapsed to an all-reject policy, so the whole history pathway was
dead weight.  The same signature appeared on M3ED (use rate 0.80 but a fixed
threshold 0.5 that ignores how much evidence is actually available).

Root cause
----------
``CandidateRiskFallback`` compares a *learned scalar risk* against a *constant*
threshold 0.5.  Two failure modes follow:

1.  The risk head can satisfy its BCE loss by predicting "risky" everywhere when
    candidates are genuinely mostly worse than the anchor (early training), and
    once saturated it never recovers because the gradient through
    ``risk < threshold`` is zero for every rejected candidate.
2.  The decision ignores *whether a gain was actually available*.  A candidate
    that is only marginally worse than a very poor anchor should still be
    usable if it carries real information.

Repair
------
We replace the constant comparison with a **causal-utility-adaptive budget**:

*  ``utility_budget`` is a learnable, bounded scalar per task.  Rather than
   fixing the accept rate, we fix the *expected benefit*: a candidate is
   accepted when its predicted risk, discounted by its measured counterfactual
   utility, falls below a threshold that adapts to the marginal-utility
   distribution of the batch.
*  A **monotone acceptance schedule** warms the router up: early epochs allow a
   larger accept budget (escape the collapse basin), then anneal toward the
   calibrated value.  The schedule is a function of the epoch only, never of
   validation/test performance, so it cannot leak or overfit.
*  A **recovery term** in the loss penalises accept-rate collapse directly.

All quantities are computed from training statistics only.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdaptiveRiskBudget(nn.Module):
    """Turn a learned risk score into an accept decision with an adaptive budget.

    Parameters
    ----------
    init_budget:
        Initial accept budget in ``(0, 1)``.  Interpreted as the target fraction
        of candidates that may be accepted.
    """

    def __init__(self, init_budget: float = 0.35, warmup_epochs: int = 3) -> None:
        super().__init__()
        init = float(min(max(init_budget, 1e-3), 1 - 1e-3))
        # logit-space so the budget stays strictly inside (0,1)
        self.budget_logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(init)))))
        self.utility_scale = nn.Parameter(torch.tensor(0.5))
        self.warmup_epochs = int(warmup_epochs)

    def budget(self, epoch: int | None = None) -> Tensor:
        base = torch.sigmoid(self.budget_logit)
        if epoch is None or self.warmup_epochs <= 0:
            return base
        # Warm-up: start generous, anneal to the learned budget.  This is a
        # schedule over the epoch index only -- it never observes dev/test.
        progress = min(max(epoch / float(self.warmup_epochs), 0.0), 1.0)
        warm = base + (1.0 - base) * (1.0 - progress) * 0.5
        return warm.clamp(1e-3, 0.95)

    def forward(
        self,
        risk: Tensor,
        cf_utility: Tensor | None,
        valid: Tensor,
        epoch: int | None = None,
    ) -> dict[str, Tensor]:
        """Return acceptance decisions and the applied threshold.

        ``risk`` and ``cf_utility`` are ``[B, K]``; ``valid`` is ``[B, K]``.
        Utility is *subtracted* from risk: a candidate with strong measured
        marginal contribution is treated as safer than its raw risk implies.

        v7 correctness fix: the threshold is computed **per sample** over that
        sample's own valid candidates.  The previous implementation took a
        quantile over the *flattened batch*, which made every acceptance
        decision depend on which other samples happened to share the batch --
        test predictions were not reproducible across batchings.  All tensor
        ops below are sample-local; the same sample now yields the same
        decision in any batch composition.
        """
        valid_bool = valid.bool()
        budget = self.budget(epoch)
        # v7.1 fix: the discount must be *bounded relative to the risk itself*.
        # The additive form ``risk - scale*util`` with scale clamped to 2.0 could
        # subtract more than the entire risk, so `adjusted` collapsed to 0 and
        # the gate accepted 99.57% of candidates while nominally budgeting 35%
        # (measured on the seed-43 CH-SIMS v2 run).  A multiplicative discount
        # says "high utility may at most halve the assessed risk", which keeps
        # the gate binding no matter how confident the utility head becomes.
        scale = self.utility_scale.clamp(0.0, 1.0)
        adjusted = risk
        if cf_utility is not None:
            util = cf_utility.to(device=risk.device, dtype=risk.dtype).clamp(0.0, 1.0)
            adjusted = risk * (1.0 - scale * util)
        adjusted = adjusted.masked_fill(~valid_bool, 1.0)

        # Per-sample quantile threshold.  Invalid candidates are parked at 1.0,
        # so after sorting the valid ones occupy the leading positions and the
        # quantile index only ever points at valid candidates.
        sorted_risk, _ = adjusted.sort(dim=1)                       # [B, K] ascending
        n_valid = valid_bool.sum(dim=1, keepdim=True).clamp_min(1)  # [B, 1]
        # Fractional index of the `budget` quantile within [0, n_valid - 1].
        pos = (budget.to(adjusted.dtype) * (n_valid.to(adjusted.dtype) - 1.0)).clamp_min(0.0)
        k_max = adjusted.size(1) - 1
        lo = pos.floor().long().clamp(max=k_max)
        hi = pos.ceil().long().clamp(max=k_max)
        frac = (pos - lo.to(adjusted.dtype))
        thr_lo = sorted_risk.gather(1, lo)
        thr_hi = sorted_risk.gather(1, hi)
        thr_sample = thr_lo * (1.0 - frac) + thr_hi * frac          # [B, 1]
        # Blend with the absolute budget so a degenerate candidate pool cannot
        # swing the threshold arbitrarily, same intent as the legacy blend.
        threshold = 0.5 * (thr_sample + budget.to(adjusted.dtype))  # [B, 1]
        accept = (adjusted <= threshold) & valid_bool
        return {
            "accept": accept,
            "adjusted_risk": adjusted,
            "accept_threshold": threshold.squeeze(1).detach(),
            "accept_budget": budget.detach(),
        }


def risk_collapse_penalty(accept_rate: Tensor, target_floor: float = 0.08) -> Tensor:
    """Penalise an all-reject policy without forcing a specific accept rate.

    Only the *lower* bound is constrained: a router that accepts nothing is
    certainly broken, whereas accepting more than the floor is judged by the
    primary task loss.
    """
    return F.relu(float(target_floor) - accept_rate).square()
