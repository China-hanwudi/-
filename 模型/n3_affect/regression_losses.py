"""L1 sentiment objectives and relative-MAE candidate risk supervision."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .regression_config import N3RegressionConfig


def n3_regression_loss(outputs: dict, targets: Tensor, cfg: N3RegressionConfig) -> dict[str, Tensor]:
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if targets.shape != outputs["prediction_norm"].shape or not targets.is_floating_point():
        raise ValueError("Expected continuous floating targets [B] or [B,1]")
    if not torch.isfinite(targets).all() or torch.any(targets.abs() > cfg.target_scale + 1e-6):
        raise ValueError("MOSEI targets must be finite continuous scores in [-3,3]")
    normalized = targets / cfg.target_scale
    primary = F.l1_loss(outputs["prediction_norm"], normalized)
    zero = primary.new_zeros(())
    current_aux = history_aux = risk = coverage = safe_coverage = risk_positive_rate = zero
    total = primary
    valid = outputs["history_mask"].bool()
    candidates = outputs["candidate_prediction_norm"].squeeze(-1)
    if outputs["route_mode"] != "current-only":
        current_aux = F.l1_loss(outputs["current_prediction_norm"], normalized)
        total = total + cfg.current_aux_loss_weight * current_aux
        if valid.any():
            expected = normalized.expand_as(candidates)
            history_aux = (candidates - expected).abs()[valid].mean()
            total = total + cfg.history_aux_loss_weight * history_aux
    risk_targets = torch.zeros_like(candidates)
    if outputs["route_mode"] == "hard-safe" and valid.any():
        with torch.no_grad():
            current_error = (outputs["current_prediction_norm"] - normalized).abs()
            candidate_error = (candidates - normalized).abs()
            risk_targets = (candidate_error > current_error + cfg.risk_margin).to(candidates.dtype)
        risk_logits = outputs["candidate_risk_logits"]
        risk = F.binary_cross_entropy_with_logits(risk_logits[valid], risk_targets[valid])
        safe_coverage = (1 - torch.sigmoid(risk_logits[valid])).mean()
        target_safe = (1 - risk_targets[valid]).mean()
        coverage = (safe_coverage - target_safe).square() + 0.10 * F.relu(0.10 - safe_coverage).square()
        risk_positive_rate = risk_targets[valid].mean()
        total = total + cfg.risk_loss_weight * risk + cfg.coverage_loss_weight * coverage
    mix_kl = mix_peak = zero
    mix = outputs.get("mix_weights")
    if mix is not None and cfg.mix_kl_weight:
        mix_kl = (mix * (mix.clamp_min(1e-8).log() + math.log(mix.size(-1)))).sum(dim=-1).mean()
        total = total + cfg.mix_kl_weight * mix_kl
    if mix is not None and cfg.mix_peak_weight:
        mix_peak = F.relu(mix.max(dim=-1).values - cfg.mix_peak_cap).mean()
        total = total + cfg.mix_peak_weight * mix_peak
    return {"loss": total, "main_l1_normalized": primary, "mae_original_scale": primary * cfg.target_scale,
            "current_aux_loss": current_aux, "history_aux_loss": history_aux, "risk_loss": risk,
            "coverage_loss": coverage, "risk_positive_rate": risk_positive_rate, "predicted_safe_coverage": safe_coverage,
            "risk_targets": risk_targets.detach(), "mix_kl": mix_kl, "mix_peak_pen": mix_peak}
