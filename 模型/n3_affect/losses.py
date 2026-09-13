"""Training losses for N3 (emotion primary + utility + VAD aux)."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import N3TrainConfig


def n3_total_loss(
    outputs: Mapping[str, Tensor],
    labels: Tensor,
    cfg: N3TrainConfig,
    utility_targets: Mapping[str, Tensor] | None = None,
    vad_targets: Tensor | None = None,
    class_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    emotion = F.cross_entropy(outputs["logits"], labels, weight=class_weight, label_smoothing=float(getattr(cfg, "label_smoothing", 0.0)))
    total = cfg.emotion_loss_weight * emotion
    zero = outputs["logits"].new_zeros(())
    smoothing = float(getattr(cfg, "label_smoothing", 0.0))
    route_mode = outputs.get("route_mode", "hard-safe")

    current_aux = zero
    if route_mode != "current-only" and "current_only_logits" in outputs and cfg.current_aux_loss_weight > 0:
        current_aux = F.cross_entropy(
            outputs["current_only_logits"], labels, weight=class_weight,
            label_smoothing=smoothing,
        )
        total = total + cfg.current_aux_loss_weight * current_aux

    history_aux = zero
    valid = outputs.get("history_mask")
    candidate_logits = outputs.get("candidate_logits")
    if valid is not None:
        valid = valid.to(dtype=torch.bool)
    if route_mode != "current-only" and candidate_logits is not None and valid is not None and valid.any():
        candidate_labels = labels[:, None].expand_as(valid)
        history_aux = F.cross_entropy(
            candidate_logits[valid], candidate_labels[valid], weight=class_weight,
            label_smoothing=smoothing,
        )
        total = total + cfg.history_aux_loss_weight * history_aux

    risk = zero
    coverage = zero
    risk_positive_rate = zero
    predicted_safe_coverage = zero
    risk_logits = outputs.get("candidate_risk_logits")
    if route_mode == "hard-safe" and risk_logits is not None and candidate_logits is not None and valid is not None and valid.any():
        oof_targets = outputs.get("oof_risk_targets")
        if oof_targets is not None:
            # Cross-fitted labels are preferred when supplied by a frozen
            # train-only OOF artifact; they never contain validation/test data.
            risk_targets = oof_targets.detach().to(device=risk_logits.device, dtype=risk_logits.dtype)
        else:
            with torch.no_grad():
                current_nll = -F.log_softmax(outputs["current_only_logits"], dim=-1).gather(
                    1, labels[:, None]
                ).squeeze(1)
                candidate_nll = -F.log_softmax(candidate_logits, dim=-1).gather(
                    2, labels[:, None, None].expand(-1, candidate_logits.size(1), 1)
                ).squeeze(2)
                risk_targets = (candidate_nll > current_nll[:, None] + cfg.risk_margin).to(risk_logits.dtype)
        risk = F.binary_cross_entropy_with_logits(risk_logits[valid], risk_targets[valid])
        predicted_safe_coverage = (1.0 - torch.sigmoid(risk_logits[valid])).mean()
        target_safe_coverage = (1.0 - risk_targets[valid]).mean()
        coverage = (predicted_safe_coverage - target_safe_coverage).square()
        # Discourage the degenerate all-reject policy.  The floor is only a
        # training regularizer; final acceptance still follows the calibrated
        # uncertainty-aware hard threshold.
        coverage = coverage + 0.10 * F.relu(0.10 - predicted_safe_coverage).square()
        risk_positive_rate = risk_targets[valid].mean()
        total = total + cfg.risk_loss_weight * risk + cfg.coverage_loss_weight * coverage
    utility = torch.zeros((), device=labels.device)
    if utility_targets is not None:
        pieces = []
        for key in ("U_T", "U_A", "U_V", "U_joint"):
            if key in utility_targets:
                pieces.append(F.mse_loss(outputs[key], utility_targets[key]))
        if "U_cross" in utility_targets:
            pieces.append(F.mse_loss(outputs["U_cross"], utility_targets["U_cross"]))
        if pieces:
            utility = torch.stack(pieces).mean()
            total = total + cfg.utility_loss_weight * utility
    # Reliability objectives for the v4 router.  These are optional and remain
    # zero for legacy checkpoints, but make the new innovation trainable:
    # (i) counterfactual utility is aligned with the observed per-modality
    # prediction quality; (ii) valid modalities should agree with the routed
    # decision without forcing identical representations.
    cf_utility = outputs.get("counterfactual_utility")
    modal_logits = outputs.get("modal_candidate_logits")
    routed = outputs.get("candidate_logits")
    cf_loss = zero
    consistency = zero
    cf_w = float(getattr(cfg, "counterfactual_loss_weight", 0.05) or 0.0)
    con_w = float(getattr(cfg, "cross_modal_consistency_weight", 0.03) or 0.0)
    if cf_utility is not None and modal_logits is not None and valid is not None and valid.any() and cf_w > 0:
        with torch.no_grad():
            modal_nll = -F.log_softmax(modal_logits, dim=-1).gather(
                -1, labels[:, None, None, None].expand(-1, modal_logits.size(1), modal_logits.size(2), 1)
            ).squeeze(-1)
            target = (-modal_nll).detach()
            target = (target - target.mean(dim=2, keepdim=True)) / target.std(dim=2, keepdim=True).clamp_min(1e-4)
        pred = cf_utility
        pred = (pred - pred.mean(dim=2, keepdim=True)) / pred.std(dim=2, keepdim=True).clamp_min(1e-4)
        cf_loss = F.smooth_l1_loss(pred[valid], target[valid])
        total = total + cf_w * cf_loss
    if modal_logits is not None and routed is not None and valid is not None and valid.any() and con_w > 0:
        routed_prob = F.softmax(routed.detach(), dim=-1)[:, :, None, :]
        modal_logprob = F.log_softmax(modal_logits, dim=-1)
        consistency = F.kl_div(modal_logprob, routed_prob.expand_as(modal_logprob), reduction="none").sum(-1)
        consistency = consistency[valid].mean()
        total = total + con_w * consistency
    vad = torch.zeros((), device=labels.device)
    if vad_targets is not None:
        vad = F.mse_loss(outputs["vad"], vad_targets)
        total = total + cfg.vad_loss_weight * vad
    mix = outputs.get("mix_weights")
    mix_kl = torch.zeros((), device=labels.device)
    mix_peak_pen = torch.zeros((), device=labels.device)
    kl_w = float(getattr(cfg, "mix_kl_weight", 0.0) or 0.0)
    if mix is not None and kl_w > 0:
        k = mix.size(-1)
        mix_kl = (mix * (mix.clamp_min(1e-8).log() + math.log(k))).sum(dim=-1).mean()
        total = total + kl_w * mix_kl
    peak_w = float(getattr(cfg, "mix_peak_weight", 0.0) or 0.0)
    peak_cap = float(getattr(cfg, "mix_peak_cap", 0.40) or 0.40)
    if mix is not None and peak_w > 0:
        mix_peak_pen = F.relu(mix.max(dim=-1).values - peak_cap).mean()
        total = total + peak_w * mix_peak_pen
    return {
        "loss": total,
        "emotion_loss": emotion,
        "current_aux_loss": current_aux,
        "history_aux_loss": history_aux,
        "risk_loss": risk,
        "coverage_loss": coverage,
        "risk_positive_rate": risk_positive_rate,
        "predicted_safe_coverage": predicted_safe_coverage,
        "utility_loss": utility,
        "counterfactual_loss": cf_loss,
        "cross_modal_consistency": consistency,
        "vad_loss": vad,
        "mix_kl": mix_kl,
        "mix_peak_pen": mix_peak_pen,
    }
