"""Training losses for N3 (emotion primary + utility + VAD aux)."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import N3TrainConfig
from .counterfactual import counterfactual_alignment_loss
from .redundancy import private_orthogonality_loss, sign_consistency_loss
from .causal_contribution import text_shortcut_penalty


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
    # ---- v5 core innovation: verifiable counterfactual modality utility ----
    # ``cf_measured_targets`` is a detached [B,3] leave-one-modality-out signal
    # computed by the model with an explicit ablated forward pass.  It replaces
    # the v4 self-referential rank target, and the router head is trained to
    # predict it.
    cf_utility = outputs.get("counterfactual_utility")
    modal_logits = outputs.get("modal_candidate_logits")
    routed = outputs.get("candidate_logits")
    cf_loss = zero
    consistency = zero
    cf_w = float(getattr(cfg, "counterfactual_loss_weight", 0.05) or 0.0)
    con_w = float(getattr(cfg, "cross_modal_consistency_weight", 0.03) or 0.0)
    measured = outputs.get("cf_measured_targets")
    if cf_utility is not None and measured is not None and valid is not None and valid.any() and cf_w > 0:
        measured = measured.to(device=cf_utility.device, dtype=cf_utility.dtype)
        # Accept [B,3], [B,1,3] or [B,K,3]; broadcast to the head's shape.
        while measured.ndim < cf_utility.ndim:
            measured = measured.unsqueeze(1)
        measured = measured.expand_as(cf_utility)
        hist_modal_mask = outputs.get("history_modality_mask")
        if hist_modal_mask is None:
            util_valid = valid.unsqueeze(-1) & torch.ones_like(cf_utility)
        else:
            util_valid = valid.unsqueeze(-1) & (hist_modal_mask > 0)
        cf_loss = counterfactual_alignment_loss(cf_utility, measured.detach(), util_valid)
        total = total + cf_w * cf_loss
    elif (cf_utility is not None and valid is not None and valid.any() and cf_w > 0
          and outputs.get("is_training_forward", False)):
        raise RuntimeError(
            "counterfactual_loss_weight > 0 requires measured leave-one-out "
            "targets during training; self-referential fallback is disabled"
        )
    if modal_logits is not None and routed is not None and valid is not None and valid.any() and con_w > 0:
        routed_prob = F.softmax(routed.detach(), dim=-1)[:, :, None, :]
        modal_logprob = F.log_softmax(modal_logits, dim=-1)
        consistency = F.kl_div(modal_logprob, routed_prob.expand_as(modal_logprob), reduction="none").sum(-1)
        consistency = consistency[valid].mean()
        total = total + con_w * consistency
    # ---- v5 secondary innovation: redundancy-aware private/shared terms ----
    sign_loss = zero
    ortho_loss = zero
    sign_w = float(getattr(cfg, "sign_consistency_weight", 0.0) or 0.0)
    ortho_w = float(getattr(cfg, "private_orthogonality_weight", 0.0) or 0.0)
    modality_sign = outputs.get("modality_sign")
    hist_modal_mask = outputs.get("history_modality_mask")
    if modality_sign is not None and hist_modal_mask is not None and sign_w > 0:
        sign_valid = hist_modal_mask.bool() * valid.unsqueeze(-1).bool()
        sign_loss = sign_consistency_loss(modality_sign, sign_valid.float())
        total = total + sign_w * sign_loss
    private_view = outputs.get("private_view")
    shared_view = outputs.get("shared_view")
    if private_view is not None and shared_view is not None and hist_modal_mask is not None and ortho_w > 0:
        ortho_valid = hist_modal_mask.float() * valid.unsqueeze(-1).float()
        ortho_loss = private_orthogonality_loss(private_view, shared_view, ortho_valid)
        total = total + ortho_w * ortho_loss
    # ---- v6 "solve point": counter the all-reject collapse -----------------
    collapse_w = float(getattr(cfg, "risk_collapse_weight", 0.0) or 0.0)
    collapse = outputs.get("risk_collapse_penalty")
    if collapse is None:
        collapse = zero
    if collapse_w > 0:
        total = total + collapse_w * collapse.mean()
    # ---- v7: text-shortcut guard (defined but never wired in v6) -----------
    ts_loss = zero
    ts_w = float(getattr(cfg, "text_shortcut_weight", 0.0) or 0.0)
    route_w = outputs.get("modality_route_weights")
    measured_raw = outputs.get("cf_measured_targets")
    if (route_w is not None and measured_raw is not None and ts_w > 0
            and hist_modal_mask is not None and valid is not None and valid.any()):
        m_ts = measured_raw.to(device=route_w.device, dtype=route_w.dtype)
        while m_ts.ndim < route_w.ndim:
            m_ts = m_ts.unsqueeze(1)
        m_ts = m_ts.expand_as(route_w)
        ts_valid = (valid.unsqueeze(-1) & (hist_modal_mask > 0)).to(route_w.dtype)
        ts_loss = text_shortcut_penalty(route_w, m_ts.detach(), ts_valid)
        total = total + ts_w * ts_loss
    # ---- v6 core: unimodal-label supervision (CH-SIMS v2) ------------------
    unimodal_logits = outputs.get("unimodal_logits")
    unimodal_labels = outputs.get("unimodal_labels")
    unimodal_loss = zero
    unimodal_w = float(getattr(cfg, "unimodal_loss_weight", 0.0) or 0.0)
    if (unimodal_logits is not None and unimodal_labels is not None
            and hist_modal_mask is not None and unimodal_w > 0):
        # unimodal_logits [B,K,3,C]; labels [B,K,3]
        target = unimodal_labels.to(torch.long)
        mask = (hist_modal_mask > 0) & valid.unsqueeze(-1)
        if mask.any():
            flat_logits = unimodal_logits.reshape(-1, unimodal_logits.size(-1))
            flat_target = target.reshape(-1)
            flat_mask = mask.reshape(-1)
            unimodal_loss = F.cross_entropy(
                flat_logits[flat_mask], flat_target[flat_mask], weight=class_weight,
                label_smoothing=smoothing,
            )
            total = total + unimodal_w * unimodal_loss
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
        "sign_consistency_loss": sign_loss,
        "private_orthogonality_loss": ortho_loss,
        "risk_collapse_loss": collapse.mean() if isinstance(collapse, Tensor) else collapse,
        "text_shortcut_loss": ts_loss,
        "unimodal_loss": unimodal_loss,
        "cf_has_measured_target": measured is not None,
        "vad_loss": vad,
        "mix_kl": mix_kl,
        "mix_peak_pen": mix_peak_pen,
    }
