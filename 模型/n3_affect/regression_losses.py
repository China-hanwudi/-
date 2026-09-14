"""L1 sentiment objectives and relative-MAE candidate risk supervision."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .regression_config import N3RegressionConfig
from .counterfactual import counterfactual_alignment_loss
from .redundancy import private_orthogonality_loss, sign_consistency_loss
from .causal_contribution import unimodal_aux_loss, text_shortcut_penalty


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
    # Risk supervision covers ALL candidates (current-as-candidate + history).
    risk_targets = torch.zeros_like(outputs["candidate_risk_logits"])
    all_cands = outputs.get("all_candidate_prediction_norm")
    all_mask = outputs.get("all_candidate_mask")
    if outputs["route_mode"] == "hard-safe" and all_mask is not None and all_mask.any():
        valid_all = all_mask.bool()
        with torch.no_grad():
            current_error = (outputs["current_prediction_norm"] - normalized).abs()
            all_err = (all_cands.squeeze(-1) - normalized).abs()
            risk_targets = (all_err > current_error + cfg.risk_margin).to(risk_targets.dtype)
        risk_logits = outputs["candidate_risk_logits"]
        risk = F.binary_cross_entropy_with_logits(risk_logits[valid_all], risk_targets[valid_all])
        safe_coverage = (1 - torch.sigmoid(risk_logits[valid_all])).mean()
        target_safe = (1 - risk_targets[valid_all]).mean()
        coverage = (safe_coverage - target_safe).square() + 0.10 * F.relu(0.10 - safe_coverage).square()
        risk_positive_rate = risk_targets[valid_all].mean()
        total = total + cfg.risk_loss_weight * risk + cfg.coverage_loss_weight * coverage
    mix_kl = mix_peak = zero
    mix = outputs.get("mix_weights")
    if mix is not None and cfg.mix_kl_weight:
        mix_kl = (mix * (mix.clamp_min(1e-8).log() + math.log(mix.size(-1)))).sum(dim=-1).mean()
        total = total + cfg.mix_kl_weight * mix_kl
    if mix is not None and cfg.mix_peak_weight:
        mix_peak = F.relu(mix.max(dim=-1).values - cfg.mix_peak_cap).mean()
        total = total + cfg.mix_peak_weight * mix_peak
    # ---- v5 core innovation: verifiable counterfactual modality utility ----
    cf_loss = sign_loss = ortho_loss = zero
    cf_utility = outputs.get("cf_predicted_utility")
    measured = outputs.get("cf_measured_targets")
    cf_w = float(getattr(cfg, "counterfactual_loss_weight", 0.0) or 0.0)
    if cf_utility is not None and measured is not None and valid.any() and cf_w > 0:
        measured = measured.to(device=cf_utility.device, dtype=cf_utility.dtype)
        # Accept [B,3], [B,1,3] or [B,K,3]; broadcast to the head's shape.
        while measured.ndim < cf_utility.ndim:
            measured = measured.unsqueeze(1)
        measured = measured.expand_as(cf_utility)
        hist_modal = outputs.get("history_modality_mask")
        util_valid = valid.unsqueeze(-1) if hist_modal is None else (valid.unsqueeze(-1) & (hist_modal > 0))
        cf_loss = counterfactual_alignment_loss(cf_utility, measured.detach(), util_valid)
        total = total + cf_w * cf_loss
    elif cf_utility is not None and valid.any() and cf_w > 0 and outputs.get("is_training_forward", False):
        raise RuntimeError(
            "counterfactual_loss_weight > 0 requires measured leave-one-out "
            "targets during training; missing target would be self-referential"
        )
    sign_w = float(getattr(cfg, "sign_consistency_weight", 0.0) or 0.0)
    ortho_w = float(getattr(cfg, "private_orthogonality_weight", 0.0) or 0.0)
    modality_sign = outputs.get("modality_sign")
    hist_modal = outputs.get("history_modality_mask")
    if modality_sign is not None and hist_modal is not None and sign_w > 0:
        sign_valid = (hist_modal > 0).float() * valid.unsqueeze(-1).float()
        sign_loss = sign_consistency_loss(modality_sign, sign_valid)
        total = total + sign_w * sign_loss
    private_view = outputs.get("private_view")
    shared_view = outputs.get("shared_view")
    if private_view is not None and shared_view is not None and hist_modal is not None and ortho_w > 0:
        ortho_valid = (hist_modal > 0).float() * valid.unsqueeze(-1).float()
        ortho_loss = private_orthogonality_loss(private_view, shared_view, ortho_valid)
        total = total + ortho_w * ortho_loss
    # ---- v6 "solve point": counter the MOSEI all-reject collapse ----------
    collapse = outputs.get("risk_collapse_penalty")
    if collapse is None:
        collapse = zero
    collapse_w = float(getattr(cfg, "risk_collapse_weight", 0.0) or 0.0)
    if collapse_w > 0:
        total = total + collapse_w * collapse.mean()
    # ---- v7: current-candidate mechanism (alive on non-dialogue data) ------
    modality_mask = outputs.get("modality_mask")
    # NOTE: the history-candidate block above reassigns ``measured`` to an
    # expanded view, so every block below re-reads the raw target.
    measured_raw = outputs.get("cf_measured_targets")
    # The current candidate's utility head is trained against the *same*
    # measured leave-one-modality-out target as the history-candidate head.
    cur_cf_loss = zero
    cur_cf = outputs.get("current_cf_predicted_utility")
    if cur_cf is not None and measured_raw is not None and modality_mask is not None and cf_w > 0:
        m_cur = measured_raw.to(device=cur_cf.device, dtype=cur_cf.dtype)
        while m_cur.ndim < cur_cf.ndim:
            m_cur = m_cur.unsqueeze(1)
        m_cur = m_cur.expand_as(cur_cf)
        cur_valid = (modality_mask.unsqueeze(1).expand_as(cur_cf) > 0)
        cur_cf_loss = counterfactual_alignment_loss(cur_cf, m_cur.detach(), cur_valid)
        total = total + cf_w * cur_cf_loss
    # Unimodal labels (CH-SIMS v2 label_T/A/V) supervise the per-modality head
    # as an ordinary auxiliary loss -- never inside a loss difference.
    uni_loss = zero
    uni_w = float(getattr(cfg, "unimodal_loss_weight", 0.0) or 0.0)
    uni_pred = outputs.get("unimodal_prediction_norm")
    uni_labels = outputs.get("unimodal_labels")
    if (uni_pred is not None and uni_labels is not None
            and modality_mask is not None and uni_w > 0):
        uni_labels = uni_labels.to(device=uni_pred.device, dtype=uni_pred.dtype)
        uni_loss = unimodal_aux_loss(
            uni_pred, uni_labels, modality_mask,
            classification=False, target_scale=cfg.target_scale,
        )
        total = total + uni_w * uni_loss
    # Text-shortcut guard: text's share of the routed decision weight may not
    # exceed its measured causal gain by more than the margin.
    ts_loss = zero
    ts_w = float(getattr(cfg, "text_shortcut_weight", 0.0) or 0.0)
    cur_w = outputs.get("current_modality_weights")
    if cur_w is not None and measured_raw is not None and modality_mask is not None and ts_w > 0:
        m_ts = measured_raw.to(device=cur_w.device, dtype=cur_w.dtype)
        while m_ts.ndim < cur_w.ndim:
            m_ts = m_ts.unsqueeze(1)
        m_ts = m_ts.expand_as(cur_w)
        ts_valid = modality_mask.unsqueeze(1).expand_as(cur_w)
        ts_loss = text_shortcut_penalty(cur_w, m_ts.detach(), ts_valid)
        total = total + ts_w * ts_loss
    return {"loss": total, "main_l1_normalized": primary, "mae_original_scale": primary * cfg.target_scale,
            "current_aux_loss": current_aux, "history_aux_loss": history_aux, "risk_loss": risk,
            "coverage_loss": coverage, "risk_positive_rate": risk_positive_rate, "predicted_safe_coverage": safe_coverage,
            "risk_targets": risk_targets.detach(), "mix_kl": mix_kl, "mix_peak_pen": mix_peak,
            "counterfactual_loss": cf_loss, "sign_consistency_loss": sign_loss,
            "private_orthogonality_loss": ortho_loss,
            "risk_collapse_loss": collapse.mean() if isinstance(collapse, Tensor) else collapse,
            "current_cf_loss": cur_cf_loss, "unimodal_loss": uni_loss,
            "text_shortcut_loss": ts_loss,
            "cf_has_measured_target": measured is not None}
