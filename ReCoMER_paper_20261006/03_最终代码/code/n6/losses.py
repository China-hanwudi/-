"""Objective for model6 (iteration 3).

Core terms (unchanged from round 1/2):
1. Task loss on the DEPLOYED output:
   cls -> cross entropy (label_smoothing=0.05, sqrt-inverse class weights
   ``w_c = sqrt(N / (C * count_c))`` capped at 2.0, computed from the train
   split labels only); reg -> MSE.  Under M3 the deployed forward is the
   mixed one and the task loss becomes the mixup task loss
   (``lam * CE(y_i) + (1 - lam) * CE(y_j)``; MSE against the mixed target).
2. ``lambda_u * MSE(mu, phi_measured)`` -- the measured exact-Shapley
   alignment of the utility MU head; exactly 0 in ``uniform``/``gate`` mode.
   ``mu`` is taken from the CLEAN forward when M3 is active (the Shapley
   target is always measured on the clean batch).
3. ``lambda_c * KL(log_softmax(full_joint) || log_softmax(deployed))`` with
   batch-mean reduction for cls (bugfix 3, 2026-09-21: previously summed over
   the batch, so the strength scaled with batch size; for reg:
   ``lambda_c * mean((joint - deployed)^2)``).
   Skipped entirely under ``--deploy joint_uniform|joint_softgate``.

Iteration-3 module terms (only when the corresponding flag is on):
4. M1: ``lambda_j * loss(full joint) + lambda_v * loss(view joint)`` -- the
   counterfactual-view training of the joint head (CE/MSE matching the task).
5. M3: ``0.1 * MSE(g_phi(mixed), lam * g_phi(i) + (1 - lam) * g_phi(j))`` --
   utility-consistency of the mixup.
6. W6 (only with ``--grl-lambda > 0``): ``lambda * CE(series_logits,
   series_id)`` -- the series-adversarial term (GRL inside the model).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import M6Config

# Non-full modality subsets used by the M1 counterfactual views.
NONFULL_SUBSETS = (
    ("T",), ("A",), ("V",),
    ("A", "T"), ("A", "V"), ("T", "V"),
)


def compute_class_weights(labels: Tensor, num_classes: int, cap: float = 2.0) -> Tensor:
    """``w_c = sqrt(N / (C * count_c))`` capped at ``cap`` (train labels only).

    A01/P0-02: out-of-range labels raise instead of being silently clamped
    (framework 10: 类别越界先报错不能clamp)."""
    y = labels.long()
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError(
            "labels out of range for num_classes=%d" % num_classes)
    counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
    n = float(labels.numel())
    w = torch.sqrt(torch.tensor(n, dtype=torch.float32) / (num_classes * counts))
    return w.clamp(max=cap)


def joint_supervision(logits: Tensor, y: Tensor, cfg: M6Config,
                      class_weight: Optional[Tensor] = None) -> Tensor:
    """Task-matching supervision for a joint-head output (M1 terms)."""
    if cfg.task == "cls":
        return F.cross_entropy(logits, y, weight=class_weight,
                               label_smoothing=cfg.label_smoothing)
    return F.mse_loss(logits.squeeze(-1), y)


def m1_view_loss(model, cur_embs: Dict[str, Tensor],
                 batch: Dict[str, Tensor], y: Tensor, cfg: M6Config,
                 class_weight: Optional[Tensor] = None) -> Tensor:
    """M1 view term: per-sample random non-full subset through the joint head.

    Each sample draws one of the 6 non-full subsets; samples sharing a draw
    are scored in one grouped ``joint_on_subset`` forward (up to 6 small
    forwards per step).  Returns the sample-weighted mean view loss.
    """
    bsz = y.shape[0]
    pick = torch.randint(0, len(NONFULL_SUBSETS), (bsz,), device=y.device)
    acc = y.new_zeros((), dtype=torch.float32)
    count = 0
    for k, subset in enumerate(NONFULL_SUBSETS):
        idx = (pick == k).nonzero(as_tuple=True)[0]
        nb = int(idx.numel())
        if nb == 0:
            continue
        sub_b = {key: val[idx] for key, val in batch.items()}
        sub_cur = {m: cur_embs[m][idx] for m in cur_embs}
        logits = model.joint_on_subset(sub_cur, sub_b, subset)
        acc = acc + joint_supervision(logits, y[idx], cfg, class_weight) * nb
        count += nb
    return acc / max(count, 1)


def total_loss(out: Dict[str, Tensor], y: Tensor, cfg: M6Config,
               class_weight: Optional[Tensor] = None,
               phi: Optional[Tensor] = None,
               mu: Optional[Tensor] = None,
               mix: Optional[Tuple[Tensor, Tensor]] = None,
               aux: Optional[Dict[str, Tensor]] = None
               ) -> Dict[str, Tensor]:
    """Assemble the total objective.

    ``out`` is the forward whose deployed output carries the task loss (the
    MIXED forward under M3).  ``mu`` overrides the utility-MSE source (the
    clean-forward mu under M3); it defaults to ``out["utility_score"]``.
    ``aux`` carries precomputed module terms: ``m1_full``, ``m1_view`` and
    ``m3_consistency`` (the latter already an MSE scalar).
    """
    deployed = out["deployed"]
    joint = out["joint"]
    if mix is not None:
        perm, lam = mix
        if cfg.task == "cls":
            ce_i = F.cross_entropy(deployed, y, weight=class_weight,
                                   label_smoothing=cfg.label_smoothing,
                                   reduction="none")
            ce_j = F.cross_entropy(deployed, y[perm], weight=class_weight,
                                   label_smoothing=cfg.label_smoothing,
                                   reduction="none")
            task = (lam * ce_i + (1.0 - lam) * ce_j).mean()
        else:
            task = F.mse_loss(deployed.squeeze(-1),
                              lam * y + (1.0 - lam) * y[perm])
    elif cfg.task == "cls":
        task = F.cross_entropy(deployed, y, weight=class_weight,
                               label_smoothing=cfg.label_smoothing)
    else:
        task = F.mse_loss(deployed.squeeze(-1), y)

    mu_src = mu if mu is not None else out.get("utility_score")
    loss_u = deployed.new_zeros(())
    if (cfg.utility == "shapley" and phi is not None and mu_src is not None):
        # phi is the per-sample standardised Shapley vector [B,3].
        loss_u = F.mse_loss(mu_src, phi)

    if cfg.deploy in ("joint_uniform", "joint_softgate", "closed_loop"):
        # The deployed path IS the joint head there, so "KL(full joint ||
        # deployed)" degenerates to zero and is skipped by design.
        loss_c = deployed.new_zeros(())
    elif cfg.task == "cls":
        logp_deployed = F.log_softmax(deployed, dim=-1)
        p_joint = F.softmax(joint, dim=-1)
        # Bugfix 3 (audit 2026-09-21): batchmean, not sum -- a summed KL made
        # the regularization strength scale with batch size.  Only affects
        # solo_weighted configs with the KL term active.
        loss_c = F.kl_div(logp_deployed, p_joint, reduction="batchmean")
    else:
        loss_c = ((joint.squeeze(-1) - deployed.squeeze(-1)) ** 2).mean()

    m1_full = deployed.new_zeros(())
    m1_view = deployed.new_zeros(())
    m3_cons = deployed.new_zeros(())
    series = deployed.new_zeros(())
    if aux is not None:
        if "m1_full" in aux:
            m1_full = aux["m1_full"]
        if "m1_view" in aux:
            m1_view = aux["m1_view"]
        if "m3_consistency" in aux:
            m3_cons = aux["m3_consistency"]
        if "series" in aux:
            series = aux["series"]

    total = (task + cfg.lambda_u * loss_u + cfg.lambda_c * loss_c
             + cfg.lambda_j * m1_full + cfg.lambda_v * m1_view
             + 0.1 * m3_cons + cfg.grl_lambda * series)
    return {"loss": total, "task": task, "utility": loss_u,
            "consistency": loss_c, "m1_full": m1_full, "m1_view": m1_view,
            "m3_consistency": m3_cons, "series": series}
