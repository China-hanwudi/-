"""Unimodal-supervised counterfactual modality contribution (v7, corrected).

Motivation
----------
CH-SIMS v2 is unusual in that *every utterance carries three aligned labels*:
one for text, one for audio, one for video, plus the multimodal label.  Almost
all prior work either ignores the unimodal labels or uses them only as extra
classification targets.  That throws away the strongest signal in the dataset.

v6 design flaw (fixed here)
---------------------------
The v6 ``causal_utility_from_labels`` computed

    causal_k = L(f(x_{-k}), y_k) - L(f(x), y)

i.e. it subtracted losses taken against **two different label spaces** (the
modality's own annotation ``y_k`` and the fused annotation ``y``).  The
difference conflates "the effect of removing modality k" with "the difference
between two independent annotations", so it is not a well-defined causal
quantity.  Reviewers with any causal-inference background reject this on sight.

v7 correction
-------------
The two quantities are separated cleanly:

* **Counterfactual marginal utility (same label).**  The causal gain of a
  modality is measured against the *fused* label only:

      causal_k = L(f(x_{-k}), y) - L(f(x), y)

  This is exactly the leave-one-modality-out quantity already measured in
  ``counterfactual.py`` / ``model.measure_counterfactual_utility``.  It stays
  the training target of the router's utility head.

* **Unimodal labels as auxiliary supervision.**  ``label_T/A/V`` supervise a
  per-modality head (this file's ``UnimodalContributionHead``) via an ordinary
  auxiliary loss.  They never enter a loss *difference*, so no cross-label
  subtraction exists anywhere in the objective.  The head's predictions also
  serve as pseudo unimodal labels for datasets without them (e.g. MOSEI).

Text-only shortcut guard
------------------------
A well-known failure on CH-SIMS/MOSEI is the model collapsing onto text.  The
``text_shortcut_penalty`` penalises the router when text's share of the
decision weight exceeds its *measured* causal gain by more than a margin.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class UnimodalContributionHead(nn.Module):
    """Predict each modality's own label from its private view.

    Because CH-SIMS v2 labels each modality separately, this head is supervised
    directly.  ``out_dim`` is the number of classes for M3ED and 1 (a
    continuous score) for CH-SIMS v2 / CMU-MOSEI.  Its outputs double as the
    ``y_m`` estimates when explicit unimodal labels are unavailable.
    """

    def __init__(self, d_model: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, modality_view: Tensor) -> Tensor:
        """``modality_view`` is ``[..., D]`` -> ``[..., out_dim]``."""
        return self.net(modality_view)


def causal_utility_same_label(
    ablation_error: Tensor,
    full_error: Tensor,
    visible: Tensor,
) -> Tensor:
    """Same-label counterfactual marginal utility.

    ``ablation_error`` ``[B, 3]`` is the task loss of the model with each
    modality removed, measured **against the fused label**; ``full_error``
    ``[B]`` is the same loss with all modalities present; ``visible`` is the
    ``[B, 3]`` modality availability mask.

    Returns the detached ``[B, 3]`` per-sample max-normalised target

        causal_k = max(L(f(x_{-k}), y) - L(f(x), y), 0)

    A modality whose removal increases the fused-label loss the most is the
    most causally important one.  Negative deltas are clamped: keeping a
    modality is never credited for a loss reduction that did not happen.
    """
    with torch.no_grad():
        if full_error.ndim > 1:
            full_error = full_error.squeeze(-1)
        causal = (ablation_error - full_error.unsqueeze(-1)).clamp_min(0.0)
        causal = causal * visible.to(dtype=causal.dtype)
        scale = causal.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return (causal / scale).detach()


def unimodal_aux_loss(
    unimodal_pred: Tensor,
    unimodal_labels: Tensor,
    valid: Tensor,
    *,
    classification: bool,
    class_weight: Tensor | None = None,
    target_scale: float = 1.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Auxiliary supervision of the per-modality head against ``label_T/A/V``.

    This is an *ordinary* auxiliary loss -- the unimodal labels only ever
    appear here, as direct targets, and never inside a loss difference.

    ``unimodal_pred`` is ``[..., 3]`` (regression, squeezed) or ``[..., 3, C]``
    (classification); ``unimodal_labels`` is ``[..., 3]``; ``valid`` is the
    ``[..., 3]`` availability mask.
    """
    if not valid.any():
        return unimodal_pred.new_zeros(())
    valid_f = valid.to(dtype=unimodal_pred.dtype)
    if classification:
        logits = unimodal_pred.reshape(-1, unimodal_pred.size(-1))
        target = unimodal_labels.reshape(-1).to(torch.long)
        keep = valid.reshape(-1).bool()
        return F.cross_entropy(
            logits[keep], target[keep],
            weight=class_weight, label_smoothing=label_smoothing,
        )
    labels = unimodal_labels.to(dtype=unimodal_pred.dtype) / float(target_scale)
    err = F.smooth_l1_loss(unimodal_pred, labels, reduction="none")
    return (err * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def text_shortcut_penalty(
    modality_weights: Tensor,
    causal_target: Tensor,
    valid: Tensor,
    *,
    margin: float = 0.35,
) -> Tensor:
    """Penalise text receiving more decision weight than its causal gain allows.

    ``modality_weights`` ``[B,K,3]``, ``causal_target`` ``[B,K,3]``.
    """
    if not valid.any():
        return modality_weights.new_zeros(())
    w = modality_weights * valid
    total = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    share = w / total
    allowed = (causal_target * valid).clamp_min(1e-6)
    allowed = allowed / allowed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    excess = F.relu(share[:, :, 0:1] - allowed[:, :, 0:1] - margin)
    masked = (excess * valid[:, :, 0:1]).sum() / valid[:, :, 0:1].sum().clamp_min(1.0)
    return masked
