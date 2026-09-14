"""Verifiable counterfactual modality utility (v5 core innovation).

Motivation
----------
The v4 package exposes ``counterfactual_utility`` from the router, but the key
was never returned by the model (``model.py``), so the auxiliary loss silently
received ``None`` and stayed at zero.  Even when wired up, the old target was a
rank-normalised negative NLL of the *same* modality, which is a self-consistency
signal rather than the marginal contribution of a modality.

v5 replaces it with a *measured* counterfactual: we take the current-context
representation, remove one modality by re-running the shared task head on a
zeroed stream, and measure the increase in the task loss.  This is a genuine
leave-one-modality-out (LOMO) quantity and it is what the router's utility head
is trained to predict.

Design constraints
------------------
1.  **No gradient double counting.**  The counterfactual forward pass is
    executed under ``torch.no_grad`` so the shared head only receives gradient
    from the primary task loss; the utility head receives gradient only from the
    alignment loss.
2.  **Cheap.**  The leave-one-out batch is a single [3B, ...] forward through the
    already-built context encoder, not a re-encoding of raw features.
3.  **Safe under missing modalities.**  Removing an already-missing modality
    yields a zero target, so the utility head is never asked to explain the
    effect of a stream that was never there.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


def _binary_kl(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    """Symmetric Bernoulli KL between two logit tensors, reduced to a scalar."""
    p = torch.sigmoid(logits_a)
    q = torch.sigmoid(logits_b)
    eps = 1e-6
    kl_pq = p * ((p.clamp_min(eps).log() - q.clamp_min(eps).log())) + \
        (1 - p) * (((1 - p).clamp_min(eps)).log() - ((1 - q).clamp_min(eps)).log())
    return kl_pq.mean()


def classification_counterfactual_targets(
    score_fn: Callable[[int], Tensor],
    labels: Tensor,
    modality_mask: Tensor,
    *,
    causal: bool = True,
) -> Tensor:
    """Leave-one-modality-out CE increase for a classifier.

    ``score_fn(k)`` must run the *same* scoring head with modality ``k`` zeroed
    and return ``[B, C]`` logits.  The returned tensor is ``[B, 3]`` and is
    detached; it is a training target only.
    """
    with torch.no_grad():
        full = score_fn(-1)
        base = torch.nn.functional.cross_entropy(full, labels, reduction="none")
        targets = []
        for k in range(3):
            if float(modality_mask[:, k].sum()) == 0:
                targets.append(torch.zeros_like(base))
                continue
            removed = torch.nn.functional.cross_entropy(score_fn(k), labels, reduction="none")
            delta = removed - base
            # "causal" clamps negative deltas to zero: keeping a modality can
            # never be credited for a loss reduction that did not happen.
            if causal:
                delta = delta.clamp_min(0.0)
            visible = modality_mask[:, k]
            delta = delta * visible
            targets.append(delta)
        stacked = torch.stack(targets, dim=-1)
        # Normalise per sample so the magnitude is comparable across batches;
        # the router head learns a ranking, not an absolute loss scale.
        scale = stacked.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return stacked / scale


def regression_counterfactual_targets(
    score_fn: Callable[[int], Tensor],
    targets_norm: Tensor,
    modality_mask: Tensor,
    *,
    causal: bool = True,
) -> Tensor:
    """Leave-one-modality-out absolute-error increase for a scalar regressor."""
    with torch.no_grad():
        if targets_norm.ndim == 1:
            targets_norm = targets_norm.unsqueeze(-1)
        full = score_fn(-1)
        base = (full - targets_norm).abs()
        collected = []
        for k in range(3):
            if float(modality_mask[:, k].sum()) == 0:
                collected.append(torch.zeros_like(base))
                continue
            removed = (score_fn(k) - targets_norm).abs()
            delta = removed - base
            if causal:
                delta = delta.clamp_min(0.0)
            collected.append(delta * modality_mask[:, k])
        stacked = torch.cat(collected, dim=-1)
        scale = stacked.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return stacked / scale


class CounterfactualUtilityAligner(nn.Module):
    """Small head that predicts the measured leave-one-out utility of a stream.

    The head consumes the same evidence representation the router already
    builds, so it adds parameters without adding a second fusion backbone.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, evidence: Tensor) -> Tensor:
        """``evidence`` is ``[B, K, 3, D]`` -> ``[B, K, 3]`` utility estimates."""
        return self.net(evidence).squeeze(-1)


def counterfactual_alignment_loss(
    predicted: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    margin_rank_weight: float = 0.5,
) -> Tensor:
    """Align predicted utility with the measured target, plus a ranking term.

    ``valid`` is ``[B, K, 3]``.  The smooth-L1 term anchors the scale; the
    margin term enforces that the stream with the largest measured
    contribution also receives the largest predicted utility.
    """
    if not valid.any():
        return predicted.new_zeros(())
    smooth = torch.nn.functional.smooth_l1_loss(predicted[valid], target[valid])
    if margin_rank_weight <= 0:
        return smooth
    # Pairwise ranking inside each (batch, candidate) triple.
    pred = predicted.clone()
    tgt = target.clone()
    pred = pred.masked_fill(~valid.bool(), -1e4)
    tgt = tgt.masked_fill(~valid.bool(), -1e4)
    best = tgt.argmax(dim=-1)
    best_pred = pred.gather(-1, best.unsqueeze(-1))
    rank = torch.nn.functional.relu(0.1 + pred - best_pred)
    rank = rank.masked_fill(~valid.bool(), 0.0)
    return smooth + margin_rank_weight * rank.sum(dim=-1).mean()
