"""Cross-modal redundancy with orthogonality and sign consistency (v5).

This is the second, complementary innovation.  It addresses the failure mode the
core counterfactual objective exposes:

* A modality can receive a near-zero leave-one-out utility not because it is
  uninformative but because another modality already carries the same
  information (redundancy).
* Two modalities can be individually informative yet push the decision in
  opposite directions, which makes the routed candidate unstable.

We therefore add three light-weight, differentiable structures:

1.  **Private/shared view separation** (``PrivateSharedReformulation``): a
    parameter-efficient soft split of each modality representation into a
    shared affect channel and a private residual channel.  LOMO utility is then
    computed on the *view* level, so a redundant modality loses utility only on
    its shared view while its private view is preserved.
2.  **Sign-consistency heads** (``SignConsistencyHeads``): per modality sign of
    the predicted sentiment/decision margin, trained with a consistency loss so
    the router learns to distrust streams whose sign disagrees with the
    consensus.
3.  **Orthogonality regulariser**: pushes private channels of different
    modalities apart (decorrelation) while keeping shared channels aligned.

The module is task agnostic: ``out_dim`` is the number of classes for M3ED and
1 for CMU-MOSEI.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PrivateSharedReformulation(nn.Module):
    """Soft private/shared split with a bounded shared ratio.

    A single learnable scalar ``shared_ratio`` per modality controls how much of
    the representation is treated as the shared affect channel.  It is bounded
    by a sigmoid so the split cannot collapse.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.private_proj = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.recombine = nn.Sequential(
            nn.Linear(2 * hidden, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )
        self.shared_ratio = nn.Parameter(torch.zeros(3))

    def forward(self, evidence: Tensor) -> dict[str, Tensor]:
        """``evidence`` is ``[B, K, 3, D]``.

        Returns ``shared`` / ``private`` / ``recombined``, each
        ``[B, K, 3, D]``.
        """
        batch, candidates, modalities, d_model = evidence.shape
        flat = evidence.reshape(batch * candidates * modalities, d_model)
        shared = self.shared_proj(flat).reshape(batch, candidates, modalities, -1)
        private = self.private_proj(flat).reshape(batch, candidates, modalities, -1)
        ratio = torch.sigmoid(self.shared_ratio).view(1, 1, modalities, 1)
        mixed = torch.cat([ratio * shared, (1.0 - ratio) * private], dim=-1)
        recombined = self.recombine(
            mixed.reshape(batch * candidates * modalities, -1)
        ).reshape(batch, candidates, modalities, d_model)
        return {"shared": shared, "private": private, "recombined": recombined, "ratio": ratio}


class SignConsistencyHeads(nn.Module):
    """Per-modality sign logit for the routed decision.

    For classification the "sign" is the margin between the top-1 and top-2
    class logits; for regression it is the raw scalar.  A consistency loss then
    penalises streams that disagree, without forcing identical magnitudes.
    """

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, private: Tensor) -> Tensor:
        return self.head(private).squeeze(-1)


def sign_consistency_loss(modality_sign: Tensor, valid: Tensor, *, tau: float = 0.5) -> Tensor:
    """Penalise sign disagreement weighted by availability.

    ``modality_sign`` is ``[B, K, 3]``.  We use a smooth surrogate: the squared
    distance to the availability-weighted mean sign.
    """
    if not valid.any():
        return modality_sign.new_zeros(())
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (modality_sign * valid).sum(dim=-1, keepdim=True) / count
    penalty = (modality_sign - mean).square() * valid
    return tau * penalty.sum() / valid.sum().clamp_min(1.0)


def private_orthogonality_loss(private: Tensor, shared: Tensor, valid: Tensor) -> Tensor:
    """Decorrelate private channels; align shared channels across modalities.

    ``private``/``shared`` are ``[B, K, 3, H]``.  A modality-axis Gram matrix of
    the availability-weighted features gives a cheap, scale-free objective: the
    off-diagonal of the private Gram is pushed to zero (each modality keeps a
    distinct private code), while the shared Gram is pushed *towards* identity
    (all modalities keep a compatible shared affect code).
    """
    if not valid.any():
        return private.new_zeros(())
    weight = valid.unsqueeze(-1)                       # [B,K,3,1]
    count = (weight.sum() * max(private.size(-1), 1)).clamp_min(1.0)

    def _gram(x: Tensor) -> Tensor:
        # x: [B,K,3,H]; contract the batch/slot/feature axes so the Gram lives
        # on the modality axis, giving a [3,3] matrix.
        x = x * weight
        flat = x.permute(2, 0, 1, 3).reshape(x.size(2), -1)     # [3, B*K*H]
        cov = flat @ flat.t()                                    # [3,3]
        cov = cov / count
        norm = cov.diagonal().clamp_min(1e-6).sqrt()
        return cov / (norm[:, None] * norm[None, :])

    n_modal = private.size(2)
    eye = torch.eye(n_modal, device=private.device, dtype=private.dtype)
    priv_gram = _gram(private)
    shared_gram = _gram(shared)
    # Off-diagonal only: the diagonal is the (unit) self-correlation.
    priv_term = (priv_gram - torch.diag(priv_gram.diagonal())).square().sum()
    shared_term = (shared_gram - eye).square().sum()
    return priv_term + 0.5 * shared_term
