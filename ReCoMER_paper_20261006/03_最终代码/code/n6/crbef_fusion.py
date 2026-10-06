"""cRBEF-style external expert fusion for the MHnoU model.

The local MHnoU checkpoint remains the primary predictor.  A separately
trained cRBEF-v2 predictor is treated as an external evidence expert and is
added in log-probability space with a bounded, sample-dependent gate::

    log p_fused = log p_mh + g(x) * (log p_crbef - log pi)

where ``pi`` is the training class prior and ``0 <= g <= max_gate``.  The
main model is therefore an explicit fallback when the external expert is
uncertain or disagrees with it.  The module accepts logits or probabilities;
the two predictors are never retrained together by this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CRBEFFusionConfig:
    num_classes: int
    max_gate: float = 1.0
    initial_gate: float = 0.5
    eps: float = 1e-6

    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("cRBEF fusion requires at least two classes")
        if self.max_gate <= 0.0:
            raise ValueError("max_gate must be positive")
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between 0 and 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def _as_log_probs(x: Tensor, eps: float) -> Tensor:
    """Convert probabilities or ordinary logits to log probabilities."""
    if x.ndim != 2:
        raise ValueError("classification predictions must have shape [N, C]")
    # A valid probability matrix has non-negative entries and row sums close
    # to one.  This keeps the adapter convenient for saved cRBEF .npz files.
    if bool(torch.isfinite(x).all()) and bool((x >= -eps).all()):
        row_sum = x.sum(dim=-1, keepdim=True)
        if bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=2e-4,
                               rtol=2e-4)):
            return x.clamp_min(eps).log()
    return F.log_softmax(x, dim=-1)


def _margin(prob: Tensor) -> Tensor:
    top = prob.topk(k=min(2, prob.shape[-1]), dim=-1).values
    return top[:, 0] - top[:, 1]


def reliability_features(main_logits: Tensor, expert_logits: Tensor,
                          class_prior: Optional[Tensor] = None,
                          eps: float = 1e-6) -> Tensor:
    """Build the six reliability statistics used by cRBEF.

    The order matches the A100 implementation: main top probability, main
    top-two margin, cRBEF top probability, strongest cRBEF log-evidence over
    the class prior, prediction agreement, and main entropy.  A sigmoid
    linear gate over these values has seven trainable parameters.
    """
    main_logits = _as_log_probs(main_logits, eps)
    expert_logits = _as_log_probs(expert_logits, eps)
    if main_logits.shape != expert_logits.shape:
        raise ValueError("main and cRBEF predictions must have equal shape")
    main = main_logits.softmax(dim=-1)
    expert = expert_logits.softmax(dim=-1)
    if class_prior is None:
        prior = torch.full((main.shape[-1],), 1.0 / main.shape[-1],
                           dtype=main.dtype, device=main.device)
    else:
        prior = torch.as_tensor(class_prior, dtype=main.dtype,
                                device=main.device).flatten()
        if prior.numel() != main.shape[-1]:
            raise ValueError("class_prior length must equal class count")
        prior = prior.clamp_min(eps)
        prior = prior / prior.sum()
    expert_evidence = expert_logits - prior.clamp_min(eps).log()
    agreement = (main.argmax(dim=-1) == expert.argmax(dim=-1)).to(main.dtype)
    main_entropy = -(main * main.clamp_min(eps).log()).sum(dim=-1)
    return torch.stack([
        main.max(dim=-1).values,
        _margin(main),
        expert.max(dim=-1).values,
        expert_evidence.max(dim=-1).values,
        agreement,
        main_entropy,
    ], dim=-1)


class CRBEFExternalFusion(nn.Module):
    """Bounded log-linear fusion of MHnoU and a cRBEF-v2 expert."""

    def __init__(self, cfg: CRBEFFusionConfig,
                 class_prior: Optional[Tensor] = None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.gate = nn.Linear(6, 1)
        # cRBEF starts from a fixed half-strength evidence blend; the seven
        # parameters are then learned while both experts remain frozen.
        with torch.no_grad():
            self.gate.weight.zero_()
            p = float(cfg.initial_gate)
            self.gate.bias.fill_(torch.logit(torch.tensor(p)))
        if class_prior is None:
            prior = torch.full((cfg.num_classes,), 1.0 / cfg.num_classes)
        else:
            prior = torch.as_tensor(class_prior, dtype=torch.float32).flatten()
            if prior.numel() != cfg.num_classes:
                raise ValueError("class_prior length must equal num_classes")
            prior = prior.clamp_min(cfg.eps)
            prior = prior / prior.sum()
        self.register_buffer("class_prior", prior, persistent=True)

    def gate_values(self, main_logits: Tensor, expert_logits: Tensor) -> Tensor:
        feat = reliability_features(main_logits, expert_logits,
                                    self.class_prior, self.cfg.eps)
        return self.cfg.max_gate * torch.sigmoid(self.gate(feat).squeeze(-1))

    def forward(self, main_logits: Tensor, expert_logits: Tensor,
                return_gate: bool = False):
        main_logits = _as_log_probs(main_logits, self.cfg.eps)
        expert_logits = _as_log_probs(expert_logits, self.cfg.eps)
        if main_logits.shape != expert_logits.shape:
            raise ValueError("main and cRBEF predictions must have equal shape")
        if main_logits.shape[-1] != self.cfg.num_classes:
            raise ValueError("prediction class count does not match config")
        gate = self.gate_values(main_logits.detach(), expert_logits.detach())
        prior_log = self.class_prior.clamp_min(self.cfg.eps).log()
        fused = main_logits + gate.unsqueeze(-1) * (expert_logits - prior_log)
        fused = F.log_softmax(fused, dim=-1)
        if return_gate:
            return fused, gate
        return fused


@torch.no_grad()
def fuse_with_fixed_gate(main_logits: Tensor, expert_logits: Tensor,
                         gate: Tensor, class_prior: Optional[Tensor] = None,
                         eps: float = 1e-6) -> Tensor:
    """Functional form useful for audits and frozen gate replay."""
    main_logits = _as_log_probs(main_logits, eps)
    expert_logits = _as_log_probs(expert_logits, eps)
    if main_logits.shape != expert_logits.shape:
        raise ValueError("main and cRBEF predictions must have equal shape")
    if gate.ndim == 1:
        gate = gate.unsqueeze(-1)
    if gate.shape != main_logits.shape[:1] + (1,):
        raise ValueError("gate must have shape [N] or [N,1]")
    if class_prior is None:
        prior = torch.full((main_logits.shape[-1],),
                           1.0 / main_logits.shape[-1],
                           dtype=main_logits.dtype, device=main_logits.device)
    else:
        prior = torch.as_tensor(class_prior, dtype=main_logits.dtype,
                                device=main_logits.device).flatten()
        prior = prior.clamp_min(eps)
        prior = prior / prior.sum()
    z = main_logits + gate.clamp_min(0.0) * (expert_logits - prior.log())
    return F.log_softmax(z, dim=-1)


def fit_gate(main_logits: Tensor, expert_logits: Tensor, labels: Tensor,
             cfg: CRBEFFusionConfig, class_prior: Optional[Tensor] = None,
             epochs: int = 100, lr: float = 1e-3,
             weight_decay: float = 1e-4) -> Tuple[CRBEFExternalFusion, Dict[str, float]]:
    """Fit only the reliability gate while both experts stay frozen."""
    if labels.ndim != 1 or labels.shape[0] != main_logits.shape[0]:
        raise ValueError("labels must be [N] and aligned with predictions")
    model = CRBEFExternalFusion(cfg, class_prior=class_prior)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    main_logits = main_logits.detach().float()
    expert_logits = expert_logits.detach().float()
    labels = labels.detach().long()
    last = float("nan")
    for _ in range(max(1, int(epochs))):
        opt.zero_grad(set_to_none=True)
        fused = model(main_logits, expert_logits)
        loss = F.nll_loss(fused, labels)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    model.eval()
    with torch.no_grad():
        _, gate = model(main_logits, expert_logits, return_gate=True)
    return model, {"train_nll": last,
                   "mean_gate": float(gate.mean()),
                   "min_gate": float(gate.min()),
                   "max_gate": float(gate.max())}


def prediction_tensor(value, *, device: str = "cpu") -> Tensor:
    """Convert a NumPy array/list to a float tensor for CLI and notebooks."""
    return torch.as_tensor(value, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# The newer ``local_current`` outer fusion from the supplied CRBEF package.
# This is deliberately kept separate from CRBEFExternalFusion: the latter is
# the seven-parameter CRBEF-internal gate, while local_current is the second
# fusion layer applied to two complete peer models.

LOCAL_CURRENT_FEATURES = 23


def _as_probabilities(x: Tensor, eps: float = 1e-7) -> Tensor:
    return _as_log_probs(x, eps).exp()


def _validate_local_inputs(P: Tensor, modality_evidence: Tensor) -> None:
    if P.ndim != 3 or P.shape[1] != 4:
        raise ValueError("P must have shape [N,4,C]: a,b,a0,b0")
    if modality_evidence.ndim != 3 or modality_evidence.shape[1] != 3:
        raise ValueError("modality_evidence must have shape [N,3,C]")
    if P.shape[0] != modality_evidence.shape[0] or P.shape[2] != modality_evidence.shape[2]:
        raise ValueError("P and modality_evidence must have matching N and C")
    if P.shape[2] < 2:
        raise ValueError("local_current requires at least two classes")


def local_current_features(P: Tensor, modality_evidence: Tensor,
                           eps: float = 1e-7) -> Tensor:
    """Build the 23 per-class features used by the supplied local_current.

    P columns are ``[a, b, a0, b0]`` where a is cRBEF, b is the complete
    MHnoU/uniform_h peer, a0 is the CRBEF reference output and b0 is the
    no-history reference output.  The final three features are the
    modality-relative evidence C for T/A/V.
    """
    P = torch.as_tensor(P, dtype=torch.float32)
    modality_evidence = torch.as_tensor(modality_evidence, dtype=torch.float32)
    _validate_local_inputs(P, modality_evidence)
    a, b, a0, b0 = (_as_probabilities(P[:, i], eps) for i in range(4))
    q = (a + b) / 2.0
    la, lb, lq, la0, lb0 = [x.clamp_min(eps).log() for x in (a, b, q, a0, b0)]
    da, db = la - la0, lb - lb0
    norm = lambda d: d / d.abs().sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = lambda p: -(p * p.clamp_min(eps).log()).sum(dim=1)
    margin = lambda p: p.topk(k=2, dim=1).values[:, 0] - p.topk(k=2, dim=1).values[:, 1]
    global_stats = torch.stack([entropy(a), entropy(b), margin(a), margin(b)], dim=1)
    flags = torch.stack([
        (a == a.max(dim=1, keepdim=True).values).to(a.dtype),
        (b == b.max(dim=1, keepdim=True).values).to(b.dtype),
    ], dim=-1)
    inner = torch.stack([a0, b0, la0, lb0, da, db, norm(da), norm(db)], dim=-1)
    base = torch.stack([a, b, q, la, lb, lq], dim=-1)
    global_stats = global_stats[:, None, :].expand(-1, P.shape[2], -1)
    evidence = modality_evidence.transpose(1, 2)
    return torch.cat([base, global_stats, flags, inner, evidence], dim=-1)


def local_current_candidate_mask(P: Tensor) -> Tensor:
    """Return the top-two/average-top candidate set used by local_current."""
    P = torch.as_tensor(P, dtype=torch.float32)
    a, b = _as_probabilities(P[:, 0]), _as_probabilities(P[:, 1])
    q = (a + b) / 2.0
    a_second = a.topk(k=2, dim=1).values[:, 1, None]
    b_second = b.topk(k=2, dim=1).values[:, 1, None]
    return ((a >= a_second) | (b >= b_second) |
            (q == q.max(dim=1, keepdim=True).values))


@dataclass
class LocalCurrentConfig:
    num_classes: int
    eta: float = 1.0
    correction_fraction: float = 0.25
    delta_scale: float = 1.5
    eps: float = 1e-7

    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("local_current requires at least two classes")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0,1]")
        if not 0.0 < self.correction_fraction <= 1.0:
            raise ValueError("correction_fraction must be in (0,1]")
        if self.delta_scale <= 0.0:
            raise ValueError("delta_scale must be positive")


class LocalCurrentFusion(nn.Module):
    """Class-wise bounded residual correction from the newer candidate.

    The network predicts one correction for each class from the 23 features,
    reallocates only the probability mass of a candidate class set, and then
    applies the selected eta/fraction.  The correction is initialized to zero,
    so an unfitted instance is exactly the equal-probability baseline.
    """

    def __init__(self, cfg: LocalCurrentConfig,
                 mean: Optional[Tensor] = None,
                 scale: Optional[Tensor] = None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.net = nn.Sequential(nn.Linear(LOCAL_CURRENT_FEATURES, 32),
                                 nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if mean is None:
            mean = torch.zeros(LOCAL_CURRENT_FEATURES)
        if scale is None:
            scale = torch.ones(LOCAL_CURRENT_FEATURES)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).flatten())
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).flatten().clamp_min(1e-5))
        if self.mean.numel() != LOCAL_CURRENT_FEATURES or self.scale.numel() != LOCAL_CURRENT_FEATURES:
            raise ValueError("local_current mean/scale must have length 23")

    def forward(self, P: Tensor, modality_evidence: Tensor,
                return_details: bool = False):
        P = torch.as_tensor(P, dtype=torch.float32)
        modality_evidence = torch.as_tensor(modality_evidence, dtype=torch.float32)
        _validate_local_inputs(P, modality_evidence)
        if P.shape[2] != self.cfg.num_classes:
            raise ValueError("P class count does not match config")
        x = local_current_features(P, modality_evidence, self.cfg.eps)
        z = ((x - self.mean) / self.scale).clamp(-10.0, 10.0)
        raw = self.net(z).squeeze(-1)
        delta = self.cfg.delta_scale * raw.tanh()
        a, b = _as_probabilities(P[:, 0]), _as_probabilities(P[:, 1])
        q = (a + b) / 2.0
        mask = local_current_candidate_mask(P).to(q.dtype)
        mass = (q * mask).sum(dim=1, keepdim=True)
        weighted = q * delta.exp() * mask
        corrected = torch.where(
            mask.bool(), mass * weighted / weighted.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps), q)
        fused = q + self.cfg.correction_fraction * float(self.cfg.eta) * (corrected - q)
        fused = fused.clamp_min(self.cfg.eps)
        fused = fused / fused.sum(dim=1, keepdim=True)
        logp = fused.log()
        if return_details:
            return logp, {"probabilities": fused, "delta": delta, "candidate_mask": mask,
                          "baseline": q, "corrected": corrected}
        return logp


def fit_local_current(P: Tensor, modality_evidence: Tensor, labels: Tensor,
                      cfg: LocalCurrentConfig, epochs: int = 40,
                      lr: float = 1e-3, weight_decay: float = 1e-2
                      ) -> Tuple[LocalCurrentFusion, Dict[str, float]]:
    """Fit only the outer local_current corrector on train/OOF predictions."""
    P = torch.as_tensor(P, dtype=torch.float32).detach()
    modality_evidence = torch.as_tensor(modality_evidence, dtype=torch.float32).detach()
    labels = torch.as_tensor(labels, dtype=torch.long).detach()
    if labels.ndim != 1 or labels.shape[0] != P.shape[0]:
        raise ValueError("labels must be [N] and aligned with P")
    x = local_current_features(P, modality_evidence, cfg.eps)
    axes = tuple(range(x.ndim - 1))
    mean = x.mean(dim=axes)
    scale = x.std(dim=axes, unbiased=False).clamp_min(1e-5)
    model = LocalCurrentFusion(cfg, mean=mean, scale=scale)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for _ in range(max(1, int(epochs))):
        opt.zero_grad(set_to_none=True)
        logp, details = model(P, modality_evidence, return_details=True)
        nll = F.nll_loss(logp, labels)
        q = details["baseline"]
        good_baseline = (q.argmax(dim=1) == labels).float()
        baseline_logp = q[torch.arange(len(labels)), labels].clamp_min(cfg.eps).log()
        fitted_logp = logp[torch.arange(len(labels)), labels]
        preserve = (good_baseline * (baseline_logp - fitted_logp).relu()).mean()
        regularizer = details["delta"].square().mean()
        loss = nll + 2.0 * preserve + 0.01 * regularizer
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        _, details = model(P, modality_evidence, return_details=True)
    return model, {"loss": history[-1], "mean_abs_delta": float(details["delta"].abs().mean()),
                   "candidate_fraction": float(details["candidate_mask"].mean())}
