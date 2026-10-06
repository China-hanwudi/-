"""UGF (Utility-Gated Fusion) model, model6.

Architecture (d_model=192, dropout=0.15); all inputs are utterance-level
pooled vectors:

1. Per-modality encoder: ``LN -> Linear(d_m, 192) -> GELU -> Dropout -> LN``
   plus a learned per-modality embedding vector added right after the linear.
   History slots (K up to 3, from ``history_index``) are encoded with the SAME
   encoder (shared weights) and masked by validity.
2. History context: per modality, single-head scaled-dot-product attention
   pooling over that modality's valid history embeddings with the current
   embedding as query (learned projection on the query); the pooled vector is
   added back via a LayerScale zero-initialised scalar.  ``use_history=False``
   (or K=0 packs) disables the whole block.
3. Three SOLO heads: ``Linear(192 -> C)`` per modality on the final embedding.
4. JOINT head: tokens = 3 current embeddings (+ history tokens when enabled)
   through a 2-layer ``nn.TransformerEncoder`` (nhead=6, dim_ff=576,
   dropout=0.15, batch_first) with subset-masked mean-pooling ->
   ``Linear(192 -> C)``.  An arbitrary modality subset S can be evaluated by
   masking the excluded modalities' tokens (their history tokens too).
5. ``UtilityHead`` g_phi: input =
   concat[3 final current embeddings (576), 3 pairwise cos-similarity scalars,
   3 solo-confidence scalars] (582) -> ``Linear(582->192) -> GELU -> Dropout ->
   Linear(192->3)``.  Solo-confidence is the max softmax prob of each detached
   solo head for cls, and ``|yhat_m - median(yhat_1, yhat_2, yhat_3)|`` of the
   detached solo predictions for reg.
6. Deployment (``--deploy``):
   * ``solo_weighted`` (default): ``w = softmax(g_phi / tau)`` with
     ``w_m = max(w_m, eps_floor)`` and renormalisation, then
     ``deployed = sum_m w_m * solo_m``.
   * ``joint_uniform``: the joint head over the (unscaled) tokens deploys
     directly -- every modality token enters uniformly.
   * ``joint_softgate``: the utility gates the fusion INPUT -- tokens
     (current and history) are scaled by the per-sample deploy weights and
     the joint head output is the deployed logits.
   * ``closed_loop``: the admitted history has already been fed back into the
     current embeddings; bounded utility weights scale the three current
     tokens and the joint head is the final predictor.  Raw history tokens
     are omitted to avoid double counting the same history.
   In the joint modes the solo-sum is NOT the deployed path, but the solo
   heads still exist (they feed the g_phi confidence features).  The KL
   consistency term is only defined for ``solo_weighted`` and is skipped in
   the joint modes.  ``utility=uniform`` fixes ``w = 1/3`` (g_phi is never
   instantiated); ``utility=gate`` instantiates g_phi without any Shapley
   supervision (lambda_u = 0); ``utility=shapley`` supervises g_phi with the
   measured exact Shapley values.

Measured utility (training-time only, ``torch.no_grad``): the 7 non-empty
subsets S of {T, A, V} are scored through the JOINT head on the current
tokens only; U(S) is computed PER SAMPLE -- ``log_softmax(logits_S)[y]`` per
row for cls, ``-|yhat_S - y|`` per row for reg, ``U(empty) = 0`` -- so the
exact 3-player Shapley values phi are [B,3].  As the g_phi MSE target each
sample's 3-vector is standardised (mean 0, std 1, std clamped at 1e-6); the
raw per-sample phi (``standardize=False``) is what the valid-only oracle
analysis deploys.  Measurement runs on the current training model state, per
training step, mirroring the reference ``measure_counterfactual_utility``
pattern.

Iteration-3 switchable modules (all default OFF, so round-1/2 arms reproduce
bit-for-bit):

* M1 counterfactual-view training: per training step, per sample, one random
  non-full subset view of the joint head; the joint head is supervised by
  ``lambda_j * loss(full joint) + lambda_v * loss(view joint)`` (CE for cls,
  MSE for reg).  The full joint is the unweighted all-token head; each view
  forward reuses ``joint_on_subset`` on the samples that drew that subset
  (up to 6 small grouped forwards -- chosen over a fused full+view batch
  because per-sample subsets do not share a token mask).
* M2 regret gating: the utility head emits (mu, sigma) per modality and the
  deploy weights become ``softmax((mu - kappa * sigma) / tau)`` (floor and
  renormalisation unchanged); the utility MSE supervises mu only.
* M3 utility-consistent mixup: per batch, sample pairs (permutation) and
  ``lambda ~ Beta(alpha, alpha)`` mix the cur_embs across pairs; the task
  loss is taken on the mixed forward and a 0.1-weighted MSE keeps
  ``g_phi(mixed)`` close to ``lambda * g_phi(i) + (1 - lambda) * g_phi(j)``
  computed on the pre-mix embeddings (detached target).  The Shapley utility
  MSE target is always measured on the clean batch.

W6 gap-fix switches (both default OFF, so all earlier arms reproduce):

* Series-adversarial head (``--grl-lambda``): the masked mean of the three
  final current embeddings is pushed through a GradientReversalLayer(lambda)
  into a small series classifier (d -> 128 -> n_series); the trainer adds
  ``lambda * CE(series_logits, series_id)`` to the objective.  Series ids are
  parsed from the pack ids (field 1 of ``<A/B>_<series>_<ep>_<utt>``) and the
  class vocabulary is the sorted unique set of the TRAIN split.  Training-only:
  the head is never executed at eval time, and old checkpoints (no head keys,
  ``grl_lambda=0``) load unchanged.

W9 StreamFusion switches (all default OFF; with the switches off or the pack
keys absent every earlier arm reproduces bit-for-bit):

* V-path (``--vpath``; needs pack key ``Vf [N,K,512]``): K frame tokens ->
  Linear(512->d) + learned frame positional embedding -> 2-layer
  TransformerEncoder -> attention pool with a learned query.  The output
  REPLACES the pooled-V current embedding; the U-gate reformulates the
  change as a gated correction ``V + g_v * (vpath_out - pooled_V)``.
* M-path (``--mpath``): SAM2-style memory attention.  The memory bank is the
  history slots of all three modalities (encoded with the shared modality
  encoders, reusing ``hist_embs`` when the history block is on), masked by
  validity x modality x speaker_same (speaker_same absent -> all readable).
  Query = masked mean of the three current embeddings; the LayerScale(0)
  read-out is added to every current embedding.  Additive on top of the
  existing history block; self-contained when the block is off.
* U-gate (``--ugate``): 242-parameter MLP on 12 cheap per-modality solo
  confidence stats (p_max / margin / entropy / agree-with-text) producing two
  sigmoid gates for the V and M corrections.  Gates are computed from the
  PRE-correction solo outputs.
* C1 blend (``--c1-alpha``): deployed = (1-a) * uniform solo-logit average +
  a * (sum_m (log_softmax(solo_m) - log pi) + log pi), pi = train class
  prior (stored in cfg / RUN_METADATA; buffer persistent=False so state
  dicts stay compatible).  cls + solo_weighted only.

Stage-2 mask fusion (``mask_fusion_fix``): with the flag on, the
``solo_weighted`` deployed sum uses ``w_present = w * modality_mask``
renormalized over present modalities, so an absent modality (zero-filled
input still yields nonzero head logits via LayerNorm beta / head bias)
contributes nothing; rows with no present modality get an explicit
uniform-posterior fallback (zero logits / 0.0) and are counted in
``out["n_mask_fallback"]`` / valid metrics ``n_mask_fallback``.  All-present
rows are bit-identical to the old path (w * 1 = w, sum(w)=1).  The flag
defaults to False in cfg so existing checkpoints keep their saved behavior.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import M6Config

MODALITIES = ("T", "A", "V")
COS_PAIRS = ((0, 1), (0, 2), (1, 2))
UTILITY_EMPTY = 0.0


class GradientReversal(torch.autograd.Function):
    """Identity forward; backward multiplies the gradient by -lambda."""

    @staticmethod
    def forward(ctx, x: Tensor, lam: float) -> Tensor:
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return -ctx.lam * grad, None


def grl(x: Tensor, lam: float) -> Tensor:
    return GradientReversal.apply(x, lam)


class VPath(nn.Module):
    """Frame-level video path (W9): K CLIP-512 frames -> temporal transformer
    -> attention pool.  Output REPLACES the pooled-V current embedding when
    --vpath is on and the pack carries ``Vf``; otherwise the module is a no-op.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float, max_k: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(512, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_k, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2,
                                             enable_nested_tensor=False)
        self.query = nn.Parameter(torch.zeros(1, d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, vf: Tensor) -> Tensor:
        # vf [B, K, 512]; every frame is valid (no padding mask)
        k = int(vf.shape[1])
        if k > self.pos.shape[1]:
            raise ValueError("V-path got K=%d frames > max_k=%d"
                             % (k, int(self.pos.shape[1])))
        h = self.proj(vf) + self.pos[:, :k]
        x = self.encoder(h)                                        # [B,K,D]
        scores = (x @ self.query.squeeze(0)) / self.scale          # [B,K]
        attn = torch.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)                 # [B,D]


class MemoryAttn(nn.Module):
    """SAM2-style memory read (W9 M-path): single-head cross-attention from
    the current fused state into the history memory bank, LayerScale(0-init)
    on the read-out so the path starts as an exact no-op."""

    def __init__(self, d_model: int, with_null: bool = False) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, fused: Tensor, mem: Tensor, mask: Tensor) -> Tensor:
        # fused [B,D]; mem [B,M,D]; mask [B,M] (1 = readable memory)
        if mem.shape[1] == 0:
            return torch.zeros_like(fused)
        q = self.q_proj(fused)
        scores = (mem @ q.unsqueeze(-1)).squeeze(-1) / self.scale
        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1) * mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (attn.unsqueeze(-1) * mem).sum(dim=1)
        return self.gamma * pooled


class UGate(nn.Module):
    """W9 micro-gate: 12 cheap per-modality solo-confidence stats -> 2 gates
    in (0,1) for the V-path and M-path corrections.  242 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, stats: Tensor) -> Tensor:
        return torch.sigmoid(self.net(stats))


class ModalityEncoder(nn.Module):
    """LN -> Linear -> (+ learned modality embedding) -> GELU -> Dropout -> LN."""

    def __init__(self, in_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.mod_emb = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(self.norm_in(x))
        h = h + self.mod_emb
        return self.norm_out(self.drop(F.gelu(h)))


def _sparsemax(logits: Tensor, mask: Tensor) -> Tensor:
    """Sparsemax over valid candidates, with zero mass on padded slots."""
    z = logits.masked_fill(mask <= 0, -1e9)
    z = z - z.max(dim=-1, keepdim=True).values
    zs = torch.sort(z, dim=-1, descending=True).values
    cssv = zs.cumsum(dim=-1) - 1.0
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    support = (zs * k.unsqueeze(0)) > cssv
    k_z = support.sum(dim=-1).clamp_min(1)
    tau = cssv.gather(1, (k_z - 1).unsqueeze(1)).squeeze(1) / k_z.to(z.dtype)
    p = (z - tau.unsqueeze(1)).clamp_min(0.0) * mask
    valid = mask.sum(dim=-1, keepdim=True) > 0
    return torch.where(valid, p, torch.zeros_like(p))


class HistoryAttnPool(nn.Module):
    """Attention-pool history slots, separating selection from admission."""

    def __init__(self, d_model: int, with_null: bool = False) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.gamma = nn.Parameter(torch.zeros(1))  # LayerScale, zero-init
        self.null_emb = nn.Parameter(torch.zeros(d_model)) if with_null else None

    def attend(self, cur: Tensor, hist: Tensor, mask: Tensor,
               use_null: bool = False, null_bias: Optional[Tensor] = None,
               normalizer: str = "softmax") -> Tuple[Tensor, Tensor, Tensor]:
        """Return pooled history, slot attention and null mass.

        ``use_null`` exists only for the frozen legacy ablation.  The normal
        and sentence-gate paths normalise over real, valid history slots.
        """
        if hist.shape[1] == 0:
            z = torch.zeros_like(cur)
            return z, mask, torch.zeros(cur.shape[0], device=cur.device)
        q = self.q_proj(cur)                                  # [B,D]
        scores = (hist @ q.unsqueeze(-1)).squeeze(-1) / self.scale   # [B,K]
        if use_null:
            if self.null_emb is None:
                raise RuntimeError("legacy null attention requested without null parameter")
            null_score = (self.null_emb @ q.unsqueeze(-1)).squeeze(-1) / self.scale
            scores = torch.cat([scores, null_score.unsqueeze(-1)], dim=-1)
            ext_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
        else:
            ext_mask = mask
        if use_null and null_bias is not None:
            scores[:, -1] = scores[:, -1] + null_bias
        scores = scores.masked_fill(ext_mask <= 0, -1e9)
        # all-masked rows need an explicit zero result: softmax(-1e9...) is
        # uniform, which would otherwise leak padded history into the pool.
        any_valid = ext_mask.sum(dim=-1, keepdim=True) > 0
        if normalizer == "sparsemax":
            attn = _sparsemax(scores, ext_mask)
        else:
            attn = torch.softmax(scores, dim=-1) * ext_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = torch.where(any_valid, attn, torch.zeros_like(attn))
        if use_null:
            null_mass = attn[:, -1]
            slot_attn = attn[:, :-1]
        else:
            null_mass = torch.zeros(cur.shape[0], device=cur.device)
            slot_attn = attn
        pooled = (slot_attn.unsqueeze(-1) * hist).sum(dim=1)  # [B,D]
        return pooled, slot_attn, null_mass

    def apply(self, cur: Tensor, pooled: Tensor, gate: Tensor) -> Tensor:
        return cur + gate.unsqueeze(-1) * self.gamma * pooled


class SentenceHistoryGate(nn.Module):
    """A deliberately small utterance-level history admission decision."""

    FEATURE_DIM = 24

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Start conservatively: evidence is retained until the gate learns a
        # counterexample.  The corresponding probability is 0.8.
        nn.init.constant_(self.net[-1].bias, math.log(4.0))

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


class HistoryUtilityCalibrator(SentenceHistoryGate):
    """Predict keep utility; its log-odds recalibrates the empty candidate."""
    pass


class JointHead(nn.Module):
    """2-layer TransformerEncoder over modality tokens + masked mean-pool."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float, num_classes: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True)
        # enable_nested_tensor=True (the default) makes the encoder return a
        # padded output whose sequence length is the max number of VALID
        # tokens per row (fully-masked trailing tokens are dropped), which
        # breaks subset evaluation; force the standard path so the output is
        # always [B, ntok, D].
        self.encoder = nn.TransformerEncoder(layer, num_layers=2,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        # tokens [B, ntok, D]; token_mask [B, ntok] (1 = include in pooling/attn)
        key_padding = token_mask <= 0
        # Degenerate safety: never let a row attend over an entirely padded set.
        all_pad = key_padding.all(dim=1)
        if bool(all_pad.any()):
            key_padding = key_padding.clone()
            key_padding[all_pad, 0] = False
        x = self.encoder(tokens, src_key_padding_mask=key_padding)
        m = token_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


class UtilityHead(nn.Module):
    """g_phi: 582-d evidence -> per-modality (mu, sigma) utility scores.

    Version 2 (iteration 3): the trunk is shared and the last layer is split
    into a mu head (supervised by the measured Shapley values) and a sigma
    head (used only by M2 regret gating).  Sigma is zero-initialised so that
    M2-off models are bit-identical to the round-1/2 single-head behaviour at
    initialisation, and so legacy single-head checkpoints can be shimmed by
    wrapping their final layer into the mu head with sigma left at zero.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model + 6, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_mu = nn.Linear(d_model, 3)
        self.head_sigma = nn.Linear(d_model, 3)
        nn.init.zeros_(self.head_sigma.weight)
        nn.init.zeros_(self.head_sigma.bias)

    def forward(self, emb_cat: Tensor, cos_sims: Tensor, conf: Tensor
                ) -> Tuple[Tensor, Tensor]:
        h = self.net(torch.cat([emb_cat, cos_sims, conf], dim=-1))
        return self.head_mu(h), self.head_sigma(h)


class ConstantContributionGate(nn.Module):
    """Fixed-preference control for the revised Innovation-2 router."""

    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(3))

    def forward(self, cur_embs, solo, mask, history_state=None):
        mu = self.mu.unsqueeze(0).expand(mask.shape[0], -1)
        return mu, torch.zeros_like(mu)


class EvidenceRouter(nn.Module):
    """Per-modality evidence router used by the revised Innovation 2.

    The router keeps the measured Shapley teacher and the bounded deployment
    formula unchanged.  It only replaces the old shared black-box MLP with
    local modality evidence plus a shared cross-modal context.
    """

    def __init__(self, d_model: int, num_classes: int, task: str,
                 history_feature_dim: int = 0) -> None:
        super().__init__()
        self.task = task
        self.history_feature_dim = int(history_feature_dim)
        self.project = nn.ModuleDict({m: nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 48), nn.GELU()
        ) for m in MODALITIES})
        evidence_dim = num_classes + 7 if task == "cls" else 11
        self.scorer = nn.Sequential(
            nn.Linear(3 * 48 + evidence_dim + self.history_feature_dim, 96), nn.GELU(),
            nn.Linear(96, 1))
        # A new router starts as the frozen uniform base.
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, cur_embs, solo, mask, history_state=None):
        mask = mask.detach().float()
        h = torch.stack([self.project[m](cur_embs[m].detach())
                         for m in MODALITIES], dim=1)
        h = h * mask.unsqueeze(-1)
        count = mask.sum(1, keepdim=True).clamp_min(1.0)
        context = h.sum(1, keepdim=True) / count.unsqueeze(-1)
        logits = torch.stack([solo[m].detach() for m in MODALITIES], dim=1)
        present = mask.unsqueeze(-1)
        if self.task == "cls":
            prob = F.softmax(logits, dim=-1)
            mean_prob = (prob * present).sum(1, keepdim=True) / count.unsqueeze(-1)
            top = prob.topk(2, dim=-1).values
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            entropy = entropy / math.log(prob.shape[-1])
            centered_logits = logits - logits.mean(-1, keepdim=True)
            scale = centered_logits.square().mean(-1, keepdim=True).sqrt()
            disagreement = (prob - mean_prob).abs().sum(-1, keepdim=True)
            agreement = (prob.argmax(-1) == mean_prob.argmax(-1)).float().unsqueeze(-1)
            stats = torch.cat([prob, top[:, :, :1], entropy,
                               top[:, :, :1] - top[:, :, 1:2], scale,
                               disagreement, agreement, present], dim=-1)
        else:
            pred = logits[:, :, 0] * mask
            mean = (pred * mask).sum(1, keepdim=True) / count
            diff = pred - mean
            spread = ((diff.square() * mask).sum(1, keepdim=True) / count).sqrt()
            pair = (pred.unsqueeze(2) - pred.unsqueeze(1)).abs() * mask.unsqueeze(1)
            stats = torch.cat([
                pred.unsqueeze(-1), diff.unsqueeze(-1), diff.abs().unsqueeze(-1),
                spread.unsqueeze(1).expand(-1, 3, -1), pair,
                pred.unsqueeze(1).expand(-1, 3, -1), present], dim=-1)
        features = torch.cat([h, context.expand_as(h), h-context,
                               stats * present], dim=-1)
        if self.history_feature_dim:
            if history_state is None:
                hs = torch.zeros(mask.shape[0], self.history_feature_dim,
                                 device=mask.device, dtype=h.dtype)
            else:
                hs = history_state.to(device=mask.device, dtype=h.dtype)
                if hs.ndim != 2 or hs.shape[0] != mask.shape[0] or hs.shape[1] != self.history_feature_dim:
                    raise ValueError("history_state must be [B,%d]" % self.history_feature_dim)
            features = torch.cat([features, hs.unsqueeze(1).expand(-1, 3, -1)], dim=-1)
        mu = self.scorer(features).squeeze(-1)
        return mu, torch.zeros_like(mu)


class ContributionCorrector(nn.Module):
    """Optional legacy residual, retained only for old ablation checkpoints."""

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({m: nn.Linear(d_model, num_classes)
                                    for m in MODALITIES})
        self.gamma = nn.Parameter(torch.zeros(3))

    def forward(self, cur_embs: Dict[str, Tensor], mu: Tensor) -> Tensor:
        centered = (mu - mu.mean(dim=-1, keepdim=True)).detach()
        delta = []
        for i, m in enumerate(MODALITIES):
            raw = self.heads[m](cur_embs[m])
            delta.append(raw * centered[:, i:i + 1] * self.gamma[i])
        return torch.stack(delta, dim=1)


class UGFModel(nn.Module):
    def __init__(self, cfg: M6Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(getattr(cfg, "%s_dim" % {"T": "text", "A": "audio", "V": "video"}[m]),
                               d, cfg.dropout)
            for m in MODALITIES
        })
        self.use_history = bool(cfg.use_history) and int(cfg.history_k) > 0
        if self.use_history:
            self.hist_pool = nn.ModuleDict(
                {m: HistoryAttnPool(d, with_null=(cfg.legacy_history_null or
                                                  cfg.history_gate_mode == "legacy_null" or
                                                  cfg.history_abstain_variant in ("legacy", "utility_softmax", "utility_semantic", "utility_sparsemax")))
                 for m in MODALITIES})
        self.history_gate = (SentenceHistoryGate(cfg.history_gate_hidden)
                             if self.use_history and cfg.history_gate_mode == "sentence"
                             else None)
        self.history_utility = (HistoryUtilityCalibrator(cfg.history_utility_hidden)
                                if self.use_history and cfg.history_abstain_variant in
                                ("utility_softmax", "utility_semantic", "utility_sparsemax")
                                else None)
        self._history_gate_epoch = 10 ** 9
        self.solo_heads = nn.ModuleDict(
            {m: nn.Linear(d, cfg.num_classes) for m in MODALITIES})
        self.joint = JointHead(d, cfg.num_heads, cfg.ff_dim, cfg.dropout,
                               cfg.num_classes)
        if cfg.utility in ("gate", "shapley"):
            if cfg.gate_architecture == "evidence":
                self.utility_head = EvidenceRouter(
                    d, cfg.num_classes, cfg.task,
                    history_feature_dim=getattr(cfg, "history_feature_dim", 0))
            elif cfg.gate_architecture == "constant":
                self.utility_head = ConstantContributionGate()
            else:
                self.utility_head = UtilityHead(d, cfg.dropout)
        else:
            self.utility_head = None
        self.contrib_corrector = (
            ContributionCorrector(d, cfg.num_classes)
            if cfg.contrib_correct else None)
        # W6: series-adversarial head (training only; never executed at eval).
        self.series_head = None
        if cfg.grl_lambda > 0.0 and int(cfg.n_series) > 0:
            self.series_head = nn.Sequential(
                nn.Linear(d, 128), nn.GELU(), nn.Linear(128, int(cfg.n_series)))
        # W9 StreamFusion modules (all no-ops when off / keys absent).
        self.vpath = VPath(d, cfg.num_heads, cfg.ff_dim, cfg.dropout) \
            if cfg.use_vpath else None
        self.mpath = MemoryAttn(d) if cfg.use_mpath else None
        self.ugate = UGate() if cfg.use_ugate else None
        if cfg.c1_alpha > 0.0 and cfg.class_prior is not None:
            prior = torch.tensor(cfg.class_prior, dtype=torch.float32)
            # persistent=False: never enters a state_dict, so old checkpoints
            # load unchanged and new ones reconstruct pi from the cfg dict.
            self.register_buffer("prior_buf", prior.clamp_min(1e-9),
                                 persistent=False)
        else:
            self.prior_buf = None
        # 2026-09-24 performance round-1 (plan 04 S3.1): additive residual
        # on the deployed solo-weighted logits.  Default "none" keeps every
        # existing checkpoint bit-identical.  Initialised in a SEPARATE branch
        # from prior_buf (P0 fix: the previous layout rebound the prior else
        # to the residual if, nulling prior_buf whenever residual == "none").
        self.residual_mode = cfg.residual
        self.resid_pairs = None
        self.resid_concat = None
        if cfg.residual == "pair":
            r = int(cfg.resid_rank)
            pairs = {}
            for a, b in (("T", "A"), ("T", "V"), ("A", "V")):
                va = nn.Linear(d, r)
                vb = nn.Linear(d, r)
                u = nn.Linear(r, cfg.num_classes)
                nn.init.zeros_(u.weight)
                nn.init.zeros_(u.bias)
                pairs[a + b] = nn.ModuleDict({"va": va, "vb": vb, "u": u})
            self.resid_pairs = nn.ModuleDict(pairs)
        elif cfg.residual == "concat":
            h = int(cfg.resid_hidden)
            w2 = nn.Linear(h, cfg.num_classes)
            nn.init.zeros_(w2.weight)
            nn.init.zeros_(w2.bias)
            self.resid_concat = nn.Sequential(
                nn.Linear(3 * d, h), nn.ReLU(), nn.Dropout(cfg.dropout), w2)

    # ------------------------------------------------------------------
    def set_history_gate_epoch(self, epoch: int) -> None:
        """Sets the training schedule state; eval always uses final behavior."""
        self._history_gate_epoch = int(epoch)

    @staticmethod
    def _normalised_entropy(prob: Tensor) -> Tensor:
        n_class = max(int(prob.shape[-1]), 2)
        return (-(prob * prob.clamp_min(1e-9).log()).sum(dim=-1)
                / math.log(float(n_class)))

    @staticmethod
    def _js_divergence(p: Tensor, q: Tensor) -> Tensor:
        mid = 0.5 * (p + q)
        kl_p = (p * (p.clamp_min(1e-9).log() - mid.clamp_min(1e-9).log())).sum(dim=-1)
        kl_q = (q * (q.clamp_min(1e-9).log() - mid.clamp_min(1e-9).log())).sum(dim=-1)
        return (0.5 * (kl_p + kl_q) / math.log(2.0)).clamp(0.0, 1.0)

    def _sentence_gate_features(self, base: Dict[str, Tensor],
                                pooled: Dict[str, Tensor],
                                slot_masks: Dict[str, Tensor],
                                attn: Dict[str, Tensor],
                                batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Build the fixed, detached 24-dimensional gate input.

        These are pre-feedback features.  Detaching prevents the gate BCE from
        training the encoders merely to make its own proxy easier to predict.
        """
        bsz = base["T"].shape[0]
        device = base["T"].device
        dtype = base["T"].dtype
        zero = torch.zeros(bsz, device=device, dtype=dtype)
        current_solo = {m: self.solo_heads[m](base[m]) for m in MODALITIES}
        history_solo = {m: self.solo_heads[m](pooled[m]) for m in MODALITIES}
        feats: List[Tensor] = []
        for mi, m in enumerate(MODALITIES):
            p_cur = torch.softmax(current_solo[m], dim=-1)
            top = torch.topk(p_cur, k=min(2, p_cur.shape[-1]), dim=-1).values
            margin = top[:, 0] - (top[:, 1] if top.shape[-1] > 1 else 0.0)
            feats.extend([self._normalised_entropy(p_cur), margin,
                          batch["modality_mask"][:, mi].to(dtype)])
        for m in MODALITIES:
            feats.append(slot_masks[m].sum(dim=-1) / max(int(slot_masks[m].shape[1]), 1))
        for m in MODALITIES:
            valid = slot_masks[m].sum(dim=-1) > 0
            js = self._js_divergence(torch.softmax(current_solo[m], dim=-1),
                                     torch.softmax(history_solo[m], dim=-1))
            ent = self._normalised_entropy(torch.softmax(history_solo[m], dim=-1))
            feats.extend([torch.where(valid, js, zero), torch.where(valid, ent, zero)])
        any_slot = batch["history_mask"] * (
            batch["history_modality_mask"].sum(dim=-1) > 0).to(dtype)
        has_history = any_slot.sum(dim=-1) > 0
        same = batch.get("speaker_same")
        if same is None:
            feats.extend([zero, zero])
        else:
            feats.extend([(same.to(dtype) * any_slot).sum(dim=-1)
                          / any_slot.sum(dim=-1).clamp_min(1.0),
                          torch.ones_like(zero)])
        feats.append(any_slot.sum(dim=-1) / max(int(any_slot.shape[1]), 1))
        text_mask = slot_masks["T"]
        raw_cur = batch["T_t"]
        raw_hist = batch["T_h"]
        norm_cur = raw_cur.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        norm_hist = raw_hist.norm(dim=-1).clamp_min(1e-9)
        cos = (raw_hist * raw_cur.unsqueeze(1)).sum(dim=-1) / (norm_hist * norm_cur)
        cos = cos.clamp(-1.0, 1.0)
        n_text = text_mask.sum(dim=-1)
        mean_cos = (cos * text_mask).sum(dim=-1) / n_text.clamp_min(1.0)
        max_cos = cos.masked_fill(text_mask <= 0, -1.0).max(dim=-1).values
        weighted = (cos * attn["T"]).sum(dim=-1)
        valid_text = n_text > 0
        feats.extend([torch.where(valid_text, mean_cos, zero),
                      torch.where(valid_text, max_cos, zero),
                      torch.where(valid_text, weighted, zero)])
        f = torch.stack(feats, dim=-1)
        if f.shape[-1] != SentenceHistoryGate.FEATURE_DIM:
            raise RuntimeError("sentence gate feature dimension changed: %d" % f.shape[-1])
        if self.cfg.history_gate_features == "base":
            f = f.clone()
            f[:, 21:24] = 0.0
        return f.detach(), has_history

    def _sentence_gate_values(self, features: Tensor, has_history: Tensor,
                              history_override: Optional[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert self.history_gate is not None
        logits = self.history_gate(features)
        prob = torch.sigmoid(logits / self.cfg.history_gate_temperature)
        prob = prob * has_history.to(prob.dtype)
        hard = (prob >= self.cfg.history_gate_threshold).to(prob.dtype) * has_history.to(prob.dtype)
        if history_override is not None:
            if history_override not in (0, 1):
                raise ValueError("history_override must be 0, 1, or None")
            applied = torch.full_like(prob, float(history_override)) * has_history.to(prob.dtype)
            hard = applied.detach()
            return logits, prob, hard, applied
        if self.training and self._history_gate_epoch < self.cfg.history_gate_warmup_epochs:
            applied = has_history.to(prob.dtype)
        elif self.training and self._history_gate_epoch < (
                self.cfg.history_gate_warmup_epochs + self.cfg.history_gate_soft_epochs):
            applied = prob
        elif self.cfg.history_gate_execution == "soft":
            applied = prob
        elif self.training:
            # straight-through hard gate: forward is binary, backward follows p.
            applied = hard + prob - prob.detach()
        else:
            applied = hard
        return logits, prob, hard, applied

    def encode(self, batch: Dict[str, Tensor], history_override: Optional[int] = None
               ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, object]]:
        """Return (current embeddings, history embeddings, W9 corrections).

        ``corr`` carries the raw correction tensors: ``delta_v`` = the V-path
        frame embedding MINUS the pooled-V encoder output (cur_embs keeps the
        pooled base), and ``corr_m`` = the LayerScale'd memory read-out.  The
        corrections are applied by ``forward`` exactly once (after the
        optional U-gate), so when the W9 switches are off this is exactly the
        old encode.
        """
        base_embs: Dict[str, Tensor] = {}
        cur_embs: Dict[str, Tensor] = {}
        hist_embs: Dict[str, Tensor] = {}
        corr: Dict[str, object] = {"delta_v": None, "corr_m": None}
        abst_list: List[Tensor] = []
        pooled: Dict[str, Tensor] = {}
        slot_masks: Dict[str, Tensor] = {}
        attention: Dict[str, Tensor] = {}
        v_pooled: Optional[Tensor] = None
        v_out: Optional[Tensor] = None
        for mi, m in enumerate(MODALITIES):
            cur = self.encoders[m](batch[m + "_t"])
            if m == "V" and self.vpath is not None and "Vf" in batch:
                # Bugfix 1 (audit 2026-09-21): keep the POOLED base in
                # cur_embs; the gated correction is applied exactly once in
                # forward() as  V + g * (frame - pooled).  Previously cur was
                # REPLACED here and the delta was added again in forward, i.e.
                # V_frame + g*(V_frame - V_pooled), so gate=0 could not
                # disable the frame branch.
                v_pooled = cur
                v_out = self.vpath(batch["Vf"])
            base_embs[m] = cur
            if self.use_history:
                hist = self.encoders[m](batch[m + "_h"])      # [B,K,D]
                slot_mask = batch["history_mask"] * batch["history_modality_mask"][:, :, mi]
                hist_embs[m] = hist
                slot_masks[m] = slot_mask
                use_null = (self.cfg.history_gate_mode == "legacy_null" or
                             self.cfg.history_abstain_variant in
                             ("legacy", "utility_softmax", "utility_semantic", "utility_sparsemax"))
                p, a, null_mass = self.hist_pool[m].attend(
                    cur, hist, slot_mask, use_null=use_null)
                pooled[m] = p
                attention[m] = a
                abst_list.append(null_mass)
            cur_embs[m] = cur
        # Core Innovation-3 variants: retain the original empty candidate,
        # then adjust only its pre-softmax score using predicted keep utility.
        utility_logit = None
        if (self.use_history and self.history_utility is not None and
                self.cfg.history_abstain_variant in
                ("utility_softmax", "utility_semantic", "utility_sparsemax")):
            features, has_history = self._sentence_gate_features(
                base_embs, pooled, slot_masks, attention, batch)
            utility_logit = self.history_utility(features)
            keep = torch.sigmoid(utility_logit / self.cfg.history_utility_temperature)
            null_bias = self.cfg.history_null_beta * (
                torch.log((1.0 - keep).clamp_min(1e-5)) -
                torch.log(keep.clamp_min(1e-5)))
            normalizer = ("sparsemax" if self.cfg.history_abstain_variant ==
                          "utility_sparsemax" else "softmax")
            pooled = {}
            attention = {}
            abst_list = []
            for m in MODALITIES:
                p, a, null_mass = self.hist_pool[m].attend(
                    base_embs[m], hist_embs[m], slot_masks[m], use_null=True,
                    null_bias=null_bias, normalizer=normalizer)
                pooled[m], attention[m] = p, a
                abst_list.append(null_mass)
            hard = (keep >= self.cfg.history_gate_threshold).to(keep.dtype) * has_history.to(keep.dtype)
            if history_override is not None:
                if history_override not in (0, 1):
                    raise ValueError("history_override must be 0, 1, or None")
                applied = torch.full_like(keep, float(history_override)) * has_history.to(keep.dtype)
                hard = applied.detach()
            elif self.training and self.cfg.history_gate_execution == "soft":
                applied = keep
            elif self.training:
                # Hard forward path with a straight-through soft gradient.
                applied = hard + keep - keep.detach()
            else:
                applied = hard
            gate_state_utility = {"history_gate_logit": utility_logit,
                                  "history_gate_prob": keep,
                                  "history_gate_hard": hard,
                                  "history_gate_applied": applied,
                                  "history_gate_features": features,
                                  "has_history": has_history}
        else:
            gate_state_utility = None
        gate_state: Dict[str, Optional[Tensor]] = {
            "history_gate_logit": None, "history_gate_prob": None,
            "history_gate_hard": None, "history_gate_applied": None,
            "history_gate_features": None, "has_history": None,
        }
        if self.use_history:
            if self.cfg.history_gate_mode == "sentence":
                if self.history_gate is None:
                    raise RuntimeError("sentence gate mode without gate module")
                features, has_history = self._sentence_gate_features(
                    base_embs, pooled, slot_masks, attention, batch)
                logit, prob, hard, applied = self._sentence_gate_values(
                    features, has_history, history_override)
                gate_state = {"history_gate_logit": logit, "history_gate_prob": prob,
                              "history_gate_hard": hard, "history_gate_applied": applied,
                              "history_gate_features": features, "has_history": has_history}
            elif gate_state_utility is not None:
                gate_state = gate_state_utility
                applied = gate_state_utility["history_gate_applied"]
            else:
                applied = torch.ones(base_embs["T"].shape[0], device=base_embs["T"].device,
                                     dtype=base_embs["T"].dtype)
                if history_override is not None:
                    applied = applied * float(history_override)
            for m in MODALITIES:
                cur_embs[m] = self.hist_pool[m].apply(base_embs[m], pooled[m], applied)
        if abst_list:
            self._abstain = torch.stack(abst_list, dim=1)   # [B,3]
        else:
            self._abstain = None
        if v_out is not None and v_pooled is not None:
            corr["delta_v"] = v_out - v_pooled
        if self.mpath is not None and batch["history_mask"].shape[1] > 0:
            mem_parts: List[Tensor] = []
            mask_parts: List[Tensor] = []
            hmm = batch["history_modality_mask"]
            ss = batch.get("speaker_same")
            if ss is None:
                ss = torch.ones_like(batch["history_mask"])
            ssf = (ss > 0).float()
            for mi, m in enumerate(MODALITIES):
                mem = hist_embs.get(m)
                if mem is None:
                    mem = self.encoders[m](batch[m + "_h"])  # shared weights
                mem_parts.append(mem)
                mask_parts.append(batch["history_mask"] * hmm[:, :, mi] * ssf)
            mem = torch.cat(mem_parts, dim=1)                # [B, 3K, D]
            mask = torch.cat(mask_parts, dim=1)
            # The memory path must obey the same history admission decision
            # as the history pool.  Otherwise rejected history can re-enter
            # through mpath and silently bypass the abstention gate.
            admission = gate_state.get("history_gate_applied")
            if admission is None:
                admission = torch.ones(mask.shape[0], device=mask.device,
                                       dtype=mask.dtype)
                if history_override is not None:
                    admission = admission * float(history_override)
            mask = mask * admission.detach().to(mask.dtype).unsqueeze(1)
            e = torch.stack([cur_embs[m] for m in MODALITIES], dim=1)
            mmk = batch["modality_mask"].unsqueeze(-1)
            fused = (e * mmk).sum(dim=1) / mmk.sum(dim=1).clamp_min(1.0)
            corr["corr_m"] = self.mpath(fused, mem, mask)
        corr.update(gate_state)
        return cur_embs, hist_embs, corr

    def gate_stats(self, solo: Dict[str, Tensor]) -> Tensor:
        """12-d cheap per-modality solo-confidence stats for the U-gate."""
        feats: List[Tensor] = []
        pred_t = solo["T"].detach().argmax(dim=-1)
        for m in MODALITIES:
            p = torch.softmax(solo[m].detach().float(), dim=-1)
            top2 = torch.topk(p, k=min(2, p.shape[-1]), dim=-1).values
            p_max = top2[:, 0]
            margin = top2[:, 0] - top2[:, 1]
            entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1)
            agree = (solo[m].detach().argmax(dim=-1) == pred_t).float()
            feats.extend([p_max, margin, entropy, agree])
        return torch.stack(feats, dim=-1)

    def _history_router_features(self, history_state=None) -> Optional[Tensor]:
        """Compact, detached history state for the evidence router."""
        if getattr(self.cfg, "history_feature_dim", 0) == 0:
            return None
        if history_state is None:
            return None
        prob = history_state.get("history_gate_prob")
        applied = history_state.get("history_gate_applied")
        has = history_state.get("has_history")
        abst = history_state.get("abstain")
        if prob is None and applied is None and has is None:
            return None
        if prob is None:
            prob = torch.zeros_like(applied if applied is not None else has)
        if applied is None:
            applied = torch.zeros_like(prob)
        if has is None:
            has = torch.zeros_like(prob)
        if abst is None:
            abst_mean = torch.zeros_like(prob)
        else:
            abst_mean = abst.detach().float().mean(dim=1)
        return torch.stack([prob, applied, has.to(prob.dtype), abst_mean], dim=-1).detach()

    def predict_utility(self, cur_embs, solo, batch, history_state=None):
        """Predict modality utility through the configured Innovation-2 head."""
        if self.cfg.gate_architecture in ("evidence", "constant"):
            hs = self._history_router_features(history_state)
            return self.utility_head(cur_embs, solo, batch["modality_mask"], hs)
        features = self.utility_features(cur_embs, solo)
        if self.cfg.gate_detach_inputs:
            features = tuple(v.detach() for v in features)
        return self.utility_head(*features)

    def build_tokens(self, cur_embs: Dict[str, Tensor],
                     hist_embs: Dict[str, Tensor],
                     batch: Dict[str, Tensor],
                     subset: Optional[Sequence[str]] = None,
                     token_weight: Optional[Tensor] = None,
                     history_gate: Optional[Tensor] = None,
                     history_hard: Optional[Tensor] = None,
                     include_history_tokens: bool = True
                     ) -> Tuple[Tensor, Tensor]:
        """Stack tokens and validity masks; ``subset`` restricts modalities.

        ``token_weight`` ([B,3] per-modality weights, optional) scales each
        token by its modality's weight -- the joint_softgate deployment path
        feeds the joint head with utility-weighted tokens (history tokens
        included); ``None`` leaves tokens unscaled.  A sentence gate applies
        once to raw history tokens.  Hard gates remove the tokens from both
        attention and masked pooling, so rejection has no token side path.
        """
        tokens: List[Tensor] = []
        masks: List[Tensor] = []
        for mi, m in enumerate(MODALITIES):
            include = 1.0 if (subset is None or m in subset) else 0.0
            w_m = None if token_weight is None else token_weight[:, mi].unsqueeze(-1)
            tok = cur_embs[m] if w_m is None else cur_embs[m] * w_m
            tokens.append(tok)
            masks.append(batch["modality_mask"][:, mi] * include)
        if self.use_history and include_history_tokens:
            for mi, m in enumerate(MODALITIES):
                if m not in hist_embs:
                    continue
                include = 1.0 if (subset is None or m in subset) else 0.0
                w_m = None if token_weight is None else token_weight[:, mi].unsqueeze(-1)
                slot_mask = (batch["history_mask"]
                             * batch["history_modality_mask"][:, :, mi])
                for j in range(hist_embs[m].shape[1]):
                    h = hist_embs[m][:, j, :]
                    h_gate = None if history_gate is None else history_gate.unsqueeze(-1)
                    tok = h if w_m is None else h * w_m
                    if h_gate is not None:
                        tok = tok * h_gate
                    hmask = slot_mask[:, j] * include
                    if history_hard is not None:
                        hmask = hmask * history_hard
                    masks.append(hmask)
                    tokens.append(tok)
        return torch.stack(tokens, dim=1), torch.stack(masks, dim=1)

    def utility_features(self, cur_embs: Dict[str, Tensor],
                         solo: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        e = [cur_embs[m] for m in MODALITIES]
        cos = [F.cosine_similarity(e[i], e[j], dim=-1).unsqueeze(-1)
               for (i, j) in COS_PAIRS]
        cos_sims = torch.cat(cos, dim=-1)                     # [B,3]
        if self.cfg.task == "cls":
            conf = torch.cat([
                F.softmax(solo[m].detach(), dim=-1).max(dim=-1, keepdim=True)[0]
                for m in MODALITIES], dim=-1)                 # [B,3]
        else:
            preds = torch.stack(
                [solo[m].detach().squeeze(-1) for m in MODALITIES], dim=-1)  # [B,3]
            med = preds.median(dim=-1, keepdim=True)[0]
            conf = (preds - med).abs()                        # [B,3]
        emb_cat = torch.cat(e, dim=-1)                        # [B, 3*d]
        return emb_cat, cos_sims, conf

    def deploy_weights(self, mu: Tensor,
                       sigma: Optional[Tensor] = None) -> Tensor:
        """w = softmax(score / tau) with floor + renormalisation.

        M2 regret gating scores ``mu - kappa * sigma``; otherwise ``mu``
        (bit-identical to the round-1/2 behaviour, and to legacy checkpoints
        whose shimmed sigma is exactly zero).
        """
        if self.cfg.use_m2 and sigma is not None:
            score = (mu - self.cfg.kappa * sigma) / self.cfg.tau
        else:
            score = mu / self.cfg.tau
        if getattr(self.cfg, "use_bounded_w", False):
            # Centered bounded weights from Innovation 2.  The scale is
            # invariant to a common utility offset and cannot collapse onto
            # one modality.
            r = mu - mu.mean(dim=-1, keepdim=True)
            lam = float(getattr(self.cfg, "bounded_lambda", 0.3))
            w = torch.full_like(mu, 1.0 / mu.shape[-1])
            w = w * (1.0 + lam * r /
                     r.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-6))
            w = w.clamp_min(0.0)
            return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        w = torch.softmax(score, dim=-1)
        w = w.clamp_min(self.cfg.eps_floor)
        return w / w.sum(dim=-1, keepdim=True)

    def forward(self, batch: Dict[str, Tensor],
                subset: Optional[Sequence[str]] = None,
                mix: Optional[Tuple[Tensor, Tensor]] = None,
                history_override: Optional[int] = None
                ) -> Dict[str, Tensor]:
        """``mix = (perm, lam)`` enables the M3 manifold-mixup pass.

        The utility-consistency target is built from g_phi applied to the two
        PRE-MIX embedding sets (detached); the mixed embeddings then drive
        the solo / joint / utility heads, so the task loss and the deployed
        path see the mixed sample.  History tokens in the joint head are
        left unmixed (mixup is defined at the cur_embs level only).
        """
        cur_embs, hist_embs, corr = self.encode(batch, history_override=history_override)
        g_mix = None
        g_mix_target = None
        if mix is not None and self.utility_head is not None:
            perm, lam = mix
            lam_col = lam.unsqueeze(-1)                       # [B,1]
            solo_orig = {m: self.solo_heads[m](cur_embs[m]) for m in MODALITIES}
            state = dict(corr, abstain=getattr(self, "_abstain", None))
            mu_i, sig_i = self.predict_utility(cur_embs, solo_orig, batch, state)
            cur_perm = {m: cur_embs[m][perm] for m in MODALITIES}
            solo_perm = {m: solo_orig[m][perm] for m in MODALITIES}
            state_perm = {k: (v[perm] if torch.is_tensor(v) and v.shape[0] == perm.shape[0] else v)
                          for k, v in state.items()}
            mu_j, sig_j = self.predict_utility(cur_perm, solo_perm, batch, state_perm)
            gi = torch.cat([mu_i, sig_i], dim=-1)             # [B,6]
            gj = torch.cat([mu_j, sig_j], dim=-1)
            g_mix_target = (lam_col * gi + (1.0 - lam_col) * gj).detach()
            cur_embs = {m: (lam_col * cur_embs[m]
                            + (1.0 - lam_col) * cur_embs[m][perm])
                        for m in MODALITIES}
        need_corr = (corr["delta_v"] is not None) or (corr["corr_m"] is not None)
        if need_corr:
            # Gates are cheap stats of the PRE-correction solo outputs; the
            # deployed heads then run on the corrected embeddings.
            tmp_solo = {m: self.solo_heads[m](cur_embs[m]) for m in MODALITIES}
            scale_v = scale_m = None
            if self.ugate is not None:
                g2 = self.ugate(self.gate_stats(tmp_solo))
                scale_v = g2[:, 0].unsqueeze(-1)
                scale_m = g2[:, 1].unsqueeze(-1)
            if corr["delta_v"] is not None:
                sv = scale_v if scale_v is not None else 1.0
                cur_embs = dict(cur_embs,
                                V=cur_embs["V"] + sv * corr["delta_v"])
            if corr["corr_m"] is not None:
                sm = scale_m if scale_m is not None else 1.0
                cur_embs = {m: cur_embs[m] + sm * corr["corr_m"]
                            for m in MODALITIES}
        solo = {m: self.solo_heads[m](cur_embs[m]) for m in MODALITIES}
        solo_stack = torch.stack([solo[m] for m in MODALITIES], dim=1)  # [B,3,C]
        if self.utility_head is not None:
            state = dict(corr, abstain=getattr(self, "_abstain", None))
            mu, sigma = self.predict_utility(cur_embs, solo, batch, state)
            w = self.deploy_weights(mu, sigma)
            if self.contrib_corrector is not None:
                solo_stack = solo_stack + self.contrib_corrector(cur_embs, mu)
                solo = {m: solo_stack[:, i, :] for i, m in enumerate(MODALITIES)}
            if g_mix_target is not None:
                g_mix = torch.cat([mu, sigma], dim=-1)        # [B,6]
        else:
            mu = None
            sigma = None
            w = torch.full((solo["T"].shape[0], 3), 1.0 / 3.0,
                           dtype=solo["T"].dtype, device=solo["T"].device)
        n_mask_fallback = 0
        history_gate = corr.get("history_gate_applied")
        history_hard = corr.get("history_gate_hard")
        token_hard_mask = (history_hard if (
            (self.cfg.history_gate_mode == "sentence" or
             self.cfg.history_abstain_variant in ("utility_softmax", "utility_semantic", "utility_sparsemax"))
            and (self.cfg.history_gate_execution == "hard" or history_override is not None)
        ) else None)
        # ``closed_loop`` is the integrated Innovation-1 path: the admitted
        # history has already changed cur_embs, the bounded utility weights
        # scale the current modality tokens, and the joint head is the final
        # predictor.  Raw history tokens are omitted there to avoid counting
        # the same admitted history twice.
        w_deploy = (w.detach() if getattr(self.cfg, "detach_utility_path", False)
                    else w)
        if self.cfg.deploy in ("joint_softgate", "closed_loop"):
            # Utility gates the FUSION INPUT: tokens (current and history)
            # scaled by the per-sample deploy weights feed the joint head;
            # its output is the deployed logits.
            tokens, tok_mask = self.build_tokens(cur_embs, hist_embs, batch,
                                                 subset, token_weight=w_deploy,
                                                 history_gate=history_gate,
                                                 history_hard=token_hard_mask,
                                                 include_history_tokens=(
                                                     self.cfg.deploy != "closed_loop"))
            joint_logits = self.joint(tokens, tok_mask)
            deployed = joint_logits
        else:
            tokens, tok_mask = self.build_tokens(cur_embs, hist_embs, batch,
                                                 subset, history_gate=history_gate,
                                                 history_hard=token_hard_mask)
            joint_logits = self.joint(tokens, tok_mask)
            if self.cfg.deploy == "joint_uniform":
                # Uniform token entry; the joint head output itself deploys.
                deployed = joint_logits
            else:  # "solo_weighted": weighted sum of the solo logits.
                if self.cfg.mask_fusion_fix:
                    # Framework candidate 2 (mask-aware output fusion): an
                    # absent modality (modality_mask=0, zero-filled features)
                    # must NOT contribute -- its head still emits nonzero
                    # logits from LayerNorm beta / head bias on zero input.
                    # Weights renormalize over PRESENT modalities only; rows
                    # with nothing present fall back to a uniform posterior
                    # (cls: zero logit vector; reg: 0.0) and are counted.
                    mm = batch["modality_mask"]
                    w_present = w_deploy * mm
                    s = w_present.sum(dim=-1, keepdim=True)
                    fallback = s.squeeze(-1) <= 0.0
                    w_present = w_present / s.clamp_min(1e-9)
                    deployed = (w_present.unsqueeze(-1) * solo_stack).sum(dim=1)
                    n_mask_fallback = int(fallback.sum().item())
                    if n_mask_fallback > 0:
                        deployed = deployed.clone()
                        deployed[fallback] = 0.0
                else:
                    deployed = (w_deploy.unsqueeze(-1) * solo_stack).sum(dim=1)    # [B,C]
                if (self.cfg.c1_alpha > 0.0 and self.cfg.task == "cls"
                        and self.prior_buf is not None):
                    # W9 C1: blend the uniform solo average with the
                    # prior-calibrated evidence term
                    #   alpha * (sum_m (log_softmax(solo_m) - log pi) + log pi)
                    alpha = self.cfg.c1_alpha
                    uniform_solo = solo_stack.mean(dim=1)
                    logpi = torch.log(self.prior_buf).to(deployed.dtype)
                    evidence = sum(
                        F.log_softmax(solo[m], dim=-1) for m in MODALITIES) \
                        - 2.0 * logpi
                    deployed = (1.0 - alpha) * uniform_solo + alpha * evidence
                if self.residual_mode != "none":
                    # 2026-09-24 performance round-1: additive residual on the
                    # deployed logits, applied AFTER the C1 blend (P0 fix:
                    # applying it before the blend silently discarded the
                    # residual whenever C1 was on).  cur_embs are the
                    # per-modality encoder outputs [B,192]; effective presence
                    # comes from modality_mask (pooled setting).
                    mm_r = batch["modality_mask"]
                    eff = {m: mm_r[:, i] for i, m in enumerate(MODALITIES)}
                    any_present = (mm_r.sum(dim=-1, keepdim=True) > 0
                                   ).to(deployed.dtype)
                    if self.residual_mode == "pair":
                        for (a, b), mod in self.resid_pairs.items():
                            gate_ab = (eff[a] * eff[b]).unsqueeze(-1)
                            if bool((gate_ab > 0).any()):
                                r = mod["u"](mod["va"](cur_embs[a])
                                             * mod["vb"](cur_embs[b]))
                                deployed = deployed + gate_ab * r
                    else:  # concat
                        cat = torch.cat(
                            [cur_embs[m] * eff[m].unsqueeze(-1)
                             for m in MODALITIES], dim=-1)
                        deployed = deployed + any_present * \
                            self.resid_concat(cat)
        series_logits = None
        if self.training and self.series_head is not None:
            # W6: the deployed fusion representation = masked mean of the three
            # final current modality embeddings, pushed through the GRL into a
            # series classifier.  Only runs in training mode; never at eval.
            e = torch.stack([cur_embs[m] for m in MODALITIES], dim=1)   # [B,3,D]
            mm = batch["modality_mask"].unsqueeze(-1)
            fused = (e * mm).sum(dim=1) / mm.sum(dim=1).clamp_min(1.0)
            series_logits = self.series_head(grl(fused, self.cfg.grl_lambda))
        out = {
            "solo": solo, "solo_stack": solo_stack,
            "joint": joint_logits, "deployed": deployed, "weights": w,
            "utility_score": mu, "utility_mu": mu, "utility_sigma": sigma,
            "g_mix": g_mix, "g_mix_target": g_mix_target,
            "series_logits": series_logits, "n_mask_fallback": n_mask_fallback,
            "cur_embs": cur_embs, "hist_embs": hist_embs,
            "abstain": getattr(self, "_abstain", None),
            "history_gate_logit": corr.get("history_gate_logit"),
            "history_gate_prob": corr.get("history_gate_prob"),
            "history_gate_hard": history_hard,
            "history_gate_applied": history_gate,
            "history_gate_features": corr.get("history_gate_features"),
            "has_history": corr.get("has_history"),
            "history_override": history_override,
            "tokens": tokens, "token_mask": tok_mask,
        }
        return out

    # ------------------------------------------------------------------
    def joint_on_subset(self, cur_embs: Dict[str, Tensor],
                        batch: Dict[str, Tensor],
                        subset: Sequence[str],
                        token_weight: Optional[Tensor] = None) -> Tensor:
        """JOINT head on current tokens, with optional deploy weights."""
        tokens = torch.stack([cur_embs[m] for m in MODALITIES], dim=1)
        mask = torch.zeros_like(batch["modality_mask"])
        for mi, m in enumerate(MODALITIES):
            if m in subset:
                mask[:, mi] = batch["modality_mask"][:, mi]
        if token_weight is not None:
            tokens = tokens * token_weight.unsqueeze(-1)
        return self.joint(tokens, mask)

    @torch.no_grad()
    def measure_shapley(self, batch: Dict[str, Tensor], y: Tensor,
                        standardize: bool = True,
                        token_weight: Optional[Tensor] = None) -> Tensor:
        """Exact 3-player Shapley of PER-SAMPLE subset utilities.

        Runs on the current model state (no grad), current tokens only.  Every
        utility U(S) is computed per sample (cls: log_softmax gathered at the
        true class per row; reg: -|yhat - y| per row), so phi is [B,3].  With
        ``standardize=True`` (training target) each sample's 3-vector is
        normalised to mean 0 / std 1 (std clamped at 1e-6); with
        ``standardize=False`` the raw per-sample phi is returned -- this is
        what the valid-only oracle deployment uses.  Training mode is
        preserved: this is called between training steps and must not leave
        the module in eval() behind.
        """
        was_training = self.training
        self.eval()
        try:
            cur_embs, _, _ = self.encode(batch)
            subsets: List[Tuple[str, ...]] = []
            for r in (1, 2, 3):
                subsets.extend(tuple(sorted(c))
                               for c in itertools.combinations(MODALITIES, r))
            u: Dict[Tuple[str, ...], Tensor] = {}
            for s in subsets:
                logits = self.joint_on_subset(cur_embs, batch, s,
                                              token_weight=token_weight)
                if self.cfg.task == "cls":
                    u[s] = F.log_softmax(logits, dim=-1).gather(
                        1, y.view(-1, 1)).squeeze(1)              # [B]
                else:
                    u[s] = -(logits.squeeze(-1) - y).abs()        # [B]
            bsz = u[subsets[0]].shape[0]
            phi = torch.zeros(bsz, 3, dtype=torch.float32, device=y.device)
            for mi in range(3):
                others = [MODALITIES[j] for j in range(3) if j != mi]
                for r in range(0, 3):
                    for s in itertools.combinations(others, r):
                        weight = (math.factorial(r) * math.factorial(2 - r)) / 6.0
                        with_m = tuple(sorted(s + (MODALITIES[mi],)))
                        gain = u[with_m]
                        if len(s) > 0:
                            gain = gain - u[tuple(sorted(s))]
                        phi[:, mi] = phi[:, mi] + weight * gain
            if standardize:
                phi = (phi - phi.mean(dim=1, keepdim=True)) / \
                    phi.std(dim=1, keepdim=True).clamp_min(1e-6)
            return phi
        finally:
            if was_training:
                self.train()

    def count_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
