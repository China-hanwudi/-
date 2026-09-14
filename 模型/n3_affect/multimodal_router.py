"""Compact, context-conditioned evidence routing for dialogue multimodality."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .counterfactual import CounterfactualUtilityAligner
from .redundancy import PrivateSharedReformulation, SignConsistencyHeads


class LowRankCrossModalMatch(nn.Module):
    """Compact multiplicative T/A/V matching used by both task adapters.

    This is a low-rank tensor-fusion style interaction: pairwise products are
    computed only for available streams, then gated by the current context.
    It preserves complementarity instead of treating modalities as competing
    independent logits.
    """

    def __init__(self, d_model: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        rank = max(8, min(hidden, d_model // 2))
        self.proj = nn.ModuleList([nn.Linear(d_model, rank, bias=False) for _ in range(3)])
        self.out = nn.Sequential(
            nn.Linear(rank * 3 + d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden), nn.Linear(hidden, out_dim)
        )
        self.gate = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, current: Tensor, history: Tensor, context: Tensor, valid: Tensor) -> Tensor:
        # current [B,3,D], history [B,K,3,D], valid [B,K,3]
        cur = [self.proj[i](current[:, i]).unsqueeze(1) for i in range(3)]
        hist = [self.proj[i](history[:, :, i]) for i in range(3)]
        pairs = [cur[0] * hist[1], cur[0] * hist[2], cur[1] * hist[2]]
        pair_valid = [valid[:, :, 0] * valid[:, :, 1], valid[:, :, 0] * valid[:, :, 2], valid[:, :, 1] * valid[:, :, 2]]
        z = torch.cat([p * m.unsqueeze(-1) for p, m in zip(pairs, pair_valid)] +
                      [context[:, None, :].expand(-1, history.size(1), -1)], dim=-1)
        return self.out(z) * torch.sigmoid(self.gate(context))[:, None]


class DynamicEvidenceRouter(nn.Module):
    """Route T/A/V evidence independently before candidate fusion."""

    def __init__(self, d_model: int, num_classes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.evidence = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )
        # MISA-style factorization: a compact shared affect channel plus
        # modality-private residual channels.  Keeping this inside the router
        # prevents a second bulky fusion backbone.
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.private_bottlenecks = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
            for _ in range(3)
        ])
        # evidence, current context, and agreement/distance/energy/availability/
        # prediction-disagreement cues
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model + 5, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Softplus()
        )
        self.modality_head = nn.Linear(d_model, num_classes)
        self.interaction = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)
        )
        self.interaction_gate = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.modality_embedding = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.cross_modal_match = LowRankCrossModalMatch(d_model, hidden, num_classes, dropout)
        self.utility_head = nn.Sequential(
            nn.Linear(2 * d_model + 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.fallback_temperature = nn.Parameter(torch.tensor(1.0))
        # ---- v5 innovations -------------------------------------------------
        # A genuine leave-one-modality-out utility head, trained against the
        # measured counterfactual computed by the caller (see counterfactual.py).
        self.cf_aligner = CounterfactualUtilityAligner(d_model, hidden, dropout)
        # Private/shared view split plus per-view redundancy signals.
        self.view_split = PrivateSharedReformulation(d_model, hidden, dropout)
        self.sign_head = SignConsistencyHeads(hidden, dropout)
        # Bounded residual scale for the measured-utility correction; starts at
        # zero so a freshly initialised model reproduces the v4 behaviour.
        self.cf_correction_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
        masked = logits.masked_fill(~valid.bool(), -1e4)
        weights = F.softmax(masked, dim=2) * valid.to(dtype=logits.dtype)
        return weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def forward(self, current: Tensor, history: Tensor, relation_repr: Tensor,
                current_context: Tensor, modality_mask: Tensor | None,
                history_modality_mask: Tensor, history_mask: Tensor) -> dict[str, Tensor]:
        batch, candidates, _, _ = history.shape
        if modality_mask is None:
            modality_mask = torch.ones(batch, 3, device=history.device, dtype=history.dtype)
        modality_mask = modality_mask.to(device=history.device, dtype=history.dtype)
        history_modality_mask = history_modality_mask.to(device=history.device, dtype=history.dtype)
        history_mask = history_mask.to(device=history.device, dtype=history.dtype)
        cur = current[:, None, :, :].expand(-1, candidates, -1, -1)
        rel = relation_repr[:, :, None, :].expand(-1, -1, 3, -1)
        evidence = self.evidence(torch.cat([cur, history, (cur - history).abs(), rel], dim=-1))
        shared_evidence = self.shared_bottleneck(evidence.mean(dim=2, keepdim=True))
        private_evidence = torch.stack(
            [self.private_bottlenecks[m](evidence[:, :, m]) for m in range(3)], dim=2
        )
        evidence = private_evidence + 0.5 * shared_evidence
        evidence = evidence + self.modality_embedding[None, None, :, :]
        cur_norm = cur.norm(dim=-1)
        hist_norm = history.norm(dim=-1)
        delta_norm = (cur - history).norm(dim=-1)
        agreement = F.cosine_similarity(cur, history, dim=-1, eps=1e-6)
        # Compute provisional per-modality predictions before routing.  The
        # Jensen-Shannon disagreement with their consensus is a label-free
        # reliability signal: contradictory, uncertain streams are downweighted
        # while complementary but consistent streams remain available.
        modal_logits = self.modality_head(evidence)
        modal_prob = F.softmax(modal_logits, dim=-1)
        valid = history_modality_mask * modality_mask[:, None, :] * history_mask[:, :, None]
        valid_count = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        scale = ((cur_norm + hist_norm) * valid).sum(dim=2, keepdim=True).div(valid_count).clamp_min(1e-6)
        consensus = (modal_prob * valid.unsqueeze(-1)).sum(dim=2, keepdim=True) / valid_count.unsqueeze(-1)
        disagreement = 0.5 * (
            modal_prob * (modal_prob.clamp_min(1e-8).log() - consensus.clamp_min(1e-8).log())
            + consensus * (consensus.clamp_min(1e-8).log() - modal_prob.clamp_min(1e-8).log())
        ).sum(dim=-1)
        disagreement = disagreement * valid
        valid_hist_mean = (hist_norm * valid).sum(dim=2, keepdim=True) / valid_count
        reliability = torch.stack([
            agreement, -(delta_norm / scale),
            hist_norm / valid_hist_mean.clamp_min(1e-6),
            history_modality_mask,
            -disagreement,
        ], dim=-1)
        context = current_context[:, None, None, :].expand(-1, candidates, 3, -1)
        gate_logits = self.gate(torch.cat([evidence, context, reliability], dim=-1)).squeeze(-1)
        uncertainty = self.uncertainty(evidence).squeeze(-1)
        utility_features = torch.cat([evidence, context,
            torch.stack([agreement, -disagreement, history_modality_mask], dim=-1)], dim=-1)
        counterfactual_utility = self.utility_head(utility_features).squeeze(-1)
        # ---- v5: measured counterfactual utility + redundancy views ---------
        # ``cf_aligner`` predicts the leave-one-modality-out utility that the
        # training loop measures with an actual ablated forward pass.  The
        # private/shared split lets the router distinguish "uninformative"
        # from "redundant" streams, and sign heads expose disagreement.
        views = self.view_split(evidence)
        cf_predicted = self.cf_aligner(evidence)
        modality_sign = self.sign_head(views["private"])
        cf_delta = cf_predicted - cf_predicted.mean(dim=2, keepdim=True)
        cf_delta = cf_delta * valid
        gate_logits = gate_logits + 0.35 * counterfactual_utility - 0.5 * uncertainty
        # Bounded, zero-initialised correction driven by the measured utility.
        gate_logits = gate_logits + torch.tanh(self.cf_correction_scale) * cf_delta
        temperature = self.fallback_temperature.clamp(0.5, 2.0)
        gate_logits = gate_logits / (temperature + 0.25 * uncertainty.detach())
        weights = self._masked_softmax(gate_logits, valid)
        weights = weights * (history_mask[:, :, None] > 0).to(dtype=weights.dtype)
        routed_logits = (weights.unsqueeze(-1) * modal_logits).sum(dim=2)
        interaction_input = torch.cat([relation_repr, current_context[:, None, :].expand(-1, candidates, -1)], dim=-1)
        interaction_logits = self.interaction(interaction_input)
        interaction_strength = torch.sigmoid(self.interaction_gate(interaction_input))
        cross_modal_logits = self.cross_modal_match(current, history, current_context, valid)
        routed_logits = routed_logits + 0.25 * interaction_strength * interaction_logits
        routed_logits = routed_logits + 0.15 * cross_modal_logits
        return {
            "candidate_logits": routed_logits, "modality_weights": weights,
            "modality_uncertainty": uncertainty, "modal_candidate_logits": modal_logits,
            "modality_disagreement": disagreement,
            "counterfactual_utility": counterfactual_utility,
            "fallback_temperature": temperature.detach(),
            "shared_evidence": shared_evidence,
            "private_evidence": private_evidence,
            "interaction_logits": interaction_logits, "interaction_strength": interaction_strength.squeeze(-1),
            "cross_modal_logits": cross_modal_logits,
            "routed_evidence": evidence,
            # v5 signals
            "cf_predicted_utility": cf_predicted,
            "cf_correction": cf_delta,
            "modality_sign": modality_sign,
            "shared_view": views["shared"],
            "private_view": views["private"],
            "shared_ratio": views["ratio"],
        }
