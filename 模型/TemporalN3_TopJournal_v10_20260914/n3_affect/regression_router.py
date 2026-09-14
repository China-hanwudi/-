"""Scalar prediction routing; T/A/V softmax remains a modality selector."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .multimodal_router import DynamicEvidenceRouter, LowRankCrossModalMatch


class ScalarDynamicEvidenceRouter(DynamicEvidenceRouter):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        # The shared original component uses num_classes as its output width.
        # Width 1 here is a scalar score, never a one-class probability.
        super().__init__(d_model=d_model, num_classes=1, hidden=hidden, dropout=dropout)
        self.cross_modal_match = LowRankCrossModalMatch(d_model, hidden, 1, dropout)

    def forward(self, current: Tensor, history: Tensor, relation_repr: Tensor,
                current_context: Tensor, modality_mask: Tensor,
                history_modality_mask: Tensor, history_mask: Tensor) -> dict[str, Tensor]:
        batch, candidates, _, _ = history.shape
        cur = current[:, None].expand(-1, candidates, -1, -1)
        rel = relation_repr[:, :, None].expand(-1, -1, 3, -1)
        evidence = self.evidence(torch.cat([cur, history, (cur - history).abs(), rel], dim=-1))
        shared = self.shared_bottleneck(evidence.mean(dim=2, keepdim=True))
        private = torch.stack([self.private_bottlenecks[m](evidence[:, :, m]) for m in range(3)], dim=2)
        evidence = private + 0.5 * shared + self.modality_embedding[None, None]
        valid = history_modality_mask * modality_mask[:, None, :] * history_mask[:, :, None]
        cur_norm, hist_norm = cur.norm(dim=-1), history.norm(dim=-1)
        valid_count = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        scale = ((cur_norm + hist_norm) * valid).sum(dim=2, keepdim=True).div(valid_count).clamp_min(1e-6)
        modal_prediction = self.modality_head(evidence).squeeze(-1)
        # Missing streams must not participate in the reliability statistics.
        # The previous unmasked consensus treated zero-padded video as a real
        # modality and suppressed the valid text/audio routes.
        consensus = (modal_prediction * valid).sum(dim=2, keepdim=True) / valid_count
        disagreement = (modal_prediction - consensus).square()
        disagreement = disagreement * valid
        dispersion = disagreement.sum(dim=2) / valid_count.squeeze(-1)
        valid_hist_mean = (hist_norm * valid).sum(dim=2, keepdim=True) / valid_count
        reliability = torch.stack([
            F.cosine_similarity(cur, history, dim=-1, eps=1e-6),
            -(cur - history).norm(dim=-1) / scale,
            hist_norm / valid_hist_mean.clamp_min(1e-6),
            history_modality_mask, -disagreement,
        ], dim=-1)
        context = current_context[:, None, None].expand(-1, candidates, 3, -1)
        gate_logits = self.gate(torch.cat([evidence, context, reliability], dim=-1)).squeeze(-1)
        uncertainty = self.uncertainty(evidence).squeeze(-1)
        # ---- v5: same measured-counterfactual head and redundancy views as the
        # classification router, so both tasks share one innovation.
        views = self.view_split(evidence)
        cf_predicted = self.cf_aligner(evidence)
        modality_sign = self.sign_head(views["private"])
        cf_delta = (cf_predicted - cf_predicted.mean(dim=2, keepdim=True)) * valid
        gate_logits = gate_logits + torch.tanh(self.cf_correction_scale) * cf_delta
        weights = self._masked_softmax(gate_logits - 0.5 * uncertainty, valid)
        interaction_input = torch.cat([relation_repr, current_context[:, None].expand(-1, candidates, -1)], dim=-1)
        interaction = self.interaction(interaction_input)
        strength = torch.sigmoid(self.interaction_gate(interaction_input))
        cross_modal = self.cross_modal_match(current, history, current_context, valid)
        candidate = (weights * modal_prediction).sum(dim=2, keepdim=True) + 0.25 * strength * interaction + 0.15 * cross_modal
        return {"candidate_prediction_norm": candidate, "modality_weights": weights,
                "modality_disagreement": disagreement, "modal_dispersion": dispersion,
                "modality_uncertainty": uncertainty, "modal_prediction_norm": modal_prediction,
                "interaction_strength": strength.squeeze(-1), "cross_modal_prediction_norm": cross_modal,
                # v5 signals
                "cf_predicted_utility": cf_predicted,
                "cf_correction": cf_delta,
                "modality_sign": modality_sign,
                "shared_view": views["shared"],
                "private_view": views["private"],
                "shared_ratio": views["ratio"]}
