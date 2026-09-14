"""End-to-end N3 model with candidate-level hard-safe history routing."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .causal_contribution import UnimodalContributionHead
from .config import N3TrainConfig
from .encoders import SixWayEncoders
from .gating import TwoLevelGate
from .hard_fallback import CandidateRiskFallback
from .multimodal_router import DynamicEvidenceRouter
from .relation import SharedThreeByThree
from .utility import BidirectionalUtilityHeads, all_effect_deltas, fuse_variants


class TheoryAuxHead(nn.Module):
    """Small VAD regression head used only as an auxiliary training signal."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, 3)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.net(x))


class HistoricalEvidenceController(nn.Module):
    """Compact multi-scale historical retrieval with bounded recency decay."""
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, hidden, bias=False)
        self.key = nn.Linear(d_model, hidden, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.boundary = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, current: Tensor, history: Tensor, mask: Tensor) -> Tensor:
        # History is canonical oldest-to-newest (left padded).  Recency decay
        # therefore assigns the largest bonus to the last valid slot.
        q = self.query(current).unsqueeze(1)
        k = self.key(history)
        sim = (q * k).sum(-1) / (k.size(-1) ** 0.5)
        dist = torch.arange(history.size(1) - 1, -1, -1, device=history.device, dtype=history.dtype)
        decay = torch.exp(-self.scale.clamp(0.05, 4.0) * dist / max(history.size(1), 1))
        boundary = torch.sigmoid(self.boundary(torch.cat([current[:, None, :].expand_as(history), history], -1)).squeeze(-1))
        score = sim + decay[None, :] - 0.5 * boundary
        score = score.masked_fill(~mask.bool(), -1e4)
        alpha = F.softmax(score, dim=1) * mask
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (alpha.unsqueeze(-1) * self.value(history)).sum(dim=1)


class N3EmotionModel(nn.Module):
    """Multimodal backbone plus explicit candidate filtering and hard fallback."""

    def __init__(self, cfg: N3TrainConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or N3TrainConfig()
        self.cfg.validate()
        d = self.cfg.d_model
        self.encoders = SixWayEncoders(self.cfg)
        self.current_fuse = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.LayerNorm(d))
        self.history_fuse = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(self.cfg.dropout), nn.LayerNorm(d)
        )
        self.relation = SharedThreeByThree(d, self.cfg.relation_rank, self.cfg.dropout)
        self.time_fuse = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.LayerNorm(d))
        mix_tau = float(getattr(self.cfg, "mix_tau", 1.25))
        self.utility = BidirectionalUtilityHeads(d, self.cfg.gate_hidden, self.cfg.dropout, mix_tau=mix_tau)
        self.gate = TwoLevelGate(d, self.cfg.gate_hidden, self.cfg.dropout)
        # Explicit T/A/V evidence paths.  The router directly produces each
        # candidate's logits; it is not a post-hoc scale on a fused vector.
        self.evidence_router = DynamicEvidenceRouter(
            d, self.cfg.num_classes, self.cfg.gate_hidden, self.cfg.dropout
        )
        self.speaker_state_net = nn.Sequential(
            nn.Linear(d * 3 + 1, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.speaker_state_cell = nn.GRUCell(d, d)
        risk_feature_dim = 8 if self.cfg.risk_feature_version == "conflict_v2" else 4
        self.hard_fallback = CandidateRiskFallback(
            d,
            feature_dim=risk_feature_dim,
            risk_threshold=self.cfg.risk_threshold,
            use_adaptive_budget=bool(getattr(self.cfg, "use_adaptive_budget", True)),
            accept_budget_init=float(getattr(self.cfg, "accept_budget_init", 0.35)),
            accept_warmup_epochs=int(getattr(self.cfg, "accept_warmup_epochs", 3)),
        )
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=self.cfg.num_heads,
                dim_feedforward=self.cfg.ffn_dim,
                dropout=self.cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=self.cfg.num_layers,
        )
        self.classifier = nn.Linear(d, self.cfg.num_classes)
        self.current_unimodal_head = UnimodalContributionHead(
            d, self.cfg.gate_hidden, self.cfg.num_classes, self.cfg.dropout
        )
        self.current_modality_gate = nn.Sequential(
            nn.Linear(d, self.cfg.gate_hidden), nn.GELU(),
            nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.gate_hidden, 1),
        )
        # Zero initialization preserves the v9 current anchor at startup.  The
        # label-anchored path earns influence only when validation training
        # supports it.
        self.current_anchor_blend = nn.Parameter(torch.zeros(1))
        self.vad_head = TheoryAuxHead(d)
        self.history_controller = HistoricalEvidenceController(d, self.cfg.gate_hidden)
        # Monotone acceptance warm-up index.  Only the training loop advances it;
        # it is a schedule over epochs, never over dev/test metrics.
        self.current_epoch: int = 0
        # v6 core: predict each modality's OWN label from its private view.  On
        # CH-SIMS v2 these targets exist, which turns the counterfactual head
        # into a label-grounded causal estimator.
        self.unimodal_head = UnimodalContributionHead(
            self.cfg.gate_hidden, self.cfg.gate_hidden, self.cfg.num_classes, self.cfg.dropout
        )

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _encode_current(self, tokens: Tensor, modality_mask: Tensor) -> Tensor:
        """Mask unavailable modality tokens in attention and mean pooling."""
        visible = modality_mask.bool()
        all_missing = ~visible.any(dim=1)
        padding = ~visible
        if all_missing.any():
            # PyTorch attention cannot consume a row with every key masked.
            padding = padding.clone()
            padding[all_missing] = False
        encoded = self.context(tokens, src_key_padding_mask=padding)
        weights = visible.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.where(all_missing[:, None], torch.zeros_like(pooled), pooled)

    def _current_logits(
        self, streams: Mapping[str, Tensor], modality_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build a mask-aware current anchor with label-anchored experts."""
        tokens = torch.stack([streams["T_t"], streams["A_t"], streams["V_t"]], dim=1)
        pooled = self._encode_current(tokens, modality_mask)
        anchor = self.classifier(pooled)
        modal_logits = self.current_unimodal_head(tokens)
        entropy = -(F.softmax(modal_logits, dim=-1) * F.log_softmax(modal_logits, dim=-1)).sum(-1)
        gate_logits = self.current_modality_gate(tokens).squeeze(-1) - entropy.detach()
        gate_logits = gate_logits.masked_fill(~modality_mask.bool(), -1e4)
        weights = F.softmax(gate_logits, dim=-1) * modality_mask.to(gate_logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        anchored = (weights.unsqueeze(-1) * modal_logits).sum(dim=1)
        blend = torch.tanh(self.current_anchor_blend)
        logits = anchor + blend * (anchored - anchor)
        return logits, pooled, modal_logits, weights

    def _ablated_current_logits(
        self, current_tokens: Tensor, modality_mask: Tensor, drop_modality: int
    ) -> Tensor:
        """Anchor logits with one current modality zeroed (leave-one-out).

        Zeroing the *encoded token* matches how a missing modality is already
        represented downstream, so the ablated anchor stays in-distribution and
        the measured delta reflects stream information rather than an
        arbitrary out-of-distribution perturbation.
        """
        ablated = current_tokens.clone()
        ablated[:, drop_modality, :] = 0.0
        visible = modality_mask.clone()
        visible[:, drop_modality] = 0.0
        streams = {"T_t": ablated[:, 0], "A_t": ablated[:, 1], "V_t": ablated[:, 2]}
        return self._current_logits(streams, visible)[0]

    def measure_counterfactual_utility(
        self, batch: Mapping[str, Tensor], labels: Tensor
    ) -> Tensor:
        """Measured leave-one-modality-out CE increase on the current anchor.

        Returns a detached ``[B, 3]`` tensor used as a training target for the
        router's counterfactual head.  Runs entirely under ``no_grad`` so it
        never contaminates the primary gradient path.
        """
        modality_mask = batch.get("modality_mask")
        if modality_mask is None:
            modality_mask = labels.new_ones(labels.size(0), 3)
        modality_mask = modality_mask.to(device=labels.device, dtype=labels.dtype)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                streams = self.encoders(batch)
                tokens = torch.stack([streams["T_t"], streams["A_t"], streams["V_t"]], dim=1)
                base = F.cross_entropy(
                    self._current_logits(streams, modality_mask)[0], labels, reduction="none"
                )
                targets = []
                for k in range(3):
                    removed = F.cross_entropy(
                        self._ablated_current_logits(tokens, modality_mask, k), labels, reduction="none"
                    )
                    delta = (removed - base).clamp_min(0.0) * modality_mask[:, k]
                    targets.append(delta)
                stacked_targets = torch.stack(targets, dim=-1)
                scale = stacked_targets.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                return stacked_targets / scale
        finally:
            self.train(was_training)

    def _candidate_logits(self, current_pool: Tensor, candidate_repr: Tensor) -> Tensor:
        """Classify each candidate without mixing candidates before filtering."""
        batch, candidates, _ = candidate_repr.shape
        fused = self.history_fuse(
            torch.cat([current_pool[:, None, :].expand(-1, candidates, -1), candidate_repr], dim=-1)
        )
        tokens = torch.stack(
            [
                current_pool[:, None, :].expand(-1, candidates, -1),
                fused,
                candidate_repr,
            ],
            dim=2,
        )
        encoded = self.context(tokens.reshape(batch * candidates, 3, -1))
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled).view(batch, candidates, -1)

    def forward(
        self,
        batch: Mapping[str, Tensor],
        texts_current: list[str] | None = None,
        texts_history: list[str] | None = None,
        route_mode: str = "hard-safe",
    ) -> dict[str, Tensor]:
        if route_mode not in {"hard-safe", "current-only", "plain-history", "soft-gate"}:
            raise ValueError(f"unknown route_mode: {route_mode}")
        streams = self.encoders(batch, texts_current=texts_current, texts_history=texts_history)
        modality_mask = batch.get("modality_mask")
        if modality_mask is None:
            modality_mask = torch.ones(
                streams["T_t"].size(0), 3, device=streams["T_t"].device,
                dtype=streams["T_t"].dtype,
            )
        else:
            modality_mask = modality_mask.to(device=streams["T_t"].device,
                                              dtype=streams["T_t"].dtype)
        if self.training and self.cfg.current_modality_dropout > 0:
            keep = (torch.rand_like(modality_mask) >= self.cfg.current_modality_dropout).to(modality_mask.dtype)
            # Never erase the whole current clip; this is a missing-modality
            # augmentation, not a missing-example augmentation.
            empty = (keep * modality_mask).sum(dim=1, keepdim=True) == 0
            keep = torch.where(empty, torch.ones_like(keep), keep)
            modality_mask = modality_mask * keep
            for i, name in enumerate(("T", "A", "V")):
                streams[f"{name}_t"] = streams[f"{name}_t"] * modality_mask[:, i:i+1]
        if batch.get("speaker_same") is not None:
            streams["speaker_same"] = batch["speaker_same"]
        for name in ("T", "A", "V"):
            if streams[f"{name}_h"].ndim == 2:
                streams[f"{name}_h"] = streams[f"{name}_h"].unsqueeze(1)
        history_mask = batch.get("history_mask")
        if history_mask is None:
            history_mask = torch.ones(
                streams["T_h"].shape[:2], device=streams["T_h"].device, dtype=streams["T_h"].dtype
            )
        if history_mask.ndim == 1:
            history_mask = history_mask.unsqueeze(1)
        history_modality_mask = batch.get("history_modality_mask")
        if history_modality_mask is None:
            history_modality_mask = history_mask.unsqueeze(-1).expand(-1, history_mask.size(1), 3)
        elif history_modality_mask.ndim == 2:
            # A [B,K] legacy mask is a slot-validity mask, not a modality mask.
            history_modality_mask = history_modality_mask.unsqueeze(-1).expand(-1, -1, 3)
        history_modality_mask = history_modality_mask.to(
            device=history_mask.device, dtype=history_mask.dtype
        )
        if self.training:
            # Corrupt only historical evidence during training.  Current-only
            # remains an uncorrupted anchor, making the fallback identifiable.
            if self.cfg.history_dropout > 0:
                keep_slots = (
                    torch.rand_like(history_mask) >= self.cfg.history_dropout
                ).to(history_mask.dtype)
                history_mask = history_mask * keep_slots
            if self.cfg.modality_dropout > 0:
                keep_mods = (
                    torch.rand_like(history_modality_mask) >= self.cfg.modality_dropout
                ).to(history_modality_mask.dtype)
                history_modality_mask = history_modality_mask * keep_mods
            history_modality_mask = history_modality_mask * history_mask.unsqueeze(-1)
        history_mask = history_mask * (history_modality_mask.sum(dim=-1) > 0).to(history_mask.dtype)
        if bool(getattr(self.cfg, "disable_history", False)):
            # Ablation "drop history": every down-stream path sees an empty
            # history, so hard-safe degenerates to current-only by construction
            # rather than by post-hoc overwriting.
            history_mask = torch.zeros_like(history_mask)
            history_modality_mask = torch.zeros_like(history_modality_mask)
        for i, name in enumerate(("T", "A", "V")):
            streams[f"{name}_h"] = (
                streams[f"{name}_h"] * history_modality_mask[:, :, i:i + 1]
            )
        has_history = (history_mask.sum(dim=1) > 0).to(dtype=streams["T_t"].dtype)

        current_variants = fuse_variants(
            self.current_fuse, streams["T_t"], streams["A_t"], streams["V_t"]
        )
        current_pool = current_variants["TAV"]
        current_logits, current_context, current_unimodal_logits, current_modality_weights = self._current_logits(
            streams, modality_mask
        )

        pooled_history: dict[str, Tensor] = {}
        denom = history_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        for name in ("T", "A", "V"):
            pooled_history[name] = (streams[f"{name}_h"] * history_mask.unsqueeze(-1)).sum(dim=1) / denom
        history_variants = fuse_variants(
            self.current_fuse, pooled_history["T"], pooled_history["A"], pooled_history["V"]
        )
        deltas = all_effect_deltas(
            current_variants,
            history_variants,
            self.time_fuse,
            streams["T_t"],
            streams["A_t"],
            streams["V_t"],
            pooled_history["T"],
            pooled_history["A"],
            pooled_history["V"],
        )
        utilities = self.utility(deltas)
        gated = self.gate(
            current_pool,
            {"T_h": pooled_history["T"], "A_h": pooled_history["A"], "V_h": pooled_history["V"]},
            utilities,
        )

        if bool(getattr(self.cfg, "disable_relation", False)):
            # Ablation "drop candidate-level 3x3 relation module": the shared
            # current/history evidence grid is removed while every downstream
            # tensor keeps its shape, so the risk features fall back to the
            # zero-evidence regime instead of the model crashing.
            _b, _k = streams["T_h"].shape[:2]
            candidate_repr = streams["T_h"].new_zeros(_b, _k, self.cfg.d_model)
            relation_scores = candidate_repr.new_zeros(_b, _k, 3, 3)
            relation_grid = candidate_repr.new_zeros(_b, _k, 3, 3, self.cfg.relation_rank * 3)
            self.relation.last_evidence = {}
        else:
            candidate_repr, relation_scores, relation_grid = self.relation(
                streams,
                has_history=has_history,
                modality_mask=modality_mask,
                history_modality_mask=history_modality_mask,
                history_mask=history_mask,
            )
        # Speaker identity is a reliability cue, not a separate fusion path.
        speaker_same = streams.get("speaker_same")
        if speaker_same is None:
            speaker_same = torch.zeros_like(history_mask)
        speaker_same = speaker_same.to(dtype=history_mask.dtype)
        # Recurrent speaker-conditioned emotion state over valid history slots.
        # Canonical slots are oldest-to-newest, so update the state in order.
        slot_joint = (
            streams["T_h"] + streams["A_h"] + streams["V_h"]
        ) / 3.0
        state = current_context
        states = [None] * slot_joint.size(1)
        for k in range(slot_joint.size(1)):
            update = self.speaker_state_net(
                torch.cat(
                    [
                        streams["T_h"][:, k],
                        streams["A_h"][:, k],
                        streams["V_h"][:, k],
                        speaker_same[:, k:k + 1],
                    ],
                    dim=-1,
                )
            )
            proposed = self.speaker_state_cell(update, state)
            state = torch.where(history_mask[:, k:k + 1].bool(), proposed, state)
            states[k] = state
        speaker_memory = torch.stack(states, dim=1)
        candidate_repr = candidate_repr + 0.10 * speaker_memory * history_mask.unsqueeze(-1)
        retrieved_history = self.history_controller(current_context, candidate_repr, history_mask)
        candidate_repr = candidate_repr + 0.12 * retrieved_history[:, None, :]
        current_modal = torch.stack([streams["T_t"], streams["A_t"], streams["V_t"]], dim=1)
        history_modal = torch.stack([streams["T_h"], streams["A_h"], streams["V_h"]], dim=2)
        routed_modal = self.evidence_router(
            current_modal,
            history_modal,
            candidate_repr,
            current_context,
            modality_mask,
            history_modality_mask,
            history_mask,
        )
        candidate_logits = routed_modal["candidate_logits"]
        # Keep the legacy aggregate key for existing loss/reporting code while
        # exposing the full [B,K,3] routing evidence for audits.
        modality_keep_prob = routed_modal["modality_weights"].mean(dim=1)
        quality = routed_modal["modality_uncertainty"].neg().exp()
        relation_strength = relation_scores.abs().mean(dim=(2, 3))
        pair_quality = relation_grid.abs().mean(dim=(2, 3, 4))
        pair_conflict = relation_grid.var(dim=(2, 3, 4), unbiased=False)
        utility_signal = utilities["U_mix"].expand(-1, candidate_repr.size(1))
        joint_keep = gated["joint_keep_prob"].expand(-1, candidate_repr.size(1))
        risk_context_features = torch.stack(
            [
                relation_strength,
                pair_conflict,
                utility_signal,
                joint_keep,
            ],
            dim=-1,
        )
        legacy_relation_features = torch.stack(
            [
                relation_strength,
                pair_quality,
                pair_conflict,
                history_mask,
            ],
            dim=-1,
        )
        if self.cfg.risk_feature_version == "conflict_v2":
            current_prob = F.softmax(current_logits, dim=-1).detach().unsqueeze(1)
            candidate_prob = F.softmax(candidate_logits, dim=-1).detach()
            current_prob = current_prob.expand_as(candidate_prob)
            mixture = 0.5 * (current_prob + candidate_prob)
            js_divergence = 0.5 * (
                (current_prob * (current_prob.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(dim=-1)
                + (candidate_prob * (candidate_prob.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(dim=-1)
            )
            current_entropy = -(current_prob * current_prob.clamp_min(1e-8).log()).sum(dim=-1)
            candidate_entropy = -(candidate_prob * candidate_prob.clamp_min(1e-8).log()).sum(dim=-1)
            conflict_features = torch.stack(
                [
                    candidate_entropy - current_entropy,
                    current_prob.amax(dim=-1) - candidate_prob.amax(dim=-1),
                    (current_logits.argmax(dim=-1).unsqueeze(1) != candidate_logits.argmax(dim=-1)).to(candidate_logits.dtype),
                    js_divergence,
                ],
                dim=-1,
            )
            candidate_features = torch.cat([risk_context_features, conflict_features], dim=-1)
        else:
            candidate_features = legacy_relation_features
        routed = self.hard_fallback.filter_candidates(
            candidate_repr,
            candidate_features,
            current_logits,
            candidate_logits,
            history_mask=history_mask,
            cf_utility=routed_modal["cf_predicted_utility"].mean(dim=-1) if "cf_predicted_utility" in routed_modal else None,
            epoch=self.current_epoch,
        )
        if route_mode == "current-only":
            routed["use_history"] = torch.zeros_like(routed["use_history"])
        elif route_mode == "plain-history":
            valid = history_mask.bool()
            routed["use_history"] = has_history.bool()
            routed["history_logits"] = torch.where(
                valid.any(dim=1, keepdim=True),
                candidate_logits.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1)
                / valid.sum(dim=1, keepdim=True).clamp_min(1),
                current_logits,
            )
        elif route_mode == "soft-gate":
            routed["use_history"] = has_history.bool()
            soft = gated["use_history"].clamp(0.0, 1.0)
            routed["history_logits"] = routed["history_logits"]
            routed["logits"] = current_logits + soft * (routed["history_logits"] - current_logits)
        routed["use_history"] = routed["use_history"] & has_history.bool()
        if route_mode != "soft-gate":
            routed["logits"] = torch.where(
                routed["use_history"].unsqueeze(-1), routed["history_logits"], current_logits
            )
        routed["route_mode"] = route_mode
        vad = self.vad_head(current_context)
        return {
            "logits": routed["logits"],
            "probs": F.softmax(routed["logits"], dim=-1),
            "current_only_logits": current_logits,
            "current_unimodal_logits": current_unimodal_logits,
            "current_modality_weights": current_modality_weights,
            "current_modality_mask": modality_mask,
            "candidate_logits": candidate_logits,
            "history_mask": history_mask,
            "history_modality_mask": history_modality_mask,
            "history_logits": routed["history_logits"],
            "vad": vad,
            "relation_grid": relation_grid,
            "relation_scores": relation_scores,
            "alignment_grid": self.relation.last_evidence.get("alignment", relation_scores),
            "complementarity_grid": self.relation.last_evidence.get("complementarity", relation_scores.new_zeros(relation_scores.shape)),
            "conflict_grid": self.relation.last_evidence.get("conflict", relation_scores.new_zeros(relation_scores.shape)),
            "candidate_features": candidate_features,
            "modality_keep_prob": modality_keep_prob,
            "modality_quality": quality,
            "modality_route_weights": routed_modal["modality_weights"],
            "cross_modal_logits": routed_modal["cross_modal_logits"],
            "modality_uncertainty": routed_modal["modality_uncertainty"],
            "modality_disagreement": routed_modal["modality_disagreement"],
            "modal_candidate_logits": routed_modal["modal_candidate_logits"],
            "interaction_strength": routed_modal["interaction_strength"],
            "speaker_state_memory": speaker_memory,
            # ---- v5: expose the counterfactual head so the auxiliary loss can
            # actually receive it (this was the disconnected key in v4).
            "counterfactual_utility": routed_modal["cf_predicted_utility"],
            "cf_predicted_utility": routed_modal["cf_predicted_utility"],
            "cf_correction": routed_modal["cf_correction"],
            "modality_sign": routed_modal["modality_sign"],
            "shared_view": routed_modal["shared_view"],
            "private_view": routed_modal["private_view"],
            "shared_ratio": routed_modal["shared_ratio"],
            # v6: per-modality label predictions (CH-SIMS v2 supervises these
            # directly; other datasets leave them as a proxy).
            "unimodal_logits": self.unimodal_head(routed_modal["private_view"]),
            # v6 diagnostics for the accept-budget repair
            "accept_threshold": routed.get("accept_threshold"),
            "accept_budget": routed.get("accept_budget"),
            "accept_rate": routed.get("accept_rate"),
            "risk_collapse_penalty": routed.get("risk_collapse_penalty"),
            "is_training_forward": self.training,
            **utilities,
            **gated,
            **routed,
        }

    def predict_label(self, batch: Mapping[str, Tensor]) -> list[str]:
        with torch.no_grad():
            idx = self.forward(batch)["logits"].argmax(dim=-1).tolist()
        return [self.cfg.emotion_label_order[i] for i in idx]

    def export_card(self) -> dict[str, Any]:
        return {
            "model_name": "CMER-HardSafe",
            "protocol_id": "cmer_hardsafe_v2",
            "text_tower": self.cfg.text_tower,
            "trainable_parameters": self.count_trainable_parameters(),
            "parameter_budget": self.cfg.parameter_budget,
            "num_classes": self.cfg.num_classes,
            "emotion_label_order": list(self.cfg.emotion_label_order),
            "evidence_channels": ["alignment", "complementarity", "conflict"],
            "history_routing": "candidate_level_dynamic_evidence_router_with_uncertainty",
            "modality_selection": "context_conditioned_TAV_softmax_with_mask_agreement_and_disagreement",
            "speaker_state": "chronological_gru_conditioned_on_same_speaker",
        }
