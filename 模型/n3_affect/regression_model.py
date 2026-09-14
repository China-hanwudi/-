"""Continuous N3 adaptation preserving candidate routing and hard fallback.

Predictions are unbounded normalized scores u, with sentiment = target_scale*u.
No labels, categorical probabilities, or categorical entropy enter forward.

v7 additions
------------
* **Current-as-candidate routing.**  On non-dialogue data (CH-SIMS v2 /
  CMU-MOSEI) there is no conversational history, so the entire evidence-routing
  stack used to sit idle and the model degenerated to its mean-pooled anchor.
  The current modality triplet is now itself routed as an evidence candidate:
  the measured-utility machinery (v5/v6) acts on the *current modalities*, and
  the risk head chooses between the anchor and the routed fusion per sample.
  With dialogue history present, the current candidate is simply concatenated
  with the K historical candidates -- one mechanism covers both regimes.
* **Continuous unimodal head.**  CH-SIMS v2 provides ``label_T/A/V``; a scalar
  per-modality head on the *current* private views is supervised against them
  (auxiliary loss only -- never inside a loss difference).
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .causal_contribution import UnimodalContributionHead
from .encoders import SixWayEncoders
from .gating import TwoLevelGate
from .hard_fallback import CandidateRiskFallback
from .regression_config import N3RegressionConfig
from .regression_router import ScalarDynamicEvidenceRouter
from .relation import SharedThreeByThree
from .utility import BidirectionalUtilityHeads, all_effect_deltas, fuse_variants


INPUT_KEYS = frozenset({"T_t", "A_t", "V_t", "T_h", "A_h", "V_h", "history_mask",
                        "modality_mask", "history_modality_mask", "speaker_same",
                        "unimodal_labels"})
REQUIRED_KEYS = INPUT_KEYS - {"speaker_same", "unimodal_labels"}


class N3SentimentModel(nn.Module):
    def __init__(self, cfg: N3RegressionConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = cfg.d_model
        self.encoders = SixWayEncoders(cfg)
        self.current_fuse = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.LayerNorm(d))
        self.time_fuse = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.LayerNorm(d))
        self.relation = SharedThreeByThree(d, cfg.relation_rank, cfg.dropout)
        self.utility = BidirectionalUtilityHeads(d, cfg.gate_hidden, cfg.dropout, mix_tau=cfg.mix_tau)
        self.gate = TwoLevelGate(d, cfg.gate_hidden, cfg.dropout)
        self.evidence_router = ScalarDynamicEvidenceRouter(d, cfg.gate_hidden, cfg.dropout)
        self.speaker_state_net = nn.Sequential(nn.Linear(d * 3 + 1, d), nn.GELU(), nn.LayerNorm(d))
        self.speaker_state_cell = nn.GRUCell(d, d)
        self.hard_fallback = CandidateRiskFallback(
            d, feature_dim=8, risk_threshold=cfg.risk_threshold,
            use_adaptive_budget=bool(getattr(cfg, "use_adaptive_budget", True)),
            accept_budget_init=float(getattr(cfg, "accept_budget_init", 0.35)),
            accept_warmup_epochs=int(getattr(cfg, "accept_warmup_epochs", 3)),
        )
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.num_heads, dim_feedforward=cfg.ffn_dim, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True), num_layers=cfg.num_layers)
        self.regressor = nn.Linear(d, 1)
        # Monotone acceptance warm-up index, advanced by the training loop only.
        self.current_epoch: int = 0
        # v7: continuous per-modality head for CH-SIMS v2 unimodal labels.  It
        # consumes the *current* private views produced by the shared
        # private/shared splitter inside the evidence router, so it adds no
        # second decomposition.
        self.unimodal_head = UnimodalContributionHead(
            cfg.gate_hidden, cfg.gate_hidden, 1, cfg.dropout
        )

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _ablated_current_prediction(self, current_tokens: Tensor, drop_modality: int) -> Tensor:
        """Current anchor with one modality token zeroed (leave-one-out)."""
        ablated = current_tokens.clone()
        ablated[:, drop_modality, :] = 0.0
        return self.regressor(self.context(ablated).mean(dim=1))

    def measure_counterfactual_utility(self, batch: Mapping[str, Tensor], targets: Tensor) -> Tensor:
        """Measured leave-one-modality-out MAE increase on the current anchor.

        Returns a detached ``[B, 3]`` tensor used to train the router's
        counterfactual head.  Values are in normalised (y/3) units and are
        rescaled per sample so the head learns a ranking across modalities.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self._validate_inputs(batch)
                streams = self.encoders(batch)
                tokens = torch.stack([streams["T_t"], streams["A_t"], streams["V_t"]], dim=1)
                if targets.ndim == 1:
                    targets = targets.unsqueeze(-1)
                modality_mask = batch["modality_mask"].to(device=targets.device, dtype=targets.dtype)
                base = (self.regressor(self.context(tokens).mean(dim=1)) - targets).abs()
                collected = []
                for k in range(3):
                    removed = (self._ablated_current_prediction(tokens, k) - targets).abs()
                    visible = modality_mask[:, k:k + 1]
                    delta = (removed - base).clamp_min(0.0) * visible
                    collected.append(delta)
                stacked = torch.cat(collected, dim=-1)
                scale = stacked.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                return stacked / scale
        finally:
            self.train(was_training)

    def _validate_inputs(self, batch: Mapping[str, Tensor]) -> None:
        if set(batch) - INPUT_KEYS or REQUIRED_KEYS - set(batch):
            raise ValueError(f"Feature-only input required; extra={set(batch)-INPUT_KEYS}, missing={REQUIRED_KEYS-set(batch)}")
        if batch["T_t"].ndim != 2 or batch["T_h"].ndim != 3:
            raise ValueError("Use current [B,D] and history [B,K,D], with explicit padded history slots")
        b, k = batch["T_h"].shape[:2]
        if not b or not k:
            raise ValueError("Batch and padded history slot count must be positive")
        for modality, dimension in (("T", self.cfg.text_dim), ("A", self.cfg.audio_dim), ("V", self.cfg.video_dim)):
            if tuple(batch[f"{modality}_t"].shape) != (b, dimension) or tuple(batch[f"{modality}_h"].shape) != (b, k, dimension):
                raise ValueError(f"Shape mismatch for modality {modality}")
        for key, shape in (("history_mask", (b, k)), ("modality_mask", (b, 3)),
                           ("history_modality_mask", (b, k, 3)), ("speaker_same", (b, k))):
            if key in batch and (tuple(batch[key].shape) != shape or not torch.all((batch[key] == 0) | (batch[key] == 1))):
                raise ValueError(f"Invalid binary mask {key}, expected {shape}")
        if not torch.all(batch["history_modality_mask"] <= batch["history_mask"].unsqueeze(-1)):
            raise ValueError("A missing history slot cannot contain an available modality")
        if "unimodal_labels" in batch:
            ul = batch["unimodal_labels"]
            if tuple(ul.shape) != (b, 3) or not ul.is_floating_point():
                raise ValueError("unimodal_labels must be a floating tensor of shape [B, 3]")

    def forward(self, batch: Mapping[str, Tensor], route_mode: str = "hard-safe") -> dict:
        if route_mode not in {"hard-safe", "current-only", "plain-history"}:
            raise ValueError("Regression supports hard-safe, current-only, or plain-history")
        self._validate_inputs(batch)
        streams = self.encoders(batch)
        history_mask = batch["history_mask"].to(dtype=streams["T_t"].dtype)
        modality_mask = batch["modality_mask"].to(dtype=history_mask.dtype)
        hist_modal_mask = batch["history_modality_mask"].to(dtype=history_mask.dtype)
        if self.training:
            if self.cfg.history_dropout:
                history_mask = history_mask * (torch.rand_like(history_mask) >= self.cfg.history_dropout)
            if self.cfg.modality_dropout:
                hist_modal_mask = hist_modal_mask * (torch.rand_like(hist_modal_mask) >= self.cfg.modality_dropout)
        hist_modal_mask = hist_modal_mask * history_mask.unsqueeze(-1)
        history_mask = history_mask * (hist_modal_mask.sum(dim=-1) > 0)
        # Explicit adaptation fix: every later path sees the same dropped/missing evidence.
        for i, name in enumerate(("T", "A", "V")):
            streams[f"{name}_h"] = streams[f"{name}_h"] * hist_modal_mask[:, :, i:i+1]
        speaker_same = batch.get("speaker_same", torch.zeros_like(history_mask)).to(dtype=history_mask.dtype)
        streams["speaker_same"] = speaker_same
        has_history = history_mask.sum(dim=1) > 0
        current_tokens = torch.stack([streams["T_t"], streams["A_t"], streams["V_t"]], dim=1)
        current_context = self.context(current_tokens).mean(dim=1)
        current = self.regressor(current_context)
        variants = fuse_variants(self.current_fuse, streams["T_t"], streams["A_t"], streams["V_t"])
        denom = history_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = {name: (streams[f"{name}_h"] * history_mask.unsqueeze(-1)).sum(dim=1) / denom for name in ("T", "A", "V")}
        hist_variants = fuse_variants(self.current_fuse, pooled["T"], pooled["A"], pooled["V"])
        deltas = all_effect_deltas(variants, hist_variants, self.time_fuse,
                                   streams["T_t"], streams["A_t"], streams["V_t"], pooled["T"], pooled["A"], pooled["V"])
        utilities = self.utility(deltas)
        gated = self.gate(variants["TAV"], {f"{name}_h": pooled[name] for name in ("T", "A", "V")}, utilities)
        candidate_repr, relation_scores, relation_grid = self.relation(
            streams, has_history=has_history.to(history_mask.dtype), modality_mask=modality_mask,
            history_modality_mask=hist_modal_mask, history_mask=history_mask)
        state = current_context
        states = [None] * history_mask.size(1)
        for k in range(history_mask.size(1)):
            update = self.speaker_state_net(torch.cat([streams["T_h"][:, k], streams["A_h"][:, k],
                                                       streams["V_h"][:, k], speaker_same[:, k:k+1]], dim=-1))
            state = torch.where(history_mask[:, k:k+1].bool(), self.speaker_state_cell(update, state), state)
            states[k] = state
        memory = torch.stack(states, dim=1)
        candidate_repr = candidate_repr + 0.10 * memory * history_mask.unsqueeze(-1)
        modal = self.evidence_router(
            current_tokens, torch.stack([streams["T_h"], streams["A_h"], streams["V_h"]], dim=2),
            candidate_repr, current_context, modality_mask, hist_modal_mask, history_mask)
        raw_candidates = modal["candidate_prediction_norm"]
        # MOSEI clips are frequently isolated; a noisy historical neighbour
        # must not make an unbounded scalar jump.  Treat history as a bounded
        # residual around the current anchor so the risk head can learn a
        # meaningful accept/reject boundary instead of rejecting everything.
        delta = raw_candidates - current[:, None, :]
        candidates = current[:, None, :] + float(self.cfg.history_delta_cap) * torch.tanh(delta)
        modal["raw_candidate_prediction_norm"] = raw_candidates
        modal["candidate_prediction_norm"] = candidates
        context_features = torch.stack([
            relation_scores.abs().mean(dim=(2, 3)), relation_grid.var(dim=(2, 3, 4), unbiased=False),
            utilities["U_mix"].expand(-1, history_mask.size(1)),
            gated["joint_keep_prob"].expand(-1, history_mask.size(1)),
        ], dim=-1)
        difference = candidates.detach().squeeze(-1) - current.detach()
        conflict_features = torch.stack([
            difference.abs(), difference,
            ((candidates.detach().squeeze(-1) >= 0) != (current.detach() >= 0)).to(current.dtype),
            modal["modal_dispersion"].detach(),
        ], dim=-1)
        features = torch.cat([context_features, conflict_features], dim=-1)
        # ---- v7: current-as-candidate --------------------------------------
        # On non-dialogue data the K historical slots are empty, so without
        # this branch the whole routing stack would be dead weight.  Route the
        # current modality triplet as its own evidence candidate; the measured
        # counterfactual utility (computed on the anchor) supervises this head,
        # and the risk head chooses per sample between the anchor and the
        # routed fusion.  With dialogue history, the current candidate is
        # simply concatenated with the K historical candidates.
        # Strict empty-history contract: a missing history cannot be converted
        # into a synthetic candidate.  This makes hard-safe exactly current-only
        # when every history slot is padding; callers can still enable the
        # current-as-candidate path explicitly for isolated-clip studies.
        allow_current_candidate = bool(getattr(self.cfg, "allow_current_candidate_without_history", False))
        ones_slot = torch.ones_like(history_mask[:, :1]) if allow_current_candidate else has_history[:, None].to(history_mask.dtype)
        cur_router = self.evidence_router(
            current_tokens,
            current_tokens.unsqueeze(1),          # triplet as one candidate [B,1,3,D]
            current_context.unsqueeze(1),          # relation repr [B,1,D]
            current_context,
            modality_mask,
            modality_mask.unsqueeze(1),            # [B,1,3]
            ones_slot,                             # [B,1]
        )
        cur_raw = cur_router["candidate_prediction_norm"]      # [B,1,1]
        cur_cf = cur_router["cf_predicted_utility"]            # [B,1,3]
        cur_weights = cur_router["modality_weights"]           # [B,1,3]
        cur_views = self.evidence_router.view_split(current_tokens.unsqueeze(1))
        cur_private = cur_views["private"].squeeze(1)          # [B,3,H]
        unimodal_pred = self.unimodal_head(cur_private).squeeze(-1)  # [B,3]
        unimodal_pred = unimodal_pred * modality_mask
        cur_cf_mean = cur_cf.mean(dim=-1)                      # [B,1]
        cur_context_features = torch.stack([
            torch.ones_like(cur_cf_mean), torch.zeros_like(cur_cf_mean),
            cur_cf_mean, torch.ones_like(cur_cf_mean),
        ], dim=-1)                                             # [B,1,4]
        cur_difference = cur_raw.detach().squeeze(-1) - current.detach()
        cur_conflict_features = torch.stack([
            cur_difference.abs(), cur_difference,
            ((cur_raw.detach().squeeze(-1) >= 0) != (current.detach() >= 0)).to(current.dtype),
            cur_router["modal_dispersion"].detach(),
        ], dim=-1)                                             # [B,1,4]
        cur_features = torch.cat([cur_context_features, cur_conflict_features], dim=-1)
        # Merge the current candidate with the historical candidates into one
        # routing decision.  All tensors keep the candidate axis explicit.
        all_raw = torch.cat([cur_raw, raw_candidates], dim=1)              # [B,1+K,1]
        all_delta = all_raw - current[:, None, :]
        all_candidates = current[:, None, :] + float(self.cfg.history_delta_cap) * torch.tanh(all_delta)
        all_repr = torch.cat([current_context.unsqueeze(1), candidate_repr], dim=1)
        all_mask = torch.cat([ones_slot, history_mask], dim=1)             # [B,1+K]
        all_features = torch.cat([cur_features, features], dim=1)
        all_cf = torch.cat([cur_cf_mean, modal["cf_predicted_utility"].mean(dim=-1)], dim=1)
        routed = self.hard_fallback.filter_candidates(
            all_repr, all_features, current, all_candidates, all_mask,
            uncertainty_coef=float(self.cfg.risk_upper_coef),
            cf_utility=all_cf,
            epoch=self.current_epoch,
        )
        if route_mode == "current-only":
            routed["use_history"] = torch.zeros_like(routed["use_history"])
            prediction = current
        elif route_mode == "plain-history":
            routed["use_history"] = has_history
            unfiltered = (candidates * history_mask.unsqueeze(-1)).sum(dim=1) / denom
            prediction = torch.where(has_history[:, None], unfiltered, current)
        else:
            prediction = routed["logits"]
        return {
            "prediction_norm": prediction, "prediction": self.cfg.target_scale * prediction,
            "current_prediction_norm": current, "current_prediction": self.cfg.target_scale * current,
            "candidate_prediction_norm": candidates, "candidate_prediction": self.cfg.target_scale * candidates,
            "raw_candidate_prediction_norm": raw_candidates,
            "history_prediction_norm": routed["history_logits"], "history_mask": history_mask,
            "history_modality_mask": hist_modal_mask, "candidate_features": features,
            "candidate_risk_logits": routed["candidate_risk_logits"], "candidate_risk": routed["candidate_risk"],
            "candidate_risk_upper": routed["candidate_risk_upper"], "safe_candidates": routed["safe_candidates"],
            "safe_count": routed["safe_count"], "use_history": routed["use_history"],
            "candidate_weights": routed["candidate_weights"], "route_mode": route_mode,
            "modality_route_weights": modal["modality_weights"], "modality_disagreement": modal["modality_disagreement"],
            "cross_modal_prediction_norm": modal.get("cross_modal_prediction_norm"),
            "modal_prediction_norm": modal["modal_prediction_norm"], "modality_uncertainty": modal["modality_uncertainty"],
            "speaker_state_memory": memory, "relation_grid": relation_grid, **utilities,
            # ---- v5: expose the measured counterfactual head + redundancy views
            "counterfactual_utility": modal["cf_predicted_utility"],
            "cf_predicted_utility": modal["cf_predicted_utility"],
            "cf_correction": modal["cf_correction"],
            "modality_sign": modal["modality_sign"],
            "shared_view": modal["shared_view"],
            "private_view": modal["private_view"],
            "shared_ratio": modal["shared_ratio"],
            # v7 current-as-candidate signals
            "current_candidate_prediction_norm": all_candidates[:, :1],
            "all_candidate_prediction_norm": all_candidates,
            "all_candidate_mask": all_mask,
            "current_cf_predicted_utility": cur_cf,
            "current_modality_weights": cur_weights,
            "unimodal_prediction_norm": unimodal_pred,
            "modality_mask": modality_mask,
            # Passthrough (detached) so the loss can supervise the unimodal
            # head without the labels ever entering the forward computation.
            "unimodal_labels": batch["unimodal_labels"].detach()
            if "unimodal_labels" in batch else None,
            # v6 accept-budget diagnostics
            "accept_threshold": routed.get("accept_threshold"),
            "accept_budget": routed.get("accept_budget"),
            "accept_rate": routed.get("accept_rate"),
            "risk_collapse_penalty": routed.get("risk_collapse_penalty"),
            "is_training_forward": self.training,
        }

    def export_card(self) -> dict:
        return {"model_name": "N3-MOSEI-Reg-v1", "task": self.cfg.task,
                "adapted_from": "CMER-HardSafe classification package", "trainable_parameters": self.count_trainable_parameters(),
                "target": "continuous sentiment [-3,3]", "training_target": "y/3",
                "output": "unbounded scalar u; reported prediction=3*u", "config": self.cfg.to_dict(),
                "risk_target": "candidate absolute error > current absolute error + normalized margin",
                "temporal_mechanism_requires_verified_past_history": True}
