"""Affect-theory and 3x3 current-history relation module for CARMA-Affect.

This module is deliberately outcome-free at inference.  The forward API sees
projected text/audio/video tokens and a requested history mask, but never sees
emotion labels.  A separate fit-train-only helper maps integer labels to a
fixed valence/arousal/dominance (VAD/PAD) design coordinate for auxiliary
supervision.

The primary branch compares the three current modalities with three pooled
strict-past history modalities (nine ordered pairs).  The capacity control has
the exact same parameters but replaces the history summary with a deterministic
current-only cyclic modality view.  It consumes no history content, although a
single history-presence bit intentionally gates the residual to match branch
activation.  Neither branch claims the fixed coordinates are a person's latent
psychological state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


RELATION_MODES = (
    "primary_history_relation",
    "vad_history_only_no_history_3x3",
    "history_presence_capacity_control",
)

# Pre-registered Mehrabian-style design coordinates.  These are fixed
# coordinates for auxiliary representation learning, not dataset estimates.
_CANONICAL_VAD = {
    "neutral": (0.00, 0.00, 0.00),
    "happy": (0.81, 0.51, 0.46),
    "sad": (-0.63, -0.27, -0.33),
    "angry": (-0.51, 0.59, 0.25),
    "surprised": (0.40, 0.67, -0.13),
    "disgusted": (-0.60, 0.35, 0.11),
    "fearful": (-0.64, 0.60, -0.43),
}
_LABEL_ALIASES = {
    "neutral": "neutral",
    "happy": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "surprised": "surprised",
    "surprise": "surprised",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "fearful": "fearful",
    "fear": "fearful",
}


@dataclass(frozen=True)
class AffectRelationConfig:
    d_model: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    auxiliary_vad_weight: float = 0.1
    use_vad_features: bool = True
    mode: Literal[
        "primary_history_relation",
        "vad_history_only_no_history_3x3",
        "history_presence_capacity_control",
    ] = "primary_history_relation"

    def validate(self) -> None:
        if self.d_model < 8 or self.hidden_dim < 8:
            raise ValueError("affect-relation widths must be at least 8")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("affect-relation dropout must be in [0, 1)")
        if not 0.0 <= self.auxiliary_vad_weight <= 1.0:
            raise ValueError("auxiliary VAD weight must be in [0, 1]")
        if self.mode not in RELATION_MODES:
            raise ValueError("unknown affect-relation mode")
        if type(self.use_vad_features) is not bool:
            raise TypeError("use_vad_features must be boolean")
        if self.use_vad_features and self.auxiliary_vad_weight <= 0.0:
            raise ValueError("VAD features require positive fit-train auxiliary weight")
        if not self.use_vad_features and self.auxiliary_vad_weight != 0.0:
            raise ValueError("no-VAD mode must disable auxiliary VAD supervision")
        if not self.use_vad_features and self.mode != "primary_history_relation":
            raise ValueError("only the primary history branch defines a no-VAD ablation")


@dataclass(frozen=True)
class AffectRelationOutput:
    relation_residual: Tensor
    utterance_vad: Tensor
    query_vad: Tensor
    history_vad: Tensor
    relation_features: Tensor
    effective_history_mask: Tensor


def _require_bool(name: str, value: Tensor) -> None:
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")


def label_vad_table(
    label_order: Sequence[str], *, device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the fixed VAD table for an exact seven-label dataset order."""

    names = tuple(str(value) for value in label_order)
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError("VAD supervision requires seven unique emotion labels")
    try:
        canonical = tuple(_LABEL_ALIASES[name] for name in names)
    except KeyError as error:
        raise ValueError(f"emotion label has no frozen VAD alias: {error.args[0]}") from error
    if len(set(canonical)) != 7 or set(canonical) != set(_CANONICAL_VAD):
        raise ValueError("emotion label order does not cover the frozen seven states")
    return torch.tensor(
        [_CANONICAL_VAD[name] for name in canonical],
        device=device,
        dtype=dtype,
    )


def fit_train_vad_auxiliary_loss(
    predicted_vad: Tensor,
    train_labels: Tensor,
    *,
    label_order: Sequence[str],
) -> Tensor:
    """Compute fit-train-only VAD MSE; callers must not pass held-out labels."""

    if predicted_vad.ndim != 2 or predicted_vad.shape[1] != 3:
        raise ValueError("predicted VAD must have shape [fit_train_rows, 3]")
    if train_labels.dtype != torch.long or train_labels.shape != (len(predicted_vad),):
        raise TypeError("fit-train labels must be torch.long and row aligned")
    if train_labels.device != predicted_vad.device:
        raise ValueError("fit-train labels and VAD predictions must share a device")
    table = label_vad_table(
        label_order, device=predicted_vad.device, dtype=predicted_vad.dtype
    )
    if torch.any(train_labels < 0) or torch.any(train_labels >= len(table)):
        raise ValueError("fit-train label is outside the frozen VAD table")
    return F.mse_loss(predicted_vad, table[train_labels])


class CausalAffectRelation(nn.Module):
    """Capacity-matched affect state and 3x3 current-history relation branch."""

    relation_feature_dim = 33  # 9 pairs x 3 relations + query VAD + VAD delta

    def __init__(self, config: AffectRelationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vad_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 3),
            nn.Tanh(),
        )
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(self.relation_feature_dim),
            nn.Linear(self.relation_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.d_model),
        )
        self.residual_scale = nn.Parameter(torch.full((config.d_model,), 1.0e-3))

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _strict_past_mask(
        history_mask: Tensor,
        valid_mask: Tensor,
        turn_ids: Tensor,
        query_indices: Tensor,
    ) -> Tensor:
        batch, length = history_mask.shape
        positions = torch.arange(length, device=history_mask.device).expand(batch, length)
        query_turn = turn_ids.gather(1, query_indices[:, None])
        strict_past = (turn_ids < query_turn) | (
            (turn_ids == query_turn) & (positions < query_indices[:, None])
        )
        return history_mask & valid_mask & strict_past

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        if mask.shape != values.shape[:3]:
            raise ValueError("relation mean mask must be [batch, length, modality]")
        weights = mask.to(dtype=values.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights[:, :, :, None]).sum(dim=1) / denominator[:, :, None]

    def forward(
        self,
        projected_modalities: Tensor,
        *,
        valid_mask: Tensor,
        history_mask: Tensor,
        turn_ids: Tensor,
        query_indices: Tensor,
        modality_mask: Tensor | None = None,
    ) -> AffectRelationOutput:
        """Build a relation residual without accepting labels or outcome targets."""

        if projected_modalities.ndim != 4 or projected_modalities.shape[2:] != (
            3,
            self.config.d_model,
        ):
            raise ValueError(
                "projected modalities must have shape [batch, length, 3, d_model]"
            )
        batch, length = projected_modalities.shape[:2]
        for name, value in (
            ("valid_mask", valid_mask),
            ("history_mask", history_mask),
        ):
            if value.shape != (batch, length):
                raise ValueError(f"{name} must have shape [batch, length]")
            _require_bool(name, value)
        if turn_ids.dtype != torch.long or turn_ids.shape != (batch, length):
            raise TypeError("turn_ids must be torch.long with shape [batch, length]")
        if query_indices.dtype != torch.long or query_indices.shape != (batch,):
            raise TypeError("query_indices must be torch.long with shape [batch]")
        if torch.any(query_indices < 0) or torch.any(query_indices >= length):
            raise ValueError("query index is outside the relation sequence")
        if any(
            value.device != projected_modalities.device
            for value in (valid_mask, history_mask, turn_ids, query_indices)
        ):
            raise ValueError("relation inputs must share one device")
        if modality_mask is None:
            modality_mask = torch.ones(
                (batch, length, 3),
                dtype=torch.bool,
                device=projected_modalities.device,
            )
        elif modality_mask.shape != (batch, length, 3) or modality_mask.dtype is not torch.bool:
            raise TypeError("modality_mask must be boolean [batch, length, 3]")
        if modality_mask.device != projected_modalities.device:
            raise ValueError("modality_mask must share the relation device")

        effective = self._strict_past_mask(
            history_mask, valid_mask, turn_ids, query_indices
        )
        row = torch.arange(batch, device=projected_modalities.device)
        query_modalities = projected_modalities[row, query_indices] * modality_mask[
            row, query_indices
        ].to(dtype=projected_modalities.dtype)[:, :, None]
        history_modalities = self._masked_mean(
            projected_modalities,
            effective[:, :, None] & modality_mask,
        )
        has_history = effective.any(dim=1)

        if self.config.mode in {
            "vad_history_only_no_history_3x3",
            "history_presence_capacity_control",
        }:
            # Same parameters and tensor widths, but no history content.  The
            # cyclic view avoids an all-zero pair vector while retaining only
            # current-utterance information.  The final residual still uses
            # has_history as a one-bit activation match to the primary branch.
            relation_history = torch.roll(query_modalities, shifts=1, dims=1)
        else:
            relation_history = history_modalities

        query_available = modality_mask[row, query_indices]
        history_available = (effective[:, :, None] & modality_mask).any(dim=1)
        if self.config.mode in {
            "vad_history_only_no_history_3x3",
            "history_presence_capacity_control",
        }:
            relation_history_available = torch.roll(
                query_available, shifts=1, dims=1
            )
        else:
            relation_history_available = history_available
        pair_available = (
            query_available[:, :, None]
            & relation_history_available[:, None, :]
        )

        query_expanded = query_modalities[:, :, None, :].expand(-1, -1, 3, -1)
        history_expanded = relation_history[:, None, :, :].expand(-1, 3, -1, -1)
        query_normalized = F.normalize(query_expanded, dim=-1, eps=1.0e-8)
        history_normalized = F.normalize(history_expanded, dim=-1, eps=1.0e-8)
        cosine = (query_normalized * history_normalized).sum(dim=-1)
        l2 = torch.sqrt(
            ((query_normalized - history_normalized) ** 2).sum(dim=-1).clamp_min(1.0e-12)
        )
        signed_delta = (query_expanded - history_expanded).mean(dim=-1)
        pair_features = torch.stack((cosine, l2, signed_delta), dim=-1)
        pair_features = pair_features.masked_fill(~pair_available[:, :, :, None], 0.0)
        pair_features = pair_features.reshape(batch, 27)

        available = modality_mask.to(dtype=projected_modalities.dtype)
        fused = (projected_modalities * available[:, :, :, None]).sum(dim=2) / (
            available.sum(dim=2, keepdim=True).clamp_min(1.0)
        )
        utterance_vad = self.vad_head(fused)
        query_vad = utterance_vad[row, query_indices]
        history_vad = (
            utterance_vad * effective.to(dtype=utterance_vad.dtype)[:, :, None]
        ).sum(dim=1) / effective.sum(dim=1, keepdim=True).clamp_min(1).to(
            dtype=utterance_vad.dtype
        )
        if self.config.mode == "history_presence_capacity_control":
            relation_history_vad = torch.roll(query_vad, shifts=1, dims=-1)
        else:
            relation_history_vad = history_vad
        if self.config.use_vad_features:
            vad_features = (query_vad, query_vad - relation_history_vad)
        else:
            zero_vad = torch.zeros_like(query_vad)
            vad_features = (zero_vad, zero_vad)
        relation_features = torch.cat(
            (pair_features, *vad_features), dim=-1
        )
        residual = self.relation_encoder(relation_features) * self.residual_scale
        residual = residual * has_history.to(dtype=residual.dtype)[:, None]
        return AffectRelationOutput(
            relation_residual=residual,
            utterance_vad=utterance_vad,
            query_vad=query_vad,
            history_vad=history_vad,
            relation_features=relation_features,
            effective_history_mask=effective,
        )
