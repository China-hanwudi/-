"""Compact causal multimodal backbone for CARMA-Affect experiments.

This module intentionally starts *after* feature extraction.  Text inputs are
fold-local SVD features and audio/video inputs are frozen WavLM/DINO features;
no pretrained model is downloaded or fine-tuned here.

The public contract has two safety properties:

1. every attention layer uses turn-aware causal masking, so a current
   prediction cannot receive information from a future utterance, even through
   an intermediate history token; and
2. ``history_mask`` is a hard key-visibility mask, so values outside a chosen
   subset cannot affect the prediction.

``forward_contexts`` evaluates arbitrary context subsets in one batch.  It is
therefore suitable for current/S/S+h/T/T-h utility calculations without five
separate model calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .causal_affect_relation import (
    AffectRelationConfig,
    CausalAffectRelation,
    label_vad_table,
)


UTILITY_CONTEXT_ORDER = ("current", "S", "S_plus_h", "T", "T_minus_h")


@dataclass(frozen=True)
class CausalBackboneConfig:
    """Dimensions and capacity limits for :class:`CausalMultimodalBackbone`."""

    text_dim: int = 256
    audio_dim: int = 1536
    video_dim: int = 768
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 384
    num_speakers: int = 64
    max_turns: int = 2048
    max_relative_turn: int = 128
    num_classes: int = 7
    dropout: float = 0.10
    layer_scale_init: float = 1.0e-3
    parameter_limit: int = 2_000_000
    affect_relation_mode: str = "disabled"
    affect_relation_hidden_dim: int = 128
    affect_relation_use_vad_features: bool = False
    auxiliary_vad_weight: float = 0.0
    emotion_label_order: tuple[str, ...] = ()

    def validate(self) -> None:
        positive = {
            "text_dim": self.text_dim,
            "audio_dim": self.audio_dim,
            "video_dim": self.video_dim,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "num_speakers": self.num_speakers,
            "max_turns": self.max_turns,
            "max_relative_turn": self.max_relative_turn,
            "num_classes": self.num_classes,
            "parameter_limit": self.parameter_limit,
            "affect_relation_hidden_dim": self.affect_relation_hidden_dim,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration fields must be positive: {invalid}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_scale_init <= 0.0:
            raise ValueError("layer_scale_init must be positive")
        if self.num_classes != 7:
            raise ValueError("CARMA-Affect currently requires exactly seven classes")
        if self.affect_relation_mode not in {
            "disabled",
            "primary_history_relation",
            "vad_history_only_no_history_3x3",
            "history_presence_capacity_control",
        }:
            raise ValueError("unknown affect-relation mode")
        if type(self.affect_relation_use_vad_features) is not bool:
            raise TypeError("affect_relation_use_vad_features must be boolean")
        if not 0.0 <= self.auxiliary_vad_weight <= 1.0:
            raise ValueError("auxiliary VAD weight must be in [0, 1]")
        if self.affect_relation_mode == "disabled":
            if (
                self.affect_relation_use_vad_features
                or self.auxiliary_vad_weight != 0.0
                or self.emotion_label_order
            ):
                raise ValueError(
                    "disabled affect relation cannot carry VAD supervision fields"
                )
        elif self.affect_relation_use_vad_features:
            if self.auxiliary_vad_weight <= 0.0:
                raise ValueError("VAD features require positive fit-train auxiliary weight")
            label_vad_table(self.emotion_label_order)
        else:
            if self.auxiliary_vad_weight != 0.0 or self.emotion_label_order:
                raise ValueError(
                    "no-VAD ablation must not carry VAD supervision fields"
                )
            if self.affect_relation_mode != "primary_history_relation":
                raise ValueError(
                    "capacity and no-3x3 controls must retain the frozen VAD branch"
                )

    def validate_dataset_label_order(self, label_order: tuple[str, ...]) -> None:
        """Bind fit-train VAD targets to the verified dataset class order."""

        observed = tuple(str(value) for value in label_order)
        if len(observed) != self.num_classes or len(set(observed)) != self.num_classes:
            raise ValueError("dataset label order must contain seven unique classes")
        if (
            self.auxiliary_vad_weight > 0.0
            and tuple(self.emotion_label_order) != observed
        ):
            raise ValueError(
                "VAD supervision label order differs from the verified dataset order"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CausalBackboneConfig":
        """Load either the published nested JSON schema or a flat mapping."""

        model_raw = payload.get("model", payload)
        if not isinstance(model_raw, Mapping):
            raise ValueError("model configuration must be a mapping")
        model = dict(model_raw)

        feature_contract = payload.get("feature_contract")
        if isinstance(feature_contract, Mapping):
            for field, modality in (
                ("text_dim", "text"),
                ("audio_dim", "audio"),
                ("video_dim", "video"),
            ):
                entry = feature_contract.get(modality)
                if isinstance(entry, Mapping) and "dim" in entry:
                    model[field] = entry["dim"]

        allowed = set(cls.__dataclass_fields__)
        selected = {key: value for key, value in model.items() if key in allowed}
        if "emotion_label_order" in selected:
            raw_order = selected["emotion_label_order"]
            if not isinstance(raw_order, (list, tuple)):
                raise ValueError("emotion_label_order must be a list of seven labels")
            selected["emotion_label_order"] = tuple(str(value) for value in raw_order)
        config = cls(**selected)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "CausalBackboneConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("backbone JSON root must be a mapping")
        return cls.from_mapping(payload)


@dataclass
class BackboneOutput:
    """Single-context predictions and the actually visible history subset."""

    logits: Tensor
    probabilities: Tensor
    effective_history_mask: Tensor
    query_vad: Tensor | None = None


@dataclass
class ContextBackboneOutput:
    """Batched-context predictions with shapes ``[batch, contexts, ...]``."""

    logits: Tensor
    probabilities: Tensor
    effective_history_mask: Tensor
    query_vad: Tensor | None = None


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _require_bool(name: str, value: Tensor) -> None:
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")


def sample_history_subset(
    history_mask: Tensor,
    *,
    valid_mask: Tensor,
    turn_ids: Tensor,
    query_indices: Tensor,
    drop_probability: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly remove eligible past items from an arbitrary base subset.

    The returned mask never contains padding, the query itself, or a future
    turn.  Supplying a seeded ``torch.Generator`` makes the sample reproducible.
    This function is public so an experiment can materialize and audit the
    exact subsets used during training.
    """

    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability must be in [0, 1]")
    _require_bool("history_mask", history_mask)
    _require_bool("valid_mask", valid_mask)
    if history_mask.shape != valid_mask.shape or history_mask.shape != turn_ids.shape:
        raise ValueError("history_mask, valid_mask, and turn_ids must have the same [B, L] shape")
    if query_indices.ndim != 1 or query_indices.shape[0] != history_mask.shape[0]:
        raise ValueError("query_indices must have shape [B]")

    batch, length = history_mask.shape
    if torch.any(query_indices < 0) or torch.any(query_indices >= length):
        raise ValueError("query index is outside the sequence")
    positions = torch.arange(length, device=history_mask.device).expand(batch, length)
    query_turn = turn_ids.gather(1, query_indices[:, None])
    is_strict_past = (turn_ids < query_turn) | (
        (turn_ids == query_turn) & (positions < query_indices[:, None])
    )
    eligible = history_mask & valid_mask & is_strict_past
    if drop_probability == 0.0:
        return eligible
    if drop_probability == 1.0:
        return torch.zeros_like(eligible)
    random_values = torch.rand(
        eligible.shape,
        device=eligible.device,
        generator=generator,
    )
    return eligible & (random_values >= drop_probability)


def pack_utility_context_masks(
    *,
    s_mask: Tensor,
    s_plus_h_mask: Tensor,
    t_mask: Tensor,
    t_minus_h_mask: Tensor,
) -> Tensor:
    """Pack current/S/S+h/T/T-h masks into one ``[B, 5, L]`` tensor."""

    masks = (s_mask, s_plus_h_mask, t_mask, t_minus_h_mask)
    for mask in masks:
        _require_bool("utility context mask", mask)
        if mask.shape != s_mask.shape:
            raise ValueError("all utility context masks must have the same shape")
    current = torch.zeros_like(s_mask)
    return torch.stack((current, *masks), dim=1)


class _FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        return self.dropout(F.gelu(self.linear(self.norm(features))))


class _TurnAwareCausalAttention(nn.Module):
    """Batch-specific causal attention ordered by (turn id, sequence index)."""

    def __init__(self, config: CausalBackboneConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.max_relative_turn = config.max_relative_turn
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.relative_turn_bias = nn.Embedding(config.max_relative_turn + 1, config.num_heads)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor, allowed: Tensor, turn_ids: Tensor) -> Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        relative_turn = (turn_ids[:, :, None] - turn_ids[:, None, :]).clamp(
            min=0,
            max=self.max_relative_turn,
        )
        bias = self.relative_turn_bias(relative_turn).permute(0, 3, 1, 2)
        scores = scores + bias.to(dtype=scores.dtype)
        scores = scores.masked_fill(~allowed[:, None, :, :], torch.finfo(scores.dtype).min)

        # Float32 softmax is deliberate: it keeps fp16/bfloat16 autocast stable.
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, width)
        return self.output_dropout(self.output(attended))


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        value, gate = self.input(tokens).chunk(2, dim=-1)
        return self.dropout(self.output(value * F.silu(gate)))


class _CausalBlock(nn.Module):
    def __init__(self, config: CausalBackboneConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = _TurnAwareCausalAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _SwiGLU(config.d_model, config.ffn_dim, config.dropout)
        self.attention_scale = nn.Parameter(
            torch.full((config.d_model,), config.layer_scale_init)
        )
        self.ffn_scale = nn.Parameter(
            torch.full((config.d_model,), config.layer_scale_init)
        )

    def forward(
        self,
        tokens: Tensor,
        *,
        allowed: Tensor,
        turn_ids: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        tokens = tokens + self.attention_scale * self.attention(
            self.attention_norm(tokens), allowed, turn_ids
        )
        tokens = tokens + self.ffn_scale * self.ffn(self.ffn_norm(tokens))
        return tokens.masked_fill(~valid_mask[:, :, None], 0.0)


class CausalMultimodalBackbone(nn.Module):
    """A sub-2M-parameter, subset-masked multimodal Transformer.

    Inputs use ``[batch, length, feature]`` layout. ``query_indices`` identifies
    the current utterance in each sequence, while ``history_mask`` selects any
    desired candidate subset.  The hard causal contract is applied after that
    selection, so accidentally marking future entries cannot expose them.
    """

    def __init__(self, config: CausalBackboneConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.text_projector = _FeatureProjector(config.text_dim, config.d_model, config.dropout)
        self.audio_projector = _FeatureProjector(config.audio_dim, config.d_model, config.dropout)
        self.video_projector = _FeatureProjector(config.video_dim, config.d_model, config.dropout)
        self.modality_embeddings = nn.Parameter(torch.empty(3, config.d_model))
        self.modality_gate = nn.Sequential(
            nn.LayerNorm(3 * config.d_model),
            nn.Linear(3 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 3),
        )
        self.speaker_embedding = nn.Embedding(config.num_speakers, config.d_model)
        self.turn_embedding = nn.Embedding(config.max_turns, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(_CausalBlock(config) for _ in range(config.num_layers))
        self.output_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        if config.affect_relation_mode == "disabled":
            self.affect_relation: CausalAffectRelation | None = None
            self.register_buffer(
                "vad_label_table",
                torch.empty(0, 3, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.affect_relation = CausalAffectRelation(
                AffectRelationConfig(
                    d_model=config.d_model,
                    hidden_dim=config.affect_relation_hidden_dim,
                    dropout=config.dropout,
                    auxiliary_vad_weight=config.auxiliary_vad_weight,
                    use_vad_features=config.affect_relation_use_vad_features,
                    mode=config.affect_relation_mode,  # type: ignore[arg-type]
                )
            )
            if config.auxiliary_vad_weight > 0.0:
                self.register_buffer(
                    "vad_label_table",
                    label_vad_table(config.emotion_label_order),
                    persistent=True,
                )
            else:
                self.register_buffer(
                    "vad_label_table",
                    torch.empty(0, 3, dtype=torch.float32),
                    persistent=False,
                )
        self.reset_parameters()

        parameter_count = self.parameter_count()
        if parameter_count >= config.parameter_limit:
            raise ValueError(
                f"backbone has {parameter_count:,} trainable parameters; "
                f"strict limit is < {config.parameter_limit:,}"
            )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.speaker_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.turn_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return count_trainable_parameters(self)

    def _validate_inputs(
        self,
        text_features: Tensor,
        audio_features: Tensor,
        video_features: Tensor,
        speaker_ids: Tensor,
        turn_ids: Tensor,
        valid_mask: Tensor,
        history_mask: Tensor,
        query_indices: Tensor,
        modality_mask: Tensor | None,
    ) -> Tensor:
        expected_prefix = text_features.shape[:2]
        if text_features.ndim != 3 or text_features.shape[-1] != self.config.text_dim:
            raise ValueError(f"text_features must have shape [B, L, {self.config.text_dim}]")
        for name, features, width in (
            ("audio_features", audio_features, self.config.audio_dim),
            ("video_features", video_features, self.config.video_dim),
        ):
            if features.ndim != 3 or features.shape[:2] != expected_prefix or features.shape[-1] != width:
                raise ValueError(f"{name} must have shape [B, L, {width}]")
        for name, values in (
            ("speaker_ids", speaker_ids),
            ("turn_ids", turn_ids),
            ("valid_mask", valid_mask),
            ("history_mask", history_mask),
        ):
            if values.shape != expected_prefix:
                raise ValueError(f"{name} must have shape [B, L]")
        _require_bool("valid_mask", valid_mask)
        _require_bool("history_mask", history_mask)
        if speaker_ids.dtype != torch.long or turn_ids.dtype != torch.long:
            raise TypeError("speaker_ids and turn_ids must have dtype torch.long")
        if query_indices.dtype != torch.long or query_indices.shape != (expected_prefix[0],):
            raise TypeError("query_indices must be torch.long with shape [B]")

        tensors = (
            audio_features,
            video_features,
            speaker_ids,
            turn_ids,
            valid_mask,
            history_mask,
            query_indices,
        )
        if any(value.device != text_features.device for value in tensors):
            raise ValueError("all inputs must be on the same device")
        if torch.any(query_indices < 0) or torch.any(query_indices >= expected_prefix[1]):
            raise ValueError("query index is outside the sequence")
        query_is_valid = valid_mask.gather(1, query_indices[:, None]).squeeze(1)
        if not torch.all(query_is_valid):
            raise ValueError("every query must identify a valid token")
        if torch.any(speaker_ids[valid_mask] < 0) or torch.any(
            speaker_ids[valid_mask] >= self.config.num_speakers
        ):
            raise ValueError("valid speaker id is outside the configured vocabulary")
        if torch.any(turn_ids[valid_mask] < 0) or torch.any(turn_ids[valid_mask] >= self.config.max_turns):
            raise ValueError("valid turn id is outside the configured embedding range")

        if modality_mask is None:
            modality_mask = torch.ones(
                (*expected_prefix, 3),
                dtype=torch.bool,
                device=text_features.device,
            )
        else:
            if modality_mask.shape != (*expected_prefix, 3):
                raise ValueError("modality_mask must have shape [B, L, 3]")
            _require_bool("modality_mask", modality_mask)
            if modality_mask.device != text_features.device:
                raise ValueError("modality_mask must be on the same device as the features")
        if not torch.all(modality_mask[valid_mask].any(dim=-1)):
            raise ValueError("each valid utterance must expose at least one modality")
        return modality_mask

    @staticmethod
    def _causal_allowed_mask(
        *,
        turn_ids: Tensor,
        valid_mask: Tensor,
        effective_history_mask: Tensor,
        query_indices: Tensor,
    ) -> Tensor:
        batch, length = turn_ids.shape
        positions = torch.arange(length, device=turn_ids.device)
        query_turn = turn_ids[:, :, None]
        key_turn = turn_ids[:, None, :]
        query_position = positions[None, :, None]
        key_position = positions[None, None, :]
        causal = (key_turn < query_turn) | (
            (key_turn == query_turn) & (key_position <= query_position)
        )

        visible_keys = effective_history_mask.clone()
        visible_keys.scatter_(1, query_indices[:, None], True)
        allowed = causal & visible_keys[:, None, :] & valid_mask[:, None, :]

        # A private diagonal for every row prevents all-masked softmax rows.
        # It cannot leak into the query because non-visible tokens remain
        # unavailable as keys to every other row.
        diagonal = torch.eye(length, dtype=torch.bool, device=turn_ids.device)
        return allowed | diagonal[None, :, :]

    def _fuse_modalities(
        self,
        text_features: Tensor,
        audio_features: Tensor,
        video_features: Tensor,
        modality_mask: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        projected = torch.stack(
            (
                self.text_projector(text_features),
                self.audio_projector(audio_features),
                self.video_projector(video_features),
            ),
            dim=2,
        )
        projected = projected + self.modality_embeddings[None, None, :, :]
        masked_projected = projected.masked_fill(~modality_mask[:, :, :, None], 0.0)
        gate_logits = self.modality_gate(masked_projected.flatten(start_dim=2))

        # Padding may have no available modality. Give it a harmless synthetic
        # text slot; its final token is zeroed immediately afterwards.
        safe_mask = modality_mask.clone()
        safe_mask[:, :, 0] |= ~valid_mask
        gate_logits = gate_logits.masked_fill(~safe_mask, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits.float(), dim=-1).to(gate_logits.dtype)
        fused = (weights[:, :, :, None] * masked_projected).sum(dim=2)
        return (
            fused.masked_fill(~valid_mask[:, :, None], 0.0),
            masked_projected.masked_fill(~valid_mask[:, :, None, None], 0.0),
        )

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        video_features: Tensor,
        speaker_ids: Tensor,
        turn_ids: Tensor,
        valid_mask: Tensor,
        history_mask: Tensor,
        query_indices: Tensor,
        modality_mask: Tensor | None = None,
        subset_dropout_p: float = 0.0,
        subset_generator: torch.Generator | None = None,
    ) -> BackboneOutput:
        modality_mask = self._validate_inputs(
            text_features,
            audio_features,
            video_features,
            speaker_ids,
            turn_ids,
            valid_mask,
            history_mask,
            query_indices,
            modality_mask,
        )
        if not 0.0 <= subset_dropout_p <= 1.0:
            raise ValueError("subset_dropout_p must be in [0, 1]")
        effective_history = sample_history_subset(
            history_mask,
            valid_mask=valid_mask,
            turn_ids=turn_ids,
            query_indices=query_indices,
            drop_probability=subset_dropout_p if self.training else 0.0,
            generator=subset_generator,
        )
        allowed = self._causal_allowed_mask(
            turn_ids=turn_ids,
            valid_mask=valid_mask,
            effective_history_mask=effective_history,
            query_indices=query_indices,
        )

        tokens, projected_modalities = self._fuse_modalities(
            text_features,
            audio_features,
            video_features,
            modality_mask,
            valid_mask,
        )
        relation_output = None
        if self.affect_relation is not None:
            relation_output = self.affect_relation(
                projected_modalities,
                valid_mask=valid_mask,
                history_mask=effective_history,
                turn_ids=turn_ids,
                query_indices=query_indices,
                modality_mask=modality_mask,
            )
            if not torch.equal(
                relation_output.effective_history_mask, effective_history
            ):
                raise AssertionError("affect relation changed the causal history mask")
            batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
            tokens = tokens.clone()
            tokens[batch_indices, query_indices] = (
                tokens[batch_indices, query_indices]
                + relation_output.relation_residual
            )
        # Padding rows may use arbitrary sentinels; clamp them before lookup.
        # Valid rows were range-checked above, so this never alters real data.
        safe_speaker_ids = speaker_ids.clamp(min=0, max=self.config.num_speakers - 1)
        safe_turn_ids = turn_ids.clamp(min=0, max=self.config.max_turns - 1)
        tokens = tokens + self.speaker_embedding(safe_speaker_ids)
        tokens = tokens + self.turn_embedding(safe_turn_ids)
        tokens = self.input_dropout(self.input_norm(tokens))
        tokens = tokens.masked_fill(~valid_mask[:, :, None], 0.0)
        for block in self.blocks:
            tokens = block(
                tokens,
                allowed=allowed,
                turn_ids=turn_ids,
                valid_mask=valid_mask,
            )

        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        current = self.output_norm(tokens[batch_indices, query_indices])
        logits = self.classifier(current)
        probabilities = torch.softmax(logits.float(), dim=-1)
        return BackboneOutput(
            logits,
            probabilities,
            effective_history,
            (
                None
                if relation_output is None or self.config.auxiliary_vad_weight == 0.0
                else relation_output.query_vad
            ),
        )

    def forward_contexts(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        video_features: Tensor,
        speaker_ids: Tensor,
        turn_ids: Tensor,
        valid_mask: Tensor,
        context_masks: Tensor,
        query_indices: Tensor,
        modality_mask: Tensor | None = None,
        subset_dropout_p: float = 0.0,
        subset_generator: torch.Generator | None = None,
    ) -> ContextBackboneOutput:
        """Evaluate ``K`` history subsets together from shared base features.

        ``context_masks`` has shape ``[B, K, L]``.  The returned logits and
        probabilities have shape ``[B, K, 7]``.  ``K`` is arbitrary, so both a
        four-context diagnostic and the five utility contexts are supported.
        """

        if context_masks.ndim != 3 or context_masks.shape[0] != text_features.shape[0]:
            raise ValueError("context_masks must have shape [B, K, L]")
        if context_masks.shape[2] != text_features.shape[1]:
            raise ValueError("context mask length must match feature sequence length")
        _require_bool("context_masks", context_masks)
        if context_masks.device != text_features.device:
            raise ValueError("context_masks must be on the same device as features")
        batch, contexts, length = context_masks.shape
        if contexts < 1:
            raise ValueError("at least one context is required")

        def repeat(value: Tensor) -> Tensor:
            return value[:, None].expand(batch, contexts, *value.shape[1:]).reshape(
                batch * contexts, *value.shape[1:]
            )

        repeated_modality_mask = repeat(modality_mask) if modality_mask is not None else None
        output = self.forward(
            text_features=repeat(text_features),
            audio_features=repeat(audio_features),
            video_features=repeat(video_features),
            speaker_ids=repeat(speaker_ids),
            turn_ids=repeat(turn_ids),
            valid_mask=repeat(valid_mask),
            history_mask=context_masks.reshape(batch * contexts, length),
            query_indices=repeat(query_indices).reshape(batch * contexts),
            modality_mask=repeated_modality_mask,
            subset_dropout_p=subset_dropout_p,
            subset_generator=subset_generator,
        )
        return ContextBackboneOutput(
            logits=output.logits.reshape(batch, contexts, self.config.num_classes),
            probabilities=output.probabilities.reshape(batch, contexts, self.config.num_classes),
            effective_history_mask=output.effective_history_mask.reshape(batch, contexts, length),
            query_vad=(
                None
                if output.query_vad is None
                else output.query_vad.reshape(batch, contexts, 3)
            ),
        )
