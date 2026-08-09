"""Trainable N3 configuration aligned with the repo freeze protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LABELS = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

ALLOWED_TEXT_TOWERS = frozenset({"composer_n3", "xlm_roberta_large"})
DEFAULT_HF_TEXT_MODEL = "FacebookAI/xlm-roberta-large"


@dataclass
class N3TrainConfig:
    """Hyperparameters for :class:`~n3_affect.model.N3EmotionModel`."""

    text_dim: int = 256
    audio_dim: int = 1536
    video_dim: int = 768
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 384
    num_classes: int = 7
    dropout: float = 0.1
    parameter_budget: int = 2_000_000
    relation_rank: int = 32
    gate_hidden: int = 64
    text_tower: str = "composer_n3"  # or "xlm_roberta_large"
    hf_text_model_id: str = DEFAULT_HF_TEXT_MODEL
    emotion_label_order: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LABELS)
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    max_epochs: int = 5
    grad_clip: float = 1.0
    emotion_loss_weight: float = 1.0
    utility_loss_weight: float = 0.2
    vad_loss_weight: float = 0.1
    seed: int = 17

    def validate(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_classes != 7:
            raise ValueError("N3 train config currently requires 7 emotion classes")
        if self.text_tower not in ALLOWED_TEXT_TOWERS:
            raise ValueError(f"unknown text_tower: {self.text_tower}")
        if len(self.emotion_label_order) != self.num_classes:
            raise ValueError("emotion_label_order length must equal num_classes")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["emotion_label_order"] = list(self.emotion_label_order)
        return data

    @classmethod
    def from_json(cls, path: str | Path) -> "N3TrainConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        arch = raw.get("architecture", {})
        dims = raw.get("input_dims", {})
        train = raw.get("training", {})
        weights = train.get("loss_weights", {})
        llm = raw.get("builtin_llm", {})
        optional = llm.get("optional_text_tower", {})
        default_mode = str(llm.get("default_mode", "composer_n3"))
        if default_mode in ALLOWED_TEXT_TOWERS:
            text_tower = default_mode
        elif optional.get("tower_key") in ALLOWED_TEXT_TOWERS:
            text_tower = str(optional["tower_key"])
        else:
            text_tower = "composer_n3"
        return cls(
            text_dim=int(dims.get("text_dim", 256)),
            audio_dim=int(dims.get("audio_dim", 1536)),
            video_dim=int(dims.get("video_dim", 768)),
            d_model=int(arch.get("d_model", 128)),
            num_heads=int(arch.get("num_heads", 4)),
            num_layers=int(arch.get("num_layers", 4)),
            ffn_dim=int(arch.get("ffn_dim", 384)),
            num_classes=int(arch.get("num_classes", 7)),
            parameter_budget=int(arch.get("parameter_budget", 2_000_000)),
            text_tower=text_tower,
            hf_text_model_id=str(optional.get("model_id", DEFAULT_HF_TEXT_MODEL)),
            emotion_label_order=tuple(raw.get("emotion_label_order", DEFAULT_LABELS)),
            lr=float(train.get("lr", 3e-4)),
            weight_decay=float(train.get("weight_decay", 1e-2)),
            batch_size=int(train.get("batch_size", 8)),
            max_epochs=int(train.get("max_epochs", 5)),
            grad_clip=float(train.get("grad_clip", 1.0)),
            emotion_loss_weight=float(weights.get("emotion", 1.0)),
            utility_loss_weight=float(weights.get("utility", 0.2)),
            vad_loss_weight=float(weights.get("vad", 0.1)),
            seed=int((train.get("seeds") or [17])[0]),
        )
