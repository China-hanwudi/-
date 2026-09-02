"""Configuration for the active Temporal N3 v4 model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemporalN3Config:
    text_dim: int = 2048
    audio_dim: int = 1536
    video_dim: int = 768
    d_model: int = 128
    relation_rank: int = 32
    gate_hidden: int = 64
    num_classes: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if min(self.text_dim, self.audio_dim, self.video_dim, self.d_model) < 1:
            raise ValueError("feature dimensions must be positive")
        if self.relation_rank < 1 or self.gate_hidden < 1 or self.num_classes < 2:
            raise ValueError("model widths and class count must be valid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
