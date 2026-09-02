"""Trainable feature projectors for the active Temporal N3 model.

Frozen multimodal feature extraction is intentionally outside this package;
the model consumes auditable T/A/V feature tensors for current and candidates.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .config import TemporalN3Config


class _Projector(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TemporalN3Encoders(nn.Module):
    """Encode six independent current/history streams and apply masks."""

    def __init__(self, cfg: TemporalN3Config) -> None:
        super().__init__()
        self.text = _Projector(cfg.text_dim, cfg.d_model, cfg.dropout)
        self.audio = _Projector(cfg.audio_dim, cfg.d_model, cfg.dropout)
        self.video = _Projector(cfg.video_dim, cfg.d_model, cfg.dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        required = ("T_t", "A_t", "V_t", "T_h", "A_h", "V_h")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"missing streams: {missing}")
        streams = {
            "T_t": self.text(batch["T_t"]), "T_h": self.text(batch["T_h"]),
            "A_t": self.audio(batch["A_t"]), "A_h": self.audio(batch["A_h"]),
            "V_t": self.video(batch["V_t"]), "V_h": self.video(batch["V_h"]),
        }
        history_mask = batch.get("history_mask")
        if history_mask is not None:
            if history_mask.ndim != 2 or history_mask.shape[0] != streams["T_t"].shape[0]:
                raise ValueError("history_mask must be [B,K]")
            for key in ("T_h", "A_h", "V_h"):
                streams[key] = streams[key] * history_mask.to(streams[key].dtype).unsqueeze(-1)
        return streams
