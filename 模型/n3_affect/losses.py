"""Training losses for N3 (emotion primary + utility + VAD aux)."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import N3TrainConfig


def n3_total_loss(
    outputs: Mapping[str, Tensor],
    labels: Tensor,
    cfg: N3TrainConfig,
    utility_targets: Mapping[str, Tensor] | None = None,
    vad_targets: Tensor | None = None,
) -> dict[str, Tensor]:
    emotion = F.cross_entropy(outputs["logits"], labels)
    total = cfg.emotion_loss_weight * emotion
    utility = torch.zeros((), device=labels.device)
    if utility_targets is not None:
        pieces = []
        for key in ("U_T", "U_A", "U_V", "U_joint"):
            if key in utility_targets:
                pieces.append(F.mse_loss(outputs[key], utility_targets[key]))
        if "U_cross" in utility_targets:
            pieces.append(F.mse_loss(outputs["U_cross"], utility_targets["U_cross"]))
        if pieces:
            utility = torch.stack(pieces).mean()
            total = total + cfg.utility_loss_weight * utility
    vad = torch.zeros((), device=labels.device)
    if vad_targets is not None:
        vad = F.mse_loss(outputs["vad"], vad_targets)
        total = total + cfg.vad_loss_weight * vad
    return {"loss": total, "emotion_loss": emotion, "utility_loss": utility, "vad_loss": vad}
