"""Training losses for N3 (emotion primary + utility + VAD aux)."""

from __future__ import annotations

import math
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
    class_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    emotion = F.cross_entropy(outputs["logits"], labels, weight=class_weight)
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
    mix = outputs.get("mix_weights")
    mix_kl = torch.zeros((), device=labels.device)
    mix_peak_pen = torch.zeros((), device=labels.device)
    kl_w = float(getattr(cfg, "mix_kl_weight", 0.0) or 0.0)
    if mix is not None and kl_w > 0:
        k = mix.size(-1)
        mix_kl = (mix * (mix.clamp_min(1e-8).log() + math.log(k))).sum(dim=-1).mean()
        total = total + kl_w * mix_kl
    peak_w = float(getattr(cfg, "mix_peak_weight", 0.0) or 0.0)
    peak_cap = float(getattr(cfg, "mix_peak_cap", 0.40) or 0.40)
    if mix is not None and peak_w > 0:
        mix_peak_pen = F.relu(mix.max(dim=-1).values - peak_cap).mean()
        total = total + peak_w * mix_peak_pen
    return {
        "loss": total,
        "emotion_loss": emotion,
        "utility_loss": utility,
        "vad_loss": vad,
        "mix_kl": mix_kl,
        "mix_peak_pen": mix_peak_pen,
    }
