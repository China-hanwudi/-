"""Utility targets (plan 00 S4.4): per-sample CE improvement of an action.

d_i(S) = CE(z_B(i), y_i) - CE(z_S(i), y_i)

This is the realised loss change of EXECUTING action S on the deployed
path.  It is NOT a causal effect, NOT per-sample WF1, NOT an upper bound.
Labels are used ONLY to build training targets / development diagnostics;
they are never an inference input.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def ce_per_sample(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y, reduction="none")


def utility_target(z_B: torch.Tensor, z_S: torch.Tensor,
                   y: torch.Tensor) -> torch.Tensor:
    return ce_per_sample(z_B, y) - ce_per_sample(z_S, y)
