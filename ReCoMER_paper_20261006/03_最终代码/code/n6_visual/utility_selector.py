"""Utility selector (plan 00 S4.4), first version.

Small Huber-regression head over: per-modality probability states
(softmax, entropy, top1-top2 margin -- class-semantic and ORDER-FIXED),
frozen visual window features and quality proxies, plus the candidate
action mask.  NO labels, correctness, sample/group IDs or corruption tags
in the inputs.  First version fits a continuous d_hat per action; no
ranking loss / IB / contrastive extras.

Deployment rule: empty action scores fixed 0; choose the legal action with
the highest predicted gain iff it exceeds the train-side-locked threshold
tau (initial 0), else fall back to the base.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .action_space import EMPTY, N_ACTIONS, action_mask, legal_actions


class UtilitySelector(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        # tau rides in state_dict (K25-08): a non-default threshold must
        # survive save/load like every other deploy-relevant field.
        self.register_buffer("tau", torch.tensor(0.0))

    def set_tau(self, v: float) -> None:
        self.tau.fill_(float(v))

    def fit_stats(self, X: np.ndarray, w: np.ndarray) -> None:
        w = w / w.sum()
        mu = (X * w[:, None]).sum(0)
        var = (((X - mu) ** 2) * w[:, None]).sum(0)
        self.mu.copy_(torch.tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.tensor(np.sqrt(var), dtype=torch.float32)
                        .clamp_min(1e-6))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)

    @torch.no_grad()
    def choose(self, x_row: np.ndarray, window_valid: np.ndarray) -> int:
        """One row: return the chosen action index (EMPTY fallback)."""
        la = legal_actions(window_valid)
        if not la[1:].any():
            return EMPTY
        best_a, best_s = EMPTY, 0.0
        for a in range(1, N_ACTIONS):
            if not la[a]:
                continue
            x = np.concatenate([x_row, action_mask(a)]).astype(np.float32)
            s = float(self(torch.tensor(x[None])).item())
            if s > best_s + 0.0 and s > float(self.tau):
                best_a, best_s = a, s
        return best_a
