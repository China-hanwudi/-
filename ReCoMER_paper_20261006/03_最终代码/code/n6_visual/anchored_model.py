"""Anchored conditional visual correction (plan 00 S4.3).

    q   = LN(Wq . concat(hT, hA, hV))                    # [B,192]
    r_S = CrossAttention(query=q, keys=V_S, values=V_S)  # [B,192]
    e_S = GELU(Wv . r_S) * sigmoid(Wc . q)               # [B,192]
    dz  = Wout . e_S                                     # [B,C]
    z   = z_B + dz

Rules encoded:
- Wout is ZERO-initialised with bias=False (correction is identically zero
  at init).  Wv/Wc are normally initialised -- no double-zero gradient
  block (acceptance test 3).
- The base is FROZEN and permanently in eval mode: wrapping
  ``_freeze_base`` sets requires_grad=False AND neutralises .train() on the
  base submodule, so an outer model.train() can never re-enable base
  dropout (acceptance test 2).
- Empty action (no window selected) or ALL windows invalid -> return z_B
  and bypass attention entirely (no NaN from all-masked attention).
- `q_only=True` is the capacity-matched control: dz from q alone
  (no visual tokens), same parameter budget as the correction path.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

D_MODEL = 192


class _FrozenBase(nn.Module):
    """Wrap a base model: parameters frozen AND eval-mode locked."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):     # eval-locked, ignores mode
        super().train(False)
        self.base.eval()
        return self

    def forward(self, *a, **k):
        return self.base(*a, **k)


class AnchoredVisualCorrector(nn.Module):
    def __init__(self, base: nn.Module, num_classes: int,
                 d_model: int = D_MODEL, heads: int = 4, q_only: bool = False):
        super().__init__()
        self.base = _FrozenBase(base)
        self.num_classes = num_classes
        self.q_only = q_only
        self.wq = nn.Linear(3 * d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        if not q_only:
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.wv = nn.Linear(d_model, d_model)
        self.wc = nn.Linear(d_model, d_model)
        self.wout = nn.Linear(d_model, num_classes, bias=False)
        nn.init.zeros_(self.wout.weight)
        self.gelu = nn.GELU()

    def correct(self, hT, hA, hV, tokens, window_valid, action_mask=None):
        """h* [B,192]; tokens [B,4,192]; window_valid [B,4];
        action_mask [B,4] (selected windows, 1=use).  Returns dz [B,C],
        zero for empty/all-invalid actions."""
        q = self.ln(self.wq(torch.cat([hT, hA, hV], dim=-1)))      # [B,192]
        if action_mask is None:
            action_mask = torch.ones_like(window_valid)
        sel = (action_mask * window_valid)                          # [B,4]
        n_sel = sel.sum(dim=1, keepdim=True)                        # [B,1]
        valid_rows = (n_sel.squeeze(1) > 0)
        dz = torch.zeros(q.shape[0], self.num_classes,
                         device=q.device, dtype=q.dtype)
        if not bool(valid_rows.any()):
            return dz
        idx = torch.nonzero(valid_rows).squeeze(1)
        if self.q_only:
            r = q[idx]
        else:
            key_mask = sel[idx] <= 0                                # True = ignore
            att, _ = self.attn(q[idx].unsqueeze(1),
                               tokens[idx], tokens[idx],
                               key_padding_mask=key_mask)
            r = att.squeeze(1)
        e = self.gelu(self.wv(r)) * torch.sigmoid(self.wc(q[idx]))
        dz[idx] = self.wout(e)
        return dz
