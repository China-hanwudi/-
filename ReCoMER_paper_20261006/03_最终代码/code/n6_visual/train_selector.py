"""Gate/selector training (P4 skeleton).

Fits the selector on the OOF cache ONLY; valid is used for pre-declared
candidate-version selection and never back-propagates into gate weights.
Actual runs belong to the separate P4 approval.
"""
from __future__ import annotations


def train_selector(selector, cache_xy, lr: float, seed: int, epochs: int = 50):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    X, d, w = cache_xy
    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=0.0)
    huber = torch.nn.HuberLoss(delta=1.0, reduction="none")
    Xt = torch.tensor(X)
    dt = torch.tensor(d)
    wt = torch.tensor(w)
    n = Xt.shape[0]
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            ix = perm[s:s + bs]
            loss = (huber(selector(Xt[ix]), dt[ix]) * wt[ix]).sum() \
                / wt[ix].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
    return selector
