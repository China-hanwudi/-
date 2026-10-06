"""Valid-only evaluation entry (refuses test by construction)."""
from __future__ import annotations

import numpy as np
import torch

ALLOWED_SPLITS = ("train", "valid")


def evaluate_valid(model_fn, ds, split: str, bs: int = 256):
    """model_fn: batch -> logits.  Returns (logits, y)."""
    if split not in ALLOWED_SPLITS:
        raise ValueError("split %r refused (valid-only evaluation)" % (split,))
    logits, ys = [], []
    with torch.no_grad():
        for s in range(0, ds.n, bs):
            b, y = ds.batch(range(s, min(s + bs, ds.n)))
            logits.append(model_fn(b).detach().cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(logits), np.concatenate(ys)
