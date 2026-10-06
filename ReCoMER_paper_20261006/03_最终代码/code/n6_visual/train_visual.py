"""Train the visual corrector ONLY (default: no utility gate).

The base stays frozen (AnchoredVisualCorrector locks it); the optimizer
receives only corrector + adapter parameters.  Selection:
clean-valid WF1 -> Macro-F1 -> Accuracy -> lower unweighted CE.
Skeleton for P3; unit-tested here for gradient-path sanity only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def corrector_parameters(model):
    return [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("base.")]


def train_corrector(model, adapter, batches, valid_xy, lr: float, seed: int,
                    out: Path, epochs: int = 30, bs: int = 64,
                    device: str = "cpu"):
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(corrector_parameters(model)
                            + list(adapter.parameters()), lr=lr,
                            weight_decay=0.01)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b, y in batches(bs):
            hT, hA, hV, z_B = model.base(b["base_in"])
            tokens, wv = adapter(b["scene_feat"], b["face_feat"],
                                 b["scene_valid"], b["face_valid"],
                                 b["unique_frame"])
            dz = model.correct(hT, hA, hV, tokens, wv)
            loss = F.cross_entropy(z_B + dz, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
            opt.step()
        hist.append({"epoch": ep, "elapsed_sec": round(time.time() - t0, 1)})
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    (out / "FINAL_RESULT.json").write_text(json.dumps(
        {"status": "TRAIN_COMPLETE", "test_read": False}, indent=1))
    return hist
