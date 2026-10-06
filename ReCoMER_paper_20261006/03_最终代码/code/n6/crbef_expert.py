"""Frozen TAV + rich-AV cRBEF expert from the supplied candidate package.

These feature heads are independent of MHnoU. CR h/a and rich audio are
different representations; native MHnoU features must never substitute them.
"""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from .crbef_source.head import Head
from .crbef_source.rich import Rich

M3ED_CLASSES = ("Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise")


class CRBEFExpert(nn.Module):
    @classmethod
    def from_state(cls, state, class_names):
        if tuple(class_names) != M3ED_CLASSES:
            raise ValueError("unsupported bundled expert class order")
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.class_names = tuple(class_names)
        model.tav, model.av, model.gate = Head(), Rich(), nn.Linear(6, 1)
        for key in ("prior", "video_mean", "video_scale"):
            model.register_buffer(key, torch.empty_like(state[key]))
        model.load_state_dict(state, strict=True)
        return model.requires_grad_(False).eval()

    def __init__(self, assets):
        super().__init__()
        assets = Path(assets)
        meta = json.loads((assets / "IMPORT_PROVENANCE.json").read_text(encoding="utf-8"))
        self.class_names = tuple(meta["class_names"])
        if self.class_names != M3ED_CLASSES:
            raise ValueError("supplied cRBEF checkpoints require the M3ED class order")
        self.tav, self.av, self.gate = Head(), Rich(), nn.Linear(6, 1)
        self.tav.load_state_dict(torch.load(assets / "TAV.pt", map_location="cpu", weights_only=True))
        self.av.load_state_dict(torch.load(assets / "AV.pt", map_location="cpu", weights_only=True))
        ck = torch.load(assets / "CR_gate.pt", map_location="cpu", weights_only=True)
        self.gate.load_state_dict(ck["gate"])
        self.register_buffer("prior", ck["prior"].float())
        with np.load(assets / "normalizers" / "V.npz", allow_pickle=False) as z:
            self.register_buffer("video_mean", torch.from_numpy(z["mean"].copy()))
            self.register_buffer("video_scale", torch.from_numpy(z["scale"].copy()))
        if self.video_mean.shape != (342,) or bool((self.video_scale <= 0).any()):
            raise ValueError("invalid original cRBEF video normalizer")
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        device = self.prior.device
        x = {k: torch.as_tensor(x[k], dtype=torch.float32, device=device)
             for k in ("h", "a", "v", "pa", "pv", "A_mean", "A_local4")}
        n = len(x["h"])
        shapes = {"h": (n, 4096), "a": (n, 1024), "v": (n, 342),
                  "pa": (n,), "pv": (n,), "A_mean": (n, 1024)}
        for k, shape in shapes.items():
            if tuple(x[k].shape) != shape or not bool(torch.isfinite(x[k]).all()):
                raise ValueError("invalid cRBEF feature " + k)
        al = x["A_local4"].flatten(1)
        if al.shape != (n, 4096) or not bool(torch.isfinite(al).all()):
            raise ValueError("A_local4 must be [N,4,1024] or [N,4096]")
        for k in ("pa", "pv"):
            if not bool(((x[k] == 0) | (x[k] == 1)).all()):
                raise ValueError(k + " must contain binary presence flags")
        # Original source recipe normalizes video only on present rows.
        v = torch.zeros_like(x["v"])
        present = x["pv"] > 0
        v[present] = ((x["v"][present].double() - self.video_mean) / self.video_scale).float()
        p = self.tav(dict(h=x["h"], a=x["a"], v=v, pa=x["pa"], pv=x["pv"])).softmax(1)
        av = self.av(x["A_mean"], al, v).softmax(1)
        top = p.topk(2, dim=1).values
        evidence = (av + 1e-9).log() - (self.prior + 1e-9).log()
        stats = torch.stack([top[:, 0], top[:, 0] - top[:, 1], av.max(1).values,
                             evidence.max(1).values, (p.argmax(1) == av.argmax(1)).float(),
                             -(p * p.clamp_min(1e-8).log()).sum(1)], dim=1)
        gate = self.gate(stats).sigmoid()
        cr = ((p + 1e-9).log() + gate * evidence).softmax(1)
        return {"CRBEF": cr, "TAV": p, "AV": av, "CR_gate": gate}


class CRBEFFeatureStore:
    """Join the original CR/rich feature files by unique utterance ID."""
    def __init__(self, cr_npz, rich_pt):
        with np.load(cr_npz, allow_pickle=False) as z:
            self.cr = {k: z[k].copy() for k in ("ids", "h", "a", "v", "pa", "pv")}
        self.rich = torch.load(rich_pt, map_location="cpu", weights_only=True)
        self.cr_lookup = self._lookup(self.cr["ids"], "CR")
        self.rich_lookup = self._lookup(self.rich["ids"], "rich audio")

    @staticmethod
    def _lookup(ids, name):
        ids = [str(v) for v in ids]
        if len(set(ids)) != len(ids):
            raise ValueError(name + " contains duplicate IDs")
        return {v: i for i, v in enumerate(ids)}

    def batch(self, ids):
        ids = [str(v) for v in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("requested utterance IDs must be unique")
        missing = [v for v in ids if v not in self.cr_lookup or v not in self.rich_lookup]
        if missing:
            raise ValueError("missing original expert features for " + repr(missing[:3]))
        ci = [self.cr_lookup[v] for v in ids]
        ri = torch.tensor([self.rich_lookup[v] for v in ids], dtype=torch.long)
        return {**{k: self.cr[k][ci] for k in ("h", "a", "v", "pa", "pv")},
                "A_mean": self.rich["A_mean"][ri], "A_local4": self.rich["A_local4"][ri]}
