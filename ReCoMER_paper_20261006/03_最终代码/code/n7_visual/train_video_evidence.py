"""Frozen sparse-face video encoder probe with matched evidence controls.

The residual head is identical across dynamic, static, CLIP and shuffled inputs.
This is a development experiment, not a claim of
novelty or a final CVPR method. No test partition is loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from n6.data import PackedDataset
from n6.evaluate import load_checkpoint
from n6_visual.anchored_model import AnchoredVisualCorrector
from n6_visual.train_corrector import (Vf16Store, adapter_tokens, average_state,
                                       base_forward, build_adapter,
                                       seed_all, sha256)


class FaceStore(Vf16Store):
    def __init__(self, split, pack_dir, mode, seed):
        super().__init__(split, pack_dir)
        self.evidence = self.d["dynamic" if mode == "shuffled" else mode].copy()
        if mode == "shuffled":
            valid = np.flatnonzero(self.d["evidence_valid"])
            rng = np.random.RandomState(seed + (0 if split == "train" else 10000))
            self.evidence[valid] = self.evidence[rng.permutation(valid)]

    def fetch(self, uids):
        out = super().fetch(uids)
        ix = [self.index[str(u)] for u in uids]
        out["evidence"] = torch.as_tensor(self.evidence[ix]).float()
        out["evidence_valid"] = torch.as_tensor(self.d["evidence_valid"][ix]).float()
        return out


def scene_reference_tokens(adapter, vb, device):
    """The reference was trained with no face input; keep that exact contract.

    A frozen parameter tensor does not imply a frozen function if new inputs
    activate its previously unused face projection. Never mutate the candidate
    input dictionary: the trainable face branch still needs its face evidence.
    """
    scene_vb = dict(vb)
    scene_vb["face_feat"] = torch.zeros_like(vb["face_feat"])
    scene_vb["face_valid"] = torch.zeros_like(vb["face_valid"])
    return adapter_tokens("v4", adapter, scene_vb, device)


class ConditionalFaceResidual(nn.Module):
    def __init__(self, d=192, classes=7):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(3 * d + classes, d),
                                   nn.LayerNorm(d), nn.GELU())
        self.register_buffer("feature_mean", torch.zeros(512))
        self.register_buffer("feature_std", torch.ones(512))
        self.feature = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d), nn.GELU())
        self.gate = nn.Linear(d, d)
        self.output = nn.Linear(d, classes, bias=False)
        nn.init.zeros_(self.output.weight)

    @torch.no_grad()
    def fit_stats(self, evidence, valid):
        x = torch.as_tensor(evidence[valid > 0], device=self.feature_mean.device)
        if len(x) < 2:
            raise ValueError("insufficient train evidence")
        self.feature_mean.copy_(x.mean(0))
        self.feature_std.copy_(x.std(0).clamp_min(1e-5))

    def forward(self, ht, ha, hv, zref, vb):
        q = self.query(torch.cat((ht, ha, hv, zref.detach().softmax(-1)), -1))
        x = (vb["evidence"] - self.feature_mean) / self.feature_std
        h = self.feature(x) * torch.sigmoid(self.gate(q))
        return self.output(h) * vb["evidence_valid"].unsqueeze(-1)


def better(m, best):
    # Checkpoint selection follows the project's exact WF1-first ordering.
    if best is None:
        return True
    def key(x):
        return (x["weighted_f1"], x["macro_f1"], x["accuracy"], -x["unweighted_ce"])
    return key(m) > key(best)


def score(logits, labels):
    pred = logits.argmax(1)
    wf1, mf1 = [], []
    n = len(labels)
    for c in range(7):
        tp = ((pred == c) & (labels == c)).sum().item()
        fp = ((pred == c) & (labels != c)).sum().item()
        fn = ((pred != c) & (labels == c)).sum().item()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        wf1.append(f1 * (labels == c).sum().item())
        mf1.append(f1)
    return {"weighted_f1": sum(wf1) / n,
            "macro_f1": float(np.mean(mf1)),
            "accuracy": float((pred == labels).float().mean()),
            "unweighted_ce": float(F.cross_entropy(logits, labels))}


@torch.no_grad()
def eval_valid(base, scene, adapter, face, data, store, device):
    face.eval()
    logits, ys = [], []
    for st in range(0, data.n, 256):
        ids = list(range(st, min(st + 256, data.n)))
        b, y = data.batch(ids)
        vb = store.fetch([data.raw["ids"][i] for i in ids])
        b = {k: v.to(device) for k, v in b.items()}
        vb = {k: v.to(device) for k, v in vb.items()}
        ht, ha, hv, zb = base_forward(base, b)
        tok, valid = scene_reference_tokens(adapter, vb, device)
        zref = zb + scene.correct(ht, ha, hv, tok, valid)
        logits.append((zref + face(ht, ha, hv, zref, vb)).cpu())
        ys.append(y.cpu())
    return score(torch.cat(logits), torch.cat(ys))


def source_hashes(root):
    names = [Path(__file__)] + sorted((root / "n6").glob("*.py")) + sorted(
        (root / "n6_visual").glob("*.py"))
    return {str(p.relative_to(root)): sha256(p) for p in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=Path, required=True)
    ap.add_argument("--scene-ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--vf16-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("dynamic", "static", "clip", "shuffled"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.out
    if (out / "FINAL_RESULT.json").exists():
        raise FileExistsError(out / "FINAL_RESULT.json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "CONFIG.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sort_keys=True, indent=2))
    seed_all(args.seed)
    device = torch.device(args.device)
    train = PackedDataset(args.data / "train.pt")
    valid = PackedDataset(args.data / "valid.pt")
    store_tr = FaceStore("train", str(args.vf16_dir), args.mode, args.seed)
    store_va = FaceStore("valid", str(args.vf16_dir), args.mode, args.seed)
    base, _, _ = load_checkpoint(str(args.base_ckpt), device)
    base.eval()
    scene_ck = torch.load(args.scene_ckpt, map_location="cpu", weights_only=True)
    assert scene_ck["arm"] == "v4" and scene_ck["seed"] == args.seed
    scene = AnchoredVisualCorrector(base, 7).to(device)
    scene.load_state_dict(scene_ck["model"], strict=True)
    scene.eval()
    adapter = build_adapter("v4", 512).to(device)
    adapter.load_state_dict(scene_ck["adapter"], strict=True)
    adapter.eval()
    for p in scene.parameters():
        p.requires_grad_(False)
    for p in adapter.parameters():
        p.requires_grad_(False)
    face = ConditionalFaceResidual().to(device)
    face.fit_stats(store_tr.evidence, store_tr.d["evidence_valid"])
    reference_metrics = eval_valid(base, scene, adapter, face, valid, store_va,
                                   device)
    reference_ok = all(abs(reference_metrics[k] - scene_ck["valid"][k]) < 1e-6
                       for k in reference_metrics)
    (out / "REFERENCE_CHECK.json").write_text(json.dumps({
        "stored_scene_metrics": scene_ck["valid"],
        "zero_residual_metrics": reference_metrics,
        "matches": reference_ok, "test_read": False}, indent=2))
    if not reference_ok:
        raise RuntimeError("zero residual fails to reproduce scene reference")
    optimizer = torch.optim.AdamW(face.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda e: float(e + 1) / 2.0),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(args.epochs - 2, 1), eta_min=1e-6)],
        milestones=[2])
    ytr = train.raw["label"].long()
    counts = torch.bincount(ytr, minlength=7).float().clamp_min(1)
    cw = torch.sqrt(torch.tensor(float(len(ytr))) / (7 * counts)).clamp(
        max=2).to(device)
    best, best_state, best_ep = None, None, -1
    snaps = deque(maxlen=3)
    swa = copy.deepcopy(face).eval()
    bad_ep, bad_loss, ce_min = 0, 0, None
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        face.train()
        perm = torch.randperm(train.n)
        running, steps = 0., 0
        for st in range(0, train.n, 64):
            ids = perm[st:st + 64].tolist()
            b, y = train.batch(ids)
            vb = store_tr.fetch([train.raw["ids"][i] for i in ids])
            b = {k: v.to(device) for k, v in b.items()}
            vb = {k: v.to(device) for k, v in vb.items()}
            y = y.to(device)
            with torch.no_grad():
                ht, ha, hv, zb = base_forward(base, b)
                tok, window_valid = scene_reference_tokens(adapter, vb, device)
                zref = zb + scene.correct(ht, ha, hv, tok, window_valid)
            dz = face(ht, ha, hv, zref, vb)
            loss = F.cross_entropy(zref + dz, y, weight=cw,
                                   label_smoothing=.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(face.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        scheduler.step()
        snaps.append({k: v.detach().clone() for k, v in face.state_dict().items()})
        eligible = len(snaps) == 3
        if eligible:
            swa.load_state_dict(average_state(list(snaps)), strict=True)
        candidate = swa if eligible else face
        m = eval_valid(base, scene, adapter, candidate, valid, store_va, device)
        hist.append({"epoch": ep, "train_loss": running / max(steps, 1),
                     "eligible": eligible, **m})
        print(json.dumps(hist[-1]), flush=True)
        if eligible and better(m, best):
            best = m
            best_state = {k: v.cpu().clone() for k, v in candidate.state_dict().items()}
            best_ep = ep
            bad_ep = 0
        elif eligible:
            bad_ep += 1
        ce_min = m["unweighted_ce"] if ce_min is None else min(
            ce_min, m["unweighted_ce"])
        bad_loss = bad_loss + 1 if m["unweighted_ce"] > ce_min + .05 else 0
        if bad_ep >= 6 or bad_loss >= 4:
            break
    if best_state is None:
        raise RuntimeError("no eligible checkpoint")
    torch.save({"face": best_state, "seed": args.seed, "mode": args.mode,
                "scene_ckpt": str(args.scene_ckpt), "valid": best}, out / "best.pt")
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    reloaded = ConditionalFaceResidual().to(device)
    reloaded.load_state_dict(ck["face"], strict=True)
    check = eval_valid(base, scene, adapter, reloaded, valid, store_va, device)
    ok = all(abs(check[k] - best[k]) < 1e-8 for k in best)
    if not ok:
        raise RuntimeError("checkpoint reload did not match")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "RUN_METADATA.json").write_text(json.dumps({
        "config_sha256": sha256(out / "CONFIG.json"),
        "checkpoint_sha256": sha256(out / "best.pt"),
        "base_ckpt_sha256": sha256(args.base_ckpt),
        "scene_ckpt_sha256": sha256(args.scene_ckpt),
        "pack_sha256": {s: sha256(args.vf16_dir / (s + ".npz"))
                        for s in ("train", "valid")},
        "data_sha256": {s: sha256(args.data / (s + ".pt"))
                        for s in ("train", "valid")},
        "source_sha256": source_hashes(Path(__file__).resolve().parents[1]),
        "lr": args.lr, "seed": args.seed, "mode": args.mode,
        "selection": "exact WF1 > MacroF1 > accuracy > lower CE",
        "scheduler": "2 warmup epochs + monotonic cosine to planned final epoch",
        "params_new": sum(p.numel() for p in face.parameters()),
        "test_read": False}, indent=2))
    (out / "FINAL_RESULT.json").write_text(json.dumps({
        "status": "TRAIN_COMPLETE", "mode": args.mode, "seed": args.seed,
        "best_valid": best, "best_epoch_index": best_ep,
        "reference_valid": reference_metrics,
        "reload_integrity": ok, "epochs_completed": len(hist),
        "test_read": False, "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
    print("FINAL_RESULT", json.dumps({"mode": args.mode,
                                       "seed": args.seed,
                                       "wf1": best["weighted_f1"]}))


if __name__ == "__main__":
    main()
