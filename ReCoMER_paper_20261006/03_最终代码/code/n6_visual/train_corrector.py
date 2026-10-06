"""P3 visual-increment trainer (work order §4): fit corrector (+adapter)
on a FROZEN champion base, default no utility gate.

Arms:
  v0  : 8-frame scene, masked mean -> 1 token (window_valid=[1,0,0,0])
  v2  : 16-frame scene, 4-window masked mean pool
  v4  : 16-frame scene, masked temporal conv (k=3) -> 4-window pool
  q0  : q-only corrector (no visual tokens), capacity-matched control

V1/V3 are NOT separately trained: with face_valid=0 (no audited face
detector, P1) they are numerically identical to V0/V2 (the face path is
masked and receives no gradient).  This is disclosed instead of burning
duplicate runs and inflating the fit count.

Recipe (locked, mirrors the audited base recipe): AdamW wd 0.01, bs 64,
max 30 epochs, warmup 2 + cosine T_max=10 (SequentialLR, epoch-end step),
SWA window 3 (partial window never eligible), early stop metric plateau
(patience 6, min_delta 1e-3) + absolute CE guard (+0.05, 4 consecutive),
selection clean-valid WF1 -> Macro-F1 -> Accuracy -> lower unweighted CE.
Loss: CE(label_smoothing=.05, sqrt-capped train class weights) on
z = z_B + dz.  Base stays frozen (eval-locked wrapper).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
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
from n6_visual.temporal_adapter import TemporalAdapter

NUM_CLASSES = None  # set from base cfg at runtime
CLASS_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness",
               "surprise"]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class TemporalConvAdapter(TemporalAdapter):
    """Masked temporal conv (k=3) WITHIN each window (window-isolated:
    no cross-window information sharing; K25-05 fix -- the first version
    convolved the whole 16-frame sequence, leaking across windows), then
    the same 4-window masked mean pool.  Invalid frames are zeroed BEFORE
    the conv and re-masked after (no motion invented from padding)."""

    def __init__(self, frame_dim: int = 512, d_model: int = 192):
        super().__init__(frame_dim, d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)

    def forward(self, scene_feat, face_feat, scene_valid, face_valid,
                unique_frame):
        B = scene_feat.shape[0]
        z = (scene_feat - self.mean) / self.std
        obs = (scene_valid * unique_frame)
        h = self.proj_scene(z) * obs.unsqueeze(-1)
        # per-window conv: reshape [B,4,4,D] -> conv each window separately
        hw = h.view(B, 4, 4, -1)
        hw = self.conv(hw.reshape(B * 4, 4, -1).transpose(1, 2)
                       ).transpose(1, 2).view(B, 4, 4, -1)
        h = hw.reshape(B, 16, -1) * obs.unsqueeze(-1)
        # face mask merges the observation mask (K25-05): face pooling only
        # over frames that are face-valid AND unique observations
        fmask = face_valid * unique_frame
        pf = self.proj_face(face_feat) * fmask.unsqueeze(-1)
        obs_w = obs.view(B, 4, 4)
        fm_w = fmask.view(B, 4, 4)
        tokens = torch.zeros(B, 4, h.shape[-1], device=h.device,
                             dtype=h.dtype)
        wvalid = torch.zeros(B, 4, device=h.device, dtype=h.dtype)
        for w in range(4):
            m = obs_w[:, w].unsqueeze(-1)
            tok = (h.view(B, 4, 4, -1)[:, w] * m).sum(1) / \
                m.sum(1).clamp_min(1.0)
            fm = fm_w[:, w].unsqueeze(-1)
            ftok = (pf.view(B, 4, 4, -1)[:, w] * fm).sum(1) / \
                fm.sum(1).clamp_min(1.0)
            tokens[:, w] = tok + ftok
            wvalid[:, w] = (m.sum(1).squeeze(-1) > 0).to(h.dtype)
        return tokens, wvalid


class Vf16Store:
    def __init__(self, split, pack_dir=None):
        pack_dir = pack_dir or "/data/emo/肖田泽科研/数据/MELD_frames16/packed"
        d = np.load(pack_dir + "/%s.npz" % split, allow_pickle=False)
        self.d = {k: d[k] for k in d.files}
        self.index = {str(u): i for i, u in enumerate(d["uid"])}

    def fetch(self, uids):
        ix = [self.index[str(u)] for u in uids]
        return {"scene_feat": torch.tensor(self.d["scene_feat"][ix]).float(),
                "face_feat": torch.tensor(self.d["face_feat"][ix]).float(),
                "scene_valid": torch.tensor(
                    self.d["scene_valid"][ix]).float(),
                "face_valid": torch.tensor(
                    self.d["face_valid"][ix]).float(),
                "unique_frame": torch.tensor(
                    self.d["unique_frame"][ix]).float()}


class Vf8Store:
    """Legacy 8-frame pack (MELD_frames Vf [N,8,512]) for the V0 arm."""

    def __init__(self, split):
        d = torch.load("/data/emo/肖田泽科研/数据/MELD_frames/packed/%s.pt"
                       % split, map_location="cpu", weights_only=True)
        self.ids = [str(i) for i in d["ids"]]
        self.index = {u: i for i, u in enumerate(self.ids)}
        self.Vf = d["Vf"].float()
        self.mm = d["modality_mask"].float()

    def fetch(self, uids):
        ix = torch.tensor([self.index[str(u)] for u in uids])
        vf = self.Vf[ix]
        fv = (vf.norm(dim=-1) > 0.5).float()
        uniq = torch.ones_like(fv)
        for r in range(vf.shape[0]):
            vi = torch.nonzero(fv[r] > 0).flatten()
            if len(vi) > 1:
                f = vf[r, vi]
                cs = f @ f.T
                off = cs[~torch.eye(len(vi), dtype=torch.bool)] > 0.9999
                if off.any():
                    # frames duplicating an EARLIER valid frame are non-unique
                    for a in range(1, len(vi)):
                        if (cs[a, :a] > 0.9999).any():
                            uniq[r, vi[a]] = 0.0
        return {"scene_feat": vf, "face_feat": torch.zeros_like(vf),
                "scene_valid": fv, "face_valid": torch.zeros_like(fv),
                "unique_frame": uniq}


import numpy as np  # noqa: E402  (top-level import kept for clarity)


def build_adapter(arm, frame_dim):
    if arm == "v4":
        return TemporalConvAdapter(frame_dim)
    return TemporalAdapter(frame_dim)


def adapter_tokens(arm, adapter, vbatch, device):
    if arm == "q0":
        # q-only control: no visual tokens; window_valid=1 so the q path
        # executes (tokens are ignored by the q_only corrector).
        B = vbatch["scene_feat"].shape[0]
        return (torch.zeros(B, 4, 192, device=device),
                torch.ones(B, 4, device=device))
    if arm == "v0":
        # 8-frame scene masked mean -> single token, window_valid=[1,0,0,0]
        z = (vbatch["scene_feat"] - adapter.mean) / adapter.std
        h = adapter.proj_scene(z)
        m = (vbatch["scene_valid"] * vbatch["unique_frame"]).unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        B = h.shape[0]
        tokens = torch.zeros(B, 4, pooled.shape[-1], device=device)
        tokens[:, 0] = pooled
        wvalid = torch.zeros(B, 4, device=device)
        wvalid[:, 0] = (m.sum(1).squeeze(-1) > 0).float()
        return tokens, wvalid
    return adapter(vbatch["scene_feat"], vbatch["face_feat"],
                   vbatch["scene_valid"], vbatch["face_valid"],
                   vbatch["unique_frame"])


@torch.no_grad()
def base_forward(base, b):
    """Frozen champion base -> (hT,hA,hV, z_B)."""
    cur = {m: base.encoders[m](b[m + "_t"]) for m in ("T", "A", "V")}
    z_B = base(b)["deployed"]
    return cur["T"], cur["A"], cur["V"], z_B


def eval_valid(base, model, adapter, arm, valid, vf, device, bs=256):
    model.eval()
    ys, preds, logits_all = [], [], []
    with torch.no_grad():
        for s in range(0, valid.n, bs):
            idx = range(s, min(s + bs, valid.n))
            b, y = valid.batch(idx)
            uids = [valid.raw["ids"][i] for i in idx]
            vb = vf.fetch(uids)
            b = {k: v.to(device) for k, v in b.items()}
            vb = {k: v.to(device) for k, v in vb.items()}
            hT, hA, hV, z_B = base_forward(base, b)
            tokens, wvalid = adapter_tokens(arm, adapter, vb, device)
            dz = model.correct(hT, hA, hV, tokens, wvalid)
            lg = z_B + dz
            logits_all.append(lg.cpu())
            ys.append(y.cpu())
    logits = torch.cat(logits_all)
    y = torch.cat(ys)
    ce = float(F.cross_entropy(logits, y))
    pred = logits.argmax(1)
    acc = float((pred == y).float().mean())
    wf1s, mf1s = [], []
    for c in range(NUM_CLASSES):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        wf1s.append(f1 * float((y == c).sum()))
        mf1s.append(f1)
    return {"weighted_f1": sum(wf1s) / max(float(len(y)), 1.0),
            "macro_f1": float(np.mean(mf1s)), "accuracy": acc,
            "unweighted_ce": ce, "logits": logits.numpy()}


def better(m, best, md=1e-3):
    if best is None:
        return True
    for k in ("weighted_f1", "macro_f1", "accuracy"):
        if m[k] > best[k] + md:
            return True
        if abs(m[k] - best[k]) > md:
            return False
    return m["unweighted_ce"] < best["unweighted_ce"] - 1e-9


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["v0", "v2", "v4", "q0"])
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--vf16-dir", default=None,
                    help="override Vf16 pack dir (e.g. the K25-01 v2 pack)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if (out_dir / "FINAL_RESULT.json").is_file():
        raise SystemExit("refusing to overwrite %s" % out_dir)
    # dir may pre-exist (the driver creates it for the log file); only a
    # completed run (FINAL_RESULT) blocks a rerun.
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(args.device)

    train = PackedDataset(Path(args.data) / "train.pt")
    valid = PackedDataset(Path(args.data) / "valid.pt")
    store_cls = Vf8Store if args.arm == "v0" else Vf16Store
    vf = store_cls("train", args.vf16_dir) if args.arm != "v0" \
        else store_cls("train")
    vf_valid = store_cls("valid", args.vf16_dir) if args.arm != "v0" \
        else store_cls("valid")

    base, bcfg, _ = load_checkpoint(args.base_ckpt, device)
    base.eval()
    global NUM_CLASSES
    NUM_CLASSES = int(bcfg.num_classes)
    model = AnchoredVisualCorrector(base, NUM_CLASSES,
                                    q_only=(args.arm == "q0")).to(device)
    frame_dim = int(vf.d["scene_feat"].shape[-1]) if args.arm != "v0" \
        else int(vf.Vf.shape[-1])
    adapter = build_adapter(args.arm, frame_dim).to(device)
    # adapter stats from train partition valid frames only
    with torch.no_grad():
        if args.arm == "v0":
            fv = (vf.Vf.norm(dim=-1) > 0.5)
            x = vf.Vf[fv]
        else:
            m = (vf.d["scene_valid"] * vf.d["unique_frame"]) > 0
            x = torch.tensor(vf.d["scene_feat"][m]).float()
        adapter.fit_stats(x, torch.ones(x.shape[0]))
    ytr = train.raw["label"].long()
    counts = torch.bincount(ytr, minlength=NUM_CLASSES).float().clamp_min(1.0)
    cw = torch.sqrt(torch.tensor(float(ytr.numel()))
                    / (NUM_CLASSES * counts)).clamp(max=2.0).to(device)

    params = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith("base.")]
    if args.arm != "q0":       # Q0 trains no visual adapter (disclosed)
        params += list(adapter.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[torch.optim.lr_scheduler.LambdaLR(
                        opt, lambda e: float(e + 1) / 2.0),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=10, eta_min=1e-6)],
        milestones=[2])

    snaps = deque(maxlen=min(3, min(args.epochs, 2)) if args.smoke else 3)
    swa = copy.deepcopy(model).eval()
    swa_ad = copy.deepcopy(adapter).eval()
    best = None
    best_state = None
    best_ep = -1
    bad_ep = 0
    bad_loss = 0
    ce_min = None
    hist = []
    t0 = time.time()
    epochs = min(args.epochs, 2) if args.smoke else args.epochs

    for ep in range(epochs):
        model.train()
        adapter.train()
        perm = torch.randperm(train.n)
        running, steps = 0.0, 0
        for st in range(0, train.n, 64):
            if args.smoke and steps >= args.smoke:
                break
            idx = perm[st:st + 64].tolist()
            b, y = train.batch(idx)
            uids = [train.raw["ids"][i] for i in idx]
            vb = vf.fetch(uids)
            b = {k: v.to(device) for k, v in b.items()}
            vb = {k: v.to(device) for k, v in vb.items()}
            y = y.to(device)
            with torch.no_grad():
                hT, hA, hV, z_B = base_forward(base, b)
            tokens, wvalid = adapter_tokens(args.arm, adapter, vb, device)
            dz = model.correct(hT, hA, hV, tokens, wvalid)
            loss = F.cross_entropy(z_B + dz, y, weight=cw,
                                   label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            running += float(loss.detach())
            steps += 1
        sched.step()

        snaps.append(({k: v.detach().clone()
                       for k, v in model.state_dict().items()},
                       {k: v.detach().clone()
                        for k, v in adapter.state_dict().items()}))
        if len(snaps) < snaps.maxlen:
            eligible = False
        else:
            eligible = True
            for sd, tgt in ((average_state([s[0] for s in snaps]), swa),
                            (average_state([s[1] for s in snaps]), swa_ad)):
                tgt.load_state_dict(sd, strict=True)
        cand_m, cand_a = (swa, swa_ad) if eligible else (model, adapter)
        m = eval_valid(base, cand_m, cand_a, args.arm, valid, vf_valid,
                       device)
        row = {"epoch": ep, "train_loss": running / max(steps, 1),
               "eligible": eligible,
               **{k: v for k, v in m.items() if k != "logits"}}
        hist.append(row)
        print(json.dumps(row), flush=True)
        if eligible and better(m, best):
            best = {k: v for k, v in m.items() if k != "logits"}
            best_state = ({k: v.cpu().clone()
                           for k, v in cand_m.state_dict().items()},
                          {k: v.cpu().clone()
                           for k, v in cand_a.state_dict().items()})
            best_ep = ep
            bad_ep = 0
        elif eligible:
            bad_ep += 1
        ce_min = m["unweighted_ce"] if ce_min is None else min(
            ce_min, m["unweighted_ce"])
        bad_loss = bad_loss + 1 if m["unweighted_ce"] > ce_min + 0.05 else 0
        if not args.smoke and bad_loss >= 4:
            break
        if not args.smoke and bad_ep >= 6:
            break

    if best_state is None:
        raise RuntimeError("no eligible candidate")
    torch.save({"model": best_state[0], "adapter": best_state[1],
                "arm": args.arm, "seed": args.seed, "epoch": best_ep,
                "base_ckpt": args.base_ckpt, "valid": best},
               out_dir / "best.pt")
    # reload integrity
    ck = torch.load(out_dir / "best.pt", map_location="cpu",
                    weights_only=True)
    m2 = AnchoredVisualCorrector(base, NUM_CLASSES,
                                 q_only=(args.arm == "q0")).to(device)
    m2.load_state_dict({k: v.to(device) for k, v in ck["model"].items()},
                       strict=False)
    a2 = build_adapter(args.arm, frame_dim).to(device)
    a2.load_state_dict({k: v.to(device) for k, v in ck["adapter"].items()})
    mi = eval_valid(base, m2, a2, args.arm, valid, vf_valid, device)
    ok = abs(mi["weighted_f1"] - best["weighted_f1"]) < 1e-6
    if not ok:
        raise RuntimeError("reload mismatch %.6f vs %.6f"
                           % (mi["weighted_f1"], best["weighted_f1"]))
    (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    (out_dir / "RUN_METADATA.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "lr": args.lr,
        "base_ckpt": args.base_ckpt,
        "base_ckpt_sha256": sha256(args.base_ckpt),
        "code": code_manifest(),
        "test_read": False,
        "params_new": int(sum(p.numel() for p in params)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=1))
    (out_dir / "FINAL_RESULT.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "best_valid": best,
        "best_epoch_index": best_ep, "reload_integrity": ok,
        "epochs_completed": len(hist), "test_read": False,
        "status": "SMOKE_PASS" if args.smoke else "TRAIN_COMPLETE",
        "elapsed_sec": round(time.time() - t0, 1)}, indent=1))
    print("FINAL_RESULT", json.dumps({"arm": args.arm, "seed": args.seed,
                                      "wf1": best["weighted_f1"]}))
    return 0


def average_state(snaps):
    out = {}
    n = float(len(snaps))
    for k in snaps[0]:
        vals = [s[k] for s in snaps]
        if vals[0].is_floating_point():
            acc = vals[0].clone()
            for v in vals[1:]:
                acc.add_(v)
            out[k] = acc / n
        else:
            out[k] = vals[-1].clone()
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def code_manifest():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


if __name__ == "__main__":
    raise SystemExit(main())
