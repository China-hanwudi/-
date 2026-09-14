"""Train/evaluate the active M3ED classification path (v9 evaluation protocol).

Only packed train.pt and valid.pt are read here.  The sealed test split is never
opened by this entry point.  Every seed/variant writes an isolated run directory
containing ``best.pt``, ``FINAL_RESULT.json``, ``history.json`` and
``RUN_METADATA.json`` with data hashes, code fingerprint, resolved config and the
ablation identity.

Checkpoint selection follows the frozen rule
``valid Weighted-F1 -> valid Macro-F1 -> valid Accuracy``; the test split is
never consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .m3ed_dataset import M3EDPackedDataset, M3ED_LABELS
from .config import N3TrainConfig
from .model import N3EmotionModel
from .losses import n3_total_loss


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def f1(labels, preds, c=7):
    cm = np.zeros((c, c), dtype=np.int64)
    for y, p in zip(labels, preds):
        if 0 <= int(y) < c and 0 <= int(p) < c:
            cm[int(y), int(p)] += 1
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    ps = cm.sum(0).astype(float)
    pr = np.divide(tp, ps, out=np.zeros_like(tp), where=ps > 0)
    rc = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    z = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(tp), where=(pr + rc) > 0)
    return (float(z.mean()), float((z * sup).sum() / max(sup.sum(), 1)),
            float((np.asarray(labels) == np.asarray(preds)).mean()),
            float(rc.mean()))


def class_weights(labels: torch.Tensor, mode: str, num_classes: int) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels.long().clamp(0, num_classes - 1), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    w = 1.0 / counts if mode == "inv" else 1.0 / counts.sqrt()
    return w / w.mean()


@torch.no_grad()
def evaluate(model, ds, device, bs, max_samples=None):
    model.eval()
    ys, ps = [], []
    loss = 0.0
    nstep = 0
    total = ds.n if max_samples is None else min(ds.n, int(max_samples))
    for s in range(0, total, bs):
        b, y = ds.batch(range(s, min(s + bs, total)))
        b = {k: v.to(device) for k, v in b.items()}
        y = y.to(device)
        out = model(b, route_mode="hard-safe")
        loss += float(n3_total_loss(out, y, model.cfg)["loss"])
        nstep += 1
        ys.extend(y.cpu().tolist())
        ps.extend(out["logits"].argmax(-1).cpu().tolist())
    ma, wf, acc, bal = f1(ys, ps, model.cfg.num_classes)
    return {"loss": loss / max(nstep, 1), "accuracy": acc, "macro_f1": ma,
            "weighted_f1": wf, "balanced_accuracy": bal, "n": len(ys)}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


def code_fingerprint(root: Path) -> dict:
    files = sorted((root / "n3_affect").glob("*.py"))
    return {p.name: sha(p)[:16] for p in files}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="main")
    # ---- ablation switches -------------------------------------------------
    ap.add_argument("--disable-history", action="store_true")
    ap.add_argument("--disable-relation", action="store_true")
    ap.add_argument("--no-counterfactual", action="store_true",
                    help="drop the measured leave-one-modality-out supervision")
    ap.add_argument("--no-redundancy", action="store_true",
                    help="drop sign-consistency and private-orthogonality terms")
    ap.add_argument("--no-adaptive-budget", action="store_true",
                    help="freeze the accept budget instead of adapting it per epoch")
    # M3ED's Fear/Disgust classes are extremely sparse.  The previous
    # unweighted objective matched the majority class but collapsed minority
    # recall (see the validation confusion audit).  Sqrt-inverse weighting is
    # a conservative default; ``none`` remains available as an explicit
    # ablation and ``inv`` as a stress test.
    ap.add_argument("--class-weight", default="sqrt_inv", choices=["none", "sqrt_inv", "inv"])
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    train = M3EDPackedDataset(args.data / "train.pt")
    valid = M3EDPackedDataset(args.data / "valid.pt")

    cfg = N3TrainConfig(
        text_dim=train.raw["T"].shape[-1], audio_dim=train.raw["A"].shape[-1],
        video_dim=train.raw["V"].shape[-1], num_classes=7,
        emotion_label_order=M3ED_LABELS,
        text_tower="composer_n3", dropout=0.1,
        batch_size=args.batch_size, max_epochs=args.epochs, seed=args.seed,
        lr=args.lr, risk_feature_version="conflict_v2",
        counterfactual_loss_weight=0.0 if args.no_counterfactual else 0.30,
        sign_consistency_weight=0.0 if args.no_redundancy else 0.05,
        private_orthogonality_weight=0.0 if args.no_redundancy else 0.05,
        risk_collapse_weight=0.10,
        unimodal_loss_weight=0.0,          # M3ED ships no unimodal labels
        use_adaptive_budget=not args.no_adaptive_budget,
        disable_history=bool(args.disable_history),
        disable_relation=bool(args.disable_relation),
        class_weight_mode=args.class_weight,
    )
    cfg.validate()
    model = N3EmotionModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)

    cw = class_weights(train.raw["label"], args.class_weight, cfg.num_classes)
    if cw is not None:
        cw = cw.to(device)

    meta = {
        "dataset": "M3ED", "task": "7_class_emotion", "tag": args.tag, "seed": args.seed,
        "device": str(device), "train_samples": train.n, "valid_samples": valid.n,
        "dims": {"T": int(cfg.text_dim), "A": int(cfg.audio_dim), "V": int(cfg.video_dim)},
        "train_sha256": sha(train.path), "valid_sha256": sha(valid.path),
        "test_read": False,
        "history_contract": "packed slots are newest-first; canonical oldest-to-newest, right-aligned K=3",
        "label_order_source": "M3ED/packed/data_audit.json label_field=EmoAnnotation.final_main_emo",
        "class_names": list(M3ED_LABELS),
        "selection_rule": "valid Weighted-F1 -> valid Macro-F1 -> valid Accuracy",
        "ablation": {
            "disable_history": bool(args.disable_history),
            "disable_relation": bool(args.disable_relation),
            "counterfactual_loss_weight": cfg.counterfactual_loss_weight,
            "sign_consistency_weight": cfg.sign_consistency_weight,
            "private_orthogonality_weight": cfg.private_orthogonality_weight,
            "risk_collapse_weight": cfg.risk_collapse_weight,
            "use_adaptive_budget": cfg.use_adaptive_budget,
            "class_weight_mode": cfg.class_weight_mode,
        },
        "resolved_config": cfg.to_dict(),
        "code_fingerprint": code_fingerprint(Path(__file__).resolve().parents[1]),
        "class_weights_applied": None if cw is None else [float(x) for x in cw.cpu()],
    }
    (args.out / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best = None
    history = []
    epochs = 1 if args.smoke else args.epochs
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        model.current_epoch = ep
        perm = torch.randperm(train.n)
        running = 0.0
        steps = 0
        for st in range(0, train.n, args.batch_size):
            if args.smoke and steps >= args.smoke:
                break
            ix = perm[st:min(st + args.batch_size, train.n)].tolist()
            b, y = train.batch(ix)
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out = model(b, route_mode="hard-safe")
            if cfg.counterfactual_loss_weight > 0:
                out["cf_measured_targets"] = model.measure_counterfactual_utility(b, y)
            loss = n3_total_loss(out, y, cfg, class_weight=cw)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach())
            steps += 1
        m = evaluate(model, valid, device, args.batch_size,
                     max_samples=(args.batch_size * 2 if args.smoke else None))
        row = {"epoch": ep, "train_loss": running / max(steps, 1),
               "elapsed_sec": round(time.time() - t0, 1), **m}
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (m["weighted_f1"], m["macro_f1"], m["accuracy"])
        if best is None or key > (best["weighted_f1"], best["macro_f1"], best["accuracy"]):
            best = dict(m)
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                        "seed": args.seed, "epoch": ep, "valid": m}, args.out / "best.pt")
            (args.out / "best_valid_metrics.json").write_text(
                json.dumps({"epoch": ep, **m}, indent=2), encoding="utf-8")
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not (args.out / "best.pt").is_file():
        raise RuntimeError("missing best.pt - run is not a valid result")
    final = {"dataset": "M3ED", "tag": args.tag, "seed": args.seed, "best_valid": best,
             "status": "SMOKE_PASS" if args.smoke else "TRAIN_COMPLETE",
             "test_read": False, "elapsed_sec": round(time.time() - t0, 1),
             "trainable_parameters": model.count_trainable_parameters()}
    (args.out / "FINAL_RESULT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("FINAL_RESULT", json.dumps(final), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
