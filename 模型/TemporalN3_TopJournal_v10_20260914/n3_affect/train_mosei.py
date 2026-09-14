"""Train CMU-MOSEI scalar sentiment on train/valid packed splits only.

Checkpoint selection uses valid MAE only; the sealed test split is never opened.
Predictions are reported in the raw [-3,3] sentiment scale (the model trains on
y/3 and emits 3*u).

Run directories contain ``best.pt``, ``FINAL_RESULT.json``, ``history.json`` and
``RUN_METADATA.json`` (data hashes, code fingerprint, resolved config, ablation
identity).
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

from .mosei_dataset import MOSEIPackedDataset
from .regression_config import N3RegressionConfig
from .regression_model import N3SentimentModel
from .regression_losses import n3_regression_loss


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, ds, device, bs, max_samples=None):
    from . import metrics as M
    model.eval()
    preds, trues = [], []
    total = ds.n if max_samples is None else min(ds.n, int(max_samples))
    for st in range(0, total, bs):
        b, t = ds.batch(range(st, min(st + bs, total)))
        b = {k: v.to(device) for k, v in b.items()}
        o = model(b, "hard-safe")
        preds.extend(o["prediction"].reshape(-1).cpu().tolist())
        trues.extend(t.reshape(-1).tolist())
    p = np.asarray(preds, float)
    y = np.asarray(trues, float)
    rep = M.regression_report(y, p, clip=None)
    return {
        "MAE": rep["mae"], "MSE": rep["mse"], "RMSE": rep["rmse"],
        "Pearson": rep["pearson"], "CCC": rep["ccc"], "R2": rep["r2"],
        "Acc2": rep["binary_gt0"]["acc2"], "F1_2": rep["binary_gt0"]["f1_2_positive"],
        "Acc2_ge0": rep["binary_ge0"]["acc2"], "F1_2_ge0": rep["binary_ge0"]["f1_2_positive"],
        "n": int(y.size),
    }


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


def code_fingerprint(root: Path) -> dict:
    return {p.name: sha(p)[:16] for p in sorted((root / "n3_affect").glob("*.py"))}


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
    ap.add_argument("--disable-history", action="store_true")
    ap.add_argument("--disable-relation", action="store_true")
    ap.add_argument("--no-counterfactual", action="store_true")
    ap.add_argument("--no-redundancy", action="store_true")
    ap.add_argument("--no-adaptive-budget", action="store_true")
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    tr = MOSEIPackedDataset(args.data / "train.pt")
    va = MOSEIPackedDataset(args.data / "valid.pt")

    cfg = N3RegressionConfig(
        text_dim=tr.raw["T"].shape[-1], audio_dim=tr.raw["A"].shape[-1],
        video_dim=tr.raw["V"].shape[-1], target_scale=3.0,
        task="mosei_sentiment_regression", batch_size=args.batch_size,
        max_epochs=args.epochs, seed=args.seed, lr=args.lr,
        counterfactual_loss_weight=0.0 if args.no_counterfactual else 0.20,
        sign_consistency_weight=0.0 if args.no_redundancy else 0.05,
        private_orthogonality_weight=0.0 if args.no_redundancy else 0.05,
        risk_collapse_weight=0.10, unimodal_loss_weight=0.0,
        global_unimodal_loss_weight=0.08, current_modality_dropout=0.12,
        use_adaptive_budget=not args.no_adaptive_budget,
        disable_history=bool(args.disable_history),
        disable_relation=bool(args.disable_relation),
    )
    cfg.validate()
    m = N3SentimentModel(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": "CMU-MOSEI", "tag": args.tag, "seed": args.seed, "device": str(device),
        "train_samples": tr.n, "valid_samples": va.n,
        "dims": {"T": cfg.text_dim, "A": cfg.audio_dim, "V": cfg.video_dim},
        "target": "raw sentiment in [-3,3]; model trains on y/3 and reports 3*u",
        "Acc2_headline_threshold": "prediction > 0 is the positive class (Acc2/F1_2); "
                                   "the MOSEI has0 (>=0) convention is reported as Acc2_ge0",
        "train_sha256": sha(tr.path), "valid_sha256": sha(va.path), "test_read": False,
        "history_contract": "packed slots are newest-first; canonical oldest-to-newest, right-aligned K=3",
        "selection_rule": "valid MAE",
        "ablation": {
            "disable_history": bool(args.disable_history),
            "disable_relation": bool(args.disable_relation),
            "counterfactual_loss_weight": cfg.counterfactual_loss_weight,
            "sign_consistency_weight": cfg.sign_consistency_weight,
            "private_orthogonality_weight": cfg.private_orthogonality_weight,
            "risk_collapse_weight": cfg.risk_collapse_weight,
            "use_adaptive_budget": cfg.use_adaptive_budget,
        },
        "resolved_config": cfg.to_dict(),
        "code_fingerprint": code_fingerprint(Path(__file__).resolve().parents[1]),
    }
    (args.out / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best = None
    hist = []
    epochs = 1 if args.smoke else args.epochs
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        m.current_epoch = ep
        perm = torch.randperm(tr.n)
        run = 0.0
        steps = 0
        for st in range(0, tr.n, args.batch_size):
            if args.smoke and steps >= args.smoke:
                break
            ix = perm[st:min(st + args.batch_size, tr.n)].tolist()
            b, y = tr.batch(ix)
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            o = m(b, "hard-safe")
            if cfg.counterfactual_loss_weight > 0:
                o["cf_measured_targets"] = m.measure_counterfactual_utility(b, y)
            z = n3_regression_loss(o, y, cfg)["loss"]
            opt.zero_grad(set_to_none=True)
            z.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
            opt.step()
            run += float(z.detach())
            steps += 1
        met = evaluate(m, va, device, args.batch_size,
                       max_samples=(args.batch_size * 2 if args.smoke else None))
        row = {"epoch": ep, "train_loss": run / max(steps, 1),
               "elapsed_sec": round(time.time() - t0, 1), **met}
        hist.append(row)
        print(json.dumps(row), flush=True)
        if best is None or met["MAE"] < best["MAE"]:
            best = dict(met)
            torch.save({"model": m.state_dict(), "cfg": cfg.to_dict(), "seed": args.seed,
                        "epoch": ep, "valid": met}, args.out / "best.pt")
            (args.out / "best_valid_metrics.json").write_text(
                json.dumps({"epoch": ep, **met}, indent=2), encoding="utf-8")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    if not (args.out / "best.pt").is_file():
        raise RuntimeError("missing best.pt - run is not a valid result")
    final = {"dataset": "CMU-MOSEI", "tag": args.tag, "seed": args.seed, "best_valid": best,
             "status": "SMOKE_PASS" if args.smoke else "TRAIN_COMPLETE", "test_read": False,
             "elapsed_sec": round(time.time() - t0, 1),
             "trainable_parameters": m.count_trainable_parameters()}
    (args.out / "FINAL_RESULT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("FINAL_RESULT", json.dumps(final), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
