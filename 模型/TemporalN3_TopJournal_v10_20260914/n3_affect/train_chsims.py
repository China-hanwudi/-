"""CH-SIMS_v2 continuous sentiment trainer (v9 evaluation protocol).

CH-SIMS v2 is the only active dataset that ships per-modality labels
(``label_T/A/V``), so it is where the label-anchored unimodal head is actually
supervised: ``regression_losses.n3_regression_loss`` routes the unimodal and the
current-candidate counterfactual terms through ``modality_mask``, so they stay
alive even though CH-SIMS has no conversational history.

Protocol
--------
* train on ``v7_packed/train.pt``; select the checkpoint by **valid MAE only**;
* predictions clipped to [-1,1] before MAE (official convention);
* Acc-2 positive class is ``y > 0`` (a 0 label is NON-positive) - the MOSEI
  ``>= 0`` convention is deliberately not applied;
* ``SEALED_TEST_DO_NOT_READ/test.pt`` is never opened here.
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

from .regression_config import N3RegressionConfig
from .regression_model import N3SentimentModel
from .regression_losses import n3_regression_loss


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChSimsPacked:
    """No modality mask and no conversational history on CH-SIMS v2."""

    def __init__(self, path: Path) -> None:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self.ids = [str(x) for x in raw.pop("ids")]
        self.data = {k: v for k, v in raw.items() if torch.is_tensor(v)}
        self.n = len(self.ids)
        self.path = Path(path)
        assert self.data["label"].shape == (self.n,)
        assert self.data["unimodal_labels"].shape == (self.n, 3)

    def batch(self, ix):
        ix = torch.as_tensor(list(ix), dtype=torch.long)
        d = self.data
        b = int(ix.numel())
        dev = d["T"].device
        return {
            "T_t": d["T"][ix], "A_t": d["A"][ix], "V_t": d["V"][ix],
            "T_h": torch.zeros(b, 1, d["T"].size(-1), device=dev),
            "A_h": torch.zeros(b, 1, d["A"].size(-1), device=dev),
            "V_h": torch.zeros(b, 1, d["V"].size(-1), device=dev),
            "history_mask": torch.zeros(b, 1, device=dev),
            "modality_mask": torch.ones(b, 3, device=dev),
            "history_modality_mask": torch.zeros(b, 1, 3, device=dev),
            "speaker_same": torch.zeros(b, 1, device=dev),
            "unimodal_labels": d["unimodal_labels"][ix],
        }, d["label"][ix]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


def code_fingerprint(root: Path) -> dict:
    return {p.name: sha(p)[:16] for p in sorted((root / "n3_affect").glob("*.py"))}


@torch.no_grad()
def evaluate(model, split, batch_size, device, max_samples=None):
    from . import metrics as M
    model.eval()
    total = split.n if max_samples is None else min(split.n, int(max_samples))
    preds, trues = [], []
    for start in range(0, total, batch_size):
        ix = torch.arange(start, min(start + batch_size, total))
        batch, y = split.batch(ix)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, route_mode="hard-safe")
        preds.extend(out["prediction"].reshape(-1).cpu().tolist())
        trues.extend(y.reshape(-1).tolist())
    p = np.asarray(preds, float)
    y = np.asarray(trues, float)
    rep = M.regression_report(y, p, clip=(-1.0, 1.0))
    return {"MAE": rep["mae"], "MAE_unclipped": rep["mae_unclipped"], "RMSE": rep["rmse"],
            "Pearson": rep["pearson"], "CCC": rep["ccc"], "R2": rep["r2"],
            "Acc2": rep["binary_gt0"]["acc2"], "F1_2": rep["binary_gt0"]["f1_2_positive"],
            "n": int(y.size)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--risk-margin", type=float, default=0.02,
                    help="relative-MAE margin for the candidate harm label; 0 reproduces "
                         "the pre-v7.1 zero-tolerance label")
    ap.add_argument("--unimodal-weight", type=float, default=0.20)
    ap.add_argument("--disable-history", action="store_true")
    ap.add_argument("--disable-relation", action="store_true")
    ap.add_argument("--no-counterfactual", action="store_true")
    ap.add_argument("--no-redundancy", action="store_true")
    ap.add_argument("--no-unimodal", action="store_true",
                    help="drop label-anchored unimodal supervision (the 'nouni' arm)")
    ap.add_argument("--no-adaptive-budget", action="store_true")
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    train = ChSimsPacked(args.data / "train.pt")
    valid = ChSimsPacked(args.data / "valid.pt")
    dims = {k: int(train.data[m].shape[-1]) for k, m in
            (("text_dim", "T"), ("audio_dim", "A"), ("video_dim", "V"))}
    print("dims:", dims, "train n:", train.n, "valid n:", valid.n, flush=True)

    cfg = N3RegressionConfig(
        **dims, target_scale=1.0, task="chsims_v2_sentiment_regression",
        batch_size=args.batch_size, max_epochs=args.epochs, seed=args.seed, lr=args.lr,
        counterfactual_loss_weight=0.0 if args.no_counterfactual else 0.30,
        unimodal_loss_weight=0.0 if args.no_unimodal else args.unimodal_weight,
        global_unimodal_loss_weight=0.04, current_modality_dropout=0.10,
        text_shortcut_weight=0.0 if args.no_unimodal else 0.05,
        risk_collapse_weight=0.10,
        sign_consistency_weight=0.0 if args.no_redundancy else 0.05,
        private_orthogonality_weight=0.0 if args.no_redundancy else 0.05,
        risk_margin=args.risk_margin,
        use_adaptive_budget=not args.no_adaptive_budget,
        disable_history=bool(args.disable_history),
        disable_relation=bool(args.disable_relation),
    )
    cfg.validate()
    model = N3SentimentModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": "CH-SIMS_v2", "tag": args.tag, "seed": args.seed, "device": str(device),
        "train_samples": train.n, "valid_samples": valid.n, "dims": dims,
        "label_protocol": "continuous sentiment [-1,1]; predictions clipped to [-1,1] "
                          "before MAE; Acc-2 positive class = y > 0 (0 is non-positive)",
        "train_sha256": sha(train.path), "valid_sha256": sha(valid.path), "test_read": False,
        "selection_rule": "valid MAE",
        "ablation": {
            "disable_history": bool(args.disable_history),
            "disable_relation": bool(args.disable_relation),
            "counterfactual_loss_weight": cfg.counterfactual_loss_weight,
            "unimodal_loss_weight": cfg.unimodal_loss_weight,
            "text_shortcut_weight": cfg.text_shortcut_weight,
            "sign_consistency_weight": cfg.sign_consistency_weight,
            "private_orthogonality_weight": cfg.private_orthogonality_weight,
            "risk_collapse_weight": cfg.risk_collapse_weight,
            "risk_margin": cfg.risk_margin,
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
        model.train()
        model.current_epoch = ep
        perm = torch.randperm(train.n)
        run = 0.0
        steps = 0
        for st in range(0, train.n, args.batch_size):
            if args.smoke and steps >= args.smoke:
                break
            ix = perm[st:min(st + args.batch_size, train.n)]
            if ix.numel() < 2:
                continue
            b, y = train.batch(ix)
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out = model(b, route_mode="hard-safe")
            if cfg.counterfactual_loss_weight > 0:
                out["cf_measured_targets"] = model.measure_counterfactual_utility(b, y)
            losses = n3_regression_loss(out, y, cfg)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run += float(losses["loss"].detach())
            steps += 1
        met = evaluate(model, valid, args.batch_size, device,
                       max_samples=(args.batch_size * 2 if args.smoke else None))
        row = {"epoch": ep, "train_loss": run / max(steps, 1),
               "elapsed_sec": round(time.time() - t0, 1), **met}
        hist.append(row)
        print(json.dumps(row), flush=True)
        if best is None or met["MAE"] < best["MAE"]:
            best = dict(met)
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "seed": args.seed,
                        "epoch": ep, "valid": met}, args.out / "best.pt")
            (args.out / "best_valid_metrics.json").write_text(
                json.dumps({"epoch": ep, **met}, indent=2), encoding="utf-8")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    if not (args.out / "best.pt").is_file():
        raise RuntimeError("missing best.pt - run is not a valid result")
    final = {"dataset": "CH-SIMS_v2", "tag": args.tag, "seed": args.seed, "best_valid": best,
             "status": "SMOKE_PASS" if args.smoke else "TRAIN_COMPLETE", "test_read": False,
             "elapsed_sec": round(time.time() - t0, 1),
             "trainable_parameters": model.count_trainable_parameters()}
    (args.out / "FINAL_RESULT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("FINAL_RESULT", json.dumps(final), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
