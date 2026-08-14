"""Train N3 on offline MELD features.

Uses ONLY the official train split. A seeded 10% slice of train is held
out as a monitor for early stopping (never official val/test). The
optimizer sees the remaining 90% only.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from n3_affect.config import N3TrainConfig
from n3_affect.losses import n3_total_loss
from n3_affect.meld_dataset import MELDFeatureDataset
from n3_affect.model import N3EmotionModel
from n3_affect.utility import EFFECT_ORDER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("train_meld")


def sample_ids(manifest: Path) -> set[str]:
    ids = set()
    with manifest.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            ids.add(r["sample_id"])
    return ids


def assert_train_only(train_manifest: Path) -> dict:
    train_ids = sample_ids(train_manifest)
    prefixes = {s.split("_dia")[0] for s in train_ids}
    if prefixes != {"train"}:
        raise SystemExit(f"train manifest contains non-train ids: {prefixes}")
    return {
        "train_n_ids": len(train_ids),
        "val_loaded": False,
        "test_loaded": False,
        "val_used_for": "not_loaded_during_training",
        "test_used_for": "not_loaded_during_training",
        "checkpoint_rule": "best_on_train_monitor_10pct_never_official_val_test",
    }


def class_weights_from_train(ds: MELDFeatureDataset, num_classes: int, device) -> torch.Tensor:
    labels = [int(r["label_id"]) for r in ds.rows]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> dict:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    n = 0
    mix_sum = None
    mix_n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        vad = batch.pop("vad")
        out = model(batch)
        use_vad = vad.abs().sum() > 0
        losses = n3_total_loss(out, labels, cfg, vad_targets=vad if use_vad else None)
        pred = out["logits"].argmax(-1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
        loss_sum += float(losses["loss"].item())
        n += 1
        if "mix_weights" in out:
            mix_sum = out["mix_weights"].detach().sum(dim=0) if mix_sum is None else mix_sum + out["mix_weights"].detach().sum(dim=0)
            mix_n += int(out["mix_weights"].size(0))
    mix_mean = None
    if mix_sum is not None and mix_n:
        mix_mean = (mix_sum / mix_n).detach().cpu()
        mix_mean = {name: float(mix_mean[i]) for i, name in enumerate(EFFECT_ORDER)}
    return {"loss": loss_sum / max(n, 1), "acc": correct / max(total, 1), "n": total, "mix_mean": mix_mean}


def copy_code_snapshot(code_dir: Path, sources: list[Path]) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if src.is_file():
            shutil.copy2(src, code_dir / src.name)


def tagged_path(path: Path, tag: str) -> Path:
    """Append .tag to the filename so run 2 does not overwrite run 1."""
    if not tag:
        return path
    return path.with_name(path.name + f".{tag}")


def infer_text_dim(feature_dir: Path) -> int:
    for p in sorted(feature_dir.glob("*.pt")):
        if p.name == "extraction_report.json":
            continue
        feat = torch.load(p, map_location="cpu", weights_only=True)
        return int(feat["T_t"].shape[-1])
    raise SystemExit(f"no feature files in {feature_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_PKG / "configs" / "n3_train_v1.json")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--train-features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-tag", type=str, default="", help="suffix appended to output filenames, e.g. 2 -> last.pt.2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-epochs", type=int, default=4)
    ap.add_argument("--monitor-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=-1.0, help="if >=0, override config dropout")
    ap.add_argument("--mix-lr-mult", type=float, default=3.0, help="learning-rate multiplier for mix_logits")
    ap.add_argument("--mix-tau", type=float, default=1.0)
    ap.add_argument("--mix-kl", type=float, default=0.0)
    ap.add_argument("--mix-peak", type=float, default=0.05, help="penalty if any mix weight exceeds mix-peak-cap")
    ap.add_argument("--mix-peak-cap", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, limit train steps per epoch")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_path = tagged_path(args.out_dir / "logs" / "pipeline.log", args.run_tag)
    logger = setup_logging(log_path)
    logger.info(f"train_meld start out_dir={args.out_dir}")
    leak = assert_train_only(args.train_manifest)
    logger.info(f"leakage_guard={leak}")
    logger.info("val and test are not loaded in this script")

    cfg = N3TrainConfig.from_json(args.config)
    cfg.text_tower = "composer_n3"
    cfg.max_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    if args.dropout >= 0:
        cfg.dropout = args.dropout
    cfg.seed = args.seed
    cfg.mix_tau = args.mix_tau
    cfg.mix_kl_weight = args.mix_kl
    cfg.mix_peak_weight = args.mix_peak
    cfg.mix_peak_cap = args.mix_peak_cap
    cfg.text_dim = infer_text_dim(args.train_features)
    cfg.validate()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ds = MELDFeatureDataset(args.train_manifest, args.train_features)
    if len(full_ds) == 0:
        raise SystemExit("no train features")
    idx = list(range(len(full_ds)))
    rng = random.Random(cfg.seed)
    rng.shuffle(idx)
    if args.monitor_frac <= 0:
        n_mon = 0
        fit_idx = idx
        mon_idx = []
    else:
        n_mon = max(1, int(round(len(idx) * args.monitor_frac)))
        mon_idx = idx[:n_mon]
        fit_idx = idx[n_mon:]
    fit_ds = full_ds.subset(fit_idx)
    mon_ds = full_ds.subset(mon_idx) if mon_idx else None
    logger.info(
        f"train_n={len(full_ds)} fit_n={len(fit_ds)} monitor_n={0 if mon_ds is None else len(mon_ds)} "
        f"device={device} text_dim={cfg.text_dim} mix_lr_mult={args.mix_lr_mult}"
    )
    logger.info("official val/test are not loaded; monitor is a slice of train or disabled")

    train_loader = DataLoader(
        fit_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    mon_loader = None
    if mon_ds is not None:
        mon_loader = DataLoader(
            mon_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

    model = N3EmotionModel(cfg).to(device)
    n_params = model.count_trainable_parameters()
    logger.info(f"trainable_parameters={n_params} budget={cfg.parameter_budget}")
    if n_params > cfg.parameter_budget:
        logger.warning(f"trainable params {n_params} exceed budget {cfg.parameter_budget}")
    class_weight = class_weights_from_train(fit_ds, cfg.num_classes, device)
    logger.info(f"class_weight_from_fit_split={class_weight.tolist()}")
    mix_params = list(model.utility.mix_parameters())
    mix_ids = {id(p) for p in mix_params}
    base_params = [p for p in model.parameters() if id(p) not in mix_ids]
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": mix_params, "lr": cfg.lr * args.mix_lr_mult, "weight_decay": 0.0},
        ],
    )
    history = []
    last_path = tagged_path(args.out_dir / "checkpoints" / "last.pt", args.run_tag)
    best_path = tagged_path(args.out_dir / "checkpoints" / "best.pt", args.run_tag)
    last_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_mon = float("inf")
    bad = 0
    best_row = None
    logger.info(
        "init_mix_weights="
        + json.dumps({k: round(v, 4) for k, v in model.utility.mixing_weights_dict().items()})
    )

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        correct = total = n = 0
        for step, batch in enumerate(train_loader):
            if args.smoke and step >= args.smoke:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            vad = batch.pop("vad")
            out = model(batch)
            use_vad = vad.abs().sum() > 0
            losses = n3_total_loss(
                out,
                labels,
                cfg,
                vad_targets=vad if use_vad else None,
                class_weight=class_weight,
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(losses["loss"].item())
            pred = out["logits"].argmax(-1)
            correct += int((pred == labels).sum().item())
            total += labels.numel()
            n += 1
        train_loss = running / max(n, 1)
        train_acc = correct / max(total, 1)
        if mon_loader is not None:
            mon_m = evaluate(model, mon_loader, device, cfg)
        else:
            mon_m = {"loss": train_loss, "acc": train_acc, "n": total, "mix_mean": None}
        mix_now = model.utility.mixing_weights_dict()
        mix_sample = mon_m.get("mix_mean") or mix_now
        hist_w = float(torch.sigmoid(model.gate.hist_logit).detach().cpu())
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "monitor_loss": mon_m["loss"],
            "monitor_acc": mon_m["acc"],
            "mix_weights_global": mix_now,
            "mix_weights": mix_sample,
            "hist_mix_weight": hist_w,
        }
        history.append(row)
        logger.info(str(row))
        logger.info("mix_weights_sample=" + json.dumps({k: round(v, 4) for k, v in mix_sample.items()}))
        ck = {
            "model": model.state_dict(),
            "cfg": cfg.to_dict(),
            "epoch": epoch,
            "mix_weights": mix_sample,
            "mix_weights_global": mix_now,
            "hist_mix_weight": hist_w,
            "effect_order": list(EFFECT_ORDER),
            "monitor": {k: v for k, v in mon_m.items() if k != "mix_mean"},
        }
        torch.save(ck, last_path)
        if mon_loader is None:
            torch.save(ck, best_path)
            best_row = row
            logger.info(f"saved_full_train {best_path} epoch={epoch} train_loss={train_loss:.4f}")
        else:
            improved = (mon_m["acc"] > best_acc + 1e-6) or (
                abs(mon_m["acc"] - best_acc) <= 1e-6 and mon_m["loss"] < best_mon
            )
            if improved:
                best_acc = mon_m["acc"]
                best_mon = mon_m["loss"]
                bad = 0
                torch.save(ck, best_path)
                best_row = row
                logger.info(
                    f"saved_best {best_path} monitor_acc={best_acc:.4f} monitor_loss={best_mon:.4f}"
                )
            else:
                bad += 1
                if epoch + 1 >= args.min_epochs and bad >= args.patience:
                    logger.info(f"early_stop epoch={epoch} patience={args.patience}")
                    break

    card = {
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_monitor": best_row,
        "train_n": len(full_ds),
        "fit_n": len(fit_ds),
        "monitor_n": 0 if mon_ds is None else len(mon_ds),
        "mix_lr_mult": args.mix_lr_mult,
        "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "device": str(device),
        "text_tower": cfg.text_tower,
        "text_dim": cfg.text_dim,
        "trainable_parameters": n_params,
        "parameter_budget": cfg.parameter_budget,
        "qwen_path": "/data/shared/qwen/Qwen3-Omni-30B-A3B-Instruct",
        "leakage_guard": leak,
        "class_weight_from_fit_split": class_weight.detach().cpu().tolist(),
        "effect_order": list(EFFECT_ORDER),
        "final_mix_weights": model.utility.mixing_weights_dict(),
        "final_hist_mix_weight": float(torch.sigmoid(model.gate.hist_logit).detach().cpu()),
        "run_tag": args.run_tag or "1",
        "note": (
            "Qwen thinker native 2048-d text + real 3-history A/V; "
            "19 measured marginal effects (current Möbius, history unimodal, temporal) "
            "mixed by learned softmax weights injected as a residual; "
            "optimizer sees 90% of official train; 10% train slice is monitor/no_grad "
            "for early stopping; official val/test never loaded during training."
        ),
    }
    (tagged_path(args.out_dir / "train_card.json", args.run_tag)).write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (tagged_path(args.out_dir / "config_snapshot.json", args.run_tag)).write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    code_sources = [
        _PKG / "n3_affect" / "config.py",
        _PKG / "n3_affect" / "model.py",
        _PKG / "n3_affect" / "encoders.py",
        _PKG / "n3_affect" / "relation.py",
        _PKG / "n3_affect" / "gating.py",
        _PKG / "n3_affect" / "utility.py",
        _PKG / "n3_affect" / "losses.py",
        _PKG / "n3_affect" / "meld_dataset.py",
        _PKG / "n3_affect" / "extract_meld_features.py",
        _PKG / "n3_affect" / "train_meld.py",
        _PKG / "n3_affect" / "eval_meld.py",
        _PKG / "n3_affect" / "generate_meld_manifests.py",
    ]
    copy_code_snapshot(tagged_path(args.out_dir / "code", args.run_tag), code_sources)
    logger.info(f"TRAIN_DONE {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
