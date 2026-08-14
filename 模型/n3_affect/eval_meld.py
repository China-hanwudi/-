"""Evaluate a saved N3 checkpoint on an official MELD split (no gradients).

Outputs metrics JSON, report text, and a per-split log file.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from n3_affect.config import N3TrainConfig
from n3_affect.losses import n3_total_loss
from n3_affect.meld_dataset import MELDFeatureDataset
from n3_affect.model import N3EmotionModel
from n3_affect.train_meld import tagged_path
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
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("eval_meld")


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> dict:
    model.eval()
    total_loss = emo_loss = 0.0
    n = 0
    all_labels, all_preds = [], []
    mix_sum = None
    mix_n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        vad = batch.pop("vad")
        out = model(batch)
        use_vad = vad.abs().sum() > 0
        losses = n3_total_loss(out, labels, cfg, vad_targets=vad if use_vad else None)
        total_loss += float(losses["loss"].item())
        emo_loss += float(losses["emotion_loss"].item())
        n += 1
        preds = out["logits"].argmax(-1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        if "mix_weights" in out:
            mix_sum = out["mix_weights"].detach().sum(dim=0) if mix_sum is None else mix_sum + out["mix_weights"].detach().sum(dim=0)
            mix_n += int(out["mix_weights"].size(0))
    report = classification_report(
        all_labels, all_preds, labels=list(range(cfg.num_classes)), target_names=list(cfg.emotion_label_order), output_dict=True, zero_division=0
    )
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mix_mean = None
    if mix_sum is not None and mix_n:
        mix_mean = (mix_sum / mix_n).cpu().tolist()
    return {
        "loss": total_loss / max(n, 1),
        "emotion_loss": emo_loss / max(n, 1),
        "accuracy": report["accuracy"],
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": report,
        "n": len(all_labels),
        "mix_mean": mix_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-tag", type=str, default="", help="suffix appended to output filenames, e.g. 2 -> train_report.txt.2")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    log_path = tagged_path(args.out_dir / f"eval_{args.split}.log", args.run_tag)
    logger = setup_logging(log_path)
    logger.info(f"eval_{args.split} start checkpoint={args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = N3TrainConfig(**ck["cfg"])
    model = N3EmotionModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    mix_weights = model.utility.mixing_weights_dict()
    hist_mix_weight = float(torch.sigmoid(model.gate.hist_logit).detach().cpu())

    ds = MELDFeatureDataset(args.manifest, args.features)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    logger.info(f"eval {args.split} n={len(ds)} checkpoint={args.checkpoint}")
    metrics = evaluate(model, loader, device, cfg)
    if metrics.get("mix_mean"):
        mix_weights = {name: float(metrics["mix_mean"][i]) for i, name in enumerate(EFFECT_ORDER)}
    logger.info("mix_weights=" + json.dumps({k: round(v, 4) for k, v in mix_weights.items()}))
    logger.info(f"hist_mix_weight={hist_mix_weight:.4f}")

    card = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": cfg.to_dict(),
        "mix_weights": mix_weights,
        "hist_mix_weight": hist_mix_weight,
        "mix_mean": metrics.get("mix_mean"),
        "metrics": {
            "loss": metrics["loss"],
            "emotion_loss": metrics["emotion_loss"],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "n": metrics["n"],
        },
        "per_class": metrics["per_class"],
    }
    (tagged_path(args.out_dir / f"{args.split}_metrics.json", args.run_tag)).write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_text = "MELD {} evaluation\n".format(args.split)
    report_text += "checkpoint: {}\n".format(args.checkpoint)
    report_text += "samples: {}\n".format(metrics['n'])
    report_text += "accuracy: {:.4f}\n".format(metrics['accuracy'])
    report_text += "f1_macro: {:.4f}\n".format(metrics['f1_macro'])
    report_text += "f1_weighted: {:.4f}\n".format(metrics['f1_weighted'])
    report_text += "loss: {:.4f}\n".format(metrics['loss'])
    report_text += "hist_mix_weight: {:.4f}\n".format(hist_mix_weight)
    report_text += "mix_weights:\n"
    for name, value in mix_weights.items():
        report_text += "  {}: {:.4f}\n".format(name, value)
    report_text += "\nper-class:\n"
    for lab in cfg.emotion_label_order:
        d = metrics["per_class"][lab]
        report_text += "  {}: precision={:.4f}, recall={:.4f}, f1={:.4f}, support={}\n".format(
            lab, d['precision'], d['recall'], d['f1-score'], int(d['support'])
        )
    (tagged_path(args.out_dir / "{}_report.txt".format(args.split), args.run_tag)).write_text(report_text, encoding="utf-8")

    logger.info(f"EVAL_DONE {args.split} acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())