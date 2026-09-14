"""Full-metric evaluation entry for the v10 exploratory mainline.

One invocation produces every metric the paper tables need:

* M3ED (7-class)  -> Weighted-F1, Macro-F1, Accuracy, Balanced Accuracy,
  per-class P/R/F1/support, Macro-P/R, MCC, top-2/3, NLL, multiclass Brier,
  ECE, confusion matrix.
* CMU-MOSEI (regression, raw [-3,3]) -> MAE, RMSE, Pearson, Spearman, CCC, R^2,
  Acc-2 / F1-2 under four named binarisation protocols, error by label quintile.
* CH-SIMS_v2 (regression, [-1,1], official clip + `y > 0` positive class) ->
  the same block, with an explicit clipped/unclipped pair.

Always writes ``eval_full.json`` and ``predictions.npz`` (ids / y_true / y_pred /
logits) so the statistics module can run paired bootstrap and Wilcoxon tests
without re-running the model.

The sealed test split is refused unless ``--allow-sealed-test`` is passed, and
a per-call receipt is written for the audit trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_DATA = Path(os.environ.get("DATA_ROOT", "data"))

DATASET_LAYOUT = {
    "m3ed": {
        "train": "M3ED/packed/train.pt",
        "valid": "M3ED/packed/valid.pt",
        "test": "M3ED/packed/test.pt",
    },
    "mosei": {
        "train": "MOSEI/packed/train.pt",
        "valid": "MOSEI/packed/valid.pt",
        "test": "MOSEI/packed/test.pt",
    },
    "chsims": {
        "train": "CH-SIMS_v2/v7_packed/train.pt",
        "valid": "CH-SIMS_v2/v7_packed/valid.pt",
        "dev": "CH-SIMS_v2/v7_packed/dev.pt",
        "test": "CH-SIMS_v2/v7_packed/SEALED_TEST_DO_NOT_READ/test.pt",
    },
}
SEALED_SPLITS = {"test"}
M3ED_CLASS_NAMES = ("Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_fingerprint(root: Path) -> dict:
    codes = sorted((root / "n3_affect").glob("*.py")) + [root / "train_sims_v7.py"]
    return {p.name: sha256_file(p)[:16] for p in codes if p.is_file()}


# ---------------------------------------------------------------------------
# dataset adapters
# ---------------------------------------------------------------------------
class ChSimsPacked:
    """CH-SIMS v2 has no modality mask and no conversational history."""

    def __init__(self, path: Path) -> None:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self.ids = [str(x) for x in raw.pop("ids")]
        self.data = {k: v for k, v in raw.items() if torch.is_tensor(v)}
        self.n = len(self.ids)
        assert self.data["label"].shape == (self.n,)
        assert self.data["unimodal_labels"].shape == (self.n, 3)
        self.path = Path(path)

    def batch(self, ix: torch.Tensor) -> tuple[dict, torch.Tensor]:
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


def build_split(dataset: str, path: Path):
    if dataset in ("m3ed", "mosei"):
        if dataset == "m3ed":
            from n3_affect.m3ed_dataset import M3EDPackedDataset as DS
        else:
            from n3_affect.mosei_dataset import MOSEIPackedDataset as DS
        ds = DS(path)
        return ds, ds.batch, [str(x) for x in ds.raw["ids"]] if "ids" in ds.raw else [str(i) for i in range(ds.n)]
    ds = ChSimsPacked(path)
    return ds, ds.batch, ds.ids


def build_model(dataset: str, ckpt: dict):
    cfg_dict = ckpt["cfg"]
    if dataset == "m3ed":
        from n3_affect.config import N3TrainConfig
        from n3_affect.model import N3EmotionModel
        cfg = N3TrainConfig(**{k: (tuple(v) if k == "emotion_label_order" else v)
                               for k, v in cfg_dict.items()})
        model = N3EmotionModel(cfg)
        return model, cfg, "classification"
    from n3_affect.regression_config import N3RegressionConfig
    from n3_affect.regression_model import N3SentimentModel
    cfg = N3RegressionConfig(**cfg_dict)
    model = N3SentimentModel(cfg)
    return model, cfg, "regression"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run(dataset: str, model, split_obj, batch_fn, batch_size: int,
        device: torch.device, route_mode: str) -> dict:
    model.eval()
    model.current_epoch = 10 ** 6  # never inside a warm-up schedule at eval time
    n = split_obj.n
    all_logits: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    mech = {"use_history": 0.0, "accept_rate": [], "n_batches": 0}
    for start in range(0, n, batch_size):
        ix = torch.arange(start, min(start + batch_size, n))
        batch, y = batch_fn(ix)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch, route_mode=route_mode)
        if "logits" in out:
            all_logits.append(out["logits"].detach().float().cpu().numpy())
            all_pred.append(out["logits"].argmax(-1).detach().cpu().numpy())
        else:
            all_pred.append(out["prediction"].detach().float().reshape(-1).cpu().numpy())
        all_true.append(y.detach().float().reshape(-1).cpu().numpy())
        if out.get("use_history") is not None:
            mech["use_history"] += float(out["use_history"].detach().float().sum())
        if out.get("accept_rate") is not None:
            ar = out["accept_rate"]
            mech["accept_rate"].append(float(ar.detach().float().mean()) if torch.is_tensor(ar) else float(ar))
        mech["n_batches"] += 1
    mech["history_use_rate"] = mech.pop("use_history") / max(n, 1)
    if mech["accept_rate"]:
        mech["accept_rate_mean"] = float(np.mean(mech["accept_rate"]))
    mech.pop("accept_rate")
    mech.pop("n_batches")
    return {
        "logits": np.concatenate(all_logits) if all_logits else None,
        "pred": np.concatenate(all_pred),
        "true": np.concatenate(all_true),
        "mech": mech,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Full-metric v9 evaluation")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_LAYOUT))
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--route-mode", default="hard-safe")
    ap.add_argument("--allow-sealed-test", action="store_true")
    args = ap.parse_args(argv)

    layout = DATASET_LAYOUT[args.dataset]
    if args.split not in layout:
        raise SystemExit(f"{args.dataset} has no split '{args.split}' (have {sorted(layout)})")
    if args.split in SEALED_SPLITS and not args.allow_sealed_test:
        raise SystemExit(
            "refusing to open a sealed split without --allow-sealed-test "
            "(one-shot frozen evaluation only)")
    data_path = args.data / layout[args.split]
    if not data_path.is_file():
        raise SystemExit(f"missing data file: {data_path}")

    sys.path.insert(0, str(args.source.resolve()))
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, cfg, task = build_model(args.dataset, ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    split_obj, batch_fn, ids = build_split(args.dataset, data_path)
    t0 = time.time()
    res = run(args.dataset, model, split_obj, batch_fn, args.batch_size, device, args.route_mode)
    elapsed = time.time() - t0

    from n3_affect import metrics as M

    report: dict = {
        "dataset": args.dataset,
        "split": args.split,
        "route_mode": args.route_mode,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_seed": ckpt.get("seed"),
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "device": str(device),
        "n": int(res["true"].size),
        "elapsed_sec": round(elapsed, 2),
        "history_use_rate": res["mech"].get("history_use_rate"),
        "accept_rate_mean": res["mech"].get("accept_rate_mean"),
        "package_fingerprint": package_fingerprint(args.source),
        "test_split_opened": bool(args.split in SEALED_SPLITS),
    }
    if task == "classification":
        report["task"] = "7_class_emotion"
        report["label_protocol"] = (
            "M3ED packed label ids 0..6; official order from data_audit.json "
            "label_field=EmoAnnotation.final_main_emo -> "
            f"{list(M3ED_CLASS_NAMES)}")
        report["metrics"] = M.classification_report(
            res["logits"], res["true"].astype(np.int64), cfg.num_classes,
            class_names=M3ED_CLASS_NAMES)
        report["selection_metric_order"] = ["weighted_f1", "macro_f1", "accuracy"]
    else:
        report["task"] = cfg.task
        clip = None
        if args.dataset == "chsims":
            report["label_protocol"] = (
                "CH-SIMS v2 continuous sentiment in [-1,1]; predictions clipped to "
                "[-1,1] before MAE (official convention); Acc-2 positive class = y > 0 "
                "(a 0 label is NON-positive). MOSEI's >=0 convention is NOT applied.")
            clip = (-1.0, 1.0)
        else:
            report["label_protocol"] = (
                "CMU-MOSEI raw sentiment in [-3,3]; model trains on y/3 and reports "
                "3*u. Acc-2 headline uses the y > 0 threshold; the >=0 (has0), non0 "
                "and zero conventions are reported alongside for auditability.")
        report["target_scale"] = float(getattr(cfg, "target_scale", 1.0))
        report["metrics"] = M.regression_report(res["true"], res["pred"], clip=clip)
        report["selection_metric_order"] = ["mae"]
    report["mechanism"] = res["mech"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        args.out / "predictions.npz",
        ids=np.asarray(ids[: res["true"].size], dtype=object),
        y_true=res["true"], y_pred=res["pred"],
        logits=res["logits"] if res["logits"] is not None else np.zeros((0, 0)),
        route_mode=np.asarray([args.route_mode]),
    )
    if args.split in SEALED_SPLITS:
        (args.out / "TEST_EVALUATION_RECEIPT.json").write_text(json.dumps({
            "sealed_split_opened": True,
            "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "data_sha256": report["data_sha256"],
            "route_mode": args.route_mode,
            "purpose": "one-shot frozen test evaluation after valid-based model selection",
        }, indent=2), encoding="utf-8")
    head = {k: v for k, v in report["metrics"].items()
            if isinstance(v, (int, float, str))}
    print("EVAL_FULL", json.dumps({"dataset": args.dataset, "split": args.split,
                                   "route_mode": args.route_mode, **head}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
