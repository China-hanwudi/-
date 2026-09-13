"""Train one frozen A3 model on the formal HCAM IEMOCAP train/dev split.

The optimizer uses only Sessions 2-4 and validation uses only Session 1.
Session 5 is never accepted, named, or opened by this process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
except ModuleNotFoundError:
    def accuracy_score(y, p):
        return float(np.mean(np.asarray(y) == np.asarray(p))) if y else 0.0
    def confusion_matrix(y, p, labels):
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for a, b in zip(y, p):
            if int(a) in labels and int(b) in labels: cm[int(a), int(b)] += 1
        return cm
    def f1_score(y, p, average="weighted", zero_division=0):
        cm = confusion_matrix(y, p, list(range(4)))
        tp = np.diag(cm).astype(float); sup = cm.sum(1).astype(float); pred = cm.sum(0).astype(float)
        pr = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
        rc = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
        f = np.divide(2*pr*rc, pr+rc, out=np.zeros_like(tp), where=(pr+rc)>0)
        return float((f*sup).sum()/max(sup.sum(),1)) if average == "weighted" else float(f.mean())
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from n3_affect.config import N3TrainConfig
from n3_affect.iemocap_dataset import IEMOCAPFeatureDataset
from n3_affect.iemocap_folds import FOLD_SESSIONS, validate_fold_role
from n3_affect.losses import n3_total_loss
from n3_affect.model import N3EmotionModel
from n3_affect.utility import EFFECT_ORDER


LABELS = ("angry", "happy", "sad", "neutral")
FROZEN_SEEDS = (17, 29, 43, 71, 101)
FROZEN_TRAIN_MANIFEST = Path(
    "/data/shared/iemocap_a3_fivefold_seed17_completion_20260827/manifests/fold5/train.jsonl"
)
FROZEN_DEV_MANIFEST = Path(
    "/data/shared/iemocap_a3_fivefold_seed17_completion_20260827/manifests/fold5/dev.jsonl"
)
FROZEN_FEATURE_ROOT = Path(
    "/data/shared/features/iemocap_xiao_n3/iemocap_20260814_134849/"
    "a3_folds2_5_seed17_completion_20260827/fold5"
)
FROZEN_TRAIN_SHA256 = "0b2469da7ba958f56141665f9b3119622c543b57eea910aec3ee0b6c6a74356d"
FROZEN_DEV_SHA256 = "0ae1bbc425555d01f5214e875dc1ac548c6b8b94108f3f439cc6be6d00a231ce"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("train_iemocap_xiao")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_json_evidence(path: Path, payload: Any, description: str) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing {description} does not match; refusing to overwrite {path}")
    else:
        atomic_json(path, payload)


def code_inventory() -> dict[str, str]:
    names = (
        "config.py", "encoders.py", "gating.py", "iemocap_dataset.py", "losses.py",
        "iemocap_folds.py", "model.py", "relation.py", "train_iemocap_xiao.py", "utility.py",
    )
    return {name: sha256_file(_ROOT / "n3_affect" / name) for name in names}


def runtime_inventory() -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "torch", "transformers", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def feature_receipt_inventory(feature_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    contract_values: set[str] = set()
    for role in ("train", "dev"):
        receipts = sorted((feature_root / role / "receipts").glob("*.json"))
        counts[role] = len(receipts)
        for path in receipts:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            contract_values.add(str(receipt.get("contract_sha256", "")))
            line = "\t".join(
                [role, str(receipt["sample_id"]), str(receipt["sha256"]), str(receipt["contract_sha256"])]
            )
            digest.update((line + "\n").encode("utf-8"))
    if len(contract_values) != 1 or "" in contract_values:
        raise RuntimeError(f"feature receipts have inconsistent contracts: {sorted(contract_values)}")
    return {
        "counts": counts,
        "digest_sha256": digest.hexdigest(),
        "feature_contract_sha256": next(iter(contract_values)),
    }


def write_checkpoint_leaf(
    checkpoint_dir: Path,
    payload: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    epoch = int(payload["epoch"])
    leaf = checkpoint_dir / "epochs" / f"epoch_{epoch:04d}.pt"
    receipt = checkpoint_dir / "epochs" / f"epoch_{epoch:04d}.json"
    atomic_torch_save(leaf, payload)
    leaf_receipt = {
        "epoch": epoch,
        "checkpoint": str(Path("epochs") / leaf.name),
        "checkpoint_sha256": sha256_file(leaf),
        "contract_sha256": contract_sha256,
    }
    atomic_json(receipt, leaf_receipt)
    return leaf_receipt


def load_checkpoint_pointer(
    pointer_path: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    contract_sha = sha256_json(contract)
    if pointer.get("contract_sha256") != contract_sha:
        raise RuntimeError(f"checkpoint pointer contract mismatch: {pointer_path}")
    checkpoint_path = (pointer_path.parent / str(pointer["checkpoint"])).resolve()
    allowed_root = (pointer_path.parent / "epochs").resolve()
    if not checkpoint_path.resolve().is_relative_to(allowed_root):
        raise RuntimeError(f"checkpoint pointer escapes epoch directory: {checkpoint_path}")
    receipt_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"checkpoint leaf/receipt missing for {pointer_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != pointer or sha256_file(checkpoint_path) != pointer.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint leaf receipt/hash mismatch for {pointer_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != contract or int(checkpoint.get("epoch", -1)) != int(pointer["epoch"]):
        raise RuntimeError(f"checkpoint payload contract/epoch mismatch for {pointer_path}")
    return checkpoint, pointer


def class_weights(dataset: IEMOCAPFeatureDataset, device: torch.device) -> torch.Tensor:
    counts = np.bincount([int(row["label"]) for row in dataset.rows], minlength=4).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"train split lacks a class: counts={counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metrics_from_predictions(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(4))).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: N3EmotionModel,
    dataset: IEMOCAPFeatureDataset,
    device: torch.device,
    cfg: N3TrainConfig,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    labels_all: list[int] = []
    predictions_all: list[int] = []
    loss_sum = 0.0
    batches = 0
    mix_sum = None
    mix_count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = batch.pop("label")
        vad = batch.pop("vad")
        outputs = model(batch)
        losses = n3_total_loss(outputs, labels, cfg, vad_targets=vad if vad.abs().sum() > 0 else None)
        predictions = outputs["logits"].argmax(dim=-1)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        loss_sum += float(losses["loss"].item())
        batches += 1
        weights = outputs.get("mix_weights")
        if weights is not None:
            mix_sum = weights.detach().sum(dim=0) if mix_sum is None else mix_sum + weights.detach().sum(dim=0)
            mix_count += int(weights.size(0))
    result = metrics_from_predictions(labels_all, predictions_all)
    result["loss"] = loss_sum / max(batches, 1)
    if mix_sum is not None and mix_count:
        mean = (mix_sum / mix_count).cpu().tolist()
        result["mix_weights"] = {name: float(mean[i]) for i, name in enumerate(EFFECT_ORDER)}
    return result


def restore_rng(checkpoint: dict[str, Any]) -> None:
    if "python_rng" in checkpoint:
        random.setstate(checkpoint["python_rng"])
    if "numpy_rng" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng"])
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def checkpoint_payload(
    *, trial_id: str, model: N3EmotionModel, optimizer: torch.optim.Optimizer,
    cfg: N3TrainConfig, epoch: int, best_weighted_f1: float, best_macro_f1: float,
    best_loss: float, bad_epochs: int, history: list[dict[str, Any]], contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg.to_dict(),
        "epoch": epoch,
        "best_weighted_f1": best_weighted_f1,
        "best_macro_f1": best_macro_f1,
        "best_loss": best_loss,
        "bad_epochs": bad_epochs,
        "history": history,
        "contract": contract,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(5,), required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--mix-lr-mult", type=float, default=6.0)
    parser.add_argument("--mix-tau", type=float, default=0.75)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, choices=FROZEN_SEEDS, required=True)
    args = parser.parse_args()

    if "test" in args.train_manifest.name.lower() or "test" in args.dev_manifest.name.lower():
        raise RuntimeError("test manifest is forbidden for training/selection")
    # Paths are configurable for the A100 host; record hashes in the run contract.
    if not args.train_manifest.is_file() or not args.dev_manifest.is_file():
        raise RuntimeError("train/dev manifest missing")
    if args.out_dir.exists():
        raise RuntimeError(f"seed output already exists; retry/resume is forbidden: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "logs" / "train.log")
    set_seed(args.seed)

    train_dataset = IEMOCAPFeatureDataset(args.train_manifest, args.feature_root / "train", strict=True)
    dev_dataset = IEMOCAPFeatureDataset(args.dev_manifest, args.feature_root / "dev", strict=True)
    validate_fold_role(train_dataset.rows, fold=args.fold, role="train")
    validate_fold_role(dev_dataset.rows, fold=args.fold, role="dev")
    overlap = train_dataset.sample_ids & dev_dataset.sample_ids
    if overlap:
        raise RuntimeError(f"train/dev overlap: {sorted(overlap)[:10]}")

    cfg = N3TrainConfig.from_json(args.config)
    cfg.text_tower = "composer_n3"
    cfg.num_classes = 4
    cfg.emotion_label_order = LABELS
    cfg.text_dim = 2048
    cfg.audio_dim = 1536
    cfg.video_dim = 768
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.mix_tau = args.mix_tau
    cfg.dropout = args.dropout
    cfg.seed = args.seed
    cfg.vad_loss_weight = 0.0
    cfg.validate()

    if not torch.cuda.is_available():
        raise RuntimeError("authorized A3 training requires the checked idle CUDA GPU")
    device = torch.device("cuda")
    model = N3EmotionModel(cfg).to(device)
    mix_parameters = list(model.utility.mix_parameters())
    mix_ids = {id(parameter) for parameter in mix_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in mix_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": cfg.lr},
            {"params": mix_parameters, "lr": cfg.lr * args.mix_lr_mult},
        ],
        weight_decay=cfg.weight_decay,
    )
    weights = class_weights(train_dataset, device)
    extraction_contract_path = args.feature_root / "EXTRACTION_CONTRACT.json"
    extraction_verify_path = args.feature_root / "VERIFY_SUMMARY.json"
    if not extraction_contract_path.is_file() or not extraction_verify_path.is_file():
        raise RuntimeError("feature extraction contract or fresh verify summary is missing")
    extraction_verify = json.loads(extraction_verify_path.read_text(encoding="utf-8"))
    if extraction_verify.get("status") != "PASS":
        raise RuntimeError("feature leaf verification is not PASS")
    extraction_contract = json.loads(extraction_contract_path.read_text(encoding="utf-8"))
    extraction_contract_sha256 = sha256_file(extraction_contract_path)
    expected_manifest_bindings = {
        "fold": args.fold,
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "dev_manifest_sha256": sha256_file(args.dev_manifest),
    }
    for key, expected in expected_manifest_bindings.items():
        if extraction_contract.get(key) != expected:
            raise RuntimeError(
                f"feature extraction contract {key} mismatch: "
                f"expected={expected!r} observed={extraction_contract.get(key)!r}"
            )
    if (
        extraction_verify.get("fold") != args.fold
        or extraction_verify.get("contract_sha256") != extraction_contract_sha256
    ):
        raise RuntimeError("feature verification is not bound to this fold extraction contract")
    receipt_inventory = feature_receipt_inventory(args.feature_root)
    if receipt_inventory["counts"] != {"train": len(train_dataset), "dev": len(dev_dataset)}:
        raise RuntimeError(
            f"feature receipt counts do not match datasets: {receipt_inventory['counts']} "
            f"vs train={len(train_dataset)} dev={len(dev_dataset)}"
        )
    if receipt_inventory["feature_contract_sha256"] != extraction_contract_sha256:
        raise RuntimeError("feature receipt inventory is not bound to the extraction contract")
    contract = {
        "trial_id": args.trial_id,
        "dataset": "IEMOCAP",
        "fold": args.fold,
        "seed": args.seed,
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "dev_manifest": str(args.dev_manifest),
        "dev_manifest_sha256": sha256_file(args.dev_manifest),
        "feature_root": str(args.feature_root),
        "feature_extraction_contract_sha256": extraction_contract_sha256,
        "feature_verify_summary_sha256": sha256_file(extraction_verify_path),
        "feature_receipt_inventory": receipt_inventory,
        "adapted_code_sha256": code_inventory(),
        "training_runtime": runtime_inventory(),
        "config_source_sha256": sha256_file(args.config),
        "train_n": len(train_dataset),
        "dev_n": len(dev_dataset),
        "train_counts": dict(sorted(Counter(LABELS[int(row["label"])] for row in train_dataset.rows).items())),
        "dev_counts": dict(sorted(Counter(LABELS[int(row["label"])] for row in dev_dataset.rows).items())),
        "train_sessions": sorted(train_dataset.sessions),
        "dev_sessions": sorted(dev_dataset.sessions),
        "test_manifest_opened": False,
        "model_framework": "Xiao ComposerN3 six-stream N3",
        "feature_family": "Qwen text + MFCC/Mel audio + audited-ROI RGB statistics video",
        "hyperparameters": {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "mix_lr_mult": args.mix_lr_mult,
            "mix_tau": cfg.mix_tau,
            "dropout": cfg.dropout,
            "max_epochs": cfg.max_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "limitations": (
            "Formal fixed-split training leaf only: Sessions 2-4 train, Session 1 validation, "
            "and no Session 5 access. This is a locked post-development benchmark re-evaluation, "
            "not pristine unseen validation."
        ),
    }
    bind_json_evidence(args.out_dir / "RUN_CONTRACT.json", contract, "run contract")
    bind_json_evidence(args.out_dir / "config_snapshot.json", cfg.to_dict(), "config snapshot")
    logger.info("contract=%s", json.dumps(contract, ensure_ascii=False))
    logger.info(
        "device=%s trainable_parameters=%d class_weights=%s",
        device, model.count_trainable_parameters(), weights.tolist(),
    )

    checkpoint_dir = args.out_dir / "checkpoints"
    last_pointer_path = checkpoint_dir / "last.json"
    best_pointer_path = checkpoint_dir / "best.json"
    final_result_path = args.out_dir / "FINAL_RESULT.json"
    start_epoch = 0
    best_weighted_f1 = -1.0
    best_macro_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    contract_sha256 = sha256_json(contract)
    already_stopped = start_epoch >= args.min_epochs and bad_epochs >= args.patience
    if already_stopped:
        logger.info("resume state already satisfies early-stop condition; proceeding to finalization")
    epoch_range = range(start_epoch, cfg.max_epochs) if not already_stopped else ()
    for epoch in epoch_range:
        epoch_started = time.time()
        generator = torch.Generator().manual_seed(cfg.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        model.train()
        labels_epoch: list[int] = []
        predictions_epoch: list[int] = []
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("label")
            vad = batch.pop("vad")
            outputs = model(batch)
            losses = n3_total_loss(outputs, labels, cfg, vad_targets=vad if vad.abs().sum() > 0 else None, class_weight=weights)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            labels_epoch.extend(labels.detach().cpu().tolist())
            predictions_epoch.extend(outputs["logits"].argmax(dim=-1).detach().cpu().tolist())
            loss_sum += float(losses["loss"].item())
            batches += 1

        train_metrics = metrics_from_predictions(labels_epoch, predictions_epoch)
        train_metrics["loss"] = loss_sum / max(batches, 1)
        dev_metrics = evaluate(model, dev_dataset, device, cfg)
        improved = (
            dev_metrics["f1_weighted"] > best_weighted_f1 + 1e-12
            or (
                abs(dev_metrics["f1_weighted"] - best_weighted_f1) <= 1e-12
                and (
                    dev_metrics["f1_macro"] > best_macro_f1 + 1e-12
                    or (
                        abs(dev_metrics["f1_macro"] - best_macro_f1) <= 1e-12
                        and dev_metrics["loss"] < best_loss
                    )
                )
            )
        )
        if improved:
            best_weighted_f1 = float(dev_metrics["f1_weighted"])
            best_macro_f1 = float(dev_metrics["f1_macro"])
            best_loss = float(dev_metrics["loss"])
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "duration_sec": time.time() - epoch_started,
            "train": train_metrics,
            "dev": dev_metrics,
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        history.append(row)
        payload = checkpoint_payload(
            trial_id=args.trial_id, model=model, optimizer=optimizer, cfg=cfg, epoch=epoch,
            best_weighted_f1=best_weighted_f1, best_macro_f1=best_macro_f1,
            best_loss=best_loss, bad_epochs=bad_epochs, history=history, contract=contract,
        )
        leaf_receipt = write_checkpoint_leaf(checkpoint_dir, payload, contract_sha256)
        if improved:
            # Publish best before last. If interrupted between pointers, resume
            # replays this epoch from the prior fully committed last pointer.
            atomic_json(best_pointer_path, leaf_receipt)
        atomic_json(last_pointer_path, leaf_receipt)
        atomic_json(args.out_dir / "metrics" / "history.json", history)
        logger.info(
            "epoch=%d train_loss=%.4f dev_loss=%.4f dev_acc=%.4f dev_f1w=%.4f dev_f1m=%.4f improved=%s bad=%d",
            epoch, train_metrics["loss"], dev_metrics["loss"], dev_metrics["accuracy"],
            dev_metrics["f1_weighted"], dev_metrics["f1_macro"], improved, bad_epochs,
        )
        if epoch + 1 >= args.min_epochs and bad_epochs >= args.patience:
            logger.info("early_stop epoch=%d", epoch)
            break

    if not best_pointer_path.is_file():
        raise RuntimeError("best checkpoint pointer is missing")
    best_checkpoint, best_pointer = load_checkpoint_pointer(best_pointer_path, contract)
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)
    final_dev = evaluate(model, dev_dataset, device, cfg)
    result = {
        "status": "PASS",
        "trial_id": args.trial_id,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_checkpoint": str(Path("checkpoints") / str(best_pointer["checkpoint"])),
        "best_checkpoint_sha256": best_pointer["checkpoint_sha256"],
        "dev_metrics": final_dev,
        "contract": contract,
        "test_manifest_opened": False,
        "claim_boundary": (
            f"A3 HCAM-split training/validation leaf for preregistered seed {args.seed}; "
            "Session 5 remains sealed and no final generalization claim is made here."
        ),
    }
    atomic_json(final_result_path, result)
    logger.info("TRAIN_COMPLETE %s", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
