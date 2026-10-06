"""Train model6 (UGF) on a packed train/valid split pair.

Only packed train.pt and valid.pt are read here; the sealed test split is
never opened (n6.data.open_split refuses any basename that is not exactly
train.pt / valid.pt, and nothing in this package references test).  Every
run writes an isolated directory containing ``best.pt``, ``FINAL_RESULT.json``,
``history.json`` and ``RUN_METADATA.json`` with data hashes, the per-file
code sha256 manifest of the n6 package, the resolved config and the utility
mode.

Checkpoint selection follows ``valid Weighted-F1 -> Macro-F1 -> Accuracy``
for cls and ``valid MAE -> RMSE -> Pearson`` (lower MAE better) for reg, with
the candidate being the uniform average of the last ``--swa-window`` epoch-end
weight sets once the window is full.
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import M6Config
from .data import (IndexedPackedDataset, PackedDataset, open_split,
                   split_sha256, TRAINABLE_SPLITS)
from .datasetspec import class_names_for, spec_for_pack
from .losses import (compute_class_weights, joint_supervision,
                     m1_view_loss, total_loss)
from .metrics import classification_report, regression_report
from .model import UGFModel

# Legacy constant kept for import compatibility with old driver scripts;
# new runs resolve names from the DatasetSpec (never from this fallback).
M3ED_CLASS_NAMES = ("Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear",
                    "Surprise")


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_manifest() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    return {p.name: sha(p) for p in sorted(root.glob("*.py")) if p.is_file()}


def load_history_utility_cache(path: str) -> Dict[int, Tuple[float, float]]:
    """Load global-indexed keep targets and weights for fit-only batches."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != "history_utility_cache_v1":
        raise SystemExit("unsupported history utility cache version")
    rows = payload.get("rows")
    if not isinstance(rows, dict):
        raise SystemExit("history utility cache lacks rows")
    out: Dict[int, Tuple[float, float]] = {}
    for key, row in rows.items():
        if not isinstance(row, dict):
            raise SystemExit("bad history utility cache row %r" % (key,))
        out[int(key)] = (float(row["target_keep"]), float(row.get("weight", 1.0)))
    return out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dialogue_manifest(path: Path, train: PackedDataset
                           ) -> Tuple[IndexedPackedDataset, IndexedPackedDataset, str]:
    """Load a fixed fit/inner-dev manifest without renumbering history slots."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "dialogue_fit_inner_dev_v1":
        raise SystemExit("unsupported dialogue manifest version")
    if payload.get("train_sha256") != split_sha256(train):
        raise SystemExit("dialogue manifest train hash does not match --data/train.pt")
    fit = payload.get("fit_indices")
    dev = payload.get("inner_dev_indices")
    if not isinstance(fit, list) or not isinstance(dev, list):
        raise SystemExit("dialogue manifest lacks fit_indices/inner_dev_indices")
    excluded = payload.get("excluded_indices", [])
    if not isinstance(excluded, list):
        raise SystemExit("dialogue manifest excluded_indices must be a list")
    fit_set, dev_set, excluded_set = set(fit), set(dev), set(excluded)
    if (fit_set & dev_set) or (fit_set & excluded_set) or (dev_set & excluded_set):
        raise SystemExit("dialogue manifest fit/dev/excluded indices must be disjoint")
    if fit_set | dev_set | excluded_set != set(range(train.n)):
        raise SystemExit("dialogue manifest fit/dev/excluded indices must cover train exactly once")
    if "ids" in train.raw:
        fit_dialogues = {str(train.raw["ids"][int(i)]).split("_utt", 1)[0] for i in fit}
        dev_dialogues = {str(train.raw["ids"][int(i)]).split("_utt", 1)[0] for i in dev}
        if fit_dialogues & dev_dialogues:
            raise SystemExit("dialogue manifest splits a dialogue across fit and inner_dev")
    manifest_hash = sha(path)
    return (IndexedPackedDataset(train, fit, "fit"),
            IndexedPackedDataset(train, dev, "inner_dev"), manifest_hash)


def series_of(uid) -> str:
    """Series id = field 1 of ``<A/B>_<series>_<ep>_<utt>`` (M3ED ids)."""
    parts = str(uid).split("_")
    return parts[1] if len(parts) > 1 else parts[0]


def average_state_dicts(snapshots) -> Dict[str, torch.Tensor]:
    snaps = list(snapshots)
    out: Dict[str, torch.Tensor] = {}
    n = float(len(snaps))
    for key in snaps[-1]:
        values = [s[key] for s in snaps]
        if values[0].is_floating_point():
            acc = values[0].clone()
            for v in values[1:]:
                acc.add_(v)
            out[key] = acc / n
        else:
            out[key] = values[-1].clone()
    return out


@torch.no_grad()
def evaluate(model, ds: PackedDataset, device, bs: int, cfg: M6Config,
             class_weight=None, max_samples: Optional[int] = None,
             series_ids: Optional[List[str]] = None):
    """Deploy-time evaluation on one split; returns (metrics, deployed logits).

    With ``series_ids`` (per-sample series strings, aligned to the split rows)
    the metrics additionally carry ``series_wf1_mean`` / ``series_wf1_min``
    (per-series weighted F1 over the valid split -- W6 selection signal).
    """
    from .metrics import per_group_weighted_f1
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    count = 0
    ys: List[int] = []
    logits_all: List[np.ndarray] = []
    fb_total = 0
    total = ds.n if max_samples is None else min(ds.n, int(max_samples))
    for_ser = (series_ids[:total] if series_ids is not None else None)
    try:
        for s in range(0, total, bs):
            b, y = ds.batch(range(s, min(s + bs, total)))
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out = model(b)
            if getattr(cfg, "mask_fusion_fix", False):
                fb_total += int(out.get("n_mask_fallback", 0))
            if cfg.task == "cls":
                l = F.cross_entropy(out["deployed"], y, weight=class_weight,
                                    label_smoothing=cfg.label_smoothing)
            else:
                l = F.mse_loss(out["deployed"].squeeze(-1), y)
            nb = int(y.numel())
            loss_sum += float(l) * nb
            count += nb
            ys.extend(y.cpu().tolist())
            logits_all.append(out["deployed"].detach().float().cpu().numpy())
    finally:
        if was_training:
            model.train()
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 0))
    if cfg.task == "cls":
        m = classification_report(
            logits, ys, cfg.num_classes,
            class_names_for(cfg, getattr(ds, "path", None)) or M3ED_CLASS_NAMES)
        if for_ser is not None:
            preds = np.argmax(logits, axis=1)
            per = per_group_weighted_f1(ys, preds, for_ser, cfg.num_classes)
            m["series_wf1_mean"] = float(np.mean(list(per.values()))) if per else 0.0
            m["series_wf1_min"] = float(np.min(list(per.values()))) if per else 0.0
            m["n_series_valid"] = len(per)
    else:
        m = regression_report(np.asarray(ys), logits.reshape(-1))
    if getattr(cfg, "mask_fusion_fix", False):
        m["n_mask_fallback"] = fb_total
    m["loss"] = loss_sum / max(count, 1)
    return m, logits


def cls_improved(m: Dict[str, float], best: Optional[Dict[str, float]],
                 md: float) -> bool:
    if best is None:
        return True
    k = (m["weighted_f1"], m["macro_f1"], m["accuracy"])
    p = (best["weighted_f1"], best["macro_f1"], best["accuracy"])
    if k[0] > p[0] + md:
        return True
    if abs(k[0] - p[0]) > md:
        return False
    if k[1] > p[1] + md:
        return True
    if abs(k[1] - p[1]) > md:
        return False
    return k[2] > p[2] + md


def reg_improved(m: Dict[str, float], best: Optional[Dict[str, float]],
                 md: float) -> bool:
    if best is None:
        return True
    if m["mae"] < best["mae"] - md:
        return True
    if m["mae"] > best["mae"] + md:
        return False
    if m["rmse"] < best["rmse"] - md:
        return True
    if m["rmse"] > best["rmse"] + md:
        return False
    return m["pearson"] > best["pearson"] + md


def series_improved(m: Dict[str, float], best: Optional[Dict[str, float]],
                    md: float) -> bool:
    """W6 selection: per-series WF1 mean -> overall WF1 -> Macro-F1."""
    if best is None:
        return True
    k = (m.get("series_wf1_mean", 0.0), m["weighted_f1"], m["macro_f1"])
    p = (best.get("series_wf1_mean", 0.0), best["weighted_f1"],
         best["macro_f1"])
    if k[0] > p[0] + md:
        return True
    if abs(k[0] - p[0]) > md:
        return False
    if k[1] > p[1] + md:
        return True
    if abs(k[1] - p[1]) > md:
        return False
    return k[2] > p[2] + md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dialogue-manifest", type=Path,
                    help="fixed train-only fit/inner_dev manifest")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--utility", default="shapley",
                    choices=["uniform", "gate", "shapley"])
    ap.add_argument("--deploy", default="solo_weighted",
                    choices=["solo_weighted", "joint_uniform", "joint_softgate", "closed_loop"],
                    help="deployment path: weighted solo-sum (default, keeps "
                         "round-1 checkpoints loadable), uniform joint head, "
                         "utility-soft-gated joint head, or the integrated "
                         "closed-loop joint path")
    # Innovation-2 revised router.  The evidence path is trained on a frozen
    # uniform base by train_gate.py; legacy remains the compatibility default.
    ap.add_argument("--gate-architecture", default="legacy",
                    choices=["legacy", "evidence", "constant"])
    ap.add_argument("--gate-detach-inputs", action="store_true")
    ap.add_argument("--detach-utility-path", action="store_true")
    ap.add_argument("--bounded-w", action="store_true",
                    help="use centered bounded contribution weights")
    ap.add_argument("--bounded-lambda", type=float, default=0.3)
    ap.add_argument("--contrib-correct", action="store_true",
                    help="legacy Shapley residual ablation")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--abstain", action="store_true")
    ap.add_argument("--abstain-lambda", type=float, default=0.1)
    ap.add_argument("--history-gate-mode", default="none",
                    choices=["none", "legacy_null", "sentence"])
    ap.add_argument("--history-gate-execution", default="hard",
                    choices=["soft", "hard"])
    ap.add_argument("--history-gate-target", default="label_match",
                    choices=["label_match", "utility"])
    ap.add_argument("--history-gate-features", default="base",
                    choices=["base", "semantic"])
    ap.add_argument("--history-gate-lambda", type=float, default=0.3)
    ap.add_argument("--history-gate-threshold", type=float, default=0.5)
    ap.add_argument("--history-gate-hidden", type=int, default=32)
    ap.add_argument("--history-gate-warmup-epochs", type=int, default=3)
    ap.add_argument("--history-gate-soft-epochs", type=int, default=2)
    ap.add_argument("--history-gate-temperature", type=float, default=1.0)
    ap.add_argument("--history-abstain-variant", default="none",
                    choices=["none", "legacy", "utility_softmax", "utility_semantic", "utility_sparsemax"],
                    help="core empty-candidate competition variant")
    ap.add_argument("--history-null-beta", type=float, default=1.0)
    ap.add_argument("--history-utility-lambda", type=float, default=0.3)
    ap.add_argument("--history-utility-hidden", type=int, default=32)
    ap.add_argument("--history-utility-temperature", type=float, default=1.0)
    ap.add_argument("--history-utility-cache", default="")
    # ---- iteration-3 switchable modules (all default OFF) ------------------
    ap.add_argument("--m1", action="store_true",
                    help="counterfactual-view training of the joint head")
    ap.add_argument("--m2", action="store_true",
                    help="regret gating: w = softmax((mu - kappa*sigma)/tau)")
    ap.add_argument("--m3", action="store_true",
                    help="utility-consistent manifold mixup")
    ap.add_argument("--lambda-j", type=float, default=0.5)
    ap.add_argument("--lambda-v", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="M3 Beta(alpha, alpha) mix coefficient")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="M2 risk-aversion coefficient")
    # ---- W6 gap-fix switches (default OFF) ---------------------------------
    ap.add_argument("--grl-lambda", type=float, default=0.0,
                    help="series-adversarial head strength (0 disables)")
    ap.add_argument("--selection-series", action="store_true",
                    help="select on per-series WF1 mean (valid only)")
    # ---- W9 StreamFusion switches (default OFF) ----------------------------
    ap.add_argument("--vpath", action="store_true",
                    help="frame-level video path (needs pack key Vf)")
    ap.add_argument("--mpath", action="store_true",
                    help="SAM2-style memory attention over history")
    ap.add_argument("--ugate", action="store_true",
                    help="micro-gates on the V/M corrections")
    ap.add_argument("--c1-alpha", type=float, default=0.0,
                    help="prior-calibrated evidence blend weight (cls)")
    ap.add_argument("--no-mask-fusion-fix", dest="mask_fusion_fix",
                    action="store_false", default=True,
                    help="disable mask-aware output fusion (default ON for "
                         "new runs; existing checkpoints keep their saved "
                         "behavior since their cfg lacks the field)")
    ap.add_argument("--residual", default="none",
                    choices=["none", "concat", "pair"],
                    help="2026-09-24 performance round-1: additive residual "
                         "on deployed logits (plan 04 S3.1)")
    ap.add_argument("--resid-rank", type=int, default=48)
    ap.add_argument("--resid-hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--cosine-epochs", type=int, default=10)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--swa-window", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-u", type=float, default=0.3)
    ap.add_argument("--lambda-c", type=float, default=0.05)
    ap.add_argument("--eps-floor", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--loss-tol", type=float, default=0.05)
    ap.add_argument("--loss-patience", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dump-valid-logits", action="store_true")
    args = ap.parse_args(argv)

    if args.abstain:
        if args.history_gate_mode not in ("none", "legacy_null"):
            raise SystemExit("--abstain conflicts with --history-gate-mode %s"
                             % args.history_gate_mode)
        args.history_gate_mode = "legacy_null"
    if args.no_history and args.history_gate_mode != "none":
        raise SystemExit("--no-history cannot be combined with a history gate")
    if args.task != "cls" and args.history_gate_mode == "sentence":
        raise SystemExit("sentence history gate v1 supports classification only")
    if args.m1 and args.deploy in ("joint_softgate", "closed_loop"):
        raise SystemExit(
            "--m1 is incompatible with weighted joint deployment; "
            "disable --m1 or use --deploy joint_uniform/solo_weighted")
    if args.history_gate_mode == "sentence":
        incompatible = []
        if args.utility != "uniform":
            incompatible.append("utility != uniform")
        if args.deploy != "solo_weighted":
            incompatible.append("deploy != solo_weighted")
        for flag, enabled in (("m1", args.m1), ("m2", args.m2), ("m3", args.m3),
                              ("vpath", args.vpath), ("mpath", args.mpath),
                              ("ugate", args.ugate), ("grl", args.grl_lambda > 0),
                              ("c1", args.c1_alpha > 0), ("residual", args.residual != "none")):
            if enabled:
                incompatible.append(flag)
        if incompatible:
            raise SystemExit("sentence history gate v1 requires the frozen uniform baseline; "
                             "unsupported switches: %s" % ", ".join(incompatible))
    if args.swa_window < 0:
        raise SystemExit("--swa-window must be >= 0")
    if (args.out / "FINAL_RESULT.json").is_file():
        raise SystemExit("refusing to overwrite the completed run %s"
                         % (args.out / "FINAL_RESULT.json"))

    seed_all(args.seed)
    device = torch.device(args.device)

    train = open_split(args.data, "train.pt", task=args.task)
    manifest_hash = None
    selection_split = "valid"
    if args.dialogue_manifest is not None:
        train, valid, manifest_hash = load_dialogue_manifest(args.dialogue_manifest, train)
        selection_split = "inner_dev"
    else:
        valid = open_split(args.data, "valid.pt", task=args.task)

    utility_cache = load_history_utility_cache(args.history_utility_cache)
    if utility_cache and args.history_abstain_variant not in (
            "utility_softmax", "utility_semantic", "utility_sparsemax"):
        raise SystemExit("history utility cache requires a utility empty-candidate variant")

    # A01/P0-02: all task geometry comes from the DatasetSpec resolved from
    # the pack path (never hardcoded).  num_classes, class names and the
    # dataset id are written into the cfg saved in best.pt.
    spec = spec_for_pack(args.data)
    if spec.task != args.task:
        raise SystemExit(
            "pack %s is %s but --task=%s" % (args.data, spec.task, args.task))

    # ---- W6: series ids (field 1 of <A/B>_<series>_<ep>_<utt>) -------------
    series_vocab = None
    if args.grl_lambda > 0.0:
        series_vocab = sorted({series_of(i) for i in train.raw["ids"]})
    valid_series = None
    if args.selection_series:
        valid_series = [series_of(i) for i in valid.raw["ids"]]

    lambda_u_eff = args.lambda_u if args.utility == "shapley" else 0.0
    class_prior = None
    if args.c1_alpha > 0.0 and args.task == "cls":
        y_tr = train.raw["label"].long()
        # Framework 10: out-of-range labels fail loudly, never clamp.
        if (y_tr < 0).any() or (y_tr >= spec.num_classes).any():
            raise SystemExit(
                "train labels out of range for %d-class spec"
                % spec.num_classes)
        counts = torch.bincount(y_tr, minlength=spec.num_classes).float()
        class_prior = [float(x) for x in (counts / counts.sum())]
    cfg = M6Config(
        text_dim=train.dims["T"], audio_dim=train.dims["A"],
        video_dim=train.dims["V"],
        task=args.task, num_classes=spec.num_classes,
        dataset_id=spec.dataset_id,
        class_names=(list(spec.class_names) if spec.class_names else None),
        d_model=args.d_model, dropout=args.dropout,
        use_history=(not args.no_history) and train.k > 0,
        use_abstain=bool(args.history_gate_mode == "legacy_null" or
                          args.history_abstain_variant in ("legacy", "utility_softmax", "utility_semantic", "utility_sparsemax")),
        history_k=train.k,
        history_gate_mode=args.history_gate_mode,
        history_gate_execution=args.history_gate_execution,
        history_gate_target=args.history_gate_target,
        history_gate_features=args.history_gate_features,
        history_gate_lambda=float(args.history_gate_lambda),
        history_gate_threshold=float(args.history_gate_threshold),
        history_gate_hidden=int(args.history_gate_hidden),
        history_gate_warmup_epochs=int(args.history_gate_warmup_epochs),
        history_gate_soft_epochs=int(args.history_gate_soft_epochs),
        history_gate_temperature=float(args.history_gate_temperature),
        history_abstain_variant=args.history_abstain_variant,
        history_null_beta=float(args.history_null_beta),
        history_utility_lambda=float(args.history_utility_lambda),
        history_utility_hidden=int(args.history_utility_hidden),
        history_utility_temperature=float(args.history_utility_temperature),
        history_utility_cache=str(args.history_utility_cache or ""),
        use_bounded_w=bool(args.bounded_w),
        bounded_lambda=float(args.bounded_lambda),
        contrib_correct=bool(args.contrib_correct),
        detach_utility_path=bool(args.detach_utility_path),
        gate_architecture=args.gate_architecture,
        gate_detach_inputs=bool(args.gate_detach_inputs),
        history_feature_dim=(4 if args.gate_architecture == "evidence" else 0),
        utility=args.utility, deploy=args.deploy,
        utility_head_version=2,
        lambda_u=lambda_u_eff, lambda_c=args.lambda_c,
        eps_floor=args.eps_floor, tau=args.tau,
        label_smoothing=0.05, class_weight_cap=2.0,
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        use_m1=bool(args.m1), use_m2=bool(args.m2), use_m3=bool(args.m3),
        lambda_j=args.lambda_j, lambda_v=args.lambda_v,
        mix_alpha=args.alpha, kappa=args.kappa,
        grl_lambda=float(args.grl_lambda),
        selection_series=bool(args.selection_series),
        n_series=(len(series_vocab) if series_vocab else 0),
        use_vpath=bool(args.vpath), use_mpath=bool(args.mpath),
        use_ugate=bool(args.ugate), c1_alpha=float(args.c1_alpha),
        class_prior=class_prior,
        bugfix_version=2,
        # mask-aware output fusion ON for new runs (framework candidate 2);
        # existing checkpoints keep their saved behavior (cfg default False)
        mask_fusion_fix=bool(args.mask_fusion_fix),
        residual=args.residual, resid_rank=args.resid_rank,
        resid_hidden=args.resid_hidden,
    )
    cfg.validate()
    model = UGFModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    warmup_epochs = max(1, args.warmup_epochs) if args.warmup_epochs > 0 else 0
    scheduler = None
    if args.cosine_epochs > 0:
        schedulers = []
        milestones = []
        if warmup_epochs > 0:
            def lr_lambda(epoch: int) -> float:
                return float(epoch + 1) / float(warmup_epochs)
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda))
            milestones.append(warmup_epochs)
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.cosine_epochs, eta_min=args.min_lr))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=schedulers, milestones=milestones)
    elif warmup_epochs > 0:
        def lr_lambda_flat(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_flat)

    class_weight = None
    if args.task == "cls":
        class_weight = compute_class_weights(
            train.raw["label"], cfg.num_classes,
            cap=cfg.class_weight_cap).to(device)

    dataset_name = args.data.resolve().parent.name
    selection_rule = (
        "%s per-series WF1 mean -> %s Weighted-F1 -> %s Macro-F1" %
        (selection_split, selection_split, selection_split)
        if (args.task == "cls" and args.selection_series)
        else "%s Weighted-F1 -> %s Macro-F1 -> %s Accuracy" %
        (selection_split, selection_split, selection_split)
        if args.task == "cls"
        else "%s MAE -> %s RMSE -> %s Pearson (lower MAE better)" %
        (selection_split, selection_split, selection_split))
    selection_metric = ("series_wf1" if (args.task == "cls" and args.selection_series)
                        else ("wf1" if args.task == "cls" else "mae"))
    meta = {
        "dataset": dataset_name,
        "task": args.task,
        "tag": args.tag,
        "seed": args.seed,
        "device": str(device),
        "train_samples": train.n,
        "valid_samples": valid.n,
        "selection_split": selection_split,
        "dialogue_manifest": (str(args.dialogue_manifest.resolve())
                              if args.dialogue_manifest is not None else None),
        "dialogue_manifest_sha256": manifest_hash,
        "dims": {m: int(train.dims[m]) for m in ("T", "A", "V")},
        "train_sha256": split_sha256(train),
        "valid_sha256": split_sha256(valid),
        "test_read": False,
        "split_guard": {
            "allowed_basenames": list(TRAINABLE_SPLITS),
            "enforced_by": "n6.data.open_split",
            "sealed_test_opened": False,
        },
        "history_contract": (
            "packed slots newest-first; canonical oldest-to-newest, "
            "right-aligned K=%d" % train.k),
        "history_gate": {
            "implementation_version": "core_empty_candidate_v2",
            "mode": cfg.history_gate_mode,
            "execution": cfg.history_gate_execution,
            "target": cfg.history_gate_target,
            "features": cfg.history_gate_features,
            "lambda": cfg.history_gate_lambda,
            "threshold": cfg.history_gate_threshold,
            "hidden": cfg.history_gate_hidden,
            "temperature": cfg.history_gate_temperature,
            "warmup_epochs": cfg.history_gate_warmup_epochs,
            "soft_epochs": cfg.history_gate_soft_epochs,
            "label_scope": (
                "training-only sentence label-match proxy" if cfg.history_gate_mode == "sentence"
                else "none"),
            "test_read": False,
        },
        "history_abstain": {
            "variant": cfg.history_abstain_variant,
            "null_beta": cfg.history_null_beta,
            "utility_lambda": cfg.history_utility_lambda,
            "utility_cache": cfg.history_utility_cache,
            "utility_cache_rows": len(utility_cache),
            "candidate_contract": "real history slots + one empty candidate before normalisation; hard empty removes history token path",
            "test_read": False,
        },
        "innovation2_router": {
            "architecture": cfg.gate_architecture,
            "bounded_weights": bool(cfg.use_bounded_w),
            "bounded_lambda": float(cfg.bounded_lambda),
            "gate_detach_inputs": bool(cfg.gate_detach_inputs),
            "detach_utility_path": bool(cfg.detach_utility_path),
            "contribution_residual": bool(cfg.contrib_correct),
        },
        "innovation1_closed_loop": {
            "active": cfg.deploy == "closed_loop",
            "final_predictor": "joint_head" if cfg.deploy == "closed_loop" else cfg.deploy,
            "history_feedback": "admitted history modifies current embeddings",
            "raw_history_tokens_in_closed_loop": False,
        },
        "selection_rule": selection_rule,
        "selection_metric": selection_metric,
        "selection_key": (["series_wf1_mean", "weighted_f1", "macro_f1"]
                          if (args.task == "cls" and args.selection_series)
                          else ["weighted_f1", "macro_f1", "accuracy"]
                          if args.task == "cls"
                          else ["mae", "-rmse", "pearson"]),
        "selection_unit": ("swa_window_average(%d)" % args.swa_window
                           if args.swa_window > 0 else "single_epoch"),
        "training_protocol": {
            "lr": args.lr, "weight_decay": args.weight_decay,
            "d_model": args.d_model, "dropout": args.dropout,
            "batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs,
            "cosine_epochs": args.cosine_epochs, "min_lr": args.min_lr,
            "max_epochs": args.epochs, "patience": args.patience,
            "min_delta": args.min_delta,
            "swa_window": args.swa_window, "grad_clip": args.grad_clip,
            "loss_tol": args.loss_tol, "loss_patience": args.loss_patience,
            "optimizer": "AdamW", "scheduler": "warmup->cosine (per epoch)",
        },
        "utility": {
            "mode": args.utility,
            "deploy": cfg.deploy,
            "lambda_u": lambda_u_eff,
            "lambda_c": args.lambda_c,
            "tau": args.tau,
            "eps_floor": args.eps_floor,
            "shapley_target": (
                "exact 3-player on measured subset utilities; per-batch "
                "standardised (mean 0, std 1 across the 3 modalities)"),
            "measurement": (
                "per training step on the current model state (no_grad), "
                "JOINT head on current tokens only"),
        },
        "modules": {
            "m1_counterfactual_view": {
                "on": bool(cfg.use_m1),
                "lambda_j": args.lambda_j, "lambda_v": args.lambda_v,
                "view": "per-sample random non-full subset, grouped "
                        "joint_on_subset forwards",
            },
            "m2_regret_gating": {
                "on": bool(cfg.use_m2),
                "kappa": args.kappa,
                "weights": "softmax((mu - kappa*sigma)/tau), floor+renorm",
                "utility_mse_supervises": "mu only",
            },
            "m3_utility_mixup": {
                "on": bool(cfg.use_m3),
                "alpha": args.alpha,
                "level": "cur_embs (post-encode), permutation pairing",
                "consistency_weight": 0.1,
                "target": "detached lam*g_phi(i) + (1-lam)*g_phi(j) on "
                          "pre-mix embeddings",
            },
            "utility_head_version": int(cfg.utility_head_version),
        },
        "series": {
            "grl_lambda": float(args.grl_lambda),
            "n_series": int(cfg.n_series),
            "selection_series": bool(args.selection_series),
            "selection_metric": selection_metric,
            "vocab": series_vocab,
            "id_parse": "field 1 of <A/B>_<series>_<ep>_<utt>",
        },
        "streamfusion": {
            "vpath": bool(cfg.use_vpath),
            "mpath": bool(cfg.use_mpath),
            "ugate": bool(cfg.use_ugate),
            "c1_alpha": float(cfg.c1_alpha),
            "class_prior": class_prior,
            "vpath_note": "replaces pooled-V embedding when pack has Vf; "
                          "no-op otherwise",
            "mpath_note": "memory = history slots of all modalities, masked "
                          "by validity x modality x speaker_same; LayerScale "
                          "zero-init read-out added to all current embs",
            "ugate_note": "242-param MLP on 12 solo-confidence stats; "
                          "end-to-end via task loss only",
            "c1_note": "cls + solo_weighted only; pi from train class counts",
        },
        "flags": {
            "no_history": bool(args.no_history),
            "dump_valid_logits": bool(args.dump_valid_logits),
            "smoke": int(args.smoke),
        },
        "early_stopping": {
            "patience": args.patience, "min_delta": args.min_delta,
            "selection": selection_rule,
            "valid_loss_tol": args.loss_tol,
            "valid_loss_patience": args.loss_patience,
        },
        "bugfix_version": 2,
        "bugfix_notes": (
            "2026-09-21 audit fixes: (1) V-path correction applied exactly "
            "once (pooled + g*(frame-pooled); previously double-applied so "
            "gate=0 kept the frame branch); (2) speaker_same reordered with "
            "the history-slot permutation; (3) KL consistency term uses "
            "batchmean (was sum, scaling with batch size).  "
            "bugfix_version=1 marks pre-fix runs."),
        "mask_fusion_fix": bool(cfg.mask_fusion_fix),
        "mask_fusion_fix_note": (
            "mask-aware output fusion (framework candidate 2): absent "
            "modalities (modality_mask=0) contribute nothing to the deployed "
            "sum; weights renormalize over present modalities; all-missing "
            "rows use a uniform-posterior fallback (zero logits / 0.0) and "
            "are counted in metrics.n_mask_fallback.  ON for new runs; "
            "existing checkpoints keep cfg default False = saved behavior."),
        "class_weights_applied": (
            None if class_weight is None
            else [float(x) for x in class_weight.cpu()]),
        "resolved_config": cfg.to_dict(),
        "code": code_manifest(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RUN_METADATA.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    best: Optional[Dict[str, float]] = None
    best_kind: Optional[str] = None
    best_epoch_index: Optional[int] = None
    bad_epochs = 0
    bad_loss = 0
    stopped_early = False
    stop_reason: Optional[str] = None
    loss_min: Optional[float] = None
    history: List[Dict[str, object]] = []
    dumped_logits: List[np.ndarray] = []
    dumped_single: List[np.ndarray] = []
    dumped_epochs: List[int] = []
    dumped_kinds: List[str] = []
    dumped_eligible: List[bool] = []

    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    # A partial SWA window is never eligible for selection; in smoke the run
    # is capped at 2 epochs, so cap the window too, otherwise no candidate
    # would ever become the champion and no best.pt would be written.  Real
    # runs (30 epochs, window 3) are unaffected.
    swa_window = (min(int(args.swa_window), epochs)
                  if args.smoke else int(args.swa_window))
    snapshots: deque = deque(maxlen=max(swa_window, 1))
    swa_model = copy.deepcopy(model).eval() if swa_window > 0 else None

    improved = (series_improved if args.selection_series
                else cls_improved if args.task == "cls" else reg_improved)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        model.set_history_gate_epoch(ep)
        perm = torch.randperm(train.n)
        running = 0.0
        steps = 0
        gate_loss_sum = 0.0
        gate_loss_count = 0
        gate_prob_sum = 0.0
        gate_hard_sum = 0.0
        gate_history_count = 0
        for st in range(0, train.n, args.batch_size):
            if args.smoke and steps >= args.smoke:
                break
            ix = perm[st:min(st + args.batch_size, train.n)].tolist()
            b, y = train.batch(ix)
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out_clean = model(b)
            out = out_clean
            mix = None
            if args.m3:
                bsz = int(y.shape[0])
                mix_perm = torch.randperm(bsz, device=device)
                beta = torch.distributions.Beta(
                    torch.tensor(args.alpha, device=device),
                    torch.tensor(args.alpha, device=device))
                lam = beta.sample((bsz,))                         # [B] on device
                mix = (mix_perm, lam)
                out = model(b, mix=mix)
            phi = None
            if args.utility == "shapley":
                phi = model.measure_shapley(
                    b, y,
                    token_weight=(out_clean["weights"].detach()
                                  if args.deploy in ("joint_softgate", "closed_loop")
                                  else None))
            aux = None
            if args.m1:
                tokens_c, mask_c = model.build_tokens(
                    out_clean["cur_embs"], out_clean["hist_embs"], b)
                m1_full = joint_supervision(
                    model.joint(tokens_c, mask_c), y, cfg, class_weight)
                m1_view = m1_view_loss(
                    model, out_clean["cur_embs"], b, y, cfg, class_weight)
                aux = {"m1_full": m1_full, "m1_view": m1_view}
            if args.m3 and out.get("g_mix") is not None:
                cons = F.mse_loss(out["g_mix"], out["g_mix_target"])
                aux = dict(aux or {}, m3_consistency=cons)
            if model.series_head is not None:
                slabels = torch.tensor(
                    [series_vocab.index(series_of(train.raw["ids"][i]))
                     for i in ix],
                    dtype=torch.long, device=device)
                aux = dict(aux or {}, series=F.cross_entropy(
                    out["series_logits"], slabels))
            loss_d = total_loss(out, y, cfg, class_weight=class_weight,
                                phi=phi, mu=out_clean["utility_mu"],
                                mix=mix, aux=aux)
            if (utility_cache and cfg.history_abstain_variant in
                    ("utility_softmax", "utility_semantic", "utility_sparsemax")
                    and out_clean.get("history_gate_logit") is not None):
                global_ix = (train.indices[torch.as_tensor(ix, dtype=torch.long)]
                             .tolist() if hasattr(train, "indices") else ix)
                targets = torch.tensor(
                    [utility_cache.get(int(i), (0.5, 0.0))[0] for i in global_ix],
                    dtype=out_clean["history_gate_logit"].dtype, device=device)
                weights = torch.tensor(
                    [utility_cache.get(int(i), (0.5, 0.0))[1] for i in global_ix],
                    dtype=out_clean["history_gate_logit"].dtype, device=device)
                has_history = out_clean.get("has_history")
                if has_history is None:
                    has_history = torch.ones_like(weights)
                weights = weights * has_history.to(weights.dtype)
                raw = F.binary_cross_entropy_with_logits(
                    out_clean["history_gate_logit"], targets, reduction="none")
                denom = weights.sum().clamp_min(1.0)
                utility_gate_loss = (raw * weights).sum() / denom
                loss_d["loss"] = loss_d["loss"] + cfg.history_utility_lambda * utility_gate_loss
                n_hist = int((weights > 0).sum().item())
                gate_loss_sum += float(utility_gate_loss.detach()) * n_hist
                gate_loss_count += n_hist
                gate_prob_sum += float((out_clean["history_gate_prob"].detach() * weights).sum())
                gate_hard_sum += float((out_clean["history_gate_hard"].detach() * weights).sum())
                gate_history_count += n_hist
            elif (cfg.history_gate_mode == "legacy_null" and out_clean.get("abstain") is not None
                    and "history_label" in b):
                hl = b["history_label"]
                valid_h = b["history_mask"] > 0
                match = ((hl == y.unsqueeze(1)) & valid_h).any(dim=1)
                target_null = (~match).float()
                abst_loss = F.binary_cross_entropy(
                    out_clean["abstain"].float().clamp(1e-6, 1.0 - 1e-6),
                    target_null.unsqueeze(1).expand(-1, 3))
                loss_d["loss"] = loss_d["loss"] + args.abstain_lambda * abst_loss
            elif (cfg.history_gate_mode == "sentence"
                  and out_clean.get("history_gate_logit") is not None):
                # Training-only label-match proxy.  It is deliberately a
                # sentence-level target and excludes rows without readable
                # history; a future frozen utility cache replaces this block.
                eligible_slot = (b["history_mask"] > 0) & (
                    b["history_modality_mask"].sum(dim=-1) > 0)
                has_history = eligible_slot.any(dim=1)
                match = ((b["history_label"] == y.unsqueeze(1))
                         & eligible_slot).any(dim=1).float()
                raw = F.binary_cross_entropy_with_logits(
                    out_clean["history_gate_logit"], match, reduction="none")
                denom = has_history.float().sum().clamp_min(1.0)
                gate_loss = (raw * has_history.float()).sum() / denom
                loss_d["loss"] = loss_d["loss"] + cfg.history_gate_lambda * gate_loss
                n_hist = int(has_history.sum().item())
                gate_loss_sum += float(gate_loss.detach()) * n_hist
                gate_loss_count += n_hist
                gate_prob_sum += float((out_clean["history_gate_prob"].detach()
                                        * has_history.float()).sum())
                gate_hard_sum += float((out_clean["history_gate_hard"].detach()
                                        * has_history.float()).sum())
                gate_history_count += n_hist
            opt.zero_grad(set_to_none=True)
            loss_d["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += float(loss_d["loss"].detach())
            steps += 1
        if scheduler is not None:
            scheduler.step()

        candidate = model
        candidate_kind = "single_epoch"
        eligible = True
        if swa_window > 0:
            snapshots.append({k: v.detach().clone()
                              for k, v in model.state_dict().items()})
            if len(snapshots) < swa_window:
                # A partial window is logged but never allowed to become the
                # champion: the candidate family must stay consistent.
                eligible = False
                candidate_kind = "swa_filling"
            else:
                assert swa_model is not None
                swa_model.load_state_dict(average_state_dicts(snapshots),
                                          strict=True)
                candidate = swa_model
                candidate_kind = "swa_window"

        candidate.set_history_gate_epoch(ep)

        max_samples = args.batch_size * 2 if args.smoke else None
        m, epoch_logits = evaluate(candidate, valid, device, args.batch_size,
                                   cfg, class_weight=class_weight,
                                   max_samples=max_samples,
                                   series_ids=valid_series)
        single_logits = epoch_logits
        if args.dump_valid_logits and candidate is not model:
            single_logits = evaluate(
                model, valid, device, args.batch_size, cfg,
                class_weight=class_weight,
                max_samples=max_samples,
                series_ids=valid_series)[1]
        if args.dump_valid_logits:
            dumped_logits.append(epoch_logits.astype(np.float32))
            dumped_single.append(single_logits.astype(np.float32))
            dumped_epochs.append(int(ep))
            dumped_kinds.append(candidate_kind)
            dumped_eligible.append(bool(eligible))

        row = {"epoch": ep, "train_loss": running / max(steps, 1),
               "elapsed_sec": round(time.time() - t0, 1),
               "candidate": candidate_kind, "eligible": eligible, **m}
        if cfg.history_gate_mode == "sentence":
            row.update({
                "history_gate_stage": (
                    "warmup" if ep < cfg.history_gate_warmup_epochs else
                    "soft" if ep < cfg.history_gate_warmup_epochs + cfg.history_gate_soft_epochs
                    else cfg.history_gate_execution),
                "history_gate_loss": gate_loss_sum / max(gate_loss_count, 1),
                "history_gate_supervised_samples": gate_loss_count,
                "history_gate_mean_keep_probability": gate_prob_sum / max(gate_history_count, 1),
                "history_gate_hard_keep_rate": gate_hard_sum / max(gate_history_count, 1),
            })
        history.append(row)
        print(json.dumps(row), flush=True)

        if eligible and improved(m, best, args.min_delta):
            best = dict(m)
            best_kind = candidate_kind
            best_epoch_index = int(ep)
            torch.save({"model": candidate.state_dict(), "cfg": cfg.to_dict(),
                        "seed": args.seed, "epoch": ep, "valid": m,
                        "candidate_kind": candidate_kind,
                        "swa_window": swa_window},
                       args.out / "best.pt")
            (args.out / "best_valid_metrics.json").write_text(
                json.dumps({"epoch": ep, "candidate": candidate_kind, **m},
                           indent=2),
                encoding="utf-8")
            bad_epochs = 0
        else:
            if eligible:
                bad_epochs += 1
        loss_min = m["loss"] if loss_min is None else min(loss_min, m["loss"])
        bad_loss = (bad_loss + 1
                    if m["loss"] > loss_min * (1.0 + args.loss_tol) else 0)
        if (not args.smoke and args.loss_patience > 0
                and bad_loss >= args.loss_patience):
            stopped_early = True
            stop_reason = "valid_loss_degraded(%d epochs > %.0f%% above min)" % (
                bad_loss, 100.0 * args.loss_tol)
            break
        if not args.smoke and bad_epochs >= max(1, args.patience):
            stopped_early = True
            stop_reason = ("valid_wf1_plateau(%d epochs)" % bad_epochs
                           if args.task == "cls"
                           else "valid_mae_plateau(%d epochs)" % bad_epochs)
            break

    if args.dump_valid_logits and dumped_logits:
        np.savez_compressed(
            args.out / "valid_logits.npz",
            logits=np.stack(dumped_logits),
            logits_single_epoch=np.stack(dumped_single),
            epochs=np.asarray(dumped_epochs, dtype=np.int64),
            candidate_kind=np.asarray(dumped_kinds, dtype=object),
            eligible=np.asarray(dumped_eligible, dtype=bool),
             ids=np.asarray(valid.raw.get("ids", []), dtype=str),
            y_true=(valid.raw["label"].numpy().astype(np.int64)
                    if args.task == "cls"
                    else valid.raw["label"].numpy().astype(np.float64)),
        )
    (args.out / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    if not (args.out / "best.pt").is_file():
        raise RuntimeError("missing best.pt - run is not a valid result")
    final = {
        "dataset": dataset_name,
        "tag": args.tag,
        "seed": args.seed,
        "task": args.task,
        "utility": args.utility,
        "deploy": cfg.deploy,
        "modules": {"m1": bool(cfg.use_m1), "m2": bool(cfg.use_m2),
                    "m3": bool(cfg.use_m3), "lambda_j": args.lambda_j,
                    "lambda_v": args.lambda_v, "alpha": args.alpha,
                    "kappa": args.kappa,
                    "vpath": bool(cfg.use_vpath), "mpath": bool(cfg.use_mpath),
                    "ugate": bool(cfg.use_ugate), "c1_alpha": float(cfg.c1_alpha)},
        "selection_metric": selection_metric,
        "grl_lambda": float(args.grl_lambda),
        "n_series": int(cfg.n_series),
        "selection_series": bool(args.selection_series),
        "bugfix_version": 2,
        "mask_fusion_fix": bool(cfg.mask_fusion_fix),
        "lambda_u": lambda_u_eff,
        "lambda_c": args.lambda_c,
        "best_valid": best,
        "best_candidate_kind": best_kind,
        "best_epoch_index": best_epoch_index,
        "selection_protocol": (
            "%s; selection unit = %s; valid only, test never read" % (
                selection_rule,
                ("uniform average of the last %d epochs" % swa_window)
                if swa_window > 0 else "single epoch")),
        "status": "SMOKE_PASS" if args.smoke else "TRAIN_COMPLETE",
        "test_read": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "valid_loss_min": loss_min,
        "swa_window": swa_window,
        "no_history": bool(args.no_history),
        "trainable_parameters": model.count_trainable_parameters(),
    }
    (args.out / "FINAL_RESULT.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8")
    print("FINAL_RESULT", json.dumps(final), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
