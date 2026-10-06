"""Train only a gate on a frozen, independently trained uniform checkpoint.

Train labels measure exactly the existing Shapley target once on this fixed
teacher. Validation labels select SWA gate checkpoints only; validation
Shapley is not computed. No test file is opened. Every gate variant sees the
same base model, features, train targets, seed, schedule and selection rule.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import open_split, split_sha256
from .evaluate import load_checkpoint
from .metrics import classification_report, regression_report
from .losses import compute_class_weights
from .model import UGFModel, MODALITIES
from .train import average_state_dicts


def build_cache(base, ds, device, teacher):
    parts = {k: [] for k in ("emb", "solo", "mask", "y", "history_state")}
    if teacher:
        parts["phi"] = []
    with torch.no_grad():
        for st in range(0, ds.n, 128):
            b, y = ds.batch(range(st, min(st + 128, ds.n)))
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out = base(b)
            parts["emb"].append(torch.stack([out["cur_embs"][m] for m in MODALITIES], 1).cpu())
            parts["solo"].append(out["solo_stack"].cpu())
            parts["mask"].append(b["modality_mask"].cpu())
            parts["y"].append(y.cpu())
            prob = out.get("history_gate_prob")
            applied = out.get("history_gate_applied")
            has = out.get("has_history")
            abst = out.get("abstain")
            if prob is None:
                prob = torch.zeros(y.shape[0], device=device)
            if applied is None:
                applied = torch.zeros_like(prob)
            if has is None:
                has = torch.zeros_like(prob)
            abst_mean = (abst.mean(dim=1) if abst is not None
                         else torch.zeros_like(prob))
            parts["history_state"].append(torch.stack(
                [prob, applied, has.to(prob.dtype), abst_mean], dim=-1).cpu())
            if teacher:
                parts["phi"].append(base.measure_shapley(b, y, standardize=True).cpu())
    return {k: torch.cat(v).to(device) for k, v in parts.items()}


def gate_forward(model, cache, idx):
    emb = {m: cache["emb"][idx, i] for i, m in enumerate(MODALITIES)}
    solo = {m: cache["solo"][idx, i] for i, m in enumerate(MODALITIES)}
    mask = cache["mask"][idx]
    hs = cache["history_state"][idx]
    state = {"history_gate_prob": hs[:, 0],
             "history_gate_applied": hs[:, 1],
             "has_history": hs[:, 2],
             "abstain": hs[:, 3].unsqueeze(1)}
    mu, sigma = model.predict_utility(
        emb, solo, {"modality_mask": mask}, state)
    w = model.deploy_weights(mu, sigma) * mask
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    logits = (w.unsqueeze(-1) * cache["solo"][idx]).sum(1)
    return mu, w, logits


def rank_loss(mu, phi, mask):
    # The original Shapley values remain unchanged. Relative ordering is an
    # additional gate-only objective; near-tie pairs carry no rank gradient.
    acc = mu.new_zeros(())
    count = mu.new_zeros(())
    for i, j in ((0, 1), (0, 2), (1, 2)):
        gap = phi[:, i] - phi[:, j]
        valid = (gap.abs() >= 0.2) & (mask[:, i] > 0) & (mask[:, j] > 0)
        value = F.softplus(-gap.sign() * (mu[:, i] - mu[:, j]))
        acc = acc + (value * valid).sum()
        count = count + valid.sum()
    return acc / count.clamp_min(1)


def report(logits, y, cfg):
    if cfg.task == "cls":
        return classification_report(logits.cpu().numpy(), y.cpu().numpy(), cfg.num_classes)
    return regression_report(y.cpu().numpy(), logits[:, 0].cpu().numpy())


def selection_key(metrics, cfg):
    if cfg.task == "cls":
        return (metrics["weighted_f1"], metrics["macro_f1"], metrics["accuracy"])
    return (-metrics["mae"], -metrics["rmse"], metrics["pearson"])


def fit_variant(base, base_sd, train, valid, args, variant, provenance):
    cfg = copy.deepcopy(base.cfg)
    cfg.utility = "shapley"
    cfg.use_bounded_w = True
    cfg.bounded_lambda = 0.3
    cfg.contrib_correct = False
    cfg.detach_utility_path = True
    cfg.gate_detach_inputs = True
    cfg.gate_architecture = ("constant" if variant.startswith("constant") else
                             "legacy" if variant.startswith("mlp") else "evidence")
    cfg.history_feature_dim = 4 if cfg.gate_architecture == "evidence" else 0
    cfg.utility_head_version = 2 if variant.startswith("mlp") else 5
    cfg.validate()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = UGFModel(cfg).to(args.device)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    expected = {"utility_head."+n for n in model.utility_head.state_dict()}
    if set(missing) != expected or unexpected:
        raise ValueError("uniform base load mismatch: %r %r" % (missing, unexpected))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("utility_head."))
    model.eval()
    opt = torch.optim.AdamW(model.utility_head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    out = args.out / variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FINAL_RESULT.json").exists():
        raise ValueError("refusing to overwrite completed gate experiment: %s" % out)
    task_calibrate = variant.endswith("_task")
    rank_weight = 0.1 if variant == "evidence_rank" or task_calibrate else 0.0
    class_weight = (compute_class_weights(train["y"], cfg.num_classes)
                    if cfg.task == "cls" else None)
    meta = dict(provenance, seed=args.seed, variant=variant, frozen_base=True,
                rank_weight=rank_weight, rank_min_standardized_gap=0.2,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                swa_window=3, gate_input_detach=True, no_contribution_residual=True,
                resolved_config=cfg.to_dict(), selection="valid SWA(3); no valid Shapley",
                gate_objective=("task + 0.3*MSE + 0.1*rank" if task_calibrate
                                else "MSE + rank_weight*rank"),
                task_gradient_to_gate_only=task_calibrate,
                test_read=False)
    (out / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    snapshots = deque(maxlen=3)
    swa_gate = copy.deepcopy(model.utility_head).eval()
    rows, best_key, best = [], None, None
    start = time.time()
    for epoch in range(args.epochs):
        model.utility_head.train()
        order = torch.randperm(len(train["y"]), device=args.device)
        loss_sum, rank_sum, task_sum, steps = 0.0, 0.0, 0.0, 0
        for st in range(0, len(order), args.batch_size):
            idx = order[st:st+args.batch_size]
            mu, _, logits = gate_forward(model, train, idx)
            mse = F.mse_loss(mu, train["phi"][idx])
            ranking = rank_loss(mu, train["phi"][idx], train["mask"][idx])
            loss = mse + rank_weight * ranking
            task = mu.new_zeros(())
            if task_calibrate:
                # The base and the cached predictions are frozen. Task loss
                # calibrates only gate parameters; it cannot change the
                # encoder, solo heads, joint teacher, or the bounded formula.
                task = (F.cross_entropy(logits, train["y"][idx],
                                        weight=class_weight, label_smoothing=cfg.label_smoothing)
                        if cfg.task == "cls" else
                        F.mse_loss(logits[:, 0], train["y"][idx]))
                loss = task + 0.3 * mse + rank_weight * ranking
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.utility_head.parameters(), 1.0)
            opt.step()
            loss_sum += float(mse.detach())
            rank_sum += float(ranking.detach())
            task_sum += float(task.detach())
            steps += 1
        scheduler.step()
        snapshots.append({k: v.detach().clone() for k, v in model.utility_head.state_dict().items()})
        row = dict(epoch=epoch, train_mse=loss_sum/steps, train_rank=rank_sum/steps,
                   train_gate_task=task_sum/steps)
        if len(snapshots) == 3:
            swa_gate.load_state_dict(average_state_dicts(snapshots), strict=True)
            live_gate = model.utility_head
            model.utility_head = swa_gate
            with torch.no_grad():
                _, w, logits = gate_forward(model, valid, slice(None))
                metrics = report(logits, valid["y"], cfg)
            key = selection_key(metrics, cfg)
            row.update(valid=metrics, mean_weight=w.mean(0).cpu().tolist())
            if best_key is None or key > best_key:
                best_key = key
                best = dict(epoch=epoch, metrics=metrics, mean_weight=row["mean_weight"])
                torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(),
                            "gate_protocol": meta}, out / "best.pt")
            model.utility_head = live_gate
        rows.append(row)
    (out / "history.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    result = dict(seed=args.seed, variant=variant, best_valid=best["metrics"],
                  mean_weight=best["mean_weight"], best_epoch_index=best["epoch"],
                  base_valid=provenance["base_valid"], test_read=False,
                  status="TRAIN_COMPLETE", elapsed_sec=time.time()-start)
    (out / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    key = "mae" if cfg.task == "reg" else "weighted_f1"
    print(json.dumps(dict(seed=args.seed, variant=variant, value=result["best_valid"][key],
                          baseline=provenance["base_valid"][key])), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--variants", default="constant,mlp,evidence,evidence_rank")
    args = p.parse_args()
    torch.set_num_threads(2)
    args.device = torch.device(args.device)
    base, cfg, _ = load_checkpoint(args.base_ckpt, args.device)
    if (cfg.utility != "uniform" or cfg.residual != "none"
            or cfg.c1_alpha != 0 or cfg.deploy != "solo_weighted"):
        raise ValueError("expected a plain uniform solo_weighted checkpoint")
    variants = args.variants.split(",")
    if any(v not in ("constant", "mlp", "evidence", "evidence_rank",
                    "constant_task", "mlp_task", "evidence_task") for v in variants):
        raise ValueError("unknown variant")
    train_ds = open_split(args.data, "train.pt", cfg.task)
    valid_ds = open_split(args.data, "valid.pt", cfg.task)
    print("CACHE_START", args.base_ckpt, flush=True)
    train = build_cache(base, train_ds, args.device, teacher=True)
    valid = build_cache(base, valid_ds, args.device, teacher=False)
    mm = valid["mask"]
    weights = mm / mm.sum(1, keepdim=True).clamp_min(1)
    logits = (weights[:, :, None] * valid["solo"]).sum(1)
    base_valid = report(logits, valid["y"], cfg)
    source_result = args.base_ckpt.parent / "FINAL_RESULT.json"
    if source_result.exists():
        saved = json.loads(source_result.read_text(encoding="utf-8"))["best_valid"]
        metric_key = "mae" if cfg.task == "reg" else "weighted_f1"
        if abs(saved[metric_key] - base_valid[metric_key]) > 1e-6:
            raise ValueError("frozen base no longer reproduces its saved validation metric")
    base_sd = base.state_dict()
    provenance = dict(base_checkpoint=str(args.base_ckpt),
                      base_checkpoint_sha256=hashlib.sha256(args.base_ckpt.read_bytes()).hexdigest(),
                      train_sha256=split_sha256(train_ds),
                      valid_sha256=split_sha256(valid_ds), base_valid=base_valid,
                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")},
                      validation_teacher_used=False,
                      shapley_teacher="unchanged measure_shapley, standardized per sample",
                      deployment="unchanged UGFModel.deploy_weights, lambda=0.3")
    print("CACHE_DONE", flush=True)
    for variant in variants:
        fit_variant(base, base_sd, train, valid, args, variant, provenance)


if __name__ == "__main__":
    main()
