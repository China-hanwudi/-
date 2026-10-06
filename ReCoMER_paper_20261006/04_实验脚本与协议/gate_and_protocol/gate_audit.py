"""Valid-only gate interventions on a frozen checkpoint, never a deployable oracle."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "代码/创新点2_门控对照/code"))
from n6.data import open_split
from n6.evaluate import load_checkpoint
from n6.metrics import classification_report
from n6.losses import total_loss


def metric(logits, y, task):
    if task == "cls":
        r = classification_report(logits.numpy(), y.numpy(), logits.shape[-1])
        return {k: r[k] for k in ("weighted_f1", "macro_f1", "accuracy", "nll")}
    return {"mae": float((logits[:, 0] - y).abs().mean())}


def summarize(ckpt, data, device):
    model, cfg, _ = load_checkpoint(ckpt, device)
    ds = open_split(data, "valid.pt", task=cfg.task)
    parts = {k: [] for k in ("y", "mask", "mu", "w", "phi", "solo", "raw", "deploy", "joint")}
    gradient_norms = None
    for st in range(0, ds.n, 64):
        b, y = ds.batch(range(st, min(st + 64, ds.n)))
        b = {k: v.to(device) for k, v in b.items()}
        y = y.to(device)
        with torch.no_grad():
            out = model(b)
            raw = torch.stack([model.solo_heads[m](out["cur_embs"][m]) for m in ("T", "A", "V")], 1)
            phi = model.measure_shapley(b, y, standardize=True)
        values = dict(y=y, mask=b["modality_mask"], mu=out["utility_mu"],
                      w=out["weights"], phi=phi, solo=out["solo_stack"], raw=raw,
                      deploy=out["deployed"], joint=out["joint"])
        for k, v in values.items():
            parts[k].append(v.detach().cpu())
        if gradient_norms is None:
            # Gradient routing on one fixed valid batch is a code-path audit,
            # not training; no optimizer steps are taken.
            g = model(b)
            ls = total_loss(g, y, cfg, phi=phi)
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            params = [p for p in model.parameters() if p.requires_grad]
            gradient_norms = {}
            for kind in ("task", "utility", "consistency"):
                grads = torch.autograd.grad(ls[kind], params, allow_unused=True, retain_graph=True)
                gradient_norms[kind] = {}
                for prefix in ("encoders.", "utility_head.", "joint.", "contrib_corrector."):
                    sq = sum(float(v.detach().square().sum()) for n, v in zip(names, grads)
                             if n.startswith(prefix) and v is not None)
                    gradient_norms[kind][prefix] = sq ** 0.5
    a = {k: torch.cat(v) for k, v in parts.items()}
    y, mm, solo, raw = a["y"], a["mask"], a["solo"], a["raw"]
    full = mm.sum(1) == 3
    n = len(y)
    w = a["w"] * mm
    w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
    uw = mm / mm.sum(1, keepdim=True).clamp_min(1)
    uniform = (uw[:, :, None] * solo).sum(1)
    phi = a["phi"].masked_fill(~mm.bool(), -1e9)
    ps = phi.argmax(1)
    ws = w.argmax(1)
    if cfg.task == "cls":
        all_p = F.softmax(solo, -1)
        su = all_p.gather(2, y[:, None, None].expand(-1, 3, 1)).squeeze(-1).clamp_min(1e-9).log()
        correct = solo.argmax(-1) == y[:, None]
        dep_correct = a["deploy"].argmax(-1) == y
        uni_correct = uniform.argmax(-1) == y
    else:
        su = -(solo[:, :, 0] - y[:, None]).abs()
    su = su.masked_fill(~mm.bool(), -1e9)
    ss = su.argmax(1)
    sorted_su = su.sort(1, descending=True).values
    strong = full & ((sorted_su[:, 0] - sorted_su[:, 1]) > (0.5 if cfg.task == "cls" else 0.2))
    oracle_w = model.deploy_weights(a["phi"].to(device)).cpu() * mm
    oracle_w /= oracle_w.sum(1, keepdim=True).clamp_min(1e-9)
    indices = torch.arange(n)
    results = {"ckpt": str(ckpt), "task": cfg.task, "n": n,
               "mask_full_fraction": float(full.float().mean()),
               "metrics": {"deployed": metric(a["deploy"], y, cfg.task),
                           "uniform_same_checkpoint": metric(uniform, y, cfg.task),
                           "joint_full": metric(a["joint"], y, cfg.task),
                           "no_corrector_same_weights": metric((w[:, :, None] * raw).sum(1), y, cfg.task),
                           "oracle_same_bounded_formula_LABELS_USED": metric((oracle_w[:, :, None] * solo).sum(1), y, cfg.task)},
               "mean_weight": w.mean(0).tolist(),
               "pred_top_fraction": torch.bincount(ws, minlength=3).float().div(n).tolist(),
               "teacher_top_fraction": torch.bincount(ps, minlength=3).float().div(n).tolist(),
               "solo_best_fraction": torch.bincount(ss, minlength=3).float().div(n).tolist(),
               "pred_teacher_match": float((ws == ps).float().mean()),
               "pred_solo_match": float((ws == ss).float().mean()),
               "teacher_solo_match": float((ps == ss).float().mean()),
               "gradient_norms_eval_batch": gradient_norms,
               "strong_solo_margin_threshold": 0.5 if cfg.task == "cls" else 0.2,
               "strong_solo_n": int(strong.sum()),
               "strong_solo_best_mean_weight": float(w[indices, ss][strong].mean()) if strong.any() else None,
               "strong_solo_pred_match": float((ws[strong] == ss[strong]).float().mean()) if strong.any() else None,
               "teacher_gap": {}, "static_interventions": {}, "pred_top_interventions": {}}
    for alpha in (1/3, 0.4333333333, 0.6, 0.8, 1.0):
        pw = torch.full_like(w, (1-alpha)/2)
        pw[indices, ws] = alpha
        pw *= mm
        pw /= pw.sum(1, keepdim=True).clamp_min(1e-9)
        results["pred_top_interventions"][str(round(alpha, 4))] = metric((pw[:, :, None]*solo).sum(1), y, cfg.task)
        for m in range(3):
            sw = torch.full_like(w, (1-alpha)/2)
            sw[:, m] = alpha
            sw *= mm
            sw /= sw.sum(1, keepdim=True).clamp_min(1e-9)
            results["static_interventions"][str(m)+"_"+str(round(alpha,4))] = metric((sw[:, :, None]*solo).sum(1), y, cfg.task)
    for m in range(3):
        results["metrics"]["solo_"+str(m)] = metric(solo[:, m], y, cfg.task)
    if cfg.task == "cls":
        exactly_one = full & (correct.sum(1) == 1)
        harmed = exactly_one & ~dep_correct
        results["dilution"] = {
            "exactly_one_solo_correct_n": int(exactly_one.sum()),
            "fusion_wrong_on_exactly_one_correct_n": int(harmed.sum()),
            "any_solo_correct_fusion_wrong_fraction": float((correct.any(1) & ~dep_correct).float().mean()),
            "all_solo_wrong_fusion_correct_fraction": float((~correct.any(1) & dep_correct).float().mean()),
            "uniform_correct_deployed_wrong_n": int((uni_correct & ~dep_correct).sum()),
            "uniform_wrong_deployed_correct_n": int((~uni_correct & dep_correct).sum()),
            "sole_correct_modality_mean_weight": [float(w[exactly_one & correct[:,m], m].mean())
                                                 if (exactly_one & correct[:,m]).any() else None for m in range(3)],
            "sole_correct_n_by_modality": [int((exactly_one & correct[:,m]).sum()) for m in range(3)],
            "sole_correct_fusion_wrong_n_by_modality": [int((harmed & correct[:,m]).sum()) for m in range(3)],
        }
    else:
        best_error = -su.max(1).values
        dep_error = (a["deploy"][:,0] - y).abs()
        results["dilution"] = {"mean_best_solo_mae_LABELS_USED": float(best_error.mean()),
                               "fusion_worse_than_best_solo_fraction": float((dep_error > best_error).float().mean()),
                               "fusion_worse_than_best_solo_by_0.2_fraction": float((dep_error > best_error + 0.2).float().mean())}
    gap = phi.sort(1, descending=True).values[:,0] - phi.sort(1, descending=True).values[:,1]
    results["teacher_gap"] = {"below_0.2_fraction": float((gap < 0.2).float().mean())}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        r = summarize(job["ckpt"], job["data"], torch.device(args.device))
        (out / (job["name"]+".json")).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(job["name"], r["metrics"]["deployed"], flush=True)


if __name__ == "__main__":
    main()
