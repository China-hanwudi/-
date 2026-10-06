"""Frozen Innovation-2 validation benchmark; NEVER opens a test split.

Re-evaluates all ten previously selected checkpoints without fitting or
choosing seeds. Metrics follow MMSA's regression conventions. Label-informed
bounds are diagnostics only and MUST NOT be called deployable performance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from n6.data import open_split, split_sha256
from n6.evaluate import load_checkpoint
from n6.metrics import regression_report

SEEDS = (7, 13, 17, 23, 29, 37, 43, 53, 71, 101)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def benchmark_metrics(y, p):
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    out = {k: regression_report(y, p)[k] for k in ("mae", "rmse", "pearson")}
    # MMSA rounds labels and predictions after clipping for Acc-7/Acc-5.
    for levels in (5, 7):
        limit = (levels - 1) // 2
        out[f"acc{levels}_pct"] = float(100 * accuracy_score(
            np.round(np.clip(y, -limit, limit)),
            np.round(np.clip(p, -limit, limit))))
    for mode in ("nonnegative", "nonzero"):
        keep = np.ones(len(y), dtype=bool) if mode == "nonnegative" else y != 0
        t = y[keep] >= 0 if mode == "nonnegative" else y[keep] > 0
        q = p[keep] >= 0 if mode == "nonnegative" else p[keep] > 0
        out[f"acc2_{mode}_pct"] = float(100 * accuracy_score(t, q))
        out[f"f1_weighted_{mode}_pct"] = float(100 * f1_score(
            t, q, average="weighted", zero_division=0))
    return out


def bounded_prediction_interval(solo, mask, lam):
    """Exact extrema under the ORIGINAL centered three-way bounded formula.

    Centered normalized scores occupy the boundary of a hexagon (and the
    epsilon-scaled interior). Its vertices are permutations of (-1,0,1).
    Presence renormalization is linear-fractional with a positive denominator,
    so extrema are achieved at vertices. The attainable scalar image is an
    interval. A target-aware projection onto that interval is a nondeployable
    lower bound on MAE for ANY gate with this frozen solo output stack.
    """
    scores = torch.tensor(list(itertools.permutations((-1., 0., 1.))),
                          device=solo.device, dtype=solo.dtype)
    weights = (1 + lam * scores)[None] / 3
    weights = weights * mask[:, None]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
    vertices = (weights * solo[:, None]).sum(-1)
    return vertices.min(-1).values, vertices.max(-1).values


@torch.no_grad()
def collect(model, ds, device):
    predictions, solos, masks = [], [], []
    for start in range(0, ds.n, 256):
        batch, _ = ds.batch(range(start, min(start + 256, ds.n)))
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        predictions.append(out["deployed"][:, 0].cpu())
        solos.append(out["solo_stack"][:, :, 0].cpu())
        masks.append(batch["modality_mask"].cpu())
    return torch.cat(predictions), torch.cat(solos), torch.cat(masks)


def self_test():
    from n6.config import M6Config
    from n6.model import UGFModel
    cfg = M6Config(task="reg", num_classes=1, utility="shapley",
                   use_bounded_w=True, bounded_lambda=.3)
    model = UGFModel(cfg)
    gen = torch.Generator().manual_seed(123)
    solo = torch.randn(8, 3, generator=gen)
    mask = torch.tensor(list(itertools.product((0., 1.), repeat=3)))
    lo, hi = bounded_prediction_interval(solo, mask, .3)
    for _ in range(100):
        mu = torch.randn(8, 3, generator=gen)
        w = model.deploy_weights(mu) * mask
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        p = (solo * w).sum(-1)
        assert (p >= lo - 1e-6).all() and (p <= hi + 1e-6).all()
    assert lo[0] == hi[0] == 0
    metric = benchmark_metrics(np.array([-1., 0., 1.]), np.array([-1., -1., 1.]))
    assert metric["acc2_nonzero_pct"] == 100
    assert abs(metric["acc2_nonnegative_pct"] - 200/3) < 1e-8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite benchmark result")
    torch.set_num_threads(2)
    self_test()
    ds = open_split(args.data, "valid.pt", "reg")
    device = torch.device("cuda")
    paths = [(seed, variant,
              args.base / f"seed{seed}" / "best.pt" if variant == "uniform" else
              args.gates / f"seed{seed}" / variant / "best.pt")
             for seed in SEEDS for variant in
             ("uniform", "constant_task", "mlp_task", "evidence_task")]
    # Lock the entire cohort before evaluating. Never pick best test/valid seed.
    cohort = [{"seed": s, "variant": v, "path": str(p), "sha256": sha(p)}
              for s, v, p in paths]
    protocol = dict(split="valid", test_read=False, seeds=list(SEEDS),
                    checkpoints=cohort, n=ds.n, split_sha256=split_sha256(ds),
                    metric_source="https://github.com/thuiar/MMSA/blob/master/src/MMSA/utils/metricsTop.py",
                    acc7_rule="round(clip(pred,-3,3)) == round(clip(label,-3,3))",
                    f1_rule="support-weighted binary F1; report both zero conventions",
                    primary="mean single-seed MAE; no best-seed or ensemble selection",
                    oracle_scope="VALID ONLY; uses labels; nondeployable capability diagnostic",
                    literature_comparison="NOT apples-to-apples: published test vs current valid; pooled/history vs sequence protocols")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out.with_suffix(".protocol.json")
    if protocol_path.exists():
        raise SystemExit("refusing to overwrite benchmark protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    y = ds.raw["label"].numpy().reshape(-1)
    for seed, variant, path in paths:
        model, cfg, _ = load_checkpoint(path, device)
        p, solo, mask = collect(model, ds, device)
        metrics = benchmark_metrics(y, p.numpy())
        reference = json.loads(path.with_name("FINAL_RESULT.json").read_text())
        expected = (reference["best_valid"] if variant != "uniform" else
                    reference.get("valid", reference.get("best_valid")))
        if expected is None:
            raise ValueError("unrecognized baseline result schema")
        assert abs(metrics["mae"] - expected["mae"]) < 1e-6
        row = dict(seed=seed, variant=variant, metrics=metrics)
        if variant == "evidence_task":
            assert cfg.use_bounded_w and cfg.bounded_lambda == .3
            assert cfg.residual == "none" and not cfg.contrib_correct
            assert cfg.deploy == "solo_weighted" and cfg.mask_fusion_fix
            lo, hi = bounded_prediction_interval(solo, mask, cfg.bounded_lambda)
            y_tensor = torch.from_numpy(y).to(lo)
            oracle = y_tensor.maximum(lo).minimum(hi)
            row["label_oracle_bounded_mae_lower_bound_NOT_DEPLOYABLE"] = float(
                (oracle - y_tensor).abs().mean())
            row["fraction_labels_outside_bounded_solo_interval"] = float(
                ((y_tensor < lo) | (y_tensor > hi)).float().mean())
            row["mean_prediction_interval_width"] = float((hi-lo).mean())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    means = {variant: {key: float(np.mean([r["metrics"][key] for r in rows
                                         if r["variant"] == variant]))
                       for key in rows[0]["metrics"]}
             for variant in ("uniform", "constant_task", "mlp_task", "evidence_task")}
    oracle_mean = float(np.mean([r["label_oracle_bounded_mae_lower_bound_NOT_DEPLOYABLE"]
                                for r in rows if r["variant"] == "evidence_task"]))
    result = dict(protocol=protocol, rows=rows, means=means,
                  label_oracle_mean_mae_lower_bound_NOT_DEPLOYABLE=oracle_mean,
                  status="VALIDATION_BENCHMARK_COMPLETE", test_read=False)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(means=means, oracle_lower_bound=oracle_mean,
                          test_read=False, status=result["status"])), flush=True)


if __name__ == "__main__":
    main()
