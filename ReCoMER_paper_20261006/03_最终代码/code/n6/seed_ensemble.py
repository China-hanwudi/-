"""Valid-set seed ensembling (W6 ``ens``): average the per-sample deployed
logits of every seed of a config and recompute the valid metrics.

Integrity gate (red-line-7 style): each seed's reloaded standalone valid
metric must reproduce its run's ``FINAL_RESULT`` record before its logits
enter the average; any mismatch fails loudly.  Valid split only -- the
sealed test split is never read anywhere in this tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .data import open_split
from .evaluate import load_checkpoint
from .metrics import classification_report, regression_report
from .train import evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True,
                    help="run dir containing seed*/best.pt")
    ap.add_argument("--seeds", default=None,
                    help="optional comma list of seed numbers to include")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to also write the JSON report")
    args = ap.parse_args(argv)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device))
    ckpts = sorted(Path(args.runs).glob("seed*/best.pt"),
                   key=lambda p: p.parent.name)
    if args.seeds:
        want = set(int(s) for s in args.seeds.split(","))
        ckpts = [c for c in ckpts if int(c.parent.name.replace("seed", "")) in want]
    if not ckpts:
        raise SystemExit("no seed*/best.pt found under %s" % args.runs)

    rows: List[dict] = []
    logits_sum: Optional[np.ndarray] = None
    cfg_ref = None
    for ck in ckpts:
        model, cfg, ckmeta = load_checkpoint(str(ck), device)
        cfg_ref = cfg
        valid = open_split(args.data, "valid.pt", task=cfg.task)
        m, logits = evaluate(model, valid, device, args.batch_size, cfg)
        seed = ckmeta.get("seed")
        fr_path = ck.parent / "FINAL_RESULT.json"
        integrity = {"checked": False, "match": None, "record": None,
                     "recomputed": None}
        if fr_path.is_file():
            fr = json.loads(fr_path.read_text(encoding="utf-8"))
            if cfg.task == "cls":
                rec = float(fr["best_valid"]["weighted_f1"])
                got = float(m["weighted_f1"])
            else:
                rec = float(fr["best_valid"]["mae"])
                got = float(m["mae"])
            integrity = {"checked": True,
                         "record": rec, "recomputed": got,
                         "match": abs(rec - got) < 1e-6}
            if not integrity["match"]:
                raise SystemExit(
                    "integrity failure %s: record=%.10f recomputed=%.10f"
                    % (ck, rec, got))
        rows.append({
            "seed": seed, "metrics": m, "integrity": integrity,
            "candidate_kind": ckmeta.get("candidate_kind"),
            "epoch": ckmeta.get("epoch"),
        })
        logits_sum = logits if logits_sum is None else logits_sum + logits
        print("seed %s wf1=%.6f integrity=%s"
              % (seed, m["weighted_f1"], integrity["match"]), flush=True)

    ens_logits = logits_sum / float(len(ckpts))
    valid = open_split(args.data, "valid.pt", task=cfg_ref.task)
    y = valid.raw["label"].numpy()
    if cfg_ref.task == "cls":
        from .datasetspec import class_names_for
        ens = classification_report(
            ens_logits, y, cfg_ref.num_classes,
            class_names_for(cfg_ref, getattr(valid, "path", None)))
        per_seed = [r["metrics"]["weighted_f1"] for r in rows]
    else:
        ens = regression_report(np.asarray(y), ens_logits.reshape(-1))
        per_seed = [r["metrics"]["mae"] for r in rows]
    report = {
        "runs": str(args.runs),
        "n_seeds": len(ckpts),
        "seeds": [r["seed"] for r in rows],
        "task": cfg_ref.task,
        "split": "valid",
        "test_read": False,
        "per_seed": rows,
        "per_seed_metric": per_seed,
        "per_seed_mean": float(np.mean(per_seed)),
        "per_seed_std": float(np.std(per_seed)),
        "ensemble": ens,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
