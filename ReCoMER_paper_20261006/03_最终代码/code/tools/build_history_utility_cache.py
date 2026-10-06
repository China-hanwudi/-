"""Build a train-only history keep-utility cache from frozen teacher folds.

For each held-out fit sample, compare the teacher loss with history disabled
and enabled.  The cache is later consumed only by student fit batches; the
student inner-dev dialogue set is excluded by the teacher manifests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from n6.data import open_split
from n6.evaluate import load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--teacher-manifest-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--loss-scale", type=float, default=0.2)
    args = ap.parse_args()
    if args.data.name != "packed":
        raise SystemExit("--data must be the packed directory")
    ds = open_split(args.data, "train.pt", task="cls")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    for fold in range(3):
        ckpt = args.teacher_root / ("fold%d" % fold) / "best.pt"
        manifest = args.teacher_manifest_dir / ("teacher_fold%d_manifest.json" % fold)
        if not ckpt.is_file() or not manifest.is_file():
            raise SystemExit("missing teacher fold %d artifacts" % fold)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        model, cfg, _ = load_checkpoint(str(ckpt), device)
        model.eval()
        heldout = [int(i) for i in payload["inner_dev_indices"]]
        for start in range(0, len(heldout), args.batch_size):
            indices = heldout[start:start + args.batch_size]
            b, y = ds.batch(indices)
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            with torch.no_grad():
                off = model(b, history_override=0)["deployed"]
                on = model(b, history_override=1)["deployed"]
                loss_off = F.cross_entropy(off, y, reduction="none")
                loss_on = F.cross_entropy(on, y, reduction="none")
                utility = loss_off - loss_on
                target = torch.sigmoid(utility / float(args.loss_scale))
            for j, sample_i in enumerate(indices):
                u = float(utility[j].cpu())
                rows[str(sample_i)] = {
                    "utility": u,
                    "target_keep": float(target[j].cpu()),
                    "weight": 1.0,
                    "teacher_fold": fold,
                }
    out = {
        "version": "history_utility_cache_v1",
        "task": "cls",
        "train_sha256": None,
        "loss_definition": "CE(history_off)-CE(history_on)",
        "target_definition": "sigmoid(utility/scale)",
        "loss_scale": float(args.loss_scale),
        "rows": rows,
        "test_read": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "out": str(args.out), "test_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
