"""Build a deterministic, dialogue-disjoint fit / inner-dev manifest.

The script accepts only a packed ``train.pt``.  It never opens valid.pt or
test.pt.  MELD IDs are expected to begin with ``dia<integer>_utt<integer>``;
unknown ID formats fail instead of being treated as independent samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch


ID_RE = re.compile(r"^(dia[^_]+)_utt\d+$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--dev-fraction", type=float, default=0.10)
    args = ap.parse_args()
    if args.train.name != "train.pt":
        raise SystemExit("manifest builder accepts train.pt only")
    if not 0.01 <= args.dev_fraction < 0.5:
        raise SystemExit("--dev-fraction must be in [0.01, 0.5)")
    raw = torch.load(args.train, map_location="cpu", weights_only=True)
    if "ids" not in raw or "label" not in raw:
        raise SystemExit("train pack needs ids and label")
    ids = [str(x) for x in raw["ids"]]
    labels = raw["label"].long().tolist()
    if len(ids) != len(labels):
        raise SystemExit("ids/label length mismatch")
    dialogue_rows = defaultdict(list)
    for i, uid in enumerate(ids):
        m = ID_RE.match(uid)
        if m is None:
            raise SystemExit("unsupported ID at index %d: %r" % (i, uid))
        dialogue_rows[m.group(1)].append(i)
    dialogue_ids = sorted(dialogue_rows)
    rng = random.Random(args.seed)
    rng.shuffle(dialogue_ids)
    target = round(len(ids) * args.dev_fraction)
    dev_dialogues = set()
    count = 0
    for did in dialogue_ids:
        if count >= target:
            break
        dev_dialogues.add(did)
        count += len(dialogue_rows[did])
    membership = ["inner_dev" if ID_RE.match(uid).group(1) in dev_dialogues else "fit"
                  for uid in ids]
    fit = [i for i, x in enumerate(membership) if x == "fit"]
    dev = [i for i, x in enumerate(membership) if x == "inner_dev"]
    if not fit or not dev:
        raise SystemExit("empty fit or inner_dev split")
    payload = {
        "version": "dialogue_fit_inner_dev_v1",
        "train_path": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "seed": args.seed,
        "requested_dev_fraction": args.dev_fraction,
        "actual_dev_fraction": len(dev) / len(ids),
        "n": len(ids),
        "fit_indices": fit,
        "inner_dev_indices": dev,
        "dialogue_counts": {"fit": len(dialogue_ids) - len(dev_dialogues),
                            "inner_dev": len(dev_dialogues)},
        "class_counts": {"fit": dict(Counter(labels[i] for i in fit)),
                         "inner_dev": dict(Counter(labels[i] for i in dev))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit("refusing to overwrite manifest %s" % args.out)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("n", "actual_dev_fraction", "dialogue_counts", "class_counts")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
