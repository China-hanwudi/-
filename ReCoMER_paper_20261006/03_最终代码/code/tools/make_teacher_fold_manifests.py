"""Create deterministic dialogue-disjoint 3-fold teacher manifests.

Only ``train.pt`` and the already frozen student manifest are read.  The
student inner-dev dialogue set is excluded from every teacher fold; each
output manifest uses two folds as fit and the remaining fold as inner-dev.
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
    ap.add_argument("--student-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    if args.train.name != "train.pt":
        raise SystemExit("teacher manifest builder accepts train.pt only")
    student = json.loads(args.student_manifest.read_text(encoding="utf-8"))
    raw = torch.load(args.train, map_location="cpu", weights_only=True)
    ids = [str(x) for x in raw["ids"]]
    labels = raw["label"].long().tolist()
    student_fit = set(int(i) for i in student["fit_indices"])
    student_dev = set(int(i) for i in student["inner_dev_indices"])
    if student_fit & student_dev or student_fit | student_dev != set(range(len(ids))):
        raise SystemExit("student manifest does not partition train indices")
    rows = defaultdict(list)
    for i in sorted(student_fit):
        m = ID_RE.match(ids[i])
        if m is None:
            raise SystemExit("unsupported ID at index %d: %r" % (i, ids[i]))
        rows[m.group(1)].append(i)
    dialogue_ids = sorted(rows)
    if len(dialogue_ids) < 3:
        raise SystemExit("need at least three fit dialogues")
    rng = random.Random(args.seed)
    rng.shuffle(dialogue_ids)
    folds = [set() for _ in range(3)]
    for n, did in enumerate(dialogue_ids):
        folds[n % 3].add(did)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_hash = sha256(args.train)
    parent_hash = sha256(args.student_manifest)
    for fold, heldout_dialogues in enumerate(folds):
        out = args.out_dir / ("teacher_fold%d_manifest.json" % fold)
        if out.exists():
            raise SystemExit("refusing to overwrite %s" % out)
        dev = [i for did in heldout_dialogues for i in rows[did]]
        fit = [i for did in dialogue_ids if did not in heldout_dialogues
               for i in rows[did]]
        payload = {
            "version": "dialogue_fit_inner_dev_v1",
            "protocol": "innovation3_teacher_preflight_v1",
            "fold": fold,
            "seed": args.seed,
            "train_path": str(args.train.resolve()),
            "train_sha256": train_hash,
            "student_manifest_sha256": parent_hash,
            "fit_indices": sorted(fit),
            "inner_dev_indices": sorted(dev),
            "excluded_indices": sorted(student_dev),
            "dialogue_counts": {"fit": len(dialogue_ids) - len(heldout_dialogues),
                                "inner_dev": len(heldout_dialogues)},
            "class_counts": {"fit": dict(Counter(labels[i] for i in fit)),
                             "inner_dev": dict(Counter(labels[i] for i in dev))},
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"fold": fold, "fit": len(fit), "inner_dev": len(dev),
                          "dialogues": payload["dialogue_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
