"""Freeze the input/source fingerprint for a completed model7 pilot run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--mode", choices=["face", "scene", "duplicate"],
                    required=True)
    args = ap.parse_args()
    run = Path(args.run)
    result = json.loads((run / "FINAL_RESULT.json").read_text())
    if result.get("status") != "TRAIN_COMPLETE" or not result.get(
            "reload_integrity"):
        raise RuntimeError("run missing training/reload evidence")
    output = run / "RUN_INPUTS.json"
    if output.exists():
        raise FileExistsError(output)
    info = {"mode": args.mode, "arm": result["arm"],
            "seed": result["seed"], "base_ckpt_sha256": sha(args.base),
            "feature_pack": {split: sha(Path(args.pack) / (split + ".npz"))
                             for split in ("train", "valid")},
            "code": {p.name: sha(p) for p in sorted(
                Path(args.code_root).glob("*.py"))},
            "test_read": False}
    output.write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
