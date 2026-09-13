"""EmotionTalk-safe adapter for the existing CMER/MELD training engine.

Only train and validation manifests are accepted.  In particular, this
adapter never discovers or opens an EmotionTalk test manifest.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

from n3_affect import train_meld as engine


def assert_emotiontalk_train_only(train_manifest: Path) -> dict:
    manifest = train_manifest.resolve()
    lowered_parts = {part.lower() for part in manifest.parts}
    if {"test", "sealed"} & lowered_parts or any(
        token in manifest.name.lower() for token in ("test", "sealed")
    ):
        raise SystemExit(f"EmotionTalk test/sealed manifest is forbidden: {manifest}")

    with manifest.open(encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty training manifest: {manifest}")

    split_values = {str(row.get("split", "")).strip().lower() for row in rows}
    if split_values != {"train"}:
        raise SystemExit(f"training manifest contains non-train split values: {sorted(split_values)}")
    invalid_ids = [str(row.get("sample_id", "")) for row in rows if not str(row.get("sample_id", "")).startswith("train_")]
    if invalid_ids:
        raise SystemExit(f"training manifest contains non-train sample IDs: {invalid_ids[:10]}")

    return {
        "dataset": "EmotionTalk",
        "train_n_ids": len({str(row["sample_id"]) for row in rows}),
        "val_loaded": False,
        "test_loaded": False,
        "val_used_for": "not_loaded_during_training",
        "test_used_for": "never_discovered_or_loaded",
        "checkpoint_rule": "weighted_f1_then_macro_f1_then_loss_on_validation_never_test",
    }


def main() -> int:
    engine.assert_train_only = assert_emotiontalk_train_only
    rc = engine.main()

    # Preserve the adapter itself alongside the engine's normal code snapshot.
    if "--out-dir" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1])
        code_dir = out_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
