from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_multimodal_external import (  # noqa: E402
    create_freeze_manifest,
    train_only,
    validate_once,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Frozen EmotionTalk multimodal external confirmation")
    result.add_argument("stage", choices=("train", "freeze", "validate"))
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--features", type=Path, default=ROOT / "artifacts" / "emotiontalk_media_features_v1.npz")
    result.add_argument("--config", type=Path, default=ROOT / "configs" / "emotiontalk_multimodal_external_v1.json")
    result.add_argument("--bundle", type=Path, default=ROOT / "artifacts" / "emotiontalk_multimodal_external_v1.joblib")
    result.add_argument("--train-summary", type=Path, default=ROOT / "artifacts" / "emotiontalk_multimodal_external_v1_train_only.json")
    result.add_argument("--freeze-manifest", type=Path, default=ROOT / "artifacts" / "emotiontalk_multimodal_external_v1_freeze.json")
    result.add_argument("--output", type=Path, default=ROOT / "artifacts" / "emotiontalk_multimodal_external_v1.json")
    result.add_argument("--per-query", type=Path, default=ROOT / "artifacts" / "emotiontalk_multimodal_external_v1_per_query.csv.gz")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.stage == "train":
        report = train_only(args.data_dir, args.features, args.config, args.bundle, args.train_summary)
    elif args.stage == "freeze":
        report = create_freeze_manifest(
            args.data_dir, args.features, args.config, args.bundle,
            [Path(__file__), ROOT / "src" / "hva_affect" / "emotiontalk_multimodal_external.py"],
            args.freeze_manifest,
        )
    else:
        report = validate_once(
            args.data_dir, args.features, args.config, args.bundle,
            args.freeze_manifest, args.output, args.per_query,
        )
    print(json.dumps(report.get("gates", report.get("hashes", report)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
