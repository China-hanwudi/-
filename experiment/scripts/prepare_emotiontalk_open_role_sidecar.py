from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_role_sidecar import prepare_emotiontalk_role_sidecars  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-time trusted generation of four non-pickled EmotionTalk feature "
            "and label sidecars for frozen train buckets 0--79 only."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--transcription", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "emotiontalk_open_role_sidecar_v2.json",
    )
    parser.add_argument(
        "--private-output-dir",
        type=Path,
        required=True,
        help="Write-once private directory for four role-separated NPZ files.",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        required=True,
        help="Aggregate-only JSON; contains no labels or row identifiers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_emotiontalk_role_sidecars(
        label_archive_path=args.labels,
        feature_path=args.features,
        transcription_path=args.transcription,
        config_path=args.config,
        private_output_dir=args.private_output_dir,
        public_manifest_path=args.public_manifest,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
