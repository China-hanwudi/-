from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_causal_backbone_loader import run_meld_causal_backbone  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the MELD causal backbone from the manifest-verified train fit and "
            "model-selection feature+label sidecars only."
        )
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        required=True,
        help="Private directory containing the role-separated MELD train sidecars.",
    )
    parser.add_argument(
        "--sidecar-manifest",
        type=Path,
        required=True,
        help="Aggregate-only manifest that hashes the sidecars and frozen source contract.",
    )
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=ROOT / "configs" / "carma_causal_backbone_meld_v1.json",
    )
    parser.add_argument(
        "--utility-config",
        type=Path,
        default=ROOT / "configs" / "bidirectional_emotion_utility_v1.json",
    )
    parser.add_argument(
        "--confirmatory-config",
        type=Path,
        default=ROOT / "configs" / "carma_confirmatory_analysis_v1.json",
    )
    parser.add_argument(
        "--private-output-dir",
        type=Path,
        required=True,
        help="Required private directory outside the public repository.",
    )
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_meld_causal_backbone(
        sidecar_dir=args.sidecar_dir,
        sidecar_manifest_path=args.sidecar_manifest,
        backbone_config_path=args.backbone_config,
        utility_config_path=args.utility_config,
        confirmatory_config_path=args.confirmatory_config,
        private_output_dir=args.private_output_dir,
        public_output_path=args.public_output,
        repository_root=REPOSITORY_ROOT,
        device_name=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
