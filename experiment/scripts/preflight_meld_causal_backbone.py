from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_multimodal_backbone import CausalBackboneConfig  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    BackboneRunConfig,
    UtilitySamplingConfig,
    validate_open_role_backbone_payload,
)
from hva_affect.meld_causal_backbone_loader import (  # noqa: E402
    preflight_meld_causal_backbone_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only MELD structural/forward preflight. It hashes but never "
            "deserializes model-selection labels and computes no performance metric."
        )
    )
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--sidecar-manifest", type=Path, required=True)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backbone_payload = json.loads(args.backbone_config.read_text(encoding="utf-8"))
    utility_payload = json.loads(args.utility_config.read_text(encoding="utf-8"))
    validate_open_role_backbone_payload(backbone_payload)
    report = preflight_meld_causal_backbone_inputs(
        sidecar_dir=args.sidecar_dir,
        manifest_path=args.sidecar_manifest,
        model_config=CausalBackboneConfig.from_mapping(backbone_payload),
        run_config=BackboneRunConfig.from_mapping(backbone_payload),
        sampling_config=UtilitySamplingConfig.from_mapping(utility_payload),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
