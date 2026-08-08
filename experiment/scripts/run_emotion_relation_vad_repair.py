from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotion_relation_vad_repair import (  # noqa: E402
    REGISTERED_OUTPUT_PATH,
    run_emotion_relation_vad_repair,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CARMA-Affect Repair 3/3 from strict physical fit/model-selection "
            "sidecars. The 299-D primary is fixed before results; a fit-only OOF "
            "gate must pass before any model-selection prediction or label scoring."
        )
    )
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=ROOT / "configs" / "carma_causal_backbone_v1.json",
    )
    parser.add_argument(
        "--repair-config",
        type=Path,
        default=ROOT / "configs" / "emotion_relation_vad_repair_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REGISTERED_OUTPUT_PATH,
        help=(
            "Write-once registered output path. The resolved repository path is "
            "frozen; alternate directories are rejected."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_emotion_relation_vad_repair(
        args.sidecar_dir,
        args.manifest,
        args.model_config,
        args.repair_config,
        args.output,
    )
    print(
        f"status={report['status']} fit_gate_passed="
        f"{report['fit_only_open_gate']['passed']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
