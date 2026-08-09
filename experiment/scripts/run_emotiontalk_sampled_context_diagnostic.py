from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_sampled_context_runner import (  # noqa: E402
    run_open_role_sampled_context_diagnostic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit existing open-role bidirectional probability checkpoints, verify the "
            "private 59-D cache, and emit aggregate sampled-context classification diagnostics."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_open_role_sampled_context_diagnostic(
        args.data_dir,
        args.feature,
        args.base_config,
        args.utility_config,
        args.cache,
        args.checkpoint_dir,
        args.output,
    )
    print(
        f"status={report['status']} models={len(report['models'])} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
