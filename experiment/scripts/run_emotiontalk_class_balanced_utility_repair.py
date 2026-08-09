from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.class_balanced_utility_repair import (  # noqa: E402
    run_open_role_class_balanced_query_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run open-role repair 2/3: fit-query class-balanced utility learning "
            "with one prediction per model-selection query. Restricted roles are "
            "not accepted as inputs, and the output is write-once aggregate JSON."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument(
        "--repair-config",
        type=Path,
        default=ROOT / "configs" / "emotiontalk_class_balanced_utility_repair_v1.json",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_open_role_class_balanced_query_policy(
        args.data_dir,
        args.feature,
        args.base_config,
        args.utility_config,
        args.repair_config,
        args.cache,
        args.checkpoint_dir,
        args.output,
    )
    print(
        f"status={report['status']} model_selection_queries="
        f"{report['experiment_counts']['model_selection_queries']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
