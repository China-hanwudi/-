from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.distributional_utility_repair import (  # noqa: E402
    run_private_distributional_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen open-role sign-by-severity distributional utility repair and "
            "write aggregate model-selection JSON only."
        )
    )
    parser.add_argument("--private-cache", type=Path, required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--repair-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_private_distributional_repair(
        args.private_cache,
        args.lineage_report,
        args.repair_config,
        args.output,
        overwrite=args.overwrite,
    )
    primary = next(
        row
        for row in report["ranking"]
        if row["name"] == "distributional_true_bidirectional"
    )
    print(
        "selected="
        f"{report['selected_open_role_model']} "
        f"true_bidirectional_rank={primary['rank']} "
        f"status={report['status']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
