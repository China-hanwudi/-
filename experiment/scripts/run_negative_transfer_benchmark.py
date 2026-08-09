from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.negative_transfer_benchmark import BenchmarkColumns, evaluate_benchmark  # noqa: E402


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()
    cli.add_argument("--input", type=Path, required=True, help="Private per-query CSV or CSV.GZ")
    cli.add_argument("--adapter", required=True, help="Adapter key in the frozen benchmark config")
    cli.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "negative_transfer_benchmark_v1.json",
    )
    cli.add_argument("--output", type=Path, required=True, help="Aggregate-only JSON output")
    return cli


def main() -> None:
    args = parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.adapter not in config["adapters"]:
        raise SystemExit(f"unknown adapter: {args.adapter}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    adapter = config["adapters"][args.adapter]
    frame = pd.read_csv(args.input)
    result = evaluate_benchmark(
        frame,
        BenchmarkColumns(
            history_count=adapter["history_count"],
            excess_loss=adapter["excess_loss"],
            cluster=(tuple(adapter["cluster"]) if isinstance(adapter["cluster"], list) else adapter["cluster"]),
        ),
        adapter["selectors"],
        config["coverage_targets"],
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    payload = {
        "protocol": config["protocol"],
        "adapter": args.adapter,
        "status": config["status"],
        "claim_boundary": config["reporting_contract"]["claim_boundary"],
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "eligible": result["data"]["eligible_history_rows"]}))


if __name__ == "__main__":
    main()
