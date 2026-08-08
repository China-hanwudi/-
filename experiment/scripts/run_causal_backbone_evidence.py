from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_backbone_evidence_runner import (  # noqa: E402
    run_fit_preflight,
)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def _mapping(values: list[tuple[str, Path]], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise SystemExit(f"duplicate {label} name: {name}")
        result[name] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Staged CARMA causal-evidence utilities. fit-preflight performs only "
            "hash/structural validation; it never trains or computes performance."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser(
        "fit-preflight",
        help="materialise fit sidecars, hash selection sidecars, and write a receipt",
    )
    fit.add_argument("--dataset", choices=("EmotionTalk", "MELD"), required=True)
    fit.add_argument("--sidecar-dir", type=Path, required=True)
    fit.add_argument("--sidecar-manifest", type=Path, required=True)
    fit.add_argument("--receipt", type=Path, required=True)
    fit.add_argument(
        "--config",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
        help="hash-bound configuration file; repeat for every frozen config",
    )
    fit.add_argument(
        "--code",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
        help="hash-bound source file; repeat for every required module",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "fit-preflight":
        raise SystemExit("unsupported command")
    result = run_fit_preflight(
        dataset=args.dataset,
        sidecar_dir=args.sidecar_dir,
        manifest_path=args.sidecar_manifest,
        receipt_path=args.receipt,
        config_paths=_mapping(args.config, "config"),
        code_paths=_mapping(args.code, "code"),
    )
    summary = {
        "schema_version": result.receipt["schema_version"],
        "status": result.receipt["status"],
        "dataset": result.receipt["dataset"],
        "fit_rows": result.fit.rows,
        "receipt_sha256": result.receipt_sha256,
        "training_run": False,
        "performance_metric_computed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
