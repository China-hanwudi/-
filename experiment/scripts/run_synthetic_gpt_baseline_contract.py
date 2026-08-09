"""Exercise the offline synthetic GPT baseline contract with mock responses only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from hva_affect.synthetic_gpt_baseline_contract import (  # noqa: E402
    SyntheticGPTContractError,
    run_synthetic_fixture_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an explicitly synthetic text fixture and cache its caller-supplied "
            "mock responses. This command has no API or production mode."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "synthetic_gpt_text_baseline_v1.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "carma_split_manifest_v1.json",
    )
    parser.add_argument(
        "--synthetic-fixture",
        type=Path,
        required=True,
        help="Explicit fixture whose filename ends in .synthetic.json.",
    )
    parser.add_argument("--hmac-key-file", type=Path, required=True)
    parser.add_argument(
        "--private-cache",
        type=Path,
        required=True,
        help="Write-once cache path outside the repository.",
    )
    parser.add_argument("--public-receipt", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-synthetic-only",
        action="store_true",
        help=(
            "Required attestation that the fixture is synthetic and contains no real, "
            "restricted, personal, or dataset-derived content."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_synthetic_fixture_contract(
            config_path=args.config,
            split_manifest_path=args.split_manifest,
            synthetic_fixture_path=args.synthetic_fixture,
            hmac_key_path=args.hmac_key_file,
            private_cache_path=args.private_cache,
            public_receipt_path=args.public_receipt,
            explicit_synthetic_acknowledgement=args.acknowledge_synthetic_only,
        )
    except SyntheticGPTContractError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
