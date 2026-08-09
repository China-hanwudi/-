from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_multimodal_sidecar import prepare_meld_role_sidecars  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify registered MELD train artifacts and create write-once, "
            "role-separated multimodal feature and label sidecars."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--train-pickle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare_meld_role_sidecars(
        train_csv_path=args.train_csv,
        train_pickle_path=args.train_pickle,
        config_path=args.config,
        private_output_dir=args.private_output_dir,
        public_manifest_path=args.public_manifest,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        "status="
        f"{report['status']} "
        f"roles={len(report['roles'])} "
        f"missing_features={report['source_contract']['missing_feature_rows']} "
        "embedded_label_audit=trusted_custodian_only_not_exposed",
        flush=True,
    )


if __name__ == "__main__":
    main()
