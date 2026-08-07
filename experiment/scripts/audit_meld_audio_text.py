from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_audio_contract import (  # noqa: E402
    extract_and_audit_split,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and featurize real MELD WAV Parquet")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "meld_audio_text_feasibility_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "meld_audio_text_preflight_v1.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["dataset_source"]
    reports = {}
    for split, parquet_key, csv_name, csv_hash in (
        ("train", "train_parquet", "train_sent_emo.csv", source["train_labels_sha256"]),
        ("dev", "dev_parquet", "dev_sent_emo.csv", source["dev_labels_sha256"]),
    ):
        frozen = source[parquet_key]
        reports[split] = extract_and_audit_split(
            [args.parquet_dir / item["name"] for item in frozen],
            args.label_dir / csv_name,
            args.cache_dir / f"meld_{split}_audio_handcrafted_v1.npz",
            expected_files=frozen,
            expected_csv_sha256=csv_hash,
            audio_config={
                **config["audio_features"],
                "version": config["audio_features"]["version"],
            },
        )
    payload = {
        "protocol": config["protocol"],
        "scope": config["scope"],
        "source": source,
        "splits": reports,
        "test_opened": False,
    }
    write_json_atomic(payload, args.output)
    print(f"wrote {args.output}")
    for split, report in reports.items():
        print(split, report["status"], report["intersection_rows"], report["missing_official_keys"])


if __name__ == "__main__":
    main()
