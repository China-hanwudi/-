from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_audio_text_risk import run_audio_text_risk, write_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--train-audio", type=Path, required=True)
    parser.add_argument("--dev-audio", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "meld_audio_text_feasibility_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "meld_audio_text_risk_v1.json"
    )
    args = parser.parse_args()
    result, per_query = run_audio_text_risk(
        args.train_csv,
        args.dev_csv,
        args.train_audio,
        args.dev_audio,
        args.config,
    )
    write_outputs(result, per_query, args.output)
    print(json.dumps(result["gates"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
