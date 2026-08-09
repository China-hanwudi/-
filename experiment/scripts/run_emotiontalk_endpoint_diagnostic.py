from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_endpoint_diagnostic import run_endpoint_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sealed-role EmotionTalk endpoint utility diagnostic")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "emotiontalk_multimodal_external_v1.json")
    parser.add_argument("--diagnostic-config", type=Path, default=ROOT / "configs" / "emotiontalk_scu_endpoint_diagnostic_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-cache", type=Path)
    args = parser.parse_args()
    result = run_endpoint_diagnostic(
        args.data_dir.resolve(),
        args.features.resolve(),
        args.base_config.resolve(),
        args.diagnostic_config.resolve(),
        args.output.resolve(),
        args.private_cache.resolve() if args.private_cache else None,
    )
    print(json.dumps({
        "gate_checks": result["gate_checks"],
        "proceed_to_stochastic_subset_augmentation": result["proceed_to_stochastic_subset_augmentation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
