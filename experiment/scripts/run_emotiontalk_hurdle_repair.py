from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_hurdle_repair import run_hurdle_repair


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two-part EmotionTalk endpoint risk repair")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "emotiontalk_multimodal_external_v1.json")
    parser.add_argument("--diagnostic-config", type=Path, default=ROOT / "configs" / "emotiontalk_scu_endpoint_diagnostic_v1.json")
    parser.add_argument("--repair-config", type=Path, default=ROOT / "configs" / "emotiontalk_hurdle_repair_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_hurdle_repair(
        args.cache.resolve(),
        args.base_config.resolve(),
        args.diagnostic_config.resolve(),
        args.repair_config.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({
        "component_metrics": result["component_metrics"],
        "gate_checks": result["gate_checks"],
        "proceed_to_stochastic_subset_augmentation": result["proceed_to_stochastic_subset_augmentation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
