from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_bidirectional_oof import run_bidirectional_oof  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe different-set bidirectional OOF supervision"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    result = run_bidirectional_oof(
        args.data_dir.resolve(),
        args.features.resolve(),
        args.base_config.resolve(),
        args.utility_config.resolve(),
        args.output.resolve(),
        args.private_cache.resolve(),
        args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
    )
    print(json.dumps({
        "status": result["status"],
        "task_counts": result["task_counts"],
        "target_summary": result["target_summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
