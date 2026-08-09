"""Optional local download for main-line Qwen3-Omni-30B-A3B-Instruct.

Never commit the downloaded weights (tens of GB). Keep outside git or under
artifacts/ only with Git LFS after explicit team decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "pretrained"
        / "qwen3-omni-30b-a3b-instruct",
    )
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc
    args.dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.model_id, local_dir=str(args.dest))
    print(f"downloaded {args.model_id} -> {path}")
    print("DO NOT git add this directory unless the team explicitly enables LFS for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
