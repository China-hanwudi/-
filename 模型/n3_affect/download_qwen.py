"""Download main-line Qwen3-4B-Instruct-2507 into local artifacts (optional)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "pretrained"
        / "qwen3-4b-instruct-2507",
    )
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc
    args.dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.model_id, local_dir=str(args.dest))
    print(f"downloaded {args.model_id} -> {path}")
    print("Note: keep large weight files local; do not commit unless using Git LFS intentionally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
