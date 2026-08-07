from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_emotiontalk_encoders import (  # noqa: E402
    audio_embedding,
    copy_member,
    face_or_full_frames,
    load_face_detector,
    read_wav,
    uniform_frames,
    video_embedding,
)
from hva_affect.data_contract import ContractError, sha256_file, write_json_atomic  # noqa: E402
from hva_affect.emotiontalk_media_contract import _load_split_keys  # noqa: E402


ALLOWED_SPLITS = ("train_corpus", "val_corpus")
AUDIO_DIM = 1536
VIDEO_DIM = 768
QUALITY_NAMES = (
    "audio_duration_seconds",
    "audio_source_rate",
    "audio_source_channels",
    "video_total_frames",
    "video_fps",
    "video_decoded_frames",
    "video_face_detection_rate",
    "video_mean_crop_area_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen EmotionTalk media features")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--wavlm", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--audio-tar", type=Path, required=True)
    parser.add_argument("--video-tar", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def config_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_allowed_keys(metadata_dir: Path, limit: int | None) -> tuple[list[str], np.ndarray]:
    splits = _load_split_keys(metadata_dir / "mm_label.npz")
    keys: list[str] = []
    names: list[str] = []
    for split in ALLOWED_SPLITS:
        selected = sorted(splits[split])
        if limit is not None:
            selected = selected[:limit]
        keys.extend(selected)
        names.extend([split] * len(selected))
    if set(keys) & splits["test_corpus"]:
        raise ContractError("sealed test key entered feature extraction")
    return keys, np.asarray(names, dtype="U16")


def load_media_index(path: Path, allowed: set[str]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["key"]
            modality = row["modality"]
            if key not in allowed or modality not in ("audio", "video"):
                continue
            pair = (key, modality)
            if pair in result:
                raise ContractError(f"duplicate media index pair: {pair}")
            if row.get("inner_member"):
                raise ContractError("canonical feature extraction must not use nested ZIP media")
            result[pair] = row["outer_member"]
    expected = {(key, modality) for key in allowed for modality in ("audio", "video")}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise ContractError(f"media index incomplete for allowed splits: {missing[:20]}")
    return result


def open_array(path: Path, shape: tuple[int, ...], dtype: str):
    if path.is_file():
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ContractError(f"resume array contract mismatch: {path.name}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def initialize_work(
    work_dir: Path, keys: list[str], splits: np.ndarray, config_digest: str
) -> dict[str, np.ndarray]:
    work_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": config_digest,
        "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "rows": len(keys),
        "allowed_splits": list(ALLOWED_SPLITS),
    }
    identity_path = work_dir / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ContractError("resume work identity does not match frozen extraction")
    else:
        write_json_atomic(identity, identity_path)
        np.save(work_dir / "keys.npy", np.asarray(keys, dtype="U20"), allow_pickle=False)
        np.save(work_dir / "splits.npy", splits, allow_pickle=False)
    return {
        "audio": open_array(work_dir / "audio.npy", (len(keys), AUDIO_DIM), "float32"),
        "video": open_array(work_dir / "video.npy", (len(keys), VIDEO_DIM), "float32"),
        "quality": open_array(work_dir / "quality.npy", (len(keys), len(QUALITY_NAMES)), "float32"),
        "audio_done": open_array(work_dir / "audio_done.npy", (len(keys),), "uint8"),
        "video_done": open_array(work_dir / "video_done.npy", (len(keys),), "uint8"),
    }


def flush_all(arrays: dict[str, np.ndarray]) -> None:
    for value in arrays.values():
        if hasattr(value, "flush"):
            value.flush()


def append_failure(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_audio(args, keys, media_index, arrays, device) -> dict:
    pending = [index for index, done in enumerate(arrays["audio_done"]) if not done]
    if not pending:
        return {"processed": 0, "seconds": 0.0, "failures": 0, "resumed_complete": True}
    processor = AutoFeatureExtractor.from_pretrained(args.wavlm, local_files_only=True)
    model = AutoModel.from_pretrained(args.wavlm, local_files_only=True).eval().to(device)
    torch.cuda.reset_peak_memory_stats()
    failures = 0
    started = time.perf_counter()
    with tarfile.open(args.audio_tar, "r:") as tar:
        # Build the tar index once; subsequent getmember calls are O(1).
        tar.getmembers()
        for ordinal, index in enumerate(pending, 1):
            key = keys[index]
            try:
                source = tar.extractfile(tar.getmember(media_index[(key, "audio")]))
                if source is None:
                    raise RuntimeError("unreadable WAV member")
                waveform, _, metadata = read_wav(source)
                feature = audio_embedding(model, processor, waveform, device)
                if feature.shape != (AUDIO_DIM,) or not np.isfinite(feature).all():
                    raise RuntimeError("invalid WavLM feature")
                arrays["audio"][index] = feature
                arrays["quality"][index, 0:3] = (
                    metadata["duration_seconds"],
                    metadata["source_rate"],
                    metadata["source_channels"],
                )
                arrays["audio_done"][index] = 1
            except Exception as exc:
                failures += 1
                append_failure(args.work_dir / "failures.jsonl", {
                    "stage": "audio", "key": key, "error": f"{type(exc).__name__}: {exc}"
                })
            if ordinal % args.checkpoint_every == 0 or ordinal == len(pending):
                flush_all(arrays)
                elapsed = time.perf_counter() - started
                print(f"AUDIO {ordinal}/{len(pending)} elapsed={elapsed:.1f}s failures={failures}", flush=True)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()
    return {"processed": len(pending), "seconds": seconds, "failures": failures, "peak_cuda_mib": peak}


def extract_video(args, keys, media_index, arrays, device) -> dict:
    pending = [index for index, done in enumerate(arrays["video_done"]) if not done]
    if not pending:
        return {"processed": 0, "seconds": 0.0, "failures": 0, "resumed_complete": True}
    processor = AutoImageProcessor.from_pretrained(args.dinov2, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(args.dinov2, local_files_only=True).eval().to(device)
    detector = load_face_detector()
    torch.cuda.reset_peak_memory_stats()
    failures = 0
    started = time.perf_counter()
    with tarfile.open(args.video_tar, "r:") as tar:
        tar.getmembers()
        for ordinal, index in enumerate(pending, 1):
            key = keys[index]
            temporary = None
            try:
                temporary = copy_member(tar, media_index[(key, "video")], ".mp4")
                frames, decode = uniform_frames(temporary, 4)
                images, faces = face_or_full_frames(frames, detector)
                feature = video_embedding(model, processor, images, device)
                if feature.shape != (VIDEO_DIM,) or not np.isfinite(feature).all():
                    raise RuntimeError("invalid DINOv2 feature")
                arrays["video"][index] = feature
                arrays["quality"][index, 3:8] = (
                    decode["total_frames"],
                    decode["fps"],
                    decode["decoded_frames"],
                    faces["face_detection_rate"],
                    faces["mean_crop_area_fraction"],
                )
                arrays["video_done"][index] = 1
            except Exception as exc:
                failures += 1
                append_failure(args.work_dir / "failures.jsonl", {
                    "stage": "video", "key": key, "error": f"{type(exc).__name__}: {exc}"
                })
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            if ordinal % args.checkpoint_every == 0 or ordinal == len(pending):
                flush_all(arrays)
                elapsed = time.perf_counter() - started
                print(f"VIDEO {ordinal}/{len(pending)} elapsed={elapsed:.1f}s failures={failures}", flush=True)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()
    return {"processed": len(pending), "seconds": seconds, "failures": failures, "peak_cuda_mib": peak}


def export_features(args, keys, splits, arrays, config_digest) -> dict:
    audio_done = np.asarray(arrays["audio_done"], dtype=bool)
    video_done = np.asarray(arrays["video_done"], dtype=bool)
    complete = audio_done & video_done
    finite = np.isfinite(arrays["audio"]).all(axis=1) & np.isfinite(arrays["video"]).all(axis=1)
    if not complete.all() or not finite.all():
        return {
            "status": "FAIL",
            "rows": len(keys),
            "audio_complete": int(audio_done.sum()),
            "video_complete": int(video_done.sum()),
            "finite_complete_rows": int((complete & finite).sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            keys=np.asarray(keys, dtype="U20"),
            splits=splits,
            audio_features=np.asarray(arrays["audio"]),
            video_features=np.asarray(arrays["video"]),
            quality=np.asarray(arrays["quality"]),
            quality_names=np.asarray(QUALITY_NAMES, dtype="U40"),
            config_sha256=np.asarray(config_digest),
        )
    os.replace(temporary, args.output)
    return {
        "status": "PASS",
        "rows": len(keys),
        "audio_complete": int(audio_done.sum()),
        "video_complete": int(video_done.sum()),
        "finite_complete_rows": int((complete & finite).sum()),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
    }


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ContractError("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden for this frozen run")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if tuple(config["allowed_splits"]) != ALLOWED_SPLITS or config["sealed_split"] != "test_corpus":
        raise ContractError("feature config split policy changed")
    if args.wavlm.name != config["audio"]["revision"] or args.dinov2.name != config["video"]["revision"]:
        raise ContractError("model snapshot revision does not match frozen config")
    digest = config_sha(args.config)
    keys, splits = load_allowed_keys(args.metadata_dir, args.limit)
    media_index = load_media_index(args.index, set(keys))
    arrays = initialize_work(args.work_dir, keys, splits, digest)

    torch.manual_seed(config["numeric"]["seed"])
    torch.cuda.manual_seed_all(config["numeric"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    overall = time.perf_counter()
    audio_report = extract_audio(args, keys, media_index, arrays, device)
    video_report = extract_video(args, keys, media_index, arrays, device)
    flush_all(arrays)
    export = export_features(args, keys, splits, arrays, digest)
    report = {
        "dataset": config["dataset"],
        "dataset_revision": config["dataset_revision"],
        "config": str(args.config),
        "config_sha256": digest,
        "allowed_splits": list(ALLOWED_SPLITS),
        "sealed_test_rows_extracted": 0,
        "limit_per_allowed_split": args.limit,
        "rows_by_split": {split: int((splits == split).sum()) for split in ALLOWED_SPLITS},
        "audio": audio_report,
        "video": video_report,
        "export": export,
        "elapsed_seconds": time.perf_counter() - overall,
        "label_fields_read": False,
        "temporary_media_retained": False,
        "work_dir_retained_for_resume_and_audit": str(args.work_dir.resolve()),
    }
    write_json_atomic(report, args.report.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if export["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
