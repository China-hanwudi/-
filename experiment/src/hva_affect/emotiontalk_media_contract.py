"""Streaming structural audit for the pinned EmotionTalk media archives.

The audit never extracts an archive wholesale. Test labels are not read: only
the split key dictionaries in the official ``mm_label.npz`` are inspected.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import wave
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

import numpy as np

from .data_contract import ContractError
from .emotiontalk_contract import SPLIT_KEYS, parse_key


MEDIA_MODALITIES = ("audio", "video")
JSON_MODALITIES = ("audio_json", "multimodal_json", "text_json", "video_json")


def _safe_posix_path(raw: str) -> str:
    normalized = raw.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ContractError(f"unsafe archive path: {raw!r}")
    if path.parts and ":" in path.parts[0]:
        raise ContractError(f"unsafe archive drive path: {raw!r}")
    return path.as_posix()


def _key_from_media_path(path: str, suffix: str) -> str | None:
    pure = PurePosixPath(path)
    if pure.suffix.lower() != suffix:
        return None
    key = pure.stem
    try:
        parse_key(key)
    except ContractError:
        return None
    return key


def _field_paths(value: object, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.add(path)
            result.update(_field_paths(child, path))
    elif isinstance(value, list):
        for child in value:
            result.update(_field_paths(child, f"{prefix}[]"))
    return result


def _load_split_keys(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        raise ContractError(f"missing official split key file: {path}")
    result: dict[str, set[str]] = {}
    with np.load(path, allow_pickle=True) as archive:
        if tuple(archive.files) != SPLIT_KEYS:
            raise ContractError("mm_label.npz split schema changed")
        for split in SPLIT_KEYS:
            array = archive[split]
            if array.shape != () or array.dtype != object:
                raise ContractError(f"mm_label.npz/{split}: malformed object array")
            corpus = array.item()
            if not isinstance(corpus, dict) or not corpus:
                raise ContractError(f"mm_label.npz/{split}: empty or malformed corpus")
            keys = set(corpus)
            for key in keys:
                parse_key(key)
            result[split] = keys
    if any(result[left] & result[right] for i, left in enumerate(SPLIT_KEYS) for right in SPLIT_KEYS[i + 1 :]):
        raise ContractError("EmotionTalk split utterance keys overlap")
    return result


def _select_probe_keys(split_keys: dict[str, set[str]], per_split: int) -> set[str]:
    if per_split <= 0:
        return set()
    selected: set[str] = set()
    for split in SPLIT_KEYS:
        ordered = sorted(split_keys[split])
        if per_split == 1:
            indices = [len(ordered) // 2]
        else:
            indices = sorted({round(i * (len(ordered) - 1) / (per_split - 1)) for i in range(per_split)})
        selected.update(ordered[index] for index in indices)
    return selected


def _probe_wav(handle: BinaryIO) -> dict:
    with wave.open(handle, "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return {
            "codec": "pcm",
            "channels": wav.getnchannels(),
            "sample_rate": rate,
            "sample_width_bytes": wav.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / rate if rate else None,
        }


def _probe_mp4(entry: BinaryIO, *, ffprobe: str, size: int) -> dict:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(entry, handle, length=1024 * 1024)
        if temporary.stat().st_size != size:
            raise ContractError("sample MP4 extraction size mismatch")
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if completed.returncode != 0:
            raise ContractError(f"ffprobe failed: {completed.stderr.strip()[:300]}")
        return json.loads(completed.stdout)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _coverage(expected: set[str], observed: set[str]) -> dict:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return {
        "expected": len(expected),
        "observed_unique": len(observed),
        "matched": len(expected & observed),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": missing[:20],
        "unexpected_examples": unexpected[:20],
        "exact_match": not missing and not unexpected,
    }


def _archive_record(manifest: dict, name: str) -> dict:
    matches = [record for record in manifest.get("files", []) if record.get("name") == name]
    if len(matches) != 1:
        raise ContractError(f"download manifest must contain exactly one {name}")
    return matches[0]


def audit_emotiontalk_media(
    manifest_path: Path,
    metadata_dir: Path,
    *,
    index_output: Path | None = None,
    probe_samples_per_split: int = 2,
    ffprobe: str | None = None,
) -> dict:
    """Audit archive inventories and key alignment without whole extraction."""

    manifest_path = manifest_path.resolve()
    metadata_dir = metadata_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "BAAI/Emotiontalk":
        raise ContractError("unexpected dataset identity in download manifest")
    if not manifest.get("revision"):
        raise ContractError("download manifest has no pinned revision")

    split_keys = _load_split_keys(metadata_dir / "mm_label.npz")
    expected_all = set().union(*split_keys.values())
    probe_keys = _select_probe_keys(split_keys, probe_samples_per_split)
    ffprobe_path = ffprobe or shutil.which("ffprobe")
    if probe_keys and not ffprobe_path:
        raise ContractError("ffprobe is required when MP4 probing is enabled")

    keysets = {name: set() for name in (*MEDIA_MODALITIES, *JSON_MODALITIES)}
    duplicates: dict[str, list[str]] = {name: [] for name in keysets}
    index_rows: list[dict] = []
    json_fields: dict[str, Counter[str]] = {name: Counter() for name in JSON_MODALITIES}
    json_errors: list[str] = []
    probes = {"audio": {}, "video": {}}
    archive_reports: dict[str, dict] = {}
    nested_video_keys: set[str] = set()
    nested_video_duplicates: list[str] = []

    def add_key(modality: str, key: str, row: dict | None = None) -> None:
        if key in keysets[modality]:
            duplicates[modality].append(key)
        keysets[modality].add(key)
        if row is not None:
            index_rows.append(row)

    archive_specs = (
        ("Audio.tar", "Audio"),
        ("Multimodal.tar", "Multimodal"),
        ("Text.tar", "Text"),
        ("Video.tar", "Video"),
    )
    for archive_name, label in archive_specs:
        record = _archive_record(manifest, archive_name)
        archive_path = Path(record["path"])
        if not archive_path.is_file():
            raise ContractError(f"missing downloaded archive: {archive_path}")
        actual_size = archive_path.stat().st_size
        if actual_size != int(record["bytes"]):
            raise ContractError(f"{archive_name}: size changed after verified download")
        if not record.get("sha256_verified"):
            raise ContractError(f"{archive_name}: manifest does not attest SHA-256 verification")

        member_paths: set[str] = set()
        extension_counts: Counter[str] = Counter()
        regular_files = directories = payload_bytes = nested_zip_entries = 0
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar:
                safe_name = _safe_posix_path(member.name)
                if safe_name in member_paths:
                    raise ContractError(f"{archive_name}: duplicate tar member {safe_name}")
                member_paths.add(safe_name)
                if member.isdir():
                    directories += 1
                    continue
                if not member.isfile():
                    continue
                regular_files += 1
                payload_bytes += member.size
                suffix = PurePosixPath(safe_name).suffix.lower() or "<none>"
                extension_counts[suffix] += 1

                if suffix == ".json":
                    key = _key_from_media_path(safe_name, ".json")
                    modality = {
                        "Audio": "audio_json",
                        "Multimodal": "multimodal_json",
                        "Text": "text_json",
                        "Video": "video_json",
                    }[label]
                    if key:
                        add_key(modality, key)
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        json_errors.append(f"{archive_name}:{safe_name}: unreadable")
                    else:
                        try:
                            value = json.load(extracted)
                            json_fields[modality].update(_field_paths(value))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            json_errors.append(f"{archive_name}:{safe_name}:{exc}")

                if label == "Audio" and suffix == ".wav":
                    key = _key_from_media_path(safe_name, ".wav")
                    if key:
                        add_key("audio", key, {
                            "key": key,
                            "modality": "audio",
                            "container": archive_name,
                            "outer_member": safe_name,
                            "inner_member": "",
                            "bytes": member.size,
                            "compressed_bytes": member.size,
                            "crc32": "",
                        })
                        if key in probe_keys:
                            extracted = tar.extractfile(member)
                            if extracted is not None:
                                probes["audio"][key] = _probe_wav(extracted)

                if label == "Multimodal" and suffix == ".mp4":
                    key = _key_from_media_path(safe_name, ".mp4")
                    if key:
                        add_key("video", key, {
                            "key": key,
                            "modality": "video",
                            "container": archive_name,
                            "outer_member": safe_name,
                            "inner_member": "",
                            "bytes": member.size,
                            "compressed_bytes": member.size,
                            "crc32": "",
                        })
                        if key in probe_keys:
                            extracted = tar.extractfile(member)
                            if extracted is not None:
                                probes["video"][key] = _probe_mp4(
                                    extracted, ffprobe=str(ffprobe_path), size=member.size
                                )

                if label == "Multimodal" and suffix == ".zip":
                    extracted_zip = tar.extractfile(member)
                    if extracted_zip is None:
                        raise ContractError(f"unreadable nested ZIP: {safe_name}")
                    with zipfile.ZipFile(extracted_zip) as nested:
                        nested_names: set[str] = set()
                        for info in nested.infolist():
                            inner_name = _safe_posix_path(info.filename)
                            if inner_name in nested_names:
                                raise ContractError(f"duplicate nested ZIP member: {inner_name}")
                            nested_names.add(inner_name)
                            if info.is_dir():
                                continue
                            nested_zip_entries += 1
                            key = _key_from_media_path(inner_name, ".mp4")
                            if not key:
                                continue
                            if key in nested_video_keys:
                                nested_video_duplicates.append(key)
                            nested_video_keys.add(key)

        archive_reports[archive_name] = {
            "path": str(archive_path),
            "bytes": actual_size,
            "sha256": record.get("sha256"),
            "sha256_previously_verified": True,
            "tar_members": len(member_paths),
            "directories": directories,
            "regular_files": regular_files,
            "payload_bytes": payload_bytes,
            "extensions": dict(sorted(extension_counts.items())),
            "nested_zip_file_entries": nested_zip_entries,
        }

    coverage = {modality: _coverage(expected_all, keys) for modality, keys in keysets.items()}
    split_coverage: dict[str, dict] = {}
    for split, expected in split_keys.items():
        split_coverage[split] = {
            modality: {
                "expected": len(expected),
                "present": len(expected & keys),
                "missing": len(expected - keys),
            }
            for modality, keys in keysets.items()
        }

    if index_output is not None:
        index_output = index_output.resolve()
        index_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = index_output.with_name(index_output.name + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "key", "modality", "container", "outer_member", "inner_member",
                "bytes", "compressed_bytes", "crc32",
            ])
            writer.writeheader()
            writer.writerows(sorted(index_rows, key=lambda row: (row["key"], row["modality"])))
        os.replace(temporary, index_output)

    duplicate_counts = {name: len(values) for name, values in duplicates.items()}
    exact = all(item["exact_match"] for item in coverage.values())
    passed = exact and not any(duplicate_counts.values()) and not json_errors
    return {
        "dataset": "BAAI/Emotiontalk",
        "revision": manifest["revision"],
        "contract_version": "1.0.0",
        "status": "PASS" if passed else "FAIL",
        "policy": {
            "whole_archive_extraction": False,
            "test_labels_used": False,
            "test_structure_keys_used": True,
            "download_hash_policy": "Reuse official LFS SHA-256 values already verified in the pinned download manifest; verify current byte lengths before scanning.",
            "forbidden_model_inputs": [
                "Audio JSON sourceAttr/emo_cap/caption fields",
                "archive emotion_result fields",
                "modality-specific gold labels",
            ],
        },
        "expected_split_rows": {split: len(keys) for split, keys in split_keys.items()},
        "archives": archive_reports,
        "coverage": coverage,
        "split_coverage": split_coverage,
        "duplicate_key_counts": duplicate_counts,
        "nested_zip_video": {
            "role": "redundant auxiliary bundle; direct Multimodal tar MP4 members are the canonical media",
            "coverage": _coverage(expected_all, nested_video_keys),
            "duplicate_inside_nested_bundle_count": len(nested_video_duplicates),
            "overlap_with_primary_video_count": len(nested_video_keys & keysets["video"]),
        },
        "json_parse_error_count": len(json_errors),
        "json_parse_error_examples": json_errors[:20],
        "json_field_presence_counts": {
            modality: dict(sorted(counter.items())) for modality, counter in json_fields.items()
        },
        "sample_probes": probes,
        "probe_selection": "deterministic evenly spaced sorted utterance keys within each split",
        "index_output": str(index_output) if index_output else None,
        "limitations": [
            "Nested ZIP central directories and selected entries are read; every MP4 payload is not fully decoded.",
            "Archive SHA-256 values are not recomputed in this pass because they were already matched to official LFS OIDs after download.",
            "Media duration/codec checks are deterministic samples, while key coverage is exhaustive.",
        ],
    }
