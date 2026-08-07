"""Contracts and label-free acoustic feature extraction for MELD train/dev.

The Parquet re-publication is used only as a transport for embedded WAV bytes.
Speaker, turn, text and gold emotion always come from the frozen official CSV.
No test file is accepted by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.io import wavfile


KEY_PATTERN = re.compile(r"^dia(?P<dialogue>\d+)_utt(?P<utterance>\d+)\.mp4$")
EXPECTED_COLUMNS = {
    "file",
    "audio",
    "video",
    "transcription",
    "major_emotion",
}


class AudioContractError(ValueError):
    """Raised when the public audio transport violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_from_name(name: str) -> tuple[int, int]:
    match = KEY_PATTERN.fullmatch(str(name))
    if not match:
        raise AudioContractError(f"invalid MELD media key: {name!r}")
    return int(match.group("dialogue")), int(match.group("utterance"))


def name_from_key(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{int(dialogue_id)}_utt{int(utterance_id)}.mp4"


def _reject_test(path: Path) -> None:
    if "test" in path.name.casefold():
        raise AudioContractError(f"test artifact is sealed: {path.name}")


def load_official_split(path: Path) -> pd.DataFrame:
    _reject_test(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Utterance", "Speaker", "Emotion", "Dialogue_ID", "Utterance_ID"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AudioContractError(f"official CSV missing columns: {missing}")
    frame = frame.copy()
    frame["media_key"] = [
        name_from_key(dialogue_id, utterance_id)
        for dialogue_id, utterance_id in zip(frame["Dialogue_ID"], frame["Utterance_ID"])
    ]
    if frame["media_key"].duplicated().any():
        duplicates = frame.loc[frame["media_key"].duplicated(), "media_key"].tolist()
        raise AudioContractError(f"duplicate official media keys: {duplicates[:5]}")
    return frame


def _normalize_transcript(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\x92", "'").replace("’", "'").replace("‘", "'")
    return " ".join(text.casefold().split())


def _pcm_to_float(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.astype(np.float64).mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        values = samples.astype(np.float64) / scale
    else:
        values = samples.astype(np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -1.0, 1.0)


def _summary(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 7
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.quantile(values, 0.1)),
        float(np.quantile(values, 0.5)),
        float(np.quantile(values, 0.9)),
    ]


AUDIO_FEATURE_NAMES = (
    "duration_seconds",
    "signal_mean",
    "signal_std",
    "signal_rms",
    "signal_abs_mean",
    "signal_peak",
    "signal_crest",
    "signal_zero_crossing_rate",
    "signal_abs_q10",
    "signal_abs_q50",
    "signal_abs_q90",
    *[f"frame_rms_{name}" for name in ("mean", "std", "min", "max", "q10", "q50", "q90")],
    *[f"frame_zcr_{name}" for name in ("mean", "std", "min", "max", "q10", "q50", "q90")],
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "band_ratio_0_200",
    "band_ratio_200_400",
    "band_ratio_400_800",
    "band_ratio_800_1600",
    "band_ratio_1600_3200",
    "band_ratio_3200_nyquist",
)


def audio_feature_vector(
    payload: bytes,
    *,
    maximum_seconds: float = 12.0,
    frame_seconds: float = 0.025,
    hop_seconds: float = 0.01,
) -> tuple[np.ndarray, dict[str, float | int]]:
    try:
        sample_rate, samples = wavfile.read(io.BytesIO(payload))
    except Exception as exc:  # pragma: no cover - exact scipy decoder error is platform-specific
        raise AudioContractError("embedded audio is not a readable WAV") from exc
    if int(sample_rate) <= 0:
        raise AudioContractError("non-positive WAV sample rate")
    channels = int(samples.shape[1]) if samples.ndim == 2 else 1
    values = _pcm_to_float(samples)
    original_samples = int(values.size)
    if original_samples == 0:
        raise AudioContractError("empty WAV payload")
    maximum_samples = max(1, int(round(float(maximum_seconds) * sample_rate)))
    values = values[:maximum_samples]
    absolute = np.abs(values)
    rms = float(np.sqrt(np.mean(values * values) + 1e-15))
    zcr = float(np.mean(values[1:] * values[:-1] < 0)) if values.size > 1 else 0.0
    base = [
        original_samples / float(sample_rate),
        float(np.mean(values)),
        float(np.std(values)),
        rms,
        float(np.mean(absolute)),
        float(np.max(absolute)),
        float(np.max(absolute) / max(rms, 1e-8)),
        zcr,
        float(np.quantile(absolute, 0.1)),
        float(np.quantile(absolute, 0.5)),
        float(np.quantile(absolute, 0.9)),
    ]

    frame_length = max(8, int(round(frame_seconds * sample_rate)))
    hop_length = max(1, int(round(hop_seconds * sample_rate)))
    if values.size < frame_length:
        padded = np.pad(values, (0, frame_length - values.size))
        frames = padded.reshape(1, -1)
    else:
        frames = np.lib.stride_tricks.sliding_window_view(values, frame_length)[::hop_length]
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-15)
    frame_zcr = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)

    window = np.hanning(values.size)
    spectrum = np.abs(np.fft.rfft(values * window)) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / float(sample_rate))
    power_sum = float(spectrum.sum())
    if power_sum <= 1e-20:
        centroid = bandwidth = rolloff = flatness = 0.0
        ratios = [0.0] * 6
    else:
        probability = spectrum / power_sum
        centroid = float(np.sum(frequencies * probability))
        bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * probability)))
        cumulative = np.cumsum(probability)
        rolloff = float(frequencies[min(int(np.searchsorted(cumulative, 0.85)), len(frequencies) - 1)])
        flatness = float(np.exp(np.mean(np.log(spectrum + 1e-20))) / (np.mean(spectrum) + 1e-20))
        edges = (0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, sample_rate / 2.0 + 1.0)
        ratios = []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (frequencies >= left) & (frequencies < right)
            ratios.append(float(spectrum[mask].sum() / power_sum))

    features = np.asarray(
        base
        + _summary(frame_rms)
        + _summary(frame_zcr)
        + [centroid, bandwidth, rolloff, flatness]
        + ratios,
        dtype=np.float64,
    )
    if features.shape != (len(AUDIO_FEATURE_NAMES),):
        raise AssertionError("audio feature dimension drift")
    metadata = {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": original_samples / float(sample_rate),
        "payload_bytes": len(payload),
    }
    return features, metadata


def _validate_parquet(path: Path) -> pq.ParquetFile:
    _reject_test(path)
    if not path.is_file():
        raise AudioContractError(f"missing Parquet: {path}")
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    if columns != EXPECTED_COLUMNS:
        raise AudioContractError(
            f"unexpected Parquet columns in {path.name}: {sorted(columns)}"
        )
    audio_type = parquet.schema_arrow.field("audio").type
    if not {field.name for field in audio_type} >= {"bytes", "path"}:
        raise AudioContractError(f"audio column lacks bytes/path in {path.name}")
    return parquet


def extract_and_audit_split(
    parquet_paths: Sequence[Path],
    official_csv: Path,
    output_npz: Path,
    *,
    expected_files: Sequence[Mapping[str, object]],
    expected_csv_sha256: str,
    audio_config: Mapping[str, object],
) -> dict:
    official = load_official_split(official_csv)
    csv_hash = sha256_file(official_csv)
    if csv_hash.casefold() != str(expected_csv_sha256).casefold():
        raise AudioContractError(
            f"official CSV hash mismatch: {csv_hash} != {expected_csv_sha256}"
        )
    expected_by_name = {str(item["name"]): item for item in expected_files}
    if {path.name for path in parquet_paths} != set(expected_by_name):
        raise AudioContractError("Parquet filenames do not match frozen configuration")

    observed_keys: list[str] = []
    observed_features: list[np.ndarray] = []
    transcript_by_key: dict[str, str] = {}
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    durations: list[float] = []
    total_payload_bytes = 0
    parquet_reports: list[dict] = []
    seen: set[str] = set()

    for path in parquet_paths:
        expected = expected_by_name[path.name]
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(expected["bytes"]):
            raise AudioContractError(f"size mismatch for {path.name}")
        if actual_hash.casefold() != str(expected["sha256"]).casefold():
            raise AudioContractError(f"SHA-256 mismatch for {path.name}")
        parquet = _validate_parquet(path)
        parquet_reports.append(
            {
                "name": path.name,
                "bytes": actual_size,
                "sha256": actual_hash,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
            }
        )
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                group_index, columns=["file", "audio", "transcription"]
            )
            for row in table.to_pylist():
                key = str(row["file"])
                key_from_name(key)
                if key in seen:
                    raise AudioContractError(f"duplicate Parquet key: {key}")
                seen.add(key)
                audio = row["audio"]
                if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
                    raise AudioContractError(f"missing embedded WAV bytes for {key}")
                feature, metadata = audio_feature_vector(
                    audio["bytes"],
                    maximum_seconds=float(audio_config["maximum_seconds"]),
                    frame_seconds=float(audio_config["frame_seconds"]),
                    hop_seconds=float(audio_config["hop_seconds"]),
                )
                observed_keys.append(key)
                observed_features.append(feature)
                transcript_by_key[key] = str(row["transcription"])
                sample_rates[int(metadata["sample_rate"])] += 1
                channels[int(metadata["channels"])] += 1
                durations.append(float(metadata["duration_seconds"]))
                total_payload_bytes += int(metadata["payload_bytes"])

    official_keys = set(official["media_key"].astype(str))
    parquet_keys = set(observed_keys)
    missing = sorted(official_keys - parquet_keys)
    unexpected = sorted(parquet_keys - official_keys)
    if unexpected:
        raise AudioContractError(f"unexpected Parquet keys: {unexpected[:5]}")

    official_text = dict(zip(official["media_key"], official["Utterance"].astype(str)))
    transcript_matches = sum(
        _normalize_transcript(official_text[key]) == _normalize_transcript(transcript_by_key[key])
        for key in parquet_keys
    )
    feature_array = np.vstack(observed_features).astype(np.float32)
    key_array = np.asarray(observed_keys, dtype=np.str_)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_suffix(output_npz.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            media_key=key_array,
            audio_features=feature_array,
            feature_names=np.asarray(AUDIO_FEATURE_NAMES, dtype=np.str_),
        )
    temporary.replace(output_npz)

    duration_array = np.asarray(durations, dtype=np.float64)
    return {
        "status": "PASS_WITH_MISSING" if missing else "PASS",
        "official_csv": {
            "name": official_csv.name,
            "sha256": csv_hash,
            "rows": int(len(official)),
        },
        "parquet": parquet_reports,
        "intersection_rows": len(observed_keys),
        "missing_official_keys": missing,
        "unexpected_keys": unexpected,
        "transcript_normalized_matches": transcript_matches,
        "transcript_normalized_match_rate": transcript_matches / len(observed_keys),
        "audio": {
            "feature_version": str(audio_config["version"]),
            "feature_dimension": len(AUDIO_FEATURE_NAMES),
            "sample_rate_counts": dict(sorted(sample_rates.items())),
            "channel_counts": dict(sorted(channels.items())),
            "duration_mean": float(duration_array.mean()),
            "duration_min": float(duration_array.min()),
            "duration_max": float(duration_array.max()),
            "embedded_payload_bytes": total_payload_bytes,
        },
        "output_npz": {
            "path": str(output_npz),
            "bytes": output_npz.stat().st_size,
            "sha256": sha256_file(output_npz),
        },
        "test_opened": False,
        "gold_label_source": "official CSV only",
        "parquet_gold_label_read": False,
        "visual_modality_present": False,
    }


def write_json_atomic(payload: Mapping[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
