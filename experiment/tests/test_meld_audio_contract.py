from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_audio_contract import (  # noqa: E402
    AUDIO_FEATURE_NAMES,
    AudioContractError,
    _reject_test,
    audio_feature_vector,
    key_from_name,
    name_from_key,
)


def _wav_bytes(sample_rate: int = 16000) -> bytes:
    time = np.arange(sample_rate // 4) / sample_rate
    samples = (0.25 * np.sin(2 * np.pi * 440 * time) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_media_key_round_trip() -> None:
    assert key_from_name("dia42_utt7.mp4") == (42, 7)
    assert name_from_key(42, 7) == "dia42_utt7.mp4"


def test_audio_features_are_finite_and_fixed_dimension() -> None:
    features, metadata = audio_feature_vector(_wav_bytes())
    assert features.shape == (len(AUDIO_FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert metadata["sample_rate"] == 16000
    assert metadata["duration_seconds"] == pytest.approx(0.25)


def test_test_artifacts_are_rejected() -> None:
    with pytest.raises(AudioContractError, match="sealed"):
        _reject_test(Path("test-00000.parquet"))
