from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_multimodal_external import (  # noqa: E402
    Blocks,
    base_features,
    restricted_donor_indices,
    selector_features,
)


def sample_blocks() -> Blocks:
    current = {
        "text": sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "audio": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        "video": np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
    }
    history = {
        "text": sparse.csr_matrix([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "audio": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "video": np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    }
    quality_names = (
        "audio_duration_seconds", "audio_source_rate", "audio_source_channels",
        "video_total_frames", "video_fps", "video_decoded_frames",
        "video_face_detection_rate", "video_mean_crop_area_fraction",
    )
    quality = np.asarray([
        [1, 44100, 2, 30, 30, 4, 1, .2],
        [2, 16000, 1, 60, 30, 4, .5, .4],
        [3, 44100, 2, 90, 30, 4, 0, 1],
    ], dtype=np.float32)
    return Blocks(current, history, quality, quality.copy(), quality_names, np.asarray([0, 1, 2]))


def test_current_only_has_zero_history_and_count() -> None:
    blocks = sample_blocks()
    actual = base_features(blocks, ("text", "audio", "video"), use_history=True)
    current = base_features(blocks, ("text", "audio", "video"), use_history=False)
    assert actual.shape == current.shape
    assert actual[2, -1] > 0
    assert current[:, -1].nnz == 0
    assert (actual != current).nnz > 0


def test_selector_modalities_have_nested_feature_schemas() -> None:
    blocks = sample_blocks()
    full = np.asarray([[.7, .3], [.4, .6], [.8, .2]])
    current = np.asarray([[.6, .4], [.5, .5], [.7, .3]])
    _, text = selector_features(blocks, ("text",), full, current)
    all_x, all_names = selector_features(blocks, ("text", "audio", "video"), full, current)
    assert set(text).issubset(set(all_names))
    assert all_x.shape == (3, len(all_names))
    assert np.isfinite(all_x).all()


def test_restricted_donor_never_uses_same_dialogue() -> None:
    rows = []
    for dialogue in range(3):
        for speaker in ("01", "02"):
            rows.append({"key": f"k{dialogue}{speaker}", "group": "G00001", "dialogue": f"{dialogue:02d}", "speaker": speaker})
    frame = pd.DataFrame(rows)
    counts = np.ones(len(frame), dtype=int)
    donor = restricted_donor_indices(frame, counts, np.random.default_rng(7), [1, 2, 3, 5, 9])
    for query, selected in enumerate(donor):
        assert frame.iloc[query]["dialogue"] != frame.iloc[selected]["dialogue"]
