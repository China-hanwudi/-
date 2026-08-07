from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_audio_contract import AUDIO_FEATURE_NAMES  # noqa: E402
from hva_affect.meld_audio_text_risk import (  # noqa: E402
    AUDIO_SELECTOR_FEATURE_NAMES,
    TEXT_SELECTOR_FEATURE_NAMES,
    aggregate_dense_history,
    audio_augmented_selector_features,
    text_selector_features,
    zero_base_features,
)


def test_dense_history_is_strict_mean_of_given_indices() -> None:
    current = np.asarray([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    history, counts = aggregate_dense_history(current, [(), (0,), (0, 1)])
    np.testing.assert_allclose(history, [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    np.testing.assert_array_equal(counts, [0, 1, 2])


def test_zero_base_retains_current_audio_but_erases_history() -> None:
    current_text = sparse.csr_matrix([[1.0, 2.0], [3.0, 4.0]])
    current_audio = np.asarray([[5.0, 6.0], [7.0, 8.0]])
    output = zero_base_features(current_text, current_audio).toarray()
    np.testing.assert_allclose(output[:, :2], current_text.toarray())
    np.testing.assert_allclose(output[:, 2:4], 0.0)
    np.testing.assert_allclose(output[:, 4], 0.0)
    np.testing.assert_allclose(output[:, 5:7], current_audio)
    np.testing.assert_allclose(output[:, 7:9], 0.0)


def test_selector_features_have_frozen_geometry_and_no_labels() -> None:
    current = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    history = sparse.csr_matrix([[0.5, 0.0], [0.0, 0.5]])
    actual = np.full((2, 7), 1.0 / 7)
    zero = actual.copy()
    text = text_selector_features(current, history, np.asarray([1, 2]), actual, zero)
    assert text.shape == (2, len(TEXT_SELECTOR_FEATURE_NAMES))
    audio_current = np.ones((2, len(AUDIO_FEATURE_NAMES)))
    audio_history = np.zeros_like(audio_current)
    augmented = audio_augmented_selector_features(text, audio_current, audio_history)
    assert augmented.shape == (2, len(AUDIO_SELECTOR_FEATURE_NAMES))
    assert np.isfinite(augmented).all()


def test_audio_selector_prefix_is_identical_text_meta() -> None:
    rng = np.random.default_rng(7)
    text = rng.normal(size=(3, len(TEXT_SELECTOR_FEATURE_NAMES)))
    current = rng.normal(size=(3, len(AUDIO_FEATURE_NAMES)))
    history = rng.normal(size=(3, len(AUDIO_FEATURE_NAMES)))
    augmented = audio_augmented_selector_features(text, current, history)
    np.testing.assert_allclose(augmented[:, : text.shape[1]], text)
