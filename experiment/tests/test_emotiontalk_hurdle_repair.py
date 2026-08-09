from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_hurdle_repair import compose_expected_regret


def test_compose_expected_regret_uses_sign_and_both_magnitudes():
    probability = np.asarray([0.1, 0.9])
    harm = np.asarray([2.0, 2.0])
    benefit = np.asarray([1.0, 1.0])
    result = compose_expected_regret(probability, harm, benefit)
    assert np.allclose(result, [-0.7, 1.7])


def test_compose_expected_regret_rejects_invalid_components():
    with np.testing.assert_raises(ValueError):
        compose_expected_regret(np.asarray([1.1]), np.asarray([1.0]), np.asarray([1.0]))
    with np.testing.assert_raises(ValueError):
        compose_expected_regret(np.asarray([0.5]), np.asarray([-1.0]), np.asarray([1.0]))
