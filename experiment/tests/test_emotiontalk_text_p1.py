from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_text_p1 import (  # noqa: E402
    _restricted_candidate_resampling,
    build_history_indices,
    interaction_sets,
    load_emotiontalk_split,
)
from hva_affect.emotiontalk_point_risk import FEATURE_NAMES, selector_features  # noqa: E402


class EmotionTalkTextP1Tests(unittest.TestCase):
    def test_test_split_fails_closed_before_file_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "test labels are sealed"):
            load_emotiontalk_split(Path("missing"), "test_corpus")

    def test_histories_are_key_ordered_not_row_ordered(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["G00001"] * 5,
                "dialogue": ["01"] * 5,
                "speaker": ["02"] * 5,
                "turn": [5, 1, 4, 2, 3],
                "_row_id": list(range(5)),
            }
        )
        histories = build_history_indices(frame)
        self.assertEqual(histories[0], (1, 3, 4, 2))
        self.assertEqual(histories[1], tuple())
        for query, prior in enumerate(histories):
            self.assertTrue(all(frame.iloc[p]["turn"] < frame.iloc[query]["turn"] for p in prior))

    def test_fixed_cardinality_roles_are_distinct_and_equal_size(self) -> None:
        histories = (tuple(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
        sets, variable_mask, fixed_mask = interaction_sets(histories)
        self.assertEqual(int(variable_mask.sum()), 3)
        self.assertEqual(int(fixed_mask.sum()), 1)
        query = 4
        self.assertEqual(len(sets["s1_placebo"][query]), 2)
        self.assertEqual(len(sets["s1_candidate"][query]), 2)
        self.assertEqual(len(sets["s2_placebo"][query]), 2)
        self.assertEqual(len(sets["s2_candidate"][query]), 2)
        self.assertEqual(sets["s1_candidate"][query], (2, 3))
        self.assertEqual(sets["s2_candidate"][query], (1, 3))

    def test_speaker_resampling_excludes_real_roles(self) -> None:
        frame = pd.DataFrame({"speaker": ["01"] * 9})
        histories = tuple(tuple(range(i)) for i in range(9))
        mask = np.asarray([len(values) >= 4 for values in histories])
        replacement = _restricted_candidate_resampling(
            np.random.default_rng(7), frame, histories, mask
        )
        for query, value in replacement.items():
            self.assertNotIn(value, set(histories[query][-3:]) | {query})

    def test_point_risk_features_are_label_free_and_fixed_width(self) -> None:
        current = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        history = sparse.csr_matrix([[0.5, 0.5], [0.0, 1.0]])
        counts = np.asarray([1, 2])
        actual = np.asarray([[0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3], [0.2] * 4 + [0.1] * 3])
        actual = actual / actual.sum(axis=1, keepdims=True)
        zero = np.full((2, 7), 1 / 7)
        features = selector_features(current, history, counts, actual, zero)
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
