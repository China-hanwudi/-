from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_text_pilot import (  # noqa: E402
    aggregate_history,
    build_history_indices,
    load_labeled_split,
    zero_history_features,
)


class MeldTextPilotTests(unittest.TestCase):
    def test_history_indices_are_same_speaker_and_strictly_past(self) -> None:
        frame = pd.DataFrame(
            {
                "Dialogue_ID": [0, 0, 0, 0],
                "Utterance_ID": [0, 1, 2, 3],
                "Speaker": ["A", "B", "A", "A"],
                "Utterance": ["a", "b", "c", "d"],
                "Emotion": ["neutral"] * 4,
                "_row_id": np.arange(4),
            }
        )
        self.assertEqual(build_history_indices(frame), ((), (), (0,), (0, 2)))

    def test_recent_and_full_aggregation(self) -> None:
        current = sparse.identity(4, format="csr", dtype=float)
        histories = ((), (), (0,), (0, 2))
        recent, recent_count = aggregate_history("recent_1", current, histories)
        full, full_count = aggregate_history("full_history", current, histories)
        self.assertEqual(recent_count.tolist(), [0, 0, 1, 1])
        self.assertEqual(full_count.tolist(), [0, 0, 1, 2])
        np.testing.assert_allclose(recent[3].toarray(), [[0, 0, 1, 0]])
        np.testing.assert_allclose(full[3].toarray(), [[0.5, 0, 0.5, 0]])

    def test_zero_history_keeps_fixed_feature_geometry(self) -> None:
        current = sparse.identity(3, format="csr", dtype=float)
        combined = zero_history_features(current)
        self.assertEqual(combined.shape, (3, 7))
        self.assertEqual(combined[:, 3:6].nnz, 0)

    def test_test_split_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "test labels are sealed"):
            load_labeled_split(Path("test_sent_emo.csv"))


if __name__ == "__main__":
    unittest.main()
