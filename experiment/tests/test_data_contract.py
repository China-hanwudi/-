from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.data_contract import (  # noqa: E402
    ContractError,
    REQUIRED_MELD_COLUMNS,
    audit_meld,
    build_same_speaker_history,
)


def row(dialogue: int, utterance: int, speaker: str, emotion: str = "neutral") -> dict[str, str]:
    return {
        "Utterance": f"text-{dialogue}-{utterance}",
        "Speaker": speaker,
        "Emotion": emotion,
        "Sentiment": "neutral",
        "Dialogue_ID": str(dialogue),
        "Utterance_ID": str(utterance),
        "Season": "1",
        "Episode": str(dialogue),
        "StartTime": "00:00:00,000",
        "EndTime": "00:00:01,000",
    }


class DataContractTests(unittest.TestCase):
    def test_history_is_same_speaker_and_strictly_past(self) -> None:
        rows = [row(0, 0, "A"), row(0, 1, "B"), row(0, 2, "A"), row(0, 3, "A")]
        pairs, counts = build_same_speaker_history(rows, "train")
        self.assertEqual(counts, [0, 0, 1, 2])
        self.assertEqual(
            [(pair.history_utterance_id, pair.query_utterance_id) for pair in pairs],
            [(0, 2), (0, 3), (2, 3)],
        )
        self.assertTrue(all(pair.history_utterance_id < pair.query_utterance_id for pair in pairs))

    def test_duplicate_query_key_fails_closed(self) -> None:
        rows = [row(0, 0, "A"), row(0, 0, "A")]
        with self.assertRaisesRegex(ContractError, "duplicate query key"):
            build_same_speaker_history(rows, "train")

    def test_empty_speaker_fails_closed(self) -> None:
        rows = [row(0, 0, "")]
        with self.assertRaisesRegex(ContractError, "empty Speaker"):
            build_same_speaker_history(rows, "train")

    def test_audit_never_computes_test_label_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            fields = sorted(REQUIRED_MELD_COLUMNS)
            for split in ("train", "dev", "test"):
                with (data_dir / f"{split}_sent_emo.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows([row(0, 0, "A", "joy"), row(0, 1, "A", "anger")])
            report = audit_meld(data_dir)
            self.assertFalse(report["splits"]["test"]["gold_label_statistics_computed"])
            self.assertIn("no test-label statistics", report["test_policy"])
            self.assertEqual(
                report["cross_split_overlap"]["train__test"]["exact_clip_interval_overlap"],
                1,
            )
            self.assertEqual(
                report["cross_split_overlap"]["train__test"]["probable_content_duplicate_pairs"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
