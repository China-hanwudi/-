from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.data_contract import ContractError  # noqa: E402
from hva_affect.emotiontalk_contract import (  # noqa: E402
    CSV_FILES,
    LABEL_FILES,
    audit_emotiontalk,
    build_history_counts,
    parse_key,
)


class EmotionTalkContractTests(unittest.TestCase):
    def test_key_parser_and_history_are_strictly_past(self) -> None:
        self.assertEqual(
            parse_key("G00001_02_03_004"), ("G00001", "02", "03", 4)
        )
        counts = build_history_counts(
            ["G00001_02_03_004", "G00001_02_03_001", "G00001_02_03_009"]
        )
        self.assertEqual(sorted(counts), [0, 1, 2])

    def test_bad_key_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "malformed utterance key"):
            parse_key("G1_2_3_4")

    def test_audit_aligns_modalities_and_hides_test_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "EmotionTalk" / "dataset" / "mm-process"
            data.mkdir(parents=True)
            corpora = {
                "train_corpus": {
                    "G00001_01_01_001": {"emo": 0, "val": 0.0},
                    "G00001_01_01_002": {"emo": 1, "val": 1.0},
                },
                "val_corpus": {"G00002_01_02_001": {"emo": 2, "val": -1.0}},
                "test_corpus": {"G00003_01_02_001": {"emo": 3, "val": -2.0}},
            }
            for name in LABEL_FILES:
                np.savez(data / name, **corpora)
            all_keys = [key for corpus in corpora.values() for key in corpus]
            for name in CSV_FILES:
                field = "name" if name == "transcription.csv" else "file_name"
                with (data / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[field, "emotion"])
                    writer.writeheader()
                    for key in all_keys:
                        suffix = "" if field == "name" else ".mp4"
                        writer.writerow({field: f"G/{key}{suffix}", "emotion": "neutral"})
            report = audit_emotiontalk(data, repository_root=root)
            self.assertEqual(report["schema_status"], "PASS")
            self.assertEqual(report["readiness"], "CONDITIONAL")
            self.assertFalse(
                report["splits"]["test_corpus"]["gold_label_statistics_computed"]
            )
            self.assertEqual(
                report["cross_split_overlap"]["val_corpus__test_corpus"]["speaker_overlap"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
