from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.data_contract import ContractError  # noqa: E402
from hva_affect.emotiontalk_media_contract import (  # noqa: E402
    _safe_posix_path,
    audit_emotiontalk_media,
)


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


class EmotionTalkMediaContractTests(unittest.TestCase):
    def test_unsafe_archive_paths_fail_closed(self) -> None:
        with self.assertRaises(ContractError):
            _safe_posix_path("../escape.mp4")

    def test_stream_audit_aligns_direct_wav_and_nested_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            metadata.mkdir()
            corpora = {
                "train_corpus": {"G00001_01_01_001": {"emo": 0, "val": 0.0}},
                "val_corpus": {"G00002_01_02_001": {"emo": 1, "val": 0.0}},
                "test_corpus": {"G00003_01_03_001": {"emo": 2, "val": 0.0}},
            }
            np.savez(metadata / "mm_label.npz", **corpora)
            keys = [key for corpus in corpora.values() for key in corpus]

            audio = root / "Audio.tar"
            with tarfile.open(audio, "w") as tar:
                for key in keys:
                    add_bytes(tar, f"Audio/json/{key}.json", json.dumps({"emotion_result": "hidden"}).encode())
                    add_bytes(tar, f"Audio/wav/{key}.wav", wav_bytes())

            multimodal = root / "Multimodal.tar"
            nested_bytes = io.BytesIO()
            with zipfile.ZipFile(nested_bytes, "w", zipfile.ZIP_DEFLATED) as nested:
                for key in keys:
                    nested.writestr(f"Multimodal/mp4/{key}.mp4", b"not-decoded-in-unit-test")
            with tarfile.open(multimodal, "w") as tar:
                add_bytes(tar, "Multimodal/mp4/bundle.zip", nested_bytes.getvalue())
                for key in keys:
                    add_bytes(tar, f"Multimodal/json/{key}.json", b"{}")
                    add_bytes(tar, f"Multimodal/mp4/{key}.mp4", b"not-decoded-in-unit-test")

            for archive_name, prefix in (("Text.tar", "Text"), ("Video.tar", "Video")):
                with tarfile.open(root / archive_name, "w:gz") as tar:
                    for key in keys:
                        add_bytes(tar, f"{prefix}/json/{key}.json", b"{}")

            files = []
            for name in ("Audio.tar", "Multimodal.tar", "Text.tar", "Video.tar"):
                path = root / name
                files.append({
                    "name": name,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sha256_verified": True,
                })
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset": "BAAI/Emotiontalk",
                "revision": "unit-test-revision",
                "files": files,
            }), encoding="utf-8")

            index = root / "index.csv.gz"
            report = audit_emotiontalk_media(
                manifest,
                metadata,
                index_output=index,
                probe_samples_per_split=0,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["coverage"]["audio"]["exact_match"])
            self.assertTrue(report["coverage"]["video"]["exact_match"])
            self.assertEqual(report["archives"]["Multimodal.tar"]["nested_zip_file_entries"], 3)
            self.assertEqual(report["nested_zip_video"]["overlap_with_primary_video_count"], 3)
            self.assertTrue(index.is_file())


if __name__ == "__main__":
    unittest.main()
