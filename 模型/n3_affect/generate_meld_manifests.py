"""Generate official MELD train/val/test manifests with strict 3-history."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LABEL_ORDER = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABEL_ORDER)}

SPLIT_DIRS = {
    "train": ("train/train_sent_emo.csv", "train/train_splits"),
    "val": ("dev_sent_emo.csv", "dev/dev_splits_complete"),
    "test": ("test_sent_emo.csv", "test/output_repeated_splits_test"),
}


def build_manifests(raw_root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for split, (csv_name, video_dir_name) in SPLIT_DIRS.items():
        csv_path = raw_root / csv_name
        video_root = raw_root / video_dir_name
        rows = []
        with csv_path.open(encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                r["Dialogue_ID"] = int(r["Dialogue_ID"])
                r["Utterance_ID"] = int(r["Utterance_ID"])
                rows.append(r)
        # sort by dialogue_id then utterance_id
        rows.sort(key=lambda r: (r["Dialogue_ID"], r["Utterance_ID"]))
        # group by dialogue
        by_dia: dict[int, list[dict]] = {}
        for r in rows:
            by_dia.setdefault(r["Dialogue_ID"], []).append(r)
        out_rows = []
        missing = []
        for dia_id, dia_rows in by_dia.items():
            for i, r in enumerate(dia_rows):
                utt_id = r["Utterance_ID"]
                sample_id = f"{split}_dia{dia_id}_utt{utt_id}"
                text = r["Utterance"].strip()
                speaker = r["Speaker"].strip()
                label = r["Emotion"].strip().lower()
                label_id = LABEL_TO_ID.get(label, -1)
                video_name = f"dia{dia_id}_utt{utt_id}.mp4"
                video_path = video_root / video_name
                video_exists = video_path.is_file()
                # gather up to 3 previous turns from same dialogue, most recent first
                hist = []
                for j in reversed(range(max(0, i - 3), i)):
                    hr = dia_rows[j]
                    h_utt_id = hr["Utterance_ID"]
                    h_video_name = f"dia{dia_id}_utt{h_utt_id}.mp4"
                    h_video_path = video_root / h_video_name
                    hist.append({
                        "dialogue_id": dia_id,
                        "utterance_id": h_utt_id,
                        "text": hr["Utterance"].strip(),
                        "video_path": str(h_video_path),
                        "speaker": hr["Speaker"].strip(),
                        "video_exists": h_video_path.is_file(),
                    })
                # strict 3-history: pad older slots with empty slots to length 3
                while len(hist) < 3:
                    hist.append({
                        "dialogue_id": -1,
                        "utterance_id": -1,
                        "text": "",
                        "video_path": "",
                        "speaker": "",
                        "video_exists": False,
                    })
                h0, h1, h2 = hist[0], hist[1], hist[2]
                out_rows.append({
                    "sample_id": sample_id,
                    "dialogue_id": dia_id,
                    "utterance_id": utt_id,
                    "split": split,
                    "text": text,
                    "video_path": str(video_path),
                    "label": label,
                    "label_id": label_id,
                    "speaker": speaker,
                    "history_n": min(i, 3),
                    "history": json.dumps(hist, ensure_ascii=False),
                    "h0_video_path": h0["video_path"],
                    "h1_video_path": h1["video_path"],
                    "h2_video_path": h2["video_path"],
                    "h0_text": h0["text"],
                    "h1_text": h1["text"],
                    "h2_text": h2["text"],
                    "h0_speaker": h0["speaker"],
                    "h1_speaker": h1["speaker"],
                    "h2_speaker": h2["speaker"],
                    "video_exists": video_exists,
                    "video_missing": int(not video_exists),
                })
                if not video_exists:
                    missing.append({
                        "sample_id": sample_id,
                        "dialogue_id": dia_id,
                        "utterance_id": utt_id,
                        "video_path": str(video_path),
                    })
        out_csv = out_dir / f"{split}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        miss_path = out_dir / f"{split}_missing.json"
        miss_path.write_text(json.dumps({
            "split": split,
            "missing_count": len(missing),
            "missing": missing,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        all_reports[split] = {
            "csv": str(out_csv),
            "n_total": len(out_rows),
            "missing": len(missing),
            "missing_json": str(miss_path),
        }
    return all_reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("/data/shared/raw/meld/MELD.Raw"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    report = build_manifests(args.raw_root, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
