"""Deterministic data contracts for HVA-Affect.

The module intentionally depends only on the Python standard library so the
data gate can run before a machine-learning environment is installed.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUIRED_MELD_COLUMNS = {
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
    "StartTime",
    "EndTime",
}
SPLITS = ("train", "dev", "test")


class ContractError(ValueError):
    """Raised when a dataset violates a frozen structural contract."""


@dataclass(frozen=True)
class HistoryPair:
    split: str
    dialogue_id: int
    query_utterance_id: int
    history_utterance_id: int
    speaker: str

    def __post_init__(self) -> None:
        if self.history_utterance_id >= self.query_utterance_id:
            raise ContractError("history must be strictly earlier than query")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, split: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContractError(f"missing split file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MELD_COLUMNS - columns)
        if missing:
            raise ContractError(f"{split} missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ContractError(f"{split} is empty")
    return rows


def _to_int(value: str, *, field: str, split: str, row_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{split} row {row_number}: {field} must be an integer"
        ) from exc


def validate_rows(rows: Sequence[Mapping[str, str]], split: str) -> None:
    seen: set[tuple[int, int]] = set()
    per_dialogue_ids: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        dialogue_id = _to_int(
            row.get("Dialogue_ID", ""), field="Dialogue_ID", split=split, row_number=index
        )
        utterance_id = _to_int(
            row.get("Utterance_ID", ""), field="Utterance_ID", split=split, row_number=index
        )
        key = (dialogue_id, utterance_id)
        if key in seen:
            raise ContractError(f"{split}: duplicate query key {key}")
        seen.add(key)
        per_dialogue_ids[dialogue_id].append(utterance_id)
        if not str(row.get("Speaker", "")).strip():
            raise ContractError(f"{split} row {index}: empty Speaker")
        if not str(row.get("Utterance", "")).strip():
            raise ContractError(f"{split} row {index}: empty Utterance")

    for dialogue_id, utterance_ids in per_dialogue_ids.items():
        ordered = sorted(utterance_ids)
        if ordered[0] < 0:
            raise ContractError(f"{split} dialogue {dialogue_id}: negative Utterance_ID")
        if len(ordered) != len(set(ordered)):
            raise ContractError(f"{split} dialogue {dialogue_id}: duplicate Utterance_ID")


def build_same_speaker_history(
    rows: Sequence[Mapping[str, str]], split: str
) -> tuple[list[HistoryPair], list[int]]:
    """Build all same-dialogue, same-speaker, strictly-past pairs.

    Returns every pair plus the number of available historical units for each
    query in chronological dialogue order. Gold emotion labels are never read.
    """

    validate_rows(rows, split)
    dialogues: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        dialogues[int(row["Dialogue_ID"])].append(row)

    pairs: list[HistoryPair] = []
    history_counts: list[int] = []
    for dialogue_id in sorted(dialogues):
        ordered = sorted(dialogues[dialogue_id], key=lambda row: int(row["Utterance_ID"]))
        prior_by_speaker: dict[str, list[int]] = defaultdict(list)
        for row in ordered:
            speaker = str(row["Speaker"])
            query_id = int(row["Utterance_ID"])
            prior_ids = prior_by_speaker[speaker]
            history_counts.append(len(prior_ids))
            for history_id in prior_ids:
                pairs.append(
                    HistoryPair(
                        split=split,
                        dialogue_id=dialogue_id,
                        query_utterance_id=query_id,
                        history_utterance_id=history_id,
                        speaker=speaker,
                    )
                )
            prior_by_speaker[speaker].append(query_id)
    return pairs, history_counts


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _normalize_content_text(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _overlap_counts(rows_by_split: Mapping[str, Sequence[Mapping[str, str]]]) -> dict:
    speakers = {
        split: {str(row["Speaker"]).strip() for row in rows}
        for split, rows in rows_by_split.items()
    }
    episodes = {
        split: {(str(row["Season"]), str(row["Episode"])) for row in rows}
        for split, rows in rows_by_split.items()
    }
    texts = {
        split: {_normalize_text(str(row["Utterance"])) for row in rows}
        for split, rows in rows_by_split.items()
    }
    clip_intervals = {
        split: {
            (
                str(row["Season"]),
                str(row["Episode"]),
                str(row["StartTime"]),
                str(row["EndTime"]),
            )
            for row in rows
        }
        for split, rows in rows_by_split.items()
    }
    rows_by_interval: dict[str, dict[tuple[str, str, str, str], list[Mapping[str, str]]]] = {}
    for split, rows in rows_by_split.items():
        grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[
                (
                    str(row["Season"]),
                    str(row["Episode"]),
                    str(row["StartTime"]),
                    str(row["EndTime"]),
                )
            ].append(row)
        rows_by_interval[split] = grouped
    result: dict[str, dict[str, int]] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        probable_duplicates = 0
        for interval in clip_intervals[left] & clip_intervals[right]:
            for left_row in rows_by_interval[left][interval]:
                for right_row in rows_by_interval[right][interval]:
                    left_text = _normalize_content_text(str(left_row["Utterance"]))
                    right_text = _normalize_content_text(str(right_row["Utterance"]))
                    if left_text and right_text:
                        similarity = difflib.SequenceMatcher(
                            None, left_text, right_text, autojunk=False
                        ).ratio()
                        if similarity >= 0.9:
                            probable_duplicates += 1
        result[f"{left}__{right}"] = {
            "speaker_overlap": len(speakers[left] & speakers[right]),
            "season_episode_overlap": len(episodes[left] & episodes[right]),
            "exact_normalized_text_overlap": len(texts[left] & texts[right]),
            "exact_clip_interval_overlap": len(
                clip_intervals[left] & clip_intervals[right]
            ),
            "probable_content_duplicate_pairs": probable_duplicates,
        }
    return result


def _history_summary(counts: Sequence[int]) -> dict[str, float | int]:
    n = len(counts)
    if n == 0:
        raise ContractError("cannot summarize empty history counts")
    return {
        "queries": n,
        "ge_1": sum(value >= 1 for value in counts),
        "ge_1_pct": round(100 * sum(value >= 1 for value in counts) / n, 4),
        "ge_2_pct": round(100 * sum(value >= 2 for value in counts) / n, 4),
        "ge_3_pct": round(100 * sum(value >= 3 for value in counts) / n, 4),
        "ge_5_pct": round(100 * sum(value >= 5 for value in counts) / n, 4),
        "mean": round(sum(counts) / n, 6),
        "max": max(counts),
    }


def audit_meld(data_dir: Path) -> dict:
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    split_reports: dict[str, dict] = {}
    for split in SPLITS:
        path = data_dir / f"{split}_sent_emo.csv"
        rows = _read_csv(path, split)
        rows_by_split[split] = rows
        pairs, counts = build_same_speaker_history(rows, split)
        split_reports[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(rows),
            "dialogues": len({row["Dialogue_ID"] for row in rows}),
            "speakers": len({row["Speaker"] for row in rows}),
            "history_pairs": len(pairs),
            "history": _history_summary(counts),
            "gold_label_statistics_computed": False,
        }

    return {
        "dataset": "MELD",
        "contract_version": "0.1.0",
        "status": "PASS",
        "test_policy": "structure_and_hash_only; no test-label statistics",
        "required_columns": sorted(REQUIRED_MELD_COLUMNS),
        "splits": split_reports,
        "cross_split_overlap": _overlap_counts(rows_by_split),
        "limitations": [
            "Raw audio/video files have not yet been inventoried.",
            "Speaker overlap is expected for recurring TV characters and must be reported.",
            "Exact text overlap is descriptive and must be investigated before modeling.",
        ],
    }


def write_json_atomic(payload: Mapping, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
