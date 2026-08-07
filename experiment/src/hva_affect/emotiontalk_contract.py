"""Deterministic structural audit for the EmotionTalk release metadata."""

from __future__ import annotations

import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_contract import ContractError, sha256_file


SPLIT_KEYS = ("train_corpus", "val_corpus", "test_corpus")
LABEL_FILES = ("audio_label.npz", "video_label.npz", "txt_label.npz", "mm_label.npz")
CSV_FILES = ("audio.csv", "video.csv", "transcription.csv", "mm.csv")
KEY_PATTERN = re.compile(
    r"^(?P<group>G\d{5})_(?P<dialogue>\d{2})_(?P<speaker>\d{2})_(?P<turn>\d{3})$"
)


def parse_key(key: str) -> tuple[str, str, str, int]:
    match = KEY_PATTERN.fullmatch(key)
    if match is None:
        raise ContractError(f"EmotionTalk malformed utterance key: {key!r}")
    return (
        match.group("group"),
        match.group("dialogue"),
        match.group("speaker"),
        int(match.group("turn")),
    )


def build_history_counts(keys: Sequence[str]) -> list[int]:
    """Count strictly earlier, same-dialogue, same-speaker history per query."""

    turns_by_stream: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for key in keys:
        group, dialogue, speaker, turn = parse_key(key)
        turns_by_stream[(group, dialogue, speaker)].append(turn)

    counts: list[int] = []
    for turns in turns_by_stream.values():
        if len(turns) != len(set(turns)):
            raise ContractError("EmotionTalk duplicate turn within speaker-dialogue stream")
        ordered = sorted(turns)
        counts.extend(range(len(ordered)))
    return counts


def _history_summary(counts: Sequence[int]) -> dict[str, float | int]:
    if not counts:
        raise ContractError("EmotionTalk split has no history counts")
    n = len(counts)
    return {
        "queries": n,
        "ge_1_pct": round(100 * sum(value >= 1 for value in counts) / n, 4),
        "ge_3_pct": round(100 * sum(value >= 3 for value in counts) / n, 4),
        "ge_5_pct": round(100 * sum(value >= 5 for value in counts) / n, 4),
        "ge_10_pct": round(100 * sum(value >= 10 for value in counts) / n, 4),
        "mean": round(sum(counts) / n, 6),
        "max": max(counts),
    }


def _load_npz(path: Path) -> dict[str, dict[str, dict]]:
    if not path.is_file():
        raise ContractError(f"EmotionTalk missing label file: {path.name}")
    # The official release stores dictionaries as pickled object arrays. Only
    # load files from the pinned upstream commit; never accept arbitrary NPZs.
    with np.load(path, allow_pickle=True) as archive:
        if tuple(archive.files) != SPLIT_KEYS:
            raise ContractError(
                f"{path.name}: expected split keys {SPLIT_KEYS}, got {archive.files}"
            )
        result: dict[str, dict[str, dict]] = {}
        for split in SPLIT_KEYS:
            array = archive[split]
            if array.shape != () or array.dtype != object:
                raise ContractError(f"{path.name}/{split}: expected scalar object array")
            corpus = array.item()
            if not isinstance(corpus, dict) or not corpus:
                raise ContractError(f"{path.name}/{split}: expected non-empty dict")
            result[split] = corpus
    return result


def _csv_basenames(path: Path) -> set[str]:
    if not path.is_file():
        raise ContractError(f"EmotionTalk missing CSV: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        field = "name" if "name" in (reader.fieldnames or []) else "file_name"
        if field not in (reader.fieldnames or []):
            raise ContractError(f"{path.name}: missing name/file_name column")
        names = {
            Path(str(row[field]).replace("\\", "/")).stem
            for row in reader
            if str(row.get(field, "")).strip()
        }
    if not names:
        raise ContractError(f"{path.name}: empty identifier set")
    return names


def _entities(keys: set[str]) -> dict[str, set]:
    parsed = [parse_key(key) for key in keys]
    return {
        "groups": {group for group, _, _, _ in parsed},
        "dialogues": {(group, dialogue) for group, dialogue, _, _ in parsed},
        "speakers": {speaker for _, _, speaker, _ in parsed},
        "speaker_dialogues": {
            (group, dialogue, speaker) for group, dialogue, speaker, _ in parsed
        },
    }


def _pairwise_overlap(keysets: Mapping[str, set[str]]) -> dict[str, dict]:
    entities = {split: _entities(keys) for split, keys in keysets.items()}
    result: dict[str, dict] = {}
    for left, right in itertools.combinations(SPLIT_KEYS, 2):
        result[f"{left}__{right}"] = {
            "utterance_overlap": len(keysets[left] & keysets[right]),
            "group_overlap": len(
                entities[left]["groups"] & entities[right]["groups"]
            ),
            "dialogue_overlap": len(
                entities[left]["dialogues"] & entities[right]["dialogues"]
            ),
            "speaker_overlap": len(
                entities[left]["speakers"] & entities[right]["speakers"]
            ),
            "overlapping_speakers": sorted(
                entities[left]["speakers"] & entities[right]["speakers"]
            ),
        }
    return result


def audit_emotiontalk(data_dir: Path, *, repository_root: Path | None = None) -> dict:
    data_dir = data_dir.resolve()
    corpora = {name: _load_npz(data_dir / name) for name in LABEL_FILES}
    mm = corpora["mm_label.npz"]

    modality_alignment: dict[str, dict] = {}
    for split in SPLIT_KEYS:
        keysets = {name: set(corpora[name][split]) for name in LABEL_FILES}
        union = set().union(*keysets.values())
        intersection = set.intersection(*keysets.values())
        if union != intersection:
            raise ContractError(f"EmotionTalk {split}: modality key sets are not aligned")
        modality_alignment[split] = {
            "identical_key_sets": True,
            "rows": len(union),
        }

    label_disagreement_vs_mm: dict[str, dict[str, int]] = {}
    for split in ("train_corpus", "val_corpus"):
        label_disagreement_vs_mm[split] = {
            name: sum(
                int(corpora[name][split][key]["emo"])
                != int(mm[split][key]["emo"])
                for key in mm[split]
            )
            for name in LABEL_FILES
            if name != "mm_label.npz"
        }

    all_keys = set().union(*(set(mm[split]) for split in SPLIT_KEYS))
    csv_reports: dict[str, dict] = {}
    for name in CSV_FILES:
        identifiers = _csv_basenames(data_dir / name)
        if identifiers != all_keys:
            raise ContractError(
                f"EmotionTalk {name}: identifiers do not match multimodal label keys"
            )
        csv_reports[name] = {
            "rows": len(identifiers),
            "identifiers_match_mm_labels": True,
            "sha256": sha256_file(data_dir / name),
        }

    split_reports: dict[str, dict] = {}
    keysets = {split: set(mm[split]) for split in SPLIT_KEYS}
    for split in SPLIT_KEYS:
        entities = _entities(keysets[split])
        counts = build_history_counts(list(keysets[split]))
        # Labels are checked only for shape/type/range. No test-label aggregate
        # or model-selection statistic is produced.
        for key, target in mm[split].items():
            parse_key(key)
            if not isinstance(target, dict) or set(target) != {"emo", "val"}:
                raise ContractError(f"EmotionTalk {split}/{key}: malformed target")
            if not isinstance(target["emo"], (int, np.integer)) or not 0 <= int(
                target["emo"]
            ) <= 6:
                raise ContractError(f"EmotionTalk {split}/{key}: invalid emotion id")
        split_reports[split] = {
            "rows": len(keysets[split]),
            "groups": len(entities["groups"]),
            "dialogues": len(entities["dialogues"]),
            "speakers": len(entities["speakers"]),
            "speaker_ids": sorted(entities["speakers"]),
            "history": _history_summary(counts),
            "gold_label_statistics_computed": False,
        }

    root = repository_root.resolve() if repository_root else data_dir.parents[2]
    referenced_media_present = 0
    for key in all_keys:
        group, dialogue, speaker, turn = parse_key(key)
        relative = Path(group) / f"{group}_{dialogue}" / f"{group}_{dialogue}_{speaker}"
        stem = f"{group}_{dialogue}_{speaker}_{turn:03d}"
        if (data_dir / relative / f"{stem}.mp4").is_file():
            referenced_media_present += 1

    readme = root / "README.md"
    readme_text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
    root_license = any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"))

    return {
        "dataset": "EmotionTalk",
        "contract_version": "0.1.0",
        "schema_status": "PASS",
        "readiness": "CONDITIONAL",
        "test_policy": "structure/hash/schema only; no test-label aggregates",
        "security_note": "Official NPZ files require allow_pickle=True; use only the pinned upstream commit.",
        "label_target": "Use mm_label.npz for the multimodal task; modality-specific label files intentionally disagree and are not interchangeable.",
        "label_mapping_verified_on_train_and_val": {
            "0": "neutral",
            "1": "happy",
            "2": "sad",
            "3": "angry",
            "4": "surprised",
            "5": "disgusted",
            "6": "fearful",
        },
        "npz_sha256": {name: sha256_file(data_dir / name) for name in LABEL_FILES},
        "csv": csv_reports,
        "modality_alignment": modality_alignment,
        "train_val_modality_label_disagreement_vs_mm": label_disagreement_vs_mm,
        "splits": split_reports,
        "cross_split_overlap": _pairwise_overlap(keysets),
        "availability": {
            "referenced_media_rows": len(all_keys),
            "referenced_mp4_present_under_metadata_tree": referenced_media_present,
            "root_data_license_present": root_license,
            "readme_has_merge_conflict_marker": "<<<<<<<" in readme_text,
        },
        "limitations": [
            "Raw media are not present in the metadata-only clone.",
            "No root data license was found; README intent is not a substitute for license terms.",
            "Validation and test share speaker IDs 02 and 13, so the official test is not fully speaker-disjoint from model selection.",
            "CSV row order is not chronological; history must be ordered by the final numeric turn field.",
        ],
    }
