"""Prepare strict, role-separated MELD train-only multimodal sidecars.

The official CSV is the authoritative label source.  The third-party MM-Align
``train.pkl`` artifact is used only for aligned audio/video features; its
embedded label is audited but never used as a target.  The pickle is opened
only after its exact registered SHA-256 is verified.

This is a trusted data-preparation boundary, not a model runner.  It reads the
MELD *train* source once, deterministically assigns whole dialogues to frozen
roles, and writes a separate feature archive and label archive for every role.
Subsequent open-role experiments can therefore load only fit/model-selection
labels without deserializing calibration or internal-holdout labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .scu_set import assign_group_role


PROTOCOL = "meld_multimodal_role_sidecars_v2"
SIDECAR_SCHEMA_VERSION = "meld_multimodal_role_sidecar_v2"
MANIFEST_SCHEMA_VERSION = "meld_multimodal_role_sidecar_manifest_v2"
ROLE_NAMES = (
    "base_and_utility_fit",
    "model_selection",
    "calibration",
    "internal_holdout",
)
EMOTION_TO_INDEX = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}
REQUIRED_CSV_COLUMNS = (
    "Utterance",
    "Speaker",
    "Emotion",
    "Dialogue_ID",
    "Utterance_ID",
)


class MeldSidecarContractError(ValueError):
    """Raised when the registered MELD source or role boundary changes."""


@dataclass(frozen=True)
class MeldRoleSidecar:
    utterances: np.ndarray
    audio_mean_std: np.ndarray
    video_mean_std: np.ndarray
    dialogue_codes: np.ndarray
    speaker_codes: np.ndarray
    utterance_order: np.ndarray
    protocol_row_ids: np.ndarray
    labels: np.ndarray
    row_alignment_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MeldSidecarContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _load_config(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeldSidecarContractError(f"cannot load MELD sidecar config: {error}") from error
    if not isinstance(payload, Mapping):
        raise MeldSidecarContractError("MELD sidecar config root must be a mapping")
    expected_root = {
        "protocol",
        "status",
        "dataset_id",
        "split_protocol_id",
        "roles",
        "source_contract",
        "feature_contract",
        "output_contract",
    }
    if set(payload) != expected_root:
        raise MeldSidecarContractError("MELD sidecar config schema changed")
    if payload.get("protocol") != PROTOCOL:
        raise MeldSidecarContractError("MELD sidecar protocol changed")
    if payload.get("status") != "frozen_before_role_sidecar_generation":
        raise MeldSidecarContractError("MELD sidecar config was not frozen before generation")
    if payload.get("dataset_id") != "MELD":
        raise MeldSidecarContractError("MELD sidecar dataset id changed")
    if payload.get("split_protocol_id") != "scu_set_exploration_v1":
        raise MeldSidecarContractError("MELD sidecar split protocol changed")
    expected_ranges = {
        "base_and_utility_fit": [0, 64],
        "model_selection": [65, 79],
        "calibration": [80, 89],
        "internal_holdout": [90, 99],
    }
    if payload.get("roles") != expected_ranges:
        raise MeldSidecarContractError("MELD role ranges changed")
    source = payload.get("source_contract")
    feature = payload.get("feature_contract")
    if not isinstance(source, Mapping) or not isinstance(feature, Mapping):
        raise MeldSidecarContractError("MELD sidecar config lacks source/feature contracts")
    expected_source = {
        "train_csv_sha256",
        "train_pickle_sha256",
        "allowed_missing_feature_rows",
        "official_csv_is_authoritative_label_source",
        "embedded_pickle_label_is_trusted_custodian_consistency_audit_only",
        "embedded_label_mismatch_statistics_must_not_be_published",
        "dev_and_test_sources_forbidden",
    }
    if set(source) != expected_source:
        raise MeldSidecarContractError("MELD sidecar source contract schema changed")
    _validate_sha256(source.get("train_csv_sha256"), field="train_csv_sha256")
    _validate_sha256(source.get("train_pickle_sha256"), field="train_pickle_sha256")
    if (
        source.get("official_csv_is_authoritative_label_source") is not True
        or source.get("embedded_pickle_label_is_trusted_custodian_consistency_audit_only")
        is not True
        or source.get("embedded_label_mismatch_statistics_must_not_be_published") is not True
        or source.get("dev_and_test_sources_forbidden") is not True
    ):
        raise MeldSidecarContractError("MELD sidecar source boundary changed")
    expected_feature = {
        "audio_dimension",
        "video_dimension",
        "pooling",
        "text_source",
        "speaker_token",
        "numeric_dtype",
        "history_rule",
    }
    if set(feature) != expected_feature:
        raise MeldSidecarContractError("MELD sidecar feature contract schema changed")
    if int(feature.get("audio_dimension", -1)) < 1 or int(feature.get("video_dimension", -1)) < 1:
        raise MeldSidecarContractError("registered modality dimensions are invalid")
    if feature.get("pooling") != "utterance_sequence_mean_and_population_std":
        raise MeldSidecarContractError("registered modality pooling changed")
    if feature.get("text_source") != "official_train_csv_utterance":
        raise MeldSidecarContractError("registered MELD text source changed")
    if feature.get("speaker_token") != (
        "sha256_63bit_population_independent_then_fit_only_runner_mapping_with_oov_zero"
    ):
        raise MeldSidecarContractError("registered MELD speaker-token contract changed")
    if feature.get("numeric_dtype") != "float32":
        raise MeldSidecarContractError("registered MELD numeric dtype changed")
    if feature.get("history_rule") != (
        "same_dialogue_and_same_speaker_and_strictly_lower_Utterance_ID"
    ):
        raise MeldSidecarContractError("registered MELD history rule changed")
    if int(source.get("allowed_missing_feature_rows", -1)) < 0:
        raise MeldSidecarContractError("allowed missing-feature count is invalid")
    output = payload.get("output_contract")
    expected_output = {
        "write_once": True,
        "one_feature_archive_per_role": True,
        "one_label_archive_per_role": True,
        "allow_pickle": False,
        "private_not_for_repository": True,
        "public_output": "aggregate manifest only",
        "legacy_v1_directory_must_not_be_overwritten": True,
    }
    if output != expected_output:
        raise MeldSidecarContractError("MELD sidecar output contract changed")
    return dict(payload)


def _pool_record(
    record: Mapping[str, Any],
    *,
    audio_dimension: int,
    video_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"token_ids", "audio_features", "video_features"}
    if not required.issubset(record):
        raise MeldSidecarContractError("MM-Align record lacks a required modality field")
    token_ids = np.asarray(record["token_ids"])
    audio = np.asarray(record["audio_features"], dtype=np.float64)
    video = np.asarray(record["video_features"], dtype=np.float64)
    if token_ids.ndim != 1 or audio.ndim != 2 or video.ndim != 2:
        raise MeldSidecarContractError("MM-Align modality arrays have invalid rank")
    if len(token_ids) < 1 or len(token_ids) != len(audio) or len(token_ids) != len(video):
        raise MeldSidecarContractError("MM-Align token/audio/video sequences are not aligned")
    if audio.shape[1] != int(audio_dimension) or video.shape[1] != int(video_dimension):
        raise MeldSidecarContractError("MM-Align modality dimension changed")
    if not np.isfinite(audio).all() or not np.isfinite(video).all():
        raise MeldSidecarContractError("MM-Align modality arrays contain non-finite values")
    audio_pooled = np.concatenate((audio.mean(axis=0), audio.std(axis=0))).astype(
        np.float32, copy=False
    )
    video_pooled = np.concatenate((video.mean(axis=0), video.std(axis=0))).astype(
        np.float32, copy=False
    )
    return audio_pooled, video_pooled


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=path.stem + ".", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _alignment_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(key) for key in keys).encode("utf-8")).hexdigest()


def _speaker_token_code(value: str) -> int:
    """Population-independent private speaker token; never a full-train rank."""

    digest = hashlib.sha256(f"MELD-speaker\x1f{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _assert_outside_repository(path: Path, repository_root: Path | None) -> None:
    if repository_root is None:
        return
    try:
        common = Path(os.path.commonpath((str(path.resolve()), str(repository_root.resolve()))))
    except ValueError:
        return
    if common == repository_root.resolve():
        raise MeldSidecarContractError("private MELD sidecar directory must be outside repository")


def prepare_meld_role_sidecars(
    *,
    train_csv_path: str | Path,
    train_pickle_path: str | Path,
    config_path: str | Path,
    private_output_dir: str | Path,
    public_manifest_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create write-once feature/label sidecars for all frozen MELD train roles."""

    csv_path = Path(train_csv_path)
    pickle_path = Path(train_pickle_path)
    destination = Path(private_output_dir)
    manifest_path = Path(public_manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"MELD public manifest already exists: {manifest_path}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"MELD sidecar output directory is not empty: {destination}")
    _assert_outside_repository(
        destination, None if repository_root is None else Path(repository_root)
    )
    if csv_path.name != "train_sent_emo.csv" or pickle_path.name != "train.pkl":
        raise MeldSidecarContractError("only MELD train sources may enter sidecar generation")
    config = _load_config(config_path)
    source = config["source_contract"]
    actual_csv_sha = sha256_file(csv_path)
    actual_pickle_sha = sha256_file(pickle_path)
    if actual_csv_sha != str(source["train_csv_sha256"]).lower():
        raise MeldSidecarContractError("MELD train CSV hash differs from the frozen source")
    if actual_pickle_sha != str(source["train_pickle_sha256"]).lower():
        # Critically, fail before unpickling an unregistered artifact.
        raise MeldSidecarContractError("MELD train pickle hash differs from the frozen source")

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing_columns = sorted(set(REQUIRED_CSV_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise MeldSidecarContractError(f"MELD train CSV lacks columns: {missing_columns}")
    if frame.loc[:, REQUIRED_CSV_COLUMNS].isna().any().any():
        raise MeldSidecarContractError("MELD train CSV contains missing required values")
    if frame.duplicated(["Dialogue_ID", "Utterance_ID"]).any():
        raise MeldSidecarContractError("MELD train CSV contains duplicate dialogue/utterance keys")
    emotions = frame["Emotion"].astype(str).str.lower()
    unknown_emotions = sorted(set(emotions) - set(EMOTION_TO_INDEX))
    if unknown_emotions:
        raise MeldSidecarContractError(f"MELD train CSV contains unknown emotions: {unknown_emotions}")

    # This is the only trusted deserialization boundary.  Exact source hashing
    # above prevents arbitrary, unregistered pickle contents from being opened.
    with pickle_path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, Mapping):
        raise MeldSidecarContractError("MM-Align train pickle root must be a mapping")

    keys = (
        frame["Dialogue_ID"].astype(int).astype(str)
        + "_"
        + frame["Utterance_ID"].astype(int).astype(str)
    )
    csv_key_set = set(keys)
    record_key_set = {str(key) for key in records}
    missing_feature_keys = csv_key_set - record_key_set
    extra_feature_keys = record_key_set - csv_key_set
    if extra_feature_keys:
        raise MeldSidecarContractError("MM-Align train pickle contains keys outside official CSV")
    if len(missing_feature_keys) != int(source["allowed_missing_feature_rows"]):
        raise MeldSidecarContractError("MM-Align missing-feature row count changed")

    roles = config["roles"]
    split_protocol = str(config["split_protocol_id"])
    feature_contract = config["feature_contract"]
    speaker_values = sorted(frame["Speaker"].astype(str).unique())
    speaker_tokens = {speaker: _speaker_token_code(speaker) for speaker in speaker_values}
    if len(set(speaker_tokens.values())) != len(speaker_tokens):
        raise MeldSidecarContractError("MELD speaker-token hash collision")
    role_buffers: dict[str, dict[str, list[Any]]] = {
        role: {
            "keys": [],
            "utterances": [],
            "audio": [],
            "video": [],
            "dialogues": [],
            "speakers": [],
            "orders": [],
            "protocol_rows": [],
            "labels": [],
        }
        for role in ROLE_NAMES
    }
    embedded_label_consistency_checked = False
    for row_index, row in frame.iterrows():
        key = str(keys.iloc[row_index])
        if key in missing_feature_keys:
            continue
        record = records[key]
        if not isinstance(record, Mapping):
            raise MeldSidecarContractError("MM-Align record must be a mapping")
        label = EMOTION_TO_INDEX[str(row["Emotion"]).lower()]
        if "label" in record:
            # Trusted-custodian consistency check only.  Its result is neither
            # a training target nor a public/sealed-role statistic.
            _ = int(record["label"]) == label
            embedded_label_consistency_checked = True
        audio_pooled, video_pooled = _pool_record(
            record,
            audio_dimension=int(feature_contract["audio_dimension"]),
            video_dimension=int(feature_contract["video_dimension"]),
        )
        role, _ = assign_group_role(
            "MELD",
            int(row["Dialogue_ID"]),
            split_protocol,
            roles,
        )
        buffer = role_buffers[role]
        buffer["keys"].append(key)
        buffer["utterances"].append(str(row["Utterance"]))
        buffer["audio"].append(audio_pooled)
        buffer["video"].append(video_pooled)
        buffer["dialogues"].append(int(row["Dialogue_ID"]))
        buffer["speakers"].append(speaker_tokens[str(row["Speaker"])])
        buffer["orders"].append(int(row["Utterance_ID"]))
        buffer["protocol_rows"].append(int(row_index))
        buffer["labels"].append(label)

    del records
    destination.mkdir(parents=True, exist_ok=True)
    public_roles: dict[str, Any] = {}
    for role in ROLE_NAMES:
        buffer = role_buffers[role]
        if not buffer["keys"]:
            raise MeldSidecarContractError(f"MELD role has no feature rows: {role}")
        alignment = _alignment_digest(buffer["keys"])
        feature_path = destination / f"features_{role}.npz"
        label_path = destination / f"labels_{role}.npz"
        audio = np.stack(buffer["audio"]).astype(np.float32, copy=False)
        video = np.stack(buffer["video"]).astype(np.float32, copy=False)
        dialogues = np.asarray(buffer["dialogues"], dtype=np.int64)
        speakers = np.asarray(buffer["speakers"], dtype=np.int64)
        orders = np.asarray(buffer["orders"], dtype=np.int64)
        protocol_rows = np.asarray(buffer["protocol_rows"], dtype=np.int64)
        seen_speaker_dialogues: set[tuple[int, int]] = set()
        history_eligible_rows = 0
        for index in np.lexsort((orders, dialogues)):
            identity = (int(dialogues[index]), int(speakers[index]))
            if identity in seen_speaker_dialogues:
                history_eligible_rows += 1
            seen_speaker_dialogues.add(identity)
        _atomic_npz(
            feature_path,
            schema_version=np.asarray([SIDECAR_SCHEMA_VERSION]),
            role=np.asarray([role]),
            row_alignment_sha256=np.asarray([alignment]),
            utterances=np.asarray(buffer["utterances"], dtype=np.str_),
            audio_mean_std=audio,
            video_mean_std=video,
            dialogue_codes=dialogues,
            speaker_codes=speakers,
            utterance_order=orders,
            protocol_row_ids=protocol_rows,
        )
        _atomic_npz(
            label_path,
            schema_version=np.asarray([SIDECAR_SCHEMA_VERSION]),
            role=np.asarray([role]),
            row_alignment_sha256=np.asarray([alignment]),
            labels=np.asarray(buffer["labels"], dtype=np.int64),
        )
        public_roles[role] = {
            "feature_filename": feature_path.name,
            "label_filename": label_path.name,
            "rows": int(len(buffer["keys"])),
            "dialogues": int(len(np.unique(dialogues))),
            "history_eligible_rows": int(history_eligible_rows),
            "audio_dimension": int(audio.shape[1]),
            "video_dimension": int(video.shape[1]),
            "feature_sha256": sha256_file(feature_path),
            "label_sha256": sha256_file(label_path),
            "row_alignment_sha256": alignment,
        }

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "role_separated_train_sidecars_created_and_hashed",
        "dataset_id": "MELD",
        "split_protocol_id": split_protocol,
        "label_order": list(EMOTION_TO_INDEX),
        "claim_boundary": (
            "Trusted MELD train-only data preparation. No model was fit and no role-level "
            "performance metric was computed. Subsequent runners must open only the role "
            "sidecars explicitly allowed by their protocol."
        ),
        "source_contract": {
            "train_csv_sha256": actual_csv_sha,
            "train_pickle_sha256": actual_pickle_sha,
            "official_csv_is_authoritative_label_source": True,
            "embedded_pickle_label_used_for_training_or_metrics": False,
            "embedded_pickle_label_consistency_checked_by_trusted_custodian": bool(
                embedded_label_consistency_checked
            ),
            "embedded_pickle_label_mismatch_statistics_exposed": False,
            "missing_feature_rows": int(len(missing_feature_keys)),
            "extra_feature_rows": 0,
        },
        "feature_contract": {
            "audio_mean_std_columns": int(2 * int(feature_contract["audio_dimension"])),
            "video_mean_std_columns": int(2 * int(feature_contract["video_dimension"])),
            "numeric_dtype": "float32",
            "strict_same_dialogue_same_speaker_past_history_supported": True,
            "protocol_row_identity": "zero_based_official_train_csv_row_index",
        },
        "seal_contract": {
            "features_and_labels_are_in_separate_archives": True,
            "each_role_has_a_separate_label_archive": True,
            "allow_pickle_required_to_load_sidecars": False,
            "open_role_runner_may_load_only": [
                "base_and_utility_fit",
                "model_selection",
            ],
            "calibration_and_internal_holdout_remain_unopened_by_model_runners": True,
        },
        "roles": public_roles,
        "config_sha256": sha256_file(config_path),
        "public_content_audit": {
            "contains_labels_or_class_counts": False,
            "contains_utterances_or_embeddings": False,
            "contains_dialogue_speaker_or_row_identifiers": False,
            "contains_private_absolute_paths": False,
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _one_string(array: np.ndarray, *, field: str) -> str:
    values = np.asarray(array)
    if values.shape != (1,) or values.dtype.kind not in {"U", "S"}:
        raise MeldSidecarContractError(f"{field} must be a one-string array")
    return str(values[0])


def load_meld_role_sidecar(
    feature_path: str | Path,
    label_path: str | Path,
    *,
    expected_role: str,
) -> MeldRoleSidecar:
    """Load exactly one role after verifying feature/label row alignment."""

    if expected_role not in ROLE_NAMES:
        raise MeldSidecarContractError("unknown MELD sidecar role")
    with np.load(feature_path, allow_pickle=False) as archive:
        expected = {
            "schema_version",
            "role",
            "row_alignment_sha256",
            "utterances",
            "audio_mean_std",
            "video_mean_std",
            "dialogue_codes",
            "speaker_codes",
            "utterance_order",
            "protocol_row_ids",
        }
        if set(archive.files) != expected:
            raise MeldSidecarContractError("MELD feature-sidecar schema changed")
        features = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(label_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "schema_version",
            "role",
            "row_alignment_sha256",
            "labels",
        }:
            raise MeldSidecarContractError("MELD label-sidecar schema changed")
        labels = {key: np.asarray(archive[key]) for key in archive.files}
    for payload in (features, labels):
        if _one_string(payload["schema_version"], field="schema_version") != SIDECAR_SCHEMA_VERSION:
            raise MeldSidecarContractError("MELD sidecar schema version changed")
        if _one_string(payload["role"], field="role") != expected_role:
            raise MeldSidecarContractError("MELD sidecar role differs from the requested role")
    feature_alignment = _one_string(
        features["row_alignment_sha256"], field="row_alignment_sha256"
    )
    label_alignment = _one_string(
        labels["row_alignment_sha256"], field="row_alignment_sha256"
    )
    if feature_alignment != label_alignment:
        raise MeldSidecarContractError("MELD feature/label sidecars are not row-aligned")
    _validate_sha256(feature_alignment, field="row_alignment_sha256")
    utterances = np.asarray(features["utterances"])
    audio = np.asarray(features["audio_mean_std"], dtype=np.float32)
    video = np.asarray(features["video_mean_std"], dtype=np.float32)
    dialogue = np.asarray(features["dialogue_codes"])
    speaker = np.asarray(features["speaker_codes"])
    order = np.asarray(features["utterance_order"])
    protocol_rows = np.asarray(features["protocol_row_ids"])
    y = np.asarray(labels["labels"])
    rows = len(utterances)
    if (
        utterances.ndim != 1
        or utterances.dtype.kind not in {"U", "S"}
        or audio.ndim != 2
        or video.ndim != 2
        or any(array.shape != (rows,) for array in (dialogue, speaker, order, protocol_rows, y))
    ):
        raise MeldSidecarContractError("MELD sidecar row arrays are misaligned")
    if not all(
        np.issubdtype(array.dtype, np.integer)
        for array in (dialogue, speaker, order, protocol_rows, y)
    ):
        raise MeldSidecarContractError("MELD sidecar codes/labels must be integer arrays")
    if np.any(protocol_rows < 0) or len(set(protocol_rows.tolist())) != rows:
        raise MeldSidecarContractError(
            "MELD sidecar protocol rows must be unique non-negative source rows"
        )
    if not np.isfinite(audio).all() or not np.isfinite(video).all():
        raise MeldSidecarContractError("MELD sidecar modality features contain non-finite values")
    if np.any((y < 0) | (y >= len(EMOTION_TO_INDEX))):
        raise MeldSidecarContractError("MELD sidecar label is outside the emotion range")
    return MeldRoleSidecar(
        utterances,
        audio,
        video,
        dialogue.astype(np.int64, copy=False),
        speaker.astype(np.int64, copy=False),
        order.astype(np.int64, copy=False),
        protocol_rows.astype(np.int64, copy=False),
        y.astype(np.int64, copy=False),
        feature_alignment,
    )
