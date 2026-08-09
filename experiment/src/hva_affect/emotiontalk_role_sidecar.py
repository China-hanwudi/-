"""Trusted creation and strict loading of EmotionTalk train-role sidecars.

The upstream release stores all train labels in one pickle and media/text in
artifacts that also contain non-open rows.  This module is the sole trusted
preparation boundary: after exact source-hash verification it writes four
physical ``allow_pickle=False`` archives (fit features, fit labels,
model-selection features, model-selection labels).  Model runners consume
only those four files and the aggregate manifest; they never open the upstream
media NPZ, transcription CSV, or label pickle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_contract import ContractError, sha256_file, write_json_atomic
from .emotiontalk_contract import parse_key
from .emotiontalk_text_p1 import LABEL_NAMES
from .scu_set import assign_group_role


PROTOCOL = "emotiontalk_role_separated_sidecars_v2"
FEATURE_SCHEMA = "emotiontalk_role_feature_sidecar_v2"
LABEL_SCHEMA = "emotiontalk_role_label_sidecar_v2"
MANIFEST_SCHEMA = "emotiontalk_role_sidecar_manifest_v2"
FIT_ROLE = "base_and_utility_fit"
SELECTION_ROLE = "model_selection"
OPEN_ROLES = (FIT_ROLE, SELECTION_ROLE)
FROZEN_ROLE_RANGES = {
    FIT_ROLE: [0, 64],
    SELECTION_ROLE: [65, 79],
    "calibration": [80, 89],
    "internal_holdout_sealed": [90, 99],
}
SOURCE_FEATURE_FIELDS = {
    "keys",
    "splits",
    "audio_features",
    "video_features",
    "quality",
    "quality_names",
    "config_sha256",
}
FEATURE_FIELDS = {
    "schema_version",
    "dataset_id",
    "role",
    "split_protocol_id",
    "row_alignment_sha256",
    "opaque_row_hashes",
    "opaque_group_hashes",
    "speaker_tokens",
    "turn_ids",
    "protocol_row_ids",
    "role_buckets",
    "texts",
    "audio_features",
    "video_features",
    "source_feature_config_sha256",
}
LABEL_FIELDS = {
    "schema_version",
    "dataset_id",
    "role",
    "split_protocol_id",
    "row_alignment_sha256",
    "labels",
    "source_label_sha256",
}
MANIFEST_ROLE_FIELDS = {
    "feature_filename",
    "label_filename",
    "rows",
    "groups",
    "history_eligible_rows",
    "audio_dimension",
    "video_dimension",
    "feature_sha256",
    "label_sha256",
    "row_alignment_sha256",
}


@dataclass(frozen=True)
class EmotionTalkRoleArrays:
    role: str
    row_hashes: np.ndarray
    group_hashes: np.ndarray
    speaker_tokens: np.ndarray
    turn_ids: np.ndarray
    protocol_row_ids: np.ndarray
    role_buckets: np.ndarray
    texts: np.ndarray
    audio: np.ndarray
    video: np.ndarray
    labels: np.ndarray
    row_alignment_sha256: str
    feature_sha256: str
    label_sha256: str


def _valid_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContractError("EmotionTalk sidecar config root must be a mapping")
    if value.get("protocol") != PROTOCOL:
        raise ContractError("EmotionTalk sidecar protocol changed")
    if value.get("status") != "frozen_before_trusted_generation":
        raise ContractError("EmotionTalk sidecar config is not frozen")
    if value.get("dataset_id") != "EmotionTalk":
        raise ContractError("EmotionTalk sidecar dataset identity changed")
    if value.get("split_protocol_id") != "scu_set_exploration_v1":
        raise ContractError("EmotionTalk sidecar split protocol changed")
    if value.get("roles") != FROZEN_ROLE_RANGES:
        raise ContractError("EmotionTalk sidecar role ranges changed")
    hashes = value.get("source_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "label_archive",
        "media_features",
        "transcription",
    }:
        raise ContractError("EmotionTalk sidecar source hash schema changed")
    for name, digest in hashes.items():
        _valid_sha256(digest, field=f"source_sha256.{name}")
    return dict(value)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _alignment_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(key) for key in keys).encode("utf-8")).hexdigest()


def _opaque_hash(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\x1f{value}".encode("utf-8")).hexdigest()


def _assert_outside_repository(path: Path, repository_root: Path | None) -> None:
    if repository_root is None:
        return
    try:
        common = Path(os.path.commonpath((str(path.resolve()), str(repository_root.resolve()))))
    except ValueError:
        return
    if common == repository_root.resolve():
        raise ContractError("private role-sidecar directory must be outside the public repository")


def _transcript_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"name", "chinese"}.issubset(reader.fieldnames or ()):
            raise ContractError("EmotionTalk transcription CSV lacks name/chinese")
        result = {
            Path(str(row["name"]).replace("\\", "/")).stem: str(row["chinese"])
            for row in reader
        }
    if not result:
        raise ContractError("EmotionTalk transcription CSV is empty")
    return result


def prepare_emotiontalk_role_sidecars(
    *,
    label_archive_path: Path,
    feature_path: Path,
    transcription_path: Path,
    config_path: Path,
    private_output_dir: Path,
    public_manifest_path: Path,
    repository_root: Path | None = None,
) -> dict:
    """Create write-once physical feature/label sidecars for roles 0--79."""

    if public_manifest_path.exists():
        raise FileExistsError(f"public manifest already exists: {public_manifest_path}")
    if private_output_dir.exists() and any(private_output_dir.iterdir()):
        raise FileExistsError(f"private sidecar directory is not empty: {private_output_dir}")
    _assert_outside_repository(private_output_dir, repository_root)
    config = _read_config(config_path)
    frozen_hashes = {
        name: _valid_sha256(value, field=name)
        for name, value in config["source_sha256"].items()
    }
    observed_hashes = {
        "label_archive": sha256_file(label_archive_path),
        "media_features": sha256_file(feature_path),
        "transcription": sha256_file(transcription_path),
    }
    if observed_hashes != frozen_hashes:
        changed = sorted(name for name in frozen_hashes if frozen_hashes[name] != observed_hashes[name])
        raise ContractError(f"EmotionTalk trusted source hash changed: {changed}")

    # Trusted source boundary.  Model code never receives these paths.
    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != SOURCE_FEATURE_FIELDS:
            raise ContractError("EmotionTalk source feature schema changed")
        keys = archive["keys"].astype(str)
        splits = archive["splits"].astype(str)
        if np.any(splits == "test_corpus"):
            raise ContractError("sealed test feature row entered source feature artifact")
        audio = archive["audio_features"].astype(np.float32, copy=True)
        video = archive["video_features"].astype(np.float32, copy=True)
        source_feature_config_sha = str(np.asarray(archive["config_sha256"]).reshape(()))
    if not (len(keys) == len(splits) == len(audio) == len(video)):
        raise ContractError("EmotionTalk source feature arrays are misaligned")
    transcript = _transcript_map(transcription_path)
    train_source = np.flatnonzero(splits == "train_corpus")
    train_keys = keys[train_source]
    if len(set(train_keys)) != len(train_keys):
        raise ContractError("duplicate EmotionTalk train key")
    if any(str(key) not in transcript for key in train_keys):
        raise ContractError("EmotionTalk train feature key lacks transcription")

    roles: list[str] = []
    buckets: list[int] = []
    groups: list[str] = []
    speakers: list[str] = []
    turns: list[int] = []
    for key in train_keys:
        group, dialogue, speaker, turn = parse_key(str(key))
        group_id = f"{group}/{dialogue}"
        role, bucket = assign_group_role(
            "EmotionTalk", group_id, str(config["split_protocol_id"]), config["roles"]
        )
        roles.append(role)
        buckets.append(int(bucket))
        groups.append(group_id)
        speakers.append(str(speaker))
        turns.append(int(turn))

    # Hash check occurs before the sole trusted unpickle.  Only open values are
    # indexed; validation/test payloads are never requested.
    with np.load(label_archive_path, allow_pickle=True) as archive:
        if "train_corpus" not in archive.files:
            raise ContractError("EmotionTalk label archive lacks train_corpus")
        payload = archive["train_corpus"]
        if payload.shape != () or payload.dtype != object:
            raise ContractError("EmotionTalk train label payload is malformed")
        train_labels = payload.item()
    if not isinstance(train_labels, dict) or set(train_labels) != set(train_keys):
        raise ContractError("EmotionTalk train label/media alignment changed")

    private_output_dir.mkdir(parents=True, exist_ok=True)
    public_roles: dict[str, dict[str, object]] = {}
    roles_array = np.asarray(roles)
    buckets_array = np.asarray(buckets, dtype=np.int16)
    groups_array = np.asarray(groups)
    speakers_array = np.asarray(speakers)
    turns_array = np.asarray(turns, dtype=np.int64)
    for role in OPEN_ROLES:
        mask = roles_array == role
        role_keys = train_keys[mask]
        if not mask.any():
            raise ContractError(f"EmotionTalk open role has no rows: {role}")
        source_rows = train_source[mask]
        labels = np.asarray([int(train_labels[str(key)]["emo"]) for key in role_keys], dtype=np.int64)
        if np.any((labels < 0) | (labels >= len(LABEL_NAMES))):
            raise ContractError("EmotionTalk open-role label is invalid")
        alignment = _alignment_sha256(role_keys)
        feature_name = f"features_{role}.npz"
        label_name = f"labels_{role}.npz"
        feature_output = private_output_dir / feature_name
        label_output = private_output_dir / label_name
        role_groups = groups_array[mask]
        role_speakers = speakers_array[mask]
        role_turns = turns_array[mask]
        _atomic_savez(
            feature_output,
            schema_version=np.asarray(FEATURE_SCHEMA),
            dataset_id=np.asarray("EmotionTalk"),
            role=np.asarray(role),
            split_protocol_id=np.asarray(str(config["split_protocol_id"])),
            row_alignment_sha256=np.asarray(alignment),
            opaque_row_hashes=np.asarray([_opaque_hash("row", str(key)) for key in role_keys]),
            opaque_group_hashes=np.asarray([_opaque_hash("group", str(value)) for value in role_groups]),
            speaker_tokens=role_speakers,
            turn_ids=role_turns,
            protocol_row_ids=np.asarray(source_rows, dtype=np.int64),
            role_buckets=buckets_array[mask],
            texts=np.asarray([transcript[str(key)] for key in role_keys]),
            audio_features=audio[source_rows],
            video_features=video[source_rows],
            source_feature_config_sha256=np.asarray(source_feature_config_sha),
        )
        _atomic_savez(
            label_output,
            schema_version=np.asarray(LABEL_SCHEMA),
            dataset_id=np.asarray("EmotionTalk"),
            role=np.asarray(role),
            split_protocol_id=np.asarray(str(config["split_protocol_id"])),
            row_alignment_sha256=np.asarray(alignment),
            labels=labels,
            source_label_sha256=np.asarray(observed_hashes["label_archive"]),
        )
        history_eligible = 0
        for query in range(len(role_keys)):
            history_eligible += int(
                np.any(
                    (role_groups == role_groups[query])
                    & (role_speakers == role_speakers[query])
                    & (role_turns < role_turns[query])
                )
            )
        public_roles[role] = {
            "feature_filename": feature_name,
            "label_filename": label_name,
            "rows": int(mask.sum()),
            "groups": int(len(set(role_groups))),
            "history_eligible_rows": int(history_eligible),
            "audio_dimension": int(audio.shape[1]),
            "video_dimension": int(video.shape[1]),
            "feature_sha256": sha256_file(feature_output),
            "label_sha256": sha256_file(label_output),
            "row_alignment_sha256": alignment,
        }
    del train_labels
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol": PROTOCOL,
        "status": "strict_open_role_feature_and_label_sidecars_created_and_hashed",
        "dataset_id": "EmotionTalk",
        "split_protocol_id": str(config["split_protocol_id"]),
        "label_order": list(LABEL_NAMES),
        "source_contract": {
            **observed_hashes,
            "feature_config_sha256": source_feature_config_sha,
            "trusted_source_boundary_only": True,
            "validation_or_test_label_payload_opened": False,
        },
        "seal_contract": {
            "model_runner_opens_upstream_media_npz_or_transcription": False,
            "open_role_runner_may_load_only": list(OPEN_ROLES),
            "calibration_holdout_validation_test_sidecars_created": False,
            "allow_pickle_required_to_load_sidecars": False,
        },
        "roles": public_roles,
        "config_sha256": sha256_file(config_path),
        "public_content_audit": {
            "contains_labels_or_class_counts": False,
            "contains_row_group_or_speaker_identifiers": False,
            "contains_transcripts_or_embeddings": False,
            "contains_private_absolute_paths": False,
        },
    }
    write_json_atomic(manifest, public_manifest_path.resolve())
    return manifest


def _scalar_text(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ContractError(f"EmotionTalk sidecar {name} must be scalar")
    return str(value.reshape(-1)[0])


def load_emotiontalk_role_sidecars(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
) -> tuple[dict[str, EmotionTalkRoleArrays], dict]:
    """Verify the public manifest and load exactly four open-role files."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_root = {
        "schema_version",
        "protocol",
        "status",
        "dataset_id",
        "split_protocol_id",
        "label_order",
        "source_contract",
        "seal_contract",
        "roles",
        "config_sha256",
        "public_content_audit",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_root:
        raise ContractError("EmotionTalk sidecar manifest schema changed")
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["protocol"] != PROTOCOL:
        raise ContractError("EmotionTalk sidecar manifest protocol changed")
    if manifest["status"] != "strict_open_role_feature_and_label_sidecars_created_and_hashed":
        raise ContractError("EmotionTalk sidecar manifest is not a completed strict artifact")
    if manifest["dataset_id"] != "EmotionTalk" or manifest["split_protocol_id"] != "scu_set_exploration_v1":
        raise ContractError("EmotionTalk sidecar manifest dataset/split changed")
    if tuple(manifest["label_order"]) != tuple(LABEL_NAMES):
        raise ContractError("EmotionTalk label order changed")
    if set(manifest["roles"]) != set(OPEN_ROLES):
        raise ContractError("EmotionTalk manifest exposes a non-open or missing role")
    source = manifest["source_contract"]
    expected_source_fields = {
        "label_archive",
        "media_features",
        "transcription",
        "feature_config_sha256",
        "trusted_source_boundary_only",
        "validation_or_test_label_payload_opened",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source_fields:
        raise ContractError("EmotionTalk manifest source contract changed")
    for field in ("label_archive", "media_features", "transcription", "feature_config_sha256"):
        _valid_sha256(source[field], field=f"source_contract.{field}")
    if source["trusted_source_boundary_only"] is not True:
        raise ContractError("EmotionTalk trusted source boundary changed")
    if source["validation_or_test_label_payload_opened"] is not False:
        raise ContractError("EmotionTalk manifest attests validation/test label access")
    seal = manifest["seal_contract"]
    expected_seal = {
        "model_runner_opens_upstream_media_npz_or_transcription",
        "open_role_runner_may_load_only",
        "calibration_holdout_validation_test_sidecars_created",
        "allow_pickle_required_to_load_sidecars",
    }
    if not isinstance(seal, Mapping) or set(seal) != expected_seal:
        raise ContractError("EmotionTalk manifest seal contract changed")
    if (
        seal["model_runner_opens_upstream_media_npz_or_transcription"] is not False
        or seal["open_role_runner_may_load_only"] != list(OPEN_ROLES)
        or seal["calibration_holdout_validation_test_sidecars_created"] is not False
        or seal["allow_pickle_required_to_load_sidecars"] is not False
    ):
        raise ContractError("EmotionTalk manifest does not enforce strict physical separation")
    _valid_sha256(manifest["config_sha256"], field="config_sha256")
    expected_public_audit = {
        "contains_labels_or_class_counts": False,
        "contains_row_group_or_speaker_identifiers": False,
        "contains_transcripts_or_embeddings": False,
        "contains_private_absolute_paths": False,
    }
    if manifest["public_content_audit"] != expected_public_audit:
        raise ContractError("EmotionTalk public-content audit changed")

    result: dict[str, EmotionTalkRoleArrays] = {}
    for role in OPEN_ROLES:
        record = manifest["roles"][role]
        if not isinstance(record, Mapping) or set(record) != MANIFEST_ROLE_FIELDS:
            raise ContractError("EmotionTalk manifest role record is malformed")
        expected_names = {
            "feature_filename": f"features_{role}.npz",
            "label_filename": f"labels_{role}.npz",
        }
        for field, expected in expected_names.items():
            if record.get(field) != expected:
                raise ContractError("EmotionTalk manifest sidecar filename changed")
        feature_path = sidecar_dir / expected_names["feature_filename"]
        label_path = sidecar_dir / expected_names["label_filename"]
        feature_sha = sha256_file(feature_path)
        label_sha = sha256_file(label_path)
        if feature_sha != _valid_sha256(record.get("feature_sha256"), field="feature_sha256"):
            raise ContractError("EmotionTalk feature sidecar hash differs from manifest")
        if label_sha != _valid_sha256(record.get("label_sha256"), field="label_sha256"):
            raise ContractError("EmotionTalk label sidecar hash differs from manifest")
        with np.load(feature_path, allow_pickle=False) as feature:
            if set(feature.files) != FEATURE_FIELDS:
                raise ContractError("EmotionTalk feature sidecar schema changed")
            feature_payload = {name: np.asarray(feature[name]) for name in feature.files}
        with np.load(label_path, allow_pickle=False) as label:
            if set(label.files) != LABEL_FIELDS:
                raise ContractError("EmotionTalk label sidecar schema changed")
            label_payload = {name: np.asarray(label[name]) for name in label.files}
        for payload, schema in ((feature_payload, FEATURE_SCHEMA), (label_payload, LABEL_SCHEMA)):
            if _scalar_text(payload, "schema_version") != schema:
                raise ContractError("EmotionTalk role sidecar version changed")
            if _scalar_text(payload, "dataset_id") != "EmotionTalk":
                raise ContractError("EmotionTalk role sidecar dataset changed")
            if _scalar_text(payload, "role") != role:
                raise ContractError("EmotionTalk role sidecar role changed")
            if _scalar_text(payload, "split_protocol_id") != "scu_set_exploration_v1":
                raise ContractError("EmotionTalk role sidecar split changed")
        alignment = _scalar_text(feature_payload, "row_alignment_sha256")
        if alignment != _scalar_text(label_payload, "row_alignment_sha256"):
            raise ContractError("EmotionTalk feature/label alignment differs")
        if alignment != _valid_sha256(record.get("row_alignment_sha256"), field="row_alignment"):
            raise ContractError("EmotionTalk row alignment differs from manifest")
        if _scalar_text(feature_payload, "source_feature_config_sha256") != str(
            source["feature_config_sha256"]
        ):
            raise ContractError("EmotionTalk feature sidecar source config differs from manifest")
        if _scalar_text(label_payload, "source_label_sha256") != str(source["label_archive"]):
            raise ContractError("EmotionTalk label sidecar source hash differs from manifest")
        rows = int(record.get("rows", -1))
        row_hashes = feature_payload["opaque_row_hashes"].astype(str)
        arrays_1d = (
            row_hashes,
            feature_payload["opaque_group_hashes"],
            feature_payload["speaker_tokens"],
            feature_payload["turn_ids"],
            feature_payload["protocol_row_ids"],
            feature_payload["role_buckets"],
            feature_payload["texts"],
            label_payload["labels"],
        )
        audio = feature_payload["audio_features"].astype(np.float32, copy=True)
        video = feature_payload["video_features"].astype(np.float32, copy=True)
        if rows <= 0 or any(np.asarray(value).shape != (rows,) for value in arrays_1d):
            raise ContractError("EmotionTalk sidecar rows differ from manifest")
        if audio.shape != (rows, int(record.get("audio_dimension", -1))) or video.shape != (
            rows,
            int(record.get("video_dimension", -1)),
        ):
            raise ContractError("EmotionTalk sidecar modality dimensions differ from manifest")
        if len(set(row_hashes)) != rows or not np.isfinite(audio).all() or not np.isfinite(video).all():
            raise ContractError("EmotionTalk feature sidecar contains duplicate/non-finite rows")
        buckets = feature_payload["role_buckets"].astype(np.int16, copy=True)
        bounds = FROZEN_ROLE_RANGES[role]
        if np.any((buckets < bounds[0]) | (buckets > bounds[1])):
            raise ContractError("EmotionTalk role sidecar bucket is outside its open range")
        labels = label_payload["labels"].astype(np.int64, copy=True)
        if np.any((labels < 0) | (labels >= len(LABEL_NAMES))):
            raise ContractError("EmotionTalk role label is invalid")
        groups = feature_payload["opaque_group_hashes"].astype(str)
        speakers = feature_payload["speaker_tokens"].astype(str)
        turns = feature_payload["turn_ids"].astype(np.int64, copy=True)
        if int(record["groups"]) != len(set(groups.tolist())):
            raise ContractError("EmotionTalk manifest group count differs from sidecar")
        history_eligible = int(
            sum(
                np.any(
                    (groups == groups[index])
                    & (speakers == speakers[index])
                    & (turns < turns[index])
                )
                for index in range(rows)
            )
        )
        if history_eligible != int(record["history_eligible_rows"]):
            raise ContractError("EmotionTalk history-eligible count differs from manifest")
        result[role] = EmotionTalkRoleArrays(
            role=role,
            row_hashes=row_hashes,
            group_hashes=groups,
            speaker_tokens=speakers,
            turn_ids=turns,
            protocol_row_ids=feature_payload["protocol_row_ids"].astype(np.int64, copy=True),
            role_buckets=buckets,
            texts=feature_payload["texts"].astype(str),
            audio=audio,
            video=video,
            labels=labels,
            row_alignment_sha256=alignment,
            feature_sha256=feature_sha,
            label_sha256=label_sha,
        )
    return result, manifest
