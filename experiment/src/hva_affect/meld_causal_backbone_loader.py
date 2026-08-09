"""Manifest-verified MELD train-role loader and causal-backbone wrapper.

The model-facing boundary accepts a sidecar directory and one aggregate
manifest.  It resolves and opens exactly four ``allow_pickle=False`` files:
fit features/labels and model-selection features/labels.  Calibration,
internal-holdout, dev, and test arrays have no API path into this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .causal_multimodal_backbone import CausalBackboneConfig, CausalMultimodalBackbone
from .data_contract import ContractError, sha256_file
from .emotiontalk_causal_backbone_runner import (
    EXPECTED_SEEDS,
    FIT_ROLE,
    FROZEN_ROLE_RANGES,
    SELECTION_ROLE,
    BackboneRunConfig,
    OpenRoleCorpus,
    UtilitySamplingConfig,
    VerifiedCorpusProvenance,
    _canonical_sha256,
    _corpus_contract_sha256,
    _role_assignment_sha256,
    create_verified_corpus_provenance,
    execute_crossfit_backbone,
    fit_fold_text_processor,
    make_crossfit_splits,
    predict_utility_contexts,
    sample_corpus_bidirectional_tasks,
    validate_open_role_backbone_payload,
)
from .meld_multimodal_sidecar import (
    EMOTION_TO_INDEX,
    MANIFEST_SCHEMA_VERSION,
    PROTOCOL,
    ROLE_NAMES,
    SIDECAR_SCHEMA_VERSION,
)
from .scu_set import assign_group_role


OPEN_ROLES = (FIT_ROLE, SELECTION_ROLE)
FEATURE_FIELDS = {
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
LABEL_FIELDS = {
    "schema_version",
    "role",
    "row_alignment_sha256",
    "labels",
}
MANIFEST_FIELDS = {
    "schema_version",
    "protocol",
    "status",
    "dataset_id",
    "split_protocol_id",
    "label_order",
    "claim_boundary",
    "source_contract",
    "feature_contract",
    "seal_contract",
    "roles",
    "config_sha256",
    "public_content_audit",
}
ROLE_RECORD_FIELDS = {
    "feature_filename",
    "label_filename",
    "rows",
    "dialogues",
    "history_eligible_rows",
    "audio_dimension",
    "video_dimension",
    "feature_sha256",
    "label_sha256",
    "row_alignment_sha256",
}
PREFLIGHT_SCHEMA = "meld_causal_backbone_nonperformance_preflight_v1"


@dataclass(frozen=True)
class _RoleArrays:
    role: str
    utterances: np.ndarray
    audio: np.ndarray
    video: np.ndarray
    dialogues: np.ndarray
    speakers: np.ndarray
    orders: np.ndarray
    protocol_rows: np.ndarray
    labels: np.ndarray
    alignment_sha256: str
    feature_sha256: str
    label_sha256: str


def _valid_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _scalar_text(payload: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ContractError(f"MELD sidecar {name} must contain one scalar string")
    return str(value.reshape(-1)[0])


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read MELD sidecar manifest: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ContractError("MELD sidecar manifest schema changed")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ContractError("MELD sidecar manifest version changed")
    if manifest["protocol"] != PROTOCOL:
        raise ContractError("MELD sidecar manifest protocol changed")
    if manifest["status"] != "role_separated_train_sidecars_created_and_hashed":
        raise ContractError("MELD sidecar manifest is not a completed artifact")
    if manifest["dataset_id"] != "MELD":
        raise ContractError("MELD sidecar manifest dataset changed")
    if manifest["split_protocol_id"] != "scu_set_exploration_v1":
        raise ContractError("MELD sidecar manifest split protocol changed")
    if tuple(manifest["label_order"]) != tuple(EMOTION_TO_INDEX):
        raise ContractError("MELD sidecar manifest label order changed")
    if not isinstance(manifest["claim_boundary"], str) or not manifest["claim_boundary"]:
        raise ContractError("MELD sidecar manifest claim boundary is empty")

    source = manifest["source_contract"]
    expected_source = {
        "train_csv_sha256",
        "train_pickle_sha256",
        "official_csv_is_authoritative_label_source",
        "embedded_pickle_label_used_for_training_or_metrics",
        "embedded_pickle_label_consistency_checked_by_trusted_custodian",
        "embedded_pickle_label_mismatch_statistics_exposed",
        "missing_feature_rows",
        "extra_feature_rows",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source:
        raise ContractError("MELD manifest source contract changed")
    _valid_sha256(source["train_csv_sha256"], field="train_csv_sha256")
    _valid_sha256(source["train_pickle_sha256"], field="train_pickle_sha256")
    if source["official_csv_is_authoritative_label_source"] is not True:
        raise ContractError("MELD official label-source contract changed")
    if source["embedded_pickle_label_used_for_training_or_metrics"] is not False:
        raise ContractError("MELD embedded pickle label entered the target contract")
    if source["embedded_pickle_label_consistency_checked_by_trusted_custodian"] is not True:
        raise ContractError("MELD trusted-custodian label consistency audit was not completed")
    if source["embedded_pickle_label_mismatch_statistics_exposed"] is not False:
        raise ContractError("MELD manifest exposes sealed-role label mismatch statistics")
    for name in ("missing_feature_rows", "extra_feature_rows"):
        if not isinstance(source[name], int) or int(source[name]) < 0:
            raise ContractError(f"MELD source count is invalid: {name}")
    if int(source["extra_feature_rows"]) != 0:
        raise ContractError("MELD manifest attests extra feature rows")

    feature = manifest["feature_contract"]
    expected_feature = {
        "audio_mean_std_columns",
        "video_mean_std_columns",
        "numeric_dtype",
        "strict_same_dialogue_same_speaker_past_history_supported",
        "protocol_row_identity",
    }
    if not isinstance(feature, Mapping) or set(feature) != expected_feature:
        raise ContractError("MELD manifest feature contract changed")
    if int(feature["audio_mean_std_columns"]) < 1 or int(feature["video_mean_std_columns"]) < 1:
        raise ContractError("MELD manifest modality dimension is invalid")
    if feature["numeric_dtype"] != "float32":
        raise ContractError("MELD manifest numeric dtype changed")
    if feature["strict_same_dialogue_same_speaker_past_history_supported"] is not True:
        raise ContractError("MELD strict-history contract changed")
    if feature["protocol_row_identity"] != "zero_based_official_train_csv_row_index":
        raise ContractError("MELD protocol-row identity contract changed")

    seal = manifest["seal_contract"]
    expected_seal = {
        "features_and_labels_are_in_separate_archives",
        "each_role_has_a_separate_label_archive",
        "allow_pickle_required_to_load_sidecars",
        "open_role_runner_may_load_only",
        "calibration_and_internal_holdout_remain_unopened_by_model_runners",
    }
    if not isinstance(seal, Mapping) or set(seal) != expected_seal:
        raise ContractError("MELD manifest seal contract changed")
    if (
        seal["features_and_labels_are_in_separate_archives"] is not True
        or seal["each_role_has_a_separate_label_archive"] is not True
        or seal["allow_pickle_required_to_load_sidecars"] is not False
        or seal["open_role_runner_may_load_only"] != list(OPEN_ROLES)
        or seal["calibration_and_internal_holdout_remain_unopened_by_model_runners"] is not True
    ):
        raise ContractError("MELD manifest does not enforce physical open-role separation")

    roles = manifest["roles"]
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_NAMES):
        raise ContractError("MELD manifest role set changed")
    audio_dim = int(feature["audio_mean_std_columns"])
    video_dim = int(feature["video_mean_std_columns"])
    for role in ROLE_NAMES:
        record = roles[role]
        if not isinstance(record, Mapping) or set(record) != ROLE_RECORD_FIELDS:
            raise ContractError("MELD manifest role record changed")
        if record["feature_filename"] != f"features_{role}.npz":
            raise ContractError("MELD manifest feature filename changed")
        if record["label_filename"] != f"labels_{role}.npz":
            raise ContractError("MELD manifest label filename changed")
        if int(record["rows"]) < 1 or int(record["dialogues"]) < 1:
            raise ContractError("MELD manifest role rows/dialogues are invalid")
        if not 0 <= int(record["history_eligible_rows"]) <= int(record["rows"]):
            raise ContractError("MELD manifest history-eligible count is invalid")
        if (int(record["audio_dimension"]), int(record["video_dimension"])) != (
            audio_dim,
            video_dim,
        ):
            raise ContractError("MELD role dimensions differ from feature contract")
        for field in ("feature_sha256", "label_sha256", "row_alignment_sha256"):
            _valid_sha256(record[field], field=f"roles.{role}.{field}")
    _valid_sha256(manifest["config_sha256"], field="config_sha256")
    audit = manifest["public_content_audit"]
    expected_audit = {
        "contains_labels_or_class_counts": False,
        "contains_utterances_or_embeddings": False,
        "contains_dialogue_speaker_or_row_identifiers": False,
        "contains_private_absolute_paths": False,
    }
    if audit != expected_audit:
        raise ContractError("MELD public-content audit changed")
    return manifest


def _load_role_pair(
    sidecar_dir: Path,
    record: Mapping[str, object],
    *,
    expected_role: str,
    model_config: CausalBackboneConfig,
) -> _RoleArrays:
    if expected_role not in OPEN_ROLES:
        raise ContractError("MELD causal loader accepts open roles only")
    feature_path = sidecar_dir / str(record["feature_filename"])
    label_path = sidecar_dir / str(record["label_filename"])
    if not feature_path.is_file() or not label_path.is_file():
        raise ContractError(f"MELD open-role sidecar is missing: {expected_role}")
    feature_sha = sha256_file(feature_path)
    label_sha = sha256_file(label_path)
    if feature_sha != str(record["feature_sha256"]):
        raise ContractError("MELD feature sidecar hash differs from manifest")
    if label_sha != str(record["label_sha256"]):
        raise ContractError("MELD label sidecar hash differs from manifest")
    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != FEATURE_FIELDS:
            raise ContractError("MELD feature sidecar schema changed")
        feature = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(label_path, allow_pickle=False) as archive:
        if set(archive.files) != LABEL_FIELDS:
            raise ContractError("MELD label sidecar schema changed")
        label = {name: np.asarray(archive[name]) for name in archive.files}
    for payload in (feature, label):
        if _scalar_text(payload, "schema_version") != SIDECAR_SCHEMA_VERSION:
            raise ContractError("MELD sidecar schema version changed")
        if _scalar_text(payload, "role") != expected_role:
            raise ContractError("MELD sidecar role does not match its manifest role")
    alignment = _scalar_text(feature, "row_alignment_sha256")
    if alignment != _scalar_text(label, "row_alignment_sha256"):
        raise ContractError("MELD feature/label row alignment differs")
    if alignment != str(record["row_alignment_sha256"]):
        raise ContractError("MELD sidecar alignment differs from manifest")
    _valid_sha256(alignment, field="row_alignment_sha256")

    utterances = feature["utterances"]
    audio_raw = feature["audio_mean_std"]
    video_raw = feature["video_mean_std"]
    dialogues_raw = feature["dialogue_codes"]
    speakers_raw = feature["speaker_codes"]
    orders_raw = feature["utterance_order"]
    protocol_rows_raw = feature["protocol_row_ids"]
    labels_raw = label["labels"]
    rows = int(record["rows"])
    if utterances.shape != (rows,) or utterances.dtype.kind not in {"U", "S"}:
        raise ContractError("MELD utterance sidecar rows or dtype changed")
    if audio_raw.dtype != np.float32 or video_raw.dtype != np.float32:
        raise ContractError("MELD sidecar modality dtype differs from manifest")
    if audio_raw.shape != (rows, model_config.audio_dim) or video_raw.shape != (
        rows,
        model_config.video_dim,
    ):
        raise ContractError("MELD sidecar modality dimensions differ from model")
    if (model_config.audio_dim, model_config.video_dim) != (
        int(record["audio_dimension"]),
        int(record["video_dimension"]),
    ):
        raise ContractError("MELD model dimensions differ from manifest")
    one_dimensional = (
        dialogues_raw,
        speakers_raw,
        orders_raw,
        protocol_rows_raw,
        labels_raw,
    )
    if any(value.shape != (rows,) for value in one_dimensional):
        raise ContractError("MELD structural or label arrays are misaligned")
    if not all(np.issubdtype(value.dtype, np.integer) for value in one_dimensional):
        raise ContractError("MELD structural codes and labels must be integer arrays")
    audio = audio_raw.astype(np.float32, copy=True)
    video = video_raw.astype(np.float32, copy=True)
    dialogues = dialogues_raw.astype(np.int64, copy=True)
    speakers = speakers_raw.astype(np.int64, copy=True)
    orders = orders_raw.astype(np.int64, copy=True)
    protocol_rows = protocol_rows_raw.astype(np.int64, copy=True)
    labels = labels_raw.astype(np.int64, copy=True)
    if not np.isfinite(audio).all() or not np.isfinite(video).all():
        raise ContractError("MELD sidecar contains non-finite modality features")
    if np.any((labels < 0) | (labels >= model_config.num_classes)):
        raise ContractError("MELD sidecar label is outside the dataset label order")
    if np.any(dialogues < 0) or np.any(speakers < 0) or np.any(orders < 0):
        raise ContractError("MELD structural codes must be non-negative")
    if np.any(protocol_rows < 0) or len(set(protocol_rows.tolist())) != rows:
        raise ContractError("MELD protocol row ids must be unique non-negative source rows")
    if len(set(zip(dialogues.tolist(), orders.tolist(), strict=True))) != rows:
        raise ContractError("MELD role sidecar duplicates dialogue/utterance order")
    if int(record["dialogues"]) != len(set(dialogues.tolist())):
        raise ContractError("MELD manifest dialogue count differs from sidecar")
    history_eligible = int(
        sum(
            np.any(
                (dialogues == dialogues[index])
                & (speakers == speakers[index])
                & (orders < orders[index])
            )
            for index in range(rows)
        )
    )
    if history_eligible != int(record["history_eligible_rows"]):
        raise ContractError("MELD manifest history-eligible count differs from sidecar")
    return _RoleArrays(
        expected_role,
        utterances.astype(str),
        audio,
        video,
        dialogues,
        speakers,
        orders,
        protocol_rows,
        labels,
        alignment,
        feature_sha,
        label_sha,
    )


def _ordered(role: _RoleArrays) -> _RoleArrays:
    ordering = np.lexsort((role.speakers, role.orders, role.dialogues))
    return _RoleArrays(
        role.role,
        role.utterances[ordering],
        role.audio[ordering],
        role.video[ordering],
        role.dialogues[ordering],
        role.speakers[ordering],
        role.orders[ordering],
        role.protocol_rows[ordering],
        role.labels[ordering],
        role.alignment_sha256,
        role.feature_sha256,
        role.label_sha256,
    )


def _history_indices(
    dialogues: np.ndarray,
    speakers: np.ndarray,
    orders: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    histories: list[tuple[int, ...]] = []
    for query in range(len(dialogues)):
        eligible = np.flatnonzero(
            (dialogues == dialogues[query])
            & (speakers == speakers[query])
            & (orders < orders[query])
        )
        histories.append(
            tuple(sorted(eligible.tolist(), key=lambda value: (int(orders[value]), value)))
        )
    return tuple(histories)


def load_meld_open_role_corpus(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    model_config: CausalBackboneConfig,
) -> tuple[OpenRoleCorpus, VerifiedCorpusProvenance]:
    """Load only manifest-verified MELD fit/selection feature and label files."""

    manifest = _read_manifest(manifest_path)
    if model_config.num_classes != len(manifest["label_order"]):
        raise ContractError("MELD model class count differs from manifest label order")
    fit = _ordered(
        _load_role_pair(
            sidecar_dir,
            manifest["roles"][FIT_ROLE],
            expected_role=FIT_ROLE,
            model_config=model_config,
        )
    )
    selection = _ordered(
        _load_role_pair(
            sidecar_dir,
            manifest["roles"][SELECTION_ROLE],
            expected_role=SELECTION_ROLE,
            model_config=model_config,
        )
    )
    values = (fit, selection)
    utterances = np.concatenate([value.utterances for value in values])
    audio = np.concatenate([value.audio for value in values], axis=0)
    video = np.concatenate([value.video for value in values], axis=0)
    dialogues = np.concatenate([value.dialogues for value in values])
    raw_speakers = np.concatenate([value.speakers for value in values])
    orders = np.concatenate([value.orders for value in values])
    protocol_rows = np.concatenate([value.protocol_rows for value in values])
    labels = np.concatenate([value.labels for value in values])
    roles = np.concatenate(
        [np.asarray([value.role] * len(value.labels)) for value in values]
    )

    fit_speakers = sorted(set(fit.speakers.tolist()))
    speaker_mapping = {int(value): index + 1 for index, value in enumerate(fit_speakers)}
    if len(speaker_mapping) + 1 > model_config.num_speakers:
        raise ContractError("fit-only MELD speaker vocabulary exceeds model config")
    speaker_mapping_sha = _canonical_sha256(
        {"oov": 0, "fit_mapping": [[value, speaker_mapping[value]] for value in fit_speakers]}
    )
    speaker_ids = np.asarray(
        [speaker_mapping.get(int(value), 0) for value in raw_speakers], dtype=np.int64
    )
    speaker_identity = np.asarray(
        [
            hashlib.sha256(f"MELD-speaker\x1f{int(value)}".encode()).hexdigest()
            for value in raw_speakers
        ]
    )
    buckets = np.empty(len(roles), dtype=np.int16)
    for index, dialogue in enumerate(dialogues):
        expected_role, bucket = assign_group_role(
            "MELD",
            int(dialogue),
            str(manifest["split_protocol_id"]),
            FROZEN_ROLE_RANGES,
        )
        if expected_role != str(roles[index]):
            raise ContractError("MELD sidecar role differs from frozen dialogue assignment")
        buckets[index] = int(bucket)
    if np.any(buckets > 79) or set(roles.astype(str)) != set(OPEN_ROLES):
        raise ContractError("sealed MELD role entered the open-role corpus")
    if np.any(orders >= model_config.max_turns):
        raise ContractError("MELD utterance order exceeds backbone turn vocabulary")
    groups = np.asarray([f"MELD/{int(value)}" for value in dialogues])
    histories = _history_indices(dialogues, raw_speakers, orders)
    keys = np.asarray(
        [
            hashlib.sha256(f"MELD\x1f{int(dialogue)}\x1f{int(order)}".encode()).hexdigest()
            for dialogue, order in zip(dialogues, orders, strict=True)
        ]
    )
    corpus = OpenRoleCorpus(
        keys=keys,
        texts=tuple(utterances.astype(str)),
        audio=audio,
        video=video,
        labels=labels,
        groups=groups,
        roles=roles,
        buckets=buckets,
        speaker_ids=speaker_ids,
        turn_ids=orders,
        histories=histories,
        protocol_row_ids=protocol_rows,
        speaker_identity=speaker_identity,
        speaker_mapping_sha256=speaker_mapping_sha,
        label_access_mode="strict_physical_meld_train_fit_selection_feature_label_sidecars",
    )
    corpus.validate(model_config)
    manifest_sha = sha256_file(manifest_path)
    source_hashes = {
        "sidecar_manifest": manifest_sha,
        "trusted_source_train_csv": str(manifest["source_contract"]["train_csv_sha256"]),
        "trusted_source_train_pickle": str(
            manifest["source_contract"]["train_pickle_sha256"]
        ),
        "sidecar_config": str(manifest["config_sha256"]),
    }
    for value in values:
        source_hashes[f"{value.role}_features"] = value.feature_sha256
        source_hashes[f"{value.role}_labels"] = value.label_sha256
    provenance = create_verified_corpus_provenance(
        dataset_id="MELD",
        manifest_schema=str(manifest["schema_version"]),
        manifest_status=str(manifest["status"]),
        manifest_sha256=manifest_sha,
        source_hashes=source_hashes,
        label_order=tuple(str(value) for value in manifest["label_order"]),
        role_rows={
            role: int(manifest["roles"][role]["rows"]) for role in OPEN_ROLES
        },
        audio_dim=model_config.audio_dim,
        video_dim=model_config.video_dim,
        role_assignment_sha256=_role_assignment_sha256(corpus),
        speaker_mapping_sha256=speaker_mapping_sha,
        corpus_contract_sha256=_corpus_contract_sha256(corpus),
        verification_origin="meld_manifest_v2",
    )
    provenance.validate(corpus, model_config)
    return corpus, provenance


def _validate_feature_without_label_deserialization(
    sidecar_dir: Path,
    record: Mapping[str, object],
    *,
    expected_role: str,
    model_config: CausalBackboneConfig,
) -> dict[str, object]:
    """Validate one feature archive and only hash (never deserialize) its label file."""

    feature_path = sidecar_dir / str(record["feature_filename"])
    label_path = sidecar_dir / str(record["label_filename"])
    if not feature_path.is_file() or not label_path.is_file():
        raise ContractError(f"MELD preflight sidecar is missing: {expected_role}")
    feature_sha = sha256_file(feature_path)
    label_sha = sha256_file(label_path)
    if feature_sha != str(record["feature_sha256"]):
        raise ContractError("MELD preflight feature hash differs from manifest")
    if label_sha != str(record["label_sha256"]):
        raise ContractError("MELD preflight label hash differs from manifest")
    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != FEATURE_FIELDS:
            raise ContractError("MELD preflight feature schema changed")
        feature = {name: np.asarray(archive[name]) for name in archive.files}
    if _scalar_text(feature, "schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ContractError("MELD preflight feature schema version changed")
    if _scalar_text(feature, "role") != expected_role:
        raise ContractError("MELD preflight feature role differs from manifest")
    if _scalar_text(feature, "row_alignment_sha256") != str(
        record["row_alignment_sha256"]
    ):
        raise ContractError("MELD preflight feature alignment differs from manifest")
    rows = int(record["rows"])
    utterances = feature["utterances"]
    audio = feature["audio_mean_std"]
    video = feature["video_mean_std"]
    dialogues = feature["dialogue_codes"]
    speakers = feature["speaker_codes"]
    orders = feature["utterance_order"]
    protocol_rows = feature["protocol_row_ids"]
    if utterances.shape != (rows,) or utterances.dtype.kind not in {"U", "S"}:
        raise ContractError("MELD preflight utterance rows changed")
    if audio.dtype != np.float32 or video.dtype != np.float32:
        raise ContractError("MELD preflight modality dtype changed")
    if audio.shape != (rows, model_config.audio_dim) or video.shape != (
        rows,
        model_config.video_dim,
    ):
        raise ContractError("MELD preflight modality dimensions changed")
    structural = (dialogues, speakers, orders, protocol_rows)
    if any(value.shape != (rows,) for value in structural) or not all(
        np.issubdtype(value.dtype, np.integer) for value in structural
    ):
        raise ContractError("MELD preflight structural arrays changed")
    if not np.isfinite(audio).all() or not np.isfinite(video).all():
        raise ContractError("MELD preflight modality features are non-finite")
    dialogues = dialogues.astype(np.int64, copy=False)
    speakers = speakers.astype(np.int64, copy=False)
    orders = orders.astype(np.int64, copy=False)
    protocol_rows = protocol_rows.astype(np.int64, copy=False)
    if np.any(dialogues < 0) or np.any(speakers < 0) or np.any(orders < 0):
        raise ContractError("MELD preflight structural code is negative")
    if np.any(protocol_rows < 0) or len(set(protocol_rows.tolist())) != rows:
        raise ContractError("MELD preflight protocol row identity changed")
    if len(set(zip(dialogues.tolist(), orders.tolist(), strict=True))) != rows:
        raise ContractError("MELD preflight duplicates a dialogue/utterance row")
    if int(record["dialogues"]) != len(set(dialogues.tolist())):
        raise ContractError("MELD preflight dialogue count differs from manifest")
    history_eligible = int(
        sum(
            np.any(
                (dialogues == dialogues[index])
                & (speakers == speakers[index])
                & (orders < orders[index])
            )
            for index in range(rows)
        )
    )
    if history_eligible != int(record["history_eligible_rows"]):
        raise ContractError("MELD preflight history count differs from manifest")
    return {
        "role": expected_role,
        "rows": rows,
        "dialogues": int(record["dialogues"]),
        "history_eligible_rows": history_eligible,
        "feature_sha256": feature_sha,
        "label_sha256_verified_without_deserialization": label_sha,
    }


def _fit_only_corpus(
    fit: _RoleArrays,
    *,
    manifest: Mapping[str, object],
    model_config: CausalBackboneConfig,
) -> OpenRoleCorpus:
    fit = _ordered(fit)
    raw_speakers = fit.speakers
    fit_speakers = sorted(set(raw_speakers.tolist()))
    speaker_mapping = {int(value): index + 1 for index, value in enumerate(fit_speakers)}
    if len(speaker_mapping) + 1 > model_config.num_speakers:
        raise ContractError("fit-only MELD preflight speaker vocabulary exceeds model config")
    speaker_mapping_sha = _canonical_sha256(
        {"oov": 0, "fit_mapping": [[value, speaker_mapping[value]] for value in fit_speakers]}
    )
    speaker_ids = np.asarray(
        [speaker_mapping[int(value)] for value in raw_speakers], dtype=np.int64
    )
    speaker_identity = np.asarray(
        [
            hashlib.sha256(f"MELD-speaker\x1f{int(value)}".encode()).hexdigest()
            for value in raw_speakers
        ]
    )
    roles = np.asarray([FIT_ROLE] * len(fit.labels))
    buckets = np.empty(len(fit.labels), dtype=np.int16)
    for index, dialogue in enumerate(fit.dialogues):
        role, bucket = assign_group_role(
            "MELD",
            int(dialogue),
            str(manifest["split_protocol_id"]),
            FROZEN_ROLE_RANGES,
        )
        if role != FIT_ROLE:
            raise ContractError("MELD preflight fit role differs from frozen assignment")
        buckets[index] = int(bucket)
    groups = np.asarray([f"MELD/{int(value)}" for value in fit.dialogues])
    histories = _history_indices(fit.dialogues, raw_speakers, fit.orders)
    keys = np.asarray(
        [
            hashlib.sha256(f"MELD\x1f{int(dialogue)}\x1f{int(order)}".encode()).hexdigest()
            for dialogue, order in zip(fit.dialogues, fit.orders, strict=True)
        ]
    )
    corpus = OpenRoleCorpus(
        keys=keys,
        texts=tuple(fit.utterances.astype(str)),
        audio=fit.audio,
        video=fit.video,
        labels=fit.labels,
        groups=groups,
        roles=roles,
        buckets=buckets,
        speaker_ids=speaker_ids,
        turn_ids=fit.orders,
        histories=histories,
        protocol_row_ids=fit.protocol_rows,
        speaker_identity=speaker_identity,
        speaker_mapping_sha256=speaker_mapping_sha,
        label_access_mode="meld_fit_labels_only_nonperformance_preflight",
    )
    corpus.validate(model_config)
    return corpus


def preflight_meld_causal_backbone_inputs(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    sampling_config: UtilitySamplingConfig,
) -> dict[str, object]:
    """Run a CPU structural/forward preflight without deserializing selection labels."""

    run_config.validate()
    manifest = _read_manifest(manifest_path)
    if model_config.num_classes != len(manifest["label_order"]):
        raise ContractError("MELD preflight model classes differ from label order")
    fit = _load_role_pair(
        sidecar_dir,
        manifest["roles"][FIT_ROLE],
        expected_role=FIT_ROLE,
        model_config=model_config,
    )
    selection_feature = _validate_feature_without_label_deserialization(
        sidecar_dir,
        manifest["roles"][SELECTION_ROLE],
        expected_role=SELECTION_ROLE,
        model_config=model_config,
    )
    corpus = _fit_only_corpus(fit, manifest=manifest, model_config=model_config)
    splits = make_crossfit_splits(
        corpus,
        outer_folds=run_config.outer_folds,
        validation_fraction=run_config.inner_validation_fraction,
        seed=EXPECTED_SEEDS[0],
    )
    tasks = sample_corpus_bidirectional_tasks(corpus, sampling_config)
    if not tasks:
        raise ContractError("MELD preflight found no non-trivial fit-only utility task")
    first_split = splits[0]
    processor = fit_fold_text_processor(
        corpus.texts,
        first_split.inner_train_indices,
        output_dim=model_config.text_dim,
        config=run_config,
        seed=EXPECTED_SEEDS[0],
    )
    text_features = processor.transform(corpus.texts)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EXPECTED_SEEDS[0])
        model = CausalMultimodalBackbone(model_config).to(torch.device("cpu")).eval()
        probability = predict_utility_contexts(
            model,
            corpus,
            text_features,
            tasks[:1],
            device=torch.device("cpu"),
            batch_size=4,
            max_history_items=run_config.max_history_items,
        )
    if probability.shape != (1, 4, model_config.num_classes) or not np.isfinite(
        probability
    ).all():
        raise ContractError("MELD preflight forward/backward probability contract failed")
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "pass_nonperformance_fit_only_no_selection_label_deserialization",
        "dataset": "MELD",
        "manifest": {
            "schema_version": manifest["schema_version"],
            "sha256": sha256_file(manifest_path),
        },
        "data_access": {
            "fit_features_deserialized": True,
            "fit_labels_deserialized_for_structural_range_validation_only": True,
            "selection_features_deserialized": True,
            "selection_labels_deserialized": False,
            "calibration_or_internal_holdout_arrays_opened": False,
            "dev_or_test_arrays_opened": False,
        },
        "fit_contract": {
            "rows": len(corpus.keys),
            "groups": len(set(corpus.groups.astype(str))),
            "history_eligible_rows": int(sum(bool(value) for value in corpus.histories)),
            "crossfit_folds": len(splits),
            "sampled_bidirectional_tasks": len(tasks),
            "protocol_row_ids_sha256": _canonical_sha256(
                corpus.protocol_row_ids.astype(int).tolist()
            ),
        },
        "selection_feature_contract": selection_feature,
        "forward_contract": {
            "contexts": ["s", "s_plus_candidate", "t", "t_minus_candidate"],
            "probability_shape": list(probability.shape),
            "finite": True,
            "metrics_or_utilities_computed": False,
        },
        "performance_claim_authorized": False,
    }


def _read_json_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be a mapping: {path.name}")
    return value


def run_meld_causal_backbone(
    *,
    sidecar_dir: Path,
    sidecar_manifest_path: Path,
    backbone_config_path: Path,
    utility_config_path: Path,
    confirmatory_config_path: Path,
    private_output_dir: Path,
    public_output_path: Path,
    repository_root: Path,
    device_name: str = "auto",
) -> dict:
    """Production MELD entry point pinned to the frozen five-seed contract."""

    backbone_payload = _read_json_mapping(backbone_config_path)
    utility_payload = _read_json_mapping(utility_config_path)
    confirmatory_payload = _read_json_mapping(confirmatory_config_path)
    validate_open_role_backbone_payload(backbone_payload)
    model_config = CausalBackboneConfig.from_mapping(backbone_payload)
    run_config = BackboneRunConfig.from_mapping(backbone_payload)
    sampling_config = UtilitySamplingConfig.from_mapping(utility_payload)
    roles = utility_payload.get("data_roles")
    if not isinstance(roles, Mapping):
        raise ContractError("utility configuration lacks frozen data roles")
    role_ranges = {name: roles.get(name) for name in FROZEN_ROLE_RANGES}
    if role_ranges != FROZEN_ROLE_RANGES:
        raise ContractError("utility role ranges changed")
    if str(roles.get("split_protocol_id", "")) != "scu_set_exploration_v1":
        raise ContractError("utility split protocol changed")
    independent = confirmatory_payload.get("independent_runs")
    if not isinstance(independent, Mapping):
        raise ContractError("confirmatory configuration lacks independent_runs")
    seeds = tuple(int(value) for value in independent.get("seeds", ()))
    if seeds != EXPECTED_SEEDS or int(independent.get("required_seed_count", 0)) != 5:
        raise ContractError("production MELD backbone requires frozen seeds 17/29/43/71/101")
    if (model_config.text_dim, model_config.audio_dim, model_config.video_dim) != (
        256,
        64,
        4096,
    ):
        raise ContractError("production MELD dimensions must be SVD256/audio64/video4096")
    if CausalMultimodalBackbone(model_config).parameter_count() >= 2_000_000:
        raise ContractError("production MELD causal backbone is not strictly under 2M parameters")
    corpus, provenance = load_meld_open_role_corpus(
        sidecar_dir=sidecar_dir,
        manifest_path=sidecar_manifest_path,
        model_config=model_config,
    )
    provenance = create_verified_corpus_provenance(
        dataset_id=provenance.dataset_id,
        manifest_schema=provenance.manifest_schema,
        manifest_status=provenance.manifest_status,
        manifest_sha256=provenance.manifest_sha256,
        source_hashes={
            **provenance.source_hashes,
            "backbone_config": sha256_file(backbone_config_path),
            "utility_config": sha256_file(utility_config_path),
            "confirmatory_config": sha256_file(confirmatory_config_path),
        },
        label_order=provenance.label_order,
        role_rows=provenance.role_rows,
        audio_dim=provenance.audio_dim,
        video_dim=provenance.video_dim,
        role_assignment_sha256=provenance.role_assignment_sha256,
        speaker_mapping_sha256=provenance.speaker_mapping_sha256,
        corpus_contract_sha256=provenance.corpus_contract_sha256,
        verification_origin=provenance.verification_origin,
    )
    provenance.validate(corpus, model_config)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    return execute_crossfit_backbone(
        corpus,
        provenance=provenance,
        model_config=model_config,
        run_config=run_config,
        sampling_config=sampling_config,
        seeds=seeds,
        private_output_dir=private_output_dir,
        public_output_path=public_output_path,
        repository_root=repository_root,
        device=device,
        private_cache_filename="meld_causal_backbone_oof_v1.npz",
    )
