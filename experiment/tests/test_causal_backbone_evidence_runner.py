from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hva_affect.causal_backbone_evidence_runner as runner
from hva_affect.causal_backbone_evidence_runner import (
    ENDPOINT_CONTEXT_NAMES,
    EXPECTED_SEEDS,
    INDEPENDENT_CURRENT_ONLY_PROTOCOL,
    PRODUCER_CACHE_SCHEMA,
    UTILITY_CONTEXT_NAMES,
    CURRENT_ONLY_FIT_ARTIFACT_SCHEMA,
    EMOTIONTALK_FEATURE_SCHEMA,
    EMOTIONTALK_LABEL_NAMES,
    EMOTIONTALK_LABEL_SCHEMA,
    EMOTIONTALK_MANIFEST_SCHEMA,
    EMOTIONTALK_PROTOCOL,
    FIT_PREFLIGHT_RECEIPT_SCHEMA,
    FIT_ROLE,
    MELD_LABEL_NAMES,
    MELD_MANIFEST_SCHEMA,
    MELD_PROTOCOL,
    MELD_SIDECAR_SCHEMA,
    SELECTION_ROLE,
    UTILITY_OOF_SCORE_SCHEMA,
    StageAContractError,
    build_checkpoint_manifest,
    load_fit_only_producer_view,
    load_verified_fold_artifacts,
    materialize_selection_features_after_receipt,
    run_fit_preflight,
    validate_current_only_fit_artifact,
    validate_fit_receipt,
    validate_utility_oof_score_artifact,
)


LABEL_NAMES = EMOTIONTALK_LABEL_NAMES
EMOTION_TO_INDEX = {name: index for index, name in enumerate(MELD_LABEL_NAMES)}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


def _role_rows(role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if role == FIT_ROLE:
        protocol = np.asarray([0, 1, 4, 5], dtype=np.int64)
        group_base = 10
    else:
        protocol = np.asarray([2, 3, 6, 7], dtype=np.int64)
        group_base = 30
    groups = np.asarray([group_base, group_base, group_base + 1, group_base + 1])
    speakers = np.asarray([1, 1, 2, 2], dtype=np.int64)
    turns = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return protocol, groups, speakers, turns


def _make_emotiontalk_sidecars(root: Path, *, poison_selection: bool = False) -> Path:
    label_order = tuple(str(value) for value in LABEL_NAMES)
    roles: dict[str, dict[str, Any]] = {}
    feature_config = _sha("emotiontalk-feature-config")
    label_source = _sha("emotiontalk-label-source")
    for role in (FIT_ROLE, SELECTION_ROLE):
        protocol, group_codes, speaker_codes, turns = _role_rows(role)
        alignment = _sha(f"EmotionTalk-{role}-alignment")
        feature_path = root / f"features_{role}.npz"
        label_path = root / f"labels_{role}.npz"
        if poison_selection and role == SELECTION_ROLE:
            _write_npz(
                feature_path,
                {
                    "schema_version": np.asarray([object() for _ in range(1)], dtype=object),
                    "dataset_id": np.asarray(object(), dtype=object),
                    "role": np.asarray(object(), dtype=object),
                    "split_protocol_id": np.asarray(object(), dtype=object),
                    "row_alignment_sha256": np.asarray(object(), dtype=object),
                    "opaque_row_hashes": np.asarray([object()], dtype=object),
                    "opaque_group_hashes": np.asarray([object()], dtype=object),
                    "speaker_tokens": np.asarray([object()], dtype=object),
                    "turn_ids": np.asarray([object()], dtype=object),
                    "protocol_row_ids": np.asarray([object()], dtype=object),
                    "role_buckets": np.asarray([object()], dtype=object),
                    "texts": np.asarray([object()], dtype=object),
                    "audio_features": np.asarray([object()], dtype=object),
                    "video_features": np.asarray([object()], dtype=object),
                    "source_feature_config_sha256": np.asarray(object(), dtype=object),
                },
            )
            _write_npz(
                label_path,
                {
                    "schema_version": np.asarray(object(), dtype=object),
                    "dataset_id": np.asarray(object(), dtype=object),
                    "role": np.asarray(object(), dtype=object),
                    "split_protocol_id": np.asarray(object(), dtype=object),
                    "row_alignment_sha256": np.asarray(object(), dtype=object),
                    "labels": np.asarray([object()], dtype=object),
                    "source_label_sha256": np.asarray(object(), dtype=object),
                },
            )
        else:
            groups = np.asarray([_sha(f"group-{value}") for value in group_codes])
            speakers = np.asarray([f"speaker-{value}" for value in speaker_codes])
            _write_npz(
                feature_path,
                {
                    "schema_version": np.asarray(EMOTIONTALK_FEATURE_SCHEMA),
                    "dataset_id": np.asarray("EmotionTalk"),
                    "role": np.asarray(role),
                    "split_protocol_id": np.asarray("scu_set_exploration_v1"),
                    "row_alignment_sha256": np.asarray(alignment),
                    "opaque_row_hashes": np.asarray(
                        [_sha(f"EmotionTalk-row-{role}-{index}") for index in range(4)]
                    ),
                    "opaque_group_hashes": groups,
                    "speaker_tokens": speakers,
                    "turn_ids": turns,
                    "protocol_row_ids": protocol,
                    "role_buckets": np.asarray(
                        [10, 10, 20, 20] if role == FIT_ROLE else [65, 65, 70, 70],
                        dtype=np.int16,
                    ),
                    "texts": np.asarray([f"emotion text {role} {index}" for index in range(4)]),
                    "audio_features": np.arange(12, dtype=np.float32).reshape(4, 3),
                    "video_features": np.arange(8, dtype=np.float32).reshape(4, 2),
                    "source_feature_config_sha256": np.asarray(feature_config),
                },
            )
            _write_npz(
                label_path,
                {
                    "schema_version": np.asarray(EMOTIONTALK_LABEL_SCHEMA),
                    "dataset_id": np.asarray("EmotionTalk"),
                    "role": np.asarray(role),
                    "split_protocol_id": np.asarray("scu_set_exploration_v1"),
                    "row_alignment_sha256": np.asarray(alignment),
                    "labels": np.asarray([0, 1, 2, 3], dtype=np.int64),
                    "source_label_sha256": np.asarray(label_source),
                },
            )
        roles[role] = {
            "feature_filename": feature_path.name,
            "label_filename": label_path.name,
            "rows": 4,
            "groups": 2,
            "history_eligible_rows": 2,
            "audio_dimension": 3,
            "video_dimension": 2,
            "feature_sha256": _file_sha(feature_path),
            "label_sha256": _file_sha(label_path),
            "row_alignment_sha256": alignment,
        }
    manifest = {
        "schema_version": EMOTIONTALK_MANIFEST_SCHEMA,
        "protocol": EMOTIONTALK_PROTOCOL,
        "status": "strict_open_role_feature_and_label_sidecars_created_and_hashed",
        "dataset_id": "EmotionTalk",
        "split_protocol_id": "scu_set_exploration_v1",
        "label_order": list(label_order),
        "source_contract": {
            "label_archive": label_source,
            "media_features": _sha("emotiontalk-media"),
            "transcription": _sha("emotiontalk-transcription"),
            "feature_config_sha256": feature_config,
            "trusted_source_boundary_only": True,
            "validation_or_test_label_payload_opened": False,
        },
        "seal_contract": {
            "model_runner_opens_upstream_media_npz_or_transcription": False,
            "open_role_runner_may_load_only": [FIT_ROLE, SELECTION_ROLE],
            "calibration_holdout_validation_test_sidecars_created": False,
            "allow_pickle_required_to_load_sidecars": False,
        },
        "roles": roles,
        "config_sha256": _sha("emotiontalk-config"),
        "public_content_audit": {
            "contains_labels_or_class_counts": False,
            "contains_row_group_or_speaker_identifiers": False,
            "contains_transcripts_or_embeddings": False,
            "contains_private_absolute_paths": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _make_meld_sidecars(root: Path, *, poison_selection: bool = False) -> Path:
    roles: dict[str, dict[str, Any]] = {}
    for role in (FIT_ROLE, SELECTION_ROLE):
        protocol, dialogues, speakers, turns = _role_rows(role)
        alignment = _sha(f"MELD-{role}-alignment")
        feature_path = root / f"features_{role}.npz"
        label_path = root / f"labels_{role}.npz"
        if poison_selection and role == SELECTION_ROLE:
            feature_values = {
                name: np.asarray([object()], dtype=object)
                for name in (
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
                )
            }
            label_values = {
                name: np.asarray([object()], dtype=object)
                for name in ("schema_version", "role", "row_alignment_sha256", "labels")
            }
            _write_npz(feature_path, feature_values)
            _write_npz(label_path, label_values)
        else:
            _write_npz(
                feature_path,
                {
                    "schema_version": np.asarray([MELD_SIDECAR_SCHEMA]),
                    "role": np.asarray([role]),
                    "row_alignment_sha256": np.asarray([alignment]),
                    "utterances": np.asarray([f"meld {role} {index}" for index in range(4)]),
                    "audio_mean_std": np.arange(12, dtype=np.float32).reshape(4, 3),
                    "video_mean_std": np.arange(8, dtype=np.float32).reshape(4, 2),
                    "dialogue_codes": dialogues,
                    "speaker_codes": speakers,
                    "utterance_order": turns,
                    "protocol_row_ids": protocol,
                },
            )
            _write_npz(
                label_path,
                {
                    "schema_version": np.asarray([MELD_SIDECAR_SCHEMA]),
                    "role": np.asarray([role]),
                    "row_alignment_sha256": np.asarray([alignment]),
                    "labels": np.asarray([0, 1, 2, 3], dtype=np.int64),
                },
            )
        roles[role] = {
            "feature_filename": feature_path.name,
            "label_filename": label_path.name,
            "rows": 4,
            "dialogues": 2,
            "history_eligible_rows": 2,
            "audio_dimension": 3,
            "video_dimension": 2,
            "feature_sha256": _file_sha(feature_path),
            "label_sha256": _file_sha(label_path),
            "row_alignment_sha256": alignment,
        }
    # The production MELD v2 manifest records the sealed roles as public
    # aggregate metadata.  Their files deliberately do not exist in this
    # fixture: open-role capabilities must validate the metadata without ever
    # resolving or touching the corresponding payload paths.
    for role in ("calibration", "internal_holdout"):
        roles[role] = {
            "feature_filename": f"features_{role}.npz",
            "label_filename": f"labels_{role}.npz",
            "rows": 4,
            "dialogues": 2,
            "history_eligible_rows": 2,
            "audio_dimension": 3,
            "video_dimension": 2,
            "feature_sha256": _sha(f"MELD-{role}-feature"),
            "label_sha256": _sha(f"MELD-{role}-label"),
            "row_alignment_sha256": _sha(f"MELD-{role}-alignment"),
        }
    manifest = {
        "schema_version": MELD_MANIFEST_SCHEMA,
        "protocol": MELD_PROTOCOL,
        "status": "role_separated_train_sidecars_created_and_hashed",
        "dataset_id": "MELD",
        "split_protocol_id": "scu_set_exploration_v1",
        "label_order": list(EMOTION_TO_INDEX),
        "claim_boundary": "synthetic train-only sidecar fixture",
        "source_contract": {
            "train_csv_sha256": _sha("meld-csv"),
            "train_pickle_sha256": _sha("meld-pickle"),
            "official_csv_is_authoritative_label_source": True,
            "embedded_pickle_label_used_for_training_or_metrics": False,
            "embedded_pickle_label_consistency_checked_by_trusted_custodian": True,
            "embedded_pickle_label_mismatch_statistics_exposed": False,
            "missing_feature_rows": 0,
            "extra_feature_rows": 0,
        },
        "feature_contract": {
            "audio_mean_std_columns": 3,
            "video_mean_std_columns": 2,
            "numeric_dtype": "float32",
            "strict_same_dialogue_same_speaker_past_history_supported": True,
            "protocol_row_identity": "zero_based_official_train_csv_row_index",
        },
        "seal_contract": {
            "features_and_labels_are_in_separate_archives": True,
            "each_role_has_a_separate_label_archive": True,
            "allow_pickle_required_to_load_sidecars": False,
            "open_role_runner_may_load_only": [FIT_ROLE, SELECTION_ROLE],
            "calibration_and_internal_holdout_remain_unopened_by_model_runners": True,
        },
        "roles": roles,
        "config_sha256": _sha("meld-config"),
        "public_content_audit": {
            "contains_labels_or_class_counts": False,
            "contains_utterances_or_embeddings": False,
            "contains_dialogue_speaker_or_row_identifiers": False,
            "contains_private_absolute_paths": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _sidecar_fixture(root: Path, dataset: str, *, poison_selection: bool = False) -> Path:
    return (
        _make_emotiontalk_sidecars(root, poison_selection=poison_selection)
        if dataset == "EmotionTalk"
        else _make_meld_sidecars(root, poison_selection=poison_selection)
    )


def _lineage_files(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    config = root / "frozen-config.json"
    code = root / "frozen-code.py"
    config.write_text('{"frozen": true}\n', encoding="utf-8")
    code.write_text("FROZEN = True\n", encoding="utf-8")
    return {
        "frozen_config": config
    }, {
        "experiment/src/hva_affect/frozen_code.py": code
    }


ENVIRONMENT = {"python": "synthetic", "numpy": "synthetic", "platform": "synthetic"}


@pytest.mark.parametrize("dataset", ["EmotionTalk", "MELD"])
def test_fit_preflight_never_np_loads_selection_payloads(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _sidecar_fixture(tmp_path, dataset, poison_selection=True)
    configs, code = _lineage_files(tmp_path)
    selection_paths = {
        tmp_path / f"features_{SELECTION_ROLE}.npz",
        tmp_path / f"labels_{SELECTION_ROLE}.npz",
    }
    opened: list[Path] = []
    original = np.load

    def guarded_load(path: object, *args: object, **kwargs: object):
        resolved = Path(path)  # type: ignore[arg-type]
        opened.append(resolved)
        if resolved in selection_paths:
            raise AssertionError("selection payload was deserialized during fit preflight")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner.np, "load", guarded_load)
    result = run_fit_preflight(
        dataset=dataset,
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=tmp_path / "fit-receipt.json",
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    assert set(path.name for path in opened) == {
        f"features_{FIT_ROLE}.npz",
        f"labels_{FIT_ROLE}.npz",
    }
    assert result.fit.rows == 4
    assert result.receipt["schema_version"] == FIT_PREFLIGHT_RECEIPT_SCHEMA
    access = result.receipt["access_contract"]
    assert access["selection_feature_deserialized"] is False  # type: ignore[index]
    assert access["selection_label_deserialized"] is False  # type: ignore[index]
    assert access["training_run"] is False  # type: ignore[index]
    assert access["performance_metric_computed"] is False  # type: ignore[index]
    validate_fit_receipt(result.receipt)


def test_meld_open_role_capabilities_never_touch_sealed_role_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _make_meld_sidecars(tmp_path)
    sealed_names = {
        f"{kind}_{role}.npz"
        for kind in ("features", "labels")
        for role in ("calibration", "internal_holdout")
    }
    assert all(not (tmp_path / name).exists() for name in sealed_names)
    touched: list[str] = []
    original = runner._file_sha256

    def guarded_hash(path: Path) -> str:
        if path.name in sealed_names:
            raise AssertionError("sealed MELD payload was touched")
        touched.append(path.name)
        return original(path)

    monkeypatch.setattr(runner, "_file_sha256", guarded_hash)
    fit_only = runner.hash_fit_role_sidecars_only(
        dataset="MELD",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
    )
    assert fit_only.fit.rows == 4
    selection_feature = runner.hash_fit_and_selection_feature_sidecars_only(
        dataset="MELD",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
    )
    assert selection_feature.selection.rows == 4
    assert sealed_names.isdisjoint(touched)


@pytest.mark.parametrize("dataset", ["EmotionTalk", "MELD"])
def test_completion_reverifies_receipt_then_opens_feature_but_not_label(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _sidecar_fixture(tmp_path, dataset)
    configs, code = _lineage_files(tmp_path)
    receipt = tmp_path / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset=dataset,
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=receipt,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    opened: list[str] = []
    original = np.load

    def spy(path: object, *args: object, **kwargs: object):
        opened.append(Path(path).name)  # type: ignore[arg-type]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner.np, "load", spy)
    view = materialize_selection_features_after_receipt(
        receipt_path=receipt,
        expected_receipt_sha256=preflight.receipt_sha256,
        dataset=dataset,
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    assert opened == [f"features_{SELECTION_ROLE}.npz"]
    assert view.labels_materialized is False
    assert len(view.texts) == 4


def test_completion_hash_mismatch_fails_before_selection_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _make_meld_sidecars(tmp_path)
    configs, code = _lineage_files(tmp_path)
    receipt = tmp_path / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset="MELD",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=receipt,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    configs["frozen_config"].write_text('{"frozen": false}\n', encoding="utf-8")
    opened: list[str] = []
    original = np.load

    def spy(path: object, *args: object, **kwargs: object):
        opened.append(Path(path).name)  # type: ignore[arg-type]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner.np, "load", spy)
    with pytest.raises(StageAContractError, match="lineage changed"):
        materialize_selection_features_after_receipt(
            receipt_path=receipt,
            expected_receipt_sha256=preflight.receipt_sha256,
            dataset="MELD",
            sidecar_dir=tmp_path,
            manifest_path=manifest,
            config_paths=configs,
            code_paths=code,
            environment=ENVIRONMENT,
        )
    assert opened == []


def test_fit_receipt_is_write_once_and_aggregate_only(tmp_path: Path) -> None:
    manifest = _make_emotiontalk_sidecars(tmp_path)
    configs, code = _lineage_files(tmp_path)
    receipt = tmp_path / "fit-receipt.json"
    first = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=receipt,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    serialized = receipt.read_text(encoding="utf-8")
    for forbidden in ("emotion text", str(tmp_path), '"labels"', '"protocol_row_ids"'):
        assert forbidden not in serialized
    assert first.receipt_sha256 == _file_sha(receipt)
    with pytest.raises(FileExistsError):
        run_fit_preflight(
            dataset="EmotionTalk",
            sidecar_dir=tmp_path,
            manifest_path=manifest,
            receipt_path=receipt,
            config_paths=configs,
            code_paths=code,
            environment=ENVIRONMENT,
        )


def test_fit_receipt_never_clobbers_a_concurrent_winner(tmp_path: Path) -> None:
    manifest = _make_emotiontalk_sidecars(tmp_path)
    configs, code = _lineage_files(tmp_path)
    receipt = tmp_path / "concurrent-fit-receipt.json"

    def attempt(_: int) -> str:
        try:
            run_fit_preflight(
                dataset="EmotionTalk",
                sidecar_dir=tmp_path,
                manifest_path=manifest,
                receipt_path=receipt,
                config_paths=configs,
                code_paths=code,
                environment=ENVIRONMENT,
            )
        except FileExistsError:
            return "exists"
        return "written"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert outcomes.count("written") == 1
    assert outcomes.count("exists") == 7
    validate_fit_receipt(json.loads(receipt.read_text(encoding="utf-8")))
    assert not list(tmp_path.glob(".concurrent-fit-receipt.json.*.tmp"))


def _uniform(shape: tuple[int, ...]) -> np.ndarray:
    result = np.ones(shape, dtype=np.float32)
    result /= shape[-1]
    return result


def _producer_mapping(*, selection_poison: bool = True) -> dict[str, np.ndarray]:
    classes = len(LABEL_NAMES)
    fit_query = np.asarray([1, 3, 5, 7], dtype=np.int64)
    fit_cluster = np.asarray([0, 0, 1, 1], dtype=np.int32)
    task_query = fit_query.copy()
    task_candidate = fit_query - 1
    s_indptr = np.zeros(len(task_query) + 1, dtype=np.int64)
    s_indices = np.asarray([], dtype=np.int64)
    t_indptr = np.arange(len(task_query) + 1, dtype=np.int64)
    t_indices = task_candidate.copy()
    task_payload = [
        {"query": int(query), "candidate": int(candidate), "s": [], "t": [int(candidate)]}
        for query, candidate in zip(task_query, task_candidate, strict=True)
    ]
    fit_endpoint = _uniform((5, 4, 2, classes))
    fit_utility_probability = _uniform((5, 4, 4, classes))
    fit_forward = np.zeros((5, 4), dtype=np.float32)
    fit_backward = np.zeros((5, 4), dtype=np.float32)
    poison = np.asarray([object()], dtype=object)
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PRODUCER_CACHE_SCHEMA),
        "dataset": np.asarray("EmotionTalk"),
        "dataset_label_order": np.asarray(LABEL_NAMES),
        "manifest_schema": np.asarray("manifest"),
        "manifest_status": np.asarray("status"),
        "manifest_sha256": np.asarray("1" * 64),
        "verified_provenance_attestation_sha256": np.asarray("2" * 64),
        "corpus_contract_sha256": np.asarray("3" * 64),
        "histories_sha256": np.asarray("4" * 64),
        "speaker_mapping_sha256": np.asarray("5" * 64),
        "runtime_environment_sha256": np.asarray("6" * 64),
        "source_identity_sha256": np.asarray("7" * 64),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "endpoint_context_names": np.asarray(ENDPOINT_CONTEXT_NAMES),
        "utility_context_names": np.asarray(UTILITY_CONTEXT_NAMES),
        "fit_query_indices": fit_query,
        "fit_cluster_codes": fit_cluster,
        "protocol_row_ids": np.arange(8, dtype=np.int64),
        "fit_endpoint_probability_oof": fit_endpoint,
        "fit_utility_probability_oof": fit_utility_probability,
        "fit_forward_utility": fit_forward,
        "fit_backward_utility": fit_backward,
        "fit_asymmetry": np.zeros_like(fit_forward),
        "fit_sign_agreement": np.ones_like(fit_forward, dtype=bool),
        "fit_task_sha256": np.asarray(_canonical_sha(task_payload)),
        "fit_task_query_indices": task_query,
        "fit_task_candidate_indices": task_candidate,
        "fit_task_s_indptr": s_indptr,
        "fit_task_s_indices": s_indices,
        "fit_task_t_indptr": t_indptr,
        "fit_task_t_indices": t_indices,
        "checkpoint_manifest_sha256": np.asarray("8" * 64),
        "utility_source": np.asarray(
            "recomputed_from_causal_backbone_probabilities_and_open_role_labels"
        ),
        "matrix_fit_endpoint_probability_oof_sha256": np.asarray(_array_sha(fit_endpoint)),
        "matrix_fit_utility_probability_oof_sha256": np.asarray(
            _array_sha(fit_utility_probability)
        ),
        "matrix_fit_forward_utility_sha256": np.asarray(_array_sha(fit_forward)),
        "matrix_fit_backward_utility_sha256": np.asarray(_array_sha(fit_backward)),
        "source_sidecar_manifest_sha256": np.asarray("9" * 64),
    }
    selection_keys = {
        "selection_query_indices",
        "selection_cluster_codes",
        "selection_endpoint_probability_fold_ensemble",
        "selection_utility_probability_fold_ensemble",
        "selection_forward_utility",
        "selection_backward_utility",
        "selection_asymmetry",
        "selection_sign_agreement",
        "selection_task_sha256",
        "matrix_selection_endpoint_probability_fold_ensemble_sha256",
        "matrix_selection_utility_probability_fold_ensemble_sha256",
        "matrix_selection_forward_utility_sha256",
        "matrix_selection_backward_utility_sha256",
        "selection_task_query_indices",
        "selection_task_candidate_indices",
        "selection_task_s_indptr",
        "selection_task_s_indices",
        "selection_task_t_indptr",
        "selection_task_t_indices",
    }
    for key in selection_keys:
        values[key] = poison if selection_poison else np.asarray([0], dtype=np.int64)
    return values


def test_fit_only_producer_view_never_deserializes_selection_poison(tmp_path: Path) -> None:
    path = tmp_path / "producer.npz"
    _write_npz(path, _producer_mapping(selection_poison=True))
    producer = load_fit_only_producer_view(path)
    assert producer.dataset == "EmotionTalk"
    assert producer.seeds == EXPECTED_SEEDS
    assert producer.fit_utility_probability.shape == (5, 4, 4, len(LABEL_NAMES))


def _checkpoint_tree(root: Path, folds: int = 2) -> None:
    for seed in EXPECTED_SEEDS:
        for fold in range(folds):
            run = root / f"seed_{seed:05d}" / f"fold_{fold:02d}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "checkpoint.pt").write_bytes(f"checkpoint-{seed}-{fold}".encode())
            (run / "text_processor.joblib").write_bytes(f"processor-{seed}-{fold}".encode())


def test_checkpoint_manifest_verifies_before_any_deserialization(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _checkpoint_tree(checkpoint_root)
    manifest = build_checkpoint_manifest(checkpoint_root, outer_folds=2)
    (checkpoint_root / "seed_00017" / "fold_00" / "checkpoint.pt").write_bytes(b"corrupt")
    called = {"checkpoint": 0, "processor": 0}

    def checkpoint_loader(_path: Path) -> object:
        called["checkpoint"] += 1
        return object()

    def processor_loader(_path: Path) -> object:
        called["processor"] += 1
        return object()

    with pytest.raises(StageAContractError, match="differs before deserialization"):
        load_verified_fold_artifacts(
            checkpoint_root,
            manifest,
            seed=17,
            fold=0,
            checkpoint_loader=checkpoint_loader,
            processor_loader=processor_loader,
        )
    assert called == {"checkpoint": 0, "processor": 0}


def test_missing_checkpoint_fails_closed_without_train_or_loader(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _checkpoint_tree(checkpoint_root)
    manifest = build_checkpoint_manifest(checkpoint_root, outer_folds=2)
    (checkpoint_root / "seed_00029" / "fold_01" / "checkpoint.pt").unlink()
    called = 0

    def loader(_path: Path) -> object:
        nonlocal called
        called += 1
        return object()

    with pytest.raises(StageAContractError, match="missing"):
        load_verified_fold_artifacts(
            checkpoint_root,
            manifest,
            seed=29,
            fold=1,
            checkpoint_loader=loader,
            processor_loader=loader,
        )
    assert called == 0


def _valid_current_fit_mapping(producer: runner.FitOnlyProducerView, manifest: runner.CheckpointManifest) -> dict[str, np.ndarray]:
    probability = _uniform((5, len(producer.fit_query_indices), len(producer.label_order)))
    fold = np.tile(np.asarray([0, 0, 1, 1], dtype=np.int32), (5, 1))
    return {
        "schema_version": np.asarray(CURRENT_ONLY_FIT_ARTIFACT_SCHEMA),
        "dataset": np.asarray(producer.dataset),
        "dataset_label_order": np.asarray(producer.label_order),
        "seeds": np.asarray(producer.seeds, dtype=np.int64),
        "outer_folds": np.asarray(manifest.outer_folds, dtype=np.int64),
        "fit_query_indices": producer.fit_query_indices.copy(),
        "fit_cluster_codes": producer.fit_cluster_codes.copy(),
        "fit_probability_oof": probability,
        "fit_fold_by_seed_query": fold,
        "producer_source_identity_sha256": np.asarray(producer.source_identity_sha256),
        "current_only_source_identity_sha256": np.asarray("a" * 64),
        "history_backbone_checkpoint_manifest_sha256": np.asarray(
            producer.checkpoint_manifest_sha256
        ),
        "checkpoint_manifest_sha256": np.asarray(manifest.manifest_sha256),
        "training_protocol": np.asarray(INDEPENDENT_CURRENT_ONLY_PROTOCOL),
        "checkpoint_namespace": np.asarray("independent_current_only"),
        "history_training_items_consumed": np.asarray(0, dtype=np.int64),
        "history_inference_items_consumed": np.asarray(0, dtype=np.int64),
        "matrix_fit_probability_oof_sha256": np.asarray(_array_sha(probability)),
        "fold_assignment_sha256": np.asarray(_array_sha(fold)),
    }


def _valid_utility_scores(producer: runner.FitOnlyProducerView) -> dict[str, np.ndarray]:
    queries = producer.fit_tasks.query_indices
    query_cluster = {
        int(query): int(cluster)
        for query, cluster in zip(
            producer.fit_query_indices, producer.fit_cluster_codes, strict=True
        )
    }
    clusters = np.asarray([query_cluster[int(query)] for query in queries], dtype=np.int64)
    fold = np.tile(clusters[None, :], (5, 1)).astype(np.int32)
    scores = np.arange(5 * len(queries), dtype=np.float64).reshape(5, len(queries)) / 100.0
    ensemble = scores.mean(axis=0)
    return {
        "schema_version": np.asarray(UTILITY_OOF_SCORE_SCHEMA),
        "dataset": np.asarray(producer.dataset),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "producer_file_sha256": np.asarray(producer.producer_file_sha256),
        "producer_source_identity_sha256": np.asarray(producer.source_identity_sha256),
        "fit_task_sha256": np.asarray(producer.fit_tasks.task_sha256),
        "fit_task_query_indices": queries.copy(),
        "fit_task_cluster_codes": clusters,
        "utility_oof_folds": np.asarray(2, dtype=np.int64),
        "fold_by_seed_task": fold,
        "decision_score_oof_by_seed": scores,
        "decision_score_oof_ensemble": ensemble,
        "matrix_decision_score_oof_by_seed_sha256": np.asarray(_array_sha(scores)),
        "matrix_decision_score_oof_ensemble_sha256": np.asarray(_array_sha(ensemble)),
        "fold_assignment_sha256": np.asarray(_array_sha(fold)),
        "feature_schema_sha256": np.asarray("b" * 64),
        "model_spec_sha256": np.asarray("c" * 64),
        "score_source_identity_sha256": np.asarray("d" * 64),
        "selection_payload_consumed": np.asarray(False),
        "labels_or_targets_serialized": np.asarray(False),
    }


def test_current_only_and_utility_oof_private_schemas_are_fail_closed(tmp_path: Path) -> None:
    producer_path = tmp_path / "producer.npz"
    _write_npz(producer_path, _producer_mapping(selection_poison=True))
    producer = load_fit_only_producer_view(producer_path)
    checkpoint_root = tmp_path / "checkpoints"
    _checkpoint_tree(checkpoint_root)
    manifest = build_checkpoint_manifest(checkpoint_root, outer_folds=2)

    current = _valid_current_fit_mapping(producer, manifest)
    validate_current_only_fit_artifact(current, producer=producer, checkpoint_manifest=manifest)
    consumed = dict(current)
    consumed["history_training_items_consumed"] = np.asarray(1, dtype=np.int64)
    with pytest.raises(StageAContractError, match="consumed history"):
        validate_current_only_fit_artifact(
            consumed, producer=producer, checkpoint_manifest=manifest
        )

    utility = _valid_utility_scores(producer)
    validate_utility_oof_score_artifact(utility, producer=producer)
    poisoned = dict(utility)
    poisoned["selection_payload_consumed"] = np.asarray(True)
    with pytest.raises(StageAContractError, match="consumed selection"):
        validate_utility_oof_score_artifact(poisoned, producer=producer)
