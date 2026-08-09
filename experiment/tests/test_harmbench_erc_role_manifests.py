from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from hva_affect.emotiontalk_role_sidecar import (  # noqa: E402
    FEATURE_SCHEMA as EMOTIONTALK_FEATURE_SCHEMA,
    FIT_ROLE,
    LABEL_SCHEMA as LEGACY_LABEL_SCHEMA,
    SELECTION_ROLE,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    HarmBenchOpenRoleError,
    compose_open_role_capabilities,
    load_emotiontalk_fit_feature_capability,
    load_emotiontalk_fit_role_capability,
    load_emotiontalk_selection_feature_capability,
    load_meld_fit_feature_capability,
    load_meld_fit_role_capability,
    load_meld_selection_feature_capability,
)
from hva_affect.harmbench_erc_role_manifests import (  # noqa: E402
    HarmBenchRoleManifestError,
    SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS,
    SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA,
    canonical_json_sha256,
    load_cross_role_feature_roster,
    load_feature_manifest,
    make_cross_role_feature_roster,
    make_feature_manifest,
    make_feature_projection,
    make_fit_training_manifest,
    read_canonical_json,
    write_canonical_json_once,
    write_emotiontalk_sanitized_fit_label_sidecar,
)
from hva_affect.meld_multimodal_sidecar import (  # noqa: E402
    SIDECAR_SCHEMA_VERSION as MELD_SIDECAR_SCHEMA,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
CLASSES = ("neutral", "happy", "sad")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_arrays(role: str, *, offset: int) -> dict[str, np.ndarray]:
    rows = 3
    bucket = 1 if role == FIT_ROLE else 66
    group_prefix = "fit" if role == FIT_ROLE else "selection"
    return {
        "schema_version": np.asarray(EMOTIONTALK_FEATURE_SCHEMA),
        "dataset_id": np.asarray("EmotionTalk"),
        "role": np.asarray(role),
        "split_protocol_id": np.asarray("scu_set_exploration_v1"),
        "row_alignment_sha256": np.asarray(SHA_A if role == FIT_ROLE else SHA_B),
        "opaque_row_hashes": np.asarray([f"row-{offset + i}" for i in range(rows)]),
        "opaque_group_hashes": np.asarray(
            [f"{group_prefix}-0", f"{group_prefix}-0", f"{group_prefix}-1"]
        ),
        "speaker_tokens": np.asarray(["speaker-a", "speaker-a", "speaker-b"]),
        "turn_ids": np.asarray([0, 1, 0], dtype=np.int64),
        "protocol_row_ids": np.arange(offset, offset + rows, dtype=np.int64),
        "role_buckets": np.asarray([bucket] * rows, dtype=np.int64),
        "texts": np.asarray(["hello", "strict past", "other"]),
        "audio_features": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        "video_features": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        "source_feature_config_sha256": np.asarray(SHA_C),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return _file_sha(path)


def _write_legacy_fit_label(path: Path, *, source_digest: str = SHA_D) -> str:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(LEGACY_LABEL_SCHEMA),
            dataset_id=np.asarray("EmotionTalk"),
            role=np.asarray(FIT_ROLE),
            split_protocol_id=np.asarray("scu_set_exploration_v1"),
            row_alignment_sha256=np.asarray(SHA_A),
            labels=np.asarray([0, 1, 2], dtype=np.int64),
            source_label_sha256=np.asarray(source_digest),
        )
    return _file_sha(path)


def _isolated_bundle(tmp_path: Path) -> dict[str, object]:
    fit_root = tmp_path / "fit_features"
    selection_root = tmp_path / "selection_features"
    target_root = tmp_path / "fit_targets"
    fit_root.mkdir()
    selection_root.mkdir()
    target_root.mkdir()
    fit_feature = fit_root / "features_base_and_utility_fit.npz"
    selection_feature = selection_root / "features_model_selection.npz"
    fit_feature_sha = _write_npz(fit_feature, _feature_arrays(FIT_ROLE, offset=0))
    selection_feature_sha = _write_npz(
        selection_feature, _feature_arrays(SELECTION_ROLE, offset=100)
    )
    fit_projection = make_feature_projection(
        dataset_id="EmotionTalk",
        role=FIT_ROLE,
        artifact_schema_version=EMOTIONTALK_FEATURE_SCHEMA,
        filename=fit_feature.name,
        sha256=fit_feature_sha,
        row_alignment_sha256=SHA_A,
        rows=3,
        independent_groups=2,
        history_eligible_rows=1,
        audio_dimension=2,
        video_dimension=3,
    )
    selection_projection = make_feature_projection(
        dataset_id="EmotionTalk",
        role=SELECTION_ROLE,
        artifact_schema_version=EMOTIONTALK_FEATURE_SCHEMA,
        filename=selection_feature.name,
        sha256=selection_feature_sha,
        row_alignment_sha256=SHA_B,
        rows=3,
        independent_groups=2,
        history_eligible_rows=1,
        audio_dimension=2,
        video_dimension=3,
    )
    roster = make_cross_role_feature_roster(
        dataset_id="EmotionTalk",
        fit_feature_projection_sha256=canonical_json_sha256(fit_projection),
        selection_feature_projection_sha256=canonical_json_sha256(selection_projection),
    )
    roster_path = tmp_path / "feature_roster.json"
    roster_sha = write_canonical_json_once(roster_path, roster)
    roster_receipt = load_cross_role_feature_roster(
        roster_path, expected_roster_sha256=roster_sha
    )
    fit_manifest_path = fit_root / "fit_feature_manifest.json"
    selection_manifest_path = selection_root / "selection_feature_manifest.json"
    fit_manifest_sha = write_canonical_json_once(
        fit_manifest_path,
        make_feature_manifest(
            feature_projection=fit_projection,
            cross_role_feature_roster_receipt=roster_receipt,
            expected_cross_role_feature_roster_sha256=roster_sha,
        ),
    )
    write_canonical_json_once(
        selection_manifest_path,
        make_feature_manifest(
            feature_projection=selection_projection,
            cross_role_feature_roster_receipt=roster_receipt,
            expected_cross_role_feature_roster_sha256=roster_sha,
        ),
    )
    legacy_label = tmp_path / "legacy_fit_labels.npz"
    legacy_sha = _write_legacy_fit_label(legacy_label)
    sanitized_label = target_root / "fit_targets.npz"
    sanitized_sha = write_emotiontalk_sanitized_fit_label_sidecar(
        source_path=legacy_label,
        destination_path=sanitized_label,
        expected_source_sha256=legacy_sha,
        expected_row_alignment_sha256=SHA_A,
    )
    training_manifest_path = target_root / "fit_training_manifest.json"
    write_canonical_json_once(
        training_manifest_path,
        make_fit_training_manifest(
            dataset_id="EmotionTalk",
            fit_feature_manifest_sha256=fit_manifest_sha,
            cross_role_feature_roster_sha256=roster_sha,
            artifact_schema_version=SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA,
            filename=sanitized_label.name,
            sha256=sanitized_sha,
            row_alignment_sha256=SHA_A,
            rows=3,
            class_names=CLASSES,
        ),
    )
    return {
        "fit_root": fit_root,
        "selection_root": selection_root,
        "target_root": target_root,
        "fit_manifest_path": fit_manifest_path,
        "selection_manifest_path": selection_manifest_path,
        "training_manifest_path": training_manifest_path,
        "roster_path": roster_path,
        "roster_sha": roster_sha,
        "roster_receipt": roster_receipt,
        "fit_projection_sha": canonical_json_sha256(fit_projection),
        "selection_projection_sha": canonical_json_sha256(selection_projection),
        "legacy_label": legacy_label,
        "sanitized_label": sanitized_label,
    }


def test_end_to_end_isolated_fit_selection_and_roster(tmp_path: Path) -> None:
    bundle = _isolated_bundle(tmp_path)
    roster = load_cross_role_feature_roster(
        bundle["roster_path"],
        expected_roster_sha256=bundle["roster_sha"],
    )
    assert roster.roster_sha256 == bundle["roster_sha"]
    assert roster.fit_feature_projection_sha256 == bundle["fit_projection_sha"]
    assert (
        roster.selection_feature_projection_sha256
        == bundle["selection_projection_sha"]
    )
    with pytest.raises(HarmBenchOpenRoleError, match="external authority"):
        load_emotiontalk_fit_feature_capability(
            capability_root=bundle["fit_root"],
            manifest_path=bundle["fit_manifest_path"],
            cross_role_feature_roster_receipt=bundle["roster_receipt"],
            expected_cross_role_feature_roster_sha256=SHA_D,
        )
    fit_features = load_emotiontalk_fit_feature_capability(
        capability_root=bundle["fit_root"],
        manifest_path=bundle["fit_manifest_path"],
        cross_role_feature_roster_receipt=bundle["roster_receipt"],
        expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
    )
    assert fit_features.any_label_archive_opened is False
    assert fit_features.cross_role_feature_roster_receipt is bundle["roster_receipt"]
    fit = load_emotiontalk_fit_role_capability(
        fit_feature_capability=fit_features,
        capability_root=bundle["target_root"],
        manifest_path=bundle["training_manifest_path"],
    )
    selection = load_emotiontalk_selection_feature_capability(
        capability_root=bundle["selection_root"],
        manifest_path=bundle["selection_manifest_path"],
        cross_role_feature_roster_receipt=bundle["roster_receipt"],
        expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
    )
    assert selection.cross_role_feature_roster_receipt is bundle["roster_receipt"]
    combined = compose_open_role_capabilities(fit, selection)
    assert combined.fit.labels.tolist() == [0, 1, 2]
    assert combined.selection.rows == 3
    assert not hasattr(selection.selection, "labels")
    assert selection.selection_label_archive_opened is False


def test_production_loader_projection_is_derived_only_from_typed_roster() -> None:
    for loader in (
        load_emotiontalk_fit_feature_capability,
        load_emotiontalk_selection_feature_capability,
        load_meld_fit_feature_capability,
        load_meld_selection_feature_capability,
    ):
        parameters = inspect.signature(loader).parameters
        assert "cross_role_feature_roster_receipt" in parameters
        assert "expected_cross_role_feature_roster_sha256" in parameters
        assert "expected_feature_projection_sha256" not in parameters


def test_typed_roster_rejects_outcome_fields_wrong_schema_and_wrong_projection(
    tmp_path: Path,
) -> None:
    bundle = _isolated_bundle(tmp_path)
    clean = read_canonical_json(bundle["roster_path"]).payload

    outcome_bearing = dict(clean)
    outcome_bearing["selection_label_sha256"] = SHA_D
    outcome_path = tmp_path / "outcome_bearing_roster.json"
    outcome_sha = write_canonical_json_once(outcome_path, outcome_bearing)
    with pytest.raises(HarmBenchRoleManifestError, match="unexpected exact schema"):
        load_cross_role_feature_roster(
            outcome_path, expected_roster_sha256=outcome_sha
        )

    wrong_schema = dict(clean)
    wrong_schema["schema_version"] = "wrong_roster_schema"
    wrong_schema_path = tmp_path / "wrong_schema_roster.json"
    wrong_schema_sha = write_canonical_json_once(wrong_schema_path, wrong_schema)
    with pytest.raises(HarmBenchRoleManifestError, match="identity changed"):
        load_cross_role_feature_roster(
            wrong_schema_path, expected_roster_sha256=wrong_schema_sha
        )

    forged_receipt = copy.copy(bundle["roster_receipt"])
    object.__setattr__(
        forged_receipt,
        "fit_feature_projection_sha256",
        SHA_D,
    )
    with pytest.raises(HarmBenchOpenRoleError, match="canonical payload"):
        load_emotiontalk_fit_feature_capability(
            capability_root=bundle["fit_root"],
            manifest_path=bundle["fit_manifest_path"],
            cross_role_feature_roster_receipt=forged_receipt,
            expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
        )

    fit_manifest = read_canonical_json(bundle["fit_manifest_path"]).payload
    wrong_projection = dict(fit_manifest["feature_projection"])
    wrong_artifact = dict(wrong_projection["artifact"])
    wrong_artifact["filename"] = "coherently_changed_features.npz"
    wrong_projection["artifact"] = wrong_artifact
    with pytest.raises(HarmBenchRoleManifestError, match="differs from typed"):
        make_feature_manifest(
            feature_projection=wrong_projection,
            cross_role_feature_roster_receipt=bundle["roster_receipt"],
            expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
        )


def test_coherent_roster_reseal_is_rejected_when_authority_is_unchanged(
    tmp_path: Path,
) -> None:
    bundle = _isolated_bundle(tmp_path)
    resealed_payload = make_cross_role_feature_roster(
        dataset_id="EmotionTalk",
        fit_feature_projection_sha256=SHA_D,
        selection_feature_projection_sha256=bundle["selection_projection_sha"],
    )
    resealed_path = tmp_path / "coherently_resealed_roster.json"
    resealed_sha = write_canonical_json_once(resealed_path, resealed_payload)
    assert resealed_sha != bundle["roster_sha"]
    with pytest.raises(HarmBenchRoleManifestError, match="external authority"):
        load_cross_role_feature_roster(
            resealed_path,
            expected_roster_sha256=bundle["roster_sha"],
        )
    resealed_receipt = load_cross_role_feature_roster(
        resealed_path, expected_roster_sha256=resealed_sha
    )
    with pytest.raises(HarmBenchOpenRoleError, match="external authority"):
        load_emotiontalk_fit_feature_capability(
            capability_root=bundle["fit_root"],
            manifest_path=bundle["fit_manifest_path"],
            cross_role_feature_roster_receipt=resealed_receipt,
            expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
        )


def test_fit_feature_and_selection_loaders_never_open_label_named_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _isolated_bundle(tmp_path)
    original = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if "label" in path.name.lower() or "target" in path.name.lower():
            raise AssertionError(f"feature loader touched target path: {path.name}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    load_emotiontalk_fit_feature_capability(
        capability_root=bundle["fit_root"],
        manifest_path=bundle["fit_manifest_path"],
        cross_role_feature_roster_receipt=bundle["roster_receipt"],
        expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
    )
    load_emotiontalk_selection_feature_capability(
        capability_root=bundle["selection_root"],
        manifest_path=bundle["selection_manifest_path"],
        cross_role_feature_roster_receipt=bundle["roster_receipt"],
        expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
    )


def test_selection_capability_is_invariant_to_legacy_outcome_mutation(
    tmp_path: Path,
) -> None:
    bundle = _isolated_bundle(tmp_path)
    first = load_emotiontalk_selection_feature_capability(
        capability_root=bundle["selection_root"],
        manifest_path=bundle["selection_manifest_path"],
        cross_role_feature_roster_receipt=bundle["roster_receipt"],
        expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
    )
    legacy_label = bundle["legacy_label"]
    _write_legacy_fit_label(legacy_label, source_digest=SHA_C)
    aggregate = tmp_path / "legacy_aggregate_manifest.json"
    aggregate.write_text(
        json.dumps(
            {
                "model_selection_label_sha256": SHA_A,
                "calibration": {"label_sha256": SHA_B},
            }
        ),
        encoding="utf-8",
    )
    aggregate.write_text(
        json.dumps(
            {
                "model_selection_label_sha256": SHA_D,
                "internal_holdout": {"label_sha256": SHA_C},
            }
        ),
        encoding="utf-8",
    )
    second = load_emotiontalk_selection_feature_capability(
        capability_root=bundle["selection_root"],
        manifest_path=bundle["selection_manifest_path"],
        cross_role_feature_roster_receipt=bundle["roster_receipt"],
        expected_cross_role_feature_roster_sha256=bundle["roster_sha"],
    )
    assert first.capability_sha256 == second.capability_sha256
    assert first.manifest_sha256 == second.manifest_sha256


def test_selection_manifest_has_no_forbidden_target_vocabulary(tmp_path: Path) -> None:
    bundle = _isolated_bundle(tmp_path)
    verified = read_canonical_json(bundle["selection_manifest_path"])
    serialized = json.dumps(verified.payload, sort_keys=True).lower()
    for token in ("label", "outcome", "calibration", "holdout", "validation", "test"):
        assert token not in serialized

    poisoned = dict(verified.payload)
    projection = dict(poisoned["feature_projection"])
    artifact = dict(projection["artifact"])
    artifact["filename"] = "labels_model_selection.npz"
    projection["artifact"] = artifact
    poisoned["feature_projection"] = projection
    poisoned["feature_projection_sha256"] = canonical_json_sha256(projection)
    poisoned_path = tmp_path / "poisoned_selection.json"
    write_canonical_json_once(poisoned_path, poisoned)
    with pytest.raises(HarmBenchRoleManifestError, match="forbidden vocabulary"):
        load_feature_manifest(
            poisoned_path,
            expected_dataset="EmotionTalk",
            expected_role=SELECTION_ROLE,
        )


def test_manifest_path_traps_duplicate_keys_and_noncanonical_json_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(HarmBenchRoleManifestError, match="basename"):
        make_feature_projection(
            dataset_id="EmotionTalk",
            role=FIT_ROLE,
            artifact_schema_version=EMOTIONTALK_FEATURE_SCHEMA,
            filename="../fit.npz",
            sha256=SHA_A,
            row_alignment_sha256=SHA_B,
            rows=1,
            independent_groups=1,
            history_eligible_rows=0,
            audio_dimension=1,
            video_dimension=1,
        )
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(HarmBenchRoleManifestError, match="duplicate JSON key"):
        read_canonical_json(duplicate)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{ "a": 1 }\n')
    with pytest.raises(HarmBenchRoleManifestError, match="canonical JSON"):
        read_canonical_json(noncanonical)


def test_emotiontalk_migration_removes_all_role_source_digest_and_is_write_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.npz"
    source_sha = _write_legacy_fit_label(source)
    destination = tmp_path / "sanitized.npz"
    digest = write_emotiontalk_sanitized_fit_label_sidecar(
        source_path=source,
        destination_path=destination,
        expected_source_sha256=source_sha,
        expected_row_alignment_sha256=SHA_A,
    )
    assert digest == _file_sha(destination)
    with np.load(destination, allow_pickle=False) as archive:
        assert set(archive.files) == SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS
        assert "source_label_sha256" not in archive.files
        assert str(np.asarray(archive["schema_version"]).reshape(-1)[0]) == (
            SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA
        )
    with pytest.raises(HarmBenchRoleManifestError, match="already exists"):
        write_emotiontalk_sanitized_fit_label_sidecar(
            source_path=source,
            destination_path=destination,
            expected_source_sha256=source_sha,
            expected_row_alignment_sha256=SHA_A,
        )


def test_symlink_manifest_is_rejected_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    write_canonical_json_once(target, {"a": 1})
    link = tmp_path / "link.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(HarmBenchRoleManifestError, match="symlink or reparse"):
        read_canonical_json(link)


def test_manifest_same_handle_repeated_read_detects_in_place_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"a":1}')
    original_open = Path.open

    class MutatingHandle:
        def __init__(self, handle: object) -> None:
            self.handle = handle
            self.first = True

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def seek(self, *args: object) -> int:
            return self.handle.seek(*args)

        def read(self, *args: object) -> bytes:
            data = self.handle.read(*args)
            if self.first:
                self.first = False
                self.handle.seek(0)
                self.handle.write(b'{"b":2}')
                self.handle.flush()
                os.fsync(self.handle.fileno())
                self.handle.seek(0)
            return data

    def mutating_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == manifest and mode == "rb":
            return MutatingHandle(original_open(path, "r+b"))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutating_open)
    with pytest.raises(HarmBenchRoleManifestError, match="bytes changed"):
        read_canonical_json(manifest)


def test_meld_uses_the_same_outcome_isolated_split_loader_contract(
    tmp_path: Path,
) -> None:
    fit_root = tmp_path / "meld_fit_features"
    selection_root = tmp_path / "meld_selection_features"
    target_root = tmp_path / "meld_fit_targets"
    fit_root.mkdir()
    selection_root.mkdir()
    target_root.mkdir()

    def meld_feature_arrays(role: str, offset: int) -> dict[str, np.ndarray]:
        prefix = 10 if role == FIT_ROLE else 20
        return {
            "schema_version": np.asarray(MELD_SIDECAR_SCHEMA),
            "role": np.asarray(role),
            "row_alignment_sha256": np.asarray(
                SHA_A if role == FIT_ROLE else SHA_B
            ),
            "utterances": np.asarray(["a", "b", "c"]),
            "audio_mean_std": np.arange(6, dtype=np.float32).reshape(3, 2),
            "video_mean_std": np.arange(9, dtype=np.float32).reshape(3, 3),
            "dialogue_codes": np.asarray([prefix, prefix, prefix + 1]),
            "speaker_codes": np.asarray([1, 1, 2]),
            "utterance_order": np.asarray([0, 1, 0]),
            "protocol_row_ids": np.arange(offset, offset + 3, dtype=np.int64),
        }

    fit_feature_path = fit_root / "features_base_and_utility_fit.npz"
    selection_feature_path = selection_root / "features_model_selection.npz"
    fit_feature_sha = _write_npz(
        fit_feature_path, meld_feature_arrays(FIT_ROLE, 0)
    )
    selection_feature_sha = _write_npz(
        selection_feature_path, meld_feature_arrays(SELECTION_ROLE, 100)
    )
    fit_projection = make_feature_projection(
        dataset_id="MELD",
        role=FIT_ROLE,
        artifact_schema_version=MELD_SIDECAR_SCHEMA,
        filename=fit_feature_path.name,
        sha256=fit_feature_sha,
        row_alignment_sha256=SHA_A,
        rows=3,
        independent_groups=2,
        history_eligible_rows=1,
        audio_dimension=2,
        video_dimension=3,
    )
    selection_projection = make_feature_projection(
        dataset_id="MELD",
        role=SELECTION_ROLE,
        artifact_schema_version=MELD_SIDECAR_SCHEMA,
        filename=selection_feature_path.name,
        sha256=selection_feature_sha,
        row_alignment_sha256=SHA_B,
        rows=3,
        independent_groups=2,
        history_eligible_rows=1,
        audio_dimension=2,
        video_dimension=3,
    )
    roster_path = tmp_path / "meld_feature_roster.json"
    roster_sha = write_canonical_json_once(
        roster_path,
        make_cross_role_feature_roster(
            dataset_id="MELD",
            fit_feature_projection_sha256=canonical_json_sha256(fit_projection),
            selection_feature_projection_sha256=canonical_json_sha256(
                selection_projection
            ),
        ),
    )
    roster_receipt = load_cross_role_feature_roster(
        roster_path, expected_roster_sha256=roster_sha
    )
    fit_manifest_path = fit_root / "fit_feature_manifest.json"
    selection_manifest_path = selection_root / "selection_feature_manifest.json"
    fit_manifest_sha = write_canonical_json_once(
        fit_manifest_path,
        make_feature_manifest(
            feature_projection=fit_projection,
            cross_role_feature_roster_receipt=roster_receipt,
            expected_cross_role_feature_roster_sha256=roster_sha,
        ),
    )
    write_canonical_json_once(
        selection_manifest_path,
        make_feature_manifest(
            feature_projection=selection_projection,
            cross_role_feature_roster_receipt=roster_receipt,
            expected_cross_role_feature_roster_sha256=roster_sha,
        ),
    )
    fit_target_path = target_root / "fit_targets.npz"
    fit_target_sha = _write_npz(
        fit_target_path,
        {
            "schema_version": np.asarray(MELD_SIDECAR_SCHEMA),
            "role": np.asarray(FIT_ROLE),
            "row_alignment_sha256": np.asarray(SHA_A),
            "labels": np.asarray([0, 1, 2], dtype=np.int64),
        },
    )
    training_manifest_path = target_root / "fit_training_manifest.json"
    write_canonical_json_once(
        training_manifest_path,
        make_fit_training_manifest(
            dataset_id="MELD",
            fit_feature_manifest_sha256=fit_manifest_sha,
            cross_role_feature_roster_sha256=roster_sha,
            artifact_schema_version=MELD_SIDECAR_SCHEMA,
            filename=fit_target_path.name,
            sha256=fit_target_sha,
            row_alignment_sha256=SHA_A,
            rows=3,
            class_names=CLASSES,
        ),
    )
    fit_features = load_meld_fit_feature_capability(
        capability_root=fit_root,
        manifest_path=fit_manifest_path,
        cross_role_feature_roster_receipt=roster_receipt,
        expected_cross_role_feature_roster_sha256=roster_sha,
    )
    fit = load_meld_fit_role_capability(
        fit_feature_capability=fit_features,
        capability_root=target_root,
        manifest_path=training_manifest_path,
    )
    selection = load_meld_selection_feature_capability(
        capability_root=selection_root,
        manifest_path=selection_manifest_path,
        cross_role_feature_roster_receipt=roster_receipt,
        expected_cross_role_feature_roster_sha256=roster_sha,
    )
    combined = compose_open_role_capabilities(fit, selection)
    assert combined.dataset_id == "MELD"
    assert combined.fit.labels.tolist() == [0, 1, 2]
    assert not hasattr(selection.selection, "labels")
