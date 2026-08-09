from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from threading import Lock
from typing import Any, Callable
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hva_affect.harmbench_erc_selection_label_curator as curator  # noqa: E402
import hva_affect.harmbench_erc_selection_labels as label_module  # noqa: E402
from hva_affect.harmbench_erc_selection_label_curator import (  # noqa: E402
    CURATOR_ATTEMPT_FILENAME,
    CuratedSelectionLabelReceipt,
    HarmBenchSelectionLabelCuratorError,
    UNIVERSAL_ARTIFACT_FILENAME,
    UNIVERSAL_MANIFEST_FILENAME,
    curate_frozen_legacy_selection_labels,
    frozen_selection_class_order_sha256,
)


FIT_SHA = hashlib.sha256(b"synthetic-fit-training-capability").hexdigest()
UNIVERSAL_MEMBERS = (
    "schema_version",
    "dataset_id",
    "role",
    "rows",
    "ordered_protocol_row_alignment_sha256",
    "class_order_sha256",
    "labels",
    "protocol_row_ids",
    "class_tokens",
)


@dataclass
class SyntheticBundle:
    dataset_id: str
    base_contract: Any
    contract: Any
    sidecar_root: Path
    external_manifest_path: Path
    output_root: Path
    feature_path: Path
    label_path: Path
    protocol_row_ids: np.ndarray
    labels: np.ndarray


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("w+b") as handle:
        np.savez_compressed(handle, **arrays)


def _feature_arrays(
    contract: Any, protocol: np.ndarray
) -> dict[str, np.ndarray]:
    rows = len(protocol)
    if contract.dataset_id == "EmotionTalk":
        return {
            "schema_version": np.asarray(contract.legacy_feature_schema),
            "dataset_id": np.asarray("EmotionTalk"),
            "role": np.asarray("model_selection"),
            "split_protocol_id": np.asarray("scu_set_exploration_v1"),
            "row_alignment_sha256": np.asarray(
                contract.legacy_row_alignment_sha256
            ),
            "opaque_row_hashes": np.asarray([f"row-{i}" for i in range(rows)]),
            "opaque_group_hashes": np.asarray(
                [f"group-{i // 2}" for i in range(rows)]
            ),
            "speaker_tokens": np.asarray([f"s{i % 2}" for i in range(rows)]),
            "turn_ids": np.arange(rows, dtype=np.int64),
            "protocol_row_ids": protocol.copy(),
            "role_buckets": np.full(rows, 70, dtype=np.int16),
            "texts": np.asarray([f"synthetic-{i}" for i in range(rows)]),
            "audio_features": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
            "video_features": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
            "source_feature_config_sha256": np.asarray("a" * 64),
        }
    return {
        "schema_version": np.asarray([contract.legacy_feature_schema]),
        "role": np.asarray(["model_selection"]),
        "row_alignment_sha256": np.asarray(
            [contract.legacy_row_alignment_sha256]
        ),
        "utterances": np.asarray([f"synthetic-{i}" for i in range(rows)]),
        "audio_mean_std": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        "video_mean_std": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        "dialogue_codes": np.asarray([i // 2 for i in range(rows)], dtype=np.int64),
        "speaker_codes": np.asarray([i % 2 for i in range(rows)], dtype=np.int64),
        "utterance_order": np.arange(rows, dtype=np.int64),
        "protocol_row_ids": protocol.copy(),
    }


def _label_arrays(contract: Any, labels: np.ndarray) -> dict[str, np.ndarray]:
    if contract.dataset_id == "EmotionTalk":
        return {
            "schema_version": np.asarray(contract.legacy_label_schema),
            "dataset_id": np.asarray("EmotionTalk"),
            "role": np.asarray("model_selection"),
            "split_protocol_id": np.asarray("scu_set_exploration_v1"),
            "row_alignment_sha256": np.asarray(
                contract.legacy_row_alignment_sha256
            ),
            "labels": labels.copy(),
            "source_label_sha256": np.asarray("b" * 64),
        }
    return {
        "schema_version": np.asarray([contract.legacy_label_schema]),
        "role": np.asarray(["model_selection"]),
        "row_alignment_sha256": np.asarray(
            [contract.legacy_row_alignment_sha256]
        ),
        "labels": labels.copy(),
    }


def _manifest_payload(contract: Any, *, feature_sha: str, label_sha: str) -> dict:
    return {
        "schema_version": contract.external_manifest_schema,
        "protocol": contract.external_manifest_protocol,
        "status": contract.external_manifest_status,
        "dataset_id": contract.dataset_id,
        "split_protocol_id": "scu_set_exploration_v1",
        "label_order": list(contract.ordered_class_tokens),
        "roles": {
            "model_selection": {
                "feature_filename": contract.selection_feature_filename,
                "label_filename": contract.selection_label_filename,
                "rows": contract.rows,
                "feature_sha256": feature_sha,
                "label_sha256": label_sha,
                "row_alignment_sha256": contract.legacy_row_alignment_sha256,
            }
        },
        "synthetic_contract_only": True,
    }


def _refreeze_bundle(
    bundle: SyntheticBundle,
    monkeypatch: pytest.MonkeyPatch,
    *,
    external_raw_transform: Callable[[bytes], bytes] | None = None,
) -> None:
    feature_sha = _file_sha(bundle.feature_path)
    label_sha = _file_sha(bundle.label_path)
    contract = replace(
        bundle.base_contract,
        rows=len(bundle.protocol_row_ids),
        selection_feature_sha256=feature_sha,
        selection_label_sha256=label_sha,
    )
    payload = _manifest_payload(contract, feature_sha=feature_sha, label_sha=label_sha)
    raw = (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    if external_raw_transform is not None:
        raw = external_raw_transform(raw)
    bundle.external_manifest_path.write_bytes(raw)
    contract = replace(
        contract,
        external_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setitem(curator._FROZEN_DATASETS, bundle.dataset_id, contract)
    bundle.contract = contract


def _make_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    *,
    feature_mutator: Callable[[dict[str, np.ndarray]], None] | None = None,
    label_mutator: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> SyntheticBundle:
    base = curator._FROZEN_DATASETS[dataset_id]
    case_root = tmp_path / dataset_id.lower()
    sidecar_root = case_root / "legacy"
    output_root = case_root / "private-output"
    public_root = case_root / "public"
    sidecar_root.mkdir(parents=True)
    output_root.mkdir()
    public_root.mkdir()
    rows = 7 if dataset_id == "EmotionTalk" else 6
    protocol = np.asarray(
        [701 + 3 * index for index in range(rows)], dtype=np.int64
    )
    labels = np.asarray(
        [index % len(base.ordered_class_tokens) for index in range(rows)],
        dtype=np.int64,
    )
    feature_values = _feature_arrays(base, protocol)
    label_values = _label_arrays(base, labels)
    if feature_mutator is not None:
        feature_mutator(feature_values)
    if label_mutator is not None:
        label_mutator(label_values)
    feature_path = sidecar_root / base.selection_feature_filename
    label_path = sidecar_root / base.selection_label_filename
    _write_npz(feature_path, feature_values)
    _write_npz(label_path, label_values)
    bundle = SyntheticBundle(
        dataset_id=dataset_id,
        base_contract=base,
        contract=base,
        sidecar_root=sidecar_root.resolve(),
        external_manifest_path=(
            public_root / base.external_manifest_filename
        ).resolve(),
        output_root=output_root.resolve(),
        feature_path=feature_path.resolve(),
        label_path=label_path.resolve(),
        protocol_row_ids=protocol,
        labels=labels,
    )
    _refreeze_bundle(bundle, monkeypatch)
    monkeypatch.setattr(curator, "_repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(curator, "_home_root", lambda: tmp_path / "home")
    monkeypatch.setattr(label_module, "_repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(label_module, "_home_root", lambda: tmp_path / "home")
    return bundle


def _curate(bundle: SyntheticBundle) -> CuratedSelectionLabelReceipt:
    return curate_frozen_legacy_selection_labels(
        dataset_id=bundle.dataset_id,
        legacy_sidecar_root=bundle.sidecar_root,
        external_manifest_path=bundle.external_manifest_path,
        private_output_root=bundle.output_root,
        expected_fit_training_capability_sha256=FIT_SHA,
    )


def test_runtime_import_surface_is_stdlib_plus_numpy_only() -> None:
    source_path = ROOT / "src" / "hva_affect" / "harmbench_erc_selection_label_curator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.append(node.module or "")
    allowed = {
        "__future__",
        "ctypes",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "stat",
        "tempfile",
        "typing",
        "zipfile",
        "numpy",
    }
    assert set(imported) <= allowed
    forbidden = {
        "prediction",
        "model",
        "metric",
        "evaluator",
        "open_roles",
        "role_manifests",
        "selection_labels",
        "checkpoint",
    }
    assert not any(fragment in name for name in imported for fragment in forbidden)


@pytest.mark.parametrize("dataset_id", ("EmotionTalk", "MELD"))
def test_both_legacy_schemas_publish_exact_loader_compatible_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset_id: str
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, dataset_id)
    receipt = _curate(bundle)
    assert isinstance(receipt, CuratedSelectionLabelReceipt)
    assert receipt.dataset_id == dataset_id
    assert receipt.rows == len(bundle.protocol_row_ids)
    assert not any(isinstance(getattr(receipt, field.name), np.ndarray) for field in fields(receipt))
    assert sorted(path.name for path in bundle.output_root.iterdir()) == sorted(
        [
            CURATOR_ATTEMPT_FILENAME,
            UNIVERSAL_ARTIFACT_FILENAME,
            UNIVERSAL_MANIFEST_FILENAME,
        ]
    )

    manifest_path = bundle.output_root / UNIVERSAL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.read_bytes() == curator._canonical_json_bytes(manifest)
    assert receipt.manifest_file_sha256 == _file_sha(manifest_path)
    artifact_path = bundle.output_root / UNIVERSAL_ARTIFACT_FILENAME
    assert receipt.artifact_file_sha256 == _file_sha(artifact_path)
    with np.load(artifact_path, allow_pickle=False) as archive:
        assert tuple(archive.files) == UNIVERSAL_MEMBERS
        assert np.array_equal(archive["labels"], bundle.labels)
        assert np.array_equal(archive["protocol_row_ids"], bundle.protocol_row_ids)
        assert tuple(archive["class_tokens"].tolist()) == (
            bundle.contract.ordered_class_tokens
        )

    metadata = label_module.load_selection_label_manifest_metadata(
        private_root=bundle.output_root,
        manifest_path=manifest_path,
        expected_manifest_sha256=receipt.manifest_file_sha256,
    )
    label_path, label_identity = label_module._validated_existing_label_file(
        metadata._origin.private_root, metadata.artifact_filename
    )
    loaded_arrays = label_module._load_label_npz_once(
        label_path,
        expected_identity=label_identity,
        expected_artifact_sha256=metadata.artifact_file_sha256,
    )
    loaded_labels, loaded_protocol, loaded_tokens = label_module._validate_loaded_arrays(
        loaded_arrays,
        metadata,
        expected_protocol_row_ids=bundle.protocol_row_ids,
        expected_class_tokens=np.asarray(
            bundle.contract.ordered_class_tokens, dtype=np.str_
        ),
    )
    assert np.array_equal(loaded_labels, bundle.labels)
    assert np.array_equal(loaded_protocol, bundle.protocol_row_ids)
    assert tuple(loaded_tokens.tolist()) == bundle.contract.ordered_class_tokens


@pytest.mark.parametrize("dataset_id", ("EmotionTalk", "MELD"))
def test_class_order_hash_is_mechanical_checkpoint_contract(
    dataset_id: str,
) -> None:
    contract = curator._FROZEN_DATASETS[dataset_id]
    expected = hashlib.sha256(
        json.dumps(
            {
                "schema_version": "harmbench_erc_checkpoint_class_order_v1",
                "dataset_id": dataset_id,
                "fit_training_capability_sha256": FIT_SHA,
                "ordered_class_tokens": list(contract.ordered_class_tokens),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert frozen_selection_class_order_sha256(
        dataset_id=dataset_id,
        expected_fit_training_capability_sha256=FIT_SHA,
    ) == expected
    assert tuple(inspect.signature(frozen_selection_class_order_sha256).parameters) == (
        "dataset_id",
        "expected_fit_training_capability_sha256",
    )


def test_only_model_selection_sidecars_are_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    forbidden_names = {
        "features_base_and_utility_fit.npz",
        "labels_base_and_utility_fit.npz",
        "features_calibration.npz",
        "labels_calibration.npz",
        "features_internal_holdout.npz",
        "labels_internal_holdout.npz",
    }
    for name in forbidden_names:
        (bundle.sidecar_root / name).write_bytes(b"must-never-be-touched")
    touched: list[str] = []
    original_open = Path.open
    original_lstat = Path.lstat
    original_resolve = Path.resolve
    original_os_lstat = os.lstat

    def trap(path: object) -> None:
        if Path(path).name in forbidden_names:
            touched.append(str(path))
            raise AssertionError("curator touched a non-selection sidecar")

    def trapped_open(path: Path, *args: object, **kwargs: object) -> Any:
        trap(path)
        return original_open(path, *args, **kwargs)

    def trapped_lstat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        trap(path)
        return original_lstat(path, *args, **kwargs)

    def trapped_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        trap(path)
        return original_resolve(path, *args, **kwargs)

    def trapped_os_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        trap(path)
        return original_os_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", trapped_open)
    monkeypatch.setattr(Path, "lstat", trapped_lstat)
    monkeypatch.setattr(Path, "resolve", trapped_resolve)
    monkeypatch.setattr(os, "lstat", trapped_os_lstat)
    _curate(bundle)
    assert touched == []


@pytest.mark.parametrize("attack", ("duplicate", "nan", "wrong_hash"))
def test_external_manifest_duplicate_nan_and_hash_attacks_fail_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "EmotionTalk")
    raw = bundle.external_manifest_path.read_bytes()
    if attack == "duplicate":
        attacked = b'{"dataset_id":"duplicate",' + raw[1:]
        bundle.external_manifest_path.write_bytes(attacked)
        bundle.contract = replace(
            bundle.contract,
            external_manifest_sha256=hashlib.sha256(attacked).hexdigest(),
        )
        monkeypatch.setitem(curator._FROZEN_DATASETS, bundle.dataset_id, bundle.contract)
        match = "duplicate JSON key"
    elif attack == "nan":
        attacked = b'{"forbidden":NaN,' + raw[1:]
        bundle.external_manifest_path.write_bytes(attacked)
        bundle.contract = replace(
            bundle.contract,
            external_manifest_sha256=hashlib.sha256(attacked).hexdigest(),
        )
        monkeypatch.setitem(curator._FROZEN_DATASETS, bundle.dataset_id, bundle.contract)
        match = "invalid JSON constant"
    else:
        bundle.external_manifest_path.write_bytes(raw + b" ")
        match = "SHA-256"
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match=match):
        _curate(bundle)
    assert list(bundle.output_root.iterdir()) == []


def test_feature_mutation_fails_before_marker_and_label_is_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    with bundle.feature_path.open("ab") as handle:
        handle.write(b"mutation")
    label_opens = 0
    original_open = Path.open

    def tracking_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal label_opens
        if path == bundle.label_path:
            label_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="feature NPZ SHA-256"):
        _curate(bundle)
    assert label_opens == 0
    assert list(bundle.output_root.iterdir()) == []


def test_label_mutation_is_terminal_after_attempt_and_cannot_silently_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    with bundle.label_path.open("ab") as handle:
        handle.write(b"mutation")
    label_opens = 0
    original_open = Path.open

    def tracking_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal label_opens
        if path == bundle.label_path and args and args[0] == "rb":
            label_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="label NPZ SHA-256"):
        _curate(bundle)
    assert label_opens == 1
    assert [path.name for path in bundle.output_root.iterdir()] == [
        CURATOR_ATTEMPT_FILENAME
    ]
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="prior attempt"):
        _curate(bundle)
    assert label_opens == 1


def test_byte_identical_path_replacement_is_detected_by_same_handle_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "EmotionTalk")
    feature_bytes = bundle.feature_path.read_bytes()
    original_open = Path.open
    replaced = False

    class ReplacingHandle:
        def __init__(self, handle: Any, path: Path) -> None:
            self._handle = handle
            self._path = path

        def __enter__(self) -> "ReplacingHandle":
            self._handle.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            nonlocal replaced
            result = self._handle.__exit__(*args)
            if not replaced:
                replaced = True
                os.replace(self._path, self._path.with_suffix(".displaced"))
                with original_open(self._path, "wb") as replacement:
                    replacement.write(feature_bytes)
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def replacing_open(path: Path, *args: object, **kwargs: object) -> Any:
        handle = original_open(path, *args, **kwargs)
        if path == bundle.feature_path and args and args[0] == "rb":
            return ReplacingHandle(handle, path)
        return handle

    monkeypatch.setattr(Path, "open", replacing_open)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="path changed identity"):
        _curate(bundle)
    assert replaced
    assert list(bundle.output_root.iterdir()) == []


def test_concurrent_curators_have_one_winner_and_one_label_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    label_opens = 0
    counter_lock = Lock()
    original_open = Path.open

    def counting_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal label_opens
        if path == bundle.label_path and args and args[0] == "rb":
            with counter_lock:
                label_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    successes: list[CuratedSelectionLabelReceipt] = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_curate, bundle), executor.submit(_curate, bundle)]
        for future in futures:
            try:
                successes.append(future.result())
            except BaseException as error:
                errors.append(error)
    assert len(successes) == 1
    assert len(errors) == 1
    assert label_opens == 1
    assert not any(path.suffix == ".tmp" for path in bundle.output_root.iterdir())


def test_manifest_publish_crash_leaves_terminal_orphan_not_rerunnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "EmotionTalk")
    original_publish = curator._publish_once

    def crash_manifest(temporary: Path, destination: Path) -> None:
        if destination.name == UNIVERSAL_MANIFEST_FILENAME:
            raise RuntimeError("synthetic crash before manifest publication")
        original_publish(temporary, destination)

    monkeypatch.setattr(curator, "_publish_once", crash_manifest)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _curate(bundle)
    names = {path.name for path in bundle.output_root.iterdir()}
    assert names == {CURATOR_ATTEMPT_FILENAME, UNIVERSAL_ARTIFACT_FILENAME}
    monkeypatch.setattr(curator, "_publish_once", original_publish)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="prior attempt"):
        _curate(bundle)
    assert not (bundle.output_root / UNIVERSAL_MANIFEST_FILENAME).exists()


def test_publication_and_directory_barrier_order_precedes_label_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    events: list[str] = []
    original_publish = curator._publish_once
    original_sync = curator._sync_directory
    original_selection_child = curator._selection_child

    def publishing(temporary: Path, destination: Path) -> None:
        events.append(f"publish:{destination.name}")
        original_publish(temporary, destination)

    def barrier(root: Path) -> None:
        events.append("directory_barrier")
        original_sync(root)

    def selection_child(
        root: Path, filename: str, *, name: str
    ) -> tuple[Path, os.stat_result]:
        if filename == bundle.contract.selection_label_filename:
            events.append("label_path_operation")
        return original_selection_child(root, filename, name=name)

    monkeypatch.setattr(curator, "_publish_once", publishing)
    monkeypatch.setattr(curator, "_sync_directory", barrier)
    monkeypatch.setattr(curator, "_selection_child", selection_child)
    _curate(bundle)
    assert events == [
        f"publish:{CURATOR_ATTEMPT_FILENAME}",
        "directory_barrier",
        "label_path_operation",
        f"publish:{UNIVERSAL_ARTIFACT_FILENAME}",
        "directory_barrier",
        f"publish:{UNIVERSAL_MANIFEST_FILENAME}",
        "directory_barrier",
    ]


def test_marker_barrier_failure_is_terminal_before_any_label_path_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "EmotionTalk")
    label_path_operations = 0
    original_selection_child = curator._selection_child

    def selection_child(
        root: Path, filename: str, *, name: str
    ) -> tuple[Path, os.stat_result]:
        nonlocal label_path_operations
        if filename == bundle.contract.selection_label_filename:
            label_path_operations += 1
        return original_selection_child(root, filename, name=name)

    def failed_barrier(_root: Path) -> None:
        raise OSError("synthetic marker directory barrier failure")

    monkeypatch.setattr(curator, "_selection_child", selection_child)
    monkeypatch.setattr(curator, "_sync_directory", failed_barrier)
    with pytest.raises(OSError, match="marker directory barrier failure"):
        _curate(bundle)
    assert label_path_operations == 0
    assert {path.name for path in bundle.output_root.iterdir()} == {
        CURATOR_ATTEMPT_FILENAME
    }
    monkeypatch.setattr(curator, "_sync_directory", lambda _root: None)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="prior attempt"):
        _curate(bundle)
    assert label_path_operations == 0


def test_manifest_directory_barrier_failure_keeps_all_publications_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    original_sync = curator._sync_directory
    barrier_count = 0

    def fail_third_barrier(root: Path) -> None:
        nonlocal barrier_count
        barrier_count += 1
        if barrier_count == 3:
            raise OSError("synthetic manifest directory barrier failure")
        original_sync(root)

    monkeypatch.setattr(curator, "_sync_directory", fail_third_barrier)
    with pytest.raises(OSError, match="manifest directory barrier failure"):
        _curate(bundle)
    assert barrier_count == 3
    assert {path.name for path in bundle.output_root.iterdir()} == {
        CURATOR_ATTEMPT_FILENAME,
        UNIVERSAL_ARTIFACT_FILENAME,
        UNIVERSAL_MANIFEST_FILENAME,
    }
    assert not any(path.suffix == ".tmp" for path in bundle.output_root.iterdir())
    monkeypatch.setattr(curator, "_sync_directory", original_sync)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="prior attempt"):
        _curate(bundle)


def test_artifact_directory_barrier_failure_keeps_terminal_orphan_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "EmotionTalk")
    original_sync = curator._sync_directory
    barrier_count = 0

    def fail_second_barrier(root: Path) -> None:
        nonlocal barrier_count
        barrier_count += 1
        if barrier_count == 2:
            raise OSError("synthetic artifact directory barrier failure")
        original_sync(root)

    monkeypatch.setattr(curator, "_sync_directory", fail_second_barrier)
    with pytest.raises(OSError, match="artifact directory barrier failure"):
        _curate(bundle)
    assert barrier_count == 2
    assert {path.name for path in bundle.output_root.iterdir()} == {
        CURATOR_ATTEMPT_FILENAME,
        UNIVERSAL_ARTIFACT_FILENAME,
    }
    monkeypatch.setattr(curator, "_sync_directory", original_sync)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="prior attempt"):
        _curate(bundle)


def test_posix_directory_barrier_opens_fsyncs_and_closes_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []

    def fake_open(path: object, flags: int) -> int:
        events.append(("open", (path, flags)))
        return 91

    def fake_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))

    def fake_close(descriptor: int) -> None:
        events.append(("close", descriptor))

    monkeypatch.setattr(curator.os, "name", "posix")
    monkeypatch.setattr(curator.os, "open", fake_open)
    monkeypatch.setattr(curator.os, "fsync", fake_fsync)
    monkeypatch.setattr(curator.os, "close", fake_close)
    curator._sync_directory(tmp_path)
    assert [event[0] for event in events] == ["open", "fsync", "close"]
    assert events[1:] == [("fsync", 91), ("close", 91)]


@pytest.mark.parametrize(
    ("target", "mutation", "message", "marker_expected"),
    (
        (
            "feature",
            lambda arrays: arrays.__setitem__(
                "protocol_row_ids",
                np.asarray([7, 7, 9, 10, 11, 12], dtype=np.int64),
            ),
            "nonnegative and unique",
            False,
        ),
        (
            "label",
            lambda arrays: arrays.__setitem__(
                "labels", np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float64)
            ),
            "exact int64",
            True,
        ),
        (
            "label",
            lambda arrays: arrays.__setitem__(
                "labels", np.asarray([0, 1, 2, 3, 4, 99], dtype=np.int64)
            ),
            "outside frozen class order",
            True,
        ),
    ),
)
def test_dtype_unique_shape_and_class_range_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation: Callable[[dict[str, np.ndarray]], None],
    message: str,
    marker_expected: bool,
) -> None:
    bundle = _make_bundle(
        tmp_path,
        monkeypatch,
        "MELD",
        feature_mutator=mutation if target == "feature" else None,
        label_mutator=mutation if target == "label" else None,
    )
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match=message):
        _curate(bundle)
    assert (bundle.output_root / CURATOR_ATTEMPT_FILENAME).exists() is marker_expected


def test_duplicate_zip_member_object_dtype_and_element_budget_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    with zipfile.ZipFile(bundle.feature_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        raw = archive.read("protocol_row_ids.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("protocol_row_ids.npy", raw)
    _refreeze_bundle(bundle, monkeypatch)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="exact ordered"):
        _curate(bundle)
    assert list(bundle.output_root.iterdir()) == []

    second = _make_bundle(tmp_path / "object", monkeypatch, "EmotionTalk")
    arrays = _feature_arrays(second.base_contract, second.protocol_row_ids)
    arrays["texts"] = np.asarray([object()] * len(second.protocol_row_ids), dtype=object)
    _write_npz(second.feature_path, arrays)
    _refreeze_bundle(second, monkeypatch)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="unsafe dtype"):
        _curate(second)
    assert list(second.output_root.iterdir()) == []

    third = _make_bundle(tmp_path / "budget", monkeypatch, "MELD")
    monkeypatch.setattr(curator, "MAX_LEGACY_ELEMENTS", 5)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="element or byte budget"):
        _curate(third)
    assert list(third.output_root.iterdir()) == []


def test_paths_roots_symlinks_and_write_once_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "EmotionTalk")
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="explicit absolute"):
        curate_frozen_legacy_selection_labels(
            dataset_id="EmotionTalk",
            legacy_sidecar_root=Path("relative"),
            external_manifest_path=bundle.external_manifest_path,
            private_output_root=bundle.output_root,
            expected_fit_training_capability_sha256=FIT_SHA,
        )
    wrong_manifest = bundle.external_manifest_path.with_name("wrong.json")
    shutil.copyfile(bundle.external_manifest_path, wrong_manifest)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="frozen absolute filename"):
        curate_frozen_legacy_selection_labels(
            dataset_id="EmotionTalk",
            legacy_sidecar_root=bundle.sidecar_root,
            external_manifest_path=wrong_manifest,
            private_output_root=bundle.output_root,
            expected_fit_training_capability_sha256=FIT_SHA,
        )

    symlink_root = tmp_path / "symlink-legacy"
    try:
        symlink_root.symlink_to(bundle.sidecar_root, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="symlink or reparse"):
        curate_frozen_legacy_selection_labels(
            dataset_id="EmotionTalk",
            legacy_sidecar_root=Path(os.path.abspath(symlink_root)),
            external_manifest_path=bundle.external_manifest_path,
            private_output_root=bundle.output_root,
            expected_fit_training_capability_sha256=FIT_SHA,
        )

    receipt = _curate(bundle)
    assert receipt.dataset_id == "EmotionTalk"
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="prior attempt"):
        _curate(bundle)


def test_output_root_must_be_outside_repo_and_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_bundle(tmp_path, monkeypatch, "MELD")
    monkeypatch.setattr(curator, "_repository_root", lambda: tmp_path)
    with pytest.raises(HarmBenchSelectionLabelCuratorError, match="outside repository"):
        _curate(bundle)


def test_public_api_has_no_raw_labels_or_caller_class_order_surface() -> None:
    assert tuple(inspect.signature(curate_frozen_legacy_selection_labels).parameters) == (
        "dataset_id",
        "legacy_sidecar_root",
        "external_manifest_path",
        "private_output_root",
        "expected_fit_training_capability_sha256",
    )
    assert "labels" not in curator.__all__
    assert all(field.name != "labels" for field in fields(CuratedSelectionLabelReceipt))
    assert "_FROZEN_DATASETS" not in curator.__all__
