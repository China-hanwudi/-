from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Callable
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import hva_affect.harmbench_erc_selection_labels as label_module  # noqa: E402
import hva_affect.harmbench_erc_selection_prelabel as prelabel_module  # noqa: E402
from test_harmbench_erc_selection_prelabel import (  # noqa: E402
    _dataset_values,
    _sha,
    state,
)
from hva_affect.harmbench_erc_selection_labels import (  # noqa: E402
    ActivatedSelectionLabelCapability,
    HarmBenchSelectionLabelError,
    SELECTION_LABEL_ARTIFACT_FILENAME,
    SELECTION_LABEL_ARTIFACT_SCHEMA,
    SELECTION_LABEL_MANIFEST_FILENAME,
    SELECTION_LABEL_MANIFEST_SCHEMA,
    SELECTION_LABEL_ROLE,
    SelectionLabelManifestMetadata,
    load_selection_label_manifest_metadata,
    selection_label_manifest_sha256,
    selection_protocol_row_alignment_sha256,
)


DATASET_ID = "EmotionTalk"
CLASS_TOKENS = ("neutral", "joy", "sadness")
CLASS_ORDER_SHA256 = _sha("class-order:EmotionTalk")
PROTOCOL_ROW_IDS = _dataset_values(DATASET_ID)[0]
LABELS = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
MANIFEST_FIELDS = {
    "schema_version",
    "artifact_schema_version",
    "dataset_id",
    "role",
    "rows",
    "ordered_protocol_row_alignment_sha256",
    "class_order_sha256",
    "artifact_filename",
    "artifact_file_sha256",
}
NPZ_MEMBERS = (
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


@pytest.fixture
def private_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(label_module, "_repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(label_module, "_home_root", lambda: tmp_path / "home")
    return root.resolve()


def _paths(root: Path) -> tuple[Path, Path]:
    return (
        root / SELECTION_LABEL_ARTIFACT_FILENAME,
        root / SELECTION_LABEL_MANIFEST_FILENAME,
    )


def _publish(root: Path) -> dict[str, object]:
    return label_module._publish_trusted_synthetic_selection_labels(
        private_root=root,
        dataset_id=DATASET_ID,
        labels=LABELS,
        protocol_row_ids=PROTOCOL_ROW_IDS,
        class_tokens=CLASS_TOKENS,
        class_order_sha256=CLASS_ORDER_SHA256,
    )


def _metadata(root: Path, manifest: dict[str, object]) -> SelectionLabelManifestMetadata:
    return load_selection_label_manifest_metadata(
        private_root=root,
        manifest_path=root / SELECTION_LABEL_MANIFEST_FILENAME,
        expected_manifest_sha256=selection_label_manifest_sha256(manifest),
    )


@pytest.fixture
def activation_factory(
    state: SimpleNamespace, tmp_path: Path
) -> Callable[[SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability]:
    counter = 0

    def activate(
        metadata: SelectionLabelManifestMetadata,
    ) -> ActivatedSelectionLabelCapability:
        nonlocal counter
        counter += 1
        output_root = tmp_path / f"ticket-attempt-{counter}"
        output_root.mkdir()
        manifests = [
            metadata if item.dataset_id == DATASET_ID else item
            for item in state.metadata
        ]
        try:
            prelabel = prelabel_module.write_selection_prelabel_bundle_once(
                private_root=output_root.resolve(),
                protocol=state.protocol,
                prediction_artifacts=state.artifacts,
                label_manifests=manifests,
            )
            attempt = prelabel_module.start_selection_evaluation_attempt(prelabel)
            tickets = prelabel_module._issue_attempt_bound_label_access_tickets(
                attempt
            )
            ticket = next(item for item in tickets if item.dataset_id == DATASET_ID)
            return label_module._activate_selection_labels_from_attempt_ticket(ticket)
        except HarmBenchSelectionLabelError:
            raise
        except Exception as error:
            raise HarmBenchSelectionLabelError(str(error)) from error

    return activate


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("w+b") as handle:
        np.savez_compressed(handle, **arrays)


def _rebind_manifest(root: Path, mutate: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
    artifact_path, manifest_path = _paths(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_file_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    if mutate is not None:
        mutate(manifest)
    manifest_path.write_bytes(label_module._canonical_json_bytes(manifest))
    return manifest


def test_exact_manifest_npz_schema_and_round_trip(
    private_root: Path,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    manifest = _publish(private_root)
    artifact_path, manifest_path = _paths(private_root)
    assert set(manifest) == MANIFEST_FIELDS
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.read_bytes() == label_module._canonical_json_bytes(manifest)
    assert manifest["schema_version"] == SELECTION_LABEL_MANIFEST_SCHEMA
    assert manifest["artifact_schema_version"] == SELECTION_LABEL_ARTIFACT_SCHEMA
    assert manifest["role"] == SELECTION_LABEL_ROLE
    assert manifest["artifact_filename"] == SELECTION_LABEL_ARTIFACT_FILENAME
    assert manifest["artifact_file_sha256"] == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    assert manifest["ordered_protocol_row_alignment_sha256"] == (
        selection_protocol_row_alignment_sha256(PROTOCOL_ROW_IDS)
    )
    forbidden_fragments = {
        "feature",
        "text",
        "audio",
        "video",
        "group",
        "speaker",
        "history",
        "path",
        "stratum",
        "prediction",
    }
    assert not any(
        fragment in key for key in manifest for fragment in forbidden_fragments
    )

    arrays = _read_arrays(artifact_path)
    assert tuple(arrays) == NPZ_MEMBERS
    assert arrays["labels"].dtype == np.dtype("int64")
    assert arrays["protocol_row_ids"].dtype == np.dtype("int64")
    assert arrays["class_tokens"].dtype.kind == "U"
    assert not any(value.dtype.kind == "O" for value in arrays.values())

    metadata = _metadata(private_root, manifest)
    capability = activation_factory(metadata)
    assert np.array_equal(capability.labels, LABELS)
    assert np.array_equal(capability.protocol_row_ids, PROTOCOL_ROW_IDS)
    assert tuple(capability.class_tokens.tolist()) == CLASS_TOKENS
    assert not capability.labels.flags.writeable
    assert not capability.protocol_row_ids.flags.writeable
    assert not capability.class_tokens.flags.writeable
    with pytest.raises(ValueError):
        capability.labels.setflags(write=True)
    assert label_module._revalidate_activated_selection_labels(capability) is capability


def test_manifest_metadata_loader_never_touches_label_artifact(
    private_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _publish(private_root)
    original_open = Path.open
    original_lstat = Path.lstat
    original_resolve = Path.resolve
    original_os_lstat = os.lstat
    touched: list[str] = []

    def forbidden(path: object) -> None:
        if Path(path).name == SELECTION_LABEL_ARTIFACT_FILENAME:
            touched.append(str(path))
            raise AssertionError("metadata loader touched the label artifact")

    def trapped_open(path: Path, *args: object, **kwargs: object) -> Any:
        forbidden(path)
        return original_open(path, *args, **kwargs)

    def trapped_lstat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        forbidden(path)
        return original_lstat(path, *args, **kwargs)

    def trapped_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        forbidden(path)
        return original_resolve(path, *args, **kwargs)

    def trapped_os_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        forbidden(path)
        return original_os_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", trapped_open)
    monkeypatch.setattr(Path, "lstat", trapped_lstat)
    monkeypatch.setattr(Path, "resolve", trapped_resolve)
    monkeypatch.setattr(os, "lstat", trapped_os_lstat)
    loaded = _metadata(private_root, manifest)
    assert loaded.artifact_filename == SELECTION_LABEL_ARTIFACT_FILENAME
    assert touched == []


def test_raw_loader_publisher_and_activation_are_not_public() -> None:
    assert "_load_label_npz_once" not in label_module.__all__
    assert "_publish_trusted_synthetic_selection_labels" not in label_module.__all__
    assert not hasattr(
        label_module, "_activate_selection_labels_after_attempt_marker_fsync"
    )
    assert "_activate_selection_labels_from_attempt_ticket" not in label_module.__all__
    assert not hasattr(label_module, "load_selection_labels")
    signature = inspect.signature(
        label_module._activate_selection_labels_from_attempt_ticket
    )
    assert tuple(signature.parameters) == ("ticket",)


def test_public_construction_and_mutated_dataclass_are_rejected(
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    manifest = _publish(private_root)
    metadata = _metadata(private_root, manifest)
    with pytest.raises(HarmBenchSelectionLabelError, match="manifest loader"):
        SelectionLabelManifestMetadata(
            schema_version=SELECTION_LABEL_MANIFEST_SCHEMA,
            artifact_schema_version=SELECTION_LABEL_ARTIFACT_SCHEMA,
            dataset_id=DATASET_ID,
            role=SELECTION_LABEL_ROLE,
            rows=len(PROTOCOL_ROW_IDS),
            ordered_protocol_row_alignment_sha256=(
                selection_protocol_row_alignment_sha256(PROTOCOL_ROW_IDS)
            ),
            class_order_sha256=CLASS_ORDER_SHA256,
            artifact_filename=SELECTION_LABEL_ARTIFACT_FILENAME,
            artifact_file_sha256=str(manifest["artifact_file_sha256"]),
            manifest_file_sha256=selection_label_manifest_sha256(manifest),
            _origin=object(),  # type: ignore[arg-type]
            _seal=object(),
        )

    original_open = Path.open
    label_opened = False

    def tracking_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal label_opened
        if path.name == SELECTION_LABEL_ARTIFACT_FILENAME:
            label_opened = True
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    with pytest.raises(HarmBenchSelectionLabelError, match="loader-sealed"):
        activation_factory(replace(metadata, rows=metadata.rows + 1))
    assert not label_opened

    capability = activation_factory(metadata)
    with pytest.raises(HarmBenchSelectionLabelError, match="loader-minted"):
        label_module._revalidate_activated_selection_labels(
            replace(capability, rows=capability.rows + 1)
        )
    with pytest.raises(HarmBenchSelectionLabelError, match="private loader"):
        ActivatedSelectionLabelCapability(
            dataset_id=capability.dataset_id,
            role=capability.role,
            rows=capability.rows,
            ordered_protocol_row_alignment_sha256=(
                capability.ordered_protocol_row_alignment_sha256
            ),
            class_order_sha256=capability.class_order_sha256,
            artifact_file_sha256=capability.artifact_file_sha256,
            manifest_file_sha256=capability.manifest_file_sha256,
            protocol_canonical_sha256=capability.protocol_canonical_sha256,
            attempt_marker_file_sha256=capability.attempt_marker_file_sha256,
            prelabel_bundle_file_sha256=capability.prelabel_bundle_file_sha256,
            prelabel_receipt_file_sha256=capability.prelabel_receipt_file_sha256,
            ticket_binding_sha256=capability.ticket_binding_sha256,
            labels=capability.labels,
            protocol_row_ids=capability.protocol_row_ids,
            class_tokens=capability.class_tokens,
            _origin=capability._origin,
            _seal=object(),
        )


def test_activation_opens_canonical_label_path_once_and_uses_same_handle(
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    manifest = _publish(private_root)
    metadata = _metadata(private_root, manifest)
    original_open = Path.open
    read_opens = 0

    def counting_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal read_opens
        if (
            path.name == SELECTION_LABEL_ARTIFACT_FILENAME
            and args
            and args[0] == "rb"
        ):
            read_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    capability = activation_factory(metadata)
    assert np.array_equal(capability.labels, LABELS)
    assert read_opens == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("label_range", "outside"),
        ("label_dtype", "int64"),
        ("row_order", "alignment"),
        ("class_order", "class-token"),
        ("object_array", "unsafe dtype"),
        ("extra_member", "exact ordered member"),
    ),
)
def test_self_consistent_private_npz_semantic_forgery_is_rejected(
    private_root: Path,
    mutation: str,
    message: str,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    _publish(private_root)
    artifact_path, _manifest_path = _paths(private_root)
    arrays = _read_arrays(artifact_path)
    if mutation == "label_range":
        arrays["labels"][0] = len(CLASS_TOKENS)
    elif mutation == "label_dtype":
        arrays["labels"] = arrays["labels"].astype(np.int32)
    elif mutation == "row_order":
        arrays["protocol_row_ids"] = arrays["protocol_row_ids"][::-1].copy()
        row_sha = selection_protocol_row_alignment_sha256(arrays["protocol_row_ids"])
        arrays["ordered_protocol_row_alignment_sha256"] = np.asarray(row_sha)
    elif mutation == "class_order":
        arrays["class_tokens"] = arrays["class_tokens"][::-1].copy()
    elif mutation == "object_array":
        arrays["class_tokens"] = np.asarray(CLASS_TOKENS, dtype=object)
    else:
        arrays["forbidden_feature"] = np.asarray([1], dtype=np.int64)
    _write_arrays(artifact_path, arrays)

    def mutate_manifest(value: dict[str, object]) -> None:
        if mutation == "row_order":
            value["ordered_protocol_row_alignment_sha256"] = str(
                arrays["ordered_protocol_row_alignment_sha256"].item()
            )

    rebound = _rebind_manifest(private_root, mutate_manifest)
    metadata = _metadata(private_root, rebound)
    with pytest.raises(HarmBenchSelectionLabelError, match=message):
        activation_factory(metadata)


def test_external_manifest_sha_blocks_self_consistent_pair_replacement_before_label_open(
    private_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _publish(private_root)
    expected = selection_label_manifest_sha256(manifest)
    artifact_path, manifest_path = _paths(private_root)
    arrays = _read_arrays(artifact_path)
    arrays["labels"] = np.asarray([2, 2, 2, 2, 2, 2], dtype=np.int64)
    _write_arrays(artifact_path, arrays)
    _rebind_manifest(private_root)
    opened = False
    original_open = Path.open

    def tracking_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal opened
        if path.name == SELECTION_LABEL_ARTIFACT_FILENAME:
            opened = True
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    with pytest.raises(HarmBenchSelectionLabelError, match="manifest SHA-256"):
        load_selection_label_manifest_metadata(
            private_root=private_root,
            manifest_path=manifest_path,
            expected_manifest_sha256=expected,
        )
    assert not opened


def test_duplicate_and_noncanonical_manifest_fail_with_matching_external_hash(
    private_root: Path,
) -> None:
    manifest = _publish(private_root)
    _artifact_path, manifest_path = _paths(private_root)
    original = manifest_path.read_bytes()
    duplicate = b'{"schema_version":"forged",' + original[1:]
    manifest_path.write_bytes(duplicate)
    with pytest.raises(HarmBenchSelectionLabelError, match="duplicate JSON key"):
        load_selection_label_manifest_metadata(
            private_root=private_root,
            manifest_path=manifest_path,
            expected_manifest_sha256=hashlib.sha256(duplicate).hexdigest(),
        )
    noncanonical = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
    manifest_path.write_bytes(noncanonical)
    with pytest.raises(HarmBenchSelectionLabelError, match="not canonical"):
        load_selection_label_manifest_metadata(
            private_root=private_root,
            manifest_path=manifest_path,
            expected_manifest_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def test_duplicate_npz_member_and_budget_bombs_fail_closed(
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    _publish(private_root)
    artifact_path, _manifest_path = _paths(private_root)
    duplicate_path = private_root / "duplicate.zip"
    shutil.copyfile(artifact_path, duplicate_path)
    with zipfile.ZipFile(duplicate_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        original_member = archive.read("labels.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("labels.npy", original_member)
    duplicate_path.replace(artifact_path)
    rebound = _rebind_manifest(private_root)
    with pytest.raises(HarmBenchSelectionLabelError, match="exact ordered member"):
        activation_factory(_metadata(private_root, rebound))

    # A strict pre-deserialization element budget is enforced from NPY headers.
    private_root_2 = private_root.parent / "private_budget"
    private_root_2.mkdir()
    _publish(private_root_2)
    budget_manifest = json.loads(
        (private_root_2 / SELECTION_LABEL_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    budget_metadata = _metadata(private_root_2, budget_manifest)
    monkeypatch.setattr(label_module, "MAX_SELECTION_LABEL_ELEMENTS", 5)
    with pytest.raises(HarmBenchSelectionLabelError, match="element or byte budget"):
        activation_factory(budget_metadata)


def test_path_replacement_with_byte_identical_copy_is_rejected(
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    manifest = _publish(private_root)
    metadata = _metadata(private_root, manifest)
    artifact_path, _manifest_path = _paths(private_root)
    original_bytes = artifact_path.read_bytes()
    original_open = Path.open
    replacement_happened = False

    class ReplacingHandle:
        def __init__(self, handle: Any, path: Path) -> None:
            self._handle = handle
            self._path = path

        def __enter__(self) -> "ReplacingHandle":
            self._handle.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            nonlocal replacement_happened
            result = self._handle.__exit__(*args)
            if not replacement_happened:
                replacement_happened = True
                displaced = self._path.with_suffix(".displaced")
                os.replace(self._path, displaced)
                with original_open(self._path, "wb") as replacement:
                    replacement.write(original_bytes)
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def replacing_open(path: Path, *args: object, **kwargs: object) -> Any:
        handle = original_open(path, *args, **kwargs)
        if (
            path == artifact_path
            and args
            and args[0] == "rb"
        ):
            return ReplacingHandle(handle, path)
        return handle

    monkeypatch.setattr(Path, "open", replacing_open)
    with pytest.raises(HarmBenchSelectionLabelError, match="path changed identity"):
        activation_factory(metadata)
    assert replacement_happened


def test_renamed_wrong_copy_symlink_and_reparse_paths_are_rejected(
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation_factory: Callable[
        [SelectionLabelManifestMetadata], ActivatedSelectionLabelCapability
    ],
) -> None:
    manifest = _publish(private_root)
    metadata = _metadata(private_root, manifest)
    artifact_path, manifest_path = _paths(private_root)
    renamed = private_root / "renamed-labels.npz"
    artifact_path.rename(renamed)
    with pytest.raises(HarmBenchSelectionLabelError, match="cannot stat"):
        activation_factory(metadata)
    shutil.copyfile(renamed, artifact_path)
    wrong_manifest = private_root / "copied-manifest.json"
    shutil.copyfile(manifest_path, wrong_manifest)
    with pytest.raises(HarmBenchSelectionLabelError, match="fixed canonical filename"):
        load_selection_label_manifest_metadata(
            private_root=private_root,
            manifest_path=wrong_manifest,
            expected_manifest_sha256=selection_label_manifest_sha256(manifest),
        )

    artifact_path.unlink()
    try:
        artifact_path.symlink_to(renamed)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(HarmBenchSelectionLabelError, match="symlink or reparse"):
        activation_factory(metadata)
    artifact_path.unlink()
    shutil.copyfile(renamed, artifact_path)
    original_is_reparse = label_module._is_reparse

    def synthetic_reparse(observed: os.stat_result) -> bool:
        return observed.st_size == artifact_path.stat().st_size or original_is_reparse(
            observed
        )

    monkeypatch.setattr(label_module, "_is_reparse", synthetic_reparse)
    with pytest.raises(HarmBenchSelectionLabelError, match="reparse"):
        activation_factory(metadata)


def test_exact_alignment_and_class_binding_rejected_before_npz_open(
    private_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata(private_root, _publish(private_root))
    original_open = Path.open
    label_opens = 0

    def tracking_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal label_opens
        if path.name == SELECTION_LABEL_ARTIFACT_FILENAME:
            label_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    with pytest.raises(HarmBenchSelectionLabelError, match="terminally"):
        label_module._activate_selection_labels_from_attempt_ticket(metadata)
    assert label_opens == 0


def test_write_once_concurrent_unique_winner_and_no_temporary_files(
    private_root: Path,
) -> None:
    def publish() -> dict[str, object]:
        return _publish(private_root)

    results: list[dict[str, object]] = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish), executor.submit(publish)]
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                errors.append(error)
    assert len(results) == 1
    assert len(errors) == 1 and isinstance(errors[0], FileExistsError)
    assert sorted(path.name for path in private_root.iterdir()) == sorted(
        [SELECTION_LABEL_ARTIFACT_FILENAME, SELECTION_LABEL_MANIFEST_FILENAME]
    )
    with pytest.raises(FileExistsError, match="write-once"):
        _publish(private_root)


@pytest.mark.parametrize(
    "bad_values",
    (
        {"labels": np.asarray([0, 1, 2, 1, 0, 2], dtype=np.int32)},
        {"protocol_row_ids": np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32)},
        {"protocol_row_ids": np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64)},
        {"class_tokens": ("neutral", "neutral")},
        {"class_order_sha256": "not-a-sha"},
    ),
)
def test_synthetic_publisher_rejects_malformed_inputs_without_output(
    private_root: Path, bad_values: dict[str, object]
) -> None:
    values: dict[str, object] = {
        "private_root": private_root,
        "dataset_id": DATASET_ID,
        "labels": LABELS,
        "protocol_row_ids": PROTOCOL_ROW_IDS,
        "class_tokens": CLASS_TOKENS,
        "class_order_sha256": CLASS_ORDER_SHA256,
    }
    values.update(bad_values)
    with pytest.raises(HarmBenchSelectionLabelError):
        label_module._publish_trusted_synthetic_selection_labels(**values)
    assert list(private_root.iterdir()) == []
