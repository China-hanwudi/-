from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hva_affect.harmbench_erc_checkpoint_artifact as artifact_module  # noqa: E402
from hva_affect.harmbench_erc_checkpoint_artifact import (  # noqa: E402
    HarmBenchCheckpointArtifactError,
    VerifiedCheckpointArtifact,
    checkpoint_artifact_receipt_payload,
    load_checkpoint_artifact,
    publish_checkpoint_artifact,
)
from hva_affect.harmbench_erc_contract import EXPECTED_TRAINING_SEEDS  # noqa: E402
from hva_affect.harmbench_erc_models import (  # noqa: E402
    CAUSAL_GRU_ID,
    CURRENT_ONLY_NAMESPACE,
    DEEPSETS_POOL_ID,
    HISTORY_NAMESPACE,
    LINEAR_POOL_ID,
    ProcessedRole,
    ProductionCurrentOnlyCheckpoint,
    ProductionHistoryCheckpoint,
    class_order_sha256,
    fit_synthetic_current_only_model,
    fit_synthetic_history_model,
)


CLASS_ORDER = ("neutral", "joy", "sadness")
SEED = int(EXPECTED_TRAINING_SEEDS[0])


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _features(rows: int = 6) -> ProcessedRole:
    generator = np.random.default_rng(20260809)
    return ProcessedRole(
        text=generator.normal(size=(rows, 256)).astype(np.float32),
        audio=generator.normal(size=(rows, 128)).astype(np.float32),
        video=generator.normal(size=(rows, 128)).astype(np.float32),
    )


def _current_checkpoint(model_id: str = LINEAR_POOL_ID) -> ProductionCurrentOnlyCheckpoint:
    features = _features()
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    low_level = fit_synthetic_current_only_model(
        model_id,
        features,
        labels,
        num_classes=len(CLASS_ORDER),
        seed=SEED,
        epochs=1,
    )
    fit_sha = _sha("fit-training")
    return ProductionCurrentOnlyCheckpoint(
        dataset_id="synthetic",
        model_id=model_id,
        model_namespace=CURRENT_ONLY_NAMESPACE,
        training_seed=SEED,
        fold=0,
        class_order=CLASS_ORDER,
        class_order_sha256=class_order_sha256(
            CLASS_ORDER,
            dataset_id="synthetic",
            fit_training_capability_sha256=fit_sha,
        ),
        fit_training_capability_sha256=fit_sha,
        fit_feature_capability_sha256=_sha("fit-feature"),
        processor_receipt_sha256=_sha("processor"),
        processed_output_receipt_sha256=_sha("processed-output"),
        crossfit_plan_sha256=_sha("crossfit-plan"),
        independence_roster_sha256=_sha("current-only-roster"),
        fit_train_protocol_row_ids_sha256=_sha("train-rows"),
        fit_heldout_protocol_row_ids_sha256=_sha("heldout-rows"),
        context_count=0,
        history_consumption_count=0,
        checkpoint=low_level,
    )


def _history_checkpoint() -> ProductionHistoryCheckpoint:
    features = _features()
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    contexts = ((), (0,), (0, 1), (1, 2), (2, 3), (3, 4))
    low_level = fit_synthetic_history_model(
        LINEAR_POOL_ID,
        features,
        labels,
        contexts,
        num_classes=len(CLASS_ORDER),
        seed=SEED,
        epochs=1,
        query_indices=tuple(range(len(labels))),
    )
    fit_sha = _sha("fit-training")
    return ProductionHistoryCheckpoint(
        dataset_id="synthetic",
        model_id=LINEAR_POOL_ID,
        model_namespace=HISTORY_NAMESPACE,
        training_seed=SEED,
        fold=0,
        class_order=CLASS_ORDER,
        class_order_sha256=class_order_sha256(
            CLASS_ORDER,
            dataset_id="synthetic",
            fit_training_capability_sha256=fit_sha,
        ),
        fit_training_capability_sha256=fit_sha,
        fit_feature_capability_sha256=_sha("fit-feature"),
        processor_receipt_sha256=_sha("processor"),
        processed_output_receipt_sha256=_sha("processed-output"),
        crossfit_plan_sha256=_sha("crossfit-plan"),
        context_training_examples_sha256=_sha("context-examples"),
        context_roster_manifest_sha256=_sha("context-roster-manifest"),
        fit_train_protocol_row_ids_sha256=_sha("train-rows"),
        fit_heldout_protocol_row_ids_sha256=_sha("heldout-rows"),
        checkpoint=low_level,
    )


def _binary_current_checkpoint() -> ProductionCurrentOnlyCheckpoint:
    base = _current_checkpoint()
    binary_order = ("negative", "positive")
    low_level = fit_synthetic_current_only_model(
        LINEAR_POOL_ID,
        _features(),
        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        num_classes=2,
        seed=SEED,
        epochs=1,
    )
    return replace(
        base,
        class_order=binary_order,
        class_order_sha256=class_order_sha256(
            binary_order,
            dataset_id=base.dataset_id,
            fit_training_capability_sha256=(
                base.fit_training_capability_sha256
            ),
        ),
        checkpoint=low_level,
    )


@pytest.mark.parametrize("model_id", [LINEAR_POOL_ID, DEEPSETS_POOL_ID, CAUSAL_GRU_ID])
def test_current_only_families_roundtrip_pickle_free_npz(
    tmp_path: Path,
    model_id: str,
) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint(model_id))
    assert artifact.receipt.model_id == model_id
    assert artifact.receipt.model_namespace == CURRENT_ONLY_NAMESPACE
    assert artifact.receipt.context_count == 0
    assert artifact.receipt.history_consumption_count == 0
    assert artifact.receipt.independence_roster_sha256 == _sha("current-only-roster")
    assert artifact.receipt.context_roster_manifest_sha256 is None
    assert artifact.receipt.context_training_examples_sha256 is None
    assert artifact.parameters
    if model_id == LINEAR_POOL_ID:
        assert {"coef_", "intercept_", "classes_"}.issubset(artifact.parameters)
    else:
        assert "text_projection.weight" in artifact.parameters
        assert "classifier.weight" in artifact.parameters
    with np.load(artifact.payload_path, allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[key].dtype.kind != "O" for key in archive.files)

    loaded = load_checkpoint_artifact(
        artifact.receipt_path,
        expected_receipt_file_sha256=artifact.receipt_file_sha256,
    )
    assert loaded.receipt == artifact.receipt
    assert tuple(loaded.parameters) == tuple(artifact.parameters)
    assert all(
        np.array_equal(loaded.parameters[name], artifact.parameters[name])
        for name in loaded.parameters
    )


def test_history_receipt_binds_aggregate_context_and_examples(tmp_path: Path) -> None:
    production = _history_checkpoint()
    artifact = publish_checkpoint_artifact(tmp_path, production)
    receipt = artifact.receipt
    assert receipt.context_roster_manifest_sha256 == production.context_roster_manifest_sha256
    assert receipt.context_training_examples_sha256 == production.context_training_examples_sha256
    assert receipt.independence_roster_sha256 is None
    assert receipt.context_count is None
    assert receipt.history_consumption_count is None


def test_artifact_module_has_no_manifest_dependency_or_legacy_binding_adapter() -> None:
    assert tuple(inspect.signature(publish_checkpoint_artifact).parameters) == (
        "private_root",
        "production_checkpoint",
    )
    assert "checkpoint_entry_binding_from_artifact" not in artifact_module.__all__
    assert not hasattr(artifact_module, "checkpoint_entry_binding_from_artifact")
    source = Path(artifact_module.__file__).read_text(encoding="utf-8")
    assert "harmbench_erc_checkpoint_manifest" not in source


def test_fake_verified_artifact_token_is_rejected(tmp_path: Path) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    with pytest.raises(HarmBenchCheckpointArtifactError, match="verified"):
        VerifiedCheckpointArtifact(
            receipt=artifact.receipt,
            receipt_path=artifact.receipt_path,
            payload_path=artifact.payload_path,
            receipt_file_sha256=artifact.receipt_file_sha256,
            parameters=artifact.parameters,
            _verification_token=object(),
        )


def test_write_once_and_repository_private_root_boundary(tmp_path: Path) -> None:
    production = _current_checkpoint()
    publish_checkpoint_artifact(tmp_path, production)
    with pytest.raises(FileExistsError, match="write-once"):
        publish_checkpoint_artifact(tmp_path, production)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="outside the repository"):
        publish_checkpoint_artifact(ROOT, _current_checkpoint())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_namespace", HISTORY_NAMESPACE, "namespace"),
        ("training_seed", 999_999, "seed/fold"),
        ("fold", 999_999, "seed/fold"),
        ("context_count", 1, "zero context/history"),
        ("history_consumption_count", 1, "zero context/history"),
    ],
)
def test_tampered_typed_current_wrapper_fails_before_write(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    production = _current_checkpoint()
    object.__setattr__(production, field, value)
    with pytest.raises(HarmBenchCheckpointArtifactError, match=message):
        publish_checkpoint_artifact(tmp_path, production)
    assert list(tmp_path.iterdir()) == []


def test_payload_byte_tampering_is_rejected(tmp_path: Path) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    raw = bytearray(artifact.payload_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    artifact.payload_path.write_bytes(raw)
    with pytest.raises(
        HarmBenchCheckpointArtifactError,
        match="file differs from external binding",
    ):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=artifact.receipt_file_sha256,
        )


def test_payload_path_replacement_during_verified_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    original_open = Path.open
    original_payload = artifact.payload_path.read_bytes()
    replacement_happened = False

    class ReplacingHandle:
        def __init__(self, handle: object, path: Path) -> None:
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
                self._path.write_bytes(original_payload)
            return result

        def fileno(self) -> int:
            return self._handle.fileno()

        def seek(self, *args: object) -> object:
            return self._handle.seek(*args)

        def read(self, *args: object) -> bytes:
            return self._handle.read(*args)

    def replacing_open(path: Path, *args: object, **kwargs: object) -> object:
        handle = original_open(path, *args, **kwargs)
        if path == artifact.payload_path and args and args[0] == "rb":
            return ReplacingHandle(handle, path)
        return handle

    monkeypatch.setattr(Path, "open", replacing_open)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="path changed"):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=artifact.receipt_file_sha256,
        )
    assert replacement_happened


def test_duplicate_receipt_key_is_rejected_even_with_matching_file_hash(
    tmp_path: Path,
) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    original = artifact.receipt_path.read_bytes()
    duplicate = b'{"schema_version":"forged",' + original[1:]
    artifact.receipt_path.write_bytes(duplicate)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="duplicate JSON key"):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=hashlib.sha256(duplicate).hexdigest(),
        )


def test_noncanonical_receipt_is_rejected_even_with_matching_file_hash(
    tmp_path: Path,
) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    payload = checkpoint_artifact_receipt_payload(artifact.receipt)
    noncanonical = json.dumps(payload, indent=2, sort_keys=False).encode("utf-8")
    artifact.receipt_path.write_bytes(noncanonical)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="canonical JSON"):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def _rewrite_signed_artifact(
    artifact: VerifiedCheckpointArtifact,
    case: str,
) -> str:
    receipt_payload = json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
    with np.load(artifact.payload_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    parameters = receipt_payload["architecture"]["parameters"]
    if case == "missing":
        removed = parameters.pop()
        arrays.pop(removed["storage_key"])
    elif case == "extra":
        parameters.append(
            {
                "name": "unexpected_parameter",
                "storage_key": "p9999",
                "dtype": np.dtype(np.float64).str,
                "shape": [1],
            }
        )
        arrays["p9999"] = np.zeros(1, dtype=np.float64)
    elif case == "renamed":
        parameters[0]["name"] = "renamed_coef_"
    elif case == "wrong_dtype":
        key = parameters[0]["storage_key"]
        arrays[key] = arrays[key].astype(np.float32)
        parameters[0]["dtype"] = arrays[key].dtype.str
    elif case == "wrong_shape":
        key = parameters[0]["storage_key"]
        arrays[key] = arrays[key][:, :-1]
        parameters[0]["shape"] = list(arrays[key].shape)
    elif case == "wrong_classes":
        arrays[parameters[2]["storage_key"]] = np.asarray(
            [2, 1, 0], dtype=np.int64
        )
    elif case == "wrong_feature_dimension":
        arrays[parameters[5]["storage_key"]] = np.asarray(
            [511], dtype=np.int64
        )
    elif case == "renamed_storage_key":
        key = parameters[0]["storage_key"]
        arrays["forged_storage_key"] = arrays.pop(key)
    elif case == "wrong_namespace":
        receipt_payload["model_namespace"] = HISTORY_NAMESPACE
        receipt_payload["architecture"]["model_namespace"] = HISTORY_NAMESPACE
    elif case == "wrong_family":
        receipt_payload["model_id"] = DEEPSETS_POOL_ID
        receipt_payload["architecture"]["family_id"] = DEEPSETS_POOL_ID
    else:  # pragma: no cover - test helper guard
        raise AssertionError(case)

    architecture_bytes = artifact_module._canonical_json_bytes(
        receipt_payload["architecture"]
    )
    arrays[artifact_module._ARCHITECTURE_STORAGE_KEY] = np.frombuffer(
        architecture_bytes, dtype=np.uint8
    ).copy()
    payload_raw = artifact_module._deterministic_npz_bytes(arrays)
    artifact.payload_path.write_bytes(payload_raw)
    receipt_payload["payload_sha256"] = hashlib.sha256(payload_raw).hexdigest()
    descriptor = {
        key: value
        for key, value in receipt_payload.items()
        if key != "receipt_sha256"
    }
    receipt_payload["receipt_sha256"] = artifact_module._canonical_json_sha256(
        descriptor
    )
    receipt_raw = artifact_module._canonical_json_bytes(receipt_payload)
    artifact.receipt_path.write_bytes(receipt_raw)
    return hashlib.sha256(receipt_raw).hexdigest()


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "extra",
        "renamed",
        "wrong_dtype",
        "wrong_shape",
        "wrong_classes",
        "wrong_feature_dimension",
        "renamed_storage_key",
        "wrong_namespace",
        "wrong_family",
    ),
)
def test_loader_rejects_resigned_exact_schema_and_identity_attacks(
    tmp_path: Path,
    case: str,
) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    receipt_file_sha256 = _rewrite_signed_artifact(artifact, case)
    with pytest.raises(HarmBenchCheckpointArtifactError):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=receipt_file_sha256,
        )


def test_loader_rejects_resigned_neural_state_dict_key_change(tmp_path: Path) -> None:
    artifact = publish_checkpoint_artifact(
        tmp_path, _current_checkpoint(DEEPSETS_POOL_ID)
    )
    receipt_file_sha256 = _rewrite_signed_artifact(artifact, "renamed")
    with pytest.raises(
        HarmBenchCheckpointArtifactError, match="exact frozen architecture"
    ):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=receipt_file_sha256,
        )


def test_family_byte_and_total_element_budget_attacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = publish_checkpoint_artifact(tmp_path, _current_checkpoint())
    byte_limits = dict(artifact_module.FAMILY_PAYLOAD_BYTE_LIMITS)
    byte_limits[LINEAR_POOL_ID] = artifact.payload_path.stat().st_size - 1
    monkeypatch.setattr(
        artifact_module, "FAMILY_PAYLOAD_BYTE_LIMITS", byte_limits
    )
    with pytest.raises(HarmBenchCheckpointArtifactError, match="byte budget"):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=artifact.receipt_file_sha256,
        )
    monkeypatch.undo()

    element_limits = dict(artifact_module.FAMILY_TOTAL_ELEMENT_LIMITS)
    element_limits[LINEAR_POOL_ID] = 1
    monkeypatch.setattr(
        artifact_module, "FAMILY_TOTAL_ELEMENT_LIMITS", element_limits
    )
    with pytest.raises(HarmBenchCheckpointArtifactError, match="parameter count"):
        load_checkpoint_artifact(
            artifact.receipt_path,
            expected_receipt_file_sha256=artifact.receipt_file_sha256,
        )


def test_linear_binary_and_multiclass_parameter_shapes_are_both_exact(
    tmp_path: Path,
) -> None:
    binary_root = tmp_path / "binary"
    binary_root.mkdir()
    binary = publish_checkpoint_artifact(binary_root, _binary_current_checkpoint())
    multiclass_root = tmp_path / "multiclass"
    multiclass_root.mkdir()
    multiclass = publish_checkpoint_artifact(multiclass_root, _current_checkpoint())
    assert binary.parameters["coef_"].shape == (1, 512)
    assert binary.parameters["classes_"].tolist() == [0, 1]
    assert multiclass.parameters["coef_"].shape == (3, 512)
    assert multiclass.parameters["classes_"].tolist() == [0, 1, 2]
