from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hva_affect.harmbench_erc_checkpoint_artifact as artifact_module  # noqa: E402
import hva_affect.harmbench_erc_checkpoint_manifest as manifest_module  # noqa: E402
from hva_affect.harmbench_erc_checkpoint_artifact import (  # noqa: E402
    VerifiedCheckpointArtifact,
    publish_checkpoint_artifact,
)
from hva_affect.harmbench_erc_checkpoint_manifest import (  # noqa: E402
    CHECKPOINT_MANIFEST_SCHEMA,
    CheckpointManifest,
    EXPECTED_CHECKPOINT_ENTRY_COUNT,
    HarmBenchCheckpointManifestError,
    VerifiedCheckpointManifest,
    build_checkpoint_manifest,
    checkpoint_manifest_payload,
    load_checkpoint_manifest,
    validate_checkpoint_manifest,
    write_checkpoint_manifest_once,
)
from hva_affect.harmbench_erc_contract import EXPECTED_TRAINING_SEEDS  # noqa: E402
from hva_affect.harmbench_erc_crossfit import (  # noqa: E402
    EXPECTED_OUTER_FOLDS,
    make_shared_group_crossfit_plan,
    resolve_shared_group_crossfit_indices,
)
from hva_affect.harmbench_erc_models import (  # noqa: E402
    CURRENT_ONLY_NAMESPACE,
    HISTORY_NAMESPACE,
    LINEAR_POOL_ID,
    ProcessedRole,
    ProductionCurrentOnlyCheckpoint,
    ProductionHistoryCheckpoint,
    class_order_sha256,
    fit_synthetic_current_only_model,
    fit_synthetic_history_model,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    make_fit_role_capability,
    make_outcome_free_role_features,
    make_synthetic_fit_feature_capability,
)


FIT_ROLE = "base_and_utility_fit"
CLASS_ORDER = ("neutral", "joy", "sadness")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def synthetic_contract() -> SimpleNamespace:
    groups: list[str] = []
    keys: list[str] = []
    texts: list[str] = []
    speakers: list[str] = []
    turns: list[int] = []
    audio: list[list[float]] = []
    video: list[list[float]] = []
    labels: list[int] = []
    for group_index in range(5):
        group = f"fit_group_{group_index}"
        for turn in range(2):
            row = len(groups)
            groups.append(group)
            keys.append(f"fit:{group}:{turn}")
            texts.append(f"synthetic checkpoint manifest row {row}")
            speakers.append(f"{group}:speaker_{turn % 2}")
            turns.append(turn)
            audio.append([float(group_index + 1), float(turn + 1)])
            video.append([float(row + 1), float((group_index + 1) * (turn + 1))])
            labels.append((group_index + turn) % len(CLASS_ORDER))
    protocol_ids = np.asarray(
        [109, 101, 108, 102, 107, 103, 106, 104, 105, 100], dtype=np.int64
    )
    features = make_outcome_free_role_features(
        dataset_id="synthetic",
        role=FIT_ROLE,
        keys=np.asarray(keys),
        texts=texts,
        audio=np.asarray(audio, dtype=np.float32),
        video=np.asarray(video, dtype=np.float32),
        groups=np.asarray(groups),
        speaker_identity=np.asarray(speakers),
        turn_ids=np.asarray(turns, dtype=np.int64),
        protocol_row_ids=protocol_ids,
        row_alignment_sha256=_sha("fit-row-alignment"),
        feature_sha256=_sha("fit-feature-file"),
    )
    feature_capability = make_synthetic_fit_feature_capability(
        fit_features=features,
        feature_manifest_sha256=_sha("sanitized-fit-feature-manifest"),
        synthetic_feature_projection_sha256=_sha("cross-role-feature-roster"),
    )
    fit = make_fit_role_capability(
        fit_feature_capability=feature_capability,
        fit_labels=np.asarray(labels, dtype=np.int64),
        fit_label_sha256=_sha("fit-label-sidecar"),
        label_order=CLASS_ORDER,
        fit_manifest_sha256=_sha("sanitized-fit-training-manifest"),
    )
    plan = make_shared_group_crossfit_plan(feature_capability)
    generator = np.random.default_rng(20260809)
    model_features = ProcessedRole(
        text=generator.normal(size=(6, 256)).astype(np.float32),
        audio=generator.normal(size=(6, 128)).astype(np.float32),
        video=generator.normal(size=(6, 128)).astype(np.float32),
    )
    model_labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    model_contexts = ((), (0,), (0, 1), (1, 2), (2, 3), (3, 4))
    return SimpleNamespace(
        fit=fit,
        plan=plan,
        model_features=model_features,
        model_labels=model_labels,
        model_contexts=model_contexts,
    )


def _row_roster_shas(
    contract: SimpleNamespace,
    seed: int,
    fold: int,
) -> tuple[str, str]:
    train, heldout = resolve_shared_group_crossfit_indices(
        contract.plan,
        contract.fit.fit.feature_capability,
        training_seed=seed,
        fold=fold,
    )
    row_ids = np.asarray(contract.fit.fit.features.protocol_row_ids)
    return (
        manifest_module._canonical_array_sha256(row_ids[train]),
        manifest_module._canonical_array_sha256(row_ids[heldout]),
    )


def _production_checkpoint(
    contract: SimpleNamespace,
    *,
    namespace: str,
    seed: int,
    fold: int,
    low_level: object,
    fit_training_sha256: str | None = None,
) -> ProductionHistoryCheckpoint | ProductionCurrentOnlyCheckpoint:
    fit_sha = (
        contract.fit.capability_sha256
        if fit_training_sha256 is None
        else fit_training_sha256
    )
    train_sha, heldout_sha = _row_roster_shas(contract, seed, fold)
    common = {
        "dataset_id": contract.fit.dataset_id,
        "model_id": LINEAR_POOL_ID,
        "model_namespace": namespace,
        "training_seed": seed,
        "fold": fold,
        "class_order": CLASS_ORDER,
        "class_order_sha256": class_order_sha256(
            CLASS_ORDER,
            dataset_id=contract.fit.dataset_id,
            fit_training_capability_sha256=fit_sha,
        ),
        "fit_training_capability_sha256": fit_sha,
        "fit_feature_capability_sha256": (
            contract.fit.fit.feature_capability.capability_sha256
        ),
        "processor_receipt_sha256": _sha(f"processor:{namespace}:{seed}:{fold}"),
        "processed_output_receipt_sha256": _sha(
            f"processed-output:{namespace}:{seed}:{fold}"
        ),
        "crossfit_plan_sha256": contract.plan.plan_sha256,
        "fit_train_protocol_row_ids_sha256": train_sha,
        "fit_heldout_protocol_row_ids_sha256": heldout_sha,
        "checkpoint": low_level,
    }
    if namespace == HISTORY_NAMESPACE:
        return ProductionHistoryCheckpoint(
            **common,
            context_training_examples_sha256=_sha(
                f"context-examples:{seed}:{fold}"
            ),
            context_roster_manifest_sha256=_sha(
                f"aggregate-context-roster:{seed}:{fold}"
            ),
        )
    return ProductionCurrentOnlyCheckpoint(
        **common,
        independence_roster_sha256=_sha(f"independence-roster:{seed}:{fold}"),
        context_count=0,
        history_consumption_count=0,
    )


def _low_levels(contract: SimpleNamespace, namespace: str) -> dict[int, object]:
    result: dict[int, object] = {}
    for raw_seed in EXPECTED_TRAINING_SEEDS:
        seed = int(raw_seed)
        if namespace == HISTORY_NAMESPACE:
            result[seed] = fit_synthetic_history_model(
                LINEAR_POOL_ID,
                contract.model_features,
                contract.model_labels,
                contract.model_contexts,
                num_classes=len(CLASS_ORDER),
                seed=seed,
                epochs=1,
                query_indices=tuple(range(len(contract.model_labels))),
            )
        else:
            result[seed] = fit_synthetic_current_only_model(
                LINEAR_POOL_ID,
                contract.model_features,
                contract.model_labels,
                num_classes=len(CLASS_ORDER),
                seed=seed,
                epochs=1,
            )
    return result


def _publish_set(
    root: Path,
    contract: SimpleNamespace,
    namespace: str,
) -> tuple[VerifiedCheckpointArtifact, ...]:
    low_levels = _low_levels(contract, namespace)
    return tuple(
        publish_checkpoint_artifact(
            root,
            _production_checkpoint(
                contract,
                namespace=namespace,
                seed=int(seed),
                fold=fold,
                low_level=low_levels[int(seed)],
            ),
        )
        for seed in EXPECTED_TRAINING_SEEDS
        for fold in range(EXPECTED_OUTER_FOLDS)
    )


@pytest.fixture(scope="module")
def artifact_sets(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_contract: SimpleNamespace,
) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("checkpoint-artifacts")
    return SimpleNamespace(
        history=_publish_set(root, synthetic_contract, HISTORY_NAMESPACE),
        current=_publish_set(root, synthetic_contract, CURRENT_ONLY_NAMESPACE),
        root=root,
    )


def _build(
    contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
    *,
    namespace: str = HISTORY_NAMESPACE,
    artifacts: tuple[VerifiedCheckpointArtifact, ...] | None = None,
) -> CheckpointManifest:
    selected = (
        artifact_sets.history
        if namespace == HISTORY_NAMESPACE
        else artifact_sets.current
    )
    return build_checkpoint_manifest(
        contract.fit,
        contract.plan,
        selected if artifacts is None else artifacts,
        model_id=LINEAR_POOL_ID,
        model_namespace=namespace,
        expected_fit_training_capability_sha256=contract.fit.capability_sha256,
        expected_crossfit_plan_sha256=contract.plan.plan_sha256,
    )


def _validate(
    manifest: CheckpointManifest,
    contract: SimpleNamespace,
    artifacts: tuple[VerifiedCheckpointArtifact, ...],
    *,
    expected_manifest_sha256: str | None = None,
) -> CheckpointManifest:
    return validate_checkpoint_manifest(
        manifest,
        contract.fit,
        contract.plan,
        artifacts,
        expected_manifest_sha256=(
            manifest.manifest_sha256
            if expected_manifest_sha256 is None
            else expected_manifest_sha256
        ),
        expected_fit_training_capability_sha256=contract.fit.capability_sha256,
        expected_crossfit_plan_sha256=contract.plan.plan_sha256,
    )


def _rehash_entry(entry: object, **changes: object) -> object:
    values = manifest_module._entry_descriptor(entry)
    values.update(changes)
    return type(entry)(
        **values,
        entry_sha256=manifest_module._canonical_json_sha256(values),
    )


def _rehash_manifest(manifest: CheckpointManifest, **changes: object) -> CheckpointManifest:
    descriptor = manifest_module._manifest_descriptor(manifest)
    typed_entries = tuple(changes.pop("typed_entries", manifest.entries))
    descriptor.update(changes)
    descriptor["entries"] = [
        manifest_module._entry_payload(entry) for entry in typed_entries
    ]
    return CheckpointManifest(
        schema_version=descriptor["schema_version"],
        dataset_id=descriptor["dataset_id"],
        model_id=descriptor["model_id"],
        model_namespace=descriptor["model_namespace"],
        training_seed_ids=tuple(descriptor["training_seed_ids"]),
        outer_folds=descriptor["outer_folds"],
        entry_count=descriptor["entry_count"],
        fit_training_capability_sha256=descriptor[
            "fit_training_capability_sha256"
        ],
        fit_feature_capability_sha256=descriptor[
            "fit_feature_capability_sha256"
        ],
        crossfit_plan_sha256=descriptor["crossfit_plan_sha256"],
        ordered_class_tokens=tuple(descriptor["ordered_class_tokens"]),
        class_order_sha256=descriptor["class_order_sha256"],
        entries=typed_entries,
        manifest_sha256=manifest_module._canonical_json_sha256(descriptor),
    )


def test_manifest_is_exact_25_seed_major_fold_minor_and_fully_bound(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    manifest = _build(synthetic_contract, artifact_sets)
    expected_pairs = tuple(
        (int(seed), fold)
        for seed in EXPECTED_TRAINING_SEEDS
        for fold in range(EXPECTED_OUTER_FOLDS)
    )
    assert manifest.schema_version == CHECKPOINT_MANIFEST_SCHEMA
    assert manifest.entry_count == EXPECTED_CHECKPOINT_ENTRY_COUNT == 25
    assert tuple((entry.training_seed, entry.fold) for entry in manifest.entries) == (
        expected_pairs
    )
    assert manifest.ordered_class_tokens == CLASS_ORDER
    first = manifest.entries[0]
    first_receipt = artifact_sets.history[0].receipt
    assert first.checkpoint_payload_sha256 == first_receipt.payload_sha256
    assert first.artifact_receipt_sha256 == first_receipt.receipt_sha256
    assert (
        first.artifact_receipt_file_sha256
        == artifact_sets.history[0].receipt_file_sha256
    )
    assert first.fit_feature_capability_sha256 == (
        synthetic_contract.fit.fit.feature_capability.capability_sha256
    )
    assert first.context_roster_manifest_sha256 is not None
    assert first.context_training_examples_sha256 is not None
    assert first.independence_roster_sha256 is None
    assert _validate(manifest, synthetic_contract, artifact_sets.history) is manifest


def test_raw_binding_api_and_legacy_adapter_are_absent() -> None:
    assert not hasattr(manifest_module, "CheckpointEntryBinding")
    assert "CheckpointEntryBinding" not in manifest_module.__all__
    assert not hasattr(artifact_module, "checkpoint_entry_binding_from_artifact")
    parameters = set(inspect.signature(build_checkpoint_manifest).parameters)
    assert "artifacts" in parameters
    assert parameters.isdisjoint(
        {
            "bindings",
            "checkpoint_sha256",
            "processor_receipt_sha256",
            "context_roster_sha256",
            "class_order",
            "train_groups",
            "heldout_groups",
        }
    )
    entry_fields = {item.name for item in fields(manifest_module.CheckpointManifestEntry)}
    assert "context_roster_sha256" not in entry_fields
    assert {
        "checkpoint_payload_sha256",
        "artifact_receipt_sha256",
        "artifact_receipt_file_sha256",
        "fit_training_capability_sha256",
        "fit_feature_capability_sha256",
        "processor_receipt_sha256",
        "processed_output_receipt_sha256",
        "fit_train_protocol_row_ids_sha256",
        "fit_heldout_protocol_row_ids_sha256",
        "class_order_sha256",
    }.issubset(entry_fields)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "reordered"])
def test_missing_duplicate_extra_and_reordered_artifacts_are_rejected(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
    mutation: str,
) -> None:
    original = list(artifact_sets.history)
    if mutation == "missing":
        changed = original[:-1]
    elif mutation == "duplicate":
        changed = [*original[:-1], original[0]]
    elif mutation == "extra":
        changed = [*original, original[-1]]
    else:
        changed = original.copy()
        changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(HarmBenchCheckpointManifestError, match="seed-major/fold-minor"):
        _build(
            synthetic_contract,
            artifact_sets,
            artifacts=tuple(changed),
        )


def test_one_wrong_seed_fold_artifact_is_rejected(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    changed = list(artifact_sets.history)
    changed[7] = artifact_sets.history[8]
    with pytest.raises(HarmBenchCheckpointManifestError, match="seed-major/fold-minor"):
        _build(synthetic_contract, artifact_sets, artifacts=tuple(changed))


def test_fake_artifact_and_fake_token_are_rejected(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    fake_object = list(artifact_sets.history)
    fake_object[0] = object()  # type: ignore[list-item]
    with pytest.raises(HarmBenchCheckpointManifestError, match="live verification"):
        _build(synthetic_contract, artifact_sets, artifacts=tuple(fake_object))

    source = artifact_sets.history[0]
    forged = object.__new__(VerifiedCheckpointArtifact)
    for name in (
        "receipt",
        "receipt_path",
        "payload_path",
        "receipt_file_sha256",
        "parameters",
    ):
        object.__setattr__(forged, name, getattr(source, name))
    object.__setattr__(forged, "_verification_token", object())
    fake_token = list(artifact_sets.history)
    fake_token[0] = forged
    with pytest.raises(HarmBenchCheckpointManifestError, match="verified"):
        _build(synthetic_contract, artifact_sets, artifacts=tuple(fake_token))


def test_valid_but_mismatched_receipt_lineage_is_rejected(
    tmp_path: Path,
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    seed = int(EXPECTED_TRAINING_SEEDS[0])
    low_level = _low_levels(synthetic_contract, HISTORY_NAMESPACE)[seed]
    mismatched = publish_checkpoint_artifact(
        tmp_path,
        _production_checkpoint(
            synthetic_contract,
            namespace=HISTORY_NAMESPACE,
            seed=seed,
            fold=0,
            low_level=low_level,
            fit_training_sha256=_sha("different-fit-training-capability"),
        ),
    )
    changed = list(artifact_sets.history)
    changed[0] = mismatched
    with pytest.raises(HarmBenchCheckpointManifestError, match="fit/feature/plan lineage"):
        _build(synthetic_contract, artifact_sets, artifacts=tuple(changed))


def test_history_and_current_fields_cannot_be_mixed(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    history = _build(synthetic_contract, artifact_sets)
    current = _build(
        synthetic_contract,
        artifact_sets,
        namespace=CURRENT_ONLY_NAMESPACE,
    )
    assert history.manifest_sha256 != current.manifest_sha256
    assert all(entry.context_count is None for entry in history.entries)
    assert all(entry.history_consumption_count is None for entry in history.entries)
    assert all(entry.independence_roster_sha256 is None for entry in history.entries)
    assert all(entry.context_count == 0 for entry in current.entries)
    assert all(entry.history_consumption_count == 0 for entry in current.entries)
    assert all(entry.independence_roster_sha256 is not None for entry in current.entries)
    assert all(entry.context_roster_manifest_sha256 is None for entry in current.entries)
    assert all(entry.context_training_examples_sha256 is None for entry in current.entries)

    mixed_artifacts = list(artifact_sets.history)
    mixed_artifacts[0] = artifact_sets.current[0]
    with pytest.raises(HarmBenchCheckpointManifestError, match="dataset/model/namespace"):
        _build(
            synthetic_contract,
            artifact_sets,
            artifacts=tuple(mixed_artifacts),
        )
    with pytest.raises(HarmBenchCheckpointManifestError, match="incomplete or mixed"):
        replace(history.entries[0], context_count=0)
    with pytest.raises(HarmBenchCheckpointManifestError, match="zero context/history"):
        replace(current.entries[0], context_roster_manifest_sha256=_sha("forbidden"))


def test_artifact_receipt_file_sha_tampering_is_rejected_live(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    changed = list(artifact_sets.history)
    changed[0] = replace(
        artifact_sets.history[0],
        receipt_file_sha256=_sha("forged-receipt-file"),
    )
    with pytest.raises(HarmBenchCheckpointManifestError, match="live verification"):
        _build(synthetic_contract, artifact_sets, artifacts=tuple(changed))

    manifest = _build(synthetic_contract, artifact_sets)
    forged_entry = _rehash_entry(
        manifest.entries[0],
        artifact_receipt_file_sha256=_sha("forged-manifest-receipt-file"),
    )
    forged = _rehash_manifest(
        manifest,
        typed_entries=(forged_entry, *manifest.entries[1:]),
    )
    with pytest.raises(HarmBenchCheckpointManifestError, match="live artifact derivation"):
        _validate(forged, synthetic_contract, artifact_sets.history)


def test_protocol_row_and_class_order_tampering_fail_live_derivation(
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    manifest = _build(synthetic_contract, artifact_sets)
    first = manifest.entries[0]
    swapped = _rehash_entry(
        first,
        fit_train_protocol_row_ids_sha256=(
            first.fit_heldout_protocol_row_ids_sha256
        ),
        fit_heldout_protocol_row_ids_sha256=(
            first.fit_train_protocol_row_ids_sha256
        ),
    )
    forged_rows = _rehash_manifest(
        manifest,
        typed_entries=(swapped, *manifest.entries[1:]),
    )
    with pytest.raises(HarmBenchCheckpointManifestError, match="live artifact derivation"):
        _validate(forged_rows, synthetic_contract, artifact_sets.history)

    reversed_order = tuple(reversed(CLASS_ORDER))
    forged_class_sha = manifest_module._class_order_sha256(
        dataset_id=manifest.dataset_id,
        fit_training_capability_sha256=manifest.fit_training_capability_sha256,
        class_order=reversed_order,
    )
    forged_entries = tuple(
        _rehash_entry(entry, class_order_sha256=forged_class_sha)
        for entry in manifest.entries
    )
    forged_classes = _rehash_manifest(
        manifest,
        ordered_class_tokens=list(reversed_order),
        class_order_sha256=forged_class_sha,
        typed_entries=forged_entries,
    )
    with pytest.raises(HarmBenchCheckpointManifestError, match="live artifact derivation"):
        _validate(forged_classes, synthetic_contract, artifact_sets.history)


def test_canonical_roundtrip_returns_sealed_manifest_and_is_write_once(
    tmp_path: Path,
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    manifest = _build(synthetic_contract, artifact_sets)
    destination = tmp_path / "checkpoint_manifest.json"
    file_sha = write_checkpoint_manifest_once(destination, manifest)
    expected_raw = manifest_module._canonical_json_bytes(
        checkpoint_manifest_payload(manifest)
    )
    assert destination.read_bytes() == expected_raw
    assert hashlib.sha256(expected_raw).hexdigest() == file_sha
    loaded = load_checkpoint_manifest(
        destination,
        synthetic_contract.fit,
        synthetic_contract.plan,
        artifact_sets.history,
        expected_file_sha256=file_sha,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_fit_training_capability_sha256=(
            synthetic_contract.fit.capability_sha256
        ),
        expected_crossfit_plan_sha256=synthetic_contract.plan.plan_sha256,
    )
    assert isinstance(loaded, VerifiedCheckpointManifest)
    assert loaded.manifest == manifest
    assert loaded.manifest_file_sha256 == file_sha
    assert loaded.manifest_path == destination
    assert manifest_module._validate_verified_checkpoint_manifest(loaded) is loaded
    with pytest.raises(HarmBenchCheckpointManifestError, match="only be created"):
        VerifiedCheckpointManifest(
            manifest=manifest,
            manifest_path=destination,
            manifest_file_sha256=file_sha,
            _verification_token=object(),
        )
    with pytest.raises(HarmBenchCheckpointManifestError, match="requires a verified"):
        manifest_module._validate_verified_checkpoint_manifest(manifest)
    with pytest.raises(FileExistsError, match="write-once"):
        write_checkpoint_manifest_once(destination, manifest)

    destination.write_bytes(expected_raw + b" ")
    with pytest.raises(HarmBenchCheckpointManifestError, match="file differs"):
        manifest_module._validate_verified_checkpoint_manifest(loaded)


def test_loader_rejects_extra_json_column_even_with_matching_file_hash(
    tmp_path: Path,
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    manifest = _build(synthetic_contract, artifact_sets)
    payload = checkpoint_manifest_payload(manifest)
    payload["entries"][0]["unexpected_column"] = "forbidden"
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    destination = tmp_path / "extra_column.json"
    destination.write_bytes(raw)
    with pytest.raises(HarmBenchCheckpointManifestError, match="entry 0 keys changed"):
        load_checkpoint_manifest(
            destination,
            synthetic_contract.fit,
            synthetic_contract.plan,
            artifact_sets.history,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_fit_training_capability_sha256=(
                synthetic_contract.fit.capability_sha256
            ),
            expected_crossfit_plan_sha256=synthetic_contract.plan.plan_sha256,
        )


def test_loader_cannot_validate_handmade_sha_only_manifest(
    tmp_path: Path,
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    manifest = _build(synthetic_contract, artifact_sets)
    forged_entry = _rehash_entry(
        manifest.entries[0],
        checkpoint_payload_sha256="f" * 64,
        artifact_receipt_sha256="e" * 64,
    )
    forged = _rehash_manifest(
        manifest,
        typed_entries=(forged_entry, *manifest.entries[1:]),
    )
    destination = tmp_path / "handmade.json"
    file_sha = write_checkpoint_manifest_once(destination, forged)
    with pytest.raises(HarmBenchCheckpointManifestError, match="live artifact derivation"):
        load_checkpoint_manifest(
            destination,
            synthetic_contract.fit,
            synthetic_contract.plan,
            artifact_sets.history,
            expected_file_sha256=file_sha,
            expected_manifest_sha256=forged.manifest_sha256,
            expected_fit_training_capability_sha256=(
                synthetic_contract.fit.capability_sha256
            ),
            expected_crossfit_plan_sha256=synthetic_contract.plan.plan_sha256,
        )


def test_writer_and_loader_reject_symlink_destinations_without_clobber(
    tmp_path: Path,
    synthetic_contract: SimpleNamespace,
    artifact_sets: SimpleNamespace,
) -> None:
    manifest = _build(synthetic_contract, artifact_sets)
    target = tmp_path / "target.json"
    original = b"do-not-clobber"
    target.write_bytes(original)
    link = tmp_path / "checkpoint_link.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(HarmBenchCheckpointManifestError, match="symlink or reparse"):
        write_checkpoint_manifest_once(link, manifest)
    assert target.read_bytes() == original
    with pytest.raises(HarmBenchCheckpointManifestError, match="symlink or reparse"):
        load_checkpoint_manifest(
            link,
            synthetic_contract.fit,
            synthetic_contract.plan,
            artifact_sets.history,
            expected_file_sha256=_sha("irrelevant-external-file-binding"),
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_fit_training_capability_sha256=(
                synthetic_contract.fit.capability_sha256
            ),
            expected_crossfit_plan_sha256=synthetic_contract.plan.plan_sha256,
        )
