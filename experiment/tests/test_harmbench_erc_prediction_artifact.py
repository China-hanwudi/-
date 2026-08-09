from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

import hva_affect.harmbench_erc_checkpoint_manifest as manifest_module  # noqa: E402
import hva_affect.harmbench_erc_prediction_artifact as prediction_module  # noqa: E402
from hva_affect.harmbench_erc_checkpoint_artifact import (  # noqa: E402
    publish_checkpoint_artifact,
)
from hva_affect.harmbench_erc_checkpoint_manifest import (  # noqa: E402
    VerifiedCheckpointManifest,
    build_checkpoint_manifest,
    load_checkpoint_manifest,
    write_checkpoint_manifest_once,
)
from hva_affect.harmbench_erc_contexts import (  # noqa: E402
    CURRENT_ONLY_STRATEGY_ID,
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    SELECTION_CONTEXT_ROLE,
    build_strict_past_context_roster,
)
from hva_affect.harmbench_erc_contract import EXPECTED_TRAINING_SEEDS  # noqa: E402
from hva_affect.harmbench_erc_crossfit import (  # noqa: E402
    EXPECTED_OUTER_FOLDS,
    make_shared_group_crossfit_plan,
    resolve_shared_group_crossfit_indices,
)
from hva_affect.harmbench_erc_models import (  # noqa: E402
    CURRENT_ONLY_NAMESPACE,
    DEEPSETS_POOL_ID,
    HISTORY_NAMESPACE,
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
    make_synthetic_selection_feature_capability,
)
from hva_affect.harmbench_erc_prediction_artifact import (  # noqa: E402
    DIALOGUE_ALL_PAST_STRATEGY_ID,
    FIT_ROLE,
    SELECTION_ROLE,
    HarmBenchPredictionArtifactError,
    LoadedPredictionArtifact,
    build_fit_fold_prediction,
    build_effective_history_current_pair,
    build_fit_oof_prediction_panel,
    build_selection_fold_prediction,
    build_selection_prediction_panel,
    load_fit_oof_prediction_artifact,
    load_selection_prediction_artifact,
    public_prediction_receipt_sha256,
    write_fit_oof_prediction_artifact,
    write_selection_prediction_artifact,
)
from hva_affect.harmbench_erc_processors import (  # noqa: E402
    fit_shared_processor,
    transform_role_features,
)


FIT_SOURCE_ROLE = "base_and_utility_fit"
SELECTION_SOURCE_ROLE = "model_selection"
CLASS_ORDER = ("neutral", "joy", "sadness")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _role_features(
    *, role: str, groups: list[str], protocol_start: int
):
    row_groups: list[str] = []
    speakers: list[str] = []
    turns: list[int] = []
    keys: list[str] = []
    texts: list[str] = []
    audio: list[list[float]] = []
    video: list[list[float]] = []
    for group_index, group in enumerate(groups):
        for turn in range(3):
            row = len(row_groups)
            row_groups.append(group)
            # Turn 1 has dialogue history but no same-speaker history; this is
            # the key distinction between common E_dialogue and strategy coverage.
            speakers.append(f"{group}:speaker:{turn % 2}")
            turns.append(turn)
            keys.append(f"{role}:{group}:{turn}")
            texts.append(f"synthetic {role} group {group_index} turn {turn}")
            audio.append([float(group_index + 1), float(turn + 1)])
            video.append([float(row + 1), float((group_index + 1) * (turn + 1))])
    rows = len(row_groups)
    protocol_ids = np.arange(protocol_start, protocol_start + rows, dtype=np.int64)
    # Deliberately make physical order differ from protocol order.
    protocol_ids = protocol_ids[::-1].copy()
    return make_outcome_free_role_features(
        dataset_id="synthetic",
        role=role,
        keys=np.asarray(keys),
        texts=texts,
        audio=np.asarray(audio, dtype=np.float32),
        video=np.asarray(video, dtype=np.float32),
        groups=np.asarray(row_groups),
        speaker_identity=np.asarray(speakers),
        turn_ids=np.asarray(turns, dtype=np.int64),
        protocol_row_ids=protocol_ids,
        row_alignment_sha256=_sha(f"{role}:row-alignment"),
        feature_sha256=_sha(f"{role}:feature-file"),
    )


def _simplex(rows: int, *, seed_index: int, fold: int) -> np.ndarray:
    result = np.empty((rows, len(CLASS_ORDER)), dtype=np.float64)
    for row in range(rows):
        raw = np.asarray(
            [
                2.0 + ((seed_index + row) % 3),
                3.0 + ((fold + row) % 2),
                4.0 + ((seed_index + fold + row) % 4),
            ],
            dtype=np.float64,
        )
        result[row] = raw / raw.sum()
    return result


@pytest.fixture(scope="module")
def sealed_contract(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    fit_features = _role_features(
        role=FIT_SOURCE_ROLE,
        groups=[f"fit_group_{index}" for index in range(5)],
        protocol_start=100,
    )
    selection_features = _role_features(
        role=SELECTION_SOURCE_ROLE,
        groups=["selection_group_0", "selection_group_1"],
        protocol_start=1_000,
    )
    roster_seed = _sha("shared-cross-role-feature-roster")
    fit_feature = make_synthetic_fit_feature_capability(
        fit_features=fit_features,
        feature_manifest_sha256=_sha("fit-feature-manifest"),
        synthetic_feature_projection_sha256=roster_seed,
    )
    selection_feature = make_synthetic_selection_feature_capability(
        selection_features=selection_features,
        manifest_sha256=_sha("selection-feature-manifest"),
        synthetic_feature_projection_sha256=roster_seed,
    )
    labels = np.arange(fit_features.rows, dtype=np.int64) % len(CLASS_ORDER)
    fit_training = make_fit_role_capability(
        fit_feature_capability=fit_feature,
        fit_labels=labels,
        fit_label_sha256=_sha("fit-label-sidecar"),
        label_order=CLASS_ORDER,
        fit_manifest_sha256=_sha("fit-training-manifest"),
    )
    plan = make_shared_group_crossfit_plan(fit_feature)
    model_generator = np.random.default_rng(20260809)
    model_features = ProcessedRole(
        text=model_generator.normal(size=(6, 256)).astype(np.float32),
        audio=model_generator.normal(size=(6, 128)).astype(np.float32),
        video=model_generator.normal(size=(6, 128)).astype(np.float32),
    )
    model_labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    low_levels = {
        int(seed): fit_synthetic_current_only_model(
            DEEPSETS_POOL_ID,
            model_features,
            model_labels,
            num_classes=len(CLASS_ORDER),
            seed=int(seed),
            epochs=1,
        )
        for seed in EXPECTED_TRAINING_SEEDS
    }
    checkpoint_root = tmp_path_factory.mktemp("prediction-checkpoints")
    artifacts = []
    fold_inputs: list[SimpleNamespace] = []
    for seed_index, raw_seed in enumerate(EXPECTED_TRAINING_SEEDS):
        seed = int(raw_seed)
        for fold in range(EXPECTED_OUTER_FOLDS):
            processor = fit_shared_processor(
                fit_feature, plan, seed=seed, fold=fold
            )
            receipt = processor.receipt
            fit_processed = transform_role_features(
                processor,
                fit_feature,
                expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
                expected_fit_feature_capability_sha256=fit_feature.capability_sha256,
                expected_transform_source_capability_sha256=(
                    fit_feature.capability_sha256
                ),
                expected_crossfit_plan_sha256=plan.plan_sha256,
                expected_seed=seed,
                expected_fold=fold,
            )
            selection_processed = transform_role_features(
                processor,
                selection_feature,
                expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
                expected_fit_feature_capability_sha256=fit_feature.capability_sha256,
                expected_transform_source_capability_sha256=(
                    selection_feature.capability_sha256
                ),
                expected_crossfit_plan_sha256=plan.plan_sha256,
                expected_seed=seed,
                expected_fold=fold,
            )
            train, heldout = resolve_shared_group_crossfit_indices(
                plan,
                fit_feature,
                training_seed=seed,
                fold=fold,
            )
            fit_ids = np.asarray(fit_features.protocol_row_ids)
            production = ProductionCurrentOnlyCheckpoint(
                dataset_id="synthetic",
                model_id=DEEPSETS_POOL_ID,
                model_namespace=CURRENT_ONLY_NAMESPACE,
                training_seed=seed,
                fold=fold,
                class_order=CLASS_ORDER,
                class_order_sha256=class_order_sha256(
                    CLASS_ORDER,
                    dataset_id="synthetic",
                    fit_training_capability_sha256=fit_training.capability_sha256,
                ),
                fit_training_capability_sha256=fit_training.capability_sha256,
                fit_feature_capability_sha256=fit_feature.capability_sha256,
                processor_receipt_sha256=receipt.processor_receipt_sha256,
                processed_output_receipt_sha256=fit_processed.output_receipt_sha256,
                crossfit_plan_sha256=plan.plan_sha256,
                independence_roster_sha256=_sha(
                    f"current-only-training-roster:{seed}:{fold}"
                ),
                fit_train_protocol_row_ids_sha256=(
                    manifest_module._canonical_array_sha256(fit_ids[train])
                ),
                fit_heldout_protocol_row_ids_sha256=(
                    manifest_module._canonical_array_sha256(fit_ids[heldout])
                ),
                context_count=0,
                history_consumption_count=0,
                checkpoint=low_levels[seed],
            )
            artifacts.append(publish_checkpoint_artifact(checkpoint_root, production))
            fold_inputs.append(
                SimpleNamespace(
                    seed=seed,
                    seed_index=seed_index,
                    fold=fold,
                    processor=processor,
                    receipt=receipt,
                    fit_processed=fit_processed,
                    selection_processed=selection_processed,
                )
            )
    manifest = build_checkpoint_manifest(
        fit_training,
        plan,
        tuple(artifacts),
        model_id=DEEPSETS_POOL_ID,
        model_namespace=CURRENT_ONLY_NAMESPACE,
        expected_fit_training_capability_sha256=fit_training.capability_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
    )
    manifest_path = checkpoint_root / "checkpoint-manifest.json"
    manifest_file_sha = write_checkpoint_manifest_once(manifest_path, manifest)
    verified = load_checkpoint_manifest(
        manifest_path,
        fit_training,
        plan,
        tuple(artifacts),
        expected_file_sha256=manifest_file_sha,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_fit_training_capability_sha256=fit_training.capability_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
    )
    fit_folds = []
    selection_folds = []
    fit_strategy_rosters = []
    selection_strategy_rosters = []
    fit_eligibility_rosters = []
    selection_eligibility_rosters = []
    for item in fold_inputs:
        fit_roster = build_strict_past_context_roster(
            fit_feature,
            fit_feature,
            item.fit_processed,
            item.receipt,
            plan,
            training_seed=item.seed,
            fold=item.fold,
            context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
            strategy_id=CURRENT_ONLY_STRATEGY_ID,
            expected_fit_plan_capability_sha256=fit_feature.capability_sha256,
            expected_source_capability_sha256=fit_feature.capability_sha256,
            expected_processor_receipt_sha256=(
                item.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                item.fit_processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
        )
        selection_roster = build_strict_past_context_roster(
            fit_feature,
            selection_feature,
            item.selection_processed,
            item.receipt,
            plan,
            training_seed=item.seed,
            fold=item.fold,
            context_role=SELECTION_CONTEXT_ROLE,
            strategy_id=CURRENT_ONLY_STRATEGY_ID,
            expected_fit_plan_capability_sha256=fit_feature.capability_sha256,
            expected_source_capability_sha256=selection_feature.capability_sha256,
            expected_processor_receipt_sha256=(
                item.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                item.selection_processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
        )
        fit_eligibility_roster = build_strict_past_context_roster(
            fit_feature,
            fit_feature,
            item.fit_processed,
            item.receipt,
            plan,
            training_seed=item.seed,
            fold=item.fold,
            context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
            strategy_id=DIALOGUE_ALL_PAST_STRATEGY_ID,
            expected_fit_plan_capability_sha256=fit_feature.capability_sha256,
            expected_source_capability_sha256=fit_feature.capability_sha256,
            expected_processor_receipt_sha256=(
                item.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                item.fit_processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
        )
        selection_eligibility_roster = build_strict_past_context_roster(
            fit_feature,
            selection_feature,
            item.selection_processed,
            item.receipt,
            plan,
            training_seed=item.seed,
            fold=item.fold,
            context_role=SELECTION_CONTEXT_ROLE,
            strategy_id=DIALOGUE_ALL_PAST_STRATEGY_ID,
            expected_fit_plan_capability_sha256=fit_feature.capability_sha256,
            expected_source_capability_sha256=selection_feature.capability_sha256,
            expected_processor_receipt_sha256=(
                item.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                item.selection_processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
        )
        fit_strategy_rosters.append(fit_roster)
        selection_strategy_rosters.append(selection_roster)
        fit_eligibility_rosters.append(fit_eligibility_roster)
        selection_eligibility_rosters.append(selection_eligibility_roster)
        fit_folds.append(
            build_fit_fold_prediction(
                verified,
                fit_feature,
                item.fit_processed,
                item.receipt,
                plan,
                fit_roster,
                fit_eligibility_roster,
                _simplex(
                    fit_roster.query_count,
                    seed_index=item.seed_index,
                    fold=item.fold,
                ),
            )
        )
        selection_folds.append(
            build_selection_fold_prediction(
                verified,
                fit_feature,
                selection_feature,
                item.selection_processed,
                item.receipt,
                plan,
                selection_roster,
                selection_eligibility_roster,
                _simplex(
                    selection_roster.query_count,
                    seed_index=item.seed_index,
                    fold=item.fold,
                ),
            )
        )
    return SimpleNamespace(
        fit_feature=fit_feature,
        selection_feature=selection_feature,
        fit_training=fit_training,
        plan=plan,
        fold_inputs=tuple(fold_inputs),
        artifacts=tuple(artifacts),
        verified=verified,
        fit_folds=tuple(fit_folds),
        selection_folds=tuple(selection_folds),
        fit_strategy_rosters=tuple(fit_strategy_rosters),
        selection_strategy_rosters=tuple(selection_strategy_rosters),
        fit_eligibility_rosters=tuple(fit_eligibility_rosters),
        selection_eligibility_rosters=tuple(selection_eligibility_rosters),
        fit_panel=build_fit_oof_prediction_panel(verified, tuple(fit_folds)),
        selection_panel=build_selection_prediction_panel(
            verified, tuple(selection_folds)
        ),
    )


@pytest.fixture(scope="module")
def history_contract(
    sealed_contract: SimpleNamespace,
    tmp_path_factory: pytest.TempPathFactory,
) -> SimpleNamespace:
    """A same-family strict-history manifest over the exact synthetic lineage."""

    generator = np.random.default_rng(20260810)
    model_features = ProcessedRole(
        text=generator.normal(size=(6, 256)).astype(np.float32),
        audio=generator.normal(size=(6, 128)).astype(np.float32),
        video=generator.normal(size=(6, 128)).astype(np.float32),
    )
    model_labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    synthetic_contexts = ((), (0,), (0, 1), (1, 2), (2, 3), (3, 4))
    low_levels = {
        int(seed): fit_synthetic_history_model(
            DEEPSETS_POOL_ID,
            model_features,
            model_labels,
            synthetic_contexts,
            num_classes=len(CLASS_ORDER),
            seed=int(seed),
            epochs=1,
            query_indices=tuple(range(len(model_labels))),
        )
        for seed in EXPECTED_TRAINING_SEEDS
    }
    checkpoint_root = tmp_path_factory.mktemp("prediction-history-checkpoints")
    artifacts = []
    fit_ids = np.asarray(
        sealed_contract.fit_feature.fit.protocol_row_ids, dtype=np.int64
    )
    for item in sealed_contract.fold_inputs:
        train, heldout = resolve_shared_group_crossfit_indices(
            sealed_contract.plan,
            sealed_contract.fit_feature,
            training_seed=item.seed,
            fold=item.fold,
        )
        production = ProductionHistoryCheckpoint(
            dataset_id="synthetic",
            model_id=DEEPSETS_POOL_ID,
            model_namespace=HISTORY_NAMESPACE,
            training_seed=item.seed,
            fold=item.fold,
            class_order=CLASS_ORDER,
            class_order_sha256=class_order_sha256(
                CLASS_ORDER,
                dataset_id="synthetic",
                fit_training_capability_sha256=(
                    sealed_contract.fit_training.capability_sha256
                ),
            ),
            fit_training_capability_sha256=(
                sealed_contract.fit_training.capability_sha256
            ),
            fit_feature_capability_sha256=(
                sealed_contract.fit_feature.capability_sha256
            ),
            processor_receipt_sha256=item.receipt.processor_receipt_sha256,
            processed_output_receipt_sha256=(
                item.fit_processed.output_receipt_sha256
            ),
            crossfit_plan_sha256=sealed_contract.plan.plan_sha256,
            context_training_examples_sha256=_sha(
                f"history-examples:{item.seed}:{item.fold}"
            ),
            context_roster_manifest_sha256=_sha(
                f"history-training-roster:{item.seed}:{item.fold}"
            ),
            fit_train_protocol_row_ids_sha256=(
                manifest_module._canonical_array_sha256(fit_ids[train])
            ),
            fit_heldout_protocol_row_ids_sha256=(
                manifest_module._canonical_array_sha256(fit_ids[heldout])
            ),
            checkpoint=low_levels[item.seed],
        )
        artifacts.append(publish_checkpoint_artifact(checkpoint_root, production))
    manifest = build_checkpoint_manifest(
        sealed_contract.fit_training,
        sealed_contract.plan,
        tuple(artifacts),
        model_id=DEEPSETS_POOL_ID,
        model_namespace=HISTORY_NAMESPACE,
        expected_fit_training_capability_sha256=(
            sealed_contract.fit_training.capability_sha256
        ),
        expected_crossfit_plan_sha256=sealed_contract.plan.plan_sha256,
    )
    manifest_path = checkpoint_root / "checkpoint-manifest.json"
    manifest_file_sha = write_checkpoint_manifest_once(manifest_path, manifest)
    verified = load_checkpoint_manifest(
        manifest_path,
        sealed_contract.fit_training,
        sealed_contract.plan,
        tuple(artifacts),
        expected_file_sha256=manifest_file_sha,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_fit_training_capability_sha256=(
            sealed_contract.fit_training.capability_sha256
        ),
        expected_crossfit_plan_sha256=sealed_contract.plan.plan_sha256,
    )
    selection_folds = []
    strategy_rosters = []
    for index, item in enumerate(sealed_contract.fold_inputs):
        strategy_roster = build_strict_past_context_roster(
            sealed_contract.fit_feature,
            sealed_contract.selection_feature,
            item.selection_processed,
            item.receipt,
            sealed_contract.plan,
            training_seed=item.seed,
            fold=item.fold,
            context_role=SELECTION_CONTEXT_ROLE,
            strategy_id="same_speaker_all_past",
            expected_fit_plan_capability_sha256=(
                sealed_contract.fit_feature.capability_sha256
            ),
            expected_source_capability_sha256=(
                sealed_contract.selection_feature.capability_sha256
            ),
            expected_processor_receipt_sha256=(
                item.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                item.selection_processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=sealed_contract.plan.plan_sha256,
        )
        strategy_rosters.append(strategy_roster)
        probability = np.empty((strategy_roster.query_count, 3), dtype=np.float64)
        for row, nonempty in enumerate(strategy_roster.context_counts):
            probability[row] = (
                np.asarray([0.01, 0.98, 0.01])
                if nonempty
                else np.asarray([0.98, 0.01, 0.01])
            )
        selection_folds.append(
            build_selection_fold_prediction(
                verified,
                sealed_contract.fit_feature,
                sealed_contract.selection_feature,
                item.selection_processed,
                item.receipt,
                sealed_contract.plan,
                strategy_roster,
                sealed_contract.selection_eligibility_rosters[index],
                probability,
            )
        )
    return SimpleNamespace(
        artifacts=tuple(artifacts),
        verified=verified,
        strategy_rosters=tuple(strategy_rosters),
        selection_folds=tuple(selection_folds),
        selection_panel=build_selection_prediction_panel(
            verified, tuple(selection_folds)
        ),
    )


@pytest.fixture
def private_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(prediction_module, "_repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(prediction_module, "_home_root", lambda: tmp_path / "home")
    return root.resolve()


def _paths(root: Path, stem: str) -> dict[str, Path]:
    return {
        "private_root": root,
        "artifact_path": root / f"{stem}.npz",
        "receipt_path": root / f"{stem}.json",
    }


def _write_and_load_selection(
    *,
    root: Path,
    stem: str,
    verified: VerifiedCheckpointManifest,
    panel: object,
) -> LoadedPredictionArtifact:
    paths = _paths(root, stem)
    receipt = write_selection_prediction_artifact(
        **paths,
        checkpoint_manifest=verified,
        panel=panel,
    )
    return load_selection_prediction_artifact(
        **paths,
        checkpoint_manifest=verified,
        expected_receipt_sha256=public_prediction_receipt_sha256(receipt),
    )


def _rewrite_npz_and_rebind_receipt(
    artifact_path: Path,
    receipt_path: Path,
    mutate: callable,
) -> str:
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    mutate(arrays)
    np.savez_compressed(artifact_path, **arrays)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["private_artifact_file_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    receipt_path.write_bytes(prediction_module._canonical_json_bytes(receipt))
    return public_prediction_receipt_sha256(receipt)


def test_writer_signatures_have_no_raw_identity_or_probability_surface() -> None:
    expected = (
        "private_root",
        "artifact_path",
        "receipt_path",
        "checkpoint_manifest",
        "panel",
    )
    assert tuple(inspect.signature(write_fit_oof_prediction_artifact).parameters) == expected
    assert tuple(inspect.signature(write_selection_prediction_artifact).parameters) == expected
    forbidden = {
        "dataset_id",
        "model_id",
        "model_namespace",
        "strategy_id",
        "training_seed_ids",
        "class_ids",
        "query_ids",
        "group_ids",
        "probabilities",
        "processor_sha256",
        "checkpoint_sha256",
        "context_count",
        "history_eligible",
    }
    assert forbidden.isdisjoint(inspect.signature(write_fit_oof_prediction_artifact).parameters)
    for builder in (build_fit_fold_prediction, build_selection_fold_prediction):
        parameters = inspect.signature(builder).parameters
        assert "strategy_context_roster" in parameters
        assert "dialogue_eligibility_roster" in parameters
        assert "history_eligible" not in parameters


def test_exact_25_order_duplicate_and_fake_manifest_fail_closed(
    sealed_contract: SimpleNamespace,
) -> None:
    verified = sealed_contract.verified
    folds = sealed_contract.fit_folds
    with pytest.raises(HarmBenchPredictionArtifactError, match="exactly 25"):
        build_fit_oof_prediction_panel(verified, folds[:-1])
    with pytest.raises(HarmBenchPredictionArtifactError, match="exactly 25"):
        build_fit_oof_prediction_panel(verified, (*folds, folds[-1]))
    duplicated = (*folds[:-1], folds[-2])
    with pytest.raises(HarmBenchPredictionArtifactError, match="order|duplicated"):
        build_fit_oof_prediction_panel(verified, duplicated)
    swapped = list(folds)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(HarmBenchPredictionArtifactError, match="order"):
        build_fit_oof_prediction_panel(verified, swapped)
    with pytest.raises(HarmBenchPredictionArtifactError, match="verified"):
        build_fit_oof_prediction_panel(object(), folds)


def test_fit_panel_is_source_ordered_exact_cover_and_group_disjoint(
    sealed_contract: SimpleNamespace,
) -> None:
    panel = sealed_contract.fit_panel
    source = sealed_contract.fit_feature.fit
    assert panel.probabilities.shape == (5, source.rows, len(CLASS_ORDER))
    assert panel.fold_assignments.shape == (5, source.rows)
    assert np.array_equal(panel.query_protocol_row_ids, source.protocol_row_ids)
    assert np.array_equal(panel.fold_assignments, sealed_contract.plan.fold_assignment)
    for seed in range(5):
        for group in set(source.groups.tolist()):
            assert len(set(panel.fold_assignments[seed][source.groups == group])) == 1


def test_fit_panel_shape_seed_row_and_group_leakage_tampering_rejected(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    base = sealed_contract.fit_panel
    wrong_shape = replace(base, probabilities=base.probabilities[0])
    with pytest.raises(HarmBenchPredictionArtifactError, match="array changed|shape"):
        write_fit_oof_prediction_artifact(
            **_paths(private_root, "wrong-shape"),
            checkpoint_manifest=sealed_contract.verified,
            panel=wrong_shape,
        )
    swapped = base.probabilities.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    swapped.setflags(write=False)
    swapped_panel = replace(base, probabilities=swapped)
    with pytest.raises(HarmBenchPredictionArtifactError, match="array changed"):
        write_fit_oof_prediction_artifact(
            **_paths(private_root, "swapped-seed"),
            checkpoint_manifest=sealed_contract.verified,
            panel=swapped_panel,
        )
    leaking = base.fold_assignments.copy()
    group = base.group_tokens[0]
    rows = np.flatnonzero(base.group_tokens == group)
    leaking[0, rows[0]] = (int(leaking[0, rows[0]]) + 1) % 5
    leaking.setflags(write=False)
    leaking_panel = replace(base, fold_assignments=leaking)
    with pytest.raises(HarmBenchPredictionArtifactError, match="array changed"):
        write_fit_oof_prediction_artifact(
            **_paths(private_root, "group-leak"),
            checkpoint_manifest=sealed_contract.verified,
            panel=leaking_panel,
        )


def test_selection_exact_25_tensor_and_live_mean_tamper_rejected(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    panel = sealed_contract.selection_panel
    rows = sealed_contract.selection_feature.selection.rows
    assert panel.per_fold_probabilities.shape == (5, 5, rows, len(CLASS_ORDER))
    assert np.array_equal(
        panel.probabilities,
        panel.per_fold_probabilities.mean(axis=1, dtype=np.float64),
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="exactly 25"):
        build_selection_prediction_panel(
            sealed_contract.verified, sealed_contract.selection_folds[:-1]
        )
    wrong_entry = sealed_contract.selection_folds[0]
    object.__setattr__(wrong_entry, "checkpoint_entry_sha256", "0" * 64)
    try:
        with pytest.raises(HarmBenchPredictionArtifactError, match="changed"):
            build_selection_prediction_panel(
                sealed_contract.verified, sealed_contract.selection_folds
            )
    finally:
        object.__setattr__(
            wrong_entry,
            "checkpoint_entry_sha256",
            sealed_contract.verified.manifest.entries[0].entry_sha256,
        )
    changed_mean = panel.probabilities.copy()
    changed_mean[0, 0] = changed_mean[0, 0][::-1]
    changed_mean.setflags(write=False)
    tampered = replace(panel, probabilities=changed_mean)
    with pytest.raises(HarmBenchPredictionArtifactError, match="array changed"):
        write_selection_prediction_artifact(
            **_paths(private_root, "mean-tamper"),
            checkpoint_manifest=sealed_contract.verified,
            panel=tampered,
        )


def test_round_trip_is_manifest_bound_readonly_and_aggregate_public(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    fit_paths = _paths(private_root, "fit")
    fit_receipt = write_fit_oof_prediction_artifact(
        **fit_paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.fit_panel,
    )
    loaded_fit = load_fit_oof_prediction_artifact(
        **fit_paths,
        checkpoint_manifest=sealed_contract.verified,
        expected_receipt_sha256=public_prediction_receipt_sha256(fit_receipt),
    )
    assert loaded_fit.model_namespace == CURRENT_ONLY_NAMESPACE
    assert loaded_fit.checkpoint_manifest_sha256 == (
        sealed_contract.verified.manifest.manifest_sha256
    )
    assert loaded_fit.fold_assignments is not None
    assert not loaded_fit.probabilities.flags.writeable
    assert not loaded_fit.query_protocol_row_ids.flags.writeable
    assert fit_receipt["entry_count"] == 25
    assert fit_receipt["class_order_sha256"] == (
        sealed_contract.verified.manifest.class_order_sha256
    )
    encoded = fit_paths["receipt_path"].read_text(encoding="utf-8")
    for private_name in ("query_protocol_row_ids", "group_tokens", "class_tokens"):
        assert private_name not in encoded

    selection_paths = _paths(private_root, "selection")
    selection_receipt = write_selection_prediction_artifact(
        **selection_paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    loaded_selection = load_selection_prediction_artifact(
        **selection_paths,
        checkpoint_manifest=sealed_contract.verified,
        expected_receipt_sha256=public_prediction_receipt_sha256(selection_receipt),
    )
    assert loaded_selection.per_fold_probabilities is not None
    assert np.array_equal(
        loaded_selection.probabilities,
        loaded_selection.per_fold_probabilities.mean(axis=1, dtype=np.float64),
    )


def test_current_only_context_fields_are_mandatory_zero_and_false(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    paths = _paths(private_root, "current-context")
    receipt = write_fit_oof_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.fit_panel,
    )
    original = public_prediction_receipt_sha256(receipt)

    def remove_count(arrays: dict[str, np.ndarray]) -> None:
        arrays.pop("context_count")

    rebound = _rewrite_npz_and_rebind_receipt(
        paths["artifact_path"], paths["receipt_path"], remove_count
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="exact schema"):
        load_fit_oof_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256=rebound,
        )
    assert rebound != original


@pytest.mark.parametrize(
    "field", ["context_count", "strategy_context_nonempty"]
)
def test_current_only_nonzero_or_true_context_tamper_rejected(
    sealed_contract: SimpleNamespace,
    private_root: Path,
    field: str,
) -> None:
    paths = _paths(private_root, f"current-{field}")
    receipt = write_fit_oof_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.fit_panel,
    )

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays[field].reshape(-1)[0] = 1 if field == "context_count" else True

    rebound = _rewrite_npz_and_rebind_receipt(
        paths["artifact_path"], paths["receipt_path"], mutate
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="context|aggregate"):
        load_fit_oof_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256=rebound,
        )
    assert rebound != public_prediction_receipt_sha256(receipt)


def test_class_namespace_npz_and_public_receipt_tampering_rejected(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    paths = _paths(private_root, "semantic-tamper")
    receipt = write_fit_oof_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.fit_panel,
    )

    def reorder_classes(arrays: dict[str, np.ndarray]) -> None:
        arrays["class_tokens"] = arrays["class_tokens"][::-1]

    rebound = _rewrite_npz_and_rebind_receipt(
        paths["artifact_path"], paths["receipt_path"], reorder_classes
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="class order"):
        load_fit_oof_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256=rebound,
        )

    public = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    public["model_namespace"] = HISTORY_NAMESPACE
    paths["receipt_path"].write_bytes(prediction_module._canonical_json_bytes(public))
    with pytest.raises(HarmBenchPredictionArtifactError):
        load_fit_oof_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256=public_prediction_receipt_sha256(public),
        )
    assert public_prediction_receipt_sha256(receipt) != rebound


def test_manifest_live_file_tamper_and_wrong_manifest_fail_before_load(
    sealed_contract: SimpleNamespace,
    tmp_path: Path,
) -> None:
    clone = tmp_path / "manifest.json"
    clone.write_bytes(sealed_contract.verified.manifest_path.read_bytes())
    copied = replace(sealed_contract.verified, manifest_path=clone.resolve())
    assert build_fit_oof_prediction_panel(copied, sealed_contract.fit_folds)
    clone.write_bytes(clone.read_bytes() + b" ")
    with pytest.raises(HarmBenchPredictionArtifactError, match="manifest"):
        build_fit_oof_prediction_panel(copied, sealed_contract.fit_folds)
    with pytest.raises(HarmBenchPredictionArtifactError, match="manifest"):
        build_fit_oof_prediction_panel(object(), sealed_contract.fit_folds)


def test_write_once_concurrent_winner_and_external_receipt_sha(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    paths = _paths(private_root, "concurrent")

    def publish() -> object:
        return write_fit_oof_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            panel=sealed_contract.fit_panel,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [future for future in (executor.submit(publish), executor.submit(publish))]
        results = []
        errors = []
        for future in outcomes:
            try:
                results.append(future.result())
            except FileExistsError as error:
                errors.append(error)
    assert len(results) == 1 and len(errors) == 1
    with pytest.raises(HarmBenchPredictionArtifactError, match="SHA-256"):
        load_fit_oof_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256="0" * 64,
        )


def test_symlink_and_private_root_boundaries(
    sealed_contract: SimpleNamespace,
    private_root: Path,
    tmp_path: Path,
) -> None:
    paths = _paths(private_root, "plain")
    receipt = write_fit_oof_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.fit_panel,
    )
    link = private_root / "receipt-link.json"
    try:
        link.symlink_to(paths["receipt_path"])
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(HarmBenchPredictionArtifactError, match="plain"):
        load_fit_oof_prediction_artifact(
            private_root=private_root,
            artifact_path=paths["artifact_path"],
            receipt_path=link,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256=public_prediction_receipt_sha256(receipt),
        )
    outside = tmp_path / "outside.npz"
    with pytest.raises(HarmBenchPredictionArtifactError, match="direct child"):
        write_fit_oof_prediction_artifact(
            private_root=private_root,
            artifact_path=outside,
            receipt_path=private_root / "outside.json",
            checkpoint_manifest=sealed_contract.verified,
            panel=sealed_contract.fit_panel,
        )


def test_fold_builder_requires_live_dialogue_all_eligibility_lineage(
    sealed_contract: SimpleNamespace,
) -> None:
    item = sealed_contract.fold_inputs[0]
    strategy = sealed_contract.selection_strategy_rosters[0]
    probability = _simplex(strategy.query_count, seed_index=0, fold=0)
    with pytest.raises(HarmBenchPredictionArtifactError, match="eligibility|strategy"):
        build_selection_fold_prediction(
            sealed_contract.verified,
            sealed_contract.fit_feature,
            sealed_contract.selection_feature,
            item.selection_processed,
            item.receipt,
            sealed_contract.plan,
            strategy,
            strategy,
            probability,
        )
    with pytest.raises(HarmBenchPredictionArtifactError, match="eligibility|verification"):
        build_selection_fold_prediction(
            sealed_contract.verified,
            sealed_contract.fit_feature,
            sealed_contract.selection_feature,
            item.selection_processed,
            item.receipt,
            sealed_contract.plan,
            strategy,
            sealed_contract.selection_eligibility_rosters[1],
            probability,
        )
    eligibility = sealed_contract.selection_eligibility_rosters[0]
    original_rows = eligibility.query_protocol_row_ids
    object.__setattr__(
        eligibility,
        "query_protocol_row_ids",
        (original_rows[0] + 999_999, *original_rows[1:]),
    )
    try:
        with pytest.raises(
            HarmBenchPredictionArtifactError, match="eligibility|verification"
        ):
            build_selection_fold_prediction(
                sealed_contract.verified,
                sealed_contract.fit_feature,
                sealed_contract.selection_feature,
                item.selection_processed,
                item.receipt,
                sealed_contract.plan,
                strategy,
                eligibility,
                probability,
            )
    finally:
        object.__setattr__(eligibility, "query_protocol_row_ids", original_rows)


def test_dialogue_eligibility_is_common_but_strategy_coverage_is_not(
    sealed_contract: SimpleNamespace,
    history_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    current = sealed_contract.selection_panel
    history = history_contract.selection_panel
    assert np.any(current.dialogue_history_eligible)
    assert not np.any(current.context_count)
    assert not np.any(current.strategy_context_nonempty)
    assert np.array_equal(
        current.dialogue_history_eligible, history.dialogue_history_eligible
    )
    common = history.dialogue_history_eligible
    same_speaker_coverage = history.strategy_context_nonempty[0, 0]
    assert np.any(common & ~same_speaker_coverage)
    assert np.array_equal(
        history.strategy_context_nonempty, history.context_count > 0
    )

    paths = _paths(private_root, "current-common-eligibility")
    receipt = write_selection_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=current,
    )
    summary = receipt["context_summary"]
    assert summary["dialogue_history_eligible_count"] > 0
    assert summary["strategy_context_nonempty_count"] == 0
    assert summary["strategy_context_count_total"] == 0
    assert summary["zero_strategy_consumption"] is True
    loaded = load_selection_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        expected_receipt_sha256=public_prediction_receipt_sha256(receipt),
    )
    assert np.any(loaded.dialogue_history_eligible)
    assert not loaded.dialogue_history_eligible.flags.writeable
    assert not loaded.strategy_context_nonempty.flags.writeable
    assert not loaded.context_count.flags.writeable


def test_old_history_eligible_private_and_public_schema_is_rejected(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    paths = _paths(private_root, "old-history-eligible")
    receipt = write_selection_prediction_artifact(
        **paths,
        checkpoint_manifest=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    old_public = dict(receipt)
    old_public["history_eligible_sha256"] = old_public.pop(
        "dialogue_history_eligible_sha256"
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="schema"):
        public_prediction_receipt_sha256(old_public)

    def old_private_schema(arrays: dict[str, np.ndarray]) -> None:
        arrays["history_eligible"] = arrays.pop("dialogue_history_eligible")

    rebound = _rewrite_npz_and_rebind_receipt(
        paths["artifact_path"], paths["receipt_path"], old_private_schema
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="exact schema"):
        load_selection_prediction_artifact(
            **paths,
            checkpoint_manifest=sealed_contract.verified,
            expected_receipt_sha256=rebound,
        )


def test_loaded_capability_seal_paths_hashes_and_path_replacement(
    sealed_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    loaded = _write_and_load_selection(
        root=private_root,
        stem="loader-seal",
        verified=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    assert loaded.private_root.is_absolute()
    assert loaded.artifact_path.is_absolute()
    assert loaded.receipt_path.is_absolute()
    assert loaded.artifact_file_sha256 == hashlib.sha256(
        loaded.artifact_path.read_bytes()
    ).hexdigest()
    assert loaded.receipt_file_sha256 == hashlib.sha256(
        loaded.receipt_path.read_bytes()
    ).hexdigest()
    with pytest.raises(HarmBenchPredictionArtifactError, match="verified loader"):
        replace(loaded, _seal=object())

    replacement = private_root / "replacement.npz"
    replacement.write_bytes(b"not-the-sealed-prediction")
    os.replace(replacement, loaded.artifact_path)
    with pytest.raises(HarmBenchPredictionArtifactError, match="SHA-256|unreadable"):
        prediction_module._revalidate_loaded_prediction_artifact(loaded)

    receipt_loaded = _write_and_load_selection(
        root=private_root,
        stem="loader-receipt-replacement",
        verified=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    receipt_replacement = private_root / "replacement.json"
    receipt_replacement.write_bytes(b"{}\n")
    os.replace(receipt_replacement, receipt_loaded.receipt_path)
    with pytest.raises(HarmBenchPredictionArtifactError, match="SHA-256"):
        prediction_module._revalidate_loaded_prediction_artifact(receipt_loaded)


def test_effective_pair_exact_fallback_and_current_anchor_reuse(
    sealed_contract: SimpleNamespace,
    history_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    current = _write_and_load_selection(
        root=private_root,
        stem="pair-current",
        verified=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    history = _write_and_load_selection(
        root=private_root,
        stem="pair-history",
        verified=history_contract.verified,
        panel=history_contract.selection_panel,
    )
    pair = build_effective_history_current_pair(history, current)
    expected_mask = np.broadcast_to(
        history.strategy_context_nonempty[0, 0][None, :],
        history.probabilities.shape[:2],
    )
    expected = np.where(
        expected_mask[..., None], history.probabilities, current.probabilities
    )
    assert history.per_fold_probabilities is not None
    assert current.per_fold_probabilities is not None
    per_fold_expected = np.where(
        expected_mask[0][None, None, :, None],
        history.per_fold_probabilities,
        current.per_fold_probabilities,
    )
    assert np.array_equal(
        per_fold_expected.mean(axis=1, dtype=np.float64), expected
    )
    assert np.array_equal(pair.use_history_mask, expected_mask)
    assert np.array_equal(pair.probabilities, expected)
    assert np.array_equal(
        pair.probabilities[~pair.use_history_mask],
        current.probabilities[~pair.use_history_mask],
    )
    assert np.array_equal(
        pair.probabilities[pair.use_history_mask],
        history.probabilities[pair.use_history_mask],
    )
    common_but_empty = history.dialogue_history_eligible[None, :] & ~expected_mask
    assert np.any(common_but_empty)
    assert np.any(
        history.probabilities[common_but_empty]
        != current.probabilities[common_but_empty]
    )
    assert np.array_equal(
        pair.probabilities[common_but_empty],
        current.probabilities[common_but_empty],
    )
    assert not pair.probabilities.flags.writeable
    assert not pair.use_history_mask.flags.writeable
    assert pair.receipt["current_panel_sha256"] == current.panel_sha256
    assert pair.receipt["effective_semantics"] == (
        "per_fold_fallback_then_five_fold_mean"
    )
    assert pair.receipt["per_fold_effective_probability_sha256"] == (
        prediction_module._array_sha256(per_fold_expected)
    )
    again = build_effective_history_current_pair(history, current)
    assert again.pair_receipt_sha256 == pair.pair_receipt_sha256
    assert np.array_equal(again.probabilities, pair.probabilities)
    with pytest.raises(HarmBenchPredictionArtifactError, match="sealed pair builder"):
        replace(pair, _seal=object())
    changed_probability = pair.probabilities.copy()
    changed_probability[0, 0] = changed_probability[0, 0][::-1]
    changed_probability.setflags(write=False)
    with pytest.raises(HarmBenchPredictionArtifactError, match="array changed"):
        prediction_module._revalidate_effective_history_current_pair(
            replace(pair, probabilities=changed_probability)
        )


def test_effective_pair_rejects_cross_family_query_class_seed_and_namespace(
    sealed_contract: SimpleNamespace,
    history_contract: SimpleNamespace,
    private_root: Path,
) -> None:
    current = _write_and_load_selection(
        root=private_root,
        stem="cross-current",
        verified=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    history = _write_and_load_selection(
        root=private_root,
        stem="cross-history",
        verified=history_contract.verified,
        panel=history_contract.selection_panel,
    )

    changed_queries = history.query_protocol_row_ids[::-1].copy()
    changed_queries.setflags(write=False)
    changed_classes = history.class_tokens[::-1].copy()
    changed_classes.setflags(write=False)
    adversarial = (
        replace(history, model_id="hb_linear_pool_v1"),
        replace(history, query_protocol_row_ids=changed_queries),
        replace(history, class_tokens=changed_classes),
        replace(history, training_seed_ids=tuple(reversed(history.training_seed_ids))),
        replace(history, model_namespace=CURRENT_ONLY_NAMESPACE),
    )
    for changed in adversarial:
        with pytest.raises(HarmBenchPredictionArtifactError, match="changed"):
            build_effective_history_current_pair(changed, current)
    with pytest.raises(HarmBenchPredictionArtifactError, match="history input"):
        build_effective_history_current_pair(current, history)


def test_effective_pair_rejects_seed_only_strategy_mask_drift(
    sealed_contract: SimpleNamespace,
    history_contract: SimpleNamespace,
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _write_and_load_selection(
        root=private_root,
        stem="seed-drift-current",
        verified=sealed_contract.verified,
        panel=sealed_contract.selection_panel,
    )
    history = _write_and_load_selection(
        root=private_root,
        stem="seed-drift-history",
        verified=history_contract.verified,
        panel=history_contract.selection_panel,
    )
    drift = history.strategy_context_nonempty.copy()
    target = int(np.flatnonzero(history.dialogue_history_eligible)[0])
    drift[1, :, target] = ~drift[0, 0, target]
    drift.setflags(write=False)
    changed = replace(history, strategy_context_nonempty=drift)

    # Isolate the pair contract from the already-tested loader revalidator so
    # this attack reaches the downstream 25-entry semantic gate itself.
    monkeypatch.setattr(
        prediction_module,
        "_revalidate_loaded_prediction_artifact",
        lambda artifact, expected_role=None: artifact,
    )
    with pytest.raises(HarmBenchPredictionArtifactError, match="25 seed/fold"):
        build_effective_history_current_pair(changed, current)
