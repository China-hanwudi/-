from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Callable

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hva_affect.harmbench_erc_prediction_artifact as prediction_module  # noqa: E402
import hva_affect.harmbench_erc_selection_labels as label_module  # noqa: E402
import hva_affect.harmbench_erc_selection_prelabel as prelabel_module  # noqa: E402
from hva_affect.harmbench_erc_contexts import (  # noqa: E402
    SELECTION_CONTEXT_ROLE,
)
from hva_affect.harmbench_erc_models import (  # noqa: E402
    CURRENT_ONLY_NAMESPACE,
    HISTORY_NAMESPACE,
)
from hva_affect.harmbench_erc_prediction_artifact import (  # noqa: E402
    LoadedPredictionArtifact,
    SELECTION_ROLE,
)
from hva_affect.harmbench_erc_protocol_v2 import (  # noqa: E402
    EXPECTED_ANCHOR_STRATEGY_ID,
    EXPECTED_CONTEXT_ROSTER_ORDER,
    EXPECTED_HISTORY_STRATEGY_ORDER,
    EXPECTED_MODEL_ORDER,
    EXPECTED_SELECTION_DATASETS,
    EXPECTED_TRAINING_SEEDS,
    load_protocol_v2,
)
from hva_affect.harmbench_erc_selection_labels import (  # noqa: E402
    SELECTION_LABEL_ARTIFACT_FILENAME,
    SELECTION_LABEL_MANIFEST_FILENAME,
    SelectionLabelManifestMetadata,
    load_selection_label_manifest_metadata,
    selection_label_manifest_sha256,
)
from hva_affect.harmbench_erc_selection_prelabel import (  # noqa: E402
    ATTEMPT_MARKER_FILENAME,
    AttemptStartedCapability,
    EXPLORATORY_STATUS,
    HarmBenchSelectionPrelabelError,
    LoadedSelectionPrelabelBundle,
    PRELABEL_BUNDLE_FILENAME,
    PRELABEL_RECEIPT_FILENAME,
    load_selection_prelabel_bundle,
    start_selection_evaluation_attempt,
    write_selection_prelabel_bundle_once,
)


CLASS_TOKENS = ("neutral", "joy", "sadness")
Q = 5


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _dataset_values(dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = 100 if dataset_id == "EmotionTalk" else 200
    queries = np.asarray([start + 7, start + 1, start + 9, start + 3, start + 5], dtype=np.int64)
    groups = np.asarray(
        [f"{dataset_id}:g0", f"{dataset_id}:g0", f"{dataset_id}:g1", f"{dataset_id}:g1", f"{dataset_id}:g1"],
        dtype=np.str_,
    )
    common = np.asarray([False, True, True, True, True], dtype=np.bool_)
    return queries, groups, common


def _strategy_depth(strategy_id: str) -> np.ndarray:
    values = {
        "independent_current_only": [0, 0, 0, 0, 0],
        "dialogue_all_past": [0, 1, 2, 4, 8],
        "same_speaker_all_past": [0, 0, 1, 2, 3],
        "recent_k3": [0, 1, 2, 3, 3],
        "similarity_top3": [0, 1, 2, 3, 3],
        "modality_balanced_top3": [0, 1, 2, 3, 3],
    }
    return np.asarray(values[strategy_id], dtype=np.int64)


def _fake_artifact(
    root: Path,
    *,
    dataset_id: str,
    model_id: str,
    strategy_id: str,
    class_order_sha256: str,
) -> LoadedPredictionArtifact:
    queries, groups, common = _dataset_values(dataset_id)
    depth = _strategy_depth(strategy_id)
    counts = np.broadcast_to(depth[None, None, :], (5, 5, Q)).copy()
    nonempty = counts > 0
    current = strategy_id == EXPECTED_ANCHOR_STRATEGY_ID
    probability_row = (
        np.asarray([0.5, 0.25, 0.25], dtype=np.float64)
        if current
        else np.asarray([0.25, 0.5, 0.25], dtype=np.float64)
    )
    per_fold = np.broadcast_to(
        probability_row[None, None, None, :], (5, 5, Q, 3)
    ).copy()
    probabilities = per_fold.mean(axis=1, dtype=np.float64)
    for value in (per_fold, probabilities, counts, nonempty):
        value.setflags(write=False)
    key = f"{dataset_id}:{model_id}:{strategy_id}"
    lineage = {
        "fit_training_capability_sha256": _sha(f"fit-training:{dataset_id}:{model_id}"),
        "fit_feature_capability_sha256": _sha(f"fit-feature:{dataset_id}:{model_id}"),
        "crossfit_plan_sha256": _sha(f"plan:{dataset_id}:{model_id}"),
        "source_capability_sha256": _sha(f"source:{dataset_id}:{model_id}"),
        "cross_role_feature_roster_sha256": _sha(f"roster:{dataset_id}:{model_id}"),
        "source_content_sha256": _sha(f"content:{dataset_id}:{model_id}"),
        "source_row_alignment_sha256": _sha(f"source-rows:{dataset_id}:{model_id}"),
        "query_roster_sha256": prediction_module._array_sha256(queries),
        "group_roster_sha256": prediction_module._array_sha256(groups),
    }
    artifact_root = root / key.replace(":", "_")
    artifact_root.mkdir(exist_ok=True)
    return LoadedPredictionArtifact(
        role=SELECTION_ROLE,
        dataset_id=dataset_id,
        model_id=model_id,
        model_namespace=(CURRENT_ONLY_NAMESPACE if current else HISTORY_NAMESPACE),
        strategy_id=strategy_id,
        context_role=SELECTION_CONTEXT_ROLE,
        training_seed_ids=tuple(int(value) for value in EXPECTED_TRAINING_SEEDS),
        fold_count=5,
        entry_count=25,
        checkpoint_manifest_sha256=_sha(f"checkpoint:{key}"),
        checkpoint_manifest_file_sha256=_sha(f"checkpoint-file:{key}"),
        class_order_sha256=class_order_sha256,
        panel_sha256=_sha(f"panel:{key}"),
        probabilities=_readonly(probabilities, dtype=np.float64),
        per_fold_probabilities=_readonly(per_fold, dtype=np.float64),
        query_protocol_row_ids=_readonly(queries, dtype=np.int64),
        group_tokens=_readonly(groups, dtype=str),
        class_tokens=_readonly(CLASS_TOKENS, dtype=str),
        fold_assignments=None,
        context_count=_readonly(counts, dtype=np.int64),
        strategy_context_nonempty=_readonly(nonempty, dtype=np.bool_),
        dialogue_history_eligible=_readonly(common, dtype=np.bool_),
        receipt=lineage,
        private_root=artifact_root.resolve(),
        artifact_path=(artifact_root / "predictions.npz").resolve(),
        receipt_path=(artifact_root / "receipt.json").resolve(),
        artifact_file_sha256=_sha(f"artifact-file:{key}"),
        receipt_file_sha256=_sha(f"receipt-file:{key}"),
        _checkpoint_manifest=object(),  # type: ignore[arg-type]
        _seal=prediction_module._LOADED_SEAL,
    )


def _fake_revalidator(
    artifact: object, *, expected_role: str | None = None
) -> LoadedPredictionArtifact:
    if (
        not isinstance(artifact, LoadedPredictionArtifact)
        or artifact._seal is not prediction_module._LOADED_SEAL
        or (expected_role is not None and artifact.role != expected_role)
    ):
        raise prediction_module.HarmBenchPredictionArtifactError("fake seal")
    return artifact


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    # All roots are synthetic.  The production policy still requires a real
    # repository-external/private location; patching only makes pytest's temp
    # tree model that location without weakening production code.
    blocked_repo = tmp_path / "blocked-repo"
    blocked_home = tmp_path / "blocked-home"
    monkeypatch.setattr(prelabel_module, "_repository_root", lambda: blocked_repo)
    monkeypatch.setattr(prelabel_module, "_home_root", lambda: blocked_home)
    monkeypatch.setattr(label_module, "_repository_root", lambda: blocked_repo)
    monkeypatch.setattr(label_module, "_home_root", lambda: blocked_home)
    monkeypatch.setattr(
        prediction_module,
        "_revalidate_loaded_prediction_artifact",
        _fake_revalidator,
    )
    monkeypatch.setattr(
        prediction_module,
        "public_prediction_receipt_sha256",
        lambda receipt: prelabel_module._canonical_sha256(dict(receipt)),
    )

    protocol = load_protocol_v2(ROOT / "configs" / "harmbench_erc_v2_candidate.json")
    prediction_root = tmp_path / "prediction-caps"
    prediction_root.mkdir()
    class_sha_by_dataset = {
        dataset_id: _sha(f"class-order:{dataset_id}")
        for dataset_id in EXPECTED_SELECTION_DATASETS
    }
    artifacts = [
        _fake_artifact(
            prediction_root,
            dataset_id=dataset_id,
            model_id=model_id,
            strategy_id=strategy_id,
            class_order_sha256=class_sha_by_dataset[dataset_id],
        )
        for dataset_id in EXPECTED_SELECTION_DATASETS
        for model_id in EXPECTED_MODEL_ORDER
        for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER
    ]
    metadata: list[SelectionLabelManifestMetadata] = []
    label_roots: dict[str, Path] = {}
    for dataset_id in EXPECTED_SELECTION_DATASETS:
        label_root = tmp_path / f"labels-{dataset_id}"
        label_root.mkdir()
        label_roots[dataset_id] = label_root
        queries, _, _ = _dataset_values(dataset_id)
        manifest = label_module._publish_trusted_synthetic_selection_labels(
            private_root=label_root.resolve(),
            dataset_id=dataset_id,
            labels=np.asarray([0, 1, 2, 1, 0], dtype=np.int64),
            protocol_row_ids=queries,
            class_tokens=CLASS_TOKENS,
            class_order_sha256=class_sha_by_dataset[dataset_id],
        )
        metadata.append(
            load_selection_label_manifest_metadata(
                private_root=label_root.resolve(),
                manifest_path=(label_root / SELECTION_LABEL_MANIFEST_FILENAME).resolve(),
                expected_manifest_sha256=selection_label_manifest_sha256(manifest),
            )
        )
    output_root = tmp_path / "prelabel-output"
    output_root.mkdir()
    return SimpleNamespace(
        protocol=protocol,
        artifacts=artifacts,
        metadata=metadata,
        label_roots=label_roots,
        output_root=output_root.resolve(),
    )


def _mutate_array(
    artifact: LoadedPredictionArtifact, name: str, mutate: Callable[[np.ndarray], None]
) -> LoadedPredictionArtifact:
    value = np.asarray(getattr(artifact, name)).copy()
    mutate(value)
    value.setflags(write=False)
    return replace(artifact, **{name: value})


def test_exact_bundle_receipt_and_irreversible_marker(state: SimpleNamespace) -> None:
    loaded = write_selection_prelabel_bundle_once(
        private_root=state.output_root,
        protocol=state.protocol,
        prediction_artifacts=tuple(reversed(state.artifacts)),
        label_manifests=tuple(reversed(state.metadata)),
    )
    assert isinstance(loaded, LoadedSelectionPrelabelBundle)
    assert loaded.bundle_path.name == PRELABEL_BUNDLE_FILENAME
    assert loaded.receipt_path.name == PRELABEL_RECEIPT_FILENAME
    bundle = json.loads(loaded.bundle_path.read_text(encoding="utf-8"))
    receipt = json.loads(loaded.receipt_path.read_text(encoding="utf-8"))
    assert loaded.bundle_path.read_bytes() == prelabel_module._canonical_json_bytes(bundle)
    assert loaded.receipt_path.read_bytes() == prelabel_module._canonical_json_bytes(receipt)
    assert bundle["evaluation_status"]["status"] == EXPLORATORY_STATUS
    assert bundle["evaluation_status"]["confirmatory_claim"] is False
    assert bundle["exact_roster"]["prediction_artifact_count"] == 36
    assert bundle["exact_roster"]["effective_hypothesis_count"] == 15
    assert bundle["exact_roster"]["effective_dataset_pair_count"] == 30
    assert len(bundle["datasets"]) == 2
    assert len(bundle["effective_pair_bindings"]) == 15
    assert all(
        len(row["dataset_pair_receipts"]) == 2
        for row in bundle["effective_pair_bindings"]
    )
    assert receipt["public_safety"] == {
        "confirmatory_claim": False,
        "contains_groups": False,
        "contains_labels": False,
        "contains_private_paths": False,
        "contains_probabilities": False,
        "contains_row_identifiers": False,
    }

    reloaded = load_selection_prelabel_bundle(
        private_root=state.output_root,
        protocol=state.protocol,
        prediction_artifacts=state.artifacts,
        label_manifests=state.metadata,
        expected_bundle_file_sha256=loaded.bundle_file_sha256,
        expected_receipt_file_sha256=loaded.receipt_file_sha256,
    )
    attempt = start_selection_evaluation_attempt(reloaded)
    assert isinstance(attempt, AttemptStartedCapability)
    assert attempt.marker_path.name == ATTEMPT_MARKER_FILENAME
    assert attempt.marker["prelabel_bundle_file_sha256"] == loaded.bundle_file_sha256
    assert attempt.marker["label_npz_access_occurred"] is False
    assert len(attempt.marker["attempt_nonce"]) == 64
    assert prelabel_module._revalidate_attempt_started_capability(attempt) is attempt
    tickets = prelabel_module._issue_attempt_bound_label_access_tickets(
        attempt
    )
    assert tuple(item.dataset_id for item in tickets) == EXPECTED_SELECTION_DATASETS
    assert all(not item.expected_protocol_row_ids.flags.writeable for item in tickets)
    activated = tuple(
        label_module._activate_selection_labels_from_attempt_ticket(ticket)
        for ticket in tickets
    )
    prelabel_module._validate_activated_label_suite_for_attempt(attempt, activated)
    assert tuple(item.dataset_id for item in activated) == EXPECTED_SELECTION_DATASETS

    with pytest.raises(HarmBenchSelectionPrelabelError, match="terminal crash-replay"):
        start_selection_evaluation_attempt(loaded)
    with pytest.raises(HarmBenchSelectionPrelabelError, match="terminal crash-replay"):
        load_selection_prelabel_bundle(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=state.artifacts,
            label_manifests=state.metadata,
            expected_bundle_file_sha256=loaded.bundle_file_sha256,
            expected_receipt_file_sha256=loaded.receipt_file_sha256,
        )


@pytest.mark.parametrize("attack", ["missing", "duplicate", "extra", "crossdataset", "crossmodel"])
def test_missing_duplicate_extra_and_typed_key_attacks_fail(
    state: SimpleNamespace, attack: str
) -> None:
    artifacts = list(state.artifacts)
    if attack == "missing":
        artifacts.pop()
    elif attack == "duplicate":
        artifacts[1] = artifacts[0]
    elif attack == "extra":
        artifacts.append(artifacts[0])
    elif attack == "crossdataset":
        artifacts[0] = replace(artifacts[0], dataset_id="MELD")
    else:
        artifacts[0] = replace(artifacts[0], model_id=EXPECTED_MODEL_ORDER[1])
    with pytest.raises(HarmBenchSelectionPrelabelError):
        write_selection_prelabel_bundle_once(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=artifacts,
            label_manifests=state.metadata,
        )


@pytest.mark.parametrize(
    "attack",
    [
        "seed",
        "row",
        "group",
        "class",
        "common_eligibility",
        "depth_25",
        "cross_model_depth",
        "current_consumption",
        "dialogue_nonempty",
    ],
)
def test_seed_row_class_common_and_depth_drift_fail(
    state: SimpleNamespace, attack: str
) -> None:
    artifacts = list(state.artifacts)
    current_index = 0
    dialogue_index = 1
    recent_second_model_index = len(EXPECTED_CONTEXT_ROSTER_ORDER) + 3
    if attack == "seed":
        artifacts[current_index] = replace(
            artifacts[current_index], training_seed_ids=(17, 29, 43, 71, 999)
        )
    elif attack == "row":
        artifacts[current_index] = _mutate_array(
            artifacts[current_index], "query_protocol_row_ids", lambda value: value.__setitem__(0, 9999)
        )
    elif attack == "group":
        artifacts[current_index] = _mutate_array(
            artifacts[current_index], "group_tokens", lambda value: value.__setitem__(0, "drift")
        )
    elif attack == "class":
        artifacts[current_index] = replace(
            artifacts[current_index], class_order_sha256=_sha("drift-class")
        )
    elif attack == "common_eligibility":
        artifacts[current_index] = _mutate_array(
            artifacts[current_index], "dialogue_history_eligible", lambda value: value.__setitem__(1, False)
        )
    elif attack == "depth_25":
        artifacts[dialogue_index] = _mutate_array(
            artifacts[dialogue_index], "context_count", lambda value: value.__setitem__((2, 3, 2), 3)
        )
        artifacts[dialogue_index] = replace(
            artifacts[dialogue_index],
            strategy_context_nonempty=_readonly(
                artifacts[dialogue_index].context_count > 0, dtype=np.bool_
            ),
        )
    elif attack == "cross_model_depth":
        artifacts[recent_second_model_index] = _mutate_array(
            artifacts[recent_second_model_index],
            "context_count",
            lambda value: value.__setitem__((slice(None), slice(None), 2), 3),
        )
        artifacts[recent_second_model_index] = replace(
            artifacts[recent_second_model_index],
            strategy_context_nonempty=_readonly(
                artifacts[recent_second_model_index].context_count > 0,
                dtype=np.bool_,
            ),
        )
    elif attack == "current_consumption":
        artifacts[current_index] = _mutate_array(
            artifacts[current_index],
            "context_count",
            lambda value: value.__setitem__((slice(None), slice(None), 1), 1),
        )
        artifacts[current_index] = replace(
            artifacts[current_index],
            strategy_context_nonempty=_readonly(
                artifacts[current_index].context_count > 0, dtype=np.bool_
            ),
        )
    else:
        artifacts[dialogue_index] = _mutate_array(
            artifacts[dialogue_index],
            "context_count",
            lambda value: value.__setitem__((slice(None), slice(None), 1), 0),
        )
        artifacts[dialogue_index] = replace(
            artifacts[dialogue_index],
            strategy_context_nonempty=_readonly(
                artifacts[dialogue_index].context_count > 0, dtype=np.bool_
            ),
        )
    with pytest.raises(HarmBenchSelectionPrelabelError):
        write_selection_prelabel_bundle_once(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=artifacts,
            label_manifests=state.metadata,
        )


def test_label_manifest_alignment_and_fake_seals_fail(state: SimpleNamespace) -> None:
    metadata = list(state.metadata)
    object.__setattr__(metadata[0], "rows", metadata[0].rows + 1)
    with pytest.raises(HarmBenchSelectionPrelabelError, match="loader-sealed"):
        write_selection_prelabel_bundle_once(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=state.artifacts,
            label_manifests=metadata,
        )

    artifacts = list(state.artifacts)
    object.__setattr__(artifacts[0], "_seal", object())
    with pytest.raises(HarmBenchSelectionPrelabelError, match="loader-sealed"):
        write_selection_prelabel_bundle_once(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=artifacts,
            label_manifests=state.metadata,
        )

    with pytest.raises(HarmBenchSelectionPrelabelError, match="loader-issued"):
        start_selection_evaluation_attempt(SimpleNamespace())  # type: ignore[arg-type]
    with pytest.raises(HarmBenchSelectionPrelabelError, match="after marker fsync"):
        AttemptStartedCapability(
            protocol_canonical_sha256=_sha("protocol"),
            prelabel_bundle_file_sha256=_sha("bundle"),
            prelabel_receipt_file_sha256=_sha("receipt"),
            marker_file_sha256=_sha("marker"),
            private_root=state.output_root,
            marker_path=state.output_root / ATTEMPT_MARKER_FILENAME,
            marker={},
            _prelabel=SimpleNamespace(),  # type: ignore[arg-type]
            _runtime=prelabel_module._AttemptRuntime(),
            _seal=object(),
        )


def test_prelabel_and_marker_never_touch_label_npz(
    state: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open
    original_lstat = Path.lstat
    original_resolve = Path.resolve
    original_os_lstat = os.lstat
    touched: list[str] = []

    def forbidden(path: object) -> None:
        if Path(path).name == SELECTION_LABEL_ARTIFACT_FILENAME:
            touched.append(str(path))
            raise AssertionError("prelabel phase touched label NPZ")

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
    loaded = write_selection_prelabel_bundle_once(
        private_root=state.output_root,
        protocol=state.protocol,
        prediction_artifacts=state.artifacts,
        label_manifests=state.metadata,
    )
    attempt = start_selection_evaluation_attempt(loaded)
    assert attempt.marker_path.exists()
    assert touched == []
    # A crash-style replay fails at the marker before any manifest/label work.
    with pytest.raises(HarmBenchSelectionPrelabelError, match="terminal crash-replay"):
        start_selection_evaluation_attempt(loaded)
    assert touched == []


def test_write_once_self_consistent_replacement_and_durability_order(
    state: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    original_sync = prelabel_module._sync_directory

    def sync(root: Path) -> None:
        events.append("directory_barrier")
        original_sync(root)

    monkeypatch.setattr(prelabel_module, "_sync_directory", sync)
    loaded = write_selection_prelabel_bundle_once(
        private_root=state.output_root,
        protocol=state.protocol,
        prediction_artifacts=state.artifacts,
        label_manifests=state.metadata,
    )
    assert events == ["directory_barrier"]
    with pytest.raises(FileExistsError):
        write_selection_prelabel_bundle_once(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=state.artifacts,
            label_manifests=state.metadata,
        )

    replacement = list(state.artifacts)
    first = replacement[0]
    replacement[0] = replace(
        first,
        panel_sha256=_sha("replacement-panel"),
        artifact_file_sha256=_sha("replacement-artifact"),
        receipt_file_sha256=_sha("replacement-receipt"),
        artifact_path=(first.private_root / "replacement.npz").resolve(),
        receipt_path=(first.private_root / "replacement.json").resolve(),
    )
    with pytest.raises(HarmBenchSelectionPrelabelError, match="live typed inputs"):
        load_selection_prelabel_bundle(
            private_root=state.output_root,
            protocol=state.protocol,
            prediction_artifacts=replacement,
            label_manifests=state.metadata,
            expected_bundle_file_sha256=loaded.bundle_file_sha256,
            expected_receipt_file_sha256=loaded.receipt_file_sha256,
        )

    attempt = start_selection_evaluation_attempt(loaded)
    assert attempt.marker_path.exists()
    assert events == ["directory_barrier", "directory_barrier"]


def test_symlink_output_and_orphan_marker_fail_closed(
    state: SimpleNamespace, tmp_path: Path
) -> None:
    target = tmp_path / "target-output"
    target.mkdir()
    link = tmp_path / "linked-output"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")
    with pytest.raises(HarmBenchSelectionPrelabelError, match="non-symlink"):
        write_selection_prelabel_bundle_once(
            private_root=link.absolute(),
            protocol=state.protocol,
            prediction_artifacts=state.artifacts,
            label_manifests=state.metadata,
        )


def test_public_api_does_not_expose_label_activation() -> None:
    assert "_issue_attempt_bound_label_access_tickets" not in prelabel_module.__all__
    assert "_revalidate_attempt_started_capability" not in prelabel_module.__all__
    assert not hasattr(prelabel_module, "activate_selection_labels")
    assert set(EXPECTED_HISTORY_STRATEGY_ORDER) == {
        "dialogue_all_past",
        "same_speaker_all_past",
        "recent_k3",
        "similarity_top3",
        "modality_balanced_top3",
    }
