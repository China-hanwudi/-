from __future__ import annotations

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from hva_affect.causal_backbone_evidence_runner import (
    EXPECTED_SEEDS,
    FIT_ROLE,
    SELECTION_ROLE,
    load_fit_only_producer_view,
    run_fit_preflight,
)
from hva_affect.causal_backbone_evidence_stage_b import (
    CurrentOnlyFoldOutput,
    UtilityOOFSeedOutput,
    _atomic_json_once,
    _atomic_savez_once,
    produce_independent_current_only_fit_oof,
    produce_utility_oof_scores,
    validate_current_only_fit_files,
    validate_utility_oof_files,
    write_fit_only_lineage,
    write_fit_protocol_map,
)
from test_causal_backbone_evidence_runner import (
    ENVIRONMENT,
    _lineage_files,
    _make_emotiontalk_sidecars,
    _producer_mapping,
    _sha,
    _write_npz,
)


def _stage_b_fixture(tmp_path: Path):
    manifest_path = _make_emotiontalk_sidecars(tmp_path, poison_selection=True)
    configs, code = _lineage_files(tmp_path)
    receipt_path = tmp_path / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    fit_map = write_fit_protocol_map(
        preflight.fit,
        receipt_path=receipt_path,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=tmp_path / "fit-map.npz",
    )
    lineage = write_fit_only_lineage(
        preflight.fit,
        fit_map=fit_map,
        receipt_path=receipt_path,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=tmp_path / "fit-lineage.npz",
    )

    values = _producer_mapping(selection_poison=True)
    # The four fit protocol rows occupy the producer's registered fit positions.
    values["protocol_row_ids"] = np.asarray([2, 0, 3, 1, 6, 4, 7, 5], dtype=np.int64)
    receipt = preflight.receipt
    fit_sidecar = receipt["sidecars"]["fit"]
    selection_sidecar = receipt["sidecars"][SELECTION_ROLE]
    values.update(
        {
            "source_sidecar_manifest_sha256": np.asarray(
                receipt["manifest"]["sha256"]
            ),
            f"source_{FIT_ROLE}_features_sha256": np.asarray(
                fit_sidecar["feature_sha256"]
            ),
            f"source_{FIT_ROLE}_labels_sha256": np.asarray(
                fit_sidecar["label_sha256"]
            ),
            "source_model_selection_features_sha256": np.asarray(
                selection_sidecar["feature_sha256"]
            ),
            "source_model_selection_labels_sha256": np.asarray(
                selection_sidecar["label_sha256"]
            ),
        }
    )
    producer_path = tmp_path / "producer.npz"
    _write_npz(producer_path, values)
    producer = load_fit_only_producer_view(producer_path)
    return preflight, receipt_path, fit_map, lineage, producer_path, producer


@pytest.mark.parametrize("kind", ["json", "npz"])
def test_stage_b_write_once_publication_is_race_safe(
    tmp_path: Path, kind: str
) -> None:
    destination = tmp_path / ("receipt.json" if kind == "json" else "artifact.npz")
    barrier = threading.Barrier(2)

    def write(index: int):
        barrier.wait()
        try:
            if kind == "json":
                digest = _atomic_json_once(destination, {"winner": index})
            else:
                digest = _atomic_savez_once(
                    destination,
                    {"winner": np.asarray(index, dtype=np.int64)},
                )
            return ("won", index, digest)
        except FileExistsError:
            return ("lost", index, None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, (0, 1)))
    assert sorted(value[0] for value in outcomes) == ["lost", "won"]
    winner = next(value[1] for value in outcomes if value[0] == "won")
    if kind == "json":
        assert json.loads(destination.read_text(encoding="utf-8")) == {
            "winner": winner
        }
    else:
        with np.load(destination, allow_pickle=False) as archive:
            assert int(np.asarray(archive["winner"]).reshape(())) == winner
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def _write_fold_checkpoint(request: object) -> None:
    root = request.checkpoint_root  # type: ignore[attr-defined]
    run = root / f"seed_{request.seed:05d}" / f"fold_{request.fold:02d}"  # type: ignore[attr-defined]
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoint.pt").write_bytes(
        f"checkpoint-{request.seed}-{request.fold}".encode()  # type: ignore[attr-defined]
    )
    (run / "text_processor.joblib").write_bytes(
        f"processor-{request.seed}-{request.fold}".encode()  # type: ignore[attr-defined]
    )


def _assert_public_receipt_is_aggregate_only(path: Path, private_root: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)
    assert str(private_root) not in serialized
    for forbidden in (
        '"row_ids"',
        '"groups"',
        '"labels"',
        '"probabilities"',
        '"scores"',
        '"paths"',
        "emotion text",
    ):
        assert forbidden not in serialized
    assert payload["public_artifact_policy"]["aggregate_only"] is True
    assert payload["public_artifact_policy"]["contains_performance_metrics"] is False


def test_stage_b_happy_path_hides_heldout_labels_and_targets(tmp_path: Path) -> None:
    preflight, receipt_path, fit_map, lineage, producer_path, producer = _stage_b_fixture(
        tmp_path
    )
    current_requests = []

    def current_callback(request):
        current_requests.append(request)
        assert request.heldout_labels_materialized is False
        assert not hasattr(request, "labels")
        assert not hasattr(request, "heldout_labels")
        assert request.fit_lineage_source_identity_sha256 == lineage.source_identity_sha256
        assert all(not history for history in request.train_histories)
        assert all(not history for history in request.heldout_histories)
        assert len(request.train_labels) == len(request.train_indices)
        _write_fold_checkpoint(request)
        probability = np.ones(
            (len(request.heldout_indices), len(preflight.fit.label_order)),
            dtype=np.float32,
        )
        probability /= probability.shape[1]
        return CurrentOnlyFoldOutput(probability, "a" * 64)

    fold_by_seed_row = np.tile(
        np.asarray([0, 0, 1, 1], dtype=np.int32), (len(EXPECTED_SEEDS), 1)
    )
    checkpoint_root = tmp_path / "current-checkpoints"
    current_artifact = tmp_path / "current-fit.npz"
    current_receipt = tmp_path / "current-fit-receipt.json"
    produced_current = produce_independent_current_only_fit_oof(
        fit=preflight.fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_path=receipt_path,
        expected_fit_preflight_receipt_sha256=preflight.receipt_sha256,
        fold_by_seed_row=fold_by_seed_row,
        outer_folds=2,
        checkpoint_root=checkpoint_root,
        artifact_path=current_artifact,
        producer_receipt_path=current_receipt,
        model_config_sha256=_sha("current-model"),
        run_config_sha256=_sha("current-run"),
        model_config_semantic_sha256=_sha("current-model-semantic"),
        run_config_semantic_sha256=_sha("current-run-semantic"),
        source_code_sha256=_sha("current-code"),
        runtime_environment_sha256=_sha("current-runtime"),
        fold_callback=current_callback,
    )
    assert len(current_requests) == len(EXPECTED_SEEDS) * 2
    current_summary = validate_current_only_fit_files(
        artifact_path=current_artifact,
        fit=preflight.fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_path=receipt_path,
        expected_fit_preflight_receipt_sha256=preflight.receipt_sha256,
        checkpoint_root=checkpoint_root,
        outer_folds=2,
    )
    assert current_summary["status"] == (
        "valid_private_fit_artifact_not_performance_evidence"
    )
    assert produced_current.artifact_sha256 == current_summary["artifact_sha256"]
    _assert_public_receipt_is_aggregate_only(current_receipt, tmp_path)

    utility_requests = []

    def utility_callback(request):
        utility_requests.append(request)
        assert request.heldout_targets_materialized is False
        assert request.selection_payload_consumed is False
        assert not hasattr(request, "heldout_forward_targets")
        assert not hasattr(request, "heldout_backward_targets")
        assert request.train_forward_targets.shape == request.train_task_indices.shape
        assert request.train_backward_targets.shape == request.train_task_indices.shape
        return UtilityOOFSeedOutput(
            np.full(len(request.heldout_task_indices), request.seed / 1000.0)
        )

    utility_folds = np.tile(
        np.asarray([0, 0, 1, 1], dtype=np.int32), (len(EXPECTED_SEEDS), 1)
    )
    utility_artifact = tmp_path / "utility-oof.npz"
    utility_receipt = tmp_path / "utility-oof-receipt.json"
    produced_utility = produce_utility_oof_scores(
        producer=producer,
        fit_map=fit_map,
        fit_preflight_receipt_path=receipt_path,
        expected_fit_preflight_receipt_sha256=preflight.receipt_sha256,
        feature_schema_sha256=_sha("utility-features"),
        model_spec_sha256=_sha("utility-model"),
        utility_oof_folds=2,
        fold_by_seed_task=utility_folds,
        artifact_path=utility_artifact,
        producer_receipt_path=utility_receipt,
        seed_callback=utility_callback,
    )
    assert len(utility_requests) == len(EXPECTED_SEEDS) * 2
    assert "selection" not in inspect.signature(produce_utility_oof_scores).parameters
    utility_summary = validate_utility_oof_files(
        artifact_path=utility_artifact,
        producer_path=producer_path,
    )
    assert utility_summary["status"] == "valid_private_fit_scores_not_performance_evidence"
    assert produced_utility.artifact_sha256 == utility_summary["artifact_sha256"]
    _assert_public_receipt_is_aggregate_only(utility_receipt, tmp_path)
