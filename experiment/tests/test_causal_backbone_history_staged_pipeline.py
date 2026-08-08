from __future__ import annotations

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="history staged pipeline requires PyTorch")

import hva_affect.causal_backbone_history_staged_pipeline as pipeline
from hva_affect.causal_backbone_evidence_runner import (
    EXPECTED_SEEDS,
    FIT_ROLE,
    SELECTION_ROLE,
    run_fit_preflight,
)
from hva_affect.causal_backbone_evidence_stage_b import write_fit_protocol_map
from hva_affect.causal_multimodal_backbone import CausalBackboneConfig
from hva_affect.emotiontalk_causal_backbone_runner import (
    BackboneRunConfig,
    PlannedCheckpointInterruption,
    UtilitySamplingConfig,
    _runtime_environment,
    fit_fold_text_processor,
)
from test_causal_backbone_evidence_runner import (
    ENVIRONMENT,
    _file_sha,
    _lineage_files,
    _make_emotiontalk_sidecars,
    _sha,
    _write_npz,
)


def _make_history_sidecars(root: Path) -> Path:
    manifest_path = _make_emotiontalk_sidecars(root, poison_selection=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for role, protocol_offset in ((FIT_ROLE, 0), (SELECTION_ROLE, 16)):
        feature_path = root / f"features_{role}.npz"
        label_path = root / f"labels_{role}.npz"
        with np.load(feature_path, allow_pickle=False) as archive:
            feature = {name: np.asarray(archive[name]) for name in archive.files}
        with np.load(label_path, allow_pickle=False) as archive:
            label = {name: np.asarray(archive[name]) for name in archive.files}
        rows = 16
        groups = np.repeat(np.arange(4), 4)
        speakers = np.repeat(np.arange(4), 4)
        turns = np.tile(np.arange(4), 4).astype(np.int64)
        feature.update(
            {
                "opaque_row_hashes": np.asarray(
                    [_sha(f"{role}-history-row-{index}") for index in range(rows)]
                ),
                "opaque_group_hashes": np.asarray(
                    [_sha(f"{role}-history-group-{value}") for value in groups]
                ),
                "speaker_tokens": np.asarray(
                    [f"{role}-speaker-{value}" for value in speakers]
                ),
                "turn_ids": turns,
                "protocol_row_ids": np.arange(
                    protocol_offset, protocol_offset + rows, dtype=np.int64
                ),
                "role_buckets": np.full(
                    rows, 10 if role == FIT_ROLE else 65, dtype=np.int16
                ),
                "texts": np.asarray(
                    [f"history aware {role} utterance {index}" for index in range(rows)]
                ),
                "audio_features": (
                    np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) / 10.0
                ),
                "video_features": (
                    np.arange(rows * 2, dtype=np.float32).reshape(rows, 2) / 10.0
                ),
            }
        )
        label["labels"] = np.arange(rows, dtype=np.int64) % 7
        _write_npz(feature_path, feature)
        _write_npz(label_path, label)
        manifest["roles"][role].update(
            {
                "rows": rows,
                "groups": 4,
                "history_eligible_rows": 12,
                "feature_sha256": _file_sha(feature_path),
                "label_sha256": _file_sha(label_path),
            }
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _configs() -> tuple[CausalBackboneConfig, BackboneRunConfig, UtilitySamplingConfig]:
    model = CausalBackboneConfig(
        text_dim=4,
        audio_dim=3,
        video_dim=2,
        d_model=8,
        num_heads=2,
        num_layers=1,
        ffn_dim=16,
        num_speakers=64,
        max_turns=32,
        max_relative_turn=8,
        num_classes=7,
        dropout=0.0,
        parameter_limit=100_000,
    )
    run = BackboneRunConfig(
        outer_folds=2,
        inner_validation_fraction=0.25,
        max_epochs=1,
        early_stopping_patience=1,
        batch_size=2,
        inference_batch_size=8,
        gradient_accumulation_steps=1,
        max_history_items=8,
        use_amp=False,
        text_min_df=1,
        text_max_df=1.0,
        text_max_features=64,
        text_svd_n_iter=2,
    )
    utility = UtilitySamplingConfig(
        draws_per_query=1,
        maximum_candidates=4,
        seed=2718,
        match_context_cardinality=True,
    )
    return model, run, utility


def _fixture(root: Path) -> dict[str, object]:
    manifest_path = _make_history_sidecars(root)
    configs, code = _lineage_files(root)
    preflight = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=root,
        manifest_path=manifest_path,
        receipt_path=root / "fit-preflight.json",
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    fit_map = write_fit_protocol_map(
        preflight.fit,
        receipt_path=preflight.receipt_path,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=root / "fit-map.npz",
    )
    lineage = preflight.receipt["lineage"]
    model, run, utility = _configs()
    execution_environment = _runtime_environment(torch.device("cpu"))
    return {
        "manifest": manifest_path,
        "configs": configs,
        "code": code,
        "preflight": preflight,
        "fit_map": fit_map,
        "config_sha": lineage["config_sha256"],
        "code_sha": lineage["code_sha256"],
        "runtime_sha": lineage["runtime_environment_sha256"],
        "execution_environment": execution_environment,
        "execution_runtime_sha": pipeline._canonical_sha256(  # noqa: SLF001
            execution_environment
        ),
        "model": model,
        "run": run,
        "utility": utility,
    }


def _context_probability(rows: int, contexts: int, classes: int) -> np.ndarray:
    result = np.full((rows, contexts, classes), 0.5 / (classes - 1), dtype=np.float32)
    for context in range(contexts):
        first = 0.2 + 0.1 * context
        result[:, context, 0] = first
        result[:, context, 1:] = (1.0 - first) / (classes - 1)
    return result


def _produce(
    root: Path,
    values: dict[str, object],
    *,
    checkpoint_status: str = "complete",
    checkpoint_attack: str | None = None,
    corrupt_processor: bool = False,
):
    checkpoint_root = root / "history-checkpoints"

    def callback(request: pipeline.HistoryFitFoldRequest):
        assert request.heldout_labels_materialized is False
        assert request.heldout_targets_materialized is False
        assert "heldout_labels" not in request.__dict__
        assert "heldout_targets" not in request.__dict__
        corpus = pipeline._corpus_from_fold_request(  # noqa: SLF001 - contract test
            request, model_config=values["model"]
        )
        assert np.all(corpus.labels[request.heldout_indices] == 0)
        run_dir = (
            request.checkpoint_root
            / f"seed_{request.seed:05d}"
            / f"fold_{request.fold:02d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        split = pipeline._split_from_outer_partition(  # noqa: SLF001
            corpus,
            outer_train=request.train_indices,
            heldout=request.heldout_indices,
            validation_fraction=values["run"].inner_validation_fraction,
            seed=request.seed,
            fold=request.fold,
        )
        processor = fit_fold_text_processor(
            corpus.texts,
            split.inner_train_indices,
            output_dim=values["model"].text_dim,
            config=values["run"],
            seed=request.seed,
        )
        processor_identity = pipeline._canonical_sha256(  # noqa: SLF001
            {
                "source_identity": request.source_identity_sha256,
                "fold": request.fold,
                "seed": request.seed,
                "inner_train_indices_sha256": pipeline._indices_sha256(  # noqa: SLF001
                    split.inner_train_indices
                ),
                "text_dim": values["model"].text_dim,
                "text_settings": {
                    name: value
                    for name, value in asdict(values["run"]).items()
                    if name.startswith("text_")
                },
            }
        )
        processor_path = run_dir / "text_processor.joblib"
        joblib.dump(
            {
                "schema_version": "fold_local_text_svd_v1",
                "identity_sha256": processor_identity,
                "processor": processor,
            },
            processor_path,
            compress=3,
        )
        if corrupt_processor:
            processor_path.write_bytes(b"not-a-joblib-processor")
        checkpoint_identity = pipeline._canonical_sha256(  # noqa: SLF001
            {
                "source_identity": request.source_identity_sha256,
                "seed": request.seed,
                "fold": request.fold,
                "inner_train": pipeline._indices_sha256(split.inner_train_indices),  # noqa: SLF001
                "inner_validation": pipeline._indices_sha256(  # noqa: SLF001
                    split.inner_validation_indices
                ),
                "outer_heldout": pipeline._indices_sha256(  # noqa: SLF001
                    split.outer_heldout_indices
                ),
                "processor_sha256": pipeline._file_sha256(processor_path),  # noqa: SLF001
                "model_config": asdict(values["model"]),
                "run_config": asdict(values["run"]),
            }
        )
        model = pipeline.CausalMultimodalBackbone(values["model"])
        state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=values["run"].learning_rate,
            weight_decay=values["run"].weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        checkpoint_payload = {
                "schema_version": "causal_backbone_atomic_checkpoint_v2",
                "status": checkpoint_status,
                "identity_sha256": checkpoint_identity,
                "epoch": 0,
                "best_epoch": 0,
                "best_validation_nll": 1.0,
                "model_state": state,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_model_state": state,
                "bad_epochs": 0,
                "early_stopped": False,
                "peak_cuda_mib": 0.0,
                "rng_state": pipeline._capture_rng_state(),  # noqa: SLF001
            }
        if checkpoint_attack == "empty_state":
            checkpoint_payload["model_state"] = {}
            checkpoint_payload["best_model_state"] = {}
        elif checkpoint_attack == "wrong_identity":
            checkpoint_payload["identity_sha256"] = "f" * 64
        torch.save(checkpoint_payload, run_dir / "checkpoint.pt")
        return pipeline.HistoryFitFoldOutput(
            endpoint_probability=_context_probability(
                len(request.heldout_indices), 2, 7
            ),
            utility_probability=_context_probability(
                len(request.heldout_tasks), 4, 7
            ),
            source_identity_sha256=request.source_identity_sha256,
        )

    produced = pipeline.produce_history_fit_only(
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        checkpoint_root=checkpoint_root,
        outcome_artifact_path=root / "history-fit-outcome.npz",
        targets_artifact_path=root / "history-fit-targets.npz",
        producer_receipt_path=root / "history-fit-receipt.json",
        model_config=values["model"],
        run_config=values["run"],
        utility_config=values["utility"],
        config_sha256=values["config_sha"],
        code_sha256=values["code_sha"],
        runtime_environment_sha256=values["runtime_sha"],
        execution_environment_sha256=values.get(
            "execution_runtime_sha", values["runtime_sha"]
        ),
        device=torch.device("cpu"),
        fold_callback=callback,
    )
    return produced, checkpoint_root


def _verify(values: dict[str, object], produced, checkpoint_root: Path):
    return pipeline.verify_history_fit_for_completion(
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        fit_outcome_artifact_path=produced.outcome_artifact_path,
        expected_fit_outcome_artifact_sha256=produced.outcome_artifact_sha256,
        fit_targets_artifact_path=produced.targets_artifact_path,
        expected_fit_targets_artifact_sha256=produced.targets_artifact_sha256,
        fit_producer_receipt_path=produced.receipt_path,
        expected_fit_producer_receipt_sha256=produced.receipt_sha256,
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        checkpoint_root=checkpoint_root,
        model_config=values["model"],
        run_config=values["run"],
        utility_config=values["utility"],
        config_sha256=values["config_sha"],
        code_sha256=values["code_sha"],
        runtime_environment_sha256=values["runtime_sha"],
        execution_environment_sha256=values.get(
            "execution_runtime_sha", values["runtime_sha"]
        ),
    )


def _selection(
    values: dict[str, object],
    root: Path,
    state: pipeline.VerifiedHistoryFitState,
    checkpoint_root: Path,
):
    return pipeline.materialize_history_selection_features_after_fit_gate(
        fit=values["preflight"].fit,
        fit_state=state,
        checkpoint_root=checkpoint_root,
        model_config=values["model"],
        run_config=values["run"],
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        dataset="EmotionTalk",
        sidecar_dir=root,
        manifest_path=values["manifest"],
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
        execution_environment=values.get("execution_environment", ENVIRONMENT),
    )


def _complete(
    root: Path,
    values: dict[str, object],
    state: pipeline.VerifiedHistoryFitState,
    checkpoint_root: Path,
    selection,
    *,
    output_root: Path | None = None,
):
    destination_root = root if output_root is None else output_root
    return pipeline.complete_history_selection_outcomes(
        fit=values["preflight"].fit,
        selection=selection,
        fit_map=values["fit_map"],
        fit_state=state,
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        checkpoint_root=checkpoint_root,
        model_config=values["model"],
        run_config=values["run"],
        utility_config=values["utility"],
        config_sha256=values["config_sha"],
        code_sha256=values["code_sha"],
        runtime_environment_sha256=values["runtime_sha"],
        execution_environment_sha256=values.get(
            "execution_runtime_sha", values["runtime_sha"]
        ),
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
        execution_environment=values.get("execution_environment", ENVIRONMENT),
        sidecar_dir=root,
        manifest_path=values["manifest"],
        selection_feature_sha256=state.selection_feature_sha256,
        artifact_path=destination_root / "history-complete-outcome.npz",
        completion_receipt_path=destination_root / "history-complete-receipt.json",
        device=torch.device("cpu"),
    )


def test_fit_api_has_no_selection_capability_and_fold_request_has_no_heldout_outcome() -> None:
    parameters = inspect.signature(pipeline.produce_history_fit_only).parameters
    assert not any("selection" in name for name in parameters)
    fields = pipeline.HistoryFitFoldRequest.__dataclass_fields__
    assert "heldout_labels" not in fields
    assert "heldout_targets" not in fields


def test_history_fit_rejects_vad_order_mismatch_before_callback(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    values["execution_environment"] = _runtime_environment(torch.device("cpu"))
    values["execution_runtime_sha"] = pipeline._canonical_sha256(  # noqa: SLF001
        values["execution_environment"]
    )
    wrong_order = (
        "neutral",
        "surprise",
        "fear",
        "sadness",
        "joy",
        "disgust",
        "anger",
    )
    relation_model = replace(
        values["model"],
        affect_relation_mode="primary_history_relation",
        affect_relation_use_vad_features=True,
        auxiliary_vad_weight=0.1,
        emotion_label_order=wrong_order,
    )
    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="VAD supervision label order",
    ):
        pipeline.produce_history_fit_only(
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            fit_preflight_receipt_path=values["preflight"].receipt_path,
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            checkpoint_root=tmp_path / "never-created-checkpoints",
            outcome_artifact_path=tmp_path / "never-created-outcome.npz",
            targets_artifact_path=tmp_path / "never-created-targets.npz",
            producer_receipt_path=tmp_path / "never-created-receipt.json",
            model_config=relation_model,
            run_config=values["run"],
            utility_config=values["utility"],
            config_sha256=values["config_sha"],
            code_sha256=values["code_sha"],
            runtime_environment_sha256=values["runtime_sha"],
            device=torch.device("cpu"),
            fold_callback=lambda _request: pytest.fail("callback ran after VAD mismatch"),
        )


def test_history_fit_rehashes_live_fit_arrays_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    values["preflight"].fit.audio[0, 0] += 1.0
    monkeypatch.setattr(
        pipeline,
        "make_real_history_fit_fold_callback",
        lambda **_kwargs: pytest.fail("trainer created after live fit mutation"),
    )
    with pytest.raises(
        pipeline.HistoryStagedPipelineError, match="live fit arrays"
    ):
        pipeline.produce_history_fit_only(
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            fit_preflight_receipt_path=values["preflight"].receipt_path,
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            checkpoint_root=tmp_path / "unused" / "checkpoints",
            outcome_artifact_path=tmp_path / "unused" / "history-fit-outcome.npz",
            targets_artifact_path=tmp_path / "unused" / "history-fit-targets.npz",
            producer_receipt_path=tmp_path / "unused" / "history-fit-receipt.json",
            model_config=values["model"],
            run_config=values["run"],
            utility_config=values["utility"],
            config_sha256=values["config_sha"],
            code_sha256=values["code_sha"],
            runtime_environment_sha256=values["runtime_sha"],
            device=torch.device("cpu"),
            fold_callback=lambda _request: pytest.fail(
                "callback ran after live fit mutation"
            ),
        )


def test_history_fit_refuses_empty_utility_task_set_before_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    monkeypatch.setattr(pipeline, "_tasks_for_role", lambda *_args: ())
    with pytest.raises(pipeline.HistoryStagedPipelineError, match="non-empty"):
        pipeline.produce_history_fit_only(
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            fit_preflight_receipt_path=values["preflight"].receipt_path,
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            checkpoint_root=tmp_path / "unused" / "checkpoints",
            outcome_artifact_path=tmp_path / "unused" / "history-fit-outcome.npz",
            targets_artifact_path=tmp_path / "unused" / "history-fit-targets.npz",
            producer_receipt_path=tmp_path / "unused" / "history-fit-receipt.json",
            model_config=values["model"],
            run_config=values["run"],
            utility_config=values["utility"],
            config_sha256=values["config_sha"],
            code_sha256=values["code_sha"],
            runtime_environment_sha256=values["runtime_sha"],
            device=torch.device("cpu"),
            fold_callback=lambda _request: pytest.fail("callback ran for empty tasks"),
        )


def test_fit_outputs_are_write_once_aggregate_only_and_physically_separated(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    produced, _ = _produce(tmp_path, values)
    target_view = pipeline.load_history_fit_targets_view(
        produced.targets_artifact_path,
        expected_fit_outcome_sha256=produced.outcome_artifact_sha256,
        expected_source_identity_sha256=produced.source_identity_sha256,
    )
    assert target_view.forward_utility.shape[0] == len(EXPECTED_SEEDS)
    assert target_view.forward_utility.shape[1] > 0
    with np.load(produced.outcome_artifact_path, allow_pickle=False) as archive:
        outcome_keys = set(archive.files)
    with np.load(produced.targets_artifact_path, allow_pickle=False) as archive:
        target_keys = set(archive.files)
    assert not any(
        token in name
        for name in outcome_keys
        for token in ("forward_utility", "backward_utility", "asymmetry", "sign_agreement")
    )
    assert not any("probability" in name or "labels" in name for name in target_keys)
    receipt = json.loads(produced.receipt_path.read_text(encoding="utf-8"))
    serialized = json.dumps(receipt)
    assert receipt["public_artifact_policy"]["aggregate_only"] is True
    assert receipt["training_contract"]["selection_payload_consumed"] is False
    assert produced.production_trainer is False
    assert receipt["status"] == pipeline.SYNTHETIC_FIT_STATUS
    assert receipt["training_contract"]["production_receipt"] is False
    assert receipt["training_contract"]["trainer_mode"] == pipeline.SYNTHETIC_TRAINER_MODE
    assert str(tmp_path) not in serialized
    assert '"probabilities"' not in serialized
    assert '"labels"' not in serialized
    with pytest.raises(FileExistsError):
        pipeline._write_json_once(produced.receipt_path, receipt)  # noqa: SLF001


def test_npz_write_once_is_atomic_under_concurrent_publish(tmp_path: Path) -> None:
    destination = tmp_path / "concurrent.npz"
    barrier = threading.Barrier(2)

    def publish(value: int):
        barrier.wait()
        try:
            pipeline._write_npz_once(  # noqa: SLF001
                destination, {"value": np.asarray(value, dtype=np.int64)}
            )
            return "written"
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (1, 2)))
    assert sorted(results) == ["refused", "written"]
    with np.load(destination, allow_pickle=False) as archive:
        assert int(archive["value"]) in {1, 2}
    assert not list(tmp_path.glob(".concurrent.npz.*.tmp"))


def test_private_root_is_external_new_and_non_reusable(tmp_path: Path) -> None:
    claimed = pipeline.claim_new_history_private_root(tmp_path / "fresh-private-run")
    assert claimed.is_dir()
    with pytest.raises(FileExistsError, match="all-new"):
        pipeline.claim_new_history_private_root(claimed)
    with pytest.raises(pipeline.HistoryStagedPipelineError, match="outside"):
        pipeline.claim_new_history_private_root(
            pipeline._canonical_repository_root() / "forbidden-private-run"  # noqa: SLF001
        )


def test_private_root_resume_requires_the_same_lineage_claim(tmp_path: Path) -> None:
    destination = tmp_path / "resumable-private-run"
    claimed = pipeline.claim_or_resume_history_private_root(
        destination,
        production_claim_sha256="a" * 64,
        allow_resume=False,
    )
    (claimed / "checkpoints").mkdir()
    assert pipeline.claim_or_resume_history_private_root(
        destination,
        production_claim_sha256="a" * 64,
        allow_resume=True,
    ) == claimed
    with pytest.raises(
        pipeline.HistoryStagedPipelineError, match="lineage claim changed"
    ):
        pipeline.claim_or_resume_history_private_root(
            destination,
            production_claim_sha256="b" * 64,
            allow_resume=True,
        )


def test_private_run_lock_refuses_a_concurrent_fit_and_auto_releases(
    tmp_path: Path,
) -> None:
    root = pipeline.claim_new_history_private_root(tmp_path / "locked-private-run")
    with pipeline.history_private_run_lock(root):
        with pytest.raises(
            pipeline.HistoryStagedPipelineError, match="holds the private run lock"
        ):
            with pipeline.history_private_run_lock(root):
                pytest.fail("concurrent lock was acquired")
    with pipeline.history_private_run_lock(root):
        pass


@pytest.mark.parametrize(
    ("checkpoint_attack", "corrupt_processor", "message"),
    (
        ("empty_state", False, "state is incomplete"),
        ("wrong_identity", False, "identity differs"),
        (None, True, "cannot deserialize fold text processor"),
    ),
)
def test_semantic_checkpoint_gate_rejects_attack_before_selection_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_attack: str | None,
    corrupt_processor: bool,
    message: str,
) -> None:
    values = _fixture(tmp_path)
    selection_feature = (tmp_path / f"features_{SELECTION_ROLE}.npz").resolve()
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() == selection_feature:
            raise AssertionError("selection feature opened before semantic checkpoint gate")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    with pytest.raises(pipeline.HistoryStagedPipelineError, match=message):
        _produce(
            tmp_path,
            values,
            checkpoint_attack=checkpoint_attack,
            corrupt_processor=corrupt_processor,
        )


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_fit_gate_rejects_missing_or_corrupt_checkpoint(
    tmp_path: Path, mutation: str
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    checkpoint = checkpoint_root / "seed_00017" / "fold_00" / "checkpoint.pt"
    if mutation == "missing":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"replaced")
    with pytest.raises((OSError, ValueError, pipeline.HistoryStagedPipelineError)):
        _verify(values, produced, checkpoint_root)


def test_fit_producer_rejects_partial_checkpoint_without_touching_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    forbidden = {
        (tmp_path / f"features_{SELECTION_ROLE}.npz").resolve(),
        (tmp_path / f"labels_{SELECTION_ROLE}.npz").resolve(),
    }
    original_hash = pipeline._file_sha256  # noqa: SLF001

    def guarded_hash(path):
        if Path(path).resolve() in forbidden:
            raise AssertionError("fit producer touched a selection payload")
        return original_hash(path)

    monkeypatch.setattr(pipeline, "_file_sha256", guarded_hash)
    with pytest.raises(
        pipeline.HistoryStagedPipelineError, match="partial or unknown"
    ):
        _produce(tmp_path, values, checkpoint_status="partial")


def test_feature_gate_opens_no_selection_label_and_binds_normalized_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection_label = (tmp_path / f"labels_{SELECTION_ROLE}.npz").resolve()
    selection_label.unlink()
    original_hash = pipeline._file_sha256  # noqa: SLF001
    original_load = np.load

    def guarded_hash(path):
        if Path(path).resolve() == selection_label:
            raise AssertionError("feature gate hashed the selection label")
        return original_hash(path)

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() == selection_label:
            raise AssertionError("feature gate deserialized the selection label")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_file_sha256", guarded_hash)
    monkeypatch.setattr(np, "load", guarded_load)
    verified = _selection(values, tmp_path, state, checkpoint_root)
    assert verified.view.labels_materialized is False
    assert verified.feature_file_sha256 == state.selection_feature_sha256
    assert verified.normalized_feature_sha256 == (
        pipeline._selection_feature_contract_sha256(verified.view)  # noqa: SLF001
    )


def test_feature_gate_rehashes_live_config_before_selection_npz_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    config_path = Path(next(iter(values["configs"].values())))
    config_path.write_text("changed after fit gate", encoding="utf-8")
    selection_feature = (tmp_path / f"features_{SELECTION_ROLE}.npz").resolve()
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() == selection_feature:
            raise AssertionError("selection feature opened before live lineage rehash")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    with pytest.raises(ValueError, match="lineage changed"):
        _selection(values, tmp_path, state, checkpoint_root)


def test_histories_reject_future_out_of_bounds_cross_group_and_cross_speaker() -> None:
    groups = np.asarray(["g", "g", "g", "other", "g"])
    speakers = np.asarray(["s", "s", "s", "s", "other"])
    turns = np.asarray([0, 1, 2, 0, 0], dtype=np.int64)
    valid = ((), (0,), (0, 1), (), ())
    assert pipeline.validate_strict_past_histories(
        groups=groups, speakers=speakers, turns=turns, histories=valid
    ) == valid
    attacks = (
        ((), (2,), (0, 1), (), ()),
        ((), (0,), (0, 99), (), ()),
        ((), (0,), (0, 1, 3), (), ()),
        ((), (0,), (0, 1, 4), (), ()),
    )
    for attack in attacks:
        with pytest.raises(pipeline.HistoryStagedPipelineError, match="strict past"):
            pipeline.validate_strict_past_histories(
                groups=groups,
                speakers=speakers,
                turns=turns,
                histories=attack,
            )


def test_real_fit_adapter_preserves_history_and_hides_heldout_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, run, _ = _configs()
    request = pipeline.HistoryFitFoldRequest(
        dataset="EmotionTalk",
        seed=17,
        fold=0,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        heldout_indices=np.asarray([4, 5, 6, 7], dtype=np.int64),
        train_texts=("a", "b", "c", "d"),
        train_audio=np.zeros((4, 3), dtype=np.float32),
        train_video=np.zeros((4, 2), dtype=np.float32),
        train_labels=np.asarray([1, 2, 3, 4], dtype=np.int64),
        train_group_tokens=np.asarray(["g0", "g0", "g1", "g1"]),
        train_speaker_tokens=np.asarray(["s0", "s0", "s1", "s1"]),
        train_turns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        train_protocol_row_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
        train_histories=((), (0,), (), (2,)),
        heldout_texts=("e", "f", "g", "h"),
        heldout_audio=np.zeros((4, 3), dtype=np.float32),
        heldout_video=np.zeros((4, 2), dtype=np.float32),
        heldout_group_tokens=np.asarray(["g2", "g2", "g3", "g3"]),
        heldout_speaker_tokens=np.asarray(["s2", "s2", "s3", "s3"]),
        heldout_turns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        heldout_protocol_row_ids=np.asarray([4, 5, 6, 7], dtype=np.int64),
        heldout_histories=((), (4,), (), (6,)),
        heldout_tasks=(),
        heldout_labels_materialized=False,
        heldout_targets_materialized=False,
        source_identity_sha256="a" * 64,
        checkpoint_root=Path("unused"),
    )
    calls: list[dict[str, object]] = []

    def trainer(corpus, split, **kwargs):
        calls.append(kwargs)
        assert kwargs["require_complete_checkpoint"] is False
        assert corpus.labels[split.outer_heldout_indices].tolist() == [0, 0, 0, 0]
        assert any(corpus.histories)
        assert np.all(corpus.speaker_ids[split.outer_heldout_indices] == 0)
        return SimpleNamespace(model=object(), text_features=np.zeros((8, 4)))

    def endpoint(_model, corpus, _text, queries, **_kwargs):
        assert [corpus.histories[int(value)] for value in queries] == [
            (),
            (4,),
            (),
            (6,),
        ]
        return _context_probability(len(queries), 2, 7)

    monkeypatch.setattr(pipeline, "train_one_fold_seed", trainer)
    monkeypatch.setattr(pipeline, "predict_current_and_all_history", endpoint)
    monkeypatch.setattr(
        pipeline,
        "predict_utility_contexts",
        lambda _m, _c, _t, tasks, **_k: _context_probability(len(tasks), 4, 7),
    )
    callback = pipeline.make_real_history_fit_fold_callback(
        model_config=model, run_config=run, device=torch.device("cpu")
    )
    output = callback(request)
    assert output.endpoint_probability.shape == (4, 2, 7)
    assert output.utility_probability.shape == (0, 4, 7)
    assert len(calls) == 1


@pytest.mark.parametrize("use_vad", [True, False], ids=["valid-vad", "no-vad"])
def test_real_tiny_train_strict_restore_and_selection_inference_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_vad: bool
) -> None:
    """Exercise the actual trainer/processor/checkpoint path without adapters."""

    values = _fixture(tmp_path)
    values["model"] = replace(
        values["model"],
        affect_relation_mode="primary_history_relation",
        affect_relation_hidden_dim=8,
        affect_relation_use_vad_features=use_vad,
        auxiliary_vad_weight=0.1 if use_vad else 0.0,
        emotion_label_order=(
            tuple(values["preflight"].fit.label_order) if use_vad else ()
        ),
    )
    production_claim = pipeline.history_production_claim_sha256(
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        model_config=values["model"],
        run_config=values["run"],
        utility_config=values["utility"],
        config_sha256=values["config_sha"],
        code_sha256=values["code_sha"],
        runtime_environment_sha256=values["runtime_sha"],
        execution_environment_sha256=values["execution_runtime_sha"],
    )
    private_root = pipeline.claim_or_resume_history_private_root(
        tmp_path / ("real-vad-private" if use_vad else "real-no-vad-private"),
        production_claim_sha256=production_claim,
        allow_resume=False,
    )
    checkpoint_root = private_root / "checkpoints"
    production_kwargs = {
        "fit": values["preflight"].fit,
        "fit_map": values["fit_map"],
        "fit_preflight_receipt_path": values["preflight"].receipt_path,
        "expected_fit_preflight_receipt_sha256": values["preflight"].receipt_sha256,
        "checkpoint_root": checkpoint_root,
        "outcome_artifact_path": private_root / "history-fit-outcome.npz",
        "targets_artifact_path": private_root / "history-fit-targets.npz",
        "producer_receipt_path": private_root / "history-fit-receipt.json",
        "model_config": values["model"],
        "run_config": values["run"],
        "utility_config": values["utility"],
        "config_sha256": values["config_sha"],
        "code_sha256": values["code_sha"],
        "runtime_environment_sha256": values["runtime_sha"],
        "execution_environment_sha256": values["execution_runtime_sha"],
        "device": torch.device("cpu"),
        "production_run_claim_sha256": production_claim,
    }
    if not use_vad:
        real_train = pipeline.train_one_fold_seed
        interrupted = False

        def interrupt_first_fold(*args, **kwargs):
            nonlocal interrupted
            if not interrupted and not kwargs.get("require_complete_checkpoint", False):
                interrupted = True
                kwargs["test_interrupt_after_epoch"] = 0
            return real_train(*args, **kwargs)

        monkeypatch.setattr(pipeline, "train_one_fold_seed", interrupt_first_fold)
        with pytest.raises(PlannedCheckpointInterruption):
            pipeline.produce_history_fit_only(**production_kwargs)
        assert interrupted
        pipeline.claim_or_resume_history_private_root(
            private_root,
            production_claim_sha256=production_claim,
            allow_resume=True,
        )
        monkeypatch.setattr(pipeline, "train_one_fold_seed", real_train)
    produced = pipeline.produce_history_fit_only(**production_kwargs)
    if not use_vad:
        fresh_root = pipeline.claim_or_resume_history_private_root(
            tmp_path / "real-no-vad-uninterrupted-private",
            production_claim_sha256=production_claim,
            allow_resume=False,
        )
        fresh_kwargs = dict(production_kwargs)
        fresh_kwargs.update(
            {
                "checkpoint_root": fresh_root / "checkpoints",
                "outcome_artifact_path": fresh_root / "history-fit-outcome.npz",
                "targets_artifact_path": fresh_root / "history-fit-targets.npz",
                "producer_receipt_path": fresh_root / "history-fit-receipt.json",
            }
        )
        uninterrupted = pipeline.produce_history_fit_only(**fresh_kwargs)
        with np.load(produced.outcome_artifact_path, allow_pickle=False) as resumed_npz:
            with np.load(
                uninterrupted.outcome_artifact_path, allow_pickle=False
            ) as uninterrupted_npz:
                for name in (
                    "fit_endpoint_probability_oof",
                    "fit_utility_probability_oof",
                    "fit_fold_by_seed_query",
                ):
                    np.testing.assert_array_equal(
                        resumed_npz[name], uninterrupted_npz[name]
                    )
    assert produced.production_trainer is True
    fit_receipt = json.loads(produced.receipt_path.read_text(encoding="utf-8"))
    assert fit_receipt["status"] == pipeline.PRODUCTION_FIT_STATUS
    assert fit_receipt["training_contract"]["production_receipt"] is True
    state = _verify(values, produced, checkpoint_root)
    assert state.production_trainer is True
    selection = _selection(values, tmp_path, state, checkpoint_root)
    completed = _complete(
        tmp_path,
        values,
        state,
        checkpoint_root,
        selection,
        output_root=private_root,
    )
    assert completed.artifact_path.is_file()
    complete_receipt = json.loads(
        completed.receipt_path.read_text(encoding="utf-8")
    )
    assert complete_receipt["completion_contract"]["production_receipt"] is True
    assert (
        complete_receipt["completion_contract"]["selection_label_deserialized"]
        is False
    )
    selection_feature_path = (
        tmp_path / f"features_{SELECTION_ROLE}.npz"
    ).resolve()
    selection_label_path = (
        tmp_path / f"labels_{SELECTION_ROLE}.npz"
    ).resolve()
    original_np_load = np.load

    def reject_raw_selection(path, *args, **kwargs):
        if Path(path).resolve() in {selection_feature_path, selection_label_path}:
            raise AssertionError(
                "production attestation opened a raw selection sidecar"
            )
        return original_np_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", reject_raw_selection)
    attestation = pipeline.verify_history_completion_production_attestation(
        completed.artifact_path,
        completed.receipt_path,
        completed.receipt_sha256,
    )
    assert attestation.dataset == "EmotionTalk"
    assert attestation.artifact_sha256 == completed.artifact_sha256
    assert attestation.completion_receipt_sha256 == completed.receipt_sha256
    assert attestation.fit_producer_receipt_sha256 == produced.receipt_sha256
    assert attestation.production_run_claim_sha256 == production_claim
    assert dict(attestation.config_sha256) == values["config_sha"]
    assert dict(attestation.code_sha256) == values["code_sha"]
    assert (
        attestation.fit_preflight_receipt_sha256
        == values["preflight"].receipt_sha256
    )
    assert (
        attestation.execution_environment_sha256
        == values["execution_runtime_sha"]
    )

    # One production run is enough to exercise the adversarial verifier matrix.
    if use_vad:
        return

    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="completion receipt file hash changed",
    ):
        pipeline.verify_history_completion_production_attestation(
            completed.artifact_path,
            completed.receipt_path,
            "f" * 64,
        )

    original_completion_receipt = completed.receipt_path.read_bytes()
    for section, field, replacement in (
        (None, "status", pipeline.SYNTHETIC_COMPLETION_STATUS),
        ("completion_contract", "production_receipt", False),
        (
            "completion_contract",
            "trainer_mode",
            pipeline.SYNTHETIC_TRAINER_MODE,
        ),
    ):
        payload = json.loads(original_completion_receipt.decode("utf-8"))
        target = payload if section is None else payload[section]
        target[field] = replacement
        completed.receipt_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        tampered_receipt_sha = _file_sha(completed.receipt_path)
        with pytest.raises(pipeline.HistoryStagedPipelineError):
            pipeline.verify_history_completion_production_attestation(
                completed.artifact_path,
                completed.receipt_path,
                tampered_receipt_sha,
            )
        completed.receipt_path.write_bytes(original_completion_receipt)

    original_artifact = completed.artifact_path.read_bytes()
    completed.artifact_path.write_bytes(original_artifact + b"tampered-completion")
    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="completion artifact hash changed",
    ):
        pipeline.verify_history_completion_production_attestation(
            completed.artifact_path,
            completed.receipt_path,
            completed.receipt_sha256,
        )
    completed.artifact_path.write_bytes(original_artifact)

    checkpoint_path = sorted(checkpoint_root.rglob("checkpoint.pt"))[0]
    original_checkpoint = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(original_checkpoint + b"tampered-checkpoint")
    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="checkpoint manifest changed",
    ):
        pipeline.verify_history_completion_production_attestation(
            completed.artifact_path,
            completed.receipt_path,
            completed.receipt_sha256,
        )
    checkpoint_path.write_bytes(original_checkpoint)

    claim_path = private_root / "history-run-claim.json"
    original_claim = claim_path.read_bytes()
    claim_payload = json.loads(original_claim.decode("utf-8"))
    claim_payload["production_claim_sha256"] = "0" * 64
    claim_path.write_text(
        json.dumps(claim_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="production history claim lineage changed",
    ):
        pipeline.verify_history_completion_production_attestation(
            completed.artifact_path,
            completed.receipt_path,
            completed.receipt_sha256,
        )
    claim_path.write_bytes(original_claim)

    original_fit_receipt = produced.receipt_path.read_bytes()
    fit_payload = json.loads(original_fit_receipt.decode("utf-8"))
    fit_payload["lineage"]["production_run_claim_sha256"] = "0" * 64
    produced.receipt_path.write_text(
        json.dumps(
            fit_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tampered_fit_receipt_sha = _file_sha(produced.receipt_path)
    completion_payload = json.loads(original_completion_receipt.decode("utf-8"))
    completion_payload["lineage"][
        "fit_producer_receipt_sha256"
    ] = tampered_fit_receipt_sha
    completed.receipt_path.write_text(
        json.dumps(
            completion_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    synchronized_completion_sha = _file_sha(completed.receipt_path)
    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="fit receipt lineage changed",
    ):
        pipeline.verify_history_completion_production_attestation(
            completed.artifact_path,
            completed.receipt_path,
            synchronized_completion_sha,
        )
    produced.receipt_path.write_bytes(original_fit_receipt)
    completed.receipt_path.write_bytes(original_completion_receipt)


def test_synthetic_completion_with_canonical_names_is_not_production_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)

    class Processor:
        def transform(self, texts):
            return np.zeros(
                (len(texts), values["model"].text_dim), dtype=np.float32
            )

    monkeypatch.setattr(
        pipeline,
        "train_one_fold_seed",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=object(),
            processor=Processor(),
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "predict_current_and_all_history",
        lambda _m, _c, _t, queries, **_kwargs: _context_probability(
            len(queries), 2, 7
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "predict_utility_contexts",
        lambda _m, _c, _t, tasks, **_kwargs: _context_probability(
            len(tasks), 4, 7
        ),
    )
    synthetic_root = tmp_path / "synthetic-canonical-names"
    synthetic_root.mkdir()
    completed = _complete(
        tmp_path,
        values,
        state,
        checkpoint_root,
        selection,
        output_root=synthetic_root,
    )
    receipt = json.loads(completed.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == pipeline.SYNTHETIC_COMPLETION_STATUS
    assert receipt["completion_contract"]["production_receipt"] is False
    with pytest.raises(
        pipeline.HistoryStagedPipelineError,
        match="not canonical production evidence",
    ):
        pipeline.verify_history_completion_production_attestation(
            completed.artifact_path,
            completed.receipt_path,
            completed.receipt_sha256,
        )


def test_complete_selection_is_checkpoint_only_outcome_free_and_does_not_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)
    calls: list[dict[str, object]] = []

    class Processor:
        def transform(self, texts):
            return np.zeros((len(texts), values["model"].text_dim), dtype=np.float32)

    def complete_loader(corpus, _split, **kwargs):
        calls.append(kwargs)
        if kwargs.get("require_complete_checkpoint") is not True:
            raise AssertionError("training path was enabled")
        assert corpus.label_access_mode.startswith("fit_train_labels_only")
        return SimpleNamespace(
            model=object(),
            processor=Processor(),
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
        )

    def endpoint_predict(_model, corpus, _text, queries, **_kwargs):
        assert corpus.label_access_mode.endswith("never_scored")
        assert np.all(corpus.labels == 0)
        assert any(corpus.histories)
        return _context_probability(len(queries), 2, 7)

    def utility_predict(_model, corpus, _text, tasks, **_kwargs):
        assert corpus.label_access_mode.endswith("never_scored")
        assert np.all(corpus.labels == 0)
        return _context_probability(len(tasks), 4, 7)

    monkeypatch.setattr(pipeline, "train_one_fold_seed", complete_loader)
    monkeypatch.setattr(pipeline, "predict_current_and_all_history", endpoint_predict)
    monkeypatch.setattr(pipeline, "predict_utility_contexts", utility_predict)
    completed = _complete(
        tmp_path, values, state, checkpoint_root, selection
    )
    assert len(calls) == len(EXPECTED_SEEDS) * values["run"].outer_folds
    assert all(call["require_complete_checkpoint"] is True for call in calls)
    view = pipeline.load_history_outcome_free_view(
        completed.artifact_path,
        fit=values["preflight"].fit,
        selection=selection.view,
        state=state,
    )
    assert view.selection_endpoint_probability_fold_ensemble.shape == (5, 16, 2, 7)
    assert view.selection_utility_probability_fold_ensemble.shape[0] == 5
    assert isinstance(view, pipeline.HistoryOutcomeFreeView)
    assert not isinstance(view, pipeline.HistoryFitTargetsView)
    with np.load(completed.artifact_path, allow_pickle=False) as archive:
        keys = set(archive.files)
    forbidden = (
        "forward_utility",
        "backward_utility",
        "asymmetry",
        "sign_agreement",
        "accuracy",
        "macro_f1",
        "nll",
        "brier",
    )
    assert not any(fragment in name for name in keys for fragment in forbidden)
    assert "labels" not in keys
    receipt = json.loads(completed.receipt_path.read_text(encoding="utf-8"))
    contract = receipt["completion_contract"]
    assert contract["complete_checkpoint_only"] is True
    assert contract["selection_label_deserialized"] is False
    assert contract["selection_utility_target_computed"] is False
    assert contract["performance_metric_computed"] is False
    assert str(tmp_path) not in json.dumps(receipt)


def test_complete_rejects_materialized_selection_labels_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    verified_selection = _selection(values, tmp_path, state, checkpoint_root)
    materialized = replace(verified_selection.view, labels_materialized=True)
    selection = replace(
        verified_selection,
        view=materialized,
        normalized_feature_sha256=pipeline._selection_feature_contract_sha256(  # noqa: SLF001
            materialized
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_one_fold_seed",
        lambda *_args, **_kwargs: pytest.fail("checkpoint loader ran after label violation"),
    )
    with pytest.raises(
        pipeline.HistoryStagedPipelineError, match="selection labels"
    ):
        _complete(tmp_path, values, state, checkpoint_root, selection)


def test_complete_rejects_in_memory_selection_group_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)
    selection.view.groups[0] = "tampered-group"
    monkeypatch.setattr(
        pipeline,
        "train_one_fold_seed",
        lambda *_args, **_kwargs: pytest.fail("loader ran after feature-view tampering"),
    )
    with pytest.raises(
        pipeline.HistoryStagedPipelineError, match="feature capability changed"
    ):
        _complete(tmp_path, values, state, checkpoint_root, selection)


def test_complete_rejects_live_code_change_after_feature_gate_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)
    code_path = Path(next(iter(values["code"].values())))
    code_path.write_text("changed after selection feature gate", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "train_one_fold_seed",
        lambda *_args, **_kwargs: pytest.fail("loader ran after live code mutation"),
    )
    with pytest.raises(
        (ValueError, pipeline.HistoryStagedPipelineError), match="lineage changed"
    ):
        _complete(tmp_path, values, state, checkpoint_root, selection)


@pytest.mark.parametrize("artifact", ["outcome", "targets", "receipt", "checkpoint"])
def test_complete_rejects_replacement_after_fit_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)
    if artifact == "outcome":
        state.fit_outcome_path.write_bytes(b"replaced-fit-outcome")
    elif artifact == "targets":
        state.fit_targets_path.write_bytes(b"replaced-fit-targets")
    elif artifact == "receipt":
        state.fit_receipt_path.write_bytes(b"replaced-fit-receipt")
    else:
        (
            checkpoint_root / "seed_00017" / "fold_00" / "checkpoint.pt"
        ).write_bytes(b"replaced-checkpoint")
    monkeypatch.setattr(
        pipeline,
        "train_one_fold_seed",
        lambda *_args, **_kwargs: pytest.fail("loader ran after TOCTOU replacement"),
    )
    with pytest.raises((OSError, ValueError, pipeline.HistoryStagedPipelineError)):
        _complete(tmp_path, values, state, checkpoint_root, selection)


def test_complete_detects_checkpoint_replacement_during_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)
    mutated = False

    class Processor:
        def transform(self, texts):
            return np.zeros((len(texts), values["model"].text_dim), dtype=np.float32)

    def loader(_corpus, _split, **kwargs):
        nonlocal mutated
        assert kwargs["require_complete_checkpoint"] is True
        if not mutated:
            mutated = True
            (
                checkpoint_root / "seed_00017" / "fold_00" / "checkpoint.pt"
            ).write_bytes(b"changed-during-inference")
        return SimpleNamespace(
            model=object(),
            processor=Processor(),
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
        )

    monkeypatch.setattr(pipeline, "train_one_fold_seed", loader)
    monkeypatch.setattr(
        pipeline,
        "predict_current_and_all_history",
        lambda _m, _c, _t, q, **_k: _context_probability(len(q), 2, 7),
    )
    monkeypatch.setattr(
        pipeline,
        "predict_utility_contexts",
        lambda _m, _c, _t, tasks, **_k: _context_probability(len(tasks), 4, 7),
    )
    with pytest.raises((OSError, ValueError, pipeline.HistoryStagedPipelineError)):
        _complete(tmp_path, values, state, checkpoint_root, selection)


def test_complete_never_deserializes_fit_targets_or_selection_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced, checkpoint_root = _produce(tmp_path, values)
    state = _verify(values, produced, checkpoint_root)
    selection = _selection(values, tmp_path, state, checkpoint_root)
    forbidden_paths = {
        state.fit_targets_path.resolve(),
        (tmp_path / f"labels_{SELECTION_ROLE}.npz").resolve(),
    }
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() in forbidden_paths:
            raise AssertionError("complete-selection deserialized a forbidden outcome")
        return original_load(path, *args, **kwargs)

    class Processor:
        def transform(self, texts):
            return np.zeros((len(texts), values["model"].text_dim), dtype=np.float32)

    def loader(_corpus, _split, **kwargs):
        assert kwargs["require_complete_checkpoint"] is True
        return SimpleNamespace(
            model=object(),
            processor=Processor(),
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
        )

    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(pipeline, "train_one_fold_seed", loader)
    monkeypatch.setattr(
        pipeline,
        "predict_current_and_all_history",
        lambda _m, _c, _t, q, **_k: _context_probability(len(q), 2, 7),
    )
    monkeypatch.setattr(
        pipeline,
        "predict_utility_contexts",
        lambda _m, _c, _t, tasks, **_k: _context_probability(len(tasks), 4, 7),
    )
    completed = _complete(tmp_path, values, state, checkpoint_root, selection)
    assert completed.artifact_path.is_file()
