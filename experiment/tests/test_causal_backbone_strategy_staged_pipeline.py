from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import hva_affect.causal_backbone_current_only_pipeline as current_pipeline
import hva_affect.causal_backbone_history_staged_pipeline as history_pipeline
import hva_affect.causal_backbone_strategy_staged_pipeline as strategy_pipeline
from hva_affect.bidirectional_utility_model import UtilityModelSpec
from hva_affect.causal_backbone_evidence_runner import (
    EXPECTED_SEEDS,
    FIT_ROLE,
    SELECTION_ROLE,
    materialize_selection_features_after_receipt,
)
from hva_affect.causal_backbone_evidence_stage_b import write_fit_only_lineage
from hva_affect.causal_backbone_history_staged_pipeline import (
    EncodedHistoryTasks,
    HistoryFitTargetsView,
    HistoryOutcomeFreeView,
)
from hva_affect.causal_backbone_strategy_staged_pipeline import (
    JOINT_EVALUATION_ROSTER,
    METHOD_ROSTER,
    REGISTERED_VARIANTS,
    StrategyStagedPipelineError,
    _claim_strategy_private_root,
    _infer_registered_method_probabilities,
    _strategy_artifact_mapping,
    _strategy_receipt,
    build_outcome_free_strategy_nonproduction_fixture,
    complete_strategy_selection,
    strategy_private_paths,
    validate_strategy_artifact_mapping,
)
from hva_affect.causal_multimodal_backbone import CausalBackboneConfig
from test_causal_backbone_evidence_runner import ENVIRONMENT
import test_causal_backbone_current_only_pipeline as current_test_support
import test_causal_backbone_history_staged_pipeline as history_test_support


SHA = "a" * 64


def _probability(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    value = rng.uniform(0.05, 1.0, size=shape)
    return (value / value.sum(axis=-1, keepdims=True)).astype(np.float32)


def _tasks(rows: int) -> EncodedHistoryTasks:
    query = np.arange(2, rows, dtype=np.int64)
    candidate = query - 1
    addition = tuple(() for _ in query)
    deletion = tuple((int(value), int(value) - 1) for value in candidate)
    return EncodedHistoryTasks(query, candidate, addition, deletion, SHA)


def _fixture(rows: int = 16):
    rng = np.random.default_rng(20260808)
    fit_tasks = _tasks(rows)
    selection_tasks = _tasks(rows)
    histories = tuple(tuple(range(index)) for index in range(rows))
    fit_cluster = np.repeat(np.arange(4), rows // 4).astype(np.int32)
    selection_cluster = fit_cluster.copy()
    classes = 3
    fit_utility = _probability(
        rng, (len(EXPECTED_SEEDS), len(fit_tasks), 4, classes)
    )
    selection_utility = _probability(
        rng, (len(EXPECTED_SEEDS), len(selection_tasks), 4, classes)
    )
    outcome = HistoryOutcomeFreeView(
        dataset="EmotionTalk",
        label_order=("a", "b", "c"),
        seeds=EXPECTED_SEEDS,
        fit_protocol_row_ids=np.arange(rows, dtype=np.int64),
        selection_protocol_row_ids=np.arange(100, 100 + rows, dtype=np.int64),
        fit_cluster_codes=fit_cluster,
        selection_cluster_codes=selection_cluster,
        fit_histories_sha256=SHA,
        selection_histories_sha256="b" * 64,
        fit_tasks=fit_tasks,
        selection_tasks=selection_tasks,
        fit_endpoint_probability_oof=_probability(
            rng, (len(EXPECTED_SEEDS), rows, 2, classes)
        ),
        selection_endpoint_probability_fold_ensemble=_probability(
            rng, (len(EXPECTED_SEEDS), rows, 2, classes)
        ),
        fit_utility_probability_oof=fit_utility,
        selection_utility_probability_fold_ensemble=selection_utility,
        source_identity_sha256="c" * 64,
        checkpoint_manifest_sha256="d" * 64,
        fit_outcome_artifact_sha256="e" * 64,
        fit_targets_artifact_sha256="f" * 64,
        artifact_sha256="1" * 64,
    )
    forward = rng.normal(0.0, 0.15, size=(len(EXPECTED_SEEDS), len(fit_tasks))).astype(
        np.float32
    )
    backward = (
        0.4 * forward
        + rng.normal(0.0, 0.12, size=forward.shape).astype(np.float32)
    )
    supervision = HistoryFitTargetsView(
        dataset="EmotionTalk",
        seeds=EXPECTED_SEEDS,
        task_sha256=SHA,
        forward_utility=forward,
        backward_utility=backward,
        asymmetry=(forward - backward).astype(np.float32),
        sign_agreement=np.sign(forward) == np.sign(backward),
        source_identity_sha256="c" * 64,
        fit_outcome_artifact_sha256="e" * 64,
        artifact_sha256="f" * 64,
    )
    specs = (
        UtilityModelSpec(
            name="bidirectional_shared_mlp",
            mode="bidirectional_shared",
            hidden_layer_sizes=(4,),
            max_iter=12,
            solver="lbfgs",
            early_stopping=False,
        ),
        UtilityModelSpec(
            name="forward_only_mlp",
            mode="forward_only",
            hidden_layer_sizes=(4,),
            max_iter=12,
            solver="lbfgs",
            early_stopping=False,
        ),
        UtilityModelSpec(
            name="backward_only_mlp",
            mode="backward_only",
            hidden_layer_sizes=(4,),
            max_iter=12,
            solver="lbfgs",
            early_stopping=False,
        ),
    )
    plan = build_outcome_free_strategy_nonproduction_fixture(
        outcome=outcome,
        supervision=supervision,
        fit_histories=histories,
        selection_histories=histories,
        fixture_identity_sha256="2" * 64,
        model_specs=specs,
    )
    return outcome, supervision, histories, plan


def _upstream(outcome, supervision, histories, *, folds: int = 2):
    fit_features = SimpleNamespace(
        feature_identity_sha256="3" * 64,
        feature_file_sha256="4" * 64,
        rows=len(histories),
        histories=histories,
    )
    selection_features = SimpleNamespace(
        feature_identity_sha256="5" * 64,
        feature_file_sha256="6" * 64,
        rows=len(histories),
        histories=histories,
    )
    history_attestation = SimpleNamespace(
        artifact_sha256=outcome.artifact_sha256,
        completion_receipt_sha256="7" * 64,
        fit_producer_receipt_sha256="8" * 64,
        production_run_claim_sha256="9" * 64,
    )
    full_anchor = SimpleNamespace(
        artifact_sha256="0" * 64,
        completion_receipt_sha256="1" * 64,
        production_run_claim_sha256="2" * 64,
    )
    current = SimpleNamespace(
        artifact_sha256="3" * 64,
        completion_receipt_sha256="4" * 64,
        fit_producer_receipt_sha256="5" * 64,
        production_run_claim_sha256="6" * 64,
        current_only_source_identity_sha256="7" * 64,
        current_checkpoint_manifest_sha256="8" * 64,
        source_code_sha256="9" * 64,
        runtime_environment_sha256="0" * 64,
        fit_protocol_row_ids=outcome.fit_protocol_row_ids.copy(),
        selection_protocol_row_ids=outcome.selection_protocol_row_ids.copy(),
    )
    fold_by_seed = np.vstack(
        [np.arange(len(histories), dtype=np.int32) % folds for _ in EXPECTED_SEEDS]
    )
    history = SimpleNamespace(
        outcome=outcome,
        supervision=supervision,
        fit_outcome=SimpleNamespace(fold_by_seed_query=fold_by_seed),
        fit_features=fit_features,
        selection_features=selection_features,
        attestation=history_attestation,
        checkpoint_manifest=SimpleNamespace(outer_folds=folds),
        checkpoint_root=Path("checkpoints"),
        model_config_sha256="a" * 64,
        run_config_sha256="b" * 64,
        utility_config_sha256="c" * 64,
        code_sha256={"history.py": "d" * 64},
        execution_environment_sha256="e" * 64,
    )
    return SimpleNamespace(
        history=history,
        current=current,
        full_history_anchor=full_anchor,
        upstream_identity_sha256="f" * 64,
    )


@pytest.fixture(scope="module")
def synthetic_plan():
    return _fixture()


def test_fixed_roster_exact_fit_coverage_and_directional_ensemble(synthetic_plan) -> None:
    _outcome, _supervision, _histories, plan = synthetic_plan
    assert plan.method_roster == METHOD_ROSTER
    assert JOINT_EVALUATION_ROSTER[0] == "independent_current_only"
    assert plan.rule.fit_selected_count == round(0.25 * plan.rule.fit_pair_count)
    assert plan.forward_rule.fit_selected_count == round(
        0.25 * plan.forward_rule.fit_pair_count
    )
    assert plan.backward_rule.fit_selected_count == round(
        0.25 * plan.backward_rule.fit_pair_count
    )
    assert np.allclose(
        plan.fit_decision_ensemble,
        np.minimum(plan.fit_forward_ensemble, plan.fit_backward_ensemble),
    )
    assert all(
        len(selected) == len(recency)
        for selected, recency in zip(
            plan.policy.selected_contexts,
            plan.policy.matched_recency_contexts,
            strict=True,
        )
    )


def test_production_api_has_no_outcome_or_callback_capability() -> None:
    parameters = inspect.signature(complete_strategy_selection).parameters
    forbidden = ("label", "target", "outcome", "callback")
    assert not [
        name for name in parameters if any(token in name.lower() for token in forbidden)
    ]


def test_private_mapping_has_no_outcome_fields_and_binds_full_anchor(synthetic_plan) -> None:
    outcome, supervision, histories, plan = synthetic_plan
    upstream = _upstream(outcome, supervision, histories)
    probabilities = {
        method: np.full(
            (len(EXPECTED_SEEDS), len(histories), len(outcome.label_order)),
            1.0 / len(outcome.label_order),
            dtype=np.float32,
        )
        for method in METHOD_ROSTER
    }
    values = _strategy_artifact_mapping(
        upstream=upstream,
        plan=plan,
        probabilities=probabilities,
        registered_variant="no_vad",
        production_claim_sha256="1" * 64,
        strategy_config_sha256={"config": "2" * 64},
        strategy_code_sha256={"code": "3" * 64},
        strategy_runtime_environment_sha256="4" * 64,
        strategy_live_lineage_sha256="5" * 64,
    )
    forbidden = ("label", "target", "gold", "accuracy", "macro_f1", "nll", "brier", "metric")
    assert not [
        name for name in values if any(token in name.lower() for token in forbidden)
    ]
    assert str(values["registered_variant"]) == "no_vad"
    assert str(values["full_current_anchor_history_artifact_sha256"]) == "0" * 64
    validate_strategy_artifact_mapping(
        values,
        upstream=upstream,
        plan=plan,
        registered_variant="no_vad",
        production_claim_sha256="1" * 64,
    )


def test_inference_restores_complete_checkpoint_for_every_seed_fold(
    synthetic_plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, supervision, histories, plan = synthetic_plan
    upstream = _upstream(outcome, supervision, histories, folds=2)
    calls = []

    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._validate_history_model_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline.verify_complete_history_checkpoint_payloads",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._dummy_fit_view",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._selection_view",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._fit_corpus_from_view",
        lambda *_args, **_kwargs: object(),
    )
    selection_corpus = SimpleNamespace(texts=tuple("x" for _ in histories))
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._selection_corpus_from_view",
        lambda *_args, **_kwargs: selection_corpus,
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._split_from_outer_partition",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline.verify_checkpoint_manifest",
        lambda *_args, **_kwargs: None,
    )

    class Processor:
        def transform(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    def restore(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
            processor=Processor(),
            model=object(),
        )

    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline.train_one_fold_seed",
        restore,
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline.predict_one_probability_per_query",
        lambda _model, _corpus, _text, queries, _contexts, **_kwargs: np.full(
            (len(queries), len(outcome.label_order)),
            1.0 / len(outcome.label_order),
            dtype=np.float32,
        ),
    )
    run_config = SimpleNamespace(
        outer_folds=2,
        inner_validation_fraction=0.2,
        inference_batch_size=8,
        max_history_items=16,
    )
    result = _infer_registered_method_probabilities(
        upstream=upstream,
        plan=plan,
        model_config=object(),
        run_config=run_config,
        device=torch.device("cpu"),
    )
    assert len(calls) == len(EXPECTED_SEEDS) * 2
    assert all(value["require_complete_checkpoint"] is True for value in calls)
    assert tuple(result) == METHOD_ROSTER


def test_private_root_claim_is_write_once_and_external(tmp_path: Path) -> None:
    root = tmp_path / "strategy"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_claim_strategy_private_root, root, "1" * 64)
            for _ in range(2)
        ]
    successes = []
    failures = []
    for future in futures:
        try:
            successes.append(future.result())
        except FileExistsError as error:
            failures.append(error)
    assert len(successes) == len(failures) == 1
    assert strategy_private_paths(successes[0])["claim"].is_file()
    assert set(REGISTERED_VARIANTS) == {
        "full",
        "no_vad",
        "no_history_3x3",
        "capacity_control",
    }


def test_receipt_attributes_one_shared_current_anchor_and_variant_history(synthetic_plan) -> None:
    outcome, supervision, histories, plan = synthetic_plan
    upstream = _upstream(outcome, supervision, histories)
    receipt = _strategy_receipt(
        upstream=upstream,
        plan=plan,
        registered_variant="capacity_control",
        production_claim_sha256="1" * 64,
        artifact_sha256="2" * 64,
        strategy_config_sha256={"config": "3" * 64},
        strategy_code_sha256={"code": "4" * 64},
        strategy_runtime_environment_sha256="5" * 64,
        strategy_live_lineage_sha256="6" * 64,
    )
    source = receipt["completion_contract"]["reference_source_contract"]
    assert source["independent_current_only"] == (
        "single_full_history_anchor_current_artifact"
    )
    assert source["all_history_diagnostic"] == (
        "registered_variant_history_checkpoints"
    )
    assert receipt["completion_contract"]["joint_evaluation_roster"] == list(
        JOINT_EVALUATION_ROSTER
    )


def test_corrupt_upstream_gate_fails_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    inferred = []

    def corrupt(_state):
        raise StrategyStagedPipelineError("corrupt receipt")

    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._assert_upstream_unchanged",
        corrupt,
    )
    monkeypatch.setattr(
        "hva_affect.causal_backbone_strategy_staged_pipeline._infer_registered_method_probabilities",
        lambda **_kwargs: inferred.append(True),
    )
    model_config = CausalBackboneConfig(
        affect_relation_mode="primary_history_relation",
        affect_relation_use_vad_features=True,
        auxiliary_vad_weight=0.1,
        emotion_label_order=(
            "neutral",
            "happy",
            "sad",
            "angry",
            "surprised",
            "disgusted",
            "fearful",
        ),
    )
    with pytest.raises(StrategyStagedPipelineError, match="corrupt receipt"):
        complete_strategy_selection(
            upstream=object(),
            registered_variant="full",
            model_config=model_config,
            run_config=object(),
            private_output_root=Path("unused"),
            config_paths={},
            code_paths={},
            environment={},
            device=torch.device("cpu"),
        )
    assert not inferred


def test_production_history_current_strategy_attestation_chain_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real production receipts/claims with synthetic checkpoints.

    The test never invokes a trainer and removes the model-selection label
    sidecar before the current completion and all three strategy-facing gates.
    """

    sidecar_root = tmp_path / "sidecars"
    sidecar_root.mkdir()
    values = history_test_support._fixture(sidecar_root)  # noqa: SLF001
    values["model"] = replace(
        values["model"],
        affect_relation_mode="primary_history_relation",
        affect_relation_hidden_dim=8,
        affect_relation_use_vad_features=True,
        auxiliary_vad_weight=0.1,
        emotion_label_order=tuple(values["preflight"].fit.label_order),
    )

    # Reuse the synthetic checkpoint writer from the history contract tests,
    # but inject it through the canonical no-callback production entry point.
    captured_history_callback: dict[str, object] = {}

    def capture_history_callback(**kwargs):
        captured_history_callback["value"] = kwargs["fold_callback"]
        return SimpleNamespace()

    with monkeypatch.context() as patcher:
        patcher.setattr(
            history_test_support.pipeline,
            "produce_history_fit_only",
            capture_history_callback,
        )
        history_test_support._produce(sidecar_root, values)  # noqa: SLF001
    history_callback = captured_history_callback["value"]

    history_claim = history_pipeline.history_production_claim_sha256(
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
    history_root = history_pipeline.claim_or_resume_history_private_root(
        tmp_path / "history-private",
        production_claim_sha256=history_claim,
        allow_resume=False,
    )
    history_paths = history_pipeline._production_private_paths(history_root)  # noqa: SLF001
    with monkeypatch.context() as patcher:
        patcher.setattr(
            history_pipeline,
            "make_real_history_fit_fold_callback",
            lambda **_kwargs: history_callback,
        )
        history_fit = history_pipeline.produce_history_fit_only(
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            fit_preflight_receipt_path=values["preflight"].receipt_path,
            expected_fit_preflight_receipt_sha256=(
                values["preflight"].receipt_sha256
            ),
            checkpoint_root=history_paths["checkpoint"],
            outcome_artifact_path=history_paths["fit_outcome"],
            targets_artifact_path=history_paths["fit_targets"],
            producer_receipt_path=history_paths["fit_receipt"],
            model_config=values["model"],
            run_config=values["run"],
            utility_config=values["utility"],
            config_sha256=values["config_sha"],
            code_sha256=values["code_sha"],
            runtime_environment_sha256=values["runtime_sha"],
            execution_environment_sha256=values["execution_runtime_sha"],
            device=torch.device("cpu"),
            production_run_claim_sha256=history_claim,
        )
    assert history_fit.production_trainer is True
    history_state = history_test_support._verify(  # noqa: SLF001
        values, history_fit, history_paths["checkpoint"]
    )
    history_selection = history_test_support._selection(  # noqa: SLF001
        values,
        sidecar_root,
        history_state,
        history_paths["checkpoint"],
    )

    class HistoryProcessor:
        def transform(self, texts):
            return np.zeros((len(texts), values["model"].text_dim), dtype=np.float32)

    def restore_history_checkpoint(_corpus, _split, **kwargs):
        assert kwargs["require_complete_checkpoint"] is True
        return SimpleNamespace(
            model=object(),
            processor=HistoryProcessor(),
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            history_pipeline, "train_one_fold_seed", restore_history_checkpoint
        )
        patcher.setattr(
            history_pipeline,
            "predict_current_and_all_history",
            lambda _model, _corpus, _text, queries, **_kwargs: (
                history_test_support._context_probability(len(queries), 2, 7)  # noqa: SLF001
            ),
        )
        patcher.setattr(
            history_pipeline,
            "predict_utility_contexts",
            lambda _model, _corpus, _text, tasks, **_kwargs: (
                history_test_support._context_probability(len(tasks), 4, 7)  # noqa: SLF001
            ),
        )
        history_complete = history_test_support._complete(  # noqa: SLF001
            sidecar_root,
            values,
            history_state,
            history_paths["checkpoint"],
            history_selection,
            output_root=history_root,
        )

    history_attestation = (
        history_pipeline.verify_history_completion_production_attestation(
            history_complete.artifact_path,
            history_complete.receipt_path,
            history_complete.receipt_sha256,
        )
    )

    # Bind the independent current-only producer to the attested full-history
    # source, then create production-form current checkpoints without training.
    current_lineage = write_fit_only_lineage(
        values["preflight"].fit,
        fit_map=values["fit_map"],
        receipt_path=values["preflight"].receipt_path,
        expected_receipt_sha256=values["preflight"].receipt_sha256,
        output_path=sidecar_root / "current-fit-lineage.npz",
    )
    values["lineage"] = current_lineage
    values["receipt"] = values["preflight"].receipt_path
    history_fit_alignment = current_pipeline.load_attested_history_fit_alignment_view(
        history_attestation,
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=(
            values["preflight"].receipt_sha256
        ),
    )
    history_alignment = (
        current_pipeline.load_attested_history_producer_alignment_view(
            history_attestation,
            fit_producer=history_fit_alignment,
        )
    )

    current_parent = tmp_path / "current"
    current_parent.mkdir()
    captured_current_callback: dict[str, object] = {}

    def capture_current_callback(**kwargs):
        captured_current_callback["value"] = kwargs["fold_callback"]
        return SimpleNamespace()

    with monkeypatch.context() as patcher:
        patcher.setattr(
            current_test_support,
            "produce_independent_current_only_fit_oof",
            capture_current_callback,
        )
        _unused, current_context = current_test_support._produce_fake_fit(  # noqa: SLF001
            current_parent, values
        )
    current_callback = captured_current_callback["value"]
    with monkeypatch.context() as patcher:
        patcher.setattr(
            current_pipeline,
            "make_real_current_only_fold_callback",
            lambda **_kwargs: current_callback,
        )
        current_fit = current_pipeline.produce_current_only_fit_with_real_trainer(
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=current_lineage,
            fit_preflight_receipt_path=values["preflight"].receipt_path,
            expected_fit_preflight_receipt_sha256=(
                values["preflight"].receipt_sha256
            ),
            checkpoint_root=current_context.checkpoint_root,
            artifact_path=current_context.paths["fit_artifact"],
            producer_receipt_path=current_context.paths["fit_receipt"],
            model_config=current_context.model,
            run_config=current_context.run_config,
            model_config_sha256=current_context.hashes["model"],
            run_config_sha256=current_context.hashes["run"],
            source_code_sha256=current_context.hashes["code"],
            runtime_environment_sha256=current_context.hashes["runtime"],
            production_run_claim_sha256=current_context.claim,
            allow_checkpoint_resume=False,
            device=torch.device("cpu"),
        )

    current_state = current_pipeline.verify_current_only_fit_for_completion(
        fit_artifact_path=current_fit.artifact_path,
        fit_producer_receipt_path=current_fit.receipt_path,
        expected_fit_producer_receipt_sha256=current_fit.receipt_sha256,
        checkpoint_root=current_context.checkpoint_root,
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        lineage=current_lineage,
        producer=history_fit_alignment,
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=(
            values["preflight"].receipt_sha256
        ),
        outer_folds=current_context.run_config.outer_folds,
        model_config=current_context.model,
        run_config=current_context.run_config,
        model_config_sha256=current_context.hashes["model"],
        run_config_sha256=current_context.hashes["run"],
        source_code_sha256=current_context.hashes["code"],
        runtime_environment_sha256=current_context.hashes["runtime"],
        production_run_claim_sha256=current_context.claim,
        device=torch.device("cpu"),
    )
    selection = materialize_selection_features_after_receipt(
        receipt_path=values["preflight"].receipt_path,
        expected_receipt_sha256=values["preflight"].receipt_sha256,
        dataset="EmotionTalk",
        sidecar_dir=sidecar_root,
        manifest_path=values["manifest"],
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
    )

    selection_label_path = (
        sidecar_root / f"labels_{SELECTION_ROLE}.npz"
    ).resolve()
    selection_label_path.unlink()
    assert not selection_label_path.exists()

    class CurrentProcessor:
        def transform(self, texts):
            return np.zeros(
                (len(texts), current_context.model.text_dim), dtype=np.float32
            )

    def restore_current_checkpoint(_corpus, _split, **kwargs):
        assert kwargs["require_complete_checkpoint"] is True
        return SimpleNamespace(
            source_identity_sha256=current_state.current_only_source_identity_sha256,
            trained=SimpleNamespace(
                model=object(),
                processor=CurrentProcessor(),
                summary={
                    "resumed_complete_checkpoint": True,
                    "resumed_partial_checkpoint": False,
                },
            ),
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            current_pipeline,
            "train_independent_current_only_fold_seed",
            restore_current_checkpoint,
        )
        patcher.setattr(
            current_pipeline,
            "predict_one_probability_per_query",
            lambda _model, _corpus, _text, queries, _contexts, **_kwargs: np.full(
                (len(queries), 7), 1.0 / 7.0, dtype=np.float32
            ),
        )
        current_complete = (
            current_pipeline.complete_current_only_selection_probabilities(
                fit=values["preflight"].fit,
                selection=selection,
                fit_map=values["fit_map"],
                lineage=current_lineage,
                fit_producer=history_fit_alignment,
                producer=history_alignment,
                fit_state=current_state,
                fit_preflight_receipt_path=values["preflight"].receipt_path,
                expected_fit_preflight_receipt_sha256=(
                    values["preflight"].receipt_sha256
                ),
                checkpoint_root=current_context.checkpoint_root,
                model_config=current_context.model,
                run_config=current_context.run_config,
                model_config_sha256=current_context.hashes["model"],
                run_config_sha256=current_context.hashes["run"],
                source_code_sha256=current_context.hashes["code"],
                runtime_environment_sha256=current_context.hashes["runtime"],
                production_run_claim_sha256=current_context.claim,
                selection_feature_sha256=values["preflight"].receipt["sidecars"][
                    SELECTION_ROLE
                ]["feature_sha256"],
                artifact_path=current_context.paths["complete_artifact"],
                completion_receipt_path=current_context.paths["complete_receipt"],
                device=torch.device("cpu"),
            )
        )

    fit_features = strategy_pipeline.load_outcome_free_role_features(
        role=FIT_ROLE,
        dataset="EmotionTalk",
        feature_path=sidecar_root / f"features_{FIT_ROLE}.npz",
        manifest_path=values["manifest"],
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
    )
    selection_features = strategy_pipeline.load_outcome_free_role_features(
        role=SELECTION_ROLE,
        dataset="EmotionTalk",
        feature_path=sidecar_root / f"features_{SELECTION_ROLE}.npz",
        manifest_path=values["manifest"],
        fit_preflight_receipt_path=values["preflight"].receipt_path,
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
    )

    original_np_load = np.load

    def reject_selection_label(path, *args, **kwargs):
        if Path(path).resolve() == selection_label_path:
            raise AssertionError("a production verifier opened model-selection labels")
        return original_np_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", reject_selection_label)

    # The intended production sequence succeeds with the label capability gone.
    history_attestation = (
        history_pipeline.verify_history_completion_production_attestation(
            history_complete.artifact_path,
            history_complete.receipt_path,
            history_complete.receipt_sha256,
        )
    )
    current_attestation = (
        strategy_pipeline.verify_current_only_completion_production_attestation(
            current_complete.artifact_path,
            current_complete.receipt_path,
            current_complete.receipt_sha256,
            history_attestation=history_attestation,
            producer_alignment=history_alignment,
        )
    )
    upstream = strategy_pipeline.verify_strategy_upstream_state(
        history_attestation=history_attestation,
        current_attestation=current_attestation,
        full_history_anchor_attestation=history_attestation,
        full_history_anchor_model_config=values["model"],
        fit_features=fit_features,
        selection_features=selection_features,
    )
    assert upstream.current.anchor_history_artifact_sha256 == (
        upstream.full_history_anchor.artifact_sha256
    )

    no_vad_anchor = replace(
        values["model"],
        affect_relation_use_vad_features=False,
        auxiliary_vad_weight=0.0,
        emotion_label_order=(),
    )
    capacity_anchor = replace(
        values["model"],
        affect_relation_mode="history_presence_capacity_control",
    )
    for non_full_anchor in (no_vad_anchor, capacity_anchor):
        with pytest.raises(
            StrategyStagedPipelineError,
            match="variant label differs from the validated model contract",
        ):
            strategy_pipeline.verify_strategy_upstream_state(
                history_attestation=history_attestation,
                current_attestation=current_attestation,
                full_history_anchor_attestation=history_attestation,
                full_history_anchor_model_config=non_full_anchor,
                fit_features=fit_features,
                selection_features=selection_features,
            )

    def rewrite_json(path: Path, mutation) -> bytes:
        original = path.read_bytes()
        payload = json.loads(original.decode("utf-8"))
        mutation(payload)
        path.write_text(
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
        return original

    # History receipt internal hash, history claim, current receipt internal
    # hash, current claim, current artifact bytes, and in-memory anchor hash all
    # fail at their first applicable production gate.
    original = rewrite_json(
        history_complete.receipt_path,
        lambda payload: payload["lineage"].__setitem__(
            "checkpoint_manifest_sha256", "0" * 64
        ),
    )
    with pytest.raises(history_pipeline.HistoryStagedPipelineError):
        history_pipeline.verify_history_completion_production_attestation(
            history_complete.artifact_path,
            history_complete.receipt_path,
            strategy_pipeline._file_sha256(history_complete.receipt_path),  # noqa: SLF001
        )
    history_complete.receipt_path.write_bytes(original)

    history_claim_path = history_root / "history-run-claim.json"
    original = rewrite_json(
        history_claim_path,
        lambda payload: payload.__setitem__("production_claim_sha256", "0" * 64),
    )
    with pytest.raises(history_pipeline.HistoryStagedPipelineError):
        history_pipeline.verify_history_completion_production_attestation(
            history_complete.artifact_path,
            history_complete.receipt_path,
            history_complete.receipt_sha256,
        )
    history_claim_path.write_bytes(original)

    original = rewrite_json(
        current_complete.receipt_path,
        lambda payload: payload["lineage"].__setitem__(
            "history_checkpoint_manifest_sha256", "0" * 64
        ),
    )
    with pytest.raises(StrategyStagedPipelineError):
        strategy_pipeline.verify_current_only_completion_production_attestation(
            current_complete.artifact_path,
            current_complete.receipt_path,
            strategy_pipeline._file_sha256(current_complete.receipt_path),  # noqa: SLF001
            history_attestation=history_attestation,
            producer_alignment=history_alignment,
        )
    current_complete.receipt_path.write_bytes(original)

    current_claim_path = current_context.paths["claim"]
    original = rewrite_json(
        current_claim_path,
        lambda payload: payload.__setitem__("production_claim_sha256", "0" * 64),
    )
    with pytest.raises(StrategyStagedPipelineError):
        strategy_pipeline.verify_current_only_completion_production_attestation(
            current_complete.artifact_path,
            current_complete.receipt_path,
            current_complete.receipt_sha256,
            history_attestation=history_attestation,
            producer_alignment=history_alignment,
        )
    current_claim_path.write_bytes(original)

    original_artifact = current_complete.artifact_path.read_bytes()
    current_complete.artifact_path.write_bytes(original_artifact + b"tampered")
    with pytest.raises(StrategyStagedPipelineError):
        strategy_pipeline.verify_current_only_completion_production_attestation(
            current_complete.artifact_path,
            current_complete.receipt_path,
            current_complete.receipt_sha256,
            history_attestation=history_attestation,
            producer_alignment=history_alignment,
        )
    current_complete.artifact_path.write_bytes(original_artifact)

    current_checkpoint_path = sorted(
        current_context.checkpoint_root.rglob("checkpoint.pt")
    )[0]
    original_checkpoint = current_checkpoint_path.read_bytes()
    current_checkpoint_path.write_bytes(original_checkpoint + b"tampered")
    with pytest.raises(StrategyStagedPipelineError):
        strategy_pipeline.verify_current_only_completion_production_attestation(
            current_complete.artifact_path,
            current_complete.receipt_path,
            current_complete.receipt_sha256,
            history_attestation=history_attestation,
            producer_alignment=history_alignment,
        )
    current_checkpoint_path.write_bytes(original_checkpoint)

    with pytest.raises(StrategyStagedPipelineError):
        strategy_pipeline.verify_strategy_upstream_state(
            history_attestation=history_attestation,
            current_attestation=replace(
                current_attestation,
                anchor_history_artifact_sha256="0" * 64,
            ),
            full_history_anchor_attestation=history_attestation,
            full_history_anchor_model_config=values["model"],
            fit_features=fit_features,
            selection_features=selection_features,
        )
    assert not selection_label_path.exists()
