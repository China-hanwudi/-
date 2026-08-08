from __future__ import annotations

import hashlib
import json
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="causal evidence tests require PyTorch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hva_affect.causal_backbone_evidence as evidence  # noqa: E402
from hva_affect.causal_backbone_evidence import (  # noqa: E402
    CURRENT_ONLY_CACHE_SCHEMA,
    EXPECTED_SEEDS,
    INDEPENDENT_CURRENT_ONLY_PROTOCOL,
    PRODUCER_CACHE_SCHEMA,
    AccuracyNoHarmContrast,
    EvidenceBundle,
    EvidenceContractError,
    HolmHypothesis,
    MethodPrediction,
    _array_sha256,
    _canonical_sha256,
    _draw_crossed_seed_shared_clusters,
    _paired_whole_cluster_randomization_arrays,
    build_current_only_artifact_mapping,
    current_only_artifact_from_mapping,
    current_only_independence_attestation_payload,
    evaluate_open_role_evidence,
    aggregate_required_dataset_reports,
    freeze_fit_oof_operating_point,
    holm_bonferroni,
    load_accuracy_no_harm_gate_from_confirmatory_config,
    paired_seed_shared_cluster_contrast,
    predeclare_holm_family,
    predeclare_accuracy_no_harm_gate,
    predeclare_cross_dataset_aggregation,
    prepare_policy_contexts,
    producer_cache_from_mapping,
    validate_aggregate_public_output,
    validate_evidence_bundle,
    write_aggregate_public_report,
)
from hva_affect.causal_multimodal_backbone import CausalBackboneConfig  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    FIT_ROLE,
    BackboneRunConfig,
    CrossfitSplit,
    OpenRoleCorpus,
    TrainedFold,
)


LABELS = tuple(f"emotion_{index}" for index in range(7))
HISTORY_IDENTITY = "1" * 64
CURRENT_IDENTITY = "2" * 64
HISTORY_CHECKPOINT = "3" * 64
CURRENT_CHECKPOINT = "4" * 64
SCORE_IDENTITY = "5" * 64


def _uniform_probability(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, 1.0 / shape[-1], dtype=np.float32)


def _task_encoding(queries: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    # Two coalition draws per ordered query-candidate pair.
    task_queries = np.repeat(queries, 2).astype(np.int64)
    candidate = np.repeat(queries - 1, 2).astype(np.int64)
    s_indptr = np.zeros(len(task_queries) + 1, dtype=np.int64)
    s_indices = np.empty(0, dtype=np.int64)
    t_indptr = np.arange(len(task_queries) + 1, dtype=np.int64)
    t_indices = candidate.copy()
    payload = [
        {"query": int(query), "candidate": int(item), "s": [], "t": [int(item)]}
        for query, item in zip(task_queries, candidate, strict=True)
    ]
    return (
        {
            "query_indices": task_queries,
            "candidate_indices": candidate,
            "s_indptr": s_indptr,
            "s_indices": s_indices,
            "t_indptr": t_indptr,
            "t_indices": t_indices,
        },
        _canonical_sha256(payload),
    )


def valid_producer_mapping(dataset: str = "EmotionTalk") -> dict[str, np.ndarray]:
    fit_query = np.asarray([1, 3, 5, 7], dtype=np.int64)
    selection_query = np.asarray([9, 11, 13, 15], dtype=np.int64)
    fit_tasks, fit_task_hash = _task_encoding(fit_query)
    selection_tasks, selection_task_hash = _task_encoding(selection_query)
    seeds = np.asarray(EXPECTED_SEEDS, dtype=np.int64)
    fit_endpoint = _uniform_probability((5, 4, 2, 7))
    selection_endpoint = _uniform_probability((5, 4, 2, 7))
    fit_utility_probability = _uniform_probability((5, 8, 4, 7))
    selection_utility_probability = _uniform_probability((5, 8, 4, 7))
    fit_forward = np.zeros((5, 8), dtype=np.float32)
    fit_backward = np.zeros((5, 8), dtype=np.float32)
    selection_forward = np.zeros((5, 8), dtype=np.float32)
    selection_backward = np.zeros((5, 8), dtype=np.float32)
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PRODUCER_CACHE_SCHEMA),
        "dataset": np.asarray(dataset),
        "dataset_label_order": np.asarray(LABELS),
        "manifest_schema": np.asarray("strict_multimodal_role_sidecars_v2"),
        "manifest_status": np.asarray("verified"),
        "manifest_sha256": np.asarray("a" * 64),
        "verified_provenance_attestation_sha256": np.asarray("b" * 64),
        "corpus_contract_sha256": np.asarray("c" * 64),
        "histories_sha256": np.asarray("d" * 64),
        "speaker_mapping_sha256": np.asarray("e" * 64),
        "runtime_environment_sha256": np.asarray("f" * 64),
        "source_identity_sha256": np.asarray(HISTORY_IDENTITY),
        "seeds": seeds,
        "endpoint_context_names": np.asarray(("current_only", "all_history")),
        "utility_context_names": np.asarray(
            ("s", "s_plus_candidate", "t", "t_minus_candidate")
        ),
        "fit_query_indices": fit_query,
        "selection_query_indices": selection_query,
        "fit_cluster_codes": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "selection_cluster_codes": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "protocol_row_ids": np.arange(16, dtype=np.int64),
        "fit_endpoint_probability_oof": fit_endpoint,
        "selection_endpoint_probability_fold_ensemble": selection_endpoint,
        "fit_utility_probability_oof": fit_utility_probability,
        "selection_utility_probability_fold_ensemble": selection_utility_probability,
        "fit_forward_utility": fit_forward,
        "fit_backward_utility": fit_backward,
        "fit_asymmetry": np.zeros_like(fit_forward),
        "fit_sign_agreement": np.ones_like(fit_forward, dtype=bool),
        "selection_forward_utility": selection_forward,
        "selection_backward_utility": selection_backward,
        "selection_asymmetry": np.zeros_like(selection_forward),
        "selection_sign_agreement": np.ones_like(selection_forward, dtype=bool),
        "fit_task_sha256": np.asarray(fit_task_hash),
        "selection_task_sha256": np.asarray(selection_task_hash),
        "checkpoint_manifest_sha256": np.asarray(HISTORY_CHECKPOINT),
        "utility_source": np.asarray(
            "recomputed_from_causal_backbone_probabilities_and_open_role_labels"
        ),
        "source_sidecar_manifest_sha256": np.asarray("6" * 64),
    }
    for prefix, encoding in (("fit", fit_tasks), ("selection", selection_tasks)):
        for name, array in encoding.items():
            values[f"{prefix}_task_{name}"] = array
    matrices = {
        "fit_endpoint_probability_oof": fit_endpoint,
        "selection_endpoint_probability_fold_ensemble": selection_endpoint,
        "fit_utility_probability_oof": fit_utility_probability,
        "selection_utility_probability_fold_ensemble": selection_utility_probability,
        "fit_forward_utility": fit_forward,
        "fit_backward_utility": fit_backward,
        "selection_forward_utility": selection_forward,
        "selection_backward_utility": selection_backward,
    }
    for name, array in matrices.items():
        values[f"matrix_{name}_sha256"] = np.asarray(_array_sha256(array))
    return values


def valid_current_mapping(
    producer: evidence.CausalProducerCache,
    *,
    fit_probability: np.ndarray | None = None,
    selection_probability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    fit_probability = (
        _uniform_probability((5, 4, 7)) if fit_probability is None else fit_probability
    )
    selection_probability = (
        _uniform_probability((5, 4, 7))
        if selection_probability is None
        else selection_probability
    )
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CURRENT_ONLY_CACHE_SCHEMA),
        "dataset": np.asarray(producer.dataset),
        "dataset_label_order": np.asarray(producer.label_order),
        "seeds": np.asarray(producer.seeds, dtype=np.int64),
        "fit_query_indices": producer.fit_query_indices.copy(),
        "selection_query_indices": producer.selection_query_indices.copy(),
        "fit_cluster_codes": producer.fit_cluster_codes.copy(),
        "selection_cluster_codes": producer.selection_cluster_codes.copy(),
        "fit_probability_oof": np.asarray(fit_probability, dtype=np.float32),
        "selection_probability_fold_ensemble": np.asarray(
            selection_probability, dtype=np.float32
        ),
        "producer_source_identity_sha256": np.asarray(HISTORY_IDENTITY),
        "current_only_source_identity_sha256": np.asarray(CURRENT_IDENTITY),
        "history_backbone_checkpoint_manifest_sha256": np.asarray(HISTORY_CHECKPOINT),
        "checkpoint_manifest_sha256": np.asarray(CURRENT_CHECKPOINT),
        "training_protocol": np.asarray(INDEPENDENT_CURRENT_ONLY_PROTOCOL),
        "checkpoint_namespace": np.asarray("independent_current_only"),
        "history_training_items_consumed": np.asarray(0, dtype=np.int64),
        "history_inference_items_consumed": np.asarray(0, dtype=np.int64),
    }
    values["matrix_fit_probability_oof_sha256"] = np.asarray(
        _array_sha256(values["fit_probability_oof"])
    )
    values["matrix_selection_probability_fold_ensemble_sha256"] = np.asarray(
        _array_sha256(values["selection_probability_fold_ensemble"])
    )
    values["independence_attestation_sha256"] = np.asarray(
        _canonical_sha256(current_only_independence_attestation_payload(values))
    )
    return values


def histories() -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    for row in range(16):
        values.append((row - 1,) if row % 2 == 1 else tuple())
    return tuple(values)


def decisive_probability(labels: np.ndarray, *, correct: bool) -> np.ndarray:
    result = np.full((5, len(labels), 7), 0.001 / 6.0, dtype=np.float32)
    for seed_index in range(5):
        for row, label in enumerate(labels):
            target = int(label) if correct else 6
            result[seed_index, row, target] = 0.999
    return result


def target_probability(targets: Sequence[int]) -> np.ndarray:
    result = np.full((5, len(targets), 7), 0.001 / 6.0, dtype=np.float32)
    for seed_index in range(5):
        for row, target in enumerate(targets):
            result[seed_index, row, int(target)] = 0.999
    return result


@pytest.mark.parametrize("dataset", ["EmotionTalk", "MELD"])
def test_producer_cache_contract_supports_both_registered_datasets(dataset: str) -> None:
    cache = producer_cache_from_mapping(valid_producer_mapping(dataset))
    assert cache.dataset == dataset
    assert cache.seeds == EXPECTED_SEEDS
    assert cache.fit_endpoint_probability.shape == (5, 4, 2, 7)
    assert len(cache.selection_tasks) == 8


def test_producer_cache_rejects_malformed_source_hash_and_matrix_hash() -> None:
    malformed = valid_producer_mapping()
    malformed["source_sidecar_manifest_sha256"] = np.asarray("not-a-hash")
    with pytest.raises(EvidenceContractError, match="SHA-256"):
        producer_cache_from_mapping(malformed)

    malformed = valid_producer_mapping()
    malformed["matrix_fit_endpoint_probability_oof_sha256"] = np.asarray("0" * 64)
    with pytest.raises(EvidenceContractError, match="matrix hash differs"):
        producer_cache_from_mapping(malformed)


def test_independent_current_only_rejects_history_identity_or_consumption() -> None:
    producer = producer_cache_from_mapping(valid_producer_mapping())
    valid = valid_current_mapping(producer)
    artifact = current_only_artifact_from_mapping(valid, producer)
    assert artifact.source_identity_sha256 == CURRENT_IDENTITY

    reused = valid_current_mapping(producer)
    reused["current_only_source_identity_sha256"] = np.asarray(HISTORY_IDENTITY)
    reused["independence_attestation_sha256"] = np.asarray(
        _canonical_sha256(current_only_independence_attestation_payload(reused))
    )
    with pytest.raises(EvidenceContractError, match="reused history-trained"):
        current_only_artifact_from_mapping(reused, producer)

    consumed = valid_current_mapping(producer)
    consumed["history_training_items_consumed"] = np.asarray(1, dtype=np.int64)
    consumed["independence_attestation_sha256"] = np.asarray(
        _canonical_sha256(current_only_independence_attestation_payload(consumed))
    )
    with pytest.raises(EvidenceContractError, match="consumed history"):
        current_only_artifact_from_mapping(consumed, producer)


def test_current_only_builder_emits_exact_self_validating_independent_schema() -> None:
    producer = producer_cache_from_mapping(valid_producer_mapping())
    fit_probability = _uniform_probability((5, 4, 7)).astype(np.float64)
    selection_probability = _uniform_probability((5, 4, 7)).astype(np.float64)
    values = build_current_only_artifact_mapping(
        producer,
        fit_probability_oof=fit_probability,
        selection_probability_fold_ensemble=selection_probability,
        current_only_source_identity_sha256=CURRENT_IDENTITY,
        checkpoint_manifest_sha256=CURRENT_CHECKPOINT,
    )
    assert set(values) == evidence._CURRENT_ONLY_KEYS
    artifact = current_only_artifact_from_mapping(values, producer)
    assert artifact.source_identity_sha256 == CURRENT_IDENTITY
    assert artifact.checkpoint_manifest_sha256 == CURRENT_CHECKPOINT
    assert str(values["checkpoint_namespace"]) == "independent_current_only"
    assert int(values["history_training_items_consumed"]) == 0
    assert int(values["history_inference_items_consumed"]) == 0

    with pytest.raises(EvidenceContractError, match="history source identity"):
        build_current_only_artifact_mapping(
            producer,
            fit_probability_oof=fit_probability,
            selection_probability_fold_ensemble=selection_probability,
            current_only_source_identity_sha256=HISTORY_IDENTITY,
            checkpoint_manifest_sha256=CURRENT_CHECKPOINT,
        )
    with pytest.raises(EvidenceContractError, match="history checkpoint"):
        build_current_only_artifact_mapping(
            producer,
            fit_probability_oof=fit_probability,
            selection_probability_fold_ensemble=selection_probability,
            current_only_source_identity_sha256=CURRENT_IDENTITY,
            checkpoint_manifest_sha256=HISTORY_CHECKPOINT,
        )


def test_accuracy_gate_loads_from_exact_hash_bound_confirmatory_config(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs" / "carma_confirmatory_analysis_v1.json"
    expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    bindings = {
        "A1_accuracy_vs_current": ("carma", "independent_current_only"),
        "A2_accuracy_vs_frozen_reference": ("carma", "coverage_matched_recency"),
    }
    gate = load_accuracy_no_harm_gate_from_confirmatory_config(
        config_path,
        method_bindings=bindings,
        expected_sha256=expected_hash,
    )
    assert gate.gate_id == "carma_confirmatory_accuracy_no_harm_v1"
    assert gate.minimum_point_difference == 0.0
    assert gate.minimum_ci95_lower == -0.005
    assert {row.contrast_id for row in gate.contrasts} == set(bindings)

    with pytest.raises(EvidenceContractError, match="config hash differs"):
        load_accuracy_no_harm_gate_from_confirmatory_config(
            config_path,
            method_bindings=bindings,
            expected_sha256="0" * 64,
        )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["effect_and_safety_gates"]["accuracy_no_harm"]["contrasts"].pop()
    reduced = tmp_path / "reduced.json"
    reduced.write_text(json.dumps(payload), encoding="utf-8")
    reduced_hash = hashlib.sha256(reduced.read_bytes()).hexdigest()
    with pytest.raises(EvidenceContractError, match="exact frozen accuracy family"):
        load_accuracy_no_harm_gate_from_confirmatory_config(
            reduced,
            method_bindings={"A1_accuracy_vs_current": bindings["A1_accuracy_vs_current"]},
            expected_sha256=reduced_hash,
        )


def tiny_model_config() -> CausalBackboneConfig:
    return CausalBackboneConfig(
        text_dim=8,
        audio_dim=10,
        video_dim=12,
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=24,
        num_speakers=4,
        max_turns=32,
        max_relative_turn=8,
        num_classes=7,
        dropout=0.0,
    )


def tiny_corpus() -> OpenRoleCorpus:
    rows = 6
    return OpenRoleCorpus(
        keys=np.asarray([f"row_{index}" for index in range(rows)]),
        texts=tuple(f"emotion row {index}" for index in range(rows)),
        audio=np.zeros((rows, 10), dtype=np.float32),
        video=np.zeros((rows, 12), dtype=np.float32),
        labels=np.arange(rows, dtype=np.int64) % 7,
        groups=np.asarray(["dialogue"] * rows),
        roles=np.asarray([FIT_ROLE] * rows),
        buckets=np.asarray([10] * rows, dtype=np.int16),
        speaker_ids=np.asarray([1] * rows, dtype=np.int64),
        turn_ids=np.arange(rows, dtype=np.int64),
        histories=tuple(tuple(range(index)) for index in range(rows)),
        speaker_identity=np.asarray(["speaker"] * rows),
        speaker_mapping_sha256="7" * 64,
    )


def test_current_only_training_and_inference_physically_strip_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = tiny_corpus()
    split = CrossfitSplit(
        fold=0,
        inner_train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        inner_validation_indices=np.asarray([3], dtype=np.int64),
        outer_heldout_indices=np.asarray([4, 5], dtype=np.int64),
    )
    seen: dict[str, object] = {}
    sentinel = TrainedFold(
        model=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
        text_features=np.zeros((6, 8), dtype=np.float32),
        checkpoint_path=tmp_path / "checkpoint.pt",
        processor_path=tmp_path / "processor.joblib",
        summary={},
    )

    def fake_train(stripped: OpenRoleCorpus, _split: CrossfitSplit, **kwargs: object):
        seen["train_histories"] = stripped.histories
        seen["subset_dropout"] = kwargs["run_config"].subset_dropout_probability  # type: ignore[union-attr]
        seen["source_identity"] = kwargs["source_identity"]
        return sentinel

    def fake_predict(
        _model: object,
        stripped: OpenRoleCorpus,
        _text: np.ndarray,
        queries: tuple[int, ...],
        contexts: tuple[tuple[int, ...], ...],
        **_kwargs: object,
    ) -> np.ndarray:
        seen["inference_histories"] = stripped.histories
        seen["contexts"] = contexts
        return _uniform_probability((len(queries), 7))

    monkeypatch.setattr(evidence, "train_one_fold_seed", fake_train)
    monkeypatch.setattr(evidence, "predict_one_probability_per_query", fake_predict)
    fold = evidence.train_independent_current_only_fold_seed(
        corpus,
        split,
        producer_source_identity_sha256=HISTORY_IDENTITY,
        model_config=tiny_model_config(),
        run_config=BackboneRunConfig(use_amp=False),
        seed=17,
        checkpoint_root=tmp_path / "independent_current_only",
        device=torch.device("cpu"),
    )
    probability = evidence.predict_independent_current_only_probability(
        fold,
        corpus,
        (4, 5),
        device=torch.device("cpu"),
        batch_size=2,
        max_history_items=8,
    )
    assert all(not row for row in seen["train_histories"])  # type: ignore[union-attr]
    assert all(not row for row in seen["inference_histories"])  # type: ignore[union-attr]
    assert seen["contexts"] == ((), ())
    assert seen["subset_dropout"] == 0.0
    assert seen["source_identity"] != HISTORY_IDENTITY
    assert probability.shape == (2, 7)


def test_fit_oof_freeze_is_exact_at_25_percent_under_boundary_ties() -> None:
    producer = producer_cache_from_mapping(valid_producer_mapping())
    fit_scores = np.ones(len(producer.fit_tasks), dtype=np.float64)
    rule = freeze_fit_oof_operating_point(
        producer.fit_tasks,
        fit_scores,
        score_source_identity_sha256=SCORE_IDENTITY,
    )
    assert rule.fit_pair_count == 4
    assert rule.fit_selected_count == 1
    assert rule.fit_realized_coverage == 0.25
    assert rule.boundary_tie_fraction == 0.25

    policy = prepare_policy_contexts(
        query_indices=producer.selection_query_indices,
        histories=histories(),
        tasks=producer.selection_tasks,
        decision_scores=np.ones(len(producer.selection_tasks), dtype=np.float64),
        score_source_identity_sha256=SCORE_IDENTITY,
        rule=rule,
    )
    assert policy.selected_pair_count == 1
    assert policy.realized_pair_coverage == 0.25
    assert all(
        len(selected) == len(recency)
        for selected, recency in zip(
            policy.selected_contexts, policy.matched_recency_contexts, strict=True
        )
    )


def test_policy_freeze_api_cannot_receive_labels_clusters_or_utilities() -> None:
    freeze_parameters = set(inspect.signature(freeze_fit_oof_operating_point).parameters)
    prepare_parameters = set(inspect.signature(prepare_policy_contexts).parameters)
    for forbidden in ("labels", "cluster_codes", "utilities", "targets"):
        assert forbidden not in freeze_parameters
        assert forbidden not in prepare_parameters


def test_crossed_draw_has_one_shared_cluster_vector_without_seed_axis() -> None:
    rng = np.random.default_rng(20260808)
    seed_draw, cluster_draw = _draw_crossed_seed_shared_clusters(
        rng, seed_count=5, cluster_count=7
    )
    assert seed_draw.shape == (5,)
    assert cluster_draw.shape == (7,)
    assert cluster_draw.ndim == 1


def make_effect_bundle(dataset: str = "EmotionTalk") -> EvidenceBundle:
    producer = producer_cache_from_mapping(valid_producer_mapping(dataset))
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    current_probability = decisive_probability(labels, correct=False)
    current = current_only_artifact_from_mapping(
        valid_current_mapping(producer, selection_probability=current_probability),
        producer,
    )
    rule = freeze_fit_oof_operating_point(
        producer.fit_tasks,
        np.linspace(1.0, 0.0, len(producer.fit_tasks)),
        score_source_identity_sha256=SCORE_IDENTITY,
    )
    policy = prepare_policy_contexts(
        query_indices=producer.selection_query_indices,
        histories=histories(),
        tasks=producer.selection_tasks,
        decision_scores=np.linspace(1.0, 0.0, len(producer.selection_tasks)),
        score_source_identity_sha256=SCORE_IDENTITY,
        rule=rule,
    )
    perfect = decisive_probability(labels, correct=True)
    methods = {
        "independent_current_only": MethodPrediction(
            current_probability, CURRENT_IDENTITY, "independent_current_only"
        ),
        "coverage_matched_recency": MethodPrediction(
            current_probability, HISTORY_IDENTITY, "coverage_matched_recency"
        ),
        "carma_bidirectional_full": MethodPrediction(
            perfect, HISTORY_IDENTITY, "selected_history"
        ),
    }
    return EvidenceBundle(
        producer=producer,
        independent_current_only=current,
        role="model_selection",
        labels=labels,
        histories=histories(),
        policy=policy,
        methods=methods,
    )


def test_one_prediction_per_query_and_independent_baseline_are_enforced() -> None:
    bundle = make_effect_bundle()
    validate_evidence_bundle(bundle)
    malformed_methods = dict(bundle.methods)
    malformed_methods["carma_bidirectional_full"] = MethodPrediction(
        np.zeros((5, 5, 7), dtype=np.float32),
        HISTORY_IDENTITY,
        "selected_history",
    )
    with pytest.raises(EvidenceContractError, match="shape"):
        validate_evidence_bundle(
            EvidenceBundle(
                producer=bundle.producer,
                independent_current_only=bundle.independent_current_only,
                role=bundle.role,
                labels=bundle.labels,
                histories=bundle.histories,
                policy=bundle.policy,
                methods=malformed_methods,
            )
        )

    forged_methods = dict(bundle.methods)
    forged_methods["independent_current_only"] = MethodPrediction(
        bundle.producer.selection_endpoint_probability[:, :, 0],
        HISTORY_IDENTITY,
        "independent_current_only",
    )
    with pytest.raises(EvidenceContractError, match="reused the history model identity"):
        validate_evidence_bundle(
            EvidenceBundle(
                producer=bundle.producer,
                independent_current_only=bundle.independent_current_only,
                role=bundle.role,
                labels=bundle.labels,
                histories=bundle.histories,
                policy=bundle.policy,
                methods=forged_methods,
            )
        )


def test_shared_cluster_bootstrap_is_deterministic_for_identity_and_known_effect() -> None:
    bundle = make_effect_bundle()
    identity = HolmHypothesis(
        "H0",
        "coverage_matched_recency",
        "coverage_matched_recency",
        "macro_f1",
        "greater",
    )
    equal = paired_seed_shared_cluster_contrast(
        bundle, identity, replicates=200, bootstrap_seed=37
    )
    assert equal["point_difference"] == 0.0
    assert equal["ci95_percentile"] == [0.0, 0.0]
    assert equal["hypothesis_test"][
        "paired_whole_cluster_randomization_p_value"
    ] == 1.0

    effect = HolmHypothesis(
        "H1",
        "carma_bidirectional_full",
        "coverage_matched_recency",
        "macro_f1",
        "greater",
    )
    first = paired_seed_shared_cluster_contrast(
        bundle, effect, replicates=200, bootstrap_seed=37
    )
    second = paired_seed_shared_cluster_contrast(
        bundle, effect, replicates=200, bootstrap_seed=37
    )
    assert first == second
    assert first["point_difference"] > 0.0
    assert first["ci95_percentile"][0] > 0.0  # type: ignore[index]
    assert first["bootstrap_design"].startswith("five_training_seeds_crossed")
    assert first["queries_within_cluster_kept_together"] is True
    assert first["hypothesis_test"]["one_swap_shared_across_five_seeds"] is True
    assert "bootstrap_directional_p_value" not in first


def test_exact_whole_cluster_randomization_detects_strong_classification_effect() -> None:
    labels = np.arange(6, dtype=np.int64)
    candidate = target_probability(labels.tolist())
    reference = target_probability(((labels + 1) % 7).tolist())
    result = _paired_whole_cluster_randomization_arrays(
        labels=labels,
        candidate=candidate,
        reference=reference,
        current=reference,
        clusters=np.arange(6, dtype=np.int64),
        eligible=np.ones(6, dtype=bool),
        metric="macro_f1",
        alternative="greater",
        replicates=1000,
        seed=17,
    )
    assert result["point_difference"] > 0.0
    assert result["exact_enumeration"] is True
    assert result["assignment_count"] == 64
    assert result["paired_whole_cluster_randomization_p_value"] <= 1.0 / 64.0
    assert result["nonlinear_metric_recomputed_each_assignment"] is True


def test_exact_randomization_p_values_are_superuniform_under_sharp_null() -> None:
    # Condition on six unordered correct/wrong prediction pairs, then enumerate
    # every equally likely observed cluster assignment under the sharp null.
    labels = np.arange(6, dtype=np.int64)
    correct = target_probability(labels.tolist())
    wrong = target_probability(((labels + 1) % 7).tolist())
    p_values: list[float] = []
    for assignment in range(1 << 6):
        swap = ((assignment >> np.arange(6)) & 1).astype(bool)
        candidate = np.where(swap[None, :, None], wrong, correct)
        reference = np.where(swap[None, :, None], correct, wrong)
        result = _paired_whole_cluster_randomization_arrays(
            labels=labels,
            candidate=candidate,
            reference=reference,
            current=reference,
            clusters=np.arange(6, dtype=np.int64),
            eligible=np.ones(6, dtype=bool),
            metric="accuracy",
            alternative="greater",
            replicates=1000,
            seed=29,
        )
        p_values.append(float(result["paired_whole_cluster_randomization_p_value"]))
    observed_size = float(np.mean(np.asarray(p_values) <= 0.05))
    assert observed_size <= 0.05 + 1.0 / 64.0
    assert float(np.mean(p_values)) >= 0.5


def test_holm_is_monotone_and_complete_family_only() -> None:
    adjusted = holm_bonferroni(
        {"H1": 0.01, "H2": 0.03, "H3": 0.04},
        declared_order=("H1", "H2", "H3"),
        alpha=0.05,
    )
    ranked = sorted(adjusted.values(), key=lambda value: int(value["holm_rank"]))
    assert [value["holm_adjusted_p_value"] for value in ranked] == sorted(
        value["holm_adjusted_p_value"] for value in ranked
    )
    with pytest.raises(EvidenceContractError, match="complete declared"):
        holm_bonferroni(
            {"H1": 0.01, "H2": 0.03},
            declared_order=("H1", "H2", "H3"),
            alpha=0.05,
        )


def test_full_evaluator_returns_aggregate_only_report_with_holm() -> None:
    bundle = make_effect_bundle()
    family = predeclare_holm_family(
        family_id="synthetic_frozen_family",
        alpha=0.05,
        hypotheses=(
            HolmHypothesis(
                "H1_macro",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "macro_f1",
                "greater",
            ),
            HolmHypothesis(
                "H2_regret",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "mean_regret",
                "less",
            ),
        ),
        analysis_config_sha256="8" * 64,
    )
    accuracy_gate = predeclare_accuracy_no_harm_gate(
        gate_id="synthetic_accuracy_no_harm",
        contrasts=(
            AccuracyNoHarmContrast(
                "A1_accuracy",
                "carma_bidirectional_full",
                "coverage_matched_recency",
            ),
        ),
        analysis_config_sha256="8" * 64,
    )
    report = evaluate_open_role_evidence(
        bundle,
        family,
        accuracy_gate,
        replicates=200,
        bootstrap_seed=20260808,
    )
    validate_aggregate_public_output(report)
    serialized = json.dumps(report, sort_keys=True)
    assert report["performance_claim_gate"]["authorized"] is False
    assert report["operating_point"]["target_candidate_pair_coverage"] == 0.25
    assert report["coverage_matched_recency_contract"][
        "same_selected_history_count_for_every_query"
    ] is True
    assert report["contrasts"]["H1_macro"]["ci95_percentile"][0] > 0.0
    assert report["contrasts"]["H2_regret"]["ci95_percentile"][1] < 0.0
    assert report["predeclared_holm_family"]["complete_family_evaluated"] is True
    assert report["mandatory_accuracy_no_harm_gate"][
        "all_predeclared_contrasts_passed"
    ] is True
    assert report["mandatory_accuracy_no_harm_gate"]["minimum_ci95_lower"] == -0.005
    assert report["mandatory_accuracy_no_harm_gate"]["contrasts"]["A1_accuracy"][
        "accuracy_improvement_supported"
    ] is True
    for forbidden in (
        "query_indices",
        "cluster_codes",
        "protocol_row_ids",
        '"labels"',
        '"predictions"',
        '"probabilities"',
        "C:\\\\",
    ):
        assert forbidden not in serialized


def test_macro_f1_point_gain_cannot_override_accuracy_no_harm_failure() -> None:
    base = make_effect_bundle()
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    reference = target_probability([0, 0, 0, 0])
    # Better class balance (and slightly better macro-F1) but only 50% accuracy
    # versus the reference's 75% accuracy.
    candidate = target_probability([0, 1, 1, 1])
    current = current_only_artifact_from_mapping(
        valid_current_mapping(base.producer, selection_probability=reference),
        base.producer,
    )
    bundle = EvidenceBundle(
        producer=base.producer,
        independent_current_only=current,
        role="model_selection",
        labels=labels,
        histories=base.histories,
        policy=base.policy,
        methods={
            "independent_current_only": MethodPrediction(
                reference, CURRENT_IDENTITY, "independent_current_only"
            ),
            "coverage_matched_recency": MethodPrediction(
                reference, HISTORY_IDENTITY, "coverage_matched_recency"
            ),
            "carma_bidirectional_full": MethodPrediction(
                candidate, HISTORY_IDENTITY, "selected_history"
            ),
        },
    )
    family = predeclare_holm_family(
        family_id="synthetic_accuracy_guard_family",
        alpha=0.05,
        hypotheses=(
            HolmHypothesis(
                "H1_macro",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "macro_f1",
                "greater",
            ),
            HolmHypothesis(
                "H2_regret",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "mean_regret",
                "less",
            ),
        ),
        analysis_config_sha256="9" * 64,
    )
    accuracy_gate = predeclare_accuracy_no_harm_gate(
        gate_id="synthetic_accuracy_guard",
        contrasts=(
            AccuracyNoHarmContrast(
                "A1_accuracy",
                "carma_bidirectional_full",
                "coverage_matched_recency",
            ),
        ),
        analysis_config_sha256="9" * 64,
    )
    report = evaluate_open_role_evidence(
        bundle,
        family,
        accuracy_gate,
        replicates=200,
        bootstrap_seed=71,
    )
    assert report["contrasts"]["H1_macro"]["point_difference"] > 0.0
    accuracy = report["mandatory_accuracy_no_harm_gate"]
    assert accuracy["all_predeclared_contrasts_passed"] is False
    assert accuracy["macro_f1_success_cannot_override_failure"] is True
    assert accuracy["contrasts"]["A1_accuracy"]["point_difference"] < 0.0
    assert accuracy["contrasts"]["A1_accuracy"][
        "noninferiority_is_not_improvement_evidence"
    ] is True


def test_single_dataset_cannot_trigger_hash_bound_dual_dataset_gate() -> None:
    family = predeclare_holm_family(
        family_id="dual_dataset_family",
        alpha=0.05,
        hypotheses=(
            HolmHypothesis(
                "H1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "macro_f1",
                "greater",
            ),
            HolmHypothesis(
                "H2",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "mean_regret",
                "less",
            ),
        ),
        analysis_config_sha256="c" * 64,
    )
    accuracy_gate = predeclare_accuracy_no_harm_gate(
        gate_id="dual_dataset_accuracy",
        contrasts=(
            AccuracyNoHarmContrast(
                "A1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
            ),
        ),
        analysis_config_sha256="c" * 64,
    )
    emotiontalk = evaluate_open_role_evidence(
        make_effect_bundle("EmotionTalk"),
        family,
        accuracy_gate,
        replicates=100,
        bootstrap_seed=43,
    )
    meld = evaluate_open_role_evidence(
        make_effect_bundle("MELD"),
        family,
        accuracy_gate,
        replicates=100,
        bootstrap_seed=43,
    )
    assert "single_dataset_not_publishable" in emotiontalk["claim_boundary"]
    assert emotiontalk["cross_dataset_claim_gate"][
        "single_dataset_can_trigger_method_success"
    ] is False
    plan = predeclare_cross_dataset_aggregation(
        plan_id="required_emotiontalk_meld_v1",
        analysis_config_sha256="c" * 64,
    )
    with pytest.raises(EvidenceContractError, match="both required dataset reports"):
        aggregate_required_dataset_reports({"EmotionTalk": emotiontalk}, plan)
    index = aggregate_required_dataset_reports(
        {"EmotionTalk": emotiontalk, "MELD": meld}, plan
    )
    assert index["both_required_datasets_present"] is True
    assert index["single_dataset_can_trigger_method_success"] is False
    assert index["method_success_authorized"] is False
    assert set(index["dataset_report_sha256"]) == {"EmotionTalk", "MELD"}


def test_public_output_validator_rejects_identifiers_arrays_and_paths() -> None:
    with pytest.raises(EvidenceContractError, match="forbidden field labels"):
        validate_aggregate_public_output({"labels": [1, 2]})
    with pytest.raises(EvidenceContractError, match="ndarray"):
        validate_aggregate_public_output({"aggregate": np.asarray([1.0])})
    with pytest.raises(EvidenceContractError, match="private/local path"):
        validate_aggregate_public_output({"artifact": "C:\\private\\cache.npz"})


def test_exact_public_whitelist_rejects_opaque_payloads_under_innocuous_names() -> None:
    bundle = make_effect_bundle()
    family = predeclare_holm_family(
        family_id="privacy_family",
        alpha=0.05,
        hypotheses=(
            HolmHypothesis(
                "H1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "macro_f1",
                "greater",
            ),
            HolmHypothesis(
                "H2",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "mean_regret",
                "less",
            ),
        ),
        analysis_config_sha256="a" * 64,
    )
    accuracy_gate = predeclare_accuracy_no_harm_gate(
        gate_id="privacy_accuracy",
        contrasts=(
            AccuracyNoHarmContrast(
                "A1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
            ),
        ),
        analysis_config_sha256="a" * 64,
    )
    report = evaluate_open_role_evidence(
        bundle, family, accuracy_gate, replicates=100, bootstrap_seed=17
    )
    opaque = json.loads(json.dumps(report))
    opaque["opaque_ids"] = ["rowhash_01", "rowhash_02"]
    with pytest.raises(EvidenceContractError, match="unknown=.*opaque_ids"):
        validate_aggregate_public_output(opaque)
    nested = json.loads(json.dumps(report))
    nested["counts"]["values"] = [0, 1, 2, 3]
    with pytest.raises(EvidenceContractError, match="unknown=.*values"):
        validate_aggregate_public_output(nested)
    vector = json.loads(json.dumps(report))
    vector["aggregate_method_metrics"]["carma_bidirectional_full"]["macro_f1"][
        "values"
    ] = [0.1, 0.2, 0.3, 0.4, 0.5]
    with pytest.raises(EvidenceContractError, match="unknown=.*values"):
        validate_aggregate_public_output(vector)


def test_public_report_writer_is_validated_atomic_and_write_once(tmp_path: Path) -> None:
    with pytest.raises(EvidenceContractError, match="schema mismatch"):
        write_aggregate_public_report(
            {"schema_version": "synthetic", "aggregate": {"mean": 1.0}},
            tmp_path / "uncontrolled.json",
        )
    bundle = make_effect_bundle()
    family = predeclare_holm_family(
        family_id="writer_family",
        alpha=0.05,
        hypotheses=(
            HolmHypothesis(
                "H1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "macro_f1",
                "greater",
            ),
            HolmHypothesis(
                "H2",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "mean_regret",
                "less",
            ),
        ),
        analysis_config_sha256="b" * 64,
    )
    accuracy_gate = predeclare_accuracy_no_harm_gate(
        gate_id="writer_accuracy",
        contrasts=(
            AccuracyNoHarmContrast(
                "A1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
            ),
        ),
        analysis_config_sha256="b" * 64,
    )
    payload = evaluate_open_role_evidence(
        bundle, family, accuracy_gate, replicates=100, bootstrap_seed=29
    )
    output = tmp_path / "evidence.json"
    write_aggregate_public_report(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        write_aggregate_public_report(payload, output)


def test_public_report_writer_never_clobbers_a_concurrent_winner(
    tmp_path: Path,
) -> None:
    bundle = make_effect_bundle()
    family = predeclare_holm_family(
        family_id="concurrent_writer_family",
        alpha=0.05,
        hypotheses=(
            HolmHypothesis(
                "H1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "macro_f1",
                "greater",
            ),
            HolmHypothesis(
                "H2",
                "carma_bidirectional_full",
                "coverage_matched_recency",
                "mean_regret",
                "less",
            ),
        ),
        analysis_config_sha256="c" * 64,
    )
    accuracy_gate = predeclare_accuracy_no_harm_gate(
        gate_id="concurrent_writer_accuracy",
        contrasts=(
            AccuracyNoHarmContrast(
                "A1",
                "carma_bidirectional_full",
                "coverage_matched_recency",
            ),
        ),
        analysis_config_sha256="c" * 64,
    )
    payload = evaluate_open_role_evidence(
        bundle,
        family,
        accuracy_gate,
        replicates=100,
        bootstrap_seed=31,
    )
    output = tmp_path / "concurrent-evidence.json"

    def attempt(_: int) -> str:
        try:
            write_aggregate_public_report(payload, output)
        except FileExistsError:
            return "exists"
        return "written"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert outcomes.count("written") == 1
    assert outcomes.count("exists") == 7
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".concurrent-evidence.json.*.tmp"))
