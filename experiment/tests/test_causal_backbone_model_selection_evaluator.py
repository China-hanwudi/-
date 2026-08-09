from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hva_affect.causal_backbone_model_selection_evaluator as evaluator
from hva_affect.causal_backbone_model_selection_evaluator import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATORY_ANALYSIS_SHA256,
    PRIMARY_VARIANT,
    PRIVATE_ARTIFACT_NAME,
    PRIVATE_RECEIPT_NAME,
    PROSPECTIVE_TARGET_MACRO_F1_GAIN,
    PUBLIC_REPORT_SCHEMA,
    RANDOMIZATION_REPLICATES,
    RANDOMIZATION_SEED,
    CompletedModelSelectionEvaluation,
    FrozenAnalysisContract,
    ModelSelectionEvaluationError,
    SelectionSidecarSource,
    StrategyProductionInput,
    VerifiedStrategyBundle,
    _EvaluationInputs,
    _VariantArrays,
    _VerifiedSelectionLabelCapability,
    _evaluate_model_selection_aggregates,
    _file_sha256,
    _load_frozen_analysis_contract,
    _load_model_selection_labels_once,
    _publish_outputs,
    _canonical_sha256,
    _derive_variant_from_live_configs,
    run_model_selection_reference_freeze,
    validate_model_selection_public_report,
    verify_model_selection_reference_freeze_receipt,
    verify_strategy_bundle_before_label_access,
)
from hva_affect.causal_backbone_strategy_staged_pipeline import (
    JOINT_EVALUATION_ROSTER,
    METHOD_ROSTER,
    REGISTERED_VARIANTS,
    VerifiedStrategyCompletionAttestation,
)


def _sha(character: str) -> str:
    return character * 64


def _upstream() -> SimpleNamespace:
    rows = np.arange(8, dtype=np.int64)
    clusters = np.repeat(np.arange(4, dtype=np.int64), 2)
    tasks = SimpleNamespace(task_sha256=_sha("a"), __len__=lambda self: 4)
    outcome = SimpleNamespace(
        dataset="MELD",
        label_order=("a", "b", "c"),
        seeds=(17, 29, 43, 71, 101),
        fit_protocol_row_ids=rows.copy(),
        selection_protocol_row_ids=rows.copy(),
        fit_cluster_codes=clusters.copy(),
        selection_cluster_codes=clusters.copy(),
        fit_histories_sha256=_sha("b"),
        selection_histories_sha256=_sha("c"),
        fit_tasks=tasks,
        selection_tasks=tasks,
    )
    features = SimpleNamespace(
        protocol_row_ids=rows.copy(),
        histories=tuple(() if index % 2 == 0 else (index - 1,) for index in rows),
        manifest_sha256=_sha("d"),
        feature_file_sha256=_sha("e"),
        row_alignment_sha256=_sha("f"),
    )
    return SimpleNamespace(
        history=SimpleNamespace(
            outcome=outcome,
            fit_outcome=SimpleNamespace(
                fold_by_seed_query=np.tile(np.arange(8) % 4, (5, 1))
            ),
            selection_features=features,
        ),
        current=SimpleNamespace(),
        full_history_anchor=SimpleNamespace(),
    )


def _attestation(variant: str, *, alignment: str = "1") -> VerifiedStrategyCompletionAttestation:
    history_sha = _sha("2") if variant == PRIMARY_VARIANT else _sha(
        {"no_vad": "3", "no_history_3x3": "4", "capacity_control": "5"}[variant]
    )
    return VerifiedStrategyCompletionAttestation(
        dataset="MELD",
        registered_variant=variant,
        artifact_path=Path(f"C:/private/{variant}/strategy-complete.npz"),
        artifact_sha256=_sha("6"),
        receipt_path=Path(f"C:/private/{variant}/strategy-complete-receipt.json"),
        receipt_sha256=_sha("7"),
        production_run_claim_sha256=_sha("8"),
        cross_variant_alignment_sha256=_sha(alignment),
        variant_history_artifact_sha256=history_sha,
        full_current_anchor_history_artifact_sha256=_sha("2"),
        current_artifact_sha256=_sha("9"),
        method_roster=METHOD_ROSTER,
        joint_evaluation_roster=JOINT_EVALUATION_ROSTER,
        base_seeds=(17, 29, 43, 71, 101),
        utility_seeds=(17, 29, 43, 71, 101),
        fit_query_count=8,
        selection_query_count=8,
        fit_task_count=4,
        selection_task_count=4,
    )


def _gate_inputs() -> dict[str, StrategyProductionInput]:
    return {
        variant: StrategyProductionInput(
            artifact_path=Path(f"C:/private/{variant}/strategy-complete.npz"),
            receipt_path=Path(
                f"C:/private/{variant}/strategy-complete-receipt.json"
            ),
            expected_receipt_sha256=_sha("7"),
            upstream=_upstream(),
            config_paths={variant: Path(f"{variant}.json")},
            code_paths={"code": Path("code.py")},
            environment={"python": "fixture"},
        )
        for variant in REGISTERED_VARIANTS
    }


def _patch_gate_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attestations: dict[str, VerifiedStrategyCompletionAttestation] | None = None,
    corrupt_config_variant: str | None = None,
) -> None:
    observed = attestations or {variant: _attestation(variant) for variant in REGISTERED_VARIANTS}

    def verify(_artifact, _receipt, _sha256, *, upstream):
        variant = next(
            key for key, value in _gate_inputs().items() if value.upstream.history.outcome.dataset == upstream.history.outcome.dataset
        )
        # Every fixture upstream is structurally equal, so use the receipt parent.
        name = Path(_artifact).parent.name
        return observed[name]

    config_hashes = {"model": _sha("a")}
    code_hashes = {"code": _sha("b")}
    runtime = _sha("c")
    live = _canonical_sha256(
        {
            "config_sha256": config_hashes,
            "code_sha256": code_hashes,
            "runtime_environment_sha256": runtime,
        }
    )
    common = {
        "fit_feature_identity_sha256": _sha("d"),
        "selection_feature_identity_sha256": _sha("e"),
        "fit_task_sha256": _sha("a"),
        "selection_task_sha256": _sha("a"),
        "history_fold_assignment_sha256": _sha("f"),
        "history_code_bundle_sha256": _sha("1"),
        "history_execution_environment_sha256": _sha("2"),
        "current_source_code_sha256": _sha("3"),
        "current_runtime_environment_sha256": _sha("4"),
        "strategy_config_bundle_sha256": _canonical_sha256(config_hashes),
        "strategy_code_bundle_sha256": _canonical_sha256(code_hashes),
        "strategy_runtime_environment_sha256": runtime,
        "strategy_live_lineage_sha256": live,
    }

    def read_receipt(attestation):
        lineage = dict(common)
        if corrupt_config_variant == attestation.registered_variant:
            lineage["strategy_config_bundle_sha256"] = _sha("0")
        return {"completion_contract": {}}, lineage

    monkeypatch.setattr(evaluator, "verify_strategy_completion_production_attestation", verify)
    monkeypatch.setattr(evaluator, "_read_attested_strategy_receipt", read_receipt)
    monkeypatch.setattr(
        evaluator,
        "_live_strategy_lineage",
        lambda **_kwargs: (config_hashes, code_hashes, runtime, live),
    )
    monkeypatch.setattr(
        evaluator,
        "_derive_variant_from_live_configs",
        lambda paths, **_kwargs: next(iter(paths)),
    )


def test_production_api_has_no_outcome_role_or_statistical_override() -> None:
    parameters = inspect.signature(run_model_selection_reference_freeze).parameters
    forbidden = {
        "labels",
        "label_path",
        "outcomes",
        "role",
        "calibration",
        "holdout",
        "validation",
        "test",
        "bootstrap_replicates",
        "bootstrap_seed",
        "randomization_replicates",
        "randomization_seed",
    }
    assert not (set(parameters) & forbidden)
    source_fields = set(SelectionSidecarSource.__dataclass_fields__)
    assert not (source_fields & {"role", "label_path", "outcome_path"})


def test_registered_variant_is_derived_from_each_actual_model_config() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    paths = {
        "full": config_dir / "carma_affect_relation_meld_full_v1.json",
        "no_vad": config_dir / "carma_affect_relation_meld_no_vad_v1.json",
        "no_history_3x3": (
            config_dir / "carma_affect_relation_meld_no_history_3x3_v1.json"
        ),
        "capacity_control": (
            config_dir / "carma_affect_relation_meld_capacity_control_v1.json"
        ),
    }
    assert {
        variant: _derive_variant_from_live_configs(
            {"backbone": path}, dataset="MELD"
        )
        for variant, path in paths.items()
    } == {variant: variant for variant in REGISTERED_VARIANTS}


def test_four_variant_gate_accepts_only_common_full_anchored_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gate_dependencies(monkeypatch)
    bundle = verify_strategy_bundle_before_label_access(_gate_inputs())
    assert tuple(bundle.attestations) == REGISTERED_VARIANTS
    assert bundle.attestations[PRIMARY_VARIANT].variant_history_artifact_sha256 == (
        bundle.full_current_anchor_history_artifact_sha256
    )
    assert bundle.current_artifact_sha256 == _sha("9")


@pytest.mark.parametrize("attack", ["variant", "alignment", "config"])
def test_variant_alignment_and_config_attacks_fail_in_prelabel_gate(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    attestations = {variant: _attestation(variant) for variant in REGISTERED_VARIANTS}
    corrupt_config = None
    if attack == "variant":
        attestations["no_vad"] = _attestation(PRIMARY_VARIANT)
    elif attack == "alignment":
        attestations["capacity_control"] = _attestation(
            "capacity_control", alignment="0"
        )
    else:
        corrupt_config = "no_vad"
    _patch_gate_dependencies(
        monkeypatch,
        attestations=attestations,
        corrupt_config_variant=corrupt_config,
    )
    with pytest.raises(ModelSelectionEvaluationError):
        verify_strategy_bundle_before_label_access(_gate_inputs())


def test_receipt_failure_stops_run_before_label_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        evaluator,
        "_load_frozen_analysis_contract",
        lambda _path: FrozenAnalysisContract(_sha("a"), _sha("b"), "family", 0.05),
    )

    def fail_gate(_strategies):
        events.append("receipt")
        raise ModelSelectionEvaluationError("tampered strategy receipt")

    monkeypatch.setattr(evaluator, "verify_strategy_bundle_before_label_access", fail_gate)
    monkeypatch.setattr(
        evaluator,
        "_verify_selection_label_capability",
        lambda *_args: events.append("label"),
    )
    with pytest.raises(ModelSelectionEvaluationError, match="tampered strategy receipt"):
        run_model_selection_reference_freeze(
            strategies={},
            selection_source=SelectionSidecarSource(
                "MELD", "unused", "unused", "unused", _sha("a"), {}, {}, {}
            ),
            confirmatory_analysis_path="unused",
            private_output_root="unused",
            public_report_path="unused.json",
        )
    assert events == ["receipt"]


def test_confirmatory_config_byte_tamper_fails_before_any_producer(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "configs" / "carma_confirmatory_analysis_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["hierarchical_bootstrap"]["replicates"] = 9999
    tampered = tmp_path / source.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelSelectionEvaluationError, match="frozen byte contract"):
        _load_frozen_analysis_contract(tampered)


def test_evaluator_consumes_the_exact_frozen_analysis_contract() -> None:
    path = Path(__file__).parents[1] / "configs" / "carma_confirmatory_analysis_v1.json"
    contract = _load_frozen_analysis_contract(path)
    assert contract.analysis_sha256 == CONFIRMATORY_ANALYSIS_SHA256
    assert contract.harm_reference_candidates == (
        "all_history",
        "coverage_matched_recency",
        "forward_only_utility",
        "backward_only_utility",
    )
    assert contract.zero_harm_denominator_action == "fail_closed_not_estimable"
    assert contract.per_seed_required_successes == 4
    assert contract.per_seed_success_conditions == (
        "macro_f1_candidate_strictly_greater_than_reference",
        "mean_regret_vs_current_non_positive",
    )


def _minimal_strategy_bundle(upstream: SimpleNamespace) -> VerifiedStrategyBundle:
    source = StrategyProductionInput("a", "b", _sha("1"), upstream, {}, {}, {})
    return VerifiedStrategyBundle(
        dataset="MELD",
        attestations={},
        inputs={PRIMARY_VARIANT: source},
        receipt_lineage={},
        cross_variant_alignment_sha256=_sha("2"),
        full_current_anchor_history_artifact_sha256=_sha("3"),
        current_artifact_sha256=_sha("4"),
        strategy_config_roster_sha256=_sha("5"),
        common_strategy_code_bundle_sha256=_sha("6"),
        common_strategy_runtime_environment_sha256=_sha("7"),
    )


def test_label_sha_tamper_fails_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label_path = tmp_path / "labels_model_selection.npz"
    np.savez(
        label_path,
        schema_version=np.asarray(evaluator._SPECS["MELD"].label_schema),
        role=np.asarray("model_selection"),
        row_alignment_sha256=np.asarray(_sha("a")),
        labels=np.asarray([0, 1], dtype=np.int64),
    )
    opened: list[bool] = []
    monkeypatch.setattr(evaluator.np, "load", lambda *_a, **_k: opened.append(True))
    upstream = SimpleNamespace(
        history=SimpleNamespace(
            selection_features=SimpleNamespace(protocol_row_ids=np.arange(2)),
            outcome=SimpleNamespace(selection_protocol_row_ids=np.arange(2)),
        )
    )
    capability = _VerifiedSelectionLabelCapability(
        "MELD", label_path, _sha("0"), _sha("a"), 2, {}, _sha("b"), _sha("c")
    )
    with pytest.raises(ModelSelectionEvaluationError, match="before deserialization"):
        _load_model_selection_labels_once(
            capability, _minimal_strategy_bundle(upstream)
        )
    assert not opened


def test_label_archive_opens_once_without_pickle_and_reorders_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "labels_model_selection.npz"
    np.savez(
        path,
        schema_version=np.asarray(evaluator._SPECS["MELD"].label_schema),
        role=np.asarray("model_selection"),
        row_alignment_sha256=np.asarray(_sha("a")),
        labels=np.asarray([0, 1, 2], dtype=np.int64),
    )
    original = np.load
    calls: list[dict[str, object]] = []

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator.np, "load", spy)
    upstream = SimpleNamespace(
        history=SimpleNamespace(
            selection_features=SimpleNamespace(
                protocol_row_ids=np.asarray([12, 10, 11], dtype=np.int64)
            ),
            outcome=SimpleNamespace(
                selection_protocol_row_ids=np.asarray([10, 11, 12], dtype=np.int64)
            ),
        )
    )
    capability = _VerifiedSelectionLabelCapability(
        "MELD",
        path,
        _file_sha256(path),
        _sha("a"),
        3,
        {},
        _sha("b"),
        _sha("c"),
    )
    labels = _load_model_selection_labels_once(
        capability, _minimal_strategy_bundle(upstream)
    )
    assert labels.tolist() == [1, 2, 0]
    assert calls == [{"allow_pickle": False}]


def _probability(labels: np.ndarray, errors: tuple[int, ...]) -> np.ndarray:
    predicted = labels.copy()
    for index in errors:
        predicted[index] = (predicted[index] + 1) % 3
    matrix = np.full((len(labels), 3), 0.04, dtype=np.float64)
    matrix[np.arange(len(labels)), predicted] = 0.92
    return np.repeat(matrix[None, :, :], 5, axis=0)


@pytest.fixture(scope="module")
def aggregate_fixture():
    labels = np.asarray([0, 1, 2] * 4, dtype=np.int64)
    protocol = np.arange(len(labels), dtype=np.int64)
    clusters = np.repeat(np.arange(4, dtype=np.int64), 3)
    eligible = np.asarray([False, True, True] * 4, dtype=bool)
    current = _probability(labels, (0, 2, 4, 6, 8))
    full_methods = {
        "bidirectional_selected_history": _probability(labels, ()),
        "forward_only_selected_history": _probability(labels, (0, 2, 4)),
        "backward_only_selected_history": _probability(labels, (0, 2, 4, 6)),
        "coverage_matched_recency": _probability(labels, (1,)),
        "all_history_diagnostic": _probability(labels, (1, 2)),
    }

    def variant(name: str, errors: tuple[int, ...]) -> _VariantArrays:
        probability = _probability(labels, errors)
        methods = dict(full_methods)
        methods["bidirectional_selected_history"] = probability
        return _VariantArrays(name, "MELD", ("a", "b", "c"), protocol, clusters, methods)

    variants = {
        "full": _VariantArrays(
            "full", "MELD", ("a", "b", "c"), protocol, clusters, full_methods
        ),
        "no_vad": variant("no_vad", (0, 2)),
        "no_history_3x3": variant("no_history_3x3", (0, 2, 4)),
        "capacity_control": variant("capacity_control", (0, 2, 4, 6)),
    }
    inputs = _EvaluationInputs(
        "MELD",
        ("a", "b", "c"),
        protocol,
        clusters,
        eligible,
        current,
        variants,
    )
    analysis = FrozenAnalysisContract(
        CONFIRMATORY_ANALYSIS_SHA256, _sha("a"), "carma_confirmatory_claim_family_v1", 0.05
    )
    aggregates = _evaluate_model_selection_aggregates(
        labels=labels, inputs=inputs, analysis=analysis
    )
    return aggregates, analysis


def test_fixed_reference_holm_accuracy_capacity_and_sensitivity(aggregate_fixture) -> None:
    aggregates, _analysis = aggregate_fixture
    assert aggregates["reference_freeze"]["frozen_reference"] == (
        "coverage_matched_recency"
    )
    assert aggregates["holm_family"]["hypothesis_order"] == [
        "H1_primary_macro_f1",
        "H2_primary_mean_regret",
        "H3_emotion_constraint_increment",
        "H4_three_by_three_relation_increment",
        "H5_current_only_increment",
    ]
    assert aggregates["holm_family"]["capacity_control_included"] is False
    assert aggregates["capacity_control"]["included_in_holm_family"] is False
    assert aggregates["accuracy_no_harm"]["contrast_order"] == [
        "A1_accuracy_vs_current",
        "A2_accuracy_vs_frozen_reference",
    ]
    assert aggregates["history_harm_rate_reduction"]["reference"] == (
        "coverage_matched_recency"
    )
    assert "current_only" not in aggregates["history_harm_rate_reduction"][
        "reference_candidates"
    ]
    assert aggregates["per_seed_success"]["required_successes"] == 4
    assert aggregates["per_seed_success"]["same_seed_for_all_conditions"] is True
    assert aggregates["per_seed_success"]["success_count"] == 5
    assert all(
        row["macro_f1_condition_passed"]
        and row["mean_regret_condition_passed"]
        and row["same_seed_joint_success"]
        for row in aggregates["per_seed_success"]["seed_results"]
    )
    sensitivity = aggregates["prospective_sensitivity"]
    assert sensitivity["assumed_effect_absolute"] == PROSPECTIVE_TARGET_MACRO_F1_GAIN
    assert sensitivity["observed_effect_used_as_assumed_effect"] is False
    assert sensitivity["observed_post_hoc_power_computed"] is False
    assert sensitivity["bootstrap_error_replicates"] == BOOTSTRAP_REPLICATES
    assert sensitivity["bootstrap_seed"] == BOOTSTRAP_SEED
    assert sensitivity["configured_randomization_replicates"] == RANDOMIZATION_REPLICATES
    assert sensitivity["configured_randomization_seed"] == RANDOMIZATION_SEED
    assert "mean_regret_vs_frozen_reference" in aggregates["aggregate_gates"]
    assert aggregates["aggregate_gates"][
        "mean_regret_vs_frozen_reference"
    ]["included_in_holm_family"] is False
    assert aggregates["stage_authorization"]["calibration_unseal_authorized"] is False


def test_zero_history_harm_denominator_publishes_fail_closed_not_estimable() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    protocol = np.arange(6, dtype=np.int64)
    clusters = np.repeat(np.arange(2, dtype=np.int64), 3)
    probability = _probability(labels, ())
    methods = {method: probability for method in METHOD_ROSTER}
    variants = {
        variant: _VariantArrays(
            variant,
            "MELD",
            ("a", "b", "c"),
            protocol,
            clusters,
            methods,
        )
        for variant in REGISTERED_VARIANTS
    }
    inputs = _EvaluationInputs(
        "MELD",
        ("a", "b", "c"),
        protocol,
        clusters,
        np.asarray([False, True, True, False, True, True]),
        probability,
        variants,
    )
    analysis = FrozenAnalysisContract(
        CONFIRMATORY_ANALYSIS_SHA256,
        _sha("a"),
        "carma_confirmatory_claim_family_v1",
        0.05,
    )
    aggregates = _evaluate_model_selection_aggregates(
        labels=labels,
        inputs=inputs,
        analysis=analysis,
    )
    harm = aggregates["history_harm_rate_reduction"]
    assert harm["estimable"] is False
    assert harm["passed"] is False
    assert harm["relative_history_harm_rate_reduction"] is None
    assert harm["failure_reason"] == (
        "zero_reference_harm_rate_fail_closed_not_estimable"
    )


def _publication_bundle() -> VerifiedStrategyBundle:
    attestations = {variant: _attestation(variant) for variant in REGISTERED_VARIANTS}
    return VerifiedStrategyBundle(
        dataset="MELD",
        attestations=attestations,
        inputs={},
        receipt_lineage={},
        cross_variant_alignment_sha256=_sha("1"),
        full_current_anchor_history_artifact_sha256=_sha("2"),
        current_artifact_sha256=_sha("3"),
        strategy_config_roster_sha256=_sha("4"),
        common_strategy_code_bundle_sha256=_sha("5"),
        common_strategy_runtime_environment_sha256=_sha("6"),
    )


def _publish_fixture(
    aggregate_fixture,
    tmp_path: Path,
) -> CompletedModelSelectionEvaluation:
    aggregates, analysis = aggregate_fixture
    capability = _VerifiedSelectionLabelCapability(
        "MELD",
        tmp_path / "unread.npz",
        _sha("7"),
        _sha("8"),
        12,
        {},
        _sha("9"),
        _sha("a"),
    )
    return _publish_outputs(
        strategies=_publication_bundle(),
        selection_capability=capability,
        analysis=analysis,
        aggregates=aggregates,
        private_output_root=tmp_path / "private",
        public_report_path=tmp_path / "public.json",
    )


def _rewrite_self_consistent_handoff(
    completed: CompletedModelSelectionEvaluation,
    *,
    mutate_artifact=None,
    mutate_receipt=None,
) -> str:
    artifact = json.loads(
        completed.private_artifact_path.read_text(encoding="utf-8")
    )
    if mutate_artifact is not None:
        mutate_artifact(artifact)
    completed.private_artifact_path.write_bytes(evaluator._json_bytes(artifact))
    receipt = json.loads(completed.private_receipt_path.read_text(encoding="utf-8"))
    receipt["lineage"]["private_reference_freeze_artifact_sha256"] = _file_sha256(
        completed.private_artifact_path
    )
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    completed.private_receipt_path.write_bytes(evaluator._json_bytes(receipt))
    return _file_sha256(completed.private_receipt_path)


def test_write_once_outputs_are_aggregate_and_have_joint_handoff(
    aggregate_fixture,
    tmp_path: Path,
) -> None:
    aggregates, analysis = aggregate_fixture
    capability = _VerifiedSelectionLabelCapability(
        "MELD",
        tmp_path / "unread.npz",
        _sha("7"),
        _sha("8"),
        12,
        {},
        _sha("9"),
        _sha("a"),
    )
    root = tmp_path / "private"
    public = tmp_path / "public.json"
    completed = _publish_outputs(
        strategies=_publication_bundle(),
        selection_capability=capability,
        analysis=analysis,
        aggregates=aggregates,
        private_output_root=root,
        public_report_path=public,
    )
    assert isinstance(completed, CompletedModelSelectionEvaluation)
    assert completed.private_artifact_path.name == PRIVATE_ARTIFACT_NAME
    assert completed.private_receipt_path.name == PRIVATE_RECEIPT_NAME
    report = json.loads(public.read_text(encoding="utf-8"))
    assert report["schema_version"] == PUBLIC_REPORT_SCHEMA
    validate_model_selection_public_report(report)
    assert "seed_results" not in report["per_seed_success"]
    assert report["per_seed_success"]["seed_level_numeric_details_withheld"] is True
    private = json.loads(
        completed.private_artifact_path.read_text(encoding="utf-8")
    )
    assert len(private["evaluation"]["per_seed_success"]["seed_results"]) == 5
    handoff = verify_model_selection_reference_freeze_receipt(
        completed.private_artifact_path,
        completed.private_receipt_path,
        completed.private_receipt_sha256,
    )
    assert handoff.dataset == "MELD"
    assert handoff.frozen_reference == "coverage_matched_recency"
    expected_gate = bool(
        all(
            private["evaluation"]["aggregate_gates"][name]
            for name in evaluator._MODEL_SELECTION_GATE_KEYS
        )
        and private["evaluation"]["holm_family"]["results"][
            "H1_primary_macro_f1"
        ]["multiplicity"]["rejected_at_familywise_alpha"]
        and private["evaluation"]["holm_family"]["results"][
            "H2_primary_mean_regret"
        ]["multiplicity"]["rejected_at_familywise_alpha"]
    )
    assert handoff.model_selection_gate_passed is expected_gate
    assert completed.model_selection_gate_passed is expected_gate
    with pytest.raises(FileExistsError):
        _publish_outputs(
            strategies=_publication_bundle(),
            selection_capability=capability,
            analysis=analysis,
            aggregates=aggregates,
            private_output_root=root,
            public_report_path=public,
        )


@pytest.mark.parametrize("gate_name", evaluator._MODEL_SELECTION_GATE_KEYS)
def test_receipt_verifier_rederives_each_aggregate_gate_fail_closed(
    aggregate_fixture,
    tmp_path: Path,
    gate_name: str,
) -> None:
    completed = _publish_fixture(aggregate_fixture, tmp_path)

    def mutate(artifact) -> None:
        gates = artifact["evaluation"]["aggregate_gates"]
        gates[gate_name] = not gates[gate_name]

    forged_receipt_sha = _rewrite_self_consistent_handoff(
        completed,
        mutate_artifact=mutate,
    )
    with pytest.raises(ModelSelectionEvaluationError, match="aggregate gate"):
        verify_model_selection_reference_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            forged_receipt_sha,
        )


def test_receipt_verifier_rejects_missing_aggregate_gate_field(
    aggregate_fixture,
    tmp_path: Path,
) -> None:
    completed = _publish_fixture(aggregate_fixture, tmp_path)

    def mutate(artifact) -> None:
        artifact["evaluation"]["aggregate_gates"].pop(
            "accuracy_no_harm_passed"
        )

    forged_receipt_sha = _rewrite_self_consistent_handoff(
        completed,
        mutate_artifact=mutate,
    )
    with pytest.raises(ModelSelectionEvaluationError, match="schema changed"):
        verify_model_selection_reference_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            forged_receipt_sha,
        )


@pytest.mark.parametrize(
    "attack",
    ("missing_h5", "missing_adjustment_field", "forged_primary_rejection"),
)
def test_receipt_verifier_rederives_complete_holm_family_fail_closed(
    aggregate_fixture,
    tmp_path: Path,
    attack: str,
) -> None:
    completed = _publish_fixture(aggregate_fixture, tmp_path)

    def mutate(artifact) -> None:
        results = artifact["evaluation"]["holm_family"]["results"]
        if attack == "missing_h5":
            results.pop("H5_current_only_increment")
        elif attack == "missing_adjustment_field":
            results["H3_emotion_constraint_increment"]["multiplicity"].pop(
                "holm_adjusted_p_value"
            )
        else:
            multiplicity = results["H1_primary_macro_f1"]["multiplicity"]
            multiplicity["rejected_at_familywise_alpha"] = not multiplicity[
                "rejected_at_familywise_alpha"
            ]

    forged_receipt_sha = _rewrite_self_consistent_handoff(
        completed,
        mutate_artifact=mutate,
    )
    with pytest.raises(ModelSelectionEvaluationError, match="[Hh]olm"):
        verify_model_selection_reference_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            forged_receipt_sha,
        )


def test_receipt_model_selection_boolean_is_not_a_trust_source(
    aggregate_fixture,
    tmp_path: Path,
) -> None:
    completed = _publish_fixture(aggregate_fixture, tmp_path)

    def mutate(receipt) -> None:
        completion = receipt["completion_contract"]
        completion["model_selection_gate_passed"] = not completion[
            "model_selection_gate_passed"
        ]

    forged_receipt_sha = _rewrite_self_consistent_handoff(
        completed,
        mutate_receipt=mutate,
    )
    with pytest.raises(ModelSelectionEvaluationError, match="handoff lineage changed"):
        verify_model_selection_reference_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            forged_receipt_sha,
        )
