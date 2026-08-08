from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hva_affect.causal_backbone_evidence_runner import (
    EXPECTED_SEEDS,
    SELECTION_ROLE,
    run_fit_preflight,
)
from hva_affect.causal_backbone_evidence_stage_b import (
    CurrentOnlyFoldOutput,
    StageBContractError,
    fit_only_lineage_values,
    materialize_verified_fit_for_stage_b,
    produce_independent_current_only_fit_oof,
    validate_fit_only_lineage_values,
    write_fit_only_lineage,
    write_fit_protocol_map,
)
from hva_affect.causal_backbone_history_staged_pipeline import (
    HistoryStagedPipelineError,
    VerifiedHistoryCompletionAttestation,
)
from hva_affect.causal_multimodal_backbone import CausalBackboneConfig
from hva_affect.emotiontalk_causal_backbone_runner import BackboneRunConfig
from test_causal_backbone_evidence_runner import (
    ENVIRONMENT,
    _lineage_files,
    _make_emotiontalk_sidecars,
    _sha,
)


def _write_fake_checkpoint(request: object) -> None:
    run = (
        request.checkpoint_root  # type: ignore[attr-defined]
        / f"seed_{request.seed:05d}"  # type: ignore[attr-defined]
        / f"fold_{request.fold:02d}"  # type: ignore[attr-defined]
    )
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoint.pt").write_bytes(b"fit-only-checkpoint")
    (run / "text_processor.joblib").write_bytes(b"fit-only-processor")


def test_fit_bootstrap_runs_without_selection_files_or_history_producer(
    tmp_path: Path,
) -> None:
    manifest = _make_emotiontalk_sidecars(tmp_path, poison_selection=True)
    configs, code = _lineage_files(tmp_path)
    receipt = tmp_path / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=receipt,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )

    # Physical absence is stronger than an np.load spy: every fit-only loader,
    # map/lineage builder, validator, and producer below must succeed without
    # even stat'ing either selection payload.
    (tmp_path / f"features_{SELECTION_ROLE}.npz").unlink()
    (tmp_path / f"labels_{SELECTION_ROLE}.npz").unlink()
    fit = materialize_verified_fit_for_stage_b(
        receipt_path=receipt,
        expected_receipt_sha256=preflight.receipt_sha256,
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    fit_map = write_fit_protocol_map(
        fit,
        receipt_path=receipt,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=tmp_path / "fit-map.npz",
    )
    lineage = write_fit_only_lineage(
        fit,
        fit_map=fit_map,
        receipt_path=receipt,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=tmp_path / "fit-lineage.npz",
    )

    callback_requests = []

    def callback(request):
        callback_requests.append(request)
        assert request.heldout_labels_materialized is False
        assert not hasattr(request, "heldout_labels")
        assert not hasattr(request, "labels")
        assert request.fit_lineage_source_identity_sha256 == (
            lineage.source_identity_sha256
        )
        _write_fake_checkpoint(request)
        probability = np.full(
            (len(request.heldout_indices), len(fit.label_order)),
            1.0 / len(fit.label_order),
            dtype=np.float32,
        )
        return CurrentOnlyFoldOutput(probability, "a" * 64)

    folds = np.tile(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        (len(EXPECTED_SEEDS), 1),
    )
    produced = produce_independent_current_only_fit_oof(
        fit=fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_path=receipt,
        expected_fit_preflight_receipt_sha256=preflight.receipt_sha256,
        fold_by_seed_row=folds,
        outer_folds=2,
        checkpoint_root=tmp_path / "current-checkpoints",
        artifact_path=tmp_path / "current-fit.npz",
        producer_receipt_path=tmp_path / "current-fit-receipt.json",
        model_config_sha256=_sha("model"),
        run_config_sha256=_sha("run"),
        model_config_semantic_sha256=_sha("model-semantic"),
        run_config_semantic_sha256=_sha("run-semantic"),
        source_code_sha256=_sha("code"),
        runtime_environment_sha256=_sha("runtime"),
        fold_callback=callback,
    )
    assert produced.artifact_path.is_file()
    assert len(callback_requests) == len(EXPECTED_SEEDS) * 2
    assert not (tmp_path / "producer.npz").exists()
    parameters = inspect.signature(produce_independent_current_only_fit_oof).parameters
    assert "producer" not in parameters
    assert "selection" not in parameters


def test_fit_lineage_rejects_selection_fields_and_claims(tmp_path: Path) -> None:
    manifest = _make_emotiontalk_sidecars(tmp_path, poison_selection=True)
    configs, code = _lineage_files(tmp_path)
    receipt = tmp_path / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=receipt,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    fit_map = write_fit_protocol_map(
        preflight.fit,
        receipt_path=receipt,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=tmp_path / "fit-map.npz",
    )
    values = fit_only_lineage_values(
        preflight.fit,
        fit_map=fit_map,
        receipt_path=receipt,
        expected_receipt_sha256=preflight.receipt_sha256,
    )
    injected = dict(values)
    injected["selection_labels"] = np.asarray([1], dtype=np.int64)
    with pytest.raises(StageBContractError, match="schema changed"):
        validate_fit_only_lineage_values(
            injected,
            fit=preflight.fit,
            fit_map=fit_map,
            receipt_path=receipt,
            expected_receipt_sha256=preflight.receipt_sha256,
        )
    claimed = dict(values)
    claimed["contains_selection_material"] = np.asarray(True)
    with pytest.raises(StageBContractError, match="differs|selection material"):
        validate_fit_only_lineage_values(
            claimed,
            fit=preflight.fit,
            fit_map=fit_map,
            receipt_path=receipt,
            expected_receipt_sha256=preflight.receipt_sha256,
        )


def _load_cli_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_causal_backbone_evidence.py"
    spec = importlib.util.spec_from_file_location("causal_evidence_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_source_snapshot(
    code: dict[str, Path], root: Path, *, manifest_path: Path | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        manifest_path=(
            next(iter(code.values())) if manifest_path is None else manifest_path
        ),
        manifest_sha256=_sha("source-snapshot-manifest"),
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        worktree_root=root,
        stable_code_paths=lambda: dict(code),
    )


def _snapshot_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--source-snapshot-manifest",
        str(tmp_path / "production-source-snapshot.json"),
        "--source-snapshot-manifest-sha256",
        _sha("source-snapshot-manifest"),
        "--source-snapshot-worktree-root",
        str(tmp_path / "detached-worktree"),
    ]


def test_fit_lineage_create_and_validate_cli_need_no_selection_or_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _make_emotiontalk_sidecars(tmp_path, poison_selection=True)
    configs, code = _lineage_files(tmp_path)
    preflight_configs = {
        **configs,
        "production_source_snapshot_v1": next(iter(code.values())),
    }
    receipt = tmp_path / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=manifest,
        receipt_path=receipt,
        config_paths=preflight_configs,
        code_paths=code,
    )
    (tmp_path / f"features_{SELECTION_ROLE}.npz").unlink()
    (tmp_path / f"labels_{SELECTION_ROLE}.npz").unlink()
    module = _load_cli_module()
    source_snapshot = _fake_source_snapshot(code, module.ROOT.parent)
    monkeypatch.setattr(module, "_verify_source_snapshot", lambda _args: source_snapshot)
    parser = module.build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    fit_dests = {
        action.dest
        for action in subparser_action.choices["current-only-fit"]._actions
    }
    assert "fit_lineage" in fit_dests
    assert "producer" not in fit_dests
    assert {"private_output_dir", "resume"}.issubset(fit_dests)
    assert "checkpoint_root" not in fit_dests
    assert {
        "source_snapshot_manifest",
        "source_snapshot_manifest_sha256",
        "source_snapshot_worktree_root",
    }.issubset(fit_dests)
    complete_dests = {
        action.dest
        for action in subparser_action.choices[
            "current-only-complete-selection"
        ]._actions
    }
    assert {
        "private_output_dir",
        "history_complete_artifact",
        "history_completion_receipt",
        "history_completion_receipt_sha256",
    }.issubset(complete_dests)
    assert not {
        "checkpoint_root",
        "fit_artifact",
        "current_only_cache",
        "completion_receipt",
    } & complete_dests
    history_artifact_action = next(
        action
        for action in subparser_action.choices[
            "current-only-complete-selection"
        ]._actions
        if action.dest == "history_complete_artifact"
        and "--history-complete-artifact" in action.option_strings
    )
    assert "history-complete-outcome.npz" in history_artifact_action.help

    common = [
        "--dataset",
        "EmotionTalk",
        "--sidecar-dir",
        str(tmp_path),
        "--sidecar-manifest",
        str(manifest),
        "--fit-receipt",
        str(receipt),
        "--fit-receipt-sha256",
        preflight.receipt_sha256,
        *_snapshot_cli_args(tmp_path),
    ]
    for name, path in configs.items():
        common.extend(["--config", f"{name}={path}"])
    fit_map_path = tmp_path / "cli-fit-map.npz"
    lineage_path = tmp_path / "cli-fit-lineage.npz"
    artifacts = [
        "--fit-map",
        str(fit_map_path),
        "--fit-lineage",
        str(lineage_path),
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_causal_backbone_evidence.py", "fit-lineage-create", *common, *artifacts],
    )
    module.main()
    created = json.loads(capsys.readouterr().out)
    assert created["selection_payload_opened"] is False
    assert created["history_producer_required"] is False
    assert fit_map_path.is_file() and lineage_path.is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_causal_backbone_evidence.py",
            "fit-lineage-validate",
            *common,
            *artifacts,
        ],
    )
    module.main()
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "fit_only_alignment_lineage_valid"
    assert validated["selection_payload_opened"] is False


def test_current_only_cli_requires_actual_frozen_config_and_exact_snapshot_code(
    tmp_path: Path,
) -> None:
    module = _load_cli_module()
    frozen_config = tmp_path / "backbone.json"
    frozen_config.write_text("{}\n", encoding="utf-8")
    other_config = tmp_path / "other.json"
    other_config.write_text("{}\n", encoding="utf-8")
    source = module.ROOT / "src" / "hva_affect"
    canonical = {
        "experiment/scripts/run_causal_backbone_evidence.py": Path(
            module.__file__
        ).resolve(),
        "experiment/src/hva_affect/causal_affect_relation.py": (
            source / "causal_affect_relation.py"
        ),
    }
    source_snapshot = _fake_source_snapshot(canonical, module.ROOT.parent)
    module._verify_frozen_production_inputs(
        backbone_config=frozen_config,
        config_paths={"registered_backbone": frozen_config},
        code_paths=canonical,
        source_snapshot=source_snapshot,
    )
    with pytest.raises(SystemExit, match="not one of the preflight-frozen configs"):
        module._verify_frozen_production_inputs(
            backbone_config=other_config,
            config_paths={"registered_backbone": frozen_config},
            code_paths=canonical,
            source_snapshot=source_snapshot,
        )
    missing_relation = dict(canonical)
    missing_relation.pop("experiment/src/hva_affect/causal_affect_relation.py")
    with pytest.raises(SystemExit, match="source snapshot"):
        module._verify_frozen_production_inputs(
            backbone_config=frozen_config,
            config_paths={"registered_backbone": frozen_config},
            code_paths=missing_relation,
            source_snapshot=source_snapshot,
        )
    wrong_relation = dict(canonical)
    wrong_relation["experiment/src/hva_affect/causal_affect_relation.py"] = (
        source / "causal_multimodal_backbone.py"
    )
    with pytest.raises(SystemExit, match="source snapshot"):
        module._verify_frozen_production_inputs(
            backbone_config=frozen_config,
            config_paths={"registered_backbone": frozen_config},
            code_paths=wrong_relation,
            source_snapshot=source_snapshot,
        )


@pytest.mark.parametrize(
    "attack",
    [
        "wrong receipt SHA",
        "synthetic receipt",
        "different private root",
        "tampered history artifact",
        "tampered production claim",
    ],
)
def test_current_only_cli_history_attestation_fails_before_selection_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    module = _load_cli_module()
    backbone_path = tmp_path / "backbone.json"
    backbone_path.write_text("{}\n", encoding="utf-8")
    private_root = tmp_path / "current-only-private"
    private_root.mkdir()
    events: list[str] = []

    import hva_affect.causal_backbone_current_only_pipeline as current_pipeline
    import hva_affect.causal_backbone_evidence_stage_b as stage_b
    import hva_affect.causal_backbone_history_staged_pipeline as history_pipeline
    import hva_affect.emotiontalk_causal_backbone_runner as backbone_runner

    source_snapshot = _fake_source_snapshot(
        {
            "experiment/scripts/run_causal_backbone_evidence.py": Path(
                module.__file__
            ).resolve()
        },
        module.ROOT.parent,
    )
    monkeypatch.setattr(module, "_verify_source_snapshot", lambda _args: source_snapshot)
    monkeypatch.setattr(module, "_verify_frozen_production_inputs", lambda **_k: None)
    monkeypatch.setattr(
        module,
        "_load_backbone_config",
        lambda _path: (CausalBackboneConfig(), BackboneRunConfig()),
    )
    monkeypatch.setattr(module, "_resolve_device", lambda _name: object())
    monkeypatch.setattr(module, "_production_code_sha256", lambda _code: "a" * 64)
    monkeypatch.setattr(
        module,
        "materialize_selection_features_after_receipt",
        lambda **_kwargs: events.append("selection")
        or pytest.fail("selection feature was accessed after failed attestation"),
    )
    monkeypatch.setattr(
        stage_b,
        "materialize_verified_fit_for_stage_b",
        lambda **_kwargs: SimpleNamespace(dataset="EmotionTalk"),
    )
    monkeypatch.setattr(stage_b, "load_fit_protocol_map", lambda *_a, **_k: object())
    monkeypatch.setattr(stage_b, "load_fit_only_lineage", lambda *_a, **_k: object())
    monkeypatch.setattr(
        current_pipeline,
        "current_only_production_claim_sha256",
        lambda **_kwargs: "b" * 64,
    )
    monkeypatch.setattr(
        current_pipeline,
        "load_attested_history_fit_alignment_view",
        lambda *_args, **_kwargs: events.append("history-fit-view")
        or pytest.fail("history view loaded after failed attestation"),
    )
    monkeypatch.setattr(
        backbone_runner,
        "_runtime_environment",
        lambda _device: {"runtime": "test"},
    )

    def reject_attestation(*_args, **_kwargs):
        events.append("attestation")
        raise HistoryStagedPipelineError(attack)

    monkeypatch.setattr(
        history_pipeline,
        "verify_history_completion_production_attestation",
        reject_attestation,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_causal_backbone_evidence.py",
            "current-only-complete-selection",
            "--dataset",
            "EmotionTalk",
            "--sidecar-dir",
            str(tmp_path),
            "--sidecar-manifest",
            str(tmp_path / "manifest.json"),
            "--fit-receipt",
            str(tmp_path / "fit-preflight.json"),
            "--fit-receipt-sha256",
            "c" * 64,
            "--fit-map",
            str(tmp_path / "fit-map.npz"),
            "--fit-lineage",
            str(tmp_path / "fit-lineage.npz"),
            "--backbone-config",
            str(backbone_path),
            "--private-output-dir",
            str(private_root),
            "--history-complete-artifact",
            str(tmp_path / "history-private" / "history-complete-outcome.npz"),
            "--fit-producer-receipt-sha256",
            "d" * 64,
            "--history-completion-receipt",
            str(tmp_path / "history-private" / "history-complete-receipt.json"),
            "--history-completion-receipt-sha256",
            "e" * 64,
            *_snapshot_cli_args(tmp_path),
        ],
    )
    with pytest.raises(HistoryStagedPipelineError, match=attack):
        module.main()
    assert events == ["attestation"]
    main_source = inspect.getsource(module.main)
    assert main_source.index(
        "history_attestation = _attest_history_completion_for_current_only"
    ) < main_source.index(
        "selection = materialize_selection_features_after_receipt"
    )


def test_current_only_history_attestation_binds_all_frozen_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module()
    import hva_affect.causal_backbone_history_staged_pipeline as history_pipeline

    config_path = tmp_path / "backbone.json"
    code_path = tmp_path / "runner.py"
    config_path.write_text("{}\n", encoding="utf-8")
    code_path.write_text("pass\n", encoding="utf-8")
    model = CausalBackboneConfig()
    run = BackboneRunConfig()
    preflight_sha = _sha("fit-preflight")
    preflight_runtime_sha = _sha("preflight-runtime")
    execution_runtime_sha = _sha("execution-runtime")
    artifact_path = (tmp_path / "history-complete-outcome.npz").resolve()
    receipt_path = (tmp_path / "history-complete-receipt.json").resolve()
    attestation = VerifiedHistoryCompletionAttestation(
        dataset="EmotionTalk",
        artifact_path=artifact_path,
        artifact_sha256=_sha("history-artifact"),
        completion_receipt_path=receipt_path,
        completion_receipt_sha256=_sha("history-receipt"),
        fit_producer_receipt_path=(tmp_path / "history-fit-receipt.json"),
        fit_producer_receipt_sha256=_sha("history-fit-receipt"),
        source_identity_sha256=_sha("history-source"),
        checkpoint_manifest_sha256=_sha("history-checkpoints"),
        production_run_claim_sha256=_sha("history-claim"),
        fit_preflight_receipt_sha256=preflight_sha,
        config_sha256={"backbone": module._sha256(config_path)},
        code_sha256={"runner": module._sha256(code_path)},
        runtime_environment_sha256=preflight_runtime_sha,
        execution_environment_sha256=execution_runtime_sha,
        model_config_sha256=module._canonical_sha256(asdict(model)),
        run_config_sha256=module._canonical_sha256(asdict(run)),
        utility_config_sha256=_sha("history-utility"),
    )
    returned: list[VerifiedHistoryCompletionAttestation] = []

    def verify(artifact, receipt, expected_sha):
        assert Path(artifact).resolve() == artifact_path
        assert Path(receipt).resolve() == receipt_path
        assert expected_sha == attestation.completion_receipt_sha256
        returned.append(attestation)
        return attestation

    monkeypatch.setattr(
        history_pipeline,
        "verify_history_completion_production_attestation",
        verify,
    )
    kwargs = {
        "artifact_path": artifact_path,
        "completion_receipt_path": receipt_path,
        "expected_completion_receipt_sha256": (
            attestation.completion_receipt_sha256
        ),
        "dataset": "EmotionTalk",
        "fit_preflight_receipt_sha256": preflight_sha,
        "config_paths": {"backbone": config_path},
        "code_paths": {"runner": code_path},
        "model_config": model,
        "run_config": run,
        "runtime_environment_sha256": preflight_runtime_sha,
        "execution_environment_sha256": execution_runtime_sha,
    }
    assert module._attest_history_completion_for_current_only(
        **kwargs
    ) is attestation
    assert returned == [attestation]

    for field, replacement in (
        ("fit_preflight_receipt_sha256", "0" * 64),
        ("config_sha256", {"backbone": "0" * 64}),
        ("code_sha256", {"runner": "0" * 64}),
        ("runtime_environment_sha256", "0" * 64),
        ("execution_environment_sha256", "0" * 64),
        ("model_config_sha256", "0" * 64),
        ("run_config_sha256", "0" * 64),
    ):
        changed = replace(attestation, **{field: replacement})
        monkeypatch.setattr(
            history_pipeline,
            "verify_history_completion_production_attestation",
            lambda *_args, _changed=changed, **_kwargs: _changed,
        )
        with pytest.raises(SystemExit, match=field):
            module._attest_history_completion_for_current_only(**kwargs)


def test_history_cli_is_staged_and_fit_has_no_selection_capability() -> None:
    module = _load_cli_module()
    parser = module.build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert {"history-fit", "history-complete-selection"}.issubset(
        subparser_action.choices
    )
    fit_dests = {
        action.dest for action in subparser_action.choices["history-fit"]._actions
    }
    assert not any("selection" in name for name in fit_dests)
    assert {
        "fit_map",
        "backbone_config",
        "utility_config",
        "private_output_dir",
    }.issubset(fit_dests)
    complete_dests = {
        action.dest
        for action in subparser_action.choices["history-complete-selection"]._actions
    }
    assert {
        "fit_outcome_sha256",
        "fit_targets_sha256",
        "fit_producer_receipt_sha256",
    }.issubset(complete_dests)


def test_history_cli_requires_frozen_utility_and_exact_code_set(tmp_path: Path) -> None:
    module = _load_cli_module()
    backbone = tmp_path / "backbone.json"
    utility = tmp_path / "utility.json"
    other = tmp_path / "other.json"
    for path in (backbone, utility, other):
        path.write_text("{}\n", encoding="utf-8")
    canonical = {
        "experiment/scripts/run_causal_backbone_evidence.py": Path(
            module.__file__
        ).resolve(),
        "experiment/src/hva_affect/causal_backbone_history_staged_pipeline.py": (
            module.ROOT
            / "src"
            / "hva_affect"
            / "causal_backbone_history_staged_pipeline.py"
        ),
    }
    source_snapshot = _fake_source_snapshot(canonical, module.ROOT.parent)
    module._verify_frozen_production_inputs(
        backbone_config=backbone,
        utility_config=utility,
        config_paths={"backbone": backbone, "utility": utility},
        code_paths=canonical,
        source_snapshot=source_snapshot,
    )
    with pytest.raises(SystemExit, match="utility-config"):
        module._verify_frozen_production_inputs(
            backbone_config=backbone,
            utility_config=other,
            config_paths={"backbone": backbone, "utility": utility},
            code_paths=canonical,
            source_snapshot=source_snapshot,
        )
    extra = dict(canonical)
    extra["experiment/src/hva_affect/unfrozen_extra.py"] = next(
        iter(canonical.values())
    )
    with pytest.raises(SystemExit, match="source snapshot"):
        module._verify_frozen_production_inputs(
            backbone_config=backbone,
            utility_config=utility,
            config_paths={"backbone": backbone, "utility": utility},
            code_paths=extra,
            source_snapshot=source_snapshot,
        )
