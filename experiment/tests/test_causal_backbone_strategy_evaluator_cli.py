from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hva_affect.causal_backbone_evidence_runner as evidence_runner
import hva_affect.causal_backbone_history_staged_pipeline as history_pipeline
import hva_affect.causal_backbone_model_selection_evaluator as evaluator
import hva_affect.causal_backbone_strategy_staged_pipeline as strategy_pipeline
import hva_affect.production_source_snapshot_v1 as source_snapshot_module
from test_causal_fit_lineage_bootstrap import (
    _fake_source_snapshot,
    _load_cli_module,
)


VARIANTS = ("full", "no_vad", "no_history_3x3", "capacity_control")
SHA = "a" * 64


def _subparsers(module) -> dict[str, argparse.ArgumentParser]:
    parser = module.build_parser()
    action = next(
        item
        for item in parser._actions
        if isinstance(item, argparse._SubParsersAction)
    )
    return action.choices


def _snapshot(module, tmp_path: Path) -> SimpleNamespace:
    manifest = tmp_path / "source-snapshot.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return _fake_source_snapshot(
        {
            "experiment/scripts/run_causal_backbone_evidence.py": Path(
                module.__file__
            ).resolve()
        },
        module.ROOT.parent,
        manifest_path=manifest,
    )


def test_new_commands_are_typed_and_expose_no_outcome_or_later_role_path() -> None:
    module = _load_cli_module()
    choices = _subparsers(module)
    assert {"strategy-complete-selection", "evaluate-model-selection"}.issubset(
        choices
    )

    snapshot_fields = {
        "source_snapshot_manifest",
        "source_snapshot_manifest_sha256",
        "source_snapshot_worktree_root",
    }
    forbidden = {
        "label",
        "label_path",
        "role",
        "calibration",
        "holdout",
        "test",
        "registered_variant",
        "code",
    }
    strategy_fields = {
        action.dest
        for action in choices["strategy-complete-selection"]._actions
    }
    evaluator_fields = {
        action.dest for action in choices["evaluate-model-selection"]._actions
    }
    assert snapshot_fields.issubset(strategy_fields)
    assert snapshot_fields.issubset(evaluator_fields)
    assert {
        "history_complete_artifact",
        "full_history_anchor_artifact",
        "current_complete_artifact",
        "backbone_config",
        "full_anchor_backbone_config",
        "full_anchor_fit_receipt",
        "full_anchor_fit_receipt_sha256",
    }.issubset(strategy_fields)
    assert {
        "history_complete_artifact",
        "current_complete_artifact",
        "strategy_complete_artifact",
        "confirmatory_analysis",
        "public_report",
    }.issubset(evaluator_fields)
    assert not forbidden & strategy_fields
    assert not forbidden & evaluator_fields

    for command, subparser in choices.items():
        if command == "create-production-source-snapshot":
            # This is the bootstrap command that creates the manifest hash;
            # it cannot require a manifest or expected manifest hash yet.
            continue
        fields = {action.dest for action in subparser._actions}
        assert snapshot_fields.issubset(fields), command
        for field in snapshot_fields:
            action = next(item for item in subparser._actions if item.dest == field)
            assert action.required is True, (command, field)

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "strategy-complete-selection",
                "--fit-receipt-sha256",
                "not-a-sha",
            ]
        )
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "evaluate-model-selection",
                "--dataset",
                "MELD",
                "--calibration",
                "forbidden.npz",
            ]
        )


def test_cli_cannot_hash_one_snapshot_while_executing_from_another_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module()
    foreign_root = tmp_path / "foreign-detached"
    (foreign_root / "experiment" / "src" / "hva_affect").mkdir(parents=True)
    attestation = SimpleNamespace(
        worktree_root=foreign_root.resolve(),
        stable_code_paths=lambda: {
            "experiment/scripts/run_causal_backbone_evidence.py": Path(
                module.__file__
            ).resolve()
        },
    )
    monkeypatch.setattr(
        source_snapshot_module,
        "verify_production_source_snapshot",
        lambda **_kwargs: attestation,
    )
    args = SimpleNamespace(
        source_snapshot_manifest=tmp_path / "snapshot.json",
        source_snapshot_manifest_sha256=SHA,
        source_snapshot_worktree_root=foreign_root,
    )
    with pytest.raises(SystemExit, match="not executing from the attested"):
        module._verify_source_snapshot(args)


def test_strategy_cli_verifies_variant_anchor_current_before_outcome_free_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    full_path = module.ROOT / "configs" / "carma_affect_relation_meld_full_v1.json"
    no_vad_path = (
        module.ROOT / "configs" / "carma_affect_relation_meld_no_vad_v1.json"
    )
    full_model, full_run = module._load_backbone_config(full_path)
    no_vad_model, no_vad_run = module._load_backbone_config(no_vad_path)
    snapshot = _snapshot(module, tmp_path)
    events: list[str] = []
    variant_history = tmp_path / "no-vad-history.npz"
    full_history = tmp_path / "full-history.npz"

    monkeypatch.setattr(module, "_verify_source_snapshot", lambda _args: snapshot)
    monkeypatch.setattr(module, "capture_runtime_environment", lambda: {"runtime": "x"})

    def attest_history(artifact, _receipt, _sha):
        events.append("variant-history" if artifact == variant_history else "full-anchor")
        return SimpleNamespace(dataset="MELD")

    monkeypatch.setattr(module, "_verify_history_completion_triplet", attest_history)
    monkeypatch.setattr(
        module,
        "_load_backbone_config",
        lambda path: (full_model, full_run) if path == full_path else (no_vad_model, no_vad_run),
    )
    def current(**kwargs):
        assert kwargs["full_fit_receipt"] == tmp_path / "full-fit.json"
        assert kwargs["full_fit_receipt_sha256"] == "b" * 64
        events.append("current")
        return object()

    monkeypatch.setattr(module, "_verify_full_anchored_current", current)

    def upstream(**_kwargs):
        events.append("upstream")
        return "no_vad", no_vad_model, no_vad_run, object()

    monkeypatch.setattr(module, "_verify_strategy_upstream", upstream)
    monkeypatch.setattr(module, "_resolve_device", lambda _name: object())

    def complete(**kwargs):
        events.append("complete")
        assert kwargs["registered_variant"] == "no_vad"
        assert kwargs["model_config"] == no_vad_model
        assert kwargs["config_paths"] == {
            "model": no_vad_path,
            "production_source_snapshot_v1": snapshot.manifest_path,
        }
        assert kwargs["code_paths"] == snapshot.stable_code_paths()
        return SimpleNamespace(
            artifact_sha256="1" * 64,
            receipt_sha256="2" * 64,
            production_run_claim_sha256="3" * 64,
            policy_sha256="4" * 64,
        )

    monkeypatch.setattr(strategy_pipeline, "complete_strategy_selection", complete)
    args = SimpleNamespace(
        config=[("model", no_vad_path)],
        dataset="MELD",
        sidecar_dir=tmp_path,
        sidecar_manifest=tmp_path / "manifest.json",
        fit_receipt=tmp_path / "no-vad-fit.json",
        fit_receipt_sha256=SHA,
        full_anchor_fit_receipt=tmp_path / "full-fit.json",
        full_anchor_fit_receipt_sha256="b" * 64,
        backbone_config=no_vad_path,
        full_anchor_backbone_config=full_path,
        history_complete_artifact=variant_history,
        history_complete_receipt=tmp_path / "no-vad-history-receipt.json",
        history_complete_receipt_sha256=SHA,
        full_history_anchor_artifact=full_history,
        full_history_anchor_receipt=tmp_path / "full-history-receipt.json",
        full_history_anchor_receipt_sha256=SHA,
        current_complete_artifact=tmp_path / "current.npz",
        current_complete_receipt=tmp_path / "current-receipt.json",
        current_complete_receipt_sha256=SHA,
        private_output_dir=tmp_path / "strategy-private",
        device="cpu",
    )
    module._run_strategy_complete_selection(args)
    assert events == [
        "variant-history",
        "full-anchor",
        "current",
        "upstream",
        "complete",
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["registered_variant"] == "no_vad"
    assert summary["variant_derived_from_model_config"] is True
    assert summary["selection_label_file_accessed"] is False
    assert summary["calibration_unseal_authorized"] is False
    assert summary["source_snapshot_manifest_sha256"] == snapshot.manifest_sha256


def test_strategy_config_mapping_has_one_model_derived_variant_and_frozen_contract(
    tmp_path: Path,
) -> None:
    module = _load_cli_module()
    full = module.ROOT / "configs" / "carma_affect_relation_meld_full_v1.json"
    no_vad = module.ROOT / "configs" / "carma_affect_relation_meld_no_vad_v1.json"
    snapshot = tmp_path / "source-snapshot.json"
    snapshot.write_text("{}\n", encoding="utf-8")

    derived, _model, _run = module._load_exact_variant_config_mapping(
        backbone_config=full,
        config_paths={"model": full, "production_source_snapshot_v1": snapshot},
        dataset="MELD",
    )
    assert derived == "full"
    with pytest.raises(SystemExit, match="exactly its one"):
        module._load_exact_variant_config_mapping(
            backbone_config=full,
            config_paths={"full": full, "second_model": no_vad},
            dataset="MELD",
        )

    tampered = tmp_path / "tampered-contract.json"
    payload = json.loads(full.read_text(encoding="utf-8"))
    payload["experimental_contract"]["variant"] = "no_vad"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="experimental contract"):
        module._load_exact_variant_config_mapping(
            backbone_config=tampered,
            config_paths={"model": tampered},
            dataset="MELD",
        )


def _evaluation_args(module, tmp_path: Path) -> SimpleNamespace:
    config_paths = {
        variant: module.ROOT
        / "configs"
        / f"carma_affect_relation_meld_{variant}_v1.json"
        for variant in VARIANTS
    }
    return SimpleNamespace(
        dataset="MELD",
        sidecar_dir=tmp_path,
        sidecar_manifest=tmp_path / "manifest.json",
        fit_receipt=[
            (variant, tmp_path / f"{variant}-fit.json") for variant in VARIANTS
        ],
        fit_receipt_sha256=[(variant, SHA) for variant in VARIANTS],
        variant_config=[
            (variant, "model", config_paths[variant]) for variant in VARIANTS
        ],
        backbone_config=list(config_paths.items()),
        history_complete_artifact=[
            (variant, tmp_path / f"{variant}-history.npz") for variant in VARIANTS
        ],
        history_completion_receipt=[
            (variant, tmp_path / f"{variant}-history-receipt.json")
            for variant in VARIANTS
        ],
        history_completion_receipt_sha256=[(variant, SHA) for variant in VARIANTS],
        current_complete_artifact=tmp_path / "current.npz",
        current_complete_receipt=tmp_path / "current-receipt.json",
        current_complete_receipt_sha256=SHA,
        strategy_complete_artifact=[
            (variant, tmp_path / f"{variant}-strategy.npz") for variant in VARIANTS
        ],
        strategy_completion_receipt=[
            (variant, tmp_path / f"{variant}-strategy-receipt.json")
            for variant in VARIANTS
        ],
        strategy_completion_receipt_sha256=[(variant, SHA) for variant in VARIANTS],
        confirmatory_analysis=(
            module.ROOT / "configs" / "carma_confirmatory_analysis_v1.json"
        ),
        private_output_dir=tmp_path / "evaluation-private",
        public_report=tmp_path / "model-selection-public.json",
    )


def test_evaluator_cli_builds_exact_four_variant_bundle_before_single_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    args = _evaluation_args(module, tmp_path)
    snapshot = _snapshot(module, tmp_path)
    events: list[str] = []
    artifact_to_variant = {
        path: variant for variant, path in args.history_complete_artifact
    }
    backbone_to_variant = {path: variant for variant, path in args.backbone_config}

    monkeypatch.setattr(module, "_verify_source_snapshot", lambda _args: snapshot)
    monkeypatch.setattr(module, "capture_runtime_environment", lambda: {"runtime": "x"})

    def attest(artifact, _receipt, _sha):
        variant = artifact_to_variant[artifact]
        events.append(f"history:{variant}")
        return SimpleNamespace(dataset="MELD", registered_variant=variant)

    monkeypatch.setattr(module, "_verify_history_completion_triplet", attest)
    monkeypatch.setattr(
        module,
        "_verify_full_anchored_current",
        lambda **_kwargs: events.append("current") or object(),
    )

    def upstream(*, paths, **_kwargs):
        variant = backbone_to_variant[paths.backbone_config]
        model, run = module._load_backbone_config(paths.backbone_config)
        events.append(f"upstream:{variant}")
        return variant, model, run, SimpleNamespace(variant=variant)

    monkeypatch.setattr(module, "_verify_strategy_upstream", upstream)

    def evaluate(**kwargs):
        events.append("evaluator")
        assert tuple(kwargs["strategies"]) == VARIANTS
        for variant, source in kwargs["strategies"].items():
            assert isinstance(source, evaluator.StrategyProductionInput)
            assert source.upstream.variant == variant
            assert source.code_paths == snapshot.stable_code_paths()
            assert (
                source.config_paths["production_source_snapshot_v1"]
                == snapshot.manifest_path
            )
        selection = kwargs["selection_source"]
        assert isinstance(selection, evaluator.SelectionSidecarSource)
        assert not hasattr(selection, "role")
        assert not hasattr(selection, "label_path")
        assert (
            selection.config_paths["production_source_snapshot_v1"]
            == snapshot.manifest_path
        )
        assert kwargs["confirmatory_analysis_path"] == args.confirmatory_analysis
        return SimpleNamespace(
            private_artifact_sha256="1" * 64,
            private_receipt_sha256="2" * 64,
            public_report_sha256="3" * 64,
            frozen_reference="all_history",
            model_selection_gate_passed=False,
            prospective_power=0.81,
            power_gate_passed=True,
        )

    monkeypatch.setattr(evaluator, "run_model_selection_reference_freeze", evaluate)
    module._run_evaluate_model_selection(args)

    assert events == [
        *[f"history:{variant}" for variant in VARIANTS],
        "history:full",
        "current",
        *[f"upstream:{variant}" for variant in VARIANTS],
        "evaluator",
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["registered_variants"] == list(VARIANTS)
    assert summary["selection_label_access_limited_to_evaluator"] is True
    assert summary["model_selection_gate_passed"] is False
    assert summary["confirmatory_claim_authorized"] is False
    assert summary["calibration_unseal_authorized"] is False
    assert summary["external_test_unseal_authorized"] is False


@pytest.mark.parametrize(
    "invalid",
    [
        "/experiment/src/hva_affect/x.py",
        "experiment\\src\\hva_affect\\x.py",
        "experiment/src//hva_affect/x.py",
        "experiment/src/hva_affect/../x.py",
        "experiment/src/hva_affect/./x.py",
        "experiment/src/hva_affect/x.txt",
        "experiment/scripts/other.py",
        "other/src/hva_affect/x.py",
        "C:/experiment/src/hva_affect/x.py",
    ],
)
def test_all_source_consumers_reject_noncanonical_or_out_of_scope_keys(
    tmp_path: Path,
    invalid: str,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        evidence_runner._named_file_hashes({invalid: source}, "code_paths")  # noqa: SLF001
    with pytest.raises(ValueError):
        history_pipeline._named_live_file_hashes({invalid: source}, "code_paths")  # noqa: SLF001
    with pytest.raises(ValueError):
        strategy_pipeline._safe_named_file_hashes({invalid: source}, "code_paths")  # noqa: SLF001


def test_source_consumers_accept_nested_posix_keys_but_config_names_stay_tokens(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    key = "experiment/src/hva_affect/future/calibration/gate.py"
    for observed in (
        evidence_runner._named_file_hashes({key: source}, "code_paths"),  # noqa: SLF001
        history_pipeline._named_live_file_hashes({key: source}, "code_paths"),  # noqa: SLF001
        strategy_pipeline._safe_named_file_hashes({key: source}, "code_paths"),  # noqa: SLF001
    ):
        assert tuple(observed) == (key,)

    for consumer in (
        evidence_runner._named_file_hashes,  # noqa: SLF001
        history_pipeline._named_live_file_hashes,  # noqa: SLF001
        strategy_pipeline._safe_named_file_hashes,  # noqa: SLF001
    ):
        with pytest.raises(ValueError):
            consumer({"config/with/slash": source}, "config_paths")

    ambiguous = {
        "experiment/src/hva_affect/Future.py": source,
        "experiment/src/hva_affect/future.py": source,
    }
    with pytest.raises(ValueError):
        evidence_runner._named_file_hashes(ambiguous, "code_paths")  # noqa: SLF001
    with pytest.raises(ValueError):
        history_pipeline._named_live_file_hashes(ambiguous, "code_paths")  # noqa: SLF001
    with pytest.raises(ValueError):
        strategy_pipeline._safe_named_file_hashes(ambiguous, "code_paths")  # noqa: SLF001

    module = _load_cli_module()
    snapshot = _snapshot(module, tmp_path)
    with pytest.raises(SystemExit, match="reserved"):
        module._bind_source_snapshot_config(  # noqa: SLF001
            {"production_source_snapshot_v1": source}, snapshot
        )
