from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hva_affect.causal_backbone_joint_model_selection_freeze as joint_module
import hva_affect.production_source_snapshot_v1 as snapshot_module
from test_causal_fit_lineage_bootstrap import _load_cli_module


SHA = "a" * 64


def _subparsers(module) -> dict[str, argparse.ArgumentParser]:
    action = next(
        item
        for item in module.build_parser()._actions
        if isinstance(item, argparse._SubParsersAction)
    )
    return action.choices


def _fake_snapshot(module, tmp_path: Path) -> SimpleNamespace:
    manifest = tmp_path / "snapshot-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        manifest_path=manifest,
        manifest_sha256="1" * 64,
        commit_sha="2" * 40,
        tree_sha="3" * 40,
        worktree_root=module.ROOT.parent.resolve(),
        stable_code_paths=lambda: {
            "experiment/scripts/run_causal_backbone_evidence.py": Path(
                module.__file__
            ).resolve()
        },
    )


def _fake_joint_attestation(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_roster=("EmotionTalk", "MELD"),
        artifact_path=tmp_path / "never-print-private-artifact.json",
        artifact_sha256="4" * 64,
        receipt_path=tmp_path / "never-print-private-receipt.json",
        receipt_sha256="5" * 64,
        public_report_sha256="6" * 64,
        analysis_config_sha256="7" * 64,
        source_snapshot_manifest_sha256="1" * 64,
        source_snapshot_git_commit="2" * 40,
        source_snapshot_git_tree="3" * 40,
        source_snapshot_code_bundle_sha256="a" * 64,
        frozen_reference_by_dataset={
            "EmotionTalk": "all_history",
            "MELD": "coverage_matched_recency",
        },
        prospective_power_by_dataset={"EmotionTalk": 0.81, "MELD": 0.83},
        power_gate_passed_by_dataset={"EmotionTalk": True, "MELD": True},
        upstream_artifact_sha256_by_dataset={
            "EmotionTalk": "8" * 64,
            "MELD": "9" * 64,
        },
        upstream_receipt_sha256_by_dataset={
            "EmotionTalk": "b" * 64,
            "MELD": "c" * 64,
        },
        upstream_public_report_sha256_by_dataset={
            "EmotionTalk": "d" * 64,
            "MELD": "e" * 64,
        },
        cross_variant_alignment_sha256_by_dataset={
            "EmotionTalk": "f" * 64,
            "MELD": "0" * 64,
        },
        model_selection_gate_attested_by_dataset={
            "EmotionTalk": True,
            "MELD": True,
        },
        model_selection_gate_passed_by_dataset={
            "EmotionTalk": True,
            "MELD": True,
        },
        calibration_stage_workflow_authorized=True,
        failure_reasons=(),
    )


def test_final_entrypoint_parsers_are_exact_typed_and_outcome_free() -> None:
    module = _load_cli_module()
    cli_source = Path(module.__file__).read_text(encoding="utf-8")
    disable_bytecode = cli_source.index("sys.dont_write_bytecode = True")
    sanitize_import_path = cli_source.index("sys.path[:] =")
    first_non_builtin_import = cli_source.index("import argparse")
    bootstrap_call = cli_source.index(
        '_bootstrap_require_plain_python_package(ROOT / "src")'
    )
    path_insert = cli_source.index("sys.path.insert")
    first_package_import = cli_source.index("from hva_affect")
    assert disable_bytecode < sanitize_import_path < first_non_builtin_import
    assert first_non_builtin_import < bootstrap_call < path_insert < first_package_import
    assert sys.dont_write_bytecode is True
    choices = _subparsers(module)
    commands = {
        "create-production-source-snapshot",
        "verify-production-source-snapshot",
        "run-joint-model-selection-freeze",
        "verify-joint-model-selection-freeze",
    }
    assert commands.issubset(choices)

    create_fields = {
        action.dest
        for action in choices["create-production-source-snapshot"]._actions
    }
    assert {
        "source_snapshot_worktree_root",
        "source_snapshot_output_manifest",
    }.issubset(create_fields)

    snapshot_fields = {
        "source_snapshot_manifest",
        "source_snapshot_manifest_sha256",
        "source_snapshot_worktree_root",
    }
    for command in commands - {"create-production-source-snapshot"}:
        fields = {action.dest for action in choices[command]._actions}
        assert snapshot_fields.issubset(fields)
        for field in snapshot_fields:
            action = next(
                item for item in choices[command]._actions if item.dest == field
            )
            assert action.required is True

    run_fields = {
        action.dest
        for action in choices["run-joint-model-selection-freeze"]._actions
    }
    assert {
        "emotiontalk_model_selection_artifact",
        "emotiontalk_model_selection_receipt",
        "emotiontalk_model_selection_receipt_sha256",
        "meld_model_selection_artifact",
        "meld_model_selection_receipt",
        "meld_model_selection_receipt_sha256",
        "joint_private_output_root",
        "joint_public_report",
    }.issubset(run_fields)
    verify_fields = {
        action.dest
        for action in choices["verify-joint-model-selection-freeze"]._actions
    }
    assert {
        "joint_private_artifact",
        "joint_private_receipt",
        "joint_private_receipt_sha256",
    }.issubset(verify_fields)

    forbidden_parameters = {
        "dataset",
        "label",
        "label_path",
        "probabilities",
        "predictions",
        "role",
        "calibration",
        "holdout",
        "test",
    }
    for command in commands:
        fields = {action.dest for action in choices[command]._actions}
        assert not forbidden_parameters & fields

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "verify-joint-model-selection-freeze",
                "--joint-private-receipt-sha256",
                "NOT-A-SHA",
            ]
        )


@pytest.mark.parametrize(
    "attack_path",
    (
        Path("ignored_native_shadow.pyd"),
        Path("ignored_native_shadow.so"),
        Path("__pycache__") / "ignored_shadow.pyc",
        Path("..") / "torch.py",
        Path("..") / "numpy.py",
    ),
)
def test_bootstrap_rejects_import_shadow_before_parser_or_package_import(
    tmp_path: Path,
    attack_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    scripts = experiment / "scripts"
    package = experiment / "src" / "hva_affect"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    attack = package / attack_path
    attack.parent.mkdir(parents=True, exist_ok=True)
    attack.write_bytes(b"unattested import shadow")
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_causal_backbone_evidence.py"
    )
    script = scripts / source.name
    shutil.copyfile(source, script)

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode != 0
    assert "production package bootstrap integrity check failed closed" in (
        result.stderr
    )
    assert "usage:" not in result.stdout.lower()
    assert "ModuleNotFoundError" not in result.stderr


def test_script_directory_cannot_shadow_stdlib_before_bootstrap(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    scripts = experiment / "scripts"
    package = experiment / "src" / "hva_affect"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "causal_backbone_evidence_runner.py").write_text(
        "def capture_runtime_environment():\n"
        "    return {}\n"
        "def materialize_selection_features_after_receipt(*args, **kwargs):\n"
        "    raise AssertionError('not called by --help')\n"
        "def run_fit_preflight(*args, **kwargs):\n"
        "    raise AssertionError('not called by --help')\n",
        encoding="utf-8",
    )
    (scripts / "argparse.py").write_text(
        "raise RuntimeError('ARGPARSE_SHADOW_EXECUTED')\n",
        encoding="utf-8",
    )
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_causal_backbone_evidence.py"
    )
    script = scripts / source.name
    shutil.copyfile(source, script)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "ARGPARSE_SHADOW_EXECUTED" not in result.stderr


def test_source_snapshot_create_and_verify_call_v1_apis_and_print_no_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    snapshot = _fake_snapshot(module, tmp_path)
    observed: list[tuple[str, object]] = []
    detached = tmp_path / "detached-worktree"
    output = tmp_path / "external" / "snapshot.json"

    def create_api(**kwargs):
        observed.append(("create", kwargs))
        return snapshot

    monkeypatch.setattr(snapshot_module, "create_production_source_snapshot", create_api)
    module._run_create_production_source_snapshot(
        SimpleNamespace(
            source_snapshot_worktree_root=detached,
            source_snapshot_output_manifest=output,
        )
    )
    create_summary = json.loads(capsys.readouterr().out)
    assert observed.pop() == (
        "create",
        {"worktree_root": detached, "output_path": output},
    )
    assert create_summary["operation"] == "create"
    assert str(detached) not in json.dumps(create_summary)
    assert str(output) not in json.dumps(create_summary)

    manifest = tmp_path / "source.json"

    def verify_api(**kwargs):
        observed.append(("verify", kwargs))
        return snapshot

    monkeypatch.setattr(snapshot_module, "verify_production_source_snapshot", verify_api)
    module._run_verify_production_source_snapshot(
        SimpleNamespace(
            source_snapshot_manifest=manifest,
            source_snapshot_manifest_sha256=SHA,
            source_snapshot_worktree_root=module.ROOT.parent,
        )
    )
    verify_summary = json.loads(capsys.readouterr().out)
    assert observed.pop() == (
        "verify",
        {
            "manifest_path": manifest,
            "expected_manifest_sha256": SHA,
            "worktree_root": module.ROOT.parent,
        },
    )
    assert verify_summary["operation"] == "verify"
    assert verify_summary["executing_from_verified_snapshot"] is True
    assert str(manifest) not in json.dumps(verify_summary)


def test_joint_run_uses_exact_two_typed_handoffs_then_reverifies_and_redacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    source_snapshot = _fake_snapshot(module, tmp_path)
    joint_attestation = _fake_joint_attestation(tmp_path)
    monkeypatch.setattr(module, "_verify_source_snapshot", lambda _args: source_snapshot)
    events: list[tuple[str, object]] = []
    completed = SimpleNamespace(
        private_artifact_path=tmp_path / "joint-model-selection-freeze.json",
        private_receipt_path=tmp_path / "joint-model-selection-freeze-receipt.json",
        private_receipt_sha256="5" * 64,
    )

    def run_api(**kwargs):
        events.append(("run", kwargs))
        return completed

    def verify_api(*args, **kwargs):
        events.append(("verify", (args, kwargs)))
        return joint_attestation

    monkeypatch.setattr(joint_module, "run_joint_model_selection_freeze", run_api)
    monkeypatch.setattr(
        joint_module, "verify_joint_model_selection_freeze_receipt", verify_api
    )
    args = SimpleNamespace(
        emotiontalk_model_selection_artifact=tmp_path / "emotiontalk-artifact.json",
        emotiontalk_model_selection_receipt=tmp_path / "emotiontalk-receipt.json",
        emotiontalk_model_selection_receipt_sha256="b" * 64,
        meld_model_selection_artifact=tmp_path / "meld-artifact.json",
        meld_model_selection_receipt=tmp_path / "meld-receipt.json",
        meld_model_selection_receipt_sha256="c" * 64,
        joint_private_output_root=tmp_path / "private-joint",
        joint_public_report=tmp_path / "public-joint.json",
    )
    module._run_joint_model_selection_freeze(args)

    assert [event[0] for event in events] == ["run", "verify"]
    run_kwargs = events[0][1]
    assert isinstance(run_kwargs, dict)
    assert tuple(run_kwargs["inputs"]) == ("EmotionTalk", "MELD")
    assert all(
        isinstance(value, joint_module.ModelSelectionReferenceFreezeInput)
        for value in run_kwargs["inputs"].values()
    )
    assert run_kwargs["inputs"]["EmotionTalk"].artifact_path == (
        args.emotiontalk_model_selection_artifact
    )
    assert run_kwargs["inputs"]["MELD"].expected_receipt_sha256 == (
        args.meld_model_selection_receipt_sha256
    )
    assert run_kwargs["source_snapshot"] is source_snapshot
    assert events[1][1] == (
        (
            completed.private_artifact_path,
            completed.private_receipt_path,
            completed.private_receipt_sha256,
        ),
        {"source_snapshot": source_snapshot},
    )

    summary = json.loads(capsys.readouterr().out)
    rendered = json.dumps(summary, sort_keys=True)
    assert summary["operation"] == "run"
    assert summary["dataset_roster"] == ["EmotionTalk", "MELD"]
    assert summary["joint_model_selection_freeze_passed"] is True
    assert summary["aggregate_only"] is True
    assert str(tmp_path) not in rendered
    for private_digest in ("8" * 64, "9" * 64, "b" * 64, "c" * 64, "f" * 64):
        assert private_digest not in rendered


def test_joint_verify_calls_hash_bound_api_and_outputs_only_aggregate_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    source_snapshot = _fake_snapshot(module, tmp_path)
    monkeypatch.setattr(
        module, "_verify_source_snapshot", lambda _args: source_snapshot
    )
    attestation = _fake_joint_attestation(tmp_path)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        joint_module,
        "verify_joint_model_selection_freeze_receipt",
        lambda *args, **kwargs: calls.append((args, kwargs)) or attestation,
    )
    artifact = tmp_path / "joint-model-selection-freeze.json"
    receipt = tmp_path / "joint-model-selection-freeze-receipt.json"
    args = SimpleNamespace(
        joint_private_artifact=artifact,
        joint_private_receipt=receipt,
        joint_private_receipt_sha256=SHA,
    )
    module._run_verify_joint_model_selection_freeze(args)
    assert calls == [
        (
            (artifact, receipt, SHA),
            {"source_snapshot": source_snapshot},
        )
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["operation"] == "verify"
    assert str(artifact) not in json.dumps(summary)
    assert "upstream_artifact_sha256_by_dataset" not in summary


def test_new_entrypoint_errors_fail_closed_without_rendering_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module()
    secret = tmp_path / "private-secret"
    monkeypatch.setattr(
        snapshot_module,
        "create_production_source_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(
            snapshot_module.ProductionSourceSnapshotError(str(secret))
        ),
    )
    with pytest.raises(SystemExit, match="failed closed") as source_error:
        module._run_create_production_source_snapshot(
            SimpleNamespace(
                source_snapshot_worktree_root=secret,
                source_snapshot_output_manifest=secret / "manifest.json",
            )
        )
    assert str(secret) not in str(source_error.value)

    monkeypatch.setattr(
        module, "_verify_source_snapshot", lambda _args: _fake_snapshot(module, tmp_path)
    )
    monkeypatch.setattr(
        joint_module,
        "run_joint_model_selection_freeze",
        lambda **_kwargs: (_ for _ in ()).throw(
            joint_module.JointModelSelectionFreezeError(str(secret))
        ),
    )
    args = SimpleNamespace(
        emotiontalk_model_selection_artifact=secret / "a.json",
        emotiontalk_model_selection_receipt=secret / "r.json",
        emotiontalk_model_selection_receipt_sha256=SHA,
        meld_model_selection_artifact=secret / "m-a.json",
        meld_model_selection_receipt=secret / "m-r.json",
        meld_model_selection_receipt_sha256=SHA,
        joint_private_output_root=secret / "joint",
        joint_public_report=secret / "public.json",
    )
    with pytest.raises(SystemExit, match="failed closed") as joint_error:
        module._run_joint_model_selection_freeze(args)
    assert str(secret) not in str(joint_error.value)


def test_main_dispatches_all_four_final_entrypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module()
    calls: list[str] = []
    handlers = {
        "_run_create_production_source_snapshot": "create",
        "_run_verify_production_source_snapshot": "verify-source",
        "_run_joint_model_selection_freeze": "run-joint",
        "_run_verify_joint_model_selection_freeze": "verify-joint",
    }
    for handler, label in handlers.items():
        monkeypatch.setattr(
            module,
            handler,
            lambda _args, observed=label: calls.append(observed),
        )

    snapshot_args = [
        "--source-snapshot-manifest",
        str(tmp_path / "snapshot.json"),
        "--source-snapshot-manifest-sha256",
        SHA,
        "--source-snapshot-worktree-root",
        str(tmp_path / "detached"),
    ]
    invocations = [
        [
            "create-production-source-snapshot",
            "--source-snapshot-worktree-root",
            str(tmp_path / "detached"),
            "--source-snapshot-output-manifest",
            str(tmp_path / "snapshot.json"),
        ],
        ["verify-production-source-snapshot", *snapshot_args],
        [
            "run-joint-model-selection-freeze",
            *snapshot_args,
            "--emotiontalk-model-selection-artifact",
            str(tmp_path / "e-a.json"),
            "--emotiontalk-model-selection-receipt",
            str(tmp_path / "e-r.json"),
            "--emotiontalk-model-selection-receipt-sha256",
            SHA,
            "--meld-model-selection-artifact",
            str(tmp_path / "m-a.json"),
            "--meld-model-selection-receipt",
            str(tmp_path / "m-r.json"),
            "--meld-model-selection-receipt-sha256",
            SHA,
            "--joint-private-output-root",
            str(tmp_path / "joint"),
            "--joint-public-report",
            str(tmp_path / "joint-public.json"),
        ],
        [
            "verify-joint-model-selection-freeze",
            *snapshot_args,
            "--joint-private-artifact",
            str(tmp_path / "joint-model-selection-freeze.json"),
            "--joint-private-receipt",
            str(tmp_path / "joint-model-selection-freeze-receipt.json"),
            "--joint-private-receipt-sha256",
            SHA,
        ],
    ]
    for invocation in invocations:
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_causal_backbone_evidence.py", *invocation],
        )
        module.main()
    assert calls == ["create", "verify-source", "run-joint", "verify-joint"]
