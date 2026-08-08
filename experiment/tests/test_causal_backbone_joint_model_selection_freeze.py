from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import hva_affect.causal_backbone_joint_model_selection_freeze as joint
from hva_affect.causal_backbone_joint_model_selection_freeze import (
    PRIVATE_ARTIFACT_NAME,
    PRIVATE_ARTIFACT_SCHEMA,
    PRIVATE_RECEIPT_NAME,
    PUBLIC_REPORT_SCHEMA,
    JointModelSelectionFreezeError,
    ModelSelectionReferenceFreezeInput,
    run_joint_model_selection_freeze,
    validate_joint_model_selection_public_report,
    verify_joint_model_selection_freeze_receipt,
)
from hva_affect.causal_backbone_model_selection_evaluator import (
    CONFIRMATORY_ANALYSIS_SHA256,
    VerifiedModelSelectionAggregateAttestation,
)
from hva_affect.production_source_snapshot_v1 import (
    ProductionSourceSnapshotAttestation,
)


_MISSING = object()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_snapshot(
    tmp_path: Path, *, suffix: str = ""
) -> ProductionSourceSnapshotAttestation:
    code_sha256 = {
        "experiment/scripts/run_causal_backbone_evidence.py": _sha(
            f"cli-source{suffix}"
        ),
        "experiment/src/hva_affect/causal_backbone_joint_model_selection_freeze.py": _sha(
            f"joint-source{suffix}"
        ),
    }
    code_paths = {
        name: tmp_path / "detached-worktree" / Path(name)
        for name in code_sha256
    }
    return ProductionSourceSnapshotAttestation(
        manifest_path=tmp_path / f"source-snapshot{suffix}.json",
        manifest_sha256=_sha(f"manifest{suffix}"),
        worktree_root=tmp_path / "detached-worktree",
        commit_sha=hashlib.sha1(f"commit{suffix}".encode("utf-8")).hexdigest(),
        tree_sha=hashlib.sha1(f"tree{suffix}".encode("utf-8")).hexdigest(),
        code_sha256=code_sha256,
        code_paths=code_paths,
    )


def _bundle(
    tmp_path: Path,
    *,
    emotion_power: float = 0.91,
    meld_power: float = 0.92,
    emotion_gate: object = True,
    meld_gate: object = True,
) -> tuple[
    dict[str, ModelSelectionReferenceFreezeInput],
    dict[str, object],
]:
    inputs: dict[str, ModelSelectionReferenceFreezeInput] = {}
    attestations: dict[str, object] = {}
    settings = {
        "EmotionTalk": (emotion_power, emotion_gate, "all_history"),
        "MELD": (meld_power, meld_gate, "coverage_matched_recency"),
    }
    for dataset, (power, gate, reference) in settings.items():
        upstream_root = tmp_path / "does-not-exist" / dataset
        artifact = upstream_root / "model-selection-reference-freeze.json"
        receipt = upstream_root / "model-selection-reference-freeze-receipt.json"
        receipt_sha = _sha(f"{dataset}-receipt")
        inputs[dataset] = ModelSelectionReferenceFreezeInput(
            artifact_path=artifact,
            receipt_path=receipt,
            expected_receipt_sha256=receipt_sha,
        )
        values = {
            "dataset": dataset,
            "artifact_path": artifact.resolve(),
            "artifact_sha256": _sha(f"{dataset}-artifact"),
            "receipt_path": receipt.resolve(),
            "receipt_sha256": receipt_sha,
            "public_report_sha256": _sha(f"{dataset}-public"),
            "analysis_config_sha256": CONFIRMATORY_ANALYSIS_SHA256,
            "cross_variant_alignment_sha256": _sha(f"{dataset}-alignment"),
            "frozen_reference": reference,
            "prospective_power": power,
            "power_gate_passed": power >= 0.8,
        }
        if gate is _MISSING:
            attestations[dataset] = SimpleNamespace(**values)
        else:
            values["model_selection_gate_passed"] = gate
            attestations[dataset] = VerifiedModelSelectionAggregateAttestation(
                **values
            )
    return inputs, attestations


def _install_verifier(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, ModelSelectionReferenceFreezeInput],
    attestations: dict[str, object],
    *,
    barrier: threading.Barrier | None = None,
) -> list[tuple[Path, Path, str]]:
    lookup = {
        (
            Path(source.artifact_path).resolve(),
            Path(source.receipt_path).resolve(),
            source.expected_receipt_sha256,
        ): attestations[dataset]
        for dataset, source in inputs.items()
    }
    calls: list[tuple[Path, Path, str]] = []
    lock = threading.Lock()

    def fake_verifier(
        artifact_path: str | Path,
        receipt_path: str | Path,
        expected_receipt_sha256: str,
    ) -> object:
        key = (
            Path(artifact_path).resolve(),
            Path(receipt_path).resolve(),
            expected_receipt_sha256,
        )
        with lock:
            calls.append(key)
        result = lookup[key]
        if barrier is not None and getattr(result, "dataset") == "EmotionTalk":
            barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(
        joint, "verify_model_selection_reference_freeze_receipt", fake_verifier
    )
    return calls


def _run(
    tmp_path: Path,
    inputs: dict[str, ModelSelectionReferenceFreezeInput],
    *,
    suffix: str = "",
    source_snapshot: ProductionSourceSnapshotAttestation | None = None,
) -> joint.CompletedJointModelSelectionFreeze:
    return run_joint_model_selection_freeze(
        inputs=inputs,
        source_snapshot=(
            source_snapshot if source_snapshot is not None else _source_snapshot(tmp_path)
        ),
        private_output_root=tmp_path / f"joint-private{suffix}",
        public_report_path=tmp_path / f"joint-public{suffix}.json",
    )


def test_exact_two_dataset_success_authorizes_only_separate_calibration_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    calls = _install_verifier(monkeypatch, inputs, attestations)

    completed = _run(tmp_path, inputs)

    assert completed.calibration_stage_workflow_authorized is True
    assert completed.failure_reasons == ()
    assert len(calls) == 2
    assert {value[0] for value in calls} == {
        Path(source.artifact_path).resolve() for source in inputs.values()
    }
    report_raw = completed.public_report_path.read_bytes()
    report = json.loads(report_raw.decode("utf-8"))
    assert report_raw == joint._canonical_json_bytes(report)
    assert report["schema_version"] == PUBLIC_REPORT_SCHEMA
    validate_joint_model_selection_public_report(report)
    authorization = report["stage_authorization"]
    assert authorization == {
        "calibration_outcome_access_authorized_by_this_layer": False,
        "confirmatory_method_success_authorized": False,
        "external_test_unseal_authorized": False,
        "failure_reasons": [],
        "internal_holdout_unseal_authorized": False,
        "reason": (
            "joint_predicate_passed_authorize_separate_calibration_workflow_only"
        ),
        "separate_calibration_stage_workflow_authorized": True,
    }
    public_text = report_raw.decode("utf-8")
    assert str(tmp_path) not in public_text
    assert not any(
        len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in _all_strings(report)
    )
    assert all(
        forbidden not in _all_keys(report)
        for forbidden in (
            "artifact_sha256",
            "receipt_sha256",
            "public_report_sha256",
            "seed_results",
            "row_ids",
            "cluster_codes",
            "probabilities",
        )
    )

    private = json.loads(completed.private_artifact_path.read_text(encoding="utf-8"))
    assert private["schema_version"] == PRIVATE_ARTIFACT_SCHEMA
    assert private["analysis_contract"]["analysis_config_sha256"] == (
        CONFIRMATORY_ANALYSIS_SHA256
    )
    for dataset in joint.REQUIRED_DATASETS:
        assert private["datasets"][dataset]["frozen_reference"] == (
            attestations[dataset].frozen_reference
        )
        assert private["datasets"][dataset]["prospective_power"] == (
            attestations[dataset].prospective_power
        )
        assert private["datasets"][dataset]["upstream_hashes"] == {
            "artifact_sha256": attestations[dataset].artifact_sha256,
            "cross_variant_alignment_sha256": (
                attestations[dataset].cross_variant_alignment_sha256
            ),
            "public_report_sha256": attestations[dataset].public_report_sha256,
            "receipt_sha256": attestations[dataset].receipt_sha256,
        }

    verified = verify_joint_model_selection_freeze_receipt(
        completed.private_artifact_path,
        completed.private_receipt_path,
        completed.private_receipt_sha256,
        source_snapshot=_source_snapshot(tmp_path),
    )
    assert verified.dataset_roster == ("EmotionTalk", "MELD")
    assert verified.calibration_stage_workflow_authorized is True
    assert verified.source_snapshot_manifest_sha256 == _source_snapshot(
        tmp_path
    ).manifest_sha256
    assert dict(verified.frozen_reference_by_dataset) == {
        "EmotionTalk": "all_history",
        "MELD": "coverage_matched_recency",
    }
    with pytest.raises(TypeError):
        verified.frozen_reference_by_dataset["MELD"] = "current_only"  # type: ignore[index]


def test_source_snapshot_is_private_and_required_to_verify_the_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    source_a = _source_snapshot(tmp_path, suffix="-source-a")
    source_b = _source_snapshot(tmp_path, suffix="-source-b")
    completed = _run(tmp_path, inputs, source_snapshot=source_a)

    artifact = json.loads(
        completed.private_artifact_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(completed.private_receipt_path.read_text(encoding="utf-8"))
    assert artifact["source_snapshot_lineage"] == receipt["lineage"][
        "source_snapshot_lineage"
    ]
    assert artifact["source_snapshot_lineage"] == {
        "code_bundle_sha256": joint._canonical_value_sha256(
            dict(sorted(source_a.code_sha256.items()))
        ),
        "git_commit": source_a.commit_sha,
        "git_tree": source_a.tree_sha,
        "manifest_sha256": source_a.manifest_sha256,
        "schema_version": "production_source_snapshot_v1",
    }
    public_report = json.loads(completed.public_report_path.read_text(encoding="utf-8"))
    assert not any("source_snapshot" in key for key in _all_keys(public_report))

    verified = verify_joint_model_selection_freeze_receipt(
        completed.private_artifact_path,
        completed.private_receipt_path,
        completed.private_receipt_sha256,
        source_snapshot=source_a,
    )
    assert verified.source_snapshot_git_commit == source_a.commit_sha
    assert verified.source_snapshot_git_tree == source_a.tree_sha
    assert verified.source_snapshot_code_bundle_sha256 == artifact[
        "source_snapshot_lineage"
    ]["code_bundle_sha256"]
    with pytest.raises(JointModelSelectionFreezeError, match="expected snapshot"):
        verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            completed.private_receipt_sha256,
            source_snapshot=source_b,
        )


def test_source_snapshot_attestation_subclass_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ForgedSourceSnapshot(ProductionSourceSnapshotAttestation):
        pass

    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    forged = ForgedSourceSnapshot(**vars(_source_snapshot(tmp_path)))
    with pytest.raises(JointModelSelectionFreezeError, match="exact typed"):
        _run(tmp_path, inputs, source_snapshot=forged)


def _all_strings(value: object) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            values.append(str(key))
            values.extend(_all_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_all_strings(nested))
    elif isinstance(value, str):
        values.append(value)
    return values


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys


def test_missing_upstream_performance_predicate_fails_closed_despite_power(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(
        tmp_path, emotion_gate=_MISSING, meld_gate=_MISSING
    )
    _install_verifier(monkeypatch, inputs, attestations)

    with pytest.raises(JointModelSelectionFreezeError, match="exact typed"):
        _run(tmp_path, inputs)

    assert not (tmp_path / "joint-private").exists()
    assert not (tmp_path / "joint-public.json").exists()


@pytest.mark.parametrize(
    ("emotion_power", "meld_power", "emotion_gate", "meld_gate", "reason"),
    [
        (0.79, 0.93, True, True, "EmotionTalk:prospective_power_below_0.80"),
        (0.91, 0.79, True, True, "MELD:prospective_power_below_0.80"),
        (0.91, 0.93, False, True, "EmotionTalk:upstream_model_selection_gate_failed"),
        (0.91, 0.93, True, False, "MELD:upstream_model_selection_gate_failed"),
    ],
)
def test_each_dataset_power_and_verified_performance_gate_is_conjunctive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    emotion_power: float,
    meld_power: float,
    emotion_gate: bool,
    meld_gate: bool,
    reason: str,
) -> None:
    inputs, attestations = _bundle(
        tmp_path,
        emotion_power=emotion_power,
        meld_power=meld_power,
        emotion_gate=emotion_gate,
        meld_gate=meld_gate,
    )
    _install_verifier(monkeypatch, inputs, attestations)

    completed = _run(tmp_path, inputs)

    assert completed.calibration_stage_workflow_authorized is False
    assert reason in completed.failure_reasons
    report = json.loads(completed.public_report_path.read_text(encoding="utf-8"))
    authorization = report["stage_authorization"]
    assert authorization["separate_calibration_stage_workflow_authorized"] is False
    assert authorization["internal_holdout_unseal_authorized"] is False
    assert authorization["external_test_unseal_authorized"] is False
    assert authorization["confirmatory_method_success_authorized"] is False


def test_input_contract_has_no_caller_performance_boolean() -> None:
    with pytest.raises(TypeError):
        ModelSelectionReferenceFreezeInput(
            artifact_path="a",
            receipt_path="b",
            expected_receipt_sha256="0" * 64,
            model_selection_gate_passed=True,  # type: ignore[call-arg]
        )


def test_roster_must_be_exact_before_any_upstream_verifier_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, _ = _bundle(tmp_path)
    monkeypatch.setattr(
        joint,
        "verify_model_selection_reference_freeze_receipt",
        lambda **_kwargs: pytest.fail("verifier must not run for a changed roster"),
    )

    with pytest.raises(JointModelSelectionFreezeError, match="exactly"):
        _run(tmp_path, {"EmotionTalk": inputs["EmotionTalk"]})
    with pytest.raises(JointModelSelectionFreezeError, match="exactly"):
        _run(
            tmp_path,
            {
                **inputs,
                "Other": inputs["EmotionTalk"],
            },
        )


def test_upstream_dataset_identity_and_confirmatory_sha_are_rechecked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    attestations["EmotionTalk"] = replace(
        attestations["EmotionTalk"], dataset="MELD"
    )
    _install_verifier(monkeypatch, inputs, attestations)
    with pytest.raises(JointModelSelectionFreezeError, match="wrong dataset"):
        _run(tmp_path, inputs)

    inputs, attestations = _bundle(tmp_path)
    attestations["MELD"] = replace(
        attestations["MELD"],
        analysis_config_sha256=_sha("different-contract"),
    )
    _install_verifier(monkeypatch, inputs, attestations)
    with pytest.raises(JointModelSelectionFreezeError, match="exact frozen"):
        _run(tmp_path, inputs)


@pytest.mark.parametrize(
    "field",
    ["artifact_sha256", "receipt_sha256", "public_report_sha256"],
)
def test_dataset_handoffs_must_have_distinct_hash_bound_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    inputs, attestations = _bundle(tmp_path)
    duplicate = getattr(attestations["EmotionTalk"], field)
    attestations["MELD"] = replace(attestations["MELD"], **{field: duplicate})
    if field == "receipt_sha256":
        source = inputs["MELD"]
        inputs["MELD"] = ModelSelectionReferenceFreezeInput(
            source.artifact_path, source.receipt_path, duplicate
        )
    _install_verifier(monkeypatch, inputs, attestations)

    with pytest.raises(JointModelSelectionFreezeError, match="independently hash-bound"):
        _run(tmp_path, inputs)


def test_all_four_upstream_artifact_and_receipt_paths_must_be_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    meld = inputs["MELD"]
    collision = Path(inputs["EmotionTalk"].receipt_path)
    inputs["MELD"] = ModelSelectionReferenceFreezeInput(
        collision, meld.receipt_path, meld.expected_receipt_sha256
    )
    attestations["MELD"] = replace(
        attestations["MELD"], artifact_path=collision.resolve()
    )
    _install_verifier(monkeypatch, inputs, attestations)

    with pytest.raises(JointModelSelectionFreezeError, match="four distinct"):
        _run(tmp_path, inputs)


def test_non_boolean_future_gate_is_rejected_not_coerced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path, meld_gate=1)
    _install_verifier(monkeypatch, inputs, attestations)
    with pytest.raises(JointModelSelectionFreezeError, match="must be a boolean"):
        _run(tmp_path, inputs)


def test_upstream_verifier_result_must_be_the_exact_attestation_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ForgedAttestationSubclass(VerifiedModelSelectionAggregateAttestation):
        pass

    inputs, attestations = _bundle(tmp_path)
    base = attestations["MELD"]
    assert isinstance(base, VerifiedModelSelectionAggregateAttestation)
    attestations["MELD"] = ForgedAttestationSubclass(**vars(base))
    _install_verifier(monkeypatch, inputs, attestations)

    with pytest.raises(JointModelSelectionFreezeError, match="exact typed"):
        _run(tmp_path, inputs)


def test_public_privacy_validator_rejects_local_path_and_digest_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path, emotion_gate=False, meld_gate=False)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    original = json.loads(completed.public_report_path.read_text(encoding="utf-8"))

    path_leak = copy.deepcopy(original)
    path_leak["stage_authorization"]["failure_reasons"] = ["C:\\private\\labels.npz"]
    with pytest.raises(JointModelSelectionFreezeError, match="local path"):
        joint._visit_aggregate(path_leak, public=True)

    hash_leak = copy.deepcopy(original)
    hash_leak["stage_authorization"]["failure_reasons"] = ["a" * 64]
    with pytest.raises(JointModelSelectionFreezeError, match="outcome hash"):
        joint._visit_aggregate(hash_leak, public=True)


def test_joint_receipt_rejects_noncanonical_or_tampered_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    receipt_payload = json.loads(
        completed.private_receipt_path.read_text(encoding="utf-8")
    )
    completed.private_receipt_path.write_text(
        json.dumps(receipt_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    changed_receipt_sha = hashlib.sha256(
        completed.private_receipt_path.read_bytes()
    ).hexdigest()
    with pytest.raises(JointModelSelectionFreezeError, match="not canonical"):
        verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            changed_receipt_sha,
            source_snapshot=_source_snapshot(tmp_path),
        )


def test_joint_receipt_rejects_canonical_private_schema_injection_even_rehashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    artifact = json.loads(
        completed.private_artifact_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(completed.private_receipt_path.read_text(encoding="utf-8"))
    artifact["labels"] = ["forbidden"]
    artifact_bytes = joint._canonical_json_bytes(artifact)
    completed.private_artifact_path.write_bytes(artifact_bytes)
    receipt["lineage"]["private_artifact_sha256"] = hashlib.sha256(
        artifact_bytes
    ).hexdigest()
    receipt_bytes = joint._canonical_json_bytes(receipt)
    completed.private_receipt_path.write_bytes(receipt_bytes)

    with pytest.raises(JointModelSelectionFreezeError, match="schema changed"):
        verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            hashlib.sha256(receipt_bytes).hexdigest(),
            source_snapshot=_source_snapshot(tmp_path),
        )


@pytest.mark.parametrize(
    ("artifact_key", "receipt_map"),
    [
        ("artifact_sha256", "upstream_artifact_sha256_by_dataset"),
        ("receipt_sha256", "upstream_receipt_sha256_by_dataset"),
        ("public_report_sha256", "upstream_public_report_sha256_by_dataset"),
    ],
)
def test_verifier_rejects_rehashed_duplicate_dataset_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_key: str,
    receipt_map: str,
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    artifact = json.loads(
        completed.private_artifact_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(completed.private_receipt_path.read_text(encoding="utf-8"))
    duplicate = artifact["datasets"]["EmotionTalk"]["upstream_hashes"][
        artifact_key
    ]
    artifact["datasets"]["MELD"]["upstream_hashes"][artifact_key] = duplicate
    receipt["lineage"][receipt_map]["MELD"] = duplicate
    artifact_bytes = joint._canonical_json_bytes(artifact)
    completed.private_artifact_path.write_bytes(artifact_bytes)
    receipt["lineage"]["private_artifact_sha256"] = hashlib.sha256(
        artifact_bytes
    ).hexdigest()
    receipt_bytes = joint._canonical_json_bytes(receipt)
    completed.private_receipt_path.write_bytes(receipt_bytes)

    with pytest.raises(JointModelSelectionFreezeError, match="independently hash-bound"):
        verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            hashlib.sha256(receipt_bytes).hexdigest(),
            source_snapshot=_source_snapshot(tmp_path),
        )


def test_verifier_rejects_numeric_sha_even_when_rehashed_and_cross_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    artifact = json.loads(
        completed.private_artifact_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(completed.private_receipt_path.read_text(encoding="utf-8"))
    numeric_digest = int("1" * 64)
    artifact["datasets"]["MELD"]["upstream_hashes"][
        "artifact_sha256"
    ] = numeric_digest
    receipt["lineage"]["upstream_artifact_sha256_by_dataset"][
        "MELD"
    ] = numeric_digest
    artifact_bytes = joint._canonical_json_bytes(artifact)
    completed.private_artifact_path.write_bytes(artifact_bytes)
    receipt["lineage"]["private_artifact_sha256"] = hashlib.sha256(
        artifact_bytes
    ).hexdigest()
    receipt_bytes = joint._canonical_json_bytes(receipt)
    completed.private_receipt_path.write_bytes(receipt_bytes)

    with pytest.raises(JointModelSelectionFreezeError, match="lowercase SHA-256"):
        verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            hashlib.sha256(receipt_bytes).hexdigest(),
            source_snapshot=_source_snapshot(tmp_path),
        )


def test_write_once_outputs_do_not_clobber_existing_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    before = {
        path: path.read_bytes()
        for path in (
            completed.private_artifact_path,
            completed.private_receipt_path,
            completed.public_report_path,
        )
    }

    with pytest.raises(FileExistsError):
        _run(tmp_path, inputs)

    assert all(path.read_bytes() == encoded for path, encoded in before.items())


def test_receipt_is_last_commit_marker_when_public_publication_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    original_atomic_write_once = joint._atomic_write_once
    private_root = tmp_path / "joint-private"
    artifact_path = private_root / PRIVATE_ARTIFACT_NAME
    receipt_path = private_root / PRIVATE_RECEIPT_NAME
    public_path = tmp_path / "joint-public.json"
    publication_order: list[Path] = []

    def fail_after_public_write(path: Path, payload: bytes) -> str:
        publication_order.append(path)
        digest = original_atomic_write_once(path, payload)
        if path == public_path:
            raise JointModelSelectionFreezeError("injected public publication fault")
        return digest

    monkeypatch.setattr(joint, "_atomic_write_once", fail_after_public_write)
    with pytest.raises(JointModelSelectionFreezeError, match="injected"):
        _run(tmp_path, inputs)

    assert publication_order == [artifact_path, public_path]
    assert artifact_path.is_file()
    assert public_path.is_file()
    assert not receipt_path.exists()
    with pytest.raises(JointModelSelectionFreezeError, match="missing"):
        verify_joint_model_selection_freeze_receipt(
            artifact_path,
            receipt_path,
            "0" * 64,
            source_snapshot=_source_snapshot(tmp_path),
        )


def test_concurrent_writers_have_exactly_one_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    barrier = threading.Barrier(2)
    _install_verifier(monkeypatch, inputs, attestations, barrier=barrier)

    def publish() -> object:
        try:
            return _run(tmp_path, inputs)
        except Exception as error:  # captured for exact winner/loser assertion
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: publish(), range(2)))

    winners = [
        value
        for value in outcomes
        if isinstance(value, joint.CompletedJointModelSelectionFreeze)
    ]
    losers = [value for value in outcomes if isinstance(value, Exception)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(losers[0], FileExistsError)
    winner = winners[0]
    verify_joint_model_selection_freeze_receipt(
        winner.private_artifact_path,
        winner.private_receipt_path,
        winner.private_receipt_sha256,
        source_snapshot=_source_snapshot(tmp_path),
    )


def test_atomic_hard_link_publication_rehashes_destination_for_toctou(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "toctou.json"
    monkeypatch.setattr(joint, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(JointModelSelectionFreezeError, match="changed during publication"):
        joint._atomic_write_once(destination, b"{}\n")


def test_verifier_binds_decoded_artifact_bytes_before_path_rehash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    completed = _run(tmp_path, inputs)
    receipt = json.loads(completed.private_receipt_path.read_text(encoding="utf-8"))
    replacement = b"{}\n"
    receipt["lineage"]["private_artifact_sha256"] = hashlib.sha256(
        replacement
    ).hexdigest()
    receipt_bytes = joint._canonical_json_bytes(receipt)
    completed.private_receipt_path.write_bytes(receipt_bytes)
    original_decode = joint._decode_canonical_json

    def swap_after_decode(path: Path) -> tuple[object, bytes]:
        payload, raw = original_decode(path)
        if path == completed.private_artifact_path:
            path.write_bytes(replacement)
        return payload, raw

    monkeypatch.setattr(joint, "_decode_canonical_json", swap_after_decode)
    with pytest.raises(JointModelSelectionFreezeError, match="diverged"):
        verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            hashlib.sha256(receipt_bytes).hexdigest(),
            source_snapshot=_source_snapshot(tmp_path),
        )


def test_verifier_binds_decoded_receipt_to_expected_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs_a, attestations_a = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs_a, attestations_a)
    completed_a = _run(tmp_path, inputs_a, suffix="-a")
    trusted_receipt_bytes = completed_a.private_receipt_path.read_bytes()

    inputs_b, attestations_b = _bundle(tmp_path, emotion_gate=False)
    _install_verifier(monkeypatch, inputs_b, attestations_b)
    completed_b = _run(tmp_path, inputs_b, suffix="-b")
    alternate_receipt_bytes = completed_b.private_receipt_path.read_bytes()
    alternate_artifact_bytes = completed_b.private_artifact_path.read_bytes()
    original_decode = joint._decode_canonical_json

    def swap_during_decode(path: Path) -> tuple[object, bytes]:
        if path == completed_a.private_receipt_path:
            path.write_bytes(alternate_receipt_bytes)
            try:
                return original_decode(path)
            finally:
                path.write_bytes(trusted_receipt_bytes)
        if path == completed_a.private_artifact_path:
            path.write_bytes(alternate_artifact_bytes)
        return original_decode(path)

    monkeypatch.setattr(joint, "_decode_canonical_json", swap_during_decode)
    with pytest.raises(JointModelSelectionFreezeError, match="changed while decoding"):
        verify_joint_model_selection_freeze_receipt(
            completed_a.private_artifact_path,
            completed_a.private_receipt_path,
            hashlib.sha256(trusted_receipt_bytes).hexdigest(),
            source_snapshot=_source_snapshot(tmp_path),
        )


def test_private_root_must_be_new_absolute_and_external_to_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    with pytest.raises(JointModelSelectionFreezeError, match="absolute"):
        run_joint_model_selection_freeze(
            inputs=inputs,
            source_snapshot=_source_snapshot(tmp_path),
            private_output_root="relative-private",
            public_report_path=tmp_path / "public.json",
        )

    monkeypatch.setattr(
        joint.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(JointModelSelectionFreezeError, match="every Git repository"):
        _run(tmp_path, inputs)


def test_private_root_inside_source_repository_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, attestations = _bundle(tmp_path)
    _install_verifier(monkeypatch, inputs, attestations)
    repository_root = Path(joint.__file__).resolve().parents[3]
    with pytest.raises(JointModelSelectionFreezeError, match="repository-external"):
        run_joint_model_selection_freeze(
            inputs=inputs,
            source_snapshot=_source_snapshot(tmp_path),
            private_output_root=repository_root / "forbidden-joint-private",
            public_report_path=tmp_path / "public.json",
        )
