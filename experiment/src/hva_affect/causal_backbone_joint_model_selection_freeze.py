"""Hash-bound two-dataset model-selection reference freeze.

This layer consumes *only* the aggregate attestations returned by
``verify_model_selection_reference_freeze_receipt``.  It never receives a
label, probability array, row identifier, or outcome-sidecar capability.  Its
sole positive authority is narrow: after both required datasets have a
verifier-attested model-selection gate and prospective power of at least 0.80,
it may authorize a separate calibration-stage workflow.  It never authorizes
confirmatory method success, holdout access, or test access.

The upstream verifier-derived ``model_selection_gate_passed`` field is a
required strict boolean.  A missing field or non-typed verifier result is
rejected; power alone is not performance evidence and cannot unseal
calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

from .causal_backbone_model_selection_evaluator import (
    CONFIRMATORY_ANALYSIS_SHA256,
    MINIMUM_PROSPECTIVE_POWER,
    REFERENCE_CANDIDATES,
    VerifiedModelSelectionAggregateAttestation,
    verify_model_selection_reference_freeze_receipt,
)
from .production_source_snapshot_v1 import (
    SNAPSHOT_SCHEMA_VERSION,
    ProductionSourceSnapshotAttestation,
)


REQUIRED_DATASETS = ("EmotionTalk", "MELD")
PRIVATE_ARTIFACT_SCHEMA = "carma_causal_backbone_joint_model_selection_freeze_private_v1"
PRIVATE_RECEIPT_SCHEMA = "carma_causal_backbone_joint_model_selection_freeze_receipt_v1"
PUBLIC_REPORT_SCHEMA = "carma_causal_backbone_joint_model_selection_freeze_public_v1"
PRIVATE_ARTIFACT_NAME = "joint-model-selection-freeze.json"
PRIVATE_RECEIPT_NAME = "joint-model-selection-freeze-receipt.json"

_STATUS = "complete_non_confirmatory_two_dataset_joint_model_selection_freeze"
_CLAIM_BOUNDARY = (
    "Aggregate-only two-dataset model-selection handoff; never confirmatory "
    "method-success evidence and never holdout or test authority."
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LOCAL_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_AGGREGATE_KEYS = frozenset(
    {
        "labels",
        "label_path",
        "outcomes",
        "outcome_path",
        "predictions",
        "probabilities",
        "protocol_row_ids",
        "row_ids",
        "row_vector",
        "cluster_codes",
        "cluster_vector",
        "group_ids",
        "speaker_ids",
        "contexts",
        "histories",
        "seed_results",
        "seed_vector",
        "private_path",
        "artifact_path",
        "receipt_path",
    }
)


class JointModelSelectionFreezeError(ValueError):
    """Raised when the joint freeze or its aggregate privacy boundary changes."""


@dataclass(frozen=True)
class ModelSelectionReferenceFreezeInput:
    """Exactly the three values accepted by the upstream receipt verifier."""

    artifact_path: str | Path
    receipt_path: str | Path
    expected_receipt_sha256: str


@dataclass(frozen=True)
class CompletedJointModelSelectionFreeze:
    """Paths and hashes for a newly published aggregate joint freeze."""

    private_artifact_path: Path
    private_artifact_sha256: str
    private_receipt_path: Path
    private_receipt_sha256: str
    public_report_path: Path
    public_report_sha256: str
    calibration_stage_workflow_authorized: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedJointModelSelectionFreezeAttestation:
    """Aggregate-only handoff for a future, separate calibration workflow."""

    dataset_roster: tuple[str, ...]
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    public_report_sha256: str
    analysis_config_sha256: str
    source_snapshot_manifest_sha256: str
    source_snapshot_git_commit: str
    source_snapshot_git_tree: str
    source_snapshot_code_bundle_sha256: str
    frozen_reference_by_dataset: Mapping[str, str]
    prospective_power_by_dataset: Mapping[str, float]
    power_gate_passed_by_dataset: Mapping[str, bool]
    upstream_artifact_sha256_by_dataset: Mapping[str, str]
    upstream_receipt_sha256_by_dataset: Mapping[str, str]
    upstream_public_report_sha256_by_dataset: Mapping[str, str]
    cross_variant_alignment_sha256_by_dataset: Mapping[str, str]
    model_selection_gate_attested_by_dataset: Mapping[str, bool]
    model_selection_gate_passed_by_dataset: Mapping[str, bool]
    calibration_stage_workflow_authorized: bool
    failure_reasons: tuple[str, ...]


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise JointModelSelectionFreezeError(
            f"required aggregate handoff is missing or symbolic: {path.name}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise JointModelSelectionFreezeError(
            f"cannot hash aggregate handoff {path.name}: {error}"
        ) from error
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise JointModelSelectionFreezeError(
            f"{field} must be one lowercase SHA-256 digest"
        )
    return cast(str, value)


def _require_git_object_id(value: object, field: str) -> str:
    if type(value) is not str or _GIT_OBJECT_ID.fullmatch(value) is None:
        raise JointModelSelectionFreezeError(
            f"{field} must be one lowercase Git object ID"
        )
    return cast(str, value)


def _canonical_value_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _validate_source_snapshot_lineage(value: object) -> None:
    expected = {
        "schema_version",
        "manifest_sha256",
        "git_commit",
        "git_tree",
        "code_bundle_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise JointModelSelectionFreezeError("source snapshot lineage schema changed")
    if value["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise JointModelSelectionFreezeError("source snapshot schema changed")
    _require_sha256(value["manifest_sha256"], "source_snapshot.manifest_sha256")
    _require_git_object_id(value["git_commit"], "source_snapshot.git_commit")
    _require_git_object_id(value["git_tree"], "source_snapshot.git_tree")
    _require_sha256(
        value["code_bundle_sha256"], "source_snapshot.code_bundle_sha256"
    )


def _source_snapshot_lineage(
    attestation: ProductionSourceSnapshotAttestation,
) -> dict[str, object]:
    """Reduce one exact verified source snapshot to path-free private lineage."""

    if type(attestation) is not ProductionSourceSnapshotAttestation:
        raise JointModelSelectionFreezeError(
            "source snapshot verifier did not return the exact typed attestation"
        )
    hashes = attestation.code_sha256
    paths = attestation.code_paths
    if (
        not isinstance(hashes, Mapping)
        or not hashes
        or not isinstance(paths, Mapping)
        or set(hashes) != set(paths)
    ):
        raise JointModelSelectionFreezeError(
            "source snapshot code roster is missing or inconsistent"
        )
    normalized: dict[str, str] = {}
    for key, digest in hashes.items():
        if (
            type(key) is not str
            or not key
            or "\\" in key
            or key.startswith("/")
            or any(part in {"", ".", ".."} for part in key.split("/"))
        ):
            raise JointModelSelectionFreezeError(
                "source snapshot code key is not repository-relative POSIX"
            )
        normalized[key] = _require_sha256(
            digest, f"source_snapshot.code_sha256.{key}"
        )
    lineage: dict[str, object] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "manifest_sha256": _require_sha256(
            attestation.manifest_sha256, "source_snapshot.manifest_sha256"
        ),
        "git_commit": _require_git_object_id(
            attestation.commit_sha, "source_snapshot.git_commit"
        ),
        "git_tree": _require_git_object_id(
            attestation.tree_sha, "source_snapshot.git_tree"
        ),
        "code_bundle_sha256": _canonical_value_sha256(dict(sorted(normalized.items()))),
    }
    _validate_source_snapshot_lineage(lineage)
    return lineage


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise JointModelSelectionFreezeError(f"{field} must be a boolean")
    return cast(bool, value)


def _require_probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointModelSelectionFreezeError(f"{field} must be a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JointModelSelectionFreezeError(f"{field} must be in [0, 1]")
    return result


def _model_selection_gate_from_verified_attestation(
    attestation: VerifiedModelSelectionAggregateAttestation,
) -> bool:
    """Accept only the exact typed, verifier-derived performance predicate."""

    if type(attestation) is not VerifiedModelSelectionAggregateAttestation:
        raise JointModelSelectionFreezeError(
            "upstream verifier did not return the exact typed attestation"
        )
    value = attestation.model_selection_gate_passed
    if type(value) is not bool:
        raise JointModelSelectionFreezeError(
            "verified model_selection_gate_passed must be a boolean"
        )
    return cast(bool, value)


def _verify_upstream_handoffs(
    inputs: Mapping[str, ModelSelectionReferenceFreezeInput],
) -> dict[str, tuple[VerifiedModelSelectionAggregateAttestation, bool, bool]]:
    if not isinstance(inputs, Mapping) or set(inputs) != set(REQUIRED_DATASETS):
        raise JointModelSelectionFreezeError(
            "joint freeze requires exactly EmotionTalk and MELD"
        )

    verified: dict[
        str, tuple[VerifiedModelSelectionAggregateAttestation, bool, bool]
    ] = {}
    all_paths: list[Path] = []
    artifact_hashes: list[str] = []
    receipt_hashes: list[str] = []
    public_hashes: list[str] = []
    for dataset in REQUIRED_DATASETS:
        source = inputs[dataset]
        if not isinstance(source, ModelSelectionReferenceFreezeInput):
            raise JointModelSelectionFreezeError(
                f"{dataset} input must be ModelSelectionReferenceFreezeInput"
            )
        # This is intentionally the only operation allowed to inspect each
        # upstream artifact/receipt.  The joint layer never parses those files.
        attestation = verify_model_selection_reference_freeze_receipt(
            artifact_path=source.artifact_path,
            receipt_path=source.receipt_path,
            expected_receipt_sha256=source.expected_receipt_sha256,
        )
        if attestation.dataset != dataset:
            raise JointModelSelectionFreezeError(
                f"{dataset} input attested the wrong dataset"
            )
        artifact_path = Path(attestation.artifact_path).resolve()
        receipt_path = Path(attestation.receipt_path).resolve()
        if (
            artifact_path != Path(source.artifact_path).resolve()
            or receipt_path != Path(source.receipt_path).resolve()
        ):
            raise JointModelSelectionFreezeError(
                f"{dataset} verifier returned different aggregate paths"
            )
        receipt_sha = _require_sha256(
            attestation.receipt_sha256, f"{dataset}.receipt_sha256"
        )
        if receipt_sha != _require_sha256(
            source.expected_receipt_sha256,
            f"{dataset}.expected_receipt_sha256",
        ):
            raise JointModelSelectionFreezeError(
                f"{dataset} verifier returned a different receipt hash"
            )
        artifact_sha = _require_sha256(
            attestation.artifact_sha256, f"{dataset}.artifact_sha256"
        )
        public_sha = _require_sha256(
            attestation.public_report_sha256,
            f"{dataset}.public_report_sha256",
        )
        _require_sha256(
            attestation.cross_variant_alignment_sha256,
            f"{dataset}.cross_variant_alignment_sha256",
        )
        analysis_sha = _require_sha256(
            attestation.analysis_config_sha256,
            f"{dataset}.analysis_config_sha256",
        )
        if analysis_sha != CONFIRMATORY_ANALYSIS_SHA256:
            raise JointModelSelectionFreezeError(
                "datasets do not use the exact frozen confirmatory contract"
            )
        if attestation.frozen_reference not in REFERENCE_CANDIDATES:
            raise JointModelSelectionFreezeError(
                f"{dataset} frozen reference is inadmissible"
            )
        power = _require_probability(
            attestation.prospective_power, f"{dataset}.prospective_power"
        )
        power_gate = _require_bool(
            attestation.power_gate_passed, f"{dataset}.power_gate_passed"
        )
        if power_gate != (power >= MINIMUM_PROSPECTIVE_POWER):
            raise JointModelSelectionFreezeError(
                f"{dataset} prospective-power gate is inconsistent"
            )
        gate_passed = _model_selection_gate_from_verified_attestation(attestation)
        verified[dataset] = (attestation, True, gate_passed)
        all_paths.extend((artifact_path, receipt_path))
        artifact_hashes.append(artifact_sha)
        receipt_hashes.append(receipt_sha)
        public_hashes.append(public_sha)

    if len(set(all_paths)) != 2 * len(REQUIRED_DATASETS):
        raise JointModelSelectionFreezeError(
            "dataset artifacts and receipts must be four distinct files"
        )
    if (
        len(set(artifact_hashes)) != len(REQUIRED_DATASETS)
        or len(set(receipt_hashes)) != len(REQUIRED_DATASETS)
        or len(set(public_hashes)) != len(REQUIRED_DATASETS)
    ):
        raise JointModelSelectionFreezeError(
            "dataset artifacts and receipts must be independently hash-bound"
        )
    if {
        item[0].analysis_config_sha256 for item in verified.values()
    } != {CONFIRMATORY_ANALYSIS_SHA256}:
        raise JointModelSelectionFreezeError(
            "datasets do not share the exact frozen confirmatory contract"
        )
    return verified


def _failure_reasons(
    verified: Mapping[
        str, tuple[VerifiedModelSelectionAggregateAttestation, bool, bool]
    ],
) -> list[str]:
    reasons: list[str] = []
    for dataset in REQUIRED_DATASETS:
        attestation, gate_attested, gate_passed = verified[dataset]
        if not attestation.power_gate_passed:
            reasons.append(f"{dataset}:prospective_power_below_0.80")
        if not gate_attested:
            raise AssertionError("a published joint freeze requires typed gate evidence")
        if not gate_passed:
            reasons.append(f"{dataset}:upstream_model_selection_gate_failed")
    return reasons


def _dataset_payloads(
    verified: Mapping[
        str, tuple[VerifiedModelSelectionAggregateAttestation, bool, bool]
    ],
    *,
    include_upstream_hashes: bool,
) -> dict[str, object]:
    payloads: dict[str, object] = {}
    for dataset in REQUIRED_DATASETS:
        attestation, gate_attested, gate_passed = verified[dataset]
        payload: dict[str, object] = {
            "dataset": dataset,
            "frozen_reference": attestation.frozen_reference,
            "prospective_power": float(attestation.prospective_power),
            "minimum_prospective_power": MINIMUM_PROSPECTIVE_POWER,
            "power_gate_passed": bool(attestation.power_gate_passed),
            "model_selection_gate_attested": gate_attested,
            "model_selection_gate_passed": gate_passed,
        }
        if include_upstream_hashes:
            payload["upstream_hashes"] = {
                "artifact_sha256": attestation.artifact_sha256,
                "receipt_sha256": attestation.receipt_sha256,
                "public_report_sha256": attestation.public_report_sha256,
                "cross_variant_alignment_sha256": (
                    attestation.cross_variant_alignment_sha256
                ),
            }
        payloads[dataset] = payload
    return payloads


def _joint_predicate(
    verified: Mapping[
        str, tuple[VerifiedModelSelectionAggregateAttestation, bool, bool]
    ],
) -> dict[str, object]:
    all_power = all(verified[name][0].power_gate_passed for name in REQUIRED_DATASETS)
    all_gates_attested = all(verified[name][1] for name in REQUIRED_DATASETS)
    all_gates_passed = all(
        verified[name][1] and verified[name][2] for name in REQUIRED_DATASETS
    )
    return {
        "exact_required_dataset_roster_passed": True,
        "same_exact_confirmatory_contract_passed": True,
        "all_dataset_prospective_power_gates_passed": all_power,
        "all_dataset_model_selection_gates_attested": all_gates_attested,
        "all_dataset_model_selection_gates_passed": all_gates_passed,
        "joint_model_selection_freeze_passed": all_power and all_gates_passed,
        "predicate_is_conjunctive": True,
    }


def _stage_authorization(
    predicate: Mapping[str, object], reasons: list[str]
) -> dict[str, object]:
    authorized = predicate["joint_model_selection_freeze_passed"] is True
    if authorized != (not reasons):
        raise AssertionError("joint failure reasons and authorization diverged")
    return {
        "separate_calibration_stage_workflow_authorized": authorized,
        "calibration_outcome_access_authorized_by_this_layer": False,
        "confirmatory_method_success_authorized": False,
        "internal_holdout_unseal_authorized": False,
        "external_test_unseal_authorized": False,
        "failure_reasons": list(reasons),
        "reason": (
            "joint_predicate_passed_authorize_separate_calibration_workflow_only"
            if authorized
            else "joint_predicate_failed_keep_all_later_roles_sealed"
        ),
    }


def _expected_failure_reasons_from_dataset_payloads(
    datasets: Mapping[str, object],
) -> list[str]:
    rows = cast(Mapping[str, Mapping[str, object]], datasets)
    reasons: list[str] = []
    for dataset in REQUIRED_DATASETS:
        row = rows[dataset]
        if row["power_gate_passed"] is not True:
            reasons.append(f"{dataset}:prospective_power_below_0.80")
        if row["model_selection_gate_attested"] is not True:
            raise JointModelSelectionFreezeError(
                f"{dataset} model-selection gate lacks typed verifier attestation"
            )
        if row["model_selection_gate_passed"] is not True:
            reasons.append(f"{dataset}:upstream_model_selection_gate_failed")
    return reasons


def _visit_aggregate(value: object, *, public: bool) -> None:
    def visit(child: object, trail: tuple[str, ...]) -> None:
        if isinstance(child, Path):
            raise JointModelSelectionFreezeError(
                f"aggregate contains a path object at {'.'.join(trail)}"
            )
        if isinstance(child, Mapping):
            for raw_key, nested in child.items():
                if not isinstance(raw_key, str):
                    raise JointModelSelectionFreezeError(
                        f"aggregate contains a non-string key at {'.'.join(trail)}"
                    )
                lowered = raw_key.lower()
                if lowered in _FORBIDDEN_AGGREGATE_KEYS:
                    raise JointModelSelectionFreezeError(
                        f"aggregate exposes forbidden field: {raw_key}"
                    )
                if public and (
                    lowered.endswith("_sha256")
                    or lowered
                    in {
                        "hash",
                        "hashes",
                        "outcome_hash",
                        "outcome_hashes",
                        "upstream_hashes",
                    }
                ):
                    raise JointModelSelectionFreezeError(
                        f"public report exposes an outcome hash field: {raw_key}"
                    )
                visit(nested, (*trail, raw_key))
            return
        if isinstance(child, (list, tuple)):
            for index, nested in enumerate(child):
                visit(nested, (*trail, str(index)))
            return
        if child is not None and not isinstance(child, (str, bool, int, float)):
            raise JointModelSelectionFreezeError(
                f"aggregate contains unsupported JSON at {'.'.join(trail)}"
            )
        if isinstance(child, float) and not math.isfinite(child):
            raise JointModelSelectionFreezeError(
                f"aggregate contains a non-finite value at {'.'.join(trail)}"
            )
        if isinstance(child, str):
            if (
                child.startswith(("/", "./", "../", "~/", "file://", "\\\\"))
                or "\\" in child
                or (public and "/" in child)
                or _LOCAL_PATH.match(child)
            ):
                raise JointModelSelectionFreezeError(
                    f"aggregate contains a local path at {'.'.join(trail)}"
                )
            if public and _SHA256.fullmatch(child.lower()) is not None:
                raise JointModelSelectionFreezeError(
                    f"public report exposes an outcome hash at {'.'.join(trail)}"
                )

    visit(value, ("root",))


def _validate_dataset_payloads(
    datasets: object, *, include_upstream_hashes: bool
) -> None:
    if not isinstance(datasets, Mapping) or set(datasets) != set(REQUIRED_DATASETS):
        raise JointModelSelectionFreezeError("joint dataset payload schema changed")
    expected = {
        "dataset",
        "frozen_reference",
        "prospective_power",
        "minimum_prospective_power",
        "power_gate_passed",
        "model_selection_gate_attested",
        "model_selection_gate_passed",
    }
    if include_upstream_hashes:
        expected.add("upstream_hashes")
    for dataset in REQUIRED_DATASETS:
        row = datasets[dataset]
        if not isinstance(row, Mapping) or set(row) != expected:
            raise JointModelSelectionFreezeError(
                f"{dataset} joint summary schema changed"
            )
        if row.get("dataset") != dataset:
            raise JointModelSelectionFreezeError(
                f"{dataset} joint summary identity changed"
            )
        if row.get("frozen_reference") not in REFERENCE_CANDIDATES:
            raise JointModelSelectionFreezeError(
                f"{dataset} joint frozen reference changed"
            )
        power = _require_probability(
            row.get("prospective_power"), f"{dataset}.prospective_power"
        )
        if row.get("minimum_prospective_power") != MINIMUM_PROSPECTIVE_POWER:
            raise JointModelSelectionFreezeError(
                f"{dataset} prospective-power threshold changed"
            )
        power_gate = _require_bool(
            row.get("power_gate_passed"), f"{dataset}.power_gate_passed"
        )
        gate_attested = _require_bool(
            row.get("model_selection_gate_attested"),
            f"{dataset}.model_selection_gate_attested",
        )
        gate_passed = _require_bool(
            row.get("model_selection_gate_passed"),
            f"{dataset}.model_selection_gate_passed",
        )
        if power_gate != (power >= MINIMUM_PROSPECTIVE_POWER):
            raise JointModelSelectionFreezeError(
                f"{dataset} prospective-power gate changed"
            )
        if not gate_attested:
            raise JointModelSelectionFreezeError(
                f"{dataset} model-selection gate lacks typed verifier attestation"
            )
        if include_upstream_hashes:
            hashes = row.get("upstream_hashes")
            expected_hashes = {
                "artifact_sha256",
                "receipt_sha256",
                "public_report_sha256",
                "cross_variant_alignment_sha256",
            }
            if not isinstance(hashes, Mapping) or set(hashes) != expected_hashes:
                raise JointModelSelectionFreezeError(
                    f"{dataset} upstream hash schema changed"
                )
            for key in expected_hashes:
                _require_sha256(hashes[key], f"{dataset}.{key}")
    if include_upstream_hashes:
        rows = cast(Mapping[str, Mapping[str, object]], datasets)
        for key in (
            "artifact_sha256",
            "receipt_sha256",
            "public_report_sha256",
        ):
            values = {
                cast(Mapping[str, object], rows[dataset]["upstream_hashes"])[key]
                for dataset in REQUIRED_DATASETS
            }
            if len(values) != len(REQUIRED_DATASETS):
                raise JointModelSelectionFreezeError(
                    f"dataset upstream {key} values must be independently hash-bound"
                )


def _validate_predicate(
    predicate: object, datasets: Mapping[str, object]
) -> bool:
    expected = {
        "exact_required_dataset_roster_passed",
        "same_exact_confirmatory_contract_passed",
        "all_dataset_prospective_power_gates_passed",
        "all_dataset_model_selection_gates_attested",
        "all_dataset_model_selection_gates_passed",
        "joint_model_selection_freeze_passed",
        "predicate_is_conjunctive",
    }
    if not isinstance(predicate, Mapping) or set(predicate) != expected:
        raise JointModelSelectionFreezeError("joint predicate schema changed")
    for key in expected:
        _require_bool(predicate[key], f"joint_predicate.{key}")
    rows = cast(Mapping[str, Mapping[str, object]], datasets)
    all_power = all(rows[name]["power_gate_passed"] is True for name in REQUIRED_DATASETS)
    all_attested = all(
        rows[name]["model_selection_gate_attested"] is True
        for name in REQUIRED_DATASETS
    )
    all_gates = all(
        rows[name]["model_selection_gate_attested"] is True
        and rows[name]["model_selection_gate_passed"] is True
        for name in REQUIRED_DATASETS
    )
    joint = all_power and all_gates
    if dict(predicate) != {
        "exact_required_dataset_roster_passed": True,
        "same_exact_confirmatory_contract_passed": True,
        "all_dataset_prospective_power_gates_passed": all_power,
        "all_dataset_model_selection_gates_attested": all_attested,
        "all_dataset_model_selection_gates_passed": all_gates,
        "joint_model_selection_freeze_passed": joint,
        "predicate_is_conjunctive": True,
    }:
        raise JointModelSelectionFreezeError("joint predicate values changed")
    return joint


def _validate_stage_authorization(
    authorization: object,
    *,
    joint_passed: bool,
    datasets: Mapping[str, object],
) -> tuple[str, ...]:
    expected = {
        "separate_calibration_stage_workflow_authorized",
        "calibration_outcome_access_authorized_by_this_layer",
        "confirmatory_method_success_authorized",
        "internal_holdout_unseal_authorized",
        "external_test_unseal_authorized",
        "failure_reasons",
        "reason",
    }
    if not isinstance(authorization, Mapping) or set(authorization) != expected:
        raise JointModelSelectionFreezeError("joint stage authorization schema changed")
    authorized = _require_bool(
        authorization["separate_calibration_stage_workflow_authorized"],
        "separate_calibration_stage_workflow_authorized",
    )
    if authorized != joint_passed:
        raise JointModelSelectionFreezeError("joint calibration authority changed")
    for field in (
        "calibration_outcome_access_authorized_by_this_layer",
        "confirmatory_method_success_authorized",
        "internal_holdout_unseal_authorized",
        "external_test_unseal_authorized",
    ):
        if authorization[field] is not False:
            raise JointModelSelectionFreezeError(
                "joint freeze improperly authorizes outcomes, success, holdout, or test"
            )
    reasons = authorization["failure_reasons"]
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise JointModelSelectionFreezeError("joint failure reasons changed")
    expected_reasons = _expected_failure_reasons_from_dataset_payloads(datasets)
    if reasons != expected_reasons or bool(reasons) == joint_passed:
        raise JointModelSelectionFreezeError("joint failure reasons are inconsistent")
    expected_reason = (
        "joint_predicate_passed_authorize_separate_calibration_workflow_only"
        if joint_passed
        else "joint_predicate_failed_keep_all_later_roles_sealed"
    )
    if authorization["reason"] != expected_reason:
        raise JointModelSelectionFreezeError("joint stage reason changed")
    return tuple(cast(list[str], reasons))


def validate_joint_model_selection_public_report(payload: Mapping[str, object]) -> None:
    """Validate the exact, path-free and outcome-hash-free public schema."""

    expected = {
        "schema_version",
        "status",
        "claim_boundary",
        "required_dataset_roster",
        "datasets",
        "joint_predicate",
        "stage_authorization",
        "public_artifact_policy",
    }
    if set(payload) != expected:
        raise JointModelSelectionFreezeError("joint public report schema changed")
    if (
        payload.get("schema_version") != PUBLIC_REPORT_SCHEMA
        or payload.get("status") != _STATUS
        or payload.get("claim_boundary") != _CLAIM_BOUNDARY
        or payload.get("required_dataset_roster") != list(REQUIRED_DATASETS)
    ):
        raise JointModelSelectionFreezeError("joint public report identity changed")
    _validate_dataset_payloads(payload.get("datasets"), include_upstream_hashes=False)
    datasets = cast(Mapping[str, object], payload["datasets"])
    joint_passed = _validate_predicate(payload.get("joint_predicate"), datasets)
    _validate_stage_authorization(
        payload.get("stage_authorization"),
        joint_passed=joint_passed,
        datasets=datasets,
    )
    if payload.get("public_artifact_policy") != {
        "aggregate_only": True,
        "contains_row_cluster_or_seed_vectors": False,
        "contains_private_paths": False,
        "contains_outcome_hashes": False,
        "contains_labels_predictions_or_probabilities": False,
        "confirmatory_method_success_authorized": False,
        "holdout_or_test_access_authorized": False,
    }:
        raise JointModelSelectionFreezeError("joint public privacy policy changed")
    _visit_aggregate(payload, public=True)


def _validate_private_artifact(payload: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "status",
        "claim_boundary",
        "required_dataset_roster",
        "analysis_contract",
        "source_snapshot_lineage",
        "datasets",
        "joint_predicate",
        "stage_authorization",
        "public_report_sha256",
        "aggregate_handoff_contract",
        "private_artifact_policy",
    }
    if set(payload) != expected:
        raise JointModelSelectionFreezeError("joint private artifact schema changed")
    if (
        payload.get("schema_version") != PRIVATE_ARTIFACT_SCHEMA
        or payload.get("status") != _STATUS
        or payload.get("claim_boundary") != _CLAIM_BOUNDARY
        or payload.get("required_dataset_roster") != list(REQUIRED_DATASETS)
        or payload.get("analysis_contract")
        != {"analysis_config_sha256": CONFIRMATORY_ANALYSIS_SHA256}
    ):
        raise JointModelSelectionFreezeError("joint private artifact identity changed")
    _validate_source_snapshot_lineage(payload.get("source_snapshot_lineage"))
    _require_sha256(payload.get("public_report_sha256"), "public_report_sha256")
    _validate_dataset_payloads(payload.get("datasets"), include_upstream_hashes=True)
    datasets = cast(Mapping[str, object], payload["datasets"])
    joint_passed = _validate_predicate(payload.get("joint_predicate"), datasets)
    _validate_stage_authorization(
        payload.get("stage_authorization"),
        joint_passed=joint_passed,
        datasets=datasets,
    )
    if payload.get("aggregate_handoff_contract") != {
        "two_required_datasets_only": True,
        "aggregate_handoff_only": True,
        "label_probability_or_outcome_path_capability_exposed": False,
        "separate_calibration_workflow_is_only_possible_positive_authority": True,
    }:
        raise JointModelSelectionFreezeError("joint aggregate handoff changed")
    if payload.get("private_artifact_policy") != {
        "aggregate_only": True,
        "contains_row_cluster_or_seed_vectors": False,
        "contains_private_paths": False,
        "contains_upstream_hashes_for_lineage_only": True,
        "contains_labels_predictions_or_probabilities": False,
    }:
        raise JointModelSelectionFreezeError("joint private policy changed")
    _visit_aggregate(payload, public=False)


def _validate_receipt(payload: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "status",
        "claim_boundary",
        "required_dataset_roster",
        "lineage",
        "completion_contract",
        "downstream_handoff",
    }
    if set(payload) != expected:
        raise JointModelSelectionFreezeError("joint receipt schema changed")
    if (
        payload.get("schema_version") != PRIVATE_RECEIPT_SCHEMA
        or payload.get("status") != _STATUS
        or payload.get("claim_boundary") != _CLAIM_BOUNDARY
        or payload.get("required_dataset_roster") != list(REQUIRED_DATASETS)
    ):
        raise JointModelSelectionFreezeError("joint receipt identity changed")
    lineage = payload.get("lineage")
    expected_lineage = {
        "private_artifact_sha256",
        "public_report_sha256",
        "analysis_config_sha256",
        "source_snapshot_lineage",
        "upstream_artifact_sha256_by_dataset",
        "upstream_receipt_sha256_by_dataset",
        "upstream_public_report_sha256_by_dataset",
        "cross_variant_alignment_sha256_by_dataset",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != expected_lineage:
        raise JointModelSelectionFreezeError("joint receipt lineage schema changed")
    for field in (
        "private_artifact_sha256",
        "public_report_sha256",
        "analysis_config_sha256",
    ):
        _require_sha256(lineage[field], field)
    if lineage["analysis_config_sha256"] != CONFIRMATORY_ANALYSIS_SHA256:
        raise JointModelSelectionFreezeError("joint receipt analysis contract changed")
    _validate_source_snapshot_lineage(lineage["source_snapshot_lineage"])
    for field in (
        "upstream_artifact_sha256_by_dataset",
        "upstream_receipt_sha256_by_dataset",
        "upstream_public_report_sha256_by_dataset",
        "cross_variant_alignment_sha256_by_dataset",
    ):
        values = lineage[field]
        if not isinstance(values, Mapping) or set(values) != set(REQUIRED_DATASETS):
            raise JointModelSelectionFreezeError(
                f"joint receipt {field} roster changed"
            )
        for dataset in REQUIRED_DATASETS:
            _require_sha256(values[dataset], f"{field}.{dataset}")
        if field != "cross_variant_alignment_sha256_by_dataset" and len(
            {values[dataset] for dataset in REQUIRED_DATASETS}
        ) != len(REQUIRED_DATASETS):
            raise JointModelSelectionFreezeError(
                f"joint receipt {field} values must be independently hash-bound"
            )

    completion = payload.get("completion_contract")
    expected_completion = {
        "frozen_reference_by_dataset",
        "prospective_power_by_dataset",
        "power_gate_passed_by_dataset",
        "model_selection_gate_attested_by_dataset",
        "model_selection_gate_passed_by_dataset",
        "joint_model_selection_freeze_passed",
        "separate_calibration_stage_workflow_authorized",
        "calibration_outcome_access_authorized_by_this_layer",
        "confirmatory_method_success_authorized",
        "internal_holdout_unseal_authorized",
        "external_test_unseal_authorized",
        "failure_reasons",
        "aggregate_only",
    }
    if not isinstance(completion, Mapping) or set(completion) != expected_completion:
        raise JointModelSelectionFreezeError("joint receipt completion schema changed")
    map_fields = (
        "frozen_reference_by_dataset",
        "prospective_power_by_dataset",
        "power_gate_passed_by_dataset",
        "model_selection_gate_attested_by_dataset",
        "model_selection_gate_passed_by_dataset",
    )
    for field in map_fields:
        values = completion[field]
        if not isinstance(values, Mapping) or set(values) != set(REQUIRED_DATASETS):
            raise JointModelSelectionFreezeError(
                f"joint receipt {field} roster changed"
            )
    for dataset in REQUIRED_DATASETS:
        if completion["frozen_reference_by_dataset"][dataset] not in REFERENCE_CANDIDATES:
            raise JointModelSelectionFreezeError("receipt frozen reference changed")
        power = _require_probability(
            completion["prospective_power_by_dataset"][dataset],
            f"receipt.{dataset}.prospective_power",
        )
        power_gate = _require_bool(
            completion["power_gate_passed_by_dataset"][dataset],
            f"receipt.{dataset}.power_gate_passed",
        )
        gate_attested = _require_bool(
            completion["model_selection_gate_attested_by_dataset"][dataset],
            f"receipt.{dataset}.model_selection_gate_attested",
        )
        gate_passed = _require_bool(
            completion["model_selection_gate_passed_by_dataset"][dataset],
            f"receipt.{dataset}.model_selection_gate_passed",
        )
        if power_gate != (power >= MINIMUM_PROSPECTIVE_POWER):
            raise JointModelSelectionFreezeError("receipt power gate changed")
        if not gate_attested:
            raise JointModelSelectionFreezeError(
                "receipt lacks typed model-selection gate attestation"
            )
    all_power = all(
        completion["power_gate_passed_by_dataset"][name] is True
        for name in REQUIRED_DATASETS
    )
    all_model = all(
        completion["model_selection_gate_attested_by_dataset"][name] is True
        and completion["model_selection_gate_passed_by_dataset"][name] is True
        for name in REQUIRED_DATASETS
    )
    joint = all_power and all_model
    if (
        completion["joint_model_selection_freeze_passed"] is not joint
        or completion["separate_calibration_stage_workflow_authorized"] is not joint
        or completion["calibration_outcome_access_authorized_by_this_layer"] is not False
        or completion["confirmatory_method_success_authorized"] is not False
        or completion["internal_holdout_unseal_authorized"] is not False
        or completion["external_test_unseal_authorized"] is not False
        or completion["aggregate_only"] is not True
    ):
        raise JointModelSelectionFreezeError("joint receipt authority changed")
    reasons = completion["failure_reasons"]
    expected_reasons: list[str] = []
    for dataset in REQUIRED_DATASETS:
        if completion["power_gate_passed_by_dataset"][dataset] is not True:
            expected_reasons.append(f"{dataset}:prospective_power_below_0.80")
        if completion["model_selection_gate_attested_by_dataset"][dataset] is not True:
            raise JointModelSelectionFreezeError(
                "receipt lacks typed model-selection gate attestation"
            )
        if completion["model_selection_gate_passed_by_dataset"][dataset] is not True:
            expected_reasons.append(
                f"{dataset}:upstream_model_selection_gate_failed"
            )
    if (
        not isinstance(reasons, list)
        or reasons != expected_reasons
        or bool(reasons) == joint
    ):
        raise JointModelSelectionFreezeError("joint receipt failure reasons changed")
    if payload.get("downstream_handoff") != {
        "verifier": "verify_joint_model_selection_freeze_receipt",
        "hash_bound_artifact_and_receipt_required": True,
        "aggregate_handoff_only": True,
        "only_possible_authorized_next_workflow": "separate_calibration_stage",
        "never_authorizes_confirmatory_holdout_or_test": True,
    }:
        raise JointModelSelectionFreezeError("joint downstream handoff changed")
    _visit_aggregate(payload, public=False)


def _repository_external_new_root(value: str | Path) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise JointModelSelectionFreezeError(
            "private joint-freeze root must be an absolute external path"
        )
    root = raw.resolve(strict=False)
    if root == Path(root.anchor) or not root.name:
        raise JointModelSelectionFreezeError("private joint-freeze root is too broad")
    try:
        parent = root.parent.resolve(strict=True)
    except OSError as error:
        raise JointModelSelectionFreezeError(
            f"private joint-freeze parent is unavailable: {error}"
        ) from error
    if not parent.is_dir():
        raise JointModelSelectionFreezeError(
            "private joint-freeze parent must be a directory"
        )
    repository_root = Path(__file__).resolve().parents[3]
    if _is_within(root, repository_root):
        raise JointModelSelectionFreezeError(
            "private joint-freeze root must be repository-external"
        )
    try:
        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--absolute-git-dir"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise JointModelSelectionFreezeError(
            f"cannot establish repository-external private root: {error}"
        ) from error
    if result.returncode == 0:
        raise JointModelSelectionFreezeError(
            "private joint-freeze root must be external to every Git repository"
        )
    if result.returncode != 128:
        raise JointModelSelectionFreezeError(
            "cannot establish repository-external private root ownership"
        )
    if root.exists() or root.is_symlink():
        raise FileExistsError("private joint-freeze root already exists")
    return root


def _public_destination(value: str | Path, private_root: Path) -> Path:
    path = Path(value).resolve(strict=False)
    if path.suffix.lower() != ".json" or not path.name:
        raise JointModelSelectionFreezeError(
            "joint public report must be a JSON file"
        )
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise JointModelSelectionFreezeError(
            f"joint public report parent is unavailable: {error}"
        ) from error
    if not parent.is_dir() or _is_within(path, private_root):
        raise JointModelSelectionFreezeError(
            "joint public report must be separate from the private root"
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError("joint public report already exists")
    return path


def _atomic_write_once(path: Path, payload: bytes) -> str:
    """Publish by hard link and rehash the destination to close TOCTOU."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"write-once output already exists: {path.name}")
    descriptor = -1
    temporary: Path | None = None
    expected = _sha256_bytes(payload)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"write-once output already exists: {path.name}"
            ) from None
        except OSError as error:
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    f"write-once output already exists: {path.name}"
                ) from error
            raise JointModelSelectionFreezeError(
                f"cannot atomically publish {path.name}: {error}"
            ) from error
        if _sha256_file(path) != expected:
            raise JointModelSelectionFreezeError(
                f"write-once output changed during publication: {path.name}"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return expected


def run_joint_model_selection_freeze(
    *,
    inputs: Mapping[str, ModelSelectionReferenceFreezeInput],
    source_snapshot: ProductionSourceSnapshotAttestation,
    private_output_root: str | Path,
    public_report_path: str | Path,
) -> CompletedJointModelSelectionFreeze:
    """Verify, freeze and publish the exact EmotionTalk+MELD joint predicate."""

    source_lineage = _source_snapshot_lineage(source_snapshot)
    verified = _verify_upstream_handoffs(inputs)
    root = _repository_external_new_root(private_output_root)
    public_path = _public_destination(public_report_path, root)

    reasons = _failure_reasons(verified)
    predicate = _joint_predicate(verified)
    authorization = _stage_authorization(predicate, reasons)
    public_report: dict[str, object] = {
        "schema_version": PUBLIC_REPORT_SCHEMA,
        "status": _STATUS,
        "claim_boundary": _CLAIM_BOUNDARY,
        "required_dataset_roster": list(REQUIRED_DATASETS),
        "datasets": _dataset_payloads(verified, include_upstream_hashes=False),
        "joint_predicate": predicate,
        "stage_authorization": authorization,
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_cluster_or_seed_vectors": False,
            "contains_private_paths": False,
            "contains_outcome_hashes": False,
            "contains_labels_predictions_or_probabilities": False,
            "confirmatory_method_success_authorized": False,
            "holdout_or_test_access_authorized": False,
        },
    }
    validate_joint_model_selection_public_report(public_report)
    public_bytes = _canonical_json_bytes(public_report)
    public_sha = _sha256_bytes(public_bytes)

    private_artifact: dict[str, object] = {
        "schema_version": PRIVATE_ARTIFACT_SCHEMA,
        "status": _STATUS,
        "claim_boundary": _CLAIM_BOUNDARY,
        "required_dataset_roster": list(REQUIRED_DATASETS),
        "analysis_contract": {
            "analysis_config_sha256": CONFIRMATORY_ANALYSIS_SHA256
        },
        "source_snapshot_lineage": source_lineage,
        "datasets": _dataset_payloads(verified, include_upstream_hashes=True),
        "joint_predicate": predicate,
        "stage_authorization": authorization,
        "public_report_sha256": public_sha,
        "aggregate_handoff_contract": {
            "two_required_datasets_only": True,
            "aggregate_handoff_only": True,
            "label_probability_or_outcome_path_capability_exposed": False,
            "separate_calibration_workflow_is_only_possible_positive_authority": True,
        },
        "private_artifact_policy": {
            "aggregate_only": True,
            "contains_row_cluster_or_seed_vectors": False,
            "contains_private_paths": False,
            "contains_upstream_hashes_for_lineage_only": True,
            "contains_labels_predictions_or_probabilities": False,
        },
    }
    _validate_private_artifact(private_artifact)
    private_bytes = _canonical_json_bytes(private_artifact)
    private_sha = _sha256_bytes(private_bytes)

    attestations = {name: verified[name][0] for name in REQUIRED_DATASETS}
    receipt: dict[str, object] = {
        "schema_version": PRIVATE_RECEIPT_SCHEMA,
        "status": _STATUS,
        "claim_boundary": _CLAIM_BOUNDARY,
        "required_dataset_roster": list(REQUIRED_DATASETS),
        "lineage": {
            "private_artifact_sha256": private_sha,
            "public_report_sha256": public_sha,
            "analysis_config_sha256": CONFIRMATORY_ANALYSIS_SHA256,
            "source_snapshot_lineage": source_lineage,
            "upstream_artifact_sha256_by_dataset": {
                name: attestations[name].artifact_sha256 for name in REQUIRED_DATASETS
            },
            "upstream_receipt_sha256_by_dataset": {
                name: attestations[name].receipt_sha256 for name in REQUIRED_DATASETS
            },
            "upstream_public_report_sha256_by_dataset": {
                name: attestations[name].public_report_sha256
                for name in REQUIRED_DATASETS
            },
            "cross_variant_alignment_sha256_by_dataset": {
                name: attestations[name].cross_variant_alignment_sha256
                for name in REQUIRED_DATASETS
            },
        },
        "completion_contract": {
            "frozen_reference_by_dataset": {
                name: attestations[name].frozen_reference for name in REQUIRED_DATASETS
            },
            "prospective_power_by_dataset": {
                name: float(attestations[name].prospective_power)
                for name in REQUIRED_DATASETS
            },
            "power_gate_passed_by_dataset": {
                name: bool(attestations[name].power_gate_passed)
                for name in REQUIRED_DATASETS
            },
            "model_selection_gate_attested_by_dataset": {
                name: verified[name][1] for name in REQUIRED_DATASETS
            },
            "model_selection_gate_passed_by_dataset": {
                name: verified[name][2] for name in REQUIRED_DATASETS
            },
            "joint_model_selection_freeze_passed": predicate[
                "joint_model_selection_freeze_passed"
            ],
            "separate_calibration_stage_workflow_authorized": authorization[
                "separate_calibration_stage_workflow_authorized"
            ],
            "calibration_outcome_access_authorized_by_this_layer": False,
            "confirmatory_method_success_authorized": False,
            "internal_holdout_unseal_authorized": False,
            "external_test_unseal_authorized": False,
            "failure_reasons": list(reasons),
            "aggregate_only": True,
        },
        "downstream_handoff": {
            "verifier": "verify_joint_model_selection_freeze_receipt",
            "hash_bound_artifact_and_receipt_required": True,
            "aggregate_handoff_only": True,
            "only_possible_authorized_next_workflow": "separate_calibration_stage",
            "never_authorizes_confirmatory_holdout_or_test": True,
        },
    }
    _validate_receipt(receipt)
    receipt_bytes = _canonical_json_bytes(receipt)
    receipt_sha = _sha256_bytes(receipt_bytes)

    # The new directory itself is claimed with no-clobber semantics before any
    # private output is published.  Each file then uses an atomic hard link.
    root.mkdir(exist_ok=False)
    artifact_path = root / PRIVATE_ARTIFACT_NAME
    receipt_path = root / PRIVATE_RECEIPT_NAME
    # The receipt is the commit marker.  Publish and reverify both artifacts
    # before it, so a failed public publication can never leave a verifier-
    # acceptable completion receipt behind.
    observed_private_sha = _atomic_write_once(artifact_path, private_bytes)
    observed_public_sha = _atomic_write_once(public_path, public_bytes)
    if (
        observed_private_sha != private_sha
        or observed_public_sha != public_sha
        or _sha256_file(artifact_path) != private_sha
        or _sha256_file(public_path) != public_sha
    ):
        raise JointModelSelectionFreezeError(
            "joint artifacts changed before receipt commit"
        )
    _atomic_write_once(receipt_path, receipt_bytes)
    return CompletedJointModelSelectionFreeze(
        private_artifact_path=artifact_path,
        private_artifact_sha256=private_sha,
        private_receipt_path=receipt_path,
        private_receipt_sha256=receipt_sha,
        public_report_path=public_path,
        public_report_sha256=public_sha,
        calibration_stage_workflow_authorized=bool(
            authorization["separate_calibration_stage_workflow_authorized"]
        ),
        failure_reasons=tuple(reasons),
    )


def _decode_canonical_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JointModelSelectionFreezeError(
            f"cannot decode aggregate joint handoff {path.name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise JointModelSelectionFreezeError("aggregate joint handoff must be an object")
    if raw != _canonical_json_bytes(payload):
        raise JointModelSelectionFreezeError(
            f"aggregate joint handoff is not canonical JSON: {path.name}"
        )
    return payload, raw


def verify_joint_model_selection_freeze_receipt(
    artifact_path: str | Path,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    *,
    source_snapshot: ProductionSourceSnapshotAttestation,
) -> VerifiedJointModelSelectionFreezeAttestation:
    """Reverify a hash-bound joint aggregate without any outcome capability."""

    expected_source_lineage = _source_snapshot_lineage(source_snapshot)
    artifact_file = Path(artifact_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    if (
        artifact_file.name != PRIVATE_ARTIFACT_NAME
        or receipt_file.name != PRIVATE_RECEIPT_NAME
        or artifact_file.parent != receipt_file.parent
    ):
        raise JointModelSelectionFreezeError("joint handoff paths are not canonical")
    expected_receipt = _require_sha256(
        expected_receipt_sha256, "expected_receipt_sha256"
    )
    if _sha256_file(receipt_file) != expected_receipt:
        raise JointModelSelectionFreezeError("joint receipt hash changed")
    receipt, receipt_raw = _decode_canonical_json(receipt_file)
    if _sha256_bytes(receipt_raw) != expected_receipt:
        raise JointModelSelectionFreezeError(
            "joint receipt changed while decoding"
        )
    artifact, artifact_raw = _decode_canonical_json(artifact_file)
    artifact_sha = _sha256_bytes(artifact_raw)
    _validate_receipt(receipt)
    _validate_private_artifact(artifact)
    lineage = cast(Mapping[str, object], receipt["lineage"])
    completion = cast(Mapping[str, object], receipt["completion_contract"])
    artifact_datasets = cast(Mapping[str, Mapping[str, object]], artifact["datasets"])
    if artifact["source_snapshot_lineage"] != lineage["source_snapshot_lineage"]:
        raise JointModelSelectionFreezeError(
            "joint artifact and receipt source snapshot lineage diverged"
        )
    if artifact["source_snapshot_lineage"] != expected_source_lineage:
        raise JointModelSelectionFreezeError(
            "joint source snapshot lineage does not match the expected snapshot"
        )
    if (
        artifact_sha != lineage["private_artifact_sha256"]
        or artifact["public_report_sha256"] != lineage["public_report_sha256"]
        or artifact["analysis_contract"]
        != {"analysis_config_sha256": lineage["analysis_config_sha256"]}
        or artifact["joint_predicate"]["joint_model_selection_freeze_passed"]
        != completion["joint_model_selection_freeze_passed"]
        or artifact["stage_authorization"][
            "separate_calibration_stage_workflow_authorized"
        ]
        != completion["separate_calibration_stage_workflow_authorized"]
        or artifact["stage_authorization"]["failure_reasons"]
        != completion["failure_reasons"]
    ):
        raise JointModelSelectionFreezeError("joint receipt and artifact diverged")
    lineage_maps = {
        "artifact_sha256": "upstream_artifact_sha256_by_dataset",
        "receipt_sha256": "upstream_receipt_sha256_by_dataset",
        "public_report_sha256": "upstream_public_report_sha256_by_dataset",
        "cross_variant_alignment_sha256": (
            "cross_variant_alignment_sha256_by_dataset"
        ),
    }
    completion_maps = {
        "frozen_reference": "frozen_reference_by_dataset",
        "prospective_power": "prospective_power_by_dataset",
        "power_gate_passed": "power_gate_passed_by_dataset",
        "model_selection_gate_attested": (
            "model_selection_gate_attested_by_dataset"
        ),
        "model_selection_gate_passed": "model_selection_gate_passed_by_dataset",
    }
    for dataset in REQUIRED_DATASETS:
        row = artifact_datasets[dataset]
        upstream = cast(Mapping[str, object], row["upstream_hashes"])
        for row_key, receipt_key in lineage_maps.items():
            receipt_map = cast(Mapping[str, object], lineage[receipt_key])
            if upstream[row_key] != receipt_map[dataset]:
                raise JointModelSelectionFreezeError(
                    f"{dataset} upstream joint lineage diverged"
                )
        for row_key, receipt_key in completion_maps.items():
            receipt_map = cast(Mapping[str, object], completion[receipt_key])
            if row[row_key] != receipt_map[dataset]:
                raise JointModelSelectionFreezeError(
                    f"{dataset} frozen joint aggregate diverged"
                )
    if (
        _sha256_file(receipt_file) != expected_receipt
        or _sha256_file(artifact_file) != artifact_sha
    ):
        raise JointModelSelectionFreezeError(
            "joint handoff changed while verifying"
        )
    frozen = cast(Mapping[str, str], completion["frozen_reference_by_dataset"])
    powers = cast(Mapping[str, float], completion["prospective_power_by_dataset"])
    upstream_artifacts = cast(
        Mapping[str, str], lineage["upstream_artifact_sha256_by_dataset"]
    )
    upstream_receipts = cast(
        Mapping[str, str], lineage["upstream_receipt_sha256_by_dataset"]
    )
    upstream_public_reports = cast(
        Mapping[str, str], lineage["upstream_public_report_sha256_by_dataset"]
    )
    cross_variant_alignments = cast(
        Mapping[str, str], lineage["cross_variant_alignment_sha256_by_dataset"]
    )
    power_gates = cast(
        Mapping[str, bool], completion["power_gate_passed_by_dataset"]
    )
    model_gate_attestations = cast(
        Mapping[str, bool], completion["model_selection_gate_attested_by_dataset"]
    )
    model_gates = cast(
        Mapping[str, bool], completion["model_selection_gate_passed_by_dataset"]
    )
    return VerifiedJointModelSelectionFreezeAttestation(
        dataset_roster=REQUIRED_DATASETS,
        artifact_path=artifact_file,
        artifact_sha256=artifact_sha,
        receipt_path=receipt_file,
        receipt_sha256=expected_receipt,
        public_report_sha256=cast(str, lineage["public_report_sha256"]),
        analysis_config_sha256=cast(str, lineage["analysis_config_sha256"]),
        source_snapshot_manifest_sha256=cast(
            str, expected_source_lineage["manifest_sha256"]
        ),
        source_snapshot_git_commit=cast(str, expected_source_lineage["git_commit"]),
        source_snapshot_git_tree=cast(str, expected_source_lineage["git_tree"]),
        source_snapshot_code_bundle_sha256=cast(
            str, expected_source_lineage["code_bundle_sha256"]
        ),
        frozen_reference_by_dataset=MappingProxyType(dict(frozen)),
        prospective_power_by_dataset=MappingProxyType(
            {name: float(powers[name]) for name in REQUIRED_DATASETS}
        ),
        power_gate_passed_by_dataset=MappingProxyType(dict(power_gates)),
        upstream_artifact_sha256_by_dataset=MappingProxyType(
            dict(upstream_artifacts)
        ),
        upstream_receipt_sha256_by_dataset=MappingProxyType(
            dict(upstream_receipts)
        ),
        upstream_public_report_sha256_by_dataset=MappingProxyType(
            dict(upstream_public_reports)
        ),
        cross_variant_alignment_sha256_by_dataset=MappingProxyType(
            dict(cross_variant_alignments)
        ),
        model_selection_gate_attested_by_dataset=MappingProxyType(
            dict(model_gate_attestations)
        ),
        model_selection_gate_passed_by_dataset=MappingProxyType(dict(model_gates)),
        calibration_stage_workflow_authorized=cast(
            bool, completion["separate_calibration_stage_workflow_authorized"]
        ),
        failure_reasons=tuple(cast(list[str], completion["failure_reasons"])),
    )
