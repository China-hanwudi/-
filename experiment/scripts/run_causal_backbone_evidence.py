from __future__ import annotations

import sys

# ``sys`` is built in and cannot be shadowed by a file beside this script.
# Remove the script directory and empty current-directory entries before the
# first non-built-in import, then explicitly add only the closed source root.
sys.dont_write_bytecode = True
_BOOTSTRAP_SCRIPT_DIRECTORY = __file__.replace("\\", "/").rsplit("/", 1)[0]
_BOOTSTRAP_SCRIPT_DIRECTORY_CASEFOLD = _BOOTSTRAP_SCRIPT_DIRECTORY.casefold().rstrip(
    "/"
)


def _bootstrap_path_is_script_directory(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold().rstrip("/")
    if not normalized:
        return True
    return normalized == _BOOTSTRAP_SCRIPT_DIRECTORY_CASEFOLD or (
        not _BOOTSTRAP_SCRIPT_DIRECTORY_CASEFOLD.startswith("/")
        and ":/" not in _BOOTSTRAP_SCRIPT_DIRECTORY_CASEFOLD
        and normalized.endswith("/" + _BOOTSTRAP_SCRIPT_DIRECTORY_CASEFOLD)
    )


sys.path[:] = [
    value
    for value in sys.path
    if type(value) is str and not _bootstrap_path_is_script_directory(value)
]

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_require_plain_python_package(source_root: Path) -> None:
    """Reject import-shadow payloads before the first package import.

    The full Git/source attestation is necessarily imported from this tree, so
    its own bootstrap cannot depend on package code.  Walk with stdlib lstat /
    scandir only, never follow links, and accept only ordinary directories and
    ordinary ``.py`` files.  In particular, ignored native extensions and pyc
    caches cannot execute ahead of the later source-snapshot verifier.
    """

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def require_plain_directory(path: Path) -> None:
        try:
            status = os.lstat(path)
        except OSError:
            raise SystemExit(
                "production package bootstrap integrity check failed closed"
            ) from None
        attributes = int(getattr(status, "st_file_attributes", 0))
        if not stat.S_ISDIR(status.st_mode) or attributes & reparse_flag:
            raise SystemExit(
                "production package bootstrap integrity check failed closed"
            )

    require_plain_directory(source_root)
    try:
        with os.scandir(source_root) as iterator:
            source_entries = list(iterator)
    except OSError:
        raise SystemExit(
            "production package bootstrap integrity check failed closed"
        ) from None
    if len(source_entries) != 1 or source_entries[0].name != "hva_affect":
        raise SystemExit(
            "production package bootstrap integrity check failed closed"
        )
    package_entry = source_entries[0]
    try:
        package_status = package_entry.stat(follow_symlinks=False)
    except OSError:
        raise SystemExit(
            "production package bootstrap integrity check failed closed"
        ) from None
    package_attributes = int(getattr(package_status, "st_file_attributes", 0))
    if (
        not stat.S_ISDIR(package_status.st_mode)
        or package_attributes & reparse_flag
    ):
        raise SystemExit(
            "production package bootstrap integrity check failed closed"
        )
    package_root = Path(package_entry.path)
    pending = [package_root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError:
            raise SystemExit(
                "production package bootstrap integrity check failed closed"
            ) from None
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError:
                raise SystemExit(
                    "production package bootstrap integrity check failed closed"
                ) from None
            attributes = int(getattr(status, "st_file_attributes", 0))
            if attributes & reparse_flag:
                raise SystemExit(
                    "production package bootstrap integrity check failed closed"
                )
            if stat.S_ISDIR(status.st_mode):
                pending.append(Path(entry.path))
            elif not (
                stat.S_ISREG(status.st_mode) and entry.name.endswith(".py")
            ):
                raise SystemExit(
                    "production package bootstrap integrity check failed closed"
                )


_bootstrap_require_plain_python_package(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_backbone_evidence_runner import (  # noqa: E402
    capture_runtime_environment,
    materialize_selection_features_after_receipt,
    run_fit_preflight,
)


_REGISTERED_VARIANTS = (
    "full",
    "no_vad",
    "no_history_3x3",
    "capacity_control",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_CONFIG_NAME = "production_source_snapshot_v1"


class _StrategyUpstreamPaths:
    __slots__ = (
        "history_artifact",
        "history_receipt",
        "history_receipt_sha256",
        "backbone_config",
        "fit_receipt",
        "fit_receipt_sha256",
        "config_paths",
    )

    def __init__(
        self,
        *,
        history_artifact: Path,
        history_receipt: Path,
        history_receipt_sha256: str,
        backbone_config: Path,
        fit_receipt: Path,
        fit_receipt_sha256: str,
        config_paths: Mapping[str, Path],
    ) -> None:
        self.history_artifact = history_artifact
        self.history_receipt = history_receipt
        self.history_receipt_sha256 = history_receipt_sha256
        self.backbone_config = backbone_config
        self.fit_receipt = fit_receipt
        self.fit_receipt_sha256 = fit_receipt_sha256
        self.config_paths = config_paths


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return value


def _variant_path(value: str) -> tuple[str, Path]:
    variant, path = _named_path(value)
    if variant not in _REGISTERED_VARIANTS:
        raise argparse.ArgumentTypeError(
            "variant must be full, no_vad, no_history_3x3, or capacity_control"
        )
    return variant, path


def _variant_sha256(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected VARIANT=SHA256")
    variant, digest = value.split("=", 1)
    if variant not in _REGISTERED_VARIANTS:
        raise argparse.ArgumentTypeError(
            "variant must be full, no_vad, no_history_3x3, or capacity_control"
        )
    return variant, _sha256_argument(digest)


def _variant_named_path(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected VARIANT:NAME=PATH")
    raw_name, raw_path = value.split("=", 1)
    variant, name = raw_name.split(":", 1)
    if variant not in _REGISTERED_VARIANTS:
        raise argparse.ArgumentTypeError(
            "variant must be full, no_vad, no_history_3x3, or capacity_control"
        )
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty VARIANT:NAME=PATH")
    return variant, name, Path(raw_path)


def _mapping(values: list[tuple[str, Path]], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise SystemExit(f"duplicate {label} name: {name}")
        result[name] = path
    return result


def _exact_variant_mapping(
    values: list[tuple[str, object]], label: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for variant, value in values:
        if variant in result:
            raise SystemExit(f"duplicate {label} variant: {variant}")
        result[variant] = value
    if tuple(result) != _REGISTERED_VARIANTS and set(result) != set(
        _REGISTERED_VARIANTS
    ):
        missing = sorted(set(_REGISTERED_VARIANTS) - set(result))
        unknown = sorted(set(result) - set(_REGISTERED_VARIANTS))
        raise SystemExit(
            f"{label} roster must contain exactly the four registered variants: "
            f"missing={missing}, unknown={unknown}"
        )
    return {variant: result[variant] for variant in _REGISTERED_VARIANTS}


def _exact_variant_named_mapping(
    values: list[tuple[str, str, Path]], label: str
) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {
        variant: {} for variant in _REGISTERED_VARIANTS
    }
    for variant, name, path in values:
        if name in result[variant]:
            raise SystemExit(f"duplicate {label} name for {variant}: {name}")
        result[variant][name] = path
    missing = [variant for variant in _REGISTERED_VARIANTS if not result[variant]]
    if missing:
        raise SystemExit(f"{label} mapping is missing variants: {missing}")
    return result


def _add_verified_receipt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("EmotionTalk", "MELD"), required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--sidecar-manifest", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument(
        "--fit-receipt-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--config",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
        help="repeat the exact hash-bound config mapping used by fit-preflight",
    )
    _add_source_snapshot_arguments(parser)


def _add_source_snapshot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-snapshot-manifest",
        type=Path,
        required=True,
        help="repository-external production_source_snapshot_v1 manifest",
    )
    parser.add_argument(
        "--source-snapshot-manifest-sha256",
        type=_sha256_argument,
        required=True,
        help="expected byte SHA-256 of the immutable source snapshot manifest",
    )
    parser.add_argument(
        "--source-snapshot-worktree-root",
        type=Path,
        required=True,
        help=(
            "exact clean detached worktree frozen by the source snapshot; this "
            "CLI and imported hva_affect package must execute from that root"
        ),
    )


def _add_current_only_common(parser: argparse.ArgumentParser) -> None:
    _add_verified_receipt_arguments(parser)
    parser.add_argument("--fit-map", type=Path, required=True)
    parser.add_argument("--fit-lineage", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")


def _add_history_common(parser: argparse.ArgumentParser) -> None:
    _add_verified_receipt_arguments(parser)
    parser.add_argument("--fit-map", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")


def _add_completion_triplet(
    parser: argparse.ArgumentParser,
    *,
    prefix: str,
    help_label: str,
) -> None:
    parser.add_argument(
        f"--{prefix}-artifact",
        type=Path,
        required=True,
        help=f"canonical external {help_label} artifact",
    )
    parser.add_argument(
        f"--{prefix}-receipt",
        type=Path,
        required=True,
        help=f"canonical external {help_label} completion receipt",
    )
    parser.add_argument(
        f"--{prefix}-receipt-sha256",
        type=_sha256_argument,
        required=True,
        help=f"expected byte SHA-256 of the {help_label} completion receipt",
    )


def _add_joint_dataset_handoff_arguments(
    parser: argparse.ArgumentParser,
    *,
    dataset: str,
    option_prefix: str,
) -> None:
    parser.add_argument(
        f"--{option_prefix}-model-selection-artifact",
        dest=f"{option_prefix}_model_selection_artifact",
        type=Path,
        required=True,
        help=f"canonical private aggregate model-selection artifact for {dataset}",
    )
    parser.add_argument(
        f"--{option_prefix}-model-selection-receipt",
        dest=f"{option_prefix}_model_selection_receipt",
        type=Path,
        required=True,
        help=f"canonical private aggregate model-selection receipt for {dataset}",
    )
    parser.add_argument(
        f"--{option_prefix}-model-selection-receipt-sha256",
        dest=f"{option_prefix}_model_selection_receipt_sha256",
        type=_sha256_argument,
        required=True,
        help=f"expected byte SHA-256 of the {dataset} model-selection receipt",
    )


def _add_exact_variant_argument(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    value_type,
    metavar: str,
    help_text: str,
) -> None:
    parser.add_argument(
        option,
        action="append",
        type=value_type,
        default=[],
        required=True,
        metavar=metavar,
        help=f"repeat exactly once for each registered variant; {help_text}",
    )


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _attest_history_completion_for_current_only(
    *,
    artifact_path: Path,
    completion_receipt_path: Path,
    expected_completion_receipt_sha256: str,
    dataset: str,
    fit_preflight_receipt_sha256: str,
    config_paths: dict[str, Path],
    code_paths: dict[str, Path],
    model_config: object,
    run_config: object,
    runtime_environment_sha256: str,
    execution_environment_sha256: str,
):
    """Verify and bind history production before any selection capability exists."""

    from hva_affect.causal_backbone_history_staged_pipeline import (
        verify_history_completion_production_attestation,
    )

    attestation = verify_history_completion_production_attestation(
        artifact_path,
        completion_receipt_path,
        expected_completion_receipt_sha256,
    )
    expected_config = {
        name: _sha256(path) for name, path in sorted(config_paths.items())
    }
    expected_code = {
        name: _sha256(path) for name, path in sorted(code_paths.items())
    }
    changed: list[str] = []
    if attestation.dataset != dataset:
        changed.append("dataset")
    if (
        attestation.fit_preflight_receipt_sha256
        != fit_preflight_receipt_sha256
    ):
        changed.append("fit_preflight_receipt_sha256")
    if dict(attestation.config_sha256) != expected_config:
        changed.append("config_sha256")
    if dict(attestation.code_sha256) != expected_code:
        changed.append("code_sha256")
    if attestation.runtime_environment_sha256 != runtime_environment_sha256:
        changed.append("runtime_environment_sha256")
    if attestation.execution_environment_sha256 != execution_environment_sha256:
        changed.append("execution_environment_sha256")
    if attestation.model_config_sha256 != _canonical_sha256(
        asdict(model_config)  # type: ignore[arg-type]
    ):
        changed.append("model_config_sha256")
    if attestation.run_config_sha256 != _canonical_sha256(
        asdict(run_config)  # type: ignore[arg-type]
    ):
        changed.append("run_config_sha256")
    if changed:
        raise SystemExit(
            "history completion differs from current-only frozen lineage: "
            f"{changed}"
        )
    return attestation


def _load_backbone_config(path: Path):
    from hva_affect.causal_multimodal_backbone import CausalBackboneConfig
    from hva_affect.emotiontalk_causal_backbone_runner import BackboneRunConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("backbone config root must be a mapping")
    model = CausalBackboneConfig.from_mapping(payload)
    run = BackboneRunConfig.from_mapping(payload)
    return model, run


def _load_utility_config(path: Path):
    from hva_affect.emotiontalk_causal_backbone_runner import UtilitySamplingConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("utility config root must be a mapping")
    return UtilitySamplingConfig.from_mapping(payload)


def _verify_source_snapshot(args):
    import hva_affect

    from hva_affect.production_source_snapshot_v1 import (
        REQUIRED_CLI_PATH,
        REQUIRED_PACKAGE_ROOT,
        verify_production_source_snapshot,
    )

    attestation = verify_production_source_snapshot(
        manifest_path=args.source_snapshot_manifest,
        expected_manifest_sha256=args.source_snapshot_manifest_sha256,
        worktree_root=args.source_snapshot_worktree_root,
    )
    code_paths = dict(attestation.stable_code_paths())
    expected_cli = code_paths.get(REQUIRED_CLI_PATH)
    expected_package = (
        attestation.worktree_root / Path(REQUIRED_PACKAGE_ROOT)
    ).resolve(strict=True)
    imported_package = Path(hva_affect.__file__).resolve(strict=True).parent
    if (
        expected_cli is None
        or Path(expected_cli).resolve(strict=True)
        != Path(__file__).resolve(strict=True)
        or attestation.worktree_root != ROOT.parent.resolve(strict=True)
        or imported_package != expected_package
    ):
        raise SystemExit(
            "production CLI/package is not executing from the attested source snapshot"
        )
    return attestation


def _production_code_sha256(code_paths: Mapping[str, Path]) -> str:
    return _canonical_sha256(
        {name: _sha256(Path(path)) for name, path in sorted(code_paths.items())}
    )


def _snapshot_summary_fields(source_snapshot) -> dict[str, object]:
    return {
        "source_snapshot_schema_version": "production_source_snapshot_v1",
        "source_snapshot_manifest_sha256": source_snapshot.manifest_sha256,
        "source_snapshot_git_commit": source_snapshot.commit_sha,
        "source_snapshot_git_tree": source_snapshot.tree_sha,
        "source_snapshot_code_bundle_sha256": _production_code_sha256(
            source_snapshot.stable_code_paths()
        ),
    }


def _run_create_production_source_snapshot(args) -> None:
    from hva_affect.production_source_snapshot_v1 import (
        ProductionSourceSnapshotError,
        create_production_source_snapshot,
    )

    try:
        attestation = create_production_source_snapshot(
            worktree_root=args.source_snapshot_worktree_root,
            output_path=args.source_snapshot_output_manifest,
        )
        snapshot_fields = _snapshot_summary_fields(attestation)
    except (ProductionSourceSnapshotError, FileExistsError, OSError):
        # Snapshot errors can contain repository-external paths.  The CLI is an
        # aggregate-only boundary, so those details are never rendered.
        raise SystemExit("production source snapshot creation failed closed") from None
    summary = {
        "schema_version": "carma_production_source_snapshot_cli_v1",
        "status": "immutable_production_source_snapshot_created",
        "operation": "create",
        "source_file_count": len(attestation.stable_code_paths()),
        "clean_detached_worktree_verified": True,
        "outcome_data_accessed": False,
        "training_run": False,
    }
    summary.update(snapshot_fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _run_verify_production_source_snapshot(args) -> None:
    from hva_affect.production_source_snapshot_v1 import ProductionSourceSnapshotError

    try:
        # This stronger wrapper calls the v1 verifier and additionally proves
        # that this CLI and imported package are executing from that snapshot.
        attestation = _verify_source_snapshot(args)
        snapshot_fields = _snapshot_summary_fields(attestation)
    except (ProductionSourceSnapshotError, FileExistsError, OSError):
        raise SystemExit("production source snapshot verification failed closed") from None
    summary = {
        "schema_version": "carma_production_source_snapshot_cli_v1",
        "status": "immutable_production_source_snapshot_verified",
        "operation": "verify",
        "source_file_count": len(attestation.stable_code_paths()),
        "clean_detached_worktree_verified": True,
        "executing_from_verified_snapshot": True,
        "outcome_data_accessed": False,
        "training_run": False,
    }
    summary.update(snapshot_fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _joint_model_selection_summary(attestation, *, operation: str) -> dict[str, object]:
    datasets = tuple(attestation.dataset_roster)
    return {
        "schema_version": "carma_joint_model_selection_freeze_cli_v1",
        "status": "aggregate_two_dataset_joint_model_selection_freeze_verified",
        "operation": operation,
        "dataset_roster": list(datasets),
        "analysis_config_sha256": attestation.analysis_config_sha256,
        "private_artifact_sha256": attestation.artifact_sha256,
        "private_receipt_sha256": attestation.receipt_sha256,
        "public_report_sha256": attestation.public_report_sha256,
        "frozen_reference_by_dataset": {
            name: attestation.frozen_reference_by_dataset[name] for name in datasets
        },
        "prospective_power_by_dataset": {
            name: float(attestation.prospective_power_by_dataset[name])
            for name in datasets
        },
        "power_gate_passed_by_dataset": {
            name: bool(attestation.power_gate_passed_by_dataset[name])
            for name in datasets
        },
        "model_selection_gate_attested_by_dataset": {
            name: bool(attestation.model_selection_gate_attested_by_dataset[name])
            for name in datasets
        },
        "model_selection_gate_passed_by_dataset": {
            name: bool(attestation.model_selection_gate_passed_by_dataset[name])
            for name in datasets
        },
        "joint_model_selection_freeze_passed": bool(
            attestation.calibration_stage_workflow_authorized
        ),
        "separate_calibration_stage_workflow_authorized": bool(
            attestation.calibration_stage_workflow_authorized
        ),
        "failure_reasons": list(attestation.failure_reasons),
        "aggregate_only": True,
        "label_probability_or_row_capability_exposed": False,
        "calibration_outcome_access_authorized_by_this_layer": False,
        "confirmatory_method_success_authorized": False,
        "internal_holdout_unseal_authorized": False,
        "external_test_unseal_authorized": False,
    }


def _run_joint_model_selection_freeze(args) -> None:
    from hva_affect.causal_backbone_joint_model_selection_freeze import (
        JointModelSelectionFreezeError,
        ModelSelectionReferenceFreezeInput,
        run_joint_model_selection_freeze,
        verify_joint_model_selection_freeze_receipt,
    )
    from hva_affect.production_source_snapshot_v1 import ProductionSourceSnapshotError

    try:
        source_snapshot = _verify_source_snapshot(args)
        completed = run_joint_model_selection_freeze(
            inputs={
                "EmotionTalk": ModelSelectionReferenceFreezeInput(
                    artifact_path=args.emotiontalk_model_selection_artifact,
                    receipt_path=args.emotiontalk_model_selection_receipt,
                    expected_receipt_sha256=(
                        args.emotiontalk_model_selection_receipt_sha256
                    ),
                ),
                "MELD": ModelSelectionReferenceFreezeInput(
                    artifact_path=args.meld_model_selection_artifact,
                    receipt_path=args.meld_model_selection_receipt,
                    expected_receipt_sha256=args.meld_model_selection_receipt_sha256,
                ),
            },
            source_snapshot=source_snapshot,
            private_output_root=args.joint_private_output_root,
            public_report_path=args.joint_public_report,
        )
        # Never trust a just-written receipt merely because this process wrote
        # it: re-enter through the public aggregate verifier before reporting.
        attestation = verify_joint_model_selection_freeze_receipt(
            completed.private_artifact_path,
            completed.private_receipt_path,
            completed.private_receipt_sha256,
            source_snapshot=source_snapshot,
        )
        snapshot_fields = _snapshot_summary_fields(source_snapshot)
    except (
        JointModelSelectionFreezeError,
        ProductionSourceSnapshotError,
        FileExistsError,
        OSError,
    ):
        raise SystemExit("joint model-selection freeze failed closed") from None
    summary = _joint_model_selection_summary(attestation, operation="run")
    summary.update(snapshot_fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _run_verify_joint_model_selection_freeze(args) -> None:
    from hva_affect.causal_backbone_joint_model_selection_freeze import (
        JointModelSelectionFreezeError,
        verify_joint_model_selection_freeze_receipt,
    )
    from hva_affect.production_source_snapshot_v1 import ProductionSourceSnapshotError

    try:
        source_snapshot = _verify_source_snapshot(args)
        attestation = verify_joint_model_selection_freeze_receipt(
            args.joint_private_artifact,
            args.joint_private_receipt,
            args.joint_private_receipt_sha256,
            source_snapshot=source_snapshot,
        )
        snapshot_fields = _snapshot_summary_fields(source_snapshot)
    except (
        JointModelSelectionFreezeError,
        ProductionSourceSnapshotError,
        FileExistsError,
        OSError,
    ):
        raise SystemExit("joint model-selection freeze verification failed closed") from None
    summary = _joint_model_selection_summary(attestation, operation="verify")
    summary.update(snapshot_fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _bind_source_snapshot_config(
    config_paths: Mapping[str, Path], source_snapshot
) -> dict[str, Path]:
    """Bind manifest/commit/tree into every downstream private lineage hash.

    The canonical manifest contains the exact Git commit, tree, and recursive
    source hashes.  Treating that immutable JSON as one reserved lineage config
    preserves the strict relative source-path code mapping while making every
    preflight/producer claim cryptographically depend on the snapshot itself.
    """

    if _SOURCE_SNAPSHOT_CONFIG_NAME in config_paths:
        raise SystemExit("source snapshot lineage config name is reserved")
    result = {str(name): Path(path) for name, path in config_paths.items()}
    result[_SOURCE_SNAPSHOT_CONFIG_NAME] = Path(
        source_snapshot.manifest_path
    ).resolve(strict=True)
    return dict(sorted(result.items()))


def _verify_frozen_production_inputs(
    *,
    backbone_config: Path,
    utility_config: Path | None = None,
    config_paths: dict[str, Path],
    code_paths: dict[str, Path],
    source_snapshot,
) -> None:
    """Require the actual trainer inputs to be the preflight-mapped files."""

    resolved_config = backbone_config.resolve(strict=True)
    frozen_configs = {path.resolve(strict=True) for path in config_paths.values()}
    if resolved_config not in frozen_configs:
        raise SystemExit("--backbone-config is not one of the preflight-frozen configs")
    if utility_config is not None and utility_config.resolve(strict=True) not in frozen_configs:
        raise SystemExit("--utility-config is not one of the preflight-frozen configs")
    canonical_paths = dict(source_snapshot.stable_code_paths())
    canonical_names = tuple(canonical_paths)
    missing = sorted(set(canonical_names) - set(code_paths))
    unknown = sorted(set(code_paths) - set(canonical_names))
    changed = [
        name
        for name in canonical_names
        if name in code_paths
        and code_paths[name].resolve(strict=True) != canonical_paths[name].resolve(strict=True)
    ]
    if missing or changed or unknown:
        raise SystemExit(
            "production code mapping differs from the verified source snapshot: "
            f"missing={missing}, changed={changed}, unknown={unknown}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Staged CARMA causal-evidence utilities. fit-preflight performs only "
            "hash/structural validation; it never trains or computes performance."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_create = subparsers.add_parser(
        "create-production-source-snapshot",
        help=(
            "freeze the recursive production CLI/package source set from one "
            "clean detached worktree into a repository-external manifest"
        ),
    )
    snapshot_create.add_argument(
        "--source-snapshot-worktree-root",
        type=Path,
        required=True,
        help="exact clean detached Git worktree to freeze",
    )
    snapshot_create.add_argument(
        "--source-snapshot-output-manifest",
        type=Path,
        required=True,
        help="new repository-external write-once snapshot manifest",
    )

    snapshot_verify = subparsers.add_parser(
        "verify-production-source-snapshot",
        help=(
            "verify one immutable source manifest and prove that this CLI/package "
            "is executing from its exact clean detached worktree"
        ),
    )
    _add_source_snapshot_arguments(snapshot_verify)

    joint_run = subparsers.add_parser(
        "run-joint-model-selection-freeze",
        help=(
            "verify the aggregate EmotionTalk and MELD model-selection receipts "
            "and publish their exact conjunctive freeze"
        ),
    )
    _add_source_snapshot_arguments(joint_run)
    _add_joint_dataset_handoff_arguments(
        joint_run, dataset="EmotionTalk", option_prefix="emotiontalk"
    )
    _add_joint_dataset_handoff_arguments(
        joint_run, dataset="MELD", option_prefix="meld"
    )
    joint_run.add_argument(
        "--joint-private-output-root",
        type=Path,
        required=True,
        help="new absolute repository-external write-once joint-freeze directory",
    )
    joint_run.add_argument(
        "--joint-public-report",
        type=Path,
        required=True,
        help="new aggregate-only public JSON report",
    )

    joint_verify = subparsers.add_parser(
        "verify-joint-model-selection-freeze",
        help="reverify one hash-bound aggregate EmotionTalk+MELD joint freeze",
    )
    _add_source_snapshot_arguments(joint_verify)
    joint_verify.add_argument(
        "--joint-private-artifact", type=Path, required=True
    )
    joint_verify.add_argument(
        "--joint-private-receipt", type=Path, required=True
    )
    joint_verify.add_argument(
        "--joint-private-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )

    fit = subparsers.add_parser(
        "fit-preflight",
        help="materialise fit sidecars, hash selection sidecars, and write a receipt",
    )
    fit.add_argument("--dataset", choices=("EmotionTalk", "MELD"), required=True)
    fit.add_argument("--sidecar-dir", type=Path, required=True)
    fit.add_argument("--sidecar-manifest", type=Path, required=True)
    fit.add_argument("--receipt", type=Path, required=True)
    fit.add_argument(
        "--config",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
        help="hash-bound configuration file; repeat for every frozen config",
    )
    _add_source_snapshot_arguments(fit)
    lineage_create = subparsers.add_parser(
        "fit-lineage-create",
        help=(
            "write the private fit protocol map and outcome-free fit lineage; "
            "selection payload files are never opened"
        ),
    )
    _add_verified_receipt_arguments(lineage_create)
    lineage_create.add_argument("--fit-map", type=Path, required=True)
    lineage_create.add_argument("--fit-lineage", type=Path, required=True)
    lineage_validate = subparsers.add_parser(
        "fit-lineage-validate",
        help=(
            "revalidate the private fit protocol map and fit lineage without "
            "opening selection payload files"
        ),
    )
    _add_verified_receipt_arguments(lineage_validate)
    lineage_validate.add_argument("--fit-map", type=Path, required=True)
    lineage_validate.add_argument("--fit-lineage", type=Path, required=True)
    current_fit = subparsers.add_parser(
        "current-only-fit",
        help=(
            "train independent history-stripped fit OOF models; selection payloads "
            "are not API inputs"
        ),
    )
    _add_current_only_common(current_fit)
    current_fit.add_argument(
        "--resume",
        action="store_true",
        help="resume partial checkpoints only when the private lineage claim matches",
    )

    complete = subparsers.add_parser(
        "current-only-complete-selection",
        help=(
            "load complete checkpoints only, open selection features (never labels), "
            "and write the strategy-consumable current-only cache"
        ),
    )
    _add_current_only_common(complete)
    history_artifact = complete.add_mutually_exclusive_group(required=True)
    history_artifact.add_argument(
        "--history-complete-artifact",
        dest="history_complete_artifact",
        type=Path,
        help=(
            "canonical repo-external staged history-complete-outcome.npz; "
            "it must match --history-completion-receipt"
        ),
    )
    history_artifact.add_argument(
        "--producer",
        dest="history_complete_artifact",
        type=Path,
        help=argparse.SUPPRESS,
    )
    complete.add_argument(
        "--fit-producer-receipt-sha256", type=_sha256_argument, required=True
    )
    complete.add_argument("--history-completion-receipt", type=Path, required=True)
    complete.add_argument(
        "--history-completion-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    history_fit = subparsers.add_parser(
        "history-fit",
        help=(
            "train canonical history-aware fit OOF models in one new external "
            "private directory; no selection payload is an API input"
        ),
    )
    _add_history_common(history_fit)
    history_fit.add_argument(
        "--resume",
        action="store_true",
        help="resume partial checkpoints only when the private lineage claim matches",
    )
    history_complete = subparsers.add_parser(
        "history-complete-selection",
        help=(
            "semantically restore every complete history checkpoint, then open "
            "selection features only and write the outcome-free cache"
        ),
    )
    _add_history_common(history_complete)
    history_complete.add_argument(
        "--fit-outcome-sha256", type=_sha256_argument, required=True
    )
    history_complete.add_argument(
        "--fit-targets-sha256", type=_sha256_argument, required=True
    )
    history_complete.add_argument(
        "--fit-producer-receipt-sha256", type=_sha256_argument, required=True
    )

    strategy_complete = subparsers.add_parser(
        "strategy-complete-selection",
        help=(
            "verify one history variant, the unique full-history current anchor, "
            "and the independent current-only cache before producing the fixed "
            "outcome-free strategy roster"
        ),
    )
    _add_verified_receipt_arguments(strategy_complete)
    strategy_complete.add_argument(
        "--backbone-config",
        type=Path,
        required=True,
        help=(
            "the variant model contract; its registered variant is derived from "
            "validated model fields and is never accepted as a caller label"
        ),
    )
    strategy_complete.add_argument(
        "--full-anchor-backbone-config",
        type=Path,
        required=True,
        help="the full model contract bound to the independent current-only cache",
    )
    strategy_complete.add_argument(
        "--full-anchor-fit-receipt",
        type=Path,
        required=True,
        help="fit-preflight receipt bound to the full-history current anchor",
    )
    strategy_complete.add_argument(
        "--full-anchor-fit-receipt-sha256",
        type=_sha256_argument,
        required=True,
        help="expected byte SHA-256 of the full-anchor fit-preflight receipt",
    )
    _add_completion_triplet(
        strategy_complete,
        prefix="history-complete",
        help_label="variant history",
    )
    _add_completion_triplet(
        strategy_complete,
        prefix="full-history-anchor",
        help_label="full-history anchor",
    )
    _add_completion_triplet(
        strategy_complete,
        prefix="current-complete",
        help_label="independent current-only",
    )
    strategy_complete.add_argument(
        "--private-output-dir", type=Path, required=True
    )
    strategy_complete.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser(
        "evaluate-model-selection",
        help=(
            "production-attest all four strategy variants, then open the one "
            "canonical model-selection label archive and freeze aggregate results"
        ),
    )
    evaluate.add_argument(
        "--dataset", choices=("EmotionTalk", "MELD"), required=True
    )
    evaluate.add_argument("--sidecar-dir", type=Path, required=True)
    evaluate.add_argument("--sidecar-manifest", type=Path, required=True)
    _add_exact_variant_argument(
        evaluate,
        "--fit-receipt",
        value_type=_variant_path,
        metavar="VARIANT=PATH",
        help_text="variant-specific fit-preflight receipt",
    )
    _add_exact_variant_argument(
        evaluate,
        "--fit-receipt-sha256",
        value_type=_variant_sha256,
        metavar="VARIANT=SHA256",
        help_text="expected fit-preflight receipt SHA-256",
    )
    evaluate.add_argument(
        "--variant-config",
        action="append",
        type=_variant_named_path,
        default=[],
        required=True,
        metavar="VARIANT:NAME=PATH",
        help=(
            "variant-specific exact fit-preflight config mapping; each variant "
            "must contain exactly one model-derivable backbone config"
        ),
    )
    _add_source_snapshot_arguments(evaluate)
    _add_exact_variant_argument(
        evaluate,
        "--backbone-config",
        value_type=_variant_path,
        metavar="VARIANT=PATH",
        help_text="actual model contract used to derive the variant",
    )
    _add_exact_variant_argument(
        evaluate,
        "--history-complete-artifact",
        value_type=_variant_path,
        metavar="VARIANT=PATH",
        help_text="canonical external history completion artifact",
    )
    _add_exact_variant_argument(
        evaluate,
        "--history-completion-receipt",
        value_type=_variant_path,
        metavar="VARIANT=PATH",
        help_text="canonical external history completion receipt",
    )
    _add_exact_variant_argument(
        evaluate,
        "--history-completion-receipt-sha256",
        value_type=_variant_sha256,
        metavar="VARIANT=SHA256",
        help_text="expected history completion receipt SHA-256",
    )
    _add_completion_triplet(
        evaluate,
        prefix="current-complete",
        help_label="full-anchored independent current-only",
    )
    _add_exact_variant_argument(
        evaluate,
        "--strategy-complete-artifact",
        value_type=_variant_path,
        metavar="VARIANT=PATH",
        help_text="canonical external strategy completion artifact",
    )
    _add_exact_variant_argument(
        evaluate,
        "--strategy-completion-receipt",
        value_type=_variant_path,
        metavar="VARIANT=PATH",
        help_text="canonical external strategy completion receipt",
    )
    _add_exact_variant_argument(
        evaluate,
        "--strategy-completion-receipt-sha256",
        value_type=_variant_sha256,
        metavar="VARIANT=SHA256",
        help_text="expected strategy completion receipt SHA-256",
    )
    evaluate.add_argument(
        "--confirmatory-analysis",
        type=Path,
        required=True,
        help="byte-frozen confirmatory analysis contract",
    )
    evaluate.add_argument("--private-output-dir", type=Path, required=True)
    evaluate.add_argument("--public-report", type=Path, required=True)
    return parser


def _history_private_paths(root: Path) -> dict[str, Path]:
    return {
        "checkpoint": root / "checkpoints",
        "fit_outcome": root / "history-fit-outcome.npz",
        "fit_targets": root / "history-fit-targets.npz",
        "fit_receipt": root / "history-fit-receipt.json",
        "complete_outcome": root / "history-complete-outcome.npz",
        "complete_receipt": root / "history-complete-receipt.json",
    }


def _run_history_command(
    *, args, fit, fit_map, configs, code, source_snapshot
) -> None:
    from hva_affect.causal_backbone_history_staged_pipeline import (
        claim_or_resume_history_private_root,
        complete_history_selection_outcomes,
        history_production_claim_sha256,
        materialize_history_selection_features_after_fit_gate,
        produce_history_fit_only,
        verify_history_fit_for_completion,
    )
    from hva_affect.emotiontalk_causal_backbone_runner import _runtime_environment

    model_config, run_config = _load_backbone_config(args.backbone_config)
    utility_config = _load_utility_config(args.utility_config)
    device = _resolve_device(args.device)
    environment = capture_runtime_environment()
    execution_environment = _runtime_environment(device)
    config_hashes = {name: _sha256(path) for name, path in sorted(configs.items())}
    code_hashes = {name: _sha256(path) for name, path in sorted(code.items())}
    runtime_sha = _canonical_sha256(environment)
    execution_runtime_sha = _canonical_sha256(execution_environment)
    production_claim = history_production_claim_sha256(
        fit=fit,
        fit_map=fit_map,
        fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        model_config=model_config,
        run_config=run_config,
        utility_config=utility_config,
        config_sha256=config_hashes,
        code_sha256=code_hashes,
        runtime_environment_sha256=runtime_sha,
        execution_environment_sha256=execution_runtime_sha,
    )
    if args.command == "history-fit":
        private_root = claim_or_resume_history_private_root(
            args.private_output_dir,
            production_claim_sha256=production_claim,
            allow_resume=bool(args.resume),
        )
    else:
        private_root = args.private_output_dir.resolve(strict=True)
        if not private_root.is_dir():
            raise SystemExit("--private-output-dir must be the history-fit directory")
    paths = _history_private_paths(private_root)

    if args.command == "history-fit":
        produced = produce_history_fit_only(
            fit=fit,
            fit_map=fit_map,
            fit_preflight_receipt_path=args.fit_receipt,
            expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
            checkpoint_root=paths["checkpoint"],
            outcome_artifact_path=paths["fit_outcome"],
            targets_artifact_path=paths["fit_targets"],
            producer_receipt_path=paths["fit_receipt"],
            model_config=model_config,
            run_config=run_config,
            utility_config=utility_config,
            config_sha256=config_hashes,
            code_sha256=code_hashes,
            runtime_environment_sha256=runtime_sha,
            device=device,
            production_run_claim_sha256=production_claim,
            execution_environment_sha256=execution_runtime_sha,
        )
        if not produced.production_trainer:
            raise SystemExit("history-fit did not use the canonical production trainer")
        summary = {
            "schema_version": "carma_history_backbone_fit_receipt_v1",
            "status": "history_fit_oof_complete_not_performance_evidence",
            "dataset": fit.dataset,
            "fit_rows": fit.rows,
            "fit_outcome_sha256": produced.outcome_artifact_sha256,
            "fit_targets_sha256": produced.targets_artifact_sha256,
            "fit_producer_receipt_sha256": produced.receipt_sha256,
            "checkpoint_manifest_sha256": produced.checkpoint_manifest.manifest_sha256,
            "canonical_production_trainer": True,
            "selection_payload_consumed": False,
            "performance_metric_computed": False,
        }
        summary.update(_snapshot_summary_fields(source_snapshot))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command != "history-complete-selection":
        raise SystemExit("unsupported history command")
    if paths["complete_outcome"].exists() or paths["complete_receipt"].exists():
        raise SystemExit("history complete-selection outputs already exist")
    fit_state = verify_history_fit_for_completion(
        fit=fit,
        fit_map=fit_map,
        fit_outcome_artifact_path=paths["fit_outcome"],
        expected_fit_outcome_artifact_sha256=args.fit_outcome_sha256,
        fit_targets_artifact_path=paths["fit_targets"],
        expected_fit_targets_artifact_sha256=args.fit_targets_sha256,
        fit_producer_receipt_path=paths["fit_receipt"],
        expected_fit_producer_receipt_sha256=args.fit_producer_receipt_sha256,
        fit_preflight_receipt_path=args.fit_receipt,
        expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        checkpoint_root=paths["checkpoint"],
        model_config=model_config,
        run_config=run_config,
        utility_config=utility_config,
        config_sha256=config_hashes,
        code_sha256=code_hashes,
        runtime_environment_sha256=runtime_sha,
        execution_environment_sha256=execution_runtime_sha,
    )
    if not fit_state.production_trainer:
        raise SystemExit("history completion refuses a synthetic fit receipt")
    selection = materialize_history_selection_features_after_fit_gate(
        fit=fit,
        fit_state=fit_state,
        checkpoint_root=paths["checkpoint"],
        model_config=model_config,
        run_config=run_config,
        fit_preflight_receipt_path=args.fit_receipt,
        dataset=args.dataset,
        sidecar_dir=args.sidecar_dir,
        manifest_path=args.sidecar_manifest,
        config_paths=configs,
        code_paths=code,
        environment=environment,
        execution_environment=execution_environment,
    )
    completed = complete_history_selection_outcomes(
        fit=fit,
        selection=selection,
        fit_map=fit_map,
        fit_state=fit_state,
        fit_preflight_receipt_path=args.fit_receipt,
        expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        checkpoint_root=paths["checkpoint"],
        model_config=model_config,
        run_config=run_config,
        utility_config=utility_config,
        config_sha256=config_hashes,
        code_sha256=code_hashes,
        runtime_environment_sha256=runtime_sha,
        execution_environment_sha256=execution_runtime_sha,
        config_paths=configs,
        code_paths=code,
        environment=environment,
        execution_environment=execution_environment,
        sidecar_dir=args.sidecar_dir,
        manifest_path=args.sidecar_manifest,
        selection_feature_sha256=fit_state.selection_feature_sha256,
        artifact_path=paths["complete_outcome"],
        completion_receipt_path=paths["complete_receipt"],
        device=device,
    )
    summary = {
        "schema_version": "carma_history_backbone_complete_receipt_v1",
        "status": "selection_outcome_free_probability_cache_complete",
        "dataset": fit.dataset,
        "artifact_sha256": completed.artifact_sha256,
        "completion_receipt_sha256": completed.receipt_sha256,
        "checkpoint_manifest_sha256": completed.checkpoint_manifest_sha256,
        "complete_checkpoint_only": True,
        "selection_label_file_accessed": False,
        "selection_utility_target_computed": False,
        "performance_metric_computed": False,
    }
    summary.update(_snapshot_summary_fields(source_snapshot))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _assert_registered_variant_contract() -> None:
    from hva_affect.causal_backbone_strategy_staged_pipeline import (
        REGISTERED_VARIANTS,
    )

    if tuple(REGISTERED_VARIANTS) != _REGISTERED_VARIANTS:
        raise SystemExit("CLI and production registered-variant rosters differ")


def _require_backbone_in_config_mapping(
    backbone_config: Path,
    config_paths: Mapping[str, Path],
    *,
    label: str,
) -> None:
    resolved = backbone_config.resolve(strict=True)
    frozen = {Path(path).resolve(strict=True) for path in config_paths.values()}
    if resolved not in frozen:
        raise SystemExit(f"{label} is not one of its fit-preflight-frozen configs")


def _load_registered_backbone_contract(path: Path, *, dataset: str):
    from hva_affect.causal_backbone_strategy_staged_pipeline import (
        derive_registered_variant,
    )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read registered backbone config: {error}") from error
    if not isinstance(payload, Mapping):
        raise SystemExit("registered backbone config root must be a mapping")
    model, run = _load_backbone_config(path)
    variant = derive_registered_variant(model)
    experiment = payload.get("experimental_contract")
    if (
        not isinstance(experiment, Mapping)
        or experiment.get("dataset_id") != dataset
        or experiment.get("variant") != variant
        or experiment.get("primary_variant") != "full"
        or experiment.get("model_selection_may_choose_variant") is not False
    ):
        raise SystemExit(
            "backbone experimental contract differs from its model-derived variant"
        )
    return variant, model, run


def _load_exact_variant_config_mapping(
    *,
    backbone_config: Path,
    config_paths: Mapping[str, Path],
    dataset: str,
):
    _require_backbone_in_config_mapping(
        backbone_config,
        config_paths,
        label="variant backbone config",
    )
    candidates: list[Path] = []
    for raw_path in config_paths.values():
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read fit-preflight config: {error}") from error
        if not isinstance(payload, Mapping):
            raise SystemExit("fit-preflight config root must be a mapping")
        model = payload.get("model")
        if isinstance(model, Mapping) and "affect_relation_mode" in model:
            candidates.append(path.resolve(strict=True))
    expected = backbone_config.resolve(strict=True)
    if len(candidates) != 1 or candidates[0] != expected:
        raise SystemExit(
            "each strategy config mapping must contain exactly its one actual "
            "model-derivable backbone config"
        )
    return _load_registered_backbone_contract(backbone_config, dataset=dataset)


def _load_strategy_feature_views(
    *,
    dataset: str,
    sidecar_dir: Path,
    sidecar_manifest: Path,
    fit_receipt: Path,
    fit_receipt_sha256: str,
):
    from hva_affect.causal_backbone_evidence_runner import FIT_ROLE, SELECTION_ROLE
    from hva_affect.causal_backbone_strategy_staged_pipeline import (
        load_outcome_free_role_features,
    )

    fit_features = load_outcome_free_role_features(
        role=FIT_ROLE,
        dataset=dataset,
        feature_path=sidecar_dir / f"features_{FIT_ROLE}.npz",
        manifest_path=sidecar_manifest,
        fit_preflight_receipt_path=fit_receipt,
        expected_fit_preflight_receipt_sha256=fit_receipt_sha256,
    )
    selection_features = load_outcome_free_role_features(
        role=SELECTION_ROLE,
        dataset=dataset,
        feature_path=sidecar_dir / f"features_{SELECTION_ROLE}.npz",
        manifest_path=sidecar_manifest,
        fit_preflight_receipt_path=fit_receipt,
        expected_fit_preflight_receipt_sha256=fit_receipt_sha256,
    )
    return fit_features, selection_features


def _verify_history_completion_triplet(
    artifact: Path,
    receipt: Path,
    receipt_sha256: str,
):
    from hva_affect.causal_backbone_history_staged_pipeline import (
        verify_history_completion_production_attestation,
    )

    return verify_history_completion_production_attestation(
        artifact,
        receipt,
        receipt_sha256,
    )


def _verify_full_anchored_current(
    *,
    full_history_anchor,
    full_fit_receipt: Path,
    full_fit_receipt_sha256: str,
    current_artifact: Path,
    current_receipt: Path,
    current_receipt_sha256: str,
):
    from hva_affect.causal_backbone_current_only_pipeline import (
        load_attested_history_fit_alignment_view,
        load_attested_history_producer_alignment_view,
    )
    from hva_affect.causal_backbone_strategy_staged_pipeline import (
        verify_current_only_completion_production_attestation,
    )

    fit_alignment = load_attested_history_fit_alignment_view(
        full_history_anchor,
        fit_preflight_receipt_path=full_fit_receipt,
        expected_fit_preflight_receipt_sha256=full_fit_receipt_sha256,
    )
    producer_alignment = load_attested_history_producer_alignment_view(
        full_history_anchor,
        fit_producer=fit_alignment,
    )
    current = verify_current_only_completion_production_attestation(
        current_artifact,
        current_receipt,
        current_receipt_sha256,
        history_attestation=full_history_anchor,
        producer_alignment=producer_alignment,
    )
    return current


def _verify_strategy_upstream(
    *,
    paths: _StrategyUpstreamPaths,
    history_attestation,
    full_history_anchor,
    current_attestation,
    full_anchor_model_config,
    dataset: str,
    sidecar_dir: Path,
    sidecar_manifest: Path,
):
    from hva_affect.causal_backbone_strategy_staged_pipeline import (
        verify_strategy_upstream_state,
    )

    registered_variant, model_config, run_config = _load_exact_variant_config_mapping(
        backbone_config=paths.backbone_config,
        config_paths=paths.config_paths,
        dataset=dataset,
    )
    fit_features, selection_features = _load_strategy_feature_views(
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        sidecar_manifest=sidecar_manifest,
        fit_receipt=paths.fit_receipt,
        fit_receipt_sha256=paths.fit_receipt_sha256,
    )
    upstream = verify_strategy_upstream_state(
        history_attestation=history_attestation,
        current_attestation=current_attestation,
        full_history_anchor_attestation=full_history_anchor,
        full_history_anchor_model_config=full_anchor_model_config,
        fit_features=fit_features,
        selection_features=selection_features,
    )
    return registered_variant, model_config, run_config, upstream


def _run_strategy_complete_selection(args) -> None:
    from hva_affect.causal_backbone_strategy_staged_pipeline import (
        complete_strategy_selection,
    )

    _assert_registered_variant_contract()
    source_snapshot = _verify_source_snapshot(args)
    configs = _bind_source_snapshot_config(
        _mapping(args.config, "config"), source_snapshot
    )
    code = dict(source_snapshot.stable_code_paths())
    environment = capture_runtime_environment()

    # No selection outcome capability exists in this command.  The order is
    # still explicit: variant history, full anchor, independent current, then
    # the outcome-free strategy producer.
    history = _verify_history_completion_triplet(
        args.history_complete_artifact,
        args.history_complete_receipt,
        args.history_complete_receipt_sha256,
    )
    full_anchor = _verify_history_completion_triplet(
        args.full_history_anchor_artifact,
        args.full_history_anchor_receipt,
        args.full_history_anchor_receipt_sha256,
    )
    full_variant, full_model, _full_run = _load_registered_backbone_contract(
        args.full_anchor_backbone_config,
        dataset=args.dataset,
    )
    if full_variant != "full":
        raise SystemExit("the current-only anchor model contract is not full")
    current = _verify_full_anchored_current(
        full_history_anchor=full_anchor,
        full_fit_receipt=args.full_anchor_fit_receipt,
        full_fit_receipt_sha256=args.full_anchor_fit_receipt_sha256,
        current_artifact=args.current_complete_artifact,
        current_receipt=args.current_complete_receipt,
        current_receipt_sha256=args.current_complete_receipt_sha256,
    )
    paths = _StrategyUpstreamPaths(
        history_artifact=args.history_complete_artifact,
        history_receipt=args.history_complete_receipt,
        history_receipt_sha256=args.history_complete_receipt_sha256,
        backbone_config=args.backbone_config,
        fit_receipt=args.fit_receipt,
        fit_receipt_sha256=args.fit_receipt_sha256,
        config_paths=configs,
    )
    variant, model_config, run_config, upstream = _verify_strategy_upstream(
        paths=paths,
        history_attestation=history,
        full_history_anchor=full_anchor,
        current_attestation=current,
        full_anchor_model_config=full_model,
        dataset=args.dataset,
        sidecar_dir=args.sidecar_dir,
        sidecar_manifest=args.sidecar_manifest,
    )
    completed = complete_strategy_selection(
        upstream=upstream,
        registered_variant=variant,
        model_config=model_config,
        run_config=run_config,
        private_output_root=args.private_output_dir,
        config_paths=configs,
        code_paths=code,
        environment=environment,
        device=_resolve_device(args.device),
    )
    summary = {
        "schema_version": "carma_causal_backbone_strategy_complete_cli_v1",
        "status": "outcome_free_strategy_probability_cache_complete",
        "dataset": history.dataset,
        "registered_variant": variant,
        "artifact_sha256": completed.artifact_sha256,
        "completion_receipt_sha256": completed.receipt_sha256,
        "production_run_claim_sha256": completed.production_run_claim_sha256,
        "policy_sha256": completed.policy_sha256,
        "variant_derived_from_model_config": True,
        "selection_label_file_accessed": False,
        "calibration_unseal_authorized": False,
        "internal_holdout_unseal_authorized": False,
        "external_test_unseal_authorized": False,
        "performance_metric_computed": False,
    }
    summary.update(_snapshot_summary_fields(source_snapshot))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _run_evaluate_model_selection(args) -> None:
    from hva_affect.causal_backbone_model_selection_evaluator import (
        SelectionSidecarSource,
        StrategyProductionInput,
        run_model_selection_reference_freeze,
    )

    _assert_registered_variant_contract()
    source_snapshot = _verify_source_snapshot(args)
    fit_receipts = _exact_variant_mapping(args.fit_receipt, "fit receipt")
    fit_receipt_hashes = _exact_variant_mapping(
        args.fit_receipt_sha256, "fit receipt SHA-256"
    )
    variant_configs = _exact_variant_named_mapping(
        args.variant_config, "variant config"
    )
    variant_configs = {
        variant: _bind_source_snapshot_config(configs, source_snapshot)
        for variant, configs in variant_configs.items()
    }
    backbone_configs = _exact_variant_mapping(
        args.backbone_config, "backbone config"
    )
    history_artifacts = _exact_variant_mapping(
        args.history_complete_artifact, "history artifact"
    )
    history_receipts = _exact_variant_mapping(
        args.history_completion_receipt, "history receipt"
    )
    history_receipt_hashes = _exact_variant_mapping(
        args.history_completion_receipt_sha256, "history receipt SHA-256"
    )
    strategy_artifacts = _exact_variant_mapping(
        args.strategy_complete_artifact, "strategy artifact"
    )
    strategy_receipts = _exact_variant_mapping(
        args.strategy_completion_receipt, "strategy receipt"
    )
    strategy_receipt_hashes = _exact_variant_mapping(
        args.strategy_completion_receipt_sha256, "strategy receipt SHA-256"
    )
    code = dict(source_snapshot.stable_code_paths())
    environment = capture_runtime_environment()

    # Derive all four names from the actual JSON model fields before accepting
    # any caller-keyed strategy bundle.
    models: dict[str, object] = {}
    for variant in _REGISTERED_VARIANTS:
        backbone = Path(backbone_configs[variant])
        configs = variant_configs[variant]
        derived, model, _run = _load_exact_variant_config_mapping(
            backbone_config=backbone,
            config_paths=configs,
            dataset=args.dataset,
        )
        if derived != variant:
            raise SystemExit(
                f"{variant} caller key differs from model-derived variant {derived}"
            )
        models[variant] = model

    # Complete all history attestations first, then independently re-attest the
    # unique full anchor and its one current-only completion.  No label loader is
    # reachable in any of these APIs.
    histories = {
        variant: _verify_history_completion_triplet(
            Path(history_artifacts[variant]),
            Path(history_receipts[variant]),
            str(history_receipt_hashes[variant]),
        )
        for variant in _REGISTERED_VARIANTS
    }
    full_anchor = _verify_history_completion_triplet(
        Path(history_artifacts["full"]),
        Path(history_receipts["full"]),
        str(history_receipt_hashes["full"]),
    )
    current = _verify_full_anchored_current(
        full_history_anchor=full_anchor,
        full_fit_receipt=Path(fit_receipts["full"]),
        full_fit_receipt_sha256=str(fit_receipt_hashes["full"]),
        current_artifact=args.current_complete_artifact,
        current_receipt=args.current_complete_receipt,
        current_receipt_sha256=args.current_complete_receipt_sha256,
    )

    strategies = {}
    for variant in _REGISTERED_VARIANTS:
        paths = _StrategyUpstreamPaths(
            history_artifact=Path(history_artifacts[variant]),
            history_receipt=Path(history_receipts[variant]),
            history_receipt_sha256=str(history_receipt_hashes[variant]),
            backbone_config=Path(backbone_configs[variant]),
            fit_receipt=Path(fit_receipts[variant]),
            fit_receipt_sha256=str(fit_receipt_hashes[variant]),
            config_paths=variant_configs[variant],
        )
        derived, _model, _run, upstream = _verify_strategy_upstream(
            paths=paths,
            history_attestation=histories[variant],
            full_history_anchor=full_anchor,
            current_attestation=current,
            full_anchor_model_config=models["full"],
            dataset=args.dataset,
            sidecar_dir=args.sidecar_dir,
            sidecar_manifest=args.sidecar_manifest,
        )
        if derived != variant:
            raise SystemExit("strategy upstream changed its model-derived variant")
        strategies[variant] = StrategyProductionInput(
            artifact_path=Path(strategy_artifacts[variant]),
            receipt_path=Path(strategy_receipts[variant]),
            expected_receipt_sha256=str(strategy_receipt_hashes[variant]),
            upstream=upstream,
            config_paths=variant_configs[variant],
            code_paths=code,
            environment=environment,
        )

    completed = run_model_selection_reference_freeze(
        strategies=strategies,
        selection_source=SelectionSidecarSource(
            dataset=args.dataset,
            sidecar_dir=args.sidecar_dir,
            manifest_path=args.sidecar_manifest,
            preflight_receipt_path=Path(fit_receipts["full"]),
            expected_preflight_receipt_sha256=str(fit_receipt_hashes["full"]),
            config_paths=variant_configs["full"],
            code_paths=code,
            environment=environment,
        ),
        confirmatory_analysis_path=args.confirmatory_analysis,
        private_output_root=args.private_output_dir,
        public_report_path=args.public_report,
    )
    summary = {
        "schema_version": "carma_model_selection_reference_freeze_cli_v1",
        "status": "aggregate_model_selection_reference_frozen",
        "dataset": args.dataset,
        "registered_variants": list(_REGISTERED_VARIANTS),
        "private_artifact_sha256": completed.private_artifact_sha256,
        "private_receipt_sha256": completed.private_receipt_sha256,
        "public_report_sha256": completed.public_report_sha256,
        "frozen_reference": completed.frozen_reference,
        "model_selection_gate_passed": completed.model_selection_gate_passed,
        "prospective_power": completed.prospective_power,
        "power_gate_passed": completed.power_gate_passed,
        "selection_label_access_limited_to_evaluator": True,
        "calibration_unseal_authorized": False,
        "internal_holdout_unseal_authorized": False,
        "external_test_unseal_authorized": False,
        "confirmatory_claim_authorized": False,
    }
    summary.update(_snapshot_summary_fields(source_snapshot))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "create-production-source-snapshot":
        _run_create_production_source_snapshot(args)
        return

    if args.command == "verify-production-source-snapshot":
        _run_verify_production_source_snapshot(args)
        return

    if args.command == "run-joint-model-selection-freeze":
        _run_joint_model_selection_freeze(args)
        return

    if args.command == "verify-joint-model-selection-freeze":
        _run_verify_joint_model_selection_freeze(args)
        return

    if args.command == "fit-preflight":
        source_snapshot = _verify_source_snapshot(args)
        result = run_fit_preflight(
            dataset=args.dataset,
            sidecar_dir=args.sidecar_dir,
            manifest_path=args.sidecar_manifest,
            receipt_path=args.receipt,
            config_paths=_bind_source_snapshot_config(
                _mapping(args.config, "config"), source_snapshot
            ),
            code_paths=dict(source_snapshot.stable_code_paths()),
        )
        summary = {
            "schema_version": result.receipt["schema_version"],
            "status": result.receipt["status"],
            "dataset": result.receipt["dataset"],
            "fit_rows": result.fit.rows,
            "receipt_sha256": result.receipt_sha256,
            "training_run": False,
            "performance_metric_computed": False,
        }
        summary.update(_snapshot_summary_fields(source_snapshot))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "strategy-complete-selection":
        _run_strategy_complete_selection(args)
        return

    if args.command == "evaluate-model-selection":
        _run_evaluate_model_selection(args)
        return

    from hva_affect.causal_backbone_evidence_stage_b import (
        FIT_ONLY_LINEAGE_SCHEMA,
        load_fit_only_lineage,
        load_fit_protocol_map,
        materialize_verified_fit_for_stage_b,
        validate_current_only_fit_files,
        write_fit_only_lineage,
        write_fit_protocol_map,
    )

    source_snapshot = _verify_source_snapshot(args)
    configs = _bind_source_snapshot_config(
        _mapping(args.config, "config"), source_snapshot
    )
    code = dict(source_snapshot.stable_code_paths())
    if args.command in {
        "current-only-fit",
        "current-only-complete-selection",
        "history-fit",
        "history-complete-selection",
    }:
        _verify_frozen_production_inputs(
            backbone_config=args.backbone_config,
            utility_config=(
                args.utility_config if args.command.startswith("history-") else None
            ),
            config_paths=configs,
            code_paths=code,
            source_snapshot=source_snapshot,
        )
    fit = materialize_verified_fit_for_stage_b(
        receipt_path=args.fit_receipt,
        expected_receipt_sha256=args.fit_receipt_sha256,
        dataset=args.dataset,
        sidecar_dir=args.sidecar_dir,
        manifest_path=args.sidecar_manifest,
        config_paths=configs,
        code_paths=code,
    )
    if args.command == "fit-lineage-create":
        fit_map_destination = args.fit_map.resolve()
        lineage_destination = args.fit_lineage.resolve()
        if fit_map_destination == lineage_destination:
            raise SystemExit("fit map and fit lineage must use different output files")
        if args.fit_map.exists() or args.fit_lineage.exists():
            raise SystemExit("fit lineage create outputs must not already exist")
        fit_map = write_fit_protocol_map(
            fit,
            receipt_path=args.fit_receipt,
            expected_receipt_sha256=args.fit_receipt_sha256,
            output_path=args.fit_map,
        )
        lineage = write_fit_only_lineage(
            fit,
            fit_map=fit_map,
            receipt_path=args.fit_receipt,
            expected_receipt_sha256=args.fit_receipt_sha256,
            output_path=args.fit_lineage,
        )
        summary = {
            "schema_version": FIT_ONLY_LINEAGE_SCHEMA,
            "status": "fit_only_alignment_lineage_created",
            "dataset": lineage.dataset,
            "fit_rows": lineage.rows,
            "fit_map_sha256": fit_map.artifact_sha256,
            "fit_lineage_sha256": lineage.artifact_sha256,
            "selection_payload_opened": False,
            "history_producer_required": False,
            "training_run": False,
            "performance_metric_computed": False,
        }
        summary.update(_snapshot_summary_fields(source_snapshot))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    fit_map = load_fit_protocol_map(
        args.fit_map,
        receipt_path=args.fit_receipt,
        expected_receipt_sha256=args.fit_receipt_sha256,
    )
    if args.command == "fit-lineage-validate":
        lineage = load_fit_only_lineage(
            args.fit_lineage,
            fit=fit,
            fit_map=fit_map,
            receipt_path=args.fit_receipt,
            expected_receipt_sha256=args.fit_receipt_sha256,
        )
        summary = {
            "schema_version": FIT_ONLY_LINEAGE_SCHEMA,
            "status": "fit_only_alignment_lineage_valid",
            "dataset": lineage.dataset,
            "fit_rows": lineage.rows,
            "fit_map_sha256": fit_map.artifact_sha256,
            "fit_lineage_sha256": lineage.artifact_sha256,
            "selection_payload_opened": False,
            "history_producer_required": False,
            "training_run": False,
            "performance_metric_computed": False,
        }
        summary.update(_snapshot_summary_fields(source_snapshot))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command in {"current-only-fit", "current-only-complete-selection"}:
        lineage = load_fit_only_lineage(
            args.fit_lineage,
            fit=fit,
            fit_map=fit_map,
            receipt_path=args.fit_receipt,
            expected_receipt_sha256=args.fit_receipt_sha256,
        )

    if args.command in {"history-fit", "history-complete-selection"}:
        _run_history_command(
            args=args,
            fit=fit,
            fit_map=fit_map,
            configs=configs,
            code=code,
            source_snapshot=source_snapshot,
        )
        return

    if args.command not in {
        "current-only-fit",
        "current-only-complete-selection",
    }:
        raise SystemExit("unsupported command")

    from hva_affect.causal_backbone_current_only_pipeline import (
        claim_or_resume_current_only_private_root,
        complete_current_only_selection_probabilities,
        current_only_private_paths,
        current_only_production_claim_sha256,
        load_attested_history_fit_alignment_view,
        load_attested_history_producer_alignment_view,
        produce_current_only_fit_with_real_trainer,
        verify_current_only_fit_for_completion,
    )
    from hva_affect.emotiontalk_causal_backbone_runner import _runtime_environment

    model_config, run_config = _load_backbone_config(args.backbone_config)
    device = _resolve_device(args.device)
    config_file_sha = _sha256(args.backbone_config)
    model_sha = _canonical_sha256(
        {"file_sha256": config_file_sha, "resolved_model": asdict(model_config)}
    )
    run_sha = _canonical_sha256(
        {"file_sha256": config_file_sha, "resolved_runner": asdict(run_config)}
    )
    runtime = _runtime_environment(device)
    runtime_sha = _canonical_sha256(runtime)
    preflight_runtime_sha = _canonical_sha256(capture_runtime_environment())
    source_code_sha = _production_code_sha256(code)
    production_claim = current_only_production_claim_sha256(
        fit=fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        model_config=model_config,
        run_config=run_config,
        model_config_sha256=model_sha,
        run_config_sha256=run_sha,
        source_code_sha256=source_code_sha,
        runtime_environment_sha256=runtime_sha,
    )

    if args.command == "current-only-fit":
        private_root = claim_or_resume_current_only_private_root(
            args.private_output_dir,
            production_claim_sha256=production_claim,
            allow_resume=bool(args.resume),
        )
    else:
        private_root = args.private_output_dir.resolve(strict=True)
        if not private_root.is_dir():
            raise SystemExit(
                "--private-output-dir must be the current-only-fit directory"
            )
    current_paths = current_only_private_paths(private_root)

    if args.command == "current-only-fit":
        produced = produce_current_only_fit_with_real_trainer(
            fit=fit,
            fit_map=fit_map,
            lineage=lineage,
            fit_preflight_receipt_path=args.fit_receipt,
            expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
            checkpoint_root=current_paths["checkpoint"],
            artifact_path=current_paths["fit_artifact"],
            producer_receipt_path=current_paths["fit_receipt"],
            model_config=model_config,
            run_config=run_config,
            model_config_sha256=model_sha,
            run_config_sha256=run_sha,
            source_code_sha256=source_code_sha,
            runtime_environment_sha256=runtime_sha,
            production_run_claim_sha256=production_claim,
            allow_checkpoint_resume=bool(args.resume),
            device=device,
        )
        summary = validate_current_only_fit_files(
            artifact_path=current_paths["fit_artifact"],
            fit=fit,
            fit_map=fit_map,
            lineage=lineage,
            fit_preflight_receipt_path=args.fit_receipt,
            expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
            checkpoint_root=current_paths["checkpoint"],
            outer_folds=run_config.outer_folds,
        )
        summary["fit_producer_receipt_sha256"] = produced.receipt_sha256
        summary["selection_payload_consumed"] = False
        summary["performance_metric_computed"] = False
        summary.update(_snapshot_summary_fields(source_snapshot))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command != "current-only-complete-selection":
        raise SystemExit("unsupported command")
    history_attestation = _attest_history_completion_for_current_only(
        artifact_path=args.history_complete_artifact,
        completion_receipt_path=args.history_completion_receipt,
        expected_completion_receipt_sha256=(
            args.history_completion_receipt_sha256
        ),
        dataset=args.dataset,
        fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        config_paths=configs,
        code_paths=code,
        model_config=model_config,
        run_config=run_config,
        runtime_environment_sha256=preflight_runtime_sha,
        execution_environment_sha256=runtime_sha,
    )
    fit_producer = load_attested_history_fit_alignment_view(
        history_attestation,
        fit_preflight_receipt_path=args.fit_receipt,
        expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
    )
    fit_state = verify_current_only_fit_for_completion(
        fit_artifact_path=current_paths["fit_artifact"],
        fit_producer_receipt_path=current_paths["fit_receipt"],
        expected_fit_producer_receipt_sha256=args.fit_producer_receipt_sha256,
        checkpoint_root=current_paths["checkpoint"],
        fit=fit,
        fit_map=fit_map,
        lineage=lineage,
        producer=fit_producer,
        fit_preflight_receipt_path=args.fit_receipt,
        expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        outer_folds=run_config.outer_folds,
        model_config=model_config,
        run_config=run_config,
        model_config_sha256=model_sha,
        run_config_sha256=run_sha,
        source_code_sha256=source_code_sha,
        runtime_environment_sha256=runtime_sha,
        production_run_claim_sha256=production_claim,
        device=device,
    )
    # Selection is opened only after the fit artifact and every checkpoint have
    # passed the semantic complete-checkpoint gate.  This API has no label path.
    selection = materialize_selection_features_after_receipt(
        receipt_path=args.fit_receipt,
        expected_receipt_sha256=args.fit_receipt_sha256,
        dataset=args.dataset,
        sidecar_dir=args.sidecar_dir,
        manifest_path=args.sidecar_manifest,
        config_paths=configs,
        code_paths=code,
    )
    alignment = load_attested_history_producer_alignment_view(
        history_attestation,
        fit_producer=fit_producer,
    )
    completed = complete_current_only_selection_probabilities(
        fit=fit,
        selection=selection,
        fit_map=fit_map,
        lineage=lineage,
        fit_producer=fit_producer,
        producer=alignment,
        fit_state=fit_state,
        fit_preflight_receipt_path=args.fit_receipt,
        expected_fit_preflight_receipt_sha256=args.fit_receipt_sha256,
        checkpoint_root=current_paths["checkpoint"],
        model_config=model_config,
        run_config=run_config,
        model_config_sha256=model_sha,
        run_config_sha256=run_sha,
        source_code_sha256=source_code_sha,
        runtime_environment_sha256=runtime_sha,
        production_run_claim_sha256=production_claim,
        selection_feature_sha256=fit_state.selection_feature_sha256,
        artifact_path=current_paths["complete_artifact"],
        completion_receipt_path=current_paths["complete_receipt"],
        device=device,
    )
    summary = {
        "schema_version": "carma_independent_current_only_private_v1",
        "status": "strategy_consumable_current_only_cache_complete_not_performance_evidence",
        "dataset": alignment.dataset,
        "seed_count": len(alignment.seeds),
        "fit_query_count": len(alignment.fit_query_indices),
        "selection_query_count": len(alignment.selection_query_indices),
        "artifact_sha256": completed.artifact_sha256,
        "completion_receipt_sha256": completed.receipt_sha256,
        "complete_checkpoint_only": True,
        "selection_label_materialized": False,
        "selection_label_file_accessed": False,
        "performance_metric_computed": False,
    }
    summary.update(_snapshot_summary_fields(source_snapshot))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
