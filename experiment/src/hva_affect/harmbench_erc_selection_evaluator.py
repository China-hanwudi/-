"""Irreversible aggregate-only HarmBench-ERC selection evaluator.

This module is the sole production bridge across the outcome boundary.  It
starts from an empty, repository-external fixed output root, publishes the
outcome-free prelabel state, durably starts exactly one attempt, consumes the
attempt's exact two label-access tickets in frozen dataset order, computes the
frozen aggregate statistics, and publishes one canonical final JSON file.

There is deliberately no resume, retry, force, cleanup, or synthetic
publication entry point.  Once the attempt marker exists, every failure is a
terminal exploratory outcome.  The final artifact contains only a validated
aggregate report and its public state bindings; label/prediction lineage,
paths, row identifiers, probabilities, and resampling draws remain private.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import harmbench_erc_selection_labels as _labels
from . import harmbench_erc_selection_prelabel as _prelabel
from . import harmbench_erc_selection_statistics as _statistics
from .harmbench_erc_prediction_artifact import LoadedPredictionArtifact
from .harmbench_erc_protocol_v2 import (
    EXPECTED_SELECTION_DATASETS,
    PROTOCOL_V2_CANONICAL_SHA256,
    ProtocolV2Contract,
)
from .harmbench_erc_selection_labels import SelectionLabelManifestMetadata


FINAL_EVALUATION_SCHEMA = "harmbench_erc_selection_evaluation_final_v1"
FINAL_EVALUATION_STATE = "final_write_once_fsync_complete"
FINAL_EVALUATION_FILENAME = _prelabel.FINAL_OUTPUT_FILENAME
MAX_FINAL_EVALUATION_BYTES = 16 * 1024 * 1024

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FINAL_SEAL = object()
_FIXED_STATE_FILENAMES = (
    _prelabel.PRELABEL_BUNDLE_FILENAME,
    _prelabel.PRELABEL_RECEIPT_FILENAME,
    _prelabel.ATTEMPT_MARKER_FILENAME,
    FINAL_EVALUATION_FILENAME,
)
_FINAL_KEYS = {
    "schema_version",
    "state",
    "irreversible",
    "resume",
    "protocol_canonical_sha256",
    "attempt_marker_sha256",
    "permanently_exploratory",
    "confirmatory",
    "statistics_sha256",
    "validated_statistics_report",
}
_MARKER_KEYS = {
    "schema_version",
    "state",
    "irreversible",
    "resume_or_rerun_permitted",
    "crash_before_final_is_terminal",
    "protocol_canonical_sha256",
    "prelabel_bundle_filename",
    "prelabel_bundle_file_sha256",
    "prelabel_receipt_filename",
    "prelabel_receipt_file_sha256",
    "attempt_nonce",
    "evaluation_status",
    "next_permitted_operation",
    "label_npz_access_occurred",
}
_MARKER_STATUS_KEYS = {
    "status",
    "confirmatory_claim",
    "calibration",
    "internal_holdout",
    "validation",
    "official_test",
    "selection_labels_opened",
    "row_metrics_computed",
    "attempt_started",
}
_FORBIDDEN_FINAL_KEY_PARTS = (
    "artifact",
    "manifest",
    "input_digest",
    "input_sha",
    "private",
    "path",
    "label",
    "probabilit",
    "row_id",
    "query_id",
    "group_token",
    "dialogue_id",
    "speaker",
    "raw_text",
    "embedding",
    "per_seed",
    "per_fold",
    "seed_draw",
    "cluster_draw",
    "resampling_draw",
)
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class HarmBenchSelectionEvaluatorError(ValueError):
    """Raised when the irreversible evaluator/final state is not exact."""


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HarmBenchSelectionEvaluatorError(
            f"{name} must be one lowercase SHA-256"
        )
    return value


def _canonical_json_payload(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HarmBenchSelectionEvaluatorError(
            f"final state is not canonical JSON data: {error}"
        ) from error


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_payload(value) + b"\n"


def _plain_json_snapshot(value: object, *, name: str) -> dict[str, object]:
    encoded = _canonical_json_payload(value)
    try:
        result = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # pragma: no cover
        raise HarmBenchSelectionEvaluatorError(
            f"{name} did not survive canonical JSON snapshotting"
        ) from error
    if not isinstance(result, dict):
        raise HarmBenchSelectionEvaluatorError(f"{name} root must be an object")
    if _canonical_json_payload(result) != encoded:
        raise HarmBenchSelectionEvaluatorError(
            f"{name} changed during canonical JSON snapshotting"
        )
    return result


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _exact_mapping(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HarmBenchSelectionEvaluatorError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise HarmBenchSelectionEvaluatorError(
            f"{name} must use only string keys"
        )
    observed = set(value)
    if observed != expected:
        raise HarmBenchSelectionEvaluatorError(
            f"{name} schema changed: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _scan_final_privacy(value: object, *, path: str = "$") -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise HarmBenchSelectionEvaluatorError(
                f"non-finite final value at {path}"
            )
        return
    if isinstance(value, str):
        if _WINDOWS_PATH.match(value) or value.startswith(("/", "file://")) or "\\" in value:
            raise HarmBenchSelectionEvaluatorError(
                f"path-like private value in final output at {path}"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise HarmBenchSelectionEvaluatorError(
                    f"non-string final key at {path}"
                )
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_FINAL_KEY_PARTS):
                raise HarmBenchSelectionEvaluatorError(
                    f"privacy-forbidden final key at {path}.{key}"
                )
            _scan_final_privacy(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise HarmBenchSelectionEvaluatorError(
                f"oversized final sequence at {path}"
            )
        for index, child in enumerate(value):
            _scan_final_privacy(child, path=f"{path}[{index}]")
        return
    raise HarmBenchSelectionEvaluatorError(
        f"unsupported final value at {path}: {type(value).__name__}"
    )


def _validated_statistics_snapshot(report: object) -> tuple[dict[str, object], str]:
    plain = _plain_json_snapshot(report, name="selection statistics report")
    try:
        _statistics.validate_selection_statistics_report(plain)
    except (TypeError, ValueError) as error:
        raise HarmBenchSelectionEvaluatorError(
            "selection statistics report failed its frozen validator"
        ) from error
    encoded = _canonical_json_payload(plain)
    return plain, hashlib.sha256(encoded).hexdigest()


def validate_final_selection_evaluation(payload: object) -> None:
    """Validate the exact aggregate-only final wrapper and embedded report."""

    root = _exact_mapping(payload, _FINAL_KEYS, name="final evaluation")
    if (
        root["schema_version"] != FINAL_EVALUATION_SCHEMA
        or root["state"] != FINAL_EVALUATION_STATE
        or root["irreversible"] is not True
        or root["resume"] is not False
        or root["permanently_exploratory"] is not True
        or root["confirmatory"] is not False
        or root["protocol_canonical_sha256"] != PROTOCOL_V2_CANONICAL_SHA256
    ):
        raise HarmBenchSelectionEvaluatorError(
            "final irreversible/exploratory state changed"
        )
    _sha256(root["attempt_marker_sha256"], name="attempt_marker_sha256")
    expected_statistics_sha = _sha256(
        root["statistics_sha256"], name="statistics_sha256"
    )
    report, observed_statistics_sha = _validated_statistics_snapshot(
        root["validated_statistics_report"]
    )
    if (
        observed_statistics_sha != expected_statistics_sha
        or report.get("protocol_canonical_sha256")
        != root["protocol_canonical_sha256"]
        or report.get("selection_result_status")
        != _statistics.EXPLORATORY_STATUS
        or report.get("confirmatory_claim") is not False
    ):
        raise HarmBenchSelectionEvaluatorError(
            "final statistics hash/status binding changed"
        )
    _scan_final_privacy(payload)


def _preflight_empty_root(
    private_root: str | Path,
) -> tuple[Path, os.stat_result]:
    """Check every fixed state file before touching any typed research input."""

    try:
        root = _prelabel._validate_output_root(private_root)  # noqa: SLF001
        root_identity = _prelabel._plain_directory_stat(  # noqa: SLF001
            root, name="selection evaluator output root"
        )
        present = tuple(
            filename
            for filename in _FIXED_STATE_FILENAMES
            if os.path.lexists(
                _prelabel._fixed_path(root, filename)  # noqa: SLF001
            )
        )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
    except (TypeError, ValueError, OSError) as error:
        raise HarmBenchSelectionEvaluatorError(
            "selection evaluator output root failed preflight"
        ) from error
    if present:
        raise HarmBenchSelectionEvaluatorError(
            "fixed evaluator state already exists; resume/retry/force/cleanup "
            f"is forbidden: {', '.join(present)}"
        )
    return root, root_identity


def _birth_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(metadata.st_size),
    )


def _publish_final_once(
    *,
    root: Path,
    root_identity: os.stat_result,
    payload: Mapping[str, object],
) -> str:
    validate_final_selection_evaluation(payload)
    encoded = _canonical_json_bytes(payload)
    if len(encoded) > MAX_FINAL_EVALUATION_BYTES:
        raise HarmBenchSelectionEvaluatorError(
            "final evaluation exceeds its fixed byte budget"
        )
    destination = _prelabel._fixed_path(  # noqa: SLF001
        root, FINAL_EVALUATION_FILENAME
    )
    if os.path.lexists(destination):
        raise FileExistsError("write-once final evaluation already exists")
    temporary: Path | None = None
    try:
        temporary = _prelabel._temporary_bytes(  # noqa: SLF001
            root, destination, encoded
        )
        temporary_identity = _birth_identity(
            _prelabel._plain_file_stat(  # noqa: SLF001
                temporary, name="temporary final evaluation"
            )
        )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
        if os.path.lexists(destination):
            raise FileExistsError("write-once final evaluation already exists")
        _prelabel._publish_once(temporary, destination)  # noqa: SLF001
        temporary = None
        published_identity = _prelabel._plain_file_stat(  # noqa: SLF001
            destination, name="published final evaluation"
        )
        if _birth_identity(published_identity) != temporary_identity:
            raise HarmBenchSelectionEvaluatorError(
                "final destination is not the exact fsynced temporary file"
            )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
        _prelabel._sync_directory(root)  # noqa: SLF001
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
        final_sha = hashlib.sha256(encoded).hexdigest()
        observed = _prelabel._decode_canonical_file(  # noqa: SLF001
            destination,
            expected_sha256=final_sha,
            maximum_bytes=MAX_FINAL_EVALUATION_BYTES,
            name="final selection evaluation",
        )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
        if observed != dict(payload):
            raise HarmBenchSelectionEvaluatorError(
                "final evaluation changed during canonical readback"
            )
        validate_final_selection_evaluation(observed)
        return final_sha
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _validate_terminal_marker(
    marker: object, *, protocol_canonical_sha256: str
) -> Mapping[str, object]:
    root = _exact_mapping(marker, _MARKER_KEYS, name="attempt marker")
    if (
        root["schema_version"] != _prelabel.ATTEMPT_MARKER_SCHEMA
        or root["state"] != "attempt_marker_write_once_fsync_complete"
        or root["irreversible"] is not True
        or root["resume_or_rerun_permitted"] is not False
        or root["crash_before_final_is_terminal"] is not True
        or root["protocol_canonical_sha256"] != protocol_canonical_sha256
        or root["prelabel_bundle_filename"]
        != _prelabel.PRELABEL_BUNDLE_FILENAME
        or root["prelabel_receipt_filename"]
        != _prelabel.PRELABEL_RECEIPT_FILENAME
        or root["label_npz_access_occurred"] is not False
    ):
        raise HarmBenchSelectionEvaluatorError(
            "terminal attempt marker state changed"
        )
    for key in (
        "prelabel_bundle_file_sha256",
        "prelabel_receipt_file_sha256",
        "attempt_nonce",
    ):
        _sha256(root[key], name=key)
    status = _exact_mapping(
        root["evaluation_status"],
        _MARKER_STATUS_KEYS,
        name="attempt marker evaluation status",
    )
    if (
        status["status"] != _prelabel.EXPLORATORY_STATUS
        or status["confirmatory_claim"] is not False
        or status["calibration"] is not False
        or status["internal_holdout"] is not False
        or status["validation"] is not False
        or status["official_test"] is not False
        or status["selection_labels_opened"] is not False
        or status["row_metrics_computed"] is not False
        or status["attempt_started"] is not True
    ):
        raise HarmBenchSelectionEvaluatorError(
            "attempt marker exploratory status changed"
        )
    return root


@dataclass(frozen=True)
class VerifiedSelectionEvaluationFinal:
    """Aggregate-only terminal capability minted by the final-file verifier."""

    protocol_canonical_sha256: str
    attempt_marker_sha256: str
    statistics_sha256: str
    final_file_sha256: str
    payload: Mapping[str, object] = field(repr=False, compare=False)
    _private_root: Path = field(repr=False, compare=False)
    _final_path: Path = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _FINAL_SEAL:
            raise HarmBenchSelectionEvaluatorError(
                "final capabilities can only be minted by the terminal verifier"
            )
        for name in (
            "protocol_canonical_sha256",
            "attempt_marker_sha256",
            "statistics_sha256",
            "final_file_sha256",
        ):
            _sha256(getattr(self, name), name=name)


def verify_terminal_selection_evaluation(
    *, private_root: str | Path, expected_final_sha256: str
) -> VerifiedSelectionEvaluationFinal:
    """Verify an already-terminal final state without attempt revalidation.

    The caller must supply the trusted final-file digest obtained at the
    write-once publication boundary.  This prevents a self-consistent later
    replacement of the marker/final pair from becoming a new trust root.
    """

    expected_final = _sha256(
        expected_final_sha256, name="expected_final_sha256"
    )
    try:
        root = _prelabel._validate_output_root(private_root)  # noqa: SLF001
        root_identity = _prelabel._plain_directory_stat(  # noqa: SLF001
            root, name="terminal selection evaluator root"
        )
        paths = {
            filename: _prelabel._fixed_path(root, filename)  # noqa: SLF001
            for filename in _FIXED_STATE_FILENAMES
        }
        for filename, path in paths.items():
            _prelabel._plain_file_stat(  # noqa: SLF001
                path, name=f"terminal fixed file {filename}"
            )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001

        final_payload = _prelabel._decode_canonical_file(  # noqa: SLF001
            paths[FINAL_EVALUATION_FILENAME],
            expected_sha256=expected_final,
            maximum_bytes=MAX_FINAL_EVALUATION_BYTES,
            name="final selection evaluation",
        )
        validate_final_selection_evaluation(final_payload)
        marker_sha = str(final_payload["attempt_marker_sha256"])
        marker = _prelabel._decode_canonical_file(  # noqa: SLF001
            paths[_prelabel.ATTEMPT_MARKER_FILENAME],
            expected_sha256=marker_sha,
            maximum_bytes=_prelabel.MAX_ATTEMPT_MARKER_BYTES,
            name="attempt marker",
        )
        live_marker = _validate_terminal_marker(
            marker,
            protocol_canonical_sha256=str(
                final_payload["protocol_canonical_sha256"]
            ),
        )
        bundle_sha = str(live_marker["prelabel_bundle_file_sha256"])
        receipt_sha = str(live_marker["prelabel_receipt_file_sha256"])
        bundle = _prelabel._decode_canonical_file(  # noqa: SLF001
            paths[_prelabel.PRELABEL_BUNDLE_FILENAME],
            expected_sha256=bundle_sha,
            maximum_bytes=_prelabel.MAX_PRELABEL_BUNDLE_BYTES,
            name="prelabel bundle",
        )
        receipt = _prelabel._decode_canonical_file(  # noqa: SLF001
            paths[_prelabel.PRELABEL_RECEIPT_FILENAME],
            expected_sha256=receipt_sha,
            maximum_bytes=_prelabel.MAX_PRELABEL_RECEIPT_BYTES,
            name="prelabel receipt",
        )
        if (
            bundle.get("schema_version") != _prelabel.PRELABEL_BUNDLE_SCHEMA
            or not isinstance(bundle.get("protocol"), Mapping)
            or bundle["protocol"].get("canonical_sha256")
            != final_payload["protocol_canonical_sha256"]
            or receipt != _prelabel._receipt_for_bundle(bundle, bundle_sha)  # noqa: SLF001
        ):
            raise HarmBenchSelectionEvaluatorError(
                "terminal prelabel/receipt/marker binding changed"
            )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
    except HarmBenchSelectionEvaluatorError:
        raise
    except (TypeError, ValueError, OSError, KeyError, AssertionError) as error:
        raise HarmBenchSelectionEvaluatorError(
            "terminal selection evaluation verification failed"
        ) from error

    return VerifiedSelectionEvaluationFinal(
        protocol_canonical_sha256=str(
            final_payload["protocol_canonical_sha256"]
        ),
        attempt_marker_sha256=marker_sha,
        statistics_sha256=str(final_payload["statistics_sha256"]),
        final_file_sha256=expected_final,
        payload=_deep_freeze(final_payload),
        _private_root=root,
        _final_path=paths[FINAL_EVALUATION_FILENAME],
        _seal=_FINAL_SEAL,
    )


def evaluate_and_publish_selection_once(
    *,
    private_root: str | Path,
    protocol: ProtocolV2Contract,
    prediction_artifacts: Sequence[LoadedPredictionArtifact],
    label_manifests: Sequence[SelectionLabelManifestMetadata],
) -> VerifiedSelectionEvaluationFinal:
    """Run and publish the exact irreversible selection evaluation once."""

    # This is intentionally the first operation.  In particular, no protocol,
    # prediction, or label-manifest attribute may be inspected before all four
    # fixed state names have been proven absent.
    root, root_identity = _preflight_empty_root(private_root)

    prelabel = _prelabel.write_selection_prelabel_bundle_once(
        private_root=root,
        protocol=protocol,
        prediction_artifacts=prediction_artifacts,
        label_manifests=label_manifests,
    )
    _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001
    attempt = _prelabel.start_selection_evaluation_attempt(prelabel)

    try:
        tickets = _prelabel._issue_attempt_bound_label_access_tickets(  # noqa: SLF001
            attempt
        )
        if (
            type(tickets) is not tuple
            or len(tickets) != len(EXPECTED_SELECTION_DATASETS)
            or tuple(ticket.dataset_id for ticket in tickets)
            != tuple(EXPECTED_SELECTION_DATASETS)
        ):
            raise HarmBenchSelectionEvaluatorError(
                "attempt did not issue the exact ordered two-ticket suite"
            )

        # A comprehension/parallel map is intentionally forbidden here.  The
        # first ticket must finish irreversibly before the second is consumed.
        activated_labels: list[_labels.ActivatedSelectionLabelCapability] = []
        for ticket in tickets:
            activated_labels.append(
                _labels._activate_selection_labels_from_attempt_ticket(  # noqa: SLF001
                    ticket
                )
            )
        activated = tuple(activated_labels)
        _prelabel._validate_activated_label_suite_for_attempt(  # noqa: SLF001
            attempt, activated
        )

        joint_inputs = _statistics.load_joint_selection_evaluation_inputs(
            attempt, activated
        )
        report = _statistics.evaluate_selection_statistics(joint_inputs)
        report_plain, statistics_sha = _validated_statistics_snapshot(report)

        # This is the final live typed-input/attempt check.  It must happen
        # before final bytes are formed; the post-publication verifier below is
        # intentionally file-only and never calls this attempt revalidator.
        live_attempt = _prelabel._revalidate_attempt_started_capability(  # noqa: SLF001
            attempt
        )
        _prelabel._validate_activated_label_suite_for_attempt(  # noqa: SLF001
            live_attempt, activated
        )
        _prelabel._assert_root_identity(root, root_identity)  # noqa: SLF001

        final_payload: dict[str, object] = {
            "schema_version": FINAL_EVALUATION_SCHEMA,
            "state": FINAL_EVALUATION_STATE,
            "irreversible": True,
            "resume": False,
            "protocol_canonical_sha256": (
                live_attempt.protocol_canonical_sha256
            ),
            "attempt_marker_sha256": live_attempt.marker_file_sha256,
            "permanently_exploratory": True,
            "confirmatory": False,
            "statistics_sha256": statistics_sha,
            "validated_statistics_report": report_plain,
        }
        validate_final_selection_evaluation(final_payload)
        final_sha = _publish_final_once(
            root=root,
            root_identity=root_identity,
            payload=final_payload,
        )
    except BaseException:
        # Never clean or roll back any post-marker state.  The on-disk marker
        # is the durable terminal record even if this process is interrupted.
        _prelabel._mark_attempt_terminal_failure(  # noqa: SLF001
            attempt, "selection_evaluator_post_marker"
        )
        raise

    # A final file makes attempt revalidation invalid by design.  The terminal
    # verifier follows only fixed files and the trusted digest from publication.
    return verify_terminal_selection_evaluation(
        private_root=root,
        expected_final_sha256=final_sha,
    )


__all__ = [
    "FINAL_EVALUATION_FILENAME",
    "FINAL_EVALUATION_SCHEMA",
    "FINAL_EVALUATION_STATE",
    "HarmBenchSelectionEvaluatorError",
    "MAX_FINAL_EVALUATION_BYTES",
    "VerifiedSelectionEvaluationFinal",
    "evaluate_and_publish_selection_once",
    "validate_final_selection_evaluation",
    "verify_terminal_selection_evaluation",
]
