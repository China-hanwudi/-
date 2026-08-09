"""Exact aggregate-only synthetic public artifact for HarmBench-ERC."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

from .harmbench_erc_contract import (
    EXPECTED_SYNTHETIC_BOOTSTRAP_REPLICATES,
    EXPECTED_SYNTHETIC_BOOTSTRAP_SEED,
)
from .harmbench_erc_inference import CELL_ENDPOINTS, PINNED_DEVELOPMENT_PROTOCOL_SHA256
from .harmbench_erc_metrics import (
    DEFAULT_HARM_THRESHOLDS,
    DEFAULT_TAIL_ALPHA,
    NLL_EPSILON,
    HarmBenchMetricError,
    ensure_finite_public_tree,
)


class HarmBenchPublicError(ValueError):
    """Raised when a public artifact violates its exact schema."""


SYNTHETIC_PUBLIC_SCHEMA = "harmbench_erc_synthetic_public_v1"
PROTOCOL_ID = "harmbench_erc_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ROOT_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "protocol_sha256",
    "dataset_id",
    "data_role",
    "model_roster",
    "strategy_roster",
    "cell",
    "contrast",
    "contract_checks",
    "public_artifact_policy",
    "stage_authorization",
}
CELL_KEYS = {
    "alignment_contract",
    "metric_contract",
    "inference_contract",
    "point",
    "bootstrap",
}
CONTRAST_KEYS = {
    "alignment_contract",
    "metric_contract",
    "contrast_direction",
    "paired_on",
    "point",
    "bootstrap",
}
BOOTSTRAP_SUMMARY_KEYS = {
    "bootstrap_mean",
    "ci95_low",
    "ci95_high",
    "finite_replicates",
    "finite_fraction",
    "minimum_finite_fraction",
    "minimum_finite_fraction_gate_applicable",
}
RATE_ENDPOINTS = {
    "coverage",
    "population_harm_rate_gt_0",
    "population_harm_rate_gt_0_05",
    "conditional_harm_rate_gt_0",
    "conditional_harm_rate_gt_0_05",
    "eligible_break_rate",
    "eligible_rescue_rate",
    "selected_break_rate",
    "selected_rescue_rate",
}


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HarmBenchPublicError(f"{name} must be an object")
    return value


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, object]:
    mapping = _mapping(value, name=name)
    observed = {str(key) for key in mapping}
    if observed != expected:
        raise HarmBenchPublicError(
            f"{name} schema changed: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )
    return mapping


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarmBenchPublicError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HarmBenchPublicError(f"{name} must be finite")
    return result


def _exact_integer(
    value: object, *, name: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarmBenchPublicError(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise HarmBenchPublicError(f"{name} is below its minimum")
    if maximum is not None and value > maximum:
        raise HarmBenchPublicError(f"{name} is above its maximum")
    return value


def _optional_number(value: object, *, name: str) -> None:
    if value is not None:
        _finite_number(value, name=name)


def _validate_alignment(value: object, *, name: str) -> None:
    alignment = _exact_keys(
        value,
        {"dataset_id", "alignment_sha256", "bootstrap_plan_sha256"},
        name=name,
    )
    if alignment["dataset_id"] != "synthetic_dialogues":
        raise HarmBenchPublicError(f"{name} dataset changed")
    for key in ("alignment_sha256", "bootstrap_plan_sha256"):
        if not isinstance(alignment[key], str) or not SHA256_PATTERN.fullmatch(alignment[key]):
            raise HarmBenchPublicError(f"{name}.{key} is malformed")


def _validate_metric_contract(value: object, *, name: str) -> None:
    metric = _exact_keys(
        value,
        {"nll_probability_floor", "harm_thresholds_nats", "tail_alpha"},
        name=name,
    )
    if metric["nll_probability_floor"] != NLL_EPSILON:
        raise HarmBenchPublicError(f"{name} NLL floor changed")
    if metric["harm_thresholds_nats"] != list(DEFAULT_HARM_THRESHOLDS):
        raise HarmBenchPublicError(f"{name} harm thresholds changed")
    if metric["tail_alpha"] != DEFAULT_TAIL_ALPHA:
        raise HarmBenchPublicError(f"{name} tail alpha changed")


def _validate_endpoint_block(
    value: object, *, name: str, contrast: bool
) -> Mapping[str, object]:
    block = _exact_keys(value, set(CELL_ENDPOINTS), name=name)
    for endpoint in CELL_ENDPOINTS:
        _optional_number(block[endpoint], name=f"{name}.{endpoint}")
        if block[endpoint] is not None and endpoint in RATE_ENDPOINTS:
            number = float(block[endpoint])
            lower, upper = (-1.0, 1.0) if contrast else (0.0, 1.0)
            if not lower <= number <= upper:
                raise HarmBenchPublicError(f"{name}.{endpoint} is outside its range")
    if all(block[endpoint] is None for endpoint in CELL_ENDPOINTS):
        raise HarmBenchPublicError(f"{name} cannot be entirely null in a complete report")
    return block


def _validate_bootstrap(
    value: object,
    *,
    name: str,
    point: Mapping[str, object],
    replicates: int,
    contrast: bool,
) -> None:
    block = _exact_keys(value, set(CELL_ENDPOINTS), name=name)
    for endpoint in CELL_ENDPOINTS:
        summary = _exact_keys(
            block[endpoint], BOOTSTRAP_SUMMARY_KEYS, name=f"{name}.{endpoint}"
        )
        for key in ("bootstrap_mean", "ci95_low", "ci95_high"):
            _optional_number(summary[key], name=f"{name}.{endpoint}.{key}")
        finite_replicates = _exact_integer(
            summary["finite_replicates"],
            name=f"{name}.{endpoint}.finite_replicates",
            minimum=0,
            maximum=replicates,
        )
        fraction = _finite_number(
            summary["finite_fraction"], name=f"{name}.{endpoint}.finite_fraction"
        )
        minimum = _finite_number(
            summary["minimum_finite_fraction"],
            name=f"{name}.{endpoint}.minimum_finite_fraction",
        )
        expected_fraction = finite_replicates / replicates
        if (
            not 0.0 <= fraction <= 1.0
            or not math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-12)
            or minimum != 0.95
        ):
            raise HarmBenchPublicError(f"{name}.{endpoint} finite-fraction contract changed")
        gate = summary["minimum_finite_fraction_gate_applicable"]
        if not isinstance(gate, bool):
            raise HarmBenchPublicError(f"{name}.{endpoint} gate flag is not boolean")
        interval = (summary["ci95_low"], summary["ci95_high"])
        if point[endpoint] is None:
            if gate or summary["bootstrap_mean"] is not None or interval != (None, None):
                raise HarmBenchPublicError(
                    f"{name}.{endpoint} null point/bootstrap contract changed"
                )
        else:
            if (
                not gate
                or fraction < minimum
                or summary["bootstrap_mean"] is None
                or interval[0] is None
                or interval[1] is None
            ):
                raise HarmBenchPublicError(
                    f"{name}.{endpoint} finite bootstrap gate failed"
                )
            if float(interval[0]) > float(interval[1]):
                raise HarmBenchPublicError(f"{name}.{endpoint} CI order changed")
            if endpoint in RATE_ENDPOINTS:
                lower, upper = (-1.0, 1.0) if contrast else (0.0, 1.0)
                for key in ("bootstrap_mean", "ci95_low", "ci95_high"):
                    number = float(summary[key])
                    if not lower <= number <= upper:
                        raise HarmBenchPublicError(
                            f"{name}.{endpoint}.{key} is outside its range"
                        )


def _validate_cell(value: object) -> tuple[Mapping[str, object], int]:
    cell = _exact_keys(value, CELL_KEYS, name="cell")
    _validate_alignment(cell["alignment_contract"], name="cell.alignment_contract")
    _validate_metric_contract(cell["metric_contract"], name="cell.metric_contract")
    inference = _exact_keys(
        cell["inference_contract"],
        {
            "unit",
            "training_seed_count",
            "cluster_count",
            "replicates",
            "random_seed",
            "shared_plan_required_for_all_dataset_cells",
            "minimum_finite_bootstrap_fraction",
            "invalid_replicates_silently_redrawn",
        },
        name="cell.inference_contract",
    )
    if (
        inference["unit"] != "training_seed_crossed_with_whole_cluster"
        or inference["training_seed_count"] != 5
        or inference["shared_plan_required_for_all_dataset_cells"] is not True
        or inference["minimum_finite_bootstrap_fraction"] != 0.95
        or inference["invalid_replicates_silently_redrawn"] is not False
    ):
        raise HarmBenchPublicError("cell inference contract changed")
    _exact_integer(
        inference["cluster_count"], name="cell.inference_contract.cluster_count", minimum=2
    )
    replicates = _exact_integer(
        inference["replicates"], name="cell.inference_contract.replicates", minimum=1
    )
    random_seed = _exact_integer(
        inference["random_seed"], name="cell.inference_contract.random_seed", minimum=0
    )
    if (
        replicates != EXPECTED_SYNTHETIC_BOOTSTRAP_REPLICATES
        or random_seed != EXPECTED_SYNTHETIC_BOOTSTRAP_SEED
    ):
        raise HarmBenchPublicError("synthetic inference profile changed")
    point = _validate_endpoint_block(cell["point"], name="cell.point", contrast=False)
    _validate_bootstrap(
        cell["bootstrap"],
        name="cell.bootstrap",
        point=point,
        replicates=replicates,
        contrast=False,
    )
    return cell, replicates


def _validate_contrast(value: object, *, replicates: int) -> Mapping[str, object]:
    contrast = _exact_keys(value, CONTRAST_KEYS, name="contrast")
    _validate_alignment(contrast["alignment_contract"], name="contrast.alignment_contract")
    _validate_metric_contract(contrast["metric_contract"], name="contrast.metric_contract")
    if contrast["contrast_direction"] != "left_minus_right" or contrast["paired_on"] != (
        "same_training_seed_draw_and_whole_cluster_draw"
    ):
        raise HarmBenchPublicError("contrast pairing changed")
    point = _validate_endpoint_block(
        contrast["point"], name="contrast.point", contrast=True
    )
    _validate_bootstrap(
        contrast["bootstrap"],
        name="contrast.bootstrap",
        point=point,
        replicates=replicates,
        contrast=True,
    )
    return contrast


def validate_synthetic_public_report(value: object) -> Mapping[str, object]:
    root = _exact_keys(value, ROOT_KEYS, name="synthetic_public_report")
    try:
        ensure_finite_public_tree(root)
    except HarmBenchMetricError as error:
        raise HarmBenchPublicError(str(error)) from error
    if (
        root["schema_version"] != SYNTHETIC_PUBLIC_SCHEMA
        or root["protocol_id"] != PROTOCOL_ID
        or root["status"] != "complete_synthetic_contract_no_real_data"
        or root["dataset_id"] != "synthetic_dialogues"
        or root["data_role"] != "synthetic_fixture"
    ):
        raise HarmBenchPublicError("synthetic report identity changed")
    if root["protocol_sha256"] != PINNED_DEVELOPMENT_PROTOCOL_SHA256:
        raise HarmBenchPublicError("protocol SHA-256 is not the pinned synthetic contract")
    if root["model_roster"] != ["synthetic_five_seed_probability_model"]:
        raise HarmBenchPublicError("synthetic model roster changed")
    if root["strategy_roster"] != [
        "independent_current_only",
        "all_strictly_past_history",
    ]:
        raise HarmBenchPublicError("synthetic strategy roster changed")
    cell, replicates = _validate_cell(root["cell"])
    contrast = _validate_contrast(root["contrast"], replicates=replicates)
    if dict(cell["alignment_contract"]) != dict(contrast["alignment_contract"]):
        raise HarmBenchPublicError("cell and contrast alignment/plan binding differs")
    checks = _exact_keys(
        root["contract_checks"],
        {
            "synthetic_only",
            "real_data_consumed",
            "alignment_bound",
            "shared_cluster_bootstrap",
            "exact_cvar_tie_regression_covered",
            "json_nan_forbidden",
            "write_once",
        },
        name="contract_checks",
    )
    expected_checks = {
        "synthetic_only": True,
        "real_data_consumed": False,
        "alignment_bound": True,
        "shared_cluster_bootstrap": True,
        "exact_cvar_tie_regression_covered": True,
        "json_nan_forbidden": True,
        "write_once": True,
    }
    if dict(checks) != expected_checks:
        raise HarmBenchPublicError("synthetic contract checks changed")
    policy = _exact_keys(
        root["public_artifact_policy"],
        {
            "aggregate_only",
            "contains_labels_predictions_probabilities_or_embeddings",
            "contains_query_row_cluster_seed_or_participant_vectors",
            "contains_private_paths_or_outcome_hashes",
        },
        name="public_artifact_policy",
    )
    if dict(policy) != {
        "aggregate_only": True,
        "contains_labels_predictions_probabilities_or_embeddings": False,
        "contains_query_row_cluster_seed_or_participant_vectors": False,
        "contains_private_paths_or_outcome_hashes": False,
    }:
        raise HarmBenchPublicError("public artifact policy changed")
    authorization = _exact_keys(
        root["stage_authorization"],
        {
            "open_role_development_authorized",
            "official_test_feature_or_prediction_authorized",
            "official_test_label_or_outcome_authorized",
            "confirmatory_claim_authorized",
        },
        name="stage_authorization",
    )
    if dict(authorization) != {
        "open_role_development_authorized": True,
        "official_test_feature_or_prediction_authorized": False,
        "official_test_label_or_outcome_authorized": False,
        "confirmatory_claim_authorized": False,
    }:
        raise HarmBenchPublicError("synthetic stage authorization changed")
    return root


def build_synthetic_public_report(
    *,
    protocol_sha256: str,
    cell: Mapping[str, object],
    contrast: Mapping[str, object],
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": SYNTHETIC_PUBLIC_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete_synthetic_contract_no_real_data",
        "protocol_sha256": protocol_sha256,
        "dataset_id": "synthetic_dialogues",
        "data_role": "synthetic_fixture",
        "model_roster": ["synthetic_five_seed_probability_model"],
        "strategy_roster": [
            "independent_current_only",
            "all_strictly_past_history",
        ],
        "cell": dict(cell),
        "contrast": dict(contrast),
        "contract_checks": {
            "synthetic_only": True,
            "real_data_consumed": False,
            "alignment_bound": True,
            "shared_cluster_bootstrap": True,
            "exact_cvar_tie_regression_covered": True,
            "json_nan_forbidden": True,
            "write_once": True,
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_labels_predictions_probabilities_or_embeddings": False,
            "contains_query_row_cluster_seed_or_participant_vectors": False,
            "contains_private_paths_or_outcome_hashes": False,
        },
        "stage_authorization": {
            "open_role_development_authorized": True,
            "official_test_feature_or_prediction_authorized": False,
            "official_test_label_or_outcome_authorized": False,
            "confirmatory_claim_authorized": False,
        },
    }
    validate_synthetic_public_report(report)
    return report


def canonical_public_bytes(report: Mapping[str, object]) -> bytes:
    try:
        snapshot_encoded = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        snapshot = json.loads(snapshot_encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise HarmBenchPublicError(f"public report is not canonical JSON: {error}") from error
    validate_synthetic_public_report(snapshot)
    return (
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(report: Mapping[str, object], path: str | Path) -> str:
    output = Path(path)
    if output.suffix.lower() != ".json" or output.is_symlink():
        raise HarmBenchPublicError("public destination must be a plain JSON path")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise HarmBenchPublicError("public destination parent must already be a plain directory")
    encoded = canonical_public_bytes(report)
    expected = hashlib.sha256(encoded).hexdigest()
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"write-once output already exists: {output.name}") from None
        observed = hashlib.sha256(output.read_bytes()).hexdigest()
        if observed != expected:
            raise HarmBenchPublicError("public output changed during publication")
        return expected
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
