"""Fail-closed model-selection evaluation and reference freezing.

This is the first production component allowed to deserialize a
``model_selection`` outcome sidecar.  Its ordering is deliberately structural:

1. validate the byte-exact confirmatory analysis contract;
2. production-attest all four registered strategy artifacts and their live
   config/code/runtime lineage;
3. prove their common current anchor and cross-variant row/task/history/fold
   alignment;
4. reload the already-attested probability caches;
5. only then derive a narrow v2-preflight-backed label capability and open the
   one canonical model-selection label archive once with ``allow_pickle=False``.

The resulting statistics are development/model-selection evidence only.  They
freeze the strongest admissible reference and assess prospective sensitivity;
they do not authorize calibration, holdout, or test access and they are not
confirmatory performance evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast

import numpy as np

from .causal_backbone_evidence import (
    _aggregate_method_metrics,
    _bootstrap_classification_difference,
    _bootstrap_mean_regret_difference,
    _classification_score_from_confusion,
    _confusion_by_seed_cluster,
    _method_seed_metrics,
    _paired_whole_cluster_randomization_arrays,
    _randomization_assignments,
    holm_bonferroni,
)
from .causal_backbone_evidence_runner import (
    SELECTION_ROLE,
    _SPECS,
    _array_sha256,
    verify_fit_receipt_inputs,
)
from .causal_backbone_strategy_staged_pipeline import (
    JOINT_EVALUATION_ROSTER,
    METHOD_ROSTER,
    REGISTERED_VARIANTS,
    VerifiedStrategyCompletionAttestation,
    VerifiedStrategyUpstreamState,
    _live_strategy_lineage,
    derive_registered_variant,
    verify_current_only_completion_production_attestation,
    verify_strategy_completion_production_attestation,
)
from .causal_multimodal_backbone import CausalBackboneConfig
from .confirmatory_contract import (
    ACCURACY_NO_HARM_CI95_LOWER_MINIMUM,
    ACCURACY_NO_HARM_POINT_MINIMUM,
    MINIMUM_MACRO_F1_GAIN,
    RANDOMIZATION_REPLICATES,
    RANDOMIZATION_SEED,
    load_json_contract,
    validate_confirmatory_analysis,
)


CONFIRMATORY_ANALYSIS_SHA256 = (
    "990c2960eebd3be2e041067addc1d821f8f07d3d87c204ae08c134cfa6fb4fa3"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_808
PROSPECTIVE_TARGET_MACRO_F1_GAIN = 0.005
MINIMUM_PROSPECTIVE_POWER = 0.8
RANDOMIZATION_ALPHA = 0.05
EXACT_RANDOMIZATION_MAX_CLUSTERS = 16

REFERENCE_CANDIDATES = (
    "current_only",
    "all_history",
    "coverage_matched_recency",
    "forward_only_utility",
    "backward_only_utility",
)
PRIMARY_VARIANT = "full"
CAPACITY_VARIANT = "capacity_control"

PRIVATE_ARTIFACT_SCHEMA = (
    "carma_causal_backbone_model_selection_reference_freeze_private_v1"
)
PRIVATE_RECEIPT_SCHEMA = (
    "carma_causal_backbone_model_selection_reference_freeze_receipt_v1"
)
PUBLIC_REPORT_SCHEMA = (
    "carma_causal_backbone_model_selection_reference_freeze_public_v1"
)
PRIVATE_ARTIFACT_NAME = "model-selection-reference-freeze.json"
PRIVATE_RECEIPT_NAME = "model-selection-reference-freeze-receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "labels",
        "predictions",
        "probabilities",
        "protocol_row_ids",
        "row_ids",
        "cluster_codes",
        "group_ids",
        "speaker_ids",
        "contexts",
        "histories",
        "private_path",
        "artifact_path",
        "receipt_path",
        "label_path",
        "seed_results",
    }
)

_HOLM_SPEC = (
    (
        "H1_primary_macro_f1",
        "carma_bidirectional_vs_frozen_strongest_admissible_baseline",
        "macro_f1",
        "greater",
    ),
    (
        "H2_primary_mean_regret",
        "carma_bidirectional_full_vs_current_only",
        "mean_regret",
        "less",
    ),
    (
        "H3_emotion_constraint_increment",
        "carma_bidirectional_full_vs_without_emotion_constraints",
        "macro_f1",
        "greater",
    ),
    (
        "H4_three_by_three_relation_increment",
        "carma_bidirectional_full_vs_without_3x3_relations",
        "macro_f1",
        "greater",
    ),
    (
        "H5_current_only_increment",
        "carma_bidirectional_full_vs_current_only",
        "macro_f1",
        "greater",
    ),
)
_ACCURACY_IDS = (
    "A1_accuracy_vs_current",
    "A2_accuracy_vs_frozen_reference",
)
_MODEL_SELECTION_GATE_KEYS = (
    "macro_f1_point_gain_passed",
    "macro_f1_ci95_lower_above_zero_passed",
    "mean_regret_vs_current_ci95_upper_non_positive_passed",
    "mean_regret_vs_frozen_reference_ci95_upper_non_positive_passed",
    "history_harm_rate_reduction_passed",
    "accuracy_no_harm_passed",
    "per_seed_success_passed",
)


class ModelSelectionEvaluationError(ValueError):
    """Raised when the model-selection access or analysis contract changes."""


@dataclass(frozen=True)
class StrategyProductionInput:
    """One strategy artifact plus the live lineage used to produce it."""

    artifact_path: str | Path
    receipt_path: str | Path
    expected_receipt_sha256: str
    upstream: VerifiedStrategyUpstreamState
    config_paths: Mapping[str, str | Path]
    code_paths: Mapping[str, str | Path]
    environment: Mapping[str, object]


@dataclass(frozen=True)
class SelectionSidecarSource:
    """Narrow source description; it intentionally accepts no role or label path."""

    dataset: str
    sidecar_dir: str | Path
    manifest_path: str | Path
    preflight_receipt_path: str | Path
    expected_preflight_receipt_sha256: str
    config_paths: Mapping[str, str | Path]
    code_paths: Mapping[str, str | Path]
    environment: Mapping[str, object]


@dataclass(frozen=True)
class FrozenAnalysisContract:
    analysis_sha256: str
    split_manifest_sha256: str
    family_id: str
    familywise_alpha: float
    harm_reference_candidates: tuple[str, ...] = (
        "all_history",
        "coverage_matched_recency",
        "forward_only_utility",
        "backward_only_utility",
    )
    harm_reference_selection_rule: str = (
        "highest_five_seed_mean_model_selection_macro_f1_then_highest_accuracy_"
        "then_lowest_mean_regret_then_lexicographic_model_id"
    )
    minimum_relative_harm_reduction: float = 0.05
    zero_harm_denominator_action: str = "fail_closed_not_estimable"
    per_seed_required_successes: int = 4
    per_seed_success_conditions: tuple[str, ...] = (
        "macro_f1_candidate_strictly_greater_than_reference",
        "mean_regret_vs_current_non_positive",
    )
    per_seed_macro_f1_threshold: float = 0.0
    per_seed_mean_regret_threshold: float = 0.0


@dataclass(frozen=True)
class VerifiedStrategyBundle:
    """All pre-label producer evidence needed to create a label capability."""

    dataset: str
    attestations: Mapping[str, VerifiedStrategyCompletionAttestation]
    inputs: Mapping[str, StrategyProductionInput]
    receipt_lineage: Mapping[str, Mapping[str, str]]
    cross_variant_alignment_sha256: str
    full_current_anchor_history_artifact_sha256: str
    current_artifact_sha256: str
    strategy_config_roster_sha256: str
    common_strategy_code_bundle_sha256: str
    common_strategy_runtime_environment_sha256: str


@dataclass(frozen=True)
class _VerifiedSelectionLabelCapability:
    dataset: str
    label_path: Path
    label_sha256: str
    row_alignment_sha256: str
    rows: int
    manifest: Mapping[str, object]
    manifest_sha256: str
    preflight_receipt_sha256: str


@dataclass(frozen=True)
class _VariantArrays:
    variant: str
    dataset: str
    label_order: tuple[str, ...]
    protocol_row_ids: np.ndarray
    cluster_codes: np.ndarray
    probabilities: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _EvaluationInputs:
    dataset: str
    label_order: tuple[str, ...]
    protocol_row_ids: np.ndarray
    cluster_codes: np.ndarray
    history_eligible: np.ndarray
    current_probability: np.ndarray
    variants: Mapping[str, _VariantArrays]


@dataclass(frozen=True)
class CompletedModelSelectionEvaluation:
    private_artifact_path: Path
    private_artifact_sha256: str
    private_receipt_path: Path
    private_receipt_sha256: str
    public_report_path: Path
    public_report_sha256: str
    frozen_reference: str
    model_selection_gate_passed: bool
    prospective_power: float
    power_gate_passed: bool


@dataclass(frozen=True)
class VerifiedModelSelectionAggregateAttestation:
    """Aggregate-only handoff accepted by a future two-dataset joint evaluator."""

    dataset: str
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    public_report_sha256: str
    analysis_config_sha256: str
    cross_variant_alignment_sha256: str
    frozen_reference: str
    model_selection_gate_passed: bool
    prospective_power: float
    power_gate_passed: bool


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ModelSelectionEvaluationError(f"required file is missing: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if _SHA256.fullmatch(digest) is None:
        raise ModelSelectionEvaluationError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _single_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ModelSelectionEvaluationError(f"{field} must contain one string")
    return str(array.reshape(-1)[0])


def _integer_vector(value: np.ndarray, field: str, *, unique: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ModelSelectionEvaluationError(f"{field} must be an integer vector")
    result = array.astype(np.int64, copy=True)
    if np.any(result < 0) or (unique and len(np.unique(result)) != len(result)):
        raise ModelSelectionEvaluationError(f"{field} contains invalid row identities")
    return result


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_frozen_analysis_contract(path: str | Path) -> FrozenAnalysisContract:
    analysis_path = Path(path)
    observed_sha = _file_sha256(analysis_path)
    if observed_sha != CONFIRMATORY_ANALYSIS_SHA256:
        raise ModelSelectionEvaluationError(
            "confirmatory analysis config differs from the frozen byte contract"
        )
    split_path = analysis_path.parent / "carma_split_manifest_v1.json"
    try:
        analysis = load_json_contract(analysis_path)
        split_manifest = load_json_contract(split_path)
        validate_confirmatory_analysis(analysis, split_manifest)
    except (OSError, ValueError) as error:
        raise ModelSelectionEvaluationError(
            f"confirmatory analysis contract is invalid: {error}"
        ) from error

    bootstrap = cast(Mapping[str, object], analysis["hierarchical_bootstrap"])
    randomization = cast(Mapping[str, object], analysis["hypothesis_testing"])
    holm = cast(Mapping[str, object], analysis["holm_family"])
    mde = cast(Mapping[str, object], analysis["mde_and_power"])
    gates = cast(Mapping[str, object], analysis["effect_and_safety_gates"])
    accuracy = cast(Mapping[str, object], gates["accuracy_no_harm"])
    harm = cast(Mapping[str, object], gates["history_harm_rate_reduction"])
    per_seed = cast(Mapping[str, object], gates["per_seed_success"])
    per_seed_thresholds = cast(Mapping[str, object], per_seed["thresholds"])
    observed_holm = tuple(
        (
            str(cast(Mapping[str, object], row)["id"]),
            str(cast(Mapping[str, object], row)["contrast"]),
            str(cast(Mapping[str, object], row)["metric"]),
            str(cast(Mapping[str, object], row)["alternative"]),
        )
        for row in cast(Sequence[object], holm["hypotheses"])
    )
    observed_accuracy_ids = tuple(
        str(cast(Mapping[str, object], row)["id"])
        for row in cast(Sequence[object], accuracy["contrasts"])
    )
    critical_contract = (
        bootstrap.get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap.get("bootstrap_seed") == BOOTSTRAP_SEED
        and randomization.get("monte_carlo_assignments") == RANDOMIZATION_REPLICATES
        and randomization.get("randomization_seed") == RANDOMIZATION_SEED
        and randomization.get("exact_enumeration_max_clusters")
        == EXACT_RANDOMIZATION_MAX_CLUSTERS
        and observed_holm == _HOLM_SPEC
        and observed_accuracy_ids == _ACCURACY_IDS
        and float(gates["minimum_macro_f1_gain_absolute"])
        == MINIMUM_MACRO_F1_GAIN
        and float(accuracy["minimum_point_difference"])
        == ACCURACY_NO_HARM_POINT_MINIMUM
        and float(accuracy["minimum_ci95_lower"])
        == ACCURACY_NO_HARM_CI95_LOWER_MINIMUM
        and float(mde["minimum_detectable_gain_absolute"])
        == PROSPECTIVE_TARGET_MACRO_F1_GAIN
        and float(mde["minimum_power"]) == MINIMUM_PROSPECTIVE_POWER
        and float(mde["two_sided_alpha"]) == RANDOMIZATION_ALPHA
        and harm.get("candidate") == "carma_bidirectional_full"
        and harm.get("reference")
        == "strongest_history_using_admissible_baseline_frozen_on_model_selection"
        and tuple(cast(Sequence[str], harm.get("reference_candidates", ())))
        == (
            "all_history",
            "coverage_matched_recency",
            "forward_only_utility",
            "backward_only_utility",
        )
        and harm.get("reference_selection_rule")
        == (
            "highest_five_seed_mean_model_selection_macro_f1_then_highest_"
            "accuracy_then_lowest_mean_regret_then_lexicographic_model_id"
        )
        and float(harm.get("minimum_relative_reduction", math.nan)) == 0.05
        and harm.get("zero_reference_harm_rate_action")
        == "fail_closed_not_estimable"
        and per_seed.get("candidate") == "carma_bidirectional_full"
        and per_seed.get("reference")
        == "strongest_admissible_baseline_frozen_on_model_selection"
        and per_seed.get("seed_count") == 5
        and per_seed.get("required_successes") == 4
        and per_seed.get("same_seed_for_all_conditions") is True
        and tuple(cast(Sequence[str], per_seed.get("success_requires_all", ())))
        == (
            "macro_f1_candidate_strictly_greater_than_reference",
            "mean_regret_vs_current_non_positive",
        )
        and float(
            per_seed_thresholds.get(
                "macro_f1_difference_strictly_greater_than", math.nan
            )
        )
        == 0.0
        and float(
            per_seed_thresholds.get(
                "mean_regret_vs_current_must_not_exceed", math.nan
            )
        )
        == 0.0
    )
    if not critical_contract:
        raise ModelSelectionEvaluationError("frozen statistical contract changed")
    return FrozenAnalysisContract(
        analysis_sha256=observed_sha,
        split_manifest_sha256=_file_sha256(split_path),
        family_id=str(holm["family_id"]),
        familywise_alpha=float(holm["familywise_alpha"]),
        harm_reference_candidates=tuple(
            str(value)
            for value in cast(Sequence[object], harm["reference_candidates"])
        ),
        harm_reference_selection_rule=str(harm["reference_selection_rule"]),
        minimum_relative_harm_reduction=float(harm["minimum_relative_reduction"]),
        zero_harm_denominator_action=str(harm["zero_reference_harm_rate_action"]),
        per_seed_required_successes=int(per_seed["required_successes"]),
        per_seed_success_conditions=tuple(
            str(value)
            for value in cast(Sequence[object], per_seed["success_requires_all"])
        ),
        per_seed_macro_f1_threshold=float(
            per_seed_thresholds["macro_f1_difference_strictly_greater_than"]
        ),
        per_seed_mean_regret_threshold=float(
            per_seed_thresholds["mean_regret_vs_current_must_not_exceed"]
        ),
    )


def _read_attested_strategy_receipt(
    attestation: VerifiedStrategyCompletionAttestation,
) -> tuple[dict[str, object], dict[str, str]]:
    path = attestation.receipt_path
    if _file_sha256(path) != attestation.receipt_sha256:
        raise ModelSelectionEvaluationError("strategy receipt changed after attestation")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelSelectionEvaluationError(f"cannot reread strategy receipt: {error}") from error
    if not isinstance(payload, dict):
        raise ModelSelectionEvaluationError("strategy receipt root must be a mapping")
    raw_lineage = payload.get("lineage")
    if not isinstance(raw_lineage, Mapping):
        raise ModelSelectionEvaluationError("strategy receipt lineage is absent")
    lineage = {
        str(name): _require_sha256(value, f"strategy receipt lineage.{name}")
        for name, value in raw_lineage.items()
    }
    if _file_sha256(path) != attestation.receipt_sha256:
        raise ModelSelectionEvaluationError("strategy receipt changed while rereading")
    return payload, lineage


def _same_array(left: object, right: object) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def _derive_variant_from_live_configs(
    config_paths: Mapping[str, str | Path],
    *,
    dataset: str,
) -> str:
    """Derive a variant from the actual validated model JSON, never its filename."""

    candidates: list[str] = []
    for raw_path in config_paths.values():
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelSelectionEvaluationError(
                f"cannot read live strategy config {path.name}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ModelSelectionEvaluationError(
                f"live strategy config {path.name} must contain a mapping"
            )
        model = payload.get("model")
        if not isinstance(model, Mapping) or "affect_relation_mode" not in model:
            continue
        try:
            derived = derive_registered_variant(CausalBackboneConfig.from_mapping(payload))
        except (TypeError, ValueError) as error:
            raise ModelSelectionEvaluationError(
                f"live strategy model contract is invalid: {error}"
            ) from error
        experiment = payload.get("experimental_contract")
        if (
            not isinstance(experiment, Mapping)
            or experiment.get("variant") != derived
            or experiment.get("primary_variant") != PRIMARY_VARIANT
            or experiment.get("model_selection_may_choose_variant") is not False
            or experiment.get("dataset_id") != dataset
        ):
            raise ModelSelectionEvaluationError(
                "live strategy experimental contract differs from its model-derived variant"
            )
        candidates.append(derived)
    if len(candidates) != 1:
        raise ModelSelectionEvaluationError(
            "each strategy input must contain exactly one derivable model config"
        )
    return candidates[0]


def verify_strategy_bundle_before_label_access(
    strategies: Mapping[str, StrategyProductionInput],
) -> VerifiedStrategyBundle:
    """Attest all four variants without accepting or resolving any label path."""

    if set(strategies) != set(REGISTERED_VARIANTS):
        raise ModelSelectionEvaluationError(
            "strategy roster must be exactly full, no_vad, no_history_3x3, capacity_control"
        )
    attestations: dict[str, VerifiedStrategyCompletionAttestation] = {}
    inputs: dict[str, StrategyProductionInput] = {}
    receipts: dict[str, dict[str, object]] = {}
    lineages: dict[str, dict[str, str]] = {}
    live_configs: dict[str, str] = {}
    live_codes: dict[str, str] = {}
    live_runtimes: dict[str, str] = {}

    for variant in REGISTERED_VARIANTS:
        source = strategies[variant]
        if not isinstance(source, StrategyProductionInput):
            raise ModelSelectionEvaluationError(f"{variant} strategy input has wrong type")
        try:
            attestation = verify_strategy_completion_production_attestation(
                source.artifact_path,
                source.receipt_path,
                source.expected_receipt_sha256,
                upstream=source.upstream,
            )
        except (OSError, ValueError) as error:
            raise ModelSelectionEvaluationError(
                f"{variant} strategy production attestation failed: {error}"
            ) from error
        if attestation.registered_variant != variant:
            raise ModelSelectionEvaluationError(
                f"strategy variant substitution detected for {variant}"
            )
        payload, lineage = _read_attested_strategy_receipt(attestation)
        try:
            config_hashes, code_hashes, runtime_sha, live_sha = _live_strategy_lineage(
                config_paths=source.config_paths,
                code_paths=source.code_paths,
                environment=source.environment,
            )
        except (OSError, ValueError) as error:
            raise ModelSelectionEvaluationError(
                f"{variant} live strategy lineage failed: {error}"
            ) from error
        expected_live = {
            "strategy_config_bundle_sha256": _canonical_sha256(config_hashes),
            "strategy_code_bundle_sha256": _canonical_sha256(code_hashes),
            "strategy_runtime_environment_sha256": runtime_sha,
            "strategy_live_lineage_sha256": live_sha,
        }
        if any(lineage.get(name) != value for name, value in expected_live.items()):
            raise ModelSelectionEvaluationError(
                f"{variant} strategy config/code/runtime lineage changed"
            )
        if _derive_variant_from_live_configs(
            source.config_paths,
            dataset=attestation.dataset,
        ) != variant:
            raise ModelSelectionEvaluationError(
                f"{variant} label differs from the live model-derived variant"
            )
        attestations[variant] = attestation
        inputs[variant] = source
        receipts[variant] = payload
        lineages[variant] = lineage
        live_configs[variant] = expected_live["strategy_config_bundle_sha256"]
        live_codes[variant] = expected_live["strategy_code_bundle_sha256"]
        live_runtimes[variant] = expected_live[
            "strategy_runtime_environment_sha256"
        ]

    full = attestations[PRIMARY_VARIANT]
    if (
        full.variant_history_artifact_sha256
        != full.full_current_anchor_history_artifact_sha256
    ):
        raise ModelSelectionEvaluationError(
            "full strategy is not the unique current-only history anchor"
        )
    scalar_fields = (
        "dataset",
        "cross_variant_alignment_sha256",
        "full_current_anchor_history_artifact_sha256",
        "current_artifact_sha256",
        "method_roster",
        "joint_evaluation_roster",
        "base_seeds",
        "utility_seeds",
        "fit_query_count",
        "selection_query_count",
        "fit_task_count",
        "selection_task_count",
    )
    for variant, attestation in attestations.items():
        if any(
            getattr(attestation, name) != getattr(full, name) for name in scalar_fields
        ):
            raise ModelSelectionEvaluationError(
                f"{variant} strategy differs from the shared evaluation alignment"
            )
        if attestation.method_roster != METHOD_ROSTER or (
            attestation.joint_evaluation_roster != JOINT_EVALUATION_ROSTER
        ):
            raise ModelSelectionEvaluationError("strategy method roster changed")

    full_outcome = inputs[PRIMARY_VARIANT].upstream.history.outcome
    full_fold = np.asarray(
        getattr(inputs[PRIMARY_VARIANT].upstream.history.fit_outcome, "fold_by_seed_query")
    )
    for variant, source in inputs.items():
        outcome = source.upstream.history.outcome
        fold = np.asarray(getattr(source.upstream.history.fit_outcome, "fold_by_seed_query"))
        if (
            outcome.dataset != full_outcome.dataset
            or outcome.label_order != full_outcome.label_order
            or outcome.seeds != full_outcome.seeds
            or outcome.fit_histories_sha256 != full_outcome.fit_histories_sha256
            or outcome.selection_histories_sha256
            != full_outcome.selection_histories_sha256
            or outcome.fit_tasks.task_sha256 != full_outcome.fit_tasks.task_sha256
            or outcome.selection_tasks.task_sha256
            != full_outcome.selection_tasks.task_sha256
            or not _same_array(outcome.fit_protocol_row_ids, full_outcome.fit_protocol_row_ids)
            or not _same_array(
                outcome.selection_protocol_row_ids,
                full_outcome.selection_protocol_row_ids,
            )
            or not _same_array(outcome.fit_cluster_codes, full_outcome.fit_cluster_codes)
            or not _same_array(
                outcome.selection_cluster_codes,
                full_outcome.selection_cluster_codes,
            )
            or not _same_array(fold, full_fold)
        ):
            raise ModelSelectionEvaluationError(
                f"{variant} query/task/history/seed/fold alignment changed"
            )

    common_receipt_fields = (
        "fit_feature_identity_sha256",
        "selection_feature_identity_sha256",
        "fit_task_sha256",
        "selection_task_sha256",
        "history_fold_assignment_sha256",
        "history_code_bundle_sha256",
        "history_execution_environment_sha256",
        "current_source_code_sha256",
        "current_runtime_environment_sha256",
        "strategy_code_bundle_sha256",
        "strategy_runtime_environment_sha256",
    )
    full_lineage = lineages[PRIMARY_VARIANT]
    for variant, lineage in lineages.items():
        if any(lineage.get(name) != full_lineage.get(name) for name in common_receipt_fields):
            raise ModelSelectionEvaluationError(
                f"{variant} shared config/code/runtime/task lineage changed"
            )
    if len(set(live_codes.values())) != 1 or len(set(live_runtimes.values())) != 1:
        raise ModelSelectionEvaluationError(
            "strategy variants were not produced by one code/runtime contract"
        )

    config_roster_sha = _canonical_sha256(dict(sorted(live_configs.items())))
    return VerifiedStrategyBundle(
        dataset=full.dataset,
        attestations=MappingProxyType(dict(attestations)),
        inputs=MappingProxyType(dict(inputs)),
        receipt_lineage=MappingProxyType(
            {name: MappingProxyType(dict(value)) for name, value in lineages.items()}
        ),
        cross_variant_alignment_sha256=full.cross_variant_alignment_sha256,
        full_current_anchor_history_artifact_sha256=(
            full.full_current_anchor_history_artifact_sha256
        ),
        current_artifact_sha256=full.current_artifact_sha256,
        strategy_config_roster_sha256=config_roster_sha,
        common_strategy_code_bundle_sha256=next(iter(live_codes.values())),
        common_strategy_runtime_environment_sha256=next(iter(live_runtimes.values())),
    )


def _verify_selection_label_capability(
    source: SelectionSidecarSource,
    strategies: VerifiedStrategyBundle,
) -> _VerifiedSelectionLabelCapability:
    """Derive the only label capability, after the four-strategy gate exists."""

    if not isinstance(strategies, VerifiedStrategyBundle):
        raise ModelSelectionEvaluationError(
            "selection outcomes require a verified four-strategy bundle"
        )
    if not isinstance(source, SelectionSidecarSource):
        raise ModelSelectionEvaluationError("selection sidecar source has wrong type")
    if source.dataset != strategies.dataset:
        raise ModelSelectionEvaluationError("selection sidecar dataset differs from strategies")
    try:
        _receipt, sidecars = verify_fit_receipt_inputs(
            receipt_path=source.preflight_receipt_path,
            expected_receipt_sha256=source.expected_preflight_receipt_sha256,
            dataset=source.dataset,
            sidecar_dir=source.sidecar_dir,
            manifest_path=source.manifest_path,
            config_paths=source.config_paths,
            code_paths=source.code_paths,
            environment=source.environment,
        )
    except (OSError, ValueError) as error:
        raise ModelSelectionEvaluationError(
            f"v2 model-selection preflight verification failed: {error}"
        ) from error

    full_features = strategies.inputs[PRIMARY_VARIANT].upstream.history.selection_features
    record = sidecars.selection
    canonical_directory = Path(source.sidecar_dir).resolve()
    label_path = record.label_path.resolve()
    if (
        label_path.name != "labels_model_selection.npz"
        or label_path.parent != canonical_directory
    ):
        raise ModelSelectionEvaluationError(
            "model-selection label capability did not resolve to the canonical filename"
        )
    if (
        sidecars.dataset != strategies.dataset
        or sidecars.manifest_sha256 != full_features.manifest_sha256
        or record.feature_sha256 != full_features.feature_file_sha256
        or record.row_alignment_sha256 != full_features.row_alignment_sha256
        or record.rows != len(full_features.protocol_row_ids)
    ):
        raise ModelSelectionEvaluationError(
            "model-selection manifest/preflight capability differs from strategy features"
        )
    expected_receipt = _require_sha256(
        source.expected_preflight_receipt_sha256,
        "expected_preflight_receipt_sha256",
    )
    return _VerifiedSelectionLabelCapability(
        dataset=sidecars.dataset,
        label_path=label_path,
        label_sha256=_require_sha256(record.label_sha256, "selection label SHA-256"),
        row_alignment_sha256=_require_sha256(
            record.row_alignment_sha256, "selection row alignment SHA-256"
        ),
        rows=record.rows,
        manifest=sidecars.manifest,
        manifest_sha256=sidecars.manifest_sha256,
        preflight_receipt_sha256=expected_receipt,
    )


def _load_model_selection_labels_once(
    capability: _VerifiedSelectionLabelCapability,
    strategies: VerifiedStrategyBundle,
) -> np.ndarray:
    """Open exactly one canonical model-selection label NPZ, once, without pickle."""

    if not isinstance(capability, _VerifiedSelectionLabelCapability):
        raise ModelSelectionEvaluationError("verified label capability is required")
    if capability.dataset != strategies.dataset:
        raise ModelSelectionEvaluationError("label capability dataset changed")
    spec = _SPECS.get(capability.dataset)
    if spec is None:
        raise ModelSelectionEvaluationError("unsupported model-selection dataset")
    if _file_sha256(capability.label_path) != capability.label_sha256:
        raise ModelSelectionEvaluationError(
            "model-selection label SHA-256 differs before deserialization"
        )
    try:
        with np.load(capability.label_path, allow_pickle=False) as archive:
            if set(archive.files) != set(spec.label_fields):
                raise ModelSelectionEvaluationError(
                    "model-selection label sidecar schema changed"
                )
            values = {name: np.asarray(archive[name]) for name in archive.files}
    except ModelSelectionEvaluationError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ModelSelectionEvaluationError(
            f"cannot deserialize model-selection labels: {error}"
        ) from error
    if _file_sha256(capability.label_path) != capability.label_sha256:
        raise ModelSelectionEvaluationError(
            "model-selection label archive changed while deserializing"
        )

    if capability.dataset == "EmotionTalk":
        if (
            _single_text(values["schema_version"], "schema_version")
            != spec.label_schema
            or _single_text(values["dataset_id"], "dataset_id") != capability.dataset
            or _single_text(values["role"], "role") != SELECTION_ROLE
            or _single_text(values["split_protocol_id"], "split_protocol_id")
            != "scu_set_exploration_v1"
        ):
            raise ModelSelectionEvaluationError(
                "EmotionTalk model-selection label contract changed"
            )
        source_contract = capability.manifest.get("source_contract")
        if not isinstance(source_contract, Mapping) or (
            _single_text(values["source_label_sha256"], "source_label_sha256")
            != str(source_contract.get("label_archive"))
        ):
            raise ModelSelectionEvaluationError(
                "EmotionTalk model-selection label source lineage changed"
            )
    else:
        if (
            _single_text(values["schema_version"], "schema_version")
            != spec.label_schema
            or _single_text(values["role"], "role") != SELECTION_ROLE
        ):
            raise ModelSelectionEvaluationError("MELD model-selection label contract changed")
    if (
        _single_text(values["row_alignment_sha256"], "row_alignment_sha256")
        != capability.row_alignment_sha256
    ):
        raise ModelSelectionEvaluationError("model-selection label row alignment changed")
    labels = np.asarray(values["labels"])
    if (
        labels.shape != (capability.rows,)
        or not np.issubdtype(labels.dtype, np.integer)
    ):
        raise ModelSelectionEvaluationError(
            "model-selection labels must be one integer per sidecar row"
        )
    labels = labels.astype(np.int64, copy=True)
    if np.any((labels < 0) | (labels >= len(spec.label_order))):
        raise ModelSelectionEvaluationError(
            "model-selection label lies outside the frozen class order"
        )

    full = strategies.inputs[PRIMARY_VARIANT].upstream
    sidecar_protocol = _integer_vector(
        full.history.selection_features.protocol_row_ids,
        "selection feature protocol rows",
        unique=True,
    )
    strategy_protocol = _integer_vector(
        full.history.outcome.selection_protocol_row_ids,
        "strategy selection protocol rows",
        unique=True,
    )
    if (
        len(sidecar_protocol) != len(labels)
        or set(sidecar_protocol.tolist()) != set(strategy_protocol.tolist())
    ):
        raise ModelSelectionEvaluationError(
            "label sidecar and strategy selection protocol rows differ"
        )
    position = {int(row): index for index, row in enumerate(sidecar_protocol)}
    return labels[np.asarray([position[int(row)] for row in strategy_protocol], dtype=np.int64)]


def _validated_probability(
    value: np.ndarray,
    *,
    shape: tuple[int, int, int],
    field: str,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or np.any(array > 1.0)
        or not np.allclose(array.sum(axis=-1), 1.0, atol=1.0e-5)
    ):
        raise ModelSelectionEvaluationError(f"{field} contains invalid probabilities")
    return array.astype(np.float64, copy=True)


def _load_attested_variant_arrays(
    attestation: VerifiedStrategyCompletionAttestation,
    upstream: VerifiedStrategyUpstreamState,
) -> _VariantArrays:
    path = attestation.artifact_path
    if _file_sha256(path) != attestation.artifact_sha256:
        raise ModelSelectionEvaluationError("strategy artifact changed after attestation")
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)

            def read(name: str) -> np.ndarray:
                if name not in keys:
                    raise ModelSelectionEvaluationError(
                        f"strategy artifact field is missing: {name}"
                    )
                return np.asarray(archive[name])

            dataset = _single_text(read("dataset"), "dataset")
            variant = _single_text(read("registered_variant"), "registered_variant")
            label_order = tuple(
                str(value)
                for value in np.asarray(read("dataset_class_order")).reshape(-1)
            )
            roster = tuple(
                str(value) for value in np.asarray(read("method_roster")).reshape(-1)
            )
            protocol = _integer_vector(
                read("selection_protocol_rows"),
                "selection_protocol_rows",
                unique=True,
            )
            clusters = _integer_vector(
                read("selection_cluster_codes"), "selection_cluster_codes"
            )
            rows = len(protocol)
            classes = len(label_order)
            probabilities: dict[str, np.ndarray] = {}
            for method in METHOD_ROSTER:
                name = f"{method}_probability_fold_ensemble"
                matrix = _validated_probability(
                    read(name),
                    shape=(len(attestation.base_seeds), rows, classes),
                    field=name,
                )
                digest_name = f"matrix_{name}_sha256"
                declared = _require_sha256(
                    _single_text(read(digest_name), digest_name), digest_name
                )
                if declared != _array_sha256(read(name)):
                    raise ModelSelectionEvaluationError(
                        f"strategy probability hash changed: {method}"
                    )
                probabilities[method] = matrix
    except ModelSelectionEvaluationError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ModelSelectionEvaluationError(
            f"cannot reload attested strategy artifact: {error}"
        ) from error
    if _file_sha256(path) != attestation.artifact_sha256:
        raise ModelSelectionEvaluationError("strategy artifact changed while reloading")
    if (
        dataset != attestation.dataset
        or variant != attestation.registered_variant
        or roster != METHOD_ROSTER
        or label_order != upstream.history.outcome.label_order
        or clusters.shape != protocol.shape
        or len(np.unique(clusters)) < 2
        or not np.array_equal(protocol, upstream.history.outcome.selection_protocol_row_ids)
        or not np.array_equal(clusters, upstream.history.outcome.selection_cluster_codes)
    ):
        raise ModelSelectionEvaluationError(
            "attested strategy probability alignment changed"
        )
    return _VariantArrays(
        variant=variant,
        dataset=dataset,
        label_order=label_order,
        protocol_row_ids=protocol,
        cluster_codes=clusters,
        probabilities=MappingProxyType(probabilities),
    )


def _collect_evaluation_inputs(strategies: VerifiedStrategyBundle) -> _EvaluationInputs:
    variant_arrays = {
        variant: _load_attested_variant_arrays(
            strategies.attestations[variant], strategies.inputs[variant].upstream
        )
        for variant in REGISTERED_VARIANTS
    }
    full = variant_arrays[PRIMARY_VARIANT]
    for variant, values in variant_arrays.items():
        if (
            values.dataset != full.dataset
            or values.label_order != full.label_order
            or not np.array_equal(values.protocol_row_ids, full.protocol_row_ids)
            or not np.array_equal(values.cluster_codes, full.cluster_codes)
        ):
            raise ModelSelectionEvaluationError(
                f"{variant} probability cache is not cross-variant aligned"
            )

    full_upstream = strategies.inputs[PRIMARY_VARIANT].upstream
    current = full_upstream.current
    try:
        live_current = verify_current_only_completion_production_attestation(
            current.artifact_path,
            current.completion_receipt_path,
            current.completion_receipt_sha256,
            history_attestation=full_upstream.full_history_anchor,
            producer_alignment=current.producer_alignment,
        )
    except (OSError, ValueError) as error:
        raise ModelSelectionEvaluationError(
            f"full-anchor current-only attestation failed: {error}"
        ) from error
    if live_current.artifact_sha256 != strategies.current_artifact_sha256:
        raise ModelSelectionEvaluationError("full-anchor current artifact changed")
    current_protocol = _integer_vector(
        live_current.selection_protocol_row_ids,
        "current selection protocol rows",
        unique=True,
    )
    if set(current_protocol.tolist()) != set(full.protocol_row_ids.tolist()):
        raise ModelSelectionEvaluationError(
            "current-only and strategy selection rows differ"
        )
    current_probability_raw = _validated_probability(
        live_current.artifact.selection_probability,
        shape=(len(live_current.artifact.seeds), len(current_protocol), len(full.label_order)),
        field="independent_current_only.selection_probability",
    )
    current_position = {int(row): index for index, row in enumerate(current_protocol)}
    reorder = np.asarray(
        [current_position[int(row)] for row in full.protocol_row_ids], dtype=np.int64
    )
    current_probability = current_probability_raw[:, reorder, :]
    current_clusters = np.asarray(
        live_current.artifact.selection_cluster_codes, dtype=np.int64
    )[reorder]
    if not np.array_equal(current_clusters, full.cluster_codes):
        raise ModelSelectionEvaluationError(
            "current-only clusters differ from full strategy clusters"
        )

    features = full_upstream.history.selection_features
    feature_protocol = _integer_vector(
        features.protocol_row_ids, "selection feature protocol rows", unique=True
    )
    if set(feature_protocol.tolist()) != set(full.protocol_row_ids.tolist()):
        raise ModelSelectionEvaluationError(
            "selection feature history rows differ from strategy rows"
        )
    eligible_by_protocol = {
        int(row): bool(history)
        for row, history in zip(feature_protocol, features.histories, strict=True)
    }
    history_eligible = np.asarray(
        [eligible_by_protocol[int(row)] for row in full.protocol_row_ids], dtype=bool
    )
    if (
        not np.any(history_eligible)
        or len(np.unique(full.cluster_codes[history_eligible])) < 2
    ):
        raise ModelSelectionEvaluationError(
            "model-selection evaluation needs two history-eligible clusters"
        )
    return _EvaluationInputs(
        dataset=full.dataset,
        label_order=full.label_order,
        protocol_row_ids=full.protocol_row_ids.copy(),
        cluster_codes=full.cluster_codes.copy(),
        history_eligible=history_eligible,
        current_probability=current_probability,
        variants=MappingProxyType(variant_arrays),
    )


def _contrast_result(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    current: np.ndarray,
    clusters: np.ndarray,
    eligible: np.ndarray,
    metric: str,
    alternative: str,
) -> tuple[dict[str, object], np.ndarray]:
    if metric in {"macro_f1", "accuracy"}:
        point, bootstrap, cluster_count = _bootstrap_classification_difference(
            labels=labels,
            candidate=candidate,
            reference=reference,
            clusters=clusters,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
            metric=metric,
        )
    elif metric == "mean_regret":
        point, bootstrap, cluster_count = _bootstrap_mean_regret_difference(
            labels=labels,
            candidate=candidate,
            reference=reference,
            current=current,
            clusters=clusters,
            eligible=eligible,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
    else:
        raise ModelSelectionEvaluationError(f"unsupported contrast metric: {metric}")
    try:
        randomization = _paired_whole_cluster_randomization_arrays(
            labels=labels,
            candidate=candidate,
            reference=reference,
            current=current,
            clusters=clusters,
            eligible=eligible,
            metric=metric,
            alternative=alternative,
            replicates=RANDOMIZATION_REPLICATES,
            seed=RANDOMIZATION_SEED,
        )
    except ValueError as error:
        raise ModelSelectionEvaluationError(
            f"paired whole-cluster randomization failed: {error}"
        ) from error
    if not math.isclose(
        point,
        float(randomization["point_difference"]),
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("bootstrap and randomization estimands differ")
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return (
        {
            "metric": metric,
            "alternative": alternative,
            "difference_definition": "candidate_minus_reference",
            "point_difference": float(point),
            "favorable_direction_point": float(
                point if alternative == "greater" else -point
            ),
            "ci95_percentile": [float(low), float(high)],
            "bootstrap_design": (
                "five_training_seeds_crossed_with_shared_whole_cluster_draw"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster_count": int(cluster_count),
            "queries_within_cluster_kept_together": True,
            "independent_query_resampling": False,
            "hypothesis_test": randomization,
        },
        bootstrap,
    )


def _paired_mean_regret_ci_only(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    current: np.ndarray,
    clusters: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, object]:
    """Prespecified safety gate CI; it is not added to or removed from Holm."""

    try:
        point, bootstrap, cluster_count = _bootstrap_mean_regret_difference(
            labels=labels,
            candidate=candidate,
            reference=reference,
            current=current,
            clusters=clusters,
            eligible=eligible,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
    except ValueError as error:
        raise ModelSelectionEvaluationError(
            f"frozen-reference mean-regret bootstrap failed: {error}"
        ) from error
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "metric": "mean_regret",
        "difference_definition": "candidate_minus_frozen_reference",
        "point_difference": float(point),
        "ci95_percentile": [float(low), float(high)],
        "ci95_upper_must_not_exceed": 0.0,
        "passed": bool(float(high) <= 0.0),
        "bootstrap_design": (
            "five_training_seeds_crossed_with_shared_whole_cluster_draw"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_count": int(cluster_count),
        "included_in_holm_family": False,
        "prespecified_safety_gate_not_unlisted_inferential_claim": True,
    }


def _macro_f1_randomization_null(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    cluster_values = np.unique(clusters)
    candidate_confusion = _confusion_by_seed_cluster(
        labels, candidate, clusters, cluster_values
    )
    reference_confusion = _confusion_by_seed_cluster(
        labels, reference, clusters, cluster_values
    )
    assignments, exact, assignment_count = _randomization_assignments(
        cluster_count=len(cluster_values),
        replicates=RANDOMIZATION_REPLICATES,
        seed=RANDOMIZATION_SEED,
    )
    null = np.empty(assignment_count, dtype=np.float64)
    for index, swap in enumerate(assignments):
        candidate_matrix = np.where(
            swap[None, :, None, None],
            reference_confusion,
            candidate_confusion,
        ).sum(axis=1)
        reference_matrix = np.where(
            swap[None, :, None, None],
            candidate_confusion,
            reference_confusion,
        ).sum(axis=1)
        null[index] = float(
            np.mean(
                [
                    _classification_score_from_confusion(
                        candidate_matrix[seed_index], "macro_f1"
                    )
                    - _classification_score_from_confusion(
                        reference_matrix[seed_index], "macro_f1"
                    )
                    for seed_index in range(candidate.shape[0])
                ]
            )
        )
    return null, exact, assignment_count


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, probability, interpolation="higher"))


def _prospective_sensitivity(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
    point: float,
    bootstrap: np.ndarray,
) -> dict[str, object]:
    null, exact, assignment_count = _macro_f1_randomization_null(
        labels=labels,
        candidate=candidate,
        reference=reference,
        clusters=clusters,
    )
    critical = _higher_quantile(np.abs(null), 1.0 - RANDOMIZATION_ALPHA)
    centered_error = np.asarray(bootstrap, dtype=np.float64) - float(point)
    simulated = PROSPECTIVE_TARGET_MACRO_F1_GAIN + centered_error
    power = float(np.mean(np.abs(simulated) > critical))
    passed = bool(power >= MINIMUM_PROSPECTIVE_POWER)
    return {
        "analysis_type": "prospective_design_sensitivity_not_observed_post_hoc_power",
        "target_metric": "macro_f1",
        "assumed_effect_absolute": PROSPECTIVE_TARGET_MACRO_F1_GAIN,
        "assumed_effect_source": "frozen_minimum_meaningful_gain_not_observed_effect",
        "bootstrap_error": "centered_whole_cluster_crossed_seed_bootstrap",
        "bootstrap_error_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "randomization_critical_value": critical,
        "randomization_alpha": RANDOMIZATION_ALPHA,
        "randomization_alternative": "two_sided_absolute_macro_f1_difference",
        "randomization_assignments": int(assignment_count),
        "randomization_exact_enumeration": bool(exact),
        "configured_randomization_replicates": RANDOMIZATION_REPLICATES,
        "configured_randomization_seed": RANDOMIZATION_SEED,
        "estimated_power": power,
        "minimum_power": MINIMUM_PROSPECTIVE_POWER,
        "power_gate_passed": passed,
        "observed_effect_used_as_assumed_effect": False,
        "observed_post_hoc_power_computed": False,
        "underpowered_action": (
            "do_not_unseal_calibration_holdout_or_external_test"
            if not passed
            else "eligible_only_for_separate_cross_dataset_freeze_review"
        ),
    }


def _evaluate_model_selection_aggregates(
    *,
    labels: np.ndarray,
    inputs: _EvaluationInputs,
    analysis: FrozenAnalysisContract,
) -> dict[str, object]:
    labels = np.asarray(labels)
    rows = len(inputs.protocol_row_ids)
    if (
        labels.shape != (rows,)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= len(inputs.label_order)))
    ):
        raise ModelSelectionEvaluationError("aligned model-selection outcomes are invalid")
    labels = labels.astype(np.int64, copy=False)
    full = inputs.variants[PRIMARY_VARIANT].probabilities[
        "bidirectional_selected_history"
    ]
    no_vad = inputs.variants["no_vad"].probabilities[
        "bidirectional_selected_history"
    ]
    no_history_3x3 = inputs.variants["no_history_3x3"].probabilities[
        "bidirectional_selected_history"
    ]
    capacity = inputs.variants[CAPACITY_VARIANT].probabilities[
        "bidirectional_selected_history"
    ]
    current = inputs.current_probability
    full_methods = inputs.variants[PRIMARY_VARIANT].probabilities
    references: dict[str, np.ndarray] = {
        "current_only": current,
        "all_history": full_methods["all_history_diagnostic"],
        "coverage_matched_recency": full_methods["coverage_matched_recency"],
        "forward_only_utility": full_methods["forward_only_selected_history"],
        "backward_only_utility": full_methods["backward_only_selected_history"],
    }
    all_methods: dict[str, np.ndarray] = {
        **references,
        "carma_bidirectional_full": full,
        "carma_bidirectional_no_vad": no_vad,
        "carma_bidirectional_no_history_3x3": no_history_3x3,
        "carma_bidirectional_capacity_control": capacity,
    }
    method_seed_metrics = {
        name: _method_seed_metrics(
            labels,
            probability,
            current,
            inputs.history_eligible,
        )
        for name, probability in sorted(all_methods.items())
    }
    method_metrics = {
        name: _aggregate_method_metrics(records)
        for name, records in method_seed_metrics.items()
    }

    ranking_rows = []
    for name in REFERENCE_CANDIDATES:
        metrics = cast(Mapping[str, Mapping[str, float]], method_metrics[name])
        ranking_rows.append(
            {
                "method": name,
                "macro_f1": float(metrics["macro_f1"]["five_seed_mean"]),
                "accuracy": float(metrics["accuracy"]["five_seed_mean"]),
                "mean_regret": float(metrics["mean_regret"]["five_seed_mean"]),
            }
        )
    ranking_rows.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["accuracy"]),
            float(row["mean_regret"]),
            str(row["method"]),
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank
    frozen_reference = str(ranking_rows[0]["method"])
    frozen_probability = references[frozen_reference]

    history_ranking = [
        dict(row)
        for row in ranking_rows
        if str(row["method"]) in analysis.harm_reference_candidates
    ]
    if tuple(sorted(row["method"] for row in history_ranking)) != tuple(
        sorted(analysis.harm_reference_candidates)
    ):
        raise ModelSelectionEvaluationError(
            "frozen history-harm reference candidates are not all available"
        )
    history_ranking.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["accuracy"]),
            float(row["mean_regret"]),
            str(row["method"]),
        )
    )
    for rank, row in enumerate(history_ranking, start=1):
        row["rank"] = rank
    harm_reference = str(history_ranking[0]["method"])
    harm_reference_rate = float(
        cast(Mapping[str, Mapping[str, float]], method_metrics[harm_reference])[
            "history_harm_rate"
        ]["five_seed_mean"]
    )
    full_harm_rate = float(
        cast(
            Mapping[str, Mapping[str, float]],
            method_metrics["carma_bidirectional_full"],
        )["history_harm_rate"]["five_seed_mean"]
    )
    if analysis.zero_harm_denominator_action != "fail_closed_not_estimable":
        raise AssertionError("unknown zero harm denominator action")
    harm_estimable = bool(harm_reference_rate > 0.0)
    relative_harm_reduction = (
        float((harm_reference_rate - full_harm_rate) / harm_reference_rate)
        if harm_estimable
        else None
    )
    harm_gate = {
        "candidate": "carma_bidirectional_full",
        "reference": harm_reference,
        "reference_candidates": list(analysis.harm_reference_candidates),
        "reference_selection_rule": analysis.harm_reference_selection_rule,
        "reference_ranking": history_ranking,
        "candidate_five_seed_mean_history_harm_rate": full_harm_rate,
        "reference_five_seed_mean_history_harm_rate": harm_reference_rate,
        "relative_history_harm_rate_reduction": relative_harm_reduction,
        "minimum_relative_reduction": analysis.minimum_relative_harm_reduction,
        "estimable": harm_estimable,
        "passed": bool(
            harm_estimable
            and cast(float, relative_harm_reduction)
            >= analysis.minimum_relative_harm_reduction
        ),
        "zero_reference_harm_rate_action": analysis.zero_harm_denominator_action,
        "failure_reason": (
            None
            if harm_estimable
            else "zero_reference_harm_rate_fail_closed_not_estimable"
        ),
    }

    full_seed_records = method_seed_metrics["carma_bidirectional_full"]
    reference_seed_records = method_seed_metrics[frozen_reference]
    if len(full_seed_records) != 5 or len(reference_seed_records) != 5:
        raise ModelSelectionEvaluationError("per-seed success requires five aligned seeds")
    per_seed_rows = []
    for seed_index, (candidate_row, reference_row) in enumerate(
        zip(full_seed_records, reference_seed_records, strict=True)
    ):
        macro_difference = float(
            candidate_row["macro_f1"] - reference_row["macro_f1"]
        )
        mean_regret = float(candidate_row["mean_regret"])
        macro_passed = bool(
            macro_difference > analysis.per_seed_macro_f1_threshold
        )
        regret_passed = bool(
            mean_regret <= analysis.per_seed_mean_regret_threshold
        )
        per_seed_rows.append(
            {
                "seed_position": seed_index,
                "macro_f1_candidate_minus_reference": macro_difference,
                "mean_regret_candidate_vs_current": mean_regret,
                "macro_f1_condition_passed": macro_passed,
                "mean_regret_condition_passed": regret_passed,
                "same_seed_joint_success": bool(macro_passed and regret_passed),
            }
        )
    success_count = int(sum(bool(row["same_seed_joint_success"]) for row in per_seed_rows))
    per_seed_gate = {
        "candidate": "carma_bidirectional_full",
        "reference": frozen_reference,
        "seed_count": 5,
        "required_successes": analysis.per_seed_required_successes,
        "same_seed_for_all_conditions": True,
        "success_requires_all": list(analysis.per_seed_success_conditions),
        "macro_f1_difference_strictly_greater_than": (
            analysis.per_seed_macro_f1_threshold
        ),
        "mean_regret_vs_current_must_not_exceed": (
            analysis.per_seed_mean_regret_threshold
        ),
        "seed_results": per_seed_rows,
        "success_count": success_count,
        "passed": bool(success_count >= analysis.per_seed_required_successes),
        "aggregate_effect_ci_harm_and_accuracy_gates_are_independent": True,
    }

    contrast_inputs = {
        "H1_primary_macro_f1": (full, frozen_probability, "macro_f1", "greater"),
        "H2_primary_mean_regret": (full, current, "mean_regret", "less"),
        "H3_emotion_constraint_increment": (full, no_vad, "macro_f1", "greater"),
        "H4_three_by_three_relation_increment": (
            full,
            no_history_3x3,
            "macro_f1",
            "greater",
        ),
        "H5_current_only_increment": (full, current, "macro_f1", "greater"),
    }
    holm_results: dict[str, dict[str, object]] = {}
    raw_p_values: dict[str, float] = {}
    h1_bootstrap: np.ndarray | None = None
    for hypothesis_id, _contrast, metric, alternative in _HOLM_SPEC:
        candidate, reference, bound_metric, bound_alternative = contrast_inputs[
            hypothesis_id
        ]
        if metric != bound_metric or alternative != bound_alternative:
            raise AssertionError("frozen hypothesis implementation drifted")
        result, bootstrap = _contrast_result(
            labels=labels,
            candidate=candidate,
            reference=reference,
            current=current,
            clusters=inputs.cluster_codes,
            eligible=inputs.history_eligible,
            metric=metric,
            alternative=alternative,
        )
        holm_results[hypothesis_id] = result
        raw_p_values[hypothesis_id] = float(
            cast(Mapping[str, object], result["hypothesis_test"])[
                "paired_whole_cluster_randomization_p_value"
            ]
        )
        if hypothesis_id == "H1_primary_macro_f1":
            h1_bootstrap = bootstrap
    adjusted = holm_bonferroni(
        raw_p_values,
        declared_order=[row[0] for row in _HOLM_SPEC],
        alpha=analysis.familywise_alpha,
    )
    for hypothesis_id, multiplicity in adjusted.items():
        holm_results[hypothesis_id]["multiplicity"] = multiplicity

    accuracy_results: dict[str, dict[str, object]] = {}
    for contrast_id, reference in (
        ("A1_accuracy_vs_current", current),
        ("A2_accuracy_vs_frozen_reference", frozen_probability),
    ):
        result, _bootstrap = _contrast_result(
            labels=labels,
            candidate=full,
            reference=reference,
            current=current,
            clusters=inputs.cluster_codes,
            eligible=inputs.history_eligible,
            metric="accuracy",
            alternative="greater",
        )
        point = float(result["point_difference"])
        ci_low = float(cast(Sequence[float], result["ci95_percentile"])[0])
        result.update(
            {
                "minimum_point_difference": ACCURACY_NO_HARM_POINT_MINIMUM,
                "minimum_ci95_lower": ACCURACY_NO_HARM_CI95_LOWER_MINIMUM,
                "passed": bool(
                    point >= ACCURACY_NO_HARM_POINT_MINIMUM
                    and ci_low >= ACCURACY_NO_HARM_CI95_LOWER_MINIMUM
                ),
                "noninferiority_is_not_improvement_evidence": True,
            }
        )
        accuracy_results[contrast_id] = result

    capacity_metrics = cast(
        Mapping[str, Mapping[str, float]],
        method_metrics["carma_bidirectional_capacity_control"],
    )
    full_metrics = cast(
        Mapping[str, Mapping[str, float]], method_metrics["carma_bidirectional_full"]
    )
    capacity_diagnostic = {
        "variant": CAPACITY_VARIANT,
        "purpose": "prespecified_capacity_and_mechanism_diagnostic_only",
        "included_in_holm_family": False,
        "may_enter_or_remove_a_holm_hypothesis": False,
        "macro_f1_point_difference_full_minus_capacity": float(
            full_metrics["macro_f1"]["five_seed_mean"]
            - capacity_metrics["macro_f1"]["five_seed_mean"]
        ),
        "accuracy_point_difference_full_minus_capacity": float(
            full_metrics["accuracy"]["five_seed_mean"]
            - capacity_metrics["accuracy"]["five_seed_mean"]
        ),
        "mean_regret_point_difference_full_minus_capacity": float(
            full_metrics["mean_regret"]["five_seed_mean"]
            - capacity_metrics["mean_regret"]["five_seed_mean"]
        ),
        "inferential_claim_authorized": False,
    }
    if h1_bootstrap is None:
        raise AssertionError("H1 bootstrap was not computed")
    h1_point = float(holm_results["H1_primary_macro_f1"]["point_difference"])
    sensitivity = _prospective_sensitivity(
        labels=labels,
        candidate=full,
        reference=frozen_probability,
        clusters=inputs.cluster_codes,
        point=h1_point,
        bootstrap=h1_bootstrap,
    )
    power_passed = bool(sensitivity["power_gate_passed"])
    frozen_reference_regret = _paired_mean_regret_ci_only(
        labels=labels,
        candidate=full,
        reference=frozen_probability,
        current=current,
        clusters=inputs.cluster_codes,
        eligible=inputs.history_eligible,
    )
    h1_result = holm_results["H1_primary_macro_f1"]
    h2_result = holm_results["H2_primary_mean_regret"]
    h1_ci = cast(Sequence[float], h1_result["ci95_percentile"])
    h2_ci = cast(Sequence[float], h2_result["ci95_percentile"])
    aggregate_gates = {
        "minimum_macro_f1_gain_absolute": MINIMUM_MACRO_F1_GAIN,
        "macro_f1_point_gain_passed": bool(
            float(h1_result["point_difference"]) >= MINIMUM_MACRO_F1_GAIN
        ),
        "macro_f1_ci95_lower_above_zero_passed": bool(float(h1_ci[0]) > 0.0),
        "mean_regret_vs_current_ci95_upper_non_positive_passed": bool(
            float(h2_ci[1]) <= 0.0
        ),
        "mean_regret_vs_frozen_reference": frozen_reference_regret,
        "mean_regret_vs_frozen_reference_ci95_upper_non_positive_passed": bool(
            frozen_reference_regret["passed"]
        ),
        "history_harm_rate_reduction_passed": bool(harm_gate["passed"]),
        "accuracy_no_harm_passed": bool(
            all(bool(result["passed"]) for result in accuracy_results.values())
        ),
        "per_seed_success_passed": bool(per_seed_gate["passed"]),
        "gates_are_conjunctive_and_independently_computed": True,
    }
    return {
        "counts": {
            "queries": rows,
            "clusters": int(len(np.unique(inputs.cluster_codes))),
            "history_eligible_queries": int(np.sum(inputs.history_eligible)),
            "history_eligible_clusters": int(
                len(np.unique(inputs.cluster_codes[inputs.history_eligible]))
            ),
            "training_seeds": int(full.shape[0]),
        },
        "reference_freeze": {
            "frozen_reference": frozen_reference,
            "candidate_roster": list(REFERENCE_CANDIDATES),
            "selection_rule": (
                "highest_five_seed_mean_model_selection_macro_f1_then_highest_"
                "accuracy_then_lowest_mean_regret_then_lexicographic_model_id"
            ),
            "ranking": ranking_rows,
            "frozen_before_calibration_unseal": True,
        },
        "method_metrics": method_metrics,
        "holm_family": {
            "family_id": analysis.family_id,
            "method": "holm_bonferroni",
            "familywise_alpha": analysis.familywise_alpha,
            "hypothesis_order": [row[0] for row in _HOLM_SPEC],
            "capacity_control_included": False,
            "results": holm_results,
            "interpretation": "non_confirmatory_model_selection_diagnostic_only",
        },
        "accuracy_no_harm": {
            "contrast_order": list(_ACCURACY_IDS),
            "results": accuracy_results,
            "all_passed": bool(
                all(bool(result["passed"]) for result in accuracy_results.values())
            ),
            "interpretation": "non_confirmatory_model_selection_diagnostic_only",
        },
        "history_harm_rate_reduction": harm_gate,
        "per_seed_success": per_seed_gate,
        "aggregate_gates": aggregate_gates,
        "capacity_control": capacity_diagnostic,
        "prospective_sensitivity": sensitivity,
        "stage_authorization": {
            "power_gate_passed": power_passed,
            "calibration_unseal_authorized": False,
            "internal_holdout_unseal_authorized": False,
            "external_test_unseal_authorized": False,
            "reason": (
                "prospective_power_below_0.8_keep_all_later_roles_sealed"
                if not power_passed
                else "single_dataset_model_selection_evaluator_has_no_unseal_authority"
            ),
            "next_required_authority": (
                "hash_bound_two_required_dataset_joint_freeze_and_separate_stage_gate"
            ),
        },
    }


def _validate_json_safe_aggregate(value: object, *, public: bool) -> None:
    def visit(child: object, trail: tuple[str, ...]) -> None:
        if isinstance(child, (np.ndarray, Path)):
            raise ModelSelectionEvaluationError(
                f"aggregate output contains a non-JSON private value at {'.'.join(trail)}"
            )
        if isinstance(child, Mapping):
            for raw_key, nested in child.items():
                key = str(raw_key)
                if public and key in _FORBIDDEN_PUBLIC_KEYS:
                    raise ModelSelectionEvaluationError(
                        f"public aggregate exposes forbidden field: {key}"
                    )
                visit(nested, (*trail, key))
            return
        if isinstance(child, (list, tuple)):
            for index, nested in enumerate(child):
                visit(nested, (*trail, str(index)))
            return
        if child is not None and not isinstance(child, (str, bool, int, float)):
            raise ModelSelectionEvaluationError(
                f"aggregate output contains unsupported JSON value at {'.'.join(trail)}"
            )
        if isinstance(child, float) and not math.isfinite(child):
            raise ModelSelectionEvaluationError(
                f"aggregate output contains non-finite number at {'.'.join(trail)}"
            )
        if public and isinstance(child, str):
            if child.startswith(("/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", child):
                raise ModelSelectionEvaluationError(
                    f"public aggregate contains a local path at {'.'.join(trail)}"
                )

    visit(value, ("root",))


def validate_model_selection_public_report(payload: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "status",
        "claim_boundary",
        "dataset",
        "role",
        "primary_variant",
        "analysis_contract",
        "counts",
        "reference_freeze",
        "method_metrics",
        "holm_family",
        "accuracy_no_harm",
        "history_harm_rate_reduction",
        "per_seed_success",
        "aggregate_gates",
        "capacity_control",
        "prospective_sensitivity",
        "stage_authorization",
        "public_artifact_policy",
    }
    if set(payload) != expected:
        raise ModelSelectionEvaluationError("public model-selection report schema changed")
    if (
        payload.get("schema_version") != PUBLIC_REPORT_SCHEMA
        or payload.get("status")
        != "complete_non_confirmatory_single_dataset_reference_freeze"
        or payload.get("role") != SELECTION_ROLE
        or payload.get("primary_variant") != PRIMARY_VARIANT
        or payload.get("dataset") not in _SPECS
    ):
        raise ModelSelectionEvaluationError("public model-selection report identity changed")
    policy = payload.get("public_artifact_policy")
    if not isinstance(policy, Mapping) or policy != {
        "aggregate_only": True,
        "contains_row_level_values": False,
        "contains_private_paths": False,
        "confirmatory_claim_authorized": False,
        "calibration_holdout_or_test_access_authorized": False,
    }:
        raise ModelSelectionEvaluationError("public aggregate privacy policy changed")
    authorization = payload.get("stage_authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(name) is not False
        for name in (
            "calibration_unseal_authorized",
            "internal_holdout_unseal_authorized",
            "external_test_unseal_authorized",
        )
    ):
        raise ModelSelectionEvaluationError("public report improperly authorizes a later role")
    holm = payload.get("holm_family")
    if not isinstance(holm, Mapping) or (
        holm.get("hypothesis_order") != [row[0] for row in _HOLM_SPEC]
        or holm.get("capacity_control_included") is not False
    ):
        raise ModelSelectionEvaluationError("public Holm family changed")
    sensitivity = payload.get("prospective_sensitivity")
    if not isinstance(sensitivity, Mapping) or (
        sensitivity.get("observed_effect_used_as_assumed_effect") is not False
        or sensitivity.get("observed_post_hoc_power_computed") is not False
        or float(sensitivity.get("assumed_effect_absolute", math.nan))
        != PROSPECTIVE_TARGET_MACRO_F1_GAIN
    ):
        raise ModelSelectionEvaluationError("public prospective sensitivity changed")
    harm = payload.get("history_harm_rate_reduction")
    if not isinstance(harm, Mapping) or (
        harm.get("reference_candidates")
        != [
            "all_history",
            "coverage_matched_recency",
            "forward_only_utility",
            "backward_only_utility",
        ]
        or harm.get("zero_reference_harm_rate_action")
        != "fail_closed_not_estimable"
        or float(harm.get("minimum_relative_reduction", math.nan)) != 0.05
    ):
        raise ModelSelectionEvaluationError("public history-harm gate changed")
    per_seed = payload.get("per_seed_success")
    if not isinstance(per_seed, Mapping) or (
        per_seed.get("seed_count") != 5
        or per_seed.get("required_successes") != 4
        or per_seed.get("same_seed_for_all_conditions") is not True
        or per_seed.get("seed_level_numeric_details_withheld") is not True
        or "seed_results" in per_seed
        or per_seed.get("success_requires_all")
        != [
            "macro_f1_candidate_strictly_greater_than_reference",
            "mean_regret_vs_current_non_positive",
        ]
    ):
        raise ModelSelectionEvaluationError("public per-seed success gate changed")
    _validate_json_safe_aggregate(payload, public=True)


def _handoff_mapping(
    value: object,
    label: str,
    *,
    exact_keys: frozenset[str] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelSelectionEvaluationError(f"{label} must be an object")
    if exact_keys is not None and set(value) != exact_keys:
        raise ModelSelectionEvaluationError(f"{label} schema changed")
    return cast(Mapping[str, object], value)


def _handoff_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ModelSelectionEvaluationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelSelectionEvaluationError(
            f"{label} must be a finite number"
        ) from error
    if not math.isfinite(result):
        raise ModelSelectionEvaluationError(f"{label} must be a finite number")
    return result


def _handoff_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ModelSelectionEvaluationError(f"{label} must be boolean")
    return value


def _handoff_ci95(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ModelSelectionEvaluationError(f"{label} must contain two bounds")
    low = _handoff_number(value[0], f"{label}[0]")
    high = _handoff_number(value[1], f"{label}[1]")
    if low > high:
        raise ModelSelectionEvaluationError(f"{label} bounds are reversed")
    return low, high


def _verified_contrast_summary(
    value: object,
    *,
    label: str,
    metric: str,
    alternative: str,
    has_multiplicity: bool,
) -> tuple[float, tuple[float, float], float]:
    keys = {
        "metric",
        "alternative",
        "difference_definition",
        "point_difference",
        "favorable_direction_point",
        "ci95_percentile",
        "bootstrap_design",
        "bootstrap_replicates",
        "bootstrap_seed",
        "cluster_count",
        "queries_within_cluster_kept_together",
        "independent_query_resampling",
        "hypothesis_test",
    }
    if has_multiplicity:
        keys.add("multiplicity")
    else:
        keys.update(
            {
                "minimum_point_difference",
                "minimum_ci95_lower",
                "passed",
                "noninferiority_is_not_improvement_evidence",
            }
        )
    result = _handoff_mapping(value, label, exact_keys=frozenset(keys))
    if (
        result.get("metric") != metric
        or result.get("alternative") != alternative
        or result.get("difference_definition") != "candidate_minus_reference"
        or result.get("bootstrap_design")
        != "five_training_seeds_crossed_with_shared_whole_cluster_draw"
        or result.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or result.get("bootstrap_seed") != BOOTSTRAP_SEED
        or result.get("queries_within_cluster_kept_together") is not True
        or result.get("independent_query_resampling") is not False
    ):
        raise ModelSelectionEvaluationError(f"{label} contrast contract changed")
    point = _handoff_number(result.get("point_difference"), f"{label}.point_difference")
    favorable = _handoff_number(
        result.get("favorable_direction_point"),
        f"{label}.favorable_direction_point",
    )
    expected_favorable = point if alternative == "greater" else -point
    if not math.isclose(favorable, expected_favorable, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ModelSelectionEvaluationError(f"{label} favorable direction changed")
    ci95 = _handoff_ci95(result.get("ci95_percentile"), f"{label}.ci95_percentile")
    cluster_count_value = result.get("cluster_count")
    if (
        isinstance(cluster_count_value, bool)
        or not isinstance(cluster_count_value, int)
        or cluster_count_value < 2
    ):
        raise ModelSelectionEvaluationError(f"{label} cluster count is invalid")

    test = _handoff_mapping(
        result.get("hypothesis_test"),
        f"{label}.hypothesis_test",
        exact_keys=frozenset(
            {
                "point_difference",
                "favorable_direction_point",
                "paired_whole_cluster_randomization_p_value",
                "test_design",
                "sharp_null",
                "seed_resampling_in_hypothesis_test",
                "one_swap_shared_across_five_seeds",
                "queries_within_cluster_kept_together",
                "nonlinear_metric_recomputed_each_assignment",
                "exact_enumeration",
                "assignment_count",
                "randomization_seed",
                "cluster_count",
            }
        ),
    )
    if (
        test.get("test_design")
        != "paired_whole_cluster_swap_shared_across_all_five_seeds"
        or test.get("sharp_null")
        != "candidate_reference_exchangeable_within_each_independent_cluster"
        or test.get("seed_resampling_in_hypothesis_test") is not False
        or test.get("one_swap_shared_across_five_seeds") is not True
        or test.get("queries_within_cluster_kept_together") is not True
        or test.get("nonlinear_metric_recomputed_each_assignment")
        is not (metric in {"macro_f1", "accuracy"})
        or test.get("cluster_count") != cluster_count_value
    ):
        raise ModelSelectionEvaluationError(f"{label} randomization contract changed")
    test_point = _handoff_number(
        test.get("point_difference"), f"{label}.hypothesis_test.point_difference"
    )
    test_favorable = _handoff_number(
        test.get("favorable_direction_point"),
        f"{label}.hypothesis_test.favorable_direction_point",
    )
    if not math.isclose(test_point, point, rel_tol=1.0e-12, abs_tol=1.0e-12) or not math.isclose(
        test_favorable, favorable, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ModelSelectionEvaluationError(f"{label} randomization estimand changed")
    p_value = _handoff_number(
        test.get("paired_whole_cluster_randomization_p_value"),
        f"{label}.hypothesis_test.p_value",
    )
    if not 0.0 <= p_value <= 1.0:
        raise ModelSelectionEvaluationError(f"{label} randomization p-value is invalid")
    exact = _handoff_bool(
        test.get("exact_enumeration"), f"{label}.hypothesis_test.exact_enumeration"
    )
    assignment_count = test.get("assignment_count")
    if isinstance(assignment_count, bool) or not isinstance(assignment_count, int):
        raise ModelSelectionEvaluationError(f"{label} assignment count is invalid")
    if exact:
        if (
            cluster_count_value > EXACT_RANDOMIZATION_MAX_CLUSTERS
            or assignment_count != 1 << cluster_count_value
            or test.get("randomization_seed") is not None
        ):
            raise ModelSelectionEvaluationError(f"{label} exact randomization changed")
    elif (
        cluster_count_value <= EXACT_RANDOMIZATION_MAX_CLUSTERS
        or assignment_count != RANDOMIZATION_REPLICATES
        or test.get("randomization_seed") != RANDOMIZATION_SEED
    ):
        raise ModelSelectionEvaluationError(f"{label} Monte Carlo randomization changed")
    return point, ci95, p_value


def _derive_model_selection_handoff(
    evaluation_value: object,
) -> tuple[str, bool, float, bool]:
    """Validate hash-bound aggregate semantics and derive all handoff gates."""

    evaluation = _handoff_mapping(
        evaluation_value,
        "evaluation",
        exact_keys=frozenset(
            {
                "counts",
                "reference_freeze",
                "method_metrics",
                "holm_family",
                "accuracy_no_harm",
                "history_harm_rate_reduction",
                "per_seed_success",
                "aggregate_gates",
                "capacity_control",
                "prospective_sensitivity",
                "stage_authorization",
            }
        ),
    )
    reference = _handoff_mapping(
        evaluation.get("reference_freeze"),
        "evaluation.reference_freeze",
        exact_keys=frozenset(
            {
                "frozen_reference",
                "candidate_roster",
                "selection_rule",
                "ranking",
                "frozen_before_calibration_unseal",
            }
        ),
    )
    frozen_reference = str(reference.get("frozen_reference"))
    if (
        frozen_reference not in REFERENCE_CANDIDATES
        or reference.get("candidate_roster") != list(REFERENCE_CANDIDATES)
        or reference.get("selection_rule")
        != (
            "highest_five_seed_mean_model_selection_macro_f1_then_highest_"
            "accuracy_then_lowest_mean_regret_then_lexicographic_model_id"
        )
        or reference.get("frozen_before_calibration_unseal") is not True
    ):
        raise ModelSelectionEvaluationError("reference-freeze aggregate changed")
    ranking_value = reference.get("ranking")
    if not isinstance(ranking_value, list) or len(ranking_value) != len(REFERENCE_CANDIDATES):
        raise ModelSelectionEvaluationError("reference-freeze ranking changed")
    ranking: list[dict[str, float | int | str]] = []
    for expected_rank, value in enumerate(ranking_value, start=1):
        row = _handoff_mapping(
            value,
            f"evaluation.reference_freeze.ranking[{expected_rank - 1}]",
            exact_keys=frozenset({"method", "macro_f1", "accuracy", "mean_regret", "rank"}),
        )
        method = str(row.get("method"))
        if method not in REFERENCE_CANDIDATES or row.get("rank") != expected_rank:
            raise ModelSelectionEvaluationError("reference-freeze ranking changed")
        ranking.append(
            {
                "method": method,
                "macro_f1": _handoff_number(row.get("macro_f1"), "ranking.macro_f1"),
                "accuracy": _handoff_number(row.get("accuracy"), "ranking.accuracy"),
                "mean_regret": _handoff_number(row.get("mean_regret"), "ranking.mean_regret"),
                "rank": expected_rank,
            }
        )
    if (
        {str(row["method"]) for row in ranking} != set(REFERENCE_CANDIDATES)
        or ranking
        != sorted(
            ranking,
            key=lambda row: (
                -float(row["macro_f1"]),
                -float(row["accuracy"]),
                float(row["mean_regret"]),
                str(row["method"]),
            ),
        )
        or ranking[0]["method"] != frozen_reference
    ):
        raise ModelSelectionEvaluationError("reference-freeze ranking is inconsistent")

    holm = _handoff_mapping(
        evaluation.get("holm_family"),
        "evaluation.holm_family",
        exact_keys=frozenset(
            {
                "family_id",
                "method",
                "familywise_alpha",
                "hypothesis_order",
                "capacity_control_included",
                "results",
                "interpretation",
            }
        ),
    )
    declared_order = [row[0] for row in _HOLM_SPEC]
    if (
        holm.get("family_id") != "carma_confirmatory_claim_family_v1"
        or holm.get("method") != "holm_bonferroni"
        or _handoff_number(holm.get("familywise_alpha"), "holm.familywise_alpha")
        != RANDOMIZATION_ALPHA
        or holm.get("hypothesis_order") != declared_order
        or holm.get("capacity_control_included") is not False
        or holm.get("interpretation")
        != "non_confirmatory_model_selection_diagnostic_only"
    ):
        raise ModelSelectionEvaluationError("Holm aggregate contract changed")
    holm_results = _handoff_mapping(
        holm.get("results"),
        "evaluation.holm_family.results",
        exact_keys=frozenset(declared_order),
    )
    raw_p_values: dict[str, float] = {}
    contrast_points: dict[str, float] = {}
    contrast_cis: dict[str, tuple[float, float]] = {}
    for hypothesis_id, _contrast, metric, alternative in _HOLM_SPEC:
        point, ci95, p_value = _verified_contrast_summary(
            holm_results[hypothesis_id],
            label=f"evaluation.holm_family.results.{hypothesis_id}",
            metric=metric,
            alternative=alternative,
            has_multiplicity=True,
        )
        contrast_points[hypothesis_id] = point
        contrast_cis[hypothesis_id] = ci95
        raw_p_values[hypothesis_id] = p_value
    recomputed_holm = holm_bonferroni(
        raw_p_values,
        declared_order=declared_order,
        alpha=RANDOMIZATION_ALPHA,
    )
    for hypothesis_id in declared_order:
        result = cast(Mapping[str, object], holm_results[hypothesis_id])
        multiplicity = _handoff_mapping(
            result.get("multiplicity"),
            f"evaluation.holm_family.results.{hypothesis_id}.multiplicity",
            exact_keys=frozenset(
                {
                    "raw_p_value",
                    "holm_adjusted_p_value",
                    "holm_rank",
                    "rejected_at_familywise_alpha",
                }
            ),
        )
        expected = recomputed_holm[hypothesis_id]
        if (
            not math.isclose(
                _handoff_number(multiplicity.get("raw_p_value"), "multiplicity.raw_p_value"),
                float(expected["raw_p_value"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _handoff_number(
                    multiplicity.get("holm_adjusted_p_value"),
                    "multiplicity.holm_adjusted_p_value",
                ),
                float(expected["holm_adjusted_p_value"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or isinstance(multiplicity.get("holm_rank"), bool)
            or not isinstance(multiplicity.get("holm_rank"), int)
            or multiplicity.get("holm_rank") != expected["holm_rank"]
            or multiplicity.get("rejected_at_familywise_alpha")
            is not expected["rejected_at_familywise_alpha"]
        ):
            raise ModelSelectionEvaluationError("Holm adjustment is inconsistent")
    primary_holm_passed = bool(
        recomputed_holm["H1_primary_macro_f1"]["rejected_at_familywise_alpha"]
        and recomputed_holm["H2_primary_mean_regret"]["rejected_at_familywise_alpha"]
    )

    accuracy = _handoff_mapping(
        evaluation.get("accuracy_no_harm"),
        "evaluation.accuracy_no_harm",
        exact_keys=frozenset({"contrast_order", "results", "all_passed", "interpretation"}),
    )
    if (
        accuracy.get("contrast_order") != list(_ACCURACY_IDS)
        or accuracy.get("interpretation")
        != "non_confirmatory_model_selection_diagnostic_only"
    ):
        raise ModelSelectionEvaluationError("accuracy no-harm aggregate changed")
    accuracy_results = _handoff_mapping(
        accuracy.get("results"),
        "evaluation.accuracy_no_harm.results",
        exact_keys=frozenset(_ACCURACY_IDS),
    )
    accuracy_passes: list[bool] = []
    for contrast_id in _ACCURACY_IDS:
        point, ci95, _p_value = _verified_contrast_summary(
            accuracy_results[contrast_id],
            label=f"evaluation.accuracy_no_harm.results.{contrast_id}",
            metric="accuracy",
            alternative="greater",
            has_multiplicity=False,
        )
        result = cast(Mapping[str, object], accuracy_results[contrast_id])
        expected_pass = bool(
            point >= ACCURACY_NO_HARM_POINT_MINIMUM
            and ci95[0] >= ACCURACY_NO_HARM_CI95_LOWER_MINIMUM
        )
        if (
            _handoff_number(
                result.get("minimum_point_difference"),
                f"{contrast_id}.minimum_point_difference",
            )
            != ACCURACY_NO_HARM_POINT_MINIMUM
            or _handoff_number(
                result.get("minimum_ci95_lower"), f"{contrast_id}.minimum_ci95_lower"
            )
            != ACCURACY_NO_HARM_CI95_LOWER_MINIMUM
            or _handoff_bool(result.get("passed"), f"{contrast_id}.passed")
            != expected_pass
            or result.get("noninferiority_is_not_improvement_evidence") is not True
        ):
            raise ModelSelectionEvaluationError("accuracy no-harm result is inconsistent")
        accuracy_passes.append(expected_pass)
    accuracy_gate_passed = bool(all(accuracy_passes))
    if _handoff_bool(accuracy.get("all_passed"), "accuracy_no_harm.all_passed") != accuracy_gate_passed:
        raise ModelSelectionEvaluationError("accuracy no-harm aggregate is inconsistent")

    harm = _handoff_mapping(
        evaluation.get("history_harm_rate_reduction"),
        "evaluation.history_harm_rate_reduction",
        exact_keys=frozenset(
            {
                "candidate",
                "reference",
                "reference_candidates",
                "reference_selection_rule",
                "reference_ranking",
                "candidate_five_seed_mean_history_harm_rate",
                "reference_five_seed_mean_history_harm_rate",
                "relative_history_harm_rate_reduction",
                "minimum_relative_reduction",
                "estimable",
                "passed",
                "zero_reference_harm_rate_action",
                "failure_reason",
            }
        ),
    )
    history_candidates = [
        "all_history",
        "coverage_matched_recency",
        "forward_only_utility",
        "backward_only_utility",
    ]
    expected_history_ranking = []
    for row in ranking:
        if str(row["method"]) in history_candidates:
            expected_history_ranking.append(
                {
                    **row,
                    "rank": len(expected_history_ranking) + 1,
                }
            )
    observed_history_ranking = harm.get("reference_ranking")
    if (
        harm.get("candidate") != "carma_bidirectional_full"
        or harm.get("reference") != expected_history_ranking[0]["method"]
        or harm.get("reference_candidates") != history_candidates
        or observed_history_ranking != expected_history_ranking
        or harm.get("reference_selection_rule")
        != (
            "highest_five_seed_mean_model_selection_macro_f1_then_highest_"
            "accuracy_then_lowest_mean_regret_then_lexicographic_model_id"
        )
        or _handoff_number(
            harm.get("minimum_relative_reduction"), "harm.minimum_relative_reduction"
        )
        != 0.05
        or harm.get("zero_reference_harm_rate_action")
        != "fail_closed_not_estimable"
    ):
        raise ModelSelectionEvaluationError("history-harm aggregate changed")
    candidate_harm = _handoff_number(
        harm.get("candidate_five_seed_mean_history_harm_rate"), "harm.candidate_rate"
    )
    reference_harm = _handoff_number(
        harm.get("reference_five_seed_mean_history_harm_rate"), "harm.reference_rate"
    )
    if not 0.0 <= candidate_harm <= 1.0 or not 0.0 <= reference_harm <= 1.0:
        raise ModelSelectionEvaluationError("history-harm rate is invalid")
    harm_estimable = reference_harm > 0.0
    expected_harm_reduction = (
        (reference_harm - candidate_harm) / reference_harm if harm_estimable else None
    )
    observed_harm_reduction = harm.get("relative_history_harm_rate_reduction")
    if harm_estimable:
        if not math.isclose(
            _handoff_number(observed_harm_reduction, "harm.relative_reduction"),
            cast(float, expected_harm_reduction),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or harm.get("failure_reason") is not None:
            raise ModelSelectionEvaluationError("history-harm reduction is inconsistent")
    elif (
        observed_harm_reduction is not None
        or harm.get("failure_reason")
        != "zero_reference_harm_rate_fail_closed_not_estimable"
    ):
        raise ModelSelectionEvaluationError("zero history-harm denominator did not fail closed")
    expected_harm_pass = bool(
        harm_estimable and cast(float, expected_harm_reduction) >= 0.05
    )
    if (
        _handoff_bool(harm.get("estimable"), "harm.estimable") != harm_estimable
        or _handoff_bool(harm.get("passed"), "harm.passed") != expected_harm_pass
    ):
        raise ModelSelectionEvaluationError("history-harm gate is inconsistent")

    per_seed = _handoff_mapping(
        evaluation.get("per_seed_success"),
        "evaluation.per_seed_success",
        exact_keys=frozenset(
            {
                "candidate",
                "reference",
                "seed_count",
                "required_successes",
                "same_seed_for_all_conditions",
                "success_requires_all",
                "macro_f1_difference_strictly_greater_than",
                "mean_regret_vs_current_must_not_exceed",
                "seed_results",
                "success_count",
                "passed",
                "aggregate_effect_ci_harm_and_accuracy_gates_are_independent",
            }
        ),
    )
    if (
        per_seed.get("candidate") != "carma_bidirectional_full"
        or per_seed.get("reference") != frozen_reference
        or per_seed.get("seed_count") != 5
        or per_seed.get("required_successes") != 4
        or per_seed.get("same_seed_for_all_conditions") is not True
        or per_seed.get("success_requires_all")
        != [
            "macro_f1_candidate_strictly_greater_than_reference",
            "mean_regret_vs_current_non_positive",
        ]
        or _handoff_number(
            per_seed.get("macro_f1_difference_strictly_greater_than"),
            "per_seed.macro_f1_threshold",
        )
        != 0.0
        or _handoff_number(
            per_seed.get("mean_regret_vs_current_must_not_exceed"),
            "per_seed.mean_regret_threshold",
        )
        != 0.0
        or per_seed.get("aggregate_effect_ci_harm_and_accuracy_gates_are_independent")
        is not True
    ):
        raise ModelSelectionEvaluationError("per-seed aggregate contract changed")
    seed_results = per_seed.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != 5:
        raise ModelSelectionEvaluationError("per-seed results changed")
    derived_seed_successes = 0
    for seed_position, value in enumerate(seed_results):
        row = _handoff_mapping(
            value,
            f"per_seed.seed_results[{seed_position}]",
            exact_keys=frozenset(
                {
                    "seed_position",
                    "macro_f1_candidate_minus_reference",
                    "mean_regret_candidate_vs_current",
                    "macro_f1_condition_passed",
                    "mean_regret_condition_passed",
                    "same_seed_joint_success",
                }
            ),
        )
        macro_passed = _handoff_number(
            row.get("macro_f1_candidate_minus_reference"), "per_seed.macro_difference"
        ) > 0.0
        regret_passed = _handoff_number(
            row.get("mean_regret_candidate_vs_current"), "per_seed.mean_regret"
        ) <= 0.0
        joint_passed = bool(macro_passed and regret_passed)
        if (
            row.get("seed_position") != seed_position
            or _handoff_bool(row.get("macro_f1_condition_passed"), "per_seed.macro_passed")
            != macro_passed
            or _handoff_bool(
                row.get("mean_regret_condition_passed"), "per_seed.regret_passed"
            )
            != regret_passed
            or _handoff_bool(row.get("same_seed_joint_success"), "per_seed.joint_passed")
            != joint_passed
        ):
            raise ModelSelectionEvaluationError("per-seed result is inconsistent")
        derived_seed_successes += int(joint_passed)
    per_seed_gate_passed = derived_seed_successes >= 4
    if (
        per_seed.get("success_count") != derived_seed_successes
        or _handoff_bool(per_seed.get("passed"), "per_seed.passed")
        != per_seed_gate_passed
    ):
        raise ModelSelectionEvaluationError("per-seed gate is inconsistent")

    gates = _handoff_mapping(
        evaluation.get("aggregate_gates"),
        "evaluation.aggregate_gates",
        exact_keys=frozenset(
            {
                "minimum_macro_f1_gain_absolute",
                *_MODEL_SELECTION_GATE_KEYS,
                "mean_regret_vs_frozen_reference",
                "gates_are_conjunctive_and_independently_computed",
            }
        ),
    )
    frozen_regret = _handoff_mapping(
        gates.get("mean_regret_vs_frozen_reference"),
        "aggregate_gates.mean_regret_vs_frozen_reference",
        exact_keys=frozenset(
            {
                "metric",
                "difference_definition",
                "point_difference",
                "ci95_percentile",
                "ci95_upper_must_not_exceed",
                "passed",
                "bootstrap_design",
                "bootstrap_replicates",
                "bootstrap_seed",
                "cluster_count",
                "included_in_holm_family",
                "prespecified_safety_gate_not_unlisted_inferential_claim",
            }
        ),
    )
    frozen_regret_ci = _handoff_ci95(
        frozen_regret.get("ci95_percentile"),
        "aggregate_gates.mean_regret_vs_frozen_reference.ci95_percentile",
    )
    frozen_regret_passed = frozen_regret_ci[1] <= 0.0
    if (
        frozen_regret.get("metric") != "mean_regret"
        or frozen_regret.get("difference_definition")
        != "candidate_minus_frozen_reference"
        or _handoff_number(
            frozen_regret.get("ci95_upper_must_not_exceed"),
            "frozen_regret.ci95_upper_must_not_exceed",
        )
        != 0.0
        or _handoff_bool(frozen_regret.get("passed"), "frozen_regret.passed")
        != frozen_regret_passed
        or frozen_regret.get("bootstrap_design")
        != "five_training_seeds_crossed_with_shared_whole_cluster_draw"
        or frozen_regret.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or frozen_regret.get("bootstrap_seed") != BOOTSTRAP_SEED
        or frozen_regret.get("included_in_holm_family") is not False
        or frozen_regret.get("prespecified_safety_gate_not_unlisted_inferential_claim")
        is not True
    ):
        raise ModelSelectionEvaluationError("frozen-reference regret gate changed")
    expected_gates = {
        "macro_f1_point_gain_passed": contrast_points["H1_primary_macro_f1"]
        >= MINIMUM_MACRO_F1_GAIN,
        "macro_f1_ci95_lower_above_zero_passed": contrast_cis[
            "H1_primary_macro_f1"
        ][0]
        > 0.0,
        "mean_regret_vs_current_ci95_upper_non_positive_passed": contrast_cis[
            "H2_primary_mean_regret"
        ][1]
        <= 0.0,
        "mean_regret_vs_frozen_reference_ci95_upper_non_positive_passed": (
            frozen_regret_passed
        ),
        "history_harm_rate_reduction_passed": expected_harm_pass,
        "accuracy_no_harm_passed": accuracy_gate_passed,
        "per_seed_success_passed": per_seed_gate_passed,
    }
    if (
        _handoff_number(
            gates.get("minimum_macro_f1_gain_absolute"),
            "aggregate_gates.minimum_macro_f1_gain_absolute",
        )
        != MINIMUM_MACRO_F1_GAIN
        or gates.get("gates_are_conjunctive_and_independently_computed") is not True
    ):
        raise ModelSelectionEvaluationError("aggregate gate contract changed")
    for gate_name, expected_pass in expected_gates.items():
        if _handoff_bool(gates.get(gate_name), f"aggregate_gates.{gate_name}") != expected_pass:
            raise ModelSelectionEvaluationError(f"aggregate gate is inconsistent: {gate_name}")
    model_selection_gate_passed = bool(
        all(expected_gates.values()) and primary_holm_passed
    )

    sensitivity = _handoff_mapping(
        evaluation.get("prospective_sensitivity"),
        "evaluation.prospective_sensitivity",
        exact_keys=frozenset(
            {
                "analysis_type",
                "target_metric",
                "assumed_effect_absolute",
                "assumed_effect_source",
                "bootstrap_error",
                "bootstrap_error_replicates",
                "bootstrap_seed",
                "randomization_critical_value",
                "randomization_alpha",
                "randomization_alternative",
                "randomization_assignments",
                "randomization_exact_enumeration",
                "configured_randomization_replicates",
                "configured_randomization_seed",
                "estimated_power",
                "minimum_power",
                "power_gate_passed",
                "observed_effect_used_as_assumed_effect",
                "observed_post_hoc_power_computed",
                "underpowered_action",
            }
        ),
    )
    prospective_power = _handoff_number(
        sensitivity.get("estimated_power"), "prospective_sensitivity.estimated_power"
    )
    power_gate_passed = prospective_power >= MINIMUM_PROSPECTIVE_POWER
    if (
        not 0.0 <= prospective_power <= 1.0
        or sensitivity.get("analysis_type")
        != "prospective_design_sensitivity_not_observed_post_hoc_power"
        or sensitivity.get("target_metric") != "macro_f1"
        or _handoff_number(
            sensitivity.get("assumed_effect_absolute"),
            "prospective_sensitivity.assumed_effect_absolute",
        )
        != PROSPECTIVE_TARGET_MACRO_F1_GAIN
        or sensitivity.get("assumed_effect_source")
        != "frozen_minimum_meaningful_gain_not_observed_effect"
        or sensitivity.get("bootstrap_error")
        != "centered_whole_cluster_crossed_seed_bootstrap"
        or sensitivity.get("bootstrap_error_replicates") != BOOTSTRAP_REPLICATES
        or sensitivity.get("bootstrap_seed") != BOOTSTRAP_SEED
        or _handoff_number(
            sensitivity.get("randomization_alpha"),
            "prospective_sensitivity.randomization_alpha",
        )
        != RANDOMIZATION_ALPHA
        or sensitivity.get("randomization_alternative")
        != "two_sided_absolute_macro_f1_difference"
        or sensitivity.get("configured_randomization_replicates")
        != RANDOMIZATION_REPLICATES
        or sensitivity.get("configured_randomization_seed") != RANDOMIZATION_SEED
        or _handoff_number(
            sensitivity.get("minimum_power"), "prospective_sensitivity.minimum_power"
        )
        != MINIMUM_PROSPECTIVE_POWER
        or _handoff_bool(
            sensitivity.get("power_gate_passed"),
            "prospective_sensitivity.power_gate_passed",
        )
        != power_gate_passed
        or sensitivity.get("observed_effect_used_as_assumed_effect") is not False
        or sensitivity.get("observed_post_hoc_power_computed") is not False
        or sensitivity.get("underpowered_action")
        != (
            "eligible_only_for_separate_cross_dataset_freeze_review"
            if power_gate_passed
            else "do_not_unseal_calibration_holdout_or_external_test"
        )
    ):
        raise ModelSelectionEvaluationError("prospective sensitivity is inconsistent")
    _handoff_number(
        sensitivity.get("randomization_critical_value"),
        "prospective_sensitivity.randomization_critical_value",
    )
    randomization_assignments = sensitivity.get("randomization_assignments")
    if (
        isinstance(randomization_assignments, bool)
        or not isinstance(randomization_assignments, int)
        or randomization_assignments < 1
        or not isinstance(sensitivity.get("randomization_exact_enumeration"), bool)
    ):
        raise ModelSelectionEvaluationError("prospective sensitivity randomization changed")

    authorization = _handoff_mapping(
        evaluation.get("stage_authorization"),
        "evaluation.stage_authorization",
        exact_keys=frozenset(
            {
                "power_gate_passed",
                "calibration_unseal_authorized",
                "internal_holdout_unseal_authorized",
                "external_test_unseal_authorized",
                "reason",
                "next_required_authority",
            }
        ),
    )
    if (
        _handoff_bool(
            authorization.get("power_gate_passed"),
            "stage_authorization.power_gate_passed",
        )
        != power_gate_passed
        or any(
            authorization.get(name) is not False
            for name in (
                "calibration_unseal_authorized",
                "internal_holdout_unseal_authorized",
                "external_test_unseal_authorized",
            )
        )
        or authorization.get("reason")
        != (
            "single_dataset_model_selection_evaluator_has_no_unseal_authority"
            if power_gate_passed
            else "prospective_power_below_0.8_keep_all_later_roles_sealed"
        )
        or authorization.get("next_required_authority")
        != "hash_bound_two_required_dataset_joint_freeze_and_separate_stage_gate"
    ):
        raise ModelSelectionEvaluationError("single-dataset stage authorization changed")
    return (
        frozen_reference,
        model_selection_gate_passed,
        prospective_power,
        power_gate_passed,
    )


def _write_bytes_once(path: Path, encoded: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"write-once output already exists: {path.name}") from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(encoded).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _publish_outputs(
    *,
    strategies: VerifiedStrategyBundle,
    selection_capability: _VerifiedSelectionLabelCapability,
    analysis: FrozenAnalysisContract,
    aggregates: Mapping[str, object],
    private_output_root: str | Path,
    public_report_path: str | Path,
) -> CompletedModelSelectionEvaluation:
    (
        frozen_reference,
        model_selection_gate_passed,
        prospective_power,
        power_gate_passed,
    ) = _derive_model_selection_handoff(aggregates)
    analysis_contract = {
        "analysis_config_sha256": analysis.analysis_sha256,
        "split_manifest_sha256": analysis.split_manifest_sha256,
        "family_id": analysis.family_id,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "randomization_replicates": RANDOMIZATION_REPLICATES,
        "randomization_seed": RANDOMIZATION_SEED,
        "holm_hypotheses": [row[0] for row in _HOLM_SPEC],
        "accuracy_contrasts": list(_ACCURACY_IDS),
    }
    private_per_seed = cast(Mapping[str, object], aggregates["per_seed_success"])
    public_per_seed = {
        name: value
        for name, value in private_per_seed.items()
        if name != "seed_results"
    }
    public_per_seed["seed_level_numeric_details_withheld"] = True
    public_report: dict[str, object] = {
        "schema_version": PUBLIC_REPORT_SCHEMA,
        "status": "complete_non_confirmatory_single_dataset_reference_freeze",
        "claim_boundary": (
            "Model-selection-only reference freeze and prospective sensitivity; "
            "not confirmatory performance evidence."
        ),
        "dataset": strategies.dataset,
        "role": SELECTION_ROLE,
        "primary_variant": PRIMARY_VARIANT,
        "analysis_contract": analysis_contract,
        "counts": aggregates["counts"],
        "reference_freeze": aggregates["reference_freeze"],
        "method_metrics": aggregates["method_metrics"],
        "holm_family": aggregates["holm_family"],
        "accuracy_no_harm": aggregates["accuracy_no_harm"],
        "history_harm_rate_reduction": aggregates[
            "history_harm_rate_reduction"
        ],
        "per_seed_success": public_per_seed,
        "aggregate_gates": aggregates["aggregate_gates"],
        "capacity_control": aggregates["capacity_control"],
        "prospective_sensitivity": aggregates["prospective_sensitivity"],
        "stage_authorization": aggregates["stage_authorization"],
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "confirmatory_claim_authorized": False,
            "calibration_holdout_or_test_access_authorized": False,
        },
    }
    validate_model_selection_public_report(public_report)
    public_bytes = _json_bytes(public_report)
    public_sha = hashlib.sha256(public_bytes).hexdigest()

    private_artifact: dict[str, object] = {
        "schema_version": PRIVATE_ARTIFACT_SCHEMA,
        "status": "complete_single_dataset_reference_freeze_not_confirmatory_evidence",
        "claim_boundary": (
            "Private aggregate reference-freeze lineage; no row-level outcome, "
            "prediction, probability, identifier, or context is stored."
        ),
        "dataset": strategies.dataset,
        "role": SELECTION_ROLE,
        "primary_variant": PRIMARY_VARIANT,
        "analysis_contract": analysis_contract,
        "lineage": {
            "model_selection_manifest_sha256": selection_capability.manifest_sha256,
            "model_selection_preflight_receipt_sha256": (
                selection_capability.preflight_receipt_sha256
            ),
            "model_selection_outcome_file_sha256": selection_capability.label_sha256,
            "model_selection_row_alignment_sha256": (
                selection_capability.row_alignment_sha256
            ),
            "cross_variant_alignment_sha256": (
                strategies.cross_variant_alignment_sha256
            ),
            "full_current_anchor_history_artifact_sha256": (
                strategies.full_current_anchor_history_artifact_sha256
            ),
            "current_artifact_sha256": strategies.current_artifact_sha256,
            "strategy_config_roster_sha256": strategies.strategy_config_roster_sha256,
            "common_strategy_code_bundle_sha256": (
                strategies.common_strategy_code_bundle_sha256
            ),
            "common_strategy_runtime_environment_sha256": (
                strategies.common_strategy_runtime_environment_sha256
            ),
            "strategy_artifact_sha256_by_variant": {
                variant: strategies.attestations[variant].artifact_sha256
                for variant in REGISTERED_VARIANTS
            },
            "strategy_receipt_sha256_by_variant": {
                variant: strategies.attestations[variant].receipt_sha256
                for variant in REGISTERED_VARIANTS
            },
            "strategy_production_claim_sha256_by_variant": {
                variant: strategies.attestations[variant].production_run_claim_sha256
                for variant in REGISTERED_VARIANTS
            },
        },
        "evaluation": dict(aggregates),
        "public_report_sha256": public_sha,
        "aggregate_handoff_contract": {
            "single_dataset_only": True,
            "accepted_by_future_joint_evaluator_via_receipt_verifier": True,
            "required_joint_dataset_roster": ["EmotionTalk", "MELD"],
            "single_dataset_may_authorize_method_success": False,
        },
        "private_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_outcome_file_hash_for_lineage_only": True,
        },
    }
    _validate_json_safe_aggregate(private_artifact, public=False)
    private_bytes = _json_bytes(private_artifact)
    private_sha = hashlib.sha256(private_bytes).hexdigest()
    receipt: dict[str, object] = {
        "schema_version": PRIVATE_RECEIPT_SCHEMA,
        "status": "complete_single_dataset_aggregate_reference_freeze_receipt",
        "dataset": strategies.dataset,
        "claim_boundary": (
            "Aggregate-only single-dataset handoff; no authority to open later roles."
        ),
        "lineage": {
            "private_reference_freeze_artifact_sha256": private_sha,
            "public_aggregate_report_sha256": public_sha,
            "analysis_config_sha256": analysis.analysis_sha256,
            "cross_variant_alignment_sha256": (
                strategies.cross_variant_alignment_sha256
            ),
        },
        "completion_contract": {
            "primary_variant": PRIMARY_VARIANT,
            "registered_variants": list(REGISTERED_VARIANTS),
            "frozen_reference": frozen_reference,
            "reference_candidates": list(REFERENCE_CANDIDATES),
            "model_selection_gate_passed": model_selection_gate_passed,
            "prospective_power": prospective_power,
            "minimum_prospective_power": MINIMUM_PROSPECTIVE_POWER,
            "power_gate_passed": power_gate_passed,
            "single_dataset_only": True,
            "aggregate_only": True,
            "confirmatory_claim_authorized": False,
            "calibration_unseal_authorized": False,
            "internal_holdout_unseal_authorized": False,
            "external_test_unseal_authorized": False,
        },
        "joint_evaluator_handoff": {
            "verifier": "verify_model_selection_reference_freeze_receipt",
            "required_dataset_roster": ["EmotionTalk", "MELD"],
            "hash_bound_artifact_and_receipt_required": True,
        },
    }
    _validate_json_safe_aggregate(receipt, public=False)
    receipt_bytes = _json_bytes(receipt)

    root = Path(private_output_root).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    public_path = Path(public_report_path).resolve()
    if (
        not root.is_absolute()
        or root == Path(root.anchor)
        or _is_within(root, repository_root)
    ):
        raise ModelSelectionEvaluationError(
            "private reference-freeze root must be a new external directory"
        )
    if public_path.suffix.lower() != ".json" or _is_within(public_path, root):
        raise ModelSelectionEvaluationError(
            "public report must be a separate JSON artifact"
        )
    if root.exists():
        raise FileExistsError("private reference-freeze root already exists")
    if public_path.exists():
        raise FileExistsError("public model-selection report already exists")
    root.mkdir(parents=True, exist_ok=False)
    artifact_path = root / PRIVATE_ARTIFACT_NAME
    receipt_path = root / PRIVATE_RECEIPT_NAME
    observed_private_sha = _write_bytes_once(artifact_path, private_bytes)
    observed_receipt_sha = _write_bytes_once(receipt_path, receipt_bytes)
    observed_public_sha = _write_bytes_once(public_path, public_bytes)
    if observed_private_sha != private_sha or observed_public_sha != public_sha:
        raise AssertionError("published model-selection output hash changed")
    return CompletedModelSelectionEvaluation(
        private_artifact_path=artifact_path,
        private_artifact_sha256=private_sha,
        private_receipt_path=receipt_path,
        private_receipt_sha256=observed_receipt_sha,
        public_report_path=public_path,
        public_report_sha256=public_sha,
        frozen_reference=frozen_reference,
        model_selection_gate_passed=model_selection_gate_passed,
        prospective_power=prospective_power,
        power_gate_passed=power_gate_passed,
    )


def run_model_selection_reference_freeze(
    *,
    strategies: Mapping[str, StrategyProductionInput],
    selection_source: SelectionSidecarSource,
    confirmatory_analysis_path: str | Path,
    private_output_root: str | Path,
    public_report_path: str | Path,
) -> CompletedModelSelectionEvaluation:
    """Run one dataset's fixed model-selection evaluation in fail-closed order."""

    analysis = _load_frozen_analysis_contract(confirmatory_analysis_path)
    strategy_bundle = verify_strategy_bundle_before_label_access(strategies)
    evaluation_inputs = _collect_evaluation_inputs(strategy_bundle)
    selection_capability = _verify_selection_label_capability(
        selection_source, strategy_bundle
    )
    labels = _load_model_selection_labels_once(
        selection_capability, strategy_bundle
    )
    aggregates = _evaluate_model_selection_aggregates(
        labels=labels,
        inputs=evaluation_inputs,
        analysis=analysis,
    )
    return _publish_outputs(
        strategies=strategy_bundle,
        selection_capability=selection_capability,
        analysis=analysis,
        aggregates=aggregates,
        private_output_root=private_output_root,
        public_report_path=public_report_path,
    )


def verify_model_selection_reference_freeze_receipt(
    artifact_path: str | Path,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
) -> VerifiedModelSelectionAggregateAttestation:
    """Verify an aggregate single-dataset handoff without any outcome capability."""

    artifact_file = Path(artifact_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    if (
        artifact_file.name != PRIVATE_ARTIFACT_NAME
        or receipt_file.name != PRIVATE_RECEIPT_NAME
        or artifact_file.parent != receipt_file.parent
    ):
        raise ModelSelectionEvaluationError("reference-freeze paths are not canonical")
    expected_receipt = _require_sha256(
        expected_receipt_sha256, "expected_receipt_sha256"
    )
    if _file_sha256(receipt_file) != expected_receipt:
        raise ModelSelectionEvaluationError("reference-freeze receipt hash changed")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelSelectionEvaluationError(
            f"cannot read reference-freeze handoff: {error}"
        ) from error
    if not isinstance(receipt, Mapping) or not isinstance(artifact, Mapping):
        raise ModelSelectionEvaluationError("reference-freeze handoff root changed")
    if set(receipt) != {
        "schema_version",
        "status",
        "dataset",
        "claim_boundary",
        "lineage",
        "completion_contract",
        "joint_evaluator_handoff",
    } or set(artifact) != {
        "schema_version",
        "status",
        "claim_boundary",
        "dataset",
        "role",
        "primary_variant",
        "analysis_contract",
        "lineage",
        "evaluation",
        "public_report_sha256",
        "aggregate_handoff_contract",
        "private_artifact_policy",
    }:
        raise ModelSelectionEvaluationError("reference-freeze handoff schema changed")
    if (
        receipt.get("schema_version") != PRIVATE_RECEIPT_SCHEMA
        or artifact.get("schema_version") != PRIVATE_ARTIFACT_SCHEMA
        or receipt.get("dataset") != artifact.get("dataset")
        or artifact.get("role") != SELECTION_ROLE
        or artifact.get("primary_variant") != PRIMARY_VARIANT
    ):
        raise ModelSelectionEvaluationError("reference-freeze handoff identity changed")
    lineage = receipt.get("lineage")
    completion = receipt.get("completion_contract")
    artifact_analysis = artifact.get("analysis_contract")
    artifact_lineage = artifact.get("lineage")
    if not all(
        isinstance(value, Mapping)
        for value in (lineage, completion, artifact_analysis, artifact_lineage)
    ):
        raise ModelSelectionEvaluationError("reference-freeze handoff contract is absent")
    lineage = cast(Mapping[str, object], lineage)
    completion = cast(Mapping[str, object], completion)
    artifact_analysis = cast(Mapping[str, object], artifact_analysis)
    artifact_lineage = cast(Mapping[str, object], artifact_lineage)
    completion = _handoff_mapping(
        completion,
        "receipt.completion_contract",
        exact_keys=frozenset(
            {
                "primary_variant",
                "registered_variants",
                "frozen_reference",
                "reference_candidates",
                "model_selection_gate_passed",
                "prospective_power",
                "minimum_prospective_power",
                "power_gate_passed",
                "single_dataset_only",
                "aggregate_only",
                "confirmatory_claim_authorized",
                "calibration_unseal_authorized",
                "internal_holdout_unseal_authorized",
                "external_test_unseal_authorized",
            }
        ),
    )
    (
        frozen_reference,
        model_selection_gate_passed,
        power,
        power_passed,
    ) = _derive_model_selection_handoff(artifact.get("evaluation"))
    artifact_sha = _file_sha256(artifact_file)
    public_sha = _require_sha256(
        lineage.get("public_aggregate_report_sha256"),
        "public_aggregate_report_sha256",
    )
    if (
        artifact_sha
        != _require_sha256(
            lineage.get("private_reference_freeze_artifact_sha256"),
            "private_reference_freeze_artifact_sha256",
        )
        or artifact.get("public_report_sha256") != public_sha
        or artifact_analysis.get("analysis_config_sha256")
        != CONFIRMATORY_ANALYSIS_SHA256
        or lineage.get("analysis_config_sha256") != CONFIRMATORY_ANALYSIS_SHA256
        or artifact_lineage.get("cross_variant_alignment_sha256")
        != lineage.get("cross_variant_alignment_sha256")
        or completion.get("primary_variant") != PRIMARY_VARIANT
        or completion.get("registered_variants") != list(REGISTERED_VARIANTS)
        or completion.get("reference_candidates") != list(REFERENCE_CANDIDATES)
        or completion.get("frozen_reference") != frozen_reference
        or _handoff_bool(
            completion.get("model_selection_gate_passed"),
            "receipt.model_selection_gate_passed",
        )
        != model_selection_gate_passed
        or not math.isclose(
            _handoff_number(
                completion.get("prospective_power"),
                "receipt.prospective_power",
            ),
            power,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or _handoff_number(
            completion.get("minimum_prospective_power"),
            "receipt.minimum_prospective_power",
        )
        != MINIMUM_PROSPECTIVE_POWER
        or _handoff_bool(
            completion.get("power_gate_passed"),
            "receipt.power_gate_passed",
        )
        != power_passed
        or completion.get("single_dataset_only") is not True
        or completion.get("aggregate_only") is not True
        or any(
            completion.get(name) is not False
            for name in (
                "confirmatory_claim_authorized",
                "calibration_unseal_authorized",
                "internal_holdout_unseal_authorized",
                "external_test_unseal_authorized",
            )
        )
    ):
        raise ModelSelectionEvaluationError("reference-freeze handoff lineage changed")
    _validate_json_safe_aggregate(artifact, public=False)
    _validate_json_safe_aggregate(receipt, public=False)
    if (
        _file_sha256(receipt_file) != expected_receipt
        or _file_sha256(artifact_file) != artifact_sha
    ):
        raise ModelSelectionEvaluationError(
            "reference-freeze handoff changed while verifying"
        )
    return VerifiedModelSelectionAggregateAttestation(
        dataset=str(artifact["dataset"]),
        artifact_path=artifact_file,
        artifact_sha256=artifact_sha,
        receipt_path=receipt_file,
        receipt_sha256=expected_receipt,
        public_report_sha256=public_sha,
        analysis_config_sha256=CONFIRMATORY_ANALYSIS_SHA256,
        cross_variant_alignment_sha256=_require_sha256(
            lineage["cross_variant_alignment_sha256"],
            "cross_variant_alignment_sha256",
        ),
        frozen_reference=frozen_reference,
        model_selection_gate_passed=model_selection_gate_passed,
        prospective_power=power,
        power_gate_passed=power_passed,
    )
