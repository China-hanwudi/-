"""Repair 3/3: emotion-posterior relations plus fixed VAD transitions.

The primary method is frozen before any model-selection result is visible:

``59-D task features + raw seven-class posterior concatenation + the complete
3 x 3 current/history modality relation grid + fixed VAD state/transitions``.

The module has three deliberately separate stages.  First, a label-free opaque
group hash split reserves 20% of the physical fit role for an internal gate;
all projectors, utility models, and thresholds are fit on the other 80% only.
Second, the gate compares the frozen 299-D primary against both the original
59-D baseline and a deterministic rank-59 299-D capacity control.  The same
utility seed must improve Macro-F1 by 0.002, not worsen NLL, and not reduce
accuracy against both references; four of five seeds must pass.  Only then is
the complete fit role retrained and the physical model-selection role opened.
Third, the already-fixed primary and explanatory ablations are scored.  No
model-selection result selects a feature layer, class mapping, VAD map, weight,
threshold, or primary variant.

VAD values are posterior expectations over pre-registered Mehrabian-style
PAD/VAD design coordinates.  They are theory-inspired engineering anchors,
not estimated psychological truth.  Gold labels are training/scoring targets
only and never enter an inference feature.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import platform
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .bidirectional_emotion_utility import (
    BidirectionalCoalitionTask,
    bidirectional_utility_targets,
)
from .bidirectional_utility_model import (
    BidirectionalUtilityCache,
    UtilitySplit,
    trainable_parameter_count,
)
from .causal_multimodal_backbone import CausalBackboneConfig
from .class_balanced_utility_repair import (
    MODEL_NAMES,
    BalancedUtilitySeedScores,
    CapacityMatchedUtilitySpec,
    ClassBalanceSpec,
    default_capacity_matched_specs,
    fit_class_balanced_seed_scores,
    fit_class_balanced_utility_model,
    group_oof_class_balanced_predictions,
)
from .data_contract import ContractError, write_json_atomic
from .emotion_probability_relations import (
    BASE_CACHE_FEATURE_NAMES,
    BASE_CACHE_SCHEMA_VERSION,
    HISTORY_CONTEXTS,
    MODALITIES,
    BaseCacheLineage,
    EmotionProbabilityBlock,
    TrainOnlyProvenance,
    align_with_59d_task_cache,
    base_cache_lineage_sha256,
    bidirectional_task_order_sha256,
    build_emotion_probability_relations,
    dataset_identity_sha256,
    emotion_class_order_sha256,
    emotion_context_schema_sha256,
    feature_names_content_sha256,
    numeric_matrix_content_sha256,
    ordered_source_sha256,
    verify_base_59d_cache,
)
from .emotiontalk_bidirectional_oof import probability_task_features, task_contexts
from .emotiontalk_causal_backbone_runner import (
    FIT_ROLE,
    SELECTION_ROLE,
    OpenRoleCorpus,
    UtilitySamplingConfig,
    VerifiedCorpusProvenance,
    _history_indices,
    _role_assignment_sha256,
    load_emotiontalk_open_role_corpus,
    sample_corpus_bidirectional_tasks,
)
from . import emotiontalk_role_sidecar as role_sidecar
from .emotiontalk_query_policy_runner import (
    aggregate_candidate_draw_scores,
    build_reversible_selected_contexts,
    coverage_matched_recency_contexts,
    fit_query_candidate_coverage_threshold,
    query_strategy_metrics,
)
from .emotiontalk_sampled_context_runner import _assert_aggregate_output
from .emotiontalk_text_p1 import LABEL_NAMES
from .meld_text_pilot import sha256_file


PROTOCOL = "emotion_relation_vad_repair_v1"
REPORT_SCHEMA_VERSION = "emotion_relation_vad_repair_report_v1"
REGISTERED_STATUS = "repair_3_of_3_frozen_before_fit_gate_and_model_selection"
CLASS_ORDER = tuple(LABEL_NAMES)
BASE_SEEDS = (17, 29, 43, 71, 101)
UTILITY_SEEDS = (17, 29, 43, 71, 101)
PRIMARY_VARIANT = "full_59d_concat_3x3_vad"
CAPACITY_CONTROL_VARIANT = "base_59d_rank59_capacity_control_299d"
VARIANT_ORDER = (
    PRIMARY_VARIANT,
    CAPACITY_CONTROL_VARIANT,
    "same_modality_relations_vad",
    "full_relations_no_vad",
    "concat_vad_no_3x3",
    "base_59d_only",
)
VARIANT_WIDTHS = MappingProxyType(
    {
        PRIMARY_VARIANT: 299,
        CAPACITY_CONTROL_VARIANT: 299,
        "same_modality_relations_vad": 227,
        "full_relations_no_vad": 272,
        "concat_vad_no_3x3": 191,
        "base_59d_only": 59,
    }
)
FIT_GATE_MACRO_F1_GAIN = 0.002
FIT_GATE_REQUIRED_SEEDS = 4
NLL_IDENTITY_TOLERANCE = 1.0e-12
FUSION_CURRENT_WEIGHT = 0.5
FIT_INTERNAL_GATE_NAMESPACE = "CARMA-Affect/Repair3/fit-internal-gate/v1"
FIT_INTERNAL_GATE_EVAL_FRACTION = 0.20
CAPACITY_CONTROL_NAMESPACE = "CARMA-Affect/Repair3/capacity-control/v1"
CAPACITY_CONTROL_MATRIX_SHA256 = (
    "20eb5664adacd98324261590d360d1ad2e30306dc0d90e0b622a388f2e1b1f36"
)
CAPACITY_CONTROL_SPEC_SHA256 = (
    "b8d18ce77d49f4b3bc59bd68c6b6ecf3fee136769cc234744eaa293f2a47e522"
)
SUPERSEDED_UNVERIFIABLE_CAPACITY_SPEC_SHA256 = (
    "d0cf08dba1f81a9393de048ce7211b4591f05dced7a4137767a8daa66c382ab0"
)
REGISTERED_SIDECAR_MANIFEST_SHA256 = (
    "bbd843876fa051c5426d0d56870adc939cdf71e1e8eaf552880ab4f89d47f530"
)
REGISTERED_MODEL_CONFIG_SHA256 = (
    "abdf905c3351f3fe265dd4741977c5760aa101fd78257cd803a4e9a504d8fce2"
)
REGISTERED_OUTPUT_FILENAME = (
    "emotiontalk_emotion_relation_vad_repair_v1_open_role.json"
)
REGISTERED_OUTPUT_REPOSITORY_RELATIVE_PATH = (
    f"artifacts/{REGISTERED_OUTPUT_FILENAME}"
)
REGISTERED_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / REGISTERED_OUTPUT_FILENAME
).resolve()
PRIMARY_OR_CONTROL_PARAMETER_COUNT = 10_162
BASE_59D_PARAMETER_COUNT = 2_482
FIT_GATE_STAGE_ACCESS_CONTRACT = MappingProxyType(
    {
        "stage_1_sha256_verify": [
            f"features_{FIT_ROLE}.npz",
            f"labels_{FIT_ROLE}.npz",
            f"features_{SELECTION_ROLE}.npz",
            f"labels_{SELECTION_ROLE}.npz",
        ],
        "stage_1_deserialize": [
            f"features_{FIT_ROLE}.npz",
            f"labels_{FIT_ROLE}.npz",
        ],
        "stage_1_hash_only_never_deserialize": [
            f"features_{SELECTION_ROLE}.npz",
            f"labels_{SELECTION_ROLE}.npz",
        ],
        "stage_1_materialized_roles": [FIT_ROLE],
        "stage_2_selection_label_materialization_requires_fit_gate_pass": True,
        "stage_2_full_alignment_and_provenance_revalidation": True,
        "fit_gate_failure_selection_feature_payload_opened": False,
        "fit_gate_failure_selection_label_deserialized": False,
        "fit_gate_failure_selection_label_payload_opened": False,
        "fit_gate_failure_selection_scored": False,
    }
)
REPAIR2_CONFIG_SHA256 = "a39886300ceb91b657ba80232fbdc74abf93ec0b4c640782ab51bd55a9abe188"
REPAIR2_RESULT_SHA256 = "547e54e1e6f525944eb6715e760e4a0af78abb9b4d0bea7d1940b4fdc0be1cf4"

VAD_COORDINATES = MappingProxyType(
    {
        "neutral": (0.00, 0.00, 0.00),
        "happy": (0.81, 0.51, 0.46),
        "sad": (-0.63, -0.27, -0.33),
        "angry": (-0.51, 0.59, 0.25),
        "surprised": (0.40, 0.67, -0.13),
        "disgusted": (-0.60, 0.35, 0.11),
        "fearful": (-0.64, 0.60, -0.43),
    }
)
VAD_COORDINATE_SYSTEM = "Mehrabian-style pleasure/valence-arousal-dominance design coordinates"
VAD_CLAIM_BOUNDARY = "fixed theory-inspired engineering anchors; not estimated psychological truth"


class EmotionRelationVADRepairError(ContractError):
    """Raised when Repair 3 violates a frozen scientific or data contract."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


CAPACITY_CONTROL_SPEC = MappingProxyType(
    {
        "application": "X_59 @ E_59x299",
        "dtype": "float64",
        "matrix_orientation": "source_by_target",
        "multiplicity": "count_target_columns_mapped_to_source_index",
        "namespace": CAPACITY_CONTROL_NAMESPACE,
        "nonzero_value": "sign / sqrt(multiplicity)",
        "schema_version": "carma_repair3_rank59_capacity_control_v1",
        "sign": (
            "sha256(namespace + unit_separator + decimal_j)."
            "first_byte_even_is_positive"
        ),
        "source_index_for_target_column": "j_mod_source_width",
        "source_width": 59,
        "target_width": 299,
    }
)

FIT_INTERNAL_GATE_SPLIT_SPEC = MappingProxyType(
    {
        "schema_version": "carma_repair3_fit_internal_gate_split_v1",
        "namespace": FIT_INTERNAL_GATE_NAMESPACE,
        "input": "unique FIT_ROLE opaque_group_hashes only",
        "score": "sha256(utf8(namespace + unit_separator + group_hash))",
        "ordering": "ascending (score_hex, group_hash)",
        "gate_eval_fraction": FIT_INTERNAL_GATE_EVAL_FRACTION,
        "gate_eval_group_count": "ceil(fraction * unique_fit_groups)",
        "gate_eval_assignment": "first ordered groups",
        "gate_train_assignment": "remaining ordered groups",
        "label_stratification": False,
        "salt_search_or_result_conditioned_resplit": False,
    }
)
FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256 = _canonical_sha256(
    dict(FIT_INTERNAL_GATE_SPLIT_SPEC)
)


def capacity_control_expansion_matrix() -> np.ndarray:
    """Return the frozen label-free rank-59 59x299 capacity-control map."""

    if _canonical_sha256(dict(CAPACITY_CONTROL_SPEC)) != CAPACITY_CONTROL_SPEC_SHA256:
        raise EmotionRelationVADRepairError("capacity-control spec hash changed")
    source_width = int(CAPACITY_CONTROL_SPEC["source_width"])
    target_width = int(CAPACITY_CONTROL_SPEC["target_width"])
    multiplicities = np.bincount(
        np.arange(target_width, dtype=np.int64) % source_width,
        minlength=source_width,
    )
    matrix = np.zeros((source_width, target_width), dtype=np.float64)
    for target in range(target_width):
        source = target % source_width
        first_byte = hashlib.sha256(
            f"{CAPACITY_CONTROL_NAMESPACE}\x1f{target}".encode("utf-8")
        ).digest()[0]
        sign = 1.0 if first_byte % 2 == 0 else -1.0
        matrix[source, target] = sign / math.sqrt(float(multiplicities[source]))
    observed = numeric_matrix_content_sha256(matrix)
    if observed != CAPACITY_CONTROL_MATRIX_SHA256:
        raise EmotionRelationVADRepairError(
            "capacity-control expansion matrix hash changed"
        )
    if np.linalg.matrix_rank(matrix) != source_width or not np.allclose(
        matrix @ matrix.T,
        np.eye(source_width, dtype=np.float64),
        rtol=0.0,
        atol=5.0e-16,
    ):
        raise EmotionRelationVADRepairError(
            "capacity-control expansion lost its rank-59 isometry"
        )
    matrix.setflags(write=False)
    return matrix


CAPACITY_CONTROL_EXPANSION = capacity_control_expansion_matrix()


def vad_coordinate_sha256(
    class_order: Sequence[str] = CLASS_ORDER,
    coordinates: Mapping[str, Sequence[float]] = VAD_COORDINATES,
) -> str:
    """Hash the ordered, named VAD design anchors and their claim boundary."""

    order = tuple(str(value) for value in class_order)
    if order != CLASS_ORDER or set(coordinates) != set(CLASS_ORDER):
        raise EmotionRelationVADRepairError("VAD class mapping is not the frozen seven-class map")
    ordered: list[list[object]] = []
    for name in order:
        vector = tuple(float(value) for value in coordinates[name])
        if len(vector) != 3 or not np.isfinite(vector).all():
            raise EmotionRelationVADRepairError("VAD coordinates must be finite three-vectors")
        ordered.append([name, *vector])
    return _canonical_sha256(
        {
            "coordinate_system": VAD_COORDINATE_SYSTEM,
            "claim_boundary": VAD_CLAIM_BOUNDARY,
            "ordered_anchors": ordered,
        }
    )


VAD_COORDINATE_SHA256 = vad_coordinate_sha256()


@dataclass(frozen=True)
class ProjectorSpec:
    """Frozen single-modality seven-class posterior producer."""

    folds: int = 5
    loss: str = "log_loss"
    penalty: str = "l2"
    alpha: float = 1.0e-4
    max_iter: int = 400
    tolerance: float = 1.0e-4
    class_weight: str = "balanced"
    average: bool = True
    text_analyzer: str = "char_wb"
    text_ngram_min: int = 2
    text_ngram_max: int = 5
    text_min_df: int = 2
    text_max_features: int = 50_000
    text_sublinear_tf: bool = True

    def validate(self) -> None:
        expected = {
            "loss": "log_loss",
            "penalty": "l2",
            "class_weight": "balanced",
            "text_analyzer": "char_wb",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise EmotionRelationVADRepairError(f"projector {name} changed")
        if self.folds != 5 or self.alpha != 1.0e-4 or self.max_iter != 400:
            raise EmotionRelationVADRepairError("projector capacity/optimization changed")
        if self.tolerance != 1.0e-4 or not self.average:
            raise EmotionRelationVADRepairError("projector convergence rule changed")
        if (self.text_ngram_min, self.text_ngram_max) != (2, 5):
            raise EmotionRelationVADRepairError("text n-gram range changed")
        if self.text_min_df != 2 or self.text_max_features != 50_000:
            raise EmotionRelationVADRepairError("text vocabulary contract changed")
        if self.text_sublinear_tf is not True:
            raise EmotionRelationVADRepairError("text sublinear-TF contract changed")

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class FrozenRepairConfig:
    raw: Mapping[str, object]
    projector: ProjectorSpec
    sampling: UtilitySamplingConfig
    balance: ClassBalanceSpec
    utility_specs: tuple[CapacityMatchedUtilitySpec, ...]


@dataclass(frozen=True)
class RolePosteriorGrid:
    """Role-local posterior probabilities; row_indices map back to the corpus."""

    role: str
    row_indices: np.ndarray
    probabilities: Mapping[str, np.ndarray]
    base_seeds: tuple[int, ...]
    fold_by_local_row: np.ndarray
    fold_assignment_sha256: str
    producer_config_sha256: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.row_indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices):
            raise EmotionRelationVADRepairError("posterior source rows must be unique and non-empty")
        if self.role not in {FIT_ROLE, SELECTION_ROLE}:
            raise EmotionRelationVADRepairError("posterior role is not open")
        if tuple(self.base_seeds) != BASE_SEEDS:
            raise EmotionRelationVADRepairError("five base seeds changed")
        validated: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            if modality not in self.probabilities:
                raise EmotionRelationVADRepairError("posterior modality is missing")
            values = np.asarray(self.probabilities[modality], dtype=np.float64)
            expected = (len(BASE_SEEDS), len(indices), len(CLASS_ORDER))
            if values.shape != expected or not np.isfinite(values).all():
                raise EmotionRelationVADRepairError(
                    f"{modality} posterior must have finite shape {expected}"
                )
            if np.any(values < -1e-12) or not np.allclose(
                values.sum(axis=2), 1.0, rtol=1e-7, atol=1e-9
            ):
                raise EmotionRelationVADRepairError("posterior left the probability simplex")
            copied = np.array(values, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            validated[modality] = copied
        folds = np.asarray(self.fold_by_local_row, dtype=np.int32)
        if folds.shape != (len(indices),):
            raise EmotionRelationVADRepairError("posterior fold assignment is not row-aligned")
        if self.role == FIT_ROLE and (np.any(folds < 0) or len(np.unique(folds)) != 5):
            raise EmotionRelationVADRepairError("fit posteriors are not complete five-fold OOF")
        if self.role == SELECTION_ROLE and np.any(folds != -1):
            raise EmotionRelationVADRepairError("selection posteriors must be full-fit predictions")
        for name, digest in (
            ("fold_assignment_sha256", self.fold_assignment_sha256),
            ("producer_config_sha256", self.producer_config_sha256),
        ):
            if len(str(digest)) != 64 or set(str(digest)) - set("0123456789abcdef"):
                raise EmotionRelationVADRepairError(f"{name} is not a lowercase SHA-256")
        indices.setflags(write=False)
        folds.setflags(write=False)
        object.__setattr__(self, "row_indices", indices)
        object.__setattr__(self, "fold_by_local_row", folds)
        object.__setattr__(self, "probabilities", MappingProxyType(validated))

    @property
    def local_by_corpus_row(self) -> Mapping[int, int]:
        return MappingProxyType(
            {int(row): local for local, row in enumerate(self.row_indices.tolist())}
        )


@dataclass(frozen=True)
class RoleFeatureSet:
    role: str
    tasks: tuple[BidirectionalCoalitionTask, ...]
    task_labels: np.ndarray
    cluster_codes: np.ndarray
    base_probability_by_seed: np.ndarray
    variants: Mapping[str, UtilitySplit]
    feature_names: Mapping[str, tuple[str, ...]]
    provenance: TrainOnlyProvenance
    base_cache_lineage_sha256: str
    vad_coordinate_sha256: str
    outcome_labels_used_for_feature_construction: bool = True


@dataclass(frozen=True)
class FitGateState:
    utility_seed: int
    threshold: float
    task_scores: np.ndarray


@dataclass(frozen=True)
class FitInternalGateSplit:
    """Opaque, group-whole, label-free assignment inside the physical fit role."""

    namespace: str
    split_spec_sha256: str
    ordered_input_group_sha256: str
    group_assignment_sha256: str
    row_assignment_sha256: str
    train_groups: tuple[str, ...]
    eval_groups: tuple[str, ...]
    train_rows: tuple[int, ...]
    eval_rows: tuple[int, ...]
    train_history_eligible_rows: int
    eval_history_eligible_rows: int
    train_class_complete: bool
    eval_class_complete: bool

    @property
    def eligible(self) -> bool:
        return bool(
            self.train_class_complete
            and self.eval_class_complete
            and self.train_history_eligible_rows > 0
            and self.eval_history_eligible_rows > 0
        )

    def aggregate_attestation(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "algorithm": (
                "SHA256(UTF8(namespace + unit_separator + opaque_group_hash)); "
                "ascending (score, group); first ceil(0.20*n_groups) gate-eval"
            ),
            "gate_eval_fraction": FIT_INTERNAL_GATE_EVAL_FRACTION,
            "split_spec_sha256": self.split_spec_sha256,
            "ordered_input_group_sha256": self.ordered_input_group_sha256,
            "group_assignment_sha256": self.group_assignment_sha256,
            "row_assignment_sha256": self.row_assignment_sha256,
            "group_counts": {
                "fit_total": len(self.train_groups) + len(self.eval_groups),
                "gate_train": len(self.train_groups),
                "gate_eval": len(self.eval_groups),
            },
            "row_counts": {
                "fit_total": len(self.train_rows) + len(self.eval_rows),
                "gate_train": len(self.train_rows),
                "gate_eval": len(self.eval_rows),
            },
            "history_eligible_row_counts": {
                "gate_train": self.train_history_eligible_rows,
                "gate_eval": self.eval_history_eligible_rows,
            },
            "class_complete": {
                "gate_train": self.train_class_complete,
                "gate_eval": self.eval_class_complete,
            },
            "label_stratification": False,
            "salt_search_or_result_conditioned_resplit": False,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class GateVariantSeedScore:
    seed: int
    threshold: float
    gate_eval_scores: np.ndarray
    train_oof_fold_by_row: np.ndarray
    train_query_candidate_pairs: int
    realized_train_query_candidate_coverage: float
    parameter_count: int
    training_artifact_sha256: str
    threshold_commitment_sha256: str


@dataclass(frozen=True)
class StagedVerifiedCorpusProvenance:
    """Tamper-evident Stage-1 lineage with selection labels still opaque bytes."""

    dataset_id: str
    manifest_schema: str
    manifest_status: str
    manifest_sha256: str
    source_hashes: Mapping[str, str]
    label_order: tuple[str, ...]
    role_rows: Mapping[str, int]
    audio_dim: int
    video_dim: int
    role_assignment_sha256: str
    speaker_mapping_sha256: str
    fit_stage_contract_sha256: str
    verification_origin: str
    verifier_attestation_sha256: str
    strict_role_feature_sidecars: bool = True
    strict_role_label_sidecars: bool = True
    selection_feature_hash_verified: bool = True
    selection_feature_payload_opened: bool = False
    selection_feature_deserialized: bool = False
    selection_label_hash_verified: bool = True
    selection_label_payload_opened: bool = False
    selection_label_deserialized: bool = False
    sealed_role_arrays_opened: bool = False
    validation_or_test_opened: bool = False

    def validate(self, corpus: OpenRoleCorpus, model_config: CausalBackboneConfig) -> None:
        """Validate the fit-gate corpus without materialising selection labels."""

        if self.verification_origin != "emotiontalk_manifest_v2_fit_gate_stage1":
            raise EmotionRelationVADRepairError(
                "staged provenance was not produced by the fit-gate verifier"
            )
        if not self.strict_role_feature_sidecars or not self.strict_role_label_sidecars:
            raise EmotionRelationVADRepairError("strict physical role sidecars are required")
        if self.sealed_role_arrays_opened or self.validation_or_test_opened:
            raise EmotionRelationVADRepairError("staged provenance attests sealed/dev/test access")
        if (
            not self.selection_feature_hash_verified
            or self.selection_feature_payload_opened
            or self.selection_feature_deserialized
        ):
            raise EmotionRelationVADRepairError(
                "selection features must be hash-verified but opaque before the fit gate"
            )
        if (
            not self.selection_label_hash_verified
            or self.selection_label_payload_opened
            or self.selection_label_deserialized
        ):
            raise EmotionRelationVADRepairError(
                "selection labels must be hash-verified but opaque before the fit gate"
            )
        if not self.dataset_id or any(character in self.dataset_id for character in "\\/\x00"):
            raise EmotionRelationVADRepairError("staged provenance dataset id is empty or unsafe")
        if not self.manifest_schema or not self.manifest_status:
            raise EmotionRelationVADRepairError("staged manifest identity is incomplete")
        for name, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("role_assignment_sha256", self.role_assignment_sha256),
            ("speaker_mapping_sha256", self.speaker_mapping_sha256),
            ("fit_stage_contract_sha256", self.fit_stage_contract_sha256),
            ("verifier_attestation_sha256", self.verifier_attestation_sha256),
        ):
            _staged_sha256(digest, field=name)
        expected_source_names = {
            "sidecar_manifest",
            "trusted_source_label_archive",
            "trusted_source_media_features",
            "trusted_source_transcription",
            f"{FIT_ROLE}_features",
            f"{FIT_ROLE}_labels",
            f"{SELECTION_ROLE}_features",
            f"{SELECTION_ROLE}_labels",
        }
        if set(self.source_hashes) != expected_source_names:
            raise EmotionRelationVADRepairError("staged source-hash schema changed")
        for name, digest in self.source_hashes.items():
            if not name or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in str(name)
            ):
                raise EmotionRelationVADRepairError("staged source-hash name is unsafe")
            _staged_sha256(digest, field=f"source_hashes.{name}")
        if self.source_hashes["sidecar_manifest"] != self.manifest_sha256:
            raise EmotionRelationVADRepairError(
                "staged manifest hash is not bound to source hashes"
            )
        if tuple(self.label_order) != CLASS_ORDER or len(self.label_order) != model_config.num_classes:
            raise EmotionRelationVADRepairError("staged label order differs from model/classes")
        if set(self.role_rows) != {FIT_ROLE, SELECTION_ROLE}:
            raise EmotionRelationVADRepairError("staged manifest role-row schema changed")
        if int(self.role_rows[FIT_ROLE]) != len(corpus.keys):
            raise EmotionRelationVADRepairError("staged fit rows differ from corpus")
        if int(self.role_rows[SELECTION_ROLE]) <= 0:
            raise EmotionRelationVADRepairError("staged selection-row commitment is empty")
        if (self.audio_dim, self.video_dim) != (model_config.audio_dim, model_config.video_dim):
            raise EmotionRelationVADRepairError("staged modality dimensions differ from model")
        _validate_staged_corpus(corpus, model_config)
        if self.speaker_mapping_sha256 != corpus.speaker_mapping_sha256:
            raise EmotionRelationVADRepairError("staged speaker mapping differs from corpus")
        if self.role_assignment_sha256 != _role_assignment_sha256(corpus):
            raise EmotionRelationVADRepairError("staged role assignment differs from corpus")
        selection_feature_sha = self.source_hashes[f"{SELECTION_ROLE}_features"]
        selection_label_sha = self.source_hashes[f"{SELECTION_ROLE}_labels"]
        if self.fit_stage_contract_sha256 != _fit_stage_contract_sha256(
            corpus,
            selection_feature_sha256=selection_feature_sha,
            selection_label_sha256=selection_label_sha,
        ):
            raise EmotionRelationVADRepairError("staged fit contract changed after verification")
        if self.verifier_attestation_sha256 != _canonical_sha256(
            _staged_provenance_attestation_payload(self)
        ):
            raise EmotionRelationVADRepairError("staged provenance attestation changed")


@dataclass(frozen=True)
class StagedOpenRoleLoad:
    """Stage-1 fit-only corpus plus opaque selection-file commitments."""

    corpus: OpenRoleCorpus
    provenance: StagedVerifiedCorpusProvenance
    manifest: Mapping[str, object]


CorpusProvenance = StagedVerifiedCorpusProvenance | VerifiedCorpusProvenance


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EmotionRelationVADRepairError(f"{name} must be a mapping")
    return value


def _load_json(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EmotionRelationVADRepairError(f"cannot read {name}: {error}") from error
    return _mapping(value, name=name)


def _staged_sha256(value: object, *, field: str) -> str:
    try:
        return role_sidecar._valid_sha256(value, field=field)
    except ContractError as error:
        raise EmotionRelationVADRepairError(str(error)) from error


def _validate_staged_manifest_and_hashes(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    model_config: CausalBackboneConfig,
) -> tuple[dict[str, object], dict[tuple[str, str], Path], dict[str, str]]:
    """Validate the complete manifest and hash all four physical role files."""

    manifest = dict(_load_json(manifest_path, name="EmotionTalk sidecar manifest"))
    required_root = {
        "schema_version",
        "protocol",
        "status",
        "dataset_id",
        "split_protocol_id",
        "label_order",
        "source_contract",
        "seal_contract",
        "roles",
        "config_sha256",
        "public_content_audit",
    }
    if set(manifest) != required_root:
        raise EmotionRelationVADRepairError("EmotionTalk sidecar manifest schema changed")
    if (
        manifest["schema_version"] != role_sidecar.MANIFEST_SCHEMA
        or manifest["protocol"] != role_sidecar.PROTOCOL
    ):
        raise EmotionRelationVADRepairError("EmotionTalk sidecar manifest protocol changed")
    if manifest["status"] != "strict_open_role_feature_and_label_sidecars_created_and_hashed":
        raise EmotionRelationVADRepairError(
            "EmotionTalk sidecar manifest is not a completed strict artifact"
        )
    if (
        manifest["dataset_id"] != "EmotionTalk"
        or manifest["split_protocol_id"] != "scu_set_exploration_v1"
    ):
        raise EmotionRelationVADRepairError("EmotionTalk sidecar dataset/split changed")
    if tuple(manifest["label_order"]) != CLASS_ORDER:
        raise EmotionRelationVADRepairError("EmotionTalk label order changed")

    source = _mapping(manifest["source_contract"], name="source_contract")
    expected_source_fields = {
        "label_archive",
        "media_features",
        "transcription",
        "feature_config_sha256",
        "trusted_source_boundary_only",
        "validation_or_test_label_payload_opened",
    }
    if set(source) != expected_source_fields:
        raise EmotionRelationVADRepairError("EmotionTalk manifest source contract changed")
    for field in ("label_archive", "media_features", "transcription", "feature_config_sha256"):
        _staged_sha256(source[field], field=f"source_contract.{field}")
    if source["trusted_source_boundary_only"] is not True:
        raise EmotionRelationVADRepairError("EmotionTalk trusted source boundary changed")
    if source["validation_or_test_label_payload_opened"] is not False:
        raise EmotionRelationVADRepairError(
            "EmotionTalk manifest attests validation/test label access"
        )

    seal = _mapping(manifest["seal_contract"], name="seal_contract")
    expected_seal_fields = {
        "model_runner_opens_upstream_media_npz_or_transcription",
        "open_role_runner_may_load_only",
        "calibration_holdout_validation_test_sidecars_created",
        "allow_pickle_required_to_load_sidecars",
    }
    if set(seal) != expected_seal_fields:
        raise EmotionRelationVADRepairError("EmotionTalk manifest seal contract changed")
    if (
        seal["model_runner_opens_upstream_media_npz_or_transcription"] is not False
        or seal["open_role_runner_may_load_only"] != [FIT_ROLE, SELECTION_ROLE]
        or seal["calibration_holdout_validation_test_sidecars_created"] is not False
        or seal["allow_pickle_required_to_load_sidecars"] is not False
    ):
        raise EmotionRelationVADRepairError(
            "EmotionTalk manifest does not enforce strict physical separation"
        )
    _staged_sha256(manifest["config_sha256"], field="config_sha256")
    expected_public_audit = {
        "contains_labels_or_class_counts": False,
        "contains_row_group_or_speaker_identifiers": False,
        "contains_transcripts_or_embeddings": False,
        "contains_private_absolute_paths": False,
    }
    if manifest["public_content_audit"] != expected_public_audit:
        raise EmotionRelationVADRepairError("EmotionTalk public-content audit changed")

    roles = _mapping(manifest["roles"], name="roles")
    if set(roles) != {FIT_ROLE, SELECTION_ROLE}:
        raise EmotionRelationVADRepairError(
            "EmotionTalk manifest exposes a non-open or missing role"
        )
    paths: dict[tuple[str, str], Path] = {}
    observed_hashes: dict[str, str] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    for role in (FIT_ROLE, SELECTION_ROLE):
        record = _mapping(roles[role], name=f"roles.{role}")
        if set(record) != role_sidecar.MANIFEST_ROLE_FIELDS:
            raise EmotionRelationVADRepairError("EmotionTalk manifest role record is malformed")
        expected_names = {
            "features": f"features_{role}.npz",
            "labels": f"labels_{role}.npz",
        }
        if (
            record.get("feature_filename") != expected_names["features"]
            or record.get("label_filename") != expected_names["labels"]
        ):
            raise EmotionRelationVADRepairError("EmotionTalk manifest sidecar filename changed")
        try:
            rows = int(record["rows"])
            groups = int(record["groups"])
            history_rows = int(record["history_eligible_rows"])
            audio_dim = int(record["audio_dimension"])
            video_dim = int(record["video_dimension"])
        except (TypeError, ValueError) as error:
            raise EmotionRelationVADRepairError(
                "EmotionTalk manifest role dimensions are malformed"
            ) from error
        if (
            rows <= 0
            or not 1 <= groups <= rows
            or not 0 <= history_rows <= rows
            or audio_dim <= 0
            or video_dim <= 0
        ):
            raise EmotionRelationVADRepairError(
                "EmotionTalk manifest role dimensions are invalid"
            )
        dimensions[role] = (audio_dim, video_dim)
        _staged_sha256(record["row_alignment_sha256"], field=f"roles.{role}.row_alignment")
        for kind, filename in expected_names.items():
            path = sidecar_dir / filename
            paths[(role, kind)] = path
            observed = sha256_file(path)
            manifest_field = "feature_sha256" if kind == "features" else "label_sha256"
            expected = _staged_sha256(
                record[manifest_field], field=f"roles.{role}.{manifest_field}"
            )
            if observed != expected:
                raise EmotionRelationVADRepairError(
                    f"EmotionTalk {kind[:-1]} sidecar hash differs from manifest"
                )
            observed_hashes[f"{role}_{kind}"] = observed
    if dimensions[FIT_ROLE] != dimensions[SELECTION_ROLE]:
        raise EmotionRelationVADRepairError("EmotionTalk role modality dimensions differ")
    if dimensions[FIT_ROLE] != (model_config.audio_dim, model_config.video_dim):
        raise EmotionRelationVADRepairError(
            "EmotionTalk sidecar dimensions differ from the causal model"
        )
    observed_hashes["sidecar_manifest"] = sha256_file(manifest_path)
    return manifest, paths, observed_hashes


def _load_staged_feature_payload(
    *,
    role: str,
    path: Path,
    record: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    """Deserialize and validate one feature sidecar without touching its labels."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != role_sidecar.FEATURE_FIELDS:
            raise EmotionRelationVADRepairError("EmotionTalk feature sidecar schema changed")
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if role_sidecar._scalar_text(payload, "schema_version") != role_sidecar.FEATURE_SCHEMA:
        raise EmotionRelationVADRepairError("EmotionTalk feature sidecar version changed")
    if role_sidecar._scalar_text(payload, "dataset_id") != "EmotionTalk":
        raise EmotionRelationVADRepairError("EmotionTalk feature sidecar dataset changed")
    if role_sidecar._scalar_text(payload, "role") != role:
        raise EmotionRelationVADRepairError("EmotionTalk feature sidecar role changed")
    if role_sidecar._scalar_text(payload, "split_protocol_id") != "scu_set_exploration_v1":
        raise EmotionRelationVADRepairError("EmotionTalk feature sidecar split changed")
    alignment = _staged_sha256(
        role_sidecar._scalar_text(payload, "row_alignment_sha256"),
        field=f"{role}.feature.row_alignment_sha256",
    )
    if alignment != _staged_sha256(
        record["row_alignment_sha256"], field=f"roles.{role}.row_alignment_sha256"
    ):
        raise EmotionRelationVADRepairError(
            "EmotionTalk feature alignment differs from manifest"
        )
    if role_sidecar._scalar_text(payload, "source_feature_config_sha256") != str(
        source["feature_config_sha256"]
    ):
        raise EmotionRelationVADRepairError(
            "EmotionTalk feature sidecar source config differs from manifest"
        )

    rows = int(record["rows"])
    row_hashes = payload["opaque_row_hashes"].astype(str)
    groups = payload["opaque_group_hashes"].astype(str)
    speakers = payload["speaker_tokens"].astype(str)
    turns = payload["turn_ids"].astype(np.int64, copy=True)
    protocol_rows = payload["protocol_row_ids"].astype(np.int64, copy=True)
    buckets = payload["role_buckets"].astype(np.int16, copy=True)
    texts = payload["texts"].astype(str)
    one_dimensional = (
        row_hashes,
        groups,
        speakers,
        turns,
        protocol_rows,
        buckets,
        texts,
    )
    audio = payload["audio_features"].astype(np.float32, copy=True)
    video = payload["video_features"].astype(np.float32, copy=True)
    if any(np.asarray(value).shape != (rows,) for value in one_dimensional):
        raise EmotionRelationVADRepairError("EmotionTalk feature rows differ from manifest")
    if audio.shape != (rows, int(record["audio_dimension"])) or video.shape != (
        rows,
        int(record["video_dimension"]),
    ):
        raise EmotionRelationVADRepairError(
            "EmotionTalk feature modality dimensions differ from manifest"
        )
    if (
        len(set(row_hashes.tolist())) != rows
        or not np.isfinite(audio).all()
        or not np.isfinite(video).all()
    ):
        raise EmotionRelationVADRepairError(
            "EmotionTalk feature sidecar contains duplicate/non-finite rows"
        )
    bounds = role_sidecar.FROZEN_ROLE_RANGES[role]
    if np.any((buckets < bounds[0]) | (buckets > bounds[1])):
        raise EmotionRelationVADRepairError(
            "EmotionTalk role sidecar bucket is outside its open range"
        )
    if int(record["groups"]) != len(set(groups.tolist())):
        raise EmotionRelationVADRepairError(
            "EmotionTalk manifest group count differs from feature sidecar"
        )
    history_eligible = int(
        sum(
            np.any(
                (groups == groups[index])
                & (speakers == speakers[index])
                & (turns < turns[index])
            )
            for index in range(rows)
        )
    )
    if history_eligible != int(record["history_eligible_rows"]):
        raise EmotionRelationVADRepairError(
            "EmotionTalk history-eligible count differs from manifest"
        )
    return {
        "row_hashes": row_hashes,
        "group_hashes": groups,
        "speaker_tokens": speakers,
        "turn_ids": turns,
        "protocol_row_ids": protocol_rows,
        "role_buckets": buckets,
        "texts": texts,
        "audio": audio,
        "video": video,
        "row_alignment_sha256": alignment,
    }


def _load_staged_fit_labels(
    *,
    path: Path,
    record: Mapping[str, object],
    source: Mapping[str, object],
    expected_alignment_sha256: str,
) -> np.ndarray:
    """Deserialize only fit labels; selection-label bytes stay opaque."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != role_sidecar.LABEL_FIELDS:
            raise EmotionRelationVADRepairError("EmotionTalk fit label sidecar schema changed")
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if role_sidecar._scalar_text(payload, "schema_version") != role_sidecar.LABEL_SCHEMA:
        raise EmotionRelationVADRepairError("EmotionTalk fit label sidecar version changed")
    if role_sidecar._scalar_text(payload, "dataset_id") != "EmotionTalk":
        raise EmotionRelationVADRepairError("EmotionTalk fit label sidecar dataset changed")
    if role_sidecar._scalar_text(payload, "role") != FIT_ROLE:
        raise EmotionRelationVADRepairError("EmotionTalk fit label sidecar role changed")
    if role_sidecar._scalar_text(payload, "split_protocol_id") != "scu_set_exploration_v1":
        raise EmotionRelationVADRepairError("EmotionTalk fit label sidecar split changed")
    alignment = _staged_sha256(
        role_sidecar._scalar_text(payload, "row_alignment_sha256"),
        field=f"{FIT_ROLE}.label.row_alignment_sha256",
    )
    if (
        alignment != expected_alignment_sha256
        or alignment
        != _staged_sha256(
            record["row_alignment_sha256"],
            field=f"roles.{FIT_ROLE}.row_alignment_sha256",
        )
    ):
        raise EmotionRelationVADRepairError("EmotionTalk fit feature/label alignment differs")
    if role_sidecar._scalar_text(payload, "source_label_sha256") != str(source["label_archive"]):
        raise EmotionRelationVADRepairError(
            "EmotionTalk fit label source hash differs from manifest"
        )
    labels = payload["labels"].astype(np.int64, copy=True)
    if labels.shape != (int(record["rows"]),):
        raise EmotionRelationVADRepairError("EmotionTalk fit label rows differ from manifest")
    if np.any((labels < 0) | (labels >= len(CLASS_ORDER))):
        raise EmotionRelationVADRepairError("EmotionTalk fit label is invalid")
    return labels


def _validate_staged_corpus(
    corpus: OpenRoleCorpus,
    model_config: CausalBackboneConfig,
) -> None:
    roles = set(corpus.roles.astype(str).tolist())
    if roles != {FIT_ROLE}:
        raise EmotionRelationVADRepairError("staged corpus must materialise only the fit role")
    if np.any((corpus.labels < 0) | (corpus.labels >= model_config.num_classes)):
        raise EmotionRelationVADRepairError("staged fit label is outside the class range")
    corpus.validate(model_config)


def _fit_stage_contract_sha256(
    corpus: OpenRoleCorpus,
    *,
    selection_feature_sha256: str,
    selection_label_sha256: str,
) -> str:
    """Bind all Stage-1 arrays plus the opaque selection-label commitment."""

    selection_feature_hash = _staged_sha256(
        selection_feature_sha256, field="selection_feature_sha256"
    )
    selection_label_hash = _staged_sha256(
        selection_label_sha256, field="selection_label_sha256"
    )
    speaker_identity = (
        np.asarray(corpus.speaker_ids, dtype=str)
        if corpus.speaker_identity is None
        else np.asarray(corpus.speaker_identity, dtype=str)
    )
    protocol_rows = (
        np.arange(len(corpus.keys), dtype=np.int64)
        if corpus.protocol_row_ids is None
        else np.asarray(corpus.protocol_row_ids, dtype=np.int64)
    )

    def numeric_vector_hash(values: np.ndarray) -> str:
        array = np.asarray(values)
        return numeric_matrix_content_sha256(array.reshape(len(array), -1))

    return _canonical_sha256(
        {
            "schema": "emotion_relation_vad_repair_fit_gate_stage1_v1",
            "keys": list(corpus.keys.astype(str)),
            "texts": list(corpus.texts),
            "audio": numeric_matrix_content_sha256(corpus.audio),
            "video": numeric_matrix_content_sha256(corpus.video),
            "fit_labels": numeric_vector_hash(corpus.labels),
            "selection_features": {
                "state": "opaque_sha256_verified_not_opened_not_deserialized",
                "sha256": selection_feature_hash,
            },
            "selection_labels": {
                "state": "opaque_sha256_verified_not_opened_not_deserialized",
                "sha256": selection_label_hash,
            },
            "groups": list(corpus.groups.astype(str)),
            "roles": list(corpus.roles.astype(str)),
            "buckets": numeric_vector_hash(corpus.buckets),
            "speaker_ids": numeric_vector_hash(corpus.speaker_ids),
            "speaker_identity": list(speaker_identity),
            "speaker_mapping_sha256": corpus.speaker_mapping_sha256,
            "turn_ids": numeric_vector_hash(corpus.turn_ids),
            "protocol_row_ids": numeric_vector_hash(protocol_rows),
            "histories": [list(map(int, values)) for values in corpus.histories],
            "role_assignment_sha256": _role_assignment_sha256(corpus),
            "stage_events": [
                "all_four_sidecar_sha256_verified",
                "fit_feature_deserialized",
                "fit_label_deserialized",
                "selection_feature_not_opened_not_deserialized",
                "selection_label_not_opened_not_deserialized",
            ],
        }
    )


def _staged_provenance_attestation_payload(
    provenance: StagedVerifiedCorpusProvenance,
) -> dict[str, object]:
    return {
        "dataset_id": provenance.dataset_id,
        "manifest_schema": provenance.manifest_schema,
        "manifest_status": provenance.manifest_status,
        "manifest_sha256": provenance.manifest_sha256,
        "source_hashes": dict(sorted(provenance.source_hashes.items())),
        "label_order": list(provenance.label_order),
        "role_rows": dict(sorted(provenance.role_rows.items())),
        "audio_dim": provenance.audio_dim,
        "video_dim": provenance.video_dim,
        "role_assignment_sha256": provenance.role_assignment_sha256,
        "speaker_mapping_sha256": provenance.speaker_mapping_sha256,
        "fit_stage_contract_sha256": provenance.fit_stage_contract_sha256,
        "verification_origin": provenance.verification_origin,
        "strict_role_feature_sidecars": provenance.strict_role_feature_sidecars,
        "strict_role_label_sidecars": provenance.strict_role_label_sidecars,
        "selection_feature_hash_verified": provenance.selection_feature_hash_verified,
        "selection_feature_payload_opened": provenance.selection_feature_payload_opened,
        "selection_feature_deserialized": provenance.selection_feature_deserialized,
        "selection_label_hash_verified": provenance.selection_label_hash_verified,
        "selection_label_payload_opened": provenance.selection_label_payload_opened,
        "selection_label_deserialized": provenance.selection_label_deserialized,
        "sealed_role_arrays_opened": provenance.sealed_role_arrays_opened,
        "validation_or_test_opened": provenance.validation_or_test_opened,
    }


def _create_staged_provenance(
    *,
    corpus: OpenRoleCorpus,
    manifest: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> StagedVerifiedCorpusProvenance:
    selection_feature_sha = source_hashes[f"{SELECTION_ROLE}_features"]
    selection_label_sha = source_hashes[f"{SELECTION_ROLE}_labels"]
    provisional = StagedVerifiedCorpusProvenance(
        dataset_id="EmotionTalk",
        manifest_schema=str(manifest["schema_version"]),
        manifest_status=str(manifest["status"]),
        manifest_sha256=str(source_hashes["sidecar_manifest"]),
        source_hashes=MappingProxyType(dict(source_hashes)),
        label_order=tuple(str(value) for value in manifest["label_order"]),
        role_rows=MappingProxyType(
            {
                role: int(_mapping(manifest["roles"], name="roles")[role]["rows"])
                for role in (FIT_ROLE, SELECTION_ROLE)
            }
        ),
        audio_dim=int(_mapping(manifest["roles"], name="roles")[FIT_ROLE]["audio_dimension"]),
        video_dim=int(_mapping(manifest["roles"], name="roles")[FIT_ROLE]["video_dimension"]),
        role_assignment_sha256=_role_assignment_sha256(corpus),
        speaker_mapping_sha256=corpus.speaker_mapping_sha256,
        fit_stage_contract_sha256=_fit_stage_contract_sha256(
            corpus,
            selection_feature_sha256=selection_feature_sha,
            selection_label_sha256=selection_label_sha,
        ),
        verification_origin="emotiontalk_manifest_v2_fit_gate_stage1",
        verifier_attestation_sha256="0" * 64,
    )
    return replace(
        provisional,
        verifier_attestation_sha256=_canonical_sha256(
            _staged_provenance_attestation_payload(provisional)
        ),
    )


def _combine_staged_role_arrays(
    role_arrays: Mapping[str, role_sidecar.EmotionTalkRoleArrays],
    *,
    model_config: CausalBackboneConfig,
) -> OpenRoleCorpus:
    fit_speakers = sorted(set(role_arrays[FIT_ROLE].speaker_tokens.astype(str)))
    speaker_mapping = {value: index + 1 for index, value in enumerate(fit_speakers)}
    if len(speaker_mapping) + 1 > model_config.num_speakers:
        raise EmotionRelationVADRepairError(
            "fit-only EmotionTalk speaker vocabulary exceeds model config"
        )
    speaker_mapping_sha = _canonical_sha256(
        {"oov": 0, "fit_mapping": [[value, speaker_mapping[value]] for value in fit_speakers]}
    )
    values = [role_arrays[FIT_ROLE]]
    ordering = np.argsort(
        np.concatenate([value.protocol_row_ids for value in values]), kind="stable"
    )

    def combined(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(value, name)) for value in values], axis=0
        )[ordering]

    keys = combined("row_hashes").astype(str)
    groups = combined("group_hashes").astype(str)
    speaker_tokens = combined("speaker_tokens").astype(str)
    roles = np.asarray([FIT_ROLE] * len(keys))
    labels = combined("labels").astype(np.int64)
    buckets = combined("role_buckets").astype(np.int16)
    turns = combined("turn_ids").astype(np.int64)
    protocol_rows = combined("protocol_row_ids").astype(np.int64)
    texts = combined("texts").astype(str)
    audio = combined("audio").astype(np.float32)
    video = combined("video").astype(np.float32)
    speaker_ids = np.asarray(
        [speaker_mapping.get(value, 0) for value in speaker_tokens], dtype=np.int64
    )
    speaker_identity = np.asarray(
        [hashlib.sha256(f"speaker\x1f{value}".encode()).hexdigest() for value in speaker_tokens]
    )
    corpus = OpenRoleCorpus(
        keys=keys,
        texts=tuple(texts),
        audio=audio,
        video=video,
        labels=labels,
        groups=groups,
        roles=roles,
        buckets=buckets,
        speaker_ids=speaker_ids,
        turn_ids=turns,
        histories=_history_indices(groups, speaker_identity, turns),
        protocol_row_ids=protocol_rows,
        speaker_identity=speaker_identity,
        speaker_mapping_sha256=speaker_mapping_sha,
        label_access_mode="fit_role_only_selection_feature_and_label_sha256_verified_opaque",
    )
    _validate_staged_corpus(corpus, model_config)
    return corpus


def load_emotiontalk_fit_gate_stage(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    model_config: CausalBackboneConfig,
) -> StagedOpenRoleLoad:
    """Load Stage 1 while keeping selection-label bytes physically opaque."""

    sidecar_dir = Path(sidecar_dir)
    manifest_path = Path(manifest_path)
    manifest, paths, observed_hashes = _validate_staged_manifest_and_hashes(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        model_config=model_config,
    )
    roles_manifest = _mapping(manifest["roles"], name="roles")
    source = _mapping(manifest["source_contract"], name="source_contract")
    fit_feature_payload = _load_staged_feature_payload(
        role=FIT_ROLE,
        path=paths[(FIT_ROLE, "features")],
        record=_mapping(roles_manifest[FIT_ROLE], name=f"roles.{FIT_ROLE}"),
        source=source,
    )
    fit_labels = _load_staged_fit_labels(
        path=paths[(FIT_ROLE, "labels")],
        record=_mapping(roles_manifest[FIT_ROLE], name=f"roles.{FIT_ROLE}"),
        source=source,
        expected_alignment_sha256=str(fit_feature_payload["row_alignment_sha256"]),
    )
    # Close the hash/load race for all four files.  Both selection files are
    # hashed a second time; neither is handed to np.load in Stage 1.
    for (role, kind), path in paths.items():
        name = f"{role}_{kind}"
        if sha256_file(path) != observed_hashes[name]:
            raise EmotionRelationVADRepairError(
                "EmotionTalk sidecar changed during staged verification"
            )

    values = fit_feature_payload
    role_arrays = {
        FIT_ROLE: role_sidecar.EmotionTalkRoleArrays(
            role=FIT_ROLE,
            row_hashes=np.asarray(values["row_hashes"]),
            group_hashes=np.asarray(values["group_hashes"]),
            speaker_tokens=np.asarray(values["speaker_tokens"]),
            turn_ids=np.asarray(values["turn_ids"]),
            protocol_row_ids=np.asarray(values["protocol_row_ids"]),
            role_buckets=np.asarray(values["role_buckets"]),
            texts=np.asarray(values["texts"]),
            audio=np.asarray(values["audio"]),
            video=np.asarray(values["video"]),
            labels=fit_labels,
            row_alignment_sha256=str(values["row_alignment_sha256"]),
            feature_sha256=observed_hashes[f"{FIT_ROLE}_features"],
            label_sha256=observed_hashes[f"{FIT_ROLE}_labels"],
        )
    }
    corpus = _combine_staged_role_arrays(role_arrays, model_config=model_config)
    source_hashes = {
        "sidecar_manifest": observed_hashes["sidecar_manifest"],
        "trusted_source_label_archive": str(source["label_archive"]),
        "trusted_source_media_features": str(source["media_features"]),
        "trusted_source_transcription": str(source["transcription"]),
        **{
            name: digest
            for name, digest in observed_hashes.items()
            if name != "sidecar_manifest"
        },
    }
    provenance = _create_staged_provenance(
        corpus=corpus,
        manifest=manifest,
        source_hashes=source_hashes,
    )
    provenance.validate(corpus, model_config)
    return StagedOpenRoleLoad(
        corpus=corpus,
        provenance=provenance,
        manifest=MappingProxyType(dict(manifest)),
    )


def _extract_fit_only_corpus(
    corpus: OpenRoleCorpus,
    *,
    model_config: CausalBackboneConfig,
) -> OpenRoleCorpus:
    fit_rows = corpus.role_indices(FIT_ROLE)
    local_by_global = {int(row): local for local, row in enumerate(fit_rows)}
    histories: list[tuple[int, ...]] = []
    for query in fit_rows:
        try:
            histories.append(
                tuple(
                    local_by_global[int(candidate)]
                    for candidate in corpus.histories[int(query)]
                )
            )
        except KeyError as error:
            raise EmotionRelationVADRepairError(
                "full corpus fit history crosses a role boundary"
            ) from error
    protocol_rows = (
        np.arange(len(corpus.keys), dtype=np.int64)
        if corpus.protocol_row_ids is None
        else np.asarray(corpus.protocol_row_ids, dtype=np.int64)
    )
    speaker_identity = (
        np.asarray(corpus.speaker_ids, dtype=str)
        if corpus.speaker_identity is None
        else np.asarray(corpus.speaker_identity, dtype=str)
    )
    result = OpenRoleCorpus(
        keys=np.asarray(corpus.keys[fit_rows]).copy(),
        texts=tuple(corpus.texts[int(row)] for row in fit_rows),
        audio=np.asarray(corpus.audio[fit_rows]).copy(),
        video=np.asarray(corpus.video[fit_rows]).copy(),
        labels=np.asarray(corpus.labels[fit_rows], dtype=np.int64).copy(),
        groups=np.asarray(corpus.groups[fit_rows]).copy(),
        roles=np.asarray(corpus.roles[fit_rows]).copy(),
        buckets=np.asarray(corpus.buckets[fit_rows]).copy(),
        speaker_ids=np.asarray(corpus.speaker_ids[fit_rows], dtype=np.int64).copy(),
        turn_ids=np.asarray(corpus.turn_ids[fit_rows], dtype=np.int64).copy(),
        histories=tuple(histories),
        protocol_row_ids=np.asarray(protocol_rows[fit_rows], dtype=np.int64).copy(),
        speaker_identity=np.asarray(speaker_identity[fit_rows]).copy(),
        speaker_mapping_sha256=corpus.speaker_mapping_sha256,
        label_access_mode="fit_role_only_selection_feature_and_label_sha256_verified_opaque",
    )
    _validate_staged_corpus(result, model_config)
    return result


def deterministic_fit_internal_gate_split(
    corpus: OpenRoleCorpus,
) -> FitInternalGateSplit:
    """Assign physical-fit groups without inspecting labels or result metrics."""

    fit_rows = corpus.role_indices(FIT_ROLE)
    if not len(fit_rows):
        raise EmotionRelationVADRepairError("fit-internal gate received no fit rows")
    fit_groups = np.asarray(corpus.groups[fit_rows], dtype=str)
    unique_groups = tuple(sorted(set(fit_groups.tolist())))
    if len(unique_groups) < 6:
        raise EmotionRelationVADRepairError(
            "fit-internal gate requires at least six opaque fit groups"
        )
    scored = tuple(
        sorted(
            (
                hashlib.sha256(
                    f"{FIT_INTERNAL_GATE_NAMESPACE}\x1f{group}".encode("utf-8")
                ).hexdigest(),
                group,
            )
            for group in unique_groups
        )
    )
    n_eval = int(math.ceil(FIT_INTERNAL_GATE_EVAL_FRACTION * len(unique_groups)))
    if not 0 < n_eval < len(unique_groups):
        raise EmotionRelationVADRepairError("fit-internal gate split is degenerate")
    eval_groups = tuple(group for _, group in scored[:n_eval])
    train_groups = tuple(group for _, group in scored[n_eval:])
    eval_set = set(eval_groups)
    train_set = set(train_groups)
    train_rows = tuple(
        int(row) for row in fit_rows if str(corpus.groups[int(row)]) in train_set
    )
    eval_rows = tuple(
        int(row) for row in fit_rows if str(corpus.groups[int(row)]) in eval_set
    )
    if set(train_rows) & set(eval_rows) or set(train_rows) | set(eval_rows) != set(
        map(int, fit_rows)
    ):
        raise AssertionError("fit-internal gate row coverage is not an exact partition")

    def history_eligible(rows: Sequence[int]) -> int:
        return int(sum(bool(corpus.histories[int(row)]) for row in rows))

    def class_complete(rows: Sequence[int]) -> bool:
        return set(np.asarray(corpus.labels[list(rows)], dtype=np.int64).tolist()) == set(
            range(len(CLASS_ORDER))
        )

    ordered_groups = tuple(group for _, group in scored)
    group_assignments = tuple(
        (group, "gate_eval" if group in eval_set else "gate_train")
        for group in ordered_groups
    )
    row_assignments = tuple(
        sorted(
            (
                str(corpus.keys[int(row)]),
                "gate_eval"
                if str(corpus.groups[int(row)]) in eval_set
                else "gate_train",
            )
            for row in fit_rows
        )
    )
    return FitInternalGateSplit(
        namespace=FIT_INTERNAL_GATE_NAMESPACE,
        split_spec_sha256=FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256,
        ordered_input_group_sha256=hashlib.sha256(
            "\n".join(ordered_groups).encode("utf-8")
        ).hexdigest(),
        group_assignment_sha256=_canonical_sha256(
            {
                "namespace": FIT_INTERNAL_GATE_NAMESPACE,
                "assignments": group_assignments,
            }
        ),
        row_assignment_sha256=_canonical_sha256(
            {
                "namespace": FIT_INTERNAL_GATE_NAMESPACE,
                "assignments": row_assignments,
            }
        ),
        train_groups=train_groups,
        eval_groups=eval_groups,
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_history_eligible_rows=history_eligible(train_rows),
        eval_history_eligible_rows=history_eligible(eval_rows),
        train_class_complete=class_complete(train_rows),
        eval_class_complete=class_complete(eval_rows),
    )


def build_fit_internal_gate_corpus(
    corpus: OpenRoleCorpus,
    split: FitInternalGateSplit,
    *,
    model_config: CausalBackboneConfig,
) -> OpenRoleCorpus:
    """Extract physical fit rows and expose fixed train/eval roles in memory."""

    if split.namespace != FIT_INTERNAL_GATE_NAMESPACE:
        raise EmotionRelationVADRepairError("fit-internal gate namespace changed")
    if split.split_spec_sha256 != FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256:
        raise EmotionRelationVADRepairError("fit-internal gate split spec changed")
    if not split.eligible:
        raise EmotionRelationVADRepairError(
            "fit_internal_gate_no_go: both partitions require all seven classes "
            "and at least one history-eligible row"
        )
    fit_rows = corpus.role_indices(FIT_ROLE)
    if set(map(int, fit_rows)) != set(split.train_rows) | set(split.eval_rows):
        raise EmotionRelationVADRepairError("fit-internal split no longer covers fit rows")
    local_by_global = {int(row): local for local, row in enumerate(fit_rows)}
    histories: list[tuple[int, ...]] = []
    for query in fit_rows:
        histories.append(
            tuple(local_by_global[int(candidate)] for candidate in corpus.histories[int(query)])
        )
    train_groups = set(split.train_groups)
    roles = np.asarray(
        [
            FIT_ROLE if str(corpus.groups[int(row)]) in train_groups else SELECTION_ROLE
            for row in fit_rows
        ]
    )
    protocol_rows = (
        np.arange(len(corpus.keys), dtype=np.int64)
        if corpus.protocol_row_ids is None
        else np.asarray(corpus.protocol_row_ids, dtype=np.int64)
    )
    speaker_identity = (
        np.asarray(corpus.speaker_ids, dtype=str)
        if corpus.speaker_identity is None
        else np.asarray(corpus.speaker_identity, dtype=str)
    )
    result = OpenRoleCorpus(
        keys=np.asarray(corpus.keys[fit_rows]).copy(),
        texts=tuple(corpus.texts[int(row)] for row in fit_rows),
        audio=np.asarray(corpus.audio[fit_rows]).copy(),
        video=np.asarray(corpus.video[fit_rows]).copy(),
        labels=np.asarray(corpus.labels[fit_rows], dtype=np.int64).copy(),
        groups=np.asarray(corpus.groups[fit_rows]).copy(),
        roles=roles,
        buckets=np.asarray(corpus.buckets[fit_rows]).copy(),
        speaker_ids=np.asarray(corpus.speaker_ids[fit_rows], dtype=np.int64).copy(),
        turn_ids=np.asarray(corpus.turn_ids[fit_rows], dtype=np.int64).copy(),
        histories=tuple(histories),
        protocol_row_ids=np.asarray(protocol_rows[fit_rows], dtype=np.int64).copy(),
        speaker_identity=np.asarray(speaker_identity[fit_rows]).copy(),
        speaker_mapping_sha256=corpus.speaker_mapping_sha256,
        label_access_mode=(
            "fit_internal_gate_train_labels_for_training_eval_labels_scoring_only"
        ),
    )
    result.validate(model_config)
    if any(
        str(result.roles[candidate]) != str(result.roles[query])
        for query, history in enumerate(result.histories)
        for candidate in history
    ):
        raise AssertionError("fit-internal gate split broke a group history")
    return result


def materialize_selection_labels_after_fit_gate(
    staged: StagedOpenRoleLoad,
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    model_config: CausalBackboneConfig,
) -> tuple[OpenRoleCorpus, VerifiedCorpusProvenance]:
    """Materialise and align selection labels only after the fit gate passes."""

    staged.provenance.validate(staged.corpus, model_config)
    if sha256_file(Path(manifest_path)) != staged.provenance.manifest_sha256:
        raise EmotionRelationVADRepairError("sidecar manifest changed after the fit gate")
    corpus, provenance = load_emotiontalk_open_role_corpus(
        sidecar_dir=Path(sidecar_dir),
        manifest_path=Path(manifest_path),
        model_config=model_config,
    )
    provenance.validate(corpus, model_config)
    comparable_provenance_fields = (
        "dataset_id",
        "manifest_schema",
        "manifest_status",
        "manifest_sha256",
        "source_hashes",
        "label_order",
        "role_rows",
        "audio_dim",
        "video_dim",
        "speaker_mapping_sha256",
    )
    for name in comparable_provenance_fields:
        staged_value = getattr(staged.provenance, name)
        full_value = getattr(provenance, name)
        if isinstance(staged_value, Mapping):
            if dict(staged_value) != dict(full_value):
                raise EmotionRelationVADRepairError(
                    f"Stage-1/Stage-2 provenance differs at {name}"
                )
        elif staged_value != full_value:
            raise EmotionRelationVADRepairError(
                f"Stage-1/Stage-2 provenance differs at {name}"
            )

    staged_corpus = staged.corpus
    full_fit_corpus = _extract_fit_only_corpus(corpus, model_config=model_config)
    for name in (
        "keys",
        "audio",
        "video",
        "groups",
        "roles",
        "buckets",
        "speaker_ids",
        "turn_ids",
        "protocol_row_ids",
        "speaker_identity",
    ):
        if not np.array_equal(
            np.asarray(getattr(staged_corpus, name)),
            np.asarray(getattr(full_fit_corpus, name)),
        ):
            raise EmotionRelationVADRepairError(
                f"Stage-1/Stage-2 corpus alignment differs at {name}"
            )
    if (
        staged_corpus.texts != full_fit_corpus.texts
        or staged_corpus.histories != full_fit_corpus.histories
    ):
        raise EmotionRelationVADRepairError("Stage-1/Stage-2 text/history alignment differs")
    if not np.array_equal(staged_corpus.labels, full_fit_corpus.labels):
        raise EmotionRelationVADRepairError("fit labels changed during Stage-2 materialisation")
    selection_rows = corpus.role_indices(SELECTION_ROLE)
    if np.any((corpus.labels[selection_rows] < 0) | (corpus.labels[selection_rows] >= len(CLASS_ORDER))):
        raise EmotionRelationVADRepairError("Stage-2 selection labels are invalid")
    staged.provenance.validate(full_fit_corpus, model_config)
    return corpus, provenance


def load_emotion_relation_vad_repair_config(path: str | Path) -> FrozenRepairConfig:
    """Load and fully validate the frozen Repair-3 registration."""

    raw = _load_json(Path(path), name="Repair-3 config")
    if raw.get("protocol") != PROTOCOL or raw.get("status") != REGISTERED_STATUS:
        raise EmotionRelationVADRepairError("Repair-3 protocol/freeze status changed")
    if raw.get("primary_variant") != PRIMARY_VARIANT:
        raise EmotionRelationVADRepairError("primary feature variant changed")
    if tuple(raw.get("variant_order", ())) != VARIANT_ORDER:
        raise EmotionRelationVADRepairError("primary/ablation registration changed")
    if _mapping(raw.get("variant_widths"), name="variant_widths") != dict(VARIANT_WIDTHS):
        raise EmotionRelationVADRepairError("registered feature geometry changed")
    if tuple(raw.get("class_order", ())) != CLASS_ORDER:
        raise EmotionRelationVADRepairError("emotion class order changed")
    if tuple(int(value) for value in raw.get("base_seeds", ())) != BASE_SEEDS:
        raise EmotionRelationVADRepairError("base seeds changed")
    if tuple(int(value) for value in raw.get("utility_seeds", ())) != UTILITY_SEEDS:
        raise EmotionRelationVADRepairError("utility seeds changed")
    if raw.get("registered_output_filename") != REGISTERED_OUTPUT_FILENAME:
        raise EmotionRelationVADRepairError("registered output filename changed")
    if raw.get("registered_output_repository_relative_path") != (
        REGISTERED_OUTPUT_REPOSITORY_RELATIVE_PATH
    ):
        raise EmotionRelationVADRepairError("registered output path changed")

    upstream = _mapping(raw.get("frozen_repair_2"), name="frozen_repair_2")
    if upstream.get("config_sha256") != REPAIR2_CONFIG_SHA256 or upstream.get(
        "result_sha256"
    ) != REPAIR2_RESULT_SHA256:
        raise EmotionRelationVADRepairError("Repair-2 frozen lineage changed")

    vad = _mapping(raw.get("vad"), name="vad")
    if vad.get("coordinate_system") != VAD_COORDINATE_SYSTEM:
        raise EmotionRelationVADRepairError("VAD coordinate system changed")
    if vad.get("claim_boundary") != VAD_CLAIM_BOUNDARY:
        raise EmotionRelationVADRepairError("VAD claim boundary changed")
    config_coordinates = _mapping(vad.get("coordinates"), name="vad.coordinates")
    coordinates = {
        str(name): tuple(float(value) for value in values)
        for name, values in config_coordinates.items()
    }
    if coordinates != dict(VAD_COORDINATES):
        raise EmotionRelationVADRepairError("VAD coordinate values changed")
    if vad.get("coordinate_sha256") != VAD_COORDINATE_SHA256:
        raise EmotionRelationVADRepairError("VAD coordinate hash changed")
    if vad.get("feature_source") != "seven_class_predicted_posterior_expectation_only":
        raise EmotionRelationVADRepairError("VAD feature source changed")

    input_locks = _mapping(raw.get("registered_input_locks"), name="registered_input_locks")
    expected_input_locks = {
        "sidecar_manifest_sha256": REGISTERED_SIDECAR_MANIFEST_SHA256,
        "model_config_sha256": REGISTERED_MODEL_CONFIG_SHA256,
        "enforced_before_sidecar_deserialization_or_model_training": True,
    }
    if dict(input_locks) != expected_input_locks:
        raise EmotionRelationVADRepairError("registered input-file locks changed")

    split_config = _mapping(raw.get("fit_internal_gate_split"), name="fit_internal_gate_split")
    expected_split_config = {
        **dict(FIT_INTERNAL_GATE_SPLIT_SPEC),
        "split_spec_sha256": FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256,
        "eligibility": (
            "both partitions contain all seven classes and at least one "
            "history-eligible row; otherwise NO-GO"
        ),
    }
    if dict(split_config) != expected_split_config:
        raise EmotionRelationVADRepairError("fit-internal gate split registration changed")

    capacity = _mapping(raw.get("capacity_control"), name="capacity_control")
    canonical_capacity_json = json.dumps(
        dict(CAPACITY_CONTROL_SPEC),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    expected_capacity = {
        "variant": CAPACITY_CONTROL_VARIANT,
        "descriptor": dict(CAPACITY_CONTROL_SPEC),
        "canonical_json": canonical_capacity_json,
        "spec_sha256": CAPACITY_CONTROL_SPEC_SHA256,
        "matrix_sha256": CAPACITY_CONTROL_MATRIX_SHA256,
        "rank": 59,
        "maximum_eet_identity_error": 5e-16,
        "superseded_handoff_only_unverifiable_spec_sha256": (
            SUPERSEDED_UNVERIFIABLE_CAPACITY_SPEC_SHA256
        ),
        "supersession_reason": (
            "The prior handoff supplied only a truncated hash and no reproducible "
            "descriptor. Before any real fit-gate result was opened, the exact "
            "canonical descriptor above was frozen so its hash can be recomputed "
            "rather than hard-coded without provenance."
        ),
    }
    if dict(capacity) != expected_capacity:
        raise EmotionRelationVADRepairError("capacity-control registration changed")
    if _canonical_sha256(dict(CAPACITY_CONTROL_SPEC)) != CAPACITY_CONTROL_SPEC_SHA256:
        raise EmotionRelationVADRepairError("capacity-control spec is not reproducible")
    expansion = capacity_control_expansion_matrix()
    if numeric_matrix_content_sha256(expansion) != CAPACITY_CONTROL_MATRIX_SHA256:
        raise EmotionRelationVADRepairError("capacity-control matrix is not reproducible")

    gate = _mapping(raw.get("fit_only_open_gate"), name="fit_only_open_gate")
    expected_gate = {
        "candidate_variant": PRIMARY_VARIANT,
        "reference_variants": ["base_59d_only", CAPACITY_CONTROL_VARIANT],
        "utility_model": "class_balanced_true_bidirectional_mlp",
        "minimum_macro_f1_gain_vs_each_reference": FIT_GATE_MACRO_F1_GAIN,
        "nll_maximum_worsening_vs_each_reference": 0.0,
        "accuracy_minimum_gain_vs_each_reference": 0.0,
        "same_seed_six_condition_intersection": True,
        "minimum_successful_utility_seeds_out_of_five": FIT_GATE_REQUIRED_SEEDS,
        "selection_prediction_if_fail": False,
        "selection_label_scoring_if_fail": False,
    }
    if dict(gate) != expected_gate:
        raise EmotionRelationVADRepairError("fit-only gate changed")

    selection_contract = _mapping(
        raw.get("model_selection_contract"), name="model_selection_contract"
    )
    expected_selection_contract = {
        "primary_fixed_before_results": True,
        "ablations_explanatory_only": True,
        "selection_can_choose_variant": False,
        "selection_can_choose_relation_layer": False,
        "selection_can_choose_class_mapping": False,
        "selection_can_choose_fusion_weight": False,
        "registered_and_stronger_gates_require_accuracy_non_worsening": True,
        "fixed_references": [
            "current_only",
            "all_history",
            "coverage_matched_recency",
            "class_balanced_backward_mlp",
        ],
        "prediction_unit": "one prediction per query",
        "seed_grid": "five utility seeds by five base seeds",
    }
    if dict(selection_contract) != expected_selection_contract:
        raise EmotionRelationVADRepairError("model-selection contract changed")

    teacher = _mapping(raw.get("optional_teacher"), name="optional_teacher")
    expected_teacher = {
        "status": "predeclared_not_downloaded_not_executed",
        "model_revision": (
            "MilaNLProc/xlm-emo-t@a6ee7c9fad08d60204e7ae437d41d392381496f0"
        ),
        "role": "four-class auxiliary text feature only, never seven-class supervision",
        "fit_oof_eligibility": {
            "minimum_successful_seeds_out_of_five": 4,
            "minimum_macro_f1_gain": FIT_GATE_MACRO_F1_GAIN,
            "nll_maximum_worsening": 0.0,
            "accuracy_minimum_gain": 0.0,
            "same_seed_macro_nll_accuracy_intersection": True,
        },
        "selection_can_choose_teacher": False,
        "selection_can_choose_layer": False,
        "selection_can_choose_class_mapping": False,
        "johnson_chinese_model": "no_go",
    }
    if dict(teacher) != expected_teacher:
        raise EmotionRelationVADRepairError("optional-teacher registration changed")

    access = _mapping(raw.get("data_access_staging"), name="data_access_staging")
    if dict(access) != dict(FIT_GATE_STAGE_ACCESS_CONTRACT):
        raise EmotionRelationVADRepairError("fit-gate data-access staging changed")

    projector_map = _mapping(raw.get("unimodal_projector"), name="unimodal_projector")
    projector = ProjectorSpec(
        **{
            name: projector_map[name]
            for name in ProjectorSpec.__dataclass_fields__
            if name in projector_map
        }
    )
    projector.validate()
    sampling = UtilitySamplingConfig.from_mapping(raw)
    if sampling != UtilitySamplingConfig(
        draws_per_query=8,
        maximum_candidates=16,
        seed=20260808,
        match_context_cardinality=True,
    ):
        raise EmotionRelationVADRepairError("counterfactual sampling registration changed")

    balance_map = _mapping(raw.get("class_balance"), name="class_balance")
    balance = ClassBalanceSpec(
        scheme=str(balance_map.get("scheme", "")),
        beta=float(balance_map.get("beta", math.nan)),
        resample_size_multiplier=float(balance_map.get("resample_size_multiplier", math.nan)),
        frequency_unit=str(balance_map.get("frequency_unit", "")),
        task_weight_rule=str(balance_map.get("task_weight_rule", "")),
        oof_frequency_scope=str(balance_map.get("oof_frequency_scope", "")),
        final_frequency_scope=str(balance_map.get("final_frequency_scope", "")),
    )
    expected_balance = ClassBalanceSpec()
    if balance != expected_balance:
        raise EmotionRelationVADRepairError("Repair-2 class balance policy changed")
    utility_specs = default_capacity_matched_specs()
    architecture = _mapping(raw.get("utility_architecture"), name="utility_architecture")
    for spec in utility_specs:
        for field_name in (
            "hidden_layer_sizes",
            "alpha",
            "max_iter",
            "tolerance",
            "activation",
            "solver",
            "batch_size",
            "learning_rate_init",
            "early_stopping",
            "validation_fraction",
            "n_iter_no_change",
        ):
            expected_value = getattr(spec, field_name)
            actual = architecture.get(field_name)
            if isinstance(expected_value, tuple):
                actual = tuple(int(value) for value in actual or ())
            if actual != expected_value:
                raise EmotionRelationVADRepairError(
                    f"Repair-2 utility architecture changed at {field_name}"
                )
    expected_parameter_counts = {
        PRIMARY_VARIANT: PRIMARY_OR_CONTROL_PARAMETER_COUNT,
        CAPACITY_CONTROL_VARIANT: PRIMARY_OR_CONTROL_PARAMETER_COUNT,
        "base_59d_only": BASE_59D_PARAMETER_COUNT,
    }
    if architecture.get("trainable_parameters") != expected_parameter_counts:
        raise EmotionRelationVADRepairError("registered utility parameter counts changed")
    for spec in utility_specs:
        if trainable_parameter_count(299, spec.hidden_layer_sizes, 2) != (
            PRIMARY_OR_CONTROL_PARAMETER_COUNT
        ):
            raise EmotionRelationVADRepairError("299-D utility capacity changed")
        if trainable_parameter_count(59, spec.hidden_layer_sizes, 2) != (
            BASE_59D_PARAMETER_COUNT
        ):
            raise EmotionRelationVADRepairError("59-D utility capacity changed")
    if float(raw.get("fusion_current_weight", math.nan)) != FUSION_CURRENT_WEIGHT:
        raise EmotionRelationVADRepairError("current/history fusion weight changed")
    return FrozenRepairConfig(raw, projector, sampling, balance, utility_specs)


def _aligned_probability(classifier: SGDClassifier, values: object) -> np.ndarray:
    classes = tuple(int(value) for value in np.asarray(classifier.classes_).tolist())
    if classes != tuple(range(len(CLASS_ORDER))):
        raise EmotionRelationVADRepairError(
            "every fit fold must contain all seven emotion classes in canonical order"
        )
    probability = np.asarray(classifier.predict_proba(values), dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(CLASS_ORDER):
        raise EmotionRelationVADRepairError("projector posterior shape changed")
    probability = np.clip(probability, 0.0, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _prepare_modality_features(
    corpus: OpenRoleCorpus,
    modality: str,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    spec: ProjectorSpec,
) -> tuple[object, object]:
    if modality == "text":
        vectorizer = TfidfVectorizer(
            analyzer=spec.text_analyzer,
            ngram_range=(spec.text_ngram_min, spec.text_ngram_max),
            min_df=spec.text_min_df,
            max_features=spec.text_max_features,
            sublinear_tf=spec.text_sublinear_tf,
            dtype=np.float64,
        )
        train = vectorizer.fit_transform([corpus.texts[int(row)] for row in train_rows])
        predicted = vectorizer.transform([corpus.texts[int(row)] for row in predict_rows])
        return train, predicted
    matrix = np.asarray(corpus.audio if modality == "audio" else corpus.video, dtype=np.float64)
    scaler = StandardScaler()
    train = scaler.fit_transform(matrix[train_rows])
    predicted = scaler.transform(matrix[predict_rows])
    return train, predicted


def _new_projector(spec: ProjectorSpec, seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss=spec.loss,
        penalty=spec.penalty,
        alpha=spec.alpha,
        max_iter=spec.max_iter,
        tol=spec.tolerance,
        class_weight=spec.class_weight,
        average=spec.average,
        random_state=int(seed),
        shuffle=True,
        early_stopping=False,
    )


def _producer_config_sha256(spec: ProjectorSpec) -> str:
    return _canonical_sha256(
        {
            "protocol": PROTOCOL,
            "producer": "three_independent_seven_class_sgd_logistic_projectors",
            "projector": spec.as_dict(),
            "base_seeds": BASE_SEEDS,
            "class_order": CLASS_ORDER,
            "fusion": {
                "modalities": MODALITIES,
                "modality_rule": "arithmetic_mean",
                "current_weight": FUSION_CURRENT_WEIGHT,
                "history_weight": 1.0 - FUSION_CURRENT_WEIGHT,
            },
        }
    )


def fit_fit_role_group_oof_posteriors(
    corpus: OpenRoleCorpus,
    spec: ProjectorSpec,
) -> RolePosteriorGrid:
    """Fit whole-group OOF projectors without generating selection predictions."""

    spec.validate()
    fit_rows = corpus.role_indices(FIT_ROLE)
    groups = np.asarray(corpus.groups[fit_rows], dtype=str)
    labels = np.asarray(corpus.labels[fit_rows], dtype=np.int64)
    n_splits = min(spec.folds, len(np.unique(groups)))
    if n_splits != 5:
        raise EmotionRelationVADRepairError("Repair 3 requires exactly five fit groups/folds")
    probabilities = {
        modality: np.full(
            (len(BASE_SEEDS), len(fit_rows), len(CLASS_ORDER)), np.nan, dtype=np.float64
        )
        for modality in MODALITIES
    }
    fold_by_local = np.full(len(fit_rows), -1, dtype=np.int32)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_local, held_local) in enumerate(
        splitter.split(fit_rows, labels, groups=groups)
    ):
        train_rows = fit_rows[train_local]
        held_rows = fit_rows[held_local]
        if set(groups[train_local]) & set(groups[held_local]):
            raise AssertionError("emotion projector OOF leaked a group")
        if len(np.unique(labels[train_local])) != len(CLASS_ORDER):
            raise EmotionRelationVADRepairError("an OOF training fold lacks an emotion class")
        fold_by_local[held_local] = fold
        for modality in MODALITIES:
            train_x, held_x = _prepare_modality_features(
                corpus, modality, train_rows, held_rows, spec
            )
            for seed_index, seed in enumerate(BASE_SEEDS):
                classifier = _new_projector(spec, seed + fold * 10_007)
                classifier.fit(train_x, labels[train_local])
                probabilities[modality][seed_index, held_local] = _aligned_probability(
                    classifier, held_x
                )
    if np.any(fold_by_local < 0) or any(
        not np.isfinite(values).all() for values in probabilities.values()
    ):
        raise AssertionError("fit OOF posterior coverage is incomplete")
    fold_hash = _canonical_sha256(
        {
            "role": FIT_ROLE,
            "protocol_row_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.asarray(corpus.protocol_row_ids, dtype=np.int64)[fit_rows]
                ).tobytes()
            ).hexdigest(),
            "fold_by_local_row": fold_by_local.tolist(),
            "group_whole_fold": True,
        }
    )
    return RolePosteriorGrid(
        role=FIT_ROLE,
        row_indices=fit_rows,
        probabilities=probabilities,
        base_seeds=BASE_SEEDS,
        fold_by_local_row=fold_by_local,
        fold_assignment_sha256=fold_hash,
        producer_config_sha256=_producer_config_sha256(spec),
    )


def fit_full_fit_predict_selection_posteriors(
    corpus: OpenRoleCorpus,
    spec: ProjectorSpec,
    *,
    fit_oof: RolePosteriorGrid,
) -> RolePosteriorGrid:
    """Predict selection only after the fit-only gate has passed.

    Selection labels are deliberately never indexed in this function.
    """

    if fit_oof.role != FIT_ROLE or fit_oof.producer_config_sha256 != _producer_config_sha256(spec):
        raise EmotionRelationVADRepairError("fit OOF producer lineage changed")
    fit_rows = corpus.role_indices(FIT_ROLE)
    selection_rows = corpus.role_indices(SELECTION_ROLE)
    fit_labels = np.asarray(corpus.labels[fit_rows], dtype=np.int64)
    if len(np.unique(fit_labels)) != len(CLASS_ORDER):
        raise EmotionRelationVADRepairError("full fit role lacks an emotion class")
    probabilities = {
        modality: np.full(
            (len(BASE_SEEDS), len(selection_rows), len(CLASS_ORDER)),
            np.nan,
            dtype=np.float64,
        )
        for modality in MODALITIES
    }
    for modality in MODALITIES:
        train_x, selection_x = _prepare_modality_features(
            corpus, modality, fit_rows, selection_rows, spec
        )
        for seed_index, seed in enumerate(BASE_SEEDS):
            classifier = _new_projector(spec, seed)
            classifier.fit(train_x, fit_labels)
            probabilities[modality][seed_index] = _aligned_probability(
                classifier, selection_x
            )
    selection_fold_hash = _canonical_sha256(
        {
            "role": SELECTION_ROLE,
            "fit_role_only": True,
            "fit_oof_fold_assignment_sha256": fit_oof.fold_assignment_sha256,
            "selection_labels_used_for_fit_or_mapping": False,
            "base_seeds": BASE_SEEDS,
        }
    )
    return RolePosteriorGrid(
        role=SELECTION_ROLE,
        row_indices=selection_rows,
        probabilities=probabilities,
        base_seeds=BASE_SEEDS,
        fold_by_local_row=np.full(len(selection_rows), -1, dtype=np.int32),
        fold_assignment_sha256=selection_fold_hash,
        producer_config_sha256=fit_oof.producer_config_sha256,
    )


def fit_gate_train_predict_gate_eval_posteriors(
    gate_corpus: OpenRoleCorpus,
    spec: ProjectorSpec,
    *,
    gate_train_oof: RolePosteriorGrid,
) -> RolePosteriorGrid:
    """Predict fit-internal gate-eval from complete gate-train only.

    The in-memory ``SELECTION_ROLE`` tag is reused solely as a role separator;
    no physical model-selection feature or label sidecar is opened here.
    """

    return fit_full_fit_predict_selection_posteriors(
        gate_corpus,
        spec,
        fit_oof=gate_train_oof,
    )


def _sample_tasks_for_role(
    corpus: OpenRoleCorpus,
    role: str,
    sampling: UtilitySamplingConfig,
) -> tuple[BidirectionalCoalitionTask, ...]:
    """Sample one role while preserving original protocol-row RNG identities."""

    allowed = set(int(value) for value in corpus.role_indices(role))
    masked_histories = tuple(
        history if query in allowed else tuple()
        for query, history in enumerate(corpus.histories)
    )
    masked = replace(corpus, histories=masked_histories)
    tasks = tuple(sample_corpus_bidirectional_tasks(masked, sampling))
    if not tasks:
        raise EmotionRelationVADRepairError(f"{role} produced no bidirectional tasks")
    for task in tasks:
        rows = {
            int(task.query_index),
            int(task.candidate_index),
            *map(int, task.addition_context),
            *map(int, task.deletion_context),
        }
        if not rows.issubset(allowed):
            raise AssertionError("role-specific task sampling crossed an open-role boundary")
    return tasks


def _role_dataset_identifier(provenance: CorpusProvenance) -> str:
    return (
        f"{provenance.dataset_id}/{provenance.manifest_schema}"
        f"@{provenance.manifest_sha256}"
    )


def _train_only_provenance(
    corpus: OpenRoleCorpus,
    corpus_provenance: CorpusProvenance,
    posterior: RolePosteriorGrid,
    tasks: Sequence[BidirectionalCoalitionTask],
) -> TrainOnlyProvenance:
    dataset = _role_dataset_identifier(corpus_provenance)
    source_hash = ordered_source_sha256(dataset, tuple(str(value) for value in corpus.keys))
    class_hash = emotion_class_order_sha256(CLASS_ORDER)
    context_hash = emotion_context_schema_sha256()
    role = posterior.role
    mode = "train_fold_oof" if role == FIT_ROLE else "train_fit_only"
    task_hash = bidirectional_task_order_sha256(
        tasks,
        dataset=dataset,
        role=role,
        source_order_sha256=source_hash,
        split_manifest_sha256=corpus_provenance.manifest_sha256,
        fold_assignment_sha256=posterior.fold_assignment_sha256,
        context_schema_sha256=context_hash,
        class_order_sha256=class_hash,
        producer_config_sha256=posterior.producer_config_sha256,
    )
    return TrainOnlyProvenance(
        mode=mode,
        dataset=dataset,
        role=role,
        dataset_sha256=dataset_identity_sha256(dataset),
        source_order_sha256=source_hash,
        split_manifest_sha256=corpus_provenance.manifest_sha256,
        fold_assignment_sha256=posterior.fold_assignment_sha256,
        task_order_sha256=task_hash,
        context_schema_sha256=context_hash,
        class_order_sha256=class_hash,
        producer_config_sha256=posterior.producer_config_sha256,
    )


def _posterior_rows(
    posterior: RolePosteriorGrid,
    modality: str,
    corpus_rows: Sequence[int],
) -> np.ndarray:
    local = posterior.local_by_corpus_row
    try:
        positions = np.asarray([local[int(row)] for row in corpus_rows], dtype=np.int64)
    except KeyError as error:
        raise EmotionRelationVADRepairError("task references a row outside posterior role") from error
    return posterior.probabilities[modality][:, positions]


def _mean_context_posterior(
    posterior: RolePosteriorGrid,
    modality: str,
    contexts: Sequence[Sequence[int]],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for context in contexts:
        rows = tuple(int(value) for value in context)
        if not rows:
            raise EmotionRelationVADRepairError("Repair-3 task history context is empty")
        values.append(np.mean(_posterior_rows(posterior, modality, rows), axis=1))
    return np.stack(values, axis=1)


def _task_context_blocks(
    posterior: RolePosteriorGrid,
    tasks: Sequence[BidirectionalCoalitionTask],
    provenance: TrainOnlyProvenance,
) -> tuple[EmotionProbabilityBlock, dict[str, EmotionProbabilityBlock]]:
    queries = tuple(int(task.query_index) for task in tasks)
    candidates = tuple((int(task.candidate_index),) for task in tasks)
    additions = tuple(tuple(task.addition_context) for task in tasks)
    deletions = tuple(tuple(task.deletion_context) for task in tasks)
    minus = tuple(
        tuple(value for value in task.deletion_context if int(value) != int(task.candidate_index))
        for task in tasks
    )
    mean_probabilities: dict[str, dict[str, np.ndarray]] = {
        "current": {},
        "candidate": {},
        "s": {},
        "t": {},
        "t_minus_candidate": {},
    }
    for modality in MODALITIES:
        mean_probabilities["current"][modality] = np.mean(
            _posterior_rows(posterior, modality, queries), axis=0
        )
        mean_probabilities["candidate"][modality] = np.mean(
            _mean_context_posterior(posterior, modality, candidates), axis=0
        )
        mean_probabilities["s"][modality] = np.mean(
            _mean_context_posterior(posterior, modality, additions), axis=0
        )
        mean_probabilities["t"][modality] = np.mean(
            _mean_context_posterior(posterior, modality, deletions), axis=0
        )
        mean_probabilities["t_minus_candidate"][modality] = np.mean(
            _mean_context_posterior(posterior, modality, minus), axis=0
        )
    orders = {modality: CLASS_ORDER for modality in MODALITIES}

    def block(name: str) -> EmotionProbabilityBlock:
        return EmotionProbabilityBlock(
            probabilities=mean_probabilities[name],
            provenance=provenance,
            class_order=CLASS_ORDER,
            modality_class_orders=orders,
        )

    current = block("current")
    history = {context: block(context) for context in HISTORY_CONTEXTS}
    return current, history


def build_vad_state_transition_features(
    current: EmotionProbabilityBlock,
    history: Mapping[str, EmotionProbabilityBlock],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return 15 state and 12 current-minus-context VAD features."""

    anchors = np.asarray([VAD_COORDINATES[name] for name in CLASS_ORDER], dtype=np.float64)
    contexts: list[tuple[str, EmotionProbabilityBlock]] = [
        ("current", current),
        *((name, history[name]) for name in HISTORY_CONTEXTS),
    ]
    states: dict[str, np.ndarray] = {}
    values: list[np.ndarray] = []
    names: list[str] = []
    dimensions = ("valence", "arousal", "dominance")
    for context_name, block in contexts:
        mean_probability = np.mean(
            np.stack([block.probabilities[modality] for modality in MODALITIES]), axis=0
        )
        expected = mean_probability @ anchors
        states[context_name] = expected
        for column, dimension in enumerate(dimensions):
            values.append(expected[:, column])
            names.append(f"vad_state__{context_name}__{dimension}")
    for context_name in HISTORY_CONTEXTS:
        transition = states["current"] - states[context_name]
        for column, dimension in enumerate(dimensions):
            values.append(transition[:, column])
            names.append(f"vad_transition__current_minus_{context_name}__{dimension}")
    matrix = np.column_stack(values).astype(np.float64, copy=False)
    if matrix.shape != (current.rows, 27) or not np.isfinite(matrix).all():
        raise AssertionError("VAD state/transition geometry changed")
    if len(names) != len(set(names)) or any("label" in name or "gold" in name for name in names):
        raise AssertionError("VAD feature names are unsafe")
    matrix.setflags(write=False)
    return matrix, tuple(names)


def _fused_context_probability(
    posterior: RolePosteriorGrid,
    queries: Sequence[int],
    contexts: Sequence[Sequence[int]],
) -> np.ndarray:
    if len(queries) != len(contexts):
        raise EmotionRelationVADRepairError("query/context probability inputs are not aligned")
    current = np.mean(
        np.stack(
            [_posterior_rows(posterior, modality, queries) for modality in MODALITIES],
            axis=0,
        ),
        axis=0,
    )
    result = np.empty_like(current)
    for row, context in enumerate(contexts):
        members = tuple(int(value) for value in context)
        if not members:
            result[:, row] = current[:, row]
            continue
        history = np.mean(
            np.stack(
                [
                    np.mean(_posterior_rows(posterior, modality, members), axis=1)
                    for modality in MODALITIES
                ],
                axis=0,
            ),
            axis=0,
        )
        result[:, row] = (
            FUSION_CURRENT_WEIGHT * current[:, row]
            + (1.0 - FUSION_CURRENT_WEIGHT) * history
        )
    result = np.clip(result, 0.0, 1.0)
    result /= result.sum(axis=2, keepdims=True)
    return result


def _task_base_probability(
    posterior: RolePosteriorGrid,
    tasks: Sequence[BidirectionalCoalitionTask],
) -> np.ndarray:
    queries = tuple(int(task.query_index) for task in tasks)
    contexts_by_task = tuple(task_contexts(task) for task in tasks)
    result = np.empty(
        (len(BASE_SEEDS), len(tasks), 4, len(CLASS_ORDER)), dtype=np.float64
    )
    for context_index in range(4):
        contexts = tuple(values[context_index] for values in contexts_by_task)
        result[:, :, context_index] = _fused_context_probability(
            posterior, queries, contexts
        )
    return result


def _deterministic_npy(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def _deterministic_base_cache_payload(
    matrix: np.ndarray,
    role: str,
) -> bytes:
    field = "fit_x" if role == FIT_ROLE else "selection_x"
    arrays = {
        "schema_version.npy": np.asarray([BASE_CACHE_SCHEMA_VERSION]),
        f"{field}.npy": np.asarray(matrix, dtype=np.float64),
        "feature_names.npy": np.asarray(BASE_CACHE_FEATURE_NAMES, dtype=str),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _deterministic_npy(arrays[name]))
    return stream.getvalue()


def _verify_in_memory_base_cache(
    matrix: np.ndarray,
    provenance: TrainOnlyProvenance,
) -> object:
    payload = _deterministic_base_cache_payload(matrix, provenance.role)
    lineage = BaseCacheLineage(
        schema_version=BASE_CACHE_SCHEMA_VERSION,
        row_count=len(matrix),
        cache_sha256=hashlib.sha256(payload).hexdigest(),
        matrix_content_sha256=numeric_matrix_content_sha256(matrix),
        feature_names_content_sha256=feature_names_content_sha256(
            BASE_CACHE_FEATURE_NAMES
        ),
        provenance=provenance,
        class_order=CLASS_ORDER,
    )
    return verify_base_59d_cache(
        payload,
        lineage=lineage,
        expected_lineage_sha256=base_cache_lineage_sha256(lineage),
    )


def _cluster_codes(values: Sequence[object]) -> np.ndarray:
    _, inverse = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def build_role_feature_set(
    corpus: OpenRoleCorpus,
    corpus_provenance: CorpusProvenance,
    posterior: RolePosteriorGrid,
    tasks: Sequence[BidirectionalCoalitionTask],
    *,
    outcome_labels_allowed: bool = True,
) -> RoleFeatureSet:
    """Construct and lineage-verify the fixed primary plus four ablations."""

    task_tuple = tuple(tasks)
    provenance = _train_only_provenance(
        corpus, corpus_provenance, posterior, task_tuple
    )
    current, history = _task_context_blocks(posterior, task_tuple, provenance)
    relation = build_emotion_probability_relations(current, history)
    vad_x, vad_names = build_vad_state_transition_features(current, history)
    probability_by_seed = _task_base_probability(posterior, task_tuple)
    base_x, base_names = probability_task_features(
        np.mean(probability_by_seed, axis=0), task_tuple, corpus.histories
    )
    if tuple(base_names) != BASE_CACHE_FEATURE_NAMES or base_x.shape[1] != 59:
        raise EmotionRelationVADRepairError("recomputed base is not canonical 59-D")
    verified = _verify_in_memory_base_cache(base_x, provenance)

    joined_full = align_with_59d_task_cache(
        verified,
        emotion_features=relation,
        emotion_group="simple_concat_plus_full_9cell",
    )
    joined_same = align_with_59d_task_cache(
        verified,
        emotion_features=relation,
        emotion_group="simple_concat_plus_same_modality",
    )
    joined_concat = align_with_59d_task_cache(
        verified,
        emotion_features=relation,
        emotion_group="simple_concat",
    )
    matrices = {
        PRIMARY_VARIANT: np.column_stack([joined_full.matrix, vad_x]),
        CAPACITY_CONTROL_VARIANT: verified.matrix @ CAPACITY_CONTROL_EXPANSION,
        "same_modality_relations_vad": np.column_stack([joined_same.matrix, vad_x]),
        "full_relations_no_vad": joined_full.matrix,
        "concat_vad_no_3x3": np.column_stack([joined_concat.matrix, vad_x]),
        "base_59d_only": verified.matrix,
    }
    names = {
        PRIMARY_VARIANT: joined_full.feature_names + vad_names,
        CAPACITY_CONTROL_VARIANT: tuple(
            (
                f"capacity_control__column_{target:03d}__"
                f"source_{target % 59:02d}"
            )
            for target in range(299)
        ),
        "same_modality_relations_vad": joined_same.feature_names + vad_names,
        "full_relations_no_vad": joined_full.feature_names,
        "concat_vad_no_3x3": joined_concat.feature_names + vad_names,
        "base_59d_only": verified.feature_names,
    }
    query_rows = np.asarray(
        [int(task.query_index) for task in task_tuple], dtype=np.int64
    )
    labels = (
        np.asarray(corpus.labels[query_rows], dtype=np.int64)
        if outcome_labels_allowed
        else np.zeros(len(query_rows), dtype=np.int64)
    )
    clusters = _cluster_codes(corpus.groups[query_rows])
    mean_probability = np.mean(probability_by_seed, axis=0)
    targets = (
        bidirectional_utility_targets(
            labels,
            mean_probability[:, 0],
            mean_probability[:, 1],
            mean_probability[:, 2],
            mean_probability[:, 3],
        )
        if outcome_labels_allowed
        else SimpleNamespace(
            forward_addition=np.zeros(len(task_tuple), dtype=np.float64),
            backward_deletion=np.zeros(len(task_tuple), dtype=np.float64),
        )
    )
    splits: dict[str, UtilitySplit] = {}
    for variant in VARIANT_ORDER:
        matrix = np.asarray(matrices[variant], dtype=np.float64)
        if matrix.shape != (len(task_tuple), VARIANT_WIDTHS[variant]):
            raise AssertionError(f"{variant} feature width changed")
        if len(names[variant]) != matrix.shape[1] or len(set(names[variant])) != len(
            names[variant]
        ):
            raise AssertionError(f"{variant} feature names changed")
        if not np.isfinite(matrix).all() or any(
            token in feature.lower()
            for feature in names[variant]
            for token in ("gold", "label", "target")
        ):
            raise EmotionRelationVADRepairError("inference feature is non-finite or supervised")
        splits[variant] = UtilitySplit.validated(
            matrix,
            targets.forward_addition,
            targets.backward_deletion,
            clusters,
            label=f"Repair-3 {posterior.role} {variant}",
        )
    return RoleFeatureSet(
        role=posterior.role,
        tasks=task_tuple,
        task_labels=labels,
        cluster_codes=clusters,
        base_probability_by_seed=probability_by_seed,
        variants=MappingProxyType(splits),
        feature_names=MappingProxyType({key: tuple(value) for key, value in names.items()}),
        provenance=provenance,
        base_cache_lineage_sha256=verified.lineage_sha256,
        vad_coordinate_sha256=VAD_COORDINATE_SHA256,
        outcome_labels_used_for_feature_construction=bool(outcome_labels_allowed),
    )


def _mean_metric_records(
    records: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    if len(records) != len(BASE_SEEDS):
        raise EmotionRelationVADRepairError("metric aggregate requires five base seeds")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records):
        raise EmotionRelationVADRepairError("base-seed metric schemas differ")
    result: dict[str, float | int] = {}
    for key in keys:
        values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
        result[key] = int(round(float(np.mean(values)))) if key == "queries" else float(
            np.mean(values)
        )
    return result


def _aggregate_utility_seed_records(
    records: Sequence[Mapping[str, float | int]],
) -> dict[str, dict[str, float]]:
    if len(records) != len(UTILITY_SEEDS):
        raise EmotionRelationVADRepairError("utility aggregate requires five seeds")
    keys = tuple(key for key in records[0] if key != "utility_seed")
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
        }
    return result


def _metric_difference(
    candidate: Mapping[str, float | int],
    reference: Mapping[str, float | int],
) -> dict[str, float]:
    keys = (
        "pooled_macro_f1",
        "pooled_accuracy",
        "pooled_nll",
        "pooled_brier",
        "mean_excess_nll_vs_current",
        "harm_rate_vs_current",
        "actual_history_coverage",
    )
    return {
        f"{key}_difference": float(candidate[key]) - float(reference[key])
        for key in keys
    }


def _strategy_records(
    corpus: OpenRoleCorpus,
    posterior: RolePosteriorGrid,
    query_rows: Sequence[int],
    contexts: Sequence[Sequence[int]],
    *,
    current_probability_by_seed: np.ndarray,
    ece_bins: int,
) -> tuple[dict[str, float | int], ...]:
    queries = tuple(int(value) for value in query_rows)
    # This is the only helper that indexes outcome labels for query scoring.
    labels = np.asarray(corpus.labels[np.asarray(queries, dtype=np.int64)], dtype=np.int64)
    clusters = _cluster_codes(corpus.groups[np.asarray(queries, dtype=np.int64)])
    probability_by_seed = _fused_context_probability(posterior, queries, contexts)
    records = tuple(
        query_strategy_metrics(
            labels,
            probability_by_seed[seed_index],
            current_probability_by_seed[seed_index],
            contexts,
            corpus.histories,
            queries,
            clusters,
            ece_bins=int(ece_bins),
        )
        for seed_index in range(len(BASE_SEEDS))
    )
    return records


def _numeric_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\x00" + array.tobytes(order="C")).hexdigest()


def _fitted_capacity_model_sha256(model: object) -> str:
    estimator = model.estimator
    state = {
        "spec": model.spec.public_dict(),
        "seed": int(model.seed),
        "x_scaler_mean": _numeric_array_sha256(model.x_scaler.mean_),
        "x_scaler_scale": _numeric_array_sha256(model.x_scaler.scale_),
        "target_mean": _numeric_array_sha256(model.target_mean),
        "target_scale": _numeric_array_sha256(model.target_scale),
        "coefs": [_numeric_array_sha256(value) for value in estimator.coefs_],
        "intercepts": [
            _numeric_array_sha256(value) for value in estimator.intercepts_
        ],
        "parameter_count": int(model.parameter_count),
    }
    return _canonical_sha256(state)


def _fit_gate_variant_seed_scores(
    train_features: RoleFeatureSet,
    eval_features: RoleFeatureSet,
    variant: str,
    model_spec: CapacityMatchedUtilitySpec,
    balance: ClassBalanceSpec,
) -> tuple[GateVariantSeedScore, ...]:
    """Fit only on gate-train; gate-eval contributes feature rows, never labels."""

    if train_features.role != FIT_ROLE or eval_features.role != SELECTION_ROLE:
        raise EmotionRelationVADRepairError("fit-internal gate feature roles changed")
    if not train_features.outcome_labels_used_for_feature_construction:
        raise EmotionRelationVADRepairError("gate-train utility targets are missing")
    if eval_features.outcome_labels_used_for_feature_construction:
        raise EmotionRelationVADRepairError(
            "gate-eval labels entered feature or utility-target construction"
        )
    if train_features.feature_names[variant] != eval_features.feature_names[variant]:
        raise EmotionRelationVADRepairError("gate train/eval feature schemas differ")
    train_split = train_features.variants[variant]
    eval_x = np.asarray(eval_features.variants[variant].x, dtype=np.float64)
    expected_parameters = trainable_parameter_count(
        train_split.x.shape[1], model_spec.hidden_layer_sizes, 2
    )
    frozen_parameters = (
        BASE_59D_PARAMETER_COUNT
        if variant == "base_59d_only"
        else PRIMARY_OR_CONTROL_PARAMETER_COUNT
    )
    if expected_parameters != frozen_parameters:
        raise EmotionRelationVADRepairError(
            f"{variant} utility capacity changed: {expected_parameters}"
        )
    states: list[GateVariantSeedScore] = []
    for seed in UTILITY_SEEDS:
        oof = group_oof_class_balanced_predictions(
            train_split,
            train_features.tasks,
            train_features.task_labels,
            model_spec,
            balance,
            seed=int(seed),
            maximum_splits=5,
        )
        threshold, pair_count, realized_coverage = (
            fit_query_candidate_coverage_threshold(
                train_features.tasks,
                oof.predictions.decision_score,
                target_coverage=0.25,
            )
        )
        fitted = fit_class_balanced_utility_model(
            train_split,
            train_features.tasks,
            train_features.task_labels,
            model_spec,
            balance,
            seed=int(seed),
        )
        if int(fitted.parameter_count) != expected_parameters:
            raise EmotionRelationVADRepairError("fitted utility capacity changed")
        prediction = fitted.predict(eval_x)
        training_artifact_sha = _canonical_sha256(
            {
                "variant": variant,
                "utility_seed": int(seed),
                "train_task_order_sha256": train_features.provenance.task_order_sha256,
                "train_base_cache_lineage_sha256": train_features.base_cache_lineage_sha256,
                "train_feature_matrix_sha256": numeric_matrix_content_sha256(
                    train_split.x
                ),
                "train_forward_target_sha256": _numeric_array_sha256(
                    train_split.forward
                ),
                "train_backward_target_sha256": _numeric_array_sha256(
                    train_split.backward
                ),
                "train_cluster_sha256": _numeric_array_sha256(
                    train_split.cluster_codes
                ),
                "train_oof_fold_sha256": _numeric_array_sha256(oof.fold_by_row),
                "model_state_sha256": _fitted_capacity_model_sha256(fitted),
            }
        )
        threshold_commitment = _canonical_sha256(
            {
                "training_artifact_sha256": training_artifact_sha,
                "target_query_candidate_coverage": 0.25,
                "threshold": float(threshold),
                "fit_query_candidate_pairs": int(pair_count),
                "realized_fit_query_candidate_coverage": float(realized_coverage),
            }
        )
        states.append(
            GateVariantSeedScore(
                seed=int(seed),
                threshold=float(threshold),
                gate_eval_scores=np.asarray(
                    prediction.decision_score, dtype=np.float64
                ),
                train_oof_fold_by_row=np.asarray(
                    oof.fold_by_row, dtype=np.int32
                ),
                train_query_candidate_pairs=int(pair_count),
                realized_train_query_candidate_coverage=float(realized_coverage),
                parameter_count=int(fitted.parameter_count),
                training_artifact_sha256=training_artifact_sha,
                threshold_commitment_sha256=threshold_commitment,
            )
        )
    return tuple(states)


def fit_only_gate_decision(
    macro_f1_gain_vs_base: Sequence[float],
    nll_worsening_vs_base: Sequence[float],
    accuracy_gain_vs_base: Sequence[float],
    macro_f1_gain_vs_capacity_control: Sequence[float],
    nll_worsening_vs_capacity_control: Sequence[float],
    accuracy_gain_vs_capacity_control: Sequence[float],
) -> tuple[bool, tuple[bool, ...]]:
    """Frozen same-seed intersection gate against both causal references."""

    arrays = tuple(
        np.asarray(tuple(values), dtype=np.float64)
        for values in (
            macro_f1_gain_vs_base,
            nll_worsening_vs_base,
            accuracy_gain_vs_base,
            macro_f1_gain_vs_capacity_control,
            nll_worsening_vs_capacity_control,
            accuracy_gain_vs_capacity_control,
        )
    )
    if any(values.shape != (5,) for values in arrays):
        raise EmotionRelationVADRepairError(
            "fit-only gate requires six aligned five-seed comparisons"
        )
    if any(not np.isfinite(values).all() for values in arrays):
        raise EmotionRelationVADRepairError("fit-only gate inputs must be finite")
    per_seed = tuple(
        bool(
            arrays[0][index] >= FIT_GATE_MACRO_F1_GAIN
            and arrays[1][index] <= NLL_IDENTITY_TOLERANCE
            and arrays[2][index] >= 0.0
            and arrays[3][index] >= FIT_GATE_MACRO_F1_GAIN
            and arrays[4][index] <= NLL_IDENTITY_TOLERANCE
            and arrays[5][index] >= 0.0
        )
        for index in range(len(UTILITY_SEEDS))
    )
    return sum(per_seed) >= FIT_GATE_REQUIRED_SEEDS, per_seed


def evaluate_fit_only_open_gate(
    corpus: OpenRoleCorpus,
    train_posterior: RolePosteriorGrid,
    eval_posterior: RolePosteriorGrid,
    train_features: RoleFeatureSet,
    eval_features: RoleFeatureSet,
    split: FitInternalGateSplit,
    balance: ClassBalanceSpec,
    utility_specs: Sequence[CapacityMatchedUtilitySpec],
    *,
    ece_bins: int,
) -> tuple[bool, dict[str, object]]:
    """Fit on gate-train and score frozen predictions once on gate-eval."""

    if train_posterior.role != FIT_ROLE or eval_posterior.role != SELECTION_ROLE:
        raise EmotionRelationVADRepairError("fit-internal gate posterior roles changed")
    if not split.eligible:
        raise EmotionRelationVADRepairError("fit_internal_gate_no_go: ineligible split")
    by_name = {spec.name: spec for spec in utility_specs}
    true_name = "class_balanced_true_bidirectional_mlp"
    if set(by_name) != set(MODEL_NAMES):
        raise EmotionRelationVADRepairError("Repair-2 utility model set changed")
    model_spec = by_name[true_name]

    # All three variants are fully trained and all gate-eval predictions and
    # contexts are frozen before the first gate-eval label is indexed below.
    states = {
        variant: _fit_gate_variant_seed_scores(
            train_features,
            eval_features,
            variant,
            model_spec,
            balance,
        )
        for variant in (
            PRIMARY_VARIANT,
            "base_59d_only",
            CAPACITY_CONTROL_VARIANT,
        )
    }
    query_rows = tuple(int(value) for value in eval_posterior.row_indices)
    empty = tuple(tuple() for _ in query_rows)
    all_history = tuple(tuple(corpus.histories[query]) for query in query_rows)
    current_probability = _fused_context_probability(
        eval_posterior, query_rows, empty
    )

    contexts_by_variant: dict[str, dict[int, tuple[tuple[int, ...], ...]]] = {}
    prediction_context_commitments: dict[str, dict[str, str]] = {}
    for variant, variant_states in states.items():
        contexts_by_variant[variant] = {}
        prediction_context_commitments[variant] = {}
        for state in variant_states:
            nested = aggregate_candidate_draw_scores(
                eval_features.tasks, state.gate_eval_scores
            )
            contexts = build_reversible_selected_contexts(
                query_rows,
                corpus.histories,
                nested,
                threshold=state.threshold,
            )
            contexts_by_variant[variant][int(state.seed)] = contexts
            prediction_context_commitments[variant][str(state.seed)] = (
                _canonical_sha256(
                    {
                        "gate_eval_score_sha256": _numeric_array_sha256(
                            state.gate_eval_scores
                        ),
                        "threshold_commitment_sha256": (
                            state.threshold_commitment_sha256
                        ),
                        "selected_contexts": contexts,
                    }
                )
            )

    # Outcome-label access starts here, after every prediction/context commit.
    def mean_for(candidate_contexts: Sequence[Sequence[int]]) -> dict[str, float | int]:
        return _mean_metric_records(
            _strategy_records(
                corpus,
                eval_posterior,
                query_rows,
                candidate_contexts,
                current_probability_by_seed=current_probability,
                ece_bins=ece_bins,
            )
        )

    current_mean = mean_for(empty)
    all_mean = mean_for(all_history)
    per_seed: list[dict[str, object]] = []
    comparisons: dict[str, list[float]] = {
        "macro_base": [],
        "nll_base": [],
        "accuracy_base": [],
        "macro_control": [],
        "nll_control": [],
        "accuracy_control": [],
    }
    for seed in UTILITY_SEEDS:
        primary = mean_for(contexts_by_variant[PRIMARY_VARIANT][int(seed)])
        base = mean_for(contexts_by_variant["base_59d_only"][int(seed)])
        control = mean_for(contexts_by_variant[CAPACITY_CONTROL_VARIANT][int(seed)])
        comparisons["macro_base"].append(
            float(primary["pooled_macro_f1"]) - float(base["pooled_macro_f1"])
        )
        comparisons["nll_base"].append(
            float(primary["pooled_nll"]) - float(base["pooled_nll"])
        )
        comparisons["accuracy_base"].append(
            float(primary["pooled_accuracy"]) - float(base["pooled_accuracy"])
        )
        comparisons["macro_control"].append(
            float(primary["pooled_macro_f1"])
            - float(control["pooled_macro_f1"])
        )
        comparisons["nll_control"].append(
            float(primary["pooled_nll"]) - float(control["pooled_nll"])
        )
        comparisons["accuracy_control"].append(
            float(primary["pooled_accuracy"]) - float(control["pooled_accuracy"])
        )
        per_seed.append(
            {
                "utility_seed": int(seed),
                "primary_five_base_seed_mean": primary,
                "base_59d_five_base_seed_mean": base,
                "rank59_capacity_control_five_base_seed_mean": control,
                "primary_minus_base_59d": _metric_difference(primary, base),
                "primary_minus_rank59_capacity_control": _metric_difference(
                    primary, control
                ),
                "primary_minus_current": _metric_difference(primary, current_mean),
                "fit_only_seed_gate_pass": False,
            }
        )
    passed, seed_decisions = fit_only_gate_decision(
        comparisons["macro_base"],
        comparisons["nll_base"],
        comparisons["accuracy_base"],
        comparisons["macro_control"],
        comparisons["nll_control"],
        comparisons["accuracy_control"],
    )
    for record, seed_passed in zip(per_seed, seed_decisions, strict=True):
        record["fit_only_seed_gate_pass"] = seed_passed
    training_commitments = {
        variant: {
            str(state.seed): {
                "training_artifact_sha256": state.training_artifact_sha256,
                "threshold_commitment_sha256": state.threshold_commitment_sha256,
                "parameter_count": state.parameter_count,
            }
            for state in variant_states
        }
        for variant, variant_states in states.items()
    }
    report: dict[str, object] = {
        "role": "physical_fit_internal_gate_eval",
        "selection_prediction_generated": False,
        "selection_label_scored": False,
        "candidate_variant": PRIMARY_VARIANT,
        "reference_variants": ["base_59d_only", CAPACITY_CONTROL_VARIANT],
        "candidate_and_references_use_same_repair_2_policy": True,
        "definition": {
            "minimum_macro_f1_gain_vs_each_reference": FIT_GATE_MACRO_F1_GAIN,
            "nll_maximum_worsening_vs_each_reference": 0.0,
            "accuracy_minimum_gain_vs_each_reference": 0.0,
            "nll_comparison_tolerance": NLL_IDENTITY_TOLERANCE,
            "same_seed_six_condition_intersection": True,
            "minimum_successful_utility_seeds_out_of_five": FIT_GATE_REQUIRED_SEEDS,
        },
        "fixed_references": {
            "current_only_five_base_seed_mean": current_mean,
            "all_history_five_base_seed_mean": all_mean,
        },
        "per_utility_seed": per_seed,
        "successful_utility_seeds_out_of_five": int(sum(seed_decisions)),
        "passed": bool(passed),
        "aggregate_attestation": {
            "split": split.aggregate_attestation(),
            "task_commitments": {
                "gate_train": train_features.provenance.task_order_sha256,
                "gate_eval": eval_features.provenance.task_order_sha256,
            },
            "fold_commitments": {
                "gate_train_base_oof": train_posterior.fold_assignment_sha256,
                "gate_eval_base_full_gate_train_prediction": (
                    eval_posterior.fold_assignment_sha256
                ),
            },
            "model_and_threshold_commitments": training_commitments,
            "gate_eval_prediction_context_commitments": (
                prediction_context_commitments
            ),
            "gate_eval_labels_used_for_projector_fit": False,
            "gate_eval_labels_used_for_feature_or_utility_fit": False,
            "gate_eval_labels_used_for_threshold": False,
            "gate_eval_labels_used_only_for_scoring_after_prediction_context_freeze": True,
            "selection_feature_payload_opened": False,
            "selection_label_payload_opened": False,
            "row_or_group_identifiers_published": False,
        },
    }
    return passed, report


def _selection_variant_scores(
    fit_features: RoleFeatureSet,
    selection_features: RoleFeatureSet,
    variant: str,
    balance: ClassBalanceSpec,
    utility_specs: Sequence[CapacityMatchedUtilitySpec],
) -> dict[str, tuple[BalancedUtilitySeedScores, ...]]:
    if fit_features.feature_names[variant] != selection_features.feature_names[variant]:
        raise EmotionRelationVADRepairError("fit/selection feature schemas differ")
    cache = BidirectionalUtilityCache(
        fit=fit_features.variants[variant],
        selection=selection_features.variants[variant],
        feature_names=fit_features.feature_names[variant],
        source_hashes={
            "fit_task_order_sha256": fit_features.provenance.task_order_sha256,
            "selection_task_order_sha256": selection_features.provenance.task_order_sha256,
            "fit_base_cache_lineage_sha256": fit_features.base_cache_lineage_sha256,
            "selection_base_cache_lineage_sha256": selection_features.base_cache_lineage_sha256,
        },
    )
    # This frozen Repair-2 API is intentionally called only after the fit gate.
    return fit_class_balanced_seed_scores(
        cache,
        fit_features.tasks,
        fit_features.task_labels,
        balance,
        utility_specs,
        maximum_splits=5,
    )


def evaluate_model_selection(
    corpus: OpenRoleCorpus,
    selection_posterior: RolePosteriorGrid,
    fit_features: RoleFeatureSet,
    selection_features: RoleFeatureSet,
    balance: ClassBalanceSpec,
    utility_specs: Sequence[CapacityMatchedUtilitySpec],
    *,
    ece_bins: int,
) -> dict[str, object]:
    """Score the fixed primary and explanatory ablations; never choose among them."""

    if selection_posterior.role != SELECTION_ROLE:
        raise EmotionRelationVADRepairError("model-selection evaluator received wrong role")
    query_rows = tuple(int(value) for value in selection_posterior.row_indices)
    empty = tuple(tuple() for _ in query_rows)
    all_history = tuple(tuple(corpus.histories[query]) for query in query_rows)
    current_probability = _fused_context_probability(
        selection_posterior, query_rows, empty
    )
    current_mean = _mean_metric_records(
        _strategy_records(
            corpus,
            selection_posterior,
            query_rows,
            empty,
            current_probability_by_seed=current_probability,
            ece_bins=ece_bins,
        )
    )
    all_mean = _mean_metric_records(
        _strategy_records(
            corpus,
            selection_posterior,
            query_rows,
            all_history,
            current_probability_by_seed=current_probability,
            ece_bins=ece_bins,
        )
    )

    variant_reports: dict[str, object] = {}
    primary_contexts: dict[int, tuple[tuple[int, ...], ...]] = {}
    primary_means: dict[int, dict[str, float | int]] = {}
    backward_means: dict[int, dict[str, float | int]] = {}
    max_nll_identity_error = 0.0
    for variant in VARIANT_ORDER:
        scores = _selection_variant_scores(
            fit_features,
            selection_features,
            variant,
            balance,
            utility_specs,
        )
        models: dict[str, object] = {}
        for model_name in MODEL_NAMES:
            per_seed: list[dict[str, float | int]] = []
            for state in scores[model_name]:
                nested = aggregate_candidate_draw_scores(
                    selection_features.tasks, state.selection_scores
                )
                contexts = build_reversible_selected_contexts(
                    query_rows,
                    corpus.histories,
                    nested,
                    threshold=state.threshold,
                )
                mean_record = _mean_metric_records(
                    _strategy_records(
                        corpus,
                        selection_posterior,
                        query_rows,
                        contexts,
                        current_probability_by_seed=current_probability,
                        ece_bins=ece_bins,
                    )
                )
                identity_error = abs(
                    (
                        float(mean_record["pooled_nll"])
                        - float(current_mean["pooled_nll"])
                    )
                    - float(mean_record["mean_excess_nll_vs_current"])
                )
                max_nll_identity_error = max(max_nll_identity_error, identity_error)
                record = {
                    "utility_seed": int(state.seed),
                    **mean_record,
                    "fit_oof_threshold": float(state.threshold),
                    "fit_query_candidate_pairs": int(
                        state.fit_query_candidate_pairs
                    ),
                    "realized_fit_query_candidate_coverage": float(
                        state.realized_fit_query_candidate_coverage
                    ),
                    "trainable_parameters": int(state.parameter_count),
                }
                per_seed.append(record)
                if variant == PRIMARY_VARIANT and model_name == (
                    "class_balanced_true_bidirectional_mlp"
                ):
                    primary_contexts[int(state.seed)] = contexts
                    primary_means[int(state.seed)] = mean_record
                if variant == PRIMARY_VARIANT and model_name == "class_balanced_backward_mlp":
                    backward_means[int(state.seed)] = mean_record
            models[model_name] = {
                "per_utility_seed": per_seed,
                "aggregate_across_utility_seeds": _aggregate_utility_seed_records(
                    per_seed
                ),
            }
        variant_reports[variant] = {
            "role": "fixed_primary" if variant == PRIMARY_VARIANT else "explanatory_ablation",
            "feature_width": VARIANT_WIDTHS[variant],
            "models": models,
        }
    if max_nll_identity_error > NLL_IDENTITY_TOLERANCE:
        raise AssertionError("pooled NLL and mean excess NLL identities disagree")

    primary_diagnostics: list[dict[str, object]] = []
    registered_success = 0
    stronger_success = 0
    for utility_seed in UTILITY_SEEDS:
        primary = primary_means[int(utility_seed)]
        backward = backward_means[int(utility_seed)]
        recency_contexts = coverage_matched_recency_contexts(
            query_rows, corpus.histories, primary_contexts[int(utility_seed)]
        )
        recency = _mean_metric_records(
            _strategy_records(
                corpus,
                selection_posterior,
                query_rows,
                recency_contexts,
                current_probability_by_seed=current_probability,
                ece_bins=ece_bins,
            )
        )
        registered = (
            float(primary["pooled_macro_f1"])
            - float(current_mean["pooled_macro_f1"])
            >= FIT_GATE_MACRO_F1_GAIN
            and float(primary["mean_excess_nll_vs_current"]) < 0.0
            and float(primary["pooled_accuracy"])
            >= float(current_mean["pooled_accuracy"])
            and float(primary["actual_history_coverage"]) >= 0.10
        )
        stronger = registered and all(
            float(primary["pooled_macro_f1"]) > float(reference["pooled_macro_f1"])
            and float(primary["pooled_nll"])
            <= float(reference["pooled_nll"]) + NLL_IDENTITY_TOLERANCE
            and float(primary["pooled_accuracy"])
            >= float(reference["pooled_accuracy"])
            for reference in (all_mean, recency, backward)
        )
        registered_success += int(registered)
        stronger_success += int(stronger)
        primary_diagnostics.append(
            {
                "utility_seed": int(utility_seed),
                "primary_five_base_seed_mean": primary,
                "coverage_matched_recency_five_base_seed_mean": recency,
                "backward_five_base_seed_mean": backward,
                "primary_minus_current": _metric_difference(primary, current_mean),
                "primary_minus_all_history": _metric_difference(primary, all_mean),
                "primary_minus_recency": _metric_difference(primary, recency),
                "primary_minus_backward": _metric_difference(primary, backward),
                "registered_current_gate_pass": bool(registered),
                "stronger_all_reference_gate_pass": bool(stronger),
            }
        )
    return {
        "selection_prediction_generated": True,
        "selection_label_scored": True,
        "variant_choice_from_selection_results": False,
        "fixed_primary_variant": PRIMARY_VARIANT,
        "fixed_references": {
            "current_only_five_base_seed_mean": current_mean,
            "all_history_five_base_seed_mean": all_mean,
        },
        "variants": variant_reports,
        "primary_reference_diagnostics": primary_diagnostics,
        "registered_current_gate": {
            "definition": {
                "minimum_macro_f1_gain_vs_current": FIT_GATE_MACRO_F1_GAIN,
                "mean_excess_nll_vs_current_strictly_below_zero": True,
                "accuracy_gain_vs_current_minimum": 0.0,
                "minimum_actual_history_coverage": 0.10,
                "minimum_successful_utility_seeds_out_of_five": 4,
            },
            "successful_utility_seeds_out_of_five": int(registered_success),
            "passed": bool(registered_success >= 4),
        },
        "stronger_all_reference_gate": {
            "references": [
                "current_only",
                "all_history",
                "coverage_matched_recency",
                "class_balanced_backward_mlp",
            ],
            "minimum_successful_utility_seeds_out_of_five": 4,
            "per_reference_accuracy_gain_minimum": 0.0,
            "successful_utility_seeds_out_of_five": int(stronger_success),
            "passed": bool(stronger_success >= 4),
        },
        "numerical_audit": {
            "nll_identity_tolerance": NLL_IDENTITY_TOLERANCE,
            "maximum_nll_identity_absolute_error": float(max_nll_identity_error),
            "passed": True,
        },
    }


def _provenance_report(provenance: CorpusProvenance) -> dict[str, object]:
    staged = isinstance(provenance, StagedVerifiedCorpusProvenance)
    return {
        "dataset_id": provenance.dataset_id,
        "manifest_schema": provenance.manifest_schema,
        "manifest_status": provenance.manifest_status,
        "manifest_sha256": provenance.manifest_sha256,
        "source_hashes": dict(sorted(provenance.source_hashes.items())),
        "label_order": list(provenance.label_order),
        "role_rows": dict(sorted(provenance.role_rows.items())),
        "audio_dim": provenance.audio_dim,
        "video_dim": provenance.video_dim,
        "role_assignment_sha256": provenance.role_assignment_sha256,
        "speaker_mapping_sha256": provenance.speaker_mapping_sha256,
        "corpus_contract_sha256": (
            None if staged else provenance.corpus_contract_sha256
        ),
        "fit_stage_contract_sha256": (
            provenance.fit_stage_contract_sha256 if staged else None
        ),
        "verification_origin": provenance.verification_origin,
        "verifier_attestation_sha256": provenance.verifier_attestation_sha256,
        "strict_role_feature_sidecars": provenance.strict_role_feature_sidecars,
        "strict_role_label_sidecars": provenance.strict_role_label_sidecars,
        "selection_feature_hash_verified": (
            provenance.selection_feature_hash_verified if staged else True
        ),
        "selection_feature_payload_opened": (
            provenance.selection_feature_payload_opened if staged else True
        ),
        "selection_feature_deserialized": (
            provenance.selection_feature_deserialized if staged else True
        ),
        "selection_label_hash_verified": (
            provenance.selection_label_hash_verified if staged else True
        ),
        "selection_label_payload_opened": (
            provenance.selection_label_payload_opened if staged else True
        ),
        "selection_label_deserialized": (
            provenance.selection_label_deserialized if staged else True
        ),
        "sealed_role_arrays_opened": provenance.sealed_role_arrays_opened,
        "validation_or_test_opened": provenance.validation_or_test_opened,
    }


def _environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "gpu_used": False,
    }


def _assert_registered_input_locks(
    manifest_path: Path,
    model_config_path: Path,
    frozen: FrozenRepairConfig,
) -> None:
    """Fail before any sidecar deserialisation or model training on drift."""

    locks = _mapping(
        frozen.raw.get("registered_input_locks"), name="registered_input_locks"
    )
    observed = {
        "sidecar_manifest_sha256": sha256_file(Path(manifest_path)),
        "model_config_sha256": sha256_file(Path(model_config_path)),
    }
    expected = {
        "sidecar_manifest_sha256": REGISTERED_SIDECAR_MANIFEST_SHA256,
        "model_config_sha256": REGISTERED_MODEL_CONFIG_SHA256,
    }
    if {name: locks.get(name) for name in expected} != expected:
        raise EmotionRelationVADRepairError("registered input lock values changed")
    if observed != expected:
        raise EmotionRelationVADRepairError(
            "registered sidecar manifest or model config SHA-256 drifted"
        )


def _assert_files_match_snapshot(
    snapshot: Mapping[str, tuple[Path, str]],
    *,
    stage: str,
) -> None:
    for name, (path, expected_sha256) in snapshot.items():
        if sha256_file(path) != expected_sha256:
            raise EmotionRelationVADRepairError(
                f"{name} changed after its bytes were registered at {stage}"
            )


def _optional_teacher_report(raw: Mapping[str, object]) -> dict[str, object]:
    """Rebuild the public optional-teacher summary from a strict whitelist."""

    teacher = _mapping(raw.get("optional_teacher"), name="optional_teacher")
    eligibility = _mapping(
        teacher.get("fit_oof_eligibility"),
        name="optional_teacher.fit_oof_eligibility",
    )
    return {
        "status": str(teacher.get("status", "")),
        "model_revision": str(teacher.get("model_revision", "")),
        "role": str(teacher.get("role", "")),
        "fit_oof_eligibility": {
            "minimum_successful_seeds_out_of_five": int(
                eligibility.get("minimum_successful_seeds_out_of_five", -1)
            ),
            "minimum_macro_f1_gain": float(
                eligibility.get("minimum_macro_f1_gain", math.nan)
            ),
            "nll_maximum_worsening": float(
                eligibility.get("nll_maximum_worsening", math.nan)
            ),
            "accuracy_minimum_gain": float(
                eligibility.get("accuracy_minimum_gain", math.nan)
            ),
            "same_seed_macro_nll_accuracy_intersection": bool(
                eligibility.get(
                    "same_seed_macro_nll_accuracy_intersection", False
                )
            ),
        },
        "selection_can_choose_teacher": bool(
            teacher.get("selection_can_choose_teacher", False)
        ),
        "selection_can_choose_layer": bool(
            teacher.get("selection_can_choose_layer", False)
        ),
        "selection_can_choose_class_mapping": bool(
            teacher.get("selection_can_choose_class_mapping", False)
        ),
        "johnson_chinese_model": str(teacher.get("johnson_chinese_model", "")),
    }


def run_emotion_relation_vad_repair(
    sidecar_dir: Path,
    manifest_path: Path,
    model_config_path: Path,
    repair_config_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run Repair 3 from verified open-role sidecars and write aggregate JSON."""

    sidecar_dir = Path(sidecar_dir)
    manifest_path = Path(manifest_path)
    model_config_path = Path(model_config_path)
    repair_config_path = Path(repair_config_path)
    output_path = Path(output_path)
    if output_path.resolve() != REGISTERED_OUTPUT_PATH:
        raise EmotionRelationVADRepairError(
            "Repair-3 output path differs from the frozen one-shot repository path"
        )
    if output_path.exists():
        raise FileExistsError(f"Repair-3 output already exists: {output_path}")
    if not sidecar_dir.is_dir():
        raise FileNotFoundError(sidecar_dir)
    for path in (manifest_path, model_config_path, repair_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_snapshot = {
        "sidecar_manifest": (manifest_path, sha256_file(manifest_path)),
        "model_config": (model_config_path, sha256_file(model_config_path)),
        "repair_config": (repair_config_path, sha256_file(repair_config_path)),
    }
    frozen = load_emotion_relation_vad_repair_config(repair_config_path)
    _assert_files_match_snapshot(input_snapshot, stage="repair-config load")
    _assert_registered_input_locks(manifest_path, model_config_path, frozen)
    model_payload = _load_json(model_config_path, name="causal backbone config")
    _assert_files_match_snapshot(input_snapshot, stage="model-config load")
    model_config = CausalBackboneConfig.from_mapping(model_payload)
    staged_load = load_emotiontalk_fit_gate_stage(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        model_config=model_config,
    )
    corpus = staged_load.corpus
    staged_provenance = staged_load.provenance
    staged_provenance.validate(corpus, model_config)
    if tuple(staged_provenance.label_order) != CLASS_ORDER:
        raise EmotionRelationVADRepairError(
            "sidecar label order differs from the pre-registered VAD/class mapping"
        )
    _assert_files_match_snapshot(input_snapshot, stage="before first model fit")

    # Stage 1: split the physical fit role without labels.  The gate-train
    # partition alone fits every projector, utility model and threshold.  The
    # physical model-selection feature/label payloads remain unopened.
    internal_split = deterministic_fit_internal_gate_split(corpus)
    gate_corpus = build_fit_internal_gate_corpus(
        corpus,
        internal_split,
        model_config=model_config,
    )
    gate_train_posterior = fit_fit_role_group_oof_posteriors(
        gate_corpus, frozen.projector
    )
    gate_eval_posterior = fit_gate_train_predict_gate_eval_posteriors(
        gate_corpus,
        frozen.projector,
        gate_train_oof=gate_train_posterior,
    )
    gate_train_tasks = _sample_tasks_for_role(
        gate_corpus, FIT_ROLE, frozen.sampling
    )
    gate_eval_tasks = _sample_tasks_for_role(
        gate_corpus, SELECTION_ROLE, frozen.sampling
    )
    gate_train_features = build_role_feature_set(
        gate_corpus,
        staged_provenance,
        gate_train_posterior,
        gate_train_tasks,
        outcome_labels_allowed=True,
    )
    gate_eval_features = build_role_feature_set(
        gate_corpus,
        staged_provenance,
        gate_eval_posterior,
        gate_eval_tasks,
        outcome_labels_allowed=False,
    )
    ece_bins = int(frozen.raw.get("ece_bins", 15))
    fit_gate_passed, fit_gate = evaluate_fit_only_open_gate(
        gate_corpus,
        gate_train_posterior,
        gate_eval_posterior,
        gate_train_features,
        gate_eval_features,
        internal_split,
        frozen.balance,
        frozen.utility_specs,
        ece_bins=ece_bins,
    )

    selection_report: dict[str, object]
    selection_features: RoleFeatureSet | None = None
    fit_posterior = gate_train_posterior
    fit_features = gate_train_features
    report_provenance: CorpusProvenance = staged_provenance
    access_event_sequence = [
        "manifest_schema_source_seal_contracts_validated",
        "fit_feature_fit_label_selection_feature_selection_label_sha256_verified",
        "fit_features_deserialized",
        "fit_labels_deserialized",
        "selection_features_not_opened_not_deserialized",
        "selection_labels_not_opened_not_deserialized",
        "physical_fit_groups_assigned_by_frozen_label_free_internal_split",
        "gate_train_projector_utility_and_threshold_fit",
        "gate_eval_predictions_and_contexts_frozen_before_gate_eval_label_scoring",
        "fit_internal_gate_evaluated",
    ]
    if not fit_gate_passed:
        access_event_sequence.append(
            "fit_gate_failed_selection_materialization_prediction_and_scoring_skipped"
        )
        selection_report = {
            "executed": False,
            "reason": "pre_registered_fit_only_gate_failed",
            "selection_prediction_generated": False,
            "selection_label_scored": False,
        }
        status = "fit_only_gate_no_go_no_selection_predictions"
    else:
        # Only a passed independent internal gate permits a full physical-fit
        # retrain.  This still precedes and does not require physical selection
        # materialisation.
        fit_posterior = fit_fit_role_group_oof_posteriors(
            corpus, frozen.projector
        )
        fit_tasks = _sample_tasks_for_role(corpus, FIT_ROLE, frozen.sampling)
        fit_features = build_role_feature_set(
            corpus,
            staged_provenance,
            fit_posterior,
            fit_tasks,
            outcome_labels_allowed=True,
        )
        access_event_sequence.append(
            "fit_internal_gate_passed_complete_physical_fit_retrained"
        )
        # Stage 2 begins only after the fit gate.  The complete loader now
        # materialises and aligns selection labels against the already-attested
        # Stage-1 corpus before any selection prediction or scoring can run.
        corpus, verified_provenance = materialize_selection_labels_after_fit_gate(
            staged_load,
            sidecar_dir=sidecar_dir,
            manifest_path=manifest_path,
            model_config=model_config,
        )
        report_provenance = verified_provenance
        access_event_sequence.append(
            "fit_gate_passed_selection_labels_materialized_and_alignment_reverified"
        )
        selection_posterior = fit_full_fit_predict_selection_posteriors(
            corpus,
            frozen.projector,
            fit_oof=fit_posterior,
        )
        selection_tasks = _sample_tasks_for_role(
            corpus, SELECTION_ROLE, frozen.sampling
        )
        selection_features = build_role_feature_set(
            corpus,
            verified_provenance,
            selection_posterior,
            selection_tasks,
        )
        selection_report = {
            "executed": True,
            **evaluate_model_selection(
                corpus,
                selection_posterior,
                fit_features,
                selection_features,
                frozen.balance,
                frozen.utility_specs,
                ece_bins=ece_bins,
            ),
        }
        access_event_sequence.extend(
            [
                "selection_predictions_generated_from_complete_fit_role",
                "selection_labels_scored",
            ]
        )
        status = (
            "model_selection_registered_gate_pass"
            if bool(selection_report["registered_current_gate"]["passed"])
            else "model_selection_no_go"
        )

    _assert_files_match_snapshot(input_snapshot, stage="after all model fitting")
    source_hashes = {
        "canonical_sidecar_manifest_sha256": report_provenance.manifest_sha256,
        "model_config_sha256": input_snapshot["model_config"][1],
        "repair_config_sha256": input_snapshot["repair_config"][1],
        "repair_3_module_sha256": sha256_file(Path(__file__)),
        "emotion_relation_module_sha256": sha256_file(
            Path(inspect.getsourcefile(build_emotion_probability_relations) or "")
        ),
        "class_balanced_policy_module_sha256": sha256_file(
            Path(inspect.getsourcefile(fit_class_balanced_seed_scores) or "")
        ),
        "fit_gate_stage_loader_module_sha256": sha256_file(
            Path(inspect.getsourcefile(load_emotiontalk_fit_gate_stage) or "")
        ),
        "full_materialization_loader_module_sha256": sha256_file(
            Path(inspect.getsourcefile(load_emotiontalk_open_role_corpus) or "")
        ),
        "repair_2_config_sha256": REPAIR2_CONFIG_SHA256,
        "repair_2_result_sha256": REPAIR2_RESULT_SHA256,
        "projector_config_sha256": fit_posterior.producer_config_sha256,
        "vad_coordinate_sha256": VAD_COORDINATE_SHA256,
        "fit_internal_gate_split_spec_sha256": FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256,
        "fit_internal_gate_group_assignment_sha256": (
            internal_split.group_assignment_sha256
        ),
        "fit_internal_gate_row_assignment_sha256": (
            internal_split.row_assignment_sha256
        ),
        "capacity_control_spec_sha256": CAPACITY_CONTROL_SPEC_SHA256,
        "capacity_control_matrix_sha256": CAPACITY_CONTROL_MATRIX_SHA256,
        "fit_task_order_sha256": fit_features.provenance.task_order_sha256,
        "fit_base_cache_lineage_sha256": fit_features.base_cache_lineage_sha256,
    }
    if selection_features is not None:
        source_hashes.update(
            {
                "selection_task_order_sha256": selection_features.provenance.task_order_sha256,
                "selection_base_cache_lineage_sha256": selection_features.base_cache_lineage_sha256,
            }
        )
    canonical_manifest_hash = _canonical_sha256(
        {
            "protocol": PROTOCOL,
            "source_hashes": source_hashes,
            "class_order": CLASS_ORDER,
            "variant_order": VARIANT_ORDER,
            "fit_gate_passed": fit_gate_passed,
        }
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "analysis_stage": "open_role_repair_3_of_3_not_confirmatory",
        "registered_output_filename": REGISTERED_OUTPUT_FILENAME,
        "registered_output_repository_relative_path": (
            REGISTERED_OUTPUT_REPOSITORY_RELATIVE_PATH
        ),
        "fixed_primary": {
            "name": PRIMARY_VARIANT,
            "width": VARIANT_WIDTHS[PRIMARY_VARIANT],
            "components": [
                "canonical_59d_task_features",
                "raw_three_modality_seven_class_posterior_concat",
                "complete_3x3_current_history_relations",
                "fixed_vad_state_and_transition",
            ],
            "chosen_before_fit_gate_or_model_selection_results": True,
            "selection_results_can_choose_variant": False,
        },
        "fit_internal_gate_split": internal_split.aggregate_attestation(),
        "capacity_control_contract": {
            "variant": CAPACITY_CONTROL_VARIANT,
            "descriptor": dict(CAPACITY_CONTROL_SPEC),
            "canonical_json": json.dumps(
                dict(CAPACITY_CONTROL_SPEC),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "spec_sha256": CAPACITY_CONTROL_SPEC_SHA256,
            "matrix_sha256": CAPACITY_CONTROL_MATRIX_SHA256,
            "rank": 59,
            "primary_parameter_count": PRIMARY_OR_CONTROL_PARAMETER_COUNT,
            "capacity_control_parameter_count": PRIMARY_OR_CONTROL_PARAMETER_COUNT,
            "base_59d_parameter_count": BASE_59D_PARAMETER_COUNT,
            "superseded_handoff_only_unverifiable_spec_sha256": (
                SUPERSEDED_UNVERIFIABLE_CAPACITY_SPEC_SHA256
            ),
            "superseded_before_any_real_fit_gate_result": True,
        },
        "ablations": [
            {
                "name": name,
                "width": VARIANT_WIDTHS[name],
                "role": "explanatory_only_not_model_selection",
            }
            for name in VARIANT_ORDER[1:]
        ],
        "emotion_domain_contract": {
            "class_order": list(CLASS_ORDER),
            "projectors": ["text", "audio", "video"],
            "fit_prediction_mode": "whole_group_out_of_fold",
            "selection_prediction_mode": "model_fit_only_on_complete_fit_role",
            "gold_labels_in_inference_features": False,
            "vad_coordinate_system": VAD_COORDINATE_SYSTEM,
            "vad_claim_boundary": VAD_CLAIM_BOUNDARY,
            "vad_coordinate_sha256": VAD_COORDINATE_SHA256,
        },
        "fit_only_open_gate": fit_gate,
        "model_selection": selection_report,
        "access_contract": {
            "loader": (
                "load_emotiontalk_fit_gate_stage_then_"
                "materialize_selection_labels_after_fit_gate"
            ),
            "strict_physical_fit_and_selection_sidecars": True,
            "raw_npz_argument": False,
            "transcription_argument": False,
            "pickle_argument": False,
            "data_directory_argument": False,
            "sealed_role_argument": False,
            "calibration_holdout_validation_test_opened": False,
            "selection_prediction_requires_fit_gate": True,
            "selection_label_scoring_requires_fit_gate": True,
            "complete_physical_fit_retraining_requires_internal_gate": True,
            "fit_internal_gate_namespace": FIT_INTERNAL_GATE_NAMESPACE,
            "fit_internal_gate_uses_physical_fit_groups_only": True,
            "gate_eval_labels_used_for_projector_utility_or_threshold": False,
            "stage_1_selection_feature_sha256_verified": True,
            "stage_1_selection_feature_payload_opened": False,
            "stage_1_selection_feature_payload_deserialized": False,
            "stage_1_selection_prediction_generated": False,
            "stage_1_selection_label_sha256_verified": True,
            "stage_1_selection_label_payload_opened": False,
            "stage_1_selection_label_payload_deserialized": False,
            "selection_feature_payload_opened_after_fit_gate": bool(
                fit_gate_passed
            ),
            "selection_label_payload_deserialized_after_fit_gate": bool(
                fit_gate_passed
            ),
            "access_event_sequence": access_event_sequence,
        },
        "fit_gate_stage_provenance": _provenance_report(staged_provenance),
        "provenance": _provenance_report(report_provenance),
        "source_hashes": source_hashes,
        "registered_input_locks": {
            "sidecar_manifest_sha256": REGISTERED_SIDECAR_MANIFEST_SHA256,
            "model_config_sha256": REGISTERED_MODEL_CONFIG_SHA256,
            "verified_before_sidecar_deserialization_or_model_training": True,
            "manifest_model_and_repair_config_reverified_after_model_fitting": True,
            "manifest_model_and_repair_config_reverified_before_report_commit": True,
        },
        "canonical_reproducibility_manifest_sha256": canonical_manifest_hash,
        "environment": _environment(),
        "optional_teacher": _optional_teacher_report(frozen.raw),
        "claim_boundary": (
            "Open-role Repair 3 only. A passed model-selection gate is not a held-out, "
            "multi-dataset, or top-conference result. Calibration, holdout, validation, "
            "test, and external datasets remain unopened."
        ),
    }
    _assert_files_match_snapshot(input_snapshot, stage="before aggregate report commit")
    _assert_aggregate_output(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, output_path.resolve())
    return report


def _validate_runner_signature() -> None:
    names = tuple(inspect.signature(run_emotion_relation_vad_repair).parameters)
    expected = (
        "sidecar_dir",
        "manifest_path",
        "model_config_path",
        "repair_config_path",
        "output_path",
    )
    if names != expected:
        raise AssertionError("Repair-3 runner signature changed")
    forbidden = {"raw", "npz", "transcription", "pickle", "data", "sealed", "test"}
    for name in names:
        if set(name.lower().split("_")) & forbidden:
            raise AssertionError("Repair-3 runner exposes a forbidden input")


_validate_runner_signature()
