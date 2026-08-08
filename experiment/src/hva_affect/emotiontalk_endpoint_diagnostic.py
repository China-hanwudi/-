from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data_contract import ContractError, write_json_atomic
from .emotiontalk_multimodal_external import (
    _align_probabilities,
    _fit_processors,
    _predict_selector,
    _risk_classifier,
    _risk_regressor,
    _slice_blocks,
    _transform_processors,
    base_features,
    build_blocks,
    load_media_split,
    load_unlabeled_frame,
    selector_features,
)
from .emotiontalk_text_p1 import LABEL_NAMES, build_history_indices
from .meld_text_pilot import make_classifier, sha256_file, true_class_loss
from .negative_transfer_benchmark import (
    BenchmarkColumns,
    evaluate_benchmark,
    policy_at_coverage,
)
from .scu_set import assign_group_role


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_diagnostic_config(config: Mapping[str, object]) -> None:
    required = {
        "protocol",
        "dataset",
        "role_protocol",
        "primary_modalities",
        "crossfit_folds",
        "base_seeds",
        "risk_seeds",
        "coverages",
        "bootstrap_replicates",
        "bootstrap_seed",
        "roles",
        "gates",
        "sealed_policy",
    }
    missing = required - set(config)
    if missing:
        raise ContractError(f"diagnostic config missing keys: {sorted(missing)}")
    expected_roles = {
        "base_and_utility_fit": [0, 64],
        "model_selection": [65, 79],
        "calibration": [80, 89],
        "internal_holdout_sealed": [90, 99],
    }
    if config["roles"] != expected_roles:
        raise ContractError("diagnostic role ranges changed")
    if any(bool(value) for value in config["sealed_policy"].values()):
        raise ContractError("a sealed target read was enabled")
    if list(config["primary_modalities"]) != ["text", "audio", "video"]:
        raise ContractError("primary diagnostic must remain three-modal")
    if int(config["crossfit_folds"]) < 2:
        raise ContractError("cross-fitting requires at least two folds")
    if len(config["base_seeds"]) < 2 or len(config["risk_seeds"]) < 2:
        raise ContractError("multi-seed diagnostic requires at least two seeds")


def assign_frame_roles(
    frame: pd.DataFrame,
    *,
    dataset: str,
    role_protocol: str,
    role_ranges: Mapping[str, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not {"group", "dialogue"}.issubset(frame.columns):
        raise ContractError("frame missing group/dialogue")
    groups = (frame["group"].astype(str) + "/" + frame["dialogue"].astype(str)).to_numpy()
    group_assignment = {
        group: assign_group_role(dataset, group, role_protocol, role_ranges)
        for group in sorted(set(groups))
    }
    roles = np.asarray([group_assignment[group][0] for group in groups], dtype=object)
    buckets = np.asarray([group_assignment[group][1] for group in groups], dtype=np.int16)
    return groups, roles, buckets


def _load_materialized_labels(
    data_dir: Path,
    all_keys: Sequence[str],
    materialized_keys: Sequence[str],
) -> np.ndarray:
    with np.load(data_dir / "mm_label.npz", allow_pickle=True) as archive:
        payload = archive["train_corpus"]
        if payload.shape != () or payload.dtype != object:
            raise ContractError("malformed train_corpus label payload")
        labels = payload.item()
    if set(labels) != set(all_keys):
        raise ContractError("label/media key mismatch for train_corpus")
    values = np.asarray([int(labels[str(key)]["emo"]) for key in materialized_keys], dtype=np.int64)
    if np.any((values < 0) | (values >= len(LABEL_NAMES))):
        raise ContractError("invalid materialized train label")
    return values


def _selected_mask(score: np.ndarray, coverage: float) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    keep = max(1, int(np.floor(float(coverage) * len(score))))
    order = np.lexsort((np.arange(len(score)), score))
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:keep]] = True
    return selected


def objective_reversal_summary(
    excess: np.ndarray,
    mean_score: np.ndarray,
    harm_score: np.ndarray,
    coverages: Sequence[float],
    gates: Mapping[str, float | int],
) -> dict:
    excess = np.asarray(excess, dtype=float)
    mean_score = np.asarray(mean_score, dtype=float)
    harm_score = np.asarray(harm_score, dtype=float)
    if not (len(excess) == len(mean_score) == len(harm_score)):
        raise ValueError("objective arrays are not aligned")
    rows: dict[str, dict] = {}
    reversal_count = 0
    for coverage in coverages:
        mean_selected = _selected_mask(mean_score, float(coverage))
        harm_selected = _selected_mask(harm_score, float(coverage))
        union = int(np.logical_or(mean_selected, harm_selected).sum())
        intersection = int(np.logical_and(mean_selected, harm_selected).sum())
        jaccard = float(intersection / union) if union else 1.0
        mean_metrics = policy_at_coverage(excess, mean_score, float(coverage))
        harm_metrics = policy_at_coverage(excess, harm_score, float(coverage))
        mean_advantage = float(harm_metrics["mean_policy_regret"] - mean_metrics["mean_policy_regret"])
        harm_advantage = float(mean_metrics["harm_rate_among_used"] - harm_metrics["harm_rate_among_used"])
        reversal = bool(
            jaccard <= float(gates["maximum_selected_set_jaccard_for_reversal"])
            and mean_advantage >= float(gates["minimum_mean_regret_advantage_for_reversal"])
            and harm_advantage >= float(gates["minimum_harm_rate_advantage_for_reversal"])
        )
        reversal_count += int(reversal)
        rows[f"{float(coverage):.2f}"] = {
            "selected_set_jaccard": jaccard,
            "mean_risk_policy": mean_metrics,
            "harm_probability_policy": harm_metrics,
            "mean_regret_advantage_of_mean_risk": mean_advantage,
            "harm_rate_advantage_of_harm_probability": harm_advantage,
            "preference_reversal": reversal,
        }
    return {
        "score_spearman": float(spearmanr(mean_score, harm_score).statistic),
        "coverages": rows,
        "reversal_coverages": int(reversal_count),
        "minimum_reversal_coverages": int(gates["minimum_reversal_coverages"]),
        "pass": bool(reversal_count >= int(gates["minimum_reversal_coverages"])),
    }


def summarize_seed_targets(targets: np.ndarray, ensemble_target: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=float)
    ensemble_target = np.asarray(ensemble_target, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != len(ensemble_target):
        raise ValueError("seed targets must be seeds by aligned rows")
    pairwise = [float(spearmanr(targets[left], targets[right]).statistic) for left, right in combinations(range(len(targets)), 2)]
    seed_to_ensemble = [float(spearmanr(target, ensemble_target).statistic) for target in targets]
    harm = targets > 0
    majority = np.mean(harm, axis=0) >= 0.5
    majority_agreement = np.mean(harm == majority[None, :], axis=0)
    unanimous = np.all(harm == harm[0][None, :], axis=0)
    return {
        "seeds": int(len(targets)),
        "rows": int(targets.shape[1]),
        "pairwise_spearman_median": float(np.median(pairwise)),
        "pairwise_spearman_min": float(np.min(pairwise)),
        "seed_to_ensemble_spearman": seed_to_ensemble,
        "majority_sign_agreement_mean": float(np.mean(majority_agreement)),
        "unanimous_sign_fraction": float(np.mean(unanimous)),
    }


def _fit_risk_heads(base_config: dict, x: np.ndarray, target: np.ndarray, seeds: Sequence[int]) -> dict:
    mean_models, harm_models = [], []
    harm = (target > 0).astype(int)
    if len(np.unique(harm)) != 2:
        raise ContractError("fit role lacks both beneficial and harmful endpoint targets")
    for seed in seeds:
        mean_model = _risk_regressor(base_config, int(seed), loss="squared_error")
        harm_model = _risk_classifier(base_config, int(seed))
        mean_model.fit(x, target)
        harm_model.fit(x, harm)
        mean_models.append(mean_model)
        harm_models.append(harm_model)
    return {"mean_models": mean_models, "harm_models": harm_models}


def _predict_risk_heads(heads: Mapping[str, Sequence], x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean([model.predict(x) for model in heads["mean_models"]], axis=0),
        "harm": np.mean([model.predict_proba(x)[:, 1] for model in heads["harm_models"]], axis=0),
    }


def _write_private_cache(
    path: Path,
    *,
    fit_x: np.ndarray,
    fit_target: np.ndarray,
    fit_seed_targets: np.ndarray,
    selection_x: np.ndarray,
    selection_target: np.ndarray,
    selection_seed_targets: np.ndarray,
    selection_counts: np.ndarray,
    selection_clusters: np.ndarray,
    feature_names: Sequence[str],
    base_config_sha256: str,
    diagnostic_config_sha256: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"private endpoint cache already exists: {path}")
    cluster_codes, _ = pd.factorize(pd.Series(selection_clusters), sort=True)
    if np.any(cluster_codes < 0):
        raise ContractError("selection clusters could not be encoded")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(["emotiontalk_endpoint_cache_v1"]),
            fit_x=np.asarray(fit_x, dtype=np.float64),
            fit_target=np.asarray(fit_target, dtype=np.float64),
            fit_seed_targets=np.asarray(fit_seed_targets, dtype=np.float64),
            selection_x=np.asarray(selection_x, dtype=np.float64),
            selection_target=np.asarray(selection_target, dtype=np.float64),
            selection_seed_targets=np.asarray(selection_seed_targets, dtype=np.float64),
            selection_counts=np.asarray(selection_counts, dtype=np.int32),
            selection_cluster_codes=np.asarray(cluster_codes, dtype=np.int32),
            feature_names=np.asarray(tuple(feature_names), dtype=str),
            base_config_sha256=np.asarray([base_config_sha256]),
            diagnostic_config_sha256=np.asarray([diagnostic_config_sha256]),
        )
    os.replace(temporary, path)


def _endpoint_probabilities(
    frame: pd.DataFrame,
    audio: np.ndarray,
    video: np.ndarray,
    quality: np.ndarray,
    quality_names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    selection_indices: np.ndarray,
    base_config: dict,
    diagnostic_config: dict,
) -> dict:
    n_classes = len(LABEL_NAMES)
    modalities = tuple(diagnostic_config["primary_modalities"])
    seeds = tuple(int(value) for value in diagnostic_config["base_seeds"])
    histories = build_history_indices(frame)
    oof_full = np.full((len(seeds), len(frame), n_classes), np.nan, dtype=np.float64)
    oof_current = np.full_like(oof_full, np.nan)
    oof_selector_x: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    splitter = GroupKFold(n_splits=int(diagnostic_config["crossfit_folds"]))
    fit_local = np.arange(len(fit_indices))
    for fold, (train_local, held_local) in enumerate(
        splitter.split(fit_local, y[fit_indices], groups[fit_indices]), start=1
    ):
        train_index = fit_indices[train_local]
        held_index = fit_indices[held_local]
        processors = _fit_processors(base_config, frame, audio, video, train_index)
        current = _transform_processors(processors, frame, audio, video)
        blocks = build_blocks(current, quality, quality_names, histories)
        full_x = base_features(blocks, modalities, use_history=True)
        current_x = base_features(blocks, modalities, use_history=False)
        for seed_index, seed in enumerate(seeds):
            model = make_classifier(base_config, seed + fold * 1000)
            model.fit(
                sparse.vstack([full_x[train_index], current_x[train_index]], format="csr"),
                np.concatenate([y[train_index], y[train_index]]),
            )
            oof_full[seed_index, held_index] = _align_probabilities(
                model, model.predict_proba(full_x[held_index]), n_classes
            )
            oof_current[seed_index, held_index] = _align_probabilities(
                model, model.predict_proba(current_x[held_index]), n_classes
            )
        held_full = np.mean(oof_full[:, held_index], axis=0)
        held_current = np.mean(oof_current[:, held_index], axis=0)
        held_features, held_names = selector_features(
            _slice_blocks(blocks, held_index), modalities, held_full, held_current
        )
        if oof_selector_x is None:
            oof_selector_x = np.full((len(frame), held_features.shape[1]), np.nan, dtype=np.float64)
            feature_names = held_names
        elif feature_names != held_names:
            raise RuntimeError("selector feature schema changed across folds")
        oof_selector_x[held_index] = held_features
        print(f"endpoint diagnostic crossfit fold {fold} complete", flush=True)
    if oof_selector_x is None or np.isnan(oof_selector_x[fit_indices]).any():
        raise RuntimeError("incomplete endpoint OOF selector features")
    if np.isnan(oof_full[:, fit_indices]).any() or np.isnan(oof_current[:, fit_indices]).any():
        raise RuntimeError("incomplete endpoint OOF probabilities")

    processors = _fit_processors(base_config, frame, audio, video, fit_indices)
    current = _transform_processors(processors, frame, audio, video)
    blocks = build_blocks(current, quality, quality_names, histories)
    full_x = base_features(blocks, modalities, use_history=True)
    current_x = base_features(blocks, modalities, use_history=False)
    selection_full = np.empty((len(seeds), len(selection_indices), n_classes), dtype=np.float64)
    selection_current = np.empty_like(selection_full)
    for seed_index, seed in enumerate(seeds):
        model = make_classifier(base_config, seed)
        model.fit(
            sparse.vstack([full_x[fit_indices], current_x[fit_indices]], format="csr"),
            np.concatenate([y[fit_indices], y[fit_indices]]),
        )
        selection_full[seed_index] = _align_probabilities(
            model, model.predict_proba(full_x[selection_indices]), n_classes
        )
        selection_current[seed_index] = _align_probabilities(
            model, model.predict_proba(current_x[selection_indices]), n_classes
        )
        print(f"endpoint diagnostic final base seed {seed} complete", flush=True)
    selection_features, selection_feature_names = selector_features(
        _slice_blocks(blocks, selection_indices),
        modalities,
        np.mean(selection_full, axis=0),
        np.mean(selection_current, axis=0),
    )
    if feature_names != selection_feature_names:
        raise RuntimeError("selection selector feature schema differs from OOF schema")
    return {
        "histories": histories,
        "oof_full": oof_full,
        "oof_current": oof_current,
        "oof_selector_x": oof_selector_x,
        "selection_full": selection_full,
        "selection_current": selection_current,
        "selection_selector_x": selection_features,
        "feature_names": feature_names,
    }


def run_endpoint_diagnostic(
    data_dir: Path,
    feature_path: Path,
    base_config_path: Path,
    diagnostic_config_path: Path,
    output_path: Path,
    private_cache_path: Path | None = None,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_path}")
    base_config = _read_json(base_config_path)
    diagnostic_config = _read_json(diagnostic_config_path)
    _validate_diagnostic_config(diagnostic_config)
    if base_config.get("sealed_split") != "test_corpus":
        raise ContractError("base config no longer seals EmotionTalk test")

    keys, audio, video, quality, quality_names, feature_config_sha = load_media_split(
        feature_path, "train_corpus"
    )
    full_frame = load_unlabeled_frame(data_dir, keys)
    groups_all, roles_all, buckets_all = assign_frame_roles(
        full_frame,
        dataset=str(diagnostic_config["dataset"]),
        role_protocol=str(diagnostic_config["role_protocol"]),
        role_ranges=diagnostic_config["roles"],
    )
    allowed_roles = {"base_and_utility_fit", "model_selection"}
    materialized = np.asarray([role in allowed_roles for role in roles_all], dtype=bool)
    if np.any(np.isin(roles_all[materialized], ["calibration", "internal_holdout_sealed"])):
        raise ContractError("sealed role entered materialized diagnostic frame")
    work_frame = full_frame.loc[materialized].copy().reset_index(drop=True)
    work_frame["_row_id"] = np.arange(len(work_frame), dtype=int)
    work_audio = audio[materialized]
    work_video = video[materialized]
    work_quality = quality[materialized]
    work_groups = groups_all[materialized]
    work_roles = roles_all[materialized]
    y = _load_materialized_labels(data_dir, keys, work_frame["key"].astype(str).tolist())
    fit_indices = np.flatnonzero(work_roles == "base_and_utility_fit")
    selection_indices = np.flatnonzero(work_roles == "model_selection")
    if set(work_groups[fit_indices]) & set(work_groups[selection_indices]):
        raise ContractError("fit and model-selection groups overlap")
    if not np.array_equal(np.sort(np.unique(y[fit_indices])), np.arange(len(LABEL_NAMES))):
        raise ContractError("fit role does not contain all emotion classes")

    probability = _endpoint_probabilities(
        work_frame,
        work_audio,
        work_video,
        work_quality,
        quality_names,
        y,
        work_groups,
        fit_indices,
        selection_indices,
        base_config,
        diagnostic_config,
    )
    fit_seed_targets = np.asarray([
        true_class_loss(y[fit_indices], probability["oof_full"][seed, fit_indices])
        - true_class_loss(y[fit_indices], probability["oof_current"][seed, fit_indices])
        for seed in range(len(diagnostic_config["base_seeds"]))
    ])
    fit_target = (
        true_class_loss(y[fit_indices], np.mean(probability["oof_full"][:, fit_indices], axis=0))
        - true_class_loss(y[fit_indices], np.mean(probability["oof_current"][:, fit_indices], axis=0))
    )
    selection_seed_targets = np.asarray([
        true_class_loss(y[selection_indices], probability["selection_full"][seed])
        - true_class_loss(y[selection_indices], probability["selection_current"][seed])
        for seed in range(len(diagnostic_config["base_seeds"]))
    ])
    selection_target = (
        true_class_loss(y[selection_indices], np.mean(probability["selection_full"], axis=0))
        - true_class_loss(y[selection_indices], np.mean(probability["selection_current"], axis=0))
    )
    fit_eligible = np.asarray([
        len(probability["histories"][index]) > 0 for index in fit_indices
    ], dtype=bool)
    selection_counts = np.asarray([
        len(probability["histories"][index]) for index in selection_indices
    ], dtype=np.int32)
    eligible = selection_counts > 0
    if private_cache_path is not None:
        _write_private_cache(
            private_cache_path,
            fit_x=probability["oof_selector_x"][fit_indices][fit_eligible],
            fit_target=fit_target[fit_eligible],
            fit_seed_targets=fit_seed_targets[:, fit_eligible],
            selection_x=probability["selection_selector_x"][eligible],
            selection_target=selection_target[eligible],
            selection_seed_targets=selection_seed_targets[:, eligible],
            selection_counts=selection_counts[eligible],
            selection_clusters=work_groups[selection_indices][eligible],
            feature_names=probability["feature_names"],
            base_config_sha256=sha256_file(base_config_path),
            diagnostic_config_sha256=sha256_file(diagnostic_config_path),
        )
    heads = _fit_risk_heads(
        base_config,
        probability["oof_selector_x"][fit_indices][fit_eligible],
        fit_target[fit_eligible],
        diagnostic_config["risk_seeds"],
    )
    score = _predict_risk_heads(heads, probability["selection_selector_x"])
    evaluation_frame = pd.DataFrame({
        "history_count": selection_counts,
        "excess_loss": selection_target,
        "cluster": work_groups[selection_indices],
        "mean_score": score["mean"],
        "harm_score": score["harm"],
    })
    benchmark = evaluate_benchmark(
        evaluation_frame,
        BenchmarkColumns("history_count", "excess_loss", "cluster"),
        {"three_modal_endpoint": {"mean": "mean_score", "harm_probability": "harm_score"}},
        diagnostic_config["coverages"],
        bootstrap_replicates=int(diagnostic_config["bootstrap_replicates"]),
        bootstrap_seed=int(diagnostic_config["bootstrap_seed"]),
    )
    reversal = objective_reversal_summary(
        selection_target[eligible],
        score["mean"][eligible],
        score["harm"][eligible],
        diagnostic_config["coverages"],
        diagnostic_config["gates"],
    )
    fit_stability = summarize_seed_targets(
        fit_seed_targets[:, fit_eligible], fit_target[fit_eligible]
    )
    selection_stability = summarize_seed_targets(
        selection_seed_targets[:, eligible], selection_target[eligible]
    )
    ranking = benchmark["selectors"]["three_modal_endpoint"]["scores"]
    gates = diagnostic_config["gates"]
    gate_checks = {
        "history_query_count": bool(int(eligible.sum()) >= int(gates["minimum_model_selection_history_queries"])),
        "cluster_count": bool(len(set(work_groups[selection_indices][eligible])) >= int(gates["minimum_model_selection_clusters"])),
        "mean_signal": bool(ranking["mean"]["ranking"]["spearman_excess"] >= float(gates["minimum_mean_score_spearman"])),
        "harm_signal": bool(ranking["harm_probability"]["ranking"]["harm_auc"] >= float(gates["minimum_harm_probability_auc"])),
        "fit_seed_stability": bool(
            fit_stability["pairwise_spearman_median"] >= float(gates["minimum_pairwise_seed_target_spearman"])
            and fit_stability["majority_sign_agreement_mean"] >= float(gates["minimum_majority_sign_agreement"])
        ),
        "selection_seed_stability": bool(
            selection_stability["pairwise_spearman_median"] >= float(gates["minimum_pairwise_seed_target_spearman"])
            and selection_stability["majority_sign_agreement_mean"] >= float(gates["minimum_majority_sign_agreement"])
        ),
        "sign_severity_preference_reversal": bool(reversal["pass"]),
    }
    role_counts = {
        role: {
            "rows": int(np.sum(roles_all == role)),
            "groups": int(len(set(groups_all[roles_all == role]))),
        }
        for role in diagnostic_config["roles"]
    }
    result = {
        "protocol": diagnostic_config["protocol"],
        "status": "train_only_model_selection_exploratory; calibration_and_internal_holdout_unread",
        "claim_boundary": "Model-selection evidence can justify subset augmentation, but cannot confirm the final method.",
        "hashes": {
            "base_config_sha256": sha256_file(base_config_path),
            "diagnostic_config_sha256": sha256_file(diagnostic_config_path),
            "feature_sha256": sha256_file(feature_path),
            "feature_config_sha256": feature_config_sha,
            "transcription_sha256": sha256_file(data_dir / "transcription.csv"),
            "labels_sha256": sha256_file(data_dir / "mm_label.npz"),
        },
        "roles": role_counts,
        "target_stability": {"fit": fit_stability, "model_selection": selection_stability},
        "benchmark": benchmark,
        "objective_reversal": reversal,
        "gate_checks": gate_checks,
        "proceed_to_stochastic_subset_augmentation": bool(all(gate_checks.values())),
        "sealed_audit": {
            "calibration_rows_used_for_training_or_metrics": 0,
            "internal_holdout_rows_used_for_training_or_metrics": 0,
            "validation_rows_used": 0,
            "test_rows_used": 0,
            "row_level_output_emitted": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, output_path.resolve())
    return result
