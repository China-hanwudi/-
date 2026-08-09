"""Leakage-safe MELD text feasibility pilot for CARMA-Affect.

This module deliberately addresses only the mechanism feasibility question.
It never opens the MELD test split and must not be interpreted as a
multimodal or submission-grade result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


NLL_PROBABILITY_FLOOR = 1e-12


METHODS = (
    "full_history",
    "recent_1",
    "recent_3",
    "decay_0_7",
    "similarity_top1",
)


@dataclass(frozen=True)
class SplitData:
    frame: pd.DataFrame
    histories: tuple[tuple[int, ...], ...]
    current: sparse.csr_matrix
    labels: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labeled_split(path: Path) -> pd.DataFrame:
    """Load train/dev only. Callers are responsible for never passing test."""
    if path.name.startswith("test"):
        raise ValueError("test labels are sealed for this pilot")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Utterance",
        "Speaker",
        "Emotion",
        "Dialogue_ID",
        "Utterance_ID",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"missing required values in {path.name}")
    frame = frame.copy()
    frame["_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def build_history_indices(frame: pd.DataFrame) -> tuple[tuple[int, ...], ...]:
    """Return same-dialogue, same-speaker, strictly earlier row positions."""
    histories: list[list[int]] = [[] for _ in range(len(frame))]
    grouped = frame.groupby("Dialogue_ID", sort=True)
    for _, dialogue in grouped:
        ordered = dialogue.sort_values("Utterance_ID", kind="stable")
        prior: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for _, row in ordered.iterrows():
            row_id = int(row["_row_id"])
            utt_id = int(row["Utterance_ID"])
            speaker = str(row["Speaker"])
            selected = [rid for previous_utt, rid in prior[speaker] if previous_utt < utt_id]
            if len(selected) != len(prior[speaker]):
                raise ValueError("non-strict history detected")
            histories[row_id] = selected
            prior[speaker].append((utt_id, row_id))
    return tuple(tuple(values) for values in histories)


def _selection_weights(
    method: str,
    query_index: int,
    candidates: Sequence[int],
    current: sparse.csr_matrix,
) -> tuple[list[int], np.ndarray]:
    if not candidates:
        return [], np.empty(0, dtype=np.float64)
    if method == "full_history":
        selected = list(candidates)
        weights = np.ones(len(selected), dtype=np.float64)
    elif method == "recent_1":
        selected = [candidates[-1]]
        weights = np.ones(1, dtype=np.float64)
    elif method == "recent_3":
        selected = list(candidates[-3:])
        weights = np.ones(len(selected), dtype=np.float64)
    elif method == "decay_0_7":
        selected = list(candidates)
        ages = np.arange(len(selected) - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-0.7 * ages)
    elif method == "similarity_top1":
        candidate_matrix = current[list(candidates)]
        scores = (candidate_matrix @ current[query_index].T).toarray().ravel()
        best = int(np.argmax(scores))
        selected = [candidates[best]]
        weights = np.ones(1, dtype=np.float64)
    else:
        raise ValueError(f"unknown method: {method}")
    weights /= weights.sum()
    return selected, weights


def aggregate_history(
    method: str,
    current: sparse.csr_matrix,
    histories: Sequence[Sequence[int]],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    selected_counts = np.zeros(len(histories), dtype=np.int32)
    for query_index, candidates in enumerate(histories):
        selected, weights = _selection_weights(method, query_index, candidates, current)
        selected_counts[query_index] = len(selected)
        rows.extend([query_index] * len(selected))
        cols.extend(selected)
        data.extend(weights.tolist())
    aggregator = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(histories), len(histories)), dtype=np.float64
    )
    return (aggregator @ current).tocsr(), selected_counts


def aggregate_custom_sets(
    current: sparse.csr_matrix, sets: Sequence[Sequence[int]]
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    counts = np.zeros(len(sets), dtype=np.int32)
    for query_index, selected in enumerate(sets):
        selected = list(selected)
        counts[query_index] = len(selected)
        if selected:
            weight = 1.0 / len(selected)
            rows.extend([query_index] * len(selected))
            cols.extend(selected)
            data.extend([weight] * len(selected))
    aggregator = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(sets), len(sets)), dtype=np.float64
    )
    return (aggregator @ current).tocsr(), counts


def combine_features(
    current: sparse.csr_matrix,
    history: sparse.csr_matrix,
    counts: np.ndarray,
) -> sparse.csr_matrix:
    count_feature = sparse.csr_matrix(np.log1p(counts).reshape(-1, 1))
    return sparse.hstack([current, history, count_feature], format="csr")


def zero_history_features(current: sparse.csr_matrix) -> sparse.csr_matrix:
    zeros = sparse.csr_matrix(current.shape, dtype=current.dtype)
    counts = np.zeros(current.shape[0], dtype=np.int32)
    return combine_features(current, zeros, counts)


def make_classifier(config: dict, seed: int) -> SGDClassifier:
    cfg = config["classifier"]
    return SGDClassifier(
        loss=cfg["loss"],
        penalty=cfg["penalty"],
        alpha=float(cfg["alpha"]),
        class_weight=cfg["class_weight"],
        average=bool(cfg["average"]),
        max_iter=int(cfg["max_iter"]),
        tol=float(cfg["tol"]),
        random_state=int(seed),
    )


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_label_ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= boundaries[index]) & (confidence <= boundaries[index + 1])
        else:
            mask = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def prediction_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int
) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    per_row_nll = true_class_loss(y_true, probabilities)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "log_loss": float(np.mean(per_row_nll)),
        "brier": multiclass_brier(y_true, probabilities),
        "ece": top_label_ece(y_true, probabilities, bins),
    }


def true_class_loss(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    selected = probabilities[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(selected, NLL_PROBABILITY_FLOOR, 1.0))


def cluster_bootstrap_difference(
    difference: np.ndarray,
    cluster_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    clusters = np.unique(cluster_ids)
    grouped = []
    for cluster in clusters:
        values = difference[cluster_ids == cluster]
        grouped.append(
            (
                float(values.sum()),
                int(values.size),
                int((values > 0).sum()),
                float(values[values > 0].sum()),
            )
        )
    grouped = np.asarray(grouped, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    harms = np.empty(replicates, dtype=np.float64)
    harm_magnitudes = np.empty(replicates, dtype=np.float64)
    for iteration in range(replicates):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        draw = grouped[sample].sum(axis=0)
        means[iteration] = draw[0] / draw[1]
        harms[iteration] = draw[2] / draw[1]
        harm_magnitudes[iteration] = draw[3] / draw[2] if draw[2] else 0.0
    point_harm = float((difference > 0).mean())
    point_harm_magnitude = float(difference[difference > 0].mean()) if (difference > 0).any() else 0.0
    return {
        "mean_excess_loss": float(difference.mean()),
        "mean_excess_loss_ci_low": float(np.quantile(means, 0.025)),
        "mean_excess_loss_ci_high": float(np.quantile(means, 0.975)),
        "harm_rate": point_harm,
        "harm_rate_ci_low": float(np.quantile(harms, 0.025)),
        "harm_rate_ci_high": float(np.quantile(harms, 0.975)),
        "mean_harm_nats": point_harm_magnitude,
        "mean_harm_nats_ci_low": float(np.quantile(harm_magnitudes, 0.025)),
        "mean_harm_nats_ci_high": float(np.quantile(harm_magnitudes, 0.975)),
    }


def cluster_bootstrap_binary(
    indicator: np.ndarray,
    cluster_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    clusters = np.unique(cluster_ids)
    grouped = []
    for cluster in clusters:
        values = indicator[cluster_ids == cluster]
        grouped.append((int(values.sum()), int(values.size)))
    grouped = np.asarray(grouped, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for iteration in range(replicates):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        draw = grouped[sample].sum(axis=0)
        estimates[iteration] = draw[0] / draw[1]
    return {
        "rate": float(indicator.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _build_vectorizer(config: dict) -> TfidfVectorizer:
    cfg = config["vectorizer"]
    return TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=int(cfg["min_df"]),
        max_df=float(cfg["max_df"]),
        max_features=int(cfg["max_features"]),
        sublinear_tf=bool(cfg["sublinear_tf"]),
        strip_accents=cfg["strip_accents"],
        lowercase=True,
        dtype=np.float64,
    )


def _align_probabilities(model: SGDClassifier, raw: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
    aligned[:, model.classes_.astype(int)] = raw
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _custom_feature_sets(
    current: sparse.csr_matrix,
    sets: Sequence[Sequence[int]],
) -> sparse.csr_matrix:
    history, counts = aggregate_custom_sets(current, sets)
    return combine_features(current, history, counts)


def run_pilot(
    train_path: Path,
    dev_path: Path,
    config_path: Path,
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_frame = load_labeled_split(train_path)
    dev_frame = load_labeled_split(dev_path)
    train_histories = build_history_indices(train_frame)
    dev_histories = build_history_indices(dev_frame)

    vectorizer = _build_vectorizer(config)
    train_current = vectorizer.fit_transform(train_frame["Utterance"].astype(str)).tocsr()
    dev_current = vectorizer.transform(dev_frame["Utterance"].astype(str)).tocsr()

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_frame["Emotion"].astype(str))
    dev_y = encoder.transform(dev_frame["Emotion"].astype(str))
    n_classes = len(encoder.classes_)
    zero_train = zero_history_features(train_current)
    zero_dev = zero_history_features(dev_current)

    seeds = [int(value) for value in config["seeds"]]
    bins = int(config["ece_bins"])
    method_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    method_zero_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    per_seed_metrics: dict[str, list[dict]] = defaultdict(list)
    fitted_models: dict[str, list[SGDClassifier]] = defaultdict(list)

    # Frozen current-only champion candidate, same feature dimensionality.
    for seed in seeds:
        model = make_classifier(config, seed)
        model.fit(zero_train, train_y)
        probabilities = _align_probabilities(model, model.predict_proba(zero_dev), n_classes)
        method_probabilities["current_only"].append(probabilities)
        per_seed_metrics["current_only"].append(
            {"seed": seed, **prediction_metrics(dev_y, probabilities, bins)}
        )

    selected_counts_by_method: dict[str, np.ndarray] = {}
    dev_features_by_method: dict[str, sparse.csr_matrix] = {}
    for method in config["history_methods"]:
        train_history, train_counts = aggregate_history(method, train_current, train_histories)
        dev_history, dev_counts = aggregate_history(method, dev_current, dev_histories)
        train_actual = combine_features(train_current, train_history, train_counts)
        dev_actual = combine_features(dev_current, dev_history, dev_counts)
        selected_counts_by_method[method] = dev_counts
        dev_features_by_method[method] = dev_actual

        if config["history_dropout_augmentation"]:
            fit_x = sparse.vstack([train_actual, zero_train], format="csr")
            fit_y = np.concatenate([train_y, train_y])
        else:
            fit_x, fit_y = train_actual, train_y

        for seed in seeds:
            model = make_classifier(config, seed)
            model.fit(fit_x, fit_y)
            actual_probability = _align_probabilities(
                model, model.predict_proba(dev_actual), n_classes
            )
            zero_probability = _align_probabilities(
                model, model.predict_proba(zero_dev), n_classes
            )
            method_probabilities[method].append(actual_probability)
            method_zero_probabilities[method].append(zero_probability)
            fitted_models[method].append(model)
            actual_metrics = prediction_metrics(dev_y, actual_probability, bins)
            zero_metrics = prediction_metrics(dev_y, zero_probability, bins)
            eligible = dev_counts > 0
            difference = (
                true_class_loss(dev_y[eligible], actual_probability[eligible])
                - true_class_loss(dev_y[eligible], zero_probability[eligible])
            )
            per_seed_metrics[method].append(
                {
                    "seed": seed,
                    **actual_metrics,
                    "within_model_zero_weighted_f1": zero_metrics["weighted_f1"],
                    "eligible_queries": int(eligible.sum()),
                    "within_model_harm_rate": float((difference > 0).mean()),
                    "within_model_mean_excess_loss": float(difference.mean()),
                }
            )

    ensemble: dict[str, np.ndarray] = {
        method: np.mean(probabilities, axis=0)
        for method, probabilities in method_probabilities.items()
    }
    ensemble_zero: dict[str, np.ndarray] = {
        method: np.mean(probabilities, axis=0)
        for method, probabilities in method_zero_probabilities.items()
    }

    method_reports: dict[str, dict] = {}
    per_query = pd.DataFrame(
        {
            "Dialogue_ID": dev_frame["Dialogue_ID"].to_numpy(),
            "Utterance_ID": dev_frame["Utterance_ID"].to_numpy(),
            "Speaker": dev_frame["Speaker"].astype(str).to_numpy(),
            "gold_emotion": dev_frame["Emotion"].astype(str).to_numpy(),
            "history_count": np.asarray([len(values) for values in dev_histories]),
        }
    )
    per_query["current_only_loss"] = true_class_loss(dev_y, ensemble["current_only"])

    for method in config["history_methods"]:
        actual_probability = ensemble[method]
        zero_probability = ensemble_zero[method]
        selected_counts = selected_counts_by_method[method]
        eligible = selected_counts > 0
        actual_loss = true_class_loss(dev_y, actual_probability)
        zero_loss = true_class_loss(dev_y, zero_probability)
        difference = actual_loss[eligible] - zero_loss[eligible]
        clusters = dev_frame.loc[eligible, "Dialogue_ID"].to_numpy()
        bootstrap = cluster_bootstrap_difference(
            difference,
            clusters,
            int(config["bootstrap_replicates"]),
            int(config["bootstrap_seed"]) + list(METHODS).index(method),
        )
        report = {
            "ensemble_metrics": prediction_metrics(dev_y, actual_probability, bins),
            "within_model_zero_metrics": prediction_metrics(dev_y, zero_probability, bins),
            "selected_history_mean": float(selected_counts.mean()),
            "eligible_queries": int(eligible.sum()),
            "within_model_history_effect": bootstrap,
            "seed_metrics": per_seed_metrics[method],
        }
        method_reports[method] = report
        per_query[f"{method}_loss"] = actual_loss
        per_query[f"{method}_zero_loss"] = zero_loss
        per_query[f"{method}_selected_count"] = selected_counts
        per_query[f"{method}_excess_loss"] = actual_loss - zero_loss

    # G1 model-dependent interaction probe using the full-history models.
    eligible_interaction = np.asarray([len(values) >= 2 for values in dev_histories])
    empty_sets: list[tuple[int, ...]] = []
    candidate_sets: list[tuple[int, ...]] = []
    context_sets: list[tuple[int, ...]] = []
    context_candidate_sets: list[tuple[int, ...]] = []
    all_except_sets: list[tuple[int, ...]] = []
    full_sets: list[tuple[int, ...]] = []
    for values in dev_histories:
        values = tuple(values)
        if len(values) >= 2:
            candidate = values[-1]
            context = values[-2]
            empty_sets.append(())
            candidate_sets.append((candidate,))
            context_sets.append((context,))
            context_candidate_sets.append((context, candidate))
            all_except_sets.append(values[:-1])
            full_sets.append(values)
        else:
            empty_sets.append(())
            candidate_sets.append(())
            context_sets.append(())
            context_candidate_sets.append(())
            all_except_sets.append(())
            full_sets.append(())

    custom = {
        "empty": _custom_feature_sets(dev_current, empty_sets),
        "candidate": _custom_feature_sets(dev_current, candidate_sets),
        "context": _custom_feature_sets(dev_current, context_sets),
        "context_candidate": _custom_feature_sets(dev_current, context_candidate_sets),
        "all_except": _custom_feature_sets(dev_current, all_except_sets),
        "full": _custom_feature_sets(dev_current, full_sets),
    }
    probe_losses: dict[str, list[np.ndarray]] = defaultdict(list)
    for model in fitted_models["full_history"]:
        for name, features in custom.items():
            probability = _align_probabilities(model, model.predict_proba(features), n_classes)
            probe_losses[name].append(true_class_loss(dev_y, probability))
    averaged_loss = {name: np.mean(values, axis=0) for name, values in probe_losses.items()}
    delta_empty = averaged_loss["empty"] - averaged_loss["candidate"]
    delta_context = averaged_loss["context"] - averaged_loss["context_candidate"]
    delta_all = averaged_loss["all_except"] - averaged_loss["full"]
    mask = eligible_interaction
    sign_flip_context = (delta_empty[mask] * delta_context[mask]) < 0
    sign_flip_all = (delta_empty[mask] * delta_all[mask]) < 0
    interaction_clusters = dev_frame.loc[mask, "Dialogue_ID"].to_numpy()
    context_ci = cluster_bootstrap_binary(
        sign_flip_context.astype(float),
        interaction_clusters,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]) + 100,
    )
    all_ci = cluster_bootstrap_binary(
        sign_flip_all.astype(float),
        interaction_clusters,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]) + 101,
    )
    interaction_report = {
        "status": "exploratory_model_dependent_probe",
        "eligible_queries": int(mask.sum()),
        "candidate_added_to_one_context_sign_flip": context_ci,
        "candidate_added_to_all_others_sign_flip": all_ci,
        "mean_abs_marginal_change_one_context": float(
            np.mean(np.abs(delta_empty[mask] - delta_context[mask]))
        ),
        "mean_abs_marginal_change_all_others": float(
            np.mean(np.abs(delta_empty[mask] - delta_all[mask]))
        ),
        "limitation": (
            "The probe is conditional on a linear text model trained with full/zero history "
            "augmentation; it is evidence of model-level conditional utility, not intrinsic "
            "multimodal human affect dynamics."
        ),
    }
    per_query["interaction_eligible"] = mask
    per_query["marginal_empty"] = delta_empty
    per_query["marginal_one_context"] = delta_context
    per_query["marginal_all_others"] = delta_all

    gates = config["gates"]
    g0_methods = []
    for method, report in method_reports.items():
        effect = report["within_model_history_effect"]
        consistent = sum(
            item["within_model_harm_rate"] >= gates["g0_min_harm_rate_ci_low"]
            for item in report["seed_metrics"]
        )
        if (
            effect["harm_rate"] >= gates["g0_min_harm_rate"]
            and effect["harm_rate_ci_low"] >= gates["g0_min_harm_rate_ci_low"]
            and effect["mean_harm_nats"] >= gates["g0_min_mean_harm_nats"]
            and consistent >= gates["g0_min_seed_consistency"]
        ):
            g0_methods.append(method)
    g0 = {"pass": bool(g0_methods), "qualifying_methods": g0_methods}

    g1_context = interaction_report["candidate_added_to_one_context_sign_flip"]
    g1_all = interaction_report["candidate_added_to_all_others_sign_flip"]
    g1 = {
        "pass_exploratory": bool(
            max(g1_context["rate"], g1_all["rate"]) >= gates["g1_min_sign_flip_rate"]
            and max(g1_context["ci_low"], g1_all["ci_low"])
            >= gates["g1_min_sign_flip_ci_low"]
            and max(
                interaction_report["mean_abs_marginal_change_one_context"],
                interaction_report["mean_abs_marginal_change_all_others"],
            )
            >= gates["g1_min_abs_marginal_change_nats"]
        ),
        "status": "exploratory_only",
    }

    best_f1 = max(
        report["ensemble_metrics"]["weighted_f1"] for report in method_reports.values()
    )
    full_harm = method_reports["full_history"]["within_model_history_effect"]["harm_rate"]
    sufficient_simple = []
    for method in ("recent_1", "recent_3", "decay_0_7", "similarity_top1"):
        report = method_reports[method]
        if (
            report["ensemble_metrics"]["weighted_f1"]
            >= best_f1 - gates["g2_best_f1_tolerance"]
            and report["within_model_history_effect"]["harm_rate"]
            <= full_harm - gates["g2_harm_reduction_vs_full"]
        ):
            sufficient_simple.append(method)
    g2 = {
        "complex_carma_needed": not bool(sufficient_simple),
        "simple_methods_meeting_rule": sufficient_simple,
    }

    current_report = {
        "ensemble_metrics": prediction_metrics(dev_y, ensemble["current_only"], bins),
        "seed_metrics": per_seed_metrics["current_only"],
    }
    result = {
        "protocol": config["protocol"],
        "scope": config["scope"],
        "data": {
            "train_file": train_path.name,
            "train_sha256": sha256_file(train_path),
            "train_rows": int(len(train_frame)),
            "dev_file": dev_path.name,
            "dev_sha256": sha256_file(dev_path),
            "dev_rows": int(len(dev_frame)),
            "test_opened": False,
            "labels": encoder.classes_.tolist(),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "config": config,
        "current_only": current_report,
        "methods": method_reports,
        "interaction_probe": interaction_report,
        "gates": {"G0": g0, "G1": g1, "G2": g2},
        "interpretation_boundary": (
            "A positive gate result supports only text-level mechanism feasibility on MELD dev. "
            "It does not validate multimodal CARMA-Affect, longitudinal personalization, "
            "or top-conference novelty."
        ),
    }
    return result, per_query


def write_outputs(result: dict, per_query: pd.DataFrame, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(output_json)
    per_query.to_csv(output_csv, index=False, compression="gzip")
