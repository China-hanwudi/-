"""Leakage-safe EmotionTalk text P1 interaction audit.

The official media archives are gated and not available locally.  This module
therefore uses only train/validation Chinese transcriptions and the official
multimodal-task label file.  It never opens the test corpus payload and cannot
be interpreted as multimodal evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from .data_contract import ContractError, write_json_atomic
from .emotiontalk_contract import parse_key
from .meld_text_pilot import (
    _align_probabilities,
    aggregate_custom_sets,
    cluster_bootstrap_binary,
    combine_features,
    make_classifier,
    prediction_metrics,
    sha256_file,
    true_class_loss,
    zero_history_features,
)


LABEL_NAMES = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "disgusted",
    "fearful",
)


def load_emotiontalk_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load train/validation only; fail closed on any test request."""

    if split not in {"train_corpus", "val_corpus"}:
        raise ValueError("EmotionTalk test labels are sealed for P1")
    transcript_path = data_dir / "transcription.csv"
    label_path = data_dir / "mm_label.npz"
    transcripts = pd.read_csv(transcript_path, encoding="utf-8-sig")
    required = {"name", "chinese"}
    if not required.issubset(transcripts.columns):
        raise ContractError(f"transcription.csv missing columns: {sorted(required)}")
    transcript_map = {
        Path(str(row["name"]).replace("\\", "/")).stem: str(row["chinese"])
        for _, row in transcripts.iterrows()
    }
    with np.load(label_path, allow_pickle=True) as archive:
        if split not in archive.files:
            raise ContractError(f"mm_label.npz missing {split}")
        payload = archive[split]
        if payload.shape != () or payload.dtype != object:
            raise ContractError(f"mm_label.npz/{split}: malformed payload")
        labels = payload.item()
    rows = []
    for key, target in labels.items():
        if key not in transcript_map:
            raise ContractError(f"missing transcription for {key}")
        group, dialogue, speaker, turn = parse_key(key)
        emotion_id = int(target["emo"])
        if not 0 <= emotion_id < len(LABEL_NAMES):
            raise ContractError(f"invalid emotion id for {key}: {emotion_id}")
        rows.append(
            {
                "key": key,
                "group": group,
                "dialogue": dialogue,
                "speaker": speaker,
                "turn": turn,
                "text": transcript_map[key],
                "emotion": LABEL_NAMES[emotion_id],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["key"].duplicated().any():
        raise ContractError(f"EmotionTalk {split}: empty or duplicate keys")
    frame["_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def build_history_indices(frame: pd.DataFrame) -> tuple[tuple[int, ...], ...]:
    """Strictly earlier history within group/dialogue/speaker streams."""

    histories: list[tuple[int, ...]] = [tuple() for _ in range(len(frame))]
    for _, stream in frame.groupby(["group", "dialogue", "speaker"], sort=True):
        ordered = stream.sort_values("turn", kind="stable")
        turns = ordered["turn"].astype(int).tolist()
        if len(turns) != len(set(turns)):
            raise ContractError("duplicate turn in EmotionTalk speaker-dialogue stream")
        prior: list[int] = []
        for _, row in ordered.iterrows():
            row_id = int(row["_row_id"])
            histories[row_id] = tuple(prior)
            prior.append(row_id)
    return tuple(histories)


def interaction_sets(
    histories: Sequence[Sequence[int]],
) -> tuple[dict[str, list[tuple[int, ...]]], np.ndarray, np.ndarray]:
    """Build variable- and fixed-cardinality interaction interventions."""

    names = (
        "empty",
        "candidate",
        "context",
        "context_candidate",
        "s1_placebo",
        "s1_candidate",
        "s2_placebo",
        "s2_candidate",
    )
    sets = {name: [] for name in names}
    variable_mask = np.zeros(len(histories), dtype=bool)
    fixed_mask = np.zeros(len(histories), dtype=bool)
    for index, values_raw in enumerate(histories):
        values = tuple(values_raw)
        if len(values) >= 2:
            variable_mask[index] = True
            candidate = values[-1]
            context = values[-2]
            sets["empty"].append(tuple())
            sets["candidate"].append((candidate,))
            sets["context"].append((context,))
            sets["context_candidate"].append((context, candidate))
        else:
            for name in ("empty", "candidate", "context", "context_candidate"):
                sets[name].append(tuple())

        if len(values) >= 4:
            fixed_mask[index] = True
            placebo = values[0]
            context_2 = values[-3]
            context_1 = values[-2]
            candidate = values[-1]
            if len({placebo, context_2, context_1, candidate}) != 4:
                raise ContractError("fixed-cardinality roles must be distinct")
            sets["s1_placebo"].append((context_1, placebo))
            sets["s1_candidate"].append((context_1, candidate))
            sets["s2_placebo"].append((context_2, placebo))
            sets["s2_candidate"].append((context_2, candidate))
        else:
            for name in ("s1_placebo", "s1_candidate", "s2_placebo", "s2_candidate"):
                sets[name].append(tuple())
    return sets, variable_mask, fixed_mask


def _restricted_candidate_resampling(
    rng: np.random.Generator,
    frame: pd.DataFrame,
    histories: Sequence[Sequence[int]],
    mask: np.ndarray,
) -> dict[int, int]:
    """Resample candidates within speaker while avoiding query-role collisions."""

    eligible = np.flatnonzero(mask)
    result: dict[int, int] = {}
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for query in eligible:
        by_speaker[str(frame.iloc[query]["speaker"])].append(int(query))
    for queries in by_speaker.values():
        pool = np.asarray([histories[q][-1] for q in queries], dtype=int)
        forbidden = [set(histories[q][-3:]) | {q} for q in queries]
        if len(queries) < 5:
            raise ContractError("too few fixed-cardinality queries for speaker resampling")
        proposed = []
        for blocked in forbidden:
            for _ in range(1000):
                value = int(rng.choice(pool))
                if value not in blocked:
                    proposed.append(value)
                    break
            else:
                raise ContractError("could not construct collision-free speaker resample")
        result.update({query: int(value) for query, value in zip(queries, proposed, strict=True)})
    return result


def _replace_candidates(
    base_sets: dict[str, list[tuple[int, ...]]],
    histories: Sequence[Sequence[int]],
    variable_mask: np.ndarray,
    fixed_mask: np.ndarray,
    replacement: dict[int, int],
) -> dict[str, list[tuple[int, ...]]]:
    candidate = [tuple() for _ in histories]
    context_candidate = [tuple() for _ in histories]
    s1_candidate = [tuple() for _ in histories]
    s2_candidate = [tuple() for _ in histories]
    for query in np.flatnonzero(variable_mask):
        # Fixed-cardinality queries receive a restricted replacement.  The
        # remaining variable-only queries keep empty interventions and are
        # excluded from the restricted null statistic.
        if int(query) not in replacement:
            continue
        value = replacement[int(query)]
        context = histories[int(query)][-2]
        candidate[int(query)] = (value,)
        context_candidate[int(query)] = (context, value)
    for query in np.flatnonzero(fixed_mask):
        value = replacement[int(query)]
        s1 = histories[int(query)][-2]
        s2 = histories[int(query)][-3]
        s1_candidate[int(query)] = (s1, value)
        s2_candidate[int(query)] = (s2, value)
    return {
        "candidate": candidate,
        "context_candidate": context_candidate,
        "s1_candidate": s1_candidate,
        "s2_candidate": s2_candidate,
        "empty": base_sets["empty"],
        "context": base_sets["context"],
        "s1_placebo": base_sets["s1_placebo"],
        "s2_placebo": base_sets["s2_placebo"],
    }


def _vectorizer(config: dict) -> TfidfVectorizer:
    cfg = config["vectorizer"]
    return TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=int(cfg["min_df"]),
        max_df=float(cfg["max_df"]),
        max_features=int(cfg["max_features"]),
        sublinear_tf=bool(cfg["sublinear_tf"]),
        dtype=np.float64,
    )


def run_p1(
    data_dir: Path, config_path: Path
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train = load_emotiontalk_split(data_dir, "train_corpus")
    validation = load_emotiontalk_split(data_dir, "val_corpus")
    train_histories = build_history_indices(train)
    validation_histories = build_history_indices(validation)

    vectorizer = _vectorizer(config)
    train_current = vectorizer.fit_transform(train["text"].astype(str)).tocsr()
    validation_current = vectorizer.transform(validation["text"].astype(str)).tocsr()
    train_history, train_counts = aggregate_custom_sets(train_current, train_histories)
    validation_history, validation_counts = aggregate_custom_sets(
        validation_current, validation_histories
    )
    train_actual = combine_features(train_current, train_history, train_counts)
    validation_actual = combine_features(
        validation_current, validation_history, validation_counts
    )
    zero_train = zero_history_features(train_current)
    zero_validation = zero_history_features(validation_current)

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train["emotion"].astype(str))
    validation_y = encoder.transform(validation["emotion"].astype(str))
    n_classes = len(encoder.classes_)
    fit_x = sparse.vstack([train_actual, zero_train], format="csr")
    fit_y = np.concatenate([train_y, train_y])
    models = []
    actual_probabilities = []
    zero_probabilities = []
    for seed in config["seeds"]:
        model = make_classifier(config, int(seed))
        model.fit(fit_x, fit_y)
        models.append(model)
        actual_probabilities.append(
            _align_probabilities(model, model.predict_proba(validation_actual), n_classes)
        )
        zero_probabilities.append(
            _align_probabilities(model, model.predict_proba(zero_validation), n_classes)
        )
    actual_probability = np.mean(actual_probabilities, axis=0)
    zero_probability = np.mean(zero_probabilities, axis=0)

    sets, variable_mask, fixed_mask = interaction_sets(validation_histories)

    def loss_for_sets(selected: list[tuple[int, ...]]) -> np.ndarray:
        history, counts = aggregate_custom_sets(validation_current, selected)
        features = combine_features(validation_current, history, counts)
        probability = np.mean(
            [
                _align_probabilities(model, model.predict_proba(features), n_classes)
                for model in models
            ],
            axis=0,
        )
        return true_class_loss(validation_y, probability)

    observed_losses = {name: loss_for_sets(selected) for name, selected in sets.items()}
    variable_empty = observed_losses["empty"] - observed_losses["candidate"]
    variable_context = observed_losses["context"] - observed_losses["context_candidate"]
    variable_flip = variable_empty[variable_mask] * variable_context[variable_mask] < 0
    fixed_1 = observed_losses["s1_placebo"] - observed_losses["s1_candidate"]
    fixed_2 = observed_losses["s2_placebo"] - observed_losses["s2_candidate"]
    fixed_flip = fixed_1[fixed_mask] * fixed_2[fixed_mask] < 0

    clusters = (
        validation.loc[fixed_mask, "group"].astype(str)
        + "_"
        + validation.loc[fixed_mask, "dialogue"].astype(str)
    ).to_numpy()
    observed_fixed_ci = cluster_bootstrap_binary(
        fixed_flip.astype(float),
        clusters,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]) + 2000,
    )

    rng = np.random.default_rng(int(config["bootstrap_seed"]) + 2100)
    null_variable_rates = []
    null_fixed_rates = []
    for _ in range(int(config["restricted_null_draws"])):
        replacement = _restricted_candidate_resampling(
            rng, validation, validation_histories, fixed_mask
        )
        null_sets = _replace_candidates(
            sets, validation_histories, variable_mask, fixed_mask, replacement
        )
        candidate_loss = loss_for_sets(null_sets["candidate"])
        context_candidate_loss = loss_for_sets(null_sets["context_candidate"])
        # Use the fixed-cardinality mask for both nulls because restricted
        # replacements are defined only where four distinct roles exist.
        null_variable_1 = observed_losses["empty"] - candidate_loss
        null_variable_2 = observed_losses["context"] - context_candidate_loss
        null_variable_rates.append(
            float(
                (
                    null_variable_1[fixed_mask] * null_variable_2[fixed_mask] < 0
                ).mean()
            )
        )
        s1_candidate_loss = loss_for_sets(null_sets["s1_candidate"])
        s2_candidate_loss = loss_for_sets(null_sets["s2_candidate"])
        null_fixed_1 = observed_losses["s1_placebo"] - s1_candidate_loss
        null_fixed_2 = observed_losses["s2_placebo"] - s2_candidate_loss
        null_fixed_rates.append(
            float((null_fixed_1[fixed_mask] * null_fixed_2[fixed_mask] < 0).mean())
        )

    null_fixed = np.asarray(null_fixed_rates)
    observed_fixed_rate = float(fixed_flip.mean())
    empirical_p = float(
        (1 + np.sum(null_fixed >= observed_fixed_rate)) / (1 + len(null_fixed))
    )
    observed_minus_null = float(observed_fixed_rate - null_fixed.mean())
    gate_cfg = config["g1_gate"]
    gate_pass = bool(
        int(fixed_mask.sum()) >= int(gate_cfg["min_fixed_cardinality_queries"])
        and observed_minus_null >= float(gate_cfg["min_observed_minus_null_rate"])
        and empirical_p <= float(gate_cfg["max_empirical_upper_tail_p"])
    )

    eligible_history = validation_counts > 0
    actual_loss = true_class_loss(validation_y, actual_probability)
    zero_loss = true_class_loss(validation_y, zero_probability)
    seed_fixed_rates = []
    for seed, model in zip(config["seeds"], models, strict=True):
        seed_losses = {}
        for name in ("s1_placebo", "s1_candidate", "s2_placebo", "s2_candidate"):
            history, counts = aggregate_custom_sets(validation_current, sets[name])
            features = combine_features(validation_current, history, counts)
            probability = _align_probabilities(
                model, model.predict_proba(features), n_classes
            )
            seed_losses[name] = true_class_loss(validation_y, probability)
        delta_1 = seed_losses["s1_placebo"] - seed_losses["s1_candidate"]
        delta_2 = seed_losses["s2_placebo"] - seed_losses["s2_candidate"]
        seed_fixed_rates.append(
            {
                "seed": int(seed),
                "fixed_cardinality_sign_flip_rate": float(
                    (delta_1[fixed_mask] * delta_2[fixed_mask] < 0).mean()
                ),
            }
        )

    per_query = validation[
        ["key", "group", "dialogue", "speaker", "turn", "emotion"]
    ].copy()
    per_query["history_count"] = validation_counts
    per_query["variable_eligible"] = variable_mask
    per_query["fixed_eligible"] = fixed_mask
    per_query["variable_marginal_empty"] = variable_empty
    per_query["variable_marginal_context"] = variable_context
    per_query["fixed_relative_utility_context_1"] = fixed_1
    per_query["fixed_relative_utility_context_2"] = fixed_2
    per_query["full_history_excess_loss"] = actual_loss - zero_loss

    result = {
        "protocol": config["protocol"],
        "scope": config["scope"],
        "data": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_sha256": {
                "transcription.csv": sha256_file(data_dir / "transcription.csv"),
                "mm_label.npz": sha256_file(data_dir / "mm_label.npz"),
            },
            "validation_history_queries": int(eligible_history.sum()),
            "variable_interaction_queries": int(variable_mask.sum()),
            "fixed_cardinality_queries": int(fixed_mask.sum()),
            "test_opened": False,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "prediction": {
            "within_model_zero": prediction_metrics(
                validation_y, zero_probability, int(config["ece_bins"])
            ),
            "full_history": prediction_metrics(
                validation_y, actual_probability, int(config["ece_bins"])
            ),
            "eligible_harm_rate": float(
                (actual_loss[eligible_history] > zero_loss[eligible_history]).mean()
            ),
            "eligible_mean_excess_loss": float(
                (actual_loss[eligible_history] - zero_loss[eligible_history]).mean()
            ),
        },
        "variable_cardinality_probe": {
            "eligible_queries": int(variable_mask.sum()),
            "observed_sign_flip_rate": float(variable_flip.mean()),
            "restricted_null_on_fixed_subset_mean": float(
                np.mean(null_variable_rates)
            ),
            "status": "descriptive_only_due_to_cardinality_change",
        },
        "fixed_cardinality_replacement_probe": {
            "eligible_queries": int(fixed_mask.sum()),
            "observed_sign_flip": observed_fixed_ci,
            "seed_rates": seed_fixed_rates,
            "speaker_stratified_candidate_resampling": {
                "stratum": "speaker",
                "null_draws": int(config["restricted_null_draws"]),
                "null_mean": float(null_fixed.mean()),
                "null_sd": float(null_fixed.std(ddof=1)),
                "null_q025": float(np.quantile(null_fixed, 0.025)),
                "null_q975": float(np.quantile(null_fixed, 0.975)),
                "observed_minus_null": observed_minus_null,
                "empirical_upper_tail_p": empirical_p,
            },
        },
        "gates": {
            "G1_text_specificity": {
                "pass": gate_pass,
                "thresholds": gate_cfg,
                "interpretation": (
                    "Pass supports text-level history-combination specificity on "
                    "EmotionTalk validation only; it is not multimodal CARMA evidence."
                ),
            }
        },
        "config": config,
        "limitations": [
            "Text transcription only; gated audio/video archives are unavailable locally.",
            "Validation is used for exploratory P1; test corpus payload is sealed.",
            "A linear averaged-history classifier may miss nonlinear human affect interactions.",
            "Restricted resampling preserves speaker but not dialogue, exact candidate marginals, or temporal distance.",
        ],
    }
    return result, per_query


def write_outputs(
    result: dict, per_query: pd.DataFrame, output_json: Path, output_csv: Path
) -> None:
    write_json_atomic(result, output_json.resolve())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(output_csv, index=False, compression="gzip", encoding="utf-8")
