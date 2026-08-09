"""Frozen multimodal external confirmation protocol for EmotionTalk.

Training and calibration use only train labels. Validation representations and
all decisions are computed before validation labels are opened. Test features
and labels are rejected. The validation command requires a hash-attested bundle
and refuses to overwrite an existing result.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, normalize

from .data_contract import ContractError, write_json_atomic
from .emotiontalk_contract import parse_key
from .emotiontalk_text_p1 import LABEL_NAMES, _vectorizer, build_history_indices
from .meld_text_pilot import (
    _align_probabilities,
    aggregate_custom_sets,
    cluster_bootstrap_binary,
    cluster_bootstrap_difference,
    make_classifier,
    prediction_metrics,
    sha256_file,
    true_class_loss,
)


EXPECTED_ARCHIVE_FIELDS = {
    "keys", "splits", "audio_features", "video_features", "quality",
    "quality_names", "config_sha256",
}


@dataclass
class Blocks:
    current: dict[str, sparse.csr_matrix | np.ndarray]
    history: dict[str, sparse.csr_matrix | np.ndarray]
    quality_current: np.ndarray
    quality_history: np.ndarray
    quality_names: tuple[str, ...]
    counts: np.ndarray


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("sealed_split") != "test_corpus":
        raise ContractError("test split must remain sealed")
    if config.get("primary_base") != "text_audio_video":
        raise ContractError("primary base changed after protocol design")
    return config


def load_media_split(path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], str]:
    if split not in {"train_corpus", "val_corpus"}:
        raise ContractError("EmotionTalk test media are sealed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARCHIVE_FIELDS:
            raise ContractError(f"media feature schema changed: {archive.files}")
        keys = archive["keys"].astype(str)
        splits = archive["splits"].astype(str)
        if np.any(splits == "test_corpus"):
            raise ContractError("sealed test features found in media artifact")
        mask = splits == split
        if not mask.any():
            raise ContractError(f"no rows for {split}")
        selected_keys = keys[mask]
        if len(set(selected_keys)) != len(selected_keys):
            raise ContractError(f"duplicate keys in {split}")
        audio = archive["audio_features"][mask].astype(np.float32, copy=True)
        video = archive["video_features"][mask].astype(np.float32, copy=True)
        quality = archive["quality"][mask].astype(np.float32, copy=True)
        quality_names = tuple(archive["quality_names"].astype(str))
        config_sha = str(archive["config_sha256"])
    if not (np.isfinite(audio).all() and np.isfinite(video).all() and np.isfinite(quality).all()):
        raise ContractError(f"non-finite media features in {split}")
    return selected_keys, audio, video, quality, quality_names, config_sha


def load_unlabeled_frame(data_dir: Path, keys: Sequence[str]) -> pd.DataFrame:
    transcripts = pd.read_csv(data_dir / "transcription.csv", encoding="utf-8-sig")
    if not {"name", "chinese"}.issubset(transcripts.columns):
        raise ContractError("transcription.csv missing name/chinese")
    text_map = {
        Path(str(row["name"]).replace("\\", "/")).stem: str(row["chinese"])
        for _, row in transcripts.iterrows()
    }
    rows = []
    for row_id, key in enumerate(keys):
        if key not in text_map:
            raise ContractError(f"missing transcript for {key}")
        group, dialogue, speaker, turn = parse_key(str(key))
        rows.append({
            "key": str(key), "group": group, "dialogue": dialogue,
            "speaker": speaker, "turn": int(turn), "text": text_map[key],
            "_row_id": row_id,
        })
    return pd.DataFrame(rows)


def load_labels_for_keys(data_dir: Path, split: str, keys: Sequence[str]) -> np.ndarray:
    if split not in {"train_corpus", "val_corpus"}:
        raise ContractError("EmotionTalk test labels are sealed")
    with np.load(data_dir / "mm_label.npz", allow_pickle=True) as archive:
        payload = archive[split]
        if payload.shape != () or payload.dtype != object:
            raise ContractError(f"malformed {split} label payload")
        labels = payload.item()
    if set(labels) != set(keys):
        raise ContractError(f"label/media key mismatch for {split}")
    values = np.asarray([int(labels[str(key)]["emo"]) for key in keys], dtype=np.int64)
    if np.any((values < 0) | (values >= len(LABEL_NAMES))):
        raise ContractError(f"invalid labels in {split}")
    return values


def _aggregate_dense(current: np.ndarray, histories: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols, values = [], [], []
    counts = np.zeros(len(histories), dtype=np.int32)
    for row, selected_raw in enumerate(histories):
        selected = tuple(selected_raw)
        counts[row] = len(selected)
        if selected:
            rows.extend([row] * len(selected))
            cols.extend(selected)
            values.extend([1.0 / len(selected)] * len(selected))
    operator = sparse.csr_matrix((values, (rows, cols)), shape=(len(histories), len(histories)), dtype=np.float32)
    return np.asarray(operator @ current, dtype=np.float32), counts


def _fit_processors(config: dict, frame: pd.DataFrame, audio: np.ndarray, video: np.ndarray, fit_indices: np.ndarray) -> dict:
    vectorizer = _vectorizer(config)
    vectorizer.fit(frame.iloc[fit_indices]["text"].astype(str))
    projection = config["media_projection"]
    result: dict[str, object] = {"vectorizer": vectorizer}
    for name, raw, components in (
        ("audio", audio, int(projection["audio_components"])),
        ("video", video, int(projection["video_components"])),
    ):
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(raw[fit_indices]).astype(np.float32, copy=False)
        pca = PCA(
            n_components=components,
            svd_solver=projection["svd_solver"],
            iterated_power=int(projection["iterated_power"]),
            random_state=int(config["bootstrap_seed"]),
        )
        pca.fit(fit_scaled)
        result[f"{name}_scaler"] = scaler
        result[f"{name}_pca"] = pca
    return result


def _transform_processors(processors: Mapping[str, object], frame: pd.DataFrame, audio: np.ndarray, video: np.ndarray) -> dict[str, sparse.csr_matrix | np.ndarray]:
    transformed: dict[str, sparse.csr_matrix | np.ndarray] = {
        "text": processors["vectorizer"].transform(frame["text"].astype(str)).tocsr()
    }
    for name, raw in (("audio", audio), ("video", video)):
        scaled = processors[f"{name}_scaler"].transform(raw).astype(np.float32, copy=False)
        projected = processors[f"{name}_pca"].transform(scaled).astype(np.float32, copy=False)
        transformed[name] = normalize(projected, norm="l2", axis=1).astype(np.float32, copy=False)
    return transformed


def build_blocks(current: dict[str, sparse.csr_matrix | np.ndarray], quality: np.ndarray, quality_names: Sequence[str], histories: Sequence[Sequence[int]]) -> Blocks:
    text_history, counts = aggregate_custom_sets(current["text"], histories)
    audio_history, audio_counts = _aggregate_dense(np.asarray(current["audio"]), histories)
    video_history, video_counts = _aggregate_dense(np.asarray(current["video"]), histories)
    quality_history, quality_counts = _aggregate_dense(quality, histories)
    if not (np.array_equal(counts, audio_counts) and np.array_equal(counts, video_counts) and np.array_equal(counts, quality_counts)):
        raise RuntimeError("history count mismatch across modalities")
    return Blocks(
        current=current,
        history={"text": text_history, "audio": audio_history, "video": video_history},
        quality_current=quality,
        quality_history=quality_history,
        quality_names=tuple(quality_names),
        counts=counts,
    )


def _slice_blocks(blocks: Blocks, indices: np.ndarray) -> Blocks:
    return Blocks(
        current={name: value[indices] for name, value in blocks.current.items()},
        history={name: value[indices] for name, value in blocks.history.items()},
        quality_current=blocks.quality_current[indices],
        quality_history=blocks.quality_history[indices],
        quality_names=blocks.quality_names,
        counts=blocks.counts[indices],
    )


def base_features(blocks: Blocks, modalities: Sequence[str], *, use_history: bool, donor_indices: np.ndarray | None = None) -> sparse.csr_matrix:
    parts: list[sparse.csr_matrix] = []
    for modality in modalities:
        current = blocks.current[modality]
        parts.append(current if sparse.issparse(current) else sparse.csr_matrix(current))
    for modality in modalities:
        history = blocks.history[modality]
        if not use_history:
            shape = history.shape
            parts.append(sparse.csr_matrix(shape, dtype=history.dtype))
        else:
            selected = history if donor_indices is None else history[donor_indices]
            parts.append(selected if sparse.issparse(selected) else sparse.csr_matrix(selected))
    counts = blocks.counts if use_history else np.zeros_like(blocks.counts)
    parts.append(sparse.csr_matrix(np.log1p(counts).reshape(-1, 1)))
    return sparse.hstack(parts, format="csr")


def _probability_features(full_probability: np.ndarray, current_probability: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    full_entropy = -(full_probability * np.log(np.clip(full_probability, 1e-12, 1))).sum(1)
    current_entropy = -(current_probability * np.log(np.clip(current_probability, 1e-12, 1))).sum(1)
    delta = full_probability - current_probability
    values: list[np.ndarray] = [
        full_probability.max(1), current_probability.max(1), full_entropy, current_entropy,
        np.abs(delta).sum(1), np.sqrt((delta ** 2).sum(1)),
        (full_probability.argmax(1) == current_probability.argmax(1)).astype(float),
    ]
    names = [
        "full_confidence", "current_confidence", "full_entropy", "current_entropy",
        "probability_l1_shift", "probability_l2_shift", "prediction_agreement",
    ]
    for prefix, probability in (("full", full_probability), ("current", current_probability)):
        for class_index in range(probability.shape[1]):
            values.append(probability[:, class_index])
            names.append(f"{prefix}_prob_{class_index}")
    return values, names


def _geometry(current: sparse.csr_matrix | np.ndarray, history: sparse.csr_matrix | np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    if sparse.issparse(current):
        dot = np.asarray(current.multiply(history).sum(1)).ravel()
        current_norm = np.sqrt(np.asarray(current.multiply(current).sum(1)).ravel())
        history_norm = np.sqrt(np.asarray(history.multiply(history).sum(1)).ravel())
    else:
        current_array, history_array = np.asarray(current), np.asarray(history)
        dot = (current_array * history_array).sum(1)
        current_norm = np.linalg.norm(current_array, axis=1)
        history_norm = np.linalg.norm(history_array, axis=1)
    cosine = dot / np.maximum(current_norm * history_norm, 1e-12)
    l2 = np.sqrt(np.maximum(current_norm ** 2 + history_norm ** 2 - 2 * dot, 0))
    return [cosine, l2, history_norm, np.abs(current_norm - history_norm)], ["cosine", "l2", "history_norm", "norm_gap"]


def selector_features(blocks: Blocks, modalities: Sequence[str], full_probability: np.ndarray, current_probability: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[np.ndarray] = [np.log1p(blocks.counts)]
    names = ["log_history_count"]
    prob_values, prob_names = _probability_features(full_probability, current_probability)
    values.extend(prob_values)
    names.extend(prob_names)
    for modality in modalities:
        geometry_values, geometry_names = _geometry(blocks.current[modality], blocks.history[modality])
        values.extend(geometry_values)
        names.extend([f"{modality}_{name}" for name in geometry_names])
    q = {name: index for index, name in enumerate(blocks.quality_names)}
    if "audio" in modalities:
        for field, transform in (
            ("audio_duration_seconds", np.log1p),
            ("audio_source_rate", np.log1p),
            ("audio_source_channels", lambda x: x),
        ):
            values.extend([transform(blocks.quality_current[:, q[field]]), transform(blocks.quality_history[:, q[field]])])
            names.extend([f"current_{field}", f"history_mean_{field}"])
    if "video" in modalities:
        for field, transform in (
            ("video_face_detection_rate", lambda x: x),
            ("video_mean_crop_area_fraction", lambda x: x),
            ("video_total_frames", np.log1p),
        ):
            values.extend([transform(blocks.quality_current[:, q[field]]), transform(blocks.quality_history[:, q[field]])])
            names.extend([f"current_{field}", f"history_mean_{field}"])
    matrix = np.column_stack(values).astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite selector feature")
    return matrix, tuple(names)


def _risk_regressor(config: dict, seed: int, *, loss: str, quantile: float | None = None):
    cfg = config["risk_model"]
    kwargs = dict(
        loss=loss, learning_rate=float(cfg["learning_rate"]), max_iter=int(cfg["max_iter"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]), min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]), random_state=int(seed),
    )
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def _risk_classifier(config: dict, seed: int):
    cfg = config["risk_model"]
    return HistGradientBoostingClassifier(
        learning_rate=float(cfg["learning_rate"]), max_iter=int(cfg["max_iter"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]), min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]), class_weight="balanced",
        random_state=int(seed),
    )


def _fit_selector(config: dict, x: np.ndarray, target: np.ndarray, fit: np.ndarray, calibration: np.ndarray, feature_names: Sequence[str]) -> dict:
    quantile = float(config["risk_quantile"])
    means, quantiles, harms = [], [], []
    for seed in config["risk_seeds"]:
        mean_model = _risk_regressor(config, int(seed), loss="squared_error")
        quantile_model = _risk_regressor(config, int(seed), loss="quantile", quantile=quantile)
        harm_model = _risk_classifier(config, int(seed))
        mean_model.fit(x[fit], target[fit])
        quantile_model.fit(x[fit], target[fit])
        harm_model.fit(x[fit], (target[fit] > 0).astype(int))
        means.append(mean_model)
        quantiles.append(quantile_model)
        harms.append(harm_model)
    calibration_quantile = np.mean([model.predict(x[calibration]) for model in quantiles], axis=0)
    residual = target[calibration] - calibration_quantile
    level = min(1.0, np.ceil((len(residual) + 1) * quantile) / len(residual))
    correction = float(np.quantile(residual, level, method="higher"))
    calibration_upper = calibration_quantile + correction
    thresholds = {
        str(coverage): float(np.quantile(calibration_upper, float(coverage), method="linear"))
        for coverage in config["risk_curve_coverages"]
    }
    return {
        "mean_models": means, "quantile_models": quantiles, "harm_models": harms,
        "conformal_correction": correction, "coverage_thresholds": thresholds,
        "feature_names": tuple(feature_names),
        "calibration_upper_coverage": float((target[calibration] <= calibration_upper).mean()),
        "calibration_rows": int(len(calibration)),
    }


def _predict_selector(selector: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean([model.predict(x) for model in selector["mean_models"]], axis=0)
    quantile = np.mean([model.predict(x) for model in selector["quantile_models"]], axis=0)
    harm = np.mean([model.predict_proba(x)[:, 1] for model in selector["harm_models"]], axis=0)
    return {"mean": mean, "upper": quantile + float(selector["conformal_correction"]), "harm": harm}


def _ensemble_probability(models: Sequence, x: sparse.csr_matrix, n_classes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    per_seed = [_align_probabilities(model, model.predict_proba(x), n_classes) for model in models]
    return np.mean(per_seed, axis=0), per_seed


def train_only(data_dir: Path, feature_path: Path, config_path: Path, bundle_path: Path, summary_path: Path) -> dict:
    config = _read_config(config_path)
    keys, audio, video, quality, quality_names, feature_config_sha = load_media_split(feature_path, "train_corpus")
    frame = load_unlabeled_frame(data_dir, keys)
    y = load_labels_for_keys(data_dir, "train_corpus", keys)
    if not np.array_equal(np.sort(np.unique(y)), np.arange(len(LABEL_NAMES))):
        raise ContractError("train split does not contain all frozen emotion classes")
    n_classes = len(LABEL_NAMES)
    histories = build_history_indices(frame)
    counts = np.asarray([len(value) for value in histories], dtype=np.int32)
    eligible = counts > 0
    groups = (frame["group"].astype(str) + "_" + frame["dialogue"].astype(str)).to_numpy()
    subsets = {name: tuple(value) for name, value in config["modality_subsets"].items()}
    oof_full = {name: np.full((len(frame), n_classes), np.nan) for name in subsets}
    oof_current = {name: np.full((len(frame), n_classes), np.nan) for name in subsets}
    oof_selector_x: dict[str, np.ndarray] = {}
    selector_feature_names: dict[str, tuple[str, ...]] = {}
    splitter = GroupKFold(n_splits=int(config["crossfit_folds"]))
    for fold, (fit_index, held_index) in enumerate(splitter.split(np.zeros(len(frame)), y, groups), start=1):
        processors = _fit_processors(config, frame, audio, video, fit_index)
        current = _transform_processors(processors, frame, audio, video)
        blocks = build_blocks(current, quality, quality_names, histories)
        for subset_name, modalities in subsets.items():
            full_x = base_features(blocks, modalities, use_history=True)
            current_x = base_features(blocks, modalities, use_history=False)
            model = make_classifier(config, int(config["seeds"][0]) + fold)
            model.fit(sparse.vstack([full_x[fit_index], current_x[fit_index]], format="csr"), np.concatenate([y[fit_index], y[fit_index]]))
            oof_full[subset_name][held_index] = _align_probabilities(model, model.predict_proba(full_x[held_index]), n_classes)
            oof_current[subset_name][held_index] = _align_probabilities(model, model.predict_proba(current_x[held_index]), n_classes)
            held_blocks = _slice_blocks(blocks, held_index)
            held_selector_x, feature_names = selector_features(
                held_blocks, modalities,
                oof_full[subset_name][held_index], oof_current[subset_name][held_index],
            )
            if subset_name not in oof_selector_x:
                oof_selector_x[subset_name] = np.full((len(frame), held_selector_x.shape[1]), np.nan)
                selector_feature_names[subset_name] = feature_names
            elif selector_feature_names[subset_name] != feature_names:
                raise RuntimeError(f"selector schema changed across folds for {subset_name}")
            oof_selector_x[subset_name][held_index] = held_selector_x
    if any(np.isnan(value).any() for value in (*oof_full.values(), *oof_current.values(), *oof_selector_x.values())):
        raise RuntimeError("incomplete OOF base predictions")
    primary = str(config["primary_base"])
    target = true_class_loss(y, oof_full[primary]) - true_class_loss(y, oof_current[primary])
    eligible_indices = np.flatnonzero(eligible)
    selector_split = GroupShuffleSplit(
        n_splits=1, test_size=float(config["selector_calibration_fraction"]),
        random_state=int(config["bootstrap_seed"]),
    )
    fit_local, calibration_local = next(selector_split.split(eligible_indices, groups=groups[eligible]))
    selector_fit, selector_calibration = eligible_indices[fit_local], eligible_indices[calibration_local]
    final_processors = _fit_processors(config, frame, audio, video, np.arange(len(frame)))
    final_current = _transform_processors(final_processors, frame, audio, video)
    final_blocks = build_blocks(final_current, quality, quality_names, histories)
    selectors, base_models, selector_train_summary, base_seed_summary = {}, {}, {}, {}
    for subset_name, modalities in subsets.items():
        selector_x = oof_selector_x[subset_name]
        feature_names = selector_feature_names[subset_name]
        selector = _fit_selector(config, selector_x, target, selector_fit, selector_calibration, feature_names)
        selectors[subset_name] = selector
        predicted = _predict_selector(selector, selector_x[selector_calibration])
        truth = target[selector_calibration]
        selector_train_summary[subset_name] = {
            "calibration_harm_auc": float(roc_auc_score((truth > 0).astype(int), predicted["harm"])),
            "calibration_mean_spearman": float(spearmanr(truth, predicted["mean"]).statistic),
            "calibration_upper_coverage": selector["calibration_upper_coverage"],
            "feature_count": len(feature_names),
        }
        full_x = base_features(final_blocks, modalities, use_history=True)
        current_x = base_features(final_blocks, modalities, use_history=False)
        models, seed_metrics = [], []
        for seed in config["seeds"]:
            model = make_classifier(config, int(seed))
            model.fit(sparse.vstack([full_x, current_x], format="csr"), np.concatenate([y, y]))
            models.append(model)
            seed_probability = _align_probabilities(model, model.predict_proba(full_x), n_classes)
            seed_metrics.append({"seed": int(seed), "resubstitution_not_evidence": True, "resubstitution_log_loss": prediction_metrics(y, seed_probability, int(config["ece_bins"]))["log_loss"]})
        base_models[subset_name] = models
        base_seed_summary[subset_name] = seed_metrics
    bundle = {
        "protocol": config["protocol"], "config": config,
        "config_sha256": sha256_file(config_path), "feature_sha256": sha256_file(feature_path),
        "feature_config_sha256": feature_config_sha,
        "transcription_sha256": sha256_file(data_dir / "transcription.csv"),
        "labels_sha256": sha256_file(data_dir / "mm_label.npz"),
        "processors": final_processors, "base_models": base_models, "selectors": selectors,
        "n_classes": n_classes, "quality_names": quality_names,
        "train_rows": int(len(frame)),
        "selector_fit_rows": int(len(selector_fit)), "selector_calibration_rows": int(len(selector_calibration)),
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, bundle_path)
    summary = {
        "protocol": config["protocol"], "stage": "train_only_complete_validation_unopened",
        "data": {"train_rows": len(frame), "eligible_train": int(eligible.sum()), "test_opened": False, "validation_labels_opened": False},
        "hashes": {
            "config_sha256": sha256_file(config_path), "feature_sha256": sha256_file(feature_path),
            "bundle_sha256": sha256_file(bundle_path), "transcription_sha256": bundle["transcription_sha256"],
            "labels_sha256": bundle["labels_sha256"], "feature_config_sha256": feature_config_sha,
        },
        "selector_calibration_diagnostics": selector_train_summary,
        "base_seed_resubstitution_diagnostics_not_evidence": base_seed_summary,
        "next_action": "Freeze code/config/data/bundle hashes, then run validation exactly once.",
    }
    write_json_atomic(summary, summary_path.resolve())
    return summary


def create_freeze_manifest(data_dir: Path, feature_path: Path, config_path: Path, bundle_path: Path, code_paths: Sequence[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"freeze manifest already exists: {output}")
    bundle = joblib.load(bundle_path)
    manifest = {
        "protocol": bundle["protocol"], "validation_authorized": True,
        "hashes": {
            "config": sha256_file(config_path), "features": sha256_file(feature_path),
            "bundle": sha256_file(bundle_path), "transcription": sha256_file(data_dir / "transcription.csv"),
            "labels": sha256_file(data_dir / "mm_label.npz"),
            **{f"code_{index}": sha256_file(path) for index, path in enumerate(code_paths)},
        },
        "paths": {f"code_{index}": str(path.resolve()) for index, path in enumerate(code_paths)},
        "validation_rule": "One run only; output must not pre-exist; no post-validation tuning.",
        "test_policy": "sealed",
    }
    write_json_atomic(manifest, output.resolve())
    return manifest


def verify_freeze_manifest(data_dir: Path, feature_path: Path, config_path: Path, bundle_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("validation_authorized") is not True or manifest.get("test_policy") != "sealed":
        raise ContractError("invalid validation freeze manifest")
    expected = dict(manifest["hashes"])
    observed = {
        "config": sha256_file(config_path), "features": sha256_file(feature_path),
        "bundle": sha256_file(bundle_path), "transcription": sha256_file(data_dir / "transcription.csv"),
        "labels": sha256_file(data_dir / "mm_label.npz"),
        **{name: sha256_file(Path(path)) for name, path in manifest["paths"].items()},
    }
    if expected != observed:
        changed = sorted(name for name in set(expected) | set(observed) if expected.get(name) != observed.get(name))
        raise ContractError(f"freeze hash mismatch: {changed}")
    return manifest


def _count_bin(counts: np.ndarray, boundaries: Sequence[int]) -> np.ndarray:
    return np.digitize(counts, np.asarray(boundaries, dtype=int), right=True)


def restricted_donor_indices(frame: pd.DataFrame, counts: np.ndarray, rng: np.random.Generator, boundaries: Sequence[int]) -> np.ndarray:
    donor = np.arange(len(frame), dtype=int)
    bins = _count_bin(counts, boundaries)
    eligible = np.flatnonzero(counts > 0)
    for query in eligible:
        same_speaker = frame["speaker"].astype(str).to_numpy() == str(frame.iloc[query]["speaker"])
        other_dialogue = (frame["group"].astype(str) + "_" + frame["dialogue"].astype(str)).to_numpy() != f"{frame.iloc[query]['group']}_{frame.iloc[query]['dialogue']}"
        candidates = eligible[same_speaker[eligible] & other_dialogue[eligible] & (bins[eligible] == bins[query])]
        if len(candidates) == 0:
            candidates = eligible[same_speaker[eligible] & other_dialogue[eligible]]
        if len(candidates) == 0:
            candidates = eligible[other_dialogue[eligible] & (bins[eligible] == bins[query])]
        if len(candidates) == 0:
            raise ContractError(f"no negative-control donor for {frame.iloc[query]['key']}")
        donor[query] = int(rng.choice(candidates))
    return donor


def _cluster_bootstrap_mean(values: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> dict:
    unique = np.unique(groups)
    totals = np.asarray([(values[groups == group].sum(), (groups == group).sum()) for group in unique], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        draw = totals[rng.integers(0, len(unique), size=len(unique))].sum(0)
        estimates[index] = draw[0] / draw[1]
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975))}


def _selector_increment_bootstrap(target: np.ndarray, all_harm: np.ndarray, text_harm: np.ndarray, all_mean: np.ndarray, text_mean: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> dict:
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    auc, rho = [], []
    truth = (target > 0).astype(int)
    for _ in range(replicates):
        sampled = np.concatenate([indices[index] for index in rng.integers(0, len(unique), size=len(unique))])
        if len(np.unique(truth[sampled])) < 2:
            continue
        auc.append(roc_auc_score(truth[sampled], all_harm[sampled]) - roc_auc_score(truth[sampled], text_harm[sampled]))
        rho.append(spearmanr(target[sampled], all_mean[sampled]).statistic - spearmanr(target[sampled], text_mean[sampled]).statistic)
    return {
        "auc_increment": float(roc_auc_score(truth, all_harm) - roc_auc_score(truth, text_harm)),
        "auc_increment_ci_low": float(np.quantile(auc, 0.025)), "auc_increment_ci_high": float(np.quantile(auc, 0.975)),
        "spearman_increment": float(spearmanr(target, all_mean).statistic - spearmanr(target, text_mean).statistic),
        "spearman_increment_ci_low": float(np.quantile(rho, 0.025)), "spearman_increment_ci_high": float(np.quantile(rho, 0.975)),
    }


def _subgroup_regret(target: np.ndarray, mask: np.ndarray, groups: np.ndarray, config: dict) -> dict:
    if mask.sum() == 0:
        return {"rows": 0}
    report = {"rows": int(mask.sum()), "harm_rate": float((target[mask] > 0).mean()), "mean_excess": float(target[mask].mean()), "p90_excess": float(np.quantile(target[mask], 0.90)), "p99_excess": float(np.quantile(target[mask], 0.99))}
    if mask.sum() >= 20 and len(np.unique(groups[mask])) >= 3:
        report["cluster_bootstrap"] = cluster_bootstrap_difference(target[mask], groups[mask], int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + int(mask.sum()))
    return report


def validate_once(data_dir: Path, feature_path: Path, config_path: Path, bundle_path: Path, manifest_path: Path, output_json: Path, output_csv: Path) -> dict:
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("validation output already exists; protocol forbids rerun/overwrite")
    verify_freeze_manifest(data_dir, feature_path, config_path, bundle_path, manifest_path)
    config = _read_config(config_path)
    bundle = joblib.load(bundle_path)
    keys, audio, video, quality, quality_names, feature_config_sha = load_media_split(feature_path, "val_corpus")
    if tuple(quality_names) != tuple(bundle["quality_names"]):
        raise ContractError("quality schema changed between train and validation")
    frame = load_unlabeled_frame(data_dir, keys)
    histories = build_history_indices(frame)
    current = _transform_processors(bundle["processors"], frame, audio, video)
    blocks = build_blocks(current, quality, quality_names, histories)
    eligible = blocks.counts > 0
    groups = (frame["group"].astype(str) + "_" + frame["dialogue"].astype(str)).to_numpy()
    subsets = {name: tuple(value) for name, value in config["modality_subsets"].items()}
    probabilities, per_seed_probabilities, selector_predictions = {}, {}, {}
    for subset_name, modalities in subsets.items():
        full_x = base_features(blocks, modalities, use_history=True)
        current_x = base_features(blocks, modalities, use_history=False)
        full_probability, full_per_seed = _ensemble_probability(bundle["base_models"][subset_name], full_x, int(bundle["n_classes"]))
        current_probability, current_per_seed = _ensemble_probability(bundle["base_models"][subset_name], current_x, int(bundle["n_classes"]))
        probabilities[subset_name] = {"full": full_probability, "current": current_probability}
        per_seed_probabilities[subset_name] = {"full": full_per_seed, "current": current_per_seed}
        selector_x, feature_names = selector_features(blocks, modalities, full_probability, current_probability)
        if tuple(feature_names) != tuple(bundle["selectors"][subset_name]["feature_names"]):
            raise ContractError(f"selector feature schema changed for {subset_name}")
        selector_predictions[subset_name] = _predict_selector(bundle["selectors"][subset_name], selector_x)
    # Label-blind negative controls are fully predicted before validation labels are opened.
    rng = np.random.default_rng(int(config["negative_control"]["seed"]))
    primary = str(config["primary_base"])
    permuted_probabilities = []
    for _ in range(int(config["negative_control"]["draws"])):
        donor = restricted_donor_indices(frame, blocks.counts, rng, config["negative_control"]["count_bins"])
        permuted_x = base_features(blocks, subsets[primary], use_history=True, donor_indices=donor)
        permuted_probability, _ = _ensemble_probability(bundle["base_models"][primary], permuted_x, int(bundle["n_classes"]))
        permuted_probabilities.append(permuted_probability)
    # This is the only point at which validation labels are opened.
    y = load_labels_for_keys(data_dir, "val_corpus", keys)
    base_reports, seed_stability = {}, {}
    targets = {}
    for subset_name in subsets:
        full_probability = probabilities[subset_name]["full"]
        current_probability = probabilities[subset_name]["current"]
        full_loss = true_class_loss(y, full_probability)
        current_loss = true_class_loss(y, current_probability)
        target = full_loss - current_loss
        targets[subset_name] = target
        base_reports[subset_name] = {
            "current_only": prediction_metrics(y, current_probability, int(config["ece_bins"])),
            "full_history": prediction_metrics(y, full_probability, int(config["ece_bins"])),
            "eligible_regret": {**cluster_bootstrap_difference(target[eligible], groups[eligible], int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + len(base_reports)), "p90_excess_loss": float(np.quantile(target[eligible], 0.90)), "p99_excess_loss": float(np.quantile(target[eligible], 0.99))},
        }
        seed_stability[subset_name] = []
        for seed, full_seed, current_seed in zip(config["seeds"], per_seed_probabilities[subset_name]["full"], per_seed_probabilities[subset_name]["current"], strict=True):
            seed_target = true_class_loss(y, full_seed) - true_class_loss(y, current_seed)
            seed_stability[subset_name].append({"seed": int(seed), "harm_rate": float((seed_target[eligible] > 0).mean()), "mean_excess_loss": float(seed_target[eligible].mean())})
    target = targets[primary]
    current_probability = probabilities[primary]["current"]
    full_probability = probabilities[primary]["full"]
    current_loss = true_class_loss(y, current_probability)
    selector_reports, policies = {}, {}
    for subset_name, predicted in selector_predictions.items():
        truth = target[eligible]
        harm_truth = (truth > 0).astype(int)
        selector_reports[subset_name] = {
            "harm_auc": float(roc_auc_score(harm_truth, predicted["harm"][eligible])),
            "harm_balanced_accuracy_at_0_5": float(balanced_accuracy_score(harm_truth, (predicted["harm"][eligible] >= 0.5).astype(int))),
            "mean_prediction_spearman": float(spearmanr(truth, predicted["mean"][eligible]).statistic),
            "validation_upper_coverage": float((truth <= predicted["upper"][eligible]).mean()),
            "calibration_upper_coverage": float(bundle["selectors"][subset_name]["calibration_upper_coverage"]),
            "conformal_correction_nats": float(bundle["selectors"][subset_name]["conformal_correction"]),
        }
        def policy(use: np.ndarray) -> dict:
            use = use & eligible
            policy_probability = current_probability.copy()
            policy_probability[use] = full_probability[use]
            difference = true_class_loss(y, policy_probability)[eligible] - current_loss[eligible]
            selected = target[use]
            return {
                "history_coverage": float(use[eligible].mean()),
                "metrics": prediction_metrics(y, policy_probability, int(config["ece_bins"])),
                "population_harm_incidence": float((difference > 0).mean()),
                "harm_rate_among_used": float((selected > 0).mean()) if len(selected) else None,
                "mean_excess_among_used": float(selected.mean()) if len(selected) else None,
                "p90_excess_among_used": float(np.quantile(selected, 0.90)) if len(selected) else None,
                "p99_excess_among_used": float(np.quantile(selected, 0.99)) if len(selected) else None,
                "cluster_bootstrap_policy_minus_current": cluster_bootstrap_difference(difference, groups[eligible], int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + int(10000 * use[eligible].mean()) + len(policies)),
            }
        selector_policies = {"strict_conformal_q90_upper_below_zero": policy(predicted["upper"] < 0)}
        for coverage, threshold in bundle["selectors"][subset_name]["coverage_thresholds"].items():
            item = policy(predicted["upper"] <= float(threshold))
            item["calibration_risk_threshold"] = float(threshold)
            selector_policies[f"calibration_target_{coverage}"] = item
        policies[subset_name] = selector_policies
    increment = _selector_increment_bootstrap(
        target[eligible], selector_predictions[primary]["harm"][eligible], selector_predictions["text"]["harm"][eligible],
        selector_predictions[primary]["mean"][eligible], selector_predictions["text"]["mean"][eligible], groups[eligible],
        int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + 9000,
    )
    permuted_losses = np.stack([true_class_loss(y, probability) for probability in permuted_probabilities])
    pairing_advantage = permuted_losses.mean(0) - true_class_loss(y, full_probability)
    pairing_report = {
        "draws": int(len(permuted_probabilities)),
        "actual_advantage_over_restricted_permutation_nats": _cluster_bootstrap_mean(pairing_advantage[eligible], groups[eligible], int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + 10000),
        "permutation_mean_harm_vs_current": float(((permuted_losses.mean(0) - current_loss)[eligible] > 0).mean()),
        "permutation_mean_excess_vs_current_nats": float((permuted_losses.mean(0) - current_loss)[eligible].mean()),
    }
    q = {name: index for index, name in enumerate(quality_names)}
    face = quality[:, q["video_face_detection_rate"]]
    rate = quality[:, q["audio_source_rate"]]
    channels = quality[:, q["audio_source_channels"]]
    sensitivity = {
        "face_detection": {
            "none": _subgroup_regret(target, eligible & (face == 0), groups, config),
            "partial": _subgroup_regret(target, eligible & (face > 0) & (face < 1), groups, config),
            "all_frames": _subgroup_regret(target, eligible & (face == 1), groups, config),
        },
        "audio_source_rate": {
            "16000": _subgroup_regret(target, eligible & (rate == 16000), groups, config),
            "44100": _subgroup_regret(target, eligible & (rate == 44100), groups, config),
        },
        "audio_channels": {
            "mono": _subgroup_regret(target, eligible & (channels == 1), groups, config),
            "stereo": _subgroup_regret(target, eligible & (channels == 2), groups, config),
        },
    }
    gates = config["gates"]
    natural = base_reports[primary]["eligible_regret"]
    strict = policies[primary]["strict_conformal_q90_upper_below_zero"]
    strict_bootstrap = strict["cluster_bootstrap_policy_minus_current"]
    natural_pass = natural["harm_rate"] >= float(gates["natural_harm"]["min_harm_rate"]) and natural["harm_rate_ci_low"] >= float(gates["natural_harm"]["min_harm_rate_ci_low"])
    selector_pass = (
        selector_reports[primary]["harm_auc"] >= float(gates["multimodal_selector_increment"]["min_all_harm_auc"])
        and increment["auc_increment"] >= float(gates["multimodal_selector_increment"]["min_auc_increment_vs_text"])
        and increment["spearman_increment"] >= float(gates["multimodal_selector_increment"]["min_spearman_increment_vs_text"])
        and increment["auc_increment_ci_low"] >= float(gates["multimodal_selector_increment"]["min_auc_increment_ci_low"])
        and increment["spearman_increment_ci_low"] >= float(gates["multimodal_selector_increment"]["min_spearman_increment_ci_low"])
    )
    strict_harm = strict["harm_rate_among_used"]
    strict_pass = strict["history_coverage"] >= float(gates["strict_q90_fallback"]["min_history_coverage"]) and strict_bootstrap["mean_excess_loss_ci_high"] <= float(gates["strict_q90_fallback"]["max_policy_mean_excess_ci_high"]) and strict_harm is not None and natural["harm_rate"] - strict_harm >= float(gates["strict_q90_fallback"]["min_harm_rate_reduction_among_used"])
    pairing_pass = pairing_report["actual_advantage_over_restricted_permutation_nats"]["ci_low"] > float(gates["pairing_specificity"]["min_actual_advantage_ci_low_nats"])
    gate_results = {
        "natural_harm_replicated": {"pass": bool(natural_pass), "thresholds": gates["natural_harm"]},
        "multimodal_selector_increment": {"pass": bool(selector_pass), "thresholds": gates["multimodal_selector_increment"]},
        "strict_q90_nontrivial_safe_fallback": {"pass": bool(strict_pass), "thresholds": gates["strict_q90_fallback"]},
        "history_pairing_specificity": {"pass": bool(pairing_pass), "thresholds": gates["pairing_specificity"]},
        "carma_method_route_go": {"pass": bool(selector_pass and strict_pass)},
        "negative_transfer_benchmark_route_go": {"pass": bool(natural_pass and pairing_pass)},
    }
    per_query = frame[["key", "group", "dialogue", "speaker", "turn"]].copy()
    per_query["label_id"] = y
    per_query["history_count"] = blocks.counts
    per_query["current_loss"] = current_loss
    per_query["full_history_loss"] = true_class_loss(y, full_probability)
    per_query["excess_loss"] = target
    per_query["permutation_mean_loss"] = permuted_losses.mean(0)
    for subset_name, predicted in selector_predictions.items():
        per_query[f"{subset_name}_predicted_mean"] = predicted["mean"]
        per_query[f"{subset_name}_predicted_upper"] = predicted["upper"]
        per_query[f"{subset_name}_predicted_harm"] = predicted["harm"]
        per_query[f"{subset_name}_strict_use"] = (predicted["upper"] < 0) & eligible
    for name, index in q.items():
        per_query[name] = quality[:, index]
    result = {
        "protocol": config["protocol"], "stage": "single_frozen_validation_complete",
        "data": {"train_rows": int(bundle["train_rows"]), "validation_rows": len(frame), "eligible_validation": int(eligible.sum()), "validation_labels_opened_once": True, "test_opened": False},
        "hashes": {"config": sha256_file(config_path), "features": sha256_file(feature_path), "bundle": sha256_file(bundle_path), "freeze_manifest": sha256_file(manifest_path), "feature_config": feature_config_sha},
        "base_modality_ablation": base_reports,
        "base_seed_stability": seed_stability,
        "selector_metrics": selector_reports,
        "selector_increment_paired_cluster_bootstrap": increment,
        "risk_coverage_policies": policies,
        "restricted_history_permutation_control": pairing_report,
        "quality_sensitivity": sensitivity,
        "gates": gate_results,
        "interpretation_contract": [
            "Engineering success does not imply the CARMA hypothesis passed.",
            "All gates and ablations are reported regardless of direction.",
            "Quality analyses are sensitivity analyses; no quality stratum was silently excluded.",
            "The official validation split is the only external confirmation used here; test remains sealed.",
            "Conformal coverage is marginal over calibration dialogue groups, not a per-speaker guarantee.",
        ],
        "config": config,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, output_json.resolve())
    per_query.to_csv(output_csv, index=False, compression="gzip", encoding="utf-8")
    return result
