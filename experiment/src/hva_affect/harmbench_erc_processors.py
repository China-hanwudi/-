"""Model-independent, fold-local processors for HarmBench-ERC.

Only a validated fit capability plus its frozen shared cross-fit plan may enter
the fit boundary; training rows are derived internally.  The fitted state is shared by every model
family so that differences between models cannot be attributed to a different
text vocabulary, media scaling, or embedding dimension.

This module intentionally contains no outcome-bearing fit surface and no disk
cache.  Receipts are immutable in-memory values; a later runner may persist
them only through its separately audited artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .harmbench_erc_crossfit import (
    SharedGroupCrossfitPlan,
    resolve_shared_group_crossfit_indices,
    validate_shared_group_crossfit_plan,
)
from .harmbench_erc_open_roles import (
    FitFeatureCapability,
    OutcomeFreeRoleFeatures,
    SelectionFeatureCapability,
    validate_fit_feature_capability,
    validate_outcome_free_role_features,
    validate_selection_feature_capability,
)


class HarmBenchProcessorError(ValueError):
    """Raised when the shared processor contract is violated."""


PROCESSOR_SCHEMA = "harmbench_erc_shared_processor_v1"
PROCESSOR_RECEIPT_SCHEMA = "harmbench_erc_processor_receipt_v1"
OUTPUT_RECEIPT_SCHEMA = "harmbench_erc_processed_role_receipt_v1"
MODALITY_ORDER = ("text", "audio", "video")
SHA256_LENGTH = 64


def _valid_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if (
        len(digest) != SHA256_LENGTH
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise HarmBenchProcessorError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_array_sha256(values: object) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    if array.dtype.kind in {"U", "S", "O"}:
        for value in array.astype(str).reshape(-1):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and not np.little_endian
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _exact_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise HarmBenchProcessorError(f"{name} must be an exact integer")
    result = int(value)
    if result < 0:
        raise HarmBenchProcessorError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class ProcessorSpec:
    """The one exact, model-independent HarmBench processor specification.

    Construction with any non-frozen value fails.  This keeps the class useful
    as a typed receipt payload without turning preprocessing into a model-level
    hyperparameter surface.
    """

    schema_version: str = PROCESSOR_SCHEMA
    text_analyzer: str = "char_wb"
    text_ngram_range: tuple[int, int] = (2, 5)
    text_max_features: int = 50_000
    text_output_dimension: int = 256
    text_svd_algorithm: str = "randomized"
    text_svd_n_iter: int = 7
    numeric_scaler: str = "fold_local_standard_scaler_ddof0"
    numeric_projection: str = "identity_pad_or_seeded_gaussian_1_over_sqrt_d"
    audio_output_dimension: int = 128
    video_output_dimension: int = 128
    row_normalization: str = "zero_safe_l2"
    output_dtype: str = "float32"
    fusion_order: tuple[str, str, str] = MODALITY_ORDER
    canonical_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        observed = {
            "schema_version": self.schema_version,
            "text_analyzer": self.text_analyzer,
            "text_ngram_range": self.text_ngram_range,
            "text_max_features": self.text_max_features,
            "text_output_dimension": self.text_output_dimension,
            "text_svd_algorithm": self.text_svd_algorithm,
            "text_svd_n_iter": self.text_svd_n_iter,
            "numeric_scaler": self.numeric_scaler,
            "numeric_projection": self.numeric_projection,
            "audio_output_dimension": self.audio_output_dimension,
            "video_output_dimension": self.video_output_dimension,
            "row_normalization": self.row_normalization,
            "output_dtype": self.output_dtype,
            "fusion_order": self.fusion_order,
        }
        expected = {
            "schema_version": PROCESSOR_SCHEMA,
            "text_analyzer": "char_wb",
            "text_ngram_range": (2, 5),
            "text_max_features": 50_000,
            "text_output_dimension": 256,
            "text_svd_algorithm": "randomized",
            "text_svd_n_iter": 7,
            "numeric_scaler": "fold_local_standard_scaler_ddof0",
            "numeric_projection": "identity_pad_or_seeded_gaussian_1_over_sqrt_d",
            "audio_output_dimension": 128,
            "video_output_dimension": 128,
            "row_normalization": "zero_safe_l2",
            "output_dtype": "float32",
            "fusion_order": MODALITY_ORDER,
        }
        if observed != expected:
            raise HarmBenchProcessorError("processor specification differs from the frozen exact spec")
        payload = self.canonical_payload()
        object.__setattr__(self, "canonical_sha256", _canonical_json_sha256(payload))

    def canonical_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "text_analyzer": self.text_analyzer,
                "text_ngram_range": list(self.text_ngram_range),
                "text_max_features": self.text_max_features,
                "text_output_dimension": self.text_output_dimension,
                "text_svd_algorithm": self.text_svd_algorithm,
                "text_svd_n_iter": self.text_svd_n_iter,
                "numeric_scaler": self.numeric_scaler,
                "numeric_projection": self.numeric_projection,
                "audio_output_dimension": self.audio_output_dimension,
                "video_output_dimension": self.video_output_dimension,
                "row_normalization": self.row_normalization,
                "output_dtype": self.output_dtype,
                "fusion_order": list(self.fusion_order),
            }
        )


FROZEN_PROCESSOR_SPEC = ProcessorSpec()


@dataclass(frozen=True)
class ProcessorReceipt:
    """Immutable identity of one fold-local fitted processor."""

    schema_version: str
    dataset_id: str
    source_role: str
    seed: int
    fold: int
    source_capability_sha256: str
    cross_role_feature_roster_sha256: str
    crossfit_plan_sha256: str
    train_protocol_row_ids: tuple[int, ...]
    train_protocol_row_ids_sha256: str
    train_group_alignment_sha256: str
    source_row_alignment_sha256: str
    source_content_sha256: str
    processor_spec_sha256: str
    fit_state_sha256: str
    processor_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != PROCESSOR_RECEIPT_SCHEMA:
            raise HarmBenchProcessorError("processor receipt schema changed")
        if not self.dataset_id or not self.source_role:
            raise HarmBenchProcessorError("processor receipt identity is empty")
        seed = _exact_nonnegative_integer(self.seed, name="receipt seed")
        fold = _exact_nonnegative_integer(self.fold, name="receipt fold")
        row_ids = tuple(
            _exact_nonnegative_integer(value, name="training protocol row id")
            for value in self.train_protocol_row_ids
        )
        if not row_ids or len(row_ids) != len(set(row_ids)):
            raise HarmBenchProcessorError("training protocol row ids must be non-empty and unique")
        row_sha = _valid_sha256(
            self.train_protocol_row_ids_sha256,
            name="train_protocol_row_ids_sha256",
        )
        if row_sha != _canonical_array_sha256(np.asarray(row_ids, dtype=np.int64)):
            raise HarmBenchProcessorError("training protocol row id SHA is inconsistent")
        descriptor = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "source_role": self.source_role,
            "seed": seed,
            "fold": fold,
            "source_capability_sha256": _valid_sha256(
                self.source_capability_sha256,
                name="source_capability_sha256",
            ),
            "cross_role_feature_roster_sha256": _valid_sha256(
                self.cross_role_feature_roster_sha256,
                name="cross_role_feature_roster_sha256",
            ),
            "crossfit_plan_sha256": _valid_sha256(
                self.crossfit_plan_sha256,
                name="crossfit_plan_sha256",
            ),
            "train_protocol_row_ids": list(row_ids),
            "train_protocol_row_ids_sha256": row_sha,
            "train_group_alignment_sha256": _valid_sha256(
                self.train_group_alignment_sha256,
                name="train_group_alignment_sha256",
            ),
            "source_row_alignment_sha256": _valid_sha256(
                self.source_row_alignment_sha256,
                name="source_row_alignment_sha256",
            ),
            "source_content_sha256": _valid_sha256(
                self.source_content_sha256, name="source_content_sha256"
            ),
            "processor_spec_sha256": _valid_sha256(
                self.processor_spec_sha256, name="processor_spec_sha256"
            ),
            "fit_state_sha256": _valid_sha256(
                self.fit_state_sha256, name="fit_state_sha256"
            ),
        }
        if _valid_sha256(
            self.processor_receipt_sha256, name="processor_receipt_sha256"
        ) != _canonical_json_sha256(descriptor):
            raise HarmBenchProcessorError("processor receipt SHA is inconsistent")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "train_protocol_row_ids", row_ids)


@dataclass(frozen=True)
class _TextProcessor:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD | None
    output_dimension: int
    effective_dimension: int


@dataclass(frozen=True)
class _NumericProcessor:
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray | None
    input_dimension: int
    output_dimension: int


@dataclass(frozen=True)
class SharedProcessor:
    """One outcome-free fitted processor shared by all downstream families."""

    spec: ProcessorSpec
    receipt: ProcessorReceipt
    _text: _TextProcessor = field(repr=False)
    _audio: _NumericProcessor = field(repr=False)
    _video: _NumericProcessor = field(repr=False)

    @property
    def text_vocabulary(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._text.vectorizer.get_feature_names_out())

    @property
    def text_effective_dimension(self) -> int:
        return self._text.effective_dimension

    @property
    def audio_mean(self) -> np.ndarray:
        return _readonly(self._audio.mean, dtype=np.float64)

    @property
    def audio_scale(self) -> np.ndarray:
        return _readonly(self._audio.scale, dtype=np.float64)

    @property
    def video_mean(self) -> np.ndarray:
        return _readonly(self._video.mean, dtype=np.float64)

    @property
    def video_scale(self) -> np.ndarray:
        return _readonly(self._video.scale, dtype=np.float64)

    def transform(
        self,
        source: FitFeatureCapability | SelectionFeatureCapability,
        *,
        expected_processor_receipt_sha256: str,
        expected_fit_feature_capability_sha256: str,
        expected_transform_source_capability_sha256: str,
        expected_crossfit_plan_sha256: str,
        expected_seed: int,
        expected_fold: int,
    ) -> "ProcessedRoleEmbeddings":
        return transform_role_features(
            self,
            source,
            expected_processor_receipt_sha256=expected_processor_receipt_sha256,
            expected_fit_feature_capability_sha256=(
                expected_fit_feature_capability_sha256
            ),
            expected_transform_source_capability_sha256=(
                expected_transform_source_capability_sha256
            ),
            expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
            expected_seed=expected_seed,
            expected_fold=expected_fold,
        )


@dataclass(frozen=True)
class ProcessedRoleEmbeddings:
    """Immutable, row-aligned processor output for one open role."""

    dataset_id: str
    role: str
    protocol_row_ids: np.ndarray
    text: np.ndarray
    audio: np.ndarray
    video: np.ndarray
    fusion: np.ndarray
    source_capability_sha256: str
    cross_role_feature_roster_sha256: str
    source_row_alignment_sha256: str
    source_content_sha256: str
    processor_receipt_sha256: str
    output_row_alignment_sha256: str
    text_sha256: str
    audio_sha256: str
    video_sha256: str
    fusion_sha256: str
    output_receipt_sha256: str

    @property
    def modality_embeddings(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {"text": self.text, "audio": self.audio, "video": self.video}
        )

    @property
    def rows(self) -> int:
        return len(self.protocol_row_ids)


def _validate_source(source: object) -> OutcomeFreeRoleFeatures:
    if not isinstance(source, OutcomeFreeRoleFeatures):
        raise HarmBenchProcessorError("source must be OutcomeFreeRoleFeatures")
    try:
        source = validate_outcome_free_role_features(source)
    except ValueError as error:
        raise HarmBenchProcessorError(f"source capability content changed: {error}") from error
    rows = source.rows
    if rows < 1:
        raise HarmBenchProcessorError("source is empty")
    if (
        len(source.texts) != rows
        or source.audio.ndim != 2
        or source.video.ndim != 2
        or source.audio.shape[0] != rows
        or source.video.shape[0] != rows
        or source.audio.shape[1] < 1
        or source.video.shape[1] < 1
        or np.asarray(source.protocol_row_ids).shape != (rows,)
    ):
        raise HarmBenchProcessorError("source modalities are not row-aligned matrices")
    if not np.isfinite(source.audio).all() or not np.isfinite(source.video).all():
        raise HarmBenchProcessorError("source modality contains non-finite values")
    if len(set(np.asarray(source.protocol_row_ids, dtype=np.int64).tolist())) != rows:
        raise HarmBenchProcessorError("source protocol row ids are not unique")
    _valid_sha256(source.row_alignment_sha256, name="source.row_alignment_sha256")
    _valid_sha256(source.content_sha256, name="source.content_sha256")
    return source


def _resolve_role_capability(
    value: FitFeatureCapability | SelectionFeatureCapability,
) -> tuple[OutcomeFreeRoleFeatures, str, str]:
    try:
        if isinstance(value, FitFeatureCapability):
            validated = validate_fit_feature_capability(value)
            return (
                validated.fit,
                validated.capability_sha256,
                validated.cross_role_feature_roster_sha256,
            )
        if isinstance(value, SelectionFeatureCapability):
            validated = validate_selection_feature_capability(value)
            return (
                validated.selection,
                validated.capability_sha256,
                validated.cross_role_feature_roster_sha256,
            )
    except ValueError as error:
        raise HarmBenchProcessorError(f"role capability changed: {error}") from error
    raise HarmBenchProcessorError("transform requires a typed fit or selection capability")


def _training_indices(values: Sequence[object], *, rows: int) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or len(raw) == 0:
        raise HarmBenchProcessorError("training indices must be a non-empty vector")
    normalized: list[int] = []
    for value in raw:
        index = _exact_nonnegative_integer(value, name="training index")
        if index >= rows:
            raise HarmBenchProcessorError("training index is outside the source")
        normalized.append(index)
    if len(normalized) != len(set(normalized)):
        raise HarmBenchProcessorError("training indices must be unique")
    return np.asarray(normalized, dtype=np.int64)


def _derived_seed(*, seed: int, fold: int, component: str) -> int:
    encoded = f"{PROCESSOR_SCHEMA}\x1f{seed}\x1f{fold}\x1f{component}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def _fit_text(
    texts: Sequence[str],
    indices: np.ndarray,
    *,
    seed: int,
    fold: int,
    spec: ProcessorSpec,
) -> _TextProcessor:
    vectorizer = TfidfVectorizer(
        analyzer=spec.text_analyzer,
        ngram_range=spec.text_ngram_range,
        lowercase=True,
        min_df=1,
        max_df=1.0,
        max_features=spec.text_max_features,
        binary=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform([str(texts[index]) for index in indices])
    except ValueError as error:
        raise HarmBenchProcessorError(f"training text vocabulary could not be fit: {error}") from error
    rows, features = matrix.shape
    if features < 1:
        raise HarmBenchProcessorError("training text vocabulary is empty")
    if features == 1:
        svd = None
        effective = 1
    else:
        # This upper bound is guaranteed not to exceed the algebraic rank
        # available from a tiny fold; the unused part of 256 is zero padded.
        effective = min(spec.text_output_dimension, rows, features - 1)
        effective = max(1, int(effective))
        svd = TruncatedSVD(
            n_components=effective,
            algorithm=spec.text_svd_algorithm,
            n_iter=spec.text_svd_n_iter,
            random_state=_derived_seed(seed=seed, fold=fold, component="text_svd"),
            tol=0.0,
        )
        try:
            svd.fit(matrix)
        except (TypeError, ValueError, FloatingPointError) as error:
            raise HarmBenchProcessorError(f"training text SVD could not be fit: {error}") from error
    return _TextProcessor(
        vectorizer=vectorizer,
        svd=svd,
        output_dimension=spec.text_output_dimension,
        effective_dimension=effective,
    )


def _fit_numeric(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
    fold: int,
    component: str,
    output_dimension: int,
) -> _NumericProcessor:
    training = np.asarray(values[indices], dtype=np.float64)
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    try:
        scaler.fit(training)
    except (TypeError, ValueError, FloatingPointError) as error:
        raise HarmBenchProcessorError(f"training {component} scaler could not be fit: {error}") from error
    mean = _readonly(scaler.mean_, dtype=np.float64)
    scale = _readonly(scaler.scale_, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise HarmBenchProcessorError(f"training {component} scaler state is invalid")
    input_dimension = int(training.shape[1])
    if input_dimension <= output_dimension:
        projection = None
    else:
        rng = np.random.default_rng(
            _derived_seed(seed=seed, fold=fold, component=f"{component}_projection")
        )
        projection = rng.standard_normal(
            (input_dimension, output_dimension), dtype=np.float64
        ) / np.sqrt(float(output_dimension))
        projection = _readonly(projection, dtype=np.float32)
    return _NumericProcessor(
        mean=mean,
        scale=scale,
        projection=projection,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
    )


def _text_state_payload(processor: _TextProcessor) -> Mapping[str, object]:
    vocabulary = processor.vectorizer.get_feature_names_out().astype(str)
    vectorizer = processor.vectorizer
    tfidf = getattr(vectorizer, "_tfidf", None)
    if tfidf is None:
        raise HarmBenchProcessorError("fitted vectorizer lost its TF-IDF transformer")
    vectorizer_params = {
        "class": f"{type(vectorizer).__module__}.{type(vectorizer).__qualname__}",
        "input": vectorizer.input,
        "encoding": vectorizer.encoding,
        "decode_error": vectorizer.decode_error,
        "strip_accents": vectorizer.strip_accents,
        "lowercase": vectorizer.lowercase,
        "preprocessor_is_none": vectorizer.preprocessor is None,
        "tokenizer_is_none": vectorizer.tokenizer is None,
        "analyzer": vectorizer.analyzer,
        "stop_words_is_none": vectorizer.stop_words is None,
        "token_pattern": vectorizer.token_pattern,
        "ngram_range": list(vectorizer.ngram_range),
        "max_df": vectorizer.max_df,
        "min_df": vectorizer.min_df,
        "max_features": vectorizer.max_features,
        "vocabulary_is_none": vectorizer.vocabulary is None,
        "binary": vectorizer.binary,
        "dtype": str(np.dtype(vectorizer.dtype)),
        "norm": vectorizer.norm,
        "use_idf": vectorizer.use_idf,
        "smooth_idf": vectorizer.smooth_idf,
        "sublinear_tf": vectorizer.sublinear_tf,
        "fixed_vocabulary": bool(getattr(vectorizer, "fixed_vocabulary_", False)),
    }
    tfidf_params = {
        "class": f"{type(tfidf).__module__}.{type(tfidf).__qualname__}",
        "norm": tfidf.norm,
        "use_idf": bool(tfidf.use_idf),
        "smooth_idf": bool(tfidf.smooth_idf),
        "sublinear_tf": bool(tfidf.sublinear_tf),
        "idf_sha256": _canonical_array_sha256(tfidf.idf_),
    }
    if processor.svd is None:
        svd_params: Mapping[str, object] | None = None
    else:
        svd_params = {
            "class": f"{type(processor.svd).__module__}.{type(processor.svd).__qualname__}",
            "n_components": processor.svd.n_components,
            "algorithm": processor.svd.algorithm,
            "n_iter": processor.svd.n_iter,
            "random_state": processor.svd.random_state,
            "tol": processor.svd.tol,
            "n_features_in": int(processor.svd.n_features_in_),
        }
    return {
        "vectorizer_params": vectorizer_params,
        "tfidf_params": tfidf_params,
        "vocabulary_sha256": _canonical_array_sha256(vocabulary),
        "idf_sha256": _canonical_array_sha256(processor.vectorizer.idf_),
        "svd_params": svd_params,
        "svd_components_sha256": (
            None
            if processor.svd is None
            else _canonical_array_sha256(processor.svd.components_)
        ),
        "output_dimension": processor.output_dimension,
        "effective_dimension": processor.effective_dimension,
    }


def _numeric_state_payload(processor: _NumericProcessor) -> Mapping[str, object]:
    return {
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "mean_sha256": _canonical_array_sha256(processor.mean),
        "scale_sha256": _canonical_array_sha256(processor.scale),
        "projection_sha256": (
            None
            if processor.projection is None
            else _canonical_array_sha256(processor.projection)
        ),
        "input_dimension": processor.input_dimension,
        "output_dimension": processor.output_dimension,
    }


def _fit_state_sha256(
    text: _TextProcessor,
    audio: _NumericProcessor,
    video: _NumericProcessor,
) -> str:
    return _canonical_json_sha256(
        {
            "text": dict(_text_state_payload(text)),
            "audio": dict(_numeric_state_payload(audio)),
            "video": dict(_numeric_state_payload(video)),
        }
    )


def validate_shared_processor(
    processor: SharedProcessor,
    *,
    expected_processor_receipt_sha256: str,
    expected_fit_feature_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_seed: int,
    expected_fold: int,
) -> SharedProcessor:
    """Rebuild the receipt and hash all fitted state before every consumer."""

    if not isinstance(processor, SharedProcessor):
        raise HarmBenchProcessorError("processor must be a SharedProcessor")
    if processor.spec != FROZEN_PROCESSOR_SPEC:
        raise HarmBenchProcessorError("processor specification changed")
    receipt = processor.receipt
    try:
        rebuilt = ProcessorReceipt(
            schema_version=receipt.schema_version,
            dataset_id=receipt.dataset_id,
            source_role=receipt.source_role,
            seed=receipt.seed,
            fold=receipt.fold,
            source_capability_sha256=receipt.source_capability_sha256,
            cross_role_feature_roster_sha256=(
                receipt.cross_role_feature_roster_sha256
            ),
            crossfit_plan_sha256=receipt.crossfit_plan_sha256,
            train_protocol_row_ids=receipt.train_protocol_row_ids,
            train_protocol_row_ids_sha256=receipt.train_protocol_row_ids_sha256,
            train_group_alignment_sha256=receipt.train_group_alignment_sha256,
            source_row_alignment_sha256=receipt.source_row_alignment_sha256,
            source_content_sha256=receipt.source_content_sha256,
            processor_spec_sha256=receipt.processor_spec_sha256,
            fit_state_sha256=receipt.fit_state_sha256,
            processor_receipt_sha256=receipt.processor_receipt_sha256,
        )
    except ValueError as error:
        raise HarmBenchProcessorError(f"processor receipt changed: {error}") from error
    if rebuilt.processor_spec_sha256 != processor.spec.canonical_sha256:
        raise HarmBenchProcessorError("processor spec receipt binding changed")
    expected = {
        "processor_receipt_sha256": _valid_sha256(
            expected_processor_receipt_sha256,
            name="expected_processor_receipt_sha256",
        ),
        "source_capability_sha256": _valid_sha256(
            expected_fit_feature_capability_sha256,
            name="expected_fit_feature_capability_sha256",
        ),
        "crossfit_plan_sha256": _valid_sha256(
            expected_crossfit_plan_sha256,
            name="expected_crossfit_plan_sha256",
        ),
        "seed": _exact_nonnegative_integer(expected_seed, name="expected_seed"),
        "fold": _exact_nonnegative_integer(expected_fold, name="expected_fold"),
    }
    if any(getattr(rebuilt, name) != value for name, value in expected.items()):
        raise HarmBenchProcessorError("processor differs from external expected binding")
    live_state = _fit_state_sha256(processor._text, processor._audio, processor._video)
    if live_state != rebuilt.fit_state_sha256:
        raise HarmBenchProcessorError("processor fit state differs from its receipt")
    return processor


def fit_shared_processor(
    fit_capability: FitFeatureCapability,
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    seed: int,
    fold: int,
    spec: ProcessorSpec = FROZEN_PROCESSOR_SPEC,
) -> SharedProcessor:
    """Fit one frozen fold processor using internally derived fit-only rows."""

    try:
        fit_capability = validate_fit_feature_capability(fit_capability)
        validate_shared_group_crossfit_plan(crossfit_plan, fit_capability)
        indices, _ = resolve_shared_group_crossfit_indices(
            crossfit_plan,
            fit_capability,
            training_seed=seed,
            fold=fold,
        )
    except ValueError as error:
        raise HarmBenchProcessorError(f"fit plan/capability changed: {error}") from error
    source = _validate_source(fit_capability.fit)
    if not isinstance(spec, ProcessorSpec) or spec != FROZEN_PROCESSOR_SPEC:
        raise HarmBenchProcessorError("only the frozen processor specification is accepted")
    seed = _exact_nonnegative_integer(seed, name="seed")
    fold = _exact_nonnegative_integer(fold, name="fold")
    text = _fit_text(
        source.texts,
        indices,
        seed=seed,
        fold=fold,
        spec=spec,
    )
    audio = _fit_numeric(
        source.audio,
        indices,
        seed=seed,
        fold=fold,
        component="audio",
        output_dimension=spec.audio_output_dimension,
    )
    video = _fit_numeric(
        source.video,
        indices,
        seed=seed,
        fold=fold,
        component="video",
        output_dimension=spec.video_output_dimension,
    )
    train_row_ids = tuple(int(value) for value in source.protocol_row_ids[indices])
    train_row_sha = _canonical_array_sha256(np.asarray(train_row_ids, dtype=np.int64))
    train_group_sha = _canonical_array_sha256(np.asarray(source.groups)[indices])
    fit_state_sha = _fit_state_sha256(text, audio, video)
    descriptor = {
        "schema_version": PROCESSOR_RECEIPT_SCHEMA,
        "dataset_id": source.dataset_id,
        "source_role": source.role,
        "seed": seed,
        "fold": fold,
        "source_capability_sha256": fit_capability.capability_sha256,
        "cross_role_feature_roster_sha256": (
            fit_capability.cross_role_feature_roster_sha256
        ),
        "crossfit_plan_sha256": crossfit_plan.plan_sha256,
        "train_protocol_row_ids": list(train_row_ids),
        "train_protocol_row_ids_sha256": train_row_sha,
        "train_group_alignment_sha256": train_group_sha,
        "source_row_alignment_sha256": source.row_alignment_sha256,
        "source_content_sha256": source.content_sha256,
        "processor_spec_sha256": spec.canonical_sha256,
        "fit_state_sha256": fit_state_sha,
    }
    receipt = ProcessorReceipt(
        **descriptor,
        processor_receipt_sha256=_canonical_json_sha256(descriptor),
    )
    return SharedProcessor(
        spec=spec,
        receipt=receipt,
        _text=text,
        _audio=audio,
        _video=video,
    )


def _zero_safe_l2(values: np.ndarray) -> np.ndarray:
    dense = np.asarray(values, dtype=np.float64)
    if dense.ndim != 2 or not np.isfinite(dense).all():
        raise HarmBenchProcessorError("embedding matrix is not finite and two-dimensional")
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    result = np.zeros(dense.shape, dtype=np.float32)
    nonzero = norms[:, 0] > 0.0
    if np.any(nonzero):
        result[nonzero] = (dense[nonzero] / norms[nonzero]).astype(np.float32)
    if not np.isfinite(result).all():
        raise HarmBenchProcessorError("L2 normalization produced non-finite values")
    result.setflags(write=False)
    return result


def _transform_text(processor: _TextProcessor, texts: Sequence[str]) -> np.ndarray:
    try:
        matrix = processor.vectorizer.transform(tuple(str(value) for value in texts))
        if processor.svd is None:
            reduced = matrix.toarray()
        else:
            reduced = processor.svd.transform(matrix)
    except (TypeError, ValueError, FloatingPointError) as error:
        raise HarmBenchProcessorError(f"text transform failed: {error}") from error
    if reduced.shape != (len(texts), processor.effective_dimension):
        raise HarmBenchProcessorError("text transform dimension changed")
    padded = np.zeros((len(texts), processor.output_dimension), dtype=np.float32)
    padded[:, : processor.effective_dimension] = np.asarray(reduced, dtype=np.float32)
    return _zero_safe_l2(padded)


def _transform_numeric(processor: _NumericProcessor, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != processor.input_dimension:
        raise HarmBenchProcessorError("numeric modality input dimension changed")
    if not np.isfinite(raw).all():
        raise HarmBenchProcessorError("numeric modality contains non-finite values")
    standardized = (raw - processor.mean) / processor.scale
    if processor.projection is None:
        reduced = standardized
    else:
        reduced = standardized @ np.asarray(processor.projection, dtype=np.float64)
    if reduced.shape[1] > processor.output_dimension:
        raise HarmBenchProcessorError("numeric projection exceeded its output dimension")
    padded = np.zeros((len(raw), processor.output_dimension), dtype=np.float32)
    padded[:, : reduced.shape[1]] = np.asarray(reduced, dtype=np.float32)
    return _zero_safe_l2(padded)


def transform_role_features(
    processor: SharedProcessor,
    source: FitFeatureCapability | SelectionFeatureCapability,
    *,
    expected_processor_receipt_sha256: str,
    expected_fit_feature_capability_sha256: str,
    expected_transform_source_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_seed: int,
    expected_fold: int,
) -> ProcessedRoleEmbeddings:
    """Transform a role without refitting or mutating the fold-local state."""

    processor = validate_shared_processor(
        processor,
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
        expected_fit_feature_capability_sha256=(
            expected_fit_feature_capability_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
        expected_seed=expected_seed,
        expected_fold=expected_fold,
    )
    source, source_capability_sha, roster_sha = _resolve_role_capability(source)
    if source_capability_sha != _valid_sha256(
        expected_transform_source_capability_sha256,
        name="expected_transform_source_capability_sha256",
    ):
        raise HarmBenchProcessorError(
            "transform source capability differs from external expected binding"
        )
    source = _validate_source(source)
    if source.dataset_id != processor.receipt.dataset_id:
        raise HarmBenchProcessorError("transform dataset differs from fit dataset")
    if roster_sha != processor.receipt.cross_role_feature_roster_sha256:
        raise HarmBenchProcessorError("transform cross-role feature roster changed")
    if source.role == processor.receipt.source_role and (
        source_capability_sha != processor.receipt.source_capability_sha256
    ):
        raise HarmBenchProcessorError("fit transform source capability changed")
    before = _fit_state_sha256(processor._text, processor._audio, processor._video)
    if before != processor.receipt.fit_state_sha256:
        raise HarmBenchProcessorError("processor fit state differs from its receipt")
    text = _transform_text(processor._text, source.texts)
    audio = _transform_numeric(processor._audio, source.audio)
    video = _transform_numeric(processor._video, source.video)
    fusion = _zero_safe_l2(np.concatenate((text, audio, video), axis=1))
    after = _fit_state_sha256(processor._text, processor._audio, processor._video)
    if after != before:
        raise HarmBenchProcessorError("transform mutated the fitted processor state")

    row_ids = _readonly(source.protocol_row_ids, dtype=np.int64)
    row_alignment = _canonical_json_sha256(
        {
            "source_row_alignment_sha256": source.row_alignment_sha256,
            "protocol_row_ids_sha256": _canonical_array_sha256(row_ids),
        }
    )
    hashes = {
        "text_sha256": _canonical_array_sha256(text),
        "audio_sha256": _canonical_array_sha256(audio),
        "video_sha256": _canonical_array_sha256(video),
        "fusion_sha256": _canonical_array_sha256(fusion),
    }
    descriptor = {
        "schema_version": OUTPUT_RECEIPT_SCHEMA,
        "dataset_id": source.dataset_id,
        "role": source.role,
        "rows": source.rows,
        "source_capability_sha256": source_capability_sha,
        "cross_role_feature_roster_sha256": roster_sha,
        "source_row_alignment_sha256": source.row_alignment_sha256,
        "source_content_sha256": source.content_sha256,
        "processor_receipt_sha256": processor.receipt.processor_receipt_sha256,
        "output_row_alignment_sha256": row_alignment,
        **hashes,
    }
    return ProcessedRoleEmbeddings(
        dataset_id=source.dataset_id,
        role=source.role,
        protocol_row_ids=row_ids,
        text=text,
        audio=audio,
        video=video,
        fusion=fusion,
        source_capability_sha256=source_capability_sha,
        cross_role_feature_roster_sha256=roster_sha,
        source_row_alignment_sha256=source.row_alignment_sha256,
        source_content_sha256=source.content_sha256,
        processor_receipt_sha256=processor.receipt.processor_receipt_sha256,
        output_row_alignment_sha256=row_alignment,
        **hashes,
        output_receipt_sha256=_canonical_json_sha256(descriptor),
    )


def validate_processed_role_embeddings(
    value: ProcessedRoleEmbeddings,
    *,
    expected_source_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_output_receipt_sha256: str,
) -> ProcessedRoleEmbeddings:
    """Reject mutated or caller-forged processor outputs before model use."""

    if not isinstance(value, ProcessedRoleEmbeddings):
        raise HarmBenchProcessorError("processed role embedding type changed")
    arrays = {
        "protocol_row_ids": np.asarray(value.protocol_row_ids),
        "text": np.asarray(value.text),
        "audio": np.asarray(value.audio),
        "video": np.asarray(value.video),
        "fusion": np.asarray(value.fusion),
    }
    if any(array.flags.writeable for array in arrays.values()):
        raise HarmBenchProcessorError("processed role embedding array is writable")
    if arrays["protocol_row_ids"].dtype != np.dtype(np.int64) or any(
        arrays[name].dtype != np.dtype(np.float32)
        for name in ("text", "audio", "video", "fusion")
    ):
        raise HarmBenchProcessorError("processed role embedding dtype changed")
    rows = len(arrays["protocol_row_ids"])
    expected_shapes = {
        "protocol_row_ids": (rows,),
        "text": (rows, FROZEN_PROCESSOR_SPEC.text_output_dimension),
        "audio": (rows, FROZEN_PROCESSOR_SPEC.audio_output_dimension),
        "video": (rows, FROZEN_PROCESSOR_SPEC.video_output_dimension),
        "fusion": (
            rows,
            FROZEN_PROCESSOR_SPEC.text_output_dimension
            + FROZEN_PROCESSOR_SPEC.audio_output_dimension
            + FROZEN_PROCESSOR_SPEC.video_output_dimension,
        ),
    }
    if rows < 1 or any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise HarmBenchProcessorError("processed role embedding shape changed")
    if any(
        not np.isfinite(arrays[name]).all()
        for name in ("text", "audio", "video", "fusion")
    ):
        raise HarmBenchProcessorError("processed role embedding contains non-finite values")
    if len(set(arrays["protocol_row_ids"].astype(np.int64).tolist())) != rows:
        raise HarmBenchProcessorError("processed protocol row ids are not unique")
    if np.any(arrays["protocol_row_ids"] < 0):
        raise HarmBenchProcessorError("processed protocol row ids must be non-negative")
    expected_fusion = _zero_safe_l2(
        np.concatenate(
            (arrays["text"], arrays["audio"], arrays["video"]), axis=1
        )
    )
    if not np.array_equal(expected_fusion, arrays["fusion"]):
        raise HarmBenchProcessorError("processed fusion is inconsistent with modalities")
    for name in ("text", "audio", "video", "fusion"):
        norms = np.linalg.norm(arrays[name].astype(np.float64), axis=1)
        if not np.all((np.isclose(norms, 0.0)) | np.isclose(norms, 1.0, atol=2e-6)):
            raise HarmBenchProcessorError("processed embedding L2 contract changed")
    source_capability_sha = _valid_sha256(
        value.source_capability_sha256, name="source_capability_sha256"
    )
    if source_capability_sha != _valid_sha256(
        expected_source_capability_sha256,
        name="expected_source_capability_sha256",
    ):
        raise HarmBenchProcessorError("processed source capability differs from expected")
    roster_sha = _valid_sha256(
        value.cross_role_feature_roster_sha256,
        name="cross_role_feature_roster_sha256",
    )
    row_alignment = _canonical_json_sha256(
        {
            "source_row_alignment_sha256": _valid_sha256(
                value.source_row_alignment_sha256,
                name="source_row_alignment_sha256",
            ),
            "protocol_row_ids_sha256": _canonical_array_sha256(
                arrays["protocol_row_ids"]
            ),
        }
    )
    hashes = {
        f"{name}_sha256": _canonical_array_sha256(arrays[name])
        for name in ("text", "audio", "video", "fusion")
    }
    expected_hashes = {
        "text_sha256": value.text_sha256,
        "audio_sha256": value.audio_sha256,
        "video_sha256": value.video_sha256,
        "fusion_sha256": value.fusion_sha256,
    }
    if row_alignment != value.output_row_alignment_sha256 or any(
        _valid_sha256(expected_hashes[name], name=name) != digest
        for name, digest in hashes.items()
    ):
        raise HarmBenchProcessorError("processed role embedding content binding changed")
    descriptor = {
        "schema_version": OUTPUT_RECEIPT_SCHEMA,
        "dataset_id": value.dataset_id,
        "role": value.role,
        "rows": rows,
        "source_capability_sha256": source_capability_sha,
        "cross_role_feature_roster_sha256": roster_sha,
        "source_row_alignment_sha256": value.source_row_alignment_sha256,
        "source_content_sha256": _valid_sha256(
            value.source_content_sha256, name="source_content_sha256"
        ),
        "processor_receipt_sha256": _valid_sha256(
            value.processor_receipt_sha256, name="processor_receipt_sha256"
        ),
        "output_row_alignment_sha256": row_alignment,
        **hashes,
    }
    if _canonical_json_sha256(descriptor) != _valid_sha256(
        value.output_receipt_sha256, name="output_receipt_sha256"
    ):
        raise HarmBenchProcessorError("processed role embedding receipt changed")
    if value.processor_receipt_sha256 != _valid_sha256(
        expected_processor_receipt_sha256,
        name="expected_processor_receipt_sha256",
    ):
        raise HarmBenchProcessorError("processed processor receipt differs from expected")
    if value.output_receipt_sha256 != _valid_sha256(
        expected_output_receipt_sha256,
        name="expected_output_receipt_sha256",
    ):
        raise HarmBenchProcessorError("processed output receipt differs from expected")
    return value


__all__ = [
    "FROZEN_PROCESSOR_SPEC",
    "HarmBenchProcessorError",
    "MODALITY_ORDER",
    "ProcessedRoleEmbeddings",
    "ProcessorReceipt",
    "ProcessorSpec",
    "SharedProcessor",
    "fit_shared_processor",
    "transform_role_features",
    "validate_processed_role_embeddings",
    "validate_shared_processor",
]
