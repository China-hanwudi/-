from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from hva_affect.emotiontalk_role_sidecar import FIT_ROLE  # noqa: E402
from hva_affect.harmbench_erc_crossfit import (  # noqa: E402
    make_shared_group_crossfit_plan,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    make_outcome_free_role_features,
    make_synthetic_fit_feature_capability,
)
from hva_affect.harmbench_erc_processor_cache import (  # noqa: E402
    HarmBenchProcessorCacheError,
    load_shared_processor_cache,
    processor_cache_receipt_file_sha256,
    write_shared_processor_cache,
)
from hva_affect.harmbench_erc_processors import fit_shared_processor  # noqa: E402


def _processor():
    rows = 10
    source = make_outcome_free_role_features(
        dataset_id="synthetic",
        role=FIT_ROLE,
        keys=[f"r{i}" for i in range(rows)],
        texts=[f"emotion text {i}" for i in range(rows)],
        audio=np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        video=np.arange(rows * 4, dtype=np.float32).reshape(rows, 4),
        groups=[f"g{i // 2}" for i in range(rows)],
        speaker_identity=[f"s{i % 2}" for i in range(rows)],
        turn_ids=[i % 2 for i in range(rows)],
        protocol_row_ids=np.arange(rows),
        row_alignment_sha256="a" * 64,
        feature_sha256="b" * 64,
    )
    capability = make_synthetic_fit_feature_capability(
        fit_features=source,
        feature_manifest_sha256="c" * 64,
        synthetic_feature_projection_sha256="d" * 64,
    )
    plan = make_shared_group_crossfit_plan(capability)
    processor = fit_shared_processor(capability, plan, seed=17, fold=0)
    return source, capability, processor


def _write(processor, target: Path):
    return write_shared_processor_cache(
        processor,
        target_directory=target,
        protocol_sha256="c" * 64,
        source_snapshot_sha256="d" * 64,
        source_capability_sha256=processor.receipt.source_capability_sha256,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(
    processor,
    target: Path,
    receipt,
    *,
    expected_cache_receipt_sha256: str | None = None,
    expected_serialized_payload_sha256: str | None = None,
):
    return load_shared_processor_cache(
        target_directory=target,
        expected_protocol_sha256="c" * 64,
        expected_source_snapshot_sha256="d" * 64,
        expected_source_capability_sha256=(
            processor.receipt.source_capability_sha256
        ),
        expected_crossfit_plan_sha256=processor.receipt.crossfit_plan_sha256,
        expected_processor_receipt_sha256=processor.receipt.processor_receipt_sha256,
        expected_cache_receipt_sha256=(
            expected_cache_receipt_sha256
            if expected_cache_receipt_sha256 is not None
            else processor_cache_receipt_file_sha256(receipt)
        ),
        expected_serialized_payload_sha256=(
            expected_serialized_payload_sha256
            if expected_serialized_payload_sha256 is not None
            else receipt.serialized_payload_sha256
        ),
    )


def _transform(processor, capability):
    receipt = processor.receipt
    return processor.transform(
        capability,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_fit_feature_capability_sha256=receipt.source_capability_sha256,
        expected_transform_source_capability_sha256=capability.capability_sha256,
        expected_crossfit_plan_sha256=receipt.crossfit_plan_sha256,
        expected_seed=receipt.seed,
        expected_fold=receipt.fold,
    )


def test_roundtrip_is_write_once_hash_bound_and_equivalent(tmp_path: Path) -> None:
    source, capability, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    restored, observed = _load(processor, target, receipt)
    assert observed == receipt
    assert processor_cache_receipt_file_sha256(receipt) == _file_sha256(
        target / "processor_cache_receipt.json"
    )
    assert restored.receipt == processor.receipt
    np.testing.assert_array_equal(
        _transform(restored, capability).fusion,
        _transform(processor, capability).fusion,
    )
    payload = json.loads((target / "processor_cache_receipt.json").read_text())
    assert set(payload).isdisjoint({"path", "labels", "outcomes"})
    assert payload["contains_labels_or_outcomes"] is False
    with pytest.raises(FileExistsError):
        _write(processor, target)


def test_repository_internal_target_is_rejected(tmp_path: Path) -> None:
    del tmp_path
    _, _, processor = _processor()
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(HarmBenchProcessorCacheError, match="outside repository"):
        _write(processor, repository / "never_create_private_processor_cache")
    assert "repository_root" not in inspect.signature(
        write_shared_processor_cache
    ).parameters
    assert "repository_root" not in inspect.signature(
        load_shared_processor_cache
    ).parameters


def test_payload_tamper_fails_before_joblib_load(tmp_path: Path, monkeypatch) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    with (target / "processor.joblib").open("ab") as handle:
        handle.write(b"tamper")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("joblib.load must not run")

    monkeypatch.setattr("hva_affect.harmbench_erc_processor_cache.joblib.load", forbidden)
    with pytest.raises(HarmBenchProcessorCacheError, match="payload SHA"):
        _load(processor, target, receipt)
    assert called is False


def test_receipt_file_tamper_fails_against_out_of_band_sha(tmp_path: Path) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    receipt_path = target / "processor_cache_receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload["fold"] = 4
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HarmBenchProcessorCacheError, match="receipt file SHA"):
        _load(processor, target, receipt)


def test_coordinated_payload_and_receipt_replacement_fails_before_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    old_receipt_file_sha = processor_cache_receipt_file_sha256(receipt)
    old_payload_sha = receipt.serialized_payload_sha256

    payload_path = target / "processor.joblib"
    joblib.dump(
        {"synthetic_replacement": True},
        payload_path,
        compress=("gzip", 3),
        protocol=5,
    )
    receipt_path = target / "processor_cache_receipt.json"
    replacement = json.loads(receipt_path.read_text(encoding="utf-8"))
    replacement["serialized_payload_sha256"] = _file_sha256(payload_path)
    descriptor = {
        key: value
        for key, value in replacement.items()
        if key != "cache_binding_sha256"
    }
    replacement["cache_binding_sha256"] = hashlib.sha256(
        _canonical_json_bytes(descriptor)
    ).hexdigest()
    receipt_path.write_bytes(_canonical_json_bytes(replacement) + b"\n")
    replacement_receipt_file_sha = _file_sha256(receipt_path)

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("joblib.load must not run")

    monkeypatch.setattr("hva_affect.harmbench_erc_processor_cache.joblib.load", forbidden)
    with pytest.raises(HarmBenchProcessorCacheError, match="receipt file SHA"):
        _load(
            processor,
            target,
            receipt,
            expected_cache_receipt_sha256=old_receipt_file_sha,
            expected_serialized_payload_sha256=old_payload_sha,
        )
    with pytest.raises(HarmBenchProcessorCacheError, match="external expected SHA"):
        _load(
            processor,
            target,
            receipt,
            expected_cache_receipt_sha256=replacement_receipt_file_sha,
            expected_serialized_payload_sha256=old_payload_sha,
        )
    assert called is False


def test_noncanonical_receipt_is_rejected_even_when_file_sha_is_expected(
    tmp_path: Path,
) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    receipt_path = target / "processor_cache_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_path.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    with pytest.raises(HarmBenchProcessorCacheError, match="not canonical"):
        _load(
            processor,
            target,
            receipt,
            expected_cache_receipt_sha256=_file_sha256(receipt_path),
        )


def test_duplicate_receipt_key_is_rejected_even_when_file_sha_is_expected(
    tmp_path: Path,
) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    receipt_path = target / "processor_cache_receipt.json"
    canonical = receipt_path.read_bytes()
    duplicate = canonical[:-2] + b',"seed":0}\n'
    receipt_path.write_bytes(duplicate)
    with pytest.raises(HarmBenchProcessorCacheError, match="duplicate JSON key"):
        _load(
            processor,
            target,
            receipt,
            expected_cache_receipt_sha256=_file_sha256(receipt_path),
        )


def test_out_of_band_expected_hashes_are_required_keyword_arguments(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(load_shared_processor_cache)
    required = (
        "expected_source_capability_sha256",
        "expected_crossfit_plan_sha256",
        "expected_processor_receipt_sha256",
        "expected_cache_receipt_sha256",
        "expected_serialized_payload_sha256",
    )
    for name in required:
        assert signature.parameters[name].default is inspect.Parameter.empty

    arguments = {
        "target_directory": tmp_path / "missing",
        "expected_protocol_sha256": "a" * 64,
        "expected_source_snapshot_sha256": "b" * 64,
        "expected_source_capability_sha256": "c" * 64,
        "expected_crossfit_plan_sha256": "d" * 64,
        "expected_processor_receipt_sha256": "e" * 64,
        "expected_cache_receipt_sha256": "f" * 64,
        "expected_serialized_payload_sha256": "0" * 64,
    }
    for name in required:
        incomplete = dict(arguments)
        del incomplete[name]
        with pytest.raises(TypeError, match=name):
            load_shared_processor_cache(**incomplete)


def test_concurrent_writers_have_exactly_one_winner(tmp_path: Path) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"

    def attempt() -> str:
        try:
            _write(processor, target)
            return "won"
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: attempt(), range(4)))
    assert results.count("won") == 1
    assert results.count("lost") == 3


def test_writer_live_validates_processor_and_source_binding_before_publish(
    tmp_path: Path,
) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "mutated"
    processor._text.vectorizer._tfidf.norm = "l1"
    with pytest.raises(HarmBenchProcessorCacheError, match="live validation"):
        _write(processor, target)
    assert not os.path.lexists(target)

    _, _, clean = _processor()
    with pytest.raises(HarmBenchProcessorCacheError, match="external expected binding"):
        write_shared_processor_cache(
            clean,
            target_directory=tmp_path / "private" / "wrong-source",
            protocol_sha256="c" * 64,
            source_snapshot_sha256="d" * 64,
            source_capability_sha256="f" * 64,
        )


def test_loader_rejects_coherently_resealed_mutated_processor_state(
    tmp_path: Path,
) -> None:
    _, _, processor = _processor()
    target = tmp_path / "private" / "processor"
    receipt = _write(processor, target)
    processor._text.vectorizer._tfidf.norm = "l1"
    payload_path = target / "processor.joblib"
    joblib.dump(processor, payload_path, compress=("gzip", 3), protocol=5)
    payload_sha = _file_sha256(payload_path)
    receipt_path = target / "processor_cache_receipt.json"
    replacement = json.loads(receipt_path.read_text(encoding="utf-8"))
    replacement["serialized_payload_sha256"] = payload_sha
    descriptor = {
        key: value
        for key, value in replacement.items()
        if key != "cache_binding_sha256"
    }
    replacement["cache_binding_sha256"] = hashlib.sha256(
        _canonical_json_bytes(descriptor)
    ).hexdigest()
    receipt_path.write_bytes(_canonical_json_bytes(replacement) + b"\n")
    with pytest.raises(HarmBenchProcessorCacheError, match="live validation"):
        _load(
            processor,
            target,
            receipt,
            expected_cache_receipt_sha256=_file_sha256(receipt_path),
            expected_serialized_payload_sha256=payload_sha,
        )


def test_broken_link_or_reparse_target_is_never_clobbered(tmp_path: Path) -> None:
    _, _, processor = _processor()
    target = tmp_path / "broken-cache-link"
    try:
        target.symlink_to(tmp_path / "missing-cache", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")
    assert os.path.lexists(target)
    with pytest.raises(HarmBenchProcessorCacheError, match="symlink or reparse"):
        _write(processor, target)
    assert target.is_symlink()


def test_public_api_has_no_label_or_outcome_parameter() -> None:
    names = set(inspect.signature(write_shared_processor_cache).parameters)
    names |= set(inspect.signature(load_shared_processor_cache).parameters)
    assert not any("label" in name or "outcome" in name for name in names)
