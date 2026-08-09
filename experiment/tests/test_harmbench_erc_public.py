from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hva_affect.harmbench_erc_contract import load_development_protocol  # noqa: E402
from hva_affect.harmbench_erc_inference import (  # noqa: E402
    bootstrap_cell_metrics,
    bootstrap_paired_strategy_contrast,
    make_shared_cluster_bootstrap_plan,
)
from hva_affect.harmbench_erc_public import (  # noqa: E402
    HarmBenchPublicError,
    atomic_write_once,
    build_synthetic_public_report,
    canonical_public_bytes,
    validate_synthetic_public_report,
)
from hva_affect.harmbench_erc_metrics import (  # noqa: E402
    HarmBenchMetricError,
    ensure_finite_public_tree,
)
from run_harmbench_erc_synthetic_contract import synthetic_bundle  # noqa: E402


CONFIG = ROOT / "configs" / "harmbench_erc_v1_draft.json"


@lru_cache(maxsize=1)
def _cached_report_fixture() -> dict:
    protocol = load_development_protocol(CONFIG)
    labels, row_ids, clusters, eligible, current, history = synthetic_bundle()
    plan = make_shared_cluster_bootstrap_plan(
        "synthetic_dialogues",
        row_ids,
        clusters,
        training_seed_count=5,
        replicates=500,
        random_seed=20260810,
    )
    cell = bootstrap_cell_metrics(
        "synthetic_dialogues",
        row_ids,
        clusters,
        labels,
        current,
        history,
        eligible,
        eligible,
        plan,
    )
    contrast = bootstrap_paired_strategy_contrast(
        "synthetic_dialogues",
        row_ids,
        clusters,
        labels,
        current,
        history,
        eligible,
        current,
        eligible,
        eligible,
        plan,
    )
    return build_synthetic_public_report(
        protocol_sha256=protocol.canonical_sha256,
        cell=cell,
        contrast=contrast,
    )


def report_fixture() -> dict:
    return copy.deepcopy(_cached_report_fixture())


def test_exact_synthetic_public_report_validates() -> None:
    report = report_fixture()
    assert validate_synthetic_public_report(report) == report
    assert report["stage_authorization"]["official_test_label_or_outcome_authorized"] is False


def test_unknown_key_private_path_and_nonfinite_value_fail_closed() -> None:
    report = report_fixture()
    report["unknown"] = True
    with pytest.raises(HarmBenchPublicError, match="schema changed"):
        validate_synthetic_public_report(report)
    report = report_fixture()
    report["status"] = "C:\\private\\labels.npy"
    with pytest.raises(HarmBenchPublicError, match="local path"):
        validate_synthetic_public_report(report)
    report = report_fixture()
    report["cell"]["point"]["delta_macro_f1"] = float("nan")
    with pytest.raises(HarmBenchPublicError, match="non-finite"):
        validate_synthetic_public_report(report)


def test_public_report_cannot_authorize_official_test() -> None:
    report = report_fixture()
    report["stage_authorization"]["official_test_feature_or_prediction_authorized"] = True
    with pytest.raises(HarmBenchPublicError, match="authorization changed"):
        validate_synthetic_public_report(report)


def test_cell_and_contrast_must_share_alignment_and_plan() -> None:
    report = report_fixture()
    report["contrast"]["alignment_contract"]["alignment_sha256"] = "0" * 64
    with pytest.raises(HarmBenchPublicError, match="alignment/plan binding"):
        validate_synthetic_public_report(report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["cell"]["inference_contract"].__setitem__("replicates", -1.25),
        lambda report: report["cell"]["inference_contract"].__setitem__("random_seed", 7),
        lambda report: report["cell"]["point"].__setitem__("coverage", 2.0),
    ],
)
def test_inference_controls_ranges_and_ci_order_fail_closed(mutation) -> None:
    report = report_fixture()
    mutation(report)
    with pytest.raises(HarmBenchPublicError):
        validate_synthetic_public_report(report)


def test_bootstrap_ci_order_fails_closed() -> None:
    report = report_fixture()
    report["cell"]["bootstrap"]["coverage"]["ci95_low"] = 0.9
    report["cell"]["bootstrap"]["coverage"]["ci95_high"] = 0.1
    with pytest.raises(HarmBenchPublicError, match="CI order"):
        validate_synthetic_public_report(report)


def test_bootstrap_summary_matches_point_gate_and_replicate_count() -> None:
    report = report_fixture()
    summary = report["cell"]["bootstrap"]["coverage"]
    summary["finite_replicates"] = 1
    summary["finite_fraction"] = 1.0
    with pytest.raises(HarmBenchPublicError, match="finite-fraction"):
        validate_synthetic_public_report(report)

    report = report_fixture()
    summary = report["cell"]["bootstrap"]["coverage"]
    summary["finite_replicates"] = 250
    summary["finite_fraction"] = 0.5
    with pytest.raises(HarmBenchPublicError, match="finite bootstrap gate"):
        validate_synthetic_public_report(report)


def test_wrong_protocol_sha_and_weaker_privacy_policy_fail_closed() -> None:
    report = report_fixture()
    report["protocol_sha256"] = "0" * 64
    with pytest.raises(HarmBenchPublicError, match="not the pinned"):
        validate_synthetic_public_report(report)
    report = report_fixture()
    report["public_artifact_policy"][
        "contains_private_paths_or_outcome_hashes"
    ] = True
    with pytest.raises(HarmBenchPublicError, match="policy changed"):
        validate_synthetic_public_report(report)


@pytest.mark.parametrize(
    "value",
    [
        r"\\server\share\private\labels.npy",
        r"\\?\C:\private\labels.npy",
        "~/private/labels.npy",
        "FILE://C:/private/labels.npy",
        "C:private\\labels.npy",
    ],
)
def test_unc_device_home_and_drive_relative_paths_fail_closed(value: str) -> None:
    with pytest.raises(HarmBenchMetricError, match="local path"):
        ensure_finite_public_tree({"status": value})


def test_atomic_writer_is_write_once(tmp_path: Path) -> None:
    report = report_fixture()
    output = tmp_path / "synthetic.json"
    digest = atomic_write_once(report, output)
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="write-once"):
        atomic_write_once(report, output)


def test_atomic_writer_competing_payloads_has_one_winner(tmp_path: Path) -> None:
    first = report_fixture()
    second = report_fixture()
    second["cell"]["point"]["delta_mean_nll"] += 1e-6
    output = tmp_path / "race.json"

    def write(index: int):
        report = first if index % 2 == 0 else second
        try:
            return ("winner", atomic_write_once(report, output))
        except FileExistsError:
            return ("lost", None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(write, range(16)))
    assert sum(status == "winner" for status, _ in outcomes) == 1
    assert output.read_bytes() in {
        canonical_public_bytes(first),
        canonical_public_bytes(second),
    }


def test_existing_destination_remains_byte_identical(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_bytes(b"preexisting")
    with pytest.raises(FileExistsError, match="write-once"):
        atomic_write_once(report_fixture(), output)
    assert output.read_bytes() == b"preexisting"
