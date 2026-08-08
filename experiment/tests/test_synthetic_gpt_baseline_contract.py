from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.synthetic_gpt_baseline_contract import (  # noqa: E402
    ANALYSIS_ID,
    CANONICAL_FIXTURE_SHA256,
    FROZEN_CONFIG_SHA256,
    PRIVATE_CACHE_SCHEMA,
    PROTOCOL_ID,
    PUBLIC_RECEIPT_SCHEMA,
    REPOSITORY_ROOT as PHYSICAL_REPOSITORY_ROOT,
    SyntheticGPTContractError,
    _request_cache_entry,
    assert_production_authorized,
    canonical_synthetic_fixture,
    run_synthetic_fixture_contract,
    validate_contract_config,
    validate_external_llm_boundary,
    validate_mock_response,
    validate_public_receipt,
    validate_synthetic_fixture,
)


CONFIG_PATH = ROOT / "configs" / "synthetic_gpt_text_baseline_v1.json"
SPLIT_MANIFEST_PATH = ROOT / "configs" / "carma_split_manifest_v1.json"
MODULE_PATH = ROOT / "src" / "hva_affect" / "synthetic_gpt_baseline_contract.py"
CLI_PATH = ROOT / "scripts" / "run_synthetic_gpt_baseline_contract.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(
    tmp_path: Path,
    payload: dict | None = None,
    *,
    name: str = "contract.synthetic.json",
) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload if payload is not None else canonical_synthetic_fixture(), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_strong_key(tmp_path: Path, *, name: str = "private-hmac-key.bin") -> Path:
    path = tmp_path / name
    path.write_bytes(bytes(range(32)))
    return path


def run_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    fixture = write_fixture(tmp_path)
    key = write_strong_key(tmp_path)
    private_cache = tmp_path / "private" / "responses.json"
    public_receipt = tmp_path / "public" / "receipt.json"
    receipt = run_synthetic_fixture_contract(
        config_path=CONFIG_PATH,
        split_manifest_path=SPLIT_MANIFEST_PATH,
        synthetic_fixture_path=fixture,
        hmac_key_path=key,
        private_cache_path=private_cache,
        public_receipt_path=public_receipt,
        explicit_synthetic_acknowledgement=True,
    )
    return private_cache, public_receipt, receipt


def receipt_expectations(private_path: Path) -> dict[str, str]:
    return {
        "expected_config_sha256": sha(CONFIG_PATH),
        "expected_split_manifest_sha256": sha(SPLIT_MANIFEST_PATH),
        "expected_private_cache_sha256": sha(private_path),
    }


def test_repository_root_is_physically_derived_and_not_caller_supplied() -> None:
    assert PHYSICAL_REPOSITORY_ROOT == REPOSITORY_ROOT
    assert "repository_root" not in inspect.signature(run_synthetic_fixture_contract).parameters
    parser_source = CLI_PATH.read_text(encoding="utf-8")
    assert "--repository-root" not in parser_source


def test_frozen_config_validates_all_nonexecution_and_identity_fields() -> None:
    config = load(CONFIG_PATH)
    validate_contract_config(config)
    assert sha(CONFIG_PATH) == FROZEN_CONFIG_SHA256
    assert config["analysis_id"] == ANALYSIS_ID
    assert config["provider_contract"]["model_snapshot"] == (
        "current_alias_unpinned_no_execution"
    )
    assert config["provider_contract"]["api_execution_enabled"] is False
    assert config["provider_contract"]["store"] is False
    assert config["provider_contract"]["tools_enabled"] is False
    assert config["provider_contract"]["batch_enabled"] is False
    assert config["context_contract"]["canonical_fixture_sha256_allowlist"] == [
        CANONICAL_FIXTURE_SHA256
    ]


def test_canonical_fixture_writes_hmac_only_requests_and_strict_public_receipt(
    tmp_path: Path,
) -> None:
    private_path, public_path, receipt = run_fixture(tmp_path)
    private = load(private_path)
    public = load(public_path)
    expected = receipt_expectations(private_path)
    validate_public_receipt(public, **expected)

    assert receipt == public
    assert private["schema_version"] == PRIVATE_CACHE_SCHEMA
    assert private["protocol_id"] == PROTOCOL_ID
    assert private["analysis_id"] == ANALYSIS_ID
    assert public["schema_version"] == PUBLIC_RECEIPT_SCHEMA
    assert private["canonical_fixture_sha256"] == CANONICAL_FIXTURE_SHA256
    assert private["frozen_contract"]["config_sha256"] == sha(CONFIG_PATH)
    assert private["frozen_contract"]["split_manifest_sha256"] == sha(
        SPLIT_MANIFEST_PATH
    )
    assert private["frozen_contract"]["split_protocol_id"] == "scu_set_exploration_v1"

    assert len(private["entries"]) == 2
    for entry in private["entries"]:
        assert set(entry) == {"request", "response"}
        assert set(entry["request"]) == {
            "hmac_sha256",
            "input_character_count",
            "input_utf8_bytes",
            "synthetic",
        }
        assert len(entry["request"]["hmac_sha256"]) == 64
        int(entry["request"]["hmac_sha256"], 16)
        assert entry["request"]["synthetic"] is True

    private_text = private_path.read_text(encoding="utf-8")
    public_text = public_path.read_text(encoding="utf-8")
    for raw_fragment in ("blue lantern", "amber cloud"):
        assert raw_fragment not in private_text
        assert raw_fragment not in public_text
    for entry in private["entries"]:
        assert entry["request"]["hmac_sha256"] not in public_text
    assert "predicted_emotion" not in public_text
    assert public["execution_audit"] == {
        "api_calls": 0,
        "network_calls": 0,
        "openai_sdk_imported": False,
        "real_sidecars_opened": 0,
        "real_dataset_text_rows_opened": 0,
    }


def test_request_hmac_binds_full_config_manifest_and_protocol_identity() -> None:
    config = load(CONFIG_PATH)
    record = validate_synthetic_fixture(canonical_synthetic_fixture())[0]
    common = {
        "record": record,
        "key": bytes(range(32)),
        "config": config,
    }
    baseline = _request_cache_entry(
        **common,
        config_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
    )["request"]["hmac_sha256"]
    config_changed = _request_cache_entry(
        **common,
        config_sha256="c" * 64,
        split_manifest_sha256="b" * 64,
    )["request"]["hmac_sha256"]
    manifest_changed = _request_cache_entry(
        **common,
        config_sha256="a" * 64,
        split_manifest_sha256="d" * 64,
    )["request"]["hmac_sha256"]
    assert len({baseline, config_changed, manifest_changed}) == 3


def test_private_cache_and_public_receipt_are_write_once(tmp_path: Path) -> None:
    private_path, public_path, _ = run_fixture(tmp_path)
    private_before = private_path.read_bytes()
    public_before = public_path.read_bytes()
    with pytest.raises(SyntheticGPTContractError, match="already exists"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=tmp_path / "contract.synthetic.json",
            hmac_key_path=tmp_path / "private-hmac-key.bin",
            private_cache_path=private_path,
            public_receipt_path=public_path,
            explicit_synthetic_acknowledgement=True,
        )
    assert private_path.read_bytes() == private_before
    assert public_path.read_bytes() == public_before


def test_missing_acknowledgement_fails_before_fixture_read(tmp_path: Path) -> None:
    with pytest.raises(SyntheticGPTContractError, match="acknowledgement"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=tmp_path / "does-not-exist.synthetic.json",
            hmac_key_path=tmp_path / "does-not-exist.key",
            private_cache_path=tmp_path / "private.json",
            public_receipt_path=tmp_path / "receipt.json",
            explicit_synthetic_acknowledgement=False,
        )


@pytest.mark.parametrize(
    ("path", "unsafe"),
    [
        (("analysis_id",), "other_analysis"),
        (("split_manifest_contract", "sha256"), "0" * 64),
        (("provider_contract", "model"), "floating-model-alias"),
        (("provider_contract", "model_snapshot"), "gpt-5.6-terra"),
        (("provider_contract", "store"), True),
        (("provider_contract", "batch_enabled"), True),
        (("provider_contract", "tools_enabled"), True),
        (("provider_contract", "reasoning", "context"), "all_turns"),
        (("prompt_contract", "system_prompt"), "changed prompt"),
        (("context_contract", "canonical_fixture_sha256_allowlist"), []),
        (("production_gate", "production_enabled"), True),
        (("production_gate", "external_llm_api_authorization"), "assumed"),
    ],
)
def test_frozen_config_rejects_any_security_or_identity_drift(
    path: tuple[str, ...], unsafe: object
) -> None:
    config = copy.deepcopy(load(CONFIG_PATH))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unsafe
    with pytest.raises(SyntheticGPTContractError):
        validate_contract_config(config)


@pytest.mark.parametrize(
    ("path", "unsafe"),
    [
        (("manifest_id",), "other_manifest"),
        (("schema_version",), "2.0.0"),
        (("status",), "open"),
        (("split_protocol_id",), "other_protocol"),
        (
            ("privacy_boundary", "external_llm_api", "raw_or_row_level_restricted_dataset_content"),
            "allowed",
        ),
        (("privacy_boundary", "external_llm_api", "allowed_payloads"), ["anything"]),
    ],
)
def test_split_manifest_identity_status_protocol_and_privacy_are_strict(
    path: tuple[str, ...], unsafe: object
) -> None:
    manifest = copy.deepcopy(load(SPLIT_MANIFEST_PATH))
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unsafe
    with pytest.raises(SyntheticGPTContractError):
        validate_external_llm_boundary(manifest)


def test_split_manifest_extra_root_field_is_rejected() -> None:
    manifest = copy.deepcopy(load(SPLIT_MANIFEST_PATH))
    manifest["free_text"] = "unsafe"
    with pytest.raises(SyntheticGPTContractError, match="schema changed"):
        validate_external_llm_boundary(manifest)


def test_production_rejected_while_external_llm_authorization_is_absent() -> None:
    with pytest.raises(SyntheticGPTContractError, match="authorization is absent"):
        assert_production_authorized(load(CONFIG_PATH), load(SPLIT_MANIFEST_PATH))


@pytest.mark.parametrize(
    ("unsafe_text", "message"),
    [
        ("[SYNTHETIC] contact invented.person@example.com", "PII"),
        (r"[SYNTHETIC] inspect C:\Users\Person\private.txt", "filesystem path"),
        ("[SYNTHETIC] inspect /home/person/private.txt", "filesystem path"),
        ("[SYNTHETIC] visit https://example.invalid/private", "URL"),
        ("[SYNTHETIC] copied from MELD", "restricted dataset"),
    ],
)
def test_pii_paths_urls_and_dataset_names_are_rejected(
    unsafe_text: str, message: str
) -> None:
    payload = canonical_synthetic_fixture()
    payload["records"][0]["current_text"] = unsafe_text
    with pytest.raises(SyntheticGPTContractError, match=message):
        validate_synthetic_fixture(payload)


def test_arbitrary_prefixed_synthetic_text_is_not_accepted() -> None:
    payload = canonical_synthetic_fixture()
    payload["records"][0]["current_text"] = (
        "[SYNTHETIC] This is arbitrary caller-controlled text with the expected prefix."
    )
    with pytest.raises(SyntheticGPTContractError, match="sole canonical"):
        validate_synthetic_fixture(payload)


def test_missing_marker_extra_dataset_field_and_false_attestation_are_rejected() -> None:
    payload = canonical_synthetic_fixture()
    payload["records"][0]["current_text"] = "not explicitly synthetic"
    with pytest.raises(SyntheticGPTContractError, match="synthetic prefix"):
        validate_synthetic_fixture(payload)

    payload = canonical_synthetic_fixture()
    payload["records"][0]["dataset_id"] = "invented"
    with pytest.raises(SyntheticGPTContractError, match="schema changed"):
        validate_synthetic_fixture(payload)

    payload = canonical_synthetic_fixture()
    payload["attestation"]["contains_real_dataset_content"] = True
    with pytest.raises(SyntheticGPTContractError, match="attestation"):
        validate_synthetic_fixture(payload)


def test_dataset_named_ancestor_path_is_rejected_before_fixture_read(tmp_path: Path) -> None:
    ancestor = tmp_path / "MELD_private_boundary"
    fixture = write_fixture(ancestor)
    key = write_strong_key(tmp_path)
    with pytest.raises(SyntheticGPTContractError, match="path ancestry"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=fixture,
            hmac_key_path=key,
            private_cache_path=tmp_path / "private.json",
            public_receipt_path=tmp_path / "receipt.json",
            explicit_synthetic_acknowledgement=True,
        )


def test_dataset_like_fixture_filename_is_rejected(tmp_path: Path) -> None:
    fixture = write_fixture(tmp_path, name="meld-sidecar.synthetic.json")
    key = write_strong_key(tmp_path)
    with pytest.raises(SyntheticGPTContractError, match="filename resembles"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=fixture,
            hmac_key_path=key,
            private_cache_path=tmp_path / "private.json",
            public_receipt_path=tmp_path / "receipt.json",
            explicit_synthetic_acknowledgement=True,
        )


def test_fixture_and_key_file_size_limits_fail_before_payload_use(tmp_path: Path) -> None:
    oversized_fixture = tmp_path / "oversized.synthetic.json"
    oversized_fixture.write_bytes(b"{" + b" " * (16 * 1024 + 1) + b"}")
    strong_key = write_strong_key(tmp_path)
    with pytest.raises(SyntheticGPTContractError, match="file-size limit"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=oversized_fixture,
            hmac_key_path=strong_key,
            private_cache_path=tmp_path / "private-a.json",
            public_receipt_path=tmp_path / "receipt-a.json",
            explicit_synthetic_acknowledgement=True,
        )

    fixture = write_fixture(tmp_path)
    oversized_key = tmp_path / "oversized-key.bin"
    oversized_key.write_bytes(bytes(range(256)) * 17)
    with pytest.raises(SyntheticGPTContractError, match="key file size"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=fixture,
            hmac_key_path=oversized_key,
            private_cache_path=tmp_path / "private-b.json",
            public_receipt_path=tmp_path / "receipt-b.json",
            explicit_synthetic_acknowledgement=True,
        )


def test_repeated_byte_weak_hmac_key_is_rejected(tmp_path: Path) -> None:
    fixture = write_fixture(tmp_path)
    weak_key = tmp_path / "weak-key.bin"
    weak_key.write_bytes(b"k" * 32)
    with pytest.raises(SyntheticGPTContractError, match="byte diversity"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=fixture,
            hmac_key_path=weak_key,
            private_cache_path=tmp_path / "private.json",
            public_receipt_path=tmp_path / "receipt.json",
            explicit_synthetic_acknowledgement=True,
        )


def test_private_cache_and_key_cannot_be_redirected_into_real_repository(
    tmp_path: Path,
) -> None:
    fixture = write_fixture(tmp_path)
    strong_key = write_strong_key(tmp_path)
    forbidden_cache = REPOSITORY_ROOT / "_synthetic_contract_must_not_write.private.json"
    assert not forbidden_cache.exists()
    with pytest.raises(SyntheticGPTContractError, match="private response cache"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=fixture,
            hmac_key_path=strong_key,
            private_cache_path=forbidden_cache,
            public_receipt_path=tmp_path / "receipt-a.json",
            explicit_synthetic_acknowledgement=True,
        )
    assert not forbidden_cache.exists()

    with pytest.raises(SyntheticGPTContractError, match="HMAC key"):
        run_synthetic_fixture_contract(
            config_path=CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            synthetic_fixture_path=fixture,
            hmac_key_path=CONFIG_PATH,
            private_cache_path=tmp_path / "outside-cache.json",
            public_receipt_path=tmp_path / "receipt-b.json",
            explicit_synthetic_acknowledgement=True,
        )


def test_mock_response_disallows_probability_drift_and_free_text() -> None:
    response = canonical_synthetic_fixture()["records"][0]["mock_response"]
    invalid = copy.deepcopy(response)
    invalid["probabilities"][0] += 0.01
    with pytest.raises(SyntheticGPTContractError, match="sum to one"):
        validate_mock_response(invalid)

    invalid = copy.deepcopy(response)
    invalid["free_text_rationale"] = "not allowed"
    with pytest.raises(SyntheticGPTContractError, match="schema changed"):
        validate_mock_response(invalid)


@pytest.mark.parametrize(
    ("path", "unsafe"),
    [
        (("frozen_contract", "model"), "free-text-model"),
        (("frozen_contract", "model_snapshot"), "pretend-pinned"),
        (("frozen_contract", "split_protocol_id"), "other"),
        (("frozen_contract", "config_sha256"), "not-a-hash"),
        (("artifact_hashes", "private_cache_sha256"), "0" * 64),
        (("aggregate", "synthetic_record_count"), 2.0),
        (("execution_audit", "network_calls"), 1),
        (("public_content_audit", "contains_mock_responses"), True),
    ],
)
def test_public_receipt_rejects_all_nested_drift_and_wrong_types(
    tmp_path: Path, path: tuple[str, ...], unsafe: object
) -> None:
    private_path, _, receipt = run_fixture(tmp_path)
    tampered = copy.deepcopy(receipt)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unsafe
    with pytest.raises(SyntheticGPTContractError):
        validate_public_receipt(tampered, **receipt_expectations(private_path))


def test_public_receipt_rejects_free_text_at_every_schema_boundary(tmp_path: Path) -> None:
    private_path, _, receipt = run_fixture(tmp_path)
    for parent_path in ((), ("frozen_contract",), ("aggregate",), ("artifact_hashes",)):
        tampered = copy.deepcopy(receipt)
        target = tampered
        for key in parent_path:
            target = target[key]
        target["free_text"] = "not allowed"
        with pytest.raises(SyntheticGPTContractError, match="schema changed"):
            validate_public_receipt(tampered, **receipt_expectations(private_path))


def test_module_and_cli_forbid_network_process_dynamic_import_and_causal_escape_hatches() -> None:
    forbidden_import_roots = {
        "asyncio",
        "ctypes",
        "ftplib",
        "multiprocessing",
        "subprocess",
        "socket",
        "ssl",
        "smtplib",
        "telnetlib",
        "webbrowser",
        "http",
        "urllib",
        "requests",
        "httpx",
        "openai",
        "importlib",
        "torch",
        "transformers",
    }
    forbidden_direct_calls = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
    }
    forbidden_os_calls = {"system", "popen", "startfile"}
    for path in (MODULE_PATH, CLI_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                assert "causal" not in node.module
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_direct_calls
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                        assert node.func.attr not in forbidden_os_calls
                        assert not node.func.attr.startswith(("exec", "spawn"))
        assert not (imported & forbidden_import_roots)


def test_cli_rejects_repository_root_override_and_requires_acknowledgement(
    tmp_path: Path,
) -> None:
    fixture = write_fixture(tmp_path)
    key = write_strong_key(tmp_path)
    private = tmp_path / "private.json"
    public = tmp_path / "public.json"
    common = [
        sys.executable,
        str(CLI_PATH),
        "--synthetic-fixture",
        str(fixture),
        "--hmac-key-file",
        str(key),
        "--private-cache",
        str(private),
        "--public-receipt",
        str(public),
    ]
    spoofed = subprocess.run(
        [*common, "--repository-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert spoofed.returncode != 0
    assert "unrecognized arguments" in spoofed.stderr

    rejected = subprocess.run(common, capture_output=True, text=True, check=False)
    assert rejected.returncode != 0
    assert "acknowledgement" in rejected.stderr
    assert not private.exists()
    assert not public.exists()

    accepted = subprocess.run(
        [*common, "--acknowledge-synthetic-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)["execution_audit"]["api_calls"] == 0
    assert private.exists()
    assert public.exists()
