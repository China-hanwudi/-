from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType

import pytest

from hva_affect.production_source_snapshot_v1 import (
    ProductionSourceSnapshotAttestation,
    ProductionSourceSnapshotError,
    create_production_source_snapshot,
    verify_production_source_snapshot,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, *, detached: bool = True) -> Path:
    root = tmp_path / "repo"
    (root / "experiment" / "scripts").mkdir(parents=True)
    package = root / "experiment" / "src" / "hva_affect"
    nested = package / "future" / "calibration"
    nested.mkdir(parents=True)
    (root / "experiment" / "scripts" / "run_causal_backbone_evidence.py").write_text(
        "from hva_affect import core\n", encoding="utf-8"
    )
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (nested / "__init__.py").write_text("\n", encoding="utf-8")
    (nested / "gate.py").write_text("GATE = 'frozen'\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Snapshot Test")
    _git(root, "config", "user.email", "snapshot@example.invalid")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "freeze")
    if detached:
        _git(root, "checkout", "-q", "--detach", "HEAD")
    return root


def _outside(tmp_path: Path, name: str = "snapshot.json") -> Path:
    output = tmp_path / "private"
    output.mkdir(exist_ok=True)
    return output / name


def _canonical_write(path: Path, payload: dict) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_snapshot_recursively_freezes_relative_paths_and_returns_typed_mapping(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path)
    created = create_production_source_snapshot(
        worktree_root=root, output_path=output
    )

    assert isinstance(created, ProductionSourceSnapshotAttestation)
    assert isinstance(created.code_paths, MappingProxyType)
    assert isinstance(created.code_sha256, MappingProxyType)
    assert "experiment/scripts/run_causal_backbone_evidence.py" in created.code_paths
    nested = "experiment/src/hva_affect/future/calibration/gate.py"
    assert nested in created.code_paths
    assert "gate.py" not in created.code_paths
    assert all("/" in key and "\\" not in key for key in created.code_paths)

    verified = verify_production_source_snapshot(
        manifest_path=output,
        expected_manifest_sha256=created.manifest_sha256,
        worktree_root=root,
    )
    assert verified == created
    assert verified.stable_code_paths() is verified.code_paths


def test_attached_or_dirty_worktree_is_rejected(tmp_path: Path) -> None:
    attached = _repository(tmp_path / "attached", detached=False)
    with pytest.raises(ProductionSourceSnapshotError, match="detached HEAD"):
        create_production_source_snapshot(
            worktree_root=attached,
            output_path=_outside(tmp_path / "attached"),
        )

    dirty = _repository(tmp_path / "dirty")
    (dirty / "experiment" / "src" / "hva_affect" / "core.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(ProductionSourceSnapshotError, match="completely clean"):
        create_production_source_snapshot(
            worktree_root=dirty,
            output_path=_outside(tmp_path / "dirty"),
        )


@pytest.mark.parametrize("hidden_flag", ["--assume-unchanged", "--skip-worktree"])
def test_snapshot_creation_rejects_index_flags_that_hide_a_rewrite(
    tmp_path: Path, hidden_flag: str
) -> None:
    root = _repository(tmp_path)
    source_name = "experiment/src/hva_affect/core.py"
    (root / Path(source_name)).write_text("VALUE = 999\n", encoding="utf-8")
    _git(root, "update-index", hidden_flag, source_name)
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(
        ProductionSourceSnapshotError,
        match="assume-unchanged/skip-worktree",
    ):
        create_production_source_snapshot(
            worktree_root=root,
            output_path=_outside(tmp_path),
        )


@pytest.mark.parametrize(
    ("relative_name", "payload"),
    [
        ("core.pyd", b"unattested extension module"),
        ("__pycache__/core.cpython-311.pyc", b"unattested bytecode"),
    ],
)
def test_ignored_importable_artifact_cannot_shadow_frozen_python_source(
    tmp_path: Path, relative_name: str, payload: bytes
) -> None:
    root = _repository(tmp_path)
    package = root / "experiment" / "src" / "hva_affect"
    artifact = package / Path(relative_name)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    (root / ".git" / "info" / "exclude").write_text(
        f"experiment/src/hva_affect/{relative_name}\n",
        encoding="utf-8",
    )
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(
        ProductionSourceSnapshotError,
        match="non-Python executable/data file",
    ):
        create_production_source_snapshot(
            worktree_root=root,
            output_path=_outside(tmp_path),
        )


@pytest.mark.parametrize(
    "relative_name",
    ["core.pyd", "__pycache__/core.cpython-311.pyc"],
)
def test_verifier_rejects_ignored_importable_artifact_added_after_freeze(
    tmp_path: Path, relative_name: str
) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path)
    created = create_production_source_snapshot(
        worktree_root=root,
        output_path=output,
    )
    artifact = root / "experiment" / "src" / "hva_affect" / Path(relative_name)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"unattested import candidate")
    (root / ".git" / "info" / "exclude").write_text(
        f"experiment/src/hva_affect/{relative_name}\n",
        encoding="utf-8",
    )
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(
        ProductionSourceSnapshotError,
        match="non-Python executable/data file",
    ):
        verify_production_source_snapshot(
            manifest_path=output,
            expected_manifest_sha256=created.manifest_sha256,
            worktree_root=root,
        )


def test_ignored_directory_symlink_or_junction_is_rejected_on_create_and_verify(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path, "valid.json")
    created = create_production_source_snapshot(
        worktree_root=root,
        output_path=output,
    )
    external = tmp_path / "external-package"
    external.mkdir()
    (external / "shadow.py").write_text("SHADOW = True\n", encoding="utf-8")
    link = root / "experiment" / "src" / "hva_affect" / "ignored_link"
    if os.name == "nt":
        linked = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(external)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if linked.returncode != 0:
            pytest.skip(f"cannot create Windows junction: {linked.stderr}")
    else:
        try:
            link.symlink_to(external, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"cannot create directory symlink: {error}")
    (root / ".git" / "info" / "exclude").write_text(
        "experiment/src/hva_affect/ignored_link/\n",
        encoding="utf-8",
    )
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(ProductionSourceSnapshotError, match="symbolic/reparse"):
        create_production_source_snapshot(
            worktree_root=root,
            output_path=_outside(tmp_path, "blocked.json"),
        )
    with pytest.raises(ProductionSourceSnapshotError, match="symbolic/reparse"):
        verify_production_source_snapshot(
            manifest_path=output,
            expected_manifest_sha256=created.manifest_sha256,
            worktree_root=root,
        )


def test_ignored_source_root_sibling_cannot_shadow_dependency_imports(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path, "valid.json")
    created = create_production_source_snapshot(
        worktree_root=root,
        output_path=output,
    )
    sibling = root / "experiment" / "src" / "torch.py"
    sibling.write_text("raise RuntimeError('shadowed torch')\n", encoding="utf-8")
    (root / ".git" / "info" / "exclude").write_text(
        "experiment/src/torch.py\n",
        encoding="utf-8",
    )
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(ProductionSourceSnapshotError, match="source root"):
        create_production_source_snapshot(
            worktree_root=root,
            output_path=_outside(tmp_path, "blocked.json"),
        )
    with pytest.raises(ProductionSourceSnapshotError, match="source root"):
        verify_production_source_snapshot(
            manifest_path=output,
            expected_manifest_sha256=created.manifest_sha256,
            worktree_root=root,
        )


def test_repository_internal_manifest_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    with pytest.raises(ProductionSourceSnapshotError, match="outside"):
        create_production_source_snapshot(
            worktree_root=root,
            output_path=root / "private" / "snapshot.json",
        )


@pytest.mark.parametrize("destination_kind", ["other_repository", "linked_worktree"])
def test_manifest_cannot_be_written_into_any_other_git_worktree(
    tmp_path: Path, destination_kind: str
) -> None:
    root = _repository(tmp_path / "source")
    if destination_kind == "other_repository":
        destination_root = _repository(tmp_path / "destination")
    else:
        destination_root = tmp_path / "linked" / "repo"
        destination_root.parent.mkdir(parents=True)
        _git(root, "worktree", "add", "-q", "--detach", str(destination_root), "HEAD")
    destination_parent = destination_root / "private"
    destination_parent.mkdir()

    with pytest.raises(ProductionSourceSnapshotError, match="every Git repository/worktree"):
        create_production_source_snapshot(
            worktree_root=root,
            output_path=destination_parent / "snapshot.json",
        )


def test_manifest_moved_into_another_repository_is_not_verifiable(tmp_path: Path) -> None:
    root = _repository(tmp_path / "source")
    output = _outside(tmp_path / "source")
    created = create_production_source_snapshot(
        worktree_root=root, output_path=output
    )
    other = _repository(tmp_path / "other")
    copied = other / "copied-snapshot.json"
    copied.write_bytes(output.read_bytes())

    with pytest.raises(ProductionSourceSnapshotError, match="every Git repository/worktree"):
        verify_production_source_snapshot(
            manifest_path=copied,
            expected_manifest_sha256=created.manifest_sha256,
            worktree_root=root,
        )


@pytest.mark.parametrize("attack", ["add", "delete", "rewrite", "hidden_rewrite"])
def test_added_deleted_or_rewritten_module_fails_closed(
    tmp_path: Path, attack: str
) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path)
    created = create_production_source_snapshot(
        worktree_root=root, output_path=output
    )
    package = root / "experiment" / "src" / "hva_affect"
    if attack == "add":
        # Ignored files do not dirty `git status`; the committed/live set check
        # must still reject the newly discoverable Python module.
        (root / ".git" / "info" / "exclude").write_text(
            "experiment/src/hva_affect/ignored_future.py\n", encoding="utf-8"
        )
        (package / "ignored_future.py").write_text("FUTURE = True\n", encoding="utf-8")
        message = "source sets differ"
    elif attack == "delete":
        (package / "core.py").unlink()
        message = "completely clean"
    elif attack == "rewrite":
        (package / "core.py").write_text("VALUE = 999\n", encoding="utf-8")
        message = "completely clean"
    else:
        (package / "core.py").write_text("VALUE = 999\n", encoding="utf-8")
        _git(
            root,
            "update-index",
            "--assume-unchanged",
            "experiment/src/hva_affect/core.py",
        )
        message = "assume-unchanged/skip-worktree"

    with pytest.raises(ProductionSourceSnapshotError, match=message):
        verify_production_source_snapshot(
            manifest_path=output,
            expected_manifest_sha256=created.manifest_sha256,
            worktree_root=root,
        )


@pytest.mark.parametrize("attack", ["new_commit", "new_tree"])
def test_snapshot_binds_exact_git_commit_and_tree(tmp_path: Path, attack: str) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path)
    created = create_production_source_snapshot(
        worktree_root=root, output_path=output
    )
    if attack == "new_tree":
        (root / "future-stage-note.txt").write_text("later stage\n", encoding="utf-8")
        _git(root, "add", "future-stage-note.txt")
        _git(
            root,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "later tree",
        )
    else:
        _git(
            root,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "later commit",
        )

    with pytest.raises(ProductionSourceSnapshotError, match="commit/tree changed"):
        verify_production_source_snapshot(
            manifest_path=output,
            expected_manifest_sha256=created.manifest_sha256,
            worktree_root=root,
        )


@pytest.mark.parametrize("attack", ["omit_nested", "basename_key"])
def test_manifest_cannot_omit_subpackages_or_use_basename_keys(
    tmp_path: Path, attack: str
) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path, "valid.json")
    create_production_source_snapshot(worktree_root=root, output_path=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    nested = "experiment/src/hva_affect/future/calibration/gate.py"
    digest = payload["source"]["files"].pop(nested)
    if attack == "basename_key":
        payload["source"]["files"]["gate.py"] = digest
    else:
        payload["source"]["file_count"] -= 1
    malicious = _outside(tmp_path, f"{attack}.json")
    malicious_sha = _canonical_write(malicious, payload)

    with pytest.raises(ProductionSourceSnapshotError):
        verify_production_source_snapshot(
            manifest_path=malicious,
            expected_manifest_sha256=malicious_sha,
            worktree_root=root,
        )


def test_v2_or_later_stage_cannot_overwrite_or_relabel_v1(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path)
    created = create_production_source_snapshot(
        worktree_root=root, output_path=output
    )
    original = output.read_bytes()
    with pytest.raises(ProductionSourceSnapshotError, match="already exists"):
        create_production_source_snapshot(worktree_root=root, output_path=output)
    assert output.read_bytes() == original

    payload = json.loads(original.decode("utf-8"))
    payload["schema_version"] = "production_source_snapshot_v2"
    v2 = _outside(tmp_path, "v2.json")
    v2_sha = _canonical_write(v2, payload)
    with pytest.raises(ProductionSourceSnapshotError, match="only production_source_snapshot_v1"):
        verify_production_source_snapshot(
            manifest_path=v2,
            expected_manifest_sha256=v2_sha,
            worktree_root=root,
        )
    assert output.read_bytes() == original
    assert hashlib.sha256(original).hexdigest() == created.manifest_sha256


def test_concurrent_writers_publish_exactly_once(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = _outside(tmp_path)

    def attempt() -> object:
        try:
            return create_production_source_snapshot(
                worktree_root=root, output_path=output
            )
        except ProductionSourceSnapshotError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: attempt(), range(2)))
    successes = [
        value
        for value in results
        if isinstance(value, ProductionSourceSnapshotAttestation)
    ]
    failures = [value for value in results if isinstance(value, ProductionSourceSnapshotError)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "already exists" in str(failures[0])
    verify_production_source_snapshot(
        manifest_path=output,
        expected_manifest_sha256=successes[0].manifest_sha256,
        worktree_root=root,
    )
