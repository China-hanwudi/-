"""Immutable source-snapshot contract for CARMA production evidence.

This module never opens a dataset and never trains a model.  It freezes one
clean detached Git worktree into a repository-external, write-once manifest.
The manifest uses repository-relative POSIX paths, so nested first-party
modules cannot be omitted or collapsed to ambiguous basenames.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = "production_source_snapshot_v1"
SNAPSHOT_STATUS = "frozen_immutable_production_source_snapshot"
REQUIRED_CLI_PATH = "experiment/scripts/run_causal_backbone_evidence.py"
REQUIRED_SOURCE_ROOT = "experiment/src"
REQUIRED_PACKAGE_ROOT = "experiment/src/hva_affect"
SNAPSHOT_CONTRACT: Mapping[str, object] = MappingProxyType(
    {
        "canonical_path_format": "repository_relative_posix",
        "required_cli": REQUIRED_CLI_PATH,
        "source_root_policy": "exactly_one_plain_hva_affect_directory",
        "required_python_tree": f"{REQUIRED_PACKAGE_ROOT}/**/*.py",
        "recursive_python_tree": True,
        "non_python_package_files_forbidden": True,
        "basename_keys_forbidden": True,
        "clean_detached_git_worktree_required": True,
        "repository_external_manifest_required": True,
        "write_policy": "atomic_write_once_no_clobber",
    }
)

_HEX_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ProductionSourceSnapshotError(ValueError):
    """Raised when a production source snapshot is not immutable or complete."""


@dataclass(frozen=True)
class ProductionSourceSnapshotAttestation:
    """Typed, immutable view of one verified v1 source snapshot."""

    manifest_path: Path
    manifest_sha256: str
    worktree_root: Path
    commit_sha: str
    tree_sha: str
    code_sha256: Mapping[str, str]
    code_paths: Mapping[str, Path]

    def stable_code_paths(self) -> Mapping[str, Path]:
        """Return canonical relative-path keys mapped to verified live files."""

        return self.code_paths


@dataclass(frozen=True)
class _WorktreeCapture:
    root: Path
    commit_sha: str
    tree_sha: str
    code_sha256: Mapping[str, str]
    code_paths: Mapping[str, Path]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ProductionSourceSnapshotError(
            f"cannot hash production source {path}: {error}"
        ) from error
    return digest.hexdigest()


def _canonical_manifest_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, field: str) -> str:
    text = str(value).lower()
    if _SHA256.fullmatch(text) is None:
        raise ProductionSourceSnapshotError(f"{field} must be one SHA-256 digest")
    return text


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_repository_external_parent(parent: Path) -> None:
    """Reject a manifest directory owned by any Git repository or worktree."""

    result = _git_process(parent, "rev-parse", "--absolute-git-dir")
    if result.returncode == 0:
        raise ProductionSourceSnapshotError(
            "snapshot manifest must be external to every Git repository/worktree"
        )
    # Git uses 128 for an ordinary directory outside a repository.  Any other
    # result is an indeterminate ownership check and therefore fails closed.
    if result.returncode != 128:
        stderr = str(result.stderr).strip()
        raise ProductionSourceSnapshotError(
            "cannot establish repository-external manifest ownership: "
            f"{stderr or 'unknown git error'}"
        )


def _git_process(root: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
            errors=None if binary else "replace",
        )
    except OSError as error:
        raise ProductionSourceSnapshotError(f"cannot execute git: {error}") from error


def _git_text(root: Path, *args: str) -> str:
    result = _git_process(root, *args)
    if result.returncode != 0:
        stderr = str(result.stderr).strip()
        raise ProductionSourceSnapshotError(
            f"git {' '.join(args)} failed: {stderr or 'unknown git error'}"
        )
    return str(result.stdout).strip()


def _git_object_id(root: Path, expression: str, field: str) -> str:
    value = _git_text(root, "rev-parse", "--verify", expression).lower()
    if _HEX_OBJECT_ID.fullmatch(value) is None:
        raise ProductionSourceSnapshotError(f"git {field} is not a canonical object id")
    return value


def _canonical_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProductionSourceSnapshotError(
            "source keys must be non-empty repository-relative POSIX paths"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) < 2
    ):
        raise ProductionSourceSnapshotError(
            "source keys must be canonical relative paths, never basenames"
        )
    return value


def _is_symbolic_or_reparse(metadata: os.stat_result) -> bool:
    reparse_mask = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_mask)


def _require_closed_source_root(root: Path, source_root: Path, package: Path) -> None:
    """Ensure the sys.path root cannot hide sibling module shadow payloads."""

    try:
        source_metadata = source_root.lstat()
        with os.scandir(source_root) as iterator:
            entries = list(iterator)
    except OSError as error:
        raise ProductionSourceSnapshotError(
            f"cannot inspect production source root: {error}"
        ) from error
    if (
        _is_symbolic_or_reparse(source_metadata)
        or not stat.S_ISDIR(source_metadata.st_mode)
        or not _is_within(source_root.resolve(strict=True), root)
        or len(entries) != 1
        or entries[0].name != package.name
    ):
        raise ProductionSourceSnapshotError(
            "production source root must contain only the plain hva_affect directory"
        )
    try:
        package_metadata = entries[0].stat(follow_symlinks=False)
    except OSError as error:
        raise ProductionSourceSnapshotError(
            f"cannot inspect production package root: {error}"
        ) from error
    if (
        _is_symbolic_or_reparse(package_metadata)
        or not stat.S_ISDIR(package_metadata.st_mode)
        or Path(entries[0].path).resolve(strict=True) != package.resolve(strict=True)
    ):
        raise ProductionSourceSnapshotError(
            "production source root must contain only the plain hva_affect directory"
        )


def _closed_package_python_files(root: Path, package: Path) -> list[Path]:
    """Enumerate a closed regular-file/regular-directory Python package tree."""

    discovered: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda value: value.name)
        except OSError as error:
            raise ProductionSourceSnapshotError(
                f"cannot enumerate production package directory {directory}: {error}"
            ) from error
        for raw_entry in entries:
            entry = Path(raw_entry.path)
            try:
                metadata = raw_entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ProductionSourceSnapshotError(
                    f"cannot inspect production package entry {entry}: {error}"
                ) from error
            if _is_symbolic_or_reparse(metadata):
                raise ProductionSourceSnapshotError(
                    "production package entries must not be symbolic/reparse points"
                )
            try:
                resolved = entry.resolve(strict=True)
            except OSError as error:
                raise ProductionSourceSnapshotError(
                    f"cannot resolve production package entry {entry}: {error}"
                ) from error
            if not _is_within(resolved, root):
                raise ProductionSourceSnapshotError(
                    "production package entry escaped the worktree"
                )
            if stat.S_ISDIR(metadata.st_mode):
                visit(entry)
            elif stat.S_ISREG(metadata.st_mode):
                if entry.suffix != ".py":
                    raise ProductionSourceSnapshotError(
                        "production package contains a non-Python executable/data file"
                    )
                discovered.append(entry)
            else:
                raise ProductionSourceSnapshotError(
                    "production package entries must be regular files/directories"
                )

    try:
        package_metadata = package.lstat()
    except OSError as error:
        raise ProductionSourceSnapshotError(
            f"cannot inspect production package root: {error}"
        ) from error
    if (
        _is_symbolic_or_reparse(package_metadata)
        or not stat.S_ISDIR(package_metadata.st_mode)
        or not _is_within(package.resolve(strict=True), root)
    ):
        raise ProductionSourceSnapshotError(
            "production package root must be a plain in-worktree directory"
        )
    visit(package)
    return sorted(discovered)


def _live_source_paths(root: Path) -> dict[str, Path]:
    cli = root / Path(REQUIRED_CLI_PATH)
    source_root = root / Path(REQUIRED_SOURCE_ROOT)
    package = root / Path(REQUIRED_PACKAGE_ROOT)
    if (
        not cli.is_file()
        or cli.is_symlink()
        or not package.is_dir()
        or package.is_symlink()
    ):
        raise ProductionSourceSnapshotError(
            "required CLI or first-party package tree is missing or symbolic"
        )
    _require_closed_source_root(root, source_root, package)
    # Ignored extension modules and bytecode are executable import candidates.
    # In particular, FileFinder prefers ``.pyd``/``.so`` over a same-named
    # ``.py``, while a forged valid ``__pycache__`` entry can replace source
    # execution.  A clean porcelain status does not report ignored files, so
    # close the package tree explicitly instead of assuming ``rglob('*.py')``
    # is the complete executable source set.
    candidates = [cli, *_closed_package_python_files(root, package)]
    if len(candidates) < 2:
        raise ProductionSourceSnapshotError("production Python tree is empty")
    result: dict[str, Path] = {}
    casefolded: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            raise ProductionSourceSnapshotError(
                "production source entries must be regular non-symbolic files"
            )
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise ProductionSourceSnapshotError(
                "production source escaped the worktree"
            ) from error
        relative = _canonical_relative_path(relative)
        folded = relative.casefold()
        if relative in result or folded in casefolded:
            raise ProductionSourceSnapshotError(
                "production source paths are duplicate or case-ambiguous"
            )
        casefolded.add(folded)
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, root):
            raise ProductionSourceSnapshotError(
                "production source escaped the worktree"
            )
        result[relative] = resolved
    if REQUIRED_CLI_PATH not in result:
        raise ProductionSourceSnapshotError("required production CLI is absent")
    return dict(sorted(result.items()))


def _tracked_source_paths(root: Path) -> set[str]:
    result = _git_process(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        "HEAD",
        "--",
        REQUIRED_CLI_PATH,
        REQUIRED_PACKAGE_ROOT,
        binary=True,
    )
    if result.returncode != 0:
        stderr = bytes(result.stderr).decode("utf-8", errors="replace").strip()
        raise ProductionSourceSnapshotError(
            f"cannot enumerate committed production sources: {stderr}"
        )
    values = bytes(result.stdout).split(b"\0")
    tracked: set[str] = set()
    for raw in values:
        if not raw:
            continue
        try:
            value = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ProductionSourceSnapshotError(
                "committed production path is not UTF-8"
            ) from error
        if value == REQUIRED_CLI_PATH or (
            value.startswith(f"{REQUIRED_PACKAGE_ROOT}/") and value.endswith(".py")
        ):
            tracked.add(_canonical_relative_path(value))
    return tracked


def _require_plain_source_index_entries(root: Path, tracked: set[str]) -> None:
    """Reject index flags that can hide live source changes from Git status.

    ``assume-unchanged`` and ``skip-worktree`` make a modified file look clean
    to the porcelain status used by the snapshot contract.  ``git ls-files -v``
    exposes both states: an ordinary tracked entry is tagged ``H`` while a
    lowercase tag denotes assume-unchanged and ``S`` denotes skip-worktree.
    Production snapshots accept only the ordinary state for every frozen file.
    """

    result = _git_process(
        root,
        "ls-files",
        "-v",
        "-z",
        "--",
        REQUIRED_CLI_PATH,
        REQUIRED_PACKAGE_ROOT,
        binary=True,
    )
    if result.returncode != 0:
        stderr = bytes(result.stderr).decode("utf-8", errors="replace").strip()
        raise ProductionSourceSnapshotError(
            f"cannot inspect production source index flags: {stderr}"
        )
    observed: dict[str, str] = {}
    casefolded: set[str] = set()
    for raw in bytes(result.stdout).split(b"\0"):
        if not raw:
            continue
        if len(raw) < 3 or raw[1:2] != b" ":
            raise ProductionSourceSnapshotError(
                "production source index status is malformed"
            )
        try:
            tag = raw[:1].decode("ascii", errors="strict")
            value = raw[2:].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ProductionSourceSnapshotError(
                "production source index status is not UTF-8"
            ) from error
        if value != REQUIRED_CLI_PATH and not (
            value.startswith(f"{REQUIRED_PACKAGE_ROOT}/") and value.endswith(".py")
        ):
            continue
        name = _canonical_relative_path(value)
        if name in observed or name.casefold() in casefolded:
            raise ProductionSourceSnapshotError(
                "production source index paths are duplicate or case-ambiguous"
            )
        casefolded.add(name.casefold())
        observed[name] = tag
    if set(observed) != tracked:
        raise ProductionSourceSnapshotError(
            "production source HEAD/index sets differ"
        )
    hidden = sorted(name for name, tag in observed.items() if tag != "H")
    if hidden:
        raise ProductionSourceSnapshotError(
            "production source uses assume-unchanged/skip-worktree index flags: "
            f"{hidden}"
        )


def _capture_clean_detached_worktree(worktree_root: str | Path) -> _WorktreeCapture:
    root = Path(worktree_root).resolve(strict=True)
    if not root.is_dir():
        raise ProductionSourceSnapshotError("worktree root must be a directory")
    top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root:
        raise ProductionSourceSnapshotError(
            "worktree_root must be the exact Git worktree top level"
        )
    symbolic = _git_process(root, "symbolic-ref", "-q", "HEAD")
    if symbolic.returncode == 0:
        raise ProductionSourceSnapshotError("production worktree must use detached HEAD")
    if symbolic.returncode != 1:
        raise ProductionSourceSnapshotError("cannot determine detached HEAD state")
    status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProductionSourceSnapshotError("production worktree must be completely clean")
    commit_sha = _git_object_id(root, "HEAD^{commit}", "commit")
    tree_sha = _git_object_id(root, "HEAD^{tree}", "tree")
    paths = _live_source_paths(root)
    tracked = _tracked_source_paths(root)
    _require_plain_source_index_entries(root, tracked)
    if set(paths) != tracked:
        missing_live = sorted(tracked - set(paths))
        uncommitted_live = sorted(set(paths) - tracked)
        raise ProductionSourceSnapshotError(
            "live and committed production source sets differ: "
            f"missing_live={missing_live}, uncommitted_live={uncommitted_live}"
        )
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    _require_plain_source_index_entries(root, tracked)
    if _git_text(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ProductionSourceSnapshotError("production worktree changed while hashing")
    if (
        _git_object_id(root, "HEAD^{commit}", "commit") != commit_sha
        or _git_object_id(root, "HEAD^{tree}", "tree") != tree_sha
        or _live_source_paths(root) != paths
    ):
        raise ProductionSourceSnapshotError("production worktree changed while capturing")
    return _WorktreeCapture(
        root=root,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        code_sha256=MappingProxyType(dict(sorted(hashes.items()))),
        code_paths=MappingProxyType(dict(sorted(paths.items()))),
    )


def _manifest_payload(capture: _WorktreeCapture) -> dict[str, object]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": SNAPSHOT_STATUS,
        "contract": dict(SNAPSHOT_CONTRACT),
        "git": {
            "commit_sha": capture.commit_sha,
            "tree_sha": capture.tree_sha,
            "detached_head": True,
            "clean_worktree": True,
        },
        "source": {
            "file_count": len(capture.code_sha256),
            "files": dict(capture.code_sha256),
        },
    }


def _external_destination(output_path: str | Path, root: Path) -> Path:
    raw = Path(output_path)
    lexical = raw.absolute()
    if _is_within(lexical, root):
        raise ProductionSourceSnapshotError(
            "snapshot manifest must be written outside the repository worktree"
        )
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise ProductionSourceSnapshotError(
            f"snapshot manifest parent is unavailable: {error}"
        ) from error
    if not parent.is_dir():
        raise ProductionSourceSnapshotError("snapshot manifest parent must be a directory")
    _require_repository_external_parent(parent)
    destination = parent / raw.name
    if not raw.name or raw.name in {".", ".."}:
        raise ProductionSourceSnapshotError("snapshot manifest requires a file name")
    if _is_within(destination, root):
        raise ProductionSourceSnapshotError(
            "snapshot manifest must be written outside the repository worktree"
        )
    return destination


def _atomic_write_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProductionSourceSnapshotError("snapshot manifest already exists")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ProductionSourceSnapshotError(
                "snapshot manifest already exists"
            ) from error
        except OSError as error:
            if path.exists() or path.is_symlink():
                raise ProductionSourceSnapshotError(
                    "snapshot manifest already exists"
                ) from error
            raise ProductionSourceSnapshotError(
                f"cannot atomically publish snapshot manifest: {error}"
            ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _validated_manifest_payload(raw: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionSourceSnapshotError(
            f"cannot decode source snapshot manifest: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ProductionSourceSnapshotError("snapshot manifest root must be an object")
    if raw != _canonical_manifest_bytes(payload):
        raise ProductionSourceSnapshotError("snapshot manifest is not canonical JSON")
    if set(payload) != {"schema_version", "status", "contract", "git", "source"}:
        raise ProductionSourceSnapshotError("snapshot manifest root schema changed")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ProductionSourceSnapshotError("only production_source_snapshot_v1 is valid")
    if payload.get("status") != SNAPSHOT_STATUS:
        raise ProductionSourceSnapshotError("snapshot manifest status changed")
    if payload.get("contract") != dict(SNAPSHOT_CONTRACT):
        raise ProductionSourceSnapshotError("snapshot manifest contract changed")
    git = payload.get("git")
    source = payload.get("source")
    if not isinstance(git, dict) or set(git) != {
        "commit_sha",
        "tree_sha",
        "detached_head",
        "clean_worktree",
    }:
        raise ProductionSourceSnapshotError("snapshot Git schema changed")
    if git.get("detached_head") is not True or git.get("clean_worktree") is not True:
        raise ProductionSourceSnapshotError("snapshot does not attest clean detached HEAD")
    for name in ("commit_sha", "tree_sha"):
        value = str(git.get(name, "")).lower()
        if _HEX_OBJECT_ID.fullmatch(value) is None:
            raise ProductionSourceSnapshotError(f"snapshot {name} is invalid")
    if not isinstance(source, dict) or set(source) != {"file_count", "files"}:
        raise ProductionSourceSnapshotError("snapshot source schema changed")
    files = source.get("files")
    count = source.get("file_count")
    if (
        not isinstance(files, dict)
        or not files
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(files)
    ):
        raise ProductionSourceSnapshotError("snapshot source count is invalid")
    observed: dict[str, str] = {}
    casefolded: set[str] = set()
    for raw_name, raw_digest in files.items():
        name = _canonical_relative_path(raw_name)
        if name.casefold() in casefolded:
            raise ProductionSourceSnapshotError("snapshot paths are case-ambiguous")
        casefolded.add(name.casefold())
        if name != REQUIRED_CLI_PATH and not (
            name.startswith(f"{REQUIRED_PACKAGE_ROOT}/") and name.endswith(".py")
        ):
            raise ProductionSourceSnapshotError(
                "snapshot contains a source outside the frozen CLI/Python tree"
            )
        observed[name] = _require_sha256(raw_digest, f"source.files.{name}")
    if REQUIRED_CLI_PATH not in observed or not any(
        name.startswith(f"{REQUIRED_PACKAGE_ROOT}/") for name in observed
    ):
        raise ProductionSourceSnapshotError("snapshot omits the CLI or Python tree")
    return payload


def _attestation(
    *, manifest_path: Path, manifest_sha256: str, capture: _WorktreeCapture
) -> ProductionSourceSnapshotAttestation:
    return ProductionSourceSnapshotAttestation(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        worktree_root=capture.root,
        commit_sha=capture.commit_sha,
        tree_sha=capture.tree_sha,
        code_sha256=MappingProxyType(dict(capture.code_sha256)),
        code_paths=MappingProxyType(dict(capture.code_paths)),
    )


def create_production_source_snapshot(
    *, worktree_root: str | Path, output_path: str | Path
) -> ProductionSourceSnapshotAttestation:
    """Create one repository-external v1 snapshot from a clean detached worktree."""

    first = _capture_clean_detached_worktree(worktree_root)
    destination = _external_destination(output_path, first.root)
    payload = _canonical_manifest_bytes(_manifest_payload(first))
    second = _capture_clean_detached_worktree(first.root)
    if (
        first.commit_sha != second.commit_sha
        or first.tree_sha != second.tree_sha
        or dict(first.code_sha256) != dict(second.code_sha256)
        or dict(first.code_paths) != dict(second.code_paths)
    ):
        raise ProductionSourceSnapshotError(
            "production worktree changed before snapshot publication"
        )
    _atomic_write_once(destination, payload)
    manifest_sha = _sha256_bytes(payload)
    if _sha256_file(destination) != manifest_sha:
        raise ProductionSourceSnapshotError("published snapshot manifest changed")
    return _attestation(
        manifest_path=destination,
        manifest_sha256=manifest_sha,
        capture=second,
    )


def verify_production_source_snapshot(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    worktree_root: str | Path,
) -> ProductionSourceSnapshotAttestation:
    """Verify a v1 manifest against one explicit clean detached worktree."""

    expected = _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
    raw_path = Path(manifest_path)
    if raw_path.is_symlink():
        raise ProductionSourceSnapshotError("snapshot manifest must not be symbolic")
    path = raw_path.resolve(strict=True)
    if not path.is_file():
        raise ProductionSourceSnapshotError("snapshot manifest must be a file")
    root = Path(worktree_root).resolve(strict=True)
    if _is_within(path, root):
        raise ProductionSourceSnapshotError(
            "snapshot manifest must remain outside the repository worktree"
        )
    _require_repository_external_parent(path.parent)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ProductionSourceSnapshotError(
            f"cannot read source snapshot manifest: {error}"
        ) from error
    if _sha256_bytes(raw) != expected:
        raise ProductionSourceSnapshotError("snapshot manifest SHA-256 changed")
    payload = _validated_manifest_payload(raw)
    capture = _capture_clean_detached_worktree(root)
    git = payload["git"]
    source = payload["source"]
    assert isinstance(git, Mapping)
    assert isinstance(source, Mapping)
    files = source["files"]
    assert isinstance(files, Mapping)
    if (
        str(git["commit_sha"]).lower() != capture.commit_sha
        or str(git["tree_sha"]).lower() != capture.tree_sha
    ):
        raise ProductionSourceSnapshotError("snapshot Git commit/tree changed")
    if dict(files) != dict(capture.code_sha256):
        manifest_names = set(str(name) for name in files)
        live_names = set(capture.code_sha256)
        raise ProductionSourceSnapshotError(
            "snapshot source set or hashes changed: "
            f"added={sorted(live_names - manifest_names)}, "
            f"removed={sorted(manifest_names - live_names)}"
        )
    final = _capture_clean_detached_worktree(root)
    if (
        final.commit_sha != capture.commit_sha
        or final.tree_sha != capture.tree_sha
        or dict(final.code_sha256) != dict(capture.code_sha256)
    ):
        raise ProductionSourceSnapshotError("worktree changed during verification")
    if _sha256_file(path) != expected:
        raise ProductionSourceSnapshotError("snapshot manifest changed while verifying")
    return _attestation(
        manifest_path=path,
        manifest_sha256=expected,
        capture=final,
    )
