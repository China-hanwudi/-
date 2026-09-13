from __future__ import annotations
from typing import Any, Iterable

FOLD_SESSIONS = {
    5: {"train": frozenset({"Session2", "Session3", "Session4"}), "dev": frozenset({"Session1"})}
}
FOLD_COUNTS = {5: {"train": 3205, "dev": 1085}}

def validate_fold(fold: int) -> None:
    if fold not in FOLD_SESSIONS:
        raise RuntimeError(f"only fold 5 is authorized, got fold={fold}")

def validate_fold_role(rows: Iterable[dict[str, Any]], *, fold: int, role: str, exact_count: bool = False) -> None:
    validate_fold(fold)
    if role not in {"train", "dev"}:
        raise RuntimeError(f"invalid role={role!r}")
    materialized = list(rows)
    sessions = {str(row["session_id"]) for row in materialized}
    expected = set(FOLD_SESSIONS[fold][role])
    if sessions != expected:
        raise RuntimeError(f"fold{fold} {role} sessions changed: expected={sorted(expected)} observed={sorted(sessions)}")
    if exact_count and len(materialized) != FOLD_COUNTS[fold][role]:
        raise RuntimeError(f"fold{fold} {role} count changed")
    if not {int(row["label"]) for row in materialized}.issubset({0,1,2,3}):
        raise RuntimeError("unexpected IEMOCAP labels")
