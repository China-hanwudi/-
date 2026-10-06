"""OOF cache builder (P4 skeleton).

Full discipline lives in the work order §5 (P4): K=3 group folds, every
label-fitted upstream refit per fold, per-row generator hashes, automatic
refusal when a row's group lies inside its teacher's training range.
This module only defines the record schema + the guard; the actual OOF
runs are a SEPARATE approval (P4).
"""
from __future__ import annotations


class OOFFoldViolation(Exception):
    pass


def guard_teacher_row(teacher_train_groups: set, row_group: str,
                      teacher_id: str) -> None:
    """Program-level refusal: a row whose group was inside the teacher's
    training groups must never be emitted by that teacher."""
    if row_group in teacher_train_groups:
        raise OOFFoldViolation(
            "teacher %s saw group %s in training; row refused"
            % (teacher_id, row_group))
