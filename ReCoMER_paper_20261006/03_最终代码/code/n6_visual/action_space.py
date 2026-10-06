"""12-action space (plan 00 S4.4): 1 empty + 4 single + 6 pairs + 1 full.

Legality: an action is legal for a row iff every selected window is valid
for that row; the EMPTY action is ALWAYS legal (guaranteed fallback).
"""
from __future__ import annotations

import itertools

import numpy as np

ACTIONS = [()] + [(i,) for i in range(4)] + \
          list(itertools.combinations(range(4), 2)) + [(0, 1, 2, 3)]
N_ACTIONS = 12
EMPTY = 0
FULL = 11


def action_mask(action_idx: int, n_windows: int = 4) -> np.ndarray:
    m = np.zeros(n_windows, dtype=np.float32)
    for w in ACTIONS[action_idx]:
        m[w] = 1.0
    return m


def legal_actions(window_valid: np.ndarray) -> np.ndarray:
    """window_valid [4] -> bool [12]; action legal iff all its windows valid.
    Empty always legal (at least one legal action guaranteed)."""
    wv = np.asarray(window_valid) > 0
    out = np.zeros(N_ACTIONS, dtype=bool)
    for i, act in enumerate(ACTIONS):
        out[i] = all(wv[w] for w in act) if act else True
    assert out[EMPTY]
    return out
