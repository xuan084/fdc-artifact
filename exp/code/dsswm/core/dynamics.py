"""KNOWN (public) transition rules. Contains no unknown parameters.

Load rule (A1 for loads): matched participants +1 (capped at nmax), unmatched -1 (floored at 0).
In static variants loads are frozen.
"""
from __future__ import annotations

import numpy as np

from .actions import ActionSpace


def next_loads(loads: np.ndarray, aspace: ActionSpace, a_idx: int, nmax: int, static: bool = False) -> np.ndarray:
    if static:
        return loads.copy()
    matched = np.concatenate([aspace.matched_left(a_idx), aspace.matched_right(a_idx)])
    up = np.minimum(loads + 1, nmax)
    down = np.maximum(loads - 1, 0)
    return np.where(matched, up, down).astype(np.int64)


def next_loads_batch(loads: np.ndarray, matched: np.ndarray, nmax: int, static: bool = False) -> np.ndarray:
    """loads, matched: (n, P)."""
    if static:
        return loads.copy()
    return np.where(matched.astype(bool), np.minimum(loads + 1, nmax), np.maximum(loads - 1, 0))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
