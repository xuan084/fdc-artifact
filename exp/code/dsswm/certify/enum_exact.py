"""Exact enumeration certifier over a finite class (ground-truth oracle for the NL certifiers).

J: (B, K) value table; mask: (B,) membership of Theta_t; k_hat chosen by the point estimate.
R_bar = max_{theta in Theta_t} max_{pi'} [J_theta(pi') - J_theta(pi_hat)]  (same theta for both policies).
"""
from __future__ import annotations

import numpy as np

from .status import Status


def certify_enum(J: np.ndarray, mask: np.ndarray, theta_hat_idx: int, eps: float, top_m: int = 5):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {"status": Status.MODEL_CONFLICT, "pi_hat": None, "r_bar": np.inf, "binding": None, "top_models": []}
    k_hat = int(np.argmax(J[theta_hat_idx]))
    Jm = J[mask]
    regret = Jm.max(1) - Jm[:, k_hat]                 # per-theta worst challenger gap
    idx = np.flatnonzero(mask)
    order = np.argsort(-regret)[:top_m]
    r_bar = float(max(regret.max(), 0.0))
    binding_theta = int(idx[order[0]])
    binding_pi = int(np.argmax(J[binding_theta]))
    status = Status.CERTIFIED if r_bar <= eps else Status.NEED_DATA
    return {"status": status, "pi_hat": k_hat, "r_bar": r_bar, "binding": binding_pi,
            "binding_theta": binding_theta, "top_models": [int(idx[o]) for o in order],
            "top_regrets": [float(regret[o]) for o in order], "set_size": int(mask.sum())}


def sup_linear_enum(points: np.ndarray, D: np.ndarray) -> np.ndarray:
    """max over an enumerated finite set of theta points of <d, theta>, for each row d."""
    return (np.atleast_2d(D) @ points.T).max(1)
