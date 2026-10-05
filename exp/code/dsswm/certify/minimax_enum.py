"""Exact-enumeration certifier with minimax-regret policy choice and the finite-class Prop.-3 trichotomy.

regret_theta(pi) = max_pi' J_theta(pi') - J_theta(pi)   (same theta for the whole trajectory and all policies)
R(pi) = max_{theta in Theta_t} regret_theta(pi);  pi_c = argmin_pi R(pi);  CERTIFIED iff R(pi_c) <= eps.
This is sound for exactly the same reason as certify_enum (sup over the confidence set); choosing the minimax
policy instead of argmax J_theta_hat can only certify earlier, never wrongly.

OUT_OF_SCOPE (finite-class analogue of Prop. 3): group Theta_t into classes that are observationally equivalent
under every *legal* interaction (identical outcome / retention probabilities on all legal factors). No future
legal data can separate members of one class. If every class is decision-ambiguous (min_pi max_{theta in class}
regret > eps), no legal interaction sequence can ever certify -> OUT_OF_SCOPE.
"""
from __future__ import annotations

import numpy as np

from .status import Status


def regret_matrix(J: np.ndarray) -> np.ndarray:
    return J.max(1, keepdims=True) - J


def certify_minimax(Reg: np.ndarray, mask: np.ndarray, eps: float, top_m: int = 5):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {"status": Status.MODEL_CONFLICT, "pi": None, "r_bar": float("inf"), "blocking": [], "set_size": 0}
    Rm = Reg[mask]
    R = Rm.max(0)
    pi = int(np.argmin(R))
    r_bar = float(max(R[pi], 0.0))
    idx = np.flatnonzero(mask)
    reg_pi = Rm[:, pi]
    order = np.argsort(-reg_pi)
    blocking = [int(idx[o]) for o in order[:top_m] if reg_pi[o] > eps]
    status = Status.CERTIFIED if r_bar <= eps else Status.NEED_DATA
    return {"status": status, "pi": pi, "r_bar": r_bar, "blocking": blocking,
            "blocking_regret": [float(reg_pi[o]) for o in order[:top_m] if reg_pi[o] > eps],
            "set_size": int(mask.sum()), "R_all": R}


def observable_signature(py: np.ndarray, pe: np.ndarray, max_level: int, decimals: int = 9) -> np.ndarray:
    """Per-theta signature of the distribution of every legal observation factor (levels <= max_level)."""
    B = py.shape[0]
    sig = np.concatenate([py[..., :max_level + 1].reshape(B, -1), pe.reshape(B, -1)], 1)
    return np.round(sig, decimals)


def group_ids(signature: np.ndarray) -> np.ndarray:
    _, inv = np.unique(signature, axis=0, return_inverse=True)
    return inv.reshape(-1)


def trichotomy(Reg: np.ndarray, mask: np.ndarray, gid: np.ndarray, eps: float):
    """Returns (all_ambiguous, n_classes, n_ambiguous). Classes restricted to Theta_t."""
    mask = np.asarray(mask, bool)
    g = gid[mask]
    Rm = Reg[mask]
    uniq, inv = np.unique(g, return_inverse=True)
    # per-class max regret for every policy, then min over policies
    worst = np.full((len(uniq), Reg.shape[1]), -np.inf)
    np.maximum.at(worst, inv, Rm)
    amb = worst.min(1) > eps
    return bool(amb.all()), int(len(uniq)), int(amb.sum())
