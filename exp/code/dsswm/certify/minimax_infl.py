"""JPC_infl: minimax-regret certification on the inflated set Theta_t^+ (learner side).

Theta_t^+ = union of the cells C(g), g in Theta_t (models.grid_ladder). For theta in C(g) every value moves by at
most eta_loc(g) (certify.eta_loc), so every regret moves by at most 2 eta_loc(g) and
    sup_{theta in Theta_t^+} regret_theta(pi) <= R_grid(pi; Theta_t) + 2 max_{g in Theta_t} eta_loc(g).
CERTIFIED iff R_grid(pi_c) + 2 eta_max <= eps. The inflation is policy-independent (eta_loc is a max over pi),
so the minimax policy is the grid minimax policy. With eta == 0 this reduces exactly to certify_minimax.
All grid-based methods (JPC, B3g, B12, B8) use this same Theta^+ and the same inflation in R2 / R3.
"""
from __future__ import annotations

import numpy as np

from .minimax_enum import certify_minimax
from .status import Status


def certify_minimax_infl(Reg: np.ndarray, mask: np.ndarray, eta: np.ndarray | float, eps: float, top_m: int = 5,
                         factor: float = 2.0):
    """factor = 2 for value-level eta_loc (locked definition); factor = 1 for the regret-level eta_loc_diff."""
    mask = np.asarray(mask, bool)
    eta = np.broadcast_to(np.asarray(eta, float), mask.shape)
    out = certify_minimax(Reg, mask, eps, top_m=top_m)
    if out["status"] == Status.MODEL_CONFLICT:
        out.update(eta_max=float("nan"), r_bar_grid=float("inf"), inflation=float("nan"))
        return out
    eta_max = float(eta[mask].max())
    r_grid = out["r_bar"]
    r_infl = r_grid + factor * eta_max
    out.update(r_bar_grid=r_grid, eta_max=eta_max, inflation=factor * eta_max, r_bar=r_infl,
               status=Status.CERTIFIED if r_infl <= eps else Status.NEED_DATA,
               blocking=out["blocking"] if r_grid > eps else [])
    return out


def inflation_floor(eta: np.ndarray, mask: np.ndarray, eps: float) -> bool:
    """True if no further data can certify: even a singleton set alive at the smallest-eta cell fails."""
    m = np.asarray(mask, bool)
    return bool(m.any() and 2.0 * float(np.asarray(eta)[m].min()) > eps)
