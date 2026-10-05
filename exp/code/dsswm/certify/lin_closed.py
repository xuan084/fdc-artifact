"""T-Lin certifier: R_bar = max_{pi'} sup_{theta in E_t} <z_pi' - z_pihat, theta> (closed form, exact).

Proposition-3 range check: if a decision difference d has a component outside range(sum_mu I_mu) of the legal
interaction library that can move the regret above eps under the a-priori parameter box, the problem is
OUT_OF_SCOPE (no legal interaction can certify it); otherwise NEED_DATA.
"""
from __future__ import annotations

import numpy as np

from .status import Status


def certify_lin(Z: np.ndarray, ell, eps: float):
    th = ell.theta_hat()
    vals = Z @ th
    k_hat = int(np.argmax(vals))
    D = Z - Z[k_hat][None]
    ub = ell.sup_linear(D)
    ub[k_hat] = 0.0
    k_bind = int(np.argmax(ub))
    r_bar = float(max(ub.max(), 0.0))
    status = Status.CERTIFIED if r_bar <= eps else Status.NEED_DATA
    return {"status": status, "pi_hat": k_hat, "r_bar": r_bar, "binding": k_bind, "ub": ub,
            "theta_tilde": ell.argsup_linear(D[k_bind]) if k_bind != k_hat else th}


def range_projector(library_rows: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthogonal projector onto span(library rows) = range(sum_mu I_mu)."""
    U, s, _ = np.linalg.svd(library_rows.T, full_matrices=False)
    U = U[:, s > tol * max(s.max(), 1.0)]
    return U @ U.T


def range_status(D: np.ndarray, Pi_range: np.ndarray, box_halfwidth: np.ndarray, eps: float, tol: float = 1e-9):
    """For each decision difference row d: OUT_OF_SCOPE if the out-of-range part can change the decision by > eps.

    The out-of-range component d_perp is unidentifiable by any legal interaction; its worst-case contribution under the
    prior box |theta_k - c_k| <= h_k is sum_k |d_perp_k| h_k (a conservative upper bound).
    """
    D = np.atleast_2d(D)
    Dp = D - D @ Pi_range.T
    unident = np.abs(Dp) @ box_halfwidth
    out = np.where((np.abs(Dp).max(1) > tol) & (unident > eps), Status.OUT_OF_SCOPE.value, Status.NEED_DATA.value)
    return out, unident
