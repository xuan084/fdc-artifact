"""Oracle reference rho* for E1-NL (harness side: receives the true parameter tables as an argument).

Relaxed information-theoretic reference for the number of additional rounds the LR-set certifier needs:
    T* = 1 / z*,   z* = max_{w in Delta(X)} min_{theta in Alt} sum_x w_x KL_x(theta* || theta) / m_theta
X = all (observable state, legal action) pairs (state choice is FREE - reachability / steering is relaxed away),
Alt = models in the current Theta_t that block certification of the truth-optimal minimax policy,
m_theta = remaining LR margin log(1/delta) - log M_t(theta) (>0 inside Theta_t).
Solved by constraint generation over Alt with HiGHS. This is a reference scale for the suspicious-gate check
("interactions below oracle rho*"), not a formal lower bound for the plug-in LR set.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ..acquire.nl_kl_dda import kl_features


def oracle_rho(true_py, true_pe, model_py, model_pe, margins, S_unique, max_rounds=30, batch=200):
    """true_*: (1,...) tables; model_*: (M,...) tables of the Alt models; margins (M,); S_unique (ncol, nf)."""
    M = len(margins)
    if M == 0:
        return {"T_star": 0.0, "z": float("inf"), "n_alt": 0, "active": 0}
    F = kl_features(true_py, true_pe, model_py, model_pe)            # (nf, M)
    A = (S_unique @ F) / np.maximum(margins, 1e-9)[None]              # (ncol, M) rate per unit margin
    ncol = A.shape[0]
    # start with the models that are hardest under the uniform design
    order = np.argsort(A.mean(0))
    act = list(order[:min(batch, M)])
    z, w = 0.0, np.full(ncol, 1.0 / ncol)
    for _ in range(max_rounds):
        sub = A[:, act]
        if (sub.max(0) <= 1e-15).any():
            return {"T_star": float("inf"), "z": 0.0, "n_alt": M, "active": len(act)}
        c = np.zeros(ncol + 1); c[-1] = -1.0
        A_ub = np.hstack([-sub.T, np.ones((len(act), 1))])
        A_eq = np.zeros((1, ncol + 1)); A_eq[0, :ncol] = 1.0
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(act)), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * ncol + [(None, None)], method="highs")
        if res.status != 0:
            break
        w, z = np.clip(res.x[:ncol], 0, None), float(res.x[-1])
        vals = w @ A
        viol = np.flatnonzero(vals < z * (1 - 1e-7) - 1e-12)
        viol = [v for v in viol[np.argsort(vals[viol])] if v not in set(act)][:batch]
        if not viol:
            break
        act += viol
    z_full = float((w @ A).min())
    return {"T_star": float(1.0 / z_full) if z_full > 0 else float("inf"), "z": z_full, "n_alt": M,
            "active": len(act), "support": int((w > 1e-6).sum())}
