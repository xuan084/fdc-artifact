"""A-Ney / A-XY: frozen non-adaptive allocations for the shared-certificate ablation (methodology 1.3, R2).

Both are estimated ONCE on the development half D (the caller passes D's finite-population cell means / variances;
this module never touches an environment) and then frozen; the certificate is the unchanged QFC certificate
(``frontier_common.QFCMethod`` with ``alloc_p``), so Thm 1 covers them (non-adaptive schedule).

* A-Ney: within-segment Neyman proportions p_{s,a} proportional to sigma_hat_{s,a}.
* A-XY : the XY-optimal design of RAGE (Fiez et al. 2019) targeted at the Q problems, solved by Frank-Wolfe
         (step 2/(k+2), stop at relative l2 change < 0.01 or 1000 iterations, as transcribed in lit_diff_r4 s3.2):
           minimise_p  max_q max_{pi' in Pi_{B_q}, pi' != pi*_q}  V_p(pi', pi*_q) / max(Delta_q(pi'), eps)^2,
           V_p(pi', pi) = sum_{s: pi'(s) != pi(s)} w_s (sigma^2_{s,pi'(s)} / p_{s,pi'(s)} + sigma^2_{s,pi(s)} / p_{s,pi(s)})
         (per-arrival variance of Delta_hat when segment-s arrivals are split by p_{s,.}); the max is smoothed by a
         log-sum-exp (temperature 50 on the normalised objective). Segment arrivals are exogenous, so the design
         variable is the within-segment split p (a product of simplices); a floor p >= p_min keeps every cell live.
Both matrices are clipped to p >= p_min and renormalised per segment.
"""
from __future__ import annotations

import numpy as np

__all__ = ["neyman_alloc", "xy_alloc", "xy_objective"]

P_MIN = 0.01


def _floor(p, p_min):
    p = np.maximum(np.asarray(p, dtype=float), p_min)
    return p / p.sum(1, keepdims=True)


def neyman_alloc(sigma2, p_min=P_MIN):
    sd = np.sqrt(np.maximum(np.asarray(sigma2, dtype=float), 1e-12))
    return _floor(sd / sd.sum(1, keepdims=True), p_min)


def _pairs(w, mu, pols, feas, eps):
    """Directions: for each problem, (pi', pi*) for all feasible pi' != pi*, with gap weight max(Delta, eps)^2."""
    seg = np.arange(pols.shape[1])[None, :]
    J = (np.asarray(w)[None, :] * np.asarray(mu)[seg, pols]).sum(1)
    D_list, A_list, B_list, den = [], [], [], []
    for q in range(feas.shape[0]):
        f = np.flatnonzero(feas[q])
        istar = f[int(np.argmax(J[f]))]
        others = f[f != istar]
        diff = pols[others] != pols[istar][None, :]
        D_list.append(diff)
        A_list.append(pols[others])
        B_list.append(np.broadcast_to(pols[istar], diff.shape))
        den.append(np.maximum(J[istar] - J[others], eps) ** 2)
    return np.vstack(D_list), np.vstack(A_list), np.vstack(B_list), np.concatenate(den)


def xy_objective(p, w, sigma2, dirs):
    diff, ap, bp, den = dirs
    S = p.shape[0]
    seg = np.arange(S)[None, :]
    term = np.asarray(w)[None, :] * (sigma2[seg, ap] / p[seg, ap] + sigma2[seg, bp] / p[seg, bp])
    return (np.where(diff, term, 0.0).sum(1)) / den


def xy_alloc(w, mu, sigma2, pols, feas, eps, p_min=P_MIN, iters=1000, temp=50.0, tol=0.01):
    w = np.asarray(w, dtype=float)
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), 1e-12)
    S, A = sigma2.shape
    dirs = _pairs(w, mu, np.asarray(pols, dtype=np.int64), np.asarray(feas, dtype=bool), eps)
    diff, ap, bp, den = dirs
    seg = np.arange(S)[None, :]
    p = np.full((S, A), 1.0 / A)
    hist = []
    for k in range(iters):
        f = xy_objective(p, w, sigma2, dirs)
        fmax = f.max()
        z = np.exp(temp * (f / fmax - 1.0))
        wt = z / z.sum()                                     # softmax weights of the smoothed max
        # gradient of sum_i wt_i f_i w.r.t. p[s, a]: - wt_i w_s sigma2 / p^2 / den_i on touched cells
        g = np.zeros((S, A))
        coef = (wt / den)[:, None] * w[None, :]
        for arr in (ap, bp):
            val = np.where(diff, -coef * sigma2[seg, arr] / p[seg, arr] ** 2, 0.0)
            np.add.at(g, (np.broadcast_to(seg, arr.shape), arr), val)
        # linear minimisation over the floored product of simplices: vertex per segment
        v = np.full((S, A), p_min)
        v[np.arange(S), np.argmin(g, 1)] = 1.0 - p_min * (A - 1)
        gamma = 2.0 / (k + 2.0)
        p_new = (1 - gamma) * p + gamma * v
        rel = np.linalg.norm(p_new - p) / np.linalg.norm(p)
        p = p_new
        hist.append(float(fmax))
        if k > 10 and rel < tol:
            break
    p = _floor(p, p_min)
    return p, {"iters": k + 1, "objective_final": float(xy_objective(p, w, sigma2, dirs).max()),
               "objective_uniform": float(xy_objective(np.full((S, A), 1.0 / A), w, sigma2, dirs).max()),
               "eps_design": float(eps)}
