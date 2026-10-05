"""Finite-population (without-replacement) empirical-Bernstein tools for the round-4 QFC certificate.

Components
----------
* ``wor_mean_cs``   -- time-uniform confidence sequence (CS) for the mean of a finite pool sampled without
  replacement: the predictable-plug-in empirical-Bernstein WoR CS of Waudby-Smith & Ramdas (2020),
  "Confidence sequences for sampling without replacement", NeurIPS 2020 (Thm 4; lambda rule PrPl-EB of
  Waudby-Smith & Ramdas, JRSS-B 2024, eq. (26)). Used as a published component; no Hoeffding-Serfling splicing.
* ``var_ucb_binary`` -- variance upper bound for a binary pool from a mean interval: max_{m in CS} m (1 - m)
  (Lemma L2 of plan/theory/qfc_lemma.md). On E_var = {mu_c in CS_c for all c, t} it dominates sigma_c^2.
* ``functional_width`` -- Lemma L1 width sqrt(2 V x) + b x / 3 for a functional sum_c a_c mu_c, with
  V = sum_c a_c^2 sigma_bar_c^2 / n_c and b = max_c |a_c| / n_c over live cells (exhausted cells dropped).

The WoR CS (two-sided, level 1 - delta)
---------------------------------------
For X_1, X_2, ... drawn WoR from a pool of size N with values in [0, 1] and mean mu, let
S_{i-1} = sum_{j<i} X_j, mu_hat_{i-1} = (1/2 + S_{i-1}) / i, w_i = N / (N - i + 1), and

  A_t = sum_{i<=t} lambda_i (X_i + S_{i-1} / (N - i + 1)),   W_t = sum_{i<=t} lambda_i w_i,
  R_t = log(2 / delta) + sum_{i<=t} (X_i - mu_hat_{i-1})^2 psi_E(lambda_i),  psi_E(l) = -log(1 - l) - l,

with predictable lambda_i in (0, c]. Then CS_t = [(A_t - R_t) / W_t, (A_t + R_t) / W_t] satisfies
P(exists t <= N : mu not in CS_t) <= delta. The running intersection and the deterministic logical bounds
[S_t / N, (S_t + N - t) / N] are intersected in (both preserve validity).
"""
from __future__ import annotations

import numpy as np

__all__ = ["wor_mean_cs", "var_ucb_binary", "functional_width", "cell_var_ucb", "delta_per_cell",
           "psi_e", "prpl_lambda"]


def psi_e(lam):
    lam = np.asarray(lam, dtype=float)
    return -np.log1p(-lam) - lam


def prpl_lambda(x: np.ndarray, delta: float, c: float = 0.5) -> np.ndarray:
    """Predictable plug-in lambda_t = min(sqrt(2 log(2/delta) / (sigma_hat^2_{t-1} t log(1 + t))), c).

    x has shape (..., n) with values in [0, 1]; sigma_hat^2_{t-1} uses only X_1..X_{t-1}.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    t = np.arange(1, n + 1, dtype=float)
    cs = np.cumsum(x, axis=-1)
    mu_t = (0.5 + cs) / (t + 1.0)                       # mu_hat_t after observing X_t
    sq = np.cumsum((x - mu_t) ** 2, axis=-1)            # sum_{i<=t} (X_i - mu_hat_i)^2
    sig_t = (0.25 + sq) / (t + 1.0)                     # sigma_hat^2_t
    sig_prev = np.concatenate([np.full(x.shape[:-1] + (1,), 0.25), sig_t[..., :-1]], axis=-1)
    lam = np.sqrt(2.0 * np.log(2.0 / delta) / (sig_prev * t * np.log1p(t)))
    return np.minimum(lam, c)


def wor_mean_cs(x_seq, N_pool: int, delta: float, c: float = 0.5, R: float = 1.0,
                running_intersection: bool = True, return_center: bool = False):
    """Two-sided WSR20 WoR empirical-Bernstein CS for the pool mean, at every time t = 1..n.

    Parameters
    ----------
    x_seq : array (..., n), the WoR draws in sampling order; values in [0, R]. Leading axes are batch axes.
    N_pool : pool size N (n <= N).
    delta : two-sided miscoverage (time-uniform over t = 1..N).
    c : cap on lambda (WSR use 1/2 or 3/4).
    R : outcome range; data are rescaled to [0, 1] internally and bounds are returned on the original scale.

    Returns
    -------
    (lo, hi) arrays of shape (..., n): lo[..., t-1] <= mu <= hi[..., t-1] for all t simultaneously w.p. >= 1-delta.
    """
    x = np.asarray(x_seq, dtype=float) / float(R)
    if x.ndim == 0:
        x = x[None]
    n = x.shape[-1]
    N = int(N_pool)
    if n > N:
        raise ValueError(f"sequence length {n} exceeds pool size {N}")
    if n == 0:
        z = np.zeros(x.shape[:-1] + (0,))
        return (z, z) if not return_center else (z, z, z)
    if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("x_seq must lie in [0, R]")
    i = np.arange(1, n + 1, dtype=float)
    S = np.cumsum(x, axis=-1)
    S_prev = S - x
    mu_prev = (0.5 + S_prev) / i                        # mu_hat_{i-1}
    lam = prpl_lambda(x, delta, c)
    rem = N - i + 1.0                                   # N - i + 1 >= 1
    A = np.cumsum(lam * (x + S_prev / rem), axis=-1)
    W = np.cumsum(lam * (N / rem), axis=-1)
    Rt = np.log(2.0 / delta) + np.cumsum((x - mu_prev) ** 2 * psi_e(lam), axis=-1)
    center = A / W
    lo = (A - Rt) / W
    hi = (A + Rt) / W
    # deterministic logical bounds of the finite pool
    lo = np.maximum(lo, S / N)
    hi = np.minimum(hi, (S + (N - i)) / N)
    if running_intersection:
        lo = np.maximum.accumulate(lo, axis=-1)
        hi = np.minimum.accumulate(hi, axis=-1)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    hi = np.maximum(hi, lo)                             # guard against float crossings at exhaustion
    if return_center:
        return lo * R, hi * R, center * R
    return lo * R, hi * R


def var_ucb_binary(cs) -> np.ndarray:
    """sigma_bar^2 = max_{m in [lo, hi]} m (1 - m) for a binary pool (Lemma L2).

    cs : (lo, hi) tuple of floats or arrays (e.g. the output of ``wor_mean_cs`` or one time-slice of it).
    """
    lo, hi = (np.asarray(v, dtype=float) for v in cs[:2])
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    straddle = (lo <= 0.5) & (hi >= 0.5)
    m = np.where(hi < 0.5, hi, lo)                      # nearest endpoint to 1/2 when the interval misses it
    out = np.where(straddle, 0.25, m * (1.0 - m))
    return out if out.ndim else float(out)


def functional_width(a, n, var_ucb, x: float, N=None, R: float = 1.0) -> float:
    """Lemma L1 width sqrt(2 V x) + b x / 3 for sum_c a_c (mu_hat_c - mu_c).

    a : coefficients per cell; n : sample counts per cell; var_ucb : variance upper bounds per cell
    (sigma_bar_c^2, on the original outcome scale); x : log(1/error) exponent (e.g. L_1);
    N : optional pool sizes -- exhausted cells (n_c == N_c) are exact and dropped; if None, all cells with
    a_c != 0 are treated as live (the planner's, weaker, form);
    R : outcome range (b is multiplied by R; var_ucb is already on the unscaled scale).
    Returns +inf if some cell with a_c != 0 has n_c == 0.
    """
    a = np.asarray(a, dtype=float)
    n = np.asarray(n, dtype=float)
    v = np.broadcast_to(np.asarray(var_ucb, dtype=float), a.shape)
    act = a != 0
    if np.any(act & (n <= 0)):
        return float("inf")
    live = act.copy()
    if N is not None:
        live &= n < np.asarray(N, dtype=float)
    if not np.any(live):
        return 0.0
    al, nl, vl = a[live], n[live], v[live]
    V = float(np.sum(al ** 2 * vl / nl))
    b = float(np.max(np.abs(al) / nl)) * R
    return float(np.sqrt(2.0 * V * x) + b * x / 3.0)


def delta_per_cell(delta_var: float = 0.01, C_var: int = 48) -> float:
    """Per-cell two-sided miscoverage for the E_var ledger. WSR20 CSs are time-uniform, so no K factor."""
    return float(delta_var) / int(C_var)


def cell_var_ucb(x_seq, N_pool: int, delta: float, c: float = 0.5):
    """Convenience: binary-pool variance UCB at every time t from the WoR CS. Returns (sigma_bar^2[t], lo, hi)."""
    lo, hi = wor_mean_cs(x_seq, N_pool, delta, c=c)
    return var_ucb_binary((lo, hi)), lo, hi
