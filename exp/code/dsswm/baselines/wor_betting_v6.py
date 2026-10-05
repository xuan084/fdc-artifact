"""Lock-v6 addendum, block A(ii): tuned time-uniform WoR betting confidence sequence + rectangle certificate.

New file (v5-frozen modules untouched).  Rival ``HC-WoR``: frozen 50/50 design (identical schedule to FDC /
B4-bal[0.5]) + per-cell hedged-capital WITHOUT-replacement confidence sequence + the same rectangle ("radius-sum")
certificate as B4-bal.  Only the per-cell CS differs from B4-bal: B4-bal uses the WSR20 predictable-plug-in empirical
Bernstein WoR CS with the untuned lambda_t = min(sqrt(2 log(2/alpha) / (sig2_{t-1} t log(1+t))), 1/2); HC-WoR uses the
betting (hedged capital) construction with a lambda schedule tuned on development seeds.

Construction (Waudby-Smith & Ramdas, "Estimating means of bounded random variables by betting", JRSSB 2024, s6
"sampling without replacement", hedged capital process; WoR conditional mean as in Waudby-Smith & Ramdas, NeurIPS 2020):
  pool of N values in [0, 1] with mean m (candidate), draws X_1, X_2, ... in uniformly random order, S_{i-1} = sum of
  the first i-1 draws,
      mu_i(m) = (N m - S_{i-1}) / (N - i + 1)                       (= E[X_i | past] when the pool mean is m)
      K+_n(m) = prod_{i<=n} (1 + lam+_i(m) (X_i - mu_i(m))),   lam+_i(m) = min(lamt_i, c / mu_i(m))
      K-_n(m) = prod_{i<=n} (1 - lam-_i(m) (X_i - mu_i(m))),   lam-_i(m) = min(lamt_i, c / (1 - mu_i(m)))
  with lamt_i >= 0 predictable (a function of X_1..X_{i-1} only) and c in (0, 1).  At the true m, (X_i - mu_i(m)) is a
  martingale difference, each factor is >= 1 - c > 0, so K+ and K- are nonnegative martingales with mean 1; the hedged
  process theta K+ + (1 - theta) K- is a nonnegative martingale and Ville's inequality gives
      P(exists n <= N: max(theta K+_n(m), (1 - theta) K-_n(m)) >= 1/alpha) <= alpha.
  Monotonicity: each factor of K+ is nonincreasing in mu (hence in m), each factor of K- nondecreasing, so
  {m: theta K+_n(m) >= 1/alpha} is a lower set and {m: (1-theta) K-_n(m) >= 1/alpha} an upper set; the CS
  {m : max(...) < 1/alpha} is an interval (lo, hi) found by two bisections.  The bisection returns the last REJECTED
  point on each side, which is outward (conservative).  theta = 1/2 (WSR default).
  Validity is time-uniform in the cell's own count, so it holds under ANY sampling rule and any stopping
  (stronger than FDC's checkpoint guarantee).  Union over the S*A cells at alpha = delta / (S*A) each -> one event of
  probability >= 1 - delta on which every rectangle bound holds for all q, k (baseline_qualification_r5 [RECT]).
  Running intersection over the evaluated counts is valid on that event.

lambda schedules (pre-declared tuning grid, dev seeds 900-949, selection rule of lock v5 rival_configs):
  'prpl'  : lamt_i = sqrt(2 log(2/alpha) / (sig2_{i-1} i log(1 + i)))           (WSR predictable plug-in)
  'nstar' : lamt_i = sqrt(2 log(2/alpha) / (sig2_{i-1} n*_c))                     (tuned to a target count n*_c)
            n*_c = min(N_c, t* w_s p_a) with t* = target_frac * tau_R and p_a the frozen design share (0.5): the
            design-implied expected count of cell c at arrival t*, a public, outcome-free quantity.
  sig2_{i-1} = (1/4 + sum_{j<i} (X_j - muhat_j)^2) / i  (WSR; muhat_j = (1/2 + S_j) / (j + 1)).
"""
from __future__ import annotations

import math

import numpy as np

from .b4_bal import balanced_alloc
from .frontier_common import Method, rect_certificate

__all__ = ["HCWoRRect", "hc_wor_cs_at", "hc_wor_cs_interval", "hc_lambda_tilde", "HC_GRID", "HC_DEFAULT"]

HC_GRID = (
    {"schedule": "prpl", "c": 0.5, "target_frac": None},
    {"schedule": "prpl", "c": 0.75, "target_frac": None},
    {"schedule": "nstar", "c": 0.75, "target_frac": 0.10},
    {"schedule": "nstar", "c": 0.75, "target_frac": 0.20},
    {"schedule": "nstar", "c": 0.75, "target_frac": 0.35},
    {"schedule": "nstar", "c": 0.75, "target_frac": 0.50},
)
HC_DEFAULT = HC_GRID[3]
_BISECT_ITERS = 40


def _sig2_prev(x):
    n = x.shape[-1]
    t = np.arange(1, n + 1, dtype=float)
    cs = np.cumsum(x)
    mu_t = (0.5 + cs) / (t + 1.0)
    sq = np.cumsum((x - mu_t) ** 2)
    sig_t = (0.25 + sq) / (t + 1.0)
    return np.concatenate([[0.25], sig_t[:-1]])


def hc_lambda_tilde(x, alpha, schedule="prpl", n_star=None):
    """Predictable lamt_i (before truncation by c / mu_i(m)) for the whole draw sequence x (values in [0, 1])."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n == 0:
        return np.zeros(0)
    t = np.arange(1, n + 1, dtype=float)
    sp = _sig2_prev(x)
    L = 2.0 * math.log(2.0 / alpha)
    if schedule == "prpl":
        return np.sqrt(L / (sp * t * np.log1p(t)))
    if schedule == "nstar":
        if n_star is None or n_star <= 0:
            raise ValueError("schedule 'nstar' needs n_star > 0")
        return np.sqrt(L / (sp * float(n_star)))
    raise ValueError(f"unknown lambda schedule {schedule!r}")


def _log_cap(x, S_prev, N, lamt, m, c, sign):
    """log K^{sign}_n(m) for the prefix x[:n] (sign = +1 or -1)."""
    i = np.arange(1, x.shape[-1] + 1, dtype=float)
    mu = (N * m - S_prev) / (N - i + 1.0)
    if np.any(mu < -1e-12) or np.any(mu > 1 + 1e-12):
        return np.inf                                       # m logically impossible given the draws -> reject
    mu = np.clip(mu, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        cap = c / (mu if sign > 0 else (1.0 - mu))
    lam = np.minimum(lamt, cap)
    return float(np.sum(np.log1p(sign * lam * (x - mu))))


def hc_wor_cs_interval(x, N, alpha, lamt, c, theta=0.5, lo0=0.0, hi0=1.0):
    """Hedged-capital WoR CS for the pool mean at count n = len(x).

    Returns (lo, hi, empty).  The two one-sided rejection regions are inverted SEPARATELY over the whole bracket
    [max(lo0, S/N), min(hi0, (S + N - n)/N)] (logical pool bounds intersected with the previous running-intersection
    bounds lo0 / hi0):
      * K+ is nonincreasing in m, so {m : theta K+(m) >= 1/alpha} is a lower set [bracket_lo, r+]; lo = the last
        REJECTED bisection point (outward), or bracket_lo when nothing is rejected;
      * K- is nondecreasing in m, so {m : (1-theta) K-(m) >= 1/alpha} is an upper set [r-, bracket_hi]; hi = the first
        rejected point (outward), or bracket_hi when nothing is rejected.
    No point (in particular not the sample mean) is assumed to be accepted.  If either function rejects the whole
    bracket, or lo > hi, the CS is EMPTY (the coverage event has already failed, probability <= alpha); then
    ``empty = True`` and the conservative logical pool interval [S/N, (S + N - n)/N] is returned (it can never cause
    an extra certification relative to any non-empty CS).
    lamt: predictable lamt_i for i = 1..n (from ``hc_lambda_tilde`` on the full order; only the prefix is used)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n == 0:
        return 0.0, 1.0, False
    s = float(x.sum())
    if n >= N:
        m = s / N
        return m, m, False
    S_prev = np.concatenate([[0.0], np.cumsum(x)[:-1]])
    lt = np.asarray(lamt[:n], dtype=float)
    thr_p = math.log(1.0 / (theta * alpha))
    thr_m = math.log(1.0 / ((1.0 - theta) * alpha))
    L0, H0 = s / N, (s + (N - n)) / N                          # logical bounds of the finite pool
    a0, b0 = max(lo0, L0), min(hi0, H0)
    if a0 > b0:
        return L0, H0, True

    def fp(m):
        return _log_cap(x, S_prev, N, lt, m, c, +1) >= thr_p    # rejected by K+

    def fm(m):
        return _log_cap(x, S_prev, N, lt, m, c, -1) >= thr_m    # rejected by K-

    # ---- lower end (K+)
    if not fp(a0):
        lo = a0
    elif fp(b0):
        return L0, H0, True
    else:
        a, b = a0, b0                                          # a rejected, b accepted
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (a + b)
            if fp(mid):
                a = mid
            else:
                b = mid
        lo = a
    # ---- upper end (K-)
    if not fm(b0):
        hi = b0
    elif fm(a0):
        return L0, H0, True
    else:
        a, b = a0, b0                                          # a accepted, b rejected
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (a + b)
            if fm(mid):
                b = mid
            else:
                a = mid
        hi = b
    if lo > hi:
        return L0, H0, True
    return float(lo), float(hi), False


def hc_wor_cs_at(x, N, alpha, lamt, c, theta=0.5, lo0=0.0, hi0=1.0):
    """(lo, hi) of ``hc_wor_cs_interval`` (empty CS -> logical pool interval)."""
    lo, hi, _ = hc_wor_cs_interval(x, N, alpha, lamt, c, theta, lo0, hi0)
    return lo, hi


class HCWoRRect(Method):
    """HC-WoR: frozen 50/50 design + per-cell hedged-capital WoR CS (time-uniform) + rectangle certificate.

    The per-cell CS is computed truth-side by the v6 harness (``streams.frontier_runner_v6``) from the cell's own
    draws (exactly the records the method has read) and exposed through ``FrontierState.cs()``."""

    alloc_kind = "fixed"
    validity = "rigorous"
    cs_kind = "hc"

    def __init__(self, schedule="nstar", c=0.75, target_frac=0.20, name="HC-WoR"):
        if schedule not in ("prpl", "nstar"):
            raise ValueError(schedule)
        if not 0.0 < c < 1.0:
            raise ValueError("c must lie in (0, 1)")
        if schedule == "nstar" and not (target_frac and 0.0 < target_frac <= 1.0):
            raise ValueError("nstar needs target_frac in (0, 1]")
        self.schedule, self.c, self.target_frac = schedule, float(c), target_frac
        self.name = name
        self.alloc_p = None
        self.share = 0.5

    def setup(self, ctx):
        if abs(float(ctx.delta) - 0.05) > 1e-12:
            raise ValueError("HC-WoR: delta fixed at 0.05")
        self.alloc_p = balanced_alloc(ctx.S, ctx.A, self.share)
        p = self.alloc_p
        self.n_star = None
        if self.schedule == "nstar":
            t_star = self.target_frac * ctx.tau_R
            self.n_star = np.minimum(ctx.N, np.maximum(1.0, t_star * ctx.w[:, None] * p))

    def cs_params(self, ctx):
        """Per-cell parameters handed to the harness (all public / outcome-free)."""
        return {"schedule": self.schedule, "c": self.c, "theta": 0.5, "alpha": ctx.delta / (ctx.S * ctx.A),
                "n_star": None if self.n_star is None else self.n_star.tolist()}

    def certify(self, ctx, st):
        return rect_certificate(ctx, st)

    def describe(self):
        d = super().describe()
        d.update({"schedule": self.schedule, "c": self.c, "target_frac": self.target_frac,
                  "certificate": "rect (hedged-capital WoR CS, delta/(S*A) per cell, time-uniform)",
                  "design": "frozen 50/50 (identical schedule to FDC)"})
        return d
