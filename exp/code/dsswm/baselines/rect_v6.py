"""Lock-v6 addendum, block A: same-strength rectangular rivals for FDC (new file; v5-frozen modules untouched).

external reviewer (result_debate_review, 2026-10-03) pointed out that every v5 rectangle rival used the per-cell WSR20 CS, which
is valid at EVERY sample size (time-uniform) and therefore pays for a stronger guarantee than FDC, which is valid only
at the K pre-specified checkpoints.  This module adds rectangle ("radius-sum") certificates that are matched to FDC in
design and guarantee strength, plus a tuned time-uniform rival:

  RECT-ck-HG    frozen 50/50 design (identical schedule to FDC / B4-bal[0.5]); per (cell, checkpoint) EXACT
                hypergeometric test-inversion interval for the pool mean, two-sided, level delta / (S A K) per
                (cell, checkpoint), i.e. alpha_side = delta / (2 S A K); running intersection over checkpoints.
                Validity: under the frozen design the counts n_c(k) are functions of the schedule A only
                (fdc_theorem s2), and conditional on A the first n_c(k) draws of pool c are a uniform WoR sample of
                fixed size, so the number of ones is exactly Hypergeometric(N_c, M_c, n_c(k)) (fdc_theorem Fact 0).
                Inverting the two exact one-sided tests gives P(mu_c not in I_c(k) | A) <= 2 alpha_side; a union over
                the S A K (cell, checkpoint) pairs gives total miscoverage <= delta (all of delta = 0.05 goes to the
                means; no variance event is needed).  On the simultaneous event every rectangle bound
                U_q(k) = max_{pi'} sum_{s: pi'(s) != pi_hat(s)} w_s (hi[s, pi'(s)] - lo[s, pi_hat(s)]) dominates
                Delta(pi'*, pi_hat) for every pi', q, k -> stream FWER <= delta at the K checkpoints (the same
                guarantee class as Theorem FDC-1).  It is an exact fixed-n test inversion (no MGF relaxation); it is NOT
                claimed to be the tightest of all valid interval constructions.

  RECT-ck-Bern  same design and union; per-cell interval from Lemma L1 (the SAME Bernstein inequality FDC uses, applied
                to a single cell with a = +-1) with the Lemma L2 variance UCB, and the SAME delta split as FDC:
                beta_c = ln(2 S A K / delta_main), x_v = ln(2 S A K / delta_var), (delta_main, delta_var) =
                (0.045, 0.005).  FDC / RECT-ck-Bern isolates the certificate GEOMETRY (direction-level quadratic form vs
                per-cell rectangle) at identical design, lemma, guarantee strength and delta split.
                Validity: fdc_theorem s3 with Pi replaced by the S A single-cell directions (+-1), i.e. E_main over
                2 S A K one-sided events at e^{-beta_c} each plus E_var unchanged.

Both certificates: exhausted cell -> exact point {mu_hat}; n = 0 -> [0, 1]; deterministic logical bounds of the finite
pool [S/N, (S + N - n)/N] are intersected (always true, no probability cost).  Certification only at the K
checkpoints (the harness calls ``certify`` only there).  No tuning parameter (the design is FDC's, the split is FDC's).

RECT-ck-HG-live (bottom of this module) spends delta only on live cells (0 < n < N) per checkpoint.
The tuned time-uniform WoR betting rival (A(ii), ``HC-WoR``) lives in ``baselines/wor_betting_v6.py``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import hypergeom

from .b4_bal import balanced_alloc
from .frontier_common import Method, pi_hat_indices, rect_U

__all__ = ["RectCkHG", "RectCkHGLive", "RectCkBern", "hg_interval_counts", "hg_mean_interval", "bern_mean_interval",
           "rect_certificate_from", "V6_DELTA", "V6_DELTA_MAIN", "V6_DELTA_VAR"]

V6_DELTA = 0.05
V6_DELTA_MAIN = 0.045
V6_DELTA_VAR = 0.005


# =============================================================================================== exact hypergeometric
def _hg_lower_M(N, n, x, a):
    """Smallest M in [x, N - n + x] with P_M(X >= x) > a  (all smaller M are rejected at level a).

    P_M(X >= x) is nondecreasing in M (the hypergeometric family is stochastically increasing in M)."""
    lo, hi = int(x), int(N - n + x)
    if x == 0:
        return 0
    # invariant: answer in [lo, hi]; P_hi(X >= x) > a always holds at hi = N - n + x? not necessarily -> check
    if hypergeom.sf(x - 1, N, hi, n) <= a:          # pragma: no cover - impossible for a < 1 (P_hi(X >= x) = 1)
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom.sf(x - 1, N, mid, n) > a:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _hg_upper_M(N, n, x, a):
    """Largest M in [x, N - n + x] with P_M(X <= x) > a  (all larger M are rejected at level a)."""
    lo, hi = int(x), int(N - n + x)
    if x == n:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(x, N, mid, n) > a:
            lo = mid
        else:
            hi = mid - 1
    return lo


def hg_interval_counts(N, n, x, alpha_side):
    """Exact two-sided test-inversion interval [M_L, M_U] for the number of ones M of a binary pool of size N, given
    x ones among n WoR draws; each tail at level alpha_side.  n = 0 -> [0, N]; n = N -> [x, x]."""
    N, n, x = int(N), int(n), int(x)
    if n <= 0:
        return 0, N
    if n >= N:
        return x, x
    return _hg_lower_M(N, n, x, alpha_side), _hg_upper_M(N, n, x, alpha_side)


def hg_mean_interval(N, n, s, alpha_side):
    """Vectorised (over cells) exact interval for the pool means; binary outcomes (sums are integer counts)."""
    N = np.asarray(N, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    s = np.rint(np.asarray(s, dtype=float)).astype(np.int64)
    lo = np.zeros(N.shape)
    hi = np.ones(N.shape)
    for idx in np.ndindex(N.shape):
        ML, MU = hg_interval_counts(N[idx], n[idx], s[idx], alpha_side)
        lo[idx] = ML / N[idx]
        hi[idx] = MU / N[idx]
    return lo, hi


# =============================================================================================== Bernstein (L1 + L2)
def bern_mean_interval(N, n, s, beta_c, x_v):
    """Per-cell Lemma L1 interval with Lemma L2 variance UCB (binary pools), clipped to the logical pool bounds."""
    from ..theory_checks.mc_l1 import bernstein_mu_ci, sigma2_ucb
    N = np.asarray(N, dtype=float)
    n = np.asarray(n, dtype=float)
    s = np.asarray(s, dtype=float)
    nn = np.maximum(n, 1.0)
    mu = np.where(n > 0, s / nn, 0.5)
    vlo, vhi = bernstein_mu_ci(mu, n, N, float(x_v))
    sig2 = sigma2_ucb(vlo, vhi)
    rad = np.sqrt(2.0 * sig2 * beta_c / nn) + beta_c / (3.0 * nn)
    lo = mu - rad
    hi = mu + rad
    lo = np.maximum(lo, s / N)                                   # logical bounds of the finite pool
    hi = np.minimum(hi, (s + (N - n)) / N)
    exact = n >= N
    lo = np.where(exact, mu, lo)
    hi = np.where(exact, mu, hi)
    empty = n <= 0
    lo = np.where(empty, 0.0, lo)
    hi = np.where(empty, 1.0, hi)
    return np.clip(lo, 0.0, 1.0), np.clip(np.maximum(hi, lo), 0.0, 1.0)


def rect_certificate_from(ctx, st, lo, hi):
    """Rectangle certificate (same formula as frontier_common.rect_certificate) from given per-cell bounds."""
    ihs = pi_hat_indices(ctx, st.mu_hat)
    out = []
    for q in range(ctx.Q):
        U = float(np.max(np.where(ctx.feas[q], rect_U(ctx, lo, hi, ihs[q]), -np.inf)))
        out.append((bool(U <= ctx.eps), int(ihs[q]), U))
    return out


# =============================================================================================== methods
class _CkRect(Method):
    alloc_kind = "fixed"
    validity = "rigorous"
    share = 0.5
    cs_kind = "none"            # the harness CS is not used (bounds from counts / sums only)

    def __init__(self):
        self.alloc_p = None
        self._lo = self._hi = None
        self.ledger = None

    def setup(self, ctx):
        if abs(float(ctx.delta) - V6_DELTA) > 1e-12:
            raise ValueError(f"{self.name}: delta is fixed at {V6_DELTA} (same as FDC)")
        if not ctx.binary:
            raise ValueError(f"{self.name}: binary pools only")
        self.alloc_p = balanced_alloc(ctx.S, ctx.A, self.share)
        self._lo = np.zeros((ctx.S, ctx.A))
        self._hi = np.ones((ctx.S, ctx.A))
        self._last_t = -1
        self.ledger = self._ledger(ctx)

    def _bounds(self, ctx, st):
        raise NotImplementedError

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("checkpoints must be visited in increasing order")
        self._last_t = st.t
        lo, hi = self._bounds(ctx, st)
        # running intersection over checkpoints: valid because the event is simultaneous over all (cell, k)
        self._lo = np.maximum(self._lo, lo)
        self._hi = np.minimum(self._hi, hi)
        self._hi = np.maximum(self._hi, self._lo)               # float guard (exhausted cells: lo == hi)
        return rect_certificate_from(ctx, st, self._lo, self._hi)

    def describe(self):
        d = super().describe()
        d.update({"design": "frozen 50/50 (balanced_alloc share 0.5; identical schedule to FDC)",
                  "guarantee": "valid at the K pre-specified checkpoints (same class as Theorem FDC-1)",
                  "ledger": self.ledger})
        return d


class RectCkHG(_CkRect):
    """Exact hypergeometric per-(cell, checkpoint) intervals, union over S*A*K, all of delta to the means."""

    name = "RECT-ck-HG"

    def _ledger(self, ctx):
        K = len(ctx.checkpoints)
        a = V6_DELTA / (2.0 * ctx.S * ctx.A * K)
        return {"alpha_side": a, "union": f"2 sides x S*A={ctx.S * ctx.A} cells x K={K} checkpoints",
                "bound": 2 * ctx.S * ctx.A * K * a}

    def _bounds(self, ctx, st):
        return hg_mean_interval(st.N, st.n, st.sum, self.ledger["alpha_side"])


class RectCkBern(_CkRect):
    """Per-cell Lemma L1 Bernstein + Lemma L2 variance UCB with FDC's delta split (geometry-only contrast)."""

    name = "RECT-ck-Bern"

    def _ledger(self, ctx):
        K = len(ctx.checkpoints)
        SA = ctx.S * ctx.A
        beta_c = math.log(2 * SA * K / V6_DELTA_MAIN)
        x_v = math.log(2 * SA * K / V6_DELTA_VAR)
        return {"beta_c": beta_c, "x_v": x_v, "delta_main": V6_DELTA_MAIN, "delta_var": V6_DELTA_VAR,
                "bound_main": 2 * SA * K * math.exp(-beta_c), "bound_var": 2 * SA * K * math.exp(-x_v)}

    def _bounds(self, ctx, st):
        return bern_mean_interval(st.N, st.n, st.sum, self.ledger["beta_c"], self.ledger["x_v"])


class RectCkHGLive(_CkRect):
    """RECT-ck-HG-live: exact hypergeometric intervals with delta spent only on LIVE cells (0 < n_c(k) < N_c).

    Spending: delta_k = delta / K per checkpoint; at checkpoint k the L_k = #{c : 0 < n_c(k) < N_c} live cells get
    alpha_side(k) = delta_k / (2 L_k) each.  Under the frozen design the counts, hence L_k, are functions of the
    outcome-free schedule A only (fdc_theorem s2), so conditional on A the union over the live (cell, k) pairs is
    sum_k sum_{live c} 2 alpha_side(k) <= sum_k delta_k = delta (equality unless some L_k = 0).  Empty cells ([0, 1]) and exhausted cells (exact
    point) have zero miscoverage and cost nothing.  When every cell is live, alpha_side(k) = delta / (2 S A K), i.e.
    exactly RECT-ck-HG; otherwise strictly larger -> weakly dominates RECT-ck-HG.  Running intersection over
    checkpoints as in RECT-ck-HG (same simultaneous event).  Valid at the K pre-specified checkpoints."""

    name = "RECT-ck-HG-live"

    def _ledger(self, ctx):
        K = len(ctx.checkpoints)
        return {"delta_per_checkpoint": V6_DELTA / K, "K": K,
                "alpha_side_rule": "delta / (2 K L_k), L_k = live cells at checkpoint k (schedule-measurable)",
                "alpha_side_all_live": V6_DELTA / (2.0 * ctx.S * ctx.A * K)}

    def _bounds(self, ctx, st):
        n, N = np.asarray(st.n), np.asarray(st.N)
        live = int(((n > 0) & (n < N)).sum())
        a = self.ledger["delta_per_checkpoint"] / (2.0 * max(live, 1))
        self.last_alpha_side = a
        return hg_mean_interval(N, n, st.sum, a)
