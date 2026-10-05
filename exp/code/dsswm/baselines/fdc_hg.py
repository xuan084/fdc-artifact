"""FDC-HG: FDC-BF with each cell's log-MGF bound replaced by the upper envelope, over the exact-HG variance box, of the
EXACT hypergeometric log-MGF (plan/fdc_hg_theorem.md).  NEW file; v5-v8 modules are imported, never modified.

Everything except the cell MGF is inherited from FDC-BF (fdc_bet.FDCBet with kind='bennett', var_box='HG', fpc=True,
rect=False, split 0.045/0.005): frozen 50/50 design (identical schedule digest), exact HG variance box with running
intersection, union ledger beta, lambda grid lambda_0 * [1/200, 10] (161 points), certification rule.

Cell bound (live cell, 0 < n < N; X ~ HG(N, M, n), t = u / n):
    Lambda(u; M) = log sum_x p_M(x) exp(t (x - n M / N))
    Lambda_bar(u) = max_{M in box} Lambda(u; M),   box = {ceil(N lo), ..., floor(N hi)}
Computable upper bound Lambda_tilde >= Lambda_bar from
  * Lemma T  safe truncation of the sum (log-concave tilted pmf -> geometric tail bounds);
  * Lemma R  interval bound in M: f(M+1)/f(M) = 1 + (e^t - 1)(n - h_t(M))/(N - M), h_t nondecreasing in M (MLR) ->
             two-line bound on every [M1, M2]; branch and bound until slack <= tol (any stopping point is still valid);
  * Lemma C  convexity in u (Lambda_bar(0) = 0) -> linear chords between grid points are upper bounds;
  * min with FDC-BF's Bennett-FPC box supremum (so FDC-HG widths <= FDC-BF widths pointwise, by construction).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import hypergeom

from .fdc_bet import FDCBet, _LAMBDA_GRID
from .frontier_common import jhat_all, pi_hat_indices
from .rect_v6 import hg_interval_counts

__all__ = ["hg_logmgf_bounds", "hg_envelope", "CellEnvelope", "direction_widths_fn", "FDCHG", "exact_logmgf"]

_SAFE_ABS = 1e-12
_SAFE_REL = 1e-9
_TAIL_REL = 1e-12
BB_TOL = 1e-3            # branch-and-bound slack target (Lambda units); affects tightness only
BB_MAX_EVAL = 400        # cap on exact evaluations per cell envelope (bound stays valid if hit)
U_RATIO = 1.05           # geometric |u| grid ratio (chord interpolation)
U_BENNETT_RANGE = (0.01, 60.0)   # |u| grid spans Bennett-sup values in this range; outside -> Bennett


# =============================================================================================== exact (reference)
def exact_logmgf(u, n, N, M):
    """Full-support reference log E exp(u (mu_hat - M/N)) in 80-bit long double, pmf from the exact neighbour-ratio
    recurrence normalised over the full support (independent of scipy's lgamma; used by tests)."""
    n, N, M = int(n), int(N), int(M)
    ld = np.longdouble
    lo, hi = max(0, n - (N - M)), min(n, M)
    y = np.arange(lo, hi, dtype=ld)
    lr = np.log(ld(M) - y) + np.log(ld(n) - y) - np.log(y + 1) - np.log(ld(N - M - n + 1) + y)
    lq = np.concatenate([np.zeros(1, dtype=ld), np.cumsum(lr)])
    xs = np.arange(lo, hi + 1, dtype=ld)
    c = ld(n) * ld(M) / ld(N)
    u = np.atleast_1d(np.asarray(u, dtype=float)).astype(ld)

    def lse(v, axis=0):
        mx = v.max(axis)
        return mx + np.log(np.exp(v - mx).sum(axis))
    num = lse(lq[:, None] + (u[None, :] / n) * (xs[:, None] - c))
    return (num - lse(lq)).astype(float)


# =============================================================================================== Lemma T
def _tilted_mode(N, M, n, t):
    """Approximate mode of the tilted pmf g(x) ~ C(M,x) C(N-M,n-x) e^{tx} (root of r_t(x) = 1); used only to place the
    summation window (validity comes from the tail bounds, not from this)."""
    w = math.exp(min(max(t, -700.0), 700.0))
    # (M - x)(n - x) w = (x + 1)(N - M - n + x + 1)  ->  a x^2 + b x + c = 0
    a = w - 1.0
    b = -(M + n) * w - (N - M - n + 2.0)
    c = M * n * w - (N - M - n + 1.0)
    if abs(a) < 1e-12:
        x = -c / b if b != 0 else n * M / N
    else:
        disc = max(b * b - 4 * a * c, 0.0)
        r1 = (-b - math.sqrt(disc)) / (2 * a)
        r2 = (-b + math.sqrt(disc)) / (2 * a)
        lo, hi = max(0, n - (N - M)), min(n, M)
        cands = [r for r in (r1, r2) if lo - 1 <= r <= hi + 1]
        x = cands[0] if cands else n * M / N
    return x


_EPS = np.finfo(float).eps


def hg_logmgf_bounds(N, M, n, t):
    """For X ~ HG(N, M, n) and an array of tilts t (= u / n): directed enclosures
        Lambda = log E exp(t (X - n M / N))   ->  (lam_lo, lam_hi)   lam_lo <= Lambda <= lam_hi
        h      = E[X e^{tX}] / E[e^{tX}]       ->  (h_lo, h_hi)       h_lo <= h <= h_hi
    The pmf is built (unnormalised) from the exact neighbour-ratio recurrence anchored at x0 (no lgamma of huge
    arguments), each log-weight carries an explicit floating-point error radius e(x), and the normalisation is done
    on the same window (so the common constant cancels exactly).  Truncation: Lemma T (log-concave tilted pmf,
    geometric tail bounds).  Upper bound uses (core_up + tilted tails_up) / core_dn of the untilted mass."""
    N, M, n = int(N), int(M), int(n)
    t = np.asarray(t, dtype=float)
    xs_lo, xs_hi = max(0, n - (N - M)), min(n, M)
    c = n * M / N                                          # centring constant (float)
    ec = 4 * _EPS * abs(c)                                 # its rounding radius
    if xs_lo == xs_hi:                                     # degenerate law: X = const
        x0 = float(xs_lo)
        lam = t * (x0 - c)
        rad = np.abs(t) * (ec + 4 * _EPS * (abs(x0) + abs(c))) + 4 * _EPS * np.abs(lam) + 1e-300
        return lam - rad, lam + rad, np.full_like(t, x0), np.full_like(t, x0)
    var = n * (M / N) * (1 - M / N) * (N - n) / max(N - 1, 1)
    sd = math.sqrt(max(var, 0.0)) + 1.0
    m_lo = _tilted_mode(N, M, n, float(t.min()))
    m_hi = _tilted_mode(N, M, n, float(t.max()))
    Lmax = math.log(max(N, 2)) + 1.0
    ext = 16.0 * sd + 30.0
    while True:
        a = max(xs_lo, int(math.floor(min(m_lo, m_hi, c) - ext)))
        b = min(xs_hi, int(math.ceil(max(m_lo, m_hi, c) + ext)))
        xs = np.arange(a, b + 1)
        x0 = int(min(max(round(c), a), b))
        y = np.arange(a, b).astype(float)                       # steps y -> y + 1
        lr = (np.log(M - y) + np.log(n - y)) - (np.log(y + 1.0) + np.log(N - M - n + y + 1.0))
        i0 = x0 - a
        lq = np.zeros(len(xs))                                  # log q(x), q(x0) = 1, accumulated FROM the anchor
        if i0 < len(lr):
            lq[i0 + 1:] = np.cumsum(lr[i0:])                    # x > x0: sum_{x0 <= y < x} lr(y)
        if i0 > 0:
            lq[:i0] = -np.cumsum(lr[:i0][::-1])[::-1]           # x < x0: -sum_{x <= y < x0} lr(y)
        k = np.abs(xs - x0).astype(float)
        # running max of |lq| along the path x0 -> x (bounds the partial sums' magnitude)
        absq = np.abs(lq)
        run = np.empty_like(absq)
        run[i0:] = np.maximum.accumulate(absq[i0:])
        run[:i0 + 1] = np.maximum.accumulate(absq[:i0 + 1][::-1])[::-1]
        eq = 2 * _EPS * (k + 1) * (16 * Lmax + run) + 1e-300     # radius of log q(x)
        d = xs - c
        td = t[None, :] * d[:, None]
        etd = np.abs(t)[None, :] * (ec + 4 * _EPS * (np.abs(xs) + abs(c)))[:, None] + 4 * _EPS * np.abs(td)
        # one exp pass: Z = sum_x q(x) e^{t d(x)}, S = sum_x x q(x) e^{t d(x)}; every per-term log radius r(x) =
        # eq(x) + etd(x) is folded in through its maximum R (sum e^{lg + r} <= e^{R} sum e^{lg}), plus the
        # exp / summation rounding radius esum.
        lg = lq[:, None] + td
        R = (eq[:, None] + etd).max(0)                          # (T,)
        R0 = float(eq.max())
        W = len(xs)
        esum = math.log1p(8 * _EPS * (W + 2))
        mx = lg.max(0)
        E = np.exp(lg - mx[None, :])
        z = E.sum(0)
        sx = xs.astype(float) @ E
        lz = mx + np.log(z)
        with np.errstate(divide="ignore"):
            lsx = mx + np.log(sx)
        num_up, num_dn = lz + R + esum, lz - R - esum
        ls_up, ls_dn = lsx + R + esum, lsx - R - esum
        mq = lq.max()
        lzq = mq + math.log(np.exp(lq - mq).sum())
        den_dn, den_up = lzq - R0 - esum, lzq + R0 + esum
        lg_up_first = lg[0] + R
        lg_up_last = lg[-1] + R
        ok = True
        log_tb = np.full_like(t, -np.inf)
        log_tbx = np.full_like(t, -np.inf)
        log_ta = np.full_like(t, -np.inf)
        log_t0 = -np.inf
        lr_eps = 1e-12                                          # upward pad on log ratios
        if b < xs_hi:
            lrb = (math.log(M - b) + math.log(n - b)) - (math.log(b + 1.0) + math.log(N - M - n + b + 1.0)) + lr_eps
            lrho = lrb + t
            if np.any(lrho >= -1e-9) or lrb >= -1e-9:
                ok = False
            else:
                rho = np.exp(lrho)
                log_tb = lg_up_last + lrho - np.log1p(-rho)
                log_tbx = lg_up_last + np.log(b * rho / (1 - rho) + rho / (1 - rho) ** 2) + 1e-12
                log_t0 = np.logaddexp(log_t0, lq[-1] + eq[-1] + lrb - math.log1p(-math.exp(lrb)))
        if a > xs_lo and ok:
            lra = (math.log(M - (a - 1)) + math.log(n - (a - 1))) - (math.log(a) + math.log(N - M - n + a))
            lrho = -(lra + t) + lr_eps                          # log g(a-1)/g(a)
            l0 = -lra + lr_eps
            if np.any(lrho >= -1e-9) or l0 >= -1e-9:
                ok = False
            else:
                rho = np.exp(lrho)
                log_ta = lg_up_first + lrho - np.log1p(-rho)
                log_t0 = np.logaddexp(log_t0, lq[0] + eq[0] + l0 - math.log1p(-math.exp(l0)))
        if ok:
            log_t = np.logaddexp(log_ta, log_tb)
            if np.all(log_t - num_dn <= math.log(_TAIL_REL)) or (a == xs_lo and b == xs_hi):
                break
        if a == xs_lo and b == xs_hi:                             # pragma: no cover - full support: tails empty
            log_t = np.full_like(t, -np.inf)
            break
        ext *= 2.0
    lam_hi = np.logaddexp(num_up, log_t) - den_dn
    lam_lo = num_dn - np.logaddexp(den_up, log_t0)
    lam_hi = lam_hi + 4 * _EPS * (np.abs(lam_hi) + np.abs(num_up) + abs(den_dn)) + 1e-300
    lam_lo = lam_lo - 4 * _EPS * (np.abs(lam_lo) + np.abs(num_dn) + abs(den_up))
    # h = sum x g / sum g (centring cancels): lower = S_dn / (Z_up + tails), upper = (S_up + x-tails) / Z_dn
    h_lo = np.exp(ls_dn - np.logaddexp(num_up, log_t)) * (1 - 16 * _EPS)
    h_lo = np.maximum(h_lo, float(xs_lo))
    num_x = np.logaddexp(np.logaddexp(ls_up, log_ta + (math.log(a) if a > 0 else -np.inf)), log_tbx)
    with np.errstate(over="ignore"):
        h_hi = np.minimum(np.exp(num_x - num_dn) * (1 + 16 * _EPS) + 1e-299, float(xs_hi))   # +abs: underflow
    return lam_lo, lam_hi, h_lo, h_hi


# =============================================================================================== Lemma R + B&B
def _log_ratio(t, p):
    """log(1 + (e^t - 1) p) = log((1 - p) + p e^t), p in [0, 1], stable for all finite t (no expm1 overflow)."""
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log1p(-p), np.log(p) + t)


def _interval_bound(t, N, n, M1, M2, e1, e2):
    """Upper bound on max_{M in [M1, M2]} Lambda(u; M) from endpoint enclosures e = (lam_lo, lam_hi, h_lo, h_hi).
    Non-finite results are returned as +inf (never NaN)."""
    L = float(M2 - M1)
    p_bar = np.clip((n - e1[2]) / (N - M2 + 1.0) * (1 + 8 * _EPS), 0.0, 1.0)
    p_low = np.clip((n - e2[3]) / float(N - M1) * (1 - 8 * _EPS), 0.0, 1.0)
    pos = t > 0
    p_hi_ratio = np.where(pos, p_bar, p_low)                 # gives the largest ratio
    p_lo_ratio = np.where(pos, p_low, p_bar)
    c = t * n / N
    sa = _log_ratio(t, p_hi_ratio) - c
    sb = _log_ratio(t, p_lo_ratio) - c
    pad_s = 8 * _EPS * (np.abs(t) + np.abs(c) + 1.0)
    sa = sa + pad_s
    sb = sb - pad_s
    l1, l2 = e1[1], e2[1]
    den = sa - sb
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = np.where(den > 1e-300, (l2 - l1 - sb * L) / den, L)
        x = np.clip(np.nan_to_num(x, nan=L), 0.0, L)
        vals = []
        for xx in (np.zeros_like(t), np.full_like(t, L), x):
            vals.append(np.minimum(l1 + sa * xx, l2 - sb * (L - xx)))
        out = np.maximum.reduce(vals)
        out = out + 16 * _EPS * (np.abs(l1) + np.abs(l2) + (np.abs(sa) + np.abs(sb)) * L) + 1e-300
    return np.where(np.isfinite(out), out, np.inf)


def hg_envelope(N, n, M_lo, M_hi, t, tol=BB_TOL, max_eval=BB_MAX_EVAL, return_info=False):
    """Rigorous upper bound on max_{M_lo <= M <= M_hi} Lambda(u; M) for every tilt in t (branch and bound).
    Any exit (convergence or evaluation cap) returns a valid upper bound; tightness <= tol only if converged."""
    N, n, M_lo, M_hi = int(N), int(n), int(M_lo), int(M_hi)
    t = np.asarray(t, dtype=float)
    ev = {}

    def get(M):
        if M not in ev:
            ev[M] = hg_logmgf_bounds(N, M, n, t)
        return ev[M]

    k0 = min(9, M_hi - M_lo + 1)
    pts = sorted(set(int(round(x)) for x in np.linspace(M_lo, M_hi, k0)) | {M_lo, M_hi})
    for M in pts:
        get(M)
    while True:
        best_lo = np.maximum.reduce([ev[M][0] for M in pts])
        up = np.maximum.reduce([ev[M][1] for M in pts])
        new = []
        for M1, M2 in zip(pts[:-1], pts[1:]):
            if M2 - M1 <= 1:
                continue
            bnd = _interval_bound(t, N, n, M1, M2, ev[M1], ev[M2])
            up = np.maximum(up, bnd)
            if np.any(bnd > best_lo + tol):
                new.append((M1 + M2) // 2)
        if not new:
            break
        if len(ev) + len(new) > max_eval:
            break
        for M in new:
            get(M)
        pts = sorted(set(pts) | set(new))
    up = np.where(np.isfinite(up), up, np.inf)
    converged = bool(np.max(up - best_lo) <= tol)              # tolerance actually met (incl. endpoint radii)
    if return_info:
        argmax = [pts[i] for i in np.argmax(np.stack([ev[M][0] for M in pts]), axis=0)]
        return up, {"n_eval": len(ev), "argmax_M": argmax, "slack": float(np.max(up - best_lo)),
                    "converged": converged}
    return up


def _phi_up(x):
    """Upper bound on phi(x) = e^x - 1 - x for x >= 0 (series with remainder for small x; no cancellation)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        small = x < 1e-3
        ser = x * x / 2 + x ** 3 / 6 + x ** 4 / 24 * np.exp(np.minimum(x, 1.0)) + np.where(x > 0, 1e-300, 0.0)
        big = np.expm1(x) - x
        out = np.where(small, ser, big) * (1 + 1e-12)
    return np.where(np.isnan(out), np.inf, out)


def var_sup_int(n, N, M_lo, M_hi):
    """Upward-rounded sup_{M_lo <= M <= M_hi} Var(X) = n M (N - M) (N - n) / (N^2 (N - 1)) from INTEGER endpoints
    (exact integer arithmetic, one outward-rounded division; no 1 - m cancellation)."""
    n, N, M_lo, M_hi = int(n), int(N), int(M_lo), int(M_hi)
    if not (0 < n < N):
        return 0.0
    best = max(M * (N - M) for M in {min(max(N // 2, M_lo), M_hi), min(max((N + 1) // 2, M_lo), M_hi)})
    if best == 0:
        return 0.0
    from fractions import Fraction
    v = Fraction(n * best * (N - n), N * N * (N - 1))
    return float(v) * (1 + 4 * _EPS)


def bennett_sup_safe(u, n, N, M_lo, M_hi):
    """Bennett-FPC box supremum with outward rounding: Var_bar(X) from integer endpoints, phi upper bound."""
    varX = var_sup_int(n, N, M_lo, M_hi)
    if varX <= 0:
        return np.zeros(np.shape(u))
    return varX * _phi_up(np.abs(np.asarray(u, dtype=float)) / n) * (1 + 4 * _EPS)


class CellEnvelope:
    """Per-(cell, checkpoint) envelope on a geometric |u| grid; evaluation by chords (Lemma C) min Bennett-sup.
    M_lo / M_hi: integer HG box endpoints (passed exactly by FDCHG; if omitted they are recovered from lo / hi)."""

    def __init__(self, n, N, lo, hi, ratio=U_RATIO, bennett_range=U_BENNETT_RANGE, tol=BB_TOL, M_lo=None, M_hi=None):
        self.n, self.N, self.lo, self.hi = int(n), int(N), float(lo), float(hi)
        if M_lo is None:                                     # outward recovery (valid for N < 2^50): may add one M
            from fractions import Fraction
            if N >= 2 ** 50:
                raise ValueError("pass integer M_lo / M_hi for N >= 2^50")
            M_lo = max(0, math.floor(Fraction(lo) * N))
            M_hi = min(int(N), math.ceil(Fraction(hi) * N))
        self.M_lo, self.M_hi = int(M_lo), int(M_hi)
        if self.M_hi < self.M_lo:                            # pragma: no cover - box is never empty
            self.M_lo, self.M_hi = 0, int(N)
        v = var_sup_int(n, N, self.M_lo, self.M_hi)          # Var_bar(X) (integer box sup, rounded up)
        self.varX = v
        self.converged = True
        if v <= 0:
            self.ug = np.zeros(0)
            self.vpos = self.vneg = np.zeros(0)
            return
        b_lo, b_hi = bennett_range

        def solve(b):                                        # grid placement only (not a validity step)
            lo_, hi_ = -30.0, 30.0
            for _ in range(80):
                mid = 0.5 * (lo_ + hi_)
                with np.errstate(over="ignore"):
                    val = v * float(_phi_up(math.exp(mid) / n))
                if val < b:
                    lo_ = mid
                else:
                    hi_ = mid
            return math.exp(hi_)
        u1, u2 = solve(b_lo), solve(b_hi)
        k = int(math.ceil(math.log(u2 / u1) / math.log(ratio))) + 1
        self.ug = u1 * ratio ** np.arange(k)
        t = np.concatenate([-self.ug[::-1], self.ug]) / n
        env, info = hg_envelope(N, n, self.M_lo, self.M_hi, t, tol=tol, return_info=True)
        self.converged = info["converged"]
        self.vneg = env[:k][::-1].copy()                     # value at -ug
        self.vpos = env[k:].copy()

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        ben = bennett_sup_safe(u, self.n, self.N, self.M_lo, self.M_hi)
        if self.ug.size == 0:
            return ben
        a = np.abs(u)
        out = np.full(u.shape, np.inf)
        xg = np.concatenate([[0.0], self.ug])
        for sel, vals in ((u >= 0, self.vpos), (u < 0, self.vneg)):
            if not sel.any():
                continue
            aa = a[sel]
            yg = np.concatenate([[0.0], vals])
            r = np.interp(aa, xg, yg, right=np.inf)          # chords (valid: convex envelope, values are upper bounds)
            r = r * (1 + 8 * _EPS) + 1e-300
            r = np.where(aa > self.ug[-1], np.inf, r)
            out[sel] = r
        out = np.where(np.isnan(out), np.inf, out)
        return np.minimum(out, ben)


# =============================================================================================== widths
def direction_widths_fn(ctx, ih, n, N, lo, hi, beta, cell_fn, grid=_LAMBDA_GRID):
    """fdc_bet.direction_widths with the per-cell bound supplied by cell_fn(s, a, u) (live cells only); all other
    steps (difference set, V_pair, lambda_0, grid, exhausted / empty handling) are copied verbatim."""
    S = ctx.S
    seg = np.arange(S)
    pols = ctx.pols
    ph = pols[ih]
    D = pols != ph[None, :]
    n = np.asarray(n, dtype=float)
    N = np.asarray(N, dtype=float)
    live = (n > 0) & (n < N)
    zero = n <= 0
    w = ctx.w
    vmax = np.clip(0.5, lo, hi)
    vmax = vmax * (1.0 - vmax)
    fp = (N - n) / np.maximum(N - 1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vcell = np.where(live, vmax * fp / np.maximum(n, 1.0), 0.0)
    a_ch = pols
    a_h = np.broadcast_to(ph[None, :], pols.shape)
    zero_touch = (D & (zero[seg[None, :], a_ch] | zero[seg[None, :], a_h])).any(1)
    w2 = (w ** 2)[None, :]
    Vpair = (D * w2 * (vcell[seg[None, :], a_ch] + vcell[seg[None, :], a_h])).sum(1)
    out = np.zeros(pols.shape[0])
    beta = np.broadcast_to(np.asarray(beta, dtype=float), (pols.shape[0],))
    act = Vpair > 0
    if act.any():
        Pi = np.flatnonzero(act)
        lam0 = np.sqrt(2.0 * beta[Pi] / Vpair[Pi])
        Lam = lam0[:, None] * grid[None, :]
        F = np.zeros_like(Lam)
        for s in range(S):
            dsel = D[Pi, s]
            if not dsel.any():
                continue
            rows = np.flatnonzero(dsel)
            for role, arms, sign in (("ch", a_ch[Pi[rows], s], -1.0), ("h", a_h[Pi[rows], s], +1.0)):
                for a in np.unique(arms):
                    r = rows[arms == a]
                    if not live[s, a]:
                        continue
                    u = sign * Lam[r] * w[s]
                    F[r] += cell_fn(s, a, u)
        with np.errstate(invalid="ignore", over="ignore"):
            val = (beta[Pi][:, None] + F) / Lam
        val = np.where(np.isfinite(val), val, np.inf)
        out[Pi] = val.min(1)
    out = np.where(D.any(1), out, 0.0)
    out = np.where(zero_touch, np.inf, out)
    return out


# =============================================================================================== method
class FDCHG(FDCBet):
    """FDC-BF with the exact-hypergeometric envelope cell MGF (Theorem FDC-HG-1)."""

    def __init__(self, name="FDC-HG", tol=BB_TOL, ratio=U_RATIO):
        super().__init__(kind="bennett", var_box="HG", fpc=True, rect=False, split=(0.045, 0.005), name=name)
        self.tol, self.ratio = float(tol), float(ratio)
        self.timing = []
        self.bb_unconverged = 0

    def setup(self, ctx):
        super().setup(ctx)
        self._Ml = np.zeros((ctx.S, ctx.A), dtype=np.int64)
        self._Mh = None

    def _box(self, ctx, st):
        """Same box as FDC-BF (exact HG inversion, running intersection), tracked in INTEGER counts; lo / hi are the
        identical floats ML / N, MU / N (division is monotone, so the running max / min commute with it)."""
        N = np.asarray(st.N, dtype=np.int64)
        n = np.asarray(st.n, dtype=np.int64)
        s = np.rint(np.asarray(st.sum, dtype=float)).astype(np.int64)
        if self._Mh is None:
            self._Mh = N.copy()
        a = self.ledger["alpha_side_var"]
        for idx in np.ndindex(N.shape):
            ML, MU = hg_interval_counts(N[idx], n[idx], s[idx], a)
            self._Ml[idx] = max(self._Ml[idx], ML)
            self._Mh[idx] = max(min(self._Mh[idx], MU), self._Ml[idx])
        self._lo = self._Ml / N
        self._hi = self._Mh / N
        return self._lo.copy(), self._hi.copy()

    def certify(self, ctx, st):
        import time
        t0 = time.perf_counter()
        if st.t < self._last_t:
            raise RuntimeError("checkpoints must be visited in increasing order")
        self._last_t = st.t
        lo, hi = self._box(ctx, st)
        Ml, Mh = self._Ml, self._Mh
        n = np.asarray(st.n)
        N = np.asarray(st.N)
        envs = {}

        def cell_fn(s, a, u):
            if (s, a) not in envs:
                envs[(s, a)] = CellEnvelope(n[s, a], N[s, a], lo[s, a], hi[s, a], ratio=self.ratio, tol=self.tol,
                                            M_lo=Ml[s, a], M_hi=Mh[s, a])
            return envs[(s, a)](u)

        mu = st.mu_hat
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        beta = self.ledger["beta"]
        cache = {}
        out = []
        for q in range(ctx.Q):
            ih = int(ihs[q])
            if ih not in cache:
                wd = direction_widths_fn(ctx, ih, n, N, lo, hi, beta, cell_fn, self.grid)
                cache[ih] = (J - J[ih]) + wd
            Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        self.timing.append(time.perf_counter() - t0)
        self.bb_unconverged += sum(not e.converged for e in envs.values())
        return out

    def describe(self):
        d = super().describe()
        d.update({"cell_mgf": "exact hypergeometric envelope over the HG variance box, min Bennett-FPC sup",
                  "bb_tol": self.tol, "u_ratio": self.ratio,
                  "guarantee": "FWER <= 0.05 at the K pre-specified checkpoints (Theorem FDC-HG-1)"})
        return d
