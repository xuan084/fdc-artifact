"""FDC-bet exploration (round 5 follow-up, R2): direction-level FDC certificates with tight finite-population
cell components.  NEW file; v5-frozen and v6 modules are imported, never modified.

Family name "FDC-bet" is the tentative name from the action plan; the constructions evaluated here are fixed-time
(checkpoint) Chernoff certificates, which dominate fixed-lambda betting at the K checkpoints (see
plan/fdc_bet_exploration.md s2.4).  Every variant keeps FDC's design (frozen 50/50 schedule, identical digest), FDC's
union (sum_q |Pi_{B_q}| x K direction events) and FDC's certification rule (U_q(k) <= eps at the K checkpoints, sticky
answers, stop at 12/15).  Only the per-direction width changes.

Setting (fdc_theorem s2, Fact 0): conditional on the outcome-free schedule A, the cells are independent uniform WoR
samples of fixed sizes n_c(k) from binary pools (N_c, M_c = N_c mu_c).  For a direction a (challenger pi', centre
pi_hat; coefficient -w_s on the challenger cell (s, pi'(s)) and +w_s on the centre cell (s, pi_hat(s)) of every
differing segment s) we need an upper bound on the upper tail of  Y = sum_c a~_c (mu_hat_c - mu_c).

Cell log-MGF bounds  Psi_c(u; m) >= log E exp(u (mu_hat_c - m))  (pool mean m, n = n_c, N = N_c):
  * 'bennett'  (Bennett with the EXACT finite-population variance).  The hypergeometric law of X_c = n mu_hat_c is a
               Poisson-binomial law (its pgf has only real zeros; Vatutin & Mikhailov 1982, Pitman 1997 s3), i.e. X_c
               is a sum of n independent Bernoulli(p_i) with sum_i p_i(1-p_i) = Var X_c = n m (1-m) (N-n)/(N-1).
               Bennett's lemma for each centred Bernoulli (|B - p| <= 1) gives
                   Psi(u; m) = n m (1-m) (N-n)/(N-1) * phi(|u|/n),   phi(x) = e^x - 1 - x.
  * 'kl'       Hoeffding (1963, Thm 4) convex order WoR <=_cx WR, binomial MGF (no FPC):
                   Psi(u; m) = n [log(1 - m + m e^{u/n}) - m u/n].
  * 'min'      pointwise minimum of the two (both are valid upper bounds, so is their minimum).
  * 'bernstein' (ablation) Bernstein form of the Bennett bound, phi(x) <= x^2 / (2 (1 - x/3)), giving the closed-form
               width sqrt(2 beta V) + b beta / 3 with V the (optionally FPC-corrected) variance -- with fpc=False and the
               L2 variance box this is EXACTLY FDC (checked by unit test).
Exhausted cells (n = N) are exact (Psi = 0); n = 0 -> +inf.

Unknown m: variance event E_var = {mu_c in B_c(k) for all c, k} with B_c(k) a two-sided per-(cell, checkpoint) box:
  'HG' exact hypergeometric test inversion (rect_v6.hg_mean_interval), alpha_side = delta_var / (2 S A K), running
       intersection over checkpoints (valid on the simultaneous event);
  'L2' FDC's Lemma L2 Bernstein inversion at x_v = ln(2 S A K / delta_var) (no running intersection, as in FDC).
On E_var, Psi_c(u; mu_c) <= Psi_bar_c(u) := sup_{m in B_c} Psi_c(u; m) (closed form: both bounds are concave in m;
for 'min' we use min(sup bennett, sup kl) >= sup min).

Width (any lambda > 0, data-dependent lambda allowed, see the proof in the exploration note s2):
    w(pi', pi_hat) = min_{lambda in Lambda(pair)} [beta + sum_c Psi_bar_c(lambda a~_c)] / lambda,
Lambda(pair) = lambda_0 * geometric grid, lambda_0 = sqrt(2 beta / Var_bar(pair)).  Empty difference set -> 0.

Optional free hybrid ('rect=True'): on E_var the rectangle bound sum_s w_s (hi[s, pi'(s)] - lo[s, pi_hat(s)]) also
holds, so U uses min(Delta_hat + w, rect) per challenger at no delta cost.

Ledger: beta = ln(sum_q |Pi_{B_q}| K / delta_main), delta_main + delta_var = delta = 0.05 (default split 0.045 /
0.005 = FDC's).  FWER <= delta at the K checkpoints, both observation windows (fdc_theorem s6).
"""
from __future__ import annotations

import math

import numpy as np

from .b4_bal import balanced_alloc
from .fdc import union_size
from .frontier_common import Method, jhat_all, pi_hat_indices
from .rect_v6 import hg_mean_interval

__all__ = ["FDCBet", "psi_bennett_fpc", "psi_kl", "psi_bar", "kl_argmax_m", "direction_widths", "bet_ledger",
           "VARIANTS", "make_variant", "FDC_BET_DELTA"]

FDC_BET_DELTA = 0.05
_LAMBDA_GRID = np.exp(np.linspace(math.log(1.0 / 200.0), math.log(10.0), 161))   # ratio ~1.069


# =============================================================================================== cell log-MGF bounds
def _phi(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.expm1(x) - x


def psi_bennett_fpc(u, n, N, m):
    """Bennett bound with exact FPC variance on log E exp(u (mu_hat - m)); arrays broadcast. n >= 1, n < N assumed."""
    n = np.asarray(n, dtype=float)
    N = np.asarray(N, dtype=float)
    m = np.asarray(m, dtype=float)
    varX = n * m * (1.0 - m) * (N - n) / np.maximum(N - 1.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        out = varX * _phi(np.abs(u) / n)
    return np.where(varX <= 0, 0.0, out)


def psi_kl(u, n, m):
    """Binomial (with-replacement) log-MGF of the mean, numerically stable for m in [0, 1]."""
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    s = np.asarray(u, dtype=float) / n
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lse = np.logaddexp(np.log1p(-m), np.log(m) + s)          # log(1 - m + m e^s)
        out = n * (lse - s * m)
    out = np.where((m <= 0) | (m >= 1), 0.0, out)
    return np.maximum(out, 0.0)                                  # true value >= 0 (Jensen); guards round-off


def kl_argmax_m(s):
    """argmax_m of the binomial term (concave in m): m* = (e^s - 1 - s) / (s (e^s - 1)), m*(0) = 1/2."""
    s = np.asarray(s, dtype=float)
    small = np.abs(s) < 1e-6
    ss = np.where(small, 1.0, s)
    with np.errstate(over="ignore", invalid="ignore"):
        em1 = np.expm1(ss)
        ms = (em1 - ss) / (ss * em1)
    ms = np.where(np.isfinite(ms), ms, np.where(ss > 0, 0.0, 1.0))
    return np.where(small, 0.5 - s / 12.0, np.clip(ms, 0.0, 1.0))


def psi_bar(u, n, N, lo, hi, kind):
    """sup_{m in [lo, hi]} Psi(u; m) for the chosen cell bound ('bennett' | 'kl' | 'min'). Live cells only."""
    mstar_v = np.clip(0.5, lo, hi)
    if kind in ("bennett", "min"):
        b = psi_bennett_fpc(u, n, N, mstar_v)
        if kind == "bennett":
            return b
    mk = np.clip(kl_argmax_m(np.asarray(u, dtype=float) / np.asarray(n, dtype=float)), lo, hi)
    k = psi_kl(u, n, mk)
    if kind == "kl":
        return k
    return np.minimum(b, k)


# =============================================================================================== ledger
def bet_ledger(ctx, delta_main=0.045, delta_var=0.005, var_box="HG"):
    K = int(len(ctx.checkpoints))
    SA = int(ctx.S * ctx.A)
    m = union_size(ctx, "feas")
    beta = math.log(m * K / delta_main)
    led = {"union_size": m, "K": K, "S_A": SA, "delta_main": delta_main, "delta_var": delta_var,
           "delta_total": delta_main + delta_var, "beta": beta, "bound_main": m * K * math.exp(-beta),
           "var_box": var_box}
    if var_box == "HG":
        led["alpha_side_var"] = delta_var / (2.0 * SA * K)
        led["bound_var"] = 2.0 * SA * K * led["alpha_side_var"]
    else:
        led["x_v"] = math.log(2 * SA * K / delta_var)
        led["bound_var"] = 2.0 * SA * K * math.exp(-led["x_v"])
    return led


# =============================================================================================== widths
def direction_widths(ctx, ih, n, N, lo, hi, beta, kind, fpc=True, grid=_LAMBDA_GRID):
    """Width w(pi', pi_hat) for every pi' in ctx.pols (vector over P), centre pi_hat = pols[ih].

    kind: 'bennett' | 'kl' | 'min' (Chernoff over a lambda grid) or 'bernstein' (closed form sqrt(2 beta V) + b beta/3;
    with fpc=False and the L2 box this is FDC's pair width)."""
    S = ctx.S
    seg = np.arange(S)
    pols = ctx.pols
    ph = pols[ih]
    D = pols != ph[None, :]                                      # (P, S)
    n = np.asarray(n, dtype=float)
    N = np.asarray(N, dtype=float)
    live = (n > 0) & (n < N)
    zero = n <= 0
    w = ctx.w
    # per-cell variance of mu_hat on the box (FPC optional)
    vmax = np.clip(0.5, lo, hi)
    vmax = vmax * (1.0 - vmax)
    fp = (N - n) / np.maximum(N - 1.0, 1.0) if fpc else np.ones_like(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        vcell = np.where(live, vmax * fp / np.maximum(n, 1.0), 0.0)          # Var(mu_hat_c) upper bound
    # gather challenger / centre cells for each (p, s)
    a_ch = pols                                                  # arm of the challenger cell
    a_h = np.broadcast_to(ph[None, :], pols.shape)
    zero_touch = (D & (zero[seg[None, :], a_ch] | zero[seg[None, :], a_h])).any(1)
    w2 = (w ** 2)[None, :]
    Vpair = (D * w2 * (vcell[seg[None, :], a_ch] + vcell[seg[None, :], a_h])).sum(1)     # (P,)
    out = np.zeros(pols.shape[0])
    beta = np.broadcast_to(np.asarray(beta, dtype=float), (pols.shape[0],))
    if kind == "bernstein":
        inv_n = np.where(live, 1.0 / np.maximum(n, 1.0), 0.0)
        bc = w[:, None] * inv_n                                  # |a_c| / n_c on live cells
        b = (D * np.maximum(bc[seg[None, :], a_ch], bc[seg[None, :], a_h])).max(1)
        out = np.sqrt(2.0 * beta * Vpair) + b * beta / 3.0
    else:
        act = Vpair > 0
        if act.any():
            Pi = np.flatnonzero(act)
            lam0 = np.sqrt(2.0 * beta[Pi] / Vpair[Pi])           # (P',)
            Lam = lam0[:, None] * grid[None, :]                  # (P', G)
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
                            continue                             # exhausted: exact, Psi = 0 (n = 0 handled below)
                        u = sign * Lam[r] * w[s]
                        F[r] += psi_bar(u, n[s, a], N[s, a], lo[s, a], hi[s, a], kind)
            with np.errstate(invalid="ignore", over="ignore"):
                val = (beta[Pi][:, None] + F) / Lam
            val = np.where(np.isfinite(val), val, np.inf)
            out[Pi] = val.min(1)
    out = np.where(D.any(1), out, 0.0)
    out = np.where(zero_touch, np.inf, out)
    return out


# =============================================================================================== method
VARIANTS = {
    # name                 kind         box    fpc    rect   (delta_main, delta_var)
    "FDC-re":            ("bernstein", "L2", False, False, (0.045, 0.005)),   # re-implementation of FDC (check)
    "FDC-FPC":           ("bernstein", "L2", True,  False, (0.045, 0.005)),   # FDC + finite-population correction
    "FDC-BF":            ("bennett",   "HG", True,  False, (0.045, 0.005)),   # Bennett-FPC + exact HG variance box
    "FDC-KLF":           ("min",       "HG", True,  False, (0.045, 0.005)),   # min(Bennett-FPC, binomial KL) + HG box
    "FDC-KLF+R":         ("min",       "HG", True,  True,  (0.045, 0.005)),   # + free rectangle on E_var
    "FDC-BF-L2":         ("bennett",   "L2", True,  False, (0.045, 0.005)),   # Bennett-FPC with FDC's L2 box
}


class FDCBet(Method):
    """Direction-level FDC certificate with a configurable cell inequality (see module docstring)."""

    alloc_kind = "fixed"
    validity = "rigorous"
    cs_kind = "none"

    def __init__(self, kind="min", var_box="HG", fpc=True, rect=False, split=(0.045, 0.005), name=None,
                 grid=None):
        if kind not in ("bernstein", "bennett", "kl", "min"):
            raise ValueError(kind)
        if var_box not in ("HG", "L2"):
            raise ValueError(var_box)
        if abs(split[0] + split[1] - FDC_BET_DELTA) > 1e-12 or min(split) <= 0:
            raise ValueError("split must be positive and sum to delta = 0.05")
        self.kind, self.var_box, self.fpc, self.rect = kind, var_box, bool(fpc), bool(rect)
        self.split = (float(split[0]), float(split[1]))
        self.grid = _LAMBDA_GRID if grid is None else np.asarray(grid, dtype=float)
        self.name = name or f"FDC-bet[{kind},{var_box},fpc={int(self.fpc)},rect={int(self.rect)}," \
                            f"{split[0]:g}/{split[1]:g}]"
        self.alloc_p = None
        self.ledger = None

    def setup(self, ctx):
        if abs(float(ctx.delta) - FDC_BET_DELTA) > 1e-12:
            raise ValueError("FDC-bet: delta fixed at 0.05")
        if not ctx.binary or ctx.A != 2:
            raise ValueError("FDC-bet: binary pools, A = 2")
        self.alloc_p = balanced_alloc(ctx.S, ctx.A, 0.5)
        self.ledger = bet_ledger(ctx, self.split[0], self.split[1], self.var_box)
        self._lo = np.zeros((ctx.S, ctx.A))
        self._hi = np.ones((ctx.S, ctx.A))
        self._last_t = -1

    def _box(self, ctx, st):
        n, N, s = st.n, st.N, st.sum
        if self.var_box == "HG":
            lo, hi = hg_mean_interval(N, n, s, self.ledger["alpha_side_var"])
            self._lo = np.maximum(self._lo, lo)                  # running intersection (simultaneous event)
            self._hi = np.maximum(np.minimum(self._hi, hi), self._lo)
            return self._lo.copy(), self._hi.copy()
        from ..theory_checks.mc_l1 import bernstein_mu_ci
        lo, hi = bernstein_mu_ci(st.mu_hat, n, np.asarray(N, dtype=float), self.ledger["x_v"])
        return lo, hi

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("checkpoints must be visited in increasing order")
        self._last_t = st.t
        lo, hi = self._box(ctx, st)
        mu = st.mu_hat
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        beta = self.ledger["beta"]
        cache = {}
        out = []
        for q in range(ctx.Q):
            ih = int(ihs[q])
            if ih not in cache:
                wd = direction_widths(ctx, ih, st.n, st.N, lo, hi, beta, self.kind, self.fpc, self.grid)
                U = (J - J[ih]) + wd
                if self.rect:
                    seg = np.arange(ctx.S)
                    ph = ctx.pols[ih]
                    D = ctx.pols != ph[None, :]
                    rect = (ctx.w[None, :] * np.where(D, hi[seg[None, :], ctx.pols] - lo[seg, ph][None, :], 0.0)).sum(1)
                    U = np.minimum(U, rect)
                cache[ih] = U
            Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        return out

    def describe(self):
        d = super().describe()
        d.update({"kind": self.kind, "var_box": self.var_box, "fpc": self.fpc, "rect": self.rect,
                  "split": self.split, "ledger": self.ledger,
                  "design": "frozen 50/50 (balanced_alloc 0.5; identical schedule to FDC)",
                  "guarantee": "FWER <= 0.05 at the K pre-specified checkpoints (Theorem FDC-bet-1)"})
        return d


def make_variant(name):
    kind, box, fpc, rect, split = VARIANTS[name]
    return FDCBet(kind=kind, var_box=box, fpc=fpc, rect=rect, split=split, name=name)
