"""B3: RAGE (Fiez, Jain, Jamieson & Ratliff, NeurIPS 2019), round-4 frontier adaptation, exact over Pi_all.

Also hosts the phased-elimination machinery shared with Peace (``peace_frontier``).

Encoding (transductive linear bandit on the cells)
--------------------------------------------------
theta = mu in R^{S A}; policy pi <-> z_pi with z_pi[s, pi(s)] = w_s; J(pi) = z_pi . mu. Segment arrivals are exogenous,
so a design is the within-segment split p (product of simplices); a cell's share of arrivals is lambda_{s,a} = w_s p_{s,a}
and A(lambda) = diag(lambda / sigma^2). For a pair (z, z'):
    ||z - z'||^2_{A(lambda)^{-1}} = V_p(z, z') = sum_{s: z(s) != z'(s)} w_s (sigma^2_{s,z(s)} / p_{s,z(s)} + sigma^2_{s,z'(s)} / p_{s,z'(s)})
(per arrival; divide by the number of arrivals N). Pairs are de-duplicated by their cell-incidence signature
(``pair_signatures``), which is exact.

RAGE (transcribed constants, idea/lit_diff_r4.md s3.2)
------------------------------------------------------
  round t = 1, 2, ...;  delta_t = delta / t^2 (x 1/Q: union over the Q problems sharing one stream -- deviation);
  lambda_t = XY-optimal design over Y(Z_t) (Frank-Wolfe, step 2/(k+2), stop at relative l2 change < 0.01 or 1000
             iterations, lambda < 1e-5 -> 0 on unused cells);  rho(Y(Z_t)) = max_{z,z' in Z_t} V_p(z, z');
  N_t = max{ ceil(8 (2^{t+1})^2 rho (1 + eps_r) log(|Z|^2 / delta_t)), r(eps_r) },  eps_r = 1/5,
        r(eps_r) = (d (d + 1) / 2 + 1) / eps_r,  d = S A;
  theta_hat_t from the round-t samples ONLY; eliminate z if exists z': (z' - z) . theta_hat_t >= 2^{-(t+2)}.
  On the good event every survivor of round t has gap < 2^{-(t+1)} (gap <= R = 1 by construction), so the eps-good
  frontier adaptation certifies problem q when 2^{-(t+1)} <= eps or |Z_q| = 1, answer = empirical best survivor.

Multi-problem frontier adaptation (written into the appendix)
  * every problem keeps its own active set Z_q (start: Pi_{B_q}, enumerated); arrivals are shared;
  * the round allocation is the uniform mixture of the per-problem designs over the problems still in play;
  * N_t is computed with each problem's rho evaluated AT THE MIXTURE (so every problem's own error bound holds with the
    allocation actually played) and the max over problems is taken;
  * the round length is rounded up to a multiple of the lock re-planning interval and is at least A x replan (so the
    within-round tracking can visit every arm of a segment); the split is realised by within-round C-tracking
    (target arm = argmax_a p_{s,a} x (segment-s round arrivals + expected batch) - n^round_{s,a});
  * pool exhaustion: re-selection by descending deficit (engine); a cell with no round-t sample is 'unknown' and the
    elimination test uses the conservative bounds 0 / R for unknown cells, so it can only eliminate less.

Variants (methodology R3)
  * -nominal: sigma^2 = 1/4 proxy + the paper constants above. "nominal": valid for i.i.d. Gaussian rounds, unproven
              under WoR + exogenous-arrival tracking.
  * -fav    : plug-in variance mu_hat (1 - mu_hat) (cumulative) + the same constants. No validity claimed.
  * -rect   : the -fav phased design / elimination as the SAMPLING rule; certification and stopping by the per-cell
              WSR20 CS rectangle (``rect_certificate``). Rigorous. For an undecided problem whose active set has
              collapsed the design set is {pi_hat_q, rect challenger_q} (so sampling keeps serving it).
"""
from __future__ import annotations

import math

import numpy as np

from .frontier_common import Method, jhat_all, pi_hat_indices, plugin_var, rect_certificate, rect_U

__all__ = ["pair_signatures", "pair_V", "xy_design", "PhasedElimination", "RAGEFrontier", "VARIANTS",
           "LAM_ZERO", "FW_ITERS", "FW_TOL"]

VARIANTS = ("rect", "nominal", "fav")
LAM_ZERO = 1e-5
FW_ITERS = 1000
FW_TOL = 0.01
RAGE_EPS_ROUND = 0.2
_SIG_CACHE: dict = {}


# =============================================================================================== pair geometry
def _pair_types(A):
    types = {}
    for a in range(A):
        for b in range(a + 1, A):
            types[(a, b)] = len(types) + 1
    T = np.zeros((A, A), dtype=np.int64)
    for (a, b), k in types.items():
        T[a, b] = T[b, a] = k
    return T, types


def pair_signatures(rows, A, block=256):
    """Unique cell-incidence signatures of all unordered pairs of distinct rows (m, S) -> bool (U, S A).

    Signature of (z, z'): cell (s, a) is included iff z(s) != z'(s) and a in {z(s), z'(s)}. Exact de-duplication."""
    rows = np.asarray(rows, dtype=np.int64)
    m, S = rows.shape
    key = (A, rows.tobytes())
    if key in _SIG_CACHE:
        return _SIG_CACHE[key]
    if m < 2:
        out = np.zeros((0, S * A), dtype=bool)
        _SIG_CACHE[key] = out
        return out
    T, types = _pair_types(A)
    base = len(types) + 1
    if base ** S >= 2 ** 62:
        raise ValueError("signature code overflow")
    pw = base ** np.arange(S - 1, -1, -1, dtype=np.int64)
    codes = []
    for i0 in range(0, m, block):
        blk = rows[i0:i0 + block]
        t = T[blk[:, None, :], rows[None, :, :]]                      # (b, m, S)
        c = (t * pw).sum(-1)
        jj = np.arange(m)[None, :]
        ii = (i0 + np.arange(blk.shape[0]))[:, None]
        codes.append(np.unique(c[jj > ii]))
    u = np.unique(np.concatenate(codes))
    u = u[u != 0]
    inv = {k: ab for ab, k in types.items()}
    digits = (u[:, None] // pw[None, :]) % base                          # (U, S)
    inc = np.zeros((u.size, S, A), dtype=bool)
    for k, (a, b) in inv.items():
        sel = digits == k
        inc[:, :, a] |= sel
        inc[:, :, b] |= sel
    out = inc.reshape(u.size, S * A)
    if len(_SIG_CACHE) > 4096:
        _SIG_CACHE.clear()
    _SIG_CACHE[key] = out
    return out


def cell_cost(w, var, p):
    """c_{s,a} = w_s sigma^2_{s,a} / p_{s,a} (inf where p = 0)."""
    with np.errstate(divide="ignore"):
        return np.where(p > 0, np.asarray(w)[:, None] * var / np.where(p > 0, p, 1.0), np.inf)


def pair_V(inc, w, var, p):
    """V_p for every signature (vector over U); 0 x inf -> 0 for cells not in the signature."""
    c = cell_cost(w, var, p).ravel()
    return np.where(inc, c[None, :], 0.0).sum(1) if inc.size else np.zeros(0)


def _support(inc, S, A):
    return inc.any(0).reshape(S, A) if inc.size else np.zeros((S, A), dtype=bool)


def _finalise(p, used):
    """lambda < 1e-5 -> 0 on cells no pair uses; used cells keep a 1e-5 floor; rows renormalised."""
    p = np.where(used, np.maximum(p, LAM_ZERO), np.where(p < LAM_ZERO, 0.0, p))
    s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1.0), 1.0 / p.shape[1])


def xy_design(inc, w, var, iters=FW_ITERS, tol=FW_TOL, temp=50.0, p0=None):
    """Frank-Wolfe XY design: minimise_p max_u V_p(u) over the product of simplices (smoothed max, temperature 50).
    Returns (p, rho = max_u V_p(u))."""
    w = np.asarray(w, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1e-12)
    S, A = var.shape
    p = np.full((S, A), 1.0 / A) if p0 is None else np.asarray(p0, dtype=float).copy()
    if inc.shape[0] == 0:
        return p, 0.0
    used = _support(inc, S, A)
    p = _finalise(p, used)
    incf = inc.astype(float)
    for k in range(iters):
        c = cell_cost(w, var, p)
        f = incf @ np.where(np.isfinite(c), c, 0.0).ravel()
        fmax = f.max()
        z = np.exp(temp * (f / fmax - 1.0))
        wt = z / z.sum()
        g = -(wt @ incf).reshape(S, A) * w[:, None] * var / np.maximum(p, LAM_ZERO) ** 2
        v = np.full((S, A), LAM_ZERO)
        v[np.arange(S), np.argmin(g, 1)] = 1.0 - LAM_ZERO * (A - 1)
        gamma = 2.0 / (k + 2.0)
        p_new = (1 - gamma) * p + gamma * v
        rel = np.linalg.norm(p_new - p) / np.linalg.norm(p)
        p = p_new
        if k > 10 and rel < tol:
            break
    p = _finalise(p, used)
    return p, float(pair_V(inc, w, var, p).max())


def policy_rows_value(ctx, rows_idx, mu):
    return jhat_all(ctx, mu)[rows_idx]


# =============================================================================================== phased elimination
class PhasedElimination(Method):
    """Shared round structure of RAGE / Peace on the frontier (see module docstring)."""
    alloc_kind = "adaptive"
    family = "phased"

    def __init__(self, variant="rect", name=None):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.name = name or f"{self.family}-{variant}"
        self.validity = {"rect": "rigorous", "nominal": "nominal", "fav": "none"}[variant]

    # ---------------------------------------------------------------- hooks (family specific)
    def _design(self, ctx, st, q, rows_idx, var, k):
        raise NotImplementedError

    def _round_size(self, ctx, st, q, rows_idx, var, p_mix, k):
        raise NotImplementedError

    def _thr(self, k):
        raise NotImplementedError

    def _stop_scale(self, k):
        raise NotImplementedError

    def _q_min(self, ctx):
        raise NotImplementedError

    # ---------------------------------------------------------------- shared
    def setup(self, ctx):
        self.Z = [np.flatnonzero(ctx.feas[q]) for q in range(ctx.Q)]
        self.Z1 = [z.size for z in self.Z]
        self.done = np.zeros(ctx.Q, dtype=bool)
        self.ans = np.full(ctx.Q, -1, dtype=np.int64)
        self.k = 0
        self.t_end = -1
        self.n0 = np.zeros((ctx.S, ctx.A), dtype=np.int64)
        self.s0 = np.zeros((ctx.S, ctx.A))
        self.p = np.full((ctx.S, ctx.A), 1.0 / ctx.A)
        self.log = []
        self.d = ctx.S * ctx.A

    def _var(self, ctx, st):
        if self.variant == "nominal":
            return np.full((ctx.S, ctx.A), 0.25 * ctx.R * ctx.R)
        return np.where(st.n > 0, plugin_var(st, ctx.R), 0.25 * ctx.R * ctx.R)

    def _design_set(self, ctx, st, q):
        Zq = self.Z[q]
        if self.variant != "rect":
            return Zq
        mu = st.mu_hat
        ih = pi_hat_indices(ctx, mu)[q]
        lo, hi = st.cs()
        U = np.where(ctx.feas[q], rect_U(ctx, lo, hi, ih), -np.inf)
        ch = int(np.argmax(U))
        return np.unique(np.concatenate([Zq, [ih, ch]]))

    def _in_play(self, st):
        if self.variant == "rect":
            return np.flatnonzero(st.undecided)
        return np.flatnonzero(~self.done)

    def _end_round(self, ctx, st):
        dn = st.n - self.n0
        known = dn > 0
        th = np.where(known, (st.sum - self.s0) / np.maximum(dn, 1), 0.0)
        lo_v = np.where(known, th, 0.0)
        hi_v = np.where(known, th, ctx.R)
        Jlo, Jhi = jhat_all(ctx, lo_v), jhat_all(ctx, hi_v)
        thr = self._thr(self.k) * ctx.R
        Jcum = jhat_all(ctx, st.mu_hat)
        n_elim = 0
        for q in range(ctx.Q):
            if self.done[q]:
                continue
            Zq = self.Z[q]
            keep = Jhi[Zq] - Jlo[Zq].max() > -thr          # eliminate z iff max_z' Jlo(z') - Jhi(z) >= thr
            n_elim += int((~keep).sum())
            self.Z[q] = Zq[keep] if keep.any() else Zq[[int(np.argmax(Jlo[Zq]))]]
            if self.Z[q].size == 1 or self._stop_scale(self.k) * ctx.R <= ctx.eps:
                self.done[q] = True
                self.ans[q] = int(self.Z[q][int(np.argmax(Jcum[self.Z[q]]))])
        self.log[-1].update({"t_end_actual": int(st.t), "eliminated": n_elim,
                             "active_sizes": [int(z.size) for z in self.Z], "done": int(self.done.sum())})

    def _start_round(self, ctx, st):
        self.k += 1
        k = self.k
        var = self._var(ctx, st)
        play = self._in_play(st)
        sets = {int(q): self._design_set(ctx, st, q) for q in play}
        sets = {q: r for q, r in sets.items() if r.size >= 2}
        if sets:
            ps = [self._design(ctx, st, q, r, var, k) for q, r in sets.items()]
            p_mix = np.mean(ps, 0)
            sizes = {q: self._round_size(ctx, st, q, r, var, p_mix, k) for q, r in sets.items()}
            N = max(sizes.values())
        else:
            p_mix, N, sizes = self.p, 0, {}
        self.p = p_mix
        L = max(int(N), int(math.ceil(self._q_min(ctx))), ctx.A * ctx.replan)
        L = int(math.ceil(L / ctx.replan) * ctx.replan)
        self.t_end = st.t + L
        self.n0 = st.n.copy()
        self.s0 = st.sum.copy()
        self.log.append({"k": k, "t_start": int(st.t), "N_k": int(N), "len": L, "n_design": len(sets),
                         "thr": self._thr(k), "stop_scale": self._stop_scale(k),
                         "N_by_q_max_q": int(max(sizes, key=sizes.get)) if sizes else -1})

    def plan(self, ctx, st):
        if self.k == 0 or st.t >= self.t_end:
            if self.k > 0:
                self._end_round(ctx, st)
            self._start_round(ctx, st)
        dn = (st.n - self.n0).astype(float)
        seg_round = dn.sum(1)
        target = self.p * (seg_round + ctx.w * ctx.replan)[:, None]
        deficit = target - dn
        return np.argsort(-deficit, axis=1, kind="stable").astype(np.int64)

    def certify(self, ctx, st):
        if self.variant == "rect":
            return rect_certificate(ctx, st)
        ih = pi_hat_indices(ctx, st.mu_hat)
        sc = self._stop_scale(max(self.k - 1, 1))
        return [(True, int(self.ans[q]), float(sc)) if self.done[q] else (False, int(ih[q]), float("inf"))
                for q in range(ctx.Q)]

    def describe(self):
        d = super().describe()
        d.update({"variant": self.variant, "rounds": getattr(self, "log", [])})
        return d


class RAGEFrontier(PhasedElimination):
    family = "B3"

    def __init__(self, variant="rect", name=None, eps_round=RAGE_EPS_ROUND, fw_iters=FW_ITERS):
        super().__init__(variant, name)
        self.eps_round = eps_round
        self.fw_iters = fw_iters

    def _design(self, ctx, st, q, rows_idx, var, k):
        inc = pair_signatures(ctx.pols[rows_idx], ctx.A)
        p, _ = xy_design(inc, ctx.w, var, iters=self.fw_iters)
        return p

    def _delta_k(self, ctx, k):
        return ctx.delta / (k * k) / ctx.Q

    def _round_size(self, ctx, st, q, rows_idx, var, p_mix, k):
        inc = pair_signatures(ctx.pols[rows_idx], ctx.A)
        rho = float(pair_V(inc, ctx.w, var, p_mix).max())
        dk = self._delta_k(ctx, k)
        return int(math.ceil(8 * (2.0 ** (k + 1)) ** 2 * rho * (1 + self.eps_round) *
                             math.log(self.Z1[q] ** 2 / dk) / ctx.R ** 2))

    def _thr(self, k):
        return 2.0 ** (-(k + 2))

    def _stop_scale(self, k):
        return 2.0 ** (-(k + 1))

    def _q_min(self, ctx):
        d = ctx.S * ctx.A
        return (d * (d + 1) / 2 + 1) / self.eps_round
