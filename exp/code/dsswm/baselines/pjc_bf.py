"""PJC-BF: phased adaptive joint certificate (round-8 rival requested by the v7 reviews, P0-3).  NEW file; v5/v6/v7
modules are imported, never modified.

Both v7 reviews asked for the strongest legitimate guaranteed rival that FDC-BF's own theorem covers: a RAGE / Peace
style phased design whose read plan is frozen per phase from past data only, with the Bennett-FPC joint
(direction-level) width.  Two constructions are provided; both are valid at the K pre-specified checkpoints with
FWER <= delta = 0.05 (same guarantee class as FDC-BF, RECT-ck-HG):

mode='local'  (PJC-BF, RAGE-style; the construction the critic describes)
    Phase boundaries are a data-free set of checkpoint indices b_1 < b_2 < ...; phase j covers the arrivals after
    T_j = ck[b_j] (T_0 = 0).  At T_j the plan for phase j (target within-segment split p_j[s, a]) is computed from the
    data observed up to T_j only, and arms are assigned by a deterministic deficit-tracking rule that uses the plan and
    the phase-j COUNTS only (never sums), so the phase-j counts n'_c(k) are functions of (F_{T_j}, future arrival
    sequence).  Conditional on that, the phase-j draws of cell c are a fixed-size uniform WoR sample from the
    remaining pool (N'_c = N_c - n_c(T_j), remaining mean m'_c = (M_c - S_c(T_j)) / N'_c), independent over cells
    (exchangeability of the unread part of a uniform permutation at a stopping time + independence of the cells and of
    the label-only arrival sequence).  The phase-local estimator
        mu_hat_c^(j) = (S_c(T_j) + N'_c * m_hat'_c) / N_c
    has error (N'_c / N_c) (m_hat'_c - m'_c), so Theorem FDC-bet-1 applies verbatim with pools (N'_c, m'_c) and cell
    coefficients a~_c N'_c / N_c.  Union: each checkpoint lies in exactly one phase, so the ledger is FDC-BF's
    (beta = ln(sum_q |Pi_{B_q}| K / delta_main), alpha_side = delta_var / (2 S A K)).  Variance boxes are exact HG
    inversions on the phase-local sample, mapped to mu-space and intersected over all checkpoints (valid on the
    simultaneous event), then mapped back to m'-space.  Statistical information from earlier phases is used only
    through the exactly known prefix sums S_c(T_j) (as in RAGE, which discards earlier-phase samples).

mode='menu'   (PJC-BF-M, all-data path union; a stronger construction we add so that the rival is not handicapped by
    discarding data)
    At each boundary the method chooses one of M data-free menu designs (control share c in `menu`).  For every
    menu path p (sequence of choices) the counts n^p_c(k) are functions of the arrival sequence only, so the GLOBAL
    estimator S_c / n_c under path p is a fixed-size WoR sample (FDC-bet-1 applies).  The realised path is data
    dependent, so the main and variance events are charged for every (path, checkpoint) pair:
        L = sum_k M^{j(k)},   j(k) = #{boundaries b < k},
        beta = ln(sum_q |Pi_{B_q}| * L / delta_main),   alpha_side = delta_var / (2 S A L).
    With M = 1 (or no boundary) this is FDC-BF up to the deterministic-tracking design.

Design rules (computed at T_j from data up to T_j; any F_{T_j}-measurable rule is legal):
  'half'    50/50 deterministic tracking (local mode: isolates the cost of the phase reset).
  'neyman'  remaining-weight Neyman split p[s, a] proportional to (N'_c / N_c) * sd_c, sd_c from the Laplace plug-in mean
            (S_c + 1) / (n_c + 2); floor p_floor per arm.  For the phase-local estimator the variance of segment s is
            sum_a (N'_a/N_a)^2 v_a (1/n'_a - 1/N'_a) (up to the (N'-1) factor), minimised at n'_a ~ (N'_a/N_a) sd_a.
  'proj'    (menu mode) the menu share minimising the projected global variance sum_s w_s^2 sum_a v_a (1/n_proj - 1/N)
            at horizon min(tau_R, 2 t).
Phase 0 always uses the 50/50 tracking design (no data yet).

Arm assignment (engine 'adaptive' mode: one arm per segment per replan batch; exhausted -> other arm): in segment s
the arm with the largest deficit p_j[s, a] * (m_s + 1) - n'_{s, a} (m_s = phase-j reads of segment s; ties -> lower
arm index) gets the batch.  Uses counts and the stored plan only.

Also here:
  RectCkBF        matched Bennett-FPC rectangle (critic P1-1 / external reviewer point 8): the SAME cell inequality as FDC-BF
                  (Bennett with exact FPC variance, sup over the SAME exact-HG variance box at alpha_side =
                  delta_var / (2 S A K)), the SAME delta split (0.045 / 0.005), the SAME frozen 50/50 design; only the
                  aggregation changes: per-cell one-sided Chernoff radius at beta_c = ln(2 S A K / delta_main) and a
                  radius-sum rectangle.  Option box=True also intersects with the (already paid) variance box.
  RectCkHGPlan    RECT-ck-HG with its own frozen read plan (uniform control share or a per-cell matrix, e.g. Neyman
                  frozen from development-half pool variances).
  HCWoRPlan       HC-WoR (v6) with its own frozen read plan.
  FDCBetPlan      FDC-BF with a non-50/50 frozen plan (descriptive only).
"""
from __future__ import annotations

import math

import numpy as np

from .b4_bal import balanced_alloc
from .fdc import union_size
from .fdc_bet import _LAMBDA_GRID, FDCBet, psi_bar
from .frontier_common import Method, jhat_all, pi_hat_indices
from .rect_v6 import RectCkHG, _CkRect, hg_mean_interval, rect_certificate_from
from .wor_betting_v6 import HCWoRRect

__all__ = ["PJCBF", "pjc_ledger", "paths_per_checkpoint", "scaled_direction_widths", "RectCkBF", "RectCkHGPlan",
           "HCWoRPlan", "FDCBetPlan", "neyman_matrix", "PJC_DELTA", "PJC_SPLIT"]

PJC_DELTA = 0.05
PJC_SPLIT = (0.045, 0.005)


# =============================================================================================== ledgers
def paths_per_checkpoint(K, boundaries, M):
    """Number of distinct menu paths that determine the counts at checkpoint k (k = 0..K-1)."""
    b = np.asarray(sorted(boundaries), dtype=int)
    return np.array([int(M) ** int((b < k).sum()) for k in range(K)], dtype=np.int64)


def pjc_ledger(ctx, mode, boundaries, M=1, split=PJC_SPLIT):
    K = int(len(ctx.checkpoints))
    SA = int(ctx.S * ctx.A)
    m = union_size(ctx, "feas")
    if mode == "local":
        L = K
    elif mode == "menu":
        L = int(paths_per_checkpoint(K, boundaries, M).sum())
    else:
        raise ValueError(mode)
    d_main, d_var = split
    beta = math.log(m * L / d_main)
    a = d_var / (2.0 * SA * L)
    return {"mode": mode, "union_size": m, "K": K, "L_events_per_direction": L, "S_A": SA, "delta_main": d_main,
            "delta_var": d_var, "delta_total": d_main + d_var, "beta": beta, "alpha_side_var": a,
            "bound_main": m * L * math.exp(-beta), "bound_var": 2.0 * SA * L * a,
            "boundaries": [int(x) for x in boundaries], "M": int(M)}


# =============================================================================================== widths
def scaled_direction_widths(ctx, ih, n, N, lo, hi, beta, scale, kind="bennett", grid=_LAMBDA_GRID):
    """Chernoff direction width with per-cell coefficient magnitude w_s * scale[s, a] on pools (N, n) whose means lie in
    [lo, hi] (the pools may be remaining pools).  scale == 1 reproduces fdc_bet.direction_widths(..., fpc=True).

    Cells with N == 0 contribute nothing (empty remaining pool: the cell mean is known exactly); live cells
    (0 < n < N) contribute sup_box Psi; n == N cells are exact; n == 0 < N touched -> +inf."""
    S = ctx.S
    seg = np.arange(S)
    pols = ctx.pols
    ph = pols[ih]
    D = pols != ph[None, :]
    n = np.asarray(n, dtype=float)
    N = np.asarray(N, dtype=float)
    coef = np.asarray(ctx.w, dtype=float)[:, None] * np.asarray(scale, dtype=float)
    live = (n > 0) & (n < N)
    zero = (n <= 0) & (N > 0) & (coef > 0)
    vmax = np.clip(0.5, lo, hi)
    vmax = vmax * (1.0 - vmax)
    fp = (N - n) / np.maximum(N - 1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vcell = np.where(live, vmax * fp / np.maximum(n, 1.0), 0.0)
    a_ch = pols
    a_h = np.broadcast_to(ph[None, :], pols.shape)
    zero_touch = (D & (zero[seg[None, :], a_ch] | zero[seg[None, :], a_h])).any(1)
    c2v = coef ** 2 * vcell
    Vpair = (D * (c2v[seg[None, :], a_ch] + c2v[seg[None, :], a_h])).sum(1)
    out = np.zeros(pols.shape[0])
    beta = float(beta)
    act = Vpair > 0
    if kind not in ("bennett", "kl", "min"):
        raise ValueError(kind)
    if act.any():
        Pi = np.flatnonzero(act)
        lam0 = np.sqrt(2.0 * beta / Vpair[Pi])
        Lam = lam0[:, None] * grid[None, :]
        F = np.zeros_like(Lam)
        for s in range(S):
            dsel = D[Pi, s]
            if not dsel.any():
                continue
            rows = np.flatnonzero(dsel)
            for arms, sign in ((a_ch[Pi[rows], s], -1.0), (a_h[Pi[rows], s], +1.0)):
                for a in np.unique(arms):
                    if not live[s, a] or coef[s, a] <= 0:
                        continue
                    r = rows[arms == a]
                    u = sign * Lam[r] * coef[s, a]
                    F[r] += psi_bar(u, n[s, a], N[s, a], lo[s, a], hi[s, a], kind)
        with np.errstate(invalid="ignore", over="ignore"):
            val = (beta + F) / Lam
        val = np.where(np.isfinite(val), val, np.inf)
        out[Pi] = val.min(1)
    out = np.where(D.any(1), out, 0.0)
    out = np.where(zero_touch, np.inf, out)
    return out


def _hg_box_safe(N, n, s, a):
    """hg_mean_interval with N == 0 cells mapped to the dummy box [0, 0]."""
    N = np.asarray(N, dtype=np.int64)
    lo = np.zeros(N.shape)
    hi = np.zeros(N.shape)
    pos = N > 0
    if pos.any():
        l, h = hg_mean_interval(N[pos], np.asarray(n)[pos], np.asarray(s)[pos], a)
        lo[pos], hi[pos] = l, h
    return lo, hi


def neyman_matrix(sigma2, weight=None, floor=0.1):
    """Within-segment Neyman shares p[s, a] ~ weight[s, a] * sd[s, a] with a per-arm floor (rows sum to 1)."""
    sd = np.sqrt(np.maximum(np.asarray(sigma2, dtype=float), 0.0))
    if weight is not None:
        sd = sd * np.asarray(weight, dtype=float)
    S, A = sd.shape
    tot = sd.sum(1, keepdims=True)
    p = np.where(tot > 0, sd / np.where(tot > 0, tot, 1.0), 1.0 / A)
    p = np.clip(p, floor, 1.0 - floor * (A - 1))
    return p / p.sum(1, keepdims=True)


# =============================================================================================== PJC-BF
class PJCBF(Method):
    """Phased adaptive joint certificate with Bennett-FPC widths (see module docstring)."""

    alloc_kind = "adaptive"
    validity = "rigorous"
    cs_kind = "none"

    def __init__(self, mode="local", boundaries=(2,), rule="neyman", rect=False, p_floor=0.1,
                 menu=(0.4, 0.45, 0.5), kind="bennett", name=None):
        if mode not in ("local", "menu"):
            raise ValueError(mode)
        if mode == "local" and rule not in ("half", "neyman"):
            raise ValueError(rule)
        if mode == "menu" and rule not in ("proj", "half"):
            raise ValueError(rule)
        if not 0.0 < p_floor <= 0.5:
            raise ValueError("p_floor in (0, 0.5]")
        self.mode, self.boundaries, self.rule = mode, tuple(sorted(int(b) for b in boundaries)), rule
        self.rect, self.p_floor, self.kind = bool(rect), float(p_floor), kind
        self.menu = tuple(float(x) for x in menu) if mode == "menu" else ()
        if mode == "menu" and any(not 0.0 < x < 1.0 for x in self.menu):
            raise ValueError("menu shares in (0, 1)")
        bnd = ",".join(str(b) for b in self.boundaries)
        self.name = name or (f"PJC-BF[local,b={bnd},{rule}{',R' if rect else ''}]" if mode == "local" else
                             f"PJC-BF-M[b={bnd},menu={'/'.join(f'{x:g}' for x in self.menu)},{rule}"
                             f"{',R' if rect else ''}]")
        self.alloc_p = None
        self.ledger = None

    # ------------------------------------------------------------------ setup
    def setup(self, ctx):
        if abs(float(ctx.delta) - PJC_DELTA) > 1e-12:
            raise ValueError("PJC-BF: delta fixed at 0.05")
        if not ctx.binary or ctx.A != 2:
            raise ValueError("PJC-BF: binary pools, A = 2")
        K = len(ctx.checkpoints)
        if any(b < 0 or b >= K - 1 for b in self.boundaries):
            raise ValueError("boundaries must be checkpoint indices in [0, K-2]")
        M = len(self.menu) if self.mode == "menu" else 1
        self.ledger = pjc_ledger(ctx, self.mode, self.boundaries, M)
        S, A = ctx.S, ctx.A
        self._p = balanced_alloc(S, A, 0.5)                  # phase-0 plan (data-free)
        self._n0 = np.zeros((S, A), dtype=np.int64)          # counts / sums at the current phase start
        self._s0 = np.zeros((S, A))
        self._lo = np.zeros((S, A))                          # running mu-space box
        self._hi = np.ones((S, A))
        self._last_t = -1
        self._phase = 0
        self.path = []                                        # menu choices (menu mode) / plans (local mode)
        self.plans = [self._p.copy()]

    # ------------------------------------------------------------------ allocation
    def plan(self, ctx, st):
        """Deficit tracking of the current phase plan from COUNTS only (never st.sum)."""
        n = np.asarray(st.n, dtype=np.int64) - self._n0
        m = n.sum(1, keepdims=True).astype(float)
        deficit = self._p * (m + 1.0) - n
        first = np.argmax(deficit, axis=1)                    # ties -> lower arm index
        pref = np.empty((ctx.S, ctx.A), dtype=np.int64)
        for s in range(ctx.S):
            a = int(first[s])
            pref[s] = [a] + [b for b in range(ctx.A) if b != a]
        return pref

    def _next_plan(self, ctx, st):
        """Plan for the next phase from data up to the boundary (F_{T_j}-measurable)."""
        n, s, N = np.asarray(st.n, float), np.asarray(st.sum, float), np.asarray(st.N, float)
        mh = (s + 1.0) / (n + 2.0)
        v = mh * (1.0 - mh)
        if self.mode == "local":
            if self.rule == "half":
                return balanced_alloc(ctx.S, ctx.A, 0.5), None
            r = np.where(N > 0, (N - n) / np.maximum(N, 1.0), 0.0)        # remaining fraction after the boundary
            return neyman_matrix(v, r, self.p_floor), None
        # menu mode: projected global variance at horizon min(tau_R, 2 t)
        if self.rule == "half":
            j = self.menu.index(0.5) if 0.5 in self.menu else 0
            return balanced_alloc(ctx.S, ctx.A, self.menu[j]), j
        H = min(float(ctx.tau_R), 2.0 * float(st.t))
        arr = (H - st.t) * np.asarray(ctx.w, float)                         # expected further arrivals per segment
        best, bj = np.inf, 0
        for j, c in enumerate(self.menu):
            p = balanced_alloc(ctx.S, ctx.A, c)
            npj = np.minimum(N, n + p * arr[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(npj > 0, v * (1.0 / np.maximum(npj, 1.0) - 1.0 / np.maximum(N, 1.0)), 0.0)
            obj = float(((np.asarray(ctx.w, float) ** 2)[:, None] * np.maximum(term, 0.0)).sum())
            if obj < best - 1e-18:
                best, bj = obj, j
        return balanced_alloc(ctx.S, ctx.A, self.menu[bj]), bj

    # ------------------------------------------------------------------ certificate
    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("checkpoints must be visited in increasing order")
        self._last_t = st.t
        k = int(np.searchsorted(ctx.checkpoints, st.t))
        if k >= len(ctx.checkpoints) or int(ctx.checkpoints[k]) != st.t:
            raise RuntimeError("certify called off the checkpoint grid")
        a_var = self.ledger["alpha_side_var"]
        beta = self.ledger["beta"]
        N = np.asarray(st.N, dtype=np.int64)
        if self.mode == "local":
            Np = N - self._n0
            npr = np.asarray(st.n, dtype=np.int64) - self._n0
            sp = np.asarray(st.sum, float) - self._s0
            lo_p, hi_p = _hg_box_safe(Np, npr, sp, a_var)
            Nf = np.maximum(N, 1).astype(float)
            lo_mu = (self._s0 + Np * lo_p) / Nf
            hi_mu = (self._s0 + Np * hi_p) / Nf
            self._lo = np.maximum(self._lo, lo_mu)
            self._hi = np.maximum(np.minimum(self._hi, hi_mu), self._lo)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_b = np.where(Np > 0, (N * self._lo - self._s0) / np.maximum(Np, 1), 0.0)
                hi_b = np.where(Np > 0, (N * self._hi - self._s0) / np.maximum(Np, 1), 0.0)
            lo_b = np.clip(lo_b, 0.0, 1.0)
            hi_b = np.clip(np.maximum(hi_b, lo_b), 0.0, 1.0)
            mprime = np.where(npr > 0, sp / np.maximum(npr, 1), np.where(st.n > 0, st.sum / np.maximum(st.n, 1), 0.5))
            mu = np.where(Np > 0, (self._s0 + Np * mprime) / Nf, self._s0 / Nf)
            scale = np.where(N > 0, Np / Nf, 0.0)
            n_use, N_use = npr, Np
        else:
            lo_g, hi_g = hg_mean_interval(N, st.n, st.sum, a_var)
            self._lo = np.maximum(self._lo, lo_g)
            self._hi = np.maximum(np.minimum(self._hi, hi_g), self._lo)
            lo_b, hi_b = self._lo.copy(), self._hi.copy()
            mu = st.mu_hat
            scale = np.ones((ctx.S, ctx.A))
            n_use, N_use = st.n, N
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        cache, out = {}, []
        seg = np.arange(ctx.S)
        for q in range(ctx.Q):
            ih = int(ihs[q])
            if ih not in cache:
                wd = scaled_direction_widths(ctx, ih, n_use, N_use, lo_b, hi_b, beta, scale, self.kind)
                U = (J - J[ih]) + wd
                if self.rect:
                    ph = ctx.pols[ih]
                    D = ctx.pols != ph[None, :]
                    rect = (ctx.w[None, :] * np.where(D, self._hi[seg[None, :], ctx.pols] -
                                                      self._lo[seg, ph][None, :], 0.0)).sum(1)
                    U = np.minimum(U, rect)
                cache[ih] = U
            Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        # phase boundary: freeze the next phase's plan from data up to now (after this checkpoint's certificate)
        if k in self.boundaries:
            p, j = self._next_plan(ctx, st)
            self._p = p
            self.plans.append(p.copy())
            if j is not None:
                self.path.append(int(j))
            if self.mode == "local":
                self._n0 = np.asarray(st.n, dtype=np.int64).copy()
                self._s0 = np.asarray(st.sum, float).copy()
            self._phase += 1
        return out

    def describe(self):
        d = super().describe()
        d.update({"mode": self.mode, "boundaries": list(self.boundaries), "rule": self.rule, "rect": self.rect,
                  "p_floor": self.p_floor, "menu": list(self.menu), "ledger": self.ledger,
                  "guarantee": "FWER <= 0.05 at the K pre-specified checkpoints (FDC-bet-1 applied per phase / "
                               "per menu path)"})
        return d


# =============================================================================================== matched rectangle
class RectCkBF(_CkRect):
    """Bennett-FPC rectangle matched to FDC-BF (same cell inequality, variance box, delta split, design)."""

    def __init__(self, box=False, share=0.5, name=None):
        super().__init__()
        self.box = bool(box)
        self.share = float(share)
        self.name = name or ("RECT-ck-BF+box" if box else "RECT-ck-BF")

    def _ledger(self, ctx):
        K = len(ctx.checkpoints)
        SA = ctx.S * ctx.A
        d_main, d_var = PJC_SPLIT
        beta_c = math.log(2 * SA * K / d_main)
        a = d_var / (2.0 * SA * K)
        return {"beta_c": beta_c, "alpha_side_var": a, "delta_main": d_main, "delta_var": d_var,
                "bound_main": 2 * SA * K * math.exp(-beta_c), "bound_var": 2 * SA * K * a}

    def setup(self, ctx):
        super().setup(ctx)
        self._vlo = np.zeros((ctx.S, ctx.A))
        self._vhi = np.ones((ctx.S, ctx.A))

    def _bounds(self, ctx, st):
        n = np.asarray(st.n, float)
        N = np.asarray(st.N, float)
        s = np.asarray(st.sum, float)
        vlo, vhi = hg_mean_interval(st.N, st.n, st.sum, self.ledger["alpha_side_var"])
        self._vlo = np.maximum(self._vlo, vlo)
        self._vhi = np.maximum(np.minimum(self._vhi, vhi), self._vlo)
        r = bennett_cell_radius(n, N, self._vlo, self._vhi, self.ledger["beta_c"])
        mu = np.where(n > 0, s / np.maximum(n, 1.0), 0.5)
        lo = np.maximum(mu - r, s / np.maximum(N, 1.0))
        hi = np.minimum(mu + r, (s + (N - n)) / np.maximum(N, 1.0))
        exact = n >= N
        lo = np.where(exact, mu, lo)
        hi = np.where(exact, mu, hi)
        empty = n <= 0
        lo = np.where(empty, 0.0, lo)
        hi = np.where(empty, 1.0, hi)
        if self.box:
            lo = np.maximum(lo, self._vlo)
            hi = np.minimum(hi, self._vhi)
        return np.clip(lo, 0.0, 1.0), np.clip(np.maximum(hi, lo), 0.0, 1.0)


def bennett_cell_radius(n, N, lo, hi, beta_c, grid=_LAMBDA_GRID):
    """One-sided Chernoff radius min_lambda (beta_c + sup_box Psi^B(lambda)) / lambda for each cell mean (coefficient 1);
    Psi^B is symmetric in the sign, so the same radius bounds both tails.  Exhausted -> 0, empty -> inf."""
    n = np.asarray(n, float)
    N = np.asarray(N, float)
    live = (n > 0) & (n < N)
    r = np.where(n >= N, 0.0, np.inf)
    if live.any():
        idx = np.flatnonzero(live.ravel())
        nn, NN = n.ravel()[idx], N.ravel()[idx]
        l, h = np.asarray(lo).ravel()[idx], np.asarray(hi).ravel()[idx]
        vm = np.clip(0.5, l, h)
        v = vm * (1 - vm) * (NN - nn) / np.maximum(NN - 1, 1) / nn
        rr = np.zeros(len(idx))
        for i in range(len(idx)):
            if v[i] <= 0:
                rr[i] = 0.0
                continue
            lam = math.sqrt(2 * beta_c / v[i]) * grid
            F = psi_bar(lam, nn[i], NN[i], l[i], h[i], "bennett")
            with np.errstate(invalid="ignore", over="ignore"):
                val = (beta_c + F) / lam
            rr[i] = float(np.nanmin(np.where(np.isfinite(val), val, np.inf)))
        rflat = r.ravel().copy()
        rflat[idx] = rr
        r = rflat.reshape(n.shape)
    return r


# =============================================================================================== own-plan wrappers
class RectCkHGPlan(RectCkHG):
    """RECT-ck-HG with its own frozen read plan: uniform control share or an (S, A) matrix (frozen before the stream)."""

    def __init__(self, share=0.5, alloc=None, name=None):
        super().__init__()
        self.share = float(share)
        self._alloc_fixed = None if alloc is None else np.asarray(alloc, dtype=float)
        self.name = name or f"RECT-ck-HG[{'mat' if alloc is not None else f'{share:g}'}]"

    def setup(self, ctx):
        super().setup(ctx)
        if self._alloc_fixed is not None:
            self.alloc_p = self._alloc_fixed / self._alloc_fixed.sum(1, keepdims=True)


class HCWoRPlan(HCWoRRect):
    """HC-WoR (v6 frozen config) with its own frozen read plan; n_star follows the plan as in v6."""

    def __init__(self, schedule, c, target_frac, share=0.5, alloc=None, name=None):
        super().__init__(schedule, c, target_frac, name=name or "HC-WoR[plan]")
        self.share = float(share)
        self._alloc_fixed = None if alloc is None else np.asarray(alloc, dtype=float)

    def setup(self, ctx):
        super().setup(ctx)
        if self._alloc_fixed is not None:
            self.alloc_p = self._alloc_fixed / self._alloc_fixed.sum(1, keepdims=True)
            if self.schedule == "nstar":
                t_star = self.target_frac * ctx.tau_R
                self.n_star = np.minimum(ctx.N, np.maximum(1.0, t_star * ctx.w[:, None] * self.alloc_p))


class FDCBetPlan(FDCBet):
    """FDC-BF certificate under a different frozen read plan (descriptive only)."""

    def __init__(self, share=0.5, alloc=None, name=None):
        super().__init__(kind="bennett", var_box="HG", fpc=True, rect=False, split=PJC_SPLIT,
                         name=name or f"FDC-BF[{'mat' if alloc is not None else f'{share:g}'}]")
        self.share = float(share)
        self._alloc_fixed = None if alloc is None else np.asarray(alloc, dtype=float)

    def setup(self, ctx):
        super().setup(ctx)
        self.alloc_p = (balanced_alloc(ctx.S, ctx.A, self.share) if self._alloc_fixed is None else
                        self._alloc_fixed / self._alloc_fixed.sum(1, keepdims=True))
