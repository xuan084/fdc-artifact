"""PJC-A: phased joint certificates with GENUINE adaptivity (post-hoc descriptive block v8-posthoc-D).  NEW file;
pjc_bf.py (bound by the v8 lock) is imported, never modified.

Why: in the v8 confirmatory block the tuned PJC members did not adapt (PJC-local* was the no-reset member, PJC-menu*
chose share 0.5 on every stream with menu {0.4, 0.5}), so "no gain from adaptivity" was tested where nothing adapted.
This module adds members whose design actually moves with the data, all inside the SAME guarantee class as FDC-BF
(FWER <= 0.05 at the K pre-specified checkpoints, split 0.045 / 0.005, Bennett-FPC direction widths, exact-HG variance
boxes).

Three families (``family``):

'reset'    (RAGE-style, phase-local estimator; PJCBF mode='local' certificate verbatim)
    Any F_{T_j}-measurable plan is legal (Theorem FDC-bet-1 per phase), so the plan may be continuous and per segment.
    rule 'neyman'  per-segment Neyman from past data (pjc_bf.neyman_matrix on the remaining fraction).
    rule 'dirseg'  per-segment direction-optimal design: coordinate descent over shares {0.10, 0.15, ..., 0.90} per
                   segment minimising the G-optimal proxy below.  Ledger = FDC-BF's (L = K).

'menu'     (keep-data, global share from a data-free menu; PJCBF mode='menu' certificate verbatim)
    rule 'proj'    pjc_bf's projected global variance.
    rule 'dirproj' the menu share minimising the G-optimal proxy.  Ledger: L = sum_k M^{j(k)}.

'segmenu'  (keep-data, PER-SEGMENT share from a data-free per-segment menu)
    At each boundary every segment independently picks a share from ``menu``; the design space per boundary has
    M^S elements, every element's counts are functions of the arrival sequence only (deficit tracking from counts),
    so the menu-path union is charged with M^S paths per boundary: L = sum_k (M^S)^{j(k)} (exact Python integers;
    beta = ln(|Pi| L / 0.045), alpha_side = 0.005 / (2 S A L)).
    rule 'neyman'  per-segment Neyman from past data, rounded to the nearest menu value.
    rule 'dirseg'  coordinate descent over menu values per segment on the G-optimal proxy.

G-optimal proxy (computed at the boundary from data up to T_j only, after that checkpoint's certificate):
    for every problem q that is not certified, every competitor i in Pi_{B_q} with U_q(i) > eps is "violating"; with
    room_qi = eps + J_hat(pi_hat_q) - J_hat(i) (floored at eps / 4) the design objective is
        max_{q, i violating} V_proj(pi_hat_q, i; design) / room_qi^2,
    V_proj = sum over disagreeing segments of the projected estimator variance of the two cells at horizon
    H = min(tau_R, 2 t) (expected further arrivals (H - t) w_s split by the design; Laplace plug-in variances).  For the
    phase-local estimator the coefficient is w_s N'/N and the pool is the remaining pool.  If every problem is
    certified, the 50/50 design is kept.

Diagnostics on the instance: ``plans`` (one (S, A) matrix per phase), ``path`` (menu indices; segmenu: tuple per
boundary), ``switched`` (some later plan differs from the phase-0 50/50 plan).
"""
from __future__ import annotations

import math

import numpy as np

from .b4_bal import balanced_alloc
from .fdc import union_size
from .frontier_common import jhat_all, pi_hat_indices
from .pjc_bf import PJC_DELTA, PJC_SPLIT, PJCBF, _hg_box_safe, neyman_matrix, scaled_direction_widths
from .rect_v6 import hg_mean_interval

__all__ = ["PJCAdapt", "adapt_ledger", "paths_per_checkpoint_exact", "SHARE_GRID", "MENU5", "MENU3"]

MENU5 = (0.3, 0.4, 0.5, 0.6, 0.7)
MENU3 = (0.3, 0.5, 0.7)
SHARE_GRID = tuple(round(0.1 + 0.05 * i, 2) for i in range(17))      # 0.10 .. 0.90


def paths_per_checkpoint_exact(K, boundaries, M_per_boundary):
    """Exact (Python int) number of design paths that determine the counts at checkpoint k = 0..K-1."""
    b = sorted(int(x) for x in boundaries)
    return [int(M_per_boundary) ** sum(1 for x in b if x < k) for k in range(K)]


def adapt_ledger(ctx, family, boundaries, M=1, split=PJC_SPLIT):
    K = int(len(ctx.checkpoints))
    S, A = int(ctx.S), int(ctx.A)
    m = union_size(ctx, "feas")
    if family == "reset":
        Mb = 1
    elif family == "menu":
        Mb = int(M)
    elif family == "segmenu":
        Mb = int(M) ** S
    else:
        raise ValueError(family)
    L = sum(paths_per_checkpoint_exact(K, boundaries, Mb)) if family != "reset" else K
    d_main, d_var = split
    beta = math.log(m) + math.log(L) - math.log(d_main)
    a = d_var / (2.0 * S * A * L)
    return {"family": family, "union_size": m, "K": K, "L_events_per_direction": str(L),
            "log_L": math.log(L), "paths_per_boundary": str(Mb), "S_A": S * A, "delta_main": d_main,
            "delta_var": d_var, "delta_total": d_main + d_var, "beta": beta, "alpha_side_var": a,
            "bound_main": math.exp(math.log(m) + math.log(L) - beta), "bound_var": 2.0 * S * A * L * a,
            "boundaries": [int(x) for x in boundaries], "M": int(M)}


class PJCAdapt(PJCBF):
    """Genuinely adaptive PJC member (see module docstring)."""

    def __init__(self, family="reset", boundaries=(4,), rule="dirseg", menu=MENU5, p_floor=0.1, kind="bennett",
                 name=None):
        rules = {"reset": ("neyman", "dirseg"), "menu": ("proj", "dirproj"), "segmenu": ("neyman", "dirseg")}
        if family not in rules or rule not in rules[family]:
            raise ValueError(f"family {family!r} / rule {rule!r}")
        mode = "local" if family == "reset" else "menu"
        super().__init__(mode=mode, boundaries=boundaries, rule="neyman" if mode == "local" else "proj",
                         rect=False, p_floor=p_floor, menu=menu, kind=kind, name="tmp")
        self.family, self.arule = family, rule
        self.menu = tuple(float(x) for x in menu) if family != "reset" else ()
        bnd = ",".join(str(b) for b in self.boundaries)
        mtag = "/".join(f"{x:g}" for x in self.menu)
        self.name = name or (f"PJC-A[reset,b={bnd},{rule}]" if family == "reset" else
                             f"PJC-A[{family},b={bnd},menu={mtag},{rule}]")
        self._seg_vals = None

    # ------------------------------------------------------------------ setup
    def setup(self, ctx):
        if abs(float(ctx.delta) - PJC_DELTA) > 1e-12:
            raise ValueError("PJC-A: delta fixed at 0.05")
        if not ctx.binary or ctx.A != 2:
            raise ValueError("PJC-A: binary pools, A = 2")
        K = len(ctx.checkpoints)
        if any(b < 0 or b >= K - 1 for b in self.boundaries):
            raise ValueError("boundaries must be checkpoint indices in [0, K-2]")
        self.ledger = adapt_ledger(ctx, self.family, self.boundaries, len(self.menu) if self.menu else 1)
        S, A = ctx.S, ctx.A
        self._p = balanced_alloc(S, A, 0.5)
        self._p0 = self._p.copy()
        self._n0 = np.zeros((S, A), dtype=np.int64)
        self._s0 = np.zeros((S, A))
        self._lo = np.zeros((S, A))
        self._hi = np.ones((S, A))
        self._last_t = -1
        self._phase = 0
        self.path = []
        self.plans = [self._p.copy()]
        self.objective_trace = []

    @property
    def switched(self):
        return any(float(np.max(np.abs(p - self._p0))) > 1e-9 for p in self.plans[1:])

    # ------------------------------------------------------------------ certificate (pjc_bf.PJCBF.certify + stash)
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
        for q in range(ctx.Q):
            ih = int(ihs[q])
            if ih not in cache:
                wd = scaled_direction_widths(ctx, ih, n_use, N_use, lo_b, hi_b, beta, scale, self.kind)
                cache[ih] = (J - J[ih]) + wd
            Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        if k in self.boundaries:
            p, j = self._adaptive_plan(ctx, st, J, ihs, cache, out)
            self._p = p
            self.plans.append(p.copy())
            self.path.append(j)
            if self.mode == "local":
                self._n0 = np.asarray(st.n, dtype=np.int64).copy()
                self._s0 = np.asarray(st.sum, float).copy()
            self._phase += 1
        return out

    # ------------------------------------------------------------------ design rules (F_{T_j}-measurable)
    def _violators(self, ctx, J, ihs, cache, out):
        """[(ih, D_mask (S,), room)] for every violating competitor of every uncertified problem."""
        res = []
        eps = float(ctx.eps)
        for q in range(ctx.Q):
            if out[q][0]:
                continue
            ih = int(ihs[q])
            U = cache[ih]
            bad = np.flatnonzero(ctx.feas[q] & (U > eps))
            if bad.size == 0:
                continue
            room = np.maximum(eps + J[ih] - J[bad], eps / 4.0)
            D = ctx.pols[bad] != ctx.pols[ih][None, :]
            res.append((ih, bad, D, room))
        return res

    def _cell_var_fn(self, ctx, st):
        """Returns f(p) -> c2v (S, A): projected coefficient^2 x estimator variance per cell under design p."""
        n = np.asarray(st.n, float)
        s = np.asarray(st.sum, float)
        N = np.asarray(st.N, float)
        w = np.asarray(ctx.w, float)
        mh = (s + 1.0) / (n + 2.0)
        v = mh * (1.0 - mh)
        H = min(float(ctx.tau_R), 2.0 * float(st.t))
        arr = max(H - float(st.t), 0.0) * w
        if self.mode == "local":
            Np = N - n
            coef2 = (w[:, None] * np.where(N > 0, Np / np.maximum(N, 1.0), 0.0)) ** 2

            def f(p):
                npj = np.minimum(Np, p * arr[:, None])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(Np > 0, v * (1.0 / np.maximum(npj, 1.0) - 1.0 / np.maximum(Np, 1.0)), 0.0)
                return coef2 * np.maximum(t, 0.0)
        else:
            coef2 = (w ** 2)[:, None] * np.ones_like(n)

            def f(p):
                npj = np.minimum(N, n + p * arr[:, None])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(N > 0, v * (1.0 / np.maximum(npj, 1.0) - 1.0 / np.maximum(N, 1.0)), 0.0)
                return coef2 * np.maximum(t, 0.0)
        return f

    @staticmethod
    def _gobj(ctx, viol, c2v):
        seg = np.arange(ctx.S)
        best = 0.0
        for ih, bad, D, room in viol:
            ph = ctx.pols[ih]
            ch = ctx.pols[bad]
            V = (D * (c2v[seg[None, :], ch] + c2v[seg, ph][None, :])).sum(1)
            best = max(best, float(np.max(V / room ** 2)))
        return best

    def _coord_descent(self, ctx, f, viol, vals, start_share):
        S = ctx.S
        sh = np.full(S, float(start_share))

        def mat(x):
            return np.stack([x, 1.0 - x], axis=1)

        cur = self._gobj(ctx, viol, f(mat(sh)))
        for _ in range(3):
            moved = False
            for s in range(S):
                b_val, b_obj = sh[s], cur
                for v in vals:
                    if v == sh[s]:
                        continue
                    x = sh.copy()
                    x[s] = v
                    o = self._gobj(ctx, viol, f(mat(x)))
                    if o < b_obj - 1e-15 * max(1.0, b_obj):
                        b_val, b_obj = v, o
                if b_val != sh[s]:
                    sh[s], cur, moved = b_val, b_obj, True
            if not moved:
                break
        return sh, cur

    def _adaptive_plan(self, ctx, st, J, ihs, cache, out):
        S, A = ctx.S, ctx.A
        n, s, N = np.asarray(st.n, float), np.asarray(st.sum, float), np.asarray(st.N, float)
        mh = (s + 1.0) / (n + 2.0)
        v = mh * (1.0 - mh)
        if self.arule == "neyman":
            r = np.where(N > 0, (N - n) / np.maximum(N, 1.0), 0.0) if self.family == "reset" else None
            p = neyman_matrix(v, r, self.p_floor)
            if self.family == "reset":
                return p, None
            vals = np.asarray(self.menu)
            idx = np.abs(p[:, 0][:, None] - vals[None, :]).argmin(1)          # nearest menu value per segment
            sh = vals[idx]
            return np.stack([sh, 1.0 - sh], axis=1), tuple(int(i) for i in idx)
        if self.arule == "proj":
            p, j = PJCBF._next_plan(self, ctx, st)
            return p, j
        viol = self._violators(ctx, J, ihs, cache, out)
        if not viol:                                                        # all certified: keep 50/50
            if self.family == "menu":
                j = self.menu.index(0.5) if 0.5 in self.menu else 0
                return balanced_alloc(S, A, self.menu[j]), j
            if self.family == "segmenu":
                j = self.menu.index(0.5) if 0.5 in self.menu else 0
                return balanced_alloc(S, A, self.menu[j]), tuple([j] * S)
            return balanced_alloc(S, A, 0.5), None
        f = self._cell_var_fn(ctx, st)
        if self.family == "menu":
            objs = [self._gobj(ctx, viol, f(balanced_alloc(S, A, c))) for c in self.menu]
            j = int(np.argmin(objs))
            self.objective_trace.append([round(o, 6) for o in objs])
            return balanced_alloc(S, A, self.menu[j]), j
        vals = SHARE_GRID if self.family == "reset" else self.menu
        start = 0.5 if 0.5 in vals else vals[len(vals) // 2]
        sh, o = self._coord_descent(ctx, f, viol, vals, start)
        self.objective_trace.append(round(o, 6))
        p = np.stack([sh, 1.0 - sh], axis=1)
        if self.family == "reset":
            return p, None
        return p, tuple(int(self.menu.index(float(x))) for x in sh)

    def describe(self):
        d = super().describe()
        d.update({"family": self.family, "adaptive_rule": self.arule, "menu": list(self.menu),
                  "guarantee": "FWER <= 0.05 at the K pre-specified checkpoints (FDC-bet-1 per phase / per design "
                               "path; segmenu charges M^S paths per boundary)"})
        return d
