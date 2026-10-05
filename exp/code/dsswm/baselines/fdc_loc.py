"""v9 candidate upgrades of FDC-BF (NEW file; locked v5-v8 modules imported, never modified).

Theory: plan/v9_candidates_theory.md (written before this code).

  FDCLoc  (candidate a, "FDC-LOC")  gap-localised union.  At checkpoint k the main exponent beta_J = ln(M K/delta_main)
          is replaced by beta_hat_k = min{beta <= beta_J : G_hat_k(beta) <= delta_main / K}, with
              G_hat_k(beta) = sum_q sum_{pi in Pi_q} exp(-beta (1 + (Dlo_{q,pi} - eps)_+ / Wbar_{q,pi}(beta))),
          Dlo the pairwise-rectangle gap lower bound on the SAME exact-HG variance boxes (E_var, no extra delta) and
          Wbar a Bernstein upper bound on the ideal width of (pi'', pi) maximised over the survivors pi''.  Theorem LOC-1.
          beta_hat_k <= beta_J always, so widths <= FDC-BF's on the same schedule / boxes.

  FDCTimeUniform (candidate b, "TU-FDC")  FDC-BF (or FDC-LOC) certified at ANY evaluation time t >= t_1 with the width
          frozen at the latest block point t_k <= t (counts, boxes and beta of t_k; Delta_hat from the current data).
          Lemma TU (reverse-martingale maximal inequality) makes the K-block ledger valid uniformly in time.  With the
          evaluation grid equal to the block grid it is numerically identical to FDC-BF (FDC-LOC).
"""
from __future__ import annotations

import math

import numpy as np

from .fdc_bet import FDCBet, direction_widths
from .frontier_common import jhat_all, pi_hat_indices
from .rect_v6 import hg_mean_interval

__all__ = ["FDCLoc", "FDCTimeUniform", "loc_beta", "pair_tables", "BETA_TOL"]

BETA_TOL = 1e-6
BETA_FLOOR = 1e-3


def pair_tables(ctx, n, N, lo, hi):
    """Pairwise quantities over all policy pairs (P x P), rows = pi' (first argument), cols = pi.

    L[i, j]  = sum_{s in D} w_s (lo[s, pi_i(s)] - hi[s, pi_j(s)])  (lower bound of J(pi_i) - J(pi_j) on E_var)
    V[i, j]  = sum_{s in D} w_s^2 (v_s0 + v_s1),  b[i, j] = max_{live c in D} w_s / n_c,  Z[i, j] = touches an n = 0 cell.
    v_c = box-sup Var(mu_hat_c) with FPC (as fdc_bet.direction_widths)."""
    S = ctx.S
    pols = ctx.pols
    w = ctx.w
    n = np.asarray(n, dtype=float)
    N = np.asarray(N, dtype=float)
    live = (n > 0) & (n < N)
    vmax = np.clip(0.5, lo, hi)
    vmax = vmax * (1.0 - vmax)
    fp = (N - n) / np.maximum(N - 1.0, 1.0)
    vcell = np.where(live, vmax * fp / np.maximum(n, 1.0), 0.0)
    bcell = np.where(live, w[:, None] / np.maximum(n, 1.0), 0.0)
    zcell = n <= 0
    P = pols.shape[0]
    L = np.zeros((P, P))
    V = np.zeros((P, P))
    b = np.zeros((P, P))
    Z = np.zeros((P, P), dtype=bool)
    for s in range(S):
        ai = pols[:, s][:, None]
        aj = pols[:, s][None, :]
        D = ai != aj
        L += np.where(D, w[s] * (lo[s, ai] - hi[s, aj]), 0.0)
        V += np.where(D, w[s] ** 2 * (vcell[s, 0] + vcell[s, 1]), 0.0)
        bs = max(bcell[s, 0], bcell[s, 1])
        b = np.where(D, np.maximum(b, bs), b)
        Z |= D & (zcell[s, 0] | zcell[s, 1])
    return L, V, b, Z


def loc_beta(ctx, n, N, lo, hi, beta_J, target, tol=BETA_TOL):
    """beta_hat = smallest beta (bisection, returned from the admissible side) with G_hat(beta) <= target; <= beta_J.

    Returns (beta_hat, info)."""
    eps = ctx.eps
    L, V, b, Z = pair_tables(ctx, n, N, lo, hi)
    xs, Vs, bs, zs = [], [], [], []
    n_surv, n_good = [], []
    for q in range(ctx.Q):
        f = np.asarray(ctx.feas[q], dtype=bool)
        Lq = np.where(f[:, None], L, -np.inf)                    # rows pi' restricted to Pi_q
        dlo = Lq.max(0)[f]                                       # gap lower bound of every pi in Pi_q
        dlo_all = Lq.max(0)
        surv = f & (dlo_all <= 0.0)                              # contains pi*_q on E_var
        n_surv.append(int(surv.sum()))
        # gap upper bound: Delta_pi <= max_{pi'' in S_q} [J(pi'') - J(pi)] <= max_{pi''} -L[pi, pi''] on E_var;
        # pi with dup <= eps are surely eps-good: not in G's index set, dropped from G_hat (Theorem LOC-1, refinement)
        dup = np.where(surv[None, :], -L, -np.inf).max(1)[f]
        keep = dup > eps
        n_good.append(int((~keep).sum()))
        Vq = np.where(surv[:, None], V, -np.inf)[:, f]           # (P, |Pi_q|) rows = survivors
        bq = np.where(surv[:, None], b, -np.inf)[:, f]
        zq = (surv[:, None] & Z)[:, f].any(0)
        xs.append(np.maximum(dlo - eps, 0.0)[keep])
        Vs.append(Vq[:, keep])
        bs.append(bq[:, keep])
        zs.append(zq[keep])

    def wbar(beta):
        out = []
        for Vq, bq, zq in zip(Vs, bs, zs):
            fin = np.isfinite(Vq)
            wq = np.where(fin, np.sqrt(2.0 * beta * np.where(fin, np.maximum(Vq, 0.0), 0.0))
                          + np.where(fin, bq, 0.0) * beta / 3.0, -np.inf)
            wq = wq.max(0) if wq.shape[1] else np.zeros(0)
            wq = np.where(zq, np.inf, wq)
            out.append(wq)
        return out

    def G(beta):
        tot = 0.0
        for x, wq in zip(xs, wbar(beta)):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(np.isfinite(wq) & (wq > 0), x / np.where(wq > 0, wq, 1.0), 0.0)
            if beta <= 0:
                tot += float(r.size)                             # conservative value at beta = 0 (never used as beta_hat)
                continue
            # zero width with beta > 0 means every touched cell is exhausted (Y == 0): excess gap > 0 is impossible
            r = np.where((wq <= 0) & (x > 0), np.inf, r)
            with np.errstate(over="ignore"):
                tot += float(np.sum(np.exp(-beta * (1.0 + r)))) if r.size else 0.0
        return tot

    M = int(np.asarray(ctx.feas).sum())
    info = {"n_survivors": n_surv, "n_sure_good": n_good, "M": M}
    if G(beta_J) > target * (1 + 1e-9):                          # cannot happen (G <= M e^-beta); defensive
        info["fallback"] = True
        return beta_J, info
    a, c = 0.0, beta_J                                           # G(c) <= target invariant
    if G(a) <= target:
        # every remaining term vanishes (e.g. all policies surely eps-good): any beta > 0 is admissible.  beta = 0 would
        # make direction_widths return +inf (0/0 on a zero lambda grid), so use the positive admissible value
        # BETA_FLOOR (G_hat is nonincreasing, so G_hat(BETA_FLOOR) <= G_hat(0) <= target).  external reviewer r1 fix.
        c = min(BETA_FLOOR, beta_J)
    else:
        while c - a > tol:
            mid = 0.5 * (a + c)
            if G(mid) <= target:
                c = mid
            else:
                a = mid
    info["G_at_beta"] = G(c)
    info["n_unresolved_e"] = float(G(c) * math.exp(c))          # effective count  sum exp(-beta x/W)
    return c, info


class FDCLoc(FDCBet):
    """FDC-LOC: FDC-BF with the gap-localised main exponent beta_hat_k (Theorem LOC-1)."""

    def __init__(self, name="FDC-LOC", localise=True, **kw):
        kw.setdefault("kind", "bennett")
        kw.setdefault("var_box", "HG")
        super().__init__(name=name, **kw)
        self.localise = bool(localise)
        self.beta_trace = []

    def _beta(self, ctx, st, lo, hi):
        beta_J = self.ledger["beta"]
        if not self.localise:
            return beta_J, {}
        K = self.ledger["K"]
        return loc_beta(ctx, st.n, st.N, lo, hi, beta_J, self.split[0] / K)

    def certify_with(self, ctx, mu, n, N, lo, hi, beta):
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        cache = {}
        out = []
        for q in range(ctx.Q):
            ih = int(ihs[q])
            if ih not in cache:
                wd = direction_widths(ctx, ih, n, N, lo, hi, beta, self.kind, self.fpc, self.grid)
                cache[ih] = (J - J[ih]) + wd
            Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        return out

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("checkpoints must be visited in increasing order")
        self._last_t = st.t
        lo, hi = self._box(ctx, st)
        beta, info = self._beta(ctx, st, lo, hi)
        self.beta_trace.append({"t": int(st.t), "beta": float(beta), "beta_J": float(self.ledger["beta"]),
                                "n_unresolved_e": info.get("n_unresolved_e"),
                                "n_survivors_sum": int(sum(info["n_survivors"])) if "n_survivors" in info else None,
                                "n_sure_good_sum": int(sum(info["n_sure_good"])) if "n_sure_good" in info else None})
        return self.certify_with(ctx, st.mu_hat, st.n, st.N, lo, hi, beta)

    def describe(self):
        d = super().describe()
        d.update({"localise": self.localise,
                  "guarantee": "FWER <= 0.05 (Theorem LOC-1); time-uniform with stale widths (Lemma TU)"})
        return d


class FDCTimeUniform(FDCLoc):
    """TU-FDC: certify at any evaluation time with the width frozen at the latest block point (Theorem TU-1).

    block_points: the frozen block grid (t values); ledger K = len(block_points) (NOT the evaluation grid).
    localise=False -> TU version of FDC-BF; localise=True -> TU version of FDC-LOC."""

    def __init__(self, block_points, name="TU-FDC", localise=False, **kw):
        super().__init__(name=name, localise=localise, **kw)
        self.block_points = np.asarray(block_points, dtype=np.int64)

    def setup(self, ctx):
        super().setup(ctx)
        K = int(len(self.block_points))
        m = self.ledger["union_size"]
        self.ledger.update({"K": K, "beta": math.log(m * K / self.split[0]),
                            "alpha_side_var": self.split[1] / (2.0 * ctx.S * ctx.A * K),
                            "block_points": self.block_points.tolist(), "K_eval": int(len(ctx.checkpoints))})
        self._snap = None
        self._next_block = 0
        self.n_stale_evals = 0

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("evaluation times must be increasing")
        self._last_t = st.t
        # advance through every block point <= t; the snapshot is taken at the LATEST block point reached.  The engine
        # calls certify only at evaluation times, so a block point that is not an evaluation time cannot be
        # snapshotted -> we require block points to be a subset of the evaluation grid.
        while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
            bp = int(self.block_points[self._next_block])
            if bp != st.t:
                raise RuntimeError("block grid must be a subset of the evaluation grid")
            lo, hi = self._box(ctx, st)                       # box only at block points (E_var on the block grid)
            beta, info = self._beta(ctx, st, lo, hi)
            self._snap = (st.n.copy(), st.N.copy(), lo, hi, beta)
            self.beta_trace.append({"t": bp, "beta": float(beta), "beta_J": float(self.ledger["beta"]),
                                    "n_unresolved_e": info.get("n_unresolved_e")})
            self._next_block += 1
        if self._snap is None:                                # before t_1: no certification
            return [(False, 0, float("inf")) for _ in range(ctx.Q)]
        n_k, N_k, lo, hi, beta = self._snap
        if int(self.block_points[self._next_block - 1]) != st.t:
            self.n_stale_evals += 1
        return self.certify_with(ctx, st.mu_hat, n_k, N_k, lo, hi, beta)


def dense_grid(ck, sub=4):
    """Evaluation grid with (sub - 1) geometric interpolants per checkpoint interval; contains ck exactly."""
    ck = np.asarray(ck, dtype=np.int64)
    pts = [int(ck[0])]
    for a, b in zip(ck[:-1], ck[1:]):
        for j in range(1, sub):
            pts.append(int(round(math.exp(math.log(a) + j * (math.log(b) - math.log(a)) / sub))))
        pts.append(int(b))
    g = np.unique(np.asarray(pts, dtype=np.int64))
    assert set(ck.tolist()) <= set(g.tolist())
    return g


__all__ += ["dense_grid", "hg_mean_interval"]
