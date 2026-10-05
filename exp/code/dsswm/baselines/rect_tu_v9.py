"""Lock v9: time-uniform matched Bennett rectangle RECT-ck-BF-TU (NEW file; pjc_bf / rect_v6 imported, never modified).

RECT-ck-BF(+box) (pjc_bf.RectCkBF) is the per-cell Bennett-FPC rectangle matched to FDC-BF: same cell inequality, same
exact-HG variance box, same (delta_main, delta_var) = (0.045, 0.005) split, same frozen design.  Its guarantee is at the
K pre-specified checkpoints.  Lemma TU (plan/v9_candidates_theory.md (b): reverse-martingale maximal inequality for a
uniform WoR cell mean, Serfling 1974) upgrades every such Chernoff certificate to ALL times t >= t_1 under the frozen
outcome-free schedule, with the radius frozen at the latest block point:

  * block points t_1 < ... < t_K = the frozen K = 20 checkpoint grid (ledger K = 20, NOT the evaluation grid);
  * at a block point t_k: variance box (exact-HG interval at alpha_side_var = delta_var / (2 S A K), running
    intersection over block points -- event E_var on the block grid) and the one-sided Bennett-FPC radius
    r_{c,k} = min_lambda (beta_c + sup_box Psi^B(lambda)) / lambda,  beta_c = ln(2 S A K / delta_main);
  * at ANY evaluation time t with k(t) = max{k : t_k <= t}: cell interval [mu_hat_c(t) - r_{c,k(t)}, mu_hat_c(t) +
    r_{c,k(t)}] intersected with the deterministic always-valid bounds [s/N, (s + N - n)/N] (and, with box=True, with
    the block-point variance box, a statement about the fixed mu_c), running intersection over evaluation times, and
    the same rectangle certificate as RECT-ck-BF.

Validity (same structure as Theorem TU-1): on E_var, r_{c,k} >= r*_{c,k} := the Chernoff radius of the TRUE cell MGF
at t_k (Bennett-FPC with the box-sup variance dominates it), which is deterministic given the schedule; Lemma TU with
coefficient vector a = +-e_c gives P(sup_{t >= t_k} +-(mu_c - mu_hat_c(t)) > r*_{c,k}) <= e^{-beta_c} for every cell
with n_c(t_k) >= 1 (empty cells have r = inf and use only the deterministic bounds).  Union over 2 S A K events:
delta_main; plus delta_var.  On the intersection every interval at every time contains mu_c, so the running
intersection does too.  FWER <= 0.05 uniformly over t >= t_1 under the frozen schedule (NOT under adaptive sampling).
On the block grid it is numerically identical to RECT-ck-BF(+box) (unit test).
"""
from __future__ import annotations

import math

import numpy as np

from .pjc_bf import PJC_SPLIT, RectCkBF, bennett_cell_radius
from .rect_v6 import hg_mean_interval, rect_certificate_from

__all__ = ["RectCkBFTU", "make_tu_pjc"]


class RectCkBFTU(RectCkBF):
    """Time-uniform RECT-ck-BF(+box): radii frozen at the latest block point, centre mu_hat(t) current."""

    def __init__(self, block_points, box=True, name="RECT-ck-BF-TU"):
        super().__init__(box=box, name=name)
        self.block_points = np.asarray(block_points, dtype=np.int64)

    def _ledger(self, ctx):
        K = int(len(self.block_points))
        SA = ctx.S * ctx.A
        d_main, d_var = PJC_SPLIT
        beta_c = math.log(2 * SA * K / d_main)
        a = d_var / (2.0 * SA * K)
        return {"beta_c": beta_c, "alpha_side_var": a, "delta_main": d_main, "delta_var": d_var,
                "bound_main": 2 * SA * K * math.exp(-beta_c), "bound_var": 2 * SA * K * a, "K": K,
                "K_eval": int(len(ctx.checkpoints)), "block_points": self.block_points.tolist()}

    def setup(self, ctx):
        super().setup(ctx)
        self._snap = None
        self._next_block = 0
        self.n_stale_evals = 0

    def _block_update(self, ctx, st):
        n = np.asarray(st.n, float)
        N = np.asarray(st.N, float)
        vlo, vhi = hg_mean_interval(st.N, st.n, st.sum, self.ledger["alpha_side_var"])
        self._vlo = np.maximum(self._vlo, vlo)
        self._vhi = np.maximum(np.minimum(self._vhi, vhi), self._vlo)
        r = bennett_cell_radius(n, N, self._vlo, self._vhi, self.ledger["beta_c"])
        self._snap = (r, self._vlo.copy(), self._vhi.copy())

    def _bounds_at(self, st):
        r, vlo, vhi = self._snap
        n = np.asarray(st.n, float)
        N = np.asarray(st.N, float)
        s = np.asarray(st.sum, float)
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
            lo = np.maximum(lo, vlo)
            hi = np.minimum(hi, vhi)
        return np.clip(lo, 0.0, 1.0), np.clip(np.maximum(hi, lo), 0.0, 1.0)

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("evaluation times must be increasing")
        self._last_t = st.t
        while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
            if int(self.block_points[self._next_block]) != st.t:
                raise RuntimeError("block grid must be a subset of the evaluation grid")
            self._block_update(ctx, st)
            self._next_block += 1
        if self._snap is None:                                   # before t_1: no certification
            return [(False, 0, float("inf")) for _ in range(ctx.Q)]
        if int(self.block_points[self._next_block - 1]) != st.t:
            self.n_stale_evals += 1
        lo, hi = self._bounds_at(st)
        self._lo = np.maximum(self._lo, lo)
        self._hi = np.minimum(self._hi, hi)
        self._hi = np.maximum(self._hi, self._lo)
        return rect_certificate_from(ctx, st, self._lo, self._hi)

    def describe(self):
        d = super().describe()
        d.update({"guarantee": "FWER <= 0.05 uniformly over t >= t_1 under the frozen schedule (Lemma TU); radii "
                               "frozen at the latest of the K = 20 block points", "box": self.box})
        return d


# =============================================================================================== TU form of PJC-local
def _pjc_local_tu_class():
    import copy as _copy

    from .frontier_common import jhat_all, pi_hat_indices
    from .pjc_bf import PJCBF, scaled_direction_widths

    class PJCLocalTUImpl(PJCBF):
        """TU-PJC: the frozen no-reset PJC-local member (mode local, boundaries (), deterministic count tracking of a
        data-free 50/50 plan -> outcome-free allocation) with widths frozen at the latest of the K = 20 block points
        (Lemma TU, exactly as TU-1 for FDC-BF: counts are deterministic given the arrival sequence).  At a block point
        the certificate IS PJC-local's (computed by PJCBF.certify on the block grid); in between, PJC's direction
        widths of the block point with the current centre.  Ledger K = 20."""

        def __init__(self, block_points, name="TU-PJC", **kw):
            kw = dict(kw)
            kw.pop("pick", None)
            kw["boundaries"] = tuple(kw.get("boundaries", ()))
            super().__init__(**kw, name=name)
            if self.mode != "local" or self.boundaries or self.rect or self.rule != "half":
                raise ValueError("TU-PJC is defined only for the no-reset local member (b=(), half, no rect)")
            self.block_points = np.asarray(block_points, dtype=np.int64)

        def setup(self, ctx):
            cb = _copy.copy(ctx)
            cb.checkpoints = self.block_points.copy()
            super().setup(cb)
            self._ctx_block = cb
            self.ledger["K_eval"] = int(len(ctx.checkpoints))
            self.ledger["block_points"] = self.block_points.tolist()
            self._snap = None
            self._next_block = 0
            self.n_stale_evals = 0
            self._tlast = -1

        def certify(self, ctx, st):
            if st.t < self._tlast:
                raise RuntimeError("evaluation times must be increasing")
            self._tlast = st.t
            out_block = None
            while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
                if int(self.block_points[self._next_block]) != st.t:
                    raise RuntimeError("block grid must be a subset of the evaluation grid")
                out_block = PJCBF.certify(self, self._ctx_block, st)          # PJC-local's own certificate
                N = np.asarray(st.N, dtype=np.int64)
                lo_b = np.clip(np.where(N > 0, self._lo, 0.0), 0.0, 1.0)
                hi_b = np.clip(np.maximum(np.where(N > 0, self._hi, 0.0), lo_b), 0.0, 1.0)
                self._snap = (np.asarray(st.n, dtype=np.int64).copy(), N.copy(), lo_b, hi_b)
                self._next_block += 1
            if out_block is not None:
                return out_block
            if self._snap is None:
                return [(False, 0, float("inf")) for _ in range(ctx.Q)]
            self.n_stale_evals += 1
            n_k, N_k, lo_b, hi_b = self._snap
            n, s = np.asarray(st.n, float), np.asarray(st.sum, float)
            # centre exactly as PJCBF.certify (no-reset member): current cell mean, 1/2 for unsampled cells of a
            # non-empty pool, 0 for empty pools (N == 0)
            mu = np.where(np.asarray(st.N) > 0, np.where(n > 0, s / np.maximum(n, 1.0), 0.5), 0.0)
            beta = self.ledger["beta"]
            J = jhat_all(ctx, mu)
            ihs = pi_hat_indices(ctx, mu)
            ones = np.ones((ctx.S, ctx.A))
            cache, out = {}, []
            for q in range(ctx.Q):
                ih = int(ihs[q])
                if ih not in cache:
                    wd = scaled_direction_widths(ctx, ih, n_k, N_k, lo_b, hi_b, beta, ones, self.kind)
                    cache[ih] = (J - J[ih]) + wd
                Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
                out.append((bool(Uq <= ctx.eps), ih, Uq))
            return out

    return PJCLocalTUImpl


def make_tu_pjc(block_points, **kw):
    return _pjc_local_tu_class()(block_points, **kw)
