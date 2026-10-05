"""RECT-BF-DP and HC-WoR-DP: per-cell rectangle certificates over implicit segment-policy classes (v10, NEW file).

RECT-BF-DP is pjc_bf.RectCkBF(+box) -- the Bennett-FPC rectangle matched to FDC-BF / FDC-DP: same cell inequality,
same exact-HG variance box (alpha_side = delta_var / (2 S A K), running intersection), same (0.045, 0.005) split, same
frozen 50/50 design -- with the enumeration over pols replaced by an exact multiple-choice knapsack DP:
    U_q = max(0, max_{pi in Pi_q, pi != pi_hat} sum_s w_s (hi[s, pi(s)] - lo[s, pi_hat(s)]) [s in D]).
The rectangle is additive over segments, so one DP is exact (no lambda / quantifier issue).  Its union is over the
2 S A K one-sided cell events (beta_C = ln(2 S A K / delta_main)), independent of |Pi_q|.

HC-WoR-DP is wor_betting_v6.HCWoRRect (hedged-capital WoR CS per cell, delta / (S A) per cell, time-uniform, CS
supplied by the v6 harness engine) with the same DP rectangle certificate.

RECT-BF-DP-TU: rect_tu_v9.RectCkBFTU (radii frozen at block points, Lemma TU) with the DP certificate.

external reviewer FDC-DP r1: B1 the centre is the WEIGHTED empirical argmax (dp_argmax(w[:, None] * mu_hat)); B4 fresh-radius
RECT-BF-DP accepts only times on the predeclared grid ctx.checkpoints (dense monitoring only via RectBFDPTU).
"""
from __future__ import annotations

import numpy as np

from .fdc_dp import dp_argmax, dp_max_cols, reduce_costs
from .pjc_bf import RectCkBF
from .rect_tu_v9 import RectCkBFTU
from .wor_betting_v6 import HCWoRRect

__all__ = ["rect_dp_certificate", "RectBFDP", "RectBFDPTU", "HCWoRDP"]


def rect_dp_certificate(ctx, mu_hat, lo, hi, undecided=None):
    """DP rectangle certificate for every problem of a SegCtx; returns [(certified, pi_hat tuple, U)]."""
    sp = ctx.sp
    cost, budgets, _ = reduce_costs(sp.cost, sp.budgets)
    S = ctx.S
    seg = np.arange(S)
    out = [None] * ctx.Q
    groups = {}
    vals = np.asarray(ctx.w, float)[:, None] * np.asarray(mu_hat, float)       # weighted centre (external reviewer r1 B1)
    for q in range(ctx.Q):
        _, ph = dp_argmax(vals, sp.cost, int(sp.budgets[q]))
        key = tuple(int(x) for x in ph)
        if undecided is not None and not undecided[q]:
            out[q] = (False, key, float("nan"))
            continue
        groups.setdefault(key, []).append(q)
    for key, qs in groups.items():
        ph = np.asarray(key, dtype=np.int64)
        T = (ctx.w[:, None] * (hi - lo[seg, ph][:, None]))[:, :, None]
        T[seg, ph, :] = 0.0
        diff = np.ones((S, ctx.A), dtype=bool)
        diff[seg, ph] = False
        f1 = dp_max_cols(T, cost, diff, int(max(budgets[q] for q in qs)))
        for q in qs:
            U = max(0.0, float(f1[0, int(budgets[q])]))
            out[q] = (bool(U <= ctx.eps), key, U)
    return out


class RectBFDP(RectCkBF):
    """RECT-BF-DP (box=True by default: the matched RECT-ck-BF+box)."""

    def __init__(self, box=True, name=None):
        super().__init__(box=box, name=name or ("RECT-BF-DP" if box else "RECT-BF-DP[nobox]"))

    def setup(self, ctx):
        super().setup(ctx)
        grid = np.asarray(ctx.checkpoints, dtype=np.int64)
        if grid.ndim != 1 or len(grid) == 0 or (grid <= 0).any() or (np.diff(grid) <= 0).any():
            raise ValueError("RECT-BF-DP: the checkpoint grid must be a non-empty strictly increasing positive grid")
        self._grid_set = set(int(x) for x in grid)

    def certify(self, ctx, st):
        if st.t <= self._last_t:
            raise RuntimeError("checkpoints must be visited in strictly increasing order")
        if int(st.t) not in self._grid_set:
            raise RuntimeError(f"RECT-BF-DP: fresh-radius certify at t={st.t} off the predeclared checkpoint grid "
                               "(use RectBFDPTU for denser monitoring)")
        self._last_t = st.t
        lo, hi = self._bounds(ctx, st)
        self._lo = np.maximum(self._lo, lo)
        self._hi = np.minimum(self._hi, hi)
        self._hi = np.maximum(self._hi, self._lo)
        return rect_dp_certificate(ctx, st.mu_hat, self._lo, self._hi, getattr(st, "undecided", None))


class RectBFDPTU(RectCkBFTU):
    """Time-uniform RECT-BF-DP (radii frozen at the latest block point)."""

    def __init__(self, block_points, box=True, name="RECT-BF-DP-TU"):
        super().__init__(block_points, box=box, name=name)

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("evaluation times must be increasing")
        self._last_t = st.t
        while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
            if int(self.block_points[self._next_block]) != st.t:
                raise RuntimeError("block grid must be a subset of the evaluation grid")
            self._block_update(ctx, st)
            self._next_block += 1
        if self._snap is None:
            return [(False, None, float("inf")) for _ in range(ctx.Q)]
        if int(self.block_points[self._next_block - 1]) != st.t:
            self.n_stale_evals += 1
        lo, hi = self._bounds_at(st)
        self._lo = np.maximum(self._lo, lo)
        self._hi = np.minimum(self._hi, hi)
        self._hi = np.maximum(self._hi, self._lo)
        return rect_dp_certificate(ctx, st.mu_hat, self._lo, self._hi, getattr(st, "undecided", None))


class HCWoRDP(HCWoRRect):
    """HC-WoR with the DP rectangle certificate (CS from the harness: needs the HC engine, see seg_v10.run_stream)."""

    def __init__(self, schedule="nstar", c=0.75, target_frac=0.20, name="HC-WoR-DP"):
        super().__init__(schedule, c, target_frac, name=name)

    def certify(self, ctx, st):
        lo, hi = st.cs()
        return rect_dp_certificate(ctx, st.mu_hat, lo, hi, getattr(st, "undecided", None))
