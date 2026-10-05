"""RECT-HG-DP: exact hypergeometric per-cell rectangle with the worst-gap DP certificate (v10-posthoc-G, G3; NEW file,
rect_v6 / rect_dp / fdc_dp imported unchanged).

Construction = rect_v6.RectCkHG (lock v6): frozen 50/50 design (identical schedule to FDC-DP), per (cell, checkpoint)
EXACT hypergeometric test-inversion interval for the pool mean at alpha_side = delta / (2 S A K) (all of delta = 0.05
to the means, no variance event), running intersection over the K checkpoints, exhausted cell -> exact point, n = 0 ->
[0, 1], logical finite-pool bounds intersected.  Certificate = rect_dp.rect_dp_certificate (exact multiple-choice
knapsack DP over the implicit class, weighted centre), identical to RECT-BF-DP's.

Validity: rect_v6 docstring -- conditional on the outcome-free schedule each cell's first n_c(k) draws are a uniform
WoR sample, so the count is exactly hypergeometric; the union over 2 S A K one-sided (cell, checkpoint) events gives
P(some mean outside its interval) <= delta, and on that event every rectangle bound dominates the true gap of every
challenger -> stream FWER <= 0.05 at the K checkpoints (same guarantee class as FDC-DP and RECT-BF-DP).  Certifying
only on the predeclared grid (checked at setup).  Parameter-free.
"""
from __future__ import annotations

import numpy as np

from .rect_dp import rect_dp_certificate
from .rect_v6 import RectCkHG

__all__ = ["RectHGDP"]


class RectHGDP(RectCkHG):
    name = "RECT-HG-DP"

    def __init__(self, name="RECT-HG-DP"):
        super().__init__()
        self.name = name

    def setup(self, ctx):
        super().setup(ctx)
        grid = np.asarray(ctx.checkpoints, dtype=np.int64)
        if grid.ndim != 1 or len(grid) == 0 or (grid <= 0).any() or (np.diff(grid) <= 0).any():
            raise ValueError("RECT-HG-DP: the checkpoint grid must be a non-empty strictly increasing positive grid")
        self._grid_set = set(int(x) for x in grid)

    def certify(self, ctx, st):
        if st.t <= self._last_t:
            raise RuntimeError("checkpoints must be visited in strictly increasing order")
        if int(st.t) not in self._grid_set:
            raise RuntimeError(f"RECT-HG-DP: certify at t={st.t} off the predeclared checkpoint grid")
        self._last_t = st.t
        lo, hi = self._bounds(ctx, st)
        self._lo = np.maximum(self._lo, lo)
        self._hi = np.minimum(self._hi, hi)
        self._hi = np.maximum(self._hi, self._lo)
        return rect_dp_certificate(ctx, st.mu_hat, self._lo, self._hi, getattr(st, "undecided", None))
