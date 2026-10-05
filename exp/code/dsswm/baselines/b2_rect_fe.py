"""B2-rect with a tunable forced-exploration constant (round 5, task r5_rival_tuning).

The r4 module ``combgame_joint.py`` is frozen (6297e7f2) and hard-codes the CombGame forced-exploration rule
"if min_a n_{s,a} < sqrt(n_s), target the least-sampled arm". Methodology s4.4 pre-declares the tuning grid
explore_c in {1.0 (published sqrt(n_s)), 0.5}: the rule becomes min_a n_{s,a} < explore_c * sqrt(n_s).

Only the sampling rule changes; the certificate is the unchanged rect_certificate (per-cell WSR20 WoR CS at
delta / (S A), radius sum), whose validity does not depend on how arms are chosen. So every grid point is rigorous.
explore_c = 1.0 reproduces CombGameJoint("rect") exactly (identity-checked in r5_rival_tuning and in
tests/test_r5_rival_tuning.py).
"""
from __future__ import annotations

import math

import numpy as np

from .combgame_joint import CombGameJoint
from .frontier_common import glr_V, jhat_all, pi_hat_indices

__all__ = ["CombGameRectFE", "B2_EXPLORE_GRID"]

B2_EXPLORE_GRID = (1.0, 0.5)


class CombGameRectFE(CombGameJoint):
    def __init__(self, explore_c=1.0, name=None):
        if not explore_c > 0:
            raise ValueError("explore_c must be > 0")
        super().__init__("rect", name=name or "B2-rect")
        self.explore_c = float(explore_c)

    def plan(self, ctx, st):  # copy of CombGameJoint.plan with the forced-exploration constant exposed
        self._round += 1
        n = st.n.astype(float)
        ns = n.sum(1)
        self._T += self._p * (ns - self._last_ns)[:, None]
        self._last_ns = ns.copy()
        var = self._var(ctx, st)
        var_s = np.maximum(var, 1e-6)
        mu = st.mu_hat
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        t = max(st.t, 1)
        gain = np.zeros((ctx.S, ctx.A))
        if (n > 0).all():
            for q in np.flatnonzero(st.undecided):
                ih = ihs[q]
                V = glr_V(ctx, st, var_s, ih)
                gap = np.maximum(ctx.eps - (J - J[ih]), 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Z = np.where(V > 0, gap ** 2 / (2 * V), np.inf)
                Z[ih] = np.inf
                Z = np.where(ctx.feas[q], Z, np.inf)
                br = int(np.argmin(Z))
                if not np.isfinite(Z[br]) or V[br] <= 0:
                    continue
                ph, pc = ctx.pols[ih], ctx.pols[br]
                d = np.flatnonzero(ph != pc)
                lam = n / t
                coef = gap[br] ** 2 / (2 * V[br] ** 2)
                for arm_row in (pc, ph):
                    a = arm_row[d]
                    gain[d, a] += coef * ctx.w[d] ** 2 * var_s[d, a] / (lam[d, a] ** 2) / t
        mx = gain.max(1, keepdims=True)
        g = np.where(mx > 0, gain / np.where(mx > 0, mx, 1.0), 0.0)
        self._cum_gain += g
        eta = math.sqrt(8 * math.log(max(ctx.A, 2)) / self._round)
        z = eta * (self._cum_gain - self._cum_gain.max(1, keepdims=True))
        p = np.exp(z)
        self._p = p / p.sum(1, keepdims=True)
        exp_arr = ctx.w * ctx.replan
        deficit = self._T + self._p * exp_arr[:, None] - n
        pref = np.zeros((ctx.S, ctx.A), dtype=np.int64)
        for s in range(ctx.S):
            order = [int(a) for a in np.argsort(-deficit[s], kind="stable")]
            if n[s].min() < self.explore_c * math.sqrt(max(ns[s], 1.0)):
                low = int(np.argmin(n[s]))
                order.remove(low)
                order = [low] + order
            pref[s] = order
        return pref

    def describe(self):
        d = super().describe()
        d["explore_c"] = self.explore_c
        return d


def _register():
    from ..streams.r5_registry import REGISTRY, MethodSpec, register
    old = REGISTRY["B2-rect"]
    register(MethodSpec(old.name, old.validity_class, old.role, old.in_R, old.source,
                        "baselines/b2_rect_fe.py (subclass of combgame_joint.py)", {"explore_c": 1.0},
                        "forced-exploration constant tunable (r5_rival_tuning); explore_c=1.0 == r4 B2-rect"),
             lambda explore_c=1.0, **kw: CombGameRectFE(explore_c=explore_c))


_register()
