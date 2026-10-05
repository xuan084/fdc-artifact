"""B5: multi-armed Qini / maq (Sverdrup, Wu, Athey & Wager 2023, "Qini curves for multi-armed treatment rules"),
descriptive baseline -- ASYMPTOTIC, NOT a finite-sample guarantee.

The ``maq`` Python package is not installable here (its PyPI sdist ships no setup.py / pyproject), so the estimator is
re-implemented on the cell sufficient statistics (segment-level treatment rules, exact for the A = 2 frontier):

* ``qini_curve``: the multi-armed Qini curve = the solution path of the budget-constrained (fractional) allocation
  problem max_pi sum_s w_s mu_hat[s, pi(s)] s.t. sum_s w_s kappa[pi(s)] <= B, traced by the convex-hull / greedy
  upgrade rule of maq (upgrade the (segment, arm) pair with the largest incremental gain per incremental cost first,
  dropping dominated arms), with plug-in delta-method standard errors per path point.
* certificate (B5 on the frontier, pool allocation = offline RCT log): at checkpoint k and problem q, pi_hat_q =
  argmax J_hat over Pi_{B_q} (segment-level, enumerated), and the problem is "certified" iff for every feasible pi'
      Delta_hat(pi', pi_hat) + z_{1 - delta / K} se(Delta_hat(pi', pi_hat)) <= eps,
  se^2 = sum_{s differing} w_s^2 (v[s, pi'(s)] / n + v[s, pi_hat(s)] / n), v = plug-in variance (exhausted cells exact).
  Fixed-sample normal intervals with a Bonferroni correction over the K checkpoints ("time Bonferroni") only -- no
  union over policies or problems, as in maq's pointwise intervals. validity = 'asymptotic'.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .frontier_common import Method, glr_V, jhat_all, pi_hat_indices, plugin_var

__all__ = ["MaqQini", "qini_curve"]


def qini_curve(w, mu_hat, n, kappa, var=None):
    """Greedy convex-hull Qini path. Returns list of dicts {cost, gain, se, upgrade=(s, from, to)} starting from the
    all-cheapest-arm rule (gain measured relative to it)."""
    w = np.asarray(w, dtype=float)
    mu = np.asarray(mu_hat, dtype=float)
    n = np.asarray(n, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    S, A = mu.shape
    var = mu * (1 - mu) if var is None else np.asarray(var, dtype=float)
    base = int(np.argmin(kappa))
    cur = np.full(S, base)
    cost = float((w * kappa[cur]).sum())
    gain = 0.0
    path = [{"cost": cost, "gain": 0.0, "se": 0.0, "upgrade": None}]
    while True:
        best = None
        for s in range(S):
            for a in range(A):
                dc = w[s] * (kappa[a] - kappa[cur[s]])
                dg = w[s] * (mu[s, a] - mu[s, cur[s]])
                if dc <= 0 or dg <= 0:
                    continue
                r = dg / dc
                if best is None or r > best[0]:
                    best = (r, s, a, dc, dg)
        if best is None:
            break
        _, s, a, dc, dg = best
        path.append({"cost": cost + dc, "gain": gain + dg, "upgrade": (int(s), int(cur[s]), int(a))})
        cost, gain = cost + dc, gain + dg
        cur[s] = a
        with np.errstate(divide="ignore", invalid="ignore"):
            se2 = sum(w[t] ** 2 * (var[t, cur[t]] / max(n[t, cur[t]], 1) + var[t, base] / max(n[t, base], 1))
                      for t in range(S) if cur[t] != base)
        path[-1]["se"] = float(np.sqrt(se2))
    return path


class MaqQini(Method):
    name = "B5"
    alloc_kind = "pool"
    validity = "asymptotic"

    def __init__(self, name="B5"):
        self.name = name

    def setup(self, ctx):
        self.z = float(norm.ppf(1.0 - ctx.delta / len(ctx.checkpoints)))

    def certify(self, ctx, st):
        var = np.where(st.n > 0, plugin_var(st, ctx.R), 0.25 * ctx.R * ctx.R)
        J = jhat_all(ctx, st.mu_hat)
        ihs = pi_hat_indices(ctx, st.mu_hat)
        out = []
        for q in range(ctx.Q):
            V = glr_V(ctx, st, var, ihs[q])
            U = np.where(np.isinf(V), np.inf, (J - J[ihs[q]]) + self.z * np.sqrt(np.where(np.isinf(V), 0.0, V)))
            Uq = float(np.max(np.where(ctx.feas[q], U, -np.inf)))
            out.append((bool(Uq <= ctx.eps), int(ihs[q]), Uq))
        return out

    def describe(self):
        d = super().describe()
        d.update({"z_time_bonferroni": getattr(self, "z", None), "note": "asymptotic, not finite-sample valid",
                  "implementation": "self-implemented (maq PyPI sdist not installable)"})
        return d
