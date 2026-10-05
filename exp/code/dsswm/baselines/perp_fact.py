"""PERP-fact: per-factor-cell perturbation certificate (PERP-style robustness check, literature neighbour;
methodology 1.6, descriptive only, run on 100 MS-H instances). Learner side: no truth.

Point model: per factor cell the KT estimate p_hat_c = (successes + 1/2) / (visits + 1) (unvisited cells: 1/2).
Leader = argmax_pi J(pi; p_hat) (exact zero-width evaluation with the FCC decoupled solver).
Certificate: the leader is declared eps-optimal iff it stays eps-optimal under every ONE-AT-A-TIME perturbation of a
single cell used by the leader's or any rival's plan to an endpoint of that cell's betting CS (FCC CS, same
delta_cell), all other cells held at p_hat:  min_{c, endpoint} [J_lead - max_{pi != lead} J_pi] >= -eps.
This ignores joint perturbations, so it is NOT a valid simultaneous certificate (that is the point of the comparison).
Acquisition and checkpoints: identical to the FCC adapter.
"""
from __future__ import annotations

import time

import numpy as np

from .ms_common import CHECKPOINTS, FCC


class PERPFact(FCC):
    name = "PERP-fact"

    def __init__(self, pub):
        super().__init__(pub)
        n = self.cert.cells.n
        self.succ = np.zeros(n)
        self.vis = np.zeros(n)
        self._used = {}

    def observe(self, obs):
        super().observe(obs)
        for c, x in self.cert.cells.obs_updates(obs):
            self.vis[c] += 1
            self.succ[c] += x

    def _J(self, plans, q, p, only=None, base=None):
        w, wr, cq = np.asarray(q.utility.w, float), float(q.utility.w_ret), float(q.utility.c_q)
        out = np.array(base, float) if base is not None else np.zeros(len(plans))
        for i, pl in enumerate(plans):
            if only is None or i in only:
                out[i] = 0.5 * sum(self.cert.solver.solve(pl, w, wr, cq, p, p))
        return out

    def perp_check(self, k):
        q = self.pub.problems[k]
        plans = self.plans(k)
        p_hat = (self.succ + 0.5) / (self.vis + 1.0)
        J0 = self._J(plans, q, p_hat)
        lead = int(np.argmax(J0))
        if len(J0) == 1:
            return True, lead, 0.0
        lo, hi = self.cert.cs.intervals()
        if k not in self._used:
            self._used[k] = [set(self.cert.solver.used_cells(pl).keys()) for pl in plans]
        uses = self._used[k]
        used = set().union(*uses)
        worst = float(J0[lead] - np.delete(J0, lead).max())
        for c in sorted(used):
            for v in (lo[c], hi[c]):
                if v == p_hat[c]:
                    continue
                p = p_hat.copy()
                p[c] = v
                J = self._J(plans, q, p, only={i for i, u in enumerate(uses) if c in u}, base=J0)
                worst = min(worst, float(J[lead] - np.delete(J, lead).max()))
        return worst >= -self.pub.eps, lead, worst

    def solve(self, k, sw, lr, handle, rng, tmax):
        t0 = time.perf_counter()
        plans = self.plans(k)
        ci, n_checks, status, pi, first, worst = 0, 0, None, None, None, None
        while True:
            if sw.new_steps >= CHECKPOINTS[ci] or sw.new_steps >= tmax:
                while ci < len(CHECKPOINTS) - 1 and CHECKPOINTS[ci] <= sw.new_steps:
                    ci += 1
                n_checks += 1
                ok, lead, worst = self.perp_check(k)
                if ok:
                    status, pi, first = "CERTIFIED", lead, sw.new_steps
                    break
                b = self.bounds(k)
                ub = b[:, 1].copy()
                ub[lead] = -np.inf
                ch = int(np.argmax(ub))
                self.acq.set_influence(plans, (lead, ch))
            if sw.new_steps >= tmax:
                status = "NEED_DATA"
                break
            a = self.acq.choose(self.code_of(handle), self.widths(), rng)
            self.step(sw, handle, a)
        return {"status": status, "pi": pi, "steps": sw.new_steps,
                "extra": {"n_checks": n_checks, "first_cert_step": first, "perp_worst_margin": worst,
                          "wall_clock_s": time.perf_counter() - t0}}
