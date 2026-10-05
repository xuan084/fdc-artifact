"""DF-sep-ledger: model-free whole-trajectory paired betting certificate with a stream-wide evidence ledger
(methodology 1.6, E-ms(c) cost reference). Learner side: no class model, no truth.

Sampling: one 'pull' of policy a = handle.reset_to(s0) + H real platform rounds under a (each round billed through
  sw.record; resets are counted by the platform in n_resets and reported, as for the r0-r3 whole-trial baselines).
Ledger: every executed trajectory is stored as (ysum_t (H,), engaged count at H) under the key
  (H, s0, behaviour fingerprint) where the fingerprint is the action of the policy at every reachable (t, state)
  (two policies with the same fingerprint from the same s0 generate identically distributed trajectories). A later
  problem re-scores matching ledger trajectories under its own utility -- the ledger is shared across the stream.
Statistic: for every unordered pair (a, b) the i-th trajectories of a and b form the i-th paired sample
  D_i = U_a,i - U_b,i in [-u_max, u_max], mapped to [0, 1]; two-sided hedged betting CS (fcc.BettingCS, WSR 2024),
  alpha_pair = delta / (n_problems * n_pairs) (Bonferroni over the problems of the stream and the pairs).
Certificate: leader = argmax empirical mean; CERTIFIED iff for every b != leader, LCB(U_leader - U_b) >= -eps.
Sampling rule: LUCB-style -- pull the leader and the challenger (largest UCB of U_b - U_leader), one trajectory each.
"""
from __future__ import annotations

import time

import numpy as np

from ..certify.fcc import BettingCS, DecoupledSolver
from .ms_common import Method, u_max


def fingerprint(solver: DecoupledSolver, policy, loads0, engaged0, H):
    pl = solver.plan(policy, loads0, engaged0, H)
    return (int(H), tuple(int(x) for x in loads0), tuple(int(x) for x in engaged0),
            tuple((tuple(int(s) for s in st), tuple(int(a) for a in ac)) for st, ac in zip(pl.states, pl.acts)))


class DFSepLedger(Method):
    name = "DF-sep-ledger"

    def __init__(self, pub, n_problems: int | None = None):
        super().__init__(pub)
        P = pub.aspace
        self.solver = DecoupledSolver(P.L, P.R, pub.nmax, P, pub.rho_ret)
        self.n_problems = int(n_problems or len(pub.problems))
        self.ledger: dict = {}
        self.n_reused_traj = 0

    def _pull(self, sw, handle, q, pol, key):
        handle.reset_to(q.loads0, q.engaged0)
        ys = np.zeros(q.H)
        obs = None
        for t in range(q.H):
            loads, eng = handle.observable_state()
            a = int(pol.act(t, loads, eng))
            obs = self.step(sw, handle, a)
            ys[t] = sum(int(o[3]) for o in obs.outcomes)
        rec = (ys, float(np.sum(obs.next_engaged)))
        self.ledger.setdefault(key, []).append(rec)
        return rec

    def solve(self, k, sw, lr, handle, rng, tmax):
        t0 = time.perf_counter()
        q = self.pub.problems[k]
        K, H = len(q.policies), q.H
        w, wr, cq = np.asarray(q.utility.w, float), float(q.utility.w_ret), float(q.utility.c_q)
        um = u_max(q)
        keys = [fingerprint(self.solver, p, q.loads0, q.engaged0, H) for p in q.policies]
        U = [[cq * (float(ys @ w) + wr * e) for ys, e in self.ledger.get(kk, [])] for kk in keys]
        reused = sum(len(u) for u in U)
        self.n_reused_traj += reused
        pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
        pidx = {p: i for i, p in enumerate(pairs)}
        alpha = self.pub.delta / (self.n_problems * max(len(pairs), 1))
        cs = BettingCS(max(len(pairs), 1), alpha)
        fed = np.zeros(len(pairs), dtype=np.int64)
        resets0 = handle.n_resets

        def feed():
            for (a, b), i in pidx.items():
                m = min(len(U[a]), len(U[b]))
                while fed[i] < m:
                    d = U[a][fed[i]] - U[b][fed[i]]
                    cs.update(i, (d + um) / (2.0 * um))
                    fed[i] += 1

        def diff_ci(a, b):                       # CI of U_a - U_b in utility units
            if a < b:
                i = pidx[(a, b)]
                return cs.lo[i] * 2 * um - um, cs.hi[i] * 2 * um - um
            lo, hi = diff_ci(b, a)
            return -hi, -lo

        status, pi, first = None, None, None
        n_pulls = 0
        while True:
            feed()
            if K == 1:
                status, pi = "CERTIFIED", 0
                break
            n = np.array([len(u) for u in U])
            if (n > 0).all():
                mu = np.array([np.mean(u) for u in U])
                lead = int(np.argmax(mu))
                lcb = [diff_ci(lead, b)[0] for b in range(K) if b != lead]
                if min(lcb) >= -self.pub.eps:
                    status, pi, first = "CERTIFIED", lead, sw.new_steps
                    break
                others = [b for b in range(K) if b != lead]
                ch = others[int(np.argmax([diff_ci(b, lead)[1] for b in others]))]
                todo = [lead, ch]
            else:
                todo = [int(a) for a in np.flatnonzero(n == n.min())[:2]]
            if sw.new_steps + H * len(todo) > tmax:
                status = "NEED_DATA"
                break
            for a in todo:
                ys, e = self._pull(sw, handle, q, q.policies[a], keys[a])
                U[a].append(cq * (float(ys @ w) + wr * e))
                n_pulls += 1
        return {"status": status, "pi": pi, "steps": sw.new_steps,
                "extra": {"n_pulls": n_pulls, "n_reused_traj": reused, "n_pairs": len(pairs), "alpha_pair": alpha,
                          "resets": int(handle.n_resets - resets0), "first_cert_step": first,
                          "wall_clock_s": time.perf_counter() - t0}}
