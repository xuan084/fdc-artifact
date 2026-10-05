"""Multi-regime probe for the static-twin `probe20` arm (methodology 2.3 / hypotheses C2). Learner side: truth-free.

probe(m) = ev(m) whose m required new steps are FORCED to a legal action sequence that drives at least one
participant to load >= 2 and keeps it there. Loads follow the known public rule (core.dynamics): matched +1 (cap nmax),
unmatched -1 (floor 0); g(n) = n, so the static twin (g = g_ret = 1 reparametrisation, exact at load 1) is
misspecified at load 2 and the probe exposes that regime.

Rule (deterministic given the public state and the probe rng):
  * candidate set = legal actions whose matching covers the maximum number of participants (2x2: the two perfect
    matchings, each with no incentive or one level-1 incentive -> 6 actions; budget/levels from the public aspace)
  * if some participant already sits at load >= 2, restrict to candidates that keep at least one such participant
    matched (load stays at nmax), preferring LEFT participants (gamma_i enters the left logit only)
  * among those, maximise the number of INFORMATIVE pairs (both partners engaged -- an outcome is drawn only then --
    and the left partner at load >= 2); uniform choice among the maximisers with the probe rng
All steps are real platform rounds recorded through ReuseSwitch.record (billed).
"""
from __future__ import annotations

import numpy as np


class MultiRegimeProbe:
    def __init__(self, aspace, nmax: int):
        self.aspace = aspace
        self.nmax = int(nmax)
        self.L, self.R = aspace.L, aspace.R
        cover = aspace.match.reshape(aspace.n, -1).sum(1)
        self.cand = np.flatnonzero(cover == cover.max())
        self.ml = np.stack([aspace.matched_left(a) for a in range(aspace.n)])
        self.mr = np.stack([aspace.matched_right(a) for a in range(aspace.n)])

    def choose(self, loads, rng, engaged=None) -> int:
        loads = np.asarray(loads, int)
        hiL = np.flatnonzero(loads[: self.L] >= 2)
        hiR = np.flatnonzero(loads[self.L:] >= 2)
        c = self.cand
        if len(hiL):
            keep = [a for a in c if self.ml[a][hiL].any()]
            c = np.array(keep) if keep else c
        elif len(hiR):
            keep = [a for a in c if self.mr[a][hiR].any()]
            c = np.array(keep) if keep else c
        if engaged is not None:
            e = np.asarray(engaged, int)
            sc = np.array([sum(1 for i, j in self.aspace.actions[a].pairs
                               if e[i] and e[self.L + j] and loads[i] >= 2) for a in c])
            c = c[sc == sc.max()]
        return int(rng.choice(c))

    def run(self, m: int, sw, handle, rng) -> dict:
        """m forced probe steps through the switch (billed). Returns regime statistics of the probe rows."""
        n_hi, n_hiL, n_hiL_active = 0, 0, 0
        for _ in range(int(m)):
            loads, eng = handle.observable_state()
            a = self.choose(loads, rng, eng)
            obs = handle.step(a)
            sw.record(obs)
            ld = np.asarray(obs.loads, int)
            n_hi += int(ld.max() >= 2)
            hl = ld[: self.L] >= 2
            n_hiL += int(hl.any())
            n_hiL_active += int(any(o[4] == 1 and hl[o[0]] for o in obs.outcomes))
        return {"probe_steps": int(m), "probe_frac_load2": n_hi / max(m, 1),
                "probe_frac_left_load2": n_hiL / max(m, 1), "probe_frac_left_load2_active": n_hiL_active / max(m, 1)}
