"""E1-Lin / E1-Static ground truth: y_ij = alpha_i + beta_j - gamma_i n_i + psi[b] + N(0, sigma^2).
Known load dynamics (A1): the state trajectory does not depend on y."""
from __future__ import annotations

import numpy as np

from ..core import provenance
from ..core.actions import ActionSpace
from ..core.dynamics import next_loads, next_loads_batch
from .base import BaseEnv


class LinEnv(BaseEnv):
    kind = "lin"

    def __init__(self, alpha, beta, gamma, psi, sigma=0.5, L=3, R=3, nmax=3, budget=1,
                 incentive_levels=(1,), static=False, seed=0):
        aspace = ActionSpace(L, R, budget, incentive_levels)
        super().__init__(L, R, nmax, engagement=False, aspace=aspace, seed=seed, static=static)
        self._alpha = np.asarray(alpha, float)
        self._beta = np.asarray(beta, float)
        self._gamma = np.zeros(L) if static else np.asarray(gamma, float)
        self._psi = np.zeros(aspace.nb)
        self._psi[1:] = np.asarray(psi, float)[: aspace.nb - 1]
        self._sigma = float(sigma)

    def known_constants(self):
        return {"sigma": self._sigma, "nmax": self.nmax, "static": self.static}

    def _mean(self, i, j, n_i, b):
        return self._alpha[i] + self._beta[j] - self._gamma[i] * n_i + self._psi[b]

    def step(self, a_idx: int):
        a = self.aspace.actions[a_idx]
        inc = {(i, j): l for i, j, l in a.incentives}
        outs = []
        for i, j in a.pairs:
            b = inc.get((i, j), 0)
            y = self._mean(i, j, self.loads[i], b) + self._sigma * self.rng.standard_normal()
            outs.append((int(i), int(j), int(b), float(y)))
        nl = next_loads(self.loads, self.aspace, a_idx, self.nmax, self.static)
        obs = provenance.issue(env_kind=self.kind, t=self.t, loads=tuple(int(x) for x in self.loads),
                               engaged=tuple([1] * self.P), action=int(a_idx), outcomes=tuple(outs),
                               next_loads=tuple(int(x) for x in nl), next_engaged=tuple([1] * self.P))
        self.loads = nl
        self.t += 1
        self.n_steps += 1
        return obs

    # ---- evaluation-only helpers (independent of learner code) ----
    def true_theta_vector(self):
        """Truth in the learner's coordinate order (alpha, beta, gamma, psi[1:]). Oracle/evaluation only."""
        return np.concatenate([self._alpha, self._beta, self._gamma, self._psi[1:]])

    def simulate_batch(self, policy, loads0, H, n_episodes, rng):
        """MC episodes from loads0 under `policy`. Returns per-step summed outcomes (n, H)."""
        loads = np.repeat(np.asarray(loads0, np.int64)[None], n_episodes, 0)
        ysum = np.zeros((n_episodes, H))
        ones = np.ones(self.P, dtype=np.int64)
        for t in range(H):
            # A1: the trajectory is deterministic, but we still evaluate per unique state.
            codes = loads @ (self.nmax + 1) ** np.arange(self.P)
            for c in np.unique(codes):
                m = codes == c
                ld = loads[m][0]
                a_idx = policy.act(t, ld, ones)
                a = self.aspace.actions[a_idx]
                inc = {(i, j): l for i, j, l in a.incentives}
                tot = np.zeros(m.sum())
                for i, j in a.pairs:
                    b = inc.get((i, j), 0)
                    tot += self._mean(i, j, ld[i], b) + self._sigma * rng.standard_normal(m.sum())
                ysum[m, t] = tot
                matched = np.concatenate([self.aspace.matched_left(a_idx), self.aspace.matched_right(a_idx)])
                loads[m] = next_loads_batch(loads[m], np.repeat(matched[None], m.sum(), 0), self.nmax, self.static)
        return ysum


class StaticLinEnv(LinEnv):
    """E1-Static: loads frozen and gamma = 0 (the load channel is removed)."""
    kind = "lin_static"

    def __init__(self, *args, **kwargs):
        kwargs["static"] = True
        super().__init__(*args, **kwargs)
