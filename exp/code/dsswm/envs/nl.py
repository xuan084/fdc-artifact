"""E1-NL ground truth (outcome-dependent dynamics).

y_ij ~ Bern(sigmoid(alpha_i + beta_j - gamma_i g(n_i) + psi_i[b] + syn_ij))   if both engaged, else y_ij = 0
engaged p:    e_p' ~ Bern(sigmoid(tau_p + c * y_p - lam * n_p))   (y_p = outcome of p's pair, 0 if none)
disengaged p: e_p' ~ Bern(rho_ret)
Loads follow the known rule in core.dynamics (driven by the matching action).
Default g(n) = n, psi_i = psi (global), syn = 0  -> in-class E1-NL. Misspecified variants in envs.mis.

Specification completions (round 1; gaps found by tier2_indep_replication, now fixed in writing and enforced here):
  (S1) A pair outcome y_ij is generated only if BOTH matched participants are engaged at the start of the round.
       Otherwise y_ij = 0 and the pair contributes no factor to the likelihood (outcome tuple flag active = 0).
  (S2) Retention of an engaged participant p: e_p' ~ Bern(sigmoid(tau_p + c * y_p - lam * n_p(t))), where n_p(t) is
       the load BEFORE this round's load update (next_loads is applied after retention is drawn).
  (S3) Trajectory utility U_q = c_q * [ sum_t w_{q,t} sum_{(i,j) in a_t} y_ij,t + w_ret * sum_p e_p(H) ], i.e. it
       includes the final-engagement (retention) term; c_q = 1 / max_{theta in class, pi in Pi_q} J_raw is fixed by
       the model class (public, data-free; streams.generator.nl_class_max_jtable).
These three statements are reproduced verbatim in the paper appendix and in plan/methodology.md (Appendix A).
"""
from __future__ import annotations

import numpy as np

from ..core import provenance
from ..core.actions import ActionSpace
from ..core.dynamics import next_loads, next_loads_batch, sigmoid
from .base import BaseEnv


class NLEnv(BaseEnv):
    kind = "nl"

    def __init__(self, alpha, beta, gamma, tauL, tauR, psi, lam, c=1.0, rho_ret=0.3, L=2, R=2, nmax=2,
                 budget=1, incentive_levels=(1,), seed=0, syn=None, g=None, psi_left=None, static=False):
        aspace = ActionSpace(L, R, budget, incentive_levels)
        super().__init__(L, R, nmax, engagement=True, aspace=aspace, seed=seed, static=static)
        nb = aspace.nb
        self._alpha = np.asarray(alpha, float)
        self._beta = np.asarray(beta, float)
        self._gamma = np.asarray(gamma, float)
        self._tau = np.concatenate([np.asarray(tauL, float), np.asarray(tauR, float)])
        psi_full = np.zeros(nb)
        psi_full[1:] = np.asarray(psi, float).reshape(-1)[: nb - 1]
        self._psi_left = (np.repeat(psi_full[None], L, 0) if psi_left is None else np.asarray(psi_left, float))
        self._lam = float(lam)
        self._c = float(c)
        self._rho_ret = float(rho_ret)
        self._syn = np.zeros((L, R)) if syn is None else np.asarray(syn, float)
        self._g = np.arange(nmax + 1, dtype=float) if g is None else np.asarray(g, float)

    def known_constants(self):
        return {"c": self._c, "rho_ret": self._rho_ret, "nmax": self.nmax}

    def _pair_logit(self, i, j, n_i, b):
        return self._alpha[i] + self._beta[j] - self._gamma[i] * self._g[n_i] + self._psi_left[i, b] + self._syn[i, j]

    def step(self, a_idx: int):
        a = self.aspace.actions[a_idx]
        inc = {(i, j): l for i, j, l in a.incentives}
        L = self.L
        y_p = np.zeros(self.P, dtype=np.int64)
        outs = []
        for i, j in a.pairs:
            b = inc.get((i, j), 0)
            active = bool(self.engaged[i] and self.engaged[L + j])
            y = 0
            if active:
                y = int(self.rng.random() < sigmoid(self._pair_logit(i, j, self.loads[i], b)))
                y_p[i] = y
                y_p[L + j] = y
            outs.append((int(i), int(j), int(b), int(y), int(active)))
        p_stay = sigmoid(self._tau + self._c * y_p - self._lam * self.loads)
        p = np.where(self.engaged == 1, p_stay, self._rho_ret)
        ne = (self.rng.random(self.P) < p).astype(np.int64)
        nl = next_loads(self.loads, self.aspace, a_idx, self.nmax, self.static)
        obs = provenance.issue(env_kind=self.kind, t=self.t, loads=tuple(int(x) for x in self.loads),
                               engaged=tuple(int(x) for x in self.engaged), action=int(a_idx), outcomes=tuple(outs),
                               next_loads=tuple(int(x) for x in nl), next_engaged=tuple(int(x) for x in ne))
        self.loads, self.engaged = nl, ne
        self.t += 1
        self.n_steps += 1
        return obs

    # ---- evaluation-only helpers ----
    def true_params(self):
        """Ground truth as a parameter dict for the exact propagator (oracle / evaluation only)."""
        return {"alpha": self._alpha.copy(), "beta": self._beta.copy(), "gamma": self._gamma.copy(),
                "tauL": self._tau[: self.L].copy(), "tauR": self._tau[self.L:].copy(),
                "psi_left": self._psi_left.copy(), "lam": self._lam, "syn": self._syn.copy(), "g": self._g.copy()}

    def simulate_batch(self, policy, loads0, engaged0, H, n_episodes, rng):
        """Vectorised MC episodes. Returns ysum (n, H) and final engaged count (n,)."""
        n, L, R, P = n_episodes, self.L, self.R, self.P
        loads = np.repeat(np.asarray(loads0, np.int64)[None], n, 0)
        eng = np.repeat(np.asarray(engaged0, np.int64)[None], n, 0)
        ysum = np.zeros((n, H))
        A = self.aspace
        for t in range(H):
            codes = self.codec.encode_batch(loads, eng)
            uniq, inv = np.unique(codes, return_inverse=True)
            acts_u = np.array([policy.act(t, *self.codec.decode(int(c))) for c in uniq], dtype=np.int64)
            acts = acts_u[inv]
            match = A.match[acts].astype(bool)            # (n, L, R)
            inc = A.inc[acts].astype(np.int64)            # (n, L, R)
            ii = np.arange(L)[None, :, None]
            logit = (self._alpha[None, :, None] + self._beta[None, None, :]
                     - self._gamma[None, :, None] * self._g[loads[:, :L]][:, :, None]
                     + self._psi_left[ii, inc] + self._syn[None])
            active = match & (eng[:, :L, None] == 1) & (eng[:, None, L:] == 1)
            y = active & (rng.random((n, L, R)) < sigmoid(logit))
            ysum[:, t] = y.sum((1, 2))
            yp = np.concatenate([y.any(2), y.any(1)], axis=1).astype(float)
            p_stay = sigmoid(self._tau[None] + self._c * yp - self._lam * loads)
            p = np.where(eng == 1, p_stay, self._rho_ret)
            new_eng = (rng.random((n, P)) < p).astype(np.int64)
            matched = np.concatenate([match.any(2), match.any(1)], axis=1)
            loads = next_loads_batch(loads, matched, self.nmax, self.static)
            eng = new_eng
        return ysum, eng.sum(1)
