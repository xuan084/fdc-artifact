"""T-Lin model class: theta = (alpha_1..L, beta_1..R, gamma_1..L, psi_1..psi_{nb-1}).

Feature of a matched pair: x(i, j, n_i, b) = e_alpha_i + e_beta_j - n_i e_gamma_i + e_psi_b (b >= 1).
Under A1 the load trajectory of a deterministic policy is known, so J_theta(pi) = <z_pi, theta> exactly.
"""
from __future__ import annotations

import numpy as np

from ..core.actions import ActionSpace
from ..core.dynamics import next_loads


class LinClass:
    def __init__(self, L: int, R: int, nb: int, nmax: int = 3, static: bool = False):
        self.L, self.R, self.nb, self.nmax, self.static = L, R, nb, nmax, static
        self.d = 2 * L + R + (nb - 1)
        self.names = ([f"alpha{i}" for i in range(L)] + [f"beta{j}" for j in range(R)]
                      + [f"gamma{i}" for i in range(L)] + [f"psi{b}" for b in range(1, nb)])

    def slices(self):
        L, R = self.L, self.R
        return {"alpha": slice(0, L), "beta": slice(L, L + R), "gamma": slice(L + R, 2 * L + R),
                "psi": slice(2 * L + R, self.d)}

    def feature(self, i, j, n_i, b) -> np.ndarray:
        x = np.zeros(self.d)
        L, R = self.L, self.R
        x[i] = 1.0
        x[L + j] = 1.0
        if not self.static:
            x[L + R + i] = -float(n_i)
        if b >= 1:
            x[2 * L + R + b - 1] = 1.0
        return x

    def obs_rows(self, obs):
        X, y = [], []
        for i, j, b, val in obs.outcomes:
            X.append(self.feature(i, j, obs.loads[i], b))
            y.append(val)
        return np.array(X).reshape(-1, self.d), np.array(y)

    def z_per_step(self, policy, loads0, H: int, utility, aspace: ActionSpace) -> np.ndarray:
        """(H, d): z_{pi,t} with utility weights and scale applied; z_pi = sum over t."""
        P = self.L + self.R
        loads = np.asarray(loads0, np.int64).copy()
        ones = np.ones(P, dtype=np.int64)
        Z = np.zeros((H, self.d))
        for t in range(H):
            a_idx = policy.act(t, loads, ones)
            a = aspace.actions[a_idx]
            inc = {(i, j): l for i, j, l in a.incentives}
            for i, j in a.pairs:
                Z[t] += self.feature(i, j, loads[i], inc.get((i, j), 0))
            Z[t] *= utility.c_q * utility.w[t]
            loads = next_loads(loads, aspace, a_idx, self.nmax, self.static)
        return Z

    def z(self, policy, loads0, H, utility, aspace) -> np.ndarray:
        return self.z_per_step(policy, loads0, H, utility, aspace).sum(0)


def ridge_estimate(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    d = X.shape[1]
    V = lam * np.eye(d) + X.T @ X
    return np.linalg.solve(V, X.T @ y), V
