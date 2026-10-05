"""Time-uniform self-normalised ellipsoid for T-Lin (Abbasi-Yadkori et al. 2011, Thm 2).

Theta_t = {theta : ||theta - theta_hat_t||_{V_t} <= sqrt(beta_t)},
sqrt(beta_t) = sigma * sqrt(2 log(1/delta) + log det(V_t) - d log lam) + sqrt(lam) * S.
`beta_scale` != 1 gives the heuristic (non-guaranteed) variant; it is flagged in `guaranteed`.
"""
from __future__ import annotations

import numpy as np

from ..core.provenance import require_authentic


class EllipsoidSet:
    def __init__(self, model, sigma: float, delta: float, S: float, lam: float = 1.0, beta_scale: float = 1.0):
        self.model = model
        self.d = model.d
        self.sigma, self.delta, self.S, self.lam = sigma, delta, S, lam
        self.beta_scale = beta_scale
        self.V = lam * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.n_obs = 0
        self.n_rounds = 0
        self.serials: list[int] = []

    @property
    def guaranteed(self) -> bool:
        return self.beta_scale == 1.0

    def update(self, obs) -> None:
        require_authentic(obs)
        X, y = self.model.obs_rows(obs)
        if len(y):
            self.V += X.T @ X
            self.b += X.T @ y
            self.n_obs += len(y)
        self.n_rounds += 1
        self.serials.append(obs.serial)

    def theta_hat(self) -> np.ndarray:
        return np.linalg.solve(self.V, self.b)

    def sqrt_beta(self) -> float:
        _, logdet = np.linalg.slogdet(self.V)
        r = self.sigma * np.sqrt(2 * np.log(1 / self.delta) + logdet - self.d * np.log(self.lam)) + np.sqrt(self.lam) * self.S
        return float(self.beta_scale * r)

    def width(self, d: np.ndarray) -> np.ndarray:
        """||d||_{V^{-1}} for rows of d."""
        d = np.atleast_2d(d)
        Vi_d = np.linalg.solve(self.V, d.T)
        return np.sqrt(np.maximum((d.T * Vi_d).sum(0), 0.0))

    def sup_linear(self, d: np.ndarray) -> np.ndarray:
        """sup_{theta in Theta_t} <d, theta> (closed form) for rows of d."""
        d = np.atleast_2d(d)
        return d @ self.theta_hat() + self.sqrt_beta() * self.width(d)

    def argsup_linear(self, d: np.ndarray) -> np.ndarray:
        th = self.theta_hat()
        w = self.width(d)[0]
        if w <= 0:
            return th
        return th + self.sqrt_beta() * np.linalg.solve(self.V, d) / w

    def contains(self, theta: np.ndarray, tol: float = 1e-9) -> bool:
        diff = theta - self.theta_hat()
        return float(diff @ self.V @ diff) <= self.sqrt_beta() ** 2 * (1 + tol)

    def state_digest(self):
        return (self.V.copy(), self.b.copy(), self.n_obs, tuple(self.serials))
