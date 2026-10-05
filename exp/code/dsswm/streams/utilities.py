"""Trajectory utility U_q = c_q * (sum_t w_t * sum_pairs y_t + w_ret * sum_p e_p(H)), scaled into [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Utility:
    w: np.ndarray        # (H,) time weights, positive
    w_ret: float         # weight on final engagement count (0 for E1-Lin)
    c_q: float           # scale

    def value_from_components(self, ysum: np.ndarray, esum_final=None) -> np.ndarray:
        """ysum (n, H); esum_final (n,) -> per-episode utility (n,)."""
        u = ysum @ self.w
        if esum_final is not None and self.w_ret:
            u = u + self.w_ret * esum_final
        return self.c_q * u

    def to_dict(self):
        return {"w": [float(x) for x in self.w], "w_ret": float(self.w_ret), "c_q": float(self.c_q)}


def sample_utility(rng: np.random.Generator, H: int, L: int, R: int, with_retention: bool, y_scale: float) -> Utility:
    """Random time profile (front/back loaded) and random retention weight."""
    tt = np.arange(H) / max(H - 1, 1) - 0.5
    slope = rng.uniform(-1.5, 1.5)
    w = np.exp(slope * tt) * rng.uniform(0.8, 1.2, size=H)
    m = min(L, R)
    if with_retention:
        w_ret = float(rng.uniform(0.0, 1.0) * w.sum() * m / (L + R))
        c_q = 1.0 / (m * w.sum() + w_ret * (L + R))
    else:
        w_ret = 0.0
        c_q = 1.0 / (y_scale * m * w.sum())
    return Utility(w=w, w_ret=w_ret, c_q=float(c_q))
