"""Observable state (loads for all L+R participants, engagement bits) and an integer codec."""
from __future__ import annotations

import numpy as np


class StateCodec:
    """code = sum_p loads[p] * N^p + N^P * sum_p e[p] * 2^p, N = nmax + 1, P = L + R."""

    def __init__(self, L: int, R: int, nmax: int, engagement: bool):
        self.L, self.R, self.P = L, R, L + R
        self.nmax, self.N = nmax, nmax + 1
        self.engagement = engagement
        self._pow_n = self.N ** np.arange(self.P)
        self._pow_e = 2 ** np.arange(self.P)
        self.n_load_codes = self.N ** self.P
        self.size = self.n_load_codes * (2 ** self.P if engagement else 1)

    def encode(self, loads, engaged=None) -> int:
        c = int(np.dot(np.asarray(loads, dtype=np.int64), self._pow_n))
        if self.engagement:
            c += self.n_load_codes * int(np.dot(np.asarray(engaged, dtype=np.int64), self._pow_e))
        return c

    def encode_batch(self, loads: np.ndarray, engaged: np.ndarray | None = None) -> np.ndarray:
        c = loads.astype(np.int64) @ self._pow_n
        if self.engagement:
            c = c + self.n_load_codes * (engaged.astype(np.int64) @ self._pow_e)
        return c

    def decode(self, code: int) -> tuple[np.ndarray, np.ndarray]:
        lc = code % self.n_load_codes
        loads = (lc // self._pow_n) % self.N
        if self.engagement:
            ec = code // self.n_load_codes
            engaged = (ec // self._pow_e) % 2
        else:
            engaged = np.ones(self.P, dtype=np.int64)
        return loads.astype(np.int64), engaged.astype(np.int64)
