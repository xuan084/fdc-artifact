"""Finite T-NL parameter class (Cartesian grid) on the GPU.

Per left participant i: (alpha_i, gamma_i, tauL_i) from a small grid; per right participant j: (beta_j, tauR_j);
global (psi, lam). |Theta| = k_L^L * k_R^R * k_psi * k_lam.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class NLGrid:
    L: int = 2
    R: int = 2
    alpha: tuple = (-1.0, 0.0, 1.0)
    gamma: tuple = (0.0, 0.5)
    tauL: tuple = (0.5, 1.5)
    beta: tuple = (-0.5, 0.5)
    tauR: tuple = (0.5, 1.5)
    psi: tuple = (0.0, 0.5, 1.0)
    lam: tuple = (0.0, 0.5)
    psi2: tuple = ()          # optional level-2 incentive effect (Type-2 problems); empty = not modelled
    extra: dict = field(default_factory=dict)

    def left_tuples(self):
        return list(itertools.product(self.alpha, self.gamma, self.tauL))

    def right_tuples(self):
        return list(itertools.product(self.beta, self.tauR))

    def global_tuples(self):
        p2 = self.psi2 if self.psi2 else (None,)
        return list(itertools.product(self.psi, self.lam, p2))

    def size(self) -> int:
        return len(self.left_tuples()) ** self.L * len(self.right_tuples()) ** self.R * len(self.global_tuples())

    def to_dict(self):
        return {k: list(getattr(self, k)) if isinstance(getattr(self, k), tuple) else getattr(self, k)
                for k in ("L", "R", "alpha", "gamma", "tauL", "beta", "tauR", "psi", "lam", "psi2")}


class NLClass:
    """Enumerated parameter class; index order = itertools.product(left^L, right^R, global)."""

    def __init__(self, grid: NLGrid, device="cuda", dtype=torch.float64):
        self.grid = grid
        self.L, self.R = grid.L, grid.R
        lt, rt, gt = grid.left_tuples(), grid.right_tuples(), grid.global_tuples()
        self.kL, self.kR, self.kG = len(lt), len(rt), len(gt)
        self.B = grid.size()
        self.device, self.dtype = device, dtype
        # mixed-radix digits
        radices = [self.kL] * self.L + [self.kR] * self.R + [self.kG]
        idx = np.arange(self.B)
        digits = []
        for r in reversed(radices):
            digits.append(idx % r)
            idx = idx // r
        digits = list(reversed(digits))
        self._radices = radices
        lt_a, rt_a, gt_a = np.array(lt, float), np.array(rt, float), np.array(gt, dtype=object)
        L, R = self.L, self.R
        alpha = np.stack([lt_a[digits[i], 0] for i in range(L)], 1)
        gamma = np.stack([lt_a[digits[i], 1] for i in range(L)], 1)
        tauL = np.stack([lt_a[digits[i], 2] for i in range(L)], 1)
        beta = np.stack([rt_a[digits[L + j], 0] for j in range(R)], 1)
        tauR = np.stack([rt_a[digits[L + j], 1] for j in range(R)], 1)
        g = digits[L + R]
        psi = np.array([gt[k][0] for k in g], float)
        lam = np.array([gt[k][1] for k in g], float)
        nb = 3 if grid.psi2 else 2
        psi_left = np.zeros((self.B, L, nb))
        psi_left[:, :, 1] = psi[:, None]
        if grid.psi2:
            psi_left[:, :, 2] = np.array([gt[k][2] for k in g], float)[:, None]
        self.nb = nb
        self.np_params = {"alpha": alpha, "beta": beta, "gamma": gamma, "tauL": tauL, "tauR": tauR,
                          "psi_left": psi_left, "lam": lam}

    def torch_params(self, sl: slice | None = None):
        sl = sl or slice(None)
        out = {k: torch.as_tensor(v[sl], device=self.device, dtype=self.dtype) for k, v in self.np_params.items()}
        return out

    def index_of(self, alpha, beta, gamma, tauL, tauR, psi, lam, psi2=None) -> int:
        """Index of a parameter vector lying on the grid (used by the generator only)."""
        lt, rt, gt = self.grid.left_tuples(), self.grid.right_tuples(), self.grid.global_tuples()
        digs = [lt.index((alpha[i], gamma[i], tauL[i])) for i in range(self.L)]
        digs += [rt.index((beta[j], tauR[j])) for j in range(self.R)]
        digs.append(gt.index((psi, lam, psi2 if self.grid.psi2 else None)))
        idx = 0
        for d, r in zip(digs, self._radices):
            idx = idx * r + d
        return idx

    def params_at(self, k: int) -> dict:
        return {key: v[k] for key, v in self.np_params.items()}
