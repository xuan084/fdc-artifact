"""PolicyCert-tab: Dann et al. 2019 style non-factored tabular interval-MDP certificate (methodology 1.6, descriptive;
isolates the contribution of the factorisation, novelty v7). Learner side: no truth.

Cells are NOT shared across states: every parameter is indexed by the full (state code, action) context,
  pair slot   (code, a, i, j)        P(y_ij = 1 | state, action)           (active pairs only)
  ret slot    (code, a, p, y_p)      P(e_p' = 1 | state, action, y_p)      (engaged participants only)
so the transition of every (state, action) is learned separately (1296 states x 15 actions x 12 slots = 233,280
parameters; union bound delta / 233,280). Within one (state, action) the transition still factorises over the
participants given the pair outcomes (the platform's known conditional-independence structure), which is exactly what
lets the SAME decoupled backward solver as FCC compute rigorous value bounds (every slot appears at most once per
(t, state), so the vertex enumeration of fcc.DecoupledSolver is unchanged; only the cell ids are remapped).
CS: the FCC hedged betting CS, stored sparsely (only visited slots carry capital arrays; unvisited slots are [0, 1]).
Acquisition and checkpoints: identical to the FCC adapter (width x influence over the tabular slots).
"""
from __future__ import annotations

import math

import numpy as np

from ..certify.fcc import BettingCS, DecoupledSolver, FCCCertifier, _down, _up
from .ms_common import FCC, CellAcquirer

N_SLOTS = 12


class SparseBettingCS(BettingCS):
    """fcc.BettingCS with lazily allocated per-cell capital arrays (same bets, same grid, same validity)."""

    def __init__(self, n_cells: int, alpha_cell: float, grid_pow: int = 11, c: float = 0.5):
        assert 0 < c < 1
        self.n_cells, self.alpha = int(n_cells), float(alpha_cell)
        self.G = 2 ** int(grid_pow)
        self.grid = np.arange(self.G + 1, dtype=float) / self.G
        self.c = float(c)
        self.log_thr = math.log(1.0 / self.alpha)
        self._K: dict = {}
        self.t = np.zeros(self.n_cells, dtype=np.int64)
        self.sx = np.zeros(self.n_cells)
        self.ssd = np.zeros(self.n_cells)
        self.lo = np.zeros(self.n_cells)
        self.hi = np.ones(self.n_cells)
        self.empty = np.zeros(self.n_cells, dtype=bool)
        self._lam_const = 2.0 * math.log(2.0 / self.alpha)

    def update(self, c_id: int, x: float):
        x = float(x)
        lam = self._lambda(c_id)
        t = int(self.t[c_id])
        mu_prev = (0.5 + self.sx[c_id]) / (t + 1)
        if c_id not in self._K:
            self._K[c_id] = [np.zeros(self.G + 1), np.zeros(self.G + 1)]
        kp, km = self._K[c_id]
        d = x - self.grid
        kp += np.log1p(lam * d)
        km += np.log1p(-lam * d)
        self.t[c_id] = t + 1
        self.sx[c_id] += x
        self.ssd[c_id] += (x - mu_prev) ** 2
        self._refresh(c_id)

    def _refresh(self, c_id: int):
        lp, lm = self._K[c_id]
        env = np.logaddexp(lp[1:], lm[:-1]) - math.log(2.0)
        keep = env < self.log_thr
        if not keep.any():
            self.empty[c_id] = True
            return
        k = np.flatnonzero(keep)
        lo = max(0.0, float(_down(self.grid[k[0]])))
        hi = min(1.0, float(_up(self.grid[k[-1] + 1])))
        nlo, nhi = max(self.lo[c_id], lo), min(self.hi[c_id], hi)
        if nlo > nhi:
            self.empty[c_id] = True
            return
        self.lo[c_id], self.hi[c_id] = nlo, nhi


class TabularSolver(DecoupledSolver):
    """DecoupledSolver with factor-cell ids remapped to (state code, action, slot) ids."""

    def __init__(self, L, R, nmax, aspace, rho_ret):
        super().__init__(L, R, nmax, aspace, rho_ret)
        self.nA = aspace.n
        self.n_codes = self.codec.n_load_codes * 2 ** self.P
        self.n_tab = self.n_codes * self.nA * N_SLOTS
        self.ZERO, self.RHO = self.n_tab, self.n_tab + 1
        self._tab_cache: dict = {}
        C = self.cells
        self._slot = np.zeros(C.n, dtype=np.int64)
        for c in range(C.n):
            if c < C.n_pair:
                r = c // C.nb
                r //= C.N
                j = r % C.R
                i = r // C.R
                self._slot[c] = i * C.R + j
            else:
                r = c - C.n_pair
                y = r % 2
                r //= 2
                p = r // C.N
                self._slot[c] = C.L * C.R + 2 * p + y

    def tab_id(self, code, a_idx, slot):
        return (int(code) * self.nA + int(a_idx)) * N_SLOTS + int(slot)

    def structure(self, code: int, a_idx: int):
        key = (code, a_idx)
        s = self._tab_cache.get(key)
        if s is not None:
            return s
        cid, mems, codes = DecoupledSolver.structure(self, code, a_idx)
        base = self.tab_id(code, a_idx, 0)
        nf = self.cells.n
        out = np.where(cid < nf, base + self._slot[np.minimum(cid, nf - 1)], cid)
        s = (out.astype(np.int64), mems, codes)
        self._tab_cache[key] = s
        return s

    def used_cells(self, plan):
        allc = np.concatenate([c.reshape(-1) for c in plan.cellids])
        allc = allc[allc < self.n_tab]
        u, n = np.unique(allc, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, n)}

    def obs_updates(self, obs):
        code = self.codec.encode(np.asarray(obs.loads), np.asarray(obs.engaged))
        a = int(obs.action)
        L, P = self.L, self.P
        out, y_p = [], [0] * P
        for (i, j, b, y, active) in obs.outcomes:
            if active:
                out.append((self.tab_id(code, a, i * self.R + j), int(y)))
                y_p[i] = y_p[L + j] = int(y)
        for p in range(P):
            if obs.engaged[p]:
                out.append((self.tab_id(code, a, L * self.R + 2 * p + y_p[p]), int(obs.next_engaged[p])))
        return out


class TabularCertifier(FCCCertifier):
    def __init__(self, L, R, nmax, aspace, rho_ret, delta, grid_pow: int = 11):
        self.solver = TabularSolver(L, R, nmax, aspace, rho_ret)
        self.cells = self.solver.cells
        self.n_params = self.solver.n_tab
        self.delta = float(delta)
        self.cs = SparseBettingCS(self.n_params, self.delta / self.n_params, grid_pow=grid_pow)

    def observe(self, obs):
        self.cs.update_many(self.solver.obs_updates(obs))


class PolicyCertTab(FCC):
    name = "PolicyCert-tab"

    def __init__(self, pub):
        super().__init__(pub)
        P = pub.aspace
        self.cert = TabularCertifier(P.L, P.R, pub.nmax, P, pub.rho_ret, pub.delta)
        self.acq = CellAcquirer(self.cert.solver, np.arange(P.n), self.cert.n_params)
        self._plans = {}
