"""FCC: factor-cell certificate for E1-NL policy values (learner side; round 4, methodology 1.4 / 1.6).

Factor cells (the contract; see exp/results/*/r4_setup_fcc/contract_table.md)
  pair cell       (i, j, n_i, b)  P(y_ij = 1 | both i and L+j engaged, load of i = n_i, incentive level b)
                                  updated only when the pair is ACTIVE (matched and both engaged at round start)
  retention cell  (p, n_p, y_p)   P(e_p' = 1 | p engaged, load of p before the load update = n_p, outcome of p's pair y_p)
                                  updated only when p is ENGAGED (y_p = 0 if p is unmatched or its pair is inactive)
  known constants rho_ret (return probability of a disengaged participant), the load rule (core.dynamics), c_q, w.

Statistics: one two-sided hedged betting confidence sequence per cell (Waudby-Smith & Ramdas 2024, JRSSB, bounded
mean by betting; predictable plug-in bets truncated at a constant c, i.e. m-INDEPENDENT bets, so the capital processes
are monotone in m and a finite grid gives a rigorous superset of the CS). Time index = the cell's own visit count.
Union over all cells: delta_cell = delta / n_cells. Running intersection over visits (valid: the CS is time-uniform).

Value bounds: decoupled (rectangular) relaxation = finite-horizon interval MDP for a FIXED deterministic policy. At every
(t, state) nature picks every cell parameter independently inside its interval; the one-step objective
    w_t * sum_active p_pair + E[V_{t+1}(s')]
is multilinear in the cell parameters used at (t, state) (each cell appears at most once per round), so its max / min
over the box is attained at a vertex and enumerated exactly (participants are grouped in <= 2-member groups; a group's
vertex set has <= 32 elements). The decoupled value upper-bounds max over the SHARED-parameter box (the shared model is
one feasible nature strategy); the shared-parameter vertex method is NOT valid because a cell can repeat inside one
trajectory (J then contains p^2, p(1-p), ...).
Optional BnB (value_bounds_bnb): bisect the <= 4 most influential cells; each node is bounded by the decoupled solver,
so the reported bound is valid at any time (no COMPUTE_UNKNOWN).
Outward rounding: per (t, state) the bound is padded by 64 ulp of the magnitude of the summed terms and then moved one
float outward with np.nextafter; CS endpoints are grid fractions k / 2^m moved outward with np.nextafter.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.actions import ActionSpace
from ..core.state import StateCodec

_EPS = np.finfo(float).eps
_PAD_ULP = 64.0


def _up(x):
    return np.nextafter(x, np.inf)


def _down(x):
    return np.nextafter(x, -np.inf)


# =============================================================================================== cell index
class CellIndex:
    """Enumerates factor cells. ids: pair cells first, then retention cells."""

    def __init__(self, L: int, R: int, nmax: int, nb: int):
        self.L, self.R, self.P, self.nmax, self.N, self.nb = L, R, L + R, nmax, nmax + 1, nb
        self.n_pair = L * R * self.N * nb
        self.n_ret = self.P * self.N * 2
        self.n = self.n_pair + self.n_ret

    def pair(self, i, j, n, b) -> int:
        return ((i * self.R + j) * self.N + n) * self.nb + b

    def ret(self, p, n, y) -> int:
        return self.n_pair + (p * self.N + n) * 2 + y

    def label(self, c: int) -> str:
        if c < self.n_pair:
            b = c % self.nb
            r = c // self.nb
            n = r % self.N
            r //= self.N
            j = r % self.R
            i = r // self.R
            return f"pair(i={i},j={j},n={n},b={b})"
        r = c - self.n_pair
        y = r % 2
        r //= 2
        n = r % self.N
        p = r // self.N
        return f"ret(p={p},n={n},y={y})"

    def kind(self, c: int) -> str:
        return "pair" if c < self.n_pair else "ret"

    def obs_updates(self, obs):
        """Cell updates of one authenticated observation, in within-round order (pair outcomes, then retention)."""
        L, P = self.L, self.P
        out = []
        y_p = [0] * P
        for (i, j, b, y, active) in obs.outcomes:
            if active:
                out.append((self.pair(i, j, int(obs.loads[i]), b), int(y)))
                y_p[i] = y_p[L + j] = int(y)
        for p in range(P):
            if obs.engaged[p]:
                out.append((self.ret(p, int(obs.loads[p]), y_p[p]), int(obs.next_engaged[p])))
        return out


def cell_probs(params: dict, cells: CellIndex, c: float) -> np.ndarray:
    """Cell-level success probabilities implied by a structural NL parameter dict (alpha, beta, gamma, tauL, tauR,
    psi_left (L, nb), lam, optional syn (L, R), g (N,)). Used for zero-width checks against the exact propagator."""
    L, R, N, nb = cells.L, cells.R, cells.N, cells.nb
    g = np.arange(N, dtype=float) if params.get("g") is None else np.asarray(params["g"], float)
    syn = np.zeros((L, R)) if params.get("syn") is None else np.asarray(params["syn"], float)
    psi_left = np.asarray(params["psi_left"], float)
    out = np.zeros(cells.n)
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
    for i in range(L):
        for j in range(R):
            for n in range(N):
                for b in range(nb):
                    out[cells.pair(i, j, n, b)] = sig(params["alpha"][i] + params["beta"][j] - params["gamma"][i] * g[n]
                                                      + psi_left[i, b] + syn[i, j])
    tau = np.concatenate([np.asarray(params["tauL"], float), np.asarray(params["tauR"], float)])
    for p in range(cells.P):
        for n in range(N):
            for y in range(2):
                out[cells.ret(p, n, y)] = sig(tau[p] + c * y - float(params["lam"]) * n)
    return out


# =============================================================================================== betting CS
class BettingCS:
    """Two-sided hedged betting CS for the means of n_cells bounded [0, 1] streams (WSR 2024, PrPl bets, truncation c).

    Capital at candidate mean m after t visits (lambda_i predictable, m-independent, 0 <= lambda_i <= c < 1):
        K+_t(m) = prod (1 + lambda_i (x_i - m)),  K-_t(m) = prod (1 - lambda_i (x_i - m)),
        M_t(m) = (K+_t(m) + K-_t(m)) / 2  -- a nonnegative martingale at the true mean (Ville: P(sup M >= 1/a) <= a).
    K+ is non-increasing and K- non-decreasing in m, so for m in [g_k, g_{k+1}]: M_t(m) >= (K+(g_{k+1}) + K-(g_k)) / 2.
    The grid interval is kept iff that lower envelope is < 1/a  =>  the kept union is a superset of the exact CS.
    """

    def __init__(self, n_cells: int, alpha_cell: float, grid_pow: int = 11, c: float = 0.5):
        assert 0 < c < 1
        self.n_cells, self.alpha = int(n_cells), float(alpha_cell)
        self.G = 2 ** int(grid_pow)
        self.grid = np.arange(self.G + 1, dtype=float) / self.G          # exact binary fractions
        self.c = float(c)
        self.log_thr = math.log(1.0 / self.alpha)
        self.logKp = np.zeros((self.n_cells, self.G + 1))
        self.logKm = np.zeros((self.n_cells, self.G + 1))
        self.t = np.zeros(self.n_cells, dtype=np.int64)
        self.sx = np.zeros(self.n_cells)
        self.ssd = np.zeros(self.n_cells)                                 # sum (x_i - mu_hat_{i-1})^2 (PrPl)
        self.lo = np.zeros(self.n_cells)
        self.hi = np.ones(self.n_cells)
        self.empty = np.zeros(self.n_cells, dtype=bool)                  # contract alarm (CS became empty)
        self._lam_const = 2.0 * math.log(2.0 / self.alpha)

    def _lambda(self, c_id: int) -> float:
        t = int(self.t[c_id]) + 1                                         # index of the upcoming visit
        var = (0.25 + self.ssd[c_id]) / t                                 # sigma_hat^2_{t-1}
        lam = math.sqrt(self._lam_const / (var * t * math.log(1.0 + t)))
        return min(self.c, lam)

    def update(self, c_id: int, x: float):
        x = float(x)
        lam = self._lambda(c_id)
        t = int(self.t[c_id])
        mu_prev = (0.5 + self.sx[c_id]) / (t + 1)
        d = x - self.grid
        self.logKp[c_id] += np.log1p(lam * d)
        self.logKm[c_id] += np.log1p(-lam * d)
        self.t[c_id] = t + 1
        self.sx[c_id] += x
        self.ssd[c_id] += (x - mu_prev) ** 2
        self._refresh(c_id)

    def update_many(self, updates):
        for c_id, x in updates:
            self.update(c_id, x)

    def _refresh(self, c_id: int):
        lp, lm = self.logKp[c_id], self.logKm[c_id]
        env = np.logaddexp(lp[1:], lm[:-1]) - math.log(2.0)               # lower envelope on [g_k, g_{k+1}]
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

    def intervals(self):
        return self.lo.copy(), self.hi.copy()


# =============================================================================================== value bounds
_VBITS = np.array(list(itertools.product((0, 1), repeat=5)), dtype=bool)        # (32, 5) vertex selector


@dataclass
class _Plan:
    states: list                       # per t: reachable state codes (t = 0..H)
    acts: list                         # per t < H: actions
    cellids: list                      # per t < H: (S_t, G, 5) extended cell ids (p, ra1, ra0, rb1, rb0)
    nxt: list                          # per t < H: (S_t, 4^G) index into states[t+1] (len(states[t+1]) = zero slot)
    esum: np.ndarray = None            # (S_H,) engaged count of terminal states


class DecoupledSolver:
    """Backward interval value iteration for a fixed deterministic policy on E1-NL (cells + known constants).

    Participants are partitioned per (state, action) into G = ceil(P/2) groups of two (matched pairs first, the rest
    paired arbitrarily, a dummy member pads an odd count). Every group is written in the pair form
        P(e_a', e_b') = p r_a1(e_a') r_b1(e_b') + (1 - p) r_a0(e_a') r_b0(e_b'),  E[y] = p
    with 5 parameters; for a non-active group p = 0 and r_a0 / r_b0 are the members' y = 0 retention cells, the known
    rho_ret (disengaged) or 0 (dummy). Extended ids: n_cells -> constant 0, n_cells + 1 -> constant rho_ret."""

    def __init__(self, L: int, R: int, nmax: int, aspace: ActionSpace, rho_ret: float, static: bool = False):
        self.L, self.R, self.P, self.nmax = L, R, L + R, nmax
        self.aspace, self.rho = aspace, float(rho_ret)
        self.cells = CellIndex(L, R, nmax, aspace.nb)
        self.codec = StateCodec(L, R, nmax, engagement=True)
        self.static = static
        self.G = (self.P + 1) // 2
        self.ZERO, self.RHO = self.cells.n, self.cells.n + 1
        self._pow2 = 2 ** np.arange(self.P)
        self._struct_cache: dict = {}
        self.n_solves = 0
        letters = "abcdefgh"[: self.G]
        ein_e = "ijklmnop"[: self.G]
        # val[s, v_1..v_G] = sum_e prod_g Q_g[s, v_g, e_g] V[s, e_1..e_G]
        self._ein = ",".join(f"s{letters[g]}{ein_e[g]}" for g in range(self.G)) + f",s{ein_e}->s{letters}"

    def _next_loads(self, loads, a_idx):
        if self.static:
            return loads.copy()
        a = self.aspace
        matched = np.concatenate([a.matched_left(a_idx), a.matched_right(a_idx)])
        return np.where(matched, np.minimum(loads + 1, self.nmax), np.maximum(loads - 1, 0)).astype(np.int64)

    def structure(self, code: int, a_idx: int):
        """(cellids (G, 5), members (G, 2), next-state codes (4^G,) with -1 for impossible combos)."""
        key = (code, a_idx)
        s = self._struct_cache.get(key)
        if s is not None:
            return s
        L, P, C, Z, RH = self.L, self.P, self.cells, self.ZERO, self.RHO
        loads, eng = self.codec.decode(code)
        a = self.aspace.actions[a_idx]
        inc = {(i, j): lv for i, j, lv in a.incentives}
        cid, mems, used = [], [], set()

        def r0(q):
            return C.ret(q, int(loads[q]), 0) if eng[q] else RH

        for (i, j) in a.pairs:
            pa, pb = i, L + j
            used.update((pa, pb))
            if eng[pa] and eng[pb]:
                cid.append((C.pair(i, j, int(loads[i]), inc.get((i, j), 0)),
                            C.ret(pa, int(loads[pa]), 1), C.ret(pa, int(loads[pa]), 0),
                            C.ret(pb, int(loads[pb]), 1), C.ret(pb, int(loads[pb]), 0)))
            else:
                cid.append((Z, Z, r0(pa), Z, r0(pb)))
            mems.append((pa, pb))
        rest = [p for p in range(P) if p not in used]
        for k in range(0, len(rest), 2):
            m = rest[k:k + 2]
            if len(m) == 2:
                cid.append((Z, Z, r0(m[0]), Z, r0(m[1])))
                mems.append((m[0], m[1]))
            else:
                cid.append((Z, Z, r0(m[0]), Z, Z))
                mems.append((m[0], -1))
        while len(cid) < self.G:
            cid.append((Z, Z, Z, Z, Z))
            mems.append((-1, -1))
        nl = self._next_loads(loads, a_idx)
        lc = int(np.dot(nl, self.codec._pow_n))
        codes = np.full(4 ** self.G, -1, dtype=np.int64)
        for flat, combo in enumerate(itertools.product(range(4), repeat=self.G)):
            e = np.zeros(P, dtype=np.int64)
            ok = True
            for g, (ma, mb) in enumerate(mems):
                for m_, bit in ((ma, combo[g] >> 1), (mb, combo[g] & 1)):
                    if m_ >= 0:
                        e[m_] = bit
                    elif bit:
                        ok = False
            if ok:
                codes[flat] = lc + self.codec.n_load_codes * int(e @ self._pow2)
        s = (np.array(cid, dtype=np.int64), np.array(mems, dtype=np.int64), codes)
        self._struct_cache[key] = s
        return s

    # ------------------------------------------------------------------ reachable states
    def plan(self, policy, loads0, engaged0, H: int) -> _Plan:
        s0 = self.codec.encode(loads0, engaged0)
        states = [np.array([s0], dtype=np.int64)]
        acts, cids, nxt_codes = [], [], []
        for t in range(H):
            cur = states[-1]
            a_t = np.array([policy.act(t, *self.codec.decode(int(c))) for c in cur], dtype=np.int64)
            st = [self.structure(int(c), int(a)) for c, a in zip(cur, a_t)]
            cids.append(np.stack([x[0] for x in st]))
            codes = np.stack([x[2] for x in st])
            nxt_codes.append(codes)
            acts.append(a_t)
            states.append(np.unique(codes[codes >= 0]))
        nxt = []
        for t in range(H):
            nx = states[t + 1]
            codes = nxt_codes[t]
            idx = np.searchsorted(nx, np.where(codes >= 0, codes, 0))
            nxt.append(np.where(codes >= 0, idx, len(nx)))
        esum = np.array([self.codec.decode(int(c))[1].sum() for c in states[H]], dtype=float)
        return _Plan(states, acts, cids, nxt, esum)

    def used_cells(self, plan: _Plan):
        """cell -> number of reachable (t, state) where it enters."""
        allc = np.concatenate([c.reshape(-1, ) for c in plan.cellids])
        allc = allc[allc < self.cells.n]
        u, n = np.unique(allc, return_counts=True)
        # count per (t, state) presence (a cell appears at most once per (t, state))
        return {int(a): int(b) for a, b in zip(u, n)}

    # ------------------------------------------------------------------ the backward pass
    def _step(self, cid, nxt, Vn, w_t, ext_lo, ext_hi):
        S = cid.shape[0]
        G = self.G
        lo5, hi5 = ext_lo[cid], ext_hi[cid]                                       # (S, G, 5)
        vals = np.where(_VBITS[None, None], hi5[:, :, None, :], lo5[:, :, None, :])  # (S, G, 32, 5)
        p, ra1, ra0, rb1, rb0 = np.moveaxis(vals, -1, 0)
        q1a = np.stack([1 - ra1, ra1], -1)
        q0a = np.stack([1 - ra0, ra0], -1)
        q1b = np.stack([1 - rb1, rb1], -1)
        q0b = np.stack([1 - rb0, rb0], -1)
        Q = (p[..., None, None] * q1a[..., :, None] * q1b[..., None, :]
             + (1 - p)[..., None, None] * q0a[..., :, None] * q0b[..., None, :]).reshape(S, G, 32, 4)
        out = {}
        for sense in ("hi", "lo"):
            Vext = np.append(Vn[sense], 0.0)
            Vg = Vext[nxt].reshape((S,) + (4,) * G)
            val = np.einsum(self._ein, *[Q[:, g] for g in range(G)], Vg, optimize=True)
            tot = val
            for g in range(G):
                shp = [S] + [1] * G
                shp[1 + g] = 32
                tot = tot + w_t * p[:, g].reshape(shp)
            flat = tot.reshape(S, -1)
            scale = abs(w_t) * G + np.max(np.abs(Vext)) + 1e-300
            pad = _PAD_ULP * _EPS * scale
            out[sense] = _up(flat.max(1) + pad) if sense == "hi" else _down(flat.min(1) - pad)
        return out

    def solve(self, plan: _Plan, w, w_ret: float, c_q: float, lo, hi):
        """(J_lo, J_hi): outward-rounded bounds of c_q * [sum_t w_t sum y + w_ret sum_p e_p(H)] under the
        decoupled (rectangular) relaxation of the cell box [lo, hi]."""
        self.n_solves += 1
        ext_lo = np.concatenate([np.asarray(lo, float), [0.0, self.rho]])
        ext_hi = np.concatenate([np.asarray(hi, float), [0.0, self.rho]])
        H = len(plan.acts)
        base = w_ret * plan.esum
        Vn = {"hi": _up(base), "lo": _down(base)}
        for t in range(H - 1, -1, -1):
            Vn = self._step(plan.cellids[t], plan.nxt[t], Vn, float(w[t]), ext_lo, ext_hi)
        a, b = c_q * Vn["lo"][0], c_q * Vn["hi"][0]
        if c_q < 0:
            a, b = b, a
        jlo = _down(a - _PAD_ULP * _EPS * abs(a))
        jhi = _up(b + _PAD_ULP * _EPS * abs(b))
        return float(jlo), float(jhi)


def value_bounds(solver: DecoupledSolver, policy, loads0, engaged0, H, utility, lo, hi, plan=None):
    plan = plan or solver.plan(policy, loads0, engaged0, H)
    return solver.solve(plan, np.asarray(utility.w, float), float(utility.w_ret), float(utility.c_q), lo, hi)


# =============================================================================================== BnB tightening
def value_bounds_bnb(solver: DecoupledSolver, plan: _Plan, utility, lo, hi, max_cells: int = 4, max_nodes: int = 64,
                     time_budget_s: float | None = None, sense: str = "hi"):
    """Branch and bound on <= max_cells most influential cells (influence = width x #(t, state) uses).
    sense 'hi': valid upper bound of max_{shared theta in box} J; 'lo': valid lower bound of min J.
    Every node is bounded by the decoupled solver, so the returned value is valid whenever the loop stops."""
    w, wr, cq = np.asarray(utility.w, float), float(utility.w_ret), float(utility.c_q)
    lo, hi = np.asarray(lo, float).copy(), np.asarray(hi, float).copy()
    use = solver.used_cells(plan)
    infl = sorted(((hi[c] - lo[c]) * n, c) for c, n in use.items() if hi[c] > lo[c])
    branch = [c for _, c in infl[::-1][:max_cells]]
    sgn = 1.0 if sense == "hi" else -1.0

    def bound(l_, h_):
        jl, jh = solver.solve(plan, w, wr, cq, l_, h_)
        return jh if sense == "hi" else jl

    def point(l_, h_):
        m = 0.5 * (l_ + h_)
        jl, jh = solver.solve(plan, w, wr, cq, m, m)          # zero width = exact shared value at the midpoint
        return jl if sense == "hi" else jh                    # conservative feasible value

    t0 = time.perf_counter()
    root = bound(lo, hi)
    out = {"decoupled": root, "branch_cells": branch, "nodes": 1}
    if not branch:
        out["bnb"] = root
        return out
    inc = point(lo, hi)
    heap = [(-sgn * root, 0, lo, hi)]
    tie = 1
    nodes = 1
    while heap and nodes < max_nodes:
        if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
            break
        if -heap[0][0] - sgn * inc <= 1e-12:      # best open node cannot beat the incumbent
            break
        _, _, l_, h_ = heapq.heappop(heap)
        c = max(branch, key=lambda x: h_[x] - l_[x])
        mid = 0.5 * (l_[c] + h_[c])
        for a, b in ((l_[c], mid), (mid, h_[c])):
            l2, h2 = l_.copy(), h_.copy()
            l2[c], h2[c] = a, b
            ub = bound(l2, h2)
            nodes += 1
            pv = point(l2, h2)
            if sgn * pv > sgn * inc:
                inc = pv
            if sgn * ub > sgn * inc:
                heapq.heappush(heap, (-sgn * ub, tie, l2, h2))
                tie += 1
    best_open = (-heap[0][0]) * sgn if heap else inc
    res = max(best_open, inc) if sense == "hi" else min(best_open, inc)
    out.update({"bnb": float(res), "incumbent": float(inc), "nodes": nodes,
                "time_s": time.perf_counter() - t0})
    return out


# =============================================================================================== certificate
class FCCCertifier:
    """Cells + CS + decoupled bounds. decide(): certify the policy with the largest lower bound if
    LB(pi_hat) >= max_{pi != pi_hat} UB(pi) - eps (epsilon-best policy identification)."""

    def __init__(self, L, R, nmax, aspace: ActionSpace, rho_ret: float, delta: float, grid_pow: int = 11,
                 static: bool = False):
        self.solver = DecoupledSolver(L, R, nmax, aspace, rho_ret, static=static)
        self.cells = self.solver.cells
        self.delta = float(delta)
        self.cs = BettingCS(self.cells.n, self.delta / self.cells.n, grid_pow=grid_pow)

    def observe(self, obs):
        self.cs.update_many(self.cells.obs_updates(obs))

    def bounds(self, problem, plans=None, bnb: bool = False, **bnb_kw):
        lo, hi = self.cs.intervals()
        plans = plans or [self.solver.plan(pi, problem.loads0, problem.engaged0, problem.H) for pi in problem.policies]
        out = []
        for pl in plans:
            if bnb:
                a = value_bounds_bnb(self.solver, pl, problem.utility, lo, hi, sense="lo", **bnb_kw)["bnb"]
                b = value_bounds_bnb(self.solver, pl, problem.utility, lo, hi, sense="hi", **bnb_kw)["bnb"]
                out.append((a, b))
            else:
                out.append(self.solver.solve(pl, np.asarray(problem.utility.w, float), float(problem.utility.w_ret),
                                             float(problem.utility.c_q), lo, hi))
        return np.array(out)

    @staticmethod
    def decide(bounds: np.ndarray, eps: float):
        LB, UB = bounds[:, 0], bounds[:, 1]
        k = int(np.argmax(LB))
        others = np.delete(UB, k)
        ok = (others.size == 0) or LB[k] >= others.max() - eps
        return ("CERTIFIED", k) if ok else ("CONTINUE", k)


# =============================================================================================== pair-difference bound
@dataclass
class _JointPlan:
    states: list                       # per t: union of states reachable under any interleaving of the two policies
    cid: tuple                         # (cid_a, cid_b): per t (S_t, G, 5)
    nxt: tuple                         # (nxt_a, nxt_b): per t (S_t, 4^G) index into states[t+1] (zero slot = len)
    same: list                         # per t (S_t,) bool: both policies take the same action
    esum: np.ndarray = None


def joint_plan(solver: DecoupledSolver, pi_a, pi_b, loads0, engaged0, H: int) -> _JointPlan:
    codec = solver.codec
    states = [np.array([codec.encode(loads0, engaged0)], dtype=np.int64)]
    cids, codes_all, same = ([], []), ([], []), []
    for t in range(H):
        cur = states[-1]
        dec = [codec.decode(int(c)) for c in cur]
        acts = [np.array([pi.act(t, *d) for d in dec], dtype=np.int64) for pi in (pi_a, pi_b)]
        same.append(acts[0] == acts[1])
        nxt_codes = []
        for side in (0, 1):
            st = [solver.structure(int(c), int(a)) for c, a in zip(cur, acts[side])]
            cids[side].append(np.stack([x[0] for x in st]))
            cd = np.stack([x[2] for x in st])
            codes_all[side].append(cd)
            nxt_codes.append(cd[cd >= 0])
        states.append(np.unique(np.concatenate(nxt_codes)))
    nxt = ([], [])
    for t in range(H):
        nx = states[t + 1]
        for side in (0, 1):
            cd = codes_all[side][t]
            idx = np.searchsorted(nx, np.where(cd >= 0, cd, 0))
            nxt[side].append(np.where(cd >= 0, idx, len(nx)))
    esum = np.array([codec.decode(int(c))[1].sum() for c in states[H]], dtype=float)
    return _JointPlan(states, cids, nxt, same, esum)


def _pdl_raw(solver, jp: _JointPlan, val: int, ch: int, w, w_ret, ext_lo, ext_hi):
    """Raw (unscaled) bounds of J(pi_ch) - J(pi_val) by the performance-difference lemma:
        J(ch) - J(val) = sum_t E_{s ~ d^ch_t}[ Q^val_t(s, ch(t, s)) - V^val_t(s) ].
    V^val / Q^val intervals come from the decoupled interval DP on the joint state set (they contain the true values
    because the true model is one feasible nature strategy); the advantage is EXACTLY 0 where both policies take the
    same action (shared cells cancel); the outer expectation over d^ch is bounded by a second decoupled DP."""
    H = len(jp.same)
    base = w_ret * jp.esum
    V = [None] * (H + 1)
    V[H] = {"hi": _up(base), "lo": _down(base)}
    for t in range(H - 1, -1, -1):
        V[t] = solver._step(jp.cid[val][t], jp.nxt[val][t], V[t + 1], float(w[t]), ext_lo, ext_hi)
    W = {"hi": np.zeros(len(jp.states[H])), "lo": np.zeros(len(jp.states[H]))}
    for t in range(H - 1, -1, -1):
        Q = solver._step(jp.cid[ch][t], jp.nxt[ch][t], V[t + 1], float(w[t]), ext_lo, ext_hi)
        a_hi = np.where(jp.same[t], 0.0, _up(Q["hi"] - V[t]["lo"]))
        a_lo = np.where(jp.same[t], 0.0, _down(Q["lo"] - V[t]["hi"]))
        E = solver._step(jp.cid[ch][t], jp.nxt[ch][t], W, 0.0, ext_lo, ext_hi)
        W = {"hi": _up(E["hi"] + a_hi), "lo": _down(E["lo"] + a_lo)}
    return float(W["lo"][0]), float(W["hi"][0]), float(V[0]["lo"][0]), float(V[0]["hi"][0])


def pair_difference_bounds(solver: DecoupledSolver, pi_a, pi_b, loads0, engaged0, H, utility, lo, hi, jp=None):
    """Valid bounds (dlo, dhi) of J(pi_a) - J(pi_b) for every shared cell parameter in the box [lo, hi]:
    intersection of the two PDL directions and of the independent difference [LB_a - UB_b, UB_a - LB_b] (each valid on
    its own, so the intersection is valid and never wider than the independent rule). Outward rounded."""
    jp = jp or joint_plan(solver, pi_a, pi_b, loads0, engaged0, H)
    ext_lo = np.concatenate([np.asarray(lo, float), [0.0, solver.rho]])
    ext_hi = np.concatenate([np.asarray(hi, float), [0.0, solver.rho]])
    w, wr, cq = np.asarray(utility.w, float), float(utility.w_ret), float(utility.c_q)
    l1, h1, bl, bh = _pdl_raw(solver, jp, val=1, ch=0, w=w, w_ret=wr, ext_lo=ext_lo, ext_hi=ext_hi)  # J_a - J_b
    l2, h2, al, ah = _pdl_raw(solver, jp, val=0, ch=1, w=w, w_ret=wr, ext_lo=ext_lo, ext_hi=ext_hi)  # J_b - J_a
    rl = max(l1, -h2, float(_down(al - bh)))
    rh = min(h1, -l2, float(_up(ah - bl)))
    a, b = cq * rl, cq * rh
    if cq < 0:
        a, b = b, a
    return float(_down(a - _PAD_ULP * _EPS * abs(a))), float(_up(b + _PAD_ULP * _EPS * abs(b)))


def decide_pdl(solver: DecoupledSolver, problem, lo, hi, eps: float, jps: dict | None = None):
    """Pair-difference certification rule (round-4 pre-lock FCC revision).
    D[j, k] = UB(J_j - J_k) from pair_difference_bounds; k_hat = argmin_k max_{j != k} D[j, k];
    eps_need = that min-max value. CERTIFIED iff eps_need <= eps. Validity: on the union CS event every D[j, k] is a
    valid upper bound for the shared true parameters, so J(pi*) - J(k_hat) <= D[pi*, k_hat] <= eps (no extra union).
    jps: optional cache {(j, k): joint_plan} for j < k. Returns (status, k_hat, eps_need, D)."""
    pols = problem.policies
    n = len(pols)
    D = np.full((n, n), -np.inf)
    for j in range(n):
        for k in range(j + 1, n):
            jp = None if jps is None else jps.get((j, k))
            if jp is None:
                jp = joint_plan(solver, pols[j], pols[k], problem.loads0, problem.engaged0, problem.H)
                if jps is not None:
                    jps[(j, k)] = jp
            dl, dh = pair_difference_bounds(solver, pols[j], pols[k], problem.loads0, problem.engaged0, problem.H,
                                            problem.utility, lo, hi, jp=jp)
            D[j, k], D[k, j] = dh, -dl
    if n == 1:
        return "CERTIFIED", 0, -np.inf, D
    colmax = D.max(axis=0)
    k = int(np.argmin(colmax))
    e = float(colmax[k])
    return ("CERTIFIED" if e <= eps else "CONTINUE"), k, e, D
