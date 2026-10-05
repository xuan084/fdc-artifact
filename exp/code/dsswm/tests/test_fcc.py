"""FCC (round 4, methodology 1.4): decoupled interval-MDP bounds, betting CS, outward rounding, zero-width exactness."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from dsswm.certify.fcc import BettingCS, CellIndex, DecoupledSolver, cell_probs, value_bounds_bnb
from dsswm.core.actions import ActionSpace
from dsswm.core.state import StateCodec
from dsswm.streams.utilities import Utility


# ------------------------------------------------------------------------------------------------ helpers
class TablePolicy:
    """Deterministic random policy pi(t, state) (keyed hash), restricted to actions with level <= max_level."""

    def __init__(self, aspace, seed, max_level=1):
        self.aspace, self.seed = aspace, seed
        self.allowed = [k for k in range(aspace.n) if aspace.max_level(k) <= max_level]
        self._c = {}

    def act(self, t, loads, engaged):
        key = (t, tuple(int(x) for x in loads), tuple(int(x) for x in engaged))
        if key not in self._c:
            h = abs(hash((self.seed,) + key)) % (2 ** 32)
            self._c[key] = int(self.allowed[np.random.default_rng(h).integers(len(self.allowed))])
        return self._c[key]


class ConstPolicy:
    def __init__(self, a):
        self.a = a

    def act(self, t, loads, engaged):
        return self.a


def forward_J(solver: DecoupledSolver, policy, loads0, eng0, H, util, theta):
    """INDEPENDENT forward evaluator of the shared-parameter value for a batch of cell vectors theta (B, n_cells):
    explicit enumeration of pair outcomes y and next engagement e' (no grouping, no vertex logic)."""
    theta = np.atleast_2d(theta)
    B = theta.shape[0]
    cells, codec, A, L, P, rho = solver.cells, solver.codec, solver.aspace, solver.L, solver.P, solver.rho
    dist = {codec.encode(loads0, eng0): np.ones(B)}
    val = np.zeros(B)
    for t in range(H):
        new = {}
        for code, pr in dist.items():
            loads, eng = codec.decode(code)
            a_idx = policy.act(t, loads, eng)
            a = A.actions[a_idx]
            inc = {(i, j): lv for i, j, lv in a.incentives}
            act_pairs = [(i, j) for (i, j) in a.pairs if eng[i] and eng[L + j]]
            nl = solver._next_loads(loads, a_idx)
            for ys in itertools.product((0, 1), repeat=len(act_pairs)):
                py = np.ones(B)
                yp = np.zeros(P, dtype=int)
                for (i, j), y in zip(act_pairs, ys):
                    q = theta[:, cells.pair(i, j, int(loads[i]), inc.get((i, j), 0))]
                    py = py * (q if y else 1 - q)
                    yp[i] = yp[L + j] = y
                val += util.w[t] * pr * py * sum(ys)
                for es in itertools.product((0, 1), repeat=P):
                    pe = np.ones(B)
                    for p_ in range(P):
                        r = theta[:, cells.ret(p_, int(loads[p_]), int(yp[p_]))] if eng[p_] else np.full(B, rho)
                        pe = pe * (r if es[p_] else 1 - r)
                    nc = codec.encode(nl, np.array(es))
                    new[nc] = new.get(nc, 0) + pr * py * pe
        dist = new
    for code, pr in dist.items():
        val += util.w_ret * pr * codec.decode(code)[1].sum()
    return util.c_q * val


def random_instance(seed):
    rng = np.random.default_rng([seed, 4001])
    L, R = [(1, 1), (1, 1), (1, 2), (2, 1), (2, 2)][rng.integers(5)]
    nmax = int(rng.integers(1, 3)) if L * R < 4 else 1
    H = int(rng.integers(2, 5)) if L * R < 4 else int(rng.integers(2, 4))
    aspace = ActionSpace(L, R, budget=1, incentive_levels=(1,))
    solver = DecoupledSolver(L, R, nmax, aspace, rho_ret=float(rng.uniform(0.1, 0.6)), static=bool(rng.random() < 0.3))
    n = solver.cells.n
    centre = rng.uniform(0, 1, n)
    width = rng.choice([0.0, 0.05, 0.3, 0.8], n)
    lo = np.clip(centre - width / 2, 0, 1)
    hi = np.clip(centre + width / 2, 0, 1)
    util = Utility(w=rng.uniform(-1, 1, H), w_ret=float(rng.uniform(-1, 1)), c_q=float(rng.uniform(0.2, 1.0)))
    loads0 = rng.integers(0, nmax + 1, L + R)
    eng0 = (rng.random(L + R) < 0.8).astype(np.int64)
    pol = TablePolicy(aspace, seed)
    return solver, pol, loads0, eng0, H, util, lo, hi, rng


def check_random_instance(seed, n_points=10_000, max_enum_cells=12):
    solver, pol, loads0, eng0, H, util, lo, hi, rng = random_instance(seed)
    plan = solver.plan(pol, loads0, eng0, H)
    jlo, jhi = solver.solve(plan, util.w, util.w_ret, util.c_q, lo, hi)
    used = sorted(solver.used_cells(plan))
    pts = lo + (hi - lo) * rng.random((n_points, solver.cells.n))
    if len(used) <= max_enum_cells:
        V = np.array(list(itertools.product((0, 1), repeat=len(used))), dtype=bool)
        vert = np.repeat(((lo + hi) / 2)[None], len(V), 0)
        vert[:, used] = np.where(V, hi[used], lo[used])
        n_vert = len(V)
    else:
        vert = np.where(rng.random((4096, solver.cells.n)) < 0.5, lo, hi)
        n_vert = -4096
    J = np.concatenate([forward_J(solver, pol, loads0, eng0, H, util, pts),
                        forward_J(solver, pol, loads0, eng0, H, util, vert)])
    return {"seed": seed, "jlo": jlo, "jhi": jhi, "Jmax": float(J.max()), "Jmin": float(J.min()),
            "n_used": len(used), "n_vertices": n_vert, "ok": bool(jhi >= J.max() and jlo <= J.min())}


# ------------------------------------------------------------------------------------------------ tests
def test_p_one_minus_p_counterexample():
    """Shared cell p in [0.2, 0.8] used twice: J = p - p^2. Vertex max 0.16 < interior 0.25 <= decoupled UB."""
    aspace = ActionSpace(1, 1, budget=1, incentive_levels=(1,))
    solver = DecoupledSolver(1, 1, 1, aspace, rho_ret=0.3, static=True)
    C = solver.cells
    lo, hi = np.zeros(C.n), np.zeros(C.n)
    pc = C.pair(0, 0, 1, 0)
    lo[pc], hi[pc] = 0.2, 0.8
    for p_ in range(2):                       # retention: stay iff y = 1 (known exactly)
        lo[C.ret(p_, 1, 1)] = hi[C.ret(p_, 1, 1)] = 1.0
    pol = ConstPolicy(aspace.index([(0, 0)]))
    util = Utility(w=np.array([1.0, -1.0]), w_ret=0.0, c_q=1.0)
    plan = solver.plan(pol, np.array([1, 1]), np.array([1, 1]), 2)
    jlo, jhi = solver.solve(plan, util.w, util.w_ret, util.c_q, lo, hi)
    ps = np.linspace(0.2, 0.8, 601)
    th = np.repeat(lo[None], len(ps), 0)
    th[:, pc] = ps
    J = forward_J(solver, pol, np.array([1, 1]), np.array([1, 1]), 2, util, th)
    assert abs(J - (ps - ps ** 2)).max() < 1e-12
    assert max(J[0], J[-1]) == pytest.approx(0.16)       # vertex method
    assert J.max() == pytest.approx(0.25)                # interior
    assert jhi >= 0.25 and jhi >= J.max()
    assert jhi == pytest.approx(0.8 - 0.8 * 0.2, abs=1e-12)  # decoupled: p0 = 0.8, p1 = 0.2
    assert jlo <= J.min()


@pytest.mark.parametrize("block", range(10))
def test_random_small_instances_upper_bound(block):
    """200 random small instances (20 per block): decoupled bounds contain the shared-parameter values at 10^4 random
    box points and at all vertices of the used cells."""
    for s in range(block * 20, block * 20 + 20):
        r = check_random_instance(s, n_points=10_000)
        assert r["ok"], r


def test_outward_rounding_exact_rationals():
    """Bounds bracket the EXACT rational value (Fractions) for zero-width dyadic intervals."""
    for s in range(20):
        solver, pol, loads0, eng0, H, util, lo, hi, rng = random_instance(1000 + s)
        theta = np.round(lo * 64) / 64
        util = Utility(w=np.round(util.w * 64) / 64, w_ret=round(util.w_ret * 64) / 64, c_q=1.0)
        solver.rho = round(solver.rho * 64) / 64
        plan = solver.plan(pol, loads0, eng0, H)
        jlo, jhi = solver.solve(plan, util.w, util.w_ret, util.c_q, theta, theta)
        exact = _exact_fraction(solver, pol, loads0, eng0, H, util, theta)
        assert Fraction(jlo) <= exact <= Fraction(jhi)
        assert jhi - jlo < 1e-12


def _exact_fraction(solver, pol, loads0, eng0, H, util, theta):
    F = Fraction
    cells, codec, A, L, P = solver.cells, solver.codec, solver.aspace, solver.L, solver.P
    th = [F(float(x)) for x in theta]
    rho = F(solver.rho)
    dist = {codec.encode(loads0, eng0): F(1)}
    val = F(0)
    for t in range(H):
        new = {}
        for code, pr in dist.items():
            loads, eng = codec.decode(code)
            a_idx = pol.act(t, loads, eng)
            a = A.actions[a_idx]
            inc = {(i, j): lv for i, j, lv in a.incentives}
            act_pairs = [(i, j) for (i, j) in a.pairs if eng[i] and eng[L + j]]
            nl = solver._next_loads(loads, a_idx)
            for ys in itertools.product((0, 1), repeat=len(act_pairs)):
                py = F(1)
                yp = [0] * P
                for (i, j), y in zip(act_pairs, ys):
                    q = th[cells.pair(i, j, int(loads[i]), inc.get((i, j), 0))]
                    py *= q if y else 1 - q
                    yp[i] = yp[L + j] = y
                val += F(float(util.w[t])) * pr * py * sum(ys)
                for es in itertools.product((0, 1), repeat=P):
                    pe = F(1)
                    for p_ in range(P):
                        r = th[cells.ret(p_, int(loads[p_]), yp[p_])] if eng[p_] else rho
                        pe *= r if es[p_] else 1 - r
                    nc = codec.encode(nl, np.array(es))
                    new[nc] = new.get(nc, F(0)) + pr * py * pe
        dist = new
    for code, pr in dist.items():
        val += F(float(util.w_ret)) * pr * int(codec.decode(code)[1].sum())
    return F(float(util.c_q)) * val


def test_nextafter_strictly_outward():
    solver, pol, loads0, eng0, H, util, lo, hi, _ = random_instance(7)
    plan = solver.plan(pol, loads0, eng0, H)
    jlo, jhi = solver.solve(plan, util.w, util.w_ret, util.c_q, lo, lo)
    J = forward_J(solver, pol, loads0, eng0, H, util, lo)[0]
    assert jlo < J < jhi


def test_zero_width_matches_exact_propagator():
    """Zero-width cell intervals at a structural theta reproduce exact/nl_propagate.py J to 1e-10 (H = 6 and 8)."""
    torch = pytest.importorskip("torch")
    from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
    from dsswm.streams.offgrid import make_offgrid_instance
    for seed in (800, 801):
        inst = make_offgrid_instance(seed, "R1", K=6)
        env = inst.env
        tp = env.true_params()
        prop = NLPropagator(2, 2, 2, env.aspace, c=1.0, rho_ret=0.3, device="cpu")
        solver = DecoupledSolver(2, 2, 2, env.aspace, rho_ret=0.3)
        theta = cell_probs(tp, solver.cells, c=1.0)
        pt = params_to_torch(tp, device="cpu")
        for prob in inst.problems:
            Jex = prop.j_table(pt, prob.policies, prob.loads0, prob.engaged0, prob.H, prob.utility)[0]
            for k, pi in enumerate(prob.policies):
                plan = solver.plan(pi, prob.loads0, prob.engaged0, prob.H)
                jlo, jhi = solver.solve(plan, prob.utility.w, prob.utility.w_ret, prob.utility.c_q, theta, theta)
                assert abs(jlo - Jex[k]) < 1e-10 and abs(jhi - Jex[k]) < 1e-10, (seed, prob.pid, k, jlo, jhi, Jex[k])
                assert jlo <= Jex[k] + 1e-13 and jhi >= Jex[k] - 1e-13


def test_bnb_valid_and_tighter():
    """BnB output is a valid upper bound (>= shared max over box points) and <= the decoupled bound."""
    aspace = ActionSpace(1, 1, budget=1, incentive_levels=(1,))
    solver = DecoupledSolver(1, 1, 1, aspace, rho_ret=0.3, static=True)
    C = solver.cells
    lo, hi = np.zeros(C.n), np.zeros(C.n)
    pc = C.pair(0, 0, 1, 0)
    lo[pc], hi[pc] = 0.2, 0.8
    for p_ in range(2):
        lo[C.ret(p_, 1, 1)] = hi[C.ret(p_, 1, 1)] = 1.0
    pol = ConstPolicy(aspace.index([(0, 0)]))
    util = Utility(w=np.array([1.0, -1.0]), w_ret=0.0, c_q=1.0)
    plan = solver.plan(pol, np.array([1, 1]), np.array([1, 1]), 2)
    r = value_bounds_bnb(solver, plan, util, lo, hi, max_cells=4, max_nodes=200)
    assert r["bnb"] >= 0.25 - 1e-12
    assert r["bnb"] <= r["decoupled"]
    assert r["bnb"] < 0.27                               # tightened from 0.64 towards 0.25
    rl = value_bounds_bnb(solver, plan, util, lo, hi, sense="lo", max_nodes=200)
    assert rl["bnb"] <= 0.16 + 1e-12


def test_betting_cs_coverage_and_shrinkage():
    rng = np.random.default_rng(42)
    mus = rng.uniform(0.02, 0.98, 300)
    cs = BettingCS(300, alpha_cell=0.05, grid_pow=10)
    miss = np.zeros(300, dtype=bool)
    for t in range(400):
        x = (rng.random(300) < mus).astype(float)
        for c in range(300):
            cs.update(c, x[c])
        lo, hi = cs.intervals()
        miss |= (mus < lo) | (mus > hi)
    assert miss.mean() <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 300)
    lo, hi = cs.intervals()
    assert np.median(hi - lo) < 0.25
    assert (lo >= 0).all() and (hi <= 1).all()


def test_cell_index_roundtrip_and_obs_updates():
    C = CellIndex(2, 2, 2, 2)
    assert C.n == 2 * 2 * 3 * 2 + 4 * 3 * 2 == 48
    ids = {C.pair(i, j, n, b) for i in range(2) for j in range(2) for n in range(3) for b in range(2)}
    ids |= {C.ret(p, n, y) for p in range(4) for n in range(3) for y in range(2)}
    assert ids == set(range(48))

    class O:
        loads, engaged = (1, 0, 2, 0), (1, 1, 1, 0)
        outcomes = ((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))
        next_engaged = (1, 0, 1, 1)
    up = C.obs_updates(O)
    assert up[0] == (C.pair(0, 0, 1, 1), 1)
    # p0 (y=1), p1 (inactive pair -> y=0), p2 (y=1); p3 disengaged -> no update (known rho)
    assert up[1:] == [(C.ret(0, 1, 1), 1), (C.ret(1, 0, 0), 0), (C.ret(2, 2, 1), 1)]
