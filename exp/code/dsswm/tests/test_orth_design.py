"""orth-replay design (evidence.orth_design): exact LMO, occupancy, column-generation feasibility / certificate."""
import itertools

import numpy as np

from dsswm.evidence.orth_design import TabularMDP, alignment, construct_design


def _toy(seed=0, S=4, A=3, K=2):
    rng = np.random.default_rng(seed)
    nxt = rng.integers(0, S, (S * A, K))
    prob = rng.random((S * A, K))
    prob /= prob.sum(1, keepdims=True)
    return TabularMDP(S, A, nxt, prob)


def test_occupancy_is_distribution():
    m = _toy()
    pol = np.random.default_rng(1).integers(0, m.nA, (5, m.nS))
    occ = m.occupancy(pol, 0)
    assert abs(occ.sum() - 1.0) < 1e-12 and (occ >= 0).all()


def test_dp_lmo_matches_brute_force():
    m = _toy(S=3, A=2)
    H = 3
    g = np.random.default_rng(2).standard_normal(m.nS * m.nA)
    pol, V = m.best_policy(g, H)
    best = np.inf
    for flat in itertools.product(range(m.nA), repeat=H * m.nS):
        p = np.array(flat).reshape(H, m.nS)
        best = min(best, m.occupancy(p, 0) @ g)
    assert abs(m.occupancy(pol, 0) @ g - best) < 1e-12
    assert abs(V[0] - best) < 1e-12


def test_construct_feasible_and_par():
    m = _toy(seed=3, S=5, A=3)
    rng = np.random.default_rng(4)
    t = rng.uniform(1, 2, m.nS * m.nA)
    c = [t * rng.uniform(0.0, 1.0, m.nS * m.nA)]
    c[0][::4] = 0.0                                   # some (s, a) carry no information along u
    T = float(m.occupancy(m.best_policy(-t, 4)[0], 0) @ t + m.occupancy(m.best_policy(t, 4)[0], 0) @ t) / 2
    perp = construct_design(m, 0, 4, t, c, T, mode="perp")
    par = construct_design(m, 0, 4, t, c, T, mode="par")
    occ = perp["occ"]
    assert abs(occ.sum() - 1) < 1e-9
    assert abs(perp["trace_err"]) <= 0.05 + 1e-9
    assert par["A"][0] >= perp["A"][0] - 1e-12
    if perp["feasible"]:
        assert perp["A"][0] <= 0.05 + 1e-9
    else:
        assert perp["converged"] and perp["certified_infeasible"]


def test_certificate_when_alignment_impossible():
    m = _toy(seed=5)
    t = np.ones(m.nS * m.nA)
    c = [0.5 * t]                                     # A = 0.5 for every design
    r = construct_design(m, 0, 3, t, c, 1.0, mode="perp")
    assert not r["feasible"] and r["certified_infeasible"] and abs(r["A"][0] - 0.5) < 1e-12


def test_alignment_definition():
    I = np.diag([4.0, 1.0])
    assert abs(alignment(I, np.array([1.0, 0.0])) - 0.8) < 1e-12
    assert abs(alignment(I, np.array([0.0, 2.0])) - 0.2) < 1e-12
