"""G-dp unit tests for certify/quadknap.py (methodology 1.4; r4 pre-lock revision R2)."""
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from dsswm.certify import quadknap as qk
from dsswm.stats.fp_eb import functional_width
from dsswm.streams.frontier import Problem, cr_problems, enumerate_policies, hr_problems


# ------------------------------------------------------------------------------------------- instance generators
def random_stats(rng, S, A, regime="unit", exhaust_p=0.1):
    w = rng.dirichlet(np.ones(S) * 2.0)
    if regime == "unit":
        mu = rng.uniform(0.0, 1.0, size=(S, A))
        n = rng.integers(1, 400, size=(S, A)).astype(float)
    else:                                   # CR-like: rare binary outcome, large counts, near ties
        base = rng.uniform(0.02, 0.06, size=(S, 1))
        mu = np.clip(base + rng.normal(0, 0.004, size=(S, A)), 0.001, 0.2)
        n = rng.integers(2_000, 400_000, size=(S, A)).astype(float)
    N = n + rng.integers(1, 1000, size=(S, A))
    ex = rng.random((S, A)) < exhaust_p
    N = np.where(ex, n, N)
    return qk.make_stats(w, mu, n, N=N, x_v=12.165)


def boundary_problem(rng, stats, pols, L1):
    """Budget-boundary instance: kappa with a free arm, budget set exactly to the cost of the worst challenger of an
    initial random budget (so the binding challenger sits on the boundary)."""
    A = stats.A
    kappa = np.concatenate([[0.0], rng.choice([0.5, 1.0, 1.5, 2.0, 0.37, 1.13], size=A - 1)])
    cost = (stats.w[None, :] * kappa[pols]).sum(1)
    p0 = Problem("b0", "visit", tuple(kappa), float(cost[rng.integers(len(pols))]))
    r = qk.qfc_certificate_enum(stats, [p0], L1, 0.0, pols=pols)[0]
    wc = np.asarray(r["worst"])
    B = float((stats.w * kappa[wc]).sum())
    return Problem("bnd", "visit", tuple(kappa), B)


# ------------------------------------------------------------------------------------------- external reviewer counterexample
def test_reviewer_counterexample_abstract():
    dhat, V, L = np.array([0.0, -1.0]), np.array([0.5, 2.0]), 1.0
    U_enum = float(np.max(dhat + np.sqrt(2 * L * V)))
    assert abs(U_enum - 1.0) < 1e-15
    U_cont, t = qk.minmax_t_cont(dhat, V, L)
    assert abs(U_cont - 13 / 12) < 1e-9
    assert abs(t - 4 / 3) < 1e-5
    assert abs((U_cont - U_enum) - 1 / 12) < 1e-9
    grid = np.geomspace(0.05, 50, qk.DEFAULT_T_POINTS)
    U_grid, _ = qk.minmax_t_grid(dhat, V, L, grid)
    assert U_grid >= 13 / 12 - 1e-12 and U_grid >= U_enum


def test_reviewer_counterexample_through_dp():
    """Same counterexample inside the cell model: S=1, A=3, pi_hat = arm 0 (exhausted => v = 0, no low-order term),
    arm 1: dhat 0, V 1/2; arm 2: dhat -1, V 2. Huge n makes b_bar L / 3 negligible (< 1e-9)."""
    n = np.array([[5.0, 1e12, 1e12]])
    stats = qk.QFCStats(w=np.array([1.0]), mu_hat=np.array([[0.0, 0.0, -1.0]]), n=n,
                        var_ucb=np.array([[0.0, 0.5e12, 2e12]]), N=np.array([[5.0, 1e13, 1e13]]))
    p = Problem("reviewer", "visit", (0.0, 0.0, 0.0), 1.0)
    r = qk.qfc_certificate_enum(stats, [p], 1.0, 0.0, pi_hat=[(0,)])[0]
    assert abs(r["U"] - 1.0) < 1e-9
    g = qk.gap_report(stats, p, 1.0, pi_hat=(0,), t_grid=np.geomspace(0.05, 50, 40))
    assert abs(g["U_enum"] - 1.0) < 1e-9
    assert abs(g["U_iii"] - 13 / 12) < 1e-8
    assert abs(g["gap_iii_minmax"] - 1 / 12) < 1e-8
    assert g["U_dp"] >= 13 / 12 - 1e-12
    assert not g["violation"]


# ------------------------------------------------------------------------------------------- enum == brute force
def brute_U(stats, problem, L1, pi_hat):
    S, A = stats.S, stats.A
    best = -np.inf
    kappa = np.asarray(problem.kappa)
    for pp in itertools.product(range(A), repeat=S):
        if (stats.w * kappa[list(pp)]).sum() > problem.budget + 1e-12:
            continue
        a = np.zeros((S, A))
        for s in range(S):
            if pp[s] != pi_hat[s]:
                a[s, pp[s]] += stats.w[s]
                a[s, pi_hat[s]] -= stats.w[s]
        dh = float((a * stats.mu_hat).sum())
        wid = functional_width(a.ravel(), stats.n.ravel(), stats.var_ucb.ravel(), L1,
                               N=None if stats.N is None else stats.N.ravel(), R=stats.R)
        best = max(best, dh + wid)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_enum_equals_brute_force(seed):
    rng = np.random.default_rng(1000 + seed)
    S, A = (6, 3) if seed % 3 == 0 else ((5, 2) if seed % 3 == 1 else (4, 3))
    stats = random_stats(rng, S, A, regime="unit" if seed % 2 else "cr")
    L1 = qk.l1_qstar(S, A, 15, 20, 0.04)
    probs = hr_problems() if A == 3 else cr_problems()
    probs = probs[:: 4]
    res = qk.qfc_certificate_enum(stats, probs, L1, 0.002)
    for r, p in zip(res, probs):
        bf = brute_U(stats, p, L1, r["pi_hat"])
        assert abs(bf - r["U"]) <= 1e-12 * max(1.0, abs(bf))


def test_pi_hat_is_feasible_argmax_and_U_nonneg():
    rng = np.random.default_rng(7)
    stats = random_stats(rng, 6, 3)
    pols = enumerate_policies(6, 3)
    for r, p in zip(qk.qfc_certificate_enum(stats, hr_problems(), 19.398, 0.01, pols=pols), hr_problems()):
        assert r["U"] >= 0.0                                   # pi' = pi_hat contributes 0
        cost = (stats.w * np.asarray(p.kappa)[list(r["pi_hat"])]).sum()
        assert cost <= p.budget + 1e-12
        assert r["certified"] == (r["U"] <= 0.01)


def test_zero_count_gives_inf_and_exhausted_cells_dropped():
    w = np.array([0.5, 0.5])
    mu = np.array([[0.1, 0.2], [0.3, 0.3]])
    n = np.array([[10.0, 0.0], [10.0, 10.0]])
    st = qk.QFCStats(w, mu, n, var_ucb=0.25)
    p = Problem("p", "visit", (0.0, 1.0), 1.0)
    assert math.isinf(qk.qfc_certificate_enum(st, [p], 10.0, 0.0)[0]["U"])
    assert math.isinf(qk.qfc_upper_dp(st, p, 10.0, pi_hat=(0, 0))["U_dp"])
    # all cells exhausted => estimates exact => U = max true regret against pi_hat, widths 0
    n2 = np.full((2, 2), 10.0)
    st2 = qk.QFCStats(w, mu, n2, var_ucb=0.25, N=n2)
    r = qk.qfc_certificate_enum(st2, [p], 10.0, 0.0)[0]
    assert abs(r["U"]) < 1e-15 and r["pi_hat"] == (1, 0)


def test_upper_dp_drops_unreachable_inf_choices():
    """An n = 0 cell that is infeasible under the (rounded-down) budget must not make U_DP infinite."""
    w = np.array([0.5, 0.5])
    mu = np.array([[0.1, 0.2], [0.3, 0.3]])
    n = np.array([[10.0, 0.0], [10.0, 10.0]])
    st = qk.QFCStats(w, mu, n, var_ucb=0.25)
    p = Problem("p", "visit", (0.0, 3.0), 1.0)                 # treating segment 0 costs 1.5 > 1
    p_ok = Problem("p", "visit", (0.0, 3.0), 1.0)
    r = qk.qfc_upper_dp(st, p, 10.0, pi_hat=(0, 0))
    assert math.isfinite(r["U_dp"])
    assert r["U_dp"] >= qk.qfc_certificate_enum(st, [p_ok], 10.0, 0.0, pi_hat=[(0, 0)])[0]["U"]


# ------------------------------------------------------------------------------------------- 100 boundary cases
def test_dp_upper_bound_100_boundary_instances():
    rng = np.random.default_rng(42)
    viol, prop_ok, n = 0, 0, 0
    for i in range(100):
        S = int(rng.integers(3, 9))
        A = 3 if (i % 2 == 0 and S <= 6) else 2
        stats = random_stats(rng, S, A, regime="unit" if i % 3 else "cr")
        pols = enumerate_policies(S, A)
        L1 = qk.l1_qstar(S, A, 15, 20, 0.04)
        p = boundary_problem(rng, stats, pols, L1)
        g = qk.gap_report(stats, p, L1, pols=pols)
        n += 1
        viol += int(g["violation"])
        prop_ok += int(g["proposal_feasible"])
        assert g["relaxed_superset"]
        assert g["dp_vs_enum_relaxed_maxabs"] <= 1e-12 * max(1.0, abs(g["U_dp"]))
        for k in ("gap_i_cost_rounding", "gap_ii_t_grid", "gap_iii_minmax", "gap_iv_low_order"):
            assert g[k] >= -1e-9, (k, g[k])
        parts = g["gap_i_cost_rounding"] + g["gap_ii_t_grid"] + g["gap_iii_minmax"] + g["gap_iv_low_order"]
        assert abs(parts - g["gap_total"]) <= 1e-12 * max(1.0, abs(g["U_dp"]))
    assert viol == 0 and prop_ok == n == 100


def test_naive_roundup_relaxation_is_detectably_invalid():
    """Power control: using rounded-UP costs as the 'relaxed' set loses the boundary challenger at least once."""
    rng = np.random.default_rng(42)
    bad = 0
    for i in range(100):
        S = int(rng.integers(3, 9))
        A = 3 if (i % 2 == 0 and S <= 6) else 2
        stats = random_stats(rng, S, A, regime="unit" if i % 3 else "cr")
        pols = enumerate_policies(S, A)
        L1 = qk.l1_qstar(S, A, 15, 20, 0.04)
        p = boundary_problem(rng, stats, pols, L1)
        r = qk.qfc_certificate_enum(stats, [p], L1, 0.0, pols=pols)[0]
        tg = qk.t_grid_default(stats, L1)
        naive = qk.qfc_upper_dp(stats, p, L1, pi_hat=r["pi_hat"], t_grid=tg, rounding="naive_up")
        mask_up = qk.relaxed_feasible_mask(pols, stats.w, p, "naive_up")
        feas = (stats.w[None, :] * np.asarray(p.kappa)[pols]).sum(1) <= p.budget + 1e-12
        bad += int(np.any(feas & ~mask_up))
        assert naive["U_dp"] is not None
    assert bad >= 1


def test_proposals_feasible_under_original_costs():
    rng = np.random.default_rng(3)
    for i in range(200):
        S = int(rng.integers(2, 17))
        A = int(rng.integers(2, 4))
        stats = random_stats(rng, S, A)
        kappa = (0.0,) + tuple(rng.uniform(0.1, 2.0, size=A - 1))
        B = float(rng.uniform(0.0, 1.5))
        pr = qk.propose_dp(stats, Problem("x", "visit", kappa, B))
        assert pr["feasible"] and pr["verified"]


def test_s16_dp_runs_and_upper_bounds_small_subproblem():
    rng = np.random.default_rng(11)
    stats = random_stats(rng, 16, 3, regime="cr")
    L1 = qk.l1_qstar(16, 3, 15, 20, 0.04)
    for p in hr_problems()[::5]:
        r = qk.qfc_upper_dp(stats, p, L1)
        assert np.isfinite(r["U_dp"]) and r["U_dp"] > 0
        assert r["proposal"]["feasible"]


def test_module_wording():
    src = (Path(qk.__file__)).read_text(encoding="utf-8")
    for bad in ("精确 DP", "精确DP", "gap→0", "gap -> 0", "exact DP", "Exact DP"):
        assert bad not in src
