"""Unit tests for the FDC-bet exploration modules (baselines/fdc_bet.py, baselines/fdc_mr.py).  Synthetic only."""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import hypergeom

from dsswm.baselines.fdc import FDCMethod
from dsswm.baselines.fdc_bet import (FDCBet, direction_widths, kl_argmax_m, make_variant, psi_bar, psi_bennett_fpc,
                                     psi_kl)
from dsswm.baselines.fdc_mr import FDCMR, mr_ledger, partition_dp, rho_profile, subset_dp_tables, RHO_GRID
from dsswm.streams import frontier_runner as frr
from dsswm.streams.frontier_runner import build_ctx, run_stream
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


# --------------------------------------------------------------------------------------------- cell MGF bounds
@pytest.mark.parametrize("N", [10, 23, 40, 101])
def test_cell_bounds_dominate_exact_hypergeometric_mgf(N):
    """log E exp(s (X - n m)) of the exact hypergeometric law <= Bennett-FPC and <= binomial (u = n s)."""
    worst_b = worst_k = -np.inf
    for n in sorted({1, 2, N // 3, N // 2, N - 2, N - 1}):
        for M in range(N + 1):
            xs = np.arange(max(0, n - (N - M)), min(n, M) + 1)
            lp = hypergeom.logpmf(xs, N, M, n)
            m = M / N
            for s in np.concatenate([-np.logspace(-3, 1.5, 25), np.logspace(-3, 1.5, 25)]):
                exact = logsumexp(lp + s * (xs - n * m))
                worst_b = max(worst_b, exact - float(psi_bennett_fpc(n * s, n, N, m)))
                worst_k = max(worst_k, exact - float(psi_kl(n * s, n, m)))
    assert worst_b <= 1e-9 and worst_k <= 1e-9, (worst_b, worst_k)


def test_kl_argmax_and_psi_bar_is_sup():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 5000))
        N = n + int(rng.integers(1, 5000))
        lo = float(rng.uniform(0, 1))
        hi = float(min(1.0, lo + rng.uniform(0, 0.5)))
        u = float(rng.normal(0, 3 * n ** 0.5)) * rng.choice([1e-2, 1, 10])
        grid = np.linspace(lo, hi, 401)
        for kind in ("bennett", "kl", "min"):
            if kind == "bennett":
                vals = psi_bennett_fpc(u, n, N, grid)
            elif kind == "kl":
                vals = psi_kl(u, n, grid)
            else:
                vals = np.minimum(psi_bennett_fpc(u, n, N, grid), psi_kl(u, n, grid))
            sup = float(psi_bar(u, n, N, lo, hi, kind))
            assert sup >= np.max(vals) - 1e-9 * (1 + abs(sup)), (kind, n, N, lo, hi, u)
    s = np.array([-5.0, -0.3, -0.01, 0.02, 0.7, 4.0])      # s = 0: f == 0, argmax arbitrary
    m = np.linspace(0, 1, 100001)
    for si in s:
        f = np.log1p(m * np.expm1(si)) - si * m
        assert abs(m[np.argmax(f)] - float(kl_argmax_m(si))) < 1e-4


def test_direction_tail_monte_carlo():
    """P(Y > w(true mu)) <= e^-beta for a 4-cell direction under exact WoR sampling (beta small, MC check)."""
    rng = np.random.default_rng(7)
    N = np.array([[80, 150], [60, 200]])
    mu = np.array([[0.1, 0.35], [0.5, 0.05]])
    pools = [[np.r_[np.ones(int(round(mu[s, a] * N[s, a]))), np.zeros(N[s, a] - int(round(mu[s, a] * N[s, a])))]
              for a in range(2)] for s in range(2)]
    mu = np.array([[p.mean() for p in row] for row in pools])
    n = np.array([[30, 70], [25, 120]])

    class C:  # minimal ctx: 2 segments, full policy class
        S, A = 2, 2
        w = np.array([0.4, 0.6])
        pols = np.array(list(product(range(2), repeat=2)))
    beta = math.log(8.0)
    for kind in ("bennett", "kl", "min", "bernstein"):
        # centre pi_hat = (0, 0), challenger (1, 1): coefficient -w on (s, 1), +w on (s, 0)
        wd = direction_widths(C, 0, n, N, mu, mu, beta, kind, fpc=(kind != "bernstein"))
        w_true = wd[3]
        hits = 0
        T = 20000
        for _ in range(T):
            muh = np.array([[rng.permutation(pools[s][a])[:n[s, a]].mean() for a in range(2)] for s in range(2)])
            Y = sum(C.w[s] * ((muh[s, 0] - mu[s, 0]) - (muh[s, 1] - mu[s, 1])) for s in range(2))
            hits += Y > w_true
        p = hits / T
        assert p <= math.exp(-beta) + 3 * math.sqrt(math.exp(-beta) / T), (kind, p)


def test_widths_edge_cases():
    class C:
        S, A = 2, 2
        w = np.array([0.5, 0.5])
        pols = np.array(list(product(range(2), repeat=2)))
    N = np.array([[100, 100], [100, 100]])
    n = np.array([[100, 40], [0, 50]])          # cell (0,0) exhausted, (1,0) empty
    lo = np.array([[0.3, 0.1], [0.0, 0.2]])
    hi = np.array([[0.3, 0.4], [1.0, 0.5]])
    wd = direction_widths(C, 0, n, N, lo, hi, 5.0, "min")
    assert wd[0] == 0.0                           # empty difference set
    assert np.isinf(wd[1]) and np.isinf(wd[3])    # touches the empty cell (1, 0)
    assert np.isfinite(wd[2]) and wd[2] > 0       # segment 0 only: (0,0) exhausted, (0,1) live
    n2 = np.array([[100, 100], [0, 50]])
    wd2 = direction_widths(C, 0, n2, N, lo, hi, 5.0, "min")
    assert wd2[2] == 0.0                          # only exhausted cells touched -> exact


# --------------------------------------------------------------------------------------------- methods on the toy
def _toy_ctx(sd=900):
    env = _toy_population(sd)
    return env, build_ctx(env, PROBS, TOY_EPS)


def test_fdc_re_reproduces_fdc_exactly():
    env, ctx = _toy_ctx(901)
    for p in range(3):
        a, ra = run_stream(env, FDCMethod(), 901000 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=True)
        b, rb = run_stream(env, make_variant("FDC-re"), 901000 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=True)
        assert a["schedule_digest"] == b["schedule_digest"] and a["cert_k"] == b["cert_k"]
        assert a["decided_pi"] == b["decided_pi"] and a["N80"] == b["N80"]
        for x, y in zip(ra, rb):
            np.testing.assert_allclose(x["U"], y["U"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("mk", [lambda: make_variant("FDC-KLF+R"), lambda: FDCMR(rho="uniform"),
                                lambda: FDCMR(rho="mixF50"), lambda: FDCMR(rho="front3")])
def test_same_schedule_and_soundness_on_toy(mk):
    env, ctx = _toy_ctx(902)
    for p in range(3):
        a, _ = run_stream(env, FDCMethod(), 902000 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        b, _ = run_stream(env, mk(), 902000 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        assert a["schedule_digest"] == b["schedule_digest"] and b["billing_ok"] and not b["fwer_event"]


def test_tighter_components_never_looser_than_fdc_per_problem():
    """FDC-FPC <= FDC pointwise (FPC factor <= 1, same box, same beta)."""
    env, ctx = _toy_ctx(903)
    import copy
    c2 = copy.copy(ctx)
    c2.eps = -1.0
    c2.stop_k = 99
    a, ra = run_stream(env, FDCMethod(), 903001, PROBS, -1.0, ctx=c2, keep_U=True)
    b, rb = run_stream(env, make_variant("FDC-FPC"), 903001, PROBS, -1.0, ctx=c2, keep_U=True)
    for x, y in zip(ra, rb):
        assert np.all(np.asarray(y["U"]) <= np.asarray(x["U"]) + 1e-12)


# --------------------------------------------------------------------------------------------- multi-resolution
@pytest.mark.parametrize("rho", RHO_GRID)
def test_mr_ledger_spends_exactly_delta_main(rho):
    class C:
        S, A = 9, 2
        checkpoints = np.arange(20)
        feas = np.ones((15, 512), dtype=bool)
    led = mr_ledger(C, rho, 0.045)
    assert abs(led["bound_main"] - 0.045) < 1e-12
    assert abs(sum(rho_profile(9, rho)) - 1) < 1e-12
    assert all(np.isfinite(b) or r == 0 for b, r in zip(led["beta_by_size"], led["rho"]))


def test_partition_dp_matches_bruteforce():
    rng = np.random.default_rng(3)
    S = 5
    base = rng.uniform(0.1, 2.0, size=1 << S)
    base[0] = 0.0
    B = partition_dp(base, subset_dp_tables(S), S)

    def best(D):
        if D == 0:
            return 0.0
        r = base[D]
        low = D & -D
        T = (D - 1) & D
        while T:
            if T & low:
                r = min(r, best(T) + best(D ^ T))
            T = (T - 1) & D
        return r
    for D in range(1 << S):
        assert abs(B[D] - best(D)) < 1e-12
