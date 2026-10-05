"""Unit tests for baselines/fdc_hg.py (FDC-HG, plan/fdc_hg_theorem.md).  Synthetic only (no dev / eval data)."""
from __future__ import annotations

import copy
import math
import time
from itertools import product

import numpy as np
import pytest

from dsswm.baselines.fdc_bet import direction_widths, make_variant, psi_bar
from dsswm.baselines.fdc_hg import (FDCHG, CellEnvelope, _interval_bound, direction_widths_fn, exact_logmgf,
                                    hg_envelope, hg_logmgf_bounds)
from dsswm.streams.frontier_runner import build_ctx, run_stream
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


def _lattice_box(rng, N):
    Ml = int(rng.integers(0, N + 1))
    Mh = min(N, Ml + int(rng.integers(0, N // 2 + 1)))
    return Ml, Mh


# --------------------------------------------------------------------------------------------- Lemma T
def test_logmgf_bounds_bracket_exact_full_support():
    rng = np.random.default_rng(11)
    for _ in range(150):
        N = int(rng.integers(3, 3000))
        n = int(rng.integers(1, N))
        M = int(rng.integers(0, N + 1))
        t = np.concatenate([-np.logspace(-4, 1, 15), np.logspace(-4, 1, 15)])
        lo, hi, h_lo, h_hi = hg_logmgf_bounds(N, M, n, t)
        ex = exact_logmgf(t * n, n, N, M)
        assert np.all(lo <= ex) and np.all(hi >= ex), (N, n, M)
        assert np.all(hi - lo <= 1e-6 + 1e-6 * np.abs(ex))
        xs = np.arange(max(0, n - (N - M)), min(n, M) + 1)
        from scipy.stats import hypergeom
        lp = hypergeom.logpmf(xs, N, M, n)
        g = np.exp(lp[:, None] + t[None, :] * (xs[:, None] - xs.mean()) - (lp[:, None] + t[None, :] * (
            xs[:, None] - xs.mean())).max(0))
        h = (xs[:, None] * g).sum(0) / g.sum(0)
        assert np.all(h_lo <= h + 1e-7 * (1 + h)) and np.all(h_hi >= h - 1e-7 * (1 + h))


def test_logmgf_bounds_large_pool_window():
    """Large N, n: truncated window with tail bounds; compare with the full-support sum."""
    for N, n, M in ((430000, 30000, 129000), (2658749, 200000, 2600), (8619, 3000, 5), (193000, 14000, 48000)):
        t = np.array([-3e-2, -3e-3, -1e-4, 1e-4, 3e-3, 3e-2])
        lo, hi, _, _ = hg_logmgf_bounds(N, M, n, t)
        ex = exact_logmgf(t * n, n, N, M)                  # long-double full-support reference
        assert np.all(hi >= ex) and np.all(lo <= ex), (N, n, M, hi - ex, ex - lo)


# --------------------------------------------------------------------------------------------- Lemma R
def test_ratio_identity_and_interval_bound():
    rng = np.random.default_rng(5)
    for _ in range(60):
        N = int(rng.integers(5, 400))
        n = int(rng.integers(1, N))
        t = np.concatenate([-np.logspace(-3, 0.7, 8), np.logspace(-3, 0.7, 8)])
        # (a) ratio identity f(M+1)/f(M) = 1 + (e^t - 1)(n - h_t(M))/(N - M)
        M = int(rng.integers(0, N))
        l0 = exact_logmgf(t * n, n, N, M) + t * n * M / N           # log f(M)
        l1 = exact_logmgf(t * n, n, N, M + 1) + t * n * (M + 1) / N
        from scipy.stats import hypergeom
        xs = np.arange(max(0, n - (N - M)), min(n, M) + 1)
        lg = hypergeom.logpmf(xs, N, M, n)[:, None] + t[None, :] * xs[:, None]
        g = np.exp(lg - lg.max(0))
        h = (xs[:, None] * g).sum(0) / g.sum(0)
        np.testing.assert_allclose(l1 - l0, np.log1p(np.expm1(t) * (n - h) / (N - M)), rtol=1e-9, atol=1e-11)
        _, _, h_lo, h_hi = hg_logmgf_bounds(N, M, n, t)
        assert np.all(h_lo <= h + 1e-9) and np.all(h_hi >= h - 1e-9)
        # (b) interval bound >= exact on every M of the interval
        M1 = int(rng.integers(0, N))
        M2 = int(rng.integers(M1 + 1, N + 1))
        e1, e2 = hg_logmgf_bounds(N, M1, n, t), hg_logmgf_bounds(N, M2, n, t)
        bnd = _interval_bound(t, N, n, M1, M2, e1, e2)
        ex = np.max([exact_logmgf(t * n, n, N, M) for M in range(M1, M2 + 1)], axis=0)
        assert np.all(bnd >= ex - 1e-12), (N, n, M1, M2)


# --------------------------------------------------------------------------------------------- envelope
def test_envelope_dominates_exact_mgf_for_every_M_in_box():
    rng = np.random.default_rng(1)
    worst = -np.inf
    for _ in range(120):
        N = int(rng.integers(5, 600))
        n = int(rng.integers(1, N))
        Ml, Mh = _lattice_box(rng, N)
        e = CellEnvelope(n, N, Ml / N, Mh / N, M_lo=Ml, M_hi=Mh)
        us = np.concatenate([-np.logspace(-2, 2.5, 40), np.logspace(-2, 2.5, 40)]) * math.sqrt(n)
        val = e(us)
        ex = np.max([exact_logmgf(us, n, N, M) for M in range(Ml, Mh + 1)], axis=0)
        worst = max(worst, float(np.max(ex - val)))
    assert worst <= 1e-12, worst


def test_envelope_below_bennett_pointwise_and_tighter_somewhere():
    rng = np.random.default_rng(2)
    gains = []
    for _ in range(80):
        N = int(rng.integers(20, 5000))
        n = int(rng.integers(1, N))
        Ml, Mh = _lattice_box(rng, N)
        lo, hi = Ml / N, Mh / N
        e = CellEnvelope(n, N, lo, hi, M_lo=Ml, M_hi=Mh)
        us = np.concatenate([-np.logspace(-3, 3, 60), np.logspace(-3, 3, 60)]) * math.sqrt(n)
        b = psi_bar(us, n, N, lo, hi, "bennett")
        v = e(us)
        assert np.all(v <= b * (1 + 1e-10) + 1e-300), (N, n, lo, hi)
        m = (b > 0.5) & (b < 40)
        if m.any():
            gains.append(float(np.median(v[m] / b[m])))
    assert np.median(gains) < 0.95


def test_bb_converges_and_info():
    up, info = hg_envelope(5000, 800, 900, 1400, np.array([-0.05, 0.05]), return_info=True)
    assert info["slack"] <= 1e-3 + 1e-9 and info["n_eval"] < 100


def test_envelope_timing_large_cell():
    t0 = time.perf_counter()
    CellEnvelope(30000, 430000, 0.29, 0.31)
    CellEnvelope(200000, 2658749, 0.0009, 0.0011)
    assert time.perf_counter() - t0 < 5.0


# --------------------------------------------------------------------------------------------- widths
class _C:
    S, A = 3, 2
    w = np.array([0.2, 0.5, 0.3])
    pols = np.array(list(product(range(2), repeat=3)))


def _rand_state(rng):
    N = rng.integers(50, 3000, size=(3, 2))
    n = np.minimum(rng.integers(1, 3000, size=(3, 2)), N - 1)
    n[0, 0] = N[0, 0]                                       # one exhausted cell
    M = (rng.uniform(0.02, 0.6, size=(3, 2)) * N).astype(int)
    Ml = np.maximum(M - (0.1 * N).astype(int), 0)
    Mh = np.minimum(M + (0.1 * N).astype(int), N)
    return n, N, Ml / N, Mh / N


def test_widths_fn_reproduces_fdc_bf_and_hg_never_wider():
    rng = np.random.default_rng(3)
    for _ in range(10):
        n, N, lo, hi = _rand_state(rng)
        beta = 9.0
        envs = {}

        def ben(s, a, u):
            return psi_bar(u, n[s, a], N[s, a], lo[s, a], hi[s, a], "bennett")

        def hg(s, a, u):
            if (s, a) not in envs:
                envs[(s, a)] = CellEnvelope(n[s, a], N[s, a], lo[s, a], hi[s, a])
            return envs[(s, a)](u)
        for ih in range(_C.pols.shape[0]):
            ref = direction_widths(_C, ih, n, N, lo, hi, beta, "bennett")
            np.testing.assert_array_equal(direction_widths_fn(_C, ih, n, N, lo, hi, beta, ben), ref)
            w_hg = direction_widths_fn(_C, ih, n, N, lo, hi, beta, hg)
            assert np.all(w_hg <= ref * (1 + 1e-10) + 1e-15)


def test_direction_tail_monte_carlo():
    """P(Y > w_HG(true M)) <= e^-beta for a 4-cell direction under exact WoR sampling (MC)."""
    rng = np.random.default_rng(7)
    N = np.array([[80, 150], [60, 200]])
    Mtrue = np.array([[8, 52], [30, 10]])
    mu = Mtrue / N
    pools = [[np.r_[np.ones(Mtrue[s, a]), np.zeros(N[s, a] - Mtrue[s, a])] for a in range(2)] for s in range(2)]
    n = np.array([[30, 70], [25, 120]])

    class C:
        S, A = 2, 2
        w = np.array([0.4, 0.6])
        pols = np.array(list(product(range(2), repeat=2)))
    beta = math.log(8.0)
    envs = {(s, a): CellEnvelope(n[s, a], N[s, a], mu[s, a], mu[s, a]) for s in range(2) for a in range(2)}
    wd = direction_widths_fn(C, 0, n, N, mu, mu, beta, lambda s, a, u: envs[(s, a)](u))
    wb = direction_widths(C, 0, n, N, mu, mu, beta, "bennett")
    assert wd[3] < wb[3]
    T = 40000
    hits = 0
    for _ in range(T):
        muh = np.array([[rng.permutation(pools[s][a])[:n[s, a]].mean() for a in range(2)] for s in range(2)])
        Y = sum(C.w[s] * ((muh[s, 0] - mu[s, 0]) - (muh[s, 1] - mu[s, 1])) for s in range(2))
        hits += Y > wd[3]
    p = hits / T
    assert p <= math.exp(-beta) + 3 * math.sqrt(math.exp(-beta) / T), p


# --------------------------------------------------------------------------------------------- method on the toy
def test_same_schedule_as_fdc_bf_and_never_looser_on_toy():
    env = _toy_population(904)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    c2 = copy.copy(ctx)
    c2.eps = -1.0
    c2.stop_k = 99
    for p in range(2):
        a, ra = run_stream(env, make_variant("FDC-BF"), 904000 + p, PROBS, -1.0, ctx=c2, keep_U=True)
        b, rb = run_stream(env, FDCHG(), 904000 + p, PROBS, -1.0, ctx=c2, keep_U=True)
        assert a["schedule_digest"] == b["schedule_digest"]
        for x, y in zip(ra, rb):
            assert np.all(np.asarray(y["U"]) <= np.asarray(x["U"]) + 1e-10)
        a, _ = run_stream(env, make_variant("FDC-BF"), 904100 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        b, _ = run_stream(env, FDCHG(), 904100 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        assert a["schedule_digest"] == b["schedule_digest"] and b["billing_ok"] and not b["fwer_event"]


# --------------------------------------------------------------------------------------------- external reviewer review r1 regressions
def _mp_logmgf(u, n, N, M, dps=60):
    """Independent high-precision oracle (mpmath, exact integer binomials)."""
    import mpmath as mp
    mp.mp.dps = dps
    t = mp.mpf(float(u)) / n
    c = mp.mpf(n) * M / N
    tot = mp.mpf(0)
    for x in range(max(0, n - (N - M)), min(n, M) + 1):
        tot += mp.binomial(M, x) * mp.binomial(N - M, n - x) * mp.e ** (t * (x - c))
    return mp.log(tot / mp.binomial(N, n))


def _mp_tilted_mean(t, n, N, M, dps=60):
    import mpmath as mp
    mp.mp.dps = dps
    t = mp.mpf(float(t))
    z = s = mp.mpf(0)
    for x in range(max(0, n - (N - M)), min(n, M) + 1):
        g = mp.binomial(M, x) * mp.binomial(N - M, n - x) * mp.e ** (t * x)
        z += g
        s += x * g
    return s / z


def test_reviewer_two_point_support_counterexample():
    N, n = 1000000, 999999
    e = CellEnvelope(n, N, 499999 / N, 0.5, M_lo=499999, M_hi=500000)
    us = np.concatenate([e.ug[:5], -e.ug[:5], e.ug[-3:]])
    val = e(us)
    for u, v in zip(us, val):
        ex = max(_mp_logmgf(u, n, N, 499999), _mp_logmgf(u, n, N, 500000))
        assert v >= float(ex), (u, v, float(ex))


def test_reviewer_tilted_mean_bounds_against_mpmath():
    for (N, M, n, t) in ((1000000, 333333, 1, 0.01), (100000, 33333, 1, 0.01), (5000, 1700, 900, -0.02),
                         (2000, 13, 600, 1.5), (300, 150, 299, -3.0)):
        _, _, h_lo, h_hi = hg_logmgf_bounds(N, M, n, np.array([t]))
        h = _mp_tilted_mean(t, n, N, M)
        assert h_lo[0] <= h <= h_hi[0], (N, M, n, t, h_lo[0], float(h), h_hi[0])


def test_logmgf_upper_bound_against_mpmath_random():
    rng = np.random.default_rng(21)
    for _ in range(40):
        N = int(rng.integers(3, 250))
        n = int(rng.integers(1, N))
        M = int(rng.integers(0, N + 1))
        t = np.array([-2.0, -0.1, -1e-3, 1e-3, 0.1, 2.0])
        lo, hi, _, _ = hg_logmgf_bounds(N, M, n, t)
        for i, ti in enumerate(t):
            ex = _mp_logmgf(ti * n, n, N, M)
            assert lo[i] <= ex <= hi[i], (N, n, M, ti)


def test_bennett_tiny_argument_is_positive():
    assert CellEnvelope(1, 2, 0.5, 0.5, M_lo=1, M_hi=1)(np.array([1e-16]))[0] > 0


def test_extreme_tilts_never_nan():
    for tt in (-40.0, -700.0, 40.0, 710.0):
        v = hg_envelope(100, 70, 0, 100, np.array([tt]))
        assert not np.isnan(v).any() and v[0] >= float(_mp_logmgf(tt * 70, 70, 100, 50)) - 1e-9


def test_integer_box_from_method_contains_truth_large_N():
    """Integer endpoints are used as given; the float-recovery path is outward (contains the truth)."""
    N, M = 100000000000, 66061961580
    e = CellEnvelope(10, N, (M - 100) / N, M / N, M_lo=M - 100, M_hi=M)
    assert e.M_hi == M and e.M_lo == M - 100
    e2 = CellEnvelope(10, N, (M - 100) / N, M / N)
    assert e2.M_lo <= M - 100 and e2.M_hi >= M


def test_fdchg_box_bitwise_equal_to_fdc_bf():
    env = _toy_population(905)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    a, b = make_variant("FDC-BF"), FDCHG()
    a.setup(ctx)
    b.setup(ctx)

    class St:
        pass
    rng = np.random.default_rng(0)
    N = np.asarray(ctx.N, dtype=np.int64)
    n = np.zeros_like(N)
    for _ in range(8):
        n = np.minimum(N, n + rng.integers(0, 40, size=N.shape))
        st = St()
        st.N, st.n = N, n
        st.sum = np.floor(n * rng.uniform(0, 1, size=N.shape))
        la, ha = a._box(ctx, st)
        lb, hb = b._box(ctx, st)
        assert np.array_equal(la, lb) and np.array_equal(ha, hb)


# --------------------------------------------------------------------------------------------- external reviewer review r2 regressions
def test_reviewer_r2_bennett_near_one():
    N = 10_000_000
    n = M = N - 1
    e = CellEnvelope(n, N, M / N, M / N, M_lo=M, M_hi=M)
    u = 999.9999
    import mpmath as mp
    mp.mp.dps = 50
    t = mp.mpf(u) / n
    true = mp.log(1 + mp.expm1(t) / N) - t / N
    assert e(np.array([u]))[0] >= float(true)


def test_reviewer_r2_anchor_accumulation():
    N, M, n = 1000000, 500000, 100000
    t = np.array([-1e-3, -1e-5, 1e-5, 1e-3])
    lo, hi, _, _ = hg_logmgf_bounds(N, M, n, t)
    ex = exact_logmgf(t * n, n, N, M)
    assert np.all(lo <= ex) and np.all(hi >= ex)


def test_reviewer_r2_underflow():
    _, _, _, h_hi = hg_logmgf_bounds(100, 1, 1, np.array([-750.0]))
    assert h_hi[0] > 0
    assert CellEnvelope(1, 2, 0.5, 0.5, M_lo=1, M_hi=1)(np.array([1e-162]))[0] > 0


def test_reviewer_r2_converged_means_tolerance():
    _, info = hg_envelope(1000000, 999999, 500000, 500000, np.array([1e8]), return_info=True)
    assert info["converged"] == (info["slack"] <= 1e-3)
