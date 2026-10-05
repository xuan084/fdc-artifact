"""Tests for the v9 candidates (dsswm/baselines/fdc_loc.py; plan/v9_candidates_theory.md)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy import stats

from dsswm.baselines.fdc_bet import make_variant
from dsswm.baselines.fdc_loc import FDCLoc, FDCTimeUniform, dense_grid, loc_beta
from dsswm.streams.frontier_runner import build_ctx
from dsswm.streams.frontier_runner_v6 import run_stream_v6
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


@pytest.fixture(scope="module")
def toy():
    env = _toy_population(900)
    return env, build_ctx(env, PROBS, TOY_EPS)


def _run(env, m, seed, ctx):
    s, rows = run_stream_v6(env, m, seed, PROBS, TOY_EPS, ctx=ctx, keep_U=True)
    return s, rows


# ------------------------------------------------------------------------------------------------ (a) FDC-LOC
def test_loc_without_localisation_equals_fdc_bf(toy):
    env, ctx = toy
    for seed in (900000, 900001):
        s1, r1 = _run(env, make_variant("FDC-BF"), seed, ctx)
        s2, r2 = _run(env, FDCLoc(localise=False), seed, ctx)
        assert s1["N80"] == s2["N80"] and s1["cert_k"] == s2["cert_k"]
        for a, b in zip(r1, r2):
            assert np.allclose(a["U"], b["U"], rtol=0, atol=0)


def test_loc_dominates_fdc_bf(toy):
    env, ctx = toy
    for seed in (900002, 900003, 900004):
        s1, r1 = _run(env, make_variant("FDC-BF"), seed, ctx)
        m = FDCLoc()
        s2, r2 = _run(env, m, seed, ctx)
        assert all(b["beta"] <= b["beta_J"] + 1e-12 for b in m.beta_trace)
        for a, b in zip(r1, r2):                      # common prefix: U_loc <= U_bf (same centre, smaller beta)
            assert np.all(np.asarray(b["U"]) <= np.asarray(a["U"]) + 1e-12)
        assert s2["N80"] <= s1["N80"]


def test_loc_beta_admissible_and_bounded(toy):
    env, ctx = toy
    rng = np.random.default_rng(0)
    N = ctx.N.astype(float)
    for _ in range(5):
        n = np.floor(N * rng.uniform(0.05, 0.9, size=N.shape))
        mu = rng.uniform(0.2, 0.8, size=N.shape)
        lo = np.clip(mu - rng.uniform(0.0, 0.2, size=N.shape), 0, 1)
        hi = np.clip(mu + rng.uniform(0.0, 0.2, size=N.shape), 0, 1)
        beta_J = math.log(int(ctx.feas.sum()) * 20 / 0.045)
        b, info = loc_beta(ctx, n, N, lo, hi, beta_J, 0.045 / 20)
        assert 0.0 <= b <= beta_J
        assert info.get("G_at_beta", 0.0) <= 0.045 / 20 * (1 + 1e-9)


# ------------------------------------------------------------------------------------------------ (b) TU-FDC
def test_dense_grid_contains_checkpoints(toy):
    _, ctx = toy
    g = dense_grid(ctx.checkpoints, 4)
    assert set(ctx.checkpoints.tolist()) <= set(g.tolist())
    assert len(g) > 3 * len(ctx.checkpoints) - 5 and np.all(np.diff(g) > 0)


def test_tu_on_block_grid_equals_fdc_bf(toy):
    env, ctx = toy
    for seed in (900005, 900006):
        s1, r1 = _run(env, make_variant("FDC-BF"), seed, ctx)
        s2, r2 = _run(env, FDCTimeUniform(ctx.checkpoints), seed, ctx)
        assert s1["N80"] == s2["N80"] and s1["cert_k"] == s2["cert_k"]
        for a, b in zip(r1, r2):
            assert np.array_equal(a["U"], b["U"])


def test_tu_dense_uses_stale_widths_and_block_ledger(toy):
    env, ctx = toy
    c2 = copy.copy(ctx)
    c2.checkpoints = dense_grid(ctx.checkpoints, 4)
    m = FDCTimeUniform(ctx.checkpoints)
    s, rows = _run(env, m, 900007, c2)
    assert m.ledger["K"] == len(ctx.checkpoints) and m.ledger["K_eval"] == len(c2.checkpoints)
    assert math.isclose(m.ledger["beta"], math.log(int(ctx.feas.sum()) * len(ctx.checkpoints) / 0.045))
    assert m.n_stale_evals > 0
    with pytest.raises(RuntimeError):                 # block points must be evaluation times
        bad = FDCTimeUniform(ctx.checkpoints + 1)
        _run(env, bad, 900008, c2)


def _exact_hg_mgf_dev(N, Mpos, n, lam):
    """log E exp(lam (mu - mu_hat_n)) for X ~ HG(N, Mpos, n) (exact, small N)."""
    x = np.arange(0, n + 1)
    p = stats.hypergeom.pmf(x, N, Mpos, n)
    return float(np.log(np.sum(p * np.exp(lam * (Mpos / N - x / n)))))


def test_lemma_tu_single_cell_monte_carlo_with_invalid_control():
    """Lemma TU: P(sup_{n >= n0} (mu - mu_hat_n) >= y) <= inf_lam e^{-lam y} E e^{lam Y_{n0}} (exact HG MGF).

    Invalid control: the exact FIXED-n quantile at n0 (P(Y_{n0} >= y_fix) <= a) monitored over all n >= n0 with a fresh
    exact fixed-n quantile at every n (no union, no maximal inequality) -- its crossing frequency must exceed a."""
    N, Mpos, n0, a = 300, 90, 30, 0.10
    lams = np.linspace(0.5, 60, 400)
    logm = np.array([_exact_hg_mgf_dev(N, Mpos, n0, l) for l in lams])
    # y with inf_lam exp(-lam y + logM(lam)) = a
    ylo, yhi = 0.0, 1.0
    for _ in range(60):
        y = 0.5 * (ylo + yhi)
        if np.min(np.exp(-lams * y + logm)) <= a:
            yhi = y
        else:
            ylo = y
    y = yhi
    # fresh exact quantile per n: smallest c_n with P(mu - mu_hat_n >= c_n) <= a
    mu = Mpos / N
    cn = {}
    for n in range(n0, N):
        x = np.arange(0, n + 1)
        cdf = stats.hypergeom.cdf(x, N, Mpos, n)        # P(X <= x)  <-> mu_hat <= x/n  <-> dev >= mu - x/n
        ok = x[cdf <= a]
        cn[n] = (mu - (ok.max() / n)) if ok.size else np.inf   # dev >= c_n  iff X <= ok.max()
    rng = np.random.default_rng(12345)
    pool = np.zeros(N)
    pool[:Mpos] = 1.0
    R = 6000
    hit_tu = hit_naive = 0
    cvec = np.array([cn[n] for n in range(n0, N)])
    for _ in range(R):
        xs = rng.permutation(pool)
        dev = mu - np.cumsum(xs)[n0 - 1:N - 1] / np.arange(n0, N)
        hit_tu += bool(np.any(dev >= y - 1e-12))
        hit_naive += bool(np.any(dev >= cvec - 1e-12))
    f_tu, f_nv = hit_tu / R, hit_naive / R
    se = math.sqrt(a * (1 - a) / R)
    assert f_tu <= a + 3 * se, (f_tu, a)
    assert f_nv > a + 3 * se, (f_nv, a)


def test_loc_beta_zero_branch_keeps_dominance():
    """external reviewer r1 counterexample: all policies surely eps-good -> beta_hat must stay > 0 and widths finite."""
    from types import SimpleNamespace
    from dsswm.baselines.fdc_bet import direction_widths
    ctx = SimpleNamespace(S=1, A=2, w=np.array([1.0]), pols=np.array([[0], [1]]), feas=np.array([[True, True]]),
                          Q=1, eps=0.1)
    n = np.array([[900.0, 900.0]])
    N = np.array([[1000.0, 1000.0]])
    lo = np.array([[0.484, 0.484]])
    hi = np.array([[0.516, 0.516]])
    beta_J = math.log(2 * 1 / 0.045)
    b, _ = loc_beta(ctx, n, N, lo, hi, beta_J, 0.045)
    assert 0 < b <= beta_J
    wl = direction_widths(ctx, 0, n, N, lo, hi, b, "bennett")
    wb = direction_widths(ctx, 0, n, N, lo, hi, beta_J, "bennett")
    assert np.all(np.isfinite(wl)) and np.all(wl <= wb + 1e-12)
