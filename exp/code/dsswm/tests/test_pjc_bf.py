"""Unit tests for baselines/pjc_bf.py (PJC-BF phased adaptive joint rival, matched Bennett rectangle).  Synthetic only."""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from dsswm.baselines.fdc_bet import direction_widths, make_variant
from dsswm.baselines.pjc_bf import (PJCBF, RectCkBF, bennett_cell_radius, paths_per_checkpoint, pjc_ledger,
                                    scaled_direction_widths)
from dsswm.baselines.frontier_common import FrontierState
from dsswm.streams.frontier_runner import build_ctx, run_stream
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


class _C:
    S, A = 2, 2
    w = np.array([0.4, 0.6])
    pols = np.array(list(product(range(2), repeat=2)))


def _pools(N, mu):
    return [[np.r_[np.ones(int(round(mu[s][a] * N[s][a]))), np.zeros(N[s][a] - int(round(mu[s][a] * N[s][a])))]
             for a in range(2)] for s in range(2)]


def test_scaled_widths_reduce_to_fdc_bet_widths():
    rng = np.random.default_rng(3)
    for _ in range(20):
        N = rng.integers(50, 400, size=(2, 2))
        n = np.minimum(N, rng.integers(1, 300, size=(2, 2)))
        lo = rng.uniform(0, 0.5, size=(2, 2))
        hi = np.minimum(1.0, lo + rng.uniform(0, 0.5, size=(2, 2)))
        for ih in range(4):
            a = direction_widths(_C, ih, n, N, lo, hi, 6.0, "bennett", fpc=True)
            b = scaled_direction_widths(_C, ih, n, N, lo, hi, 6.0, np.ones((2, 2)), "bennett")
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_phase_local_tail_monte_carlo_with_adversarial_phase2_counts():
    """Two phases; phase-2 counts depend on phase-1 data (adversarially).  The phase-local deviation
    Y = sum_c a_c (mu_hat_c^(2) - mu_c) must satisfy P(Y > w(true remaining means)) <= e^-beta."""
    rng = np.random.default_rng(11)
    N = [[120, 200], [90, 160]]
    mu = [[0.2, 0.45], [0.6, 0.1]]
    pools = _pools(N, mu)
    Nn = np.array(N)
    M = np.array([[p.sum() for p in row] for row in pools])
    muv = M / Nn
    beta = math.log(8.0)
    n1 = np.array([[20, 30], [15, 25]])
    T = 20000
    hits = 0
    centre, chall = 0, 3                          # pi_hat = (0, 0), challenger (1, 1)
    for _ in range(T):
        perms = [[rng.permutation(pools[s][a]) for a in range(2)] for s in range(2)]
        s1 = np.array([[perms[s][a][:n1[s, a]].sum() for a in range(2)] for s in range(2)])
        # adversarial phase-2 design: read a lot from cells that looked high in phase 1, little otherwise
        n2 = np.where(s1 / n1 > muv, 60, 8)
        Np = Nn - n1
        s2 = np.array([[perms[s][a][n1[s, a]:n1[s, a] + n2[s, a]].sum() for a in range(2)] for s in range(2)])
        mprime = (M - s1) / Np                    # true remaining means (F_T1-measurable given the truth)
        mhat = (s1 + Np * s2 / n2) / Nn
        Y = sum(_C.w[s] * ((mhat[s, 0] - muv[s, 0]) - (mhat[s, 1] - muv[s, 1])) for s in range(2))
        wd = scaled_direction_widths(_C, centre, n2, Np, mprime, mprime, beta, Np / Nn, "bennett")
        hits += Y > wd[chall]
    p = hits / T
    assert p <= math.exp(-beta) + 3 * math.sqrt(math.exp(-beta) / T), p


def test_ledgers_spend_exactly_delta():
    env = _toy_population(900)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    K = len(ctx.checkpoints)
    loc = pjc_ledger(ctx, "local", (2,))
    ref = make_variant("FDC-BF")
    ref.setup(ctx)
    assert abs(loc["beta"] - ref.ledger["beta"]) < 1e-12
    assert abs(loc["alpha_side_var"] - ref.ledger["alpha_side_var"]) < 1e-15
    for b, M in (((0,), 3), ((2, 6), 3), ((2,), 2), ((), 4)):
        led = pjc_ledger(ctx, "menu", b, M)
        L = paths_per_checkpoint(K, b, M)
        assert led["L_events_per_direction"] == int(L.sum())
        assert abs(led["bound_main"] - 0.045) < 1e-12 and abs(led["bound_var"] - 0.005) < 1e-12
    assert paths_per_checkpoint(5, (1, 3), 3).tolist() == [1, 1, 3, 3, 9]


def test_plan_uses_counts_only():
    env = _toy_population(901)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    m = PJCBF(mode="local", boundaries=(0,), rule="neyman")
    m.setup(ctx)
    m._p = np.array([[0.3, 0.7]] * ctx.S)
    n = np.array([[10, 40], [25, 20], [5, 5]])
    a = m.plan(ctx, FrontierState(100, n, np.zeros((3, 2)), ctx.N, np.ones(ctx.Q, bool), None))
    b = m.plan(ctx, FrontierState(100, n, np.random.default_rng(0).integers(0, 5, (3, 2)), ctx.N,
                                  np.ones(ctx.Q, bool), None))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kw", [dict(mode="local", boundaries=(0,), rule="neyman"),
                                dict(mode="local", boundaries=(2, 5, 8), rule="neyman", rect=True),
                                dict(mode="menu", boundaries=(0, 4), rule="proj", menu=(0.35, 0.5, 0.65))])
def test_pjc_runs_sound_and_adapts_on_toy(kw):
    adapted = 0
    for sd in (900, 903, 907):
        env = _toy_population(sd)
        ctx = build_ctx(env, PROBS, TOY_EPS)
        for p in range(2):
            m = PJCBF(**kw)
            s, _ = run_stream(env, m, 1000 * sd + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
            assert s["billing_ok"] and not s["fwer_event"] and s["n_cert"] > 0
            adapted += any(abs(float(x[:, 0].mean()) - 0.5) > 1e-9 for x in m.plans[1:])
    assert adapted > 0


def test_bennett_cell_radius_monte_carlo():
    rng = np.random.default_rng(5)
    N, Mk, n = 300, 75, 60
    pool = np.r_[np.ones(Mk), np.zeros(N - Mk)]
    m = Mk / N
    beta = math.log(10.0)
    r = float(bennett_cell_radius(np.array([n]), np.array([N]), np.array([m]), np.array([m]), beta)[0])
    T = 40000
    dev = np.array([rng.permutation(pool)[:n].mean() - m for _ in range(T)])
    for side in (dev, -dev):
        p = float((side > r).mean())
        assert p <= math.exp(-beta) + 3 * math.sqrt(math.exp(-beta) / T), p


def test_rect_ck_bf_ledger_and_toy_soundness():
    env = _toy_population(902)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    m = RectCkBF()
    m.setup(ctx)
    assert abs(m.ledger["bound_main"] - 0.045) < 1e-12 and abs(m.ledger["bound_var"] - 0.005) < 1e-12
    fb = make_variant("FDC-BF")
    for p in range(3):
        a, _ = run_stream(env, RectCkBF(box=True), 902000 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        b, _ = run_stream(env, fb, 902000 + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        assert a["schedule_digest"] == b["schedule_digest"] and a["billing_ok"] and not a["fwer_event"]
