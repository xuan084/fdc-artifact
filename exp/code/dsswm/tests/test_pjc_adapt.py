"""Unit tests for baselines/pjc_adapt.py (post-hoc block v8-posthoc-D: genuinely adaptive PJC members).  Synthetic only."""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from dsswm.baselines.fdc_bet import make_variant
from dsswm.baselines.pjc_adapt import MENU3, MENU5, PJCAdapt, adapt_ledger, paths_per_checkpoint_exact
from dsswm.baselines.pjc_bf import paths_per_checkpoint, pjc_ledger, scaled_direction_widths
from dsswm.streams.frontier_runner import build_ctx, run_stream
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


class _C:
    S, A = 2, 2
    w = np.array([0.4, 0.6])
    pols = np.array(list(product(range(2), repeat=2)))


def test_ledgers_spend_exactly_delta_and_match_pjc_bf():
    env = _toy_population(900)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    K = len(ctx.checkpoints)
    ref = make_variant("FDC-BF")
    ref.setup(ctx)
    r = adapt_ledger(ctx, "reset", (2, 5))
    assert abs(r["beta"] - ref.ledger["beta"]) < 1e-12
    assert abs(r["alpha_side_var"] - ref.ledger["alpha_side_var"]) < 1e-18
    for b in ((2,), (2, 6), (1, 4, 8)):
        m = adapt_ledger(ctx, "menu", b, 5)
        p = pjc_ledger(ctx, "menu", b, 5)
        assert abs(m["beta"] - p["beta"]) < 1e-12 and abs(m["alpha_side_var"] - p["alpha_side_var"]) < 1e-18
        sg = adapt_ledger(ctx, "segmenu", b, 3)
        L = sum(paths_per_checkpoint_exact(K, b, 3 ** ctx.S))
        assert sg["L_events_per_direction"] == str(L)
        for led in (m, sg, r):
            assert abs(led["bound_main"] - 0.045) < 1e-9 and abs(led["bound_var"] - 0.005) < 1e-12
    # exact integer path counts agree with pjc_bf's int64 counter where the latter does not overflow
    assert paths_per_checkpoint_exact(7, (1, 3), 4) == paths_per_checkpoint(7, (1, 3), 4).tolist()
    # huge segment menus (5^9 per boundary, 3 boundaries) stay exact and finite
    big = sum(paths_per_checkpoint_exact(20, (2, 5, 8), 5 ** 9))
    assert big > 2 ** 63 and math.isfinite(math.log(big))


def test_union_ledger_monte_carlo_per_segment_menu_adversarial_path():
    """Keep-data estimator, per-segment menu with 2 choices x 2 segments = 4 paths per boundary.  Phase-2 counts are
    chosen ADVERSARIALLY from phase-1 data.  Each fixed path obeys the Chernoff tail e^-beta; the realised path is
    covered by the 4-path union: P(Y > w_path(beta0 + ln 4)) <= e^-beta0."""
    rng = np.random.default_rng(17)
    N = np.array([[150, 220], [120, 180]])
    mu = np.array([[0.25, 0.5], [0.55, 0.15]])
    pools = [[np.r_[np.ones(int(round(mu[s, a] * N[s, a]))), np.zeros(N[s, a] - int(round(mu[s, a] * N[s, a])))]
              for a in range(2)] for s in range(2)]
    M = np.array([[p.sum() for p in row] for row in pools])
    muv = M / N
    n1 = np.array([[20, 20], [20, 20]])
    extra = 60
    menu = (0.25, 0.75)
    beta0 = math.log(10.0)
    beta = beta0 + math.log(4.0)
    centre, chall = 0, 3
    T = 20000
    hits_union, hits_fixed = 0, 0
    for _ in range(T):
        perms = [[rng.permutation(pools[s][a]) for a in range(2)] for s in range(2)]
        s1 = np.array([[perms[s][a][:n1[s, a]].sum() for a in range(2)] for s in range(2)])
        # adversary: per segment, read more from the arm whose phase-1 mean overshot the truth
        choice = [0 if s1[s, 0] / n1[s, 0] <= muv[s, 0] else 1 for s in range(2)]
        n = n1.copy()
        for s in range(2):
            n[s, 0] += int(round(menu[choice[s]] * extra))
            n[s, 1] += extra - int(round(menu[choice[s]] * extra))
        sums = np.array([[perms[s][a][:n[s, a]].sum() for a in range(2)] for s in range(2)])
        mhat = sums / n
        Y = sum(_C.w[s] * ((mhat[s, 0] - muv[s, 0]) - (mhat[s, 1] - muv[s, 1])) for s in range(2))
        wd = scaled_direction_widths(_C, centre, n, N, muv, muv, beta, np.ones((2, 2)), "bennett")
        hits_union += Y > wd[chall]
        # fixed path (choice (0, 0) always): the single-path bound at beta0
        nf = n1 + np.array([[15, 45], [15, 45]])
        sf = np.array([[perms[s][a][:nf[s, a]].sum() for a in range(2)] for s in range(2)])
        Yf = sum(_C.w[s] * ((sf[s, 0] / nf[s, 0] - muv[s, 0]) - (sf[s, 1] / nf[s, 1] - muv[s, 1])) for s in range(2))
        wf = scaled_direction_widths(_C, centre, nf, N, muv, muv, beta0, np.ones((2, 2)), "bennett")
        hits_fixed += Yf > wf[chall]
    slack = 3 * math.sqrt(math.exp(-beta0) / T)
    assert hits_union / T <= math.exp(-beta0) + slack, hits_union / T
    assert hits_fixed / T <= math.exp(-beta0) + slack, hits_fixed / T


@pytest.mark.parametrize("kw", [dict(family="reset", boundaries=(0,), rule="neyman"),
                                dict(family="reset", boundaries=(2, 5), rule="dirseg"),
                                dict(family="menu", boundaries=(0, 4), rule="dirproj", menu=MENU5),
                                dict(family="menu", boundaries=(2,), rule="proj", menu=MENU5),
                                dict(family="segmenu", boundaries=(2,), rule="dirseg", menu=MENU3),
                                dict(family="segmenu", boundaries=(0, 3), rule="neyman", menu=MENU5)])
def test_adapt_runs_sound_on_toy(kw):
    switched = 0
    for sd in (900, 903, 907):
        env = _toy_population(sd)
        ctx = build_ctx(env, PROBS, TOY_EPS)
        for p in range(2):
            m = PJCAdapt(**kw)
            s, _ = run_stream(env, m, 1000 * sd + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
            assert s["billing_ok"] and not s["fwer_event"] and s["n_cert"] > 0
            assert len(m.plans) == 1 + sum(1 for b in m.boundaries if b < len(ctx.checkpoints))
            switched += m.switched
            if m.family == "segmenu":
                assert all(isinstance(x, tuple) and len(x) == ctx.S for x in m.path)
                for P in m.plans[1:]:
                    assert set(np.round(P[:, 0], 9)) <= set(m.menu)
    if kw["rule"] == "neyman" and kw["family"] == "reset":
        assert switched > 0


def test_no_adapt_member_reduces_to_fdc_bf_on_toy():
    """menu family with menu (0.5,) only and no boundary: identical certificate path to PJC no-reset member."""
    from dsswm.baselines.pjc_bf import PJCBF
    env = _toy_population(904)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    a, _ = run_stream(env, PJCAdapt("menu", (), "dirproj", menu=(0.5,)), 904001, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
    b, _ = run_stream(env, PJCBF("local", (), "half"), 904001, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
    assert a["cert_k"] == b["cert_k"] and a["N80"] == b["N80"]


def test_design_is_data_free_of_future_and_counts_only_tracking():
    """plan() reads counts only (inherited); the boundary plan only depends on data up to the boundary."""
    from dsswm.baselines.frontier_common import FrontierState
    env = _toy_population(905)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    m = PJCAdapt("reset", (0,), "dirseg")
    m.setup(ctx)
    m._p = np.array([[0.3, 0.7]] * ctx.S)
    n = np.array([[10, 40], [25, 20], [5, 5]])
    a = m.plan(ctx, FrontierState(100, n, np.zeros((3, 2)), ctx.N, np.ones(ctx.Q, bool), None))
    b = m.plan(ctx, FrontierState(100, n, np.random.default_rng(0).integers(0, 5, (3, 2)), ctx.N,
                                  np.ones(ctx.Q, bool), None))
    assert np.array_equal(a, b)
