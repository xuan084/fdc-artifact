"""Unit tests for baselines/multiarm_v9.py (general-A FDC-BF / PJC-BF / matched rectangle) and envs/hillstrom_v9.py."""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from dsswm.baselines import multiarm_v9 as ma
from dsswm.baselines.fdc_bet import direction_widths, make_variant, psi_bar
from dsswm.baselines.pjc_bf import PJCBF, RectCkBF
from dsswm.streams import frontier as fr
from dsswm.streams.frontier_runner import SyntheticPoolEnv, run_stream
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


class _C3:
    S, A = 2, 3
    w = np.array([0.35, 0.65])
    pols = np.array(list(product(range(3), repeat=2)))


def _same(s1, s2):
    for k in ("N80", "cert_k", "decided_pi", "schedule_digest", "n_false", "n_cert"):
        assert s1[k] == s2[k], k


# ------------------------------------------------------------------------------------------- A = 2 reductions
@pytest.mark.parametrize("sd", [900, 905])
def test_fdc_bf_a_equals_fdc_bf_for_two_arms(sd):
    env = _toy_population(sd)
    s1, _ = run_stream(env, make_variant("FDC-BF"), 1000 * sd + 1, PROBS, TOY_EPS, keep_U=False)
    s2, _ = run_stream(env, ma.fdc_bf_a(), 1000 * sd + 1, PROBS, TOY_EPS, keep_U=False)
    _same(s1, s2)


def test_pjc_a_bal_equals_pjc_half_for_two_arms():
    env = _toy_population(902)
    for b in ((), (2,)):
        s1, _ = run_stream(env, PJCBF(mode="local", boundaries=b, rule="half"), 902_003, PROBS, TOY_EPS, keep_U=False)
        s2, _ = run_stream(env, ma.PJCBFA(boundaries=b, rule="bal"), 902_003, PROBS, TOY_EPS, keep_U=False)
        _same(s1, s2)
    s1, _ = run_stream(env, PJCBF(mode="local", boundaries=(2,), rule="neyman"), 902_004, PROBS, TOY_EPS, keep_U=False)
    s2, _ = run_stream(env, ma.PJCBFA(boundaries=(2,), rule="neyman"), 902_004, PROBS, TOY_EPS, keep_U=False)
    _same(s1, s2)


def test_rect_bf_a_equals_rect_bf_for_two_arms():
    env = _toy_population(903)
    s1, _ = run_stream(env, RectCkBF(), 903_002, PROBS, TOY_EPS, keep_U=False)
    s2, _ = run_stream(env, ma.RectCkBFA(), 903_002, PROBS, TOY_EPS, keep_U=False)
    _same(s1, s2)


# ------------------------------------------------------------------------------------------- A = 3 coefficients
def test_direction_widths_a3_match_explicit_coefficient_vector():
    """For A = 3 the width must equal min_lambda (beta + sum_c Psi_bar(lambda a_c)) / lambda with the explicit
    coefficient vector a_c = -w_s on (s, pi'(s)), +w_s on (s, pi_hat(s)) for differing s, 0 elsewhere (in particular 0
    on the third arm of a differing segment and on every arm of an agreeing segment)."""
    rng = np.random.default_rng(5)
    grid = np.exp(np.linspace(math.log(1 / 200), math.log(10), 161))
    for _ in range(15):
        N = rng.integers(60, 400, size=(2, 3))
        n = np.minimum(N - 1, rng.integers(5, 300, size=(2, 3)))
        lo = rng.uniform(0, 0.5, size=(2, 3))
        hi = np.minimum(1.0, lo + rng.uniform(0.01, 0.5, size=(2, 3)))
        beta = 6.0
        for ih in range(9):
            got = direction_widths(_C3, ih, n, N, lo, hi, beta, "bennett", fpc=True, grid=grid)
            ph = _C3.pols[ih]
            for p in range(9):
                pc = _C3.pols[p]
                a = np.zeros((2, 3))
                for s in range(2):
                    if pc[s] != ph[s]:
                        a[s, pc[s]] -= _C3.w[s]
                        a[s, ph[s]] += _C3.w[s]
                if not a.any():
                    assert got[p] == 0.0
                    continue
                assert np.count_nonzero(a) == 2 * int((pc != ph).sum())
                v = sum(a[s, b] ** 2 * (np.clip(0.5, lo[s, b], hi[s, b]) * (1 - np.clip(0.5, lo[s, b], hi[s, b]))
                                        * (N[s, b] - n[s, b]) / (N[s, b] - 1) / n[s, b])
                        for s in range(2) for b in range(3) if a[s, b] != 0)
                lam = math.sqrt(2 * beta / v) * grid
                F = np.zeros_like(lam)
                for s, b in zip(*np.nonzero(a)):
                    F += psi_bar(lam * a[s, b], n[s, b], N[s, b], lo[s, b], hi[s, b], "bennett")
                ref = float(np.min((beta + F) / lam))
                assert got[p] == pytest.approx(ref, rel=1e-10, abs=1e-14)


def test_direction_tail_monte_carlo_a3():
    """Fixed-size WoR samples from 2 x 3 binary pools; for every challenger direction the deviation
    Y = sum_c a_c (mu_hat_c - mu_c) exceeds the width at the TRUE means (point box) with frequency <= e^-beta."""
    rng = np.random.default_rng(17)
    N = np.array([[150, 220, 180], [260, 140, 200]])
    mu = np.array([[0.2, 0.45, 0.3], [0.6, 0.15, 0.5]])
    M = np.rint(mu * N).astype(int)
    muv = M / N
    n = np.array([[30, 50, 40], [45, 25, 60]])
    beta = math.log(20.0)                                   # bound 0.05 per direction
    ih = 4                                                  # pi_hat = (1, 1)
    w = direction_widths(_C3, ih, n, N, muv, muv, beta, "bennett", fpc=True)
    ph = _C3.pols[ih]
    A_dir = []
    for p in range(9):
        pc = _C3.pols[p]
        a = np.zeros((2, 3))
        for s in range(2):
            if pc[s] != ph[s]:
                a[s, pc[s]] -= _C3.w[s]
                a[s, ph[s]] += _C3.w[s]
        A_dir.append(a)
    T = 20000
    hits = np.zeros(9)
    for _ in range(T):
        x = rng.hypergeometric(M, N - M, n)
        dev = x / n - muv
        hits += np.array([float((A_dir[p] * dev).sum() > w[p]) if A_dir[p].any() else 0.0 for p in range(9)])
    freq = hits / T
    assert freq.max() <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / T)


# ------------------------------------------------------------------------------------------- A = 3 toy FWER (short)
def _toy3(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2000, 8000, size=(3, 3))
    base = rng.uniform(0.15, 0.6, size=3)
    up = rng.normal(0.0, 0.10, size=(3, 2))
    mu = np.column_stack([base, np.clip(base[:, None] + up, 0.01, 0.99)])
    return SyntheticPoolEnv(sizes, mu, seed=seed, n_min=200)


TOY3_PROBS = [fr.Problem(f"toy3|B{b:.2f}", "visit", (0.0, 1.0, 1.0), b) for b in (0.2, 0.35, 0.5, 0.65, 0.8)]


@pytest.mark.parametrize("mk", [lambda: ma.fdc_bf_a(), lambda: ma.PJCBFA(boundaries=(2,), rule="neyman"),
                                lambda: ma.RectCkBFA(), lambda: ma.make_plan_rect("bal", 3, 3)])
def test_toy3_no_false_certification_short(mk):
    ev = cert = 0
    for sd in (900, 901, 902):
        env = _toy3(sd)
        for p in range(3):
            s, _ = run_stream(env, mk(), 1000 * sd + p, TOY3_PROBS, 0.02, keep_U=False)
            assert s["billing_ok"]
            ev += int(s["fwer_event"])
            cert += s["n_cert"]
    assert ev == 0 and cert > 0


def test_fdc_bf_a_three_arm_design_and_ledger():
    env = _toy3(904)
    m = ma.fdc_bf_a()
    from dsswm.streams.frontier_runner import build_ctx
    ctx = build_ctx(env, TOY3_PROBS, 0.02)
    m.setup(ctx)
    np.testing.assert_allclose(m.alloc_p, np.full((3, 3), 1 / 3))
    led = m.ledger
    assert led["union_size"] == int(ctx.feas.sum())
    assert led["S_A"] == 9
    assert abs(led["bound_main"] - 0.045) < 1e-12 and abs(led["bound_var"] - 0.005) < 1e-12
    with pytest.raises(ValueError):
        ma.fdc_bf_a(alloc=np.array([[0.5, 0.5, 0.0]] * 3)).setup(ctx)


def test_plan_matrices_rows_sum_to_one():
    s2 = np.array([[0.1, 0.2, 0.15], [0.05, 0.25, 0.2]])
    for tag in ma.PLAN_TAGS_A3:
        P = ma.plan_matrix(tag, 2, 3, s2)
        assert P.shape == (2, 3) and np.allclose(P.sum(1), 1) and (P > 0).all()
    np.testing.assert_allclose(ma.plan_matrix("0.5", 2, 3)[0], [0.5, 0.25, 0.25])


# ------------------------------------------------------------------------------------------- Hillstrom env
def test_hillstrom_eval_half_is_refused_without_v9_gate():
    from dsswm.envs.hillstrom_v9 import HillstromV9Env
    with pytest.raises(PermissionError):
        HillstromV9Env("CR6", "eval")
    with pytest.raises(PermissionError):
        HillstromV9Env("CR6", "eval", eval_task_id="v9c_full_a")


def test_hillstrom_hashed_split_label_only_and_stable():
    from dsswm.envs import hillstrom_v9 as hv
    m1 = hv.hashed_dev_mask(1000)
    m2 = hv.hashed_dev_mask(1000)
    assert np.array_equal(m1, m2) and 400 < m1.sum() < 600
    assert not np.array_equal(m1, hv.hashed_dev_mask(1000, salt="other"))
    summ = hv.split_summary()
    assert summ["dev_rows"] + summ["eval_rows"] == 64000
    assert summ["dev_rows"] == 32119


def test_hillstrom_dev_env_segments_and_problems():
    from dsswm.envs import hillstrom_v9 as hv
    env = hv.HillstromV9Env("CR6", "dev", outcomes=("visit",))
    assert env.S == 6 and env.A == 3 and env.N == 32119 and env.pool_sizes.min() > 300
    assert env.half == "dev"
    probs = hv.hv9_problems("visit", "F3")
    assert len(probs) == 15 and probs[0].kappa == (0.0, 1.5, 1.0)
