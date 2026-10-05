"""Round-4 statistics: one-sided CP, WSR20 WoR empirical-Bernstein CS, variance UCB, L1 width, acceptance stats."""
import math

import numpy as np
import pytest
from scipy.stats import beta

from dsswm.stats.acceptance_r4 import (reviewer_fcr_counterexample, completion_lower, e3_uncensored, e_v_over_r,
                                       fcr_old, holm, jpc_failure_reproduced, layer_safety, paired_diff_positive,
                                       paired_log_ratio, penalized_cost_log_ratio, s1_fwer, s2_fcr_old)
from dsswm.stats.cp import clopper_pearson, cp_lower_one_sided, cp_upper_one_sided
from dsswm.stats.fp_eb import delta_per_cell, functional_width, var_ucb_binary, wor_mean_cs


# ------------------------------------------------------------------------------------------------------ CP
def test_cp_upper_one_sided_value():
    assert abs(cp_upper_one_sided(4, 200) - 0.04518) < 1e-5
    assert cp_upper_one_sided(4, 200) <= 0.05 < cp_upper_one_sided(5, 200)


def test_cp_edges_and_lower():
    assert cp_upper_one_sided(200, 200) == 1.0
    assert cp_lower_one_sided(0, 200) == 0.0
    assert abs(cp_upper_one_sided(0, 200) - (1 - 0.05 ** (1 / 200))) < 1e-12
    assert abs(cp_lower_one_sided(200, 200) - 0.05 ** (1 / 200)) < 1e-12
    assert cp_lower_one_sided(4, 200) < 4 / 200 < cp_upper_one_sided(4, 200)
    with pytest.raises(ValueError):
        cp_upper_one_sided(5, 4)


def test_old_clopper_pearson_unchanged():
    lo, hi = clopper_pearson(4, 200)
    assert lo == float(beta.ppf(0.025, 4, 197)) and hi == float(beta.ppf(0.975, 5, 196))
    assert clopper_pearson(0, 0) == (0.0, 1.0)


# ------------------------------------------------------------------------------------------------------ WoR CS
def _perms(pool, reps, rng):
    return np.array([rng.permutation(pool) for _ in range(reps)])


@pytest.mark.parametrize("N,mu,delta", [(400, 0.3, 0.05), (600, 0.04, 0.05), (300, 0.5, 0.2)])
def test_wor_cs_time_uniform_coverage_binary(N, mu, delta):
    rng = np.random.default_rng(42)
    pool = np.zeros(N)
    pool[: int(round(mu * N))] = 1.0
    m = pool.mean()
    X = _perms(pool, 10_000, rng)
    lo, hi = wor_mean_cs(X, N, delta)
    miss = ((lo > m + 1e-12) | (hi < m - 1e-12)).any(axis=1).mean()
    assert 1 - miss >= 1 - delta


def test_wor_cs_coverage_nonbinary_and_range():
    rng = np.random.default_rng(42)
    N, R, delta = 400, 5.0, 0.1
    pool = R * rng.beta(0.5, 2.0, N)
    X = _perms(pool, 10_000, rng)
    lo, hi = wor_mean_cs(X, N, delta, R=R)
    m = pool.mean()
    assert ((lo > m + 1e-12) | (hi < m - 1e-12)).any(axis=1).mean() <= delta
    with pytest.raises(ValueError):
        wor_mean_cs(np.array([R + 1.0]), N, delta, R=R)


def test_wor_cs_structure():
    rng = np.random.default_rng(42)
    N = 200
    pool = (rng.random(N) < 0.35).astype(float)
    x = rng.permutation(pool)
    lo, hi = wor_mean_cs(x, N, 0.05)
    assert np.all(np.diff(lo) >= -1e-15) and np.all(np.diff(hi) <= 1e-15)   # running intersection
    assert np.all(lo <= hi)
    assert abs(lo[-1] - pool.mean()) < 1e-12 and abs(hi[-1] - pool.mean()) < 1e-12  # exhaustion is exact
    lo_s, hi_s = wor_mean_cs(x[:50], N, 0.05)
    np.testing.assert_allclose(lo_s, lo[:50])                                # prefix-consistent (predictable)
    with pytest.raises(ValueError):
        wor_mean_cs(np.zeros(N + 1), N, 0.05)


def test_wor_cs_tighter_than_wr_late():
    """Near exhaustion the WoR CS must be narrower than the same CS run with a huge pool (with-replacement limit)."""
    rng = np.random.default_rng(42)
    N = 1000
    pool = (rng.random(N) < 0.3).astype(float)
    x = rng.permutation(pool)[:900]
    lo, hi = wor_mean_cs(x, N, 0.05)
    lo2, hi2 = wor_mean_cs(x, 10 ** 9, 0.05)
    assert (hi[-1] - lo[-1]) < 0.7 * (hi2[-1] - lo2[-1])


def test_delta_per_cell():
    assert delta_per_cell(0.01, 48) == pytest.approx(0.01 / 48)


# ------------------------------------------------------------------------------------------------------ var UCB
def test_var_ucb_binary_cases():
    assert var_ucb_binary((0.4, 0.7)) == 0.25
    assert var_ucb_binary((0.1, 0.2)) == pytest.approx(0.16)
    assert var_ucb_binary((0.8, 0.95)) == pytest.approx(0.16)
    assert var_ucb_binary((0.3, 0.3)) == pytest.approx(0.21)
    np.testing.assert_allclose(var_ucb_binary((np.array([0.1, 0.6]), np.array([0.2, 0.9]))), [0.16, 0.24])


def test_var_ucb_dominates_true_variance_on_E_var():
    rng = np.random.default_rng(42)
    for N, mu in [(300, 0.1), (300, 0.45), (500, 0.8)]:
        pool = np.zeros(N)
        pool[: int(round(mu * N))] = 1.0
        m = pool.mean()
        X = _perms(pool, 2000, rng)
        lo, hi = wor_mean_cs(X, N, 0.05)
        cov = ~((lo > m + 1e-12) | (hi < m - 1e-12)).any(axis=1)
        v = var_ucb_binary((lo, hi))
        assert cov.mean() >= 0.95
        assert np.all(v[cov] >= m * (1 - m) - 1e-12)


# ------------------------------------------------------------------------------------------------------ L1 width
def test_functional_width_formula():
    a = np.array([0.5, -0.5, 0.0])
    n = np.array([100, 50, 0])
    v = np.array([0.2, 0.25, 0.25])
    x = 19.4
    V = 0.25 * 0.2 / 100 + 0.25 * 0.25 / 50
    b = 0.5 / 50
    assert functional_width(a, n, v, x) == pytest.approx(math.sqrt(2 * V * x) + b * x / 3)
    assert functional_width(a, [0, 50, 0], v, x) == float("inf")
    # exhausted cell 2 (n == N) is exact and dropped
    w_live = functional_width(a, n, v, x, N=[1000, 50, 10])
    assert w_live == pytest.approx(math.sqrt(2 * 0.25 * 0.2 / 100 * x) + 0.5 / 100 * x / 3)
    assert w_live <= functional_width(a, n, v, x)
    assert functional_width(a, [10, 50, 0], v, x, N=[10, 50, 5]) == 0.0
    assert functional_width(a, n, v, x, R=2.0) > functional_width(a, n, v, x)


def test_l1_width_with_var_ucb_mc():
    """Two-cell functional, WoR draws, sigma_bar from the CS: violation rate <= e^{-x} (+ MC slack)."""
    rng = np.random.default_rng(42)
    x = 3.0
    pools = [(rng.random(800) < 0.2).astype(float), (rng.random(600) < 0.5).astype(float)]
    a = np.array([1.0, -1.0])
    n = np.array([120, 90])
    mus = np.array([p.mean() for p in pools])
    reps, viol = 4000, 0
    for _ in range(reps):
        dev, vb = 0.0, []
        for c, p in enumerate(pools):
            s = rng.permutation(p)[: n[c]]
            lo, hi = wor_mean_cs(s, len(p), 0.01)
            vb.append(var_ucb_binary((lo[-1], hi[-1])))
            dev += a[c] * (s.mean() - mus[c])
        viol += dev > functional_width(a, n, np.array(vb), x)
    assert viol / reps <= math.exp(-x) * 1.2 + 0.01


# ------------------------------------------------------------------------------------------------------ acceptance
def test_reviewer_fcr_counterexample():
    nf, nc = reviewer_fcr_counterexample()
    s1 = s1_fwer(nf > 0)
    assert s1["k"] == 10 and s1["rate"] == pytest.approx(0.05)
    assert e_v_over_r(nf, nc) == pytest.approx(0.05)
    assert fcr_old(nf, nc) == pytest.approx(150 / 1670)
    assert abs(fcr_old(nf, nc) - 0.0898) < 1e-4
    s2 = s2_fcr_old(nf, nc)
    assert s2["fcr_old"] == pytest.approx(0.0898, abs=1e-4)
    assert s2["upper"] > 0.05 and not s2["pass"]
    assert not layer_safety(nf, nc)["safe"]


def test_s1_s2_pass_case_and_determinism():
    nf = np.zeros(200)
    nf[:3] = 1
    nc = np.full(200, 12.0)
    out = layer_safety(nf, nc)
    assert out["S1"]["k"] == 3 and out["S1"]["pass"]
    assert out["S2"]["pass"] and out["safe"]
    assert s2_fcr_old(nf, nc)["upper"] == s2_fcr_old(nf, nc)["upper"]          # seed 42 fixed
    assert s2_fcr_old(nf, nc)["upper"] >= fcr_old(nf, nc)
    z = s2_fcr_old(np.zeros(5), np.zeros(5))
    assert z["fcr_old"] == 0.0 and z["upper"] == 0.0
    with pytest.raises(ValueError):
        s2_fcr_old([2], [1])
    five = np.zeros(200)
    five[:5] = 1
    assert not s1_fwer(five > 0)["pass"]


def test_jpc_failure_reproduced():
    nf = np.zeros(200)
    nf[:40] = 3
    nc = np.full(200, 10.0)
    assert jpc_failure_reproduced(nf, nc)["reproduced"]
    nf2 = np.zeros(200)
    nf2[:2] = 1
    assert not jpc_failure_reproduced(nf2, nc)["reproduced"]


def test_e3_and_completion():
    cens = np.array([False] * 130 + [True] * 70)
    e3 = e3_uncensored(cens)
    assert e3["point"] == pytest.approx(0.65) and e3["cp_lower"] >= 0.5 and e3["pass"]
    assert not e3_uncensored(np.array([False] * 110 + [True] * 90))["pass"]
    rng = np.random.default_rng(1)
    r = np.clip(rng.normal(0.6, 0.25, 200), 0, 1)
    c = completion_lower(r)
    assert c["lower"] < c["mean"] and c["pass"]


def test_paired_log_ratio_and_censoring():
    rng = np.random.default_rng(3)
    nx = rng.integers(5000, 60000, 200).astype(float)
    nm = nx * 0.6 * np.exp(rng.normal(0, 0.2, 200))
    out = paired_log_ratio(nm, nx, tau=64000)
    assert out["pass"] and out["ci_hi"] < math.log(0.8) and out["p_one_sided"] < 0.01
    same = paired_log_ratio(nx, nx, tau=64000)
    assert not same["pass"] and same["p_one_sided"] == 1.0
    cens = paired_log_ratio([None, 1000.0], [64000.0, float("inf")], tau=64000, B=100)
    assert cens["mean"] == pytest.approx((0.0 + math.log(1000 / 64000)) / 2)
    with pytest.raises(ValueError):
        paired_log_ratio([None], [10.0], tau=None)


def test_penalized_cost_both_fail():
    out = penalized_cost_log_ratio([1.0, 2.0], [1.0, 2.0], [False, True], [False, True], tau=100.0, B=200)
    assert out["n_both_fail"] == 1
    assert out["mean"] == pytest.approx(math.log(100.0) / 2)
    one = penalized_cost_log_ratio([5.0], [10.0], [False], [True], tau=100.0, B=10)
    assert one["mean"] == pytest.approx(math.log(100.0 / 10.0))


def test_paired_diff_positive():
    rng = np.random.default_rng(5)
    xm = np.clip(rng.normal(0.7, 0.2, 200), 0, 1)
    xx = np.clip(xm - 0.2 + rng.normal(0, 0.1, 200), 0, 1)
    out = paired_diff_positive(xm, xx)
    assert out["pass"] and out["ci_lo"] > 0 and out["p_one_sided"] < 0.01


def test_holm():
    out = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(out["p_adj"], [0.03, 0.06, 0.06, 0.02])
    assert out["reject"] == [True, False, False, True]
    assert holm([0.9, 0.8])["p_adj"] == [1.0, 1.0]
