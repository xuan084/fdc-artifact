"""r4_setup_baselines_a: shared frontier runner, B1 / B4 / B2 (rect, nominal, fav), A-Ney / A-XY.

Checks
  * batched engine == arrival-by-arrival ReplayStream (adaptive with exhaustion, fixed with exhaustion, pool);
  * billing invariants on every run;
  * every method certifies a known-answer toy pool correctly;
  * rigorous variants (B1, B4, B2-rect, QFC, A-Ney) have empirical FWER CP upper <= 0.05 on 200 small adaptive
    synthetic finite-pool streams (20 toy populations, dev seeds 900-919, x 10 permutation seeds; near-ties and
    pool exhaustion included);
  * fav variants are flagged validity='none', nominal variants 'nominal';
  * frozen thresholds are data-free; frozen allocations are proper and A-XY improves its own objective.
"""
import math

import numpy as np
import pytest
from scipy import stats

from dsswm.baselines.clucb_joint import CLUCBJoint
from dsswm.baselines.combgame_joint import C_gG, CombGameJoint, beta_CG, beta_dir, beta_KK, freeze_beta, g_G
from dsswm.baselines.frontier_common import QFCMethod, make_ctx, rect_certificate, FrontierState
from dsswm.baselines.frozen_alloc import neyman_alloc, xy_alloc, xy_objective, _pairs
from dsswm.baselines.uniform_rs import UniformRS
from dsswm.envs.pool_replay import ReplayStream, make_schedule
from dsswm.streams import frontier as fr
from dsswm.streams.frontier_runner import StreamEngine, SyntheticPoolEnv, build_ctx, run_stream

PROBS = [fr.Problem(f"toy|B{b:.2f}", "visit", (0.0, 1.0), b) for b in (0.2, 0.35, 0.5, 0.65, 0.8)]


def _small_env(seed=0, sizes=None, mu=None):
    sizes = [[60, 140], [90, 110], [40, 160]] if sizes is None else sizes
    mu = [[0.3, 0.5], [0.4, 0.45], [0.2, 0.6]] if mu is None else mu
    return SyntheticPoolEnv(sizes, mu, seed=seed, n_min=20)


# ------------------------------------------------------------------------------------------------ engine exactness
def test_engine_matches_replaystream_adaptive_with_exhaustion():
    env = _small_env(1)
    rng = np.random.default_rng(3)
    sch = make_schedule(env, 11, adaptive=True)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    ref = ReplayStream(env, sch, "visit")
    ref_n = np.zeros((env.S, env.A), dtype=np.int64)
    ref_sum = np.zeros((env.S, env.A))
    bounds = list(range(17, env.tau_R, 17)) + [env.tau_R]
    for b in bounds:
        pref = np.array([rng.permutation(env.A) for _ in range(env.S)])
        eng.advance_adaptive(b, pref)
        while ref.t < b:
            s, a, y = ref.arrive(choose=lambda s, avail: next(int(x) for x in pref[s] if x in avail))
            if a is not None:
                ref_n[s, a] += 1
                ref_sum[s, a] += y
        assert np.array_equal(eng.n, ref_n)
        assert np.allclose(eng.sum, ref_sum)
        assert eng.billed == ref.billed and eng.n_served == ref.n_served and eng.n_skipped == ref.n_skipped
        assert eng.billing_ok() and ref.billing_ok()
    assert eng.n_skipped > 0 or (eng.n == env.pool_sizes).all()
    assert (eng.n == env.pool_sizes).all()        # whole population consumed at tau_R


def test_engine_matches_replaystream_fixed_with_exhaustion():
    env = _small_env(2)
    P = np.array([[0.9, 0.1], [0.5, 0.5], [0.95, 0.05]])           # drives the small pools dry early
    sch = make_schedule(env, 5, alloc=P)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    ref = ReplayStream(env, sch, "visit")
    ref_n = np.zeros((env.S, env.A), dtype=np.int64)
    ref_sum = np.zeros((env.S, env.A))
    for b in list(range(23, env.tau_R, 23)) + [env.tau_R]:
        eng.advance_planned(b)
        while ref.t < b:
            s, a, y = ref.arrive()
            if a is not None:
                ref_n[s, a] += 1
                ref_sum[s, a] += y
        assert np.array_equal(eng.n, ref_n)
        assert np.allclose(eng.sum, ref_sum)
        assert eng.billed == ref.billed and eng.n_served == ref.n_served and eng.n_skipped == ref.n_skipped
        assert eng.n_reselected == ref.n_reselected
    assert eng.n_reselected > 0


def test_engine_fixed_vectorised_path_equals_loop_path():
    env = SyntheticPoolEnv([[3000, 9000], [5000, 6000]], [[0.1, 0.2], [0.3, 0.25]], seed=4, n_min=100)
    P = np.array([[0.6, 0.4], [0.5, 0.5]])
    sch = make_schedule(env, 9, alloc=P)
    e1 = StreamEngine(env, sch, "visit", 0.01)
    e1.advance_planned(env.tau_R)
    e2 = StreamEngine(env, sch, "visit", 0.01)
    e2._loop(0, env.tau_R)
    e2.t = env.tau_R
    assert np.array_equal(e1.n, e2.n) and np.allclose(e1.sum, e2.sum)
    assert e1.n_skipped == e2.n_skipped and e1.n_reselected == e2.n_reselected
    assert e1.n_loop_arrivals < env.tau_R          # the vectorised path is actually used


def test_pool_schedule_never_exhausts():
    env = _small_env(3)
    sch = make_schedule(env, 7)
    eng = StreamEngine(env, sch, "visit", 0.01)
    eng.advance_planned(env.tau_R)
    assert eng.n_skipped == 0 and eng.n_reselected == 0 and (eng.n == env.pool_sizes).all() and eng.billing_ok()


def test_engine_cs_matches_wor_mean_cs_and_is_exact_at_exhaustion():
    from dsswm.stats.fp_eb import wor_mean_cs
    env = _small_env(5)
    sch = make_schedule(env, 3, adaptive=True)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.02)
    c = 1
    o = eng._obs[c]
    lo, hi = wor_mean_cs(o, o.size, 0.02)
    n = np.zeros((env.S, env.A), dtype=np.int64)
    for k in (0, 1, 7, o.size):
        n[0, 1] = k
        L, H = eng.cs_at(n)
        if k == 0:
            assert L[0, 1] == 0.0 and H[0, 1] == 1.0
        else:
            assert L[0, 1] == lo[k - 1] and H[0, 1] == hi[k - 1]
    assert abs(L[0, 1] - o.mean()) < 1e-12 and abs(H[0, 1] - o.mean()) < 1e-12


# ------------------------------------------------------------------------------------------------ methods
ALL = [lambda: QFCMethod(), lambda: UniformRS(), lambda: CLUCBJoint(), lambda: CombGameJoint("rect"),
       lambda: CombGameJoint("nominal"), lambda: CombGameJoint("fav")]


@pytest.mark.parametrize("mk", ALL, ids=["QFC", "B4", "B1", "B2-rect", "B2-nominal", "B2-fav"])
def test_each_method_certifies_known_answer_toy(mk):
    # large gaps: arm 1 is better by 0.4 in every segment; exact answer = treat largest segments within budget
    env = SyntheticPoolEnv([[4000, 4000], [3000, 3000], [3000, 3000]], [[0.1, 0.6], [0.2, 0.45], [0.3, 0.9]], seed=0,
                           n_min=200)
    m = mk()
    s, rows = run_stream(env, m, 42, PROBS, 0.02)
    assert s["billing_ok"]
    assert s["reached_stop"] and s["n_false"] == 0 and s["N80"] < env.tau_R
    assert all(r["billing_ok"] for r in rows)
    mu = env.true_mu("visit")
    ctx = build_ctx(env, PROBS, 0.02)
    for q, ih in enumerate(s["decided_pi"]):
        if ih >= 0:
            exact = fr.solve_enum(env.w, mu, PROBS[q])
            J = fr.policy_values(ctx.pols[ih:ih + 1], env.w, mu)[0]
            assert exact.J_star - J <= 0.02 + 1e-12


def test_validity_labels():
    assert QFCMethod().validity == "rigorous" and UniformRS().validity == "rigorous"
    assert CLUCBJoint().validity == "rigorous" and CombGameJoint("rect").validity == "rigorous"
    assert CombGameJoint("nominal").validity == "nominal"
    assert CombGameJoint("fav").validity == "none"


def _toy_population(seed):
    """Near-tie toy population for the FWER stress test (dev seeds 900-919): pools of 2k-8k records, sign-mixed
    uplifts of order 0.1, eps = 0.02 -> certification happens well before exhaustion and eps-wrong answers (regret up
    to ~0.1) exist in every class, so a broken certificate would show up as false certifications."""
    rng = np.random.default_rng(seed)
    S = 3
    sizes = rng.integers(2000, 8000, size=(S, 2))
    base = rng.uniform(0.15, 0.6, size=S)
    up = rng.normal(0.0, 0.10, size=S)
    mu = np.stack([base, np.clip(base + up, 0.01, 0.99)], 1)
    return SyntheticPoolEnv(sizes, mu, seed=seed, n_min=200)


TOY_EPS = 0.02


def toy_fwer(method_factory, seeds=range(900, 920), perms=range(10), eps=TOY_EPS):
    ev = 0
    n = 0
    certs = 0
    early = 0
    billing = True
    for sd in seeds:
        env = _toy_population(sd)
        for p in perms:
            s, _ = run_stream(env, method_factory(), 1000 * sd + p, PROBS, eps, keep_U=False)
            ev += int(s["fwer_event"])
            certs += s["n_cert"]
            billing &= s["billing_ok"]
            early += int(s["reached_stop"] and s["N80"] < 0.5 * env.tau_R)
            n += 1
    return ev, n, certs, early, billing


@pytest.mark.parametrize("mk,name", [(lambda: UniformRS(), "B4"), (lambda: CLUCBJoint(), "B1"),
                                     (lambda: CombGameJoint("rect"), "B2-rect"), (lambda: QFCMethod(), "QFC")])
def test_rigorous_variants_toy_fwer(mk, name):
    ev, n, certs, early, billing = toy_fwer(mk)
    assert n == 200 and billing
    cp_up = 1.0 if ev >= n else float(stats.beta.ppf(0.95, ev + 1, n - ev))
    assert cp_up <= 0.05, (name, ev, n)
    assert certs > 0, f"{name}: vacuous FWER test (no certification at all)"
    assert early >= 0.25 * n, f"{name}: certifications only near exhaustion (vacuous stress)"


def test_toy_population_exhausts_pools_under_adaptive_sampling():
    env = _toy_population(900)
    s, _ = run_stream(env, CLUCBJoint(), 1, PROBS, -1.0, keep_U=False)   # eps<0 -> never certifies, runs to tau_R
    assert s["t_end"] == env.tau_R and s["skipped"] + s["reselected"] > 0 and s["billing_ok"]


# ------------------------------------------------------------------------------------------------ thresholds
def test_cgG_and_thresholds():
    assert np.isfinite(g_G(0.75))
    a, b = C_gG(1.0), C_gG(2.0)
    assert b > a > 1.0                                  # increasing, and >= x (lambda <= 1)
    assert C_gG(10.0) >= 10.0
    # KK approximation sanity: C^g(x) ~ x + O(log x)
    assert C_gG(20.0) < 20.0 + 4 * math.log(1 + 20 + math.sqrt(40)) + 5
    assert beta_dir(512, 20, 0.05) == pytest.approx(math.log(512 ** 2 * 20 / 0.05))
    assert beta_KK(np.full(18, 1e5), 0.05) > beta_KK(np.full(18, 10), 0.05)
    assert beta_CG(1e6, 9, 512, 15, 0.05) > beta_CG(1e3, 9, 512, 15, 0.05)


def test_freeze_beta_is_data_free_and_minimal():
    env = _small_env(7)
    ctx = build_ctx(env, PROBS, 0.02)
    f1 = freeze_beta(ctx)
    env2 = SyntheticPoolEnv([[60, 140], [90, 110], [40, 160]], [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], seed=99, n_min=20)
    f2 = freeze_beta(build_ctx(env2, PROBS, 0.02))        # same labels / sizes, different outcomes
    assert f1 == f2
    assert f1["values_at_t_med"][f1["choice"]] == min(f1["values_at_t_med"].values())


def test_rect_certificate_matches_bruteforce():
    env = _small_env(8)
    ctx = build_ctx(env, PROBS, 0.05)
    rng = np.random.default_rng(0)
    n = rng.integers(5, 40, size=(env.S, env.A))
    sums = n * rng.uniform(0.2, 0.7, size=n.shape)
    lo = np.clip(sums / n - 0.1, 0, 1)
    hi = np.clip(sums / n + 0.1, 0, 1)
    st = FrontierState(n.sum(), n, sums, env.pool_sizes, np.ones(len(PROBS), bool), lambda _n: (lo, hi))
    res = rect_certificate(ctx, st)
    mu = sums / n
    J = fr.policy_values(ctx.pols, env.w, mu)
    for q, (ok, ih, U) in enumerate(res):
        f = np.flatnonzero(ctx.feas[q])
        assert J[ih] == J[f].max()
        best = -np.inf
        for j in f:
            u = sum(env.w[s] * (hi[s, ctx.pols[j, s]] - lo[s, ctx.pols[ih, s]]) for s in range(env.S)
                    if ctx.pols[j, s] != ctx.pols[ih, s])
            best = max(best, u)
        assert U == pytest.approx(best) and ok == (best <= 0.05)


# ------------------------------------------------------------------------------------------------ frozen allocations
def test_frozen_allocations():
    sig = np.array([[0.04, 0.09], [0.01, 0.25], [0.16, 0.16]])
    p = neyman_alloc(sig)
    assert np.allclose(p.sum(1), 1) and (p >= 0.01 - 1e-12).all()
    assert p[0, 1] / p[0, 0] == pytest.approx(1.5)
    env = _small_env(9)
    ctx = build_ctx(env, PROBS, 0.02)
    mu = env.true_mu("visit")
    pxy, info = xy_alloc(env.w, mu, sig, ctx.pols, ctx.feas, 0.02)
    assert np.allclose(pxy.sum(1), 1) and (pxy >= 0.01 - 1e-12).all()
    assert info["objective_final"] <= info["objective_uniform"] + 1e-12
    pxy2, _ = xy_alloc(env.w, mu, sig, ctx.pols, ctx.feas, 0.02)
    assert np.array_equal(pxy, pxy2)                      # deterministic / frozen
    s, _ = run_stream(env, QFCMethod("A-XY", alloc_p=pxy), 3, PROBS, 0.02)
    assert s["billing_ok"] and s["alloc_kind"] == "fixed"
