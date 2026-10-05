"""r5_baseline_qualification: new rigorous rivals (B4-bal, Hait-SW, Molitor-WoR) and the r5 registry.

Checks
  * registry: every member of R resolves, is labelled rigorous and has the declared allocation kind;
    non-rigorous / ablation methods are never in R; tuning-grid points stay rigorous;
  * B4-bal: proper shares, realised share before exhaustion, billing with exhaustion, invalid shares rejected;
  * Hait-SW: share law (gamma = 2/3 default, floor, Laplace at n = 0), one-batch-lag predictability, plan returns
    permutations, realised shares track the targets, billing with exhaustion;
  * Molitor-WoR: certificate equals a brute-force loop, dominates (is never tighter than) the rectangle bound for the
    same leader, single-feasible-policy edge case;
  * every new rival certifies a known-answer toy correctly; a quick 40-stream near-tie FWER smoke (0 false streams).
"""
import numpy as np
import pytest

from dsswm.baselines.b4_bal import B4Bal, balanced_alloc
from dsswm.baselines.frontier_common import FrontierState, make_ctx, pi_hat_indices, rect_U
from dsswm.baselines.hait_sw import HaitSW, hait_shares
from dsswm.baselines.molitor_wor import MolitorWoR, molitor_certificate
from dsswm.streams import frontier as fr
from dsswm.streams import r5_registry as reg
from dsswm.streams.frontier_runner import StreamEngine, SyntheticPoolEnv, build_ctx, run_stream
from dsswm.envs.pool_replay import make_schedule
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population

NEW = {"B4-bal": lambda: B4Bal(), "Hait-SW": lambda: HaitSW(), "Molitor-WoR": lambda: MolitorWoR()}


# ------------------------------------------------------------------------------------------------ registry
def test_registry_rigorous_set_resolves_and_is_rigorous():
    for name in reg.RIGOROUS_SET_R:
        sp = reg.spec(name)
        assert sp.in_R and sp.role == "rival_R" and sp.validity_class == "rigorous"
        m = reg.make_method(name)
        assert m.validity == "rigorous", name
        assert m.name == name, (name, m.name)
    assert set(reg.names("rival_R")) == set(reg.RIGOROUS_SET_R)


def test_registry_excludes_nominal_plugin_and_ablation_from_R():
    for name, sp in reg.REGISTRY.items():
        if sp.validity_class != "rigorous" or sp.role in ("ablation", "main", "descriptive", "excluded"):
            assert name not in reg.RIGOROUS_SET_R
    for name in ("QFC-half", "B2-nominal", "B2-fav", "Peace-fav", "B3-fav", "B5"):
        assert name not in reg.RIGOROUS_SET_R
    assert reg.make_method("B2-nominal").validity == "nominal"
    assert reg.make_method("B2-fav").validity == "none"


def test_registry_tuning_grid_points_are_rigorous():
    for name in ("B4-bal", "Hait-SW"):
        for kw in reg.TUNING_GRID[name]:
            m = reg.make_method(name, **kw)
            assert m.validity == "rigorous" and m.name == name


def test_registry_alloc_kinds():
    assert reg.make_method("B4-bal").alloc_kind == "fixed"
    assert reg.make_method("Hait-SW").alloc_kind == "adaptive"
    assert reg.make_method("Molitor-WoR").alloc_kind == "pool"
    assert reg.make_method("B4").alloc_kind == "pool"


def test_registry_qfc_half_is_fifty_fifty_fixed():
    env = SyntheticPoolEnv([[300, 300], [200, 400]], [[0.3, 0.5], [0.4, 0.45]], seed=1, n_min=20)
    ctx = build_ctx(env, PROBS, 0.02)
    m = reg.make_method("QFC-half")
    m.setup(ctx)
    assert m.alloc_kind == "fixed" and np.allclose(m.alloc_p, 0.5) and m.alloc_p.shape == (2, 2)


# ------------------------------------------------------------------------------------------------ B4-bal
def test_balanced_alloc_rows():
    p = balanced_alloc(3, 2, 0.4)
    assert p.shape == (3, 2) and np.allclose(p[:, 0], 0.4) and np.allclose(p.sum(1), 1)
    p3 = balanced_alloc(2, 3, 0.5)
    assert np.allclose(p3[0], [0.5, 0.25, 0.25])
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            balanced_alloc(2, 2, bad)


@pytest.mark.parametrize("share", [0.40, 0.45, 0.50])
def test_b4bal_realised_share_and_billing(share):
    env = SyntheticPoolEnv([[20000, 20000], [15000, 25000]], [[0.3, 0.5], [0.4, 0.45]], seed=2, n_min=200)
    m = B4Bal(share=share)
    ctx = build_ctx(env, PROBS, 0.02)
    m.setup(ctx)
    sch = make_schedule(env, 11, alloc=m.alloc_p)
    eng = StreamEngine(env, sch, "visit", 0.05 / 4)
    eng.advance_planned(20000)                         # no exhaustion yet
    real = eng.n[:, 0] / eng.n.sum(1)
    assert np.allclose(real, share, atol=0.015)
    eng.advance_planned(env.tau_R)                     # through exhaustion
    assert eng.billing_ok() and (eng.n == eng.N).all()


# ------------------------------------------------------------------------------------------------ Hait-SW
def test_hait_shares_law_floor_laplace():
    n = np.array([[1000, 1000], [0, 0]])
    sums = np.array([[100, 500], [0, 0]])                # means 0.1 / 0.5 -> sigma 0.3 / 0.5
    p = hait_shares(n, sums, gamma=2 / 3, p_min=0.0)
    m = (sums + 1) / (n + 2)
    sig = np.sqrt(m * (1 - m))
    expect = sig[0] ** (2 / 3) / (sig[0] ** (2 / 3)).sum()
    assert np.allclose(p[0], expect)
    assert np.allclose(p[1], 0.5)                        # Laplace at n = 0: equal shares
    pn = hait_shares(n, sums, gamma=1.0, p_min=0.0)      # Neyman option
    assert np.allclose(pn[0], sig[0] / sig[0].sum())
    assert pn[0, 0] < p[0, 0] < 0.5                       # 2/3 law is between Neyman and equal
    pf = hait_shares(np.array([[1000, 1000]]), np.array([[0, 500]]), p_min=0.10)
    assert pf.min() >= 0.10 - 1e-12 and np.allclose(pf.sum(1), 1)
    with pytest.raises(ValueError):
        hait_shares(n, sums, p_min=0.6)


def test_hait_uses_one_batch_lagged_data():
    env = SyntheticPoolEnv([[300, 300], [200, 400]], [[0.3, 0.5], [0.4, 0.45]], seed=1, n_min=20)
    ctx = build_ctx(env, PROBS, 0.02)
    m = HaitSW(p_min=0.0)
    m.setup(ctx)
    und = np.ones(ctx.Q, dtype=bool)
    st0 = FrontierState(0, np.zeros((2, 2)), np.zeros((2, 2)), env.pool_sizes, und, lambda n: None)
    m.plan(ctx, st0)
    assert np.allclose(m._p, 0.5)                        # nothing observed before the previous batch
    n1, s1 = np.array([[50, 50], [40, 60]]), np.array([[2, 25], [20, 30]])
    m.plan(ctx, FrontierState(100, n1, s1, env.pool_sizes, und, lambda n: None))
    assert np.allclose(m._p, 0.5)                        # lag: uses the snapshot of call 0 (empty)
    n2, s2 = np.array([[100, 100], [80, 120]]), np.array([[5, 50], [40, 60]])
    m.plan(ctx, FrontierState(200, n2, s2, env.pool_sizes, und, lambda n: None))
    assert np.allclose(m._p, hait_shares(n1, s1, p_min=0.0))   # call 2 uses call-1 data, not call-2 data
    assert not np.allclose(m._p, hait_shares(n2, s2, p_min=0.0))


def test_hait_plan_permutations_tracking_and_billing():
    env = SyntheticPoolEnv([[6000, 6000], [5000, 7000], [3000, 9000]], [[0.05, 0.5], [0.4, 0.45], [0.2, 0.6]],
                           seed=3, n_min=200)
    m = HaitSW(p_min=0.02)
    s, rows = run_stream(env, m, 7, PROBS, 0.0, keep_U=False)      # eps = 0: runs to tau_R (exhaustion)
    assert s["billing_ok"] and s["n_plans"] > 10 and all(r["billing_ok"] for r in rows)
    assert s["t_end"] == env.tau_R or s["reached_stop"]
    # realised share before exhaustion tracks the lagged 2/3-law targets: segment 0 (sigma 0.22 vs 0.5) is
    # under-sampled on the control arm
    first = min(rows, key=lambda r: abs(r["t"] - 12000))
    n = np.array(first["n_cells"]).reshape(3, 2)
    assert n[0, 0] / n[0].sum() < 0.47


# ------------------------------------------------------------------------------------------------ Molitor-WoR
def _rand_state(ctx, rng):
    n = rng.integers(5, 50, size=(ctx.S, ctx.A))
    lo = rng.uniform(0.0, 0.5, size=(ctx.S, ctx.A))
    hi = lo + rng.uniform(0.01, 0.4, size=(ctx.S, ctx.A))
    mu = (lo + hi) / 2
    return FrontierState(100, n, mu * n, np.full((ctx.S, ctx.A), 1000), np.ones(ctx.Q, bool), lambda _n: (lo, hi)), lo, hi


def test_molitor_matches_bruteforce_and_dominates_rectangle():
    env = SyntheticPoolEnv([[300, 300], [200, 400], [250, 250]], [[0.3, 0.5], [0.4, 0.45], [0.2, 0.6]], seed=1,
                           n_min=20)
    ctx = build_ctx(env, PROBS, 0.02)
    rng = np.random.default_rng(0)
    for _ in range(20):
        st, lo, hi = _rand_state(ctx, rng)
        res = molitor_certificate(ctx, st)
        for q in range(ctx.Q):
            best = None
            for i in range(ctx.P):
                if not ctx.feas[q, i]:
                    continue
                L = sum(ctx.w[s] * lo[s, ctx.pols[i, s]] for s in range(ctx.S))
                if best is None or L > best[0] + 1e-15:
                    best = (L, i)
            L0, ih = best
            U = max([sum(ctx.w[s] * hi[s, ctx.pols[j, s]] for s in range(ctx.S)) - L0
                     for j in range(ctx.P) if ctx.feas[q, j] and j != ih], default=0.0)
            assert res[q][1] == ih and abs(res[q][2] - U) < 1e-12
            # domination: Molitor's bound >= the rectangle bound for the same leader
            Ur = np.where(ctx.feas[q], rect_U(ctx, lo, hi, ih), -np.inf)
            Ur[ih] = -np.inf
            if np.isfinite(Ur).any():
                assert res[q][2] >= Ur.max() - 1e-12


def test_molitor_single_feasible_policy():
    env = SyntheticPoolEnv([[300, 300], [200, 400]], [[0.3, 0.5], [0.4, 0.45]], seed=1, n_min=20)
    probs = [fr.Problem("toy|B0", "visit", (0.0, 1.0), 0.0)]       # budget 0 -> only the all-control policy
    ctx = build_ctx(env, probs, 0.02)
    assert ctx.feas[0].sum() == 1
    st, _, _ = _rand_state(ctx, np.random.default_rng(1))
    ok, ih, U = molitor_certificate(ctx, st)[0]
    assert ok and U == 0.0 and ctx.feas[0, ih]


# ------------------------------------------------------------------------------------------------ end-to-end
@pytest.mark.parametrize("name", list(NEW))
def test_new_rival_certifies_known_answer_toy(name):
    env = SyntheticPoolEnv([[4000, 4000], [3000, 3000], [3000, 3000]], [[0.1, 0.6], [0.2, 0.45], [0.3, 0.9]], seed=0,
                           n_min=200)
    s, rows = run_stream(env, NEW[name](), 42, PROBS, 0.02)
    assert s["billing_ok"] and s["validity"] == "rigorous"
    assert s["reached_stop"] and s["n_false"] == 0 and s["N80"] < env.tau_R
    mu = env.true_mu("visit")
    ctx = build_ctx(env, PROBS, 0.02)
    for q, ih in enumerate(s["decided_pi"]):
        if ih >= 0:
            exact = fr.solve_enum(env.w, mu, PROBS[q])
            J = fr.policy_values(ctx.pols[ih:ih + 1], env.w, mu)[0]
            assert exact.J_star - J <= 0.02 + 1e-12


@pytest.mark.parametrize("name", list(NEW))
def test_new_rival_toy_fwer_smoke(name):
    ev = 0
    for sd in (900, 901, 902, 903):
        env = _toy_population(sd)
        for p in range(10):
            s, _ = run_stream(env, NEW[name](), 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
            ev += int(s["fwer_event"])
            assert s["billing_ok"]
    assert ev == 0
