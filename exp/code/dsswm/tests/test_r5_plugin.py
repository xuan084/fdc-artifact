"""r5_setup_plugin_variants: tuned plug-in descriptive comparators (B2-fav-tight, FIX-bal-fav, Peace-fav-bal).

Checks
  * registry: the three names resolve through ``r5_registry.make_method``, are labelled validity 'none' / role
    'descriptive', are never in the rigorous rival set R, and have the declared allocation kinds;
  * beta_tight = ln(sum_q |Pi_Bq| K / delta) from ctx (CR9 dev: ~= 14.12);
  * B2-fav-tight is exactly CombGame('fav') with beta_override = beta_tight (row-by-row identical streams);
  * FIX-bal-fav: frozen 50/50 design, certificate equals glr_certificate(plug-in var, beta_tight); billing with
    exhaustion;
  * Peace-fav-bal: frozen balanced design, rounds start at t = 0 and close only at checkpoints, deterministic,
    billing with exhaustion, certifies a known-answer toy correctly;
  * CR9 dev seed 900: B2-fav-tight / FIX-bal-fav identical (every checkpoint row, incl. U) to the offline contrarian
    script ``idea/r5_offline/contrarian/offline_ctr.py`` (skipped if the dataset or the script is absent).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.baselines.b4_bal import balanced_alloc
from dsswm.baselines.combgame_joint import CombGameJoint
from dsswm.baselines.frontier_common import FrontierState, glr_certificate, plugin_var
from dsswm.baselines.plugin_r5 import B2FavTight, FixBalFav, PeaceFavBal, beta_tight
from dsswm.streams import frontier as fr
from dsswm.streams import r5_registry as reg
from dsswm.streams.frontier_runner import SyntheticPoolEnv, build_ctx, run_stream
from dsswm.tests.test_baselines_r4a import PROBS

NAMES = ("B2-fav-tight", "FIX-bal-fav", "Peace-fav-bal")
WS = Path(__file__).resolve().parents[4]
OFFLINE = WS / "idea/r5_offline/contrarian/offline_ctr.py"


def _easy_env(seed=0):
    return SyntheticPoolEnv([[4000, 4000], [3000, 3000], [3000, 3000]], [[0.1, 0.6], [0.2, 0.45], [0.3, 0.9]],
                            seed=seed, n_min=200)


def _exhaust_env(seed=1):
    # small, unbalanced pools: the balanced design exhausts the small arm pools before tau_R
    return SyntheticPoolEnv([[60, 400], [300, 80], [50, 250]], [[0.3, 0.5], [0.4, 0.45], [0.2, 0.6]], seed=seed,
                            n_min=20)


# ------------------------------------------------------------------------------------------------ registry
def test_registry_resolves_plugin_variants_descriptive_not_in_R():
    kinds = {"B2-fav-tight": "adaptive", "FIX-bal-fav": "fixed", "Peace-fav-bal": "fixed"}
    for n in NAMES:
        sp = reg.spec(n)
        assert sp.validity_class == "none" and sp.role == "descriptive" and not sp.in_R
        assert n not in reg.RIGOROUS_SET_R
        m = reg.make_method(n)
        assert m.name == n and m.validity == "none" and m.alloc_kind == kinds[n]
    assert set(reg.names("rival_R")) == set(reg.RIGOROUS_SET_R)


def test_validity_stays_none_after_setup():
    env = _easy_env()
    ctx = build_ctx(env, PROBS, 0.02)
    for n in NAMES:
        m = reg.make_method(n)
        m.setup(ctx)
        assert m.validity == "none"


# ------------------------------------------------------------------------------------------------ beta
def test_beta_tight_formula():
    env = _easy_env()
    ctx = build_ctx(env, PROBS, 0.02)
    assert beta_tight(ctx) == pytest.approx(math.log(int(ctx.feas.sum()) * len(ctx.checkpoints) / ctx.delta))
    b2, fx = B2FavTight(), FixBalFav()
    b2.setup(ctx)
    fx.setup(ctx)
    assert b2.beta_override == fx.beta == beta_tight(ctx)
    fb = FixBalFav(beta=10.0)
    fb.setup(ctx)
    assert fb.beta == 10.0


# ------------------------------------------------------------------------------------------------ B2-fav-tight
def test_b2_fav_tight_equals_combgame_fav_with_override():
    env = _exhaust_env(2)
    ctx = build_ctx(env, PROBS, 0.02)
    ref = CombGameJoint("fav", beta_override=beta_tight(ctx), name="ref")
    for seed in (5, 6):
        s1, r1 = run_stream(env, B2FavTight(), seed, PROBS, 0.02)
        s2, r2 = run_stream(env, CombGameJoint("fav", beta_override=beta_tight(ctx), name="ref"), seed, PROBS, 0.02)
        assert s1["billing_ok"] and len(r1) == len(r2)
        for x, y in zip(r1, r2):
            for k in ("n_cert", "n_cells", "U", "billed", "reselected"):
                assert x[k] == y[k], k
        assert s1["N80"] == s2["N80"] and s1["decided_pi"] == s2["decided_pi"]
    assert ref.validity == "none"


# ------------------------------------------------------------------------------------------------ FIX-bal-fav
def test_fix_bal_fav_design_and_certificate():
    env = _easy_env()
    ctx = build_ctx(env, PROBS, 0.02)
    m = FixBalFav()
    m.setup(ctx)
    assert np.allclose(m.alloc_p, balanced_alloc(ctx.S, ctx.A, 0.5))
    rng = np.random.default_rng(0)
    n = rng.integers(50, 500, size=(ctx.S, ctx.A))
    sums = rng.binomial(n, 0.4).astype(float)
    st = FrontierState(int(n.sum()), n, sums, ctx.N, np.ones(ctx.Q, bool), lambda nn: None)
    assert m.certify(ctx, st) == glr_certificate(ctx, st, plugin_var(st, ctx.R), beta_tight(ctx))


def test_fix_bal_fav_realised_share_and_billing():
    env = _easy_env(3)
    m = FixBalFav()
    s, rows = run_stream(env, m, 11, PROBS, 0.0)        # eps = 0: never certifies -> runs to tau_R
    assert s["billing_ok"]
    first = np.asarray(rows[0]["n_cells"]).reshape(env.S, env.A)
    share = first[:, 0] / first.sum(1)
    assert np.all(np.abs(share - 0.5) < 0.1)
    s2, _ = run_stream(_exhaust_env(), FixBalFav(), 3, PROBS, 0.02)
    assert s2["billing_ok"]


def test_pool_variant_is_pool_alloc():
    m = FixBalFav(share=None, name="POOL-fav-tight")
    assert m.alloc_kind == "pool" and m.alloc_p is None


# ------------------------------------------------------------------------------------------------ Peace-fav-bal
def test_peace_fav_bal_rounds_checkpoint_aligned_and_deterministic():
    env = _exhaust_env(4)
    ctx = build_ctx(env, PROBS, 0.02)
    m1, m2 = PeaceFavBal(n_eta=300), PeaceFavBal(n_eta=300)
    s1, r1 = run_stream(env, m1, 9, PROBS, 0.02)
    s2, r2 = run_stream(env, m2, 9, PROBS, 0.02)
    assert s1["billing_ok"] and s1["validity"] == "none"
    assert [x["U"] for x in r1] == [x["U"] for x in r2] and s1["N80"] == s2["N80"]
    assert np.allclose(m1.p, balanced_alloc(ctx.S, ctx.A, 0.5))
    assert m1.log[0]["t_start"] == 0
    ck = set(int(c) for c in ctx.checkpoints)
    for r in m1.log[1:]:
        assert r["t_start"] in ck
    for r in m1.log:
        if "t_end_actual" in r:
            assert r["t_end_actual"] in ck and r["t_end_actual"] >= r["t_start"] + r["len"]
    with pytest.raises(RuntimeError):
        m1.plan(ctx, None)


def test_peace_fav_bal_known_answer_toy():
    env = _easy_env()
    s, _ = run_stream(env, PeaceFavBal(n_eta=500), 42, PROBS, 0.02)
    assert s["billing_ok"] and s["reached_stop"] and s["n_false"] == 0
    mu = env.true_mu("visit")
    ctx = build_ctx(env, PROBS, 0.02)
    for q, ih in enumerate(s["decided_pi"]):
        if ih >= 0:
            exact = fr.solve_enum(env.w, mu, PROBS[q])
            assert exact.J_star - fr.policy_values(ctx.pols[ih:ih + 1], env.w, mu)[0] <= 0.02 + 1e-12


# ------------------------------------------------------------------------------------------------ CR9 identity
def _offline():
    if not OFFLINE.exists():
        pytest.skip("offline contrarian script absent")
    sys.path.insert(0, str(OFFLINE.parent))
    try:
        import offline_ctr as oc
        oc.init()
    except Exception as e:  # dataset missing
        pytest.skip(f"CR9 dev layer unavailable: {e}")
    ctx = oc.G["ctx"]
    oc.TB = math.log(int(ctx.feas.sum()) * len(ctx.checkpoints) / ctx.delta)
    return oc


@pytest.mark.parametrize("new,off", [("FIX-bal-fav", "FIX0.50-fav-tight"), ("B2-fav-tight", "B2-tightbeta")])
def test_cr9_dev_identity_with_offline_script(new, off):
    oc = _offline()
    G, ctx = oc.G, oc.G["ctx"]
    assert beta_tight(ctx) == pytest.approx(14.1189, abs=1e-3)
    s1, r1 = run_stream(G["env"], reg.make_method(new), 900, ctx.problems, oc.EPS, ctx=ctx, J_true=G["Jt"],
                        J_star=G["Js"])
    s2, r2 = run_stream(G["env"], oc.mk(off), 900, ctx.problems, oc.EPS, ctx=ctx, J_true=G["Jt"], J_star=G["Js"])
    assert len(r1) == len(r2)
    for x, y in zip(r1, r2):
        for k in ("n_cert", "n_false", "n_cells", "U", "billed", "served", "skipped", "reselected"):
            assert x[k] == y[k], (k, x["k"])
    for k in ("N80", "censored", "decided_pi", "cert_k", "billed", "schedule_digest"):
        assert s1[k] == s2[k], k
    assert s1["validity"] == s2["validity"] == "none"
