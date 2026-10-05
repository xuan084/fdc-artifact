"""Unit tests for the lock-v6 addendum code (rect_v6, wor_betting_v6, frontier_runner_v6, lenta_v6, prereg_v6,
v6_analysis).  Fast: synthetic pools only, except one structural Lenta check (labels / split, no eval outcomes)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy.stats import hypergeom

from dsswm.baselines.b4_bal import B4Bal
from dsswm.baselines.rect_v6 import (RectCkBern, RectCkHG, RectCkHGLive, V6_DELTA_MAIN, V6_DELTA_VAR, bern_mean_interval,
                                     hg_interval_counts)
from dsswm.baselines.wor_betting_v6 import HC_GRID, HCWoRRect, hc_lambda_tilde, hc_wor_cs_at
from dsswm.envs.data_v6 import DATA_FILES, data_drift, file_sha256
from dsswm.stats import prereg, prereg_v6
from dsswm.stats.v6_analysis import boot_idx, decide_block_a, decide_block_c, paired
from dsswm.streams import frontier_runner as frr
from dsswm.streams.frontier_runner import SyntheticPoolEnv, build_ctx, run_stream
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6
from dsswm.streams import frontier as fr
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


# --------------------------------------------------------------------------------------------- exact hypergeometric
@pytest.mark.parametrize("N,n", [(30, 7), (40, 20), (25, 1), (50, 49)])
def test_hg_interval_exact_coverage(N, n):
    """For every true M the exact miscoverage P_M(M not in I(X)) <= 2 a (computed by summing the pmf)."""
    a = 0.02
    I = {x: hg_interval_counts(N, n, x, a) for x in range(0, n + 1)}
    for M in range(0, N + 1):
        miss = 0.0
        for x in range(max(0, n - (N - M)), min(n, M) + 1):
            lo, hi = I[x]
            if not lo <= M <= hi:
                miss += hypergeom.pmf(x, N, M, n)
        assert miss <= 2 * a + 1e-12, (N, n, M, miss)


def test_hg_interval_edges_and_monotone():
    assert hg_interval_counts(100, 0, 0, 0.01) == (0, 100)
    assert hg_interval_counts(100, 100, 37, 0.01) == (37, 37)
    lo_prev = -1
    for x in range(0, 21):
        lo, hi = hg_interval_counts(200, 20, x, 0.01)
        assert x <= lo <= hi <= 200 - 20 + x and lo >= lo_prev
        lo_prev = lo


# --------------------------------------------------------------------------------------------- Bernstein checkpoint
def test_bern_interval_is_lemma_l1_radius():
    N, n, s = np.array([1000.0]), np.array([200.0]), np.array([30.0])
    beta, xv = 9.0, 11.0
    lo, hi = bern_mean_interval(N, n, s, beta, xv)
    from dsswm.theory_checks.mc_l1 import bernstein_mu_ci, sigma2_ucb
    sig2 = sigma2_ucb(*bernstein_mu_ci(np.array([0.15]), n, N, xv))
    rad = math.sqrt(2 * sig2[0] * beta / 200) + beta / 600
    assert lo[0] == pytest.approx(max(0.15 - rad, 30 / 1000))
    assert hi[0] == pytest.approx(min(0.15 + rad, (30 + 800) / 1000))
    lo, hi = bern_mean_interval(np.array([50.0]), np.array([50.0]), np.array([10.0]), beta, xv)
    assert lo[0] == hi[0] == pytest.approx(0.2)


def test_ck_ledgers_sum_to_delta():
    env = _toy_population(901)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    b = RectCkBern()
    b.setup(ctx)
    assert b.ledger["bound_main"] == pytest.approx(V6_DELTA_MAIN) and b.ledger["bound_var"] == pytest.approx(V6_DELTA_VAR)
    h = RectCkHG()
    h.setup(ctx)
    assert h.ledger["bound"] == pytest.approx(0.05)


# --------------------------------------------------------------------------------------------- hedged capital WoR CS
def test_hc_cs_exhausted_and_empty():
    x = np.array([1.0, 0.0, 1.0, 1.0])
    lam = hc_lambda_tilde(x, 0.05, "prpl")
    assert hc_wor_cs_at(x, 4, 0.05, lam, 0.5) == (0.75, 0.75)
    assert hc_wor_cs_at(x[:0], 4, 0.05, lam, 0.5) == (0.0, 1.0)


def test_hc_cs_contains_logical_bounds_and_is_interval():
    rng = np.random.default_rng(0)
    pool = (rng.random(3000) < 0.2).astype(float)
    x = rng.permutation(pool)
    lam = hc_lambda_tilde(x, 0.05, "nstar", n_star=500)
    for n in (10, 100, 500, 2000, 2999):
        lo, hi = hc_wor_cs_at(x[:n], 3000, 0.05, lam, 0.75)
        s = x[:n].sum()
        assert s / 3000 - 1e-12 <= lo <= hi <= (s + 3000 - n) / 3000 + 1e-12


def test_hc_cs_time_uniform_coverage_mc():
    """Monte Carlo: the running CS misses the true pool mean at SOME n in {checkpoints} w.p. <= alpha (+ MC slack)."""
    rng = np.random.default_rng(1)
    N, alpha = 400, 0.2
    pool = np.zeros(N)
    pool[:60] = 1.0
    mu = pool.mean()
    ns = [5, 10, 20, 40, 80, 160, 320, 399]
    for sched, kw in (("prpl", {}), ("nstar", {"n_star": 80})):
        miss = 0
        R = 300
        for _ in range(R):
            x = rng.permutation(pool)
            lam = hc_lambda_tilde(x, alpha, sched, **kw)
            bad = False
            for n in ns:
                lo, hi = hc_wor_cs_at(x[:n], N, alpha, lam, 0.75)
                bad |= not (lo <= mu + 1e-12 and mu - 1e-12 <= hi)
            miss += bad
        assert miss / R <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / R), (sched, miss)


def test_hc_tuned_narrower_than_wsr20_near_target():
    """At the target count the tuned betting CS is not wider than the untuned WSR20 PrPl-EB CS (sanity of tuning)."""
    from dsswm.stats.fp_eb import wor_mean_cs
    rng = np.random.default_rng(2)
    N = 200_000
    pool = (rng.random(N) < 0.05).astype(float)
    x = rng.permutation(pool)
    n = 40_000
    a = 0.05 / 18
    lam = hc_lambda_tilde(x, a, "nstar", n_star=n)
    lo, hi = hc_wor_cs_at(x[:n], N, a, lam, 0.75)
    wl, wh = wor_mean_cs(x[:n], N, a)
    assert lo <= pool.mean() <= hi
    assert hi - lo <= (wh[-1] - wl[-1]) * 1.0001


# --------------------------------------------------------------------------------------------- harness
def test_run_stream_v6_identical_for_non_hc_methods():
    env = _toy_population(903)
    for mk in (lambda: B4Bal(0.5),):
        s1, r1 = run_stream(env, mk(), 7, PROBS, TOY_EPS)
        s2, r2 = run_stream_v6(env, mk(), 7, PROBS, TOY_EPS)
        for k in s1:
            if not k.startswith("sec"):
                assert s1[k] == s2[k], k


def test_hc_patch_restored_and_same_design_digest():
    env = _toy_population(904)
    orig = frr.StreamEngine
    s_hc, _ = run_stream_v6(env, HCWoRRect("prpl", 0.5, None), 11, PROBS, TOY_EPS)
    assert frr.StreamEngine is orig
    s_b4, _ = run_stream(env, B4Bal(0.5), 11, PROBS, TOY_EPS)
    for m in (RectCkHG(), RectCkHGLive(), RectCkBern()):
        s, _ = run_stream_v6(env, m, 11, PROBS, TOY_EPS)
        assert s["schedule_digest"] == s_b4["schedule_digest"] == s_hc["schedule_digest"]
        assert s["billing_ok"]


@pytest.mark.parametrize("mk", [RectCkHG, RectCkHGLive, RectCkBern, lambda: HCWoRRect(**HC_GRID[3]),
                                lambda: HCWoRRect(**HC_GRID[0])])
def test_v6_rivals_toy_no_false_cert_and_nonvacuous(mk):
    ev = certs = early = 0
    for sd in range(905, 909):
        env = _toy_population(sd)
        for p in range(5):
            s, _ = run_stream_v6(env, mk(), 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
            ev += int(s["fwer_event"])
            certs += s["n_cert"]
            early += int(s["reached_stop"] and s["N80"] < 0.5 * env.tau_R)
            assert s["billing_ok"]
    assert ev == 0 and certs > 0 and early >= 5


def test_known_answer_toy():
    env = SyntheticPoolEnv([[4000, 4000], [3000, 3000], [3000, 3000]], [[0.1, 0.6], [0.2, 0.45], [0.3, 0.9]], seed=0,
                           n_min=200)
    for m in (RectCkHG(), RectCkHGLive(), RectCkBern(), HCWoRRect(**HC_GRID[3])):
        s, _ = run_stream_v6(env, m, 42, PROBS, 0.02)
        assert s["reached_stop"] and s["n_false"] == 0 and s["N80"] < env.tau_R


def test_continuation_prefix_matches_stop_at_12():
    env = _toy_population(910)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    s1, r1 = run_stream_v6(env, B4Bal(0.5), 5, PROBS, TOY_EPS, ctx=ctx)
    s2, r2 = run_stream_v6(env, B4Bal(0.5), 5, PROBS, TOY_EPS, ctx=ctx_with_stop(ctx, ctx.Q))
    assert ctx.stop_k != ctx.Q
    for a, b in zip(r1, r2):
        assert a["n_cert"] == b["n_cert"] and a["new"] == b["new"]
    assert [r["n_cert"] for r in r2][len(r1) - 1] >= ctx.stop_k if s1["reached_stop"] else True


# --------------------------------------------------------------------------------------------- Lenta LR9
def test_lr9_structure_and_eval_guard():
    from dsswm.envs.base import TruthAccessError
    from dsswm.envs.lenta_v6 import LentaLayerEnv, lr9_problems
    dev = LentaLayerEnv("dev")
    ev = LentaLayerEnv("eval")
    assert dev.S == ev.S == 9 and dev.A == ev.A == 2
    assert abs(dev.N - ev.N) <= 9 and dev.N + ev.N == 687029
    assert ev.structure_only and not dev.structure_only
    with pytest.raises(TruthAccessError):
        ev.true_mu()
    with pytest.raises(TruthAccessError):
        ev.outcomes_view()
    with pytest.raises(ValueError):
        LentaLayerEnv("full")
    assert len(lr9_problems()) == 15
    assert (dev.pool_sizes > 1000).all() and (ev.pool_sizes > 1000).all()


# --------------------------------------------------------------------------------------------- prereg v6
def _fake_addendum(v5):
    add = {"version": "5+v6-addendum", "status": "locked", "addendum_to": {"version": 5, "sha256": v5["sha256"]},
           "v5_verdict_statement": "x", "blocks": {}, "seed_manifest": {}, "replica_check": {},
           "eval_tasks": {"v6a_full": {}}, "frozen_configs": {"CR9": {}}, "input_sha256": {"plan/prereg_lock.json":
           prereg._sha_file(prereg.LOCK_PATH)}, "code_sha256": {"run_r5s_v6.py": "0" * 64}, "git_commit": "a" * 40,
           "data_sha256": {n: {"path": p, "sha256": "0" * 64} for n, p in DATA_FILES.items()}}
    add["sha256"] = prereg.canonical_hash(add)
    return add


def test_prereg_v6_gate_logic():
    v5 = prereg.load_lock(prereg.lock_path(5))
    add = _fake_addendum(v5)
    assert prereg_v6.check_addendum(add, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False) is add
    with pytest.raises(RuntimeError):
        prereg_v6.check_addendum(add, "v6x_unknown", v5_lock=v5, verify_code=False, verify_data=False)
    bad = copy.deepcopy(add)
    bad["status"] = "draft"
    bad["sha256"] = prereg.canonical_hash(bad)
    with pytest.raises(RuntimeError):
        prereg_v6.check_addendum(bad, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False)
    bad = copy.deepcopy(add)
    bad["addendum_to"]["sha256"] = "0" * 64
    bad["sha256"] = prereg.canonical_hash(bad)
    with pytest.raises(RuntimeError):
        prereg_v6.check_addendum(bad, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False)
    tampered = copy.deepcopy(add)
    tampered["eval_tasks"]["v6b_full"] = {}
    with pytest.raises(RuntimeError):
        prereg_v6.check_addendum(tampered, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False)


def test_draft_never_authorises():
    ok, why = prereg_v6.addendum_gate("v6a_full", path=prereg_v6.DRAFT_PATH, verify_code=False)
    assert not ok


# --------------------------------------------------------------------------------------------- analysis rules
def _pr(r):
    return {"ub95_one_sided": r}


def test_decision_rules():
    from dsswm.stats.v6_analysis import A_FAMILY, V5_R
    ok = {"status": "pass"}
    fam = {m: _pr(0.5) for m in A_FAMILY}
    assert decide_block_a(fam, ok)["verdict"] == "keep"
    assert decide_block_a({**fam, "HC-WoR": _pr(0.70)}, ok)["verdict"] == "downgrade"
    assert decide_block_a({**fam, "HC-WoR": _pr(0.99)}, ok)["verdict"] == "downgrade"
    assert decide_block_a({**fam, "HC-WoR": _pr(1.0)}, ok)["verdict"] == "withdraw"
    assert decide_block_a(fam, {"status": "fail"})["verdict"] == "withdraw"          # replica failure forced
    c = {m: _pr(0.6) for m in V5_R}
    assert decide_block_c(c, ok)["verdict"] == "n100_pass"
    assert decide_block_c({**c, "B1": _pr(1.01)}, ok)["verdict"] == "narrow_to_N80"
    assert decide_block_c(c, {"status": "fail"})["verdict"] == "narrow_to_N80"


@pytest.mark.parametrize("bad", ["empty", "partial", "extra", "nan", "noreplica", "pending"])
def test_decision_rules_reject_incomplete(bad):
    from dsswm.stats.v6_analysis import A_FAMILY, V5_R
    fam = {m: _pr(0.5) for m in A_FAMILY}
    c = {m: _pr(0.5) for m in V5_R}
    rep = {"status": "pass"}
    if bad == "empty":
        fam, c = {}, {}
    elif bad == "partial":
        fam = {"RECT-ck-HG": _pr(0.5)}
        c = {"B1": _pr(0.5)}
    elif bad == "extra":
        fam = {**fam, "X": _pr(0.5)}
        c = {**c, "X": _pr(0.5)}
    elif bad == "nan":
        fam = {**fam, "HC-WoR": _pr(float("nan"))}
        c = {**c, "B1": _pr(float("nan"))}
    elif bad == "noreplica":
        rep = None
    else:
        rep = {"status": None}
    with pytest.raises(ValueError):
        decide_block_a(fam, rep)
    with pytest.raises(ValueError):
        decide_block_c(c, rep)


def _rows(methods, seeds, val=100.0, eps=0.001, key="N80_pen"):
    return [{"method": m, "seed": s, "eps": eps, key: val} for m in methods for s in seeds]


@pytest.mark.parametrize("bad", ["missing", "dup", "nan", "zero", "extra_method", "extra_seed"])
def test_build_matrix_rejects(bad):
    from dsswm.stats.v6_analysis import build_matrix
    rows = _rows(["FDC", "B4-bal"], range(5))
    if bad == "missing":
        rows = rows[:-1]
    elif bad == "dup":
        rows = rows + rows[:1]
    elif bad == "nan":
        rows[0]["N80_pen"] = float("nan")
    elif bad == "zero":
        rows[0]["N80_pen"] = 0
    elif bad == "extra_method":
        rows += _rows(["X"], range(5))
    else:
        rows += _rows(["FDC"], [99])
    with pytest.raises(ValueError):
        build_matrix(rows, ["FDC", "B4-bal"], range(5), [0.001], "N80_pen")


def test_analyse_block_end_to_end_dev_spec():
    from dsswm.stats.v6_analysis import A_FAMILY, analyse_block
    ms = ("FDC", "B4-bal") + A_FAMILY
    rows = _rows(["FDC"], range(20), 50.0) + _rows(ms[1:], range(20), 100.0)
    spec = {"methods": ms, "family": A_FAMILY, "seeds": tuple(range(20)), "eps": (0.001,), "key": "N80_pen"}
    out = analyse_block("A", rows, {"status": "pass"}, spec=spec, B=500)
    assert out["decision"]["verdict"] == "keep"
    with pytest.raises(ValueError):
        analyse_block("A", rows[:-1], {"status": "pass"}, spec=spec, B=500)


# --------------------------------------------------------------------------------------------- replica rules
def test_replica_rules():
    from dsswm.stats import v6_replica as R
    base = [{"method": m, "seed": s, "eps": 0.001, "N80_pen": 10, "sec": 1.0, "schedule_digest": "d"}
            for m in ("FDC", "B4-bal") for s in range(3)]
    rep = [dict(r, sec=9.0) for r in base]
    assert R.check_r1(base, rep, ["FDC", "B4-bal"], [0.001], range(3))["pass"]      # timing excluded
    rep[0]["N80_pen"] = 11
    assert not R.check_r1(base, rep, ["FDC", "B4-bal"], [0.001], range(3))["pass"]
    assert not R.check_r1(base, rep[1:], ["FDC", "B4-bal"], [0.001], range(3))["pass"]
    assert R.check_r2(base, ("FDC", "B4-bal"))["pass"]
    bad = [dict(r) for r in base]
    bad[0]["schedule_digest"] = "e"
    assert not R.check_r2(bad, ("FDC", "B4-bal"))["pass"]
    assert not R.check_r2(base, ("FDC", "B4-bal", "HC-WoR"))["pass"]                   # group member missing
    assert "B4-bal" not in R.COMPARISON_GROUPS["v6b_full_b"]


def test_replica_r3_mapping():
    from dsswm.stats import v6_replica as R
    v5 = [{"method": "B1", "seed": 1, "N80_pen": 100, "completed": True, "rs_k_stop": 2, "n_cert_curve": [0, 5, 12],
           "cert_k": [1, 2, -1]}]
    fw = [{"method": "FDC", "seed": 1, "N100_pen": 50}]
    v6 = [{"method": "B1", "seed": 1, "N80_pen": 100, "k80": 2, "n_cert_curve": [0, 5, 12, 15], "cert_k": [1, 2, 3]},
          {"method": "FDC", "seed": 1, "N100_pen": 50}]
    assert R.check_r3(v6, v5, fw, ["B1"], [1])["pass"]
    v6b = [dict(v6[0], cert_k=[1, 2, 2]), v6[1]]
    assert not R.check_r3(v6b, v5, fw, ["B1"], [1])["pass"]
    v6c = [v6[0], dict(v6[1], N100_pen=51)]
    assert not R.check_r3(v6c, v5, fw, ["B1"], [1])["pass"]


# --------------------------------------------------------------------------------------------- HC fix / HG-live
def test_hc_reviewer_counterexample_lower_root():
    x = np.r_[np.ones(100), np.zeros(100)]
    a = 0.05 / 18
    lam = hc_lambda_tilde(x, a, "nstar", n_star=200)
    from dsswm.baselines.wor_betting_v6 import hc_wor_cs_interval
    lo, hi, empty = hc_wor_cs_interval(x, 1000, a, lam, 0.75)
    assert not empty and lo == pytest.approx(0.552007, abs=1e-5) and lo > 0.5


def test_hc_empty_intersection_is_conservative():
    from dsswm.baselines.wor_betting_v6 import hc_wor_cs_interval
    x = np.ones(100)
    lam = hc_lambda_tilde(x, 0.05, "prpl")
    lo, hi, empty = hc_wor_cs_interval(x, 1000, 0.05, lam, 0.75, lo0=0.0, hi0=0.11)
    assert empty and (lo, hi) == (pytest.approx(0.1), pytest.approx(1.0))


def test_hg_live_weakly_dominates_and_spends_delta():
    from dsswm.baselines.rect_v6 import RectCkHGLive
    env = _toy_population(906)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    m = RectCkHGLive()
    m.setup(ctx)
    K, SA = len(ctx.checkpoints), ctx.S * ctx.A

    class St:
        pass
    st = St()
    st.N = np.asarray(env.pool_sizes)
    st.n = st.N // 2
    st.sum = st.n * 0.3
    m._bounds(ctx, st)
    assert m.last_alpha_side == pytest.approx(0.05 / (2 * SA * K))      # all live -> identical to RECT-ck-HG
    st.n = st.N.copy()
    st.n[0, 0] = st.N[0, 0] // 2
    st.sum = np.floor(st.n * 0.3)
    m._bounds(ctx, st)
    assert m.last_alpha_side == pytest.approx(0.05 / (2 * 1 * K))      # one live cell gets the whole delta_k
    lo, hi = m._bounds(ctx, st)
    exh = st.n >= st.N
    assert np.allclose(lo[exh], hi[exh])


def test_lr9_eval_cannot_be_unlocked_without_addendum():
    from dsswm.envs.base import TruthAccessError
    from dsswm.envs.lenta_v6 import LentaLayerEnv
    with pytest.raises(TruthAccessError):
        LentaLayerEnv("eval", eval_task_id="v6b_full_b")
    with pytest.raises(TypeError):
        LentaLayerEnv("eval", allow_eval=True)


def test_prereg_v6_rejects_empty_code_and_input_drift(tmp_path):
    v5 = prereg.load_lock(prereg.lock_path(5))
    add = _fake_addendum(v5)
    add["code_sha256"] = {}
    add["sha256"] = prereg.canonical_hash(add)
    with pytest.raises(RuntimeError):
        prereg_v6.check_addendum(add, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False)
    add = _fake_addendum(v5)
    add["input_sha256"] = {"plan/prereg_lock.json": "0" * 64}
    add["sha256"] = prereg.canonical_hash(add)
    with pytest.raises(RuntimeError):
        prereg_v6.check_addendum(add, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False)


def test_finalize_refuses_overwrite_and_bad_commit(tmp_path):
    out = tmp_path / "lock.json"
    out.write_text("{}")
    with pytest.raises(RuntimeError, match="never overwritten"):
        prereg_v6.finalize_addendum("a" * 40, out_path=out)
    v5 = prereg.load_lock(prereg.lock_path(5))
    with pytest.raises(RuntimeError):
        prereg_v6.commit_code_mismatch({"code_sha256": {"run_r5s_v6.py": "0" * 64}}, "f" * 40)
    head = __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=str(prereg.CODE_ROOT)).stdout.strip()
    bad = prereg_v6.commit_code_mismatch({"code_sha256": {"run_r5s_v6.py": "0" * 64}}, head)
    assert bad == ["run_r5s_v6.py"]
    assert v5["version"] == 5



# --------------------------------------------------------------------------------------------- rev. 3 (external reviewer re-check)
_FULL = prereg.WS_ROOT / "exp/results/full"


def _v5_record(fname, method, seed):
    import json
    for l in (_FULL / fname / "results.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["method"] == method and int(r["seed"]) == seed:
            return r
    raise LookupError((fname, method, seed))


def _continuation(b):
    from dsswm.stats.v6_replica import v5_k_stop
    ks = v5_k_stop(b)
    c5 = b["n_cert_curve"]
    k_cut = ks if ks is not None else len(c5) - 1
    pre = list(c5[:k_cut + 1])
    tail = []
    last = pre[-1]
    for _ in range(20 - len(pre)):
        last = min(15, last + 1)
        tail.append(last)
    cert = [k if k >= 0 else min(19, k_cut + 1) for k in b["cert_k"]]
    return {"method": b["method"], "seed": int(b["seed"]), "N80_pen": b["N80_pen"], "k80": ks,
            "n_cert_curve": pre + tail, "cert_k": cert}


@pytest.mark.parametrize("fname,method,seed", [("r5_cr_main_a", "Peace-rect", 30037),
                                               ("r5_cr_main_b", "Peace-rect", 30112),
                                               ("r5_cr_main_b", "Peace-rect", 30137),
                                               ("r5_cr_main_b", "B4-bal", 30150)])
def test_r3_real_v5_records(fname, method, seed):
    """Real v5 records of both schemas (top-level rs_k_stop / nested run_stream.k_stop); a legal continuation that
    certifies more problems after the v5 stop must pass even though the v5 curve is padded after the stop."""
    from dsswm.stats.v6_replica import check_r3, v5_k_stop
    b = _v5_record(fname, method, seed)
    if fname == "r5_cr_main_b":
        assert "rs_k_stop" not in b and "k_stop" in b["run_stream"]
    else:
        assert "rs_k_stop" in b
    a = _continuation(b)
    ks = v5_k_stop(b)
    if ks is not None and ks < 19 and b["n_cert_curve"][ks] < 15:
        assert a["n_cert_curve"] != b["n_cert_curve"]           # the padded v5 tail differs from a real continuation
    assert check_r3([a], [b], [], [method], [seed])["pass"]
    bad = dict(a, n_cert_curve=[a["n_cert_curve"][0] + 1] + a["n_cert_curve"][1:])
    assert not check_r3([bad], [b], [], [method], [seed])["pass"]
    bad = dict(a, N80_pen=a["N80_pen"] + 1)
    assert not check_r3([bad], [b], [], [method], [seed])["pass"]


_A_GROUP = ("FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR")


def _task_rows(methods=_A_GROUP, seeds=range(12), eps=(0.001,)):
    return [{"method": m, "seed": s, "eps": e, "N80_pen": 100 + s, "schedule_digest": f"d{s}", "sec": 0.1}
            for m in methods for e in eps for s in seeds]


_M, _E, _S = _A_GROUP, (0.001,), tuple(range(12))


def _replica_of(rows):
    return [dict(r, sec=9.9) for r in rows if r["seed"] < 10]


def _kw(rows, rep_rows, **over):
    from dsswm.stats.v6_replica import content_hash
    kw = dict(task="v6a_full", is_eval=True, current_rows=rows, replica_rows=rep_rows, methods=_M, eps=_E, seeds=_S,
              add_sha="f" * 64, code_sha256={"x": "1"}, data_sha256="dd",
              sealed_content_sha256=content_hash(rows))      # stands for a verified git seal of these rows
    kw.update(over)
    return kw


def _report(rows, rep_rows=None, task="v6a_full", is_eval=True, add="f" * 64):
    from dsswm.stats import v6_replica as R
    rep_rows = _replica_of(rows) if rep_rows is None else rep_rows
    return R.build_report(task, is_eval, rows, rep_rows, _M, _E, _S, add, {"x": "1"}, "dd")


def test_replica_report_refused_when_incomplete_or_unbound():
    rows = _task_rows()
    with pytest.raises(ValueError):
        _report(rows[:-1])                                       # main matrix incomplete
    with pytest.raises(ValueError):
        _report(rows + rows[:1])                                 # duplicate
    with pytest.raises(ValueError):
        _report(rows, rep_rows=_replica_of(rows)[:-1])           # replica matrix incomplete
    with pytest.raises(ValueError):
        _report(rows, add=None)                                  # eval without addendum sha
    with pytest.raises(ValueError):
        _report(rows, task="v6c_full_b")                         # C eval needs the v5 references for R3


@pytest.mark.parametrize("bad", ["wrong_task", "none_sha", "empty_checks", "appended", "replaced", "data", "code",
                                 "seeds", "pass_with_fail", "missing"])
def test_validate_report_refuses_stale_or_unbound(bad):
    import copy as _c
    from dsswm.stats import v6_replica as R
    rows = _task_rows()
    rep_rows = _replica_of(rows)
    rep = _report(rows)
    kw = _kw(rows, rep_rows)
    assert R.validate_report(rep, **kw) == "pass"
    rep2 = _c.deepcopy(rep)
    if bad == "wrong_task":
        rep2["task_id"] = "v6c_full_b"
    elif bad == "none_sha":
        rep2["addendum_sha256"] = None
    elif bad == "empty_checks":
        rep2["checks"] = []
    elif bad == "appended":
        kw["current_rows"] = rows + [dict(rows[0], seed=99)]
    elif bad == "replaced":
        kw["current_rows"] = [dict(rows[0], N80_pen=1)] + rows[1:]
    elif bad == "data":
        kw["data_sha256"] = "ee"
    elif bad == "code":
        kw["code_sha256"] = {"x": "2"}
    elif bad == "seeds":
        kw["seeds"] = tuple(range(13))
    elif bad == "pass_with_fail":
        rep2["checks"][0]["pass"] = False
    else:
        rep2 = None
    with pytest.raises(ValueError):
        R.validate_report(rep2, **kw)


def test_reviewer_counterexample_edited_report_cannot_flip_verdict():
    """(a) external reviewer final check: R2 genuinely fails (one schedule mismatch); editing BOTH the report status and the R2
    'pass' flag (keeping a non-empty 'bad') must be refused, and the honest report must yield 'fail' -> withdraw."""
    import copy as _c
    from dsswm.stats import v6_replica as R
    from dsswm.stats.v6_analysis import A_FAMILY
    rows = _task_rows()
    rows[3] = dict(rows[3], schedule_digest="MISMATCH")       # FDC seed 3 differs from B4-bal seed 3
    rep_rows = _replica_of(rows)
    honest = _report(rows, rep_rows)
    r2 = [c for c in honest["checks"] if c["rule"] == "R2"][0]
    assert honest["status"] == "fail" and not r2["pass"] and r2["bad"]
    assert R.validate_report(honest, **_kw(rows, rep_rows)) == "fail"
    assert decide_block_a({m: _pr(0.5) for m in A_FAMILY}, {"status": "fail"})["verdict"] == "withdraw"
    forged = _c.deepcopy(honest)
    forged["status"] = "pass"
    for c in forged["checks"]:
        if c["rule"] == "R2":
            c["pass"] = True                                   # 'bad' left non-empty, exactly as in the counterexample
    with pytest.raises(ValueError):
        R.validate_report(forged, **_kw(rows, rep_rows))


def test_tampered_replica_rows_refused():
    """(b) a genuine 'pass' report no longer validates once the replica rows behind it are altered (even if the
    alteration would itself make R1 fail)."""
    from dsswm.stats import v6_replica as R
    rows = _task_rows()
    rep_rows = _replica_of(rows)
    rep = _report(rows, rep_rows)
    assert rep["status"] == "pass"
    tampered = [dict(rep_rows[0], N80_pen=12345)] + rep_rows[1:]
    with pytest.raises(ValueError):
        R.validate_report(rep, **_kw(rows, tampered))
    with pytest.raises(ValueError):
        R.validate_report(rep, **_kw(rows, rep_rows[:-1]))     # replica row dropped


def test_honest_pass_still_passes():
    """(c) honest report: validate_report returns the RECOMPUTED 'pass'."""
    from dsswm.stats import v6_replica as R
    rows = _task_rows()
    rep_rows = _replica_of(rows)
    assert R.validate_report(_report(rows, rep_rows), **_kw(rows, rep_rows)) == "pass"


def test_r3_recomputed_with_v5_references():
    from dsswm.stats import v6_replica as R
    b = _v5_record("r5_cr_main_a", "Peace-rect", 30037)
    a = _continuation(b)
    a.update({"eps": 0.001, "schedule_digest": "p", "sec": 1.0})
    fdc = {"method": "FDC", "seed": 30037, "eps": 0.001, "N100_pen": 7, "schedule_digest": "p", "N80_pen": 5}
    fw = [{"method": "FDC", "seed": 30037, "N100_pen": 7}]
    rows = [a, fdc]
    seeds = (30037,)
    rep = R.build_report("v6c_full_a", True, rows, rows, ("Peace-rect", "FDC"), (0.001,), seeds, "f" * 64, {}, "d",
                         [b], fw)
    kw = dict(task="v6c_full_a", is_eval=True, current_rows=rows, replica_rows=rows, methods=("Peace-rect", "FDC"),
              eps=(0.001,), seeds=seeds, add_sha="f" * 64, code_sha256={}, data_sha256="d",
              sealed_content_sha256=R.content_hash(rows))
    assert R.validate_report(rep, v5_main=[b], v5_fwer=fw, **kw) == "pass"
    fw_bad = [{"method": "FDC", "seed": 30037, "N100_pen": 8}]
    with pytest.raises(ValueError):                             # v5 reference changed -> report does not reconcile
        R.validate_report(rep, v5_main=[b], v5_fwer=fw_bad, **kw)


def test_r2_requires_all_planned_streams():
    from dsswm.stats.v6_replica import check_r2
    rows = _task_rows(("FDC", "B4-bal"), seeds=range(10))
    assert check_r2(rows, ("FDC", "B4-bal"), range(10), (0.001,))["pass"]
    assert not check_r2(rows, ("FDC", "B4-bal"), range(200), (0.001,))["pass"]


def test_data_binding(tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"abc")
    frozen = {"x": {"path": str(f), "sha256": file_sha256(f)}}
    assert data_drift(frozen) == []
    f.write_bytes(b"abcd")
    assert data_drift(frozen) == ["x"]
    assert data_drift({"y": {"path": str(tmp_path / "nope"), "sha256": "0"}}) == ["y (missing)"]


def test_addendum_refuses_wrong_raw_data_hash():
    v5 = prereg.load_lock(prereg.lock_path(5))
    add = _fake_addendum(v5)
    add["sha256"] = prereg.canonical_hash(add)
    with pytest.raises(RuntimeError, match="raw data"):
        prereg_v6.check_addendum(add, "v6a_full", v5_lock=v5, verify_code=False)
    add2 = _fake_addendum(v5)
    add2["data_sha256"] = {}
    add2["sha256"] = prereg.canonical_hash(add2)
    with pytest.raises(RuntimeError, match="data_sha256"):
        prereg_v6.check_addendum(add2, "v6a_full", v5_lock=v5, verify_code=False, verify_data=False)


# --------------------------------------------------------------------------------------------- rev. 5: eval seals
@pytest.fixture
def git_repo(tmp_path):
    import subprocess as sp
    root = tmp_path / "repo"
    (root / "seals").mkdir(parents=True)
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["commit", "-q", "--allow-empty", "-m", "init"]):
        sp.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def _sealed(git_repo, rows):
    from dsswm.stats.v6_seal import make_seal, write_and_commit_seal
    seal = make_seal("v6a_full", rows, _M, _E, _S, {"x": "1"}, "dd", "f" * 64)
    return write_and_commit_seal(seal, git_repo / "seals")


def _vs(git_repo, rows):
    from dsswm.stats.v6_seal import verify_seal
    return verify_seal(git_repo / "seals", "v6a_full", rows, _M, _E, _S, {"x": "1"}, "dd", "f" * 64)


def test_r2_runs_on_full_matrix_beyond_r1_seeds():
    from dsswm.stats import v6_replica as R
    rows = _task_rows()
    rows[10] = dict(rows[10], schedule_digest="MISMATCH")      # FDC seed 10: outside the R1 seeds 0-9
    checks = R.compute_checks("v6a_full", True, rows, _replica_of(rows), _M, _E, _S)
    r1 = [c for c in checks if c["rule"] == "R1"][0]
    r2 = [c for c in checks if c["rule"] == "R2"][0]
    assert r1["pass"] and not r2["pass"] and r2["n_streams"] == len(_S)


def test_sealed_honest_run_passes(git_repo):
    from dsswm.stats import v6_replica as R
    rows = _task_rows()
    sha = _sealed(git_repo, rows)
    info = _vs(git_repo, rows)
    assert info["seal_commit"] == sha
    rep_rows = _replica_of(rows)
    assert R.validate_report(_report(rows, rep_rows), **_kw(
        rows, rep_rows, sealed_content_sha256=info["seal"]["results_content_sha256"])) == "pass"


def test_reviewer_confirm_counterexample_edited_main_row_and_report_refused(git_repo):
    """Original run: FDC seed 10 (outside R1) has a schedule mismatch -> R2 fails -> withdraw.  The attacker edits that
    main row AND rebuilds the report (checks, status, results_content_sha256) so that everything reconciles: the
    current rows no longer match the git-committed seal -> refused."""
    from dsswm.stats import v6_replica as R
    rows = _task_rows()
    rows[10] = dict(rows[10], schedule_digest="MISMATCH")
    _sealed(git_repo, rows)
    rep_rows = _replica_of(rows)
    honest = _report(rows, rep_rows)
    assert honest["status"] == "fail"
    sealed = _vs(git_repo, rows)["seal"]["results_content_sha256"]
    assert R.validate_report(honest, **_kw(rows, rep_rows, sealed_content_sha256=sealed)) == "fail"
    edited = [dict(r) for r in rows]
    edited[10]["schedule_digest"] = "d10"                      # restore agreement with the other methods
    forged = _report(edited, rep_rows)                         # fully self-consistent forged report
    assert forged["status"] == "pass"
    with pytest.raises(ValueError):
        R.validate_report(forged, **_kw(edited, rep_rows, sealed_content_sha256=sealed))
    with pytest.raises(ValueError):
        _vs(git_repo, edited)
    with pytest.raises(ValueError):                            # eval without any seal is refused too
        R.validate_report(forged, **_kw(edited, rep_rows, sealed_content_sha256=None))


def test_seal_edited_after_commit_refused(git_repo):
    import json as _j
    rows = _task_rows()
    _sealed(git_repo, rows)
    edited = [dict(r, N80_pen=r["N80_pen"] + 1) if r["seed"] == 11 else r for r in rows]
    sp = git_repo / "seals" / "v6a_full.seal.json"
    seal = _j.loads(sp.read_text())
    from dsswm.stats.v6_replica import content_hash
    seal["results_content_sha256"] = content_hash(edited)      # attacker re-points the seal at edited rows
    sp.write_text(_j.dumps(seal, indent=1, sort_keys=True))
    with pytest.raises(ValueError, match="edited after sealing"):
        _vs(git_repo, edited)


def test_seal_commit_must_exist_in_history(git_repo):
    import json as _j
    rows = _task_rows()
    _sealed(git_repo, rows)
    rp = git_repo / "seals" / "v6a_full.seal_ref.json"
    ref = _j.loads(rp.read_text())
    ref["seal_commit"] = "0" * 40
    rp.write_text(_j.dumps(ref))
    with pytest.raises(ValueError, match="does not exist"):
        _vs(git_repo, rows)


def test_refuse_to_seal_incomplete(git_repo):
    from dsswm.stats.v6_seal import make_seal
    with pytest.raises(ValueError):
        make_seal("v6a_full", _task_rows()[:-1], _M, _E, _S, {}, "d", "f" * 64)
