"""Round-4 real layer: pool replay (envs/pool_replay.py) + problem definitions (streams/frontier.py)."""
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsswm.envs.base import TruthAccessError
from dsswm.envs.pool_replay import (PoolReplayEnv, ReplayStream, Schedule, load_table, make_schedule,
                                    occurrence_rank, replay_fixed, sha256_file, DATA_PATHS)
from dsswm.streams import frontier as fr

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def hr6_dev():
    return PoolReplayEnv("HR6", "dev")


@pytest.fixture(scope="module")
def lr8_dev():
    return PoolReplayEnv("LR8", "dev")


# ------------------------------------------------------------------ data / pools
def test_datasets_are_real_and_untouched():
    import json
    for name, p in DATA_PATHS.items():
        prov = json.loads((p.parent / "PROVENANCE.json").read_text())
        assert prov["fallback"] is False
        if "tidy_sha256" in prov:
            assert sha256_file(p) == prov["tidy_sha256"]


@pytest.mark.parametrize("layer", ["HR6", "HR16", "LR8"])
def test_pool_sizes_sum_to_rows(layer):
    full = PoolReplayEnv(layer, "full")
    dev, ev = PoolReplayEnv(layer, "dev"), PoolReplayEnv(layer, "eval")
    n_rows = len(load_table("hillstrom" if layer.startswith("HR") else "lenta"))
    assert full.pool_sizes.sum() == n_rows == full.N == full.tau_R
    assert dev.pool_sizes.sum() == dev.N and ev.pool_sizes.sum() == ev.N
    assert np.array_equal(dev.pool_sizes + ev.pool_sizes, full.pool_sizes)
    assert len(np.intersect1d(dev.row_index, ev.row_index)) == 0
    assert np.array_equal(np.sort(np.r_[dev.row_index, ev.row_index]), np.arange(n_rows))
    assert np.abs(dev.pool_sizes - ev.pool_sizes).max() <= 3   # HR6 pools merge <= 3 S16 strata, each off by <= 1
    assert (full.pool_sizes > 0).all() and np.isclose(full.w.sum(), 1.0)


def test_split_is_stratified_by_finest_segmentation():
    """HR split is stratified by S16 x arm, so every S16 x arm pool differs by at most 1 row between halves."""
    dev, ev = PoolReplayEnv("HR16", "dev"), PoolReplayEnv("HR16", "eval")
    assert np.abs(dev.pool_sizes - ev.pool_sizes).max() <= 1
    dl, el = PoolReplayEnv("LR8", "dev"), PoolReplayEnv("LR8", "eval")
    assert np.abs(dl.pool_sizes - el.pool_sizes).max() <= 1
    # deterministic in the split seed
    assert np.array_equal(PoolReplayEnv("HR6", "dev").row_index, dev.row_index)


def test_reviewer_segment_counts_reproduce():
    """external reviewer round-3 review: dev definition {0},{1,2},{3..6} -> 6 / 16 segments; synthesiser cut [-1,1,3,10] -> 5 / 13.
    S=16 smallest (segment, arm) cell on the full table = 285 rows (methodology s1.1)."""
    df = load_table("hillstrom")
    assert len(np.unique(fr.seg_hr6(df))) == 6
    lab16, pairs = fr.seg_hr16(df)
    assert len(np.unique(lab16)) == 16 and len(pairs) == 16
    assert PoolReplayEnv("HR16", "full").pool_sizes.min() == 285
    h = pd.cut(df.x_history_segment, [-1, 1, 3, 10], labels=False)
    assert df.groupby([h, df.x_newbie]).ngroups == 5
    assert df.groupby([h, df.x_newbie, df.x_mens * 2 + df.x_womens]).ngroups == 13
    assert fr.hs3(np.arange(7)).tolist() == [0, 1, 1, 2, 2, 2, 2]


def test_lenta_s8_and_quartiles():
    df = load_table("lenta")
    edges = fr.lenta_age_edges(df.x_age.to_numpy())
    lab = fr.seg_lr8(df, edges)
    assert len(np.unique(lab)) == 8
    q = lab // 2
    for k in range(3):                                   # right-inclusive quartile bins on the full table
        assert df.x_age.to_numpy()[q == k].max() <= edges[k] < df.x_age.to_numpy()[q == k + 1].min()


# ------------------------------------------------------------------ schedules
@pytest.mark.parametrize("kind", ["pool", "fixed", "adaptive"])
def test_schedule_invariant_to_outcomes(hr6_dev, kind):
    alloc = np.full((hr6_dev.S, hr6_dev.A), 1.0 / 3) if kind == "fixed" else None
    for seed in (900, 901, 977):
        a = make_schedule(hr6_dev, seed, alloc=alloc, adaptive=(kind == "adaptive"))
        b = make_schedule(hr6_dev.with_permuted_outcomes(seed + 1), seed, alloc=alloc, adaptive=(kind == "adaptive"))
        assert a.digest() == b.digest()
        assert not np.array_equal(hr6_dev.outcomes_view("visit"),
                                  hr6_dev.with_permuted_outcomes(seed + 1).outcomes_view("visit"))
    assert make_schedule(hr6_dev, 900).digest() != make_schedule(hr6_dev, 901).digest()


def test_schedule_marginals_pool_proportional(hr6_dev):
    sch = make_schedule(hr6_dev, 42)
    assert np.array_equal(np.bincount(sch.seg_seq, minlength=hr6_dev.S), hr6_dev.seg_sizes)
    cell = sch.seg_seq.astype(int) * hr6_dev.A + sch.arm_seq
    assert np.array_equal(np.bincount(cell, minlength=hr6_dev.S * hr6_dev.A), hr6_dev.pool_sizes.ravel())
    for c, p in enumerate(sch.pool_perm):
        assert np.array_equal(np.sort(p), np.arange(hr6_dev.pool_sizes.ravel()[c]))


def test_lenta_pool_proportional_never_exhausts_before_tau(lr8_dev):
    for seed in (900, 950, 999):
        sch = make_schedule(lr8_dev, seed)
        cell = sch.seg_seq.astype(np.int64) * lr8_dev.A + sch.arm_seq
        rank = occurrence_rank(cell)
        assert (rank < lr8_dev.pool_sizes.ravel()[cell]).all()      # every request finds a record
        n, _, _ = replay_fixed(lr8_dev, sch, "response_att", lr8_dev.checkpoints())
        assert np.array_equal(n[-1], lr8_dev.pool_sizes.ravel())
    ctrl_share = lr8_dev.pool_sizes[:, 0].sum() / lr8_dev.N
    assert 0.2 < ctrl_share < 0.3


def test_occurrence_rank():
    c = np.array([2, 0, 2, 2, 1, 0])
    assert occurrence_rank(c).tolist() == [0, 0, 1, 2, 0, 1]


# ------------------------------------------------------------------ engine: billing / exhaustion
def test_engine_matches_vectorised_replay(hr6_dev):
    sch = make_schedule(hr6_dev, 903)
    ck = hr6_dev.checkpoints()
    n, s1, _ = replay_fixed(hr6_dev, sch, "visit", ck)
    st = ReplayStream(hr6_dev, sch, "visit")
    C = hr6_dev.S * hr6_dev.A
    cnt, tot, k = np.zeros(C, int), np.zeros(C), 0
    while st.t < st.tau_R:
        s, a, y = st.arrive()
        cnt[s * hr6_dev.A + a] += 1
        tot[s * hr6_dev.A + a] += y
        if st.t == ck[k]:
            assert np.array_equal(cnt, n[k]) and np.allclose(tot, s1[k])
            k += 1
    assert k == len(ck) and st.billing_ok() and st.n_skipped == 0 and st.n_reselected == 0
    assert np.allclose(s1[-1] / n[-1], hr6_dev.true_mu("visit").ravel())   # full replay recovers the population


def _run_greedy(env, seed, pref=1):
    sch = make_schedule(env, seed, adaptive=True)
    st = ReplayStream(env, sch, "visit")
    log = []
    while st.t < st.tau_R:
        log.append(st.arrive(lambda s, avail: pref if pref in avail else avail[0]))
    return st, log


def test_exhaustion_rules_deterministic_and_billed(hr6_dev):
    st1, log1 = _run_greedy(hr6_dev, 911)
    st2, log2 = _run_greedy(hr6_dev, 911)
    assert log1 == log2 and st1.billing_ok() and st1.billed == hr6_dev.tau_R
    # always-Mens drains every Mens pool, then re-selects; segment totals equal pool totals so nothing is skipped
    assert (st1.remaining == 0).all() and st1.n_skipped == 0
    first_non_mens = [i for i, (s, a, y) in enumerate(log1) if a != 1]
    assert first_non_mens and all(st1.remaining[:, 1] == 0)


def test_skip_when_segment_exhausted_still_billed(hr6_dev):
    """Hand-built schedule with more segment-0 arrivals than segment-0 rows: overflow arrivals are skipped and billed."""
    env = hr6_dev
    n0 = int(env.seg_sizes[0])
    T = n0 + 50
    sch0 = make_schedule(env, 1, adaptive=True)
    sch = Schedule(1, np.zeros(T, dtype=np.int16), np.full(T, -1, dtype=np.int8), np.zeros(T), sch0.pool_perm,
                   "adaptive")
    st = ReplayStream(env, sch, "visit")
    st.tau_R = T
    out = [st.arrive(lambda s, avail: avail[-1]) for _ in range(T)]
    assert st.n_skipped == 50 and st.billed == T and st.n_served == n0 and st.billing_ok()
    assert all(a is None for _, a, _ in out[n0:])


def test_fixed_allocation_reselect_rule(hr6_dev):
    p = np.zeros((hr6_dev.S, hr6_dev.A))
    p[:, 1] = 1.0                                         # all mass on Mens -> forced re-selection later
    sch = make_schedule(hr6_dev, 5, alloc=p)
    st = ReplayStream(hr6_dev, sch, "visit")
    while st.t < st.tau_R:
        st.arrive()
    assert st.billing_ok() and st.n_reselected > 0 and (st.remaining == 0).all()
    with pytest.raises(RuntimeError):
        pass_through = make_schedule(hr6_dev, 5, alloc=p)
        pass_through.alloc = "pool"
        replay_fixed(hr6_dev, pass_through, "visit", hr6_dev.checkpoints())


# ------------------------------------------------------------------ truth isolation
def test_handle_hides_truth(hr6_dev):
    h = hr6_dev.handle(make_schedule(hr6_dev, 7), "visit")
    for bad in ("_env", "_sch", "_y", "true_mu", "truth_answers", "pool_rows", "_ReplayHandle__stream"):
        with pytest.raises(TruthAccessError):
            getattr(h, bad)
    s, a, y = h.arrive()
    assert h.billed == 1 and h.S == 6 and h.pool_sizes.sum() == hr6_dev.N
    with pytest.raises(TruthAccessError):
        h.t = 0


def test_frontier_is_truth_free():
    tree = ast.parse((PKG / "streams" / "frontier.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            assert not any("envs" in m or "generator" in m for m in mods), mods
    code = ("import sys; sys.path.insert(0, %r); import dsswm.streams.frontier\n" % str(PKG.parent) +
            "assert not [m for m in sys.modules if m.startswith('dsswm.envs')]; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ------------------------------------------------------------------ exact solvers / problems
def test_problem_sets():
    hr, lr = fr.hr_problems(), fr.lr_problems()
    assert len(hr) == 15 and len({p.qid for p in hr}) == 15
    assert len(lr) == 9 and all(p.kappa[0] == 0 for p in lr)


def test_dp_equals_enumeration_random():
    rng = np.random.default_rng(0)
    for _ in range(100):
        S, A = int(rng.integers(2, 7)), int(rng.integers(2, 4))
        n = rng.integers(1, 500, S)
        mu = rng.random((S, A))
        kappa = (0.0,) + tuple(float(x) for x in rng.choice([0.5, 1.0, 1.5, 2.0], A - 1))
        p = fr.Problem("t", "y", kappa, float(rng.choice([0.1, 0.3, 0.5, 0.9, 2.0])))
        ans = fr.solve_enum(n / n.sum(), mu, p)
        J, pi, c = fr.solve_dp(n, mu, p)
        assert abs(J - ans.J_star) < 1e-12 and c <= p.budget + 1e-12
        assert abs(fr.policy_values(np.array([pi]), n / n.sum(), mu)[0] - J) < 1e-12


def test_mitm_equals_dp_and_enum():
    rng = np.random.default_rng(1)
    for _ in range(10):
        S = int(rng.integers(4, 9))
        n = rng.integers(50, 3000, S)
        mu = rng.random((S, 3)) * 0.3
        p = fr.Problem("t", "y", fr.HR_COST_TIERS["C2"], 0.6)
        a = fr.solve_mitm(n, mu, p)
        b = fr.solve_enum(n / n.sum(), mu, p)
        assert abs(a.J_star - b.J_star) < 1e-12 and a.eps_set_size == b.eps_set_size and a.n_feasible == b.n_feasible
        assert abs(fr.solve_dp(n, mu, p)[0] - a.J_star) < 1e-12


def test_truth_answers_hr6_and_trivial_policies(hr6_dev):
    probs = fr.hr_problems()
    ans = hr6_dev.truth_answers(probs)
    mu = hr6_dev.true_mu("visit")
    pooled = hr6_dev.true_pooled_mu("visit")
    for p, a in zip(probs, ans):
        assert a.dp_agrees and a.cost_star <= p.budget + 1e-12
        for pi in (fr.trivial_pooled_greedy(hr6_dev.w, pooled, p), fr.trivial_all_arm(hr6_dev.w, p, 1)):
            c = fr.policy_costs(np.array([pi]), hr6_dev.w, p.kappa)[0]
            v = fr.policy_values(np.array([pi]), hr6_dev.w, mu)[0]
            assert c <= p.budget + 1e-12 and v <= a.J_star + 1e-15


def test_checkpoints():
    ck = fr.checkpoints(1000, 31999)
    assert len(ck) == 20 and ck[0] == 1000 and ck[-1] == 31999 and np.all(np.diff(ck) > 0)
    assert len(fr.checkpoints(5000, 687029)) == 20
