"""r4 pre-lock revision: real Criteo Uplift v2.1 pool-replay layers CR9 / CR12 and the Hillstrom HR8 secondary layer."""
import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from dsswm.envs.pool_replay import (DATA_PATHS, LAYERS, PoolReplayEnv, ReplayStream, load_table, make_schedule,
                                    occurrence_rank, replay_fixed, split_strata)
from dsswm.streams import frontier as fr


@pytest.fixture(scope="module")
def cr9_dev():
    return PoolReplayEnv("CR9", "dev")


@pytest.fixture(scope="module")
def cr9_eval():
    return PoolReplayEnv("CR9", "eval")


@pytest.fixture(scope="module")
def hr8_dev():
    return PoolReplayEnv("HR8", "dev")


# ------------------------------------------------------------------ data / segmentations
def test_criteo_is_real_v21():
    prov = json.loads((DATA_PATHS["criteo"].parent / "PROVENANCE.json").read_text())
    assert prov["fallback"] is False and prov["n_rows"] == 13_979_592
    assert prov["sha256_gz"].startswith("2716e1bf")
    h = hashlib.sha256()
    with open(DATA_PATHS["criteo"].parent / "criteo-research-uplift-v2.1.csv.gz", "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    assert h.hexdigest() == prov["sha256_gz"]
    assert len(load_table("criteo")) == prov["n_rows"]


def test_criteo_segment_definitions():
    df = load_table("criteo")
    modes = fr.criteo_modes(df)
    # f6 / f8 duplicate f0 / f2 after binarisation -> dropped (methodology R1)
    for dup, base in fr.CR_DUPLICATES.items():
        assert np.array_equal(df[dup].to_numpy() != modes[dup], df[base].to_numpy() != modes[base])
    l9, d9 = fr.seg_cr9(df, modes)
    l12, d12, med = fr.seg_cr12(df, modes)
    assert len(d9) == 9 and np.array_equal(np.unique(l9), np.arange(9))
    assert len(d12) == 12 and np.array_equal(np.unique(l12), np.arange(12))
    assert d9[0] == "f0nm=0|f2nm=0|f3nm=0|f9nm=0"
    # every non-mode feature used by CR9 has minority share >= 15% (covariate-only selection rule)
    for f in fr.CR9_FEATURES:
        assert (df[f].to_numpy() != modes[f]).mean() >= 0.15
    # segment labels never depend on outcomes / treatment
    assert set(med) == {"f0", "f2"}


def test_hr8_segments():
    df = load_table("hillstrom")
    lab, pairs = fr.seg_hr8(df)
    assert len(pairs) == 8 and (0, 2) not in pairs and np.array_equal(np.unique(lab), np.arange(8))
    # HR16 refines HR8, so the shared HR split stratification is valid for HR8
    l16, _ = fr.seg_hr16(df)
    assert all(len(np.unique(lab[l16 == k])) == 1 for k in range(16))


@pytest.mark.parametrize("layer", ["CR9", "CR12", "HR8"])
def test_pool_sizes_and_split(layer):
    full, dev, ev = (PoolReplayEnv(layer, h) for h in ("full", "dev", "eval"))
    n_rows = len(load_table(LAYERS[layer]["dataset"]))
    assert full.N == n_rows == full.pool_sizes.sum()
    assert np.array_equal(dev.pool_sizes + ev.pool_sizes, full.pool_sizes)
    assert len(np.intersect1d(dev.row_index, ev.row_index)) == 0 and dev.N + ev.N == n_rows
    assert (dev.pool_sizes > 0).all() and (ev.pool_sizes > 0).all()
    assert dev.S == {"CR9": 9, "CR12": 12, "HR8": 8}[layer] and dev.A == LAYERS[layer]["A"]
    # each pool merges at most a few joint strata, each off by <= 1 row
    assert np.abs(dev.pool_sizes - ev.pool_sizes).max() <= 8


def test_cr_split_stratified_by_joint_cell_and_shared():
    """CR9 and CR12 halves are the same row sets (joint CR12 x CR9 stratification); joint cells differ by <= 1."""
    a, b = PoolReplayEnv("CR9", "dev"), PoolReplayEnv("CR12", "dev")
    assert np.array_equal(a.row_index, b.row_index)
    df = load_table("criteo")
    strata = split_strata("CR9", df)
    arm = df["treatment"].to_numpy().astype(np.int64)
    key = strata * 2 + arm
    full = np.bincount(key)
    dev = np.bincount(key[a.row_index], minlength=len(full))
    assert np.abs(2 * dev - full).max() <= 1
    # HR8 uses the HR split (identical to HR6 / HR16 halves)
    assert np.array_equal(PoolReplayEnv("HR8", "dev").row_index, PoolReplayEnv("HR16", "dev").row_index)


# ------------------------------------------------------------------ schedules / billing
@pytest.mark.parametrize("seed", [900, 901, 902])
def test_cr9_schedule_invariant_to_outcomes(cr9_dev, seed):
    a = make_schedule(cr9_dev, seed)
    b = make_schedule(cr9_dev.with_permuted_outcomes(10_000 + seed), seed)
    assert a.digest() == b.digest()
    assert make_schedule(cr9_dev, seed + 1).digest() != a.digest()


def test_cr9_pool_proportional_counts_and_no_exhaustion(cr9_dev):
    sch = make_schedule(cr9_dev, 900)
    cell = sch.seg_seq.astype(np.int64) * 2 + sch.arm_seq.astype(np.int64)
    sizes = cr9_dev.pool_sizes.ravel()
    assert np.array_equal(np.bincount(cell, minlength=len(sizes)), sizes)          # whole log read once
    assert int((occurrence_rank(cell) >= sizes[cell]).sum()) == 0
    n, _, _ = replay_fixed(cr9_dev, sch, "visit", cr9_dev.checkpoints())
    assert np.array_equal(n[-1], sizes) and (n.sum(1) == cr9_dev.checkpoints()).all()


def test_cr9_engine_matches_vectorised_prefix(cr9_dev):
    ck = cr9_dev.checkpoints()[:4]
    sch = make_schedule(cr9_dev, 905)
    n, s1, _ = replay_fixed(cr9_dev, sch, "visit", cr9_dev.checkpoints())
    st = ReplayStream(cr9_dev, sch, "visit")
    cnt, tot = np.zeros(18, dtype=np.int64), np.zeros(18)
    for k, t in enumerate(ck):
        while st.t < t:
            s, a, y = st.arrive()
            cnt[s * 2 + a] += 1
            tot[s * 2 + a] += y
        assert np.array_equal(cnt, n[k]) and np.allclose(tot, s1[k]) and st.billed == t
    assert st.billing_ok() and st.n_skipped == 0 and st.n_reselected == 0


def test_cr9_adaptive_exhaustion_billing(cr9_dev):
    """Always-control adaptive rule exhausts control pools (~15% of each segment); re-selection + billing stay exact."""
    st = ReplayStream(cr9_dev, make_schedule(cr9_dev, 900, adaptive=True), "visit")
    stop = int(0.2 * cr9_dev.tau_R)
    while st.t < stop:
        st.arrive(lambda s, avail: 0 if 0 in avail else avail[0])
    assert st.billing_ok() and st.billed == stop and st.n_skipped == 0
    assert (st.remaining[:, 0] == 0).any()


def test_hr8_full_replay_and_exhaustion(hr8_dev):
    sch = make_schedule(hr8_dev, 900)
    n, s1, _ = replay_fixed(hr8_dev, sch, "visit", hr8_dev.checkpoints())
    st = ReplayStream(hr8_dev, sch, "visit")
    while st.t < st.tau_R:
        st.arrive()
    assert st.billing_ok() and np.array_equal(n[-1], hr8_dev.pool_sizes.ravel())
    st = ReplayStream(hr8_dev, make_schedule(hr8_dev, 900, adaptive=True), "visit")
    while st.t < st.tau_R:
        st.arrive(lambda s, avail: 1 if 1 in avail else avail[0])
    assert st.billing_ok() and st.billed == hr8_dev.tau_R and (st.remaining[:, 1] == 0).all()


# ------------------------------------------------------------------ problems / truth
def test_cr_problem_set_and_constants():
    ps = fr.cr_problems()
    assert len(ps) == 15 and [p.budget for p in ps] == [round(0.10 + 0.05 * i, 2) for i in range(15)]
    assert all(p.kappa == (0.0, 1.0) and p.outcome == "visit" for p in ps)
    assert len({p.qid for p in ps}) == 15
    assert fr.replan_interval(6_989_793) == 3495 and fr.replan_interval(32_000) == 200
    assert fr.replan_interval(400_000) == 200 and fr.replan_interval(400_001) == 201
    assert LAYERS["CR9"]["n_min"] == 50_000 and LAYERS["CR9"]["eps_grid"] == (0.001, 0.0015, 0.002, 0.003)


def test_cr_checkpoints(cr9_dev):
    ck = cr9_dev.checkpoints()
    assert len(ck) == 20 and ck[0] == 50_000 and ck[-1] == cr9_dev.tau_R and (np.diff(ck) > 0).all()
    assert cr9_dev.replan_interval == max(200, math.ceil(cr9_dev.tau_R / 2000))


def _brute_force(w, mu, p):
    best, arg = -np.inf, None
    vals = []
    for pi in itertools.product(range(mu.shape[1]), repeat=mu.shape[0]):
        cost = sum(w[s] * p.kappa[a] for s, a in enumerate(pi))
        if cost <= p.budget + fr.FEAS_TOL:
            v = sum(w[s] * mu[s, a] for s, a in enumerate(pi))
            vals.append(v)
            if v > best:
                best, arg = v, pi
    return best, arg, len(vals), vals


@pytest.mark.parametrize("half", ["dev", "eval"])
def test_cr9_enum_equals_brute_force(half, cr9_dev, cr9_eval):
    env = cr9_dev if half == "dev" else cr9_eval
    mu = env.true_mu("visit")
    for p, ans in zip(fr.cr_problems(), env.truth_answers(fr.cr_problems())):
        J, pi, nf, vals = _brute_force(env.w, mu, p)
        assert abs(J - ans.J_star) <= 1e-12 and pi == ans.pi_star and nf == ans.n_feasible
        assert ans.dp_agrees
        for e in env.eps_grid:
            assert ans.eps_set_size[e] == sum(v >= J - e for v in vals)


def test_hr8_truth_and_dp(hr8_dev):
    ans = hr8_dev.truth_answers(fr.hr_problems("visit"))
    assert len(ans) == 15 and all(a.dp_agrees for a in ans)


def test_trivial_policies_cr(cr9_dev):
    mu, pooled = cr9_dev.true_mu("visit"), cr9_dev.true_pooled_mu("visit")
    for p in fr.cr_problems():
        a = fr.trivial_all_arm(cr9_dev.w, p, 1)
        b = fr.trivial_pooled_greedy(cr9_dev.w, pooled, p)
        assert fr.policy_costs(np.array([a]), cr9_dev.w, p.kappa)[0] <= p.budget + fr.FEAS_TOL
        if pooled[1] > pooled[0]:          # positive pooled uplift: sign policy == treat-by-w_s-desc
            assert a == b
    assert pooled[1] > pooled[0]


def test_old_layers_unchanged():
    """New layers must not change the HR6 / HR16 / LR8 constants frozen by r4_setup_pool_replay."""
    assert LAYERS["HR6"]["n_min"] == 1000 and LAYERS["LR8"]["n_min"] == 5000
    assert "eps_grid" not in LAYERS["HR6"] and PoolReplayEnv("HR6", "dev").eps_grid == fr.EPS_GRID
    assert fr.EPS_GRID == (0.0025, 0.005, 0.0075, 0.01)
