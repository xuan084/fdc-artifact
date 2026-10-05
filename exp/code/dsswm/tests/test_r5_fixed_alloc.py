"""r5_t0_alloc_audit: regression tests for the fixed-allocation path (make_schedule -> StreamEngine.advance_planned /
_resel -> QFC certificate) used by QFC-half / A-Ney / r5 frozen-share variants.

Covers: asymmetric shares (0.3, 0.7) / (0.7, 0.3) arm-index agreement with the arrival-by-arrival ReplayStream
(column 0 = arm 0 = control, column 1 = arm 1 = treatment); re-selection to the remaining arm after exhaustion; count
conservation on the 50/50 and 0.40 paths; schedule digest invariance to outcome permutation; and the diagnosed
mechanism behind QFC-ney's N80 = tau_R (a zero share starves a cell, n = 0 -> U = +inf until the sibling pool runs dry).
"""
import numpy as np
import pytest

from dsswm.baselines.frontier_common import QFCMethod
from dsswm.baselines.frozen_alloc import neyman_alloc
from dsswm.certify.quadknap import make_stats, qfc_certificate_enum
from dsswm.envs.pool_replay import ReplayStream, make_schedule
from dsswm.streams import frontier as fr
from dsswm.streams.frontier_runner import StreamEngine, SyntheticPoolEnv, build_ctx, run_stream

ASYM = np.array([[0.3, 0.7], [0.7, 0.3], [0.3, 0.7]])
PROBS = [fr.Problem(f"toy|B{b:.2f}", "visit", (0.0, 1.0), b) for b in (0.2, 0.35, 0.5, 0.65, 0.8)]


def _env(sizes=((300, 900), (700, 500), (400, 400)), mu=((0.3, 0.5), (0.4, 0.45), (0.2, 0.6)), seed=0, n_min=20):
    return SyntheticPoolEnv([list(r) for r in sizes], [list(r) for r in mu], seed=seed, n_min=n_min)


def _replay_counts(env, sch, bounds):
    """Arrival-by-arrival reference: (n, sum, billed, served, skipped, reselected) at every bound."""
    ref = ReplayStream(env, sch, "visit")
    n = np.zeros((env.S, env.A), dtype=np.int64)
    sm = np.zeros((env.S, env.A))
    out = []
    for b in bounds:
        while ref.t < b:
            s, a, y = ref.arrive()
            if a is not None:
                n[s, a] += 1
                sm[s, a] += y
        out.append((n.copy(), sm.copy(), ref.billed, ref.n_served, ref.n_skipped, ref.n_reselected))
    assert ref.billing_ok()
    return out


@pytest.mark.parametrize("P", [ASYM, ASYM[:, ::-1].copy()])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_asymmetric_shares_engine_equals_replaystream(P, seed):
    env = _env(seed=seed)
    sch = make_schedule(env, 100 + seed, alloc=P)
    bounds = list(range(37, env.tau_R, 37)) + [env.tau_R]
    ref = _replay_counts(env, sch, bounds)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    for b, (n, sm, billed, served, skipped, resel) in zip(bounds, ref):
        eng.advance_planned(b)
        assert np.array_equal(eng.n, n)
        assert np.allclose(eng.sum, sm)
        assert (eng.billed, eng.n_served, eng.n_skipped, eng.n_reselected) == (billed, served, skipped, resel)
        assert eng.billing_ok()
    assert eng.n_reselected > 0                                 # exhaustion re-selection exercised
    assert np.array_equal(eng.n, env.pool_sizes)                # tau_R = N: whole population served


def test_asymmetric_shares_large_vectorised_path_equals_replaystream():
    env = _env(sizes=((6000, 20000), (15000, 9000)), mu=((0.1, 0.2), (0.3, 0.25)), seed=4, n_min=100)
    P = np.array([[0.3, 0.7], [0.7, 0.3]])
    sch = make_schedule(env, 77, alloc=P)
    bounds = [1000, 7000, 20000, 35000, env.tau_R]
    ref = _replay_counts(env, sch, bounds)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    for b, (n, sm, billed, served, skipped, resel) in zip(bounds, ref):
        eng.advance_planned(b)
        assert np.array_equal(eng.n, n) and np.allclose(eng.sum, sm)
        assert (eng.n_skipped, eng.n_reselected) == (skipped, resel)
    assert eng.n_loop_arrivals < env.tau_R                     # chunked (vectorised) path really used


def test_arm_index_order_column0_is_arm0():
    """Before any exhaustion the realised within-segment share of arm a equals column a of the allocation matrix."""
    env = _env(sizes=((40000, 40000), (40000, 40000)), mu=((0.0, 1.0), (1.0, 0.0)), seed=5, n_min=100)
    P = np.array([[0.3, 0.7], [0.7, 0.3]])
    sch = make_schedule(env, 3, alloc=P)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    eng.advance_planned(40000)                                  # ~20000 per segment: no pool can be dry yet
    assert eng.n_reselected == 0
    share = eng.n / eng.n.sum(1, keepdims=True)
    assert np.allclose(share, P, atol=0.02)
    # outcome identity: cell (0,1) and (1,0) are all-ones, so sums identify which pool each arm index drew from
    assert eng.sum[0, 1] == eng.n[0, 1] and eng.sum[0, 0] == 0
    assert eng.sum[1, 0] == eng.n[1, 0] and eng.sum[1, 1] == 0
    # degenerate shares map to the right column
    for col in (0, 1):
        Pd = np.zeros((2, 2))
        Pd[:, col] = 1.0
        s2 = make_schedule(env, 3, alloc=Pd)
        assert (s2.arm_seq == col).all()


def test_exhaustion_reselects_to_remaining_arm():
    env = _env(sizes=((50, 500), (300, 300)), mu=((0.5, 0.5), (0.5, 0.5)), seed=6)
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    sch = make_schedule(env, 8, alloc=P)
    ref = ReplayStream(env, sch, "visit")
    seen_exhausted = False
    while ref.t < env.tau_R:
        s = ref.current_segment()
        planned = ref.planned_arm()
        dry = ref.remaining[s, planned] == 0
        _, a, _ = ref.arrive()
        if dry and a is not None:
            assert a != planned and ref.remaining[s, a] >= 0     # moved to the other, non-exhausted arm
            seen_exhausted = True
    assert seen_exhausted
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    eng.advance_planned(env.tau_R)
    assert np.array_equal(eng.n, env.pool_sizes) and eng.n_skipped == 0 and eng.n_reselected == ref.n_reselected
    # _resel with a single available arm always returns it, whatever the uniform
    for u in (0.0, 0.3, 0.999999):
        assert eng._resel(0, [1], u) == 1 and eng._resel(0, [0], u) == 0


@pytest.mark.parametrize("p0", [0.5, 0.40])
def test_count_conservation_half_and_040(p0):
    env = _env(sizes=((800, 2400), (1500, 1500), (600, 3000)), seed=7, n_min=50)
    P = np.tile([p0, 1 - p0], (env.S, 1))
    for seed in (11, 12):
        sch = make_schedule(env, seed, alloc=P)
        eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
        for b in list(env.checkpoints(10)) + [env.tau_R]:
            eng.advance_planned(int(b))
            assert eng.billed == eng.t == min(int(b), env.tau_R)
            assert int(eng.n.sum()) == eng.n_served
            assert eng.n_served + eng.n_skipped == eng.billed
            assert (eng.n <= env.pool_sizes).all() and eng.billing_ok()
            # per segment, served = arrivals of that segment so far (supply = demand per segment, nothing skipped)
            seg_arr = np.bincount(sch.seg_seq[: eng.t].astype(np.int64), minlength=env.S)
            assert np.array_equal(eng.n.sum(1), seg_arr)
        assert eng.n_skipped == 0 and np.array_equal(eng.n, env.pool_sizes)
        s, _ = run_stream(env, QFCMethod("QFC-x", alloc_p=P), seed, PROBS, 0.02)
        assert s["billing_ok"] and s["alloc_kind"] == "fixed"


@pytest.mark.parametrize("P", [None, np.full((3, 2), 0.5), np.tile([0.4, 0.6], (3, 1)), ASYM])
def test_schedule_digest_invariant_to_outcome_permutation(P):
    env = _env(seed=8)
    env2 = env.with_permuted_outcomes(123)
    assert not np.array_equal(env.outcomes_view("visit"), env2.outcomes_view("visit"))
    for seed in (0, 1, 900):
        a = make_schedule(env, seed, alloc=P)
        b = make_schedule(env2, seed, alloc=P)
        assert a.digest() == b.digest()
        assert np.array_equal(a.arm_seq, b.arm_seq) and np.array_equal(a.seg_seq, b.seg_seq)


# ------------------------------------------------------------------------------------------------ diagnosed mechanism
def _starved_env():
    # segment 0: control pool has mean exactly 0 (sd = 0) -> unfloored Neyman share 0, like CR9 dev segment 0
    return _env(sizes=((200, 1000), (2000, 2000), (2000, 2000)), mu=((0.0, 0.05), (0.3, 0.35), (0.4, 0.3)), seed=9,
                n_min=50)


def test_zero_share_starves_cell_and_blocks_certificate_until_sibling_exhausts():
    env = _starved_env()
    mu = env.true_mu("visit")
    sd = np.sqrt(mu * (1 - mu))
    P = sd / sd.sum(1, keepdims=True)                         # the offline script's unfloored oracle Neyman
    assert P[0, 0] == 0.0
    ctx = build_ctx(env, PROBS, 0.02)
    m = QFCMethod("QFC-ney-raw", alloc_p=P)
    m.setup(ctx)
    sch = make_schedule(env, 900, alloc=P)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    before = 0
    for t in ctx.checkpoints:
        eng.advance_planned(int(t))
        if eng.n[0, 1] < env.pool_sizes[0, 1]:                 # sibling (treatment) pool still live
            assert eng.n[0, 0] == 0                            # starved: never drawn, never re-selected
            mh = np.where(eng.n > 0, eng.sum / np.maximum(eng.n, 1), 0.5)
            st = make_stats(ctx.w, mh, eng.n, N=eng.N, x_v=m.params["x_v"])
            res = qfc_certificate_enum(st, ctx.problems, m.params["L1"], ctx.eps, pols=ctx.pols)
            assert all(np.isinf(r["U"]) and not r["certified"] for r in res)
            before += 1
    assert before >= 1
    assert eng.n[0, 0] == env.pool_sizes[0, 0]                 # served only after the sibling ran dry


def test_neyman_floor_keeps_every_cell_live():
    env = _starved_env()
    p = neyman_alloc(env.true_sigma2("visit"), p_min=0.01)
    # clip-then-renormalise: a clipped zero ends at p_min / (1 + p_min) = 0.0099 (not exactly p_min); still live
    assert (p >= 0.01 / 1.01 - 1e-12).all() and np.allclose(p.sum(1), 1)
    sch = make_schedule(env, 900, alloc=p)
    eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
    eng.advance_planned(env.tau_R // 2)
    assert eng.n[0, 0] > 0
