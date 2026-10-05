"""r5_fdc_impl: unit tests for FDCMethod / FDCAblation (baselines/fdc.py; methodology s2, plan/theory/fdc_theorem.md).

(1) ledger values on the CR9 development ctx (beta = 14.2242, x_v = 11.8776, sum_q |Pi_{B_q}| = 3386, K = 20, S*A = 18);
(2) FDC == FDCAblation('half', 'feas', 'r5') stream by stream (N80, cert_k, decided_pi, digest) on 10 CR9 dev streams;
(3) FDCAblation('pool', 'qstar', 'r4') == r4 QFCMethod() live, and == the stored r4 G1-shape QFC streams
    (exp/results/pilots/r4_g1_shape_a/streams.jsonl); the default runs dev seeds 900-909, set FDC_R4_IDENTITY_ALL=1 to
    run all 900-999 (the run script run_r5_fdc_impl.py always runs all 100);
(4) overrides downgrade validity to 'none' and append '-explore'; FDCMethod takes no constructor arguments;
(5) monotonicity: per problem, U_FDC <= U_{QFC-half} on the same state (smaller beta and x_v);
(6) boundaries: n = 0 -> U = +inf; exhausted cells exact (width 0); mu_hat in {0, 1} -> sigma_bar^2 > 0; tau_R -> U = 0.
Only CR9 dev halves and dev seeds 900-999 are touched.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from dsswm.baselines.b4_bal import balanced_alloc
from dsswm.baselines.fdc import FACTORIAL_CELLS, FDCAblation, FDCMethod, fdc_ledger
from dsswm.baselines.frontier_common import FrontierState, QFCMethod, make_ctx
from dsswm.certify.quadknap import make_stats, pair_terms, qfc_certificate_enum, _width
from dsswm.streams import frontier as fr
from dsswm.streams.frontier_runner import SyntheticPoolEnv, build_ctx, run_stream, true_policy_values

RES = Path(__file__).resolve().parents[3] / "results"
R4_STREAMS = RES / "pilots" / "r4_g1_shape_a" / "streams.jsonl"
FDC_SEEDS = list(range(900, 910))
R4_SEEDS = list(range(900, 1000)) if os.environ.get("FDC_R4_IDENTITY_ALL") else list(range(900, 910))


@pytest.fixture(scope="module")
def cr9():
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", "dev")
    eps = float(json.loads((RES / "r4_gates" / "eps.json").read_text())["eps_star"])
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    return env, ctx, J, Js


def _run(cr9, method, seed):
    env, ctx, J, Js = cr9
    s, _ = run_stream(env, method, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=False)
    return s


def _key(s):
    return (s["N80"], tuple(s["cert_k"]), tuple(s["decided_pi"]), s["schedule_digest"], s["n_false"], s["billed"])


# ------------------------------------------------------------------------------------------------ (1) ledger
def test_ledger_cr9_dev(cr9):
    _, ctx, _, _ = cr9
    assert ctx.eps == 0.001 and ctx.delta == 0.05 and len(ctx.checkpoints) == 20 and (ctx.S, ctx.A) == (9, 2)
    m = FDCMethod()
    m.setup(ctx)
    p = m.params
    assert p["union_size"] == 3386 and int(ctx.feas.sum()) == 3386
    assert m.ledger_info["per_q_feasible"] == [14, 44, 69, 111, 144, 186, 212, 243, 256, 269, 300, 326, 368, 401, 443]
    assert p["K"] == 20 and p["C_var"] == 18
    assert round(p["L1"], 4) == 14.2242 and abs(p["L1"] - math.log(3386 * 20 / 0.045)) < 1e-12
    assert round(p["x_v"], 4) == 11.8776 and abs(p["x_v"] - math.log(720 / 0.005)) < 1e-12
    assert (p["delta_main"], p["delta_var"]) == (0.045, 0.005)
    led = m.ledger_info
    assert abs(led["bound_main"] - 0.045) < 1e-12 and abs(led["bound_var"] - 0.005) < 1e-12
    assert abs(led["bound_main"] + led["bound_var"] - ctx.delta) < 1e-12
    # the hard-coded beta = 14.1 of the pre-r5 offline script would violate the ledger
    assert 3386 * 20 * math.exp(-14.1) + 0.005 > 0.0559
    # design: frozen 50/50, identical to B4-bal(0.5)
    assert m.alloc_kind == "fixed" and np.array_equal(m.alloc_p, np.full((9, 2), 0.5))
    assert np.array_equal(m.alloc_p, balanced_alloc(9, 2, 0.5))
    assert m.validity == "rigorous" and m.name == "FDC"


def test_ledger_factorial_cells(cr9):
    _, ctx, _, _ = cr9
    from dsswm.baselines.frontier_common import qfc_default_params
    r4 = qfc_default_params(ctx.S, ctx.A, ctx.Q, K=20)
    vals = {}
    for cell in FACTORIAL_CELLS:
        a = FDCAblation(*cell)
        a.setup(ctx)
        assert a.validity == "rigorous" and not a.name.endswith("-explore"), cell
        assert a.alloc_kind == ("pool" if cell[0] == "pool" else "fixed")
        vals[cell] = (a.params["L1"], a.params["x_v"])
    assert vals[("pool", "qstar", "r4")] == (r4["L1"], r4["x_v"])            # bitwise r4 ledger
    assert round(vals[("pool", "qstar", "r4")][0], 3) == 15.161
    assert vals[("half", "feas", "r5")] == (math.log(3386 * 20 / 0.045), math.log(720 / 0.005))
    assert abs(vals[("half", "qstar", "r5")][0] - math.log(7680 * 20 / 0.045)) < 1e-9
    assert abs(vals[("half", "feas", "r4")][0] - math.log(3386 * 20 / 0.04)) < 1e-12
    assert vals[("half", "feas", "r4")][1] == r4["x_v"]


# ------------------------------------------------------------------------------------------------ (2) FDC identity
@pytest.mark.parametrize("seed", FDC_SEEDS)
def test_fdc_equals_ablation_half_feas_r5(cr9, seed):
    a = _run(cr9, FDCMethod(), seed)
    b = _run(cr9, FDCAblation("half", "feas", "r5"), seed)
    assert _key(a) == _key(b)
    assert a["n_false"] == 0 and a["billing_ok"]
    assert a["skipped"] == 0                                   # permutation schedule: double exhaustion never skips


# ------------------------------------------------------------------------------------------------ (3) r4 identity
@pytest.fixture(scope="module")
def r4_stored():
    rows = [json.loads(l) for l in R4_STREAMS.read_text().splitlines() if l.strip()]
    return {r["perm_seed"]: r for r in rows if r["method"] == "QFC" and r.get("layer") == "CR9"}


@pytest.mark.parametrize("seed", R4_SEEDS)
def test_ablation_pool_qstar_r4_equals_r4_qfc(cr9, r4_stored, seed):
    a = _run(cr9, FDCAblation("pool", "qstar", "r4"), seed)
    b = _run(cr9, QFCMethod(), seed)
    assert _key(a) == _key(b)
    st = r4_stored[seed]
    assert (a["N80"], a["cert_k"], a["decided_pi"], a["schedule_digest"], a["n_false"]) == \
        (st["N80"], st["cert_k"], st["decided_pi"], st["schedule_digest"], st["n_false"])


# ------------------------------------------------------------------------------------------------ (4) validity guard
def test_fdc_constructor_rejects_overrides():
    for kw in ({"beta": 14.1}, {"x_v": 11.0}, {"share": 0.45}, {"delta": 0.1}, {"params": {}}, {"alloc_p": None},
               {"name": "FDC2"}, {"K_override": 30}):
        with pytest.raises(TypeError):
            FDCMethod(**kw)


@pytest.mark.parametrize("kw,reason", [
    ({"beta_override": 14.1}, "beta=14.1"),
    ({"x_v_override": 11.0}, "x_v=11"),
    ({"share": 0.40}, "share=0.4"),
    ({"share": 0.45}, "share=0.45"),
    ({"K_override": 10}, "K_override=10<K=20"),
])
def test_ablation_override_downgrades(cr9, kw, reason):
    _, ctx, _, _ = cr9
    a = FDCAblation("half", "feas", "r5", **kw)
    a.setup(ctx)
    assert a.validity == "none" and a.name.endswith("-explore")
    assert any(r.startswith(reason) for r in a.explore_reasons), a.explore_reasons
    assert a.describe()["validity"] == "none"


def test_ablation_legal_overrides_stay_rigorous(cr9):
    _, ctx, _, _ = cr9
    for kw in ({"share": 0.5}, {"K_override": 20}, {"K_override": 60}):
        a = FDCAblation("half", "feas", "r5", **kw)
        a.setup(ctx)
        assert a.validity == "rigorous" and not a.name.endswith("-explore"), kw
    a = FDCAblation("half", "feas", "r5", K_override=60)
    a.setup(ctx)
    assert abs(a.params["L1"] - math.log(3386 * 60 / 0.045)) < 1e-12      # conservative: larger union


def test_r4_ledger_downgrades_when_cvar_undercounts():
    env = SyntheticPoolEnv(np.full((25, 2), 40), np.full((25, 2), 0.3), seed=0, n_min=10)   # S*A = 50 > 48
    probs = [fr.Problem("toy|B0.50", "visit", (0.0, 1.0), 0.5)]
    pols = np.vstack([np.zeros(25, dtype=np.int64), np.ones(25, dtype=np.int64)])
    ctx = make_ctx(env.w, env.pool_sizes, probs, 0.01, 0.05, env.checkpoints(20), env.replan_interval, env.tau_R,
                   pols=pols)
    a = FDCAblation("pool", "feas", "r4")
    a.setup(ctx)
    assert a.validity == "none" and any(r.startswith("C_var=48<S*A=50") for r in a.explore_reasons)
    b = FDCAblation("pool", "feas", "r5")
    b.setup(ctx)
    assert b.validity == "rigorous" and b.params["C_var"] == 50


def test_fdc_rejects_other_delta(cr9):
    env, ctx, _, _ = cr9
    ctx2 = build_ctx(env, ctx.problems, ctx.eps, delta=0.1)
    with pytest.raises(ValueError):
        FDCMethod().setup(ctx2)
    a = FDCAblation("half", "feas", "r5")
    a.setup(ctx2)
    assert a.validity == "none"


def test_setup_is_per_ctx():
    """Reusing one FDC object on two ctxs recomputes the ledger and the index map (no stale cache)."""
    probs = [fr.Problem(f"toy|B{b:.2f}", "visit", (0.0, 1.0), b) for b in (0.3, 0.6)]
    out = []
    m = FDCMethod()
    for S in (3, 4):
        env = SyntheticPoolEnv(np.full((S, 2), 200), np.full((S, 2), 0.3), seed=S, n_min=20)
        ctx = make_ctx(env.w, env.pool_sizes, probs, 0.01, 0.05, env.checkpoints(20), env.replan_interval, env.tau_R)
        m.setup(ctx)
        assert m.params["union_size"] == int(ctx.feas.sum()) and m.params["C_var"] == 2 * S
        assert m.alloc_p.shape == (S, 2)
        st = FrontierState(ctx.checkpoints[3], np.full((S, 2), 50), np.full((S, 2), 15.0), env.pool_sizes,
                           np.ones(2, bool), lambda n: None)
        res = m.certify(ctx, st)
        assert all(0 <= ih < ctx.P for _, ih, _ in res)
        out.append(m.params["L1"])
    assert out[0] != out[1]


# ------------------------------------------------------------------------------------------------ (5) monotonicity
def _states_cr9(cr9, seed, method):
    """Certificate-time states (n, sum) of one stream under ``method``'s design, read from the engine."""
    from dsswm.envs.pool_replay import make_schedule
    from dsswm.streams.frontier_runner import StreamEngine
    env, ctx, _, _ = cr9
    method.setup(ctx)
    sch = make_schedule(env, seed, alloc=method.alloc_p if method.alloc_kind == "fixed" else None)
    eng = StreamEngine(env, sch, "visit", delta_cell=ctx.delta / (env.S * env.A))
    for b in ctx.checkpoints:
        eng.advance_planned(int(b))
        yield FrontierState(eng.t, eng.n, eng.sum, eng.N, np.ones(ctx.Q, bool), eng.cs_at)


@pytest.mark.parametrize("seed", [900, 901])
def test_fdc_width_le_qfc_half(cr9, seed):
    _, ctx, _, _ = cr9
    fdc, qh = FDCMethod(), FDCAblation("half", "qstar", "r4")
    fdc.setup(ctx)
    qh.setup(ctx)
    assert fdc.params["L1"] < qh.params["L1"] and fdc.params["x_v"] < qh.params["x_v"]
    n_states = 0
    for st in _states_cr9(cr9, seed, FDCMethod()):
        a = fdc.certify(ctx, st)
        b = qh.certify(ctx, st)
        for (ca, ia, Ua), (cb, ib, Ub) in zip(a, b):
            assert ia == ib                                    # same centre (depends on mu_hat only)
            assert Ua <= Ub + 1e-15
            assert cb <= ca                                    # QFC-half certified => FDC certified
        # pairwise widths and variance UCBs, every challenger
        sf = make_stats(ctx.w, st.mu_hat, st.n, N=st.N, x_v=fdc.params["x_v"])
        sq = make_stats(ctx.w, st.mu_hat, st.n, N=st.N, x_v=qh.params["x_v"])
        assert np.all(sf.var_ucb <= sq.var_ucb + 1e-15)
        ih = ia
        _, Vf, bf = pair_terms(sf, ctx.pols, ctx.pols[ih])
        _, Vq, bq = pair_terms(sq, ctx.pols, ctx.pols[ih])
        wf, wq = _width(Vf, bf, fdc.params["L1"]), _width(Vq, bq, qh.params["L1"])
        fin = np.isfinite(wq)
        assert np.all(wf[fin] <= wq[fin] + 1e-15) and np.array_equal(np.isfinite(wf), fin)
        n_states += 1
    assert n_states == 20


def test_width_monotone_in_beta_and_xv():
    rng = np.random.default_rng(42)
    S, A = 5, 2
    w = rng.dirichlet(np.ones(S))
    n = rng.integers(5, 400, (S, A)).astype(float)
    mu = rng.uniform(0, 1, (S, A))
    pols = fr.enumerate_policies(S, A)
    prev = None
    for beta, xv in [(8.0, 6.0), (10.0, 8.0), (14.2242, 11.8776), (15.161, 12.165)]:
        st = make_stats(w, mu, n, x_v=xv)
        _, V, b = pair_terms(st, pols, pols[3])
        wd = _width(V, b, beta)
        if prev is not None:
            assert np.all(wd >= prev - 1e-15)
        prev = wd


# ------------------------------------------------------------------------------------------------ (6) boundaries
PROBS5 = [fr.Problem(f"toy|B{b:.2f}", "visit", (0.0, 1.0), b) for b in (0.2, 0.35, 0.5, 0.65, 0.8)]


def _toy_ctx(sizes, mu, seed=0, n_min=20):
    env = SyntheticPoolEnv([list(r) for r in sizes], [list(r) for r in mu], seed=seed, n_min=n_min)
    ctx = make_ctx(env.w, env.pool_sizes, PROBS5, 0.0, 0.05, env.checkpoints(20), env.replan_interval, env.tau_R,
                   stop_frac=0.8)
    return env, ctx


def test_n_zero_gives_infinite_width():
    env, ctx = _toy_ctx(((300, 900), (700, 500), (400, 400)), ((0.3, 0.5), (0.4, 0.45), (0.2, 0.6)))
    m = FDCMethod()
    m.setup(ctx)
    n = np.array([[0, 40], [60, 50], [30, 30]])
    sm = np.array([[0.0, 20.0], [24.0, 22.0], [6.0, 18.0]])
    st = FrontierState(int(n.sum()), n, sm, env.pool_sizes, np.ones(ctx.Q, bool), lambda x: None)
    res = m.certify(ctx, st)
    stats = make_stats(ctx.w, st.mu_hat, st.n, N=st.N, x_v=m.params["x_v"])
    raw = qfc_certificate_enum(stats, ctx.problems, m.params["L1"], ctx.eps, pols=ctx.pols)
    n_inf = 0
    for q, ((ok, ih, U), r) in enumerate(zip(res, raw)):
        feas = np.flatnonzero(ctx.feas[q])
        touches = (ctx.pols[feas, 0] != ctx.pols[ih, 0]).any()
        if touches:     # some feasible challenger differs from pi_hat in segment 0 -> touches the n = 0 cell
            assert U == math.inf and not ok and r["U"] == math.inf
            n_inf += 1
        else:           # e.g. a singleton class: no challenger touches the empty cell -> finite
            assert np.isfinite(U)
    assert n_inf >= 3
    assert m.params["x_v"] == math.log(2 * 6 * 20 / 0.005)


def test_exhausted_cells_are_exact_and_mu_hat_extremes():
    env, ctx = _toy_ctx(((30, 50), (40, 40), (50, 30)), ((0.0, 1.0), (0.4, 0.45), (0.2, 0.6)))
    m = FDCMethod()
    m.setup(ctx)
    N = env.pool_sizes
    # segment 0 fully exhausted, others partial; mu_hat in {0, 1} on live cells of segment 1
    n = np.array([[30, 50], [10, 10], [20, 20]])
    sm = np.array([[0.0, 50.0], [0.0, 10.0], [4.0, 12.0]])
    stats = make_stats(ctx.w, sm / n, n, N=N, x_v=m.params["x_v"])
    assert stats.var_ucb[1, 0] > 0 and stats.var_ucb[1, 1] > 0           # L2: mu_hat in {0,1} -> sigma_bar^2 > 0
    pols = ctx.pols
    ih = int(np.flatnonzero((pols == np.array([0, 0, 0])).all(1))[0])
    dh, V, b = pair_terms(stats, pols, pols[ih])
    only0 = int(np.flatnonzero((pols == np.array([1, 0, 0])).all(1))[0])  # differs only in the exhausted segment
    assert V[only0] == 0.0 and b[only0] == 0.0 and _width(V, b, m.params["L1"])[only0] == 0.0
    assert dh[only0] == pytest.approx(ctx.w[0] * (1.0 - 0.0))           # exact difference


def test_tau_R_certifies_exactly_and_full_stream():
    env, ctx = _toy_ctx(((300, 900), (700, 500), (400, 400)), ((0.3, 0.5), (0.4, 0.45), (0.2, 0.6)))
    m = FDCMethod()
    m.setup(ctx)
    st = FrontierState(env.tau_R, env.pool_sizes, (env.pool_sizes * np.asarray(env.true_mu("visit"))),
                       env.pool_sizes, np.ones(ctx.Q, bool), lambda x: None)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    for q, (ok, ih, U) in enumerate(m.certify(ctx, st)):
        assert ok and abs(U) < 1e-12
        assert J[ih] == pytest.approx(J[ctx.feas[q]].max())
    # whole stream at eps = 0: no false certification, billing conserved, every pool served at tau_R
    for seed in range(5):
        s, rows = run_stream(env, FDCMethod(), 100 + seed, ctx.problems, ctx.eps, ctx=ctx)
        assert s["n_false"] == 0 and s["billing_ok"] and s["skipped"] == 0
        assert s["alloc_kind"] == "fixed" and s["validity"] == "rigorous"


def test_early_exhaustion_reselection_path():
    """Control pools much smaller than half the segment: 50/50 exhausts control early and re-selects to treatment."""
    env, ctx = _toy_ctx(((40, 900), (60, 700), (30, 500)), ((0.1, 0.3), (0.2, 0.25), (0.0, 0.4)), n_min=10)
    s, rows = run_stream(env, FDCMethod(), 7, ctx.problems, ctx.eps, ctx=ctx)
    assert s["reselected"] > 0 and s["billing_ok"] and s["n_false"] == 0
    last = np.array(rows[-1]["n_cells"]).reshape(3, 2)
    assert np.all(last <= env.pool_sizes)


# ------------------------------------------------------------------------------------------------ proof-code rows
def test_l2_inversion_contains_exact_set_and_covers():
    """Lemma L2 as coded (make_stats -> bernstein_mu_ci, sigma2_ucb): the outward-bisected interval contains the exact
    inversion set {m : |mu_hat - m| <= sqrt(2 m(1-m) x/n) + x/(3n)} (fine grid), sigma_bar^2 >= max m(1-m) over it,
    and the true mean of a binary WoR pool is covered at the FDC exponent x_v = 11.8776 (no misses expected)."""
    from dsswm.theory_checks.mc_l1 import bernstein_mu_ci, sigma2_ucb
    x = math.log(720 / 0.005)
    grid = np.linspace(0.0, 1.0, 200001)
    for n, k in [(1, 0), (1, 1), (5, 0), (5, 5), (40, 3), (400, 17), (5000, 2500), (20000, 0)]:
        mh = k / n
        lo, hi = bernstein_mu_ci(np.array([mh]), np.array([n]), np.array([np.inf]), x)
        inside = grid[np.abs(mh - grid) <= np.sqrt(2 * grid * (1 - grid) * x / n) + x / (3 * n)]
        assert lo[0] <= inside.min() + 1e-12 and hi[0] >= inside.max() - 1e-12
        s2 = sigma2_ucb(lo, hi)[0]
        assert s2 >= (inside * (1 - inside)).max() - 1e-12
        if mh in (0.0, 1.0):
            assert s2 > 0                                       # mu_hat in {0,1}: sigma_bar^2 > 0 (plug-in gives 0)
    rng = np.random.default_rng(42)
    miss = 0
    for _ in range(300):
        Np = int(rng.integers(50, 3000))
        pool = (rng.random(Np) < rng.uniform(0, 0.2)).astype(float)
        mu = pool.mean()
        perm = rng.permutation(pool)
        ns = np.unique(np.linspace(1, Np - 1, 12).astype(int))
        mh = np.array([perm[:m].mean() for m in ns])
        lo, hi = bernstein_mu_ci(mh, ns, np.full(len(ns), Np), x)
        miss += int(((mu < lo - 1e-12) | (mu > hi + 1e-12)).sum())
    assert miss == 0
    # exhausted -> point mass (exact), n = 0 -> [0, 1]
    lo, hi = bernstein_mu_ci(np.array([0.3, 0.5]), np.array([10, 0]), np.array([10, 10]), x)
    assert (lo[0], hi[0]) == (0.3, 0.3) and (lo[1], hi[1]) == (0.0, 1.0)


def test_pi_hat_tie_rule_lowest_index():
    env, ctx = _toy_ctx(((300, 900), (700, 500), (400, 400)), ((0.3, 0.5), (0.4, 0.45), (0.2, 0.6)))
    m = FDCMethod()
    m.setup(ctx)
    n = np.full((3, 2), 100)
    st = FrontierState(600, n, np.full((3, 2), 30.0), env.pool_sizes, np.ones(ctx.Q, bool), lambda x: None)
    res = m.certify(ctx, st)                                   # all mu_hat equal -> J_hat ties on every policy
    for q, (_, ih, _) in enumerate(res):
        assert ih == int(np.flatnonzero(ctx.feas[q])[0])


def test_checkpoints_and_stop_rule(cr9):
    env, ctx, J, Js = cr9
    assert ctx.checkpoints.tolist() == [50000, 64847, 84103, 109077, 141466, 183474, 237955, 308614, 400254, 519107,
                                        673252, 873169, 1132450, 1468723, 1904849, 2470480, 3204070, 4155495,
                                        5389439, 6989793]
    assert ctx.stop_k == 12 and ctx.Q == 15
    s, rows = run_stream(env, FDCMethod(), 903, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=True)
    assert [r["t"] for r in rows] == ctx.checkpoints[:len(rows)].tolist()   # certificates only at checkpoints
    assert rows[-1]["n_cert"] >= 12 and all(r["n_cert"] < 12 for r in rows[:-1])
    assert s["N80"] == rows[-1]["t"] and not s["censored"]
    # sticky answers: cert_k is the first checkpoint whose U_q <= eps
    for q, k in enumerate(s["cert_k"]):
        Us = [r["U"][q] for r in rows]
        first = next((i for i, u in enumerate(Us) if u <= ctx.eps), -1)
        assert k == first


def test_error_penalised_N80():
    """A wrong certification sets N80 = tau_R (harness rule); constructed with an always-certifying invalid method."""
    from dsswm.baselines.frontier_common import Method
    env, ctx = _toy_ctx(((300, 900), (700, 500), (400, 400)), ((0.3, 0.5), (0.4, 0.45), (0.2, 0.6)))

    class Worst(Method):
        name, alloc_kind, validity = "worst", "fixed", "none"

        def setup(self, c):
            self.alloc_p = np.full((c.S, c.A), 0.5)

        def certify(self, c, st):
            J = true_policy_values(c.pols, env.w, env.true_mu("visit"))
            return [(True, int(np.flatnonzero(c.feas[q])[np.argmin(J[c.feas[q]])]), 0.0) for q in range(c.Q)]

    s, _ = run_stream(env, Worst(), 1, ctx.problems, ctx.eps, ctx=ctx)
    assert s["n_false"] > 0 and s["N80"] == ctx.tau_R and s["censored"]
