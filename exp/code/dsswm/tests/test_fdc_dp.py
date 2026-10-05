"""Tests for FDC-DP / RECT-BF-DP (dsswm/baselines/fdc_dp.py, rect_dp.py; plan/fdc_dp_theory.md).

  * counting DP == enumeration; DP argmax / DP max == brute force
  * branch-and-bound (scheme b) == enumeration exactly (max and decision), scheme (a) >= enumeration (conservative)
  * on Bennett-FPC / exact-HG cell tables at S <= 9: FDC-DP(b) == enumeration FDC-BF evaluated with the same lambda set
    (independent per-policy implementation), FDC-DP(a) never certifies more; agreement with native FDC-BF
    (fdc_bet.direction_widths, per-direction lambda grid) reported and bounded
  * RECT-BF-DP == enumeration rectangle (rect_U formula)
  * toy Monte Carlo FWER with an invalid power control (a few minutes)
  * runtime per checkpoint at S = 64 (synthetic population with real-data-like pool sizes)
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest

from dsswm.baselines.fdc_bet import direction_widths, psi_bar
from dsswm.baselines.fdc_dp import (FDCDP, FDCDPTimeUniform, SegProblems, bnb, cell_tables, count_policies,
                                    dp_argmax, dp_ledger, dp_max_cols, enum_U, joint_columns, lambda_grid,
                                    make_seg_ctx, reduce_costs, scheme_a_U)
from dsswm.baselines.frontier_common import FrontierCtx
from dsswm.baselines.rect_dp import RectBFDP, rect_dp_certificate
from dsswm.baselines.rect_v6 import hg_mean_interval
from dsswm.envs.seg_v10 import run_stream_seg, seg_problems, true_opt
from dsswm.streams.frontier_runner import SyntheticPoolEnv


def _pols(S, A):
    return np.array(list(itertools.product(range(A), repeat=S)), dtype=np.int64)


# ------------------------------------------------------------------------------------------------ DP primitives
def test_count_policies_matches_enumeration():
    rng = np.random.default_rng(1)
    for _ in range(60):
        S, A = int(rng.integers(1, 8)), int(rng.integers(2, 4))
        cost = rng.integers(0, 5, size=(S, A)) * int(rng.integers(1, 4))
        B = rng.integers(-1, int(cost.max(1).sum()) + 2, size=4)
        pols = _pols(S, A)
        c = cost[np.arange(S)[None, :], pols].sum(1)
        assert count_policies(cost, B) == [int((c <= b).sum()) for b in B]


def test_count_policies_large_S_closed_form():
    S = 64
    cost = np.stack([np.zeros(S, int), np.full(S, 8)], 1)
    B = np.array([int(math.floor(b * 8 * S + 1e-9)) for b in (0.1, 0.5, 0.8)])
    got = count_policies(cost, B)
    want = [sum(math.comb(S, j) for j in range(int(b // 8) + 1)) for b in B]
    assert got == want
    led = dp_ledger(SegProblems(cost, B, ["a", "b", "c"]), S, 2, 20)
    assert abs(led["beta"] - (math.log(sum(want)) + math.log(20 / 0.045))) < 1e-9
    assert led["beta"] <= led["beta_upper_SlnA"] + 1e-9


def test_dp_argmax_and_max_match_brute_force():
    rng = np.random.default_rng(2)
    for _ in range(150):
        S, A, J = int(rng.integers(1, 7)), int(rng.integers(2, 4)), int(rng.integers(1, 4))
        cost = rng.integers(0, 4, size=(S, A))
        vals = rng.normal(size=(S, A))
        cap = int(rng.integers(0, int(cost.max(1).sum()) + 1))
        pols = _pols(S, A)
        c = cost[np.arange(S)[None, :], pols].sum(1)
        v = vals[np.arange(S)[None, :], pols].sum(1)
        best, pi = dp_argmax(vals, cost, cap)
        feas = c <= cap
        if not feas.any():
            assert pi is None
            continue
        assert abs(best - v[feas].max()) < 1e-12
        assert int(cost[np.arange(S), pi].sum()) <= cap and abs(vals[np.arange(S), pi].sum() - best) < 1e-12
        # dp_max_cols: max over pi != ph
        ph = pi
        T = rng.normal(size=(S, A, J))
        T[np.arange(S), ph] = 0.0
        diff = np.ones((S, A), bool)
        diff[np.arange(S), ph] = False
        f1 = dp_max_cols(T, cost, diff, cap)
        tv = T[np.arange(S)[None, :], pols].sum(1)                    # (P, J)
        ok = feas & ~(pols == ph[None, :]).all(1)
        for cc in range(cap + 1):
            m = ok & (c <= cc)
            want = tv[m].max(0) if m.any() else np.full(J, -np.inf)
            assert np.allclose(f1[:, cc], want, atol=1e-12)


def _rand_cols(rng, S, A, J, p_inf=0.15):
    cost = rng.integers(0, 4, size=(S, A))
    ph = rng.integers(0, A, size=S)
    T = rng.normal(0, 1, size=(S, A, J))
    if rng.random() < p_inf:
        T[rng.integers(S), rng.integers(A), :] = np.inf
    T[np.arange(S), ph] = 0.0
    const = rng.normal(0, 1, size=J)
    diff = np.ones((S, A), bool)
    diff[np.arange(S), ph] = False
    cap = int(cost[np.arange(S), ph].sum()) + int(rng.integers(0, 6))
    return const, T, cost, diff, cap, ph


def test_bnb_exact_and_scheme_a_conservative_random():
    rng = np.random.default_rng(3)
    for _ in range(300):
        S, A, J = int(rng.integers(2, 8)), int(rng.integers(2, 4)), int(rng.integers(1, 6))
        const, T, cost, diff, cap, ph = _rand_cols(rng, S, A, J)
        e, _ = enum_U(const, T, cost, cap, ph)
        r = bnb(const, T, cost, diff, cap)
        assert r["status"] == "exact" and (r["U"] == e or abs(r["U"] - e) < 1e-9)
        ua = scheme_a_U(const, T, cost, diff, [cap])[0]
        assert ua >= e - 1e-9
        for thr in (e - 1e-7, e + 1e-7, e - 0.3, e + 0.3):
            if np.isfinite(thr):
                d = bnb(const, T, cost, diff, cap, thr=thr)
                assert (d["status"] == "certified") == (e <= thr)


# ------------------------------------------------------------------------------------------------ FDC-BF cell tables
def _rand_cells(rng, S, A=2, p_exh=0.1, p_zero=0.03):
    N = rng.integers(40, 400, size=(S, A))
    mu = rng.uniform(0.05, 0.95, size=(S, A))
    M = np.rint(mu * N).astype(int)
    n = np.floor(N * rng.uniform(0.05, 0.9, size=(S, A))).astype(int)
    n = np.where(rng.random((S, A)) < p_exh, N, n)
    n = np.where(rng.random((S, A)) < p_zero, 0, n)
    s = rng.hypergeometric(M, N - M, np.maximum(n, 1)) * (n > 0)
    s = np.minimum(s, n)
    return N, n, s


def _instance(rng, S, beta=None, alpha=1e-4):
    N, n, s = _rand_cells(rng, S)
    lo, hi = hg_mean_interval(N, n, s, alpha)
    w = rng.dirichlet(np.ones(S) * 3)
    mu = np.where(n > 0, s / np.maximum(n, 1), 0.5)
    k1 = rng.integers(1, 4, size=S)
    cost = np.stack([np.zeros(S, int), k1], 1)
    B = np.sort(rng.integers(0, int(k1.sum()) + 1, size=3))
    beta = float(rng.uniform(4, 12)) if beta is None else beta
    return dict(N=N, n=n, s=s, lo=lo, hi=hi, w=w, mu=mu, cost=cost, B=B, beta=beta)


def _fdcbf_same_grid_enum(I, ph, lam, cap, pols):
    """Independent per-policy FDC-BF[Lambda]: W = min over lam of (beta + sum Psi_bar)/lambda, zero_touch -> inf,
    all-deterministic difference -> width 0; U = max over feasible pi != ph of Delta_hat + W."""
    S = len(I["w"])
    n, N = I["n"].astype(float), I["N"].astype(float)
    vmax = np.clip(0.5, I["lo"], I["hi"])
    vmax = vmax * (1 - vmax)
    live = (n > 0) & (n < N)
    var0 = live & (vmax * (N - n) <= 0)
    det = (n >= N) | var0
    best = -np.inf
    cst = I["cost"][np.arange(S)[None, :], pols].sum(1)
    for i in range(len(pols)):
        pi = pols[i]
        if (pi == ph).all() or cst[i] > cap:
            continue
        D = np.flatnonzero(pi != ph)
        dh = float(sum(I["w"][s] * (I["mu"][s, pi[s]] - I["mu"][s, ph[s]]) for s in D))
        cells = [(s, pi[s]) for s in D] + [(s, ph[s]) for s in D]
        if any(n[c] <= 0 for c in cells):
            W = np.inf
        elif all(det[c] for c in cells):
            W = 0.0
        else:
            F = np.zeros(len(lam))
            for (s, a) in cells:
                if live[s, a] and not var0[s, a]:
                    F += psi_bar(lam * I["w"][s], n[s, a], N[s, a], I["lo"][s, a], I["hi"][s, a], "bennett")
            W = float(np.min((I["beta"] + F) / lam))
        best = max(best, dh + W)
    return best


@pytest.mark.parametrize("S", [3, 5, 7, 9])
def test_fdc_dp_equals_enumeration_fdcbf_same_lambda_set(S):
    rng = np.random.default_rng(100 + S)
    pols = _pols(S, 2)
    n_dec = 0
    for _ in range(12 if S < 9 else 6):
        I = _instance(rng, S)
        for q, cap in enumerate(I["B"]):
            _, ph = dp_argmax(I["w"][:, None] * I["mu"], I["cost"], int(cap))
            eps = 0.05
            lam = lambda_grid(I["beta"], eps, I["w"], I["n"], I["N"], I["lo"], I["hi"])
            P, det = cell_tables(I["w"], I["n"], I["N"], I["lo"], I["hi"], lam)
            const, T, diff = joint_columns(I["w"], I["mu"], ph, P, det, lam, I["beta"])
            ref = _fdcbf_same_grid_enum(I, ph, lam, int(cap), pols)
            rb = bnb(const, T, I["cost"], diff, int(cap))
            assert (rb["U"] == ref) or abs(rb["U"] - ref) <= 1e-9 * max(1.0, abs(ref)), (rb, ref)
            ua = scheme_a_U(const, T, I["cost"], diff, [int(cap)])[0]
            assert ua >= ref - 1e-12
            for thr in (ref * 0.999, ref * 1.001, ref + 1e-3, ref - 1e-3):
                if np.isfinite(thr) and thr >= 0:
                    d = bnb(const, T, I["cost"], diff, int(cap), thr=thr)
                    assert (d["status"] == "certified") == (ref <= thr)
                    if ua <= thr:                       # scheme (a) certifies only if (b) does
                        assert d["status"] == "certified"
                    n_dec += 1
    assert n_dec > 0


def test_fdc_dp_vs_native_fdcbf_grid_agreement():
    """Native FDC-BF uses a per-direction lambda grid; FDC-DP a global one.  Both are valid; their U differ only by
    grid discretisation (reported; bounded here at 0.5 % of max(|U|, 0.01))."""
    rng = np.random.default_rng(7)
    S = 7
    pols = _pols(S, 2)
    rel = []
    for _ in range(15):
        I = _instance(rng, S, beta=float(rng.uniform(6, 12)))
        if (I["n"] <= 0).any():
            continue
        for cap in I["B"]:
            _, ph = dp_argmax(I["w"][:, None] * I["mu"], I["cost"], int(cap))
            cst = I["cost"][np.arange(S)[None, :], pols].sum(1)
            feas = (cst <= cap)[None, :]
            ctx = FrontierCtx(w=I["w"], S=S, A=2, pols=pols, problems=["q"], eps=0.05, delta=0.05,
                              checkpoints=np.arange(20), replan=1, feas=feas, N=I["N"], tau_R=1, stop_k=1)
            ih = int(np.flatnonzero((pols == ph[None, :]).all(1))[0])
            wd = direction_widths(ctx, ih, I["n"], I["N"], I["lo"], I["hi"], I["beta"], "bennett", True)
            J = (I["w"][None, :] * I["mu"][np.arange(S)[None, :], pols]).sum(1)
            Un = np.where(feas[0] & (np.arange(len(pols)) != ih), J - J[ih] + wd, -np.inf).max()
            lam = lambda_grid(I["beta"], 0.05, I["w"], I["n"], I["N"], I["lo"], I["hi"])
            P, det = cell_tables(I["w"], I["n"], I["N"], I["lo"], I["hi"], lam)
            const, T, diff = joint_columns(I["w"], I["mu"], ph, P, det, lam, I["beta"])
            Ub = bnb(const, T, I["cost"], diff, int(cap))["U"]
            if np.isfinite(Un):
                rel.append(abs(Ub - Un) / max(abs(Un), 1e-2))     # relative, floored at the eps scale
    print({"n": len(rel), "max_rel_diff": max(rel), "median_rel_diff": float(np.median(rel))})
    assert rel and max(rel) < 5e-3, max(rel)


# ------------------------------------------------------------------------------------------------ RECT-BF-DP
def test_rect_dp_equals_enumeration():
    rng = np.random.default_rng(11)
    for _ in range(40):
        S = int(rng.integers(2, 8))
        lo = rng.uniform(0, 0.5, size=(S, 2))
        hi = lo + rng.uniform(0, 0.5, size=(S, 2))
        mu = (lo + hi) / 2 + rng.normal(0, 0.01, size=(S, 2))
        w = rng.dirichlet(np.ones(S))
        k1 = rng.integers(1, 4, size=S)
        sp = SegProblems(np.stack([np.zeros(S, int), k1], 1), np.sort(rng.integers(0, k1.sum() + 1, size=3)),
                         ["a", "b", "c"])
        ctx = make_seg_ctx(w, np.full((S, 2), 100), sp, 0.05, 0.05, np.arange(1, 21), 1000)
        out = rect_dp_certificate(ctx, mu, lo, hi)
        pols = _pols(S, 2)
        for q in range(3):
            _, ph = dp_argmax(w[:, None] * mu, sp.cost, int(sp.budgets[q]))
            D = pols != ph[None, :]
            up = (w[None, :] * np.where(D, hi[np.arange(S)[None, :], pols] - lo[np.arange(S), ph][None, :], 0)).sum(1)
            feas = sp.cost[np.arange(S)[None, :], pols].sum(1) <= sp.budgets[q]
            assert abs(out[q][2] - max(0.0, up[feas].max())) < 1e-12
            assert out[q][1] == tuple(int(x) for x in ph)


# ------------------------------------------------------------------------------------------------ toy FWER
def _toy_env(seed, S=6):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(600, 2000, size=(S, 2))
    base = rng.uniform(0.2, 0.6, size=S)
    up = rng.normal(0.0, 0.08, size=S)
    mu = np.stack([base, np.clip(base + up, 0.01, 0.99)], 1)
    env = SyntheticPoolEnv(sizes, mu, seed=seed, n_min=200)
    env.problems = seg_problems(env.w, budgets=(0.2, 0.35, 0.5, 0.65, 0.8))
    return env


class _Invalid(FDCDP):
    """INVALID power control: beta forced to 0.5 (no union, sub-nominal exponent)."""
    validity = "none"

    def setup(self, ctx):
        super().setup(ctx)
        self.ledger["beta"] = 0.5


def test_toy_monte_carlo_fwer_with_invalid_control():
    eps = 0.02
    makers = {"FDC-DP(b)": lambda: FDCDP("b"), "FDC-DP(a)": lambda: FDCDP("a"), "RECT-BF-DP": lambda: RectBFDP(),
              "INVALID": lambda: _Invalid("a", name="INVALID")}
    false = {k: 0 for k in makers}
    n = 0
    cert = {k: 0 for k in makers}
    n80 = {k: [] for k in makers}
    t0 = time.perf_counter()
    for sd in range(900, 912):
        env = _toy_env(sd)
        Js, mu = true_opt(env)
        ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, eps, 0.05, env.checkpoints(20), env.tau_R)
        for p in range(8):
            n += 1
            for k, f in makers.items():
                s, _ = run_stream_seg(env, f(), 1000 * sd + p, ctx, Js, mu)
                false[k] += int(s["fwer_event"])
                cert[k] += s["n_cert"]
                n80[k].append(s["N80_raw"])
    print({"streams": n, "false": false, "certs": cert, "sec": round(time.perf_counter() - t0, 1)})
    for k in ("FDC-DP(b)", "FDC-DP(a)", "RECT-BF-DP"):
        assert false[k] == 0, (k, false)
    assert false["INVALID"] >= 1, false               # the toy can detect a broken certificate
    # same schedule and data: (b) certifies a superset of (a) at every checkpoint, so it never stops later
    assert all(b <= a for a, b in zip(n80["FDC-DP(a)"], n80["FDC-DP(b)"]))
    print({k: float(np.exp(np.mean(np.log(v)))) for k, v in n80.items()})


def test_tu_wrapper_identical_on_block_grid():
    env = _toy_env(905)
    Js, mu = true_opt(env)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, 0.02, 0.05, env.checkpoints(20), env.tau_R)
    for seed in (1, 2):
        s1, r1 = run_stream_seg(env, FDCDP("b"), seed, ctx, Js, mu)
        s2, r2 = run_stream_seg(env, FDCDPTimeUniform(ctx.checkpoints, scheme="b"), seed, ctx, Js, mu)
        assert s1["N80_pen"] == s2["N80_pen"] and s1["cert_k"] == s2["cert_k"]
        for a, b in zip(r1, r2):
            assert a["U"] == b["U"]


# ------------------------------------------------------------------------------------------------ runtime at S = 64
def test_runtime_per_checkpoint_S64():
    rng = np.random.default_rng(64)
    S = 64
    sizes = rng.integers(400, 700, size=(S, 2))                # X5-SR64-like pools (~545 per cell)
    base = rng.uniform(0.55, 0.68, size=S)
    mu = np.stack([base, np.clip(base + rng.normal(0.02, 0.04, size=S), 0.01, 0.99)], 1)
    env = SyntheticPoolEnv(sizes, mu, seed=64, n_min=2000)
    env.problems = seg_problems(env.w)
    Js, mut = true_opt(env)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, 0.02, 0.05, env.checkpoints(20), env.tau_R)
    ctx.stop_k = 16                                            # never stop early: time every checkpoint
    m = FDCDP("b")
    s, rows = run_stream_seg(env, m, 7, ctx, Js, mut)
    secs = [r["sec_cert"] for r in rows]
    print({"S": S, "K": len(secs), "median_sec": float(np.median(secs)), "max_sec": float(np.max(secs)),
           "nodes": sum(x["nodes"] for x in m.cert_stats), "node_limit_hits": sum(x["node_limit_hits"]
                                                                                  for x in m.cert_stats),
           "beta": m.ledger["beta"], "cols_max": max(x["n_cols"] for x in m.cert_stats)})
    assert np.median(secs) < 10.0 and s["n_false"] == 0


def test_reduce_costs():
    c, b, g = reduce_costs(np.array([[0, 8], [0, 16]]), np.array([7, 8, 23]))
    assert g == 8 and c.tolist() == [[0, 1], [0, 2]] and b.tolist() == [0, 1, 2]


# ------------------------------------------------------------------------------------------------ external reviewer FDC-DP r1 fixes
def _reviewer_counterexample():
    w = np.array([0.9, 0.1])
    mu = np.array([[0.0, 0.4], [0.0, 0.5]])
    sp = SegProblems(np.array([[0, 1], [0, 1]]), np.array([1]), ["B1"])
    N = np.full((2, 2), 50)
    ctx = make_seg_ctx(w, N, sp, 0.1, 0.05, np.arange(1, 21) * 10, 200)
    return ctx, mu, N


def test_b1_weighted_centre_reviewer_counterexample():
    """external reviewer r1 B1: w = (0.9, 0.1), control (0, 0), treatment (0.4, 0.5), unit costs, budget 1, all pools exhausted:
    the centre must be (1, 0) (value 0.36), not the unweighted argmax (0, 1) (value 0.05); eps = 0.1 certifies."""
    ctx, mu, N = _reviewer_counterexample()
    for scheme in ("a", "b"):
        m = FDCDP(scheme)
        m.setup(ctx)
        ok, pi, U = m.certify_with(ctx, mu, N, N, mu.copy(), mu.copy())[0]
        assert pi == (1, 0) and ok and U <= 0.1
    ok, pi, U = rect_dp_certificate(ctx, mu, mu.copy(), mu.copy())[0]
    assert pi == (1, 0) and ok and U == 0.0


def test_b1_end_to_end_weighted_centre_equals_enumeration():
    """End to end through FDCDP.certify_with (unequal weights): centre = enumerated weighted argmax, decision =
    enumeration FDC-BF on the same lambda set, scheme (a) never certifies more; RECT-BF-DP centre likewise."""
    rng = np.random.default_rng(2024)
    n_ok = n_no = 0
    for it in range(40):
        S = int(rng.integers(3, 8))
        I = _instance(rng, S)
        I["w"] = rng.dirichlet(np.ones(S) * 0.7)                          # strongly unequal weights
        sp = SegProblems(I["cost"], I["B"], [f"q{j}" for j in range(len(I["B"]))])
        eps = float(rng.choice([0.01, 0.03, 0.08, 0.2]))
        ctx = make_seg_ctx(I["w"], I["N"], sp, eps, 0.05, np.arange(1, 21), 1000)
        mb, ma = FDCDP("b"), FDCDP("a")
        mb.setup(ctx)
        ma.setup(ctx)
        I["beta"] = mb.ledger["beta"]
        ob = mb.certify_with(ctx, I["mu"], I["n"], I["N"], I["lo"], I["hi"])
        oa = ma.certify_with(ctx, I["mu"], I["n"], I["N"], I["lo"], I["hi"])
        orc = rect_dp_certificate(ctx, I["mu"], I["lo"], I["hi"])
        pols = _pols(S, 2)
        J = (I["w"][None, :] * I["mu"][np.arange(S)[None, :], pols]).sum(1)
        cst = I["cost"][np.arange(S)[None, :], pols].sum(1)
        lam = lambda_grid(I["beta"], eps, I["w"], I["n"], I["N"], I["lo"], I["hi"])
        for q, cap in enumerate(I["B"]):
            feas = cst <= cap
            ph = pols[np.flatnonzero(feas)[np.argmax(J[feas])]]
            assert ob[q][1] == tuple(int(x) for x in ph) == oa[q][1] == orc[q][1]
            ref = _fdcbf_same_grid_enum(I, ph, lam, int(cap), pols)
            if mb.last_detail[q]["decision"] != "abstain_node_limit":
                assert ob[q][0] == bool(max(ref, 0.0) <= eps)
            assert (not oa[q][0]) or ob[q][0]
            n_ok += int(ob[q][0])
            n_no += int(not ob[q][0])
    assert n_ok > 0 and n_no > 0


def _branch_only_case():
    """A (ctx, cell data, q) where scheme (a) fails but scheme (b) certifies (searched deterministically)."""
    rng = np.random.default_rng(77)
    for it in range(4000):
        S = int(rng.integers(4, 8))
        I = _instance(rng, S)
        if (I["n"] <= 0).any():
            continue
        sp = SegProblems(I["cost"], I["B"], [f"q{j}" for j in range(len(I["B"]))])
        for eps in (0.02, 0.05, 0.1):
            ctx = make_seg_ctx(I["w"], I["N"], sp, eps, 0.05, np.arange(1, 21), 1000)
            m = FDCDP("b", exact_U=True)
            m.setup(ctx)
            out = m.certify_with(ctx, I["mu"], I["n"], I["N"], I["lo"], I["hi"])
            for q, d in enumerate(m.last_detail):
                if d["decision"] == "certified_b":
                    return ctx, I, q, out, m.last_detail
    raise AssertionError("no branch-only certificate found")


def test_b3_branch_only_certificate_output_contract():
    ctx, I, q, out, det = _branch_only_case()
    d = det[q]
    ok, pi, U = out[q]
    assert ok and d["decision"] == "certified_b"
    assert U == d["U_a"] and d["U_a"] > ctx.eps                 # third element = scheme-(a) bound, NOT exact
    assert d["U_exact"] is not None and d["U_exact"] <= ctx.eps < d["U_a"]


def test_b3_forced_node_cap_abstains():
    ctx, I, q, _, _ = _branch_only_case()
    m = FDCDP("b", node_limit=1)
    m.setup(ctx)
    out = m.certify_with(ctx, I["mu"], I["n"], I["N"], I["lo"], I["hi"])
    d = m.last_detail[q]
    assert d["decision"] == "abstain_node_limit" and not out[q][0]
    assert sum(x["node_limit_hits"] for x in m.cert_stats) >= 1


def test_b3_violator_lower_bound():
    rng = np.random.default_rng(5)
    seen = 0
    for it in range(200):
        I = _instance(rng, 6)
        sp = SegProblems(I["cost"], I["B"], ["a", "b", "c"])
        ctx = make_seg_ctx(I["w"], I["N"], sp, 0.01, 0.05, np.arange(1, 21), 1000)
        m = FDCDP("b", exact_U=True)
        m.setup(ctx)
        m.certify_with(ctx, I["mu"], I["n"], I["N"], I["lo"], I["hi"])
        for d in m.last_detail:
            if d["decision"] == "violator_b" and d["U_exact"] is not None and d["U_lower"] is not None:
                assert d["U_lower"] <= d["U_exact"] + 1e-12 and d["U_exact"] <= d["U_a"] + 1e-12
                assert d["U_lower"] > ctx.eps
                seen += 1
    assert seen > 0


def test_b4_fresh_width_certify_rejects_off_grid_times():
    from dsswm.baselines.frontier_common import FrontierState
    env = _toy_env(906)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, 0.02, 0.05, env.checkpoints(20), env.tau_R)
    ck = ctx.checkpoints
    n = np.minimum(env.pool_sizes, 50)
    st = lambda t: FrontierState(t, n, n * 0.4, env.pool_sizes, np.ones(ctx.Q, bool), None)   # noqa: E731
    for m in (FDCDP("b"), FDCDP("a"), RectBFDP()):
        m.setup(ctx)
        m.certify(ctx, st(int(ck[0])))
        with pytest.raises(RuntimeError):
            m.certify(ctx, st(int(ck[0]) + 1))                   # off the predeclared grid
        with pytest.raises(RuntimeError):
            m.certify(ctx, st(int(ck[0])))                       # not strictly increasing
        m.certify(ctx, st(int(ck[1])))
    bad = make_seg_ctx(env.w, env.pool_sizes, env.problems, 0.02, 0.05, ck[::-1], env.tau_R)
    with pytest.raises(ValueError):
        FDCDP("b").setup(bad)
    with pytest.raises(ValueError):
        RectBFDP().setup(bad)


def test_b2_stream_horizon_to_15_and_n80_at_12():
    env = _toy_env(907)
    Js, mu = true_opt(env)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, 0.02, 0.05, env.checkpoints(20), env.tau_R,
                       stop_frac=1.0)
    s, rows = run_stream_seg(env, FDCDP("b"), 3, ctx, Js, mu)
    assert s["stop_k"] == ctx.Q and s["n80_k"] == int(math.ceil(0.8 * ctx.Q))
    if s["k80"] is not None:
        assert rows[s["k80"]]["n_cert"] >= s["n80_k"] and (s["k80"] == 0 or rows[s["k80"] - 1]["n_cert"] < s["n80_k"])
        assert s["exhaustion_at_k80"] is not None
    assert s["n80_lt_tau"] == (s["N80_pen"] < s["tau_R"])
    assert len(rows) >= (s["k80"] or 0) + 1
