"""Tests for FDC-PW (prior-weighted FDC-DP ledger; plan/fdc_pw_theory.md)."""
from __future__ import annotations

import itertools
import math

import numpy as np

from dsswm.baselines.fdc_dp import FDCDP, FDCDPTimeUniform, count_policies, make_seg_ctx, reduce_costs
from dsswm.baselines.fdc_pw import FDCPW, FDCPWTimeUniform, shell_counts, shell_rho
from dsswm.envs.seg_v10 import run_stream_seg, true_opt
from dsswm.tests.test_fdc_dp import _toy_env


def _enum(cost, cap):
    S, A = cost.shape
    for pi in itertools.product(range(A), repeat=S):
        if sum(cost[s, a] for s, a in enumerate(pi)) <= cap:
            yield np.asarray(pi)


def test_shell_counts_match_enumeration_and_total():
    rng = np.random.default_rng(3)
    for _ in range(40):
        S = int(rng.integers(2, 8))
        cost = np.zeros((S, 2), dtype=np.int64)
        cost[:, 1] = rng.integers(1, 5, size=S)
        if rng.random() < 0.3:
            cost[:, 0] = rng.integers(0, 3, size=S)
        cap = int(rng.integers(0, int(cost.max(1).sum()) + 1))
        ref = rng.integers(0, 2, size=S)
        Nd = shell_counts(cost, cap, ref)
        brute = np.zeros(S + 1)
        for pi in _enum(cost, cap):
            brute[int(np.sum(pi != ref))] += 1
        assert np.allclose(Nd, brute, rtol=1e-8, atol=0)
        assert (Nd >= brute).all()                                    # inflation is conservative
        tot = count_policies(cost, [cap])[0] if isinstance(count_policies(cost, [cap]), tuple) else None
        if tot is not None and isinstance(tot, (list, tuple)):
            assert abs(sum(tot) - brute.sum()) < 0.5


def test_rho_sums_to_one_and_weights_sum_at_most_one():
    for S in (1, 5, 16, 64):
        assert abs(shell_rho(S).sum() - 1.0) < 1e-12
    rng = np.random.default_rng(5)
    for _ in range(20):
        S = int(rng.integers(2, 8))
        cost = np.zeros((S, 2), dtype=np.int64)
        cost[:, 1] = rng.integers(1, 4, size=S)
        cap = int(rng.integers(0, int(cost[:, 1].sum()) + 1))
        ref = rng.integers(0, 2, size=S)
        Nd = shell_counts(cost, cap, ref)
        rho = shell_rho(S)
        tot = sum(rho[int(np.sum(pi != ref))] / Nd[int(np.sum(pi != ref))] for pi in _enum(cost, cap))
        assert tot <= 1.0 + 1e-12


def _ctx(env, eps=0.02):
    return make_seg_ctx(env.w, env.pool_sizes, env.problems, eps, 0.05, env.checkpoints(20), env.tau_R)


def test_uniform_weights_reduce_to_fdc_dp():
    """With beta_d forced to FDC-DP's beta_J and only the argmax centre, FDC-PW reproduces FDC-DP exactly."""
    env = _toy_env(907)
    Js, mu = true_opt(env)
    ctx = _ctx(env)

    class _Uniform(FDCPW):
        def setup(self, c):
            super().setup(c)
            for p in self.pw:
                p["beta_d"] = np.full_like(p["beta_d"], self.ledger["beta"])
            self._bminmax = (self.ledger["beta"], self.ledger["beta"])

    for seed in (1, 2, 3):
        s1, r1 = run_stream_seg(env, FDCDP("b"), seed, ctx, Js, mu)
        s2, r2 = run_stream_seg(env, _Uniform(np.zeros((env.S, 2)), centres="argmax", scheme="b"), seed, ctx, Js, mu)
        assert s1["N80_pen"] == s2["N80_pen"] and s1["cert_k"] == s2["cert_k"]


def test_beta_at_reference_is_small_and_worst_case_penalty_bounded():
    env = _toy_env(903, S=6)
    ctx = _ctx(env)
    m = FDCPW(np.zeros((env.S, 2)), scheme="a")
    m.setup(ctx)
    base = m.ledger["pw_base"]
    S = env.S
    for q, p in enumerate(m.pw):
        assert abs(p["beta_d"][0] - (base - math.log(shell_rho(S)[0]) + math.log(p["N_d"][0]))) < 1e-9
        cost, budgets, _ = reduce_costs(env.problems.cost, env.problems.budgets)
        Mq = sum(p["N_d"])
        fin = p["beta_d"][np.isfinite(p["beta_d"])]
        assert fin.max() <= base + math.log(Mq) + math.log(2 * (S + 1)) + 1e-9


class _InvalidPW(FDCPW):
    validity = "none"

    def setup(self, c):
        super().setup(c)
        for p in self.pw:
            p["beta_d"] = np.full_like(p["beta_d"], 0.5)
        self._bminmax = (0.5, 0.5)


def test_toy_monte_carlo_fwer_oracle_and_adversarial_reference():
    """Valid for ANY frozen reference: oracle (true uplift), adversarial (anti-uplift) and zero references all give
    no false stream; an invalid beta = 0.5 control is caught."""
    eps = 0.02
    false = {}
    n80 = {}
    for sd in range(900, 912):
        env = _toy_env(sd)
        Js, mu = true_opt(env)
        ctx = _ctx(env, eps)
        up = (mu[:, 1] - mu[:, 0])[:, None] * np.array([[0.0, 1.0]])
        makers = {"FDC-DP(b)": lambda: FDCDP("b"),
                  "PW-oracle": lambda: FDCPW(env.w[:, None] * up, scheme="b"),
                  "PW-adversarial": lambda: FDCPW(-env.w[:, None] * up, scheme="b"),
                  "PW-zero": lambda: FDCPW(np.zeros((env.S, 2)), scheme="b"),
                  "INVALID": lambda: _InvalidPW(np.zeros((env.S, 2)), scheme="a", name="INVALID")}
        for p in range(8):
            for k, f in makers.items():
                s, _ = run_stream_seg(env, f(), 1000 * sd + p, ctx, Js, mu)
                false[k] = false.get(k, 0) + int(s["fwer_event"])
                n80.setdefault(k, []).append(s["N80_raw"])
    print(false, {k: float(np.exp(np.mean(np.log(v)))) for k, v in n80.items()})
    for k in ("FDC-DP(b)", "PW-oracle", "PW-adversarial", "PW-zero"):
        assert false[k] == 0, (k, false)
    assert false["INVALID"] >= 1, false


def test_tu_pw_identical_on_block_grid():
    env = _toy_env(905)
    Js, mu = true_opt(env)
    ctx = _ctx(env)
    ref = np.zeros((env.S, 2))
    for seed in (1, 2):
        s1, r1 = run_stream_seg(env, FDCPW(ref, scheme="b"), seed, ctx, Js, mu)
        s2, r2 = run_stream_seg(env, FDCPWTimeUniform(ctx.checkpoints, ref, scheme="b"), seed, ctx, Js, mu)
        assert s1["N80_pen"] == s2["N80_pen"] and s1["cert_k"] == s2["cert_k"]
        for a, b in zip(r1, r2):
            assert a["U"] == b["U"]
    _ = FDCDPTimeUniform


def test_dp_argmax_ball_matches_brute_force():
    from dsswm.baselines.fdc_pw import dp_argmax_ball
    rng = np.random.default_rng(11)
    for _ in range(60):
        S = int(rng.integers(2, 8))
        cost = np.zeros((S, 2), dtype=np.int64)
        cost[:, 1] = rng.integers(1, 4, size=S)
        cap = int(rng.integers(0, int(cost[:, 1].sum()) + 1))
        ref = rng.integers(0, 2, size=S)
        vals = rng.normal(size=(S, 2))
        out = dp_argmax_ball(vals, cost, cap, ref, range(S + 1))
        for r in range(S + 1):
            best = max((vals[np.arange(S), pi].sum() for pi in _enum(cost, cap) if np.sum(pi != ref) <= r),
                       default=None)
            if best is None:
                assert out[r] is None
                continue
            pi = out[r]
            assert np.sum(pi != ref) <= r and sum(cost[s, a] for s, a in enumerate(pi)) <= cap
            assert abs(vals[np.arange(S), pi].sum() - best) < 1e-9
