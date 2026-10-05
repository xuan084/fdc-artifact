"""Tests for FDC-LS (shell-localised ledger; plan/fdc_ls_theory.md)."""
from __future__ import annotations

import itertools
import math

import numpy as np

from dsswm.baselines.fdc_dp import FDCDP, count_policies, dp_max_cols, joint_columns, make_seg_ctx
from dsswm.baselines.fdc_ls import FDCLS, FDCLSTimeUniform, dp_max_cols_shell, ls_betas, ls_rho
from dsswm.envs.seg_v10 import run_stream_seg, true_opt
from dsswm.tests.test_fdc_dp import _toy_env


def test_rho_and_weights_sum_at_most_one_for_every_optimum():
    for S in (1, 4, 16, 64):
        assert abs(ls_rho(S).sum() - 1.0) < 1e-12
    rng = np.random.default_rng(2)
    for _ in range(15):
        S = int(rng.integers(2, 7))
        A = 2
        Q, K, dm = 3, 20, 0.045
        M = Q * 2 ** S
        beta, beta_far, D = ls_betas(S, A, Q, K, M, dm)
        for star in itertools.product(range(A), repeat=S):
            tot = 0.0
            for pi in itertools.product(range(A), repeat=S):
                d = sum(x != y for x, y in zip(pi, star))
                if d:
                    tot += Q * math.exp(-beta[d]) * K / dm          # Q problems x K checkpoints, normalised
            assert tot <= 1.0 + 1e-9
        assert all(beta[d] <= beta_far + 1e-12 for d in range(1, S + 1))


def test_dp_max_cols_shell_matches_brute_force():
    rng = np.random.default_rng(4)
    for _ in range(40):
        S, A, J = int(rng.integers(2, 7)), 2, 3
        T = rng.normal(size=(S, A, J))
        cost = np.zeros((S, A), dtype=np.int64)
        cost[:, 1] = rng.integers(1, 3, size=S)
        cap = int(rng.integers(0, int(cost[:, 1].sum()) + 1))
        ph = rng.integers(0, 2, size=S)
        diff = np.ones((S, A), bool)
        diff[np.arange(S), ph] = False
        T[np.arange(S), ph] = 0.0
        D = int(rng.integers(0, S + 1))
        F = dp_max_cols_shell(T, cost, diff, cap, D)
        for c in range(cap + 1):
            for j in range(J):
                best = [-np.inf] * (D + 2)
                for pi in itertools.product(range(A), repeat=S):
                    if sum(cost[s, a] for s, a in enumerate(pi)) > c:
                        continue
                    d = sum(pi[s] != ph[s] for s in range(S))
                    slot = d if d <= D else D + 1
                    best[slot] = max(best[slot], sum(T[s, pi[s], j] for s in range(S)))
                for slot in range(D + 2):
                    assert (np.isneginf(F[slot, j, c]) and np.isneginf(best[slot])) or abs(F[slot, j, c] - best[slot]) < 1e-9
        # merged over shells it reproduces FDC-DP's challenger maximum
        f1 = dp_max_cols(T, cost, diff, cap)
        assert np.allclose(np.where(np.isfinite(f1), f1, -1e300), np.where(np.isfinite(F[1:].max(0)), F[1:].max(0), -1e300))


def _ctx(env, eps=0.02):
    return make_seg_ctx(env.w, env.pool_sizes, env.problems, eps, 0.05, env.checkpoints(20), env.tau_R)


class _LSNoShell(FDCLS):
    """D = 0: every challenger uses beta_far = beta_J + ln 2."""

    def _ls_setup(self, ctx):
        super()._ls_setup(ctx)
        b, bf, _ = self._ls
        self._ls = (b, bf, 0)
        self._bmin = bf


class _DPa(FDCDP):
    def setup(self, ctx):
        super().setup(ctx)
        self.ledger["beta"] = self.ledger["beta"] + math.log(2)


def test_no_shell_reduces_to_fdc_dp_a_with_ln2():
    env = _toy_env(907)
    Js, mu = true_opt(env)
    ctx = _ctx(env)
    for seed in (1, 2):
        s1, r1 = run_stream_seg(env, _DPa("a"), seed, ctx, Js, mu)
        s2, r2 = run_stream_seg(env, _LSNoShell(), seed, ctx, Js, mu)
        assert s1["N80_pen"] == s2["N80_pen"] and s1["cert_k"] == s2["cert_k"]


class _InvalidLS(FDCLS):
    validity = "none"

    def _ls_setup(self, ctx):
        super()._ls_setup(ctx)
        b, bf, D = self._ls
        self._ls = (np.where(np.isnan(b), b, 0.5), 0.5, D)
        self._bmin = 0.5


def test_toy_monte_carlo_fwer():
    eps = 0.02
    false, n80 = {}, {}
    for sd in range(900, 912):
        env = _toy_env(sd)
        Js, mu = true_opt(env)
        ctx = _ctx(env, eps)
        makers = {"FDC-DP(a)": lambda: FDCDP("a"), "FDC-LS(a)": lambda: FDCLS(),
                  "INVALID": lambda: _InvalidLS(name="INVALID")}
        for p in range(8):
            for k, f in makers.items():
                s, _ = run_stream_seg(env, f(), 1000 * sd + p, ctx, Js, mu)
                false[k] = false.get(k, 0) + int(s["fwer_event"])
                n80.setdefault(k, []).append(s["N80_raw"])
    print(false, {k: float(np.exp(np.mean(np.log(v)))) for k, v in n80.items()})
    assert false["FDC-DP(a)"] == 0 and false["FDC-LS(a)"] == 0
    assert false["INVALID"] >= 1


def test_tu_ls_identical_on_block_grid():
    env = _toy_env(905)
    Js, mu = true_opt(env)
    ctx = _ctx(env)
    for seed in (1, 2):
        s1, r1 = run_stream_seg(env, FDCLS(), seed, ctx, Js, mu)
        s2, r2 = run_stream_seg(env, FDCLSTimeUniform(ctx.checkpoints), seed, ctx, Js, mu)
        assert s1["N80_pen"] == s2["N80_pen"] and s1["cert_k"] == s2["cert_k"]
    _ = (count_policies, joint_columns)
