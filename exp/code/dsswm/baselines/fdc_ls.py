"""FDC-LS: FDC-DP with a union ledger localised in Hamming shells around the (unknown) optimum (NEW file, v11;
plan/fdc_ls_theory.md).

FDC-DP's event for index (q, pi, k) concerns the direction (pi*_q, pi), where pi*_q is fixed though unknown.  A union
bound needs only weights that are fixed before the data, so they may depend on pi*_q.  Allocate

    w_q(pi) = 1/2 * 1/M  +  1/2 * rho_d / (Q * C(S, d) (A-1)^d),     d = Hamming(pi, pi*_q) >= 1,

with M = sum_q |Pi_q| and rho_d = 0.5 * 2^-d / (1 - 2^-S) + 0.5 / S over d = 1..S.  Since #{pi : Hamming(pi, pi*) = d}
<= C(S, d) (A-1)^d, sum_{q, pi} w_q(pi) <= 1/2 + 1/2 = 1 (the centre pi*_q itself never needs an event).  Index level:
delta_main * w_q(pi) / K, exponent

    beta_d = ln(K / delta_main) - ln(1/(2M) + rho_d / (2 Q C(S, d) (A-1)^d))  <=  min(beta_J, beta_shell_d) + ln 2.

In a certificate with centre pi_hat, the event used is the one for index pi_hat, i.e. d = Hamming(pi_hat, pi*_q); the
challenger playing pi*_q differs from pi_hat in exactly d segments.  So evaluating each challenger pi with
beta_{Hamming(pi, pi_hat)} gives U >= the value at pi = pi*_q, which bounds J(pi*_q) - J(pi_hat) on the event: validity
as in Theorem DP-1.  The challenger-dependent exponent is handled exactly by a knapsack DP with a flip-count state
(d = 1..D, plus one merged state "> D" that uses the conservative bound beta_J + ln 2 >= beta_d for every d > D, where D is
the largest d with beta_shell_d + ln 2 < beta_J + ln 2).  Scheme (a) only (min over lambda outside the max): conservative.
"""
from __future__ import annotations

import math

import numpy as np

from .fdc_dp import (FDCDP, FDCDPTimeUniform, _NEG, cell_tables, dp_argmax, joint_columns, lambda_grid,
                     reduce_costs)


def ls_rho(S):
    d = np.arange(1, S + 1, dtype=float)
    return 0.5 * 2.0 ** (-d) / (1.0 - 2.0 ** (-S)) + 0.5 / S


def ls_betas(S, A, Q, K, M, delta_main):
    """beta_d for d = 1..S (index 0 unused, = nan) and D (largest d with a local exponent below beta_J + ln 2)."""
    rho = ls_rho(S)
    beta = np.full(S + 1, np.nan)
    base = math.log(K / delta_main)
    for d in range(1, S + 1):
        lc = math.lgamma(S + 1) - math.lgamma(d + 1) - math.lgamma(S - d + 1) + d * math.log(A - 1)
        # ln(1/(2M) + rho_d / (2 Q C)) computed stably
        a = -math.log(2 * M)
        b = math.log(rho[d - 1]) - math.log(2 * Q) - lc
        m = max(a, b)
        beta[d] = base - (m + math.log(math.exp(a - m) + math.exp(b - m)))
    beta_far = base + math.log(2 * M)                 # >= beta_d for all d (drops the shell term)
    loc = [d for d in range(1, S + 1) if beta[d] < beta_far - 1e-12]
    D = max(loc) if loc else 0
    return beta, beta_far, D


def _shift_max3(dst, src, t, k):
    """dst[..., k:] = max(dst[..., k:], src[..., :C+1-k] + t[None, :, None]) on (Dn, J, C+1) arrays."""
    C1 = dst.shape[-1]
    if k >= C1:
        return
    with np.errstate(invalid="ignore"):
        v = src[..., :C1 - k] + t[None, :, None]
    v = np.where(np.isnan(v), _NEG, v)
    np.maximum(dst[..., k:], v, out=dst[..., k:])


def dp_max_cols_shell(T, cost, diff, cap, D):
    """F[d, j, c], d = 0..D+1: max over pi with sum cost <= c of sum_s T[s, pi(s), j], with exactly d differing segments
    for d <= D and > D differing segments in the last slot (d = 0 slot = the centre path)."""
    S, A, J = T.shape
    F = np.full((D + 2, J, cap + 1), _NEG)
    F[0] = 0.0
    for s in range(S):
        N = np.full_like(F, _NEG)
        for a in range(A):
            k = int(cost[s, a])
            t = T[s, a]
            if diff[s, a]:
                _shift_max3(N[1:D + 1], F[0:D], t, k)                      # d -> d + 1 (d + 1 <= D)
                _shift_max3(N[D + 1:D + 2], F[D:D + 2].max(axis=0, keepdims=True), t, k)   # -> "> D"
            else:
                _shift_max3(N, F, t, k)
        F = N
    return F


class _LSMixin:
    def _ls_setup(self, ctx):
        K = int(self.ledger.get("K", len(self._grid_set)))
        M = int(self.ledger["union_size"])
        beta_d, beta_far, D = ls_betas(ctx.S, ctx.A, ctx.Q, K, M, self.split[0])
        self._ls = (beta_d, beta_far, D)
        self.ledger.update({"ls_D": int(D), "ls_beta_1": float(beta_d[1]), "ls_beta_far": float(beta_far),
                            "ls_beta_D": float(beta_d[D]) if D else None,
                            "ls_rule": "w = 1/2 uniform + 1/2 rho_d / (Q C(S,d)(A-1)^d), d = Hamming to pi*_q"})
        self._bmin = float(beta_d[1]) if D else beta_far

    def _ls_lam(self, ctx, n, N, lo, hi):
        l1 = lambda_grid(self._bmin, ctx.eps, ctx.w, n, N, lo, hi, self.grid_ratio)
        l2 = lambda_grid(self._ls[1], ctx.eps, ctx.w, n, N, lo, hi, self.grid_ratio)
        return np.unique(np.concatenate([l1, l2]))

    def certify_with(self, ctx, mu, n, N, lo, hi, undecided=None, rect=None, t=None, tables=None):
        import time
        if rect is not None:
            raise NotImplementedError
        t0 = time.perf_counter()
        sp = ctx.sp
        cost, budgets, _ = reduce_costs(sp.cost, sp.budgets)
        if tables is None:
            lam = self._ls_lam(ctx, n, N, lo, hi)
            P, det = cell_tables(ctx.w, n, N, lo, hi, lam)
        else:
            lam, P, det = tables
        beta_d, beta_far, D = self._ls
        vals = np.asarray(ctx.w, float)[:, None] * np.asarray(mu, float)
        out = [None] * ctx.Q
        groups = {}
        for q in range(ctx.Q):
            _, ph = dp_argmax(vals, sp.cost, int(sp.budgets[q]))
            if ph is None:
                raise RuntimeError(f"problem {q}: empty policy class")
            if undecided is not None and not undecided[q]:
                out[q] = (False, tuple(int(x) for x in ph), float("nan"))
                continue
            groups.setdefault(tuple(int(x) for x in ph), []).append(q)
        st = {"t": t, "n_centres": len(groups), "a_certified": 0, "b_certified": 0, "nodes": 0, "bnb_calls": 0,
              "node_limit_hits": 0}
        for key, qs in groups.items():
            ph = np.asarray(key, dtype=np.int64)
            const, T, diff = joint_columns(ctx.w, mu, ph, P, det, lam, 0.0)     # beta enters per shell below
            capmax = max(int(budgets[q]) for q in qs)
            F = dp_max_cols_shell(T, cost, diff, capmax, D)                     # (D+2, J, cap+1)
            nl = len(lam)
            consts = np.zeros((D + 2, T.shape[2]))
            for d in range(1, D + 1):
                consts[d, :nl] = beta_d[d] / lam
            consts[D + 1, :nl] = beta_far / lam
            for q in qs:
                c = int(budgets[q])
                V = F[1:, :, c] + consts[1:]                     # -inf (infeasible) + finite = -inf; +inf stays
                U = max(0.0, float(np.min(V.max(axis=0))))
                ok = U <= ctx.eps
                if ok:
                    st["a_certified"] += 1
                out[q] = (bool(ok), key, float(U))
        st["sec"] = round(time.perf_counter() - t0, 4)
        self.cert_stats.append(st)
        return out


class FDCLS(_LSMixin, FDCDP):
    """FDC-LS(a): checkpoint-valid shell-localised FDC-DP (scheme (a))."""

    def __init__(self, **kw):
        kw.setdefault("name", "FDC-LS(a)")
        super().__init__(scheme="a", **kw)

    def setup(self, ctx):
        super().setup(ctx)
        self.ledger["K"] = len(self._grid_set)
        self._ls_setup(ctx)


class FDCLSTimeUniform(_LSMixin, FDCDPTimeUniform):
    """TU-FDC-LS(a): Lemma TU reading (widths frozen at block points)."""

    def __init__(self, block_points, **kw):
        kw.setdefault("name", "TU-FDC-LS(a)")
        super().__init__(block_points, scheme="a", **kw)

    def setup(self, ctx):
        super().setup(ctx)
        self.ledger["K"] = len(self.block_points)
        self._ls_setup(ctx)

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("evaluation times must be increasing")
        self._last_t = st.t
        while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
            if int(self.block_points[self._next_block]) != st.t:
                raise RuntimeError("block grid must be a subset of the evaluation grid")
            lo, hi = self._box(st)
            lam = self._ls_lam(ctx, st.n, st.N, lo, hi)
            P, det = cell_tables(ctx.w, st.n, st.N, lo, hi, lam)
            self._snap = (np.asarray(st.n).copy(), np.asarray(st.N).copy(), lo, hi, (lam, P, det))
            self._next_block += 1
        if self._snap is None:
            return [(False, None, float("inf")) for _ in range(ctx.Q)]
        n_k, N_k, lo, hi, tables = self._snap
        return self.certify_with(ctx, st.mu_hat, n_k, N_k, lo, hi, getattr(st, "undecided", None), t=st.t,
                                 tables=tables)
