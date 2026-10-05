"""Shared truth-side runner for the round-4 real-layer frontier problems (CR9 primary, HR8 secondary, CR12 / toy).

Protocol (methodology 1.2 / 1.5 / R1, identical for every method)
-----------------------------------------------------------------
* Pool protocol of ``envs.pool_replay``: a complete schedule is pre-generated from the permutation seed and labels only
  (arrival segments, within-pool record orders; the planned arm for 'pool' / 'fixed' allocations). Adaptive methods
  pick arms online, but only at batch boundaries: every ``replan`` arrivals (lock constant
  max(200, ceil(tau_R / 2000))) the method returns a per-segment arm preference order; all arrivals of segment s in
  the batch go to the first non-exhausted arm in that order (the method's own re-selection rule). Fixed allocations
  re-select exhausted arms with the pre-drawn uniform (p[s, :] renormalised over the non-exhausted arms), exactly as
  ``ReplayStream``. A fully exhausted segment's arrival is skipped. Every arrival is billed.
* K = 20 pre-fixed log-spaced checkpoints (env.checkpoints). Methods certify only there. A problem's answer is frozen
  at its first certification (sticky). The stream stops at the first checkpoint with >= stop_k (12/15) certified
  problems. N80 = that checkpoint if no certification so far is eps-wrong, else tau_R (censored); never reaching
  stop_k -> tau_R (censored).
* Per-cell WSR20 WoR empirical-Bernstein CSs (``stats.fp_eb.wor_mean_cs``) are computed once per stream over each
  pool's full record order (the CS at count n depends on the first n draws only), two-sided miscoverage
  delta_cell = delta / (S A) per cell; they are exposed to methods via ``FrontierState.cs()``.

Engine exactness: the batched engine is checked against the arrival-by-arrival ``ReplayStream`` in
``tests/test_baselines_r4a.py`` (counts, sums, billing counters identical).

Output rows: one per (stream, method, checkpoint reached): see ``run_stream``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..baselines.frontier_common import FrontierState, make_ctx
from ..envs.pool_replay import PoolReplayEnv, make_schedule
from ..stats.fp_eb import wor_mean_cs
from . import frontier as fr

__all__ = ["StreamEngine", "run_stream", "SyntheticPoolEnv", "true_policy_values", "write_jsonl", "build_ctx",
           "make_frontier_method", "BASELINES_B"]

_CHUNK = 65_536


# =============================================================================================== engine
class StreamEngine:
    def __init__(self, env, sch, outcome, delta_cell, R=1.0):
        self.env, self.sch = env, sch
        self.S, self.A = env.S, env.A
        self.N = env.pool_sizes.astype(np.int64)
        self.R = float(R)
        self.delta_cell = float(delta_cell)
        y = env.outcomes_view(outcome)
        C = self.S * self.A
        self._obs = []
        self._cum = []
        for c in range(C):
            o = np.asarray(y[env.pool_rows[c][sch.pool_perm[c]]], dtype=float)
            self._obs.append(o)
            self._cum.append(np.concatenate([[0.0], np.cumsum(o)]))
        self._cs_lo = [None] * C
        self._cs_hi = [None] * C
        self.seg_seq = np.asarray(sch.seg_seq, dtype=np.int64)
        self.arm_seq = np.asarray(sch.arm_seq, dtype=np.int64)
        self.resel_u = sch.resel_u
        self.kind = sch.alloc
        self.alloc_p = sch.alloc_p
        self._seg_pos = None
        self.n = np.zeros((self.S, self.A), dtype=np.int64)
        self.sum = np.zeros((self.S, self.A))
        self.t = 0
        self.billed = 0
        self.n_served = 0
        self.n_skipped = 0
        self.n_reselected = 0
        self.n_loop_arrivals = 0

    # ------------------------------------------------------------------ CS
    def _cell_cs(self, c):
        if self._cs_lo[c] is None:
            o = self._obs[c]
            if o.size:
                lo, hi = wor_mean_cs(o, int(o.size), self.delta_cell, R=self.R)
            else:
                lo = hi = np.zeros(0)
            self._cs_lo[c] = np.concatenate([[0.0], lo])
            self._cs_hi[c] = np.concatenate([[self.R], hi])
        return self._cs_lo[c], self._cs_hi[c]

    def cs_at(self, n):
        lo = np.zeros((self.S, self.A))
        hi = np.zeros((self.S, self.A))
        for c in range(self.S * self.A):
            s, a = divmod(c, self.A)
            L, H = self._cell_cs(c)
            lo[s, a], hi[s, a] = L[n[s, a]], H[n[s, a]]
        return lo, hi

    # ------------------------------------------------------------------ bookkeeping
    def _take(self, s, a, m):
        c = s * self.A + a
        n0 = self.n[s, a]
        self.sum[s, a] += self._cum[c][n0 + m] - self._cum[c][n0]
        self.n[s, a] = n0 + m

    def remaining(self):
        return self.N - self.n

    def billing_ok(self):
        return bool(self.billed == self.t and self.n_served + self.n_skipped == self.billed and
                    int(self.n.sum()) == self.n_served and (self.n <= self.N).all())

    # ------------------------------------------------------------------ adaptive batches
    def _seg_counts(self, t0, t1):
        if self._seg_pos is None:
            self._seg_pos = [np.flatnonzero(self.seg_seq == s) for s in range(self.S)]
        return np.array([np.searchsorted(p, t1) - np.searchsorted(p, t0) for p in self._seg_pos], dtype=np.int64)

    def advance_adaptive(self, t1, pref):
        t1 = min(int(t1), self.env.tau_R)
        m_all = self._seg_counts(self.t, t1)
        pref = np.asarray(pref, dtype=np.int64)
        for s in range(self.S):
            m = int(m_all[s])
            first = True
            for a in pref[s]:
                if m <= 0:
                    break
                r = int(self.N[s, a] - self.n[s, a])
                take = min(m, r)
                if take > 0:
                    self._take(s, int(a), take)
                    self.n_served += take
                    if not first:
                        self.n_reselected += take
                    m -= take
                first = False
            self.n_skipped += m
        self.billed += t1 - self.t
        self.t = t1

    # ------------------------------------------------------------------ fixed / pool schedules
    def _resel(self, s, avail, u):
        p = self.alloc_p[s, avail] if self.alloc_p is not None else np.ones(len(avail))
        p = p / p.sum() if p.sum() > 0 else np.full(len(avail), 1.0 / len(avail))
        return avail[min(int((u > np.cumsum(p)).sum()), len(avail) - 1)]

    def _loop(self, t0, t1):
        for t in range(t0, t1):
            s = int(self.seg_seq[t])
            avail = [a for a in range(self.A) if self.n[s, a] < self.N[s, a]]
            self.billed += 1
            if not avail:
                self.n_skipped += 1
                continue
            a = int(self.arm_seq[t])
            if self.n[s, a] >= self.N[s, a]:
                self.n_reselected += 1
                a = self._resel(s, avail, float(self.resel_u[t]))
            self._take(s, a, 1)
            self.n_served += 1
        self.n_loop_arrivals += t1 - t0

    def advance_planned(self, t1):
        t1 = min(int(t1), self.env.tau_R)
        size = _CHUNK
        while self.t < t1:
            b = min(self.t + size, t1)
            seg = self.seg_seq[self.t:b]
            arm = self.arm_seq[self.t:b].copy()
            exh = self.n >= self.N
            n_resel = 0
            skip = np.zeros(b - self.t, dtype=bool)
            if exh.any():
                bad = exh[seg, arm]
                if bad.any():
                    for s in np.unique(seg[bad]):
                        idx = np.flatnonzero(bad & (seg == s))
                        avail = [a for a in range(self.A) if not exh[s, a]]
                        if not avail:
                            skip[idx] = True
                            continue
                        arm[idx] = [self._resel(int(s), avail, float(u)) for u in self.resel_u[self.t + idx]]
                        n_resel += idx.size
            cell = seg * self.A + arm
            cnt = np.bincount(cell[~skip], minlength=self.S * self.A).reshape(self.S, self.A)
            if np.any(self.n + cnt > self.N):
                if b - self.t > 1024:            # a pool runs dry inside this chunk: shrink, then exact loop
                    size = max(1024, (b - self.t) // 2)
                    continue
                self._loop(self.t, b)
                size = _CHUNK
            else:
                for c in np.flatnonzero(cnt.ravel()):
                    s, a = divmod(int(c), self.A)
                    self._take(s, a, int(cnt[s, a]))
                self.n_served += int(cnt.sum())
                self.n_skipped += int(skip.sum())
                self.n_reselected += int(n_resel)
                self.billed += b - self.t
            self.t = b


# =============================================================================================== runner
def true_policy_values(pols, w, mu):
    return fr.policy_values(pols, w, mu)


def build_ctx(env, problems, eps, delta=0.05, K=20, binary=True, R=1.0, pols=None):
    return make_ctx(env.w, env.pool_sizes, problems, eps, delta, env.checkpoints(K), env.replan_interval, env.tau_R,
                    binary=binary, R=R, pols=pols)


def run_stream(env, method, perm_seed, problems, eps, outcome="visit", delta=0.05, K=20, ctx=None, J_true=None,
               J_star=None, keep_U=True):
    """Run one method on one stream. Returns (summary dict, rows list).

    J_true: (P,) true policy values (truth side, for scoring only); J_star: (Q,) true optimum per problem.
    """
    t_start = time.perf_counter()
    if ctx is None:
        ctx = build_ctx(env, problems, eps, delta, K)
    if J_true is None:
        J_true = true_policy_values(ctx.pols, env.w, env.true_mu(outcome))
    if J_star is None:
        J_star = np.array([J_true[ctx.feas[q]].max() for q in range(ctx.Q)])
    method.setup(ctx)
    adaptive = method.alloc_kind == "adaptive"
    sch = make_schedule(env, perm_seed, alloc=method.alloc_p if method.alloc_kind == "fixed" else None,
                        adaptive=adaptive)
    eng = StreamEngine(env, sch, outcome, delta_cell=delta / (env.S * env.A), R=ctx.R)
    ck = ctx.checkpoints
    if adaptive:
        bounds = np.unique(np.concatenate([ck, np.arange(ctx.replan, ctx.tau_R, ctx.replan)]))
    else:
        bounds = ck
    undecided = np.ones(ctx.Q, dtype=bool)
    decided_pi = np.full(ctx.Q, -1, dtype=np.int64)
    cert_k = np.full(ctx.Q, -1, dtype=np.int64)
    false_q = np.zeros(ctx.Q, dtype=bool)
    rows = []
    pref = None
    n_plans = 0
    t_plan = t_cert = 0.0
    stop_k_idx = None
    ck_set = {int(x): i for i, x in enumerate(ck)}

    def state():
        return FrontierState(eng.t, eng.n, eng.sum, eng.N, undecided, eng.cs_at, decided_pi)

    for b in bounds:
        b = int(b)
        if adaptive:
            if pref is None or eng.t % ctx.replan == 0:
                t0 = time.perf_counter()
                pref = np.asarray(method.plan(ctx, state()), dtype=np.int64)
                t_plan += time.perf_counter() - t0
                n_plans += 1
                if pref.shape != (ctx.S, ctx.A) or any(sorted(r) != list(range(ctx.A)) for r in pref.tolist()):
                    raise ValueError(f"{method.name}: plan must return an (S, A) arm permutation per segment")
            eng.advance_adaptive(b, pref)
        else:
            eng.advance_planned(b)
        if b not in ck_set:
            continue
        k = ck_set[b]
        t0 = time.perf_counter()
        res = method.certify(ctx, state())
        t_cert += time.perf_counter() - t0
        new = []
        for q, (ok, ih, U) in enumerate(res):
            if ok and undecided[q]:
                undecided[q] = False
                decided_pi[q] = ih
                cert_k[q] = k
                false_q[q] = bool(J_star[q] - J_true[ih] > ctx.eps + 1e-12)
                new.append(q)
        n_cert = int((~undecided).sum())
        row = {"k": k, "t": b, "n_cert": n_cert, "n_false": int(false_q.sum()), "new": new,
               "new_false": [q for q in new if false_q[q]], "billed": eng.billed, "served": eng.n_served,
               "skipped": eng.n_skipped, "reselected": eng.n_reselected, "billing_ok": eng.billing_ok(),
               "n_cells": eng.n.ravel().tolist()}
        if keep_U:
            row["U"] = [float(r[2]) for r in res]
        rows.append(row)
        if n_cert >= ctx.stop_k:
            stop_k_idx = k
            break
    reached = stop_k_idx is not None
    any_false = bool(false_q.any())
    n80 = int(ck[stop_k_idx]) if (reached and not any_false) else int(ctx.tau_R)
    summ = {"method": method.name, "validity": method.validity, "alloc_kind": method.alloc_kind,
            "perm_seed": int(perm_seed), "schedule_digest": sch.digest()[:16], "eps": ctx.eps, "delta": ctx.delta,
            "reached_stop": reached, "stop_k": ctx.stop_k, "k_stop": stop_k_idx, "N80": n80,
            "censored": bool(not reached or any_false), "n_cert": int((~undecided).sum()),
            "n_false": int(false_q.sum()), "fwer_event": any_false, "cert_k": cert_k.tolist(),
            "decided_pi": decided_pi.tolist(), "false_q": [int(q) for q in np.flatnonzero(false_q)],
            "billed": eng.billed, "t_end": eng.t, "served": eng.n_served, "skipped": eng.n_skipped,
            "reselected": eng.n_reselected, "loop_arrivals": eng.n_loop_arrivals,
            "billing_ok": eng.billing_ok() and all(r["billing_ok"] for r in rows), "n_plans": n_plans,
            "sec_plan": round(t_plan, 3), "sec_cert": round(t_cert, 3),
            "sec_total": round(time.perf_counter() - t_start, 3)}
    for r in rows:
        r.update({"method": method.name, "perm_seed": int(perm_seed)})
    return summ, rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


# =============================================================================================== method registry (r4_setup_baselines_b)
BASELINES_B = ("B3-rect", "B3-nominal", "B3-fav", "Peace-rect", "Peace-nominal", "Peace-fav", "B5", "H8-JPC1",
               "H8-Tboot", "H8-MisLid", "H8-MisLid0")


def make_frontier_method(name, env=None, **kw):
    """Factory for the r4_setup_baselines_b methods. ``env`` is used ONLY for its public segment descriptions
    (seg_desc, needed by the H8 feature maps); nothing truth-side is passed to a method."""
    seg_desc = getattr(env, "seg_desc", None)
    if name.startswith("B3-"):
        from ..baselines.rage_frontier import RAGEFrontier
        return RAGEFrontier(name[3:], **kw)
    if name.startswith("Peace-"):
        from ..baselines.peace_frontier import PeaceFrontier
        return PeaceFrontier(name[6:], **kw)
    if name == "B5":
        from ..baselines.maq_desc import MaqQini
        return MaqQini()
    from ..baselines import h8_models as h8
    if name == "H8-JPC1":
        return h8.JPC1Step(seg_desc)
    if name == "H8-Tboot":
        return h8.TLearnerBoot(seg_desc, **kw)
    if name == "H8-MisLid":
        return h8.MisLid1Step(seg_desc)
    if name == "H8-MisLid0":
        return h8.MisLid1Step(seg_desc, eps_mis=0.0, name="H8-MisLid0")
    raise KeyError(name)


# =============================================================================================== synthetic toy env
class SyntheticPoolEnv(PoolReplayEnv):
    """Small synthetic finite-pool population with the PoolReplayEnv interface (unit tests / toy FWER).

    pool_sizes (S, A) ints, mu (S, A) target cell means; binary outcomes with exactly round(mu * size) ones per pool,
    so the finite-population mean is known exactly. Segment weights follow the pool sizes."""

    def __init__(self, pool_sizes, mu, seed=0, n_min=None, outcome="visit"):  # noqa: D401 - no super().__init__
        pool_sizes = np.asarray(pool_sizes, dtype=np.int64)
        S, A = pool_sizes.shape
        rng = np.random.default_rng(seed)
        seg, arm, y = [], [], []
        for s in range(S):
            for a in range(A):
                m = int(pool_sizes[s, a])
                ones = int(round(float(mu[s][a]) * m))
                v = np.zeros(m)
                v[:ones] = 1.0
                seg.append(np.full(m, s))
                arm.append(np.full(m, a))
                y.append(rng.permutation(v))
        self.layer, self.half, self.split_seed = "TOY", "full", seed
        self.A, self.S = A, S
        self.seg_desc = [f"s{s}" for s in range(S)]
        self.seg = np.concatenate(seg).astype(np.int64)
        self.arm = np.concatenate(arm).astype(np.int64)
        yy = np.concatenate(y)
        yy.setflags(write=False)
        self._y = {outcome: yy}
        self.clip = {}
        self.binary_outcomes = (outcome,)
        self.N = int(len(self.seg))
        self.row_index = np.arange(self.N)
        self.cell = self.seg * A + self.arm
        self.pool_sizes = pool_sizes.copy()
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(S * A)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = int(n_min) if n_min is not None else max(20, self.N // 50)
        self.eps_grid = (0.01,)
        self.replan_interval = max(10, self.N // 200)
