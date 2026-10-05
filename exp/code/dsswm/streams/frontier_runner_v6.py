"""Lock-v6 addendum harness: thin wrapper around the v5-frozen ``frontier_runner.run_stream`` (new file).

* ``run_stream_v6(env, method, ...)`` runs EXACTLY the frozen ``run_stream`` (same schedule, engine, billing, sticky
  answers, stop rule, N80 / censoring definitions).  For methods with ``cs_kind == 'hc'`` the engine's per-cell CS
  provider ``cs_at`` is swapped, for the duration of the call, by a subclass that returns the hedged-capital WoR CS
  (``baselines.wor_betting_v6``) computed from each cell's own draws; everything else is inherited unchanged.  For any
  other method the call is the frozen function itself (no patch).
* ``stop_k`` override (block C continuation, N100): the frozen runner stops at ctx.stop_k; continuation runs use a
  ctx copy with stop_k = Q, so the first stop_k-crossing of the original rule is read off the same trajectory
  (``n_cert`` curve), i.e. the prefix up to the original stop is identical by construction (checked by the replica
  rule of the addendum).
"""
from __future__ import annotations

import contextlib
import copy

import numpy as np

from ..baselines.wor_betting_v6 import hc_lambda_tilde, hc_wor_cs_interval
from . import frontier_runner as _fr

__all__ = ["run_stream_v6", "hc_engine_class", "ctx_with_stop"]


def hc_engine_class(params):
    """StreamEngine subclass whose cs_at(n) returns the hedged-capital WoR CS (running intersection per cell)."""
    base = _fr.StreamEngine

    class HCEngine(base):
        _hc = params

        def _hc_init(self):
            if getattr(self, "_hc_ready", False):
                return
            C = self.S * self.A
            self._hc_lamt = [None] * C
            self._hc_lo = np.zeros(C)
            self._hc_hi = np.full(C, 1.0)
            self._hc_last_n = np.full(C, -1, dtype=np.int64)
            self._hc_last = [None] * C
            self.hc_empty_events = 0
            self._hc_ready = True

        def _hc_cell(self, c, n):
            p = self._hc
            if self._hc_lamt[c] is None:
                o = self._obs[c] / self.R
                ns = None if p["n_star"] is None else np.asarray(p["n_star"], dtype=float).ravel()[c]
                self._hc_lamt[c] = hc_lambda_tilde(o, p["alpha"], p["schedule"], ns)
            if n == self._hc_last_n[c]:
                return self._hc_last[c]
            if n < self._hc_last_n[c]:
                raise RuntimeError("HC CS: counts must be nondecreasing within a stream")
            x = self._obs[c][:n] / self.R
            lo, hi, empty = hc_wor_cs_interval(x, int(self.N.ravel()[c]), p["alpha"], self._hc_lamt[c], p["c"],
                                               p["theta"], lo0=float(self._hc_lo[c]), hi0=float(self._hc_hi[c]))
            self._hc_last_n[c] = n
            if empty:
                # empty CS (coverage event already failed): return the logical pool interval, keep the running state
                self.hc_empty_events += 1
                self._hc_last[c] = (lo, hi)
                return lo, hi
            lo = max(lo, self._hc_lo[c])                       # running intersection (time-uniform event)
            hi = min(hi, self._hc_hi[c])
            self._hc_lo[c], self._hc_hi[c] = lo, hi
            self._hc_last[c] = (lo, hi)
            return lo, hi

        def cs_at(self, n):
            self._hc_init()
            lo = np.zeros((self.S, self.A))
            hi = np.zeros((self.S, self.A))
            for c in range(self.S * self.A):
                s, a = divmod(c, self.A)
                L, H = self._hc_cell(c, int(n[s, a]))
                lo[s, a], hi[s, a] = L * self.R, H * self.R
            return lo, hi

    return HCEngine


@contextlib.contextmanager
def _patched_engine(cls):
    orig = _fr.StreamEngine
    _fr.StreamEngine = cls
    try:
        yield
    finally:
        _fr.StreamEngine = orig


def ctx_with_stop(ctx, stop_k):
    c2 = copy.copy(ctx)
    c2.stop_k = int(stop_k)
    return c2


def run_stream_v6(env, method, perm_seed, problems, eps, outcome="visit", delta=0.05, K=20, ctx=None, J_true=None,
                  J_star=None, keep_U=True):
    if ctx is None:
        ctx = _fr.build_ctx(env, problems, eps, delta, K)
    if getattr(method, "cs_kind", None) == "hc":
        method.setup(ctx)                       # n_star needs the ctx; run_stream calls setup again (idempotent)
        cls = hc_engine_class(method.cs_params(ctx))
        with _patched_engine(cls):
            return _fr.run_stream(env, method, perm_seed, problems, eps, outcome=outcome, delta=delta, K=K, ctx=ctx,
                                  J_true=J_true, J_star=J_star, keep_U=keep_U)
    return _fr.run_stream(env, method, perm_seed, problems, eps, outcome=outcome, delta=delta, K=K, ctx=ctx,
                          J_true=J_true, J_star=J_star, keep_U=keep_U)
