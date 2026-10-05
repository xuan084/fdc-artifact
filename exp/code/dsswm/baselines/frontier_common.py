"""Learner-side primitives shared by every round-4 real-layer method (r4_setup_baselines_a / _b).

This module is truth-free (it never imports ``dsswm.envs``); the truth-side harness lives in
``dsswm.streams.frontier_runner``. A method sees only:

  * ``FrontierCtx``   -- public problem data: segment weights w, the enumerated policy class ``pols`` (P, S), the
                         problem list, eps, delta, checkpoints, the re-planning interval, per-problem feasibility masks,
                         pool sizes N (public in the pool protocol), stop rule (stop_k of Q problems).
  * ``FrontierState`` -- per-stream observations at the current arrival count: counts n, sums, pool sizes N,
                         the undecided-problem mask, and (lazily) the per-cell WSR20 without-replacement CS at the
                         current counts (``st.cs()``).

Method protocol (``Method``)
---------------------------
  name, alloc_kind in {'pool', 'fixed', 'adaptive'}, validity in {'rigorous', 'nominal', 'none'}, alloc_p (S, A) for
  'fixed'; setup(ctx); plan(ctx, st) -> pref (S, A) int array (adaptive only; row s is the method's arm preference
  order in segment s for the next batch: column 0 is the target arm, later columns are its own re-selection order when
  a pool is exhausted); certify(ctx, st) -> list of (certified: bool, pi_index: int, U: float) for every problem.
  The harness freezes a problem's answer at its first certification (sticky) and stops at stop_k certifications.

Certificates provided here
--------------------------
  * ``rect_certificate``  -- cell-level rectangle ("radius-sum") certificate from time-uniform per-cell CSs:
        U_q = max_{pi' in Pi_{B_q}} sum_{s: pi'(s) != pi_hat(s)} w_s (hi[s, pi'(s)] - lo[s, pi_hat(s)]).
    On the event {mu_c in CS_c(n) for every cell c and every n} (probability >= 1 - delta by a union over the S*A
    cells; WSR20 WoR CSs are time-uniform in the cell's own sample count, and the cell's draws are a uniformly random
    permutation of its pool whatever the sampling rule does), Delta(pi', pi_hat) <= U_q for every pi', so U_q <= eps
    implies an eps-correct answer. Rigorous under adaptive counts and pool exhaustion.
  * ``glr_certificate``   -- functional GLR stopping (eps - Delta_hat)_+^2 / (2 V) >= beta for every feasible
    challenger, i.e. U_q = max_{pi'} Delta_hat + sqrt(2 beta V) <= eps (the algebraic identity of methodology 1.5).
    V = sum over differing segments of w_s^2 (var[s, pi'(s)] / n + var[s, pi_hat(s)] / n); exhausted cells are exact
    (contribute 0); n = 0 -> +inf. Used by the nominal (sigma^2 = 1/4) and fav (plug-in variance) variants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..streams.frontier import FEAS_TOL, enumerate_policies

__all__ = ["FrontierCtx", "FrontierState", "Method", "make_ctx", "jhat_all", "pi_hat_indices", "rect_U",
           "rect_certificate", "glr_V", "glr_certificate", "plugin_var", "QFCMethod", "qfc_default_params"]


# =============================================================================================== context / state
@dataclass
class FrontierCtx:
    w: np.ndarray                 # (S,)
    S: int
    A: int
    pols: np.ndarray              # (P, S) int
    problems: list
    eps: float
    delta: float
    checkpoints: np.ndarray       # (K,)
    replan: int
    feas: np.ndarray              # (Q, P) bool
    N: np.ndarray                 # (S, A) pool sizes (public)
    tau_R: int
    stop_k: int
    binary: bool = True
    R: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def Q(self):
        return len(self.problems)

    @property
    def P(self):
        return self.pols.shape[0]


def make_ctx(w, N, problems, eps, delta, checkpoints, replan, tau_R, stop_frac=0.8, binary=True, R=1.0, pols=None):
    w = np.asarray(w, dtype=float)
    N = np.asarray(N, dtype=np.int64)
    S, A = N.shape
    pols = enumerate_policies(S, A).astype(np.int64) if pols is None else np.asarray(pols, dtype=np.int64)
    feas = np.zeros((len(problems), pols.shape[0]), dtype=bool)
    for q, p in enumerate(problems):
        cost = (w[None, :] * np.asarray(p.kappa, dtype=float)[pols]).sum(1)
        feas[q] = cost <= p.budget + FEAS_TOL
    stop_k = int(math.ceil(stop_frac * len(problems) - 1e-9))
    return FrontierCtx(w=w, S=S, A=A, pols=pols, problems=list(problems), eps=float(eps), delta=float(delta),
                       checkpoints=np.asarray(checkpoints, dtype=np.int64), replan=int(replan), feas=feas, N=N,
                       tau_R=int(tau_R), stop_k=stop_k, binary=binary, R=float(R))


class FrontierState:
    """Observation state handed to methods (copies; methods cannot alter the stream)."""

    def __init__(self, t, n, sums, N, undecided, cs_fn, decided_pi=None):
        self.t = int(t)
        self.n = np.asarray(n, dtype=np.int64).copy()
        self.sum = np.asarray(sums, dtype=float).copy()
        self.N = np.asarray(N, dtype=np.int64)
        self.undecided = np.asarray(undecided, dtype=bool).copy()
        self.decided_pi = None if decided_pi is None else np.asarray(decided_pi).copy()
        self._cs_fn = cs_fn
        self._cs = None

    @property
    def mu_hat(self):
        """Empirical cell means; cells with n = 0 get 1/2 (uninformative placeholder)."""
        return np.where(self.n > 0, self.sum / np.maximum(self.n, 1), 0.5)

    @property
    def remaining(self):
        return self.N - self.n

    def cs(self):
        """(lo, hi) of the per-cell WSR20 WoR CS at the current counts (lazy; same delta per cell for all methods)."""
        if self._cs is None:
            self._cs = self._cs_fn(self.n)
        return self._cs


class Method:
    name = "method"
    alloc_kind = "adaptive"
    validity = "none"
    alloc_p = None

    def setup(self, ctx: FrontierCtx):
        pass

    def plan(self, ctx: FrontierCtx, st: FrontierState):
        raise NotImplementedError

    def certify(self, ctx: FrontierCtx, st: FrontierState):
        raise NotImplementedError

    def describe(self):
        return {"name": self.name, "alloc_kind": self.alloc_kind, "validity": self.validity}


# =============================================================================================== shared helpers
def jhat_all(ctx: FrontierCtx, mu_hat):
    seg = np.arange(ctx.S)[None, :]
    return (ctx.w[None, :] * np.asarray(mu_hat)[seg, ctx.pols]).sum(1)


def pi_hat_indices(ctx: FrontierCtx, mu_hat):
    """Per problem: argmax of J_hat over Pi_{B_q} (lowest enumeration index on ties)."""
    J = jhat_all(ctx, mu_hat)
    return np.array([int(np.argmax(np.where(ctx.feas[q], J, -np.inf))) for q in range(ctx.Q)], dtype=np.int64)


def rect_U(ctx: FrontierCtx, lo, hi, ih):
    """Rectangle upper bound on Delta(pi', pi_hat) for all pi' (vector over P); pi_hat = pols[ih]."""
    seg = np.arange(ctx.S)
    ph = ctx.pols[ih]
    diff = ctx.pols != ph[None, :]
    up = hi[seg[None, :], ctx.pols] - lo[seg, ph][None, :]
    return (ctx.w[None, :] * np.where(diff, up, 0.0)).sum(1)


def rect_certificate(ctx: FrontierCtx, st: FrontierState):
    lo, hi = st.cs()
    ihs = pi_hat_indices(ctx, st.mu_hat)
    out = []
    for q in range(ctx.Q):
        U = float(np.max(np.where(ctx.feas[q], rect_U(ctx, lo, hi, ihs[q]), -np.inf)))
        out.append((bool(U <= ctx.eps), int(ihs[q]), U))
    return out


def plugin_var(st: FrontierState, R=1.0):
    """Plug-in variance mu_hat (1 - mu_hat) (binary) -- fav variants only; no validity claimed."""
    m = st.mu_hat / R
    return (R * R) * m * (1.0 - m)


def glr_V(ctx: FrontierCtx, st: FrontierState, var, ih):
    """V(pi', pi_hat) for every pi' (vector over P). Exhausted cells contribute 0; n = 0 -> inf."""
    seg = np.arange(ctx.S)
    n = st.n.astype(float)
    live = n < st.N
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(n <= 0, np.inf, np.where(live, (ctx.w ** 2)[:, None] * np.asarray(var) / np.maximum(n, 1), 0.0))
    ph = ctx.pols[ih]
    diff = ctx.pols != ph[None, :]
    return np.where(diff, v[seg[None, :], ctx.pols] + v[seg, ph][None, :], 0.0).sum(1)


def glr_certificate(ctx: FrontierCtx, st: FrontierState, var, beta):
    mu = st.mu_hat
    J = jhat_all(ctx, mu)
    ihs = pi_hat_indices(ctx, mu)
    out = []
    for q in range(ctx.Q):
        V = glr_V(ctx, st, var, ihs[q])
        with np.errstate(invalid="ignore"):
            U = (J - J[ihs[q]]) + np.sqrt(2.0 * beta * V)
        U = np.where(np.isinf(V), np.inf, U)
        Uq = float(np.max(np.where(ctx.feas[q], U, -np.inf)))
        out.append((bool(Uq <= ctx.eps), int(ihs[q]), Uq))
    return out


# =============================================================================================== QFC (main method)
def qfc_default_params(S, A, Q, K=20, delta_main=0.04, delta_var=0.01, C_var=48):
    """Lock ledger (plan/theory/qfc_lemma.md s4/s9): L1 = Q*-union exponent; x_v = ln(2 C_var K / delta_var)."""
    from ..certify.quadknap import l1_qstar
    return {"L1": l1_qstar(S, A, Q, K, delta_main), "x_v": math.log(2 * C_var * K / delta_var),
            "delta_main": delta_main, "delta_var": delta_var, "C_var": C_var, "K": K}


class QFCMethod(Method):
    """QFC certificate (exact enumeration) with a non-adaptive allocation: pool proportions (main method, alloc_p
    None) or a frozen fixed allocation matrix (A-Ney / A-XY ablations). Covered by Thm 1 (non-adaptive schedule)."""
    validity = "rigorous"

    def __init__(self, name="QFC", alloc_p=None, params=None):
        self.name = name
        self.alloc_kind = "pool" if alloc_p is None else "fixed"
        self.alloc_p = None if alloc_p is None else np.asarray(alloc_p, dtype=float)
        self.params = params

    def setup(self, ctx):
        if self.params is None:
            self.params = qfc_default_params(ctx.S, ctx.A, ctx.Q, K=len(ctx.checkpoints))

    def certify(self, ctx, st):
        from ..certify.quadknap import make_stats, qfc_certificate_enum
        stats = make_stats(ctx.w, st.mu_hat, st.n, N=st.N, x_v=self.params["x_v"], binary=ctx.binary, R=ctx.R)
        res = qfc_certificate_enum(stats, ctx.problems, self.params["L1"], ctx.eps, pols=ctx.pols)
        idx = {tuple(int(x) for x in r): i for i, r in enumerate(ctx.pols)} if not hasattr(self, "_idx") else self._idx
        self._idx = idx
        return [(bool(r["certified"]), int(idx[r["pi_hat"]]), float(r["U"])) for r in res]

    def describe(self):
        d = super().describe()
        d.update({"params": self.params, "alloc_p": None if self.alloc_p is None else self.alloc_p.tolist()})
        return d
