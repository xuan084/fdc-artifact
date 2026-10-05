"""Round-4 multi-step layer: learner-side method interface + shared helpers (r3 JPC, always-dynamic JPC, FCC adapter).

LEARNER SIDE: this module (and every dsswm.baselines.* module used by the r4 multi-step harness) must not import the
harness / truth modules (dsswm.streams.ms_r4, dsswm.streams.offgrid, replica, run_* scripts). A method sees only
  pub      MSPublic: the stream problems (public Problem objects, c_q = class-max normalisation), the public class
           objects and class J tables of the learner classes, eps, delta, known constants (rho_ret, c, nmax)
  handle   EnvHandle (step / observable_state / reset_to only; truth access raises TruthAccessError)
  sw, lr   the r3 ReuseSwitch (`full` arm) and the problem's switched evidence set; every new platform round must go
           through sw.record (the only billed path)

Method protocol
  m = Method(pub);  m.make_set_factory() -> SeqLRSet factory for the switch;  m.begin_stream(init_obs)
  m.solve(k, sw, lr, handle, rng, tmax) -> {"status", "pi", "steps", "extra"}  (one problem, one call)
Checkpoints (methodology 1.6): certificate checks at new-step counts {0} + 20 geometric points in [1, TAU=3000],
continued with the same ratio up to T_MAX (6000).
"""
from __future__ import annotations

import copy
import time

import numpy as np

from ..certify.fcc import FCCCertifier
from .switched_nl import PublicNL, solve_set_method

TAU = 3000
T_MAX = 6000


def checkpoints(tau: int = TAU, tmax: int = T_MAX, n: int = 20) -> np.ndarray:
    g = np.round(np.geomspace(1, tau, n)).astype(int)
    for i in range(1, n):                                  # strictly increasing integer ladder (20 distinct points)
        g[i] = max(g[i], g[i - 1] + 1)
    assert g[-1] == tau
    r = (tau / 1.0) ** (1.0 / (n - 1))
    ext, x = [], float(tau)
    while x * r < tmax:
        x *= r
        ext.append(int(round(x)))
    return np.unique(np.concatenate([[0], g, ext, [tmax]])).astype(int)


CHECKPOINTS = checkpoints()


class MSPublic:
    """Truth-free per-stream objects handed to a learner."""

    def __init__(self, problems, classes: dict, eps: float, delta: float, rho_ret: float, c_known: float, nmax: int,
                 default_class: str, layer: str):
        self.problems = list(problems)
        self.classes = classes            # name -> PublicNL (with_problems already applied)
        self.eps, self.delta = float(eps), float(delta)
        self.rho_ret, self.c_known, self.nmax = float(rho_ret), float(c_known), int(nmax)
        self.default_class = default_class
        self.layer = layer
        any_pub = classes[default_class]
        self.aspace = any_pub.aspace
        self.prop = any_pub.prop

    def nl(self, cls: str | None = None) -> PublicNL:
        return self.classes[cls or self.default_class]


def u_max(q) -> float:
    return float(q.utility.c_q) * (2.0 * float(np.sum(q.utility.w)) + 4.0 * float(q.utility.w_ret))


class Method:
    name = "base"
    learner_class: str | None = None          # None -> pub.default_class

    def __init__(self, pub: MSPublic):
        self.pub = pub
        self.n_obs = 0

    def nlpub(self) -> PublicNL:
        return self.pub.nl(self.learner_class)

    def make_set_factory(self):
        return self.nlpub().make_set_factory("JPC")

    def begin_stream(self, init_obs):
        for o in init_obs:
            self.observe(o)

    def observe(self, obs):
        self.n_obs += 1

    def step(self, sw, handle, a):
        obs = handle.step(int(a))
        sw.record(obs)
        self.observe(obs)
        return obs

    def solve(self, k, sw, lr, handle, rng, tmax):
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ JPC family
class JPC(Method):
    """r3 locked JPC (SeqLRSet log 1/delta + exact minimax certificate + DDA), called as is. On MS-S the r3 learner class
    is the static twin (r3_static); on MS-H / MS-R3 / MS-F it is G_1."""
    name = "JPC"

    def solve(self, k, sw, lr, handle, rng, tmax):
        return solve_set_method(self.nlpub(), k, sw, lr, handle, rng, tmax, method="JPC")


class AlwaysDynamicJPC(JPC):
    """r3 dynamic class G_1 everywhere, no guard (differs from JPC only on MS-S, where r3 JPC uses the static class)."""
    name = "always-dynamic-JPC"
    learner_class = "dyn_G1"


def with_eps(pub: PublicNL, eps: float) -> PublicNL:
    out = copy.copy(pub)
    out.eps = float(eps)
    return out


# ------------------------------------------------------------------------------------------------ cell acquisition
class CellAcquirer:
    """Width x influence greedy design over legal actions (shared by FCC / PolicyCert-tab / PERP-fact so the three
    differ only in the certificate). Influence of a cell = #(t, state) uses in the plans of the two contending policies
    (leader by lower bound, challenger by upper bound). Ties / all-zero scores -> uniform legal action (rng)."""

    def __init__(self, solver, legal, n_cells: int):
        self.solver, self.legal, self.n = solver, np.asarray(legal), int(n_cells)
        self.infl = None

    def set_influence(self, plans, idx):
        n = self.n
        infl = np.zeros(n)
        for i in idx:
            for c, m in self.solver.used_cells(plans[i]).items():
                if c < n:
                    infl[c] += m
        self.infl = infl

    def choose(self, code, width, rng):
        n = width.shape[0]
        best, best_s = [], -1.0
        for a in self.legal:
            cid = self.solver.structure(int(code), int(a))[0].reshape(-1)
            cid = cid[cid < n]
            s = float(np.sum(width[cid] * (1.0 + (self.infl[cid] if self.infl is not None else 0.0))))
            if s > best_s + 1e-15:
                best, best_s = [int(a)], s
            elif abs(s - best_s) <= 1e-15:
                best.append(int(a))
        if best_s <= 0:
            return int(rng.choice(self.legal))
        return int(best[0] if len(best) == 1 else rng.choice(best))


def contenders(bounds: np.ndarray):
    LB, UB = bounds[:, 0], bounds[:, 1]
    k = int(np.argmax(LB))
    if len(UB) == 1:
        return k, k
    ub = UB.copy()
    ub[k] = -np.inf
    return k, int(np.argmax(ub))


class FCC(Method):
    """FCC wiring adapter (methodology 1.6): FCCCertifier (per-cell betting CS, union over cells, decoupled bounds),
    width x influence acquisition, certificate checks at CHECKPOINTS. The acquisition rule is a wiring default that the
    G-fac gate may replace; the certificate is the r4_setup_fcc one, unchanged."""
    name = "FCC"
    learner_class = "dyn_G1"
    bnb = False

    def __init__(self, pub: MSPublic):
        super().__init__(pub)
        P = pub.aspace
        self.cert = FCCCertifier(P.L, P.R, pub.nmax, P, pub.rho_ret, pub.delta)
        self.acq = CellAcquirer(self.cert.solver, np.arange(P.n), self.cert.cells.n)
        self._plans = {}

    def observe(self, obs):
        super().observe(obs)
        self.cert.observe(obs)

    def plans(self, k):
        if k not in self._plans:
            q = self.pub.problems[k]
            self._plans[k] = [self.cert.solver.plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
        return self._plans[k]

    def bounds(self, k):
        return self.cert.bounds(self.pub.problems[k], plans=self.plans(k), bnb=self.bnb)

    def widths(self):
        lo, hi = self.cert.cs.intervals()
        return hi - lo

    def code_of(self, handle):
        return self.cert.solver.codec.encode(*handle.observable_state())

    def solve(self, k, sw, lr, handle, rng, tmax):
        t0 = time.perf_counter()
        eps = self.pub.eps
        plans = self.plans(k)
        ci, n_checks, status, pi, first = 0, 0, None, None, None
        b = None
        while True:
            if sw.new_steps >= CHECKPOINTS[ci] or sw.new_steps >= tmax:
                while ci < len(CHECKPOINTS) - 1 and CHECKPOINTS[ci] <= sw.new_steps:
                    ci += 1
                b = self.bounds(k)
                n_checks += 1
                st, kk = FCCCertifier.decide(b, eps)
                if st == "CERTIFIED":
                    status, pi, first = "CERTIFIED", kk, sw.new_steps
                    break
                lead, ch = contenders(b)
                self.acq.set_influence(plans, (lead, ch))
            if sw.new_steps >= tmax:
                status = "NEED_DATA"
                break
            a = self.acq.choose(self.code_of(handle), self.widths(), rng)
            self.step(sw, handle, a)
        return {"status": status, "pi": pi, "steps": sw.new_steps,
                "extra": {"n_checks": n_checks, "first_cert_step": first, "wall_clock_s": time.perf_counter() - t0,
                          "gap_end": cert_gap(b), "cs_empty_cells": int(np.sum(self.cert.cs.empty))}}


def cert_gap(b):
    """max_{pi != leader} UB(pi) - LB(leader) (certified iff <= eps); None without bounds or with one policy."""
    if b is None or len(b) < 2:
        return None
    lead, _ = contenders(b)
    return float(np.delete(b[:, 1], lead).max() - b[lead, 0])
