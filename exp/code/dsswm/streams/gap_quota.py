"""Round-3 NL-R0 gap-quota problem streams (harness side: this module CREATES ground truth and reads it to place
problems; learner code must never import it). Methodology sections 2.1 and 2.5.

Instance (statistical unit, seed s):
  theta*  uniform on the learner grid G_1 (|Theta| = 13824, realizable): independent uniform draws of each left
          tuple, right tuple and (psi, lam) with rng [s, 41]  (uniform on the Cartesian product)
  pool    candidate problems j = 0, 1, 2, ... drawn with rng [s, 43, j] by the round-0 generator code path
          (H, loads0, engaged0, aff, policy family, utility with retention); class J table (exact propagator over
          Theta, class-max normalisation c_q, truth-free, cached) -> true top-2 gap Delta_j = J*(1) - J*(2)
  layers  tie: Delta < eps,  near: eps <= Delta < 2 eps,  clear: Delta >= 2 eps   (eps = 0.02)
Streams (all CRN-paired across methods / arms):
  quota        5 tie + 5 near + 5 clear: sequential rejection sampling over the pool, each layer accepts the first 5
               candidates of its layer in pool order; a layer that is still unfilled after 2000 pool draws makes the
               instance `quota_fail` (it is then replaced by the next reserve seed, in seed order: `fill_instances`)
  no_tie       15 clear problems (Delta >= 2 eps), first 15 clear candidates of the pool (same 2000-draws-per-slot cap)
  low_overlap  quota composition, but chosen greedily from the pool by the PUBLIC decision direction
               D_q = grad_v (J(pi_a) - J(pi_b)) at the class-mean vector v_bar (pi_a, pi_b = top-2 policies at v_bar):
               each next problem minimises max_prev |cos(D_q, D_prev)| among candidates of the not-yet-full layers
  offset_null  the quota problems with utility U'_q(traj; pi) = (U_q(traj) + b_pi) / (1 + max b),
               b_pi = max_pi' J*(pi') - J*(pi) >= 0 (harness-computed, handed to the learner as a PUBLIC utility term):
               all candidates have equal true value; other class members still disagree, so data is needed to certify
               an eps-tie. Class tables transform with the same affine map.
Stream order: base order = draw order of the selected problems shuffled by rng [s, 44]; stream t in {0, 1, 2} uses
the permutation rng [s, 45, t] of the base order and noise seed STREAMS[t] = 42 / 123 / 456. The sibling platform of
the `vol` arm uses noise seed + 7919 (evidence.reuse_switch.SHADOW_SEED_OFFSET), same theta*, same problems.
"""
from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..envs.nl import NLEnv
from ..evidence.reuse_switch import SHADOW_SEED_OFFSET
from ..models.nl_class import NLClass
from .generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, Problem, _initial_data, _sample_policies
from .utilities import Utility, sample_utility

EPS_NL = 0.02
QUOTA = 5
LAYERS = ("tie", "near", "clear")
MAX_DRAWS_PER_LAYER = 2000
STREAM_NOISE = (42, 123, 456)
KINDS = ("quota", "no_tie", "low_overlap", "offset_null")


def gap_layer(delta: float, eps: float = EPS_NL) -> str:
    if delta < eps:
        return "tie"
    if delta < 2 * eps:
        return "near"
    return "clear"


def top2_gap(j: np.ndarray) -> float:
    s = np.sort(np.asarray(j, float))[::-1]
    return float(s[0] - s[1])


def nl_r0_truth(seed: int, grid=DEFAULT_NL_S_GRID) -> dict:
    rng = np.random.default_rng([seed, 41])
    lt, rt, gt = grid.left_tuples(), grid.right_tuples(), grid.global_tuples()
    lefts = [lt[int(rng.integers(len(lt)))] for _ in range(grid.L)]
    rights = [rt[int(rng.integers(len(rt)))] for _ in range(grid.R)]
    g = gt[int(rng.integers(len(gt)))]
    return {"alpha": [x[0] for x in lefts], "gamma": [x[1] for x in lefts], "tauL": [x[2] for x in lefts],
            "beta": [x[0] for x in rights], "tauR": [x[1] for x in rights], "psi": float(g[0]), "lam": float(g[1])}


def make_env(seed: int, truth: dict, noise_seed: int, cfg=NL_DEFAULTS, grid=DEFAULT_NL_S_GRID) -> NLEnv:
    return NLEnv(np.array(truth["alpha"]), np.array(truth["beta"]), np.array(truth["gamma"]), np.array(truth["tauL"]),
                 np.array(truth["tauR"]), [truth["psi"]], truth["lam"], c=cfg["c"], rho_ret=cfg["rho_ret"], L=grid.L,
                 R=grid.R, nmax=cfg["nmax"], budget=cfg["budget"], incentive_levels=(1,),
                 seed=int(seed) * 1000 + int(noise_seed))


def candidate_problem(seed: int, j: int, aspace, cfg=NL_DEFAULTS, L=2, R=2) -> Problem:
    prng = np.random.default_rng([seed, 43, j])
    H = int(prng.choice(cfg["H_choices"]))
    loads0 = prng.integers(0, cfg["nmax"] + 1, L + R)
    eng0 = (prng.random(L + R) < cfg["p_engaged0"]).astype(np.int64)
    aff = prng.standard_normal((L, R))
    pols = _sample_policies(prng, aspace, H, aff, cfg["nmax"], cfg["n_pol"])
    util = sample_utility(prng, H, L, R, with_retention=True, y_scale=1.0)
    return Problem(f"r3q{seed}_c{j}", 1, H, loads0, eng0, pols, util, aff, meta={"cand_index": j})


def _atomic_save(path: Path, arr):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)


@dataclass
class Candidate:
    j: int
    problem: Problem
    J: np.ndarray            # class table, class-max normalised (truth-free)
    delta: float             # harness: true top-2 gap
    layer: str               # harness
    D: np.ndarray | None = None


class QuotaBuilder:
    """Per-instance candidate pool + stream selection. `prop_tab` computes class J tables (GPU ok); `prop_cpu` is
    used only for the public decision directions (autograd on CPU)."""

    def __init__(self, seed: int, ncl: NLClass, prop_tab, cache_dir, eps: float = EPS_NL, prop_cpu=None,
                 max_draws: int = MAX_DRAWS_PER_LAYER, quota: int = QUOTA, grid=DEFAULT_NL_S_GRID):
        self.seed, self.ncl, self.prop, self.eps = int(seed), ncl, prop_tab, eps
        self.prop_cpu = prop_cpu
        self.max_draws, self.quota, self.grid = max_draws, quota, grid
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.truth = nl_r0_truth(seed, grid)
        self.ti = ncl.index_of(tuple(self.truth["alpha"]), tuple(self.truth["beta"]), tuple(self.truth["gamma"]),
                               tuple(self.truth["tauL"]), tuple(self.truth["tauR"]), self.truth["psi"],
                               self.truth["lam"])
        self._env0 = make_env(seed, self.truth, STREAM_NOISE[0])
        self.aspace = self._env0.aspace
        self._params = None
        self.pool: list[Candidate] = []
        self.sec_tables = 0.0
        self.n_table_computed = 0
        self._sel: dict = {}

    # ------------------------------------------------------------------------------ pool
    def candidate(self, j: int) -> Candidate:
        while len(self.pool) <= j:
            jj = len(self.pool)
            q = candidate_problem(self.seed, jj, self.aspace)
            path = self.cache_dir / f"{q.pid}_raw.npy"
            if path.exists():
                raw = np.load(path)
            else:
                import time
                if self._params is None:
                    self._params = self.ncl.torch_params()
                t0 = time.perf_counter()
                u1 = Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0)
                raw = self.prop.j_table(self._params, q.policies, q.loads0, q.engaged0, q.H, u1)
                self.sec_tables += time.perf_counter() - t0
                self.n_table_computed += 1
                _atomic_save(path, raw)
            c_q = 1.0 / float(raw.max())
            q.utility.c_q = c_q
            m = min(self.grid.L, self.grid.R)
            q.meta["u_max"] = c_q * (m * float(q.utility.w.sum()) + q.utility.w_ret * (self.grid.L + self.grid.R))
            q.meta["normalisation"] = "class_max"
            J = raw * c_q
            d = top2_gap(J[self.ti])
            self.pool.append(Candidate(jj, q, J, d, gap_layer(d, self.eps)))
        return self.pool[j]

    def decision_direction(self, c: Candidate) -> np.ndarray:
        """Public D_q at the class-mean vector (no truth): gradient of the top-2 contrast at v_bar."""
        if c.D is None:
            from ..baselines.glm_linearised import class_theta_vecs, value_and_grad
            if not hasattr(self, "_vbar"):
                self._vbar = class_theta_vecs(self.ncl.np_params).mean(0)
            q = c.problem
            plans = [self.prop_cpu.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
            Jl, G = value_and_grad(self.prop_cpu, plans, self._vbar, q.utility, self.grid.L, self.grid.R)
            o = np.argsort(-Jl)
            c.D = G[o[0]] - G[o[1]]
        return c.D

    # ------------------------------------------------------------------------------ selection
    def select(self, kind: str = "quota") -> dict:
        if kind == "offset_null":
            kind = "quota"
        if kind in self._sel:
            return self._sel[kind]
        need = ({"clear": 3 * self.quota} if kind == "no_tie" else {l: self.quota for l in LAYERS})
        draws_used = {l: 0 for l in need}
        cap = {l: self.max_draws * (need[l] // self.quota) for l in need}
        picked = {l: [] for l in need}
        j = 0
        fail = False
        if kind in ("quota", "no_tie"):
            while any(len(picked[l]) < need[l] for l in need):
                if any(len(picked[l]) < need[l] and draws_used[l] >= cap[l] for l in need):
                    fail = True
                    break
                c = self.candidate(j)
                j += 1
                for l in need:
                    if len(picked[l]) < need[l]:
                        draws_used[l] += 1
                if c.layer in need and len(picked[c.layer]) < need[c.layer]:
                    picked[c.layer].append(c)
            chosen = sorted([c for l in need for c in picked[l]], key=lambda c: c.j)
        elif kind == "low_overlap":
            base = self.select("quota")
            if base["quota_fail"]:
                self._sel[kind] = {**base, "kind": kind}
                return self._sel[kind]
            n_pool = base["n_draws"]
            pool = [self.candidate(i) for i in range(n_pool)]
            counts = {l: 0 for l in LAYERS}
            chosen = []
            first = min((c for c in pool if c.layer == "tie"), key=lambda c: c.j)
            chosen.append(first); counts[first.layer] += 1
            while len(chosen) < 3 * self.quota:
                best, bval = None, np.inf
                prevD = [self._unit(self.decision_direction(c)) for c in chosen]
                for c in pool:
                    if c in chosen or counts[c.layer] >= self.quota:
                        continue
                    u = self._unit(self.decision_direction(c))
                    val = max(abs(float(u @ p)) for p in prevD)
                    if val < bval - 1e-15:
                        best, bval = c, val
                chosen.append(best); counts[best.layer] += 1
            j = n_pool
            draws_used = {l: n_pool for l in LAYERS}
        else:
            raise ValueError(kind)
        out = {"kind": kind, "quota_fail": fail, "n_draws": j, "draws_used": draws_used,
               "chosen": [] if fail else chosen,
               "layer_counts": {l: sum(c.layer == l for c in chosen) for l in LAYERS}}
        if not fail:
            base_perm = np.random.default_rng([self.seed, 44]).permutation(len(chosen))
            out["base"] = [chosen[i] for i in base_perm] if kind != "low_overlap" else chosen
        self._sel[kind] = out
        return out

    @staticmethod
    def _unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def fill_rate(self, kind="quota") -> float:
        s = self.select(kind)
        tot = 3 * self.quota
        return sum(s["layer_counts"].values()) / tot if not s["quota_fail"] else \
            sum(min(v, self.quota if kind != "no_tie" else tot) for v in s["layer_counts"].values()) / tot

    # ------------------------------------------------------------------------------ streams
    def stream(self, stream: int, kind: str = "quota", sibling: bool = False, n_problems: int | None = None):
        sel = self.select(kind)
        if sel["quota_fail"]:
            raise RuntimeError(f"instance {self.seed}: quota_fail for kind={kind}")
        base = sel["base"]
        if kind == "low_overlap":
            order = list(range(len(base)))        # the greedy order IS the stream order (all streams share it)
        else:
            order = list(np.random.default_rng([self.seed, 45, stream]).permutation(len(base)))
        cands = [base[i] for i in order][: n_problems or len(base)]
        noise = STREAM_NOISE[stream] + (SHADOW_SEED_OFFSET if sibling else 0)
        env = make_env(self.seed, self.truth, noise)
        init = _initial_data(env, NL_DEFAULTS["n0"], np.random.default_rng([self.seed, 12]))
        problems, Js, meta = [], [], []
        for c in cands:
            q, J = c.problem, c.J
            if kind == "offset_null":
                q, J = offset_null(q, J, self.ti)
            problems.append(q)
            Js.append(J)
            meta.append({"pid": q.pid, "cand_index": c.j, "true_gap": top2_gap(J[self.ti]), "gap_layer": c.layer,
                         "true_gap_orig": c.delta, "pi_star": int(np.argmax(J[self.ti]))})
        return R3Stream(seed=self.seed, stream=stream, kind=kind, noise_seed=noise, sibling=sibling, env=env,
                        init_obs=init, problems=problems, J=Js, harness=meta, theta_index=self.ti,
                        truth=dict(self.truth))

    def overlap(self, problems_cands) -> list:
        """cos overlap of each problem's public decision direction with the previous ones (max |cos| and the
        cosine to span(prev))."""
        out, prev = [], []
        for c in problems_cands:
            u = self._unit(self.decision_direction(c))
            if not prev:
                out.append({"cos_max": None, "cos_span": None})
            else:
                P = np.stack(prev, 1)
                Q, _ = np.linalg.qr(P)
                proj = Q @ (Q.T @ u)
                out.append({"cos_max": max(abs(float(u @ p)) for p in prev), "cos_span": float(np.linalg.norm(proj))})
            prev.append(u)
        return out


@dataclass
class R3Stream:
    seed: int
    stream: int
    kind: str
    noise_seed: int
    sibling: bool
    env: NLEnv                      # ground truth: harness only
    init_obs: list
    problems: list                  # public Problem objects
    J: list                         # public class J tables (transformed for offset_null)
    harness: list                   # per problem: true gap / layer (scoring only)
    theta_index: int                # harness only
    truth: dict = field(default_factory=dict)


def offset_null(q: Problem, J: np.ndarray, ti: int):
    """Offset-null copy of problem q and its class J table: all candidates have equal true value."""
    Jt = J[ti]
    b = float(Jt.max()) - Jt
    s = 1.0 + float(b.max())
    q2 = copy.copy(q)
    q2.meta = {**q.meta, "offset_b": b.tolist(), "offset_scale": s, "offset_null": True,
               "u_max": (q.meta.get("u_max", 1.0) + float(b.max())) / s}
    q2.pid = q.pid + "_null"
    J2 = (J + b[None, :]) / s
    return q2, J2


def fill_instances(requested, ok_fn, reserve_start: int, max_reserve: int = 1000):
    """Deterministic replacement of quota_fail instances: walk the requested seeds in order; each failing seed is
    replaced by the next unused reserve seed (reserve_start, reserve_start + 1, ...) that passes `ok_fn`."""
    out, log, r = [], [], reserve_start
    for s in requested:
        if ok_fn(s):
            out.append(s)
            continue
        while r < reserve_start + max_reserve:
            cand = r
            r += 1
            if ok_fn(cand):
                out.append(cand)
                log.append({"failed": int(s), "replacement": int(cand)})
                break
        else:
            raise RuntimeError("reserve seeds exhausted")
    return out, log
