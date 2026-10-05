"""Pre-registered instance / problem-stream generator (harness side: it creates ground truth).

An environment instance = true population theta* + shared initial data (n0 rounds of uniform random legal
actions executed on the real env) + a problem stream of K problems. Instances are the statistical unit.
Learner code receives only: EnvHandle, the authenticated initial observations and the public Problem objects.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field

import numpy as np

from ..envs.lin import LinEnv, StaticLinEnv
from ..envs.nl import NLEnv
from ..models.nl_class import NLClass, NLGrid
from .policies import FAMILIES, sample_policy
from .utilities import Utility, sample_utility

LIN_DEFAULTS = dict(L=3, R=3, sigma=1.5, nmax=3, budget=1, n0=20, K=10, H_choices=(4, 6, 8), n_pol=(4, 8),
                    prior=dict(alpha=(0.0, 1.0), beta=(0.0, 1.0), gamma=(0.0, 0.6), psi=(0.0, 0.4), psi2=(0.0, 0.8)),
                    y_scale=2.4, static=False)
NL_DEFAULTS = dict(L=2, R=2, nmax=2, budget=1, c=1.0, rho_ret=0.3, n0=20, K=10, H_choices=(6, 8), n_pol=(4, 8),
                   p_engaged0=0.85, kappa_mode="natural")


@dataclass
class Problem:
    pid: str
    ptype: int
    H: int
    loads0: np.ndarray
    engaged0: np.ndarray
    policies: list
    utility: Utility
    aff: np.ndarray
    meta: dict = field(default_factory=dict)

    def public_dict(self):
        return {"pid": self.pid, "ptype": self.ptype, "H": self.H, "loads0": self.loads0.tolist(),
                "engaged0": self.engaged0.tolist(), "policies": [p.name for p in self.policies],
                "utility": self.utility.to_dict(), "meta": self.meta}


@dataclass
class Instance:
    seed: int
    noise_seed: int
    env_kind: str
    env: object                 # ground truth: evaluation code only
    init_obs: list
    problems: list
    truth: dict                 # ground truth summary: evaluation code only
    cfg: dict


def generator_hash() -> str:
    from . import policies, utilities
    src = inspect.getsource(inspect.getmodule(generator_hash)) + inspect.getsource(policies) + inspect.getsource(utilities)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _sample_policies(rng, aspace, H, aff, nmax, n_pol, level=1, families=FAMILIES):
    k = int(rng.integers(n_pol[0], n_pol[1] + 1))
    pols, names, tries = [], set(), 0
    while len(pols) < k and tries < 200:
        tries += 1
        fam = str(rng.choice(families))
        p = sample_policy(fam, rng, aspace, H, aff, nmax, level=level)
        if p.name in names:
            continue
        names.add(p.name)
        pols.append(p)
    return pols


def _initial_data(env, n0, rng, max_level=1):
    allowed = [k for k in range(env.aspace.n) if env.aspace.max_level(k) <= max_level]
    return [env.step(int(rng.choice(allowed))) for _ in range(n0)]


def make_lin_instance(seed: int, noise_seed: int = 42, ptypes=(1,), **overrides) -> Instance:
    cfg = {**LIN_DEFAULTS, **overrides}
    cfg["prior"] = {**LIN_DEFAULTS["prior"], **overrides.get("prior", {})}
    rng = np.random.default_rng([seed, 1])
    L, R, pr = cfg["L"], cfg["R"], cfg["prior"]
    alpha = rng.uniform(*pr["alpha"], L)
    beta = rng.uniform(*pr["beta"], R)
    gamma = rng.uniform(*pr["gamma"], L)
    psi = np.array([rng.uniform(*pr["psi"]), rng.uniform(*pr["psi2"])])
    Env = StaticLinEnv if cfg["static"] else LinEnv
    env = Env(alpha, beta, gamma, psi, sigma=cfg["sigma"], L=L, R=R, nmax=cfg["nmax"], budget=cfg["budget"],
              incentive_levels=(1, 2), seed=int(seed) * 1000 + int(noise_seed))
    init = _initial_data(env, cfg["n0"], np.random.default_rng([seed, 2]))
    prng = np.random.default_rng([seed, 3])
    problems = []
    for k in range(cfg["K"]):
        ptype = int(prng.choice(ptypes))
        H = int(prng.choice(cfg["H_choices"]))
        loads0 = prng.integers(0, cfg["nmax"] + 1, L + R)
        aff = prng.standard_normal((L, R))
        pols = _sample_policies(prng, env.aspace, H, aff, cfg["nmax"], cfg["n_pol"])
        if ptype == 2:   # new incentive level b=2: never used in the initial data
            inc_fams = ("front_incentive", "uniform_incentive", "late_incentive")
            n2 = int(prng.integers(1, 3))
            pols = pols[: max(len(pols) - n2, 2)] + [sample_policy(str(prng.choice(inc_fams)), prng, env.aspace, H, aff,
                                                                   cfg["nmax"], level=2) for _ in range(n2)]
        util = sample_utility(prng, H, L, R, with_retention=False, y_scale=cfg["y_scale"])
        problems.append(Problem(f"lin{seed}_q{k}", ptype, H, loads0, np.ones(L + R, dtype=np.int64), pols, util, aff))
    truth = {"theta": env.true_theta_vector().tolist(), "sigma": cfg["sigma"]}
    return Instance(seed, noise_seed, env.kind, env, init, problems, truth, cfg)


# Calibrated on dev instances (seeds 200-239) in setup_env_core pilot; |Theta| = 13824.
DEFAULT_NL_S_GRID = NLGrid(L=2, R=2, alpha=(-1.5, 0.0, 1.5), gamma=(0.0, 0.75), tauL=(0.5, 1.5), beta=(-1.0, 1.0),
                           tauR=(0.5, 1.5), psi=(0.0, 0.75, 1.5), lam=(0.0, 0.5))
# E1-NL-M (3x3) reduced grid for the exact-enumeration feasibility check; |Theta| = 2048.
DEFAULT_NL_M_GRID = NLGrid(L=3, R=3, alpha=(-1.0, 1.0), gamma=(0.0, 0.75), tauL=(1.0,), beta=(0.0,),
                           tauR=(0.5, 1.5), psi=(0.0, 1.5), lam=(0.0, 0.5))


def make_nl_instance(seed: int, noise_seed: int = 42, grid: NLGrid | None = None, nl_class: NLClass | None = None,
                     ptypes=(1,), **overrides) -> Instance:
    cfg = {**NL_DEFAULTS, **overrides}
    grid = grid or DEFAULT_NL_S_GRID
    L, R = grid.L, grid.R
    rng = np.random.default_rng([seed, 11])
    lt, rt = grid.left_tuples(), grid.right_tuples()
    km = cfg["kappa_mode"]

    def pick_left():
        cand = lt
        if km == "zero":
            cand = [x for x in lt if x[1] == 0.0]
        elif km == "high":
            cand = [x for x in lt if x[1] == max(grid.gamma)]
        return cand[int(rng.integers(len(cand)))]

    lefts = [pick_left() for _ in range(L)]
    rights = [rt[int(rng.integers(len(rt)))] for _ in range(R)]
    psis = list(grid.psi) if km != "high" else [p for p in grid.psi if p >= np.median(grid.psi)]
    lams = list(grid.lam) if km == "natural" else ([0.0] if km == "zero" else [max(grid.lam)])
    psi = float(psis[int(rng.integers(len(psis)))])
    lam = float(lams[int(rng.integers(len(lams)))])
    alpha = np.array([x[0] for x in lefts]); gamma = np.array([x[1] for x in lefts]); tauL = np.array([x[2] for x in lefts])
    beta = np.array([x[0] for x in rights]); tauR = np.array([x[1] for x in rights])
    env = NLEnv(alpha, beta, gamma, tauL, tauR, [psi], lam, c=cfg["c"], rho_ret=cfg["rho_ret"], L=L, R=R,
                nmax=cfg["nmax"], budget=cfg["budget"], incentive_levels=(1,), seed=int(seed) * 1000 + int(noise_seed))
    init = _initial_data(env, cfg["n0"], np.random.default_rng([seed, 12]))
    prng = np.random.default_rng([seed, 13])
    problems = []
    for k in range(cfg["K"]):
        H = int(prng.choice(cfg["H_choices"]))
        loads0 = prng.integers(0, cfg["nmax"] + 1, L + R)
        eng0 = (prng.random(L + R) < cfg["p_engaged0"]).astype(np.int64)
        aff = prng.standard_normal((L, R))
        pols = _sample_policies(prng, env.aspace, H, aff, cfg["nmax"], cfg["n_pol"])
        util = sample_utility(prng, H, L, R, with_retention=True, y_scale=1.0)
        problems.append(Problem(f"nl{seed}_q{k}", 1, H, loads0, eng0, pols, util, aff))
    truth = {"alpha": alpha.tolist(), "beta": beta.tolist(), "gamma": gamma.tolist(), "tauL": tauL.tolist(),
             "tauR": tauR.tolist(), "psi": psi, "lam": lam}
    if nl_class is not None:
        truth["theta_index"] = nl_class.index_of(tuple(alpha), tuple(beta), tuple(gamma), tuple(tauL), tuple(tauR), psi, lam)
    return Instance(seed, noise_seed, env.kind, env, init, problems, truth, cfg)


def nl_class_max_jtable(problem: Problem, prop, params, cache_path=None):
    """Exact J table over the class with the utility scale fixed a priori by the model class:
    c_q = 1 / max_{theta in Theta, pi in Pi_q} J_raw  (public, data-free), so J_theta(pi) in [0, 1] on the class.
    Sets problem.utility.c_q and problem.meta['u_max'] (upper bound of a single-trajectory utility)."""
    import os
    import numpy as _np
    raw_u = Utility(w=problem.utility.w, w_ret=problem.utility.w_ret, c_q=1.0)
    if cache_path and os.path.exists(cache_path):
        raw = _np.load(cache_path)
    else:
        raw = prop.j_table(params, problem.policies, problem.loads0, problem.engaged0, problem.H, raw_u)
        if cache_path:
            _np.save(cache_path, raw)
    c_q = 1.0 / float(raw.max())
    problem.utility.c_q = c_q
    m = min(prop.L, prop.R)
    problem.meta["u_max"] = c_q * (m * float(problem.utility.w.sum()) + problem.utility.w_ret * prop.P)
    problem.meta["normalisation"] = "class_max"
    return raw * c_q
