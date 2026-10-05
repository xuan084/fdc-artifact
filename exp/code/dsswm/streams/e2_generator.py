"""E2 instance / problem-stream generator (harness side: creates ground truth).

theta* drawn uniformly from the generator box (strictly inside the model-class box):
  alpha U[-1.5,1.5], gamma U[0,0.75], tauL U[0.5,1.5], beta U[-1,1], tauR U[0.5,1.5], psi U[0,1.5], lam U[0,0.5]
Initial data: n0 rounds of uniformly random legal actions (random matching of size 3..6, incentive on one random
matched pair w.p. 1/2), shared by all methods. Type-1 problem stream: K problems with random s0, H=8, 4..8 policies
from the eight families, random utility weights (with retention term), U_q in [0, 1].
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field

import numpy as np

from ..envs.e2 import E2Env
from ..models import e2_model as M
from .e2_policies import FAMILIES, sample_e2_policy
from .utilities import sample_utility

E2_DEFAULTS = dict(n0=40, K=10, H=8, n_pol=(4, 8), p_engaged0=0.85)
GEN_BOX = dict(alpha=(-1.5, 1.5), gamma=(0.0, 0.75), tauL=(0.5, 1.5), beta=(-1.0, 1.0), tauR=(0.5, 1.5),
               psi=(0.0, 1.5), lam=(0.0, 0.5))


@dataclass
class E2Problem:
    pid: str
    H: int
    loads0: np.ndarray
    engaged0: np.ndarray
    policies: list
    utility: object
    aff: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class E2Instance:
    seed: int
    noise_seed: int
    env: E2Env           # ground truth (evaluation code only)
    init_obs: list
    problems: list
    cfg: dict


def e2_generator_hash() -> str:
    from . import e2_policies, utilities
    src = inspect.getsource(inspect.getmodule(e2_generator_hash)) + inspect.getsource(e2_policies) + \
        inspect.getsource(utilities)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def random_action(rng, L=6, R=6):
    k = int(rng.integers(3, 7))
    lefts = rng.choice(L, k, replace=False)
    rights = rng.choice(R, k, replace=False)
    mt = np.zeros((L, R), bool); mt[lefts, rights] = True
    inc = np.zeros((L, R), np.int64)
    if rng.random() < 0.5:
        q = int(rng.integers(k)); inc[lefts[q], rights[q]] = 1
    return mt, inc


def make_e2_instance(seed: int, noise_seed: int = 42, device="cuda", **overrides) -> E2Instance:
    cfg = {**E2_DEFAULTS, **overrides}
    rng = np.random.default_rng([seed, 21])
    gb = GEN_BOX
    v = np.concatenate([rng.uniform(*gb["alpha"], M.L), rng.uniform(*gb["gamma"], M.L), rng.uniform(*gb["tauL"], M.L),
                        rng.uniform(*gb["beta"], M.R), rng.uniform(*gb["tauR"], M.R),
                        [rng.uniform(*gb["psi"])], [rng.uniform(*gb["lam"])]])
    env = E2Env(v, seed=int(seed) * 1000 + int(noise_seed), device=device)
    irng = np.random.default_rng([seed, 22])
    init = [env.step(*random_action(irng)) for _ in range(cfg["n0"])]
    prng = np.random.default_rng([seed, 23])
    problems = []
    for k in range(cfg["K"]):
        H = int(cfg["H"])
        loads0 = prng.integers(0, M.NMAX + 1, M.P)
        eng0 = (prng.random(M.P) < cfg["p_engaged0"]).astype(np.int64)
        aff = prng.standard_normal((M.L, M.R))
        n = int(prng.integers(cfg["n_pol"][0], cfg["n_pol"][1] + 1))
        pols, names, tries = [], set(), 0
        while len(pols) < n and tries < 200:
            tries += 1
            p = sample_e2_policy(str(prng.choice(FAMILIES)), prng, H, aff, device=device)
            if p.name in names:
                continue
            names.add(p.name); pols.append(p)
        util = sample_utility(prng, H, M.L, M.R, with_retention=True, y_scale=1.0)
        m = min(M.L, M.R)
        problems.append(E2Problem(f"e2_{seed}_q{k}", H, loads0, eng0, pols, util, aff,
                                  meta={"u_max": 1.0, "m": m}))
    return E2Instance(seed, noise_seed, env, init, problems, cfg)
