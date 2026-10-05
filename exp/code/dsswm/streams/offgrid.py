"""Round-1 off-grid truth ladder (harness side: this module CREATES ground truth; learner code must not import it).

Truth levels (methodology section 2.2), all CRN-paired on the same instance seed:
  R1  continuous truth in the kappa_high sub-box, independent coordinates:
        alpha_i ~ U[-1.5, 1.5], beta_j ~ U[-1, 1], tau_p ~ U[0.5, 1.5], gamma_i ~ U[0.5, 0.75],
        lam ~ U[0.35, 0.5], psi ~ U[0.75, 1.5]
  R0  the R1 draw snapped coordinate-wise to the nearest point of the learner grid G_1 (Cartesian grid, so the
      coordinate-wise nearest point is the Voronoi nearest point). By construction gamma -> 0.75, lam -> 0.5,
      psi -> {0.75, 1.5}: the round-0 kappa_high support.
  R2 / R3 use the R1 truth (only the learner / certifier differ).
Matched twin (section 2.3): R0 truth with every gamma_i and lam set to 0, all other coordinates identical.
Dynamic dose g in {0, 0.5, 1}: gamma and lam multiplied by g (base = R0 truth unless `base='R1'`).

CRN: for one instance seed, every level / twin / dose shares
  * the initial-data action sequence (rng [seed, 12]) and the env noise seed (seed * 1000 + noise_seed),
  * the problem stream (rng [seed, 13], the round-0 generator code path; hash 794acc9d48ff8c75 unchanged),
so the first 10 problems of an off-grid instance equal round-0 `make_nl_instance(seed)` problems.
Streams: 3 per instance, (noise_seed, problem order) = (42, perm0 = identity), (123, perm1), (456, perm2);
perm_k for k >= 1 is a permutation drawn from rng [seed, 31, k].
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..envs.nl import NLEnv
from ..models.nl_class import NLGrid
from .generator import (DEFAULT_NL_S_GRID, NL_DEFAULTS, Instance, Problem, _initial_data, _sample_policies)
from .utilities import sample_utility

R1_BOX = {"alpha": (-1.5, 1.5), "beta": (-1.0, 1.0), "tau": (0.5, 1.5), "gamma": (0.5, 0.75),
          "lam": (0.35, 0.5), "psi": (0.75, 1.5)}
STREAMS = ((42, 0), (123, 1), (456, 2))
K_STREAM = 15
LEVELS = ("R0", "R1", "R2", "R3")


def sample_r1_truth(seed: int, L: int = 2, R: int = 2) -> dict:
    rng = np.random.default_rng([seed, 21])
    b = R1_BOX
    return {"alpha": rng.uniform(*b["alpha"], L), "beta": rng.uniform(*b["beta"], R),
            "tauL": rng.uniform(*b["tau"], L), "tauR": rng.uniform(*b["tau"], R),
            "gamma": rng.uniform(*b["gamma"], L), "lam": float(rng.uniform(*b["lam"])),
            "psi": float(rng.uniform(*b["psi"]))}


def _nearest(x, values):
    v = np.asarray(values, float)
    x = np.asarray(x, float)
    return v[np.argmin(np.abs(x[..., None] - v), axis=-1)]


def snap_to_grid(theta: dict, grid: NLGrid = DEFAULT_NL_S_GRID) -> dict:
    """Coordinate-wise nearest grid point (= Voronoi nearest point of a Cartesian grid)."""
    return {"alpha": _nearest(theta["alpha"], grid.alpha), "beta": _nearest(theta["beta"], grid.beta),
            "tauL": _nearest(theta["tauL"], grid.tauL), "tauR": _nearest(theta["tauR"], grid.tauR),
            "gamma": _nearest(theta["gamma"], grid.gamma), "lam": float(_nearest(theta["lam"], grid.lam)),
            "psi": float(_nearest(theta["psi"], grid.psi))}


def truth_for(seed: int, level: str = "R1", twin: bool = False, dose: float = 1.0, base: str | None = None,
              grid: NLGrid = DEFAULT_NL_S_GRID) -> dict:
    r1 = sample_r1_truth(seed, grid.L, grid.R)
    if base is None:
        base = "R0" if (twin or level == "R0") else "R1"
    th = snap_to_grid(r1, grid) if base == "R0" else {k: (np.array(v) if isinstance(v, np.ndarray) else v)
                                                       for k, v in r1.items()}
    th = {k: (np.asarray(v, float).copy() if isinstance(v, np.ndarray) else float(v)) for k, v in th.items()}
    g = 0.0 if twin else float(dose)
    th["gamma"] = th["gamma"] * g
    th["lam"] = th["lam"] * g
    return th


def truth_hash(th: dict) -> str:
    d = {k: (np.round(np.asarray(v, float), 12).tolist()) for k, v in sorted(th.items())}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def stream_perm(seed: int, k: int, K: int = K_STREAM) -> np.ndarray:
    if k == 0:
        return np.arange(K)
    return np.random.default_rng([seed, 31, k]).permutation(K)


def _problems(seed: int, aspace, L: int, R: int, K: int, cfg: dict):
    prng = np.random.default_rng([seed, 13])          # identical code path to generator.make_nl_instance
    problems = []
    for k in range(K):
        H = int(prng.choice(cfg["H_choices"]))
        loads0 = prng.integers(0, cfg["nmax"] + 1, L + R)
        eng0 = (prng.random(L + R) < cfg["p_engaged0"]).astype(np.int64)
        aff = prng.standard_normal((L, R))
        pols = _sample_policies(prng, aspace, H, aff, cfg["nmax"], cfg["n_pol"])
        util = sample_utility(prng, H, L, R, with_retention=True, y_scale=1.0)
        problems.append(Problem(f"nl{seed}_q{k}", 1, H, loads0, eng0, pols, util, aff, meta={"gen_index": k}))
    return problems


def make_offgrid_instance(seed: int, level: str = "R1", stream: int = 0, twin: bool = False, dose: float = 1.0,
                          base: str | None = None, grid: NLGrid = DEFAULT_NL_S_GRID, K: int = K_STREAM,
                          static: bool = False, **overrides) -> Instance:
    """One (instance, stream) of the off-grid ladder. `level` in R0..R3 (R2/R3 share the R1 truth)."""
    assert level in LEVELS, level
    cfg = {**NL_DEFAULTS, **overrides, "K": K}
    noise_seed, k_perm = STREAMS[stream]
    L, R = grid.L, grid.R
    th = truth_for(seed, "R0" if level == "R0" else "R1", twin=twin, dose=dose, base=base, grid=grid)
    env = NLEnv(th["alpha"], th["beta"], th["gamma"], th["tauL"], th["tauR"], [th["psi"]], th["lam"], c=cfg["c"],
                rho_ret=cfg["rho_ret"], L=L, R=R, nmax=cfg["nmax"], budget=cfg["budget"], incentive_levels=(1,),
                seed=int(seed) * 1000 + int(noise_seed), static=static)
    init = _initial_data(env, cfg["n0"], np.random.default_rng([seed, 12]))
    probs = _problems(seed, env.aspace, L, R, K, cfg)
    perm = stream_perm(seed, k_perm, K)
    problems = [probs[i] for i in perm]
    truth = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in th.items()}
    truth["sha256"] = truth_hash(th)
    truth["level"] = level
    truth["twin"] = bool(twin)
    truth["dose"] = 0.0 if twin else float(dose)
    cfg = {**cfg, "stream": stream, "noise_seed": noise_seed, "perm": perm.tolist(), "level": level}
    return Instance(seed, noise_seed, env.kind, env, init, problems, truth, cfg)


def manifest_row(seed: int) -> dict:
    """Truth hashes only (no learner runs): R0 / R1 / twin / dose-0.5 truths of one instance."""
    out = {"seed": int(seed)}
    for name, kw in {"R0": dict(level="R0"), "R1": dict(level="R1"), "twin": dict(level="R0", twin=True),
                     "dose0.5_R0": dict(level="R0", dose=0.5), "dose0.5_R1": dict(level="R1", dose=0.5)}.items():
        th = truth_for(seed, **kw)
        out[name] = truth_hash(th)
    out["perms"] = [stream_perm(seed, k).tolist() for _, k in STREAMS]
    return out
