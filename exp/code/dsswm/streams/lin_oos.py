"""E1-Lin non-near-tie OUT_OF_SCOPE family (methodology section 2.5; harness side: creates ground truth).

Two Type-2 constructions on E1-Lin 3x3 (sigma = 1.5, eps = 0.05, round-0 prior box unless stated):
  (a) 'squeeze'  psi2 prior box widened to [-0.8, 0.8] (crowding-out tier), candidates mix level-1 / level-2
                 incentives (generator ptype 2), level-2 probes ILLEGAL -> psi2 lies in R(legal library)^perp.
  (b) 'newpart'  right participant j_new never enters a legal probe nor the shared initial data (initial rounds
                 drawn uniformly from the legal actions only); candidates are the generator ptype-1 policies over
                 all participants of an E1-Lin 3x4 platform (j_new = 3) PLUS a 'swap' twin of each of them (j_new
                 exchanged with a random legal right participant), so the ranking depends on beta_{j_new}. NB: beta_{j_new} and the alpha/beta
                 gauge direction both lie in R^perp, so the unidentified component of a candidate difference is in
                 general NOT coordinate aligned ('aligned' = False in the manifest); learner-side OOS lower bounds
                 must handle the fibre (an LP), not a coordinate box.
Truth label (independent of any learner code): V* = min_pi max_pi' sup{<z_pi' - z_pi, theta> : theta in prior box,
P_R theta = P_R theta*} solved by HiGHS LP; OUT_OF_SCOPE iff V* > eps; near tie iff |V* - eps| < 0.2 eps.
The family keeps only truly OOS, non-near-tie problems (V* >= 1.2 eps).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ..certify.lin_closed import range_projector
from ..envs.lin import LinEnv
from ..models.lin_class import LinClass
from .generator import LIN_DEFAULTS, Instance, make_lin_instance

LIN_OOS_EPS = 0.05
NEAR_TIE_FRAC = 0.2
SQUEEZE_PSI2_BOX = (-0.8, 0.8)
NEWPART_R = 4         # E1-Lin 3x4 for the new-participant family; right participant 3 is the new one
NEWPART_J = 3


class SwapRightPolicy:
    """Public candidate: the base policy's action with right participants j_a and j_b exchanged in every pair
    (and incentive). On E1-Lin the right load does not enter the outcome mean, so a base / swap twin differs only
    through beta_{j_a} - beta_{j_b} on the rounds where exactly one of them is matched."""

    def __init__(self, base, j_a: int, j_b: int, aspace):
        self.base, self.ja, self.jb, self.aspace = base, int(j_a), int(j_b), aspace
        self.level = getattr(base, "level", 1)
        self.family = getattr(base, "family", "?") + "_swap"

    @property
    def name(self):
        return f"{self.base.name}|swap_r{self.ja}r{self.jb}"

    def uses_level(self, lv: int) -> bool:
        return self.base.uses_level(lv)

    def _sw(self, j):
        return self.jb if j == self.ja else (self.ja if j == self.jb else j)

    def act(self, t, loads, engaged) -> int:
        a = self.aspace.actions[self.base.act(t, loads, engaged)]
        pairs = tuple((i, self._sw(j)) for i, j in a.pairs)
        incs = tuple((i, self._sw(j), l) for i, j, l in a.incentives)
        return self.aspace.index(pairs, incs)
FAMILIES = ("squeeze", "newpart")


class LinLegality:
    def __init__(self, maxlev: int = 1, excluded_right=(), excluded_left=()):
        self.maxlev = int(maxlev)
        self.excluded_right = tuple(excluded_right)
        self.excluded_left = tuple(excluded_left)

    def is_legal(self, aspace, a_idx) -> bool:
        if aspace.max_level(a_idx) > self.maxlev:
            return False
        a = aspace.actions[a_idx]
        return not any(i in self.excluded_left or j in self.excluded_right for i, j in a.pairs)

    def legal_actions(self, aspace) -> np.ndarray:
        return np.array([a for a in range(aspace.n) if self.is_legal(aspace, a)], dtype=np.int64)

    def library(self, lc: LinClass) -> np.ndarray:
        rows = [lc.feature(i, j, n, b) for i in range(lc.L) for j in range(lc.R) for n in range(lc.nmax + 1)
                for b in range(self.maxlev + 1) if i not in self.excluded_left and j not in self.excluded_right]
        return np.array(rows)

    def to_dict(self):
        return {"maxlev": self.maxlev, "excluded_right": list(self.excluded_right),
                "excluded_left": list(self.excluded_left)}


def prior_box(lc: LinClass, prior: dict):
    lo, hi = np.zeros(lc.d), np.zeros(lc.d)
    for k, n in enumerate(lc.names):
        key = "psi2" if n == "psi2" else ("psi" if n == "psi1" else n.rstrip("0123456789"))
        lo[k], hi[k] = prior[key]
    return lo, hi


def truth_vstar(Z: np.ndarray, theta: np.ndarray, PR: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float,
                return_argmax: bool = False) -> dict:
    """Harness: V* by LP (HiGHS) over the prior box intersected with the identified fibre of theta*."""
    U, s, _ = np.linalg.svd(PR)
    NR = U[:, s > 0.5]
    A_eq, b_eq = NR.T, NR.T @ theta
    K = len(Z)
    W = np.zeros((K, K))
    arg = {}
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            d = Z[b] - Z[a]
            if np.abs(d).max() < 1e-14:
                continue
            r = linprog(-d, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lo, hi)), method="highs")
            if r.status != 0:
                raise RuntimeError(f"truth LP failed: {r.message}")
            W[a, b] = -r.fun
            if return_argmax:
                arg[(a, b)] = r.x
    Va = W.max(1)
    vstar = float(Va.min())
    out = {"V_star": vstar, "label": "OUT_OF_SCOPE" if vstar > eps else "NEED_DATA",
           "near_tie": bool(abs(vstar - eps) < NEAR_TIE_FRAC * eps), "pi_safe": int(np.argmin(Va)), "W": W}
    if return_argmax:
        out["argmax"] = arg
    return out


def _alignment(Z, PR):
    D = (Z[None] - Z[:, None]).reshape(-1, Z.shape[1])
    Dp = D - D @ PR
    perp_coord = np.linalg.norm(PR, axis=0) < 1e-9
    return bool(np.all(np.abs(Dp[:, ~perp_coord]) < 1e-9)), float(np.abs(Dp).max())


def _rebuild_with_legal_init(inst: Instance, legality: LinLegality) -> Instance:
    """Same truth / noise seed / problems; initial rounds re-drawn uniformly from the LEGAL actions only."""
    e = inst.env
    cfg = inst.cfg
    env = LinEnv(e._alpha, e._beta, e._gamma, e._psi[1:], sigma=cfg["sigma"], L=cfg["L"], R=cfg["R"],
                 nmax=cfg["nmax"], budget=cfg["budget"], incentive_levels=(1, 2),
                 seed=int(inst.seed) * 1000 + int(inst.noise_seed))
    allowed = legality.legal_actions(env.aspace)
    rng = np.random.default_rng([inst.seed, 2])
    init = [env.step(int(rng.choice(allowed))) for _ in range(cfg["n0"])]
    return Instance(inst.seed, inst.noise_seed, env.kind, env, init, inst.problems, inst.truth, cfg)


def make_oos_instance(family: str, seed: int, noise_seed: int = 42, K: int = 10):
    """Returns (instance, legality, prior). Learner-visible: env handle, init_obs, problems, legality, prior box."""
    if family == "squeeze":
        prior = {**LIN_DEFAULTS["prior"], "psi2": SQUEEZE_PSI2_BOX}
        inst = make_lin_instance(seed, noise_seed=noise_seed, ptypes=(2,), K=K, prior={"psi2": SQUEEZE_PSI2_BOX})
        leg = LinLegality(maxlev=1)
        inst.cfg["oos_family"] = family
        return inst, leg, prior
    if family == "newpart":
        prior = dict(LIN_DEFAULTS["prior"])
        inst = make_lin_instance(seed, noise_seed=noise_seed, ptypes=(1,), K=K, R=NEWPART_R)
        leg = LinLegality(maxlev=1, excluded_right=(NEWPART_J,))
        inst = _rebuild_with_legal_init(inst, leg)
        prng = np.random.default_rng([seed, 31])
        for q in inst.problems:
            partners = [int(prng.integers(0, NEWPART_J)) for _ in q.policies]
            q.policies = list(q.policies) + [SwapRightPolicy(p, NEWPART_J, jb, inst.env.aspace)
                                             for p, jb in zip(q.policies, partners)]
            q.meta["swap_partners"] = partners
        inst.cfg["oos_family"] = family
        return inst, leg, prior
    raise ValueError(family)


def family_manifest(family: str, seeds, noise_seed: int = 42, K: int = 10, eps: float = LIN_OOS_EPS) -> list[dict]:
    """One record per candidate problem; 'keep' marks truly OOS, non-near-tie problems (V* >= 1.2 eps)."""
    out = []
    for seed in seeds:
        inst, leg, prior = make_oos_instance(family, seed, noise_seed=noise_seed, K=K)
        env = inst.env
        lc = LinClass(env.L, env.R, env.aspace.nb, nmax=env.nmax)
        lo, hi = prior_box(lc, prior)
        theta = env.true_theta_vector()
        assert np.all(theta >= lo - 1e-12) and np.all(theta <= hi + 1e-12)
        PR = range_projector(leg.library(lc))
        X0 = np.concatenate([lc.obs_rows(o)[0] for o in inst.init_obs], 0)
        init_legal = all(leg.is_legal(env.aspace, o.action) for o in inst.init_obs)
        for k, q in enumerate(inst.problems):
            Z = np.stack([lc.z(p, q.loads0, q.H, q.utility, env.aspace) for p in q.policies])
            tl = truth_vstar(Z, theta, PR, lo, hi, eps)
            aligned, dperp = _alignment(Z, PR)
            uses_new = int(sum(any(NEWPART_J == j for j in _pairs_used(p, q, env)) for p in q.policies)) \
                if family == "newpart" else None
            Jt = Z @ theta
            out.append({"family": family, "seed": int(seed), "noise_seed": int(noise_seed), "problem_index": k,
                        "pid": q.pid, "H": q.H, "n_policies": len(q.policies),
                        "policies": [p.name for p in q.policies],
                        "n_level2_policies": int(sum(getattr(p, "level", 1) == 2 for p in q.policies)),
                        "n_policies_using_new_participant": uses_new,
                        "V_star": tl["V_star"], "true_label": tl["label"], "near_tie": tl["near_tie"],
                        "keep": bool(tl["label"] == "OUT_OF_SCOPE" and not tl["near_tie"]),
                        "aligned": aligned, "d_perp_max": dperp, "pi_safe": tl["pi_safe"],
                        "true_gap_J": float(Jt.max() - np.sort(Jt)[-2]) if len(Jt) > 1 else 0.0,
                        "init_data_legal": bool(init_legal), "init_rank": int(np.linalg.matrix_rank(X0)),
                        "legality": leg.to_dict(), "prior_box": {"lo": lo.tolist(), "hi": hi.tolist()},
                        "eps": eps})
    return out


def _pairs_used(policy, q, env):
    """Pairs matched along the (deterministic, A1) load trajectory of a policy from s0."""
    from ..core.dynamics import next_loads
    loads = np.asarray(q.loads0, np.int64).copy()
    ones = np.ones(env.P, dtype=np.int64)
    used = set()
    for t in range(q.H):
        a = policy.act(t, loads, ones)
        for i, j in env.aspace.actions[a].pairs:
            used.add(j)
        loads = next_loads(loads, env.aspace, a, env.nmax, False)
    return used
