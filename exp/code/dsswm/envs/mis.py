"""E1-Mis: the learner keeps the E1-NL-S class, the truth adds structure outside it.

m1 pair synergy: logit += s_ij (not absorbable by individual effects; s has zero row/col means)
m2 saturating fatigue: -gamma_i * g(n), g concave (g(n) = gmax * (1 - exp(-n/scale)) normalised)
m3 hidden two-type mixture: psi takes one of two values according to a latent left-participant type
Strength is a scalar knob `strength`; calibration to eta_min / eps is done by downstream tasks.
"""
from __future__ import annotations

import numpy as np

from .nl import NLEnv


def _double_center(s):
    s = s - s.mean(1, keepdims=True)
    return s - s.mean(0, keepdims=True)


def make_mis_env(kind: str, base: dict, strength: float, rng: np.random.Generator, **env_kwargs) -> NLEnv:
    L, R = len(base["alpha"]), len(base["beta"])
    nmax = env_kwargs.get("nmax", 2)
    kw = dict(base)
    if kind == "m1":
        s = _double_center(rng.standard_normal((L, R)))
        s = s / max(np.abs(s).max(), 1e-12)
        kw["syn"] = strength * s
    elif kind == "m2":
        n = np.arange(nmax + 1, dtype=float)
        conc = nmax * (1 - np.exp(-n / 0.5)) / (1 - np.exp(-nmax / 0.5))   # concave, same endpoint
        kw["g"] = (1 - strength) * n + strength * conc if strength <= 1 else conc
    elif kind == "m3":
        psi = float(np.asarray(base["psi"]).reshape(-1)[0])
        types = rng.integers(0, 2, size=L)
        nb = 2
        psi_left = np.zeros((L, nb))
        psi_left[:, 1] = np.where(types == 1, psi + strength, max(psi - strength, -2.0))
        kw["psi_left"] = psi_left
    else:
        raise ValueError(kind)
    env = NLEnv(**kw, **env_kwargs)
    env.kind = f"nl_mis_{kind}"
    return env


# ---------------------------------------------------------------- m4: temporal drift (unregistered direction)
M4_T_REF = 1000.0            # registered drift time scale (platform rounds)
M4_NOMINAL_STEPS = 200       # registered nominal new steps per problem for the stream-level calibration schedule


class DriftNLEnv(NLEnv):
    """m4: alpha_i(t) = alpha_i + s * t_global / T_ref (t_global = platform round counter env.t).
    The learner class keeps static alpha; no registered falsification direction contains time."""
    kind = "nl_mis_m4"

    def __init__(self, *args, drift_s: float = 0.0, t_ref: float = M4_T_REF, **kwargs):
        super().__init__(*args, **kwargs)
        self._drift_s = float(drift_s)
        self._t_ref = float(t_ref)
        self._alpha0 = self._alpha.copy()

    def _shift(self, t):
        return self._drift_s * float(t) / self._t_ref

    def step(self, a_idx: int):
        self._alpha = self._alpha0 + self._shift(self.t)
        return super().step(a_idx)

    def simulate_batch(self, policy, loads0, engaged0, H, n_episodes, rng):
        self._alpha = self._alpha0 + self._shift(self.t)
        return super().simulate_batch(policy, loads0, engaged0, H, n_episodes, rng)

    def true_params_at(self, t):
        """Truth frozen at platform round t (exact up to the within-horizon drift s * H / T_ref)."""
        tp = self.true_params()
        tp["alpha"] = self._alpha0 + self._shift(t)
        return tp


def m4_schedule(n0: int, K: int, nominal_steps: int = M4_NOMINAL_STEPS):
    """Nominal start round of problem k in the calibration schedule."""
    return [n0 + k * nominal_steps for k in range(K)]


def calibrate_m4(J_class_list, J_true_fn, eta_target: float, t_sched, t_ref: float = M4_T_REF, s_max: float = 50.0,
                 rel_tol: float = 1e-3):
    """Bisection on the drift slope s >= 0 so that
        eta_min(s) = min_theta max_{k, pi} |J_true(k, pi; s) - J_theta(k, pi)| = eta_target
    over the whole problem stream (problem k evaluated with alpha frozen at its nominal start round t_sched[k]).
    J_class_list: list of (B, n_pi_k) class J tables; J_true_fn(alpha_shift) -> list of (n_pi_k,) true J vectors.
    Returns dict(s, eta_min, reached, iters, theta_min)."""
    Jall = np.concatenate(J_class_list, 1)

    def eta_min(s):
        jt = np.concatenate([J_true_fn(k, s * t_sched[k] / t_ref) for k in range(len(J_class_list))])
        dev = np.abs(Jall - jt[None]).max(1)
        b = int(np.argmin(dev))
        return float(dev[b]), b

    e0, b0 = eta_min(0.0)
    if eta_target <= e0:
        return {"s": 0.0, "eta_min": e0, "reached": bool(abs(e0 - eta_target) <= rel_tol * max(eta_target, 1e-12)
                                                          or eta_target <= e0), "iters": 0, "theta_min": b0,
                "eta_min_at_0": e0}
    lo, hi, it = 0.0, 0.1, 0
    while eta_min(hi)[0] < eta_target and hi < s_max:
        lo, hi = hi, hi * 2
        it += 1
    v = None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        v, _b = eta_min(mid)
        it += 1
        if abs(v - eta_target) <= rel_tol * eta_target:
            break
        if v < eta_target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi) if v is None or abs(v - eta_target) > rel_tol * eta_target else mid
    v, b = eta_min(s)
    return {"s": float(s), "eta_min": v, "reached": bool(abs(v - eta_target) <= 5 * rel_tol * eta_target), "iters": it,
            "theta_min": b, "eta_min_at_0": e0}


def make_m4_env(base: dict, drift_s: float, t_ref: float = M4_T_REF, **env_kwargs) -> DriftNLEnv:
    """base: in-class truth dict (alpha, beta, gamma, tauL, tauR, psi, lam) as in generator truth."""
    return DriftNLEnv(base["alpha"], base["beta"], base["gamma"], base["tauL"], base["tauR"], base["psi"],
                      base["lam"], drift_s=drift_s, t_ref=t_ref, **env_kwargs)
