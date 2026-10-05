"""B1 / B2: whole-trial best-arm identification with anytime-valid confidence sequences (LUCB sampling).

Each 'pull' of arm k is one full H-step trial of candidate policy k from the problem's s0 (cost H env steps).
Trajectory utilities lie in [0, u_max]. Stopping rule (same epsilon-certificate as JPC):
    stop when  max_{k != k_hat} UCB_k - LCB_{k_hat} <= eps   (k_hat = best empirical mean)
B1: Hoeffding CS with a union bound over arms and sample sizes (delta_n = 6 delta / (pi^2 K n^2)).
B2: proxy-centred residual audit: per-arm estimate J_proxy(k) + mean residual (U - J_proxy(k)),
    with an empirical-Bernstein (Maurer-Pontil) CS on the residual, same union bound.
    NB: with a fixed s0 and no covariates the proxy is a per-arm constant, so it does not reduce the residual
    variance; B2 differs from B1 by the variance-adaptive CS and by proxy-informed warm start of the leader.
Arm pulls are drawn by a `sampler(k, n)` callback that returns n trajectory utilities from the real platform.
History reuse: `prior` maps arm -> previously observed utilities (re-scored under the new utility).
"""
from __future__ import annotations

import math

import numpy as np


def _delta_n(delta, K, n):
    return 6.0 * delta / (math.pi ** 2 * K * np.maximum(n, 1) ** 2)


def hoeffding_rad(n, u_range, delta, K):
    n = np.maximum(n, 1)
    return u_range * np.sqrt(np.log(2.0 / _delta_n(delta, K, n)) / (2.0 * n))


def eb_rad(n, var, u_range, delta, K):
    n = np.maximum(n, 2)
    lg = np.log(4.0 / _delta_n(delta, K, n))     # two-sided
    return np.sqrt(2.0 * var * lg / n) + 7.0 * u_range * lg / (3.0 * (n - 1))


class _Stream:
    """Pre-drawn trajectory pool per arm (consumed in order) to keep CRN pairing across B1 and B2."""

    def __init__(self, sampler, K, chunk=512):
        self.sampler, self.chunk = sampler, chunk
        self.pool = [np.zeros(0) for _ in range(K)]
        self.pos = np.zeros(K, dtype=int)

    def draw(self, k, b=1):
        while self.pos[k] + b > len(self.pool[k]):
            n_new = max(self.chunk, len(self.pool[k]), b)          # geometric growth keeps sampling vectorised
            self.pool[k] = np.concatenate([self.pool[k], self.sampler(k, n_new)])
        v = self.pool[k][self.pos[k]:self.pos[k] + b]
        self.pos[k] += b
        return v


def lucb(sampler, K, H, eps, delta, u_max, max_steps, kind="hoeffding", proxy=None, prior=None, init_pulls=2,
         batch_frac=0.02):
    """Returns dict(status, pi, steps, pulls, censored, history={arm: utilities}).
    Batched LUCB: each round pulls leader and challenger max(1, batch_frac * n_k) times (overshoot <= batch_frac)."""
    stream = _Stream(sampler, K)
    obs = [list(prior.get(k, [])) if prior else [] for k in range(K)]
    n_reused = sum(len(o) for o in obs)
    proxy = np.zeros(K) if proxy is None else np.asarray(proxy, float)
    n = np.array([len(o) for o in obs], float)
    s1 = np.array([np.sum(np.asarray(o) - proxy[k]) for k, o in enumerate(obs)], float)
    s2 = np.array([np.sum((np.asarray(o) - proxy[k]) ** 2) for k, o in enumerate(obs)], float)
    steps = 0
    pulls = 0

    def pull(k, b=1):
        nonlocal steps, pulls
        b = int(max(1, min(b, (max_steps - steps) // H)))
        v = stream.draw(k, b)
        r = v - proxy[k]
        n[k] += b; s1[k] += r.sum(); s2[k] += (r * r).sum()
        steps += H * b
        pulls += b

    for k in range(K):
        while n[k] < init_pulls and steps + H <= max_steps:
            pull(k)
    gap = float("inf")
    while True:
        if (n < 1).any():
            return {"status": "NEED_DATA", "pi": None, "steps": steps, "pulls": pulls, "censored": True,
                    "n_reused": n_reused, "n_per_arm": n.copy(), "gap": gap}
        mu = proxy + s1 / n
        if kind == "hoeffding":
            rad = hoeffding_rad(n, u_max, delta, K)
        else:  # empirical Bernstein on residuals around the proxy
            var = np.where(n > 1, (s2 - s1 ** 2 / n) / np.maximum(n - 1, 1), u_max ** 2 / 4)
            rad = eb_rad(n, np.maximum(var, 0.0), u_max, delta, K)
        lcb, ucb = mu - rad, mu + rad
        k_hat = int(np.argmax(mu))
        ucb_o = ucb.copy(); ucb_o[k_hat] = -np.inf
        ch = int(np.argmax(ucb_o))
        gap = float(ucb_o[ch] - lcb[k_hat]) if K > 1 else -np.inf
        if gap <= eps:
            return {"status": "CERTIFIED", "pi": k_hat, "steps": steps, "pulls": pulls, "censored": False,
                    "n_reused": n_reused, "n_per_arm": n.copy(), "gap": gap}
        if steps + 2 * H > max_steps:
            return {"status": "NEED_DATA", "pi": k_hat, "steps": steps, "pulls": pulls, "censored": True,
                    "n_reused": n_reused, "n_per_arm": n.copy(), "gap": gap}
        b = int(batch_frac * min(n[k_hat], n[ch]))
        pull(k_hat, b)
        pull(ch, b)
