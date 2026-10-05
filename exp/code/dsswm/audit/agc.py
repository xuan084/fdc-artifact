"""Audit-Gated Certification (AGC): on-policy audit of (pi_hat, binding challengers) before CERTIFIED.

Learner-side module: the platform is reached only through `sampler(arm, n)`, which must execute n real whole
trials of candidate `arm` from the problem's s0 and return their trajectory utilities (cost H env steps each).

Estimator (PPI form): mu_a = J_model(a) + mean_n(U - J_model(a)); empirical-Bernstein anytime CS on the residual
with a union bound over ALL K candidates of the problem and all sample sizes (delta_n = 6 delta / (pi^2 K n^2)),
so any adaptively chosen audited set is covered simultaneously. NB: with a fixed s0 and a deterministic policy the
model control variate is a per-arm constant, so it shifts but does not shrink the residual variance.

Decision rule on D(a) = J(a) - J(pi_hat) for every audited challenger a:
  PASS      all UCB(D(a)) <= eps                       -> CERTIFIED pi_hat (audit compatible with the model)
  CONFLICT  some LCB(D(a)) > eps                       -> model refuted on an audited pair (MODEL_CONFLICT);
            then fallback direct BAI (LUCB rule) on the audited competitive set {pi_hat} + audited challengers,
            continuing with the same audit data: CERTIFIED leader when max_{k != leader} UCB_k - LCB_leader <= eps.
  censored  budget exhausted                            -> NEED_DATA
Guarantee is falsification-type: it covers only the audited pairs; it does not show that eta is small.
"""
from __future__ import annotations

import math

import numpy as np


def _delta_n(delta, K, n):
    return 6.0 * delta / (math.pi ** 2 * K * np.maximum(n, 1) ** 2)


def eb_radius(n, var, u_range, delta, K):
    n = np.maximum(n, 2)
    lg = np.log(4.0 / _delta_n(delta, K, n))
    return np.sqrt(2.0 * var * lg / n) + 7.0 * u_range * lg / (3.0 * (n - 1))


def agc_audit(sampler, pi_hat: int, challengers, J_model, K_union: int, H: int, eps: float, delta_audit: float,
              u_max: float, max_steps: int, init_pulls: int = 4, batch_frac: float = 0.02):
    arms = [int(pi_hat)] + [int(a) for a in challengers if int(a) != int(pi_hat)]
    A = len(arms)
    proxy = np.asarray([J_model[a] for a in arms], float)
    n = np.zeros(A); s1 = np.zeros(A); s2 = np.zeros(A)
    steps = 0
    pulls = 0
    phase = "audit"
    conflict_at = None

    def pull(i, b):
        nonlocal steps, pulls
        b = int(max(1, min(b, (max_steps - steps) // H)))
        if b <= 0:
            return
        v = np.asarray(sampler(arms[i], b), float)
        r = v - proxy[i]
        n[i] += b; s1[i] += r.sum(); s2[i] += (r * r).sum()
        steps += H * b
        pulls += b

    out = {"arms": arms, "n_audited_pairs": A - 1}
    if A == 1:
        out.update(status="CERTIFIED", pi=int(pi_hat), steps=0, pulls=0, censored=False, phase="no_challenger",
                   conflict=False)
        return out
    for i in range(A):
        pull(i, init_pulls)
    while True:
        mu = proxy + s1 / n
        var = np.where(n > 1, (s2 - s1 ** 2 / n) / np.maximum(n - 1, 1), u_max ** 2 / 4)
        rad = eb_radius(n, np.maximum(var, 0.0), u_max, delta_audit, K_union)
        lcb, ucb = mu - rad, mu + rad
        if phase == "audit":
            d_ucb = ucb[1:] - lcb[0]
            d_lcb = lcb[1:] - ucb[0]
            if (d_ucb <= eps).all():
                out.update(status="CERTIFIED", pi=int(pi_hat), steps=steps, pulls=pulls, censored=False,
                           phase="audit_pass", conflict=False, mu=mu.tolist(), n=n.tolist())
                return out
            if (d_lcb > eps).any():
                phase = "fallback"
                conflict_at = steps
                continue
            if steps + 2 * H > max_steps:
                out.update(status="NEED_DATA", pi=int(pi_hat), steps=steps, pulls=pulls, censored=True,
                           phase="audit_censored", conflict=False, mu=mu.tolist(), n=n.tolist())
                return out
            ch = 1 + int(np.argmax(d_ucb))
            b = int(batch_frac * min(n[0], n[ch]))
            pull(0, b)
            pull(ch, b)
        else:
            lead = int(np.argmax(mu))
            ucb_o = ucb.copy(); ucb_o[lead] = -np.inf
            ch = int(np.argmax(ucb_o))
            if ucb_o[ch] - lcb[lead] <= eps:
                out.update(status="CERTIFIED", pi=int(arms[lead]), steps=steps, pulls=pulls, censored=False,
                           phase="fallback_cert", conflict=True, conflict_at=conflict_at, mu=mu.tolist(), n=n.tolist())
                return out
            if steps + 2 * H > max_steps:
                out.update(status="MODEL_CONFLICT", pi=None, steps=steps, pulls=pulls, censored=True,
                           phase="fallback_censored", conflict=True, conflict_at=conflict_at, mu=mu.tolist(),
                           n=n.tolist())
                return out
            b = int(batch_frac * min(n[lead], n[ch]))
            pull(lead, b)
            pull(ch, b)
