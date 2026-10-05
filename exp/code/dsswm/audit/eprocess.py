"""Stepwise e-process falsification audit (HF4).

Model-environment consistency at tolerance eta, tested with per-STEP Bellman residuals instead of whole trials:
for an audited policy pi and the model point theta_m (e.g. the class MLE inside Theta_t) let V^m_t(s) be the exact
model value-to-go of pi (backward DP over the whole observable state space, so a segment can start from whatever
state the platform is in -- no reset / steering to s0 needed). A real step from s_t under a_t = pi(t, s_t) gives
    r_t = u_t + V^m_{t+1}(s_{t+1}) - V^m_t(s_t),   u_t = c_q w_t sum_{active pairs} y,   V^m_H(s) = c_q w_ret sum_p e_p
Under the model E[r_t | F_{t-1}] = 0 and sum_t r_t telescopes to U - V^m_0(s_0).
Null H0(eta): |E_true[r_t | F_{t-1}]| <= eta / H at every audited step (sufficient for |J_true - J_m| <= eta for the
audited policies from any start state). Two one-sided betting e-processes
    E^+_n = prod (1 + lam_s (r_s - eta/H)),   E^-_n = prod (1 + lam_s (-r_s - eta/H)),
predictable lam_s in [0, 1/(2 (hi - m0))] (aGRAPA, clipped), are non-negative supermartingales under H0, and so is
their average; MODEL_CONFLICT when (E^+ + E^-)/2 >= 1/delta_audit (Ville). A fixed step budget without an alarm is
reported as PASS -- a falsification-type audit (no positive guarantee that eta is small).
"""
from __future__ import annotations

import math

import numpy as np


class BettingEProcess:
    """One-sided test of H0: E[X_s | past] <= m0 for X in [lo, hi]."""

    def __init__(self, m0: float, lo: float, hi: float, c: float = 0.5):
        assert hi > m0, "tolerance must lie below the upper range"
        self.m0, self.lo, self.hi = float(m0), float(lo), float(hi)
        self.lam_max = c / max(hi - m0, 1e-12)
        self.log_e = 0.0
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0

    def _lam(self):
        if self.n < 2:
            return 0.0
        mu = self.s1 / self.n - self.m0
        var = max(self.s2 / self.n - (self.s1 / self.n) ** 2, 1e-12)
        lam = mu / (var + mu * mu)          # aGRAPA
        return float(min(max(lam, 0.0), self.lam_max))

    def update(self, x: float) -> float:
        lam = self._lam()
        self.log_e += math.log(max(1.0 + lam * (x - self.m0), 1e-300))
        self.n += 1
        self.s1 += x
        self.s2 += x * x
        return self.log_e


class TwoSidedEProcess:
    def __init__(self, tol: float, lo: float, hi: float):
        self.up = BettingEProcess(tol, lo, hi)
        self.dn = BettingEProcess(tol, -hi, -lo)

    def update(self, x: float) -> float:
        a, b = self.up.update(x), self.dn.update(-x)
        m = max(a, b)
        return m + math.log(0.5 * (math.exp(a - m) + math.exp(b - m)))


def policy_value_tables(prop, LT_row: np.ndarray, EY_row: np.ndarray, policy, H: int, utility):
    """Exact model value-to-go V[t, code] (t = 0..H) of a deterministic policy on every observable state."""
    S = prop.codec.size
    V = np.zeros((H + 1, S))
    for code in range(S):
        V[H, code] = utility.c_q * utility.w_ret * prop.codec.decode(code)[1].sum()
    for t in range(H - 1, -1, -1):
        for code in range(S):
            ld, en = prop.codec.decode(code)
            a = policy.act(t, ld, en)
            fidx, const, nxt, eyidx = prop._struct(code, a)
            p = np.exp(LT_row[fidx].sum(1) + const)
            p = p / p.sum()
            V[t, code] = utility.c_q * utility.w[t] * EY_row[eyidx].sum() + float(p @ V[t + 1, nxt])
    return V


class EProcessAudit:
    """Audit the model on a set of policies (e.g. pi_hat and its binding challengers) with real steps."""

    def __init__(self, prop, LT_row, EY_row, policies, H, utility, eta: float, delta_audit: float):
        self.prop, self.H, self.utility = prop, H, utility
        self.policies = list(policies)
        self.V = [policy_value_tables(prop, LT_row, EY_row, pi, H, utility) for pi in self.policies]
        umax = utility.c_q * float(np.max(utility.w)) * min(prop.L, prop.R)
        lo = min(float(V[1:].min() - V[:-1].max()) for V in self.V)
        hi = max(float(umax + V[1:].max() - V[:-1].min()) for V in self.V)
        self.lo, self.hi = lo, hi
        self.tol = eta / H
        self.ep = TwoSidedEProcess(self.tol, lo, hi)
        self.threshold = math.log(1.0 / delta_audit)
        self.steps = 0
        self.log_e_max = 0.0

    def run(self, env_handle, max_steps: int, record=None) -> dict:
        """Round-robin H-step segments of the audited policies from the current platform state."""
        k = 0
        while self.steps < max_steps:
            pi, V = self.policies[k % len(self.policies)], self.V[k % len(self.policies)]
            for t in range(self.H):
                if self.steps >= max_steps:
                    break
                ld, en = env_handle.observable_state()
                c0 = self.prop.codec.encode(ld, en)
                obs = env_handle.step(pi.act(t, ld, en))
                if record is not None:
                    record(obs)
                u = self.utility.c_q * self.utility.w[t] * sum(o[3] for o in obs.outcomes if o[4])
                c1 = self.prop.codec.encode(obs.next_loads, obs.next_engaged)
                r = u + V[t + 1, c1] - V[t, c0]
                le = self.ep.update(float(r))
                self.steps += 1
                self.log_e_max = max(self.log_e_max, le)
                if le >= self.threshold:
                    return {"status": "MODEL_CONFLICT", "steps": self.steps, "log_e": le}
            k += 1
        return {"status": "PASS", "steps": self.steps, "log_e": self.log_e_max}
