"""Replica ground-truth environments and exact J (methodology.md sec. 2.1, 2.2, 3).

Conventions decided from the text (spec-gap decisions are listed in SPEC_GAPS and in summary.json):
  * participants are indexed p = 0..L-1 (left) then L..L+R-1 (right); a pair (i, j) uses left i, right j.
  * a policy is an opaque callable policy(t, loads, engaged) -> (pairs, incentives) where pairs is a
    tuple of (i, j) and incentives a dict {(i, j): level}; level 0 = no incentive.
  * load rule (sec. 2.1, A1): matched participants +1 capped at nmax, unmatched -1 floored at 0.
No import of the main implementation is allowed here.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

SPEC_GAPS = {
    "lin_psi2": "Text gives theta in R^10 (single psi). Locked prereg adds psi2 for incentive level 2; "
                "replica uses theta = (alpha_1..L, beta_1..R, gamma_1..L, psi_1, psi_2).",
    "nl_y_when_disengaged": "Text does not say whether a matched pair with a disengaged member produces an "
                            "outcome. Variant 'both_engaged' (y=0 unless both engaged) vs 'always'.",
    "nl_retention_load_time": "Retention uses n_i(t) (pre-update load, text-literal) vs n_i(t+1).",
    "nl_retention_term": "Text sec. 2.1 utility has no retention term; problem data carry w_ret. "
                         "Replica uses U = c_q*(sum_t w_t sum_pairs y + w_ret * sum_p e_p(H)).",
    "nl_right_side": "Right participants have no gamma; their retention is sigmoid(tau_j + c*y_j - lam*n_j).",
    "nl_load_when_disengaged": "Load rule applied to every participant purely by the action (engagement-independent).",
}

# Round-1 status (methodology.md sec. 2.1, completed E1-NL spec, written into the paper appendix).
SPEC_GAPS_R2_STATUS = {
    "nl_y_when_disengaged": "RESOLVED by sec. 2.1 item 1: y_ij = 0 unless both matched members are engaged "
                            "(default y_mode='both_engaged').",
    "nl_retention_load_time": "RESOLVED by sec. 2.1 item 2: retention uses n_p(t), the load BEFORE this round's "
                              "update (default ret_load='pre').",
    "nl_retention_term": "RESOLVED by sec. 2.1 item 3: U_q = c_q [sum_t w_t sum y + w_ret sum_p e_p(H)]; c_q is "
                         "public data (class-max normalisation), taken from the problem as data.",
    "nl_right_side": "STILL IMPLICIT in the text (right side has no gamma; same retention form); replica choice kept.",
    "nl_load_when_disengaged": "STILL IMPLICIT in the text (sec. 2.1 A1 load rule is stated per action only); "
                               "replica choice kept (engagement-independent).",
    "nl_disengaged_return": "Disengaged participants re-engage with probability rho_ret (public constant), "
                            "independent of y and load; replica choice kept.",
    "m4_drift": "sec. 2.4: alpha_i(t) = alpha_i + s * t_global / T_ref. Replica implements both the text-literal "
                "per-round drift (t_global = t_start + t) and the frozen-at-start approximation.",
    "lin_psi2": "unchanged (E1-Lin not re-tested in round 2)",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def next_loads(loads, pairs, L, R, nmax):
    matched = [False] * (L + R)
    for i, j in pairs:
        matched[i] = True
        matched[L + j] = True
    return tuple(min(n + 1, nmax) if m else max(n - 1, 0) for n, m in zip(loads, matched))


# ------------------------------------------------------------------ E1-Lin
class ReplicaLin:
    """y_ij,t = alpha_i + beta_j - gamma_i n_i(t) + psi_b + N(0, sigma^2); loads independent of y."""

    def __init__(self, theta, L=3, R=3, nmax=3, sigma=1.5, seed=0):
        th = np.asarray(theta, float)
        self.L, self.R, self.nmax, self.sigma = L, R, nmax, sigma
        self.alpha = th[:L]
        self.beta = th[L:L + R]
        self.gamma = th[L + R:2 * L + R]
        self.psi = np.concatenate([[0.0], th[2 * L + R:]])  # psi[0] = 0 (no incentive)
        self.rng = np.random.default_rng(seed)

    def mean(self, i, j, n_i, b):
        return self.alpha[i] + self.beta[j] - self.gamma[i] * n_i + self.psi[b]

    def exact_J(self, policy, loads0, engaged0, H, w, w_ret, c_q):
        loads = tuple(int(x) for x in loads0)
        eng = tuple(int(x) for x in engaged0)
        tot = 0.0
        for t in range(H):
            pairs, inc = policy(t, loads, eng)
            s = 0.0
            for (i, j) in pairs:
                s += self.mean(i, j, loads[i], inc.get((i, j), 0))
            tot += w[t] * s
            loads = next_loads(loads, pairs, self.L, self.R, self.nmax)
        # no engagement dynamics in E1-Lin: sum_p e_p(H) is constant = sum(engaged0)
        return c_q * (tot + w_ret * float(sum(eng)))

    def rollout(self, policy, loads0, engaged0, H, w, w_ret, c_q):
        loads = tuple(int(x) for x in loads0)
        eng = tuple(int(x) for x in engaged0)
        tot = 0.0
        for t in range(H):
            pairs, inc = policy(t, loads, eng)
            for (i, j) in pairs:
                tot += w[t] * (self.mean(i, j, loads[i], inc.get((i, j), 0)) + self.sigma * self.rng.standard_normal())
            loads = next_loads(loads, pairs, self.L, self.R, self.nmax)
        return c_q * (tot + w_ret * float(sum(eng)))


# ------------------------------------------------------------------ E1-NL
class ReplicaNL:
    """E1-NL: y ~ Bern(sigmoid(alpha_i + beta_j - gamma_i n_i + psi_b)); engagement transitions."""

    def __init__(self, alpha, beta, gamma, tauL, tauR, psi, lam, c=1.0, rho_ret=0.3, L=2, R=2, nmax=2,
                 y_mode="both_engaged", ret_load="pre", seed=0, syn=None, psi_left=None):
        """syn: optional (L, R) pair-synergy logit term (round-3 m1r held-out family, replica-side).
        psi_left: optional (L, n_levels) per-left-participant incentive effect, psi_left[i, 0] = 0 (round-3 m3)."""
        self.L, self.R, self.nmax = L, R, nmax
        self.alpha = np.asarray(alpha, float)
        self.beta = np.asarray(beta, float)
        self.gamma = np.asarray(gamma, float)
        self.tau = np.concatenate([np.asarray(tauL, float), np.asarray(tauR, float)])
        psi = np.atleast_1d(np.asarray(psi, float))
        self.psi = np.concatenate([[0.0], psi])  # psi[b], b in {0, 1, ...}
        self.lam, self.c, self.rho_ret = float(lam), float(c), float(rho_ret)
        self.y_mode, self.ret_load = y_mode, ret_load
        self.syn = None if syn is None else np.asarray(syn, float).reshape(L, R)
        self.psi_left = None if psi_left is None else np.asarray(psi_left, float).reshape(L, -1)
        self.rng = np.random.default_rng(seed)

    def p_y(self, i, j, n_i, b, da=0.0):
        """da: additive alpha shift at this round (m4 drift, sec. 2.4); 0 for in-class / off-grid truths."""
        ps = self.psi[b] if self.psi_left is None else self.psi_left[i, b]
        sy = 0.0 if self.syn is None else self.syn[i, j]
        return sigmoid(self.alpha[i] + da + self.beta[j] - self.gamma[i] * n_i + ps + sy)

    def _active(self, i, j, eng):
        if self.y_mode == "always":
            return True
        return bool(eng[i]) and bool(eng[self.L + j])

    def _q_stay(self, p, e, y, n_pre, n_post):
        if not e:
            return self.rho_ret
        n = n_pre if self.ret_load == "pre" else n_post
        return sigmoid(self.tau[p] + self.c * y - self.lam * n)

    def exact_J(self, policy, loads0, engaged0, H, w, w_ret, c_q, alpha_shift=None):
        """alpha_shift: None, a constant, or a callable t -> shift (per-round m4 drift)."""
        P = self.L + self.R
        if alpha_shift is None:
            shift = lambda t: 0.0  # noqa: E731
        elif callable(alpha_shift):
            shift = alpha_shift
        else:
            shift = lambda t, _c=float(alpha_shift): _c  # noqa: E731
        dist = {(tuple(int(x) for x in loads0), tuple(int(x) for x in engaged0)): 1.0}
        reward = 0.0
        for t in range(H):
            new = {}
            da = float(shift(t))
            for (loads, eng), pr in dist.items():
                pairs, inc = policy(t, loads, eng)
                nl = next_loads(loads, pairs, self.L, self.R, self.nmax)
                act = [(i, j, self.p_y(i, j, loads[i], inc.get((i, j), 0), da))
                       for (i, j) in pairs if self._active(i, j, eng)]
                reward += pr * w[t] * sum(q for _, _, q in act)
                for ys in itertools.product((0, 1), repeat=len(act)):
                    py = 1.0
                    yp = [0] * P
                    for (i, j, q), y in zip(act, ys):
                        py *= q if y else 1.0 - q
                        yp[i] = y
                        yp[self.L + j] = y
                    if py == 0.0:
                        continue
                    qs = [self._q_stay(p, eng[p], yp[p], loads[p], nl[p]) for p in range(P)]
                    for es in itertools.product((0, 1), repeat=P):
                        pe = py
                        for q, e in zip(qs, es):
                            pe *= q if e else 1.0 - q
                        if pe == 0.0:
                            continue
                        key = (nl, es)
                        new[key] = new.get(key, 0.0) + pr * pe
            dist = new
        e_final = sum(pr * sum(es) for (_, es), pr in dist.items())
        return c_q * (reward + w_ret * e_final)

    def step(self, loads, eng, pairs, inc):
        """One sampled platform round. Returns (outcomes, next_loads, next_engaged)."""
        P = self.L + self.R
        nl = next_loads(loads, pairs, self.L, self.R, self.nmax)
        yp = [0] * P
        outs = []
        for (i, j) in pairs:
            b = inc.get((i, j), 0)
            if self._active(i, j, eng):
                y = int(self.rng.random() < self.p_y(i, j, loads[i], b))
                yp[i] = y
                yp[self.L + j] = y
                outs.append((i, j, b, y))
        ne = tuple(int(self.rng.random() < self._q_stay(p, eng[p], yp[p], loads[p], nl[p])) for p in range(P))
        return outs, nl, ne

    def rollout(self, policy, loads0, engaged0, H, w, w_ret, c_q):
        loads = tuple(int(x) for x in loads0)
        eng = tuple(int(x) for x in engaged0)
        tot = 0.0
        for t in range(H):
            pairs, inc = policy(t, loads, eng)
            outs, loads, eng = self.step(loads, eng, pairs, inc)
            tot += w[t] * sum(o[3] for o in outs)
        return c_q * (tot + w_ret * float(sum(eng)))


# ------------------------------------------------------------------ round-3 held-out family m1r (replica side)
def replica_synergy_pattern(seed, L=2, R=2):
    """m1r pair-synergy pattern generated on the replica side (rng [seed, 9071], independent of the main package's
    m1 pattern rng [seed, 61]): double-centred (zero row / column means), max-abs normalised to 1."""
    s = np.random.default_rng([int(seed), 9071]).standard_normal((L, R))
    s = s - s.mean(1, keepdims=True)
    s = s - s.mean(0, keepdims=True)
    return s / max(float(np.abs(s).max()), 1e-12)
