"""B12: finite-class Track-and-Stop that observes only whole-trial utilities (learner side).

Observation model: one pull of arm k = one H-step trial of candidate policy pi_k from the problem's s0 (cost H
env steps); the learner sees only the scalar trajectory utility U in [0, u_max], E_theta[U] = J_theta(pi_k).
Evidence (anytime-valid, finite class, no union bound over theta or arms): for every grid point theta,
    M_t(theta) = prod_k sqrt(rho / (n_k + rho)) * exp( S_k(theta)^2 / (2 sigma^2 (n_k + rho)) ),
    S_k(theta) = sum_{pulls of k} (U - J_theta(pi_k)),   sigma = u_max / 2  (Hoeffding: U is sigma-sub-Gaussian),
the normal-mixture supermartingale (mixing lambda ~ N(0, 1/(sigma^2 rho))) per arm; the product over arms of these
interleaved supermartingales is a supermartingale under theta*, so by Ville
    Theta_t = {theta : log M_t(theta) < log(1/delta)}   covers theta* (in-class) w.p. >= 1 - delta for all t.
Evidence from earlier problems is carried as an additive log M prior (the product stays a supermartingale), so
B12 reuses its own ledger across a problem stream exactly like JPC (but at whole-trial granularity).
Certification: the same grid minimax-regret certifier with the same Theta^+ inflation (certify.minimax_infl).
Sampling: Track-and-Stop at theta_hat = argmin_theta log M_t(theta): allocation
    w* = argmax_w min_{theta' in Alt} sum_k w_k (J_theta_hat(k) - J_theta'(k))^2
(Alt = alive theta' under which pi_hat has inflated regret > eps), solved by Frank-Wolfe, with D-tracking,
sqrt(t) forced exploration and geometric batching (batch = max(1, batch_frac * t) pulls).
"""
from __future__ import annotations

import math

import numpy as np

from ..certify.minimax_enum import regret_matrix
from ..certify.minimax_infl import certify_minimax_infl
from ..certify.status import Status


class B12TaS:
    def __init__(self, J: np.ndarray, u_max: float, delta: float, eps: float, eta=None, rho: float = 50.0,
                 prior_logM=None, batch_frac: float = 0.05, max_alt: int = 4096, fw_iters: int = 60):
        self.J = np.asarray(J, float)                      # (B, K) normalised class J table
        self.B, self.K = self.J.shape
        self.Reg = regret_matrix(self.J)
        self.sigma = float(u_max) / 2.0
        self.delta, self.eps, self.rho = float(delta), float(eps), float(rho)
        self.eta = np.zeros(self.B) if eta is None else np.asarray(eta, float)
        self.prior = np.zeros(self.B) if prior_logM is None else np.asarray(prior_logM, float).copy()
        self.n = np.zeros(self.K)
        self.S = np.zeros(self.K)
        self.batch_frac, self.max_alt, self.fw_iters = batch_frac, max_alt, fw_iters

    # ---------------- evidence ----------------
    def logM(self) -> np.ndarray:
        out = self.prior.copy()
        s2 = self.sigma ** 2
        for k in np.flatnonzero(self.n > 0):
            Sk = self.S[k] - self.n[k] * self.J[:, k]
            out += 0.5 * math.log(self.rho / (self.n[k] + self.rho)) + Sk ** 2 / (2 * s2 * (self.n[k] + self.rho))
        return out

    def mask(self, lm=None) -> np.ndarray:
        lm = self.logM() if lm is None else lm
        return lm < math.log(1.0 / self.delta)

    def certify(self, mask=None):
        mask = self.mask() if mask is None else mask
        return certify_minimax_infl(self.Reg, mask, self.eta, self.eps)

    def update(self, k: int, utilities) -> None:
        u = np.asarray(utilities, float)
        self.n[k] += len(u)
        self.S[k] += float(u.sum())

    # ---------------- sampling ----------------
    def allocation(self, lm, mask, pi_hat: int) -> np.ndarray:
        th = int(np.argmin(lm))
        alt = np.flatnonzero(mask & (self.Reg[:, pi_hat] + 2.0 * self.eta > self.eps))
        if len(alt) == 0:
            return np.full(self.K, 1.0 / self.K)
        D = (self.J[th][None] - self.J[alt]) ** 2           # (A, K)
        if len(alt) > self.max_alt:
            keep = np.argsort(D.mean(1))[: self.max_alt]
            D = D[keep]
        w = np.full(self.K, 1.0 / self.K)
        for it in range(self.fw_iters):
            a = int(np.argmin(D @ w))
            e = np.zeros(self.K)
            e[int(np.argmax(D[a]))] = 1.0
            g = 2.0 / (it + 2.0)
            w = (1 - g) * w + g * e
        return w

    def next_pulls(self, w) -> np.ndarray:
        t = float(self.n.sum())
        b = max(1, int(self.batch_frac * t))
        forced = self.n < math.sqrt(t + b) - self.K / 2.0
        if forced.any():
            out = np.zeros(self.K, dtype=int)
            out[forced] = max(1, b // int(forced.sum()))
            return out
        target = np.maximum((t + b) * w - self.n, 0.0)
        if target.sum() <= 0:
            target = w.copy()
        out = np.floor(target / target.sum() * b).astype(int)
        while out.sum() < b:
            out[int(np.argmax(target - out))] += 1
        return out

    def run(self, sampler, H: int, max_steps: int, init_pulls: int = 1):
        """sampler(k, n) -> n utilities from real platform trials of arm k (harness side). Returns a result dict."""
        steps, it = 0, 0
        for k in range(self.K):
            if init_pulls and self.n[k] < init_pulls:
                u = sampler(k, init_pulls)
                self.update(k, u)
                steps += H * len(u)
        while True:
            lm = self.logM()
            mask = self.mask(lm)
            c = self.certify(mask)
            if c["status"] in (Status.CERTIFIED, Status.MODEL_CONFLICT):
                break
            if steps >= max_steps:
                break
            w = self.allocation(lm, mask, int(c["pi"]))
            pulls = self.next_pulls(w)
            for k in np.flatnonzero(pulls):
                u = sampler(int(k), int(pulls[k]))
                self.update(int(k), u)
                steps += H * len(u)
            it += 1
        censored = c["status"] not in (Status.CERTIFIED, Status.MODEL_CONFLICT)
        return {"status": c["status"].value, "pi": c["pi"], "steps": int(steps), "pulls": self.n.astype(int).tolist(),
                "censored": bool(censored), "set_size": int(c["set_size"]), "r_bar": float(c["r_bar"]),
                "eta_max": float(c.get("eta_max", 0.0)), "iterations": it, "logM": lm}
