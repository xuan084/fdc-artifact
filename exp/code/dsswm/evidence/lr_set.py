"""Anytime-valid sequential likelihood-ratio confidence set over a finite class (universal inference).

M_t(theta) = prod_s q_s(x_s) / p_theta(x_s), with q_s predictable (built from x_1..x_{s-1} only).
Under theta*, M_t(theta*) is a non-negative martingale, so by Ville P(exists t: M_t(theta*) >= 1/delta) <= delta.
Theta_t = {theta : log M_t(theta) < log(1/delta)}.
numerator = 'plugin'  : q_s = p_{theta_hat_{s-1}} (MLE on the class; first step uses the uniform mixture)
numerator = 'mixture' : q_s = Bayes predictive under the uniform prior (also predictable)

threshold (round 1, HR2 accounting):
  'ui'        (default) Theta_t = {theta : log q_{1:t} - log p_theta < log(1/delta)}  (anytime-valid, Ville)
  'chi2_deff' fixed-n Wilks set on the same finite class:
              Theta_t = {theta : max_theta' log p_theta' - log p_theta < 0.5 * chi2_{d_eff, 1-delta}}
              NOT anytime-valid; it isolates the threshold factor (UI vs chi^2) at fixed support and shape.
              d_eff is the design-Fisher participation ratio tr(I)^2 / tr(I^2), locked in r2_prereg_lock.
"""
from __future__ import annotations

import math

import torch
from scipy.stats import chi2

from ..core.provenance import require_authentic


class SeqLRSet:
    def __init__(self, propagator, LT: torch.Tensor, delta: float, numerator: str = "plugin",
                 threshold: str = "ui", d_eff: float | None = None):
        self.prop = propagator
        self.LT = LT
        self.B = LT.shape[0]
        self.delta = delta
        self.numerator = numerator
        self.cum = torch.zeros(self.B, device=LT.device, dtype=LT.dtype)
        self.log_num = 0.0
        self.n_rounds = 0
        self.serials: list[int] = []
        if threshold not in ("ui", "chi2_deff"):
            raise ValueError(threshold)
        if threshold == "chi2_deff" and not d_eff:
            raise ValueError("threshold='chi2_deff' needs d_eff > 0")
        self.threshold = threshold
        self.d_eff = None if d_eff is None else float(d_eff)

    def _predictive_log(self, ll: torch.Tensor) -> float:
        if self.n_rounds == 0 or self.numerator == "mixture":
            # Bayes predictive: logsumexp(cum + ll) - logsumexp(cum)   (uniform prior)
            return float(torch.logsumexp(self.cum + ll, 0) - torch.logsumexp(self.cum, 0))
        k = int(torch.argmax(self.cum))
        return float(ll[k])

    def update(self, obs) -> None:
        require_authentic(obs)
        ll = self.prop.loglik(self.LT, [obs])[:, 0]
        self.log_num += self._predictive_log(ll)
        self.cum = self.cum + ll
        self.n_rounds += 1
        self.serials.append(obs.serial)

    def log_ratio(self) -> torch.Tensor:
        return self.log_num - self.cum

    def cutoff(self) -> float:
        if self.threshold == "ui":
            return math.log(1.0 / self.delta)
        return 0.5 * float(chi2.ppf(1.0 - self.delta, self.d_eff))

    def statistic(self) -> torch.Tensor:
        if self.threshold == "ui":
            return self.log_ratio()
        return self.cum.max() - self.cum

    def mask(self) -> torch.Tensor:
        return self.statistic() < self.cutoff()

    def mle(self) -> int:
        return int(torch.argmax(self.cum))

    def size(self) -> int:
        return int(self.mask().sum())

    def state_digest(self):
        return (self.cum.detach().cpu().numpy().copy(), self.log_num, tuple(self.serials))


def participation_ratio(I) -> float:
    """d_eff = tr(I)^2 / tr(I^2) of a (design) Fisher matrix; used to set the chi2_deff threshold."""
    import numpy as np
    I = np.asarray(I, float)
    t2 = float(np.trace(I @ I))
    return float(np.trace(I) ** 2 / t2) if t2 > 0 else 0.0
