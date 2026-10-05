"""MisLid-ms: JPC's parametric class + misspecification-adaptive inflation in the spirit of Reda et al. 2021
(MisLid, misspecified linear top-m identification), transported to the multi-step factor-cell setting
(methodology 1.6). Learner side: known quantities only (dynamic rule, rho_ret, eps, delta); no truth.

Online misspecification level (probability units; anytime-valid upper bound):
    eta_hat_t = max over VISITED factor cells c of  max(|lo_c - p_c(theta_hat)|, |hi_c - p_c(theta_hat)|)
  = |empirical frequency - best class fit| + CS radius, where [lo_c, hi_c] is the per-cell betting CS of FCC (same
  delta_cell = delta / #cells) and theta_hat is the class MLE (best fit). On the event that every cell CS covers,
  eta_hat_t >= max_c |p*_c - p_c(theta_hat)|.
Confidence-set inflation (linear in eta_hat to first order): a model theta is kept iff
    log q_{1:t} - log p_theta(x_{1:t}) - sum_c n_c(t) * log(1 + eta_hat_t / m_c(theta)) < log(1/delta),
  m_c(theta) = max(min(p_c(theta), 1 - p_c(theta)), M_FLOOR), n_c(t) = #updates of cell c inside the evidence set.
  (Per Bernoulli update |log p*(x) - log p_theta(x)| <= log(1 + eta / m) when |p* - p_theta| <= eta, so a model within
  eta of the truth on every cell stays in the set with probability >= 1 - delta - delta_cells.)
Inflation multiple kappa (infl_mult, the single hyper-parameter, frozen on the dev block over {1, 2}; default 1 = the
  setup version): eta_hat is replaced by kappa * eta_hat in both the set inflation and the certificate threshold.
Variants: eta_mode 'ucb' (default, the valid upper bound above) and 'lcb' (eta_hat = max_c dist(p_c(theta_hat),
  [lo_c, hi_c]), the smallest misspecification consistent with the data; NOT valid, a less conservative variant).
Certificate threshold: eps - 2 H eta_hat_t^val with eta_hat^val = c_q (2 max_t w_t + 4 w_ret / H) eta_hat_t, the
  per-round reward-scale conversion of a per-cell probability error (2 pair cells per round, 4 retention cells at the
  horizon); 2 H eta^val = 2 c_q (2 H max_t w_t + 4 w_ret) eta_hat. If the threshold is <= 0 the method cannot certify
  and keeps acquiring with the r3 DDA rule until T_max.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..acquire.nl_kl_dda import dda_choose
from ..certify.fcc import BettingCS, CellIndex
from ..certify.minimax_enum import certify_minimax
from ..evidence.lr_set import SeqLRSet
from .ms_common import CHECKPOINTS, JPC

M_FLOOR = 0.02
ETA_GRID = 1e-4
ETA_MODES = ("ucb", "lcb")
CHECKPOINT_SET = set(int(x) for x in CHECKPOINTS)


def class_cell_probs(np_params: dict, cells: CellIndex, c: float) -> np.ndarray:
    """(B, n_cells) cell success probabilities of every class model (vectorised fcc.cell_probs)."""
    al, be, ga = np_params["alpha"], np_params["beta"], np_params["gamma"]
    tau = np.concatenate([np_params["tauL"], np_params["tauR"]], 1)
    psl, lam = np_params["psi_left"], np_params["lam"]
    B = al.shape[0]
    out = np.zeros((B, cells.n))
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    for i in range(cells.L):
        for j in range(cells.R):
            for n in range(cells.N):
                for b in range(cells.nb):
                    out[:, cells.pair(i, j, n, b)] = sig(al[:, i] + be[:, j] - ga[:, i] * n + psl[:, i, b])
    for p in range(cells.P):
        for n in range(cells.N):
            for y in range(2):
                out[:, cells.ret(p, n, y)] = sig(tau[:, p] + c * y - lam * n)
    return out


class InflatedLRSet(SeqLRSet):
    """SeqLRSet whose statistic is deflated by sum_c n_c log(1 + eta / m_c(theta)); eta is pushed by the method."""

    def __init__(self, propagator, LT, delta, cells: CellIndex, logm: np.ndarray):
        super().__init__(propagator, LT, delta)
        self.cells = cells
        self.n_c = np.zeros(cells.n)
        self.inv_m = torch.as_tensor(1.0 / logm, dtype=LT.dtype)          # (B, n_cells) 1 / m_c(theta)
        self.eta = 0.0
        self._eta_c, self._L = None, None

    def update(self, obs) -> None:
        super().update(obs)
        for c, _ in self.cells.obs_updates(obs):
            self.n_c[c] += 1

    def inflation(self) -> torch.Tensor:
        if self.eta <= 0:
            return torch.zeros(self.B, dtype=self.cum.dtype)
        e = math.ceil(self.eta / ETA_GRID) * ETA_GRID          # rounded UP (conservative) to cache the log table
        if e != self._eta_c:
            self._eta_c, self._L = e, torch.log1p(e * self.inv_m)
        nc = torch.as_tensor(self.n_c, dtype=self.cum.dtype)
        return self._L @ nc

    def statistic(self) -> torch.Tensor:
        return self.log_ratio() - self.inflation()


class MisLidMS(JPC):
    name = "MisLid-ms"

    def __init__(self, pub, learner_class: str | None = None, eta_mode: str = "ucb", infl_mult: float = 1.0):
        super().__init__(pub)
        if eta_mode not in ETA_MODES:
            raise ValueError(eta_mode)
        if infl_mult < 1.0:
            raise ValueError("infl_mult must be >= 1")
        self.eta_mode = eta_mode
        self.infl_mult = float(infl_mult)          # the single MisLid-ms hyper-parameter (inflation multiple, {1, 2})
        self.name = "MisLid-ms" if eta_mode == "ucb" else f"MisLid-ms-{eta_mode}"
        if self.infl_mult != 1.0:
            self.name += f"-k{infl_mult:g}"
        if learner_class is not None:
            self.learner_class = learner_class
        nl = self.nlpub()
        P = pub.aspace
        self.cells = CellIndex(P.L, P.R, pub.nmax, P.nb)
        self.CP = class_cell_probs(nl.ncl.np_params, self.cells, pub.c_known)
        self.m = np.maximum(np.minimum(self.CP, 1.0 - self.CP), M_FLOOR)
        self.cs = BettingCS(self.cells.n, pub.delta / self.cells.n)
        self.visits = np.zeros(self.cells.n, dtype=np.int64)
        self.eta_trace = []

    def make_set_factory(self):
        nl = self.nlpub()
        return lambda: InflatedLRSet(nl.prop, nl.LT, nl.delta, self.cells, self.m)

    def observe(self, obs):
        super().observe(obs)
        ups = self.cells.obs_updates(obs)
        self.cs.update_many(ups)
        for c, _ in ups:
            self.visits[c] += 1

    def eta_hat(self, k_hat: int) -> float:
        v = self.visits > 0
        if not v.any():
            return 0.0
        lo, hi = self.cs.intervals()
        p = self.CP[k_hat]
        if self.eta_mode == "ucb":            # sup distance from the model to the CS: deviation + radius (valid)
            return float(np.max(np.maximum(np.abs(lo[v] - p[v]), np.abs(hi[v] - p[v]))))
        # lcb: distance from the model to the CS interval (0 if inside) -- the smallest misspecification consistent with
        # the data; NOT a valid upper bound (variant reported separately)
        return float(np.max(np.maximum(np.maximum(lo[v] - p[v], p[v] - hi[v]), 0.0)))

    def solve(self, k, sw, lr, handle, rng, tmax):
        t0 = time.perf_counter()
        nl = self.nlpub()
        q = self.pub.problems[k]
        Reg = nl.Reg[k]
        eps, top_m = self.pub.eps, nl.top_m
        conv = 2.0 * float(q.utility.c_q) * (2.0 * q.H * float(np.max(q.utility.w)) + 4.0 * float(q.utility.w_ret))
        inner = lr.inner
        status, cert, eta, thr, first, n_checks = None, None, 0.0, eps, None, 0
        thr_max, eta_min = -math.inf, math.inf
        while True:                                       # checked every step, like the r3 JPC it extends
            n_checks += 1
            inner.eta = 0.0
            kh = inner.mle()
            eta = self.infl_mult * self.eta_hat(kh)
            inner.eta = eta
            thr = eps - conv * eta
            thr_max, eta_min = max(thr_max, thr), min(eta_min, eta)
            mask = lr.mask().numpy()
            if not mask.any():
                status = "MODEL_CONFLICT"
                break
            cert = certify_minimax(Reg, mask, thr if thr > 0 else eps, top_m)
            if thr > 0 and cert["status"].value == "CERTIFIED":
                status, first = "CERTIFIED", sw.new_steps
                break
            if sw.new_steps in CHECKPOINT_SET:
                self.eta_trace.append((k, sw.new_steps, eta, thr))
            if sw.new_steps >= tmax:
                status = "NEED_DATA"
                break
            blk = cert["blocking"] or [int(i) for i in np.flatnonzero(mask)[:top_m]]
            margins = np.maximum(nl.log_thr - lr.statistic().numpy()[blk], 1e-6)
            code = nl.prop.codec.encode(*handle.observable_state())
            a, _ = dda_choose(code, nl.py[kh:kh + 1], nl.pe[kh:kh + 1], nl.LT_np[kh], nl.py[blk], nl.pe[blk],
                              margins, nl.inc, nl.prop, nl.legal, rng)
            self.step(sw, handle, a)
        pi = cert["pi"] if status == "CERTIFIED" else None
        return {"status": status, "pi": pi, "steps": sw.new_steps,
                "extra": {"eta_hat_end": eta, "infl_mult": self.infl_mult, "eps_cert_end": thr,
                          "eps_cert_max": thr_max, "eta_hat_min": eta_min, "conv_2H": conv, "n_checks": n_checks,
                          "first_cert_step": first, "set_size": int(lr.mask().numpy().sum()),
                          "wall_clock_s": time.perf_counter() - t0}}
