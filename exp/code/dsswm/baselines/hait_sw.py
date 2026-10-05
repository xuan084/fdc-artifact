"""Hait-SW (round 5): stratum-resolved anytime-valid architecture of Hait (arXiv 2609.37873) ported to the
budgeted-frontier, without-replacement pool replay.

Source passages (verified with the arXiv MCP, 2026-10-03; quoted in plan/baseline_qualification_r5.md):
  * s4.2 eq. (4.4): target half-width W(n) = sum_j q_j b_j(n_j) from simultaneous local CSs (additive aggregation);
  * Prop. 4.2 / eqs. (4.8), (4.11): width-optimal (boundary-rate) allocation for root-n variance-adaptive local CSs,
    n_j proportional to (q_j sigma_j)^{2/3} -- "Neither rule is the Neyman allocation q_j sigma_j";
  * Lemma 6.1 + Thm 6.2: the aggregated interval is time-uniformly valid "under any predictable adaptive allocation
    rule" when every local CS has time-uniform error <= alpha / J;
  * s7, Cor. 7.1: familywise ("ever") best-system identification from the simultaneous local event.

Port (each change is listed in the qualification table):
  1. strata = cells (s, a). Segments arrive exogenously (Criteo log order), so the method only chooses the arm
     inside an arriving segment. Both arms of segment s enter the rectangle width of every challenger that differs
     in s with the same weight w_s, so the within-segment version of (4.11) is p_{s,a} proportional to
     sigma_hat_{s,a}^{gamma} with gamma = 2/3 (Hait's law; DEFAULT). gamma = 1 (Neyman, as written in methodology
     s4.2) is available as a tuning option; it is NOT Hait's width-optimal rule for this architecture.
  2. sigma_hat: predictable Laplace-smoothed plug-in, m = (sum + 1) / (n + 2), sigma_hat = R sqrt(m (1 - m)),
     computed from the data observed BEFORE THE PREVIOUS re-plan batch (one-batch lag, methodology s4.2 wording);
     the allocation is therefore F_{t-1}-measurable (predictable) -- in fact validity below does not even need that.
  3. exploration floor p_min (Hait's vanishing floor eps_t = min{1, 2 t^{-0.6}} is replaced by a constant floor,
     tuned on {0.02, 0.05, 0.10}): p = p_min + (1 - A p_min) * p_raw.
  4. shares are realised by C-tracking at the re-plan batch boundaries (the harness gives each segment's batch
     arrivals to the first non-exhausted arm in the returned preference order): target arm = largest deficit
     T_{s,a} + p_{s,a} E[batch arrivals] - n_{s,a}; re-selection on exhaustion = descending deficit.
  5. local CSs: PrPl-EB (with-replacement martingale) -> WSR20 WoR empirical-Bernstein CS (Waudby-Smith & Ramdas
     2020, Thm 4 -- the without-replacement version of the same predictable-plug-in betting construction), per cell
     at delta / (S A) two-sided; the union over the S A cells replaces Hait's alpha / (H J) split (tighter here).
  6. decision = rect_certificate: for each problem q, U_q = max_{pi' in Pi_{B_q}} sum_{s: pi'(s) != pi_hat(s)}
     w_s (hi[s, pi'(s)] - lo[s, pi_hat(s)]), which is the s7 pairwise rule (7.1) with the eps-relaxation, applied to
     every feasible challenger simultaneously (all pairs share the same cell event, so no H factor is needed).

Validity: on the cell event {mu_c in CS_c(n) for all c, n} (prob >= 1 - delta; each cell's draws are a uniformly
random permutation of its pool whatever the predictable arm choice, so WSR20 Thm 4 applies in the cell's local time
-- the WoR analogue of Hait Lemma 6.1), every certification is eps-correct at every checkpoint. Rigorous.
"""
from __future__ import annotations

import numpy as np

from .frontier_common import Method, rect_certificate

__all__ = ["HaitSW", "HAIT_PMIN_GRID", "HAIT_GAMMA_GRID", "hait_shares"]

HAIT_PMIN_GRID = (0.02, 0.05, 0.10)
HAIT_GAMMA_GRID = (2.0 / 3.0, 1.0)


def hait_shares(n, sums, R=1.0, gamma=2.0 / 3.0, p_min=0.05):
    """Within-segment allocation shares p (S, A) from (lagged) counts and sums."""
    n = np.asarray(n, dtype=float)
    m = (np.asarray(sums, dtype=float) / R + 1.0) / (n + 2.0)
    sig = R * np.sqrt(np.clip(m * (1.0 - m), 0.0, None))
    raw = sig ** gamma
    tot = raw.sum(1, keepdims=True)
    A = n.shape[1]
    raw = np.where(tot > 0, raw / np.where(tot > 0, tot, 1.0), 1.0 / A)
    if not 0.0 <= p_min * A <= 1.0:
        raise ValueError("p_min must satisfy 0 <= A p_min <= 1")
    return p_min + (1.0 - A * p_min) * raw


class HaitSW(Method):
    alloc_kind = "adaptive"
    validity = "rigorous"

    def __init__(self, p_min=0.05, gamma=2.0 / 3.0, lag=1, name="Hait-SW"):
        self.p_min = float(p_min)
        self.gamma = float(gamma)
        self.lag = int(lag)
        self.name = name

    def setup(self, ctx):
        self._T = np.zeros((ctx.S, ctx.A))
        self._p = np.full((ctx.S, ctx.A), 1.0 / ctx.A)
        self._last_ns = np.zeros(ctx.S)
        self._hist = []          # snapshots (n, sum) at past plan calls, oldest first
        self._shares_log = []

    def plan(self, ctx, st):
        n = st.n.astype(float)
        ns = n.sum(1)
        self._T += self._p * (ns - self._last_ns)[:, None]        # credit last batch with last targets
        self._last_ns = ns.copy()
        self._hist.append((st.n.copy(), st.sum.copy()))
        if len(self._hist) > self.lag + 1:
            self._hist.pop(0)
        n_lag, s_lag = self._hist[0] if len(self._hist) > self.lag else (np.zeros_like(st.n), np.zeros_like(st.sum))
        self._p = hait_shares(n_lag, s_lag, ctx.R, self.gamma, self.p_min)
        if len(self._shares_log) < 50 or len(self._shares_log) % 200 == 0:
            self._shares_log.append(self._p[:, 0].round(4).tolist())
        deficit = self._T + self._p * (ctx.w * ctx.replan)[:, None] - n
        return np.argsort(-deficit, axis=1, kind="stable").astype(np.int64)

    def certify(self, ctx, st):
        return rect_certificate(ctx, st)

    def describe(self):
        d = super().describe()
        d.update({"p_min": self.p_min, "gamma": self.gamma, "lag_batches": self.lag,
                  "allocation": "p_{s,a} ~ sigma_hat^gamma (Laplace plug-in, lagged), floor p_min, C-tracking",
                  "certificate": "rect (WSR20 WoR, delta/(S*A) per cell)"})
        return d
