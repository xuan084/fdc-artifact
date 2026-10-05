"""Molitor-WoR (round 5): anytime-valid optimal-policy identification of Molitor (arXiv 2606.17515) ported to the
without-replacement pool replay.

Source passages (verified with the arXiv MCP, 2026-10-03; quoted in plan/baseline_qualification_r5.md):
  * Prop. 1 (Waudby-Smith et al. 2024): per-policy anytime-valid lower / upper CS for nu(pi) from logged bandit data
    (with-replacement / i.i.d. contextual-bandit stream, rewards in [0, 1]);
  * Theorem 1: with per-policy CSs at level alpha / m (union over Pi), the candidate set
        S_t = {pi : U_t(pi) >= max_{pi'} L_t(pi')}
    contains the optimal set uniformly over time w.p. >= 1 - alpha; s3.1 notes the eps-optimal extension.

Port:
  1. data = the RCT log read in pool-proportional random order (the logging policy, alloc_kind = 'pool'), exactly B4's
     schedule;
  2. per-policy CS = sum_s w_s [lo_{s, pi(s)}, hi_{s, pi(s)}] built from the per-cell WSR20 WoR CSs (delta / (S A)
     per cell). On the cell event every policy CS covers simultaneously, which replaces the union over Pi
     (alpha / m) of Theorem 1 -- tighter here (|Pi| = 512 policies vs 18 cells). Prop. 1's i.i.d. CS is replaced by
     WSR20 because the replay is without replacement;
  3. eps-relaxed Theorem 1 rule per problem q (class Pi_{B_q}): leader pi_hat_q = argmax_{Pi_{B_q}} LCB (the policy
     defining max L_t in S_t; ties -> lowest index), and
        U_q = max_{pi' in Pi_{B_q}, pi' != pi_hat_q} UCB(pi') - LCB(pi_hat_q),   certify iff U_q <= eps.
     (methodology s4.2 writes the max over all of Pi_{B_q}; including pi' = pi_hat only adds the leader's own
     CS width and is never needed for validity, so the faithful pi' != pi_hat form -- more favourable to the
     rival -- is used.) If Pi_{B_q} = {pi_hat}, U_q = 0.
Validity: on the cell event, J(pi*) <= UCB(pi*) and J(pi_hat) >= LCB(pi_hat), so pi* != pi_hat implies
J(pi*) - J(pi_hat) <= U_q <= eps. Rigorous. Dominated by construction by B4 (same schedule; the per-policy interval
keeps the CS width of the segments where pi' and pi_hat agree) -- reported as such.
"""
from __future__ import annotations

import numpy as np

from .frontier_common import Method

__all__ = ["MolitorWoR", "molitor_certificate"]


def molitor_certificate(ctx, st):
    lo, hi = st.cs()
    seg = np.arange(ctx.S)[None, :]
    lcb = (ctx.w[None, :] * lo[seg, ctx.pols]).sum(1)
    ucb = (ctx.w[None, :] * hi[seg, ctx.pols]).sum(1)
    out = []
    for q in range(ctx.Q):
        f = ctx.feas[q]
        ih = int(np.argmax(np.where(f, lcb, -np.inf)))
        others = f.copy()
        others[ih] = False
        U = float(np.max(ucb[others]) - lcb[ih]) if others.any() else 0.0
        out.append((bool(U <= ctx.eps), ih, U))
    return out


class MolitorWoR(Method):
    name = "Molitor-WoR"
    alloc_kind = "pool"
    validity = "rigorous"

    def certify(self, ctx, st):
        return molitor_certificate(ctx, st)

    def describe(self):
        d = super().describe()
        d.update({"certificate": "per-policy CS = sum_s w_s cell WSR20 CS; U = max_{pi'!=pi_hat} UCB - LCB(pi_hat)",
                  "leader": "argmax LCB over Pi_{B_q}"})
        return d
