"""B1: CLUCB-frontier-joint (Chen, Lin, King, Lyu & Chen, NeurIPS 2014, joint multi-problem version), round 4.

Sampling (every batch of ``ctx.replan`` arrivals): for every undecided problem q, pi_hat_q = argmax of J_hat over
Pi_{B_q}; the CLUCB challenger is pi~_q = argmax_{pi' in Pi_{B_q}} sum_{s: pi'(s) != pi_hat(s)} w_s (hi[s, pi'(s)] -
lo[s, pi_hat(s)]) (the CLUCB adjusted-weight oracle with the cell CS endpoints in place of mu_hat +/- rad). The cells
of the union over undecided q of the symmetric differences pi_hat_q (+) pi~_q are the candidates; arrivals are
exogenous segments, so in segment s the target arm is the candidate arm with the largest CS width (if segment s has
no candidate cell, the arm with the largest CS width -- the arrival is billed either way). Re-selection order on pool
exhaustion: descending CS width.

Certificate: ``rect_certificate`` -- per-cell WSR20 time-uniform WoR CS radius sum (rigorous for adaptive counts).
Stopping = CLUCB stopping (U_q <= eps); the harness freezes answers and stops at 12/15.
"""
from __future__ import annotations

import numpy as np

from .frontier_common import Method, pi_hat_indices, rect_U, rect_certificate

__all__ = ["CLUCBJoint", "clucb_pref"]


def clucb_pref(ctx, st):
    lo, hi = st.cs()
    width = hi - lo
    ihs = pi_hat_indices(ctx, st.mu_hat)
    cand = np.zeros((ctx.S, ctx.A), dtype=bool)
    for q in np.flatnonzero(st.undecided):
        U = np.where(ctx.feas[q], rect_U(ctx, lo, hi, ihs[q]), -np.inf)
        br = int(np.argmax(U))
        ph, pc = ctx.pols[ihs[q]], ctx.pols[br]
        d = ph != pc
        s_idx = np.flatnonzero(d)
        cand[s_idx, ph[s_idx]] = True
        cand[s_idx, pc[s_idx]] = True
    pref = np.zeros((ctx.S, ctx.A), dtype=np.int64)
    for s in range(ctx.S):
        order = list(np.argsort(-width[s], kind="stable"))
        if cand[s].any():
            first = max(np.flatnonzero(cand[s]), key=lambda a: (width[s, a], -a))
            order.remove(first)
            order = [int(first)] + [int(a) for a in order]
        pref[s] = order
    return pref


class CLUCBJoint(Method):
    name = "B1"
    alloc_kind = "adaptive"
    validity = "rigorous"

    def plan(self, ctx, st):
        return clucb_pref(ctx, st)

    def certify(self, ctx, st):
        return rect_certificate(ctx, st)
