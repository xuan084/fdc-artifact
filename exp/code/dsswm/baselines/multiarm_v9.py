"""General-A (A >= 2) versions of FDC-BF, PJC-BF-local and the frozen-plan rectangles (v9 block C, Hillstrom A = 3).

NEW file: fdc_bet.py, pjc_bf.py, rect_v6.py, wor_betting_v6.py (v6-v8 bound) are imported, never modified.

Audit of the A = 2 restriction (2026-10-04)
------------------------------------------
* ``fdc_bet.direction_widths`` is already written for general A: a policy difference pi' - pi_hat touches, in every
  segment s with pi'(s) != pi_hat(s), exactly two cells, the challenger cell (s, pi'(s)) with coefficient -w_s and the
  centre cell (s, pi_hat(s)) with coefficient +w_s; cells (s, a) with a not in {pi'(s), pi_hat(s)} have coefficient 0.
  For A = 3 the two touched cells of a segment are still distinct pools, so conditional on the outcome-free schedule
  the cell sums are independent hypergeometric variables (Fact 0) and the Chernoff bound of Theorem FDC-bet-1 applies
  to Y = sum_c a_c (mu_hat_c - mu_c) unchanged.  The union (sum_q |Pi_{B_q}| x K direction events; variance box over
  S A K (cell, checkpoint) pairs) is counted from the ctx for any A.  The only A = 2 lock is the guard in
  ``FDCBet.setup`` (plus the design ``balanced_alloc(S, A, 0.5)``, which for A = 3 would be 0.5 / 0.25 / 0.25).
* ``PJCBF``: phase-local estimator, deficit tracking, Neyman rule and ledger are general in A; only the guard in
  ``setup`` and the 'half' design (control 0.5) are A = 2 specific.
* ``RectCkHG`` / ``RectCkBF`` / ``HCWoRRect`` / ``RectCkHGPlan`` / ``HCWoRPlan`` have no A guard; their per-cell
  intervals and the rectangle bound are general in A.  ``share`` is the CONTROL share (column 0) and the other A - 1
  arms split the rest equally (``balanced_alloc``).
* The runner (frontier_runner.run_stream / StreamEngine / make_schedule) and the frontier solvers are general in A.

Design convention for A = 3 (pre-declared before any Hillstrom certification run)
-------------------------------------------------------------------------------
'bal' = uniform 1/A per arm in every segment, the natural generalisation of FDC's 50/50 (it is also the RCT's own
assignment ratio).  FDC-BF-A uses 'bal' and is NOT tuned; RECT-ck-BF-A (matched rectangle) uses the same design.
Rivals choose their own frozen plan from PLAN_TAGS_A3 on development seeds (v5 rule).
"""
from __future__ import annotations

import numpy as np

from .b4_bal import balanced_alloc
from .fdc_bet import FDC_BET_DELTA, FDCBet, bet_ledger
from .pjc_bf import PJC_SPLIT, PJCBF, HCWoRPlan, RectCkBF, RectCkHGPlan, neyman_matrix, pjc_ledger

__all__ = ["uniform_alloc", "plan_matrix", "FDCBetA", "PJCBFA", "RectCkBFA", "make_plan_rect", "make_plan_hc",
           "PLAN_TAGS_A3", "fdc_bf_a"]

# control share tags ('bal' = 1/A each arm) and frozen variance plans from the dev half
PLAN_TAGS_A3 = ("bal", "0.2", "0.4", "0.5", "0.6", "ney", "ney23")


def uniform_alloc(S, A):
    return np.full((S, A), 1.0 / A)


def plan_matrix(tag, S, A, sigma2=None):
    """Frozen (S, A) read plan for a tag.  'ney' / 'ney23' need dev-half cell variances sigma2 (outcome-free w.r.t.
    any eval half)."""
    if tag == "bal":
        return uniform_alloc(S, A)
    if tag in ("ney", "ney23"):
        if sigma2 is None:
            raise ValueError("Neyman plans need dev-half cell variances")
        s2 = np.maximum(np.asarray(sigma2, dtype=float), 0.0)
        return neyman_matrix(s2 if tag == "ney" else s2 ** (2.0 / 3.0), None, 0.1)
    return balanced_alloc(S, A, float(tag))


class FDCBetA(FDCBet):
    """FDC-bet certificate for general A with a frozen, data-free (or dev-frozen) design matrix.

    Identical to ``FDCBet`` except: no A = 2 guard; design = ``alloc`` (default uniform 1/A).  With A = 2 and the
    default design this is exactly FDCBet (balanced_alloc(S, 2, 0.5) = uniform) -- checked by unit test."""

    def __init__(self, kind="bennett", var_box="HG", fpc=True, rect=False, split=(0.045, 0.005), alloc=None,
                 name=None, grid=None):
        super().__init__(kind=kind, var_box=var_box, fpc=fpc, rect=rect, split=split, name=name, grid=grid)
        self._alloc_fixed = None if alloc is None else np.asarray(alloc, dtype=float)

    def setup(self, ctx):
        if abs(float(ctx.delta) - FDC_BET_DELTA) > 1e-12:
            raise ValueError("FDC-bet: delta fixed at 0.05")
        if not ctx.binary or ctx.A < 2:
            raise ValueError("FDC-bet-A: binary pools, A >= 2")
        if self._alloc_fixed is None:
            self.alloc_p = uniform_alloc(ctx.S, ctx.A)
        else:
            if self._alloc_fixed.shape != (ctx.S, ctx.A) or np.any(self._alloc_fixed <= 0):
                raise ValueError("design must be (S, A) with positive entries (a zero share starves a cell)")
            self.alloc_p = self._alloc_fixed / self._alloc_fixed.sum(1, keepdims=True)
        self.ledger = bet_ledger(ctx, self.split[0], self.split[1], self.var_box)
        self._lo = np.zeros((ctx.S, ctx.A))
        self._hi = np.ones((ctx.S, ctx.A))
        self._last_t = -1

    def describe(self):
        d = super().describe()
        d["design"] = "frozen plan " + ("uniform 1/A" if self._alloc_fixed is None else "matrix")
        return d


def fdc_bf_a(alloc=None, name="FDC-BF"):
    """FDC-BF (Bennett-FPC + exact HG variance box, 0.045 / 0.005) for general A; default design uniform 1/A."""
    return FDCBetA(kind="bennett", var_box="HG", fpc=True, rect=False, split=(0.045, 0.005), alloc=alloc, name=name)


class PJCBFA(PJCBF):
    """PJC-BF (local mode) for general A.  Rule 'bal' replaces 'half': phase plans are uniform 1/A (deterministic
    tracking); rule 'neyman' as in PJCBF.  Phase 0 always uses the uniform plan.  Menu mode is not offered for A > 2."""

    def __init__(self, boundaries=(2,), rule="neyman", rect=False, p_floor=0.1, kind="bennett", name=None):
        if rule not in ("bal", "neyman"):
            raise ValueError(rule)
        super().__init__(mode="local", boundaries=boundaries, rule="half" if rule == "bal" else rule, rect=rect,
                         p_floor=p_floor, kind=kind, name=name)
        self.rule_a = rule
        if name is None:
            bnd = ",".join(str(b) for b in self.boundaries)
            self.name = f"PJC-BF[local,b={bnd},{rule}{',R' if rect else ''}]"

    def setup(self, ctx):
        if abs(float(ctx.delta) - 0.05) > 1e-12:
            raise ValueError("PJC-BF: delta fixed at 0.05")
        if not ctx.binary or ctx.A < 2:
            raise ValueError("PJC-BF-A: binary pools, A >= 2")
        K = len(ctx.checkpoints)
        if any(b < 0 or b >= K - 1 for b in self.boundaries):
            raise ValueError("boundaries must be checkpoint indices in [0, K-2]")
        self.ledger = pjc_ledger(ctx, "local", self.boundaries, 1, PJC_SPLIT)
        S, A = ctx.S, ctx.A
        self._p = uniform_alloc(S, A)
        self._n0 = np.zeros((S, A), dtype=np.int64)
        self._s0 = np.zeros((S, A))
        self._lo = np.zeros((S, A))
        self._hi = np.ones((S, A))
        self._last_t = -1
        self._phase = 0
        self.path = []
        self.plans = [self._p.copy()]

    def _next_plan(self, ctx, st):
        if self.rule_a == "bal":
            return uniform_alloc(ctx.S, ctx.A), None
        return super()._next_plan(ctx, st)


class RectCkBFA(RectCkBF):
    """Matched Bennett-FPC rectangle with FDC-BF-A's design (uniform 1/A by default)."""

    def __init__(self, box=False, alloc=None, name=None):
        super().__init__(box=box, share=0.5, name=name)
        self._alloc_fixed = None if alloc is None else np.asarray(alloc, dtype=float)

    def setup(self, ctx):
        super().setup(ctx)
        self.alloc_p = (uniform_alloc(ctx.S, ctx.A) if self._alloc_fixed is None else
                        self._alloc_fixed / self._alloc_fixed.sum(1, keepdims=True))


def make_plan_rect(tag, S, A, sigma2=None, name=None):
    """RECT-ck-HG with its own frozen read plan (tag from PLAN_TAGS_A3)."""
    return RectCkHGPlan(alloc=plan_matrix(tag, S, A, sigma2), name=name or f"RECT-ck-HG[{tag}]")


def make_plan_hc(schedule, c, target_frac, tag, S, A, sigma2=None, name=None):
    """HC-WoR (hedged-capital WoR CS + rectangle) with its own frozen read plan; n_star follows the plan."""
    tf = "-" if target_frac is None else f"{target_frac:g}"
    return HCWoRPlan(schedule, c, target_frac, alloc=plan_matrix(tag, S, A, sigma2),
                     name=name or f"HC-WoR{{{schedule},{c:g},{tf}}}[{tag}]")
