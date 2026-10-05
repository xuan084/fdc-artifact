"""Round-3 wording-tier decision (methodology 5.6; candidates.json cand_f.wording_tiers). Frozen before the v3 lock.

Thresholds are copied from idea/hypotheses.md / methodology 5.2 and must not be edited after the lock.
Decision order (methodology 5.6):
  1. Tier C if
       (a) clear-layer ratio S(JPC,full)/S(B3,full) on NL-R0 has 95% CI upper >= 1 ("CI contains 1"; a CI
           entirely above 1 is also C), or
       (b) zero-effect saving >= main non-tie saving, where saving = -mean_i log((S_JPC_full+1)/(S_B3_full+1))
           (zero-effect: all problems of the offset-null stream; main: non-tie problems of the NL-R0 quota stream), or
       (c) A1 is falsified on NL-R0 (A1 log-ratio 95% CI upper >= log 0.8).
  2. Tier A if A2 passes on BOTH layers AND clear-layer CI upper < 1 AND C1 passes.
  3. Otherwise tier B (A1 passes but A2 is not a two-layer pass, or the evidence is inconclusive).
Q wording is orthogonal: Q1 pass -> Lambda_hat_perp in main text as an ex-ante audit quantity; otherwise Thm 1-3 are
stated as conditional theorems with the negative result "no incremental ex-ante predictive power".

Operationalisation choices that the lock must confirm (hypotheses.md is silent): the clear-layer switch and the A1
falsification are read on NL-R0 (the headline layer, where JPC-vs-B3 is defined); E1-Lin values are reported only.
"""
from __future__ import annotations

import math

LOG06 = math.log(0.6)
LOG08 = math.log(0.8)
LOG125 = math.log(1.25)

TIER_TEXT = {
    "A": "joint evidence benefits from reuse significantly more than continuous baselines; reuse is also the main "
         "failure channel of static models (non-tie numbers)",
    "B": "reuse is a generic dividend; JPC's residual advantage holds only within realizable finite classes; "
         "headline moves to risk (C1/C2) and mechanism (Q)",
    "C": "savings are mainly epsilon-tie identification; paper becomes 'risks and mechanism of reuse'; 0.196/0.474 "
         "reported only as natural-generator conditional readings",
}
Q_TEXT = {
    True: "Lambda_hat_perp enters the main text as an ex-ante audit quantity",
    False: "Thm 1-3 stated as conditional theorems + negative result: no incremental ex-ante predictive power",
}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) or math.isinf(x) else None


def decide_tier_r3(inp: dict) -> dict:
    """inp keys (missing / None = not evaluable):
      A1_NL_ci_upper           log-ratio CI upper of JPC full/off, NL-R0
      A1_pass_NL, A1_pass_Lin  bool
      A2_pass_NL, A2_pass_Lin  bool
      clear_ratio_ci_upper     CI upper of S(JPC,full)/S(B3,full), clear layer, NL-R0 (ratio scale)
      saving_null, saving_main float (log scale, positive = JPC cheaper)
      C1_pass                  bool
      Q1_pass                  bool
    """
    reasons = []
    a1u = _f(inp.get("A1_NL_ci_upper"))
    cu = _f(inp.get("clear_ratio_ci_upper"))
    sn, sm = _f(inp.get("saving_null")), _f(inp.get("saving_main"))
    if cu is not None and cu >= 1.0:
        reasons.append("clear-layer JPC/B3 CI upper %.3f >= 1" % cu)
    if sn is not None and sm is not None and sn >= sm:
        reasons.append("zero-effect saving %.3f >= main non-tie saving %.3f" % (sn, sm))
    if a1u is not None and a1u >= LOG08:
        reasons.append("A1 falsified on NL-R0 (CI upper %.3f >= log 0.8)" % a1u)
    missing = [k for k in ("clear_ratio_ci_upper", "saving_null", "saving_main", "A1_NL_ci_upper")
               if _f(inp.get(k)) is None]
    q = inp.get("Q1_pass")
    qw = Q_TEXT[bool(q)] if q is not None else "Q1 not evaluable -> " + Q_TEXT[False]
    if reasons:
        tier, why = "C", "; ".join(reasons)
    elif (inp.get("A2_pass_NL") is True and inp.get("A2_pass_Lin") is True and cu is not None and cu < 1.0
          and inp.get("C1_pass") is True):
        tier, why = "A", "A2 passes on both layers, clear-layer CI upper < 1, C1 passes"
    else:
        parts = []
        if not (inp.get("A1_pass_NL") or inp.get("A1_pass_Lin")):
            parts.append("A1 not confirmed on either layer")
        if not (inp.get("A2_pass_NL") is True and inp.get("A2_pass_Lin") is True):
            parts.append("A2 not a two-layer pass")
        if inp.get("C1_pass") is not True:
            parts.append("C1 not passed")
        tier, why = "B", "; ".join(parts) or "default"
    return {"tier": tier, "reason": why, "wording": TIER_TEXT[tier], "q_wording": qw,
            "not_evaluable_inputs": missing}
