"""Three-tier wording decision (hypotheses.md "三档结论措辞"; methodology 6.3). Frozen by r2_prereg_lock.

This file is hashed into plan/prereg_lock.json (field `tier_decision.code_sha256`). r2_analysis_aggregate must import
`decide_tier` from here and assert the hash before use. Thresholds are the pre-registered values of idea/hypotheses.md:
  HR1 pass at an off-grid level  <=>  ratio CI upper < 0.8  AND  completion >= 0.80  AND  FCR CP upper <= 2*delta,
  ratio taken vs the strongest non-shared valid denominator of that level (denominator rule, methodology 4.3).
  Tier A: HR1 passes at R2 or R3, AND vs the same-threshold control (B3-UI if implemented, else the HR2
          'chi2 -> UI threshold' cell) the CI upper is also < 0.8.
  Tier B: HR1 passes only vs B3 (some level), and vs the same-threshold control the CI upper is >= 0.8.
  Tier C: HR1 fails at both R2 and R3, OR the G0 resolution gate failed.
Only levels listed in the locked Holm family A (field `holm_families.A`) count as "HR1 passes at R2 / R3".
"""
from __future__ import annotations

RATIO_UPPER = 0.8
COMPLETION_MIN = 0.80
DELTA = 0.05
FCR_CP_UPPER_MAX = 2 * DELTA


def hr1_pass(cell: dict) -> bool:
    """cell: {'ratio_ci_upper', 'completion', 'fcr_cp_upper'} for JPC(_infl) vs one denominator at one level."""
    if cell is None:
        return False
    try:
        return (float(cell["ratio_ci_upper"]) < RATIO_UPPER and float(cell["completion"]) >= COMPLETION_MIN
                and float(cell["fcr_cp_upper"]) <= FCR_CP_UPPER_MAX)
    except (KeyError, TypeError, ValueError):
        return False


def decide_tier(g0_passed: bool, family_a_levels: list, by_level: dict) -> dict:
    """by_level[level] = {'vs_best_valid': cell, 'vs_B3': cell, 'vs_same_threshold': cell} (cells as in hr1_pass).

    Returns {'tier': 'A'|'B'|'C', 'reason': str, 'passing_levels': [...]}.
    """
    offgrid = [lv for lv in ("R2", "R3") if lv in family_a_levels]
    if not g0_passed:
        return {"tier": "C", "reason": "G0 resolution gate failed (hypotheses.md tier C: 'or G0 failed')",
                "passing_levels": []}
    passing = [lv for lv in offgrid if hr1_pass((by_level.get(lv) or {}).get("vs_best_valid"))]
    if passing:
        st = [lv for lv in passing
              if (by_level[lv].get("vs_same_threshold") or {}).get("ratio_ci_upper", 1e9) < RATIO_UPPER]
        if st:
            return {"tier": "A", "reason": "HR1 passes off-grid and vs same-threshold control", "passing_levels": st}
    b3_only = [lv for lv in offgrid if hr1_pass((by_level.get(lv) or {}).get("vs_B3"))]
    if passing or b3_only:
        return {"tier": "B", "reason": "HR1 passes only vs B3 / not vs the same-threshold control",
                "passing_levels": sorted(set(passing) | set(b3_only))}
    return {"tier": "C", "reason": "HR1 fails at every off-grid level in family A", "passing_levels": []}
