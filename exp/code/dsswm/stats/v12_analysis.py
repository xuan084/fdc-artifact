"""Lock-v12 addendum: pre-registered rules and analysis (NEW file; plan/v12_obd2_plan.md).  Every v11 rule is imported
unchanged from ``v11_analysis`` (methods, eps grid, HC grid, rival-success rule, THRESH rule, decision, cell analysis);
v12 adds only the campaign seeds, the pre-stated block-status gate, the descriptive fallback eps and two pre-stated
descriptive statistics (full-frontier ratio, uncensored-subset ratio).

Block-status gate (``block_status``; plan s6), on the campaign's dev half at the rule-selected eps, from the dev truth:
confirmatory iff (i) the rule selects some eps of the grid, (ii) all-control is not eps-optimal in at least one of the
15 problems, and (iii) the mean share of feasible policies that are eps-optimal is <= 0.5; otherwise descriptive.
If (i) fails the block runs at FALLBACK_EPS = the largest grid eps (descriptive).
"""
from __future__ import annotations

import numpy as np

from .v6_analysis import boot_idx, paired
from .v11_analysis import (DESCRIPTIVE, DEV_SEEDS, EPS_GRID, HC_GRID, HC_NAMES, HC_TUNED, K_GRID, METHODS, N80_K,
                           PRIMARY, RECT_HG, RIVAL, RULE_MIN_SUCCESS, THRESH_CAP, THRESH_MARGIN, _by, _vec,
                           analyse_cell, decide, geomean, select_eps, select_hc, thresh_rule)

__all__ = ["PRIMARY", "RIVAL", "HC_TUNED", "RECT_HG", "DESCRIPTIVE", "METHODS", "EPS_GRID", "HC_GRID", "HC_NAMES",
           "RULE_MIN_SUCCESS", "DEV_SEEDS", "EVAL_SEEDS", "K_GRID", "N80_K", "THRESH_CAP", "THRESH_MARGIN",
           "FALLBACK_EPS", "SHARE_MAX", "VALIDITY_LABEL", "select_hc", "select_eps", "block_eps", "block_status",
           "thresh_rule", "decide", "decide_block", "analyse_cell", "analyse_cell_v12", "geomean"]

EVAL_SEEDS = {"women": tuple(range(39200, 39400)), "men": tuple(range(39400, 39600))}
FALLBACK_EPS = max(EPS_GRID)
SHARE_MAX = 0.5
# RECT-HG-DP has a checkpoint-strength guarantee only; v12 rows carry this label instead of the class's 'rigorous'.
VALIDITY_LABEL = {RECT_HG: "checkpoint_strength_CP"}


def block_eps(rule):
    """The eps the block runs at: the rule-selected eps, else FALLBACK_EPS (block descriptive by gate (i))."""
    e = rule.get("selected_eps")
    return float(e) if e is not None else float(FALLBACK_EPS)


def block_status(rule, share):
    """Pre-stated gate (plan s6).  ``share`` = obd_v11_eval.eps_opt_share(dev env, dev J*, dev mu, block eps)."""
    n_q = len(share["per_problem"])
    c1 = rule.get("selected_eps") is not None
    c2 = int(share["n_problems_all_control_eps_optimal"]) < n_q
    c3 = float(share["mean_share_eps_optimal"]) <= SHARE_MAX
    return {"status": "confirmatory" if (c1 and c2 and c3) else "descriptive",
            "criteria": {"i_rule_selects_eps": bool(c1),
                         "ii_all_control_not_eps_optimal_in_some_problem": bool(c2),
                         "iii_mean_eps_optimal_share_le_0.5": bool(c3)},
            "eps": block_eps(rule), "selected_eps": rule.get("selected_eps"),
            "n_problems": n_q, "n_problems_all_control_eps_optimal": int(share["n_problems_all_control_eps_optimal"]),
            "mean_share_eps_optimal": float(share["mean_share_eps_optimal"]),
            "per_problem_share": share["per_problem"], "n_policies": int(share["n_policies"]),
            "rule": "confirmatory iff (i) the rival-success rule selects an eps of the grid, (ii) all-control is not "
                    "eps-optimal in at least one of the 15 problems, (iii) mean eps-optimal policy share <= 0.5 "
                    "(dev half, dev truth, rule-selected eps); else descriptive (plan/v12_obd2_plan.md s6)"}


def decide_block(ratio, primary_false_streams, replica, thresh, status):
    """Confirmatory block: v11 ``decide`` unchanged, labelled with the frozen block status.
    Descriptive block: the inputs are validated by the same function, but NO verdict is issued: the record carries
    ``verdict = 'descriptive'``, THRESH marked not applicable, and only the descriptive quantities (ratio, UB95, CI,
    primary false streams, replica status); no positive / negative wording (plan s6; coordinator 2026-10-05)."""
    if status not in ("confirmatory", "descriptive"):
        raise ValueError(f"invalid block status {status!r}")
    d = decide(ratio, primary_false_streams, replica, thresh)
    if status == "confirmatory":
        d["block_status"] = status
        d["confirmatory"] = True
        d["thresh_applicable"] = True
        return d
    return {"verdict": "descriptive", "block_status": status, "confirmatory": False, "thresh_applicable": False,
            "thresh_computed": thresh, "ratio": float(ratio["geomean_ratio"]), "ub95": float(ratio["ub95_one_sided"]),
            "ci95_two_sided": ratio.get("ci95_two_sided"), "primary_false_streams": primary_false_streams,
            "replica_status": d["replica_status"],
            "rule": "descriptive block (pre-stated block-status gate failed on dev): ratio, UB95, false streams, "
                    "eps-optimal share, exhaustion, full-frontier and uncensored-subset readings are reported; no "
                    "verdict and no THRESH comparison"}


def analyse_cell_v12(rows, seeds, eps, B=None):
    """v11 ``analyse_cell`` plus the pre-stated descriptive statistics (plan s7):
    * full-frontier ratio: paired geomean of N_stop_pen (rows to 15/15; tau if never or after a false certificate),
      PRIMARY / RIVAL, same bootstrap (horizon-sensitive; never the endpoint);
    * uncensored-subset ratio: PRIMARY / RIVAL N80_pen on the streams where neither had an exhausted pool at N80
      (exhaustion_at_k80.all == 0 for both), bootstrap B = 10^4 seed 42 on that subset."""
    kw = {} if B is None else {"B": B}
    cell = analyse_cell(rows, seeds, eps, **kw)
    R = _by(rows)
    P = _vec(R, PRIMARY, seeds, eps)
    Q = _vec(R, RIVAL, seeds, eps)
    nB = cell["bootstrap"]["B"]
    full = paired([r["N_stop_pen"] for r in P], [r["N_stop_pen"] for r in Q], boot_idx(len(seeds), B=nB))
    full.update({"geomean_rows_primary": geomean([r["N_stop_pen"] for r in P]),
                 "geomean_rows_rival": geomean([r["N_stop_pen"] for r in Q]),
                 "rival_reaches_15_only_at_tau": int(sum(r["N_stop_pen"] >= r["tau_R"] for r in Q)),
                 "primary_reaches_15_only_at_tau": int(sum(r["N_stop_pen"] >= r["tau_R"] for r in P)),
                 "role": "descriptive (pre-stated; horizon-sensitive)"})
    keep = np.array([((p.get("exhaustion_at_k80") or {}).get("all", 1.0) == 0) and
                     ((q.get("exhaustion_at_k80") or {}).get("all", 1.0) == 0) for p, q in zip(P, Q)], dtype=bool)
    n = int(keep.sum())
    if n >= 2:
        unc = paired(np.array([r["N80_pen"] for r in P], float)[keep], np.array([r["N80_pen"] for r in Q], float)[keep],
                     boot_idx(n, B=nB))
    else:
        unc = {"geomean_ratio": None, "ub95_one_sided": None, "n": n}
    unc.update({"n_streams_in_subset": n, "n_streams": len(seeds),
                "subset_equals_rival_before_tau": bool(np.array_equal(keep, np.array([bool(q["n80_lt_tau"])
                                                                                     for q in Q]))),
                "role": "descriptive (pre-stated in plan/v12_obd2_plan.md s7)"})
    cell["descriptive_v12"] = {"full_frontier": full, "uncensored_subset": unc}
    return cell


_ = (HC_GRID, HC_NAMES, DESCRIPTIVE, select_eps, select_hc, thresh_rule)
