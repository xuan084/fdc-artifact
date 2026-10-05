"""Lock-v10 addendum: pre-registered analysis (new file; frozen with the v10 addendum).

Statistic (as v6-v9, ``v6_analysis.paired`` / ``boot_idx``): per-stream log ratio d_i = log N80_pen(primary, i) -
log N80_pen(r, i); point estimate exp(mean d) = paired geometric-mean ratio; paired percentile bootstrap over the 200
streams, B = 10^4, seed 42, ONE resample matrix for every cell, comparator and endpoint; one-sided 95 % UB =
exp(q_0.95 of the bootstrap mean); two-sided 95 % CI reported.

Endpoint.  N80_pen = the K = 20 checkpoint at which 12 of the 15 problems are first certified, if no false certificate
exists by then, else tau_R (v9 rule); the stream continues to 15/15 or tau_R and a 'false stream' is any false
certificate over the WHOLE run (external reviewer FDC-DP r1 B2).

Methods (every one on the frozen 50/50 design, the K = 20 block grid, delta 0.05):
  PRIMARY  'TU-FDC-DP(b)'  FDC-DP scheme (b) read under its time-uniform guarantee (FDCDPTimeUniform, block points =
           the K = 20 grid = the monitoring grid; on that grid numerically identical to FDC-DP(b), unit test)
  RIVAL    'RECT-BF-DP-TU' matched Bennett-FPC rectangle with radii frozen at block points (RectBFDPTU; identical to
           RECT-BF-DP on the block grid), parameter-free
  descriptive 'HC-WoR-DP' (hedged-capital WoR CS rectangle; frozen v8 S = 9 configuration transferred UNTUNED, plan
           0.5) and 'FDC-DP(a)' (single-column scheme).

Decision per confirmatory block (IUT over the block's cells; blocks are reported separately; there is NO joint claim
unless both blocks pass):
  'positive_result_achieved' iff for EVERY registered cell c of the block
      (1) UB95(TU-FDC-DP(b) / RECT-BF-DP-TU, N80_pen) < 0.80, and
      (2) TU-FDC-DP(b) has 0 / 200 false streams (whole run),
  and (3) the replica check passes for every task of the block.
  Otherwise 'positive_result_not_achieved' with failing components (per cell).  Each cell is also reported on its own.
Claims are restricted to the named rectangle implementations (RECT-BF-DP-TU; HC-WoR-DP as an untuned transfer).
Descriptive cells: same statistics, no verdict.  Lenta exhaustion qualification: the mean fraction of exhausted cells
(all / control / treatment) at each method's N80 checkpoint is reported for every cell.
"""
from __future__ import annotations

import math

import numpy as np

from .v6_analysis import B_BOOT, boot_idx, cp_upper, paired

__all__ = ["PRIMARY", "RIVAL", "DESCRIPTIVE", "METHODS", "THRESHOLD", "CELLS", "DESC_CELLS", "BLOCK_SEEDS",
           "cell_spec", "decide_block", "decide_joint", "analyse_cell", "analyse_block", "matrix"]

PRIMARY = "TU-FDC-DP(b)"
RIVAL = "RECT-BF-DP-TU"
DESCRIPTIVE = ("HC-WoR-DP", "FDC-DP(a)")
METHODS = (PRIMARY, RIVAL) + DESCRIPTIVE
THRESHOLD = 0.80
BLOCK_SEEDS = {"A": tuple(range(38000, 38200)), "B": tuple(range(38200, 38400))}
# confirmatory cells = the eps chosen by the pre-stated rival-success rule on dev (v2 dev, post external reviewer fixes)
CELLS = {
    "A": {"v10a_full_s16": {"data": "x5", "S": 16, "eps": 0.03},
          "v10a_full_s32": {"data": "x5", "S": 32, "eps": 0.04}},
    "B": {"v10b_full_s16": {"data": "lenta", "S": 16, "eps": 0.004},
          "v10b_full_s32": {"data": "lenta", "S": 32, "eps": 0.006},
          "v10b_full_s64": {"data": "lenta", "S": 64, "eps": 0.008}},
}
# descriptive cells (rivals censored on dev under the rule); seeds of their table's block
DESC_CELLS = {"v10d_full_x5s64": {"data": "x5", "S": 64, "eps": 0.05, "block": "A"}}


def cell_spec(task):
    for b, cells in CELLS.items():
        if task in cells:
            return dict(cells[task], block=b, role="confirmatory")
    if task in DESC_CELLS:
        return dict(DESC_CELLS[task], role="descriptive")
    raise KeyError(task)


def matrix(rows, methods, seeds, eps, key="N80_pen"):
    by = {(r["method"], int(r["seed"])): r for r in rows if abs(float(r["eps"]) - eps) < 1e-15}
    out = {}
    for m in methods:
        v = [by.get((m, int(s))) for s in seeds]
        if any(x is None for x in v):
            raise ValueError(f"matrix: {m} lacks {sum(x is None for x in v)} of {len(seeds)} streams at eps {eps}")
        out[m] = np.array([float(x[key]) for x in v])
    return out


def _check_ub(name, v):
    u = (v or {}).get("ub95_one_sided")
    if u is None or not math.isfinite(u) or u <= 0:
        raise ValueError(f"invalid UB for {name}: {u!r}")
    return float(u)


def decide_block(block, per_cell: dict, replica: dict, cells=None, thr=THRESHOLD) -> dict:
    """per_cell: {task: {'ratio': paired(...) of PRIMARY / RIVAL, 'primary_false_streams': int}} for exactly the
    registered cells of the block; replica: {'status': 'pass' | 'fail', ...} over all tasks of the block."""
    reg = tuple((CELLS[block] if cells is None else cells).keys())
    if set(per_cell) != set(reg):
        raise ValueError(f"cell results {sorted(per_cell)} != registered {sorted(reg)}")
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r})")
    out_cells = {}
    fails = []
    for t in reg:
        c = per_cell[t]
        ub = _check_ub(t, c.get("ratio"))
        fs = c.get("primary_false_streams")
        if not isinstance(fs, int) or isinstance(fs, bool) or fs < 0:
            raise ValueError(f"{t}: primary_false_streams must be a non-negative int")
        cf = []
        if ub >= thr:
            cf.append("rival_superiority_failed")
        if fs > 0:
            cf.append("validity_failure")
        out_cells[t] = {"ub95": ub, "ratio": float(c["ratio"]["geomean_ratio"]), "primary_false_streams": fs,
                        "pass": not cf, "failing": cf,
                        "sub_label": (None if ub < thr else ("faster_below_1" if ub < 1.0 else "not_faster"))}
        fails += [f"{t}:{x}" for x in cf]
    if st != "pass":
        fails.append("replica_fail")
    return {"block": block, "verdict": "positive_result_achieved" if not fails else "positive_result_not_achieved",
            "failing_components": fails, "cells": out_cells, "replica_status": st,
            "UB_star": max(v["ub95"] for v in out_cells.values()),
            "governing_cell": max(out_cells, key=lambda k: out_cells[k]["ub95"]),
            "rule": f"for every registered cell: UB95({PRIMARY}/{RIVAL}, paired geometric-mean N80_pen) < {thr} AND 0 "
                    f"{PRIMARY} false streams (whole run); AND replica pass for every task of the block (IUT)"}


def decide_joint(dec_a: dict, dec_b: dict) -> dict:
    both = dec_a["verdict"] == dec_b["verdict"] == "positive_result_achieved"
    return {"verdict": "positive_on_both_blocks" if both else "no_joint_claim",
            "A": dec_a["verdict"], "B": dec_b["verdict"],
            "rule": "blocks are reported separately; a joint statement across X5 and Lenta is made only if BOTH blocks "
                    "are positive (IUT across blocks); no other cross-block combination"}


def _mean_exh(rr):
    vals = [r.get("exhaustion_at_k80") for r in rr]
    vals = [v for v in vals if v]
    if not vals:
        return None
    return {k: float(np.mean([v[k] for v in vals])) for k in ("all", "ctrl", "treat")} | {"n_streams": len(vals)}


def analyse_cell(task, rows, seeds, eps, idx=None, B=B_BOOT, methods=METHODS):
    idx = boot_idx(len(seeds), B=B) if idx is None else idx
    M = matrix(rows, methods, seeds, eps)
    by = {(r["method"], int(r["seed"])): r for r in rows if abs(float(r["eps"]) - eps) < 1e-15}
    out = {"task": task, "eps": float(eps), "n_streams": len(seeds), "comparisons": {}, "methods": {}}
    for m in methods:
        if m != PRIMARY:
            out["comparisons"][f"{PRIMARY}/{m}"] = paired(M[PRIMARY], M[m], idx)
    if "HC-WoR-DP" in methods:
        out["comparisons"][f"{RIVAL}/HC-WoR-DP"] = paired(M[RIVAL], M["HC-WoR-DP"], idx)
    for m in methods:
        rr = [by[(m, int(s))] for s in seeds]
        tau = rr[0]["tau_R"]
        k = int(sum(bool(r["fwer_event"]) for r in rr))
        d = {"validity": rr[0].get("validity"), "tau_R": tau, "false_streams": k, "cp_ub95": cp_upper(k, len(rr)),
             "false_streams_by_k80": int(sum(bool(r.get("false_by_k80")) for r in rr)),
             "geomean_N80_pen": float(np.exp(np.mean(np.log(M[m])))),
             "geomean_N80_over_tau": float(np.exp(np.mean(np.log(M[m] / tau)))),
             "share_N80_lt_tau": float(np.mean([bool(r["n80_lt_tau"]) for r in rr])),
             "share_reached_15": float(np.mean([bool(r["reached_stop"]) for r in rr])),
             "exhaustion_at_k80_mean": _mean_exh(rr)}
        if "node_limit_hits" in rr[0]:
            calls = int(sum(r["bnb_calls"] for r in rr))
            d.update({"node_limit_hits": int(sum(r["node_limit_hits"] for r in rr)), "bnb_calls": calls,
                      "b_only_certs": int(sum(r["b_only_certs"] for r in rr))})
        secs = [r.get("rs_sec_cert") for r in rr]
        if all(isinstance(x, dict) and x.get("by_k") for x in secs):
            ck = [v for x in secs for v in x["by_k"]]
            d["sec_cert_per_ck_median"] = float(np.median(ck))
            d["sec_cert_per_ck_max"] = float(np.max(ck))
        out["methods"][m] = d
    out["primary_false_streams"] = out["methods"][PRIMARY]["false_streams"]
    out["all_rigorous_false_streams_total"] = int(sum(v["false_streams"] for v in out["methods"].values()
                                                      if v["validity"] == "rigorous"))
    return out


def analyse_block(block, rows_by_task: dict, replica: dict, seeds=None, cells=None, B=B_BOOT):
    cells = CELLS[block] if cells is None else cells
    seeds = BLOCK_SEEDS[block] if seeds is None else tuple(seeds)
    if set(rows_by_task) != set(cells):
        raise ValueError(f"rows for {sorted(rows_by_task)} != registered cells {sorted(cells)}")
    idx = boot_idx(len(seeds), B=B)
    per = {t: analyse_cell(t, rows_by_task[t], seeds, cells[t]["eps"], idx=idx) for t in cells}
    dec = decide_block(block, {t: {"ratio": per[t]["comparisons"][f"{PRIMARY}/{RIVAL}"],
                                   "primary_false_streams": per[t]["primary_false_streams"]} for t in cells},
                       replica, cells=cells)
    return {"block": block, "n_streams": len(seeds), "cells": per, "decision": dec,
            "cell_specs": {t: dict(c) for t, c in cells.items()}}
