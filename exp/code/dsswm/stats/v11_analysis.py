"""Lock-v11 addendum: pre-registered rules and analysis (new file; frozen with the v11 addendum; plan/v11_obd_plan.md).

Methods (all on the frozen outcome-free 50/50 design, the K = 40 checkpoint grid, delta 0.05, run to 15/15 or tau):
  PRIMARY      'TU-FDC-DP(b)'      FDCDPTimeUniform(block points = the K = 40 grid, scheme 'b')
  RIVAL        'RECT-BF-DP-TU'     RectBFDPTU(block points = the K = 40 grid, box=True), parameter-free
  descriptive  'HC-WoR-DP[tuned]'  HCWoRDP('nstar', c, tf) with (c, tf) tuned on dev by ``select_hc`` (block-G3 rule)
  descriptive  'RECT-HG-DP'        exact hypergeometric cells at delta / (2 S A K); checkpoint-strength (CP) guarantee

eps rule (``select_eps``; never sees FDC-DP rows): smallest eps of EPS_GRID at which the dev-best of {RECT-BF-DP-TU,
HC-WoR-DP tuned at that eps} (lower dev geometric-mean N80_pen; tie -> RECT-BF-DP-TU) has strict N80_pen < tau on
>= 40 / 50 dev streams; none -> no confirmatory cell.
HC rule (``select_hc``): per eps, the config of HC_GRID with the lowest dev geometric-mean N80_pen; ties -> grid order.
THRESH rule (``thresh_rule``): THRESH = min(0.90, UB95_dev + 0.10), rounded UP to 3 decimals, UB95_dev = one-sided
UB95 of the primary / rival paired geometric-mean N80_pen ratio on the 50 dev streams of the dev eps cell (B = 10^4,
seed 42).
Statistic: paired geometric-mean ratio of N80_pen, paired percentile bootstrap over the 200 eval streams (B = 10^4,
seed 42, one resample matrix for every comparator), one-sided UB95 (``v6_analysis.paired``).
Decision (``decide``): 'positive_result_achieved' iff UB95(PRIMARY / RIVAL) < THRESH AND PRIMARY has 0 / 200 false
streams over the whole run AND the replica check passes; otherwise 'positive_result_not_achieved' with the failing
components.
"""
from __future__ import annotations

import math

import numpy as np

from .v6_analysis import B_BOOT, BOOT_SEED, boot_idx, cp_upper, paired

__all__ = ["PRIMARY", "RIVAL", "HC_TUNED", "RECT_HG", "DESCRIPTIVE", "METHODS", "EPS_GRID", "HC_GRID", "HC_NAMES",
           "RULE_MIN_SUCCESS", "DEV_SEEDS", "EVAL_SEEDS", "K_GRID", "N80_K", "THRESH_CAP", "THRESH_MARGIN",
           "select_hc", "select_eps", "thresh_rule", "decide", "analyse_cell", "matrix", "geomean", "hc_name"]

PRIMARY = "TU-FDC-DP(b)"
RIVAL = "RECT-BF-DP-TU"
HC_TUNED = "HC-WoR-DP[tuned]"
RECT_HG = "RECT-HG-DP"
DESCRIPTIVE = (HC_TUNED, RECT_HG)
METHODS = (PRIMARY, RIVAL) + DESCRIPTIVE
EPS_GRID = (3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3)
HC_GRID = tuple((c, tf) for c in (0.5, 0.75) for tf in (0.2, 0.4, 0.6, 0.8))   # = run_v10_posthoc_G.HC_GRID


def hc_name(c, tf):
    return f"HC-WoR-DP[c={c},tf={tf}]"


HC_NAMES = tuple(hc_name(c, tf) for c, tf in HC_GRID)
RULE_MIN_SUCCESS = 40
DEV_SEEDS = tuple(range(950, 1000))
EVAL_SEEDS = tuple(range(39000, 39200))
K_GRID = 40
N80_K = 12
THRESH_CAP = 0.90
THRESH_MARGIN = 0.10
_FDC_PREFIXES = ("TU-FDC", "FDC")


def geomean(v):
    v = np.asarray(v, dtype=float)
    return float(np.exp(np.mean(np.log(v))))


def _by(rows):
    out = {}
    for r in rows:
        k = (r["method"], int(r["seed"]), float(r["eps"]))
        if k in out:
            raise ValueError(f"duplicate row {k}")
        out[k] = r
    return out


def _vec(R, m, seeds, eps, key="N80_pen"):
    v = [R.get((m, int(s), float(eps))) for s in seeds]
    if any(x is None for x in v):
        raise ValueError(f"{m} lacks {sum(x is None for x in v)} of {len(seeds)} streams at eps {eps}")
    return v


def select_hc(rows, seeds=DEV_SEEDS, grid=EPS_GRID):
    """Per eps: HC config with the lowest dev geomean N80_pen over HC_GRID; ties -> grid order."""
    R = _by([r for r in rows if r["method"] in HC_NAMES])
    out = {}
    for e in grid:
        g = {}
        for name, (c, tf) in zip(HC_NAMES, HC_GRID):
            rr = _vec(R, name, seeds, e)
            g[name] = {"c": c, "tf": tf, "geomean_N80_pen": geomean([r["N80_pen"] for r in rr]),
                       "successes": int(sum(r["N80_pen"] < r["tau_R"] for r in rr)),
                       "false_streams": int(sum(bool(r["fwer_event"]) for r in rr))}
        best = min(HC_NAMES, key=lambda n: (g[n]["geomean_N80_pen"], HC_NAMES.index(n)))
        out[repr(float(e))] = {"name": best, "c": g[best]["c"], "tf": g[best]["tf"], "grid": g}
    return out


def select_eps(rows, hc_sel, seeds=DEV_SEEDS, grid=EPS_GRID, min_success=RULE_MIN_SUCCESS):
    """v10 rival-success rule with the per-eps tuned HC.  Refuses FDC-DP rows (the rule never reads FDC-DP)."""
    if any(str(r["method"]).startswith(_FDC_PREFIXES) for r in rows):
        raise ValueError("select_eps: FDC-DP rows must not be passed to the rival-success rule")
    R = _by(rows)
    trace = []
    for e in sorted(grid):
        hc = hc_sel[repr(float(e))]["name"]
        cand = {RIVAL: RIVAL, "HC-WoR-DP[tuned]": hc}
        cnt = {}
        for lab, m in cand.items():
            rr = _vec(R, m, seeds, e)
            cnt[lab] = {"method": m, "successes": int(sum(r["N80_pen"] < r["tau_R"] for r in rr)),
                        "geomean_N80_pen": geomean([r["N80_pen"] for r in rr]), "n": len(rr)}
        order = (RIVAL, "HC-WoR-DP[tuned]")
        best = min(order, key=lambda m: (cnt[m]["geomean_N80_pen"], order.index(m)))
        ok = cnt[best]["successes"] >= min_success
        trace.append({"eps": float(e), "best": best, "best_method": cnt[best]["method"], "per_rect": cnt,
                      "qualifies": bool(ok)})
        if ok:
            return {"selected_eps": float(e), "rival": best, "rival_method": cnt[best]["method"],
                    "successes": cnt[best]["successes"], "n": len(seeds), "trace": trace}
    return {"selected_eps": None, "rival": None, "rival_method": None, "successes": None, "n": len(seeds),
            "trace": trace}


def thresh_rule(ub95_dev):
    u = float(ub95_dev)
    if not (math.isfinite(u) and u > 0):
        raise ValueError(f"invalid dev UB95 {ub95_dev!r}")
    return min(THRESH_CAP, math.ceil((u + THRESH_MARGIN) * 1000 - 1e-9) / 1000)


def decide(ratio, primary_false_streams, replica, thresh):
    u = (ratio or {}).get("ub95_one_sided")
    if u is None or not math.isfinite(u) or u <= 0:
        raise ValueError(f"invalid UB {u!r}")
    if not (isinstance(thresh, float) and 0 < thresh < 1):
        raise ValueError(f"invalid THRESH {thresh!r}")
    fs = primary_false_streams
    if not isinstance(fs, int) or isinstance(fs, bool) or fs < 0:
        raise ValueError("primary_false_streams must be a non-negative int")
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r})")
    fails = []
    if u >= thresh:
        fails.append("rival_superiority_failed")
    if fs > 0:
        fails.append("validity_failure")
    if st != "pass":
        fails.append("replica_fail")
    return {"verdict": "positive_result_achieved" if not fails else "positive_result_not_achieved",
            "failing_components": fails, "ub95": float(u), "ratio": float(ratio["geomean_ratio"]), "thresh": thresh,
            "primary_false_streams": fs, "replica_status": st,
            "sub_label": None if u < thresh else ("faster_below_1" if u < 1.0 else "not_faster"),
            "rule": f"UB95({PRIMARY}/{RIVAL}, paired geometric-mean N80_pen) < THRESH = {thresh} AND 0 {PRIMARY} false "
                    f"streams (whole run) AND replica pass"}


def matrix(rows, methods, seeds, eps, key="N80_pen"):
    R = _by(rows)
    return {m: np.array([float(x[key]) for x in _vec(R, m, seeds, eps)]) for m in methods}


def _mean_exh(rr):
    vals = [r.get("exhaustion_at_k80") for r in rr]
    vals = [v for v in vals if v]
    if not vals:
        return None
    return {k: float(np.mean([v[k] for v in vals])) for k in ("all", "ctrl", "treat")} | {"n_streams": len(vals)}


def analyse_cell(rows, seeds, eps, idx=None, B=B_BOOT, methods=METHODS):
    idx = boot_idx(len(seeds), B=B) if idx is None else idx
    M = matrix(rows, methods, seeds, eps)
    R = _by(rows)
    out = {"eps": float(eps), "n_streams": len(seeds), "bootstrap": {"B": int(len(idx)), "seed": BOOT_SEED},
           "comparisons": {}, "methods": {}}
    for m in methods:
        if m != PRIMARY:
            out["comparisons"][f"{PRIMARY}/{m}"] = paired(M[PRIMARY], M[m], idx)
    for m in methods:
        if m not in (PRIMARY, RIVAL):
            out["comparisons"][f"{RIVAL}/{m}"] = paired(M[RIVAL], M[m], idx)
    for m in methods:
        rr = _vec(R, m, seeds, eps)
        tau = rr[0]["tau_R"]
        k = int(sum(bool(r["fwer_event"]) for r in rr))
        d = {"validity": rr[0].get("validity"), "tau_R": tau, "false_streams": k, "cp_ub95": cp_upper(k, len(rr)),
             "false_streams_by_k80": int(sum(bool(r.get("false_by_k80")) for r in rr)),
             "geomean_N80_pen": geomean(M[m]), "geomean_N80_over_tau": geomean(M[m] / tau),
             "share_N80_lt_tau": float(np.mean([bool(r["n80_lt_tau"]) for r in rr])),
             "share_reached_15": float(np.mean([bool(r["reached_stop"]) for r in rr])),
             "exhaustion_at_k80_mean": _mean_exh(rr)}
        if rr[0].get("node_limit_hits") is not None:
            d.update({"node_limit_hits": int(sum(r["node_limit_hits"] for r in rr)),
                      "bnb_calls": int(sum(r["bnb_calls"] for r in rr)),
                      "b_only_certs": int(sum(r["b_only_certs"] for r in rr))})
        if rr[0].get("hc_config") is not None:
            d["hc_config"] = rr[0]["hc_config"]
        out["methods"][m] = d
    out["primary_false_streams"] = out["methods"][PRIMARY]["false_streams"]
    return out
