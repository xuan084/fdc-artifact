"""Lock-v7 addendum: pre-registered analysis (new file; frozen with the v7 addendum).

Statistics as v6 (``v6_analysis.paired`` / ``boot_idx``): per-stream log ratio d_i = log N(FDC-BF, i) - log N(r, i);
point estimate exp(mean d); paired percentile bootstrap over streams, B = 10^4, seed 42, the SAME resample matrix for
every comparator and endpoint of a block; one-sided 95% UB = exp(q_0.95); two-sided 95% CI.

Block A (CONFIRMATORY, CR9 eval half, eps 0.001, seeds 33000-33199, primary endpoint N80_pen):
  primary family F = {RECT-ck-HG, RECT-ck-HG-live, HC-WoR} (same design, same-or-stronger guarantee rectangles);
  UB* = max_{r in F} UB95(FDC-BF / r).
  'positive_result_achieved' iff  UB* < 0.80  AND  FDC-BF has 0/200 false streams (any false certificate up to the
  end of its run, stop at 15/15 or tau_R)  AND  the replica check passed  (intersection-union test: every component
  must pass; no multiplicity correction needed for the conjunction).
  Otherwise 'positive_result_not_achieved' with the failing components listed; sub-labels
     'faster_below_1'  (UB* < 1 but >= 0.80),  'not_faster' (UB* >= 1),  'validity_failure' (FDC-BF false stream > 0),
     'replica_fail'.
  Secondary (descriptive, no verdict): FDC-BF/r for every other registered rival (RECT-ck-Bern, FDC v5, B4-bal, the
  nine v5 rigorous rivals) on N80_pen; the same ratios on x12 (interpolated 12th crossing, unpenalised) and on
  N100_pen (15/15, penalised); FDC-MR[front3]/r on the same endpoints; per-method geomean N/tau_R, share at tau_R,
  false streams with Clopper-Pearson 95% UB.
Block B (DESCRIPTIVE, Lenta LR9 eval half, outcome-exposed log, seeds 34000-34199, eps 0.002 / 0.003 / 0.004 all
  reported): FDC-BF/r and FDC-MR[front3]/r for every method, per eps; no verdict.  replica 'fail' -> 'replica_fail'
  (appendix only).
"""
from __future__ import annotations

import math

import numpy as np

from .v6_analysis import B_BOOT, boot_idx, build_matrix, cp_upper, paired

__all__ = ["A_FAMILY", "A_THRESHOLD", "A_METHODS", "B_METHODS", "BLOCK_SPEC", "decide_block_a", "analyse_block",
           "PRIMARY", "SECONDARY_METHOD", "V5_R"]

PRIMARY = "FDC-BF"
SECONDARY_METHOD = "FDC-MR[front3]"
A_FAMILY = ("RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR")
A_THRESHOLD = 0.80
V5_R = ("B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool")
A_METHODS = (PRIMARY, SECONDARY_METHOD, "FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR",
             "B1", "B4", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool")
B_METHODS = (PRIMARY, SECONDARY_METHOD, "FDC", "B4-bal", "QFC-pool", "B2-rect", "Peace-rect", "Hait-SW",
             "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR")
BLOCK_SPEC = {
    "A": {"methods": A_METHODS, "family": A_FAMILY, "seeds": tuple(range(33000, 33200)), "eps": (0.001,),
          "keys": ("N80_pen", "x12", "N100_pen")},
    "B": {"methods": B_METHODS, "family": None, "seeds": tuple(range(34000, 34200)), "eps": (0.002, 0.003, 0.004),
          "keys": ("N80_pen", "x12")},
}


def _false_streams(rows, methods, eps, n):
    out = {}
    for e in eps:
        for m in methods:
            rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - e) < 1e-15]
            k = int(sum(bool(r["fwer_event"]) for r in rr))
            k80 = int(sum(int(r.get("n_false_at_k80") or 0) > 0 for r in rr))
            out[f"{m}@{e}"] = {"false_streams": k, "n": len(rr), "cp_ub95": cp_upper(k, n),
                               "false_streams_by_12of15": k80}
    return out


def _describe(rows, methods, eps, keys):
    out = {}
    for e in eps:
        for m in methods:
            rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - e) < 1e-15]
            tau = rr[0]["tau_R"]
            d = {"n": len(rr), "share_N80_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr]))}
            for k in keys:
                v = np.array([r[k] for r in rr], dtype=float)
                d[f"geomean_{k}_over_tau"] = float(np.exp(np.mean(np.log(v / tau))))
            out[f"{m}@{e}"] = d
    return out


def decide_block_a(per_rival: dict, primary_false_streams: int, replica: dict, family=A_FAMILY,
                   threshold=A_THRESHOLD) -> dict:
    if set(per_rival) != set(family):
        raise ValueError(f"per-rival results {sorted(per_rival)} != registered family {sorted(family)}")
    ub = {}
    for k, v in per_rival.items():
        u = v.get("ub95_one_sided")
        if u is None or not math.isfinite(u) or u <= 0:
            raise ValueError(f"invalid UB for {k}: {u!r}")
        ub[k] = float(u)
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r})")
    if not isinstance(primary_false_streams, int) or primary_false_streams < 0:
        raise ValueError("primary_false_streams must be a non-negative int")
    worst = max(ub, key=ub.get)
    u = ub[worst]
    fails = []
    if u >= threshold:
        fails.append("faster_below_1" if u < 1.0 else "not_faster")
    if primary_false_streams > 0:
        fails.append("validity_failure")
    if st != "pass":
        fails.append("replica_fail")
    return {"verdict": "positive_result_achieved" if not fails else "positive_result_not_achieved",
            "failing_components": fails, "UB_star": u, "governing_rival": worst, "per_rival_ub": ub,
            "primary_false_streams": primary_false_streams, "replica_status": st,
            "rule": f"UB* = max_r in {list(family)} UB95(FDC-BF/r, N80_pen) < {threshold} AND 0 FDC-BF false streams "
                    "AND replica pass (IUT)"}


def analyse_block(block, rows, replica, spec=None, B=B_BOOT):
    sp = dict(BLOCK_SPEC[block] if spec is None else spec)
    mats = {k: build_matrix(rows, sp["methods"], sp["seeds"], sp["eps"], k) for k in sp["keys"]}
    idx = boot_idx(len(sp["seeds"]), B=B)
    out = {"block": block, "keys": list(sp["keys"]), "n_streams": len(sp["seeds"]), "per_eps": {}}
    for e in sp["eps"]:
        pe = {}
        for ref in (PRIMARY, SECONDARY_METHOD):
            for k in sp["keys"]:
                M = mats[k]
                a = M[(ref, float(e))]
                pe[f"{ref}|{k}"] = {m: paired(a, M[(m, float(e))], idx) for m in sp["methods"] if m != ref}
        out["per_eps"][str(e)] = pe
    out["false_streams"] = _false_streams(rows, sp["methods"], sp["eps"], len(sp["seeds"]))
    out["describe"] = _describe(rows, sp["methods"], sp["eps"], sp["keys"])
    if block == "A":
        e0 = str(sp["eps"][0])
        prim = out["per_eps"][e0][f"{PRIMARY}|N80_pen"]
        fs = out["false_streams"][f"{PRIMARY}@{float(sp['eps'][0])}"]["false_streams"]
        out["decision"] = decide_block_a({m: prim[m] for m in sp["family"]}, fs, replica, sp["family"])
    else:
        st = (replica or {}).get("status")
        if st not in ("pass", "fail"):
            raise ValueError(f"replica check not completed (status={st!r})")
        out["decision"] = {"verdict": "descriptive" if st == "pass" else "replica_fail"}
    return out
