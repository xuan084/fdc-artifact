"""Lock-v9 addendum: pre-registered analysis (new file; frozen with the v9 addendum).

Statistics as v6 / v7 / v8 (``v6_analysis.paired`` / ``boot_idx``): per-stream log ratio d_i = log N(primary, i) -
log N(r, i); point estimate exp(mean d); paired percentile bootstrap over streams, B = 10^4, seed 42, the SAME resample
matrix for every comparator and endpoint of a block; one-sided 95% UB = exp(q_0.95); two-sided 95% CI.

Grids.  Blocks A and B are run on two evaluation grids of the same streams:
  '@D'  the 4x-dense monitoring grid (``fdc_loc.dense_grid(ck, 4)``: 3 geometric interpolants per interval of the frozen
        K = 20 checkpoint grid ck; 77 evaluation times on both tables), every method monitored at the SAME times;
  '@K'  the frozen K = 20 checkpoint grid (the grid of every earlier block).
Method names in the analysis carry the grid suffix (rows of a dense task -> name + '@D', of a K20 task -> name + '@K').

PRIMARY = 'TU-FDC@D': FDC-BF code unchanged, widths frozen at the latest of the 20 block points (Theorem TU-1,
time-uniform under the frozen schedule), certified at every dense evaluation time.

Block A (CONFIRMATORY; Criteo CR9 eval half, seeds 37000-37199, eps 0.001, endpoint N80_pen):
  rivals R_A = {HC-WoR@D (frozen v7/v8 CR9 config; time-uniform under any predictable sampling), RECT-ck-BF-TU@D
  (time-uniform matched Bennett rectangle, Lemma TU)}.
  'positive_result_achieved' iff (1) UB95(TU-FDC@D / r) < 0.85 for every r in R_A, (2) TU-FDC@D 0/200 false streams
  (any false certificate up to the end of its run), (3) replica pass for every block-A task (R1 + R2 + R2b and the
  cross-task R2b).  Otherwise 'positive_result_not_achieved' with failing components.
Block B (CONFIRMATORY; X5 RetailHero eval half, seeds 37200-37399, eps 0.02):
  rivals R_B = {HC-WoR@D (frozen X5-tuned v8 config), RECT-ck-BF-TU@D}, threshold 0.50; non-inferiority family
  G_B = {PJC-local@D (frozen X5-tuned v8 config = the no-reset member, monitored at the same 77 times with its own
  K = 77 checkpoint ledger), TU-PJC@D (the same member read under Lemma TU: widths frozen at the latest of the 20 block
  points, K = 20 ledger)}, margin 1.05 for BOTH (IUT); plus 0 false streams and replica pass.
Joint headline ('time-uniform speed claim on both confirmatory real tables') requires BOTH A and B positive (IUT).
Block C (DESCRIPTIVE, Hillstrom 3-arm, outcome-exposed): ratios, CIs, false streams, share at tau_R, width ratio at
  checkpoint index K-2; the pre-registered WORDING rule (not a test): if UB95(FDC-BF/r) < 0.95 for every tuned
  rectangle r (RECT-ck-HG*, HC-WoR*) at the primary eps and FDC-BF has 0 false streams -> 'faster_descriptive', else
  'no_speed_advantage'.
"""
from __future__ import annotations

import math

import numpy as np

from .v6_analysis import B_BOOT, boot_idx, cp_upper, paired
from .v8_analysis import build_matrix_by_eps

__all__ = ["PRIMARY", "DENSE_METHODS", "K20_METHODS", "HEAVY_K20", "CONF_SPEC", "C_SPEC", "decide_confirmatory",
           "decide_joint", "word_block_c", "analyse_confirmatory", "analyse_c", "suffix_rows"]

PRIMARY = "TU-FDC@D"
DENSE_METHODS = ("TU-FDC", "HC-WoR", "RECT-ck-BF-TU", "PJC-local", "TU-PJC", "FDC-BF[K77]", "RECT-ck-HG", "TU-LOC")
K20_METHODS = ("FDC-BF", "HC-WoR", "RECT-ck-HG", "RECT-ck-BF", "RECT-ck-BF+box", "PJC-local", "FDC-MR[front3]")
HEAVY_K20 = ("FDC-HG", "FDC-LOC")
KEYS = ("N80_pen", "x12", "N100_pen")
CONF_SPEC = {
    "A": {"layer": "CR9", "eps": 0.001, "seeds": tuple(range(37000, 37200)),
          "tasks": {"v9a_full_d": ("D", DENSE_METHODS), "v9a_full_k": ("K", K20_METHODS),
                    "v9a_full_x": ("K", HEAVY_K20)},
          "rivals": ("HC-WoR@D", "RECT-ck-BF-TU@D"), "threshold": 0.85, "ni": (), "ni_margin": None},
    "B": {"layer": "X9", "eps": 0.02, "seeds": tuple(range(37200, 37400)),
          "tasks": {"v9b_full_d": ("D", DENSE_METHODS), "v9b_full_k": ("K", K20_METHODS),
                    "v9b_full_x": ("K", HEAVY_K20)},
          "rivals": ("HC-WoR@D", "RECT-ck-BF-TU@D"), "threshold": 0.50, "ni": ("PJC-local@D", "TU-PJC@D"), "ni_margin": 1.05},
}
C_B_METHODS = ("FDC-BF", "RECT-ck-HG[0.2]", "HC-WoR{nstar,0.75,1}[0.2]", "PJC-BF[local,b=4,neyman]", "RECT-ck-BF",
               "RECT-ck-BF+box", "FDC-BF[0.5]", "FDC-BF[ney]", "RECT-ck-HG[bal]")
C_A_METHODS = ("FDC-BF", "RECT-ck-HG[bal]", "HC-WoR{prpl,0.5,-}[bal]", "PJC-BF[local,b=,bal]", "RECT-ck-BF",
               "RECT-ck-BF+box", "PJC-BF[local,b=2,neyman]")
C_SPEC = {
    "v9c_full_b": {"layer": "HCZ6", "design": "B (CZ6 / F2 womens half price / visit)", "eps": (0.01, 0.0125, 0.015),
                   "primary_eps": 0.0125, "seeds": tuple(range(37400, 37600)), "methods": C_B_METHODS,
                   "tuned_rectangles": ("RECT-ck-HG[0.2]", "HC-WoR{nstar,0.75,1}[0.2]"),
                   "pjc": "PJC-BF[local,b=4,neyman]", "role": "main descriptive (block C)"},
    "v9c_full_a": {"layer": "HCR6", "design": "A (CR6 / F3 / visit)", "eps": (0.006, 0.0075, 0.01),
                   "primary_eps": 0.0075, "seeds": tuple(range(37600, 37800)), "methods": C_A_METHODS,
                   "tuned_rectangles": ("RECT-ck-HG[bal]", "HC-WoR{prpl,0.5,-}[bal]"),
                   "pjc": "PJC-BF[local,b=,bal]", "role": "supplement (task too hard for rivals on dev)"},
}
C_WORD_THRESHOLD = 0.95
C_K_DIAG = 18                                   # checkpoint index K-2 (the last checkpoint before tau_R), K = 20


def suffix_rows(rows, grid):
    return [dict(r, method=f"{r['method']}@{grid}") for r in rows]


def decide_confirmatory(block, per_rival: dict, per_ni: dict, primary_false_streams: int, replica: dict,
                        spec: dict | None = None) -> dict:
    sp = CONF_SPEC[block] if spec is None else spec
    R, G, thr, margin = tuple(sp["rivals"]), tuple(sp["ni"]), float(sp["threshold"]), sp["ni_margin"]
    if set(per_rival) != set(R):
        raise ValueError(f"rival results {sorted(per_rival)} != registered {sorted(R)}")
    if set(per_ni) != set(G):
        raise ValueError(f"non-inferiority results {sorted(per_ni)} != registered {sorted(G)}")
    ub = {}
    for k, v in list(per_rival.items()) + list(per_ni.items()):
        u = v.get("ub95_one_sided")
        if u is None or not math.isfinite(u) or u <= 0:
            raise ValueError(f"invalid UB for {k}: {u!r}")
        ub[k] = float(u)
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r})")
    if not isinstance(primary_false_streams, int) or primary_false_streams < 0:
        raise ValueError("primary_false_streams must be a non-negative int")
    wR = max(R, key=lambda m: ub[m])
    fails = []
    if ub[wR] >= thr:
        fails.append("rival_superiority_failed")
    wG = None
    if G:
        wG = max(G, key=lambda m: ub[m])
        if ub[wG] >= margin:
            fails.append("joint_rival_not_inferior_failed")
    if primary_false_streams > 0:
        fails.append("validity_failure")
    if st != "pass":
        fails.append("replica_fail")
    sub = None
    if "rival_superiority_failed" in fails:
        sub = "faster_below_1" if ub[wR] < 1.0 else "not_faster"
    return {"block": block, "verdict": "positive_result_achieved" if not fails else "positive_result_not_achieved",
            "failing_components": fails, "rival_sub_label": sub, "UB_star": ub[wR], "governing_rival": wR,
            "UB_star_NI": ub[wG] if wG else None, "governing_NI_rival": wG, "per_rival_ub": ub,
            "primary_false_streams": primary_false_streams, "replica_status": st,
            "rule": f"max_r in {list(R)} UB95({PRIMARY}/r, N80_pen, eps {sp['eps']}) < {thr}"
                    + (f" AND max_r in {list(G)} UB95 < {margin}" if G else "")
                    + f" AND 0 {PRIMARY} false streams AND replica pass (IUT, all required)"}


def decide_joint(dec_a: dict, dec_b: dict) -> dict:
    both = dec_a["verdict"] == dec_b["verdict"] == "positive_result_achieved"
    return {"verdict": "positive_on_both_confirmatory_tables" if both else "not_positive_on_both",
            "A": dec_a["verdict"], "B": dec_b["verdict"],
            "rule": "the time-uniform speed headline requires blocks A AND B positive (IUT across tables)"}


def _false_streams(rows, methods, eps, n):
    out = {}
    for m in methods:
        rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - eps) < 1e-15]
        k = int(sum(bool(r["fwer_event"]) for r in rr))
        k80 = int(sum(int(r.get("n_false_at_k80") or 0) > 0 for r in rr))
        out[m] = {"false_streams": k, "n": len(rr), "cp_ub95": cp_upper(k, n), "false_streams_by_12of15": k80,
                  "validity": rr[0].get("validity") if rr else None}
    return out


def _describe(rows, methods, eps, keys):
    out = {}
    for m in methods:
        rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - eps) < 1e-15]
        tau = rr[0]["tau_R"]
        d = {"n": len(rr), "share_N80_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])), "tau_R": tau,
             "K_eval": rr[0].get("K_eval"),
             "mean_sec_per_stream": float(np.mean([r["sec"] for r in rr])) if all("sec" in r for r in rr) else None}
        for k in keys:
            if all(r.get(k) is not None for r in rr):
                v = np.array([r[k] for r in rr], dtype=float)
                d[f"geomean_{k}_over_tau"] = float(np.exp(np.mean(np.log(v / tau))))
        out[m] = d
    return out


def analyse_confirmatory(block, rows, replica, spec=None, B=B_BOOT):
    """rows: already grid-suffixed rows of every task of the block."""
    sp = dict(CONF_SPEC[block] if spec is None else spec)
    e = float(sp["eps"])
    methods = tuple(f"{m}@{g}" for _, (g, ms) in sp["tasks"].items() for m in ms)
    mats = {k: build_matrix_by_eps(rows, {e: methods}, sp["seeds"], k) for k in KEYS}
    idx = boot_idx(len(sp["seeds"]), B=B)
    out = {"block": block, "eps": e, "keys": list(KEYS), "n_streams": len(sp["seeds"]), "primary": PRIMARY,
           "comparisons": {}}
    refs = [PRIMARY, "FDC-BF@K"]
    for ref in refs:
        for k in KEYS:
            M = mats[k]
            a = M[(ref, e)]
            out["comparisons"][f"{ref}|{k}"] = {m: paired(a, M[(m, e)], idx) for m in methods if m != ref}
    out["false_streams"] = _false_streams(rows, methods, e, len(sp["seeds"]))
    out["describe"] = _describe(rows, methods, e, KEYS)
    pe = out["comparisons"][f"{PRIMARY}|N80_pen"]
    fs = out["false_streams"][PRIMARY]["false_streams"]
    out["decision"] = decide_confirmatory(block, {m: pe[m] for m in sp["rivals"]}, {m: pe[m] for m in sp["ni"]},
                                          fs, replica, spec=sp)
    out["all_rigorous_false_streams_total"] = int(sum(v["false_streams"] for v in out["false_streams"].values()))
    return out


def word_block_c(per_rect: dict, fdc_false_streams: int, rectangles=None, thr=C_WORD_THRESHOLD) -> dict:
    if rectangles is not None and set(per_rect) != set(rectangles):
        raise ValueError(f"rectangle results {sorted(per_rect)} != registered {sorted(rectangles)}")
    if not per_rect:
        raise ValueError("no rectangle results")
    if not isinstance(fdc_false_streams, int) or fdc_false_streams < 0:
        raise ValueError("fdc_false_streams must be a non-negative int")
    ub = {}
    for k, v in per_rect.items():
        u = v.get("ub95_one_sided")
        if u is None or not math.isfinite(u) or u <= 0:
            raise ValueError(f"invalid UB for {k}: {u!r}")
        ub[k] = float(u)
    ok = all(u < thr for u in ub.values()) and fdc_false_streams == 0
    return {"wording": "faster_descriptive" if ok else "no_speed_advantage", "per_rectangle_ub": ub,
            "fdc_false_streams": fdc_false_streams,
            "rule": f"UB95(FDC-BF/r) < {thr} for every tuned rectangle r AND 0 FDC-BF false streams -> "
                    "'on the 3-arm Hillstrom log, FDC-BF is also faster than the tuned rectangles (descriptive)'; "
                    "else 'no speed advantage on Hillstrom' (+ running-bound ratios). Wording only, no test."}


def _u12(r, k):
    c = r.get("u12_curve") or []
    return c[k] if k < len(c) else float("nan")


def analyse_c(task, rows, replica, spec=None, B=B_BOOT):
    sp = dict(C_SPEC[task] if spec is None else spec)
    methods = tuple(sp["methods"])
    mbe = {float(e): methods for e in sp["eps"]}
    mats = {k: build_matrix_by_eps(rows, mbe, sp["seeds"], k) for k in ("N80_pen", "x12")}
    idx = boot_idx(len(sp["seeds"]), B=B)
    out = {"task": task, "role": sp["role"], "design": sp["design"], "n_streams": len(sp["seeds"]), "per_eps": {}}
    for e in mbe:
        pe = {}
        for k in ("N80_pen", "x12"):
            M = mats[k]
            pe[f"FDC-BF|{k}"] = {m: paired(M[("FDC-BF", e)], M[(m, e)], idx) for m in methods if m != "FDC-BF"}
        # running-certificate-bound diagnostic at checkpoint index K-2 (12th smallest running-min U_q), CONDITIONAL on
        # both trajectories reaching that checkpoint with finite positive bounds; eligible / total counts reported;
        # 'unavailable' when fewer than 10 eligible pairs
        by = {(r["method"], int(r["seed"])): r for r in rows if abs(float(r["eps"]) - e) < 1e-15}
        wd = {}
        for m in methods:
            if m == "FDC-BF":
                continue
            uf = np.array([_u12(by[("FDC-BF", s)], C_K_DIAG) for s in sp["seeds"]])
            ur = np.array([_u12(by[(m, s)], C_K_DIAG) for s in sp["seeds"]])
            ok = np.isfinite(uf) & np.isfinite(ur) & (uf > 0) & (ur > 0)
            if ok.sum() >= 10:
                d = np.log(uf[ok]) - np.log(ur[ok])
                wd[m] = {"ratio": float(np.exp(d.mean())), "eligible": int(ok.sum()), "total": len(sp["seeds"]),
                         "checkpoint_index": C_K_DIAG}
            else:
                wd[m] = {"ratio": None, "eligible": int(ok.sum()), "total": len(sp["seeds"]),
                         "checkpoint_index": C_K_DIAG, "status": "unavailable"}
        pe["running_bound_ratio_k18_FDC-BF_over"] = wd
        pe["false_streams"] = _false_streams(rows, methods, e, len(sp["seeds"]))
        pe["describe"] = _describe(rows, methods, e, ("N80_pen", "x12"))
        out["per_eps"][str(e)] = pe
    pp = out["per_eps"][str(float(sp["primary_eps"]))]
    out["wording"] = word_block_c({m: pp["FDC-BF|N80_pen"][m] for m in sp["tuned_rectangles"]},
                                  pp["false_streams"]["FDC-BF"]["false_streams"], rectangles=sp["tuned_rectangles"])
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r})")
    out["decision"] = {"verdict": "descriptive" if st == "pass" else "replica_fail"}
    return out
