"""Lock-v8 addendum: pre-registered analysis (new file; frozen with the v8 addendum).

Statistics as v6 / v7 (``v6_analysis.paired`` / ``boot_idx``): per-stream log ratio d_i = log N(FDC-BF, i) -
log N(r, i); point estimate exp(mean d); paired percentile bootstrap over streams, B = 10^4, seed 42, the SAME resample
matrix for every comparator and endpoint of a block; one-sided 95% UB = exp(q_0.95); two-sided 95% CI.

Block A (CONFIRMATORY, X5 RetailHero X9 eval half, seeds 35000-35199; decision at eps = 0.02, endpoint N80_pen):
  F = {RECT-ck-HG*, RECT-ck-HG-live, HC-WoR*}  (rectangles; * = read plan / schedule tuned on X5 dev 900-949)
  G = {PJC-local*, PJC-menu*}                    (phased adaptive joint certificates, tuned on X5 dev 900-949)
  'positive_result_achieved' iff
     (1) UB95(FDC-BF / r) < 0.60 for every r in F            (superiority over rectangles, IUT)
     (2) UB95(FDC-BF / r) < 1.05 for every r in G            (non-inferiority, margin 5%, IUT)
     (3) FDC-BF has 0/200 false streams at eps 0.02 (any false certificate up to the end of its run)
     (4) replica (R1 + R2 + R2b) pass for every block-A task.
  Otherwise 'positive_result_not_achieved' with the failing components:
     'rectangle_superiority_failed' (sub-label 'faster_below_1' if max_F UB < 1, else 'not_faster'),
     'joint_adaptive_rival_not_inferior_failed', 'validity_failure', 'replica_fail'.
  Secondary (descriptive, no verdict): G superiority (UB < 1); every other method's FDC-BF/r on N80_pen, x12, N100_pen
  at all three eps (0.015, 0.02, 0.03 where run); FDC-MR[front3]/r; per method geomean N/tau_R, share at tau_R, false
  streams with Clopper-Pearson 95% UB.
Block B (DESCRIPTIVE, CR12 eval half, seeds 35200-35399, eps 0.001, stop 12/15): ratios, false streams, per-checkpoint
  certificate time.  replica 'fail' -> 'replica_fail' (appendix flag only).
Block C (DESCRIPTIVE, POST HOC, CR9 eval half, v7 block-A streams 33000-33199, eps 0.001, stop 15/15): the new v8 rows
  joined with the git-sealed v7 block-A rows of the same streams; FDC-BF re-run rows are compared field by field with
  the sealed v7 FDC-BF rows (reproduction check).
"""
from __future__ import annotations

import math

import numpy as np

from .v6_analysis import B_BOOT, boot_idx, cp_upper, paired

__all__ = ["PRIMARY", "SECONDARY_METHOD", "F_FAMILY", "G_FAMILY", "F_THRESHOLD", "G_MARGIN", "DECISION_EPS",
           "A_EPS", "A_CORE", "A_DESCR", "B_METHODS", "C_NEW", "BLOCK_SPEC", "decide_block_a", "analyse_block",
           "build_matrix_by_eps", "per_ckpt_time"]

PRIMARY = "FDC-BF"
SECONDARY_METHOD = "FDC-MR[front3]"
F_FAMILY = ("RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR")
G_FAMILY = ("PJC-local", "PJC-menu")
F_THRESHOLD = 0.60
G_MARGIN = 1.05
DECISION_EPS = 0.02
A_EPS = (0.015, 0.02, 0.03)
A_CORE = (PRIMARY, SECONDARY_METHOD, "FDC", "RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR", "PJC-local", "PJC-menu",
          "RECT-ck-BF", "RECT-ck-BF+box")
PLUGINS = ("B2-fav", "B3-fav", "Peace-fav", "B5", "B2-fav-tight", "FIX-bal-fav", "Peace-fav-bal")
LIT_RIGOROUS = ("B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool")
A_DESCR = PLUGINS + LIT_RIGOROUS
B_METHODS = (PRIMARY, SECONDARY_METHOD, "FDC", "RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR", "RECT-ck-BF+box",
             "PJC-local", "PJC-menu")
C_NEW = (PRIMARY, "PJC-local", "PJC-menu", "RECT-ck-BF", "RECT-ck-BF+box", "RECT-ck-HG[ney]", "HC-WoR[ney]") + PLUGINS
C_V7 = ("FDC-BF", "FDC-MR[front3]", "FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR",
        "B1", "B4", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool")
BLOCK_SPEC = {
    "A": {"methods_by_eps": {0.015: A_CORE, 0.02: A_CORE + A_DESCR, 0.03: A_CORE},
          "seeds": tuple(range(35000, 35200)), "keys": ("N80_pen", "x12", "N100_pen")},
    "B": {"methods_by_eps": {0.001: B_METHODS}, "seeds": tuple(range(35200, 35400)), "keys": ("N80_pen", "x12")},
    # block C: v8 rows use the method name with suffix '@v8' for the FDC-BF re-run; v7 rows keep their names
    "C": {"methods_by_eps": {0.001: tuple(m if m != PRIMARY else "FDC-BF@v8" for m in C_NEW) + C_V7},
          "seeds": tuple(range(33000, 33200)), "keys": ("N80_pen", "x12", "N100_pen")},
}


def build_matrix_by_eps(rows, methods_by_eps, seeds, key):
    """{(method, eps): array over seeds}.  Refuses missing / extra / duplicate rows and invalid endpoints."""
    seeds = list(seeds)
    sset = set(seeds)
    want = {(m, float(e)) for e, ms in methods_by_eps.items() for m in ms}
    got = {}
    for r in rows:
        k = (r["method"], float(r["eps"]))
        if k not in want:
            raise ValueError(f"unregistered (method, eps) in rows: {k}")
        if int(r["seed"]) not in sset:
            raise ValueError(f"unregistered seed {r['seed']} for {k}")
        d = got.setdefault(k, {})
        if int(r["seed"]) in d:
            raise ValueError(f"duplicate row {k} seed {r['seed']}")
        v = r.get(key)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"invalid endpoint {key}={v!r} for {k} seed {r['seed']}")
        d[int(r["seed"])] = float(v)
    for k in sorted(want):
        if k not in got or len(got[k]) != len(seeds):
            raise ValueError(f"incomplete: {k} has {len(got.get(k, {}))}/{len(seeds)} seeds")
    return {k: np.array([got[k][s] for s in seeds]) for k in want}


def per_ckpt_time(rows, methods_by_eps):
    """Mean certificate seconds per visited checkpoint (rs_sec_cert / n_checkpoints_visited), descriptive."""
    out = {}
    for e, ms in methods_by_eps.items():
        for m in ms:
            rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - e) < 1e-15]
            v = [r["rs_sec_cert"] / max(1, r["n_checkpoints_visited"]) for r in rr
                 if r.get("rs_sec_cert") is not None and r.get("n_checkpoints_visited")]
            if v:
                out[f"{m}@{e}"] = {"mean_sec_per_checkpoint": float(np.mean(v)),
                                   "mean_sec_per_stream": float(np.mean([r["sec"] for r in rr]))}
    return out


def _false_streams(rows, methods_by_eps, n):
    out = {}
    for e, ms in methods_by_eps.items():
        for m in ms:
            rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - e) < 1e-15]
            k = int(sum(bool(r["fwer_event"]) for r in rr))
            k80 = int(sum(int(r.get("n_false_at_k80") or 0) > 0 for r in rr))
            out[f"{m}@{e}"] = {"false_streams": k, "n": len(rr), "cp_ub95": cp_upper(k, n),
                               "false_streams_by_12of15": k80, "validity": rr[0].get("validity") if rr else None}
    return out


def _describe(rows, methods_by_eps, keys):
    out = {}
    for e, ms in methods_by_eps.items():
        for m in ms:
            rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - e) < 1e-15]
            tau = rr[0]["tau_R"]
            d = {"n": len(rr), "share_N80_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])), "tau_R": tau}
            for k in keys:
                v = np.array([r[k] for r in rr], dtype=float)
                d[f"geomean_{k}_over_tau"] = float(np.exp(np.mean(np.log(v / tau))))
            out[f"{m}@{e}"] = d
    return out


def decide_block_a(per_F: dict, per_G: dict, primary_false_streams: int, replica: dict, F=F_FAMILY, G=G_FAMILY,
                   f_thr=F_THRESHOLD, g_margin=G_MARGIN) -> dict:
    if set(per_F) != set(F):
        raise ValueError(f"F results {sorted(per_F)} != registered {sorted(F)}")
    if set(per_G) != set(G):
        raise ValueError(f"G results {sorted(per_G)} != registered {sorted(G)}")
    ub = {}
    for k, v in list(per_F.items()) + list(per_G.items()):
        u = v.get("ub95_one_sided")
        if u is None or not math.isfinite(u) or u <= 0:
            raise ValueError(f"invalid UB for {k}: {u!r}")
        ub[k] = float(u)
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r})")
    if not isinstance(primary_false_streams, int) or primary_false_streams < 0:
        raise ValueError("primary_false_streams must be a non-negative int")
    wF = max(F, key=lambda m: ub[m])
    wG = max(G, key=lambda m: ub[m])
    fails = []
    if ub[wF] >= f_thr:
        fails.append("rectangle_superiority_failed")
    if ub[wG] >= g_margin:
        fails.append("joint_adaptive_rival_not_inferior_failed")
    if primary_false_streams > 0:
        fails.append("validity_failure")
    if st != "pass":
        fails.append("replica_fail")
    sub = None
    if "rectangle_superiority_failed" in fails:
        sub = "faster_below_1" if ub[wF] < 1.0 else "not_faster"
    return {"verdict": "positive_result_achieved" if not fails else "positive_result_not_achieved",
            "failing_components": fails, "rectangle_sub_label": sub,
            "UB_star_F": ub[wF], "governing_rival_F": wF, "UB_star_G": ub[wG], "governing_rival_G": wG,
            "per_rival_ub": ub, "G_superiority_descriptive": {m: ub[m] < 1.0 for m in G},
            "primary_false_streams": primary_false_streams, "replica_status": st,
            "rule": f"max_r in {list(F)} UB95(FDC-BF/r, N80_pen, eps {DECISION_EPS}) < {f_thr} AND max_r in {list(G)} "
                    f"UB95(FDC-BF/r) < {g_margin} AND 0 FDC-BF false streams AND replica pass (IUT, all required)"}


def analyse_block(block, rows, replica, spec=None, B=B_BOOT):
    sp = dict(BLOCK_SPEC[block] if spec is None else spec)
    mbe = {float(e): tuple(ms) for e, ms in sp["methods_by_eps"].items()}
    keys = [k for k in sp["keys"]]
    mats = {k: build_matrix_by_eps(rows, mbe, sp["seeds"], k) for k in keys}
    idx = boot_idx(len(sp["seeds"]), B=B)
    prim = "FDC-BF@v8" if block == "C" else PRIMARY
    out = {"block": block, "keys": keys, "n_streams": len(sp["seeds"]), "per_eps": {}}
    for e, ms in mbe.items():
        pe = {}
        refs = [prim] + ([SECONDARY_METHOD] if SECONDARY_METHOD in ms else [])
        if block == "C":
            refs = [prim] + (["FDC-BF"] if "FDC-BF" in ms else [])   # sealed v7 FDC-BF rows: second reference
        for ref in refs:
            for k in keys:
                M = mats[k]
                a = M[(ref, e)]
                pe[f"{ref}|{k}"] = {m: paired(a, M[(m, e)], idx) for m in ms if m != ref}
        out["per_eps"][str(e)] = pe
    out["false_streams"] = _false_streams(rows, mbe, len(sp["seeds"]))
    out["describe"] = _describe(rows, mbe, keys)
    out["per_checkpoint_time"] = per_ckpt_time(rows, mbe)
    if block == "A":
        pe = out["per_eps"][str(DECISION_EPS)][f"{PRIMARY}|N80_pen"]
        fs = out["false_streams"][f"{PRIMARY}@{DECISION_EPS}"]["false_streams"]
        out["decision"] = decide_block_a({m: pe[m] for m in F_FAMILY}, {m: pe[m] for m in G_FAMILY}, fs, replica)
    else:
        st = (replica or {}).get("status")
        if st not in ("pass", "fail"):
            raise ValueError(f"replica check not completed (status={st!r})")
        out["decision"] = {"verdict": "descriptive" if st == "pass" else "replica_fail"}
    return out
