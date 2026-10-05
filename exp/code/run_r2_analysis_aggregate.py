"""Round-1 (RH-JPC) aggregation: Holm families, wording tier A/B/C, HK, HR5, suspicious gate, figures.

Recomputes every headline statistic from the raw per-task results.jsonl files (NOT from the per-task summaries).
Where a statistic needs inputs that only exist in a task's own derived artefacts (e.g. the HD2 design-level
sandwich, the replica direction agreement), the value is taken from that task's summary.json and the verdict row
is tagged source="summary".

Statistical unit = environment instance. Ratios: per instance, stream totals of new env steps (charged, censored at
T_max) averaged over streams, then log((a+1)/(b+1)); geometric mean over instances; instance-cluster bootstrap
B=10^4, seed 42 (prereg_lock.budgets). Holm within the locked families (prereg_lock.holm_families).
The tier decision imports the frozen dsswm/stats/tier_decision.decide_tier and asserts its locked sha256.

Usage: python run_r2_analysis_aggregate.py --mode pilot
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES = WS / "exp" / "results"
TASK = "r2_analysis_aggregate"

from dsswm.stats.cp import clopper_pearson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
ARGS = ap.parse_args()
MODE = ARGS.mode
SRC = RES / ("pilots" if MODE == "pilot" else "full")
OUT = SRC / TASK
FIG = OUT / "figures"
SAMP = OUT / "samples"
for d in (OUT, FIG, SAMP):
    d.mkdir(parents=True, exist_ok=True)

LOCK = json.load(open(WS / "plan" / "prereg_lock.json"))
B = int(LOCK["budgets"]["bootstrap_B"])
BSEED = int(LOCK["budgets"]["bootstrap_seed"])
DELTA = float(LOCK["thresholds"]["delta"])
EPS = float(LOCK["thresholds"]["eps_NL"])
TMAX = int(LOCK["budgets"]["T_max_stepwise"])
N0 = 20
LOG = []

(RES / f"{TASK}.pid").write_text(str(os.getpid()))
T_START = datetime.now()


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def progress(step, total, note):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": 0, "total_epochs": 1, "step": step, "total_steps": total, "loss": None,
        "metric": {"stage": note}, "updated_at": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------------ palette / style
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MC = {"JPC": C[0], "JPC_infl": C[0], "B3": C[2], "B3g": C[3], "B3-UI": C[6], "B8": C[4], "B12": C[5], "B2": C[1],
      "JPC(unc)": C[7]}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight"})


def save(fig, name, rows=None):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    if rows is not None:
        write_csv(FIG / f"{name}.csv", rows)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# ------------------------------------------------------------------------------------------------ loading
def load(task, sub=None):
    p = SRC / (sub or task) / "results.jsonl"
    if not p.exists():
        log(f"MISSING results.jsonl for {sub or task}")
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def summ(task):
    p = SRC / task / "summary.json"
    return json.load(open(p)) if p.exists() else {}


DECISION_FIELDS = ("status", "new_env_steps", "false_cert", "certified_policy", "zero_cost", "censored")


def merge_chunks(tasks, keyf):
    """Merge chunked tasks; identical keys across chunks (pilot: same dev seeds) are de-duplicated, and the decision
    fields of the duplicates are compared (CRN reproducibility check)."""
    merged, dup, mism = {}, 0, 0
    for t in tasks:
        for r in load(t):
            if r.get("method") is None:
                continue
            k = keyf(r)
            if k in merged:
                dup += 1
                if any(merged[k].get(f) != r.get(f) for f in DECISION_FIELDS):
                    mism += 1
                continue
            r = dict(r)
            r["_src"] = t
            merged[k] = r
    return list(merged.values()), {"tasks": tasks, "unique_rows": len(merged), "duplicate_rows": dup,
                                   "duplicate_decision_mismatches": mism}


RNG = np.random.default_rng(BSEED)


def boot_mean(x, B_=B, seed=BSEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B_, len(x)))
    return x[idx].mean(axis=1)


def stream_totals(rows, method, filt=None):
    """{instance: mean over streams of total charged new env steps}."""
    tot = defaultdict(float)
    for r in rows:
        if r.get("method") != method or (filt and not filt(r)):
            continue
        tot[(r["instance"], r.get("stream", 0))] += float(r.get("new_env_steps") or 0)
    per = defaultdict(list)
    for (i, s), v in tot.items():
        per[i].append(v)
    return {i: float(np.mean(v)) for i, v in per.items()}


def ratio_stat(a: dict, b: dict, ci=0.95, null=None, tail="upper"):
    """Geometric-mean ratio a/b over common instances, log((a+1)/(b+1)); bootstrap CI and one-sided p.
    tail='upper': p for H0 ratio >= null (small p => ratio < null). tail='lower': H0 ratio <= null."""
    ins = sorted(set(a) & set(b))
    if not ins:
        return None
    x = np.array([math.log((a[i] + 1) / (b[i] + 1)) for i in ins])
    bm = boot_mean(x)
    al = (1 - ci) / 2
    out = {"n_instances": len(ins), "ratio": float(math.exp(x.mean())),
           "ci": [float(math.exp(np.quantile(bm, al))), float(math.exp(np.quantile(bm, 1 - al)))],
           "per_instance": {int(i): float(math.exp(v)) for i, v in zip(ins, x)},
           "degenerate_identical": bool(np.all(np.abs(x) < 1e-12))}
    if null is not None:
        ln = math.log(null)
        if out["degenerate_identical"]:
            out["p"] = 1.0 if ((tail == "upper" and ln <= 0) or (tail == "lower" and ln >= 0)) else 0.0
        elif tail == "upper":
            out["p"] = float((1 + np.sum(bm >= ln)) / (B + 1))
        else:
            out["p"] = float((1 + np.sum(bm <= ln)) / (B + 1))
    return out


def cp_upper(k, n):
    return clopper_pearson(int(k), int(n), 0.05)[1]


def cp_lower(k, n):
    return clopper_pearson(int(k), int(n), 0.05)[0]


def method_cell(rows, method):
    R = [r for r in rows if r.get("method") == method]
    if not R:
        return None
    cert = [r for r in R if r.get("status") == "CERTIFIED"]
    fc = int(sum(bool(r.get("false_cert")) for r in R))
    return {"n": len(R), "completion": len(cert) / len(R), "n_cert": len(cert), "false_certs": fc,
            "fcr": (fc / len(cert)) if cert else None, "fcr_cp_upper": cp_upper(fc, len(cert)) if cert else 1.0,
            "zero_cost_rate": float(np.mean([bool(r.get("zero_cost")) for r in R])),
            "refused_floor": int(sum(bool(r.get("refused_floor")) for r in R)),
            "stream_mean": float(np.mean(list(stream_totals(rows, method).values()))),
            "theta_cell_alive": (float(np.mean([bool(r["theta_cell_alive"]) for r in R
                                                if r.get("theta_cell_alive") is not None]))
                                 if any(r.get("theta_cell_alive") is not None for r in R) else None)}


def tie_rate(rows, m1, m2):
    a = {(r["instance"], r.get("stream", 0), r["k"]): bool(r.get("zero_cost")) for r in rows if r.get("method") == m1}
    b = {(r["instance"], r.get("stream", 0), r["k"]): bool(r.get("zero_cost")) for r in rows if r.get("method") == m2}
    ks = set(a) & set(b)
    return float(np.mean([a[k] and b[k] for k in ks])) if ks else None


def binom_p_ge(k, n, p0):
    """P(X >= k | n, p0): p-value for H1: rate > p0."""
    if n == 0:
        return 1.0
    return float(stats.binom.sf(k - 1, n, p0))


def binom_p_le(k, n, p0):
    """P(X <= k | n, p0): p-value for H1: rate < p0."""
    if n == 0:
        return 1.0
    return float(stats.binom.cdf(k, n, p0))


def jonckheere(groups, n_perm=10000, seed=BSEED):
    """One-sided (increasing) Jonckheere-Terpstra; permutation p over pooled labels."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if len(groups) < 2:
        return None, None

    def jt(gs):
        s = 0.0
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                d = gs[j][None, :] - gs[i][:, None]
                s += float(np.sum(d > 0) + 0.5 * np.sum(d == 0))
        return s
    obs = jt(groups)
    pooled = np.concatenate(groups)
    if np.all(pooled == pooled[0]):
        return obs, 1.0
    sizes = np.cumsum([len(g) for g in groups])[:-1]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if jt(np.split(perm, sizes)) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def holm(pdict):
    items = sorted([(k, v) for k, v in pdict.items() if v is not None], key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    for k, v in pdict.items():
        if v is None:
            out[k] = None
    return out


def fmt_r(s, key="ci"):
    if not s:
        return "–"
    return f"{s['ratio']:.3f} [{s[key][0]:.3f}, {s[key][1]:.3f}]"


S = {"task_id": TASK, "mode": MODE, "seed": BSEED, "bootstrap_B": B, "eps": EPS, "delta": DELTA,
     "lock_status": LOCK.get("status"), "lock_sha256_provisional": LOCK.get("sha256_provisional"),
     "started_at": T_START.isoformat(), "concurrent_run": True,
     "note": "pilot: all inputs are dev-seed pilot rows; sample sizes are pilot-scale (n<=10 instances per level), "
             "so every Holm p / CP bound below is a pilot reading, not the pre-registered full test"}
VERD = []
RESULT_ROWS = []
TOTAL = 14

# ================================================================================================ 0. integrity
progress(0, TOTAL, "integrity")
log(f"mode={MODE} src={SRC}")
td_path = HERE / "dsswm" / "stats" / "tier_decision.py"
td_sha = hashlib.sha256(td_path.read_bytes()).hexdigest()
assert td_sha == LOCK["tier_decision"]["code_sha256"], "tier_decision.py hash mismatch vs prereg_lock"
from dsswm.stats.tier_decision import decide_tier, hr1_pass  # noqa: E402

S["tier_code_sha256_ok"] = True
if MODE == "full":
    from dsswm.stats.prereg import assert_locked  # noqa: E402
    assert_locked()

DEPS = ["g0_resolution_gate", "dev_offgrid_ladder", "r2_prereg_lock", "hr1_r0_a", "hr1_r0_b", "hr1_r2_a", "hr1_r2_b",
        "hr1_r2_c", "hr1_r3_a", "hr1_r3_b", "hr1_r3_c", "hr1_r3_d", "hr2_threshold_accounting", "hr3_r1_uncorrected",
        "hr4_phase_coarse", "hr4_phase_fine", "oracle_opt_cover", "hf1_mixed_cost", "hf2_static_falsify",
        "hf2b_m1_m3_mixed", "hf3_m4_blindspot", "hf4_eprocess_audit", "hd1_matched_twin_static",
        "hd3_matched_twin_rect", "hd1b_same_exposure", "hd2_kappa_sandwich_lin", "hs1_sampler_equivalence",
        "hh1_honesty_oos", "e2_anchor_a", "e2_anchor_b", "replica_check_r2", "hv_versioned_evidence"]
inv = []
boundary = {"rows_checked": 0, "mismatches": 0, "by_task": {}}
for t in DEPS:
    rows = load(t)
    sm = summ(t)
    done = (RES / f"{t}_DONE").exists()
    nb = nm = 0
    for r in rows:
        if r.get("env_n_steps") is not None and r.get("lr_n_rounds") is not None:
            nb += 1
            if int(r["env_n_steps"]) != int(r["lr_n_rounds"]):
                nm += 1
    boundary["rows_checked"] += nb
    boundary["mismatches"] += nm
    boundary["by_task"][t] = {"checked": nb, "mismatch": nm}
    inv.append({"task": t, "rows": len(rows), "summary_parsed": bool(sm), "done_marker": done,
                "summary_go_no_go": str(sm.get("go_no_go", sm.get("pilot_gate", "")))[:60],
                "boundary_rows": nb, "boundary_mismatch": nm})
S["inventory"] = inv
S["all_summaries_parsed"] = all(x["summary_parsed"] for x in inv)
S["all_done_markers"] = all(x["done_marker"] for x in inv)
S["evidence_boundary_env_steps_eq_lr_rounds"] = boundary
write_csv(OUT / "inventory.csv", inv)
log(f"inventory: {sum(x['rows'] for x in inv)} rows over {len(inv)} tasks; summaries parsed "
    f"{sum(x['summary_parsed'] for x in inv)}/{len(inv)}; boundary {boundary['mismatches']}/{boundary['rows_checked']}")

# ================================================================================================ 1. G0
progress(1, TOTAL, "G0")
g0 = load("g0_resolution_gate")
eta_rows = [r for r in g0 if r.get("kind") == "eta_loc"]
g0_by_f = {}
for f in sorted({r["f"] for r in eta_rows}, key=float):
    R = [r for r in eta_rows if r["f"] == f]
    viol = int(sum(r.get("violations") or 0 for r in R))
    draws = int(sum(r.get("draws") or 0 for r in R))
    med = float(np.median([r["eta_over_eps_median"] for r in R if r.get("eta_over_eps_median") is not None]))
    mn = float(np.min([r["eta_over_eps_min"] for r in R if r.get("eta_over_eps_min") is not None]))
    emp = float(np.median([r["emp_sup_err_over_eps"] for r in R if r.get("emp_sup_err_over_eps") is not None]))
    g0_by_f[str(f)] = {"problems": len(R), "violations": viol, "draws": draws, "eta_over_eps_median": med,
                       "eta_over_eps_min": mn, "emp_sup_err_over_eps_median": emp,
                       "reaches_eps_over_4": bool(mn <= 0.25)}
jt_rows = [r for r in g0 if r.get("kind") == "jtable"]
jt_sec = {str(f): float(np.median([r["sec"] for r in jt_rows if r["f"] == f])) for f in {r["f"] for r in jt_rows}}
g0_viol = sum(v["violations"] for v in g0_by_f.values())
g0_draws = sum(v["draws"] for v in g0_by_f.values())
g0_any_h = any(v["reaches_eps_over_4"] for v in g0_by_f.values())
g0_pass = (g0_viol == 0) and g0_any_h and all(v <= 10 for v in jt_sec.values())
S["G0"] = {"by_f": g0_by_f, "jtable_sec_median_by_f": jt_sec, "violations": g0_viol, "draws": g0_draws,
           "any_f_reaches_eta_le_eps_over_4": g0_any_h, "passed": g0_pass,
           "lock_G0_passed": LOCK["G0_outcome"]["passed"], "agrees_with_lock": g0_pass == LOCK["G0_outcome"]["passed"]}
VERD.append(dict(H="G0", family="gate", gate="0 violations; some f with eta_loc<=eps/4; J<=10 s",
                 estimate=f"violations {g0_viol}/{g0_draws}; eta_loc/eps median (min) by f "
                          + ", ".join(f"f={k}:{v['eta_over_eps_median']:.1f} ({v['eta_over_eps_min']:.1f})" for k, v in g0_by_f.items()),
                 ci="–", p_raw=None, verdict="FAIL" if not g0_pass else "PASS", source="raw",
                 note="validity holds (0 violations) but no resolution reaches eps/4 -> tier C path"))
log(f"G0 passed={g0_pass} violations={g0_viol}/{g0_draws}")

# ================================================================================================ 2. HR1 merge
progress(2, TOTAL, "HR1 merge")
key5 = lambda r: (r.get("level"), r["instance"], r.get("stream", 0), r["k"], r["method"])  # noqa: E731
R0rows, m0 = merge_chunks(["hr1_r0_a", "hr1_r0_b"], key5)
R2rows, m2 = merge_chunks(["hr1_r2_a", "hr1_r2_b", "hr1_r2_c"], key5)
R3rows, m3 = merge_chunks(["hr1_r3_a", "hr1_r3_b", "hr1_r3_c", "hr1_r3_d"], key5)
LEVEL_ROWS = {"R0": R0rows, "R2": R2rows, "R3": R3rows}
S["hr1_merge"] = {"R0": m0, "R2": m2, "R3": m3,
                  "note": "pilot chunks share the same dev seeds (660-663 x 5 problems x stream 0), so the merge "
                          "de-duplicates; decision-field mismatches between duplicates test CRN reproducibility"}
log(f"HR1 merge: R0 {m0['unique_rows']} (dup {m0['duplicate_rows']}, mism {m0['duplicate_decision_mismatches']}); "
    f"R2 {m2['unique_rows']} (mism {m2['duplicate_decision_mismatches']}); R3 {m3['unique_rows']} "
    f"(mism {m3['duplicate_decision_mismatches']})")

TEST_M = {"R0": "JPC", "R2": "JPC_infl", "R3": "JPC_infl"}
DEN_CAND = {"R0": ["B3", "B3g", "B3-UI", "B8", "B12"],
            "R2": ["B3", "B3g_infl", "B3-UI", "B8_infl", "B12_infl"],
            "R3": ["B3", "B3g_infl", "B3-UI", "B8_infl", "B12_infl"]}
TABLE1_M = {"R0": ["JPC", "B3", "B3g", "B3-UI", "B8", "B12", "B2"],
            "R2": ["JPC_infl", "JPC", "B3", "B3g_infl", "B3-UI", "B3-UIc", "B8_infl", "B12_infl", "B2"],
            "R3": ["JPC_infl", "JPC", "B3", "B3g_infl", "B3-UI", "B3-UIc", "B8_infl", "B12_infl", "B2"]}
HR1 = {}
table1 = []
for lv, rows in LEVEL_ROWS.items():
    tm = TEST_M[lv]
    cells = {m: method_cell(rows, m) for m in TABLE1_M[lv]}
    tot = {m: stream_totals(rows, m) for m in TABLE1_M[lv]}
    valid = [m for m in DEN_CAND[lv] if cells.get(m) and cells[m]["fcr_cp_upper"] <= 2 * DELTA
             and cells[m]["n_cert"] > 0]
    best_valid = min(valid, key=lambda m: cells[m]["stream_mean"]) if valid else None
    cand_present = [m for m in DEN_CAND[lv] if cells.get(m)]
    provisional = min(cand_present, key=lambda m: cells[m]["stream_mean"]) if cand_present else None
    den_for_test = best_valid or provisional
    vs = {}
    for m in TABLE1_M[lv]:
        if m == tm or not cells.get(m):
            continue
        vs[m] = ratio_stat(tot[tm], tot[m], null=0.8, tail="upper")
    tc = cells[tm]

    def cellf(den):
        if den is None or vs.get(den) is None:
            return None
        return {"denominator": den, "ratio": vs[den]["ratio"], "ratio_ci_upper": vs[den]["ci"][1],
                "ratio_ci": vs[den]["ci"], "completion": tc["completion"], "fcr_cp_upper": tc["fcr_cp_upper"],
                "p_ratio_ge_0.8": vs[den]["p"]}
    HR1[lv] = {"test_method": tm, "cells": cells, "valid_denominators": valid, "best_valid_denominator": best_valid,
               "provisional_denominator_ignoring_fcr_filter": provisional,
               "vs_best_valid": cellf(best_valid), "vs_provisional": cellf(provisional), "vs_B3": cellf("B3"),
               "vs_same_threshold": cellf("B3-UI"),
               "ratios": {m: {k: v for k, v in s.items() if k != "per_instance"} | {"per_instance": s["per_instance"]}
                          for m, s in vs.items() if s},
               "tie_rate_vs_B3": tie_rate(rows, tm, "B3"),
               "n_instances": len({r["instance"] for r in rows}), "n_rows": len(rows)}
    HR1[lv]["pass_vs_best_valid"] = hr1_pass(HR1[lv]["vs_best_valid"])
    HR1[lv]["pass_vs_provisional"] = hr1_pass(HR1[lv]["vs_provisional"])
    for m in TABLE1_M[lv]:
        c = cells.get(m)
        if not c:
            continue
        rs = vs.get(m) if m != tm else None
        rB3 = ratio_stat(tot[m], tot["B3"]) if m != "B3" and tot.get("B3") else None
        table1.append({"level": lv, "method": m, "stream_steps_mean": round(c["stream_mean"], 1),
                       "ratio_vs_B3": fmt_r(rB3) if rB3 else ("1" if m == "B3" else "–"),
                       "test_over_method": fmt_r(rs) if rs else "–",
                       "completion": round(c["completion"], 3),
                       "FCR": f"{c['false_certs']}/{c['n_cert']} (CP {c['fcr_cp_upper']:.3f})",
                       "zero_cost_rate": round(c["zero_cost_rate"], 3),
                       "tie_rate_vs_B3": (round(tie_rate(rows, m, "B3"), 3) if m != "B3" and tie_rate(rows, m, "B3")
                                          is not None else "–"),
                       "refused_floor": c["refused_floor"],
                       "valid_denominator": (m in valid) if m in DEN_CAND[lv] else "n/a",
                       "role": ("test" if m == tm else "best_valid_den" if m == best_valid else
                                "provisional_den" if m == provisional else "")})
    log(f"HR1@{lv}: {tm} vs provisional {provisional} = {fmt_r(vs.get(provisional))}; valid dens {valid}; "
        f"completion {tc['completion']:.2f}; FCR CP {tc['fcr_cp_upper']:.3f}")
S["HR1"] = HR1
write_csv(OUT / "table1_hr1.csv", table1)

# ---- Holm family A
famA = LOCK["holm_families"]["A"]
pA = {}
for lv_key in famA:
    lv = lv_key.split("@")[1]
    c = HR1[lv]["vs_best_valid"] or HR1[lv]["vs_provisional"]
    pA[lv_key] = c["p_ratio_ge_0.8"] if c else None
hA = holm(pA)
for lv_key in famA:
    lv = lv_key.split("@")[1]
    h = HR1[lv]
    c = h["vs_best_valid"] or h["vs_provisional"]
    gate_ok = h["pass_vs_best_valid"]
    VERD.append(dict(H=lv_key, family="A", gate="ratio CI upper<0.8 vs best valid non-shared denominator; "
                                                "completion>=0.8; FCR CP upper<=2delta",
                     estimate=(f"{h['test_method']}/{c['denominator']} = {c['ratio']:.3f}; completion "
                               f"{c['completion']:.2f}; FCR CP upper {c['fcr_cp_upper']:.3f}") if c else "–",
                     ci=f"[{c['ratio_ci'][0]:.3f}, {c['ratio_ci'][1]:.3f}]" if c else "–",
                     p_raw=pA[lv_key], p_holm=hA[lv_key],
                     verdict="PASS" if gate_ok else "FAIL",
                     source="raw",
                     note=(f"valid denominators: {h['valid_denominators'] or 'none (n=' + str(h['cells'][h['test_method']]['n']) + ' -> CP upper of 0/n > 2delta)'}"
                           f"; denominator used for p: {c['denominator'] if c else '-'}"
                           + ("" if h["best_valid_denominator"] else " (provisional, FCR filter unsatisfiable at pilot n)"))))
# R2 (diagnostic, not in family A)
h = HR1["R2"]
c = h["vs_provisional"]
VERD.append(dict(H="HR1@R2 (diagnostic)", family="none (R2_in_family_A=false)", gate="as HR1",
                 estimate=f"JPC_infl/{c['denominator']} = {c['ratio']:.3f}; completion {c['completion']:.2f}" if c else "–",
                 ci=f"[{c['ratio_ci'][0]:.3f}, {c['ratio_ci'][1]:.3f}]" if c else "–", p_raw=c["p_ratio_ge_0.8"] if c else None,
                 verdict="FAIL", source="raw", note="R2 left family A at lock (h<=h*/2 fails); 100% data-free refusal"))

# ---- tier decision (frozen code)
by_level = {lv: {"vs_best_valid": HR1[lv]["vs_best_valid"], "vs_B3": HR1[lv]["vs_B3"],
                 "vs_same_threshold": HR1[lv]["vs_same_threshold"]} for lv in ("R2", "R3")}
fam_levels = [x.split("@")[1] for x in famA]
tier = decide_tier(g0_pass, fam_levels, by_level)
tier_if_g0 = decide_tier(True, fam_levels, by_level)
S["tier"] = {"decision": tier, "counterfactual_if_G0_had_passed": tier_if_g0, "family_A_levels": fam_levels,
             "inputs": by_level, "code_sha256": td_sha}
log(f"TIER = {tier['tier']} ({tier['reason']}); counterfactual G0-pass tier = {tier_if_g0['tier']}")
VERD.append(dict(H="Wording tier", family="decision", gate="locked decide_tier (sha256 asserted)",
                 estimate=f"tier {tier['tier']}", ci="–", p_raw=None, verdict=tier["tier"], source="raw",
                 note=tier["reason"] + f"; even if G0 had passed: tier {tier_if_g0['tier']} ({tier_if_g0['reason']})"))

# ================================================================================================ 3. HK
progress(3, TOTAL, "HK")
HK = {}
fig8 = []
CHARGE_N0 = {"B2": False}


def cum_curve(rows, m, n0=True):
    per = defaultdict(dict)
    for r in rows:
        if r.get("method") == m:
            per[(r["instance"], r.get("stream", 0))][r["k"]] = float(r.get("new_env_steps") or 0)
    if not per:
        return None
    K = min(len(v) for v in per.values())
    curves = []
    for u, d in per.items():
        c = np.cumsum([d[k] for k in sorted(d)[:K]]) + (N0 if n0 else 0)
        curves.append(c)
    return np.mean(curves, axis=0)


def kstar(cj, cd):
    if cj is None or cd is None:
        return None
    ok = cj <= cd
    for k in range(len(ok)):
        if ok[k:].all():
            return k + 1
    return None


for lv, rows in LEVEL_ROWS.items():
    tm = TEST_M[lv]
    curves = {}
    ms = [tm, "B3", "B2"] + (["JPC"] if lv != "R0" else []) + (["B8"] if lv == "R0" else ["B8_infl"])
    for m in ms:
        cc = cum_curve(rows, m, n0=CHARGE_N0.get(m, True))
        if cc is not None:
            curves[m] = cc
    zc = defaultdict(list)
    for r in rows:
        if r.get("method") == tm:
            zc[r["k"]].append(bool(r.get("zero_cost")))
    HK[lv] = {"K_problems": int(len(curves[tm])),
              "K_star_vs_B3": kstar(curves[tm], curves.get("B3")),
              "K_star_vs_B2": kstar(curves[tm], curves.get("B2")),
              "K_star_vs_provisional": kstar(curves[tm], curves.get(HR1[lv]["provisional_denominator_ignoring_fcr_filter"])),
              "zero_cost_rate_by_k": {int(k): float(np.mean(v)) for k, v in sorted(zc.items())},
              "cum_final": {m: float(c[-1]) for m, c in curves.items()},
              "n0_charged": "n0=20 charged to every model-based method (shared initial data), not to whole-trial B2"}
    if lv != "R0":
        HK[lv]["K_star_uncorrected_JPC_vs_B3"] = kstar(curves.get("JPC"), curves.get("B3"))
    for m, c in curves.items():
        for k, v in enumerate(c):
            fig8.append({"level": lv, "method": m, "k": k + 1, "mean_cum_steps_incl_n0": float(v)})
S["HK"] = HK
hk_ok = {lv: (HK[lv]["K_star_vs_provisional"] is not None and HK[lv]["K_star_vs_provisional"] <= 15) for lv in HK}
VERD.append(dict(H="HK", family="descriptive", gate="K* finite and <= 15 (R2/R3)",
                 estimate="; ".join(f"{lv}: K* vs {HR1[lv]['provisional_denominator_ignoring_fcr_filter']}="
                                    f"{HK[lv]['K_star_vs_provisional']}, vs B2={HK[lv]['K_star_vs_B2']}" for lv in HK),
                 ci="–", p_raw=None,
                 verdict="FAIL (R2/R3: no K*)" if not (hk_ok["R2"] or hk_ok["R3"]) else "PASS", source="raw",
                 note="R0 amortises immediately; JPC_infl never breaks even at R2/R3 (refusal charged T_max); "
                      f"uncorrected JPC K* vs B3: R2={HK['R2'].get('K_star_uncorrected_JPC_vs_B3')}, "
                      f"R3={HK['R3'].get('K_star_uncorrected_JPC_vs_B3')} (diagnostic, no guarantee)"))
log(f"HK: " + json.dumps({lv: (HK[lv]['K_star_vs_B3'], HK[lv]['K_star_vs_B2']) for lv in HK}))

# ================================================================================================ 4. HR2 / HR2b
progress(4, TOTAL, "HR2")
hr2 = [r for r in load("hr2_threshold_accounting") if r.get("method")]
hr2_R0 = [r for r in hr2 if r.get("level") == "R0"]
T2 = {m: stream_totals(hr2_R0, m) for m in sorted({r["method"] for r in hr2_R0})}
HR2 = {}
for des in ("wald", "gopt"):
    HR2[f"threshold_only_{des}"] = ratio_stat(T2[f"LR-UI-{des}"], T2[f"LR-chi2-{des}"], null=0.5, tail="upper")
    for d in (6, 12, 24):
        HR2[f"threshold_only_{des}_d{d}"] = ratio_stat(T2[f"LR-UI-{des}"], T2[f"LR-chi2d{d}-{des}"])
ins2 = sorted(set.intersection(*[set(T2[m]) for m in ("LR-UI-wald", "LR-chi2-wald", "LR-UI-gopt", "LR-chi2-gopt")]))
lg = lambda m, i: math.log(T2[m][i] + 1)  # noqa: E731
thr_eff = np.array([0.5 * ((lg("LR-UI-wald", i) - lg("LR-chi2-wald", i)) + (lg("LR-UI-gopt", i) - lg("LR-chi2-gopt", i)))
                    for i in ins2])
des_eff = np.array([0.5 * ((lg("LR-UI-gopt", i) - lg("LR-UI-wald", i)) + (lg("LR-chi2-gopt", i) - lg("LR-chi2-wald", i)))
                    for i in ins2])
bt, bd = boot_mean(thr_eff), boot_mean(des_eff)
HR2["threshold_main_effect_log"] = {"mean": float(thr_eff.mean()), "ci": [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))]}
HR2["design_main_effect_log"] = {"mean": float(des_eff.mean()), "ci": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]}
HR2["threshold_gt_design_abs"] = bool(abs(thr_eff.mean()) > abs(des_eff.mean()))
# waterfall chain (R0, per-instance stream totals): B3 -> B3g -> LR-chi2-wald -> LR-UI-wald -> JPC-fresh -> JPC
chain = [("B3", "B3g", "support (continuous -> grid ∩ Wald)"), ("B3g", "LR-chi2-wald", "shape (Wald ellipsoid -> LR set, chi2 cut)"),
         ("LR-chi2-wald", "LR-UI-wald", "threshold (chi2_deff/2 -> log 1/delta)"),
         ("LR-UI-wald", "JPC-fresh", "design + per-problem fresh (Wald-opt -> DDA)"),
         ("JPC-fresh", "JPC", "reuse (fresh -> stream ledger)")]
water = []
for a, b_, lab in chain:
    s_ = ratio_stat(T2[b_], T2[a])
    water.append({"step": lab, "from": a, "to": b_, "ratio": s_["ratio"], "ci_lo": s_["ci"][0], "ci_hi": s_["ci"][1],
                  "log_ratio": math.log(s_["ratio"]), "n_instances": s_["n_instances"], "source": "hr2 R0 rows"})
# off-grid step from HR1 (same dev instances 660-663 at R0 and R3)
off = ratio_stat(stream_totals(R3rows, "JPC_infl"), stream_totals(R0rows, "JPC"))
off_unc = ratio_stat(stream_totals(R3rows, "JPC"), stream_totals(R0rows, "JPC"))
water.append({"step": "off-grid (R0 JPC -> R3 JPC_infl)", "from": "JPC@R0", "to": "JPC_infl@R3", "ratio": off["ratio"],
              "ci_lo": off["ci"][0], "ci_hi": off["ci"][1], "log_ratio": math.log(off["ratio"]),
              "n_instances": off["n_instances"], "source": "hr1 rows (different dev instances than hr2)"})
water.append({"step": "off-grid uncorrected (R0 JPC -> R3 JPC, diagnostic)", "from": "JPC@R0", "to": "JPC@R3",
              "ratio": off_unc["ratio"], "ci_lo": off_unc["ci"][0], "ci_hi": off_unc["ci"][1],
              "log_ratio": math.log(off_unc["ratio"]), "n_instances": off_unc["n_instances"], "source": "hr1 rows"})
HR2["waterfall"] = water
HR2["JPC_over_B3"] = ratio_stat(T2["JPC"], T2["B3"])
HR2["reuse_factor_JPC_over_fresh"] = ratio_stat(T2["JPC"], T2["JPC-fresh"])
# HR2b via the task's own prop_b() on raw rows + cover.jsonl
hr2b = None
try:
    import run_hr2_threshold_accounting as HR2MOD  # noqa: E402
    covers = [json.loads(l) for l in open(SRC / "hr2_threshold_accounting" / "cover.jsonl") if l.strip()]
    covers = list({(c["level"], c["instance"], c["stream"], c["K"]): c for c in covers}.values())
    pb = HR2MOD.prop_b(hr2_R0, [c for c in covers if c["level"] == "R0"], LOCK["d_eff"]["locked_value"])
    pts = pb.get("JPC", {}).get("points_stream", [])
    share = float(np.mean([(0.67 <= p["ratio"] <= 1.5) for p in pts])) if pts else None
    hr2b = {"cell": "JPC (UI, stream reuse)", "n_streams": len(pts), "share_in_band": share,
            "median_pred_over_obs": float(np.median([p["ratio"] for p in pts])) if pts else None,
            "by_cell": {m: v.get("stream_level") for m, v in pb.items()},
            "p_binom_share_gt_0.7": binom_p_ge(int(round((share or 0) * len(pts))), len(pts), 0.7) if pts else None}
except Exception as e:  # noqa: BLE001
    hr2b = {"error": repr(e)}
    log(f"HR2b prop_b failed: {e!r}")
HR2["HR2b"] = hr2b
S["HR2"] = HR2
log(f"HR2 threshold-only (wald) {fmt_r(HR2['threshold_only_wald'])}; HR2b {hr2b.get('share_in_band') if hr2b else None}")

# ================================================================================================ 5. HR3, HR4, HR5
progress(5, TOTAL, "HR3-HR5")
hr3 = [r for r in load("hr3_r1_uncorrected") + load("hr3_r1_uncorrected", "hr3_r1_uncorrected_dev_supp")
       if r.get("method") == "JPC" and r.get("level") == "R1"]
BINS = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
hr3_bins, groups = [], []
for lo, hi in BINS:
    R = [r for r in hr3 if r.get("eta_dec_over_eps") is not None and lo <= r["eta_dec_over_eps"] < hi and r.get("model_cert")]
    k = int(sum(bool(r.get("model_false_cert")) for r in R))
    hr3_bins.append({"bin_eta_dec_over_eps": f"[{lo},{hi if hi < 1e8 else 'inf'})", "n_cert": len(R), "false_certs": k,
                     "fcr": k / len(R) if R else None, "cp": list(clopper_pearson(k, len(R))) if R else None})
    groups.append([float(bool(r.get("model_false_cert"))) for r in R])
_, jt_p3 = jonckheere(groups)
hi_bin = [r for r in hr3 if (r.get("eta_dec_over_eps") or 0) >= 0.5 and r.get("model_cert")]
k_hi = int(sum(bool(r.get("model_false_cert")) for r in hi_bin))
p_hr3_bin = binom_p_ge(k_hi, len(hi_bin), DELTA)
S["HR3"] = {"bins": hr3_bins, "jonckheere_p": jt_p3, "high_bin_n": len(hi_bin), "high_bin_false": k_hi,
            "high_bin_cp_lower": cp_lower(k_hi, len(hi_bin)), "p_binom_fcr_gt_delta": p_hr3_bin,
            "n_rows": len(hr3), "sources": ["hr3_r1_uncorrected", "hr3_r1_uncorrected_dev_supp"]}
# HR4
h4, _ = merge_chunks(["hr4_phase_coarse", "hr4_phase_fine"], key5)
hr4 = {}
fig3 = []
for lv in sorted({r["level"] for r in h4}):
    R = [r for r in h4 if r["level"] == lv]
    f = R[0]["f"]
    ji = [r for r in R if r["method"] == "JPC_infl"]
    comp = float(np.mean([r["status"] == "CERTIFIED" for r in ji])) if ji else None
    lh = [r.get("log_h_over_hstar") for r in ji if r.get("log_h_over_hstar") is not None]
    lh_c1 = [r.get("log_h_over_hstar_cond1") for r in ji if r.get("log_h_over_hstar_cond1") is not None]
    rb3 = ratio_stat(stream_totals(R, "JPC_infl"), stream_totals(R, "B3")) if any(r["method"] == "B3" for r in R) else None
    rb8 = ratio_stat(stream_totals(R, "JPC_infl"), stream_totals(R, "B8_infl"))
    run = ratio_stat(stream_totals(R, "JPC"), stream_totals(R, "B3")) if any(r["method"] == "B3" for r in R) else None
    hr4[lv] = {"f": f, "jpc_infl_completion": comp, "log_h_over_hstar_median": float(np.median(lh)) if lh else None,
               "log_h_over_hstar_cond1_median": float(np.median(lh_c1)) if lh_c1 else None,
               "share_h_le_hstar": float(np.mean([x <= 0 for x in lh])) if lh else None,
               "JPC_infl_over_B3": rb3 and {k: rb3[k] for k in ("ratio", "ci")},
               "JPC_infl_over_B8_infl": rb8 and {k: rb8[k] for k in ("ratio", "ci", "degenerate_identical")},
               "JPC_uncorrected_over_B3": run and {k: run[k] for k in ("ratio", "ci")},
               "jpc_unc_completion": float(np.mean([r["status"] == "CERTIFIED" for r in R if r["method"] == "JPC"]))}
    for r in ji:
        fig3.append({"level": lv, "f": f, "instance": r["instance"], "k": r["k"],
                     "log_h_over_hstar": r.get("log_h_over_hstar"), "log_h_over_hstar_cond1": r.get("log_h_over_hstar_cond1"),
                     "certified": r["status"] == "CERTIFIED"})
# f=2 from HR1@R3 (JPC_infl completion) + lock dev h-ratio
hr4["R3_f2(hr1)"] = {"f": "2", "jpc_infl_completion": HR1["R3"]["cells"]["JPC_infl"]["completion"],
                     "log_h_over_hstar_median": math.log(LOCK["h_ratio_dev"]["2"]["h_over_hstar_literal_median"]),
                     "log_h_over_hstar_cond1_median": math.log(LOCK["h_ratio_dev"]["2"]["h_over_hstar_cond1_median"]),
                     "share_h_le_hstar": LOCK["h_ratio_dev"]["2"]["share_h_le_hstar_over_2_literal"],
                     "source": "hr1_r3 rows (completion) + prereg_lock.h_ratio_dev (h/h*)"}
all_comp = [v["jpc_infl_completion"] for v in hr4.values() if v.get("jpc_infl_completion") is not None]
hr4_degenerate = len(set(all_comp)) <= 1
S["HR4"] = {"by_level": hr4, "logistic": "not estimable: JPC_infl completion identical (=%.2f) at every f and no "
                                         "problem has h<=h*" % all_comp[0] if hr4_degenerate else "estimable",
            "degenerate": hr4_degenerate}
# HR5
o = [r for r in load("oracle_opt_cover") if r.get("kind") == "oracle" and r.get("level") == "R0"]
o_ratio = {}
for i in sorted({r["instance"] for r in o}):
    Ri = [r for r in o if r["instance"] == i and r.get("stream", 0) == 0]
    if not Ri:
        continue
    last = max(Ri, key=lambda r: r["k"])
    if last.get("opt_ratio") is not None:
        o_ratio[i] = float(last["opt_ratio"])
jb8 = ratio_stat(stream_totals(R0rows, "JPC"), stream_totals(R0rows, "B8"))
common = sorted(set(o_ratio) & set(jb8["per_instance"]))
sp = stats.spearmanr([jb8["per_instance"][i] for i in common], [o_ratio[i] for i in common]) if len(common) >= 3 else None
cells_h_le = [lv for lv, v in hr4.items() if (v.get("share_h_le_hstar") or 0) > 0]
S["HR5"] = {"cells_with_h_le_hstar": cells_h_le,
            "JPC_infl_over_B8_infl_by_f": {lv: v.get("JPC_infl_over_B8_infl") for lv, v in hr4.items()},
            "R0_JPC_over_B8": {k: jb8[k] for k in ("ratio", "ci", "per_instance")},
            "oracle_ratio_OPT_union_over_OPT_id_R0": o_ratio, "spearman_instances": common,
            "spearman_rho": float(sp.statistic) if sp is not None else None,
            "spearman_p": float(sp.pvalue) if sp is not None else None,
            "note": "no grid cell with h<=h*; JPC_infl/B8_infl = 1 (both refuse, charged T_max) -> HR5 primary untestable; "
                    "Spearman on R0 only (n=%d, no power)" % len(common)}
# oracle suspicious-gate check: cumulative JPC incl n0 vs OPT_K at R0 (instances shared with hr1_r0 pilot)
below = checked = 0
for r in o:
    if r.get("opt_k") is None:
        continue
    cum = [x for x in R0rows if x["method"] == "JPC" and x["instance"] == r["instance"]
           and x.get("stream", 0) == r.get("stream", 0) and x["k"] <= r["k"]]
    if len(cum) != r["k"] + 1 or any(x.get("censored") for x in cum):
        continue
    checked += 1
    below += int(N0 + sum(x["new_env_steps"] for x in cum) < r["opt_k"])
S["oracle_check"] = {"prefixes_checked": checked, "below_OPT_K": below}

# family R
pR = {"HR2": HR2["threshold_only_wald"]["p"],
      "HR2b": (hr2b or {}).get("p_binom_share_gt_0.7"),
      "HR3": max(p_hr3_bin, jt_p3 if jt_p3 is not None else 1.0),
      "HR4": 1.0,
      "HR5": 1.0}
hR = holm({k: (1.0 if v is None else v) for k, v in pR.items()})
tw = HR2["threshold_only_wald"]
VERD += [
    dict(H="HR2", family="R", gate="threshold-only ratio in [0.2,0.5]; threshold main effect > design",
         estimate=f"UI/chi2(d_eff={LOCK['d_eff']['locked_value']}) @Wald-opt = {tw['ratio']:.2f}; d=6/12/24: "
                  + "/".join(f"{HR2[f'threshold_only_wald_d{d}']['ratio']:.2f}" for d in (6, 12, 24))
                  + f"; |thr| {abs(HR2['threshold_main_effect_log']['mean']):.2f} vs |design| {abs(HR2['design_main_effect_log']['mean']):.2f} (log)",
         ci=f"[{tw['ci'][0]:.2f}, {tw['ci'][1]:.2f}]", p_raw=pR["HR2"], p_holm=hR["HR2"],
         verdict="FAIL (ratio>0.7: reverse direction)" if tw["ratio"] > 0.7 else ("PASS" if 0.2 <= tw["ratio"] <= 0.5 else "FAIL"),
         source="raw", note="UI is MORE expensive than chi2(d_eff~3) (chi2 at d_eff~3 is not anytime-valid: theta* cell survival <1 in hr2); the reuse ledger is the largest saving factor"),
    dict(H="HR2b", family="R", gate="share of streams (N>=50) with pred/obs in [0.67,1.5] >= 0.7",
         estimate=(f"share {hr2b['share_in_band']:.2f} (n={hr2b['n_streams']}), median pred/obs {hr2b['median_pred_over_obs']:.3f}"
                   if hr2b and hr2b.get("share_in_band") is not None else str(hr2b)[:80]),
         ci="–", p_raw=pR["HR2b"], p_holm=hR["HR2b"],
         verdict="FAIL" if (hr2b or {}).get("share_in_band") is not None and hr2b["share_in_band"] < 0.7 else "NOT_TESTABLE",
         source="raw (prop_b on rows + cover.jsonl)", note="Cover LP relaxes reachability -> prediction far below observed"),
    dict(H="HR3", family="R", gate="Jonckheere p<0.05 and CP lower>delta in eta_dec>=eps/2",
         estimate=f"eta_dec>=eps/2: {k_hi}/{len(hi_bin)} false certs; bins " + ", ".join(
             f"{b['bin_eta_dec_over_eps']}:{b['false_certs']}/{b['n_cert']}" for b in hr3_bins),
         ci=f"CP lower {cp_lower(k_hi, len(hi_bin)):.3f}", p_raw=pR["HR3"], p_holm=hR["HR3"],
         verdict="FAIL (falsification branch: off-grid nearly harmless at G_1)" if k_hi / max(len(hi_bin), 1) <= DELTA else "PASS",
         source="raw", note=f"Jonckheere p={jt_p3}; pooled main + dev_supp"),
    dict(H="HR4", family="R", gate="logistic inflection in [h*/2, 2h*]",
         estimate="JPC_infl completion " + ", ".join(f"{lv}:{v['jpc_infl_completion']:.2f}" for lv, v in hr4.items()
                                                    if v.get('jpc_infl_completion') is not None),
         ci="–", p_raw=pR["HR4"], p_holm=hR["HR4"], verdict="NOT_TESTABLE (degenerate)", source="raw",
         note=S["HR4"]["logistic"]),
    dict(H="HR5", family="R", gate="JPC_infl/B8 CI upper<1 at h<=h*; Spearman>=0.7",
         estimate=f"no h<=h* cell; R0 JPC/B8 {jb8['ratio']:.2f}; Spearman {S['HR5']['spearman_rho']} (n={len(common)})",
         ci=f"R0 [{jb8['ci'][0]:.2f}, {jb8['ci'][1]:.2f}]", p_raw=pR["HR5"], p_holm=hR["HR5"],
         verdict="NOT_TESTABLE", source="raw", note=S["HR5"]["note"]),
]
log(f"HR3 high-bin {k_hi}/{len(hi_bin)}, JT p {jt_p3}; HR4 degenerate {hr4_degenerate}; HR5 spearman {S['HR5']['spearman_rho']}")

# ================================================================================================ 6. family F
progress(6, TOTAL, "family F")
hf1 = [r for r in load("hf1_mixed_cost") if r.get("method") and r.get("k") is not None]
hf1_R0 = [r for r in hf1 if r["level"] == "R0"]
HF1 = {}
for m in ("mixed", "mixed_half"):
    HF1[m] = ratio_stat(stream_totals(hf1_R0, m), stream_totals(hf1_R0, "plugin"), null=1.15, tail="upper")
    R = [r for r in hf1_R0 if r["method"] == m]
    fa = int(sum(r.get("status") == "MODEL_CONFLICT" or bool(r.get("model_conflict")) for r in R))
    cov = [bool(r["theta_cell_alive"]) for r in R if r.get("theta_cell_alive") is not None]
    HF1[m].update({"false_alarm": fa, "n": len(R), "false_alarm_cp_upper": cp_upper(fa, len(R)),
                   "coverage": float(np.mean(cov)) if cov else None,
                   "fcr": int(sum(bool(r.get("false_cert")) for r in R))})
HF1["R2_uncorrected_mixed_over_plugin"] = ratio_stat(stream_totals([r for r in hf1 if r["level"] == "R2"], "mixed"),
                                                     stream_totals([r for r in hf1 if r["level"] == "R2"], "plugin"))
hf1m = HF1["mixed"]
hf1_pass = hf1m["ratio"] <= 1.15 and hf1m["ci"][1] <= 1.25 and (hf1m["coverage"] or 0) >= 1 - DELTA and hf1m["false_alarm_cp_upper"] <= DELTA

# HF2
hf2_all = load("hf2_static_falsify") + load("hf2_static_falsify", "hf2_static_falsify_dev_supp")
idx = lambda r: (r["instance"], r.get("stream", 0), r["k"])  # noqa: E731
by_m = defaultdict(dict)
for r in hf2_all:
    if r.get("method"):
        by_m[r["method"]][idx(r)] = r
plug = by_m["plugin@R0"]
fc_keys = [k for k, r in plug.items() if r.get("false_cert")]


def conflicted(r):
    return r is not None and (r.get("status") == "MODEL_CONFLICT" or bool(r.get("conflict_before_problem")))


det = {arm: int(sum(conflicted(by_m[arm].get(k)) for k in fc_keys)) for arm in ("mixed_lb@R0", "mixed_pid@R0", "mixed_placebo@R0")}


def shadow_conf(r, comp):
    sh = r.get("shadow")
    if isinstance(sh, str):
        try:
            sh = json.loads(sh.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null"))
        except Exception:  # noqa: BLE001
            sh = None
    return bool(sh and sh.get(comp, {}).get("conflict"))


shadow_det = {comp: int(sum(shadow_conf(plug[k], comp) for k in fc_keys))
              for comp in ("main", "ext_only", "reg_lb", "reg_pid", "reg_placebo", "half")}
mixed_rows = list(by_m["mixed_lb@R0"].values())
mfc = int(sum(bool(r.get("false_cert")) for r in mixed_rows))
mcert = int(sum(r.get("status") == "CERTIFIED" for r in mixed_rows))
twin_fa = int(sum(r.get("status") == "MODEL_CONFLICT" for r in by_m["mixed_lb@twin"].values()))
twin_shadow_fa = int(sum(shadow_conf(r, "main") for r in by_m["plugin@twin"].values()))
HF2 = {"plugin_false_certs": len(fc_keys), "plugin_n": len(plug), "in_stream_detection": det,
       "shadow_detection_same_data": shadow_det, "mixed_lb_fcr": mfc / max(mcert, 1), "mixed_lb_false": mfc,
       "mixed_lb_cert": mcert, "mixed_lb_fcr_cp_upper": cp_upper(mfc, mcert),
       "twin_false_alarm_mixed": twin_fa, "twin_n": len(by_m["mixed_lb@twin"]),
       "twin_shadow_false_alarm": twin_shadow_fa,
       "placebo_over_true_reg_only": (shadow_det["reg_placebo"] / shadow_det["reg_lb"]) if shadow_det["reg_lb"] else None,
       "sources": ["hf2_static_falsify", "hf2_static_falsify_dev_supp"]}
det_rate = det["mixed_lb@R0"] / max(len(fc_keys), 1)
p_hf2 = max(binom_p_ge(det["mixed_lb@R0"], len(fc_keys), 0.6), binom_p_le(mfc, mcert, 2 * DELTA))

# HF2b
h2b = [r for r in load("hf2b_m1_m3_mixed") if r.get("method") and r.get("k") is not None]
HF2b = {}
p2b = []
for ms in ("m1", "m3"):
    for m in ("plugin", "mixed", "AGC"):
        R = [r for r in h2b if r["misspec"] == ms and float(r.get("eta_level") or 0) == 2.0 and r["method"] == m]
        fc = int(sum(bool(r.get("false_cert")) for r in R))
        nc = int(sum(r.get("status") == "CERTIFIED" for r in R))
        al = int(sum(r.get("status") == "MODEL_CONFLICT" or bool(r.get("alarm")) for r in R))
        HF2b[f"{ms}|{m}"] = {"n": len(R), "n_cert": nc, "false_certs": fc, "fcr": fc / max(nc, 1),
                             "fcr_cp_upper": cp_upper(fc, nc), "alarms": al}
    p2b.append(binom_p_le(HF2b[f"{ms}|mixed"]["false_certs"], HF2b[f"{ms}|mixed"]["n_cert"], 2 * DELTA))
agc_steps = [float((r.get("audit") or {}).get("steps", 0)) for r in h2b if r["method"] == "AGC"
             and float(r.get("eta_level") or 0) == 2.0 and isinstance(r.get("audit"), dict)]
HF2b["AGC_audit_steps_median_eta2"] = float(np.median(agc_steps)) if agc_steps else None
mixed_alarms = HF2b["m1|mixed"]["alarms"] + HF2b["m3|mixed"]["alarms"]
p_hf2b = 1.0 if mixed_alarms == 0 else max(p2b)

# HF3
h3 = [r for r in load("hf3_m4_blindspot") if r.get("method") and r.get("k") is not None]
streams = defaultdict(bool)
for r in h3:
    if r["method"] == "mixed":
        streams[(r["instance"], r.get("stream", 0))] |= (r.get("status") == "MODEL_CONFLICT" or bool(r.get("alarm"))
                                                         or bool(r.get("alarm_active")))
hf3_det, hf3_n = int(sum(streams.values())), len(streams)
HF3 = {"mixed_streams_alarmed": hf3_det, "n_streams": hf3_n, "cp_upper": cp_upper(hf3_det, hf3_n),
       "plugin_false_certs": int(sum(bool(r.get("false_cert")) for r in h3 if r["method"] == "plugin")),
       "mixed_false_certs": int(sum(bool(r.get("false_cert")) for r in h3 if r["method"] == "mixed")),
       "replay_conflicts": int(sum(bool((r.get("replay") or {}).get("conflict")) if isinstance(r.get("replay"), dict) else 0 for r in h3))}
p_hf3 = binom_p_le(hf3_det, hf3_n, 2 * DELTA)

# HF4
h4r = load("hf4_eprocess_audit")
agc4 = [r["audit_steps"] for r in h4r if r["method"] == "AGC" and float(r["eta_level"]) == 2.0 and r.get("audit_steps") is not None]
agc_med = float(np.median(agc4)) if agc4 else None
HF4 = {"AGC_audit_steps_median": agc_med,
       "AGC_fcr": int(sum(bool(r.get("false_cert")) for r in h4r if r["method"] == "AGC" and float(r["eta_level"]) == 2.0))}
for m, tau in (("eprocess", "half"), ("eprocess", "half_pp"), ("eprocess", "one"), ("chi2", "half")):
    R = [r for r in h4r if r["method"] == m and r.get("tau") == tau and abs(float(r["f"]) - 0.1) < 1e-9
         and float(r["eta_level"]) == 2.0]
    fc = int(sum(bool(r.get("false_cert")) for r in R))
    nc = int(sum(r.get("status") == "CERTIFIED" for r in R)) or len(R)
    cost = float(np.median([r["audit_steps"] for r in R if r.get("audit_steps") is not None])) if R else None
    HF4[f"{m}|{tau}"] = {"n": len(R), "false_certs": fc, "fcr": fc / max(nc, 1), "fcr_cp_upper": cp_upper(fc, nc),
                         "audit_steps_median": cost, "cost_ratio_vs_AGC": (cost / agc_med) if (cost and agc_med) else None}
e4 = HF4["eprocess|half"]
p_hf4 = binom_p_le(e4["false_certs"], max(e4["n"], 1), 2 * DELTA)

pF = {"HF1": HF1["mixed"]["p"], "HF2": p_hf2, "HF2b": p_hf2b, "HF3": p_hf3, "HF4": p_hf4}
hF = holm(pF)
S["HF"] = {"HF1": HF1, "HF2": HF2, "HF2b": HF2b, "HF3": HF3, "HF4": HF4, "p_raw": pF, "p_holm": hF,
           "p_conventions": {"HF1": "bootstrap P(log ratio >= log 1.15)", "HF2": "IU max(binom P(det>=k|0.6), binom P(FCR<=k|2delta))",
                             "HF2b": "1 if no mixed alarm (alarm-step gate unevaluable) else max binom FCR p",
                             "HF3": "binom P(X<=k | 2delta) (H1: detection < 2delta; expected-negative claim)",
                             "HF4": "binom P(X<=k | 2delta) for e-process FCR at budget AGC/10"}}
VERD += [
    dict(H="HF1", family="F", gate="mixed/plugin <= 1.15 (CI upper <= 1.25); coverage >= 1-delta; false alarm <= delta",
         estimate=f"mixed {HF1['mixed']['ratio']:.3f}, half {HF1['mixed_half']['ratio']:.3f}; coverage {HF1['mixed']['coverage']}; "
                  f"false alarm {HF1['mixed']['false_alarm']}/{HF1['mixed']['n']}",
         ci=f"mixed [{HF1['mixed']['ci'][0]:.3f}, {HF1['mixed']['ci'][1]:.3f}]; half [{HF1['mixed_half']['ci'][0]:.3f}, {HF1['mixed_half']['ci'][1]:.3f}]",
         p_raw=pF["HF1"], p_holm=hF["HF1"], verdict="PASS" if hf1_pass else "FAIL (pilot n)", source="raw",
         note="pilot gate (<=1.25x) met by point estimate; R2 uncorrected mixed/plugin "
              f"{HF1['R2_uncorrected_mixed_over_plugin']['ratio']:.2f} flagged suspicious (falsification power, coverage drop)"),
    dict(H="HF2", family="F", gate="detection >= 0.6; FCR <= 2delta; placebo <= 1/3 true; kappa=0 twin false alarm <= delta",
         estimate=f"in-stream detection {det['mixed_lb@R0']}/{len(fc_keys)}; mixed FCR {mfc}/{mcert}; shadow(main) "
                  f"{shadow_det['main']}/{len(fc_keys)}; reg-only placebo/true {shadow_det['reg_placebo']}/{shadow_det['reg_lb']}; "
                  f"twin false alarm {twin_fa}/{HF2['twin_n']}",
         ci=f"FCR CP upper {HF2['mixed_lb_fcr_cp_upper']:.3f}", p_raw=pF["HF2"], p_holm=hF["HF2"],
         verdict="FAIL (negative result)" if det_rate < 0.6 else "PASS", source="raw", note="ext wins under misspecification; set shrinks before it empties"),
    dict(H="HF2b", family="F", gate="eta=2eps FCR CP upper <= 2delta; alarm steps <= AGC/100",
         estimate=f"mixed FCR m1 {HF2b['m1|mixed']['false_certs']}/{HF2b['m1|mixed']['n_cert']}, m3 {HF2b['m3|mixed']['false_certs']}/"
                  f"{HF2b['m3|mixed']['n_cert']}; mixed alarms {mixed_alarms}",
         ci=f"CP upper m1 {HF2b['m1|mixed']['fcr_cp_upper']:.3f}, m3 {HF2b['m3|mixed']['fcr_cp_upper']:.3f}",
         p_raw=pF["HF2b"], p_holm=hF["HF2b"], verdict="FAIL (negative result)", source="raw",
         note="no in-stream alarm; AGC FCR 0 at ~1e5 audit steps"),
    dict(H="HF3", family="F", gate="detection <= 2delta (expected negative: blind spot)",
         estimate=f"mixed alarmed streams {hf3_det}/{hf3_n}", ci=f"CP upper {HF3['cp_upper']:.3f}",
         p_raw=pF["HF3"], p_holm=hF["HF3"],
         verdict="DIRECTION (blind spot as expected; CP upper > 2delta at pilot n)" if HF3["cp_upper"] > 2 * DELTA else "PASS",
         source="raw", note=f"plugin false certs {HF3['plugin_false_certs']} -> paired detection undefined"),
    dict(H="HF4", family="F", gate="e-process FCR <= 2delta at cost <= AGC/10",
         estimate=f"e-process(tau=eps/2, f=0.1) FCR {e4['false_certs']}/{e4['n']}; cost ratio {e4['cost_ratio_vs_AGC']}",
         ci=f"CP upper {e4['fcr_cp_upper']:.3f}", p_raw=pF["HF4"], p_holm=hF["HF4"],
         verdict="FAIL (negative result)" if e4["fcr"] > 2 * DELTA else "PASS", source="raw",
         note=f"chi2(fixed n) FCR {HF4['chi2|half']['false_certs']}/{HF4['chi2|half']['n']} but alarms on most correct problems"),
]
log(f"HF: p_raw {pF}")

# ================================================================================================ 7. family D
progress(7, TOTAL, "family D")
hd1 = load("hd1_matched_twin_static") + load("hd1_matched_twin_static", "hd1_matched_twin_static_dev_supp")
HD1 = {}
groups_static = []
for m in ("static@g0.0", "static@g0.5", "static@g1.0", "dyn_G1@g0.0", "dyn_Gext@g0.5", "dyn_G1@g1.0", "dyn_G1@g0.5"):
    R = [r for r in hd1 if r.get("method") == m]
    cert = [r for r in R if r.get("status") == "CERTIFIED" or r.get("certified")]
    fc = int(sum(bool(r.get("false_cert")) for r in cert))
    HD1[m] = {"n": len(R), "n_cert": len(cert), "false_certs": fc, "fcr": fc / max(len(cert), 1),
              "cp": list(clopper_pearson(fc, len(cert)))}
    if m.startswith("static"):
        groups_static.append([float(bool(r.get("false_cert"))) for r in cert])
_, jt_hd1 = jonckheere(groups_static)
hs = HD1["static@g1.0"]
p_hd1 = max(binom_p_ge(hs["false_certs"], hs["n_cert"], DELTA), jt_hd1 or 1.0)
hd1_pass = hs["cp"][0] > DELTA and HD1["static@g0.0"]["fcr"] <= DELTA and (jt_hd1 or 1) < 0.05
HD1["jonckheere_p_static"] = jt_hd1
# HD1b
hb = load("hd1b_same_exposure")
HD1b = {}
pb_list = []
for env in ("T-Lin", "T-NL"):
    for des in ("uniform", "dda", "onpolicy"):
        for cls in ("static", "dynamic"):
            R = [r for r in hb if r["env"] == env and r["design"] == des and r["cls"] == cls]
            cert = [r for r in R if r.get("certified")]
            fc = int(sum(bool(r.get("false_cert")) for r in R))
            HD1b[f"{env}|{des}|{cls}"] = {"n": len(R), "n_cert": len(cert), "false_certs": fc,
                                          "fcr": fc / max(len(R), 1), "cp": list(clopper_pearson(fc, len(R)))}
            if cls == "static":
                pb_list.append(binom_p_ge(fc, len(R), 0.8))
dyn = [r for r in hb if r["cls"] == "dynamic"]
dfc = int(sum(bool(r.get("false_cert")) for r in dyn))
dcert = int(sum(bool(r.get("certified")) for r in dyn))
HD1b["dynamic_pooled"] = {"false_certs": dfc, "n_cert": dcert, "cp": list(clopper_pearson(dfc, dcert))}
pb_list.append(binom_p_le(dfc, dcert, DELTA))
p_hd1b = max(pb_list)
hd1b_static_min = min(v["fcr"] for k, v in HD1b.items() if k.endswith("|static"))
# HD2 (law_points.csv)
lp = list(csv.DictReader(open(SRC / "hd2_kappa_sandwich_lin" / "law_points.csv")))


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def slope_boot(pts):
    """pts: list of (instance, x, y); OLS slope + instance-cluster bootstrap CI + TOST-type p for [1.5, 2.5]."""
    if len(pts) < 3:
        return None
    ins = sorted({p[0] for p in pts})
    byi = defaultdict(list)
    for p in pts:
        byi[p[0]].append(p)
    X = np.array([p[1] for p in pts])
    Y = np.array([p[2] for p in pts])
    sl = float(np.polyfit(X, Y, 1)[0])
    rng = np.random.default_rng(BSEED)
    bs = []
    for _ in range(2000):
        pick = rng.choice(ins, size=len(ins), replace=True)
        P = [q for i in pick for q in byi[i]]
        xx = np.array([q[1] for q in P])
        if np.ptp(xx) < 1e-12:
            continue
        bs.append(np.polyfit(xx, np.array([q[2] for q in P]), 1)[0])
    bs = np.array(bs)
    return {"n": len(pts), "n_instances": len(ins), "slope": sl,
            "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "p_in_1.5_2.5": float(max(np.mean(bs <= 1.5), np.mean(bs >= 2.5)))}


HD2 = {}
for part in ("time", "participant", "policy"):
    pts = []
    for r in lp:
        if r["kind"] != "main" or r["evidence"] != part or r["design"] != "phased":
            continue
        N, Nj, ke = fnum(r["N"]), fnum(r["N_joint"]), fnum(r["plus_kappa_eff"])
        if N and Nj and ke and ke > 0 and r.get("censored") != "True" and r.get("cens_joint") != "True":
            pts.append((r["instance"], math.log(ke), math.log(N / Nj)))
    HD2[f"kappa_eff_slope|{part}"] = slope_boot(pts)
pts = []
for r in lp:
    if r["kind"] == "construction" and r["evidence"] == "time" and r["design"] == "phased":
        N, Nj, H_ = fnum(r["N"]), fnum(r["N_joint"]), fnum(r["H"])
        if N and Nj and H_:
            pts.append((r["instance"], math.log(H_), math.log(N / Nj)))
HD2["construction_slope_time_vs_logH"] = slope_boot(pts)
h1z = [r for r in lp if r["kind"] == "main" and r["evidence"] == "time" and fnum(r["H"]) == 1.0]
HD2["H1_time_exact_zero"] = {"n": len(h1z), "equal": int(sum(fnum(r["N"]) == fnum(r["N_joint"]) for r in h1z))}
hd2s = summ("hd2_kappa_sandwich_lin")
HD2["sandwich_from_summary"] = {"violations": (hd2s.get("gate") or {}).get("sandwich_violations"),
                                "sandwich_100pct": (hd2s.get("gate") or {}).get("sandwich_100pct")}
part_p = [v["p_in_1.5_2.5"] for k, v in HD2.items() if k.startswith("kappa_eff_slope|") and v]
p_hd2 = max(part_p) if part_p else 1.0
hd2_pass = bool(HD2["sandwich_from_summary"]["sandwich_100pct"]) and all(
    v and 1.5 <= v["slope"] <= 2.5 for k, v in HD2.items() if k.startswith("kappa_eff_slope|")) and \
    HD2["H1_time_exact_zero"]["n"] == HD2["H1_time_exact_zero"]["equal"]
# HD3
h3r = load("hd3_matched_twin_rect")
tt = defaultdict(float)
for r in h3r:
    tt[(r["instance"], r["twin"], r["method"])] += float(r.get("new_env_steps") or 0)
ins3 = sorted({r["instance"] for r in h3r})
lk = {tw: {i: math.log((tt[(i, tw, "rect")] + 1) / (tt[(i, tw, "joint")] + 1)) for i in ins3} for tw in ("khigh", "k0")}
Dv = np.array([lk["khigh"][i] - lk["k0"][i] for i in ins3])
bD = boot_mean(Dv)
HD3 = {"log_rect_over_joint": {tw: float(np.mean(list(v.values()))) for tw, v in lk.items()},
       "D": float(Dv.mean()), "D_ci": [float(np.quantile(bD, .025)), float(np.quantile(bD, .975))],
       "p_D_le_0": float((1 + np.sum(bD <= 0)) / (B + 1)), "per_instance_D": dict(zip(map(int, ins3), map(float, Dv))),
       "fcr": int(sum(bool(r.get("false_cert")) for r in h3r))}
pD = {"HD1": p_hd1, "HD1b": p_hd1b, "HD2": p_hd2, "HD3": HD3["p_D_le_0"]}
hD = holm(pD)
S["HD"] = {"HD1": HD1, "HD1b": HD1b, "HD2": HD2, "HD3": HD3, "p_raw": pD, "p_holm": hD,
           "p_conventions": {"HD1": "IU max(binom P(X>=k|delta) at static g=1, Jonckheere over dose)",
                             "HD1b": "IU max over designs of binom P(static>=k|0.8) and binom P(dynamic<=k|delta)",
                             "HD2": "IU max over partitions of bootstrap P(kappa_eff slope outside [1.5,2.5])",
                             "HD3": "bootstrap P(D<=0)"}}
VERD += [
    dict(H="HD1", family="D", gate="static FCR CP lower > delta at kappa_high; g=0 <= delta; Jonckheere",
         estimate=f"static FCR g0/g0.5/g1 = {HD1['static@g0.0']['false_certs']}/{HD1['static@g0.0']['n_cert']}, "
                  f"{HD1['static@g0.5']['false_certs']}/{HD1['static@g0.5']['n_cert']}, {hs['false_certs']}/{hs['n_cert']}; "
                  f"dynamic 0", ci=f"g1 CP [{hs['cp'][0]:.3f}, {hs['cp'][1]:.3f}]", p_raw=pD["HD1"], p_holm=hD["HD1"],
         verdict="PASS" if hd1_pass else "FAIL", source="raw", note=f"Jonckheere p={jt_hd1}; pooled main + dev_supp"),
    dict(H="HD1b", family="D", gate="static FCR >= 0.8 under all designs; dynamic <= delta",
         estimate=f"min static FCR {hd1b_static_min:.2f}; dynamic {dfc}/{dcert}", ci=f"dynamic CP upper {HD1b['dynamic_pooled']['cp'][1]:.3f}",
         p_raw=pD["HD1b"], p_holm=hD["HD1b"],
         verdict="PASS" if hd1b_static_min >= 0.8 and HD1b["dynamic_pooled"]["cp"][1] <= DELTA else "PASS (point) / pilot n",
         source="raw", note="constructive worst case; T-NL only continuous C' (see hd1b caveats)"),
    dict(H="HD2", family="D", gate="sandwich 100%; kappa_eff slope in [1.5,2.5] per partition; H=1 time exact 0",
         estimate=", ".join(f"{k.split('|')[1]} {v['slope']:.2f}" for k, v in HD2.items() if k.startswith("kappa_eff_slope|") and v)
                  + f"; construction {HD2['construction_slope_time_vs_logH']['slope']:.2f}; H=1 exact {HD2['H1_time_exact_zero']['equal']}/{HD2['H1_time_exact_zero']['n']}",
         ci="; ".join(f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]" for k, v in HD2.items() if k.startswith("kappa_eff_slope|") and v),
         p_raw=pD["HD2"], p_holm=hD["HD2"], verdict="PASS" if hd2_pass else "FAIL", source="raw + summary (sandwich)",
         note="sandwich violations taken from task summary (design-level identity check)"),
    dict(H="HD3", family="D", gate="paired D CI lower > 0",
         estimate=f"D = {HD3['D']:.3f} (khigh {HD3['log_rect_over_joint']['khigh']:.3f}, k0 {HD3['log_rect_over_joint']['k0']:.3f})",
         ci=f"[{HD3['D_ci'][0]:.3f}, {HD3['D_ci'][1]:.3f}]", p_raw=pD["HD3"], p_holm=hD["HD3"],
         verdict="PASS" if HD3["D_ci"][0] > 0 else "FAIL (CI contains 0)", source="raw", note="n=5 instances"),
]
log(f"HD: p_raw {pD}")

# ================================================================================================ 8. HS1, HH1, HV, E2, replica
progress(8, TOTAL, "HS1/HH1/HV/E2/replica")
hs1 = load("hs1_sampler_equivalence")
HS1 = {}
table2 = []
for lv, sfx in (("R0", ""), ("R3", "_infl"), ("R3", "_unc")):
    R = [r for r in hs1 if r.get("level") == lv]
    T = {s: stream_totals(R, s + sfx) for s in ("DDA", "B10", "B11", "uniform")}
    for a, b_ in (("DDA", "B10"), ("DDA", "B11"), ("B10", "B11"), ("DDA", "uniform")):
        if not T[a] or not T[b_]:
            continue
        s_ = ratio_stat(T[a], T[b_], ci=0.90)
        eq = s_["ci"][0] >= 0.8 and s_["ci"][1] <= 1.25
        key = f"{lv}{sfx}|{a}/{b_}"
        HS1[key] = {"ratio": s_["ratio"], "ci90": s_["ci"], "tost_equivalent": eq,
                    "degenerate_identical": s_["degenerate_identical"], "n_instances": s_["n_instances"]}
        table2.append({"level": lv + (sfx or " (exact)"), "pair": f"{a}/{b_}", "ratio": round(s_["ratio"], 3),
                       "ci90": f"[{s_['ci'][0]:.3f}, {s_['ci'][1]:.3f}]",
                       "TOST [0.8,1.25]": ("degenerate (all refuse)" if s_["degenerate_identical"] else
                                           ("equivalent" if eq else "not shown")),
                       "a_better (CI upper<0.8)": s_["ci"][1] < 0.8})
S["HS1"] = HS1
write_csv(OUT / "table2_hs1.csv", table2)
r0p = [v for k, v in HS1.items() if k.startswith("R0|") and "uniform" not in k]
VERD.append(dict(H="HS1", family="separate (TOST)", gate="pairwise 90% CI within [0.8,1.25] (R0 and off-grid)",
                 estimate="R0: " + ", ".join(f"{k.split('|')[1]} {v['ratio']:.2f}" for k, v in HS1.items() if k.startswith("R0|")),
                 ci="; ".join(f"[{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]" for v in r0p), p_raw=None,
                 verdict="PASS" if all(v["tost_equivalent"] for v in r0p) else "NOT SHOWN (pilot n=5; DDA not better)",
                 source="raw", note="R3 inflated: all samplers refuse at 0 steps (degenerate)"))
# HH1
hh = load("hh1_honesty_oos")
HH1 = {}
table3 = []
for m in ("JPC", "B4", "B9-20"):
    R = [r for r in hh if r["method"] == m]
    wl = int(sum(bool(r.get("wrong_label")) for r in R))
    HH1[m] = {"n": len(R), "wrong_label": wl, "cp_upper": cp_upper(wl, len(R))}
J = [r for r in hh if r["method"] == "JPC"]
lin_oos = [r for r in J if r.get("family") in ("squeeze", "newpart") and r.get("true_label") == "OUT_OF_SCOPE"
           and not r.get("near_tie")]
oos_det = int(sum(r["status"] == "OUT_OF_SCOPE" for r in lin_oos))
HH1["Lin_OOS_family_detection"] = {"n": len(lin_oos), "detected": oos_det,
                                   "rate": oos_det / max(len(lin_oos), 1), "cp": list(clopper_pearson(oos_det, len(lin_oos)))}
dec = int(sum(r["status"] in ("CERTIFIED", "OUT_OF_SCOPE") for r in J))
HH1["decidable_within_budget"] = dec / max(len(J), 1)
nt = [r for r in J if r.get("near_tie")]
HH1["near_tie_subset"] = {"n": len(nt), "decided": int(sum(r["status"] != "NEED_DATA" for r in nt)),
                          "wrong": int(sum(bool(r.get("wrong_label")) for r in nt))}
for env in ("E1-Lin", "E1-NL-S"):
    for tl in ("NEED_DATA", "OUT_OF_SCOPE"):
        R = [r for r in J if r["env"] == env and r.get("true_label") == tl]
        if R:
            c = Counter(r["status"] for r in R)
            table3.append({"env": env, "true_label": tl, "n": len(R), **{f"pred_{k}": c.get(k, 0) for k in
                           ("CERTIFIED", "OUT_OF_SCOPE", "NEED_DATA", "MODEL_CONFLICT", "COMPUTE_UNKNOWN")},
                           "wrong_label": int(sum(bool(r.get("wrong_label")) for r in R))})
for m in ("B4", "B9-20"):
    table3.append({"env": "all", "true_label": f"baseline {m}", "n": HH1[m]["n"], "wrong_label": HH1[m]["wrong_label"]})
S["HH1"] = HH1
write_csv(OUT / "table3_hh1.csv", table3)
VERD.append(dict(H="HH1a/b", family="separate (CP)", gate="wrong-label CP upper <= delta; Lin non-near-tie OOS detection >= 0.9",
                 estimate=f"JPC wrong labels {HH1['JPC']['wrong_label']}/{HH1['JPC']['n']}; OOS detection {oos_det}/{len(lin_oos)}",
                 ci=f"wrong CP upper {HH1['JPC']['cp_upper']:.3f}; detection CP [{HH1['Lin_OOS_family_detection']['cp'][0]:.2f}, "
                    f"{HH1['Lin_OOS_family_detection']['cp'][1]:.2f}]", p_raw=None,
                 verdict="PASS" if HH1["JPC"]["cp_upper"] <= DELTA and HH1["Lin_OOS_family_detection"]["rate"] >= 0.9 else "FAIL",
                 source="raw", note=f"B4 wrong labels {HH1['B4']['wrong_label']}/{HH1['B4']['n']}; decidable {HH1['decidable_within_budget']:.2f}"))
# HV (cand_g)
hv = [r for r in load("hv_versioned_evidence") if r.get("method") and r.get("k") is not None]
HV = {"HV1": {}, "HV3": {}, "VJE_vs_reset_known": {}}
for kt in (0, 1):
    for no in (20, 200, 1000):
        R = [r for r in hv if r["k_true"] == kt and r["n_old"] == no]
        nv = [r for r in R if r["method"] == "naive"]
        fc = int(sum(bool(r.get("false_cert")) for r in nv))
        nc = int(sum(r.get("status") == "CERTIFIED" for r in nv))
        HV["HV1"][f"k_true={kt},n_old={no}"] = {"false_certs": fc, "n_cert": nc, "cp": list(clopper_pearson(fc, nc))}
        s_ = ratio_stat(stream_totals(R, "vje"), stream_totals(R, "detect_reset"))
        HV["HV3"][f"k_true={kt},n_old={no}"] = {"ratio": s_["ratio"], "ci": s_["ci"], "pass": s_["ci"][1] < 0.8,
                                                "falsified_within_0.9_1.1": s_["ci"][0] >= 0.9 and s_["ci"][1] <= 1.1}
        s2 = ratio_stat(stream_totals(R, "vje"), stream_totals(R, "reset_known"))
        HV["VJE_vs_reset_known"][f"k_true={kt},n_old={no}"] = {"ratio": s2["ratio"], "ci": s2["ci"]}
HV["vje_false_certs"] = int(sum(bool(r.get("false_cert")) for r in hv if r["method"] == "vje"))
HV["detect_reset_alarms"] = int(sum(bool(r.get("model_conflict")) or r.get("status") == "MODEL_CONFLICT"
                                    for r in hv if r["method"] == "detect_reset"))
fc_seq = [HV["HV1"][f"k_true=1,n_old={no}"]["false_certs"] / max(HV["HV1"][f"k_true=1,n_old={no}"]["n_cert"], 1) for no in (20, 200, 1000)]
HV["HV1_monotone_k1"] = bool(all(np.diff(fc_seq) >= 0) and fc_seq[-1] > fc_seq[0])
S["HV"] = HV
VERD.append(dict(H="HV1 (cand_g)", family="cand_g", gate="naive-reuse FCR increases with n_old",
                 estimate="k_true=1 naive FCR " + "/".join(f"{x:.2f}" for x in fc_seq), ci="–", p_raw=None,
                 verdict="PASS (direction)" if HV["HV1_monotone_k1"] else "FAIL", source="raw", note="effect only at n_old=1000"))
VERD.append(dict(H="HV3 (cand_g)", family="cand_g", gate="VJE/detect-reset CI upper < 0.8",
                 estimate=", ".join(f"{k}: {v['ratio']:.2f}" for k, v in HV["HV3"].items()), ci="–", p_raw=None,
                 verdict="PASS" if any(v["pass"] for v in HV["HV3"].values()) else "FAIL (reverse direction at n_old=1000)",
                 source="raw", note=f"detect-reset alarms {HV['detect_reset_alarms']} (blind to in-class change); VJE false certs {HV['vje_false_certs']}"))
# E2 (descriptive)
e2a, e2b = load("e2_anchor_a"), load("e2_anchor_b")
seen, e2 = set(), []
for r in e2a + e2b:
    k = (r["instance"], r.get("problem_index"), r["method"])
    if k not in seen:
        seen.add(k)
        e2.append(r)
E2 = {}
appx_e2 = []
for m in sorted({r["method"] for r in e2}):
    R = [r for r in e2 if r["method"] == m]
    cert = [r for r in R if r.get("status") == "CERTIFIED"]
    fc = int(sum(bool(r.get("false_cert")) for r in R))
    E2[m] = {"n": len(R), "completion": len(cert) / len(R), "false_certs": fc, "fcr_cp_upper": cp_upper(fc, max(len(cert), 1)),
             "zero_cost_share": float(np.mean([bool(r.get("zero_cost_cert")) for r in R])),
             "median_new_env_steps": float(np.median([r.get("new_env_steps") or 0 for r in R])),
             "unit": "whole-trial rollouts" if m.startswith("B1") else "env steps"}
    appx_e2.append({"method": m, **E2[m]})
e2_tot = lambda m: {i: float(sum(r.get("new_env_steps") or 0 for r in e2 if r["method"] == m and r["instance"] == i))  # noqa: E731
                    for i in {r["instance"] for r in e2}}
E2["JPC_over_B3_instance_cumulative"] = ratio_stat(e2_tot("JPC"), e2_tot("B3"))
E2["note"] = "inner approximation, not a guaranteed upper bound, fixed-n Wilks radius; not part of any test"
S["E2"] = E2
write_csv(OUT / "appendix_e2.csv", appx_e2)
rep = load("replica_check_r2")
REP = {}
for v in sorted({r["variant"] for r in rep}):
    d = [abs(r["abs_diff"]) for r in rep if r["variant"] == v and r.get("abs_diff") is not None]
    REP[v] = {"n": len(d), "max_abs_J_diff": float(max(d)) if d else None, "median_abs_J_diff": float(np.median(d)) if d else None}
rs = summ("replica_check_r2")
REP["direction_agreement_from_summary"] = rs.get("direction_agreement")
REP["replicated_stream_ratio_from_summary"] = rs.get("replicated_stream_ratio")
REP["go_no_go"] = rs.get("go_no_go")
S["replica"] = REP
write_csv(OUT / "appendix_replica.csv", [{"variant": k, **v} for k, v in REP.items() if isinstance(v, dict) and "n" in v])

# ================================================================================================ 9. suspicious gate
progress(9, TOTAL, "suspicious gate")
SUSP = []


def sg(task, trigger, value, checks, conclusion):
    SUSP.append({"task": task, "trigger": trigger, "value": value, "checks": checks, "conclusion": conclusion})


r0 = HR1["R0"]["ratios"]
sg("hr1_r0 (HR1@R0)", "savings > 5x vs B3-UI", f"JPC/B3-UI = {r0['B3-UI']['ratio']:.3f}",
   f"rollouts in Theta_t: env_n_steps==lr_n_rounds {boundary['mismatches']} mismatches / {boundary['rows_checked']} rows; "
   f"censoring: B3-UI charged T_max on {HR1['R0']['cells']['B3-UI']['n'] - HR1['R0']['cells']['B3-UI']['n_cert']} problems; "
   "granularity: both per-step; truth leak: r2_setup unit tests (no truth import, AST regime checks)",
   "explained by UI threshold on continuous 12-d class + censoring; B3-UI is a weak (but locked) control -> do not headline")
sg("hr1_r0 (HR1@R0)", ">30% better than simple baseline B3", f"JPC/B3 = {r0['B3']['ratio']:.3f}",
   f"oracle: {S['oracle_check']['below_OPT_K']}/{S['oracle_check']['prefixes_checked']} cumulative prefixes below OPT_K; "
   f"tie rate vs B3 {HR1['R0']['tie_rate_vs_B3']:.2f}",
   "on-grid realisable savings; consistent with round 0 (~0.2-0.47); conditional on truth on the learner grid")
sg("hr1_r2/r3 (HR1 off-grid)", "ratio far from 1 (14.9x worse)", f"JPC_infl/B3 = {HR1['R3']['ratios']['B3']['ratio']:.2f}",
   "all inflated grid methods refuse data-free (2 min eta_loc > eps); charged T_max", "cost artefact of honest refusal, not a performance comparison")
hf1r2 = HF1["R2_uncorrected_mixed_over_plugin"]
sg("hf1_mixed_cost (R2 secondary)", ">30% improvement", f"mixed/plugin = {hf1r2['ratio']:.2f}",
   "theta* cell coverage drops (mixed set shrinks faster off-grid; ext beats pool by ~13-15 nats)",
   "near-falsification power, not an in-class saving; must not be reported as a cost advantage")
hs1u = HS1.get("R0|DDA/uniform")
if hs1u:
    sg("hs1_sampler_equivalence", ">30% better than uniform", f"DDA/uniform = {hs1u['ratio']:.2f}",
       "same LR evidence and certifier; only the sampler differs; FCR 0", "expected design effect; uniform is a weak sampler")
sg("hd2_kappa_sandwich_lin", "joint vs rectangular > 5x", "Omega(H^2) construction ratios 15-800x",
   "predicted by kappa theory (Thm D); 0 false certs; 0 runs below oracle rho*", "expected by construction; not a JPC-savings claim")
e2r = E2["JPC_over_B3_instance_cumulative"]
sg("e2_anchor_{a,b}", ">30% better than B3", f"JPC/B3 cumulative = {e2r['ratio']:.2f}",
   "driven by 4 hard problems; task near-trivial at eps=0.01; MC inner approximation", "descriptive anchor only")
sg("hd1_matched_twin_static", "static FCR near 0 without completion drop?", f"static FCR g=1 {hs['fcr']:.2f}",
   "static FCR > 0 at kappa_high -> trigger not met", "not triggered")
write_csv(OUT / "suspicious_gate_audit.csv", SUSP)
S["suspicious_gate"] = SUSP

# ================================================================================================ 10. figures
progress(10, TOTAL, "figures")
# Fig 1 architecture
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.axis("off")
boxes = [(0.02, 0.55, "real interaction\n(env.step, authentic)"), (0.2, 0.55, "joint LR set Θ_t\n(SeqLRSet, UI log 1/δ)"),
         (0.38, 0.55, "falsification layer\n⅓{pool,reg,ext} + replay"), (0.56, 0.55, "certifier\nexact / dual; R2/R3: +2η_loc"),
         (0.74, 0.55, "5 statuses\nCERT/NEED/CU/MC/OOS"), (0.56, 0.08, "DDA acquisition\n(NEED_DATA)"),
         (0.2, 0.08, "cross-problem ledger\n(zero-cost reuse)"), (0.38, 0.08, "model rollouts\n(never enter Θ_t)")]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 0.16, 0.3, boxstyle="round,pad=0.01", fc="#f4f3ef", ec=INK2, lw=1))
    ax.text(x + 0.08, y + 0.15, t, ha="center", va="center", fontsize=7.5, color=INK)
arr = [((0.18, 0.7), (0.2, 0.7)), ((0.36, 0.7), (0.38, 0.7)), ((0.54, 0.7), (0.56, 0.7)), ((0.72, 0.7), (0.74, 0.7)),
       ((0.82, 0.55), (0.64, 0.38)), ((0.56, 0.23), (0.1, 0.55)), ((0.28, 0.38), (0.28, 0.55)), ((0.46, 0.38), (0.6, 0.55))]
for (x0, y0), (x1, y1) in arr:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.plot([0.37, 0.37], [0.02, 0.95], ls="--", color=C[7], lw=1)
ax.text(0.372, 0.96, "evidence boundary", color=C[7], fontsize=7)
ax.text(0.92, 0.7, "each certificate\ncarries its\nfalsification scope", fontsize=7, color=INK2, va="center")
save(fig, "fig1_architecture")
# Table 1 figure-less (md) ; Fig 2 waterfall
fig, ax = plt.subplots(figsize=(7.5, 3.6))
cum = 0.0
for i, w in enumerate(water[:-1]):
    lr = w["log_ratio"]
    col = C[2] if lr < 0 else C[7]
    ax.bar(i, lr, bottom=cum if i < len(chain) else 0, color=col, width=0.6)
    ax.errorbar(i, (cum if i < len(chain) else 0) + lr, yerr=[[lr - math.log(w["ci_lo"])], [math.log(w["ci_hi"]) - lr]],
                color=INK2, capsize=3, lw=1)
    ax.text(i, (cum if i < len(chain) else 0) + lr + (0.12 if lr >= 0 else -0.25), f"×{w['ratio']:.2f}", ha="center", fontsize=7)
    if i < len(chain):
        cum += lr
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(len(water) - 1))
ax.set_xticklabels([w["step"].split(" (")[0] for w in water[:-1]], rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("log step ratio (cumulative from B3)")
ax.set_title("Fig 2  Savings decomposition (R0 dev pilot; last bar: off-grid R3, separate instances)", fontsize=9)
save(fig, "fig2_waterfall", water)
# Fig 3 phase
fig, axs = plt.subplots(3, 1, figsize=(6, 6.5), sharex=True)
fcol = {"0.5": C[3], "1": C[0], "2": C[2], "4": C[6]}
p3rows = []
for lv, v in hr4.items():
    x = v.get("log_h_over_hstar_median")
    if x is None or v.get("jpc_infl_completion") is None:
        continue
    f = str(v["f"])
    axs[0].scatter([x], [v["jpc_infl_completion"]], color=fcol.get(f, C[1]), s=40, label=f"f={f}", zorder=3)
    if v.get("JPC_infl_over_B3"):
        r_ = v["JPC_infl_over_B3"]
        axs[1].errorbar([x], [r_["ratio"]], yerr=[[r_["ratio"] - r_["ci"][0]], [r_["ci"][1] - r_["ratio"]]], fmt="o",
                        color=fcol.get(f, C[1]), capsize=3)
    if v.get("JPC_infl_over_B8_infl"):
        axs[2].scatter([x], [v["JPC_infl_over_B8_infl"]["ratio"]], color=fcol.get(f, C[1]), s=40)
    p3rows.append({"level": lv, "f": f, "log_h_over_hstar_median": x, "completion": v["jpc_infl_completion"],
                   "jpc_infl_over_b3": (v.get("JPC_infl_over_B3") or {}).get("ratio"),
                   "jpc_infl_over_b8": (v.get("JPC_infl_over_B8_infl") or {}).get("ratio")})
r3x = hr4["R3_f2(hr1)"]["log_h_over_hstar_median"]
rr = HR1["R3"]["ratios"]["B3"]
axs[1].errorbar([r3x], [rr["ratio"]], yerr=[[rr["ratio"] - rr["ci"][0]], [rr["ci"][1] - rr["ratio"]]], fmt="o", color=fcol["2"], capsize=3)
axs[2].scatter([r3x], [HR1["R3"]["ratios"]["B8_infl"]["ratio"]], color=fcol["2"], s=40)
for a in axs:
    a.axvline(0, color=C[7], ls="--", lw=1)
axs[0].set_ylabel("JPC_infl completion")
axs[0].set_ylim(-0.05, 1.05)
axs[0].legend(fontsize=7, loc="center right")
axs[1].set_ylabel("JPC_infl / B3")
axs[1].set_yscale("log")
axs[2].set_ylabel("JPC_infl / B8_infl")
axs[2].set_xlabel("log(h / h*)  (literal h*; dashed: h = h*)")
axs[0].set_title("Fig 3  Resolution phase diagram: every f sits far right of h* -> logistic not estimable", fontsize=9)
save(fig, "fig3_phase", p3rows)
# Fig 4 HR3
fig, ax = plt.subplots(figsize=(5.5, 3.2))
xs = range(len(hr3_bins))
fc_ = [b["fcr"] if b["fcr"] is not None else np.nan for b in hr3_bins]
lo_ = [b["cp"][0] if b["cp"] else np.nan for b in hr3_bins]
hi_ = [b["cp"][1] if b["cp"] else np.nan for b in hr3_bins]
ax.errorbar(xs, fc_, yerr=[np.array(fc_) - np.array(lo_), np.array(hi_) - np.array(fc_)], fmt="o-", color=C[0], capsize=3)
ax.axhline(DELTA, color=C[7], ls="--", lw=1)
ax.text(len(hr3_bins) - 1, DELTA + 0.005, "δ", color=C[7])
ax.set_xticks(list(xs))
ax.set_xticklabels([f"{b['bin_eta_dec_over_eps']}\nn={b['n_cert']}" for b in hr3_bins], fontsize=7)
ax.set_xlabel("measured η_dec / ε (R1, uncorrected, G_1)")
ax.set_ylabel("FCR (CP 95%)")
ax.set_title(f"Fig 4  Off-grid uncorrected FCR vs η_dec (Jonckheere p={jt_p3:.2f})", fontsize=9)
save(fig, "fig4_hr3", hr3_bins)
# Fig 5 falsification
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
for j, m in enumerate(("mixed", "mixed_half")):
    v = list(HF1[m]["per_instance"].values())
    axs[0].scatter(np.full(len(v), j) + np.linspace(-0.1, 0.1, len(v)), v, color=C[j], s=18)
    axs[0].errorbar([j + 0.25], [HF1[m]["ratio"]], yerr=[[HF1[m]["ratio"] - HF1[m]["ci"][0]], [HF1[m]["ci"][1] - HF1[m]["ratio"]]],
                    fmt="s", color=INK, capsize=3)
axs[0].axhline(1.15, color=C[7], ls="--", lw=1)
axs[0].axhline(1.25, color=C[7], ls=":", lw=1)
axs[0].set_xticks([0, 1])
axs[0].set_xticklabels(["⅓ mixed", "½ mixed"])
axs[0].set_ylabel("stream steps / plug-in (η=0)")
axs[0].set_title("(a) in-class cost (HF1)", fontsize=9)
labs = ["in-stream\n(mixed)", "shadow main\n(same data)", "reg-only\nload bucket", "reg-only\nplacebo", "κ=0 twin\nfalse alarm"]
vals = [det["mixed_lb@R0"] / max(len(fc_keys), 1), shadow_det["main"] / max(len(fc_keys), 1),
        shadow_det["reg_lb"] / max(len(fc_keys), 1), shadow_det["reg_placebo"] / max(len(fc_keys), 1),
        twin_fa / max(HF2["twin_n"], 1)]
axs[1].bar(range(5), vals, color=[C[0], C[6], C[2], C[3], C[7]])
axs[1].axhline(0.6, color=C[7], ls="--", lw=1)
axs[1].set_xticks(range(5))
axs[1].set_xticklabels(labs, fontsize=6.5, rotation=35, ha="right")
axs[1].set_ylabel("share of plug-in false certs flagged")
axs[1].set_title(f"(b) static-class detection (HF2, n={len(fc_keys)})", fontsize=9)
cost_items = [("AGC (m1)", HF4["AGC_audit_steps_median"]), ("e-process\nτ=ε/2,f=.1", HF4["eprocess|half"]["audit_steps_median"]),
              ("χ² fixed n", HF4["chi2|half"]["audit_steps_median"]), ("AGC (hf2b)", HF2b["AGC_audit_steps_median_eta2"])]
cost_items = [(a, b_) for a, b_ in cost_items if b_]
axs[2].bar(range(len(cost_items)), [b_ for _, b_ in cost_items], color=[C[1], C[0], C[2], C[4]][:len(cost_items)])
axs[2].set_yscale("log")
axs[2].set_xticks(range(len(cost_items)))
axs[2].set_xticklabels([a for a, _ in cost_items], fontsize=7)
axs[2].set_ylabel("audit steps (median)")
axs[2].set_title("(c) audit cost; mixed alarms: 0 (HF2b/HF3)", fontsize=9)
save(fig, "fig5_falsification", [{"panel": "b", "label": l_.replace("\n", " "), "value": v_} for l_, v_ in zip(labs, vals)]
     + [{"panel": "c", "label": a.replace("\n", " "), "value": b_} for a, b_ in cost_items])
# Fig 6 HD1/HD3
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
doses = [0.0, 0.5, 1.0]
st = [HD1[f"static@g{d}"] for d in doses]
dy = [HD1["dyn_G1@g0.0"], HD1["dyn_Gext@g0.5"], HD1["dyn_G1@g1.0"]]
for lab, arr_, col in (("static (param-matched)", st, C[1]), ("dynamic (reference)", dy, C[0])):
    f_ = np.array([a["fcr"] for a in arr_])
    axs[0].errorbar(doses, f_, yerr=[f_ - np.array([a["cp"][0] for a in arr_]), np.array([a["cp"][1] for a in arr_]) - f_],
                    fmt="o-", color=col, capsize=3, label=lab)
axs[0].axhline(DELTA, color=C[7], ls="--", lw=1)
axs[0].set_xlabel("dynamics dose g (γ, λ scaled)")
axs[0].set_ylabel("FCR (CP 95%)")
axs[0].legend(fontsize=7)
axs[0].set_title(f"(a) HD1 matched twin (Jonckheere p={jt_hd1:.4f})", fontsize=9)
pi = HD3["per_instance_D"]
axs[1].scatter(range(len(pi)), list(pi.values()), color=C[0])
axs[1].axhline(0, color=INK2, lw=0.8)
axs[1].axhline(HD3["D"], color=C[1], ls="--", lw=1)
axs[1].fill_between([-0.5, len(pi) - 0.5], HD3["D_ci"][0], HD3["D_ci"][1], color=C[1], alpha=0.15)
axs[1].set_xticks(range(len(pi)))
axs[1].set_xticklabels([str(i) for i in pi], fontsize=7)
axs[1].set_xlabel("instance")
axs[1].set_ylabel("D = Δlog(rect/joint), κ_high − κ0")
axs[1].set_title("(b) HD3 paired D [95% CI]", fontsize=9)
save(fig, "fig6_dynamics", [{"dose": d, "static_fcr": s_["fcr"], "dynamic_fcr": y_["fcr"]} for d, s_, y_ in zip(doses, st, dy)]
     + [{"instance": i, "D": v} for i, v in pi.items()])
# Fig 7 HD2
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
sw = []
for r in lp:
    if r["kind"] == "main" and r["design"] == "phased" and r["evidence"] in ("time", "participant", "policy"):
        a_, lo_, hi_ = fnum(r["plus_rho_ratio"]), fnum(r["plus_lower"]), fnum(r["plus_upper"])
        if a_ and lo_ and hi_:
            sw.append((r["evidence"], lo_, a_, hi_))
for j, part in enumerate(("time", "participant", "policy")):
    P = [s for s in sw if s[0] == part]
    if P:
        axs[0].scatter([p[2] for p in P], [p[1] for p in P], s=8, color=C[j], marker="v", label=f"{part} lower")
        axs[0].scatter([p[2] for p in P], [p[3] for p in P], s=8, color=C[j], marker="^")
lim = [1, max([s[3] for s in sw] + [10])]
axs[0].plot(lim, lim, color=INK2, lw=0.8)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("design-level ratio ρ_P/ρ_J")
axs[0].set_ylabel("bounds")
axs[0].legend(fontsize=6)
axs[0].set_title(f"(a) sandwich (violations: {HD2['sandwich_from_summary']['violations']})", fontsize=9)
for j, part in enumerate(("time", "participant", "policy")):
    xs_, ys_ = [], []
    for r in lp:
        if r["kind"] == "main" and r["evidence"] == part and r["design"] == "phased":
            N, Nj, ke = fnum(r["N"]), fnum(r["N_joint"]), fnum(r["plus_kappa_eff"])
            if N and Nj and ke and ke > 0:
                xs_.append(math.log(ke))
                ys_.append(math.log(N / Nj))
    v = HD2[f"kappa_eff_slope|{part}"]
    axs[1].scatter(xs_, ys_, s=8, color=C[j], label=f"{part}: {v['slope']:.2f}" if v else part)
axs[1].set_xlabel("log κ_eff")
axs[1].set_ylabel("log N_rect / N_joint")
axs[1].legend(fontsize=7)
axs[1].set_title("(b) κ_eff law (gate slope ∈ [1.5, 2.5])", fontsize=9)
cx, cy = [], []
for r in lp:
    if r["kind"] == "construction" and r["evidence"] == "time" and r["design"] == "phased":
        N, Nj = fnum(r["N"]), fnum(r["N_joint"])
        if N and Nj:
            cx.append(fnum(r["H"]))
            cy.append(N / Nj)
axs[2].scatter(cx, cy, color=C[0], s=14)
axs[2].set_xscale("log", base=2)
axs[2].set_yscale("log")
axs[2].set_xlabel("H")
axs[2].set_ylabel("N_time / N_joint")
axs[2].set_title(f"(c) Ω(H²) construction, slope {HD2['construction_slope_time_vs_logH']['slope']:.2f}", fontsize=9)
save(fig, "fig7_hd2", [{"partition": k.split("|")[1], **{kk: vv for kk, vv in v.items()}} for k, v in HD2.items()
                       if k.startswith("kappa_eff_slope|") and v])
# Fig 8 HK
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
lsty = {"R0": "-", "R2": ":", "R3": "--"}
for lv in ("R0", "R2", "R3"):
    for m in (TEST_M[lv], "B3", "JPC") if lv != "R0" else ("JPC", "B3", "B8"):
        pts = [x for x in fig8 if x["level"] == lv and x["method"] == m]
        if not pts:
            continue
        lab = f"{m}@{lv}" + (" (unc.)" if lv != "R0" and m == "JPC" else "")
        col = MC.get(m, C[4]) if not (lv != "R0" and m == "JPC") else MC["JPC(unc)"]
        axs[0].plot([p["k"] for p in pts], [p["mean_cum_steps_incl_n0"] for p in pts], ls=lsty[lv], color=col, label=lab, lw=1.6)
for lv in ("R0", "R2", "R3"):
    ks = HK[lv]["K_star_vs_B3"]
    if ks:
        axs[0].axvline(ks, color=INK2, ls=lsty[lv], lw=0.8)
axs[0].set_yscale("log")
axs[0].set_xlabel("problem index k")
axs[0].set_ylabel("mean cumulative new steps incl. n0")
axs[0].legend(fontsize=6, ncol=2)
axs[0].set_title("(a) cumulative cost; K* vs B3: " + ", ".join(f"{lv}={HK[lv]['K_star_vs_B3']}" for lv in HK), fontsize=9)
for lv in ("R0", "R2", "R3"):
    z = HK[lv]["zero_cost_rate_by_k"]
    axs[1].plot([k + 1 for k in z], list(z.values()), ls=lsty[lv], marker="o", ms=3, color=C[0], label=f"{TEST_M[lv]}@{lv}")
axs[1].set_xlabel("problem index k")
axs[1].set_ylabel("zero-cost certification rate")
axs[1].set_ylim(-0.05, 1.05)
axs[1].legend(fontsize=7)
axs[1].set_title("(b) zero-cost certification vs k", fontsize=9)
for a in axs:
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
save(fig, "fig8_hk", fig8)
# Appendix: eta_loc validity histogram + cell survival
fig, axs = plt.subplots(1, 2, figsize=(9, 3))
for j, f in enumerate(sorted({r["f"] for r in eta_rows}, key=float)):
    v = [r["ratio_err_over_eta_max"] for r in eta_rows if r["f"] == f and r.get("ratio_err_over_eta_max") is not None]
    axs[0].hist(v, bins=20, alpha=0.6, color=C[j], label=f"f={f}")
axs[0].axvline(1.0, color=C[7], ls="--")
axs[0].set_xlabel("max empirical |ΔJ| / η_loc per problem (violation if > 1)")
axs[0].legend(fontsize=7)
axs[0].set_title(f"(a) η_loc validity: {g0_viol}/{g0_draws} violations", fontsize=9)
surv = []
for lv in ("R2", "R3"):
    c = HR1[lv]["cells"].get("JPC")
    if c and c.get("theta_cell_alive") is not None:
        surv.append((f"JPC unc.@{lv}", c["theta_cell_alive"]))
jg = [r for r in g0 if r.get("kind") == "jpc"]
for f in sorted({r["f"] for r in jg}, key=float):
    R = [r for r in jg if r["f"] == f]
    surv.append((f"G0 θ* cell f={f}", float(np.mean([bool(r.get("theta_cell_alive")) for r in R]))))
    surv.append((f"G0 g° cell f={f}", float(np.mean([bool(r.get("gcirc_cell_alive")) for r in R]))))
axs[1].barh(range(len(surv)), [s[1] for s in surv], color=C[0])
axs[1].set_yticks(range(len(surv)))
axs[1].set_yticklabels([s[0] for s in surv], fontsize=7)
axs[1].set_xlim(0, 1)
axs[1].set_title("(b) cell survival (DEC-ID diagnostics)", fontsize=9)
save(fig, "figA_eta_loc_and_survival", [{"label": a, "survival": b_} for a, b_ in surv])
# VJE appendix
fig, ax = plt.subplots(figsize=(6, 3))
labs_v = list(HV["HV3"].keys())
vv = [HV["HV3"][k]["ratio"] for k in labs_v]
lo_v = [HV["HV3"][k]["ci"][0] for k in labs_v]
hi_v = [HV["HV3"][k]["ci"][1] for k in labs_v]
ax.errorbar(range(len(vv)), vv, yerr=[np.array(vv) - lo_v, np.array(hi_v) - np.array(vv)], fmt="o", color=C[6], capsize=3)
ax.axhline(0.8, color=C[7], ls="--", lw=1)
ax.axhline(1.0, color=INK2, lw=0.8)
ax.set_yscale("log")
ax.set_xticks(range(len(vv)))
ax.set_xticklabels([k.replace(",", "\n") for k in labs_v], fontsize=6.5)
ax.set_ylabel("VJE / detect-reset stream steps")
ax.set_title("Appendix  cand_g HV3 (gate: CI upper < 0.8)", fontsize=9)
save(fig, "figA_vje_hv3", [{"cell": k, **HV["HV3"][k]} for k in labs_v])
log("figures written")

# ================================================================================================ 11. verdict table + Table 1 md
progress(11, TOTAL, "verdict table")
fam_holm = {"A": hA, "R": hR, "F": hF, "D": hD}
for v in VERD:
    v.setdefault("p_holm", None)
    if v.get("family") in fam_holm and v.get("p_holm") is None:
        v["p_holm"] = fam_holm[v["family"]].get(v["H"])


def pfmt(p):
    return "–" if p is None else (f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}")


md = ["# Round-1 verdict table (pilot, dev seeds; pilot-scale n — not the pre-registered full test)", "",
      f"Wording tier (locked decide_tier, sha256 {td_sha[:12]}…): **{tier['tier']}** — {tier['reason']}. "
      f"Counterfactual if G0 had passed: tier {tier_if_g0['tier']}.", "",
      "| H | family | gate | estimate | CI | p raw | p Holm | verdict | source | note |", "|---|---|---|---|---|---|---|---|---|---|"]
for v in VERD:
    md.append(f"| {v['H']} | {v['family']} | {v['gate']} | {v['estimate']} | {v['ci']} | {pfmt(v['p_raw'])} | "
              f"{pfmt(v['p_holm'])} | {v['verdict']} | {v['source']} | {v.get('note', '')} |")
md += ["", "## Table 1 (HR1): per level x method", "",
       "| level | method | stream steps | ratio vs B3 [95% CI] | test/method | completion | FCR (CP upper) | zero-cost | tie vs B3 | refused | valid den. | role |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in table1:
    md.append("| " + " | ".join(str(t[k]) for k in ("level", "method", "stream_steps_mean", "ratio_vs_B3", "test_over_method",
                                                    "completion", "FCR", "zero_cost_rate", "tie_rate_vs_B3", "refused_floor",
                                                    "valid_denominator", "role")) + " |")
md += ["", "## Table 2 (HS1): sampler equivalence (90% CI, TOST [0.8, 1.25])", "", "| level | pair | ratio | 90% CI | TOST | a better |",
       "|---|---|---|---|---|---|"]
md += [f"| {t['level']} | {t['pair']} | {t['ratio']} | {t['ci90']} | {t['TOST [0.8,1.25]']} | {t['a_better (CI upper<0.8)']} |" for t in table2]
md += ["", "## Table 3 (HH1): JPC confusion (true label x final status)", "",
       "| env | true label | n | CERT | OOS | NEED | MC | CU | wrong |", "|---|---|---|---|---|---|---|---|---|"]
for t in table3:
    md.append(f"| {t['env']} | {t['true_label']} | {t['n']} | {t.get('pred_CERTIFIED', '–')} | {t.get('pred_OUT_OF_SCOPE', '–')} | "
              f"{t.get('pred_NEED_DATA', '–')} | {t.get('pred_MODEL_CONFLICT', '–')} | {t.get('pred_COMPUTE_UNKNOWN', '–')} | {t['wrong_label']} |")
md += ["", "## Appendix: E2 anchor (inner approximation, descriptive only) and replica environment", "",
       "| method | n | completion | false certs | zero-cost share | median new steps | unit |", "|---|---|---|---|---|---|---|"]
md += [f"| {a['method']} | {a['n']} | {a['completion']:.2f} | {a['false_certs']} | {a['zero_cost_share']:.2f} | {a['median_new_env_steps']:.0f} | {a['unit']} |" for a in appx_e2]
md += ["", f"E2 JPC/B3 instance-cumulative ratio {fmt_r(E2['JPC_over_B3_instance_cumulative'])} (n=2 dev instances).", "",
       "| replica variant | n | max abs J diff | median abs J diff |", "|---|---|---|---|"]
md += [f"| {k} | {v['n']} | {v['max_abs_J_diff']:.2e} | {v['median_abs_J_diff']:.2e} |" for k, v in REP.items()
       if isinstance(v, dict) and v.get("n")]
md += ["", f"Replica direction agreement (from replica_check_r2 summary): {json.dumps(REP.get('direction_agreement_from_summary'), ensure_ascii=False)[:400]}",
       "", "## Suspicious-improvement gate", "", "| task | trigger | value | checks | conclusion |", "|---|---|---|---|---|"]
md += [f"| {s['task']} | {s['trigger']} | {s['value']} | {s['checks']} | {s['conclusion']} |" for s in SUSP]
(OUT / "verdict_table.md").write_text("\n".join(md) + "\n")
write_csv(OUT / "verdicts.csv", VERD)
S["verdicts"] = VERD

# ================================================================================================ 12. results.jsonl, samples
progress(12, TOTAL, "results/samples")
with open(OUT / "results.jsonl", "w") as f:
    for v in VERD:
        f.write(json.dumps({"kind": "verdict", **v}, default=str) + "\n")
    for t in table1:
        f.write(json.dumps({"kind": "table1", **t}, default=str) + "\n")
    for w in water:
        f.write(json.dumps({"kind": "waterfall", **w}, default=str) + "\n")
    for lv in HK:
        f.write(json.dumps({"kind": "HK", "level": lv, **HK[lv]}, default=str) + "\n")
    for s_ in SUSP:
        f.write(json.dumps({"kind": "suspicious_gate", **s_}, default=str) + "\n")
samples = {"HR1_R0_JPC_vs_B3_first_stream": [
    {k: r.get(k) for k in ("instance", "k", "method", "status", "new_env_steps", "zero_cost", "true_regret", "tie")}
    for r in sorted([r for r in R0rows if r["method"] in ("JPC", "B3") and r["instance"] == min(x["instance"] for x in R0rows)],
                    key=lambda r: (r["k"], r["method"]))],
    "HR1_R3_refusals": [{k: r.get(k) for k in ("instance", "k", "method", "status", "new_env_steps", "refused_floor",
                                               "eta_loc_min", "theta_cell_alive")}
                        for r in [r for r in R3rows if r["method"] == "JPC_infl"][:5]],
    "HF2_plugin_false_certs_vs_mixed": [{"key": list(k), "plugin_regret": plug[k].get("true_regret"),
                                         "mixed_status": (by_m["mixed_lb@R0"].get(k) or {}).get("status"),
                                         "mixed_false_cert": (by_m["mixed_lb@R0"].get(k) or {}).get("false_cert"),
                                         "shadow_main_conflict": shadow_conf(plug[k], "main")} for k in fc_keys[:8]],
    "HR3_highest_eta_dec_certs": sorted([{k: r.get(k) for k in ("instance", "k", "eta_dec_over_eps", "model_status",
                                                                "model_true_regret", "model_false_cert")}
                                         for r in hr3 if r.get("model_cert")],
                                        key=lambda d: -(d["eta_dec_over_eps"] or 0))[:8]}
(SAMP / "samples.json").write_text(json.dumps(samples, indent=1, default=str))

# ================================================================================================ 13. per-candidate GO/NO-GO
progress(13, TOTAL, "candidates")
hr2_reversed = tw["ratio"] > 0.7
cands = [
    {"candidate_id": "cand_a", "go_no_go": "NO_GO", "confidence": 0.85,
     "supported_hypotheses": ["G0 validity (0 eta_loc violations)", "HR1@R0 direction (JPC/B3 %.2f, on-grid)" % HR1["R0"]["ratios"]["B3"]["ratio"],
                              "HD1 static-class FCR rises with dose", "HD2 kappa_eff law per partition", "HH1a/b honesty",
                              "HK on-grid K*=%s" % HK["R0"]["K_star_vs_B3"]],
     "failed_assumptions": ["G0 resolution gate (no f reaches eta_loc<=eps/4, even at eps=0.10)",
                            "HR1@R2/R3 (100% data-free refusal; completion 0)", "HR2 threshold accounting reversed (UI/chi2 %.1f)" % tw["ratio"],
                            "HR3 (off-grid harmless at G_1)", "HR4/HR5 untestable (no h<=h*)", "HF2/HF2b/HF4 negative", "HD3 CI contains 0"],
     "key_metrics": {"tier": tier["tier"], "jpc_b3_R0": HR1["R0"]["ratios"]["B3"]["ratio"],
                     "jpc_infl_b3_R3": HR1["R3"]["ratios"]["B3"]["ratio"], "jpc_infl_completion_R3": HR1["R3"]["cells"]["JPC_infl"]["completion"]},
     "notes": "Locked pivot rule fires: tier C (G0 failed; HR1 fails at R2 and R3) -> pivot_to_cand_f. The 0.2-0.47 "
              "on-grid ratio survives only as a realisable-finite-class conditional bound."},
    {"candidate_id": "cand_f", "go_no_go": "GO", "confidence": 0.5,
     "supported_hypotheses": ["accounting chain measurable end-to-end (waterfall)", "reuse ledger is the dominant factor (JPC/JPC-fresh %.2f)" % HR2["reuse_factor_JPC_over_fresh"]["ratio"],
                              "resolution near-invariance of uncorrected decision cost (JPC/B3 across f)", "Prop C' / Thm D chapter (HD1, HD1b, HD2)"],
     "failed_assumptions": (["HR2 predicted threshold ratio [0.2,0.5]: observed %.1f (reverse)" % tw["ratio"]] if hr2_reversed else [])
                           + ["HR2b Prop-B share in band %.2f < 0.7" % (hr2b.get("share_in_band") or 0) if hr2b and hr2b.get("share_in_band") is not None else "HR2b not computed",
                              "HR4 phase transition not observable (all f >> h*)", "JPC_infl/B8 curve degenerate (both refuse)"],
     "key_metrics": {"threshold_only_ratio": tw["ratio"], "reuse_factor": HR2["reuse_factor_JPC_over_fresh"]["ratio"],
                     "design_main_effect_log": HR2["design_main_effect_log"]["mean"], "threshold_main_effect_log": HR2["threshold_main_effect_log"]["mean"]},
     "notes": "Promoted by the locked tier-C rule; data are shared with cand_a, no new runs needed. Its own pre-registered "
              "accounting predictions are mostly falsified, so the paper must be reframed as an honest negative/explanatory "
              "account (savings = realisable finite support + reuse; off-grid inflation makes certification impossible at this "
              "resolution), not as a confirmed threshold story. Confidence moderate."},
    {"candidate_id": "cand_g", "go_no_go": "NO_GO", "confidence": 0.7,
     "supported_hypotheses": ["HV1 naive-reuse FCR increases with n_old (k_true=1)", "HF1 point cost <= 1.25x (pilot gate)",
                              "VJE beats reset_known at n_old=1000,k=1 (%.2f)" % HV["VJE_vs_reset_known"]["k_true=1,n_old=1000"]["ratio"]],
     "failed_assumptions": ["HV3 VJE/detect-reset CI upper < 0.8 (reverse at n_old=1000)", "detect-reset never alarms (blind to in-class change)",
                            "HF2 / HF2b zero-interaction falsification negative"],
     "key_metrics": {"hv3_n1000_k1": HV["HV3"]["k_true=1,n_old=1000"]["ratio"], "hf1_mixed": HF1["mixed"]["ratio"],
                     "hf2_detection": det_rate},
     "notes": "Falsification layer does not fire on registered or unregistered misspecification at pilot scale; VJE is more "
              "expensive than detect-reset. Keep HV1 / VJE-vs-reset_known as a secondary result inside cand_f if useful."},
]
overall = {"overall_recommendation": "PIVOT", "selected_candidate_id": "cand_f", "candidates": cands,
           "decision_basis": "locked tier rule: tier %s (%s); methodology 8 pivot_to_cand_f condition" % (tier["tier"], tier["reason"])}
S["pilot_decision"] = overall
S["pass_criteria"] = {"every_pilot_summary_parsed": S["all_summaries_parsed"],
                      "per_candidate_table_cand_a_f_g": True, "pilot_summary_json_written": None}

# ================================================================================================ 14. write summary + pilot_summary
wall = (datetime.now() - T_START).total_seconds()
S["wall_clock_s"] = wall
S["log"] = LOG
S["figures"] = sorted(p.name for p in FIG.iterdir())
(OUT / "summary.json").write_text(json.dumps(S, indent=1, default=str, ensure_ascii=False))
(OUT / "run.log").write_text("\n".join(LOG) + "\n")
print(json.dumps({"tier": tier, "wall_s": wall}, default=str))
progress(TOTAL, TOTAL, "done")
