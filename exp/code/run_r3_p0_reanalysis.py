"""r3_p0_reanalysis: EXPLORATORY (post-hoc) P0 re-analysis of the round-3 full results.

Status: EXPLORATORY / post-hoc. Nothing here is confirmatory; nothing here changes the locked Tier C verdict
(plan/prereg_lock.json v3, exp/results/full/r3_analysis_aggregate). It runs NO environment interactions and draws NO
new seeds: it only re-reads existing results.jsonl / predictors.jsonl / summary.json files.

Items (reviews/result_debate_review.md section 5, idea/result_debate/verdict.md external reviewer addendum):
  1  metric-consistency table for headline numbers in writing/outline.md (estimand, layer, tau, sample)
  2  orth on constructible-only problems: log(orth/off), log(orth/vol) excluding ORTH_INFEASIBLE / trivial
  3  T0: computable Rem bound / eps p90 vs oracle exact |Rem| / eps p90 (dev seeds 720-739; r3_p4)
  4  HD2: per-partition kappa_eff slopes vs beta-corrected slope; which gate failed
  5  Q1: AUC of frozen S2 and raw S1 (Lambda_perp) with instance-cluster bootstrap; positives in eta_dec <= eps
  6  Q2 restricted: static_open rows only (predictors recorded at problem open), cluster-robust; GLOBAL placebo
  7  Q3 error model refit controlling for estimated gap (-log gap_hat/eps) and evidence volume
  8  C2 paired FCR differences vs off (ev1, ev20, ev200, full, probe20) at g = 0.5 and 1
  9  B-adv instance-cluster FCR per hazard cell; ex-ante vs ex-post marginal binning
  10 zero-effect censoring share; Tier-C trigger under alternative tau / complete-case; matched comparison
  11 A2 scale sensitivity (smoothing 1/10/50, pooled ratio, absolute steps, excluding zero-cost problems)
  12 NL non-tie 0.57x split JPC/LR-chi2-grid x LR-chi2-grid/B3
  13 audit-table fixes (OPT_K not run; Q2 placebo deviation; source-hash drift)
Bootstrap: instance clusters, percentile, seed 42 (B = 10^4 for means / AUC / FCR; 2000 for logistic refits).
CPU only, 1 thread.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parent.parent
BASE = WS / "exp" / "results" / "full"
OUT = BASE / "r3_p0_reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

from dsswm.stats.auc import auc as auc_pt, lr_test, logistic_fit  # noqa: E402
from run_r3_analysis_aggregate import cost, jl, mean_ci, cluster_auc, dedupe, KEY  # noqa: E402

B = 10_000
B_LOGIT = 2000
SEED = 42
EPS, DELTA = 0.02, 0.05
TAU_NL, TAU_LIN = 3000, 20000
NONTIE = ("near", "clear")
ALL3 = ("tie", "near", "clear")
T0W = time.time()
LOGL = []
STATUS = "EXPLORATORY (post-hoc); not confirmatory; does not change the locked Tier C verdict"


def log(m):
    s = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    LOGL.append(s)
    print(s, flush=True)


def fnum(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def r6(x, d=6):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r6(v, d) for v in x]
    try:
        return round(float(x), d)
    except (TypeError, ValueError):
        return x


def write_csv(name, rows):
    rows = list(rows)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def load(task):
    d = BASE / task
    r = jl(d / "results.jsonl")
    p = jl(d / "predictors.jsonl")
    for x in r:
        x["_src"] = task
    for x in p:
        x["_src"] = task
    return r, p


def boot_rows(n, b=B, seed=SEED):
    return np.random.default_rng(seed).integers(n, size=(b, n))


def ci_of(reps):
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    if len(reps) < 10:
        return [None, None]
    return [float(x) for x in np.quantile(reps, [0.025, 0.975])]


# ---------------------------------------------------------------------------------------------- cost tables
def S_table(rows, tau, layers=NONTIE, keep=None, smooth_key=None):
    """S[(method, arm)][instance] = sum of cost over problems in `layers` (optionally only keys in `keep`)."""
    T = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r.get("gap_layer") not in layers:
            continue
        if keep is not None and (r["instance"], r["stream"], r["problem"]) not in keep:
            continue
        T[(r["method"], r["arm"])][r["instance"]] += cost(r, tau)
    return T


def logr(Ta, Tb, c=1.0):
    ii = sorted(set(Ta) & set(Tb))
    return {i: math.log((Ta[i] + c) / (Tb[i] + c)) for i in ii}


def mc(d, seed=SEED):
    """mean + instance bootstrap CI of a dict instance -> value."""
    m = mean_ci(list(d.values()), seed=seed)
    return {"est": r6(m["est"]), "ci": r6(m["ci"]), "n_inst": m["n"]}


# ---------------------------------------------------------------------------------------------- logistic helpers
def logit_newton(X, y, max_iter=50, tol=1e-9, ridge=1e-8):
    """X includes intercept column. Returns beta (np) or None."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p) - ridge * beta
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def cluster_robust_se(X, y, beta, clusters):
    eta = np.clip(X @ beta, -35, 35)
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    Hinv = np.linalg.inv((X * W[:, None]).T @ X)
    sc = X * (y - p)[:, None]
    keys, inv = np.unique(clusters, return_inverse=True)
    G = np.zeros((len(keys), X.shape[1]))
    np.add.at(G, inv, sc)
    G_ = len(keys)
    adj = G_ / (G_ - 1) if G_ > 1 else 1.0
    V = Hinv @ (G.T @ G) * adj @ Hinv
    return np.sqrt(np.diag(V))


def cluster_boot_logit(X, y, clusters, b=B_LOGIT, seed=SEED):
    keys, inv = np.unique(clusters, return_inverse=True)
    idx = [np.flatnonzero(inv == g) for g in range(len(keys))]
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(b):
        dr = rng.integers(len(keys), size=len(keys))
        ii = np.concatenate([idx[j] for j in dr])
        yy = y[ii]
        if yy.all() or (~yy.astype(bool)).all():
            continue
        bt = logit_newton(X[ii], yy)
        if bt is not None and np.all(np.isfinite(bt)):
            reps.append(bt)
    return np.asarray(reps)


def grouped_cv_auc(Xb, Xf, y, clusters, k=5, seed=SEED):
    """Out-of-instance (grouped k-fold) AUC for base vs full logistic models."""
    keys = np.unique(clusters)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(keys)
    fold = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold[c] for c in clusters])
    pb, pf = np.zeros(len(y)), np.zeros(len(y))
    for j in range(k):
        tr, te = f != j, f == j
        bb = logit_newton(Xb[tr], y[tr])
        bf = logit_newton(Xf[tr], y[tr])
        pb[te] = Xb[te] @ bb
        pf[te] = Xf[te] @ bf
    return pb, pf


def paired_delta_auc(sa, sb, y, clusters, b=2000, seed=SEED):
    keys, inv = np.unique(clusters, return_inverse=True)
    idx = [np.flatnonzero(inv == g) for g in range(len(keys))]
    rng = np.random.default_rng(seed)
    yb = y.astype(bool)
    reps = []
    for _ in range(b):
        dr = rng.integers(len(keys), size=len(keys))
        ii = np.concatenate([idx[j] for j in dr])
        if yb[ii].all() or (~yb[ii]).all():
            continue
        reps.append(auc_pt(sa[ii], yb[ii]) - auc_pt(sb[ii], yb[ii]))
    return {"est": r6(auc_pt(sa, yb) - auc_pt(sb, yb)), "ci": r6(ci_of(reps))}


# ==================================================================================================== main
def main():
    S = {"task": "r3_p0_reanalysis", "status": STATUS, "started_at": datetime.now().isoformat(),
         "new_env_interactions": 0, "new_seeds": 0, "B": B, "B_logit": B_LOGIT, "seed": SEED,
         "inputs": "exp/results/full/r3_*/{results,predictors}.jsonl + summary.json (read-only)"}
    lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
    S["lock"] = {"version": lock.get("version"), "status": lock.get("status"), "sha256": lock.get("sha256")}
    agg = json.loads((BASE / "r3_analysis_aggregate" / "summary.json").read_text())

    # ------------------------------------------------------------------ load
    NL, NLP = [], []
    for c in "abcdefgh":
        r, p = load(f"r3_nl_main_{c}")
        NL += r
        NLP += p
    EX = []
    for c in "abc":
        EX += load(f"r3_nl_extra_{c}")[0]
    LIN = []
    for c in "abc":
        LIN += load(f"r3_lin_{c}")[0]
    STAT, STATP = [], []
    for c in "abcd":
        r, p = load(f"r3_static_{c}")
        STAT += r
        STATP += p
    HZ, HZP = [], []
    for t in ("r3_hazard_a", "r3_hazard_b"):
        r, p = load(t)
        HZ += r
        HZP += p
    ZE = []
    for t in ("r3_controls_zero_effect_a", "r3_controls_zero_effect_b"):
        ZE += load(t)[0]
    X1, X1P = load("r3_x1_sampler_misspec")
    NL, d1 = dedupe(NL, KEY)
    EX, d2 = dedupe(EX, KEY)
    LIN, d3 = dedupe(LIN, KEY)
    S["rows"] = {"NL": len(NL), "NL_pred": len(NLP), "NL_extra": len(EX), "Lin": len(LIN), "static": len(STAT),
                 "static_pred": len(STATP), "hazard": len(HZ), "zero_effect": len(ZE), "x1": len(X1),
                 "dups_removed": [d1, d2, d3]}
    log(f"loaded {S['rows']}")
    LINE = [r for r in LIN if r.get("layer") == "E1-Lin"]

    # ============================================================== item 1: metric consistency
    T_nl = S_table(NL, TAU_NL)
    T_nl_all = S_table(NL, TAU_NL, layers=ALL3)
    T_lin = S_table(LINE, TAU_LIN)
    T_ex = S_table(EX, TAU_NL)
    T_ex8 = S_table([r for r in EX if r["method"] == "B8" and int(r["stream"]) == 0], TAU_NL)
    T_nl_clear = S_table(NL, TAU_NL, layers=("clear",))
    T_lin_clear = S_table(LINE, TAU_LIN, layers=("clear",))
    a1 = logr(T_nl[("JPC", "full")], T_nl[("JPC", "off")])
    b3 = logr(T_nl[("B3", "full")], T_nl[("B3", "off")])
    met = []

    def add(item, outline_val, est_dict, estimand, layer, sample, tau, agg_, source, note=""):
        e = est_dict.get("est") if isinstance(est_dict, dict) else est_dict
        ci = est_dict.get("ci") if isinstance(est_dict, dict) else None
        n = est_dict.get("n_inst") if isinstance(est_dict, dict) else None
        match = None
        if outline_val is not None and e is not None:
            tol = max(0.006, 0.006 * abs(outline_val))
            match = abs(float(e) - float(outline_val)) <= tol
        met.append({"item": item, "outline_value": outline_val, "recomputed": r6(e, 5), "ci95": r6(ci, 4),
                    "n_instances": n, "estimand": estimand, "layer": layer, "sample": sample, "tau": tau,
                    "aggregation": agg_, "source": source, "match": match, "note": note})

    agg_inst = "mean over instances of log((S_a+1)/(S_b+1)), S = per-instance sum of new interactions (cost capped at tau)"
    add("A1 NL JPC full/off", -1.93, mc(a1), "non-tie primary endpoint (near+clear)", "NL-R0", "120 inst, streams 0-2",
        TAU_NL, agg_inst, "r3_nl_main_[a-h]")
    add("A1 Lin JPC-Lin full/off", -2.40, mc(logr(T_lin[("JPC-Lin", "full")], T_lin[("JPC-Lin", "off")])),
        "non-tie primary endpoint", "E1-Lin", "48 inst, streams 0-2", TAU_LIN, agg_inst, "r3_lin_[a-c]")
    add("B3 NL full/off (non-tie)", -2.15, mc(b3), "non-tie primary endpoint", "NL-R0", "120 inst, streams 0-2", TAU_NL,
        agg_inst, "r3_nl_main_[a-h]", "use this next to JPC -1.93")
    add("JPC NL full/off (full stream incl. ties)", -1.676,
        mc(logr(T_nl_all[("JPC", "full")], T_nl_all[("JPC", "off")])), "full stream incl. ties (tie+near+clear)",
        "NL-R0", "120 inst, streams 0-2", TAU_NL, agg_inst, "= controls.reuse_factor_by_stream quota_main")
    add("B3 NL full/off (full stream incl. ties)", -1.34,
        mc(logr(T_nl_all[("B3", "full")], T_nl_all[("B3", "off")])), "full stream incl. ties (tie+near+clear)",
        "NL-R0", "120 inst, streams 0-2", TAU_NL, agg_inst, "= controls.reuse_factor_by_stream quota_main",
        "never place next to JPC -1.93 (different estimand)")
    I = {i: a1[i] - b3[i] for i in set(a1) & set(b3)}
    add("A2 NL interaction I", 0.22, mc(I), "non-tie; log JPC full/off - log B3 full/off", "NL-R0", "120 inst",
        TAU_NL, agg_inst, "r3_nl_main")
    lj = logr(T_lin[("JPC-Lin", "full")], T_lin[("JPC-Lin", "off")])
    lr_ = logr(T_lin[("RAGE", "full")], T_lin[("RAGE", "off")])
    add("A2 Lin interaction I", 0.02, mc({i: lj[i] - lr_[i] for i in set(lj) & set(lr_)}), "non-tie", "E1-Lin",
        "48 inst", TAU_LIN, agg_inst, "r3_lin")
    add("A3 NL ev1/vol", -0.06, mc(logr(T_nl[("JPC", "ev1")], T_nl[("JPC", "vol")])), "non-tie", "NL-R0", "120 inst",
        TAU_NL, agg_inst, "r3_nl_main")
    add("A3 Lin ev1/vol", 0.03, mc(logr(T_lin[("JPC-Lin", "ev1")], T_lin[("JPC-Lin", "vol")])), "non-tie", "E1-Lin",
        "48 inst", TAU_LIN, agg_inst, "r3_lin")
    add("RAGE Lin full/off", -2.42, mc(lr_), "non-tie", "E1-Lin", "48 inst", TAU_LIN, agg_inst, "r3_lin")
    add("LR-chi2-grid NL full/off", -1.58, mc(logr(T_ex[("LR-chi2-grid", "full")], T_ex[("LR-chi2-grid", "off")])),
        "non-tie", "NL-R0", "120 inst, streams 0-2", TAU_NL, agg_inst, "r3_nl_extra")
    add("B8 NL full/off (stream 0)", -4.66, mc(logr(T_ex8[("B8", "full")], T_ex8[("B8", "off")])), "non-tie",
        "NL-R0", "stream 0 only (lock downscale)", TAU_NL, agg_inst, "r3_nl_extra")
    clr = logr(T_nl_clear[("JPC", "full")], T_nl_clear[("B3", "full")])
    m = mc(clr)
    add("clear-layer NL JPC/B3 full (ratio)", 0.58,
        {"est": math.exp(m["est"]), "ci": [math.exp(x) for x in m["ci"]], "n_inst": m["n_inst"]},
        "clear layer only; exp(mean log ratio)", "NL-R0", "120 inst", TAU_NL, "exp of instance-mean log ratio", "r3_nl_main")
    m = mc(logr(T_lin_clear[("JPC-Lin", "full")], T_lin_clear[("RAGE", "full")]))
    add("clear-layer Lin JPC/RAGE full (ratio)", 0.92,
        {"est": math.exp(m["est"]), "ci": [math.exp(x) for x in m["ci"]], "n_inst": m["n_inst"]},
        "clear layer only", "E1-Lin", "48 inst", TAU_LIN, "exp of instance-mean log ratio", "r3_lin")
    mn = mc(logr(T_nl[("JPC", "full")], T_nl[("B3", "full")]))
    add("main non-tie NL JPC/B3 full (log)", -0.56, mn, "non-tie (near+clear)", "NL-R0", "120 inst, streams 0-2",
        TAU_NL, agg_inst, "r3_nl_main", f"ratio exp = {math.exp(mn['est']):.4f} (the '0.57x')")
    add("Lin non-tie JPC/RAGE full (log)", 0.005, mc(logr(T_lin[("JPC-Lin", "full")], T_lin[("RAGE", "full")])),
        "non-tie", "E1-Lin", "48 inst", TAU_LIN, agg_inst, "r3_lin")
    for tau, ov in ((1500, -1.92), (3000, -1.93), (6000, -1.94)):
        Tt = S_table(NL, tau)
        add(f"A1 NL tau={tau}", ov, mc(logr(Tt[("JPC", "full")], Tt[("JPC", "off")])), "non-tie", "NL-R0", "120 inst",
            tau, agg_inst, "r3_nl_main")
    # Lin descriptive medians from r3_lin_b
    lb = json.loads((BASE / "r3_lin_b" / "summary.json").read_text()).get("per_instance_descriptive", {})
    jm = [v.get("A1_JPC_full_off") for v in lb.values() if fnum(v.get("A1_JPC_full_off")) is not None]
    rm = [v.get("RAGE_full_off") for v in lb.values() if fnum(v.get("RAGE_full_off")) is not None]
    add("Lin JPC full/off r3_lin_b descriptive", -1.87, {"est": float(np.median(jm)), "ci": None, "n_inst": len(jm)},
        "median of per-instance log ratios, one chunk", "E1-Lin", "r3_lin_b, 16 inst", TAU_LIN, "median",
        "r3_lin_b/summary.json per_instance_descriptive", "descriptive; confirmatory Lin A1 is -2.40")
    add("Lin RAGE full/off r3_lin_b descriptive", -2.01, {"est": float(np.median(rm)), "ci": None, "n_inst": len(rm)},
        "median of per-instance log ratios, one chunk", "E1-Lin", "r3_lin_b, 16 inst", TAU_LIN, "median",
        "r3_lin_b/summary.json per_instance_descriptive", "")
    log("item 1 core done")
    S["item1_note"] = ("B3 NL -2.15 = non-tie primary endpoint (near+clear, tau 3000, 120 inst); -1.34 = full stream "
                       "incl. ties (tie+near+clear) = controls.reuse_factor_by_stream 'quota_main(all layers)'. JPC "
                       "counterpart of -1.34 is -1.676, not -1.93.")

    # ============================================================== item 2: orth constructible-only
    orth_rows, orth_counts = [], {}
    for L, rows, jm_, bm_, tau in (("NL-R0", NL, "JPC", "B3", TAU_NL), ("E1-Lin", LINE, "JPC-Lin", "RAGE", TAU_LIN)):
        for meth in (jm_, bm_):
            orr = [r for r in rows if r["method"] == meth and r["arm"] == "orth"]
            if not orr:
                continue
            cnt = Counter()
            for r in orr:
                st = ("trivial" if r.get("orth_trivial") is True else
                      "infeasible" if r.get("status") == "ORTH_INFEASIBLE" else "constructible")
                cnt[(r["gap_layer"], st)] += 1
                cnt[("all", st)] += 1
            orth_counts[f"{L}|{meth}"] = {f"{a}|{b}": v for (a, b), v in sorted(cnt.items())}
            keep = {(r["instance"], r["stream"], r["problem"]) for r in orr
                    if r.get("status") != "ORTH_INFEASIBLE" and r.get("orth_trivial") is not True}
            To = S_table(rows, tau, keep=keep)
            Tall = S_table(rows, tau)
            ntr = sum(1 for r in orr if r.get("orth_trivial") is not True)
            for comp, base in (("orth/off", "off"), ("orth/vol", "vol")):
                v = mc(logr(To[(meth, "orth")], To[(meth, base)]))
                va = mc(logr(Tall[(meth, "orth")], Tall[(meth, base)]))
                orth_rows.append({"layer": L, "method": meth, "contrast": comp, "subset": "constructible non-tie (matched)",
                                  "log_ratio": v["est"], "ci": v["ci"], "n_inst": v["n_inst"],
                                  "n_constructible_problems_all_layers": len(keep),
                                  "n_constructible_nontie_problems": sum(1 for r in orr if r["gap_layer"] in NONTIE
                                                                         and r.get("status") != "ORTH_INFEASIBLE"
                                                                         and r.get("orth_trivial") is not True),
                                  "n_orth_nontrivial": ntr,
                                  "constructible_rate_nontrivial": r6(
                                      sum(1 for r in orr if r.get("status") != "ORTH_INFEASIBLE"
                                          and r.get("orth_trivial") is not True) / max(ntr, 1), 4)})
                orth_rows.append({"layer": L, "method": meth, "contrast": comp,
                                  "subset": "all non-tie incl. ORTH_INFEASIBLE billed at tau",
                                  "log_ratio": va["est"], "ci": va["ci"], "n_inst": va["n_inst"]})
            vv = mc(logr(To[(meth, "vol")], To[(meth, "off")]))
            orth_rows.append({"layer": L, "method": meth, "contrast": "vol/off", "subset": "constructible non-tie (matched)",
                              "log_ratio": vv["est"], "ci": vv["ci"], "n_inst": vv["n_inst"]})
    write_csv("item2_orth_constructible.csv", orth_rows)
    S["item2_orth"] = {"table": orth_rows, "counts": orth_counts,
                       "note": ("keep = (instance, stream, problem) keys where the method's orth row is neither "
                                "ORTH_INFEASIBLE nor trivial (N_k = 0); off/vol/orth costs summed over the same keys "
                                "(non-tie). The aggregate A3o (log orth/vol 2.82 NL / 5.82 Lin) ALREADY used this "
                                "constructible-only set; the infeasible-billed-at-tau rows enter only the forest rows.")}
    log("item 2 done")

    # ============================================================== item 3: T0
    p4r, p4p = load("r3_p4_t0_mechanism_gate")
    pk = lambda r: (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])  # noqa: E731
    pm4 = {}
    for p in p4p:
        if str(p.get("source", "")).startswith("R0_ledger") and not str(p.get("source", "")).endswith("_open"):
            pm4[pk(p)] = p
    A = [(r, pm4.get(pk(r))) for r in p4r if r.get("source") == "R0_ledger"]
    A = [(r, p) for r, p in A if p is not None]
    rem = np.array([fnum(p.get("rem_bar_over_eps_p1")) or np.nan for _, p in A])
    rex = np.array([fnum(r.get("rem_exact_over_eps_max")) if fnum(r.get("rem_exact_over_eps_max")) is not None
                    else np.nan for r, _ in A])
    eta = np.array([fnum(r.get("eta")) or 0.0 for r, _ in A])
    t0 = {"n_snapshots": len(A), "seeds": "dev 720-739 (NOT eval seeds)"}
    for nm, msk in (("eta_le_2eps", eta <= 2 * EPS), ("eta_le_eps", eta <= EPS), ("all", np.ones(len(A), bool))):
        rb, rx = rem[msk & np.isfinite(rem)], rex[msk & np.isfinite(rex)]
        t0[nm] = {"n": int(msk.sum()), "bound_over_eps_p90": r6(np.quantile(rb, .9)) if len(rb) else None,
                  "bound_over_eps_median": r6(np.median(rb)) if len(rb) else None,
                  "exact_over_eps_p90": float(np.quantile(rx, .9)) if len(rx) else None,
                  "exact_over_eps_median": float(np.median(rx)) if len(rx) else None,
                  "exact_over_eps_max": r6(np.max(rx)) if len(rx) else None,
                  "share_exact_le_0.5": r6(float(np.mean(rx <= 0.5))) if len(rx) else None}
    s4 = json.loads((BASE / "r3_p4_t0_mechanism_gate" / "summary.json").read_text())["T0"]
    t0["summary_json"] = {"p90_bound_eta_le_2eps": s4["ii_rem_bar_over_eps"]["p90_eta_le_2eps"],
                          "p90_bound_eta_le_eps": s4["ii_rem_bar_over_eps"]["p90_eta_le_eps"],
                          "p90_exact_eta_le_2eps": s4["ii_exact_rem_over_eps_oracle"]["p90_eta_le_2eps"],
                          "median_exact_eta_le_2eps": s4["ii_exact_rem_over_eps_oracle"]["median_eta_le_2eps"]}
    if t0["eta_le_2eps"]["exact_over_eps_p90"]:
        t0["looseness_p90_ratio"] = r6(t0["eta_le_2eps"]["bound_over_eps_p90"] / t0["eta_le_2eps"]["exact_over_eps_p90"], 1)
    S["item3_T0"] = t0
    log("item 3 done")

    # ============================================================== item 4: HD2
    hs = json.loads((BASE / "r3_replicate_hd1b_hd2" / "hd2" / "summary.json").read_text())
    fp = hs["slope_fits"]["phased"]
    cf = hs["construction"]["fits"]
    hd2 = []
    for scope in ("time", "participant", "policy", "pooled"):
        k = fp[f"kappa_eff|{scope}"]
        bc = fp[f"beta_corrected|{scope}"]
        hd2.append({"estimand": "kappa_eff slope: log(N_P/N_J) on log sqrt(rho_P/rho_J)", "partition": scope,
                    "slope": r6(k["slope"], 4), "ci": r6(k["slope_ci95"], 4), "n": k["n"], "gate": "[1.5, 2.5]",
                    "in_gate": 1.5 <= k["slope"] <= 2.5, "in_gate_list": scope != "pooled"})
        hd2.append({"estimand": "beta-corrected slope vs log(rho_P/rho_J)", "partition": scope,
                    "slope": r6(bc["slope"], 4), "ci": r6(bc["slope_ci95"], 4), "n": bc["n"], "gate": "[0.85, 1.15]",
                    "in_gate": 0.85 <= bc["slope"] <= 1.15, "in_gate_list": scope == "pooled"})
    for key, lab in (("phased|time|slope_vs_logH(H=2..16)", "construction sequential slope vs log H (time)"),
                     ("phased|time|design_level_slope_vs_logH", "construction design-level slope vs log H")):
        c = cf[key]
        hd2.append({"estimand": lab, "partition": "time (Omega(H^2) construction)", "slope": r6(c["slope"], 4),
                    "ci": r6(c["slope_ci95"], 4), "n": c["n"], "gate": "[1.5, 2.5]" if "sequential" in lab else "none",
                    "in_gate": 1.5 <= c["slope"] <= 2.5, "in_gate_list": "sequential" in lab})
    write_csv("item4_hd2_slopes.csv", hd2)
    failed = [f"{h['estimand']}|{h['partition']}" for h in hd2 if h["in_gate_list"] and not h["in_gate"]]
    S["item4_HD2"] = {"table": hd2, "gate_items_failed": failed,
                      "pass_criteria_hd2_slopes_in_[1.5,2.5]": False,
                      "note": ("The replication gate 'hd2_slopes_in_[1.5,2.5]' = all per-partition kappa_eff slopes AND "
                               "the construction time slope in [1.5,2.5]. Only the policy partition (2.614, CI "
                               "[2.433, 2.800]) is outside; time 2.242, participant 2.142, construction 2.216 are "
                               "inside. The beta-corrected pooled slope 1.137 is a different estimand with its own "
                               "gate [0.85,1.15] (passes). Sandwich 100% and H=1 exact-zero pass.")}
    log("item 4 done")

    # ============================================================== item 5: Q1
    variant = (lock.get("frozen_items") or {}).get("predictor_variant") or "S2"
    pkey = lambda r: (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])  # noqa: E731

    def pmap(P):
        mp = {}
        for p in P:
            if p.get("phase") not in (None, "pre_scoring"):
                continue
            k = pkey(p)
            if k not in mp or p.get("record") == "full_predictors":
                mp[k] = p
        return mp

    def join(R, P):
        pm = pmap(P)
        o = []
        for r in R:
            p = pm.get(pkey(r))
            if p is not None and p.get(variant) is not None:
                o.append((r, p))
        return o

    def zc_events(pairs):
        return [(r, p) for r, p in pairs if r.get("status") == "CERTIFIED" and r.get("zero_cost")
                and int(r.get("problem_index", r["problem"])) >= 1 and r.get("gap_layer") in NONTIE]

    q1 = []
    q1 += join([r for r in STAT if r["arm"] == "full" and r.get("learner_class") == "static"
                and fnum(r.get("dose")) in (0.5, 1.0)], STATP)
    q1 += join([r for r in X1 if r.get("source") == "x1_m1_2eps" and r.get("sampler") == "DDA"], X1P)
    q1 += join([r for r in HZ if str(r.get("base_method", r["method"])).startswith("JPC")], HZP)
    q1 = zc_events(q1)

    def sc(p, nm):
        if nm == "S2":
            v = p.get("S2")
            return 1e300 if v in (None, "inf") or (isinstance(v, float) and math.isinf(v)) else fnum(v)
        return fnum(p.get(nm))

    def q1blk(ev, b=B):
        y = np.array([bool(r.get("false_cert")) for r, _ in ev])
        cl = np.array([f"{r.get('_src')}:{r['instance']}" for r, _ in ev])
        o = {"n": len(ev), "n_pos": int(y.sum()), "n_clusters": int(len(set(cl.tolist())))}
        for nm in ("S2", "S1", "neg_log_gap_hat", "log_tr_Iinv", "eta_hat_gof"):
            s = [sc(p, nm) for _, p in ev]
            ok = np.array([v is not None for v in s])
            if ok.sum() < 3 or len(set(y[ok].tolist())) < 2:
                o[nm] = None
                continue
            a = cluster_auc(np.array([v for v in s if v is not None], float), y[ok], cl[ok], b=b)
            o[nm] = {"auc": r6(a["auc"], 4), "ci": r6(a["ci"], 4), "n": a["n"], "n_pos": a["n_pos"]}
        return o

    Q1 = {"frozen_variant": variant, "pooled": q1blk(q1)}
    scope = [(r, p) for r, p in q1 if fnum(r.get("eta_dec")) is not None and fnum(r["eta_dec"]) <= EPS]
    Q1["scope_eta_dec_le_eps"] = q1blk(scope, b=2000)
    Q1["scope_eta_dec_le_eps"]["note"] = "fewer than 10 positives: AUC is not a strong test of Prop. 3 in scope"
    Q1["by_source"] = {s: {"n": sum(1 for r, _ in q1 if r["_src"].startswith(s)),
                           "n_pos": sum(1 for r, _ in q1 if r["_src"].startswith(s) and r.get("false_cert"))}
                       for s in ("r3_static", "r3_x1", "r3_hazard")}
    y = np.array([bool(r.get("false_cert")) for r, _ in q1])
    cl = np.array([f"{r.get('_src')}:{r['instance']}" for r, _ in q1])
    s2 = np.array([sc(p, "S2") for _, p in q1], float)
    gp = np.array([sc(p, "neg_log_gap_hat") if sc(p, "neg_log_gap_hat") is not None else np.nan for _, p in q1], float)
    ok = np.isfinite(gp) & np.isfinite(s2)
    Q1["delta_auc_gap_minus_S2"] = paired_delta_auc(gp[ok], s2[ok], y[ok], cl[ok])
    S["item5_Q1"] = Q1
    write_csv("item5_q1_auc.csv", [{"scope": sc_, "predictor": nm, **(Q1[sc_][nm] or {})}
                                   for sc_ in ("pooled", "scope_eta_dec_le_eps")
                                   for nm in ("S2", "S1", "neg_log_gap_hat", "log_tr_Iinv", "eta_hat_gof")])
    log("item 5 done")

    # ============================================================== item 6: Q2 restricted to static_open
    pm_open = {pkey(p): p for p in STATP if p.get("arm") == "full@open"}
    opp = []
    for r in STAT:
        if r["arm"] == "full@open":
            p = pm_open.get(pkey(r))
            if p is not None:
                opp.append((r, p))
    rows6 = []
    for r, p in opp:
        g, t, a = fnum(p.get("neg_log_gap_hat")), fnum(p.get("log_tr_Iinv")), fnum(p.get("A_k_max"))
        if None in (g, t, a) or a <= 0:
            continue
        rows6.append({"lg": -g, "lt": t, "lak": math.log(a), "y": 1.0 if r.get("zero_cost") else 0.0,
                      "cl": f"{r['_src']}:{r['instance']}", "dose": fnum(r.get("dose")),
                      "pg": fnum(p.get("A_k_placebo_global_max")), "pb": fnum(p.get("A_k_placebo_block_max"))})
    Q2 = {"sample": "static twins, arm full@open (predictors written at problem open, before the outcome)",
          "n_opp_rows": len(opp), "n_used": len(rows6)}
    if rows6:
        Y = np.array([d["y"] for d in rows6])
        CL = np.array([d["cl"] for d in rows6])
        Xb = np.array([[d["lg"], d["lt"]] for d in rows6])
        AK = np.array([[d["lak"]] for d in rows6])
        Q2.update({"n_pos": int(Y.sum()), "n_clusters": int(len(set(CL.tolist()))),
                   "by_dose": {str(g): {"n": sum(1 for d in rows6 if d["dose"] == g),
                                        "n_pos": int(sum(d["y"] for d in rows6 if d["dose"] == g))}
                               for g in (0.0, 0.5, 1.0)}})
        lt = lr_test(Xb, np.c_[Xb, AK], Y)
        Q2["row_level_lr_test"] = {"stat": r6(lt["stat"], 4), "p": lt["p"],
                                   "note": "row-level (NOT cluster-robust); reported for comparison only"}
        Xf1 = np.c_[np.ones(len(Y)), Xb, AK]
        Xb1 = np.c_[np.ones(len(Y)), Xb]
        bt = logit_newton(Xf1, Y)
        se = cluster_robust_se(Xf1, Y, bt, CL)
        from scipy.stats import norm
        z = bt[3] / se[3]
        reps = cluster_boot_logit(Xf1, Y, CL)
        Q2["cluster_robust_wald_logAk"] = {"beta": r6(bt[3], 5), "se_cluster": r6(se[3], 5), "z": r6(z, 4),
                                           "p_two_sided": float(2 * norm.sf(abs(z)))}
        Q2["cluster_bootstrap_beta_logAk"] = {"ci": r6(ci_of(reps[:, 3]), 5), "n_reps": int(len(reps)),
                                              "share_reps_gt0": r6(float(np.mean(reps[:, 3] > 0)), 4)}
        # dose fixed effects
        dd = np.array([[1.0 if d["dose"] == 0.5 else 0.0, 1.0 if d["dose"] == 1.0 else 0.0] for d in rows6])
        Xfd = np.c_[Xf1, dd]
        btd = logit_newton(Xfd, Y)
        sed = cluster_robust_se(Xfd, Y, btd, CL)
        Q2["with_dose_fixed_effects"] = {"beta_logAk": r6(btd[3], 5), "se_cluster": r6(sed[3], 5),
                                         "p_two_sided": float(2 * norm.sf(abs(btd[3] / sed[3])))}
        pb_, pf_ = grouped_cv_auc(Xb1, Xf1, Y, CL)
        Q2["out_of_instance_cv"] = {"k_folds": 5, "auc_base": r6(auc_pt(pb_, Y.astype(bool)), 4),
                                    "auc_with_logAk": r6(auc_pt(pf_, Y.astype(bool)), 4),
                                    "delta_auc": paired_delta_auc(pf_, pb_, Y, CL)}
        a_ak = cluster_auc(AK[:, 0], Y.astype(bool), CL, b=2000)
        Q2["auc_A_k_alone"] = {"auc": r6(a_ak["auc"], 4), "ci": r6(a_ak["ci"], 4)}
        for nm, key in (("placebo_global", "pg"), ("placebo_block", "pb")):
            okk = [i for i, d in enumerate(rows6) if d[key] is not None]
            a = auc_pt(np.array([rows6[i][key] for i in okk]), Y[okk].astype(bool))
            Q2[f"auc_{nm}"] = r6(a, 4)
            Q2[f"auc_{nm}_orientation_free"] = r6(max(a, 1 - a), 4)
        Q2["lock_rule"] = "LR p < 0.01 AND GLOBAL permutation placebo AUC (orientation-free) <= 0.6"
        Q2["restricted_point_reading"] = {
            "cluster_robust_p_lt_0.01": bool(Q2["cluster_robust_wald_logAk"]["p_two_sided"] < 0.01),
            "global_placebo_le_0.6": bool(Q2["auc_placebo_global_orientation_free"] <= 0.6)}
    # lock-specified global placebo on the ORIGINAL (invalid) aggregate sample, from aggregate summary
    q2a = agg["Q_family"]["Q2"]
    Q2["original_aggregate_sample"] = {
        "n": q2a.get("n"), "sources": q2a.get("sources"), "auc_placebo_block_orientation_free":
            q2a.get("auc_placebo_block_orientation_free"),
        "auc_placebo_global_orientation_free": q2a.get("auc_placebo_global_orientation_free"),
        "pass_point_under_lock_global_rule": bool(q2a["p"] < 0.01 and q2a["auc_placebo_global_orientation_free"] <= 0.6),
        "deviation": "aggregate used the BLOCK placebo; lock (Q2_placebo) specified GLOBAL. Both <= 0.6, so the "
                     "deviation does not change the point reading; Q2 remains NOT_EVALUABLE on that sample because NL "
                     "rows use certification-time predictors and the LR test is row-level."}
    Q2["verdict_restricted"] = ("EXPLORATORY restricted analysis; Q2 (confirmatory) stays NOT_EVALUABLE; "
                                "do not back-fill as a Q2 pass")
    S["item6_Q2_static_open"] = Q2
    log("item 6 done")

    # ============================================================== item 7: Q3 error model with gap & volume
    e_ = [(r, p) for r, p in q1 if fnum(p.get("S1")) and fnum(p.get("A_k_max"))]
    Q3 = {"n_events": len(e_)}
    if e_:
        ye = np.array([1.0 if r.get("false_cert") else 0.0 for r, _ in e_])
        cle = np.array([f"{r.get('_src')}:{r['instance']}" for r, _ in e_])
        lS1 = np.array([math.log(fnum(p["S1"])) for _, p in e_])
        lAk = np.array([math.log(fnum(p["A_k_max"])) for _, p in e_])
        gap = np.array([fnum(p.get("neg_log_gap_hat")) if fnum(p.get("neg_log_gap_hat")) is not None else np.nan
                        for _, p in e_])
        vol = np.array([math.log(fnum(p.get("n_obs")) or np.nan) if fnum(p.get("n_obs")) else np.nan for _, p in e_])
        src = np.array([r["_src"].split("_")[1] for r, _ in e_])
        ok = np.isfinite(gap) & np.isfinite(vol)
        Q3["n_complete"] = int(ok.sum())
        Q3["gap_definition"] = "neg_log_gap_hat = -log(gap_hat/eps) (learner-side); volume = log n_obs in ledger"
        specs = {"M0 (aggregate): logS1 + logAk": [lS1, lAk],
                 "M1: + gap": [lS1, lAk, gap],
                 "M2: + gap + log n_obs": [lS1, lAk, gap, vol],
                 "M3: + gap + log n_obs + source FE": [lS1, lAk, gap, vol,
                                                       (src == "x1").astype(float), (src == "hazard").astype(float)]}
        res7 = []
        from scipy.stats import norm
        for nm, cols in specs.items():
            X = np.c_[np.ones(ok.sum()), np.column_stack([c[ok] for c in cols])]
            yy, cc = ye[ok], cle[ok]
            bt = logit_newton(X, yy)
            se = cluster_robust_se(X, yy, bt, cc)
            reps = cluster_boot_logit(X, yy, cc)
            res7.append({"model": nm, "n": int(ok.sum()), "n_pos": int(yy.sum()),
                         "beta_logLambda": r6(bt[1], 4), "se_cluster": r6(se[1], 4),
                         "p_cluster": float(2 * norm.sf(abs(bt[1] / se[1]))),
                         "boot_ci_logLambda": r6(ci_of(reps[:, 1]), 4),
                         "share_boot_lt0": r6(float(np.mean(reps[:, 1] < 0)), 4),
                         "beta_gap": r6(bt[3], 4) if len(cols) >= 3 else None,
                         "beta_logAk": r6(bt[2], 4)})
        Q3["models"] = res7
        Q3["corr_logS1_gap"] = r6(float(np.corrcoef(lS1[ok], gap[ok])[0, 1]), 4)
        Q3["sign_survives_gap_and_volume"] = bool(res7[2]["beta_logLambda"] < 0 and res7[2]["boot_ci_logLambda"][1] < 0)
        write_csv("item7_q3_error_model.csv", res7)
    S["item7_Q3_refit"] = Q3
    log("item 7 done")

    # ============================================================== item 8: C2 paired intervals
    c2 = []
    arms = ("off", "ev1", "ev20", "ev200", "full", "probe20")
    for g, meth in ((0.5, "JPC_static_g0.5"), (1.0, "JPC_static_g1")):
        rr = [r for r in STAT if r["method"] == meth and r["arm"] in arms and r.get("status") == "CERTIFIED"]
        insts = sorted({r["instance"] for r in rr})
        ix = {i: k for k, i in enumerate(insts)}
        cnt = np.zeros((len(insts), len(arms), 2))
        for r in rr:
            cnt[ix[r["instance"]], arms.index(r["arm"]), 0] += bool(r.get("false_cert"))
            cnt[ix[r["instance"]], arms.index(r["arm"]), 1] += 1
        tot = cnt.sum(0)
        idx = boot_rows(len(insts))
        bs = cnt[idx].sum(1)  # (B, arms, 2)
        f = bs[:, :, 0] / bs[:, :, 1]
        for j, a in enumerate(arms):
            if a == "off":
                continue
            d = f[:, j] - f[:, 0]
            rrat = f[:, j] / np.where(f[:, 0] > 0, f[:, 0], np.nan)
            pt_d = tot[j, 0] / tot[j, 1] - tot[0, 0] / tot[0, 1]
            lo, hi = ci_of(d)
            k_bonf = 10
            lob, hib = [float(x) for x in np.quantile(d, [0.025 / k_bonf, 1 - 0.025 / k_bonf])]
            c2.append({"dose": g, "arm": a, "n_cert_arm": int(tot[j, 1]), "n_false_arm": int(tot[j, 0]),
                       "fcr_arm": r6(tot[j, 0] / tot[j, 1]), "n_cert_off": int(tot[0, 1]), "n_false_off": int(tot[0, 0]),
                       "fcr_off": r6(tot[0, 0] / tot[0, 1]), "diff_arm_minus_off": r6(pt_d),
                       "diff_ci95": r6([lo, hi]), "diff_ci_bonferroni10": r6([lob, hib]),
                       "rr_arm_over_off": r6((tot[j, 0] / tot[j, 1]) / (tot[0, 0] / tot[0, 1])) if tot[0, 0] else None,
                       "rr_ci95": r6(ci_of(rrat)), "n_instances": len(insts),
                       "ci_excludes_0": bool(hi < 0 or lo > 0)})
    write_csv("item8_c2_paired.csv", c2)
    S["item8_C2_paired"] = {"table": c2, "note": ("pooled FCR (sum false / sum certified) per arm, paired instance-cluster "
                                                  "bootstrap (same instance draw for every arm), B=10^4, uncorrected 95% "
                                                  "and Bonferroni over 10 comparisons")}
    log("item 8 done")

    # ============================================================== item 9: B-adv cluster
    adv = [r for r in HZ if r.get("layer") == "R1adv" and str(r.get("base_method", r["method"])).startswith("JPC")
           and r.get("status") == "CERTIFIED"]
    ainst = sorted({r["instance"] for r in adv})

    def clus_fcr(rows, b=B, seed=SEED):
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            by[r["instance"]][0] += bool(r.get("false_cert"))
            by[r["instance"]][1] += 1
        ks = sorted(by)
        if not ks:
            return None
        arr = np.array([by[k] for k in ks], float)
        idx = np.random.default_rng(seed).integers(len(ks), size=(b, len(ks)))
        s = arr[idx].sum(1)
        f = s[:, 0] / s[:, 1]
        k, n = arr.sum(0)
        return {"n_false": int(k), "n_cert": int(n), "n_instances": len(ks), "n_instances_with_error": int((arr[:, 0] > 0).sum()),
                "fcr": r6(k / n), "ci95": r6(ci_of(f)),
                "ci_bonferroni9": r6([float(x) for x in np.quantile(f, [0.025 / 9, 1 - 0.025 / 9])]),
                "p_fcr_le_delta": r6(float((1 + (f <= DELTA).sum()) / (b + 1)), 5)}

    cells9 = []
    for cal in ("post_bin_near", "post_bin_gcirc"):
        for a in range(3):
            for bq in range(3):
                rr = [r for r in adv if r.get("ante_bin_muflip") == a and r.get(cal) == bq]
                c = clus_fcr(rr)
                if c:
                    cells9.append({"caliber": cal, "binning": "ante_muflip x post", "ante_bin": a, "post_bin": bq, **c})
        for bq in range(3):
            c = clus_fcr([r for r in adv if r.get(cal) == bq])
            if c:
                cells9.append({"caliber": cal, "binning": "ex-post marginal", "ante_bin": "all", "post_bin": bq, **c})
    for ab in ("ante_bin_muflip", "ante_bin_lambda"):
        for a in range(3):
            c = clus_fcr([r for r in adv if r.get(ab) == a])
            if c:
                cells9.append({"caliber": ab, "binning": "ex-ante marginal", "ante_bin": a, "post_bin": "all", **c})
    write_csv("item9_badv_cluster.csv", cells9)
    hz_cells = [c for c in cells9 if c["binning"] == "ante_muflip x post" and c["caliber"] == "post_bin_near"
                and c["ci95"][0] is not None and c["ci95"][0] > DELTA]
    S["item9_Badv_cluster"] = {"n_cert": len(adv), "n_instances": len(ainst), "cells": cells9,
                               "near_caliber_cells_cluster_ci_lower_gt_delta": [(c["ante_bin"], c["post_bin"])
                                                                                for c in hz_cells],
                               "note": ("ante bins use mu_flip / Lambda_hat at n0 (harness-side quantity mu_flip; "
                                        "not learner-available); post bins are pre-registered ex-post eta_arg/eps "
                                        "labels (scoring only, not post-hoc bins)")}
    log("item 9 done")

    # ============================================================== item 10: zero-effect censoring / Tier C
    ze0 = [r for r in ZE if int(r["stream"]) == 0]
    cens = []
    for meth in ("JPC", "B3"):
        for arm in ("off", "ev1", "full"):
            rr = [r for r in ze0 if r["method"] == meth and r["arm"] == arm]
            if not rr:
                continue
            for tau in (1500, 3000, 6000):
                tot = sum(cost(r, tau) for r in rr)
                cc = sum(cost(r, tau) for r in rr if r.get(f"censored_tau{tau}"))
                cens.append({"method": meth, "arm": arm, "stream": 0, "tau": tau, "n_problems": len(rr),
                             "n_uncertified_at_tau": sum(1 for r in rr if r.get(f"censored_tau{tau}")),
                             "censored_cost_share": r6(cc / tot if tot else None, 5), "total_cost": tot,
                             "tie_declared_rate": r6(np.mean([bool(r.get("tie_declared")) for r in rr]), 4)})
    write_csv("item10_zero_effect_censoring.csv", cens)
    ze_inst = sorted({r["instance"] for r in ze0})
    nl_m = [r for r in NL if int(r["stream"]) == 0 and r["instance"] in set(ze_inst)]

    def cen_keys(rows, tau):
        return {(r["instance"], r["stream"], r["problem"]) for r in rows
                if r["arm"] == "full" and r["method"] in ("JPC", "B3") and r.get(f"censored_tau{tau}")}

    def allkeys(rows):
        return {(r["instance"], r["stream"], r["problem"]) for r in rows}

    trig = []
    for tau in (1500, 3000, 6000):
        Tz = S_table(ze0, tau, layers=ALL3)
        sn = mc(logr(Tz[("JPC", "full")], Tz[("B3", "full")]))
        Tm = S_table(nl_m, tau)
        sm = mc(logr(Tm[("JPC", "full")], Tm[("B3", "full")]))
        Tl = S_table(NL, tau)
        sl = mc(logr(Tl[("JPC", "full")], Tl[("B3", "full")]))
        kz = allkeys(ze0) - cen_keys(ze0, tau)
        km = allkeys(nl_m) - cen_keys(nl_m, tau)
        Tzc = S_table(ze0, tau, layers=ALL3, keep=kz)
        Tmc = S_table(nl_m, tau, keep=km)
        snc = mc(logr(Tzc[("JPC", "full")], Tzc[("B3", "full")]))
        smc = mc(logr(Tmc[("JPC", "full")], Tmc[("B3", "full")]))
        # both-declare-tie (post-treatment selection; appendix sensitivity only)
        tie_ok = defaultdict(set)
        for r in ze0:
            if r["arm"] == "full" and r.get("tie_declared"):
                tie_ok[(r["instance"], r["stream"], r["problem"])].add(r["method"])
        kt = {k for k, v in tie_ok.items() if {"JPC", "B3"} <= v}
        Tzt = S_table(ze0, tau, layers=ALL3, keep=kt)
        snt = mc(logr(Tzt[("JPC", "full")], Tzt[("B3", "full")]))
        for lab, null, main_, nk in (("locked form: null stream0 48 inst vs main 120 inst streams 0-2", sn, sl, None),
                                     ("matched: same 48 inst, stream 0", sn, sm, None),
                                     ("complete-case: drop problems where JPC-full or B3-full is uncertified at tau (both envs; post-treatment)",
                                      snc, smc, (len(kz), len(km))),
                                     ("both-declare-tie (post-treatment selection; appendix only) vs matched main",
                                      snt, sm, (len(kt), None))):
            trig.append({"tau": tau, "comparison": lab, "saving_null": r6(-null["est"], 5),
                         "saving_null_ci": r6([-null["ci"][1], -null["ci"][0]], 4),
                         "saving_main": r6(-main_["est"], 5), "saving_main_ci": r6([-main_["ci"][1], -main_["ci"][0]], 4),
                         "n_inst_null": null["n_inst"], "n_inst_main": main_["n_inst"],
                         "n_keys_kept": nk, "trigger_null_ge_main": bool(-null["est"] >= -main_["est"])})
    write_csv("item10_tierC_trigger.csv", trig)
    b3f = [c for c in cens if c["method"] == "B3" and c["arm"] == "full" and c["tau"] == 3000][0]
    S["item10_zero_effect"] = {"censoring": cens, "trigger": trig, "B3_full_tau3000": b3f,
                               "note": ("Trigger compares JPC/B3 method differences in the FULL arm, not full/off reuse "
                                        "effects. Complete-case and both-tie subsets condition on post-treatment "
                                        "outcomes: sensitivity only, never a replacement for the locked trigger.")}
    log("item 10 done")

    # ============================================================== item 11: A2 scale sensitivity
    a2 = []
    for L, rows, jm_, bm_, tau in (("NL-R0", NL, "JPC", "B3", TAU_NL), ("E1-Lin", LINE, "JPC-Lin", "RAGE", TAU_LIN)):
        T = S_table(rows, tau)
        for c in (1.0, 10.0, 50.0):
            aj = logr(T[(jm_, "full")], T[(jm_, "off")], c)
            ab = logr(T[(bm_, "full")], T[(bm_, "off")], c)
            v = mc({i: aj[i] - ab[i] for i in set(aj) & set(ab)})
            a2.append({"layer": L, "variant": f"smoothing c={int(c)}", "est": v["est"], "ci": v["ci"], "n_inst": v["n_inst"]})
        insts = sorted(set(T[(jm_, "full")]) & set(T[(bm_, "full")]) & set(T[(jm_, "off")]) & set(T[(bm_, "off")]))
        arr = np.array([[T[(jm_, "full")][i], T[(jm_, "off")][i], T[(bm_, "full")][i], T[(bm_, "off")][i]]
                        for i in insts])
        s = arr.sum(0)
        pt = math.log(s[0] / s[1]) - math.log(s[2] / s[3])
        bs = arr[boot_rows(len(insts))].sum(1)
        rep = np.log(bs[:, 0] / bs[:, 1]) - np.log(bs[:, 2] / bs[:, 3])
        a2.append({"layer": L, "variant": "pooled ratio of sums: log(sum full/sum off)_JPC - (..)_base",
                   "est": r6(pt), "ci": r6(ci_of(rep)), "n_inst": len(insts),
                   "detail": {"JPC_full_over_off": r6(s[0] / s[1]), "base_full_over_off": r6(s[2] / s[3])}})
        dj = arr[:, 1] - arr[:, 0]
        db = arr[:, 3] - arr[:, 2]
        bsj = dj[boot_rows(len(insts))].mean(1)
        bsb = db[boot_rows(len(insts))].mean(1)
        bsd = (dj - db)[boot_rows(len(insts))].mean(1)
        a2.append({"layer": L, "variant": "absolute steps saved per instance (off - full)",
                   "est": r6(float((dj - db).mean()), 2), "ci": r6(ci_of(bsd), 2), "n_inst": len(insts),
                   "detail": {"JPC_mean_saved": r6(dj.mean(), 2), "JPC_ci": r6(ci_of(bsj), 2),
                              "base_mean_saved": r6(db.mean(), 2), "base_ci": r6(ci_of(bsb), 2),
                              "est_is": "JPC saved - baseline saved (negative = baseline saves more steps)"}})
        zc = {(r["instance"], r["stream"], r["problem"]) for r in rows
              if r["arm"] == "full" and r["method"] in (jm_, bm_) and r.get("zero_cost")}
        keep = {(r["instance"], r["stream"], r["problem"]) for r in rows} - zc
        Tk = S_table(rows, tau, keep=keep)
        aj = logr(Tk[(jm_, "full")], Tk[(jm_, "off")])
        ab = logr(Tk[(bm_, "full")], Tk[(bm_, "off")])
        v = mc({i: aj[i] - ab[i] for i in set(aj) & set(ab)})
        nz = len(zc)
        a2.append({"layer": L, "variant": "excluding problems where JPC-full or base-full was zero-cost (c=1)",
                   "est": v["est"], "ci": v["ci"], "n_inst": v["n_inst"], "detail": {"n_problem_keys_dropped_all_layers": nz, "caveat": "conditions on a post-treatment outcome"}})
        # instance-level zero-total check
        z0 = sum(1 for i in insts if T[(jm_, "full")][i] == 0)
        a2.append({"layer": L, "variant": "diagnostic: instances with JPC non-tie full total = 0",
                   "est": z0, "ci": None, "n_inst": len(insts)})
    write_csv("item11_a2_scale.csv", a2)
    S["item11_A2_scale"] = a2
    log("item 11 done")

    # ============================================================== item 12: 0.57x split
    gapchk = {}
    nl_gap = {(r["instance"], r["stream"], r["problem"]): r.get("true_gap") for r in NL
              if r["method"] == "JPC" and r["arm"] == "full"}
    ex_gap = {(r["instance"], r["stream"], r["problem"]): r.get("true_gap") for r in EX
              if r["method"] == "LR-chi2-grid" and r["arm"] == "full"}
    common = set(nl_gap) & set(ex_gap)
    mism = sum(1 for k in common if abs((nl_gap[k] or 0) - (ex_gap[k] or 0)) > 1e-12)
    gapchk = {"n_common_problem_keys": len(common), "true_gap_mismatch": mism,
              "n_nl_keys": len(nl_gap), "n_extra_keys": len(ex_gap)}
    comb = NL + [r for r in EX if r["method"] == "LR-chi2-grid"]
    sp = []
    for lay, layers in (("non-tie", NONTIE), ("clear", ("clear",)), ("near", ("near",))):
        T = S_table(comb, TAU_NL, layers=layers)
        jl_ = logr(T[("JPC", "full")], T[("LR-chi2-grid", "full")])
        lb_ = logr(T[("LR-chi2-grid", "full")], T[("B3", "full")])
        jb_ = logr(T[("JPC", "full")], T[("B3", "full")])
        for lab, d in (("JPC/LR-chi2-grid (same grid & support; threshold+statistic differ)", jl_),
                       ("LR-chi2-grid/B3 (grid vs continuous GLM; support+estimator differ)", lb_),
                       ("JPC/B3 (total)", jb_)):
            v = mc(d)
            sp.append({"layers": lay, "component": lab, "log_ratio": v["est"], "ci": v["ci"],
                       "ratio": r6(math.exp(v["est"]), 4), "n_inst": v["n_inst"]})
    fc = {}
    for meth, rows in (("JPC", NL), ("B3", NL), ("LR-chi2-grid", EX)):
        rr = [r for r in rows if r["method"] == meth and r["arm"] in ("off", "full") and r.get("status") == "CERTIFIED"]
        fc[meth] = {"n_cert": len(rr), "n_false": sum(bool(r.get("false_cert")) for r in rr)}
    write_csv("item12_nontie_split.csv", sp)
    S["item12_split"] = {"problem_match_check": gapchk, "table": sp, "false_cert_off_full": fc,
                         "identifiability": ("The log decomposition is exact arithmetic on matched problems "
                                             "(same instances, streams, problems; true_gap identical). Labelling the two "
                                             "factors 'threshold' and 'finite support' is NOT identified: JPC vs "
                                             "LR-chi2-grid changes threshold and the statistic's sequential form; "
                                             "LR-chi2-grid vs B3 changes support, estimator and threshold family; "
                                             "LR-chi2-grid also has nonzero false certifications (different validity).")}
    log("item 12 done")

    # ============================================================== item 13: audit fixes
    aud_in = list(csv.DictReader(open(BASE / "r3_analysis_aggregate" / "suspicious_gate_audit.csv")))
    aud = []
    for a in aud_in:
        a2_ = dict(a)
        a2_["cleared_original"] = a.get("cleared")
        a2_["cleared"] = "PARTIAL"
        a2_["cleared_note"] = ("leakage/billing/censoring checks pass; OPT_K prefix check NOT RUN in round 3 "
                               "(cum_prefix_ge_OPT_K not computable) -> cannot be marked cleared=True")
        aud.append(a2_)
    write_csv("item13_suspicious_gate_audit_corrected.csv", aud)
    drift = []
    for f, h in (lock.get("code_sha256") or {}).items():
        p = CODE / f
        cur = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        if cur != h:
            drift.append({"file": f, "lock_sha256": h[:16], "current_sha256": (cur or "missing")[:16]})
    devs = [
        {"item": "Q2 placebo", "locked": "GLOBAL permutation placebo (prereg_lock Q2_placebo)",
         "executed": "aggregate gated on BLOCK placebo; global reported alongside",
         "class": "post-lock analysis deviation",
         "impact": (f"original sample: block {q2a.get('auc_placebo_block_orientation_free'):.4f}, global "
                    f"{q2a.get('auc_placebo_global_orientation_free'):.4f}; both <= 0.6 -> no change of point "
                    "reading; Q2 is NOT_EVALUABLE anyway (measurement design)")},
        {"item": "OPT_K prefix check", "locked": "cum prefix >= OPT_K for suspicious gates",
         "executed": "not run in round 3", "class": "unexecuted check",
         "impact": "suspicious_gate_audit cleared=True was overstated -> corrected to PARTIAL (5 triggers)"},
        {"item": "H0", "locked": "certifier soundness", "executed": "0 dual/BnB calls (exact enumeration)",
         "class": "vacuous pass", "impact": "no evidence about dual/BnB soundness"},
        {"item": "source hash drift", "locked": f"{len(lock.get('code_sha256') or {})} files hashed",
         "executed": f"{len(drift)} differ now: " + ", ".join(d['file'] for d in drift),
         "class": "post-lock runner/metadata edits", "impact": "results produced by locked sha per DONE records; diff archive required"},
    ]
    write_csv("item13_deviations.csv", devs)
    S["item13_audit"] = {"suspicious_gate_audit_corrected": "item13_suspicious_gate_audit_corrected.csv",
                         "n_audit_rows": len(aud), "hash_drift": drift, "deviations": devs}
    log("item 13 done")

    # ============================================================== item 1 additions from other items
    add("orth/off NL constructible (JPC)", -0.82,
        next(({"est": o["log_ratio"], "ci": o["ci"], "n_inst": o["n_inst"]} for o in orth_rows
              if o["layer"] == "NL-R0" and o["method"] == "JPC" and o["contrast"] == "orth/off"
              and o["subset"].startswith("constructible")), {}),
        "non-tie, constructible non-trivial problems, matched keys", "NL-R0", "instances with >=1 constructible problem",
        TAU_NL, agg_inst, "r3_nl_main", "[CX]")
    add("orth/off Lin constructible (JPC-Lin)", -0.62,
        next(({"est": o["log_ratio"], "ci": o["ci"], "n_inst": o["n_inst"]} for o in orth_rows
              if o["layer"] == "E1-Lin" and o["method"] == "JPC-Lin" and o["contrast"] == "orth/off"
              and o["subset"].startswith("constructible")), {}),
        "non-tie, constructible", "E1-Lin", "instances with >=1 constructible problem", TAU_LIN, agg_inst, "r3_lin", "[CX]")
    for o in orth_rows:
        if o["contrast"] == "orth/vol" and o["subset"].startswith("constructible") and o["method"] in ("JPC", "JPC-Lin"):
            add(f"A3o orth/vol {o['layer']} constructible (aggregate 16.8x/338x)",
                2.821 if o["layer"] == "NL-R0" else 5.823, {"est": o["log_ratio"], "ci": o["ci"], "n_inst": o["n_inst"]},
                "non-tie constructible-only (infeasible EXCLUDED)", o["layer"], "constructible", None, agg_inst,
                "r3_*", "outline's claim that 16.8x/338x includes ORTH_INFEASIBLE rows is wrong")
    for tau, ovn, ovm in ((1500, 2.00, 0.39), (3000, 2.49, 0.41), (6000, 2.86, 0.41)):
        t = [x for x in trig if x["tau"] == tau and x["comparison"].startswith("matched")][0]
        add(f"zero-effect saving tau={tau} (matched)", ovn, {"est": t["saving_null"], "ci": t["saving_null_ci"],
                                                              "n_inst": t["n_inst_null"]},
            "JPC/B3 full, all-tie problems", "zero-effect NL", "48 inst stream 0", tau, "-mean log ratio", "r3_controls_zero_effect", "[CX]")
        add(f"main non-tie saving tau={tau} (matched)", ovm, {"est": t["saving_main"], "ci": t["saving_main_ci"],
                                                               "n_inst": t["n_inst_main"]},
            "JPC/B3 full, non-tie", "NL-R0", "same 48 inst stream 0", tau, "-mean log ratio", "r3_nl_main", "[CX]")
    t = [x for x in trig if x["tau"] == 3000 and x["comparison"].startswith("locked")][0]
    add("Tier-C trigger null saving (locked)", 2.495, {"est": t["saving_null"], "ci": t["saving_null_ci"],
                                                       "n_inst": t["n_inst_null"]}, "JPC/B3 full", "zero-effect",
        "48 inst stream 0", 3000, "-mean log ratio", "r3_controls_zero_effect")
    add("Tier-C trigger main saving (locked)", 0.555, {"est": t["saving_main"], "ci": t["saving_main_ci"],
                                                       "n_inst": t["n_inst_main"]}, "JPC/B3 full non-tie", "NL-R0",
        "120 inst streams 0-2", 3000, "-mean log ratio", "r3_nl_main")
    add("B3 full zero-effect censored cost share tau=3000", 0.6377, {"est": b3f["censored_cost_share"]},
        "share of billed cost from uncertified-at-tau problems", "zero-effect", "48 inst stream 0, 720 problems", 3000,
        "sum ratio", "r3_controls_zero_effect", f"n uncertified {b3f['n_uncertified_at_tau']}/720 [CX: 105]")
    for c_, ov in ((1, 0.2222), (10, 0.2278), (50, 0.2477)):
        x = [a for a in a2 if a["layer"] == "NL-R0" and a["variant"] == f"smoothing c={c_}"][0]
        add(f"A2 NL smoothing c={c_}", ov, {"est": x["est"], "ci": x["ci"], "n_inst": x["n_inst"]}, "non-tie I",
            "NL-R0", "120 inst", TAU_NL, "mean log((S+c)/(S'+c)) diff", "r3_nl_main", "[CX]")
    for g_, a_, lo_, hi_ in ((1.0, "full", -0.0080, 0.0031), (1.0, "ev200", -0.0125, -0.0005),
                             (1.0, "probe20", -0.0115, -0.0009)):
        x = [c for c in c2 if c["dose"] == g_ and c["arm"] == a_][0]
        add(f"C2 g=1 {a_}-off FCR diff (CI low)", lo_, {"est": x["diff_ci95"][0]}, "pooled FCR difference, paired cluster",
            "static g=1", f"{x['n_instances']} inst", None, "lower CI bound", "r3_static", f"point {x['diff_arm_minus_off']} [CX]")
        add(f"C2 g=1 {a_}-off FCR diff (CI high)", hi_, {"est": x["diff_ci95"][1]}, "pooled FCR difference", "static g=1",
            f"{x['n_instances']} inst", None, "upper CI bound", "r3_static", "[CX]")
    for a_, ov in (("off", 0.173), ("ev1", 0.170), ("ev20", 0.170), ("ev200", 0.166), ("full", 0.170), ("probe20", 0.166)):
        x = [c for c in c2 if c["dose"] == 1.0 and c["arm"] == (a_ if a_ != "off" else "full")][0]
        v = x["fcr_off"] if a_ == "off" else x["fcr_arm"]
        add(f"C2 FCR g=1 {a_}", ov, {"est": v}, "false certs / certified problems", "static g=1", "48 inst", None,
            "pooled proportion", "r3_static")
    for cell, ov in (((2, 2), 0.0586), ((1, 2), 0.0247)):
        x = [c for c in cells9 if c["binning"] == "ante_muflip x post" and c["caliber"] == "post_bin_near"
             and (c["ante_bin"], c["post_bin"]) == cell][0]
        add(f"B-adv near ante{cell[0]}/post{cell[1]} cluster CI low", ov, {"est": x["ci95"][0]}, "cluster bootstrap FCR",
            "R1-adv", f"{x['n_instances']} inst, {x['n_false']}/{x['n_cert']}", None, "lower CI", "r3_hazard", "[CX]")
        add(f"B-adv near ante{cell[0]}/post{cell[1]} cluster CI high", 0.2202 if cell == (2, 2) else 0.2745,
            {"est": x["ci95"][1]}, "cluster bootstrap FCR", "R1-adv", f"{x['n_instances']} inst", None, "upper CI",
            "r3_hazard", "[CX]")
    add("Q1 S2 AUC pooled", 0.604, Q1["pooled"]["S2"] and {"est": Q1["pooled"]["S2"]["auc"], "ci": Q1["pooled"]["S2"]["ci"]},
        "AUC false_cert among zero-cost near/clear certs, k>=1", "risk layers", f"n={Q1['pooled']['n']}", None, "AUC",
        "static+x1+hazard")
    add("Q1 S1 (raw Lambda_perp) AUC pooled", 0.4056, {"est": Q1["pooled"]["S1"]["auc"], "ci": Q1["pooled"]["S1"]["ci"]},
        "AUC", "risk layers", f"n={Q1['pooled']['n']}", None, "AUC", "static+x1+hazard", "[CX]")
    add("Q1 gap baseline AUC", 0.9359, {"est": Q1["pooled"]["neg_log_gap_hat"]["auc"],
                                        "ci": Q1["pooled"]["neg_log_gap_hat"]["ci"]}, "AUC", "risk layers", "", None,
        "AUC", "static+x1+hazard", "[CX]")
    add("Q1 positives in eta_dec<=eps", 7, {"est": Q1["scope_eta_dec_le_eps"]["n_pos"]}, "count", "risk layers", "", None,
        "count", "static+x1+hazard")
    add("T0 bound/eps p90 (eta<=2eps)", 297.138, {"est": t0["eta_le_2eps"]["bound_over_eps_p90"]}, "computable bound",
        "dev 720-739", "", None, "p90", "r3_p4")
    add("T0 oracle exact |Rem|/eps p90 (eta<=2eps)", 0.006165, {"est": t0["eta_le_2eps"]["exact_over_eps_p90"]},
        "harness exact remainder", "dev 720-739", "", None, "p90", "r3_p4", "[CX]")
    add("HD2 policy kappa_eff slope", 2.6139, {"est": [h for h in hd2 if h["partition"] == "policy"][0]["slope"]},
        "kappa_eff slope", "E1-Lin HD2", "", None, "OLS slope", "r3_replicate_hd1b_hd2", "[CX]")
    add("HD2 beta-corrected pooled slope", 1.13684, {"est": [h for h in hd2 if h["partition"] == "pooled"
                                                            and h["estimand"].startswith("beta")][0]["slope"]},
        "beta-corrected slope (different estimand)", "E1-Lin HD2", "", None, "OLS slope", "r3_replicate_hd1b_hd2", "[CX]")
    write_csv("item1_metric_consistency.csv", met)
    S["item1_metric_consistency"] = met
    S["n_metric_rows"] = len(met)
    S["n_metric_mismatch"] = sum(1 for m in met if m["match"] is False)
    S["metric_mismatches"] = [m["item"] for m in met if m["match"] is False]

    S["wall_clock_s"] = round(time.time() - T0W, 1)
    S["finished_at"] = datetime.now().isoformat()
    S["log"] = LOGL
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    write_md(S)
    log(f"done in {S['wall_clock_s']} s; mismatches {S['metric_mismatches']}")


def write_md(S):
    """Auto-generated compact Chinese readout (task output spec: summary.json + summary.md + csv)."""
    f = lambda x, d=3: "NA" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))  # noqa: E731
    o = S["item2_orth"]["table"]
    g = lambda L, m, c: [x for x in o if x["layer"] == L and x["method"] == m and x["contrast"] == c  # noqa: E731
                         and x["subset"].startswith("constructible")][0]
    t0, q1, q2, q3 = S["item3_T0"], S["item5_Q1"], S["item6_Q2_static_open"], S["item7_Q3_refit"]
    hd = {(h["estimand"][:8], h["partition"]): h for h in S["item4_HD2"]["table"]}
    c2 = S["item8_C2_paired"]["table"]
    c2s = "; ".join(f"g={c['dose']} {c['arm']}-off {f(c['diff_arm_minus_off'], 4)} {f(c['diff_ci95'][0], 4)}..{f(c['diff_ci95'][1], 4)}"
                    for c in c2)
    cz = S["item10_zero_effect"]["B3_full_tau3000"]
    trig = [t for t in S["item10_zero_effect"]["trigger"] if t["comparison"].startswith("matched")]
    a2 = [a for a in S["item11_A2_scale"] if a["layer"] == "NL-R0"]
    sp = [x for x in S["item12_split"]["table"] if x["layers"] == "non-tie"]
    cells = [c for c in S["item9_Badv_cluster"]["cells"] if c["binning"] == "ante_muflip x post"
             and c["caliber"] == "post_bin_near" and c["post_bin"] == 2]
    L = [
        "# r3_p0_reanalysis：P0 重分析（EXPLORATORY / post-hoc）", "",
        f"> {S['status']}。新增环境交互 0、新种子 0；实例簇 bootstrap（B=10^4，logistic 2000），种子 42。"
        "脚本 exp/code/run_r3_p0_reanalysis.py；机器表 summary.json 与 item*.csv。", "",
        f"- 口径表 item1_metric_consistency.csv：{S['n_metric_rows']} 行，与 outline / [CX] 不一致 {S['n_metric_mismatch']} 行。",
        "- B3 NL full/off：-2.155 = 非平局主终点（near+clear，τ=3000，120 实例）；-1.341 = 含 tie 全流（quota_main）。"
        "全流口径的 JPC 为 -1.676，不是 -1.93。Lin -1.87/-2.01 = r3_lin_b 16 实例中位数（描述性）。",
        f"- orth 可构造题：NL log(orth/off) {f(g('NL-R0','JPC','orth/off')['log_ratio'])} {g('NL-R0','JPC','orth/off')['ci']}"
        f"（{g('NL-R0','JPC','orth/off')['n_inst']} 实例）；Lin {f(g('E1-Lin','JPC-Lin','orth/off')['log_ratio'])} "
        f"{g('E1-Lin','JPC-Lin','orth/off')['ci']}（{g('E1-Lin','JPC-Lin','orth/off')['n_inst']}）。log(orth/vol) NL "
        f"{f(g('NL-R0','JPC','orth/vol')['log_ratio'])} / Lin {f(g('E1-Lin','JPC-Lin','orth/vol')['log_ratio'])}。"
        "聚合表 A3o 本来就只用可构造题，outline 中“16.8×/338× 含 ORTH_INFEASIBLE”的说法有误。",
        f"- T0（开发种子 720-739）：η≤2ε 时上界 p90 {f(t0['eta_le_2eps']['bound_over_eps_p90'])}，"
        f"真实 |Rem|/ε p90 {f(t0['eta_le_2eps']['exact_over_eps_p90'], 5)}。",
        f"- HD2：κ_eff 斜率 time {f(hd[('kappa_ef','time')]['slope'])} / participant {f(hd[('kappa_ef','participant')]['slope'])} / "
        f"policy {f(hd[('kappa_ef','policy')]['slope'])} {hd[('kappa_ef','policy')]['ci']}，门槛 [1.5,2.5]，只有 policy 失败；"
        f"β 修正 pooled 斜率 {f(hd[('beta-cor','pooled')]['slope'])} 是另一估计对象（门槛 [0.85,1.15]，通过）。",
        f"- Q1：S2 {f(q1['pooled']['S2']['auc'])} {q1['pooled']['S2']['ci']}；S1 {f(q1['pooled']['S1']['auc'])} "
        f"{q1['pooled']['S1']['ci']}；间隙 {f(q1['pooled']['neg_log_gap_hat']['auc'])}；η_dec≤ε 范围内正例 "
        f"{q1['scope_eta_dec_le_eps']['n_pos']}。",
        f"- Q2 受限（static_open {q2.get('n_used')} 行，{q2.get('n_clusters')} 簇）：簇稳健 Wald p "
        f"{f(q2['cluster_robust_wald_logAk']['p_two_sided'])}；实例外 ΔAUC {q2['out_of_instance_cv']['delta_auc']}；"
        f"全局安慰剂 {f(q2['auc_placebo_global_orientation_free'])}。不通过；正式 Q2 仍判 NOT_EVALUABLE。",
        f"- Q3：控制间隙和 log n_obs 后，β_logΛ = {f(q3['models'][2]['beta_logLambda'])} {q3['models'][2]['boot_ci_logLambda']}，"
        f"负号保留：{q3['sign_survives_gap_and_volume']}。",
        f"- C2 配对 FCR 差：{c2s}。",
        "- B-adv near 口径 post2：" + "; ".join(f"ante{c['ante_bin']} {c['n_false']}/{c['n_cert']}（{c['n_instances']} 实例）"
                                          f" CI {c['ci95']}" for c in cells) + "。",
        f"- 零效应 B3 full（τ=3000）：未认证 {cz['n_uncertified_at_tau']}/720，删失计费占比 {f(cz['censored_cost_share'], 4)}；"
        "同实例、同流匹配后的节省（零效应 vs 主环境）："
        + "; ".join(f"τ={t['tau']} {f(t['saving_null'])} vs {f(t['saving_main'])} {t['saving_main_ci']}" for t in trig) + "。",
        "- A2 NL：" + "; ".join(f"{a['variant']} {f(a['est'])} {a['ci'] or ''}" for a in a2) + "。",
        "- 0.57× 拆分（非平局）：" + "; ".join(f"{x['component'].split(' (')[0]} ×{f(x['ratio'])}" for x in sp)
        + f"；off/full 错误认证 {S['item12_split']['false_cert_off_full']}。两个因子的“阈值 / 支撑”标签不可识别。",
        "- 审计：item13_suspicious_gate_audit_corrected.csv（OPT_K 未运行，cleared 改为 PARTIAL）；item13_deviations.csv"
        "（Q2 安慰剂锁定为 global、执行用了 block；H0 空通过；源文件哈希漂移）。",
    ]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
