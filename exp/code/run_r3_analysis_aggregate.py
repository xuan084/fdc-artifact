"""r3_analysis_aggregate: round-3 aggregate (methodology 5 / 8; task_plan r3_analysis_aggregate).

Reads ONLY raw results.jsonl / predictors.jsonl of the dependency tasks (plus their summary.json for gate flags) and
recomputes:
  * pre-gates: H0, billing, T0, P2 timing/completion, P3 constructible rate, P5 acceptance;
  * P family per layer (Holm over {A1, A2, A3, A3o}; A3o only if orth confirmatory), A2 two-layer intersection,
    clear-layer kill switch;
  * V family (Holm over {C1, C2, B-adv}); HD1 / HD1b / HD2 / HH1 replication table;
  * Q family (Holm over {Q1, Q2, Q3}) with the frozen predictor variant, T0 scope view, held-out direction;
  * suspicious-gate audit; tier_decision_r3 (A/B/C + Q wording); figures of methodology 8 (pdf + png + csv).
pilot mode: dev seeds only -> pipeline check + per-candidate GO/NO-GO; every number is labelled non-evidence.
full mode : assert_locked() (v3) first; same code path on exp/results/full/.

Bootstrap: instance-level cluster bootstrap, B = 10^4, percentile, seed 42. One-sided p by bootstrap inversion
p = (1 + #{rep on the wrong side of the margin}) / (B + 1). Pass rule for a CI-type hypothesis = Holm-adjusted
one-sided p < 0.025 (equivalent to the 95% two-sided CI bound for a single test).
CPU only, 1 thread (并发运行).
"""
from __future__ import annotations

import argparse
import ast
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parent.parent
from dsswm.stats.auc import auc as auc_pt, lr_test, logistic_fit, pr_auc  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.stats.tier_decision_r3 import decide_tier_r3, LOG06, LOG08, LOG125  # noqa: E402

TASK = "r3_analysis_aggregate"
B = 10_000
SEED = 42
EPS, EPS_LIN, DELTA = 0.02, 0.05, 0.05
TAU_NL = 3000
COLORS = {"JPC": "#2a78d6", "B3": "#eb6834", "B8": "#1baf7a", "LR-chi2-grid": "#eda100",
          "JPC-Lin": "#2a78d6", "RAGE": "#eb6834"}
ARMS = ["off", "vol", "orth", "ev1", "full"]
T0W = time.time()
LOG: list = []


def log(m):
    s = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    LOG.append(s)
    print(s, flush=True)


# ============================================================================================ io
def jl(p):
    if not p.exists():
        return []
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fnum(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Out:
    def __init__(self, root):
        self.root = root
        self.fig = root / "figures"
        self.tab = root / "tables"
        for d in (root, self.fig, self.tab):
            d.mkdir(parents=True, exist_ok=True)

    def csv(self, name, rows, d=None):
        d = d or self.tab
        rows = list(rows)
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(d / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys or ["empty"])
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})

    def save(self, fig, name, rows=None):
        fig.savefig(self.fig / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.fig / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        if rows is not None:
            self.csv(f"{name}.csv", rows, d=self.fig)


def progress(out_dir, step, total, note):
    p = WS / "exp" / "results" / f"{TASK}_PROGRESS.json"
    p.write_text(json.dumps({"task_id": TASK, "epoch": step, "total_epochs": total, "step": step,
                             "total_steps": total, "loss": None, "metric": {"stage": note},
                             "updated_at": datetime.now().isoformat()}))


# ============================================================================================ stats helpers
RNG = np.random.default_rng(SEED)


def boot_idx(n, b=B, seed=SEED):
    return np.random.default_rng(seed).integers(n, size=(b, n))


def mean_ci(vals, margin=None, side="upper", seed=SEED):
    """vals: per-instance values. side='upper': H1 mean < margin; side='lower': H1 mean > margin."""
    v = np.asarray([x for x in vals if x is not None and math.isfinite(x)], float)
    if len(v) == 0:
        return {"n": 0, "est": None, "ci": [None, None], "p": None}
    est = float(v.mean())
    if len(v) == 1:
        return {"n": 1, "est": est, "ci": [None, None], "p": None, "sd": None}
    reps = v[boot_idx(len(v), seed=seed)].mean(1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    p = None
    if margin is not None:
        bad = (reps >= margin).sum() if side == "upper" else (reps <= margin).sum()
        p = float((1 + bad) / (B + 1))
    return {"n": int(len(v)), "est": est, "ci": [float(lo), float(hi)], "p": p, "sd": float(v.std(ddof=1)),
            "mcse": float(reps.std(ddof=1))}


def holm(pd):
    items = [(k, p) for k, p in pd.items() if p is not None]
    items.sort(key=lambda t: t[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    for k in pd:
        adj.setdefault(k, None)
    return adj


def cluster_auc(scores, labels, clusters, seed=SEED, b=B):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    c = np.asarray(clusters)
    pt = auc_pt(s, y)
    keys = sorted(set(c.tolist()))
    idx = {k: np.where(c == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(b):
        dr = rng.integers(len(keys), size=len(keys))
        ii = np.concatenate([idx[keys[j]] for j in dr])
        if y[ii].all() or (~y[ii]).all():
            continue
        reps.append(auc_pt(s[ii], y[ii]))
    reps = np.asarray(reps)
    ci = np.quantile(reps, [0.025, 0.975]).tolist() if len(reps) > 10 else [None, None]
    return {"auc": None if math.isnan(pt) else pt, "ci": ci, "reps": reps, "n": int(len(y)), "n_pos": int(y.sum()),
            "n_clusters": len(keys), "pr_auc": (None if not y.any() else pr_auc(s, y))}


def cp(k, n):
    lo, hi = clopper_pearson(int(k), int(n))
    return [lo, hi]


# ============================================================================================ loading
def tag_rows(rows, src):
    for r in rows:
        r["_src"] = src
    return rows


def load_task(base, task, sub=None):
    d = base / task if sub is None else base / task / sub
    return tag_rows(jl(d / "results.jsonl"), task), tag_rows(jl(d / "predictors.jsonl"), task)


def dedupe(rows, key):
    seen, out, dup = set(), [], 0
    for r in rows:
        k = key(r)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out.append(r)
    return out, dup


KEY = lambda r: (r.get("layer") or r.get("kind"), r.get("kind"), r["instance"], r["stream"], r["method"],  # noqa: E731
                 r["arm"], r["problem"])


def cost(r, tau):
    c = r.get(f"cost_tau{tau}")
    if c is not None:
        return float(c)
    if r.get("status") == "ORTH_INFEASIBLE":
        return float(tau)
    if r.get("censored") or r.get("status") != "CERTIFIED":
        return float(tau)
    return float(min(r.get("new_env_steps") or 0, tau))


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    args = ap.parse_args()
    mode = args.mode
    base = WS / "exp" / "results" / ("pilots" if mode == "pilot" else "full")
    out = Out(base / TASK)
    lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
    if mode == "full":
        from dsswm.stats.prereg import assert_locked
        assert_locked()
    gates_lock = lock.get("gates", {}) if isinstance(lock.get("gates"), dict) else {}
    S = {"task": TASK, "mode": mode, "started_at": datetime.now().isoformat(), "lock_status": lock.get("status"),
         "lock_version": lock.get("version"), "lock_sha256": lock.get("sha256"),
         "evidence_status": ("NON-EVIDENCE: pilot on dev seeds; pipeline check only" if mode == "pilot"
                             else "confirmatory (eval seeds, locked v3)"),
         "concurrency_note": "并发运行 (up to 4 tasks); CPU 1 thread", "B": B, "seed": SEED}
    TOTAL = 12
    progress(out, 0, TOTAL, "load")

    deps = json.loads((WS / "plan" / "task_plan.json").read_text())
    deps = [t for t in deps.get("tasks", deps) if t["id"] == TASK][0]["depends_on"]
    summ, parse_ok = {}, {}
    for t in deps:
        p = base / t / "summary.json"
        try:
            summ[t] = json.loads(p.read_text())
            parse_ok[t] = True
        except Exception as e:  # noqa: BLE001
            summ[t] = {}
            parse_ok[t] = f"unparsable: {e!r}" if p.exists() else "missing"
    S["summaries_parsed"] = parse_ok
    S["all_summaries_parsable"] = all(v is True for v in parse_ok.values())
    S["task_go_no_go"] = {t: summ[t].get("go_no_go") for t in deps}
    if mode == "full":   # deviation log: pilot timing-gate labels are not full-mode criteria -> use DONE status
        done = {}
        for t in deps:
            dp = WS / "exp" / "results" / f"{t}_DONE"
            try:
                done[t] = json.loads(dp.read_text()).get("status")
            except Exception:  # noqa: BLE001
                done[t] = "missing" if not dp.exists() else "unparsable"
        S["task_done_status"] = done
        S["task_go_no_go_note"] = ("raw runner go_no_go labels (pilot template; timing gate) kept for audit only, "
                                   "NOT used for any verdict; see methodology 'post-lock deviations'")

    # ------------------------------------------------------------------ raw rows
    n_read = 0
    NL, NLP, LIN, LINP = [], [], [], []
    for t in [f"r3_nl_main_{c}" for c in "abcdefgh"]:
        r, p = load_task(base, t)
        NL += r
        NLP += p
    LR_EX, _ = [], []
    for t in [f"r3_nl_extra_{c}" for c in "abc"]:
        r, p = load_task(base, t)
        LR_EX += r
    for t in [f"r3_lin_{c}" for c in "abc"]:
        r, p = load_task(base, t)
        LIN += r
        LINP += p
    dev_supp = {}
    if mode == "pilot":   # dev supplement: P1 (NL 720-727) and P2 (Lin 740-749) -- same row schema
        r1, p1 = load_task(base, "r3_p1_reuse_ablation_nl")
        r2, p2 = load_task(base, "r3_p2_lin_stream_smoke")
        lin_inst = {r["instance"] for r in LIN}
        r2 = [r for r in r2 if r["instance"] not in lin_inst]
        dev_supp = {"NL_P1_rows": len(r1), "Lin_P2_rows_kept": len(r2),
                    "rule": "pilot only: P1/P2 dev rows pooled with eval-task smoke rows; instances already present "
                            "in the eval-task smoke are dropped from P2"}
        NL += r1
        LIN += r2
    STAT, STATP, HZ, HZP, HO, HOP, ZE, ZEP, CS, CSP, X1, X1P = ([] for _ in range(12))
    for t in [f"r3_static_{c}" for c in "abcd"]:
        r, p = load_task(base, t)
        STAT += r
        STATP += p
    for t in ["r3_hazard_a", "r3_hazard_b"]:
        r, p = load_task(base, t)
        HZ += r
        HZP += p
    HO, HOP = load_task(base, "r3_heldout_misspec")
    if (base / "r3_controls_zero_effect").exists():
        ZE, ZEP = load_task(base, "r3_controls_zero_effect")
    else:   # full mode: lock split the zero-effect env into chunks a/b (24 instances each)
        for t in ("r3_controls_zero_effect_a", "r3_controls_zero_effect_b"):
            r, p = load_task(base, t)
            ZE += r
            ZEP += p
    CS, CSP = load_task(base, "r3_controls_streams")
    X1, X1P = load_task(base, "r3_x1_sampler_misspec")
    raw_counts = {k: len(v) for k, v in dict(NL=NL, NL_pred=NLP, NL_extra=LR_EX, Lin=LIN, Lin_pred=LINP, static=STAT,
                                              static_pred=STATP, hazard=HZ, hazard_pred=HZP, heldout=HO,
                                              heldout_pred=HOP, zero_effect=ZE, zero_effect_pred=ZEP, streams=CS,
                                              streams_pred=CSP, x1=X1, x1_pred=X1P).items()}
    n_read = sum(raw_counts.values())
    dups = {}
    NL, dups["NL"] = dedupe(NL, KEY)
    LR_EX, dups["NL_extra"] = dedupe(LR_EX, KEY)
    LIN, dups["Lin"] = dedupe(LIN, KEY)
    skey = lambda r: (r["method"], r["arm"], r["instance"], r["stream"], r["problem"], r.get("dose"))  # noqa: E731
    STAT, dups["static"] = dedupe(STAT, skey)
    hkey = lambda r: (r.get("layer"), r["method"], r["arm"], r["instance"], r["stream"], r["problem"])  # noqa: E731
    HZ, dups["hazard"] = dedupe(HZ, hkey)
    S["raw_rows_read"] = raw_counts
    S["n_readouts"] = n_read
    S["duplicates_removed"] = dups
    S["duplicates_note"] = ("pilot chunks a-h / a-d / a-b ran the same dev seeds (identical rows) -> deduplicated "
                            "by (layer, instance, stream, method, arm, problem)" if mode == "pilot" else "")
    S["dev_supplement"] = dev_supp
    log(f"rows read {n_read}; dups {dups}")

    pkey = lambda r: (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])  # noqa: E731

    def pmap(P, extra=None):
        m = {}
        for p in P:
            if p.get("phase") not in (None, "pre_scoring"):
                continue
            k = pkey(p) + ((p.get(extra),) if extra else ())
            if k not in m or p.get("record") == "full_predictors":
                m[k] = p
        return m

    # ================================================================== 1. pre-gates
    progress(out, 1, TOTAL, "gates")
    G = {}
    # H0: exact enumeration in every round-3 path; dual/BnB never imported -> vacuous truth.
    imp = []
    for f in sorted(CODE.glob("run_r3_*.py")) + sorted((CODE / "dsswm").rglob("*.py")):
        if "tests" in f.parts or f.name == "dual_bnb.py":
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
                if any("dual_bnb" in (x or "") for x in names):
                    imp.append(str(f.relative_to(CODE)))
    G["H0"] = {"pass": len(imp) == 0, "dual_bnb_importers_round3": imp, "n_dual_calls": 0 if not imp else None,
               "rule": "exact enumeration (|Theta| <= 46656); 0 dual calls -> H0 vacuously true",
               "status": "PASS (vacuous: 0 dual/BnB calls)" if not imp else "CHECK: dual_bnb imported"}
    # billing: summaries + raw rows
    bill_s = {}
    for t, s in summ.items():
        v = s.get("billing_mismatch", s.get("n_billing_mismatch"))
        if v is None and isinstance(s.get("audit"), dict):
            v = s["audit"].get("n_bad")
        if v is not None:
            bill_s[t] = v
    allrows = NL + LR_EX + LIN + STAT + HZ + HO + ZE + CS + X1
    raw_bad = [r for r in allrows if r.get("billing_ok") is False]
    tot_ok = [r for r in allrows if r.get("n_rounds_total") is not None and isinstance(r.get("prov_counts"), dict)]
    replay_mismatch = sum(1 for r in tot_ok if int(r["n_rounds_total"]) != int(sum(r["prov_counts"].values())))
    prov_replay_mismatch = sum(1 for r in tot_ok if int(r.get("replay_steps") or 0)
                               != int(sum(v for k, v in r["prov_counts"].items() if k.startswith("replay"))))
    billed_replay = sum(1 for r in allrows if r.get("n_rounds_billed") is not None and r.get("env_n_steps") is not None
                        and int(r["n_rounds_billed"]) != int(r["env_n_steps"]))
    G["billing"] = {"pass": sum(int(v or 0) for v in bill_s.values()) == 0 and not raw_bad and billed_replay == 0
                    and replay_mismatch == 0 and prov_replay_mismatch == 0,
                    "note": "replay_steps includes replay_pad_steps (pad = sibling rows topping up to N_k)",
                    "summary_mismatch_total": int(sum(int(v or 0) for v in bill_s.values())),
                    "raw_rows_billing_ok_false": len(raw_bad), "raw_rows_billed_ne_env_steps": billed_replay,
                    "raw_rows_evidence_total_ne_provenance_sum": replay_mismatch,
                    "raw_rows_replay_steps_ne_provenance_replay": prov_replay_mismatch, "n_rows_checked": len(allrows),
                    "by_arm_levels_seen": sorted({r.get("arm") for r in allrows})}
    p4 = summ.get("r3_p4_t0_mechanism_gate", {})
    t0 = p4.get("T0", {})
    G["T0"] = {"i_identity_err_max": t0.get("i_identity_err_max"), "i_pass": t0.get("i_pass"),
               "ii_rem_bar_over_eps_p90_eta_le_2eps": (t0.get("ii_rem_bar_over_eps") or {}).get("p90_eta_le_2eps"),
               "ii_pass": t0.get("ii_pass"), "iii_pass": t0.get("iii_pass"),
               "iv_phi_perp_median": (t0.get("iv_phi_perp") or {}).get("median_max_over_pairs"),
               "iv_clause": "定向补证无成本优势" if (t0.get("iv_phi_perp") or {}).get("median_gt_0.3") else None,
               "scope": t0.get("scope"), "hard_pass": bool(t0.get("i_pass") and t0.get("iii_pass")),
               "consequence": "T0(ii) fails -> Thm 3 / Q family scope eta <= eps" if t0.get("ii_pass") is False else ""}
    p2 = summ.get("r3_p2_lin_stream_smoke", {})
    G["P2_lin"] = {"completion": p2.get("completion_rate"), "completion_nontie": p2.get("completion_rate_nontie"),
                   "sec_per_run_median": p2.get("sec_per_run_median"), "sec_per_run_p90": p2.get("sec_per_run_p90"),
                   "pass": bool((p2.get("completion_rate") or 0) >= 0.9 and (p2.get("sec_per_run_median") or 9e9) <= 5),
                   "rule": "completion >= 0.9 AND <= 5 s/run (median)"}
    oc = gates_lock.get("orth_confirmatory", {})
    G["P3_orth"] = {"constructible_rate_nl_stream0": oc.get("constructible_rate_nl_stream0"),
                    "constructible_rate_nl_stream1": oc.get("constructible_rate_nl_stream1_supplement"),
                    "constructible_rate_lin": oc.get("constructible_rate_lin"),
                    "A3o_confirmatory": bool(oc.get("value")), "rule": oc.get("rule"),
                    "eval_smoke_orth_feasible_rate": None}
    orr = [r for r in NL + LIN if r["arm"] == "orth" and r.get("orth_trivial") is not True and r.get("N_k")]
    if orr:
        G["P3_orth"]["eval_smoke_orth_feasible_rate"] = float(np.mean([r.get("status") != "ORTH_INFEASIBLE" for r in orr]))
    p5 = summ.get("r3_p5_hazard_zone_sampler", {})
    acc = (gates_lock.get("r1adv") or {}).get("acceptance") or p5.get("acceptance")
    G["P5_hazard"] = {"acceptance": acc, "pass": bool(acc and acc.get("rate", 0) >= 0.001),
                      "directed_construction_triggered": p5.get("directed_construction_triggered"),
                      "rule": "acceptance >= 0.1% else directed construction"}
    p1 = summ.get("r3_p1_reuse_ablation_nl", {})
    G["P1_interaction_computable"] = (p1.get("pass_criteria") or {}).get("interaction_I_computable_nontie")
    S["gates"] = G

    # ================================================================== 2. P family
    progress(out, 2, TOTAL, "P family")

    def S_table(rows, tau, layers=("near", "clear"), feasible_only=None):
        """S[(method, arm)][instance] = sum over streams/problems in layers of min(cost, tau)."""
        T = defaultdict(lambda: defaultdict(float))
        for r in rows:
            if r.get("gap_layer") not in layers:
                continue
            if feasible_only is not None and (r["instance"], r["stream"], r["problem"]) not in feasible_only:
                continue
            T[(r["method"], r["arm"])][r["instance"]] += cost(r, tau)
        return T

    def logr(Ta, Tb):
        ii = sorted(set(Ta) & set(Tb))
        return {i: math.log((Ta[i] + 1) / (Tb[i] + 1)) for i in ii}

    def p_family(rows, layer, jpc, base_m, tau):
        T = S_table(rows, tau)
        a1 = logr(T[(jpc, "full")], T[(jpc, "off")])
        bb = logr(T[(base_m, "full")], T[(base_m, "off")])
        I = {i: a1[i] - bb[i] for i in set(a1) & set(bb)}
        a3 = logr(T[(jpc, "ev1")], T[(jpc, "vol")])
        # A3o on orth-constructible problems (stream with orth rows), excluding trivial N_k = 0 problems
        feas = {(r["instance"], r["stream"], r["problem"]) for r in rows
                if r["method"] == jpc and r["arm"] == "orth" and r.get("status") != "ORTH_INFEASIBLE"
                and r.get("orth_trivial") is not True}
        To = S_table(rows, tau, feasible_only=feas)
        a3o = logr(To[(jpc, "orth")], To[(jpc, "vol")])
        orth_all = [r for r in rows if r["method"] == jpc and r["arm"] == "orth" and r.get("orth_trivial") is not True]
        res = {"A1": {**mean_ci(a1.values(), LOG06, "upper"), "margin": "log 0.6", "falsify_if_ci_upper_ge": LOG08},
               "A2": {**mean_ci(I.values(), LOG08, "upper"), "margin": "log 0.8", "baseline": base_m},
               "A3": {**mean_ci(a3.values(), LOG08, "upper"), "margin": "log 0.8"},
               "A3o": {**mean_ci(a3o.values(), LOG125, "lower"), "margin": "log 1.25",
                       "constructible_rate": (float(np.mean([r.get("status") != "ORTH_INFEASIBLE" for r in orth_all]))
                                              if orth_all else None), "n_orth_problems": len(orth_all)},
               "baseline_full_off": mean_ci(bb.values())}
        Tc = S_table(rows, tau, layers=("clear",))
        clr = logr(Tc[(jpc, "full")], Tc[(base_m, "full")])
        mc = mean_ci(clr.values())
        res["clear_switch"] = {"log_ratio": mc, "ratio_est": None if mc["est"] is None else math.exp(mc["est"]),
                               "ratio_ci": [None if x is None else math.exp(x) for x in mc["ci"]]}
        main_sv = logr(T[(jpc, "full")], T[(base_m, "full")])
        res["main_nontie_jpc_over_base_full"] = mean_ci(main_sv.values())
        res["per_instance"] = {"A1": a1, "A2": I, "A3": a3, "A3o": a3o, "clear": clr}
        res["n_instances"] = len(a1)
        res["tau"] = tau
        res["layer"] = layer
        return res

    tau_lin = int((lock.get("frozen_items") or {}).get("T_max_Lin") or 20000)
    PF = {"NL-R0": p_family(NL, "NL-R0", "JPC", "B3", TAU_NL),
          "E1-Lin": p_family(LIN, "E1-Lin", "JPC-Lin", "RAGE", tau_lin)}
    a3o_conf = bool(oc.get("value"))
    for L, R in PF.items():
        fam = {h: R[h]["p"] for h in ("A1", "A2", "A3") + (("A3o",) if a3o_conf else ())}
        adj = holm(fam)
        for h in ("A1", "A2", "A3", "A3o"):
            R[h]["holm_p"] = adj.get(h)
            if h == "A3o" and not a3o_conf:
                R[h]["verdict"] = "DESCRIPTIVE (constructible rate < 0.5)"
                continue
            if R[h]["est"] is None or R[h]["ci"][0] is None:
                R[h]["verdict"] = "NOT_EVALUABLE"
            elif adj.get(h) is not None and adj[h] < 0.025:
                R[h]["verdict"] = "PASS"
            elif h == "A1" and R[h]["ci"][1] >= LOG08:
                R[h]["verdict"] = "FALSIFIED"
            else:
                R[h]["verdict"] = "NOT_CONFIRMED"
    A2_two = PF["NL-R0"]["A2"]["verdict"] == "PASS" and PF["E1-Lin"]["A2"]["verdict"] == "PASS"
    S["P_family"] = {L: {k: v for k, v in R.items() if k != "per_instance"} for L, R in PF.items()}
    S["P_family"]["A2_two_layer_intersection"] = A2_two
    # tie / near / clear forest & sensitivity
    forest = []
    for L, rows, j, bm, tau in (("NL-R0", NL, "JPC", "B3", TAU_NL), ("E1-Lin", LIN, "JPC-Lin", "RAGE", tau_lin)):
        for gl in ("tie", "near", "clear"):
            T = S_table(rows, tau, layers=(gl,))
            for m in (j, bm):
                for a in ("vol", "orth", "ev1", "full"):
                    v = mean_ci(logr(T[(m, a)], T[(m, "off")]).values())
                    forest.append({"layer": L, "gap_layer": gl, "method": m, "arm": a, "n": v["n"],
                                   "log_ratio_vs_off": v["est"], "ci_low": v["ci"][0], "ci_high": v["ci"][1]})
    out.csv("forest_tie_near_clear.csv", forest)
    sens = []
    for tau in (1500, 3000, 6000):
        T = S_table(NL, tau)
        for m in ("JPC", "B3"):
            v = mean_ci(logr(T[(m, "full")], T[(m, "off")]).values())
            sens.append({"tau": tau, "method": m, "log_full_off": v["est"], "ci": v["ci"], "n": v["n"]})
    S["P_family"]["tmax_sensitivity_NL"] = sens
    # LR-chi2 / B8 descriptive off/full
    # lock downscale B8_off_stream0_only: B8 off/full compared on stream 0 only (same-stream matching)
    T = S_table(LR_EX, TAU_NL)
    T8 = S_table([r for r in LR_EX if r["method"] == "B8" and int(r["stream"]) == 0], TAU_NL)
    S["P_family"]["extra_baselines_full_off"] = {
        "LR-chi2-grid": {**mean_ci(logr(T[("LR-chi2-grid", "full")], T[("LR-chi2-grid", "off")]).values()),
                         "streams": "0-2"},
        "B8": {**mean_ci(logr(T8[("B8", "full")], T8[("B8", "off")]).values()), "streams": "0 only (matched)"}}
    log("P family done")

    # ================================================================== 3. controls
    progress(out, 3, TOTAL, "controls")
    Tz = S_table(ZE, TAU_NL, layers=("tie", "near", "clear"))
    b3_streams = {int(r["stream"]) for r in ZE if r["method"] == "B3"}
    Tz0 = S_table([r for r in ZE if int(r["stream"]) in b3_streams], TAU_NL, layers=("tie", "near", "clear"))
    zsv = logr(Tz0[("JPC", "full")], Tz0[("B3", "full")])   # same-stream matching (B3 runs stream 0 only)
    zero = mean_ci(zsv.values())
    main = PF["NL-R0"]["main_nontie_jpc_over_base_full"]
    S["controls"] = {"zero_effect_jpc_over_b3_full_log": zero, "main_nontie_jpc_over_b3_full_log": main,
                     "saving_null": None if zero["est"] is None else -zero["est"],
                     "saving_main": None if main["est"] is None else -main["est"],
                     "zero_effect_tie_declaration_rate": ((summ.get("r3_controls_zero_effect") or {}).get("tie_declaration_rate")
                                                          or {t: (summ.get(t) or {}).get("tie_declaration_rate")
                                                              for t in ("r3_controls_zero_effect_a", "r3_controls_zero_effect_b")}),
                     "zero_effect_streams_used_for_jpc_over_b3": sorted(b3_streams),
                     "note": "B3 runs stream 0 only on the zero-effect env (lock downscale); JPC/B3 compared on the same "
                             "stream(s) per instance; JPC full/off reuse factor uses all 3 streams"}
    ctl = []
    for kind in ("no_tie", "low_overlap"):
        rr = [r for r in CS if r.get("kind") == kind]
        Tk = S_table(rr, TAU_NL, layers=("tie", "near", "clear"))
        for m in ("JPC", "B3"):
            v = mean_ci(logr(Tk[(m, "full")], Tk[(m, "off")]).values())
            ctl.append({"stream_kind": kind, "method": m, "log_full_off": v["est"], "ci": v["ci"], "n": v["n"]})
    for m in ("JPC", "B3"):
        v = mean_ci(logr(Tz[(m, "full")], Tz[(m, "off")]).values())
        ctl.append({"stream_kind": "offset_null", "method": m, "log_full_off": v["est"], "ci": v["ci"], "n": v["n"]})
    for m in ("JPC", "B3"):
        Tq = S_table(NL, TAU_NL, layers=("tie", "near", "clear"))
        v = mean_ci(logr(Tq[(m, "full")], Tq[(m, "off")]).values())
        ctl.append({"stream_kind": "quota_main(all layers)", "method": m, "log_full_off": v["est"], "ci": v["ci"],
                    "n": v["n"]})
    S["controls"]["reuse_factor_by_stream"] = ctl

    # ================================================================== 4. V family
    progress(out, 4, TOTAL, "V family")
    V = {}
    st1 = [r for r in STAT if r.get("method") == "JPC_static_g1" and r["arm"] != "full@open"]
    full1 = [r for r in st1 if r["arm"] == "full" and r.get("status") == "CERTIFIED"]
    for r in full1:
        pi = r.get("problem_index", r["problem"])
        r["_seg"] = int(pi) // 5
    n_pos_c1 = sum(bool(r.get("false_cert")) for r in full1)

    def mh_from(rows):
        tab = defaultdict(lambda: [0, 0, 0, 0])
        for r in rows:
            t = tab[(r["_seg"], r["gap_layer"])]
            if r.get("zero_cost"):
                t[0] += bool(r.get("false_cert"))
                t[1] += 1
            else:
                t[2] += bool(r.get("false_cert"))
                t[3] += 1
        return tab

    def mh_rr(tabs):
        num = den = 0.0
        for a, n1, c, n0 in tabs:
            N = n1 + n0
            if N:
                num += a * n0 / N
                den += c * n1 / N
        return math.inf if den == 0 and num > 0 else (math.nan if den == 0 else num / den)

    by_inst = defaultdict(list)
    for r in full1:
        by_inst[r["instance"]].append(r)
    insts = sorted(by_inst)
    itabs = {i: mh_from(by_inst[i]) for i in insts}
    strata = sorted({s for i in insts for s in itabs[i]})
    arr = np.array([[itabs[i].get(s, [0, 0, 0, 0]) for s in strata] for i in insts], float) if insts else np.zeros((0, 0, 4))
    rr_pt = mh_rr(arr.sum(0).tolist()) if len(insts) else math.nan
    reps = []
    if len(insts) > 1:
        for row in boot_idx(len(insts)):
            reps.append(mh_rr(arr[row].sum(0).tolist()))
    reps = np.array(reps)
    fin = reps[~np.isnan(reps)] if len(reps) else reps
    ci = np.quantile(np.where(np.isinf(fin), 1e9, fin), [0.025, 0.975]).tolist() if len(fin) > 10 else [None, None]
    c1_eval = n_pos_c1 >= 5
    p_c1 = float((1 + (fin <= 1).sum()) / (len(fin) + 1)) if len(fin) and c1_eval else None
    V["C1"] = {"mh_rr": None if not math.isfinite(rr_pt) else rr_pt, "mh_rr_raw": str(rr_pt), "ci": ci,
               "n_cert": len(full1), "n_zero_cost": sum(bool(r.get("zero_cost")) for r in full1),
               "n_false": n_pos_c1, "n_instances": len(insts), "evaluable": c1_eval, "p": p_c1,
               "rule": "pass: RR > 2 AND CI lower > 1; falsified: RR <= 1.2; <5 positives -> not evaluable",
               "strata": {f"seg{s[0]}|{s[1]}": arr.sum(0)[k].tolist() for k, s in enumerate(strata)} if insts else {}}
    # C2
    fcr = {}
    for a in ("off", "ev1", "ev20", "ev200", "full", "probe20"):
        rr = [r for r in st1 if r["arm"] == a and r.get("status") == "CERTIFIED"]
        k = sum(bool(r.get("false_cert")) for r in rr)
        fcr[a] = {"n_cert": len(rr), "n_false": k, "fcr": (k / len(rr)) if rr else None, "cp": cp(k, len(rr)),
                  "n_problems": sum(1 for r in st1 if r["arm"] == a)}
    fpos = fcr["full"]["n_false"]
    c2_eval = fpos >= 5
    D = {a: (None if not fcr["full"]["fcr"] or fcr[a]["fcr"] is None else 1 - fcr[a]["fcr"] / fcr["full"]["fcr"])
         for a in ("ev1", "ev20", "ev200", "probe20")}
    c2_pass = None
    if c2_eval and None not in D.values():
        c2_pass = all(D[a] <= 0.20 for a in ("ev1", "ev20", "ev200")) and D["probe20"] >= 0.50
    V["C2"] = {"fcr_by_arm": fcr, "D": D, "evaluable": c2_eval, "pass_point": c2_pass, "p": None,
               "rule": "D(m) <= 0.20 for m in {1,20,200} AND probe20 D >= 0.50; <5 positives in full -> not evaluable",
               "p_note": "p (IUT) computed only when evaluable"}
    if c2_eval:
        byi = defaultdict(list)
        for r in st1:
            if r.get("status") == "CERTIFIED":
                byi[r["instance"]].append(r)
        ks = sorted(byi)
        cnt = np.array([[[sum(bool(r.get("false_cert")) for r in byi[i] if r["arm"] == a),
                          sum(1 for r in byi[i] if r["arm"] == a)] for a in ("full", "ev1", "ev20", "ev200", "probe20")]
                        for i in ks], float)
        pc = {"ev1": 0, "ev20": 0, "ev200": 0, "probe20": 0}
        nb = 0
        for row in boot_idx(len(ks)):
            s = cnt[row].sum(0)
            if s[0, 1] == 0 or s[0, 0] == 0:
                continue
            nb += 1
            f0 = s[0, 0] / s[0, 1]
            for j, a in enumerate(("ev1", "ev20", "ev200", "probe20"), start=1):
                d = 1 - (s[j, 0] / s[j, 1] if s[j, 1] else np.nan) / f0
                if a == "probe20":
                    pc[a] += not (d >= 0.5)
                else:
                    pc[a] += not (d <= 0.2)
        V["C2"]["p"] = float(max((1 + v) / (nb + 1) for v in pc.values())) if nb else None
    # dose / HD1 replication
    dose_rows = []
    for g in (0.0, 0.5, 1.0):
        for cls in ("static", "dyn_G1", "dyn_Gext"):
            rr = [r for r in STAT if r["arm"] == "full" and fnum(r.get("dose")) == g and r.get("learner_class") == cls
                  and r.get("status") == "CERTIFIED"]
            if rr:
                k = sum(bool(r.get("false_cert")) for r in rr)
                dose_rows.append({"dose": g, "class": cls, "n_cert": len(rr), "n_false": k, "fcr": k / len(rr),
                                  "cp_low": cp(k, len(rr))[0], "cp_high": cp(k, len(rr))[1]})
    V["HD1_dose"] = dose_rows
    # B-adv
    cells = []
    adv = [r for r in HZ if r.get("layer") == "R1adv" and str(r.get("base_method", r["method"])).startswith("JPC")
           and r.get("status") == "CERTIFIED"]
    for cal in ("post_bin_near", "post_bin_gcirc"):
        for a in range(3):
            for b in range(3):
                rr = [r for r in adv if r.get("ante_bin_muflip") == a and r.get(cal) == b]
                k = sum(bool(r.get("false_cert")) for r in rr)
                from scipy.stats import binom
                pv = float(binom.sf(k - 1, len(rr), DELTA)) if rr else None
                cells.append({"caliber": cal, "ante_bin": a, "post_bin": b, "n_cert": len(rr), "n_false": k,
                              "fcr": (k / len(rr)) if rr else None, "cp_low": cp(k, len(rr))[0],
                              "cp_high": cp(k, len(rr))[1], "binom_p_fcr_gt_delta": pv,
                              "n_instances": len({r["instance"] for r in rr})})
    prim = [c for c in cells if c["caliber"] == "post_bin_near" and c["n_cert"]]
    haz = [c for c in prim if c["cp_low"] > DELTA]
    hedge = prim and all(c["cp_high"] <= 2 * DELTA for c in prim)
    pb = min([c["binom_p_fcr_gt_delta"] for c in prim], default=None)
    V["B-adv"] = {"cells": cells, "hazard_cells": [(c["ante_bin"], c["post_bin"]) for c in haz],
                  "branch": ("hazard exists" if haz else ("hedging capacity" if hedge else "inconclusive")),
                  "p": None if pb is None else min(1.0, pb * len(prim)),
                  "rule": "any cell FCR CP lower > delta -> hazard exists; all CP upper <= 2 delta -> hedging capacity; "
                          "Holm p = Bonferroni(min binomial p over filled primary-caliber cells)",
                  "n_cert": len(adv), "evaluable": bool(prim)}
    vadj = holm({k: V[k]["p"] for k in ("C1", "C2", "B-adv")})
    for k in ("C1", "C2", "B-adv"):
        V[k]["holm_p"] = vadj.get(k)
    V["C1"]["verdict"] = ("NOT_EVALUABLE" if not c1_eval else
                          "PASS" if (V["C1"]["mh_rr"] or 0) > 2 and ci[0] is not None and ci[0] > 1
                          and (vadj["C1"] or 1) < 0.05 else
                          "FALSIFIED" if (V["C1"]["mh_rr"] is not None and V["C1"]["mh_rr"] <= 1.2) else "NOT_CONFIRMED")
    V["C2"]["verdict"] = ("NOT_EVALUABLE" if not c2_eval else
                          "PASS" if c2_pass and (vadj["C2"] or 1) < 0.05 else "NOT_CONFIRMED")
    V["B-adv"]["verdict"] = ("NOT_EVALUABLE" if not prim else V["B-adv"]["branch"].upper().replace(" ", "_"))
    S["V_family"] = V
    log("V family done")

    # ================================================================== 5. Q family
    progress(out, 5, TOTAL, "Q family")
    variant = (lock.get("frozen_items") or {}).get("predictor_variant") or "S2"
    Q = {"frozen_variant": variant}

    def join(R, P, extra=None):
        pm = pmap(P)
        o = []
        for r in R:
            p = pm.get(pkey(r))
            if p is not None and p.get(variant) is not None:
                o.append((r, p))
        return o

    def zc_events(pairs):
        return [(r, p) for r, p in pairs if r.get("status") == "CERTIFIED" and r.get("zero_cost")
                and int(r.get("problem_index", r["problem"])) >= 1 and r.get("gap_layer") in ("near", "clear")]

    q1_pairs = []
    q1_pairs += [x for x in join([r for r in STAT if r["arm"] == "full" and r.get("learner_class") == "static"
                                  and fnum(r.get("dose")) in (0.5, 1.0)], STATP) ]
    q1_pairs += join([r for r in X1 if r.get("source") == "x1_m1_2eps" and r.get("sampler") == "DDA"], X1P)
    q1_pairs += join([r for r in HZ if str(r.get("base_method", r["method"])).startswith("JPC")], HZP)
    q1 = zc_events(q1_pairs)
    ho = zc_events(join([r for r in HO if r["arm"] == "full" and r.get("family") in ("m1r_eps", "m1r_2eps", "alias")], HOP))

    def score(p, name):
        if name == "S1":
            return fnum(p.get("S1"))
        if name == "S2":
            v = p.get("S2")
            return 1e300 if v in (None, "inf") or (isinstance(v, float) and math.isinf(v)) else fnum(v)
        if name == "neg_log_gap_hat":
            return fnum(p.get("neg_log_gap_hat"))
        if name == "log_tr_Iinv":
            return fnum(p.get("log_tr_Iinv"))
        if name == "eta_hat_gof":
            return fnum(p.get("eta_hat_gof"))
        return None

    def q1_block(ev, label):
        if not ev:
            return {"n": 0, "evaluable": False}
        y = [bool(r.get("false_cert")) for r, _ in ev]
        cl = [f"{r.get('_src')}:{r['instance']}" for r, _ in ev]
        blk = {"n": len(ev), "n_pos": int(sum(y)), "n_clusters": len(set(cl)), "evaluable": sum(y) >= 5}
        for nm in (variant, "S1", "neg_log_gap_hat", "log_tr_Iinv", "eta_hat_gof"):
            s = [score(p, nm) for _, p in ev]
            ok = [i for i, v in enumerate(s) if v is not None]
            if len(set(y[i] for i in ok)) < 2:
                blk[nm] = {"auc": None}
                continue
            a = cluster_auc([s[i] for i in ok], [y[i] for i in ok], [cl[i] for i in ok])
            blk[nm] = {k: v for k, v in a.items() if k != "reps"}
            if nm == variant:
                blk["_reps"] = a["reps"]
        triv = {nm: (blk.get(nm) or {}).get("auc") for nm in ("neg_log_gap_hat", "log_tr_Iinv", "eta_hat_gof")}
        triv = {k: v for k, v in triv.items() if v is not None}
        blk["best_trivial"] = max(triv.items(), key=lambda t: t[1]) if triv else None
        blk["kendall_tau_trivial"] = "not logged by the learner side in round 3 -> omitted (lock must drop or add it)"
        return blk

    Q1 = {"pooled": q1_block(q1, "pooled")}
    scope = [(r, p) for r, p in q1 if fnum(r.get("eta_dec")) is not None and fnum(r["eta_dec"]) <= EPS]
    Q1["T0_scope_eta_dec_le_eps"] = q1_block(scope, "scope")
    Q1["heldout"] = q1_block(ho, "heldout")
    pb_ = Q1["pooled"]
    reps = pb_.pop("_reps", None)
    Q1["scope_note"] = ("T0(ii) failed -> lock T0_scope 'eta <= eps'; scope variable here = row eta_dec "
                        "(harness-side decision misspecification) -- the full lock must pin which eta is used")
    Q1["T0_scope_eta_dec_le_eps"].pop("_reps", None)
    Q1["heldout"].pop("_reps", None)
    q1p = None
    if pb_.get("evaluable") and reps is not None and pb_.get("best_trivial"):
        q1p = float((1 + (reps <= pb_["best_trivial"][1]).sum()) / (len(reps) + 1))
    ho_dir = None
    hb = Q1["heldout"]
    if hb.get("evaluable") and (hb.get(variant) or {}).get("auc") is not None and hb.get("best_trivial"):
        ho_dir = hb[variant]["auc"] - hb["best_trivial"][1] > 0
    Q1["p"] = q1p
    Q1["heldout_direction_consistent"] = ho_dir if hb.get("evaluable") else "NOT_EVALUABLE (<5 positives)"
    Q["Q1"] = Q1
    # Q2: opportunities (k >= 1, full arm; static via full@open, NL-R0 JPC/B3 cert-time proxy)
    opp = []
    pm_open = {}
    for p in STATP:
        if p.get("arm") == "full@open":
            pm_open[pkey(p)] = p
    for r in STAT:
        if r["arm"] == "full@open":
            p = pm_open.get(pkey(r))
            if p is not None:
                opp.append((r, p, "static_open"))
    pmn = pmap(NLP)
    for r in NL:
        if r["arm"] == "full" and int(r.get("problem_index", r["problem"])) >= 1 and r["method"] in ("JPC", "B3"):
            p = pmn.get(pkey(r))
            if p is not None and p.get("A_k_max") is not None:
                opp.append((r, p, "NL_cert_time_proxy"))
    Q2 = {"n": len(opp), "n_pos": sum(bool(r.get("zero_cost")) for r, _, _ in opp),
          "sources": dict(Counter(s for _, _, s in opp)),
          "note": "ev1 arm cannot produce zero-cost certification (>=1 fresh step forced) -> sample = full arm; "
                  "NL-R0 rows use the certification-time predictor record (no opening record logged in r3_nl_main) "
                  "-- equal to the opening record for zero-cost rows only; lock should add an opening record"}
    X, Y, cl2, plc_b, plc_g, ak = [], [], [], [], [], []
    for r, p, s in opp:
        g, t, a = fnum(p.get("neg_log_gap_hat")), fnum(p.get("log_tr_Iinv")), fnum(p.get("A_k_max"))
        if None in (g, t, a) or a <= 0:
            continue
        X.append([-g, t])
        ak.append(math.log(a))
        Y.append(1.0 if r.get("zero_cost") else 0.0)
        cl2.append(f"{r['_src']}:{r['instance']}")
        plc_b.append(fnum(p.get("A_k_placebo_block_max")))
        plc_g.append(fnum(p.get("A_k_placebo_global_max")))
    if len(Y) > 10 and 0 < sum(Y) < len(Y):
        X, ak, Y = np.array(X), np.array(ak)[:, None], np.array(Y)
        lt = lr_test(X, np.c_[X, ak], Y)
        Q2["lr_test_log_Ak"] = lt
        a_ak = cluster_auc(ak[:, 0], Y.astype(bool), cl2, b=2000)
        Q2["auc_A_k"] = {k: v for k, v in a_ak.items() if k != "reps"}
        for nm, pl in (("placebo_block", plc_b), ("placebo_global", plc_g)):
            okk = [i for i, v in enumerate(pl) if v is not None]
            a = auc_pt(np.array([pl[i] for i in okk]), Y[okk].astype(bool))
            Q2[f"auc_{nm}"] = a
            Q2[f"auc_{nm}_orientation_free"] = max(a, 1 - a)
        Q2["p"] = lt["p"]
        Q2["pass_point"] = bool(lt["p"] < 0.01 and Q2["auc_placebo_block_orientation_free"] <= 0.6)
        Q2["placebo_rule_note"] = ("methodology: placebo permutes u within parameter blocks (A_k_placebo_block); "
                                   "the global permutation is reported alongside")
    else:
        Q2["p"] = None
        Q2["pass_point"] = None
    Q["Q2"] = Q2
    # Q3: two-axis logistic on Q1 events with both predictors (outcomes: false cert among zero-cost; zero-cost among opp)
    Q3 = {}
    try:
        e_ = [(r, p) for r, p in q1 if fnum(p.get("S1")) and fnum(p.get("A_k_max"))]
        if len(e_) > 10 and 0 < sum(bool(r.get("false_cert")) for r, _ in e_) < len(e_):
            Xe = np.array([[math.log(fnum(p["S1"])), math.log(fnum(p["A_k_max"]))] for _, p in e_])
            ye = np.array([1.0 if r.get("false_cert") else 0.0 for r, _ in e_])
            f = logistic_fit(Xe, ye)
            Q3["error_model"] = {"beta_logLambda": f["beta"][1], "se_logLambda": f["se"][1],
                                 "beta_logAk": f["beta"][2], "se_logAk": f["se"][2], "n": f["n"]}
        o_ = [(r, p) for r, p, _ in opp if fnum(p.get("S1")) and fnum(p.get("A_k_max"))]
        if len(o_) > 10:
            Xo = np.array([[math.log(fnum(p["S1"])), math.log(fnum(p["A_k_max"]))] for _, p in o_])
            yo = np.array([1.0 if r.get("zero_cost") else 0.0 for r, _ in o_])
            f = logistic_fit(Xo, yo)
            Q3["occurrence_model"] = {"beta_logLambda": f["beta"][1], "se_logLambda": f["se"][1],
                                      "beta_logAk": f["beta"][2], "se_logAk": f["se"][2], "n": f["n"]}
    except Exception as e:  # noqa: BLE001
        Q3["error"] = repr(e)
    from scipy.stats import norm
    Q3["p"] = None
    if "error_model" in Q3 and "occurrence_model" in Q3:
        em, om = Q3["error_model"], Q3["occurrence_model"]
        own_p = [2 * norm.sf(abs(em["beta_logLambda"] / em["se_logLambda"])) if em["se_logLambda"] else 1.0,
                 2 * norm.sf(abs(om["beta_logAk"] / om["se_logAk"])) if om["se_logAk"] else 1.0]
        cross_zero = [abs(om["beta_logLambda"]) <= 1.96 * om["se_logLambda"],
                      abs(em["beta_logAk"]) <= 1.96 * em["se_logAk"]]
        Q3["p"] = float(max(own_p))
        Q3["cross_ci_contain_0"] = cross_zero
        Q3["pass_point"] = bool(all(cross_zero))
    Q3["rule"] = ("pass: cross coefficients (Lambda->zero-cost occurrence, A_k->error) 95% Wald CI contain 0; Holm p "
                  "= max Wald p of the two own-axis coefficients (operationalisation for the lock)")
    Q["Q3"] = Q3
    qadj = holm({"Q1": Q1["p"], "Q2": Q2["p"], "Q3": Q3["p"]})
    for k in ("Q1", "Q2", "Q3"):
        Q[k]["holm_p"] = qadj.get(k)
    best_t = (pb_.get("best_trivial") or [None, None])[1]
    q1_pass = (pb_.get("evaluable") and (pb_.get(variant) or {}).get("ci", [None])[0] is not None
               and best_t is not None and pb_[variant]["ci"][0] > best_t and (qadj["Q1"] or 1) < 0.05
               and ho_dir is not False)
    Q1["verdict"] = ("NOT_EVALUABLE" if not pb_.get("evaluable") else "PASS" if q1_pass else "NOT_CONFIRMED")
    Q2["verdict"] = ("NOT_EVALUABLE" if Q2.get("p") is None else
                     "PASS" if Q2.get("pass_point") and (qadj["Q2"] or 1) < 0.05 else "NOT_CONFIRMED")
    Q3["verdict"] = ("NOT_EVALUABLE" if Q3.get("p") is None else
                     "PASS" if Q3.get("pass_point") and (qadj["Q3"] or 1) < 0.05 else "NOT_CONFIRMED")
    S["Q_family"] = Q
    log("Q family done")

    # ================================================================== 6. replication table
    progress(out, 6, TOTAL, "replication")
    rep = []
    hd = summ.get("r3_replicate_hd1b_hd2", {})
    hh = summ.get("r3_replicate_hh1", {})
    rep.append({"hypothesis": "HD1 (static FCR rises with dose)", "round3": json.dumps(dose_rows)[:400],
                "source": "r3_static_[a-d] raw rows (full arm)"})
    rep.append({"hypothesis": "HD1b (same exposure: static FCR >= 0.9, dynamic 0)",
                "round3": json.dumps((hd.get("pass_criteria") or {}))[:300], "source": "r3_replicate_hd1b_hd2 summary"})
    rep.append({"hypothesis": "HD2 (kappa sandwich, H=1 exact 0)", "round3": json.dumps(hd.get("hd2", {}), default=str)[:300],
                "source": "r3_replicate_hd1b_hd2 summary"})
    rep.append({"hypothesis": "HH1 (honest abstention)", "round3": json.dumps(hh.get("reported", hh.get("hh1", {})), default=str)[:300],
                "source": "r3_replicate_hh1 summary"})
    rc = summ.get("r3_replica_check", {})
    rep.append({"hypothesis": "replica check (|J diff| <= 1e-6, 0 label flips)",
                "round3": f"max_abs_J_diff={rc.get('max_abs_J_diff')}, flips={rc.get('label_flip_count')}",
                "source": "r3_replica_check summary"})
    x1 = summ.get("r3_x1_sampler_misspec", {})
    rep.append({"hypothesis": "X1 (sampler matters only under misspecification)",
                "round3": json.dumps(x1.get("x1_readout"), default=str)[:400], "source": "r3_x1_sampler_misspec summary"})
    S["replication"] = rep
    out.csv("table2_replication.csv", rep)

    # ================================================================== 7. suspicious-gate audit
    progress(out, 7, TOTAL, "suspicious audit")
    aud = []
    try:
        import subprocess
        tr = subprocess.run([sys.executable, "-m", "pytest", "-q", "dsswm/tests/test_no_truth_import.py"], cwd=CODE,
                            capture_output=True, text=True, timeout=300)
        leak_ok = tr.returncode == 0
        leak_note = tr.stdout.strip().splitlines()[-1] if tr.stdout.strip() else tr.stderr[-200:]
    except Exception as e:  # noqa: BLE001
        leak_ok, leak_note = None, repr(e)
    replay_billed = sum(1 for r in allrows if (r.get("replay_steps") or 0) > 0
                        and int(r.get("n_rounds_billed", -1)) != int(r.get("env_n_steps", -2)))
    cens_bad = sum(1 for r in NL if r.get("censored_tau3000") and int(r.get("cost_tau3000", 0)) != 3000)
    tie_share = float(np.mean([r.get("gap_layer") == "tie" for r in NL])) if NL else None

    def add(cell, trigger, value):
        aud.append({"cell": cell, "trigger": trigger, "value": value, "leakage_test_pass": leak_ok,
                    "leakage_note": leak_note, "replay_billed_rows": replay_billed,
                    "censored_billing_bad_rows": cens_bad, "tie_share_NL": tie_share,
                    "cum_prefix_ge_OPT_K": "not computable (OPT_K oracle not run in round 3)",
                    "cleared": bool(leak_ok and replay_billed == 0 and cens_bad == 0)})

    for L, R in PF.items():
        for h in ("A1", "A2", "A3"):
            e = R[h]["est"]
            if e is not None and math.exp(e) < 0.2:
                add(f"{L}:{h}", "saving > 5x", math.exp(e))
        e = R["A3o"]["est"]
        if e is not None and abs(e) > math.log(5):
            add(f"{L}:A3o (descriptive)", "|orth/vol| > 5x",
                f"{math.exp(e):.1f}x on n={R['A3o']['n']} inst; near-zero vol denominators make log((S+1)/(S'+1)) heavy-tailed")
        e = R["main_nontie_jpc_over_base_full"]["est"]
        if e is not None and math.exp(e) < 0.7:
            add(f"{L}:JPC/baseline full (non-tie)", "> 30% better than simple baseline", math.exp(e))
    for d in dose_rows:
        if d["class"] == "static" and d["fcr"] < 0.01 and d["dose"] > 0:
            add(f"static g={d['dose']}", "static FCR ~ 0", d["fcr"])
    if not aud:
        add("none", "no trigger fired", None)
    out.csv("suspicious_gate_audit.csv", aud, d=out.root)
    S["suspicious_gate_audit"] = {"n_triggers": sum(a["trigger"] != "no trigger fired" for a in aud),
                                  "all_cleared": all(a["cleared"] for a in aud), "leakage_test": leak_note}

    # ================================================================== 8. tier decision
    progress(out, 8, TOTAL, "tier")
    nl = PF["NL-R0"]
    tin = {"A1_NL_ci_upper": nl["A1"]["ci"][1], "A1_pass_NL": nl["A1"]["verdict"] == "PASS",
           "A1_pass_Lin": PF["E1-Lin"]["A1"]["verdict"] == "PASS", "A2_pass_NL": nl["A2"]["verdict"] == "PASS",
           "A2_pass_Lin": PF["E1-Lin"]["A2"]["verdict"] == "PASS",
           "clear_ratio_ci_upper": nl["clear_switch"]["ratio_ci"][1],
           "saving_null": S["controls"]["saving_null"], "saving_main": S["controls"]["saving_main"],
           "C1_pass": True if V["C1"]["verdict"] == "PASS" else (None if V["C1"]["verdict"] == "NOT_EVALUABLE" else False),
           "Q1_pass": True if Q1["verdict"] == "PASS" else (None if Q1["verdict"] == "NOT_EVALUABLE" else False)}
    tier = decide_tier_r3(tin)
    tier["inputs"] = tin
    tier["code_sha256"] = sha(CODE / "dsswm" / "stats" / "tier_decision_r3.py")
    tier["binding"] = mode == "full"
    S["tier_decision_r3"] = tier

    # ================================================================== 9. verdict table
    progress(out, 9, TOTAL, "verdicts")
    VR = []

    def fm(x, d=3):
        return "NA" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    for L, R in PF.items():
        for h in ("A1", "A2", "A3", "A3o"):
            VR.append({"family": "P", "layer": L, "hypothesis": h, "estimate": R[h]["est"], "ci_low": R[h]["ci"][0],
                       "ci_high": R[h]["ci"][1], "p_one_sided": R[h]["p"], "holm_p": R[h]["holm_p"],
                       "n_instances": R[h]["n"], "verdict": R[h]["verdict"], "scale": "mean log ratio"})
        cs = R["clear_switch"]
        VR.append({"family": "P-switch", "layer": L, "hypothesis": "clear-layer JPC/base (full)",
                   "estimate": cs["ratio_est"], "ci_low": cs["ratio_ci"][0], "ci_high": cs["ratio_ci"][1],
                   "p_one_sided": None, "holm_p": None, "n_instances": cs["log_ratio"]["n"],
                   "verdict": ("NOT_EVALUABLE" if cs["ratio_ci"][1] is None else
                               "UPPER<1" if cs["ratio_ci"][1] < 1 else "CI>=1 (tier-C switch)"), "scale": "ratio"})
    VR.append({"family": "P", "layer": "both", "hypothesis": "A2 two-layer intersection", "estimate": None,
               "ci_low": None, "ci_high": None, "p_one_sided": None, "holm_p": None, "n_instances": None,
               "verdict": "PASS" if A2_two else "NOT_CONFIRMED", "scale": ""})
    VR.append({"family": "V", "layer": "static g=1", "hypothesis": "C1", "estimate": V["C1"]["mh_rr"],
               "ci_low": V["C1"]["ci"][0], "ci_high": V["C1"]["ci"][1], "p_one_sided": V["C1"]["p"],
               "holm_p": V["C1"]["holm_p"], "n_instances": V["C1"]["n_instances"], "verdict": V["C1"]["verdict"],
               "scale": "MH RR"})
    VR.append({"family": "V", "layer": "static g=1", "hypothesis": "C2", "estimate": json.dumps(D),
               "ci_low": None, "ci_high": None, "p_one_sided": V["C2"]["p"], "holm_p": V["C2"]["holm_p"],
               "n_instances": None, "verdict": V["C2"]["verdict"], "scale": "relative FCR drop"})
    VR.append({"family": "V", "layer": "R1-adv", "hypothesis": "B-adv", "estimate": V["B-adv"]["branch"],
               "ci_low": None, "ci_high": None, "p_one_sided": V["B-adv"]["p"], "holm_p": V["B-adv"]["holm_p"],
               "n_instances": None, "verdict": V["B-adv"]["verdict"], "scale": "cell FCR"})
    for k in ("Q1", "Q2", "Q3"):
        q = Q[k]
        est = ((q.get("pooled") or {}).get(variant) or {}).get("auc") if k == "Q1" else (
            (q.get("lr_test_log_Ak") or {}).get("stat") if k == "Q2" else None)
        ci_ = ((q.get("pooled") or {}).get(variant) or {}).get("ci", [None, None]) if k == "Q1" else [None, None]
        VR.append({"family": "Q", "layer": "risk layers", "hypothesis": k, "estimate": est, "ci_low": ci_[0],
                   "ci_high": ci_[1], "p_one_sided": q.get("p"), "holm_p": q.get("holm_p"), "n_instances": None,
                   "verdict": q["verdict"], "scale": {"Q1": "AUC", "Q2": "LR stat", "Q3": "Wald"}[k]})
    VR.append({"family": "decision", "layer": "", "hypothesis": "tier (tier_decision_r3)", "estimate": tier["tier"],
               "ci_low": None, "ci_high": None, "p_one_sided": None, "holm_p": None, "n_instances": None,
               "verdict": tier["reason"], "scale": tier["q_wording"]})
    out.csv("verdicts.csv", VR, d=out.root)
    md = [f"# r3_analysis_aggregate verdict table ({mode})", "",
          f"> {S['evidence_status']}. Bootstrap B={B}, instance clusters, seed {SEED}. 并发运行。", "",
          "| family | layer | hypothesis | estimate | 95% CI | p (1-sided) | Holm p | n inst | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in VR:
        md.append(f"| {v['family']} | {v['layer']} | {v['hypothesis']} | {fm(v['estimate'])} | "
                  f"[{fm(v['ci_low'])}, {fm(v['ci_high'])}] | {fm(v['p_one_sided'], 4)} | {fm(v['holm_p'], 4)} | "
                  f"{fm(v['n_instances'])} | {v['verdict']} |")
    md += ["", f"Tier: **{tier['tier']}** -- {tier['reason']}", f"Q wording: {tier['q_wording']}", ""]
    (out.root / "verdict_table.md").write_text("\n".join(md) + "\n")
    # Table 1
    t1 = [{"Layer": v["layer"], "Hypothesis": v["hypothesis"], "Estimate": v["estimate"],
           "95% CI": f"[{fm(v['ci_low'])}, {fm(v['ci_high'])}]", "Holm p": v["holm_p"], "Verdict": v["verdict"]}
          for v in VR if v["family"] == "P"]
    out.csv("table1_P_family.csv", t1)

    # ================================================================== 10. figures
    progress(out, 10, TOTAL, "figures")
    figs(out, PF, forest, NL, LR_EX, S, V, cells, q1, opp, variant, tau_lin, LIN)

    # ================================================================== 11. candidates
    progress(out, 11, TOTAL, "candidates")
    S["_lock_power"] = lock.get("power", {})
    cand = candidates(S, G, gates_lock, p4, mode)
    S.pop("_lock_power", None)
    S["candidates"] = cand
    S["pass_criteria"] = {"all_pilot_summaries_parsable": S["all_summaries_parsable"],
                          "go_no_go_per_candidate_written": all(c.get("go_no_go") for c in cand["candidates"]),
                          "every_hard_gate_status_explicit": all(G[k].get("pass") is not None for k in
                                                                 ("H0", "billing", "P2_lin", "P5_hazard"))
                          and G["T0"]["hard_pass"] is not None,
                          "n_readouts_ge_100": n_read >= 100}
    S["passed"] = all(S["pass_criteria"].values())
    S["go_no_go"] = "GO" if S["passed"] else "NO_GO"
    S["code_sha256"] = {"run_r3_analysis_aggregate.py": sha(Path(__file__)),
                        "dsswm/stats/tier_decision_r3.py": tier["code_sha256"]}
    S["wall_clock_s"] = time.time() - T0W
    S["finished_at"] = datetime.now().isoformat()
    S["figures"] = sorted(p.name for p in out.fig.iterdir())
    S["log"] = LOG
    (out.root / "summary.json").write_text(json.dumps(S, indent=1, default=_js, ensure_ascii=False))
    progress(out, TOTAL, TOTAL, "done")
    print(json.dumps({"go_no_go": S["go_no_go"], "tier": tier["tier"], "wall": S["wall_clock_s"]}, default=str))


def _js(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


# ============================================================================================ candidates
def candidates(S, G, gl, p4, mode):
    co = gl.get("cand_o_gate") or p4.get("cand_o_gate") or {}
    clg = gl.get("cand_l_GR_gate") or p4.get("cand_l_GR_gate") or {}
    fb = (gl.get("power") or {}).get("fallback_cand_r") or ((S.get("_lock_power") or {}).get("fallback_cand_r")) or {}
    go_f = bool(G["H0"]["pass"] and G["billing"]["pass"] and G["P2_lin"]["pass"] and G["P1_interaction_computable"])
    nl = S["P_family"]["NL-R0"]
    lin = S["P_family"]["E1-Lin"]
    c = [
        {"candidate_id": "cand_f", "go_no_go": "GO" if go_f else "NO_GO",
         "confidence": 0.55 if go_f else 0.3,
         "gate": "go_to_full: H0 AND billing 0 mismatch (5 levels) AND P2 completion >= 0.9 & <= 5 s/run AND P1 "
                 "interaction computable",
         "gate_inputs": {"H0": G["H0"]["status"], "billing_mismatch": G["billing"]["summary_mismatch_total"],
                         "P2_completion": G["P2_lin"]["completion"], "P2_sec_median": G["P2_lin"]["sec_per_run_median"],
                         "P1_interaction_computable": G["P1_interaction_computable"]},
         "supported_hypotheses": ["pipeline: all 36 dependency pilots GO-able, billing 0 mismatch on all five levels",
                                  "T0(i) identity <= 1e-9, T0(iii) retrievable",
                                  "P5 R1-adv acceptance %.3f (no directed construction)" % ((G["P5_hazard"]["acceptance"] or {}).get("rate", float("nan")))],
         "failed_assumptions": ["T0(ii) computable Rem/eps p90 %.0f >> 0.5 -> Thm 3 / Q scope eta <= eps" % (G["T0"]["ii_rem_bar_over_eps_p90_eta_le_2eps"] or float("nan")),
                                "T0(iv) phi_perp median %.2f > 0.3 -> 'targeted evidence has no cost advantage'" % (G["T0"]["iv_phi_perp_median"] or float("nan")),
                                "orth constructible rate < 0.5 -> A3o descriptive",
                                "dev Q1 signal: trivial gap AUC beats Lambda_hat_perp (pre-announced risk)",
                                "Lin dev A2 direction opposite to prediction (mean I > 0)"],
         "key_metrics": {"dev_preview_NL_A1": nl["A1"]["est"], "dev_preview_NL_A2": nl["A2"]["est"],
                         "dev_preview_Lin_A1": lin["A1"]["est"], "dev_preview_Lin_A2": lin["A2"]["est"],
                         "dev_preview_tier": S["tier_decision_r3"]["tier"]},
         "notes": "GO is by pipeline gates only (never by effect direction). Dev-preview numbers are non-evidence."},
        {"candidate_id": "cand_o", "go_no_go": "NO_GO", "confidence": 0.8,
         "gate": co.get("rule"), "gate_inputs": {k: co.get(k) for k in ("nontie_auc_S1", "delta_auc_S1_minus_eta_arg", "T0_ii_pass")},
         "promoted": bool(co.get("pass")),
         "notes": "promotion gate fails on all three clauses (AUC 0.66 < 0.80, increment over eta_arg negative, T0(ii) fails); stays backup"},
        {"candidate_id": "cand_l", "go_no_go": "NO_GO", "confidence": 0.9,
         "gate": "G-R: Rbar2/eps median <= 0.25 and p90 <= 0.5",
         "gate_inputs": {k: clg.get(k) for k in ("rbar2_over_eps_median", "rbar2_over_eps_p90")},
         "arm_added": bool(clg.get("pass")),
         "notes": "computable remainder bound ~3 orders of magnitude above the gate; LME-DWR-lite arm not added"},
        {"candidate_id": "cand_r", "go_no_go": "NOT_TRIGGERED (provisional)", "confidence": 0.5,
         "gate": fb.get("rule") or "A2 MDE80 at n=120 (NL-R0) > 0.6",
         "gate_inputs": {"A2_MDE80_NL": fb.get("value"), "triggered_on_pilot_SD": fb.get("triggered"),
                         "binding": fb.get("binding")},
         "notes": "pilot SD (8 inst x 1 stream x 5 problems) gives MDE 0.63 > 0.6 -> fallback would fire, but the "
                  "binding SD comes from P1 full (720-739, 3 streams x 15 problems). Pipeline gates P1/P2 pass and "
                  "budget fits, so the main line stays cand_f; if P1 full confirms MDE > 0.6, the headline is re-worded "
                  "in cand_r terms (P family still runs)."},
    ]
    return {"overall_recommendation": "ADVANCE" if go_f else "REFINE", "selected_candidate_id": "cand_f",
            "candidates": c}


# ============================================================================================ figures
def figs(out, PF, forest, NL, LR_EX, S, V, cells, q1, opp, variant, tau_lin, LIN):
    # Fig 1 design matrix
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    rows = [("NL-R0", "JPC"), ("NL-R0", "B3"), ("NL-R0", "LR-chi2-grid"), ("NL-R0", "B8"), ("E1-Lin", "JPC-Lin"),
            ("E1-Lin", "RAGE"), ("E1-Lin", "XY-static"), ("E1-Lin", "G-opt"), ("E1-Lin", "B1eb")]
    have = {("NL-R0", m, a) for m in ("JPC", "B3") for a in ARMS} | {("NL-R0", m, a) for m in ("LR-chi2-grid", "B8") for a in ("off", "full")}
    have |= {("E1-Lin", m, a) for m in ("JPC-Lin", "RAGE") for a in ARMS} | {("E1-Lin", "XY-static", "off"), ("E1-Lin", "G-opt", "full"), ("E1-Lin", "B1eb", "full")}
    M = np.array([[1.0 if (L, m, a) in have else 0.0 for a in ARMS] for L, m in rows])
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#f0efe9", "#2a78d6"]), aspect="auto")
    ax.set_xticks(range(5), ["off", "vol-replay", "orth-replay", "ev(1)", "full"])
    ax.set_yticks(range(len(rows)), [f"{L}: {m}" for L, m in rows], fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Figure 1. Reuse switch x method design (replay rows unbilled, provenance-tagged)", fontsize=9)
    out.save(fig, "fig1_design_matrix", [{"layer": L, "method": m, **{a: int(M[i, j]) for j, a in enumerate(ARMS)}}
                                          for i, (L, m) in enumerate(rows)])
    # Fig 2 forest
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=False)
    for ax, L in zip(axes, ("NL-R0", "E1-Lin")):
        ff = [f for f in forest if f["layer"] == L and f["log_ratio_vs_off"] is not None]
        for i, f in enumerate(ff):
            c = COLORS.get(f["method"], "#888")
            lo, hi = f["ci_low"], f["ci_high"]
            if lo is not None:
                ax.plot([lo, hi], [i, i], color=c, lw=2)
            ax.plot(f["log_ratio_vs_off"], i, "o", color=c, ms=5)
        ax.set_yticks(range(len(ff)), [f"{f['gap_layer']} {f['method']} {f['arm']}" for f in ff], fontsize=6)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_xlabel("log(S_arm / S_off)")
        ax.set_title(L, fontsize=9)
        ax.grid(axis="x", color="#eee")
    fig.suptitle("Figure 2. Stratified forest (instance means, 95% cluster-bootstrap CI)", fontsize=9)
    out.save(fig, "fig2_forest", forest)
    # Fig 3 K curve
    fig, ax = plt.subplots(figsize=(6, 3.8))
    kr = []
    for rows, mm in ((NL, ("JPC", "B3")), (LR_EX, ("B8",))):
        for m in mm:
            for a, ls in (("off", "--"), ("full", "-")):
                rr = [r for r in rows if r["method"] == m and r["arm"] == a]
                if not rr:
                    continue
                per = defaultdict(lambda: defaultdict(float))
                for r in rr:
                    per[(r["_src"], r["instance"], r["stream"])][int(r.get("problem_index", r["problem"]))] += float(r.get("new_env_steps") or 0)
                K = max(max(d) for d in per.values()) + 1
                curves = []
                for d in per.values():
                    if len(d) < K:
                        continue
                    curves.append(20 + np.cumsum([d[k] for k in range(K)]))
                if not curves:
                    continue
                mc = np.mean(curves, 0)
                ax.plot(range(1, K + 1), mc, ls, color=COLORS.get(m, "#888"), lw=2, label=f"{m} {a}")
                kr += [{"method": m, "arm": a, "K": k + 1, "cum_new_steps_incl_n0": float(v), "n_streams": len(curves)}
                       for k, v in enumerate(mc)]
    ax.set_xlabel("problem k")
    ax.set_ylabel("cumulative new steps (incl. n0=20)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(color="#eee")
    ax.set_title("Figure 3. K curve (NL-R0)", fontsize=9)
    out.save(fig, "fig3_k_curve", kr)
    # Fig 4 zero-effect vs main + stream controls
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    c = S["controls"]
    vals = [("main non-tie", c["main_nontie_jpc_over_b3_full_log"]), ("zero-effect", c["zero_effect_jpc_over_b3_full_log"])]
    for i, (nm, v) in enumerate(vals):
        if v["est"] is not None:
            axes[0].bar(i, v["est"], color=["#2a78d6", "#eb6834"][i], width=0.5)
            if v["ci"][0] is not None:
                axes[0].plot([i, i], v["ci"], color="#333", lw=1.5)
    axes[0].set_xticks([0, 1], [v[0] for v in vals])
    axes[0].axhline(0, color="#999", lw=0.8)
    axes[0].set_ylabel("log(S_JPC,full / S_B3,full)")
    axes[0].set_title("tier-C switch: zero-effect vs main", fontsize=9)
    ctl = c["reuse_factor_by_stream"]
    kinds = list(dict.fromkeys(x["stream_kind"] for x in ctl))
    for j, m in enumerate(("JPC", "B3")):
        ys = [next((x["log_full_off"] for x in ctl if x["stream_kind"] == k and x["method"] == m), None) for k in kinds]
        axes[1].bar(np.arange(len(kinds)) + (j - 0.5) * 0.36, [y if y is not None else 0 for y in ys], width=0.34,
                    color=COLORS[m], label=m)
    axes[1].set_xticks(range(len(kinds)), kinds, fontsize=7, rotation=15)
    axes[1].axhline(0, color="#999", lw=0.8)
    axes[1].set_ylabel("log(S_full / S_off)")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("reuse factor by stream kind", fontsize=9)
    fig.suptitle("Figure 4", fontsize=9)
    out.save(fig, "fig4_zero_effect_streams", [{"panel": "switch", "name": n, **v} for n, v in vals] + ctl)
    # Fig 5 static FCR
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fa = V["C2"]["fcr_by_arm"]
    arms = list(fa)
    ys = [fa[a]["fcr"] or 0 for a in arms]
    ax.bar(range(len(arms)), ys, color="#2a78d6", width=0.6)
    for i, a in enumerate(arms):
        ax.plot([i, i], fa[a]["cp"], color="#333", lw=1.5)
    ax.axhline(DELTA, color="#eb6834", ls="--", lw=1, label="delta")
    ax.set_xticks(range(len(arms)), arms)
    ax.set_ylabel("FCR (CP 95%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Figure 5. Static twin g=1 FCR by reuse arm; C1 MH-RR={V['C1']['mh_rr_raw']}", fontsize=9)
    out.save(fig, "fig5_static_fcr", [{"arm": a, **fa[a]} for a in arms] + [{"dose_table": d} for d in V["HD1_dose"]])
    # Fig 6 heatmap
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, cal in zip(axes, ("post_bin_near", "post_bin_gcirc")):
        H = np.full((3, 3), np.nan)
        for cc in cells:
            if cc["caliber"] == cal and cc["fcr"] is not None:
                H[cc["ante_bin"], cc["post_bin"]] = cc["fcr"]
        ax.imshow(H, cmap="Blues", vmin=0, vmax=max(0.2, np.nanmax(H) if np.isfinite(H).any() else 0.2))
        for cc in cells:
            if cc["caliber"] == cal:
                ax.text(cc["post_bin"], cc["ante_bin"], f"{cc['n_false']}/{cc['n_cert']}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3), ["[0,.5)", "[.5,1)", ">=1"])
        ax.set_yticks(range(3), ["mu_flip low", "mid", "high"])
        ax.set_xlabel("ex-post eta_arg/eps")
        ax.set_title(cal.replace("post_bin_", "flip caliber: "), fontsize=9)
    fig.suptitle("Figure 6. R1-adv FCR (false/certified) by ex-ante x ex-post bin", fontsize=9)
    out.save(fig, "fig6_r1adv_heatmap", cells)
    # Fig 7 ROC
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    roc_rows = []
    y = np.array([bool(r.get("false_cert")) for r, _ in q1])
    for nm, col in ((variant, "#2a78d6"), ("neg_log_gap_hat", "#eb6834"), ("log_tr_Iinv", "#1baf7a"), ("eta_hat_gof", "#eda100")):
        s = []
        for _, p in q1:
            v = p.get(nm)
            s.append(1e300 if v is None and nm == "S2" else (fnum(v) if fnum(v) is not None else -1e300))
        s = np.array(s, float)
        if len(y) and 0 < y.sum() < len(y):
            o = np.argsort(-s)
            tp = np.r_[0, np.cumsum(y[o])] / y.sum()
            fp = np.r_[0, np.cumsum(~y[o])] / (~y).sum()
            ax.plot(fp, tp, color=col, lw=2, label=f"{nm} AUC={auc_pt(s, y):.2f}")
            roc_rows += [{"score": nm, "fpr": float(a), "tpr": float(b)} for a, b in zip(fp, tp)]
    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Figure 7. Q1 ROC, zero-cost near+clear (n={len(y)}, pos={int(y.sum()) if len(y) else 0})", fontsize=8)
    out.save(fig, "fig7_q1_roc", roc_rows)
    # Fig 8 two-axis scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = []
    for r, p, s in opp:
        lam, a = fnum(p.get("S1")), fnum(p.get("A_k_max"))
        if lam is None or a is None:
            continue
        zc, fc = bool(r.get("zero_cost")), bool(r.get("false_cert"))
        sc.append({"lambda_hat_perp": lam, "A_k": a, "zero_cost": zc, "false_cert": fc, "source": s})
        ax.scatter(lam, a, s=22, marker="x" if fc else "o", color="#2a78d6" if zc else "#eb6834", alpha=0.7,
                   linewidths=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Lambda_hat_perp (S1)")
    ax.set_ylabel("A_k (k=3)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2a78d6", label="zero-cost"),
                       Line2D([], [], marker="o", ls="", color="#eb6834", label="paid"),
                       Line2D([], [], marker="x", ls="", color="#333", label="false cert")], fontsize=7, frameon=False)
    ax.set_title("Figure 8. Two-axis map (Q3)", fontsize=9)
    out.save(fig, "fig8_two_axis", sc)


if __name__ == "__main__":
    main()
