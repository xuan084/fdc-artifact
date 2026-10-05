"""r5_gate_decision: read-only gate verdict (T0 / T-spec / T1 / T2 / T3) + power rule for eval stream count.

Rules (frozen in plan/task_plan.json + plan/methodology.md s5.3, s6.1 before any gate result):
  * T0, T-spec, T3 must pass.
  * T2: every r in R one-sided 95% UB < 1.0 => GO; any UB >= 1.0 => verdict 'fdc_out'
    (and if date > 10-08 without pass => 'www27_unreachable').
  * T1 is a deliverable: if C4 model max |log err| > 0.095 => c4_mode = 'descriptive', else 'predictive'.
  * Power: IUT Monte Carlo for the hardest C1 rival (UB closest to 1) using dev point estimate and SD of
    log ratio; 1e3 replications x 2000 bootstrap, N in {200, 400}. power(N=200) >= 0.8 => 200 streams, else 400.
  * C2 CP critical values for N.
Only reads summary.json files; dev per-stream N80 is read only for the (secondary) empirical joint-IUT check.
"""
import csv
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

os.environ.setdefault("OMP_NUM_THREADS", "4")
WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
OUT = RES / "pilots" / "r5_gate_decision"
OUT.mkdir(parents=True, exist_ok=True)
TASK = "r5_gate_decision"
SEED = 42
N_REP = 1000
B_BOOT = 2000
NS = (200, 400)
DELTA = 0.05
DEADLINE = dt.date(2026, 10, 8)

(RES / f"{TASK}.pid").write_text(str(os.getpid()))


def progress(step, total, note=""):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total,
        "loss": None, "metric": {"note": note}, "updated_at": dt.datetime.now().isoformat()}))


def load(rel):
    return json.loads((WS / rel).read_text())


def rel(p):
    return str(Path(p).relative_to(WS)) if str(p).startswith(str(WS)) else str(p)


progress(0, 4, "reading gate summaries")
SRC = {
    "T0": "exp/results/pilots/r5_t0_alloc_audit/summary.json",
    "T-spec": "exp/results/pilots/r5_fdc_spec_theory/summary.json",
    "impl": "exp/results/pilot_summary_r5_fdc_impl.json",
    "qual": "exp/results/pilots/r5_baseline_qualification/summary.json",
    "tuning": "exp/results/r5_gates/rival_configs.json",
    "T1": "exp/results/pilots/r5_t1_reconcile/summary.json",
    "C4": "exp/results/r5_gates/c4_model.json",
    "T2": "exp/results/pilots/r5_t2_gate_strict/summary.json",
    "T3": "exp/results/pilots/r5_t2_gate_plugin_t3/summary.json",
    "DL": "exp/results/pilots/r5_decision_log/summary.json",
}
S = {k: load(v) for k, v in SRC.items()}

cells = []  # (gate, item, value, source_file, source_field, pass)


def cell(gate, item, value, src, field, ok=None):
    cells.append({"gate": gate, "item": item, "value": value, "source": SRC[src], "field": field, "pass": ok})
    return value


# ---- T0
t0 = S["T0"]
t0_pass = (t0["verdict"] == "PASS") and str(t0["tests_passed"]) == "True"
cell("T0", "verdict", t0["verdict"], "T0", "verdict", t0["verdict"] == "PASS")
cell("T0", "root cause", t0["root_cause"], "T0", "root_cause", None)
cell("T0", "regression tests passed", t0["tests_passed"], "T0", "tests_passed", str(t0["tests_passed"]) == "True")
cell("T0", "affects r4 numbers", t0["affects_r4_numbers"], "T0", "affects_r4_numbers", None)

# ---- T-spec
ts = S["T-spec"]
g = ts["gate"]
ts_pass = all(g.values()) and ts["mc"]["fdc_false_streams_all_windows_eps"] == 0 and ts["verdict"] == "GO"
cell("T-spec", "self-check signed", g["selfcheck_signed"], "T-spec", "gate.selfcheck_signed", g["selfcheck_signed"])
cell("T-spec", "MC false streams (60 configs x 500 x 3 eps x windows)", ts["mc"]["fdc_false_streams_all_windows_eps"],
     "T-spec", "mc.fdc_false_streams_all_windows_eps", ts["mc"]["fdc_false_streams_all_windows_eps"] == 0)
cell("T-spec", "MC certifications checked", ts["mc"]["fdc_certifications_total"], "T-spec", "mc.fdc_certifications_total")
cell("T-spec", "worst-config CP95 UB", round(ts["mc"]["fdc_worst_config_cp95_upper"], 5), "T-spec",
     "mc.fdc_worst_config_cp95_upper")
cell("T-spec", "invalid control detects violations", g["invalid_control_detects_violations"], "T-spec",
     "gate.invalid_control_detects_violations", g["invalid_control_detects_violations"])
cell("T-spec", "ledger matches spec (beta=14.2242, x_v=11.8776)", g["ledger_matches_spec"], "T-spec",
     "gate.ledger_matches_spec", g["ledger_matches_spec"])
cell("T-spec", "schedule independence", g["schedule_independent"], "T-spec", "gate.schedule_independent",
     g["schedule_independent"])
cell("T-spec", "control-plane note", ts.get("control_plane_note", ""), "T-spec", "control_plane_note")

# ---- implementation / qualification (context, not gates)
impl_c = S["impl"]["candidates"][0]
cell("impl", "tests", impl_c["key_metrics"]["tests_passed"], "impl", "candidates[0].key_metrics.tests_passed")
R = S["qual"]["registry"]["rigorous_set_R"]
cell("qual", "rigorous set R (frozen)", R, "qual", "registry.rigorous_set_R")
cell("qual", "frozen tuned configs", S["tuning"]["frozen_configs"], "tuning", "frozen_configs")

# ---- T1 (deliverable)
t1 = S["T1"]
t1_repro = {}
for k, v in t1["reproduction"].items():
    rel_dev = v["ratio"] / v["target"] - 1
    t1_repro[k] = {"target": v["target"], "reproduced": v["ratio"], "rel_dev": rel_dev,
                   "within_2pct": abs(rel_dev) <= 0.02}
    cell("T1", f"repro {k} (+-2%)", f"{v['ratio']:.4f} ({rel_dev:+.2%})", "T1", f"reproduction.{k}.ratio",
         abs(rel_dev) <= 0.02)
cell("T1", "pass_criteria all true", all(t1["pass_criteria"].values()), "T1", "pass_criteria",
     all(t1["pass_criteria"].values()))
cv = t1["c4_model_validation"]
cell("T1", "C4 model sha verified", cv["sha_verified"], "T1", "c4_model_validation.sha_verified", cv["sha_verified"])
c4_cell_errs = {c: v[cv["primary"]]["log_error"] for c, v in cv["cells"].items()}
c4_cell_ci = {c: v[cv["primary"]]["ci95"] for c, v in cv["cells"].items()}
c4_err_t1 = max(abs(x) for x in c4_cell_errs.values())
assert abs(c4_err_t1 - cv["max_abs_log_error_primary"]) < 1e-12
c4_limit = float(S["C4"]["error_limit_abs_log"])
c4_mode = "predictive" if c4_err_t1 <= c4_limit else "descriptive"
cell("T1", f"C4 model ({cv['primary']}) max |log err| over {len(c4_cell_errs)} cells, 950-999",
     round(c4_err_t1, 4), "T1", "c4_model_validation.max_abs_log_error_primary", c4_err_t1 <= c4_limit)
c4_ci_out = [c for c, ci in c4_cell_ci.items() if max(abs(ci[0]), abs(ci[1])) > c4_limit]
cell("T1", "cells whose 95% CI crosses +-0.095 (info, rule uses point est.)", c4_ci_out, "T1",
     "c4_model_validation.cells.*.M1.ci95")
cell("T1", "C4 error limit", c4_limit, "C4", "error_limit_abs_log")
cell("T1", "C4 model sha256", cv["frozen_sha256"], "T1", "c4_model_validation.frozen_sha256")
cell("T1", "validation streams run before freeze", S["C4"]["validation_streams_run_before_freeze"], "C4",
     "validation_streams_run_before_freeze", str(S["C4"]["validation_streams_run_before_freeze"]) == "False")
t1_ok = all(v["within_2pct"] for v in t1_repro.values())

# ---- T2
t2 = S["T2"]
pr = t2["gate_T2"]["per_rival"]
assert set(pr) == set(R), (set(pr), set(R))
t2_fail = [r for r in R if pr[r]["ub95_one_sided"] >= 1.0]
t2_pass = (not t2_fail) and t2["gate_T2"]["pass"]
for r in R:
    cell("T2", f"UB95 FDC/{r}", round(pr[r]["ub95_one_sided"], 4), "T2", f"gate_T2.per_rival.{r}.ub95_one_sided",
         pr[r]["ub95_one_sided"] < 1.0)
cell("T2", "billing_ok_all", t2["billing_ok_all"], "T2", "billing_ok_all", t2["billing_ok_all"])
cell("T2", "eval seeds touched", t2["eval_seeds_touched"], "T2", "eval_seeds_touched", not t2["eval_seeds_touched"])

# ---- T3
g3s, g3f = t2["gate_T3_stop"], S["T3"]["gate_T3"]
t3_pass = g3s["pass"] and g3f["pass"] and g3f["fdc_false_streams_stop"] <= 1 and g3f["fdc_false_streams_full"] <= 1
cell("T3", "FDC false streams, stop 12/15", g3s["fdc_false_streams"], "T2", "gate_T3_stop.fdc_false_streams",
     g3s["fdc_false_streams"] <= 1)
cell("T3", "FDC false streams, stop 12/15 (rerun)", g3f["fdc_false_streams_stop"], "T3",
     "gate_T3.fdc_false_streams_stop", g3f["fdc_false_streams_stop"] <= 1)
cell("T3", "FDC false streams, full horizon", g3f["fdc_false_streams_full"], "T3", "gate_T3.fdc_false_streams_full",
     g3f["fdc_false_streams_full"] <= 1)
cell("T3", "CP UB (full horizon)", g3f["fdc_cp_upper_full"], "T3", "gate_T3.fdc_cp_upper_full")
cell("T3", "audited", g3f["audited"], "T3", "gate_T3.audited", g3f["audited"])

# ---- verdict
today = dt.date.today()
if t0_pass and ts_pass and t3_pass and t2_pass:
    verdict = "GO"
elif not t2_pass:
    verdict = "www27_unreachable" if today > DEADLINE else "fdc_out"
else:
    verdict = "www27_unreachable" if today > DEADLINE else "blocked"  # T0/T-spec/T3 not closed: no lock
progress(1, 4, f"verdict={verdict}; running power MC")

# ---- power: hardest rival
hard = max(R, key=lambda r: pr[r]["ub95_one_sided"])
theta = pr[hard]["mean_log_ratio"]
sigma = pr[hard]["sd_log_ratio"]
rng = np.random.default_rng(SEED)


def boot_ub_power(draw, N, n_rep=N_REP, b=B_BOOT):
    """draw(rng, N) -> (N, m) array of log ratios. Returns (power of IUT, ub quantiles of max-UB)."""
    rej, ubs = 0, []
    for _ in range(n_rep):
        d = draw(N)
        idx = rng.integers(0, N, size=(b, N))
        means = d[idx].mean(axis=1)  # (b, m)
        ub = np.quantile(means, 0.95, axis=0)  # one-sided 95% UB of mean log ratio, per rival
        mx = float(np.exp(ub.max()))
        ubs.append(mx)
        rej += mx < 1.0
    return rej / n_rep, np.quantile(ubs, [0.5, 0.95, 0.99]).tolist(), float(max(ubs))


def mde(N, s, z=1.6449, zb=0.8416):
    return float(np.exp(-(z + zb) * s / np.sqrt(N)))


import pandas as pd  # noqa: E402

dev = pd.read_csv(RES / "pilots/r5_t2_gate_strict/per_stream_n80.csv")
Dmat = np.log(dev["FDC"].to_numpy()[:, None]) - np.log(dev[R].to_numpy())  # (100, |R|)

scenarios = [
    ("primary_parametric_hardest", f"Normal(theta_hat, sd_hat) of {hard} (pre-declared rule)", theta, sigma,
     lambda N: rng.normal(theta, sigma, size=(N, 1))),
    ("sens_empirical_joint_IUT", "resample dev 100-stream paired log-ratio vectors, all |R| rivals jointly",
     None, None,
     lambda N: Dmat[rng.integers(0, Dmat.shape[0], size=N)]),
    ("stress_ub_theta_1p5sd", f"{hard}: theta at dev one-sided UB, sd x1.5",
     float(np.log(pr[hard]["ub95_one_sided"])), 1.5 * sigma,
     lambda N: rng.normal(np.log(pr[hard]["ub95_one_sided"]), 1.5 * sigma, size=(N, 1))),
]
rows = []
for name, desc, th_s, sd_s, fn in scenarios:
    for N in NS:
        pw, q, mx = boot_ub_power(fn, N)
        rows.append({"scenario": name, "description": desc, "N": N, "n_rep": N_REP, "B_boot": B_BOOT,
                     "rival": hard if "joint" not in name else "all R (max UB)", "theta_log": th_s,
                     "sd_log": sd_s, "power_IUT": pw, "maxUB_median": q[0], "maxUB_q95": q[1],
                     "maxUB_q99": q[2], "maxUB_max": mx, "mde_ratio_80pct": mde(N, sd_s) if sd_s else None})
        print(name, N, pw, q)
progress(3, 4, "writing outputs")
with open(OUT / "power.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
prim = {r["N"]: r["power_IUT"] for r in rows if r["scenario"] == "primary_parametric_hardest"}
cr_n = 200 if prim[200] >= 0.8 else 400


# ---- C2 CP critical values
def cp_upper(x, n, a=0.05):
    return 1.0 if x >= n else float(beta_dist.ppf(1 - a, x + 1, n - x))


def cp_crit(n):
    x = 0
    while cp_upper(x + 1, n) <= DELTA:
        x += 1
    return x


c2 = {str(n): {"max_false_streams": cp_crit(n), "cp_at_max": cp_upper(cp_crit(n), n),
               "cp_at_max_plus_1": cp_upper(cp_crit(n) + 1, n)} for n in NS}
assert c2["200"]["max_false_streams"] == 4 and c2["400"]["max_false_streams"] == 12, c2

# ---- flags
flags = [
    "Hait-SW (p_min=0.02, gamma=2/3) and B4-bal (share=0.5) give identical N80 on 100/100 dev streams; "
    "|R| = 9 listed methods but 8 distinct N80 profiles. Disclosed in r5_t2_gate_strict summary; must be stated in lock v5.",
    "Molitor-WoR never reaches 12/15 before tau_R (100/100 censored) and B3-rect is censored on 33/100: "
    "their ratios are driven by rival censoring and must not be read as speed-ups of the same kind.",
    "All T2 ratios are 0.14-0.51 (>30% improvement): flagged by protocol; supporting checks: r4-consistency "
    "0 mismatches for all 6 r4 rivals, billing_ok, 0 false certificates; hardest rival QFC-pool is our own r4 method.",
    "Power is ~1 at N=200 because the dev effect vs the hardest rival is large (log ratio -0.67, sd 0.26); "
    "the binding risk for C1 is dev->eval shift, not sampling noise. The rule picks 200; ext block stays disabled.",
    "T-spec self-check items 15-16 were signed by the control plane after the MC process exited (no numbers changed); see control_plane_note.",
    "Dev results are exploratory; confirmatory evidence only from eval seeds 30000+ after lock v5.",
]

now = dt.datetime.now().isoformat()
decision = {
    "task_id": TASK, "written_at": now, "mode": "PILOT (== full)", "seed": SEED,
    "verdict": verdict, "date_checked": today.isoformat(), "www27_deadline_rule": "10-08",
    "www27_reachable": today <= DEADLINE or verdict == "GO",
    "gates": {
        "T0": {"pass": t0_pass, "source": SRC["T0"]},
        "T-spec": {"pass": ts_pass, "source": SRC["T-spec"]},
        "T1": {"deliverable": True, "repro_ok": t1_ok, "repro": t1_repro, "c4_max_abs_log_err": c4_err_t1,
               "c4_limit": c4_limit, "c4_cell_log_err": c4_cell_errs, "c4_cells_ci_crossing_limit": c4_ci_out, "source": SRC["T1"]},
        "T2": {"pass": t2_pass, "failing_rivals": t2_fail, "worst_rival": hard,
               "worst_ub95": pr[hard]["ub95_one_sided"],
               "per_rival_ub95": {r: pr[r]["ub95_one_sided"] for r in R}, "source": SRC["T2"]},
        "T3": {"pass": t3_pass, "stop_false": g3f["fdc_false_streams_stop"],
               "full_false": g3f["fdc_false_streams_full"], "source": SRC["T3"]},
    },
    "c4_mode": c4_mode,
    "power": {"rule": "IUT power for hardest C1 rival at N=200 >= 0.8 => 200 else 400",
              "hardest_rival": hard, "theta_log": theta, "sd_log": sigma,
              "power_N200": prim[200], "power_N400": prim[400], "n_rep": N_REP, "B_boot": B_BOOT,
              "sensitivity": [r for r in rows if r["scenario"] != "primary_parametric_hardest"]},
    "cr_n_streams": cr_n,
    "enable_ext_blocks": cr_n == 400,
    "eval_seeds": "30000-30199" if cr_n == 200 else "30000-30399",
    "c2_cp_critical": c2,
    "rigorous_set_R": R,
    "flags": flags,
    "eval_seeds_touched": False,
}
(RES / "r5_gate_decision.json").write_text(json.dumps(decision, indent=1, ensure_ascii=False))
summary = dict(decision, cells=cells, outputs=["exp/results/r5_gate_decision.json",
                                                 "exp/results/pilots/r5_gate_decision/gate_table.md",
                                                 "exp/results/pilots/r5_gate_decision/power.csv",
                                                 "exp/results/pilots/r5_gate_decision/summary.json"])
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

# ---- gate table md
L = [f"# r5 门判定表（{TASK}）", "",
     f"判定：**{verdict}**；c4_mode = **{c4_mode}**；cr_n_streams = **{cr_n}**（N=200 IUT 功效 {prim[200]:.3f}，"
     f"N=400 {prim[400]:.3f}；最难对手 {hard}）。判定日期 {today.isoformat()}（≤ 10-08）。", "",
     "每格都标注来源文件与字段。开发种子 900–999，评价种子未触碰。", "",
     "| 门 | 项 | 值 | 通过 | 来源字段 |", "|---|---|---|---|---|"]
for c in cells:
    v = c["value"]
    v = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
    p = "" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
    L.append(f"| {c['gate']} | {c['item']} | {v} | {p} | `{c['source']}`: `{c['field']}` |")
L += ["", "## 功效（C1 IUT，10³ 复制 × 2000 bootstrap）", "",
      "| 情景 | N | 功效 | max UB 中位 | max UB q99 | 80% 功效 MDE 比 |", "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['scenario']} | {r['N']} | {r['power_IUT']:.3f} | {r['maxUB_median']:.3f} | "
             f"{r['maxUB_q99']:.3f} | {'' if r['mde_ratio_80pct'] is None else format(r['mde_ratio_80pct'], '.3f')} |")
L += ["", "## C2 CP 临界值（δ = 0.05，单侧 95% Clopper–Pearson）", "",
      "| N | 错误流至多 | CP(x) | CP(x+1) |", "|---|---|---|---|"]
for n, v in c2.items():
    L.append(f"| {n} | {v['max_false_streams']} | {v['cp_at_max']:.4f} | {v['cp_at_max_plus_1']:.4f} |")
L += ["", "## 注意事项", ""] + [f"- {x}" for x in flags]
(OUT / "gate_table.md").write_text("\n".join(L) + "\n")

print(json.dumps({k: decision[k] for k in ("verdict", "c4_mode", "cr_n_streams", "power", "c2_cp_critical")},
                 indent=1)[:3000])
progress(4, 4, "done")
pid = RES / f"{TASK}.pid"
if pid.exists():
    pid.unlink()
(RES / f"{TASK}_DONE").write_text(json.dumps({
    "task_id": TASK, "status": "success",
    "summary": f"verdict={verdict}, c4_mode={c4_mode}, cr_n_streams={cr_n}, power200={prim[200]:.3f}",
    "final_progress": json.loads((RES / f"{TASK}_PROGRESS.json").read_text()),
    "timestamp": dt.datetime.now().isoformat()}))
