"""r4_prereg_lock: write plan/prereg_lock.json version 4 (methodology §2, §2.1, §4.1, §5, R1-R7, F1-F5).

Inputs (read only): exp/results/r4_gate_decision.json, exp/results/r4_gates/{eps,ms_frozen}.json, the dev-stream rows of
pilots/r4_g1_shape_{a,b,c} (power + CR9 unit prices), the setup / gate summaries (HR8, MS unit prices), plan/feasibility_r4/
(sha256 only), pilots/r4_setup_fcc/contract_table.md (sha256). No evaluation seed is touched; no method is run.

Steps
1. Power (E2, §2.1): paired dev log ratios d_X = log N80_QFC - log N80_X for the 5 primary opponents (B1, B4, B2/B3/Peace
   primary variants per r4_gate_decision). Each replicate resamples N dev streams jointly (pairing and cross-opponent
   correlation kept), recentres every d_X to log 0.70, computes the stream-bootstrap 95% CI and one-sided p = P*(mean >=
   log 0.8) (B_boot draws), applies Holm over the real number of comparisons (5: CR9 x 5 opponents; lr_in_E2=false) and
   counts E2 pass = every comparison Holm-rejected AND CI upper < log 0.8. N in {200, 400}. power(200) < 0.8 ->
   cr_n_streams = 400 and r4_cr_main_ext enabled (pre-committed rule).
2. Unit prices: per-method dev sec/stream (concurrent 4-slot timings) -> projected minutes per confirmatory chunk
   = units x sum(sec) / workers / 60 x 1.2 + max single unit / 60 + 2 min fixed. Chunks > 55 min are bisected on their
   stream / instance range (suffix _s1.._sK, K a power of 2) and the split is written to the lock.
3. Lock: version 4, status 'locked' or 'locked_no_eval' (both pillars dead at r4_gate_decision), sha256; git commit of
   code + lock, commit hash recorded, re-hash, second commit (same procedure as r3).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
PIL = RES_ROOT / "pilots"
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "r4_prereg_lock"
WORKERS = 4
HARD_MAX = 55.0
SAFETY = 1.2
FIXED_MIN = 2.0
LOG08, LOG07 = math.log(0.8), math.log(0.7)

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")


def load(p: Path):
    return json.loads(Path(p).read_text())


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(WS.resolve()))


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total,
        "loss": None, "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                       "final_progress": fp,
                                                       "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------------ dev stream data
def dev_rows():
    """method -> {perm_seed: row} on CR9 dev streams 900-999 (G1 parts a/b/c)."""
    out: dict[str, dict[int, dict]] = {}
    for f in ("r4_g1_shape_a/streams.jsonl", "r4_g1_shape_b/results.jsonl", "r4_g1_shape_c/results.jsonl"):
        for line in open(PIL / f):
            r = json.loads(line)
            if r.get("layer", "CR9") != "CR9" or "N80" not in r:
                continue
            out.setdefault(r["method"], {})[int(r["perm_seed"])] = r
    return out


def power_block(rows, primary, n_list=(200, 400), reps=1000, B=2000, seed=42):
    opps = ["B1", "B4", primary["B2"], primary["B3"], primary["Peace"]]
    seeds = sorted(rows["QFC"])
    for o in opps:
        assert sorted(rows[o]) == seeds, o
    taus = {r.get("tau_R") for d in rows.values() for r in d.values() if r.get("tau_R")}
    assert len(taus) == 1, taus                      # one dev half -> one tau_R
    tau_R = taus.pop()

    def n80(r):
        return float(tau_R if (r["N80"] is None or r["censored"]) else r["N80"])

    q = np.log([n80(rows["QFC"][s]) for s in seeds])
    D = np.stack([q - np.log([n80(rows[o][s]) for s in seeds]) for o in opps])     # (5, 100)
    dev = {o: {"mean_log_ratio": float(D[i].mean()), "geomean_ratio": float(math.exp(D[i].mean())),
               "sd_log_ratio": float(D[i].std(ddof=1)), "frac_zero_diff": float(np.mean(D[i] == 0)),
               "censored_opponent": int(sum(bool(rows[o][s]["censored"]) for s in seeds))}
           for i, o in enumerate(opps)}
    corr = np.corrcoef(D).round(4).tolist() if np.all(D.std(axis=1) > 0) else None
    Dc = D - D.mean(axis=1, keepdims=True) + LOG07
    rng = np.random.default_rng(seed)
    m = len(opps)
    res = {}
    csv_rows = []
    for N in n_list:
        rej = np.zeros((reps, m), bool)
        cipass = np.zeros((reps, m), bool)
        for r in range(reps):
            idx = rng.integers(0, D.shape[1], size=N)
            d = Dc[:, idx]                                   # (m, N)
            bidx = rng.integers(0, N, size=(B, N))
            bm = d[:, bidx].mean(axis=2)                     # (m, B)
            p = (bm >= LOG08).mean(axis=1)
            hi = np.quantile(bm, 0.975, axis=1)
            order = np.argsort(p, kind="mergesort")
            adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
            a = np.empty(m)
            a[order] = adj
            rej[r] = a <= 0.05
            cipass[r] = hi < LOG08
        both = rej & cipass
        joint = float(np.mean(both.all(axis=1)))
        per = {o: float(both[:, i].mean()) for i, o in enumerate(opps)}
        # analytic cross-check: normal approximation, Bonferroni alpha/m (conservative for Holm), one-sided
        from scipy.stats import norm
        z = norm.ppf(1 - 0.05 / m)
        ana = {o: float(norm.cdf((LOG08 - LOG07) / (dev[o]["sd_log_ratio"] / math.sqrt(N)) - z)) for o in opps}
        res[str(N)] = {"joint_power_E2": joint, "per_comparison_power": per, "analytic_bonferroni_per": ana,
                       "analytic_bonferroni_joint_lower": float(np.prod(list(ana.values())))}
        for o in opps:
            csv_rows.append({"n_streams": N, "comparison": f"QFC_vs_{o}", "dev_sd_log_ratio": dev[o]["sd_log_ratio"],
                             "dev_geomean_ratio": dev[o]["geomean_ratio"], "true_ratio_sim": 0.70,
                             "power_holm_mc": per[o], "power_analytic_bonf": ana[o], "joint_power_E2_mc": joint,
                             "reps": reps, "B_boot": B})
    p200 = res["200"]["joint_power_E2"]
    decision = 200 if p200 >= 0.8 else 400
    return {"source": "pilots/r4_g1_shape_{a,b,c} dev streams 900-999 (CR9, eps*=0.001)",
            "opponents_primary": opps, "n_comparisons_holm": m,
            "n_comparisons_note": "real count: CR9 x {B1, B2-primary, B3-primary, Peace-primary, B4}; lr_in_E2=false, HR8 E1-only",
            "method": ("joint stream resampling of dev paired log ratios, recentred to log 0.70 per opponent; per replicate "
                       "stream bootstrap (B draws) -> 95% CI upper and p = P*(mean >= log 0.8); Holm alpha 0.05; "
                       "E2 pass = all comparisons Holm-rejected AND CI upper < log 0.8"),
            "reps": reps, "B_boot": B, "seed": seed, "tau_R_dev": tau_R, "dev": dev, "dev_corr": corr,
            "results": res, "rule": "power(N=200) < 0.8 -> cr_n_streams = 400 (+30200-30399, r4_cr_main_ext enabled)",
            "cr_n_streams": decision, "ext_enabled": decision == 400,
            "caveat": ("N80 lives on 20 log checkpoints (adjacent ratio ~1.30), so dev log ratios are lattice-valued; the "
                       "simulation keeps that lattice. Power is evaluated at a hypothetical true ratio 0.70 as "
                       "pre-registered, NOT at the dev-observed ratios (B2-fav 1.67, Peace-fav 0.99 at which E2 cannot pass).")
            }, csv_rows


# ------------------------------------------------------------------------------------------------ unit prices
def price_stats(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return {"mean": float(a.mean()), "p90": float(np.quantile(a, 0.9)), "max": float(a.max()), "n": int(a.size)}


def unit_prices(rows):
    cr9 = {m: price_stats([r.get("sec_wall", r.get("sec_total")) for r in d.values()]) for m, d in rows.items()}
    for m in cr9:
        cr9[m]["source"] = "pilots/r4_g1_shape_{a,b,c} dev 900-999 (4-slot concurrent)"
    ba = load(PIL / "r4_setup_baselines_a/summary.json")
    bb = load(PIL / "r4_setup_baselines_b/summary.json")
    hr8 = {}
    for s, src in ((ba, "r4_setup_baselines_a"), (bb, "r4_setup_baselines_b")):
        for m, v in s.get("HR8", {}).get("timing", {}).items():
            hr8[m] = {"mean": v["sec_per_stream_mean"], "max": v["sec_per_stream_max"], "n": v.get("streams", 4),
                      "source": f"pilots/{src} HR8 smoke"}
    cr_h8 = {m: {"mean": v["sec_per_stream_mean"], "max": v["sec_per_stream_max"], "n": v["streams"],
                 "source": "pilots/r4_setup_baselines_b CR9 smoke"}
             for m, v in bb["CR9"]["timing"].items() if m.startswith("H8")}
    gd = load(RES_ROOT / "r4_gate_decision.json")
    cr12 = {m: {"mean": v, "max": v * 1.5, "n": None, "source": "r4_gate_decision unit_prices_sec.CR12 (max = 1.5 x mean, assumed)"}
            for m, v in gd["unit_prices_sec"]["CR12_sec_per_stream"].items()}
    haz = load(PIL / "r4_g_haz/summary.json")["sec_per_stream"]
    fa = load(PIL / "r4_g_fac_a/summary.json")["runtime"]
    fb = load(PIL / "r4_g_fac_b/summary.json")
    msb = load(PIL / "r4_setup_ms_baselines/summary.json")["smoke"]["cells"]
    ms = {
        "MS-H|FCC": {"mean": fa["sec_per_instance_median"], "max": fa["sec_per_instance_max"], "source": "r4_g_fac_a (T_ext, 8 ck)"},
        "MS-S|FCC": {"mean": fb["per_layer"]["MS-S"]["sec_per_instance"]["mean"], "max": fb["per_layer"]["MS-S"]["sec_per_instance"]["max"], "source": "r4_g_fac_b"},
        "MS-R3|FCC": {"mean": fb["per_layer"]["MS-R3"]["sec_per_instance"]["mean"], "max": fb["per_layer"]["MS-R3"]["sec_per_instance"]["max"], "source": "r4_g_fac_b smoke (5)"},
        "MS-F|FCC": {"mean": fb["per_layer"]["MS-F"]["sec_per_instance"]["mean"], "max": fb["per_layer"]["MS-F"]["sec_per_instance"]["max"], "source": "r4_g_fac_b smoke (5)"},
    }
    for L in ("MS-H", "MS-S"):
        for m, key in (("JPC", f"main|{L}|JPC"), ("Guard_c3", f"main|{L}|Guard_c3"), ("MisLid-ms", f"mislid|{L}|MisLid-ms")):
            ms[f"{L}|{m}"] = {"mean": haz[key]["mean"], "max": haz[key]["max"], "source": "r4_g_haz (T_max 6000)"}
    for L in ("MS-H", "MS-S", "MS-R3", "MS-F"):
        for m in ("DF-sep-ledger", "PolicyCert-tab", "always-dynamic-JPC", "PERP-fact", "JPC"):
            k = f"{L}|{m}"
            if k in msb and k not in ms:
                v = msb[k]["sec_per_instance_projected_15"]
                ms[k] = {"mean": v, "max": v * 1.5, "source": "r4_setup_ms_baselines smoke (2 problems x15, T_max 6000; max assumed 1.5x)"}
    # T_ext descriptive methods: no direct pilot price at T_ext = 200k -> extrapolated, flagged
    f_ratio = ms["MS-H|FCC"]["mean"] / msb["MS-H|FCC"]["sec_per_instance_projected_15"]
    ms["MS-H|PolicyCert-tab@T_ext"] = {"mean": msb["MS-H|PolicyCert-tab"]["sec_per_instance_projected_15"] * f_ratio,
                                       "max": msb["MS-H|PolicyCert-tab"]["sec_per_instance_projected_15"] * f_ratio * 1.5,
                                       "source": "EXTRAPOLATED: smoke@6000 x (FCC T_ext / FCC smoke) ratio; task must self-time first 4 instances"}
    v = msb["MS-H|DF-sep-ledger"]["sec_per_instance_projected_15"] * (200000 / 6000)
    ms["MS-H|whole-trial-BAI@T_ext"] = {"mean": v, "max": v * 1.5,
                                        "source": "EXTRAPOLATED: DF-sep smoke@6000 scaled linearly to 200k steps (no BAI pilot price)"}
    # one instance of BAI@T_ext alone exceeds 55 min -> its schedulable unit is (instance, problem)
    ms["MS-H|whole-trial-BAI@T_ext/problem"] = {"mean": v / 15, "max": v / 15 * 1.5,
                                                "source": "EXTRAPOLATED: BAI@T_ext per instance / 15 problems"}
    lr = {"QFC": {"mean": 0.6352 + cr9["QFC"]["mean"], "max": 2 * (0.6352 + cr9["QFC"]["mean"]),
                  "source": "EXTRAPOLATED: LR8 schedule 0.635 s (r4_setup_pool_replay) + CR9 QFC certify cost; no LR QFC pilot"}}
    return {"CR9": cr9, "CR12": cr12, "HR8": hr8, "CR_H8": cr_h8, "LR8": lr, "MS": ms,
            "note": "并发运行（4 槽），计时偏高；EXTRAPOLATED 条目没有直接 pilot 单价"}


def chunk_spec(primary):
    P = primary
    cr_ext_others = ["A-Ney", "A-XY", "B5"] + [v for b in ("B2", "B3") for v in (f"{b}-rect", f"{b}-nominal", f"{b}-fav") if v != P[b]]
    return [
        # task, layer, (lo, hi) seed range, methods, unit
        ("r4_cr_main_a", "CR9", (30000, 30199), ["QFC", "B1", "B4", "A-Ney", "A-XY", "B5"], "stream"),
        ("r4_cr_main_b", "CR9", (30000, 30199), ["B2-rect", "B2-nominal", "B2-fav"], "stream"),
        ("r4_cr_main_c1", "CR9", (30000, 30199), ["B3-rect", "B3-nominal"], "stream"),
        ("r4_cr_main_c2", "CR9", (30000, 30199), ["B3-fav"], "stream"),
        ("r4_cr_main_d1", "CR9", (30000, 30199), ["Peace-rect", "Peace-nominal"], "stream"),
        ("r4_cr_main_d2", "CR9", (30000, 30199), ["Peace-fav"], "stream"),
        ("r4_cr_main_ext", "CR9", (30200, 30399), ["QFC", "B1", "B4", P["B2"], P["B3"], P["Peace"]], "stream"),
        ("r4_cr_main_ext_b", "CR9", (30200, 30399), cr_ext_others, "stream"),
        ("r4_cr_main_ext_c", "CR9", (30200, 30399), [v for v in ("Peace-rect", "Peace-nominal", "Peace-fav") if v != P["Peace"]], "stream"),
        ("r4_cr12_scale", "CR12", (30000, 30099), ["QFC", "QFC-DP", "B1", "B4"], "stream"),
        ("r4_cr_h8", "CR_H8", (30000, 30199), ["H8-JPC1", "H8-Tboot", "H8-MisLid"], "stream"),
        ("r4_hr8_secondary", "HR8", (32000, 32099), ["QFC", "B1", "B4", P["B2"], P["B3"], P["Peace"]], "stream"),
        ("r4_lr_e1", "LR8", (31000, 31099), ["QFC"], "stream"),
        ("r4_ms_h_a", "MS", (20000, 20049), ["MS-H|FCC", "MS-H|JPC"], "instance"),
        ("r4_ms_h_b", "MS", (20050, 20099), ["MS-H|FCC", "MS-H|JPC"], "instance"),
        ("r4_ms_h_c", "MS", (20000, 20099), ["MS-H|Guard_c3", "MS-H|MisLid-ms", "MS-H|DF-sep-ledger"], "instance"),
        ("r4_ms_s_a", "MS", (20200, 20249), ["MS-S|FCC", "MS-S|JPC"], "instance"),
        ("r4_ms_s_b", "MS", (20250, 20299), ["MS-S|FCC", "MS-S|JPC"], "instance"),
        ("r4_ms_s_c", "MS", (20200, 20299), ["MS-S|Guard_c3", "MS-S|MisLid-ms", "MS-S|DF-sep-ledger"], "instance"),
        ("r4_ms_r3_a", "MS", (20400, 20449), ["MS-R3|FCC", "MS-R3|JPC"], "instance"),
        ("r4_ms_r3_b", "MS", (20450, 20499), ["MS-R3|FCC", "MS-R3|JPC"], "instance"),
        ("r4_ms_f_a", "MS", (20600, 20649), ["MS-F|FCC", "MS-F|JPC"], "instance"),
        ("r4_ms_f_b", "MS", (20650, 20699), ["MS-F|FCC", "MS-F|JPC"], "instance"),
        ("r4_ms_desc", "MS", (20000, 20049), ["MS-H|PolicyCert-tab@T_ext", "MS-H|always-dynamic-JPC",
                                              "MS-H|PERP-fact"], "instance"),
        # whole-trial BAI at T_ext: unit = (instance, problem); index u -> instance 20000 + u // 15, problem u % 15
        ("r4_ms_desc_bai", "MS", (0, 749), ["MS-H|whole-trial-BAI@T_ext/problem"], "instance_x_problem"),
    ]


def project_one(n, secs, maxs):
    return n * sum(secs) / WORKERS / 60.0 * SAFETY + max(maxs) / 60.0 + FIXED_MIN


def projections(prices, primary, gate, cr_n_streams):
    skipped = set(gate["confirmatory_tasks"]["skipped_by_gate"])
    out, csv_rows = {}, []
    for task, layer, (lo, hi), methods, unit in chunk_spec(primary):
        table = prices[layer]
        secs, maxs, srcs = [], [], set()
        for m in methods:
            p = table[m]
            secs.append(p["mean"]); maxs.append(p["max"]); srcs.add(p["source"])
        n = hi - lo + 1
        full = project_one(n, secs, maxs)
        parts, k = [(lo, hi)], 0
        while project_one(math.ceil(n / 2 ** k), secs, maxs) > HARD_MAX and 2 ** k < n:
            k += 1
        if k:
            step = math.ceil(n / 2 ** k)
            parts = [(a, min(a + step - 1, hi)) for a in range(lo, hi + 1, step)]
        part_min = max(project_one(b - a + 1, secs, maxs) for a, b in parts)
        split = None if k == 0 else [{"task_id": f"{task}_s{i + 1}", "seeds": f"{a}-{b}", "projected_min": round(project_one(b - a + 1, secs, maxs), 1)}
                                     for i, (a, b) in enumerate(parts)]
        status = ("skipped_by_gate" if task in skipped else
                  ("skipped_by_lock (cr_n_streams=200)" if cr_n_streams == 200 else "enabled_by_power")
                  if task.startswith("r4_cr_main_ext") else
                  "lock_defined_split_of_r4_ms_desc (cut order 1; ms2 dead)" if task == "r4_ms_desc_bai" else
                  "permitted_descriptive" if task in gate["confirmatory_tasks"].get("may_still_run_descriptive", []) else
                  "not_scheduled")
        out[task] = {"layer": layer, "seeds": f"{lo}-{hi}", "unit": unit, "n_units": n, "methods": methods,
                     "sec_per_unit_sum_mean": round(sum(secs), 3), "max_unit_sec": round(max(maxs), 2),
                     "projected_min_full": round(full, 1), "split_needed": k > 0, "split": split,
                     "projected_min_max_part": round(part_min, 1), "extrapolated": any("EXTRAPOLATED" in s for s in srcs),
                     "price_sources": sorted(srcs), "status_after_gate": status}
        csv_rows.append({"task_id": task, "layer": layer, "seeds": f"{lo}-{hi}", "n_units": n, "methods": "+".join(methods),
                         "sec_per_unit_sum_mean": round(sum(secs), 3), "projected_min_full": round(full, 1),
                         "split_parts": len(parts), "projected_min_max_part": round(part_min, 1),
                         "extrapolated": out[task]["extrapolated"], "status_after_gate": status})
    return out, csv_rows


# ------------------------------------------------------------------------------------------------ lock content
def code_sha():
    files = sorted((HERE / "dsswm").rglob("*.py")) + sorted(HERE.glob("run_r4_*.py"))
    return {str(p.relative_to(HERE)): sha_file(p) for p in files if "__pycache__" not in p.parts}


def build_lock(gate, eps, msf, power, proj, prices):
    from dsswm.streams.frontier import CR_BUDGETS, CR_EPS_GRID, HR_BUDGETS, HR_COST_TIERS, checkpoints
    both_dead = gate["real_pillar"] == "dead" and gate["ms2_status"] != "alive"
    feas = {rel(p): sha_file(p) for p in sorted((WS / "plan" / "feasibility_r4").iterdir())
            if p.is_file()}
    fcc_diag = {k: v for k, v in feas.items() if "fcc_redesign_diag" in k}
    n_cr = power["cr_n_streams"]
    from dsswm.stats.cp import cp_upper_one_sided
    k200 = max(k for k in range(0, 50) if cp_upper_one_sided(k, 200) <= 0.05)
    k400 = max(k for k in range(0, 80) if cp_upper_one_sided(k, 400) <= 0.05)
    k100 = max(k for k in range(0, 20) if cp_upper_one_sided(k, 100) <= 0.05)
    tauR_dev = power["tau_R_dev"]
    hashes = {p: sha_file(WS / p) for p in ("plan/methodology.md", "plan/task_plan.json", "plan/pilot_plan.json",
                                            "idea/hypotheses.md", "idea/proposal.md", "idea/candidates.json",
                                            "exp/results/r4_gate_decision.json", "exp/results/r4_gates/eps.json",
                                            "exp/results/r4_gates/ms_frozen.json",
                                            "exp/results/pilots/r4_setup_fcc/contract_table.md",
                                            "plan/history/round3_final/prereg_lock.json")
              if (WS / p).exists()}
    for p in sorted((WS / "plan" / "theory").glob("*")):
        if p.is_file():
            hashes[rel(p)] = sha_file(p)
    eval_tasks = {t: v["status_after_gate"] for t, v in proj.items()}
    lock = {
        "version": 4,
        "status": "locked_no_eval" if both_dead else "locked",
        "mode": "pilot",
        "written_by": "r4_prereg_lock",
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "cand_qfc",
        "note": ("Round-4 lock v4. Both pillars were judged dead at r4_gate_decision (real: G1-shape failed vs B2-fav 1.669 "
                 "and Peace-fav 0.987; MS2: dead_fac, MS-H non-vacuity 0.047 < 0.20). The lock is written for the record "
                 "(status=locked_no_eval); every confirmatory task is skipped_by_gate. Only the listed descriptive/safety "
                 "tasks may still be scheduled by the control plane; they cannot change any verdict."),
        "eval_seeds_touched": False,
        "declarations": {
            "pre_pilot_revision": ("修订（Pilot 后锁前修订 R1-R7，2026-10-02）是在任何方法对比之前、只依据宽度/平凡性诊断作出的；"
                                   "诊断文件 plan/feasibility_r4/ 的 sha256 见 feasibility_r4_sha256。"),
            "fcc_revision": ("FCC 锁前修订 F1-F5（2026-10-03）只依据 FCC 自身宽度/间隙诊断（plan/feasibility_r4/fcc_redesign_diag.*，"
                             "开发种子 800-803，探索性），未运行任何基线、未看到方法对基线结果、未触碰评价种子；sha256 见 fcc_redesign_diag_sha256。"),
            "lock_after_gates": ("本 lock 写于全部开发门（开发种子 800-999）之后；冻结值（ε*、主比较变体、Guard c、MisLid κ）由开发门按预注册规则产生，"
                                 "门阈值未改。评价种子 20000-20799 / 30000-30399 / 31000-31199 / 32000-32099 未触碰。"),
            "data_choice_disclosure": "CR9 选择时看过评价半的平凡性与宽度诊断（真值层面，不涉及方法比较）（methodology R1）。",
        },
        "feasibility_r4_sha256": feas,
        "fcc_redesign_diag_sha256": fcc_diag,
        "input_sha256": hashes,
        "gate_decision": {k: gate[k] for k in ("real_pillar", "ms_pillar", "ms2_status", "positive_result_reachable_this_round",
                                                "lr_in_E2", "eps_star", "eps_star_lr", "eps_star_hr8", "primary_variant",
                                                "guard_c", "guard_c_flag", "mislid_hparam", "pivot", "pivot_alternatives",
                                                "pivot_not_viable", "user_req2_partial", "gates")},
        "constants": {
            "delta": 0.05,
            "delta_ledger_real": {"main_pairs_x_checkpoints": 0.04, "variance_cs": 0.01},
            "real_primary_layer": "CR9",
            "CR9": {"data": "Criteo Uplift v2.1 (HF criteo/criteo-uplift), shared/datasets/criteo_uplift_real, fallback=false",
                    "segments": "non-mode indicators of f0, f2, f3, f9 (f6==f0, f8==f2 dropped); 9 non-empty segments",
                    "A": 2, "arms": ["control", "treatment"], "Pi_size": 512, "certificate_sup": "exact enumeration",
                    "outcome": "visit", "problems": [{"q": i, "treat_cost": 1.0, "budget": b} for i, b in enumerate(CR_BUDGETS)],
                    "N80_target": "12/15", "eps_grid": list(CR_EPS_GRID), "eps_star": eps["eps_star"],
                    "checkpoints": {"n_min": 50000, "K": 20, "spacing": "log", "to": "tau_R",
                                    "dev_example": checkpoints(50000, tauR_dev, 20).tolist() if tauR_dev else None},
                    "replan_every": "max(200, ceil(tau_R/2000)) arrivals", "split_seed": 4242,
                    "split_strata": "(CR12 segment x arm)", "billing": "every arrival billed; exhaustion reselect; censor at tau_R"},
            "QFC": {"union": "Q*-union x = S ln A + ln(Q K / delta_main) (qfc_lemma §9 option 1)",
                    "x_CR9": 9 * math.log(2) + math.log(15 * 20 / 0.04),
                    "variance": "Lemma L2 Bernstein inversion UCB (binary), delta_var=0.01 over cells x checkpoints",
                    "exhausted_cells": "removed exactly from V, b", "finite_population_corrections": "NOT used (unproven)",
                    "allocation": "pool proportions (non-adaptive pre-generated schedule)"},
            "HR8": {"segments": "Hillstrom hs3 x category", "Pi_size": 6561, "role": "E1 (safety) + descriptive N80 only",
                    "eps_star_hr8": gate["eps_star_hr8"], "eps_hr8_note": gate["hr8_status"]["detail"],
                    "cost_tiers": HR_COST_TIERS, "budgets": list(HR_BUDGETS), "own_delta_ledger": True},
            "LR8": {"role": "E1 only (QFC), eps 0.0025, 100 streams", "lr_in_E2": False},
            "CR12": {"role": "descriptive scale / S-gap only", "own_delta_ledger": True},
            "FCC": {"certificate": "dsswm.certify.fcc.decide_pdl (pair-difference PDL, uniform random behaviour, no BnB)",
                    "eps": 0.02, "delta": 0.05, "H": "original H in {6, 8}, never shortened",
                    "T_ext": 200000, "checkpoints": [1000, 3000, 6000, 20000, 60000, 100000, 150000, 200000],
                    "no_certification_off_checkpoints": True, "instances_per_layer": 100, "streams_per_instance": 1,
                    "noise_seed": 42, "problems_per_instance": 15},
            "JPC_r3": {"T_max": 6000, "tau": 3000, "eps": 0.02, "version": "r3 locked, via r4 harness"},
            "guard_c": msf.get("guard_c"), "mislid_kappa": msf.get("mislid_ms_inflation_multiple"),
        },
        "variants": {
            "opponents": {"B1": "CLUCB-frontier-joint, WSR20 WoR cell CS radius sum (rigorous)",
                          "B4": "uniform A/B (pool) + cell CS radius sum (rigorous)",
                          "B2": ["B2-rect", "B2-nominal", "B2-fav"], "B3": ["B3-rect", "B3-nominal", "B3-fav"],
                          "Peace": ["Peace-rect", "Peace-nominal", "Peace-fav"],
                          "B5": "maq / multi-arm Qini, descriptive only (not finite-sample valid)",
                          "A-Ney / A-XY": "QFC certificate + frozen non-adaptive allocation (ablation)"},
            "B2_nominal_threshold": "min over {beta_dir, beta_KK, beta_CG(d0=2S)} at the planned median arrivals, frozen before data",
            "primary_variant_rule": ("on the SAME evaluation block, for each of B2, B3, Peace take the first variant in the order "
                                     "fav -> nominal -> rect that satisfies S1 AND S2; written before unblinding, immutable"),
            "dev_reference_primary": gate["primary_variant"],
            "B2_min": "deleted",
        },
        "acceptance": {
            "S1": {"rule": "streams with >= 1 false certification k; one-sided exact CP upper <= 0.05",
                   "max_k": {"n100": k100, "n200": k200, "n400": k400}},
            "S2": "FCR_old = sum n_false / sum n_cert; one-sided 95% instance-cluster bootstrap upper (B=1e4, seed 42) <= 0.05",
            "report": "E[V/(R v 1)]",
            "E1": "CR9: QFC satisfies S1 AND S2 (HR8 and LR8 E1 are separate safety reports with their own delta ledger)",
            "E2": ("CR9: for each X in {B1, B2-primary, B3-primary, Peace-primary, B4}: paired mean log(N80_QFC / N80_X) "
                   "95% stream-bootstrap CI upper < log 0.80; Holm over the 5 comparisons with one-sided bootstrap "
                   "p = P*(mean >= log 0.8); censored N80 := tau_R"),
            "E3": "CR9: QFC uncensored N80 share point >= 0.6 AND one-sided 95% CP lower >= 0.5",
            "E1_ms_prime": ("MS-H and MS-S: FCC(PDL) at T_ext satisfies S1 (<= 1/100 false streams, CP upper 0.0466) AND S2 "
                            "AND non-vacuity (>= 20% of problems certified within T_ext)"),
            "E_ms_d": "same eval block: r3 JPC (T_max 6000) fails S1 or S2 (FWER CP lower > delta OR FCR_old cluster lower > delta)",
            "E_ms_abc": "deleted (diagnosed unreachable before the lock; F2)",
            "descriptive_only": ["Guard (c frozen)", "MisLid-ms (kappa frozen)", "DF-sep-ledger", "PolicyCert-tab",
                                 "whole-trial BAI", "always-dynamic JPC", "PERP-fact", "H8 in-model certifiers", "B5", "MS-R3", "MS-F"],
            "verdicts": {"positive_result": "E1 AND E2 AND E3 (CR9) -> '正结果达成', else '正结果未达成'",
                         "ms2_secondary": "E1-ms' AND E-ms(d) on MS-H and MS-S -> '多步次要主张成立', else '不成立'",
                         "independent": True, "no_alternative_exit": True},
            "suspicious_checks": ["QFC/B1 or QFC/B4 N80 ratio < 0.3 -> truth-leak AST audit",
                                  "any method FWER 0 with uncensored > 0.95 -> eps looseness cross-check",
                                  "FCC completion > 0.9 -> contract / zero-width audit"],
        },
        "power": power,
        "cr_n_streams": n_cr,
        "hr_n_streams": n_cr,
        "hr_n_streams_note": "legacy key read by r4_cr_main_ext (== cr_n_streams)",
        "unit_prices_sec": prices,
        "timing_projection": {"rule": (f"projected = units x sum(mean sec/unit) / {WORKERS} workers / 60 x {SAFETY} + max unit / 60 "
                                       f"+ {FIXED_MIN} min; > {HARD_MAX} min -> bisect the seed range (suffix _s1.._sK) until each part "
                                       f"<= {HARD_MAX}; the control plane adds the suffixed tasks"),
                              "per_task": proj},
        "seed_manifest": {
            "row_split": 4242, "real_dev": "900-999", "ms_dev": "800-899", "motivation_only": "600-799",
            "r3_eval_forbidden": "10000-10799",
            "CR9_eval": "30000-30199" + (" + 30200-30399 (ext)" if n_cr == 400 else ""),
            "CR12_eval": "30000-30099", "CR_H8": "30000-30199", "HR8_eval": "32000-32099", "LR8_eval": "31000-31099",
            "MS-H": "20000-20099", "MS-S": "20200-20299", "MS-R3": "20400-20499", "MS-F": "20600-20699",
            "MS_desc": "20000-20049", "post_unblinding_exploratory": ">= 40000",
            "assert_locked": "assert_locked(version=4, task_id=<task>) at start-up of every evaluation task",
        },
        "cut_order": [
            "1. skip r4_ms_desc (and its lock-defined split r4_ms_desc_bai)",
            "2. HR8 secondary layer down to 50 streams (orig.: HR secondary 50 streams)",
            "3. orig. 'B2-min and B2-fav only 100 streams' -> B2-min deleted; r4 adaptation: CR12 scale layer down to 50 streams "
            "(fav can be a primary variant under the r4 rule, so it is never cut)",
            "4. LR: QFC only (already the r4 design; nothing further to cut)",
            "primary-endpoint methods and stream counts are never cut",
        ],
        "contract_table": {"path": "exp/results/pilots/r4_setup_fcc/contract_table.md",
                           "written_before_any_fcc_result": True,
                           "m4_drift": "violates stationarity; FCC makes no claim on m4"},
        "eval_tasks": eval_tasks,
        "confirmatory_tasks": gate["confirmatory_tasks"],
        "no_eval_permitted_tasks": list(gate["confirmatory_tasks"].get("may_still_run_descriptive", [])) if both_dead else [],
        "code_sha256": code_sha(),
        "git_commit": None,
        "sha256": None,
    }
    return lock


def git_commit_lock(lock):
    from dsswm.stats.prereg import canonical_hash
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=WS, capture_output=True, text=True).stdout.strip()
    if not repo:
        return lock, "no git repo"
    real = WS.resolve()
    lock["sha256"] = canonical_hash(lock)
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    paths = [real / "plan" / "prereg_lock.json", real / "exp" / "code" / "dsswm", real / "plan" / "feasibility_r4",
             *sorted((real / "exp" / "code").glob("run_r4_*.py"))]
    subprocess.run(["git", "add", "--", *map(str, paths)], cwd=repo, check=True)
    r = subprocess.run(["git", "commit", "-m", "feat(r4_prereg_lock): lock v4 (locked_no_eval) + frozen r4 code\n\n"
                        "Co-Authored-By: Assistant <noreply@example.org>"], cwd=repo, capture_output=True, text=True)
    msg = r.stdout[-300:] + r.stderr[-300:]
    lock["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    lock["sha256"] = canonical_hash(lock)
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    subprocess.run(["git", "add", "--", str(real / "plan" / "prereg_lock.json")], cwd=repo, check=True)
    r2 = subprocess.run(["git", "commit", "-m", "feat(r4_prereg_lock): record code commit hash in lock v4\n\n"
                         "Co-Authored-By: Assistant <noreply@example.org>"], cwd=repo, capture_output=True, text=True)
    return lock, msg + r2.stdout[-200:]


def update_gpu_progress(start, status, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    d = load(p) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
    end = datetime.now()
    lst = "completed" if status == "success" else "failed"
    if TASK not in d.setdefault(lst, []):
        d[lst].append(TASK)
    d.setdefault("running", {}).pop(TASK, None)
    d.setdefault("timings", {})[TASK] = {"planned_min": 35, "actual_min": max(1, round((end - start).total_seconds() / 60)),
                                         "start_time": start.isoformat(), "end_time": end.isoformat(),
                                         "config_snapshot": snapshot}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-git", action="store_true")
    a = ap.parse_args()
    start = datetime.now()
    t0 = time.perf_counter()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    try:
        progress(0, 5, "load")
        gate = load(RES_ROOT / "r4_gate_decision.json")
        eps = load(RES_ROOT / "r4_gates" / "eps.json")
        msf = load(RES_ROOT / "r4_gates" / "ms_frozen.json")
        rows = dev_rows()
        progress(1, 5, "power")
        tp = time.perf_counter()
        power, pcsv = power_block(rows, gate["primary_variant"], reps=a.reps, B=a.boot)
        power["sec"] = round(time.perf_counter() - tp, 1)
        progress(2, 5, "unit_prices", {"power200": power["results"]["200"]["joint_power_E2"]})
        prices = unit_prices(rows)
        proj, ucsv = projections(prices, gate["primary_variant"], gate, power["cr_n_streams"])
        progress(3, 5, "lock")
        lock = build_lock(gate, eps, msf, power, proj, prices)
        lock["mode"] = a.mode
        from dsswm.stats.prereg import canonical_hash, check_lock, validate_lock
        git_msg = "skipped (--no-git)"
        if a.no_git:
            lock["sha256"] = canonical_hash(lock)
            LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
        else:
            lock, git_msg = git_commit_lock(lock)
        progress(4, 5, "validate")
        on_disk = load(LOCK)
        validate_lock(on_disk, 4)
        checks = {"validate_lock_v4": True}
        try:
            check_lock(on_disk, 4, task_id="r4_cr_main_a")
            checks["confirmatory_blocked"] = False
        except RuntimeError as e:
            checks["confirmatory_blocked"] = True
            checks["confirmatory_block_msg"] = str(e)[:200]
        if on_disk["no_eval_permitted_tasks"]:
            checks["descriptive_permitted"] = bool(check_lock(on_disk, 4, task_id=on_disk["no_eval_permitted_tasks"][0]))
        v3 = WS / "plan" / "history" / "round3_final" / "prereg_lock.json"
        checks["v3_archive_still_valid_with_version3"] = bool(check_lock(load(v3), 3)) if v3.exists() else None
        over = [t for t, v in proj.items() if v["projected_min_max_part"] > HARD_MAX]
        splits = {t: v["split"] for t, v in proj.items() if v["split_needed"]}
        with open(out_dir / "power.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pcsv[0])); w.writeheader(); w.writerows(pcsv)
        with open(out_dir / "unit_prices.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ucsv[0])); w.writeheader(); w.writerows(ucsv)
        passed = checks["validate_lock_v4"] and not over
        summary = {
            "task_id": TASK, "mode": a.mode, "candidate_id": "cand_qfc", "started_at": start.isoformat(),
            "finished_at": datetime.now().isoformat(), "wall_s": round(time.perf_counter() - t0, 1),
            "lock_path": "plan/prereg_lock.json", "lock_version": on_disk["version"], "lock_status": on_disk["status"],
            "lock_sha256": on_disk["sha256"], "git_commit": on_disk["git_commit"], "git_msg": git_msg,
            "metrics": {"lock_written": True, "power_hr": power["results"]["200"]["joint_power_E2"],
                        "power_hr_400": power["results"]["400"]["joint_power_E2"],
                        "cr_n_streams": power["cr_n_streams"],
                        "chunk_projection_max_min": max(v["projected_min_max_part"] for v in proj.values()),
                        "chunk_projection_max_min_unsplit": max(v["projected_min_full"] for v in proj.values())},
            "pass_criteria": {"lock_validates_check_lock_v4": checks["validate_lock_v4"],
                              "every_chunk_le_55_or_split_recorded": not over, "tasks_split": splits},
            "checks": checks, "pass": passed, "go_no_go": "GO" if passed else "NO_GO",
            "power": {k: power[k] for k in ("opponents_primary", "n_comparisons_holm", "dev", "results", "cr_n_streams",
                                            "ext_enabled", "caveat", "reps", "B_boot", "sec")},
            "projection_table": {t: {k: v[k] for k in ("seeds", "n_units", "projected_min_full", "split_needed",
                                                       "projected_min_max_part", "extrapolated", "status_after_gate")}
                                 for t, v in proj.items()},
            "integrity": {"eval_seeds_touched": False, "methods_run": False,
                          "inputs": "gate summaries + dev-stream rows (900-999) only"},
            "timing_note": "单价来自并发运行（4 槽）的 pilot 实测，计时偏高；EXTRAPOLATED 条目无直接 pilot 单价",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
        update_gpu_progress(start, "success", {"mode": a.mode, "power_reps": a.reps, "boot_B": a.boot,
                                               "cpu_workers": 1, "gpu_count": 0,
                                               "note": "CPU only; gpu slot is a scheduling token"})
        progress(5, 5, "done", {"passed": passed})
        mark_done("success", f"lock v4 {on_disk['status']} sha256={on_disk['sha256'][:12]}; power200="
                             f"{summary['metrics']['power_hr']:.3f}; cr_n_streams={power['cr_n_streams']}; "
                             f"max chunk {summary['metrics']['chunk_projection_max_min']} min; pass={passed}")
        print(json.dumps({k: summary[k] for k in ("metrics", "pass_criteria", "checks", "lock_status")}, ensure_ascii=False,
                         indent=1, default=str))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        update_gpu_progress(start, "failed", {"error": repr(e)[:300]})
        mark_done("failed", repr(e)[:500])
        raise


if __name__ == "__main__":
    main()
