"""r2_prereg_lock (governance): write plan/prereg_lock.json version 2 before any evaluation seed (methodology 6.1).

Inputs (dev seeds only): g0_resolution_gate, dev_offgrid_ladder, r2_setup_offgrid, r2_setup_falsify summaries (and
results.jsonl of G0), idea/hypotheses.md (thresholds copied verbatim), plan/task_plan.json (evaluation manifest).
New computation: d_eff = participation ratio tr(I)^2/tr(I^2) of the realised design Fisher at the grid MLE, after the
JPC/DDA run on fresh dev problems (seeds 640-649 x 10 problems, f=1, levels R0 and R1, stream 0, seed 42), using the
same code path as g0_resolution_gate.jpc_instance. Evaluation seeds (>= 10000) are never instantiated.

pilot: status=provisional (round-0 lock kept under `round0`).  full: status=locked + sha256; refuses to overwrite a
locked file.  Usage: run_r2_prereg_lock.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
TASK = "r2_prereg_lock"
R2_SCOPE = {"hr1_r2_a": [10000, 10042], "hr1_r2_b": [10043, 10047], "hr1_r2_c": "skip"}
R2_SCOPE_N = {"hr1_r2_a": 43, "hr1_r2_b": 5, "hr1_r2_c": 0}
RES_ROOT = WS / "exp" / "results"
LOCK = WS / "plan" / "prereg_lock.json"
HYP = WS / "idea" / "hypotheses.md"
EPS, DELTA = 0.02, 0.05
DEFF_SEEDS = list(range(640, 650))
DEFF_NPROB = 10


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp,
                                                        "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------- d_eff on dev problems
def _jpc_level(seed, level, cq, jtrue):
    import run_g0_resolution_gate as G
    orig = G.make_offgrid_instance
    G.make_offgrid_instance = lambda s, lvl, stream=0: orig(s, level, stream=stream)   # force truth level
    try:
        rows, _ = G.jpc_instance(seed, DEFF_NPROB, (1,), cq, jtrue, 3000, False)
    finally:
        G.make_offgrid_instance = orig
    for r in rows:
        r["level"] = level
    return rows


def compute_deff(log, workers):
    import torch
    import run_g0_resolution_gate as G
    from joblib import Parallel, delayed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prop = G.NLPropagator(2, 2, G.NMAX, G.make_offgrid_instance(DEFF_SEEDS[0], "R1").env.aspace, G.C_KNOWN,
                          G.RHO_RET, device=dev)
    probs, probs0 = [], []
    for s in DEFF_SEEDS:
        assert s < 10000
        i1 = G.make_offgrid_instance(s, "R1", stream=0)
        i0 = G.make_offgrid_instance(s, "R0", stream=0)
        probs += [(s, q, prop, G.vstar_of(i1.truth)) for q in i1.problems[:DEFF_NPROB]]
        probs0 += [(q, G.vstar_of(i0.truth)) for q in i0.problems[:DEFF_NPROB]]
    rows = []
    t0 = time.time()
    eta_stats, cq, jtrue1 = G.gpu_phase(probs, (1,), {"all": (1,)}, 2000, log, rows, dev)
    jtrue0 = {q.pid: G.j_values(prop, G.build_plans(prop, q), v[None], q.utility.w, q.utility.w_ret)[0] * cq[q.pid]
              for q, v in probs0}
    log(f"d_eff prep (J tables / eta_loc f=1 / truth J) {time.time() - t0:.0f}s; eta validity violations "
        f"{eta_stats['1']['violations']}/{eta_stats['1']['draws']}")
    del prop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t1 = time.time()
    jobs = [(s, "R0", jtrue0) for s in DEFF_SEEDS] + [(s, "R1", jtrue1) for s in DEFF_SEEDS]
    res = Parallel(n_jobs=workers, backend="loky")(delayed(_jpc_level)(s, lv, cq, jt) for s, lv, jt in jobs)
    jrows = [r for rr in res for r in rr]
    log(f"d_eff JPC runs: {len(jrows)} (problem, level) rows in {time.time() - t1:.0f}s")
    return jrows, rows, eta_stats


def deff_stats(vals):
    v = np.asarray(vals, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "q10": float(np.quantile(v, .1)),
            "q90": float(np.quantile(v, .9)), "min": float(v.min()), "max": float(v.max())}


# ------------------------------------------------------------------------------------------------ helpers
def verbatim_thresholds(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\| (G0|H0|HR\d+b?|HF\d+b?|HD\d+b?|HS1|HH1[ab]|HK)[ （(]", line)
        if m:
            out[m.group(1)] = line
    tiers = [ln for ln in text.splitlines() if ln.startswith("- **档 ")]
    gate = [ln for ln in text.splitlines() if ln.startswith("- 任何格子") or ln.startswith("- 核查清单")]
    header = [ln for ln in text.splitlines() if ln.startswith("> ")]
    return {"rows": out, "tier_wording": tiers, "suspicious_gate": gate, "header_notes": header}


def eval_manifest(plan: dict) -> dict:
    tasks = {}
    for t in plan["tasks"]:
        txt = t["description"] + " " + json.dumps(t.get("full", {}), ensure_ascii=False)
        rng = sorted({(int(a), int(b)) for a, b in re.findall(r"(\d{5})-(\d{5})", txt)})
        if t["id"] in ("r2_setup_offgrid", "r2_setup_falsify"):
            continue                                     # setup tasks: build/unit-test only, no evaluation outcome
        if rng:
            tasks[t["id"]] = [list(r) for r in rng]
    seeds = sorted({s for rr in tasks.values() for a, b in rr for s in range(a, b + 1)})
    main = list(range(10000, 10128))
    body = {"generator_hash": None, "per_task_ranges": tasks, "all_eval_seeds": seeds,
            "main_contrast_seeds": [main[0], main[-1]], "others_subset": [10000, 10047], "e2_seeds": [20000, 20023],
            "streams": [[42, "perm0"], [123, "perm1"], [456, "perm2"]], "K_problems": 15, "n0": 20}
    return body


def projected_minutes(summary: dict):
    """Look for a full-run wall-clock projection in a pilot summary (several naming conventions)."""
    for path in (("timing_projection", "projected_full_total_min"), ("timing_projection", "full_wall_min_projected"),
                 ("timing_projection", "full_wall_min_at_4_workers"), ("projected_full_wall_min",),
                 ("timing", "projected_full_wall_min"), ("timing_projection", "projected_full_wall_min")):
        cur = summary
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, (int, float)):
            return float(cur)
    return None


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    sub = "pilots" if pilot else "full"
    out_dir = RES_ROOT / sub / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    T0 = time.time()
    try:
        old = json.loads(LOCK.read_text())
        if old.get("status") == "locked" and old.get("version") == 2:
            raise RuntimeError("prereg_lock.json v2 is already locked (read-only); refusing to overwrite")
        round0 = old.get("round0", old) if old.get("version") == 2 else old
        log(f"start mode={args.mode}; existing lock version={old.get('version', 0)} status={old.get('status')}")
        progress(1, 5, "read dependency outputs")
        dep = {}
        dep_files = {}
        for t in ("g0_resolution_gate", "dev_offgrid_ladder", "r2_setup_offgrid", "r2_setup_falsify"):
            p = RES_ROOT / sub / t / "summary.json"
            if not p.exists() and not pilot:
                raise RuntimeError(f"full-mode lock needs full output of {t}")
            dep[t] = json.loads(p.read_text())
            dep_files[t] = {"path": str(p.relative_to(WS)), "sha256": sha_file(p), "mode": dep[t].get("mode")}
            assert not dep[t].get("eval_seeds_touched", False), f"{t} touched evaluation seeds"
        g0, lad, sof, sfa = (dep[k] for k in ("g0_resolution_gate", "dev_offgrid_ladder", "r2_setup_offgrid",
                                               "r2_setup_falsify"))
        hyp_text = HYP.read_text()
        plan = json.loads((WS / "plan" / "task_plan.json").read_text())

        # ---------------------------------------------------------------- d_eff on fresh dev problems
        progress(2, 5, "d_eff on dev problems 640-649 x 10 (R0, R1)")
        jrows, gpu_rows, eta_chk = compute_deff(log, args.workers)
        with open(out_dir / "results.jsonl", "w") as fh:
            for r in gpu_rows + jrows:
                fh.write(json.dumps(r, default=float) + "\n")
        by_lv = {lv: [r for r in jrows if r["level"] == lv] for lv in ("R0", "R1")}
        deff = {lv: deff_stats([r["d_eff"] for r in R]) for lv, R in by_lv.items()}
        for lv, R in by_lv.items():
            deff[lv].update({"completion": float(np.mean([r["status"] == "CERTIFIED" for r in R])),
                             "false_certs": int(sum(r["false_cert"] for r in R)),
                             "n_identified_dirs_median": float(np.median([r["n_identified_dirs"] for r in R])),
                             "n_rounds_total_median": float(np.median([r["n_rounds_total"] for r in R])),
                             "h_star_cond1_median": float(np.median([r["h_star_cond1"] for r in R])),
                             "h_star_identified_median": float(np.median([r["h_star_identified"] for r in R]))})
        g0_rows = [json.loads(ln) for ln in open(RES_ROOT / sub / "g0_resolution_gate" / "results.jsonl")]
        g0_deff = [r["d_eff"] for r in g0_rows if r.get("kind") == "jpc" and r.get("f") == 1]
        deff["G0_R1_f1_devseeds_600_611"] = deff_stats(g0_deff)
        deff_locked = round(deff["R0"]["median"], 2)
        log("d_eff medians: " + ", ".join(f"{k} {v['median']:.2f} [{v['q10']:.2f},{v['q90']:.2f}] n={v['n']}"
                                          for k, v in deff.items()))
        samp = [{k: r[k] for k in ("level", "instance", "problem", "status", "new_env_steps", "n_rounds_total",
                                   "d_eff", "fisher_cond", "n_identified_dirs", "h_star", "h_star_identified",
                                   "h_star_cond1", "set_size", "theta_cell_alive", "true_regret", "false_cert")}
                for r in jrows[:6] + [r for r in jrows if r["level"] == "R1"][:6]]
        (out_dir / "samples" / "deff_samples.json").write_text(json.dumps(samp, indent=1, default=float))

        # ---------------------------------------------------------------- f*, family A, h*
        progress(3, 5, "derive locked decisions")
        gate = g0["gate"]
        g0_passed = bool(gate.get("pass_eps_0.02"))
        f_reach = gate["by_eps"]["0.02"]["f_reaching_eps_over_4"]
        jt = g0["jtable_sec_per_problem_by_f"]
        eta_by_f = g0["metrics"]["eta_loc_over_eps_by_f"]
        if f_reach and any(str(f) in ("2", "4") for f in f_reach):
            f_star = int(min(int(f) for f in f_reach if str(f) in ("2", "4")))
            f_rule = "smallest f in {2,4} with median eta_loc <= eps/4 and J table <= 10 s/problem (G0)"
        else:
            f_star = 2
            f_rule = ("G0: no f reaches eta_loc <= eps/4 at eps in {0.02,0.05,0.10}; fallback f*=2 (methodology risk "
                      "table: 'R3 固定用 f=2'). Reason: f=4 lowers median eta_loc/eps only "
                      f"{eta_by_f['2']:.1f} -> {eta_by_f['4']:.1f} (-{100 * (1 - eta_by_f['4'] / eta_by_f['2']):.0f}%) "
                      f"at {jt['4'] / jt['2']:.1f}x J-table time, ~5x eta_loc time and >5 GB GPU peak "
                      "(r2_setup_offgrid note)")
        j1 = g0["jpc_r1"]
        R2_family_A = bool(j1["1"]["share_h_le_hstar_over_2"] > 0.5)
        R3_hcond = bool(j1["2"]["share_h_le_hstar_over_2"] > 0.5)
        ratio_variants = {f: {"h_over_hstar_literal_median": j1[f]["h_over_hstar_median"],
                              "h_over_hstar_identified_median": j1[f]["h_cell_halfdiag"] / j1[f]["h_star_identified"]["median"],
                              "h_over_hstar_cond1_median": j1[f]["h_over_hstar_cond1_median"],
                              "share_h_le_hstar_over_2_literal": j1[f]["share_h_le_hstar_over_2"]}
                          for f in ("1", "2")}
        famA = ["HR1@R0"] + (["HR1@R2"] if R2_family_A else []) + ["HR1@R3"]

        # ---------------------------------------------------------------- downscale decisions
        tp = lad["timing_projection"]["cpu_sec_per_problem"]
        def hr1_proj(level_key, methods, n_inst):
            sec = sum(tp[level_key][m] for m in methods if m in tp[level_key])
            return sec * n_inst * 3 * 15 / 4 / 60.0
        man = eval_manifest(plan)
        downscale = {}
        proj = {
            "hr1_r0_a": (hr1_proj("R0", ["JPC", "B3", "B3g", "B12"], 64), "dev_offgrid_ladder R0 cpu s/problem x 64 inst x 3 streams x 15 / 4 workers; B8, B3-UI, B2 not in ladder -> +unknown"),
            "hr1_r0_b": (hr1_proj("R0", ["JPC", "B3", "B3g", "B12"], 64), "same as hr1_r0_a"),
            "hr1_r2_a": (hr1_proj("R2", ["JPC_infl", "B3", "B3g_infl", "B12_infl"], 43), "R2 grid methods refuse at once (data-free floor); B3 dominates"),
            "hr1_r3_a": (hr1_proj("R3_f2", ["JPC_infl", "B3", "B3g_infl", "B12_infl", "JPC"], 32), "R3 f*=2 incl. uncorrected JPC diagnostic"),
            "hr4_phase_fine": ((g0["eta_loc"]["4"]["eta_sec_per_problem_mean"] + jt["4"]) * 32 * 15 / 60.0,
                               "f=4 eta_loc + J table per eval problem (data-free, per problem not per stream), GPU serial; methods refuse at once"),
        }
        for t in plan["tasks"]:
            tid = t["id"]
            if tid in (TASK, "r2_setup_offgrid", "r2_setup_falsify", "g0_resolution_gate", "dev_offgrid_ladder"):
                continue
            ps = RES_ROOT / "pilots" / tid / "summary.json"
            psum = json.loads(ps.read_text()) if ps.exists() else {}
            own = projected_minutes(psum) if psum else None
            planned = man["per_task_ranges"].get(tid)
            n_plan = sum(b - a + 1 for a, b in planned) if planned else None
            src = None
            if own is not None:
                m, src = own, f"own pilot projection ({ps.relative_to(WS)})"
                n_basis = (psum.get("timing_projection") or {}).get("n_full_instances")
                if tid.startswith(("hr1_r2_", "hr1_r3_")) and n_basis is None:
                    n_basis = 32                      # run_hr1_chunk.py hard-coded 32 before 2026-10-02 fix
                    src += " [basis assumed 32 instances: pre-fix hard-coded value]"
                scope = R2_SCOPE_N.get(tid) if not R2_family_A else None
                n_eff = scope if scope is not None else n_plan
                if n_basis and n_eff is not None and n_basis != n_eff:
                    m = m * n_eff / n_basis
                    src += f" rescaled x{n_eff}/{n_basis} instances"
            elif tid in proj or tid.replace("_b", "_a").replace("_c", "_a").replace("_d", "_a") in proj:
                key = tid if tid in proj else tid[:-1] + "a"
                m, src = proj[key][0], "dev_offgrid_ladder timing: " + proj[key][1]
            else:
                m = None
            if m is None:
                downscale[tid] = {"projected_full_min": None, "decision": "none_pending_own_pilot_timing",
                                  "source": "no pilot timing yet; rule applied at full-mode lock"}
            elif m > 60 and planned:
                k = int(math.ceil(m / 50.0))
                lo, hi = planned[0][0], planned[-1][1]
                if tid in R2_SCOPE and not R2_family_A and isinstance(R2_SCOPE[tid], list):
                    lo, hi = R2_SCOPE[tid]
                n = hi - lo + 1
                edges = [lo + (i * n) // k for i in range(k + 1)]
                downscale[tid] = {"projected_full_min": round(m, 1), "decision": "split",
                                  "n_subchunks": k, "subchunks": [[edges[i], edges[i + 1] - 1] for i in range(k)],
                                  "projected_min_per_subchunk": round(m / k, 1),
                                  "rule": ("projected > 60 min -> split into ceil(proj/50) contiguous sub-chunks of the "
                                           "same instance range (instance count, methods, streams unchanged, so the "
                                           "pre-registered sample size / power is kept); sub-chunk ids <task>_s<i>, "
                                           "results merged by r2_analysis_aggregate"),
                                  "source": src}
            else:
                downscale[tid] = {"projected_full_min": round(m, 1), "decision": "keep", "source": src}
        # methodology 6.1: R2 not in family A -> 48-instance diagnostic only (not a timing decision)
        r2_scope = None
        if not R2_family_A:
            r2_scope = {"instances": [10000, 10047], **R2_SCOPE, "reason": "methodology 6.1: R2 fails h <= h*/2 -> leaves family A, "
                                                      "48-instance diagnostic only"}
            for tid in ("hr1_r2_b", "hr1_r2_c"):
                downscale[tid]["scope_change"] = r2_scope[tid]
                if downscale[tid]["decision"] in ("keep", "none_pending_own_pilot_timing") or tid == "hr1_r2_c":
                    downscale[tid]["decision"] = "scope_change_R2_family_exit"
            downscale["hr1_r2_c"].pop("subchunks", None); downscale["hr1_r2_c"].pop("n_subchunks", None)
        downscale["hr4_phase_fine"]["note"] = ("f*=2 -> hr4_phase_fine runs f=4; f=4 J-table GPU peak 5.5 GB: lower "
                                               "NLPropagator mem_budget_elems to stay <= 5 GB")

        # ---------------------------------------------------------------- tuning grids
        r0b5 = round0.get("B5_tuning_nl_acquisition_factorial", {})
        tuning = {
            "protocol": "dev seeds only (0-999), <= 6 points per method, selection = min mean log(1+new env steps) "
                        "subject to FCR CP upper <= 2*delta; methods without free hyper-parameters use 1 point",
            "JPC": {"grid": [{"m": 5, "explore": 0.05, "two_step": True, "allocation": "TaS LP"}],
                    "selected": {"m": 5, "explore": 0.05, "two_step": True, "allocation": "TaS LP"},
                    "note": "pre-registered DDA default, deliberately untuned (engineering choice, not a contribution)"},
            "JPC_infl": {"grid": "same as JPC", "selected": "same as JPC",
                         "certify": "R_grid(Theta_t) + 2 max_alive eta_loc <= eps; honest refusal if 2 min eta_loc > eps"},
            "B3": {"grid": [{"beta": "sqrt(chi2_{0.95}(12))"}], "selected": {"beta": "wald_beta(d=12, delta=0.05)",
                                                                             "design": "choose_design_action (Wald-opt)"}},
            "B3g": {"grid": [{"explore": 0.05}], "selected": {"explore": 0.05, "design": "max-min blocking gain",
                                                              "set": "Wald ellipsoid intersect G_f, exact J"}},
            "B3-UI": {"grid": [{"beta_UI": "sqrt(2(log(1/delta) - (log q - l(v_hat))))"}],
                      "selected": {"threshold": "UI log(1/delta)", "design": "same as B3"},
                      "note": "quadratic/linearised continuous UI set (favourable to baseline)"},
            "B8": {"grid": [{"tau": "eps/2"}], "selected": {"tau": "eps/2"},
                   "note": "tau=eps (B8tau) reported only as favourable sensitivity, not a denominator"},
            "B12": {"grid": [{"ledger_prior": "reuse logM across problems"}], "selected": {"ledger_prior": True,
                                                                                         "max_steps": 2310000}},
            "B10": {"grid": [{"criterion": "G-opt coverage"}], "selected": {"criterion": "G-opt coverage"}},
            "B11": {"grid_source": "round0 B5 TaskDirected tuning (dev instances 500, 501)",
                    "grid_scores": r0b5.get("dev_scores", {}).get("TaskDirected"),
                    "selected": r0b5.get("selected", {}).get("TaskDirected")},
            "round0_B5_other_acquisitions": {k: v for k, v in r0b5.get("selected", {}).items() if k != "TaskDirected"},
            "uniform": {"grid": [{}], "selected": {}},
            "B1": {"grid": [{"bound": "Hoeffding LUCB"}], "selected": {"max_steps": 2310000}},
            "B2": {"grid": [{"bound": "residual + empirical Bernstein"}], "selected": {"max_steps": 2310000}},
            "B4": {"grid": [{}], "selected": {"rule": "point-estimate argmax"}},
            "B9": {"grid": [{"M": 5}, {"M": 20}, {"M": 100}], "selected": "all three reported (not tuned)"},
            "AGC": {"grid": [{}], "selected": {"bound": "whole-trial empirical Bernstein"}},
            "eprocess": {"grid": [{}], "selected": {"module": "dsswm/audit/eprocess.py"}},
        }

        # ---------------------------------------------------------------- thresholds (verbatim) and decision code
        vt = verbatim_thresholds(hyp_text)
        for k, ln in vt["rows"].items():
            assert ln in hyp_text
        tier_src = HERE / "dsswm" / "stats" / "tier_decision.py"
        from dsswm.stats.tier_decision import decide_tier
        tier_now = decide_tier(g0_passed, famA, {})
        man = eval_manifest(plan)
        man["generator_hash"] = g0["generator_hash"]
        man_sha = hashlib.sha256(json.dumps(man, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        code_hash = {p: sha_file(HERE / p) for p in (
            "dsswm/certify/eta_loc.py", "dsswm/certify/minimax_infl.py", "dsswm/certify/minimax_enum.py",
            "dsswm/evidence/lr_set.py", "dsswm/acquire/nl_kl_dda.py", "dsswm/baselines/b3_ui.py",
            "dsswm/baselines/b12_tas.py", "dsswm/baselines/glm_linearised.py", "dsswm/models/grid_ladder.py",
            "dsswm/streams/offgrid.py", "dsswm/streams/generator.py", "dsswm/stats/tier_decision.py",
            "dsswm/stats/prereg.py", "run_g0_resolution_gate.py", "run_dev_offgrid_ladder.py")}

        lock = {
            "version": 2,
            "status": "provisional" if pilot else "locked",
            "mode": args.mode,
            "written_by": f"{TASK} ({args.mode})",
            "written_at": datetime.now().isoformat(),
            "note": ("Round-1 (RH-JPC) pre-registration. Pilot: provisional, may be rewritten by the full-mode run. "
                     "Full: status=locked + sha256, read-only; every evaluation task calls "
                     "dsswm.stats.prereg.assert_locked() at start-up and refuses to run otherwise."),
            "eval_seeds_touched": False,
            "sources": dep_files,
            "hypotheses_md": {"path": "idea/hypotheses.md", "sha256": sha_file(HYP)},
            "thresholds_verbatim": vt,
            "thresholds": {
                "eps_NL": EPS, "eps_Lin": 0.05, "eps_E2": 0.01, "delta": DELTA,
                "eps_relaxation": {"allowed_set": [0.05, 0.10], "applied": False,
                                   "reason": "G0 unreachable at 0.05 and 0.10 as well (best median eta_loc "
                                             f"{gate['by_eps']['0.02']['best_median_eta']:.3f}, best empirical lower "
                                             f"bound {gate['by_eps']['0.02']['best_emp_lower_bound']:.3f} > 0.10/4) -> "
                                             "tier C path; eps stays 0.02, relaxed eps reported as sensitivity only"},
                "G0": {"violations": 0, "eta_loc_le": "eps/4", "jtable_sec_le": 10},
                "HR1": {"ratio_ci_upper_lt": 0.8, "completion_ge": 0.80, "fcr_cp_upper_le": "2*delta"},
                "HR2": {"threshold_only_ratio_in": [0.2, 0.5], "fail_if_ratio_gt": 0.7},
                "HR2b": {"pred_ratio_in": [0.67, 1.5], "share_ge": 0.70, "only_N_ge": 50},
                "HR3": {"jonckheere_p_lt": 0.05, "bin": "eta_dec >= eps/2", "fcr_cp_lower_gt": "delta"},
                "HR4": {"inflection_in": ["h*/2", "2h*"]},
                "HR5": {"ratio_ci_upper_lt": 1.0, "spearman_ge": 0.7},
                "HF1": {"ratio_le": 1.15, "ratio_ci_upper_le": 1.25, "coverage_ge": "1-delta", "false_alarm_le": "delta"},
                "HF2": {"detect_ge": 0.60, "fcr_le": "2*delta", "placebo_le": "1/3 of true labels",
                        "kappa0_false_alarm_le": "delta"},
                "HF2b": {"fcr_cp_upper_le": "2*delta", "alarm_steps_le": "AGC/100", "eta": "2*eps"},
                "HF3": {"detect_le": "2*delta"},
                "HF4": {"fcr_le": "2*delta", "cost_le": "AGC/10"},
                "HD1": {"kappa_high_fcr_cp_lower_gt": "delta", "twin_fcr_le": "delta", "monotone": "Jonckheere"},
                "HD1b": {"static_fcr_ge": 0.8, "dynamic_fcr_le": "delta", "n": 2000},
                "HD2": {"sandwich": 1.0, "beta_slope_in": [0.85, 1.15], "kappa_eff_slope_in": [1.5, 2.5],
                        "omega_H2_slope_in": [1.5, 2.5]},
                "HD3": {"paired_D_ci_lower_gt": 0},
                "HS1": {"tost_90ci_in": [0.8, 1.25]},
                "HH1a": {"cp_upper_le": "delta"}, "HH1b": {"oos_detect_ge": 0.9, "near_tie": "|V*-eps| < 0.2 eps"},
                "HK": {"K_star_le": 15},
                "suspicious_gate": {"savings_gt_x": 5, "over_simple_baseline_gt": 0.30},
            },
            "G0_outcome": {"passed": g0_passed, "go_no_go": g0["go_no_go"], "violations": gate["violations_total"],
                           "draws": gate["draws_total"], "eta_loc_over_eps_median_by_f": eta_by_f,
                           "best_emp_lower_bound_over_eps": gate["by_eps"]["0.02"]["best_emp_lower_bound"] / EPS,
                           "implication": "tier C is already implied by hypotheses.md ('或 G0 失败')"},
            "f_star": f_star,
            "f_star_rule": f_rule,
            "hr4_fine_f": 4 if f_star == 2 else 2,
            "R2_in_family_A": R2_family_A,
            "R2_rule": "R2 in family A iff share of dev problems with h <= h*/2 (literal h*) > 0.5 (G0, f=1); "
                       "otherwise R2 = 48-instance diagnostic only (methodology 6.1)",
            "R3_h_le_hstar_half": R3_hcond,
            "R3_note": ("R3 premise h <= h*/2 also fails on dev (G0, f=2); R3 stays in family A as planned "
                        "(methodology only removes R2), the failed premise is reported with every R3 number"),
            "h_ratio_dev": ratio_variants,
            "R2_scope": r2_scope,
            "eta_loc_method": {
                "definition": "eta_loc(g;q) >= max_pi sup_{theta in C(g)} |J_theta(pi) - J_g(pi)|",
                "computation": "sum_k |dJ_g/dv_k| w_k(g) + 1/2 w(g)^T Hbar w(g); exact autograd gradient at every "
                               "grid point (float64); Hbar = numerical max |central-difference Hessian| over sampled "
                               "box vertices / interior / f=1 grid points x safety 1.5; w = per-coordinate clipped "
                               "Voronoi half-widths; normalised by public class-max c_q of G_1",
                "module": "dsswm/certify/eta_loc.py", "data_free": True,
                "validity_check": f"0/{gate['draws_total']} violations (G0) + "
                                  f"{eta_chk['1']['violations']}/{eta_chk['1']['draws']} on d_eff dev problems",
                "certify_rule": "R_grid(Theta_t) + 2 max_{g alive} eta_loc(g) <= eps (dsswm/certify/minimax_infl.py)",
                "applies_to": "all grid methods (JPC_infl, B3g_infl, B12_infl, B8_infl) at R2/R3; B3, B3-UI not inflated"},
            "h_star_method": {
                "primary": "literal: h*(q) = eps / (L_J(q) sqrt(cond(I_xi))), I_xi = realised design Fisher (all rounds "
                           "incl. n0) at the grid MLE after the JPC/DDA run, cond = lmax / max(lmin, 1e-12 lmax), "
                           "L_J = normalised Lipschitz bound from eta_loc gradient parts",
                "secondary_reported": ["identified subspace: cond over eigenvalues >= 1e-6 lmax",
                                       "cond = 1 (most favourable)"],
                "h": "median Euclidean half-diagonal of G_f cells (raw coordinates)",
                "note": "literal h* is ~7e-9 because I_xi has ~1 rank-deficient direction (11/12 identified); HR4 is "
                        "evaluated on the primary definition (threshold unchanged), the secondary ones are descriptive"},
            "d_eff": {
                "definition": "participation ratio tr(I)^2 / tr(I^2) of the realised design Fisher at the grid MLE "
                              "(dsswm/evidence/lr_set.participation_ratio), raw parameter coordinates",
                "locked_value": deff_locked,
                "locked_rule": "median over the 100 fresh dev problems (seeds 640-649 x 10, f=1) at R0 = HR2 primary level",
                "dev_estimates": deff,
                "sensitivity_set": [6, 12, 24],
                "sensitivity_note": ("pre-registered {6,12,24} kept unchanged; measured d_eff (~3) lies below it, so the "
                                     "chi2 cell with the measured value is the primary and {6,12,24} is reported as the "
                                     "pre-registered sensitivity band"),
            },
            "budgets": {"T_max_stepwise": 3000, "T_max_whole_trial": 2310000, "rmst_tau": 3000,
                        "censoring": "censored at T_max lower bound; refusal = censored, charged T_max",
                        "bootstrap_B": 10000, "bootstrap_seed": 42},
            "tuning": tuning,
            "eval_manifest": man,
            "eval_manifest_sha256": man_sha,
            "B3_UI_in_denominator": bool(sof.get("b3_ui", {}).get("implemented", False)),
            "B3_UI_note": "implemented and unit-tested in r2_setup_offgrid -> enters the denominator candidates and is "
                          "the same-threshold control of tier A",
            "denominator_rule": "per R-level: drop methods in {B3, B3g, B3-UI, B8, B12} with FCR CP upper > 2*delta, "
                                "take min total stream steps; B10, B11, uniform never denominators",
            "holm_families": {"A": famA, "R": ["HR2", "HR2b", "HR3", "HR4", "HR5"],
                              "F": ["HF1", "HF2", "HF2b", "HF3", "HF4"], "D": ["HD1", "HD1b", "HD2", "HD3"],
                              "A_test": "H0: stream ratio >= 0.8, one-sided cluster-bootstrap p, Holm within family"},
            "tier_decision": {"module": "dsswm/stats/tier_decision.py", "function": "decide_tier",
                              "code_sha256": sha_file(tier_src), "code": tier_src.read_text(),
                              "dev_implied_tier": tier_now},
            "downscale_decisions": downscale,
            "setup_gates": {"r2_setup_offgrid": {"go_no_go": sof["go_no_go"],
                                                 "unit_tests_passed": sof["unit_tests"].get("passed_after_eta_diff_test")},
                            "r2_setup_falsify": {"go_no_go": sfa["go_no_go"], "gate": sfa["gate"]},
                            "dev_offgrid_ladder": {"go_no_go": lad["go_no_go"], "crashes": lad["crashes"],
                                                   "jpc_infl_completion": lad["pass_criteria"]["jpc_infl_completion_by_arm"]}},
            "code_sha256": code_hash,
            "round0": round0,
        }
        nulls = [k for k, v in lock.items() if v is None and k != "R2_scope"]
        if R2_family_A is False:
            assert lock["R2_scope"] is not None
        assert not nulls, f"null top-level fields: {nulls}"
        from dsswm.stats.prereg import canonical_hash
        if not pilot:
            lock["sha256"] = canonical_hash(lock)
        else:
            lock["sha256_provisional"] = canonical_hash(lock)
        LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False, default=float))
        log(f"wrote {LOCK.relative_to(WS)} version 2 status={lock['status']}")

        # ---------------------------------------------------------------- verify
        progress(4, 5, "verify")
        back = json.loads(LOCK.read_text())
        verbatim_ok = all(ln in hyp_text for ln in back["thresholds_verbatim"]["rows"].values()) and \
            all(ln in hyp_text for ln in back["thresholds_verbatim"]["tier_wording"])
        assert back["round0"] == round0
        try:
            from dsswm.stats.prereg import assert_locked
            assert_locked(LOCK)
            assert_refuses = False
        except RuntimeError as e:
            assert_refuses = str(e)
        summary = {
            "task_id": TASK, "mode": args.mode, "seed": 42, "eval_seeds_touched": False, "concurrent_run": True,
            "lock_status": back["status"], "lock_sha256": back.get("sha256") or back.get("sha256_provisional"),
            "lock_sha256_kind": "locked" if not pilot else "provisional (not binding)",
            "lock_file_sha256": sha_file(LOCK),
            "f_star": f_star, "R2_in_family_A": R2_family_A, "R3_h_le_hstar_half": R3_hcond,
            "family_A": famA, "d_eff": deff_locked, "d_eff_dev": deff, "g0_passed": g0_passed,
            "dev_implied_tier": tier_now, "B3_UI_in_denominator": lock["B3_UI_in_denominator"],
            "downscale_decisions": downscale, "eval_manifest_sha256": man_sha,
            "n_eval_seeds_in_manifest": len(man["all_eval_seeds"]),
            "thresholds_verbatim_ok": verbatim_ok, "n_threshold_rows": len(back["thresholds_verbatim"]["rows"]),
            "assert_locked_refuses_provisional": assert_refuses,
            "top_level_null_fields": [k for k, v in back.items() if v is None],
            "pass_criteria": {"status_provisional": back["status"] == ("provisional" if pilot else "locked"),
                              "every_field_non_null": not [k for k, v in back.items() if v is None],
                              "thresholds_byte_identical": verbatim_ok,
                              "n_deff_problems_ge_100": deff["R0"]["n"] >= 100},
            "wall_clock_s": time.time() - T0,
        }
        summary["go_no_go"] = "GO" if all(summary["pass_criteria"].values()) else "NO_GO"
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
        log(f"done in {summary['wall_clock_s']:.0f}s: {summary['go_no_go']} f*={f_star} R2_in_A={R2_family_A} "
            f"d_eff={deff_locked} tier(dev)={tier_now['tier']}")
        progress(5, 5, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: lock v2 {back['status']}, f*={f_star}, R2_in_A={R2_family_A}, "
                             f"d_eff={deff_locked}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        (out_dir / "summary.json").write_text(json.dumps({"task_id": TASK, "mode": args.mode, "error": repr(e)}))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
