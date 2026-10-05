"""r5_baseline_qualification (PILOT): new rigorous rivals B4-bal / Hait-SW / Molitor-WoR + r5 registry.

Steps
  1. unit tests: test_r5_rivals (new), test_baselines_r4a + test_no_truth_import (r4 regression);
  2. toy FWER: 200 streams per configuration on the r4 near-tie synthetic finite-pool populations
     (test_baselines_r4a._toy_population, dev seeds 900-919 x permutation seeds 0-9, eps = 0.02, 5 problems,
     stop 4/5; adaptive pool exhaustion included). Gate per new rigorous rival: 0/200 false streams (one-sided CP
     upper 0.0149) for EVERY tuning-grid point (B4-bal share x3, Hait-SW p_min x gamma = 6, Molitor-WoR x1).
     Reference rows: B4 (r4 rigorous) and NAIVE-control (plug-in GLR, beta = ln(1/delta): harness power check --
     a broken certificate must show up as false streams);
  3. billing consistency on every toy and timing stream;
  4. CR9 development-half timing smoke on dev seeds 900-903 (eps* = 0.001, 15 problems, K = 20): ONLY seconds per
     stream and billing are recorded; N80 / certification outcomes are deliberately not written (no comparison).
CPU only; <= 4 worker processes; BLAS threads pinned to 1; timings are from a concurrent run (other r5 tasks share
the 20-core host).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
TASK = "r5_baseline_qualification"
PY = __import__('sys').executable

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream  # noqa: E402

FILES = ["dsswm/baselines/b4_bal.py", "dsswm/baselines/hait_sw.py", "dsswm/baselines/molitor_wor.py",
         "dsswm/streams/r5_registry.py", "dsswm/tests/test_r5_rivals.py", "run_r5_baseline_qualification.py"]
NEW_RIVALS = ("B4-bal", "Hait-SW", "Molitor-WoR")
TIMING_SEEDS = (900, 901, 902, 903)
CR_EPS = 0.001


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def progress(rd, step, total, metric=None):
    (rd / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def run_pytest(args, timeout=2400):
    t = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], cwd=CODE, capture_output=True,
                       text=True, timeout=timeout)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}


def configs():
    out = []
    for kw in reg.TUNING_GRID["B4-bal"]:
        out.append((f"B4-bal[share={kw['share']:.2f}]", "B4-bal", kw))
    for kw in reg.TUNING_GRID["Hait-SW"]:
        out.append((f"Hait-SW[p_min={kw['p_min']:.2f},gamma={'2/3' if kw['gamma'] < 0.9 else '1'}]", "Hait-SW", kw))
    out.append(("Molitor-WoR", "Molitor-WoR", {}))
    out.append(("B4 (r4 ref)", "B4", {}))
    out.append(("NAIVE-control", None, {}))
    return out


def _mk(base, kw):
    if base is None:
        from dsswm.baselines.combgame_joint import CombGameJoint
        return CombGameJoint("fav", beta_override=math.log(1 / 0.05), name="NAIVE-control")
    return reg.make_method(base, **kw)


def _toy_job(args):
    label, base, kw, sd = args
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    out = []
    for p in range(10):
        m = _mk(base, kw)
        s, _ = run_stream(env, m, 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
        out.append({"config": label, "method": m.name, "validity": m.validity, "population_seed": sd,
                    "perm_seed": s["perm_seed"], "fwer_event": s["fwer_event"], "n_cert": s["n_cert"],
                    "n_false": s["n_false"], "reached_stop": s["reached_stop"], "N80_over_tau": s["N80"] / env.tau_R,
                    "billing_ok": s["billing_ok"], "skipped": s["skipped"], "reselected": s["reselected"],
                    "sec": s["sec_total"]})
    return out


_ENV = {}


def _timing_job(args):
    name, sd = args
    env, problems = _ENV["env"], _ENV["problems"]
    ctx = build_ctx(env, problems, CR_EPS)
    m = reg.make_method(name)
    s, rows = run_stream(env, m, sd, problems, CR_EPS, ctx=ctx, J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=False)
    # timing only: certification outcomes / N80 are not returned (no method comparison in this task)
    return {"method": name, "dev_seed": sd, "sec_total": s["sec_total"], "sec_plan": s["sec_plan"],
            "sec_cert": s["sec_cert"], "billing_ok": s["billing_ok"], "concurrent_run": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot")
    ap.add_argument("--skip-timing", action="store_true")
    a = ap.parse_args()
    rd = WS / "exp" / "results"
    out = rd / ("pilots" if a.mode == "pilot" else "full") / TASK
    out.mkdir(parents=True, exist_ok=True)
    (rd / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "started_at": started,
               "timing_note": "concurrent run (<=4 workers; 3 other r5 tasks share the 20-core host)"}
    progress(rd, 0, 4, {"stage": "unit tests"})
    summary["tests"] = {"r5_rivals": run_pytest(["dsswm/tests/test_r5_rivals.py"]),
                        "baselines_r4a": run_pytest(["dsswm/tests/test_baselines_r4a.py"]),
                        "no_truth_import": run_pytest(["dsswm/tests/test_no_truth_import.py"])}
    tests_ok = all(v["returncode"] == 0 for v in summary["tests"].values())
    progress(rd, 1, 4, {"stage": "toy FWER", "tests_ok": tests_ok})

    # ------------------------------------------------------------------ toy FWER
    t0 = time.time()
    cfgs = configs()
    jobs = [(lab, base, kw, sd) for lab, base, kw in cfgs for sd in range(900, 920)]
    with get_context("fork").Pool(4) as pool:
        res = [r for chunk in pool.map(_toy_job, jobs, chunksize=1) for r in chunk]
    with open(out / "toy_fwer_streams.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    toy, csv_rows = {}, []
    for lab, base, _ in cfgs:
        rr = [r for r in res if r["config"] == lab]
        ev = sum(r["fwer_event"] for r in rr)
        row = {"config": lab, "method": rr[0]["method"], "validity": rr[0]["validity"], "streams": len(rr),
               "false_streams": ev, "fwer_cp_upper": round(cp_upper(ev, len(rr)), 4),
               "certs_total": sum(r["n_cert"] for r in rr), "false_certs": sum(r["n_false"] for r in rr),
               "reached_stop": sum(r["reached_stop"] for r in rr),
               "median_N80_over_tau": round(float(np.median([r["N80_over_tau"] for r in rr])), 4),
               "streams_with_exhaustion": sum(r["skipped"] + r["reselected"] > 0 for r in rr),
               "billing_mismatch": sum(not r["billing_ok"] for r in rr),
               "gate_0_of_200": (ev == 0) if base in NEW_RIVALS else None}
        toy[lab] = row
        csv_rows.append(row)
    with open(out / "toy_fwer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    new_rows = [r for r in csv_rows if r["gate_0_of_200"] is not None]
    summary["toy_fwer"] = {"eps": 0.02, "delta": 0.05, "problems": 5, "stop_k": 4,
                           "populations": "r4 near-tie toy (_toy_population) dev seeds 900-919 x perm 0-9",
                           "sec": round(time.time() - t0, 1), "by_config": toy,
                           "all_new_rivals_0_of_200": all(r["gate_0_of_200"] for r in new_rows),
                           "cp_upper_0_of_200": round(cp_upper(0, 200), 4),
                           "naive_control_detects": toy["NAIVE-control"]["false_streams"] > 0}
    billing_ok = all(r["billing_mismatch"] == 0 for r in csv_rows)
    progress(rd, 2, 4, {"stage": "CR9 timing", "toy_gate": summary["toy_fwer"]["all_new_rivals_0_of_200"]})

    # ------------------------------------------------------------------ CR9 timing smoke (dev 900-903)
    timing = []
    if not a.skip_timing:
        from dsswm.envs.pool_replay import PoolReplayEnv
        from dsswm.streams.frontier_runner import true_policy_values
        t0 = time.time()
        env = PoolReplayEnv("CR9", "dev")
        problems = fr.cr_problems()
        ctx = build_ctx(env, problems, CR_EPS)
        J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
        _ENV.update(env=env, problems=problems, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]))
        sec_env = round(time.time() - t0, 1)
        jobs = [(n, sd) for n in NEW_RIVALS for sd in TIMING_SEEDS]
        with get_context("fork").Pool(4) as pool:
            timing = pool.map(_timing_job, jobs, chunksize=1)
        with open(out / "timing.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(timing[0]))
            w.writeheader()
            w.writerows(timing)
        summary["timing"] = {"layer": "CR9", "half": "dev", "eps": CR_EPS, "K": 20, "seeds": list(TIMING_SEEDS),
                             "sec_env_build": sec_env, "tau_R": int(env.tau_R), "replan": int(env.replan_interval),
                             "by_method": {n: {"sec_per_stream_mean": round(float(np.mean(
                                 [r["sec_total"] for r in timing if r["method"] == n])), 2),
                                 "sec_per_stream_max": round(float(np.max(
                                     [r["sec_total"] for r in timing if r["method"] == n])), 2),
                                 "billing_ok": all(r["billing_ok"] for r in timing if r["method"] == n)}
                                 for n in NEW_RIVALS},
                             "note": "seconds only; N80 / certification outcomes / checkpoint reached / plan counts intentionally not "
                                     "recorded (they reveal stopping times)"}
        billing_ok &= all(r["billing_ok"] for r in timing)
    progress(rd, 3, 4, {"stage": "summary"})

    qual = WS / "plan" / "baseline_qualification_r5.md"
    summary["qualification_table"] = {"path": "plan/baseline_qualification_r5.md", "exists": qual.exists()}
    summary["billing_ok"] = bool(billing_ok)
    summary["unit_tests_pass"] = bool(tests_ok)
    summary["registry"] = {"rigorous_set_R": list(reg.RIGOROUS_SET_R),
                           "all_names": list(reg.REGISTRY), "tuning_grid": {k: v for k, v in reg.TUNING_GRID.items()}}
    summary["file_sha256"] = {p: sha(CODE / p) for p in FILES}
    try:
        summary["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True,
                                             text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        summary["git_head"] = None
    gate = bool(tests_ok and summary["toy_fwer"]["all_new_rivals_0_of_200"] and billing_ok and qual.exists())
    summary["verdict"] = "GO" if gate else "NO_GO"
    summary["finished_at"] = datetime.now().isoformat()
    summary["wall_min"] = round((time.time() - t_start) / 60, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    progress(rd, 4, 4, {"verdict": summary["verdict"]})
    print(json.dumps({"verdict": summary["verdict"], "tests": summary["tests"],
                      "toy": {k: (v["false_streams"], v["streams"], v["median_N80_over_tau"]) for k, v in toy.items()},
                      "timing": summary.get("timing", {}).get("by_method"), "billing_ok": billing_ok}, indent=1))


if __name__ == "__main__":
    main()
