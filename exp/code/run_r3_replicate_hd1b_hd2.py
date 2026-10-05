"""r3_replicate_hd1b_hd2: replication of HD1b (same-exposure construction) and HD2 (kappa_eff sandwich, E1-Lin).

Runs the UNCHANGED round-2 science code (run_hd1b_same_exposure.py, run_hd2_kappa_sandwich_lin.py); the only change
is the seed source (and output location), passed through their round-3 CLI overrides.

Pilot (smoke, dev seeds, no eval seed touched):
  HD1b  dev 734-737 x 3 designs (uniform / dda / onpolicy), T-Lin noise 42 + T-NL stream 0
  HD2   dev 734-735 x H in {1,4} x 4 problems (noise 42) + construction dev 734-735 x H in {2,4} (path smoke)
  pass: 0 crashes AND HD1b static FCR >= 0.8 on every (family, design) cell AND HD2 H=1 N_time == N_joint exactly
Full (requires the v3 lock with status 'locked'; seeds read from the lock's eval_manifest / seed_blocks):
  HD1b  eval 10600-10629 x 3 designs x 3 noise seeds (Lin 42/123/456, NL streams 0/1/2)
  HD2   eval 10630-10669 reduced: main H in {1,4,8} x 4 problems x noise 42/123/456; construction 10630-10649
        x H in {2,4,8,16}
  pass: HD1b static >= 0.9 and dynamic 0; HD2 kappa_eff slopes in [1.5,2.5], H=1 exact 0

Usage: run_r3_replicate_hd1b_hd2.py --mode {pilot,full} [--workers 4]
Concurrent run (4 slots share 20 cores + 1 RTX 4090): timings are "concurrent".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "r3_replicate_hd1b_hd2"
PY = sys.executable


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


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
                                                       "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def block(spec):
    a, b = (int(x) for x in str(spec).split("-"))
    return a, b


def seed_plan(mode):
    if mode == "pilot":
        return {"hd1b": {"seeds": "734-737", "noise": "42", "streams": "0"},
                "hd2": {"seeds": "734-735", "cons": "734-735", "hs": "1,4", "cons_hs": "2,4", "noise": "42"},
                "source": "task_plan pilot.dev_seeds 734-737 (dev block 720-760)"}
    lock = json.loads(LOCK.read_text())
    assert lock.get("version") == 3 and lock.get("status") == "locked", "full mode requires the locked v3 prereg"
    man = lock.get("eval_manifest") or {}
    blocks = dict(lock["seed_blocks"]["eval"])
    if isinstance(man, dict):                                   # manifest entries override the block summary
        for k in ("hd1b", "hd2_hh1"):
            v = man.get(k)
            if isinstance(v, str) and "-" in v:
                blocks[k] = v
    a1, b1 = block(blocks["hd1b"])
    a2, b2 = block(blocks["hd2_hh1"])
    assert (a1, b1) == (10600, 10629) and (a2, b2) == (10630, 10669), (blocks["hd1b"], blocks["hd2_hh1"])
    return {"hd1b": {"seeds": f"{a1}-{b1}", "noise": "42,123,456", "streams": "0,1,2"},
            "hd2": {"seeds": f"{a2}-{b2}", "cons": f"{a2}-{a2 + 19}", "hs": "1,4,8", "cons_hs": "2,4,8,16",
                    "noise": "42,123,456"},
            "source": "plan/prereg_lock.json v3 (locked) seed_blocks.eval / eval_manifest",
            "lock_sha256": lock.get("sha256")}


def run(cmd, logf):
    t0 = time.time()
    with open(logf, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
    return rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    out = RES_ROOT / ("pilots" if pilot else "full") / TASK
    out.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    (out / "start_time.txt").write_text(datetime.now().isoformat())
    T0 = time.time()
    summ = {"task_id": TASK, "mode": args.mode, "seed": 42, "concurrent_run": True,
            "note": "并发运行（4 槽并行，每任务 4 worker）；计时偏高", "eval_seeds_touched": not pilot}
    try:
        sp = seed_plan(args.mode)
        summ["seed_plan"] = sp
        progress(1, 3, "hd1b", {"seeds": sp["hd1b"]["seeds"]})
        h = sp["hd1b"]
        rc1, t1 = run([PY, "run_hd1b_same_exposure.py", "--mode", args.mode, "--workers", str(args.workers),
                       "--seeds", h["seeds"], "--noise-seeds", h["noise"], "--nl-streams", h["streams"],
                       "--out-dir", str(out / "hd1b"), "--no-protocol"], out / "hd1b_stdout.log")
        summ["hd1b_rc"], summ["hd1b_wall_s"] = rc1, t1
        progress(2, 3, "hd2", {"seeds": sp["hd2"]["seeds"], "hd1b_rc": rc1})
        h = sp["hd2"]
        rc2, t2 = run([PY, "run_hd2_kappa_sandwich_lin.py", "--mode", args.mode, "--workers", str(args.workers),
                       "--seeds", h["seeds"], "--cons-seeds", h["cons"], "--hs", h["hs"], "--cons-hs", h["cons_hs"],
                       "--noise-seeds", h["noise"], "--out-dir", str(out / "hd2"), "--no-protocol"],
                      out / "hd2_stdout.log")
        summ["hd2_rc"], summ["hd2_wall_s"] = rc2, t2
        s1 = json.loads((out / "hd1b" / "summary.json").read_text()) if (out / "hd1b" / "summary.json").exists() else {}
        s2 = json.loads((out / "hd2" / "summary.json").read_text()) if (out / "hd2" / "summary.json").exists() else {}
        sf = s1.get("static_fcr_by_design", {})
        dfp = s1.get("dynamic_fcr_pooled", {})
        g2 = s2.get("gate", {})
        summ["hd1b"] = {"static_fcr_by_design": {k: v.get("fcr") for k, v in sf.items()},
                        "static_fcr_cp95": {k: v.get("cp95") for k, v in sf.items()},
                        "dynamic_fcr_pooled": dfp, "pass_criteria_round2": s1.get("pass_criteria"),
                        "go_no_go_round2_rule": s1.get("go_no_go"), "crashes": s1.get("crashes"),
                        "n_runs": s1.get("n_runs"), "n_rows": s1.get("n_rows"),
                        "nl_construction_infeasible": s1.get("nl_construction_infeasible"),
                        "lin_ds_max_abs": s1.get("lin_ds_max_abs"), "fwl": s1.get("fwl"),
                        "timing_projection": s1.get("timing_projection"), "error": s1.get("error")}
        summ["hd2"] = {"gate": g2, "h1_exact_zero": s2.get("h1_exact_zero"), "sandwich_violations":
                       (s2.get("sandwich") or {}).get("violations"), "n_errors": s2.get("n_errors"),
                       "n_sequential_runs": s2.get("n_sequential_runs"), "n_problem_records": s2.get("n_problem_records"),
                       "validity": s2.get("validity"), "construction_fits": (s2.get("construction") or {}).get("fits"),
                       "suspicious_flags": s2.get("suspicious_flags")}
        thr = 0.8 if pilot else 0.9
        static_vals = [v for v in summ["hd1b"]["static_fcr_by_design"].values()]
        crashes = (rc1 != 0) + (rc2 != 0) + int(s1.get("crashes") or 0) + int(s2.get("n_errors") or 0)
        pc = {"zero_crashes": crashes == 0,
              f"hd1b_static_fcr_ge_{thr}_all_cells": bool(static_vals) and all(v is not None and v >= thr
                                                                                for v in static_vals),
              "hd1b_dynamic_fcr_zero" if not pilot else "hd1b_dynamic_fcr_le_delta":
                  (dfp.get("false") == 0) if not pilot else (dfp.get("fcr") is not None and dfp["fcr"] <= 0.05),
              "hd2_h1_exact_zero": bool((s2.get("h1_exact_zero") or {}).get("pass")),
              "hd2_sandwich_100pct": bool(g2.get("sandwich_100pct"))}
        if not pilot:
            ke = list((g2.get("kappa_eff_slope_by_partition") or {}).values()) + [g2.get("construction_slope_time")]
            pc["hd2_slopes_in_[1.5,2.5]"] = all(v is not None and 1.5 <= v <= 2.5 for v in ke)
        summ["n_problem_method_runs"] = int(s1.get("n_runs") or 0) + int(s2.get("n_sequential_runs") or 0)
        summ["pass_criteria"] = pc
        summ["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summ["wall_clock_s"] = time.time() - T0
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        progress(3, 3, "done", {"go_no_go": summ["go_no_go"]})
        mark_done("success" if crashes == 0 else "failed", f"{args.mode}: {summ['go_no_go']} {pc}")
        print(json.dumps({"go_no_go": summ["go_no_go"], "pass_criteria": pc}, indent=1))
    except Exception as e:  # noqa: BLE001
        summ["error"] = repr(e)
        summ["traceback"] = traceback.format_exc()
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
