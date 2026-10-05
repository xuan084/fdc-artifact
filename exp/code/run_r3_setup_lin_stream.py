"""r3_setup_lin_stream: setup + pilot for the round-3 E1-Lin layer.

New modules: dsswm/streams/lin_stream.py (15-problem gap-quota stream at eps_Lin = 0.05, 30% (s0, Pi)-sharing twins,
cos(D_q, span(prev)), Lin-Static twin stream, cell runner + sibling platform), dsswm/baselines/lin_rage.py (JPC-Lin
= EllipsoidSet + certify_lin + DDA; RAGE extracted from run_hd2_kappa_sandwich_lin; XY-static; G-opt; B1eb whole-trial
LUCB with (s0, Pi) trial reuse), all through the five-level reuse switch; dsswm/tests/test_lin_stream.py.

pilot: (1) full unit-test suite; (2) quota / shared fraction on dev 740-749 (3 streams each); (3) ellipsoid closed form
vs enumerated sup on dev 740-741 x 3 streams (every pairwise candidate difference); (4) smoke on dev 740-741 stream 0:
{JPC-Lin, RAGE} x {off, vol, ev1, full}, XY-static x off, G-opt x full, B1eb x full (8 problems each), Lin-Static JPC-Lin
x {off, ev1, full} (4 problems), orth hook (no provider) JPC-Lin x 4 problems; T_max_Lin = 20000.
Pass: tests pass AND closed-form error <= 1e-9 AND billing mismatch 0 (and no crash).
full: rerun the unit tests and write the code sha256 for the lock.
Usage: run_r3_setup_lin_stream.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
TASK = "r3_setup_lin_stream"
PY = sys.executable

CODE_FILES = ["dsswm/streams/lin_stream.py", "dsswm/baselines/lin_rage.py", "dsswm/tests/test_lin_stream.py",
              "dsswm/tests/test_no_truth_import.py", "run_r3_setup_lin_stream.py"]
QUOTA_SEEDS = list(range(740, 750))
SMOKE_SEEDS = [740, 741]
N_SMOKE = 8
N_STATIC = 4
T_MAX = 20000
CELLS = [("JPC-Lin", ("off", "vol", "ev1", "full")), ("RAGE", ("off", "vol", "ev1", "full")), ("XY-static", ("off",)),
         ("G-opt", ("full",)), ("B1eb", ("full",))]


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


def run_tests(out_dir):
    t0 = time.perf_counter()
    xml = out_dir / "pytest_junit.xml"
    r = subprocess.run([PY, "-m", "pytest", "-q", "dsswm/tests", f"--junitxml={xml}"], cwd=HERE, capture_output=True,
                       text=True, env={**os.environ, "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0"})
    (out_dir / "pytest_stdout.txt").write_text(r.stdout[-20000:] + "\n" + r.stderr[-5000:])
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    ts = root if root.tag == "testsuite" else root.find("testsuite")
    n, f, e, s = (int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    new = sum(1 for tc in ts.iter("testcase") if "test_lin_stream" in (tc.get("classname") or ""))
    return {"returncode": r.returncode, "tests": n, "failures": f, "errors": e, "skipped": s,
            "passed": n - f - e - s, "new_tests_in_test_lin_stream": new, "sec": time.perf_counter() - t0,
            "all_passed": r.returncode == 0 and f == 0 and e == 0}


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


# --------------------------------------------------------------------------------------------- workers
def run_job(seed, static, method, arms, n_problems, tmax, orth_hook=False):
    from dsswm.streams.lin_stream import LinStreamBuilder, run_lin_cell
    b = LinStreamBuilder(seed)
    rows, samples, errs, cache = [], [], [], {}
    for arm in arms:
        t0 = time.perf_counter()
        try:
            rr = run_lin_cell(b, 0, method, arm, tmax=tmax, static=static, n_problems=n_problems,
                              sibling_cache=cache, samples=samples, orth_provider=None)
            for r in rr:
                r["cell_sec"] = time.perf_counter() - t0
                r["orth_hook_smoke"] = orth_hook
            rows += rr
        except Exception as e:  # noqa: BLE001 - crashes are counted, not hidden
            errs.append({"seed": seed, "static": static, "method": method, "arm": arm, "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
    return rows, samples, errs


def quota_check():
    from dsswm.streams.gap_quota import fill_instances
    from dsswm.streams.lin_stream import LinStreamBuilder
    recs, ok = [], {}
    for s in QUOTA_SEEDS:
        t0 = time.perf_counter()
        b = LinStreamBuilder(s)
        sel = b.select()
        rec = {"seed": s, "quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"],
               "n_fresh_draws": sel["n_fresh_draws"], "n_twin_tries": sel["n_twin_tries"], "rejected": sel["rejected"],
               "layer_counts": sel["layer_counts"], "kind_counts": sel["kind_counts"], "n_shared_target": b.n_shared,
               "sec": time.perf_counter() - t0}
        ok[s] = not sel["quota_fail"]
        if not sel["quota_fail"]:
            per = []
            for t in range(3):
                st = b.stream(t)
                per.append({"stream": t, "shared_frac": sum(m["shared"] for m in st.harness) / len(st.harness),
                            "cos_span_mean": float(np.mean([m["cos_span"] for m in st.harness[1:]])),
                            "cos_max_mean": float(np.mean([m["cos_max"] for m in st.harness[1:]])),
                            "cos_span_shared_mean": float(np.mean([m["cos_span"] for m in st.harness[1:]
                                                                   if m["shared"]] or [np.nan])),
                            "cos_span_fresh_mean": float(np.mean([m["cos_span"] for m in st.harness[1:]
                                                                  if not m["shared"]] or [np.nan])),
                            "H": [q.H for q in st.problems], "n_policies": [len(q.policies) for q in st.problems],
                            "twin_variants": dict(Counter(m["twin_variant"] for m in st.harness if m["twin_variant"]))})
                if t == 0:
                    sts = b.stream(0, static=True, same_problems=True)
                    rec["static_layers_same_problems"] = dict(Counter(m["gap_layer"] for m in sts.harness))
            rec["streams"] = per
        tq = time.perf_counter()
        ss = b.select(True)
        rec["static_quota"] = {"quota_fail": ss["quota_fail"], "n_draws": ss["n_draws"],
                               "layer_counts": ss["layer_counts"], "kind_counts": ss["kind_counts"],
                               "shared_frac_stream0": None if ss["quota_fail"] else b.shared_fraction(0, True),
                               "sec": time.perf_counter() - tq}
        recs.append(rec)
    seeds_ok, repl = fill_instances(QUOTA_SEEDS, lambda s: ok[s] if s in ok else
                                    not LinStreamBuilder(s).select()["quota_fail"], reserve_start=750)
    fr = [p["shared_frac"] for r in recs if "streams" in r for p in r["streams"]]
    static_fill = float(np.mean([not r["static_quota"]["quota_fail"] for r in recs]))
    static_same_nontie = [sum(v for k, v in r["static_layers_same_problems"].items() if k != "tie")
                          for r in recs if "static_layers_same_problems" in r]
    return recs, {"seeds": QUOTA_SEEDS, "instance_fill_rate": float(np.mean(list(ok.values()))),
                  "replacement_log": repl, "seeds_after_replacement": seeds_ok, "eps_Lin": 0.05, "per_layer": 5,
                  "max_draws": 4000, "shared_problem_frac_mean": float(np.mean(fr)),
                  "shared_problem_frac_min": float(min(fr)), "shared_problem_frac_max": float(max(fr)),
                  "shared_frac_within_0.30pm0.05": bool(all(abs(f - 0.30) <= 0.05 for f in fr)
                                                         and abs(np.mean(fr) - 0.30) <= 0.05),
                  "draws_median": float(np.median([r["n_draws"] for r in recs])),
                  "static_quota_fill_rate": static_fill,
                  "static_same_problems_nontie_per_stream": static_same_nontie,
                  "sec_total": float(sum(r["sec"] for r in recs)),
                  "cos_span_mean": float(np.nanmean([p["cos_span_mean"] for r in recs if "streams" in r
                                                     for p in r["streams"]])),
                  "cos_span_shared_mean": float(np.nanmean([p["cos_span_shared_mean"] for r in recs if "streams" in r
                                                            for p in r["streams"]])),
                  "cos_span_fresh_mean": float(np.nanmean([p["cos_span_fresh_mean"] for r in recs if "streams" in r
                                                           for p in r["streams"]]))}


def closed_form_check():
    from dsswm.streams.lin_stream import LinStreamBuilder
    from dsswm.tests.test_lin_stream import closed_form_error
    out = []
    for s in SMOKE_SEEDS:
        b = LinStreamBuilder(s)
        for t in range(3):
            out.append({"seed": s, "stream": t, "err": closed_form_error(b, t, n_extra=200, seed=t)})
    return out


# --------------------------------------------------------------------------------------------- pilot
def pilot(out_dir, workers):
    from joblib import Parallel, delayed
    summary = {"task": TASK, "mode": "pilot", "started_at": datetime.now().isoformat(),
               "note": "并发运行（与 r3_p1 / r3_p3 / r3_p5 共享 20 核 CPU）；计时偏高。CPU only。"}
    progress(1, 5, "unit_tests")
    summary["unit_tests"] = run_tests(out_dir)
    print("tests", summary["unit_tests"], flush=True)

    progress(2, 5, "quota_fill")
    recs, q = quota_check()
    with open(out_dir / "quota_fill.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")
    summary["quota"] = q
    print("quota", json.dumps(q, default=str)[:600], flush=True)

    progress(3, 5, "closed_form")
    cf = closed_form_check()
    summary["closed_form"] = {"cases": cf, "max_err": max(c["err"] for c in cf)}
    print("closed form", summary["closed_form"]["max_err"], flush=True)

    progress(4, 5, "smoke")
    jobs = [(s, False, m, arms, N_SMOKE, T_MAX) for s in SMOKE_SEEDS for m, arms in CELLS]
    jobs += [(s, True, "JPC-Lin", ("off", "ev1", "full"), N_STATIC, T_MAX) for s in SMOKE_SEEDS]
    jobs += [(740, False, "JPC-Lin", ("orth",), 4, T_MAX, True)]
    t0 = time.perf_counter()
    res = Parallel(n_jobs=workers, backend="loky")(delayed(run_job)(*j) for j in jobs)
    smoke_sec = time.perf_counter() - t0
    rows = [r for rr, _, _ in res for r in rr]
    samples = [x for _, ss, _ in res for x in ss]
    errs = [e for _, _, ee in res for e in ee]
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:40], indent=1, default=str))
    if errs:
        (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
    summary["smoke"] = summarise_smoke(rows, errs, smoke_sec)
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
        "gpu_name": "none (CPU only)", "vram_total_mb": 0, "max_batch_size": "n/a", "vram_used_mb": 0,
        "utilization_pct": None, "note": "E1-Lin learner loop is CPU-only numpy (11-dim ellipsoid); gpu slot = token"}))
    progress(5, 5, "summary")
    return summary


def summarise_smoke(rows, errs, sec):
    bill_mis = sum(not r["billing_ok"] for r in rows)
    vol = [r for r in rows if r["arm"] == "vol"]
    vol_ok = all(r["replay_steps"] == (r["n_rounds_billed"] - 20 - r["new_env_steps"]) for r in vol)
    dyn = [r for r in rows if r["layer"] == "E1-Lin" and not r["orth_hook_smoke"]]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["layer"], r["method"], r["arm"])].append(r)
    by_cell = {}
    for (lay, m, a), rr in sorted(cells.items()):
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        sh = [r for r in rr if r["shared"]]
        by_cell[f"{lay}|{m}|{a}"] = {
            "n": len(rr), "status": dict(Counter(r["status"] for r in rr)),
            "S_nontie_tau20000": int(sum(r["cost_tau20000"] for r in nt)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "new_steps_median": float(np.median([r["new_env_steps"] for r in rr])),
            "new_steps_shared": [r["new_env_steps"] for r in sh],
            "zero_cost": int(sum(r["zero_cost"] for r in rr)), "false_cert": int(sum(r["false_cert"] for r in rr)),
            "censored_tmax": int(sum(r["truncated_at_tmax"] for r in rr)),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "pad_steps_total": int(sum(r["replay_pad_steps"] for r in rr)),
            "median_wall_s": float(np.median([r["wall_clock_s"] for r in rr])),
            "max_wall_s": float(max(r["wall_clock_s"] for r in rr)),
            "sec_per_1000_steps": float(1000 * sum(r["wall_clock_s"] for r in rr)
                                        / max(1, sum(r["new_env_steps"] for r in rr)))}
        if m == "B1eb":
            by_cell[f"{lay}|{m}|{a}"]["reused_trials_shared"] = [r.get("x_n_reused_trials") for r in sh]
    inter = {}
    for s in SMOKE_SEEDS:
        S = {(m, a): sum(r["cost_tau20000"] for r in dyn if r["instance"] == s and r["method"] == m and r["arm"] == a
                         and r["gap_layer"] != "tie") for m in ("JPC-Lin", "RAGE") for a in ("off", "vol", "ev1", "full")}
        lr = lambda m, a, b: float(np.log((S[(m, a)] + 1) / (S[(m, b)] + 1)))  # noqa: E731
        inter[s] = {"logratio_JPC_full_off": lr("JPC-Lin", "full", "off"),
                    "logratio_RAGE_full_off": lr("RAGE", "full", "off"),
                    "logratio_JPC_vol_off": lr("JPC-Lin", "vol", "off"),
                    "logratio_JPC_full_vol": lr("JPC-Lin", "full", "vol"),
                    "I_full_off": lr("JPC-Lin", "full", "off") - lr("RAGE", "full", "off")}
    orth = [r for r in rows if r["orth_hook_smoke"]]
    n_pm = len({(r["instance"], r["layer"], r["pid"], r["method"]) for r in rows})
    return {"n_runs": len(rows), "n_problem_method_pairs": n_pm, "crashes": len(errs), "billing_mismatch": bill_mis,
            "vol_replay_equals_Nk": vol_ok, "replay_never_billed": bill_mis == 0 and vol_ok,
            "false_cert_total": int(sum(r["false_cert"] for r in rows)),
            "orth_hook": {"n": len(orth), "status": dict(Counter(r["status"] for r in orth))},
            "sec": sec, "by_cell": by_cell, "interaction_per_instance_descriptive": inter,
            "per_problem_wall_s_p90": float(np.percentile([r["wall_clock_s"] for r in rows], 90)),
            "tie_layer_new_steps": {f"{m}|{a}": [r["new_env_steps"] for r in dyn if r["method"] == m and r["arm"] == a
                                                 and r["gap_layer"] == "tie"] for m, arms in CELLS for a in arms}}


def update_gpu_progress(status, start_iso, wall_min, snapshot):
    import fcntl
    gp = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(gp.read_text()) if gp.exists() else {}
        for k, v in (("completed", []), ("failed", []), ("running", {}), ("timings", {})):
            d.setdefault(k, v)
        key = "completed" if status == "success" else "failed"
        if TASK not in d[key]:
            d[key].append(TASK)
        d["running"].pop(TASK, None)
        d["timings"][TASK] = {"planned_min": 12, "actual_min": int(round(wall_min)), "start_time": start_iso,
                              "end_time": datetime.now().isoformat(), "config_snapshot": snapshot}
        tmp = gp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, gp)
        fcntl.flock(lf, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-gpu-progress", action="store_true")
    a = ap.parse_args()
    sub = "pilots" if a.mode == "pilot" else "full"
    out_dir = RES_ROOT / sub / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    t0 = time.perf_counter()
    status = "failed"
    try:
        if a.mode == "pilot":
            summary = pilot(out_dir, a.workers)
            sm = summary["smoke"]
            crit = {"unit_tests_passed": summary["unit_tests"]["all_passed"],
                    "closed_form_err_le_1e-9": summary["closed_form"]["max_err"] <= 1e-9,
                    "billing_mismatch_zero": sm["billing_mismatch"] == 0 and sm["crashes"] == 0}
        else:
            summary = {"task": TASK, "mode": "full", "unit_tests": run_tests(out_dir)}
            crit = {"unit_tests_passed": summary["unit_tests"]["all_passed"]}
        summary["code_sha256"] = code_sha()
        summary["pass_criteria"] = crit
        summary["passed"] = all(crit.values())
        summary["go_no_go"] = "GO" if summary["passed"] else "NO_GO"
        summary["metrics"] = {
            "unit_tests_passed": summary["unit_tests"]["passed"],
            "shared_problem_frac": summary.get("quota", {}).get("shared_problem_frac_mean"),
            "closed_form_err": summary.get("closed_form", {}).get("max_err"),
            "billing_mismatch": summary.get("smoke", {}).get("billing_mismatch"),
            "quota_fill_rate": summary.get("quota", {}).get("instance_fill_rate")}
        summary["wall_clock_s"] = time.perf_counter() - t0
        summary["finished_at"] = datetime.now().isoformat()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(json.dumps(summary["pass_criteria"]), summary["go_no_go"], flush=True)
        status = "success" if summary["passed"] else "failed"
        mark_done(status, f"{summary['go_no_go']} {crit}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise
    finally:
        if not a.no_gpu_progress:
            update_gpu_progress(status, start_iso, (time.perf_counter() - t0) / 60,
                                {"mode": a.mode, "layer": "E1-Lin 3x3 sigma=1.5 eps=0.05", "smoke_seeds": SMOKE_SEEDS,
                                 "quota_seeds": QUOTA_SEEDS, "n_smoke_problems": N_SMOKE, "tmax": T_MAX,
                                 "methods": [m for m, _ in CELLS], "cpu_workers": a.workers, "gpu_count": 0,
                                 "concurrent": "r3_p1 / r3_p3 / r3_p5"})


if __name__ == "__main__":
    main()
