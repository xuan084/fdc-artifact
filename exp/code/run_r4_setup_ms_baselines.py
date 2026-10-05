"""r4_setup_ms_baselines: multi-step harness r4 (MS-H truth-label hazard sampler, MS-S, MS-R3, MS-F; one stream per
instance) + Guard, MisLid-ms, DF-sep-ledger, PolicyCert-tab, PERP-fact, always-dynamic JPC adapters (methodology 1.6).

Harness side (placements, truth, scoring). Learner code = dsswm.baselines.{ms_common, guard, mislid_ms, dfsep_ledger,
policycert_tab, perp_fact} (no truth imports; checked by dsswm/tests/test_ms_r4.py).

pilot (setup smoke, NO scientific readout; dev seeds only, zero evaluation seeds):
  1. unit tests dsswm/tests/test_ms_r4.py
  2. MS-H placements on dev 800-819 (acceptance: >= 15 truth-label problems per instance, cap 4000 proposals)
  3. r3 JPC replay: r3_hazard_a pilot dev instances 750-752 (R1-adv, stream 0, 45 rows) through the new harness,
     compared row by row with the r3 record (status, new steps, certified policy, false_cert, regret)
  4. smoke: 100 (instance, method) runs on dev 800-819 across the 4 layers (5 instances per layer x 5 method slots),
     first SMOKE_PROBLEMS problems of each stream at the full T_max (timing -> sec_per_instance_method projection)
  5. unit tests again (MS-H placement filter test now active)
full: rerun the unit tests and record sha256 of the code files.

Usage: run_r4_setup_ms_baselines.py --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores + the RTX 4090 with other round-4 tasks): timings are "concurrent" (并发运行).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import fcntl  # noqa: E402
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
RES = WS / "exp" / "results"
TASK = "r4_setup_ms_baselines"
PLANNED_MIN = 60
DEV = list(range(800, 820))
LAYER_SEEDS = {"MS-H": DEV[0:5], "MS-S": DEV[5:10], "MS-R3": DEV[10:15], "MS-F": DEV[15:20]}
SMOKE_PROBLEMS = 2
R3_REPLAY_SEEDS = (750, 751, 752)
R3_PILOT = RES / "pilots" / "r3_hazard_a"
CODE_FILES = ["dsswm/streams/ms_r4.py", "dsswm/baselines/ms_common.py", "dsswm/baselines/guard.py",
              "dsswm/baselines/mislid_ms.py", "dsswm/baselines/dfsep_ledger.py", "dsswm/baselines/policycert_tab.py",
              "dsswm/baselines/perp_fact.py", "dsswm/tests/test_ms_r4.py", "run_r4_setup_ms_baselines.py",
              "run_r3_hazard.py", "dsswm/baselines/switched_nl.py", "dsswm/evidence/reuse_switch.py",
              "dsswm/certify/fcc.py"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(step, total, phase, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def update_gpu_progress(status, start_iso, wall_min, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    with open(WS / "exp" / "gpu_progress.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        other = "failed" if key == "completed" else "completed"
        if TASK in d.get(other, []):
            d[other].remove(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": PLANNED_MIN, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def update_shared_summary(entry, md):
    with open(RES / "pilot_summary.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            pj = RES / "pilot_summary.json"
            d = json.loads(pj.read_text()) if pj.exists() else {"tasks": {}}
            d.setdefault("tasks", {})[TASK] = entry
            tmp = pj.with_name(pj.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))
            os.replace(tmp, pj)
            pm = RES / "pilot_summary.md"
            txt = pm.read_text() if pm.exists() else ""
            head = f"## {TASK} "
            if head in txt:
                s = txt.index(head)
                e = txt.find("\n## ", s + 1)
                txt = txt[:s] + md.strip() + "\n" + (txt[e:] if e >= 0 else "")
            else:
                txt = txt.rstrip() + "\n\n" + md.strip() + "\n"
            pm.write_text(txt)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


def run_unit_tests(out_dir, tag):
    t0 = time.time()
    xml = out_dir / f"pytest_ms_r4_{tag}.xml"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "dsswm/tests/test_ms_r4.py", f"--junitxml={xml}"],
                       cwd=HERE, capture_output=True, text=True, timeout=1200)
    import xml.etree.ElementTree as ET
    ts = ET.parse(xml).getroot()
    ts = ts if ts.tag == "testsuite" else ts.find("testsuite")
    n, f, e, s = (int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    return {"returncode": r.returncode, "tests": n, "failures": f, "errors": e, "skipped": s, "passed": n - f - e - s,
            "all_pass": r.returncode == 0 and f == 0 and e == 0, "tail": r.stdout.strip().splitlines()[-4:],
            "seconds": round(time.time() - t0, 1)}


# ============================================================================================ workers
def _replay_job(seed):
    from dsswm.streams import ms_r4 as M
    plc = json.loads((R3_PILOT / "parts" / f"place_i{seed}.json").read_text())["R1adv"]
    inst = M.build_msr3(seed, plc)
    rows, info = M.run_stream(inst, "JPC")
    return rows, info


def _smoke_job(layer, seed, methods, plc, out_dir):
    import torch
    torch.set_num_threads(1)
    from dsswm.streams import ms_r4 as M
    t0 = time.perf_counter()
    errs, rows, infos = [], [], []
    try:
        if layer == "MS-H":
            inst = M.build_msh(seed, plc)
        elif layer == "MS-R3":
            inst = M.build_msr3(seed, plc)
        elif layer == "MS-S":
            inst = M.build_mss(seed)
        else:
            inst = M.build_msf(seed)
    except Exception as e:  # noqa: BLE001
        return {"layer": layer, "seed": seed, "rows": [], "infos": [], "build_sec": time.perf_counter() - t0,
                "errors": [{"stage": "build", "error": repr(e), "tb": traceback.format_exc()[-2000:]}]}
    build_sec = time.perf_counter() - t0
    for m in methods:
        try:
            r, inf = M.run_stream(inst, m, n_problems=SMOKE_PROBLEMS,
                                  decision_path=out_dir / "decisions" / f"{layer}_i{seed}_{m}.jsonl")
            rows += r
            infos.append(inf)
        except Exception as e:  # noqa: BLE001
            errs.append({"layer": layer, "seed": seed, "method": m, "stage": "run", "error": repr(e),
                         "tb": traceback.format_exc()[-2000:]})
    return {"layer": layer, "seed": seed, "rows": rows, "infos": infos, "build_sec": build_sec, "errors": errs,
            "instance_info": {k: v for k, v in inst.info.items() if k != "calibration"} |
            ({"calibration": {k: inst.info["calibration"][k] for k in ("strength", "eta_min_over_eps", "calibrated",
                                                                         "replica_vs_main_max_abs_diff")}}
             if "calibration" in inst.info else {})}


def method_slots(layer):
    from dsswm.streams.ms_r4 import METHODS
    pool = [m for m in METHODS if layer == "MS-H" or m != "PERP-fact"]
    return [pool[i % len(pool)] for i in range(25)]


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "seed": 42, "started": start_iso, "candidate_id": "cand_qfc",
               "timing_note": "并发运行 (concurrent with other round-4 tasks; <= 4 CPU workers, BLAS threads = 1; "
                              "GPU only for MS-H/MS-R3 placement MC and class J tables, shared 4090)"}
    status = "success"
    try:
        if a.mode == "full":
            progress(0, 1, "unit_tests")
            summary["unit_tests"] = run_unit_tests(out_dir, "full")
            summary["code_sha256"] = code_sha()
            summary["pass"] = summary["unit_tests"]["all_pass"]
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
            status = "success" if summary["pass"] else "failed"
            return
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from dsswm.streams import ms_r4 as M
        progress(0, 5, "unit_tests_pre")
        summary["unit_tests_pre"] = run_unit_tests(out_dir, "pre")
        log(f"unit tests (pre): {summary['unit_tests_pre']}")

        # ---------------------------------------------------------------- 2. placements (GPU, main)
        progress(1, 5, "placements")
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        edges_mu = lock["frozen_items"]["r1adv_bin_edges"]["mu_flip"]
        pdir = out_dir / "placements"
        pdir.mkdir(exist_ok=True)
        P = M.MSPlacer(edges_mu)
        msh, place_sec = {}, {}
        for s in DEV:
            fn = pdir / f"msh_i{s}.json"
            t0 = time.perf_counter()
            if fn.exists():
                msh[s] = json.loads(fn.read_text())
            else:
                msh[s] = P.place_msh(s)
                fn.write_text(json.dumps(msh[s]))
            place_sec[s] = time.perf_counter() - t0
            log(f"MS-H i{s}: fill={msh[s]['fill']} draw={msh[s]['draw']} proposals={msh[s]['proposals']} "
                f"acc={msh[s]['proposals_accepted']} ({place_sec[s]:.1f}s)")
            P.drop_seed(s)
        msr3 = {}
        for s in LAYER_SEEDS["MS-R3"]:
            fn = pdir / f"msr3_i{s}.json"
            if fn.exists():
                msr3[s] = json.loads(fn.read_text())
            else:
                msr3[s] = P.place_msr3(s)
                fn.write_text(json.dumps(msr3[s]))
            P.drop_seed(s)
        del P
        t0 = time.perf_counter()
        quota = M.prepare_quota(LAYER_SEEDS["MS-S"] + LAYER_SEEDS["MS-F"])
        quota_sec = time.perf_counter() - t0
        fills = np.array([msh[s]["fill"] for s in DEV])
        acc = {"n_instances": len(DEV), "fill_per_instance": {s: msh[s]["fill"] for s in DEV},
               "n_full_15": int((fills >= M.Q).sum()), "frac_full_15": float((fills >= M.Q).mean()),
               "proposals_per_instance": {s: msh[s]["proposals"] for s in DEV},
               "draws_scanned": {s: msh[s]["draws_scanned"] for s in DEV},
               "accept_rate_per_proposal_mean": float(np.mean([msh[s]["acceptance_rate"] for s in DEV])),
               "eta_near_over_eps_min_accepted": float(min(r["eta_arg_near_exante"] / M.EPS for s in DEV
                                                            for r in msh[s]["per_problem"])),
               "mu_flip_min_accepted": float(min(r["mu_flip"] for s in DEV for r in msh[s]["per_problem"])),
               "gap_layers_accepted": dict(Counter(r["gap_layer"] for s in DEV for r in msh[s]["per_problem"])),
               "d_desc_in_band_share": float(np.mean([M.D_BAND[0] <= r["true_gap"] / M.EPS <= M.D_BAND[1]
                                                      for s in DEV for r in msh[s]["per_problem"]])),
               "placement_sec_mean": float(np.mean(list(place_sec.values()))),
               "placement_sec_max": float(np.max(list(place_sec.values())))}
        summary["msh_acceptance"] = acc
        summary["msr3_placement"] = {s: {k: msr3[s][k] for k in ("draw", "fill_per_bin", "underfilled")}
                                     for s in msr3}
        summary["quota_prepare"] = {"per_seed": quota, "sec": quota_sec}
        log(f"MS-H acceptance: {acc['n_full_15']}/{len(DEV)} instances with 15 problems")

        # ---------------------------------------------------------------- 3 + 4. replay + smoke (CPU pool)
        progress(2, 5, "replay_and_smoke")
        jobs = []
        for layer, seeds in LAYER_SEEDS.items():
            slots = method_slots(layer)
            for i, s in enumerate(seeds):
                plc = msh.get(s) if layer == "MS-H" else (msr3.get(s) if layer == "MS-R3" else None)
                jobs.append((layer, s, slots[5 * i:5 * i + 5], plc))
        replay_rows, replay_infos, smoke = [], [], []
        n_tot = len(jobs) + len(R3_REPLAY_SEEDS)
        done = 0
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_replay_job, s): ("replay", s) for s in R3_REPLAY_SEEDS}
            futs.update({ex.submit(_smoke_job, *j, out_dir): ("smoke", j[0], j[1]) for j in jobs})
            for f in as_completed(futs):
                tag = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001
                    r = {"fatal": repr(e), "tb": traceback.format_exc()[-2000:]}
                if tag[0] == "replay":
                    if "fatal" in r:
                        summary.setdefault("fatal", []).append({"tag": tag, **r})
                    else:
                        replay_rows += r[0]
                        replay_infos.append(r[1])
                else:
                    smoke.append(r if "fatal" not in r else {"layer": tag[1], "seed": tag[2], "rows": [],
                                                              "infos": [], "errors": [r]})
                done += 1
                progress(2, 5, "replay_and_smoke", {"done": done, "total": n_tot, "last": str(tag)})
                log(f"{tag} done ({done}/{n_tot})")

        # ---------------------------------------------------------------- analysis
        old = [json.loads(l) for l in open(R3_PILOT / "results.jsonl")]
        old = {(r["instance"], r["problem"]): r for r in old if r["layer"] == "R1adv" and r["base_method"] == "JPC"
               and r["stream"] == 0}
        cmp_keys = ("pid", "status", "new_env_steps", "certified_policy", "false_cert")
        mism = []
        for r in replay_rows:
            o = old.get((r["instance"], r["problem"]))
            bad = [k for k in cmp_keys if o is None or r[k] != o[k]]
            if o is not None and (r["true_regret"] is None) != (o["true_regret"] is None):
                bad.append("true_regret")
            elif o is not None and r["true_regret"] is not None and abs(r["true_regret"] - o["true_regret"]) > 1e-9:
                bad.append("true_regret")
            if bad:
                mism.append({"instance": r["instance"], "problem": r["problem"], "fields": bad})
        summary["r3_replay"] = {"n_rows": len(replay_rows), "n_r3_rows": len(old), "n_mismatch": len(mism),
                                "identical": len(mism) == 0 and len(replay_rows) >= 20, "mismatches": mism[:20],
                                "compared_fields": list(cmp_keys) + ["true_regret (|diff| <= 1e-9)"],
                                "source": "exp/results/pilots/r3_hazard_a (dev 750-752, R1-adv, stream 0, JPC full)"}
        rows = [r for j in smoke for r in j["rows"]]
        errs = [e for j in smoke for e in j["errors"]]
        with open(out_dir / "smoke_rows.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
        with open(out_dir / "replay_rows.jsonl", "w") as fh:
            for r in replay_rows:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
        if errs:
            (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
        infos = [i for j in smoke for i in j["infos"]]
        per_lm = defaultdict(list)
        for r in rows:
            per_lm[(r["layer"], r["method"])].append(r)
        cells = {}
        for (layer, m), rr in sorted(per_lm.items()):
            sec_p = float(np.mean([r["wall_clock_s"] for r in rr]))
            cells[f"{layer}|{m}"] = {
                "n_problem_rows": len(rr), "status_counts": dict(Counter(r["status"] for r in rr)),
                "n_cert": int(sum(r["status"] == "CERTIFIED" for r in rr)),
                "n_false_cert": int(sum(r["false_cert"] for r in rr)),
                "mean_new_steps": float(np.mean([r["new_env_steps"] for r in rr])),
                "sec_per_problem": sec_p, "sec_per_instance_projected_15": sec_p * M.Q}
        sec_im = {}
        for m in M.METHODS:
            v = [c["sec_per_instance_projected_15"] for k, c in cells.items() if k.split("|")[1] == m]
            if v:
                sec_im[m] = {"mean": float(np.mean(v)), "max": float(np.max(v))}
        billing_mis = int(sum(not r["billing_ok"] for r in rows) + sum(not r["billing_ok"] for r in replay_rows)
                          + sum(r["env_n_steps"] != r["n_rounds_billed"] for r in rows + replay_rows))
        wa_ok = all(i["write_ahead_ok"] for i in infos + replay_infos)
        n_runs = len(infos)
        summary["smoke"] = {"n_instance_method_runs": n_runs, "n_problem_rows": len(rows), "n_errors": len(errs),
                            "problems_per_run": SMOKE_PROBLEMS, "layers": {l: s for l, s in LAYER_SEEDS.items()},
                            "methods_covered": sorted({r["method"] for r in rows}),
                            "cells": cells, "build_sec": {f"{j['layer']}|{j['seed']}": j.get("build_sec")
                                                          for j in smoke},
                            "instance_info": {f"{j['layer']}|{j['seed']}": j.get("instance_info") for j in smoke},
                            "env_resets_total": int(sum(i["env_resets"] for i in infos))}
        summary["sec_per_instance_method"] = sec_im
        summary["billing_mismatch"] = billing_mis
        summary["write_ahead_ok"] = wa_ok
        samples = []
        for layer in LAYER_SEEDS:
            for r in [x for x in rows if x["layer"] == layer][:3]:
                samples.append({k: r.get(k) for k in ("layer", "instance", "method", "problem", "pid", "H",
                                                      "n_policies", "status", "certified_policy", "new_env_steps",
                                                      "true_regret", "false_cert", "true_gap", "d_desc",
                                                      "eta_near_over_eps", "mu_flip")})
        (out_dir / "samples.json").write_text(json.dumps(samples, indent=1, default=str))

        # ---------------------------------------------------------------- 5. unit tests (MS-H placement test active)
        progress(4, 5, "unit_tests_post")
        summary["unit_tests"] = run_unit_tests(out_dir, "post")
        ut = summary["unit_tests"]
        crit = {"all_unit_tests_pass": bool(ut["all_pass"] and summary["unit_tests_pre"]["all_pass"]),
                "r3_replay_identical_ge20": bool(summary["r3_replay"]["identical"]),
                "msh_acceptance_ge15_on_ge90pct": bool(acc["frac_full_15"] >= 0.9),
                "billing_mismatch_0": billing_mis == 0, "write_ahead_ok": wa_ok,
                "smoke_runs_ge100": n_runs >= 100, "no_errors": len(errs) == 0}
        summary["pass_criteria"] = crit
        summary["pass"] = all(crit[k] for k in ("all_unit_tests_pass", "r3_replay_identical_ge20",
                                                "msh_acceptance_ge15_on_ge90pct", "billing_mismatch_0"))
        summary["code_sha256"] = code_sha()
        summary["wall_min"] = (time.time() - t_start) / 60
        summary["end"] = datetime.now().isoformat()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        go = "GO" if summary["pass"] else "NO_GO"
        entry = {"task_id": TASK, "candidate_id": "cand_qfc", "go_no_go": go, "confidence": 0.8 if summary["pass"]
                 else 0.4, "gate": "setup pilot: unit tests AND r3 JPC replay identical (>=20 dev rows) AND MS-H "
                                   ">=15 problems/instance on >=90% instances AND billing mismatch 0",
                 "key_metrics": {"unit_tests_passed": ut["passed"], "unit_tests_total": ut["tests"],
                                 "r3_replay_rows": len(replay_rows), "r3_replay_mismatch": len(mism),
                                 "msh_full_15": f"{acc['n_full_15']}/{len(DEV)}", "billing_mismatch": billing_mis,
                                 "smoke_runs": n_runs, "sec_per_instance_method": sec_im},
                 "pass_criteria": crit, "notes": "smoke only, no scientific readout (dev seeds 800-819, 750-752)"}
        lines = [f"## {TASK} ({go})", "",
                 f"- 单测：{ut['passed']}/{ut['tests']} 通过（pre {summary['unit_tests_pre']['passed']}/"
                 f"{summary['unit_tests_pre']['tests']}）",
                 f"- r3 JPC 经新 harness 重放：{len(replay_rows)} 行，不一致 {len(mism)} 行",
                 f"- MS-H 真值标签接受：{acc['n_full_15']}/{len(DEV)} 个实例填满 15 题"
                 f"（每提议接受率均值 {acc['accept_rate_per_proposal_mean']:.3f}）",
                 f"- 冒烟 {n_runs} 个 (实例, 方法) 运行，错误 {len(errs)}，计费不一致 {billing_mis}，write-ahead "
                 f"{'OK' if wa_ok else 'FAIL'}",
                 "- 每实例每方法投影秒数（15 题，并发运行）：" + ", ".join(f"{m} {v['mean']:.0f}s"
                                                             for m, v in sec_im.items()),
                 "- 只做冒烟与计时，无科学读数。"]
        update_shared_summary(entry, "\n".join(lines))
        status = "success" if summary["pass"] else "failed"
    except Exception as e:  # noqa: BLE001
        status = "failed"
        summary["fatal_error"] = {"error": repr(e), "tb": traceback.format_exc()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(traceback.format_exc())
    finally:
        wall = (time.time() - t_start) / 60
        update_gpu_progress(status, start_iso, wall, {"mode": a.mode, "dev_seeds": "800-819 + r3 750-752",
                                                      "workers": a.workers, "smoke_problems_per_run": SMOKE_PROBLEMS,
                                                      "tmax": 6000, "gpu": "RTX 4090 shared (placement MC only)",
                                                      "note": "并发运行"})
        mark_done(status, f"{TASK} {a.mode}: pass={summary.get('pass')}")
        log(f"done status={status} wall={wall:.1f} min")


if __name__ == "__main__":
    main()
