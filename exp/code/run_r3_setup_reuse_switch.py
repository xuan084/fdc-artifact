"""r3_setup_reuse_switch: setup + pilot for the round-3 five-level reuse switch.

New modules: dsswm/evidence/reuse_switch.py (off / vol / orth hook / ev(m) / full, shadow ledger, provenance, billing
asserts), dsswm/baselines/switched_nl.py (JPC, LR-chi2-grid, B3, B8 wired into the switch), dsswm/streams/gap_quota.py
(tie/near/clear quota, quota_fail replacement, offset-null, no-tie and low-overlap streams), dsswm/streams/r3_harness.py
(cell runner + sibling platform), dsswm/stats/mh.py (stratified MH-RR + cluster bootstrap), stats/prereg.py (lock v3).

pilot: (1) full unit-test suite; (2) quota fill on dev 734-741 (quota / no_tie / low_overlap); (3) offset-null true
gaps on dev 734-735 x 15 problems (30); (4) smoke: dev 734-735 x 8 problems x stream 0 x {JPC, B3} x
{off, vol, ev1, full} (128 runs, T_max = 6000) + integration smoke LR-chi2-grid / B8 x {off, full} and offset-null
JPC / B3 x {off, full} on dev 734 x 4 problems.  Pass: tests pass AND billing mismatch 0 AND quota fill >= 0.95 AND
offset-null max |true gap| <= 1e-12.
full: rerun the unit tests and write the code sha256 for the lock.
Usage: run_r3_setup_reuse_switch.py --mode {pilot,full} [--workers 4]
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
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3"
TASK = "r3_setup_reuse_switch"
PY = sys.executable

CODE_FILES = ["dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py", "dsswm/streams/gap_quota.py",
              "dsswm/streams/r3_harness.py", "dsswm/stats/mh.py", "dsswm/stats/prereg.py",
              "dsswm/tests/test_reuse_switch.py", "run_r3_setup_reuse_switch.py"]
QUOTA_SEEDS = list(range(734, 742))
SMOKE_SEEDS = [734, 735]
SMOKE_METHODS = ["JPC", "B3"]
SMOKE_ARMS = ["off", "vol", "ev1", "full"]
N_SMOKE = 8


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
    new = sum(1 for tc in ts.iter("testcase") if "test_reuse_switch" in (tc.get("classname") or ""))
    return {"returncode": r.returncode, "tests": n, "failures": f, "errors": e, "skipped": s,
            "passed": n - f - e - s, "new_tests_in_test_reuse_switch": new, "sec": time.perf_counter() - t0,
            "all_passed": r.returncode == 0 and f == 0 and e == 0}


def code_sha():
    out = {}
    for f in CODE_FILES:
        out[f] = hashlib.sha256((HERE / f).read_bytes()).hexdigest()
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


# --------------------------------------------------------------------------------------------- workers
_W = {}


def _worker_ctx():
    if "pub" not in _W:
        torch.set_num_threads(1)
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        from dsswm.streams.gap_quota import QuotaBuilder
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        b = QuotaBuilder(SMOKE_SEEDS[0], ncl, None, CACHE)
        prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
        _W.update(ncl=ncl, prop=prop, pub=PublicNL(prop, ncl, [], [], 1.0, 2, 0.02, 0.05), builders={})
    return _W


def run_job(seed, kind, method, arms, n_problems, tmax):
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.r3_harness import run_cell
    W = _worker_ctx()
    key = seed
    if key not in W["builders"]:
        W["builders"][key] = QuotaBuilder(seed, W["ncl"], None, CACHE, prop_cpu=W["prop"])
    b = W["builders"][key]
    sel = b.select("quota")
    ov = [o["cos_max"] for o in b.overlap([sel["base"][i] for i in
                                           np.random.default_rng([seed, 45, 0]).permutation(15)])][:n_problems]
    rows, samples, errs = [], [], []
    for arm in arms:
        t0 = time.perf_counter()
        try:
            rr = run_cell(b, W["pub"], 0, kind, method, arm, tmax=tmax, n_problems=n_problems, overlap=ov,
                          samples=samples)
            for r in rr:
                r["cell_sec"] = time.perf_counter() - t0
            rows += rr
        except Exception as e:  # noqa: BLE001 - crashes are counted, not hidden
            errs.append({"seed": seed, "kind": kind, "method": method, "arm": arm, "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
    return rows, samples, errs


# --------------------------------------------------------------------------------------------- pilot
def pilot(out_dir, workers):
    from joblib import Parallel, delayed
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.gap_quota import QuotaBuilder, fill_instances
    summary = {"task": TASK, "mode": "pilot", "started_at": datetime.now().isoformat(),
               "note": "并发运行（与 r3_setup_mechanism 共享 CPU/GPU）；计时偏高"}
    progress(1, 5, "unit_tests")
    summary["unit_tests"] = run_tests(out_dir)
    print("tests", summary["unit_tests"], flush=True)

    # ---- quota fill / offset-null (GPU tables)
    progress(2, 5, "quota_fill")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(QUOTA_SEEDS[0], ncl, None, CACHE)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    qrows = []
    t0 = time.perf_counter()
    builders = {}
    for s in QUOTA_SEEDS:
        b = QuotaBuilder(s, ncl, propg, CACHE, prop_cpu=propc)
        builders[s] = b
        rec = {"seed": s}
        for kind in ("quota", "no_tie", "low_overlap"):
            tk = time.perf_counter()
            sel = b.select(kind)
            rec[kind] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"],
                         "layer_counts": sel["layer_counts"], "sec": time.perf_counter() - tk}
        rec["pool_layers"] = dict(Counter(c.layer for c in b.pool))
        rec["tables_computed"] = b.n_table_computed
        rec["table_sec"] = b.sec_tables
        rand = b.overlap(b.select("quota")["base"])
        low = b.overlap(b.select("low_overlap")["base"])
        rec["cos_max_mean_quota"] = float(np.mean([o["cos_max"] for o in rand[1:]]))
        rec["cos_max_mean_low_overlap"] = float(np.mean([o["cos_max"] for o in low[1:]]))
        rec["cos_span_mean_quota"] = float(np.mean([o["cos_span"] for o in rand[1:]]))
        qrows.append(rec)
        print("quota", s, rec["quota"], rec["no_tie"]["n_draws"], rec["pool_layers"], flush=True)
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    with open(out_dir / "quota_fill.jsonl", "w") as f:
        for r in qrows:
            f.write(json.dumps(r) + "\n")
    n_ok = {k: sum(not r[k]["quota_fail"] for r in qrows) for k in ("quota", "no_tie", "low_overlap")}
    slot = {k: float(np.mean([sum(r[k]["layer_counts"].values()) / 15 for r in qrows]))
            for k in ("quota", "no_tie", "low_overlap")}
    seeds_ok, repl = fill_instances(QUOTA_SEEDS, lambda s: not builders[s].select("quota")["quota_fail"]
                                    if s in builders else True, reserve_start=742)
    summary["quota"] = {"seeds": QUOTA_SEEDS, "instance_fill_rate": {k: v / len(qrows) for k, v in n_ok.items()},
                        "slot_fill_rate": slot, "replacement_log": repl, "eps": 0.02, "per_layer": 5,
                        "max_draws_per_layer": 2000,
                        "draws_median": {k: float(np.median([r[k]["n_draws"] for r in qrows])) for k in
                                         ("quota", "no_tie")},
                        "draws_max": {k: int(max(r[k]["n_draws"] for r in qrows)) for k in ("quota", "no_tie")},
                        "pool_layer_frac": {l: float(np.mean([r["pool_layers"].get(l, 0) / sum(r["pool_layers"].values())
                                                              for r in qrows])) for l in ("tie", "near", "clear")},
                        "cos_max_mean": {"quota": float(np.mean([r["cos_max_mean_quota"] for r in qrows])),
                                         "low_overlap": float(np.mean([r["cos_max_mean_low_overlap"] for r in qrows]))},
                        "sec_total": time.perf_counter() - t0}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
        "gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "vram_total_mb": (torch.cuda.get_device_properties(0).total_memory / 2 ** 20) if dev == "cuda" else 0,
        "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call)", "vram_used_mb": vram,
        "utilization_pct": None, "note": "GPU only for class J-tables of candidate problems; learner loop on CPU"}))

    progress(3, 5, "offset_null")
    null = []
    for s in SMOKE_SEEDS:
        st = builders[s].stream(0, "offset_null")
        for q, J, hm in zip(st.problems, st.J, st.harness):
            jt = J[st.theta_index]
            null.append({"seed": s, "pid": q.pid, "true_gap_null": float(jt.max() - jt.min()),
                         "orig_gap": hm["true_gap_orig"], "orig_layer": hm["gap_layer"],
                         "class_spread_max": float((J.max(1) - J.min(1)).max()),
                         "frac_theta_nontie": float(((J.max(1) - J.min(1)) >= 0.02).mean())})
    with open(out_dir / "offset_null.jsonl", "w") as f:
        for r in null:
            f.write(json.dumps(r) + "\n")
    summary["offset_null"] = {"n_problems": len(null), "max_abs_true_gap": max(r["true_gap_null"] for r in null),
                              "median_frac_theta_nontie": float(np.median([r["frac_theta_nontie"] for r in null]))}
    del ncl, propg
    torch.cuda.empty_cache() if dev == "cuda" else None

    # ---- smoke runs
    progress(4, 5, "smoke")
    jobs = [(s, "quota", m, SMOKE_ARMS, N_SMOKE, 6000) for s in SMOKE_SEEDS for m in SMOKE_METHODS]
    jobs += [(734, "quota", m, ["off", "full"], 4, 6000) for m in ("LR-chi2-grid", "B8")]
    jobs += [(734, "offset_null", m, ["off", "full"], 4, 6000) for m in ("JPC", "B3")]
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
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:24], indent=1, default=str))
    if errs:
        (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
    summary["smoke"] = summarise_smoke(rows, errs, smoke_sec)
    progress(5, 5, "summary")
    return summary


def summarise_smoke(rows, errs, sec):
    main = [r for r in rows if r["kind"] == "quota" and r["method"] in SMOKE_METHODS and r["instance"] in SMOKE_SEEDS
            and r["problem_index"] < N_SMOKE]
    bill_mis = sum(not r["billing_ok"] for r in rows)
    vol = [r for r in rows if r["arm"] == "vol"]
    vol_ok = all(r["replay_steps"] == (r["n_rounds_billed"] - 20 - r["new_env_steps"]) for r in vol)
    cells = defaultdict(list)
    for r in main:
        cells[(r["method"], r["arm"])].append(r)
    by_cell = {}
    for (m, a), rr in sorted(cells.items()):
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        by_cell[f"{m}|{a}"] = {
            "n": len(rr), "status": dict(Counter(r["status"] for r in rr)),
            "S_nontie_tau3000": int(sum(r["cost_tau3000"] for r in nt)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "zero_cost": int(sum(r["zero_cost"] for r in rr)), "false_cert": int(sum(r["false_cert"] for r in rr)),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "pad_steps_total": int(sum(r["replay_pad_steps"] for r in rr)),
            "median_wall_s": float(np.median([r["wall_clock_s"] for r in rr]))}
    # interaction I = [log S(JPC,full)/S(JPC,off)] - [log S(B3,full)/S(B3,off)] on non-tie problems, per instance
    inter = {}
    for s in SMOKE_SEEDS:
        S = {(m, a): sum(r["cost_tau3000"] for r in main if r["instance"] == s and r["method"] == m and r["arm"] == a
                         and r["gap_layer"] != "tie") for m in SMOKE_METHODS for a in SMOKE_ARMS}
        lr = lambda m, a, b: float(np.log((S[(m, a)] + 1) / (S[(m, b)] + 1)))  # noqa: E731
        inter[s] = {"logratio_JPC_full_off": lr("JPC", "full", "off"), "logratio_B3_full_off": lr("B3", "full", "off"),
                    "logratio_JPC_vol_off": lr("JPC", "vol", "off"), "logratio_JPC_full_vol": lr("JPC", "full", "vol"),
                    "I_full_off": lr("JPC", "full", "off") - lr("B3", "full", "off")}
    extra = defaultdict(list)
    for r in rows:
        if r not in main:
            extra[f"{r['kind']}|{r['method']}|{r['arm']}"].append(r)
    return {"n_runs": len(rows), "n_main_runs": len(main), "crashes": len(errs), "billing_mismatch": bill_mis,
            "vol_replay_equals_Nk": vol_ok, "replay_never_billed": bill_mis == 0 and vol_ok,
            "sec": sec, "by_cell": by_cell, "interaction_per_instance": inter,
            "extra_cells": {k: {"n": len(v), "status": dict(Counter(r["status"] for r in v)),
                                "new_steps": [r["new_env_steps"] for r in v],
                                "false_cert": int(sum(r["false_cert"] for r in v))} for k, v in extra.items()},
            "tie_layer_new_steps": {f"{m}|{a}": [r["new_env_steps"] for r in main if r["method"] == m and r["arm"] == a
                                                 and r["gap_layer"] == "tie"] for m in SMOKE_METHODS
                                    for a in SMOKE_ARMS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sub = "pilots" if a.mode == "pilot" else "full"
    out_dir = RES_ROOT / sub / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t0 = time.perf_counter()
    try:
        if a.mode == "pilot":
            summary = pilot(out_dir, a.workers)
            sm = summary["smoke"]
            crit = {"unit_tests_passed": summary["unit_tests"]["all_passed"],
                    "billing_mismatch_zero": sm["billing_mismatch"] == 0 and sm["crashes"] == 0,
                    "quota_fill_rate_ge_0.95": summary["quota"]["instance_fill_rate"]["quota"] >= 0.95,
                    "offset_null_max_gap_le_1e-12": summary["offset_null"]["max_abs_true_gap"] <= 1e-12}
        else:
            summary = {"task": TASK, "mode": "full", "unit_tests": run_tests(out_dir)}
            crit = {"unit_tests_passed": summary["unit_tests"]["all_passed"]}
        summary["code_sha256"] = code_sha()
        summary["pass_criteria"] = crit
        summary["passed"] = all(crit.values())
        summary["go_no_go"] = "GO" if summary["passed"] else "NO_GO"
        summary["metrics"] = {
            "unit_tests_passed": summary["unit_tests"]["passed"],
            "billing_mismatch": summary.get("smoke", {}).get("billing_mismatch"),
            "quota_fill_rate": summary.get("quota", {}).get("instance_fill_rate", {}).get("quota"),
            "offset_null_max_gap": summary.get("offset_null", {}).get("max_abs_true_gap")}
        summary["wall_clock_s"] = time.perf_counter() - t0
        summary["finished_at"] = datetime.now().isoformat()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(json.dumps(summary["pass_criteria"]), summary["go_no_go"], flush=True)
        mark_done("success" if summary["passed"] else "failed", f"{summary['go_no_go']} {crit}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
