"""r5_cr_main_b: confirmatory evaluation block B for C1/C2 (FDC + rigorous rivals R), CR9 eval half.

FULL  (--mode full):  lock gate (dsswm.stats.prereg.lock_gate, version 5) -> on refusal summary.status='skipped_by_lock'
                      and exit; else CR9 eval half, eps* = 0.001, K = 20, eval seeds 30100-30199, methods
                      ['FDC', 'B1', 'B4', 'B4-bal', 'B2-rect', 'B3-rect', 'Peace-rect', 'Hait-SW', 'Molitor-WoR',
                      'QFC-pool'], rival configs from the lock (rival_configs.json, sha256 checked). Every (method,
                      stream) row is written ahead to results.jsonl (fsync) with all run_stream summary fields + code
                      hashes; a restart skips rows already present. No analysis inside the block: the summary only
                      reports completeness, billing, errors and the sha256 of results.jsonl.
PILOT (--mode pilot): same code path on DEV streams 900-909 (never eval seeds): smoke + timing, lock-gate refusal path
                      tested on tampered copies of the lock, structural check of the eval-half ctx against the lock
                      (no truth, no streams), reproducibility check against r5_t2_gate_strict (same dev streams).
CPU only, 4 worker processes, BLAS threads pinned to 1; timings are from a concurrent run (other r5 blocks share the
host).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp/results"
TASK = "r5_cr_main_b"
METHODS = ["FDC", "B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool"]
EVAL_SEEDS = list(range(30100, 30200))
PILOT_SEEDS = list(range(900, 910))
N_WORKERS = 4

from dsswm.stats import prereg  # noqa: E402

_ENV = {}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_digest(lock):
    """sha256 over the frozen dsswm module hashes (as checked by assert_locked) + this runner."""
    items = sorted((k, v) for k, v in lock["code_sha256"].items() if k.startswith("dsswm/"))
    blob = json.dumps(items) + sha_file(Path(__file__))
    return {"dsswm_frozen_sha256": hashlib.sha256(json.dumps(items).encode()).hexdigest(),
            "runner_sha256": sha_file(Path(__file__)),
            "block_code_sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "freeze_commit": lock["git_commit"], "lock_sha256": lock["sha256"]}


def git_head():
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(CODE), capture_output=True, text=True)
    q = subprocess.run(["git", "status", "--porcelain", "--", "dsswm"], cwd=str(CODE), capture_output=True, text=True)
    return {"head": p.stdout.strip(), "dsswm_dirty": q.stdout.strip().splitlines()[:20]}


def progress(done, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, txt):
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
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------------ lock handling
def rival_kwargs(lock):
    rc_path = WS / lock["rival_configs"]["path"]
    got = sha_file(rc_path)
    if got != lock["rival_configs"]["sha256"]:
        raise RuntimeError(f"rival_configs.json sha256 {got} != lock {lock['rival_configs']['sha256']}")
    rc = json.loads(rc_path.read_text())
    kw = {m: dict(rc["frozen_configs"].get(m) or {}) for m in METHODS if m != "FDC"}
    kw["FDC"] = {}
    return kw


def test_lock_refusal_paths(lock_file: Path):
    """Pilot only: the gate must refuse tampered locks and accept the real one (nothing is written to plan/)."""
    real = json.loads(lock_file.read_text())
    out = {}
    cases = {}
    a = copy.deepcopy(real)
    a["status"] = "locked_no_eval"
    a["sha256"] = prereg.canonical_hash(a)
    cases["status_locked_no_eval"] = a
    b = copy.deepcopy(real)
    b["cr_n_streams"] = 400  # content change without re-hashing
    cases["hash_mismatch"] = b
    c = copy.deepcopy(real)
    c["version"] = 4
    cases["wrong_version"] = c
    d = copy.deepcopy(real)
    d["code_sha256"] = dict(d["code_sha256"])
    d["code_sha256"]["dsswm/baselines/fdc.py"] = "0" * 64
    d["sha256"] = prereg.canonical_hash(d)
    cases["code_drift"] = d
    with tempfile.TemporaryDirectory() as td:
        for name, lk in cases.items():
            p = Path(td) / f"{name}.json"
            p.write_text(json.dumps(lk))
            ok, why = prereg.lock_gate(TASK, version=5, path=p)
            out[name] = {"refused": not ok, "reason": None if ok else str(why)[:240]}
    ok, lk = prereg.lock_gate(TASK, version=5)
    out["real_lock"] = {"accepted": ok, "status": lk.get("status") if ok else lk}
    out["all_pass"] = all(v["refused"] for k, v in out.items() if k not in ("real_lock",)) and out["real_lock"]["accepted"]
    return out


# ------------------------------------------------------------------------------------------------ workers
def init_env(half, eps):
    from dsswm.envs.pool_replay import PoolReplayEnv
    from dsswm.streams import frontier as fr
    from dsswm.streams.frontier_runner import build_ctx, true_policy_values
    env = PoolReplayEnv("CR9", half)
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    _ENV.update(env=env, ctx=ctx, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]), eps=eps)


def job(a):
    from dsswm.streams import r5_registry as reg
    from dsswm.streams.frontier_runner import run_stream
    name, kw, seed, cd = a
    ctx = _ENV["ctx"]
    try:
        m = reg.make_method(name, **kw)
        t0 = time.perf_counter()
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=_ENV["J"],
                             J_star=_ENV["Js"], keep_U=False)
        ck = ctx.checkpoints
        n80_raw = int(ck[s["k_stop"]]) if s["reached_stop"] else int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))
        return {"task_id": TASK, "method": name, "params": kw, "seed": seed, "N80_pen": int(s["N80"]),
                "N80_raw": n80_raw, "completed": bool(s["reached_stop"]), "fwer_event": bool(s["fwer_event"]),
                "cert_k": s["cert_k"], "run_stream": s, "n_cert_curve": curve,
                "sec": round(time.perf_counter() - t0, 3), "code": cd, "written_at": datetime.now().isoformat(),
                "error": None}
    except Exception:  # noqa: BLE001
        return {"task_id": TASK, "method": name, "params": kw, "seed": seed, "error": traceback.format_exc()}


def load_done(rfile: Path):
    done = {}
    if rfile.exists():
        for line in rfile.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue  # torn last line from a crash: ignored, the row is re-run
            if not x.get("error"):
                done[(x["method"], x["seed"])] = x
    return done


def run_block(seeds, kws, out: Path, cd, label):
    rfile = out / "results.jsonl"
    done = load_done(rfile)
    if rfile.exists():  # rewrite without torn lines so the file stays valid jsonl
        rfile.write_text("".join(json.dumps(x) + "\n" for x in done.values()))
    order = ["Peace-rect", "B3-rect"] + [m for m in METHODS if m not in ("Peace-rect", "B3-rect")]
    jobs = [(m, kws[m], s, cd) for m in order for s in seeds if (m, s) not in done]
    total = len(seeds) * len(METHODS)
    print(f"[{label}] {len(done)} rows already present, {len(jobs)} to run", flush=True)
    errs = []
    n = len(done)
    progress(n, total)
    if jobs:
        with get_context("fork").Pool(N_WORKERS) as pool, open(rfile, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                if r.get("error"):
                    errs.append(r)
                    with open(out / "errors.jsonl", "a") as g:
                        g.write(json.dumps(r) + "\n")
                else:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    done[(r["method"], r["seed"])] = r
                n += 1
                if n % 10 == 0 or n == total:
                    progress(n, total, {"rows_done": len(done), "errors": len(errs)})
                    print(f"[{label} {n}/{total}] last={r['method']} seed={r['seed']} errors={len(errs)}", flush=True)
    return done, errs


def completeness(done, seeds, kws):
    missing = [(m, s) for m in METHODS for s in seeds if (m, s) not in done]
    rows = [done[(m, s)] for m in METHODS for s in seeds if (m, s) in done]
    bill_bad = [(r["method"], r["seed"]) for r in rows if not r["run_stream"]["billing_ok"]]
    return rows, missing, bill_bad


# ------------------------------------------------------------------------------------------------ modes
def main_full():
    out = RES / "full" / TASK
    out.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    ok, lock = prereg.lock_gate(TASK, version=5)
    if not ok:
        summ = {"task_id": TASK, "status": "skipped_by_lock", "reason": lock, "at": t_start.isoformat()}
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        return summ
    et = lock["eval_tasks"][TASK]
    assert [int(x) for x in et["seeds"].split("-")] == [EVAL_SEEDS[0], EVAL_SEEDS[-1]], et["seeds"]
    assert et["methods"] == METHODS and et["layer"] == "CR9" and et["half"] == "eval" and et["K"] == 20
    kws = rival_kwargs(lock)
    cd = code_digest(lock)
    init_env("eval", float(et["eps"]))
    ctx = _ENV["ctx"]
    sc = lock["structural_ctx"]["CR9_eval"]
    assert int(ctx.tau_R) == sc["tau_R"] and ctx.Q == sc["Q"] and ctx.stop_k == sc["stop_k"]
    assert [int(x) for x in ctx.checkpoints] == lock["fdc_spec"]["checkpoints"]["CR9_eval"]
    (out / "start_time.txt").write_text(t_start.isoformat())
    done, errs = run_block(EVAL_SEEDS, kws, out, cd, "eval")
    rows, missing, bill_bad = completeness(done, EVAL_SEEDS, kws)
    t_end = datetime.now()
    status = "complete" if (not missing and not bill_bad and not errs) else "incomplete"
    summ = {"task_id": TASK, "status": status, "mode": "full (confirmatory eval block)", "lock_version": 5,
            "lock_sha256": lock["sha256"], "freeze_commit": lock["git_commit"], "git": git_head(), "code": cd,
            "setting": {"layer": "CR9", "half": "eval", "eps": ctx.eps, "K": len(ctx.checkpoints),
                        "stop_k": int(ctx.stop_k), "Q": ctx.Q, "delta": ctx.delta, "tau_R": int(ctx.tau_R),
                        "seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n_streams": len(EVAL_SEEDS)},
            "methods": METHODS, "params": kws, "n_rows": len(rows), "n_expected": len(METHODS) * len(EVAL_SEEDS),
            "missing": missing[:50], "billing_not_ok": bill_bad[:50], "errors_this_run": len(errs),
            "results_sha256": sha_file(out / "results.jsonl"),
            "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
            "wall_min_this_run": round((t_end - t_start).total_seconds() / 60, 1),
            "timing_note": "concurrent run (4 workers; other r5 eval blocks share the 20-core host)",
            "analysis": "none inside the block (pre-registered SAP is applied by the analysis task)"}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


def main_pilot():
    out = RES / "pilots" / TASK
    out.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    lock_tests = test_lock_refusal_paths(prereg.LOCK_PATH)
    ok, lock = prereg.lock_gate(TASK, version=5)
    if not ok:
        raise RuntimeError(f"real lock refused: {lock}")
    et = lock["eval_tasks"][TASK]
    kws = rival_kwargs(lock)
    cd = code_digest(lock)
    # structural-only check of the eval half (no truth, no outcomes, no streams)
    from dsswm.envs.pool_replay import PoolReplayEnv
    from dsswm.streams import frontier as fr
    from dsswm.streams.frontier_runner import build_ctx
    ev = PoolReplayEnv("CR9", "eval")
    ectx = build_ctx(ev, fr.cr_problems("visit"), float(et["eps"]))
    sc = lock["structural_ctx"]["CR9_eval"]
    eval_struct = {"tau_R_match": int(ectx.tau_R) == sc["tau_R"], "Q_match": ectx.Q == sc["Q"],
                   "checkpoints_match": [int(x) for x in ectx.checkpoints] == lock["fdc_spec"]["checkpoints"]["CR9_eval"],
                   "pool_sizes_match": np.asarray(ev.pool_sizes).tolist() == sc["pool_sizes"],
                   "seeds_in_lock": et["seeds"], "methods_match": et["methods"] == METHODS}
    del ev, ectx
    # dev smoke run on the identical code path
    init_env("dev", float(et["eps"]))
    ctx = _ENV["ctx"]
    assert [int(x) for x in ctx.checkpoints] == lock["fdc_spec"]["checkpoints"]["CR9_dev"]
    rfile = out / "results.jsonl"
    if rfile.exists() and os.environ.get("PILOT_FRESH", "1") == "1":
        rfile.unlink()
    t_run = time.perf_counter()
    done, errs = run_block(PILOT_SEEDS, kws, out, cd, "pilot-dev")
    wall_run = time.perf_counter() - t_run
    rows, missing, bill_bad = completeness(done, PILOT_SEEDS, kws)
    # reproducibility vs r5_t2_gate_strict (same code, same dev streams)
    ref = {}
    for line in (RES / "pilots/r5_t2_gate_strict/results.jsonl").read_text().splitlines():
        x = json.loads(line)
        ref[(x["method"], x["seed"])] = x
    mism = [{"method": r["method"], "seed": r["seed"], "now": r["N80_pen"], "t2": ref[(r["method"], r["seed"])]["N80_pen"]}
            for r in rows if (r["method"], r["seed"]) in ref and (
                r["N80_pen"] != ref[(r["method"], r["seed"])]["N80_pen"]
                or r["cert_k"] != ref[(r["method"], r["seed"])]["cert_k"]
                or r["run_stream"]["schedule_digest"] != ref[(r["method"], r["seed"])]["schedule_digest"])]
    # timing projection for the full block (100 streams, 4 workers, longest-processing-time list schedule)
    sec = {m: float(np.mean([done[(m, s)]["sec"] for s in PILOT_SEEDS])) for m in METHODS}
    smax = {m: float(np.max([done[(m, s)]["sec"] for s in PILOT_SEEDS])) for m in METHODS}
    loads = np.zeros(N_WORKERS)
    for m in sorted(METHODS, key=lambda m: -sec[m]):
        for _ in EVAL_SEEDS:
            loads[np.argmin(loads)] += sec[m]
    proj_min = float(loads.max() / 60 * 1.2 + 2)  # +20% safety, +2 min setup (env + truth + lock checks)
    per_method = {}
    for m in METHODS:
        rr = [done[(m, s)] for s in PILOT_SEEDS]
        per_method[m] = {"params": kws[m], "sec_mean": round(sec[m], 2), "sec_max": round(smax[m], 2),
                         "geomean_N80_pen_dev": float(np.exp(np.mean(np.log([x["N80_pen"] for x in rr])))),
                         "completed": sum(x["completed"] for x in rr), "fwer_streams": sum(x["fwer_event"] for x in rr),
                         "billing_ok_all": all(x["run_stream"]["billing_ok"] for x in rr)}
    samples = [{k: done[(m, s)][k] for k in ("method", "seed", "N80_pen", "N80_raw", "completed", "fwer_event", "cert_k")}
               | {"decided_pi": done[(m, s)]["run_stream"]["decided_pi"],
                  "billing_ok": done[(m, s)]["run_stream"]["billing_ok"]}
               for s in (900, 905, 909) for m in ("FDC", "QFC-pool", "Peace-rect", "B4-bal")]
    crit = {"billing_ok_all": not bill_bad, "no_exceptions": not errs and not missing,
            "projected_full_min_le_55": proj_min <= 55, "lock_assertion_path_tested": lock_tests["all_pass"],
            "eval_structure_matches_lock": all(v for k, v in eval_struct.items() if k != "seeds_in_lock"),
            "reproduces_t2_gate_strict": not mism}
    go = all(crit.values())
    t_end = datetime.now()
    summ = {"task_id": TASK, "mode": "pilot (dev streams 900-909; eval seeds untouched)",
            "go_no_go": "GO" if go else "NO_GO", "criteria": crit, "lock_gate_tests": lock_tests,
            "eval_structure_check": eval_struct, "projected_full_block_min": round(proj_min, 1),
            "projection_rule": "mean dev sec/stream per method x 100 streams, LPT on 4 workers, x1.2 + 2 min",
            "pilot_run_wall_min": round(wall_run / 60, 2), "n_rows": len(rows), "n_expected": len(METHODS) * len(PILOT_SEEDS),
            "repro_vs_t2_gate_strict": {"n_compared": sum((r["method"], r["seed"]) in ref for r in rows),
                                        "n_mismatch": len(mism), "mismatches": mism[:20]},
            "per_method_dev": per_method, "samples": samples, "code": cd, "git": git_head(),
            "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
            "timing_note": "concurrent run (4 workers; r5_cr_main_a / plugin_a / plugin_b pilots share the host)",
            "eval_seeds_touched": False}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    mode = ap.parse_args().mode
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    try:
        s = main_full() if mode == "full" else main_pilot()
        if mode == "full":
            txt = f"full status={s['status']} rows={s.get('n_rows')}/{s.get('n_expected')}"
            mark_done("success" if s["status"] in ("complete", "skipped_by_lock") else "failed", txt)
        else:
            txt = (f"pilot {s['go_no_go']} | proj full {s['projected_full_block_min']} min | "
                   f"criteria={s['criteria']}")
            mark_done("success", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
