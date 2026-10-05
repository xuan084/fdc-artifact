"""r5_cr_main_{a,b}: confirmatory C1/C2 evaluation block (FDC + rigorous set R) on CR9 eval streams.

Usage:
  python run_r5_cr_main.py --task r5_cr_main_a --mode full    # eval seeds from lock v5 eval_tasks[task]
  python run_r5_cr_main.py --task r5_cr_main_a --mode pilot   # smoke/timing on DEV seeds 900-909 (never eval seeds)

Protocol (lock v5, plan/prereg_lock.json):
* start-up: dsswm.stats.prereg.lock_gate(task_id) (== assert_locked(version=5, task_id) incl. frozen-code drift check).
  On refusal: write summary.json with status='skipped_by_lock' and exit 0 (no stream is run).
* setting: CR9, eps* = 0.001 (r4_gates/eps.json, cross-checked with the lock), K = 20, Q = 15, stop at 12/15,
  delta = 0.05. full mode uses the eval half; pilot mode uses the dev half.
* methods / seeds / rival configs: taken from the lock (eval_tasks[task].methods, eval_tasks[task].seeds,
  rival_configs.frozen_configs); the sha256 of exp/results/r5_gates/rival_configs.json must match the lock.
* every (method, stream) writes all run_stream summary fields + endpoint fields + code hash to a write-ahead
  results.jsonl (fsync per row); a restart skips the (method, seed) pairs already present (resumable).
* NO analysis inside the block (no bootstrap, no ratios): only completeness / billing / hash checks.
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run: timings biased up).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import subprocess
import sys
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
GATES = RES / "r5_gates"
DEV_PILOT_SEEDS = list(range(900, 910))
N_WORKERS = 4

from dsswm.stats import prereg  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/clucb_joint.py",
         "dsswm/baselines/uniform_rs.py", "dsswm/baselines/b4_bal.py", "dsswm/baselines/combgame_joint.py",
         "dsswm/baselines/b2_rect_fe.py", "dsswm/baselines/rage_frontier.py", "dsswm/baselines/peace_frontier.py",
         "dsswm/baselines/hait_sw.py", "dsswm/baselines/molitor_wor.py", "dsswm/streams/r5_registry.py",
         "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py", "dsswm/certify/quadknap.py",
         "dsswm/envs/pool_replay.py", "dsswm/stats/prereg.py", "run_r5_cr_main.py"]
_ENV = {}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hashes():
    per = {p: sha_file(CODE / p) for p in FILES if (CODE / p).exists()}
    combined = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return per, combined


def git_info():
    def g(*a):
        return subprocess.run(["git", *a], cwd=str(CODE), capture_output=True, text=True).stdout.strip()
    return {"head": g("rev-parse", "HEAD"), "dirty_files": g("status", "--porcelain", "--", ".").splitlines()[:50]}


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def progress(task, done, total, metric=None):
    (RES / f"{task}_PROGRESS.json").write_text(json.dumps({
        "task_id": task, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def init_env(half, eps):
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", half)
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    _ENV.update(env=env, ctx=ctx, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]), eps=eps)


def job(a):
    name, kw, seed, code_sha = a
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
        out = {"method": name, "params": kw, "seed": int(seed)}
        out.update({f"rs_{k}": v for k, v in s.items()})  # every run_stream summary field, verbatim
        out.update({"N80_pen": int(s["N80"]), "N80_raw": n80_raw, "N80_over_tau": s["N80"] / ctx.tau_R,
                    "completed": bool(s["reached_stop"]), "fwer_event": bool(s["fwer_event"]),
                    "censored": bool(s["censored"]), "cert_k": s["cert_k"], "billing_ok": bool(s["billing_ok"]),
                    "schedule_digest": s["schedule_digest"], "n_cert_curve": curve,
                    "sec": round(time.perf_counter() - t0, 3), "code_sha256": code_sha, "error": None})
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "params": kw, "seed": int(seed), "error": traceback.format_exc()}


def mark_done(task, status, txt):
    pid = RES / f"{task}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{task}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES / f"{task}_DONE").write_text(json.dumps({"task_id": task, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def skipped_summary(task, out, reason, mode):
    out.mkdir(parents=True, exist_ok=True)
    s = {"task_id": task, "mode": mode, "status": "skipped_by_lock", "reason": reason,
         "written_at": datetime.now().isoformat(), "eval_seeds_touched": False}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def run_block(task, mode, out, lock, seeds=None, half=None, write_markers=True):
    et = lock["eval_tasks"][task]
    methods = list(et["methods"])
    if seeds is None:
        seeds = parse_range(et["seeds"])
    if half is None:
        half = et["half"]
    assert et["layer"] == "CR9" and int(et["K"]) == 20
    # rival configs: lock copy, and the file on disk must still hash to the lock value
    rc_file = WS / lock["rival_configs"]["path"]
    if sha_file(rc_file) != lock["rival_configs"]["sha256"]:
        raise RuntimeError("rival_configs.json sha256 differs from lock v5")
    frozen = lock["rival_configs"]["frozen_configs"]
    kws = {m: ({} if m == "FDC" else dict(frozen.get(m) or {})) for m in methods}
    eps = float(json.loads((RES / "r4_gates" / "eps.json").read_text())["eps_star"])
    if abs(eps - float(et["eps"])) > 1e-15:
        raise RuntimeError(f"eps.json {eps} != lock {et['eps']}")
    out.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    sf = out / "start_time.txt"
    if not sf.exists():
        sf.write_text(t_start.isoformat())
    init_env(half, eps)
    ctx = _ENV["ctx"]
    assert abs(ctx.eps - 0.001) < 1e-12 and len(ctx.checkpoints) == 20 and ctx.stop_k == 12 and ctx.Q == 15
    sref = lock["structural_ctx"][f"CR9_{half}"]
    if int(ctx.tau_R) != int(sref["tau_R"]) or np.asarray(ctx.pool_sizes if hasattr(ctx, "pool_sizes") else
                                                          _ENV["env"].pool_sizes).tolist() != sref["pool_sizes"]:
        raise RuntimeError(f"structural ctx differs from lock CR9_{half}")
    per_sha, code_sha = code_hashes()
    print(f"[{task}/{mode}] half={half} tau_R={ctx.tau_R} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) "
          f"methods={methods} code_sha={code_sha[:16]}", flush=True)

    # write-ahead results file: resume from completed rows
    rfile = out / "results.jsonl"
    done = set()
    if rfile.exists():
        keep = []
        for l in rfile.read_text().splitlines():
            try:
                x = json.loads(l)
            except ValueError:
                continue  # torn last line from a crash
            if x.get("error") is None and x.get("code_sha256") == code_sha and (x["method"], x["seed"]) not in done:
                done.add((x["method"], x["seed"]))
                keep.append(l)
        rfile.write_text("".join(k + "\n" for k in keep))  # drop torn / stale-code / duplicate lines
    order = [m for m in ("Peace-rect", "B3-rect") if m in methods] + \
            [m for m in methods if m not in ("Peace-rect", "B3-rect")]
    jobs = [(m, kws[m], s, code_sha) for m in order for s in seeds if (m, s) not in done]
    total = len(methods) * len(seeds)
    print(f"resume: {len(done)} done, {len(jobs)} to run", flush=True)
    if write_markers:
        progress(task, len(done), total)
    errs = []
    n_done = len(done)
    if jobs:
        with get_context("fork").Pool(N_WORKERS) as pool, open(rfile, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                if r.get("error"):
                    errs.append(r)
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if write_markers and (n_done % 10 == 0 or n_done == total):
                    progress(task, n_done, total, {"runs_done": n_done, "errors": len(errs)})
                if n_done % 25 == 0 or n_done == total:
                    print(f"[{n_done}/{total}] last={r['method']} seed={r['seed']} sec={r['sec']} "
                          f"errors={len(errs)}", flush=True)
    if errs:
        (out / "errors.log").write_text("\n\n".join(f"{e['method']} {e['seed']}\n{e['error']}" for e in errs))
        raise RuntimeError(f"{len(errs)} stream jobs failed (see errors.log)")

    # completeness / billing / hashing only -- NO analysis inside the block
    rows = {}
    for l in rfile.read_text().splitlines():
        try:
            x = json.loads(l)
        except ValueError:
            continue
        if x.get("error") is None and x.get("code_sha256") == code_sha:
            rows[(x["method"], x["seed"])] = x
    missing = [(m, s) for m in methods for s in seeds if (m, s) not in rows]
    bill_bad = [(m, s) for (m, s), x in rows.items() if not x["billing_ok"]]
    # canonical (sorted) results file hash, independent of completion order
    canon = "\n".join(json.dumps({k: v for k, v in rows[(m, s)].items() if k not in ("sec", "rs_sec_plan",
                                                                                    "rs_sec_cert", "rs_sec_total")},
                                 sort_keys=True) for m in methods for s in seeds if (m, s) in rows)
    t_end = datetime.now()
    timing = {m: {"sec_mean": float(np.mean([rows[(m, s)]["sec"] for s in seeds if (m, s) in rows] or [0])),
                  "sec_max": float(np.max([rows[(m, s)]["sec"] for s in seeds if (m, s) in rows] or [0]))}
              for m in methods}
    summary = {"task_id": task, "mode": mode, "status": "complete" if not missing and not bill_bad else "incomplete",
               "started_at": sf.read_text().strip(), "ended_at": t_end.isoformat(),
               "wall_min_this_invocation": round((t_end - t_start).total_seconds() / 60, 2),
               "timing_note": "concurrent run (4 workers, BLAS=1; other tasks share the 20-core host): biased up",
               "setting": {"layer": "CR9", "half": half, "eps": ctx.eps, "K": len(ctx.checkpoints),
                           "stop_k": int(ctx.stop_k), "Q": ctx.Q, "delta": ctx.delta, "tau_R": int(ctx.tau_R),
                           "seeds": [seeds[0], seeds[-1]], "n_streams": len(seeds)},
               "methods": methods, "params": kws,
               "lock": {"version": lock["version"], "status": lock["status"], "sha256": lock["sha256"],
                        "git_commit": lock["git_commit"]},
               "n_rows": len(rows), "expected_rows": total, "missing": missing[:50], "n_missing": len(missing),
               "billing_ok_all": not bill_bad, "billing_bad": bill_bad[:50],
               "results_canonical_sha256": hashlib.sha256(canon.encode()).hexdigest(),
               "results_file_sha256": sha_file(rfile), "code_sha256_combined": code_sha, "code_sha256": per_sha,
               "git": git_info(), "per_method_timing_sec": timing,
               "eval_seeds_touched": half == "eval",
               "note": "no analysis inside the block (no ratios / bootstrap); analysis happens in r5_analysis_aggregate"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["r5_cr_main_a", "r5_cr_main_b"])
    ap.add_argument("--mode", required=True, choices=["pilot", "full"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    task = a.task
    out = Path(a.out) if a.out else RES / ("pilots" if a.mode == "pilot" else "full") / task
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    ok, lock = prereg.lock_gate(task, version=5)
    if not ok:
        s = skipped_summary(task, out, lock, a.mode)
        mark_done(task, "success", f"skipped_by_lock: {lock}")
        print(json.dumps(s), flush=True)
        return
    try:
        if a.mode == "full":
            s = run_block(task, "full", out, lock)
        else:
            s = run_block(task, "pilot", out, lock, seeds=DEV_PILOT_SEEDS, half="dev")
        txt = f"{s['status']} rows={s['n_rows']}/{s['expected_rows']} billing_ok={s['billing_ok_all']}"
        mark_done(task, "success" if s["status"] == "complete" else "failed", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done(task, "failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
