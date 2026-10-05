"""r5_cr12_scale: supporting scale layer CR12 (|Pi| = 4096), FDC vs rigorous rivals B1 / B4 / B4-bal / Hait-SW.

FULL  (--mode full):  lock gate (dsswm.stats.prereg.lock_gate, version 5) -> on refusal summary.status='skipped_by_lock'
                      and exit; else CR12 eval half, eps* = 0.001, K = 20, eval seeds 30000-30049, methods
                      ['FDC', 'B1', 'B4', 'B4-bal', 'Hait-SW'], rival configs from the lock (rival_configs.json, sha256
                      checked). FDC ledger computed by FDCMethod.setup(ctx) on CR12 (sum_q |Pi_{B_q}| recomputed) and
                      asserted equal to fdc_spec.ledger.CR12_eval (beta / x_v to 1e-9). Every (method, stream) row is
                      written ahead to results.jsonl (fsync) with all run_stream summary fields + code hashes + the
                      per-checkpoint certificate wall time in ms (scale evidence); a restart skips rows already present.
                      Supporting result: not a gate, not in C1, no analysis inside the block.
PILOT (--mode pilot): same code path on CR12 DEV streams 900-919 (never eval seeds; 5 methods x 20 = 100 runs): smoke +
                      timing, lock-gate refusal path tested on tampered copies of the lock, structural check of the
                      CR12 eval-half ctx against the lock (no truth, no streams), determinism re-run of a subset.
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
TASK = "r5_cr12_scale"
METHODS = ["FDC", "B1", "B4", "B4-bal", "Hait-SW"]
LAYER = "CR12"
EVAL_SEEDS = list(range(30000, 30050))
PILOT_SEEDS = list(range(900, 920))
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
    env = PoolReplayEnv(LAYER, half)
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
        cert_ms = []
        _orig = m.certify

        def _timed(c, st, _o=_orig):
            t = time.perf_counter()
            r = _o(c, st)
            cert_ms.append(round((time.perf_counter() - t) * 1000, 2))
            return r
        m.certify = _timed
        t0 = time.perf_counter()
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=_ENV["J"],
                             J_star=_ENV["Js"], keep_U=False)
        ck = ctx.checkpoints
        n80_raw = int(ck[s["k_stop"]]) if s["reached_stop"] else int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))
        return {"task_id": TASK, "method": name, "params": kw, "seed": seed, "N80_pen": int(s["N80"]),
                "N80_raw": n80_raw, "completed": bool(s["reached_stop"]), "fwer_event": bool(s["fwer_event"]),
                "cert_k": s["cert_k"], "run_stream": s, "n_cert_curve": curve, "cert_ms_per_checkpoint": cert_ms,
                "fdc_params": ({k: m.params[k] for k in ("L1", "x_v")} if name == "FDC" else None),
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
    order = ["B4-bal", "B1"] + [m for m in METHODS if m not in ("B4-bal", "B1")]
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
def check_ledger(ctx, lock, key):
    """FDC ledger recomputed from ctx (sum_q |Pi_{B_q}| on CR12) vs the locked CR12_eval ledger."""
    from dsswm.baselines.fdc import FDCMethod, fdc_ledger
    led = fdc_ledger(ctx)
    m = FDCMethod()
    m.setup(ctx)
    ref = lock["fdc_spec"]["ledger"]["CR12_eval"]
    return {"ctx": key, "union_size": int(led["union_size"]), "beta": float(m.params["L1"]), "x_v": float(m.params["x_v"]),
            "n_policies": int(ctx.P), "lock_union_size": ref["union_size_sum_q_Pi_Bq"], "lock_beta": ref["beta"],
            "lock_x_v": ref["x_v"], "union_match": int(led["union_size"]) == ref["union_size_sum_q_Pi_Bq"],
            "beta_match_1e9": abs(m.params["L1"] - ref["beta"]) < 1e-9, "x_v_match_1e9": abs(m.params["x_v"] - ref["x_v"]) < 1e-9,
            "n_policies_match": int(ctx.P) == ref["n_policies"] == 4096}


def struct_check(ctx, env, sc):
    return {"tau_R_match": int(ctx.tau_R) == sc["tau_R"], "Q_match": ctx.Q == sc["Q"], "stop_k_match": int(ctx.stop_k) == sc["stop_k"],
            "K_match": len(ctx.checkpoints) == sc["K"], "n_policies_match": int(ctx.P) == sc["n_policies"],
            "S_A_match": int(ctx.S * ctx.A) == sc["S_A"],
            "pool_sizes_match": np.asarray(env.pool_sizes).tolist() == sc["pool_sizes"]}


def ms_stats(rows):
    out = {}
    for m in METHODS:
        v = [x for r in rows if r["method"] == m for x in r.get("cert_ms_per_checkpoint", [])]
        if v:
            out[m] = {"n_checkpoints": len(v), "ms_mean": round(float(np.mean(v)), 2),
                      "ms_median": round(float(np.median(v)), 2), "ms_max": round(float(np.max(v)), 2)}
    return out


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
    assert et["methods"] == METHODS and et["layer"] == LAYER and et["half"] == "eval" and et["K"] == 20
    kws = rival_kwargs(lock)
    cd = code_digest(lock)
    init_env("eval", float(et["eps"]))
    ctx = _ENV["ctx"]
    sc = lock["structural_ctx"]["CR12_eval"]
    st = struct_check(ctx, _ENV["env"], sc)
    assert all(st.values()), st
    led = check_ledger(ctx, lock, "CR12_eval")
    assert led["union_match"] and led["beta_match_1e9"] and led["x_v_match_1e9"] and led["n_policies_match"], led
    (out / "start_time.txt").write_text(t_start.isoformat())
    done, errs = run_block(EVAL_SEEDS, kws, out, cd, "eval")
    rows, missing, bill_bad = completeness(done, EVAL_SEEDS, kws)
    t_end = datetime.now()
    status = "complete" if (not missing and not bill_bad and not errs) else "incomplete"
    summ = {"task_id": TASK, "status": status, "mode": "full (supporting eval block; not a gate, not in C1)",
            "lock_version": 5, "lock_sha256": lock["sha256"], "freeze_commit": lock["git_commit"], "git": git_head(),
            "code": cd, "fdc_ledger_check": led, "structural_check": st,
            "setting": {"layer": LAYER, "half": "eval", "eps": ctx.eps, "K": len(ctx.checkpoints),
                        "stop_k": int(ctx.stop_k), "Q": ctx.Q, "delta": ctx.delta, "tau_R": int(ctx.tau_R),
                        "n_policies": int(ctx.P), "checkpoints": [int(x) for x in ctx.checkpoints],
                        "seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n_streams": len(EVAL_SEEDS)},
            "methods": METHODS, "params": kws, "n_rows": len(rows), "n_expected": len(METHODS) * len(EVAL_SEEDS),
            "missing": missing[:50], "billing_not_ok": bill_bad[:50], "errors_this_run": len(errs),
            "cert_ms_per_checkpoint": ms_stats(rows),
            "results_sha256": sha_file(out / "results.jsonl"),
            "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
            "wall_min_this_run": round((t_end - t_start).total_seconds() / 60, 1),
            "timing_note": "concurrent run (4 workers; other r5 eval blocks share the 20-core host); ms timings biased up",
            "analysis": "none inside the block (supporting; summarised by the analysis task)"}
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
    # structural-only check of the CR12 eval half (no truth, no outcomes, no streams)
    from dsswm.envs.pool_replay import PoolReplayEnv
    from dsswm.streams import frontier as fr
    from dsswm.streams.frontier_runner import build_ctx
    ev = PoolReplayEnv(LAYER, "eval")
    ectx = build_ctx(ev, fr.cr_problems("visit"), float(et["eps"]))
    eval_struct = struct_check(ectx, ev, lock["structural_ctx"]["CR12_eval"])
    eval_struct["methods_match"] = et["methods"] == METHODS
    eval_struct["layer_half_K_match"] = et["layer"] == LAYER and et["half"] == "eval" and et["K"] == 20
    eval_struct["seeds_match"] = [int(x) for x in et["seeds"].split("-")] == [EVAL_SEEDS[0], EVAL_SEEDS[-1]]
    eval_ledger = check_ledger(ectx, lock, "CR12_eval")
    del ev, ectx
    # dev smoke run on the identical code path
    init_env("dev", float(et["eps"]))
    ctx = _ENV["ctx"]
    dev_ledger = check_ledger(ctx, lock, "CR12_dev")
    rfile = out / "results.jsonl"
    if rfile.exists() and os.environ.get("PILOT_FRESH", "1") == "1":
        rfile.unlink()
    t_run = time.perf_counter()
    done, errs = run_block(PILOT_SEEDS, kws, out, cd, "pilot-dev")
    wall_run = time.perf_counter() - t_run
    rows, missing, bill_bad = completeness(done, PILOT_SEEDS, kws)
    # determinism: re-run seeds 900 / 901 for every method in-process and compare
    mism = []
    rep_seeds = PILOT_SEEDS[:2]
    for m in METHODS:
        for s in rep_seeds:
            r2 = job((m, kws[m], s, cd))
            r1 = done.get((m, s))
            if r2.get("error") or r1 is None or r2["N80_pen"] != r1["N80_pen"] or r2["cert_k"] != r1["cert_k"] \
                    or r2["run_stream"]["schedule_digest"] != r1["run_stream"]["schedule_digest"] \
                    or r2["run_stream"]["decided_pi"] != r1["run_stream"]["decided_pi"]:
                mism.append({"method": m, "seed": s, "err": r2.get("error")})
    # timing projection for the full block (50 streams, 4 workers, LPT list schedule)
    sec = {m: float(np.mean([done[(m, s)]["sec"] for s in PILOT_SEEDS])) for m in METHODS}
    smax = {m: float(np.max([done[(m, s)]["sec"] for s in PILOT_SEEDS])) for m in METHODS}
    loads = np.zeros(N_WORKERS)
    for m in sorted(METHODS, key=lambda m: -sec[m]):
        for _ in EVAL_SEEDS:
            loads[np.argmin(loads)] += sec[m]
    proj_min = float(loads.max() / 60 * 1.2 + max(smax.values()) / 60 + 2)
    per_method = {}
    for m in METHODS:
        rr = [done[(m, s)] for s in PILOT_SEEDS]
        per_method[m] = {"params": kws[m], "sec_mean": round(sec[m], 2), "sec_max": round(smax[m], 2),
                         "geomean_N80_pen_dev": float(np.exp(np.mean(np.log([x["N80_pen"] for x in rr])))),
                         "geomean_N80_raw_dev": float(np.exp(np.mean(np.log([x["N80_raw"] for x in rr])))),
                         "completed": sum(x["completed"] for x in rr), "fwer_streams": sum(x["fwer_event"] for x in rr),
                         "billing_ok_all": all(x["run_stream"]["billing_ok"] for x in rr)}
    samples = [{k: done[(m, s)][k] for k in ("method", "seed", "N80_pen", "N80_raw", "completed", "fwer_event", "cert_k")}
               | {"decided_pi": done[(m, s)]["run_stream"]["decided_pi"],
                  "billing_ok": done[(m, s)]["run_stream"]["billing_ok"],
                  "cert_ms_per_checkpoint": done[(m, s)]["cert_ms_per_checkpoint"]}
               for s in (900, 910, 919) for m in METHODS]
    crit = {"billing_ok_all": not bill_bad, "no_exceptions": not errs and not missing,
            "projected_full_min_le_55": proj_min <= 55, "lock_assertion_path_tested": lock_tests["all_pass"],
            "eval_structure_matches_lock": all(eval_struct.values()),
            "eval_ledger_matches_lock": all(eval_ledger[k] for k in ("union_match", "beta_match_1e9", "x_v_match_1e9",
                                                                     "n_policies_match")),
            "deterministic_rerun": not mism, "n_runs_ge_100": len(rows) >= 100}
    go = all(crit.values())
    t_end = datetime.now()
    summ = {"task_id": TASK, "mode": "pilot (CR12 dev streams 900-919; eval seeds untouched)",
            "go_no_go": "GO" if go else "NO_GO", "criteria": crit, "lock_gate_tests": lock_tests,
            "eval_structure_check": eval_struct, "eval_ledger_check": eval_ledger, "dev_ledger": dev_ledger,
            "dev_setting": {"tau_R": int(ctx.tau_R), "Q": ctx.Q, "stop_k": int(ctx.stop_k), "n_policies": int(ctx.P),
                            "checkpoints": [int(x) for x in ctx.checkpoints]},
            "projected_full_block_min": round(proj_min, 1),
            "projection_rule": "mean dev sec/stream per method x 50 streams, LPT on 4 workers, x1.2 + max/60 + 2 min",
            "pilot_run_wall_min": round(wall_run / 60, 2), "n_rows": len(rows), "n_expected": len(METHODS) * len(PILOT_SEEDS),
            "determinism_rerun": {"seeds": rep_seeds, "n_compared": len(METHODS) * len(rep_seeds), "n_mismatch": len(mism),
                                  "mismatches": mism[:10]},
            "per_method_dev": per_method, "cert_ms_per_checkpoint_dev": ms_stats(rows), "samples": samples,
            "code": cd, "git": git_head(), "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
            "timing_note": "concurrent run (4 workers; r5_cr_factorial / fwer_audit / k60_sens share the host)",
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
