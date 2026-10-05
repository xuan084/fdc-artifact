"""r5_cr_factorial: confirmatory evaluation block C4 -- 2 x 2 x 2 factorial (design x union x ledger) + B4-bal.

FULL  (--mode full):  lock gate (dsswm.stats.prereg.lock_gate, version 5) -> on refusal summary.status='skipped_by_lock'
                      and exit; else CR9 eval half, eps* = 0.001, K = 20, eval seeds 30000-30199, methods = the 8
                      FDCAblation cells F[design,union,ledger] + B4-bal (share frozen in rival_configs.json, sha256
                      checked against the lock).
                      BEFORE any evaluation result is read (including this block's own results.jsonl on a resume), the
                      frozen exp/results/r5_gates/c4_model.json is loaded (sha256 checked against the lock) and the
                      per-cell predictions are written to predictions.json with a timestamp:
                        primary   = the frozen file's pred_geomean_primary per cell, verbatim (model M1, dev-calibrated);
                        secondary = the same frozen model form M1 (frozen constant c) applied to the deterministic
                                    equivalent t_det of the CR9 eval half (true cell means, expected design counts),
                                    reported as a transfer sensitivity only.
                      A resume never rewrites predictions.json (its sha256 is re-checked instead).
                      Every (method, stream) row is written ahead to results.jsonl (fsync) with all run_stream fields +
                      code hashes; a restart skips rows already present. No analysis inside the block: the summary
                      reports completeness, billing, errors, the identities F[half,feas,r5] == FDC and
                      F[pool,qstar,r4] == QFC-pool against whatever main-block rows exist, and the results sha256.
PILOT (--mode pilot): same code path on DEV streams 900-909 (never eval seeds): smoke + timing, lock-gate refusal path
                      on tampered lock copies, structural check of the eval-half ctx against the lock, prediction code
                      path on the DEV half (secondary t_det must reproduce the frozen dev t_det), reproducibility vs
                      r5_t1_reconcile (all 9 configs, dev 900-909) and vs the r5_cr_main_b pilot (FDC, QFC-pool).
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
import importlib.util
import json
import math
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
TASK = "r5_cr_factorial"
EVAL_SEEDS = list(range(30000, 30200))
PILOT_SEEDS = list(range(900, 910))
N_WORKERS = 4

from dsswm.baselines.fdc import FACTORIAL_CELLS, FDCAblation  # noqa: E402
from dsswm.stats import prereg  # noqa: E402

CELL_NAME = {c: f"F[{c[0]},{c[1]},{c[2]}]" for c in FACTORIAL_CELLS}
NAME_CELL = {v: k for k, v in CELL_NAME.items()}
METHODS = [CELL_NAME[c] for c in FACTORIAL_CELLS] + ["B4-bal"]
IDENTITY = {"F[half,feas,r5]": "FDC", "F[pool,qstar,r4]": "QFC-pool"}
# r5_t1_reconcile config labels of the same objects (dev reference)
T1_LABEL = {**{n: n for n in METHODS}, "F[half,feas,r5]": "FDC",
            "F[half,qstar,r4]": "L4b QFC-half@K20 = FDCx[half,qstar,r4]", "B4-bal": "B4-bal[share=0.50]"}
T1_RECONCILE = CODE / "run_r5_t1_reconcile.py"

_ENV = {}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_digest(lock):
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
def method_kwargs(lock):
    rc_path = WS / lock["rival_configs"]["path"]
    got = sha_file(rc_path)
    if got != lock["rival_configs"]["sha256"]:
        raise RuntimeError(f"rival_configs.json sha256 {got} != lock {lock['rival_configs']['sha256']}")
    rc = json.loads(rc_path.read_text())
    kw = {n: {"design": c[0], "union": c[1], "ledger": c[2]} for c, n in CELL_NAME.items()}
    kw["B4-bal"] = dict(rc["frozen_configs"]["B4-bal"])
    return kw


def check_lock_task(lock):
    et = lock["eval_tasks"][TASK]
    assert [int(x) for x in et["seeds"].split("-")] == [EVAL_SEEDS[0], EVAL_SEEDS[-1]], et["seeds"]
    assert et["methods"] == ["FDCAblation x8", "B4-bal"], et["methods"]
    assert et["layer"] == "CR9" and et["half"] == "eval" and et["K"] == 20 and int(et["n_streams"]) == 200
    fac = lock["c4"]["factorial"]
    assert [tuple(s.split(",")) for s in fac["cells"]] == list(FACTORIAL_CELLS), fac["cells"]
    assert {f"F[{k}]": v for k, v in fac["identities"].items()} == IDENTITY, fac["identities"]
    return et


def test_lock_refusal_paths(lock_file: Path):
    """Pilot only: the gate must refuse tampered locks and accept the real one (nothing is written to plan/)."""
    real = json.loads(lock_file.read_text())
    out, cases = {}, {}
    a = copy.deepcopy(real)
    a["status"] = "locked_no_eval"
    a["sha256"] = prereg.canonical_hash(a)
    cases["status_locked_no_eval"] = a
    b = copy.deepcopy(real)
    b["cr_n_streams"] = 400
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
    out["all_pass"] = all(v["refused"] for k, v in out.items() if k != "real_lock") and out["real_lock"]["accepted"]
    return out


# ------------------------------------------------------------------------------------------------ C4 predictions
def load_frozen_model(lock):
    pm = lock["c4"]["prediction_model"]
    p = WS / pm["path"]
    got = sha_file(p)
    if got != pm["sha256_file"]:
        raise RuntimeError(f"c4_model.json sha256 {got} != lock {pm['sha256_file']}")
    m = json.loads(p.read_text())
    body = {k: v for k, v in m.items() if k not in ("sha256", "sha256_scope")}
    internal = hashlib.sha256(json.dumps(body, sort_keys=True, indent=1).encode()).hexdigest()
    if internal != pm["sha256_internal"] or m["sha256"] != internal:
        raise RuntimeError("c4_model.json internal sha256 mismatch")
    return m, got


def _load_t1():
    spec = importlib.util.spec_from_file_location("r5_t1_reconcile_c4", T1_RECONCILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_predictions(lock, half, env, ctx):
    """Frozen-model per-cell predictions. Reads no stream result of any kind."""
    m, file_sha = load_frozen_model(lock)
    t1_sha16 = sha_file(T1_RECONCILE)[:16]
    if t1_sha16 != m["code_sha256_16"]["run_r5_t1_reconcile.py"]:
        raise RuntimeError(f"run_r5_t1_reconcile.py sha16 {t1_sha16} != frozen {m['code_sha256_16']}")
    t1 = _load_t1()
    mu = env.true_mu("visit")
    c1 = float(m["model"]["variants"]["M1"]["c"])
    cells = {}
    t0 = time.perf_counter()
    for c in FACTORIAL_CELLS:
        name = CELL_NAME[c]
        fz = [v for v in m["model"]["cells"].values() if tuple(v["cell"]) == c]
        assert len(fz) == 1, (c, len(fz))
        fz = fz[0]
        prm = t1.cell_params(c, ctx)
        td = float(t1.t_det(env, ctx, prm, mu))
        cells[name] = {"cell": list(c), "pred_geomean_N80_primary": float(fz["pred_geomean_primary"]),
                       "pred_log_primary": math.log(float(fz["pred_geomean_primary"])),
                       "frozen_dev_t_det": float(fz["t_det"]),
                       f"t_det_{half}": td, f"t_det_{half}_over_tau": td / float(ctx.tau_R),
                       "pred_log_secondary_M1_transfer": math.log(td) + c1,
                       "pred_geomean_N80_secondary_M1_transfer": math.exp(math.log(td) + c1),
                       "beta": prm["beta"], "x_v": prm["x_v"]}
    # Shapley/design-share prediction implied by the primary predictions (log scale, FDC vs QFC-pool cell)
    lp = {CELL_NAME[c]: cells[CELL_NAME[c]]["pred_log_primary"] for c in FACTORIAL_CELLS}
    return {"task_id": TASK, "written_at": datetime.now().isoformat(),
            "written_before_any_eval_result_read": True,
            "model_file": lock["c4"]["prediction_model"]["path"], "model_file_sha256": file_sha,
            "model_sha256_internal": m["sha256"], "model_frozen_at": m["frozen_at"], "primary_variant": m["model"]["primary"],
            "M1_c": c1, "error_limit_abs_log": m["error_limit_abs_log"],
            "rule": lock["c4"]["prediction_model"]["rule"],
            "primary_definition": "frozen c4_model.json pred_geomean_primary per cell, verbatim (dev-calibrated M1)",
            "secondary_definition": f"frozen M1 form (log g = log t_det + c, c frozen) with t_det recomputed on the "
                                    f"CR9 {half} half (true cell means, expected design counts); sensitivity only",
            "secondary_half": half, "t_det_sec": round(time.perf_counter() - t0, 1),
            "cells": cells, "predicted_shapley_primary": shapley(lp),
            "t1_reconcile_sha16": t1_sha16}


def shapley(lp):
    """Shapley over {D, U, L} of log N80(F[pool,qstar,r4]) - log N80(F[half,feas,r5]) (positive = speed-up)."""
    lv = {"D": ("pool", "half"), "U": ("qstar", "feas"), "L": ("r4", "r5")}
    fs = ["D", "U", "L"]

    def v(S):
        c = tuple(lv[f][1 if f in S else 0] for f in fs)
        return lp[CELL_NAME[c]]
    base = v(set())
    tot = base - v(set(fs))
    phi = {}
    for f in fs:
        others = [g for g in fs if g != f]
        s = 0.0
        for r in range(3):
            for sub in __import__("itertools").combinations(others, r):
                w = math.factorial(len(sub)) * math.factorial(2 - len(sub)) / 6
                s += w * (v(set(sub)) - v(set(sub) | {f}))
        phi[f] = s
    return {"total_log_speedup": tot, "phi": phi, "design_share": phi["D"] / tot if tot else None}


def write_predictions_once(out: Path, preds):
    p = out / "predictions.json"
    if p.exists():
        old = json.loads(p.read_text())
        same = all(abs(old["cells"][n]["pred_log_primary"] - preds["cells"][n]["pred_log_primary"]) < 1e-12
                   and abs(old["cells"][n]["pred_log_secondary_M1_transfer"]
                           - preds["cells"][n]["pred_log_secondary_M1_transfer"]) < 1e-9 for n in old["cells"])
        if not same:
            raise RuntimeError("predictions.json exists and differs from a recomputation; refusing to overwrite")
        return old, sha_file(p), False
    p.write_text(json.dumps(preds, indent=1))
    (out / "predictions.sha256").write_text(sha_file(p) + "  predictions.json\n")
    return preds, sha_file(p), True


# ------------------------------------------------------------------------------------------------ workers
def init_env(half, eps):
    from dsswm.envs.pool_replay import PoolReplayEnv
    from dsswm.streams import frontier as fr
    from dsswm.streams.frontier_runner import build_ctx, true_policy_values
    env = PoolReplayEnv("CR9", half)
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    _ENV.update(env=env, ctx=ctx, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]), eps=eps)


def make(name, kw):
    if name in NAME_CELL:
        return FDCAblation(kw["design"], kw["union"], kw["ledger"])
    from dsswm.streams import r5_registry as reg
    return reg.make_method(name, **kw)


def job(a):
    from dsswm.streams.frontier_runner import run_stream
    name, kw, seed, cd = a
    ctx = _ENV["ctx"]
    try:
        m = make(name, kw)
        t0 = time.perf_counter()
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=_ENV["J"],
                             J_star=_ENV["Js"], keep_U=False)
        ck = ctx.checkpoints
        n80_raw = int(ck[s["k_stop"]]) if s["reached_stop"] else int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))
        return {"task_id": TASK, "method": name, "params": kw, "seed": seed, "method_name": m.name,
                "validity": m.validity, "N80_pen": int(s["N80"]), "N80_raw": n80_raw,
                "completed": bool(s["reached_stop"]), "fwer_event": bool(s["fwer_event"]), "cert_k": s["cert_k"],
                "run_stream": s, "n_cert_curve": curve, "sec": round(time.perf_counter() - t0, 3), "code": cd,
                "written_at": datetime.now().isoformat(), "error": None}
    except Exception:  # noqa: BLE001
        return {"task_id": TASK, "method": name, "params": kw, "seed": seed, "error": traceback.format_exc()}


def load_done(rfile: Path):
    done = {}
    if rfile.exists():
        for line in rfile.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if not x.get("error"):
                done[(x["method"], x["seed"])] = x
    return done


def run_block(seeds, kws, out: Path, cd, label):
    rfile = out / "results.jsonl"
    done = load_done(rfile)
    if rfile.exists():
        rfile.write_text("".join(json.dumps(x) + "\n" for x in done.values()))
    jobs = [(m, kws[m], s, cd) for m in METHODS for s in seeds if (m, s) not in done]
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
                if n % 20 == 0 or n == total:
                    progress(n, total, {"rows_done": len(done), "errors": len(errs)})
                    print(f"[{label} {n}/{total}] last={r['method']} seed={r['seed']} errors={len(errs)}", flush=True)
    return done, errs


def completeness(done, seeds):
    missing = [(m, s) for m in METHODS for s in seeds if (m, s) not in done]
    rows = [done[(m, s)] for m in METHODS for s in seeds if (m, s) in done]
    bill_bad = [(r["method"], r["seed"]) for r in rows if not r["run_stream"]["billing_ok"]]
    return rows, missing, bill_bad


def _digest(x):
    """schedule digest from either row schema (nested run_stream, or r5_cr_main_a's flat schedule_digest/rs_*)."""
    if "run_stream" in x:
        return x["run_stream"]["schedule_digest"]
    return x.get("schedule_digest", x.get("rs_schedule_digest"))


def identity_vs(done, ref_rows, seeds):
    """F[half,feas,r5] vs FDC and F[pool,qstar,r4] vs QFC-pool, stream by stream (N80_pen, cert_k, schedule digest)."""
    out = {}
    for cell, other in IDENTITY.items():
        n = mism = 0
        bad = []
        for s in seeds:
            a, b = done.get((cell, s)), ref_rows.get((other, s))
            if a is None or b is None:
                continue
            n += 1
            if (a["N80_pen"] != b["N80_pen"] or a["cert_k"] != b["cert_k"]
                    or _digest(a) != _digest(b)):
                mism += 1
                bad.append({"seed": s, "cell": a["N80_pen"], other: b["N80_pen"]})
        out[f"{cell}=={other}"] = {"n_compared": n, "n_mismatch": mism, "mismatches": bad[:10]}
    return out


def read_rows(p: Path):
    ref = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if not x.get("error"):
                ref[(x["method"], x["seed"])] = x
    return ref


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
    et = check_lock_task(lock)
    kws = method_kwargs(lock)
    cd = code_digest(lock)
    init_env("eval", float(et["eps"]))
    ctx = _ENV["ctx"]
    sc = lock["structural_ctx"]["CR9_eval"]
    assert int(ctx.tau_R) == sc["tau_R"] and ctx.Q == sc["Q"] and ctx.stop_k == sc["stop_k"]
    assert [int(x) for x in ctx.checkpoints] == lock["fdc_spec"]["checkpoints"]["CR9_eval"]
    # predictions BEFORE any evaluation result (incl. this block's own results.jsonl) is read
    preds, pred_sha, fresh = write_predictions_once(out, make_predictions(lock, "eval", _ENV["env"], ctx))
    print(f"[predictions] {'written' if fresh else 'kept (resume)'} sha256={pred_sha} at {preds['written_at']}",
          flush=True)
    if not (out / "start_time.txt").exists():
        (out / "start_time.txt").write_text(t_start.isoformat())
    done, errs = run_block(EVAL_SEEDS, kws, out, cd, "eval")
    rows, missing, bill_bad = completeness(done, EVAL_SEEDS)
    main_rows = {**read_rows(RES / "full/r5_cr_main_a/results.jsonl"), **read_rows(RES / "full/r5_cr_main_b/results.jsonl")}
    ident = identity_vs(done, main_rows, EVAL_SEEDS)
    validity = {m: sorted({done[(m, s)]["validity"] for s in EVAL_SEEDS if (m, s) in done}) for m in METHODS}
    t_end = datetime.now()
    status = "complete" if (not missing and not bill_bad and not errs) else "incomplete"
    summ = {"task_id": TASK, "status": status, "mode": "full (confirmatory eval block)", "lock_version": 5,
            "lock_sha256": lock["sha256"], "freeze_commit": lock["git_commit"], "git": git_head(), "code": cd,
            "setting": {"layer": "CR9", "half": "eval", "eps": ctx.eps, "K": len(ctx.checkpoints),
                        "stop_k": int(ctx.stop_k), "Q": ctx.Q, "delta": ctx.delta, "tau_R": int(ctx.tau_R),
                        "seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n_streams": len(EVAL_SEEDS)},
            "methods": METHODS, "params": kws, "validity": validity,
            "predictions": {"path": str((out / "predictions.json").relative_to(WS)), "sha256": pred_sha,
                            "written_at": preds["written_at"], "block_started_at": (out / "start_time.txt").read_text()},
            "n_rows": len(rows), "n_expected": len(METHODS) * len(EVAL_SEEDS),
            "missing": missing[:50], "billing_not_ok": bill_bad[:50], "errors_this_run": len(errs),
            "identity_vs_main_blocks": ident,
            "identity_note": "computed against the main-block rows present at the end of this run; the analysis task "
                             "re-checks once r5_cr_main_a/b are complete",
            "results_sha256": sha_file(out / "results.jsonl"),
            "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
            "wall_min_this_run": round((t_end - t_start).total_seconds() / 60, 1),
            "timing_note": "concurrent run (4 workers; other r5 eval blocks share the 20-core host)",
            "analysis": "none inside the block (pre-registered SAP is applied by r5_analysis_aggregate)"}
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
    et = check_lock_task(lock)
    kws = method_kwargs(lock)
    cd = code_digest(lock)
    from dsswm.envs.pool_replay import PoolReplayEnv
    from dsswm.streams import frontier as fr
    from dsswm.streams.frontier_runner import build_ctx
    ev = PoolReplayEnv("CR9", "eval")
    ectx = build_ctx(ev, fr.cr_problems("visit"), float(et["eps"]))
    sc = lock["structural_ctx"]["CR9_eval"]
    eval_struct = {"tau_R_match": int(ectx.tau_R) == sc["tau_R"], "Q_match": ectx.Q == sc["Q"],
                   "checkpoints_match": [int(x) for x in ectx.checkpoints] == lock["fdc_spec"]["checkpoints"]["CR9_eval"],
                   "pool_sizes_match": np.asarray(ev.pool_sizes).tolist() == sc["pool_sizes"],
                   "seeds_in_lock": et["seeds"], "cells_and_identities_match_lock": True}
    del ev, ectx
    init_env("dev", float(et["eps"]))
    ctx = _ENV["ctx"]
    assert [int(x) for x in ctx.checkpoints] == lock["fdc_spec"]["checkpoints"]["CR9_dev"]
    # prediction code path on the DEV half: secondary t_det must reproduce the frozen dev t_det
    pdir = out / "pred_dryrun"
    pdir.mkdir(exist_ok=True)
    for f in ("predictions.json", "predictions.sha256"):
        if (pdir / f).exists():
            (pdir / f).unlink()
    preds, pred_sha, _ = write_predictions_once(pdir, make_predictions(lock, "dev", _ENV["env"], ctx))
    tdet_err = {n: abs(math.log(v["t_det_dev"] / v["frozen_dev_t_det"])) for n, v in preds["cells"].items()}
    _, _, fresh2 = write_predictions_once(pdir, make_predictions(lock, "dev", _ENV["env"], ctx))  # resume path
    pred_check = {"t_det_dev_reproduces_frozen": max(tdet_err.values()) < 1e-9, "max_abs_log_tdet_diff": max(tdet_err.values()),
                  "resume_keeps_file": not fresh2, "sha256": pred_sha, "t_det_sec": preds["t_det_sec"],
                  "predicted_shapley_primary": preds["predicted_shapley_primary"]}
    rfile = out / "results.jsonl"
    if rfile.exists() and os.environ.get("PILOT_FRESH", "1") == "1":
        rfile.unlink()
    t_run = time.perf_counter()
    done, errs = run_block(PILOT_SEEDS, kws, out, cd, "pilot-dev")
    wall_run = time.perf_counter() - t_run
    rows, missing, bill_bad = completeness(done, PILOT_SEEDS)
    # reproducibility vs r5_t1_reconcile (dev, all 9 configs) and vs r5_cr_main_b pilot (FDC / QFC-pool)
    t1 = {}
    for line in (RES / "pilots/r5_t1_reconcile/results.jsonl").read_text().splitlines():
        x = json.loads(line)
        t1[(x["config"], x["seed"])] = x
    mism_t1, n_t1 = [], 0
    for r in rows:
        ref = t1.get((T1_LABEL[r["method"]], r["seed"]))
        if ref is None:
            continue
        n_t1 += 1
        if (r["N80_pen"] != ref["N80_pen"] or r["run_stream"]["k_stop"] != ref["k_stop"]
                or r["run_stream"]["schedule_digest"] != ref["schedule_digest"]):
            mism_t1.append({"method": r["method"], "seed": r["seed"], "now": r["N80_pen"], "t1": ref["N80_pen"]})
    ident = identity_vs(done, read_rows(RES / "pilots/r5_cr_main_b/results.jsonl"), PILOT_SEEDS)
    ident_ok = all(v["n_compared"] == len(PILOT_SEEDS) and v["n_mismatch"] == 0 for v in ident.values())
    sec = {m: float(np.mean([done[(m, s)]["sec"] for s in PILOT_SEEDS])) for m in METHODS}
    smax = {m: float(np.max([done[(m, s)]["sec"] for s in PILOT_SEEDS])) for m in METHODS}
    loads = np.zeros(N_WORKERS)
    for m in sorted(METHODS, key=lambda m: -sec[m]):
        for _ in EVAL_SEEDS:
            loads[np.argmin(loads)] += sec[m]
    proj_min = float(loads.max() / 60 * 1.2 + 2 + preds["t_det_sec"] / 60)
    per_method = {}
    for m in METHODS:
        rr = [done[(m, s)] for s in PILOT_SEEDS]
        per_method[m] = {"params": kws[m], "validity": sorted({x["validity"] for x in rr}),
                         "method_name": rr[0]["method_name"], "sec_mean": round(sec[m], 2), "sec_max": round(smax[m], 2),
                         "geomean_N80_pen_dev": float(np.exp(np.mean(np.log([x["N80_pen"] for x in rr])))),
                         "completed": sum(x["completed"] for x in rr), "fwer_streams": sum(x["fwer_event"] for x in rr),
                         "billing_ok_all": all(x["run_stream"]["billing_ok"] for x in rr)}
    samples = [{k: done[(m, s)][k] for k in ("method", "seed", "N80_pen", "N80_raw", "completed", "fwer_event", "cert_k")}
               | {"decided_pi": done[(m, s)]["run_stream"]["decided_pi"],
                  "billing_ok": done[(m, s)]["run_stream"]["billing_ok"]}
               for s in (900, 905, 909) for m in ("F[half,feas,r5]", "F[pool,qstar,r4]", "F[pool,feas,r5]", "B4-bal")]
    crit = {"billing_ok_all": not bill_bad, "no_exceptions": not errs and not missing,
            "projected_full_min_le_55": proj_min <= 55, "lock_assertion_path_tested": lock_tests["all_pass"],
            "eval_structure_matches_lock": all(v for k, v in eval_struct.items() if k != "seeds_in_lock"),
            "reproduces_t1_reconcile": n_t1 == len(rows) and not mism_t1,
            "identities_vs_main_b_pilot": ident_ok,
            "prediction_path_ok": pred_check["t_det_dev_reproduces_frozen"] and pred_check["resume_keeps_file"]}
    go = all(crit.values())
    t_end = datetime.now()
    summ = {"task_id": TASK, "mode": "pilot (dev streams 900-909; eval seeds untouched)",
            "go_no_go": "GO" if go else "NO_GO", "criteria": crit, "lock_gate_tests": lock_tests,
            "eval_structure_check": eval_struct, "prediction_dryrun_dev": pred_check,
            "projected_full_block_min": round(proj_min, 1),
            "projection_rule": "mean dev sec/stream per method x 200 streams, LPT on 4 workers, x1.2 + 2 min + t_det time",
            "pilot_run_wall_min": round(wall_run / 60, 2), "n_rows": len(rows), "n_expected": len(METHODS) * len(PILOT_SEEDS),
            "repro_vs_t1_reconcile": {"n_compared": n_t1, "n_mismatch": len(mism_t1), "mismatches": mism_t1[:20]},
            "identity_vs_main_b_pilot": ident,
            "per_method_dev": per_method, "samples": samples, "code": cd, "git": git_head(),
            "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
            "timing_note": "concurrent run (4 workers; other r5 eval-block pilots share the host)",
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
            txt = f"pilot {s['go_no_go']} | proj full {s['projected_full_block_min']} min | criteria={s['criteria']}"
            mark_done("success", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
