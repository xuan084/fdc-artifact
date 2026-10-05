"""Lock-v7 ADDENDUM runner: confirmatory test of FDC-BF on fresh eval seeds (new file; v5 / v6 files untouched).

  A  CONFIRMATORY  CR9 eval half, eps 0.001, NEW seeds 33000-33199; one run per stream with stop_k = 15 (N80_pen, x12
     and N100_pen read off the same trajectory; the 12/15 prefix is identical to a stop-at-12 run by construction).
  B  DESCRIPTIVE   Lenta LR9 eval half (outcome-exposed log), NEW seeds 34000-34199, eps 0.002 / 0.003 / 0.004, stop
     at 12/15 (as v6 block B).

Usage (cwd = exp/code)
  run_r5s_v7.py --task v7a_pilot | v7b_pilot            # dev runner check, seeds 950-999 (dev seeds only)
  run_r5s_v7.py --task v7a_full_a | v7a_full_b | v7b_full_a | v7b_full_b   # EVAL: refused unless the v7 addendum is
                                                                           # locked (dsswm.stats.prereg_v7)
  run_r5s_v7.py --task <task> --replica                 # R1 (first 10 seeds re-run) + R2 report
  run_r5s_v7.py --analyse A|B [--dev]
Every (method, eps, seed) row is written ahead to results.jsonl (fsync), resumable.  CPU only, BLAS threads 1,
N_WORKERS (default 4) worker processes.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp/results"
GATES = RES / "v6_gates"
N_WORKERS = 4
DEV = range(900, 1000)

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from dsswm.stats.v7_analysis import A_METHODS, B_METHODS  # noqa: E402

NO_PARAM = {"FDC-BF", "FDC-MR[front3]", "FDC", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "B1", "B4", "B3-rect",
            "Peace-rect", "Molitor-WoR", "QFC-pool"}
TUNED = {"CR9": ("B4-bal", "Hait-SW", "B2-rect", "HC-WoR"), "LR9": ("B4-bal", "Hait-SW", "B2-rect", "HC-WoR")}
LR9_EPS_GRID = (0.002, 0.003, 0.004)
X12_RANK = 12                       # 12 of 15 problems (the N80 stop of lock v5)

CODE_FILES = ["run_r5s_v7.py",
              "dsswm/baselines/fdc_bet.py", "dsswm/baselines/fdc_mr.py",
              "dsswm/envs/lenta_v7.py", "dsswm/stats/prereg_v7.py", "dsswm/stats/v7_analysis.py",
              "dsswm/stats/v7_replica.py", "dsswm/stats/v7_seal.py",
              # v6 modules used unchanged (also bound by the v6 addendum, re-checked by the v7 gate)
              "dsswm/baselines/rect_v6.py", "dsswm/baselines/wor_betting_v6.py",
              "dsswm/streams/frontier_runner_v6.py", "dsswm/envs/lenta_v6.py", "dsswm/envs/data_v6.py",
              "dsswm/stats/prereg_v6.py", "dsswm/stats/v6_analysis.py", "dsswm/stats/v6_replica.py",
              "dsswm/stats/v6_seal.py"]
LOCK_ONLY_FILES = ["build_v7_addendum_draft.py", "dsswm/tests/test_fdc_bet.py", "dsswm/tests/test_v7_addendum.py"]


def _rng(a, b):
    return list(range(a, b + 1))


A_REST = [m for m in A_METHODS if m != "Peace-rect"]
B_REST = [m for m in B_METHODS if m != "Peace-rect"]
TASKS = {
    # dev runner checks (dev seeds 950-999 only; these seeds were used to SELECT FDC-BF)
    "v7a_pilot": dict(block="A", layer="CR9", half="dev", seeds=_rng(950, 999), eps=(0.001,), stop_k=15,
                      methods=list(A_METHODS)),
    "v7b_pilot": dict(block="B", layer="LR9", half="dev", seeds=_rng(950, 999), eps=LR9_EPS_GRID, stop_k=None,
                      methods=list(B_METHODS)),
    # eval tasks (v7 addendum lock required)
    "v7a_full_a": dict(block="A", layer="CR9", half="eval", seeds=_rng(33000, 33199), eps=(0.001,), stop_k=15,
                       methods=["Peace-rect"]),
    "v7a_full_b": dict(block="A", layer="CR9", half="eval", seeds=_rng(33000, 33199), eps=(0.001,), stop_k=15,
                       methods=A_REST),
    "v7b_full_a": dict(block="B", layer="LR9", half="eval", seeds=_rng(34000, 34199), eps=LR9_EPS_GRID, stop_k=None,
                       methods=["Peace-rect"]),
    "v7b_full_b": dict(block="B", layer="LR9", half="eval", seeds=_rng(34000, 34199), eps=LR9_EPS_GRID, stop_k=None,
                       methods=B_REST),
}
BLOCK_TASKS = {"A": ("v7a_full_a", "v7a_full_b"), "B": ("v7b_full_a", "v7b_full_b")}
DEV_BLOCK_TASKS = {"A": ("v7a_pilot",), "B": ("v7b_pilot",)}
SEALS = RES / "full" / "v7_seals"
_ENV: dict = {}


# =============================================================================================== helpers
def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hashes():
    per = {p: sha_file(CODE / p) for p in CODE_FILES if (CODE / p).exists()}
    return per, hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()


def progress(task, done, total, metric=None):
    (RES / f"{task}_PROGRESS.json").write_text(json.dumps({
        "task_id": task, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(task, status, txt):
    pf = RES / f"{task}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES / f"{task}_DONE").write_text(json.dumps({"task_id": task, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def load_configs(layer, lock=None):
    """Frozen rival configs (never a default fallback): CR9 = v5 frozen configs + v6 tuned HC-WoR; LR9 = v6 LR9 gate
    file.  With a locked addendum they must equal lock['frozen_configs'][layer]."""
    if layer == "CR9":
        cfg = {k: dict(v) for k, v in json.loads((WS / "plan/prereg_lock.json").read_text())["rival_configs"]
               ["frozen_configs"].items()}
        cfg["HC-WoR"] = json.loads((GATES / "hc_config_cr9.json").read_text())["selected"]
    else:
        cfg = json.loads((GATES / "rival_configs_lr9.json").read_text())["selected"]
    for m in TUNED[layer]:
        if not cfg.get(m):
            raise RuntimeError(f"no frozen config for tuned rival {m} on {layer}")
    if lock is not None:
        locked = (lock.get("frozen_configs") or {}).get(layer)
        if locked is None or {m: locked.get(m) for m in TUNED[layer]} != {m: cfg[m] for m in TUNED[layer]}:
            raise RuntimeError(f"frozen configs on disk differ from the locked addendum ({layer})")
    return cfg


def make(name, kw):
    if name == "FDC-BF":
        from dsswm.baselines.fdc_bet import make_variant
        return make_variant("FDC-BF")
    if name == "FDC-MR[front3]":
        from dsswm.baselines.fdc_mr import FDCMR
        return FDCMR(name="FDC-MR[front3]", rho="front3")
    if name == "RECT-ck-HG":
        from dsswm.baselines.rect_v6 import RectCkHG
        return RectCkHG()
    if name == "RECT-ck-HG-live":
        from dsswm.baselines.rect_v6 import RectCkHGLive
        return RectCkHGLive()
    if name == "RECT-ck-Bern":
        from dsswm.baselines.rect_v6 import RectCkBern
        return RectCkBern()
    if name == "HC-WoR":
        from dsswm.baselines.wor_betting_v6 import HC_DEFAULT, HCWoRRect
        p = dict(HC_DEFAULT)
        p.update(kw or {})
        return HCWoRRect(p["schedule"], p["c"], p["target_frac"])
    return reg.make_method(name, **(kw or {}))


def build_env(layer, half, eval_task_id=None):
    if layer == "CR9":
        from dsswm.envs.pool_replay import PoolReplayEnv
        return PoolReplayEnv("CR9", half), fr.cr_problems("visit"), "visit"
    from dsswm.envs.lenta_v6 import lr9_problems
    from dsswm.envs.lenta_v7 import LentaLayerEnvV7
    return LentaLayerEnvV7(half, eval_task_id=eval_task_id), lr9_problems(), "response_att"


def init_env(layer, half, eps_list, eval_task_id=None):
    env, probs, outcome = build_env(layer, half, eval_task_id)
    ctx0 = build_ctx(env, probs, eps_list[0])
    J = true_policy_values(ctx0.pols, env.w, env.true_mu(outcome))
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    ctxs = {e: build_ctx(env, probs, e) for e in eps_list}
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, ctxs=ctxs)


def x_rank(U_rows, ck, eps, rank):
    """Interpolated arrival count at which the rank-th problem's sticky U crosses eps (log-linear in the checkpoint
    count, as run_fdc_bet.x12).  None if fewer than ``rank`` problems cross within the run."""
    if not U_rows:
        return None
    U = np.minimum.accumulate(np.array(U_rows, dtype=float), 0)
    lc = np.log(np.asarray(ck[:len(U_rows)], dtype=float))
    cr = []
    for q in range(U.shape[1]):
        u = U[:, q]
        idx = np.flatnonzero(u <= eps)
        if len(idx) == 0:
            cr.append(np.inf)
            continue
        k = idx[0]
        if k == 0 or not np.isfinite(u[k - 1]) or u[k] <= 0 or eps <= 0:
            cr.append(lc[k])
            continue
        a, b = math.log(u[k - 1]), math.log(max(u[k], 1e-300))
        t = (a - math.log(eps)) / (a - b) if a > b else 1.0
        cr.append(lc[k - 1] + min(max(t, 0.0), 1.0) * (lc[k] - lc[k - 1]))
    v = np.sort(cr)[rank - 1]
    return float(math.exp(v)) if np.isfinite(v) else None


def job(a):
    name, kw, seed, eps, stop_k, code_sha, add_sha, data_sha = a
    ctx = _ENV["ctxs"][eps]
    if stop_k is not None:
        ctx = ctx_with_stop(ctx, stop_k)
    try:
        m = make(name, kw)
        t0 = time.perf_counter()
        s, rows = run_stream_v6(_ENV["env"], m, seed, ctx.problems, ctx.eps, outcome=_ENV["outcome"], ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        curve_full = curve + [curve[-1] if curve else 0] * (len(ck) - len(curve))

        def first_at(target):
            for k, c in enumerate(curve):
                if c >= target:
                    return k
            return None

        k12 = first_at(12)
        n80_pen = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        k15 = first_at(15)
        xx = x_rank([r["U"] for r in rows], ck, ctx.eps, X12_RANK)
        out = {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "stop_k": int(ctx.stop_k)}
        out.update({f"rs_{k}": v for k, v in s.items()})
        out.update({"N80_pen": n80_pen, "N80_raw": int(ck[k12]) if k12 is not None else tau,
                    "k80": k12, "completed80": k12 is not None,
                    "n_false_at_k80": int(nfalse[k12]) if k12 is not None else int(s["n_false"]),
                    "x12": float(xx) if xx is not None else float(tau),
                    "N100_pen": (int(ck[k15]) if (k15 is not None and nfalse[k15] == 0) else tau)
                    if stop_k == 15 else None,
                    "k100": k15 if stop_k == 15 else None,
                    "fwer_event": bool(s["fwer_event"]), "n_false": int(s["n_false"]), "cert_k": s["cert_k"],
                    "decided_pi": s["decided_pi"], "billing_ok": bool(s["billing_ok"]),
                    "schedule_digest": s["schedule_digest"], "n_cert_curve": curve_full, "tau_R": tau,
                    "sec": round(time.perf_counter() - t0, 3), "code_sha256": code_sha,
                    "addendum_sha256": add_sha, "data_sha256": data_sha, "error": None})
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def run_jobs(task, out, jobs_all, add_sha=None, data_sha=None):
    rfile = out / "results.jsonl"
    _, code_sha = code_hashes()
    done = set()
    if rfile.exists():
        keep = []
        for l in rfile.read_text().splitlines():
            try:
                x = json.loads(l)
            except ValueError:
                continue
            key = (x["method"], json.dumps(x["params"], sort_keys=True), x["seed"], x["eps"])
            if (x.get("error") is None and x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
                    and x.get("data_sha256") == data_sha and key not in done):
                done.add(key)
                keep.append(l)
        rfile.write_text("".join(k + "\n" for k in keep))
    jobs = [j for j in jobs_all if (j[0], json.dumps(j[1], sort_keys=True), j[2], float(j[3])) not in done]
    total = len(jobs_all)
    n_done = total - len(jobs)
    print(f"[{task}] {total} jobs, {len(jobs)} to run", flush=True)
    errs = []
    if jobs:
        jobs = [j[:5] + (code_sha, add_sha, data_sha) for j in jobs]
        rp = out / "replica_report.json"
        if rp.exists():
            rp.unlink()
        with get_context("fork").Pool(N_WORKERS) as pool, open(rfile, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                if r.get("error"):
                    errs.append(r)
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if n_done % 10 == 0 or n_done == total:
                    progress(task, n_done, total, {"runs_done": n_done, "errors": len(errs)})
    if errs:
        (out / "errors.log").write_text("\n\n".join(f"{e['method']} {e['seed']}\n{e['error']}" for e in errs))
        raise RuntimeError(f"{len(errs)} jobs failed (see {out / 'errors.log'})")
    rows = [x for x in (json.loads(l) for l in rfile.read_text().splitlines())
            if x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
            and x.get("data_sha256") == data_sha]
    return rows, code_sha


def make_jobs(T, seeds, cfg):
    jobs = []
    for m in T["methods"]:
        if m in NO_PARAM:
            kw = {}
        else:
            if not cfg.get(m):
                raise RuntimeError(f"no frozen config for {m}")
            kw = dict(cfg[m])
        for e in T["eps"]:
            for s in seeds:
                jobs.append((m, kw, s, e, T["stop_k"]))
    heavy = ("Peace-rect", "HC-WoR", "FDC-MR[front3]")
    jobs.sort(key=lambda j: (j[0] not in heavy,))
    return jobs


def read_rows(path, add_sha=None):
    p = Path(path)
    if not p.exists():
        return []
    return [x for x in (json.loads(l) for l in p.read_text().splitlines())
            if x.get("error") is None and (add_sha is None or x.get("addendum_sha256") == add_sha)]


def current_rows(path, add_sha, data_sha):
    _, code_sha = code_hashes()
    return [r for r in read_rows(path, add_sha) if r.get("code_sha256") == code_sha
            and r.get("addendum_sha256") == add_sha and r.get("data_sha256") == data_sha]


def data_sha_for(layer, lock):
    from dsswm.envs.data_v6 import layer_data_sha
    return layer_data_sha(layer, frozen=None if lock is None else lock["data_sha256"])


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v7 import addendum_gate
    ok, info = addendum_gate(task)
    if not ok:
        (out / "summary.json").write_text(json.dumps({"task_id": task, "status": "skipped_by_lock", "reason": info,
                                                      "written_at": t_start.isoformat(), "eval_touched": False},
                                                     indent=1))
        mark_done(task, "skipped_by_lock", info)
        print(f"[{task}] skipped_by_lock: {info}")
        return None
    return info


def replica(task, T, lock, add_sha):
    from dsswm.stats import v7_replica as R
    is_eval = T["half"] == "eval"
    base = RES / ("full" if is_eval else "pilots") / task
    data_sha = data_sha_for(T["layer"], lock)
    main_rows = current_rows(base / "results.jsonl", add_sha, data_sha)
    from dsswm.stats.v6_replica import check_matrix
    m = check_matrix(main_rows, T["methods"], T["eps"], T["seeds"])
    if not m["complete"]:
        raise SystemExit(f"[{task}] replica refused: planned matrix incomplete {m}")
    if is_eval:
        from dsswm.stats.v7_seal import verify_seal
        verify_seal(SEALS, task, main_rows, T["methods"], T["eps"], T["seeds"], code_hashes()[0], data_sha, add_sha)
    rout = base / "replica"
    rout.mkdir(parents=True, exist_ok=True)
    seeds = T["seeds"][:R.R1_N_SEEDS]
    cfg = load_configs(T["layer"], lock)
    init_env(T["layer"], T["half"], tuple(T["eps"]), eval_task_id=task if is_eval else None)
    run_jobs(task + "_replica", rout, make_jobs(T, seeds, cfg), add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    rep_rows = current_rows(rout / "results.jsonl", add_sha, data_sha)
    rep = R.build_report(task, is_eval, main_rows, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha, per_sha,
                         data_sha)
    rep["written_at"] = datetime.now().isoformat()
    (base / "replica_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"task": task, "replica": rep["status"]}))


def analyse(block, dev):
    from dsswm.stats import v7_analysis as VA
    from dsswm.stats import v7_replica as R
    if dev:
        tasks, add_sha, lock, base = DEV_BLOCK_TASKS[block], None, None, RES / "pilots"
        T0 = TASKS[tasks[0]]
        spec = dict(VA.BLOCK_SPEC[block])
        spec.update(seeds=tuple(T0["seeds"]))
    else:
        from dsswm.stats.prereg_v7 import load_locked_addendum
        lock = load_locked_addendum(None)
        add_sha = lock["sha256"]
        tasks, spec, base = BLOCK_TASKS[block], None, RES / "full"
    per_sha, _ = code_hashes()
    rows, reps, sts = [], [], []
    for t in tasks:
        T = TASKS[t]
        data_sha = data_sha_for(T["layer"], lock)
        cur = current_rows(base / t / "results.jsonl", add_sha, data_sha)
        rf = base / t / "replica_report.json"
        rep = json.loads(rf.read_text()) if rf.exists() else None
        rep_rows = current_rows(base / t / "replica" / "results.jsonl", add_sha, data_sha)
        sealed = None
        if not dev:
            from dsswm.stats.v7_seal import verify_seal
            sealed = verify_seal(SEALS, t, cur, T["methods"], T["eps"], T["seeds"], per_sha, data_sha,
                                 add_sha)["seal"]["results_content_sha256"]
        sts.append(R.validate_report(rep, t, not dev, cur, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha,
                                     per_sha, data_sha, sealed_content_sha256=sealed))
        rows += cur
        reps.append(rep)
    replica_status = {"status": "pass" if all(x == "pass" for x in sts) else "fail"}
    res = VA.analyse_block(block, rows, replica_status, spec=spec)
    res.update({"mode": "dev_runner_check" if dev else "eval", "addendum_sha256": add_sha, "tasks": list(tasks),
                "replica_statuses": dict(zip(tasks, sts)),
                "replica_content_sha256": [r["results_content_sha256"] for r in reps],
                "code_sha256": per_sha, "written_at": datetime.now().isoformat()})
    out = base / f"v7_analysis_{block}{'_dev' if dev else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"block": block, "decision": res["decision"]}, default=str))


def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--replica", action="store_true")
    ap.add_argument("--analyse", choices=("A", "B"))
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 per process; host rule)")
    args = ap.parse_args()
    N_WORKERS = max(1, min(4, args.workers))
    if args.analyse:
        analyse(args.analyse, args.dev)
        return
    task = args.task
    T = TASKS[task]
    is_eval = T["half"] == "eval"
    out = RES / ("full" if is_eval else "pilots") / task
    out.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    lock, add_sha = None, None
    if is_eval:
        lock = gate_or_skip(task, out, t_start)
        if lock is None:
            return
        add_sha = lock["sha256"]
    else:
        assert all(s in DEV for s in T["seeds"]), "dev tasks may only use seeds 900-999"
    if args.replica:
        replica(task, T, lock, add_sha)
        return
    if is_eval:
        from dsswm.stats.v7_seal import seal_exists
        if seal_exists(SEALS, task):
            raise SystemExit(f"[{task}] already sealed: a sealed eval task is never re-run or resumed")
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    eps_list = tuple(T["eps"])
    cfg = load_configs(T["layer"], lock)
    data_sha = data_sha_for(T["layer"], lock)
    init_env(T["layer"], T["half"], eps_list, eval_task_id=task if is_eval else None)
    env = _ENV["env"]
    jobs = make_jobs(T, T["seeds"], cfg)
    rows, code_sha = run_jobs(task, out, jobs, add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    t_end = datetime.now()
    from dsswm.stats.v6_replica import check_r2
    from dsswm.stats.v7_replica import COMPARISON_GROUPS
    summary = {"task_id": task, "status": "complete", "block": T["block"], "layer": T["layer"], "half": T["half"],
               "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]), "eps": list(eps_list),
               "stop_k": T["stop_k"], "tau_R": int(env.tau_R), "n_rows": len(rows), "expected_rows": len(jobs),
               "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
               "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
               "cpu_sec_sum": round(float(sum(r["sec"] for r in rows)), 1),
               "timing_note": f"{N_WORKERS} worker processes, BLAS=1; concurrent with other v7 tasks (biased up)",
               "code_sha256": per_sha, "code_sha256_combined": code_sha, "addendum_sha256": add_sha,
               "data_sha256": data_sha, "eval_touched": bool(is_eval),
               "configs": {m: (None if m in NO_PARAM else cfg.get(m)) for m in T["methods"]},
               "R2_design_identity": check_r2(rows, COMPARISON_GROUPS.get(task, ()), T["seeds"], eps_list)}
    if is_eval:
        from dsswm.stats.v7_seal import make_seal, write_and_commit_seal
        seal = make_seal(task, rows, T["methods"], eps_list, T["seeds"], per_sha, data_sha, add_sha)
        summary["seal"] = {"results_content_sha256": seal["results_content_sha256"],
                           "seal_commit": write_and_commit_seal(seal, SEALS),
                           "seal_path": str((SEALS / f"{task}.seal.json").relative_to(WS))}
    summary["per_method_cpu_sec"] = {m: round(float(sum(r["sec"] for r in rows if r["method"] == m)), 1)
                                     for m in sorted({r["method"] for r in rows})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pid = RES / f"{task}.pid"
    if pid.exists():
        pid.unlink()
    mark_done(task, "success", f"{len(rows)} rows, wall {summary['wall_min']} min")
    print(json.dumps({k: summary[k] for k in ("task_id", "n_rows", "wall_min", "cpu_sec_sum")}), flush=True)


if __name__ == "__main__":
    main()
