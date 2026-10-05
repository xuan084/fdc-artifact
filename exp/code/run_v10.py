"""Lock-v10 ADDENDUM runner (new file; v5-v9 files untouched and imported unchanged).

FDC-DP over exponentially large segment-policy classes on frozen uplift-score segmentations (seg_v10_eval), K = 20
block grid, frozen 50/50 design, delta 0.05, streams run to 15/15 or tau_R, endpoint N80_pen at 12/15.
Methods: TU-FDC-DP(b) (primary), RECT-BF-DP-TU (primary rival), HC-WoR-DP (untuned transfer), FDC-DP(a)
(descriptive).

  A  CONFIRMATORY  X5 RetailHero eval half (3rd use; fresh seeds 38000-38199): cells v10a_full_s16 / _s32
  B  CONFIRMATORY  Lenta LR9 eval half (fresh seeds 38200-38399): cells v10b_full_s16 / _s32 / _s64
  D  DESCRIPTIVE   X5 S = 64 (rivals censored on dev): v10d_full_x5s64 (seeds 38000-38199)
  The cells / eps are those of dsswm.stats.v10_analysis.CELLS (selected on dev by the pre-stated rule).
  Pilot twins (dev halves, dev seeds 950-999): replace 'full' by 'pilot' in the task id.

Usage (cwd = exp/code)
  run_v10.py --freeze-seg                         # (once, dev only) write exp/results/v10_gates/seg_v10_frozen.json
  run_v10.py --task v10a_pilot_s16 [--workers 4]  # dev runner check
  run_v10.py --task v10a_full_s16                 # EVAL (v10 lock required)
  run_v10.py --task <task> --replica
  run_v10.py --analyse A|B|D [--dev]
Rows are written ahead to results.jsonl (fsync), resumable (except sealed eval tasks).  CPU only, BLAS threads 1,
<= 4 worker processes per runner process.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
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
GATES8 = RES / "v8_gates"
GATES10 = RES / "v10_gates"
SEALS = RES / "full" / "v10_seals"
N_WORKERS = 4
DEV = range(950, 1000)
K_GRID = 20
STOP_FRAC = 1.0                          # run to 15/15 (or tau_R)
N80_K = 12                               # endpoint: 12/15 certified

from dsswm.stats import v10_analysis as VA  # noqa: E402

CODE_FILES = ["run_v10.py", "dsswm/baselines/fdc_dp.py", "dsswm/baselines/rect_dp.py", "dsswm/envs/seg_v10.py",
              "dsswm/envs/seg_v10_eval.py", "dsswm/stats/prereg_v10.py", "dsswm/stats/v10_analysis.py",
              "dsswm/stats/v10_replica.py", "dsswm/stats/v10_seal.py",
              # inherited dependencies (imported, unchanged)
              "dsswm/baselines/rect_tu_v9.py", "dsswm/baselines/pjc_bf.py", "dsswm/baselines/wor_betting_v6.py",
              "dsswm/baselines/fdc_bet.py", "dsswm/baselines/b4_bal.py", "dsswm/baselines/rect_v6.py",
              "dsswm/baselines/frontier_common.py", "dsswm/streams/frontier_runner.py",
              "dsswm/streams/frontier_runner_v6.py", "dsswm/streams/frontier.py", "dsswm/envs/pool_replay.py",
              "dsswm/envs/lenta_v6.py", "dsswm/envs/data_v6.py", "dsswm/envs/data_v8.py", "dsswm/envs/base.py",
              "dsswm/stats/prereg.py", "dsswm/stats/prereg_v6.py", "dsswm/stats/prereg_v7.py",
              "dsswm/stats/prereg_v8.py", "dsswm/stats/prereg_v9.py", "dsswm/stats/v6_analysis.py",
              "dsswm/stats/v6_replica.py", "dsswm/stats/v8_replica.py", "dsswm/stats/v6_seal.py",
              "dsswm/stats/v8_seal.py", "dsswm/stats/v9_seal.py"]
LOCK_ONLY_FILES = ["build_v10_addendum_draft.py", "dsswm/tests/test_v10_addendum.py", "dsswm/tests/test_fdc_dp.py",
                   "run_fdc_dp_dev.py"]


def _rng(a, b):
    return list(range(a, b + 1))


TASKS = {}
for _b, _cells in VA.CELLS.items():
    for _t, _c in _cells.items():
        TASKS[_t] = dict(_c, block=_b, role="confirmatory", half="eval", seeds=list(VA.BLOCK_SEEDS[_b]))
        TASKS[_t.replace("_full_", "_pilot_")] = dict(_c, block=_b, role="confirmatory", half="dev",
                                                      seeds=_rng(950, 999))
for _t, _c in VA.DESC_CELLS.items():
    TASKS[_t] = dict(_c, role="descriptive", half="eval", seeds=list(VA.BLOCK_SEEDS[_c["block"]]))
    TASKS[_t.replace("_full_", "_pilot_")] = dict(_c, role="descriptive", half="dev", seeds=_rng(950, 999))
for _t, _T in TASKS.items():
    _T["layer"] = f"{_T['data'].upper()}-SR{_T['S']}"
    _T["methods"] = list(VA.METHODS)
BLOCK_TASKS = {"A": tuple(VA.CELLS["A"]), "B": tuple(VA.CELLS["B"]), "D": tuple(VA.DESC_CELLS)}
DEV_BLOCK_TASKS = {k: tuple(t.replace("_full_", "_pilot_") for t in v) for k, v in BLOCK_TASKS.items()}
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


def hc_config(data):
    """Frozen v8 HC-WoR schedule parameters (dev-tuned on CR9 / X9 at S = 9), transferred UNTUNED, 50/50 plan."""
    key = "cr" if data == "lenta" else "x9"
    c = json.loads((GATES8 / f"{key}_configs.json").read_text())["selected"]["HC-WoR"]
    return {"schedule": c["schedule"], "c": float(c["c"]), "target_frac": c.get("target_frac"), "plan": "0.5",
            "source": f"exp/results/v8_gates/{key}_configs.json (S = 9 tuning; plan forced to 0.5; untuned transfer)"}


def frozen_configs():
    from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
    return {"HC-WoR-DP": {"x5": hc_config("x5"), "lenta": hc_config("lenta")},
            "TU-FDC-DP(b)": {"scheme": "b", "node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO,
                             "split": [0.045, 0.005], "block_points": "the K = 20 grid env.checkpoints(20)"},
            "FDC-DP(a)": {"scheme": "a", "grid_ratio": GRID_RATIO, "split": [0.045, 0.005]},
            "RECT-BF-DP-TU": {"box": True, "split": [0.045, 0.005], "block_points": "the K = 20 grid"},
            "design": "balanced_alloc(S, A, 0.5) for every method", "delta": 0.05, "K": K_GRID,
            "checkpoints": "fr.checkpoints(n_min, tau_R, 20): n_min 2000 (X5) / 5000 (Lenta), tau_R = replay N",
            "stop_k": "Q = 15 (stop_frac 1.0)", "n80_k": N80_K}


def frozen_segmentation_hashes():
    from dsswm.envs.seg_v10_eval import FROZEN_JSON
    fz = json.loads(FROZEN_JSON.read_text())
    out = {"seg_v10_frozen.json": sha_file(FROZEN_JSON)}
    for d, v in fz["data"].items():
        out[v["model_pickle"]] = v["model_pickle_sha256"]
        if sha_file(GATES10 / v["model_pickle"]) != v["model_pickle_sha256"]:
            raise RuntimeError(f"{v['model_pickle']} differs from the frozen json")
    return out


def load_frozen(lock=None):
    cfg, seg = frozen_configs(), frozen_segmentation_hashes()
    if lock is not None:
        if lock.get("frozen_configs") != json.loads(json.dumps(cfg)):
            raise RuntimeError("frozen configs differ from the locked v10 addendum")
        if lock.get("frozen_segmentation") != seg:
            raise RuntimeError("frozen segmentation / score models differ from the locked v10 addendum")
    return cfg, seg


def make(name, data, ck, cfg):
    from dsswm.baselines.fdc_dp import FDCDP, FDCDPTimeUniform
    from dsswm.baselines.rect_dp import HCWoRDP, RectBFDPTU
    if name == "TU-FDC-DP(b)":
        c = cfg["TU-FDC-DP(b)"]
        return FDCDPTimeUniform(ck, scheme="b", node_limit=c["node_limit"], grid_ratio=c["grid_ratio"])
    if name == "RECT-BF-DP-TU":
        return RectBFDPTU(ck, box=True)
    if name == "FDC-DP(a)":
        return FDCDP("a", grid_ratio=cfg["FDC-DP(a)"]["grid_ratio"])
    if name == "HC-WoR-DP":
        c = cfg["HC-WoR-DP"][data]
        return HCWoRDP(c["schedule"], c["c"], c["target_frac"])
    raise KeyError(name)


def schedule_arrival_digest(sch):
    """sha256 (16 hex) of the arrival segment sequence and every pool's record order (as run_r5s_v8)."""
    h = hashlib.sha256(np.ascontiguousarray(sch.seg_seq).tobytes())
    for p in sch.pool_perm:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()[:16]


class _Capture:
    """Records the schedule the stream ACTUALLY consumed (wraps seg_v10.make_schedule for one run)."""

    def __enter__(self):
        from dsswm.envs import seg_v10 as SV
        self._m, self._orig, self.seen = SV, SV.make_schedule, []

        def wrapped(*a, **k):
            sch = self._orig(*a, **k)
            self.seen.append(sch)
            return sch

        SV.make_schedule = wrapped
        return self

    def __exit__(self, *exc):
        self._m.make_schedule = self._orig
        return False

    def digest(self):
        if len(self.seen) != 1:
            raise RuntimeError(f"expected exactly one schedule per run, captured {len(self.seen)}")
        return schedule_arrival_digest(self.seen[0])


def init_env(T, lock=None, eval_task_id=None):
    from dsswm.baselines.fdc_dp import make_seg_ctx
    from dsswm.envs.seg_v10 import env_digest, true_opt
    from dsswm.envs.seg_v10_eval import SegEnvV10Frozen
    fsha = None if lock is None else lock["frozen_segmentation"]["seg_v10_frozen.json"]
    env = SegEnvV10Frozen(T["data"], T["S"], T["half"], eval_task_id=eval_task_id, frozen_sha256=fsha)
    assert env.half == T["half"]
    Js, mu = true_opt(env)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, T["eps"], 0.05, env.checkpoints(K_GRID), env.tau_R,
                       env.replan_interval, stop_frac=STOP_FRAC)
    _ENV.clear()
    _ENV.update(env=env, Js=Js, mu=mu, ctx=ctx, data=T["data"], digest=env_digest(env), cfg=load_frozen(lock)[0])


def job(a):
    name, seed, eps, code_sha, add_sha, data_sha = a
    from dsswm.envs.seg_v10 import run_stream_seg
    ctx = _ENV["ctx"]
    try:
        m = make(name, _ENV["data"], ctx.checkpoints.copy(), _ENV["cfg"])
        t0 = time.perf_counter()
        with _Capture() as cap:
            s, rows = run_stream_seg(_ENV["env"], m, seed, ctx, _ENV["Js"], _ENV["mu"], n80_k=N80_K)
        sec = round(time.perf_counter() - t0, 3)
        out = {"method": name, "seed": int(seed), "eps": float(eps)}
        out.update({k: s[k] for k in ("validity", "schedule_digest", "reached_stop", "stop_k", "n80_k", "k_stop",
                                      "k80", "N80_pen", "N80_raw", "n80_lt_tau", "false_by_k80",
                                      "exhaustion_at_k80", "N_stop_pen", "n_cert", "n_false", "fwer_event",
                                      "cert_k", "decided_pi", "billing_ok", "n_cert_curve", "tau_R")})
        out.update({"arrival_digest": cap.digest(), "env_digest": _ENV["digest"], "K_eval": int(len(ctx.checkpoints)),
                    "sec": sec, "rs_sec_cert": {"total": s["sec_cert"], "by_k": s["sec_cert_by_k"]},
                    "rs_sec_total": s["sec_total"], "code_sha256": code_sha, "addendum_sha256": add_sha,
                    "data_sha256": data_sha, "error": None})
        if hasattr(m, "n_stale_evals"):
            out["n_stale_evals"] = int(m.n_stale_evals)
        cs = getattr(m, "cert_stats", None)
        if cs:
            out.update({"bnb_nodes": int(sum(x["nodes"] for x in cs)), "bnb_calls": int(sum(x["bnb_calls"] for x in cs)),
                        "node_limit_hits": int(sum(x["node_limit_hits"] for x in cs)),
                        "b_only_certs": int(sum(x["b_certified"] for x in cs)),
                        "a_certs": int(sum(x["a_certified"] for x in cs)),
                        "beta": float(m.ledger["beta"]), "ledger_K": int(m.ledger["K"])})
        led = getattr(m, "ledger", None)
        if isinstance(led, dict) and "beta_c" in led:
            out.update({"beta_c": float(led["beta_c"]), "ledger_K": int(led["K"])})
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def run_jobs(task, out, jobs_all, add_sha=None, data_sha=None):
    rfile = out / "results.jsonl"
    _, code_sha = code_hashes()
    done = set()
    if rfile.exists():
        keep = []
        for line in rfile.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            key = (x["method"], x["seed"], x["eps"])
            if (x.get("error") is None and x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
                    and x.get("data_sha256") == data_sha and key not in done):
                done.add(key)
                keep.append(line)
        rfile.write_text("".join(k + "\n" for k in keep))
    jobs = [j for j in jobs_all if (j[0], j[1], float(j[2])) not in done]
    total = len(jobs_all)
    n_done = total - len(jobs)
    print(f"[{task}] {total} jobs, {len(jobs)} to run", flush=True)
    errs = []
    if jobs:
        jobs = [j[:3] + (code_sha, add_sha, data_sha) for j in jobs]
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
    rows = [x for x in (json.loads(line) for line in rfile.read_text().splitlines())
            if x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
            and x.get("data_sha256") == data_sha]
    return rows, code_sha


def make_jobs(T, seeds):
    jobs = [(m, s, T["eps"]) for m in T["methods"] for s in seeds]
    jobs.sort(key=lambda j: j[0] != "HC-WoR-DP")
    return jobs


def current_rows(path, add_sha, data_sha):
    _, code_sha = code_hashes()
    p = Path(path)
    if not p.exists():
        return []
    return [x for x in (json.loads(line) for line in p.read_text().splitlines())
            if x.get("error") is None and x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
            and x.get("data_sha256") == data_sha]


def data_sha_for(data, lock):
    from dsswm.envs.seg_v10_eval import layer_data_sha_v10
    return layer_data_sha_v10(data, frozen=None if lock is None else lock["data_sha256"])


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v10 import addendum_gate
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
    from dsswm.stats import v10_replica as R
    from dsswm.stats.v6_replica import check_matrix
    is_eval = T["half"] == "eval"
    base = RES / ("full" if is_eval else "pilots") / task
    data_sha = data_sha_for(T["data"], lock)
    eps = (T["eps"],)
    main_rows = current_rows(base / "results.jsonl", add_sha, data_sha)
    m = check_matrix(main_rows, T["methods"], eps, T["seeds"])
    if not m["complete"]:
        raise SystemExit(f"[{task}] replica refused: planned matrix incomplete {m}")
    if is_eval:
        from dsswm.stats.v10_seal import verify_seal
        verify_seal(SEALS, task, main_rows, T["methods"], eps, T["seeds"], code_hashes()[0], data_sha, add_sha)
    rout = base / "replica"
    rout.mkdir(parents=True, exist_ok=True)
    seeds = T["seeds"][:R.R1_N_SEEDS]
    init_env(T, lock, eval_task_id=task if is_eval else None)
    run_jobs(task + "_replica", rout, make_jobs(T, seeds), add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    rep_rows = current_rows(rout / "results.jsonl", add_sha, data_sha)
    rep = R.build_report(task, is_eval, main_rows, rep_rows, T["methods"], eps, T["seeds"], add_sha, per_sha,
                         data_sha)
    rep["written_at"] = datetime.now().isoformat()
    (base / "replica_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"task": task, "replica": rep["status"]}))


def _task_rows_and_status(t, dev, add_sha, lock, per_sha):
    from dsswm.stats import v10_replica as R
    T = TASKS[t]
    base = RES / ("pilots" if dev else "full")
    data_sha = data_sha_for(T["data"], lock)
    eps = (T["eps"],)
    cur = current_rows(base / t / "results.jsonl", add_sha, data_sha)
    rf = base / t / "replica_report.json"
    rep = json.loads(rf.read_text()) if rf.exists() else None
    rep_rows = current_rows(base / t / "replica" / "results.jsonl", add_sha, data_sha)
    sealed = None
    if not dev:
        from dsswm.stats.v10_seal import verify_seal
        sealed = verify_seal(SEALS, t, cur, T["methods"], eps, T["seeds"], per_sha, data_sha,
                             add_sha)["seal"]["results_content_sha256"]
    st = R.validate_report(rep, t, not dev, cur, rep_rows, T["methods"], eps, T["seeds"], add_sha, per_sha,
                           data_sha, sealed_content_sha256=sealed)
    return cur, st, rep


def analyse(block, dev):
    from dsswm.stats.v6_analysis import boot_idx
    if dev:
        tasks, add_sha, lock = DEV_BLOCK_TASKS[block], None, None
        base = RES / "pilots"
    else:
        from dsswm.stats.prereg_v10 import load_locked_addendum
        lock = load_locked_addendum(None)
        add_sha = lock["sha256"]
        tasks, base = BLOCK_TASKS[block], RES / "full"
    per_sha, _ = code_hashes()
    cfg, seg = load_frozen(lock)
    out = {"block": block, "mode": "dev_runner_check" if dev else "eval", "addendum_sha256": add_sha,
           "tasks": list(tasks), "code_sha256": per_sha, "frozen_segmentation": seg}
    rows_by, sts, reps = {}, {}, []
    for t in tasks:
        cur, st, rep = _task_rows_and_status(t, dev, add_sha, lock, per_sha)
        rows_by[t] = cur
        sts[t] = st
        reps.append(rep["results_content_sha256"])
    replica_status = {"status": "pass" if all(s == "pass" for s in sts.values()) else "fail", "task_statuses": sts}
    seeds = TASKS[tasks[0]]["seeds"]
    if block in ("A", "B"):
        cells = {t: {k: TASKS[t][k] for k in ("data", "S", "eps")} for t in tasks}
        res = VA.analyse_block(block, rows_by, replica_status, seeds=seeds, cells=cells)
    else:
        idx = boot_idx(len(seeds))
        res = {"block": "D", "role": "descriptive (rivals censored on dev); no verdict",
               "cells": {t: VA.analyse_cell(t, rows_by[t], seeds, TASKS[t]["eps"], idx=idx) for t in tasks}}
    res.update(out)
    res.update({"replica_status_combined": replica_status, "replica_content_sha256": reps,
                "frozen_configs_used": cfg, "written_at": datetime.now().isoformat()})
    fname = f"v10_analysis_{block}{'_dev' if dev else ''}.json"
    (base / fname).write_text(json.dumps(res, indent=1, default=str))
    dec = res.get("decision", {}).get("verdict") if "decision" in res else "descriptive"
    short = {t: (round(c["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["geomean_ratio"], 3),
                 round(c["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["ub95_one_sided"], 3),
                 c["primary_false_streams"]) for t, c in res["cells"].items()}
    print(json.dumps({"block": block, "decision": dec, "cells": short, "replica": replica_status["status"]}))
    return res


# =============================================================================================== main
def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--replica", action="store_true")
    ap.add_argument("--analyse", choices=("A", "B", "D"))
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze-seg", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 per process; host rule)")
    args = ap.parse_args()
    N_WORKERS = max(1, min(4, args.workers))
    if args.freeze_seg:
        from dsswm.envs.seg_v10_eval import freeze_segmentations
        r = freeze_segmentations()
        print(json.dumps({d: v["model_pickle_sha256"][:16] for d, v in r["data"].items()}))
        return
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
        assert all(s in DEV for s in T["seeds"]), "dev tasks may only use seeds 950-999"
    if args.replica:
        replica(task, T, lock, add_sha)
        return
    if is_eval:
        from dsswm.stats.v10_seal import seal_exists
        if seal_exists(SEALS, task):
            raise SystemExit(f"[{task}] already sealed: a sealed eval task is never re-run or resumed")
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    data_sha = data_sha_for(T["data"], lock)
    init_env(T, lock, eval_task_id=task if is_eval else None)
    env = _ENV["env"]
    jobs = make_jobs(T, T["seeds"])
    rows, code_sha = run_jobs(task, out, jobs, add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    t_end = datetime.now()
    from dsswm.stats.v6_replica import check_r2
    from dsswm.stats.v10_replica import check_r2b, groups_for
    eps = (T["eps"],)
    summary = {"task_id": task, "status": "complete", "block": T["block"], "role": T["role"], "layer": T["layer"],
               "half": T["half"], "S": T["S"], "eps": T["eps"], "K": K_GRID,
               "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]), "tau_R": int(env.tau_R),
               "pool_sizes_min": int(env.pool_sizes.min()), "n_rows": len(rows), "expected_rows": len(jobs),
               "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
               "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
               "cpu_sec_sum": round(float(sum(r["sec"] for r in rows)), 1),
               "timing_note": f"{N_WORKERS} worker processes, BLAS=1; concurrent with other tasks (biased up)",
               "code_sha256": per_sha, "code_sha256_combined": code_sha, "addendum_sha256": add_sha,
               "data_sha256": data_sha, "eval_touched": bool(is_eval),
               "R2_design_identity": [check_r2(rows, g, T["seeds"], eps) for g in groups_for(task)],
               "R2b_arrival_identity": check_r2b(rows, T["seeds"], eps, T["methods"])}
    if is_eval:
        from dsswm.stats.v10_seal import make_seal, write_and_commit_seal
        seal = make_seal(task, rows, T["methods"], eps, T["seeds"], per_sha, data_sha, add_sha)
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
