"""Lock-v6 ADDENDUM blocks (supplementary to lock v5; cannot change the v5 verdict).

  A  same-strength rectangular rival stress test (CR9 eval half, NEW seeds 31000-31199)
  B  Lenta LR9 replication (descriptive; LR9 eval half, NEW seeds 32000-32199, eps grid reported in full)
  C  N100 continuation of the v5 rigorous rivals R on the v5 eval streams 30000-30199 (stop_k = 15)

Usage
  run_r5s_v6.py --task v6a_tune        # dev: HC-WoR lambda grid on CR9 dev seeds 900-949 -> exp/results/v6_gates
  run_r5s_v6.py --task v6a_pilot       # dev: all block-A methods on CR9 dev seeds 950-999 (estimates + runtime)
  run_r5s_v6.py --task v6b_tune        # dev: LR9 rival grids (B4-bal, Hait-SW, B2-rect, HC-WoR) at eps 0.003, 900-949
  run_r5s_v6.py --task v6b_pilot       # dev: LR9 eps grid x methods on seeds 950-999
  run_r5s_v6.py --task v6c_pilot       # dev: CR9 dev continuation timing on seeds 950-957
  run_r5s_v6.py --task v6a_full | v6b_full_a | v6b_full_b | v6c_full_a | v6c_full_b    # EVAL: refused unless the v6 addendum is
                                                                         # locked (dsswm.stats.prereg_v6.addendum_gate)
Dev tasks may only use seeds 900-999 (asserted).  Eval tasks write summary.status='skipped_by_lock' when refused.
Every (method, eps, seed) row is written ahead to results.jsonl (fsync), resumable; no analysis inside a block.
CPU only, 4 worker processes, BLAS threads 1.
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
GATES = RES / "v6_gates"
N_WORKERS = 4
DEV = range(900, 1000)

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402

V5_R = ["B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool"]
A_METHODS = ["FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR"]
B_METHODS = ["FDC", "B4-bal", "QFC-pool", "B2-rect", "Peace-rect", "Hait-SW", "RECT-ck-HG", "RECT-ck-HG-live",
             "RECT-ck-Bern", "HC-WoR"]
NO_PARAM = {"FDC", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "B1", "B4", "B3-rect", "Peace-rect",
            "Molitor-WoR", "QFC-pool"}
TUNED = {"CR9": ("B4-bal", "Hait-SW", "B2-rect", "HC-WoR"), "LR9": ("B4-bal", "Hait-SW", "B2-rect", "HC-WoR")}
LR9_EPS_GRID = (0.002, 0.003, 0.004)
LR9_TUNE_EPS = 0.003

CODE_FILES = ["run_r5s_v6.py", "dsswm/baselines/rect_v6.py", "dsswm/baselines/wor_betting_v6.py",
              "dsswm/streams/frontier_runner_v6.py", "dsswm/envs/lenta_v6.py", "dsswm/stats/prereg_v6.py",
              "dsswm/stats/v6_analysis.py", "dsswm/stats/v6_replica.py",
              "dsswm/envs/data_v6.py", "dsswm/stats/v6_seal.py"]


def _rng(a, b):
    return list(range(a, b + 1))


TASKS = {
    # ---------------------------------------------------------------- dev (pilot / tuning) tasks
    "v6a_tune": dict(layer="CR9", half="dev", seeds=_rng(900, 949), eps=(0.001,), stop_k=None, kind="tune_hc",
                     methods=["HC-WoR"]),
    "v6a_pilot": dict(layer="CR9", half="dev", seeds=_rng(950, 999), eps=(0.001,), stop_k=None, methods=A_METHODS),
    "v6b_tune": dict(layer="LR9", half="dev", seeds=_rng(900, 949), eps=(LR9_TUNE_EPS,), stop_k=None, kind="tune_lr9",
                     methods=["B4-bal", "Hait-SW", "B2-rect", "HC-WoR"]),
    "v6b_pilot": dict(layer="LR9", half="dev", seeds=_rng(950, 999), eps=LR9_EPS_GRID, stop_k=None,
                      methods=B_METHODS),
    "v6b_hc_sens": dict(layer="LR9", half="dev", seeds=_rng(900, 949), eps=(LR9_TUNE_EPS,), stop_k=None,
                        kind="sens_hc", methods=["HC-WoR"]),
    "v6c_pilot": dict(layer="CR9", half="dev", seeds=_rng(950, 957), eps=(0.001,), stop_k=15, methods=["FDC"] + V5_R),
    # ---------------------------------------------------------------- eval tasks (addendum lock required)
    "v6a_full": dict(layer="CR9", half="eval", seeds=_rng(31000, 31199), eps=(0.001,), stop_k=None,
                     methods=A_METHODS),
    "v6b_full_a": dict(layer="LR9", half="eval", seeds=_rng(32000, 32199), eps=LR9_EPS_GRID, stop_k=None,
                       methods=["Peace-rect"]),
    "v6b_full_b": dict(layer="LR9", half="eval", seeds=_rng(32000, 32199), eps=LR9_EPS_GRID, stop_k=None,
                       methods=[m for m in B_METHODS if m != "Peace-rect"]),
    "v6c_full_a": dict(layer="CR9", half="eval", seeds=_rng(30000, 30199), eps=(0.001,), stop_k=15,
                       methods=["Peace-rect"]),
    "v6c_full_b": dict(layer="CR9", half="eval", seeds=_rng(30000, 30199), eps=(0.001,), stop_k=15,
                       methods=["FDC"] + [m for m in V5_R if m != "Peace-rect"]),
}

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


def v5_frozen_configs():
    lock = json.loads((WS / "plan/prereg_lock.json").read_text())
    return {k: dict(v) for k, v in lock["rival_configs"]["frozen_configs"].items()}


def load_configs(layer, lock=None):
    """Frozen rival configs, never a default fallback.  CR9: v5 frozen configs + v6 tuned HC-WoR
    (v6_gates/hc_config_cr9.json); LR9: v6_gates/rival_configs_lr9.json.  With a locked addendum the configs are
    taken from lock['frozen_configs'][layer] and must equal the gate files on disk (whose hashes the gate checked)."""
    if layer == "CR9":
        f = GATES / "hc_config_cr9.json"
        if not f.exists():
            raise RuntimeError(f"missing frozen config {f} (run v6a_tune first)")
        cfg = v5_frozen_configs()
        cfg["HC-WoR"] = json.loads(f.read_text())["selected"]
    else:
        f = GATES / "rival_configs_lr9.json"
        if not f.exists():
            raise RuntimeError(f"missing frozen config {f} (run v6b_tune first)")
        cfg = json.loads(f.read_text())["selected"]
    for m in TUNED[layer]:
        if not cfg.get(m):
            raise RuntimeError(f"no frozen config for tuned rival {m} on {layer}")
    if lock is not None:
        locked = (lock.get("frozen_configs") or {}).get(layer)
        if locked is None or {m: locked.get(m) for m in TUNED[layer]} != {m: cfg[m] for m in TUNED[layer]}:
            raise RuntimeError(f"frozen configs on disk differ from the locked addendum ({layer})")
    return cfg


def make(name, kw):
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
    from dsswm.envs.lenta_v6 import LentaLayerEnv, lr9_problems
    return LentaLayerEnv(half, eval_task_id=eval_task_id), lr9_problems(), "response_att"


def init_env(layer, half, eps_list, eval_task_id=None):
    env, probs, outcome = build_env(layer, half, eval_task_id)
    ctx0 = build_ctx(env, probs, eps_list[0])
    J = true_policy_values(ctx0.pols, env.w, env.true_mu(outcome))
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    ctxs = {e: build_ctx(env, probs, e) for e in eps_list}
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, ctxs=ctxs)


def job(a):
    name, kw, seed, eps, stop_k, code_sha, add_sha, data_sha = a
    ctx = _ENV["ctxs"][eps]
    if stop_k is not None:
        ctx = ctx_with_stop(ctx, stop_k)
    try:
        m = make(name, kw)
        t0 = time.perf_counter()
        s, rows = run_stream_v6(_ENV["env"], m, seed, ctx.problems, ctx.eps, outcome=_ENV["outcome"], ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=False)
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
        out = {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "stop_k": int(ctx.stop_k)}
        out.update({f"rs_{k}": v for k, v in s.items()})
        out.update({"N80_pen": n80_pen, "N80_raw": int(ck[k12]) if k12 is not None else tau,
                    "k80": k12, "completed80": k12 is not None,
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


def run_jobs(task, out, jobs_all, write_markers=True, add_sha=None, data_sha=None):
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
        if rp.exists():                     # new rows invalidate an existing replica report
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
                if write_markers and (n_done % 10 == 0 or n_done == total):
                    progress(task, n_done, total, {"runs_done": n_done, "errors": len(errs)})
    if errs:
        (out / "errors.log").write_text("\n\n".join(f"{e['method']} {e['seed']}\n{e['error']}" for e in errs))
        raise RuntimeError(f"{len(errs)} jobs failed (see {out / 'errors.log'})")
    rows = []
    for l in rfile.read_text().splitlines():
        x = json.loads(l)
        if (x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
                and x.get("data_sha256") == data_sha):
            rows.append(x)
    return rows, code_sha


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def tune_select(rows, grid_of, seeds):
    """v5 selection rule: argmin geomean N80_pen over the tuning seeds among grid points with 0 false streams and
    billing_ok on every stream; ties -> first grid item."""
    sel, table = {}, {}
    for m, grid in grid_of.items():
        best = None
        table[m] = []
        for g in grid:
            key = json.dumps(g, sort_keys=True)
            rr = [r for r in rows if r["method"] == m and json.dumps(r["params"], sort_keys=True) == key]
            ok = len(rr) == len(seeds) and all(r["billing_ok"] for r in rr) and not any(r["fwer_event"] for r in rr)
            gm = geo([r["N80_pen"] for r in rr]) if rr else None
            table[m].append({"params": g, "geomean_N80_pen": gm, "n": len(rr), "eligible": ok,
                             "n_false_streams": int(sum(r["fwer_event"] for r in rr))})
            if ok and (best is None or gm < best[0] - 1e-9):
                best = (gm, g)
        sel[m] = None if best is None else best[1]
    return sel, table


def describe_rows(rows, methods, eps_list):
    out = {}
    for e in eps_list:
        for m in methods:
            rr = [r for r in rows if r["method"] == m and abs(r["eps"] - e) < 1e-15]
            if not rr:
                continue
            tau = rr[0]["tau_R"]
            d = {"n": len(rr), "geomean_N80_pen_over_tau": geo([r["N80_pen"] / tau for r in rr]),
                 "share_N80_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])),
                 "false_streams": int(sum(r["fwer_event"] for r in rr)),
                 "billing_ok_all": all(r["billing_ok"] for r in rr),
                 "sec_mean": float(np.mean([r["sec"] for r in rr])), "sec_max": float(np.max([r["sec"] for r in rr]))}
            if rr[0].get("N100_pen") is not None:
                d["geomean_N100_pen_over_tau"] = geo([r["N100_pen"] / tau for r in rr])
            out[f"{m}@{e}"] = d
    return out


def dev_ratios(rows, methods, eps_list, ref="FDC", key="N80_pen"):
    from dsswm.stats.v6_analysis import boot_idx, paired
    out = {}
    for e in eps_list:
        base = {r["seed"]: r[key] for r in rows if r["method"] == ref and abs(r["eps"] - e) < 1e-15}
        seeds = sorted(base)
        idx = boot_idx(len(seeds))
        for m in methods:
            if m == ref:
                continue
            comp = {r["seed"]: r[key] for r in rows if r["method"] == m and abs(r["eps"] - e) < 1e-15}
            if set(comp) != set(seeds) or not seeds:
                continue
            out[f"FDC/{m}@{e}"] = paired([base[s] for s in seeds], [comp[s] for s in seeds], idx)
    return out


# =============================================================================================== main
BLOCK_TASKS = {"A": ("v6a_full",), "B": ("v6b_full_a", "v6b_full_b"), "C": ("v6c_full_a", "v6c_full_b")}
SEALS = RES / "full" / "v6_seals"
DEV_BLOCK_TASKS = {"A": ("v6a_pilot",), "B": ("v6b_pilot",), "C": ("v6c_pilot",)}


def _bound_methods_check(task, T):
    if task.startswith("v6c_"):
        allm = set(TASKS["v6c_full_a"]["methods"]) | set(TASKS["v6c_full_b"]["methods"])
        if allm != {"FDC", *V5_R} or tuple(V5_R) != tuple(json.loads((WS / "plan/prereg_lock.json").read_text())
                                                           ["rigorous_set_R"]):
            raise RuntimeError("block C must bind exactly FDC + the nine v5 rivals R")


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v6 import addendum_gate
    ok, info = addendum_gate(task)
    if not ok:
        s = {"task_id": task, "status": "skipped_by_lock", "reason": info, "written_at": t_start.isoformat(),
             "eval_touched": False}
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        mark_done(task, "skipped_by_lock", info)
        print(f"[{task}] skipped_by_lock: {info}")
        return None
    return info


def make_jobs(T, seeds, cfg, grid_of):
    jobs = []
    if grid_of is not None:
        for m, grid in grid_of.items():
            for g in grid:
                for e in T["eps"]:
                    for s in seeds:
                        jobs.append((m, g, s, e, T["stop_k"]))
        return jobs
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
    heavy = ("Peace-rect", "HC-WoR")
    jobs.sort(key=lambda j: (j[0] not in heavy,))
    return jobs


def current_rows(path, add_sha, data_sha):
    """Rows of a v6 task produced by the CURRENT code, addendum and data (what a report must certify)."""
    _, code_sha = code_hashes()
    return [r for r in read_rows(path, add_sha) if r.get("code_sha256") == code_sha and r.get("addendum_sha256")
            == add_sha and r.get("data_sha256") == data_sha]


def v5_refs():
    full = RES / "full"
    v5 = read_rows(full / "r5_cr_main_a/results.jsonl") + read_rows(full / "r5_cr_main_b/results.jsonl")
    return v5, read_rows(full / "r5_cr_fwer_audit/results.jsonl")


def data_sha_for(layer, lock):
    from dsswm.envs.data_v6 import layer_data_sha
    return layer_data_sha(layer, frozen=None if lock is None else lock["data_sha256"])


def read_rows(path, add_sha=None):
    rows = []
    for l in Path(path).read_text().splitlines():
        x = json.loads(l)
        if x.get("error") is None and (add_sha is None or x.get("addendum_sha256") == add_sha):
            rows.append(x)
    return rows


def replica(task, T, lock, add_sha, t_start):
    """R1 (rerun first 10 seeds, independent process) + R2 (+ R3 for block-C eval tasks); the report is refused unless
    the task's full planned matrix exists, and it is bound to task, addendum, code, data and result content."""
    from dsswm.stats import v6_replica as R
    is_eval = T["half"] == "eval"
    base = RES / ("full" if is_eval else "pilots") / task
    data_sha = data_sha_for(T["layer"], lock)
    main_rows = current_rows(base / "results.jsonl", add_sha, data_sha)
    m = R.check_matrix(main_rows, T["methods"], T["eps"], T["seeds"])
    if not m["complete"]:
        raise SystemExit(f"[{task}] replica refused: planned matrix incomplete {m}")
    if is_eval:
        from dsswm.stats.v6_seal import verify_seal
        verify_seal(SEALS, task, main_rows, T["methods"], T["eps"], T["seeds"], code_hashes()[0], data_sha, add_sha)
    rout = base / "replica"
    rout.mkdir(parents=True, exist_ok=True)
    seeds = T["seeds"][:R.R1_N_SEEDS]
    cfg = load_configs(T["layer"], lock) if T.get("kind") is None else {}
    init_env(T["layer"], T["half"], tuple(T["eps"]), eval_task_id=task if is_eval else None)
    rep_rows, _ = run_jobs(task + "_replica", rout, make_jobs(T, seeds, cfg, None), add_sha=add_sha,
                           data_sha=data_sha)
    v5m = v5f = None
    if "R3" in R.required_rules(task, is_eval):
        v5m, v5f = v5_refs()
    per_sha, _ = code_hashes()
    rep_rows = current_rows(rout / "results.jsonl", add_sha, data_sha)
    rep = R.build_report(task, is_eval, main_rows, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha, per_sha,
                         data_sha, v5m, v5f)
    rep["written_at"] = datetime.now().isoformat()
    (base / "replica_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"task": task, "replica": rep["status"]}))


def analyse(block, dev):
    from dsswm.stats import v6_analysis as VA
    from dsswm.stats import v6_replica as R
    if dev:
        tasks, add_sha, lock = DEV_BLOCK_TASKS[block], None, None
        T0 = TASKS[tasks[0]]
        spec = dict(VA.BLOCK_SPEC[block])
        spec.update(seeds=tuple(T0["seeds"]), methods=tuple(T0["methods"]), eps=tuple(T0["eps"]))
        base = RES / "pilots"
    else:
        from dsswm.stats.prereg_v6 import load_locked_addendum
        lock = load_locked_addendum(None)          # schema, hashes, code, INPUT and DATA binding re-checked here
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
        v5m = v5f = None
        if "R3" in R.required_rules(t, not dev):
            v5m, v5f = v5_refs()
        sealed = None
        if not dev:
            from dsswm.stats.v6_seal import verify_seal
            sealed = verify_seal(SEALS, t, cur, T["methods"], T["eps"], T["seeds"], per_sha, data_sha,
                                 add_sha)["seal"]["results_content_sha256"]
        sts.append(R.validate_report(rep, t, not dev, cur, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha,
                                     per_sha, data_sha, v5m, v5f, sealed_content_sha256=sealed))
        rows += cur
        reps.append(rep)
    replica_status = {"status": "pass" if all(x == "pass" for x in sts) else "fail"}
    res = VA.analyse_block(block, rows, replica_status, spec=spec)
    res.update({"mode": "dev_exploratory" if dev else "eval", "addendum_sha256": add_sha, "tasks": list(tasks),
                "replica_statuses": dict(zip(tasks, sts)),
                "replica_content_sha256": [r["results_content_sha256"] for r in reps],
                "code_sha256": per_sha, "written_at": datetime.now().isoformat()})
    out = base / f"v6_analysis_{block}{'_dev' if dev else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"block": block, "decision": res["decision"]}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--replica", action="store_true", help="run R1-R3 for --task (after the task completed)")
    ap.add_argument("--analyse", choices=("A", "B", "C"))
    ap.add_argument("--dev", action="store_true", help="with --analyse: exploratory analysis of the dev pilots")
    args = ap.parse_args()
    if args.analyse:
        analyse(args.analyse, args.dev)
        return
    task = args.task
    T = TASKS[task]
    _bound_methods_check(task, T)
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
        replica(task, T, lock, add_sha, t_start)
        return
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    eps_list = tuple(T["eps"])
    kind = T.get("kind")
    cfg = load_configs(T["layer"], lock) if kind is None else {}
    data_sha = data_sha_for(T["layer"], lock)        # raw data verified against the lock (eval) / recorded (dev)
    init_env(T["layer"], T["half"], eps_list, eval_task_id=task if is_eval else None)
    env = _ENV["env"]
    if kind == "tune_hc":
        from dsswm.baselines.wor_betting_v6 import HC_GRID
        grid_of = {"HC-WoR": [dict(g) for g in HC_GRID]}
    elif kind == "tune_lr9":
        from dsswm.baselines.wor_betting_v6 import HC_GRID
        v5grid = json.loads((WS / "plan/prereg_lock.json").read_text())["rival_configs"]["grid"]
        grid_of = {"B4-bal": v5grid["B4-bal"], "Hait-SW": v5grid["Hait-SW"],
                   "B2-rect": [{"explore_c": 1.0}, {"explore_c": 0.5}], "HC-WoR": [dict(g) for g in HC_GRID]}
    elif kind == "sens_hc":
        grid_of = {"HC-WoR": [{"schedule": "nstar", "c": 0.75, "target_frac": t} for t in (0.5, 0.65, 0.8, 1.0)]}
    else:
        grid_of = None
    jobs = make_jobs(T, T["seeds"], cfg, grid_of)
    rows, code_sha = run_jobs(task, out, jobs, add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    t_end = datetime.now()
    summary = {"task_id": task, "status": "complete", "layer": T["layer"], "half": T["half"],
               "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]), "eps": list(eps_list),
               "stop_k": T["stop_k"], "tau_R": int(env.tau_R), "n_rows": len(rows), "expected_rows": len(jobs),
               "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
               "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
               "cpu_sec_sum": round(float(sum(r["sec"] for r in rows)), 1),
               "timing_note": "4 worker processes, BLAS=1; concurrent with other host load (biased up)",
               "code_sha256": per_sha, "code_sha256_combined": code_sha, "addendum_sha256": add_sha,
               "data_sha256": data_sha,
               "eval_touched": bool(is_eval)}
    if grid_of is None:
        from dsswm.stats.v6_replica import COMPARISON_GROUPS, check_r2
        summary["R2_design_identity"] = check_r2(rows, COMPARISON_GROUPS.get(task, ()), T["seeds"], eps_list)
    if kind in ("tune_hc", "tune_lr9"):
        sel, table = tune_select(rows, grid_of, T["seeds"])
        summary["tuning"] = {"selected": sel, "table": table,
                             "rule": "argmin geomean N80_pen over tuning seeds among grid points with 0 false streams "
                                     "and billing_ok; ties -> first grid item (lock v5 rule)"}
        GATES.mkdir(parents=True, exist_ok=True)
        gname = "hc_config_cr9.json" if kind == "tune_hc" else "rival_configs_lr9.json"
        payload = {"task_id": task, "frozen_at": t_end.isoformat(), "layer": T["layer"], "eps": list(eps_list),
                   "seeds": [T["seeds"][0], T["seeds"][-1]], "selected": sel["HC-WoR"] if kind == "tune_hc" else sel,
                   "table": table, "rule": summary["tuning"]["rule"], "code_sha256": per_sha}
        (GATES / gname).write_text(json.dumps(payload, indent=1))
        summary["gate_file"] = str((GATES / gname).relative_to(WS))
    elif kind == "sens_hc":
        _, table = tune_select(rows, grid_of, T["seeds"])
        summary["sensitivity_report_only"] = table
    else:
        summary["configs"] = {m: (None if m in NO_PARAM else cfg.get(m)) for m in T["methods"]}
        if not is_eval:   # dev pilots only: exploratory estimates (never for eval blocks inside the block)
            summary["describe"] = describe_rows(rows, T["methods"], eps_list)
            key = "N100_pen" if T["stop_k"] == 15 else "N80_pen"
            summary["dev_ratios_exploratory"] = dev_ratios(rows, T["methods"], eps_list, key=key)
    if is_eval:
        from dsswm.stats.v6_seal import make_seal, write_and_commit_seal
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
