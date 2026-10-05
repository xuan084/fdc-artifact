"""r5_cr_k60_sens: report-only sensitivity block -- common finer certificate grid K = 60 (CR9 eval half).

Usage:
  python run_r5_cr_k60_sens.py --mode full    # eval seeds 30000-30099 from lock v5 eval_tasks['r5_cr_k60_sens']
  python run_r5_cr_k60_sens.py --mode pilot   # smoke/timing on DEV seeds 900-909 only (eval seeds never touched)

Protocol (lock v5, plan/prereg_lock.json, eval_tasks['r5_cr_k60_sens']):
* start-up: dsswm.stats.prereg.lock_gate(task) (== assert_locked(version=5, task_id) incl. frozen-code drift check);
  on refusal summary.status='skipped_by_lock' and exit 0.
* setting: CR9, eps* = 0.001, Q = 15, stop at 12/15, delta = 0.05, K = 60 log-spaced checkpoints from 50,000 to
  tau_R (fr.checkpoints, the same rule as the K = 20 grid). The K = 60 grid is common to every method; every method
  recomputes its own time-union budget from len(ctx.checkpoints) = 60 by its own rule (FDC: beta = ln(sum_q
  |Pi_Bq| * 60 / 0.045), x_v = ln(2 S A * 60 / 0.005); QFC-pool: r4 Q*-union ledger at K = 60; B2-fav-tight:
  ln(sum_q |Pi_Bq| * 60 / delta); the rect rivals use per-cell anytime WoR CSs, which need no time union).
* methods / seeds / rival configs: from the lock (rival_configs.json sha256 checked).
* write-ahead results.jsonl (fsync per row, resumable; rows keyed by code hash); NO analysis inside the block.
* report only: not a gate, never mixed with K = 20 numbers.
The stream engine / job / bookkeeping is reused verbatim from run_r5_cr_main.py; only K and the ledger audit differ.
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run: timings biased up).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import run_r5_cr_main as crm  # noqa: E402
from dsswm.stats import prereg  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402

TASK = "r5_cr_k60_sens"
RES = crm.RES
WS = crm.WS
N_WORKERS = crm.N_WORKERS
DEV_PILOT_SEEDS = crm.DEV_PILOT_SEEDS
crm.FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/b4_bal.py",
             "dsswm/baselines/uniform_rs.py", "dsswm/baselines/combgame_joint.py", "dsswm/baselines/b2_rect_fe.py",
             "dsswm/baselines/peace_frontier.py", "dsswm/baselines/plugin_r5.py", "dsswm/evidence/ext_plugin.py",
             "dsswm/streams/r5_registry.py", "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py",
             "dsswm/certify/quadknap.py", "dsswm/envs/pool_replay.py", "dsswm/stats/prereg.py",
             "run_r5_cr_main.py", "run_r5_cr_k60_sens.py"]


def init_env(half, eps, K):
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", half)
    ctx = build_ctx(env, fr.cr_problems("visit"), eps, K=K)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    crm._ENV.update(env=env, ctx=ctx, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]), eps=eps)


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def job(a):
    """crm.job (identical row schema) + the method's own post-setup description (its K-dependent budget)."""
    r = crm.job(a)
    if r.get("error") is None:
        try:
            m = reg.make_method(a[0], **a[1])
            m.setup(crm._ENV["ctx"])
            d = m.describe() if hasattr(m, "describe") else {}
            r["method_budget"] = _jsonable({k: v for k, v in d.items() if k != "alloc_p"})
        except Exception:  # noqa: BLE001
            r["method_budget"] = {"error": traceback.format_exc()[-400:]}
        r["task_id"] = TASK
    return r


def ledger_audit(ctx):
    """Expected K = 60 budgets (from the formulas in the lock) vs what each method actually sets up."""
    from dsswm.baselines.fdc import fdc_ledger
    from dsswm.baselines.frontier_common import qfc_default_params
    K = len(ctx.checkpoints)
    U = int(np.asarray(ctx.feas).sum())
    SA = int(ctx.S * ctx.A)
    exp_fdc = {"beta": math.log(U * K / 0.045), "x_v": math.log(2 * SA * K / 0.005)}
    led = fdc_ledger(ctx)
    fdc = reg.make_method("FDC")
    fdc.setup(ctx)
    p = dict(fdc.params or {})
    qfc = reg.make_method("QFC-pool")
    qfc.setup(ctx)
    qexp = qfc_default_params(ctx.S, ctx.A, ctx.Q, K=K)
    out = {"K": K, "union_size_sum_q_Pi_Bq": U, "S_A": SA,
           "FDC_expected": exp_fdc, "FDC_ledger": {k: led[k] for k in ("beta", "x_v", "bound_main", "bound_var", "K")},
           "FDC_method_params": _jsonable(p),
           "FDC_ok": abs(led["beta"] - exp_fdc["beta"]) < 1e-12 and abs(led["x_v"] - exp_fdc["x_v"]) < 1e-12
                     and abs(float(p.get("L1", p.get("beta", np.nan))) - exp_fdc["beta"]) < 1e-12
                     and abs(float(p.get("x_v", np.nan)) - exp_fdc["x_v"]) < 1e-12,
           "QFC_pool_params": _jsonable(qfc.params), "QFC_pool_expected": _jsonable(qexp),
           "QFC_pool_ok": abs(qfc.params["L1"] - qexp["L1"]) < 1e-12 and qfc.params["K"] == K,
           "B2_fav_tight_expected_beta": math.log(U * K / float(ctx.delta))}
    try:
        b2 = reg.make_method("B2-fav-tight")
        b2.setup(ctx)
        out["B2_fav_tight_describe"] = _jsonable({k: v for k, v in b2.describe().items() if k != "alloc_p"})
    except Exception:  # noqa: BLE001
        out["B2_fav_tight_describe"] = {"error": traceback.format_exc()[-400:]}
    return out


def run_block(mode, out, lock, seeds=None, half=None):
    et = lock["eval_tasks"][TASK]
    methods = list(et["methods"])
    K = int(et["K"])
    if seeds is None:
        seeds = crm.parse_range(et["seeds"])
    if half is None:
        half = et["half"]
    assert et["layer"] == "CR9" and K == 60
    rc_file = WS / lock["rival_configs"]["path"]
    if crm.sha_file(rc_file) != lock["rival_configs"]["sha256"]:
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
    init_env(half, eps, K)
    ctx = crm._ENV["ctx"]
    ck = [int(x) for x in ctx.checkpoints]
    assert abs(ctx.eps - 0.001) < 1e-12 and len(ck) == K and ctx.stop_k == 12 and ctx.Q == 15
    sref = lock["structural_ctx"][f"CR9_{half}"]
    if int(ctx.tau_R) != int(sref["tau_R"]) or np.asarray(crm._ENV["env"].pool_sizes).tolist() != sref["pool_sizes"]:
        raise RuntimeError(f"structural ctx differs from lock CR9_{half}")
    assert ck[0] == 50000 and ck[-1] == int(ctx.tau_R) and all(b > a for a, b in zip(ck, ck[1:]))
    audit = ledger_audit(ctx)
    (out / "ledger_audit.json").write_text(json.dumps(audit, indent=1))
    if not (audit["FDC_ok"] and audit["QFC_pool_ok"]):
        raise RuntimeError(f"K={K} ledger audit failed: {audit}")
    per_sha, code_sha = crm.code_hashes()
    print(f"[{TASK}/{mode}] half={half} K={K} tau_R={ctx.tau_R} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) "
          f"methods={methods} FDC beta={audit['FDC_ledger']['beta']:.4f} code_sha={code_sha[:16]}", flush=True)

    rfile = out / "results.jsonl"
    done = set()
    if rfile.exists():
        keep = []
        for l in rfile.read_text().splitlines():
            try:
                x = json.loads(l)
            except ValueError:
                continue
            if x.get("error") is None and x.get("code_sha256") == code_sha and (x["method"], x["seed"]) not in done:
                done.add((x["method"], x["seed"]))
                keep.append(l)
        rfile.write_text("".join(k + "\n" for k in keep))
    order = [m for m in ("Peace-rect",) if m in methods] + [m for m in methods if m != "Peace-rect"]
    jobs = [(m, kws[m], s, code_sha) for m in order for s in seeds if (m, s) not in done]
    total = len(methods) * len(seeds)
    print(f"resume: {len(done)} done, {len(jobs)} to run", flush=True)
    crm.progress(TASK, len(done), total)
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
                if n_done % 10 == 0 or n_done == total:
                    crm.progress(TASK, n_done, total, {"runs_done": n_done, "errors": len(errs)})
                if n_done % 25 == 0 or n_done == total:
                    print(f"[{n_done}/{total}] last={r['method']} seed={r['seed']} sec={r['sec']} "
                          f"errors={len(errs)}", flush=True)
    if errs:
        (out / "errors.log").write_text("\n\n".join(f"{e['method']} {e['seed']}\n{e['error']}" for e in errs))
        raise RuntimeError(f"{len(errs)} stream jobs failed (see errors.log)")

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
    canon = "\n".join(json.dumps({k: v for k, v in rows[(m, s)].items() if k not in ("sec", "rs_sec_plan",
                                                                                    "rs_sec_cert", "rs_sec_total")},
                                 sort_keys=True) for m in methods for s in seeds if (m, s) in rows)
    t_end = datetime.now()
    timing = {m: {"sec_mean": float(np.mean([rows[(m, s)]["sec"] for s in seeds if (m, s) in rows] or [0])),
                  "sec_max": float(np.max([rows[(m, s)]["sec"] for s in seeds if (m, s) in rows] or [0]))}
              for m in methods}
    summary = {"task_id": TASK, "mode": mode, "status": "complete" if not missing and not bill_bad else "incomplete",
               "role": "report-only sensitivity (common K = 60 grid); not a gate; never mixed with K = 20 numbers",
               "started_at": sf.read_text().strip(), "ended_at": t_end.isoformat(),
               "wall_min_this_invocation": round((t_end - t_start).total_seconds() / 60, 2),
               "timing_note": "concurrent run (4 workers, BLAS=1; other r5 blocks share the 20-core host): biased up",
               "setting": {"layer": "CR9", "half": half, "eps": ctx.eps, "K": K, "checkpoints": ck,
                           "stop_k": int(ctx.stop_k), "Q": ctx.Q, "delta": ctx.delta, "tau_R": int(ctx.tau_R),
                           "seeds": [seeds[0], seeds[-1]], "n_streams": len(seeds)},
               "ledger_audit": audit, "methods": methods, "params": kws,
               "lock": {"version": lock["version"], "status": lock["status"], "sha256": lock["sha256"],
                        "git_commit": lock["git_commit"]},
               "n_rows": len(rows), "expected_rows": total, "missing": missing[:50], "n_missing": len(missing),
               "billing_ok_all": not bill_bad, "billing_bad": bill_bad[:50],
               "results_canonical_sha256": hashlib.sha256(canon.encode()).hexdigest(),
               "results_file_sha256": crm.sha_file(rfile), "code_sha256_combined": code_sha, "code_sha256": per_sha,
               "git": crm.git_info(), "per_method_timing_sec": timing, "eval_seeds_touched": half == "eval",
               "note": "no analysis inside the block; analysis happens in r5_analysis_aggregate"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary, rows


def lock_path_test():
    real = prereg.load_lock(prereg.lock_path(5))
    cases = {"real_lock": prereg.lock_gate(TASK, version=5)[0]}

    def gate(lk):
        try:
            prereg.check_lock(lk, 5, TASK)
            return {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "reason": str(e)[:300]}
    d = copy.deepcopy(real); d["status"] = "draft"; cases["status_draft"] = gate(d)
    d = copy.deepcopy(real); d["status"] = "locked_no_eval"; cases["locked_no_eval"] = gate(d)
    d = copy.deepcopy(real); d["eval_tasks"][TASK]["K"] = 20; cases["hash_tamper_K"] = gate(d)
    d = copy.deepcopy(real); d["version"] = 4; cases["version_mismatch"] = gate(d)
    cases["code_drift_detected_on_wrong_root"] = bool(prereg.frozen_code_drift(real, CODE / "__nonexistent__"))
    cases["code_drift_real_root"] = prereg.frozen_code_drift(real)
    ok = cases["real_lock"] and all(not cases[k]["ok"] for k in ("status_draft", "locked_no_eval", "hash_tamper_K",
                                                                     "version_mismatch")) \
        and cases["code_drift_detected_on_wrong_root"] and not cases["code_drift_real_root"]
    return {"all_pass": bool(ok), "cases": cases}


def pilot_extras(out, s, rows, lock):
    et = lock["eval_tasks"][TASK]
    methods = list(et["methods"])
    seeds = DEV_PILOT_SEEDS
    # timing projection for 100 eval streams: LPT on 4 workers, x1.2 safety + 2 min setup
    sec = {m: float(np.mean([rows[(m, x)]["sec"] for x in seeds])) for m in methods}
    loads = np.zeros(N_WORKERS)
    for m in sorted(methods, key=lambda m: -sec[m]):
        for _ in range(int(et["n_streams"])):
            loads[np.argmin(loads)] += sec[m]
    proj = float(loads.max() / 60 * 1.2 + 2)
    # descriptive dev-only sanity: K = 60 vs the K = 20 dev pilot rows of the same seeds (never eval)
    k20 = {}
    for t in ("r5_cr_main_a", "r5_cr_main_b", "r5_cr_plugin_a", "r5_cr_plugin_b"):
        p = RES / "pilots" / t / "results.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                try:
                    x = json.loads(l)
                except ValueError:
                    continue
                if x.get("error") is None and x["seed"] in seeds:
                    k20.setdefault((x["method"], x["seed"]), x)
    per = {}
    for m in methods:
        rr = [rows[(m, x)] for x in seeds]
        d = {"sec_mean": round(sec[m], 2), "sec_max": round(max(r["sec"] for r in rr), 2),
             "geomean_N80_pen_K60_dev": float(np.exp(np.mean(np.log([r["N80_pen"] for r in rr])))),
             "completed": sum(r["completed"] for r in rr), "fwer_streams": sum(r["fwer_event"] for r in rr),
             "billing_ok_all": all(r["billing_ok"] for r in rr),
             "budget": rr[0].get("method_budget", {}).get("params")}
        ref = [k20[(m, x)] for x in seeds if (m, x) in k20]
        if len(ref) == len(seeds):
            d["geomean_N80_pen_K20_dev_same_seeds"] = float(np.exp(np.mean(np.log([r["N80_pen"] for r in ref]))))
            d["completed_K20_dev"] = sum(r["completed"] for r in ref)
        per[m] = d
    samples = [{k: rows[(m, x)][k] for k in ("method", "seed", "N80_pen", "N80_raw", "completed", "fwer_event",
                                              "cert_k", "billing_ok")}
               for x in (900, 905, 909) for m in methods]
    lt = json.loads((out / "lock_path_test.json").read_text())
    crit = {"billing_ok_all": s["billing_ok_all"], "no_exceptions": s["n_missing"] == 0,
            "projected_full_min_le_55": proj <= 55, "lock_assertion_path_tested": lt["all_pass"],
            "ledger_audit_K60_ok": s["ledger_audit"]["FDC_ok"] and s["ledger_audit"]["QFC_pool_ok"]}
    go = all(crit.values())
    rep = {"task_id": TASK, "go_no_go": "GO" if go else "NO_GO", "criteria": crit,
           "projected_full_block_min": round(proj, 1),
           "projection_rule": "mean dev sec/stream per method x 100 streams, LPT on 4 workers, x1.2 + 2 min",
           "per_method_dev": per, "samples": samples, "eval_seeds_touched": False,
           "timing_note": "concurrent run (r5_cr_factorial / fwer_audit / cr12_scale share the host)",
           "k20_comparison_note": "dev seeds 900-909 only, descriptive sanity; not reported with eval numbers"}
    (out / "pilot_checks.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "full"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    ok, lock = prereg.lock_gate(TASK, version=5)
    if not ok:
        s = crm.skipped_summary(TASK, out, lock, a.mode)
        crm.mark_done(TASK, "success", f"skipped_by_lock: {lock}")
        print(json.dumps(s), flush=True)
        return
    try:
        if a.mode == "full":
            s, _ = run_block("full", out, lock)
            txt = f"full {s['status']} rows={s['n_rows']}/{s['expected_rows']} billing_ok={s['billing_ok_all']}"
            crm.mark_done(TASK, "success" if s["status"] == "complete" else "failed", txt)
        else:
            out.mkdir(parents=True, exist_ok=True)
            if (out / "results.jsonl").exists() and os.environ.get("PILOT_FRESH", "1") == "1":
                (out / "results.jsonl").unlink()
            lt = lock_path_test()
            (out / "lock_path_test.json").write_text(json.dumps(lt, indent=1))
            print(f"lock path test all_pass={lt['all_pass']}", flush=True)
            t0 = time.perf_counter()
            s, rows = run_block("pilot", out, lock, seeds=DEV_PILOT_SEEDS, half="dev")
            rep = pilot_extras(out, s, rows, lock)
            rep["pilot_run_wall_min"] = round((time.perf_counter() - t0) / 60, 2)
            (out / "pilot_checks.json").write_text(json.dumps(rep, indent=1, default=str))
            txt = f"pilot {rep['go_no_go']} | proj full {rep['projected_full_block_min']} min | {rep['criteria']}"
            crm.mark_done(TASK, "success", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        crm.mark_done(TASK, "failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
