"""r4_g_fac_a: Gate G-fac' part A -- FCC(PDL) safety + non-vacuity + runtime at T_ext = 200k on MS-H dev (800-839).

Methodology "FCC 锁前修订（r4，2026-10-02）" F4 (overrides §1.6 / §3 G-fac):
  FCC = certify.fcc.decide_pdl (pair-difference certificate), UNIFORM random behaviour, no BnB.
  MS-H dev instances 800-839 (placements frozen by r4_g_haz), ONE stream per instance (noise seed 42, n0 = 20 initial
  rounds as in the harness), 15 problems, original horizons H in {6, 8} (asserted by MSInstance), eps = 0.02,
  delta = 0.05, T_ext = 200000 new interaction rounds, 8 fixed checkpoints {1k,3k,6k,20k,60k,100k,150k,200k}.
  All 15 problems are checked on the same evidence at every checkpoint (one uniform behaviour stream serves every
  problem: the behaviour does not depend on the problem). A problem is decided at its FIRST certifying checkpoint
  (N_cert); the decision is never revised. No decision at a non-checkpoint time.
  Gate G-fac' (verdict in r4_gate_decision): dev false streams 0/40 AND cert rate at eps=0.02 >= 0.20 AND per-instance
  runtime projection keeps an evaluation chunk (50 instances, 4 workers, FCC + JPC) <= 55 min.
  Recorded: eps_need curves, true top-2 gap and candidate spread, N_cert / 6000, decoupling amplification (3 problems,
  BnB depth 8 approximation). FCC only -- no baseline is run. eps / T_ext / checkpoints are fixed (never tuned).

Usage: run_r4_g_fac_a.py [--mode pilot] [--workers 4]
Concurrent run: <= 4 CPU workers, BLAS threads = 1, CPU only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES = WS / "exp" / "results"
TASK = "r4_g_fac_a"
PLANNED_MIN = 40
SEEDS = list(range(800, 840))
EPS, DELTA = 0.02, 0.05
T_EXT = 200_000
CPS = [1000, 3000, 6000, 20000, 60000, 100000, 150000, 200000]
TMAX_R3 = 6000
NONVAC_MIN = 0.20
CHUNK_MIN_LIMIT = 55.0
EVAL_CHUNK_INSTANCES = 50
EVAL_WORKERS = 4
BEHAVIOUR_CODE = 161                      # FCC method code in ms_r4.METHOD_CODE
HAZ_PLC = RES / "pilots" / "r4_g_haz" / "placements"
HAZ_PARTS = RES / "pilots" / "r4_g_haz" / "parts"
CODE_FILES = ["dsswm/certify/fcc.py", "dsswm/streams/ms_r4.py", "dsswm/stats/cp.py", "run_r4_g_fac_a.py"]


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


# ============================================================================================ worker
def _instance_job(seed, plc, out_dir):
    """One MS-H instance: one uniform-behaviour stream to T_EXT, decide_pdl for all 15 problems at every checkpoint."""
    fn = out_dir / "parts" / f"i{seed}.json"
    if fn.exists():
        return {"seed": seed, "skipped": True}
    import torch
    torch.set_num_threads(1)
    from dsswm.certify.fcc import FCCCertifier, cell_probs, decide_pdl
    from dsswm.streams import ms_r4 as M
    t0 = time.perf_counter()
    try:
        inst = M.build_msh(seed, plc)
        env, init = inst.make_platform()
        P = env.aspace
        cert = FCCCertifier(P.L, P.R, 2, P, rho_ret=0.3, delta=DELTA)
        sol = cert.solver
        for o in init:
            cert.observe(o)
        theta = cell_probs(env.true_params(), cert.cells, c=1.0)
        # truth cross-check: certifier forward model at the true cell parameters vs the harness J_true
        plans = [[sol.plan(pi, q.loads0, q.engaged0, q.H) for pi in q.policies] for q in inst.problems]
        jt_sol = [np.array([sol.solve(pl, q.utility.w, q.utility.w_ret, q.utility.c_q, theta, theta)[0] for pl in pls])
                  for q, pls in zip(inst.problems, plans)]
        truth_diff = max(float(np.max(np.abs(a - np.asarray(b)))) for a, b in zip(jt_sol, inst.J_true))
        del plans
        build_s = time.perf_counter() - t0
        rng = np.random.default_rng([seed, M.NOISE, BEHAVIOUR_CODE])
        jps = [dict() for _ in inst.problems]
        nq = len(inst.problems)
        first = [None] * nq                   # (checkpoint, k_hat) of the first certification
        curves = [[] for _ in range(nq)]
        anytime_false = [0] * nq              # certifying checkpoints whose k_hat is false (anytime caliber)
        contain_viol = 0
        cp_rows = []
        done, step_s, dec_s = 0, 0.0, 0.0
        for T in CPS:
            ts = time.perf_counter()
            for _ in range(T - done):
                cert.observe(env.step(int(rng.integers(P.n))))
            done = T
            step_s += time.perf_counter() - ts
            td = time.perf_counter()
            lo, hi = cert.cs.intervals()
            cover = bool(((theta >= lo) & (theta <= hi)).all())
            for k, q in enumerate(inst.problems):
                Jt = np.asarray(inst.J_true[k], float)
                st, kh, e, D = decide_pdl(sol, q, lo, hi, EPS, jps=jps[k])
                # PDL containment check of every true pair difference (diagnostic; truth only read by the harness)
                tv = Jt[:, None] - Jt[None, :]
                mask = np.isfinite(D)
                contain_viol += int(np.sum(D[mask] < tv[mask] - 1e-9))
                reg = float(Jt.max() - Jt[kh])
                cert_now = st == "CERTIFIED"
                if cert_now and reg > EPS:
                    anytime_false[k] += 1
                if cert_now and first[k] is None:
                    first[k] = (T, int(kh), reg)
                curves[k].append(float(e))
                cp_rows.append({"T": T, "problem": k, "eps_need": float(e), "k_hat": int(kh), "certified": cert_now,
                                "regret_k_hat": reg})
            dec_s += time.perf_counter() - td
            cp_rows.append({"T": T, "cs_cover_truth": cover, "cell_width_med": float(np.median(hi - lo))})
        rows = []
        for k, q in enumerate(inst.problems):
            Jt = np.asarray(inst.J_true[k], float)
            srt = np.sort(Jt)
            cert_ = first[k] is not None
            reg = first[k][2] if cert_ else None
            rows.append({"instance": seed, "layer": "MS-H", "method": "FCC-PDL-unif", "problem": k,
                         "pid": q.pid, "H": int(q.H), "n_policies": len(q.policies), "noise_seed": M.NOISE,
                         "status": "CERTIFIED" if cert_ else "NEED_DATA", "certified_policy": first[k][1] if cert_
                         else None, "N_cert": first[k][0] if cert_ else None,
                         "N_cert_over_6000": first[k][0] / TMAX_R3 if cert_ else None,
                         "true_regret": reg, "false_cert": bool(cert_ and reg > EPS),
                         "anytime_false_checkpoints": anytime_false[k],
                         "true_gap": float(srt[-1] - srt[-2]), "spread": float(srt[-1] - srt[0]),
                         "gap_layer": M.gap_layer(float(srt[-1] - srt[-2])),
                         "eps_need_curve": curves[k], "checkpoints": CPS, "eps": EPS, "delta": DELTA,
                         "T_ext": T_EXT, **inst.meta[k]})
        rec = {"seed": seed, "rows": rows, "cp_rows": cp_rows, "truth_solver_max_abs_diff": truth_diff,
               "pdl_containment_violations": contain_viol, "n_cells": int(cert.cells.n),
               "env_steps": int(env.n_steps), "env_resets": int(env.n_resets),
               "build_s": build_s, "step_s": step_s, "decide_s": dec_s, "sec": time.perf_counter() - t0,
               "error": None}
    except Exception as e:  # noqa: BLE001
        rec = {"seed": seed, "rows": [], "error": repr(e), "tb": traceback.format_exc()[-3000:],
               "sec": time.perf_counter() - t0}
    tmp = fn.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, fn)
    return {"seed": seed, "skipped": False, "sec": rec["sec"], "error": rec["error"]}


def _amplification_job(seed, plc, n_problems=3, max_cells=8, max_nodes=96, budget_s=40.0):
    """Decoupling amplification at T_EXT on the first 3 problems of instance `seed` (k_hat policy and the PDL pair
    (k_hat, runner-up)): decoupled width / shared-parameter width, where the shared width is approximated (a) from above
    by BnB on the max_cells most influential cells (valid bounds -> amplification lower estimate) and (b) from below by
    vertex local search + random box points (-> amplification upper estimate)."""
    import torch
    torch.set_num_threads(1)
    from dsswm.certify.fcc import FCCCertifier, decide_pdl, value_bounds_bnb
    from dsswm.streams import ms_r4 as M
    inst = M.build_msh(seed, plc)
    env, init = inst.make_platform()
    P = env.aspace
    cert = FCCCertifier(P.L, P.R, 2, P, rho_ret=0.3, delta=DELTA)
    for o in init:
        cert.observe(o)
    rng = np.random.default_rng([seed, M.NOISE, BEHAVIOUR_CODE])       # same behaviour stream as the main job
    for _ in range(T_EXT):
        cert.observe(env.step(int(rng.integers(P.n))))
    lo, hi = cert.cs.intervals()
    sol = cert.solver
    rs = np.random.default_rng([seed, 8806])
    out = []
    for k, q in enumerate(inst.problems[:n_problems]):
        st, kh, e, D = decide_pdl(sol, q, lo, hi, EPS)
        plan = sol.plan(q.policies[kh], q.loads0, q.engaged0, q.H)
        u = q.utility
        jl, jh = sol.solve(plan, u.w, u.w_ret, u.c_q, lo, hi)
        used = sorted(sol.used_cells(plan))
        pts = []
        for _ in range(300):
            th = lo + (hi - lo) * rs.random(len(lo))
            if rs.random() < 0.5:
                th = np.where(rs.random(len(lo)) < 0.5, lo, hi)
            pts.append(sol.solve(plan, u.w, u.w_ret, u.c_q, th, th)[0])
        for sense in (1, -1):
            for _r in range(6):
                th = np.where(rs.random(len(lo)) < 0.5, lo, hi)
                cur = sol.solve(plan, u.w, u.w_ret, u.c_q, th, th)[0]
                improved = True
                while improved:
                    improved = False
                    for c in used:
                        th2 = th.copy()
                        th2[c] = hi[c] if th[c] == lo[c] else lo[c]
                        v = sol.solve(plan, u.w, u.w_ret, u.c_q, th2, th2)[0]
                        if sense * v > sense * cur + 1e-15:
                            th, cur, improved = th2, v, True
                pts.append(cur)
        tb = time.perf_counter()
        bh = value_bounds_bnb(sol, plan, u, lo, hi, max_cells=max_cells, max_nodes=max_nodes, time_budget_s=budget_s,
                              sense="hi")
        bl = value_bounds_bnb(sol, plan, u, lo, hi, max_cells=max_cells, max_nodes=max_nodes, time_budget_s=budget_s,
                              sense="lo")
        dec_w = jh - jl
        bnb_w = bh["bnb"] - bl["bnb"]
        sh_w = max(pts) - min(pts)
        colmax = D.max(axis=0)
        out.append({"instance": seed, "problem": k, "pid": q.pid, "H": int(q.H), "k_hat": int(kh), "eps_need_T_ext": e,
                    "n_used_cells": len(used), "dec_lo": jl, "dec_hi": jh, "dec_width": dec_w,
                    "bnb_lo": bl["bnb"], "bnb_hi": bh["bnb"], "bnb_width": bnb_w,
                    "bnb_nodes": bh["nodes"] + bl["nodes"], "bnb_s": time.perf_counter() - tb,
                    "shared_search_width": sh_w,
                    "amp_lower_dec_over_bnb": dec_w / max(bnb_w, 1e-12),
                    "amp_upper_dec_over_search": dec_w / max(sh_w, 1e-12),
                    "pdl_runner_up_colmax_sorted": sorted(float(x) for x in colmax)[:3]})
    return out


def run_pool(fn, jobs, workers, phase, step, total):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = 0
    res = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
                msg = "cached" if isinstance(r, dict) and r.get("skipped") else (
                    f"{r.get('sec', 0):.0f}s err={r.get('error')}" if isinstance(r, dict) else "ok")
            except Exception as e:  # noqa: BLE001
                r, msg = None, f"FATAL {e!r}"
            res.append(r)
            done += 1
            progress(step, total, phase, {"done": done, "total": len(jobs), "last": f"i{j[0]}"})
            log(f"[{phase}] i{j[0]} {msg} ({done}/{len(jobs)})")
    return res


def cp_upper(x, n, alpha=0.05):
    from scipy.stats import beta
    return 1.0 if x >= n else float(beta.ppf(1 - alpha, x + 1, n - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    for sub in ("parts", "placements", "samples"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "seed": 42, "started": start_iso, "candidate_id": "cand_qfc",
               "gate": "G-fac' part A (MS-H)", "dev_seeds": [SEEDS[0], SEEDS[-1]],
               "constants": {"eps": EPS, "delta": DELTA, "T_ext": T_EXT, "checkpoints": CPS, "problems_per_stream": 15,
                             "streams_per_instance": 1, "noise_seed": 42, "n0_initial_rounds": 20,
                             "behaviour": "uniform random over the action space, rng [seed, 42, 161]",
                             "certificate": "decide_pdl (pair-difference, decoupled interval DP, no BnB)",
                             "round_count": "new interaction rounds after the n0 = 20 initial rounds",
                             "horizons": "original H in {6, 8} (asserted)"},
               "timing_note": "并发运行 (<= 4 CPU workers, BLAS threads = 1, CPU only; r4_g_fac_b ran concurrently)"}
    status = "success"
    try:
        # ------------------------------------------------------------------ 0. placements (frozen by r4_g_haz)
        progress(0, 4, "placements")
        plc = {}
        for s in SEEDS:
            dst = out_dir / "placements" / f"msh_i{s}.json"
            if not dst.exists():
                shutil.copy(HAZ_PLC / f"msh_i{s}.json", dst)
            plc[s] = json.loads(dst.read_text())
        summary["placement"] = {"source": "exp/results/pilots/r4_g_haz/placements (identical deterministic MS-H)",
                                "n_full_15": int(sum(plc[s]["fill"] >= 15 for s in SEEDS)),
                                "underfilled": [s for s in SEEDS if plc[s]["fill"] < 15]}
        # ------------------------------------------------------------------ 1. main runs
        log(f"main: {len(SEEDS)} instances x T_ext={T_EXT}, {len(CPS)} checkpoints, workers={a.workers}")
        t_main = time.time()
        run_pool(_instance_job, [(s, plc[s], out_dir) for s in SEEDS], a.workers, "main", 1, 4)
        main_wall = time.time() - t_main
        recs = [json.loads((out_dir / "parts" / f"i{s}.json").read_text()) for s in SEEDS]
        errs = [r for r in recs if r["error"]]
        if errs:
            summary["errors"] = [{"seed": r["seed"], "error": r["error"], "tb": r.get("tb")} for r in errs]
            log(f"ERRORS on {len(errs)} instances: {[r['seed'] for r in errs]}")
        ok = [r for r in recs if not r["error"]]
        rows = [x for r in ok for x in r["rows"]]
        with open(out_dir / "results.jsonl", "w") as fh:
            for x in rows:
                fh.write(json.dumps(x, default=str) + "\n")
        # ------------------------------------------------------------------ 2. amplification (3 problems, instance 800)
        progress(2, 4, "amplification")
        t_amp = time.time()
        try:
            amp = run_pool(_amplification_job, [(SEEDS[0], plc[SEEDS[0]])], 1, "amplification", 2, 4)[0]
        except Exception as e:  # noqa: BLE001
            amp = [{"error": repr(e)}]
        summary["amplification"] = {"instance": SEEDS[0], "T": T_EXT, "problems": amp,
                                    "bnb": "max_cells=8, max_nodes=96, 40 s budget per side",
                                    "sec": time.time() - t_amp}
        # ------------------------------------------------------------------ 3. gate statistics
        progress(3, 4, "stats")
        n_inst = len(ok)
        false_streams = [r["seed"] for r in ok if any(x["false_cert"] for x in r["rows"])]
        anytime_false_streams = [r["seed"] for r in ok if any(x["anytime_false_checkpoints"] > 0 for x in r["rows"])]
        n_q = len(rows)
        n_cert = sum(x["status"] == "CERTIFIED" for x in rows)
        cert_rate = n_cert / n_q if n_q else float("nan")
        n_false = sum(x["false_cert"] for x in rows)
        E = np.array([x["eps_need_curve"] for x in rows])
        gaps = np.array([x["true_gap"] for x in rows])
        spreads = np.array([x["spread"] for x in rows])
        curve = {}
        for i, T in enumerate(CPS):
            e = E[:, i]
            curve[str(T)] = {"eps_need_q10": float(np.quantile(e, .1)), "eps_need_q25": float(np.quantile(e, .25)),
                             "eps_need_q50": float(np.median(e)), "eps_need_q75": float(np.quantile(e, .75)),
                             "cum_cert_rate_eps0.02": float(np.mean([x["N_cert"] is not None and x["N_cert"] <= T
                                                                     for x in rows])),
                             "cert_rate_eps0.05_at_T": float(np.mean(e <= 0.05)),
                             "frac_eps_need_ge_spread": float(np.mean(e >= spreads))}
        ncert = np.array([x["N_cert"] for x in rows if x["N_cert"] is not None], float)
        per_inst_rate = [np.mean([x["status"] == "CERTIFIED" for x in r["rows"]]) for r in ok]
        by_layer = {}
        for g in ("tie", "near", "clear"):
            rr = [x for x in rows if x["gap_layer"] == g]
            if rr:
                by_layer[g] = {"n": len(rr), "cert_rate": float(np.mean([x["status"] == "CERTIFIED" for x in rr])),
                               "false": int(sum(x["false_cert"] for x in rr))}
        by_H = {}
        for H in (6, 8):
            rr = [x for x in rows if x["H"] == H]
            if rr:
                by_H[str(H)] = {"n": len(rr), "cert_rate": float(np.mean([x["status"] == "CERTIFIED" for x in rr]))}
        # runtime projection: eval chunk = 50 instances on 4 workers, FCC + JPC (JPC timing from r4_g_haz MS-H)
        sec = np.array([r["sec"] for r in ok])
        jpc = [json.loads(f.read_text())["sec"] for f in sorted(HAZ_PARTS.glob("main_MS-H_i*_JPC.json"))]
        jpc_med = float(np.median(jpc)) if jpc else 0.0
        waves = math.ceil(EVAL_CHUNK_INSTANCES / EVAL_WORKERS)
        proj_med = waves * (float(np.median(sec)) + jpc_med) / 60
        proj_p90 = waves * (float(np.percentile(sec, 90)) + float(np.percentile(jpc, 90) if jpc else 0)) / 60
        # LPT simulation with the observed per-instance FCC times (resampled to 50) + JPC median
        rs = np.random.default_rng(42)
        sims = []
        for _ in range(2000):
            jobs = np.sort(rs.choice(sec, EVAL_CHUNK_INSTANCES) + rs.choice(jpc if jpc else [0.0],
                                                                             EVAL_CHUNK_INSTANCES))[::-1]
            load = np.zeros(EVAL_WORKERS)
            for t in jobs:
                load[np.argmin(load)] += t
            sims.append(load.max() / 60)
        proj_sim95 = float(np.percentile(sims, 95))
        runtime_ok = proj_p90 <= CHUNK_MIN_LIMIT
        safety_ok = len(false_streams) == 0
        nonvac_ok = cert_rate >= NONVAC_MIN
        gate_pass = bool(safety_ok and nonvac_ok and runtime_ok and not errs)
        summary.update({
            "n_instances": n_inst, "n_problems": n_q, "errors_n": len(errs),
            "fcc_false_streams": len(false_streams), "false_stream_ids": false_streams,
            "fwer_cp_upper_one_sided": cp_upper(len(false_streams), n_inst),
            "anytime_false_streams": len(anytime_false_streams),
            "n_certified": n_cert, "n_false_cert": n_false,
            "cert_rate_eps0.02": cert_rate,
            "cert_rate_per_instance_q": {q: float(np.quantile(per_inst_rate, q / 100)) for q in (10, 25, 50, 75, 90)},
            "instances_with_zero_cert": int(sum(r == 0 for r in per_inst_rate)),
            "eps_need_curve": curve,
            "true_gap_quantiles": {q: float(np.quantile(gaps, q / 100)) for q in (10, 25, 50, 75, 90)},
            "spread_quantiles": {q: float(np.quantile(spreads, q / 100)) for q in (10, 25, 50, 75, 90)},
            "frac_spread_le_eps": float(np.mean(spreads <= EPS)),
            "N_cert_over_6000": ({"n": int(len(ncert)), "min": float(ncert.min() / TMAX_R3),
                                  "q50": float(np.median(ncert) / TMAX_R3), "max": float(ncert.max() / TMAX_R3),
                                  "hist": {str(T): int((ncert == T).sum()) for T in CPS}} if len(ncert) else None),
            "by_gap_layer": by_layer, "by_H": by_H,
            "validity_diagnostics": {
                "pdl_containment_violations": int(sum(r["pdl_containment_violations"] for r in ok)),
                "cs_cover_truth_all_checkpoints": bool(all(c["cs_cover_truth"] for r in ok for c in r["cp_rows"]
                                                           if "cs_cover_truth" in c)),
                "truth_solver_max_abs_diff": float(max(r["truth_solver_max_abs_diff"] for r in ok))},
            "runtime": {"sec_per_instance_median": float(np.median(sec)),
                        "sec_per_instance_p90": float(np.percentile(sec, 90)), "sec_per_instance_max": float(sec.max()),
                        "step_s_median": float(np.median([r["step_s"] for r in ok])),
                        "decide_s_median": float(np.median([r["decide_s"] for r in ok])),
                        "main_wall_min": main_wall / 60,
                        "jpc_msh_sec_median_from_r4_g_haz": jpc_med,
                        "eval_chunk": f"{EVAL_CHUNK_INSTANCES} instances, {EVAL_WORKERS} workers, FCC + JPC",
                        "proj_chunk_min_median": proj_med, "proj_chunk_min_p90": proj_p90,
                        "proj_chunk_min_lpt_sim_p95": proj_sim95, "limit_min": CHUNK_MIN_LIMIT,
                        "note": "并发运行 timings (r4_g_fac_b concurrently on the other CPU slots)"},
            "gate": {"safety_0_false_streams": safety_ok, "nonvacuity_cert_rate_ge_0.20": nonvac_ok,
                     "runtime_chunk_le_55min": runtime_ok, "pass_part_a": gate_pass,
                     "verdict_owner": "r4_gate_decision (combines part a MS-H with part b MS-S/MS-R3/MS-F)"},
            "code_sha256": code_sha()})
        # samples: 8 representative (instance, problem) eps_need curves + per-checkpoint rows of instance 800
        samp = sorted(rows, key=lambda x: (x["status"] != "CERTIFIED", x["true_gap"]))
        pick = samp[:3] + samp[len(samp) // 2 - 1: len(samp) // 2 + 2] + samp[-2:]
        (out_dir / "samples" / "eps_need_curves.json").write_text(json.dumps(pick, indent=1, default=str))
        (out_dir / "samples" / "i800_checkpoint_rows.json").write_text(json.dumps(ok[0]["cp_rows"], indent=1))
        log(f"G-fac' A: false streams {len(false_streams)}/{n_inst}, cert rate {cert_rate:.3f}, "
            f"proj chunk p90 {proj_p90:.1f} min -> pass={gate_pass}")
    except Exception as e:  # noqa: BLE001
        status = "failed"
        summary["fatal"] = repr(e)
        summary["tb"] = traceback.format_exc()[-4000:]
        log(f"FATAL {e!r}")
    wall = (time.time() - t_start) / 60
    summary["wall_min"] = wall
    summary["finished"] = datetime.now().isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    g = summary.get("gate", {})
    rt = summary.get("runtime", {})
    entry = {"task_id": TASK, "candidate_id": "cand_qfc", "status": status,
             "go_no_go": "GO" if g.get("pass_part_a") else "NO_GO",
             "key_metrics": {"fcc_false_streams": summary.get("fcc_false_streams"),
                             "n_instances": summary.get("n_instances"),
                             "cert_rate_eps0.02": summary.get("cert_rate_eps0.02"),
                             "proj_chunk_min_p90": rt.get("proj_chunk_min_p90"),
                             "sec_per_instance_median": rt.get("sec_per_instance_median")},
             "summary_path": f"exp/results/{'pilots' if a.mode == 'pilot' else 'full'}/{TASK}/summary.json"}
    curve = summary.get("eps_need_curve", {})
    ctab = "\n".join(f"| {T} | {v['eps_need_q50']:.4f} | {v['eps_need_q25']:.4f} | {v['cum_cert_rate_eps0.02']:.3f} |"
                     for T, v in curve.items())
    md = f"""## {TASK} (G-fac' A, MS-H 800-839, FCC-PDL 均匀行为, T_ext=200k)
- 状态: {status}; 门 A 判定: {'通过' if g.get('pass_part_a') else '未通过'}（最终裁决在 r4_gate_decision）
- 错误流: {summary.get('fcc_false_streams')}/{summary.get('n_instances')}（FWER 单侧 CP 上界 {summary.get('fwer_cp_upper_one_sided')}）；anytime 口径错误流 {summary.get('anytime_false_streams')}
- ε=0.02 认证率: {summary.get('cert_rate_eps0.02')}（门槛 ≥ 0.20）；认证 {summary.get('n_certified')}/{summary.get('n_problems')}
- 运行时: 单实例中位 {rt.get('sec_per_instance_median')} s，p90 {rt.get('sec_per_instance_p90')} s；评价块投影 p90 {rt.get('proj_chunk_min_p90')} min（≤ 55）。并发运行计时
- N_cert/6000: {summary.get('N_cert_over_6000')}

| T | eps_need 中位 | eps_need q25 | 累计认证率 ε=0.02 |
|---|---|---|---|
{ctab}
"""
    update_shared_summary(entry, md)
    snap = {"model": "FCC-PDL (decoupled interval DP)", "dataset": "MS-H dev 800-839", "T_ext": T_EXT,
            "checkpoints": len(CPS), "problems": 15, "workers": a.workers, "gpu_model": "none (CPU)", "gpu_count": 0}
    update_gpu_progress(status, start_iso, wall, snap)
    mark_done(status, f"{TASK}: pass_part_a={g.get('pass_part_a')} false_streams={summary.get('fcc_false_streams')} "
                      f"cert_rate={summary.get('cert_rate_eps0.02')} proj_p90={rt.get('proj_chunk_min_p90')}")
    log(f"done ({wall:.1f} min) status={status}")


if __name__ == "__main__":
    main()
