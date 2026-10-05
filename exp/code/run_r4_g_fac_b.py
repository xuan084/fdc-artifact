"""r4_g_fac_b: Gate G-fac' part B -- FCC(PDL) safety + non-vacuity + runtime at T_ext = 200k.

Methodology "FCC 锁前修订（r4，2026-10-02）" F2 / F4 / F5 (overrides §1.6, §3 G-fac row):
  FCC = decide_pdl (pair-difference certificate, PDL), UNIFORM random behaviour, NO BnB in the rule.
  MS-S dev instances 860-889 (30, one stream each, noise seed 42, 15 problems), eps = 0.02, delta = 0.05,
  ORIGINAL horizons H in {6, 8} (asserted by MSInstance), T_ext = 200000 new rounds after the n0 = 20 initial rounds,
  8 fixed checkpoints {1k, 3k, 6k, 20k, 60k, 100k, 150k, 200k}. Certification only at checkpoints; a problem is
  decided at its FIRST checkpoint with eps_need <= eps (policy k_hat at that checkpoint, never revised).
  Smoke (record only, not in the gate): MS-R3 dev 890-894, MS-F dev 895-899, same configuration.
  Gate G-fac' (part B share; verdict in r4_gate_decision): MS-S false streams 0/30 AND MS-S cert rate at eps=0.02
  >= 0.20 AND projected evaluation chunk (50 instances FCC + JPC, 4 workers) <= 55 min.
  Recorded: eps_need curve, true top-2 gap and candidate spread, N_cert / 6000, PDL bound validity vs truth, CS
  coverage of the true cell probabilities, decoupling amplification (3 problems of MS-S 860 at T_ext, shared-parameter
  range by random + vertex local search; BnB depth 8 approximation).
  ONLY FCC is run (no baselines). Nothing (eps, T_ext, checkpoints) is tuned on these results.
  Behaviour rng = default_rng([seed, 42, 161]) (161 = METHOD_CODE['FCC']).
  Write-ahead: per-checkpoint decisions (k_hat, eps_need, D) are logged to decisions/ BEFORE truth is read; truth
  scoring happens after the stream.

Usage: run_r4_g_fac_b.py [--workers 4] [--mode pilot]
Concurrent run: <= 4 CPU workers, BLAS threads = 1; GPU (shared 4090) only for MS-R3 placements.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES = WS / "exp" / "results"
TASK = "r4_g_fac_b"
PLANNED_MIN = 40
EPS, DELTA = 0.02, 0.05
T_EXT = 200_000
CPS = [1_000, 3_000, 6_000, 20_000, 60_000, 100_000, 150_000, 200_000]
LAYER_SEEDS = {"MS-S": list(range(860, 890)), "MS-R3": list(range(890, 895)), "MS-F": list(range(895, 900))}
GATE_LAYER = "MS-S"
AMP_SEED, AMP_PROBLEMS = 860, 3
BEHAVIOUR_CODE = 161
EVAL_CHUNK_INSTANCES, EVAL_WORKERS, CHUNK_LIMIT_MIN = 50, 4, 55
GHAZ = RES / "pilots" / "r4_g_haz"
CODE_FILES = ["dsswm/streams/ms_r4.py", "dsswm/certify/fcc.py", "run_r4_g_fac_b.py"]


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


# ============================================================================================ amplification
def amplification(sol, q, lo, hi, rng, budget_s=90.0):
    """Decoupled width vs. a shared-parameter range estimate (random box points + vertex local search, as in
    r4_setup_fcc) and BnB depth-8 tightening, for policy 0 of problem q."""
    from dsswm.certify.fcc import value_bounds_bnb
    u = q.utility
    w, wr, cq = np.asarray(u.w, float), float(u.w_ret), float(u.c_q)
    plan = sol.plan(q.policies[0], q.loads0, q.engaged0, q.H)
    jl, jh = sol.solve(plan, w, wr, cq, lo, hi)
    used = sorted(sol.used_cells(plan))
    t0 = time.perf_counter()
    pts = []
    for _ in range(300):
        th = lo + (hi - lo) * rng.random(len(lo))
        if rng.random() < 0.5:
            th = np.where(rng.random(len(lo)) < 0.5, lo, hi)
        pts.append(sol.solve(plan, w, wr, cq, th, th)[0])
    for sense in (1, -1):
        for _r in range(6):
            th = np.where(rng.random(len(lo)) < 0.5, lo, hi)
            cur = sol.solve(plan, w, wr, cq, th, th)[0]
            improved = True
            while improved and time.perf_counter() - t0 < budget_s:
                improved = False
                for c in used:
                    th2 = th.copy()
                    th2[c] = hi[c] if th[c] == lo[c] else lo[c]
                    v = sol.solve(plan, w, wr, cq, th2, th2)[0]
                    if sense * v > sense * cur + 1e-15:
                        th, cur, improved = th2, v, True
            pts.append(cur)
    t1 = time.perf_counter()
    bh = value_bounds_bnb(sol, plan, u, lo, hi, max_cells=8, max_nodes=256, time_budget_s=60, sense="hi")
    bl = value_bounds_bnb(sol, plan, u, lo, hi, max_cells=8, max_nodes=256, time_budget_s=60, sense="lo")
    rng_w = max(max(pts) - min(pts), 1e-12)
    return {"pid": q.pid, "H": int(q.H), "n_used_cells": len(used), "dec_lo": float(jl), "dec_hi": float(jh),
            "shared_est_min": float(min(pts)), "shared_est_max": float(max(pts)),
            "amplification_width": float((jh - jl) / rng_w),
            "bnb8_lo": float(bl["bnb"]), "bnb8_hi": float(bh["bnb"]),
            "amplification_width_bnb8": float((bh["bnb"] - bl["bnb"]) / rng_w),
            "bnb_nodes": int(bh["nodes"] + bl["nodes"]), "search_s": t1 - t0, "bnb_s": time.perf_counter() - t1}


# ============================================================================================ worker
def _job(layer, seed, plc, out_dir, do_amp):
    import torch
    torch.set_num_threads(1)
    from dsswm.certify.fcc import FCCCertifier, cell_probs, decide_pdl
    from dsswm.streams import ms_r4 as M
    fn = out_dir / "parts" / f"{layer}_i{seed}.json"
    if fn.exists():
        return {"layer": layer, "seed": seed, "skipped": True}
    t_all = time.perf_counter()
    rec = {"layer": layer, "seed": seed, "error": None}
    try:
        t0 = time.perf_counter()
        inst = {"MS-S": lambda: M.build_mss(seed), "MS-R3": lambda: M.build_msr3(seed, plc),
                "MS-F": lambda: M.build_msf(seed)}[layer]()
        rec["build_sec"] = time.perf_counter() - t0
        env, init = inst.make_platform()
        P = env.aspace
        cert = FCCCertifier(P.L, P.R, 2, P, rho_ret=0.3, delta=DELTA)
        sol = cert.solver
        for o in init:
            cert.observe(o)
        h = env.handle()
        rng = np.random.default_rng([int(seed), M.NOISE, BEHAVIOUR_CODE])
        jps = [{} for _ in inst.problems]
        dec_path = out_dir / "decisions" / f"{layer}_i{seed}.jsonl"
        dec_path.unlink(missing_ok=True)
        first = [None] * len(inst.problems)          # (checkpoint, k_hat, eps_need) at first certification
        ck_log = []
        done, t_steps, t_checks = 0, 0.0, []
        for T in CPS:
            ts = time.perf_counter()
            for _ in range(T - done):
                cert.observe(h.step(int(rng.integers(P.n))))
            t_steps += time.perf_counter() - ts
            done = T
            lo, hi = cert.cs.intervals()
            tc = time.perf_counter()
            dec = []
            for qi, q in enumerate(inst.problems):
                st, k, e, D = decide_pdl(sol, q, lo, hi, EPS, jps=jps[qi])
                dec.append({"q": qi, "status": st, "k_hat": int(k), "eps_need": float(e), "D": D.tolist()})
                if first[qi] is None and st == "CERTIFIED":
                    first[qi] = (T, int(k), float(e))
            t_checks.append(time.perf_counter() - tc)
            ck = {"T": T, "decisions": dec, "logged_at": time.time(), "cs_lo": None}
            with open(dec_path, "a") as fh:                    # write-ahead, before any truth is read
                fh.write(json.dumps({"layer": layer, "instance": seed, "T": T, "logged_at": ck["logged_at"],
                                     "decisions": [{k_: v for k_, v in d.items() if k_ != "D"} for d in dec],
                                     "first_cert": first}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            ck["cs"] = (lo.copy(), hi.copy()) if (do_amp and T == CPS[-1]) else None
            ck_log.append(ck)
        stream_sec = time.perf_counter() - t_all
        # -------------------------------------------------------------- truth scoring (after the stream)
        scored_at = time.time()
        try:
            theta = cell_probs(env.true_params(), cert.cells, c=1.0)
        except Exception:  # noqa: BLE001  (MS-F platform: contract may not map to cells)
            theta = None
        rows, curves = [], []
        for ck in ck_log:
            assert ck["logged_at"] <= scored_at
            lo, hi = ck["cs"] if ck["cs"] is not None else (None, None)
            for d in ck["decisions"]:
                qi = d["q"]
                Jt = np.asarray(inst.J_true[qi], float)
                D = np.asarray(d["D"], float)
                n = len(Jt)
                viol = 0
                for j in range(n):
                    for k in range(n):
                        if j != k and D[j, k] < Jt[j] - Jt[k] - 1e-9:
                            viol += 1
                curves.append({"T": ck["T"], "q": qi, "eps_need": d["eps_need"], "k_hat": d["k_hat"],
                               "regret_k_hat": float(Jt.max() - Jt[d["k_hat"]]), "pdl_violations": viol})
        cs_cover = None
        if theta is not None:
            lo, hi = cert.cs.intervals()
            cs_cover = bool(((theta >= lo - 1e-12) & (theta <= hi + 1e-12)).all())
        for qi, q in enumerate(inst.problems):
            Jt = np.asarray(inst.J_true[qi], float)
            s = np.sort(Jt)
            gap = float(s[-1] - s[-2]) if len(s) > 1 else float("inf")
            spread = float(s[-1] - s[0])
            f = first[qi]
            cert_ok = f is not None
            regret = float(Jt.max() - Jt[f[1]]) if cert_ok else None
            cv = {c["T"]: c["eps_need"] for c in curves if c["q"] == qi}
            rows.append({"instance": seed, "layer": layer, "method": "FCC-PDL", "problem": qi, "pid": q.pid,
                         "H": int(q.H), "n_policies": len(q.policies), "status": "CERTIFIED" if cert_ok else "NEED_DATA",
                         "certified_policy": f[1] if cert_ok else None, "N_cert": f[0] if cert_ok else None,
                         "N_cert_over_6000": (f[0] / 6000) if cert_ok else None,
                         "eps_need_at_cert": f[2] if cert_ok else None, "true_regret": regret,
                         "false_cert": bool(cert_ok and regret > EPS), "correct_cert": bool(cert_ok and regret <= EPS),
                         "true_gap": gap, "spread": spread, "gap_layer": M.gap_layer(gap),
                         "eps_need_curve": {str(t): cv[t] for t in CPS},
                         "eps_need_ge_spread_at_Text": bool(cv[CPS[-1]] >= spread),
                         "pdl_violations": int(sum(c["pdl_violations"] for c in curves if c["q"] == qi)),
                         "eps": EPS, "delta": DELTA, "T_ext": T_EXT, **inst.meta[qi]})
        rec.update({"rows": rows, "curves": curves, "cs_cover_truth_at_Text": cs_cover,
                    "stream_sec": stream_sec, "steps_sec": t_steps, "check_sec": t_checks,
                    "n_problems": len(inst.problems), "env_steps": int(env.n_steps), "env_resets": int(env.n_resets),
                    "info": {k: v for k, v in inst.info.items() if k != "calibration"}})
        if do_amp:
            lo, hi = ck_log[-1]["cs"]
            arng = np.random.default_rng([seed, 8803])
            rec["amplification"] = [amplification(sol, q, lo, hi, arng) for q in inst.problems[:AMP_PROBLEMS]]
    except Exception as e:  # noqa: BLE001
        rec.update({"error": repr(e), "tb": traceback.format_exc()[-3000:], "rows": []})
    rec["sec"] = time.perf_counter() - t_all
    tmp = fn.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, fn)
    return {"layer": layer, "seed": seed, "skipped": False, "sec": rec["sec"], "error": rec["error"]}


# ============================================================================================ aggregation
def layer_block(recs):
    rows = [r for rec in recs for r in rec["rows"]]
    if not rows:
        return {"n_streams": len(recs), "n_errors": sum(r["error"] is not None for r in recs)}
    nf = [sum(x["false_cert"] for x in rec["rows"]) for rec in recs if rec["rows"]]
    ncert = sum(x["status"] == "CERTIFIED" for x in rows)
    out = {"n_streams": len(nf), "n_errors": int(sum(r["error"] is not None for r in recs)), "n_problems": len(rows),
           "n_cert": int(ncert), "n_false": int(sum(nf)), "false_streams": int(sum(x > 0 for x in nf)),
           "cert_rate_eps0.02": ncert / len(rows),
           "completion_mean": float(np.mean([sum(x["correct_cert"] for x in rec["rows"]) / len(rec["rows"])
                                             for rec in recs if rec["rows"]])),
           "pdl_violations_total": int(sum(x["pdl_violations"] for x in rows)),
           "cs_cover_truth_all": [rec.get("cs_cover_truth_at_Text") for rec in recs].count(False) == 0,
           "cs_cover_truth_counts": {str(k): [rec.get("cs_cover_truth_at_Text") for rec in recs].count(k)
                                     for k in (True, False, None)}}
    from dsswm.stats.cp import cp_upper_one_sided as _cpu
    try:
        out["false_stream_cp_upper_95"] = float(_cpu(out["false_streams"], out["n_streams"], 0.05))
    except Exception:  # noqa: BLE001
        out["false_stream_cp_upper_95"] = None
    curve = {}
    for T in CPS:
        e = np.array([x["eps_need_curve"][str(T)] for x in rows])
        sp = np.array([x["spread"] for x in rows])
        curve[str(T)] = {"eps_need_q25": float(np.quantile(e, .25)), "eps_need_q50": float(np.median(e)),
                         "eps_need_q75": float(np.quantile(e, .75)),
                         "cert_rate_eps0.02_cum": float(np.mean([x["N_cert"] is not None and x["N_cert"] <= T
                                                                 for x in rows])),
                         "frac_eps_need_ge_spread": float(np.mean(e >= sp))}
    out["eps_need_curve"] = curve
    g = np.array([x["true_gap"] for x in rows])
    sp = np.array([x["spread"] for x in rows])
    out["true_gap_quantiles"] = {q: float(np.quantile(g, q / 100)) for q in (10, 25, 50, 75, 90)}
    out["spread_quantiles"] = {q: float(np.quantile(sp, q / 100)) for q in (10, 25, 50, 75, 90)}
    out["frac_spread_gt_eps"] = float(np.mean(sp > EPS))
    nc = [x["N_cert_over_6000"] for x in rows if x["N_cert"] is not None]
    out["N_cert_over_6000"] = {"n": len(nc), "median": float(np.median(nc)) if nc else None,
                               "min": float(min(nc)) if nc else None, "max": float(max(nc)) if nc else None,
                               "hist_by_checkpoint": {str(T): int(sum(x["N_cert"] == T for x in rows)) for T in CPS}}
    by_gap = defaultdict(lambda: {"n": 0, "cert": 0, "false": 0})
    for x in rows:
        b = by_gap[x["gap_layer"]]
        b["n"] += 1
        b["cert"] += x["status"] == "CERTIFIED"
        b["false"] += x["false_cert"]
    out["by_gap_layer"] = dict(by_gap)
    out["by_H"] = {str(H): {"n": int(sum(x["H"] == H for x in rows)),
                            "cert": int(sum(x["H"] == H and x["status"] == "CERTIFIED" for x in rows))}
                   for H in (6, 8)}
    secs = [rec["stream_sec"] for rec in recs if rec.get("stream_sec")]
    out["sec_per_instance"] = {"mean": float(np.mean(secs)), "max": float(np.max(secs)), "n": len(secs),
                               "steps_sec_mean": float(np.mean([rec["steps_sec"] for rec in recs if rec.get("rows")])),
                               "check_sec_mean": float(np.mean([sum(rec["check_sec"]) for rec in recs
                                                                if rec.get("rows")])),
                               "build_sec_mean": float(np.mean([rec.get("build_sec", 0) for rec in recs])),
                               "note": "并发运行 (4 workers)"}
    return out


def jpc_sec_mss():
    """JPC per-instance wall clock on MS-S from r4_g_haz (the eval chunk runs FCC + JPC on the same instances)."""
    secs = []
    for fn in sorted((GHAZ / "parts").glob("*MS-S_i*_JPC.json")):
        try:
            r = json.loads(fn.read_text())
            if r.get("info"):
                secs.append(float(r["info"]["sec"]))
        except (ValueError, KeyError, TypeError):
            pass
    return {"n": len(secs), "mean": float(np.mean(secs)) if secs else None,
            "max": float(np.max(secs)) if secs else None, "source": "r4_g_haz parts (MS-S, T_max = 6000)"}


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    for sub in ("parts", "decisions", "placements", "samples"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "seed": 42, "started": start_iso, "candidate_id": "cand_qfc",
               "hypotheses": ["G-fac'"], "dev_seeds": {k: [v[0], v[-1]] for k, v in LAYER_SEEDS.items()},
               "constants": {"eps": EPS, "delta": DELTA, "T_ext": T_EXT, "checkpoints": CPS, "problems_per_stream": 15,
                             "noise_seed": 42, "streams_per_instance": 1, "behaviour": "uniform random",
                             "behaviour_rng": "[seed, 42, 161]", "rule": "decide_pdl (no BnB)", "H": [6, 8],
                             "rounds_count": "new rounds after the n0 = 20 initial rounds"},
               "timing_note": "并发运行 (<= 4 CPU workers, BLAS threads = 1; GPU only for MS-R3 placements)",
               "methodology_ref": "plan/methodology.md FCC 锁前修订（r4，2026-10-02）F2/F4/F5"}
    status = "success"
    try:
        from dsswm.streams import ms_r4 as M
        progress(0, 4, "placements_quota")
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        edges_mu = lock["frozen_items"]["r1adv_bin_edges"]["mu_flip"]
        pdir = out_dir / "placements"
        msr3 = {}
        P = None
        for s in LAYER_SEEDS["MS-R3"]:
            fn = pdir / f"msr3_i{s}.json"
            if fn.exists():
                msr3[s] = json.loads(fn.read_text())
            else:
                if P is None:
                    P = M.MSPlacer(edges_mu)
                t0 = time.perf_counter()
                msr3[s] = P.place_msr3(s)
                fn.write_text(json.dumps(msr3[s]))
                P.drop_seed(s)
                log(f"MS-R3 i{s}: placed ({time.perf_counter() - t0:.1f}s) underfilled={msr3[s].get('underfilled')}")
        del P
        t0 = time.perf_counter()
        q1 = M.prepare_quota(LAYER_SEEDS["MS-S"])
        q2 = M.prepare_quota(LAYER_SEEDS["MS-F"], need_static=False)
        summary["quota"] = {"n_quota_fail": int(sum(v["quota_fail"] for v in {**q1, **q2}.values())),
                            "sec": time.perf_counter() - t0}
        summary["msr3_placement"] = {s: {k: msr3[s].get(k) for k in ("draw", "fill_per_bin", "underfilled")}
                                     for s in msr3}
        log(f"quota ready: {summary['quota']}")

        progress(1, 4, "fcc_streams")
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = [("MS-S", s, None, s == AMP_SEED) for s in LAYER_SEEDS["MS-S"]]
        jobs += [("MS-R3", s, msr3[s], False) for s in LAYER_SEEDS["MS-R3"]]
        jobs += [("MS-F", s, None, False) for s in LAYER_SEEDS["MS-F"]]
        jobs.sort(key=lambda j: (not j[3], j[0] != "MS-S"))       # amplification instance first
        done = 0
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_job, j[0], j[1], j[2], out_dir, j[3]): j for j in jobs}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    r = f.result()
                    msg = "cached" if r.get("skipped") else f"{r['sec']:.0f}s err={r.get('error')}"
                except Exception as e:  # noqa: BLE001
                    msg = f"FATAL {e!r}"
                done += 1
                progress(1, 4, "fcc_streams", {"done": done, "total": len(jobs), "last": f"{j[0]} i{j[1]}"})
                log(f"[fcc] {j[0]} i{j[1]} {msg} ({done}/{len(jobs)})")

        progress(2, 4, "aggregate")
        recs = defaultdict(list)
        for fn in sorted((out_dir / "parts").glob("*.json")):
            r = json.loads(fn.read_text())
            recs[r["layer"]].append(r)
        with open(out_dir / "results.jsonl", "w") as fh:
            for layer in LAYER_SEEDS:
                for r in sorted(recs[layer], key=lambda x: x["seed"]):
                    for row in r["rows"]:
                        fh.write(json.dumps(row, default=str) + "\n")
        per_layer = {layer: layer_block(recs[layer]) for layer in LAYER_SEEDS}
        summary["per_layer"] = per_layer
        errs = {layer: [(r["seed"], r["error"]) for r in recs[layer] if r["error"]] for layer in LAYER_SEEDS}
        summary["errors"] = errs
        amp = [x for r in recs["MS-S"] for x in (r.get("amplification") or [])]
        summary["amplification"] = {
            "instance": AMP_SEED, "T": T_EXT, "problems": amp,
            "width_ratio_median": float(np.median([x["amplification_width"] for x in amp])) if amp else None,
            "width_ratio_bnb8_median": float(np.median([x["amplification_width_bnb8"] for x in amp])) if amp else None,
            "note": "shared-parameter range is an inner estimate (random + vertex search), so ratios are upper "
                    "estimates of the true decoupling amplification"}

        # ---------------------------------------------------------------- gate (part B share)
        ms = per_layer[GATE_LAYER]
        jpc = jpc_sec_mss()
        fcc_sec = ms["sec_per_instance"]["mean"]
        fcc_max = ms["sec_per_instance"]["max"]
        jsec = jpc["mean"] or 0.0
        proj = (fcc_sec + jsec) * EVAL_CHUNK_INSTANCES / EVAL_WORKERS / 60
        proj_max = (fcc_max + (jpc["max"] or 0.0)) * EVAL_CHUNK_INSTANCES / EVAL_WORKERS / 60
        c_safe = ms["false_streams"] == 0 and ms["n_errors"] == 0 and ms["n_streams"] == len(LAYER_SEEDS["MS-S"])
        c_nonvac = ms["cert_rate_eps0.02"] >= 0.20
        c_time = proj <= CHUNK_LIMIT_MIN
        summary["runtime_projection"] = {
            "fcc_sec_per_instance_mean": fcc_sec, "fcc_sec_per_instance_max": fcc_max, "jpc": jpc,
            "chunk": f"{EVAL_CHUNK_INSTANCES} instances x (FCC + JPC), {EVAL_WORKERS} workers",
            "projected_chunk_min_mean": proj, "projected_chunk_min_worst": proj_max, "limit_min": CHUNK_LIMIT_MIN,
            "note": "并发运行 timings; the projection assumes the same 4-worker concurrency"}
        summary["gate_G_fac_prime_part_b"] = {
            "layer": GATE_LAYER, "false_streams_zero": c_safe, "cert_rate_ge_0.20": c_nonvac,
            "projected_chunk_le_55min": c_time, "pass": bool(c_safe and c_nonvac and c_time),
            "pass_criteria": "MS-S dev false streams 0/30 AND cert rate at eps=0.02 >= 0.20 AND projected eval chunk "
                             "<= 55 min (final verdict with part A in r4_gate_decision)",
            "smoke_layers_recorded_only": ["MS-R3", "MS-F"]}
        summary["pass"] = summary["gate_G_fac_prime_part_b"]["pass"]

        # ---------------------------------------------------------------- samples
        sdir = out_dir / "samples"
        allrows = [x for layer in LAYER_SEEDS for r in recs[layer] for x in r["rows"]]
        certs = [x for x in allrows if x["status"] == "CERTIFIED"]
        rng = np.random.default_rng(42)
        pick = [certs[i] for i in rng.choice(len(certs), size=min(8, len(certs)), replace=False)] if certs else []
        unc = [x for x in allrows if x["status"] != "CERTIFIED"]
        pick += [unc[i] for i in rng.choice(len(unc), size=min(4, len(unc)), replace=False)] if unc else []
        (sdir / "sample_problems.json").write_text(json.dumps(pick, indent=1, default=str))
        if any(r.get("curves") for r in recs["MS-S"]):
            r0 = sorted(recs["MS-S"], key=lambda x: x["seed"])[0]
            (sdir / f"curves_MS-S_i{r0['seed']}.json").write_text(json.dumps(r0["curves"], indent=1))
        summary["code_sha256"] = code_sha()
    except Exception as e:  # noqa: BLE001
        status = "failed"
        summary["fatal"] = repr(e)
        summary["traceback"] = traceback.format_exc()[-4000:]
        log(summary["traceback"])
    wall = (time.time() - t_start) / 60
    summary["wall_min"] = wall
    summary["end"] = datetime.now().isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    progress(4, 4, "done", {"pass": summary.get("pass")})
    pl = summary.get("per_layer", {})
    ms = pl.get(GATE_LAYER, {})
    g = summary.get("gate_G_fac_prime_part_b", {})
    rp = summary.get("runtime_projection", {})
    entry = {"candidate_id": "cand_qfc", "go_no_go": "GO" if summary.get("pass") else "NO_GO",
             "gate": "G-fac' part B", "pass": summary.get("pass"),
             "key_metrics": {"MS-S_false_streams": ms.get("false_streams"), "MS-S_n_streams": ms.get("n_streams"),
                             "MS-S_cert_rate_eps0.02": ms.get("cert_rate_eps0.02"),
                             "projected_chunk_min": rp.get("projected_chunk_min_mean"),
                             "MS-R3_smoke": {k: pl.get("MS-R3", {}).get(k) for k in ("false_streams", "cert_rate_eps0.02")},
                             "MS-F_smoke": {k: pl.get("MS-F", {}).get(k) for k in ("false_streams", "cert_rate_eps0.02")},
                             "amplification_median": summary.get("amplification", {}).get("width_ratio_median")},
             "status": status, "path": f"exp/results/pilots/{TASK}/summary.json"}
    md = (f"## {TASK} (G-fac' part B, FCC-PDL @ T_ext=200k, MS-S 860-889 + smoke)\n"
          f"- 状态: {status}; 门(part B): {'PASS' if summary.get('pass') else 'FAIL'} "
          f"(错误流 0/30: {g.get('false_streams_zero')}, 认证率≥0.20: {g.get('cert_rate_ge_0.20')}, "
          f"块≤55min: {g.get('projected_chunk_le_55min')})\n"
          f"- MS-S: 错误流 {ms.get('false_streams')}/{ms.get('n_streams')}, ε=0.02 认证率 {ms.get('cert_rate_eps0.02')}, "
          f"投影评价块 {rp.get('projected_chunk_min_mean')} min（并发运行）\n"
          f"- 冒烟 MS-R3 / MS-F: {entry['key_metrics']['MS-R3_smoke']} / {entry['key_metrics']['MS-F_smoke']}\n")
    update_shared_summary(entry, md)
    mark_done(status, f"G-fac' part B pass={summary.get('pass')}; MS-S false_streams={ms.get('false_streams')}, "
                      f"cert_rate={ms.get('cert_rate_eps0.02')}, proj_chunk_min={rp.get('projected_chunk_min_mean')}")
    update_gpu_progress(status, start_iso, wall, {
        "model": "FCC-PDL (decoupled interval DP, uniform behaviour)", "instances": 40, "problems_per_instance": 15,
        "T_ext": T_EXT, "checkpoints": len(CPS), "workers": a.workers, "gpu_model": "RTX 4090 (placements only)",
        "gpu_count": 0, "cpu_only": True})
    log(f"done: status={status} pass={summary.get('pass')} wall={wall:.1f} min")


if __name__ == "__main__":
    main()
