"""r3_nl_extra_[a-c]: NL-R0 generic-reuse controls, LR-chi2-grid and B8 x {off, full}. Methodology 2.1 / 5.1 (descriptive).

Design  NL-R0 (E1-NL-S G_1, realisable), gap-quota streams (tie / near / clear x 5), CRN identical to r3_nl_main:
        the per-(instance, stream, method) job of run_r3_nl_main.py is reused unchanged (same platform copy, same
        noise seed, same n0 rows, same problem order, acquisition rng [seed, noise, method_code]); only the method
        set differs. Arms off / full only, so no sibling / orth platform is built.
  LR-chi2-grid  same grid, same certificate and DDA design, Wald-type fixed-n chi2_{d_eff} threshold.
  B8            full identification (width_J < tau = eps/2) on the LR set.
Instances  chunk a = eval 10000-10039, b = 10040-10079, c = 10080-10119. quota_fail replacement reproduces the
           r3_nl_main assignment exactly: each eval seed belongs to an r3_nl_main 15-seed chunk; fill_instances is
           walked over that chunk's prefix with that chunk's reserve start, so a replaced seed maps to the same
           reserve seed as in r3_nl_main (CRN across the two tasks).
Downscale  B8|off on streams [0, b8_off_streams); full mode reads prereg_lock downscale_decisions
           ("r3_nl_extra_[a-c]": preset "B8_off_stream0_only" -> 1 stream, otherwise 3).
Logging    predictors.jsonl (learner, fsync'ed, write-ahead) / results.jsonl (harness, after the stream), as r3_nl_main.

Pilot  smoke + timing only (NO scientific readout): dev seeds 736-739, stream 0, 8 problems, {LR-chi2-grid, B8} x
       {off, full} (128 runs); projects the full wall-clock of one chunk with the lock rule and lists the downscale
       option if > 55 min.
Usage: run_r3_nl_extra.py --chunk a --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores + the RTX 4090 with other round-3 tasks): timings are "concurrent".
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r3_nl_main as M  # noqa: E402

WS, RES_ROOT, N0 = M.WS, M.RES_ROOT, M.N0
METHODS = ("LR-chi2-grid", "B8")
ARMS = ("off", "full")
CHUNKS = {c: (10000 + 40 * i, 10040 + 40 * i) for i, c in enumerate("abc")}
PILOT = {"seeds": [736, 737, 738, 739], "streams": [0], "n_problems": 8}
SAFETY = 1.35                                  # lock timing rule for r3_nl_extra_[a-c]
CODE_FILES = M.CODE_FILES + ["run_r3_nl_extra.py"]


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


def main_chunk_of(seed):
    for c, (lo, hi) in M.CHUNKS.items():
        if lo <= seed < hi:
            return c
    raise ValueError(seed)


def tables_crn(seeds, mode):
    """Instance assignment with r3_nl_main-identical quota_fail replacement (full) / dev reserve (pilot)."""
    if mode == "pilot":
        return M.tables(seeds, M.DEV_RESERVE)
    by_chunk = defaultdict(list)
    for s in seeds:
        by_chunk[main_chunk_of(s)].append(s)
    used, repl, qinfo, tinfo = [], [], {}, {"sec": 0.0, "vram_peak_mb": 0.0, "parts": {}}
    for c, ss in sorted(by_chunk.items()):
        lo, _hi = M.CHUNKS[c]
        prefix = list(range(lo, max(ss) + 1))              # walk the r3_nl_main chunk prefix (same replacement order)
        u, r, qi, ti = M.tables(prefix, M.RESERVE[c])
        mp = dict(zip(prefix, u))
        used += [mp[s] for s in ss]
        repl += [x | {"nl_main_chunk": c} for x in r if x["failed"] in ss]
        qinfo.update(qi)
        tinfo["sec"] += ti["sec"]
        tinfo["vram_peak_mb"] = max(tinfo["vram_peak_mb"], ti["vram_peak_mb"])
        tinfo["parts"][c] = ti
    return used, repl, qinfo, tinfo


def k_curves(R, seeds, n_problems):
    """Descriptive K curve: cumulative cost incl. n0 (paid once per stream) vs problem index; K* = first K from which
    cum_full <= cum_off for every later K (per instance-stream, on the common off/full streams)."""
    cur, kstar = {}, {}
    idx = defaultdict(dict)
    for r in R:
        idx[(r["method"], r["arm"], r["instance"], r["stream"])][r["problem"]] = r["new_env_steps"]
    for m in METHODS:
        for a in ARMS:
            curves = []
            for (mm, aa, i, s), d in idx.items():
                if mm == m and aa == a and len(d) == n_problems:
                    curves.append(N0 + np.cumsum([d[k] for k in range(n_problems)]))
            if curves:
                cur[f"{m}|{a}"] = {"n_streams": len(curves), "mean_cum": np.mean(curves, 0).round(2).tolist()}
        ks = []
        for (mm, aa, i, s), d in idx.items():
            if mm != m or aa != "full" or (m, "off", i, s) not in idx:
                continue
            o = idx[(m, "off", i, s)]
            if len(d) != n_problems or len(o) != n_problems:
                continue
            cf = N0 + np.cumsum([d[k] for k in range(n_problems)])
            co = N0 + np.cumsum([o[k] for k in range(n_problems)])
            ok = cf <= co
            kk = next((K + 1 for K in range(n_problems) if ok[K:].all()), None)
            ks.append(kk)
        kstar[m] = {"n_pairs": len(ks), "K_star": ks,
                    "n_never": sum(k is None for k in ks)}
    return cur, kstar


def analyse(R):
    by = defaultdict(list)
    for r in R:
        by[(r["method"], r["arm"])].append(r)
    cells = {}
    for (m, a), rr in sorted(by.items()):
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        cells[f"{m}|{a}"] = {
            "n": len(rr), "n_nontie": len(nt), "status_counts": dict(Counter(r["status"] for r in rr)),
            "n_certified": len(cert), "n_zero_cost": sum(r["zero_cost"] for r in rr),
            "n_false_cert": sum(r["false_cert"] for r in rr),
            "fcr_descriptive": (sum(r["false_cert"] for r in cert) / len(cert)) if cert else None,
            "nontie_steps_tau3000_total": int(sum(r["cost_tau3000"] for r in nt)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "billing_mismatch": sum(not r["billing_ok"] for r in rr),
            "sec_per_problem_method": float(np.mean([r["wall_clock_s"] for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "sec_per_problem": float(np.mean([r["wall_clock_s"] + r["predictor_sec"] for r in rr]))}
    ratio = {}
    for m in METHODS:
        off, full = cells.get(f"{m}|off"), cells.get(f"{m}|full")
        if off and full:
            ratio[m] = {"off_full_ratio_nontie_tau3000": (off["nontie_steps_tau3000_total"] + 1)
                        / (full["nontie_steps_tau3000_total"] + 1),
                        "note": "pilot, dev seeds, stream 0 - pipeline check only, NOT a readout"}
    return cells, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", default="a", choices=sorted(CHUNKS))
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    task = f"r3_nl_extra_{a.chunk}"
    M.TASK = task                                   # M.progress / M.mark_done / M.tables / shared summary use it
    (RES_ROOT / f"{task}.pid").write_text(str(os.getpid()))
    t_all = time.perf_counter()
    start = datetime.now()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / task
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    try:
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        ds = lock.get("frozen_items", {}).get("downscale_decisions", {}).get("r3_nl_extra_[a-c]", [])
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            ds = lock.get("frozen_items", {}).get("downscale_decisions", {}).get("r3_nl_extra_[a-c]", [])
            lo, hi = CHUNKS[a.chunk]
            seeds, streams, n_problems = list(range(lo, hi)), [0, 1, 2], 15
        else:
            seeds, streams, n_problems = PILOT["seeds"], PILOT["streams"], PILOT["n_problems"]
        b8_off_streams = 1 if any("B8" in str(x) and "stream0" in str(x).replace(" ", "") for x in ds) else 3
        M.progress(0, 1, "tables")
        used, repl, qinfo, tinfo = tables_crn(seeds, a.mode)
        M.log(f"tables {tinfo.get('sec')} replacements {repl}")
        jobs = []
        for s in used:
            for st in streams:
                for m in METHODS:
                    arms = tuple(x for x in ARMS if not (m == "B8" and x == "off" and st >= b8_off_streams))
                    jobs.append((s, st, m, n_problems, arms))
        jobs.sort(key=lambda j: (not (j[2] == "B8" and "off" in j[4]), j[0], j[1]))   # B8|off is slowest
        t_runs = time.perf_counter()
        outs = M.run_pool(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, O, meta, errs, missing, wa = M.merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        au = M.audit(R)
        cells, ratio = analyse(R)
        kc, kstar = k_curves(R, used, n_problems)
        # ---- projection of one full chunk (lock rule): sum(sec/problem x slots x 1.35) / (4 x 60) + allowance
        n_full = 40 * 3 * 15

        def project(b8s):
            pc = []
            for c, v in cells.items():
                sl = 40 * b8s * 15 if c == "B8|off" else n_full
                pc.append({"cell": f"quota|{c}", "sec_per_problem": v["sec_per_problem"], "problem_slots": sl,
                           "safety": SAFETY, "cpu_s": v["sec_per_problem"] * sl * SAFETY,
                           "source": f"this pilot ({n_problems} problems/stream, concurrent)"})
            return pc, sum(c["cpu_s"] for c in pc)
        proj_cells, cpu_s = project(b8_off_streams)
        table_min = 40 * M.p1_table_sec_per_seed() / 60
        allowance = 1.0 + table_min
        proj_min = cpu_s / (4 * 60) + allowance
        proj_full_design = project(3)[1] / 240 + allowance
        proj_b8s0 = project(1)[1] / 240 + allowance
        downscale = None
        if proj_min > 55.0:
            downscale = {"option": "lock preset B8_off_stream0_only", "projected_min": proj_b8s0}
        expected = sum(len(j[4]) * n_problems for j in jobs)
        crashes = len(fatal) + len(missing) + len(errs)
        bill_mm = sum(not r["billing_ok"] for r in R)
        pass_flags = {"zero_crashes": crashes == 0 and len(R) == expected,
                      "billing_zero_mismatch": bill_mm == 0 and au["n_bad"] == 0 and au["replay_billed_total"] == 0,
                      "predictors_before_results_every_certification": bool(
                          wa.get("write_ahead_ok") and wa["n_cert_without_predictor"] == 0
                          and wa["n_cert_predictor_late"] == 0 and wa["n_full_ev1_cert_without_full_predictors"] == 0),
                      # criterion: projected <= 55 min, else a downscale option for the lock must be reported
                      "projected_full_le_55min_or_downscale_reported": proj_min <= 55.0 or (
                          downscale is not None and downscale["projected_min"] <= 55.0),
                      "n_runs_ge_100": len(R) >= 100}
        go = all(pass_flags.values())
        requires_downscale = proj_min > 55.0
        lockproj = lock.get("timing_projection", {}).get("per_task", {}).get("r3_nl_extra_[a-c]", {})
        summary = {
            "task": task, "mode": a.mode, "seeds": used, "streams": streams, "n_problems": n_problems,
            "methods": list(METHODS), "arms": list(ARMS), "b8_off_streams": b8_off_streams, "replacements": repl,
            "quota_info": qinfo, "tables": tinfo, "n_jobs": len(jobs), "n_runs": len(R),
            "n_expected_runs": expected, "n_predictor_lines": len(P),
            "crashes": {"fatal_jobs": fatal, "missing_parts": missing, "n_errors": len(errs), "errors_head": errs[:5]},
            "billing_mismatch": bill_mm, "audit": au, "write_ahead": wa, "cells": cells,
            "off_full_ratio_by_method_descriptive": ratio, "K_curve_descriptive": kc, "K_star_descriptive": kstar,
            "runs_wall_sec": runs_sec, "total_wall_sec": time.perf_counter() - t_all,
            "timing_projection": {"rule": "sum(sec/problem x slots x 1.35) / (4 workers x 60) + allowance "
                                          "(1 min + GPU pool tables of 40 fresh instances at the r3_p1 rate)",
                                  "cells": proj_cells, "cpu_s": cpu_s, "allowance_min": allowance,
                                  "projected_min_full_chunk": proj_min,
                                  "projected_min_full_design_b8off_3streams": proj_full_design,
                                  "projected_min_b8off_stream0_only": proj_b8s0,
                                  "lock_projection_min": lockproj.get("projected_min_chosen"),
                                  "lock_downscale_applied": lockproj.get("downscale_applied"),
                                  "downscale_option": downscale,
                                  "caveat": f"pilot = {n_problems} problems/stream (full 15: full-arm ledgers grow "
                                            "with k); table allowance is an upper bound (pools mostly cached by "
                                            "r3_nl_main); 并发运行"},
            "pass_criteria": pass_flags, "go_no_go": "GO" if go else "NO_GO",
            "requires_downscale_for_full": requires_downscale,
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; ratios are descriptive)",
            "lock_status_at_run": lock.get("status"), "lock_downscale_decisions": ds, "code_sha256": code_sha(),
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        M.log(f"summary: go={go} flags={pass_flags} proj={proj_min:.1f} min runs={len(R)}")
        if a.mode == "pilot":
            entry = {"candidate_id": "shared", "go_no_go": "GO" if go else "NO_GO", "pass_criteria": pass_flags,
                     "projected_full_min": proj_min, "projected_full_min_b8off_stream0_only": proj_b8s0,
                     "n_runs": len(R), "downscale_option": downscale, "requires_downscale_for_full": requires_downscale,
                     "scientific_readout": "none (smoke + timing)"}
            md = (f"## {task} (pilot, dev 736-739, stream 0, {n_problems} problems, LR-chi2-grid/B8 x off/full)\n"
                  f"- 结论: **{'GO' if go else 'NO_GO'}**；runs={len(R)}/{expected}，崩溃={crashes}，"
                  f"计费不一致={bill_mm}，审计异常={au['n_bad']}\n"
                  f"- write-ahead: 认证 {wa.get('n_certified')} 次，缺预测记录 {wa.get('n_cert_without_predictor')}，"
                  f"晚于评分 {wa.get('n_cert_predictor_late')}\n"
                  f"- 单分块 full 预计 {proj_min:.1f} min（B8 off {b8_off_streams} 流；仅流 0 时 {proj_b8s0:.1f} min；"
                  f"lock 预估 {lockproj.get('projected_min_chosen')}）；并发运行，计时偏高\n"
                  + (f"- 需降规模: full 必须采用 lock preset B8_off_stream0_only（task_plan 描述已写明；lock "
                     f"downscale_decisions 当前为空，需在 full 前写入）\n" if requires_downscale else ""))
            M.update_shared_summary(entry, md)
        M.mark_done("success" if go else "partial", f"{task} {a.mode}: GO={go} runs={len(R)} proj={proj_min:.1f}min")
        return summary
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        M.log(tb)
        (out_dir / "fatal.txt").write_text(tb)
        M.mark_done("failed", f"{task} {a.mode} crashed: {e!r}")
        raise


if __name__ == "__main__":
    main()
