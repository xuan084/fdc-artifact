"""r5_cr_fwer_audit: confirmatory C2 full-horizon FWER audit + N100 (FDC; B2-fav-tight descriptive), CR9 eval half.

Usage:
  python run_r5_cr_fwer_audit.py --mode full    # eval seeds 30000-30199 from lock v5 eval_tasks[r5_cr_fwer_audit]
  python run_r5_cr_fwer_audit.py --mode pilot   # smoke/timing on DEV seeds 900-949 only (never eval seeds)

Protocol (lock v5, plan/prereg_lock.json):
* start-up: dsswm.stats.prereg.lock_gate(task_id, version=5) (== assert_locked incl. frozen-code drift check).
  On refusal: summary.json with status='skipped_by_lock', exit 0, no stream is run.
* setting: CR9, eps* = 0.001 (r4_gates/eps.json, cross-checked with the lock), K = 20, Q = 15, delta = 0.05.
* full horizon: NO stop at 12/15. The ctx passed to run_stream is dataclasses.replace(ctx, stop_k=Q) -- the same
  construction as the dev gate r5_t2_gate_plugin_t3 (T3) whose unit prices are in the lock: the stream runs until
  15/15 are certified (after which nothing can change: answers are sticky) or tau_R. No method reads ctx.stop_k
  (only the harness does), so method behaviour is identical to the stop-at-12/15 blocks; the pilot verifies this
  by comparing the 12/15 prefix with the r5_cr_main_a / r5_cr_plugin_a dev pilot rows.
* per (method, stream) row: every run_stream summary field (prefix rs_), the correctness and true gap of every one of
  the 15 certificates, per-problem (per-budget) certification checkpoint/time, N100 (pen/raw), the 12/15 prefix
  endpoints (N80_pen, N80_raw, completed, fwer_event_by_12of15) and the full-horizon fwer_event, code hashes.
  Write-ahead results.jsonl (fsync per row), resumable.
* NO analysis inside the block (no CP bound, no ratios): only completeness / billing / hash checks.
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run: timings biased up).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import dataclasses
import hashlib
import json
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
TASK = "r5_cr_fwer_audit"
N_WORKERS = 4
DEV_PILOT_SEEDS = list(range(900, 950))  # 50 dev streams x 2 methods = 100 stream-method runs
# lock label -> registry name
LABELS = {"FDC (full horizon)": "FDC", "B2-fav-tight (full horizon, descriptive)": "B2-fav-tight"}

import run_r5_cr_main as crm  # noqa: E402  (helpers only: progress / mark_done / skipped_summary / git_info)
from dsswm.stats import prereg  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/combgame_joint.py",
         "dsswm/baselines/plugin_r5.py", "dsswm/evidence/ext_plugin.py", "dsswm/streams/r5_registry.py",
         "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py", "dsswm/certify/quadknap.py",
         "dsswm/envs/pool_replay.py", "dsswm/stats/prereg.py", "run_r5_cr_main.py", "run_r5_cr_fwer_audit.py"]
_ENV = {}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hashes():
    per = {p: sha_file(CODE / p) for p in FILES if (CODE / p).exists()}
    return per, hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()


def init_env(half, eps):
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", half)
    c20 = build_ctx(env, fr.cr_problems("visit"), eps)
    cfull = dataclasses.replace(c20, stop_k=c20.Q)  # full horizon: stop only at 15/15 (else tau_R)
    J = true_policy_values(c20.pols, env.w, env.true_mu("visit"))
    _ENV.update(env=env, c20=c20, ctx=cfull, J=J, Js=np.array([J[c20.feas[q]].max() for q in range(c20.Q)]), eps=eps)


def job(a):
    name, kw, seed, code_sha = a
    ctx, c20, J, Js = _ENV["ctx"], _ENV["c20"], _ENV["J"], _ENV["Js"]
    try:
        m = reg.make_method(name, **kw)
        t0 = time.perf_counter()
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=False)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))
        certs = []
        for q in range(ctx.Q):
            k = s["cert_k"][q]
            if k < 0:
                certs.append({"q": q, "budget": float(ctx.problems[q].budget), "cert_k": -1, "t": None,
                              "decided_pi": -1, "gap": None, "false": None})
                continue
            pi = s["decided_pi"][q]
            gap = float(Js[q] - J[pi])
            certs.append({"q": q, "budget": float(ctx.problems[q].budget), "cert_k": int(k), "t": int(ck[k]),
                          "decided_pi": int(pi), "gap": gap, "false": bool(gap > ctx.eps + 1e-12)})
        # 12/15 prefix of the full-horizon run (== the stop-at-12/15 window)
        k12 = next((i for i, c in enumerate(curve) if c >= c20.stop_k), None)
        false_by_k12 = sum(1 for c in certs if c["false"] and (k12 is None or c["cert_k"] <= k12))
        n80_raw = int(ck[k12]) if k12 is not None else tau
        n80_pen = n80_raw if (k12 is not None and false_by_k12 == 0) else tau
        out = {"method": name, "method_label": [k for k, v in LABELS.items() if v == name][0], "window": "full_horizon",
               "params": kw, "seed": int(seed)}
        out.update({f"rs_{k}": v for k, v in s.items()})  # every run_stream summary field, verbatim
        out.update({
            # full-horizon endpoints (stop_k = 15)
            "N100_pen": int(s["N80"]),  # run_stream's 'N80' field is the stop_k-generic endpoint; stop_k = 15 here
            "N100_raw": int(ck[s["k_stop"]]) if s["reached_stop"] else tau,
            "completed_15": bool(s["reached_stop"]), "k15": s["k_stop"],
            "fwer_event": bool(s["fwer_event"]), "fwer_event_full": bool(s["fwer_event"]),
            "n_cert": int(s["n_cert"]), "n_false": int(s["n_false"]), "false_q": s["false_q"],
            # stop-window (12/15) prefix endpoints
            "k12": k12, "N80_pen": int(n80_pen), "N80_raw": int(n80_raw), "completed": k12 is not None,
            "fwer_event_by_12of15": bool(false_by_k12 > 0), "false_certs_by_12of15": int(false_by_k12),
            "false_certs_after_12of15": int(s["n_false"] - false_by_k12),
            "cert_k": s["cert_k"], "cert_t": [c["t"] for c in certs], "decided_pi": s["decided_pi"],
            "certificates": certs, "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"],
            "n_cert_curve": curve, "n_false_curve": [r["n_false"] for r in rows] + [rows[-1]["n_false"] if rows else 0]
            * (len(ck) - len(rows)), "sec": round(time.perf_counter() - t0, 3), "code_sha256": code_sha,
            "error": None})
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "params": kw, "seed": int(seed), "error": traceback.format_exc()}


def lock_path_test():
    """Refusal path on in-memory tampered copies (no file is modified); the real lock must pass."""
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
    d = copy.deepcopy(real); d["eval_tasks"][TASK]["seeds"] = "30000-30001"; cases["hash_tamper"] = gate(d)
    d = copy.deepcopy(real); d["version"] = 4; cases["version_mismatch"] = gate(d)
    cases["code_drift_detected_on_wrong_root"] = bool(prereg.frozen_code_drift(real, CODE / "__nonexistent__"))
    cases["code_drift_real_root"] = prereg.frozen_code_drift(real)
    ok = cases["real_lock"] and all(not cases[k]["ok"] for k in ("status_draft", "locked_no_eval", "hash_tamper",
                                                                     "version_mismatch")) \
        and cases["code_drift_detected_on_wrong_root"] and not cases["code_drift_real_root"]
    # skipped_by_lock write path (scratch dir, never the real output dir)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sk = crm.skipped_summary(TASK, Path(td), "test refusal", "pilot")
        cases["skipped_summary_written"] = json.loads((Path(td) / "summary.json").read_text())["status"] == sk["status"] \
            == "skipped_by_lock"
    ok = ok and cases["skipped_summary_written"]
    return {"all_pass": bool(ok), "cases": cases}


def run_block(mode, out, lock, seeds=None, half=None):
    et = lock["eval_tasks"][TASK]
    assert set(et["methods"]) == set(LABELS), et["methods"]
    methods = [LABELS[x] for x in et["methods"]]
    if seeds is None:
        seeds = crm.parse_range(et["seeds"])
        assert seeds[0] == 30000 and seeds[-1] == 30199 and len(seeds) == int(et["n_streams"]) == 200
    if half is None:
        half = et["half"]
    assert et["layer"] == "CR9" and int(et["K"]) == 20
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
    ctx, c20 = _ENV["ctx"], _ENV["c20"]
    assert abs(ctx.eps - 0.001) < 1e-12 and len(ctx.checkpoints) == 20 and c20.stop_k == 12 and ctx.stop_k == 15 \
        and ctx.Q == 15
    sref = lock["structural_ctx"][f"CR9_{half}"]
    if int(ctx.tau_R) != int(sref["tau_R"]) or np.asarray(_ENV["env"].pool_sizes).tolist() != sref["pool_sizes"]:
        raise RuntimeError(f"structural ctx differs from lock CR9_{half}")
    if [int(x) for x in ctx.checkpoints] != lock["fdc_spec"]["checkpoints"][f"CR9_{half}"]:
        raise RuntimeError("checkpoints differ from lock")
    per_sha, code_sha = code_hashes()
    print(f"[{TASK}/{mode}] half={half} tau_R={ctx.tau_R} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) "
          f"methods={methods} stop_k={ctx.stop_k} (full horizon) code_sha={code_sha[:16]}", flush=True)

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
        rfile.write_text("".join(k + "\n" for k in keep))
    jobs = [(m, kws[m], s, code_sha) for m in methods for s in seeds if (m, s) not in done]
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
               "started_at": sf.read_text().strip(), "ended_at": t_end.isoformat(),
               "wall_min_this_invocation": round((t_end - t_start).total_seconds() / 60, 2),
               "timing_note": "concurrent run (4 workers, BLAS=1; other r5 blocks share the 20-core host): biased up",
               "setting": {"layer": "CR9", "half": half, "eps": ctx.eps, "K": len(ctx.checkpoints),
                           "window": "full horizon (stop_k = Q = 15, else tau_R); 12/15 prefix also recorded",
                           "stop_k_run": int(ctx.stop_k), "stop_k_prefix": int(c20.stop_k), "Q": ctx.Q,
                           "delta": ctx.delta, "tau_R": int(ctx.tau_R),
                           "checkpoints": [int(x) for x in ctx.checkpoints],
                           "budgets": [float(p.budget) for p in ctx.problems],
                           "seeds": [seeds[0], seeds[-1]], "n_streams": len(seeds)},
               "methods": methods, "method_labels": et["methods"], "params": kws,
               "lock": {"version": lock["version"], "status": lock["status"], "sha256": lock["sha256"],
                        "git_commit": lock["git_commit"]},
               "row_fields": {"N100_pen": "first checkpoint with 15/15 certified if no certificate is false, else tau_R",
                              "N100_raw": "first checkpoint with 15/15 certified (tau_R if never)",
                              "fwer_event": "any false certificate over the full horizon (== fwer_event_full)",
                              "fwer_event_by_12of15": "any false certificate issued up to the 12/15 checkpoint "
                                                      "(whole run if 12/15 never reached)",
                              "N80_pen/N80_raw/completed": "12/15-prefix endpoints, lock definitions"},
               "n_rows": len(rows), "expected_rows": total, "missing": missing[:50], "n_missing": len(missing),
               "billing_ok_all": not bill_bad, "billing_bad": bill_bad[:50],
               "results_canonical_sha256": hashlib.sha256(canon.encode()).hexdigest(),
               "results_file_sha256": sha_file(rfile), "code_sha256_combined": code_sha, "code_sha256": per_sha,
               "git": crm.git_info(), "per_method_timing_sec": timing, "eval_seeds_touched": half == "eval",
               "note": "no analysis inside the block (no CP bound / ratios); analysis happens in r5_analysis_aggregate"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary, rows


def pilot_checks(out, rows, seeds):
    """Dev-only consistency checks (no eval data): T3 full-horizon reproduction + 12/15-prefix vs dev pilot rows."""
    chk = {}
    t3 = {}
    for l in (RES / "pilots/r5_t2_gate_plugin_t3/fullhorizon.jsonl").read_text().splitlines():
        x = json.loads(l)
        t3[(x["method"], x["seed"])] = x
    mism = []
    n = 0
    for (m, s), r in rows.items():
        if (m, s) not in t3:
            continue
        n += 1
        o = t3[(m, s)]
        pairs = [("N100_pen", "N_pen"), ("N100_raw", "N_raw"), ("fwer_event", "fwer_event"), ("cert_k", "cert_k"),
                 ("decided_pi", "decided_pi"), ("n_cert_curve", "n_cert_curve"), ("schedule_digest", "schedule_digest"),
                 ("k12", "k12_in_run"), ("false_certs_by_12of15", "false_certs_by_k12")]
        bad = [a for a, b in pairs if r[a] != o[b]]
        if bad:
            mism.append({"method": m, "seed": s, "fields": bad})
    chk["repro_vs_t3_fullhorizon"] = {"n_compared": n, "n_mismatch": len(mism), "mismatches": mism[:20]}
    # 12/15 prefix vs stop-window dev pilot rows (FDC: r5_cr_main_a; B2-fav-tight: r5_cr_plugin_a)
    ref = {}
    for task, meth in (("r5_cr_main_a", "FDC"), ("r5_cr_plugin_a", "B2-fav-tight")):
        for l in (RES / "pilots" / task / "results.jsonl").read_text().splitlines():
            x = json.loads(l)
            if x.get("method") == meth and x.get("error") is None:
                ref[(meth, x["seed"])] = x
    mism = []
    n = 0
    for (m, s), r in rows.items():
        if (m, s) not in ref:
            continue
        n += 1
        o = ref[(m, s)]
        kk = o["rs_k_stop"]
        bad = []
        if r["N80_pen"] != o["N80_pen"]:
            bad.append("N80_pen")
        if r["N80_raw"] != o["N80_raw"]:
            bad.append("N80_raw")
        if r["completed"] != o["completed"]:
            bad.append("completed")
        if r["fwer_event_by_12of15"] != o["fwer_event"]:
            bad.append("fwer_event_by_12of15")
        if r["schedule_digest"] != o["schedule_digest"]:
            bad.append("schedule_digest")
        lim = (kk + 1) if kk is not None else 20
        if r["n_cert_curve"][:lim] != o["n_cert_curve"][:lim]:
            bad.append("n_cert_curve_prefix")
        if any(o["cert_k"][q] >= 0 and (r["cert_k"][q] != o["cert_k"][q]
                                         or r["decided_pi"][q] != o["rs_decided_pi"][q]) for q in range(15)):
            bad.append("certificates_prefix")
        if bad:
            mism.append({"method": m, "seed": s, "fields": bad})
    chk["prefix_vs_stop_window_dev_pilots"] = {"n_compared": n, "n_mismatch": len(mism), "mismatches": mism[:20]}
    return chk


def main_pilot(lock):
    out = RES / "pilots" / TASK
    out.mkdir(parents=True, exist_ok=True)
    lt = lock_path_test()
    (out / "lock_path_test.json").write_text(json.dumps(lt, indent=1))
    print(f"lock path test all_pass={lt['all_pass']}", flush=True)
    rfile = out / "results.jsonl"
    if rfile.exists() and os.environ.get("PILOT_FRESH", "1") == "1":
        rfile.unlink()
    t0 = time.perf_counter()
    s, rows = run_block("pilot", out, lock, seeds=DEV_PILOT_SEEDS, half="dev")
    wall = time.perf_counter() - t0
    chk = pilot_checks(out, rows, DEV_PILOT_SEEDS)
    # resume test: truncate to 37 rows + a torn line, rerun, canonical hash must be identical
    canon0 = s["results_canonical_sha256"]
    lines = rfile.read_text().splitlines()
    rfile.write_text("\n".join(lines[:37]) + "\n" + lines[37][: len(lines[37]) // 2])
    s2, _ = run_block("pilot", out, lock, seeds=DEV_PILOT_SEEDS, half="dev")
    resume = {"truncated_to": 37, "torn_line": True, "rerun_rows": s2["n_rows"],
              "canonical_hash_equal": s2["results_canonical_sha256"] == canon0}
    (out / "resume_test.json").write_text(json.dumps(resume, indent=1))
    # projection for the full block: 200 eval streams x 2 methods, LPT on 4 workers, x1.2 + max + 2 min
    methods = s["methods"]
    sec = {m: [rows[(m, x)]["sec"] for x in DEV_PILOT_SEEDS] for m in methods}
    loads = np.zeros(N_WORKERS)
    for m in sorted(methods, key=lambda m: -np.mean(sec[m])):
        for _ in range(200):
            loads[np.argmin(loads)] += float(np.mean(sec[m]))
    proj = float(loads.max() / 60 * 1.2 + max(max(v) for v in sec.values()) / 60 + 2)
    per = {}
    for m in methods:
        rr = [rows[(m, x)] for x in DEV_PILOT_SEEDS]
        per[m] = {"sec_mean": round(float(np.mean(sec[m])), 3), "sec_max": round(float(np.max(sec[m])), 3),
                  "completed_15": sum(x["completed_15"] for x in rr), "completed_12": sum(x["completed"] for x in rr),
                  "fwer_streams_full": sum(x["fwer_event"] for x in rr),
                  "fwer_streams_by_12of15": sum(x["fwer_event_by_12of15"] for x in rr),
                  "n_cert_hist": {str(k): int(v) for k, v in zip(*np.unique([x["n_cert"] for x in rr],
                                                                             return_counts=True))},
                  "billing_ok_all": all(x["billing_ok"] for x in rr)}
    samples = [{k: rows[(m, x)][k] for k in ("method", "seed", "N100_pen", "N100_raw", "N80_pen", "N80_raw", "k12",
                                             "completed_15", "fwer_event", "fwer_event_by_12of15", "cert_t",
                                             "n_cert_curve")}
               | {"false_certs": [c for c in rows[(m, x)]["certificates"] if c["false"]]}
               for x in (900, 905, 917, 933, 949) for m in methods]
    (out / "samples.json").write_text(json.dumps(samples, indent=1))
    crit = {"billing_ok_all": s["billing_ok_all"], "no_exceptions": s["status"] == "complete",
            "n_runs_ge_100": s["n_rows"] >= 100, "projected_full_min_le_55": proj <= 55,
            "lock_assertion_path_tested": lt["all_pass"],
            "reproduces_t3_full_horizon": chk["repro_vs_t3_fullhorizon"]["n_mismatch"] == 0
            and chk["repro_vs_t3_fullhorizon"]["n_compared"] > 0,
            "prefix_matches_stop_window_pilots": chk["prefix_vs_stop_window_dev_pilots"]["n_mismatch"] == 0
            and chk["prefix_vs_stop_window_dev_pilots"]["n_compared"] > 0,
            "resume_ok": resume["canonical_hash_equal"]}
    go = all(crit.values())
    ps = {"task_id": TASK, "mode": "pilot (dev streams 900-949; eval seeds untouched)", "go_no_go": "GO" if go else "NO_GO",
          "criteria": crit, "checks": chk, "resume_test": resume, "lock_path_test": lt,
          "projected_full_block_min": round(proj, 1),
          "projection_rule": "mean dev sec/stream per method x 200 streams, LPT on 4 workers, x1.2 + max/60 + 2 min",
          "pilot_run_wall_min": round(wall / 60, 2), "per_method_dev": per, "block_summary": s,
          "timing_note": "concurrent run (r5_cr_factorial / r5_cr_k60_sens / r5_cr12_scale share the host)",
          "eval_seeds_touched": False,
          "full_cmd": "cd exp/code && python run_r5_cr_fwer_audit.py --mode full"}
    (out / "pilot_summary.json").write_text(json.dumps(ps, indent=1, default=str))
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "full"])
    a = ap.parse_args()
    out = RES / ("pilots" if a.mode == "pilot" else "full") / TASK
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
            txt = f"{s['status']} rows={s['n_rows']}/{s['expected_rows']} billing_ok={s['billing_ok_all']}"
            crm.mark_done(TASK, "success" if s["status"] == "complete" else "failed", txt)
        else:
            s = main_pilot(lock)
            txt = (f"pilot {s['go_no_go']} | proj full {s['projected_full_block_min']} min | criteria={s['criteria']}")
            crm.mark_done(TASK, "success", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        crm.mark_done(TASK, "failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
