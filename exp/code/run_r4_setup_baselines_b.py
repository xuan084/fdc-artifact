"""r4_setup_baselines_b (PILOT): B3 RAGE / Peace (rect, nominal, fav), B5 maq-Qini (descriptive), H8 in-model
certifiers (JPC one-step additive logistic, T-learner + bootstrap, MisLid online / eps_mis = 0).

Steps
  1. unit tests (test_baselines_r4b, test_no_truth_import; full suite at the end);
  2. toy FWER: 200 small synthetic finite-pool streams per method (20 near-tie populations, toy dev seeds 920-939,
     x 10 permutation seeds; adaptive pool exhaustion included), CP one-sided upper bound, billing mismatches;
  3. Peace Gaussian-width MC standard error: 2000 eta draws at the XY design of every problem's full feasible set on
     CR9 dev / HR8 dev (sigma^2 = 1/4 and dev-half plug-in variances), plus every round logged by the timing runs;
  4. per-stream cost on CR9 dev (eps 0.002) and HR8 dev (eps 0.0125), TIMING seeds 880-883 (outside all dev / eval
     seeds); per-method stop / censoring counts are recorded as implementation facts (e.g. nominal constants that
     never stop within tau_R), NOT as a method comparison (no N80 ratios are summarised; not a gate).
CPU only; <= 4 workers; BLAS threads pinned to 1; timings are from a concurrent run.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
TASK = "r4_setup_baselines_b"
PY = __import__('sys').executable

from dsswm.baselines.maq_desc import qini_curve  # noqa: E402
from dsswm.baselines.peace_frontier import gaussian_width  # noqa: E402
from dsswm.baselines.rage_frontier import pair_signatures, xy_design  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import (BASELINES_B, build_ctx, make_frontier_method, run_stream,  # noqa: E402
                                           true_policy_values, write_jsonl)

FILES = ["dsswm/baselines/rage_frontier.py", "dsswm/baselines/peace_frontier.py", "dsswm/baselines/maq_desc.py",
         "dsswm/baselines/h8_models.py", "dsswm/streams/frontier_runner.py", "dsswm/tests/test_baselines_r4b.py",
         "run_r4_setup_baselines_b.py"]
VALID = ["B3-rect", "Peace-rect"]
TIMING_SEEDS = (880, 881, 882, 883)
_G = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def progress(rd, step, total, metric=None):
    (rd / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def run_pytest(args, timeout=2400):
    t = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", *args], cwd=CODE, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}


def _jd(o):
    return o.tolist() if hasattr(o, "tolist") else (float(o) if isinstance(o, np.floating) else str(o))


# ---------------------------------------------------------------------------------------------- toy FWER
def _toy_job(args):
    name, sd = args
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    out = []
    for p in range(10):
        m = make_frontier_method(name, env)
        t0 = time.perf_counter()
        s, _ = run_stream(env, m, 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
        out.append({k: s[k] for k in ("method", "perm_seed", "fwer_event", "n_cert", "n_false", "billing_ok",
                                      "skipped", "reselected", "N80", "censored", "t_end")} |
                   {"tau_R": env.tau_R, "sec": round(time.perf_counter() - t0, 3)})
    return out


# ---------------------------------------------------------------------------------------------- real layers
def _real_job(args):
    name, sd = args
    env, problems, eps, ctx, J, Js = (_G[k] for k in ("env", "problems", "eps", "ctx", "J", "Js"))
    m = make_frontier_method(name, env)
    s, rows = run_stream(env, m, sd, problems, eps, ctx=ctx, J_true=J, J_star=Js)
    d = m.describe()
    gw = [g.get("W_mc_rel_se") for g in getattr(m, "gw_log", [])]
    return name, sd, s, rows, d, gw


def gw_se_table(env, ctx, n_eta=2000, seed=777):
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n_eta, ctx.S, ctx.A))
    out = []
    sig_plug = env.true_sigma2("visit")              # dev-half plug-in variance (setup diagnostic only)
    for q in range(ctx.Q):
        rows = ctx.pols[np.flatnonzero(ctx.feas[q])]
        if rows.shape[0] < 2:
            continue
        inc = pair_signatures(rows, ctx.A)
        for lab, var in (("sigma2_1/4", np.full((ctx.S, ctx.A), 0.25)), ("plugin_dev", np.maximum(sig_plug, 1e-6))):
            p, _ = xy_design(inc, ctx.w, var)
            W, se = gaussian_width(rows, ctx.w, var, p, eta)
            out.append({"q": q, "var": lab, "n_pol": int(rows.shape[0]), "W": W, "mc_rel_se": se})
    return out


def real_layer(layer, eps_run, problems, out_dir, rd, methods):
    from dsswm.envs.pool_replay import PoolReplayEnv
    t0 = time.time()
    env = PoolReplayEnv(layer, "dev")
    sec_env = time.time() - t0
    ctx = build_ctx(env, problems, eps_run)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    t1 = time.time()
    gw = gw_se_table(env, ctx)
    sec_gw = time.time() - t1
    _G.update({"env": env, "problems": problems, "eps": eps_run, "ctx": ctx, "J": J, "Js": Js})
    jobs = [(m, sd) for m in methods for sd in TIMING_SEEDS]
    jobs.sort(key=lambda j: 0 if j[0].startswith("Peace") else 1)            # longest first
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_real_job, jobs, chunksize=1)
    per = {}
    (out_dir / "samples").mkdir(exist_ok=True)
    gw_runs = []
    for name, sd, s, rows, d, g in res:
        for r in rows:
            r["layer"], r["stream"] = layer, sd
        write_jsonl(rows, out_dir / "results.jsonl")
        (out_dir / "samples" / f"{layer}_{name}_{sd}.json").write_text(json.dumps(
            {"summary": s, "trajectory": rows, "method": d}, default=_jd))
        gw_runs += [x for x in g if x is not None]
        per.setdefault(name, []).append(s)
    timing = {}
    for name in methods:
        ss = per[name]
        secs = [s["sec_total"] for s in ss]
        timing[name] = {"sec_per_stream_mean": round(float(np.mean(secs)), 3),
                        "sec_per_stream_max": round(float(np.max(secs)), 3),
                        "billing_ok": bool(all(s["billing_ok"] for s in ss)),
                        "reached_stop": int(sum(s["reached_stop"] for s in ss)),
                        "fwer_events": int(sum(s["fwer_event"] for s in ss)),
                        "n_cert_mean": float(np.mean([s["n_cert"] for s in ss])),
                        "skipped_total": int(sum(s["skipped"] for s in ss)),
                        "reselected_total": int(sum(s["reselected"] for s in ss)),
                        "streams": len(ss)}
    qini = qini_curve(env.w, env.true_mu("visit"), env.pool_sizes, problems[0].kappa) if env.A == 2 else None
    return {"layer": layer, "half": "dev", "N": env.N, "tau_R": env.tau_R, "replan": env.replan_interval,
            "S": env.S, "A": env.A, "P": ctx.P, "Q": ctx.Q, "eps_timing": eps_run, "timing_seeds": list(TIMING_SEEDS),
            "sec_env_build": round(sec_env, 1), "seg_desc": env.seg_desc,
            "feasible_sizes": [int(f.sum()) for f in ctx.feas],
            "gw_mc_se_static": {"sec": round(sec_gw, 1), "max_rel_se": max(x["mc_rel_se"] for x in gw),
                                "rows": gw},
            "gw_mc_se_runs": {"n": len(gw_runs), "max_rel_se": max(gw_runs) if gw_runs else None},
            "qini_dev_full_pool_truth": qini, "timing": timing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot")
    ap.add_argument("--skip-full-suite", action="store_true")
    a = ap.parse_args()
    rd = WS / "exp" / "results"
    out = rd / ("pilots" if a.mode == "pilot" else "full") / TASK
    out.mkdir(parents=True, exist_ok=True)
    for f in ("results.jsonl", "toy_fwer_streams.jsonl"):
        if (out / f).exists():
            (out / f).unlink()
    (rd / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "started_at": datetime.now().isoformat(),
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)"}
    progress(rd, 0, 5)
    summary["tests"] = {"baselines_r4b": run_pytest(["dsswm/tests/test_baselines_r4b.py"]),
                        "no_truth_import": run_pytest(["dsswm/tests/test_no_truth_import.py"])}
    progress(rd, 1, 5)
    # toy FWER
    t0 = time.time()
    jobs = [(m, sd) for m in BASELINES_B for sd in range(920, 940)]
    jobs.sort(key=lambda j: 0 if j[0].startswith("Peace") else 1)
    with get_context("fork").Pool(4) as pool:
        res = [r for chunk in pool.map(_toy_job, jobs, chunksize=1) for r in chunk]
    toy = {}
    for m in BASELINES_B:
        rr = [r for r in res if r["method"] == m]
        ev = sum(r["fwer_event"] for r in rr)
        nc = sum(r["n_cert"] for r in rr)
        nf = sum(r["n_false"] for r in rr)
        toy[m] = {"validity": make_frontier_method(m).validity, "streams": len(rr), "fwer_events": ev,
                  "fwer": ev / len(rr), "fwer_cp_upper": cp_upper(ev, len(rr)), "certs_total": nc, "false_total": nf,
                  "fcr_old": nf / nc if nc else 0.0, "billing_mismatch": sum(not r["billing_ok"] for r in rr),
                  "streams_with_exhaustion": sum(r["skipped"] + r["reselected"] > 0 for r in rr),
                  "reached_stop": sum(not r["censored"] for r in rr),
                  "stopped_before_half_tau": sum((not r["censored"]) and r["N80"] < 0.5 * r["tau_R"] for r in rr),
                  "sec_per_stream_mean": round(float(np.mean([r["sec"] for r in rr])), 3)}
    write_jsonl(res, out / "toy_fwer_streams.jsonl")
    summary["toy_fwer"] = {"eps": 0.02, "delta": 0.05, "problems": 5, "stop_k": 4,
                           "populations": "toy dev seeds 920-939 x perm 0-9", "sec": round(time.time() - t0, 1),
                           "by_method": toy}
    progress(rd, 2, 5, {"toy_done": True})
    summary["CR9"] = real_layer("CR9", 0.002, fr.cr_problems(), out, rd, list(BASELINES_B))
    progress(rd, 3, 5)
    summary["HR8"] = real_layer("HR8", 0.0125, fr.hr_problems(), out, rd, list(BASELINES_B))
    progress(rd, 4, 5)
    if not a.skip_full_suite:
        summary["tests"]["full_suite"] = run_pytest(["dsswm/tests"], timeout=3600)
    billing_mm = sum(toy[m]["billing_mismatch"] for m in toy) + sum(
        int(not v["billing_ok"]) for L in ("CR9", "HR8") for v in summary[L]["timing"].values())
    gw_max = max(x for x in (summary["CR9"]["gw_mc_se_static"]["max_rel_se"],
                             summary["HR8"]["gw_mc_se_static"]["max_rel_se"],
                             summary["CR9"]["gw_mc_se_runs"]["max_rel_se"],
                             summary["HR8"]["gw_mc_se_runs"]["max_rel_se"]) if x is not None)
    crit = {"all_unit_tests_pass": all(v["returncode"] == 0 for v in summary["tests"].values()),
            "valid_variants": VALID,
            "valid_variants_toy_fwer_cp_upper_le_0.05": all(toy[m]["fwer_cp_upper"] <= 0.05 for m in VALID),
            "peace_gw_mc_rel_se_max": gw_max, "peace_gw_mc_se_lt_2pct": bool(gw_max < 0.02),
            "billing_mismatch": billing_mm}
    crit["pass"] = bool(crit["all_unit_tests_pass"] and crit["valid_variants_toy_fwer_cp_upper_le_0.05"] and
                        crit["peace_gw_mc_se_lt_2pct"] and billing_mm == 0)
    summary["pass_criteria"] = crit
    summary["go_no_go"] = "GO" if crit["pass"] else "NO_GO"
    summary["code_sha256"] = {f: sha(CODE / f) for f in FILES}
    summary["sec_total"] = round(time.time() - t_start, 1)
    summary["finished_at"] = datetime.now().isoformat()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jd))
    print(json.dumps(crit), summary["sec_total"])


if __name__ == "__main__":
    main()
