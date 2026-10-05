"""r4_setup_baselines_a (PILOT): shared frontier runner + B1 / B4 / B2-{rect,nominal,fav} + A-Ney / A-XY.

Steps
  1. unit tests (test_baselines_r4a, test_no_truth_import, full suite);
  2. toy FWER: 200 small synthetic finite-pool streams per method (20 near-tie populations, dev seeds 900-919, x 10
     permutation seeds; adaptive pool exhaustion included), CP one-sided upper bound, billing mismatches;
  3. frozen designs on the development halves: A-Ney (Neyman) and A-XY (Frank-Wolfe XY design) for CR9 (every eps of
     the CR grid) and HR8 (HR8 grid); frozen B2-nominal threshold choice for CR9 / HR8 (data-free);
  4. per-stream cost (seconds) on CR9 dev and HR8 dev with TIMING seeds 880-883 (outside the G-eps / G1 dev seeds
     900-999 and all evaluation seeds); method-vs-method N80 is NOT summarised here (not a gate, no comparison).
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
TASK = "r4_setup_baselines_a"
PY = __import__('sys').executable

from dsswm.baselines.clucb_joint import CLUCBJoint  # noqa: E402
from dsswm.baselines.combgame_joint import CombGameJoint, freeze_beta  # noqa: E402
from dsswm.baselines.frontier_common import QFCMethod  # noqa: E402
from dsswm.baselines.frozen_alloc import neyman_alloc, xy_alloc  # noqa: E402
from dsswm.baselines.uniform_rs import UniformRS  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values, write_jsonl  # noqa: E402

FILES = ["dsswm/streams/frontier_runner.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/clucb_joint.py",
         "dsswm/baselines/uniform_rs.py", "dsswm/baselines/combgame_joint.py", "dsswm/baselines/frozen_alloc.py",
         "dsswm/tests/test_baselines_r4a.py", "run_r4_setup_baselines_a.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def progress(rd, step, total, metric=None):
    (rd / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def run_pytest(args, timeout=1800):
    t = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", *args], cwd=CODE, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}


# ---------------------------------------------------------------------------------------------- toy FWER
def _toy_job(args):
    name, sd = args
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    out = []
    for p in range(10):
        m = make_method(name, env)
        s, _ = run_stream(env, m, 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
        out.append({k: s[k] for k in ("method", "perm_seed", "fwer_event", "n_cert", "n_false", "billing_ok",
                                      "skipped", "reselected", "N80", "censored", "t_end")} | {"tau_R": env.tau_R})
    return out


def make_method(name, env=None, allocs=None):
    if name == "QFC":
        return QFCMethod()
    if name == "B4":
        return UniformRS()
    if name == "B1":
        return CLUCBJoint()
    if name == "NAIVE-control":       # harness power check: plug-in GLR at beta = ln(1/delta), no union, no time
        import math
        return CombGameJoint("fav", beta_override=math.log(1 / 0.05), name="NAIVE-control")
    if name.startswith("B2-"):
        return CombGameJoint(name[3:])
    if name == "A-Ney":
        p = allocs["A-Ney"] if allocs else neyman_alloc(env.true_sigma2("visit"))
        return QFCMethod("A-Ney", alloc_p=p)
    if name == "A-XY":
        return QFCMethod("A-XY", alloc_p=allocs["A-XY"])
    raise KeyError(name)


TOY_METHODS = ["QFC", "A-Ney", "B4", "B1", "B2-rect", "B2-nominal", "B2-fav", "NAIVE-control"]
REAL_METHODS = ["QFC", "A-Ney", "A-XY", "B4", "B1", "B2-rect", "B2-nominal", "B2-fav"]
TIMING_SEEDS = (880, 881, 882, 883)


# ---------------------------------------------------------------------------------------------- real layers
def real_layer(layer, eps_grid, eps_run, problems, out_dir, rd):
    from dsswm.envs.pool_replay import PoolReplayEnv
    t0 = time.time()
    env = PoolReplayEnv(layer, "dev")
    sec_env = time.time() - t0
    mu, sig2 = env.true_mu("visit"), env.true_sigma2("visit")
    ctx = build_ctx(env, problems, eps_run)
    J = true_policy_values(ctx.pols, env.w, mu)
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    ney = neyman_alloc(sig2)
    xy = {}
    for e in eps_grid:
        p, info = xy_alloc(env.w, mu, sig2, ctx.pols, ctx.feas, e)
        xy[str(e)] = {"p": p.tolist(), **info}
    beta = freeze_beta(ctx)
    allocs = {"A-Ney": ney, "A-XY": np.array(xy[str(eps_run)]["p"])}
    timing = {}
    for name in REAL_METHODS:
        secs, bill, sk, loops = [], True, 0, 0
        for sd in TIMING_SEEDS:
            m = make_method(name, env, allocs)
            s, rows = run_stream(env, m, sd, problems, eps_run, ctx=ctx, J_true=J, J_star=Js)
            secs.append(s["sec_total"])
            bill &= s["billing_ok"]
            sk += s["skipped"]
            loops += s["loop_arrivals"]
            for r in rows:
                r["layer"], r["stream"] = layer, sd
            write_jsonl(rows, out_dir / "results.jsonl")
            (out_dir / "samples").mkdir(exist_ok=True)
            (out_dir / "samples" / f"{layer}_{name}_{sd}.json").write_text(json.dumps(
                {"summary": s, "trajectory": rows, "method": m.describe()}, default=lambda o: o.tolist()
                if hasattr(o, "tolist") else str(o)))
        timing[name] = {"sec_per_stream_mean": round(float(np.mean(secs)), 3),
                        "sec_per_stream_max": round(float(np.max(secs)), 3), "billing_ok": bool(bill),
                        "skipped_total": int(sk), "loop_arrivals_total": int(loops)}
        progress(rd, 0, 1, {"layer": layer, "method": name})
    return {"layer": layer, "half": "dev", "N": env.N, "tau_R": env.tau_R, "replan": env.replan_interval,
            "S": env.S, "A": env.A, "P": ctx.P, "Q": ctx.Q, "checkpoints": ctx.checkpoints.tolist(),
            "sec_env_build": round(sec_env, 1), "eps_timing": eps_run, "timing_seeds": list(TIMING_SEEDS),
            "frozen_A_Ney": ney.tolist(), "frozen_A_XY": xy, "B2_threshold_frozen": beta,
            "delta_cell_cs": 0.05 / (env.S * env.A), "timing": timing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot")
    ap.add_argument("--skip-full-suite", action="store_true")
    a = ap.parse_args()
    rd = WS / "exp" / "results"
    out = rd / ("pilots" if a.mode == "pilot" else "full") / TASK
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.jsonl").exists():
        (out / "results.jsonl").unlink()
    (rd / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    started = datetime.now().isoformat()
    summary = {"task_id": TASK, "mode": a.mode, "started_at": started,
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)"}
    progress(rd, 0, 4)
    summary["tests"] = {"baselines_r4a": run_pytest(["dsswm/tests/test_baselines_r4a.py"]),
                        "no_truth_import": run_pytest(["dsswm/tests/test_no_truth_import.py"])}
    progress(rd, 1, 4)
    # toy FWER
    t0 = time.time()
    jobs = [(m, sd) for m in TOY_METHODS for sd in range(900, 920)]
    with get_context("fork").Pool(4) as pool:
        res = [r for chunk in pool.map(_toy_job, jobs) for r in chunk]
    toy = {}
    for m in TOY_METHODS:
        rr = [r for r in res if r["method"] == m]
        ev = sum(r["fwer_event"] for r in rr)
        nc = sum(r["n_cert"] for r in rr)
        nf = sum(r["n_false"] for r in rr)
        toy[m] = {"streams": len(rr), "fwer_events": ev, "fwer": ev / len(rr), "fwer_cp_upper": cp_upper(ev, len(rr)),
                  "certs_total": nc, "false_total": nf, "fcr_old": nf / nc if nc else 0.0,
                  "billing_mismatch": sum(not r["billing_ok"] for r in rr),
                  "streams_with_exhaustion": sum(r["skipped"] + r["reselected"] > 0 for r in rr),
                  "reached_stop": sum(not r["censored"] for r in rr),
                  "stopped_before_half_tau": sum((not r["censored"]) and r["N80"] < 0.5 * r["tau_R"] for r in rr)}
    write_jsonl(res, out / "toy_fwer_streams.jsonl")
    summary["toy_fwer"] = {"eps": 0.02, "delta": 0.05, "problems": 5, "stop_k": 4,
                           "populations": "dev seeds 900-919 x perm 0-9", "sec": round(time.time() - t0, 1),
                           "by_method": toy}
    progress(rd, 2, 4)
    # real layers
    summary["CR9"] = real_layer("CR9", fr.CR_EPS_GRID, 0.002, fr.cr_problems(), out, rd)
    progress(rd, 3, 4)
    summary["HR8"] = real_layer("HR8", (0.01, 0.0125, 0.015, 0.0175), 0.0125, fr.hr_problems(), out, rd)
    if not a.skip_full_suite:
        summary["tests"]["full_suite"] = run_pytest(["dsswm/tests"], timeout=3000)
    # pass criteria
    valid = ["QFC", "A-Ney", "B4", "B1", "B2-rect"]
    billing_mm = sum(toy[m]["billing_mismatch"] for m in toy) + sum(
        int(not v["billing_ok"]) for L in ("CR9", "HR8") for v in summary[L]["timing"].values())
    crit = {"all_unit_tests_pass": all(v["returncode"] == 0 for v in summary["tests"].values()),
            "valid_variants_toy_fwer_cp_upper_le_0.05": all(toy[m]["fwer_cp_upper"] <= 0.05 for m in valid),
            "billing_mismatch": billing_mm}
    crit["pass"] = crit["all_unit_tests_pass"] and crit["valid_variants_toy_fwer_cp_upper_le_0.05"] and billing_mm == 0
    summary["pass_criteria"] = crit
    summary["go_no_go"] = "GO" if crit["pass"] else "NO_GO"
    summary["code_sha256"] = {f: sha(CODE / f) for f in FILES}
    summary["sec_total"] = round(time.time() - t_start, 1)
    summary["finished_at"] = datetime.now().isoformat()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist()
                                                 if hasattr(o, "tolist") else str(o)))
    print(json.dumps(crit), summary["sec_total"])


if __name__ == "__main__":
    main()
