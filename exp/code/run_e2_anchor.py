"""e2_anchor_{a,b}: E2 continuous external anchor (descriptive only, enters no test). Round 1 (RH-JPC).

Usage: run_e2_anchor.py --task-id e2_anchor_a --mode {pilot,full} [--workers 1] [--problems K]

Reuses run_tier2_e2_scale.py (v2: per-row MC radius folded into the dual objective) unchanged, with:
  * TASK = task id (PID / PROGRESS / DONE files follow the scheduler protocol);
  * EPS = 0.01 (round-1 pre-registered E2 epsilon, plan/prereg_lock.json eps_E2), delta = 0.04 model + 0.01 MC;
  * MC rollouts chunked at <= MC_CHUNK episodes per GPU call (stage 4/5 batches shrunk) so one worker stays <= 5 GB
    on the shared RTX 4090 (CRN still holds: one generator stream per method, consumed in the same order);
  * a PLUGIN reference arm: point-estimate pick at v_hat after the shared n0 rounds, zero new steps, always
    "certifies"; its error rate is the non-triviality of eps (it is NOT a certifier and is excluded from ratios);
  * seeds: pilot = dev instances 698-699; full = the task's eval range from plan/prereg_lock.json
    (requires status=locked via dsswm.stats.prereg.assert_locked()).
All E2 numbers are an INNER approximation (sparse product grid inside the continuous LR set), not a guaranteed
upper bound, with the fixed-n Wilks radius. Truth = 1e6-rollout MC estimates.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

import run_tier2_e2_scale as T  # noqa: E402
from dsswm.models import e2_model as M  # noqa: E402
from dsswm.streams.e2_generator import E2_DEFAULTS, e2_generator_hash  # noqa: E402

EPS_E2 = 0.01
MC_CHUNK = 262144
PILOT_SEEDS = [698, 699]
METHODS_DEFAULT = "JPC,B3,B1,B1eb,PLUGIN"


def _task_from_argv():
    for i, a in enumerate(sys.argv):
        if a == "--task-id" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--task-id="):
            return a.split("=", 1)[1]
    return os.environ.get("E2_ANCHOR_TASK", "e2_anchor_a")


# ---------------------------------------------------------------- module patches (applied at import -> spawn children too)
TASK_ID = _task_from_argv()
os.environ["E2_ANCHOR_TASK"] = TASK_ID
T.TASK = TASK_ID
T.EPS = EPS_E2
T.SAMPLE_ALL = True


def mc_members_chunked(Vm, q, N, gen):
    """Same contract as run_tier2_e2_scale.mc_members, but episodes are drawn and rolled out in chunks of MC_CHUNK."""
    Vt = torch.tensor(Vm, device=T.DEV, dtype=torch.float32)
    K = len(q.policies)
    csum = np.zeros((len(Vm), K, M.P))
    tots = []
    done = 0
    while done < N:
        n = min(MC_CHUNK, N - done)
        U = torch.rand((n, q.H, M.L, M.R), device=T.DEV, generator=gen)
        E = torch.rand((n, q.H, M.P), device=T.DEV, generator=gen)
        Cm, Tot = [], []
        for pol in q.policies:
            c = M.rollout_contrib(Vt, pol, q.loads0, q.engaged0, q.H, U, E, q.utility.w, q.utility.w_ret, q.utility.c_q)
            Cm.append(c.sum(1).double().cpu().numpy()); Tot.append(c.sum(2))
            del c
        csum += np.stack(Cm, 1)
        tots.append(torch.stack(Tot, 1))
        del U, E
        done += n
    return csum / N, (tots[0] if len(tots) == 1 else torch.cat(tots, 2))


T.mc_members = mc_members_chunked


def run_plugin(ctx, log=None):
    inst = ctx.fresh()
    data = M.E2Data()
    for o in inst.init_obs:
        data.add(o)
    gen = torch.Generator(device=T.DEV); gen.manual_seed(3_000_003 * ctx.seed + ctx.noise)
    t0 = time.perf_counter()
    v_hat, _ = M.fit_mle(data, device="cpu")
    rows = []
    for k, q in enumerate(ctx.problems):
        t1 = time.perf_counter()
        Jh, _, _ = M.values_and_grads(v_hat, q.policies, q.loads0, q.engaged0, q.H, q.utility.w, q.utility.w_ret,
                                      q.utility.c_q, T.N_GRAD, gen, device=T.DEV)
        pi = int(np.argmax(Jh))
        rows.append(T.score_row(ctx, k, "PLUGIN", "CERTIFIED", pi, 0, False,
                                {"wall_clock_s": time.perf_counter() - t1, "rollouts": len(q.policies) * T.N_GRAD,
                                 "zero_cost_cert": True, "note": "reference only: point-estimate pick, no certificate"}))
        if log:
            log(f"  inst {ctx.seed} q{k} PLUGIN pick={pi} regret={rows[-1]['true_regret']:.4f}")
    return rows


def run_instance(seed, noise, n_problems, tmax, tmax_trial, methods, out_dir):
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    logf = open(Path(out_dir) / f"worker_{seed}_{noise}.log", "a")

    def log(msg):
        logf.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); logf.flush()

    t0 = time.time()
    rows, samples, err, mwall = [], [], None, {}
    try:
        ctx = T.Ctx(seed, noise, n_problems)
        mwall["truth"] = time.time() - t0
        log(f"inst {seed} truth done {mwall['truth']:.1f}s")
        for m in methods:
            tm = time.time()
            if m == "JPC":
                rows += T.run_jpc(ctx, tmax, samples=samples, log=log)
            elif m == "B3":
                rows += T.run_b3(ctx, tmax, log=log)
            elif m == "B1":
                rows += T.run_trials(ctx, tmax_trial, "hoeffding", log=log)
            elif m == "B1eb":
                rows += T.run_trials(ctx, tmax_trial, "eb", log=log)
            elif m == "PLUGIN":
                rows += run_plugin(ctx, log=log)
            mwall[m] = time.time() - tm
            log(f"inst {seed} {m} done {mwall[m]:.1f}s peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    except Exception:
        err = traceback.format_exc()
        log(err)
    torch.cuda.empty_cache()
    return {"seed": seed, "noise": noise, "rows": rows, "samples": samples, "error": err, "method_wall_s": mwall,
            "wall_s": time.time() - t0, "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


def _job(a):
    return run_instance(*a)


def ratios(rows):
    """Per-problem (steps+1) ratios and per-instance cumulative-step ratios of JPC over each baseline."""
    idx = {}
    for r in rows:
        idx.setdefault((r["instance"], r["noise_seed"], r["problem_index"]), {})[r["method"]] = r
    out = {}
    for b in ("B3", "B1eb", "B1"):
        pp = [(d["JPC"]["new_env_steps"] + 1) / (d[b]["new_env_steps"] + 1) for d in idx.values() if "JPC" in d and b in d]
        both = [(d["JPC"]["new_env_steps"] + 1) / (d[b]["new_env_steps"] + 1) for d in idx.values()
                if "JPC" in d and b in d and not d["JPC"]["censored"] and not d[b]["censored"]]
        cum = {}
        for (i, nz, k), d in idx.items():
            if "JPC" in d and b in d:
                c = cum.setdefault((i, nz), [0, 0])
                c[0] += d["JPC"]["new_env_steps"]; c[1] += d[b]["new_env_steps"]
        cr = [a / max(bb, 1) for a, bb in cum.values()]
        if pp:
            out[f"JPC_over_{b}"] = {
                "per_problem_ratio_median": float(np.median(pp)), "q25": float(np.quantile(pp, .25)),
                "q75": float(np.quantile(pp, .75)), "n": len(pp),
                "per_problem_ratio_median_both_completed": float(np.median(both)) if both else None,
                "n_both_completed": len(both),
                "cumulative_ratio_per_instance": [float(x) for x in cr],
                "cumulative_ratio_median": float(np.median(cr)) if cr else None,
                "note": "(steps+1) ratio; censored runs enter at their cap; whole-trial baselines count rollouts"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default=TASK_ID, choices=["e2_anchor_a", "e2_anchor_b"])
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=3000)
    ap.add_argument("--tmax-trial", type=int, default=20_000_000)
    ap.add_argument("--methods", default=METHODS_DEFAULT)
    args = ap.parse_args()
    task = args.task_id
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / task
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (res_root / f"{task}.pid").write_text(str(os.getpid()))
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    if args.mode == "full":
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        lo, hi = lock["eval_manifest"]["per_task_ranges"][task][0]
        seeds = list(range(lo, hi + 1))
        n_prob = args.problems or 8
    else:
        seeds = PILOT_SEEDS
        n_prob = args.problems or 10
    noises = [42]
    methods = args.methods.split(",")
    log(f"start task={task} mode={args.mode} seeds={seeds} problems={n_prob} methods={methods} eps={T.EPS} "
        f"delta={T.DELTA} (model {T.DELTA_MODEL} + MC {T.DELTA_MC}) R_LR={T.R_LR:.2f} N_stages={T.N_STAGES} "
        f"mc_chunk={MC_CHUNK} n0={E2_DEFAULTS['n0']} workers={args.workers} (concurrent with other tasks)")
    jobs = [(s, nz, n_prob, args.tmax, args.tmax_trial, methods, str(out_dir)) for nz in noises for s in seeds]
    T.progress(res_root, 0, len(jobs), "running")
    t_run = time.time()
    results = []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap_unordered(_job, jobs):
            results.append(res)
            log(f"instance {res['seed']} done wall={res['wall_s']:.0f}s rows={len(res['rows'])} "
                f"mem={res['max_mem_gb']:.2f}GB method_wall={ {k: round(v) for k, v in res['method_wall_s'].items()} } "
                f"err={'yes' if res['error'] else 'no'}")
            T.progress(res_root, len(results), len(jobs), "running", {"elapsed_s": round(time.time() - t_run, 1)})
    rows = sorted([r for res in results for r in res["rows"]],
                  key=lambda r: (r["method"], r["instance"], r["noise_seed"], r["problem_index"]))
    errors = [{"seed": r["seed"], "error": r["error"]} for r in results if r["error"]]
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "samples" / "jpc_traces.json").write_text(json.dumps([s for res in results for s in res["samples"]], indent=1))
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
    b1 = sorted(r["new_env_steps"] for r in rows if r["method"] == "B1" and not r["censored"])
    n_b1 = sum(1 for r in rows if r["method"] == "B1")
    if n_b1 and len(b1) >= math.ceil(0.5 * n_b1):
        tmax_common, rule = int(b1[math.ceil(0.5 * n_b1) - 1]), "smallest T with B1 completion >= 50%"
    else:
        tmax_common, rule = args.tmax_trial, "B1 completion < 50% even at tmax_trial; T_max = tmax_trial"
    summ = T.summarize(rows, methods, tmax_common, args.tmax)
    rat = ratios(rows)
    wall = time.time() - t_run
    peak = max([r["max_mem_gb"] for r in results] + [0.0])
    n_runs = len(rows)
    cu = sum(r["status"] == "COMPUTE_UNKNOWN" for r in rows if r["method"] != "PLUGIN")
    n_cert_methods = sum(1 for r in rows if r["method"] != "PLUGIN")
    cu_rate = cu / max(n_cert_methods, 1)
    # full projection: 12 eval instances x 8 problems, 1 worker; scale from measured per-instance wall
    per_inst = [r["wall_s"] * (8 / n_prob) for r in results]
    projected_full_min = (float(np.mean(per_inst)) * 12 / max(args.workers, 1) / 60.0) if per_inst else None
    checks = {"zero_crashes": not errors and n_runs == len(seeds) * n_prob * len(methods),
              "compute_unknown_lt_5pct": cu_rate < 0.05, "peak_gpu_le_5GB": peak <= 5.0,
              "projected_full_le_60min": projected_full_min is not None and projected_full_min <= 60.0}
    j = summ.get("JPC", {})
    summary = {
        "task_id": task, "mode": args.mode, "role": "external direction anchor; enters no test (methodology 2.6)",
        "env": "E2 (6x6, H=8, continuous theta in R^32)", "eps": T.EPS, "delta": T.DELTA, "delta_model": T.DELTA_MODEL,
        "delta_mc": T.DELTA_MC, "R_LR": T.R_LR, "beta_wald": T.BETA_WALD, "instance_seeds": seeds,
        "noise_seeds": noises, "problems_per_instance": n_prob, "n0": E2_DEFAULTS["n0"],
        "e2_generator_hash": e2_generator_hash(), "tmax_step_methods": args.tmax, "tmax_trial_methods": args.tmax_trial,
        "T_max_common": tmax_common, "T_max_rule": rule,
        "mc": {"N_grad": T.N_GRAD, "N_stages": T.N_STAGES, "N_refine_caps": [min(c, 10**6) for c in T.N_REFINE],
               "N_B3": T.N_B3, "N_true": T.N_TRUE, "mc_chunk": MC_CHUNK},
        "methods": summ, "ratios": rat, "paired_legacy": T.paired(rows),
        "mc_error_share_JPC": {"certified_mean": j.get("mc_error_share_certified_mean"),
                               "certified_median": j.get("mc_error_share_certified_median"),
                               "all_mean": j.get("mc_error_share_mean"),
                               "note": "descriptive only (no longer a gate in round 1)"},
        "plugin_pick_error_rate": summ.get("PLUGIN", {}).get("fcr"),
        "pilot_checks": checks, "compute_unknown_rate_certifiers": cu_rate, "n_runs": n_runs,
        "projected_full_min": projected_full_min,
        "timing_projection": {"projected_full_total_min": projected_full_min, "workers": args.workers,
                              "full_instances": 12, "full_problems": 8},
        "projection_basis": "mean per-instance wall (all methods incl. 1e6-rollout truth) x 8/problems x 12 instances / workers",
        "go_no_go": "GO" if all(checks.values()) else "NO_GO",
        "caveats": ["INNER approximation: JPC certificate over a decision-directed sparse product grid inside the "
                    "continuous LR set; R_bar is not a guaranteed upper bound over Theta_t.",
                    "LR radius is the fixed-n Wilks value chi2_{32,0.96}/2 (not anytime-valid); B3 uses the same level.",
                    "Truth = 1e6-rollout MC estimates per policy (SE reported); no exact truth.",
                    "B3 ignores linearisation error and gradient MC noise (favourable to B3).",
                    "PLUGIN is a reference arm (point-estimate pick, no certificate); its 'FCR' is the plug-in error rate.",
                    "Wall-clock measured concurrently with up to 3 other tasks sharing CPU and the RTX 4090."],
        "wall_clock_total_s": wall, "errors": len(errors), "worker_max_mem_gb": peak,
        "per_instance": [{"seed": r["seed"], "wall_s": r["wall_s"], "method_wall_s": r["method_wall_s"],
                          "max_mem_gb": r["max_mem_gb"]} for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    try:
        name = torch.cuda.get_device_name(0)
        tot = torch.cuda.get_device_properties(0).total_memory / 2**20
    except Exception:
        name, tot = "unknown", 0
    (res_root / f"{task}_gpu_profile.json").write_text(json.dumps({
        "gpu_name": name, "vram_total_mb": round(tot), "max_batch_size": MC_CHUNK,
        "batch_unit": "MC episodes per GPU rollout call (stage 4/5 chunked)", "vram_used_mb": round(peak * 1024),
        "utilization_pct": round(100 * peak * 1024 / tot, 1) if tot else None,
        "note": "per-task cap 5 GB on a shared 4090 (project overlay); not saturating by design"}, indent=1))
    log(json.dumps({m: {k: v for k, v in d.items() if k in ("completion_rate", "rmst_new_env_steps_tau_step",
                                                             "compute_unknown_rate", "fcr", "n_false_cert",
                                                             "mc_error_share_certified_mean", "wall_clock_s_total")}
                    for m, d in summ.items()}))
    log(f"ratios={json.dumps({k: (v['per_problem_ratio_median'], v['cumulative_ratio_median']) for k, v in rat.items()})}")
    log(f"checks={checks} projected_full_min={projected_full_min} GO/NO-GO={summary['go_no_go']}")
    T.mark_done(res_root, "success" if not errors else "failed",
                f"E2 anchor {args.mode} {summary['go_no_go']}: JPC completion={j.get('completion_rate')} "
                f"peak={peak:.2f}GB projected_full={projected_full_min}")


if __name__ == "__main__":
    main()
