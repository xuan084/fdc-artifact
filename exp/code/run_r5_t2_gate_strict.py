"""r5_t2_gate_strict (PILOT == FULL for this dev gate): final FDC vs every rigorous rival in R on CR9 dev streams.

Setting: CR9 dev half, eps* = 0.001 (r4_gates/eps.json), K = 20 checkpoints, Q = 15, stop at 12/15, delta = 0.05,
dev seeds 900-999 (100 paired streams). Methods: FDC (dsswm/baselines/fdc.py, final) + R = {B1, B4, B4-bal, B2-rect,
B3-rect, Peace-rect, Hait-SW, Molitor-WoR, QFC-pool}. Tunable rivals use the frozen configs in
exp/results/r5_gates/rival_configs.json; everything else uses the r4 frozen configs (registry defaults).

Primary endpoint N80_pen (methodology s5): checkpoint at which >= 12/15 problems are certified; tau_R if never reached
or if any certificate issued up to that point is wrong. N80_raw = stopping checkpoint without the error penalty.

Gate T2: for every r in R, one-sided 95% paired-bootstrap upper bound (B = 1e4, seed 42, resampling streams) of
geomean(N80_pen FDC / N80_pen r) < 1.0.
Gate T3 (stop window): FDC false streams <= 1/100, every false stream audited (audit/), billing_ok on all streams.
Consistency: QFC-pool, B1, B4 (r4_g1_shape_a), B2-rect, B3-rect (r4_g1_shape_b), Peace-rect (r4_g1_shape_c) N80 per
stream vs the stored r4 results.
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run). No evaluation seed is touched.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
TASK = "r5_t2_gate_strict"
RES = WS / "exp/results"
OUT = RES / "pilots" / TASK
AUDIT = OUT / "audit"
GATES = RES / "r5_gates"
SEEDS = list(range(900, 1000))
if os.environ.get("T2_SMOKE"):  # pipeline smoke test only: 3 dev seeds, output to a scratch dir, no markers
    SEEDS = [900, 901, 902]
    TASK = "r5_t2_gate_strict_SMOKE"
    OUT = Path(os.environ["T2_SMOKE"])
    AUDIT = OUT / "audit"
B_BOOT = 10_000
BOOT_SEED = 42

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

RIVALS = list(reg.RIGOROUS_SET_R)
METHODS = ["FDC"] + RIVALS
# r4 stored results used for the per-stream consistency check (method name in the r4 file, file)
R4_REF = {"QFC-pool": ("QFC", "r4_g1_shape_a/streams.jsonl"), "B1": ("B1", "r4_g1_shape_a/streams.jsonl"),
          "B4": ("B4", "r4_g1_shape_a/streams.jsonl"), "B2-rect": ("B2-rect", "r4_g1_shape_b/results.jsonl"),
          "B3-rect": ("B3-rect", "r4_g1_shape_b/results.jsonl"),
          "Peace-rect": ("Peace-rect", "r4_g1_shape_c/results.jsonl")}
FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/clucb_joint.py",
         "dsswm/baselines/uniform_rs.py", "dsswm/baselines/b4_bal.py", "dsswm/baselines/combgame_joint.py",
         "dsswm/baselines/b2_rect_fe.py", "dsswm/baselines/rage_frontier.py", "dsswm/baselines/peace_frontier.py",
         "dsswm/baselines/hait_sw.py", "dsswm/baselines/molitor_wor.py", "dsswm/streams/r5_registry.py",
         "dsswm/streams/frontier_runner.py", "dsswm/certify/quadknap.py", "dsswm/envs/pool_replay.py",
         "run_r5_t2_gate_strict.py"]
_ENV = {}


def sha(p):
    return hashlib.sha256((CODE / p).read_bytes()).hexdigest()[:16]


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def progress(done, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def frozen_kwargs():
    rc = json.loads((GATES / "rival_configs.json").read_text())
    return {m: dict(rc["frozen_configs"].get(m) or {}) for m in RIVALS}, rc


def make(name, kw):
    return reg.make_method(name, **kw)


def init_env():
    from dsswm.envs.pool_replay import PoolReplayEnv
    eps = float(json.loads((RES / "r4_gates" / "eps.json").read_text())["eps_star"])
    env = PoolReplayEnv("CR9", "dev")
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    _ENV.update(env=env, ctx=ctx, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]), eps=eps)


def job(a):
    name, kw, seed = a
    ctx = _ENV["ctx"]
    try:
        m = make(name, kw)
        t0 = time.perf_counter()
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=_ENV["J"],
                             J_star=_ENV["Js"], keep_U=False)
        ck = ctx.checkpoints
        n80_raw = int(ck[s["k_stop"]]) if s["reached_stop"] else int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))  # stream stops at k_stop: carry forward
        return {"method": name, "params": kw, "seed": seed, "validity": m.validity, "alloc_kind": m.alloc_kind,
                "N80_pen": int(s["N80"]), "N80_raw": n80_raw, "N80_over_tau": s["N80"] / ctx.tau_R,
                "completed": bool(s["reached_stop"]), "fwer_event": bool(s["fwer_event"]),
                "censored": bool(s["censored"]), "n_cert": s["n_cert"], "n_false": s["n_false"],
                "false_q": s["false_q"], "k_stop": s["k_stop"], "cert_k": s["cert_k"], "decided_pi": s["decided_pi"],
                "billing": {"billed": s["billed"], "served": s["served"], "skipped": s["skipped"],
                            "reselected": s["reselected"], "loop_arrivals": s["loop_arrivals"], "t_end": s["t_end"]},
                "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"],
                "n_cert_curve": curve, "sec": round(time.perf_counter() - t0, 3), "error": None}
    except Exception:  # noqa: BLE001
        return {"method": name, "params": kw, "seed": seed, "error": traceback.format_exc()}


def audit_job(a):
    """Re-run one false stream with U kept; record every certificate and its true gap."""
    name, kw, seed = a
    ctx, J, Js = _ENV["ctx"], _ENV["J"], _ENV["Js"]
    m = make(name, kw)
    s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=True)
    certs = []
    for q in range(ctx.Q):
        if s["cert_k"][q] < 0:
            continue
        pi = s["decided_pi"][q]
        gap = float(Js[q] - J[pi])
        certs.append({"q": q, "cert_k": s["cert_k"][q], "t": int(ctx.checkpoints[s["cert_k"][q]]), "decided_pi": pi,
                      "J_star": float(Js[q]), "J_decided": float(J[pi]), "gap": gap, "eps": ctx.eps,
                      "false": bool(gap > ctx.eps + 1e-12),
                      "U_at_cert": rows[s["cert_k"][q]]["U"][q] if s["cert_k"][q] < len(rows) else None})
    return {"method": name, "seed": seed, "summary": s, "certificates": certs,
            "rows": [{k: r[k] for k in ("k", "t", "n_cert", "n_false", "new", "new_false", "n_cells", "U")}
                     for r in rows]}


def paired_stats(fdc, riv, idx_boot):
    d = np.log(fdc) - np.log(riv)
    boot = d[idx_boot].mean(1)
    return {"geomean_ratio": float(np.exp(d.mean())),
            "ub95_one_sided": float(np.exp(np.quantile(boot, 0.95))),
            "ci95_two_sided": [float(np.exp(np.quantile(boot, 0.025))), float(np.exp(np.quantile(boot, 0.975)))],
            "frac_faster": float((d < 0).mean()), "frac_tied": float((d == 0).mean()),
            "frac_slower": float((d > 0).mean()), "mean_log_ratio": float(d.mean()),
            "sd_log_ratio": float(d.std(ddof=1)), "n": int(len(d))}


def git_info():
    def g(*a):
        p = subprocess.run(["git", *a], cwd=str(CODE), capture_output=True, text=True)
        return p.stdout.strip()
    return {"head": g("rev-parse", "HEAD"), "dirty_files": g("status", "--porcelain", "--", ".").splitlines()[:50]}


def plot(rows, ck, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(METHODS):
        C = np.array([r["n_cert_curve"] for r in rows if r["method"] == m], dtype=float)
        med = np.median(C, 0)
        lo, hi = np.quantile(C, 0.25, 0), np.quantile(C, 0.75, 0)
        lw = 2.4 if m == "FDC" else 1.1
        ax.step(ck, med, where="post", color="black" if m == "FDC" else cmap(i % 10), lw=lw, label=m)
        if m == "FDC":
            ax.fill_between(ck, lo, hi, step="post", color="black", alpha=0.12)
    ax.axhline(12, ls=":", color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("arrivals (log scale)")
    ax.set_ylabel("problems certified (median over 100 dev streams)")
    ax.set_title("CR9 dev, eps*=0.001: certified problems vs arrivals (FDC IQR shaded; curves frozen after stop)",
                 fontsize=8.5)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = datetime.now()
    (OUT / "start_time.txt").write_text(t_start.isoformat())
    kws, rc = frozen_kwargs()
    kws["FDC"] = {}
    init_env()
    ctx = _ENV["ctx"]
    assert abs(ctx.eps - 0.001) < 1e-12 and len(ctx.checkpoints) == 20 and ctx.stop_k == 12 and ctx.Q == 15
    tau = int(ctx.tau_R)
    print(f"tau_R={tau} K={len(ctx.checkpoints)} stop_k={ctx.stop_k} Q={ctx.Q} eps={ctx.eps}", flush=True)
    describe = {}
    for m in METHODS:
        inst = make(m, kws[m])
        inst.setup(ctx)
        try:
            describe[m] = json.loads(json.dumps(inst.describe(), default=str))
        except Exception as e:  # noqa: BLE001
            describe[m] = {"name": m, "describe_error": repr(e)}
        describe[m]["registry"] = {"validity_class": reg.spec(m).validity_class, "role": reg.spec(m).role,
                                   "source": reg.spec(m).source, "params": kws[m]}
    # slow methods first so that the pool tail is short
    order = ["Peace-rect", "B3-rect"] + [m for m in METHODS if m not in ("Peace-rect", "B3-rect")]
    jobs = [(m, kws[m], s) for m in order for s in SEEDS]
    total = len(jobs)
    progress(0, total)
    rows, errs = [], []
    rfile = OUT / "results.jsonl"
    rfile.write_text("")
    with get_context("fork").Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            if r.get("error"):
                errs.append(r)
            else:
                rows.append(r)
                with open(rfile, "a") as f:
                    f.write(json.dumps(r) + "\n")
            if (i + 1) % 20 == 0 or i + 1 == total:
                progress(i + 1, total, {"runs_done": i + 1, "errors": len(errs)})
                print(f"[{i+1}/{total}] last={r['method']} seed={r['seed']} errors={len(errs)}", flush=True)
    if errs:
        (OUT / "errors.log").write_text("\n\n".join(f"{e['method']} {e['seed']}\n{e['error']}" for e in errs))
        raise RuntimeError(f"{len(errs)} stream jobs failed (see errors.log)")

    by = {(r["method"], r["seed"]): r for r in rows}
    N = {m: np.array([by[(m, s)]["N80_pen"] for s in SEEDS], dtype=float) for m in METHODS}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(SEEDS), size=(B_BOOT, len(SEEDS)))  # same resamples for every rival (paired by stream)
    per_method = {}
    for m in METHODS:
        rr = [by[(m, s)] for s in SEEDS]
        fs = sum(x["fwer_event"] for x in rr)
        logs = np.log(N[m])
        per_method[m] = {"validity": rr[0]["validity"], "params": kws[m], "geomean_N80_pen": float(np.exp(logs.mean())),
                         "geomean_N80_over_tau": float(np.exp(logs.mean()) / tau),
                         "geomean_N80_raw": float(np.exp(np.log([x["N80_raw"] for x in rr]).mean())),
                         "median_N80_pen": float(np.median(N[m])), "completed": int(sum(x["completed"] for x in rr)),
                         "censored": int(sum(x["censored"] for x in rr)), "false_streams": int(fs),
                         "fwer_cp_upper": round(cp_upper(fs, len(rr)), 4),
                         "false_certs": int(sum(x["n_false"] for x in rr)),
                         "billing_ok_all": all(x["billing_ok"] for x in rr),
                         "sec_per_stream_mean": float(np.mean([x["sec"] for x in rr])),
                         "n80_checkpoint_hist": {str(int(k)): int(v) for k, v in
                                                 zip(*np.unique(N[m].astype(int), return_counts=True))}}
    paired = {r: paired_stats(N["FDC"], N[r], idx) for r in RIVALS}
    for r in RIVALS:
        paired[r]["T2_pass"] = paired[r]["ub95_one_sided"] < 1.0
    with open(OUT / "paired_ratios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rival", "params", "geomean_ratio_FDC_over_r", "ci95_lo", "ci95_hi", "ub95_one_sided",
                    "frac_faster", "frac_tied", "frac_slower", "sd_log_ratio", "T2_pass", "rival_false_streams"])
        for r in RIVALS:
            p = paired[r]
            w.writerow([r, json.dumps(kws[r]), p["geomean_ratio"], p["ci95_two_sided"][0], p["ci95_two_sided"][1],
                        p["ub95_one_sided"], p["frac_faster"], p["frac_tied"], p["frac_slower"], p["sd_log_ratio"],
                        p["T2_pass"], per_method[r]["false_streams"]])
    with open(OUT / "per_stream_n80.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + METHODS)
        for i, s in enumerate(SEEDS):
            w.writerow([s] + [int(N[m][i]) for m in METHODS])

    # T3: audit every false stream (FDC mandatory; rivals too, for the record)
    false_jobs = [(m, kws[m], s) for m in METHODS for s in SEEDS if by[(m, s)]["fwer_event"]]
    audits = []
    if false_jobs:
        with get_context("fork").Pool(min(4, len(false_jobs))) as pool:
            audits = pool.map(audit_job, false_jobs)
        for a in audits:
            (AUDIT / f"{a['method']}_seed{a['seed']}.json").write_text(json.dumps(a, indent=1, default=str))
    (AUDIT / "index.json").write_text(json.dumps({
        "false_streams": [{"method": a["method"], "seed": a["seed"],
                           "false_certs": [c for c in a["certificates"] if c["false"]]} for a in audits],
        "note": "every stream with a wrong certificate at or before the stop checkpoint (FDC and rivals)"}, indent=1))
    fdc_false = per_method["FDC"]["false_streams"]

    # consistency with stored r4 results
    cons = {}
    for m, (r4name, rel) in R4_REF.items():
        stored = {}
        for l in (RES / "pilots" / rel).read_text().splitlines():
            x = json.loads(l)
            if x.get("method") == r4name and x.get("layer", "CR9") == "CR9" and "N80" in x:
                stored[x["perm_seed"]] = x
        mism = [{"seed": s, "r5": by[(m, s)]["N80_pen"], "r4": stored[s]["N80"]}
                for s in SEEDS if s in stored and by[(m, s)]["N80_pen"] != stored[s]["N80"]]
        dig = sum(1 for s in SEEDS if s in stored and by[(m, s)]["schedule_digest"] != stored[s]["schedule_digest"])
        cons[m] = {"r4_method": r4name, "source": f"exp/results/pilots/{rel}", "n_compared": sum(s in stored for s in SEEDS),
                   "n80_mismatch": len(mism), "digest_mismatch": dig, "mismatches": mism[:20]}

    ck = np.asarray(ctx.checkpoints)
    plot(rows, ck, OUT / "certified_vs_arrivals.png")
    curves = {m: {"median": np.median([by[(m, s)]["n_cert_curve"] for s in SEEDS], 0).tolist(),
                  "q25": np.quantile([by[(m, s)]["n_cert_curve"] for s in SEEDS], 0.25, 0).tolist(),
                  "q75": np.quantile([by[(m, s)]["n_cert_curve"] for s in SEEDS], 0.75, 0).tolist()} for m in METHODS}
    (OUT / "curves.json").write_text(json.dumps({"checkpoints": ck.tolist(), "curves": curves}, indent=1))
    samples = [{k: by[(m, s)][k] for k in ("method", "seed", "N80_pen", "N80_raw", "k_stop", "cert_k", "decided_pi",
                                          "n_false", "billing_ok")} for s in (SEEDS[0], SEEDS[len(SEEDS) // 2], SEEDS[-1]) for m in METHODS]
    (OUT / "samples.json").write_text(json.dumps(samples, indent=1))
    configs = {"setting": {"layer": "CR9", "half": "dev", "eps": ctx.eps, "K": len(ck), "stop_k": int(ctx.stop_k),
                           "Q": ctx.Q, "delta": ctx.delta, "tau_R": tau, "seeds": [SEEDS[0], SEEDS[-1]],
                           "n_streams": len(SEEDS), "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                                                                  "unit": "stream (paired; same resamples for all r)"}},
               "methods": describe, "rival_configs_source": "exp/results/r5_gates/rival_configs.json",
               "rival_configs_frozen_at": rc.get("frozen_at"), "git": git_info(),
               "code_sha256_16": {p: sha(p) for p in FILES if (CODE / p).exists()}}
    (OUT / "configs.json").write_text(json.dumps(configs, indent=1, default=str))

    t2_all = all(paired[r]["T2_pass"] for r in RIVALS)
    bill = all(per_method[m]["billing_ok_all"] for m in METHODS)
    t3 = fdc_false <= 1 and all(any(a["method"] == "FDC" and a["seed"] == s for a in audits)
                                for s in SEEDS if by[("FDC", s)]["fwer_event"])
    t_end = datetime.now()
    summary = {"task_id": TASK, "mode": "pilot (== full: dev gate, same streams)", "started_at": t_start.isoformat(),
               "ended_at": t_end.isoformat(), "wall_min": round((t_end - t_start).total_seconds() / 60, 1),
               "timing_note": "concurrent run (4 workers; other r5 tasks share the 20-core host)",
               "setting": configs["setting"],
               "gate_T2": {"rule": "for every r in R: one-sided 95% paired-bootstrap UB of geomean(N80_pen FDC/r) < 1.0",
                           "pass": t2_all, "failing_rivals": [r for r in RIVALS if not paired[r]["T2_pass"]],
                           "per_rival": paired},
               "gate_T3_stop": {"rule": "FDC false streams <= 1/100, each audited", "fdc_false_streams": fdc_false,
                                "fdc_cp_upper": per_method["FDC"]["fwer_cp_upper"], "pass": t3,
                                "audit_dir": os.path.relpath(AUDIT, WS),
                                "all_false_streams": [(a["method"], a["seed"]) for a in audits]},
               "billing_ok_all": bill, "per_method": per_method, "consistency_vs_r4": cons,
               "go_no_go": "GO" if (t2_all and t3 and bill) else "NO_GO",
               "eval_seeds_touched": False, "outputs": ["summary.json", "results.jsonl", "paired_ratios.csv",
                                                        "per_stream_n80.csv", "configs.json", "curves.json",
                                                        "samples.json", "audit/", "certified_vs_arrivals.png"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def mark_done(status, txt):
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
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


if __name__ == "__main__":
    try:
        s = main()
        g = s["gate_T2"]
        txt = (f"{s['go_no_go']} | T2 pass={g['pass']} failing={g['failing_rivals']} | "
               f"T3 FDC false={s['gate_T3_stop']['fdc_false_streams']}/100")
        mark_done("success", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
