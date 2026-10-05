"""r4_g_eps_select: Gate G-eps (+ G-size precheck) on the CR9 dev half D_CR (methodology R1 / R4, s3).

QFC (pool-proportion schedule, exact enumeration, Q*-union L1, Lemma L2 variance UCB) on 100 dev permutation streams
(seeds 900-999) of CR9 visit, 15 budgets. The certificate value U_q(k) and the empirical answer pi_hat_q(k) at each of
the K = 20 checkpoints do not depend on eps, so every stream is replayed ONCE through all 20 checkpoints (QFC
'recorder' that never certifies inside the harness) and the sticky / stop-at-12-of-15 / N80 / censoring rule of
``frontier_runner.run_stream`` is then applied for each eps of the grid. Equivalence with a genuine ``run_stream``
call (QFCMethod, stopping) is checked on a subset of streams for every eps.

eps* = smallest grid value with (uncensored share of QFC N80 >= 0.7) AND (non-trivial problems >= 5/15 on the dev
guard table AND on the eval-half guard table; R4: non-trivial iff min over simple policies of true regret > eps).
No eps* in the grid -> G-size FAIL (recorded). The same rule is applied to HR8 visit (Hillstrom hs3 x category,
15 problems, grid {0.01, 0.0125, 0.015, 0.0175}) and eps*_HR8 is recorded only (no gate).

Only dev halves and dev seeds 900-999 are replayed; the eval half is touched only through the truth-side trivial guard
table (as prescribed). No baseline is run.

Usage: run_r4_g_eps_select.py --mode {pilot,full} [--workers 4] [--check-streams 8]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
TASK = "r4_g_eps_select"
DEV_SEEDS = list(range(900, 1000))
CR_GRID = (0.001, 0.0015, 0.002, 0.003)
HR8_GRID = (0.01, 0.0125, 0.015, 0.0175)
USHARE_MIN = 0.7
NONTRIV_MIN = 5
CODE_FILES = ["run_r4_g_eps_select.py", "dsswm/baselines/frontier_common.py", "dsswm/certify/quadknap.py",
              "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py", "dsswm/envs/pool_replay.py",
              "dsswm/theory_checks/mc_l1.py"]

from dsswm.baselines.frontier_common import QFCMethod  # noqa: E402
from dsswm.envs.pool_replay import DATA_PATHS, PoolReplayEnv, sha256_file  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

ENVS: dict = {}
CTX: dict = {}


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                       "final_progress": fp,
                                                       "timestamp": datetime.now().isoformat()}))


def problems_for(layer):
    return fr.cr_problems("visit") if layer.startswith("CR") else fr.hr_problems("visit")


# ------------------------------------------------------------------------------------------- QFC recorder
class QFCRecorder(QFCMethod):
    """QFC that records (pi_hat index, U) for every problem at every checkpoint and never certifies inside the
    harness, so the stream runs through all K checkpoints. U and pi_hat are eps-independent."""

    def __init__(self):
        super().__init__(name="QFC")
        self.hist = []

    def certify(self, ctx, st):
        res = super().certify(ctx, st)
        self.hist.append([(ih, U) for (_, ih, U) in res])
        return [(False, ih, U) for (_, ih, U) in res]


def derive(hist, ck, eps, stop_k, J_true, J_star, tau_R):
    """Apply run_stream's sticky / stop / N80 rule for one eps to the recorded history."""
    Q = len(J_star)
    undecided = np.ones(Q, dtype=bool)
    false_q = np.zeros(Q, dtype=bool)
    decided = np.full(Q, -1)
    cert_k = np.full(Q, -1)
    curve = []
    stop = None
    for k, rec in enumerate(hist):
        for q, (ih, U) in enumerate(rec):
            if U <= eps and undecided[q]:
                undecided[q] = False
                decided[q] = ih
                cert_k[q] = k
                false_q[q] = bool(J_star[q] - J_true[ih] > eps + 1e-12)
        n_cert = int((~undecided).sum())
        curve.append(n_cert)
        if stop is None and n_cert >= stop_k:
            stop = k
            break
    # certification curve without stopping (for the appendix plot): sticky over all recorded checkpoints
    und2 = np.ones(Q, dtype=bool)
    full_curve = []
    for rec in hist:
        for q, (ih, U) in enumerate(rec):
            if U <= eps:
                und2[q] = False
        full_curve.append(int((~und2).sum()))
    reached = stop is not None
    anyf = bool(false_q.any())
    n80 = int(ck[stop]) if (reached and not anyf) else int(tau_R)
    return {"eps": eps, "reached_stop": reached, "k_stop": stop, "N80": n80, "N80_frac": n80 / tau_R,
            "censored": bool(not reached or anyf),
            # methodology 1.2: reaching 12/15 only at tau_R (all pools exhausted -> exact, zero width) counts as
            # censored at tau_R; this strict flag is the one used by the gate
            "censored_strict": bool(not reached or anyf or n80 >= tau_R), "n_cert": int((~undecided).sum()), "n_false": int(false_q.sum()),
            "fwer_event": anyf, "cert_k": cert_k.tolist(), "decided_pi": decided.tolist(),
            "false_q": [int(q) for q in np.flatnonzero(false_q)], "full_curve": full_curve}


def stream_job(args):
    layer, seeds, grid = args
    env = ENVS[(layer, "dev")]
    ctx = CTX[layer]
    J_true, J_star = ctx.extra["J_true"], ctx.extra["J_star"]
    out = []
    for seed in seeds:
        try:
            m = QFCRecorder()
            t0 = time.perf_counter()
            summ, rows = run_stream(env, m, seed, ctx.problems, eps=grid[0], ctx=ctx, J_true=J_true, J_star=J_star)
            sec = time.perf_counter() - t0
            if len(m.hist) != len(ctx.checkpoints) or not summ["billing_ok"]:
                raise RuntimeError(f"incomplete replay: {len(m.hist)} checkpoints, billing_ok={summ['billing_ok']}")
            U = np.array([[u for (_, u) in rec] for rec in m.hist])            # (K, Q)
            per_eps = {str(e): derive(m.hist, ctx.checkpoints, e, ctx.stop_k, J_true, J_star, env.tau_R)
                       for e in grid}
            out.append({"layer": layer, "perm_seed": seed, "schedule_digest": summ["schedule_digest"],
                        "billing_ok": summ["billing_ok"], "sec_stream": round(sec, 2),
                        "sec_cert": summ["sec_cert"], "U": U.tolist(),
                        "pi_hat": [[int(ih) for (ih, _) in rec] for rec in m.hist], "per_eps": per_eps})
        except Exception as ex:  # noqa: BLE001
            out.append({"layer": layer, "perm_seed": seed, "error": repr(ex), "tb": traceback.format_exc()})
    return out


def check_job(args):
    """Genuine run_stream with a stopping QFCMethod for given (layer, seed, eps)."""
    layer, seed, eps = args
    env = ENVS[(layer, "dev")]
    base = CTX[layer]
    ctx = build_ctx(env, base.problems, eps, K=len(base.checkpoints))
    summ, _ = run_stream(env, QFCMethod(), seed, ctx.problems, eps=eps, ctx=ctx, J_true=base.extra["J_true"],
                         J_star=base.extra["J_star"], keep_U=False)
    return {"layer": layer, "perm_seed": seed, "eps": eps, "N80": summ["N80"], "censored": summ["censored"],
            "n_false": summ["n_false"], "cert_k": summ["cert_k"], "decided_pi": summ["decided_pi"],
            "k_stop": summ["k_stop"]}


# ------------------------------------------------------------------------------------------- guard
def guard_table(env, grid):
    probs = problems_for(env.layer)
    ans = env.truth_answers(probs, eps_grid=grid)
    mu, pooled = env.true_mu("visit"), env.true_pooled_mu("visit")
    rows = []
    for p, a in zip(probs, ans):
        if env.layer.startswith("CR"):
            triv = {"treat_by_w_desc": fr.trivial_all_arm(env.w, p, 1),
                    "pooled_uplift_sign": fr.trivial_pooled_greedy(env.w, pooled, p)}
        else:
            triv = {"pooled_greedy": fr.trivial_pooled_greedy(env.w, pooled, p),
                    "all_arm1": fr.trivial_all_arm(env.w, p, 1)}
        g = {"half": env.half, "qid": p.qid, "J_star": a.J_star, "second_gap": a.second_gap,
             "eps_set_size": {str(e): a.eps_set_size[e] for e in grid}}
        for name, pi in triv.items():
            g[f"regret_{name}"] = float(a.J_star - fr.policy_values(np.array([pi]), env.w, mu)[0])
        g["min_trivial_regret"] = min(g[f"regret_{n}"] for n in triv)
        rows.append(g)
    return rows, {str(e): int(sum(r["min_trivial_regret"] > e for r in rows)) for e in grid}


# ------------------------------------------------------------------------------------------- aggregate
def aggregate(layer, results, grid, guard_counts, tau_R, ck):
    ok = [r for r in results if "error" not in r]
    agg = {}
    for e in grid:
        pe = [r["per_eps"][str(e)] for r in ok]
        n80 = np.array([p["N80"] for p in pe], dtype=float)
        cens = np.array([p["censored_strict"] for p in pe])
        cens_loose = np.array([p["censored"] for p in pe])
        unc = n80[~cens]
        fw = np.array([p["fwer_event"] for p in pe])
        curves = np.array([p["full_curve"] for p in pe], dtype=float)
        g_dev, g_eval = guard_counts["dev"][str(e)], guard_counts["eval"][str(e)]
        a = {"n_streams": len(pe), "uncensored_share": float((~cens).mean()) if pe else float("nan"),
             "n_censored": int(cens.sum()), "n_stop_only_at_tauR": int((cens & ~cens_loose).sum()),
             "uncensored_share_harness_flag": float((~cens_loose).mean()) if pe else float("nan"), "n_never_reached_stop": int(sum(not p["reached_stop"] for p in pe)),
             "dev_fwer": float(fw.mean()) if pe else float("nan"), "n_fwer_events": int(fw.sum()),
             "n_false_cert_total": int(sum(p["n_false"] for p in pe)),
             "N80_median": float(np.median(n80)) if pe else None,
             "N80_median_frac_tauR": float(np.median(n80) / tau_R) if pe else None,
             "N80_uncensored_median": float(np.median(unc)) if unc.size else None,
             "N80_uncensored_geomean": float(np.exp(np.log(unc).mean())) if unc.size else None,
             "N80_uncensored_min": float(unc.min()) if unc.size else None,
             "N80_uncensored_max": float(unc.max()) if unc.size else None,
             "k_stop_hist": {str(k): int(v) for k, v in zip(*np.unique(
                 [p["k_stop"] if p["k_stop"] is not None else -1 for p in pe], return_counts=True))},
             "n_nontrivial_dev": g_dev, "n_nontrivial_eval": g_eval,
             "mean_certified_curve": curves.mean(0).tolist() if curves.size else [],
             "checkpoints": [int(x) for x in ck]}
        a["size_ok"] = bool(a["uncensored_share"] >= USHARE_MIN)
        a["guard_ok"] = bool(g_dev >= NONTRIV_MIN and g_eval >= NONTRIV_MIN)
        a["eligible"] = a["size_ok"] and a["guard_ok"]
        agg[str(e)] = a
    elig = [e for e in grid if agg[str(e)]["eligible"]]
    return agg, (min(elig) if elig else None)


def plot(aggs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (layer, agg, Q) in zip(axes, aggs):
        for e, a in agg.items():
            ck = np.array(a["checkpoints"])
            ax.plot(ck, np.array(a["mean_certified_curve"]) / Q, marker="o", ms=3, label=f"eps={e}")
        ax.axhline(0.8, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("arrivals")
        ax.set_ylabel(f"certified / {Q}")
        ax.set_title(f"QFC, {layer} visit, dev half (100 streams)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "certified_vs_arrivals.png"
    fig.savefig(p, dpi=130)
    return str(p)


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-streams", type=int, default=8)
    ap.add_argument("--n-streams", type=int, default=100)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    seeds = DEV_SEEDS[:args.n_streams]
    summary = {"task_id": TASK, "mode": args.mode, "started_at": datetime.now().isoformat(),
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)",
               "seeds": [seeds[0], seeds[-1]], "n_streams": len(seeds), "cr_grid": list(CR_GRID),
               "hr8_grid": list(HR8_GRID), "rule": "eps* = min eps in grid with uncensored_share >= 0.7 (N80 < tau_R and no false cert) AND "
               "n_nontrivial >= 5/15 on dev AND eval guard tables (min-over-simple-policies regret > eps)"}
    summary["code_sha256"] = {f: sha256_file(HERE / f) for f in CODE_FILES}
    summary["data_sha256"] = {"criteo_tidy_pkl": sha256_file(DATA_PATHS["criteo"]),
                              "hillstrom_tidy_pkl": sha256_file(DATA_PATHS["hillstrom"])}
    progress(0, 4, "envs")
    t = time.time()
    for layer in ("CR9", "HR8"):
        for half in ("dev", "eval"):
            ENVS[(layer, half)] = PoolReplayEnv(layer, half)
    summary["sec_build_envs"] = round(time.time() - t, 1)

    guards, guard_counts = {}, {}
    for layer, grid in (("CR9", CR_GRID), ("HR8", HR8_GRID)):
        env = ENVS[(layer, "dev")]
        ctx = build_ctx(env, problems_for(layer), grid[0])
        J_true = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
        ctx.extra["J_true"] = J_true
        ctx.extra["J_star"] = np.array([J_true[ctx.feas[q]].max() for q in range(ctx.Q)])
        CTX[layer] = ctx
        guard_counts[layer] = {}
        for half in ("dev", "eval"):
            rows, cnt = guard_table(ENVS[(layer, half)], grid)
            guards[f"{layer}|{half}"] = rows
            guard_counts[layer][half] = cnt
    summary["guard_counts"] = guard_counts
    summary["layer_info"] = {l: {"S": ENVS[(l, "dev")].S, "A": ENVS[(l, "dev")].A, "P": int(CTX[l].P),
                                 "tau_R_dev": int(ENVS[(l, "dev")].tau_R), "stop_k": CTX[l].stop_k,
                                 "checkpoints": [int(x) for x in CTX[l].checkpoints]}
                             for l in ("CR9", "HR8")}
    for l in ("CR9", "HR8"):
        m = QFCMethod()
        m.setup(CTX[l])
        summary["layer_info"][l]["params"] = m.params
    (out_dir / "guard_tables.json").write_text(json.dumps(guards, indent=1))

    progress(1, 4, "streams", {"n_streams": len(seeds)})
    jobs = [("CR9", seeds[i::args.workers], CR_GRID) for i in range(args.workers)]
    jobs += [("HR8", seeds[i::args.workers], HR8_GRID) for i in range(args.workers)]
    t = time.time()
    results = {"CR9": [], "HR8": []}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(stream_job, jobs):
            for r in res:
                results[r["layer"]].append(r)
    summary["sec_streams"] = round(time.time() - t, 1)
    errs = [r for l in results for r in results[l] if "error" in r]
    summary["n_errors"] = len(errs)
    if errs:
        summary["errors_sample"] = errs[:3]
    with open(out_dir / "results.jsonl", "w") as f:
        for l in ("CR9", "HR8"):
            for r in sorted(results[l], key=lambda r: r["perm_seed"]):
                f.write(json.dumps(r) + "\n")

    # ---- equivalence check vs genuine stopping run_stream
    progress(2, 4, "equivalence_check")
    chk_jobs = [(l, s, e) for l, grid in (("CR9", CR_GRID), ("HR8", HR8_GRID))
                for s in seeds[:args.check_streams] for e in grid]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        chk = list(ex.map(check_job, chk_jobs))
    byk = {(r["layer"], r["perm_seed"]): r for l in results for r in results[l] if "error" not in r}
    mism = []
    for c in chk:
        d = byk[(c["layer"], c["perm_seed"])]["per_eps"][str(c["eps"])]
        same = (d["N80"] == c["N80"] and d["censored"] == c["censored"] and d["n_false"] == c["n_false"] and
                d["cert_k"] == c["cert_k"] and d["decided_pi"] == c["decided_pi"])
        if not same:
            mism.append({"check": c, "derived": {k: d[k] for k in ("N80", "censored", "n_false", "cert_k")}})
    summary["equivalence_check"] = {"n_checked": len(chk), "n_mismatch": len(mism), "mismatches": mism[:5]}

    # ---- aggregate
    progress(3, 4, "aggregate")
    aggs, eps_star = {}, {}
    for l, grid in (("CR9", CR_GRID), ("HR8", HR8_GRID)):
        agg, es = aggregate(l, results[l], grid, guard_counts[l], ENVS[(l, "dev")].tau_R, CTX[l].checkpoints)
        aggs[l], eps_star[l] = agg, es
    summary["per_eps"] = aggs
    summary["eps_star"] = eps_star
    summary["plot"] = plot([("CR9", aggs["CR9"], CTX["CR9"].Q), ("HR8", aggs["HR8"], CTX["HR8"].Q)], out_dir)
    # samples: 5 CR9 streams at eps*, U trajectories + answers
    es = eps_star["CR9"] if eps_star["CR9"] is not None else CR_GRID[-1]
    samp = []
    for r in sorted(results["CR9"], key=lambda r: r["perm_seed"])[:5]:
        if "error" in r:
            continue
        pe = r["per_eps"][str(es)]
        samp.append({"perm_seed": r["perm_seed"], "eps": es, "N80": pe["N80"], "k_stop": pe["k_stop"],
                     "cert_k": pe["cert_k"], "false_q": pe["false_q"],
                     "decided_pi": [CTX["CR9"].pols[i].tolist() if i >= 0 else None for i in pe["decided_pi"]],
                     "pi_star": [CTX["CR9"].pols[int(np.argmax(np.where(CTX["CR9"].feas[q],
                                                                         CTX["CR9"].extra["J_true"], -np.inf)))].tolist()
                                 for q in range(CTX["CR9"].Q)],
                     "U_at_stop": r["U"][pe["k_stop"]] if pe["k_stop"] is not None else None})
    (out_dir / "samples" / "cr9_streams_at_eps_star.json").write_text(json.dumps(samp, indent=1))

    gate_pass = bool(eps_star["CR9"] is not None and summary["n_errors"] == 0 and
                     summary["equivalence_check"]["n_mismatch"] == 0)
    summary["G_size_precheck"] = "PASS" if eps_star["CR9"] is not None else "FAIL"
    summary["pass"] = gate_pass
    summary["wall_min"] = round((time.time() - t_start) / 60, 2)
    summary["finished_at"] = datetime.now().isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    gates_dir = RES_ROOT / "r4_gates"
    gates_dir.mkdir(exist_ok=True)
    c = aggs["CR9"]
    eps_json = {
        "task_id": TASK, "written_at": datetime.now().isoformat(), "mode": args.mode,
        "layer": "CR9", "outcome": "visit", "half": "dev", "dev_seeds": "900-999", "n_streams": len(seeds),
        "method": "QFC (pool proportions, exact enumeration, Q*-union L1, Lemma L2 variance UCB)",
        "grid": list(CR_GRID), "rule": summary["rule"],
        "eps_star": eps_star["CR9"], "G_eps": "PASS" if eps_star["CR9"] is not None else "FAIL",
        "G_size_precheck": summary["G_size_precheck"],
        "G_size_note": "G-size is formally judged in r4_g1_shape at eps*; this is the QFC-only precheck on the "
                       "same dev streams",
        "per_eps": {e: {k: c[e][k] for k in ("uncensored_share", "n_stop_only_at_tauR", "uncensored_share_harness_flag", "dev_fwer", "n_fwer_events",
                                             "N80_uncensored_median", "N80_uncensored_geomean",
                                             "N80_median_frac_tauR", "n_nontrivial_dev", "n_nontrivial_eval",
                                             "size_ok", "guard_ok", "eligible")} for e in c},
        "tau_R_dev": int(ENVS[("CR9", "dev")].tau_R),
        "HR8_record_only": {"grid": list(HR8_GRID), "eps_star_HR8": eps_star["HR8"],
                            "per_eps": {e: {k: aggs["HR8"][e][k] for k in (
                                "uncensored_share", "dev_fwer", "N80_uncensored_median", "n_nontrivial_dev",
                                "n_nontrivial_eval", "eligible")} for e in aggs["HR8"]},
                            "tau_R_dev": int(ENVS[("HR8", "dev")].tau_R)},
        "integrity": {"n_errors": summary["n_errors"],
                      "equivalence_check_mismatch": summary["equivalence_check"]["n_mismatch"],
                      "eval_seeds_touched": False, "baselines_run": False},
        "code_sha256": summary["code_sha256"]}
    (gates_dir / "eps.json").write_text(json.dumps(eps_json, indent=1, default=str))
    mark_done("success" if gate_pass else "failed",
              f"eps*_CR9={eps_star['CR9']} eps*_HR8={eps_star['HR8']} G-size={summary['G_size_precheck']}")
    print(json.dumps({"eps_star": eps_star, "pass": gate_pass, "wall_min": summary["wall_min"]}))


if __name__ == "__main__":
    main()
