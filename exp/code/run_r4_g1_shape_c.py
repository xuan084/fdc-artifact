"""r4_g1_shape_c: Gate G1-shape part C (lifeline 1) -- QFC vs Peace (Katz-Samuels et al. 2020), variants
rect / nominal / fav (Gaussian width with 2000 eta draws, exact LMO over the 512 enumerated CR9 policies), on the same
100 CR9 dev streams (seeds 900-999) and eps* as r4_g1_shape_a / r4_g1_shape_b.

Methodology 'Pilot 后锁前修订' R1/R3 (overrides s1.5). Peace may NOT be removed from the opponent set.
Primary comparison variant per opponent (opponent-favourable rule written into the lock): in order fav -> nominal ->
rect, the first variant satisfying S1 AND S2 on the block (S1: one-sided exact CP upper of the stream FWER <= delta;
S2: one-sided 95% instance-cluster bootstrap upper of FCR_old <= delta). On the dev block this is a REFERENCE choice
only; the confirmatory choice is re-made on the evaluation block.

Gate (recorded; verdict in r4_gate_decision): paired geometric-mean N80 ratio QFC / X_primary <= 0.67 for X = Peace.
N80 = first checkpoint with >= 12/15 certified and no eps-wrong certification so far; otherwise tau_R (censored).

QFC is re-run here with exactly the r4_g_eps_select / r4_g1_shape_a code path (QFCMethod, genuine stopping
run_stream) on the same streams -- deterministic, so it must match part A stream by stream; the match is checked
against part A's results.jsonl if it exists when this task finishes (recorded, not required).

Only the dev half and dev seeds 900-999 are touched. Truth (J_true) is used only for scoring.

Usage: run_r4_g1_shape_c.py --mode {pilot,full} [--workers 4] [--n-streams 100]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
TASK = "r4_g1_shape_c"
DEV_SEEDS = list(range(900, 1000))
GATE_RATIO = 0.67
DELTA = 0.05
OPPONENTS = {"Peace": ("Peace-fav", "Peace-nominal", "Peace-rect")}
METHODS = ["QFC"] + [m for v in OPPONENTS.values() for m in v]
CODE_FILES = ["run_r4_g1_shape_c.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/peace_frontier.py",
              "dsswm/baselines/rage_frontier.py", "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py",
              "dsswm/envs/pool_replay.py", "dsswm/stats/acceptance_r4.py"]

from dsswm.baselines.combgame_joint import CombGameJoint  # noqa: E402
from dsswm.baselines.frontier_common import QFCMethod  # noqa: E402
from dsswm.envs.pool_replay import DATA_PATHS, PoolReplayEnv, sha256_file  # noqa: E402
from dsswm.stats import acceptance_r4 as acc  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, make_frontier_method, run_stream, true_policy_values  # noqa: E402

G: dict = {}


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


def make_method(name, env):
    if name == "QFC":
        return QFCMethod()
    if name.startswith("B2-"):
        return CombGameJoint(name[3:])
    return make_frontier_method(name, env)


def job(args):
    name, seed = args
    env, ctx = G["env"], G["ctx"]
    try:
        m = make_method(name, env)
        t0 = time.perf_counter()
        summ, rows = run_stream(env, m, seed, ctx.problems, eps=ctx.eps, ctx=ctx, J_true=G["J_true"],
                                J_star=G["J_star"], keep_U=True)
        summ["sec_wall"] = round(time.perf_counter() - t0, 3)
        summ["curve"] = [r["n_cert"] for r in rows]
        summ["U_last"] = rows[-1].get("U") if rows else None
        summ["n_cells_last"] = rows[-1]["n_cells"] if rows else None
        gl = getattr(m, "gw_log", None)
        if gl:
            summ["gw_n_rounds"] = len(gl)
            summ["gw_mc_rel_se_max"] = float(max(g["W_mc_rel_se"] for g in gl))
            if any("tau_alg2_mc_rel_se" in g for g in gl):
                summ["tau_alg2_mc_rel_se_max"] = float(max(g.get("tau_alg2_mc_rel_se", 0.0) for g in gl))
            summ["gw_log_tail"] = gl[-5:]
        return summ
    except Exception as ex:  # noqa: BLE001
        return {"method": name, "perm_seed": seed, "error": repr(ex), "tb": traceback.format_exc()}


def method_stats(rs, tau_R):
    n80 = np.array([r["N80"] for r in rs], float)
    cens = np.array([r["censored"] or r["N80"] >= tau_R for r in rs])
    nf = np.array([r["n_false"] for r in rs], float)
    nc = np.array([r["n_cert"] for r in rs], float)
    saf = acc.layer_safety(nf, nc, DELTA)
    return {"n": len(rs), "validity": rs[0]["validity"],
            "N80_geomean": float(np.exp(np.log(n80).mean())), "N80_median": float(np.median(n80)),
            "N80_q25": float(np.quantile(n80, 0.25)), "N80_q75": float(np.quantile(n80, 0.75)),
            "N80_median_frac_tauR": float(np.median(n80) / tau_R),
            "uncensored_share": float((~cens).mean()), "n_censored": int(cens.sum()),
            "n_reached_stop": int(sum(r["reached_stop"] for r in rs)),
            "n_cert_mean": float(nc.mean()), "fwer_k": saf["S1"]["k"], "fwer": saf["S1"]["rate"],
            "fwer_cp_upper": saf["S1"]["cp_upper"], "fcr_old": saf["S2"]["fcr_old"],
            "fcr_old_upper": saf["S2"]["upper"], "E_V_over_R": saf["E_V_over_R"], "S1": saf["S1"]["pass"],
            "S2": saf["S2"]["pass"], "safe_S1_and_S2": saf["safe"],
            "billing_ok_all": bool(all(r["billing_ok"] for r in rs)),
            "skipped_total": int(sum(r["skipped"] for r in rs)),
            "reselected_total": int(sum(r["reselected"] for r in rs)),
            "sec_per_stream_mean": float(np.mean([r["sec_wall"] for r in rs])),
            "sec_per_stream_max": float(np.max([r["sec_wall"] for r in rs]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-streams", type=int, default=100)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    eps_json = json.loads((RES_ROOT / "r4_gates" / "eps.json").read_text())
    eps = float(eps_json["eps_star"])
    seeds = DEV_SEEDS[:args.n_streams]
    summary = {"task_id": TASK, "mode": args.mode, "started_at": datetime.now().isoformat(),
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)",
               "layer": "CR9", "outcome": "visit", "half": "dev", "seeds": [seeds[0], seeds[-1]],
               "n_streams": len(seeds), "eps_star": eps, "eps_source": "exp/results/r4_gates/eps.json",
               "methods": METHODS, "gate_ratio": GATE_RATIO,
               "primary_variant_rule": "per opponent, in order fav -> nominal -> rect, the first variant with S1 AND S2 "
                                       "on the block (dev = reference only; confirmatory = eval block)",
               "note_variants": "R3: Peace-{rect,nominal,fav}; B2/B3 in r4_g1_shape_b; QFC re-run (deterministic) and checked vs parts A/B"}
    summary["code_sha256"] = {f: sha256_file(HERE / f) for f in CODE_FILES}
    summary["data_sha256"] = {"criteo_tidy_pkl": sha256_file(DATA_PATHS["criteo"])}
    progress(0, 4, "env")
    env = PoolReplayEnv("CR9", "dev")
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J_true = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    G.update(env=env, ctx=ctx, J_true=J_true, J_star=np.array([J_true[ctx.feas[q]].max() for q in range(ctx.Q)]))
    tau_R = int(env.tau_R)
    summary["layer_info"] = {"S": env.S, "A": env.A, "P": int(ctx.P), "Q": int(ctx.Q), "tau_R_dev": tau_R,
                             "stop_k": ctx.stop_k, "replan": int(ctx.replan),
                             "checkpoints": [int(x) for x in ctx.checkpoints]}

    progress(1, 4, "streams", {"n_jobs": len(METHODS) * len(seeds)})
    # slow-ish methods first for load balance
    jobs = [(m, s) for m in METHODS[::-1] for s in seeds]
    results = {m: [] for m in METHODS}
    t = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results[r["method"]].append(r)
            done += 1
            if done % 25 == 0:
                progress(1, 4, "streams", {"done": done, "n_jobs": len(jobs)})
    summary["sec_streams"] = round(time.time() - t, 1)
    errs = [r for m in METHODS for r in results[m] if "error" in r]
    summary["n_errors"] = len(errs)
    if errs:
        summary["errors_sample"] = errs[:3]
    with open(out_dir / "results.jsonl", "w") as f:
        for m in METHODS:
            for r in sorted(results[m], key=lambda r: r["perm_seed"]):
                f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")

    progress(2, 4, "aggregate")
    ok = {m: sorted([r for r in results[m] if "error" not in r], key=lambda r: r["perm_seed"]) for m in METHODS}
    stats = {m: method_stats(ok[m], tau_R) for m in METHODS if ok[m]}
    summary["per_method"] = stats
    qfc_n80 = {r["perm_seed"]: r["N80"] for r in ok["QFC"]}

    def ratio(mname):
        common = [r["perm_seed"] for r in ok[mname] if r["perm_seed"] in qfc_n80]
        a = np.array([qfc_n80[s] for s in common], float)
        b = np.array([r["N80"] for r in ok[mname] if r["perm_seed"] in qfc_n80], float)
        lr = acc.paired_log_ratio(a, b, tau=tau_R)
        return {"n_paired": len(common), "geomean_ratio_QFC_over_X": float(math.exp(lr["mean"])),
                "ci95": [float(math.exp(lr["ci_lo"])), float(math.exp(lr["ci_hi"]))],
                "mean_log_ratio": lr["mean"], "p_one_sided_vs_log0.8": lr["p_one_sided"],
                "frac_streams_QFC_faster": float(np.mean(a < b)), "frac_ties": float(np.mean(a == b))}

    summary["ratios"] = {m: ratio(m) for m in METHODS if m != "QFC" and ok[m]}
    gate = {}
    for opp, order in OPPONENTS.items():
        prim = next((v for v in order if v in stats and stats[v]["safe_S1_and_S2"]), None)
        rec = {"order": list(order), "dev_reference_primary": prim,
               "variant_safety": {v: {"S1": stats[v]["S1"], "S2": stats[v]["S2"], "fwer_k": stats[v]["fwer_k"],
                                      "fwer_cp_upper": stats[v]["fwer_cp_upper"],
                                      "fcr_old_upper": stats[v]["fcr_old_upper"]} for v in order if v in stats}}
        if prim is None:
            rec["verdict"] = "UNDETERMINED (no variant satisfies S1 AND S2 on dev)"
            rec["pass"] = None
        else:
            r = summary["ratios"][prim]
            rec["geomean_ratio"] = r["geomean_ratio_QFC_over_X"]
            rec["ci95"] = r["ci95"]
            rec["pass"] = bool(r["geomean_ratio_QFC_over_X"] <= GATE_RATIO)
            rec["verdict"] = "PASS" if rec["pass"] else "FAIL"
        rec["all_variants_ratio"] = {v: summary["ratios"][v]["geomean_ratio_QFC_over_X"]
                                     for v in order if v in summary["ratios"]}
        rec["worst_case_ratio_over_variants"] = max(rec["all_variants_ratio"].values())
        gate[opp] = rec
    summary["gate_G1_shape_c"] = gate
    summary["gate_pass_all"] = bool(all(g["pass"] for g in gate.values()))
    summary["variant_dev_fwer"] = {v: {"fwer": stats[v]["fwer"], "fwer_cp_upper": stats[v]["fwer_cp_upper"],
                                       "fcr_old_upper": stats[v]["fcr_old_upper"]} for v in stats}
    summary["sec_per_stream"] = {m: stats[m]["sec_per_stream_mean"] for m in stats}
    summary["peace_mc"] = {m: {"gw_mc_rel_se_max": max((r.get("gw_mc_rel_se_max", 0.0) for r in ok[m]), default=None),
                               "tau_alg2_mc_rel_se_max": max((r.get("tau_alg2_mc_rel_se_max", 0.0) for r in ok[m]),
                                                             default=None),
                               "rounds_mean": float(np.mean([r.get("gw_n_rounds", 0) for r in ok[m]]))}
                           for m in METHODS if m.startswith("Peace") and ok[m]}

    # ---- consistency with part A's QFC (if available)
    pa = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / "r4_g1_shape_a" / "streams.jsonl"
    chk = {"partA_results_path": str(pa.relative_to(WS)), "available": pa.exists()}
    if pa.exists():
        try:
            a_q = {}
            for line in pa.read_text().splitlines():
                d = json.loads(line)
                if d.get("method") == "QFC" and d.get("layer") == "CR9":
                    a_q[d["perm_seed"]] = d["N80"]
            common = [s for s in qfc_n80 if s in a_q]
            mism = [s for s in common if a_q[s] != qfc_n80[s]]
            chk.update({"n_common": len(common), "n_mismatch": len(mism), "mismatch_seeds": mism[:10]})
        except Exception as ex:  # noqa: BLE001
            chk["error"] = repr(ex)
    summary["qfc_partA_consistency"] = chk
    pb = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / "r4_g1_shape_b" / "results.jsonl"
    chkb = {"partB_results_path": str(pb.relative_to(WS)), "available": pb.exists()}
    if pb.exists():
        b_q = {}
        for line in pb.read_text().splitlines():
            d = json.loads(line)
            if d.get("method") == "QFC" and "N80" in d:
                b_q[d["perm_seed"]] = d["N80"]
        common = [s for s in qfc_n80 if s in b_q]
        chkb.update({"n_common": len(common), "n_mismatch": sum(b_q[s] != qfc_n80[s] for s in common)})
    summary["qfc_partB_consistency"] = chkb

    # ---- samples: 5 streams, per method N80 / cert checkpoints / decided policy vs optimum
    samp = []
    for s in seeds[:5]:
        row = {"perm_seed": s, "pi_star": [ctx.pols[int(np.argmax(np.where(ctx.feas[q], J_true, -np.inf)))].tolist()
                                           for q in range(ctx.Q)], "methods": {}}
        for m in METHODS:
            r = next((x for x in ok[m] if x["perm_seed"] == s), None)
            if r is None:
                continue
            row["methods"][m] = {"N80": r["N80"], "k_stop": r["k_stop"], "n_cert": r["n_cert"],
                                 "n_false": r["n_false"], "cert_k": r["cert_k"],
                                 "decided_pi": [ctx.pols[i].tolist() if i >= 0 else None for i in r["decided_pi"]],
                                 "curve": r["curve"], "U_last": r["U_last"], "n_cells_last": r["n_cells_last"]}
        samp.append(row)
    (out_dir / "samples" / "cr9_streams_900_904.json").write_text(json.dumps(samp, indent=1))

    # ---- plot: mean certified curve per method
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ck = np.array(ctx.checkpoints)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for m in METHODS:
            curves = np.zeros((len(ok[m]), len(ck)))
            for i, r in enumerate(ok[m]):
                c = r["curve"]
                curves[i, :len(c)] = c
                if len(c) < len(ck):
                    curves[i, len(c):] = c[-1]
            ax.plot(ck, curves.mean(0) / ctx.Q, marker="o", ms=2.5, label=m, lw=2 if m == "QFC" else 1)
        ax.axhline(0.8, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("arrivals")
        ax.set_ylabel("certified / 15 (sticky, frozen after stop)")
        ax.set_title(f"CR9 visit dev, eps={eps}, 100 streams")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "certified_vs_arrivals.png", dpi=130)
        summary["plot"] = str((out_dir / "certified_vs_arrivals.png").relative_to(WS))
    except Exception as ex:  # noqa: BLE001
        summary["plot_error"] = repr(ex)

    summary["integrity"] = {"n_errors": summary["n_errors"], "eval_seeds_touched": False,
                            "billing_ok_all": bool(all(s["billing_ok_all"] for s in stats.values()))}
    summary["wall_min"] = round((time.time() - t_start) / 60, 2)
    summary["finished_at"] = datetime.now().isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    progress(4, 4, "done")
    ok_run = summary["n_errors"] == 0 and summary["integrity"]["billing_ok_all"]
    gs = "; ".join(f"{o}: {g['dev_reference_primary']} ratio={g.get('geomean_ratio', float('nan')):.3f} {g['verdict']}"
                   for o, g in gate.items())
    mark_done("success" if ok_run else "failed", f"eps*={eps}; {gs}")
    print(json.dumps({"gate": {o: (g["dev_reference_primary"], g.get("geomean_ratio"), g["verdict"])
                               for o, g in gate.items()},
                      "ratios": {m: round(v["geomean_ratio_QFC_over_X"], 3) for m, v in summary["ratios"].items()},
                      "wall_min": summary["wall_min"], "errors": summary["n_errors"]}))


if __name__ == "__main__":
    main()
