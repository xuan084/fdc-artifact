"""Tier 2: independent re-implementation cross-check (pilot = J-truth smoke test).

Bridge rules (what is taken from the main implementation, as *data* or black boxes only):
  * instance definitions (theta*, loads0, engaged0, H, utility weights) from dsswm.streams.generator
    with the same instance seeds -- the generator is the pre-registered harness, not the method;
  * candidate policies are passed as opaque callables policy.act(t, loads, engaged) -> action index,
    decoded through ActionSpace.actions into (pairs, incentives);
  * main J_theta* values via dsswm's public J routines, for comparison only.
The replica (exp/code/replica/) never imports dsswm and computes J from the methodology text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
sys.path.insert(0, str(HERE))

from replica.model import SPEC_GAPS, ReplicaLin, ReplicaNL  # noqa: E402

TASK = "tier2_indep_replication"
RES_ROOT = WS / "exp" / "results"


def report_progress(step, total, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": 0, "total_epochs": 1, "step": step, "total_steps": total,
        "loss": None, "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({
        "task_id": TASK, "status": status, "summary": summary, "final_progress": fp,
        "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------ bridge (main side)
def make_policy_callable(pol, aspace):
    cache = {}
    acts = aspace.actions

    def f(t, loads, eng):
        key = (t, loads, eng)
        r = cache.get(key)
        if r is None:
            a = acts[int(pol.act(t, np.array(loads), np.array(eng)))]
            r = (tuple(tuple(int(x) for x in p) for p in a.pairs),
                 {(int(i), int(j)): int(lv) for (i, j, lv) in a.incentives})
            cache[key] = r
        return r
    return f


def lin_instance_rows(seed):
    from dsswm.streams import generator as G
    from dsswm.models.lin_class import LinClass
    inst = G.make_lin_instance(seed)
    theta = np.asarray(inst.truth["theta"], float)
    cfg = inst.cfg
    aspace = inst.env.aspace
    lc = LinClass(cfg["L"], cfg["R"], len(theta) - 2 * cfg["L"] - cfg["R"] + 1, nmax=cfg["nmax"])
    rep = ReplicaLin(theta, L=cfg["L"], R=cfg["R"], nmax=cfg["nmax"], sigma=cfg["sigma"])
    rows = []
    for q in inst.problems:
        u = q.utility
        for k, pol in enumerate(q.policies):
            t0 = time.time()
            j_main = float(lc.z(pol, q.loads0, q.H, u, aspace) @ theta)
            t1 = time.time()
            j_rep = rep.exact_J(make_policy_callable(pol, aspace), q.loads0, q.engaged0, q.H,
                                np.asarray(u.w, float), float(u.w_ret), float(u.c_q))
            t2 = time.time()
            rows.append({"env": "E1-Lin", "instance": seed, "problem": q.pid, "H": int(q.H), "policy_idx": k,
                         "policy": pol.name, "J_main": j_main, "J_replica": j_rep,
                         "abs_diff": abs(j_main - j_rep), "t_main_s": t1 - t0, "t_rep_s": t2 - t1})
    return rows


VARIANTS = [("both_engaged", "pre"), ("both_engaged", "post"), ("always", "pre"), ("always", "post")]


def nl_instance_rows(seed, variants):
    import torch  # noqa: F401
    torch.set_num_threads(2)
    from dsswm.streams import generator as G
    from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
    inst = G.make_nl_instance(seed)
    env, cfg, tr = inst.env, inst.cfg, inst.truth
    prop = NLPropagator(cfg["L"], cfg["R"], cfg["nmax"], env.aspace, cfg["c"], cfg["rho_ret"])
    tp = env.true_params() if callable(env.true_params) else env.true_params
    reps = {v: ReplicaNL(tr["alpha"], tr["beta"], tr["gamma"], tr["tauL"], tr["tauR"], tr["psi"], tr["lam"],
                         c=cfg["c"], rho_ret=cfg["rho_ret"], L=cfg["L"], R=cfg["R"], nmax=cfg["nmax"],
                         y_mode=v[0], ret_load=v[1]) for v in variants}
    rows = []
    for q in inst.problems:
        u = q.utility
        t0 = time.time()
        Jm = np.asarray(prop.j_table(params_to_torch(tp), q.policies, q.loads0, q.engaged0, q.H, u)).reshape(-1)
        t1 = time.time()
        for k, pol in enumerate(q.policies):
            pc = make_policy_callable(pol, env.aspace)
            row = {"env": "E1-NL-S", "instance": seed, "problem": q.pid, "H": int(q.H), "policy_idx": k,
                   "policy": pol.name, "J_main": float(Jm[k]), "t_main_s": (t1 - t0) / len(q.policies)}
            for v, rep in reps.items():
                ts = time.time()
                jr = rep.exact_J(pc, q.loads0, q.engaged0, q.H, np.asarray(u.w, float), float(u.w_ret), float(u.c_q))
                tag = f"{v[0]}|{v[1]}"
                row[f"J_replica[{tag}]"] = jr
                row[f"abs_diff[{tag}]"] = abs(jr - float(Jm[k]))
                row[f"t_rep_s[{tag}]"] = time.time() - ts
            rows.append(row)
    return rows


def mc_check(kind, seed, n_roll, n_problems, rng_seed=42):
    """Internal consistency: sampled replica rollouts vs replica exact J (|z| < 4 expected)."""
    from dsswm.streams import generator as G
    out = []
    if kind == "lin":
        inst = G.make_lin_instance(seed)
        cfg = inst.cfg
        rep = ReplicaLin(inst.truth["theta"], L=cfg["L"], R=cfg["R"], nmax=cfg["nmax"], sigma=cfg["sigma"], seed=rng_seed)
    else:
        inst = G.make_nl_instance(seed)
        cfg, tr = inst.cfg, inst.truth
        rep = ReplicaNL(tr["alpha"], tr["beta"], tr["gamma"], tr["tauL"], tr["tauR"], tr["psi"], tr["lam"],
                        c=cfg["c"], rho_ret=cfg["rho_ret"], seed=rng_seed)
    for q in inst.problems[:n_problems]:
        u = q.utility
        pc = make_policy_callable(q.policies[0], inst.env.aspace)
        args = (q.loads0, q.engaged0, q.H, np.asarray(u.w, float), float(u.w_ret), float(u.c_q))
        ex = rep.exact_J(pc, *args)
        xs = np.array([rep.rollout(pc, *args) for _ in range(n_roll)])
        se = xs.std(ddof=1) / np.sqrt(n_roll)
        out.append({"env": kind, "instance": seed, "problem": q.pid, "exact": ex, "mc_mean": float(xs.mean()),
                    "mc_se": float(se), "z": float((xs.mean() - ex) / se) if se > 0 else 0.0, "n_roll": n_roll})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "full":
        raise SystemExit("full mode (replicated JPC vs B1 on 16 eval instances) not implemented in this pilot run")

    out = RES_ROOT / "pilots" / TASK
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    (out / "start_time.txt").write_text(datetime.now().isoformat())
    t_start = time.time()
    report_progress(0, 4, {"stage": "lin"})

    from joblib import Parallel, delayed
    seeds = list(range(10))  # pilot dev instance seeds 0..9 (same as main pilot tasks), 10 problems each
    lin_rows = [r for rs in Parallel(n_jobs=args.workers)(delayed(lin_instance_rows)(s) for s in seeds) for r in rs]
    report_progress(1, 4, {"stage": "nl"})
    nl_rows = [r for rs in Parallel(n_jobs=args.workers)(delayed(nl_instance_rows)(s, VARIANTS) for s in seeds)
               for r in rs]
    report_progress(2, 4, {"stage": "mc"})
    mc = [r for rs in Parallel(n_jobs=args.workers)(
        delayed(mc_check)(k, s, 20000, 2) for k in ("lin", "nl") for s in (0, 1)) for r in rs]
    report_progress(3, 4, {"stage": "aggregate"})

    with open(out / "results.jsonl", "w") as f:
        for r in lin_rows + nl_rows:
            f.write(json.dumps(r) + "\n")
    (out / "mc_consistency.json").write_text(json.dumps(mc, indent=1))

    tol = 1e-9
    lin_diff = np.array([r["abs_diff"] for r in lin_rows])
    var_stats = {}
    for v in VARIANTS:
        tag = f"{v[0]}|{v[1]}"
        d = np.array([r[f"abs_diff[{tag}]"] for r in nl_rows])
        per_prob = {}
        for r in nl_rows:
            per_prob.setdefault((r["instance"], r["problem"]), []).append(r[f"abs_diff[{tag}]"])
        var_stats[tag] = {"max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
                          "n_policy_values": int(d.size), "n_problems": len(per_prob),
                          "n_problems_pass": int(sum(max(x) < tol for x in per_prob.values()))}
    primary = "both_engaged|pre"
    best = min(var_stats, key=lambda k: var_stats[k]["max_abs_diff"])
    lin_probs = {}
    for r in lin_rows:
        lin_probs.setdefault((r["instance"], r["problem"]), []).append(r["abs_diff"])

    # policy argmax / ranking agreement (direction sanity)
    def rank_agree(rows, key):
        g = {}
        for r in rows:
            g.setdefault((r["instance"], r["problem"]), []).append((r["J_main"], r[key]))
        # tie-aware: replica's argmax must be a main-optimal policy (ties within tol, e.g. duplicate policies)
        agree = [int(v[int(np.argmax([b for _, b in v]))][0] >= max(a for a, _ in v) - 1e-9) for v in g.values()]
        return float(np.mean(agree))

    mc_z = np.array([m["z"] for m in mc])
    nl_pass = var_stats[primary]["max_abs_diff"] < tol
    lin_pass = float(lin_diff.max()) < tol
    summary = {
        "task_id": TASK, "mode": "pilot", "seed": 42, "instance_seeds": seeds,
        "tolerance": tol,
        "E1-Lin": {"n_problems": len(lin_probs), "n_policy_values": int(lin_diff.size),
                   "jtrue_max_abs_diff": float(lin_diff.max()), "jtrue_median_abs_diff": float(np.median(lin_diff)),
                   "n_problems_pass": int(sum(max(x) < tol for x in lin_probs.values())),
                   "argmax_agreement": rank_agree(lin_rows, "J_replica"), "pass": lin_pass},
        "E1-NL-S": {"primary_variant": primary, "variants": var_stats, "best_matching_variant": best,
                    "jtrue_max_abs_diff": var_stats[primary]["max_abs_diff"],
                    "argmax_agreement": rank_agree(nl_rows, f"J_replica[{primary}]"), "pass": bool(nl_pass)},
        "mc_internal_consistency": {"n": len(mc), "max_abs_z": float(np.abs(mc_z).max()),
                                    "pass": bool(np.abs(mc_z).max() < 4)},
        "spec_gaps": SPEC_GAPS,
        "bridge": "instances from dsswm.streams.generator (same seeds); policies as opaque act() callables; "
                  "main J via LinClass.z / NLPropagator.j_table. Replica package imports no dsswm code.",
        "independence_caveat": "Replica author saw dsswm module/class docstrings and signatures while building "
                               "the data bridge (not function bodies). The NL docstring states y=0 unless both "
                               "engaged and the w_ret*sum_p E[e_p(H)] utility term, so those two spec gaps were "
                               "not resolved blind. Policy semantics are NOT independently replicated.",
        "not_done_in_pilot": "replicated JPC vs B1 on 16 eval instances (full mode only); "
                             "replicated_rmst_ratio and direction_agreement are therefore null",
        "replicated_rmst_ratio": None, "direction_agreement": None,
        "upstream_gate_note": "Task is gated on Tier-1 pass; analysis_aggregate verdict was REFINE "
                              "(Holm rejects only H3). This pilot smoke test was run as dispatched.",
        "wall_clock_s": time.time() - t_start,
        "runtime_note": "CPU only (4 joblib workers, concurrent with other tasks); main J for NL on GPU 0",
    }
    summary["go_no_go"] = "GO" if (lin_pass and nl_pass and summary["mc_internal_consistency"]["pass"]) else "NO_GO"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # samples: 5 representative rows
    samp = lin_rows[:2] + nl_rows[:3] + sorted(nl_rows, key=lambda r: -r[f"abs_diff[{primary}]"])[:2]
    (out / "samples" / "jtrue_pairs.json").write_text(json.dumps(samp, indent=1))
    report_progress(4, 4, {"stage": "done", "go_no_go": summary["go_no_go"]})
    print(json.dumps({k: summary[k] for k in ("E1-Lin", "E1-NL-S", "mc_internal_consistency", "go_no_go")}, indent=1))
    mark_done("success", f"pilot J-truth cross-check: {summary['go_no_go']}; "
                         f"Lin max diff {summary['E1-Lin']['jtrue_max_abs_diff']:.2e}, "
                         f"NL[{primary}] max diff {summary['E1-NL-S']['jtrue_max_abs_diff']:.2e}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # write DONE on failure too
        import traceback
        traceback.print_exc()
        mark_done("failed", repr(e))
        raise
