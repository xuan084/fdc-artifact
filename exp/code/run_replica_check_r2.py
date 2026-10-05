"""replica_check_r2: independent replica re-test after the completed E1-NL spec (methodology 2.1).

Pilot (this file): exact J_theta* cross-check main vs replica, tolerance 1e-9, on 100 dev problems:
  * offgrid  : R1 continuous truth (methodology 2.2), dev seeds 640-643, stream 0, first 10 problems   (40)
  * twin     : matched twin = R0 truth with gamma = lam = 0 (methodology 2.3), dev seeds 644-646          (30)
  * m4       : in-class kappa_high truth + temporal drift alpha_i(t) = alpha_i + s t_global / T_ref
               (methodology 2.4), dev seeds 690-692 (hf3 pilot seeds; s = hf3 calibrated slope for stream 0),
               problem k at its nominal start t_nom = n0 + 200 k                                       (30)
  * secondary (not counted in the 100): dose g = 0.5 on the R1 truth, dev seeds 640-641 (20 problems)
m4: the main implementation scores problem k with alpha FROZEN at the problem start round. The replica computes
both the frozen value (compared at 1e-9) and the text-literal per-round drift (t_global = t_nom + t), and reports
the frozen-approximation error |J_literal - J_frozen| separately (it is a property of the main harness, not a bug).

Bridge (main side, data / black boxes only): instances from dsswm.streams.{offgrid,generator} (harness), policies
as opaque act() callables decoded through ActionSpace.actions, main J via NLPropagator.j_table at the truth params.
The replica package exp/code/replica never imports dsswm.
Full mode (JPC_infl vs B3 on replica env, eval 10000-10015) is not implemented here: it needs a replica-backed
platform adapter for the R2 runner; see summary.json 'not_done_in_pilot'.
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
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
sys.path.insert(0, str(HERE))

from replica.model import SPEC_GAPS, SPEC_GAPS_R2_STATUS, ReplicaNL  # noqa: E402

TASK = "replica_check_r2"
RES_ROOT = WS / "exp" / "results"
TOL = 1e-9
N0, T_NOMINAL_STEP = 20, 200


def report_progress(step, total, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": 0, "total_epochs": 1, "step": step, "total_steps": total,
        "loss": None, "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
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


def _hf3_slopes():
    """Calibrated m4 slope s per (instance, stream 0) from the hf3 pilot (data only)."""
    out = {}
    p = RES_ROOT / "pilots" / "hf3_m4_blindspot" / "results.jsonl"
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("rec") == "row" and r.get("stream") == 0 and r.get("misspec") == "m4":
                out.setdefault(int(r["instance"]), float(r["strength"]))
    return out


def build_case(kind, seed, n_prob, slope=None):
    """Returns (problems, aspace, truth_params(dict of numpy), cfg, meta_fn(k) -> dict)."""
    from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, make_nl_instance
    from dsswm.streams.offgrid import make_offgrid_instance
    if kind == "offgrid":
        inst = make_offgrid_instance(seed, "R1", stream=0)
    elif kind == "dose0.5":
        inst = make_offgrid_instance(seed, "R1", stream=0, dose=0.5)
    elif kind == "twin":
        inst = make_offgrid_instance(seed, "R0", stream=0, twin=True)
    elif kind == "m4":
        from dsswm.models.nl_class import NLClass
        inst = make_nl_instance(seed, noise_seed=42, nl_class=NLClass(DEFAULT_NL_S_GRID, device="cpu"),
                                kappa_mode="high", K=15)
    else:
        raise ValueError(kind)
    main_tp = inst.env.true_params()                     # main-side parameter dict (for the main J only)
    tr = inst.truth                                      # harness truth dict (replica input, plain data)
    tp = {k: np.asarray(tr[k], float) for k in ("alpha", "beta", "gamma", "tauL", "tauR", "psi", "lam")}
    cfg = {**NL_DEFAULTS, **{k: v for k, v in inst.cfg.items() if k in ("c", "rho_ret", "nmax")}}
    return inst.problems[:n_prob], inst.env.aspace, tp, cfg, main_tp


def case_rows(kind, seed, n_prob, slope=None):
    import torch
    torch.set_num_threads(1)
    from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
    from dsswm.envs.mis import M4_T_REF
    problems, aspace, tp, cfg, main_tp = build_case(kind, seed, n_prob)
    prop = NLPropagator(2, 2, cfg["nmax"], aspace, cfg["c"], cfg["rho_ret"], device="cpu")
    psi = np.atleast_1d(tp["psi"])
    rows = []
    for k, q in enumerate(problems):
        u = q.utility
        w, w_ret, c_q = np.asarray(u.w, float), float(u.w_ret), float(u.c_q)
        shift0 = 0.0
        tp_k = main_tp
        if kind == "m4":
            t_nom = N0 + T_NOMINAL_STEP * k
            shift0 = slope * t_nom / M4_T_REF
            tp_k = dict(main_tp)
            tp_k["alpha"] = np.asarray(main_tp["alpha"], float) + shift0
        t0 = time.time()
        Jm = np.asarray(prop.j_table(params_to_torch(tp_k, device="cpu"), q.policies, q.loads0, q.engaged0, q.H, u)
                        ).reshape(-1)
        t_main = time.time() - t0
        rep = ReplicaNL(tp["alpha"], tp["beta"], tp["gamma"], tp["tauL"], tp["tauR"], psi, float(tp["lam"]),
                        c=cfg["c"], rho_ret=cfg["rho_ret"], L=2, R=2, nmax=cfg["nmax"],
                        y_mode="both_engaged", ret_load="pre")
        for a, pol in enumerate(q.policies):
            pc = make_policy_callable(pol, aspace)
            ts = time.time()
            jr = rep.exact_J(pc, q.loads0, q.engaged0, q.H, w, w_ret, c_q, alpha_shift=shift0)
            t_rep = time.time() - ts
            row = {"rec": "row", "task": TASK, "variant": kind, "instance": int(seed), "stream": 0,
                   "problem": q.pid, "k": k, "H": int(q.H), "policy_idx": a, "policy": pol.name,
                   "J_main": float(Jm[a]), "J_replica": float(jr), "abs_diff": abs(float(Jm[a]) - float(jr)),
                   "t_main_s": t_main / len(q.policies), "t_rep_s": t_rep,
                   "truth": {"gamma": tp["gamma"].tolist(), "lam": float(tp["lam"]), "psi": psi.tolist()}}
            if kind == "m4":
                t_nom = N0 + T_NOMINAL_STEP * k
                jl = rep.exact_J(pc, q.loads0, q.engaged0, q.H, w, w_ret, c_q,
                                 alpha_shift=lambda t, _s=slope, _t0=t_nom: _s * (_t0 + t) / M4_T_REF)
                row.update({"drift_s": slope, "t_nom": t_nom, "alpha_shift_start": shift0,
                            "within_horizon_shift": slope * (q.H - 1) / M4_T_REF,
                            "J_replica_literal_drift": float(jl),
                            "frozen_approx_err": abs(float(jl) - float(jr))})
            rows.append(row)
    return rows


def mc_check(kind, seed, n_roll, n_problems, rng_seed=42):
    problems, aspace, tp, cfg, _ = build_case(kind, seed, n_problems)
    rep = ReplicaNL(tp["alpha"], tp["beta"], tp["gamma"], tp["tauL"], tp["tauR"], np.atleast_1d(tp["psi"]),
                    float(tp["lam"]), c=cfg["c"], rho_ret=cfg["rho_ret"], nmax=cfg["nmax"], seed=rng_seed)
    out = []
    for q in problems:
        u = q.utility
        pc = make_policy_callable(q.policies[0], aspace)
        args = (q.loads0, q.engaged0, q.H, np.asarray(u.w, float), float(u.w_ret), float(u.c_q))
        ex = rep.exact_J(pc, *args)
        xs = np.array([rep.rollout(pc, *args) for _ in range(n_roll)])
        se = xs.std(ddof=1) / np.sqrt(n_roll)
        out.append({"variant": kind, "instance": seed, "problem": q.pid, "exact": ex, "mc_mean": float(xs.mean()),
                    "mc_se": float(se), "z": float((xs.mean() - ex) / se) if se > 0 else 0.0, "n_roll": n_roll})
    return out


def power_control(kind, seed, n_prob=10):
    """Negative control: the round-0 alternative spec readings must FAIL the 1e-9 test on round-2 truths."""
    import torch
    torch.set_num_threads(1)
    from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
    problems, aspace, tp, cfg, main_tp = build_case(kind, seed, n_prob)
    prop = NLPropagator(2, 2, cfg["nmax"], aspace, cfg["c"], cfg["rho_ret"], device="cpu")
    out = {}
    for ym, rl in (("always", "pre"), ("both_engaged", "post")):
        rep = ReplicaNL(tp["alpha"], tp["beta"], tp["gamma"], tp["tauL"], tp["tauR"], np.atleast_1d(tp["psi"]),
                        float(tp["lam"]), c=cfg["c"], rho_ret=cfg["rho_ret"], nmax=cfg["nmax"], y_mode=ym, ret_load=rl)
        d = []
        for q in problems:
            u = q.utility
            Jm = np.asarray(prop.j_table(params_to_torch(main_tp, device="cpu"), q.policies, q.loads0, q.engaged0,
                                         q.H, u)).reshape(-1)
            for a, pol in enumerate(q.policies):
                jr = rep.exact_J(make_policy_callable(pol, aspace), q.loads0, q.engaged0, q.H, np.asarray(u.w, float),
                                 float(u.w_ret), float(u.c_q))
                d.append(abs(jr - float(Jm[a])))
        # with lam = 0 (twin) retention does not depend on load, so the pre/post reading is structurally identical
        expect_fail = not (rl == "post" and float(tp["lam"]) == 0.0)
        out[f"{kind}:{seed}:{ym}|{rl}"] = {"max_abs_J_diff": float(max(d)), "fails_tol": bool(max(d) > TOL),
                                           "expected_to_fail": expect_fail}
    return out


def rank_agree(rows):
    g = {}
    for r in rows:
        g.setdefault((r["instance"], r["problem"]), []).append((r["J_main"], r["J_replica"]))
    agree = [int(v[int(np.argmax([b for _, b in v]))][0] >= max(a for a, _ in v) - 1e-9) for v in g.values()]
    return float(np.mean(agree)) if agree else None


def stats(rows):
    d = np.array([r["abs_diff"] for r in rows])
    per = {}
    for r in rows:
        per.setdefault((r["instance"], r["problem"]), []).append(r["abs_diff"])
    return {"n_problems": len(per), "n_policy_values": int(d.size), "max_abs_J_diff": float(d.max()),
            "median_abs_J_diff": float(np.median(d)), "n_problems_pass": int(sum(max(x) <= TOL for x in per.values())),
            "argmax_agreement": rank_agree(rows), "pass": bool(d.max() <= TOL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "full":
        raise SystemExit("full mode (JPC_infl vs B3 on the replica platform, eval 10000-10015) not implemented yet")

    out = RES_ROOT / "pilots" / TASK
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    (out / "start_time.txt").write_text(start_iso)
    t_start = time.time()
    report_progress(0, 3, {"stage": "jcheck"})

    slopes = _hf3_slopes()
    jobs = ([("offgrid", s, 10, None) for s in range(640, 644)] + [("twin", s, 10, None) for s in range(644, 647)]
            + [("m4", s, 10, slopes[s]) for s in range(690, 693)] + [("dose0.5", s, 10, None) for s in (640, 641)])
    from joblib import Parallel, delayed
    rows = [r for rs in Parallel(n_jobs=args.workers)(delayed(case_rows)(*j) for j in jobs) for r in rs]
    report_progress(1, 3, {"stage": "mc"})
    mc = [r for rs in Parallel(n_jobs=args.workers)(
        delayed(mc_check)(k, s, 20000, 2) for k, s in (("offgrid", 640), ("twin", 644), ("offgrid", 641),
                                                        ("twin", 645))) for r in rs]
    pcs = {}
    for d in Parallel(n_jobs=args.workers)(delayed(power_control)(k, s) for k, s in (("offgrid", 640), ("twin", 644))):
        pcs.update(d)
    report_progress(2, 3, {"stage": "aggregate"})

    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "mc_consistency.json").write_text(json.dumps(mc, indent=1))

    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    var_stats = {v: stats(rs) for v, rs in by.items()}
    primary = [r for r in rows if r["variant"] in ("offgrid", "twin", "m4")]
    prim = stats(primary)
    m4 = by["m4"]
    fe = np.array([r["frozen_approx_err"] for r in m4])
    m4_frozen = {"max_abs_J_literal_minus_frozen": float(fe.max()), "median": float(np.median(fe)),
                 "max_over_eps": float(fe.max() / 0.02),
                 "slopes": {str(s): slopes[s] for s in range(690, 693)},
                 "note": "Main harness scores m4 with alpha frozen at the problem start; text-literal per-round drift "
                         "differs by this amount (within-horizon shift s*(H-1)/T_ref)."}
    twin_ok = all(max(r["truth"]["gamma"]) == 0.0 and r["truth"]["lam"] == 0.0 for r in by["twin"])
    offgrid_offgrid = float(np.mean([any(abs(g - 0.75) > 1e-12 for g in r["truth"]["gamma"])
                                     for r in by["offgrid"]]))
    mc_z = np.array([m["z"] for m in mc])
    go = prim["pass"] and bool(np.abs(mc_z).max() < 4) and twin_ok and all(v["fails_tol"] == v["expected_to_fail"] for v in pcs.values())
    summary = {
        "task_id": TASK, "mode": "pilot", "seed": 42, "tolerance": TOL,
        "dev_seeds": {"offgrid": [640, 643], "twin": [644, 646], "m4": [690, 692], "dose0.5_secondary": [640, 641]},
        "primary_100": prim, "variants": var_stats,
        "max_abs_J_diff": prim["max_abs_J_diff"],
        "m4_frozen_vs_literal_drift": m4_frozen,
        "sanity": {"twin_gamma_lam_zero": twin_ok, "offgrid_gamma_off_grid_share": offgrid_offgrid},
        "power_control_alt_spec_readings": pcs,
        "mc_internal_consistency": {"n": len(mc), "max_abs_z": float(np.abs(mc_z).max()),
                                    "pass": bool(np.abs(mc_z).max() < 4)},
        "replica_config": {"y_mode": "both_engaged", "ret_load": "pre",
                           "basis": "methodology.md sec. 2.1 items 1-3 (completed spec), sec. 2.3 twin, sec. 2.4 m4"},
        "spec_gaps_round0": SPEC_GAPS, "spec_gaps_round2_status": SPEC_GAPS_R2_STATUS,
        "bridge": "instances from dsswm.streams.offgrid / generator (harness, same seeds); policies as opaque act() "
                  "callables; main J via NLPropagator.j_table at the truth params (m4: alpha + s*t_nom/T_ref). "
                  "Replica package imports no dsswm code.",
        "independence_caveat": "Policy semantics are reused as black boxes (NOT independently replicated). The three "
                               "sec. 2.1 items were written down after the round-0 replica saw dsswm docstrings, so "
                               "they are a documented, not a blind, resolution. For m4 the replica author read the "
                               "main DriftNLEnv (one-line drift formula) while building the bridge. Instance "
                               "generation (truth draws, problems, utilities, c_q) is shared harness code.",
        "not_done_in_pilot": "full mode: JPC_infl vs B3 (R2) on the replica platform for eval 10000-10015 "
                             "(direction_agreement, replicated_stream_ratio) needs a replica-backed platform adapter "
                             "for the R2 runner; null here.",
        "direction_agreement": None, "replicated_stream_ratio": None,
        "wall_clock_s": time.time() - t_start, "start_time": start_iso,
        "runtime_note": "CPU only, 4 joblib workers, concurrent with other tasks",
    }
    summary["go_no_go"] = "GO" if go else "NO_GO"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    samp = []
    for v in ("offgrid", "twin", "m4", "dose0.5"):
        samp += by[v][:2]
    samp += sorted(primary, key=lambda r: -r["abs_diff"])[:2]
    samp += sorted(m4, key=lambda r: -r["frozen_approx_err"])[:2]
    (out / "samples" / "jtrue_pairs.json").write_text(json.dumps(samp, indent=1))
    report_progress(3, 3, {"stage": "done", "go_no_go": summary["go_no_go"], "max_abs_J_diff": prim["max_abs_J_diff"]})
    print(json.dumps({k: summary[k] for k in ("primary_100", "variants", "m4_frozen_vs_literal_drift", "sanity",
                                              "power_control_alt_spec_readings",
                                              "mc_internal_consistency", "go_no_go", "wall_clock_s")}, indent=1))
    mark_done("success", f"pilot J re-check (offgrid/twin/m4, 100 problems): {summary['go_no_go']}; "
                         f"max |dJ| = {prim['max_abs_J_diff']:.2e}; m4 frozen-approx err max "
                         f"{m4_frozen['max_abs_J_literal_minus_frozen']:.2e}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        mark_done("failed", repr(e))
        raise
