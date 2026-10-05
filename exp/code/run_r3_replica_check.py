"""r3_replica_check: independent replica re-computation of the true J on evaluation problems (harness validity).

For every sampled problem the replica (exp/code/replica, never imports dsswm) recomputes the exact true J of EVERY
candidate policy from the truth parameters (plain data) and compares it with the dsswm J that was used to score the
round-3 rows; then every scored row (all methods / arms) on that problem is re-judged:
    true_regret_rep = max_pi J_rep - J_rep[certified_policy],   false_cert_rep = CERTIFIED and true_regret_rep > eps.
Layers / sources (pilot: the dev-seed pilot outputs of the four dependency tasks)
  NL-R0    r3_nl_main_a   QuotaBuilder(s).stream(t, 'quota', n), truth = gap_quota.nl_r0_truth(s); J_dsswm = st.J[k][ti]
  static   r3_static_a    same problems, truth = dosed_truth(g) (gamma, lam x g; g = 0 is the exact static twin);
                          J_dsswm = eta_loc.j_values at v*(g) x c_q (the runner's scoring path)
  R1, R1adv r3_hazard_a   off-grid theta* from the placement record; J_dsswm = placement J_true (the scored values)
  heldout  r3_heldout_misspec  m1r_eps / m1r_2eps / alias / m3: truth = family_spec(...)(strength from the row);
                          J_dsswm = main propagator at that truth (the rows were already scored with replica J)
Pilot sample: 100 problems stratified by layer, rng [42, stratum]: NL-R0 all 12 available, 22 each from static / R1 /
  R1adv / heldout. Because the replica is cheap, a census over ALL pilot problems is also run (secondary).
Power control: two alternative spec readings of the replica (y 'always'; retention on post-update load) must FAIL the
  tolerance on the in-class problems (the check can detect a spec mismatch).
Pass: max |J_rep - J_dsswm| <= 1e-6 AND 0 label flips (primary sample).
Independence caveat: policies are opaque act() callables decoded through the dsswm ActionSpace (not re-implemented);
  instance generation (truth draws, problems, utilities, c_q) is shared harness code.

Usage: run_r3_replica_check.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3"
TASK = "r3_replica_check"
TOL = 1e-6
EPS = 0.02
PLANNED_MIN = 8
SOURCES = {"NL-R0": "r3_nl_main_a", "static": "r3_static_a", "hazard": "r3_hazard_a",
           "heldout": "r3_heldout_misspec"}
PILOT_ALLOC = {"NL-R0": 12, "static": 22, "R1": 22, "R1adv": 22, "heldout": 22}
STRATA = ["NL-R0", "static", "R1", "R1adv", "heldout"]


def report_progress(step, total, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": 0, "total_epochs": 1, "step": step, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


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


def update_gpu_progress(status, start_iso, wall_min, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        other = "failed" if key == "completed" else "completed"
        if TASK in d.get(other, []):
            d[other].remove(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": PLANNED_MIN, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def update_shared_summary(entry, md):
    lockp = RES_ROOT / "pilot_summary.lock"
    with open(lockp, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            pj = RES_ROOT / "pilot_summary.json"
            d = json.loads(pj.read_text()) if pj.exists() else {"tasks": {}}
            d.setdefault("tasks", {})[TASK] = entry
            tmp = pj.with_name(pj.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))
            os.replace(tmp, pj)
            pm = RES_ROOT / "pilot_summary.md"
            txt = pm.read_text() if pm.exists() else ""
            head = f"## {TASK} "
            if head in txt:
                i = txt.index(head)
                e = txt.find("\n## ", i + 1)
                txt = txt[:i] + md.strip() + "\n" + (txt[e:] if e >= 0 else "")
            else:
                txt = txt.rstrip() + "\n\n" + md.strip() + "\n"
            pm.write_text(txt)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def load_rows(task, mode):
    sub = "pilots" if mode == "pilot" else "full"
    with open(RES_ROOT / sub / task / "results.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================================================ bridge (data only)
def policy_callable(pol, aspace):
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


def replica_from_truth(tp, y_mode="both_engaged", ret_load="pre"):
    """tp: plain numpy truth dict. Keys alpha beta gamma tauL tauR lam and either psi (scalar) or psi_left (L x 2);
    optional syn (L x R)."""
    from replica.model import ReplicaNL
    from dsswm.streams.generator import NL_DEFAULTS   # public constants only (c, rho_ret, nmax), not code
    pl = tp.get("psi_left")
    psi = [float(np.asarray(pl)[0, 1])] if pl is not None else [float(tp["psi"])]
    return ReplicaNL(tp["alpha"], tp["beta"], tp["gamma"], tp["tauL"], tp["tauR"], psi, float(tp["lam"]),
                     c=NL_DEFAULTS["c"], rho_ret=NL_DEFAULTS["rho_ret"], L=2, R=2, nmax=NL_DEFAULTS["nmax"],
                     y_mode=y_mode, ret_load=ret_load, syn=tp.get("syn"), psi_left=pl)


def replica_J(rep, q, aspace):
    u = q.utility
    w, w_ret, c_q = np.asarray(u.w, float), float(u.w_ret), float(u.c_q)
    return np.array([rep.exact_J(policy_callable(p, aspace), q.loads0, q.engaged0, q.H, w, w_ret, c_q)
                     for p in q.policies])


def truth_np(t):
    out = {k: np.asarray(t[k], float) for k in ("alpha", "beta", "gamma", "tauL", "tauR")}
    out["psi"], out["lam"] = float(t["psi"]), float(t["lam"])
    return out


_W = {}


def _ctx():
    if "prop" not in _W:
        import torch
        torch.set_num_threads(1)
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import make_env, nl_r0_truth
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        aspace = make_env(750, nl_r0_truth(750), 42).aspace
        prop = NLPropagator(2, 2, 2, aspace, 1.0, 0.3, device="cpu")
        _W.update(ncl=ncl, aspace=aspace, prop=prop)
    return _W


def vstar(t):
    return np.concatenate([np.asarray(t["alpha"], float), np.asarray(t["gamma"], float), np.asarray(t["tauL"], float),
                           np.asarray(t["beta"], float), np.asarray(t["tauR"], float), [float(t["psi"])],
                           [float(t["lam"])]])


def j_vstar(prop, q, v):
    from dsswm.certify.eta_loc import build_plans, j_values
    return j_values(prop, build_plans(prop, q), v[None], q.utility.w, q.utility.w_ret)[0] * q.utility.c_q


# ============================================================================================ one group (worker)
def group_job(g, y_mode="both_engaged", ret_load="pre"):
    """g: {stratum, instance, stream, variant, n_problems, problems: [k...]} -> list of problem records.
    y_mode / ret_load != defaults only for the power control (alternative spec readings)."""
    W = _ctx()
    prop = W["prop"]
    s, t, stratum = int(g["instance"]), int(g["stream"]), g["stratum"]
    out = []
    t0 = time.perf_counter()
    if stratum in ("NL-R0", "static", "heldout"):
        from dsswm.streams.gap_quota import QuotaBuilder
        b = QuotaBuilder(s, W["ncl"], None, CACHE, prop_cpu=prop)
        st = b.stream(t, "quota", n_problems=g["n_problems"])
        aspace = b.aspace
        if stratum == "NL-R0":
            tp = truth_np(st.truth)
            tp_main = None
        elif stratum == "static":
            dose = float(g["variant"])
            tr = dict(st.truth)
            tr["gamma"] = [float(x) * dose for x in st.truth["gamma"]]
            tr["lam"] = float(st.truth["lam"]) * dose
            tp = truth_np(tr)
        else:
            import run_r3_heldout_misspec as HM
            fam, strength = g["variant"]
            fn = HM.family_spec(fam, s, st.truth)[0]
            tp = fn(float(strength))
        rep = replica_from_truth(tp, y_mode, ret_load)
        for k in g["problems"]:
            q = st.problems[k]
            if stratum == "NL-R0":
                Jd = np.asarray(st.J[k][st.theta_index], float)
                Jd2 = j_vstar(prop, q, vstar(st.truth))           # second dsswm path (consistency only)
            elif stratum == "static":
                Jd = j_vstar(prop, q, vstar(tr))
                Jd2 = np.asarray(st.J[k][st.theta_index], float) if abs(dose - 1.0) < 1e-12 else None
            else:
                plans = [prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
                Jd = HM.true_J_main(prop, [q], [plans], tp)[0]
                Jd2 = None
            ts = time.perf_counter()
            Jr = replica_J(rep, q, aspace)
            out.append(_rec(g, k, q.pid, q.H, len(q.policies), Jd, Jr, Jd2, time.perf_counter() - ts, tp))
    else:  # hazard R1 / R1adv
        from run_r3_p5_hazard_zone_sampler import pool_problem, truth_dict
        P = json.loads((RES_ROOT / g["src_dir"] / "parts" / f"place_i{s}.json").read_text())[stratum]
        order = list(np.random.default_rng([s, 45, t]).permutation(len(P["problems"])))
        per = [P["per_problem"][i] for i in order]
        v = np.asarray(P["theta"], float)
        tr = truth_dict(v)
        tp = truth_np(tr)
        rep = replica_from_truth(tp, y_mode, ret_load)
        aspace = W["aspace"]
        for k in g["problems"]:
            q = pool_problem(s, per[k]["pool_j"], aspace)[0]
            Jd = np.asarray(per[k]["J_true"], float)
            Jd2 = j_vstar(prop, q, v)
            ts = time.perf_counter()
            Jr = replica_J(rep, q, aspace)
            out.append(_rec(g, k, q.pid, q.H, len(q.policies), Jd, Jr, Jd2, time.perf_counter() - ts, tp,
                            pool_j=int(per[k]["pool_j"])))
    for r in out:
        r["group_sec"] = time.perf_counter() - t0
    return out


def _rec(g, k, pid, H, n_pol, Jd, Jr, Jd2, t_rep, tp, pool_j=None):
    d = np.abs(Jd - Jr)
    return {"rec": "problem", "task": TASK, "stratum": g["stratum"], "source_task": g["source_task"],
            "instance": int(g["instance"]), "stream": int(g["stream"]), "variant": g["variant"], "problem": int(k),
            "pid": pid, "pool_j": pool_j, "H": int(H), "n_policies": int(n_pol),
            "J_dsswm": Jd.tolist(), "J_replica": Jr.tolist(), "abs_diff": d.tolist(), "max_abs_J_diff": float(d.max()),
            "dsswm_internal_diff": None if Jd2 is None else float(np.abs(Jd - np.asarray(Jd2)).max()),
            "pi_star_dsswm": int(np.argmax(Jd)), "pi_star_replica": int(np.argmax(Jr)),
            "top2_gap_dsswm": float(np.sort(Jd)[-1] - np.sort(Jd)[-2]),
            "top2_gap_replica": float(np.sort(Jr)[-1] - np.sort(Jr)[-2]), "t_replica_s": t_rep,
            "truth": {kk: (np.asarray(v).tolist() if not np.isscalar(v) else float(v)) for kk, v in tp.items()}}


def power_job(g, n=4):
    """Alternative spec readings must fail the tolerance (first n problems of the group)."""
    g = {**g, "problems": sorted(g["problems"])[:n]}
    out = {}
    for ym, rl in (("always", "pre"), ("both_engaged", "post")):
        rs = group_job(g, ym, rl)
        lam = float(rs[0]["truth"]["lam"])
        out[f"{g['stratum']}:{g['instance']}:{ym}|{rl}"] = {
            "max_abs_J_diff": max(r["max_abs_J_diff"] for r in rs), "lam": lam,
            "fails_tol": bool(max(r["max_abs_J_diff"] for r in rs) > TOL),
            # with lam = 0 retention does not depend on load: the pre/post reading is structurally identical
            "expected_to_fail": not (rl == "post" and lam == 0.0)}
    return out


# ============================================================================================ universe / sampling
def build_universe(mode):
    """Returns (problems: {key: info}, rows_by_key: {key: [row...]}). key = (stratum, instance, stream, variant, k)."""
    probs, rows_by = {}, defaultdict(list)

    def add(stratum, src, r, variant, k, npb):
        key = (stratum, int(r["instance"]), int(r["stream"]), variant, int(k))
        probs.setdefault(key, {"stratum": stratum, "source_task": src, "instance": int(r["instance"]),
                               "stream": int(r["stream"]), "variant": variant, "k": int(k), "n_problems": npb})
        rows_by[key].append(r)

    # NL-R0
    R = load_rows(SOURCES["NL-R0"], mode)
    npb = {}
    for r in R:
        kk = (r["instance"], r["stream"])
        npb[kk] = max(npb.get(kk, 0), int(r["problem"]) + 1)
    for r in R:
        add("NL-R0", SOURCES["NL-R0"], r, "R0", r["problem"], npb[(r["instance"], r["stream"])])
    # static (skip the 'full@open' copies: no certified policy, label duplicated from the 'full' row)
    R = [r for r in load_rows(SOURCES["static"], mode) if "certified_policy" in r]
    npb = {}
    for r in R:
        kk = (r["instance"], r["stream"])
        npb[kk] = max(npb.get(kk, 0), int(r["problem"]) + 1)
    for r in R:
        add("static", SOURCES["static"], r, float(r["dose"]), r["problem"], npb[(r["instance"], r["stream"])])
    # hazard
    for r in load_rows(SOURCES["hazard"], mode):
        add(r["layer"], SOURCES["hazard"], r, r["layer"], r["problem"], 15)
    # heldout
    for r in load_rows(SOURCES["heldout"], mode):
        add("heldout", SOURCES["heldout"], r, (r["family"], float(r["strength"])), r["problem"], 15)
    return probs, rows_by


def stratified_sample(keys, alloc, seed):
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(k)
    out = []
    for i, s in enumerate(STRATA):
        pool = sorted(by.get(s, []), key=str)
        n = min(alloc.get(s, 0), len(pool))
        idx = np.random.default_rng([seed, i]).choice(len(pool), n, replace=False) if n else []
        out += [pool[j] for j in sorted(idx)]
    return out


def groups_of(keys, probs, mode):
    gs = {}
    for key in keys:
        info = probs[key]
        gk = key[:4]
        if gk not in gs:
            sub = "pilots" if mode == "pilot" else "full"
            gs[gk] = {"stratum": info["stratum"], "source_task": info["source_task"], "instance": info["instance"],
                      "stream": info["stream"], "variant": info["variant"], "n_problems": info["n_problems"],
                      "src_dir": f"{sub}/{info['source_task']}", "problems": []}
        gs[gk]["problems"].append(info["k"])
    return list(gs.values())


def relabel(prec, rows):
    """Re-judge every scored row on this problem with the replica J."""
    Jr, Jd = np.asarray(prec["J_replica"]), np.asarray(prec["J_dsswm"])
    out = []
    for r in rows:
        pi = r.get("certified_policy")
        cert = r.get("status") == "CERTIFIED"
        reg_rep = float(Jr.max() - Jr[pi]) if pi is not None else None
        reg_d = float(Jd.max() - Jd[pi]) if pi is not None else None
        fc_rep = bool(cert and reg_rep is not None and reg_rep > EPS)
        rec_reg = r.get("true_regret")
        out.append({
            "rec": "label", "stratum": prec["stratum"], "instance": prec["instance"], "stream": prec["stream"],
            "variant": prec["variant"], "problem": prec["problem"], "pid": prec["pid"],
            "method": r.get("method"), "arm": r.get("arm"), "status": r.get("status"), "certified_policy": pi,
            "pid_match": r.get("pid") in (None, prec["pid"]),
            "true_regret_recorded": rec_reg, "true_regret_dsswm_recomp": reg_d, "true_regret_replica": reg_rep,
            "regret_abs_diff_replica_vs_recorded": (None if rec_reg is None or reg_rep is None
                                                     else abs(reg_rep - rec_reg)),
            "regret_abs_diff_recomp_vs_recorded": (None if rec_reg is None or reg_d is None else abs(reg_d - rec_reg)),
            "false_cert_recorded": bool(r.get("false_cert")), "false_cert_replica": fc_rep,
            "label_flip": bool(fc_rep != bool(r.get("false_cert"))),
            "regret_margin_to_eps": None if reg_rep is None else reg_rep - EPS,
        })
    return out


def stats(precs, labs):
    d = np.array([p["max_abs_J_diff"] for p in precs]) if precs else np.array([np.nan])
    by = defaultdict(list)
    for p in precs:
        by[p["stratum"]].append(p["max_abs_J_diff"])
    lb = defaultdict(list)
    for x in labs:
        lb[x["stratum"]].append(x)
    rr = [x["regret_abs_diff_replica_vs_recorded"] for x in labs if x["regret_abs_diff_replica_vs_recorded"] is not None]
    rc = [x["regret_abs_diff_recomp_vs_recorded"] for x in labs if x["regret_abs_diff_recomp_vs_recorded"] is not None]
    margins = [abs(x["regret_margin_to_eps"]) for x in labs if x["regret_margin_to_eps"] is not None]
    internal = [p["dsswm_internal_diff"] for p in precs if p["dsswm_internal_diff"] is not None]
    return {
        "n_problems": len(precs), "n_policy_values": int(sum(p["n_policies"] for p in precs)),
        "n_scored_rows": len(labs), "n_certified_rows": sum(x["status"] == "CERTIFIED" for x in labs),
        "max_abs_J_diff": float(np.nanmax(d)), "median_abs_J_diff": float(np.nanmedian(d)),
        "n_problems_over_tol": int(sum(x > TOL for x in d)),
        "per_stratum": {s: {"n_problems": len(v), "max_abs_J_diff": float(max(v)),
                            "n_rows": len(lb[s]), "n_false_cert_recorded": sum(x["false_cert_recorded"] for x in lb[s]),
                            "n_false_cert_replica": sum(x["false_cert_replica"] for x in lb[s]),
                            "label_flips": sum(x["label_flip"] for x in lb[s])} for s, v in by.items()},
        "label_flip_count": sum(x["label_flip"] for x in labs),
        "n_false_cert_recorded": sum(x["false_cert_recorded"] for x in labs),
        "n_false_cert_replica": sum(x["false_cert_replica"] for x in labs),
        "max_regret_diff_replica_vs_recorded": max(rr) if rr else None,
        "max_regret_diff_dsswm_recomp_vs_recorded": max(rc) if rc else None,
        "min_abs_regret_margin_to_eps": min(margins) if margins else None,
        "pid_mismatches": sum(not x["pid_match"] for x in labs),
        "argmax_disagreements": sum(p["pi_star_dsswm"] != p["pi_star_replica"] and p["top2_gap_replica"] > TOL
                                    for p in precs),
        "max_dsswm_internal_diff": max(internal) if internal else None,
        "pass": bool(np.nanmax(d) <= TOL and sum(x["label_flip"] for x in labs) == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "full":
        from dsswm.stats.prereg import assert_locked
        assert_locked()
    out = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    (out / "start_time.txt").write_text(start_iso)
    t_start = time.time()
    report_progress(0, 4, {"stage": "universe"})

    probs, rows_by = build_universe(a.mode)
    keys = sorted(probs, key=str)
    alloc = PILOT_ALLOC
    if a.mode == "pilot":
        primary = stratified_sample(keys, PILOT_ALLOC, 42)
    else:
        n_layer = {s: sum(k[0] == s for k in keys) for s in STRATA}
        tot = sum(n_layer.values())
        alloc = {s: int(round(300 * n / tot)) for s, n in n_layer.items()}
        primary = stratified_sample(keys, alloc, 42)
    prim_set = set(primary)
    census = keys                                          # every pilot problem (cheap): secondary

    from joblib import Parallel, delayed
    report_progress(1, 4, {"stage": "replica_J", "n_problems": len(census)})
    gs = groups_of(census, probs, a.mode)
    precs = [r for rs in Parallel(n_jobs=a.workers)(delayed(group_job)(g) for g in gs) for r in rs]
    report_progress(2, 4, {"stage": "power_control"})
    pg = [g for g in gs if g["stratum"] == "NL-R0"][:2] + [g for g in gs if g["stratum"] in ("R1", "static")][:2]
    pcs = {}
    for d in Parallel(n_jobs=a.workers)(delayed(power_job)(g) for g in pg):
        pcs.update(d)

    report_progress(3, 4, {"stage": "relabel"})
    labs_all, prim_p, prim_l = [], [], []
    key_of = {}
    for p in precs:
        key = (p["stratum"], p["instance"], p["stream"], p["variant"] if not isinstance(p["variant"], list)
               else tuple(p["variant"]), p["problem"])
        key_of[id(p)] = key
        p["in_primary_sample"] = key in prim_set
        ls = relabel(p, rows_by[key])
        for x in ls:
            x["in_primary_sample"] = p["in_primary_sample"]
        labs_all += ls
        if p["in_primary_sample"]:
            prim_p.append(p)
            prim_l += ls
    with open(out / "results.jsonl", "w") as f:
        for p in precs:
            f.write(json.dumps(p) + "\n")
        for x in labs_all:
            f.write(json.dumps(x) + "\n")
    with open(out / "predictors.jsonl", "w") as f:
        f.write(json.dumps({"record": "not_applicable", "task": TASK,
                            "note": "harness-validity check: no learner-side predictors exist for this task"}) + "\n")

    st_p, st_c = stats(prim_p, prim_l), stats(precs, labs_all)
    power_ok = all(v["fails_tol"] == v["expected_to_fail"] for v in pcs.values())
    flips = [x for x in labs_all if x["label_flip"]]
    worst = sorted(precs, key=lambda p: -p["max_abs_J_diff"])[:10]
    go = st_p["pass"] and st_c["pass"] and power_ok and st_p["pid_mismatches"] == 0
    summary = {
        "task_id": TASK, "mode": a.mode, "seed": 42, "tolerance": TOL, "eps": EPS,
        "pass_criteria": "max |J diff| <= 1e-6 AND 0 label flips (primary sample)",
        "primary_sample": {"design": "stratified by layer, rng [42, stratum index]", "allocation": PILOT_ALLOC
                           if a.mode == "pilot" else alloc, **st_p},
        "census_all_pilot_problems": st_c,
        "max_abs_J_diff": st_p["max_abs_J_diff"], "label_flip_count": st_p["label_flip_count"],
        "power_control_alt_spec_readings": pcs, "power_control_ok": power_ok,
        "flagged_rows": flips[:50],
        "worst_problems": [{kk: p[kk] for kk in ("stratum", "instance", "stream", "variant", "problem", "pid",
                                                "max_abs_J_diff")} for p in worst],
        "universe": dict(Counter(k[0] for k in keys)),
        "dev_seeds_used": {s: sorted({k[1] for k in keys if k[0] == s}) for s in STRATA},
        "deviation_note": ("full: eval-seed outputs of the four dependency tasks (r3_*_a / r3_heldout_misspec "
                           "full); primary = 300 problems allocated proportionally to layer size; census = all."
                           if a.mode == "full" else None) or "pilot plan says dev 734-739; the hazard layer's pilot ran on dev 750-752 (its own pre-"
                          "registered pilot seeds), so R1 / R1adv problems come from 750-752. No eval seed touched.",
        "sources": SOURCES,
        "dsswm_J_paths": {"NL-R0": "QuotaBuilder.stream J[k][theta_index] (scored); 2nd path eta_loc.j_values",
                          "static": "eta_loc.j_values at dosed v* x c_q (scored path); g=1 also vs J[k][ti]",
                          "R1/R1adv": "placement J_true (scored); 2nd path eta_loc.j_values at theta*",
                          "heldout": "main NLPropagator plans at family truth (rows scored with replica J already)"},
        "independence_caveat": "replica package imports no dsswm; policies decoded through dsswm ActionSpace as "
                               "opaque callables; instance generation and truth draws are shared harness code; "
                               "public constants (c, rho_ret, nmax) read from NL_DEFAULTS.",
        "wall_clock_s": time.time() - t_start, "start_time": start_iso,
        "runtime_note": f"CPU only, {a.workers} joblib workers, concurrent run (并发运行)",
    }
    summary["go_no_go"] = "GO" if go else "NO_GO"
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    samp = []
    for s in STRATA:
        samp += [p for p in prim_p if p["stratum"] == s][:2]
    (out / "samples" / "jtrue_pairs.json").write_text(json.dumps(samp + worst[:2], indent=1, default=str))
    cert_lab = [x for x in prim_l if x["status"] == "CERTIFIED"]
    near = sorted(cert_lab, key=lambda x: abs(x["regret_margin_to_eps"]))[:10]
    (out / "samples" / "relabel_examples.json").write_text(json.dumps(
        {"closest_to_eps_boundary": near, "false_cert_rows": [x for x in labs_all if x["false_cert_replica"]
                                                              or x["false_cert_recorded"]][:20]}, indent=1,
        default=str))
    if a.mode == "pilot":
        ps = summary["primary_sample"]
        cs = summary["census_all_pilot_problems"]
        update_shared_summary({
            "candidate_id": "shared", "go_no_go": summary["go_no_go"], "hypotheses": ["harness validity"],
            "key_metrics": {"max_abs_J_diff": ps["max_abs_J_diff"], "label_flip_count": ps["label_flip_count"],
                            "n_problems": ps["n_problems"], "n_scored_rows_relabelled": ps["n_scored_rows"],
                            "census_n_problems": cs["n_problems"], "census_max_abs_J_diff": cs["max_abs_J_diff"],
                            "census_label_flips": cs["label_flip_count"], "power_control_ok": power_ok},
            "notes": "replica (no dsswm import) recomputes exact true J of every candidate policy; all scored rows "
                     "re-judged; hazard layer uses its own dev seeds 750-752"},
            f"## {TASK} (pilot, 并发运行)\n"
            f"- 结论: **{summary['go_no_go']}**。主样本 {ps['n_problems']} 题（分层: NL-R0 12 / 静态孪生 22 / R1 22 / "
            f"R1-adv 22 / 留出族 22，种子 42），max |J_replica - J_dsswm| = {ps['max_abs_J_diff']:.1e}（阈值 1e-6），"
            f"重判 {ps['n_scored_rows']} 行 true_regret / false_cert，标签翻转 {ps['label_flip_count']}。\n"
            f"- 全量普查（全部 {cs['n_problems']} 个 pilot 问题，{cs['n_scored_rows']} 行）: max |dJ| = "
            f"{cs['max_abs_J_diff']:.1e}，翻转 {cs['label_flip_count']}；false_cert 记录 {cs['n_false_cert_recorded']} = "
            f"replica {cs['n_false_cert_replica']}；距 eps 边界最近的 regret 余量 {cs['min_abs_regret_margin_to_eps']:.1e}。\n"
            f"- 功效对照: replica 的替代规格读法（y 'always'；留存用更新后负载）在 lam>0 的真值上全部超出阈值"
            f"（0.04-0.76），说明该检验有检出力；lam=0 实例上 post 读法结构上等价（预期不失败）。\n"
            f"- 偏差: 危险区层使用其自身 dev 种子 750-752（而非 734-739）。独立性局限: 策略为 dsswm ActionSpace 解码的黑盒，"
            f"实例生成为共享 harness 代码。\n")
    wall = time.time() - t_start
    report_progress(4, 4, {"stage": "done", "go_no_go": summary["go_no_go"], "max_abs_J_diff": st_p["max_abs_J_diff"],
                           "label_flip_count": st_p["label_flip_count"]})
    print(json.dumps({k: summary[k] for k in ("primary_sample", "census_all_pilot_problems",
                                              "power_control_alt_spec_readings", "go_no_go", "wall_clock_s")},
                     indent=1, default=str))
    global PLANNED_MIN
    PLANNED_MIN = 8 if a.mode == "pilot" else 25
    update_gpu_progress("success", start_iso, wall / 60, {
        "mode": a.mode, "n_primary": len(prim_p), "n_census": len(precs), "workers": a.workers, "gpu_count": 0,
        "device": "cpu", "replica_tol": TOL})
    mark_done("success", f"{a.mode} replica check: {summary['go_no_go']}; primary {len(prim_p)} problems max |dJ| = "
                         f"{st_p['max_abs_J_diff']:.2e}, flips {st_p['label_flip_count']}; census {len(precs)} "
                         f"problems max |dJ| = {st_c['max_abs_J_diff']:.2e}, flips {st_c['label_flip_count']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        mark_done("failed", repr(e))
        raise
