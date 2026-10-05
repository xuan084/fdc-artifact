"""r2_setup_falsify: round-1 falsification-layer infrastructure (methodology 4.2 / 2.4 / 2.5) -- checks and manifests.

Usage: run_r2_setup_falsify.py --mode {pilot,full} [--workers 4]

pilot (dev seeds 600-699 only):
  T  unit tests  dsswm/tests/test_falsify_layer.py + test_no_truth_import.py (pytest subprocess)
  FA in-class false alarm: 100 dev instances (natural kappa generator, dynamic in-class E1-NL-S learner,
     |Theta| = 13824) x 10 noise streams = 1000 streams; n0 = 20 shared rounds + N_FA uniform legal rounds, public
     problem id every 40 rounds. Main arm = {pool, reg(load bucket), ext} (1/3 each). Secondary arms on streams
     0-1 of every instance (200 streams): reg = problem id, reg = placebo, 1/2-1/2 {pool, ext}.
     MODEL_CONFLICT rate with Clopper-Pearson upper bound; theta* coverage; min_t (log M_mixed - log M_pool).
  SP static-class power preview (HF2 direction only, not a gate): 20 kappa_high dev instances, static learner
     (gamma = lam = 0 on the same grid), 1 stream, arms main / placebo / problem id / half / reg-only (load bucket,
     placebo, problem id) / ext-only. NB the placebo replaces only the reg component, so with ext present the
     placebo arm inherits ext's detections; the reg-only arms isolate the regime contrast.
  M4 m4 calibration: 8 kappa_high dev instances x K = 15 problems; slope s such that eta_min = 2 eps over the
     problem stream (nominal schedule t_k = n0 + 200 k, T_ref = 1000).
  OOS Lin non-near-tie OOS family: squeeze + newpart, 5 dev instances x 10 problems each (100 candidate problems);
     HiGHS truth V*, keep V* >= 1.2 eps.
  VJE cand_g smoke: in-class change at a known boundary (k_true in {0,1}, n_old in {20, 200}), 2 instances:
     coverage of the true versioned pair, |Theta_new| for VJE / detect-reset / naive reuse.
  EP e-process smoke: 4 instances, model = old-epoch MLE, truth in-class vs m1 synergy (strength 2.5): alarm steps.
full: eval seeds 10000-10047: Lin OOS family manifests and m4 calibrations for every eval instance.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import fcntl
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.audit.eprocess import EProcessAudit  # noqa: E402
from dsswm.envs.mis import M4_NOMINAL_STEPS, M4_T_REF, calibrate_m4, m4_schedule  # noqa: E402
from dsswm.envs.nl import NLEnv  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import PlaceboRegime, load_bucket, problem_id_regime  # noqa: E402
from dsswm.evidence.versioned import DetectReset, VersionedJointLR  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, make_nl_instance, nl_class_max_jtable  # noqa: E402
from dsswm.streams.lin_oos import LIN_OOS_EPS, family_manifest  # noqa: E402

TASK = "r2_setup_falsify"
EPS = 0.02
DELTA = 0.05
NMAX, C_KNOWN, RHO_RET = NL_DEFAULTS["nmax"], NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"]
N_FA = 480                    # uniform legal rounds per false-alarm stream (after the n0 = 20 shared rounds)
PID_EVERY = 40
PLACEBO_FREQS = (0.45, 0.35, 0.20)
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables_f1"       # G_1 = DEFAULT_NL_S_GRID class tables (truth-free)
K_M4 = 15


# ------------------------------------------------------------------ scheduler protocol
def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


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
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def update_gpu_progress(status, start_iso, wall_min, snapshot, planned):
    p = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": planned, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))


def cp_lower(k, n, a=0.05):
    return 0.0 if k <= 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))


# ------------------------------------------------------------------ shared context (per worker)
_CTX = {}


def ctx():
    if "ncl" not in _CTX:
        torch.set_num_threads(1)
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        P = ncl.torch_params()
        Ps = {k: v.clone() for k, v in P.items()}
        Ps["gamma"] = Ps["gamma"] * 0
        Ps["lam"] = Ps["lam"] * 0
        _CTX.update(ncl=ncl, P=P, Ps=Ps)
    return _CTX


def class_tables(aspace, static=False):
    c = ctx()
    key = ("LT", static)
    if key not in c:
        prop = NLPropagator(2, 2, NMAX, aspace, C_KNOWN, RHO_RET, device="cpu")
        LT, EY = prop.tables(c["Ps"] if static else c["P"])
        c[key] = (prop, LT, EY)
    return c[key]


def arm_set(arm, prop, LT, seed):
    if arm == "main":
        return MixedLRSet(prop, LT, DELTA, regime_fn=load_bucket, name="main")
    if arm == "pid":
        return MixedLRSet(prop, LT, DELTA, regime_fn=problem_id_regime, name="pid")
    if arm == "placebo":
        return MixedLRSet(prop, LT, DELTA, regime_fn=PlaceboRegime(seed, PLACEBO_FREQS), name="placebo")
    if arm == "half":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "ext"), name="half")
    if arm == "reg_only":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "reg"), name="reg_only")
    if arm == "reg_only_placebo":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "reg"), regime_fn=PlaceboRegime(seed, PLACEBO_FREQS),
                          name="reg_only_placebo")
    if arm == "reg_only_pid":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "reg"), regime_fn=problem_id_regime,
                          name="reg_only_pid")
    if arm == "ext_only":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "ext"), weights=(0.5, 0.5), name="ext_only")
    raise ValueError(arm)


def run_stream(inst, arms, n_steps, act_seed, static=False, record_trace=False):
    """One CRN stream: every arm sees the same real rounds (one platform)."""
    c = ctx()
    prop, LT, _ = class_tables(inst.env.aspace, static)
    sets = {a: arm_set(a, prop, LT, inst.seed) for a in arms}
    pool = SeqLRSet(prop, LT, DELTA)
    h = inst.env.handle()
    rng = np.random.default_rng(act_seed)
    trace = []
    rounds = [(o, None) for o in inst.init_obs]
    t0 = time.perf_counter()
    for s in range(n_steps):
        rounds.append((h.step(int(rng.integers(h.aspace.n))), f"q{s // PID_EVERY}"))
    for r, (o, pid) in enumerate(rounds):
        pool.update(o)
        for ms in sets.values():
            ms.update(o, problem_id=pid)
        if record_trace and (r % 50 == 0 or r == len(rounds) - 1):
            trace.append({"round": r + 1, "pool_size": pool.size(),
                          **{f"{a}_size": ms.size() for a, ms in sets.items()},
                          **{f"{a}_adv": {k: round(v, 2) for k, v in ms.component_advantage().items()}
                             for a, ms in sets.items()}})
    assert h.n_steps == len(inst.init_obs) + n_steps
    ti = inst.truth.get("theta_index") if not static else None
    out = {}
    for a, ms in sets.items():
        out[a] = {"conflict": ms.conflict_round is not None, "conflict_round": ms.conflict_round,
                  "size": ms.size(), "pool_size": pool.size(),
                  "theta_star_in_set": (bool(ms.mask()[ti]) if ti is not None else None),
                  "theta_star_in_pool": (bool(pool.mask()[ti]) if ti is not None else None),
                  "min_gap_vs_pool": ms.min_gap_vs_pool, "log_w_pool": float(ms.log_w[0]),
                  "component_adv": ms.component_advantage(), "scope": ms.falsification_scope(),
                  "ext_state": ms.ext.state() if ms.ext is not None else None}
    return out, trace, time.perf_counter() - t0


# ------------------------------------------------------------------ jobs
def fa_job(seed, n_streams, n_secondary):
    rows, samples, errs = [], [], []
    ncl = ctx()["ncl"]
    t_job = time.time()
    for k in range(n_streams):
        noise = 42 + 1000 * k
        try:
            inst = make_nl_instance(seed, noise_seed=noise, nl_class=ncl, kappa_mode="natural", K=1)
            arms = ["main"] + (["pid", "placebo", "half"] if k < n_secondary else [])
            res, trace, wall = run_stream(inst, arms, N_FA, [seed, k, 77], record_trace=(k == 0 and seed < 602))
            for a, r in res.items():
                rows.append({"block": "FA", "instance": seed, "stream": k, "noise_seed": noise, "arm": a,
                             "kappa_mode": "natural", "learner": "dynamic_in_class", "n_rounds": 20 + N_FA,
                             "status": "MODEL_CONFLICT" if r["conflict"] else "OK",
                             "wall_clock_s": wall / len(arms), **r})
            if trace:
                samples.append({"block": "FA", "instance": seed, "stream": k, "truth": inst.truth, "trace": trace})
        except Exception as e:  # noqa: BLE001
            errs.append(f"FA {seed}/{k}: {e!r} {traceback.format_exc()[-400:]}")
    return {"rows": rows, "samples": samples, "errors": errs, "wall_s": time.time() - t_job, "kind": "FA",
            "seed": seed}


def sp_job(seed):
    rows, samples, errs = [], [], []
    t_job = time.time()
    try:
        inst = make_nl_instance(seed, noise_seed=42, nl_class=ctx()["ncl"], kappa_mode="high", K=1)
        arms = ["main", "placebo", "pid", "half", "reg_only", "reg_only_placebo", "reg_only_pid", "ext_only"]
        res, trace, wall = run_stream(inst, arms, 980, [seed, 0, 78], static=True, record_trace=(seed < 602))
        for a, r in res.items():
            rows.append({"block": "SP", "instance": seed, "stream": 0, "arm": a, "kappa_mode": "high",
                         "learner": "static_same_grid", "n_rounds": 1000,
                         "status": "MODEL_CONFLICT" if r["conflict"] else "OK", "wall_clock_s": wall / len(arms), **r})
        if trace:
            samples.append({"block": "SP", "instance": seed, "truth": inst.truth, "trace": trace})
    except Exception as e:  # noqa: BLE001
        errs.append(f"SP {seed}: {e!r} {traceback.format_exc()[-400:]}")
    return {"rows": rows, "samples": samples, "errors": errs, "wall_s": time.time() - t_job, "kind": "SP",
            "seed": seed}


def class_J(q, prop, P):
    """Class J table on G_1 (shared truth-free cache exp/cache/jtables_f1/<pid>_raw.npy, atomic write)."""
    path = JT_CACHE / f"{q.pid}_raw.npy"
    if path.exists():
        raw = np.load(path)
        if raw.shape == (P["alpha"].shape[0], len(q.policies)):
            return nl_class_max_jtable(q, prop, P, cache_path=str(path))
    J = nl_class_max_jtable(q, prop, P)
    raw = J / q.utility.c_q
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npy")
    np.save(tmp, raw)
    os.replace(tmp, path)
    return J


def m4_job(seed, K=K_M4):
    t_job = time.time()
    try:
        c = ctx()
        inst = make_nl_instance(seed, noise_seed=42, nl_class=c["ncl"], kappa_mode="high", K=K)
        prop, LT, EY = class_tables(inst.env.aspace)
        J = [class_J(q, prop, c["P"]) for q in inst.problems]
        base = inst.env.true_params()
        plans = [[prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] for q in inst.problems]
        cache = {}

        def J_true(k, shift):
            key = round(float(shift), 12)
            if key not in cache:
                tp = dict(base)
                tp["alpha"] = base["alpha"] + shift
                cache[key] = prop.tables(params_to_torch(tp, device="cpu"))
            LTt, EYt = cache[key]
            u = inst.problems[k].utility
            return np.array([float(prop._run_plan(pl, es, LTt, EYt, u.w, u.w_ret, u.c_q)[0]) for pl, es in plans[k]])

        sched = m4_schedule(NL_DEFAULTS["n0"], K)
        cal = calibrate_m4(J, J_true, 2 * EPS, sched)
        # sanity: in-class (s = 0) eta_min is exactly 0 because theta* is on the grid (kappa_high generator)
        row = {"block": "M4", "instance": seed, "K": K, "eta_target": 2 * EPS, "T_ref": M4_T_REF,
               "nominal_steps": M4_NOMINAL_STEPS, "schedule": sched, **cal,
               "max_alpha_shift": cal["s"] * sched[-1] / M4_T_REF,
               "max_within_horizon_shift": cal["s"] * max(q.H for q in inst.problems) / M4_T_REF,
               "truth": inst.truth, "wall_clock_s": time.time() - t_job}
        return {"rows": [row], "samples": [], "errors": [], "wall_s": time.time() - t_job, "kind": "M4", "seed": seed}
    except Exception as e:  # noqa: BLE001
        return {"rows": [], "samples": [], "errors": [f"M4 {seed}: {e!r} {traceback.format_exc()[-600:]}"],
                "wall_s": time.time() - t_job, "kind": "M4", "seed": seed}


def oos_job(family, seeds):
    t_job = time.time()
    try:
        m = family_manifest(family, seeds)
        for r in m:
            r["block"] = "OOS"
        return {"rows": m, "samples": [], "errors": [], "wall_s": time.time() - t_job, "kind": "OOS", "seed": family}
    except Exception as e:  # noqa: BLE001
        return {"rows": [], "samples": [], "errors": [f"OOS {family}: {e!r} {traceback.format_exc()[-600:]}"],
                "wall_s": time.time() - t_job, "kind": "OOS", "seed": family}


def vje_job(seed, n_old, k_true, n_new=300):
    t_job = time.time()
    try:
        c = ctx()
        ncl = c["ncl"]
        inst = make_nl_instance(seed, noise_seed=42, nl_class=ncl, kappa_mode="high", K=1)
        prop, LT, _ = class_tables(inst.env.aspace)
        t = inst.truth
        vj = VersionedJointLR(ncl, prop, LT, DELTA)
        naive = SeqLRSet(prop, LT, DELTA)
        dr = DetectReset(lambda: MixedLRSet(prop, LT, DELTA))
        h = inst.env.handle()
        rng = np.random.default_rng([seed, n_old, k_true, 5])
        old = list(inst.init_obs)[:n_old] + [h.step(int(rng.integers(h.aspace.n))) for _ in range(max(0, n_old - 20))]
        for o in old:
            vj.update(o); naive.update(o); dr.update(o)
        vj.start_new_epoch()
        beta_new = list(t["beta"])
        if k_true == 1:
            beta_new[1] = -beta_new[1]
        env2 = NLEnv(t["alpha"], beta_new, t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"],
                     seed=int(seed) * 1000 + 7)
        h2 = env2.handle()
        for _ in range(n_new):
            o = h2.step(int(rng.integers(h2.aspace.n)))
            vj.update(o); naive.update(o); dr.update(o)
        ti_old = t["theta_index"]
        ti_new = ncl.index_of(tuple(t["alpha"]), tuple(beta_new), tuple(t["gamma"]), tuple(t["tauL"]),
                              tuple(t["tauR"]), t["psi"], t["lam"])
        row = {"block": "VJE", "instance": seed, "n_old": n_old, "k_true": k_true, "n_new": n_new,
               "vje_covers_true_pair": vj.contains(ti_old, ti_new), "vje_new_set_size": vj.size(),
               "vje_theta_new_in_set": bool(vj.new_mask()[ti_new]), "vje_changed_slots": vj.changed_slots_alive(),
               "naive_size": naive.size(), "naive_theta_new_in_set": bool(naive.mask()[ti_new]),
               "detect_reset_alarms": dr.alarms, "detect_reset_size": dr.size(),
               "detect_reset_theta_new_in_set": bool(dr.mask()[ti_new]), "V": vj.V,
               "wall_clock_s": time.time() - t_job}
        return {"rows": [row], "samples": [], "errors": [], "wall_s": time.time() - t_job, "kind": "VJE", "seed": seed}
    except Exception as e:  # noqa: BLE001
        return {"rows": [], "samples": [], "errors": [f"VJE {seed}: {e!r} {traceback.format_exc()[-600:]}"],
                "wall_s": time.time() - t_job, "kind": "VJE", "seed": seed}


def ep_job(seed, strength, budget=3000):
    t_job = time.time()
    try:
        c = ctx()
        inst = make_nl_instance(seed, noise_seed=42, nl_class=c["ncl"], kappa_mode="high", K=1)
        prop, LT, EY = class_tables(inst.env.aspace)
        t = inst.truth
        S = np.random.default_rng([seed, 61]).standard_normal((2, 2))
        S = S - S.mean(1, keepdims=True); S = S - S.mean(0, keepdims=True); S = S / np.abs(S).max()
        env = NLEnv(t["alpha"], t["beta"], t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"],
                    seed=int(seed) * 1000 + 42, syn=strength * S)
        h = env.handle()
        rng = np.random.default_rng([seed, 9])
        lr = SeqLRSet(prop, LT, DELTA)
        for _ in range(200):
            lr.update(h.step(int(rng.integers(h.aspace.n))))
        q = inst.problems[0]
        k = lr.mle()
        aud = EProcessAudit(prop, LT[k].numpy(), EY[k].numpy(), q.policies[:3], q.H, q.utility, eta=2 * EPS,
                            delta_audit=DELTA)
        n0 = h.n_steps
        res = aud.run(h, budget)
        assert h.n_steps - n0 == res["steps"]
        row = {"block": "EP", "instance": seed, "m1_strength": strength, "model_theta": k,
               "model_is_theta_star": bool(k == t["theta_index"]), "audit_status": res["status"],
               "audit_steps": res["steps"], "log_e": res["log_e"], "residual_range": [aud.lo, aud.hi],
               "eta": 2 * EPS, "wall_clock_s": time.time() - t_job}
        return {"rows": [row], "samples": [], "errors": [], "wall_s": time.time() - t_job, "kind": "EP", "seed": seed}
    except Exception as e:  # noqa: BLE001
        return {"rows": [], "samples": [], "errors": [f"EP {seed}: {e!r} {traceback.format_exc()[-600:]}"],
                "wall_s": time.time() - t_job, "kind": "EP", "seed": seed}


# ------------------------------------------------------------------ analysis
def analyse(rows, tests, mode, wall):
    out = {"task_id": TASK, "mode": mode, "wall_clock_s": wall, "note": "并发运行（4 槽并行，每任务 4 worker）",
           "eps": EPS, "delta": DELTA}
    out["unit_tests"] = tests
    fa = [r for r in rows if r["block"] == "FA"]
    if fa:
        arms = {}
        for a in ("main", "pid", "placebo", "half"):
            rr = [r for r in fa if r["arm"] == a]
            if not rr:
                continue
            k = sum(r["conflict"] for r in rr)
            cov = sum(bool(r["theta_star_in_set"]) for r in rr)
            covp = sum(bool(r["theta_star_in_pool"]) for r in rr)
            infl = [r["size"] / max(r["pool_size"], 1) for r in rr]
            arms[a] = {"n_streams": len(rr), "model_conflict": k, "false_alarm_rate": k / len(rr),
                       "false_alarm_cp_upper": cp_upper(k, len(rr)), "theta_star_coverage": cov / len(rr),
                       "theta_star_coverage_pool": covp / len(rr),
                       "min_gap_vs_pool": float(min(r["min_gap_vs_pool"] for r in rr)),
                       "gap_bound": float(rr[0]["log_w_pool"]),
                       "gap_bound_holds": bool(all(r["min_gap_vs_pool"] >= r["log_w_pool"] - 1e-9 for r in rr)),
                       "set_size_ratio_vs_pool_median": float(np.median(infl)),
                       "set_size_ratio_vs_pool_p90": float(np.quantile(infl, 0.9)),
                       "component_adv_median": {c: float(np.median([r["component_adv"][c] for r in rr]))
                                                for c in rr[0]["component_adv"]}}
        out["inclass_false_alarm"] = arms
    sp = [r for r in rows if r["block"] == "SP"]
    if sp:
        d = {}
        for a in sorted({r["arm"] for r in sp}):
            rr = [r for r in sp if r["arm"] == a]
            k = sum(r["conflict"] for r in rr)
            cr = [r["conflict_round"] for r in rr if r["conflict"]]
            d[a] = {"n": len(rr), "detected": k, "rate": k / len(rr), "cp_lower": cp_lower(k, len(rr)),
                    "alarm_round_median": float(np.median(cr)) if cr else None}
        out["static_power_preview"] = d
    m4 = [r for r in rows if r["block"] == "M4"]
    if m4:
        out["m4_calibration"] = {"n_instances": len(m4), "reached": int(sum(r["reached"] for r in m4)),
                                 "reached_frac": float(np.mean([r["reached"] for r in m4])),
                                 "s_median": float(np.median([r["s"] for r in m4])),
                                 "s_range": [float(min(r["s"] for r in m4)), float(max(r["s"] for r in m4))],
                                 "eta_min_at_s0_max": float(max(r["eta_min_at_0"] for r in m4)),
                                 "max_alpha_shift_median": float(np.median([r["max_alpha_shift"] for r in m4])),
                                 "per_instance": {str(r["instance"]): {"s": r["s"], "eta_min": r["eta_min"],
                                                                      "reached": r["reached"]} for r in m4}}
    oos = [r for r in rows if r["block"] == "OOS"]
    if oos:
        fam = {}
        for f in ("squeeze", "newpart"):
            rr = [r for r in oos if r["family"] == f]
            if not rr:
                continue
            fam[f] = {"n_candidates": len(rr), "true_oos": int(sum(r["true_label"] == "OUT_OF_SCOPE" for r in rr)),
                      "near_tie": int(sum(r["near_tie"] for r in rr)), "kept": int(sum(r["keep"] for r in rr)),
                      "kept_aligned": int(sum(r["keep"] and r["aligned"] for r in rr)),
                      "V_star_kept_median": float(np.median([r["V_star"] for r in rr if r["keep"]]))
                      if any(r["keep"] for r in rr) else None}
        out["lin_oos"] = {"families": fam, "n_candidates": len(oos), "kept_total": int(sum(r["keep"] for r in oos)),
                          "eps": LIN_OOS_EPS}
    vje = [r for r in rows if r["block"] == "VJE"]
    if vje:
        out["vje_smoke"] = [{k: r[k] for k in ("instance", "n_old", "k_true", "vje_covers_true_pair",
                                               "vje_new_set_size", "naive_size", "naive_theta_new_in_set",
                                               "detect_reset_alarms", "detect_reset_size")} for r in vje]
    ep = [r for r in rows if r["block"] == "EP"]
    if ep:
        out["eprocess_smoke"] = [{k: r[k] for k in ("instance", "m1_strength", "model_is_theta_star",
                                                    "audit_status", "audit_steps")} for r in ep]
    # ---- gate
    g = {}
    g["unit_tests_passed"] = bool(tests.get("failed", 1) == 0 and tests.get("passed", 0) > 0)
    if fa:
        mm = out["inclass_false_alarm"]["main"]
        g["inclass_false_alarm_cp_upper"] = mm["false_alarm_cp_upper"]
        g["inclass_false_alarm_ok"] = bool(mm["false_alarm_cp_upper"] <= DELTA)
        g["log3_bound_holds"] = bool(mm["gap_bound_holds"])
    if m4:
        g["m4_calibration_reached"] = out["m4_calibration"]["reached_frac"]
    if oos:
        g["lin_oos_cases_generated"] = out["lin_oos"]["kept_total"]
        g["lin_oos_ok"] = bool(out["lin_oos"]["kept_total"] >= 10)
    if mode == "pilot":
        g["pass"] = bool(g["unit_tests_passed"] and g.get("inclass_false_alarm_ok") and g.get("lin_oos_ok"))
    else:
        g["pass"] = bool(oos and m4 and out["m4_calibration"]["reached_frac"] >= 0.9)
    out["gate"] = g
    out["go_no_go"] = "GO" if g["pass"] else "NO_GO"
    return out


def run_unit_tests(out_dir):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "dsswm/tests/test_falsify_layer.py",
           "dsswm/tests/test_no_truth_import.py"]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=env, timeout=1200)
    (out_dir / "unit_tests.txt").write_text(p.stdout + "\n" + p.stderr)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    import re
    passed = int(re.search(r"(\d+) passed", tail).group(1)) if re.search(r"(\d+) passed", tail) else 0
    failed = int(re.search(r"(\d+) failed", tail).group(1)) if re.search(r"(\d+) failed", tail) else 0
    errors = int(re.search(r"(\d+) error", tail).group(1)) if re.search(r"(\d+) error", tail) else 0
    return {"passed": passed, "failed": failed + errors, "returncode": p.returncode, "summary_line": tail,
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fa-instances", type=int, default=100)
    ap.add_argument("--fa-streams", type=int, default=10)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-progress-update", action="store_true")
    ap.add_argument("--test-out", default=None, help="smoke run: write here, no scheduler markers")
    ap.add_argument("--m4-instances", type=int, default=8)
    ap.add_argument("--sp-instances", type=int, default=20)
    a = ap.parse_args()
    start = time.time(); start_iso = datetime.now().isoformat()
    markers = a.test_out is None
    if markers:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    else:
        global progress
        progress = lambda *args, **kw: None  # noqa: E731
    out_dir = Path(a.test_out) if a.test_out else RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    JT_CACHE.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "run.log", "a")

    def L(msg):
        log.write(f"[{datetime.now().isoformat()}] {msg}\n"); log.flush()

    L(f"start mode={a.mode}")
    progress(0, 1, "unit_tests")
    tests = {"passed": 0, "failed": 1, "summary_line": "skipped"} if a.skip_tests else run_unit_tests(out_dir)
    L(f"unit tests: {tests.get('summary_line')}")
    jobs = []
    if a.mode == "pilot":
        dev = list(range(600, 700))
        jobs += [delayed(m4_job)(s) for s in range(600, 600 + a.m4_instances)]
        jobs += [delayed(oos_job)(f, list(range(600, 605))) for f in ("squeeze", "newpart")]
        jobs += [delayed(vje_job)(s, n_old, k) for s in (600, 601) for n_old in (20, 200) for k in (0, 1)]
        jobs += [delayed(ep_job)(s, st) for s in (600, 601) for st in (0.0, 2.5)]
        jobs += [delayed(sp_job)(s) for s in range(600, 600 + a.sp_instances)]
        jobs += [delayed(fa_job)(s, a.fa_streams, 2) for s in dev[: a.fa_instances]]
    else:
        ev = list(range(10000, 10048))
        jobs += [delayed(oos_job)(f, ev[i:i + 12]) for f in ("squeeze", "newpart") for i in range(0, 48, 12)]
        jobs += [delayed(m4_job)(s) for s in ev]
    results = []
    par = Parallel(n_jobs=a.workers, return_as="generator_unordered")
    for i, res in enumerate(par(jobs)):
        results.append(res)
        L(f"job {res['kind']} {res['seed']} wall={res['wall_s']:.1f}s rows={len(res['rows'])} errors={len(res['errors'])}")
        progress(i + 1, len(jobs), "running", {"last": f"{res['kind']}/{res['seed']}"})
    rows = [r for res in results for r in res["rows"]]
    errors = [e for res in results for e in res["errors"]]
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)) + "\n")
    # manifests for downstream tasks
    oos = [r for r in rows if r["block"] == "OOS"]
    with open(out_dir / ("lin_oos_manifest_dev.jsonl" if a.mode == "pilot" else "lin_oos_manifest_eval.jsonl"), "w") as f:
        for r in sorted(oos, key=lambda r: (r["family"], r["seed"], r["problem_index"])):
            f.write(json.dumps(r) + "\n")
    m4 = sorted([r for r in rows if r["block"] == "M4"], key=lambda r: r["instance"])
    (out_dir / ("m4_calibration_dev.json" if a.mode == "pilot" else "m4_calibration_eval.json")).write_text(
        json.dumps({"eta_target": 2 * EPS, "T_ref": M4_T_REF, "nominal_steps": M4_NOMINAL_STEPS, "K": K_M4,
                    "generator": "make_nl_instance(kappa_mode='high', noise_seed=42)", "instances": m4},
                   indent=1, default=float))
    samples = [s for res in results for s in res["samples"]]
    vje_ep = [r for r in rows if r["block"] in ("VJE", "EP")]
    kept = [r for r in oos if r["keep"]][:6]
    (out_dir / "samples" / "traces.json").write_text(json.dumps(samples[:8], indent=1, default=float))
    (out_dir / "samples" / "vje_eprocess_rows.json").write_text(json.dumps(vje_ep, indent=1, default=float))
    (out_dir / "samples" / "lin_oos_kept_examples.json").write_text(json.dumps(kept, indent=1, default=float))
    wall = time.time() - start
    summ = analyse(rows, tests, a.mode, wall)
    summ["errors"] = errors[:20]
    summ["n_errors"] = len(errors)
    summ["job_wall_s"] = {f"{res['kind']}|{res['seed']}": res["wall_s"] for res in results}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    L(f"done wall={wall:.1f}s gate={summ['gate']}")
    log.close()
    status = "success" if not errors else "failed"
    if not markers:
        print(json.dumps(summ["gate"], indent=1))
        return
    mark_done(status, f"{summ['go_no_go']} gate={json.dumps(summ['gate'])}")
    if not a.no_progress_update:
        update_gpu_progress(status, start_iso, wall / 60,
                            {"env": "E1-NL-S (|Theta|=13824) + E1-Lin OOS", "mode": a.mode,
                             "fa_streams": a.fa_instances * a.fa_streams, "fa_rounds": 20 + N_FA,
                             "workers": a.workers, "gpu_count": 0, "note": "CPU only, concurrent with other tasks"},
                            13 if a.mode == "pilot" else 45)
    print(json.dumps(summ["gate"], indent=1))


if __name__ == "__main__":
    main()
