"""hf3_m4_blindspot: HF3 blind-spot boundary -- unregistered m4 temporal drift under the mixed numerator (methodology
2.4 / 4.2; expected NEGATIVE result: detection rate <= 2 delta).

Usage: run_hf3_m4_blindspot.py --mode {pilot,full} [--workers 4] [--instances a-b] [--problems K] [--streams 0,1,2]

Truth: kappa_dyn = high E1-NL-S instance (on the G_1 grid) with m4 drift alpha_i(t) = alpha_i + s * t_global / T_ref
(dsswm.envs.mis.DriftNLEnv, T_ref = 1000 platform rounds, t_global = public platform round counter incl. n0).
s is calibrated per (instance, stream) with dsswm.envs.mis.calibrate_m4 so that
    eta_min = min_theta max_{k, pi} |J_true(k, pi; alpha frozen at nominal start n0 + 200 k) - J_theta(k, pi)| = 2 eps
over the K problems of that stream (registered nominal schedule M4_NOMINAL_STEPS = 200).
Scoring truth of problem k for an arm = alpha frozen at the ACTUAL platform round at which that arm starts problem k
(exact up to the within-horizon drift s * H / T_ref, recorded).

Problem-stream protocol (methodology 3): n0 = 20 shared real rounds, then K Type-1 problems, stream = (noise seed,
problem-order permutation); each arm on its own fresh platform copy (CRN), re-using its OWN ledger across problems.
Arms (pre-registered DDA acquisition, T_max = 3000 new steps / problem, eps = 0.02, delta = 0.05):
  plugin     JPC on the in-class plug-in SeqLRSet; empty set -> MODEL_CONFLICT -> B2 fallback.
  mixed      JPC on MixedLRSet {pool, reg(load bucket), ext} 1/3 each (the registered HF3 arm). Theta_t empty ->
             alarm -> B2; replay check before every zero-cost certificate from the reused ledger.
  mixed_pid  DIAGNOSTIC (not a gate): same as mixed but reg = public problem-id regime. The problem id is public but
             is a coarse proxy for time; any extra detection here quantifies how much of the blind spot is closed by
             a time-correlated public label (reported separately, never pooled into HF3).
HF3 detection rate = fraction of streams in which the mixed arm raised MODEL_CONFLICT (CP interval); HF3 holds
("blind spot") if the detection rate is <= 2 delta. If > 2 delta: leakage checks run (always run anyway):
  (i)  AST check: regime functions read only PublicFactor fields (pytest test_regime_functions_read_public_fields_only)
  (ii) load_bucket output invariant to the public round index t (exhaustive over loads x t grid)
  (iii) ext design has no time/round feature: feature vectors from the ExtPlugin are identical for factors that differ
       only in the round index, and REGISTERED_DIRECTIONS contains no time direction.
Horizon diagnostic (not a gate): fresh platform, n0 + nominal-span uniform legal rounds (= K * 200), arms main,
ext_only, reg_only, pid_regime -- does the layer refute the class when sampling does not stop at certification?
Pilot: dev seeds 690-699, stream 0, K = 10. Full: assert_locked(), eval seeds 10000-10047 x 3 streams x 15 problems.
CPU only (gpu slot is a scheduling token); timings are measured under concurrent runs.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix  # noqa: E402
from dsswm.envs.mis import M4_NOMINAL_STEPS, M4_T_REF, DriftNLEnv, calibrate_m4, m4_schedule  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import load_bucket, problem_id_regime  # noqa: E402
from dsswm.evidence.replay import replay_check  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, generator_hash, make_nl_instance, nl_class_max_jtable  # noqa: E402
from dsswm.streams.offgrid import K_STREAM, STREAMS, stream_perm  # noqa: E402

TASK = "hf3_m4_blindspot"
EPS, DELTA = 0.02, 0.05
ETA_TARGET = 2 * EPS
TOP_M = 5
N0 = NL_DEFAULTS["n0"]
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = 2_310_000
ARMS = ("plugin", "mixed", "mixed_pid")
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables_f1"      # G_1 = DEFAULT_NL_S_GRID class tables (truth-free, by problem)
PLANNED_MIN = {"pilot": 10, "full": 40}
PILOT_SEEDS = list(range(690, 700))


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


# ------------------------------------------------------------------ instance context
_W = {}


def worker_tables():
    if "ncl" not in _W:
        torch.set_num_threads(1)
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        _W["ncl"] = ncl
        _W["params"] = ncl.torch_params()
    return _W


def class_J(q, prop, P):
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


class Ctx:
    def __init__(self, seed, stream, K):
        W = worker_tables()
        self.seed, self.stream = seed, stream
        self.noise, k_perm = STREAMS[stream]
        self.ncl = W["ncl"]
        self.base = make_nl_instance(seed, noise_seed=self.noise, nl_class=self.ncl, kappa_mode="high", K=K_STREAM)
        perm = stream_perm(seed, k_perm, K_STREAM)
        self.perm = perm.tolist()
        self.problems = [self.base.problems[i] for i in perm][:K]
        self.aspace = self.base.env.aspace
        self.prop = NLPropagator(2, 2, NMAX, self.aspace, C_KNOWN, RHO_RET, device="cpu")
        if "LT" not in W:
            W["LT"] = self.prop.tables(W["params"])[0]
            W["py"], W["pe"] = class_prob_tables(self.ncl.np_params, C_KNOWN, NMAX, self.aspace.nb)
            W["inc"] = IncidenceIndex(self.prop.codec, self.aspace, NMAX)
        self.LT = W["LT"]
        self.LT_np = self.LT.numpy()
        self.py, self.pe, self.inc = W["py"], W["pe"], W["inc"]
        self.J = [class_J(q, self.prop, W["params"]) for q in self.problems]
        self.Reg = [regret_matrix(J) for J in self.J]
        self.Jall = np.concatenate(self.J, 1)
        self.nA = self.aspace.n
        self.legal = np.arange(self.nA)
        self.ti = self.base.truth["theta_index"]           # harness only (scoring)
        self.base_tp = self.base.env.true_params()           # in-class truth (alpha0), harness only
        self.plans = [[self.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
                      for q in self.problems]
        self._tab_cache = {}

    # harness side ---------------------------------------------------
    def J_true_k(self, k, shift):
        key = round(float(shift), 12)
        if key not in self._tab_cache:
            tp = dict(self.base_tp)
            tp["alpha"] = self.base_tp["alpha"] + shift
            self._tab_cache[key] = self.prop.tables(params_to_torch(tp, device="cpu"))
        LT, EY = self._tab_cache[key]
        u = self.problems[k].utility
        return np.array([float(self.prop._run_plan(pl, es, LT, EY, u.w, u.w_ret, u.c_q)[0]) for pl, es in self.plans[k]])

    def calibrate(self):
        sched = m4_schedule(N0, len(self.problems))
        cal = calibrate_m4(self.J, self.J_true_k, ETA_TARGET, sched)
        self._tab_cache.clear()
        return cal, sched

    def make_env(self, s):
        t = self.base.truth
        env = DriftNLEnv(t["alpha"], t["beta"], t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"], c=C_KNOWN,
                         rho_ret=RHO_RET, L=2, R=2, nmax=NMAX, budget=NL_DEFAULTS["budget"], incentive_levels=(1,),
                         seed=int(self.seed) * 1000 + int(self.noise), drift_s=s, t_ref=M4_T_REF)
        rng = np.random.default_rng([self.seed, 12])          # == generator._initial_data stream
        init = [env.step(int(rng.choice(self.legal))) for _ in range(N0)]
        return env, init


# ------------------------------------------------------------------ whole-trial samplers (real platform, frozen at env.t)
def trial_sampler(env, q, key, store=None):
    counters = {}

    def sampler(a, n):
        c = counters.get(a, 0)
        counters[a] = c + 1
        r = np.random.default_rng([*key, a, c])
        ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
        if store is not None:
            st = store.setdefault(a, ([], []))
            st[0].append(np.asarray(ys)); st[1].append(np.asarray(es))
        return q.utility.value_from_components(ys, es)
    return sampler


def run_b2(ctx, env, k, history, key):
    q = ctx.problems[k]
    keys = [(p.name, tuple(int(x) for x in q.loads0), tuple(int(x) for x in q.engaged0), q.H) for p in q.policies]
    prior = {}
    for a, kk in enumerate(keys):
        if kk in history:
            ys, es = history[kk]
            prior[a] = list(q.utility.value_from_components(ys, es))
    store = {}
    smp = trial_sampler(env, q, key, store)
    res = lucb(smp, len(q.policies), q.H, EPS, DELTA, float(q.meta["u_max"]), TMAX_TRIAL, kind="eb", prior=prior)
    for a, kk in enumerate(keys):
        if a in store:
            ys = np.concatenate(store[a][0]); es = np.concatenate(store[a][1])
            n_used = int(res["n_per_arm"][a]) - len(prior.get(a, []))
            ys, es = ys[:n_used], es[:n_used]
            if kk in history:
                ys = np.concatenate([history[kk][0], ys]); es = np.concatenate([history[kk][1], es])
            history[kk] = (ys, es)
    res["fallback"] = "B2"
    return res


# ------------------------------------------------------------------ one arm over the whole problem stream
def run_arm(ctx, s, arm, want_samples=False):
    env, init = ctx.make_env(s)
    h = env.handle()
    is_mixed = arm in ("mixed", "mixed_pid")
    regime = problem_id_regime if arm == "mixed_pid" else load_bucket
    if is_mixed:
        E = MixedLRSet(ctx.prop, ctx.LT, DELTA, regime_fn=regime, name=arm)
    else:
        E = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    thr = LOG_THR
    ledger = []
    alarm = None

    def feed(o, pid):
        nonlocal alarm
        if is_mixed:
            E.update(o, problem_id=pid)
            if alarm is None and E.conflict_round is not None:
                alarm = {"round": int(E.conflict_round), "new_steps": int(max(0, E.conflict_round - N0)),
                         "problem_k": None, "alpha_shift_at_alarm": float(s * E.conflict_round / M4_T_REF),
                         "component_adv": {c: round(float(v), 3) for c, v in E.component_advantage().items()}}
        else:
            E.update(o)
        ledger.append((o, pid))

    for o in init:
        feed(o, None)
    if alarm is not None:
        alarm["problem_k"] = 0
    rng = np.random.default_rng([ctx.seed, ctx.noise, 4, 101 + ARMS.index(arm)])
    b2_hist = {}
    rows, samples = [], []
    cum_new = 0
    starts, Jts = [], []
    plugin_conflict_k = None
    for k, q in enumerate(ctx.problems):
        Reg, J = ctx.Reg[k], ctx.J[k]
        t0 = time.perf_counter()
        t_start = int(env.t)
        shift0 = s * t_start / M4_T_REF
        Jtq = ctx.J_true_k(k, shift0)                      # scoring truth: alpha frozen at the actual start round
        starts.append(t_start); Jts.append(Jtq)
        steps, status, traj = 0, None, []
        replay = None
        n_explore = 0
        while True:
            lrat = E.log_ratio().numpy()
            mask = lrat < thr
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
            if is_mixed and sv == "CERTIFIED" and steps == 0 and k > 0:
                rc = replay_check(ctx.prop, ctx.LT, ledger, q.pid, DELTA, regime_fn=regime, env_handle=h)
                replay = {"conflict": bool(rc["conflict"]), "size": int(rc["size"]), "n_replayed": int(rc["n_replayed"]),
                          "mask_equal_incremental": bool(np.array_equal(rc["mask"].numpy(), mask))}
                if rc["conflict"]:
                    sv = "MODEL_CONFLICT"
            if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                status = sv
                break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            code = ctx.prop.codec.encode(*h.observable_state())
            blk = cert["blocking"] or certify_minimax(Reg, mask, 0.0, TOP_M)["blocking"]
            kh = E.mle()
            if not blk:
                a = int(rng.choice(ctx.legal)); mode = "random"
            else:
                margins = thr - lrat[blk]
                a, info = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk],
                                     ctx.pe[blk], margins, ctx.inc, ctx.prop, ctx.legal, rng)
                mode = info["mode"]
            n_explore += int(mode != "dda")
            obs = h.step(a)
            steps += 1
            feed(obs, q.pid)
            if alarm is not None and alarm["problem_k"] is None:
                alarm["problem_k"] = k
            if want_samples and len(traj) < 30:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "star_alive": bool(mask[ctx.ti]), "mode": mode,
                             "alpha_shift": round(float(s * env.t / M4_T_REF), 5)})
        assert env.n_steps == len(ledger) == E.n_rounds, "every evidence round must be a real platform round"
        cum_new += steps
        if arm == "plugin" and status == "MODEL_CONFLICT" and plugin_conflict_k is None:
            plugin_conflict_k = k
        mask = (E.log_ratio().numpy() < thr)
        model_pi = cert["pi"] if status == "CERTIFIED" else None
        pi, charged = model_pi, None
        fallback = None
        if status == "MODEL_CONFLICT":
            res = run_b2(ctx, env, k, b2_hist, (ctx.seed, ctx.noise, 4, k, 55, ARMS.index(arm)))
            fb_pi = int(res["pi"]) if res["status"] == "CERTIFIED" else None
            fallback = {"method": "B2", "status": res["status"], "steps": int(res["steps"]), "pulls": int(res["pulls"]),
                        "n_reused": int(res["n_reused"]), "pi": fb_pi,
                        "true_regret": float(Jtq.max() - Jtq[fb_pi]) if fb_pi is not None else None}
            if res["status"] == "CERTIFIED":
                status, pi, charged = "CERTIFIED", fb_pi, steps + int(res["steps"])
            else:
                pi, charged = None, max(steps + int(res["steps"]), TMAX_TRIAL)
        cert_ok = status == "CERTIFIED"
        if charged is None:
            charged = steps if cert_ok else max(steps, TMAX_STEP)
        tr = float(Jtq.max() - Jtq[pi]) if (cert_ok and pi is not None) else None
        Jt_end = ctx.J_true_k(k, s * env.t / M4_T_REF)
        row = {"task": TASK, "instance": ctx.seed, "stream": ctx.stream, "noise_seed": ctx.noise, "misspec": "m4",
               "eta_level": 2.0, "strength": s, "problem": q.pid, "k": k, "method": arm, "design": "DDA",
               "status": status, "new_env_steps": int(charged), "steps_consumed_model": int(steps),
               "censored": not cert_ok, "certified_policy": pi, "true_regret": tr,
               "true_regret_at_end_round": (float(Jt_end.max() - Jt_end[pi]) if (cert_ok and pi is not None) else None),
               "false_cert": bool(cert_ok and tr is not None and tr > EPS),
               "zero_cost": bool(cert_ok and charged == 0),
               "t_start": t_start, "alpha_shift_start": shift0, "t_nominal": N0 + k * M4_NOMINAL_STEPS,
               "within_horizon_shift": s * q.H / M4_T_REF,
               "eta_q_theta_star": float(np.abs(Jtq - J[ctx.ti]).max()),
               "eta_q_min": float(np.abs(J - Jtq[None]).max(1).min()),
               "theta_cell_alive": bool(mask[ctx.ti]) if mask.any() else False, "set_size": int(mask.sum()),
               "true_best": int(np.argmax(Jtq)), "true_top2_gap": float(np.sort(Jtq)[-1] - np.sort(Jtq)[-2]),
               "H": q.H, "n_policies": len(q.policies), "eps": EPS, "delta": DELTA,
               "fallback": fallback, "replay": replay, "alarm_active": bool(alarm is not None),
               "model_pi": model_pi,
               "falsification_scope": (E.falsification_scope() if is_mixed else {"components": []}),
               "cum_new_steps_model": int(cum_new), "explore_steps": n_explore,
               "lr_n_rounds": int(E.n_rounds), "env_n_steps": int(env.n_steps), "rollouts": 0,
               "wall_clock_s": time.perf_counter() - t0}
        rows.append(row)
        if want_samples and k < 3:
            samples.append({**row, "J_true": np.round(Jtq, 4).tolist(), "J_class_at_theta_star": np.round(J[ctx.ti], 4).tolist(),
                            "trajectory_head": traj})
    jt_all = np.concatenate(Jts)
    dev = np.abs(ctx.Jall - jt_all[None]).max(1)
    stream = {"instance": ctx.seed, "stream": ctx.stream, "misspec": "m4", "method": arm, "strength": s,
              "alarm": alarm, "plugin_conflict_k": plugin_conflict_k,
              "stream_charged_steps": int(sum(r["new_env_steps"] for r in rows)),
              "stream_model_steps": int(cum_new), "final_round": int(env.t),
              "max_alpha_shift_reached": float(s * env.t / M4_T_REF),
              "starts": starts, "realized_eta_min_over_eps": float(dev.min() / EPS),
              "realized_eta_theta_star_over_eps": float(dev[ctx.ti] / EPS),
              "n_false_cert": int(sum(r["false_cert"] for r in rows)),
              "min_gap_vs_pool": float(E.min_gap_vs_pool) if is_mixed else None,
              "component_adv_final": ({c: round(float(v), 3) for c, v in E.component_advantage().items()}
                                      if is_mixed else None),
              "ext_state": (E.ext.state() if is_mixed and E.ext is not None else None)}
    return rows, samples, stream


def run_unit(seed, stream, K, want_samples):
    t0 = time.time()
    out = {"seed": seed, "stream": stream, "rows": [], "streams": [], "samples": [], "errors": []}
    try:
        ctx = Ctx(seed, stream, K)
        cal, sched = ctx.calibrate()
        s = cal["s"]
        out["calib"] = {"instance": seed, "stream": stream, "misspec": "m4", "strength": s,
                        "eta_min": cal["eta_min"], "eta_min_over_eps": cal["eta_min"] / EPS,
                        "eta_min_at_0": cal["eta_min_at_0"], "theta_min": cal["theta_min"],
                        "theta_min_is_theta_star": bool(cal["theta_min"] == ctx.ti), "iters": cal["iters"],
                        "reached": cal["reached"], "schedule": sched, "perm": ctx.perm[:K],
                        "max_alpha_shift_nominal": s * sched[-1] / M4_T_REF, "calib_s": time.time() - t0}
        for arm in ARMS:
            try:
                ta = time.time()
                rows, smp, st = run_arm(ctx, s, arm, want_samples=want_samples)
                st["wall_s"] = time.time() - ta
                out["rows"] += rows
                out["samples"] += smp
                out["streams"].append(st)
            except Exception as e:  # noqa: BLE001
                out["errors"].append({"where": f"{arm}", "error": repr(e), "tb": traceback.format_exc()[-1500:]})
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"where": "unit", "error": repr(e), "tb": traceback.format_exc()[-1500:]})
    out["wall_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------ detection-horizon diagnostic (not a gate)
HORIZON_ARMS = ("main", "ext_only", "reg_only", "pid_regime")
HORIZON_PID_EVERY = M4_NOMINAL_STEPS


def horizon_unit(seed, stream, K, n_rounds):
    t0 = time.time()
    try:
        ctx = Ctx(seed, stream, K)
        cal, _ = ctx.calibrate()
        s = cal["s"]
        env, init = ctx.make_env(s)
        h = env.handle()
        sets = {"main": MixedLRSet(ctx.prop, ctx.LT, DELTA, regime_fn=load_bucket, name="main"),
                "ext_only": MixedLRSet(ctx.prop, ctx.LT, DELTA, components=("pool", "ext"), weights=(0.5, 0.5)),
                "reg_only": MixedLRSet(ctx.prop, ctx.LT, DELTA, components=("pool", "reg"), weights=(0.5, 0.5)),
                "pid_regime": MixedLRSet(ctx.prop, ctx.LT, DELTA, regime_fn=problem_id_regime, name="pid")}
        plug = SeqLRSet(ctx.prop, ctx.LT, DELTA)
        plug_empty = None
        for o in init:
            for m in sets.values():
                m.update(o, problem_id=None)
            plug.update(o)
        rng = np.random.default_rng([seed, ctx.noise, 4, 909])
        trace = []
        for r in range(1, n_rounds + 1):
            o = h.step(int(rng.integers(h.aspace.n)))
            for m in sets.values():
                m.update(o, problem_id=f"h{r // HORIZON_PID_EVERY}")
            plug.update(o)
            if plug_empty is None and not bool((plug.log_ratio().numpy() < LOG_THR).any()):
                plug_empty = N0 + r
            if r % 500 == 0:
                trace.append({"round": N0 + r, **{a: {c: round(float(v), 2) for c, v in m.component_advantage().items()}
                                                  for a, m in sets.items()}})
        assert env.n_steps == N0 + n_rounds
        rec = {"instance": seed, "stream": stream, "strength": s, "eta_min_over_eps": cal["eta_min"] / EPS,
               "n_rounds_total": N0 + n_rounds, "design": "uniform_legal",
               "max_alpha_shift": s * (N0 + n_rounds) / M4_T_REF, "plugin_empty_round": plug_empty,
               "arms": {a: {"conflict_round": m.conflict_round,
                            "theta_star_pt_alive": bool(m.mask()[ctx.ti]),
                            "component_adv_final": {c: round(float(v), 2) for c, v in m.component_advantage().items()}}
                        for a, m in sets.items()},
               "trace": trace, "wall_s": time.time() - t0}
        return {"rec": rec, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"rec": None, "error": {"where": f"horizon {seed}/{stream}", "error": repr(e),
                                       "tb": traceback.format_exc()[-1500:]}}


# ------------------------------------------------------------------ leakage checks
def leakage_check():
    from dsswm.evidence import ext_plugin as XP
    from dsswm.evidence.mixed_lr import factorize  # noqa: F401  (import check only)
    from dsswm.evidence.regimes import PUBLIC_FIELDS, PublicFactor
    out = {}
    # (i) AST check (unit test)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x",
                        "dsswm/tests/test_falsify_layer.py::test_regime_functions_read_public_fields_only",
                        "dsswm/tests/test_falsify_layer.py::test_m4_drift_env"],
                       cwd=str(HERE), capture_output=True, text=True, timeout=600)
    out["ast_regime_public_fields_test"] = {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-1:]}
    # (ii) load_bucket invariant to the public round index t
    viol = 0
    for load in range(0, 4):
        for kind in ("pair", "ret"):
            vals = {load_bucket(PublicFactor(kind, 0, 1, 0, load, 0, "q", t)) for t in (0, 1, 7, 20, 199, 2000, 10 ** 6)}
            viol += int(len(vals) != 1)
    out["load_bucket_time_invariant"] = viol == 0
    out["public_fields"] = list(PUBLIC_FIELDS)
    # (iii) ext has no time feature: design depends only on (kind, i, j, p, load, level, value, y_partner)
    ext = XP.ExtPlugin(2, 2, NMAX, 2, C_KNOWN)
    import inspect
    sig_pair = list(inspect.signature(ext._x_pair).parameters)
    sig_ret = list(inspect.signature(ext._x_ret).parameters)
    src = inspect.getsource(XP.ExtPlugin._design)
    out["ext_feature_signatures"] = {"pair": sig_pair, "ret": sig_ret}
    out["ext_design_reads_t"] = (".t)" in src) or (".t," in src) or ("f.t" in src)
    out["ext_registered_directions"] = list(XP.REGISTERED_DIRECTIONS)
    time_words = ("t", "time", "round", "epoch", "drift")
    out["ext_no_time_direction"] = not any(d in time_words for d in XP.REGISTERED_DIRECTIONS)
    out["ext_no_time_feature"] = (not out["ext_design_reads_t"]) and ("t" not in sig_pair) and ("t" not in sig_ret)
    out["pass"] = bool(out["ast_regime_public_fields_test"]["returncode"] == 0 and out["load_bucket_time_invariant"]
                       and out["ext_no_time_direction"] and out["ext_no_time_feature"])
    out["note"] = ("problem_id regime (mixed_pid diagnostic arm) is public but time-correlated by construction; it is "
                   "not part of the registered HF3 arm (reg = load bucket)")
    return out


# ------------------------------------------------------------------ analysis
def _cp(k, n):
    if n == 0:
        return None, None
    lo, hi = clopper_pearson(int(k), int(n))
    return float(lo), float(hi)


def _med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def summarize(rows, streams, calibs, mode, wall, hz, leak):
    S = {"task": TASK, "mode": mode, "eps": EPS, "delta": DELTA, "eta_target_over_eps": ETA_TARGET / EPS,
         "arms": {}, "n_rows": len(rows), "n_streams": len(streams), "wall_s": wall,
         "timing_note": "wall clock measured under concurrent runs (4 tasks share 20 cores)"}
    for arm in ARMS:
        R = [r for r in rows if r["method"] == arm]
        ST = [s for s in streams if s["method"] == arm]
        if not R:
            continue
        n_cert = sum(r["status"] == "CERTIFIED" for r in R)
        fc = sum(r["false_cert"] for r in R)
        lo, hi = _cp(fc, n_cert)
        c = {"n_problems": len(R), "n_streams": len(ST), "n_cert": n_cert, "completion": n_cert / len(R),
             "n_false_cert": int(fc), "fcr": fc / n_cert if n_cert else None, "fcr_cp_lower": lo, "fcr_cp_upper": hi,
             "streams_with_false_cert": sum(s["n_false_cert"] > 0 for s in ST),
             "false_cert_at_end_round": sum(1 for r in R if r["true_regret_at_end_round"] is not None
                                            and r["true_regret_at_end_round"] > EPS),
             "status_counts": {k2: sum(r["status"] == k2 for r in R) for k2 in sorted({r["status"] for r in R})},
             "stream_charged_steps_median": _med([s["stream_charged_steps"] for s in ST]),
             "stream_model_steps_median": _med([s["stream_model_steps"] for s in ST]),
             "final_round_median": _med([s["final_round"] for s in ST]),
             "max_alpha_shift_reached_median": _med([s["max_alpha_shift_reached"] for s in ST]),
             "realized_eta_min_over_eps_median": _med([s["realized_eta_min_over_eps"] for s in ST]),
             "realized_eta_min_over_eps_range": ([float(min(s["realized_eta_min_over_eps"] for s in ST)),
                                                  float(max(s["realized_eta_min_over_eps"] for s in ST))] if ST else None),
             "zero_cost_rate": sum(r["zero_cost"] for r in R) / len(R),
             "theta_star_pt_alive_rate": sum(r["theta_cell_alive"] for r in R) / len(R),
             "true_regret_certified_median": _med([r["true_regret"] for r in R]),
             "n_fallback": sum(r["fallback"] is not None for r in R),
             "false_cert_by_k": [sum(r["false_cert"] for r in R if r["k"] == k) for k in range(max(r["k"] for r in R) + 1)]}
        if arm == "plugin":
            pc = [s["plugin_conflict_k"] for s in ST if s["plugin_conflict_k"] is not None]
            c["detection_streams"] = len(pc)
            c["detection_rate"] = len(pc) / len(ST) if ST else None
            c["detection_rate_cp"] = _cp(len(pc), len(ST))
            c["detection_note"] = "plug-in 'detection' = in-class LR set became empty (MODEL_CONFLICT) at some problem"
        else:
            al = [s["alarm"] for s in ST if s["alarm"] is not None]
            c["detection_streams"] = len(al)
            c["detection_rate"] = len(al) / len(ST) if ST else None
            c["detection_rate_cp"] = _cp(len(al), len(ST))
            c["alarm_rounds"] = [a["round"] for a in al]
            c["alarm_problem_k"] = [a["problem_k"] for a in al]
            c["alarm_alpha_shift"] = [round(a["alpha_shift_at_alarm"], 4) for a in al]
            c["alarm_winning_component"] = {}
            for a in al:
                adv = {k2: v for k2, v in a["component_adv"].items() if k2 != "pool"}
                w = max(adv, key=adv.get) if adv else None
                c["alarm_winning_component"][w] = c["alarm_winning_component"].get(w, 0) + 1
            c["false_cert_before_alarm"] = sum(r["false_cert"] and not r["alarm_active"] for r in R)
            c["component_adv_final_median"] = {comp: _med([s["component_adv_final"].get(comp) for s in ST
                                                           if s["component_adv_final"]])
                                               for comp in ("reg", "ext")}
            fb = [r["fallback"] for r in R if r["fallback"] is not None]
            c["post_fallback"] = {"n": len(fb), "n_cert": sum(f["status"] == "CERTIFIED" for f in fb),
                                  "n_regret_gt_eps": sum(f["true_regret"] is not None and f["true_regret"] > EPS for f in fb)}
            rp = [r["replay"] for r in R if r["replay"] is not None]
            c["replay_checks"] = len(rp)
            c["replay_conflicts"] = sum(x["conflict"] for x in rp)
            c["replay_mask_equal_incremental"] = all(x["mask_equal_incremental"] for x in rp) if rp else None
        S["arms"][arm] = c
    S["calibration"] = {"n": len(calibs), "reached_all": all(x["reached"] for x in calibs),
                        "reached_frac": float(np.mean([x["reached"] for x in calibs])) if calibs else None,
                        "strength_median": _med([x["strength"] for x in calibs]),
                        "strength_range": ([float(min(x["strength"] for x in calibs)),
                                            float(max(x["strength"] for x in calibs))] if calibs else None),
                        "eta_min_over_eps_median": _med([x["eta_min_over_eps"] for x in calibs]),
                        "max_alpha_shift_nominal_median": _med([x["max_alpha_shift_nominal"] for x in calibs]),
                        "theta_min_is_theta_star_frac": float(np.mean([x["theta_min_is_theta_star"] for x in calibs]))
                        if calibs else None}
    if "mixed" in S["arms"] and "plugin" in S["arms"]:
        sm, sp = S["arms"]["mixed"]["stream_charged_steps_median"], S["arms"]["plugin"]["stream_charged_steps_median"]
        S["stream_ratio_mixed_over_plugin"] = (sm / sp) if (sm is not None and sp) else None
    m = S["arms"].get("mixed", {})
    det = m.get("detection_rate")
    det_cp = m.get("detection_rate_cp", (None, None))
    S["hf3"] = {"detection_rate": det, "detection_rate_cp": det_cp, "threshold_2delta": 2 * DELTA,
                "blind_spot_holds_point": (det is not None and det <= 2 * DELTA),
                "blind_spot_refuted_cp": (det_cp[0] is not None and det_cp[0] > 2 * DELTA),
                "fcr_plugin": S["arms"].get("plugin", {}).get("fcr"),
                "fcr_plugin_cp_upper": S["arms"].get("plugin", {}).get("fcr_cp_upper"),
                "fcr_mixed": m.get("fcr"), "fcr_mixed_cp_upper": m.get("fcr_cp_upper"),
                "diagnostic_mixed_pid_detection_rate": S["arms"].get("mixed_pid", {}).get("detection_rate"),
                "leakage_check_pass": leak.get("pass") if leak else None,
                "note": ("expected negative: detection <= 2 delta means the unregistered time direction is a true blind "
                         "spot. Pilot n is small; formal evaluation pools with hr3_r1_uncorrected's mixed arm in full.")}
    if hz:
        d = {"n": len(hz), "n_rounds_total": hz[0]["n_rounds_total"],
             "max_alpha_shift_median": _med([x["max_alpha_shift"] for x in hz]),
             "plugin_empty_rate": sum(x["plugin_empty_round"] is not None for x in hz) / len(hz),
             "plugin_empty_rounds": [x["plugin_empty_round"] for x in hz]}
        for a in HORIZON_ARMS:
            cr = [x["arms"][a]["conflict_round"] for x in hz]
            hit = [c for c in cr if c is not None]
            d[a] = {"alarm_rate": len(hit) / len(hz), "alarm_rate_cp": _cp(len(hit), len(hz)),
                    "alarm_rounds": cr, "alarm_round_median_among_alarmed": _med(hit),
                    "theta_star_pt_alive_rate": sum(x["arms"][a]["theta_star_pt_alive"] for x in hz) / len(hz)}
        S["horizon_diagnostic"] = d
    S["leakage_check"] = leak
    return S


def plots(rows, streams, hz, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    labels, vals, los, his = [], [], [], []
    for arm in ARMS:
        ST = [s for s in streams if s["method"] == arm]
        if not ST:
            continue
        n = sum((s["alarm"] is not None) if arm != "plugin" else (s["plugin_conflict_k"] is not None) for s in ST)
        lo, hi = _cp(n, len(ST))
        labels.append(arm); vals.append(n / len(ST)); los.append(lo); his.append(hi)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["tab:gray", "tab:green", "tab:olive"][:len(labels)])
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none", ecolor="k")
    ax.axhline(2 * DELTA, color="r", ls="--", lw=0.8, label="2 delta")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("stream detection rate (CP 95%)")
    ax.set_title("m4 drift (eta_min = 2 eps): detection"); ax.legend(fontsize=7)
    ax = axs[1]
    for arm, col in zip(ARMS, ("tab:gray", "tab:green", "tab:olive")):
        R = [r for r in rows if r["method"] == arm and r["true_regret"] is not None]
        ax.scatter([r["alpha_shift_start"] for r in R], [r["true_regret"] for r in R], s=10, color=col, alpha=0.6,
                   label=arm)
    ax.axhline(EPS, color="r", ls="--", lw=0.8, label="eps")
    ax.set_xlabel("alpha shift at problem start"); ax.set_ylabel("true regret of certified policy")
    ax.set_title("Certified regret vs accumulated drift"); ax.legend(fontsize=7)
    ax = axs[2]
    if hz:
        for a, col in zip(HORIZON_ARMS, ("tab:green", "tab:blue", "tab:orange", "tab:olive")):
            for x_ in hz:
                tr = x_["trace"]
                ax.plot([t["round"] for t in tr], [t[a].get("reg", t[a].get("ext", 0)) if a != "ext_only"
                                                   else t[a].get("ext", 0) for t in tr], color=col, alpha=0.4, lw=0.8)
            ax.plot([], [], color=col, label=a)
        ax.axhline(math.log(1 / DELTA), color="r", ls="--", lw=0.8, label="log(1/delta)")
        ax.set_xlabel("platform round (uniform design)"); ax.set_ylabel("component log-advantage vs pool")
        ax.set_title("Horizon diagnostic"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_detection_and_regret.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--streams", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--horizon", type=int, default=-1, help="uniform rounds (-1 = K * 200 nominal span, 0 = off)")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        rng_ = lock["eval_manifest"]["per_task_ranges"][TASK][0]
        seeds = list(range(rng_[0], rng_[1] + 1))
    else:
        seeds = list(PILOT_SEEDS)
    if args.instances:
        seeds = parse_range(args.instances)
    if not pilot:
        assert all(s >= 10000 for s in seeds)
    else:
        assert all(600 <= s <= 699 for s in seeds), "pilot uses dev seeds 600-699 only"
    K = args.problems or (10 if pilot else 15)
    streams_ids = [int(x) for x in (args.streams or ("0" if pilot else "0,1,2")).split(",")]
    n_hz = K * M4_NOMINAL_STEPS if args.horizon < 0 else args.horizon
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_f, done_f = out_dir / "results.jsonl", out_dir / "units_done.jsonl"
    log_f = out_dir / "run.log"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_f, "a") as fh:
            fh.write(line + "\n")

    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    T0 = time.time()
    progress(0, 1, "start")
    status = "success"
    try:
        leak = leakage_check()
        log(f"leakage_check: {json.dumps(leak)}")
        done = set()
        if done_f.exists():
            for ln in done_f.read_text().splitlines():
                u = json.loads(ln)
                done.add((u["seed"], u["stream"]))
        jobs = [(s, st) for s in seeds for st in streams_ids if (s, st) not in done]
        log(f"mode={args.mode} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) streams={streams_ids} K={K} arms={ARMS} "
            f"T_max={TMAX_STEP} T_ref={M4_T_REF} eta_target={ETA_TARGET} workers={args.workers}; units todo={len(jobs)} "
            f"done={len(done)}; generator_hash={generator_hash()}")
        total = len(jobs) + len(done)
        if jobs and not args.resummarize:
            samp_units = {(seeds[0], streams_ids[0]), (seeds[1 % len(seeds)], streams_ids[0])}
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(run_unit)(s, st, K, (s, st) in samp_units) for (s, st) in jobs)
            n_err = 0
            for i, out in enumerate(gen):
                with open(res_f, "a") as fh:
                    for r in out["rows"]:
                        fh.write(json.dumps({"rec": "row", **r}) + "\n")
                    for st in out["streams"]:
                        fh.write(json.dumps({"rec": "stream", **st}) + "\n")
                    if "calib" in out:
                        fh.write(json.dumps({"rec": "calib", **out["calib"]}) + "\n")
                for sm in out["samples"]:
                    (out_dir / "samples" / f"m4_i{out['seed']}_s{out['stream']}_{sm['method']}_k{sm['k']}.json"
                     ).write_text(json.dumps(sm, indent=1, default=str))
                for e in out["errors"]:
                    n_err += 1
                    log(f"ERROR unit {out['seed']}/{out['stream']} {e['where']}: {e['error']}\n{e['tb']}")
                if not out["errors"]:
                    with open(done_f, "a") as fh:
                        fh.write(json.dumps({"seed": out["seed"], "stream": out["stream"], "wall_s": out["wall_s"]}) + "\n")
                al = {s_["method"]: (s_["alarm"]["round"] if s_["alarm"] else None) if s_["method"] != "plugin"
                      else s_["plugin_conflict_k"] for s_ in out["streams"]}
                log(f"unit {i + 1}/{len(jobs)} {out['seed']} s{out['stream']} {out['wall_s']:.0f}s  "
                    f"strength={out.get('calib', {}).get('strength')} reached={out.get('calib', {}).get('reached')}  "
                    f"detect={al}  fc={[sum(r['false_cert'] for r in out['rows'] if r['method'] == a) for a in ARMS]}")
                progress(len(done) + i + 1, total, "units", {"errors": n_err})
        hz_f = out_dir / "horizon.jsonl"
        if n_hz > 0 and not args.resummarize:
            hz_done = set()
            if hz_f.exists():
                for ln in hz_f.read_text().splitlines():
                    x = json.loads(ln)
                    hz_done.add((x["instance"], x["stream"]))
            hjobs = [(sd, streams_ids[0]) for sd in seeds if (sd, streams_ids[0]) not in hz_done]
            log(f"horizon diagnostic: {len(hjobs)} jobs x {n_hz} uniform rounds")
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(horizon_unit)(sd, st, K, n_hz) for (sd, st) in hjobs)
            for i, o in enumerate(gen):
                if o["error"]:
                    log(f"ERROR horizon {o['error']['where']}: {o['error']['error']}\n{o['error']['tb']}")
                    continue
                with open(hz_f, "a") as fh:
                    fh.write(json.dumps(o["rec"]) + "\n")
                x = o["rec"]
                log(f"horizon {i + 1}/{len(hjobs)} {x['instance']} plugin_empty={x['plugin_empty_round']} "
                    f"conflict={ {a: v['conflict_round'] for a, v in x['arms'].items()} } {x['wall_s']:.0f}s")
                progress(total, total, "horizon", {"done": i + 1, "of": len(hjobs)})
        rows, streams, calibs = [], [], []
        for ln in res_f.read_text().splitlines():
            d = json.loads(ln)
            rec = d.pop("rec")
            (rows if rec == "row" else streams if rec == "stream" else calibs).append(d)
        hz = [json.loads(ln) for ln in hz_f.read_text().splitlines()] if hz_f.exists() else []
        summ = summarize(rows, streams, calibs, args.mode, time.time() - T0, hz, leak)
        summ.update({"seeds": [seeds[0], seeds[-1]], "K": K, "streams": streams_ids, "T_max_step": TMAX_STEP,
                     "T_max_trial": TMAX_TRIAL, "T_ref": M4_T_REF, "nominal_steps": M4_NOMINAL_STEPS,
                     "generator_hash": generator_hash(), "eval_seeds_touched": not pilot})
        crashed = sum(1 for ln in log_f.read_text().splitlines() if "ERROR unit" in ln) if log_f.exists() else 0
        summ["pilot_gate"] = {"end_to_end": bool(rows) and crashed == 0, "n_unit_errors": crashed,
                              "m4_calibration_reached": bool(summ["calibration"]["reached_all"]),
                              "detection_rate_reported": summ["hf3"]["detection_rate"] is not None}
        summ["pilot_gate"]["pass"] = all(summ["pilot_gate"][k] for k in ("end_to_end", "m4_calibration_reached",
                                                                         "detection_rate_reported"))
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        try:
            plots(rows, streams, hz, out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed: {e!r}")
        log(f"summary written; pilot_gate={summ['pilot_gate']}; hf3={json.dumps(summ['hf3'], default=str)}; "
            f"wall {time.time() - T0:.0f}s")
        result_summary = json.dumps({"pilot_gate": summ["pilot_gate"], "hf3": summ["hf3"]}, default=str)[:1500]
    except Exception as e:  # noqa: BLE001
        status = "failed"
        result_summary = f"{e!r}"
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    wall = time.time() - T0
    mark_done(status, result_summary)
    update_gpu_progress(status, start_iso, wall / 60,
                        {"env": "E1-Mis-m4 temporal drift (E1-NL-S class |Theta|=13824)", "mode": args.mode,
                         "instances": len(seeds), "K": K, "streams": streams_ids, "arms": list(ARMS),
                         "eta_target_over_eps": 2, "T_ref": M4_T_REF, "horizon_rounds": n_hz,
                         "workers": args.workers, "gpu_count": 0, "note": "CPU only, concurrent with other tasks"},
                        PLANNED_MIN[args.mode])
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
