"""hf2b_m1_m3_mixed: HF2b -- registered misspecification m1 / m3 under the mixed numerator vs AGC (methodology 2.4 / 4.2).

Usage: run_hf2b_m1_m3_mixed.py --mode {pilot,full} [--workers 4] [--instances a-b] [--problems K] [--etas 0,2]

Truth (round-0 implementation, copied from run_misspec_m2_m3.py): kappa_dyn = high E1-NL-S instance (on the G_1 grid)
plus an out-of-class term of strength s >= 0
  m1  pair synergy  logit += s * S_ij  (S double centred, max|S| = 1)        -- registered ext direction 'syn'
  m3  hidden two-type psi: psi_left[i, b=1] = psi + s * tau_i, tau_i in {-1,+1} -- registered reg direction (load
      bucket regime) per methodology 2.4 (and ext 'psi_i')
s is calibrated per (instance, stream) so that eta_min = min_theta max_{q in stream, pi} |J_true - J_theta| =
(eta/eps) * eps over the WHOLE problem stream of that (instance, stream). eta/eps in {0, 1, 2} (pilot {0, 2}).

Problem-stream protocol (methodology 3): n0 = 20 shared real rounds, then K Type-1 problems (stream = (noise seed,
problem-order permutation) in ((42, perm0), (123, perm1), (456, perm2)); perm over the 15-problem generator stream,
first K kept). Each arm runs on its own fresh platform copy (CRN) and re-uses its OWN ledger across the problems.
Arms (all use the pre-registered DDA acquisition, T_max = 3000 new steps / problem, eps = 0.02, delta = 0.05):
  plugin  JPC on the in-class plug-in SeqLRSet (falsification scope = {}); empty set -> MODEL_CONFLICT -> B2.
  mixed   JPC on MixedLRSet {pool, reg(load bucket), ext} (1/3 each). Theta_t empty -> MODEL_CONFLICT (alarm) ->
          B2 (residual + empirical-Bernstein whole-trial BAI) on that problem and on every later problem whose set
          stays empty; fallback steps charged. Replay check (dsswm.evidence.replay) before every zero-cost
          certificate obtained from the reused ledger.
  AGC     round-0 JPC+AGC: model certificate from the delta/2 plug-in set, then an AGC audit (delta/2,
          whole-trial empirical Bernstein) of pi_hat vs the competitive set + top-2 worst-gap challengers; audit
          steps charged on every certified problem. Empty delta/2 set -> B2 fallback.
Metrics: FCR = false certs / certs (Clopper-Pearson), alarm steps (mixed; real rounds incl. n0 at the first empty
set, and new steps after n0), AGC audit steps per audited problem, cost ratio = median alarm / median AGC audit,
post-fallback true regret, stream-level charged steps.
Pilot: dev seeds 600-604, stream 0, K = 6, kinds {m1, m3}, eta in {0, 2}. Full: assert_locked(), eval seeds
10000-10031 x 3 streams x 10 problems x {m1, m3} x eta in {0, 1, 2}; incremental results.jsonl + units_done.jsonl.
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
from dsswm.audit.agc import agc_audit  # noqa: E402
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix  # noqa: E402
from dsswm.envs.nl import NLEnv  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import load_bucket  # noqa: E402
from dsswm.evidence.replay import replay_check  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, generator_hash, make_nl_instance, nl_class_max_jtable  # noqa: E402
from dsswm.streams.offgrid import K_STREAM, STREAMS, stream_perm  # noqa: E402

TASK = "hf2b_m1_m3_mixed"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
N0 = NL_DEFAULTS["n0"]
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
LOG_THR = math.log(1.0 / DELTA)
LOG_THR2 = math.log(2.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = 2_310_000
ARMS = ("plugin", "mixed", "AGC")
KIND_IDS = {"m1": 1, "m3": 3}
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables_f1"      # G_1 = DEFAULT_NL_S_GRID class tables (truth-free, by problem)
PLANNED_MIN = {"pilot": 14, "full": 55}


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


# ------------------------------------------------------------------ instance context (harness + learner tables)
def synergy_pattern(seed, L=2, R=2):
    s = np.random.default_rng([seed, 61]).standard_normal((L, R))
    s = s - s.mean(1, keepdims=True)
    s = s - s.mean(0, keepdims=True)
    return s / max(np.abs(s).max(), 1e-12)


def m3_types(seed, L=2):
    t = np.array([-1.0, 1.0] * ((L + 1) // 2))[:L]
    return np.random.default_rng([seed, 62]).permutation(t)


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
    def __init__(self, seed, kind, stream, K):
        W = worker_tables()
        self.seed, self.kind, self.stream = seed, kind, stream
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
        self.S = synergy_pattern(seed)
        self.types = m3_types(seed)
        self.plans = [[self.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
                      for q in self.problems]

    # harness side ---------------------------------------------------
    def _mis_kwargs(self, s):
        if self.kind == "m1":
            return {"syn": s * self.S}
        if self.kind == "m3":
            psi = float(np.asarray(self.base.truth["psi"]).reshape(-1)[0])
            pl = np.zeros((2, self.aspace.nb))
            pl[:, 1:] = (psi + s * self.types)[:, None]
            return {"psi_left": pl}
        raise ValueError(self.kind)

    def true_params(self, s):
        tp = self.base.env.true_params()
        tp.update(self._mis_kwargs(s))
        return tp

    def make_env(self, s):
        t = self.base.truth
        env = NLEnv(t["alpha"], t["beta"], t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"], c=C_KNOWN,
                    rho_ret=RHO_RET, L=2, R=2, nmax=NMAX, budget=NL_DEFAULTS["budget"], incentive_levels=(1,),
                    seed=int(self.seed) * 1000 + int(self.noise), **self._mis_kwargs(s))
        rng = np.random.default_rng([self.seed, 12])          # == generator._initial_data stream
        init = [env.step(int(rng.choice(self.legal))) for _ in range(N0)]
        return env, init

    def J_true(self, s):
        LT, EY = self.prop.tables(params_to_torch(self.true_params(s), device="cpu"))
        out = []
        for q, plans in zip(self.problems, self.plans):
            u = q.utility
            out.append(np.array([float(self.prop._run_plan(pl, es, LT, EY, u.w, u.w_ret, u.c_q)[0]) for pl, es in plans]))
        return out

    def eta_min(self, s):
        jt = np.concatenate(self.J_true(s))
        dev = np.abs(self.Jall - jt[None]).max(1)
        k = int(np.argmin(dev))
        return float(dev[k]), k

    def calibrate(self, target, tol=1e-4, s_cap=20.0):
        if target <= 0:
            return 0.0, 0.0, self.ti, 0, True
        lo, hi, it = 0.0, 0.05, 0
        while self.eta_min(hi)[0] < target and hi < s_cap:
            lo, hi = hi, min(hi * 2, s_cap)
            it += 1
        if self.eta_min(hi)[0] < target:
            v, k = self.eta_min(hi)
            return hi, v, k, it, False
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            v = self.eta_min(mid)[0]
            it += 1
            if v < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-5 or abs(v - target) < tol * target:
                break
        s = 0.5 * (lo + hi)
        v, k = self.eta_min(s)
        return s, v, k, it, bool(abs(v - target) <= 0.01 * target)


# ------------------------------------------------------------------ whole-trial samplers (real platform)
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
    """B2 fallback: residual + EB whole-trial LUCB, re-using this arm's stored whole trials (same s0, H, policy)."""
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
def run_arm(ctx, s, Jt, arm, eta_lvl, want_samples=False):
    env, init = ctx.make_env(s)
    h = env.handle()
    if arm == "mixed":
        E = MixedLRSet(ctx.prop, ctx.LT, DELTA, regime_fn=load_bucket, name="main")
        thr = LOG_THR
    elif arm == "plugin":
        E = SeqLRSet(ctx.prop, ctx.LT, DELTA)
        thr = LOG_THR
    else:
        E = SeqLRSet(ctx.prop, ctx.LT, DELTA / 2)
        thr = LOG_THR2
    ledger = []
    alarm = None

    def feed(o, pid):
        nonlocal alarm
        if arm == "mixed":
            E.update(o, problem_id=pid)
            if alarm is None and E.conflict_round is not None:
                alarm = {"round": int(E.conflict_round), "new_steps": int(max(0, E.conflict_round - N0)),
                         "problem_k": None, "component_adv": {c: round(float(v), 3)
                                                              for c, v in E.component_advantage().items()}}
        else:
            E.update(o)
        ledger.append((o, pid))

    for o in init:
        feed(o, None)
    if alarm is not None:
        alarm["problem_k"] = 0
    rng = np.random.default_rng([ctx.seed, ctx.noise, KIND_IDS[ctx.kind], int(eta_lvl * 100), 101 + ARMS.index(arm)])
    b2_hist = {}
    rows, samples = [], []
    cum_new = 0
    agc_first_conflict = None
    for k, q in enumerate(ctx.problems):
        Reg, J = ctx.Reg[k], ctx.J[k]
        t0 = time.perf_counter()
        steps, status, traj = 0, None, []
        replay = None
        n_explore = 0
        while True:
            lrat = E.log_ratio().numpy()
            mask = lrat < thr
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
            if arm == "mixed" and sv == "CERTIFIED" and steps == 0 and k > 0:
                # zero-cost certificate from the reused ledger -> replay check on the full ledger (no interaction)
                rc = replay_check(ctx.prop, ctx.LT, ledger, q.pid, DELTA, regime_fn=load_bucket, env_handle=h)
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
                             "mle": int(kh), "star_alive": bool(mask[ctx.ti]), "mode": mode})
        assert env.n_steps == len(ledger) == E.n_rounds, "every evidence round must be a real platform round"
        cum_new += steps
        Jtq = Jt[k]
        mask = (E.log_ratio().numpy() < thr)
        model_pi = cert["pi"] if status == "CERTIFIED" else None
        pi, charged, extra = model_pi, None, {}
        fallback = None
        audit = None
        if status == "MODEL_CONFLICT":
            res = run_b2(ctx, env, k, b2_hist, (ctx.seed, ctx.noise, KIND_IDS[ctx.kind], int(eta_lvl * 100), k, 55,
                                               ARMS.index(arm)))
            fb_pi = int(res["pi"]) if res["status"] == "CERTIFIED" else None
            fallback = {"method": "B2", "status": res["status"], "steps": int(res["steps"]), "pulls": int(res["pulls"]),
                        "n_reused": int(res["n_reused"]), "pi": fb_pi,
                        "true_regret": float(Jtq.max() - Jtq[fb_pi]) if fb_pi is not None else None}
            if res["status"] == "CERTIFIED":
                status, pi, charged = "CERTIFIED", fb_pi, steps + int(res["steps"])
            else:
                pi, charged = None, max(steps + int(res["steps"]), TMAX_TRIAL)
        elif status == "CERTIFIED" and arm == "AGC":
            Jm = J[mask]
            comp = sorted(set(np.unique(Jm.argmax(1)).tolist()) - {model_pi})
            pair_gap = (Jm - Jm[:, [model_pi]]).max(0)
            pair_gap[model_pi] = -np.inf
            top = [int(x) for x in np.argsort(-pair_gap)[:2] if np.isfinite(pair_gap[x])]
            audited = sorted(set(comp) | set(top))
            env_a, _ = ctx.make_env(s)          # audit trials on a platform copy with the same truth
            smp = trial_sampler(env_a, q, (ctx.seed, ctx.noise, KIND_IDS[ctx.kind], int(eta_lvl * 100), k, 91))
            au = agc_audit(smp, model_pi, audited, J[E.mle()], len(q.policies), q.H, EPS, DELTA / 2,
                           float(q.meta["u_max"]), TMAX_TRIAL)
            audit = {"status": au["status"], "steps": int(au["steps"]), "trials": int(au["pulls"]),
                     "phase": au["phase"], "conflict": bool(au.get("conflict")), "n_audited": len(audited),
                     "audited_contains_true_best": bool(int(np.argmax(Jtq)) in [model_pi] + audited),
                     "model_pi_regret": float(Jtq.max() - Jtq[model_pi])}
            if audit["conflict"] and agc_first_conflict is None:
                agc_first_conflict = {"problem_k": k, "cum_model_steps": cum_new,
                                      "cum_audit_steps": int(sum(r["audit"]["steps"] for r in rows if r.get("audit"))
                                                             + int(au.get("conflict_at") or au["steps"]))}
            if au["status"] == "CERTIFIED":
                pi, charged = int(au["pi"]), steps + int(au["steps"])
            else:
                status, pi = au["status"], None
                charged = max(steps + int(au["steps"]), TMAX_TRIAL)
        cert_ok = status == "CERTIFIED"
        if charged is None:
            charged = steps if cert_ok else max(steps, TMAX_STEP)
        tr = float(Jtq.max() - Jtq[pi]) if (cert_ok and pi is not None) else None
        row = {"task": TASK, "instance": ctx.seed, "stream": ctx.stream, "noise_seed": ctx.noise, "misspec": ctx.kind,
               "eta_level": eta_lvl, "strength": s, "problem": q.pid, "k": k, "method": arm, "design": "DDA",
               "status": status, "new_env_steps": int(charged), "steps_consumed_model": int(steps),
               "censored": not cert_ok, "certified_policy": pi, "true_regret": tr,
               "false_cert": bool(cert_ok and tr is not None and tr > EPS),
               "zero_cost": bool(cert_ok and charged == 0), "eta_loc": 0.0,
               "eta_q_theta_star": float(np.abs(Jtq - J[ctx.ti]).max()),
               "theta_cell_alive": bool(mask[ctx.ti]) if mask.any() else False, "set_size": int(mask.sum()),
               "true_best": int(np.argmax(Jtq)), "true_top2_gap": float(np.sort(Jtq)[-1] - np.sort(Jtq)[-2]),
               "H": q.H, "n_policies": len(q.policies), "eps": EPS, "delta": DELTA,
               "fallback": fallback, "audit": audit, "replay": replay,
               "alarm_active": bool(alarm is not None), "model_pi": model_pi,
               "falsification_scope": (E.falsification_scope() if arm == "mixed" else {"components": []}),
               "cum_new_steps_model": int(cum_new), "explore_steps": n_explore,
               "lr_n_rounds": int(E.n_rounds), "env_n_steps": int(env.n_steps), "rollouts": 0,
               "wall_clock_s": time.perf_counter() - t0, **extra}
        rows.append(row)
        if want_samples and k < 2:
            samples.append({**row, "J_true": np.round(Jtq, 4).tolist(), "J_class_at_theta_star": np.round(J[ctx.ti], 4).tolist(),
                            "trajectory_head": traj})
    stream = {"instance": ctx.seed, "stream": ctx.stream, "misspec": ctx.kind, "eta_level": eta_lvl, "method": arm,
              "alarm": alarm, "agc_first_conflict": agc_first_conflict,
              "stream_charged_steps": int(sum(r["new_env_steps"] for r in rows)),
              "stream_model_steps": int(cum_new),
              "min_gap_vs_pool": float(E.min_gap_vs_pool) if arm == "mixed" else None,
              "component_adv_final": ({c: round(float(v), 3) for c, v in E.component_advantage().items()}
                                      if arm == "mixed" else None),
              "ext_state": (E.ext.state() if arm == "mixed" and E.ext is not None else None)}
    return rows, samples, stream


def run_unit(seed, kind, eta_lvl, stream, K, want_samples):
    t0 = time.time()
    out = {"seed": seed, "kind": kind, "eta_level": eta_lvl, "stream": stream, "rows": [], "streams": [],
           "samples": [], "errors": []}
    try:
        ctx = Ctx(seed, kind, stream, K)
        s, eta_min, k_min, it, reached = ctx.calibrate(eta_lvl * EPS)
        Jt = ctx.J_true(s)
        out["calib"] = {"instance": seed, "stream": stream, "misspec": kind, "eta_level": eta_lvl, "strength": s,
                        "eta_min": eta_min, "eta_min_over_eps": eta_min / EPS, "theta_min": k_min,
                        "theta_min_is_theta_star": bool(k_min == ctx.ti), "iters": it, "reached": reached,
                        "types": ctx.types.tolist() if kind == "m3" else None,
                        "S": ctx.S.tolist() if kind == "m1" else None, "perm": ctx.perm[:K],
                        "eta_theta_star_over_eps": float(max(np.abs(Jt[k] - ctx.J[k][ctx.ti]).max()
                                                             for k in range(len(Jt))) / EPS),
                        "calib_s": time.time() - t0}
        for arm in ARMS:
            try:
                ta = time.time()
                rows, smp, st = run_arm(ctx, s, Jt, arm, eta_lvl, want_samples=want_samples)
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
HORIZON_ARMS = ("main", "ext_only", "reg_only")
HORIZON_PID_EVERY = 200


def horizon_unit(seed, kind, eta_lvl, stream, K, n_rounds):
    """How many real rounds does the zero-interaction layer need to refute the class when sampling does NOT stop at
    the first certificate? Fresh platform copy, n0 shared rounds + n_rounds uniform legal rounds (public problem id
    every 200 rounds), arms main {pool, reg, ext}, ext_only {pool, ext} (1/2-1/2), reg_only {pool, reg} (1/2-1/2).
    Same calibrated strength as the stream units. Diagnostic only: the stream protocol is what HF2b evaluates."""
    t0 = time.time()
    try:
        ctx = Ctx(seed, kind, stream, K)
        s, eta_min, *_ = ctx.calibrate(eta_lvl * EPS)
        env, init = ctx.make_env(s)
        h = env.handle()
        sets = {"main": MixedLRSet(ctx.prop, ctx.LT, DELTA, regime_fn=load_bucket, name="main"),
                "ext_only": MixedLRSet(ctx.prop, ctx.LT, DELTA, components=("pool", "ext"), weights=(0.5, 0.5)),
                "reg_only": MixedLRSet(ctx.prop, ctx.LT, DELTA, components=("pool", "reg"), weights=(0.5, 0.5))}
        for o in init:
            for m in sets.values():
                m.update(o, problem_id=None)
        rng = np.random.default_rng([seed, ctx.noise, KIND_IDS[kind], int(eta_lvl * 100), 909])
        trace = []
        for r in range(1, n_rounds + 1):
            o = h.step(int(rng.integers(h.aspace.n)))
            for m in sets.values():
                m.update(o, problem_id=f"h{r // HORIZON_PID_EVERY}")
            if r % 1000 == 0:
                trace.append({"round": N0 + r, **{a: {c: round(float(v), 2) for c, v in m.component_advantage().items()}
                                                  for a, m in sets.items()}})
        assert env.n_steps == N0 + n_rounds
        rec = {"instance": seed, "stream": stream, "misspec": kind, "eta_level": eta_lvl, "strength": s,
               "eta_min_over_eps": eta_min / EPS, "n_rounds_total": N0 + n_rounds, "design": "uniform_legal",
               "arms": {a: {"conflict_round": m.conflict_round,
                            "theta_star_pt_alive": bool(m.mask()[ctx.ti]),
                            "component_adv_final": {c: round(float(v), 2) for c, v in m.component_advantage().items()}}
                        for a, m in sets.items()},
               "trace": trace, "wall_s": time.time() - t0}
        return {"rec": rec, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"rec": None, "error": {"where": f"horizon {seed}/{kind}/{eta_lvl}", "error": repr(e),
                                       "tb": traceback.format_exc()[-1500:]}}


# ------------------------------------------------------------------ analysis
def _cp(k, n):
    if n == 0:
        return None, None
    lo, hi = clopper_pearson(int(k), int(n))
    return float(lo), float(hi)


def _med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def summarize_horizon(hz, kinds, etas, cells):
    out = {}
    for kind in kinds:
        for e in etas:
            H = [x for x in hz if x["misspec"] == kind and x["eta_level"] == e]
            if not H:
                continue
            d = {"n": len(H), "n_rounds_total": H[0]["n_rounds_total"]}
            for a in HORIZON_ARMS:
                cr = [x["arms"][a]["conflict_round"] for x in H]
                hit = [c for c in cr if c is not None]
                d[a] = {"alarm_rate": len(hit) / len(H), "alarm_rate_cp": _cp(len(hit), len(H)),
                        "alarm_round_median_among_alarmed": _med(hit), "alarm_rounds": cr,
                        "alarm_round_median_censored": (float(np.median([c if c is not None else float("inf")
                                                                         for c in cr])))}
            ag = cells.get(f"{kind}_eta{e:g}", {}).get("AGC", {}).get("audit_steps_median")
            m = d["main"]["alarm_round_median_censored"]
            d["main_alarm_median_over_agc_audit_median"] = (m / ag) if (ag and np.isfinite(m)) else None
            out[f"{kind}_eta{e:g}"] = d
    return out


def summarize(rows, streams, calibs, kinds, etas, mode, wall):
    S = {"task": TASK, "mode": mode, "eps": EPS, "delta": DELTA, "cells": {}, "hf2b": {},
         "n_rows": len(rows), "n_streams": len(streams), "wall_s": wall,
         "timing_note": "wall clock measured under concurrent runs (4 tasks share 20 cores)"}
    for kind in kinds:
        for e in etas:
            key = f"{kind}_eta{e:g}"
            cell = {}
            for arm in ARMS:
                R = [r for r in rows if r["misspec"] == kind and r["eta_level"] == e and r["method"] == arm]
                ST = [s for s in streams if s["misspec"] == kind and s["eta_level"] == e and s["method"] == arm]
                if not R:
                    continue
                n_cert = sum(r["status"] == "CERTIFIED" for r in R)
                fc = sum(r["false_cert"] for r in R)
                lo, hi = _cp(fc, n_cert)
                c = {"n_problems": len(R), "n_streams": len(ST), "n_cert": n_cert, "completion": n_cert / len(R),
                     "n_false_cert": int(fc), "fcr": fc / n_cert if n_cert else None, "fcr_cp_lower": lo,
                     "fcr_cp_upper": hi, "false_cert_per_problem": fc / len(R),
                     "status_counts": {k2: sum(r["status"] == k2 for r in R) for k2 in sorted({r["status"] for r in R})},
                     "stream_charged_steps_mean": float(np.mean([s["stream_charged_steps"] for s in ST])) if ST else None,
                     "stream_charged_steps_median": _med([s["stream_charged_steps"] for s in ST]),
                     "stream_model_steps_median": _med([s["stream_model_steps"] for s in ST]),
                     "zero_cost_rate": sum(r["zero_cost"] for r in R) / len(R),
                     "theta_star_pt_alive_rate": sum(r["theta_cell_alive"] for r in R) / len(R),
                     "mean_true_regret_certified": _med([r["true_regret"] for r in R]),
                     "n_fallback": sum(r["fallback"] is not None for r in R)}
                if arm == "mixed":
                    al = [s["alarm"] for s in ST if s["alarm"] is not None]
                    c["alarm_streams"] = len(al)
                    c["alarm_rate"] = len(al) / len(ST) if ST else None
                    c["alarm_rate_cp"] = _cp(len(al), len(ST))
                    c["alarm_round_median"] = _med([a["round"] for a in al])
                    c["alarm_new_steps_median"] = _med([a["new_steps"] for a in al])
                    c["alarm_rounds"] = [a["round"] for a in al]
                    c["alarm_problem_k"] = [a["problem_k"] for a in al]
                    c["alarm_winning_component"] = {}
                    for a in al:
                        adv = {k2: v for k2, v in a["component_adv"].items() if k2 != "pool"}
                        w = max(adv, key=adv.get) if adv else None
                        c["alarm_winning_component"][w] = c["alarm_winning_component"].get(w, 0) + 1
                    fb = [r["fallback"] for r in R if r["fallback"] is not None]
                    c["post_fallback"] = {"n": len(fb), "n_cert": sum(f["status"] == "CERTIFIED" for f in fb),
                                          "true_regret_mean": (float(np.mean([f["true_regret"] for f in fb if f["true_regret"] is not None]))
                                                               if any(f["true_regret"] is not None for f in fb) else None),
                                          "true_regret_max": (float(max(f["true_regret"] for f in fb if f["true_regret"] is not None))
                                                              if any(f["true_regret"] is not None for f in fb) else None),
                                          "n_regret_gt_eps": sum(f["true_regret"] is not None and f["true_regret"] > EPS for f in fb),
                                          "steps_median": _med([f["steps"] for f in fb])}
                    # false certificates issued before the alarm (the window the layer did not protect)
                    c["false_cert_before_alarm"] = sum(r["false_cert"] and not r["alarm_active"] for r in R)
                    rp = [r["replay"] for r in R if r["replay"] is not None]
                    c["replay_checks"] = len(rp)
                    c["replay_conflicts"] = sum(x["conflict"] for x in rp)
                    c["replay_mask_equal_incremental"] = all(x["mask_equal_incremental"] for x in rp) if rp else None
                if arm == "AGC":
                    au = [r["audit"] for r in R if r["audit"] is not None]
                    c["n_audits"] = len(au)
                    c["audit_steps_median"] = _med([a["steps"] for a in au])
                    c["audit_steps_mean"] = float(np.mean([a["steps"] for a in au])) if au else None
                    c["audit_conflicts"] = sum(a["conflict"] for a in au)
                    c["audit_phase_counts"] = {p: sum(a["phase"] == p for a in au) for p in sorted({a["phase"] for a in au})}
                    c["stream_audit_steps_median"] = _med([sum(r["audit"]["steps"] for r in R
                                                               if r["audit"] and (r["instance"], r["stream"]) == (s["instance"], s["stream"]))
                                                           for s in ST])
                    fcs = [s["agc_first_conflict"] for s in ST if s["agc_first_conflict"] is not None]
                    c["agc_detect_streams"] = len(fcs)
                cell[arm] = c
            cal = [x for x in calibs if x["misspec"] == kind and x["eta_level"] == e]
            cell["calibration"] = {"n": len(cal), "reached_all": all(x["reached"] for x in cal),
                                   "strength_median": _med([x["strength"] for x in cal]),
                                   "eta_min_over_eps_median": _med([x["eta_min_over_eps"] for x in cal]),
                                   "eta_theta_star_over_eps_median": _med([x["eta_theta_star_over_eps"] for x in cal])}
            if "mixed" in cell and "AGC" in cell:
                am, ag = cell["mixed"]["alarm_round_median"], cell["AGC"]["audit_steps_median"]
                cell["cost_ratio_alarm_round_over_agc_audit"] = (am / ag) if (am is not None and ag) else None
                an = cell["mixed"]["alarm_new_steps_median"]
                cell["cost_ratio_alarm_newsteps_over_agc_audit"] = (an / ag) if (an is not None and ag) else None
                sm, sa = cell["mixed"]["stream_charged_steps_median"], cell["AGC"]["stream_charged_steps_median"]
                cell["stream_ratio_mixed_over_agc"] = (sm / sa) if (sm is not None and sa) else None
            if "mixed" in cell and "plugin" in cell:
                sm, sp = cell["mixed"]["stream_charged_steps_median"], cell["plugin"]["stream_charged_steps_median"]
                cell["stream_ratio_mixed_over_plugin"] = (sm / sp) if (sm is not None and sp) else None
            S["cells"][key] = cell
    # HF2b gates at eta = 2 eps (evaluated per family)
    for kind in kinds:
        c = S["cells"].get(f"{kind}_eta2")
        if not c or "mixed" not in c:
            continue
        hi = c["mixed"]["fcr_cp_upper"]
        ratio = c.get("cost_ratio_alarm_round_over_agc_audit")
        S["hf2b"][kind] = {"mixed_fcr": c["mixed"]["fcr"], "mixed_fcr_cp_upper": hi,
                           "fcr_gate": (hi is not None and hi <= 2 * DELTA),
                           "alarm_round_median": c["mixed"]["alarm_round_median"],
                           "agc_audit_steps_median": c["AGC"]["audit_steps_median"] if "AGC" in c else None,
                           "cost_ratio": ratio, "cost_gate": (ratio is not None and ratio <= 0.01),
                           "alarm_rate": c["mixed"]["alarm_rate"],
                           "plugin_fcr": c["plugin"]["fcr"] if "plugin" in c else None,
                           "agc_fcr": c["AGC"]["fcr"] if "AGC" in c else None,
                           "winning_component": c["mixed"]["alarm_winning_component"],
                           "note": "pilot sizes give wide CP intervals; gates are evaluated formally only in full"
                           if mode == "pilot" else ""}
    return S


def plots(rows, streams, out_dir, kinds, etas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axs[0]
    for kind, mk in zip(kinds, ("o", "s")):
        for e, col in zip(etas, ("tab:blue", "tab:orange", "tab:red")):
            xs, ys = [], []
            for st in streams:
                if st["misspec"] != kind or st["eta_level"] != e or st["method"] != "mixed" or st["alarm"] is None:
                    continue
                aud = [r["audit"]["steps"] for r in rows if r["method"] == "AGC" and r["misspec"] == kind
                       and r["eta_level"] == e and r["instance"] == st["instance"] and r["stream"] == st["stream"]
                       and r["audit"]]
                if aud:
                    xs.append(max(np.median(aud), 1)); ys.append(max(st["alarm"]["round"], 1))
            if xs:
                ax.scatter(xs, ys, marker=mk, color=col, label=f"{kind} eta={e:g}eps", alpha=0.8)
    lim = [1, 1e7]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.plot(lim, [v / 100 for v in lim], "k:", lw=0.8, label="y = x/100 (HF2b gate)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AGC audit steps per problem (median over the stream)")
    ax.set_ylabel("mixed-numerator alarm round (incl. n0)")
    ax.set_title("Alarm vs audit cost (per stream)")
    ax.legend(fontsize=7)
    ax = axs[1]
    labels, vals, his = [], [], []
    for kind in kinds:
        for e in etas:
            for arm in ARMS:
                R = [r for r in rows if r["misspec"] == kind and r["eta_level"] == e and r["method"] == arm]
                nc = sum(r["status"] == "CERTIFIED" for r in R)
                fc = sum(r["false_cert"] for r in R)
                if nc:
                    labels.append(f"{kind}\n{e:g}\n{arm}")
                    vals.append(fc / nc)
                    his.append(_cp(fc, nc)[1])
    x = np.arange(len(labels))
    ax.bar(x, vals, color=(["tab:gray", "tab:green", "tab:purple"] * (len(labels) // 3 + 1))[:len(labels)])
    ax.scatter(x, his, marker="_", color="k", s=80, label="CP upper")
    ax.axhline(2 * DELTA, color="r", ls="--", lw=0.8, label="2 delta")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("FCR (false certs / certs)")
    ax.set_title("FCR by family, eta, arm")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "alarm_vs_agc_and_fcr.png", dpi=130)
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
    ap.add_argument("--etas", default=None)
    ap.add_argument("--kinds", default="m1,m3")
    ap.add_argument("--streams", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--horizon", type=int, default=10000, help="uniform rounds of the detection-horizon diagnostic (0 = off)")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        rng_ = lock["eval_manifest"]["per_task_ranges"][TASK][0]
        seeds = list(range(rng_[0], rng_[1] + 1))
    else:
        seeds = list(range(600, 605))
    if args.instances:
        seeds = parse_range(args.instances)
    if not pilot:
        assert all(s >= 10000 for s in seeds)
    else:
        assert all(600 <= s <= 699 for s in seeds), "pilot uses dev seeds 600-699 only"
    K = args.problems or (6 if pilot else 10)
    etas = [float(x) for x in (args.etas or ("0,2" if pilot else "0,1,2")).split(",")]
    kinds = args.kinds.split(",")
    streams_ids = [int(x) for x in (args.streams or ("0" if pilot else "0,1,2")).split(",")]
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
        done = set()
        if done_f.exists():
            for ln in done_f.read_text().splitlines():
                u = json.loads(ln)
                done.add((u["seed"], u["kind"], u["eta_level"], u["stream"]))
        jobs = [(s, kd, e, st) for s in seeds for kd in kinds for e in etas for st in streams_ids
                if (s, kd, e, st) not in done]
        log(f"mode={args.mode} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) kinds={kinds} etas={etas} streams={streams_ids} "
            f"K={K} arms={ARMS} T_max={TMAX_STEP} workers={args.workers}; units todo={len(jobs)} done={len(done)}; "
            f"generator_hash={generator_hash()}")
        total = len(jobs) + len(done)
        if jobs and not args.resummarize:
            n_samples_units = {(seeds[0], kd, max(etas), streams_ids[0]) for kd in kinds} | \
                              {(seeds[0], kd, min(etas), streams_ids[0]) for kd in kinds}
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(run_unit)(s, kd, e, st, K, (s, kd, e, st) in n_samples_units) for (s, kd, e, st) in jobs)
            n_err = 0
            for i, out in enumerate(gen):
                with open(res_f, "a") as fh:
                    for r in out["rows"]:
                        fh.write(json.dumps({"rec": "row", **r}) + "\n")
                    for st in out["streams"]:
                        fh.write(json.dumps({"rec": "stream", **st}) + "\n")
                    if "calib" in out:
                        fh.write(json.dumps({"rec": "calib", **out["calib"]}) + "\n")
                for j, sm in enumerate(out["samples"]):
                    (out_dir / "samples" / f"{out['kind']}_eta{out['eta_level']:g}_i{out['seed']}_s{out['stream']}_{sm['method']}_k{sm['k']}.json"
                     ).write_text(json.dumps(sm, indent=1, default=str))
                for e in out["errors"]:
                    n_err += 1
                    log(f"ERROR unit {out['seed']}/{out['kind']}/{out['eta_level']}/{out['stream']} {e['where']}: "
                        f"{e['error']}\n{e['tb']}")
                if not out["errors"]:
                    with open(done_f, "a") as fh:
                        fh.write(json.dumps({"seed": out["seed"], "kind": out["kind"], "eta_level": out["eta_level"],
                                             "stream": out["stream"], "wall_s": out["wall_s"]}) + "\n")
                al = [s["alarm"]["round"] if s["alarm"] else None for s in out["streams"] if s["method"] == "mixed"]
                log(f"unit {i + 1}/{len(jobs)} {out['seed']} {out['kind']} eta={out['eta_level']:g} s{out['stream']} "
                    f"{out['wall_s']:.0f}s  strength={out.get('calib', {}).get('strength')}  mixed alarm={al}  "
                    f"fc={[sum(r['false_cert'] for r in out['rows'] if r['method'] == a) for a in ARMS]}")
                progress(len(done) + i + 1, total, "units", {"errors": n_err})
        hz_f = out_dir / "horizon.jsonl"
        if args.horizon > 0 and not args.resummarize:
            hz_done = set()
            if hz_f.exists():
                for ln in hz_f.read_text().splitlines():
                    x = json.loads(ln)
                    hz_done.add((x["instance"], x["misspec"], x["eta_level"], x["stream"]))
            hjobs = [(sd, kd, e, streams_ids[0]) for sd in seeds for kd in kinds for e in (min(etas), max(etas))
                     if (sd, kd, e, streams_ids[0]) not in hz_done]
            log(f"horizon diagnostic: {len(hjobs)} jobs x {args.horizon} uniform rounds")
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(horizon_unit)(sd, kd, e, st, K, args.horizon) for (sd, kd, e, st) in hjobs)
            for i, o in enumerate(gen):
                if o["error"]:
                    log(f"ERROR horizon {o['error']['where']}: {o['error']['error']}\n{o['error']['tb']}")
                    continue
                with open(hz_f, "a") as fh:
                    fh.write(json.dumps(o["rec"]) + "\n")
                x = o["rec"]
                log(f"horizon {i + 1}/{len(hjobs)} {x['instance']} {x['misspec']} eta={x['eta_level']:g} "
                    f"conflict={ {a: v['conflict_round'] for a, v in x['arms'].items()} } {x['wall_s']:.0f}s")
                progress(total, total, "horizon", {"done": i + 1, "of": len(hjobs)})
        rows, streams, calibs = [], [], []
        for ln in res_f.read_text().splitlines():
            d = json.loads(ln)
            rec = d.pop("rec")
            (rows if rec == "row" else streams if rec == "stream" else calibs).append(d)
        summ = summarize(rows, streams, calibs, kinds, etas, args.mode, time.time() - T0)
        if hz_f.exists():
            hz = [json.loads(ln) for ln in hz_f.read_text().splitlines()]
            summ["horizon_diagnostic"] = summarize_horizon(hz, kinds, etas, summ["cells"])
        summ.update({"seeds": [seeds[0], seeds[-1]], "K": K, "streams": streams_ids, "kinds": kinds, "etas": etas,
                     "T_max_step": TMAX_STEP, "T_max_trial": TMAX_TRIAL, "generator_hash": generator_hash(),
                     "eval_seeds_touched": not pilot, "n_errors_logged": None})
        al_eta2 = [summ["cells"].get(f"{kd}_eta2", {}).get("mixed", {}).get("alarm_streams", 0) for kd in kinds]
        agc_fc = sum(c.get("AGC", {}).get("n_false_cert", 0) for c in summ["cells"].values())
        crashed = sum(1 for ln in log_f.read_text().splitlines() if "ERROR unit" in ln) if log_f.exists() else 0
        summ["pilot_gate"] = {"end_to_end": bool(rows) and crashed == 0, "n_unit_errors": crashed,
                              "mixed_alarm_at_eta2_some_family": any(a > 0 for a in al_eta2),
                              "agc_fcr_zero": agc_fc == 0}
        summ["pilot_gate"]["pass"] = all(summ["pilot_gate"][k] for k in ("end_to_end", "mixed_alarm_at_eta2_some_family",
                                                                         "agc_fcr_zero"))
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        try:
            plots(rows, streams, out_dir, kinds, etas)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed: {e!r}")
        log(f"summary written; pilot_gate={summ['pilot_gate']}; wall {time.time() - T0:.0f}s")
        result_summary = json.dumps({"pilot_gate": summ["pilot_gate"], "hf2b": summ["hf2b"]}, default=str)[:1500]
    except Exception as e:  # noqa: BLE001
        status = "failed"
        result_summary = f"{e!r}"
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    wall = time.time() - T0
    mark_done(status, result_summary)
    update_gpu_progress(status, start_iso, wall / 60,
                        {"env": "E1-Mis-m1 / E1-Mis-m3 (E1-NL-S class |Theta|=13824)", "mode": args.mode,
                         "instances": len(seeds), "K": K, "etas": etas, "streams": streams_ids, "arms": list(ARMS),
                         "workers": args.workers, "gpu_count": 0, "note": "CPU only, concurrent with other tasks"},
                        PLANNED_MIN[args.mode])
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
