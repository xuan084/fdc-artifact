"""hf4_eprocess_audit: HF4 / P6 -- stepwise e-process falsification audit vs AGC (whole-trial EB) vs fixed-n residual
chi-square, under the registered misspecification m1 (pair synergy) at eta/eps in {0, 2}.

Usage: run_hf4_eprocess_audit.py --mode {pilot,full} [--workers 4] [--instances a-b] [--problems K] [--etas 0,2]

Truth / calibration / platform: identical to hf2b_m1_m3_mixed (imported: Ctx, trial_sampler, run_b2), m1 only.

Two phases.
  Phase 1 (per (instance, eta) stream, sequential over the K problems with ledger reuse): the round-0 JPC model stage
    of the AGC arm -- in-class plug-in SeqLRSet at delta/2, pre-registered DDA acquisition, T_max = 3000 new steps /
    problem; empty set -> MODEL_CONFLICT -> B2 (no audit). Output per problem: model_pi, MLE index theta_m, audited
    set A = competitive set u top-2 worst-gap challengers (exactly the hf2b AGC rule).
  Phase 2 (per certified problem, parallel; every audit on its OWN fresh platform copy with the same truth):
    AGC       dsswm.audit.agc.agc_audit (delta/2), whole trials from s0 -- the cost reference (steps S_AGC).
    eprocess  dsswm.audit.eprocess: two-sided betting e-process on per-step Bellman residuals of the model point
              theta_m, tolerance tau (on J; per step tau/H), delta_audit = delta/2, round-robin H-step segments of
              {pi_hat} u A from the CURRENT platform state (no reset). Run anytime up to cap = min(S_AGC, EP_CAP);
              alarm step recorded. Verdict at budget B: MODEL_CONFLICT iff alarm <= B, else PASS (anytime-valid,
              so one run gives every budget). tau in {eps/2 (primary), eps (sensitivity)}.
    chi2      fixed-n residual chi-square: n whole trials per audited arm from s0, residual U - J_m(a),
              T = sum_a n * max(|mean_a| - tau, 0)^2 / var_a vs chi2_{|A|}(1 - delta/2); n = floor(B / (H |A|)).
  Budgets B = f * S_AGC (matched per problem), f in {0.001, 0.01, 0.1, 1}; f = 0.1 is the HF4 cost gate.
  PASS -> certified pi_hat; MODEL_CONFLICT -> B2 fallback (one B2 run per problem on its own platform copy, shared by
  every audit verdict that needs it; its steps are charged separately and reported).
Metrics: FCR (false certs / certs, CP), detection rate on false-cert-eligible problems (true regret(pi_hat) > eps),
  false-alarm rate on correct problems, audit steps (median), cost ratio vs AGC, e-process steps-to-alarm on
  eligible problems (the non-circular detection-cost readout), harness diagnostic max_a |J_true(a) - J_m(a)|.
Pilot: dev seeds 600-604, stream 0, K = 10, eta in {0, 2} (5 x 10 x 2 = 100 problem units).
Full: assert_locked(), eval seeds 10000-10031 x stream 0 x K = 10 x eta {0, 2}.
CPU only (gpu slot is a scheduling token); timings are measured under concurrent runs.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import chi2 as chi2_dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

import run_hf2b_m1_m3_mixed as HB  # noqa: E402
from dsswm.acquire.nl_kl_dda import dda_choose  # noqa: E402
from dsswm.audit.agc import agc_audit  # noqa: E402
from dsswm.audit.eprocess import TwoSidedEProcess, policy_value_tables  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import generator_hash  # noqa: E402

TASK = "hf4_eprocess_audit"
EPS, DELTA = HB.EPS, HB.DELTA
DA = DELTA / 2                       # audit level (matches AGC)
TOP_M = HB.TOP_M
TMAX_STEP, TMAX_TRIAL = HB.TMAX_STEP, HB.TMAX_TRIAL
LOG_THR2 = math.log(2.0 / DELTA)
EP_CAP = 300_000                     # per-problem e-process step cap (anytime run); min(S_AGC, EP_CAP)
CHI_CAP = 2_400_000
TAUS = {"half": EPS / 2, "one": EPS}
PRIMARY_TAU = "half"
# e-process variants: registered pooled audit at tau in {eps/2, eps}; per-policy averaged variant at eps/2 (sensitivity)
EP_VARIANTS = {"half": (EPS / 2, False), "one": (EPS, False), "half_pp": (EPS / 2, True)}
FRACS = (0.001, 0.01, 0.1, 1.0)
GATE_F = 0.1
CHECKPOINTS = (100, 300, 1000, 3000, 10_000, 30_000, 100_000, 300_000)
RES_ROOT = WS / "exp" / "results"
PLANNED_MIN = {"pilot": 14, "full": 55}
KIND = "m1"


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
    HB.TASK, old = TASK, HB.TASK
    try:
        HB.update_gpu_progress(status, start_iso, wall_min, snapshot, planned)
    finally:
        HB.TASK = old


def theta_tables(ctx, k):
    P = HB.worker_tables()["params"]
    nT = P["alpha"].shape[0]
    sub = {kk: (v[k:k + 1] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == nT else v) for kk, v in P.items()}
    LT, EY = ctx.prop.tables(sub)
    return LT[0].numpy(), EY[0].numpy()


# ------------------------------------------------------------------ phase 1: model stage over the stream
def phase1(seed, eta_lvl, stream, K):
    t0 = time.time()
    out = {"seed": seed, "eta_level": eta_lvl, "stream": stream, "problems": [], "errors": []}
    try:
        ctx = HB.Ctx(seed, KIND, stream, K)
        s, eta_min, k_min, it, reached = ctx.calibrate(eta_lvl * EPS)
        Jt = ctx.J_true(s)
        out["calib"] = {"instance": seed, "stream": stream, "eta_level": eta_lvl, "strength": s, "eta_min": eta_min,
                        "eta_min_over_eps": eta_min / EPS, "theta_min": k_min, "reached": reached, "iters": it}
        env, init = ctx.make_env(s)
        h = env.handle()
        E = SeqLRSet(ctx.prop, ctx.LT, DELTA / 2)
        for o in init:
            E.update(o)
        rng = np.random.default_rng([seed, ctx.noise, 1, int(eta_lvl * 100), 404])
        b2_hist = {}
        cum = 0
        for k, q in enumerate(ctx.problems):
            Reg, J = ctx.Reg[k], ctx.J[k]
            steps = 0
            while True:
                lrat = E.log_ratio().numpy()
                mask = lrat < LOG_THR2
                cert = certify_minimax(Reg, mask, EPS, TOP_M)
                sv = cert["status"].value
                if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                    break
                if steps >= TMAX_STEP:
                    sv = "NEED_DATA"
                    break
                code = ctx.prop.codec.encode(*h.observable_state())
                blk = cert["blocking"] or certify_minimax(Reg, mask, 0.0, TOP_M)["blocking"]
                kh = E.mle()
                if not blk:
                    a = int(rng.choice(ctx.legal))
                else:
                    a, _ = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk],
                                      ctx.pe[blk], LOG_THR2 - lrat[blk], ctx.inc, ctx.prop,
                                      ctx.legal, rng)
                obs = h.step(a)
                steps += 1
                E.update(obs)
            cum += steps
            Jtq = Jt[k]
            mask = E.log_ratio().numpy() < LOG_THR2
            rec = {"instance": seed, "stream": stream, "eta_level": eta_lvl, "strength": s, "k": k, "problem": q.pid,
                   "H": q.H, "n_policies": len(q.policies), "model_status": sv, "model_steps": steps,
                   "cum_model_steps": cum, "set_size": int(mask.sum()), "theta_star_alive": bool(mask[ctx.ti]),
                   "true_best": int(np.argmax(Jtq)), "J_true": Jtq.tolist(), "u_max": float(q.meta["u_max"])}
            if sv == "CERTIFIED":
                mp = int(cert["pi"])
                Jm = J[mask]
                comp = sorted(set(np.unique(Jm.argmax(1)).tolist()) - {mp})
                pg = (Jm - Jm[:, [mp]]).max(0)
                pg[mp] = -np.inf
                top = [int(x) for x in np.argsort(-pg)[:2] if np.isfinite(pg[x])]
                aud = sorted(set(comp) | set(top))
                th = int(E.mle())
                rec.update(model_pi=mp, theta_m=th, audited=aud, theta_m_is_star=bool(th == ctx.ti),
                           model_pi_regret=float(Jtq.max() - Jtq[mp]),
                           eligible_false=bool(Jtq.max() - Jtq[mp] > EPS),
                           Jm_audited=[float(J[th, a]) for a in [mp] + aud],
                           dev_audited=float(max(abs(Jtq[a] - J[th, a]) for a in [mp] + aud)),
                           dev_all=float(np.abs(Jtq - J[th]).max()))
            elif sv == "MODEL_CONFLICT":
                res = HB.run_b2(ctx, env, k, b2_hist, (seed, ctx.noise, 1, int(eta_lvl * 100), k, 55, 7))
                pi = int(res["pi"]) if res["status"] == "CERTIFIED" else None
                rec.update(model_pi=None, fallback={"status": res["status"], "steps": int(res["steps"]), "pi": pi,
                                                    "true_regret": float(Jtq.max() - Jtq[pi]) if pi is not None else None})
            out["problems"].append(rec)
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"where": "phase1", "error": repr(e), "tb": traceback.format_exc()[-1500:]})
    out["wall_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------ phase 2: audits of one certified problem
def run_eprocess(ctx, s, q, theta_m, pols, tau, cap, per_policy=False):
    LTr, EYr = theta_tables(ctx, theta_m)
    H = q.H
    Vs = [policy_value_tables(ctx.prop, LTr, EYr, q.policies[a], H, q.utility) for a in pols]
    umax = q.utility.c_q * float(np.max(q.utility.w)) * min(ctx.prop.L, ctx.prop.R)
    lo = min(float(V[1:].min() - V[:-1].max()) for V in Vs)
    hi = max(float(umax + V[1:].max() - V[:-1].min()) for V in Vs)
    if per_policy:      # one two-sided e-process per audited policy, averaged (still an e-process under H0)
    
        eps_ = [TwoSidedEProcess(tau / H, float(V[1:].min() - V[:-1].max()), float(umax + V[1:].max() - V[:-1].min()))
                for V in Vs]
        les = [0.0] * len(pols)
    else:
        ep = TwoSidedEProcess(tau / H, lo, hi)
    thr = math.log(1.0 / DA)
    env, _ = ctx.make_env(s)
    hd = env.handle()
    steps, k, alarm, le_max = 0, 0, None, 0.0
    trace = {}
    cps = [c for c in CHECKPOINTS if c <= cap] + [cap]
    ci = 0
    rsum = 0.0
    while steps < cap and alarm is None:
        i = k % len(pols)
        pi, V = q.policies[pols[i]], Vs[i]
        for t in range(H):
            if steps >= cap:
                break
            ld, en = hd.observable_state()
            c0 = ctx.prop.codec.encode(ld, en)
            obs = hd.step(pi.act(t, ld, en))
            u = q.utility.c_q * q.utility.w[t] * sum(o[3] for o in obs.outcomes if o[4])
            c1 = ctx.prop.codec.encode(obs.next_loads, obs.next_engaged)
            r = u + V[t + 1, c1] - V[t, c0]
            rsum += r
            if per_policy:
                les[i] = eps_[i].update(float(r))
                mx = max(les)
                le = mx + math.log(sum(math.exp(x - mx) for x in les) / len(les))
            else:
                le = ep.update(float(r))
            steps += 1
            le_max = max(le_max, le)
            while ci < len(cps) and steps >= cps[ci]:
                trace[str(cps[ci])] = round(le, 3)
                ci += 1
            if le >= thr:
                alarm = steps
                break
        k += 1
    return {"tau": tau, "per_policy": per_policy, "cap": int(cap), "alarm_step": alarm, "steps_run": int(steps), "log_e_max": round(le_max, 3),
            "log_e_trace": trace, "range": [lo, hi], "mean_residual_per_step": rsum / max(steps, 1),
            "threshold": thr}


def run_chi2(ctx, s, q, theta_m, pols, budgets, key):
    J = ctx.J[ctx.problems.index(q)]
    A = len(pols)
    n_need = max(max(2, int(b // (q.H * A))) for b in budgets)
    n_need = min(n_need, CHI_CAP // (q.H * A))
    env, _ = ctx.make_env(s)
    smp = HB.trial_sampler(env, q, key)
    res = {a: np.asarray(smp(a, n_need), float) - J[theta_m, a] for a in pols}
    out = {}
    for b in budgets:
        n = int(max(2, min(n_need, b // (q.H * A))))
        T = {}
        for tn, tau in TAUS.items():
            stat = 0.0
            for a in pols:
                x = res[a][:n]
                m, v = float(x.mean()), float(x.var(ddof=1)) + 1e-12
                stat += n * max(abs(m) - tau, 0.0) ** 2 / v
            T[tn] = {"stat": stat, "reject": bool(stat > chi2_dist.ppf(1 - DA, A))}
        out[str(b)] = {"n_per_arm": n, "steps": int(n * q.H * A), **T}
    return out


def phase2(rec, K):
    t0 = time.time()
    seed, eta_lvl, stream, k = rec["instance"], rec["eta_level"], rec["stream"], rec["k"]
    out = {"key": [seed, eta_lvl, stream, k], "errors": []}
    try:
        ctx = HB.Ctx(seed, KIND, stream, K)
        s = rec["strength"]
        q = ctx.problems[k]
        J = ctx.J[k]
        Jtq = np.asarray(rec["J_true"])
        mp, th, aud = rec["model_pi"], rec["theta_m"], rec["audited"]
        base_key = (seed, ctx.noise, 1, int(eta_lvl * 100), k)
        # AGC (reference)
        env_a, _ = ctx.make_env(s)
        smp = HB.trial_sampler(env_a, q, (*base_key, 91))
        ta = time.time()
        au = agc_audit(smp, mp, aud, J[th], len(q.policies), q.H, EPS, DA, float(q.meta["u_max"]), TMAX_TRIAL)
        agc_pi = au.get("pi")
        agc = {"status": au["status"], "steps": int(au["steps"]), "phase": au["phase"], "conflict": bool(au.get("conflict")),
               "pi": agc_pi, "wall_s": time.time() - ta,
               "true_regret": float(Jtq.max() - Jtq[agc_pi]) if (au["status"] == "CERTIFIED" and agc_pi is not None) else None}
        S_agc = max(agc["steps"], 1)
        budgets = [max(1, int(f * S_agc)) for f in FRACS]
        pols = [mp] + list(aud)
        # e-process (anytime, two tolerances)
        ep = {}
        for tn, (tau, pp) in EP_VARIANTS.items():
            te = time.time()
            r = run_eprocess(ctx, s, q, th, pols, tau, min(S_agc, EP_CAP), per_policy=pp)
            r["wall_s"] = time.time() - te
            ep[tn] = r
        # fixed-n chi-square
        tc = time.time()
        chi = run_chi2(ctx, s, q, th, pols, budgets, (*base_key, 95))
        chi_wall = time.time() - tc
        # B2 fallback (only if some verdict needs it)
        need_b2 = any(r["alarm_step"] is not None for r in ep.values()) or \
            any(v[tn]["reject"] for v in chi.values() for tn in TAUS)
        b2 = None
        if need_b2:
            env_b, _ = ctx.make_env(s)
            tb = time.time()
            res = HB.run_b2(ctx, env_b, k, {}, (*base_key, 97))
            pi = int(res["pi"]) if res["status"] == "CERTIFIED" else None
            b2 = {"status": res["status"], "steps": int(res["steps"]), "pi": pi, "wall_s": time.time() - tb,
                  "true_regret": float(Jtq.max() - Jtq[pi]) if pi is not None else None}
        out.update({"agc": agc, "eprocess": ep, "chi2": chi, "chi2_wall_s": chi_wall, "b2": b2, "budgets": budgets,
                    "pols": pols})
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"where": "phase2", "error": repr(e), "tb": traceback.format_exc()[-1500:]})
    out["wall_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------ per-problem verdict table
def verdicts(rec, a2):
    """Rows (method, tau, f) -> status, cert pi, false cert, audit steps, fallback steps."""
    Jt = np.asarray(rec["J_true"])
    mp = rec["model_pi"]
    rows = []
    base = {"instance": rec["instance"], "stream": rec["stream"], "eta_level": rec["eta_level"], "k": rec["k"],
            "problem": rec["problem"], "H": rec["H"], "model_steps": rec["model_steps"],
            "eligible_false": rec["eligible_false"], "model_pi_regret": rec["model_pi_regret"],
            "dev_audited": rec["dev_audited"], "n_audited": len(rec["audited"])}
    b2 = a2["b2"]

    def fb(status_conflict):
        if not status_conflict:
            return "CERTIFIED", mp, 0
        if b2 is None or b2["status"] != "CERTIFIED":
            return "MODEL_CONFLICT", None, (b2["steps"] if b2 else 0)
        return "CERTIFIED", b2["pi"], b2["steps"]

    ag = a2["agc"]
    tr = ag["true_regret"]
    rows.append({**base, "method": "AGC", "tau": None, "f": 1.0, "status": ag["status"], "certified_policy": ag["pi"],
                 "true_regret": tr, "false_cert": bool(ag["status"] == "CERTIFIED" and tr is not None and tr > EPS),
                 "alarm": bool(ag["conflict"]), "audit_steps": ag["steps"], "fallback_steps": 0,
                 "censored": ag["status"] != "CERTIFIED"})
    for f, b in zip(FRACS, a2["budgets"]):
        for tn in EP_VARIANTS:
            e = a2["eprocess"][tn]
            al = e["alarm_step"] is not None and e["alarm_step"] <= b
            st, pi, fs = fb(al)
            tr = float(Jt.max() - Jt[pi]) if pi is not None else None
            rows.append({**base, "method": "eprocess", "tau": tn, "f": f, "budget": b, "status": st,
                         "certified_policy": pi, "true_regret": tr,
                         "false_cert": bool(st == "CERTIFIED" and tr is not None and tr > EPS), "alarm": bool(al),
                         "audit_steps": int(e["alarm_step"] if al else min(b, e["steps_run"])),
                         "budget_censored_by_cap": bool(b > e["cap"] and not al), "fallback_steps": fs,
                         "censored": st != "CERTIFIED"})
        for tn in TAUS:
            c = a2["chi2"][str(b)]
            al = c[tn]["reject"]
            st, pi, fs = fb(al)
            tr = float(Jt.max() - Jt[pi]) if pi is not None else None
            rows.append({**base, "method": "chi2", "tau": tn, "f": f, "budget": b, "status": st, "certified_policy": pi,
                         "true_regret": tr, "false_cert": bool(st == "CERTIFIED" and tr is not None and tr > EPS),
                         "alarm": bool(al), "audit_steps": c["steps"], "fallback_steps": fs,
                         "censored": st != "CERTIFIED"})
    return rows


def _cp(k, n):
    if n == 0:
        return None, None
    lo, hi = clopper_pearson(int(k), int(n))
    return float(lo), float(hi)


def _med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def summarize(p1, a2map, etas, mode, wall):
    S = {"task": TASK, "mode": mode, "eps": EPS, "delta": DELTA, "delta_audit": DA, "misspec": KIND,
         "taus": TAUS, "ep_variants": EP_VARIANTS, "primary_tau": PRIMARY_TAU, "fracs": FRACS, "gate_f": GATE_F, "ep_cap": EP_CAP,
         "cells": {}, "hf4": {}, "wall_s": wall,
         "timing_note": "wall clock measured under concurrent runs (4 tasks share 20 cores)"}
    allrows = []
    for e in etas:
        P = [r for r in p1 if r["eta_level"] == e]
        cert = [r for r in P if r["model_status"] == "CERTIFIED"]
        rows = []
        for r in cert:
            key = (r["instance"], r["eta_level"], r["stream"], r["k"])
            if key in a2map and not a2map[key]["errors"]:
                rows += verdicts(r, a2map[key])
        allrows += rows
        cell = {"n_problems": len(P), "n_model_certified": len(cert),
                "model_status_counts": {s: sum(r["model_status"] == s for r in P) for s in sorted({r["model_status"] for r in P})},
                "n_audited_problems": len({(x["instance"], x["k"]) for x in rows}),
                "model_only_false_cert": sum(r["eligible_false"] for r in cert),
                "model_only_fcr": (sum(r["eligible_false"] for r in cert) / len(cert)) if cert else None,
                "model_only_fcr_cp": _cp(sum(r["eligible_false"] for r in cert), len(cert)),
                "dev_audited_median": _med([r["dev_audited"] for r in cert]),
                "dev_audited_over_eps_max": (max(r["dev_audited"] for r in cert) / EPS) if cert else None,
                "theta_m_is_star_rate": (sum(r["theta_m_is_star"] for r in cert) / len(cert)) if cert else None,
                "model_steps_median": _med([r["model_steps"] for r in P]), "methods": {}}
        agc_rows = [x for x in rows if x["method"] == "AGC"]
        S_agc_med = _med([x["audit_steps"] for x in agc_rows])
        cell["agc_audit_steps_median"] = S_agc_med
        groups = {("AGC", None, 1.0): agc_rows}
        for f in FRACS:
            for tn in EP_VARIANTS:
                for m in ("eprocess", "chi2"):
                    groups[(m, tn, f)] = [x for x in rows if x["method"] == m and x["tau"] == tn and x["f"] == f]
        for (m, tn, f), R in groups.items():
            if not R:
                continue
            nc = sum(x["status"] == "CERTIFIED" for x in R)
            fc = sum(x["false_cert"] for x in R)
            elig = [x for x in R if x["eligible_false"]]
            good = [x for x in R if not x["eligible_false"]]
            med = _med([x["audit_steps"] for x in R])
            c = {"n": len(R), "n_cert": nc, "completion": nc / len(R), "n_false_cert": fc,
                 "fcr": fc / nc if nc else None, "fcr_cp": _cp(fc, nc),
                 "n_eligible_false": len(elig), "detect_rate": (sum(x["alarm"] for x in elig) / len(elig)) if elig else None,
                 "false_alarm_rate": (sum(x["alarm"] for x in good) / len(good)) if good else None,
                 "false_alarm_cp": _cp(sum(x["alarm"] for x in good), len(good)),
                 "audit_steps_median": med, "audit_steps_mean": float(np.mean([x["audit_steps"] for x in R])),
                 "cost_ratio_vs_agc": (med / S_agc_med) if (med is not None and S_agc_med) else None,
                 "cost_ratio_vs_agc_matched_median": _med([x["audit_steps"] / max(a["audit_steps"], 1) for x in R
                                                           for a in agc_rows if (a["instance"], a["k"]) == (x["instance"], x["k"])]),
                 "fallback_steps_median_among_alarmed": _med([x["fallback_steps"] for x in R if x["alarm"]]),
                 "n_budget_censored_by_cap": sum(x.get("budget_censored_by_cap", False) for x in R)}
            cell["methods"][f"{m}|{tn}|f{f:g}" if m != "AGC" else "AGC"] = c
        # e-process anytime detection cost on eligible problems
        dc = {}
        for tn in EP_VARIANTS:
            al, cens = [], 0
            fa = []
            for r in cert:
                key = (r["instance"], r["eta_level"], r["stream"], r["k"])
                if key not in a2map or a2map[key]["errors"]:
                    continue
                epr = a2map[key]["eprocess"][tn]
                sa = a2map[key]["agc"]["steps"]
                if r["eligible_false"]:
                    if epr["alarm_step"] is None:
                        cens += 1
                    else:
                        al.append((epr["alarm_step"], epr["alarm_step"] / max(sa, 1)))
                else:
                    fa.append(epr["alarm_step"])
            dc[tn] = {"n_eligible": len(al) + cens, "n_alarmed_within_cap": len(al), "n_censored": cens,
                      "alarm_step_median": _med([a for a, _ in al]),
                      "alarm_over_matched_agc_median": _med([b for _, b in al]),
                      "alarm_steps": [a for a, _ in al],
                      "false_alarm_steps_on_correct": [x for x in fa if x is not None],
                      "n_correct": len(fa)}
        cell["eprocess_detection_cost"] = dc
        S["cells"][f"eta{e:g}"] = cell
    c2 = S["cells"].get("eta2")
    if c2:
        ref = c2["methods"].get("AGC", {})
        g = {}
        for m in ("eprocess", "chi2"):
            for tn in EP_VARIANTS:
                x = c2["methods"].get(f"{m}|{tn}|f{GATE_F:g}")
                if not x:
                    continue
                hi = x["fcr_cp"][1]
                g[f"{m}|{tn}"] = {"fcr": x["fcr"], "fcr_cp_upper": hi, "fcr_gate": bool(hi is not None and hi <= 2 * DELTA),
                                  "fcr_point_le_2delta": bool(x["fcr"] is not None and x["fcr"] <= 2 * DELTA),
                                  "cost_ratio_vs_agc": x["cost_ratio_vs_agc"],
                                  "cost_gate": bool(x["cost_ratio_vs_agc"] is not None and x["cost_ratio_vs_agc"] <= 0.1),
                                  "detect_rate": x["detect_rate"], "n_eligible_false": x["n_eligible_false"]}
                # smallest budget fraction meeting the FCR point gate
                fmin = None
                for f in FRACS:
                    y = c2["methods"].get(f"{m}|{tn}|f{f:g}")
                    if y and y["fcr"] is not None and y["fcr"] <= 2 * DELTA and y["detect_rate"] is not None \
                            and (y["n_false_cert"] == 0 or y["fcr_cp"][1] <= 2 * DELTA):
                        fmin = f
                        break
                g[f"{m}|{tn}"]["smallest_f_meeting_fcr"] = fmin
        S["hf4"] = {"agc_fcr": ref.get("fcr"), "agc_audit_steps_median": ref.get("audit_steps_median"), "arms": g,
                    "model_only_fcr": c2["model_only_fcr"],
                    "note": "pilot CP intervals are wide; formal gate only in full" if mode == "pilot" else ""}
    return S, allrows


def plots(summ, out_dir, etas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, e in zip(axs, etas):
        c = summ["cells"].get(f"eta{e:g}")
        if not c:
            continue
        labels, vals, fcr = [], [], []
        for key, m in [("AGC", "AGC")] + [(f"{mm}|{PRIMARY_TAU}|f{f:g}", f"{mm}\nf={f:g}") for mm in ("eprocess", "chi2")
                                         for f in FRACS]:
            x = c["methods"].get(key)
            if x and x["audit_steps_median"]:
                labels.append(m); vals.append(x["audit_steps_median"]); fcr.append(x["fcr"])
        xs = np.arange(len(labels))
        ax.bar(xs, vals, color=["tab:purple"] + ["tab:green"] * len(FRACS) + ["tab:gray"] * len(FRACS))
        dc = c["eprocess_detection_cost"][PRIMARY_TAU]["alarm_step_median"]
        if dc:
            ax.axhline(dc, color="tab:red", ls="--", lw=0.8, label="e-process median alarm step (eligible)")
            ax.legend(fontsize=7)
        for xi, v, fr in zip(xs, vals, fcr):
            ax.text(xi, v * 1.15, f"FCR {fr:.2f}" if fr is not None else "", ha="center", fontsize=6, rotation=90)
        ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("audit steps per problem (median, log)")
        ax.set_title(f"m1 eta={e:g}eps: audit cost (tau=eps/2)")
    fig.tight_layout()
    fig.savefig(out_dir / "audit_cost_bar.png", dpi=130)
    plt.close(fig)


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--etas", default="0,2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resummarize", action="store_true")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        r_ = lock["eval_manifest"]["per_task_ranges"][TASK][0]
        seeds = list(range(r_[0], r_[1] + 1))
    else:
        seeds = list(range(600, 605))
    if args.instances:
        seeds = parse_range(args.instances)
    if pilot:
        assert all(600 <= s <= 699 for s in seeds), "pilot uses dev seeds 600-699 only"
    else:
        assert all(s >= 10000 for s in seeds)
    K = args.problems or 10
    etas = [float(x) for x in args.etas.split(",")]
    stream = 0
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    p1_f, p2_f, log_f = out_dir / "phase1.jsonl", out_dir / "phase2.jsonl", out_dir / "run.log"

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
        p1 = [json.loads(ln) for ln in p1_f.read_text().splitlines()] if p1_f.exists() else []
        done1 = {(r["seed"], r["eta_level"]) for r in p1}
        jobs1 = [(s, e) for s in seeds for e in etas if (s, e) not in done1]
        log(f"mode={args.mode} seeds={seeds[0]}-{seeds[-1]} etas={etas} K={K} taus={TAUS} fracs={FRACS} "
            f"EP_CAP={EP_CAP} workers={args.workers}; phase1 todo={len(jobs1)}; generator_hash={generator_hash()}")
        if jobs1 and not args.resummarize:
            for o in Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                    delayed(phase1)(s, e, stream, K) for s, e in jobs1):
                for er in o["errors"]:
                    log(f"ERROR phase1 {o['seed']}/{o['eta_level']}: {er['error']}\n{er['tb']}")
                if not o["errors"]:
                    with open(p1_f, "a") as fh:
                        fh.write(json.dumps(o) + "\n")
                    p1.append(o)
                st = [r["model_status"] for r in o["problems"]]
                log(f"phase1 {o['seed']} eta={o['eta_level']:g} {o['wall_s']:.0f}s strength={o.get('calib', {}).get('strength')} "
                    f"status={st} eligible_false={[r.get('eligible_false') for r in o['problems']]}")
        probs = [r for o in p1 for r in o["problems"]]
        a2 = [json.loads(ln) for ln in p2_f.read_text().splitlines()] if p2_f.exists() else []
        a2map = {tuple(x["key"]): x for x in a2 if not x["errors"]}
        jobs2 = [r for r in probs if r["model_status"] == "CERTIFIED"
                 and (r["instance"], r["eta_level"], r["stream"], r["k"]) not in a2map]
        # longest first: eta=0 (no alarms -> run to cap)
        jobs2.sort(key=lambda r: r["eta_level"])
        log(f"phase2 todo={len(jobs2)} done={len(a2map)}")
        total = len(jobs2) + len(a2map)
        if jobs2 and not args.resummarize:
            n_err = 0
            for i, o in enumerate(Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                    delayed(phase2)(r, K) for r in jobs2)):
                for er in o["errors"]:
                    n_err += 1
                    log(f"ERROR phase2 {o['key']}: {er['error']}\n{er['tb']}")
                with open(p2_f, "a") as fh:
                    fh.write(json.dumps(o) + "\n")
                if not o["errors"]:
                    a2map[tuple(o["key"])] = o
                    ep = o["eprocess"]
                    log(f"phase2 {i + 1}/{len(jobs2)} {o['key']} {o['wall_s']:.0f}s AGC={o['agc']['status']}/{o['agc']['steps']} "
                        f"ep_alarm={ {t: v['alarm_step'] for t, v in ep.items()} } b2={o['b2']['status'] if o['b2'] else None}")
                progress(len(a2map), total, "phase2", {"errors": n_err})
        summ, rows = summarize(probs, a2map, etas, args.mode, time.time() - T0)
        with open(out_dir / "results.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps({"task": TASK, "misspec": KIND, **r}) + "\n")
        # samples: up to 8 representative problems (eligible-false first)
        pick = sorted([r for r in probs if r["model_status"] == "CERTIFIED"],
                      key=lambda r: (-int(r["eligible_false"]), -r["eta_level"], r["instance"], r["k"]))
        n_s = 0
        for r in pick:
            key = (r["instance"], r["eta_level"], r["stream"], r["k"])
            if key in a2map and n_s < 8 and (n_s < 5 or r["eta_level"] == 0):
                (out_dir / "samples" / f"eta{r['eta_level']:g}_i{r['instance']}_k{r['k']}.json").write_text(
                    json.dumps({"phase1": r, "audits": a2map[key]}, indent=1, default=str))
                n_s += 1
        crashed = sum(1 for ln in log_f.read_text().splitlines() if ln.startswith("[") and "ERROR" in ln)
        summ.update({"seeds": [seeds[0], seeds[-1]], "K": K, "stream": stream, "etas": etas, "n_problem_units": len(probs),
                     "n_phase2": len(a2map), "generator_hash": generator_hash(), "eval_seeds_touched": not pilot,
                     "calib": [o.get("calib") for o in p1]})
        cr = summ["hf4"].get("arms", {}).get(f"eprocess|{PRIMARY_TAU}", {}).get("cost_ratio_vs_agc")
        summ["pilot_gate"] = {"end_to_end": bool(rows) and crashed == 0, "n_errors": crashed,
                              "cost_ratio_measured": cr is not None,
                              "pass": bool(rows) and crashed == 0 and cr is not None}
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        try:
            plots(summ, out_dir, etas)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed: {e!r}")
        log(f"summary written; pilot_gate={summ['pilot_gate']}; hf4={json.dumps(summ['hf4'], default=str)[:1200]}; "
            f"wall {time.time() - T0:.0f}s")
        result_summary = json.dumps({"pilot_gate": summ["pilot_gate"], "hf4": summ["hf4"]}, default=str)[:1500]
    except Exception as e:  # noqa: BLE001
        status = "failed"
        result_summary = f"{e!r}"
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    wall = time.time() - T0
    mark_done(status, result_summary)
    update_gpu_progress(status, start_iso, wall / 60,
                        {"env": "E1-Mis-m1 (E1-NL-S class |Theta|=13824)", "mode": args.mode, "instances": len(seeds),
                         "K": K, "etas": etas, "stream": stream, "audits": ["AGC", "eprocess", "chi2"],
                         "taus": list(TAUS.values()), "ep_cap": EP_CAP, "workers": args.workers, "gpu_count": 0,
                         "note": "CPU only, concurrent with other tasks"}, PLANNED_MIN[args.mode])
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
