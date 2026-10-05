"""misspec_m1: H7 / H7b misspecification phase diagram on E1-Mis-m1 (pair synergy outside the E1-NL-S class).

Usage: run_misspec_m1.py --mode {pilot,full} [--workers 4] [--instances N] [--etas 0,0.5,2] [--designs ...]

Truth: kappa_dyn=high E1-NL-S instance + logit += s * S_ij (S double-centred, max|S|=1), strength s calibrated per
instance so that eta_min = min_{theta in Theta} max_{q, pi} |J_true(q,pi) - J_theta(q,pi)| = (eta_min/eps) * eps
(over the instance's whole problem stream). The learner keeps the in-class E1-NL-S grid (|Theta| = 13824).

Per (instance, eta level, problem, design) one run from the shared n0=20 initial data (no cross-problem reuse, so
the design is the only thing that differs between runs; CRN: same platform seed for all designs):
  designs  DDA (JPC acquisition) / uniform random legal / on-policy (round-robin whole trials of the candidates
           from s0) / DDA+10% uniform
  readouts JPC model-only, adaptive stopping (first CERTIFIED of the delta LR set, T_max=600) and fixed n in
           N_FIX (status of the certifier after exactly n new steps)
           JPC+AGC: first CERTIFIED of the delta/2 LR set, then AGC audit (delta/2) of pi_hat vs. binding
           challengers with real whole trials (dsswm.audit.agc)
  theta°   pseudo-true parameter of the realised design: argmin_theta sum_{rounds} KL(P_true(.|s,a) || P_theta(.|s,a))
           (exact expected per-round KL, finite class); eta_dec = max_pi |J_true - J_theta°| on the problem
  overlap  sum_rows min(p_design(s,a), p_cand(s,a)), p_cand = exact on-policy occupancy of the candidates under truth
B1 / B2 (whole-trial direct BAI, misspecification-immune reference) per (instance, eta, problem).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import fcntl
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.acquire.nl_kl_dda import (IncidenceIndex, class_prob_tables, dda_choose, kl_features,  # noqa: E402
                                     next_state_dist, single_prob_tables)
from dsswm.audit.agc import agc_audit  # noqa: E402
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix  # noqa: E402
from dsswm.envs.nl import NLEnv  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.stats.km_rmst import rmst  # noqa: E402
from dsswm.streams.generator import (DEFAULT_NL_S_GRID, NL_DEFAULTS, generator_hash, make_nl_instance,  # noqa: E402
                                     nl_class_max_jtable)

TASK = "misspec_m1"
EPS = 0.02
DELTA = 0.05
TOP_M = 5
NOISE = 42
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
LOG_THR = math.log(1.0 / DELTA)
LOG_THR2 = math.log(2.0 / DELTA)          # delta_model = delta/2 for JPC+AGC
T_MAX = 600                               # adaptive budget of the step-level designs (as nl_acquisition_factorial)
N_FIX = (150, 300)
T_TRIAL = 2_305_116                       # B1-calibrated T_max of nl_reuse_kappa_high (whole-trial methods)
DESIGN_IDS = {"DDA": 0, "uniform": 1, "onpolicy": 2, "DDA_mix10": 3}
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables"
ENV_NAME = "E1-NL-S-kappaHigh"           # class J tables are truth-free -> shared cache of nl_reuse_kappa_high


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


# ------------------------------------------------------------------ instance context
def synergy_pattern(seed, L=2, R=2):
    s = np.random.default_rng([seed, 61]).standard_normal((L, R))
    s = s - s.mean(1, keepdims=True)
    s = s - s.mean(0, keepdims=True)
    return s / max(np.abs(s).max(), 1e-12)


class Ctx:
    def __init__(self, seed, n_problems):
        torch.set_num_threads(1)
        self.seed = seed
        self.ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        self.base = make_nl_instance(seed, noise_seed=NOISE, nl_class=self.ncl, kappa_mode="high", K=n_problems)
        self.problems = self.base.problems
        self.aspace = self.base.env.aspace
        self.prop = NLPropagator(2, 2, NMAX, self.aspace, C_KNOWN, RHO_RET, device="cpu")
        self.params = self.ncl.torch_params()
        self.LT = self.prop.tables(self.params)[0]
        self.LT_np = self.LT.numpy()
        self.J = [nl_class_max_jtable(q, self.prop, self.params, cache_path=str(JT_CACHE / f"{ENV_NAME}_{q.pid}_raw.npy"))
                  for q in self.problems]
        self.Reg = [regret_matrix(J) for J in self.J]
        self.py, self.pe = class_prob_tables(self.ncl.np_params, C_KNOWN, NMAX, self.aspace.nb)
        self.inc = IncidenceIndex(self.prop.codec, self.aspace, NMAX)
        self.nA = self.aspace.n
        self.legal = np.arange(self.nA)
        self.ti = self.base.truth["theta_index"]           # harness only (scoring)
        self.S = synergy_pattern(seed)
        self.plans = [[self.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
                      for q in self.problems]
        self.Jall = np.concatenate(self.J, 1)              # (B, sum K)

    # harness side ---------------------------------------------------
    def true_params(self, s):
        tp = self.base.env.true_params()
        tp["syn"] = s * self.S
        return tp

    def make_env(self, s):
        t = self.base.truth
        env = NLEnv(t["alpha"], t["beta"], t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"], c=C_KNOWN,
                    rho_ret=RHO_RET, L=2, R=2, nmax=NMAX, budget=NL_DEFAULTS["budget"], incentive_levels=(1,),
                    seed=int(self.seed) * 1000 + NOISE, syn=s * self.S)
        rng = np.random.default_rng([self.seed, 12])          # == generator._initial_data stream
        init = [env.step(int(rng.choice(self.legal))) for _ in range(NL_DEFAULTS["n0"])]
        return env, init

    def J_true(self, s):
        LT, EY = self.prop.tables(params_to_torch(self.true_params(s), device="cpu"))
        out = []
        for q, plans in zip(self.problems, self.plans):
            u = q.utility
            out.append(np.array([float(self.prop._run_plan(pl, es, LT, EY, u.w, u.w_ret, u.c_q)[0]) for pl, es in plans]))
        return out, LT[0].numpy()

    def eta_min(self, s):
        Jt, _ = self.J_true(s)
        jt = np.concatenate(Jt)
        dev = np.abs(self.Jall - jt[None]).max(1)
        k = int(np.argmin(dev))
        return float(dev[k]), k

    def calibrate(self, target, tol=1e-4):
        if target <= 0:
            return 0.0, 0.0, self.ti, 0
        lo, hi, it = 0.0, 0.05, 0
        while self.eta_min(hi)[0] < target and hi < 20:
            lo, hi = hi, hi * 2
            it += 1
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
        return s, v, k, it

    def occupancy(self, k, LT_true_row):
        """Exact (state, action) occupancy of the uniform mixture over candidates of problem k under the truth."""
        q = self.problems[k]
        occ = np.zeros(self.prop.codec.size * self.nA)
        for pol in q.policies:
            dist = {self.prop.codec.encode(q.loads0, q.engaged0): 1.0}
            for t in range(q.H):
                nd = {}
                for code, p in dist.items():
                    ld, en = self.prop.codec.decode(code)
                    a = pol.act(t, ld, en)
                    occ[code * self.nA + a] += p / (q.H * len(q.policies))
                    nxt, pn = next_state_dist(self.prop, LT_true_row, code, a)
                    for c2, p2 in zip(nxt, pn):
                        if p2 > 0:
                            nd[int(c2)] = nd.get(int(c2), 0.0) + p * p2
                dist = nd
        return occ


# ------------------------------------------------------------------ one design run (model-only + delta/2 event)
def pseudo_true(ctx, F, row_counts, Jtrue_q, k):
    feat = row_counts @ ctx.inc.S
    kl = feat @ F                                           # (B,)
    m = float(kl.min())
    arg = np.flatnonzero(kl <= m + 1e-9 * max(1.0, m))
    dev = np.abs(ctx.J[k][arg] - Jtrue_q[None]).max(1)
    return {"theta0": int(arg[np.argmin(dev)]), "theta0_set_size": int(len(arg)), "eta_dec": float(dev.min()),
            "eta_dec_max": float(dev.max()), "theta0_is_theta_star_class_pt": bool(ctx.ti in set(arg.tolist())),
            "kl_min_per_round": m / max(row_counts.sum(), 1)}


def overlap(row_counts, occ):
    p = row_counts / max(row_counts.sum(), 1)
    return float(np.minimum(p, occ).sum())


def run_design(ctx, s, eta_lvl, k, design, lr0, init_rows, Jtrue_q, occ, F, rec_traj=False):
    q = ctx.problems[k]
    Reg, J = ctx.Reg[k], ctx.J[k]
    env, init = ctx.make_env(s)
    h = env.handle()
    lr = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    lr.cum = lr0.cum.clone(); lr.log_num = lr0.log_num; lr.n_rounds = lr0.n_rounds; lr.serials = list(lr0.serials)
    rows_cnt = init_rows.copy()
    probe_cnt = np.zeros_like(init_rows)
    rng = np.random.default_rng([ctx.seed, NOISE, int(eta_lvl * 100), k, DESIGN_IDS[design], 303])
    t0 = time.perf_counter()

    def score(pi):
        if pi is None:
            return None, False
        reg = float(Jtrue_q.max() - Jtrue_q[pi])
        return reg, reg > EPS

    ad = None          # adaptive (delta)
    ag = None          # adaptive (delta/2) event for AGC
    fixed = {}
    trial = {"pol": None, "t": 0, "n": 0}
    traj = []
    t = 0
    n_explore = 0
    while True:
        lrat = lr.log_ratio().numpy()
        mask = lrat < LOG_THR
        mask2 = lrat < LOG_THR2
        cert = certify_minimax(Reg, mask, EPS, TOP_M) if mask.any() else None
        if ad is None:
            if cert is None:
                ad = {"status": "MODEL_CONFLICT", "t": t, "pi": None}
            elif cert["status"].value == "CERTIFIED":
                ad = {"status": "CERTIFIED", "t": t, "pi": cert["pi"], "r_bar": cert["r_bar"]}
            elif t >= T_MAX:
                ad = {"status": "NEED_DATA", "t": t, "pi": cert["pi"]}
            if ad is not None:
                ad["regret"], ad["false_cert"] = score(ad["pi"] if ad["status"] == "CERTIFIED" else None)
                ad["set_size"] = int(mask.sum())
                ad.update(pseudo_true(ctx, F, rows_cnt, Jtrue_q, k))
                ad["overlap"] = overlap(rows_cnt, occ)
                ad["overlap_probe"] = overlap(probe_cnt, occ) if probe_cnt.sum() else None
                ad["theta_star_pt_in_set"] = bool(mask[ctx.ti])
        if ag is None:
            if not mask2.any():
                ag = {"status": "MODEL_CONFLICT", "t": t, "pi": None}
            else:
                c2 = certify_minimax(Reg, mask2, EPS, TOP_M)
                if c2["status"].value == "CERTIFIED":
                    pi = c2["pi"]
                    Jm = J[mask2]
                    comp = sorted(set(np.unique(Jm.argmax(1)).tolist()) - {pi})
                    pair_gap = (Jm - Jm[:, [pi]]).max(0)          # worst-case model gap of each challenger
                    pair_gap[pi] = -np.inf
                    top = [int(x) for x in np.argsort(-pair_gap)[:2] if np.isfinite(pair_gap[x])]
                    audited = sorted(set(comp) | set(top))
                    ag = {"status": "CERTIFIED", "t": t, "pi": pi, "mle": lr.mle(), "audited": audited,
                          "competitive": comp, "pair_gap_top": [float(pair_gap[x]) for x in top]}
                elif t >= T_MAX:
                    ag = {"status": "NEED_DATA", "t": t, "pi": c2["pi"], "mle": lr.mle()}
            if ag is not None:
                ag.update({"eta_dec": pseudo_true(ctx, F, rows_cnt, Jtrue_q, k)["eta_dec"]})
        if t in N_FIX:
            if cert is None:
                fx = {"status": "MODEL_CONFLICT", "pi": None}
            else:
                fx = {"status": cert["status"].value, "pi": cert["pi"], "r_bar": cert["r_bar"]}
            fx["regret"], fx["false_cert"] = score(fx["pi"] if fx["status"] == "CERTIFIED" else None)
            fx["regret_pi"] = score(fx["pi"])[0]
            fx.update(pseudo_true(ctx, F, rows_cnt, Jtrue_q, k))
            fx["overlap"] = overlap(rows_cnt, occ)
            fx["overlap_probe"] = overlap(probe_cnt, occ)
            fx["set_size"] = int(mask.sum())
            fixed[t] = fx
        if ad is not None and ag is not None and t >= max(N_FIX):
            break
        if (cert is None and not mask2.any()) and t >= max(N_FIX):
            break
        if t >= T_MAX:
            break
        # ---- choose the next real interaction
        code = ctx.prop.codec.encode(*h.observable_state())
        if design == "uniform" or (design == "DDA_mix10" and rng.random() < 0.10):
            a = int(rng.choice(ctx.legal))
            n_explore += 1
        elif design == "onpolicy":
            if trial["pol"] is None or trial["t"] >= q.H:
                h.reset_to(q.loads0, q.engaged0)
                trial = {"pol": q.policies[trial["n"] % len(q.policies)], "t": 0, "n": trial["n"] + 1}
                code = ctx.prop.codec.encode(*h.observable_state())
            ld, en = h.observable_state()
            a = int(trial["pol"].act(trial["t"], ld, en))
            trial["t"] += 1
        else:
            dm, thr = (mask, LOG_THR) if mask.any() else (mask2, LOG_THR2)
            if not dm.any():
                a = int(rng.choice(ctx.legal))
            else:
                cc = certify_minimax(Reg, dm, EPS, TOP_M)
                blk = cc["blocking"] or certify_minimax(Reg, dm, 0.0, TOP_M)["blocking"]
                kh = lr.mle()
                if not blk:
                    a = int(rng.choice(ctx.legal))
                else:
                    margins = thr - lrat[blk]
                    a, info = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk],
                                         ctx.pe[blk], margins, ctx.inc, ctx.prop, ctx.legal, rng)
                    n_explore += int(info["mode"] != "dda")
        obs = h.step(a)
        lr.update(obs)
        rows_cnt[code * ctx.nA + a] += 1
        probe_cnt[code * ctx.nA + a] += 1
        t += 1
        if rec_traj and len(traj) < 80:
            traj.append({"t": t, "a": int(a), "set_size": int(mask.sum()), "r_bar": None if cert is None else cert["r_bar"],
                         "pi": None if cert is None else cert["pi"], "mle": lr.mle()})
    assert env.n_steps == NL_DEFAULTS["n0"] + t, "every evidence round must be a real platform round"
    assert lr.n_rounds == env.n_steps
    return {"adaptive": ad, "agc_event": ag, "fixed": fixed, "steps_run": t, "n_resets": env.n_resets,
            "explore_steps": n_explore, "wall_s": time.perf_counter() - t0, "traj": traj}


# ------------------------------------------------------------------ whole-trial samplers (harness: the real platform)
def trial_sampler(env, q, key):
    counters = {}

    def sampler(a, n):
        c = counters.get(a, 0)
        counters[a] = c + 1
        r = np.random.default_rng([*key, a, c])
        ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
        return q.utility.value_from_components(ys, es)
    return sampler


# ------------------------------------------------------------------ job = (instance, eta level)
def run_job(seed, eta_lvl, n_problems, designs, with_bai):
    t_job = time.time()
    out = {"seed": seed, "eta_level": eta_lvl, "rows": [], "agc": [], "bai": [], "samples": [], "errors": []}
    try:
        ctx = Ctx(seed, n_problems)
        s, eta_min, k_min, it = ctx.calibrate(eta_lvl * EPS)
        Jt, LT_true_row = ctx.J_true(s)
        env0, init = ctx.make_env(s)
        lr0 = SeqLRSet(ctx.prop, ctx.LT, DELTA)
        init_rows = np.zeros(ctx.prop.codec.size * ctx.nA)
        for o in init:
            lr0.update(o)
            init_rows[ctx.prop.codec.encode(o.loads, o.engaged) * ctx.nA + o.action] += 1
        tp = ctx.true_params(s)
        tpy, tpe = single_prob_tables(tp, C_KNOWN, NMAX, ctx.aspace.nb)
        F = kl_features(tpy, tpe, ctx.py, ctx.pe)                       # (nf, B): KL(P_true || P_theta) features
        th_hat0 = lr0.mle()
        out["calib"] = {"strength": s, "eta_min": eta_min, "eta_min_over_eps": eta_min / EPS, "theta_min": k_min,
                        "theta_min_is_theta_star": bool(k_min == ctx.ti), "iters": it,
                        "S": ctx.S.tolist(), "eta_theta_star": float(max(np.abs(Jt[k] - ctx.J[k][ctx.ti]).max()
                                                                          for k in range(len(Jt))))}
        audit_memo = {}
        for k, q in enumerate(ctx.problems):
            Jtrue_q = Jt[k]
            occ = ctx.occupancy(k, LT_true_row)
            base = {"instance": seed, "eta_level": eta_lvl, "eta_min": eta_min, "strength": s, "problem_index": k,
                    "pid": q.pid, "H": q.H, "n_policies": len(q.policies), "eps": EPS, "delta": DELTA,
                    "eta_q_theta_star": float(np.abs(Jtrue_q - ctx.J[k][ctx.ti]).max()),
                    "true_best": int(np.argmax(Jtrue_q)), "true_gap": float(np.sort(Jtrue_q)[-1] - np.sort(Jtrue_q)[-2]),
                    "class_best_at_theta_star": int(np.argmax(ctx.J[k][ctx.ti])),
                    "regret_of_theta_star_policy": float(Jtrue_q.max() - Jtrue_q[int(np.argmax(ctx.J[k][ctx.ti]))])}
            for d in designs:
                try:
                    r = run_design(ctx, s, eta_lvl, k, d, lr0, init_rows, Jtrue_q, occ, F,
                                   rec_traj=(k < 1 and seed < 2))
                except Exception as e:  # noqa: BLE001
                    out["errors"].append({"where": f"design {d} q{k}", "error": repr(e), "tb": traceback.format_exc()})
                    continue
                ad = r["adaptive"]
                out["rows"].append({**base, "method": "JPC", "design": d, "readout": "adaptive",
                                    "status": ad["status"], "new_env_steps": int(ad["t"]),
                                    "censored": ad["status"] != "CERTIFIED", "certified_policy": ad["pi"],
                                    "true_regret": ad["regret"], "false_cert": bool(ad["false_cert"]),
                                    **{kk: ad.get(kk) for kk in ("eta_dec", "eta_dec_max", "theta0", "theta0_set_size",
                                                                 "theta0_is_theta_star_class_pt", "kl_min_per_round",
                                                                 "overlap", "overlap_probe", "set_size",
                                                                 "theta_star_pt_in_set")},
                                    "wall_clock_s": r["wall_s"], "rollouts": 0, "n_resets": r["n_resets"]})
                for n, fx in r["fixed"].items():
                    out["rows"].append({**base, "method": "JPC", "design": d, "readout": f"fixed_n{n}",
                                        "status": fx["status"], "new_env_steps": int(n), "censored": fx["status"] != "CERTIFIED",
                                        "certified_policy": fx["pi"], "true_regret": fx["regret"],
                                        "false_cert": bool(fx["false_cert"]), "regret_of_selected": fx["regret_pi"],
                                        **{kk: fx.get(kk) for kk in ("eta_dec", "eta_dec_max", "theta0", "theta0_set_size",
                                                                     "theta0_is_theta_star_class_pt", "kl_min_per_round",
                                                                     "overlap", "overlap_probe", "set_size")}})
                # ---- JPC + AGC
                ag = r["agc_event"]
                arow = {**base, "method": "JPC_AGC", "design": d, "readout": "adaptive",
                        "model_status": ag["status"], "model_steps": int(ag["t"]), "eta_dec": ag.get("eta_dec")}
                if ag["status"] == "CERTIFIED":
                    key = (k, ag["pi"], tuple(ag["audited"]))
                    if key not in audit_memo:
                        env_a, _ = ctx.make_env(s)
                        smp = trial_sampler(env_a, q, (seed, NOISE, int(eta_lvl * 100), k, 91))
                        audit_memo[key] = agc_audit(smp, ag["pi"], ag["audited"], ctx.J[k][ag["mle"]], len(q.policies),
                                                    q.H, EPS, DELTA / 2, float(q.meta["u_max"]), T_TRIAL)
                    au = audit_memo[key]
                    pi = au["pi"] if au["status"] == "CERTIFIED" else None
                    reg = float(Jtrue_q.max() - Jtrue_q[pi]) if pi is not None else None
                    aud_best = max([ag["pi"]] + ag["audited"], key=lambda x: Jtrue_q[x])
                    arow.update({"status": au["status"], "certified_policy": pi, "true_regret": reg,
                                 "false_cert": bool(reg is not None and reg > EPS),
                                 "false_cert_audited_pairs": bool(pi is not None and Jtrue_q[aud_best] - Jtrue_q[pi] > EPS),
                                 "audit_steps": int(au["steps"]), "audit_trials": int(au["pulls"]),
                                 "new_env_steps": int(ag["t"]) + int(au["steps"]), "censored": au["status"] != "CERTIFIED",
                                 "audit_phase": au["phase"], "audit_conflict": bool(au.get("conflict")),
                                 "n_audited": len(ag["audited"]), "audited_contains_true_best":
                                 bool(base["true_best"] in [ag["pi"]] + ag["audited"]),
                                 "model_pi": ag["pi"], "model_pi_regret": float(Jtrue_q.max() - Jtrue_q[ag["pi"]])})
                else:
                    arow.update({"status": ag["status"], "certified_policy": None, "true_regret": None, "false_cert": False,
                                 "false_cert_audited_pairs": False, "audit_steps": 0, "audit_trials": 0,
                                 "new_env_steps": int(ag["t"]), "censored": True, "audit_phase": "no_model_cert"})
                out["agc"].append(arow)
                if r["traj"]:
                    out["samples"].append({"instance": seed, "eta_level": eta_lvl, "problem": q.public_dict(),
                                           "design": d, "trajectory_head": r["traj"], "adaptive": ad,
                                           "agc_event": ag, "J_true": Jtrue_q.tolist(),
                                           "J_theta_star_class_pt": ctx.J[k][ctx.ti].tolist()})
            if with_bai:
                for kind, mname, tag in (("hoeffding", "B1", 55), ("eb", "B2", 56)):
                    try:
                        env_b, _ = ctx.make_env(s)
                        smp = trial_sampler(env_b, q, (seed, NOISE, int(eta_lvl * 100), k, tag))
                        t1 = time.perf_counter()
                        proxy = ctx.J[k][th_hat0] if kind == "eb" else None
                        res = lucb(smp, len(q.policies), q.H, EPS, DELTA, float(q.meta["u_max"]), T_TRIAL, kind=kind,
                                   proxy=proxy)
                        pi = res["pi"] if res["status"] == "CERTIFIED" else None
                        reg = float(Jtrue_q.max() - Jtrue_q[pi]) if pi is not None else None
                        out["bai"].append({**base, "method": mname, "design": "whole_trial", "readout": "adaptive",
                                           "status": res["status"], "new_env_steps": int(res["steps"]),
                                           "censored": bool(res["censored"]), "certified_policy": pi, "true_regret": reg,
                                           "false_cert": bool(reg is not None and reg > EPS), "trials": int(res["pulls"]),
                                           "wall_clock_s": time.perf_counter() - t1})
                    except Exception as e:  # noqa: BLE001
                        out["errors"].append({"where": f"{mname} q{k}", "error": repr(e), "tb": traceback.format_exc()})
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"where": "job", "error": repr(e), "tb": traceback.format_exc()})
    out["wall_s"] = time.time() - t_job
    return out


# ------------------------------------------------------------------ analysis
def _cp(k, n):
    lo, hi = clopper_pearson(int(k), int(n))
    return lo, hi


def _cell(rows, tau):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    fc = sum(r["false_cert"] for r in rows)
    cert = sum(r["status"] == "CERTIFIED" for r in rows)
    st = np.array([r["new_env_steps"] for r in rows], float)
    cen = np.array([r["censored"] for r in rows], bool)
    lo, hi = _cp(fc, n)
    out = {"n": n, "fcr": fc / n, "fcr_cp_lo": lo, "fcr_cp_upper": hi, "n_false_cert": int(fc), "cert_rate": cert / n,
           "fcr_given_cert": (fc / cert) if cert else None, "rmst": rmst(st, cen, tau) if tau else None,
           "median_steps_certified": float(np.median(st[~cen])) if (~cen).any() else None,
           "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}}
    ed = [r["eta_dec"] for r in rows if r.get("eta_dec") is not None]
    if ed:
        out["eta_dec_mean_over_eps"] = float(np.mean(ed) / EPS)
        out["eta_dec_median_over_eps"] = float(np.median(ed) / EPS)
    ov = [r["overlap"] for r in rows if r.get("overlap") is not None]
    if ov:
        out["overlap_mean"] = float(np.mean(ov))
    return out


def _boot_diff(rows_a, rows_b, B=2000, seed=0):
    """Instance-cluster bootstrap of FCR(a) - FCR(b) (paired by instance)."""
    inst = sorted({r["instance"] for r in rows_a} | {r["instance"] for r in rows_b})
    ga = {i: [r["false_cert"] for r in rows_a if r["instance"] == i] for i in inst}
    gb = {i: [r["false_cert"] for r in rows_b if r["instance"] == i] for i in inst}

    def stat(ids):
        a = sum(sum(ga[i]) for i in ids) / max(sum(len(ga[i]) for i in ids), 1)
        b = sum(sum(gb[i]) for i in ids) / max(sum(len(gb[i]) for i in ids), 1)
        return a - b
    rng = np.random.default_rng(seed)
    bs = [stat(list(rng.choice(inst, len(inst)))) for _ in range(B)]
    return {"diff": stat(inst), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def analyse(rows, agc, bai, calib, etas, designs, out_dir, wall):
    from scipy.stats import spearmanr
    summ = {"cells": {}, "agc": {}, "bai": {}}
    phase = []
    for e in etas:
        for d in designs:
            for ro in ["adaptive"] + [f"fixed_n{n}" for n in N_FIX]:
                rr = [r for r in rows if r["eta_level"] == e and r["design"] == d and r["readout"] == ro]
                c = _cell(rr, T_MAX if ro == "adaptive" else None)
                summ["cells"][f"eta{e}|{d}|{ro}"] = c
                phase.append({"eta_level": e, "method": "JPC_model_only", "design": d, "readout": ro, **{
                    k: c.get(k) for k in ("n", "fcr", "fcr_cp_upper", "cert_rate", "eta_dec_mean_over_eps",
                                          "eta_dec_median_over_eps", "overlap_mean", "rmst", "median_steps_certified")}})
            ra = [r for r in agc if r["eta_level"] == e and r["design"] == d]
            if ra:
                c = _cell(ra, T_TRIAL)
                n = len(ra)
                fa = sum(r["false_cert_audited_pairs"] for r in ra)
                c.update({"fcr_audited_pairs": fa / n, "fcr_audited_pairs_cp_upper": _cp(fa, n)[1],
                          "audit_conflict_rate": float(np.mean([bool(r.get("audit_conflict")) for r in ra])),
                          "median_audit_steps": float(np.median([r["audit_steps"] for r in ra if r["audit_steps"] > 0]))
                          if any(r["audit_steps"] > 0 for r in ra) else 0.0,
                          "model_cert_rate_half_delta": float(np.mean([r["model_status"] == "CERTIFIED" for r in ra])),
                          "model_pi_false_rate": float(np.mean([(r.get("model_pi_regret") or 0) > EPS for r in ra])),
                          "audited_contains_true_best": float(np.mean([bool(r.get("audited_contains_true_best")) for r in ra
                                                                       if r["model_status"] == "CERTIFIED"] or [np.nan]))})
                summ["agc"][f"eta{e}|{d}"] = c
                phase.append({"eta_level": e, "method": "JPC_AGC", "design": d, "readout": "adaptive", **{
                    k: c.get(k) for k in ("n", "fcr", "fcr_cp_upper", "cert_rate", "eta_dec_mean_over_eps",
                                          "eta_dec_median_over_eps", "rmst", "median_steps_certified")},
                              "fcr_audited_pairs": c["fcr_audited_pairs"], "median_audit_steps": c["median_audit_steps"]})
        for m in ("B1", "B2"):
            rb = [r for r in bai if r["eta_level"] == e and r["method"] == m]
            if rb:
                c = _cell(rb, T_TRIAL)
                summ["bai"][f"eta{e}|{m}"] = c
                phase.append({"eta_level": e, "method": m, "design": "whole_trial", "readout": "adaptive",
                              **{k: c.get(k) for k in ("n", "fcr", "fcr_cp_upper", "cert_rate", "rmst",
                                                       "median_steps_certified")}})
    import csv
    keys = sorted({k for p in phase for k in p})
    with open(out_dir / "phase.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for p in phase:
            w.writerow(p)

    main3 = [d for d in ("DDA", "uniform", "onpolicy") if d in designs]
    # ---- gate (a): design spread of eta_dec (fixed n = 300 readout: same budget for every design)
    spread = {}
    per_prob = []
    for e in etas:
        if e == 0:
            continue
        rr = [r for r in rows if r["eta_level"] == e and r["readout"] == "fixed_n300" and r["design"] in main3]
        means = {d: float(np.mean([r["eta_dec"] for r in rr if r["design"] == d])) for d in main3}
        spread[str(e)] = {"design_means_over_eps": {d: v / EPS for d, v in means.items()},
                          "spread_over_eps": (max(means.values()) - min(means.values())) / EPS}
        by = {}
        for r in rr:
            by.setdefault((r["instance"], r["problem_index"]), []).append(r["eta_dec"])
        sp = [(max(v) - min(v)) / EPS for v in by.values() if len(v) == len(main3)]
        per_prob += sp
        spread[str(e)]["per_problem_spread_median_over_eps"] = float(np.median(sp))
        spread[str(e)]["per_problem_spread_frac_ge_0.25"] = float(np.mean(np.array(sp) >= 0.25))
        # adaptive readout too
        ra = [r for r in rows if r["eta_level"] == e and r["readout"] == "adaptive" and r["design"] in main3]
        spread[str(e)]["adaptive_design_means_over_eps"] = {
            d: float(np.mean([r["eta_dec"] for r in ra if r["design"] == d]) / EPS) for d in main3}
    gate_a_val = max([v["spread_over_eps"] for v in spread.values()] or [0.0])
    gate_a_pp = float(np.median(per_prob)) if per_prob else 0.0
    # ---- gate (b): AGC FCR at eta 0
    agc0 = [r for r in agc if r["eta_level"] == 0 and r["design"] in main3]
    gate_b = sum(r["false_cert"] for r in agc0) == 0
    # ---- gate (c): model-only FCR increases from eta 0 to max eta (adaptive, pooled designs)
    fcr_by_eta = {}
    for e in etas:
        rr = [r for r in rows if r["eta_level"] == e and r["readout"] == "adaptive" and r["design"] in main3]
        fcr_by_eta[str(e)] = _cell(rr, T_MAX)
    gate_c = fcr_by_eta[str(max(etas))]["fcr"] > fcr_by_eta[str(min(etas))]["fcr"]
    # ---- FCR vs measured eta_dec (binned), eta*
    bins = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
    curve = {}
    for meth, src in (("JPC_model_only", [r for r in rows if r["readout"] == "adaptive"]), ("JPC_AGC", agc),
                      ("JPC_model_only_n300", [r for r in rows if r["readout"] == "fixed_n300"])):
        for d in designs:
            rr = [r for r in src if r["design"] == d and r.get("eta_dec") is not None]
            for lo, hi in zip(bins[:-1], bins[1:]):
                b = [r for r in rr if lo <= r["eta_dec"] / EPS < hi]
                if b:
                    c = _cell(b, None)
                    curve[f"{meth}|{d}|[{lo},{hi})"] = {"n": c["n"], "fcr": c["fcr"], "fcr_cp_upper": c["fcr_cp_upper"],
                                                         "cert_rate": c["cert_rate"],
                                                         "mean_steps": float(np.mean([r["new_env_steps"] for r in b]))}
    eta_star = {}
    for d in main3:
        es = [e for e in etas if summ["cells"][f"eta{e}|{d}|adaptive"]["fcr"] > DELTA]
        eta_star[d] = min(es) if es else None
    pooled_curve = {}
    rr = [r for r in rows if r["readout"] == "adaptive" and r["design"] in main3]
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [r for r in rr if lo <= r["eta_dec"] / EPS < hi]
        if b:
            pooled_curve[f"[{lo},{hi})"] = _cell(b, None)["fcr"], len(b)
    eta_star_dec = None
    for k_, (f, n) in pooled_curve.items():
        if f > DELTA and n >= 10:
            eta_star_dec = k_
            break
    # ---- H7b: design dependence of false certification
    h7b = {}
    for e in etas:
        for ro in ("adaptive", "fixed_n300", "fixed_n150"):
            rr = {d: [r for r in rows if r["eta_level"] == e and r["readout"] == ro and r["design"] == d] for d in designs}
            ent = {}
            for other in ("uniform", "onpolicy", "DDA_mix10"):
                if other in designs and "DDA" in designs:
                    ent[f"DDA_minus_{other}"] = _boot_diff(rr["DDA"], rr[other])
            h7b[f"eta{e}|{ro}"] = ent
    # matched on eta_dec: within eta_dec bins, DDA vs uniform FCR
    h7b_matched = {}
    for ro in ("adaptive", "fixed_n300"):
        for lo, hi in zip(bins[:-1], bins[1:]):
            cell = {}
            for d in main3:
                b = [r for r in rows if r["readout"] == ro and r["design"] == d and lo <= r["eta_dec"] / EPS < hi and r["eta_level"] > 0]
                if b:
                    cell[d] = {"n": len(b), "fcr": float(np.mean([r["false_cert"] for r in b]))}
            if cell:
                h7b_matched[f"{ro}|[{lo},{hi})"] = cell
    # ---- overlap mediator
    med = {}
    for ro in ("adaptive", "fixed_n300"):
        rr = [r for r in rows if r["readout"] == ro and r["eta_level"] > 0 and r["design"] in main3]
        ov = np.array([r["overlap"] for r in rr]); fc = np.array([r["false_cert"] for r in rr], float)
        ed = np.array([r["eta_dec"] for r in rr])
        rho1 = spearmanr(ov, fc)
        rho2 = spearmanr(ov, ed)
        cells = {}
        for r in rr:
            cells.setdefault((r["instance"], r["eta_level"], r["design"]), []).append(r)
        cov = np.array([np.mean([x["overlap"] for x in v]) for v in cells.values()])
        cfc = np.array([np.mean([x["false_cert"] for x in v]) for v in cells.values()])
        rho3 = spearmanr(cov, cfc)
        med[ro] = {"spearman_overlap_vs_false_cert_runs": [float(rho1.statistic), float(rho1.pvalue), len(rr)],
                   "spearman_overlap_vs_eta_dec_runs": [float(rho2.statistic), float(rho2.pvalue), len(rr)],
                   "spearman_overlap_vs_fcr_cells(instance x eta x design)": [float(rho3.statistic), float(rho3.pvalue),
                                                                               len(cells)],
                   "overlap_mean_by_design": {d: float(np.mean([r["overlap"] for r in rr if r["design"] == d]))
                                              for d in main3}}
    # ---- in-class soundness at eta 0 (all designs, all readouts)
    eta0 = [r for r in rows if r["eta_level"] == 0]
    sound0 = {"n": len(eta0), "false_cert": int(sum(r["false_cert"] for r in eta0)),
              "theta_star_pt_in_set_adaptive": float(np.mean([bool(r.get("theta_star_pt_in_set")) for r in eta0
                                                               if r["readout"] == "adaptive"]))}
    # ---- AGC cost relative to model-only and B1/B2
    cost = {}
    for e in etas:
        jm = [r for r in rows if r["eta_level"] == e and r["readout"] == "adaptive" and r["design"] == "DDA"]
        ja = [r for r in agc if r["eta_level"] == e and r["design"] == "DDA"]
        b1 = [r for r in bai if r["eta_level"] == e and r["method"] == "B1"]
        b2 = [r for r in bai if r["eta_level"] == e and r["method"] == "B2"]
        cost[str(e)] = {"JPC_DDA_rmst_T600": rmst([r["new_env_steps"] for r in jm], [r["censored"] for r in jm], T_MAX) if jm else None,
                        "AGC_DDA_mean_total_steps": float(np.mean([r["new_env_steps"] for r in ja])) if ja else None,
                        "AGC_DDA_median_audit_steps_when_audited": float(np.median([r["audit_steps"] for r in ja if r["audit_steps"] > 0]))
                        if any(r["audit_steps"] > 0 for r in ja) else None,
                        "B1_median_steps": float(np.median([r["new_env_steps"] for r in b1])) if b1 else None,
                        "B2_median_steps": float(np.median([r["new_env_steps"] for r in b2])) if b2 else None,
                        "AGC_over_B2_median_ratio": None}
        if ja and b2:
            pa = {(r["instance"], r["problem_index"]): r["new_env_steps"] for r in ja}
            pb = {(r["instance"], r["problem_index"]): r["new_env_steps"] for r in b2}
            ks = [k_ for k_ in pa if k_ in pb]
            cost[str(e)]["AGC_over_B2_median_ratio"] = float(np.median([(pa[k_] + 1) / (pb[k_] + 1) for k_ in ks]))
    summ.update({
        "task_id": TASK, "mode": "pilot", "env": "E1-Mis-m1 (pair synergy) on E1-NL-S kappa_dyn=high", "eps": EPS,
        "delta": DELTA, "delta_split_AGC": "delta_model = delta_audit = delta/2", "T_max_step_designs": T_MAX,
        "N_fix": list(N_FIX), "T_max_whole_trial": T_TRIAL, "eta_levels_over_eps": etas, "designs": designs,
        "generator_hash": generator_hash(), "calibration": calib, "gate_a_design_spread": spread,
        "gate_b_agc_fcr_eta0": {"n": len(agc0), "false_cert": int(sum(r["false_cert"] for r in agc0))},
        "fcr_model_only_by_eta_pooled": {k: {kk: v.get(kk) for kk in ("n", "fcr", "fcr_cp_lo", "fcr_cp_upper", "cert_rate",
                                                                    "eta_dec_mean_over_eps")} for k, v in fcr_by_eta.items()},
        "fcr_vs_eta_dec_binned": curve, "fcr_vs_eta_dec_pooled": pooled_curve, "eta_star_level_by_design": eta_star,
        "eta_star_dec_bin_pooled": eta_star_dec, "H7b_design_dependence": h7b, "H7b_matched_on_eta_dec": h7b_matched,
        "overlap_mediator": med, "in_class_soundness_eta0": sound0, "interaction_cost": cost, "wall_clock_s": wall})
    gate = {"a_max_design_spread_eta_dec_over_eps(n300)": gate_a_val, "a_pass": bool(gate_a_val >= 0.25),
            "a_per_problem_median_spread_over_eps": gate_a_pp,
            "b_agc_fcr_eta0_zero": bool(gate_b),
            "c_model_only_fcr_increases": bool(gate_c),
            "c_values": {k: v["fcr"] for k, v in fcr_by_eta.items()}}
    gate["all_pass"] = bool(gate["a_pass"] and gate["b_agc_fcr_eta0_zero"] and gate["c_model_only_fcr_increases"])
    summ["gate"] = gate
    summ["go_no_go"] = "GO" if gate["all_pass"] else "NO_GO"
    return summ


def plots(rows, agc, bai, etas, designs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"DDA": "C0", "uniform": "C1", "onpolicy": "C2", "DDA_mix10": "C3"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d in designs:
        for meth, src, ls in (("model-only", [r for r in rows if r["readout"] == "adaptive"], "-"), ("AGC", agc, "--")):
            xs, ys = [], []
            for e in etas:
                rr = [r for r in src if r["design"] == d and r["eta_level"] == e]
                if rr:
                    xs.append(np.mean([r["eta_dec"] for r in rr if r.get("eta_dec") is not None]) / EPS)
                    ys.append(np.mean([r["false_cert"] for r in rr]))
            ax[0].plot(xs, ys, ls, marker="o", color=cols.get(d), label=f"{meth} {d}")
    xs, ys = [], []
    for e in etas:
        rb = [r for r in bai if r["eta_level"] == e and r["method"] == "B1"]
        rr = [r for r in rows if r["eta_level"] == e and r["readout"] == "adaptive"]
        if rb:
            xs.append(np.mean([r["eta_dec"] for r in rr]) / EPS)
            ys.append(np.mean([r["false_cert"] for r in rb]))
    ax[0].plot(xs, ys, ":", marker="s", color="k", label="B1")
    ax[0].axhline(DELTA, color="gray", lw=0.8)
    ax[0].set_xlabel("measured eta_dec / eps (mean per level)"); ax[0].set_ylabel("FCR")
    ax[0].legend(fontsize=7)
    for d in designs:
        xs, ys = [], []
        for e in etas:
            rr = [r for r in agc if r["design"] == d and r["eta_level"] == e]
            rc = [r for r in rr if r["status"] == "CERTIFIED"]
            if rc:
                xs.append(np.mean([r["eta_dec"] for r in rr if r.get("eta_dec") is not None]) / EPS)
                ys.append(np.median([r["new_env_steps"] for r in rc]))
        ax[1].plot(xs, ys, marker="o", color=cols.get(d), label=f"AGC {d}")
    for m, mk in (("B1", "s"), ("B2", "^")):
        xs, ys = [], []
        for e in etas:
            rb = [r for r in bai if r["eta_level"] == e and r["method"] == m]
            rr = [r for r in rows if r["eta_level"] == e and r["readout"] == "adaptive"]
            rb = [r for r in rb if r["status"] == "CERTIFIED"]
            if rb:
                xs.append(np.mean([r["eta_dec"] for r in rr]) / EPS)
                ys.append(np.median([r["new_env_steps"] for r in rb]))
        ax[1].plot(xs, ys, ":", marker=mk, color="k", label=m)
    ax[1].set_yscale("log"); ax[1].set_xlabel("measured eta_dec / eps"); ax[1].set_ylabel("median new env steps (certified runs)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fcr_vs_eta_dec.png", dpi=130)
    plt.close(fig)


def update_gpu_progress(status, start_iso, wall_min, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": 15, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--problems", type=int, default=10)
    ap.add_argument("--etas", default=None)
    ap.add_argument("--designs", default="DDA,uniform,onpolicy,DDA_mix10")
    ap.add_argument("--no-bai", action="store_true")
    ap.add_argument("--no-progress-update", action="store_true")
    a = ap.parse_args()
    start = time.time(); start_iso = datetime.now().isoformat()
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    etas = [float(x) for x in a.etas.split(",")] if a.etas else ([0.0, 0.5, 2.0] if a.mode == "pilot" else [0.0, 0.25, 0.5, 1.0, 2.0])
    seeds = list(range(a.instances or (10 if a.mode == "pilot" else 30))) if a.mode == "pilot" else \
        list(range(10000, 10000 + (a.instances or 30)))
    designs = a.designs.split(",")
    log = open(out_dir / "run.log", "a")
    log.write(f"[{start_iso}] start mode={a.mode} seeds={seeds} etas={etas} designs={designs}\n"); log.flush()
    jobs = [(s, e) for e in etas for s in seeds]
    progress(0, len(jobs), "running")
    results = []
    par = Parallel(n_jobs=a.workers, return_as="generator_unordered")
    for i, res in enumerate(par(delayed(run_job)(s, e, a.problems, designs, not a.no_bai) for s, e in jobs)):
        results.append(res)
        log.write(f"[{datetime.now().isoformat()}] job seed={res['seed']} eta={res['eta_level']} wall={res['wall_s']:.1f}s "
                  f"rows={len(res['rows'])} errors={len(res['errors'])}\n"); log.flush()
        progress(i + 1, len(jobs), "running", {"last": f"{res['seed']}/{res['eta_level']}"})
    rows = [r for res in results for r in res["rows"]]
    agc = [r for res in results for r in res["agc"]]
    bai = [r for res in results for r in res["bai"]]
    errors = [e for res in results for e in res["errors"]]
    calib = {f"{res['seed']}|{res['eta_level']}": res.get("calib") for res in results}
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows + agc + bai:
            f.write(json.dumps(r, default=float) + "\n")
    samples = [s for res in results for s in res["samples"]][:10]
    (out_dir / "samples" / "traces.json").write_text(json.dumps(samples, default=float, indent=1))
    wall = time.time() - start
    summ = analyse(rows, agc, bai, calib, etas, designs, out_dir, wall)
    summ["errors"] = errors[:20]
    summ["n_errors"] = len(errors)
    summ["n_instances"] = len(seeds)
    summ["instance_seeds"] = seeds
    summ["problems_per_instance"] = a.problems
    summ["job_wall_s"] = {f"{res['seed']}|{res['eta_level']}": res["wall_s"] for res in results}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    try:
        plots(rows, agc, bai, etas, designs, out_dir)
    except Exception as e:  # noqa: BLE001
        log.write(f"plot error {e!r}\n")
    log.write(f"[{datetime.now().isoformat()}] done wall={wall:.1f}s go={summ['go_no_go']} gate={summ['gate']}\n")
    log.close()
    status = "success" if not errors else "success"
    mark_done(status, f"{summ['go_no_go']} gate={summ['gate']}")
    if not a.no_progress_update:
        update_gpu_progress(status, start_iso, wall / 60, {"env": "E1-Mis-m1", "instances": len(seeds),
                                                           "problems": a.problems, "etas": etas, "designs": designs,
                                                           "T_max": T_MAX, "workers": a.workers, "gpu_count": 0,
                                                           "note": "CPU only, concurrent with other tasks"})
    print(json.dumps(summ["gate"], indent=1))


if __name__ == "__main__":
    main()
