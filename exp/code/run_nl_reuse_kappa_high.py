"""nl_reuse_kappa_high: H3 / H5c / H8 main reuse-stream comparison on E1-NL-S (kappa_dyn = high, in-class).

Usage: run_nl_reuse_kappa_high.py --mode {pilot,full} [--workers 4] [--instances N] [--tmax T]

Per environment instance (= statistical unit): theta* (kappa_mode='high'), shared n0=20 initial rounds, a Type-1
stream of K problems processed in order with an evidence ledger that persists across problems, then Type-2 problems
(new incentive level b=2, never seen in the initial data).
Methods (each on its own copy of the platform with the same seed -> common random numbers):
  JPC     joint LR set over the finite class + exact minimax certificate + DDA probes (1-2 step, LP tracking)
  ORACLE  same certifier, probes chosen with the TRUE theta* as reference (oracle-guided DDA)
  B1      whole-trial LUCB, Hoeffding anytime CS, history re-use for identical (pi, s0, H)
  B2      whole-trial LUCB on residuals around a point-estimate proxy, empirical-Bernstein anytime CS
  B3      GLM / linearised transductive design (Wald ellipsoid, same per-step observations, same eps rule)
  B3g     Wald ellipsoid intersected with the finite grid + exact J (isolates 'LR set vs ellipsoid')
  B4      point-estimate SWM: argmax J_theta_hat on the initial data, no certificate (reports false confidence)
  rho*    relaxed oracle information reference (free state choice) at every JPC problem start
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # <= 4 CPU workers in total; each worker single-threaded

import argparse
import json
import math
import os
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
                                     single_prob_tables)
from dsswm.baselines.glm_linearised import (certify_linear, choose_design_action, class_theta_vecs,  # noqa: E402
                                            fisher_rounds, value_and_grad, wald_beta)
from dsswm.baselines.oracle_rho_nl import oracle_rho  # noqa: E402
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.minimax_enum import (certify_minimax, group_ids, observable_signature,  # noqa: E402
                                        regret_matrix, trichotomy)
from dsswm.envs.nl import NLEnv  # noqa: E402
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.nl_class import NLClass, NLGrid  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.stats.km_rmst import kaplan_meier, rmst  # noqa: E402
from dsswm.streams.generator import (DEFAULT_NL_S_GRID, NL_DEFAULTS, Problem, generator_hash,  # noqa: E402
                                     make_nl_instance, nl_class_max_jtable)
from dsswm.streams.policies import sample_policy, FAMILIES  # noqa: E402
from dsswm.streams.utilities import sample_utility  # noqa: E402

TASK = "nl_reuse_kappa_high"
ENV_NAME = "E1-NL-S-kappaHigh"
DELTA = 0.05
EPS = 0.02           # E1-NL pilot epsilon (control-plane decision after setup_env_core) == pre-registered full eps
TOP_M = 5
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
T2_GRID = NLGrid(**{**{k: getattr(DEFAULT_NL_S_GRID, k) for k in ("L", "R", "alpha", "gamma", "tauL", "beta", "tauR",
                                                                  "psi", "lam")}, "psi2": (0.0, 0.75, 1.5)})
LOG_THR = math.log(1.0 / DELTA)


# ------------------------------------------------------------------ scheduler protocol
def progress(res_root, step, total, phase, metric=None):
    (res_root / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(res_root, status, summary):
    pid = res_root / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = res_root / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (res_root / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------ shared per-instance context
class Ctx:
    """Learner-side public objects for one instance + harness-side truth (kept separate, used for scoring only)."""

    def __init__(self, seed, noise_seed, n_problems, cache_dir, ncl):
        torch.set_num_threads(1)
        self.seed, self.noise = seed, noise_seed
        base = make_nl_instance(seed, noise_seed=noise_seed, nl_class=ncl, kappa_mode="high", K=n_problems)
        self.base = base
        self.ncl = ncl
        self.aspace = base.env.aspace
        self.prop = NLPropagator(2, 2, NMAX, self.aspace, C_KNOWN, RHO_RET, device="cpu")
        self.params = ncl.torch_params()
        self.LT = self.prop.tables(self.params)[0]
        self.LT_np = self.LT.numpy()
        self.problems = base.problems
        self.J, self.Reg = [], []
        for q in self.problems:
            J = nl_class_max_jtable(q, self.prop, self.params, cache_path=str(cache_dir / f"{ENV_NAME}_{q.pid}_raw.npy"))
            self.J.append(J)
            self.Reg.append(regret_matrix(J))
        self.ti = base.truth["theta_index"]                       # harness only
        self.py, self.pe = class_prob_tables(ncl.np_params, C_KNOWN, NMAX, self.aspace.nb)
        self.inc = IncidenceIndex(self.prop.codec, self.aspace, NMAX)
        self.legal = np.arange(self.aspace.n)
        sig = observable_signature(self.py, self.pe, max_level=1)
        self.gid = group_ids(sig)
        self.all_singletons = len(np.unique(self.gid)) == ncl.B
        self.vecs = class_theta_vecs(ncl.np_params)

    def fresh_instance(self):
        """A new copy of the platform (same seeds => same initial data and same per-step random stream)."""
        return make_nl_instance(self.seed, noise_seed=self.noise, nl_class=self.ncl, kappa_mode="high",
                                K=len(self.problems))


def score_row(ctx, k, method, status, pi, steps, censored, extra=None, J=None, ti=None, ptype=1):
    J = ctx.J[k] if J is None else J
    ti = ctx.ti if ti is None else ti
    Jt = J[ti]
    regret = float(Jt.max() - Jt[pi]) if pi is not None else None
    certified = status == "CERTIFIED"
    row = {"instance": ctx.seed, "noise_seed": ctx.noise, "problem_index": k, "ptype": ptype, "method": method,
           "status": status, "new_env_steps": int(steps), "censored": bool(censored), "certified_policy": pi,
           "true_regret": regret, "false_cert": bool(certified and regret is not None and regret > EPS),
           "eps": EPS, "delta": DELTA}
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ set-based methods (JPC / ORACLE)
def run_jpc(ctx, tmax, oracle=False, log=None, samples=None):
    inst = ctx.fresh_instance()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    led = EvidenceLedger([lr])
    for o in inst.init_obs:
        led.record(o, "initial")
    all_obs = list(inst.init_obs)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 777 if oracle else 101])
    true_py = single_prob_tables(env.true_params(), C_KNOWN, NMAX, ctx.aspace.nb) if oracle else None
    true_LT_row = ctx.prop.tables(params_single(env.true_params()))[0][0].numpy() if oracle else None
    rows = []
    method = "ORACLE_DDA" if oracle else "JPC"
    for k, q in enumerate(ctx.problems):
        Reg = ctx.Reg[k]
        t0 = time.perf_counter()
        steps, n_two, n_explore, n_zero = 0, 0, 0, 0
        mask0 = lr.mask().numpy()
        size0 = int(mask0.sum())
        # relaxed oracle information reference rho* given the evidence at problem start (JPC only)
        rho_ref = None
        if not oracle:
            rho_ref = rho_star(ctx, k, lr, mask0)
        status = None
        traj = []
        while True:
            mask = lr.mask().numpy()
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            if cert["status"].value == "CERTIFIED":
                status = "CERTIFIED"
                break
            if cert["status"].value == "MODEL_CONFLICT":
                status = "MODEL_CONFLICT"
                break
            if not ctx.all_singletons and steps % 10 == 0:
                amb, _, _ = trichotomy(Reg, mask, ctx.gid, EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= tmax:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            blk = cert["blocking"]
            lrat = lr.log_ratio().numpy()
            margins = LOG_THR - lrat[blk]
            code = ctx.prop.codec.encode(*h.observable_state())
            if oracle:
                ref_py, ref_pe = true_py
                LT_row = true_LT_row
            else:
                ref_py, ref_pe = ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1]
                LT_row = ctx.LT_np[kh]
            a, info = dda_choose(code, ref_py, ref_pe, LT_row, ctx.py[blk], ctx.pe[blk], margins, ctx.inc, ctx.prop,
                                 ctx.legal, rng)
            n_two += int(info.get("two_step") or 0)
            n_explore += int(info["mode"] == "explore")
            n_zero += int(info["mode"] == "zero_info")
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            all_obs.append(obs)
            steps += 1
            if samples is not None and len(traj) < 400:
                traj.append({"step": steps, "action": a, "set_size": int(mask.sum()), "r_bar": cert["r_bar"],
                             "pi": cert["pi"], "mle": kh, "mode": info["mode"], "z": info.get("z")})
        assert env.n_steps == len(all_obs), "step accounting mismatch"
        assert lr.n_rounds == len(all_obs), "evidence must contain exactly the real platform rounds"
        mask = lr.mask().numpy()
        cert = certify_minimax(Reg, mask, EPS, TOP_M)
        extra = {"set_size_start": size0, "set_size_end": int(mask.sum()), "zero_cost_cert": bool(steps == 0 and
                 status == "CERTIFIED"), "theta_star_in_set": bool(mask[ctx.ti]), "r_bar_end": cert["r_bar"],
                 "two_step_frac": n_two / max(steps, 1), "explore_steps": n_explore, "zero_info_steps": n_zero,
                 "wall_clock_s": time.perf_counter() - t0, "rollouts": 0, "H": q.H, "n_policies": len(q.policies)}
        if rho_ref is not None:
            extra.update({"rho_star": rho_ref["T_star"], "rho_star_n_alt": rho_ref["n_alt"]})
        rows.append(score_row(ctx, k, method, status, cert["pi"], steps, status != "CERTIFIED", extra))
        if samples is not None and k < 3 and not oracle and ctx.seed < 2:
            samples.append({"instance": ctx.seed, "problem": q.public_dict(), "method": method, "status": status,
                            "steps": steps, "trajectory_head": traj[:60], "J_true": ctx.J[k][ctx.ti].tolist(),
                            "certified_policy": cert["pi"], "final_set_size": int(mask.sum())})
    return rows, {"obs": all_obs, "env": env, "lr": lr}


def params_single(p):
    from dsswm.exact.nl_propagate import params_to_torch
    return params_to_torch(p, device="cpu")


def rho_star(ctx, k, lr, mask):
    """Relaxed oracle reference (harness side): uses theta* only to define the reference distribution."""
    Reg = ctx.Reg[k]
    J = ctx.J[k]
    pi_star = int(np.argmax(J[ctx.ti]))
    alt = np.flatnonzero(mask & (Reg[:, pi_star] > EPS))
    if len(alt) == 0:
        return {"T_star": 0.0, "n_alt": 0}
    margins = LOG_THR - lr.log_ratio().numpy()[alt]
    if not hasattr(ctx, "S_unique"):
        ctx.S_unique = np.unique(ctx.inc.S, axis=0).astype(np.float64)
    tp = ctx.py[ctx.ti:ctx.ti + 1], ctx.pe[ctx.ti:ctx.ti + 1]
    r = oracle_rho(tp[0], tp[1], ctx.py[alt], ctx.pe[alt], margins, ctx.S_unique)
    return r


# ------------------------------------------------------------------ B3 / B3g (GLM linearised)
def run_b3(ctx, tmax, grid_mode=False):
    inst = ctx.fresh_instance()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(ctx.prop, ctx.LT, DELTA)          # used only for the grid MLE theta_hat
    led = EvidenceLedger([lr])
    for o in inst.init_obs:
        led.record(o, "initial")
    hist = {}                                       # (state code, action) -> count
    for o in inst.init_obs:
        key = (ctx.prop.codec.encode(o.loads, o.engaged), o.action)
        hist[key] = hist.get(key, 0) + 1
    d = ctx.vecs.shape[1]
    beta = wald_beta(d, DELTA)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 303 if not grid_mode else 304])
    method = "B3g" if grid_mode else "B3"
    cache_V = {}

    def build_V(kh):
        v = ctx.vecs[kh]
        V = np.eye(d)
        by_code = {}
        for (code, a), n in hist.items():
            by_code.setdefault(code, []).append((a, n))
        for code, lst in by_code.items():
            F = fisher_rounds(v, ctx.prop.codec, ctx.aspace, code, [a for a, _ in lst], C_KNOWN, 2, 2)
            V += np.einsum("k,kde->de", np.array([n for _, n in lst], float), F)
        return V

    rows = []
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()
        plans = [ctx.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
        grad_cache = {}
        steps = 0
        kh_prev = None
        V = None
        status = None
        kappa_p = None
        lin_err = None
        while True:
            kh = lr.mle()
            v = ctx.vecs[kh]
            if kh != kh_prev:
                V = build_V(kh)
                kh_prev = kh
            if kh not in grad_cache:
                Jlin, G = value_and_grad(ctx.prop, plans, v, q.utility, 2, 2)
                grad_cache[kh] = (Jlin, G)
                if lin_err is None:
                    lin_err = float(np.abs(Jlin - ctx.J[k][kh]).max())
            Jlin, G = grad_cache[kh]
            Vinv = np.linalg.inv(V)
            lc = certify_linear(ctx.J[k][kh], G, Vinv, beta, EPS)
            if kappa_p is None:
                kappa_p = kappa_participant(lc, Vinv)
            if grid_mode:
                diff = ctx.vecs - v[None]
                maha = np.einsum("bd,de,be->b", diff, V, diff)
                gmask = maha <= beta ** 2
                cert = certify_minimax(ctx.Reg[k], gmask, EPS, TOP_M)
                done = cert["status"].value == "CERTIFIED"
                pi = cert["pi"]
            else:
                done = lc["certified"]
                pi = lc["pi"]
            if done:
                status = "CERTIFIED"
                break
            if steps >= tmax:
                status = "NEED_DATA"
                break
            code = ctx.prop.codec.encode(*h.observable_state())
            Fa = fisher_rounds(v, ctx.prop.codec, ctx.aspace, code, ctx.legal, C_KNOWN, 2, 2)
            if grid_mode:
                blk = cert["blocking"]
                if rng.random() < 0.05 or not blk:
                    a = int(rng.choice(ctx.legal))
                else:
                    Db = ctx.vecs[blk] - v[None]
                    marg = np.maximum(beta ** 2 - np.einsum("bd,de,be->b", Db, V, Db), 1e-9)
                    gain = np.einsum("bd,ade,be->ab", Db, Fa, Db) / marg[None]
                    a = int(ctx.legal[int(np.argmax(gain.min(1)))])
            else:
                blkD = lc["D"][lc["ub"] > EPS]
                a = choose_design_action(Fa, V, blkD, ctx.legal, rng)
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            key = (code, a)
            hist[key] = hist.get(key, 0) + 1
            V = V + Fa[list(ctx.legal).index(a)]
            steps += 1
        extra = {"wall_clock_s": time.perf_counter() - t0, "rollouts": 0, "H": q.H, "n_policies": len(q.policies),
                 "kappa_participant": kappa_p, "lin_value_err": lin_err, "wald_beta": beta,
                 "zero_cost_cert": bool(steps == 0 and status == "CERTIFIED")}
        rows.append(score_row(ctx, k, method, status, pi, steps, status != "CERTIFIED", extra))
    return rows


BLOCKS = {"left0": [0, 2, 4], "left1": [1, 3, 5], "right0": [6, 8], "right1": [7, 9], "global": [10, 11]}


def kappa_participant(lc, Vinv):
    """kappa_P(d) = sum_blocks ||d_b||_{V^-1} / ||d||_{V^-1} for the binding challenger direction (linearised)."""
    ub = lc["ub"].copy()
    ub[lc["pi"]] = -np.inf
    b = int(np.argmax(ub))
    dvec = lc["D"][b]
    tot = math.sqrt(max(dvec @ Vinv @ dvec, 0.0))
    if tot <= 1e-15:
        return None
    s = 0.0
    for idx in BLOCKS.values():
        db = np.zeros_like(dvec); db[idx] = dvec[idx]
        s += math.sqrt(max(db @ Vinv @ db, 0.0))
    return s / tot


# ------------------------------------------------------------------ B1 / B2 / B4
def run_trials(ctx, tmax_trial, kind):
    inst = ctx.fresh_instance()
    env = inst.env                                  # harness: the trial sampler IS the platform
    lr = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    for o in inst.init_obs:
        lr.update(o)
    kh0 = lr.mle()
    method = "B1" if kind == "hoeffding" else "B2"
    history = {}
    rows = []
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()
        keys = [(p.name, tuple(int(x) for x in q.loads0), tuple(int(x) for x in q.engaged0), q.H) for p in q.policies]
        prior = {}
        for a, key in enumerate(keys):
            if key in history:
                ys, es = history[key]
                prior[a] = list(q.utility.value_from_components(ys.astype(float), es.astype(float)))
        counters = {}

        def sampler(a, n, q=q, k=k):
            c = counters.get(a, 0)
            counters[a] = c + 1
            r = np.random.default_rng([ctx.seed, ctx.noise, k, a, c, 55])
            ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
            store = sampler.raw.setdefault(a, ([], []))
            store[0].append(ys.astype(np.int8)); store[1].append(es.astype(np.int8))
            return q.utility.value_from_components(ys, es)
        sampler.raw = {}
        proxy = ctx.J[k][kh0] if kind == "eb" else None
        res = lucb(sampler, len(q.policies), q.H, EPS, DELTA, float(q.meta["u_max"]), tmax_trial,
                   kind=("hoeffding" if kind == "hoeffding" else "eb"), proxy=proxy, prior=prior)
        # store the trajectories actually consumed for re-use by later problems
        for a, key in enumerate(keys):
            if a in sampler.raw:
                ys = np.concatenate(sampler.raw[a][0]); es = np.concatenate(sampler.raw[a][1])
                n_used = int(res["n_per_arm"][a]) - len(prior.get(a, []))
                ys, es = ys[:n_used], es[:n_used]
                if key in history:
                    ys = np.concatenate([history[key][0], ys]); es = np.concatenate([history[key][1], es])
                history[key] = (ys, es)
        extra = {"wall_clock_s": time.perf_counter() - t0, "rollouts": int(res["pulls"]), "trials": int(res["pulls"]),
                 "reused_trials": int(res["n_reused"]), "H": q.H, "n_policies": len(q.policies),
                 "final_gap": res.get("gap"), "zero_cost_cert": bool(res["steps"] == 0 and res["status"] == "CERTIFIED")}
        rows.append(score_row(ctx, k, method, res["status"], res["pi"], res["steps"], res["censored"], extra))
    return rows


def run_b4(ctx):
    inst = ctx.fresh_instance()
    lr = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    for o in inst.init_obs:
        lr.update(o)
    kh = lr.mle()
    rows = []
    for k, q in enumerate(ctx.problems):
        pi = int(np.argmax(ctx.J[k][kh]))
        rows.append(score_row(ctx, k, "B4", "CERTIFIED", pi, 0, False,
                              {"note": "point estimate, claims certainty without a certificate", "H": q.H,
                               "n_policies": len(q.policies), "rollouts": 0, "wall_clock_s": 0.0,
                               "zero_cost_cert": True}))
    return rows


# ------------------------------------------------------------------ Type-2 (new incentive level b = 2)
def t2_problems(seed, aspace2, n=3):
    """q0 / q1: 3 level-1 + 2 level-2 candidates; q2: 5 level-2 candidates only (exercises OUT_OF_SCOPE)."""
    prng = np.random.default_rng([seed, 31])
    probs = []
    L, R = aspace2.L, aspace2.R
    for k in range(n):
        H = int(prng.choice((6, 8)))
        loads0 = prng.integers(0, NMAX + 1, L + R)
        eng0 = (prng.random(L + R) < 0.85).astype(np.int64)
        aff = prng.standard_normal((L, R))
        pols, names = [], set()
        while len(pols) < (3 if k < 2 else 0):
            p = sample_policy(str(prng.choice(FAMILIES)), prng, aspace2, H, aff, NMAX, level=1)
            if p.name not in names:
                names.add(p.name); pols.append(p)
        while len(pols) < 5:
            p = sample_policy(str(prng.choice(("front_incentive", "uniform_incentive", "late_incentive"))), prng,
                              aspace2, H, aff, NMAX, level=2)
            if p.name not in names:
                names.add(p.name); pols.append(p)
        util = sample_utility(prng, H, L, R, with_retention=True, y_scale=1.0)
        probs.append(Problem(f"nl{seed}_t2q{k}", 2, H, loads0, eng0, pols, util, aff))
    return probs


def t2_jtables(seeds, cache_dir, log):
    """GPU phase (main process): exact J tables on the extended class (psi2 grid) for Type-2 problems."""
    ncl2 = NLClass(T2_GRID, device="cuda")
    params2 = ncl2.torch_params()
    out = {}
    from dsswm.core.actions import ActionSpace
    aspace2 = ActionSpace(2, 2, 1, (1, 2))
    prop2 = NLPropagator(2, 2, NMAX, aspace2, C_KNOWN, RHO_RET, device="cuda")
    for s in seeds:
        for q in t2_problems(s, aspace2):
            path = cache_dir / f"E1-NL-S-T2_{q.pid}_raw.npy"
            if not path.exists():
                raw = prop2.j_table(params2, q.policies, q.loads0, q.engaged0, q.H,
                                    type(q.utility)(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0))
                np.save(path, raw)
            out[q.pid] = str(path)
    log(f"Type-2 J tables ready ({len(out)} problems, |Theta_ext|={ncl2.B})")
    del params2
    torch.cuda.empty_cache()


def run_type2(ctx, jpc_state, tmax, cache_dir):
    """Type-2: JPC continues from its Type-1 ledger on the extended class; B4 for comparison."""
    from dsswm.core.actions import ActionSpace
    ncl2 = NLClass(T2_GRID, device="cpu")
    params2 = ncl2.torch_params()
    aspace2 = ActionSpace(2, 2, 1, (1, 2))
    prop2 = NLPropagator(2, 2, NMAX, aspace2, C_KNOWN, RHO_RET, device="cpu")
    LT2 = prop2.tables(params2)[0]
    LT2_np = LT2.numpy()
    py2, pe2 = class_prob_tables(ncl2.np_params, C_KNOWN, NMAX, 3)
    inc2 = IncidenceIndex(prop2.codec, aspace2, NMAX)
    tr = ctx.base.truth
    psi2_true = float(np.random.default_rng([ctx.seed, 32]).choice(T2_GRID.psi2))
    ti2 = ncl2.index_of(tuple(tr["alpha"]), tuple(tr["beta"]), tuple(tr["gamma"]), tuple(tr["tauL"]), tuple(tr["tauR"]),
                        tr["psi"], tr["lam"], psi2_true)
    env1 = jpc_state["env"]
    rows = []
    probs = t2_problems(ctx.seed, aspace2)
    variants = [("T2b_no_level2_probes", probs[1], 1), ("T2c_level2_only_no_level2_probes", probs[2], 1),
                ("T2a_level2_probes_legal", probs[0], 2)]
    # LR set on the extended class, rebuilt from the same authenticated observations (initial + Type-1 probes)
    base_lr = SeqLRSet(prop2, LT2, DELTA)
    for o in jpc_state["obs"]:
        base_lr.update(o)
    init_lr = SeqLRSet(prop2, LT2, DELTA)
    for o in jpc_state["obs"][:NL_DEFAULTS["n0"]]:
        init_lr.update(o)
    for vname, q, maxlev in variants:
        J = nl_class_max_jtable(q, prop2, params2, cache_path=str(cache_dir / f"E1-NL-S-T2_{q.pid}_raw.npy"))
        Reg = regret_matrix(J)
        legal = np.array([a for a in range(aspace2.n) if aspace2.max_level(a) <= maxlev])
        gid = group_ids(observable_signature(py2, pe2, max_level=maxlev))
        lr = SeqLRSet(prop2, LT2, DELTA)
        lr.cum, lr.log_num, lr.n_rounds = base_lr.cum.clone(), base_lr.log_num, base_lr.n_rounds
        env2 = NLEnv(np.array(tr["alpha"]), np.array(tr["beta"]), np.array(tr["gamma"]), np.array(tr["tauL"]),
                     np.array(tr["tauR"]), [tr["psi"], psi2_true], tr["lam"], c=C_KNOWN, rho_ret=RHO_RET, L=2, R=2,
                     nmax=NMAX, budget=1, incentive_levels=(1, 2), seed=ctx.seed * 1000 + ctx.noise + 7 + maxlev)
        env2.loads, env2.engaged = env1.loads.copy(), env1.engaged.copy()
        h2 = env2.handle()
        # ground-truth label (finite-class Prop. 3): is the truth's legal-equivalence class decision-ambiguous?
        mask0 = lr.mask().numpy()
        members = (gid == gid[ti2]) & mask0
        amb_true = bool(Reg[members].max(0).min() > EPS) if members.any() else True
        c0 = certify_minimax(Reg, mask0, EPS, TOP_M)
        label = "OUT_OF_SCOPE" if amb_true else ("CERTIFIABLE_NOW" if c0["status"].value == "CERTIFIED" else "NEED_DATA")
        rng = np.random.default_rng([ctx.seed, ctx.noise, 909, maxlev])
        steps, status, n_l2 = 0, None, 0
        t0 = time.perf_counter()
        while True:
            mask = lr.mask().numpy()
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            if cert["status"].value in ("CERTIFIED", "MODEL_CONFLICT"):
                status = cert["status"].value
                break
            if steps % 10 == 0:
                amb, ncls, namb = trichotomy(Reg, mask, gid, EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= tmax:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            blk = cert["blocking"]
            margins = LOG_THR - lr.log_ratio().numpy()[blk]
            code = prop2.codec.encode(*h2.observable_state())
            a, info = dda_choose(code, py2[kh:kh + 1], pe2[kh:kh + 1], LT2_np[kh], py2[blk], pe2[blk], margins, inc2,
                                 prop2, legal, rng)
            lr.update(h2.step(a))
            n_l2 += int(aspace2.max_level(a) == 2)
            steps += 1
        cert = certify_minimax(Reg, lr.mask().numpy(), EPS, TOP_M)
        pi = cert["pi"]
        Jt = J[ti2]
        reg = float(Jt.max() - Jt[pi]) if pi is not None else None
        common = {"instance": ctx.seed, "noise_seed": ctx.noise, "problem_index": q.pid, "ptype": 2, "variant": vname,
                  "true_label": label, "eps": EPS, "delta": DELTA, "psi2_true": psi2_true,
                  "n_policies": len(q.policies), "H": q.H}
        rows.append({**common, "method": "JPC", "status": status, "new_env_steps": steps,
                     "censored": status not in ("CERTIFIED", "OUT_OF_SCOPE"), "certified_policy": pi,
                     "true_regret": reg, "false_cert": bool(status == "CERTIFIED" and reg > EPS),
                     "status_at_start": c0["status"].value, "wall_clock_s": time.perf_counter() - t0,
                     "level2_probe_steps": int(n_l2)})
        kh0 = init_lr.mle()
        pi4 = int(np.argmax(J[kh0]))
        reg4 = float(Jt.max() - Jt[pi4])
        rows.append({**common, "method": "B4", "status": "CERTIFIED", "new_env_steps": 0, "censored": False,
                     "certified_policy": pi4, "true_regret": reg4, "false_cert": bool(reg4 > EPS)})
    return rows


# ------------------------------------------------------------------ one instance
def run_instance(seed, noise, n_problems, tmax, tmax_trial, cache_dir, methods, with_t2):
    t0 = time.time()
    ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    ctx = Ctx(seed, noise, n_problems, cache_dir, ncl)
    rows, t2rows, samples, errors, timing = [], [], [], [], {}
    jpc_state = None
    for m in methods:
        tm = time.time()
        try:
            if m == "JPC":
                r, jpc_state = run_jpc(ctx, tmax, samples=samples)
            elif m == "ORACLE_DDA":
                r, _ = run_jpc(ctx, tmax, oracle=True)
            elif m == "B3":
                r = run_b3(ctx, tmax)
            elif m == "B3g":
                r = run_b3(ctx, tmax, grid_mode=True)
            elif m == "B1":
                r = run_trials(ctx, tmax_trial, "hoeffding")
            elif m == "B2":
                r = run_trials(ctx, tmax_trial, "eb")
            elif m == "B4":
                r = run_b4(ctx)
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001 - record and continue with the other methods
            errors.append({"instance": seed, "method": m, "error": repr(e), "tb": traceback.format_exc()})
        timing[m] = time.time() - tm
    if with_t2 and jpc_state is not None:
        tm = time.time()
        try:
            t2rows = run_type2(ctx, jpc_state, tmax, cache_dir)
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "method": "TYPE2", "error": repr(e), "tb": traceback.format_exc()})
        timing["TYPE2"] = time.time() - tm
    return {"seed": seed, "rows": rows, "t2rows": t2rows, "samples": samples, "errors": errors, "timing": timing,
            "wall_s": time.time() - t0, "truth": ctx.base.truth}


# ------------------------------------------------------------------ aggregation
def summarize(rows, t2rows, tmax_common, methods):
    out = {"methods": {}}
    by = {m: [r for r in rows if r["method"] == m] for m in methods}
    km_rows = []
    for m, rs in by.items():
        if not rs:
            continue
        steps = np.array([min(r["new_env_steps"], tmax_common) if not r["censored"] else min(r["new_env_steps"], tmax_common)
                          for r in rs], float)
        cens = np.array([r["censored"] or r["new_env_steps"] > tmax_common for r in rs])
        if m == "B4":
            cens = np.zeros(len(rs), bool)
        n_cert = int(sum(r["status"] == "CERTIFIED" for r in rs))
        n_false = int(sum(r["false_cert"] for r in rs))
        lo, hi = clopper_pearson(n_false, n_cert) if n_cert else (0.0, 1.0)
        regrets = [r["true_regret"] for r in rs if r["true_regret"] is not None]
        ut, sv = kaplan_meier(steps, cens)
        for t, s in zip(ut, sv):
            km_rows.append({"method": m, "steps": float(t), "surv": float(s)})
        out["methods"][m] = {
            "n": len(rs), "rmst_new_env_steps": rmst(steps, cens, tmax_common) if m != "B4" else 0.0,
            "completion_rate": float(np.mean(~cens)) if m != "B4" else 1.0,
            "median_steps_uncensored": float(np.median(steps[~cens])) if (~cens).any() else None,
            "mean_steps": float(steps.mean()), "n_certified": n_cert, "n_false_cert": n_false,
            "fcr": n_false / n_cert if n_cert else None, "fcr_cp_upper": hi,
            "true_regret_mean": float(np.mean(regrets)) if regrets else None,
            "true_regret_max": float(np.max(regrets)) if regrets else None,
            "frac_regret_gt_eps": float(np.mean([x > EPS for x in regrets])) if regrets else None,
            "zero_cost_cert_frac": float(np.mean([bool(r.get("zero_cost_cert")) for r in rs])),
            "wall_clock_s_total": float(sum(r.get("wall_clock_s", 0.0) or 0.0 for r in rs)),
            "status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in
                              ("CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE", "MODEL_CONFLICT", "COMPUTE_UNKNOWN")},
        }
    return out, km_rows


def paired_ratios(rows, tmax_common):
    """Per-problem JPC / baseline new-step ratios (censored baselines enter at their (lower-bound) step count)."""
    key = lambda r: (r["instance"], r["problem_index"])  # noqa: E731
    idx = {}
    for r in rows:
        idx.setdefault(key(r), {})[r["method"]] = r
    out = {"per_problem": []}
    for kk, d in sorted(idx.items()):
        if "JPC" not in d:
            continue
        j = d["JPC"]["new_env_steps"]
        rec = {"instance": kk[0], "problem_index": kk[1], "JPC": j, "JPC_censored": d["JPC"]["censored"]}
        for b in ("B1", "B2", "B3", "B3g", "ORACLE_DDA"):
            if b in d:
                rec[b] = d[b]["new_env_steps"]
                rec[b + "_censored"] = d[b]["censored"]
        if "B3" in d:
            rec["kappa_participant"] = d["B3"].get("kappa_participant")
        rec["rho_star"] = d["JPC"].get("rho_star")
        bases = [rec[b] for b in ("B1", "B2", "B3") if b in rec]
        if bases:
            rec["best_B123_per_problem"] = min(bases)
        out["per_problem"].append(rec)
    return out


def ratio_stats(pp, num="JPC", den="best_B123_per_problem", add=1.0):
    vals = [(r[num] + add) / (r[den] + add) for r in pp if den in r and r.get(den) is not None]
    if not vals:
        return None
    v = np.array(vals)
    return {"median": float(np.median(v)), "mean_log": float(np.mean(np.log(v))), "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75)), "n": len(v), "note": f"(steps+{add}) ratio to avoid 0/0"}


def cumcost(rows, methods, n0):
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["instance"]), []).append(r)
    for (m, inst), rs in by.items():
        rs = sorted(rs, key=lambda r: r["problem_index"])
        base = n0 if m in ("JPC", "ORACLE_DDA", "B3", "B3g", "B4") else 0
        c = base
        for r in rs:
            c += r["new_env_steps"]
            out.append({"method": m, "instance": inst, "k": r["problem_index"] + 1, "cum_steps": c,
                        "includes_n0": bool(base)})
    return out


def extended_analysis(rows, pp, tau_step, n0, methods):
    """Well-defined restatements of the pre-registered ratio (the literal per-problem median is tie-dominated)."""
    from dsswm.stats.bootstrap import cluster_bootstrap
    out = {}
    # RMST at the per-step-method cap (whole-trial methods are essentially all censored at this horizon)
    out["rmst_at_step_cap"] = {"tau": tau_step}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs or m == "B4":
            continue
        st = np.array([min(r["new_env_steps"], tau_step) for r in rs], float)
        ce = np.array([r["censored"] or r["new_env_steps"] > tau_step for r in rs])
        out["rmst_at_step_cap"][m] = {"rmst": rmst(st, ce, tau_step), "completion": float(np.mean(~ce))}
    nz = [r for r in pp if max(r["JPC"], r.get("best_B123_per_problem", 0)) > 0]
    both0 = sum(1 for r in pp if r["JPC"] == 0 and r.get("best_B123_per_problem", 1) == 0)
    v = np.array([r["JPC"] / max(r["best_B123_per_problem"], 1) for r in nz]) if nz else np.array([])
    out["ties_both_zero"] = int(both0)
    out["nontrivial_per_problem"] = {"n": len(nz), "median_ratio": float(np.median(v)) if len(v) else None,
                                     "frac_ratio_lt_0.8": float(np.mean(v < 0.8)) if len(v) else None,
                                     "frac_JPC_strictly_fewer": float(np.mean([r["JPC"] < r["best_B123_per_problem"]
                                                                               for r in nz])) if nz else None}
    # per-instance stream totals (the unit of analysis for a reuse stream)
    insts = sorted({r["instance"] for r in pp})
    tot = {}
    for b in ("B1", "B2", "B3", "B3g", "ORACLE_DDA"):
        per = {}
        for i in insts:
            ri = [r for r in pp if r["instance"] == i and b in r]
            if not ri:
                continue
            per[i] = np.array([[r["JPC"], r[b]] for r in ri], float)
        if not per:
            continue
        def stat(arrs, add_n0=0.0):
            a = np.concatenate(arrs)
            return float(np.log((a[:, 0].sum() + add_n0 * len(arrs)) / max(a[:, 1].sum() + add_n0 * len(arrs), 1.0)))
        add = n0 if b in ("B3", "B3g", "ORACLE_DDA") else 0.0
        est, lo, hi = cluster_bootstrap(per, lambda arrs: stat(arrs), B=2000, seed=1)
        inst_ratios = [float(per[i][:, 0].sum() / max(per[i][:, 1].sum(), 1.0)) for i in per]
        n_cens = sum(1 for r in rows if r["method"] == b and r["censored"])
        tot[b] = {"total_ratio_new_steps": float(math.exp(est)), "ci95": [float(math.exp(lo)), float(math.exp(hi))],
                  "total_ratio_incl_n0": float((sum(per[i][:, 0].sum() for i in per) + n0 * len(per)) /
                                               max(sum(per[i][:, 1].sum() for i in per) + add * len(per), 1.0)),
                  "median_instance_ratio": float(np.median(inst_ratios)),
                  "instances_JPC_fewer": int(sum(x < 1 for x in inst_ratios)), "n_instances": len(per),
                  "baseline_censored_problems": n_cens,
                  "note": "censored baseline problems enter at their cap (lower bound) -> ratio is conservative"}
    out["stream_totals_vs"] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=3000, help="per-problem cap for per-step methods")
    ap.add_argument("--tmax-trial", type=int, default=10000000, help="per-problem cap for whole-trial methods")
    ap.add_argument("--tmax-common", type=int, default=None, help="RMST horizon (default: calibrated from B1)")
    ap.add_argument("--methods", default="JPC,ORACLE_DDA,B3,B3g,B1,B2,B4")
    ap.add_argument("--no-t2", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(1)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    cache_dir = WS / "exp" / "cache" / "jtables"
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    (out_dir / "start_time.txt").write_text(start_iso)
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    n_inst = args.instances or (10 if args.mode == "pilot" else 48)
    n_prob = args.problems or (10 if args.mode == "pilot" else 15)
    seeds = list(range(n_inst)) if args.mode == "pilot" else list(range(10000, 10000 + n_inst))
    noises = [42] if args.mode == "pilot" else [42, 123, 456]
    methods = args.methods.split(",")
    log(f"start mode={args.mode} instances={n_inst} problems={n_prob} methods={methods} eps={EPS} delta={DELTA} "
        f"tmax={args.tmax} tmax_trial={args.tmax_trial} generator_hash={generator_hash()} (concurrent run, 4 workers)")
    progress(res_root, 0, n_inst * len(noises) + 2, "type2_jtables")
    if not args.no_t2:
        t2_jtables(seeds, cache_dir, log)
    t_run = time.time()
    jobs = [(s, nz) for nz in noises for s in seeds]
    results = []
    done = 0
    for res in Parallel(n_jobs=args.workers, return_as="generator_unordered", verbose=0)(
            delayed(run_instance)(s, nz, n_prob, args.tmax, args.tmax_trial, cache_dir, methods, not args.no_t2)
            for s, nz in jobs):
        results.append(res)
        done += 1
        rs = res["rows"]
        jj = [r for r in rs if r["method"] == "JPC"]
        log(f"instance {res['seed']} done in {res['wall_s']:.0f}s timing={ {k: round(v) for k, v in res['timing'].items()} } "
            f"JPC steps={[r['new_env_steps'] for r in jj]} errors={len(res['errors'])}")
        for e in res["errors"]:
            log(f"ERROR {e['method']} inst {e['instance']}: {e['error']}\n{e['tb']}")
        progress(res_root, done + 1, len(jobs) + 2, "instances", {"done": done, "of": len(jobs)})
    rows = [r for res in results for r in res["rows"]]
    t2rows = [r for res in results for r in res["t2rows"]]
    samples = [s for res in results for s in res["samples"]]
    errors = [e for res in results for e in res["errors"]]
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows + t2rows:
            f.write(json.dumps(r, default=float) + "\n")
    (out_dir / "samples" / "jpc_trajectories.json").write_text(json.dumps(samples[:8], indent=1, default=float))
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))

    # ---- T_max calibration (pre-registered rule: B1 dev completion >= 50%)
    b1 = sorted(r["new_env_steps"] for r in rows if r["method"] == "B1" and not r["censored"])
    n_b1 = sum(r["method"] == "B1" for r in rows)
    if args.tmax_common:
        tmax_common = args.tmax_common
        tmax_rule = "given"
    elif n_b1 and len(b1) >= math.ceil(0.5 * n_b1):
        tmax_common = int(b1[math.ceil(0.5 * n_b1) - 1])
        tmax_rule = "smallest T with B1 completion >= 50% on these dev instances"
    else:
        tmax_common = args.tmax_trial
        tmax_rule = "B1 completion < 50% even at tmax_trial; T_max = tmax_trial"
    summ, km_rows = summarize(rows, t2rows, tmax_common, methods)
    pp = paired_ratios(rows, tmax_common)
    with open(out_dir / "km.csv", "w") as f:
        f.write("method,steps,surv\n")
        for r in km_rows:
            f.write(f"{r['method']},{r['steps']},{r['surv']}\n")
    cc = cumcost(rows, methods, NL_DEFAULTS["n0"])
    with open(out_dir / "cumcost.csv", "w") as f:
        f.write("method,instance,k,cum_steps,includes_n0\n")
        for r in cc:
            f.write(f"{r['method']},{r['instance']},{r['k']},{r['cum_steps']},{r['includes_n0']}\n")
    # breakeven K*: smallest k with mean cumulative JPC cost (incl. n0) <= mean cumulative cost of the best baseline
    K = n_prob
    mean_cum = {}
    for m in methods:
        arr = [np.mean([r["cum_steps"] for r in cc if r["method"] == m and r["k"] == k]) for k in range(1, K + 1)
               if any(r["method"] == m and r["k"] == k for r in cc)]
        if arr:
            mean_cum[m] = [float(x) for x in arr]
    kstar = {}
    for b in ("B1", "B2", "B3"):
        if "JPC" in mean_cum and b in mean_cum:
            ks = [k + 1 for k in range(min(len(mean_cum["JPC"]), len(mean_cum[b])))
                  if mean_cum["JPC"][k] <= mean_cum[b][k]]
            kstar[b] = ks[0] if ks else None
    # gate
    r_best = ratio_stats(pp["per_problem"])
    rmst_b = {b: summ["methods"][b]["rmst_new_env_steps"] for b in ("B1", "B2", "B3") if b in summ["methods"]}
    best_b = min(rmst_b, key=rmst_b.get) if rmst_b else None
    r_vs_bestrmst = ratio_stats(pp["per_problem"], den=best_b) if best_b else None
    jpc = summ["methods"].get("JPC", {})
    t2_jpc = [r for r in t2rows if r["method"] == "JPC"]
    t2_false = int(sum(r["false_cert"] for r in t2_jpc))
    t2_label_match = float(np.mean([r["status"] == r["true_label"] or (r["true_label"] == "NEED_DATA" and
                                    r["status"] in ("CERTIFIED", "NEED_DATA")) or (r["true_label"] == "CERTIFIABLE_NOW"
                                    and r["status"] == "CERTIFIED") for r in t2_jpc])) if t2_jpc else None
    t2_honest = float(np.mean([r["status_at_start"] in ("NEED_DATA", "OUT_OF_SCOPE") for r in t2_jpc])) if t2_jpc else None
    ext = extended_analysis(rows, pp["per_problem"], args.tmax, NL_DEFAULTS["n0"], methods)
    # suspicious gate (> 5x savings, below oracle rho*, > 30% over a simple baseline) - flagged, then checked below
    susp = []
    if r_best and r_best["median"] < 0.2:
        susp.append(f"JPC/best-baseline median ratio {r_best['median']:.3f} (< 1/5: savings > 5x)")
    for b, d in ext["stream_totals_vs"].items():
        if b != "ORACLE_DDA" and d["total_ratio_new_steps"] < 0.2:
            susp.append(f"stream-total new steps JPC/{b} = {d['total_ratio_new_steps']:.3g} (savings > 5x)")
    rho_ratios = [r["JPC"] / r["rho_star"] for r in pp["per_problem"] if r.get("rho_star") and r["rho_star"] > 0
                  and math.isfinite(r["rho_star"]) and not r["JPC_censored"]]
    if rho_ratios and np.median(rho_ratios) < 1.0:
        susp.append(f"JPC steps below relaxed oracle rho* (median ratio {np.median(rho_ratios):.2f})")
    jpc_vs_oracle = ratio_stats(pp["per_problem"], den="ORACLE_DDA")
    pass_ratio = bool(r_best and r_best["median"] < 0.8)
    pass_comp = bool(jpc.get("completion_rate", 0) >= 0.8)
    pass_fc = bool(jpc.get("n_false_cert", 1) == 0)
    pass_t2 = bool(t2_false == 0) if t2_jpc else None
    from scipy.stats import spearmanr
    kp = [(r["kappa_participant"], (r["JPC"] + 1) / (r["B3"] + 1)) for r in pp["per_problem"]
          if r.get("kappa_participant") is not None and "B3" in r]
    h5c = None
    if len(kp) > 5:
        sp = spearmanr([a for a, _ in kp], [b for _, b in kp])
        h5c = {"spearman_kappaP_vs_ratio_JPC_B3": float(sp.statistic), "p": float(sp.pvalue), "n": len(kp),
               "median_ratio_JPC_B3": float(np.median([b for _, b in kp])),
               "median_kappaP": float(np.median([a for a, _ in kp]))}
    summary = {
        "task_id": TASK, "mode": args.mode, "env": ENV_NAME, "kappa_dyn": "high", "eta": 0, "eps": EPS,
        "delta": DELTA, "n_instances": n_inst, "noise_seeds": noises, "problems_per_instance": n_prob,
        "instance_seeds": seeds, "generator_hash": generator_hash(), "tmax_step_methods": args.tmax,
        "tmax_trial_methods": args.tmax_trial, "T_max_common": tmax_common, "T_max_rule": tmax_rule,
        "methods": summ["methods"], "ratio_JPC_vs_best_of_B1B2B3_per_problem": r_best,
        "best_baseline_by_rmst": best_b, "ratio_JPC_vs_best_baseline_by_rmst": r_vs_bestrmst,
        "ratio_JPC_vs_ORACLE_DDA": jpc_vs_oracle,
        "ratio_JPC_vs_B3": ratio_stats(pp["per_problem"], den="B3"),
        "ratio_JPC_vs_B3g": ratio_stats(pp["per_problem"], den="B3g"),
        "ratio_JPC_vs_B1": ratio_stats(pp["per_problem"], den="B1"),
        "ratio_JPC_vs_B2": ratio_stats(pp["per_problem"], den="B2"),
        "rho_star_ratio_median": float(np.median(rho_ratios)) if rho_ratios else None,
        "mean_cumulative_cost": mean_cum, "breakeven_K_star": kstar, "H5c": h5c,
        "type2": {"n": len(t2_jpc), "jpc_false_cert": t2_false, "status_counts": {
            s: int(sum(r["status"] == s for r in t2_jpc)) for s in ("CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE")},
            "true_label_counts": {s: int(sum(r["true_label"] == s for r in t2_jpc)) for s in
                                  ("CERTIFIABLE_NOW", "NEED_DATA", "OUT_OF_SCOPE")},
            "label_consistency": t2_label_match, "zero_cost_refusal_rate": t2_honest,
            "b4_false_confidence": float(np.mean([r["false_cert"] for r in t2rows if r["method"] == "B4"])) if t2rows else None},
        "extended": ext,
        "suspicious_flags": susp,
        "gate": {"median_ratio_lt_0.8": pass_ratio, "jpc_completion_ge_0.8": pass_comp, "jpc_false_cert_0": pass_fc,
                 "type2_false_cert_0": pass_t2},
        "errors": len(errors), "wall_clock_s": time.time() - t_run,
        "notes": [
            "Concurrent run (4 workers, other tasks on the same machine): wall-clock numbers are inflated.",
            "Certificate: minimax-regret policy over Theta_t (sound; can only certify earlier than argmax-theta_hat).",
            "Whole-trial baselines draw full trials from the platform's own vectorised sampler (same distribution as "
            "env.step), cost = H steps per trial; steering to s0 is free for them (favourable to B1/B2).",
            "B3 uses a fixed-n Wald radius and ignores linearisation error (favourable, not anytime-valid).",
            "rho* is a relaxed reference (free state choice), not a formal lower bound for the plug-in LR set.",
        ],
    }
    go = pass_ratio and pass_comp and pass_fc and (pass_t2 is not False)
    summary["go_no_go"] = "GO" if go else "NO_GO"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "paired.json").write_text(json.dumps(pp, indent=1, default=float))
    log(f"summary: {json.dumps({k: summary[k] for k in ('gate', 'go_no_go', 'T_max_common', 'suspicious_flags')})}")
    mark_done(res_root, "success", f"{summary['go_no_go']} median ratio={r_best['median'] if r_best else None}")
    return summary


if __name__ == "__main__":
    main()
