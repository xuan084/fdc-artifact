"""[NON-CANONICAL for R3: hr1_r3_* chunks use run_hr1_chunk.py so that all chunks share one code path.]
HR1 main contrast (family A), shared runner for hr1_r3_{a,b,c,d} and hr1_r0_{a,b} (methodology 3 / 4.3 / 6.1).

Levels (from the locked off-grid ladder, plan/prereg_lock.json):
  R0  truth snapped to G_1, grid G_1, no correction:      JPC, B3, B3g, B3-UI, B8, B12, B2
  R3  continuous truth, grid G_{f*} (f* = 2, locked),      JPC_infl, B3, B3g_infl, B3-UI, B8_infl, B12_infl, B2
      local inflation (certify.minimax_infl):            + JPC (uncorrected on G_2, diagnostic only, never a denominator)
Grid methods at R3 use the locked inflated certificate R_grid(Theta_t) + 2 max_{g alive} eta_loc(g) <= eps and refuse
honestly (data-free) if 2 min_g eta_loc(g) > eps. Continuous-class methods (B3, B3-UI) are never inflated.
Refusal / timeout = censored and charged T_max (stepwise 3000; whole-trial 2.31e6); consumed steps reported separately.
Stream protocol: n0 = 20 shared rounds, K = 15 problems per stream, every method re-uses its OWN ledger across the
problems of a stream and runs on its own fresh platform copy, CRN on (instance seed, stream noise seed).
JPC / JPC_infl MODEL_CONFLICT (empty set) -> B2 whole-trial fallback on that problem, fallback steps charged.
B3-UI: same Wald-opt design as B3 (Fisher at the grid MLE); the UI certificate (continuous Newton MLE, expensive) is
evaluated on a geometric schedule (every step for the first 10 steps, then whenever steps >= 1.10 x last check), i.e.
charged steps overshoot the first certifiable step by <= 10% (disadvantage to the baseline, reported).
Evidence-boundary assertions: after every problem env.n_steps == lr.n_rounds (== n0 + cumulative probe steps).
H0: the certifier used here is exact enumeration over the alive grid set (certify_minimax), so no dual / BnB upper
bound is computed in this task; H0 dual >= exact sup is checked by h0_certifier_soundness (recorded in the summary).
Pilot: dev seeds 660-663 x 5 problems, stream 0 (smoke + timing). Full: locked eval ranges x 3 streams x 15 problems;
assert_locked() at start-up; incremental results.jsonl + units_done.jsonl (resumable per (instance, stream, group)).

Usage: run_hr1_main.py --task hr1_r3_a --mode {pilot,full} [--workers 4] [--smoke] [--resummarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache"
LOCK = WS / "plan" / "prereg_lock.json"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = int(2.31e6)
UI_GROWTH = 1.10

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.baselines.b3_ui import certify_b3_ui  # noqa: E402
from dsswm.baselines.b12_tas import B12TaS  # noqa: E402
from dsswm.baselines.glm_linearised import (certify_linear, choose_design_action, fisher_rounds,  # noqa: E402
                                            value_and_grad, wald_beta)
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.eta_loc import build_plans, eta_loc, j_values  # noqa: E402
from dsswm.certify.minimax_enum import (certify_minimax, group_ids, observable_signature, regret_matrix,  # noqa: E402
                                        trichotomy)
from dsswm.certify.minimax_infl import certify_minimax_infl, inflation_floor  # noqa: E402
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.grid_ladder import cell_index_of, class_vectors, ladder_grid  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import NL_DEFAULTS, generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

C_KNOWN, RHO_RET, NMAX, N0 = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"], NL_DEFAULTS["n0"]

GROUPS = {
    "R3": [["JPC_infl", "B3g_infl", "B8_infl", "B12_infl"], ["B3"], ["B3-UI"], ["B2"], ["JPC"]],
    "R0": [["JPC"], ["B3"], ["B3g"], ["B3-UI"], ["B8"], ["B12"], ["B2"]],
}
DIAGNOSTIC = {"R3": {"JPC"}, "R0": set()}
DENOM_CANDS = ("B3", "B3g", "B3g_infl", "B3-UI", "B8", "B8_infl", "B12", "B12_infl")
WHOLE_TRIAL = ("B12", "B12_infl", "B2")


def fkey(f):
    return f"{f:g}"


def atomic_save(path: Path, arr):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)


def vstar_of(truth: dict) -> np.ndarray:
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


def U1(q):
    return type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})()


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
def precompute(seeds, n_prob, streams, fs, need_eta, dev, log):
    """J tables (+ eta_loc for inflated levels) per (problem, f), cached truth-free; c_q (G_1 max); harness J_true."""
    prop = NLPropagator(2, 2, NMAX, make_offgrid_instance(seeds[0], "R1").env.aspace, C_KNOWN, RHO_RET, device=dev)
    meta = {"cq": {}, "jtrue": {}, "eta_min_over_eps": {}, "sec": {}, "gap12": {}}
    probs, seen = [], set()
    for s in seeds:
        inst0 = make_offgrid_instance(s, "R0")
        inst1 = make_offgrid_instance(s, "R1")
        by_pid = {q.pid: q for q in inst1.problems}
        for st in streams:
            for q in make_offgrid_instance(s, "R1", stream=st).problems[:n_prob]:
                if q.pid not in seen:
                    seen.add(q.pid)
                    probs.append((s, by_pid[q.pid], vstar_of(inst0.truth), vstar_of(inst1.truth)))
    for f in fs:
        ncl = NLClass(ladder_grid(f), device=dev)
        params = None
        d = CACHE / f"jtables_f{fkey(f)}"
        d.mkdir(parents=True, exist_ok=True)
        t_j, t_e = [], []
        for (s, q, v0, v1) in probs:
            pj = d / f"{q.pid}_raw.npy"
            if not pj.exists():
                params = params or ncl.torch_params()
                t0 = time.perf_counter()
                raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, U1(q))
                atomic_save(pj, raw.astype(np.float32) if f == 4 else raw)
                t_j.append(time.perf_counter() - t0)
            pe = d / f"{q.pid}_etaloc_raw.npy"
            if f in need_eta and not pe.exists():
                t0 = time.perf_counter()
                eta, _, _, info = eta_loc(prop, ncl, q, return_parts=True)
                atomic_save(d / f"{q.pid}_LJ.npy", np.array([info["parts"]["L_J"]]))
                atomic_save(pe, eta)
                t_e.append(time.perf_counter() - t0)
        meta["sec"][fkey(f)] = {"jtable_new_mean": float(np.mean(t_j)) if t_j else None, "n_jtable_new": len(t_j),
                                "eta_new_mean": float(np.mean(t_e)) if t_e else None, "n_eta_new": len(t_e)}
        log(f"tables f={fkey(f)} |Theta|={ncl.B}: new J {len(t_j)} ({meta['sec'][fkey(f)]['jtable_new_mean']}), "
            f"new eta {len(t_e)} ({meta['sec'][fkey(f)]['eta_new_mean']})")
        del ncl, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t0 = time.perf_counter()
    for (s, q, v0, v1) in probs:
        c = 1.0 / float(np.load(CACHE / "jtables_f1" / f"{q.pid}_raw.npy").max())
        meta["cq"][q.pid] = c
        plans = build_plans(prop, q)
        jt = j_values(prop, plans, np.stack([v0, v1]), q.utility.w, q.utility.w_ret) * c
        meta["jtrue"][q.pid] = {"R0": jt[0], "R1": jt[1]}
        for lv, row in (("R0", jt[0]), ("R1", jt[1])):
            srt = np.sort(np.asarray(row))[::-1]
            meta["gap12"].setdefault(lv, {})[q.pid] = float(srt[0] - srt[1]) if len(srt) > 1 else float("inf")
        for f in need_eta:
            meta["eta_min_over_eps"].setdefault(fkey(f), []).append(
                float(np.load(CACHE / f"jtables_f{fkey(f)}" / f"{q.pid}_etaloc_raw.npy").min()) * c / EPS)
    meta["sec"]["jtrue_total"] = time.perf_counter() - t0
    meta["n_problems_tables"] = len(probs)
    del prop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
class Arm:
    """Learner-side objects for one (instance, stream, level, grid f) + harness truth (scoring only)."""

    def __init__(self, seed, stream, level, f, n_prob, meta, need_lr=True, infl=False):
        torch.set_num_threads(1)
        self.seed, self.stream, self.level, self.f = seed, stream, level, f
        self.truth_level = "R0" if level == "R0" else "R1"
        inst = make_offgrid_instance(seed, level, stream=stream)
        self.noise_seed = inst.noise_seed
        self.aspace = inst.env.aspace
        self.problems = inst.problems[:n_prob]
        self.ncl = NLClass(ladder_grid(f), device="cpu")
        self.V = class_vectors(self.ncl)
        self.star = int(cell_index_of(vstar_of(inst.truth)[None], self.ncl)[0])      # harness only
        self.prop = NLPropagator(2, 2, NMAX, self.aspace, C_KNOWN, RHO_RET, device="cpu")
        d = CACHE / f"jtables_f{fkey(f)}"
        self.J, self.Reg, self.eta, self.Jt, self.cq, self.gap12 = [], [], [], [], [], []
        for q in self.problems:
            c = meta["cq"][q.pid]
            J = np.load(d / f"{q.pid}_raw.npy").astype(float) * c
            self.J.append(J)
            self.Reg.append(regret_matrix(J))
            self.eta.append(np.load(d / f"{q.pid}_etaloc_raw.npy") * c if infl else np.zeros(J.shape[0]))
            self.Jt.append(np.asarray(meta["jtrue"][q.pid][self.truth_level]))
            self.cq.append(c)
            self.gap12.append(meta["gap12"][self.truth_level][q.pid])
        if need_lr:
            self.LT = self.prop.tables(self.ncl.torch_params())[0]
            self.LT_np = self.LT.numpy()
            self.py, self.pe = class_prob_tables(self.ncl.np_params, C_KNOWN, NMAX, self.aspace.nb)
            self.inc = IncidenceIndex(self.prop.codec, self.aspace, NMAX)
            self.legal = np.arange(self.aspace.n)
            self.gid = group_ids(observable_signature(self.py, self.pe, max_level=1))
            self.all_single = len(np.unique(self.gid)) == self.ncl.B

    def fresh(self):
        return make_offgrid_instance(self.seed, self.level, stream=self.stream)

    def u_max(self, k):
        q = self.problems[k]
        return self.cq[k] * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)

    def row(self, k, method, status, pi, steps, extra=None, charged=None):
        Jt = self.Jt[k]
        tr = float(Jt.max() - Jt[pi]) if pi is not None else None
        cert = status == "CERTIFIED"
        tmax = TMAX_TRIAL if method in WHOLE_TRIAL else TMAX_STEP
        if charged is None:
            charged = steps if cert else max(steps, tmax)
        r = {"kind": "hr1", "instance": self.seed, "stream": self.stream, "noise_seed": self.noise_seed,
             "level": self.level, "f": self.f, "problem": self.problems[k].pid, "k": k, "method": method,
             "diagnostic_only": method in DIAGNOSTIC[self.level], "status": status, "new_env_steps": int(charged),
             "steps_consumed": int(steps), "censored": not cert, "certified_policy": pi, "true_regret": tr,
             "false_cert": bool(cert and tr is not None and tr > EPS), "zero_cost": bool(cert and steps == 0),
             "true_top2_gap": self.gap12[k], "tie": bool(self.gap12[k] < EPS), "H": self.problems[k].H,
             "n_policies": len(self.problems[k].policies), "eps": EPS, "delta": DELTA,
             "eta_loc_min_over_eps": float(self.eta[k].min()) / EPS,
             "eta_loc_star_over_eps": float(self.eta[k][self.star]) / EPS}
        if extra:
            r.update(extra)
        return r


def floor_any(eta, eps=EPS):
    """Data-free: no set (not even a singleton) can pass the inflated condition."""
    return bool(2.0 * float(np.min(eta)) > eps)


def b2_sampler_factory(arm, env, k, history, keys):
    q, c = arm.problems[k], arm.cq[k]
    counters = {}

    def sampler(a, n):
        cc = counters.get(a, 0)
        counters[a] = cc + 1
        r = np.random.default_rng([arm.seed, arm.noise_seed, k, a, cc, 55])
        ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
        st = sampler.raw.setdefault(a, ([], []))
        st[0].append(np.asarray(ys)); st[1].append(np.asarray(es))
        return c * (ys @ q.utility.w + q.utility.w_ret * es)
    sampler.raw = {}
    return sampler


def run_b2_problem(arm, env, k, history, proxy):
    """B2: residual + empirical-Bernstein whole-trial LUCB, re-using this method's stored trajectories (same s0, H)."""
    q, c = arm.problems[k], arm.cq[k]
    keys = [(p.name, tuple(int(x) for x in q.loads0), tuple(int(x) for x in q.engaged0), q.H) for p in q.policies]
    prior = {}
    for a, key in enumerate(keys):
        if key in history:
            ys, es = history[key]
            prior[a] = list(c * (ys.astype(float) @ q.utility.w + q.utility.w_ret * es.astype(float)))
    sampler = b2_sampler_factory(arm, env, k, history, keys)
    res = lucb(sampler, len(q.policies), q.H, EPS, DELTA, arm.u_max(k), TMAX_TRIAL, kind="eb", proxy=proxy,
               prior=prior)
    for a, key in enumerate(keys):
        if a in sampler.raw:
            ys = np.concatenate(sampler.raw[a][0]); es = np.concatenate(sampler.raw[a][1])
            n_used = int(res["n_per_arm"][a]) - len(prior.get(a, []))
            ys, es = ys[:n_used], es[:n_used]
            if key in history:
                ys = np.concatenate([history[key][0], ys]); es = np.concatenate([history[key][1], es])
            history[key] = (ys, es)
    return res


def run_jpc(arm: Arm, infl: bool, want_samples=False):
    method = "JPC_infl" if infl else "JPC"
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)
    led = EvidenceLedger([lr])
    all_obs = list(inst.init_obs)
    for o in inst.init_obs:
        led.record(o, "initial")
    rng = np.random.default_rng([arm.seed, arm.noise_seed, 101])
    b2_hist = {}
    rows, samples = [], []
    for k, q in enumerate(arm.problems):
        Reg, eta = arm.Reg[k], arm.eta[k]
        t0 = time.perf_counter()
        steps, status, traj, refused = 0, None, [], False
        while True:
            if infl and floor_any(eta):
                status, refused = "NEED_DATA", True             # honest refusal: no data can ever certify
                break
            mask = lr.mask().numpy()
            if infl:
                cert = certify_minimax_infl(Reg, mask, eta, EPS, TOP_M)
                if cert["status"].value == "NEED_DATA" and inflation_floor(eta, mask, EPS):
                    status, refused = "NEED_DATA", True
                    break
            else:
                cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
            if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                status = sv
                break
            if not arm.all_single and steps % 10 == 0:
                amb, _, _ = trichotomy(Reg, mask, arm.gid, EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            blk = cert["blocking"] or [int(i) for i in np.flatnonzero(mask)[np.argsort(-eta[mask])[:TOP_M]]]
            margins = LOG_THR - lr.log_ratio().numpy()[blk]
            code = arm.prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, arm.py[kh:kh + 1], arm.pe[kh:kh + 1], arm.LT_np[kh], arm.py[blk], arm.pe[blk],
                                 margins, arm.inc, arm.prop, arm.legal, rng)
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            all_obs.append(obs)
            steps += 1
            if want_samples and len(traj) < 40:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "star_alive": bool(mask[arm.star]), "mode": info["mode"]})
        assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
        mask = lr.mask().numpy()
        gc = certify_minimax(Reg, mask, EPS, TOP_M)
        pi = gc["pi"] if status == "CERTIFIED" else None
        extra = {"set_size": int(mask.sum()), "r_bar_grid": float(gc["r_bar"]),
                 "theta_cell_alive": bool(mask[arm.star]),
                 "eta_loc_alive_max_over_eps": float(eta[mask].max()) / EPS if mask.any() else None,
                 "refused_floor": refused, "uncertified_recommendation": None if status == "CERTIFIED" else gc["pi"],
                 "lr_n_rounds": int(lr.n_rounds), "env_n_steps": int(env.n_steps), "fallback": None}
        charged = None
        if status == "MODEL_CONFLICT":                # methodology: empty set -> B2 whole-trial fallback, cost charged
            res = run_b2_problem(arm, env, k, b2_hist, None)
            extra["fallback"] = {"method": "B2", "status": res["status"], "steps": int(res["steps"])}
            if res["status"] == "CERTIFIED":
                status, pi = "CERTIFIED", int(res["pi"])
                charged = steps + int(res["steps"])
            else:
                charged = max(steps + int(res["steps"]), TMAX_TRIAL)
        extra["wall_clock_s"] = time.perf_counter() - t0
        rows.append(arm.row(k, method, status, pi, steps, extra, charged=charged))
        if want_samples and k < 2:
            samples.append({**rows[-1], "J_true": arm.Jt[k].round(4).tolist(), "trajectory_head": traj})
    return rows, samples


class _B3Design:
    """Shared Wald-opt GLM design state of B3 / B3g / B3-UI (Fisher at the grid MLE, re-used across problems)."""

    def __init__(self, arm, inst):
        self.arm = arm
        self.hist = {}
        for o in inst.init_obs:
            key = (arm.prop.codec.encode(o.loads, o.engaged), o.action)
            self.hist[key] = self.hist.get(key, 0) + 1
        self.d = arm.V.shape[1]
        self.beta = wald_beta(self.d, DELTA)
        self.cache = {}                                  # (grid MLE index, state code) -> Fisher of every legal action

    def F_all(self, kh, code):
        key = (int(kh), int(code))
        F = self.cache.get(key)
        if F is None:
            if len(self.cache) > 20000:
                self.cache.clear()
            F = fisher_rounds(self.arm.V[kh], self.arm.prop.codec, self.arm.aspace, code, self.arm.legal, C_KNOWN, 2, 2)
            self.cache[key] = F
        return F

    def build_V(self, kh):
        Vm = np.eye(self.d)
        for (code, a), n in self.hist.items():
            Vm += n * self.F_all(kh, code)[a]
        return Vm


def run_b3(arm: Arm, variant: str):
    """variant in {'B3', 'B3g', 'B3g_infl', 'B3-UI'}."""
    grid_mode = variant.startswith("B3g")
    infl = variant == "B3g_infl"
    ui = variant == "B3-UI"
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)          # grid MLE theta_hat (linearisation point) + UI numerator
    led = EvidenceLedger([lr])
    all_obs = list(inst.init_obs)
    for o in inst.init_obs:
        led.record(o, "initial")
    des = _B3Design(arm, inst)
    beta = des.beta
    rng = np.random.default_rng([arm.seed, arm.noise_seed, {"B3": 303, "B3g": 304, "B3g_infl": 304, "B3-UI": 305}[variant]])
    rows = []
    v_hat_prev = None
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        eta = arm.eta[k]
        plans = [arm.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
        U = type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": arm.cq[k]})()
        grad_cache, steps, kh_prev, Vm, status, refused = {}, 0, None, None, None, False
        gmask, pi, next_ui, n_ui_calls, ui_sec, ui_beta = None, None, 0, 0, 0.0, None
        while True:
            if infl and floor_any(eta):
                status, refused = "NEED_DATA", True
                break
            kh = lr.mle()
            v = arm.V[kh]
            if kh != kh_prev:
                Vm = des.build_V(kh)
                kh_prev = kh
            if kh not in grad_cache:
                grad_cache[kh] = value_and_grad(arm.prop, plans, v, U, 2, 2)
            Jlin, G = grad_cache[kh]
            Vinv = np.linalg.inv(Vm)
            lc = certify_linear(arm.J[k][kh], G, Vinv, beta, EPS)
            done = False
            if grid_mode:
                diff = arm.V - v[None]
                gmask = np.einsum("bd,de,be->b", diff, Vm, diff) <= beta ** 2
                cert = (certify_minimax_infl(arm.Reg[k], gmask, eta, EPS, TOP_M) if infl
                        else certify_minimax(arm.Reg[k], gmask, EPS, TOP_M))
                if infl and cert["status"].value == "NEED_DATA" and inflation_floor(eta, gmask, EPS):
                    status, refused = "NEED_DATA", True
                    break
                done = cert["status"].value == "CERTIFIED"
                pi = cert["pi"]
            elif ui:
                if steps >= next_ui or steps >= TMAX_STEP:
                    tu = time.perf_counter()
                    c = certify_b3_ui(arm.prop, plans, all_obs, lr.log_num,
                                      v_hat_prev if v_hat_prev is not None else v, arm.ncl.grid, U, DELTA, EPS)
                    ui_sec += time.perf_counter() - tu
                    n_ui_calls += 1
                    v_hat_prev = c["v_hat"]
                    ui_beta = c["beta"]
                    next_ui = steps + 1 if steps < 10 else max(steps + 1, int(math.ceil(steps * UI_GROWTH)))
                    if c["status"].value == "MODEL_CONFLICT":
                        status = "MODEL_CONFLICT"
                        break
                    done = c["status"].value == "CERTIFIED"
                    pi = c["pi"]
            else:
                done = lc["certified"]
                pi = lc["pi"]
            if done:
                status = "CERTIFIED"
                break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            code = arm.prop.codec.encode(*h.observable_state())
            Fa = des.F_all(kh, code)
            if grid_mode:
                blk = cert["blocking"]
                if rng.random() < 0.05 or not blk:
                    a = int(rng.choice(arm.legal))
                else:
                    Db = arm.V[blk] - v[None]
                    marg = np.maximum(beta ** 2 - np.einsum("bd,de,be->b", Db, Vm, Db), 1e-9)
                    gain = np.einsum("bd,ade,be->ab", Db, Fa, Db) / marg[None]
                    a = int(arm.legal[int(np.argmax(gain.min(1)))])
            else:
                a = choose_design_action(Fa, Vm, lc["D"][lc["ub"] > EPS], arm.legal, rng)
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            all_obs.append(obs)
            des.hist[(code, a)] = des.hist.get((code, a), 0) + 1
            Vm = Vm + Fa[a]
            steps += 1
        assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
        extra = {"refused_floor": refused, "wald_beta": beta, "wall_clock_s": time.perf_counter() - t0,
                 "lr_n_rounds": int(lr.n_rounds), "env_n_steps": int(env.n_steps)}
        if grid_mode and gmask is not None:
            extra.update({"set_size": int(gmask.sum()), "theta_cell_alive": bool(gmask[arm.star])})
        if ui:
            extra.update({"ui_calls": n_ui_calls, "ui_sec": ui_sec, "beta_ui": ui_beta,
                          "ui_schedule_growth": UI_GROWTH})
        rows.append(arm.row(k, variant, status, pi if status == "CERTIFIED" else None, steps, extra))
    return rows


def run_b8(arm: Arm, infl: bool, tau=EPS / 2):
    """B8: full identification on the (inflated) set -- width_J(Theta_t^+) < tau -- then plan with theta_hat."""
    method = "B8_infl" if infl else "B8"
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)
    led = EvidenceLedger([lr])
    all_obs = list(inst.init_obs)
    for o in inst.init_obs:
        led.record(o, "initial")
    rng = np.random.default_rng([arm.seed, arm.noise_seed, 808, int(round(tau * 1e4))])
    rows = []
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        J, eta = arm.J[k], arm.eta[k]
        steps, status, width, refused = 0, None, None, False
        while True:
            if infl and 2.0 * float(eta.min()) >= tau:      # data-free: even a singleton is wider than tau
                status, refused = "NEED_DATA", True
                break
            mask = lr.mask().numpy()
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                status = "MODEL_CONFLICT"
                break
            Jm = J[idx]
            width = float((Jm.max(0) - Jm.min(0)).max()) + (2.0 * float(eta[idx].max()) if infl else 0.0)
            if width < tau:
                cert = (certify_minimax_infl(arm.Reg[k], mask, eta, EPS, TOP_M) if infl
                        else certify_minimax(arm.Reg[k], mask, EPS, TOP_M))
                status = "CERTIFIED" if cert["status"].value == "CERTIFIED" else "IDENTIFIED_UNCERTIFIED"
                break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            dist = np.abs(Jm - J[kh][None]).max(1)
            order = np.argsort(-dist)
            blk = [int(idx[o]) for o in order[:TOP_M] if dist[o] > tau / 2] or [int(idx[o]) for o in order[:TOP_M]]
            margins = LOG_THR - lr.log_ratio().numpy()[blk]
            code = arm.prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, arm.py[kh:kh + 1], arm.pe[kh:kh + 1], arm.LT_np[kh], arm.py[blk], arm.pe[blk],
                                 margins, arm.inc, arm.prop, arm.legal, rng)
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            all_obs.append(obs)
            steps += 1
        assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
        mask = lr.mask().numpy()
        pi = int(np.argmax(J[lr.mle()]))
        extra = {"tau": tau, "id_width_end": width, "refused_floor": refused, "set_size": int(mask.sum()),
                 "theta_cell_alive": bool(mask[arm.star]), "wall_clock_s": time.perf_counter() - t0,
                 "lr_n_rounds": int(lr.n_rounds), "env_n_steps": int(env.n_steps)}
        rows.append(arm.row(k, method, status, pi if status == "CERTIFIED" else None, steps, extra))
    return rows


def run_b12(arm: Arm, infl: bool):
    method = "B12_infl" if infl else "B12"
    inst = arm.fresh()
    env = inst.env                                     # harness: the trial sampler IS the platform
    rows = []
    prior = None
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        c = arm.cq[k]
        eta = arm.eta[k] if infl else None
        sim_rng = np.random.default_rng([arm.seed, arm.noise_seed, 77, k])

        def sampler(kk, n, q=q, c=c, sim_rng=sim_rng):
            ys, es = env.simulate_batch(q.policies[kk], q.loads0, q.engaged0, q.H, n, sim_rng)
            return c * (ys @ q.utility.w + q.utility.w_ret * es)

        b = B12TaS(arm.J[k], u_max=arm.u_max(k), delta=DELTA, eps=EPS, eta=eta, prior_logM=prior)
        if infl and floor_any(eta):
            mask = b.mask()
            r = {"status": "NEED_DATA", "pi": None, "steps": 0, "refused": True, "pulls": [0] * len(q.policies)}
        else:
            r = b.run(sampler, q.H, max_steps=TMAX_TRIAL)
            r["refused"] = False
            mask = b.mask()
        prior = b.logM()
        st = r["status"]
        rows.append(arm.row(k, method, st, r["pi"] if st == "CERTIFIED" else None, r["steps"],
                            {"refused_floor": r["refused"], "set_size": int(mask.sum()),
                             "theta_cell_alive": bool(mask[arm.star]), "trials": int(np.sum(r["pulls"])),
                             "wall_clock_s": time.perf_counter() - t0}))
    return rows


def run_b2(arm: Arm):
    inst = arm.fresh()
    env = inst.env
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)
    for o in inst.init_obs:
        lr.update(o)
    kh0 = lr.mle()                                    # proxy (control variate) only; validity does not depend on it
    history, rows = {}, []
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        res = run_b2_problem(arm, env, k, history, arm.J[k][kh0])
        st = res["status"]
        rows.append(arm.row(k, "B2", st, res["pi"] if st == "CERTIFIED" else None, res["steps"],
                            {"trials": int(res["pulls"]), "reused_trials": int(res["n_reused"]),
                             "final_gap": res.get("gap"), "wall_clock_s": time.perf_counter() - t0}))
    return rows


def run_unit(seed, stream, level, f, methods, n_prob, meta, want_samples):
    torch.set_num_threads(1)
    t0 = time.time()
    need_lr = any(m not in ("B12", "B12_infl") for m in methods)
    infl = level != "R0"
    arm = Arm(seed, stream, level, f, n_prob, meta, need_lr=need_lr, infl=infl)
    build = time.time() - t0
    rows, samples, errors, msec = [], [], [], {}
    for m in methods:
        tm = time.time()
        try:
            if m in ("JPC", "JPC_infl"):
                r, s = run_jpc(arm, m == "JPC_infl", want_samples)
                samples += s
            elif m in ("B3", "B3g", "B3g_infl", "B3-UI"):
                r = run_b3(arm, m)
            elif m in ("B8", "B8_infl"):
                r = run_b8(arm, m == "B8_infl")
            elif m in ("B12", "B12_infl"):
                r = run_b12(arm, m == "B12_infl")
            elif m == "B2":
                r = run_b2(arm)
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "stream": stream, "level": level, "f": f, "method": m, "error": repr(e),
                           "tb": traceback.format_exc()})
        msec[m] = time.time() - tm
    return rows, samples, errors, {"instance": seed, "stream": stream, "methods": methods, "build_sec": build,
                                   "method_sec": msec, "sec": time.time() - t0}


# ------------------------------------------------------------------------------------ summaries
def summarize(rows, n0=N0):
    out = {}
    for m in sorted({r["method"] for r in rows}):
        R = [r for r in rows if r["method"] == m]
        cert = [r for r in R if r["status"] == "CERTIFIED"]
        nfc = sum(r["false_cert"] for r in R)
        lo, hi = clopper_pearson(nfc, len(R), 0.05)
        alive = [r["theta_cell_alive"] for r in R if r.get("theta_cell_alive") is not None]
        streams = sorted({(r["instance"], r["stream"]) for r in R})
        tot = [sum(r["new_env_steps"] for r in R if (r["instance"], r["stream"]) == u) for u in streams]
        ks = sorted({r["k"] for r in R})
        out[m] = {
            "n": len(R), "n_streams": len(streams), "completion": len(cert) / len(R),
            "diagnostic_only": bool(R[0].get("diagnostic_only")),
            "status_counts": {s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R})},
            "refused_floor": int(sum(bool(r.get("refused_floor")) for r in R)),
            "stream_steps_charged_mean": float(np.mean(tot)),
            "cum_cost_with_n0_mean": float(np.mean(tot)) + n0,
            "steps_consumed_median": float(np.median([r["steps_consumed"] for r in R])),
            "steps_consumed_stream_mean": float(np.mean([sum(r["steps_consumed"] for r in R
                                                             if (r["instance"], r["stream"]) == u) for u in streams])),
            "false_certs": int(nfc), "fcr": nfc / len(R), "fcr_cp": [lo, hi], "fcr_cp_upper": hi,
            "fcr_among_certified": nfc / max(len(cert), 1),
            "max_true_regret_certified": max([r["true_regret"] for r in cert], default=None),
            "tie_rate": float(np.mean([r["tie"] for r in R])),
            "tie_rate_among_certified": float(np.mean([r["tie"] for r in cert])) if cert else None,
            "theta_cell_survival": float(np.mean(alive)) if alive else None,
            "zero_cost_rate": float(np.mean([r["zero_cost"] for r in R])),
            "zero_cost_rate_by_k": {int(k): float(np.mean([r["zero_cost"] for r in R if r["k"] == k])) for k in ks},
            "eta_loc_min_over_eps_median": float(np.median([r["eta_loc_min_over_eps"] for r in R])),
            "fallbacks_B2": int(sum(1 for r in R if r.get("fallback"))),
            "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in R]))}
    return out


def stream_ratios(rows, summ, numerator):
    """Chunk-local, PRELIMINARY: instance-cluster log ratio of stream totals (charged, +1). The binding denominator
    is chosen by r2_analysis_aggregate after merging all chunks."""
    cands = [m for m in summ if m in DENOM_CANDS]
    valid = [m for m in cands if summ[m]["fcr_cp_upper"] <= 2 * DELTA]
    best = min(valid, key=lambda m: summ[m]["stream_steps_charged_mean"]) if valid else None
    zfc = [m for m in cands if summ[m]["false_certs"] == 0]
    best_diag = min(zfc, key=lambda m: summ[m]["stream_steps_charged_mean"]) if zfc else None
    insts = sorted({r["instance"] for r in rows})
    rec = {"numerator": numerator, "valid_denominators": valid, "best_valid": best,
           "best_zero_false_cert_diag": best_diag, "excluded_fcr": [m for m in cands if m not in valid],
           "note": "chunk-local preliminary; binding denominator computed after merging chunks"}
    if numerator not in summ:
        return rec

    def inst_log(m, i):
        R = [r for r in rows if r["method"] == m and r["instance"] == i]
        sts = sorted({r["stream"] for r in R})
        return float(np.mean([math.log(sum(r["new_env_steps"] for r in R if r["stream"] == s) + 1) for s in sts]))

    for lab, den in [("vs_B3", "B3"), ("vs_B3-UI", "B3-UI"), ("vs_best_valid", best),
                     ("vs_best_zero_fc_diag", best_diag)]:
        if den is None or den not in summ:
            rec[lab] = None
            continue
        lr_ = np.array([inst_log(numerator, i) - inst_log(den, i) for i in insts])
        rng = np.random.default_rng(42)
        bs = [lr_[rng.integers(len(lr_), size=len(lr_))].mean() for _ in range(10000)]
        rec[lab] = {"denominator": den, "ratio_geo_mean": float(math.exp(lr_.mean())),
                    "ci95": [float(math.exp(np.quantile(bs, .025))), float(math.exp(np.quantile(bs, .975)))],
                    "per_instance": [float(math.exp(x)) for x in lr_]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true")
    args = ap.parse_args()
    TASK = args.task
    pilot = args.mode == "pilot"
    level = "R3" if TASK.startswith("hr1_r3") else ("R0" if TASK.startswith("hr1_r0") else None)
    assert level, f"unsupported task {TASK}"
    lock = json.loads(LOCK.read_text())
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    f = int(lock.get("f_star", 2)) if level == "R3" else 1
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    pid_path = RES_ROOT / f"{TASK}.pid"
    prog_path = RES_ROOT / f"{TASK}_PROGRESS.json"
    real = not (args.smoke or args.resummarize)
    if real:
        pid_path.write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def progress(step, total, phase, metric=None):
        if not real:
            return
        prog_path.write_text(json.dumps({
            "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
            "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))

    def mark_done(status, summary_txt):
        if pid_path.exists():
            pid_path.unlink()
        fp = {}
        if prog_path.exists():
            try:
                fp = json.loads(prog_path.read_text())
            except ValueError:
                pass
        (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary_txt,
                                                           "final_progress": fp,
                                                           "timestamp": datetime.now().isoformat()}))

    if pilot:
        seeds, n_prob, streams = list(range(660, 664)), 5, [0]
        assert max(seeds) < 10000, "dev seeds only in pilot"
    else:
        rngs = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds = [s for a, b in rngs for s in range(a, b + 1)]
        n_prob, streams = int(lock["eval_manifest"]["K_problems"]), list(range(len(STREAMS)))
    if args.smoke:
        seeds, n_prob = seeds[:1], 2
    groups = GROUPS[level]
    numerator = "JPC_infl" if level == "R3" else "JPC"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    log(f"start {TASK} mode={args.mode} level={level} f={f} device={dev} seeds={seeds[0]}-{seeds[-1]} "
        f"x streams {streams} x {n_prob} problems; groups={groups}; eps={EPS} delta={DELTA} T_max step={TMAX_STEP} "
        f"trial={TMAX_TRIAL}; generator_hash={generator_hash()} lock status={lock.get('status')}")
    summary = {"task_id": TASK, "mode": args.mode, "level": level, "f": f, "seed": 42, "eps": EPS, "delta": DELTA,
               "instances": [seeds[0], seeds[-1]], "n_instances": len(seeds), "n_problems_per_stream": n_prob,
               "streams": streams, "eval_seeds_touched": not pilot, "generator_hash": generator_hash(),
               "lock_status": lock.get("status"), "lock_sha256": lock.get("sha256") or lock.get("sha256_provisional"),
               "concurrent_run": True, "tmax_stepwise": TMAX_STEP, "tmax_whole_trial": TMAX_TRIAL,
               "groups": groups, "numerator": numerator, "b3_ui_schedule_growth": UI_GROWTH,
               "h0_note": "certifier = exact enumeration on the alive grid set; no dual/BnB bound used in this task "
                          "(H0 dual>=exact checked in h0_certifier_soundness)"}
    res_path = out_dir / "results.jsonl"
    done_path = out_dir / "units_done.jsonl"
    try:
        if not args.resummarize:
            progress(1, 4, "gpu: J tables + eta_loc")
            fs = (1, f) if f != 1 else (1,)
            meta = precompute(seeds, n_prob, streams, fs, (f,) if level == "R3" else (), dev, log)
            summary["table_sec"] = meta["sec"]
            summary["n_problems_tables"] = meta["n_problems_tables"]
            if meta["eta_min_over_eps"]:
                summary["eta_loc_min_over_eps"] = {k: {"median": float(np.median(v)), "min": float(np.min(v))}
                                                   for k, v in meta["eta_min_over_eps"].items()}
                log(f"min_g eta_loc/eps per problem (floor check, need <= 0.5): {summary['eta_loc_min_over_eps']}")
            done_units = set()
            if done_path.exists():
                for x in open(done_path):
                    u = json.loads(x)
                    done_units.add((u["instance"], u["stream"], tuple(u["methods"])))
            jobs = [(s, st, tuple(g)) for s in seeds for st in streams for g in groups
                    if (s, st, tuple(g)) not in done_units]
            log(f"units: {len(jobs)} to run, {len(done_units)} already done (resume)")
            progress(2, 4, "cpu: streams", {"units_total": len(jobs), "units_done": 0})
            from joblib import Parallel, delayed
            n_err, unit_sec = 0, []
            gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
                delayed(run_unit)(s, st, level, f, list(g), n_prob, meta, s == seeds[0] and st == 0)
                for (s, st, g) in jobs)
            all_samples = []
            for i, (rr, ss, ee, us) in enumerate(gen):
                with open(res_path, "a") as fh:
                    for r in rr:
                        fh.write(json.dumps(r, default=float) + "\n")
                for e in ee:
                    log(f"ERROR {e['instance']} s{e['stream']} {e['method']}: {e['error']}\n{e['tb']}")
                n_err += len(ee)
                with open(out_dir / "errors.jsonl", "a") as fh:
                    for e in ee:
                        fh.write(json.dumps(e) + "\n")
                if not ee:
                    with open(done_path, "a") as fh:
                        fh.write(json.dumps({"instance": us["instance"], "stream": us["stream"],
                                             "methods": us["methods"]}) + "\n")
                with open(out_dir / "unit_sec.jsonl", "a") as fh:
                    fh.write(json.dumps(us) + "\n")
                unit_sec.append(us)
                all_samples += ss
                progress(2, 4, "cpu: streams", {"units_total": len(jobs), "units_done": i + 1, "errors": n_err})
            if all_samples:
                (samples_dir / "samples.json").write_text(json.dumps(all_samples, indent=1, default=float))
            summary["crashes_this_run"] = n_err
            log(f"streams done: {len(jobs)} units, {n_err} errors, {time.time() - T0:.0f}s")
        else:
            old = json.loads((out_dir / "summary.json").read_text())
            summary = {**old, **{k: v for k, v in summary.items() if k not in old}}
        progress(3, 4, "summary")
        rows = [json.loads(x) for x in open(res_path)]
        # de-duplicate (resume may re-run a unit that crashed half-way): keep the last row per key
        dedup = {}
        for r in rows:
            dedup[(r["instance"], r["stream"], r["method"], r["k"])] = r
        rows = list(dedup.values())
        errs = [json.loads(x) for x in open(out_dir / "errors.jsonl")] if (out_dir / "errors.jsonl").exists() else []
        units_all = [json.loads(x) for x in open(out_dir / "unit_sec.jsonl")] if (out_dir / "unit_sec.jsonl").exists() else []
        summ = summarize(rows)
        ratios = stream_ratios(rows, summ, numerator)
        summary["by_method"] = summ
        summary["stream_ratios_prelim"] = ratios
        summary["n_rows"] = len(rows)
        summary["crashes_total_logged"] = len(errs)
        expected = len(seeds) * len(streams) * n_prob * sum(len(g) for g in groups)
        summary["rows_expected"] = expected
        summary["complete"] = len(rows) == expected
        acc_ok = all(r.get("lr_n_rounds") is None or r["lr_n_rounds"] == r["env_n_steps"] for r in rows)
        summary["evidence_boundary_assertions_ok"] = acc_ok and not any("step accounting" in e["error"] for e in errs)
        # timing projection (wall): per-method cpu sec per problem + per-unit build, over 4 workers; + GPU tables
        build = float(np.mean([u["build_sec"] for u in units_all])) if units_all else 0.0
        cpu_pp = {m: v["wall_clock_per_problem_s"] for m, v in summ.items()}
        n_inst_full = 32 if level == "R3" else 64
        n_units_full = n_inst_full * 3 * len(groups)
        full_cpu = sum(cpu_pp.values()) * n_inst_full * 3 * 15 + build * n_units_full
        ts = meta["sec"] if not args.resummarize else summary.get("table_sec", {})
        per_prob_gpu = sum((ts.get(fkey(ff), {}) or {}).get("jtable_new_mean") or 0.0 for ff in ((1, f) if f != 1 else (1,)))
        per_prob_gpu += (ts.get(fkey(f), {}) or {}).get("eta_new_mean") or 0.0 if level == "R3" else 0.0
        gpu_s = per_prob_gpu * n_inst_full * 15 + (ts.get("jtrue_total", 0.0) / max(summary.get("n_problems_tables", 1), 1)) * n_inst_full * 15
        summary["timing_projection"] = {
            "cpu_sec_per_problem_by_method": cpu_pp, "unit_build_sec_mean": build,
            "full_cpu_sec_total": full_cpu, "full_cpu_wall_min_at_4_workers": full_cpu / 4 / 60,
            "full_gpu_tables_min": gpu_s / 60, "full_wall_min_projected": full_cpu / 4 / 60 + gpu_s / 60,
            "basis": f"{n_inst_full} instances x 3 streams x 15 problems; pilot per-problem wall-clock means",
            "note": "concurrent run (up to 4 tasks on 20 cores); per-problem cost of B3 / B3-UI grows with steps and "
                    "pilot instances are dev seeds -> projection is approximate"}
        summary["timing_projection"]["max_method_sec_per_unit"] = (
            max((max(u["method_sec"].values()) for u in units_all), default=None))
        pc = {"zero_crashes": len(errs) == 0,
              "evidence_boundary_assertions_hold": summary["evidence_boundary_assertions_ok"],
              "projected_full_wall_le_60_min": summary["timing_projection"]["full_wall_min_projected"] <= 60.0,
              "all_rows_present": summary["complete"]}
        summary["pass_criteria"] = pc
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for m, v in summ.items():
            log(f"{m:9s} n={v['n']:3d} completion {v['completion']:.2f} refused {v['refused_floor']:3d} stream steps "
                f"{v['stream_steps_charged_mean']:.0f} consumed/stream {v['steps_consumed_stream_mean']:.0f} "
                f"FCR {v['false_certs']}/{v['n']} CPup {v['fcr_cp_upper']:.3f} tie {v['tie_rate']:.2f} "
                f"theta-cell {v['theta_cell_survival']} zero-cost {v['zero_cost_rate']:.2f} "
                f"wall/problem {v['wall_clock_per_problem_s']:.2f}s")
        log(f"prelim ratios: {json.dumps({k: (v['ratio_geo_mean'], v['ci95']) if isinstance(v, dict) and 'ci95' in v else v for k, v in ratios.items()}, default=str)}")
        log(f"projection: {summary['timing_projection']['full_wall_min_projected']:.1f} min; {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        if real:
            mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        if real:
            mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
