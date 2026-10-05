"""dev_offgrid_ladder (P1, cand_a): end-to-end off-grid ladder on dev seeds (methodology 2.2 / 3 / 4.3 / 6.1).

Levels x methods (stream protocol of section 3: n0 = 20 shared rounds, problem stream with each method re-using its
OWN ledger across problems, every method on its own fresh platform copy, CRN on the instance seed):
  R0  truth snapped to G_1          JPC, B3, B3g, B12                (no correction; realisable upper bound)
  R1  continuous truth, G_1         JPC, B3, B3g, B12                (no correction; unsafe diagnostic arm)
  R2  continuous truth, G_1         JPC_infl, B3g_infl, B12_infl     (+ B3: continuous class, never inflated)
  R3  continuous truth, G_{f*}      JPC_infl, B3g_infl, B12_infl for f* = 2 (all methods) and f* = 4 (JPC_infl);
                                    + JPC uncorrected on G_2 (diagnostic: theta*-cell survival / FCR on the finer grid)
Inflated certification (certify.minimax_infl, locked definition): R_grid(Theta_t) + 2 max_{g alive} eta_loc(g) <= eps.
Honest refusal: if even the smallest-eta cell cannot pass (2 min_g eta_loc(g) > eps; data-free, so no further data
can ever certify) the inflated method refuses at once. A refusal is censored and charged T_max (methodology 3:
"拒答一律按删失处理"; stepwise T_max = 3000, whole-trial T_max = 2.31e6). The actual steps consumed (0) are reported
separately, so nothing is hidden.
B3 is a continuous-class method (methodology 2.2: not inflated); R1 / R2 / R3 share the same continuous truth, so its
R1 run is the R2 / R3 B3 row (identical code path, identical seeds -> rows are copies, flagged `shared_from`).
Scoring (harness side only): true J at theta* by the exact propagator, normalised with the public class-max c_q of G_1.
Dev seeds only (pilot 630-634 x 6 problems, full 630-653 x 15 problems), stream 0 (seed 42, identity order).

Usage: run_dev_offgrid_ladder.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
TASK = "dev_offgrid_ladder"
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = int(2.31e6)

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.baselines.b12_tas import B12TaS  # noqa: E402
from dsswm.baselines.glm_linearised import (certify_linear, choose_design_action, fisher_rounds,  # noqa: E402
                                            value_and_grad, wald_beta)
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
from dsswm.streams.offgrid import make_offgrid_instance  # noqa: E402

C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]


def fkey(f):
    return f"{f:g}"


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


def vstar_of(truth: dict) -> np.ndarray:
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


def U1(q):
    return type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})()


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
def precompute(seeds, n_prob, fs, dev, log):
    """J tables and eta_loc for every (problem, f) (cached, truth-free); c_q (G_1 class max); harness J_true."""
    prop = NLPropagator(2, 2, NMAX, make_offgrid_instance(seeds[0], "R1").env.aspace, C_KNOWN, RHO_RET, device=dev)
    meta = {"cq": {}, "jtrue": {}, "eta_min_over_eps": {}, "sec": {}}
    probs = []
    for s in seeds:
        inst0 = make_offgrid_instance(s, "R0")
        inst1 = make_offgrid_instance(s, "R1")
        for q in inst1.problems[:n_prob]:
            probs.append((s, q, vstar_of(inst0.truth), vstar_of(inst1.truth)))
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
                np.save(pj, raw.astype(np.float32) if f == 4 else raw)
                t_j.append(time.perf_counter() - t0)
            pe = d / f"{q.pid}_etaloc_raw.npy"
            if not pe.exists():
                t0 = time.perf_counter()
                eta, _, _, info = eta_loc(prop, ncl, q, return_parts=True)
                np.save(pe, eta)
                np.save(d / f"{q.pid}_LJ.npy", np.array([info["parts"]["L_J"]]))
                t_e.append(time.perf_counter() - t0)
        meta["sec"][fkey(f)] = {"jtable_new_mean": float(np.mean(t_j)) if t_j else None, "n_jtable_new": len(t_j),
                                "eta_new_mean": float(np.mean(t_e)) if t_e else None, "n_eta_new": len(t_e)}
        log(f"tables f={fkey(f)} |Theta|={ncl.B}: new J {len(t_j)} ({meta['sec'][fkey(f)]['jtable_new_mean']}), "
            f"new eta {len(t_e)} ({meta['sec'][fkey(f)]['eta_new_mean']})")
        del ncl, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for (s, q, v0, v1) in probs:
        c = 1.0 / float(np.load(CACHE / "jtables_f1" / f"{q.pid}_raw.npy").max())
        meta["cq"][q.pid] = c
        plans = build_plans(prop, q)
        jt = j_values(prop, plans, np.stack([v0, v1]), q.utility.w, q.utility.w_ret) * c
        meta["jtrue"][q.pid] = {"R0": jt[0], "R1": jt[1]}
        for f in fs:
            meta["eta_min_over_eps"].setdefault(fkey(f), []).append(
                float(np.load(CACHE / f"jtables_f{fkey(f)}" / f"{q.pid}_etaloc_raw.npy").min()) * c / EPS)
    del prop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
class Arm:
    """Learner-side objects for one (instance, level, grid f) + harness truth for scoring only."""

    def __init__(self, seed, level, f, n_prob, meta, need_lr=True):
        torch.set_num_threads(1)
        self.seed, self.level, self.f = seed, level, f
        self.truth_level = "R0" if level == "R0" else "R1"
        self.n_prob = n_prob
        inst = make_offgrid_instance(seed, level, stream=0)
        self.aspace = inst.env.aspace
        self.problems = inst.problems[:n_prob]
        self.ncl = NLClass(ladder_grid(f), device="cpu")
        self.V = class_vectors(self.ncl)
        self.star = int(cell_index_of(vstar_of(inst.truth)[None], self.ncl)[0])      # harness only
        self.prop = NLPropagator(2, 2, NMAX, self.aspace, C_KNOWN, RHO_RET, device="cpu")
        d = CACHE / f"jtables_f{fkey(f)}"
        self.J, self.Reg, self.eta, self.Jt, self.cq = [], [], [], [], []
        for q in self.problems:
            c = meta["cq"][q.pid]
            J = np.load(d / f"{q.pid}_raw.npy").astype(float) * c
            self.J.append(J)
            self.Reg.append(regret_matrix(J))
            self.eta.append(np.load(d / f"{q.pid}_etaloc_raw.npy") * c)
            self.Jt.append(np.asarray(meta["jtrue"][q.pid][self.truth_level]))
            self.cq.append(c)
        if need_lr:
            self.LT = self.prop.tables(self.ncl.torch_params())[0]
            self.LT_np = self.LT.numpy()
            self.py, self.pe = class_prob_tables(self.ncl.np_params, C_KNOWN, NMAX, self.aspace.nb)
            self.inc = IncidenceIndex(self.prop.codec, self.aspace, NMAX)
            self.legal = np.arange(self.aspace.n)
            self.gid = group_ids(observable_signature(self.py, self.pe, max_level=1))
            self.all_single = len(np.unique(self.gid)) == self.ncl.B

    def fresh(self):
        return make_offgrid_instance(self.seed, self.level, stream=0)

    def row(self, k, method, status, pi, steps, extra=None, charged=None):
        Jt = self.Jt[k]
        tr = float(Jt.max() - Jt[pi]) if pi is not None else None
        cert = status == "CERTIFIED"
        tmax = TMAX_TRIAL if method.startswith("B12") else TMAX_STEP
        if charged is None:
            charged = steps if cert else max(steps, tmax)
        r = {"kind": "ladder", "instance": self.seed, "stream": 0, "noise_seed": 42, "level": self.level,
             "f": self.f, "problem": self.problems[k].pid, "k": k, "method": method, "status": status,
             "new_env_steps": int(charged), "steps_consumed": int(steps), "censored": not cert,
             "certified_policy": pi, "true_regret": tr, "false_cert": bool(cert and tr is not None and tr > EPS),
             "zero_cost": bool(cert and steps == 0), "H": self.problems[k].H,
             "n_policies": len(self.problems[k].policies), "eps": EPS, "delta": DELTA}
        if extra:
            r.update(extra)
        return r


def floor_any(eta, eps=EPS):
    """Data-free: no set (not even a singleton) can pass the inflated condition."""
    return bool(2.0 * float(np.min(eta)) > eps)


def run_jpc(arm: Arm, infl: bool, want_samples=False):
    method = "JPC_infl" if infl else "JPC"
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA) if hasattr(arm, "LT") else None
    led = EvidenceLedger([lr]) if lr is not None else None
    all_obs = list(inst.init_obs)
    if led is not None:
        for o in inst.init_obs:
            led.record(o, "initial")
    rng = np.random.default_rng([arm.seed, 0, 101])
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
        if lr is not None:
            assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
            mask = lr.mask().numpy()
            gc = certify_minimax(Reg, mask, EPS, TOP_M)
            pi = gc["pi"] if status == "CERTIFIED" else (gc["pi"])
            extra = {"set_size": int(mask.sum()), "r_bar_grid": float(gc["r_bar"]),
                     "theta_cell_alive": bool(mask[arm.star]),
                     "eta_loc_alive_max": float(eta[mask].max()) if mask.any() else None}
        else:
            pi, extra = None, {"set_size": None, "r_bar_grid": None, "theta_cell_alive": None,
                               "eta_loc_alive_max": None}
        if status != "CERTIFIED":
            pi_rep = pi                                  # reported (not certified) recommendation, for diagnostics
            pi = None
        else:
            pi_rep = pi
        extra.update({"refused_floor": refused, "eta_loc_min": float(eta.min()), "eta_loc_star_cell": float(eta[arm.star]),
                      "uncertified_recommendation": pi_rep if status != "CERTIFIED" else None,
                      "wall_clock_s": time.perf_counter() - t0, "rollouts": 0})
        rows.append(arm.row(k, method, status, pi, steps, extra))
        if want_samples and k < 2:
            samples.append({**rows[-1], "J_true": arm.Jt[k].round(4).tolist(), "trajectory_head": traj})
    return rows, samples


def run_b3(arm: Arm, grid_mode: bool, infl: bool = False):
    method = ("B3g_infl" if infl else "B3g") if grid_mode else "B3"
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)          # grid MLE theta_hat only (linearisation point)
    led = EvidenceLedger([lr])
    for o in inst.init_obs:
        led.record(o, "initial")
    hist = {}
    for o in inst.init_obs:
        key = (arm.prop.codec.encode(o.loads, o.engaged), o.action)
        hist[key] = hist.get(key, 0) + 1
    d = arm.V.shape[1]
    beta = wald_beta(d, DELTA)
    rng = np.random.default_rng([arm.seed, 42, 304 if grid_mode else 303])

    def build_V(v):
        Vm = np.eye(d)
        by_code = {}
        for (code, a), n in hist.items():
            by_code.setdefault(code, []).append((a, n))
        for code, lst in by_code.items():
            F = fisher_rounds(v, arm.prop.codec, arm.aspace, code, [a for a, _ in lst], C_KNOWN, 2, 2)
            Vm += np.einsum("k,kde->de", np.array([n for _, n in lst], float), F)
        return Vm

    rows = []
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        eta = arm.eta[k]
        plans = [arm.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
        U = type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": arm.cq[k]})()
        grad_cache, steps, kh_prev, Vm, status, refused = {}, 0, None, None, None, False
        gmask = None
        while True:
            if infl and floor_any(eta):
                status, refused = "NEED_DATA", True
                break
            kh = lr.mle()
            v = arm.V[kh]
            if kh != kh_prev:
                Vm = build_V(v)
                kh_prev = kh
            if kh not in grad_cache:
                grad_cache[kh] = value_and_grad(arm.prop, plans, v, U, 2, 2)
            Jlin, G = grad_cache[kh]
            Vinv = np.linalg.inv(Vm)
            lc = certify_linear(arm.J[k][kh], G, Vinv, beta, EPS)
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
            Fa = fisher_rounds(v, arm.prop.codec, arm.aspace, code, arm.legal, C_KNOWN, 2, 2)
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
            hist[(code, a)] = hist.get((code, a), 0) + 1
            Vm = Vm + Fa[list(arm.legal).index(a)]
            steps += 1
        extra = {"refused_floor": refused, "wald_beta": beta, "wall_clock_s": time.perf_counter() - t0,
                 "rollouts": 0}
        if grid_mode and gmask is not None:
            extra.update({"set_size": int(gmask.sum()), "theta_cell_alive": bool(gmask[arm.star])})
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
        u_max = c * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)
        sim_rng = np.random.default_rng([arm.seed, 42, 77, k])

        def sampler(kk, n, q=q, c=c, sim_rng=sim_rng):
            ys, es = env.simulate_batch(q.policies[kk], q.loads0, q.engaged0, q.H, n, sim_rng)
            return c * (ys @ q.utility.w + q.utility.w_ret * es)

        b = B12TaS(arm.J[k], u_max=u_max, delta=DELTA, eps=EPS, eta=eta, prior_logM=prior)
        if infl and floor_any(eta):
            mask = b.mask()
            r = {"status": "NEED_DATA", "pi": None, "steps": 0, "set_size": int(mask.sum()), "refused": True,
                 "pulls": [0] * len(q.policies)}
        else:
            r = b.run(sampler, q.H, max_steps=TMAX_TRIAL)
            r["refused"] = False
            mask = b.mask()
        prior = b.logM()
        st = r["status"]
        rows.append(arm.row(k, method, st, r["pi"] if st == "CERTIFIED" else None, r["steps"],
                            {"refused_floor": r["refused"], "set_size": int(mask.sum()),
                             "theta_cell_alive": bool(mask[arm.star]), "trials": int(np.sum(r["pulls"])),
                             "rollouts": int(np.sum(r["pulls"])), "wall_clock_s": time.perf_counter() - t0}))
    return rows


UNITS = [("R0", 1, ["JPC", "B3", "B3g", "B12"]),
         ("R1", 1, ["JPC", "B3", "B3g", "B12"]),
         ("R2", 1, ["JPC_infl", "B3g_infl", "B12_infl"]),
         ("R3", 2, ["JPC_infl", "B3g_infl", "B12_infl", "JPC"]),
         ("R3", 4, ["JPC_infl"])]


def run_unit(seed, level, f, methods, n_prob, meta, want_samples):
    torch.set_num_threads(1)
    t0 = time.time()
    need_lr = any(m in ("JPC", "B3", "B3g", "B3g_infl") for m in methods) or f != 4
    arm = Arm(seed, level, f, n_prob, meta, need_lr=need_lr)
    rows, samples, errors = [], [], []
    for m in methods:
        try:
            if m == "JPC":
                r, s = run_jpc(arm, False, want_samples)
                samples += s
            elif m == "JPC_infl":
                r, s = run_jpc(arm, True, want_samples)
                samples += s
            elif m == "B3":
                r = run_b3(arm, False)
            elif m == "B3g":
                r = run_b3(arm, True)
            elif m == "B3g_infl":
                r = run_b3(arm, True, True)
            elif m == "B12":
                r = run_b12(arm, False)
            elif m == "B12_infl":
                r = run_b12(arm, True)
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "level": level, "f": f, "method": m, "error": repr(e),
                           "tb": traceback.format_exc()})
    return rows, samples, errors, {"instance": seed, "level": level, "f": f, "sec": time.time() - t0}


# ------------------------------------------------------------------------------------ summaries
def arm_key(r):
    return f"{r['level']}" if r["level"] in ("R0", "R1", "R2") else f"R3_f{fkey(r['f'])}"


def summarize(rows):
    out = {}
    for ak in sorted({arm_key(r) for r in rows}):
        out[ak] = {}
        for m in sorted({r["method"] for r in rows if arm_key(r) == ak}):
            R = [r for r in rows if arm_key(r) == ak and r["method"] == m]
            cert = [r for r in R if r["status"] == "CERTIFIED"]
            nfc = sum(r["false_cert"] for r in R)
            lo, hi = clopper_pearson(nfc, len(R), 0.05)
            alive = [r["theta_cell_alive"] for r in R if r.get("theta_cell_alive") is not None]
            insts = sorted({r["instance"] for r in R})
            out[ak][m] = {
                "n": len(R), "completion": len(cert) / len(R),
                "status_counts": {s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R})},
                "refused_floor": int(sum(bool(r.get("refused_floor")) for r in R)),
                "stream_steps_charged_mean": float(np.mean([sum(r["new_env_steps"] for r in R if r["instance"] == i)
                                                            for i in insts])),
                "steps_consumed_median": float(np.median([r["steps_consumed"] for r in R])),
                "false_certs": int(nfc), "fcr": nfc / len(R), "fcr_cp": [lo, hi], "fcr_cp_upper": hi,
                "fcr_among_certified": nfc / max(len(cert), 1),
                "max_true_regret_certified": max([r["true_regret"] for r in cert], default=None),
                "theta_cell_survival": float(np.mean(alive)) if alive else None,
                "zero_cost_rate": float(np.mean([r["zero_cost"] for r in R])),
                "wall_clock_per_stream_s": float(np.mean([sum(r["wall_clock_s"] for r in R if r["instance"] == i)
                                                          for i in insts]))}
    return out


def stream_ratios(rows, summ):
    """Instance-level log ratio of stream totals (charged steps, +1), JPC(-infl) vs B3 and vs best valid."""
    out = {}
    for ak, ms in summ.items():
        num = "JPC" if ak in ("R0", "R1") else "JPC_infl"
        if num not in ms:
            continue
        cands = [m for m in ms if m.split("_")[0] in ("B3", "B3g", "B12")]
        valid = [m for m in cands if ms[m]["fcr_cp_upper"] <= 2 * DELTA]
        best = min(valid, key=lambda m: ms[m]["stream_steps_charged_mean"]) if valid else None
        # diagnostic only (pilot n too small for CP upper <= 2 delta: 0/30 gives 0.116): 0 observed false certs
        zfc = [m for m in cands if ms[m]["false_certs"] == 0]
        best_diag = min(zfc, key=lambda m: ms[m]["stream_steps_charged_mean"]) if zfc else None
        insts = sorted({r["instance"] for r in rows if arm_key(r) == ak})

        def tot(m, i):
            return sum(r["new_env_steps"] for r in rows if arm_key(r) == ak and r["method"] == m and r["instance"] == i)

        rec = {"numerator": num, "valid_denominators": valid, "best_valid": best, "best_zero_false_cert_diag": best_diag,
               "excluded_fcr": [m for m in cands if m not in valid]}
        for lab, den in (("vs_B3", "B3"), ("vs_best_valid", best), ("vs_best_zero_fc_diag", best_diag)):
            if den is None or den not in ms:
                rec[lab] = None
                continue
            lr_ = np.array([math.log((tot(num, i) + 1) / (tot(den, i) + 1)) for i in insts])
            rng = np.random.default_rng(0)
            bs = [lr_[rng.integers(len(lr_), size=len(lr_))].mean() for _ in range(2000)]
            rec[lab] = {"denominator": den, "ratio_geo_mean": float(math.exp(lr_.mean())),
                        "ci95": [float(math.exp(np.quantile(bs, .025))), float(math.exp(np.quantile(bs, .975)))],
                        "per_instance": [float(math.exp(x)) for x in lr_]}
        out[ak] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true", help="recompute summary.json from results.jsonl only")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if not (args.smoke or args.resummarize):
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    seeds = list(range(630, 635)) if pilot else list(range(630, 654))
    n_prob = 6 if pilot else 15
    if args.smoke:
        seeds, n_prob = [630], 2
    assert max(seeds) < 10000, "dev seeds only"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    log(f"start mode={args.mode} device={dev} seeds={seeds[0]}-{seeds[-1]} x {n_prob} problems, stream 0, "
        f"eps={EPS}, delta={DELTA}, T_max step={TMAX_STEP} trial={TMAX_TRIAL}, generator_hash={generator_hash()}")
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA,
               "dev_seeds": [seeds[0], seeds[-1]], "n_problems_per_instance": n_prob, "streams": [0],
               "eval_seeds_touched": False, "generator_hash": generator_hash(), "concurrent_run": True,
               "tmax_stepwise": TMAX_STEP, "tmax_whole_trial": TMAX_TRIAL, "units": [list(map(str, u[:2])) + u[2]
                                                                                     for u in UNITS]}
    if args.resummarize:
        rows = [json.loads(x) for x in open(out_dir / "results.jsonl")]
        old = json.loads((out_dir / "summary.json").read_text())
        old["by_arm"] = summarize(rows)
        old["stream_ratios"] = stream_ratios(rows, old["by_arm"])
        for ak, rec in old["stream_ratios"].items():
            d_ = rec["vs_best_zero_fc_diag"]
            log(f"resummarize {ak}: best valid {rec['best_valid']}; diag vs {rec['best_zero_false_cert_diag']}: "
                f"{d_ and round(d_['ratio_geo_mean'], 3)} CI {d_ and [round(x, 3) for x in d_['ci95']]}")
        (out_dir / "summary.json").write_text(json.dumps(old, indent=1, default=float))
        return
    try:
        progress(1, 4, "gpu: J tables + eta_loc")
        meta = precompute(seeds, n_prob, (1, 2, 4), dev, log)
        summary["eta_loc_min_over_eps"] = {f: {"median": float(np.median(v)), "min": float(np.min(v))}
                                           for f, v in meta["eta_min_over_eps"].items()}
        summary["table_sec"] = meta["sec"]
        log(f"min_g eta_loc/eps per problem (data-free floor check, need <= 0.5): "
            f"{ {f: round(v['min'], 2) for f, v in summary['eta_loc_min_over_eps'].items()} }")
        progress(2, 4, "cpu: ladder")
        from joblib import Parallel, delayed
        jobs = [(s, lv, f, ms) for s in seeds for (lv, f, ms) in UNITS]
        jobs.sort(key=lambda j: -len(j[3]))
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(run_unit)(s, lv, f, ms, n_prob, meta, s == seeds[0]) for (s, lv, f, ms) in jobs)
        rows = [r for rr, _, _, _ in res for r in rr]
        samples = [x for _, ss, _, _ in res for x in ss]
        errors = [e for _, _, ee, _ in res for e in ee]
        unit_sec = [u for _, _, _, u in res]
        # B3 is continuous-class and never inflated: R1 run == R2 / R3 B3 row (same truth, seeds, code path)
        for r in [r for r in rows if r["method"] == "B3" and r["level"] == "R1"]:
            for lv, f in (("R2", 1), ("R3", 2), ("R3", 4)):
                rows.append({**r, "level": lv, "f": f, "shared_from": "R1"})
        log(f"ladder: {len(rows)} rows, {len(errors)} errors, {time.time() - T0:.0f}s")
        for e in errors:
            log(f"ERROR {e['instance']} {e['level']} f={e['f']} {e['method']}: {e['error']}\n{e['tb']}")
        with open(out_dir / "results.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
        (samples_dir / "samples.json").write_text(json.dumps(samples, indent=1, default=float))
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        progress(3, 4, "summary")
        summ = summarize(rows)
        ratios = stream_ratios(rows, summ)
        summary["by_arm"] = summ
        summary["stream_ratios"] = ratios
        summary["n_cells"] = len({(r["instance"], r["problem"], arm_key(r)) for r in rows})
        summary["crashes"] = len(errors)
        summary["unit_sec"] = unit_sec
        # timing projection: CPU-sec per (problem, method) by arm -> full = 24 inst x 15 problems x 1 stream
        tp = {}
        for ak, ms in summ.items():
            tp[ak] = {m: v["wall_clock_per_stream_s"] / n_prob for m, v in ms.items()}
        full_cpu = sum(v * 24 * 15 for a in tp.values() for v in a.values())
        summary["timing_projection"] = {"cpu_sec_per_problem": tp, "full_24x15_cpu_sec_total": full_cpu,
                                        "full_wall_min_at_4_workers": full_cpu / 4 / 60,
                                        "note": "concurrent run (4 parallel tasks on 20 cores); tables cached"}
        infl_arms = {ak: summ[ak]["JPC_infl"]["completion"] for ak in summ if "JPC_infl" in summ[ak]}
        best_infl = max(infl_arms.values()) if infl_arms else 0.0
        fcr_all = all(v.get("fcr_cp_upper") is not None for a in summ.values() for v in a.values())
        pc = {"pipeline_end_to_end_0_crashes": len(errors) == 0,
              "jpc_infl_completion_ge_0.5_at_some_f": best_infl >= 0.5, "jpc_infl_completion_by_arm": infl_arms,
              "fcr_cp_upper_reported_for_every_method": fcr_all}
        summary["pass_criteria"] = pc
        summary["go_no_go"] = "GO" if all(v for k, v in pc.items() if isinstance(v, bool)) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for ak, ms in summ.items():
            for m, v in ms.items():
                log(f"{ak:7s} {m:9s} completion {v['completion']:.2f} refused {v['refused_floor']:2d} "
                    f"stream steps {v['stream_steps_charged_mean']:.0f} FCR {v['false_certs']}/{v['n']} "
                    f"CPup {v['fcr_cp_upper']:.3f} theta-cell {v['theta_cell_survival']} zero-cost "
                    f"{v['zero_cost_rate']:.2f} wall/stream {v['wall_clock_per_stream_s']:.1f}s")
        for ak, rec in ratios.items():
            d_ = rec["vs_best_zero_fc_diag"]
            log(f"ratio {ak} diag vs {rec['best_zero_false_cert_diag']} (0 false certs): "
                f"{d_ and round(d_['ratio_geo_mean'], 3)} CI {d_ and [round(x, 3) for x in d_['ci95']]}")
            log(f"ratio {ak}: vs B3 {rec['vs_B3'] and round(rec['vs_B3']['ratio_geo_mean'], 3)}, vs best valid "
                f"({rec['best_valid']}) {rec['vs_best_valid'] and round(rec['vs_best_valid']['ratio_geo_mean'], 3)}")
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        if not args.smoke:
            mark_done("success", f"{args.mode}: P1 {summary['go_no_go']} (JPC_infl completion {infl_arms})")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        if not args.smoke:
            mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
