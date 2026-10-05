"""HR1 main contrast, one chunk (family A, cand_a; methodology 2.2 / 3 / 4.3 / 5 / 6.1).

Written for hr1_r3_b (R3: continuous R1 truth, learner grid G_{f*}, f* from plan/prereg_lock.json, local inflation);
the task id is a CLI argument so the identical code path can serve every R3 chunk (instance ranges are read from the
lock's eval_manifest, never hard-coded).

Methods (each on its own fresh platform copy, CRN on (instance, stream); each method re-uses its OWN ledger across the
15-problem stream; n0 = 20 shared initial rounds):
  JPC_infl   joint LR set on G_{f*} + inflated minimax certificate R_grid + 2 max_alive eta_loc <= eps + DDA
             (honest refusal when the data-free floor 2 min_g eta_loc > eps holds: no data can ever certify)
  B3         GLM linearisation + Wald chi2_12 (continuous class; never inflated)
  B3-UI      continuous class with the SAME UI threshold log(1/delta) (same-threshold control; never inflated);
             design identical to B3; the UI certificate is evaluated on a geometric check schedule (every
             max(1, 3% of the rounds seen) steps; a sparser check of an anytime-valid set only costs steps)
  B3g_infl   Wald ellipsoid intersect G_{f*}, exact grid J, inflated certificate
  B8_infl    full identification (J-width over the alive set < tau = eps/2), then the inflated certificate
  B12_infl   finite-class whole-trial Track-and-Stop, inflated certificate
  B2         whole-trial residual + empirical-Bernstein LUCB (appendix row; model-free, not inflated)
  JPC        uncorrected JPC on G_{f*} (diagnostic only: theta*-cell survival / FCR on the finer grid; not in HR1)
Refusals / censoring: charged T_max (stepwise 3000, whole-trial 2.31e6); actual steps consumed reported separately.
B3 / B3-UI linearise at / start from the G_1 grid MLE (identical code path to the R1/R2 rows of dev_offgrid_ladder).
Scoring (harness only): true J at the continuous theta* by the exact propagator, normalised by the public G_1
class-max c_q. Evidence boundary: after every problem of every LR-based method  env.n_steps == lr.n_rounds.
Certification always uses exact enumeration over |Theta| <= 64000 rows, so no dual/BnB call is made (H0 vacuous;
counted as dual_calls = 0).

Pilot: dev instances 660-663 x 5 problems x 1 stream (seed 42).  Full: lock range x 3 streams x 15 problems
(requires dsswm.stats.prereg.assert_locked()). Incremental: one results.jsonl line per (instance, stream, method,
problem); finished (instance, stream, method) units are skipped on restart.

Level: hr1_r2_* -> R2 (continuous truth, grid G_1, inflated); hr1_r3_* -> R3 (grid G_{f*}).
Usage: run_hr1_chunk.py --task hr1_r3_b --mode {pilot,full} [--workers 4] [--smoke] [--resummarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
PILOT_SEEDS = [660, 661, 662, 663]
# Ladder level of this run: R3 (grid G_{f*}) or R2 (grid G_1); set by main() from the task id and exported via the
# environment so joblib/loky workers see the same value. R2 and R3 share the R1 continuous truth (streams.offgrid),
# so the instance is identical; only the learner grid f and the row label differ.
LEVEL = os.environ.get("HR1_LEVEL", "R3")

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.baselines.b12_tas import B12TaS  # noqa: E402
from dsswm.baselines.b3_ui import certify_b3_ui  # noqa: E402
from dsswm.certify.eta_loc import torch_params_from_vectors  # noqa: E402
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
GROUPS = [("grid", ["JPC_infl", "B3g_infl", "B8_infl", "JPC"]), ("B3", ["B3"]), ("B3-UI", ["B3-UI"]),
          ("B3-UIc", ["B3-UIc"]),
          ("B12_infl", ["B12_infl"]), ("B2", ["B2"])]
DENOM_CANDS = ["B3", "B3g_infl", "B3-UI", "B8_infl", "B12_infl"]
WHOLE_TRIAL = ("B12", "B12_infl", "B2", "B1")
UI_CHECK_FRAC = 0.05          # overridden by --ui-check-frac (passed to workers explicitly)


def fkey(f):
    return f"{f:g}"


def vstar_of(truth: dict) -> np.ndarray:
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


def U1(q):
    return type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})()


def atomic_save(path: Path, arr):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
def precompute(seeds, n_prob, fstar, dev, log):
    """G_1 J tables (c_q, B3 / B3-UI / B2 proxy) and G_{f*} J tables + eta_loc; truth-free, cached, file-locked
    (concurrent chunks may share instances); harness J_true at the continuous theta*."""
    prop = NLPropagator(2, 2, NMAX, make_offgrid_instance(seeds[0], LEVEL).env.aspace, C_KNOWN, RHO_RET, device=dev)
    meta = {"cq": {}, "jtrue": {}, "eta_min_over_eps": [], "sec": {}}
    probs = []
    for s in seeds:
        inst = make_offgrid_instance(s, LEVEL, stream=0)          # stream 0 = identity order: all 15 problems
        for q in inst.problems[:n_prob]:
            probs.append((s, q, vstar_of(inst.truth)))
    for f, need_eta in ((1, False), (fstar, True)):
        ncl = NLClass(ladder_grid(f), device=dev)
        params = None
        d = CACHE / f"jtables_f{fkey(f)}"
        d.mkdir(parents=True, exist_ok=True)
        t_j, t_e = [], []
        with open(d / ".hr1_chunk.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for (s, q, v) in probs:
                pj = d / f"{q.pid}_raw.npy"
                if not pj.exists():
                    params = params or ncl.torch_params()
                    t0 = time.perf_counter()
                    raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, U1(q))
                    atomic_save(pj, raw.astype(np.float32) if f == 4 else raw)
                    t_j.append(time.perf_counter() - t0)
                pe = d / f"{q.pid}_etaloc_raw.npy"
                if need_eta and not pe.exists():
                    t0 = time.perf_counter()
                    eta, _, _, info = eta_loc(prop, ncl, q, return_parts=True)
                    atomic_save(d / f"{q.pid}_LJ.npy", np.array([info["parts"]["L_J"]]))
                    atomic_save(pe, eta)
                    t_e.append(time.perf_counter() - t0)
            fcntl.flock(lk, fcntl.LOCK_UN)
        meta["sec"][fkey(f)] = {"jtable_new_mean": float(np.mean(t_j)) if t_j else None, "n_jtable_new": len(t_j),
                                "eta_new_mean": float(np.mean(t_e)) if t_e else None, "n_eta_new": len(t_e),
                                "Theta": int(ncl.B)}
        log(f"tables f={fkey(f)} |Theta|={ncl.B}: new J {len(t_j)} ({meta['sec'][fkey(f)]['jtable_new_mean']}), "
            f"new eta {len(t_e)} ({meta['sec'][fkey(f)]['eta_new_mean']})")
        del ncl, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for (s, q, v) in probs:
        c = 1.0 / float(np.load(CACHE / "jtables_f1" / f"{q.pid}_raw.npy").max())
        meta["cq"][q.pid] = c
        plans = build_plans(prop, q)
        meta["jtrue"][q.pid] = j_values(prop, plans, v[None], q.utility.w, q.utility.w_ret)[0] * c
        meta["eta_min_over_eps"].append(
            float(np.load(CACHE / f"jtables_f{fkey(fstar)}" / f"{q.pid}_etaloc_raw.npy").min()) * c / EPS)
    if torch.cuda.is_available():
        meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del prop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
class Arm:
    """Learner-side objects for one (instance, stream, grid f) + harness truth for scoring only."""

    def __init__(self, seed, stream, f, n_prob, meta, need_lr=True, need_eta=True):
        torch.set_num_threads(1)
        self.seed, self.stream, self.f, self.n_prob = seed, stream, f, n_prob
        self.noise = STREAMS[stream][0]
        inst = make_offgrid_instance(seed, LEVEL, stream=stream)
        self.aspace = inst.env.aspace
        self.problems = inst.problems[:n_prob] if stream == 0 else [q for q in inst.problems
                                                                    if q.pid in meta["cq"]][:n_prob]
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
            self.eta.append(np.load(d / f"{q.pid}_etaloc_raw.npy") * c if need_eta else np.zeros(len(J)))
            self.Jt.append(np.asarray(meta["jtrue"][q.pid]))
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
        return make_offgrid_instance(self.seed, LEVEL, stream=self.stream)

    def row(self, k, method, status, pi, steps, extra=None, charged=None):
        Jt = self.Jt[k]
        tr = float(Jt.max() - Jt[pi]) if pi is not None else None
        cert = status == "CERTIFIED"
        tmax = TMAX_TRIAL if method in WHOLE_TRIAL else TMAX_STEP
        if charged is None:
            charged = steps if cert else max(steps, tmax)
        q = self.problems[k]
        r = {"kind": "hr1", "instance": self.seed, "stream": self.stream, "noise_seed": self.noise, "level": LEVEL,
             "f": self.f, "problem": q.pid, "gen_index": int(q.meta.get("gen_index", -1)), "k": k, "method": method,
             "status": status, "new_env_steps": int(charged), "steps_consumed": int(steps), "censored": not cert,
             "certified_policy": pi, "true_regret": tr, "false_cert": bool(cert and tr is not None and tr > EPS),
             "zero_cost": bool(cert and steps == 0), "H": q.H, "n_policies": len(q.policies), "eps": EPS,
             "delta": DELTA, "true_gap_top2": float(np.sort(Jt)[-1] - np.sort(Jt)[-2]) if len(Jt) > 1 else None}
        if extra:
            r.update(extra)
        return r


def floor_any(eta, eps=EPS):
    """Data-free: no set (not even a singleton) can pass the inflated condition."""
    return bool(2.0 * float(np.min(eta)) > eps)


def _init_lr(arm):
    inst = arm.fresh()
    env = inst.env
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)
    led = EvidenceLedger([lr])
    for o in inst.init_obs:
        led.record(o, "initial")
    return inst, env, env.handle(), lr, led


def run_jpc(arm: Arm, infl: bool, want_samples=False):
    method = "JPC_infl" if infl else "JPC"
    inst, env, h, lr, led = _init_lr(arm)
    all_obs = list(inst.init_obs)
    rng = np.random.default_rng([arm.seed, arm.noise, 101])
    rows, samples = [], []
    for k, q in enumerate(arm.problems):
        Reg, eta = arm.Reg[k], arm.eta[k]
        t0 = time.perf_counter()
        steps, status, traj, refused = 0, None, [], False
        while True:
            if infl and floor_any(eta):
                status, refused = "NEED_DATA", True
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
                 "eta_loc_alive_max": float(eta[mask].max()) if mask.any() else None,
                 "refused_floor": refused, "eta_loc_min": float(eta.min()), "eta_loc_star_cell": float(eta[arm.star]),
                 "uncertified_recommendation": gc["pi"] if status != "CERTIFIED" else None,
                 "env_n_steps": int(env.n_steps), "lr_n_rounds": int(lr.n_rounds),
                 "wall_clock_s": time.perf_counter() - t0, "rollouts": 0, "dual_calls": 0}
        rows.append(arm.row(k, method, status, pi, steps, extra))
        if want_samples and k < 2:
            samples.append({**rows[-1], "J_true": arm.Jt[k].round(4).tolist(), "trajectory_head": traj})
    return rows, samples


def run_b3(arm: Arm, mode: str, want_samples=False, ui_frac=UI_CHECK_FRAC):
    """mode in {'B3', 'B3g_infl', 'B3-UI', 'B3-UIc'} (B3g_infl: grid intersect, inflated; B3-UI: UI threshold with
    the locked G_1 grid plug-in numerator, same design as B3; B3-UIc: same UI set but with a CONTINUOUS predictable
    plug-in numerator q_t = p(x_t | v_hat_ui at the last check), initial n0 rounds scored by the grid plug-in --
    any predictable numerator keeps Ville validity; pilot sensitivity row, not in the locked denominator set)."""
    grid_mode, ui, uic = mode == "B3g_infl", mode in ("B3-UI", "B3-UIc"), mode == "B3-UIc"
    inst, env, h, lr, led = _init_lr(arm)
    all_obs = list(inst.init_obs)
    hist = {}
    for o in inst.init_obs:
        key = (arm.prop.codec.encode(o.loads, o.engaged), o.action)
        hist[key] = hist.get(key, 0) + 1
    d = arm.V.shape[1]
    beta = wald_beta(d, DELTA)
    rng = np.random.default_rng([arm.seed, arm.noise, {"B3": 303, "B3g_infl": 304, "B3-UI": 305, "B3-UIc": 305}[mode]])
    log_num_c, LT_c = lr.log_num, None

    def tables_at(vv):
        return arm.prop.tables(torch_params_from_vectors(torch.as_tensor(np.asarray(vv, float))[None], 2, 2,
                                                         arm.aspace.nb))[0]

    def build_V(v):
        Vm = np.eye(d)
        by_code = {}
        for (code, a), n in hist.items():
            by_code.setdefault(code, []).append((a, n))
        for code, lst in by_code.items():
            F = fisher_rounds(v, arm.prop.codec, arm.aspace, code, [a for a, _ in lst], C_KNOWN, 2, 2)
            Vm += np.einsum("k,kde->de", np.array([n for _, n in lst], float), F)
        return Vm

    rows, samples = [], []
    v_ui = None
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        eta = arm.eta[k]
        plans = [arm.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]
        U = type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": arm.cq[k]})()
        grad_cache, steps, kh_prev, Vm, status, refused = {}, 0, None, None, None, False
        gmask, pi, next_ui, n_ui, ui_beta, ui_sec, traj = None, None, 0, 0, None, 0.0, []
        while True:
            if grid_mode and floor_any(eta):
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
                cert = certify_minimax_infl(arm.Reg[k], gmask, eta, EPS, TOP_M)
                if cert["status"].value == "NEED_DATA" and inflation_floor(eta, gmask, EPS):
                    status, refused = "NEED_DATA", True
                    break
                done, pi = cert["status"].value == "CERTIFIED", cert["pi"]
            elif ui:
                done = False
                if steps >= next_ui:
                    tu = time.perf_counter()
                    uc = certify_b3_ui(arm.prop, plans, all_obs, log_num_c if uic else lr.log_num,
                                       v if v_ui is None else v_ui, arm.ncl.grid, U, DELTA, EPS)
                    ui_sec += time.perf_counter() - tu
                    n_ui += 1
                    v_ui, ui_beta = uc["v_hat"], uc["beta"]
                    if uic:
                        LT_c = tables_at(v_ui)                 # plug-in for FUTURE rounds only (predictable)
                    if uc["status"].value == "MODEL_CONFLICT":
                        status = "MODEL_CONFLICT"
                        break
                    done, pi = uc["status"].value == "CERTIFIED", uc["pi"]
                    next_ui = steps + max(1, int(ui_frac * len(all_obs)))
                    if want_samples and len(traj) < 40:
                        traj.append({"step": steps, "ui_beta": round(float(ui_beta), 3),
                                     "r_bar": round(float(uc["r_bar"]), 4), "pi": uc["pi"]})
            else:
                done, pi = lc["certified"], lc["pi"]
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
            all_obs.append(obs)
            if uic:
                LT_c = tables_at(v) if LT_c is None else LT_c
                log_num_c += float(arm.prop.loglik(LT_c, [obs]).sum())
            hist[(code, a)] = hist.get((code, a), 0) + 1
            Vm = Vm + Fa[list(arm.legal).index(a)]
            steps += 1
        assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
        extra = {"refused_floor": refused, "wald_beta": beta, "wall_clock_s": time.perf_counter() - t0, "rollouts": 0,
                 "env_n_steps": int(env.n_steps), "lr_n_rounds": int(lr.n_rounds), "dual_calls": 0}
        if grid_mode and gmask is not None:
            extra.update({"set_size": int(gmask.sum()), "theta_cell_alive": bool(gmask[arm.star]),
                          "eta_loc_min": float(eta.min())})
        if ui:
            extra.update({"ui_beta_end": ui_beta, "ui_checks": n_ui, "ui_sec": ui_sec, "ui_check_frac": ui_frac})
        rows.append(arm.row(k, mode, status, pi if status == "CERTIFIED" else None, steps, extra))
        if want_samples and ui and k < 2:
            samples.append({**rows[-1], "J_true": arm.Jt[k].round(4).tolist(), "ui_trajectory_head": traj})
    return rows, samples


def run_b8(arm: Arm, tau=EPS / 2):
    """Full identification (J-width over the alive set < tau), then the inflated certificate (same Theta^+)."""
    method = "B8_infl"
    inst, env, h, lr, led = _init_lr(arm)
    rng = np.random.default_rng([arm.seed, arm.noise, 808, int(round(tau * 1e4)), False])
    rows = []
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        J, eta = arm.J[k], arm.eta[k]
        steps, status, width, refused = 0, None, None, False
        while True:
            if floor_any(eta):
                status, refused = "NEED_DATA", True
                break
            mask = lr.mask().numpy()
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                status = "MODEL_CONFLICT"
                break
            Jm = J[idx]
            width = float((Jm.max(0) - Jm.min(0)).max())
            if width < tau:
                cert = certify_minimax_infl(arm.Reg[k], mask, eta, EPS, TOP_M)
                status = "CERTIFIED" if cert["status"].value == "CERTIFIED" else "IDENTIFIED_UNCERTIFIED"
                if status != "CERTIFIED" and inflation_floor(eta, mask, EPS):
                    refused = True
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
            a, _ = dda_choose(code, arm.py[kh:kh + 1], arm.pe[kh:kh + 1], arm.LT_np[kh], arm.py[blk], arm.pe[blk],
                              margins, arm.inc, arm.prop, arm.legal, rng)
            led.record(h.step(a), "probe", q.pid)
            steps += 1
        assert env.n_steps == lr.n_rounds, "step accounting mismatch"
        mask = lr.mask().numpy()
        pi = int(np.argmax(J[lr.mle()])) if status == "CERTIFIED" else None
        rows.append(arm.row(k, method, status, pi, steps,
                            {"tau": tau, "id_width_end": width, "set_size": int(mask.sum()),
                             "theta_cell_alive": bool(mask[arm.star]), "refused_floor": refused,
                             "eta_loc_min": float(eta.min()), "env_n_steps": int(env.n_steps),
                             "lr_n_rounds": int(lr.n_rounds), "wall_clock_s": time.perf_counter() - t0,
                             "rollouts": 0, "dual_calls": 0}))
    return rows


def _u_max(q, c):
    return c * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)


def run_b12(arm: Arm):
    method = "B12_infl"
    env = arm.fresh().env
    rows, prior = [], None
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        c, eta = arm.cq[k], arm.eta[k]
        sim_rng = np.random.default_rng([arm.seed, arm.noise, 77, k])

        def sampler(kk, n, q=q, c=c, sim_rng=sim_rng):
            ys, es = env.simulate_batch(q.policies[kk], q.loads0, q.engaged0, q.H, n, sim_rng)
            return c * (ys @ q.utility.w + q.utility.w_ret * es)

        b = B12TaS(arm.J[k], u_max=_u_max(q, c), delta=DELTA, eps=EPS, eta=eta, prior_logM=prior)
        if floor_any(eta):
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
                             "theta_cell_alive": bool(mask[arm.star]), "eta_loc_min": float(eta.min()),
                             "trials": int(np.sum(r["pulls"])), "rollouts": int(np.sum(r["pulls"])),
                             "wall_clock_s": time.perf_counter() - t0}))
    return rows


def run_b2(arm: Arm):
    """Whole-trial residual + EB LUCB (appendix). Proxy = G_1 grid-MLE J after n0 (a per-arm centring constant)."""
    method = "B2"
    inst, env, h, lr, led = _init_lr(arm)
    kh0 = lr.mle()
    history, rows = {}, []
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        c = arm.cq[k]
        keys = [(p.name, tuple(int(x) for x in q.loads0), tuple(int(x) for x in q.engaged0), q.H) for p in q.policies]
        prior = {}
        for a, key in enumerate(keys):
            if key in history:
                ys, es = history[key]
                prior[a] = list(c * (ys.astype(float) @ q.utility.w + q.utility.w_ret * es.astype(float)))
        counters = {}

        def sampler(a, n, q=q, k=k, c=c):
            cc = counters.get(a, 0)
            counters[a] = cc + 1
            r = np.random.default_rng([arm.seed, arm.noise, k, a, cc, 55])
            ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
            store = sampler.raw.setdefault(a, ([], []))
            store[0].append(ys.astype(np.int8))
            store[1].append(es.astype(np.int8))
            return c * (ys @ q.utility.w + q.utility.w_ret * es)
        sampler.raw = {}
        res = lucb(sampler, len(q.policies), q.H, EPS, DELTA, _u_max(q, c), TMAX_TRIAL, kind="eb",
                   proxy=arm.J[k][kh0], prior=prior)
        for a, key in enumerate(keys):
            if a in sampler.raw:
                ys = np.concatenate(sampler.raw[a][0])
                es = np.concatenate(sampler.raw[a][1])
                n_used = int(res["n_per_arm"][a]) - len(prior.get(a, []))
                ys, es = ys[:n_used], es[:n_used]
                if key in history:
                    ys = np.concatenate([history[key][0], ys])
                    es = np.concatenate([history[key][1], es])
                history[key] = (ys, es)
        st = res["status"]
        rows.append(arm.row(k, method, st, res["pi"] if st == "CERTIFIED" else None, res["steps"],
                            {"trials": int(res["pulls"]), "reused_trials": int(res["n_reused"]),
                             "rollouts": int(res["pulls"]), "final_gap": res.get("gap"),
                             "wall_clock_s": time.perf_counter() - t0}))
    return rows


def run_unit(seed, stream, group, methods, n_prob, meta, fstar, want_samples, ui_frac=UI_CHECK_FRAC):
    torch.set_num_threads(1)
    t0 = time.time()
    rows, samples, errors = [], [], []
    try:
        if group == "grid":
            arm = Arm(seed, stream, fstar, n_prob, meta)
        elif group == "B12_infl":
            arm = Arm(seed, stream, fstar, n_prob, meta, need_lr=False)
        else:                                            # B3, B3-UI, B2: G_1 linearisation / proxy (no inflation)
            arm = Arm(seed, stream, 1, n_prob, meta, need_eta=False)
    except Exception as e:  # noqa: BLE001
        return rows, samples, [{"instance": seed, "stream": stream, "method": group, "error": repr(e),
                                "tb": traceback.format_exc()}], {"instance": seed, "stream": stream, "group": group,
                                                                 "methods": methods, "sec": time.time() - t0}
    t_arm = time.time() - t0
    for m in methods:
        try:
            if m in ("JPC", "JPC_infl"):
                r, s = run_jpc(arm, m == "JPC_infl", want_samples)
                samples += s
            elif m in ("B3", "B3g_infl", "B3-UI", "B3-UIc"):
                r, s = run_b3(arm, m, want_samples, ui_frac)
                samples += s
            elif m == "B8_infl":
                r = run_b8(arm)
            elif m == "B12_infl":
                r = run_b12(arm)
            elif m == "B2":
                r = run_b2(arm)
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "stream": stream, "method": m, "error": repr(e),
                           "tb": traceback.format_exc()})
    return rows, samples, errors, {"instance": seed, "stream": stream, "group": group, "methods": methods,
                                   "arm_sec": t_arm, "sec": time.time() - t0}


# ------------------------------------------------------------------------------------ summaries
def summarize(rows, n_prob):
    out = {}
    for m in sorted({r["method"] for r in rows}):
        R = [r for r in rows if r["method"] == m]
        cert = [r for r in R if r["status"] == "CERTIFIED"]
        nfc = sum(r["false_cert"] for r in R)
        lo, hi = clopper_pearson(nfc, len(R), 0.05)
        alive = [r["theta_cell_alive"] for r in R if r.get("theta_cell_alive") is not None]
        units = sorted({(r["instance"], r["stream"]) for r in R})
        zc_by_k = {}
        for kk in range(n_prob):
            Rk = [r for r in R if r["k"] == kk]
            if Rk:
                zc_by_k[kk] = float(np.mean([r["zero_cost"] for r in Rk]))
        cum = {}
        for u in units:
            Ru = sorted([r for r in R if (r["instance"], r["stream"]) == u], key=lambda r: r["k"])
            cum[f"{u[0]}_{u[1]}"] = (N0 + np.cumsum([r["new_env_steps"] for r in Ru])).tolist()
        out[m] = {
            "n": len(R), "completion": len(cert) / len(R),
            "status_counts": {s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R})},
            "refused_floor": int(sum(bool(r.get("refused_floor")) for r in R)),
            "stream_steps_charged_mean": float(np.mean([sum(r["new_env_steps"] for r in R
                                                            if (r["instance"], r["stream"]) == u) for u in units])),
            "steps_consumed_median": float(np.median([r["steps_consumed"] for r in R])),
            "false_certs": int(nfc), "fcr": nfc / len(R), "fcr_cp": [lo, hi], "fcr_cp_upper": hi,
            "fcr_among_certified": nfc / max(len(cert), 1),
            "max_true_regret_certified": max([r["true_regret"] for r in cert], default=None),
            "theta_cell_survival": float(np.mean(alive)) if alive else None,
            "zero_cost_rate": float(np.mean([r["zero_cost"] for r in R])),
            "zero_cost_rate_by_k": zc_by_k,
            "cum_cost_with_n0_mean": np.mean(list(cum.values()), 0).tolist() if cum else None,
            "eta_loc_min_over_eps_median": (float(np.median([r["eta_loc_min"] for r in R if "eta_loc_min" in r])) / EPS
                                            if any("eta_loc_min" in r for r in R) else None),
            "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in R])),
            "wall_clock_per_stream_s": float(np.mean([sum(r["wall_clock_s"] for r in R
                                                          if (r["instance"], r["stream"]) == u) for u in units]))}
    return out


def stream_ratios(rows, summ):
    """Per-chunk descriptive ratios (instance level: mean over streams of the stream totals, then log ratio).
    The binding HR1 denominator is chosen by r2_analysis_aggregate after merging all chunks."""
    num = "JPC_infl"
    if num not in summ:
        return {}
    insts = sorted({r["instance"] for r in rows})

    def tot(m, i):
        streams = sorted({r["stream"] for r in rows if r["instance"] == i and r["method"] == m})
        return float(np.mean([sum(r["new_env_steps"] for r in rows if r["method"] == m and r["instance"] == i
                                  and r["stream"] == s) for s in streams])) if streams else None

    cands = [m for m in DENOM_CANDS if m in summ]
    valid = [m for m in cands if summ[m]["fcr_cp_upper"] <= 2 * DELTA]
    best = min(valid, key=lambda m: summ[m]["stream_steps_charged_mean"]) if valid else None
    zfc = [m for m in cands if summ[m]["false_certs"] == 0]
    best_diag = min(zfc, key=lambda m: summ[m]["stream_steps_charged_mean"]) if zfc else None
    rec = {"numerator": num, "valid_denominators_chunk": valid, "best_valid_chunk": best,
           "best_zero_false_cert_diag": best_diag, "excluded_fcr": [m for m in cands if m not in valid]}
    for den in cands + (["B2"] if "B2" in summ else []):
        lr_ = np.array([math.log((tot(num, i) + 1) / (tot(den, i) + 1)) for i in insts
                        if tot(num, i) is not None and tot(den, i) is not None])
        if not len(lr_):
            continue
        rng = np.random.default_rng(42)
        bs = [lr_[rng.integers(len(lr_), size=len(lr_))].mean() for _ in range(2000)]
        tie = [r1["zero_cost"] and r2["zero_cost"] for r1 in rows if r1["method"] == num for r2 in rows
               if r2["method"] == den and (r2["instance"], r2["stream"], r2["problem"]) ==
               (r1["instance"], r1["stream"], r1["problem"])]
        rec[f"vs_{den}"] = {"denominator": den, "ratio_geo_mean": float(math.exp(lr_.mean())),
                            "ci95_chunk_bootstrap2000": [float(math.exp(np.quantile(bs, .025))),
                                                         float(math.exp(np.quantile(bs, .975)))],
                            "tie_rate": float(np.mean(tie)) if tie else None,
                            "per_instance": [float(math.exp(x)) for x in lr_]}
    return rec


def full_seeds(lock, task):
    """Eval instances of a chunk in full mode. For R2 chunks the lock's R2_scope (family-A exit, methodology 6.1)
    takes precedence over eval_manifest.per_task_ranges; returns None when the scope says "skip"."""
    if task.startswith("hr1_r2") and task in (lock.get("R2_scope") or {}):
        sc = lock["R2_scope"][task]
        if sc == "skip":
            return None
        return list(range(int(sc[0]), int(sc[1]) + 1))
    rng_ = lock["eval_manifest"]["per_task_ranges"][task]
    return [s for a, b in rng_ for s in range(a, b + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hr1_r3_b")
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--ui-check-frac", type=float, default=UI_CHECK_FRAC,
                    help="B3-UI(c) certificate check every max(1, frac * rounds seen) steps")
    ap.add_argument("--diag-streams", default="0,1,2", help="streams on which the uncorrected JPC diagnostic runs")
    ap.add_argument("--uic-streams", default="0,1,2", help="streams on which the B3-UIc sensitivity row runs")
    ap.add_argument("--tag", default="", help="output directory suffix (timing / sensitivity variants)")
    args = ap.parse_args()
    TASK = args.task
    pilot = args.mode == "pilot"
    lock = json.loads(LOCK.read_text())
    assert lock.get("version") == 2, "prereg lock v2 required"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    global LEVEL
    if TASK.startswith("hr1_r2"):
        LEVEL = "R2"
    elif TASK.startswith("hr1_r3"):
        LEVEL = "R3"
    else:
        raise SystemExit(f"run_hr1_chunk.py serves hr1_r2_* / hr1_r3_* only, got {TASK}")
    os.environ["HR1_LEVEL"] = LEVEL                     # inherited by loky workers spawned below
    fstar = 1 if LEVEL == "R2" else int(lock["f_star"])  # R2 = continuous truth on G_1 (methodology 2.2)
    full_seed_list = full_seeds(lock, TASK)
    n_full_inst = len(full_seed_list) if full_seed_list is not None else 0
    if not pilot and full_seed_list is None and not args.smoke:
        reason = f"R2_scope[{TASK}] = skip: {lock.get('R2_scope', {}).get('reason', 'R2 left family A')}"
        print(reason, flush=True)
        (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": "skipped", "summary": reason,
                                                            "final_progress": {},
                                                            "timestamp": datetime.now().isoformat()}))
        pidf = RES_ROOT / f"{TASK}.pid"
        if pidf.exists():
            pidf.unlink()
        return
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else "") + args.tag)
    diag_streams = {int(x) for x in args.diag_streams.split(",") if x != ""}
    uic_streams = {int(x) for x in args.uic_streams.split(",") if x != ""}
    quiet = args.smoke or bool(args.tag)                 # smoke / variant runs never touch PID / PROGRESS / DONE
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    def progress(step, total, phase, metric=None):
        if quiet:
            return
        (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
            "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
            "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))

    def mark_done(status, summary):
        if quiet:
            return
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
                                                            "final_progress": fp,
                                                            "timestamp": datetime.now().isoformat()}))

    if not (quiet or args.resummarize):
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if pilot:
        seeds, n_prob, streams = PILOT_SEEDS, 5, [0]
        assert max(seeds) < 10000, "dev seeds only in pilot"
    else:
        seeds = full_seed_list
        n_prob, streams = int(lock["eval_manifest"]["K_problems"]), [0, 1, 2]
    if args.smoke:
        seeds, n_prob = seeds[:1], 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "level": LEVEL, "f_star": fstar, "grid_f": fstar, "eps": EPS,
               "delta": DELTA, "instances": [seeds[0], seeds[-1]], "n_instances": len(seeds),
               "n_problems_per_stream": n_prob, "streams": streams, "eval_seeds_touched": not pilot,
               "generator_hash": generator_hash(), "concurrent_run": True, "tmax_stepwise": TMAX_STEP,
               "tmax_whole_trial": TMAX_TRIAL, "lock_status": lock.get("status"),
               "lock_sha256": lock.get("sha256") or lock.get("sha256_provisional"),
               "groups": [[g, ms] for g, ms in GROUPS], "denominator_candidates": DENOM_CANDS,
               "note_denominator": "binding HR1 denominator chosen by r2_analysis_aggregate on the merged chunks",
               "note_R3_premise": "R3 premise h <= h*/2 fails on dev (prereg_lock R3_note); reported with every number"}
    if LEVEL == "R2":
        summary.pop("note_R3_premise")
        summary["note_R2_scope"] = ("R2 fails h <= h*/2 on dev -> R2_in_family_A=false: 48-instance diagnostic only "
                                    "(prereg_lock R2_scope); the 'f_star' field here is the R2 grid f=1, not lock f*")
        summary["R2_in_family_A"] = lock.get("R2_in_family_A")
    if args.resummarize:
        rows = [json.loads(x) for x in open(res_path)]
        old = json.loads((out_dir / "summary.json").read_text())
        old["by_method"] = summarize(rows, n_prob)
        old["stream_ratios"] = stream_ratios(rows, old["by_method"])
        (out_dir / "summary.json").write_text(json.dumps(old, indent=1, default=float))
        return
    log(f"start task={TASK} mode={args.mode} device={dev} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) x {n_prob} "
        f"problems x streams {streams}, f*={fstar}, lock={lock.get('status')}, generator_hash={generator_hash()}")
    try:
        progress(1, 4, "gpu: J tables + eta_loc")
        meta = precompute(seeds, n_prob if streams == [0] else 15, fstar, dev, log)
        summary["eta_loc_min_over_eps"] = {"median": float(np.median(meta["eta_min_over_eps"])),
                                           "min": float(np.min(meta["eta_min_over_eps"])),
                                           "floor_rule": "refuse iff 2 min eta_loc > eps, i.e. min/eps > 0.5"}
        summary["table_sec"] = meta["sec"]
        summary["gpu_peak_mb"] = meta.get("gpu_peak_mb")
        log(f"min_g eta_loc/eps: median {summary['eta_loc_min_over_eps']['median']:.2f}, "
            f"min {summary['eta_loc_min_over_eps']['min']:.2f} (floor refuse if > 0.5); gpu peak {meta.get('gpu_peak_mb')}")
        done_units = set()
        if res_path.exists():
            prev = [json.loads(x) for x in open(res_path)]
            cnt = {}
            for r in prev:
                cnt[(r["instance"], r["stream"], r["method"])] = cnt.get((r["instance"], r["stream"], r["method"]), 0) + 1
            done_units = {u for u, c in cnt.items() if c >= n_prob}
            log(f"resume: {len(done_units)} finished (instance, stream, method) units in results.jsonl")
        jobs = []
        for s in seeds:
            for st in streams:
                for g, ms in GROUPS:
                    ms = [m for m in ms if not (m == "JPC" and st not in diag_streams)
                          and not (m == "B3-UIc" and st not in uic_streams)]
                    todo = [m for m in ms if (s, st, m) not in done_units]
                    if todo:
                        jobs.append((s, st, g, todo))
        order = {"B3-UI": 0, "B3-UIc": 1, "grid": 2, "B2": 3, "B12_infl": 4, "B3": 5}
        jobs.sort(key=lambda j: order[j[2]])
        progress(2, 4, "cpu: methods", {"jobs": len(jobs)})
        from joblib import Parallel, delayed
        errors, samples, unit_sec = [], [], []
        n_done = 0
        gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_unit)(s, st, g, ms, n_prob, meta, fstar, s == seeds[0] and st == 0, args.ui_check_frac)
            for (s, st, g, ms) in jobs)
        with open(res_path, "a") as fh:
            for rr, ss, ee, us in gen:
                bad = {(e["instance"], e["stream"], e["method"]) for e in ee}
                for r in rr:
                    if (r["instance"], r["stream"], r["method"]) not in bad:
                        fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                errors += ee
                samples += ss
                unit_sec.append(us)
                n_done += 1
                for e in ee:
                    log(f"ERROR {e['instance']} s{e['stream']} {e['method']}: {e['error']}\n{e['tb']}")
                log(f"unit {n_done}/{len(jobs)}: inst {us['instance']} stream {us['stream']} {us['group']} "
                    f"{us['sec']:.1f}s")
                progress(2, 4, "cpu: methods", {"jobs_done": n_done, "jobs": len(jobs)})
        rows = [json.loads(x) for x in open(res_path)]
        (out_dir / "samples" / "samples.json").write_text(json.dumps(samples, indent=1, default=float))
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        progress(3, 4, "summary")
        summ = summarize(rows, n_prob)
        ratios = stream_ratios(rows, summ)
        summary.update({"by_method": summ, "stream_ratios": ratios, "crashes": len(errors), "unit_sec": unit_sec,
                        "n_rows": len(rows)})
        # evidence-boundary assertions (also asserted inline; here re-checked on the written rows)
        eb = [r for r in rows if "env_n_steps" in r]
        eb_ok = all(r["env_n_steps"] == r["lr_n_rounds"] for r in eb)
        summary["evidence_boundary"] = {"rows_checked": len(eb), "env_n_steps_eq_lr_n_rounds": eb_ok,
                                        "dual_calls": int(sum(r.get("dual_calls", 0) for r in rows)),
                                        "H0_note": "exact enumeration on |Theta| <= 64000 throughout; no dual/BnB "
                                                   "call -> UB >= exact sup holds trivially (UB == exact)"}
        # timing projection: wall-clock per (problem, method) -> full chunk = n_full_inst x 3 streams x 15 problems
        per_unit = {}
        for us in unit_sec:
            per_unit.setdefault(us["group"], []).append(us["sec"])
        # full = n_full_inst instances x 15 problems x (#streams the group runs in full); the grid group's time is ~all JPC
        # diagnostic (inflated methods refuse at once), so it scales with --diag-streams
        n_full_streams = {"grid": len(diag_streams) or 0.05, "B3-UIc": len(uic_streams)}
        grp_full = {g: float(np.mean(v)) / n_prob * 15 * n_full_inst * n_full_streams.get(g, 3) for g, v in per_unit.items()}
        full_cpu = sum(grp_full.values())
        summary["timing_projection"] = {
            "unit_sec_mean_by_group": {g: float(np.mean(v)) for g, v in per_unit.items()},
            "projected_full_cpu_sec_by_group": grp_full, "projected_full_cpu_sec": full_cpu,
            "projected_full_wall_min_at_4_workers": full_cpu / 4 / 60,
            "note": "linear in problems/stream (15 vs pilot), concurrent run on 20 cores (4 parallel tasks); "
                    "J tables / eta_loc for the eval instances add GPU time (see table_sec x 480 problems)"}
        tj = (meta["sec"].get(fkey(fstar), {}).get("jtable_new_mean") or 0) + \
             (meta["sec"].get(fkey(fstar), {}).get("eta_new_mean") or 0) + \
             (meta["sec"].get("1", {}).get("jtable_new_mean") or 0)
        summary["timing_projection"]["projected_full_gpu_table_min"] = tj * n_full_inst * 15 / 60
        proj = full_cpu / 4 / 60 + tj * n_full_inst * 15 / 60
        summary["timing_projection"]["projected_full_total_min"] = proj
        summary["timing_projection"]["n_full_instances"] = n_full_inst
        pc = {"zero_crashes": len(errors) == 0, "evidence_boundary_assertions_hold": eb_ok,
              "projected_full_wall_le_60_min": proj <= 60,
              "all_methods_present": all(m in summ for _, ms in GROUPS for m in ms)}
        summary["pass_criteria"] = pc
        summary["run_options"] = {"ui_check_frac": args.ui_check_frac, "diag_streams_full": sorted(diag_streams),
                                  "uic_streams_full": sorted(uic_streams), "tag": args.tag}
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for m, v in summ.items():
            log(f"{m:9s} completion {v['completion']:.2f} {v['status_counts']} refused {v['refused_floor']:2d} "
                f"stream steps {v['stream_steps_charged_mean']:.0f} FCR {v['false_certs']}/{v['n']} "
                f"CPup {v['fcr_cp_upper']:.3f} theta-cell {v['theta_cell_survival']} zero-cost "
                f"{v['zero_cost_rate']:.2f} wall/problem {v['wall_clock_per_problem_s']:.2f}s")
        for k_, v in ratios.items():
            if isinstance(v, dict):
                log(f"ratio JPC_infl {k_}: {v['ratio_geo_mean']:.3f} CI {[round(x, 3) for x in v['ci95_chunk_bootstrap2000']]}"
                    f" tie {v['tie_rate']}")
        log(f"timing projection full: {proj:.1f} min ({grp_full})")
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}; projected full {proj:.1f} min")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
