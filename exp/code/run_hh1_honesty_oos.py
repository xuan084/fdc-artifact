"""hh1_honesty_oos: HH1a (no wrong labels) / HH1b (decidability within budget, non-near-tie OOS detection).

Usage: run_hh1_honesty_oos.py --mode {pilot,full} [--workers 4]

Blocks
  E1-Lin OOS family (new, dsswm/streams/lin_oos.py; harness keeps only truly OOS, non-near-tie problems V* >= 1.2 eps):
     squeeze : psi2 prior box [-0.8, 0.8], level-2 probes illegal (psi2 in R^perp, coordinate aligned)
     newpart : E1-Lin 3x4, right participant 3 never probed; ranking depends on beta_3 (NOT coordinate aligned)
  E1-Lin standard Type-2 (round-0 generator ptype 2) x 2 legality regimes {probes_legal, probes_illegal}
  E1-NL-S Type-2 T2a-T2e (round-0 construction, extended finite class |Theta|=41472; finite-class Prop. 3 truth label)

JPC-Lin (learner side; never reads theta*):
  evidence   Abbasi-Yadkori ellipsoid E (theoretical radius, S = ||max(|lo|,|hi|)|| of the public prior box)
  certify    UB(a,b) = min(sup_E <d,th>, sup_E <P_R d,th> + sup_box <d_perp,th>);  CERTIFIED iff min_a max_b UB <= eps
  OOS        sound fibre lower bound (handles non-aligned R^perp, e.g. the alpha/beta gauge + beta_new):
               LB(a,b) = inf_E <P_R d,th> + max{ <d, u> : u in R^perp, lo + m <= P_R th_hat + u <= hi - m }
             with m_k = sqrt(beta) * sqrt((P_R V^-1 P_R)_kk) (coordinate half-width of P_R th over E).  For any th* in
             E (prob >= 1 - delta) the point P_R th* + u lies in box and in th*'s identified fibre, so LB <= W*(a,b).
             OUT_OF_SCOPE iff min_a max_b LB > eps.  (When the out-of-range part is coordinate aligned, the round-0
             box bound is also valid and the max of both is used.)
  probes     phased transductive design (round-0 h1 style): at doubling phase boundaries a Frank-Wolfe min-max design
             over the LEGAL atoms (i, j, n, b) for the undecided identified directions, tracked by the largest-deficit
             legal action at the current loads on 50% of rounds, uniform legal action on the other 50% (pure tracking
             stalls on load-state reachability on dev seeds; see summary lin_chooser_note).
JPC-NL: LR set + exact minimax certificate + finite-class trichotomy + KL-DDA (round-0 jpc_nl_loop, unchanged).
Baselines (zero cost): B4 point estimate (claims certainty), B9-20 bootstrap-ensemble agreement.
Wrong label = false certificate (CERTIFIED with true regret > eps) OR OUT_OF_SCOPE on a truly NEED_DATA problem.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
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
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables  # noqa: E402
from dsswm.certify.lin_closed import range_projector  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, group_ids, observable_signature, regret_matrix  # noqa: E402
from dsswm.core.actions import ActionSpace  # noqa: E402
from dsswm.envs.nl import NLEnv  # noqa: E402
from dsswm.evidence.ellipsoid import EllipsoidSet  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import LIN_DEFAULTS, NL_DEFAULTS, generator_hash, make_lin_instance, \
    make_nl_instance, nl_class_max_jtable  # noqa: E402
from dsswm.streams.lin_oos import (LIN_OOS_EPS, NEAR_TIE_FRAC, LinLegality, family_manifest,  # noqa: E402
                                   make_oos_instance, prior_box, truth_vstar)
from run_honesty_trichotomy import (T2_GRID, b9_nl, jpc_nl_loop, nl_t2_all_problems,  # noqa: E402
                                    nl_t2_jtables)

TASK = "hh1_honesty_oos"
DELTA = 0.05
EPS_LIN = LIN_OOS_EPS          # 0.05 (round-0 lock for E1-Lin)
EPS_NL = 0.02
TMAX_LIN = int(os.environ.get("HH1_TMAX_LIN", 500000))
TMAX_LIN_ROUND0 = 150000      # round-0 budget, decidability also reported at this cap
TMAX_NL = 3000
CERT_CHECK_FRAC = 0.01         # certificate re-check every max(1, 1% of steps)
OOS_CHECK_FRAC = 0.05          # OOS LP re-check every max(1, 5% of steps)
EXPLORE = 0.05
PHASE0 = 64
CHOOSER = os.environ.get("HH1_CHOOSER", "mix")   # track | greedy | mix (50% FW-tracking + 50% uniform legal) | uniform
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
JT_CACHE = WS / "exp" / "cache" / "jtables"
RES_ROOT = WS / "exp" / "results"
PRE = json.loads((WS / "plan" / "prereg_lock.json").read_text())
# v1 lock: top-level "E1-Lin"; v2 lock: under "round0"; v3 lock carries neither -> the frozen round-0 value
# (dsswm.baselines.lin_rage.S_LIN, identical number)
S_LOCK = float((PRE.get("E1-Lin") or (PRE.get("round0") or {}).get("E1-Lin")
                or {"ellipsoid_S": 2.8071337695236402})["ellipsoid_S"])
PROTOCOL = True   # round-3 wrapper passes --no-protocol: it owns the PID/PROGRESS/DONE/gpu_progress files
STATUSES = ("CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE", "MODEL_CONFLICT", "COMPUTE_UNKNOWN")


# ------------------------------------------------------------------ scheduler protocol
def progress(step, total, phase, metric=None):
    if not PROTOCOL:
        return
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    if not PROTOCOL:
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
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ================================================================== E1-Lin learner geometry
class Geom:
    """Public geometry of a legality regime: range projector of the legal probe library, legal atoms/actions."""

    def __init__(self, lc: LinClass, aspace: ActionSpace, leg: LinLegality):
        L, R, N, nb = lc.L, lc.R, lc.nmax + 1, lc.nb
        self.L, self.R, self.N, self.nb, self.lc = L, R, N, nb, lc
        self.Ftab = np.zeros((L * R * N * nb, lc.d))
        self.atom_legal = np.zeros(L * R * N * nb, bool)
        for i in range(L):
            for j in range(R):
                for n in range(N):
                    for b in range(nb):
                        k = self.fidx(i, j, n, b)
                        self.Ftab[k] = lc.feature(i, j, n, b)
                        self.atom_legal[k] = (b <= leg.maxlev and i not in leg.excluded_left
                                              and j not in leg.excluded_right)
        self.PR = range_projector(leg.library(lc))
        U, s, _ = np.linalg.svd(self.PR)
        self.NR = U[:, s > 0.5]
        self.Nperp = U[:, s <= 0.5]
        self.perp_coord = np.linalg.norm(self.PR, axis=0) < 1e-9
        self.legal = leg.legal_actions(aspace)
        P = min(L, R)
        n = aspace.n
        self.ai = np.zeros((n, P), np.int64)
        self.aj = np.zeros((n, P), np.int64)
        self.ab = np.zeros((n, P), np.int64)
        self.am = np.zeros((n, P), bool)
        for a_idx, a in enumerate(aspace.actions):
            inc = {(i, j): l for i, j, l in a.incentives}
            for s_, (i, j) in enumerate(a.pairs):
                self.ai[a_idx, s_], self.aj[a_idx, s_], self.ab[a_idx, s_] = i, j, inc.get((i, j), 0)
                self.am[a_idx, s_] = True

    def fidx(self, i, j, n, b):
        return ((i * self.R + j) * self.N + n) * self.nb + b

    def action_atoms(self, loads):
        n_i = np.asarray(loads)[self.ai]
        return ((self.ai * self.R + self.aj) * self.N + n_i) * self.nb + self.ab


def cert_bounds(ell, Z, geom, lo, hi, eps):
    th = ell.theta_hat()
    Vi = np.linalg.inv(ell.V)
    sb = ell.sqrt_beta()
    D = Z[None, :, :] - Z[:, None, :]                       # D[a, b] = z_b - z_a
    G = D @ geom.PR
    Dp = D - G
    aligned = bool(np.all(np.abs(Dp[..., ~geom.perp_coord]) < 1e-9))
    box_sup = np.maximum(Dp * lo, Dp * hi).sum(-1)
    wD = np.sqrt(np.maximum(np.einsum("abd,de,abe->ab", D, Vi, D), 0))
    wG = np.sqrt(np.maximum(np.einsum("abd,de,abe->ab", G, Vi, G), 0))
    UB = np.minimum(D @ th + sb * wD, G @ th + sb * wG + box_sup)
    LBcheap = G @ th - sb * wG + box_sup                    # upper bound on the sound LP lower bound
    K = len(Z)
    UB[np.arange(K), np.arange(K)] = 0.0
    LBcheap[np.arange(K), np.arange(K)] = -np.inf
    Rbar = UB.max(1)
    pi_c = int(np.argmin(Rbar))
    return {"pi": pi_c, "r_bar": float(max(Rbar[pi_c], 0.0)), "certified": bool(Rbar[pi_c] <= eps), "UB": UB,
            "LBcheap": LBcheap, "G": G, "D": D, "wG": wG, "Vi": Vi, "th": th, "sb": sb, "aligned": aligned}


def oos_check(res, geom, lo, hi, eps, full=False):
    """Sound OOS test. Returns (oos, value, n_lp). value = min_a max_b LB when full=True (no early exit),
    otherwise the bound of the first candidate that fails (diagnostic only)."""
    th, sb, Vi, G, D, wG = res["th"], res["sb"], res["Vi"], res["G"], res["D"], res["wG"]
    PR, Np = geom.PR, geom.Nperp
    c = PR @ th
    m = sb * np.sqrt(np.maximum(np.diag(PR @ Vi @ PR), 0.0))
    # rows where R^perp has no component: u_k = 0, so (P_R th* + u)_k = th*_k lies in the box automatically
    act = np.abs(Np).sum(1) > 1e-9
    ub_c = (hi - m - c)[act]
    lb_c = (lo + m - c)[act]
    A_ub = np.vstack([Np[act], -Np[act]])
    b_ub = np.concatenate([ub_c, -lb_c])
    inf_id = G @ th - sb * wG                               # inf_E <P_R d, th>
    K = D.shape[0]
    LBc = res["LBcheap"]
    n_lp, worst = 0, []
    for a in range(K):
        if LBc[a].max() <= eps and not full:
            return False, float(LBc[a].max()), n_lp
        best = -np.inf
        for b in np.argsort(-LBc[a]):
            if b == a or (LBc[a, b] <= eps and not full) or LBc[a, b] == -np.inf:
                break
            if full and LBc[a, b] <= best:
                break
            dperp = D[a, b] - G[a, b]
            if Np.shape[1] == 0 or not act.any():
                val = inf_id[a, b]
            else:
                r = linprog(-(Np.T @ dperp), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * Np.shape[1],
                            method="highs")
                n_lp += 1
                val = inf_id[a, b] + (-r.fun) if r.status == 0 else -np.inf
            if res["aligned"]:                              # round-0 box bound is also sound when aligned
                val = max(val, float(LBc[a, b]))
            if val > best:
                best = val
            if best > eps and not full:
                break
        worst.append(best)
        if best <= eps and not full:
            return False, float(best), n_lp
    return bool(min(worst) > eps), float(min(worst)), n_lp


def fw_design(T, V, n_phase, geom, iters=150):
    """Frank-Wolfe min-max transductive design over legal atoms: min_l max_t ||g_t||^2_{(V + n A(l))^-1}."""
    F = geom.Ftab[geom.atom_legal]
    k = len(F)
    lam = np.full(k, 1.0 / k)
    for it in range(iters):
        A = V + n_phase * (F.T * lam) @ F
        Ai = np.linalg.inv(A)
        AiT = Ai @ T.T                                       # (d, m)
        vals = (T * AiT.T).sum(1)
        t = int(np.argmax(vals))
        grad = -n_phase * (F @ AiT[:, t]) ** 2               # d/dl_k of g_t^T A^-1 g_t
        j = int(np.argmin(grad))
        g = 2.0 / (it + 3.0)
        lam *= (1 - g)
        lam[j] += g
    full = np.zeros(len(geom.Ftab))
    full[geom.atom_legal] = lam
    return full


def greedy_atom_scores(T, res, geom):
    """Round-0 style greedy information score per atom (relative width reduction over undecided directions)."""
    Vi = res["Vi"]
    W = np.sqrt(np.maximum(np.einsum("md,de,me->m", T, Vi, T), 1e-30))
    FV = geom.Ftab @ Vi
    qd = (FV * geom.Ftab).sum(1)
    proj = FV @ T.T
    sc = ((proj ** 2) / (1.0 + qd[:, None]) / (W[None] ** 2)).sum(1)
    return np.where(geom.atom_legal, sc, 0.0)


def target_dirs(res, eps):
    UB, LBc, G, wG = res["UB"], res["LBcheap"], res["G"], res["wG"]
    pi = res["pi"]
    tg = [(pi, b) for b in np.flatnonzero(UB[pi] > eps)]
    for a in range(UB.shape[0]):
        if a == pi:
            continue
        b = int(np.argmax(LBc[a]))
        if LBc[a, b] > -np.inf:
            tg.append((a, b))
    T = [G[a, b] for a, b in tg if wG[a, b] > 1e-10]
    return np.array(T) if T else None


def lin_build(spec):
    if spec["block"] == "OOS_family":
        inst, leg, prior = make_oos_instance(spec["family"], spec["seed"], noise_seed=spec["noise"], K=10)
    else:
        inst = make_lin_instance(spec["seed"], noise_seed=spec["noise"], ptypes=(2,), K=5)
        leg = LinLegality(maxlev=2 if spec["regime"] == "probes_legal" else 1)
        prior = dict(LIN_DEFAULTS["prior"])
    return inst, leg, prior


def lin_case(spec, samples_wanted=False):
    torch.set_num_threads(1)
    t_job = time.time()
    eps = EPS_LIN
    inst, leg, prior = lin_build(spec)
    env = inst.env
    lc = LinClass(env.L, env.R, env.aspace.nb, nmax=env.nmax)
    lo, hi = prior_box(lc, prior)
    S = float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))
    q = inst.problems[spec["k"]]
    geom = Geom(lc, env.aspace, leg)
    Z = np.stack([lc.z(p, q.loads0, q.H, q.utility, env.aspace) for p in q.policies])
    # ---- harness truth (theta* used ONLY here and in evaluation fields)
    theta = env.true_theta_vector()
    assert np.all(theta >= lo - 1e-12) and np.all(theta <= hi + 1e-12)
    tl = truth_vstar(Z, theta, geom.PR, lo, hi, eps)
    Jt = Z @ theta
    X0 = np.concatenate([lc.obs_rows(o)[0] for o in inst.init_obs], 0)
    PD = range_projector(X0)
    Dall = (Z[None] - Z[:, None]).reshape(-1, lc.d)
    base = {"env": "E1-Lin", "block": spec["block"], "family": spec.get("family"), "instance": spec["seed"],
            "noise_seed": spec["noise"], "problem_index": spec["k"], "pid": q.pid, "regime": spec["regime"],
            "variant": f"Lin_{spec.get('family') or 'std'}_{spec['regime']}", "ptype": 2,
            "true_label": tl["label"], "V_star": tl["V_star"], "near_tie": tl["near_tie"], "eps": eps,
            "delta": DELTA, "type2_proper": bool(np.abs(Dall - Dall @ PD).max() > 1e-9),
            "d_out_of_legal_range_max": float(np.abs(Dall - Dall @ geom.PR).max()), "n_policies": len(q.policies),
            "H": q.H, "S_box": S, "legality": leg.to_dict()}
    rows = []
    # ---- JPC adaptive (fresh platform copy, same seed -> same truth, init data and noise stream)
    inst2, _, _ = lin_build(spec)
    h = inst2.env.handle()
    n0 = h.n_steps
    ell = EllipsoidSet(lc, sigma=h.known_constants()["sigma"], delta=DELTA, S=S, lam=1.0)
    for o in inst2.init_obs:
        ell.update(o)
    rng = np.random.default_rng([spec["seed"], spec["noise"], spec["k"], 13 if spec["regime"] == "probes_legal" else 14,
                                 0 if spec["block"] == "standard" else 1])
    t0 = time.perf_counter()
    steps, status, start_status, start_pi = 0, None, None, None
    next_cert, next_oos, n_lp_tot, aligned_all = 0, 0, 0, True
    phase_end, lam_t, cnt, phase_atoms = 0, None, None, 0
    traj, lb_min = [], -np.inf
    illegal, next_score, atom_sc = 0, 0, None
    while True:
        if steps >= next_cert or steps >= phase_end:
            res = cert_bounds(ell, Z, geom, lo, hi, eps)
            aligned_all &= res["aligned"]
        if steps >= next_cert:
            next_cert = steps + max(1, int(CERT_CHECK_FRAC * steps))
            cur = "CERTIFIED" if res["certified"] else "NEED_DATA"
            if cur == "NEED_DATA" and steps >= next_oos:
                next_oos = steps + max(1, int(OOS_CHECK_FRAC * steps))
                oos, lb_min, nlp = oos_check(res, geom, lo, hi, eps)
                n_lp_tot += nlp
                if oos:
                    cur = "OUT_OF_SCOPE"
            if start_status is None:
                start_status, start_pi = cur, res["pi"]
            if cur != "NEED_DATA":
                status = cur
                break
            if steps >= TMAX_LIN:
                status = "NEED_DATA"
                break
            if len(traj) < 40 and (steps <= 5 or steps in (100, 1000, 5000, 20000, 50000, 100000)
                                   or steps % 10000 == 0):
                traj.append({"step": steps, "r_bar": res["r_bar"], "oos_lb_worst_min": lb_min, "pi": res["pi"]})
        if steps >= phase_end:                              # new design phase (doubling)
            n_phase = max(PHASE0, steps)
            phase_end = steps + n_phase
            T = target_dirs(res, eps)
            lam_t = fw_design(T, ell.V, n_phase, geom) if T is not None else None
            cnt = np.zeros(len(geom.Ftab))
            phase_atoms = 0
        loads = h.observable_state()[0]
        if CHOOSER == "greedy" and steps >= next_score:
            T = target_dirs(res, eps)
            atom_sc = greedy_atom_scores(T, res, geom) if T is not None else None
            next_score = steps + max(1, int(CERT_CHECK_FRAC * steps))
        if lam_t is None or rng.random() < EXPLORE or CHOOSER == "uniform" or (
                CHOOSER == "mix" and rng.random() < 0.5) or (CHOOSER == "greedy" and atom_sc is None):
            a = int(rng.choice(geom.legal))
        elif CHOOSER == "greedy":
            at = geom.action_atoms(loads)
            sc = np.where(geom.am, atom_sc[at], 0.0).sum(1)
            a = int(geom.legal[int(np.argmax(sc[geom.legal] + 1e-12 * rng.random(len(geom.legal))))])
        else:
            at = geom.action_atoms(loads)                   # (n_actions, P)
            tgt = lam_t * (phase_atoms + 3)
            sc = np.where(geom.am, (tgt - cnt)[at], 0.0).sum(1)
            a = int(geom.legal[int(np.argmax(sc[geom.legal] + 1e-9 * rng.random(len(geom.legal))))])
        if not leg.is_legal(h.aspace, a):
            illegal += 1
        obs = h.step(a)
        ell.update(obs)
        if cnt is not None:
            at_a = geom.action_atoms(loads)[a][geom.am[a]]
            np.add.at(cnt, at_a, 1)
            phase_atoms += len(at_a)
        steps += 1
    if status != "OUT_OF_SCOPE":
        _, lb_min, _ = oos_check(res, geom, lo, hi, eps, full=True)   # diagnostic: sound min_a max_b LB at stop
    assert h.n_steps - n0 == steps, "step accounting mismatch"
    assert ell.n_rounds == len(inst2.init_obs) + steps, "evidence must contain exactly the real rounds"
    assert illegal == 0, "illegal probe"
    pi = res["pi"]
    reg = float(Jt.max() - Jt[pi])
    wrong = bool((status == "CERTIFIED" and reg > eps) or (status == "OUT_OF_SCOPE" and tl["label"] != "OUT_OF_SCOPE"))
    rows.append({**base, "method": "JPC", "protocol": "adaptive", "status": status, "status_at_start": start_status,
                 "new_env_steps": steps, "censored": status == "NEED_DATA", "certified_policy": pi,
                 "true_regret": reg, "false_cert": bool(status == "CERTIFIED" and reg > eps),
                 "wrong_oos": bool(status == "OUT_OF_SCOPE" and tl["label"] != "OUT_OF_SCOPE"), "wrong_label": wrong,
                 "start_wrong_label": bool(start_status == "OUT_OF_SCOPE" and tl["label"] != "OUT_OF_SCOPE") or bool(
                     start_status == "CERTIFIED" and float(Jt.max() - Jt[start_pi]) > eps),
                 "theta_star_in_set": bool(ell.contains(theta)), "aligned": aligned_all, "n_lp": n_lp_tot,
                 "oos_lb_worst_min_end": lb_min, "r_bar_end": res["r_bar"], "wall_clock_s": time.perf_counter() - t0})
    samples = []
    if samples_wanted:
        samples.append({"env": "E1-Lin", "block": spec["block"], "family": spec.get("family"), "pid": q.pid,
                        "regime": spec["regime"], "policies": [p.name for p in q.policies], "J_true": Jt.tolist(),
                        "true_label": tl["label"], "V_star": tl["V_star"], "start_status": start_status,
                        "final_status": status, "steps": steps, "certified_policy": pi, "trajectory_head": traj})
    # ---- zero-cost baselines (independent of the regime; recorded once per (problem, regime) for simplicity)
    X = np.concatenate([lc.obs_rows(o)[0] for o in inst.init_obs])
    y = np.concatenate([lc.obs_rows(o)[1] for o in inst.init_obs])
    th_r = np.linalg.solve(np.eye(lc.d) + X.T @ X, X.T @ y)
    pi4 = int(np.argmax(Z @ th_r))
    reg4 = float(Jt.max() - Jt[pi4])
    rows.append({**base, "method": "B4", "protocol": "zero_cost", "status": "CERTIFIED", "new_env_steps": 0,
                 "certified_policy": pi4, "true_regret": reg4, "false_cert": bool(reg4 > eps), "wrong_oos": False,
                 "wrong_label": bool(reg4 > eps), "claims_on_oos": tl["label"] == "OUT_OF_SCOPE"})
    brng = np.random.default_rng([spec["seed"], spec["noise"], spec["k"], 99])
    picks = []
    for _ in range(20):
        w = brng.poisson(1.0, len(y)).astype(float)
        thb = np.linalg.solve(np.eye(lc.d) + (X * w[:, None]).T @ X, (X * w[:, None]).T @ y)
        picks.append(int(np.argmax(Z @ thb)))
    vals, cntv = np.unique(picks, return_counts=True)
    maj = int(vals[np.argmax(cntv)])
    st9 = "CERTIFIED" if len(vals) == 1 else "NEED_DATA"
    reg9 = float(Jt.max() - Jt[maj])
    rows.append({**base, "method": "B9-20", "protocol": "zero_cost", "status": st9, "new_env_steps": 0,
                 "certified_policy": maj, "true_regret": reg9, "false_cert": bool(st9 == "CERTIFIED" and reg9 > eps),
                 "wrong_oos": False, "wrong_label": bool(st9 == "CERTIFIED" and reg9 > eps),
                 "claims_on_oos": bool(st9 == "CERTIFIED" and tl["label"] == "OUT_OF_SCOPE"),
                 "agree_frac": float(cntv.max() / 20)})
    return {"kind": "lin_" + spec["block"], "seed": spec["seed"], "rows": rows, "samples": samples,
            "wall_s": time.time() - t_job}


# ================================================================== E1-NL-S Type-2 (round-0 construction)
def nl_t2_job(seed, noise, eps, tmax, samples_wanted):
    torch.set_num_threads(1)
    t_job = time.time()
    ncl2 = NLClass(T2_GRID, device="cpu")
    params2 = ncl2.torch_params()
    aspace2 = ActionSpace(2, 2, 1, (1, 2))
    prop2 = NLPropagator(2, 2, NMAX, aspace2, C_KNOWN, RHO_RET, device="cpu")
    LT2 = prop2.tables(params2)[0]
    LT2_np = LT2.numpy()
    py2, pe2 = class_prob_tables(ncl2.np_params, C_KNOWN, NMAX, 3)
    inc2 = IncidenceIndex(prop2.codec, aspace2, NMAX)
    gids = {lv: group_ids(observable_signature(py2, pe2, max_level=lv)) for lv in (1, 2)}
    base = make_nl_instance(seed, noise_seed=noise, kappa_mode="high", K=1)
    tr = base.truth
    psi2_true = float(np.random.default_rng([seed, 32]).choice(T2_GRID.psi2))
    ti2 = ncl2.index_of(tuple(tr["alpha"]), tuple(tr["beta"]), tuple(tr["gamma"]), tuple(tr["tauL"]), tuple(tr["tauR"]),
                        tr["psi"], tr["lam"], psi2_true)
    init_obs = base.init_obs
    lr0 = SeqLRSet(prop2, LT2, DELTA)
    for o in init_obs:
        lr0.update(o)
    rows, samples = [], []

    def new_env(salt):
        e = NLEnv(np.array(tr["alpha"]), np.array(tr["beta"]), np.array(tr["gamma"]), np.array(tr["tauL"]),
                  np.array(tr["tauR"]), [tr["psi"], psi2_true], tr["lam"], c=C_KNOWN, rho_ret=RHO_RET, L=2, R=2,
                  nmax=NMAX, budget=1, incentive_levels=(1, 2), seed=seed * 1000 + noise + salt)
        e.loads, e.engaged = base.env.loads.copy(), base.env.engaged.copy()
        return e

    for vi, (vname, q, maxlev) in enumerate(nl_t2_all_problems(seed, aspace2)):
        J = nl_class_max_jtable(q, prop2, params2, cache_path=str(JT_CACHE / f"E1-NL-S-T2_{q.pid}_raw.npy"))
        Reg = regret_matrix(J)
        Jt = J[ti2]
        legal = np.array([a for a in range(aspace2.n) if aspace2.max_level(a) <= maxlev])
        gid = gids[maxlev]
        mask0 = lr0.mask().numpy()
        members = (gid == gid[ti2]) & mask0
        vstar = float(Reg[members].max(0).min()) if members.any() else float("inf")
        label = "OUT_OF_SCOPE" if vstar > eps else "NEED_DATA"
        members_l1 = (gids[1] == gids[1][ti2])
        common = {"env": "E1-NL-S", "block": "NL_T2", "family": None, "ptype": 2, "instance": seed,
                  "noise_seed": noise, "problem_index": vi, "pid": q.pid, "variant": vname,
                  "regime": "probes_legal" if maxlev == 2 else "probes_illegal", "true_label": label,
                  "V_star": vstar, "near_tie": bool(abs(vstar - eps) < NEAR_TIE_FRAC * eps), "eps": eps,
                  "delta": DELTA, "psi2_true": psi2_true,
                  "type2_proper": bool(Reg[members_l1].max(0).min() > eps), "n_policies": len(q.policies),
                  "H": q.H, "truth_class_size": int(members.sum())}
        lr = SeqLRSet(prop2, LT2, DELTA)
        lr.cum, lr.log_num, lr.n_rounds = lr0.cum.clone(), lr0.log_num, lr0.n_rounds
        env2 = new_env(7 + vi * 10 + maxlev)
        h2 = env2.handle()
        rng = np.random.default_rng([seed, noise, 909, vi])
        traj = [] if samples_wanted and vi in (1, 3) else None
        t0 = time.perf_counter()
        status, steps, start, cert = jpc_nl_loop(Reg, lr, h2, prop2, py2, pe2, LT2_np, inc2, legal, gid, eps, tmax,
                                                 rng, traj=traj)
        assert lr.n_rounds == len(init_obs) + steps and env2.n_steps == steps, "step accounting mismatch"
        pi = cert["pi"]
        reg = float(Jt.max() - Jt[pi]) if pi is not None else None
        fc = bool(status == "CERTIFIED" and reg > eps)
        woos = bool(status == "OUT_OF_SCOPE" and label != "OUT_OF_SCOPE")
        sreg = float(Jt.max() - Jt[start[1]]) if start[1] is not None else None
        rows.append({**common, "method": "JPC", "protocol": "adaptive", "status": status, "status_at_start": start[0],
                     "new_env_steps": steps, "censored": status == "NEED_DATA", "certified_policy": pi,
                     "true_regret": reg, "false_cert": fc, "wrong_oos": woos, "wrong_label": fc or woos,
                     "start_wrong_label": bool((start[0] == "OUT_OF_SCOPE" and label != "OUT_OF_SCOPE") or
                                               (start[0] == "CERTIFIED" and sreg is not None and sreg > eps)),
                     "theta_star_in_set": bool(lr.mask().numpy()[ti2]), "set_size_end": int(lr.mask().sum()),
                     "wall_clock_s": time.perf_counter() - t0})
        if traj is not None:
            samples.append({"env": "E1-NL-S", "instance": seed, "pid": q.pid, "variant": vname,
                            "policies": [p.name for p in q.policies], "J_true": Jt.tolist(), "true_label": label,
                            "V_star": vstar, "psi2_true": psi2_true, "start_status": start[0], "final_status": status,
                            "steps": steps, "certified_policy": pi, "trajectory_head": traj})
        kh0 = lr0.mle()
        pi4 = int(np.argmax(J[kh0]))
        reg4 = float(Jt.max() - Jt[pi4])
        rows.append({**common, "method": "B4", "protocol": "zero_cost", "status": "CERTIFIED", "new_env_steps": 0,
                     "certified_policy": pi4, "true_regret": reg4, "false_cert": bool(reg4 > eps), "wrong_oos": False,
                     "wrong_label": bool(reg4 > eps), "claims_on_oos": label == "OUT_OF_SCOPE"})
        st9, maj, agree = b9_nl(LT2, prop2, init_obs, J, [seed, noise, 929, vi])
        reg9 = float(Jt.max() - Jt[maj])
        rows.append({**common, "method": "B9-20", "protocol": "zero_cost", "status": st9, "new_env_steps": 0,
                     "certified_policy": maj, "true_regret": reg9, "false_cert": bool(st9 == "CERTIFIED" and reg9 > eps),
                     "wrong_oos": False, "wrong_label": bool(st9 == "CERTIFIED" and reg9 > eps),
                     "claims_on_oos": bool(st9 == "CERTIFIED" and label == "OUT_OF_SCOPE"), "agree_frac": agree})
    return {"kind": "nl_t2", "seed": seed, "rows": rows, "samples": samples, "wall_s": time.time() - t_job}


# ================================================================== case selection
def oos_manifest_seed(family, seed, noise):
    return family_manifest(family, [seed], noise_seed=noise, K=10)


def select_oos_cases(seeds, noise, n_target, n_newpart_max, workers, log):
    jobs = [delayed(oos_manifest_seed)(f, s, noise) for s in seeds for f in ("squeeze", "newpart")]
    man = [r for part in Parallel(n_jobs=workers)(jobs) for r in part]
    kept = [r for r in man if r["keep"]]
    newp = sorted([r for r in kept if r["family"] == "newpart"], key=lambda r: (r["seed"], r["problem_index"]))
    sq = sorted([r for r in kept if r["family"] == "squeeze"], key=lambda r: (r["seed"], r["problem_index"]))
    chosen = newp[:n_newpart_max]
    chosen += sq[:n_target - len(chosen)]
    log(f"OOS manifest: {len(man)} candidates, kept squeeze={len(sq)} newpart={len(newp)}; chosen {len(chosen)}")
    return man, chosen


# ================================================================== analysis
def _cp(k, n):
    lo, hi = clopper_pearson(k, n) if n else (0.0, 1.0)
    return [lo, hi]


def block_stats(rs):
    n = len(rs)
    nw = sum(r["wrong_label"] for r in rs)
    dec = sum(r["status"] != "NEED_DATA" for r in rs)
    oos_t = [r for r in rs if r["true_label"] == "OUT_OF_SCOPE"]
    oos_nt = [r for r in oos_t if not r["near_tie"]]
    nd_t = [r for r in rs if r["true_label"] == "NEED_DATA"]
    return {
        "n": n, "wrong_label": int(nw), "wrong_label_rate": nw / max(n, 1), "wrong_label_cp": _cp(nw, n),
        "false_cert": int(sum(r["false_cert"] for r in rs)), "wrong_oos": int(sum(r["wrong_oos"] for r in rs)),
        "decidable_within_budget": dec / max(n, 1), "decidable_cp": _cp(dec, n),
        "decidable_within_round0_lin_cap_150k": sum(r["status"] != "NEED_DATA" and r["new_env_steps"] <= TMAX_LIN_ROUND0
                                                    for r in rs) / max(n, 1),
        "label_counts": {L: sum(r["true_label"] == L for r in rs) for L in ("NEED_DATA", "OUT_OF_SCOPE")},
        "n_near_tie": int(sum(r["near_tie"] for r in rs)),
        "oos_true_n": len(oos_t), "oos_detected": int(sum(r["status"] == "OUT_OF_SCOPE" for r in oos_t)),
        "oos_true_non_near_tie_n": len(oos_nt),
        "oos_detected_non_near_tie": int(sum(r["status"] == "OUT_OF_SCOPE" for r in oos_nt)),
        "oos_detection_rate_non_near_tie": (sum(r["status"] == "OUT_OF_SCOPE" for r in oos_nt) / len(oos_nt))
        if oos_nt else None,
        "oos_detection_cp_non_near_tie": _cp(sum(r["status"] == "OUT_OF_SCOPE" for r in oos_nt), len(oos_nt)),
        "need_data_certified": int(sum(r["status"] == "CERTIFIED" for r in nd_t)), "need_data_n": len(nd_t),
        "steps_median_decided": float(np.median([r["new_env_steps"] for r in rs if r["status"] != "NEED_DATA"]))
        if dec else None,
        "oos_detection_steps_median": float(np.median([r["new_env_steps"] for r in oos_t
                                                       if r["status"] == "OUT_OF_SCOPE"]))
        if any(r["status"] == "OUT_OF_SCOPE" for r in oos_t) else None,
        "final_status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in STATUSES},
        "start_status_counts": {s: int(sum(r.get("status_at_start") == s for r in rs)) for s in STATUSES},
        "start_wrong_label": int(sum(r.get("start_wrong_label", False) for r in rs)),
        "theta_star_in_set_end": float(np.mean([r["theta_star_in_set"] for r in rs])) if rs else None,
    }


def analyse(rows, out_dir, meta):
    out = {"task_id": TASK, **meta}
    jpc = [r for r in rows if r["method"] == "JPC"]
    blocks = {"all": jpc,
              "Lin_OOS_family": [r for r in jpc if r["block"] == "OOS_family"],
              "Lin_OOS_squeeze": [r for r in jpc if r["family"] == "squeeze"],
              "Lin_OOS_newpart": [r for r in jpc if r["family"] == "newpart"],
              "Lin_standard": [r for r in jpc if r["block"] == "standard"],
              "Lin_standard_probes_legal": [r for r in jpc if r["block"] == "standard" and r["regime"] == "probes_legal"],
              "Lin_standard_probes_illegal": [r for r in jpc if r["block"] == "standard"
                                              and r["regime"] == "probes_illegal"],
              "E1_Lin_all": [r for r in jpc if r["env"] == "E1-Lin"],
              "E1_NL_S": [r for r in jpc if r["env"] == "E1-NL-S"]}
    out["JPC"] = {k: block_stats(v) for k, v in blocks.items()}
    out["JPC_near_tie_subset"] = block_stats([r for r in jpc if r["near_tie"]])
    out["JPC_non_near_tie"] = block_stats([r for r in jpc if not r["near_tie"]])
    out["E1_NL_S_by_variant"] = {v: block_stats([r for r in blocks["E1_NL_S"] if r["variant"] == v])
                                 for v in sorted({r["variant"] for r in blocks["E1_NL_S"]})}
    out["baselines"] = {}
    for m in ("B4", "B9-20"):
        for env in ("E1-Lin", "E1-NL-S"):
            rs = [r for r in rows if r["method"] == m and r["env"] == env]
            if m == "B4" or True:
                pass
            # zero-cost Lin baselines are regime independent: de-duplicate (instance, family, problem)
            seen, uniq = set(), []
            for r in rs:
                key = (r["block"], r.get("family"), r["instance"], r["problem_index"],
                       r["regime"] if env == "E1-NL-S" else None)
                if key not in seen:
                    seen.add(key); uniq.append(r)
            nc = sum(r["status"] == "CERTIFIED" for r in uniq)
            nw = sum(r["wrong_label"] for r in uniq)
            oos = [r for r in rs if r["true_label"] == "OUT_OF_SCOPE"]
            out["baselines"][f"{env}|{m}"] = {
                "n_unique_problems": len(uniq), "claims_certified": nc, "wrong_label": nw,
                "wrong_label_cp": _cp(nw, len(uniq)), "fcr": nw / max(nc, 1), "fcr_cp": _cp(nw, nc),
                "n_true_oos_cases": len(oos), "claims_certainty_on_true_oos": int(sum(r["claims_on_oos"] for r in oos)),
                "oos_detected": 0}
    a = out["JPC"]["all"]
    fam = out["JPC"]["Lin_OOS_family"]
    oos_trig_all = sum(r["status"] == "OUT_OF_SCOPE" and r["true_label"] == "OUT_OF_SCOPE" and not r["near_tie"]
                       for r in jpc)
    gate = {"pilot_ge10_true_non_near_tie_oos_triggered": bool(oos_trig_all >= 10),
            "pilot_zero_wrong_labels": bool(a["wrong_label"] == 0),
            "HH1a_wrong_label_cp_upper_le_delta": bool(a["wrong_label_cp"][1] <= DELTA),
            "HH1b_lin_oos_family_detection_ge_0.9": bool((fam["oos_detection_rate_non_near_tie"] or 0) >= 0.9)}
    out["n_true_non_near_tie_oos_triggered"] = int(oos_trig_all)
    out["gate"] = gate
    out["go_no_go"] = "GO" if gate["pilot_ge10_true_non_near_tie_oos_triggered"] and gate["pilot_zero_wrong_labels"] \
        else "NO_GO"
    # confusion table
    lines = ["env,block,true_label,status_kind,status,count"]
    for bname in ("Lin_OOS_squeeze", "Lin_OOS_newpart", "Lin_standard_probes_legal", "Lin_standard_probes_illegal",
                  "E1_NL_S"):
        rs = blocks[bname]
        for L in ("NEED_DATA", "OUT_OF_SCOPE"):
            for kind in ("status_at_start", "status"):
                for s in STATUSES:
                    c = sum(r["true_label"] == L and r.get(kind) == s for r in rs)
                    if c:
                        lines.append(f"{rs[0]['env']},{bname},{L},{'start' if kind != 'status' else 'final'},{s},{c}")
    (out_dir / "confusion.csv").write_text("\n".join(lines) + "\n")
    out["misclassified_or_undecided"] = [
        {k: r.get(k) for k in ("env", "block", "family", "pid", "variant", "regime", "true_label", "V_star", "near_tie",
                               "status", "status_at_start", "new_env_steps", "true_regret", "oos_lb_worst_min_end",
                               "r_bar_end")}
        for r in jpc if r["wrong_label"] or r["status"] == "NEED_DATA"
        or (r["true_label"] == "OUT_OF_SCOPE" and r["status"] != "OUT_OF_SCOPE")]
    return out


def plot_confusion(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jpc = [r for r in rows if r["method"] == "JPC"]
    cols = ["CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE"]
    panels = []
    for ttl, rs in (("E1-Lin OOS family", [r for r in jpc if r["block"] == "OOS_family"]),
                    ("E1-Lin standard Type-2", [r for r in jpc if r["block"] == "standard"]),
                    ("E1-NL-S Type-2 (T2a-e)", [r for r in jpc if r["env"] == "E1-NL-S"])):
        rl = ["NEED_DATA", "OUT_OF_SCOPE"]
        M = np.array([[sum(r["true_label"] == L and r["status"] == s for r in rs) for s in cols] for L in rl])
        panels.append((ttl, rl, M))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.0))
    for ax, (ttl, rl, M) in zip(axes, panels):
        ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > 0.6 * max(M.max(), 1) else "black", fontsize=9)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["CERT", "NEED", "OOS"], fontsize=8)
        ax.set_yticks(range(len(rl)))
        ax.set_yticklabels(rl, fontsize=8)
        ax.set_title(ttl + ": final", fontsize=8)
        ax.set_xlabel("output status", fontsize=8)
    axes[0].set_ylabel("true label", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def update_gpu_progress(status, start_iso, wall_min, snapshot, planned):
    import fcntl
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


# ================================================================== main
def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out += list(range(a, b + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-update-progress", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # round-3 overrides (r3_replicate_hh1): seed source / output location only; science code unchanged
    ap.add_argument("--seeds", default=None, help="instance seeds for OOS scan, Lin std and NL blocks ('a-b' or 'a,b')")
    ap.add_argument("--std-seeds", default=None, help="override Lin standard Type-2 seeds only")
    ap.add_argument("--noise-seeds", default=None, help="noise seeds, comma list")
    ap.add_argument("--n-oos", type=int, default=None)
    ap.add_argument("--n-newpart", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-protocol", action="store_true",
                    help="do not write hh1 PID/PROGRESS/DONE/gpu_progress (caller owns the scheduler protocol)")
    args = ap.parse_args()
    global PROTOCOL
    PROTOCOL = not args.no_protocol
    pilot = args.mode == "pilot"
    out_dir = Path(args.out_dir) if args.out_dir else RES_ROOT / ("pilots" if pilot else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    if PROTOCOL:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_file = out_dir / "start_time.txt"
    if not start_file.exists():
        start_file.write_text(datetime.now().isoformat())
    start_iso = start_file.read_text().strip()
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    t_all = time.time()
    if pilot:
        oos_seeds = list(range(600, 650))
        std_seeds = list(range(600, 604))
        nl_seeds = list(range(600, 610))
        noises = [42]
        n_oos, n_newpart = 30, 12
    else:
        lock_status = PRE.get("status")
        assert lock_status == "locked", f"prereg lock status={lock_status}; full mode requires locked"
        oos_seeds = list(range(10000, 10048))
        std_seeds = list(range(10000, 10048))
        nl_seeds = list(range(10000, 10048))
        noises = [42, 123, 456]
        n_oos, n_newpart = 10 ** 6, 10 ** 6
    if args.seeds:
        oos_seeds = std_seeds = nl_seeds = parse_seeds(args.seeds)
    if args.std_seeds:
        std_seeds = parse_seeds(args.std_seeds)
    if pilot:
        assert max(oos_seeds + std_seeds + nl_seeds) < 10000, "pilot mode must not touch evaluation seeds"
    if args.noise_seeds:
        noises = [int(x) for x in args.noise_seeds.split(",")]
    if args.n_oos is not None:
        n_oos = args.n_oos
    if args.n_newpart is not None:
        n_newpart = args.n_newpart
    if args.smoke:
        oos_seeds, std_seeds, nl_seeds, n_oos = oos_seeds[:4], std_seeds[:1], nl_seeds[:1], 3
    log(f"start mode={args.mode} oos_scan_seeds={len(oos_seeds)} std_seeds={len(std_seeds)} nl_seeds={len(nl_seeds)} "
        f"noises={noises} workers={args.workers} gen_hash={generator_hash()} S_lock={S_LOCK:.4f}")
    progress(1, 4, "oos_manifest")
    man, chosen = select_oos_cases(oos_seeds, 42, n_oos, n_newpart, args.workers, log)
    with open(out_dir / "oos_manifest_scanned.jsonl", "w") as f:
        for r in man:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("prior_box",)}, default=float) + "\n")
    progress(2, 4, "nl_t2_jtables_gpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        jt_info = nl_t2_jtables(nl_seeds, log)
        gp = torch.cuda.get_device_properties(0)
        ((RES_ROOT / f"{TASK}_gpu_profile.json") if PROTOCOL else (out_dir / "gpu_profile.json")).write_text(json.dumps({
            "gpu_name": gp.name, "vram_total_mb": gp.total_memory / 2 ** 20, "max_batch_size": None,
            "vram_used_mb": jt_info["max_vram_mb"], "utilization_pct": 100 * jt_info["max_vram_mb"] * 2 ** 20 / gp.total_memory,
            "note": "GPU only for exact NL J tables (batch = whole class); Lin/NL loops are CPU (4 workers), "
                    "VRAM probing not applicable (no batch-size knob)"}))
    else:
        jt_info = {"note": "no cuda; J tables computed on CPU lazily"}
    progress(3, 4, "cases")
    specs = []
    for r in chosen:
        for nz in noises:
            specs.append({"block": "OOS_family", "family": r["family"], "seed": r["seed"], "k": r["problem_index"],
                          "noise": nz, "regime": "probes_illegal"})
    for s in std_seeds:
        for k in range(5):
            for reg in ("probes_legal", "probes_illegal"):
                for nz in noises:
                    specs.append({"block": "standard", "seed": s, "k": k, "noise": nz, "regime": reg})
    jobs = []
    for i, sp in enumerate(specs):
        jobs.append(delayed(lin_case)(sp, i % 9 == 0))
    for s in nl_seeds:
        for nz in noises:
            jobs.append(delayed(nl_t2_job)(s, nz, EPS_NL, TMAX_NL, s == nl_seeds[0] and nz == 42))

    def safe(job):
        f, a, kw = job
        try:
            return f(*a, **kw)
        except Exception as e:  # noqa: BLE001
            return {"kind": f.__name__, "seed": a[0] if not isinstance(a[0], dict) else a[0]["seed"], "rows": [],
                    "samples": [], "error": repr(e), "tb": traceback.format_exc(), "spec": a[0], "wall_s": None}
    log(f"running {len(specs)} Lin cases + {len(nl_seeds) * len(noises)} NL jobs (x5 variants)")
    # NL jobs first (longest), then Lin
    jobs = jobs[len(specs):] + jobs[:len(specs)]
    results = Parallel(n_jobs=args.workers, verbose=0)(delayed(safe)(j) for j in jobs)
    rows, samples, errors, walls = [], [], [], {}
    for r in results:
        rows += r["rows"]
        samples += r["samples"]
        if r.get("error"):
            errors.append({k: r.get(k) for k in ("kind", "seed", "error", "tb", "spec")})
            log(f"ERROR {r['kind']} seed={r['seed']}: {r['error']}")
        walls.setdefault(r["kind"], []).append(r["wall_s"])
    log(f"cases done: {len(rows)} rows, {len(errors)} errors, wall {time.time() - t_all:.1f}s")
    progress(4, 4, "analysis")
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1, default=str))
    (out_dir / "samples" / "trajectories.json").write_text(json.dumps(samples, indent=1, default=float))
    wall = time.time() - t_all
    meta = {"mode": args.mode, "seed": 42, "eps_lin": EPS_LIN, "eps_nl": EPS_NL, "delta": DELTA,
            "tmax_lin": TMAX_LIN, "tmax_nl": TMAX_NL, "explore": EXPLORE, "phase0": PHASE0, "lin_chooser": CHOOSER,
            "lin_chooser_note": ("dev-seed comparison (lin604_q9, lin600_q1, lin601_q7, lin600_q2 @300k): pure FW "
                                 "tracking stalls on load-state reachability (0/4 decided); greedy 2/4; uniform 2/4; "
                                 "mix 50/50 3/4 -> mix chosen. Tuned on pilot dev seeds (disclosed)."),
            "cert_check_frac": CERT_CHECK_FRAC, "oos_check_frac": OOS_CHECK_FRAC, "S_lock_standard": S_LOCK,
            "generator_hash": generator_hash(), "oos_scan_seeds": [oos_seeds[0], oos_seeds[-1]],
            "oos_chosen": [{"family": r["family"], "pid": r["pid"], "V_star": r["V_star"], "aligned": r["aligned"]}
                           for r in chosen],
            "std_seeds": [std_seeds[0], std_seeds[-1]], "nl_seeds": [nl_seeds[0], nl_seeds[-1]], "noises": noises,
            "jtables": jt_info, "wall_s": wall,
            "job_wall_s_median": {k: float(np.median([w for w in v if w is not None])) if any(
                w is not None for w in v) else None for k, v in walls.items()},
            "job_wall_s_max": {k: float(max([w for w in v if w is not None])) if any(
                w is not None for w in v) else None for k, v in walls.items()},
            "n_errors": len(errors), "concurrent_note": "并发运行：4 CPU worker，与其它任务共享 20 核，墙钟可能偏高"}
    summ = analyse(rows, out_dir, meta)
    try:
        plot_confusion(rows, out_dir)
    except Exception as e:  # noqa: BLE001
        log(f"plot failed: {e!r}")
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    log(f"GO/NO-GO: {summ['go_no_go']} gate={summ['gate']}")
    status = "success" if not errors else "success_with_errors"
    if args.smoke:
        return
    mark_done(status, f"{summ['go_no_go']}; wrong={summ['JPC']['all']['wrong_label']}/{summ['JPC']['all']['n']}; "
                      f"oos_trig={summ['n_true_non_near_tie_oos_triggered']}")
    if PROTOCOL and not args.no_update_progress:
        update_gpu_progress("success", start_iso,
                            (time.time() - datetime.fromisoformat(start_iso).timestamp()) / 60,
                            {"envs": ["E1-Lin OOS squeeze/newpart", "E1-Lin Type-2 std x2 legality", "E1-NL-S T2a-e"],
                             "lin_cases": len(specs), "nl_cases": len(nl_seeds) * 5 * len(noises),
                             "tmax_lin": TMAX_LIN, "tmax_nl": TMAX_NL, "cpu_workers": args.workers,
                             "gpu_model": "RTX 4090 (NL J tables only)", "gpu_count": 1, "concurrent": True,
                             "script_wall_clock_s": round(wall, 1)}, 50 if not pilot else 14)


if __name__ == "__main__":
    main()
