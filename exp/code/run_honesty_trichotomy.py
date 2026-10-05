"""honesty_trichotomy: H6 honest refusal on Type-2 / Type-3 problems and the Prop.-3 identifiability trichotomy.

Usage: run_honesty_trichotomy.py --mode {pilot,full} [--workers 4]

E1-Lin Type-2 (new incentive level b=2, never used in the initial data; generator ptype=2):
  every problem is evaluated under two legality regimes of the probe library:
    probes_legal   : level-2 incentives may be probed          -> R = range(sum I_mu) contains psi2
    probes_illegal : only level <= 1 probes (deployment-only)  -> psi2 in R^perp
  Analytic truth label (harness, uses theta*; solved independently with LP/HiGHS):
    V* = min_pi max_pi' sup{ <z_pi' - z_pi, theta> : theta in prior box, P_R theta = P_R theta* }
    OUT_OF_SCOPE iff V* > eps, else NEED_DATA (eventually certifiable with legal data).
  JPC-Lin (learner, no theta*): ellipsoid (Abbasi-Yadkori, theoretical beta) + public prior box.
    UB(pi,pi') = min( sup_E <d,th>, sup_E <P_R d,th> + sup_box <d_perp,th> )           (sound)
    LB(pi,pi') = inf_E <P_R d,th> + sup_box <d_perp,th>   (valid lower bound of the limit worst regret when the
                 out-of-range component is coordinate aligned, which is asserted)
    CERTIFIED iff min_pi max_pi' UB <= eps;  OUT_OF_SCOPE iff min_pi max_pi' LB > eps;  else NEED_DATA -> probe.
  Probes: greedy information on the undecided P_R d directions, per-step, legal actions of the regime only.
E1-NL-S Type-2 (extended finite class with psi2 grid {0, .75, 1.5}, |Theta| = 41472), 5 variants per instance:
    T2a mixed level-1/2 candidates, level-2 probes legal       T2b mixed candidates, no level-2 probes
    T2c level-2-only candidates, no level-2 probes            T2d level-1/level-2 twin candidates, no level-2 probes
    T2e twin candidates, level-2 probes legal
  Truth label: finite-class Prop. 3 (theta*'s legal-observational-equivalence class decision-ambiguous -> OOS).
  JPC: LR set + exact minimax certificate + trichotomy check + KL-DDA probes (legal set of the variant).
Type-3 (no inherited guarantee; status distribution only):
    T3pop : ledger from population A, the problem is posed on a new population B (in-class, different theta*)
    T3m1  : pair-synergy mechanism (m1, strength 2.0 logit, outside the class); ledger from the m1 platform
Baselines: B4 point-estimate SWM (claims certainty), B9-20 ensemble agreement at zero cost (Lin: Poisson-bootstrap
  ridge, NL: Poisson-bootstrap class MLE). FCR with Clopper-Pearson in two protocols: adaptive stopping (JPC loop)
  and fixed n (non-adaptive uniform legal probes, a single certification at n_fix).
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
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.certify.lin_closed import range_projector  # noqa: E402
from dsswm.certify.minimax_enum import (certify_minimax, group_ids, observable_signature,  # noqa: E402
                                        regret_matrix, trichotomy)
from dsswm.core.actions import ActionSpace  # noqa: E402
from dsswm.envs.nl import NLEnv  # noqa: E402
from dsswm.evidence.ellipsoid import EllipsoidSet  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.models.nl_class import NLClass, NLGrid  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import (DEFAULT_NL_S_GRID, LIN_DEFAULTS, NL_DEFAULTS, Problem, generator_hash,  # noqa: E402
                                     make_lin_instance, make_nl_instance, nl_class_max_jtable)
from dsswm.streams.policies import FAMILIES, Policy, sample_policy  # noqa: E402
from dsswm.streams.utilities import sample_utility  # noqa: E402

TASK = "honesty_trichotomy"
DELTA = 0.05
EPS_LIN = 0.05            # E1-Lin pilot eps (control-plane decision after setup_env_core)
EPS_NL = 0.02             # E1-NL pilot eps (= pre-registered full eps)
TOP_M = 5
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
T2_GRID = NLGrid(**{**{k: getattr(DEFAULT_NL_S_GRID, k) for k in ("L", "R", "alpha", "gamma", "tauL", "beta", "tauR",
                                                                  "psi", "lam")}, "psi2": (0.0, 0.75, 1.5)})
LOG_THR = math.log(1.0 / DELTA)
PRE = json.loads((WS / "plan" / "prereg_lock.json").read_text())
# v1 lock: top-level "E1-Lin"; v2 lock: under "round0"; v3 lock carries neither -> the frozen round-0 values
# (plan/history/round2/prereg_lock.json round0.E1-Lin; identical numbers, dsswm.baselines.lin_rage.S_LIN)
_PRE_LIN = (PRE.get("E1-Lin") or (PRE.get("round0") or {}).get("E1-Lin")
            or {"ellipsoid_S": 2.8071337695236402, "sigma": 1.5})
S_BOUND = float(_PRE_LIN["ellipsoid_S"])
SIGMA_LIN = float(_PRE_LIN["sigma"])
LIN_K = 5                 # Type-2 problems per E1-Lin instance
TMAX_LIN = 150000         # probe cap (rounds) per E1-Lin case
LIN_BLOCK_FRAC = 0.01     # certificate re-evaluated every max(1, 1% of steps so far) rounds (<=1% overshoot)
NFIX_LIN = 20000          # fixed-n protocol (uniform legal probes, one certification)
TMAX_NL = 3000            # probe cap (rounds) per E1-NL case (= step cap used in nl_reuse_kappa_*)
NFIX_NL = 1000
M1_STRENGTH = 2.0
JT_CACHE = WS / "exp" / "cache" / "jtables"
RES_ROOT = WS / "exp" / "results"
ENV_T1 = "E1-NL-S-kappaHigh"


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


# ================================================================== E1-Lin
def lin_box(d_names):
    pr = LIN_DEFAULTS["prior"]
    lo, hi = np.zeros(len(d_names)), np.zeros(len(d_names))
    for k, n in enumerate(d_names):
        key = "psi2" if n == "psi2" else ("psi" if n == "psi1" else n.rstrip("0123456789"))
        lo[k], hi[k] = pr[key]
    return lo, hi


class LinGeom:
    """Public (learner-side) geometry of a legality regime: range projector of the probe library."""

    def __init__(self, lc: LinClass, aspace: ActionSpace, maxlev: int):
        L, R, N, nb = lc.L, lc.R, lc.nmax + 1, lc.nb
        self.L, self.R, self.N, self.nb = L, R, N, nb
        self.lc, self.maxlev = lc, maxlev
        self.Ftab = np.zeros((L * R * N * nb, lc.d))
        for i in range(L):
            for j in range(R):
                for n in range(N):
                    for b in range(nb):
                        self.Ftab[self.fidx(i, j, n, b)] = lc.feature(i, j, n, b)
        lib = np.array([lc.feature(i, j, n, b) for i in range(L) for j in range(R) for n in range(N)
                        for b in range(maxlev + 1)])
        self.PR = range_projector(lib)
        # coordinates whose unit vector lies entirely in R^perp (box on them is exact for the unidentified part)
        self.perp_coord = np.linalg.norm(self.PR, axis=0) < 1e-9
        self.legal = np.array([a for a in range(aspace.n) if aspace.max_level(a) <= maxlev])
        P = 3
        self.ai = np.zeros((aspace.n, P), np.int64)
        self.aj = np.zeros((aspace.n, P), np.int64)
        self.ab = np.zeros((aspace.n, P), np.int64)
        self.am = np.zeros((aspace.n, P), bool)
        for a_idx, a in enumerate(aspace.actions):
            inc = {(i, j): l for i, j, l in a.incentives}
            for s, (i, j) in enumerate(a.pairs):
                self.ai[a_idx, s], self.aj[a_idx, s], self.ab[a_idx, s] = i, j, inc.get((i, j), 0)
                self.am[a_idx, s] = True

    def fidx(self, i, j, n, b):
        return ((i * self.R + j) * self.N + n) * self.nb + b

    def action_feature_index(self, loads):
        n_i = np.asarray(loads)[self.ai]
        return ((self.ai * self.R + self.aj) * self.N + n_i) * self.nb + self.ab


def lin_bounds(ell, Z, geom, lo, hi, eps):
    th = ell.theta_hat()
    Vi = np.linalg.inv(ell.V)
    sb = ell.sqrt_beta()
    D = Z[None, :, :] - Z[:, None, :]                     # D[a, b] = z_b - z_a  (challenger b vs candidate a)
    G = D @ geom.PR
    Dp = D - G
    aligned = bool(np.all(np.abs(Dp[..., ~geom.perp_coord]) < 1e-9))
    box_sup = np.maximum(Dp * lo, Dp * hi).sum(-1)
    wD = np.sqrt(np.maximum(np.einsum("abd,de,abe->ab", D, Vi, D), 0))
    wG = np.sqrt(np.maximum(np.einsum("abd,de,abe->ab", G, Vi, G), 0))
    UB = np.minimum(D @ th + sb * wD, G @ th + sb * wG + box_sup)
    LB = G @ th - sb * wG + box_sup
    K = len(Z)
    UB[np.arange(K), np.arange(K)] = 0.0
    LB[np.arange(K), np.arange(K)] = 0.0
    Rbar = UB.max(1)
    pi_c = int(np.argmin(Rbar))
    lb_worst = LB.max(1)
    oos = bool(aligned and (lb_worst > eps).all())
    return {"pi": pi_c, "r_bar": float(max(Rbar[pi_c], 0.0)), "certified": bool(Rbar[pi_c] <= eps), "oos": oos,
            "aligned": aligned, "UB": UB, "LB": LB, "G": G, "wG": wG, "Vi": Vi, "lb_min": float(lb_worst.min())}


def lin_truth_label(Z, theta, geom, lo, hi, eps):
    """Harness: V* by LP (HiGHS) over the prior box intersected with the identified fibre of theta*."""
    U, s, _ = np.linalg.svd(geom.PR)
    NR = U[:, s > 0.5]                                    # orthonormal basis of R
    A_eq, b_eq = NR.T, NR.T @ theta
    K = len(Z)
    W = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            d = Z[b] - Z[a]
            if np.abs(d).max() < 1e-14:
                continue
            r = linprog(-d, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lo, hi)), method="highs")
            if r.status != 0:
                raise RuntimeError(f"truth LP failed: {r.message}")
            W[a, b] = -r.fun
    Va = W.max(1)
    vstar = float(Va.min())
    return {"V_star": vstar, "label": "OUT_OF_SCOPE" if vstar > eps else "NEED_DATA",
            "near_tie": bool(abs(vstar - eps) < 0.2 * eps), "pi_safe": int(np.argmin(Va))}


def lin_choose(geom, res, loads, eps, rng, explore=0.05):
    """Greedy per-step probe: sum of relative width reductions over undecided identified directions."""
    if rng.random() < explore:
        return int(rng.choice(geom.legal)), "explore"
    UB, LB, G, wG, Vi = res["UB"], res["LB"], res["G"], res["wG"], res["Vi"]
    pi = res["pi"]
    tg = []
    for b in np.flatnonzero(UB[pi] > eps):                 # certification direction(s) of the minimax candidate
        tg.append((pi, b))
    for a in range(UB.shape[0]):                           # OOS-test direction of every other candidate
        if a == pi:
            continue
        b = int(np.argmax(LB[a]))
        if LB[a, b] <= eps:
            tg.append((a, b))
    T = [G[a, b] for a, b in tg if wG[a, b] > 1e-10]
    W = np.array([wG[a, b] for a, b in tg if wG[a, b] > 1e-10])
    if not T:
        return int(rng.choice(geom.legal)), "zero_info"
    T = np.array(T)
    FV = geom.Ftab @ Vi
    q = (FV * geom.Ftab).sum(1)
    proj = FV @ T.T
    s_pair = ((proj ** 2) / (1.0 + q[:, None]) / (W[None] ** 2)).sum(1)
    fi = geom.action_feature_index(loads)
    sc = np.where(geom.am, s_pair[fi], 0.0).sum(1)
    leg = geom.legal
    return int(leg[int(np.argmax(sc[leg]))]), "greedy"


def lin_job(seed, noise, eps, tmax, nfix, samples_wanted):
    torch.set_num_threads(1)
    t_job = time.time()
    inst0 = make_lin_instance(seed, noise_seed=noise, ptypes=(2,), K=LIN_K)
    env0 = inst0.env
    lc = LinClass(3, 3, env0.aspace.nb, nmax=3)
    lo, hi = lin_box(lc.names)
    theta = env0.true_theta_vector()                      # harness only
    assert np.all(theta >= lo - 1e-12) and np.all(theta <= hi + 1e-12)
    geoms = {"probes_legal": LinGeom(lc, env0.aspace, 2), "probes_illegal": LinGeom(lc, env0.aspace, 1)}
    # range of the CURRENT data (initial rounds only) -> Type-2 'proper' vs 'control'
    X0 = np.concatenate([lc.obs_rows(o)[0] for o in inst0.init_obs], 0)
    PD = range_projector(X0)
    rows, samples = [], []
    for k, q in enumerate(inst0.problems):
        Z = np.stack([lc.z(p, q.loads0, q.H, q.utility, env0.aspace) for p in q.policies])
        Jt = Z @ theta
        Dall = (Z[None] - Z[:, None]).reshape(-1, lc.d)
        d_out_data = float(np.abs(Dall - Dall @ PD).max())
        n_lvl2 = int(sum(p.level == 2 for p in q.policies))
        for regime, geom in geoms.items():
            tl = lin_truth_label(Z, theta, geom, lo, hi, eps)
            d_perp = float(np.abs(Dall - Dall @ geom.PR).max())
            base = {"env": "E1-Lin", "ptype": 2, "instance": seed, "noise_seed": noise, "problem_index": k,
                    "pid": q.pid, "regime": regime, "variant": f"Lin_{regime}", "true_label": tl["label"],
                    "V_star": tl["V_star"], "near_tie": tl["near_tie"], "eps": eps, "delta": DELTA,
                    "type2_proper": bool(d_out_data > 1e-9), "d_out_of_data_range_max": d_out_data,
                    "d_out_of_legal_range_max": d_perp, "n_policies": len(q.policies), "n_level2_policies": n_lvl2,
                    "H": q.H}
            # ---------------- JPC (adaptive)
            inst = make_lin_instance(seed, noise_seed=noise, ptypes=(2,), K=LIN_K)
            h = inst.env.handle()
            n_steps0 = h.n_steps
            ell = EllipsoidSet(lc, sigma=h.known_constants()["sigma"], delta=DELTA, S=S_BOUND, lam=1.0)
            for o in inst.init_obs:
                ell.update(o)
            rng = np.random.default_rng([seed, noise, k, 11 if regime == "probes_legal" else 12])
            t0 = time.perf_counter()
            steps, status, start_status, traj, n_l2 = 0, None, None, [], 0
            aligned_all = True
            next_check = 0
            while True:
                if steps < next_check:
                    a, mode = lin_choose(geom, res, h.observable_state()[0], eps, rng)
                    ell.update(h.step(a))
                    n_l2 += int(h.aspace.max_level(a) == 2)
                    steps += 1
                    continue
                next_check = steps + max(1, int(LIN_BLOCK_FRAC * steps))
                res = lin_bounds(ell, Z, geom, lo, hi, eps)
                aligned_all &= res["aligned"]
                cur = "CERTIFIED" if res["certified"] else ("OUT_OF_SCOPE" if res["oos"] else "NEED_DATA")
                if start_status is None:
                    start_status = cur
                    start_pi = res["pi"]
                if cur != "NEED_DATA":
                    status = cur
                    break
                if steps >= tmax:
                    status = "NEED_DATA"
                    break
                a, mode = lin_choose(geom, res, h.observable_state()[0], eps, rng)
                obs = h.step(a)
                ell.update(obs)
                n_l2 += int(h.aspace.max_level(a) == 2)
                steps += 1
                if len(traj) < 60 and (steps <= 10 or steps % 200 == 0):
                    traj.append({"step": steps, "action": a, "mode": mode, "r_bar": res["r_bar"],
                                 "lb_min": res["lb_min"], "pi": res["pi"]})
            assert h.n_steps - n_steps0 == steps, "step accounting mismatch"
            assert ell.n_rounds == len(inst.init_obs) + steps, "evidence must contain exactly the real rounds"
            pi = res["pi"]
            reg = float(Jt.max() - Jt[pi])
            rows.append({**base, "method": "JPC", "protocol": "adaptive", "status": status,
                         "status_at_start": start_status, "new_env_steps": steps,
                         "censored": status == "NEED_DATA", "certified_policy": pi, "true_regret": reg,
                         "false_cert": bool(status == "CERTIFIED" and reg > eps),
                         "start_policy_regret": float(Jt.max() - Jt[start_pi]),
                         "theta_star_in_set": bool(ell.contains(theta)), "aligned": aligned_all,
                         "level2_probe_steps": n_l2, "wall_clock_s": time.perf_counter() - t0,
                         "r_bar_end": res["r_bar"], "lb_min_end": res["lb_min"]})
            if samples_wanted and len(samples) < 4 and (k < 2):
                samples.append({"env": "E1-Lin", "instance": seed, "pid": q.pid, "regime": regime,
                                "policies": [p.name for p in q.policies], "J_true": Jt.tolist(),
                                "true_label": tl["label"], "V_star": tl["V_star"], "start_status": start_status,
                                "final_status": status, "steps": steps, "certified_policy": pi,
                                "trajectory_head": traj})
            # ---------------- JPC fixed-n (uniform legal probes, one certification at n_fix)
            inst = make_lin_instance(seed, noise_seed=noise, ptypes=(2,), K=LIN_K)
            h = inst.env.handle()
            ellf = EllipsoidSet(lc, sigma=h.known_constants()["sigma"], delta=DELTA, S=S_BOUND, lam=1.0)
            for o in inst.init_obs:
                ellf.update(o)
            frng = np.random.default_rng([seed, noise, k, 21 if regime == "probes_legal" else 22])
            for _ in range(nfix):
                ellf.update(h.step(int(frng.choice(geom.legal))))
            rf = lin_bounds(ellf, Z, geom, lo, hi, eps)
            sf = "CERTIFIED" if rf["certified"] else ("OUT_OF_SCOPE" if rf["oos"] else "NEED_DATA")
            regf = float(Jt.max() - Jt[rf["pi"]])
            rows.append({**base, "method": "JPC", "protocol": f"fixed_n{nfix}", "status": sf,
                         "new_env_steps": nfix, "censored": False, "certified_policy": rf["pi"], "true_regret": regf,
                         "false_cert": bool(sf == "CERTIFIED" and regf > eps)})
        # ---------------- baselines at zero cost (independent of the probe regime)
        X, y = np.concatenate([lc.obs_rows(o)[0] for o in inst0.init_obs]), np.concatenate(
            [lc.obs_rows(o)[1] for o in inst0.init_obs])
        th_r = np.linalg.solve(np.eye(lc.d) + X.T @ X, X.T @ y)
        pi4 = int(np.argmax(Z @ th_r))
        common = {"env": "E1-Lin", "ptype": 2, "instance": seed, "noise_seed": noise, "problem_index": k, "pid": q.pid,
                  "regime": "any", "variant": "Lin_zero_cost", "eps": eps, "type2_proper": bool(d_out_data > 1e-9),
                  "true_label": None}
        reg4 = float(Jt.max() - Jt[pi4])
        rows.append({**common, "method": "B4", "protocol": "zero_cost", "status": "CERTIFIED", "new_env_steps": 0,
                     "certified_policy": pi4, "true_regret": reg4, "false_cert": bool(reg4 > eps),
                     "theta_hat_psi2": float(th_r[-1])})
        brng = np.random.default_rng([seed, noise, k, 99])
        picks = []
        for _ in range(20):
            w = brng.poisson(1.0, len(y)).astype(float)
            thb = np.linalg.solve(np.eye(lc.d) + (X * w[:, None]).T @ X, (X * w[:, None]).T @ y)
            picks.append(int(np.argmax(Z @ thb)))
        vals, cnt = np.unique(picks, return_counts=True)
        maj = int(vals[np.argmax(cnt)])
        st9 = "CERTIFIED" if len(vals) == 1 else "NEED_DATA"
        reg9 = float(Jt.max() - Jt[maj])
        rows.append({**common, "method": "B9-20", "protocol": "zero_cost", "status": st9, "new_env_steps": 0,
                     "certified_policy": maj, "true_regret": reg9, "false_cert": bool(st9 == "CERTIFIED" and reg9 > eps),
                     "agree_frac": float(cnt.max() / 20)})
    return {"kind": "lin", "seed": seed, "rows": rows, "samples": samples, "wall_s": time.time() - t_job}


# ================================================================== E1-NL-S Type-2
def t2_problems(seed, aspace2, n=3):
    """Identical to nl_reuse_kappa_high.t2_problems (same pids -> shared J-table cache)."""
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


def twin_problem(seed, aspace2):
    """T2d/T2e: three incentive-using level-1 policies and their level-2 twins (same schedule, level 2)."""
    prng = np.random.default_rng([seed, 41])
    H = int(prng.choice((6, 8)))
    loads0 = prng.integers(0, NMAX + 1, aspace2.L + aspace2.R)
    eng0 = (prng.random(aspace2.L + aspace2.R) < 0.85).astype(np.int64)
    aff = prng.standard_normal((aspace2.L, aspace2.R))
    base, names = [], set()
    tries = 0
    while len(base) < 3 and tries < 200:
        tries += 1
        p = sample_policy(str(prng.choice(("front_incentive", "uniform_incentive", "late_incentive"))), prng, aspace2,
                          H, aff, NMAX, level=1)
        if p.name not in names:
            names.add(p.name); base.append(p)
    twins = [Policy(p.family, {**p.params, "level": 2}, aspace2, H, aff, rng_seed=0) for p in base]
    util = sample_utility(prng, H, aspace2.L, aspace2.R, with_retention=True, y_scale=1.0)
    return Problem(f"nl{seed}_t2twin", 2, H, loads0, eng0, base + twins, util, aff)


def nl_t2_all_problems(seed, aspace2):
    p = t2_problems(seed, aspace2)
    tw = twin_problem(seed, aspace2)
    # (variant, problem, max probe level)
    return [("T2a_mixed_level2_probes_legal", p[0], 2), ("T2b_mixed_no_level2_probes", p[1], 1),
            ("T2c_level2_only_no_level2_probes", p[2], 1), ("T2d_twins_no_level2_probes", tw, 1),
            ("T2e_twins_level2_probes_legal", tw, 2)]


def nl_t2_jtables(seeds, log):
    """GPU phase (main process): exact J tables on the extended class; cached by pid."""
    ncl2 = NLClass(T2_GRID, device="cuda")
    params2 = ncl2.torch_params()
    aspace2 = ActionSpace(2, 2, 1, (1, 2))
    prop2 = NLPropagator(2, 2, NMAX, aspace2, C_KNOWN, RHO_RET, device="cuda")
    n_new, t0 = 0, time.time()
    for s in seeds:
        seen = set()
        for _, q, _ in nl_t2_all_problems(s, aspace2):
            if q.pid in seen:
                continue
            seen.add(q.pid)
            path = JT_CACHE / f"E1-NL-S-T2_{q.pid}_raw.npy"
            if not path.exists():
                raw = prop2.j_table(params2, q.policies, q.loads0, q.engaged0, q.H,
                                    type(q.utility)(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0))
                np.save(path, raw)
                n_new += 1
    log(f"Type-2 NL J tables ready (new {n_new}, |Theta_ext|={ncl2.B}, {time.time() - t0:.1f}s)")
    vram = torch.cuda.max_memory_allocated() / 2 ** 20
    del params2
    torch.cuda.empty_cache()
    return {"n_new_tables": n_new, "seconds": time.time() - t0, "max_vram_mb": vram, "Theta_ext": ncl2.B}


def jpc_nl_loop(Reg, lr, h, prop, py, pe, LT_np, inc, legal, gid, eps, tmax, rng, check_oos=True, traj=None):
    steps, status, start = 0, None, None
    while True:
        mask = lr.mask().numpy()
        cert = certify_minimax(Reg, mask, eps, TOP_M)
        cur = cert["status"].value
        if cur not in ("CERTIFIED", "MODEL_CONFLICT") and check_oos and steps % 10 == 0:
            amb, _, _ = trichotomy(Reg, mask, gid, eps)
            if amb:
                cur = "OUT_OF_SCOPE"
        if start is None:
            start = (cur, cert["pi"])
        if cur != "NEED_DATA":
            status = cur
            break
        if steps >= tmax:
            status = "NEED_DATA"
            break
        kh = lr.mle()
        blk = cert["blocking"]
        margins = LOG_THR - lr.log_ratio().numpy()[blk]
        code = prop.codec.encode(*h.observable_state())
        a, info = dda_choose(code, py[kh:kh + 1], pe[kh:kh + 1], LT_np[kh], py[blk], pe[blk], margins, inc, prop,
                             legal, rng)
        lr.update(h.step(a))
        steps += 1
        if traj is not None and len(traj) < 60 and (steps <= 10 or steps % 25 == 0):
            traj.append({"step": steps, "action": int(a), "mode": info["mode"], "set_size": int(mask.sum()),
                         "r_bar": cert["r_bar"], "pi": cert["pi"]})
    cert = certify_minimax(Reg, lr.mask().numpy(), eps, TOP_M)
    return status, steps, start, cert


def b9_nl(LT, prop, init_obs, J, seed_key, M=20):
    wrng = np.random.default_rng(seed_key)
    ll = prop.loglik(LT, list(init_obs)).numpy()          # (B, n0)
    W = wrng.poisson(1.0, (M, ll.shape[1])).astype(float)
    cum = W @ ll.T + wrng.random((M, ll.shape[0])) * 1e-9
    km = np.argmax(cum, 1)
    pis = np.argmax(J[km], 1)
    vals, cnt = np.unique(pis, return_counts=True)
    return ("CERTIFIED" if len(vals) == 1 else "NEED_DATA"), int(vals[np.argmax(cnt)]), float(cnt.max() / M)


def nl_t2_job(seed, noise, eps, tmax, nfix, samples_wanted):
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

    def copy_lr():
        lr = SeqLRSet(prop2, LT2, DELTA)
        lr.cum, lr.log_num, lr.n_rounds = lr0.cum.clone(), lr0.log_num, lr0.n_rounds
        return lr

    for vi, (vname, q, maxlev) in enumerate(nl_t2_all_problems(seed, aspace2)):
        J = nl_class_max_jtable(q, prop2, params2, cache_path=str(JT_CACHE / f"E1-NL-S-T2_{q.pid}_raw.npy"))
        Reg = regret_matrix(J)
        Jt = J[ti2]
        legal = np.array([a for a in range(aspace2.n) if aspace2.max_level(a) <= maxlev])
        gid = gids[maxlev]
        mask0 = lr0.mask().numpy()
        members = (gid == gid[ti2]) & mask0
        amb_true = bool(Reg[members].max(0).min() > eps) if members.any() else True
        # decision relevance of psi2 given the data: does the class-equivalence under level<=1 data matter?
        members_l1 = (gids[1] == gids[1][ti2])
        psi2_relevant = bool(Reg[members_l1].max(0).min() > eps)
        label = "OUT_OF_SCOPE" if amb_true else "NEED_DATA"
        common = {"env": "E1-NL-S", "ptype": 2, "instance": seed, "noise_seed": noise, "problem_index": vi,
                  "pid": q.pid, "variant": vname, "regime": "probes_legal" if maxlev == 2 else "probes_illegal",
                  "true_label": label, "eps": eps, "delta": DELTA, "psi2_true": psi2_true,
                  "type2_proper": psi2_relevant, "n_policies": len(q.policies), "H": q.H,
                  "truth_class_size": int(members.sum())}
        # ---------------- JPC adaptive
        lr = copy_lr()
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
        rows.append({**common, "method": "JPC", "protocol": "adaptive", "status": status, "status_at_start": start[0],
                     "new_env_steps": steps, "censored": status == "NEED_DATA", "certified_policy": pi,
                     "true_regret": reg, "false_cert": bool(status == "CERTIFIED" and reg > eps),
                     "start_policy_regret": float(Jt.max() - Jt[start[1]]) if start[1] is not None else None,
                     "theta_star_in_set": bool(lr.mask().numpy()[ti2]), "set_size_end": int(lr.mask().sum()),
                     "wall_clock_s": time.perf_counter() - t0})
        if traj is not None:
            samples.append({"env": "E1-NL-S", "instance": seed, "pid": q.pid, "variant": vname,
                            "policies": [p.name for p in q.policies], "J_true": Jt.tolist(), "true_label": label,
                            "psi2_true": psi2_true, "start_status": start[0], "final_status": status,
                            "steps": steps, "certified_policy": pi, "trajectory_head": traj})
        # ---------------- JPC fixed n (uniform legal probes; one certification + trichotomy at n_fix)
        lrf = copy_lr()
        envf = new_env(500 + vi * 10 + maxlev)
        hf = envf.handle()
        frng = np.random.default_rng([seed, noise, 919, vi])
        for _ in range(nfix):
            lrf.update(hf.step(int(frng.choice(legal))))
        mf = lrf.mask().numpy()
        cf = certify_minimax(Reg, mf, eps, TOP_M)
        sf = cf["status"].value
        if sf == "NEED_DATA" and trichotomy(Reg, mf, gid, eps)[0]:
            sf = "OUT_OF_SCOPE"
        regf = float(Jt.max() - Jt[cf["pi"]]) if cf["pi"] is not None else None
        rows.append({**common, "method": "JPC", "protocol": f"fixed_n{nfix}", "status": sf, "new_env_steps": nfix,
                     "censored": False, "certified_policy": cf["pi"], "true_regret": regf,
                     "false_cert": bool(sf == "CERTIFIED" and regf > eps)})
        # ---------------- baselines (zero cost)
        kh0 = lr0.mle()
        pi4 = int(np.argmax(J[kh0]))
        reg4 = float(Jt.max() - Jt[pi4])
        rows.append({**common, "method": "B4", "protocol": "zero_cost", "status": "CERTIFIED", "new_env_steps": 0,
                     "certified_policy": pi4, "true_regret": reg4, "false_cert": bool(reg4 > eps)})
        st9, maj, agree = b9_nl(LT2, prop2, init_obs, J, [seed, noise, 929, vi])
        reg9 = float(Jt.max() - Jt[maj])
        rows.append({**common, "method": "B9-20", "protocol": "zero_cost", "status": st9, "new_env_steps": 0,
                     "certified_policy": maj, "true_regret": reg9, "false_cert": bool(st9 == "CERTIFIED" and reg9 > eps),
                     "agree_frac": agree})
    return {"kind": "nl_t2", "seed": seed, "rows": rows, "samples": samples, "wall_s": time.time() - t_job}


# ================================================================== Type-3 (population switch, m1 mechanism)
def synergy_pattern(seed, L=2, R=2):
    s = np.random.default_rng([seed, 61]).standard_normal((L, R))
    s = s - s.mean(1, keepdims=True)
    s = s - s.mean(0, keepdims=True)
    return s / max(np.abs(s).max(), 1e-12)


def t3_job(seed, noise, eps, tmax, samples_wanted, pop_offset=500):
    torch.set_num_threads(1)
    t_job = time.time()
    ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    params = ncl.torch_params()
    A = make_nl_instance(seed, noise_seed=noise, nl_class=ncl, kappa_mode="high", K=10)
    aspace = A.env.aspace
    prop = NLPropagator(2, 2, NMAX, aspace, C_KNOWN, RHO_RET, device="cpu")
    LT = prop.tables(params)[0]
    LT_np = LT.numpy()
    py, pe = class_prob_tables(ncl.np_params, C_KNOWN, NMAX, aspace.nb)
    inc = IncidenceIndex(prop.codec, aspace, NMAX)
    legal = np.arange(aspace.n)
    gid = group_ids(observable_signature(py, pe, max_level=1))
    q = A.problems[0]
    J = nl_class_max_jtable(q, prop, params, cache_path=str(JT_CACHE / f"{ENV_T1}_{q.pid}_raw.npy"))
    Reg = regret_matrix(J)
    rows, samples = [], []
    # -------- T3pop: ledger from population A, platform now hosts population B (in-class, different theta*)
    B = make_nl_instance(seed + pop_offset, noise_seed=noise, nl_class=ncl, kappa_mode="high", K=1)
    tiA, tiB = A.truth["theta_index"], B.truth["theta_index"]
    JtB = J[tiB]
    lr = SeqLRSet(prop, LT, DELTA)
    for o in A.init_obs:
        lr.update(o)
    h = B.env.handle()
    rng = np.random.default_rng([seed, noise, 1301])
    traj = [] if samples_wanted else None
    status, steps, start, cert = jpc_nl_loop(Reg, lr, h, prop, py, pe, LT_np, inc, legal, gid, eps, tmax, rng,
                                             traj=traj)
    pi = cert["pi"]
    reg = float(JtB.max() - JtB[pi]) if pi is not None else None
    kh0 = int(torch.argmax(SeqLRSet_from(prop, LT, A.init_obs).cum))
    pi4 = int(np.argmax(J[kh0]))
    common = {"env": "E1-NL-S", "ptype": 3, "instance": seed, "noise_seed": noise, "pid": q.pid, "eps": eps,
              "delta": DELTA, "true_label": None}
    rows.append({**common, "variant": "T3pop_population_switch", "method": "JPC", "protocol": "adaptive",
                 "status": status, "status_at_start": start[0], "new_env_steps": steps, "certified_policy": pi,
                 "true_regret": reg, "false_cert": bool(status == "CERTIFIED" and reg is not None and reg > eps),
                 "start_policy_regret_on_B": float(JtB.max() - JtB[start[1]]) if start[1] is not None else None,
                 "theta_B_in_set_end": bool(lr.mask().numpy()[tiB]), "theta_A_in_set_end": bool(lr.mask().numpy()[tiA]),
                 "set_size_end": int(lr.mask().sum()), "same_theta": bool(tiA == tiB),
                 "regret_gap_A_vs_B_best": float(JtB.max() - JtB[int(np.argmax(J[tiA]))])})
    reg4 = float(JtB.max() - JtB[pi4])
    rows.append({**common, "variant": "T3pop_population_switch", "method": "B4", "protocol": "zero_cost",
                 "status": "CERTIFIED", "new_env_steps": 0, "certified_policy": pi4, "true_regret": reg4,
                 "false_cert": bool(reg4 > eps)})
    if traj is not None:
        samples.append({"env": "E1-NL-S", "variant": "T3pop", "instance": seed, "pid": q.pid, "start_status": start[0],
                        "final_status": status, "steps": steps, "J_true_B": JtB.tolist(), "trajectory_head": traj})
    # -------- T3m1: pair-synergy truth (outside the class); ledger built on the m1 platform itself
    t = A.truth
    S = synergy_pattern(seed)
    env = NLEnv(t["alpha"], t["beta"], t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"], c=C_KNOWN,
                rho_ret=RHO_RET, L=2, R=2, nmax=NMAX, budget=NL_DEFAULTS["budget"], incentive_levels=(1,),
                seed=int(seed) * 1000 + noise, syn=M1_STRENGTH * S)
    irng = np.random.default_rng([seed, 12])
    init = [env.step(int(irng.choice(legal))) for _ in range(NL_DEFAULTS["n0"])]
    tp = env.true_params()
    LTt, EYt = prop.tables(params_to_torch(tp, device="cpu"))
    u = q.utility
    Jm = np.array([float(prop._run_plan(*prop.build_policy_plan(p, q.loads0, q.engaged0, q.H), LTt, EYt, u.w,
                                        u.w_ret, u.c_q)[0]) for p in q.policies])
    eta_q = float(np.abs(J - Jm[None]).max(1).min())       # min_theta max_pi |J_true - J_theta| on this problem
    lr = SeqLRSet(prop, LT, DELTA)
    for o in init:
        lr.update(o)
    h = env.handle()
    rng = np.random.default_rng([seed, noise, 1311])
    status, steps, start, cert = jpc_nl_loop(Reg, lr, h, prop, py, pe, LT_np, inc, legal, gid, eps, tmax, rng)
    pi = cert["pi"]
    reg = float(Jm.max() - Jm[pi]) if pi is not None else None
    kh0 = int(torch.argmax(SeqLRSet_from(prop, LT, init).cum))
    pi4 = int(np.argmax(J[kh0]))
    rows.append({**common, "variant": "T3m1_pair_synergy", "method": "JPC", "protocol": "adaptive", "status": status,
                 "status_at_start": start[0], "new_env_steps": steps, "certified_policy": pi, "true_regret": reg,
                 "false_cert": bool(status == "CERTIFIED" and reg is not None and reg > eps),
                 "set_size_end": int(lr.mask().sum()), "eta_q": eta_q, "eta_q_over_eps": eta_q / eps,
                 "m1_strength": M1_STRENGTH})
    reg4 = float(Jm.max() - Jm[pi4])
    rows.append({**common, "variant": "T3m1_pair_synergy", "method": "B4", "protocol": "zero_cost",
                 "status": "CERTIFIED", "new_env_steps": 0, "certified_policy": pi4, "true_regret": reg4,
                 "false_cert": bool(reg4 > eps), "eta_q": eta_q})
    return {"kind": "t3", "seed": seed, "rows": rows, "samples": samples, "wall_s": time.time() - t_job}


def SeqLRSet_from(prop, LT, obs):
    lr = SeqLRSet(prop, LT, DELTA)
    for o in obs:
        lr.update(o)
    return lr


# ================================================================== analysis
def _cp(k, n):
    lo, hi = clopper_pearson(k, n) if n else (0.0, 1.0)
    return [lo, hi]


STATUSES = ("CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE", "MODEL_CONFLICT", "COMPUTE_UNKNOWN")


def analyse(rows, out_dir, meta):
    def sel(**kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    out = {"task_id": TASK, **meta}
    # ---------------- E1-Lin trichotomy
    lin = sel(env="E1-Lin", method="JPC", protocol="adaptive")

    def lin_correct(r):
        if r["true_label"] == "OUT_OF_SCOPE":
            return r["status"] == "OUT_OF_SCOPE"
        return r["status"] == "CERTIFIED" and r["true_regret"] <= r["eps"]
    acc = [lin_correct(r) for r in lin]
    nontie = [lin_correct(r) for r in lin if not r["near_tie"]]
    # start-status determination: OOS declared at start must be truly OOS; CERTIFIED at start must be correct
    start_oos_wrong = sum(r["status_at_start"] == "OUT_OF_SCOPE" and r["true_label"] != "OUT_OF_SCOPE" for r in lin)
    final_oos_wrong = sum(r["status"] == "OUT_OF_SCOPE" and r["true_label"] != "OUT_OF_SCOPE" for r in lin)
    lab = {L: sum(r["true_label"] == L for r in lin) for L in ("NEED_DATA", "OUT_OF_SCOPE")}
    nd = [r for r in lin if r["true_label"] == "NEED_DATA"]
    out["E1_Lin"] = {
        "n_cases": len(lin), "n_problems": len({r["pid"] for r in lin}), "label_counts": lab,
        "label_counts_by_regime": {g: {L: sum(r["true_label"] == L and r["regime"] == g for r in lin)
                                       for L in ("NEED_DATA", "OUT_OF_SCOPE")} for g in ("probes_legal", "probes_illegal")},
        "trichotomy_accuracy_final": float(np.mean(acc)) if acc else None,
        "trichotomy_accuracy_final_excl_near_tie": float(np.mean(nontie)) if nontie else None,
        "n_near_tie": int(sum(r["near_tie"] for r in lin)),
        "n_misclassified": int(len(acc) - sum(acc)),
        "misclassified": [{k: r[k] for k in ("pid", "regime", "true_label", "V_star", "status", "status_at_start",
                                             "new_env_steps", "true_regret")} for r, a in zip(lin, acc) if not a],
        "oos_declared_wrongly_start": int(start_oos_wrong), "oos_declared_wrongly_final": int(final_oos_wrong),
        "start_status_vs_label": {L: {s: sum(r["true_label"] == L and r["status_at_start"] == s for r in lin)
                                      for s in ("CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE")} for L in lab},
        "need_data_certified_after_probes": int(sum(r["status"] == "CERTIFIED" for r in nd)),
        "need_data_n": len(nd),
        "need_data_probe_steps_median": float(np.median([r["new_env_steps"] for r in nd if r["status"] == "CERTIFIED"]))
        if any(r["status"] == "CERTIFIED" for r in nd) else None,
        "need_data_probe_steps_by_regime_median": {g: float(np.median(v)) if v else None for g, v in {
            g: [r["new_env_steps"] for r in nd if r["regime"] == g and r["status"] == "CERTIFIED"]
            for g in ("probes_legal", "probes_illegal")}.items()},
        "oos_detection_steps_median": float(np.median([r["new_env_steps"] for r in lin if r["true_label"] ==
                                                        "OUT_OF_SCOPE" and r["status"] == "OUT_OF_SCOPE"]))
        if any(r["true_label"] == "OUT_OF_SCOPE" and r["status"] == "OUT_OF_SCOPE" for r in lin) else None,
        "level2_probe_steps_in_illegal_regime": int(sum(r["level2_probe_steps"] for r in lin
                                                        if r["regime"] == "probes_illegal")),
        "aligned_all": bool(all(r["aligned"] for r in lin)),
        "theta_star_in_set_end": float(np.mean([r["theta_star_in_set"] for r in lin])),
        "n_type2_proper": int(sum(r["type2_proper"] for r in lin)),
    }
    # ---------------- Type-2 honesty (both envs)
    t2 = [r for r in rows if r["ptype"] == 2 and r["method"] == "JPC" and r["protocol"] == "adaptive"]
    honest = {}
    for name, rs in (("all", t2), ("E1-Lin", [r for r in t2 if r["env"] == "E1-Lin"]),
                     ("E1-NL-S", [r for r in t2 if r["env"] == "E1-NL-S"])):
        prop_rs = [r for r in rs if r["type2_proper"]]
        n_cert = sum(r["status"] == "CERTIFIED" for r in rs)
        n_false = sum(r["false_cert"] for r in rs)
        zc = [r for r in rs if r["status_at_start"] == "CERTIFIED"]
        honest[name] = {
            "n": len(rs), "n_type2_proper": len(prop_rs),
            "false_cert_adaptive": int(n_false), "false_cert_rate_all": n_false / max(len(rs), 1),
            "false_cert_rate_all_cp": _cp(n_false, len(rs)),
            "fcr_adaptive": n_false / max(n_cert, 1), "fcr_adaptive_cp": _cp(n_false, n_cert), "n_certified": n_cert,
            "refusal_rate_at_start_all": float(np.mean([r["status_at_start"] in ("NEED_DATA", "OUT_OF_SCOPE")
                                                        for r in rs])) if rs else None,
            "refusal_rate_at_start_type2_proper": float(np.mean([r["status_at_start"] in ("NEED_DATA", "OUT_OF_SCOPE")
                                                                 for r in prop_rs])) if prop_rs else None,
            "zero_cost_certified_n": len(zc),
            "zero_cost_certified_false": int(sum((r.get("start_policy_regret") or 0) > r["eps"] for r in zc)),
            "always_refuse_rate_final": float(np.mean([r["status"] != "CERTIFIED" for r in rs])) if rs else None,
            "always_refuse_rate_final_truth_need_data": float(np.mean(
                [r["status"] != "CERTIFIED" for r in rs if r["true_label"] == "NEED_DATA"]))
            if any(r["true_label"] == "NEED_DATA" for r in rs) else None,
            "final_status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in STATUSES},
            "start_status_counts": {s: int(sum(r["status_at_start"] == s for r in rs)) for s in STATUSES},
        }
    out["type2_honesty"] = honest
    # fixed-n protocol
    fx = [r for r in rows if r["ptype"] == 2 and r["method"] == "JPC" and str(r["protocol"]).startswith("fixed")]
    out["fcr_fixed_n"] = {}
    for name, rs in (("all", fx), ("E1-Lin", [r for r in fx if r["env"] == "E1-Lin"]),
                     ("E1-NL-S", [r for r in fx if r["env"] == "E1-NL-S"])):
        nc = sum(r["status"] == "CERTIFIED" for r in rs)
        nf = sum(r["false_cert"] for r in rs)
        out["fcr_fixed_n"][name] = {"n": len(rs), "n_certified": nc, "n_false": nf, "fcr": nf / max(nc, 1),
                                    "fcr_cp": _cp(nf, nc), "n_fix": sorted({r["protocol"] for r in rs}),
                                    "status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in STATUSES}}
    # baselines
    out["baselines_type2"] = {}
    for env in ("E1-Lin", "E1-NL-S"):
        for m in ("B4", "B9-20"):
            rs = [r for r in rows if r["ptype"] == 2 and r["env"] == env and r["method"] == m]
            if not rs:
                continue
            nc = sum(r["status"] == "CERTIFIED" for r in rs)
            nf = sum(r["false_cert"] for r in rs)
            pr = [r for r in rs if r["type2_proper"]]
            out["baselines_type2"][f"{env}|{m}"] = {
                "n": len(rs), "claims_certified": nc, "false_cert": nf, "false_cert_rate_all": nf / len(rs),
                "fcr": nf / max(nc, 1), "fcr_cp": _cp(nf, nc), "refusal_rate": 1 - nc / len(rs),
                "false_cert_type2_proper": int(sum(r["false_cert"] for r in pr)), "n_type2_proper": len(pr)}
    # NL per-variant
    nl = [r for r in t2 if r["env"] == "E1-NL-S"]
    out["E1_NL_S_by_variant"] = {}
    for v in sorted({r["variant"] for r in nl}):
        rs = [r for r in nl if r["variant"] == v]
        out["E1_NL_S_by_variant"][v] = {
            "n": len(rs), "labels": {L: sum(r["true_label"] == L for r in rs) for L in ("NEED_DATA", "OUT_OF_SCOPE")},
            "type2_proper": sum(r["type2_proper"] for r in rs),
            "start": {s: sum(r["status_at_start"] == s for r in rs) for s in STATUSES if
                      any(r["status_at_start"] == s for r in rs)},
            "final": {s: sum(r["status"] == s for r in rs) for s in STATUSES if any(r["status"] == s for r in rs)},
            "false_cert": sum(r["false_cert"] for r in rs),
            "label_match_final": float(np.mean([(r["status"] == "OUT_OF_SCOPE") == (r["true_label"] == "OUT_OF_SCOPE")
                                                and (r["status"] != "NEED_DATA") for r in rs])),
            "median_steps": float(np.median([r["new_env_steps"] for r in rs]))}
    nl_acc = [((r["true_label"] == "OUT_OF_SCOPE" and r["status"] == "OUT_OF_SCOPE") or
               (r["true_label"] == "NEED_DATA" and r["status"] == "CERTIFIED" and r["true_regret"] <= r["eps"]))
              for r in nl]
    out["E1_NL_S_trichotomy_accuracy_final"] = float(np.mean(nl_acc)) if nl_acc else None
    out["E1_NL_S_oos_declared_wrongly"] = int(sum(r["status"] == "OUT_OF_SCOPE" and r["true_label"] != "OUT_OF_SCOPE"
                                                  for r in nl))
    # ---------------- Type-3
    t3 = [r for r in rows if r["ptype"] == 3]
    out["type3"] = {}
    for v in sorted({r["variant"] for r in t3}):
        for m in ("JPC", "B4"):
            rs = [r for r in t3 if r["variant"] == v and r["method"] == m]
            if not rs:
                continue
            nc = sum(r["status"] == "CERTIFIED" for r in rs)
            nf = sum(r["false_cert"] for r in rs)
            d = {"n": len(rs), "status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in STATUSES},
                 "n_certified": nc, "false_cert": nf, "false_cert_rate_all": nf / len(rs), "fcr": nf / max(nc, 1),
                 "fcr_cp": _cp(nf, nc), "model_conflict_share": sum(r["status"] == "MODEL_CONFLICT" for r in rs) / len(rs)}
            if m == "JPC":
                d["start_status_counts"] = {s: int(sum(r.get("status_at_start") == s for r in rs)) for s in STATUSES}
                d["median_steps"] = float(np.median([r["new_env_steps"] for r in rs]))
                if v.startswith("T3m1"):
                    d["eta_q_over_eps_median"] = float(np.median([r["eta_q_over_eps"] for r in rs]))
                    d["eta_q_over_eps_range"] = [float(min(r["eta_q_over_eps"] for r in rs)),
                                                 float(max(r["eta_q_over_eps"] for r in rs))]
                if v.startswith("T3pop"):
                    d["same_theta_draws"] = int(sum(r["same_theta"] for r in rs))
                    d["theta_B_in_set_end"] = int(sum(r["theta_B_in_set_end"] for r in rs))
            out["type3"][f"{v}|{m}"] = d
    # ---------------- gate
    lin_acc = out["E1_Lin"]["trichotomy_accuracy_final"]
    t2_false = honest["all"]["false_cert_adaptive"]
    refusal = honest["all"]["refusal_rate_at_start_type2_proper"]
    gate = {"E1_Lin_trichotomy_accuracy_eq_1": bool(lin_acc == 1.0), "type2_false_cert_eq_0": bool(t2_false == 0),
            "type2_refusal_rate_ge_0.9 (type2-proper, at start)": bool(refusal is not None and refusal >= 0.9),
            "fcr_cp_upper_le_2delta_adaptive": bool(honest["all"]["fcr_adaptive_cp"][1] <= 2 * DELTA),
            "fcr_cp_upper_le_2delta_fixed_n": bool(out["fcr_fixed_n"]["all"]["fcr_cp"][1] <= 2 * DELTA)}
    out["gate"] = gate
    out["go_no_go"] = "GO" if all(list(gate.values())[:3]) else "NO_GO"
    # ---------------- confusion tables
    lines = ["env,variant_group,protocol,true_label,status_kind,status,count"]
    for env in ("E1-Lin", "E1-NL-S"):
        rs = [r for r in t2 if r["env"] == env]
        groups = sorted({r["regime"] for r in rs})
        for g in groups:
            for L in ("NEED_DATA", "OUT_OF_SCOPE"):
                for kind in ("status_at_start", "status"):
                    for s in STATUSES:
                        c = sum(r["true_label"] == L and r["regime"] == g and r[kind] == s for r in rs)
                        if c:
                            lines.append(f"{env},{g},adaptive,{L},{'start' if kind != 'status' else 'final'},{s},{c}")
    for r_env in ("E1-Lin", "E1-NL-S"):
        rs = [r for r in fx if r["env"] == r_env]
        for L in ("NEED_DATA", "OUT_OF_SCOPE"):
            for s in STATUSES:
                c = sum(r["true_label"] == L and r["status"] == s for r in rs)
                if c:
                    lines.append(f"{r_env},all,fixed_n,{L},final,{s},{c}")
    for v in sorted({r["variant"] for r in t3}):
        for kind in ("status_at_start", "status"):
            for s in STATUSES:
                c = sum(r["variant"] == v and r["method"] == "JPC" and r.get(kind) == s for r in t3)
                if c:
                    lines.append(f"E1-NL-S,{v},adaptive,TYPE3,{'start' if kind != 'status' else 'final'},{s},{c}")
    (out_dir / "confusion.csv").write_text("\n".join(lines) + "\n")
    return out


def plot_confusion(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t2 = [r for r in rows if r["ptype"] == 2 and r["method"] == "JPC" and r["protocol"] == "adaptive"]
    t3 = [r for r in rows if r["ptype"] == 3 and r["method"] == "JPC"]
    cols = ["CERTIFIED", "NEED_DATA", "OUT_OF_SCOPE", "MODEL_CONFLICT"]
    panels = []
    for env in ("E1-Lin", "E1-NL-S"):
        rs = [r for r in t2 if r["env"] == env]
        for kind, ttl in (("status_at_start", "zero-cost (start)"), ("status", "final (after legal probes)")):
            rows_lab = ["NEED_DATA", "OUT_OF_SCOPE"]
            M = np.array([[sum(r["true_label"] == L and r[kind] == s for r in rs) for s in cols] for L in rows_lab])
            panels.append((f"{env} Type-2: {ttl}", rows_lab, M))
    vs = sorted({r["variant"] for r in t3})
    M = np.array([[sum(r["variant"] == v and r["status"] == s for r in t3) for s in cols] for v in vs])
    panels.append(("Type-3 (no guarantee): final", [v.split("_")[0] for v in vs], M))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.2))
    for ax, (ttl, rl, M) in zip(axes, panels):
        ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > 0.6 * max(M.max(), 1) else "black", fontsize=9)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["CERT", "NEED", "OOS", "CONFL"], fontsize=8)
        ax.set_yticks(range(len(rl)))
        ax.set_yticklabels(rl, fontsize=8)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("output status", fontsize=8)
    axes[0].set_ylabel("true identifiability label", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


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


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lin-instances", type=int, default=None)
    ap.add_argument("--nl-instances", type=int, default=None)
    ap.add_argument("--t3-instances", type=int, default=None)
    ap.add_argument("--no-update-progress", action="store_true")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_file = out_dir / "start_time.txt"
    start_iso = start_file.read_text().strip() if start_file.exists() else datetime.now().isoformat()
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    t_all = time.time()
    if pilot:
        lin_seeds = list(range(args.lin_instances or 10))
        nl_seeds = list(range(args.nl_instances or 10))
        t3_seeds = list(range(args.t3_instances or 10))
        nl_noises = [42]
    else:
        lin_seeds = list(range(10000, 10000 + (args.lin_instances or 100)))
        nl_seeds = list(range(10000, 10000 + (args.nl_instances or 40)))
        t3_seeds = list(range(10000, 10000 + (args.t3_instances or 200)))
        nl_noises = [42, 123, 456]
    log(f"start mode={args.mode} lin={len(lin_seeds)}x{LIN_K}x2 regimes, nl={len(nl_seeds)}x5x{len(nl_noises)}, "
        f"t3={len(t3_seeds)}x2, workers={args.workers}, gen_hash={generator_hash()}")
    progress(1, 4, "nl_t2_jtables_gpu")
    jt_info = nl_t2_jtables(nl_seeds, log)
    progress(2, 4, "cases")
    jobs = []
    for s in nl_seeds:
        for nz in nl_noises:
            jobs.append(delayed(nl_t2_job)(s, nz, EPS_NL, TMAX_NL, NFIX_NL, s < 2 and nz == 42))
    for s in lin_seeds:
        jobs.append(delayed(lin_job)(s, 42, EPS_LIN, TMAX_LIN, NFIX_LIN, s < 2))
    for s in t3_seeds:
        jobs.append(delayed(t3_job)(s, 42, EPS_NL, TMAX_NL, s < 2))

    def safe(job):
        f, a, kw = job
        try:
            return f(*a, **kw)
        except Exception as e:  # noqa: BLE001
            return {"kind": f.__name__, "seed": a[0], "rows": [], "samples": [], "error": repr(e),
                    "tb": traceback.format_exc(), "wall_s": None}
    results = Parallel(n_jobs=args.workers, verbose=0)(delayed(safe)(j) for j in jobs)
    rows, samples, errors, walls = [], [], [], {}
    for r in results:
        rows += r["rows"]
        samples += r["samples"]
        if r.get("error"):
            errors.append({k: r[k] for k in ("kind", "seed", "error", "tb")})
            log(f"ERROR {r['kind']} seed={r['seed']}: {r['error']}")
        walls.setdefault(r["kind"], []).append(r["wall_s"])
    log(f"cases done: {len(rows)} rows, {len(errors)} errors, wall {time.time() - t_all:.1f}s")
    progress(3, 4, "analysis")
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
    (out_dir / "samples" / "trajectories.json").write_text(json.dumps(samples, indent=1))
    wall = time.time() - t_all
    meta = {"mode": args.mode, "seed": 42, "eps_lin": EPS_LIN, "eps_nl": EPS_NL, "delta": DELTA,
            "tmax_lin": TMAX_LIN, "nfix_lin": NFIX_LIN, "tmax_nl": TMAX_NL, "nfix_nl": NFIX_NL,
            "m1_strength": M1_STRENGTH, "ellipsoid_S": S_BOUND, "sigma_lin": SIGMA_LIN,
            "generator_hash": generator_hash(), "lin_seeds": [lin_seeds[0], lin_seeds[-1]],
            "nl_seeds": [nl_seeds[0], nl_seeds[-1]], "t3_seeds": [t3_seeds[0], t3_seeds[-1]],
            "nl_noises": nl_noises, "jtables": jt_info, "wall_s": wall,
            "job_wall_s_median": {k: float(np.median([w for w in v if w is not None])) if any(
                w is not None for w in v) else None for k, v in walls.items()},
            "n_errors": len(errors), "concurrent_note": "4 CPU workers; wall-clock may be inflated by concurrent tasks"}
    summ = analyse(rows, out_dir, meta)
    try:
        plot_confusion(rows, out_dir)
    except Exception as e:  # noqa: BLE001
        log(f"plot failed: {e!r}")
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"GO/NO-GO: {summ['go_no_go']}  gate={summ['gate']}")
    progress(4, 4, "done", {"go_no_go": summ["go_no_go"]})
    status = "success" if not errors else "success_with_errors"
    mark_done(status, f"{summ['go_no_go']}; lin_acc={summ['E1_Lin']['trichotomy_accuracy_final']}; "
                      f"t2_false={summ['type2_honesty']['all']['false_cert_adaptive']}")
    if not args.no_update_progress:
        update_gpu_progress("success", start_iso, (time.time() - datetime.fromisoformat(start_iso).timestamp()) / 60,
                            {"envs": ["E1-Lin", "E1-NL-S", "E1-Mis-m1"], "lin_problems": len(lin_seeds) * LIN_K,
                             "lin_regimes": 2, "nl_t2_problems": len(nl_seeds) * 5 * len(nl_noises),
                             "t3_problems": len(t3_seeds) * 2, "Theta_ext_size": jt_info["Theta_ext"],
                             "eps_lin": EPS_LIN, "eps_nl": EPS_NL, "tmax_lin": TMAX_LIN, "tmax_nl": TMAX_NL,
                             "cpu_workers": args.workers, "gpu_model": "RTX 4090 (J tables only)", "gpu_count": 1,
                             "pilot_script_wall_clock_s": round(wall, 1), "concurrent": True},
                            30)


if __name__ == "__main__":
    main()
