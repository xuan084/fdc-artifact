"""hd1b_same_exposure: Prop. C' worst-case family -- same-exposure, different-timing policies (HD1b).

Construction-only (worst-case instance statement, NOT an empirical claim). For each construction instance we build one
problem whose candidate policies are open-loop schedules that match ONE pair (i, j) on exactly k of H steps (same
exposure) at different times (different load trajectories), and run three designs for a fixed n = 2000 real
env.step() rounds, then certify once (plus read-only checkpoints at n in {250, 500, 1000}):
  uniform   uniformly random legal action from the current platform state (one long chain, no resets)
  dda       directed design driven by the DYNAMIC class (T-Lin: greedy XY direction on the most-blocking challenger;
            T-NL: the pre-registered nl_kl_dda.dda_choose on the dynamic G_1 LR set), chain from (loads0, engaged0)
  onpolicy  episodes of the candidate policies themselves (uniform over candidates, reset_to(loads0, engaged0) per
            episode; resets are free and identical for both classes)
Both the static and the dynamic certifier see exactly the same observations of a run.

T-Lin  (E1-Lin, L=R=3, sigma=1.5, nmax=3; construction prior gamma ~ U[0.4, 0.6]):
  static class = LinClass(static=True) (no load column), dynamic class = LinClass (true class).
  Evidence = Abbasi-Yadkori ellipsoid (lam=1, S from the lock); certificate: pi_hat = plug-in argmax (ties ->
  lowest index), CERTIFIED iff max_k sup_{Theta_t} <z_k - z_pi_hat, theta> <= eps (eps = 0.05).
  Same exposure + constant utility weights => d^s_k = 0 EXACTLY for every pair in the group, so the static class
  assigns every member the same value for every theta (checked numerically: ds_max_abs).
  Family 'lin_multi'  : 1 truth-optimal timing + up to 9 timings with true regret >= 1.5 eps, order shuffled by a
                        truth-free rng. The static certifier cannot separate them, so it certifies the lowest-index
                        member: predicted FCR = (#bad)/(#members) = 0.9 for any design (n -> infinity).
  Family 'lin_pair'   : the theorem's literal pair [worst bad, good] (adversarial orientation of the static tie;
                        predicted FCR = 1 for any design).
  Tie-break-free readout 'static_false_equiv_rate': share of (good, bad) pairs with true |gap| > eps that the
                        static confidence set declares eps-equivalent (sup over Theta_t of |gap| <= eps).
  FWL check (Prop. C' (C1)): closed form b_k(xi) = -<d^n_k - B_xi^T d^s_k, gamma*>, B_xi = (X_s^T X_s)^+ X_s^T X_n
  from the run's own design matrix, against (a) the noise-free static OLS bias (exact identity when d^s_k is in the
  row space of X_s) and (b) the realised ridge static estimate (noisy; z-scored against its exact sd). Pairs: the
  construction pairs (d^s = 0) and natural-generator policy pairs (d^s != 0).

T-NL   (E1-NL-S, R0 kappa_high truth on G_1, |Theta| = 13824; eps = 0.02):
  static class = the load-free reparametrisation of run_hd1_matched_twin_static (g = g_ret = 1), dynamic = G_1.
  Evidence = in-class plug-in SeqLRSet (UI threshold log 1/delta), certificate = exact minimax (certify_minimax).
  Exact d^s = 0 is NOT attainable in T-NL: the static class keeps the engagement/retention dynamics, so changing the
  timing changes the static value for some theta (reported: nl_min_static_sup_gap over all same-exposure pairs).
  We therefore use the pre-registered fallback, the continuous version of C': 1 truth-optimal timing + up to 5
  timings with true regret >= 1.5 eps that STRICTLY dominate it under every static theta (min_theta static gap > 0),
  so |b_k| >= |<d^n, gamma*>| holds with the static sign pointing the wrong way for every design; predicted
  static FCR = 1 among certificates (MODEL_CONFLICT / NEED_DATA are reported separately as censored).
Dynamic classes contain the truth (T-Lin: well specified; T-NL: R0 truth on G_1) => dynamic FCR <= delta.

Usage: run_hd1b_same_exposure.py --mode {pilot,full} [--workers 4] [--smoke]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
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
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "hd1b_same_exposure"
DELTA = 0.05
EPS_LIN, EPS_NL = 0.05, 0.02
N_DESIGN = 2000
CHECKPOINTS = (250, 500, 1000, 2000)
DESIGNS = ("uniform", "dda", "onpolicy")
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
LAM = 1.0
H_LIN, K_LIN = 8, 4
H_NL, K_NL = 8, 4
M_LIN_BAD, M_NL_BAD = 9, 5
BAD_MARGIN = 1.5
PILOT_LIN_SEEDS = list(range(700, 720))     # dev seeds (< 10000)
PILOT_NL_SEEDS = list(range(700, 710))
FULL_SEEDS = list(range(30))                # construction instances (offset below); noise seeds 42/123/456
FULL_OFFSET = 10000
FULL_NOISE = (42, 123, 456)

from dsswm.acquire.nl_kl_dda import dda_choose  # noqa: E402
from dsswm.core.dynamics import next_loads  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix  # noqa: E402
from dsswm.evidence.ellipsoid import EllipsoidSet  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import LIN_DEFAULTS, generator_hash, make_lin_instance  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402
from dsswm.streams.utilities import Utility  # noqa: E402
from run_hd1_matched_twin_static import (C_KNOWN, G1, NMAX, RHO_RET, check_static_equivalence,  # noqa: E402
                                         class_torch_params, ctx, truth_in_class)

PRE = json.loads(LOCK.read_text())
# v1 lock: top-level "E1-Lin"; v2 lock: under "round0"; v3 lock carries neither -> the frozen round-0 value
# (dsswm.baselines.lin_rage.S_LIN, identical number)
S_BOUND = float((PRE.get("E1-Lin") or (PRE.get("round0") or {}).get("E1-Lin")
                 or {"ellipsoid_S": 2.8071337695236402})["ellipsoid_S"])
LIN_PRIOR = {"gamma": (0.4, 0.6)}


class OpenLoop:
    """Deterministic open-loop schedule (action index per step)."""
    family = "schedule"

    def __init__(self, seq, steps, H):
        self.seq = [int(a) for a in seq]
        self.steps = tuple(int(s) for s in steps)
        self.H = H
        self.name = "T" + "".join("1" if t in self.steps else "0" for t in range(H))

    def act(self, t, loads, engaged):
        return self.seq[t]


def schedules(aspace, i, j, H, k):
    a_on, a_off = aspace.index([(i, j)]), aspace.index([])
    return [OpenLoop([a_on if t in S else a_off for t in range(H)], S, H) for S in itertools.combinations(range(H), k)]


def load_exposure(steps, H, nmax, l0=0):
    """S = sum over match times of the left participant's load (the d^n coordinate up to sign and scale)."""
    n, S = l0, 0
    for t in range(H):
        if t in steps:
            S += n
            n = min(n + 1, nmax)
        else:
            n = max(n - 1, 0)
    return S


def cp(k, n):
    if n == 0:
        return [None, None]
    lo, hi = clopper_pearson(k, n, 0.05)
    return [float(lo), float(hi)]


# ============================================================================================ T-Lin
def lin_setup(seed, noise_seed):
    inst = make_lin_instance(seed, noise_seed, ptypes=(1,), K=3, prior=LIN_PRIOR)
    env, asp = inst.env, inst.env.aspace
    lc_d = LinClass(env.L, env.R, asp.nb, env.nmax, static=False)
    lc_s = LinClass(env.L, env.R, asp.nb, env.nmax, static=True)
    theta = env.true_theta_vector()
    crng = np.random.default_rng([seed, 501])                  # truth-free construction rng
    i, j = int(crng.integers(env.L)), int(crng.integers(env.R))
    util = Utility(w=np.ones(H_LIN), w_ret=0.0, c_q=1.0 / (LIN_DEFAULTS["y_scale"] * H_LIN))
    loads0 = np.zeros(env.P, np.int64)
    pool = schedules(asp, i, j, H_LIN, K_LIN)
    Zd = np.array([lc_d.z(p, loads0, H_LIN, util, asp) for p in pool])
    Jt = Zd @ theta
    reg = Jt.max() - Jt
    good = [g for g in range(len(pool)) if reg[g] < 1e-12]
    bad = [b for b in range(len(pool)) if reg[b] >= BAD_MARGIN * EPS_LIN]
    srng = np.random.default_rng([seed, 502])                  # truth-free selection/shuffle rng
    g_pick = int(srng.choice(good))
    b_pick = [int(x) for x in srng.choice(bad, size=min(M_LIN_BAD, len(bad)), replace=False)]
    members = [g_pick] + b_pick
    srng.shuffle(members)
    worst = int(max(b_pick, key=lambda b: reg[b]))
    fam = {"lin_multi": [pool[m] for m in members], "lin_pair": [pool[worst], pool[g_pick]]}
    # natural-generator pairs for the FWL check (d^s != 0)
    nat = [q for q in inst.problems]
    return dict(inst=inst, env=env, asp=asp, lc_d=lc_d, lc_s=lc_s, theta=theta, pair=(i, j), util=util,
                loads0=loads0, fam=fam, nat=nat,
                S_of={pool[m].name: load_exposure(pool[m].steps, H_LIN, env.nmax) for m in members})


def lin_certify(ell, Z, eps):
    th = ell.theta_hat()
    J = Z @ th
    pi = int(np.flatnonzero(J >= J.max() - 1e-12 * max(1.0, abs(J.max())))[0])
    D = Z - Z[pi]
    ub = ell.sup_linear(D)
    ub[pi] = 0.0
    r = float(ub.max())
    return {"pi": pi, "r_bar": r, "certified": bool(r <= eps), "J_hat": J}


def lin_false_equiv(ell, Z, Jt, eps):
    """(good, bad) pairs with true |gap| > eps that Theta_t declares eps-equivalent (both one-sided sups <= eps)."""
    g = int(np.argmax(Jt))
    k_all, k_eq = 0, 0
    for b in range(len(Z)):
        if abs(Jt[g] - Jt[b]) > eps:
            d = Z[b] - Z[g]
            up = ell.sup_linear(np.stack([d, -d]))
            k_all += 1
            k_eq += bool(up.max() <= eps)
    return k_eq, k_all


def lin_fwl(X, theta, ell_s, lc, pairs_d, sigma):
    sl = lc.slices()
    cols_n = np.arange(sl["gamma"].start, sl["gamma"].stop)
    cols_s = np.setdiff1d(np.arange(lc.d), cols_n)
    Xs, Xn = X[:, cols_s], X[:, cols_n]
    gam = theta[cols_n]
    B = np.linalg.lstsq(Xs, Xn, rcond=None)[0]
    mu = X @ theta
    th0 = np.linalg.lstsq(Xs, mu, rcond=None)[0]             # noise-free static OLS (min-norm)
    A = Xs.T @ Xs + LAM * np.eye(len(cols_s))
    Ai = np.linalg.inv(A)
    th_r = ell_s.theta_hat()[cols_s]
    m_r = Ai @ Xs.T @ mu
    U, sv, Vt = np.linalg.svd(Xs, full_matrices=False)
    rs = Vt[sv > 1e-8 * sv.max()]
    out = []
    for name, d in pairs_d:
        ds, dn = d[cols_s], d[cols_n]
        gap = float(d @ theta)
        pred = float(-(dn - B.T @ ds) @ gam)
        meas0 = float(ds @ th0 - gap)
        meas = float(ds @ th_r - gap)
        exp_r = float(ds @ m_r - gap)
        sd = float(sigma * math.sqrt(max(ds @ Ai @ (Xs.T @ Xs) @ Ai @ ds, 0.0)))
        resid = float(np.linalg.norm(ds - rs.T @ (rs @ ds)))
        out.append({"pair": name, "ds_norm": float(np.linalg.norm(ds)), "dn_norm": float(np.linalg.norm(dn)),
                    "ds_in_rowspace": bool(resid < 1e-8 * max(1.0, np.linalg.norm(ds))), "true_gap": gap,
                    "b_pred_fwl": pred, "bias_noisefree_ols": meas0, "bias_ridge_measured": meas,
                    "bias_ridge_expected": exp_r, "sd_ridge": sd,
                    "err_identity": abs(meas0 - pred), "err_measured_vs_fwl": abs(meas - pred),
                    "z_measured_vs_expected": (meas - exp_r) / sd if sd > 0 else None})
    return out


def lin_action_rows(asp):
    return [[(i, j, {(a, b): l for a, b, l in asp.actions[k].incentives}.get((i, j), 0)) for i, j in asp.actions[k].pairs]
            for k in range(asp.n)]


def lin_action_basis(lc, asp, arows):
    """x(a, loads) = base_a - sum_i loads[i] * G_a[i] (features are affine in the left loads)."""
    nA = asp.n
    base, G = np.zeros((nA, lc.d)), np.zeros((nA, lc.L, lc.d))
    for k in range(nA):
        for i, j, b in arows[k]:
            base[k] += lc.feature(i, j, 0, b)
            G[k, i] += lc.feature(i, j, 0, b) - lc.feature(i, j, 1, b)
    return base, G


def lin_dda_choose(basis, asp, loads, Vi, dk, nmax):
    """Two-step steer-then-observe greedy (mirrors the 1-step / 2-step columns of nl_kl_dda): value of a column =
    information on the blocking direction dk per platform round; loads are a known deterministic function of the
    action (A1), so the 2-step look-ahead is exact. Returns the first action of the best column."""
    base, G = basis
    L = G.shape[1]

    def info(ld):
        X = base - np.einsum("kid,i->kd", G, np.asarray(ld, float)[:L])
        XV = X @ Vi
        return (XV @ dk) ** 2 / (1.0 + (XV * X).sum(1))
    one = info(loads)
    best_k, best_v = int(np.argmax(one)), float(one.max())
    for k in range(asp.n):
        v = (one[k] + float(info(next_loads(np.asarray(loads, np.int64), asp, k, nmax)).max())) / 2.0
        if v > best_v + 1e-15:
            best_k, best_v = k, v
    return best_k


def run_lin(seed, noise_seed, design, want_sample):
    t_run = time.perf_counter()
    S = lin_setup(seed, noise_seed)
    env, asp, lc_d, lc_s, util, loads0, theta = S["env"], S["asp"], S["lc_d"], S["lc_s"], S["util"], S["loads0"], S["theta"]
    sigma = float(LIN_DEFAULTS["sigma"])
    h = env.handle()
    ell_d = EllipsoidSet(lc_d, sigma=sigma, delta=DELTA, S=S_BOUND, lam=LAM)
    ell_s = EllipsoidSet(lc_s, sigma=sigma, delta=DELTA, S=S_BOUND, lam=LAM)
    fam = S["fam"]
    Zd = {f: np.array([lc_d.z(p, loads0, H_LIN, util, asp) for p in pols]) for f, pols in fam.items()}
    Zs = {f: np.array([lc_s.z(p, loads0, H_LIN, util, asp) for p in pols]) for f, pols in fam.items()}
    Jt = {f: Zd[f] @ theta for f in fam}
    ds_max = max(float(np.abs(Zs[f] - Zs[f][0]).max()) for f in fam)
    rng = np.random.default_rng([seed, noise_seed, 77, DESIGNS.index(design)])
    arows = lin_action_rows(asp)
    basis = lin_action_basis(lc_d, asp, arows)
    pols = fam["lin_multi"]
    X, n = [], 0
    ckpt = {}
    ones = np.ones(env.P, np.int64)
    n_steps0 = int(env.n_steps)            # n0 instance-creation rounds (not used as evidence here)
    h.reset_to(loads0)
    ep_pol, ep_t = None, 0
    n_dda_explore = 0
    while n < N_DESIGN:
        if design == "onpolicy":
            if ep_pol is None or ep_t >= H_LIN:
                ep_pol, ep_t = pols[int(rng.integers(len(pols)))], 0
                h.reset_to(loads0)
            a = ep_pol.act(ep_t, None, None)
            ep_t += 1
        elif design == "uniform":
            a = int(rng.integers(asp.n))
        else:   # dda: greedy XY step on the most-blocking challenger of the dynamic plug-in
            if rng.random() < 0.05:
                a = int(rng.integers(asp.n))
                n_dda_explore += 1
            else:
                Z = Zd["lin_multi"]
                c = lin_certify(ell_d, Z, EPS_LIN)
                D = Z - Z[c["pi"]]
                ub = ell_d.sup_linear(D)
                ub[c["pi"]] = -np.inf
                dk = D[int(np.argmax(ub))]
                Vi = np.linalg.inv(ell_d.V)
                loads, _ = h.observable_state()
                a = lin_dda_choose(basis, asp, loads, Vi, dk, env.nmax)
        obs = h.step(int(a))
        ell_d.update(obs)
        ell_s.update(obs)
        Xo, _ = lc_d.obs_rows(obs)
        if len(Xo):
            X.append(Xo)
        n += 1
        if n in CHECKPOINTS:
            rec = {}
            for f in fam:
                for cname, ell, Z in (("static", ell_s, Zs[f]), ("dynamic", ell_d, Zd[f])):
                    c = lin_certify(ell, Z, EPS_LIN)
                    tr = float(Jt[f].max() - Jt[f][c["pi"]])
                    rec[f"{f}|{cname}"] = {"certified": c["certified"], "pi": c["pi"], "r_bar": c["r_bar"],
                                           "true_regret": tr, "false_cert": bool(c["certified"] and tr > EPS_LIN)}
            ckpt[n] = rec
    assert env.n_steps - n_steps0 == ell_d.n_rounds == ell_s.n_rounds == N_DESIGN, "step accounting mismatch"
    X = np.vstack(X)
    rows = []
    for f in fam:
        for cname, ell, Z in (("static", ell_s, Zs[f]), ("dynamic", ell_d, Zd[f])):
            c = lin_certify(ell, Z, EPS_LIN)
            tr = float(Jt[f].max() - Jt[f][c["pi"]])
            keq, kall = lin_false_equiv(ell, Z, Jt[f], EPS_LIN)
            rows.append({"kind": "hd1b", "env": "T-Lin", "family": f, "instance": seed, "noise_seed": noise_seed,
                         "design": design, "cls": cname, "n": N_DESIGN, "status": "CERTIFIED" if c["certified"] else "NEED_DATA",
                         "certified": c["certified"], "certified_policy": c["pi"],
                         "certified_policy_name": fam[f][c["pi"]].name, "true_regret": tr,
                         "false_cert": bool(c["certified"] and tr > EPS_LIN), "r_bar": c["r_bar"],
                         "n_members": len(Z), "n_bad": int((Jt[f].max() - Jt[f] > EPS_LIN).sum()),
                         "pred_fcr_static_tie": float((Jt[f].max() - Jt[f][0]) > EPS_LIN),
                         "false_equiv_pairs": keq, "eps_separated_pairs": kall,
                         "ds_max_abs": ds_max, "eps": EPS_LIN, "delta": DELTA,
                         "checkpoints": {str(k): v[f"{f}|{cname}"] for k, v in ckpt.items()},
                         "pair": list(S["pair"]), "n_resets": int(env.n_resets), "env_n_steps": int(env.n_steps - n_steps0),
                         "dda_explore": n_dda_explore})
    # FWL check
    pairs_d = []
    Zm = Zd["lin_multi"]
    g = int(np.argmax(Jt["lin_multi"]))
    for b in range(len(Zm)):
        if b != g:
            pairs_d.append((f"cons:{fam['lin_multi'][b].name}-{fam['lin_multi'][g].name}", Zm[b] - Zm[g]))
    for q in S["nat"]:
        Zn = np.array([lc_d.z(p, q.loads0, q.H, q.utility, asp) for p in q.policies])
        for b in range(1, len(Zn)):
            pairs_d.append((f"nat:{q.pid}:{b}-0", Zn[b] - Zn[0]))
    fwl = lin_fwl(X, theta, ell_s, lc_d, pairs_d, sigma)
    for r in fwl:
        r["kind_pair"] = "construction" if r["pair"].startswith("cons") else "natural"
    fwl_row = {"kind": "hd1b_fwl", "env": "T-Lin", "instance": seed, "noise_seed": noise_seed, "design": design,
               "pairs": fwl}
    sample = None
    if want_sample:
        sample = {"env": "T-Lin", "instance": seed, "design": design, "pair": list(S["pair"]),
                  "members": [p.name for p in fam["lin_multi"]], "load_exposure_S": S["S_of"],
                  "J_true": Jt["lin_multi"].round(4).tolist(),
                  "J_static_hat": (Zs["lin_multi"] @ ell_s.theta_hat()).round(4).tolist(),
                  "J_dynamic_hat": (Zd["lin_multi"] @ ell_d.theta_hat()).round(4).tolist(),
                  "rows": [{k: r[k] for k in ("family", "cls", "certified", "certified_policy_name", "true_regret",
                                              "false_cert", "r_bar")} for r in rows],
                  "fwl_construction_head": [x for x in fwl if x["kind_pair"] == "construction"][:3],
                  "fwl_natural_head": [x for x in fwl if x["kind_pair"] == "natural"][:3]}
    return rows, fwl_row, sample, time.perf_counter() - t_run


# ============================================================================================ T-NL
def nl_truth_params(tr, dev):
    t = {k: torch.as_tensor(np.asarray(tr[k], float)[None], device=dev, dtype=torch.float64)
         for k in ("alpha", "beta", "gamma", "tauL", "tauR")}
    t["psi_left"] = torch.as_tensor(np.array([[[0.0, tr["psi"]]] * 2], float), device=dev, dtype=torch.float64)
    t["lam"] = torch.as_tensor([float(tr["lam"])], device=dev, dtype=torch.float64)
    return t


def nl_precompute(seeds, streams, dev, log):
    """GPU: class J tables (static / dynamic G_1) for every timing pool + harness truth J; construction selection."""
    t0 = time.perf_counter()
    inst0 = make_offgrid_instance(seeds[0], "R0")
    asp = inst0.env.aspace
    prop = NLPropagator(2, 2, NMAX, asp, C_KNOWN, RHO_RET, device=dev)
    ncl = NLClass(G1, device=dev)
    Ps, Pd = class_torch_params("static", ncl, dev), ncl.torch_params()
    meta = {"static_equiv_maxdiff": check_static_equivalence(prop, ncl, dev), "cons": {}}
    U1 = Utility(w=np.ones(H_NL), w_ret=0.0, c_q=1.0)
    loads0, eng0 = np.zeros(4, np.int64), np.ones(4, np.int64)
    min_ds_sup = []
    for s in seeds:
        inst = make_offgrid_instance(s, "R0")
        tr = inst.truth
        assert truth_in_class(tr, "dyn_G1") and not truth_in_class(tr, "static")
        crng = np.random.default_rng([s, 601])
        tried = []
        order = [(H_NL, K_NL)] + [(8, 3), (6, 3)]
        pairs = [(int(crng.integers(2)), int(crng.integers(2)))]
        pairs += [p for p in itertools.product(range(2), range(2)) if p != pairs[0]]
        chosen = None
        for (H, k) in order:
            for (i, j) in pairs:
                pool = schedules(asp, i, j, H, k)
                U = Utility(w=np.ones(H), w_ret=0.0, c_q=1.0)
                Js = prop.j_table(Ps, pool, loads0, eng0, H, U)
                Jd = prop.j_table(Pd, pool, loads0, eng0, H, U)
                Jt = prop.j_table(nl_truth_params(tr, dev), pool, loads0, eng0, H, U)[0]
                if (H, k) == (H_NL, K_NL) and (i, j) == pairs[0]:
                    sup = np.abs(Js[:, :, None] - Js[:, None, :]).max(0)
                    np.fill_diagonal(sup, np.inf)
                    min_ds_sup.append(float(sup.min() / Jd.max()))
                reg = Jt.max() - Jt
                gs = [g for g in range(len(pool)) if reg[g] < 1e-12]
                srng = np.random.default_rng([s, 602, H, k, i, j])
                g = int(srng.choice(gs))
                c_pool = 1.0 / Jd.max()
                dom = [b for b in range(len(pool)) if reg[b] * c_pool >= BAD_MARGIN * EPS_NL
                       and (Js[:, b] - Js[:, g]).min() > 0]
                tried.append({"H": H, "k": k, "pair": [i, j], "n_dominating_bad": len(dom)})
                if dom:
                    bp = [int(x) for x in srng.choice(dom, size=min(M_NL_BAD, len(dom)), replace=False)]
                    members = [g] + bp
                    srng.shuffle(members)
                    pols = [pool[m] for m in members]
                    Jd_m = Jd[:, members]
                    c = 1.0 / float(Jd_m.max())                    # public class-max scale on the problem
                    chosen = {"H": H, "k": k, "pair": [i, j], "members": [pool[m].name for m in members],
                              "steps": [list(pool[m].steps) for m in members], "c_q": c,
                              "Js": (Js[:, members] * c).astype(np.float64), "Jd": (Jd_m * c).astype(np.float64),
                              "Jt": (Jt[members] * c).astype(np.float64),
                              "static_gap_vs_good_min": [float(((Js[:, b] - Js[:, g]) * c).min()) for b in bp],
                              "static_gap_vs_good_max": [float(((Js[:, b] - Js[:, g]) * c).max()) for b in bp],
                              "true_regret": [float((Jt.max() - Jt[m]) * c) for m in members],
                              "load_exposure_S": [load_exposure(pool[m].steps, H, NMAX) for m in members]}
                    break
            if chosen:
                break
        meta["cons"][s] = {"chosen": chosen, "tried": tried, "truth": tr}
    meta["nl_min_static_sup_gap"] = min_ds_sup
    meta["sec"] = time.perf_counter() - t0
    if torch.cuda.is_available():
        meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    del prop, ncl, Ps, Pd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"NL precompute {meta['sec']:.1f}s; min static sup-gap over same-exposure pairs {np.round(min_ds_sup, 4)}")
    return meta


def run_nl(seed, stream, design, cons, want_sample):
    torch.set_num_threads(1)
    t_run = time.perf_counter()
    inst = make_offgrid_instance(seed, "R0", stream=stream)
    env = inst.env
    asp = env.aspace
    h = env.handle()
    H = cons["H"]
    pols = [OpenLoop([asp.index([tuple(cons["pair"])]) if t in st else asp.index([]) for t in range(H)], st, H)
            for st in cons["steps"]]
    loads0, eng0 = np.zeros(4, np.int64), np.ones(4, np.int64)
    Cs, Cd = ctx("static", asp), ctx("dyn_G1", asp)
    lr_s, lr_d = SeqLRSet(Cs["prop"], Cs["LT"], DELTA), SeqLRSet(Cd["prop"], Cd["LT"], DELTA)
    Reg_s, Reg_d = regret_matrix(cons["Js"]), regret_matrix(cons["Jd"])
    Jt = np.asarray(cons["Jt"])
    noise = STREAMS[stream][0]
    rng = np.random.default_rng([seed, noise, 88, DESIGNS.index(design)])
    prop = Cd["prop"]
    n_steps0 = int(env.n_steps)            # n0 instance-creation rounds (not used as evidence here)
    h.reset_to(loads0, eng0)
    ep_pol, ep_t, n = None, 0, 0
    ckpt = {}
    modes = {}

    def cert_all():
        out = {}
        for cname, lr, Reg in (("static", lr_s, Reg_s), ("dynamic", lr_d, Reg_d)):
            m = lr.mask().numpy()
            c = certify_minimax(Reg, m, EPS_NL, TOP_M)
            sv = c["status"].value
            pi = c["pi"] if sv == "CERTIFIED" else None
            tr = float(Jt.max() - Jt[pi]) if pi is not None else None
            out[cname] = {"status": sv, "certified": sv == "CERTIFIED", "pi": pi,
                          "r_bar": float(c["r_bar"]) if np.isfinite(c["r_bar"]) else None, "true_regret": tr,
                          "false_cert": bool(pi is not None and tr > EPS_NL), "set_size": int(m.sum()),
                          "mle": lr.mle(), "minimax_pi": c["pi"]}
        return out

    while n < N_DESIGN:
        if design == "onpolicy":
            if ep_pol is None or ep_t >= H:
                ep_pol, ep_t = pols[int(rng.integers(len(pols)))], 0
                h.reset_to(loads0, eng0)
            a = ep_pol.act(ep_t, None, None)
            ep_t += 1
            modes["onpolicy"] = modes.get("onpolicy", 0) + 1
        elif design == "uniform":
            a = int(rng.integers(asp.n))
            modes["uniform"] = modes.get("uniform", 0) + 1
        else:
            mask = lr_d.mask().numpy()
            cert = certify_minimax(Reg_d, mask, EPS_NL, TOP_M)
            kh = lr_d.mle()
            blk = cert["blocking"] or [int(i) for i in np.flatnonzero(mask)[:TOP_M]]
            margins = LOG_THR - lr_d.log_ratio().numpy()[blk]
            code = prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, Cd["py"][kh:kh + 1], Cd["pe"][kh:kh + 1], Cd["LT_np"][kh], Cd["py"][blk],
                                 Cd["pe"][blk], margins, Cd["inc"], prop, Cd["legal"], rng)
            modes[info["mode"]] = modes.get(info["mode"], 0) + 1
        obs = h.step(int(a))
        lr_s.update(obs)
        lr_d.update(obs)
        n += 1
        if n in CHECKPOINTS:
            ckpt[n] = cert_all()
    assert env.n_steps - n_steps0 == lr_s.n_rounds == lr_d.n_rounds == N_DESIGN, "step accounting mismatch"
    fin = ckpt[N_DESIGN]
    rows = []
    for cname in ("static", "dynamic"):
        f = fin[cname]
        Jc = cons["Js"] if cname == "static" else cons["Jd"]
        g = int(np.argmax(Jt))
        mle = f["mle"]
        rows.append({"kind": "hd1b", "env": "T-NL", "family": "nl_dominated", "instance": seed, "noise_seed": noise,
                     "stream": stream, "design": design, "cls": cname, "n": N_DESIGN, "status": f["status"],
                     "certified": f["certified"], "certified_policy": f["pi"],
                     "certified_policy_name": cons["members"][f["pi"]] if f["pi"] is not None else None,
                     "true_regret": f["true_regret"], "false_cert": f["false_cert"], "r_bar": f["r_bar"],
                     "set_size": f["set_size"], "n_members": len(pols),
                     "n_bad": int((Jt.max() - Jt > EPS_NL).sum()), "eps": EPS_NL, "delta": DELTA,
                     "mle_gap_bias": [float((Jc[mle, b] - Jc[mle, g]) - (Jt[b] - Jt[g])) for b in range(len(pols))
                                      if b != g],
                     "checkpoints": {str(k): v[cname] for k, v in ckpt.items()},
                     "pair": cons["pair"], "H": H, "k": cons["k"], "design_modes": modes,
                     "n_resets": int(env.n_resets), "env_n_steps": int(env.n_steps - n_steps0)})
    sample = None
    if want_sample:
        sample = {"env": "T-NL", "instance": seed, "design": design, "pair": cons["pair"], "H": H,
                  "members": cons["members"], "load_exposure_S": cons["load_exposure_S"],
                  "J_true": Jt.round(4).tolist(), "true_regret": np.round(cons["true_regret"], 4).tolist(),
                  "static_gap_vs_good_min": np.round(cons["static_gap_vs_good_min"], 4).tolist(),
                  "J_static_at_static_mle": np.asarray(cons["Js"][fin["static"]["mle"]]).round(4).tolist(),
                  "J_dynamic_at_dynamic_mle": np.asarray(cons["Jd"][fin["dynamic"]["mle"]]).round(4).tolist(),
                  "final": fin}
    return rows, None, sample, time.perf_counter() - t_run


def run_job(job):
    try:
        if job[0] == "lin":
            return run_lin(*job[1:]) + (None, job)
        return run_nl(*job[1:]) + (None, job)
    except Exception as e:  # noqa: BLE001
        return [], None, None, 0.0, {"job": [str(x) for x in job[:4]], "error": repr(e), "tb": traceback.format_exc()}, job


# ============================================================================================ analysis
def cell(rows):
    cert = [r for r in rows if r["certified"]]
    fc = [r for r in cert if r["false_cert"]]
    sc = {}
    for r in rows:
        sc[r["status"]] = sc.get(r["status"], 0) + 1
    out = {"n": len(rows), "status_counts": sc, "certs": len(cert), "false_certs": len(fc),
           "completion": len(cert) / len(rows) if rows else None,
           "fcr": len(fc) / len(cert) if cert else None, "fcr_cp95": cp(len(fc), len(cert)),
           "false_per_run": len(fc) / len(rows) if rows else None}
    if rows and "false_equiv_pairs" in rows[0]:
        ke, ka = sum(r["false_equiv_pairs"] for r in rows), sum(r["eps_separated_pairs"] for r in rows)
        out["false_equiv_rate"] = ke / ka if ka else None
        out["false_equiv_pairs"] = [ke, ka]
        out["pred_fcr_static_tie_mean"] = float(np.mean([r["pred_fcr_static_tie"] for r in rows]))
    ck = {}
    for k in CHECKPOINTS:
        cc = [r["checkpoints"][str(k)] for r in rows if str(k) in r["checkpoints"]]
        ce = [c for c in cc if c["certified"]]
        ck[str(k)] = {"certs": len(ce), "false": sum(c["false_cert"] for c in ce),
                      "fcr": (sum(c["false_cert"] for c in ce) / len(ce)) if ce else None}
    out["by_checkpoint"] = ck
    return out


def analyse(rows, fwl_rows):
    out = {"by_cell": {}}
    keys = sorted({(r["env"], r["family"], r["design"], r["cls"]) for r in rows})
    for k in keys:
        rr = [r for r in rows if (r["env"], r["family"], r["design"], r["cls"]) == k]
        out["by_cell"]["|".join(k)] = cell(rr)
    sfd, dfd = {}, {}
    for env, fam in (("T-Lin", "lin_multi"), ("T-Lin", "lin_pair"), ("T-NL", "nl_dominated")):
        for d in DESIGNS:
            s = out["by_cell"].get(f"{env}|{fam}|{d}|static")
            dd = out["by_cell"].get(f"{env}|{fam}|{d}|dynamic")
            if s:
                sfd[f"{fam}|{d}"] = {"fcr": s["fcr"], "cp95": s["fcr_cp95"], "certs": s["certs"], "n": s["n"],
                                     "completion": s["completion"], "false_equiv_rate": s.get("false_equiv_rate")}
            if dd:
                dfd[f"{fam}|{d}"] = {"fcr": dd["fcr"], "cp95": dd["fcr_cp95"], "certs": dd["certs"], "n": dd["n"],
                                     "completion": dd["completion"]}
    out["static_fcr_by_design"] = sfd
    out["dynamic_fcr"] = dfd
    dyn = [r for r in rows if r["cls"] == "dynamic"]
    dc = [r for r in dyn if r["certified"]]
    out["dynamic_fcr_pooled"] = {"false": sum(r["false_cert"] for r in dc), "certs": len(dc), "n": len(dyn),
                                 "fcr": (sum(r["false_cert"] for r in dc) / len(dc)) if dc else None,
                                 "cp95": cp(sum(r["false_cert"] for r in dc), len(dc))}
    # FWL
    allp = [p for fr in fwl_rows for p in fr["pairs"]]
    cons = [p for p in allp if p["kind_pair"] == "construction"]
    nat = [p for p in allp if p["kind_pair"] == "natural"]
    nat_id = [p for p in nat if p["ds_in_rowspace"]]
    zz = np.array([p["z_measured_vs_expected"] for p in allp if p["z_measured_vs_expected"] is not None])
    by_design = {}
    for d in DESIGNS:
        pp = [p for fr in fwl_rows if fr["design"] == d for p in fr["pairs"] if p["kind_pair"] == "construction"]
        by_design[d] = {"n_pairs": len(pp), "b_pred_mean": float(np.mean([p["b_pred_fwl"] for p in pp])) if pp else None,
                        "true_gap_mean": float(np.mean([p["true_gap"] for p in pp])) if pp else None,
                        "max_abs_b_plus_gap": float(max(abs(p["b_pred_fwl"] + p["true_gap"]) for p in pp)) if pp else None}
    out["fwl"] = {
        "n_pairs": len(allp), "n_construction": len(cons), "n_natural": len(nat), "n_natural_identified": len(nat_id),
        "construction_ds_norm_max": float(max(p["ds_norm"] for p in cons)) if cons else None,
        "construction_static_gap_estimate_max_abs": float(max(abs(p["bias_ridge_measured"] + p["true_gap"]) for p in cons)) if cons else None,
        "construction_b_equals_minus_true_gap_max_err": float(max(abs(p["b_pred_fwl"] + p["true_gap"]) for p in cons)) if cons else None,
        "construction_by_design": by_design,
        "identity_err_max_identified": float(max(p["err_identity"] for p in cons + nat_id)) if (cons or nat_id) else None,
        "measured_vs_fwl_abs_err_median": float(np.median([p["err_measured_vs_fwl"] for p in cons + nat_id])) if (cons or nat_id) else None,
        "measured_vs_fwl_abs_err_p90": float(np.quantile([p["err_measured_vs_fwl"] for p in cons + nat_id], 0.9)) if (cons or nat_id) else None,
        "natural_abs_bias_median": float(np.median([abs(p["b_pred_fwl"]) for p in nat_id])) if nat_id else None,
        "z_abs_le_2_share": float(np.mean(np.abs(zz) <= 2)) if len(zz) else None,
        "z_mean": float(zz.mean()) if len(zz) else None, "z_sd": float(zz.std()) if len(zz) else None,
        "note": "identity = noise-free static OLS bias vs closed form (exact when d^s in rowspace(X_s)); measured = "
                "realised ridge (lam=1) static estimate, z-scored against its exact Gaussian sd around the ridge "
                "expectation; construction pairs have d^s = 0 so b = -<d^n, gamma*> = -true gap for every design"}
    return out


def plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    sf, df = summary["static_fcr_by_design"], summary["dynamic_fcr"]
    keys = [k for k in sf]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(keys))
    sv = [sf[k]["fcr"] if sf[k]["fcr"] is not None else np.nan for k in keys]
    dv = [df[k]["fcr"] if k in df and df[k]["fcr"] is not None else np.nan for k in keys]
    ax.bar(x - 0.2, sv, 0.4, label="static class")
    ax.bar(x + 0.2, dv, 0.4, label="dynamic class")
    ax.axhline(0.8, ls="--", c="k", lw=0.8)
    ax.axhline(DELTA, ls=":", c="r", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("FCR at n = 2000")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("HD1b: same-exposure, different-timing construction")
    fig.tight_layout()
    p = out_dir / "hd1b_fcr_by_design.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(WS))


# ============================================================================================ main
def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    # round-3 replication overrides (r3_replicate_hd1b_hd2): seed source + output location; defaults = round-2 run
    ap.add_argument("--seeds", default=None, help="instance seeds for BOTH T-Lin and T-NL, 'a-b' inclusive or 'a,b,c'")
    ap.add_argument("--noise-seeds", default=None, help="T-Lin noise seeds, comma list")
    ap.add_argument("--nl-streams", default=None, help="T-NL stream indices, comma list")
    ap.add_argument("--out-dir", default=None, help="output directory (default results/{pilots,full}/TASK)")
    ap.add_argument("--no-protocol", action="store_true",
                    help="do not write PID/PROGRESS/DONE/gpu_profile (a wrapper task owns them)")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    quiet = args.smoke or args.no_protocol
    out_dir = Path(args.out_dir) if args.out_dir else \
        RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_path, fwl_path = out_dir / "results.jsonl", out_dir / "fwl.jsonl"
    if not quiet:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def progress(step, total, phase, metric=None):
        if quiet:
            return
        (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
            "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
            "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))

    def mark_done(status, summ):
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
        (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summ,
                                                           "final_progress": fp, "timestamp": datetime.now().isoformat()}))

    if pilot:
        lin_seeds, nl_seeds, lin_noise, nl_streams = PILOT_LIN_SEEDS, PILOT_NL_SEEDS, [42], [0]
        assert max(lin_seeds + nl_seeds) < 10000
    else:
        lock = PRE
        assert lock.get("status") == "locked", "full mode requires a locked prereg (r2_prereg_lock)"
        lin_seeds = nl_seeds = [FULL_OFFSET + 900 + s for s in FULL_SEEDS]
        lin_noise, nl_streams = list(FULL_NOISE), [0, 1, 2]
    if args.seeds:
        lin_seeds = nl_seeds = parse_seeds(args.seeds)
        if pilot:
            assert max(lin_seeds) < 10000, "pilot mode must not touch evaluation seeds"
    if args.noise_seeds:
        lin_noise = [int(x) for x in args.noise_seeds.split(",")]
    if args.nl_streams:
        nl_streams = [int(x) for x in args.nl_streams.split(",")]
    if args.smoke:
        lin_seeds, nl_seeds = lin_seeds[:1], nl_seeds[:1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "delta": DELTA, "eps": {"T-Lin": EPS_LIN, "T-NL": EPS_NL},
               "n_design": N_DESIGN, "designs": list(DESIGNS), "lin_seeds": [lin_seeds[0], lin_seeds[-1]],
               "nl_seeds": [nl_seeds[0], nl_seeds[-1]], "lin_noise_seeds": lin_noise, "nl_streams": nl_streams,
               "construction": {"T-Lin": f"pair (i,j) truth-free; H={H_LIN}, k={K_LIN}; loads0=0; w=1; c_q=1/(y_scale H); "
                                         f"1 good + <= {M_LIN_BAD} bad (true regret >= {BAD_MARGIN} eps), shuffled; "
                                         f"gamma prior {LIN_PRIOR['gamma']}",
                                "T-NL": f"R0 truth (G_1); H={H_NL}, k={K_NL} (fallbacks (8,3),(6,3)); loads0=0, engaged0=1; "
                                        f"w=1, w_ret=0; 1 good + <= {M_NL_BAD} bad that strictly dominate it under every "
                                        f"static theta (continuous C')"},
               "claim_scope": "worst-case construction only (Prop. C'); not an empirical claim",
               "eval_seeds_touched": not pilot, "generator_hash": generator_hash(), "concurrent_run": True,
               "note": "并发运行（4 槽并行，每任务 4 worker）；计时偏高", "lock_status": PRE.get("status")}
    try:
        log(f"start {TASK} mode={args.mode} device={dev}; Lin seeds {lin_seeds[0]}-{lin_seeds[-1]} x noise {lin_noise}; "
            f"NL seeds {nl_seeds[0]}-{nl_seeds[-1]} x streams {nl_streams}; designs {DESIGNS}; n={N_DESIGN}")
        progress(1, 4, "gpu: NL class J tables + construction selection")
        meta = nl_precompute(nl_seeds, nl_streams, dev, log)
        assert meta["static_equiv_maxdiff"] < 1e-9, "static reparametrisation mismatch"
        summary["nl_precompute_sec"] = meta["sec"]
        summary["static_equiv_maxdiff"] = meta["static_equiv_maxdiff"]
        summary["nl_min_static_sup_gap_over_same_exposure_pairs"] = meta["nl_min_static_sup_gap"]
        summary["nl_exact_ds0_feasible"] = bool(min(meta["nl_min_static_sup_gap"]) < 1e-9)
        summary["nl_construction"] = {str(s): ({k: v for k, v in c["chosen"].items() if k not in ("Js", "Jd", "Jt")}
                                               if c["chosen"] else {"chosen": None}) | {"tried": c["tried"]}
                                      for s, c in meta["cons"].items()}
        if not quiet and meta.get("gpu_name"):
            (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
                "gpu_name": meta["gpu_name"], "vram_total_mb": meta["vram_total_mb"], "max_batch_size": None,
                "vram_used_mb": meta["gpu_peak_mb"], "utilization_pct": 100 * meta["gpu_peak_mb"] / meta["vram_total_mb"],
                "note": "GPU only for truth-free NL class J tables over 13824 thetas x <= 70 timings (one full-class "
                        "batch, exact propagator; batch probing not applicable); learners CPU; shared 4090"}))
        jobs = [("lin", s, ns, d, s == lin_seeds[0] or d == "dda" and s == lin_seeds[1 % len(lin_seeds)])
                for s in lin_seeds for ns in lin_noise for d in DESIGNS]
        nl_ok = [s for s in nl_seeds if meta["cons"][s]["chosen"]]
        jobs = [("nl", s, st, d, meta["cons"][s]["chosen"], s == nl_ok[0]) for s in nl_ok for st in nl_streams
                for d in DESIGNS] + jobs
        summary["nl_construction_infeasible"] = [s for s in nl_seeds if not meta["cons"][s]["chosen"]]
        progress(2, 4, "cpu: designs x certifiers", {"jobs": len(jobs)})
        from joblib import Parallel, delayed
        rows, fwl_rows, samples, errors, tsec = [], [], [], [], {}
        gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_job)(j) for j in jobs)
        nd = 0
        with open(res_path, "w") as fh, open(fwl_path, "w") as fw:
            for rr, fr, ss, sec, ee, job in gen:
                nd += 1
                if ee:
                    errors.append(ee)
                    log(f"ERROR {ee['job']}: {ee['error']}\n{ee['tb']}")
                    continue
                for r in rr:
                    fh.write(json.dumps(r, default=float) + "\n")
                rows += rr
                if fr:
                    fw.write(json.dumps(fr, default=float) + "\n")
                    fwl_rows.append(fr)
                if ss:
                    samples.append(ss)
                tsec.setdefault(job[0] + "|" + job[3], []).append(sec)
                st = {r["cls"] + "/" + r["family"]: (r["status"], r["false_cert"]) for r in rr}
                log(f"job {nd}/{len(jobs)} {job[0]} inst {job[1]} {job[3]}: {sec:.1f}s {st}")
                progress(2, 4, "cpu: designs x certifiers", {"jobs_done": nd, "jobs": len(jobs)})
        (out_dir / "samples" / "samples.json").write_text(json.dumps(samples, indent=1, default=float))
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        progress(3, 4, "summary")
        an = analyse(rows, fwl_rows)
        summary.update(an)
        summary["crashes"] = len(errors)
        summary["n_rows"] = len(rows)
        summary["n_runs"] = nd - len(errors)
        summary["lin_ds_max_abs"] = float(max(r["ds_max_abs"] for r in rows if r["env"] == "T-Lin"))
        summary["timing"] = {k: {"mean_s": float(np.mean(v)), "n": len(v)} for k, v in tsec.items()}
        if pilot:
            per_lin = float(np.mean([np.mean(v) for k, v in tsec.items() if k.startswith("lin")]))
            per_nl = float(np.mean([np.mean(v) for k, v in tsec.items() if k.startswith("nl")])) if nl_ok else 0.0
            cpu_full = 30 * 3 * 3 * (per_lin + per_nl)
            summary["timing_projection"] = {"per_run_s": {"lin": per_lin, "nl": per_nl},
                                            "projected_full_cpu_sec": cpu_full,
                                            "projected_full_wall_min_at_4_workers": cpu_full / 4 / 60,
                                            "note": "30 instances x 3 designs x 3 noise seeds, Lin + NL; concurrent run"}
        sf = an["static_fcr_by_design"]
        lin_ok = all((sf.get(f"lin_multi|{d}", {}).get("fcr") or 0) >= 0.8 for d in DESIGNS)
        nl_ok_ = all((sf.get(f"nl_dominated|{d}", {}).get("fcr") or 0) >= 0.8 for d in DESIGNS)
        dfp = an["dynamic_fcr_pooled"]
        fw = an["fwl"]
        pc = {"zero_crashes": len(errors) == 0,
              "lin_ds_zero": summary["lin_ds_max_abs"] < 1e-9,
              "static_fcr_ge_0.8_all_designs_T-Lin": lin_ok,
              "static_fcr_ge_0.8_all_designs_T-NL_continuous": nl_ok_,
              "dynamic_fcr_le_delta": dfp["fcr"] is not None and dfp["fcr"] <= DELTA,
              "fwl_identity_exact": fw["identity_err_max_identified"] is not None and fw["identity_err_max_identified"] < 1e-8}
        summary["pass_criteria"] = pc
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        summary["figure"] = plot(summary, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for k, v in an["by_cell"].items():
            log(f"{k:40s} {v['status_counts']} FCR {v['false_certs']}/{v['certs']} cp95 {v['fcr_cp95']} "
                f"false-equiv {v.get('false_equiv_rate')}")
        log(f"FWL: {json.dumps({k: v for k, v in fw.items() if k != 'note'}, default=float)}")
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}; static FCR "
                             f"{ {k: (None if v['fcr'] is None else round(v['fcr'], 3)) for k, v in sf.items()} }; "
                             f"dynamic pooled {dfp['false']}/{dfp['certs']}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
