"""h1_kappa_law_lin: kappa prediction law, Omega(H^2) construction and static vanishing on E1-Lin.

Sequential (eps, delta)-certification on E1-Lin with a generative single-observation interaction library
(atom = one platform round that matches one pair (i, j) at left load n_i with incentive b; the harness puts the
platform into the required load state; resets are not charged and are identical for every evidence type).

Evidence types (all share the same initial data n0=20, the same library, the same stopping rule form
"max_k UB_k <= eps" with pi_hat = plug-in argmax, the same Abbasi-Yadkori radius family):
  joint        UB_k = <d_k, th> + sqrt(beta) ||d_k||_{V^-1}
  rect_<P>     B6-sup: UB_k = <d_k, th> + sqrt(beta) sum_p ||d_kp||_{V^-1}   (P in time / participant / policy)
  cpe          B6-CPE: time-inhomogeneous class theta_h (one copy per step), joint width across policies in the
               lifted space (dimension d*H, radius with S*sqrt(H)); an atom is attributed to one step h.
  b3_rage      B3: transductive RAGE (Fiez et al. 2019), phase-wise fresh least squares + elimination.
Designs: each evidence uses its own Fiez-style phased plug-in optimal design (mirror-descent min-max design,
recomputed at doubling epochs, 2% uniform mixing, largest-remainder tracking); side table: shared uniform design.
Oracle rho*: 2 sigma^2 log(1/2.4 delta) min_{eps-good a} min_xi max_k ||z_a - z_k||^2_{A^-1} / (J_a - J_k + eps)^2.

Observation sampling: per-atom sufficient statistics (count, sum) are drawn from the exact Gaussian law of the
platform (harness side). This is distributionally identical to repeated env.step() calls; a validation block
compares it with real env.step() observations and runs a few end-to-end joint certifications through env.step().

Usage: run_h1_kappa_law_lin.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.streams.generator import LIN_DEFAULTS, generator_hash, make_lin_instance  # noqa: E402
from dsswm.streams.utilities import Utility  # noqa: E402
from run_h2_kappa_distribution import (Library, analyse_problem, lin_contrib, lin_library,  # noqa: E402
                                       optimise, reduced_basis)

TASK = "h1_kappa_law_lin"
EPS, DELTA, LAM = 0.05, 0.05, 1.0
PARTS = ("time", "participant", "policy")
PRE = json.loads((WS / "plan" / "prereg_lock.json").read_text())
# v1 lock: top-level "E1-Lin"; v2 lock: under "round0"; v3 lock carries neither -> the frozen round-0 value
# (dsswm.baselines.lin_rage.S_LIN, identical number)
S_BOUND = float((PRE.get("E1-Lin") or (PRE.get("round0") or {}).get("E1-Lin")
                 or {"ellipsoid_S": 2.8071337695236402})["ellipsoid_S"])


# ----------------------------------------------------------------------------------------------- protocol files
def progress(res_dir, step, total, phase, metric=None):
    (res_dir / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(res_dir, status, summary):
    pid = res_dir / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = res_dir / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (res_dir / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                      "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ----------------------------------------------------------------------------------------------- construction
class FixedSchedule:
    """Deterministic open-loop schedule: match pair (0,0) at the listed steps, otherwise idle."""

    def __init__(self, name, steps, aspace, H):
        self.family, self.name_, self.steps, self.aspace, self.H = "schedule", name, set(steps), aspace, H
        self.a_on = aspace.index([(0, 0)])
        self.a_off = aspace.index([])

    @property
    def name(self):
        return self.name_

    def act(self, t, loads, engaged):
        return self.a_on if t in self.steps else self.a_off


def construction_policies(aspace, H):
    """'Same exposure, different timing'. Returns (policies, loads0).
    H >= 4: A = {0,1} U {4,6,..,H-2}, B = {1,3,..,H-1}, load0 = 0  ->  d = -e_gamma0 (one extra unit of load).
    H = 2 : A = {0}, B = {1}, load0(participant 0) = 1          ->  d = -e_gamma0.
    Per-step blocks are +-(e_alpha0 + e_beta0) at Theta(H) steps, which cancel in the sum."""
    if H == 2:
        A, B, l0 = [0], [1], 1
    else:
        A, B, l0 = [0, 1] + list(range(4, H - 1, 2)), list(range(1, H, 2)), 0
    assert len(A) == len(B)
    loads0 = np.zeros(6, np.int64)
    loads0[0] = l0
    return [FixedSchedule(f"A{A}", A, aspace, H), FixedSchedule(f"B{B}", B, aspace, H)], loads0


# ----------------------------------------------------------------------------------------------- problem prep
def prep_problem(env, lc, policies, loads0, H, util, init_obs, theta):
    """Everything the sequential runs need (full coordinates d) + kappa analysis (reduced coordinates)."""
    rows, names = lin_library(lc, 2, lc.static)              # Type-1 library: b in {0,1}
    rows = np.asarray(rows, float)
    Q = reduced_basis(rows)                                   # (d, r)
    lib = Library(rows @ Q, np.ones(len(rows)), np.arange(len(rows)), len(rows), Q)
    mu = np.array([env._mean(i, j, n, b) for (i, j, n, b) in names])   # harness side (truth)
    CL = []
    for p in policies:
        a, _ = lin_contrib(lc, p, loads0, H, util, env.aspace)
        CL.append(a)
    CL = np.array(CL)                                         # (n, H, L, d)
    Z = CL.sum((1, 2))
    J = Z @ theta
    n = len(policies)
    T = CL.sum(2)                                             # (n, H, d)
    P = CL.sum(1)                                             # (n, L, d)
    D = {}
    D["joint"] = (Z[None, :, :] - Z[:, None, :])[:, :, None, :]                 # [ref, k, 1, d]
    D["time"] = T[None] - T[:, None]                                           # [ref, k, H, d]
    D["participant"] = P[None] - P[:, None]                                    # [ref, k, L, d]
    D["policy"] = np.stack([np.broadcast_to(Z[None, :, :], (n, n, Z.shape[1])),
                            -np.broadcast_to(Z[:, None, :], (n, n, Z.shape[1]))], 2)  # [ref, k, 2, d]
    X0, y0 = [], []
    t_of = []
    for t, obs in enumerate(init_obs):
        Xo, yo = lc.obs_rows(obs)
        X0.append(Xo); y0.append(yo); t_of += [t] * len(yo)
    X0, y0, t_of = np.vstack(X0), np.concatenate(y0), np.array(t_of)
    resid = float(np.abs(Z - (Z @ Q) @ Q.T).max())
    return dict(rows=rows, names=names, Q=Q, lib=lib, mu=mu, CL=CL, Z=Z, J=J, D=D, H=H, n=n, d=lc.d,
                X0=X0, y0=y0, t0=t_of, sigma=env._sigma, range_resid=resid)


def lifted(pp):
    """B6-CPE lifted problem: theta_lift = (theta_1..theta_H), atoms (h, a)."""
    H, d, rows, Q = pp["H"], pp["d"], pp["rows"], pp["Q"]
    M, r = rows.shape[0], Q.shape[1]
    rows_l = np.zeros((M * H, d * H))
    for h in range(H):
        rows_l[h * M:(h + 1) * M, h * d:(h + 1) * d] = rows
    Ql = np.zeros((d * H, r * H))
    for h in range(H):
        Ql[h * d:(h + 1) * d, h * r:(h + 1) * r] = Q
    lib = Library(rows_l @ Ql, np.ones(M * H), np.arange(M * H), M * H, Ql)
    T = pp["CL"].sum(2)                                        # (n, H, d)
    Zl = T.reshape(pp["n"], H * d)
    Dl = (Zl[None, :, :] - Zl[:, None, :])[:, :, None, :]
    X0 = np.zeros((len(pp["y0"]), d * H))
    for m, (x, t) in enumerate(zip(pp["X0"], pp["t0"])):
        h = int(t) % H
        X0[m, h * d:(h + 1) * d] = x
    return dict(rows=rows_l, lib=lib, mu=np.tile(pp["mu"], H), Z=Zl, D=Dl, X0=X0, y0=pp["y0"], dd=d * H,
                S=S_BOUND * np.sqrt(H))


def ev_view(pp, ev):
    if ev == "cpe":
        return lifted(pp)
    return dict(rows=pp["rows"], lib=pp["lib"], mu=pp["mu"], Z=pp["Z"], D=pp["D"][ev], X0=pp["X0"], y0=pp["y0"],
                dd=pp["d"], S=S_BOUND)


# ----------------------------------------------------------------------------------------------- sequential certification
def geom_grid(T_max, ratio=1.01):
    g, x = [0], 1.0
    while g[-1] < T_max:
        nxt = max(g[-1] + 1, int(np.ceil(x)))
        g.append(min(nxt, int(T_max)))
        x *= ratio
    return g


def largest_remainder(t, m):
    base = np.floor(t).astype(np.int64)
    r = int(m - base.sum())
    if r > 0:
        idx = np.argsort(-(t - base))[:r]
        base[idx] += 1
    return base


def run_seq(ev_name, v, J, sigma, eps, design_mode, seed, T_max, grid_ratio=1.01, iters=300, n_phase0=32):
    rng = np.random.default_rng(seed)
    X, lib, mu, Z, D, dd = v["rows"], v["lib"], v["mu"], v["Z"], v["D"], v["dd"]
    M = X.shape[0]
    V = LAM * np.eye(dd) + v["X0"].T @ v["X0"]
    bvec = v["X0"].T @ v["y0"]
    counts = np.zeros(M, np.int64)
    phase_counts = np.zeros(M, np.int64)
    n = 0
    xi = np.full(M, 1.0 / M)
    next_phase = 0
    n_designs = 0
    t0 = time.perf_counter()
    log2dl = 2 * np.log(1 / DELTA)

    def certify():
        th = np.linalg.solve(V, bvec)
        vals = Z @ th
        ref = int(np.argmax(vals))
        Vi = np.linalg.inv(V)
        Dr = D[ref]                                   # (n, nb, dd)
        w = np.sqrt(np.maximum(np.einsum("kbi,ij,kbj->kb", Dr, Vi, Dr), 0.0)).sum(1)
        _, logdet = np.linalg.slogdet(V)
        sb = sigma * np.sqrt(log2dl + logdet - dd * np.log(LAM)) + np.sqrt(LAM) * v["S"]
        ub = (vals - vals[ref]) + sb * w
        ub[ref] = 0.0
        return ref, ub, th, sb

    ref, ub, th, sb = certify()
    traj = []
    for n_t in geom_grid(T_max, grid_ratio):
        if n_t > n:
            m = n_t - n
            if design_mode == "uniform":
                tgt = (phase_counts.sum() + m) * xi
            else:
                tgt = (phase_counts.sum() + m) * xi
            deficit = np.maximum(tgt - phase_counts, 0.0)
            if deficit.sum() <= 0:
                deficit = xi.copy()
            inc = largest_remainder(deficit / deficit.sum() * m, m)
            c = inc[inc > 0]
            idx = np.nonzero(inc)[0]
            s = rng.normal(c * mu[idx], sigma * np.sqrt(c))
            Xi = X[idx]
            V += (Xi.T * c) @ Xi
            bvec += Xi.T @ s
            counts += inc
            phase_counts += inc
            n = n_t
            ref, ub, th, sb = certify()
        if len(traj) < 60 and (n == 0 or n >= 1.3 * (traj[-1][0] if traj else 1)):
            traj.append((int(n), float(ub.max()), int(ref)))
        if ub.max() <= eps:
            return dict(N=int(n), censored=False, pi_hat=ref, regret=float(J.max() - J[ref]),
                        false_cert=bool(J.max() - J[ref] > eps), n_designs=n_designs,
                        wall_s=time.perf_counter() - t0, sqrt_beta=float(sb), traj=traj)
        if design_mode == "phased" and n >= next_phase:
            active = [k for k in range(len(Z)) if k != ref and ub[k] > eps]
            Dhat = np.array([max(float((Z[ref] - Z[k]) @ th), 0.0) for k in active]) + eps
            Q = lib.Q
            blocks = []
            for k in active:
                B = D[ref][k] @ Q
                keep = np.abs(B).max(1) > 1e-13
                blocks.append(B[keep] if keep.any() else B[:1])
            xs, _ = optimise(lib, blocks, Dhat, iters=iters if n_designs else 2 * iters, xi0=None)
            xi = 0.98 * xs + 0.02 / M
            phase_counts[:] = 0
            n_designs += 1
            next_phase = n_phase0 if n == 0 else 2 * n
        if n >= T_max:
            break
    return dict(N=int(n), censored=True, pi_hat=ref, regret=float(J.max() - J[ref]), false_cert=False,
                n_designs=n_designs, wall_s=time.perf_counter() - t0, sqrt_beta=float(sb), traj=traj)


def run_rage(pp, eps, seed, T_max, iters=400, omega=0.1):
    """B3: transductive RAGE (Fiez et al. 2019) with fresh data per phase; eps-stop when 2^-l <= eps/2."""
    rng = np.random.default_rng(seed)
    lib, mu, Z, J, sigma = pp["lib"], pp["mu"], pp["Z"], pp["J"], pp["sigma"]
    G = lib.G
    Zr = Z @ pp["Q"]
    nZ = len(Z)
    S = list(range(nZ))
    total, l = 0, 1
    t0 = time.perf_counter()
    rec = int(np.argmax(Zr @ np.zeros(Zr.shape[1])))
    while True:
        pairs = [(a, b) for ii, a in enumerate(S) for b in S[ii + 1:] if np.abs(Zr[a] - Zr[b]).max() > 1e-13]
        if not pairs:
            rec = S[0]
            break
        blocks = [(Zr[a] - Zr[b])[None] for a, b in pairs]
        xi, f = optimise(lib, blocks, np.ones(len(blocks)), iters=iters)
        N_l = int(np.ceil(2 * sigma ** 2 * (2 ** (l + 1)) ** 2 * f * (1 + omega) * np.log(4 * l ** 2 * nZ ** 2 / DELTA)))
        N_l = max(N_l, lib.r + 1)
        if total + N_l > T_max:
            return dict(N=int(T_max), censored=True, pi_hat=rec, regret=float(J.max() - J[rec]), false_cert=False,
                        phases=l, wall_s=time.perf_counter() - t0)
        cnt = largest_remainder(N_l * xi, N_l)
        idx = np.nonzero(cnt)[0]
        c = cnt[idx]
        s = rng.normal(c * mu[idx], sigma * np.sqrt(c))
        A = (G[idx].T * c) @ G[idx] + 1e-9 * np.eye(lib.r)
        th = np.linalg.solve(A, G[idx].T @ s)
        total += N_l
        vals = Zr @ th
        best = max(vals[k] for k in S)
        S = [k for k in S if best - vals[k] < 2.0 ** (-l)]
        rec = max(S, key=lambda k: vals[k])
        if len(S) == 1 or 2.0 ** (-l) <= eps / 2:
            break
        l += 1
    return dict(N=int(total), censored=False, pi_hat=int(rec), regret=float(J.max() - J[rec]),
                false_cert=bool(J.max() - J[rec] > eps), phases=l, wall_s=time.perf_counter() - t0)


def oracle_rho(pp, eps, iters=800):
    lib, Zr, J, sigma = pp["lib"], pp["Z"] @ pp["Q"], pp["J"], pp["sigma"]
    best = np.inf
    for a in range(len(J)):
        if J[a] < J.max() - eps:
            continue
        ks = [k for k in range(len(J)) if k != a and np.abs(Zr[a] - Zr[k]).max() > 1e-13]
        if not ks:
            return 0.0
        blocks = [(Zr[a] - Zr[k])[None] for k in ks]
        gaps = np.array([J[a] - J[k] + eps for k in ks])
        _, val = optimise(lib, blocks, gaps, iters=iters)
        best = min(best, val)
    return float(2 * sigma ** 2 * np.log(1 / (2.4 * DELTA)) * best)


# ----------------------------------------------------------------------------------------------- jobs
def solve_problem(job):
    """job: dict(kind, seed, H, k, static, T_max, evs, designs, noise)."""
    t0 = time.perf_counter()
    kind, seed, H = job["kind"], job["seed"], job["H"]
    try:
        if kind == "construction":
            inst = make_lin_instance(seed, ptypes=(1,), H_choices=(max(H, 1),), K=1,
                                     prior={"gamma": (0.3, 0.6)})
            env = inst.env
            lc = LinClass(3, 3, env.aspace.nb, nmax=3, static=False)
            pols, loads0 = construction_policies(env.aspace, H)
            util = Utility(w=np.ones(H), w_ret=0.0, c_q=1.0)
            pid = f"constr{seed}_H{H}"
        else:
            inst = make_lin_instance(seed, ptypes=(1,), H_choices=(H,), K=job["K"], static=job["static"])
            env = inst.env
            q = inst.problems[job["k"]]
            lc = LinClass(3, 3, env.aspace.nb, nmax=3, static=job["static"])
            pols, loads0, util = q.policies, q.loads0, q.utility
            pid = q.pid + ("_static" if job["static"] else "")
        theta = env.true_theta_vector()
        pp = prep_problem(env, lc, pols, loads0, H, util, inst.init_obs, theta)
        Zr = pp["Z"] @ pp["Q"]
        k_star = int(np.argmax(pp["J"]))
        base = {"kind": kind, "env": "E1-Static" if job.get("static") else "E1-Lin", "instance": seed,
                "problem": pid, "H": H, "n_policies": pp["n"], "range_resid": pp["range_resid"],
                "J_true": pp["J"].tolist(), "pi_star": k_star, "policies": [p.name for p in pols]}
        if not any(np.abs(Zr[k_star] - Zr[k]).max() > 1e-12 for k in range(pp["n"]) if k != k_star):
            return [{**base, "degenerate": True}], time.perf_counter() - t0
        # kappa analysis (h2 definitions, theta* known, Delta floored at eps)
        CLr = [C @ pp["Q"] for C in pp["CL"]]
        ka = analyse_problem(CLr, pp["J"], EPS, pp["lib"], parts=PARTS, iters=800, want_rect=True)
        ka.pop("_A_J"); ka.pop("_A_U")
        rho_star = oracle_rho(pp, EPS)
        base.update({"degenerate": False, "rho_star": rho_star, **{f"ka_{k}": v for k, v in ka.items()}})
        out = []
        noise = job["noise"]
        rseed = [seed, H, job.get("k", 0), int(job.get("static", False)), noise]
        for design in job["designs"]:
            for ev in job["evs"]:
                if ev == "b3_rage":
                    if design != "phased":
                        continue
                    r = run_rage(pp, EPS, rseed, job["T_max"])
                else:
                    v = ev_view(pp, ev)
                    r = run_seq(ev, v, pp["J"], pp["sigma"], EPS, design, rseed, job["T_max"])
                traj = r.pop("traj", None)
                row = {**base, "evidence": ev, "design": design, "noise_seed": noise, **r,
                       "below_oracle": bool((not r["censored"]) and r["N"] < rho_star and r["N"] > 0)}
                if traj is not None and ev in ("joint", "rect_time", "time") and design == "phased":
                    row["traj"] = traj
                out.append(row)
        return out, time.perf_counter() - t0
    except Exception as e:  # keep running; record the failure
        return [{"kind": kind, "instance": seed, "H": H, "problem": job.get("k"), "error": repr(e),
                 "tb": traceback.format_exc()}], time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- validation
def validate_sampler(n_real=4000, seed=7):
    """Real env.step() observations vs the exact law used by the sufficient-statistic sampler."""
    inst = make_lin_instance(3, ptypes=(1,), H_choices=(4,), K=1)
    env = inst.env
    lc = LinClass(3, 3, env.aspace.nb, nmax=3)
    rows, names = lin_library(lc, 2, False)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(names), 6, replace=False)
    out = []
    for a in pick:
        i, j, nl, b = names[a]
        loads = np.zeros(6, np.int64)
        loads[i] = nl
        act = env.aspace.index([(i, j)], [(i, j, 1)] if b else [])
        ys = []
        for _ in range(n_real):
            env.reset_to(loads)
            ob = env.step(act)
            ys.append(ob.outcomes[0][3])
            assert ob.loads[i] == nl and ob.outcomes[0][:3] == (i, j, b)
        ys = np.array(ys)
        mu = env._mean(i, j, nl, b)
        z = (ys.mean() - mu) / (env._sigma / np.sqrt(n_real))
        out.append({"atom": [int(i), int(j), int(nl), int(b)], "mean_real": float(ys.mean()), "mean_law": float(mu),
                    "z_mean": float(z), "sd_real": float(ys.std(ddof=1)), "sd_law": float(env._sigma)})
    return out


def end_to_end_env_step(seeds=(1, 2, 4), H=4, T_max=60000):
    """Joint evidence + uniform design through real env.step() (with reset_to), vs the sufficient-stat runs."""
    from dsswm.evidence.ellipsoid import EllipsoidSet
    rows_out = []
    for seed in seeds:
        inst = make_lin_instance(seed, ptypes=(1,), H_choices=(H,), K=1)
        env = inst.env
        q = inst.problems[0]
        lc = LinClass(3, 3, env.aspace.nb, nmax=3)
        pp = prep_problem(env, lc, q.policies, q.loads0, H, q.utility, inst.init_obs, env.true_theta_vector())
        ell = EllipsoidSet(lc, sigma=env._sigma, delta=DELTA, S=S_BOUND, lam=LAM)
        for ob in inst.init_obs:
            ell.update(ob)
        h = env.handle()
        names = pp["names"]
        Z = pp["Z"]
        M = len(names)
        n, grid = 0, geom_grid(T_max)
        rng = np.random.default_rng(seed)
        order = rng.permutation(np.tile(np.arange(M), T_max // M + 1))
        res = None
        gi = 0
        while n <= T_max:
            th = ell.theta_hat()
            ref = int(np.argmax(Z @ th))
            ub = (Z - Z[ref]) @ th + ell.sqrt_beta() * ell.width(Z - Z[ref])
            ub[ref] = 0
            if ub.max() <= EPS:
                res = n
                break
            target = grid[min(gi + 1, len(grid) - 1)]
            gi += 1
            while n < target:
                i, j, nl, b = names[order[n]]
                loads = np.zeros(6, np.int64)
                loads[i] = nl
                h.reset_to(loads)
                ell.update(h.step(env.aspace.index([(i, j)], [(i, j, 1)] if b else [])))
                n += 1
        sims = [run_seq("joint", ev_view(pp, "joint"), pp["J"], pp["sigma"], EPS, "uniform", [seed, s], T_max)["N"]
                for s in range(10)]
        rows_out.append({"seed": seed, "H": H, "N_env_step": res, "env_n_steps_check": int(h.n_steps),
                         "ellipsoid_n_rounds": int(ell.n_rounds), "N_suffstat_runs": sims,
                         "N_suffstat_median": float(np.median(sims))})
    return rows_out


# ----------------------------------------------------------------------------------------------- analysis
def fit_slope(x, y, groups, B=2000, seed=0):
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    if len(x) < 3 or np.ptp(x) == 0:
        return {"n": int(len(x))}
    A = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    bs = []
    for _ in range(B):
        g = rng.choice(ug, len(ug))
        idx = np.concatenate([np.nonzero(groups == gg)[0] for gg in g])
        if np.ptp(x[idx]) == 0:
            continue
        bs.append(np.linalg.lstsq(A[idx], y[idx], rcond=None)[0][0])
    from scipy.stats import spearmanr
    rho = spearmanr(x, y).correlation
    return {"n": int(len(x)), "slope": float(sl), "intercept": float(ic),
            "slope_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))] if bs else None,
            "spearman": float(rho)}


def make_figures(df, cons, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"time": "#1f77b4", "participant": "#ff7f0e", "policy": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kcol, lab in ((axes[0], "kappa", "kappa (pre-registered: joint-opt design, binding pair)"),
                          (axes[1], "kappa_eff", "kappa_eff = sqrt(rho_P / rho_joint)")):
        for p, c in cols.items():
            g = df[df.partition == p]
            ax.scatter(np.log(g[kcol] ** 2), g.log_ratio, s=12, alpha=0.7, color=c, label=p)
        lim = [0, max(1.0, float(np.nanmax(np.log(df[kcol] ** 2))))]
        ax.plot(lim, lim, "k--", lw=0.8, label="ratio = kappa^2")
        ax.set_xlabel(f"log {lab}^2", fontsize=8)
        ax.set_ylabel("log(N_rect / N_joint)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "kappa_law_scatter.png", dpi=130)
    plt.close(fig)
    if cons is not None and len(cons):
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        for ev, c in (("time", "#1f77b4"), ("cpe", "#9467bd")):
            g = cons[cons.evidence == ev].groupby("H").ratio.median()
            ax.plot(g.index, g.values, "o-", color=c, label=f"{'B6-sup time' if ev == 'time' else 'B6-CPE'} / joint")
        Hs = np.array(sorted(cons.H.unique()))
        g = cons[cons.evidence == "time"].groupby("H").ratio.median()
        ax.plot(Hs, g.values[0] * (Hs / Hs[0]) ** 2, "k--", lw=0.8, label="H^2 reference")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("H"); ax.set_ylabel("N_rect / N_joint (median)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "construction_ratio_vs_H.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import pandas as pd
    from joblib import Parallel, delayed
    pilot = args.mode == "pilot"
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if pilot else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    t_start = time.time()
    try:
        main_evs = ["joint", "time", "participant", "policy", "cpe", "b3_rage"]
        if pilot:
            noises = [42]
            # 108 problems: H in {1,4,8} x dev seeds 0..8 x 4 problems
            jobs = [dict(kind="main", seed=s, H=H, k=k, K=4, static=False, T_max=int(1e7), evs=main_evs,
                         designs=("phased", "uniform"), noise=nz)
                    for H in (1, 4, 8) for s in range(9) for k in range(4) for nz in noises]
            static_jobs = [dict(kind="static", seed=s, H=H, k=k, K=2, static=True, T_max=int(1e7),
                                evs=["joint", "time", "participant", "policy"], designs=("phased", "uniform"),
                                noise=42) for H in (4, 8) for s in range(20, 25) for k in range(2)]
            cons_jobs = [dict(kind="construction", seed=s, H=H, T_max=int(1e8),
                              evs=["joint", "time", "participant", "cpe"], designs=("phased", "uniform"), noise=42)
                         for H in (2, 4, 8, 16) for s in range(10)]
        else:
            noises = [42, 123, 456]
            jobs = [dict(kind="main", seed=10000 + 100 * hi + s, H=H, k=k, K=4, static=False, T_max=int(1e7),
                         evs=main_evs, designs=("phased", "uniform"), noise=nz)
                    for hi, H in enumerate((1, 4, 8)) for s in range(100) for k in range(4) for nz in noises]
            jobs += [dict(kind="main", seed=10500 + 100 * hi + s, H=H, k=k, K=4, static=False, T_max=int(1e7),
                          evs=main_evs, designs=("phased",), noise=42)
                     for hi, H in enumerate((2, 16)) for s in range(25) for k in range(4)]
            static_jobs = [dict(kind="static", seed=10900 + s, H=H, k=k, K=2, static=True, T_max=int(1e7),
                                evs=["joint", "time", "participant", "policy"], designs=("phased", "uniform"),
                                noise=42) for H in (4, 8) for s in range(25) for k in range(2)]
            cons_jobs = [dict(kind="construction", seed=10000 + s, H=H, T_max=int(1e8),
                              evs=["joint", "time", "participant", "cpe"], designs=("phased", "uniform"), noise=42)
                         for H in (2, 4, 8, 16) for s in range(40)]
        all_jobs = cons_jobs + static_jobs + jobs
        progress(res_root, 0, len(all_jobs), "start")
        t_val = time.perf_counter()
        val_sampler = validate_sampler()
        val_e2e = end_to_end_env_step()
        t_val = time.perf_counter() - t_val
        progress(res_root, 0, len(all_jobs), "validation done")
        rows, secs, errors = [], [], []
        done = 0
        chunk = 24
        for c0 in range(0, len(all_jobs), chunk):
            part = all_jobs[c0:c0 + chunk]
            res = Parallel(n_jobs=args.workers, verbose=0)(delayed(solve_problem)(j) for j in part)
            for r, t in res:
                secs.append(t)
                for row in r:
                    (errors if "error" in row else rows).append(row)
            done += len(part)
            progress(res_root, done, len(all_jobs), "problems", {"rows": len(rows), "errors": len(errors)})
        wall = time.time() - t_start
        with open(out_dir / "results.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "traj"}) + "\n")
        if errors:
            (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        summary = analyse(pd, rows, errors, out_dir, pilot, wall, t_val, val_sampler, val_e2e, secs, args.workers)
        mark_done(res_root, "success", summary.get("verdict_line", ""))
        print(json.dumps(summary.get("gate", {}), indent=1))
    except Exception:
        tb = traceback.format_exc()
        (out_dir / "run_error.txt").write_text(tb)
        print(tb)
        mark_done(res_root, "failed", tb[-500:])
        raise


def analyse(pd, rows, errors, out_dir, pilot, wall, t_val, val_sampler, val_e2e, secs, workers):
    df = pd.DataFrame([r for r in rows if not r.get("degenerate")])
    n_degen = sum(1 for r in rows if r.get("degenerate"))
    key = ["kind", "problem", "H", "design", "noise_seed"]
    jn = df[df.evidence == "joint"][key + ["N", "censored"]].rename(columns={"N": "N_joint", "censored": "cens_joint"})
    m = df.merge(jn, on=key, how="left")
    m["ratio"] = (m.N + 1) / (m.N_joint + 1)
    m["log_ratio"] = np.log(m.ratio)
    summary = {"task_id": TASK, "mode": "pilot" if pilot else "full", "seed": 42, "eps": EPS, "delta": DELTA,
               "ellipsoid_S": S_BOUND, "sigma": float(LIN_DEFAULTS["sigma"]), "generator_hash": generator_hash(),
               "n_rows": int(len(rows)), "n_errors": len(errors), "n_degenerate_problems": int(n_degen),
               "wall_clock_s": wall, "validation_wall_s": t_val, "workers": workers,
               "problem_seconds_median": float(np.median(secs)), "problem_seconds_max": float(np.max(secs)),
               "interaction_library": "generative single-observation atoms x(i,j,n_i,b), b in {0,1}, n_i in 0..3 "
                                      "(72 atoms; 18 in E1-Static); 1 atom = 1 platform round; state set by reset_to "
                                      "(uncharged, identical for all evidence types)",
               "validation": {"sampler_vs_env_step": val_sampler, "end_to_end_env_step": val_e2e}}
    # ---------------- main problems: kappa law
    main = m[(m.kind == "main")]
    law_rows = []
    for design in ("phased", "uniform"):
        g = main[(main.design == design) & main.evidence.isin(PARTS)]
        for _, r in g.iterrows():
            p = r.evidence
            law_rows.append({"design": design, "partition": p, "problem": r.problem, "instance": r.instance,
                             "H": r.H, "N_rect": r.N, "N_joint": r.N_joint, "censored": bool(r.censored or r.cens_joint),
                             "ratio": r.ratio, "log_ratio": r.log_ratio,
                             "kappa": r[f"ka_kappa_{p}"], "kappa0": r[f"ka_kappa0_{p}"],
                             "kappainf": r[f"ka_kappainf_{p}"], "kappa_eff": r[f"ka_kappa_eff_{p}"],
                             "kappa_uniform": r[f"ka_kappa_{p}_uniform"], "rho_ratio": r[f"ka_rho_ratio_{p}"],
                             "both_zero": bool(r.N == 0 and r.N_joint == 0)})
    law = pd.DataFrame(law_rows)
    law.to_csv(out_dir / "law_points.csv", index=False)
    fits = {}
    for design in ("phased", "uniform"):
        g0 = law[(law.design == design) & ~law.censored & ~law.both_zero]
        fd = {}
        for kname in ("kappa", "kappa0", "kappainf", "kappa_eff", "kappa_uniform"):
            for scope, gg in [("pooled", g0)] + [(p, g0[g0.partition == p]) for p in PARTS]:
                gg = gg[np.isfinite(gg[kname]) & (gg[kname] > 0)]
                fd[f"{kname}|{scope}"] = fit_slope(np.log(gg[kname]), gg.log_ratio, gg.instance)
            # large-N subset: joint needs >= 200 new rounds (asymptotic regime)
            gl = g0[(g0.N_joint >= 200) & np.isfinite(g0[kname])]
            fd[f"{kname}|pooled_Njoint>=200"] = fit_slope(np.log(gl[kname]), gl.log_ratio, gl.instance)
        sand = g0.assign(lo=2 * np.log(g0.kappa0), hi=2 * np.log(g0.kappainf))
        fd["sandwich_share_in_[k0^2,kinf^2]"] = {p: float(((s.log_ratio >= s.lo - 0.05) & (s.log_ratio <= s.hi + 0.05)).mean())
                                                 for p, s in sand.groupby("partition")}
        fd["sim_vs_design_level"] = fit_slope(np.log(g0.rho_ratio), g0.log_ratio, g0.instance)
        fits[design] = fd
    summary["slope_fits"] = fits
    summary["slope_note"] = ("slope is d log(N_rect/N_joint) / d log(kappa); the theory N_rect/N_joint ~ kappa^2 gives "
                             "slope 2 against log kappa (equivalently slope 1 against log kappa^2). The pre-registered "
                             "interval [1.5, 2.5] is read as the slope against log kappa.")
    # ratio tables
    tab = {}
    for design in ("phased", "uniform"):
        g = main[main.design == design]
        for (H, ev), s in g.groupby(["H", "evidence"]):
            tab.setdefault(design, {}).setdefault(int(H), {})[ev] = {
                "n": int(len(s)), "censored": int(s.censored.sum()),
                "N_median": float(s.N.median()), "ratio_vs_joint_median": float(s.ratio.median()),
                "ratio_q25": float(s.ratio.quantile(.25)), "ratio_q75": float(s.ratio.quantile(.75)),
                "false_cert": int(s.false_cert.sum()), "below_oracle": int(s.below_oracle.sum()),
                "N_over_rho_star_median": float((s.N / s.rho_star.clip(lower=1e-9)).median())}
    summary["ratio_tables"] = tab
    # ---------------- static vanishing
    st = {}
    h1 = main[(main.H == 1)]
    stat = m[m.kind == "static"]
    for name, g in (("H=1", h1), ("E1-Static(H=4,8)", stat)):
        d = {}
        for design in ("phased", "uniform"):
            for p in PARTS:
                s = g[(g.design == design) & (g.evidence == p)]
                s = s[~(s.N.eq(0) & s.N_joint.eq(0))]
                if not len(s):
                    continue
                gap = (s.ratio - 1) * 100
                d[f"{design}|{p}"] = {"n": int(len(s)), "gap_pct_median": float(gap.median()),
                                      "gap_pct_mean": float(gap.mean()), "share_gap_ge_10pct": float((gap >= 10).mean()),
                                      "N_joint_median": float(s.N_joint.median()), "N_rect_median": float(s.N.median()),
                                      "kappa_median": float(s[f"ka_kappa_{p}"].median())}
        st[name] = d
    summary["static_vanishing"] = st
    # ---------------- construction
    cons = m[m.kind == "construction"].copy()
    ctab = {}
    for (design, ev, H), s in cons.groupby(["design", "evidence", "H"]):
        ctab.setdefault(design, {}).setdefault(ev, {})[int(H)] = {
            "n": int(len(s)), "censored": int(s.censored.sum()), "N_median": float(s.N.median()),
            "ratio_median": float(s.ratio.median()), "kappa_time_median": float(s.ka_kappa_time.median()),
            "rho_ratio_time_median": float(s.ka_rho_ratio_time.median()),
            "N_over_rho_star_median": float((s.N / s.rho_star.clip(lower=1e-9)).median())}
    summary["construction"] = ctab
    cfit = {}
    for design in ("phased", "uniform"):
        for ev in ("time", "cpe"):
            s = cons[(cons.design == design) & (cons.evidence == ev) & ~cons.censored]
            if len(s):
                cfit[f"{design}|{ev}|loglog_slope_vs_H(all H)"] = fit_slope(np.log(s.H), s.log_ratio, s.instance)
                s2 = s[s.H >= 4]
                cfit[f"{design}|{ev}|loglog_slope_vs_H(H>=4)"] = fit_slope(np.log(s2.H), s2.log_ratio, s2.instance)
                med = s.groupby("H").ratio.median()
                Hs = med.index.values
                cfit[f"{design}|{ev}|local_slopes"] = {f"{Hs[i]}->{Hs[i+1]}": float(np.log(med.values[i+1] / med.values[i]) /
                                                                                np.log(Hs[i+1] / Hs[i]))
                                                       for i in range(len(Hs) - 1)}
                # fit ratio = (a + b H)^2 / c^2 shape: sqrt(ratio) linear in H
                A = np.vstack([s.H.values, np.ones(len(s))]).T
                cfit[f"{design}|{ev}|sqrt_ratio_linear_in_H"] = dict(zip(["b", "a"], np.linalg.lstsq(
                    A, np.sqrt(s.ratio.values), rcond=None)[0].tolist()))
    summary["construction_fits"] = cfit
    # construction: d has zero alpha/beta components (joint does not need to identify alpha)
    # ---------------- validity
    seqs = m[m.evidence != "b3_rage"]
    summary["validity"] = {"false_certifications": int(m.false_cert.sum()), "n_runs": int(len(m)),
                           "censored_runs": int(m.censored.sum()),
                           "censored_by_evidence": {k: int(v) for k, v in m.groupby("evidence").censored.sum().items()},
                           "below_oracle_runs": int(m.below_oracle.sum()),
                           "min_N_over_rho_star_uncensored_positive": float(
                               (seqs[(seqs.N > 0) & ~seqs.censored].N / seqs[(seqs.N > 0) & ~seqs.censored].rho_star).min()),
                           "zero_cost_cert_share_joint": float((main[main.evidence == "joint"].N == 0).mean())}
    # B3 sanity
    b3 = main[(main.evidence == "b3_rage")]
    summary["b3_rage_vs_joint"] = {"ratio_median": float(b3.ratio.median()), "ratio_q25": float(b3.ratio.quantile(.25)),
                                   "ratio_q75": float(b3.ratio.quantile(.75)), "false_cert": int(b3.false_cert.sum()),
                                   "censored": int(b3.censored.sum())}
    # ---------------- gate
    fp = fits["phased"]
    prim = fp["kappa|pooled"]
    st_time = [v["gap_pct_median"] for k, v in st.get("E1-Static(H=4,8)", {}).items() if k.endswith("|time")]
    st_time += [v["gap_pct_median"] for k, v in st.get("H=1", {}).items() if k.endswith("|time")]
    gate = {"slope_primary(kappa, pooled, phased)": prim.get("slope"),
            "slope_in_[1.5,2.5]": bool(prim.get("slope") is not None and 1.5 <= prim["slope"] <= 2.5),
            "spearman_primary": prim.get("spearman"), "spearman_ge_0.8": bool((prim.get("spearman") or 0) >= 0.8),
            "static_time_gap_pct_max_median": float(max(st_time)) if st_time else None,
            "static_gap_lt_10pct": bool(st_time and max(abs(x) for x in st_time) < 10),
            "no_run_below_oracle": bool(summary["validity"]["below_oracle_runs"] == 0)}
    gate["pass_literal"] = bool(gate["slope_in_[1.5,2.5]"] and gate["spearman_ge_0.8"] and gate["static_gap_lt_10pct"]
                                and gate["no_run_below_oracle"])
    summary["gate"] = gate
    summary["verdict_line"] = f"pilot gate literal pass={gate['pass_literal']}"
    # suspicious gate
    sus = []
    for design, t in tab.items():
        for H, evs in t.items():
            for ev, s in evs.items():
                if ev != "joint" and s["ratio_vs_joint_median"] > 5:
                    sus.append(f"{design} H={H} {ev}: joint saves {s['ratio_vs_joint_median']:.1f}x (median)")
    summary["suspicious_flags"] = sus
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    # samples: representative trajectories
    samp = [r for r in rows if r.get("traj") is not None][:10]
    (out_dir / "samples" / "trajectories.json").write_text(json.dumps(
        [{k: r[k] for k in ("problem", "H", "evidence", "design", "N", "pi_hat", "regret", "policies", "J_true", "traj")
          if k in r} for r in samp], indent=1, default=float))
    lawp = law[law.design == "phased"]
    make_figures(lawp[~lawp.censored & ~lawp.both_zero], cons[cons.design == "phased"], out_dir)
    return summary


if __name__ == "__main__":
    main()
