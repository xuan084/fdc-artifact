"""hd2_kappa_sandwich_lin: corrected kappa theory (HD2) on E1-Lin (eps=0.05, sigma=1.5).

What is tested (reuses run_h1_kappa_law_lin.py for the problem preparation, the B6 evidence types, the sequential
certification loop and the Omega(H^2) construction):

1. Design-level sandwich (a theorem; any violation is an implementation error). For every problem and partition P,
   with gaps g_k (two variants: g = Delta + eps, matched to the stopping rule; g = max(Delta, eps), the h2 definition)
       f^J_k(xi) = ||z* - z_k||^2_{A(xi)^-1} / g_k^2,   f^P_k(xi) = (sum_b ||B^P_kb||_{A(xi)^-1})^2 / g_k^2,
       kappa_k(xi) = sqrt(f^P_k / f^J_k) >= 1,
       rho_J = min_{xi in C} max_k f^J_k,  rho_P = min_{xi in C} max_k f^P_k  (C = all computed designs + uniform),
       xi_J, xi_P the minimisers, bind(xi_P) = argmax_k f^J_k(xi_P).
   Chain:  kappa_bind(xi_P)^2 <= rho_P / rho_J <= rho_P(xi_J) / rho_J <= max_k kappa_k(xi_J)^2.
2. beta(N) correction: the sequential rule stops when sqrt(beta(N)) * w_k(V_N) <= Delta_k + eps, so
   N ~ beta(N) * rho. Corrected sequential log ratio = log(N_P / N_J) - log(beta_P(N_P) / beta_J(N_J)), regressed on
   log(rho_P / rho_J): pre-registered slope in [0.85, 1.15]. Also the fixed-point prediction N* = beta(N*) rho.
3. kappa_eff = sqrt(rho_P / rho_J); slope of log(N_rect / N_joint) on log kappa_eff, per partition: [1.5, 2.5].
4. Omega(H^2) construction (same exposure, different timing), H in {2,4,8,16}: slope of log(N_time/N_joint) on
   log H: [1.5, 2.5]. B6-CPE reported alongside.
5. H = 1: the time partition has a single block, so N_time == N_joint exactly (same design path, same noise).
6. B3 sanity: RAGE (Fiez et al. 2019) XY-allocation over the surviving set + the same joint stopping rule.
Binding-pair kappa (pre-registered kappa at the joint-optimal design) is reported as a secondary metric.

Usage: run_hd2_kappa_sandwich_lin.py --mode {pilot,full} [--workers 4]
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
from run_h2_kappa_distribution import objective, optimise, partition_blocks  # noqa: E402
from run_h1_kappa_law_lin import (S_BOUND, construction_policies, end_to_end_env_step, ev_view,  # noqa: E402
                                  fit_slope, geom_grid, largest_remainder, oracle_rho, prep_problem, validate_sampler)

TASK = "hd2_kappa_sandwich_lin"
EPS, DELTA, LAM = 0.05, 0.05, 1.0
PARTS = ("time", "participant", "policy")
CONS_PARTS = ("time", "participant")
REL_TOL = 1e-9
PRE = json.loads((WS / "plan" / "prereg_lock.json").read_text())


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


# ----------------------------------------------------------------------------------------------- design level
def beta_of(V, dd, sigma, S):
    _, logdet = np.linalg.slogdet(V)
    return (sigma * np.sqrt(2 * np.log(1 / DELTA) + logdet - dd * np.log(LAM)) + np.sqrt(LAM) * S) ** 2


def fixed_point_N(pp, xi, rho, it=60):
    """N* = beta(N*) * rho with V_N = lam I + X0'X0 + N A_full(xi) (prior information ignored in the widths)."""
    rows = pp["rows"]
    A = (rows.T * xi) @ rows
    V0 = LAM * np.eye(pp["d"]) + pp["X0"].T @ pp["X0"]
    N = 0.0
    for _ in range(it):
        Nn = beta_of(V0 + N * A, pp["d"], pp["sigma"], S_BOUND) * rho
        if abs(Nn - N) <= 1e-9 * max(Nn, 1.0):
            N = Nn
            break
        N = Nn
    return float(N)


def design_level(pp, parts, iters=800):
    """Sandwich + rho quantities on the true theta, two gap variants. Returns a flat dict of results."""
    Q, lib, Z, J, CL = pp["Q"], pp["lib"], pp["Z"], pp["J"], pp["CL"]
    n = len(J)
    ks = int(np.argmax(J))
    Zr = Z @ Q
    ch = [k for k in range(n) if k != ks and np.abs(Zr[ks] - Zr[k]).max() > 1e-12]
    Craw = [C @ Q for C in CL]                                        # (H, L, r) per policy
    Delta = np.array([J[ks] - J[k] for k in ch])
    jblocks = [(Zr[ks] - Zr[k])[None] for k in ch]
    pblocks = {p: [partition_blocks(Craw[ks], Craw[k], p) for k in ch] for p in parts}
    out = {"n_challengers": len(ch), "Delta_min": float(Delta.min())}
    xi_u = np.full(lib.n, 1.0 / lib.n)
    for gname, g in (("plus", Delta + EPS), ("floor", np.maximum(Delta, EPS))):
        xiJ, _ = optimise(lib, jblocks, g, iters=iters)
        cands = {"J": xiJ, "U": xi_u}
        for p in parts:
            cands[p], _ = optimise(lib, pblocks[p], g, iters=iters, xi0=xiJ)
        fJ = {c: objective(lib, x, jblocks, g) for c, x in cands.items()}
        cJ = min(fJ, key=lambda c: fJ[c].max())
        rho_J = float(fJ[cJ].max())
        out[f"{gname}|rho_J"] = rho_J
        out[f"{gname}|xiJ_from"] = cJ
        if gname == "plus":
            out["plus|Nstar_J"] = fixed_point_N(pp, cands[cJ], rho_J)
        # pre-registered binding-pair kappa (joint-optimal design, joint-binding pair)
        bJ = int(np.argmax(fJ[cJ]))
        for p in parts:
            fP = {c: objective(lib, x, pblocks[p], g) for c, x in cands.items()}
            cP = min(fP, key=lambda c: fP[c].max())
            rho_P = float(fP[cP].max())
            kap_P = np.sqrt(fP[cP] / fJ[cP])                          # kappa_k(xi_P)
            kap_J = np.sqrt(fP[cJ] / fJ[cJ])                          # kappa_k(xi_J)
            bP = int(np.argmax(fJ[cP]))                                # joint-binding pair under xi_P
            L = float(kap_P[bP] ** 2)
            R = rho_P / rho_J
            M = float(fP[cJ].max()) / rho_J
            U = float((kap_J ** 2).max())
            kmin = float(min(kap_P.min(), kap_J.min()))
            viol = {"L<=R": bool(L <= R * (1 + REL_TOL)), "R<=M": bool(R <= M * (1 + REL_TOL)),
                    "M<=U": bool(M <= U * (1 + REL_TOL)), "kappa>=1": bool(kmin >= 1 - 1e-9)}
            pre = f"{gname}|{p}|"
            out.update({pre + "rho_P": rho_P, pre + "rho_ratio": R, pre + "lower": L, pre + "mid": M,
                        pre + "upper": U, pre + "kappa_min": kmin, pre + "sandwich_ok": bool(all(viol.values())),
                        pre + "checks": viol, pre + "xiP_from": cP,
                        pre + "kappa_bind_J": float(kap_J[bJ]),           # pre-registered kappa
                        pre + "kappa_eff": float(np.sqrt(R))})
            if gname == "plus":
                out[pre + "Nstar"] = fixed_point_N(pp, cands[cP], rho_P)
    return out


# ----------------------------------------------------------------------------------------------- sequential
def run_seq(v, J, sigma, eps, design_mode, seed, T_max, grid_ratio=1.01, iters=300, n_phase0=32):
    """run_h1_kappa_law_lin.run_seq plus design_mode 'rage' (XY-allocation over the surviving set, unit weights)."""
    rng = np.random.default_rng(seed)
    X, lib, mu, Z, D, dd = v["rows"], v["lib"], v["mu"], v["Z"], v["D"], v["dd"]
    M = X.shape[0]
    V = LAM * np.eye(dd) + v["X0"].T @ v["X0"]
    bvec = v["X0"].T @ v["y0"]
    phase_counts = np.zeros(M, np.int64)
    n, n_designs, next_phase = 0, 0, 0
    xi = np.full(M, 1.0 / M)
    t0 = time.perf_counter()
    log2dl = 2 * np.log(1 / DELTA)

    def certify():
        th = np.linalg.solve(V, bvec)
        vals = Z @ th
        ref = int(np.argmax(vals))
        Vi = np.linalg.inv(V)
        Dr = D[ref]
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
            tgt = (phase_counts.sum() + m) * xi
            deficit = np.maximum(tgt - phase_counts, 0.0)
            if deficit.sum() <= 0:
                deficit = xi.copy()
            inc = largest_remainder(deficit / deficit.sum() * m, m)
            idx = np.nonzero(inc)[0]
            c = inc[idx]
            s = rng.normal(c * mu[idx], sigma * np.sqrt(c))
            Xi = X[idx]
            V += (Xi.T * c) @ Xi
            bvec += Xi.T @ s
            phase_counts += inc
            n = n_t
            ref, ub, th, sb = certify()
        if len(traj) < 60 and (n == 0 or n >= 1.3 * (traj[-1][0] if traj else 1)):
            traj.append((int(n), float(ub.max()), int(ref)))
        if ub.max() <= eps:
            return dict(N=int(n), censored=False, pi_hat=ref, regret=float(J.max() - J[ref]),
                        false_cert=bool(J.max() - J[ref] > eps), n_designs=n_designs,
                        wall_s=time.perf_counter() - t0, sqrt_beta=float(sb), traj=traj)
        if design_mode in ("phased", "rage") and n >= next_phase:
            active = [k for k in range(len(Z)) if k != ref and ub[k] > eps]
            Q = lib.Q
            if design_mode == "phased":
                Dhat = np.array([max(float((Z[ref] - Z[k]) @ th), 0.0) for k in active]) + eps
                blocks = []
                for k in active:
                    B = D[ref][k] @ Q
                    keep = np.abs(B).max(1) > 1e-13
                    blocks.append(B[keep] if keep.any() else B[:1])
            else:   # RAGE XY-allocation: all pairs within the surviving set, unit weights
                S = [ref] + active
                Zr = Z @ Q
                blocks = [(Zr[a] - Zr[b])[None] for ii, a in enumerate(S) for b in S[ii + 1:]
                          if np.abs(Zr[a] - Zr[b]).max() > 1e-13]
                if not blocks:
                    blocks = [(Zr[ref] - Zr[active[0]])[None]]
                Dhat = np.ones(len(blocks))
            xs, _ = optimise(lib, blocks, Dhat, iters=iters if n_designs else 2 * iters, xi0=None)
            xi = 0.98 * xs + 0.02 / M
            phase_counts[:] = 0
            n_designs += 1
            next_phase = n_phase0 if n == 0 else 2 * n
        if n >= T_max:
            break
    return dict(N=int(n), censored=True, pi_hat=ref, regret=float(J.max() - J[ref]), false_cert=False,
                n_designs=n_designs, wall_s=time.perf_counter() - t0, sqrt_beta=float(sb), traj=traj)


# ----------------------------------------------------------------------------------------------- jobs
def solve_problem(job):
    """job: dict(kind, seed, H, k, K, T_max, runs=[(ev, design)], noises=[...])."""
    t0 = time.perf_counter()
    kind, seed, H = job["kind"], job["seed"], job["H"]
    out = []
    try:
        for noise in job["noises"]:
            if kind == "construction":
                inst = make_lin_instance(seed, noise_seed=noise, ptypes=(1,), H_choices=(max(H, 1),), K=1,
                                         prior={"gamma": (0.3, 0.6)})
                env = inst.env
                lc = LinClass(3, 3, env.aspace.nb, nmax=3, static=False)
                pols, loads0 = construction_policies(env.aspace, H)
                util = Utility(w=np.ones(H), w_ret=0.0, c_q=1.0)
                pid = f"constr{seed}_H{H}"
                parts = CONS_PARTS
            else:
                inst = make_lin_instance(seed, noise_seed=noise, ptypes=(1,), H_choices=(H,), K=job["K"])
                env = inst.env
                q = inst.problems[job["k"]]
                lc = LinClass(3, 3, env.aspace.nb, nmax=3, static=False)
                pols, loads0, util = q.policies, q.loads0, q.utility
                pid = q.pid
                parts = PARTS
            theta = env.true_theta_vector()
            pp = prep_problem(env, lc, pols, loads0, H, util, inst.init_obs, theta)
            Zr = pp["Z"] @ pp["Q"]
            k_star = int(np.argmax(pp["J"]))
            base = {"kind": kind, "instance": seed, "problem": pid, "k": job.get("k", 0), "H": H,
                    "noise_seed": noise, "n_policies": pp["n"], "range_resid": pp["range_resid"],
                    "J_true": pp["J"].tolist(), "pi_star": k_star, "policies": [p.name for p in pols]}
            if not any(np.abs(Zr[k_star] - Zr[k]).max() > 1e-12 for k in range(pp["n"]) if k != k_star):
                out.append({**base, "degenerate": True})
                continue
            if noise == job["noises"][0] or kind == "construction":
                dl = design_level(pp, parts)          # theta*, design only: noise-independent for main problems
                rho_star = oracle_rho(pp, EPS)
            base.update({"degenerate": False, "rho_star": rho_star, "design": dl})
            rseed = [seed, H, job.get("k", 0), noise]
            runs = []
            for ev, design in job["runs"]:
                v = ev_view(pp, ev)
                r = run_seq(v, pp["J"], pp["sigma"], EPS, design, rseed, job["T_max"])
                r["evidence"], r["design"] = ev, design
                r["below_oracle"] = bool((not r["censored"]) and 0 < r["N"] < rho_star)
                runs.append(r)
            out.append({**base, "runs": runs})
        return out, time.perf_counter() - t0
    except Exception as e:  # keep running; record the failure
        return [{"kind": kind, "instance": seed, "H": H, "k": job.get("k"), "error": repr(e),
                 "tb": traceback.format_exc()}], time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- analysis
def flatten(rows):
    """One record per (problem, noise, evidence, design) with the joint (same design) reference attached."""
    recs = []
    for r in rows:
        if r.get("degenerate"):
            continue
        by = {(x["evidence"], x["design"]): x for x in r["runs"]}
        for (ev, des), x in by.items():
            jref = by.get(("joint", des if des != "rage" else "phased"))
            rec = {"kind": r["kind"], "instance": r["instance"], "problem": r["problem"], "H": r["H"],
                   "noise_seed": r["noise_seed"], "evidence": ev, "design": des, "N": x["N"],
                   "censored": x["censored"], "false_cert": x["false_cert"], "regret": x["regret"],
                   "sqrt_beta": x["sqrt_beta"], "below_oracle": x["below_oracle"], "rho_star": r["rho_star"],
                   "wall_s": x["wall_s"]}
            if jref is not None:
                rec.update({"N_joint": jref["N"], "cens_joint": jref["censored"], "sb_joint": jref["sqrt_beta"]})
            dl = r["design"]
            if ev in PARTS or (r["kind"] == "construction" and ev in CONS_PARTS):
                for g in ("plus", "floor"):
                    for f in ("rho_ratio", "lower", "upper", "kappa_eff", "kappa_bind_J"):
                        rec[f"{g}_{f}"] = dl.get(f"{g}|{ev}|{f}")
                rec["Nstar_ratio"] = dl.get(f"plus|{ev}|Nstar", np.nan) / dl["plus|Nstar_J"]
            recs.append(rec)
    return recs


def analyse(pd, rows, errors, out_dir, pilot, wall, t_val, val_sampler, val_e2e, secs, workers, n_jobs_main):
    good = [r for r in rows if not r.get("degenerate")]
    df = pd.DataFrame(flatten(rows))
    df["ratio"] = (df.N + 1) / (df.N_joint + 1)
    df["log_ratio"] = np.log(df.ratio)
    ok = (~df.censored) & (~df.cens_joint.fillna(True).astype(bool)) & (df.N > 0) & (df.N_joint > 0)
    df["beta_corr_log_ratio"] = np.where(ok, np.log(df.N / df.N_joint) - 2 * np.log(df.sqrt_beta / df.sb_joint),
                                         np.nan)
    summary = {"task_id": TASK, "mode": "pilot" if pilot else "full", "seed": 42, "eps": EPS, "delta": DELTA,
               "ellipsoid_S": S_BOUND, "sigma": float(LIN_DEFAULTS["sigma"]), "generator_hash": generator_hash(),
               "prereg_status": PRE.get("status"), "n_problem_records": len(rows), "n_errors": len(errors),
               "n_degenerate": int(sum(1 for r in rows if r.get("degenerate"))),
               "n_sequential_runs": int(len(df)), "wall_clock_s": wall, "validation_wall_s": t_val,
               "workers": workers, "timing_note": "concurrent run (4 tasks share 20 cores); per-task 4 workers",
               "problem_seconds_median": float(np.median(secs)), "problem_seconds_max": float(np.max(secs)),
               "validation": {"sampler_vs_env_step": val_sampler, "end_to_end_env_step": val_e2e}}
    # ---------------- 1. sandwich (design level, theorem)
    sand = {}
    viol_examples = []
    n_checks = n_viol = 0
    for r in good:
        dl = r["design"]
        parts = PARTS if r["kind"] == "main" else CONS_PARTS
        for g in ("plus", "floor"):
            for p in parts:
                okp = dl[f"{g}|{p}|sandwich_ok"]
                n_checks += 1
                n_viol += int(not okp)
                d = sand.setdefault(f"{g}|{p}", {"n": 0, "violations": 0, "lower_tight": 0, "mid_eq_ratio": 0,
                                                 "upper_over_lower_median": [], "xiP_not_own": 0})
                d["n"] += 1
                d["violations"] += int(not okp)
                d["lower_tight"] += int(abs(dl[f"{g}|{p}|lower"] / dl[f"{g}|{p}|rho_ratio"] - 1) < 1e-6)
                d["mid_eq_ratio"] += int(abs(dl[f"{g}|{p}|mid"] / dl[f"{g}|{p}|rho_ratio"] - 1) < 1e-6)
                d["upper_over_lower_median"].append(dl[f"{g}|{p}|upper"] / dl[f"{g}|{p}|lower"])
                d["xiP_not_own"] += int(dl[f"{g}|{p}|xiP_from"] != p)
                if not okp and len(viol_examples) < 10:
                    viol_examples.append({"problem": r["problem"], "noise": r["noise_seed"], "gap": g, "part": p,
                                          "checks": dl[f"{g}|{p}|checks"],
                                          **{x: dl[f"{g}|{p}|{x}"] for x in ("lower", "rho_ratio", "mid", "upper",
                                                                              "kappa_min")}})
    for d in sand.values():
        d["upper_over_lower_median"] = float(np.median(d["upper_over_lower_median"]))
    summary["sandwich"] = {"n_checks": n_checks, "violations": n_viol, "violation_rate": n_viol / max(n_checks, 1),
                           "rel_tol": REL_TOL, "by_gap_partition": sand, "violation_examples": viol_examples,
                           "note": "xiP_not_own = the best design for the rectangular objective came from another "
                                   "candidate (optimiser suboptimality diagnostic); lower/upper use the selected designs"}
    # ---------------- 2/3. slopes (main, phased primary; uniform side)
    main = df[(df.kind == "main") & df.evidence.isin(PARTS)]
    fits = {}
    for design in ("phased", "uniform"):
        g0 = main[(main.design == design) & ok]
        fd = {}
        for scope, gg in [("pooled", g0)] + [(p, g0[g0.evidence == p]) for p in PARTS]:
            gg = gg[np.isfinite(gg.plus_rho_ratio) & (gg.plus_rho_ratio > 0)]
            gl = gg[gg.N_joint >= 200]
            x = np.log(gg.plus_rho_ratio)
            fd[f"beta_corrected|{scope}"] = fit_slope(x, gg.beta_corr_log_ratio, gg.instance)
            fd[f"uncorrected_vs_rho|{scope}"] = fit_slope(x, np.log(gg.N / gg.N_joint), gg.instance)
            fd[f"beta_corrected|{scope}|Njoint>=200"] = fit_slope(np.log(gl.plus_rho_ratio), gl.beta_corr_log_ratio,
                                                                 gl.instance)
            fd[f"fixed_point|{scope}"] = fit_slope(np.log(gg.Nstar_ratio), np.log(gg.N / gg.N_joint), gg.instance)
            fd[f"kappa_eff|{scope}"] = fit_slope(np.log(gg.plus_kappa_eff), np.log(gg.N / gg.N_joint), gg.instance)
            fd[f"kappa_eff_floor|{scope}"] = fit_slope(np.log(gg.floor_kappa_eff), np.log(gg.N / gg.N_joint),
                                                      gg.instance)
            kb = gg[np.isfinite(gg.floor_kappa_bind_J) & (gg.floor_kappa_bind_J > 0)]
            fd[f"kappa_bind(secondary)|{scope}"] = fit_slope(np.log(kb.floor_kappa_bind_J), np.log(kb.N / kb.N_joint),
                                                            kb.instance)
            inb = gg[(gg.evidence.isin(PARTS))]
            fd[f"seq_ratio_in_design_sandwich_share|{scope}"] = float(
                ((np.log(inb.N / inb.N_joint) - 2 * np.log(inb.sqrt_beta / inb.sb_joint) >= np.log(inb.plus_lower) - 0.1)
                 & (np.log(inb.N / inb.N_joint) - 2 * np.log(inb.sqrt_beta / inb.sb_joint)
                    <= np.log(inb.plus_upper) + 0.1)).mean()) if len(inb) else None
        fits[design] = fd
    summary["slope_fits"] = fits
    summary["slope_note"] = ("beta_corrected: log(N_P/N_J) - log(beta_P(N_P)/beta_J(N_J)) vs log(rho_P/rho_J), "
                             "pre-registered [0.85,1.15]; kappa_eff: log(N_P/N_J) vs log sqrt(rho_P/rho_J), "
                             "pre-registered [1.5,2.5]; rho with gaps Delta+eps (stopping-rule matched); "
                             "kappa_bind = pre-registered binding-pair kappa at the joint-optimal design (h2 gaps).")
    # ---------------- 4. construction
    cons = df[df.kind == "construction"]
    ctab, cfit = {}, {}
    for (design, ev, H), s in cons.groupby(["design", "evidence", "H"]):
        ctab.setdefault(design, {}).setdefault(ev, {})[int(H)] = {
            "n": int(len(s)), "censored": int(s.censored.sum()), "N_median": float(s.N.median()),
            "ratio_median": float(s.ratio.median()),
            "rho_ratio_median": float(s.plus_rho_ratio.median()) if "plus_rho_ratio" in s and s.plus_rho_ratio.notna().any() else None}
    for design in ("phased", "uniform"):
        for ev in ("time", "participant", "cpe"):
            s = cons[(cons.design == design) & (cons.evidence == ev) & ~cons.censored & ~cons.cens_joint.astype(bool)]
            if len(s):
                cfit[f"{design}|{ev}|slope_vs_logH(H=2..16)"] = fit_slope(np.log(s.H), s.log_ratio, s.instance)
                s2 = s[s.H >= 4]
                cfit[f"{design}|{ev}|slope_vs_logH(H>=4)"] = fit_slope(np.log(s2.H), s2.log_ratio, s2.instance)
                if ev in CONS_PARTS:
                    cfit[f"{design}|{ev}|design_level_slope_vs_logH"] = fit_slope(
                        np.log(s.H), np.log(s.plus_rho_ratio), s.instance)
    summary["construction"] = {"table": ctab, "fits": cfit}
    # ---------------- 5. H=1 exact zero (time vs joint)
    h1 = df[(df.kind == "main") & (df.H == 1) & (df.evidence == "time")]
    diffs = (h1.N - h1.N_joint).abs()
    h1_dl = [abs(r["design"]["plus|time|rho_ratio"] - 1) for r in good if r["kind"] == "main" and r["H"] == 1]
    summary["h1_exact_zero"] = {"n_runs": int(len(h1)), "n_nonzero_gap": int((diffs > 0).sum()),
                                "max_abs_gap": int(diffs.max()) if len(h1) else None,
                                "by_design": {d: int(((s.N - s.N_joint).abs() > 0).sum()) for d, s in h1.groupby("design")},
                                "design_level_max_|rho_time/rho_J-1|": float(max(h1_dl)) if h1_dl else None,
                                "pass": bool(len(h1) > 0 and (diffs == 0).all())}
    # ---------------- ratio tables, baselines, validity
    tab = {}
    for (design, H, ev), s in df[df.kind == "main"].groupby(["design", "H", "evidence"]):
        tab.setdefault(design, {}).setdefault(int(H), {})[ev] = {
            "n": int(len(s)), "censored": int(s.censored.sum()), "N_median": float(s.N.median()),
            "ratio_vs_joint_median": float(s.ratio.median()), "ratio_q25": float(s.ratio.quantile(.25)),
            "ratio_q75": float(s.ratio.quantile(.75)), "false_cert": int(s.false_cert.sum()),
            "below_oracle": int(s.below_oracle.sum())}
    summary["ratio_tables"] = tab
    rg = df[(df.kind == "main") & (df.evidence == "joint") & (df.design == "rage")]
    summary["b3_rage_design_joint_stop"] = {
        "n": int(len(rg)), "ratio_vs_joint_phased_median": float(rg.ratio.median()) if len(rg) else None,
        "ratio_q25": float(rg.ratio.quantile(.25)) if len(rg) else None,
        "ratio_q75": float(rg.ratio.quantile(.75)) if len(rg) else None,
        "share_rage_le_phased": float((rg.N <= rg.N_joint).mean()) if len(rg) else None,
        "false_cert": int(rg.false_cert.sum()), "censored": int(rg.censored.sum())}
    cpe = df[(df.kind == "main") & (df.evidence == "cpe") & (df.design == "phased")]
    summary["b6_cpe"] = {f"H={int(H)}": {"n": int(len(s)), "ratio_vs_joint_median": float(s.ratio.median()),
                                         "censored": int(s.censored.sum())} for H, s in cpe.groupby("H")}
    seqs = df[(df.N > 0) & ~df.censored]
    summary["validity"] = {"false_certifications": int(df.false_cert.sum()), "n_runs": int(len(df)),
                           "fcr_upper_cp95": cp_upper(int(df.false_cert.sum()), int((~df.censored).sum())),
                           "censored_runs": int(df.censored.sum()),
                           "censored_by_evidence": {k: int(v) for k, v in df.groupby("evidence").censored.sum().items()},
                           "below_oracle_runs": int(df.below_oracle.sum()),
                           "min_N_over_rho_star": float((seqs.N / seqs.rho_star).min()) if len(seqs) else None}
    # ---------------- gate
    fp = fits["phased"]
    bc = fp["beta_corrected|pooled"]
    ke = {p: fp[f"kappa_eff|{p}"].get("slope") for p in PARTS}
    cs = cfit.get("phased|time|slope_vs_logH(H=2..16)", {})
    gate = {"zero_crashes": len(errors) == 0,
            "n_problem_method_runs_ge_100": bool(len(df) >= 100),
            "sandwich_violations": n_viol, "sandwich_100pct": bool(n_viol == 0),
            "h1_time_gap_exact_zero": summary["h1_exact_zero"]["pass"],
            "beta_corrected_slope_pooled": bc.get("slope"), "beta_corrected_slope_ci": bc.get("slope_ci95"),
            "beta_slope_in_[0.85,1.15]": bool(bc.get("slope") is not None and 0.85 <= bc["slope"] <= 1.15),
            "kappa_eff_slope_by_partition": ke,
            "kappa_eff_slope_pooled": fp["kappa_eff|pooled"].get("slope"),
            "kappa_eff_slope_ci_pooled": fp["kappa_eff|pooled"].get("slope_ci95"),
            "construction_slope_time": cs.get("slope"), "construction_slope_ci": cs.get("slope_ci95"),
            "no_false_cert": bool(df.false_cert.sum() == 0)}
    gate["pass"] = bool(gate["sandwich_100pct"] and gate["h1_time_gap_exact_zero"] and gate["zero_crashes"])
    summary["gate"] = gate
    # timing projection for full (main jobs carry 3 noise streams, design level computed once)
    main_secs = [t for t in secs[-n_jobs_main:]] if n_jobs_main else []
    per_job = float(np.mean(main_secs)) if main_secs else None
    if per_job is not None:
        full_jobs = 48 * 3 * 4
        summary["projected_full_min"] = float((full_jobs * per_job * 2.7 + 80 * np.mean(secs[:20]) * 1.0)
                                              / workers / 60 + t_val / 60)
        summary["projection_note"] = ("full = 576 main problems x 3 noise streams (design level once, sequential x3, "
                                      "factor 2.7) + 80 construction runs; concurrent timing")
    summary["verdict_line"] = (f"sandwich viol={n_viol}/{n_checks}; H1 exact-zero={gate['h1_time_gap_exact_zero']}; "
                               f"beta slope={bc.get('slope')}; pass={gate['pass']}")
    sus = []
    for design, t in tab.items():
        for H, evs in t.items():
            for ev, s in evs.items():
                if ev != "joint" and s["ratio_vs_joint_median"] > 5:
                    sus.append(f"{design} H={H} {ev}: joint saves {s['ratio_vs_joint_median']:.1f}x (median) -- "
                               "expected by kappa theory (rectangularisation), not a JPC saving claim")
    summary["suspicious_flags"] = sus
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    df.to_csv(out_dir / "law_points.csv", index=False)
    # samples
    samp = []
    for r in good[:: max(1, len(good) // 8)][:8]:
        samp.append({"problem": r["problem"], "H": r["H"], "kind": r["kind"], "J_true": r["J_true"],
                     "policies": r["policies"], "rho_star": r["rho_star"],
                     "design_level": {k: v for k, v in r["design"].items() if k.startswith("plus|")},
                     "runs": [{k: x[k] for k in ("evidence", "design", "N", "pi_hat", "regret", "censored",
                                                 "sqrt_beta", "traj")} for x in r["runs"]]})
    (out_dir / "samples" / "samples.json").write_text(json.dumps(samp, indent=1, default=float))
    make_figures(df, ok, out_dir)
    return summary


def cp_upper(k, n, a=0.05):
    from scipy.stats import beta
    return float(beta.ppf(1 - a / 2, k + 1, n - k)) if n > 0 and k < n else 1.0


def make_figures(df, ok, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"time": "#1f77b4", "participant": "#ff7f0e", "policy": "#2ca02c"}
    g = df[(df.kind == "main") & (df.design == "phased") & df.evidence.isin(PARTS) & ok]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for p, c in cols.items():
        s = g[g.evidence == p]
        x = np.log(s.plus_rho_ratio)
        axes[0].errorbar(x, s.beta_corr_log_ratio, fmt="o", ms=3, alpha=0.6, color=c, label=p)
        axes[0].vlines(x, np.log(s.plus_lower), np.log(s.plus_upper), color=c, alpha=0.15, lw=0.8)
        axes[1].scatter(np.log(s.plus_kappa_eff), np.log(s.N / s.N_joint), s=10, alpha=0.6, color=c, label=p)
    lim = [0, max(1.0, float(np.nanmax(np.log(g.plus_rho_ratio)))) if len(g) else 1.0]
    axes[0].plot(lim, lim, "k--", lw=0.8, label="slope 1")
    axes[0].set_xlabel("log rho_P / rho_J (design level)")
    axes[0].set_ylabel("beta-corrected log N_P / N_J")
    axes[0].set_title("bars: design-level sandwich [kappa_bind^2, max kappa^2]", fontsize=8)
    axes[1].plot([0, lim[1] / 2], [0, lim[1]], "k--", lw=0.8, label="slope 2")
    axes[1].set_xlabel("log kappa_eff")
    axes[1].set_ylabel("log N_rect / N_joint")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sandwich_scatter.png", dpi=130)
    plt.close(fig)
    cons = df[(df.kind == "construction") & (df.design == "phased")]
    if len(cons):
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        for ev, c in (("time", "#1f77b4"), ("cpe", "#9467bd"), ("participant", "#ff7f0e")):
            m = cons[cons.evidence == ev].groupby("H").ratio.median()
            if len(m):
                ax.plot(m.index, m.values, "o-", color=c, label=f"{'B6-CPE' if ev == 'cpe' else 'B6-sup ' + ev} / joint")
        m = cons[cons.evidence == "time"].groupby("H").ratio.median()
        Hs = np.array(sorted(cons.H.unique()))
        if len(m):
            ax.plot(Hs, m.values[0] * (Hs / Hs[0]) ** 2, "k--", lw=0.8, label="H^2 reference")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("H")
        ax.set_ylabel("N_rect / N_joint (median)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "construction_ratio_vs_H.png", dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
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
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    # round-3 replication overrides (r3_replicate_hd1b_hd2); defaults reproduce the round-2 run
    ap.add_argument("--seeds", default=None, help="main-problem instance seeds, 'a-b' inclusive or comma list")
    ap.add_argument("--cons-seeds", default=None, help="construction instance seeds ('none' = skip construction)")
    ap.add_argument("--hs", default=None, help="main H values, comma list (default 1,4,8)")
    ap.add_argument("--cons-hs", default=None, help="construction H values, comma list (default 2,4,8,16)")
    ap.add_argument("--noise-seeds", default=None, help="main noise seeds, comma list")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-protocol", action="store_true",
                    help="write PID/PROGRESS/DONE into out_dir instead of exp/results (a wrapper task owns those)")
    args = ap.parse_args()
    import pandas as pd
    from joblib import Parallel, delayed
    pilot = args.mode == "pilot"
    res_root = WS / "exp" / "results"
    out_dir = Path(args.out_dir) if args.out_dir else res_root / ("pilots" if pilot else "full") / TASK
    if args.no_protocol:
        res_root = out_dir
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    t_start = time.time()
    try:
        if not pilot:
            from dsswm.stats.prereg import assert_locked
            assert_locked()
        main_runs = [("joint", "phased"), ("joint", "uniform"), ("joint", "rage")] + \
                    [(ev, d) for ev in ("time", "participant", "policy", "cpe") for d in ("phased", "uniform")]
        cons_runs = [(ev, d) for ev in ("joint", "time", "participant", "cpe") for d in ("phased", "uniform")]
        if pilot:
            seeds, noises, cseeds = range(600, 609), [42], range(600, 605)
        else:
            seeds, noises, cseeds = range(10000, 10048), [42, 123, 456], range(10000, 10020)
        hs, cons_hs = (1, 4, 8), (2, 4, 8, 16)
        if args.seeds:
            seeds = parse_seeds(args.seeds)
        if args.cons_seeds:
            cseeds = [] if args.cons_seeds == "none" else parse_seeds(args.cons_seeds)
        if args.noise_seeds:
            noises = [int(x) for x in args.noise_seeds.split(",")]
        if args.hs:
            hs = tuple(int(x) for x in args.hs.split(","))
        if args.cons_hs:
            cons_hs = tuple(int(x) for x in args.cons_hs.split(","))
        if pilot:
            assert max(list(seeds) + list(cseeds)) < 10000, "pilot mode must not touch evaluation seeds"
        jobs = [dict(kind="main", seed=s, H=H, k=k, K=4, T_max=int(1e7), runs=main_runs, noises=noises)
                for H in hs for s in seeds for k in range(4)]
        cons_jobs = [dict(kind="construction", seed=s, H=H, T_max=int(1e8), runs=cons_runs, noises=[42])
                     for H in cons_hs for s in cseeds]
        (out_dir / "config.json").write_text(json.dumps({
            "mode": args.mode, "main_seeds": list(seeds), "cons_seeds": list(cseeds), "noises": list(noises),
            "H_main": list(hs), "H_cons": list(cons_hs), "n_main_jobs": len(jobs), "n_cons_jobs": len(cons_jobs)},
            indent=1))
        all_jobs = cons_jobs + jobs
        progress(res_root, 0, len(all_jobs), "start")
        t_val = time.perf_counter()
        val_sampler = validate_sampler()
        val_e2e = end_to_end_env_step()
        t_val = time.perf_counter() - t_val
        progress(res_root, 0, len(all_jobs), "validation done")
        rows, secs, errors = [], [], []
        done, chunk = 0, 16
        with open(out_dir / "results.jsonl", "w") as f:
            for c0 in range(0, len(all_jobs), chunk):
                part = all_jobs[c0:c0 + chunk]
                res = Parallel(n_jobs=args.workers, verbose=0)(delayed(solve_problem)(j) for j in part)
                for r, t in res:
                    secs.append(t)
                    for row in r:
                        if "error" in row:
                            errors.append(row)
                        else:
                            rows.append(row)
                            f.write(json.dumps({**row, "runs": [{k: v for k, v in x.items() if k != "traj"}
                                                                for x in row.get("runs", [])]}, default=float) + "\n")
                f.flush()
                done += len(part)
                progress(res_root, done, len(all_jobs), "problems", {"rows": len(rows), "errors": len(errors)})
        wall = time.time() - t_start
        if errors:
            (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        summary = analyse(pd, rows, errors, out_dir, pilot, wall, t_val, val_sampler, val_e2e, secs, args.workers,
                          len(jobs))
        mark_done(res_root, "success", summary.get("verdict_line", ""))
        print(json.dumps(summary.get("gate", {}), indent=1, default=float))
    except Exception:
        tb = traceback.format_exc()
        (out_dir / "run_error.txt").write_text(tb)
        print(tb)
        mark_done(res_root, "failed", tb[-500:])
        raise


if __name__ == "__main__":
    main()
