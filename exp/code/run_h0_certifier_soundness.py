"""h0_certifier_soundness: dual / BnB certifier vs exact enumeration (hard soundness gate).

Usage: run_h0_certifier_soundness.py --mode {pilot,full}
Pilot: 100 (instance, Theta_t snapshot, problem) triples on E1-NL-S (+ 20 E1-NL-M extra triples, not gated)
       + T-Lin assertion (dual bound == ellipsoid closed form) on E1-Lin problems.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.certify.dual_bnb import block_digits, certify_dual, lin_copy_dual, lin_mu_dual  # noqa: E402
from dsswm.certify.enum_exact import certify_enum  # noqa: E402
from dsswm.certify.lin_closed import certify_lin  # noqa: E402
from dsswm.core.dynamics import next_loads  # noqa: E402
from dsswm.evidence.ellipsoid import EllipsoidSet  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_contrib import contrib_table  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.streams.generator import (DEFAULT_NL_M_GRID, DEFAULT_NL_S_GRID, LIN_DEFAULTS, NL_DEFAULTS,  # noqa: E402
                                     generator_hash, make_lin_instance, make_nl_instance)
from dsswm.streams.utilities import Utility  # noqa: E402

TASK = "h0_certifier_soundness"
TOL = 1e-9
DELTA = 0.05


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
                                                        "final_progress": fp,
                                                        "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------ NL snapshots (GPU phase)
def nl_snapshots(env_name, grid, seeds, t_list, probs_per_snap, cache_dir, log):
    ncl = NLClass(grid)
    params = ncl.torch_params()
    digits = block_digits(ncl._radices, ncl.B)
    radices = [int(r) for r in ncl._radices]
    prop = None
    jobs = []
    for s in seeds:
        inst = make_nl_instance(s, grid=grid, nl_class=ncl)
        env = inst.env
        if prop is None:
            prop = NLPropagator(grid.L, grid.R, NL_DEFAULTS["nmax"], env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"])
        LT, _ = prop.tables(params)
        lr = SeqLRSet(prop, LT, DELTA)
        for o in inst.init_obs:
            lr.update(o)
        rng = np.random.default_rng([s, 4242])
        snaps = {}
        t = len(inst.init_obs)
        for tt in sorted(t_list):
            while t < tt:
                lr.update(env.step(int(rng.integers(env.aspace.n))))   # real env.step() data only
                t += 1
            snaps[tt] = (lr.mask().cpu().numpy().copy(), lr.mle())
        ti = inst.truth["theta_index"]
        contrib_cache = {}
        for si, tt in enumerate(sorted(t_list)):
            mask, kh = snaps[tt]
            for r in range(probs_per_snap):
                q = inst.problems[(probs_per_snap * si + r) % len(inst.problems)]
                if q.pid not in contrib_cache:
                    raw_u = Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0)
                    t0 = time.perf_counter()
                    raw = contrib_table(prop, params, q.policies, q.loads0, q.engaged0, q.H, raw_u)
                    ct_s = time.perf_counter() - t0
                    Jraw = raw.sum(-1)
                    cpath = cache_dir / f"{env_name}_{q.pid}_raw.npy"
                    # always verify attribution against the independent J-table propagation
                    Jchk = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, raw_u)
                    cache_err = float(np.abs(Jchk - Jraw).max())
                    if not cpath.exists():
                        np.save(cpath, Jchk)
                    c_q = 1.0 / float(Jraw.max())
                    contrib_cache[q.pid] = (raw * c_q, ct_s, cache_err)
                C, ct_s, cache_err = contrib_cache[q.pid]
                jobs.append({"env": env_name, "instance": s, "t": tt, "problem": q.pid, "H": q.H,
                             "n_policies": len(q.policies), "policies": [p.name for p in q.policies],
                             "C": C, "mask": mask, "mle": kh, "theta_star": ti, "digits": digits, "radices": radices,
                             "contrib_s": ct_s, "attr_vs_jtable_err": cache_err})
        log(f"{env_name} instance {s}: snapshots done, set sizes "
            f"{[int(snaps[tt][0].sum()) for tt in sorted(t_list)]}")
    return jobs, ncl


def eval_triple(job, eps_list):
    C, mask, kh = job["C"], job["mask"], job["mle"]
    J = C.sum(-1)
    eps = eps_list[0]
    t0 = time.perf_counter()
    ex = certify_enum(J, mask, kh, eps)
    ex_s = time.perf_counter() - t0
    out = {k: v for k, v in job.items() if k not in ("C", "mask", "digits", "radices")}
    out["set_size"] = int(mask.sum())
    out["theta_star_in_set"] = bool(mask[job["theta_star"]])
    out["exact_r_bar"] = ex["r_bar"]
    out["exact_s"] = ex_s
    k_hat = ex["pi_hat"]
    out["pi_hat"] = k_hat
    res = {}
    for nh in (0, 1, 2):
        r = certify_dual(C, mask, k_hat, job["digits"], job["radices"], eps, n_hubs=nh, exact=True,
                         bnb_scope="challenger")
        res[nh] = r
    r0, r1, r2 = res[0], res[1], res[2]
    out["dual_r_bar"] = r0["r_bar_ub"]
    out["bnb1_r_bar"] = r1["r_bar_ub"]
    out["bnb2_r_bar"] = r2["r_bar_ub"]
    out["lb_incumbent"] = max(r0["r_bar_lb"], r1["r_bar_lb"], r2["r_bar_lb"])
    out["rect_r_bar"] = max([0.0] + [c["rect"] for c in r0["per_challenger"]])
    out["dual_s"], out["bnb1_s"], out["bnb2_s"] = r0["wall_clock_s"], r1["wall_clock_s"], r2["wall_clock_s"]
    pcs = []
    for c0, c1, c2 in zip(r0["per_challenger"], r1["per_challenger"], r2["per_challenger"]):
        pcs.append({"challenger": c0["challenger"], "exact": c0["exact"], "dual": c0["dual"], "lp_value": c0["lp_value"],
                    "rect": c0["rect"], "bnb1": c1["bnb"], "bnb2": c2["bnb"], "incumbent": c0["incumbent"],
                    "kappa_participant": c0["kappa_participant"], "lp_ok": c0["lp_ok"],
                    "bnb1_hubs": c1["bnb_hubs"], "bnb2_hubs": c2["bnb_hubs"],
                    "bnb2_n_leaves": c2.get("bnb_n_leaves"), "bnb2_n_lp": c2.get("bnb_n_lp")})
    out["per_challenger"] = pcs
    # soundness (per challenger and certificate level)
    viol = []
    for c in pcs:
        for key in ("dual", "bnb1", "bnb2", "rect"):
            if c[key] < c["exact"] - TOL:
                viol.append({"challenger": c["challenger"], "bound": key, "value": c[key], "exact": c["exact"]})
        if c["incumbent"] > c["exact"] + TOL:
            viol.append({"challenger": c["challenger"], "bound": "incumbent_above_exact", "value": c["incumbent"],
                         "exact": c["exact"]})
        if c["lp_value"] is not None and abs(c["lp_value"] - c["dual"]) > 1e-6:
            viol.append({"challenger": c["challenger"], "bound": "lp_vs_lagrangian_mismatch(not a soundness issue)",
                         "value": c["dual"], "exact": c["lp_value"], "benign": True})
    for key in ("dual_r_bar", "bnb1_r_bar", "bnb2_r_bar"):
        if out[key] < out["exact_r_bar"] - TOL:
            viol.append({"bound": key, "value": out[key], "exact": out["exact_r_bar"]})
    out["violations"] = viol
    out["n_soundness_violations"] = sum(1 for v in viol if not v.get("benign"))
    # statuses
    for e in eps_list:
        ex_st = "CERTIFIED" if out["exact_r_bar"] <= e else "NEED_DATA"
        out[f"exact_status_eps{e}"] = ex_st
        for name in ("dual", "bnb1", "bnb2"):
            ub = out[f"{name}_r_bar"]
            st = "CERTIFIED" if ub <= e else ("NEED_DATA" if out["lb_incumbent"] > e else "COMPUTE_UNKNOWN")
            out[f"{name}_status_eps{e}"] = st
    # gaps
    den = max(out["exact_r_bar"], eps)
    for name in ("dual", "bnb1", "bnb2", "rect"):
        out[f"{name}_rel_gap"] = (out[f"{name}_r_bar"] - out["exact_r_bar"]) / den
    # binding challenger kappa (challenger achieving exact r_bar)
    b = max(pcs, key=lambda c: c["exact"]) if pcs else None
    out["kappa_participant_binding"] = b["kappa_participant"] if b else float("nan")
    out["binding_challenger"] = b["challenger"] if b else None
    return out


# ------------------------------------------------------------------ T-Lin assertion
def lin_parts(lc, policy, loads0, H, utility, aspace):
    L, R, P = lc.L, lc.R, lc.L + lc.R
    loads = np.asarray(loads0, np.int64).copy()
    parts = np.zeros((P, lc.d))
    ones = np.ones(P, dtype=np.int64)
    for t in range(H):
        a_idx = policy.act(t, loads, ones)
        a = aspace.actions[a_idx]
        inc = {(i, j): l for i, j, l in a.incentives}
        s = utility.c_q * utility.w[t]
        for i, j in a.pairs:
            b = inc.get((i, j), 0)
            xi = np.zeros(lc.d); xj = np.zeros(lc.d)
            xi[i] = 1.0
            xi[L + R + i] = -float(loads[i])
            xj[L + j] = 1.0
            if b >= 1:
                xi[2 * L + R + b - 1] += 0.5
                xj[2 * L + R + b - 1] += 0.5
            parts[i] += s * xi
            parts[L + j] += s * xj
        loads = next_loads(loads, aspace, a_idx, lc.nmax, lc.static)
    return parts


def lin_check(seeds):
    cfg = dict(LIN_DEFAULTS)
    pr = cfg["prior"]
    hi = [max(abs(a), abs(b)) for a, b in (pr["alpha"],) * 3 + (pr["beta"],) * 3 + (pr["gamma"],) * 3
          + (pr["psi"], pr["psi2"])]
    S = float(np.sqrt(np.sum(np.square(hi))))
    rows = []
    for s in seeds:
        inst = make_lin_instance(s)
        env = inst.env
        lc = LinClass(3, 3, env.aspace.nb, nmax=3)
        ell = EllipsoidSet(lc, sigma=cfg["sigma"], delta=DELTA, S=S)
        for o in inst.init_obs:
            ell.update(o)
        th, V, beta = ell.theta_hat(), ell.V, ell.sqrt_beta() ** 2
        for q in inst.problems:
            parts = [lin_parts(lc, p, q.loads0, q.H, q.utility, env.aspace) for p in q.policies]
            Z = np.array([pp.sum(0) for pp in parts])
            Zref = np.array([lc.z(p, q.loads0, q.H, q.utility, env.aspace) for p in q.policies])
            cert = certify_lin(Z, ell, 0.05)
            k = cert["pi_hat"]
            for c in range(len(q.policies)):
                if c == k:
                    continue
                d = Z[c] - Z[k]
                dp = parts[c] - parts[k]
                closed = float(ell.sup_linear(d)[0])
                mu = lin_mu_dual(d, th, V, beta)
                cp = lin_copy_dual(dp, th, V, beta)
                w = float(ell.width(d)[0])
                degenerate = w < 1e-9 * max(1.0, float(np.abs(dp).sum()))
                kap = float("nan") if degenerate else float(sum(ell.width(dp[p])[0] for p in range(dp.shape[0])) / w)
                rows.append({"instance": s, "problem": q.pid, "H": q.H, "challenger": c, "closed": closed,
                             "mu_dual": mu, "copy_dual_numeric": cp["opt_numeric"], "copy_dual_analytic": cp["opt_analytic"],
                             "rect_nu0": cp["rect_nu0"], "attr_err": float(np.abs(Z - Zref).max()),
                             "err_mu": abs(mu - closed), "err_copy": abs(cp["opt_numeric"] - closed),
                             "err_copy_analytic": abs(cp["opt_analytic"] - closed),
                             "kappa_participant": kap,
                             "degenerate_d": bool(degenerate),
                             "rect_width_ratio": float("nan") if degenerate else
                             (cp["rect_nu0"] - float(d @ th)) / (closed - float(d @ th))})
    return rows


# ------------------------------------------------------------------ summary helpers
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return {str(q): float(np.quantile(x, q)) for q in qs} if x else {}


def summarize(rows, eps_list):
    out = {"n_triples": len(rows), "soundness_violations": int(sum(r["n_soundness_violations"] for r in rows))}
    for name in ("dual", "bnb1", "bnb2", "rect"):
        out[f"{name}_rel_gap_median"] = _med([r[f"{name}_rel_gap"] for r in rows])
        out[f"{name}_rel_gap_quantiles"] = _q([r[f"{name}_rel_gap"] for r in rows])
        big = [r for r in rows if r["exact_r_bar"] >= eps_list[0]]
        out[f"{name}_ratio_median_exact_ge_eps"] = _med([r[f"{name}_r_bar"] / r["exact_r_bar"] for r in big])
        out[f"{name}_abs_gap_median"] = _med([r[f"{name}_r_bar"] - r["exact_r_bar"] for r in rows])
    out["n_exact_ge_eps"] = int(sum(r["exact_r_bar"] >= eps_list[0] for r in rows))
    pcs = [c for r in rows for c in r["per_challenger"]]
    out["n_challenger_bounds"] = len(pcs)
    for name in ("dual", "bnb1", "bnb2", "rect"):
        rel = [(c[name] - c["exact"]) / max(abs(c["exact"]), eps_list[0]) for c in pcs]
        out[f"challenger_{name}_rel_gap_median"] = _med(rel)
        out[f"challenger_{name}_exact_frac"] = float(np.mean([c[name] - c["exact"] <= 1e-9 for c in pcs])) if pcs else None
    out["lp_vs_lagrangian_max_abs_diff"] = float(max([abs(c["lp_value"] - c["dual"]) for c in pcs
                                                      if c["lp_value"] is not None] + [0.0]))
    out["lp_failures"] = int(sum(not c["lp_ok"] for c in pcs))
    for e in eps_list:
        st = {}
        for name in ("exact", "dual", "bnb1", "bnb2"):
            vals = [r[f"{name}_status_eps{e}"] for r in rows]
            st[name] = {s: int(vals.count(s)) for s in sorted(set(vals))}
        st["compute_unknown_rate"] = {n: float(np.mean([r[f"{n}_status_eps{e}"] == "COMPUTE_UNKNOWN" for r in rows]))
                                      for n in ("dual", "bnb1", "bnb2")}
        st["false_certify"] = {n: int(sum(r[f"{n}_status_eps{e}"] == "CERTIFIED" and r[f"exact_status_eps{e}"] != "CERTIFIED"
                                          for r in rows)) for n in ("dual", "bnb1", "bnb2")}
        st["lost_certifications"] = {n: int(sum(r[f"{n}_status_eps{e}"] != "CERTIFIED" and r[f"exact_status_eps{e}"] == "CERTIFIED"
                                                for r in rows)) for n in ("dual", "bnb1", "bnb2")}
        out[f"status_eps{e}"] = st
    # kappa correlation (challenger level, rectangular and optimised dual slack)
    kp = [(c["kappa_participant"], (c["rect"] - c["exact"]) / max(abs(c["exact"]), eps_list[0]),
           (c["dual"] - c["exact"]) / max(abs(c["exact"]), eps_list[0])) for c in pcs
          if c["kappa_participant"] is not None and np.isfinite(c["kappa_participant"])]
    if len(kp) > 5:
        k, gr, gd = map(np.array, zip(*kp))
        out["spearman_gap_vs_kappa_participant"] = {
            "rect_nu0": float(spearmanr(k, gr).statistic), "dual_opt": float(spearmanr(k, gd).statistic), "n": len(kp),
            "kappa_quantiles": _q(list(k))}
    tri = [(r["kappa_participant_binding"], r["dual_rel_gap"], r["rect_rel_gap"]) for r in rows
           if np.isfinite(r["kappa_participant_binding"])]
    if len(tri) > 5:
        k, gd, gr = map(np.array, zip(*tri))
        out["spearman_triple_level"] = {"dual_opt": float(spearmanr(k, gd).statistic),
                                        "rect_nu0": float(spearmanr(k, gr).statistic), "n": len(tri)}
    out["wall_clock_s_median"] = {n: _med([r[f"{n}_s"] for r in rows]) for n in ("exact", "dual", "bnb1", "bnb2")}
    out["wall_clock_s_max"] = {n: float(max(r[f"{n}_s"] for r in rows)) for n in ("exact", "dual", "bnb1", "bnb2")}
    out["by_t"] = {}
    for t in sorted(set(r["t"] for r in rows)):
        rr = [r for r in rows if r["t"] == t]
        out["by_t"][str(t)] = {"n": len(rr), "set_size_median": _med([r["set_size"] for r in rr]),
                               "exact_r_bar_median": _med([r["exact_r_bar"] for r in rr]),
                               "dual_rel_gap_median": _med([r["dual_rel_gap"] for r in rr]),
                               "bnb2_rel_gap_median": _med([r["bnb2_rel_gap"] for r in rr]),
                               "rect_rel_gap_median": _med([r["rect_rel_gap"] for r in rr]),
                               "theta_star_coverage": float(np.mean([r["theta_star_in_set"] for r in rr]))}
    out["theta_star_coverage"] = float(np.mean([r["theta_star_in_set"] for r in rows]))
    out["attr_vs_jtable_max_err"] = float(max(r["attr_vs_jtable_err"] for r in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    T0 = time.time()
    start_iso = datetime.now().isoformat()
    np.random.seed(42)
    torch.manual_seed(42)
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    cache_dir = WS / "exp" / "cache" / "jtables"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    (out_dir / "start_time.txt").write_text(start_iso)
    logf = open(out_dir / "run.log", "w")

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    eps_list = [0.05, 0.02]
    if args.mode == "pilot":
        s_seeds, t_list, pps = list(range(10)), [20, 50, 100, 200, 400], 2
        m_seeds, m_t = list(range(4)), [20, 50, 100, 200, 400]
        lin_seeds = list(range(10))
    else:
        s_seeds, t_list, pps = list(range(10000, 10060)), [20, 50, 100, 200, 400], 2
        m_seeds, m_t = list(range(10000, 10045)), [20, 50, 100, 200, 400]
        lin_seeds = list(range(10000, 10040))
    TOTAL = 6
    progress(res_root, 0, TOTAL, "tlin_check")
    lin_rows = lin_check(lin_seeds)
    lin_sum = {"n_checks": len(lin_rows),
               "max_err_mu_dual": float(max(r["err_mu"] for r in lin_rows)),
               "max_err_copy_dual_numeric": float(max(r["err_copy"] for r in lin_rows)),
               "max_err_copy_dual_analytic": float(max(r["err_copy_analytic"] for r in lin_rows)),
               "max_attr_err": float(max(r["attr_err"] for r in lin_rows)),
               "max_abs_rect_ratio_minus_kappa": float(np.nanmax([abs(r["rect_width_ratio"] - r["kappa_participant"])
                                                                  for r in lin_rows])),
               "kappa_participant_quantiles": _q([r["kappa_participant"] for r in lin_rows])}
    lin_sum["pass"] = bool(lin_sum["max_err_mu_dual"] < 1e-6 and lin_sum["max_err_copy_dual_numeric"] < 1e-6
                           and lin_sum["max_err_copy_dual_analytic"] < 1e-6)
    log(f"T-Lin check: {json.dumps(lin_sum)}")

    progress(res_root, 1, TOTAL, "nl_s_snapshots")
    jobs_s, ncl_s = nl_snapshots("E1-NL-S", DEFAULT_NL_S_GRID, s_seeds, t_list, pps, cache_dir, log)
    progress(res_root, 2, TOTAL, "nl_m_snapshots")
    jobs_m, ncl_m = nl_snapshots("E1-NL-M", DEFAULT_NL_M_GRID, m_seeds, m_t, 1, cache_dir, log)
    gpu_mem = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
    log(f"GPU phase done: {len(jobs_s)} NL-S + {len(jobs_m)} NL-M triples, peak VRAM {gpu_mem:.0f} MB, "
        f"{time.time() - T0:.1f}s")

    progress(res_root, 3, TOTAL, "dual_bnb_nl_s", {"n": len(jobs_s)})
    rows_s = Parallel(n_jobs=args.workers, verbose=0)(delayed(eval_triple)(j, eps_list) for j in jobs_s)
    log(f"NL-S dual/BnB done {time.time() - T0:.1f}s")
    progress(res_root, 4, TOTAL, "dual_bnb_nl_m", {"n": len(jobs_m)})
    rows_m = Parallel(n_jobs=args.workers, verbose=0)(delayed(eval_triple)(j, eps_list) for j in jobs_m)
    log(f"NL-M dual/BnB done {time.time() - T0:.1f}s")

    progress(res_root, 5, TOTAL, "summary")
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows_s + rows_m:
            f.write(json.dumps(r, default=float) + "\n")
    with open(out_dir / "tlin_check.jsonl", "w") as f:
        for r in lin_rows:
            f.write(json.dumps(r) + "\n")
    sum_s = summarize(rows_s, eps_list)
    sum_m = summarize(rows_m, eps_list) if rows_m else {}
    viol_all = sum_s["soundness_violations"] + sum_m.get("soundness_violations", 0)
    gate = {
        "soundness_violations_eq_0": viol_all == 0,
        "dual_rel_gap_median_lt_0.20": bool(sum_s["dual_rel_gap_median"] is not None and sum_s["dual_rel_gap_median"] < 0.20),
        "tlin_dual_eq_closed_1e-6": lin_sum["pass"],
    }
    go = all(gate.values())
    # samples: most / least tight + typical
    srt = sorted(rows_s, key=lambda r: r["dual_rel_gap"])
    pick = [srt[0], srt[len(srt) // 4], srt[len(srt) // 2], srt[3 * len(srt) // 4], srt[-1]]
    pick += sorted(rows_s, key=lambda r: -r["kappa_participant_binding"] if np.isfinite(r["kappa_participant_binding"]) else 0)[:2]
    pick += [r for r in rows_m[:2]]
    for k, smp in enumerate(pick):
        (out_dir / "samples" / f"sample_{k:02d}_{smp['env']}_{smp['problem']}_t{smp['t']}.json").write_text(
            json.dumps(smp, indent=1, default=float))
    summary = {
        "task_id": TASK, "mode": args.mode, "seed": 42, "generator_hash": generator_hash(),
        "eps_list": eps_list, "delta": DELTA, "soundness_tol": TOL,
        "design": {"E1-NL-S": {"instances": s_seeds, "t_snapshots": t_list, "problems_per_snapshot": pps,
                               "Theta_size": int(ncl_s.B), "blocks": "left1, left2, right1, right2, global",
                               "radices": [int(x) for x in ncl_s._radices]},
                   "E1-NL-M": {"instances": m_seeds, "t_snapshots": m_t, "problems_per_snapshot": 1,
                               "Theta_size": int(ncl_m.B), "radices": [int(x) for x in ncl_m._radices], "gated": False},
                   "capacity_coupled": "not run in pilot (variant not yet implemented; scheduled for full)",
                   "snapshot_data": "n0=20 shared initial rounds + uniform random legal actions on the real env up to t",
                   "rel_gap_definition": "(UB - exact R_bar) / max(exact R_bar, eps=0.05), certificate level",
                   "dual": "participant-copy Lagrangian with block-marginal consistency multipliers (LP via HiGHS, "
                           "bound re-evaluated as L(nu) with sum-zero nu)",
                   "bnb": "branch on the block of the 1 or 2 participants with widest attributed decision-gap range"},
        "metrics": {"E1-NL-S": sum_s, "E1-NL-M": sum_m, "T-Lin": lin_sum},
        "gate": gate, "go_no_go": "GO" if go else "NO_GO",
        "gpu": {"name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "peak_vram_mb": gpu_mem},
        "wall_clock_total_s": time.time() - T0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (res_root / f"{TASK}_gpu_profile.json").write_text(json.dumps({
        "gpu_name": summary["gpu"]["name"], "vram_total_mb": 24564, "max_batch_size": int(ncl_s.B),
        "batch_unit": "theta per contribution-table chunk (whole class in <=4 chunks)",
        "vram_used_mb": gpu_mem, "utilization_pct": 100 * gpu_mem / 24564,
        "note": "GPU only builds per-participant J attribution tables; dual LPs and BnB are CPU-bound (joblib "
                f"{args.workers} workers). VRAM is not the bottleneck."}, indent=1))
    log(f"GATE {gate} -> {summary['go_no_go']}; total {time.time() - T0:.1f}s")
    progress(res_root, 6, TOTAL, "done", {"go_no_go": summary["go_no_go"]})
    mark_done(res_root, "success", f"{summary['go_no_go']}; violations={viol_all}; "
                                   f"dual_gap_med={sum_s['dual_rel_gap_median']}")
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # always leave a DONE marker
        import traceback
        traceback.print_exc()
        mark_done(WS / "exp" / "results", "failed", repr(e))
        raise
