"""g0_resolution_gate (P0): resolution gate for the off-grid (R1) ladder (methodology 2.2, hypotheses G0 / HR4).

Per resolution f in {0.5, 1, 2, 4} (G_f refines only alpha / psi):
  * eta_loc(h)/eps per problem (median over cells, value at the theta* cell);
  * validity: 1e4 random theta per f (uniform cell, uniform inside the cell) -> |J_theta - J_g| <= eta_loc(g), 0 violations;
  * J-table seconds per problem.
Per f in JPC_FS (uncorrected JPC on G_f, R1 continuous truth, dev seeds only):
  * DEC-ID: survival of the theta* cell and of the KL-projection g° cell at the JPC stopping time of each problem;
    g° = argmin_g sum_t KL(P_theta*(.|s_t,a_t) || P_g(.|s_t,a_t)) over the realised design (exact per-round KL);
  * eta_dec(g°) = max_pi |J_true - J_g°| (and the same for the nearest grid point: baseline 'nearest vs KL-proj.');
  * R1 FCR (uncorrected certification, true regret under the continuous theta*);
  * h* = eps / (L_J sqrt(cond I_xi)), I_xi = design Fisher (all rounds so far) at the grid MLE; d_eff = PR(I_xi);
  * would JPC_infl certify the final set? (R_grid + 2 max_alive eta_loc <= eps') for eps' in {0.02, 0.05, 0.10}.
Gate (hypotheses.md G0): 0 violations AND some f with median eta_loc <= eps/4 AND J-table <= 10 s/problem at that f;
if none at eps=0.02, report reachability at eps in {0.05, 0.10}.
Dev seeds only: pilot 600-611 x 10 problems, full 600-699 x 15 problems. Evaluation seeds are never touched.

Usage: run_g0_resolution_gate.py --mode {pilot,full} [--workers 4]
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
TASK = "g0_resolution_gate"
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache"
EPS, DELTA = 0.02, 0.05
EPS_RELAX = (0.02, 0.05, 0.10)
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
NAMES = ["alpha0", "alpha1", "gamma0", "gamma1", "tauL0", "tauL1", "beta0", "beta1", "tauR0", "tauR1", "psi", "lam"]

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose, kl_features, single_prob_tables  # noqa: E402,E501
from dsswm.baselines.glm_linearised import fisher_rounds  # noqa: E402
from dsswm.certify.eta_loc import build_plans, eta_loc, j_values  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, group_ids, observable_signature, regret_matrix, trichotomy  # noqa: E402,E501
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet, participation_ratio  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.grid_ladder import LADDER, cell_index_of, class_vectors, half_widths, ladder_grid, sample_in_cells  # noqa: E402,E501
from dsswm.models.nl_class import NLClass  # noqa: E402
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


def U(q):
    return type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})()


# ------------------------------------------------------------------------------------------------ phase 1 (GPU)
def gpu_phase(probs, fs, eta_fs, n_draws, log, rows, dev, seed=42):
    """J tables, eta_loc and the validity check per f; returns per-f stats + per-problem c_q / J_true."""
    rng = np.random.default_rng(seed)
    prop = probs[0][2]
    cq, jtrue = {}, {}
    for (s, q, _, vst) in [(a, b, c, d) for (a, b, c, d) in probs]:
        d1 = CACHE / "jtables_f1" / f"{q.pid}_raw.npy"
        if not d1.exists():
            ncl1 = NLClass(ladder_grid(1), device=dev)
            np.save(d1, prop.j_table(ncl1.torch_params(), q.policies, q.loads0, q.engaged0, q.H, U(q)))
        cq[q.pid] = 1.0 / float(np.load(d1).max())
        jtrue[q.pid] = j_values(prop, build_plans(prop, q), vst[None], q.utility.w, q.utility.w_ret)[0] * cq[q.pid]
    stats = {}
    for f in fs:
        ncl = NLClass(ladder_grid(f), device=dev)
        params = None
        d = CACHE / f"jtables_f{fkey(f)}"
        d.mkdir(parents=True, exist_ok=True)
        W = half_widths(ncl)
        h_euclid = float(np.median(np.linalg.norm(W, axis=1)))
        jt_sec, e_med, e_star, e_min, viol, ratios, emp, lj, et_sec = [], [], [], [], 0, [], [], [], []
        P = probs if f in eta_fs["all"] else probs[: eta_fs[f]]
        per_draw = int(math.ceil(n_draws / len(P)))
        for (s, q, _, vst) in probs:
            p = d / f"{q.pid}_raw.npy"
            if not p.exists():
                params = params or ncl.torch_params()
                t0 = time.perf_counter()
                raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, U(q))
                jt_sec.append(time.perf_counter() - t0)
                np.save(p, raw.astype(np.float32) if f == 4 else raw)
                rows.append({"kind": "jtable", "f": f, "problem": q.pid, "sec": jt_sec[-1], "Theta": ncl.B})
        for (s, q, _, vst) in P:
            c = cq[q.pid]
            pe = d / f"{q.pid}_etaloc_raw.npy"
            t0 = time.perf_counter()
            eta, J, hb, info = eta_loc(prop, ncl, q, return_parts=True)
            et_sec.append(time.perf_counter() - t0)
            np.save(pe, eta)
            np.save(d / f"{q.pid}_LJ.npy", np.array([info["parts"]["L_J"]]))
            en = eta * c
            idx = rng.integers(ncl.B, size=per_draw)
            th = sample_in_cells(ncl, idx, rng)
            Jt = j_values(prop, build_plans(prop, q), th, q.utility.w, q.utility.w_ret)
            err = np.abs(Jt - J[idx]).max(1) * c
            r = err / en[idx]
            nv = int((err > en[idx] + 1e-12).sum())
            viol += nv
            ratios.append(r)
            emp.append(float(err.max()))
            star = int(cell_index_of(vst[None], ncl)[0])
            e_med.append(float(np.median(en)))
            e_min.append(float(en.min()))
            e_star.append(float(en[star]))
            lj.append(info["parts"]["L_J"] * c)
            rows.append({"kind": "eta_loc", "f": f, "problem": q.pid, "Theta": ncl.B, "sec": et_sec[-1],
                         "eta_over_eps_median": e_med[-1] / EPS, "eta_over_eps_min": e_min[-1] / EPS,
                         "eta_over_eps_theta_star_cell": e_star[-1] / EPS, "emp_sup_err_over_eps": emp[-1] / EPS,
                         "ratio_err_over_eta_max": float(r.max()), "ratio_err_over_eta_median": float(np.median(r)),
                         "violations": nv, "draws": per_draw, "L_J_norm": lj[-1],
                         "grad_term_by_coord": dict(zip(NAMES, (np.asarray(info["parts"]["grad_term_by_coord"])
                                                                 * c / EPS).round(3).tolist())),
                         "hess_term_max_over_eps": info["parts"]["hess_term_max"] * c / EPS})
        rr = np.concatenate(ratios)
        hist, edges = np.histogram(rr, bins=20, range=(0, 1))
        em = np.array(e_med)
        stats[fkey(f)] = {
            "Theta": ncl.B, "alpha_step": 3.0 / (2 * f), "psi_step": 1.5 / (2 * f),
            "h_cell_halfdiag_median": h_euclid, "n_problems_eta": len(P),
            "eta_over_eps_median": float(np.median(em)) / EPS,
            "eta_over_eps_q10_q90": [float(np.quantile(em, .1)) / EPS, float(np.quantile(em, .9)) / EPS],
            "eta_over_eps_min_cell_median": float(np.median(e_min)) / EPS,
            "eta_over_eps_theta_star_cell_median": float(np.median(e_star)) / EPS,
            "share_problems_median_eta_le_eps_over_4": {fkey(e): float(np.mean(em <= e / 4)) for e in EPS_RELAX},
            "share_problems_min_cell_eta_le_eps_over_4": {fkey(e): float(np.mean(np.array(e_min) <= e / 4))
                                                          for e in EPS_RELAX},
            "emp_sup_err_over_eps_median": float(np.median(emp)) / EPS,
            "share_problems_emp_err_le_eps_over_4": {fkey(e): float(np.mean(np.array(emp) <= e / 4))
                                                     for e in EPS_RELAX},
            "violations": viol, "draws": int(len(rr)),
            "ratio_err_over_eta": {"max": float(rr.max()), "median": float(np.median(rr)),
                                   "q99": float(np.quantile(rr, .99)), "hist_counts": hist.tolist(),
                                   "hist_edges": edges.round(3).tolist()},
            "L_J_norm_median": float(np.median(lj)),
            "jtable_sec_per_problem_new": float(np.mean(jt_sec)) if jt_sec else None,
            "eta_sec_per_problem_mean": float(np.mean(et_sec))}
        log(f"f={fkey(f)} |Theta|={ncl.B}: median eta/eps={stats[fkey(f)]['eta_over_eps_median']:.1f} "
            f"(theta* cell {stats[fkey(f)]['eta_over_eps_theta_star_cell_median']:.1f}), empirical sup err/eps "
            f"{stats[fkey(f)]['emp_sup_err_over_eps_median']:.2f}, violations {viol}/{len(rr)}, max err/eta "
            f"{rr.max():.3f}, eta {np.mean(et_sec):.2f}s/problem on {len(P)} problems")
        del ncl, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return stats, cq, jtrue


# ------------------------------------------------------------------------------------------------ phase 2 (CPU, JPC)
def jpc_instance(seed, n_prob, fs, cq, jtrue, tmax, want_samples):
    torch.set_num_threads(1)
    out, samples = [], []
    for f in fs:
        inst = make_offgrid_instance(seed, "R1", stream=0)
        env = inst.env
        h = env.handle()
        aspace = env.aspace
        ncl = NLClass(ladder_grid(f), device="cpu")
        prop = NLPropagator(2, 2, NMAX, aspace, C_KNOWN, RHO_RET, device="cpu")
        LT = prop.tables(ncl.torch_params())[0]
        LT_np = LT.numpy()
        py, pe = class_prob_tables(ncl.np_params, C_KNOWN, NMAX, aspace.nb)
        inc = IncidenceIndex(prop.codec, aspace, NMAX)
        legal = np.arange(aspace.n)
        gid = group_ids(observable_signature(py, pe, max_level=1))
        all_single = len(np.unique(gid)) == ncl.B
        V = class_vectors(ncl)
        tp = env.true_params()                                           # harness side only
        vst = vstar_of(inst.truth)
        star = int(cell_index_of(vst[None], ncl)[0])
        tpy, tpe = single_prob_tables(tp, C_KNOWN, NMAX, aspace.nb)
        Fkl = kl_features(tpy, tpe, py, pe)                              # (nf, B)
        lr = SeqLRSet(prop, LT, DELTA)
        led = EvidenceLedger([lr])
        for o in inst.init_obs:
            led.record(o, "initial")
        all_obs = list(inst.init_obs)
        rng = np.random.default_rng([seed, 0, 101])
        for k, q in enumerate(inst.problems[:n_prob]):
            d = CACHE / f"jtables_f{fkey(f)}"
            c = cq[q.pid]
            J = np.load(d / f"{q.pid}_raw.npy").astype(float) * c
            Reg = regret_matrix(J)
            eta = np.load(d / f"{q.pid}_etaloc_raw.npy") * c
            LJ = float(np.load(d / f"{q.pid}_LJ.npy")[0]) * c
            Jt = jtrue[q.pid]
            t0 = time.perf_counter()
            steps, status, traj = 0, None, []
            while True:
                mask = lr.mask().numpy()
                cert = certify_minimax(Reg, mask, EPS, TOP_M)
                sv = cert["status"].value
                if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                    status = sv
                    break
                if not all_single and steps % 10 == 0:
                    amb, _, _ = trichotomy(Reg, mask, gid, EPS)
                    if amb:
                        status = "OUT_OF_SCOPE"
                        break
                if steps >= tmax:
                    status = "NEED_DATA"
                    break
                kh = lr.mle()
                blk = cert["blocking"]
                margins = LOG_THR - lr.log_ratio().numpy()[blk]
                code = prop.codec.encode(*h.observable_state())
                a, info = dda_choose(code, py[kh:kh + 1], pe[kh:kh + 1], LT_np[kh], py[blk], pe[blk], margins, inc,
                                     prop, legal, rng)
                obs = h.step(a)
                led.record(obs, "probe", q.pid)
                all_obs.append(obs)
                steps += 1
                if want_samples and len(traj) < 40:
                    traj.append({"step": steps, "a": a, "set": int(mask.sum()), "r_bar": round(cert["r_bar"], 4),
                                 "mle": kh, "star_alive": bool(mask[star]), "mode": info["mode"]})
            assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
            mask = lr.mask().numpy()
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            # realised design: counts per (state code, action)
            pairs = {}
            for o in all_obs:
                key = (prop.codec.encode(o.loads, o.engaged), int(o.action))
                pairs[key] = pairs.get(key, 0) + 1
            cnt = np.zeros(inc.nf)
            for (code, a), n in pairs.items():
                cnt += n * inc.S[code * inc.nA + a]
            kl = cnt @ Fkl                                               # (B,) design KL from theta* to every g
            gc = int(np.argmin(kl))
            kh = lr.mle()
            I = np.zeros((12, 12))
            for (code, a), n in pairs.items():
                I += n * fisher_rounds(V[kh], prop.codec, aspace, code, [a], C_KNOWN, 2, 2)[0]
            ev = np.linalg.eigvalsh(I)
            lmax = float(ev.max())
            lmin = float(max(ev.min(), 1e-12 * lmax))
            cond = lmax / lmin
            hstar = EPS / (LJ * math.sqrt(cond))
            idn = ev[ev >= 1e-6 * lmax]
            cond_id = float(idn.max() / idn.min())
            hstar_id = EPS / (LJ * math.sqrt(cond_id))                 # identified-subspace variant (diagnostic)
            hstar_ub = EPS / LJ                                          # cond = 1: most favourable h*
            pi = cert["pi"]
            tr = float(Jt.max() - Jt[pi]) if pi is not None else None
            alive_eta = float(eta[mask].max()) if mask.any() else float("nan")
            row = {"kind": "jpc", "instance": seed, "f": f, "problem": q.pid, "k": k, "status": status,
                   "new_env_steps": steps, "censored": status != "CERTIFIED", "certified_policy": pi,
                   "true_regret": tr, "false_cert": bool(status == "CERTIFIED" and tr is not None and tr > EPS),
                   "set_size": int(mask.sum()), "r_bar_grid": cert["r_bar"],
                   "theta_cell_alive": bool(mask[star]), "gcirc_cell_alive": bool(mask[gc]),
                   "gcirc_eq_nearest": bool(gc == star), "kl_gcirc": float(kl[gc]), "kl_nearest": float(kl[star]),
                   "eta_dec_gcirc": float(np.abs(Jt - J[gc]).max()), "eta_dec_nearest": float(np.abs(Jt - J[star]).max()),
                   "eta_loc_star_cell": float(eta[star]), "eta_loc_gcirc_cell": float(eta[gc]),
                   "eta_loc_alive_max": alive_eta, "eta_dec_over_eta_loc_star": float(np.abs(Jt - J[star]).max() / eta[star]),
                   "infl_certifiable": {fkey(e): bool(mask.any() and cert["r_bar"] + 2 * alive_eta <= e) for e in EPS_RELAX},
                   "floor_star_cell_ok": {fkey(e): bool(2 * eta[star] <= e) for e in EPS_RELAX},
                   "fisher_cond": cond, "fisher_lmin": float(ev.min()), "fisher_lmax": lmax,
                   "d_eff": participation_ratio(I), "L_J_norm": LJ, "h_star": hstar,
                   "fisher_cond_identified": cond_id, "n_identified_dirs": int(len(idn)), "h_star_identified": hstar_id,
                   "h_star_cond1": hstar_ub,
                   "n_rounds_total": len(all_obs), "wall_clock_s": time.perf_counter() - t0}
            out.append(row)
            if want_samples and k < 3:
                samples.append({**row, "J_true": Jt.round(4).tolist(), "J_gcirc": J[gc].round(4).tolist(),
                                "J_nearest": J[star].round(4).tolist(), "theta_star": vst.round(4).tolist(),
                                "gcirc": V[gc].tolist(), "nearest": V[star].tolist(), "mle": V[kh].tolist(),
                                "trajectory_head": traj})
    return out, samples


def summarize_jpc(rows, fs, eta_stats):
    from dsswm.stats.cp import clopper_pearson
    s = {}
    for f in fs:
        R = [r for r in rows if r["f"] == f]
        if not R:
            continue
        cert = [r for r in R if r["status"] == "CERTIFIED"]
        nfc = sum(r["false_cert"] for r in R)
        lo, hi = clopper_pearson(nfc, len(R), 0.05) if R else (None, None)
        hs = np.array([r["h_star"] for r in R])
        hcell = eta_stats[fkey(f)]["h_cell_halfdiag_median"]
        edec = np.array([r["eta_dec_gcirc"] for r in R])
        s[fkey(f)] = {
            "n_problems": len(R), "completion": len(cert) / len(R),
            "status_counts": {k: sum(r["status"] == k for r in R) for k in sorted({r["status"] for r in R})},
            "steps_median": float(np.median([r["new_env_steps"] for r in R])),
            "r1_fcr": nfc / len(R), "r1_fcr_cp": [lo, hi],
            "r1_fcr_among_certified": nfc / max(len(cert), 1),
            "theta_cell_survival": float(np.mean([r["theta_cell_alive"] for r in R])),
            "gcirc_cell_survival": float(np.mean([r["gcirc_cell_alive"] for r in R])),
            "gcirc_eq_nearest_rate": float(np.mean([r["gcirc_eq_nearest"] for r in R])),
            "eta_dec_gcirc_over_eps_median": float(np.median(edec)) / EPS,
            "eta_dec_nearest_over_eps_median": float(np.median([r["eta_dec_nearest"] for r in R])) / EPS,
            "share_eta_dec_ge_eps_half": float(np.mean(edec >= EPS / 2)),
            "eta_dec_over_eta_loc_star_median": float(np.median([r["eta_dec_over_eta_loc_star"] for r in R])),
            "infl_certifiable_rate": {fkey(e): float(np.mean([r["infl_certifiable"][fkey(e)] for r in R]))
                                      for e in EPS_RELAX},
            "floor_star_cell_ok_rate": {fkey(e): float(np.mean([r["floor_star_cell_ok"][fkey(e)] for r in R]))
                                        for e in EPS_RELAX},
            "h_star": {"median": float(np.median(hs)), "q10": float(np.quantile(hs, .1)),
                       "q90": float(np.quantile(hs, .9))},
            "h_star_identified": {"median": float(np.median([r["h_star_identified"] for r in R])),
                                  "q10": float(np.quantile([r["h_star_identified"] for r in R], .1)),
                                  "q90": float(np.quantile([r["h_star_identified"] for r in R], .9))},
            "h_star_cond1_median": float(np.median([r["h_star_cond1"] for r in R])),
            "n_identified_dirs_median": float(np.median([r["n_identified_dirs"] for r in R])),
            "h_over_hstar_cond1_median": float(np.median([hcell / r["h_star_cond1"] for r in R])),
            "h_cell_halfdiag": hcell, "h_over_hstar_median": float(np.median(hcell / hs)),
            "share_h_le_hstar_over_2": float(np.mean(hcell <= hs / 2)),
            "fisher_cond_median": float(np.median([r["fisher_cond"] for r in R])),
            "d_eff_median": float(np.median([r["d_eff"] for r in R])),
            "d_eff_q10_q90": [float(np.quantile([r["d_eff"] for r in R], .1)),
                              float(np.quantile([r["d_eff"] for r in R], .9))],
            "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in R]))}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="1 seed x 2 problems, results to a smoke dir")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    seeds = list(range(600, 612)) if pilot else list(range(600, 700))
    n_prob = 10 if pilot else 15
    if args.smoke:
        seeds, n_prob = [600], 2
    assert max(seeds) < 10000, "dev seeds only"
    eta_fs = {"all": (0.5, 1, 2), 4: 2 if args.smoke else (60 if pilot else 300)}
    jpc_fs = (1, 2)
    tmax = NL_DEFAULTS.get("T_max_stepwise", 3000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    log(f"start mode={args.mode} device={dev} seeds={seeds[0]}-{seeds[-1]} x {n_prob} problems, jpc f={jpc_fs}, "
        f"tmax={tmax}, generator_hash={generator_hash()}")
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA,
               "dev_seeds": [seeds[0], seeds[-1]], "n_problems_per_instance": n_prob, "eval_seeds_touched": False,
               "generator_hash": generator_hash(), "concurrent_run": True, "tmax_stepwise": tmax}
    rows = []
    try:
        progress(1, 4, "gpu: J tables / eta_loc / validity")
        prop = NLPropagator(2, 2, NMAX, make_offgrid_instance(seeds[0], "R1").env.aspace, C_KNOWN, RHO_RET, device=dev)
        probs = []
        for s in seeds:
            inst = make_offgrid_instance(s, "R1", stream=0)
            vst = vstar_of(inst.truth)
            probs += [(s, q, prop, vst) for q in inst.problems[:n_prob]]
        eta_stats, cq, jtrue = gpu_phase(probs, LADDER, eta_fs, 10000, log, rows, dev)
        summary["eta_loc"] = eta_stats
        # setup-phase J-table timing (same machine, same day) for f where tables were cached
        try:
            st = json.loads((RES_ROOT / "pilots" / "r2_setup_offgrid" / "summary.json").read_text())
            summary["jtable_sec_per_problem_by_f"] = st["metrics"]["jtable_sec_per_problem_by_f"]
        except Exception:  # noqa: BLE001
            summary["jtable_sec_per_problem_by_f"] = {}
        del prop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress(2, 4, "cpu: JPC on R1 (DEC-ID / FCR / h*)")
        from joblib import Parallel, delayed
        t1 = time.time()
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(jpc_instance)(s, n_prob, jpc_fs, cq, jtrue, tmax, i < 3) for i, s in enumerate(seeds))
        jrows = [r for rr, _ in res for r in rr]
        samp = [x for _, ss in res for x in ss]
        rows += jrows
        log(f"JPC phase: {len(jrows)} (problem, f) rows in {time.time() - t1:.0f}s")
        (samples_dir / "jpc_samples.json").write_text(json.dumps(samp, indent=1, default=float))
        jsum = summarize_jpc(jrows, jpc_fs, eta_stats)
        summary["jpc_r1"] = jsum
        for f, v in jsum.items():
            log(f"JPC R1 f={f}: completion {v['completion']:.2f}, FCR {v['r1_fcr']:.3f} CP {v['r1_fcr_cp']}, "
                f"theta* cell survival {v['theta_cell_survival']:.2f}, g° survival {v['gcirc_cell_survival']:.2f}, "
                f"eta_dec(g°)/eps med {v['eta_dec_gcirc_over_eps_median']:.2f}, h* med {v['h_star']['median']:.2e}, "
                f"h/h* med {v['h_over_hstar_median']:.1f}, infl-cert {v['infl_certifiable_rate']}, d_eff "
                f"{v['d_eff_median']:.2f}")
        progress(3, 4, "gate")
        jt = summary["jtable_sec_per_problem_by_f"]
        gate = {}
        for e in EPS_RELAX:
            ok_f = [f for f in LADDER if eta_stats[fkey(f)]["eta_over_eps_median"] * EPS <= e / 4
                    and float(jt.get(fkey(f), 0.0)) <= 10]
            gate[fkey(e)] = {"f_reaching_eps_over_4": [fkey(f) for f in ok_f], "reachable": bool(ok_f),
                             "best_median_eta": min(eta_stats[fkey(f)]["eta_over_eps_median"] * EPS for f in LADDER),
                             "best_emp_lower_bound": min(eta_stats[fkey(f)]["emp_sup_err_over_eps_median"] * EPS
                                                         for f in LADDER)}
        viol = int(sum(eta_stats[fkey(f)]["violations"] for f in LADDER))
        passed = viol == 0 and gate["0.02"]["reachable"]
        summary["gate"] = {"violations_total": viol, "draws_total": int(sum(eta_stats[fkey(f)]["draws"] for f in LADDER)),
                           "by_eps": gate, "pass_eps_0.02": bool(passed),
                           "pass_any_eps": bool(viol == 0 and any(g["reachable"] for g in gate.values())),
                           "f_star_candidate": (gate["0.02"]["f_reaching_eps_over_4"] or [None])[0]}
        summary["metrics"] = {
            "eta_loc_over_eps_by_f": {fkey(f): eta_stats[fkey(f)]["eta_over_eps_median"] for f in LADDER},
            "eta_bar_violations": viol,
            "theta_cell_survival": {f: v["theta_cell_survival"] for f, v in jsum.items()},
            "gcirc_cell_survival": {f: v["gcirc_cell_survival"] for f, v in jsum.items()},
            "h_star_distribution": {f: v["h_star"] for f, v in jsum.items()},
            "r1_fcr": {f: v["r1_fcr"] for f, v in jsum.items()},
            "jtable_sec": jt}
        summary["go_no_go"] = "GO" if passed else "NO_GO"
        with open(out_dir / "results.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        log(f"done in {time.time() - T0:.0f}s: gate {summary['go_no_go']} (violations {viol}; by eps "
            f"{ {k: v['reachable'] for k, v in gate.items()} })")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        if not args.smoke:
            mark_done("success", f"{args.mode}: G0 {summary['go_no_go']}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        if not args.smoke:
            mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
