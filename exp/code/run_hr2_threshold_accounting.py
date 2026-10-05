"""HR2 / HR2b single-factor accounting (methodology 4.3 / 6.2; task hr2_threshold_accounting).

Same grid G_1, same J tables, same certifier (exact minimax enumeration on the alive grid set; at R2 the locked
local-inflation certificate). 2x2 factorial on a grid-LR confidence set:
    threshold in {UI  log(1/delta)  (anytime-valid, Ville),  Wald/Wilks  0.5 chi2_{d_eff, 1-delta} (fixed-n)}
  x design    in {Wald-opt (B3's transductive design on the linearised GLM Fisher at the grid MLE),
                  G-opt   (B10's reward-free coverage design over the whole interaction library)}
d_eff = 2.98 (locked, r2_prereg_lock) is the primary chi2 cell; {6, 12, 24} is the pre-registered sensitivity band
(chi2 cells only). Extra arms: JPC (UI + DDA, ledger re-used over the stream) and JPC-fresh (UI + DDA, every
problem restarts from the n0 = 20 initial rounds on a fresh platform copy) -> reuse factor; B3 / B3g for the
savings waterfall (full mode: re-used from hr1_r0_a when its results exist for the same CRN instances).

Proposition B (prediction, not identity):  N_stream ~= beta_eff x Cover(U_k Alt_eps(q_k)) x (eps/(eps-2 eta_dec))^2
  Cover = min sum_x n_x  s.t.  sum_x n_x KL_x(theta*, theta) >= 1  for every theta in U_{k<=K} Alt_eps(q_k),
          x = (observable state, legal action), exact per-step KL (dsswm.acquire.nl_kl_dda), reachability relaxed;
  Alt_eps(q) = {theta in G_1 : regret_theta(pi*_q) > eps}, pi*_q = true optimum (harness only, oracle quantity);
  beta_eff = the cell's cutoff (log(1/delta) for UI, chi2_{d}/2 for Wilks);  eta_dec = 0 at R0.
  N_obs = n0 + cumulative charged probe steps; only N_obs >= 50 counted (HR2b).
Charging: certified -> steps; NEED_DATA / OUT_OF_SCOPE / MODEL_CONFLICT / refusal -> censored, charged T_max = 3000.

Usage: run_hr2_threshold_accounting.py --mode {pilot,full} [--workers 4] [--smoke] [--resummarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.stats import chi2  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_hr1_main as H  # noqa: E402  (same Arm / tables / JPC / B3 code path as hr1_r0_*)
from dsswm.acquire.nl_kl_dda import kl_features  # noqa: E402
from dsswm.baselines.glm_linearised import certify_linear, choose_design_action, value_and_grad  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, trichotomy  # noqa: E402
from dsswm.certify.minimax_infl import certify_minimax_infl, inflation_floor  # noqa: E402
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS  # noqa: E402

TASK = "hr2_threshold_accounting"
WS, RES_ROOT, LOCK = H.WS, H.RES_ROOT, H.LOCK
EPS, DELTA, TOP_M, TMAX = H.EPS, H.DELTA, H.TOP_M, H.TMAX_STEP
N0 = H.N0
D_SENS = (6, 12, 24)
EXPLORE = 0.05
GLIB_CACHE = 6


def cell_name(thr, design, d=None, d_primary=None):
    if thr == "ui":
        return f"LR-UI-{design}"
    return f"LR-chi2-{design}" if d == d_primary else f"LR-chi2d{d:g}-{design}"


def beta_eff(thr, d):
    return math.log(1.0 / DELTA) if thr == "ui" else 0.5 * float(chi2.ppf(1.0 - DELTA, d))


# ---------------------------------------------------------------------------------------- learner cells
def _glib(arm, des, kh):
    """G-opt library: Fisher of every (observable state, legal action) at the grid MLE kh -> (S*A, d*d)."""
    c = arm.__dict__.setdefault("_glib", {})
    if kh not in c:
        if len(c) >= GLIB_CACHE:
            c.pop(next(iter(c)))
        rep = arm.__dict__.get("_glib_rep")
        if rep is None:
            # (state, action) pairs with identical KL incidence rows have identical observation laws -> identical
            # Fisher; the max over the library is unchanged by keeping one representative per distinct row.
            S = arm.inc.S.reshape(arm.prop.codec.size, arm.inc.nA, arm.inc.nf)[:, arm.legal].reshape(-1, arm.inc.nf)
            _, rep = np.unique(S, axis=0, return_index=True)
            arm._glib_rep = rep = np.sort(rep)
        F = np.stack([des.F_all(kh, code) for code in range(arm.prop.codec.size)])
        c[kh] = F.reshape(-1, des.d * des.d)[rep]
    return c[kh]


def run_lr_cell(arm, thr, design, d_eff, name, infl, want_samples=False):
    """Grid-LR set with threshold thr in {'ui','chi2'} and design in {'wald','gopt'}; certifier exact minimax."""
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA, threshold="ui" if thr == "ui" else "chi2_deff",
                  d_eff=None if thr == "ui" else d_eff)
    led = EvidenceLedger([lr])
    all_obs = list(inst.init_obs)
    for o in inst.init_obs:
        led.record(o, "initial")
    des = H._B3Design(arm, inst)
    beta_w = des.beta                                    # Wald radius used ONLY to define the B3 design directions
    rng = np.random.default_rng([arm.seed, arm.noise_seed, {"wald": 2100, "gopt": 2200}[design]])
    rows, samples = [], []
    for k, q in enumerate(arm.problems):
        t0 = time.perf_counter()
        Reg, eta = arm.Reg[k], arm.eta[k]
        plans = [arm.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] if design == "wald" else None
        U = type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": arm.cq[k]})()
        grad_cache, steps, kh_prev, Vm, status, refused, traj = {}, 0, None, None, None, False, []
        n_explore = 0
        while True:
            if infl and H.floor_any(eta):
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
            if steps >= TMAX:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            if kh != kh_prev:
                Vm = des.build_V(kh)
                kh_prev = kh
            code = arm.prop.codec.encode(*h.observable_state())
            Fa = des.F_all(kh, code)
            if design == "wald":
                if kh not in grad_cache:
                    grad_cache[kh] = value_and_grad(arm.prop, plans, arm.V[kh], U, 2, 2)
                _, G = grad_cache[kh]
                lc = certify_linear(arm.J[k][kh], G, np.linalg.inv(Vm), beta_w, EPS)
                Dblk = lc["D"][lc["ub"] > EPS]
                if len(Dblk) == 0:
                    n_explore += 1
                a = choose_design_action(Fa, Vm, Dblk, arm.legal, rng, explore=EXPLORE)
            else:
                if rng.random() < EXPLORE:
                    a = int(rng.choice(arm.legal))
                    n_explore += 1
                else:
                    Fl = _glib(arm, des, kh)
                    Vi = np.linalg.inv(Vm[None] + Fa[arm.legal])                    # (nA, d, d)
                    vals = (Fl @ Vi.reshape(len(arm.legal), -1).T).max(0)
                    a = int(arm.legal[int(np.argmin(vals))])
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            all_obs.append(obs)
            des.hist[(code, a)] = des.hist.get((code, a), 0) + 1
            Vm = Vm + Fa[a]
            steps += 1
            if want_samples and len(traj) < 30 and (steps <= 10 or steps % 10 == 0):
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "star_alive": bool(mask[arm.star])})
        assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
        mask = lr.mask().numpy()
        gc = certify_minimax(Reg, mask, EPS, TOP_M)
        pi = cert["pi"] if status == "CERTIFIED" else None
        extra = {"threshold": thr, "design": design, "d_eff": d_eff, "cutoff": lr.cutoff(),
                 "set_size": int(mask.sum()), "r_bar_grid": float(gc["r_bar"]), "theta_cell_alive": bool(mask[arm.star]),
                 "refused_floor": refused, "n_explore_or_empty_D": n_explore, "wall_clock_s": time.perf_counter() - t0,
                 "lr_n_rounds": int(lr.n_rounds), "env_n_steps": int(env.n_steps)}
        rows.append(arm.row(k, name, status, pi, steps, extra))
        if want_samples and k < 2:
            samples.append({**rows[-1], "J_true": arm.Jt[k].round(4).tolist(), "trajectory_head": traj})
    return rows, samples


def _sub_arm(arm, k):
    s = copy.copy(arm)
    for f in ("problems", "J", "Reg", "eta", "Jt", "cq", "gap12"):
        setattr(s, f, [getattr(arm, f)[k]])
    return s


def run_jpc_fresh(arm, infl):
    """Reuse factor arm: every problem restarts from the n0 initial rounds on a fresh platform copy (no ledger reuse)."""
    rows = []
    for k in range(len(arm.problems)):
        r, _ = H.run_jpc(_sub_arm(arm, k), infl)
        r[0].update({"k": k, "method": "JPC-fresh_infl" if infl else "JPC-fresh"})
        rows.append(r[0])
    return rows


# ---------------------------------------------------------------------------------------- Proposition B cover
def cover_unit(arm):
    """Oracle Cover(U_{k<=K} Alt_eps(q_k)) for K = 1..n_prob (harness-only: uses theta*'s grid cell)."""
    t0 = time.perf_counter()
    star = arm.star
    S = arm.inc.S.reshape(arm.prop.codec.size, arm.inc.nA, arm.inc.nf)[:, arm.legal].reshape(-1, arm.inc.nf)
    S_u = np.unique(S, axis=0)
    S_u = S_u[S_u.sum(1) > 0].astype(float)
    U = np.zeros(arm.ncl.B, bool)
    out = []
    for k in range(len(arm.problems)):
        pistar = int(np.argmax(arm.Jt[k]))
        alt = arm.Reg[k][:, pistar] > EPS
        alt[star] = False
        U |= alt
        idx = np.flatnonzero(U)
        rec = {"kind": "cover", "instance": arm.seed, "stream": arm.stream, "level": arm.level, "K": k + 1,
               "problem": arm.problems[k].pid, "n_alt_k": int(alt.sum()), "n_alt_union": int(len(idx)),
               "star_reg_at_pistar": float(arm.Reg[k][star, pistar]), "n_distinct_x": int(len(S_u))}
        if len(idx) == 0:
            rec.update({"cover": 0.0, "lp_status": "empty", "n_unidentifiable": 0})
            out.append(rec)
            continue
        F = kl_features(arm.py[star:star + 1], arm.pe[star:star + 1], arm.py[idx], arm.pe[idx])   # (nf, M)
        KX = S_u @ F                                                                              # (nx, M)
        mx = KX.max(0)
        unid = mx <= 1e-12
        KX = KX[:, ~unid]
        # drop exact-duplicate constraints (same KL vector) to shrink the LP
        KXr = np.unique(np.round(KX.T, 12), axis=0)
        tl = time.perf_counter()
        # cutting planes: start from the hardest-to-separate models, add violated constraints until feasible (exact)
        act = list(np.argsort(KXr.max(1))[:200])
        res, n_iter = None, 0
        while True:
            n_iter += 1
            A = KXr[act]
            res = linprog(np.ones(KXr.shape[1]), A_ub=-A, b_ub=-np.ones(len(act)), bounds=(0, None), method="highs")
            if res.status != 0:
                break
            viol = KXr @ res.x < 1.0 - 1e-7
            if not viol.any():
                break
            vi = np.flatnonzero(viol)
            new = vi[np.argsort((KXr[vi] @ res.x))[:500]]
            act = sorted(set(act) | set(int(i) for i in new))
        rec.update({"cover": float(res.fun) if res.status == 0 else None, "lp_status": int(res.status),
                    "n_unidentifiable": int(unid.sum()), "n_constraints": int(KXr.shape[0]),
                    "n_active_constraints": len(act), "cutting_plane_iters": n_iter,
                    "lp_sec": time.perf_counter() - tl,
                    "cover_support": int((res.x > 1e-9).sum()) if res.status == 0 else None})
        out.append(rec)
    for r in out:
        r["unit_cover_sec"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------------------- unit dispatcher
def run_unit(seed, stream, level, methods, n_prob, meta, d_primary, want_samples):
    torch.set_num_threads(1)
    H.DIAGNOSTIC.setdefault("R2", set())
    t0 = time.time()
    infl = level != "R0"
    arm = H.Arm(seed, stream, level, 1, n_prob, meta, need_lr=True, infl=infl)
    build = time.time() - t0
    rows, samples, errors, msec = [], [], [], {}
    for m in methods:
        tm = time.time()
        try:
            if m.startswith("LR-"):
                _, thr_s, design = m.split("-")
                if thr_s == "UI":
                    thr, d = "ui", None
                elif thr_s == "chi2":
                    thr, d = "chi2", d_primary
                else:
                    thr, d = "chi2", float(thr_s[len("chi2d"):])
                r, s = run_lr_cell(arm, thr, design, d, m + ("_infl" if infl else ""), infl, want_samples)
                samples += s
            elif m == "JPC":
                r, s = H.run_jpc(arm, infl, want_samples)
                samples += s
            elif m == "JPC-fresh":
                r = run_jpc_fresh(arm, infl)
            elif m in ("B3", "B3g"):
                r = H.run_b3(arm, "B3g_infl" if (m == "B3g" and infl) else m)
            elif m == "COVER":
                r = cover_unit(arm)
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "stream": stream, "level": level, "method": m, "error": repr(e),
                           "tb": traceback.format_exc()})
        msec[m] = time.time() - tm
    return rows, samples, errors, {"instance": seed, "stream": stream, "level": level, "methods": methods,
                                   "build_sec": build, "method_sec": msec, "sec": time.time() - t0}


def groups_for(level, d_primary, run_b3=True):
    g = [["LR-UI-wald", "LR-chi2-wald"], ["LR-UI-gopt", "LR-chi2-gopt"]]
    if level == "R0":
        g += [[f"LR-chi2d{d:g}-wald" for d in D_SENS], [f"LR-chi2d{d:g}-gopt" for d in D_SENS],
              ["JPC", "JPC-fresh"], ["COVER"]]
        if run_b3:
            g += [["B3", "B3g"]]
    else:
        g += [["JPC", "JPC-fresh"]]
    return g


# ---------------------------------------------------------------------------------------- analysis
def _boot(x, B=10000, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(len(x), size=(B, len(x)))].mean(1)
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def inst_logs(rows, m):
    """instance -> mean over streams of log(stream charged total + 1)."""
    out = {}
    R = [r for r in rows if r["method"] == m]
    for i in sorted({r["instance"] for r in R}):
        sts = sorted({r["stream"] for r in R if r["instance"] == i})
        out[i] = float(np.mean([math.log(sum(r["new_env_steps"] for r in R if r["instance"] == i and r["stream"] == s) + 1)
                                for s in sts]))
    return out


def log_contrast(rows, plus, minus, label):
    """sum(+log) - sum(-log) per instance (paired, CRN), geometric-mean ratio + instance bootstrap CI."""
    L = {m: inst_logs(rows, m) for m in plus + minus}
    insts = sorted(set.intersection(*[set(v) for v in L.values()])) if L else []
    if not insts or any(not L[m] for m in L):
        return {"label": label, "available": False}
    w = 1.0 / max(len(plus), 1)
    x = np.array([w * (sum(L[m][i] for m in plus) - sum(L[m][i] for m in minus)) for i in insts])
    lo, hi = _boot(x)
    return {"label": label, "available": True, "plus": plus, "minus": minus, "n_instances": len(insts),
            "log_mean": float(x.mean()), "ratio_geo_mean": float(math.exp(x.mean())), "ci95": [math.exp(lo), math.exp(hi)],
            "log_ci95": [lo, hi], "per_instance": [float(math.exp(v)) for v in x]}


def summarize_methods(rows):
    out = {}
    for m in sorted({r["method"] for r in rows}):
        R = [r for r in rows if r["method"] == m]
        cert = [r for r in R if r["status"] == "CERTIFIED"]
        nfc = sum(r["false_cert"] for r in R)
        lo, hi = clopper_pearson(nfc, len(R), 0.05)
        units = sorted({(r["instance"], r["stream"]) for r in R})
        tot = [sum(r["new_env_steps"] for r in R if (r["instance"], r["stream"]) == u) for u in units]
        alive = [r["theta_cell_alive"] for r in R if r.get("theta_cell_alive") is not None]
        out[m] = {"n": len(R), "n_streams": len(units), "completion": len(cert) / len(R),
                  "status_counts": {s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R})},
                  "refused_floor": int(sum(bool(r.get("refused_floor")) for r in R)),
                  "stream_steps_charged_mean": float(np.mean(tot)), "stream_steps_charged_median": float(np.median(tot)),
                  "steps_consumed_median": float(np.median([r["steps_consumed"] for r in R])),
                  "false_certs": int(nfc), "fcr_cp_upper": hi, "theta_cell_survival": float(np.mean(alive)) if alive else None,
                  "zero_cost_rate": float(np.mean([r["zero_cost"] for r in R])),
                  "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in R]))}
    return out


def prop_b(rows, covers, d_primary):
    """HR2b: pred/obs for each cell; streams (and cumulative prefixes) with N_obs >= 50, uncensored prefix only."""
    cov = {(c["instance"], c["stream"], c["K"]): c["cover"] for c in covers if c.get("cover") is not None}
    cells = {}
    for m in sorted({r["method"] for r in rows}):
        if m.startswith("LR-UI") or m == "JPC":                # JPC-fresh: no reuse -> union cover not applicable
            thr, d = "ui", None
        elif m.startswith("LR-chi2"):
            seg = m.split("-")[1]
            thr, d = "chi2", (d_primary if seg == "chi2" else float(seg[len("chi2d"):]))
        else:
            continue
        b = beta_eff(thr, d)
        R = sorted([r for r in rows if r["method"] == m], key=lambda r: (r["instance"], r["stream"], r["k"]))
        pts_stream, pts_prefix = [], []
        for u in sorted({(r["instance"], r["stream"]) for r in R}):
            Ru = [r for r in R if (r["instance"], r["stream"]) == u]
            cum, ok = N0, True
            for r in Ru:
                ok = ok and not r["censored"]
                cum += r["new_env_steps"]
                key = (u[0], u[1], r["k"] + 1)
                if key in cov and ok and cum >= 50:
                    pts_prefix.append({"instance": u[0], "stream": u[1], "K": r["k"] + 1, "N_obs": cum,
                                       "N_pred": b * cov[key], "ratio": b * cov[key] / cum})
            Kfin = (u[0], u[1], len(Ru))
            if Kfin in cov and ok and cum >= 50:
                pts_stream.append({"instance": u[0], "stream": u[1], "N_obs": cum, "N_pred": b * cov[Kfin],
                                   "ratio": b * cov[Kfin] / cum})
        def share(p):
            if not p:
                return None
            r = np.array([x["ratio"] for x in p])
            return {"n": len(p), "share_in_band": float(np.mean((r >= 0.67) & (r <= 1.5))), "median_ratio": float(np.median(r)),
                    "q10": float(np.quantile(r, .1)), "q90": float(np.quantile(r, .9))}
        cells[m] = {"beta_eff": b, "stream_level": share(pts_stream), "prefix_level": share(pts_prefix),
                    "points_stream": pts_stream}
    return cells


def analyze(rows, covers, d_primary, level):
    sfx = "" if level == "R0" else "_infl"
    n = lambda base: base + sfx  # noqa: E731
    C = {}
    for des in ("wald", "gopt"):
        C[f"threshold_only_ratio_{des}"] = log_contrast(rows, [n(f"LR-UI-{des}")], [n(f"LR-chi2-{des}")],
                                                        f"UI/chi2(d={d_primary}) at {des} design")
        for d in D_SENS:
            C[f"threshold_only_ratio_{des}_d{d}"] = log_contrast(rows, [n(f"LR-UI-{des}")], [n(f"LR-chi2d{d}-{des}")],
                                                                 f"UI/chi2(d={d}) at {des} design")
    for thr, nm in (("ui", "UI"), ("chi2", "chi2")):
        C[f"design_only_ratio_{thr}"] = log_contrast(rows, [n(f"LR-{nm}-gopt")], [n(f"LR-{nm}-wald")],
                                                     f"G-opt/Wald-opt at {nm} threshold")
    C["threshold_main_effect"] = log_contrast(rows, [n("LR-UI-wald"), n("LR-UI-gopt")],
                                              [n("LR-chi2-wald"), n("LR-chi2-gopt")], "UI vs chi2 averaged over designs")
    C["design_main_effect"] = log_contrast(rows, [n("LR-UI-gopt"), n("LR-chi2-gopt")],
                                           [n("LR-UI-wald"), n("LR-chi2-wald")], "G-opt vs Wald averaged over thresholds")
    for d in D_SENS:
        C[f"threshold_main_effect_d{d}"] = log_contrast(rows, [n("LR-UI-wald"), n("LR-UI-gopt")],
                                                        [n(f"LR-chi2d{d}-wald"), n(f"LR-chi2d{d}-gopt")],
                                                        f"UI vs chi2(d={d}) averaged over designs")
    # interaction: (UI-g - UI-w) - (chi2-g - chi2-w), per instance
    A = {m: inst_logs(rows, n(m)) for m in ("LR-UI-wald", "LR-UI-gopt", "LR-chi2-wald", "LR-chi2-gopt")}
    insts = sorted(set.intersection(*[set(v) for v in A.values()])) if all(A.values()) else []
    if insts:
        x = np.array([(A["LR-UI-gopt"][i] - A["LR-UI-wald"][i]) - (A["LR-chi2-gopt"][i] - A["LR-chi2-wald"][i])
                      for i in insts])
        C["interaction_log"] = {"mean": float(x.mean()), "ci95": _boot(x)}
        te = np.array([0.5 * (A["LR-UI-wald"][i] + A["LR-UI-gopt"][i] - A["LR-chi2-wald"][i] - A["LR-chi2-gopt"][i])
                       for i in insts])
        de = np.array([0.5 * (A["LR-UI-gopt"][i] + A["LR-chi2-gopt"][i] - A["LR-UI-wald"][i] - A["LR-chi2-wald"][i])
                       for i in insts])
        diff = np.abs(te.mean()) - np.abs(de.mean())
        rng = np.random.default_rng(42)
        bsd = []
        for _ in range(10000):
            ii = rng.integers(len(insts), size=len(insts))
            bsd.append(abs(te[ii].mean()) - abs(de[ii].mean()))
        C["abs_threshold_minus_abs_design"] = {"mean": float(diff), "ci95": [float(np.quantile(bsd, .025)),
                                                                             float(np.quantile(bsd, .975))],
                                               "threshold_gt_design": bool(diff > 0)}
    if level == "R0":
        C["reuse_factor_ratio"] = log_contrast(rows, ["JPC"], ["JPC-fresh"], "JPC stream / JPC fresh-start")
        C["dda_vs_wald_at_UI"] = log_contrast(rows, ["JPC"], ["LR-UI-wald"], "DDA/Wald-opt at UI grid LR")
        wf = [("support: B3g/B3", ["B3g"], ["B3"]), ("shape: chi2-grid-LR(wald)/B3g", ["LR-chi2-wald"], ["B3g"]),
              ("threshold: UI-grid-LR(wald)/chi2-grid-LR(wald)", ["LR-UI-wald"], ["LR-chi2-wald"]),
              ("design: JPC(DDA)/UI-grid-LR(wald)", ["JPC"], ["LR-UI-wald"]),
              ("reuse: JPC(stream)/JPC-fresh", ["JPC"], ["JPC-fresh"]),
              ("total: JPC/B3", ["JPC"], ["B3"]), ("total: JPC/B3g", ["JPC"], ["B3g"])]
        C["waterfall"] = {lab: log_contrast(rows, p, mm, lab) for lab, p, mm in wf}
        pb = prop_b(rows, covers, d_primary)
        C["propB"] = {m: {k: v for k, v in x.items() if k != "points_stream"} for m, x in pb.items()}
        C["propB_points"] = {m: x["points_stream"] for m, x in pb.items()}
        th = C["threshold_only_ratio_wald"]
        if th.get("available"):
            C["propB_threshold_ratio"] = {"pred": beta_eff("ui", None) / beta_eff("chi2", d_primary),
                                          "obs": th["ratio_geo_mean"],
                                          "pred_over_obs": beta_eff("ui", None) / beta_eff("chi2", d_primary) / th["ratio_geo_mean"],
                                          "by_d": {d: {"pred": beta_eff("ui", None) / beta_eff("chi2", d),
                                                       "obs": (C[f"threshold_only_ratio_wald_d{d}"] or {}).get("ratio_geo_mean")}
                                                   for d in D_SENS}}
    return C


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    lock = json.loads(LOCK.read_text())
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    d_primary = float(lock["d_eff"]["locked_value"])
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    pid_path, prog_path = RES_ROOT / f"{TASK}.pid", RES_ROOT / f"{TASK}_PROGRESS.json"
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
        if real:
            prog_path.write_text(json.dumps({"task_id": TASK, "epoch": step, "total_epochs": total, "step": step,
                                             "total_steps": total, "loss": None, "metric": {"phase": phase, **(metric or {})},
                                             "updated_at": datetime.now().isoformat()}))

    def mark_done(status, txt):
        if pid_path.exists():
            pid_path.unlink()
        fp = {}
        if prog_path.exists():
            try:
                fp = json.loads(prog_path.read_text())
            except ValueError:
                pass
        (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": txt,
                                                           "final_progress": fp, "timestamp": datetime.now().isoformat()}))

    if pilot:
        seeds_r0, seeds_r2, n_prob, streams = list(range(670, 676)), list(range(670, 672)), 5, [0]
        assert max(seeds_r0) < 10000
    else:
        rngs = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds_r0 = [s for a, b in rngs for s in range(a, b + 1)]
        seeds_r2 = seeds_r0[:24]
        n_prob, streams = int(lock["eval_manifest"]["K_problems"]), list(range(len(STREAMS)))
    if args.smoke:
        seeds_r0, seeds_r2, n_prob = seeds_r0[:1], seeds_r2[:1], 2
    # B3 / B3g: full mode re-uses hr1_r0_a rows on the same CRN instances when available
    b3_rows_ext = []
    if not pilot:
        p = RES_ROOT / "full" / "hr1_r0_a" / "results.jsonl"
        if p.exists():
            b3_rows_ext = [json.loads(x) for x in open(p)]
            b3_rows_ext = [r for r in b3_rows_ext if r["method"] in ("B3", "B3g") and r["instance"] in set(seeds_r0)
                           and r["k"] < n_prob]
    have_ext = len(b3_rows_ext) == 2 * len(seeds_r0) * len(streams) * n_prob
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    G = {"R0": groups_for("R0", d_primary, run_b3=not have_ext), "R2": groups_for("R2", d_primary)}
    log(f"start {TASK} mode={args.mode} device={dev} R0 seeds {seeds_r0[0]}-{seeds_r0[-1]} R2 seeds {seeds_r2[0]}-"
        f"{seeds_r2[-1]} streams {streams} x {n_prob} problems; d_eff primary={d_primary} sens={D_SENS}; "
        f"B3/B3g {'from hr1_r0_a' if have_ext else 'run here'}; groups={G}")
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA, "f": 1,
               "seeds_R0": [seeds_r0[0], seeds_r0[-1]], "seeds_R2": [seeds_r2[0], seeds_r2[-1]],
               "n_problems_per_stream": n_prob, "streams": streams, "eval_seeds_touched": not pilot,
               "generator_hash": generator_hash(), "lock_status": lock.get("status"), "d_eff_primary": d_primary,
               "d_eff_sensitivity": list(D_SENS), "cutoffs": {"UI": beta_eff("ui", None),
                                                              **{f"chi2_d{d:g}": beta_eff("chi2", d) for d in (d_primary,) + D_SENS}},
               "b3_source": "hr1_r0_a" if have_ext else "run_here", "concurrent_run": True, "tmax_stepwise": TMAX,
               "groups": G, "notes": [
                   "chi2_deff set is fixed-n Wilks (NOT anytime-valid); it isolates the threshold factor only",
                   "Wald-opt design = B3 transductive step on linearised GLM Fisher at the grid MLE (directions D from "
                   "certify_linear with wald_beta(12)); G-opt = B10 coverage design over all (state, action)",
                   "Cover LP relaxes state reachability (oracle, harness-only); eta_dec = 0 at R0",
                   "MODEL_CONFLICT in LR cells is censored (charged T_max), no B2 fallback; JPC keeps its locked B2 fallback"]}
    res_path, done_path, cov_path = out_dir / "results.jsonl", out_dir / "units_done.jsonl", out_dir / "cover.jsonl"
    try:
        if not args.resummarize:
            progress(1, 4, "gpu: J tables (+ eta_loc f=1 for R2)")
            meta = H.precompute(seeds_r0, n_prob, streams, (1,), (), dev, log)
            meta2 = H.precompute(seeds_r2, n_prob, streams, (1,), (1,), dev, log)
            for kk in ("cq", "jtrue", "gap12"):
                if kk == "gap12":
                    for lv, d in meta2["gap12"].items():
                        meta["gap12"].setdefault(lv, {}).update(d)
                else:
                    meta[kk].update(meta2[kk])
            summary["table_sec"] = {"R0": meta["sec"], "R2": meta2["sec"]}
            summary["eta_loc_min_over_eps_R2"] = ({k: {"median": float(np.median(v)), "min": float(np.min(v))}
                                                   for k, v in meta2["eta_min_over_eps"].items()})
            if torch.cuda.is_available():
                gp = {"gpu_name": torch.cuda.get_device_name(0),
                      "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / 2**20),
                      "max_batch_size": None, "vram_used_mb": int(torch.cuda.max_memory_allocated() / 2**20),
                      "utilization_pct": round(100 * torch.cuda.max_memory_allocated() /
                                               torch.cuda.get_device_properties(0).total_memory, 2),
                      "note": "CPU-dominant task; GPU only for missing J tables / eta_loc (per-problem exact "
                              "propagation over 13824 grid rows), no batch-size probe applicable"}
                (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(gp))
            done_units = set()
            if done_path.exists():
                for x in open(done_path):
                    u = json.loads(x)
                    done_units.add((u["instance"], u["stream"], u["level"], tuple(u["methods"])))
            jobs = [(s, st, lv, tuple(g)) for lv, ss in (("R0", seeds_r0), ("R2", seeds_r2)) for s in ss
                    for st in streams for g in G[lv] if (s, st, lv, tuple(g)) not in done_units]
            # longest groups first (G-opt / sensitivity) for better load balance
            jobs.sort(key=lambda j: (0 if "gopt" in j[3][0] else 1 if "wald" in j[3][0] else 2))
            log(f"units: {len(jobs)} to run, {len(done_units)} already done")
            progress(2, 4, "cpu: streams", {"units_total": len(jobs), "units_done": 0})
            from joblib import Parallel, delayed
            gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
                delayed(run_unit)(s, st, lv, list(g), n_prob, meta, d_primary, s == seeds_r0[0] and st == 0 and lv == "R0")
                for (s, st, lv, g) in jobs)
            n_err, all_samples = 0, []
            for i, (rr, ss, ee, us) in enumerate(gen):
                with open(res_path, "a") as fh, open(cov_path, "a") as fc:
                    for r in rr:
                        (fc if r.get("kind") == "cover" else fh).write(json.dumps(r, default=float) + "\n")
                for e in ee:
                    log(f"ERROR {e['instance']} s{e['stream']} {e['level']} {e['method']}: {e['error']}\n{e['tb']}")
                n_err += len(ee)
                with open(out_dir / "errors.jsonl", "a") as fh:
                    for e in ee:
                        fh.write(json.dumps(e) + "\n")
                if not ee:
                    with open(done_path, "a") as fh:
                        fh.write(json.dumps({k: us[k] for k in ("instance", "stream", "level", "methods")}) + "\n")
                with open(out_dir / "unit_sec.jsonl", "a") as fh:
                    fh.write(json.dumps(us) + "\n")
                all_samples += ss
                log(f"unit {i + 1}/{len(jobs)} {us['level']} {us['instance']} s{us['stream']} {us['methods']} "
                    f"{us['sec']:.1f}s")
                progress(2, 4, "cpu: streams", {"units_total": len(jobs), "units_done": i + 1, "errors": n_err})
            if all_samples:
                (out_dir / "samples" / "samples.json").write_text(json.dumps(all_samples, indent=1, default=float))
            summary["crashes_this_run"] = n_err
        else:
            old = json.loads((out_dir / "summary.json").read_text())
            summary = {**old, **{k: v for k, v in summary.items() if k not in old}}
        progress(3, 4, "summary")
        dedup = {}
        for x in open(res_path):
            r = json.loads(x)
            dedup[(r["level"], r["instance"], r["stream"], r["method"], r["k"])] = r
        rows = list(dedup.values()) + [r for r in b3_rows_ext]
        covers = list({(c["level"], c["instance"], c["stream"], c["K"]): c
                       for c in (json.loads(x) for x in open(cov_path))}.values()) if cov_path.exists() else []
        errs = [json.loads(x) for x in open(out_dir / "errors.jsonl")] if (out_dir / "errors.jsonl").exists() else []
        units_all = [json.loads(x) for x in open(out_dir / "unit_sec.jsonl")] if (out_dir / "unit_sec.jsonl").exists() else []
        by_level = {}
        for lv in ("R0", "R2"):
            R = [r for r in rows if r["level"] == lv]
            if R:
                by_level[lv] = {"by_method": summarize_methods(R),
                                "contrasts": analyze(R, [c for c in covers if c["level"] == lv], d_primary, lv)}
        summary["by_level"] = by_level
        summary["cover_summary"] = {
            "n": len(covers), "lp_ok": int(sum(c.get("lp_status") in (0, "empty") for c in covers)),
            "cover_by_K_median": {int(K): float(np.median([c["cover"] for c in covers if c["K"] == K and c.get("cover") is not None]))
                                  for K in sorted({c["K"] for c in covers})},
            "n_alt_union_by_K_median": {int(K): float(np.median([c["n_alt_union"] for c in covers if c["K"] == K]))
                                        for K in sorted({c["K"] for c in covers})},
            "unidentifiable_any": int(sum(c.get("n_unidentifiable", 0) > 0 for c in covers)),
            "monotone_in_K": all(
                all(a["cover"] <= b["cover"] + 1e-6 for a, b in zip(seq, seq[1:]))
                for seq in ([sorted([c for c in covers if (c["instance"], c["stream"]) == u and c.get("cover") is not None],
                                    key=lambda c: c["K"]) for u in {(c["instance"], c["stream"]) for c in covers}]))}
        expected = sum(len(ss) * len(streams) * n_prob * sum(len(g) for g in G[lv] if g != ["COVER"])
                       for lv, ss in (("R0", seeds_r0), ("R2", seeds_r2)))
        summary["n_rows"] = len(dedup)
        summary["rows_expected_this_task"] = expected
        summary["complete"] = len(dedup) == expected
        summary["crashes_total_logged"] = len(errs)
        acc_ok = all(r.get("lr_n_rounds") is None or r["lr_n_rounds"] == r["env_n_steps"] for r in rows)
        summary["evidence_boundary_assertions_ok"] = acc_ok and not any("step accounting" in e["error"] for e in errs)
        # timing projection for full mode (4 workers): sum of unit seconds scaled by problems ratio and instances
        if units_all:
            sec_by_lv = {lv: sum(u["sec"] for u in units_all if u["level"] == lv) for lv in ("R0", "R2")}
            n_by_lv = {"R0": len(seeds_r0) * len(streams), "R2": len(seeds_r2) * len(streams)}
            full_n = {"R0": 48 * 3, "R2": 24 * 3}
            scale = 15 / n_prob
            full_cpu = sum(sec_by_lv[lv] / max(n_by_lv[lv], 1) * full_n[lv] * scale for lv in sec_by_lv)
            b3_sec = sum(u["method_sec"].get("B3", 0) + u["method_sec"].get("B3g", 0) for u in units_all
                         if u["level"] == "R0") / max(n_by_lv["R0"], 1) * full_n["R0"] * scale
            per_m = {}
            for u in units_all:
                for m, s in u["method_sec"].items():
                    per_m.setdefault(f"{u['level']}:{m}", []).append(s)
            summary["timing_projection"] = {
                "full_cpu_sec": full_cpu, "full_wall_min_at_4_workers": full_cpu / 4 / 60,
                "full_wall_min_without_B3_B3g_(reuse_hr1_r0_a)": (full_cpu - b3_sec) / 4 / 60,
                "method_sec_per_stream_mean": {k: float(np.mean(v)) for k, v in per_m.items()},
                "max_unit_sec": max(u["sec"] for u in units_all),
                "basis": "linear in problems/stream (15 vs pilot 5); 48 R0 + 24 R2 instances x 3 streams; concurrent run",
                "note": "steps per problem fall with k under ledger reuse -> linear scaling is pessimistic for stream arms, "
                        "optimistic for JPC-fresh"}
        r0 = by_level.get("R0", {}).get("contrasts", {})
        thr_w = r0.get("threshold_only_ratio_wald", {})
        pc = {"zero_crashes": len(errs) == 0, "evidence_boundary_assertions_hold": summary["evidence_boundary_assertions_ok"],
              "2x2_end_to_end": summary["complete"],
              "threshold_only_ratio_computable": bool(thr_w.get("available")),
              "main_effects_computable": bool(r0.get("threshold_main_effect", {}).get("available")
                                              and r0.get("design_main_effect", {}).get("available")),
              "cover_lp_all_solved": summary["cover_summary"]["lp_ok"] == len(covers) and len(covers) > 0}
        summary["pass_criteria"] = pc
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        if thr_w.get("available"):
            summary["hr2_direction_preview_not_gated"] = {
                "threshold_only_ratio_wald": thr_w["ratio_geo_mean"], "ci95": thr_w["ci95"],
                "in_[0.2,0.5]": 0.2 <= thr_w["ratio_geo_mean"] <= 0.5, "gt_0.7": thr_w["ratio_geo_mean"] > 0.7,
                "threshold_gt_design": r0.get("abs_threshold_minus_abs_design", {}).get("threshold_gt_design")}
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for lv, v in by_level.items():
            for m, s in v["by_method"].items():
                log(f"{lv} {m:18s} n={s['n']:3d} compl {s['completion']:.2f} stream steps {s['stream_steps_charged_mean']:.0f} "
                    f"(median {s['stream_steps_charged_median']:.0f}) FCR {s['false_certs']}/{s['n']} theta-cell "
                    f"{s['theta_cell_survival']} wall/problem {s['wall_clock_per_problem_s']:.2f}s")
            for kname, c in v["contrasts"].items():
                if isinstance(c, dict) and c.get("available"):
                    log(f"{lv} {kname}: {c['ratio_geo_mean']:.3f} CI {np.round(c['ci95'], 3).tolist()}")
        log(f"pass: {summary['go_no_go']} {pc}; projection {summary.get('timing_projection', {}).get('full_wall_min_at_4_workers')}")
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
