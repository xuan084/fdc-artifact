"""Oracle ceilings for HR5 and the suspicious-improvement gate (task oracle_opt_cover).

HARNESS-ONLY: every quantity here needs theta* and is NOT available to any learner.

For each (instance, stream, level in {R0, R2}) on the grid G_1 (|Theta| = 13824), with the same J tables / eta_loc as
hr1_r0_* / hr1_r2_*:
  KL_x(theta*, theta) = exact per-step KL at x = (observable state, legal action)       (dsswm.acquire.nl_kl_dda)
  Cover(A)  = min sum_x n_x   s.t.  sum_x n_x KL_x(theta*, theta) >= 1  for every theta in A  (reachability relaxed,
              unit cost c_x = 1 env step per interaction; exact LP via HiGHS + cutting planes)
  OPT(A)    = log(1/delta) x Cover(A)  (env steps; same LR cutoff as the UI certificate)
  Alt_eps(q):  R0: {theta != theta*_cell : Reg_theta(pi*_q) > eps}
               R2: {theta in G_1 : Reg_theta(pi*_q) + 2 eta_loc(theta) > eps}   (locked inflated certificate:
                   any such theta left alive blocks R_grid + 2 max_alive eta_loc <= eps for pi*_q)
  Alt_id(q) (B8 full identification, tau = eps/2):
               R0: {theta : ||J_theta(q) - J_{theta*}(q)||_inf >= tau}
               R2: {theta : ||J_theta(q) - J_true(q)||_inf + 2 eta_loc(theta) >= tau}
  pi*_q = argmax_pi J_true(q)  (R0: snapped truth, R2: continuous R1 truth).
  OPT_K       = OPT(U_{k<=K} Alt_eps(q_k)),  K = 1..n_prob
  OPT_id_K    = OPT(U_{k<=K} Alt_id(q_k))
  rho*_k      = OPT(Alt_eps(q_k))  (single problem, no reuse, fresh start; the numerical rho*)
  rho*_id_k   = OPT(Alt_id(q_k))
  opt_ratio_K = OPT_K / OPT_id_K   (oracle analogue of the JPC/B8 stream ratio, HR5 Spearman >= 0.7)
  reuse_gain_K = sum_{k<=K} rho*_k / OPT_K  (oracle reuse ceiling)
  OPT(Alt_param) (R0 only, secondary): every observationally distinguishable theta != theta*_cell.
Models with zero KL on every x (observationally equivalent to theta*) cannot be excluded by any data: they are dropped
from the LP and counted (n_unidentifiable); if any lies in an Alt set the problem is 'not identifiable' (the learner's
trichotomy returns OUT_OF_SCOPE there), recorded via n_unidentifiable > 0.
R2 data-free floor: if 2 min eta_loc > eps no certificate exists (OPT = inf, flagged 'floor'); if no grid model is
eps-compatible (Reg + 2 eta <= eps for pi*) the oracle set would be empty (flagged 'no_survivor').
Suspicious gate: any observed method stream with n0 + cumulative charged steps < OPT_K (uncensored prefix) is flagged.
HR5: Spearman over instances of (JPC/B8 observed stream ratio, oracle opt_ratio at K_final), per level.

Usage: run_oracle_opt_cover.py --mode {pilot,full} [--workers 4] [--smoke] [--resummarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
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
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_hr1_main as H  # noqa: E402
from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, kl_features, single_prob_tables  # noqa: E402
from dsswm.streams.generator import generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

TASK = "oracle_opt_cover"
WS, RES_ROOT, LOCK = H.WS, H.RES_ROOT, H.LOCK
EPS, DELTA = H.EPS, H.DELTA
TAU = EPS / 2
LOGD = math.log(1.0 / DELTA)
N0 = H.N0
C_KNOWN, NMAX = H.C_KNOWN, H.NMAX


# ---------------------------------------------------------------------------------------- covering LP
def cover_lp(KX, idx, init=200, add=500, max_iter=200, prev=None, skip=False):
    """min 1^T n s.t. KX[:, idx]^T n >= 1, n >= 0. KX: (nx, B) rates. Returns dict (exact via cutting planes).
    prev: result of the same LP on a SUBSET of idx (the K-1 union); if its optimum is feasible for the superset it is
    also optimal there (the superset LP is at least as large) and is reused without re-solving.
    skip: infeasible by construction (floor / no survivor) -> no LP."""
    if skip:
        return {"cover": None, "lp_status": "skipped_infeasible", "n_unid": 0, "n_cons": 0}
    if prev is not None and prev.get("x") is not None and len(idx):
        K = KX[:, idx]
        ok = K.max(0) > 1e-12
        if (prev["x"] @ K[:, ok] >= 1.0 - 1e-7).all():
            return {**prev, "reused": True, "n_unid": int((~ok).sum()), "lp_sec": 0.0}
    if len(idx) == 0:
        return {"cover": 0.0, "lp_status": "empty", "n_unid": 0, "n_cons": 0}
    K = KX[:, idx]
    unid = K.max(0) <= 1e-12
    K = K[:, ~unid]
    if K.shape[1] == 0:
        return {"cover": 0.0, "lp_status": "all_unidentifiable", "n_unid": int(unid.sum()), "n_cons": 0}
    Kr = np.unique(np.round(K.T, 12), axis=0)                       # (m, nx) distinct constraints
    act = list(np.argsort(Kr.max(1))[:init])
    t0, it, res = time.perf_counter(), 0, None
    while it < max_iter:
        it += 1
        res = linprog(np.ones(Kr.shape[1]), A_ub=-Kr[act], b_ub=-np.ones(len(act)), bounds=(0, None), method="highs")
        if res.status != 0:
            break
        v = Kr @ res.x
        viol = np.flatnonzero(v < 1.0 - 1e-7)
        if len(viol) == 0:
            break
        new = viol[np.argsort(v[viol])][:add]
        act = sorted(set(act) | set(int(i) for i in new))
    ok = res is not None and res.status == 0
    x = res.x if ok else None
    out = {"cover": float(res.fun) if ok else None, "lp_status": int(res.status) if res is not None else -1,
           "n_unid": int(unid.sum()), "n_cons": int(Kr.shape[0]), "n_active": len(act), "iters": it,
           "lp_sec": time.perf_counter() - t0}
    if ok:
        out["min_slack"] = float((Kr @ x).min())                    # >= 1 - 1e-7 when exact
        out["support"] = int((x > 1e-9).sum())
        out["x"] = x
    return out


# ---------------------------------------------------------------------------------------- unit
def run_unit(seed, stream, level, n_prob, meta, want_samples):
    torch.set_num_threads(1)
    t0 = time.time()
    infl = level == "R2"
    arm = H.Arm(seed, stream, level, 1, n_prob, meta, need_lr=False, infl=infl)
    inst = make_offgrid_instance(seed, level, stream=stream)
    aspace = inst.env.aspace
    py, pe = class_prob_tables(arm.ncl.np_params, C_KNOWN, NMAX, aspace.nb)
    inc = IncidenceIndex(arm.prop.codec, aspace, NMAX)
    legal = np.arange(aspace.n)
    S = inc.S.reshape(arm.prop.codec.size, inc.nA, inc.nf)[:, legal].reshape(-1, inc.nf)
    S_u = np.unique(S, axis=0)
    S_u = S_u[S_u.sum(1) > 0].astype(float)
    # truth tables: R0 snapped truth (sanity: equals the star cell), R2 continuous R1 truth
    tpy, tpe = single_prob_tables(inst.env.true_params(), C_KNOWN, NMAX, aspace.nb)
    F = kl_features(tpy, tpe, py, pe)                               # (nf, B)
    KX = S_u @ F                                                    # (nx, B)
    star = arm.star
    kl_star_max = float(KX[:, star].max())                          # R0: ~0 (truth == cell); R2: >0 (off-grid)
    B = arm.ncl.B
    base = {"instance": seed, "stream": stream, "level": level, "n_distinct_x": int(len(S_u)), "B": B,
            "kl_truth_to_star_cell_max": kl_star_max}
    eta = arm.eta
    variants = ["infl", "unc"] if level == "R2" else [""]
    st = {v: {"U": np.zeros(B, bool), "Uid": np.zeros(B, bool), "sum_rho": 0.0, "sum_rho_id": 0.0,
              "prev_bad": False, "prev_bad_id": False} for v in variants}
    rows, samples = [], []

    def opt(r, bad):
        return None if (bad or r["cover"] is None) else LOGD * r["cover"]

    for k in range(len(arm.problems)):
        Jt, J, Reg, e = arm.Jt[k], arm.J[k], arm.Reg[k], eta[k]
        pistar = int(np.argmax(Jt))
        rec = {**base, "kind": "oracle", "K": k + 1, "k": k, "problem": arm.problems[k].pid,
               "n_policies": len(arm.problems[k].policies), "H": arm.problems[k].H, "pistar": pistar,
               "true_top2_gap": arm.gap12[k], "star_reg_at_pistar": float(Reg[star, pistar]),
               "eta_min_over_eps": float(e.min()) / EPS, "eta_star_over_eps": float(e[star]) / EPS, "lp_sec": 0.0}
        for v in variants:
            sx = "" if v in ("", "infl") else "_unc"
            S_ = st[v]
            if v == "":
                alt = Reg[:, pistar] > EPS
                alt[star] = False
                alt_id = np.abs(J - J[star][None]).max(1) >= TAU
                alt_id[star] = False
                floor = floor_id = False
            elif v == "infl":
                alt = Reg[:, pistar] + 2.0 * e > EPS
                alt_id = np.abs(J - Jt[None]).max(1) + 2.0 * e >= TAU
                floor, floor_id = bool(2.0 * float(e.min()) > EPS), bool(2.0 * float(e.min()) >= TAU)
            else:                                    # R2 uncorrected (diagnostic; learner JPC on G_1 without eta)
                alt = Reg[:, pistar] > EPS
                alt_id = np.abs(J - Jt[None]).max(1) >= TAU
                floor = floor_id = False
            no_surv, no_surv_id = bool(alt.all()), bool(alt_id.all())
            S_["U"] |= alt
            S_["Uid"] |= alt_id
            bad, bad_id = floor or no_surv, floor_id or no_surv_id
            rs = cover_lp(KX, np.flatnonzero(alt), skip=bad)
            rsi = cover_lp(KX, np.flatnonzero(alt_id), skip=bad_id)
            S_["prev_bad"] = S_["prev_bad"] or bad          # union infeasible once any k' <= K was
            S_["prev_bad_id"] = S_["prev_bad_id"] or bad_id
            ru = cover_lp(KX, np.flatnonzero(S_["U"]), prev=S_.get("ru"), skip=S_["prev_bad"])
            rui = cover_lp(KX, np.flatnonzero(S_["Uid"]), prev=S_.get("rui"), skip=S_["prev_bad_id"])
            S_["ru"], S_["rui"] = ru, rui
            rho, rho_id = opt(rs, bad), opt(rsi, bad_id)
            S_["sum_rho"] = S_["sum_rho"] + rho if (rho is not None and S_["sum_rho"] is not None) else None
            S_["sum_rho_id"] = S_["sum_rho_id"] + rho_id if (rho_id is not None and S_["sum_rho_id"] is not None) else None
            opt_k, opt_id_k = opt(ru, S_["prev_bad"]), opt(rui, S_["prev_bad_id"])
            rec.update({
                f"floor{sx}": floor, f"floor_id{sx}": floor_id, f"no_survivor{sx}": no_surv,
                f"no_survivor_id{sx}": no_surv_id, f"n_alt_k{sx}": int(alt.sum()), f"n_alt_id_k{sx}": int(alt_id.sum()),
                f"n_alt_union{sx}": int(S_["U"].sum()), f"n_alt_id_union{sx}": int(S_["Uid"].sum()),
                f"n_unid_alt_k{sx}": rs["n_unid"], f"n_unid_alt_union{sx}": ru["n_unid"],
                f"n_unid_alt_id_union{sx}": rui["n_unid"],
                f"cover_k{sx}": ru["cover"], f"cover_id_k{sx}": rui["cover"],
                f"opt_k{sx}": opt_k, f"opt_id_k{sx}": opt_id_k, f"rho_star{sx}": rho, f"rho_star_id{sx}": rho_id,
                f"opt_ratio{sx}": (opt_k / opt_id_k) if (opt_k is not None and opt_id_k) else None,
                f"sum_rho_star{sx}": S_["sum_rho"], f"sum_rho_star_id{sx}": S_["sum_rho_id"],
                f"reuse_gain{sx}": (S_["sum_rho"] / opt_k) if (S_["sum_rho"] is not None and opt_k) else None,
                f"reuse_gain_id{sx}": (S_["sum_rho_id"] / opt_id_k) if (S_["sum_rho_id"] is not None and opt_id_k) else None,
                f"lp_status{sx}": {"single": rs["lp_status"], "single_id": rsi["lp_status"],
                                   "union": ru["lp_status"], "union_id": rui["lp_status"]},
                f"lp_min_slack{sx}": {nm: r.get("min_slack") for nm, r in (("single", rs), ("single_id", rsi),
                                                                         ("union", ru), ("union_id", rui))},
                f"lp_cons{sx}": {"union": ru["n_cons"], "union_id": rui["n_cons"]},
                f"lp_reused{sx}": {"union": bool(ru.get("reused")), "union_id": bool(rui.get("reused"))}})
            rec["lp_sec"] += sum(r.get("lp_sec", 0.0) for r in (rs, rsi, ru, rui))
            if want_samples and k in (0, len(arm.problems) - 1) and ru.get("x") is not None:
                x = ru["x"]
                top = np.argsort(-x)[:8]
                samples.append({"instance": seed, "stream": stream, "level": level, "variant": v or "R0", "K": k + 1,
                                "opt_k": opt_k, "opt_id_k": opt_id_k, "rho_star": rho,
                                "J_true": np.round(Jt, 4).tolist(), "pistar": pistar,
                                "top_allocation": [{"x_row": int(i), "n_x_steps": float(LOGD * x[i]),
                                                    "n_features_touched": int(S_u[i].sum())} for i in top if x[i] > 0],
                                "alt_sizes": {"alt_k": int(alt.sum()), "alt_union": int(S_["U"].sum()),
                                              "alt_id_union": int(S_["Uid"].sum())}})
        rows.append(rec)
    extra = []
    if level == "R0":
        idx = np.flatnonzero(np.arange(B) != star)
        rp = cover_lp(KX, idx)
        extra.append({**base, "kind": "oracle_param", "cover_param": rp["cover"],
                      "opt_param": LOGD * rp["cover"] if rp["cover"] is not None else None,
                      "n_unid": rp["n_unid"], "lp_status": rp["lp_status"], "n_cons": rp["n_cons"],
                      "lp_sec": rp.get("lp_sec")})
    return rows + extra, samples, {"instance": seed, "stream": stream, "level": level, "sec": time.time() - t0}


# ---------------------------------------------------------------------------------------- analysis
def load_observed(pilot, seeds):
    """Observed per-row hr1 results (JPC/B8 etc.) for the same instances, if they exist."""
    sub = "pilots" if pilot else "full"
    rows, seen = [], set()
    for pat in ("hr1_r0_*", "hr1_r2_*", "hr4_phase_*"):
        for p in sorted(glob.glob(str(RES_ROOT / sub / pat / "results.jsonl"))):
            if "smoke" in p or "downscaled" in p:
                continue
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("instance") in seeds and r.get("level") in ("R0", "R2") and r.get("f", 1) == 1:
                    key = (r["method"], r["instance"], r["stream"], r["level"], r["k"])
                    if key in seen:                      # same CRN unit run by several chunks: keep the first
                        continue
                    seen.add(key)
                    r["_src"] = Path(p).parent.name
                    rows.append(r)
    return rows


VARIANTS = (("R0", "R0", ""), ("R2_infl", "R2", ""), ("R2_unc", "R2", "_unc"))


def analyze(orows, obs, n_prob):
    O = [r for r in orows if r["kind"] == "oracle"]
    S = {"lp": {}, "monotone": {}, "by_variant": {}, "suspicious_gate": {}, "hr5": {}}
    S["nan_found"] = any(isinstance(v, float) and math.isnan(v) for r in O for v in r.values())

    def q(v):
        v = [x for x in v if x is not None and np.isfinite(x)]
        return None if not v else {"n": len(v), "median": float(np.median(v)), "q10": float(np.quantile(v, .1)),
                                   "q90": float(np.quantile(v, .9)), "min": float(min(v)), "max": float(max(v))}

    OK = (0, "empty", "all_unidentifiable", "skipped_infeasible")
    for name, lv, sx in VARIANTS:
        R = [r for r in O if r["level"] == lv]
        if not R:
            continue
        st = [s for r in R for s in r[f"lp_status{sx}"].values()]
        sl = [v for r in R for v in r[f"lp_min_slack{sx}"].values() if v is not None]
        S["lp"][name] = {"n_lp": len(st), "n_ok_or_trivial": int(sum(s in OK for s in st)),
                         "n_failed": int(sum(s not in OK for s in st)), "min_slack": float(min(sl)) if sl else None}
        nviol = {"opt_k": 0, "opt_id_k": 0}
        units = sorted({(r["instance"], r["stream"]) for r in R})
        fin = []
        for u in units:
            Ru = sorted([r for r in R if (r["instance"], r["stream"]) == u], key=lambda r: r["K"])
            fin.append(Ru[-1])
            for a, b in zip(Ru, Ru[1:]):
                for key in nviol:
                    x, y = a[key + sx], b[key + sx]
                    if x is not None and y is not None and y < x * (1 - 1e-6) - 1e-9:
                        nviol[key] += 1
        S["monotone"][name] = {"opt_k": nviol["opt_k"] == 0, "opt_id_k": nviol["opt_id_k"] == 0, "violations": nviol}
        g = lambda rr, k: [r[k + sx] for r in rr]  # noqa: E731
        byK = {}
        for K in sorted({r["K"] for r in R}):
            RK = [r for r in R if r["K"] == K]
            byK[K] = {"opt_k": q(g(RK, "opt_k")), "opt_id_k": q(g(RK, "opt_id_k")), "opt_ratio": q(g(RK, "opt_ratio")),
                      "reuse_gain": q(g(RK, "reuse_gain")), "n_alt_union": q(g(RK, "n_alt_union"))}
        S["by_variant"][name] = {
            "n_units": len(units), "n_problems": len(R),
            "share_floor": float(np.mean(g(R, "floor"))), "share_floor_id": float(np.mean(g(R, "floor_id"))),
            "share_no_survivor": float(np.mean(g(R, "no_survivor"))),
            "share_no_survivor_id": float(np.mean(g(R, "no_survivor_id"))),
            "share_unidentifiable_in_alt_k": float(np.mean([x > 0 for x in g(R, "n_unid_alt_k")])),
            "share_empty_alt_k": float(np.mean([x == 0 for x in g(R, "n_alt_k")])),
            "share_opt_k_finite": float(np.mean([x is not None for x in g(R, "opt_k")])),
            "rho_star": q(g(R, "rho_star")), "rho_star_id": q(g(R, "rho_star_id")),
            "opt_k_final": q(g(fin, "opt_k")), "opt_id_k_final": q(g(fin, "opt_id_k")),
            "opt_ratio_final": q(g(fin, "opt_ratio")), "reuse_gain_final": q(g(fin, "reuse_gain")),
            "eta_min_over_eps": q([r["eta_min_over_eps"] for r in R]),
            "kl_truth_to_star_cell_max": q([r["kl_truth_to_star_cell_max"] for r in fin]), "by_K": byK}
    P = [r for r in orows if r["kind"] == "oracle_param"]
    if P:
        S["opt_param_R0"] = q([r["opt_param"] for r in P])
    # suspicious gate: observed n0 + cumulative charged steps (uncensored prefix) vs OPT_K of the matching variant
    cov = {(r["instance"], r["stream"], r["level"], r["K"]): r for r in O}
    flags, checked = [], {}
    for lv in ("R0", "R2"):
        Ob = [r for r in obs if r["level"] == lv]
        for m in sorted({r["method"] for r in Ob}):
            sx = "_unc" if (lv == "R2" and not m.endswith("_infl")) else ""
            for u in sorted({(r["instance"], r["stream"]) for r in Ob if r["method"] == m}):
                Ru = sorted([r for r in Ob if r["method"] == m and (r["instance"], r["stream"]) == u], key=lambda r: r["k"])
                cum, ok = N0, True
                for r in Ru:
                    ok = ok and not r["censored"]
                    cum += r["new_env_steps"]
                    c = cov.get((u[0], u[1], lv, r["k"] + 1))
                    if c is None or not ok or c.get("opt_k" + sx) is None:
                        continue
                    checked[f"{lv}:{m}"] = checked.get(f"{lv}:{m}", 0) + 1
                    if cum < c["opt_k" + sx]:
                        flags.append({"method": m, "level": lv, "variant": sx or "main", "instance": u[0], "stream": u[1],
                                      "K": r["k"] + 1, "N_obs": cum, "OPT_K": c["opt_k" + sx],
                                      "ratio": cum / c["opt_k" + sx], "src": r["_src"]})
    S["suspicious_gate"] = {"n_prefixes_checked": checked, "n_below_opt_k": len(flags), "flags": flags[:50],
                            "by_method": {m: sum(f["method"] == m for f in flags) for m in sorted({f["method"] for f in flags})},
                            "N_obs_over_OPT_K_by_method": {},
                            "note": "N_obs = n0 + cumulative charged steps on an uncensored prefix; OPT_K relaxes "
                                    "reachability, starts from zero data and fixes the oracle pi*, so it is a loose "
                                    "ceiling; N_obs < OPT_K -> inspect (different eps-good policy, n0 information, or bug)"}
    # observed / oracle margin per method (median over prefixes)
    for lv in ("R0", "R2"):
        Ob = [r for r in obs if r["level"] == lv]
        for m in sorted({r["method"] for r in Ob}):
            sx = "_unc" if (lv == "R2" and not m.endswith("_infl")) else ""
            ratios = []
            for u in sorted({(r["instance"], r["stream"]) for r in Ob if r["method"] == m}):
                Ru = sorted([r for r in Ob if r["method"] == m and (r["instance"], r["stream"]) == u], key=lambda r: r["k"])
                cum, ok = N0, True
                for r in Ru:
                    ok = ok and not r["censored"]
                    cum += r["new_env_steps"]
                    c = cov.get((u[0], u[1], lv, r["k"] + 1))
                    if c is not None and ok and c.get("opt_k" + sx):
                        ratios.append(cum / c["opt_k" + sx])
            if ratios:
                S["suspicious_gate"]["N_obs_over_OPT_K_by_method"][f"{lv}:{m}"] = q(ratios)
    for name, lv, sx, num, den in (("R0", "R0", "", "JPC", "B8"), ("R2_infl", "R2", "", "JPC_infl", "B8_infl")):
        Ob = [r for r in obs if r["level"] == lv and r["method"] in (num, den)]
        pts, skipped = [], []
        for i in sorted({r["instance"] for r in Ob}):
            lr, sts = {}, []
            for m in (num, den):
                sts = sorted({r["stream"] for r in Ob if r["instance"] == i and r["method"] == m})
                if not sts:
                    break
                lr[m] = float(np.mean([math.log(sum(r["new_env_steps"] for r in Ob if r["instance"] == i and
                                                    r["method"] == m and r["stream"] == s) + 1) for s in sts]))
            if len(lr) < 2:
                continue
            Kobs = max(r["k"] for r in Ob if r["instance"] == i) + 1
            orr = [cov[(i, s, lv, Kobs)]["opt_ratio" + sx] for s in sts if (i, s, lv, Kobs) in cov]
            orr = [x for x in orr if x is not None]
            if not orr:
                skipped.append(i)
                continue
            pts.append({"instance": i, "K": Kobs, "obs_ratio": math.exp(lr[num] - lr[den]),
                        "oracle_ratio": float(np.exp(np.mean(np.log(orr))))})
        rho_s = p = None
        if len(pts) >= 3:
            rr = spearmanr([x["obs_ratio"] for x in pts], [x["oracle_ratio"] for x in pts])
            rho_s = float(rr[0]) if np.isfinite(rr[0]) else None
            p = float(rr[1]) if np.isfinite(rr[1]) else None
        S["hr5"][name] = {"numerator": num, "denominator": den, "n_instances": len(pts), "spearman": rho_s,
                          "p_value": p, "pass_ge_0p7": None if rho_s is None else bool(rho_s >= 0.7),
                          "instances_without_finite_oracle": skipped, "points": pts}
    return S


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
        # main pilot block: 10 dev instances x 10 problems (cached J / eta tables) + HR5 cross-check block on the
        # hr1_r0 / hr1_r2 pilot instances (660-663 x 5 problems) so the Spearman / suspicious-gate pipeline is exercised
        blocks = [(list(range(640, 650)), 10, [0]), (list(range(660, 664)), 5, [0])]
        assert all(s < 10000 for b in blocks for s in b[0])
    else:
        rngs = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds = [s for a, b in rngs for s in range(a, b + 1)]
        blocks = [(seeds, int(lock["eval_manifest"]["K_problems"]), list(range(len(STREAMS))))]
    if args.smoke:
        blocks = [(blocks[0][0][:1], 3, [0])]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    log(f"start {TASK} mode={args.mode} device={dev} blocks={[(b[0][0], b[0][-1], b[1], b[2]) for b in blocks]}")
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA, "tau_id": TAU, "f": 1,
               "levels": ["R0", "R2"], "blocks": [{"seeds": [b[0][0], b[0][-1]], "n_prob": b[1], "streams": b[2]}
                                                  for b in blocks],
               "eval_seeds_touched": not pilot, "generator_hash": generator_hash(), "lock_status": lock.get("status"),
               "concurrent_run": True, "harness_only": True,
               "definitions": {
                   "cover": "min sum_x n_x s.t. sum_x n_x KL_x(theta*,theta) >= 1 for theta in A; x=(obs state, legal "
                            "action), unit cost per env step, reachability relaxed, exact HiGHS LP + cutting planes",
                   "OPT": "log(1/delta) * cover (env steps)",
                   "Alt_eps_R0": "Reg_theta(pi*) > eps, theta != theta*_cell",
                   "Alt_eps_R2": "Reg_theta(pi*) + 2 eta_loc(theta) > eps (locked inflated certificate)",
                   "Alt_id": "||J_theta - J_ref||_inf (+2 eta at R2) >= tau = eps/2 (B8)",
                   "rho_star": "OPT(Alt_eps(q_k)) single problem, fresh start",
                   "opt_ratio": "OPT_K / OPT_id_K", "reuse_gain": "sum_k rho*_k / OPT_K"}}
    res_path, done_path = out_dir / "results.jsonl", out_dir / "units_done.jsonl"
    seeds_all = sorted({s for b in blocks for s in b[0]})
    try:
        if not args.resummarize:
            progress(1, 3, "tables (cached J / eta_loc f=1; J_true)")
            meta = {"cq": {}, "jtrue": {}, "eta_min_over_eps": {}, "sec": {}, "gap12": {}}
            for (ss, n_prob, streams) in blocks:
                m = H.precompute(ss, n_prob, streams, (1,), (1,), dev, log)
                for kk in ("cq", "jtrue"):
                    meta[kk].update(m[kk])
                for lv, d in m["gap12"].items():
                    meta["gap12"].setdefault(lv, {}).update(d)
                summary.setdefault("table_sec", []).append(m["sec"])
            if torch.cuda.is_available():
                gp = {"gpu_name": torch.cuda.get_device_name(0),
                      "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / 2**20),
                      "max_batch_size": None, "vram_used_mb": int(torch.cuda.max_memory_allocated() / 2**20),
                      "utilization_pct": round(100 * torch.cuda.max_memory_allocated() /
                                               torch.cuda.get_device_properties(0).total_memory, 2),
                      "note": "CPU-only LP task; GPU used only for missing J / eta_loc tables (none missing when "
                              "cached); no batch-size probe applicable"}
                (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(gp))
            done_units = set()
            if done_path.exists():
                for x in open(done_path):
                    u = json.loads(x)
                    done_units.add((u["instance"], u["stream"], u["level"]))
            jobs = [(s, st, lv, n_prob) for (ss, n_prob, streams) in blocks for s in ss for st in streams
                    for lv in ("R0", "R2") if (s, st, lv) not in done_units]
            log(f"units: {len(jobs)} to run, {len(done_units)} already done")
            progress(2, 3, "cpu: covering LPs", {"units_total": len(jobs), "units_done": 0})
            from joblib import Parallel, delayed
            first = jobs[0][:3] if jobs else None
            gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
                delayed(lambda s, st, lv, n: (s, st, lv, *_safe(s, st, lv, n, meta, (s, st, lv) == first or
                                                                 (st == 0 and s in (640, 645)))))(s, st, lv, n)
                for (s, st, lv, n) in jobs)
            n_err = 0
            for i, (s, st, lv, rr, ss, err, us) in enumerate(gen):
                with open(res_path, "a") as fh:
                    for r in rr:
                        fh.write(json.dumps(r, default=float) + "\n")
                with open(out_dir / "samples" / "samples.jsonl", "a") as fh:
                    for x in ss:
                        fh.write(json.dumps(x, default=float) + "\n")
                if err:
                    n_err += 1
                    log(f"ERROR {s} s{st} {lv}: {err}")
                    with open(out_dir / "errors.jsonl", "a") as fh:
                        fh.write(json.dumps({"instance": s, "stream": st, "level": lv, "error": err}) + "\n")
                else:
                    with open(done_path, "a") as fh:
                        fh.write(json.dumps({"instance": s, "stream": st, "level": lv}) + "\n")
                log(f"unit {i + 1}/{len(jobs)} {lv} {s} s{st} {us.get('sec', 0):.1f}s")
                progress(2, 3, "cpu: covering LPs", {"units_total": len(jobs), "units_done": i + 1, "errors": n_err})
        rows = [json.loads(x) for x in open(res_path)] if res_path.exists() else []
        progress(3, 3, "analysis")
        obs = load_observed(pilot, set(seeds_all))
        S = analyze(rows, obs, None)
        summary.update(S)
        summary["n_rows"] = len(rows)
        summary["observed_sources"] = sorted({r["_src"] for r in obs})
        summary["errors"] = sum(1 for _ in open(out_dir / "errors.jsonl")) if (out_dir / "errors.jsonl").exists() else 0
        summary["wall_s"] = time.time() - T0
        n_units_done = len({(r["instance"], r["stream"], r["level"]) for r in rows if r["kind"] == "oracle"})
        unit_secs = []
        # projected full: 48 instances x 3 streams x 2 levels x 15 problems
        lps = [r["lp_sec"] for r in rows if r["kind"] == "oracle"]
        summary["timing"] = {"n_units": n_units_done, "lp_sec_per_problem_mean": float(np.mean(lps)) if lps else None}
        pc = summary["lp"]
        ok_all = all(v["n_failed"] == 0 for v in pc.values()) and not S["nan_found"] and \
            all(v["opt_k"] and v["opt_id_k"] for v in S["monotone"].values())
        summary["pass_criteria"] = {"lp_solved_all": all(v["n_failed"] == 0 for v in pc.values()),
                                    "opt_k_monotone": all(v["opt_k"] and v["opt_id_k"] for v in S["monotone"].values()),
                                    "no_nan": not S["nan_found"], "pass": bool(ok_all)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        log(f"done: pass={ok_all} wall={summary['wall_s']:.0f}s")
        if real:
            mark_done("success" if ok_all and summary["errors"] == 0 else "partial",
                      f"pass={ok_all}; errors={summary['errors']}; rows={len(rows)}")
    except Exception as e:  # noqa: BLE001
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
        if real:
            mark_done("failed", repr(e))
        raise


def _safe(s, st, lv, n, meta, want):
    try:
        rr, ss, us = run_unit(s, st, lv, n, meta, want)
        return rr, ss, None, us
    except Exception as e:  # noqa: BLE001
        return [], [], repr(e) + "\n" + traceback.format_exc(), {}


if __name__ == "__main__":
    main()
