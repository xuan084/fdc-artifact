"""Width proposition (plan/width_proposition.md): predicted FDC-BF / rectangle rows ratios from DEV cell statistics.

NEW file, descriptive only.  Reads DEV halves only (CR9, CR12: PoolReplayEnv(layer, "dev"); X9: X5LayerEnv("dev")).
No evaluation outcome is loaded; observed ratios are read from the sealed analysis JSONs for comparison.

Fluid model (no sampling noise): frozen 50/50 plan with expected counts
    A_s(t) = w_s t,   n_{s,a}(t) = min(N_{s,a}, max(A_s / 2, A_s - N_{s,1-a})),
estimates equal to the dev pool means (mu_hat = mu, Delta_hat = Delta), centre = true optimum of each problem.
Problem q is certified at the first t with max_{pi' in Pi_q} [Delta(pi', pi*_q) + W(pi', pi*_q; t)] <= eps; N80 is the
12th smallest certification time (continuous t, log-interpolated on a fine grid) and its checkpoint-rounded value.

Predictors (W = width of the joint certificate J or of a rectangle R):
  Corollary 2 (exact, analytic): leading Gaussian terms W_J^2 = 2 beta_J sum a_c^2 v_c, W_R^2 = 2 beta_C (sum |a_c|
         sqrt v_c)^2 with UNCAPPED proportional counts n_c = (w_s/2) t and v_c = vtilde_c u(t); u = 1/t (scale model,
         T = inf) or u = 1/t - 1/T with one common horizon T = tau_R.  Reports r12 = u^R_[12] / u^J_[12], the binding-
         direction sandwich, the common-horizon ratio r/(1 - x(1 - r)) and its sandwich interval.  These are exact
         statements about the idealised model (no grid).
  G-FPC  leading terms with the ACTUAL pools (capped counts, re-selection, exhaustion, cell-specific FPC), log grid.
  BF     fluid run of the width code: FDC-BF width (fdc_bet.direction_widths) versus the matched rectangle RECT-ck-BF
         (pjc_bf.bennett_cell_radius, same HG variance boxes, beta_C = ln(2 S A K / delta_main)), boxes from exact HG
         inversion at fluid counts (alpha_var = delta_var / (2 S A K)).  Omits running intersections across checkpoints.
  HG     FDC-BF (as BF) versus RECT-ck-HG (exact HG inversion at alpha = delta / (2 S A K)), fluid counts.
The "heuristic_point" substitutes the q12 joint-binding rho^2 for r12; it is a heuristic, not a corollary value.

Usage (cwd = exp/code):  .venv/bin/python3 predict_width_ratio.py [--grid 160]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/width_prop"
FULL = WS / "exp/results/full"

from dsswm.baselines.fdc import union_size  # noqa: E402
from dsswm.baselines.fdc_bet import direction_widths  # noqa: E402
from dsswm.baselines.frontier_common import jhat_all, pi_hat_indices, rect_U  # noqa: E402
from dsswm.baselines.pjc_bf import bennett_cell_radius  # noqa: E402
from dsswm.baselines.rect_v6 import hg_mean_interval  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx  # noqa: E402

DELTA, D_MAIN, D_VAR = 0.05, 0.045, 0.005
STOP_K = 12
LOGS = (("CR9", (0.001,)), ("X9", (0.015, 0.02, 0.03)), ("CR12", (0.001,)))


# ------------------------------------------------------------------------------------------------ data (DEV only)
def load(layer):
    if layer == "X9":
        from dsswm.envs.x5_v8 import X5LayerEnv
        env = X5LayerEnv("dev")
    else:
        from dsswm.envs.pool_replay import PoolReplayEnv
        env = PoolReplayEnv(layer, "dev")
    assert env.half == "dev"
    from dsswm.streams import frontier as fr
    return env, fr.cr_problems("visit"), env.true_mu("visit")


def fluid_counts(w, N, t):
    """Expected counts of the frozen 50/50 plan after t arrivals (vectorised over t)."""
    t = np.atleast_1d(np.asarray(t, float))
    A = w[None, :] * t[:, None]                                  # (T, S)
    n0 = np.minimum(N[None, :, 0], np.maximum(A / 2, A - N[None, :, 1]))
    n1 = np.minimum(N[None, :, 1], np.maximum(A / 2, A - N[None, :, 0]))
    return np.stack([n0, n1], -1)                                # (T, S, 2)


# ------------------------------------------------------------------------------------------------ Gaussian predictor
def gauss_U(ctx, mu, ih, n, N, beta_j, beta_c, fpc):
    """Leading-term widths for all challengers at one t. Returns U_J, U_R, V (sum a^2 v), L (sum |a| sqrt v)."""
    S = ctx.S
    seg = np.arange(S)
    ph = ctx.pols[ih]
    D = ctx.pols != ph[None, :]
    sig2 = mu * (1 - mu)
    live = (n > 0) & (n < N)
    f = (N - n) / np.maximum(N - 1, 1) if fpc else np.ones_like(n)
    v = np.where(live, sig2 * f / np.maximum(n, 1e-12), 0.0)
    vch = v[seg[None, :], ctx.pols]
    vh = v[seg, ph][None, :]
    w = ctx.w[None, :]
    V = (D * w ** 2 * (vch + vh)).sum(1)
    L = (D * w * (np.sqrt(vch) + np.sqrt(vh))).sum(1)
    J = jhat_all(ctx, mu)
    gap = J - J[ih]                                              # Delta(pi', pi*) <= 0 on the feasible class
    return gap + np.sqrt(2 * beta_j * V), gap + math.sqrt(2 * beta_c) * L, V, L


# ------------------------------------------------------------------------------------------------ Corollary 2 models
def analytic_u(ctx, mu, ihs, beta_j, beta_c):
    """Exact Corollary 2 quantities under (S1)-(S5) with uncapped proportional counts n_c = (w_s / 2) t:
    squared widths C^J_a u(t), C^R_a u(t) with vtilde_c = mu_c (1 - mu_c) / (w_s / 2) (N/(N-1) ~ 1 omitted);
    u_q = min_a theta_a^2 / C_a. Returns u^J, u^R (per problem) and binding-direction summaries (no time grid)."""
    seg = np.arange(ctx.S)
    vt = mu * (1 - mu) / (np.maximum(ctx.w, 1e-300)[:, None] / 2.0)
    J = jhat_all(ctx, mu)
    uJ, uR, bind = np.full(ctx.Q, np.inf), np.full(ctx.Q, np.inf), []
    for q in range(ctx.Q):
        ih = int(ihs[q])
        ph = ctx.pols[ih]
        D = ctx.pols != ph[None, :]
        vch, vh = vt[seg[None, :], ctx.pols], vt[seg, ph][None, :]
        w = ctx.w[None, :]
        V = (D * w ** 2 * (vch + vh)).sum(1)
        L = (D * w * (np.sqrt(vch) + np.sqrt(vh))).sum(1)
        theta = ctx.eps - (J - J[ih])
        ok = ctx.feas[q] & (V > 0)
        CJ, CR = 2 * beta_j * V, 2 * beta_c * L ** 2
        aJ = np.where(ok, theta ** 2 / np.where(ok, CJ, 1.0), np.inf)
        aR = np.where(ok, theta ** 2 / np.where(ok, CR, 1.0), np.inf)
        pJ, pR = int(np.argmin(aJ)), int(np.argmin(aR))
        uJ[q], uR[q] = aJ[pJ], aR[pR]
        rec = {"q": q}
        for nm, pp in (("J", pJ), ("R", pR)):
            k2 = float(V[pp] / L[pp] ** 2)
            rec[f"binding_{nm}"] = {"pi": ctx.pols[pp].tolist(), "D": int(D[pp].sum()), "kappa2": k2,
                                    "m_eff": 1.0 / k2, "rho2": beta_j / beta_c * k2, "theta": float(theta[pp])}
        bind.append(rec)
    return uJ, uR, bind


def kth_largest(x, k=STOP_K):
    return float(np.sort(x)[::-1][k - 1])


def g_fpc(r, x):
    """Corollary 2(c): N80^J / N80^R = r / (1 - x (1 - r))."""
    return r / (1.0 - x * (1.0 - r))


# ------------------------------------------------------------------------------------------------ certification times
def cert_times(Ufun, ctx, ihs, tgrid):
    """First (log-interpolated) t at which max_{feasible} U <= eps, per problem. Ufun(ih, k) -> U vector over P."""
    Q = ctx.Q
    Uq = np.full((len(tgrid), Q), np.inf)
    cache = {}
    for k in range(len(tgrid)):
        for q in range(Q):
            ih = int(ihs[q])
            if (ih, k) not in cache:
                cache[(ih, k)] = Ufun(ih, k)
            Uq[k, q] = np.max(np.where(ctx.feas[q], cache[(ih, k)], -np.inf))
        if k > 0:
            for key in [kk for kk in cache if kk[1] == k - 1]:
                del cache[key]
    tq = np.full(Q, np.inf)
    lt = np.log(tgrid)
    for q in range(Q):
        ok = np.flatnonzero(Uq[:, q] <= ctx.eps)
        if not len(ok):
            continue
        k = ok[0]
        if k == 0:
            tq[q] = tgrid[0]
            continue
        u0, u1 = Uq[k - 1, q], Uq[k, q]
        frac = (u0 - ctx.eps) / (u0 - u1) if np.isfinite(u0) and u0 > u1 else 1.0
        tq[q] = float(np.exp(lt[k - 1] + frac * (lt[k] - lt[k - 1])))
    return tq, Uq


def n80(tq, cps):
    t = float(np.sort(tq)[STOP_K - 1])
    if not np.isfinite(t):
        return math.inf, math.inf
    i = np.searchsorted(cps, t - 1e-9)
    return t, float(cps[min(i, len(cps) - 1)])


# ------------------------------------------------------------------------------------------------ binding directions
# ------------------------------------------------------------------------------------------------ main
def observed():
    def g(f, key, m, eps):
        d = json.load(open(FULL / f))
        try:
            x = d["per_eps"][eps][key][m]
            return {"ratio": x["geomean_ratio"], "ci95": x["ci95_two_sided"], "ub95": x["ub95_one_sided"], "src": f}
        except KeyError:
            return None
    obs = {"CR9@0.001": {"RECT-ck-BF": g("v8_analysis_C.json", "FDC-BF|N80_pen", "RECT-ck-BF", "0.001"),
                         "RECT-ck-BF+box": g("v8_analysis_C.json", "FDC-BF|N80_pen", "RECT-ck-BF+box", "0.001"),
                         "RECT-ck-HG": g("v7_analysis_A.json", "FDC-BF|N80_pen", "RECT-ck-HG", "0.001")},
           "CR12@0.001": {"RECT-ck-BF": None,
                          "RECT-ck-BF+box": g("v8_analysis_B.json", "FDC-BF|N80_pen", "RECT-ck-BF+box", "0.001"),
                          "RECT-ck-HG": g("v8_analysis_B.json", "FDC-BF|N80_pen", "RECT-ck-HG", "0.001")}}
    for e in ("0.015", "0.02", "0.03"):
        obs[f"X9@{float(e):g}"] = {m: g("v8_analysis_A.json", "FDC-BF|N80_pen", m, e)
                                   for m in ("RECT-ck-BF", "RECT-ck-BF+box", "RECT-ck-HG")}
    return obs


def run_log(layer, eps_list, ngrid):
    env, probs, mu = load(layer)
    N = np.asarray(env.pool_sizes, float)
    res = {}
    for eps in eps_list:
        ctx = build_ctx(env, probs, eps)
        K = len(ctx.checkpoints)
        SA = ctx.S * ctx.A
        M = union_size(ctx, "feas")
        beta_j = math.log(M * K / D_MAIN)
        beta_c = math.log(2 * SA * K / D_MAIN)
        a_var = D_VAR / (2 * SA * K)
        a_hg = DELTA / (2 * SA * K)
        cps = np.asarray(ctx.checkpoints, float)
        ihs = pi_hat_indices(ctx, mu)
        tg = np.exp(np.linspace(math.log(cps[0]), math.log(cps[-1]), ngrid))
        tg[-1] = cps[-1]
        nT = fluid_counts(ctx.w, N, tg)
        rec = {"layer": layer, "half": "dev", "eps": eps, "tau_R": int(env.tau_R), "S": ctx.S, "union_M": M, "K": K,
               "beta_J": beta_j, "beta_C": beta_c, "beta_ratio": beta_j / beta_c,
               "beta_ratio_over_2SA": beta_j / beta_c / (2 * ctx.S)}
        # ---- Corollary 2, exact (analytic; no time grid): scaling model (T = inf) and common horizon T = tau_R
        T = float(env.tau_R)
        uJ, uR, bind = analytic_u(ctx, mu, ihs, beta_j, beta_c)
        rJ = np.array([b["binding_J"]["rho2"] for b in bind])
        rR = np.array([b["binding_R"]["rho2"] for b in bind])
        with np.errstate(divide="ignore", invalid="ignore"):
            per_r = uR / uJ                                          # = t^J_q / t^R_q in the T = inf model
        tol = 1e-9
        inside = (per_r >= rR * (1 - tol)) & (per_r <= rJ * (1 + tol))
        r12 = kth_largest(uR) / kth_largest(uJ)
        r_lo, r_hi = float(rR.min()), float(rJ.max())
        tJ_ch = 1.0 / (uJ + 1.0 / T)
        tR_ch = 1.0 / (uR + 1.0 / T)
        N80J_ch, N80R_ch = float(np.sort(tJ_ch)[STOP_K - 1]), float(np.sort(tR_ch)[STOP_K - 1])
        x_ch = N80R_ch / T
        tJ_sc = 1.0 / uJ
        order = np.argsort(tJ_sc)[:STOP_K]
        q12 = int(order[-1])
        Tc = N / (np.maximum(ctx.w, 1e-12)[:, None] / 2.0)
        rec["corollary2"] = {
            "binding": bind,
            "per_problem_r_scale": per_r.tolist(), "per_problem_inside_sandwich_exact": inside.tolist(),
            "all_inside": bool(inside.all()),
            "r12": r12, "sandwich_all_q": [r_lo, r_hi],
            "outer_interval": [beta_j / beta_c / (2 * ctx.S), beta_j / beta_c],
            "scale_model_N80_ratio": r12,
            "common_horizon_model": {"T": T, "N80_J": N80J_ch, "N80_R": N80R_ch, "x_R": x_ch,
                                     "N80_ratio": N80J_ch / N80R_ch, "closed_form_check": g_fpc(r12, x_ch),
                                     "interval_from_sandwich": [g_fpc(r_lo, x_ch), g_fpc(r_hi, x_ch)]},
            "cell_horizon_over_tauR_range": [float(Tc.min() / T), float(Tc.max() / T)],
            "m_eff_binding_J_first12_median": float(np.median([bind[q]["binding_J"]["m_eff"] for q in order])),
            "m_eff_binding_R_first12_median": float(np.median([bind[q]["binding_R"]["m_eff"] for q in order])),
            "q12_scale": q12, "m_eff_bindingJ_q12": bind[q12]["binding_J"]["m_eff"],
            "rho2_bindingJ_q12": bind[q12]["binding_J"]["rho2"], "D_bindingJ_q12": bind[q12]["binding_J"]["D"],
            "heuristic_point_rho2q12_in_closed_form": g_fpc(bind[q12]["binding_J"]["rho2"], x_ch)}
        # ---- G-FPC: leading terms with the ACTUAL pools (capped counts, re-selection, exhaustion, cell FPC), grid
        tJ, _ = cert_times(lambda ih, k: gauss_U(ctx, mu, ih, nT[k], N, beta_j, beta_c, True)[0], ctx, ihs, tg)
        tR, _ = cert_times(lambda ih, k: gauss_U(ctx, mu, ih, nT[k], N, beta_j, beta_c, True)[1], ctx, ihs, tg)
        cJ, dJ = n80(tJ, cps)
        cR, dR = n80(tR, cps)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_eff = (1 / tR - 1 / T) / (1 / tJ - 1 / T)
        ins_tol = (r_eff >= rR - 1e-3) & (r_eff <= rJ + 1e-3)
        ins_ex = (r_eff >= rR) & (r_eff <= rJ)
        rec["G-FPC"] = {"N80_J": cJ, "N80_R": cR, "ratio_cont": cJ / cR, "N80_J_ck": dJ, "N80_R_ck": dR,
                        "ratio_ck": dJ / dR, "t_q_J": tJ.tolist(), "t_q_R": tR.tolist(),
                        "r_eff_common_T_per_problem": r_eff.tolist(),
                        "n_inside_sandwich_tol1e-3": int(ins_tol.sum()), "n_inside_sandwich_exact": int(ins_ex.sum()),
                        "note": "actual pools: common-horizon assumption only approximate; log-grid interpolation"}
        # ---- code-faithful fluid predictors (BF and HG), evaluated on a coarser grid
        cgrid = tg[:: max(1, ngrid // 80)]
        if cgrid[-1] != tg[-1]:
            cgrid = np.append(cgrid, tg[-1])
        nC = np.rint(fluid_counts(ctx.w, N, cgrid)).astype(np.int64)
        Ni = N.astype(np.int64)
        sums = np.rint(nC * mu[None]).astype(np.int64)
        boxes, rectBF, rectBFb, rectHG = [], [], [], []
        for k in range(len(cgrid)):
            n, s = nC[k], sums[k]
            vlo, vhi = hg_mean_interval(Ni, n, s, a_var)
            boxes.append((vlo, vhi))
            r = bennett_cell_radius(n.astype(float), N, vlo, vhi, beta_c)
            m = np.where(n > 0, s / np.maximum(n, 1), 0.5)
            lo = np.maximum(m - r, s / N)
            hi = np.minimum(m + r, (s + (N - n)) / N)
            ex = n >= Ni
            lo, hi = np.where(ex, m, lo), np.where(ex, m, hi)
            rectBF.append((np.clip(lo, 0, 1), np.clip(np.maximum(hi, lo), 0, 1)))
            lob, hib = np.maximum(lo, vlo), np.minimum(hi, vhi)
            rectBFb.append((np.clip(lob, 0, 1), np.clip(np.maximum(hib, lob), 0, 1)))
            rectHG.append(hg_mean_interval(Ni, n, s, a_hg))
        J = jhat_all(ctx, mu)

        def U_fdc(ih, k):
            lo, hi = boxes[k]
            return (J - J[ih]) + direction_widths(ctx, ih, nC[k].astype(float), N, lo, hi, beta_j, "bennett")

        tJ, _ = cert_times(U_fdc, ctx, ihs, cgrid)
        cJ, dJ = n80(tJ, cps)
        rec["fluid_FDC-BF"] = {"N80": cJ, "N80_ck": dJ, "t_q": tJ.tolist()}
        for tag, rb in (("RECT-ck-BF", rectBF), ("RECT-ck-BF+box", rectBFb), ("RECT-ck-HG", rectHG)):
            tR, _ = cert_times(lambda ih, k: rect_U(ctx, rb[k][0], rb[k][1], ih), ctx, ihs, cgrid)
            cR, dR = n80(tR, cps)
            rec[f"fluid_{tag}"] = {"N80": cR, "N80_ck": dR, "ratio_cont": cJ / cR, "ratio_ck": dJ / dR,
                                   "t_q": tR.tolist(), "per_problem_ratio": (tJ / tR).tolist()}
        # Proposition 1 remainder terms eta_J, eta_R at the fluid FDC-BF stopping time, for each problem's J-binding
        # direction (box-sup variance proxies from HG boxes at fluid counts; exact definitions of the proposition)
        t80 = rec["fluid_FDC-BF"]["N80"]
        if np.isfinite(t80):
            n80c = np.rint(fluid_counts(ctx.w, N, t80)[0]).astype(np.int64)
            s80 = np.rint(n80c * mu).astype(np.int64)
            vlo, vhi = hg_mean_interval(Ni, n80c, s80, a_var)
            mm = np.clip(0.5, vlo, vhi)
            live = (n80c > 0) & (n80c < Ni)
            vprox = np.where(live, mm * (1 - mm) * (N - n80c) / np.maximum(N - 1, 1) / np.maximum(n80c, 1), 0.0)
            bc = np.where(live, 1.0 / np.maximum(n80c, 1), 0.0)
            etas = []
            for q in range(ctx.Q):
                bq = rec["corollary2"]["binding"][q]
                if not bq:
                    continue
                pi1 = np.asarray(bq["binding_J"]["pi"])
                ph = ctx.pols[int(ihs[q])]
                cells = [(s_, int(pi1[s_])) for s_ in range(ctx.S) if pi1[s_] != ph[s_]] + \
                        [(s_, int(ph[s_])) for s_ in range(ctx.S) if pi1[s_] != ph[s_]]
                a = np.array([ctx.w[s_] for s_, _ in cells])
                g = np.array([a[i] * math.sqrt(vprox[c]) for i, c in enumerate(cells)])
                ab = np.array([a[i] * bc[c] for i, c in enumerate(cells)])
                if g.sum() <= 0:
                    continue
                etas.append({"q": q, "eta_J": float(ab.max() * math.sqrt(beta_j) / (3 * math.sqrt(2) * np.linalg.norm(g))),
                             "eta_R": float(ab.sum() * math.sqrt(beta_c) / (3 * math.sqrt(2) * g.sum())),
                             "m_eff_box": float(g.sum() ** 2 / (g ** 2).sum())})
            rec["eta_at_fluid_N80"] = {"t": t80, "per_problem": etas,
                                       "eta_J_max": max(e["eta_J"] for e in etas),
                                       "eta_R_max": max(e["eta_R"] for e in etas),
                                       "eta_J_median": float(np.median([e["eta_J"] for e in etas])),
                                       "eta_R_median": float(np.median([e["eta_R"] for e in etas]))}
        # Gaussian-limit HG prediction (Remark 3): beta_C replaced by z_alpha^2 / 2 at alpha = delta / (2 S A K)
        from scipy.stats import norm
        z_hg = float(norm.isf(a_hg))
        z_c = float(norm.isf(D_MAIN / (2 * SA * K)))
        z_j = float(norm.isf(D_MAIN / (M * K)))
        rec["gauss_limit"] = {"z_J": z_j, "z_C": z_c, "z_HG": z_hg, "zJ2_over_zC2": (z_j / z_c) ** 2,
                              "beta_ratio": beta_j / beta_c,
                              "HG_over_BF_rows_noFPC_known_var": z_hg ** 2 / (2 * beta_c),
                              "note": "heuristic diagnostics (fixed levels; no FPC; known variances)"}
        res[f"{layer}@{eps:g}"] = rec
        c2 = rec["corollary2"]
        print(f"{layer}@{eps:g}: bJ={beta_j:.2f} bC={beta_c:.2f} scale r12={c2['r12']:.3f} sandwich={c2['sandwich_all_q']} "
              f"inside={c2['all_inside']} CH={c2['common_horizon_model']['N80_ratio']:.3f} (x={c2['common_horizon_model']['x_R']:.2f}, "
              f"int={c2['common_horizon_model']['interval_from_sandwich']}) G-FPC={rec['G-FPC']['ratio_cont']:.3f} "
              f"BF={rec['fluid_RECT-ck-BF']['ratio_cont']:.3f} HG={rec['fluid_RECT-ck-HG']['ratio_cont']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=160)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"note": "descriptive; DEV halves only (CR9/CR12 PoolReplayEnv dev, X9 X5LayerEnv dev); fluid model; "
                   "observed ratios are FDC-BF / method paired geomean N80_pen on the confirmatory / descriptive "
                   "evaluation blocks (read from sealed analysis JSONs)",
           "written_at": datetime.now().isoformat(timespec="seconds"), "grid": args.grid, "stop_k": STOP_K,
           "ledger": {"delta": DELTA, "delta_main": D_MAIN, "delta_var": D_VAR}, "logs": {}}
    for layer, eps in LOGS:
        out["logs"].update(run_log(layer, eps, args.grid))
    out["observed"] = observed()
    tab = []
    for key, rec in out["logs"].items():
        ob = out["observed"].get(key, {})
        c2 = rec["corollary2"]
        ch = c2["common_horizon_model"]
        tab.append({"log": key, "beta_J/beta_C": rec["beta_ratio"],
                    "m_eff_bindingJ_first12_med": c2["m_eff_binding_J_first12_median"],
                    "m_eff_bindingJ_q12": c2["m_eff_bindingJ_q12"], "rho2_bindingJ_q12": c2["rho2_bindingJ_q12"],
                    "scale_r12": c2["r12"], "sandwich": c2["sandwich_all_q"], "all_inside_exact": c2["all_inside"],
                    "CH_x": ch["x_R"], "CH_ratio": ch["N80_ratio"], "CH_interval": ch["interval_from_sandwich"],
                    "horizon_range": c2["cell_horizon_over_tauR_range"],
                    "eta_J_med": (rec.get("eta_at_fluid_N80") or {}).get("eta_J_median"),
                    "eta_R_med": (rec.get("eta_at_fluid_N80") or {}).get("eta_R_median"),
                    "eta_max": max((rec.get("eta_at_fluid_N80") or {}).get("eta_J_max", 0),
                                   (rec.get("eta_at_fluid_N80") or {}).get("eta_R_max", 0)),
                    "pred_G_FPC": rec["G-FPC"]["ratio_cont"],
                    "pred_fluid_BF": rec["fluid_RECT-ck-BF"]["ratio_cont"],
                    "pred_fluid_BF_ck": rec["fluid_RECT-ck-BF"]["ratio_ck"],
                    "pred_fluid_BFbox": rec["fluid_RECT-ck-BF+box"]["ratio_cont"],
                    "pred_fluid_HG": rec["fluid_RECT-ck-HG"]["ratio_cont"],
                    "pred_fluid_HG_ck": rec["fluid_RECT-ck-HG"]["ratio_ck"],
                    "obs_BF": (ob.get("RECT-ck-BF") or {}).get("ratio"),
                    "obs_BFbox": (ob.get("RECT-ck-BF+box") or {}).get("ratio"),
                    "obs_HG": (ob.get("RECT-ck-HG") or {}).get("ratio")})
    out["table"] = tab
    (OUT / "prediction.json").write_text(json.dumps(out, indent=1, default=float))
    for r in tab:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
