"""HARNESS-SIDE mechanism checks (reads ground truth). Learner code must NEVER import this module
(tests/test_leverage.py checks it by AST and by the runtime import closure).

T0(i) identity self-check (Thm 1):  dJ* - dJ_theta = <h_theta, r> + Rem,   r = mu* - mu_theta (per cell).
  * dJ* and dJ_theta come from the ORIGINAL exact propagator (exact/nl_propagate.j_table, tables built from the
    parameter dicts -- an independent code path from the mu-parametrised propagator of mechanism/leverage.py);
  * <h, r> uses the reverse-mode leverage of leverage.py;
  * Rem is computed independently as the integral form of the Taylor remainder along mu(s) = mu_theta + s r,
        Rem = int_0^1 (1 - s) g''(s) ds,   g(s) = dJ(mu(s)),
    by Gauss-Legendre quadrature (g is a polynomial in s of degree <= H (min(L,R) + P) + 1, the rule is exact).
  identity_err = | <h, r> + Rem_GL - (dJ* - dJ_theta) |   (pre-registered tolerance 1e-9).
Oracle references (comparison only, never competitors): Lambda_Delta = |<(h/xi)^perp, r>_xi| / eps, ||r||_xi,
exact Rem and the validity ratio |Rem| / Rem_bar.
"""
from __future__ import annotations

import numpy as np
import torch

from ..exact.nl_propagate import params_to_torch
from .leverage import orth_leverage


def nl_mu_truth(prop, truth_params: dict) -> np.ndarray:
    """Per-cell true means mu*(x) (cell layout of leverage.NLLeverage) from the truth parameter dict."""
    P = params_to_torch(truth_params, device=prop.device)
    if truth_params.get("g_ret") is not None:
        P["g_ret"] = torch.as_tensor(np.asarray(truth_params["g_ret"], float), device=prop.device, dtype=prop.dtype)
    LT, EY = prop.tables(P)
    mu_pair = EY[0, :prop.U_pair].cpu().numpy()
    N = prop.N
    re = []
    for p in range(prop.P):
        for y in range(2):
            for n in range(N):
                re.append(float(torch.exp(LT[0, prop.re_idx(p, y, n, 1)])))
    return np.concatenate([mu_pair, np.asarray(re)])


def _jtable_pair(prop, params_t: dict, problem, k1: int, k2: int) -> float:
    J = prop.j_table(params_t, [problem.policies[k1], problem.policies[k2]], problem.loads0, problem.engaged0,
                     problem.H, problem.utility)
    return float(J[0, 0] - J[0, 1])


def gl_remainder(lev, mu_hat: np.ndarray, r: np.ndarray, problem, k1: int, k2: int, n_nodes: int | None = None):
    """int_0^1 (1-s) g''(s) ds by Gauss-Legendre, g(s) = J_k1 - J_k2 at mu_hat + s r (double reverse-mode on scalar s)."""
    deg = problem.H * (min(lev.L, lev.R) + lev.P) + 1
    n = n_nodes or (deg // 2 + 2)
    xs, ws = np.polynomial.legendre.leggauss(n)
    ss, ws = 0.5 * (xs + 1.0), 0.5 * ws
    dev, dt = lev.dev, lev.dt
    mu0 = torch.as_tensor(mu_hat, device=dev, dtype=dt).reshape(1, -1)
    rt = torch.as_tensor(r, device=dev, dtype=dt).reshape(1, -1)
    u = problem.utility
    tot = 0.0
    for s_val, w in zip(ss, ws):
        s = torch.tensor(float(s_val), device=dev, dtype=dt, requires_grad=True)
        LT, EY = lev.lt_from_mu(mu0 + s * rt)
        g = 0.0
        for k, sign in ((k1, 1.0), (k2, -1.0)):
            plan, esum = lev.plan(problem, k)
            g = g + sign * lev._run(plan, esum, LT, EY, u.w, u.w_ret, u.c_q)[0].sum()
        g1, = torch.autograd.grad(g, s, create_graph=True)
        g2, = torch.autograd.grad(g1, s)
        tot += float(w) * (1.0 - float(s_val)) * float(g2)
    return tot, n


def nl_identity_check(lev, problem, k1: int, k2: int, theta_hat_params: dict, truth_params: dict,
                      xi: np.ndarray | None = None, eps: float = 0.02, n_nodes: int | None = None) -> dict:
    """T0(i) on one (theta_hat, truth, contrast). theta_hat_params: class-row parameter dict (np, single theta)."""
    prop = lev.prop
    mu_hat = lev.mu_from_theta(lev.theta_vector(theta_hat_params))
    mu_star = nl_mu_truth(prop, truth_params)
    r = mu_star - mu_hat
    dJ_star = _jtable_pair(prop, _to_t(prop, truth_params), problem, k1, k2)
    dJ_hat = _jtable_pair(prop, params_to_torch(theta_hat_params, device=prop.device), problem, k1, k2)
    cl = lev.contrast_leverage(mu_hat, problem, k1, k2)
    first = float(cl["h"] @ r)
    rem_gl, n = gl_remainder(lev, mu_hat, r, problem, k1, k2, n_nodes)
    lhs = dJ_star - dJ_hat
    out = {"dJ_star": dJ_star, "dJ_hat": dJ_hat, "dJ_hat_mu": cl["dJ"], "first_order": first, "rem_gl": rem_gl,
           "rem_exact": lhs - first, "identity_err": abs(first + rem_gl - lhs),
           "plugin_consistency_err": abs(cl["dJ"] - dJ_hat), "gl_nodes": n, "r_inf": float(np.abs(r).max())}
    if xi is not None:
        ol = orth_leverage(cl["h"], xi, lev.Phi)
        supp = xi > 0
        lam_delta = abs(float(np.sum(xi[supp] * ol["resid"][supp] * r[supp]))
                        + float(np.sum(cl["h"][~supp] * r[~supp])))
        out.update({"lambda_delta": lam_delta / eps, "r_norm_xi": float(np.sqrt(np.sum(xi * r * r))),
                    "score_inner": float(np.abs(lev.Phi[supp].T @ (xi[supp] * r[supp])).max())})
    return out


def _to_t(prop, truth_params: dict) -> dict:
    P = params_to_torch(truth_params, device=prop.device)
    if truth_params.get("g_ret") is not None:
        P["g_ret"] = torch.as_tensor(np.asarray(truth_params["g_ret"], float), device=prop.device, dtype=prop.dtype)
    return P


def rem_bound_validity(lev, problem, k: int, mu_hat: np.ndarray, a: np.ndarray, rem_bar: float, n_samples: int = 50,
                       seed: int = 0) -> dict:
    """Max over sampled mu* in the box (|mu* - mu_hat| <= a; vertices and interior) of |Rem| / Rem_bar (must be <= 1)."""
    rng = np.random.default_rng(seed)
    g = lev.policy_leverage(mu_hat, problem, k)
    worst = 0.0
    for i in range(n_samples):
        r = a * (rng.choice([-1.0, 1.0], len(a)) if i % 2 == 0 else rng.uniform(-1, 1, len(a)))
        ms = np.clip(mu_hat + r, 1e-12, 1 - 1e-12)
        r = ms - mu_hat
        rem = float(lev.J_mu(ms, problem, [k]).reshape(-1)[0]) - g["J"] - float(g["grad"] @ r)
        worst = max(worst, abs(rem))
    return {"rem_sampled_max": worst, "rem_bar": rem_bar, "ratio": worst / rem_bar if rem_bar > 0 else (
        0.0 if worst == 0 else np.inf)}


def lin_identity_check(llev, problem, k1: int, k2: int, theta_hat: np.ndarray, theta_star: np.ndarray, aspace,
                       truth_class) -> dict:
    """Lin: dJ is linear in mu, so dJ* - dJ_hat = <h, r> exactly (Rem = 0); dJ from z-vectors of the TRUE class."""
    h = llev.contrast_leverage(problem, k1, k2, aspace)
    Phi_t = np.stack([truth_class.feature(*c) for c in llev.cell_info])
    mu_star = Phi_t @ np.asarray(theta_star, float)
    mu_hat = llev.Phi @ np.asarray(theta_hat, float)
    r = mu_star - mu_hat
    z1 = truth_class.z(problem.policies[k1], problem.loads0, problem.H, problem.utility, aspace)
    z2 = truth_class.z(problem.policies[k2], problem.loads0, problem.H, problem.utility, aspace)
    dJ_star = float((z1 - z2) @ theta_star)
    dJ_hat = float(h @ mu_hat)
    return {"dJ_star": dJ_star, "dJ_hat": dJ_hat, "first_order": float(h @ r),
            "identity_err": abs(float(h @ r) - (dJ_star - dJ_hat))}
