"""eta_loc(g; q): data-free upper bound of max_pi sup_{theta in C(g)} |J_theta(pi) - J_g(pi)| (learner side).

For a grid point g with clipped Voronoi cell C(g) (models.grid_ladder) and any theta in C(g), Taylor's theorem with
Lagrange remainder gives, per candidate policy pi,
    |J_theta(pi) - J_g(pi)| <= sum_k |dJ_g(pi)/dv_k| * w_k(g) + 1/2 * w(g)^T Hbar(pi) w(g),
where w_k(g) = max(g_k - lo_k, hi_k - g_k) is the per-coordinate half-width of C(g) measured from g and
Hbar(pi)_kl >= sup_{theta in class box} |d^2 J_theta(pi) / dv_k dv_l|.
  * gradients are exact autograd through the exact propagator (GPU float64), at every grid point;
  * Hbar is numerical (methodology 2.2): the max of |central-difference Hessian of the autograd gradient| over the
    class-box vertices sampled at random, random interior points and all f=1 grid points of the sampled set,
    multiplied by a safety factor (default 1.5). It is not a certified interval bound; validity is checked
    empirically (tests/test_eta_loc.py, g0_resolution_gate: random theta in random cells, 0 violations required).
  * the vertex/centre gradient scheme of the methodology draft needs 2^12 vertices per cell; the expansion around
    g with a Hessian slack is the same "gradient bound + Hessian slack" construction at O(1) cost per cell.
eta_loc depends only on the model class (grid, box), the problem (policies, s0, H, utility weights) and the
public propagator. It never sees data or ground truth, so it is precomputed and locked before evaluation.
Values are returned for the RAW utility (c_q = 1); the normalised bound is c_q * eta_raw.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..models.grid_ladder import BOX, _box_axes, class_vectors, half_widths
from ..models.nl_class import NLClass


def torch_params_from_vectors(V: torch.Tensor, L: int, R: int, nb: int) -> dict:
    """Differentiable (n, d) vectors -> NLPropagator.tables params."""
    n = V.shape[0]
    psi = V[:, 3 * L + 2 * R]
    cols = [torch.zeros(n, L, dtype=V.dtype, device=V.device), psi[:, None].expand(n, L)]
    cols += [torch.zeros(n, L, dtype=V.dtype, device=V.device)] * (nb - 2)
    return {"alpha": V[:, 0:L], "gamma": V[:, L:2 * L], "tauL": V[:, 2 * L:3 * L], "beta": V[:, 3 * L:3 * L + R],
            "tauR": V[:, 3 * L + R:3 * L + 2 * R], "psi_left": torch.stack(cols, -1), "lam": V[:, 3 * L + 2 * R + 1]}


def build_plans(prop, problem):
    return [prop.build_policy_plan(pi, problem.loads0, problem.engaged0, problem.H) for pi in problem.policies]


def j_values(prop, plans, V: np.ndarray, w, w_ret, chunk: int = 16384) -> np.ndarray:
    """Exact raw J (n, K) at arbitrary continuous vectors."""
    out = np.zeros((V.shape[0], len(plans)))
    with torch.no_grad():
        for s in range(0, V.shape[0], chunk):
            Vt = torch.as_tensor(V[s:s + chunk], device=prop.device, dtype=prop.dtype)
            LT, EY = prop.tables(torch_params_from_vectors(Vt, prop.L, prop.R, prop.nb))
            for k, (pl, es) in enumerate(plans):
                out[s:s + chunk, k] = prop._run_plan(pl, es, LT, EY, w, w_ret, 1.0).cpu().numpy()
    return out


def j_and_grad(prop, plans, V: np.ndarray, w, w_ret, chunk: int = 8192):
    """Exact raw J (n, K) and dJ/dv (n, K, d) by autograd."""
    n, d = V.shape
    J = np.zeros((n, len(plans)))
    G = np.zeros((n, len(plans), d))
    for s in range(0, n, chunk):
        Vt = torch.as_tensor(V[s:s + chunk], device=prop.device, dtype=prop.dtype).requires_grad_(True)
        LT, EY = prop.tables(torch_params_from_vectors(Vt, prop.L, prop.R, prop.nb))
        for k, (pl, es) in enumerate(plans):
            Jk = prop._run_plan(pl, es, LT, EY, w, w_ret, 1.0)
            (g,) = torch.autograd.grad(Jk.sum(), Vt, retain_graph=k < len(plans) - 1)
            J[s:s + chunk, k] = Jk.detach().cpu().numpy()
            G[s:s + chunk, k] = g.cpu().numpy()
    return J, G


def hessian_bound(prop, plans, grid, w, w_ret, n_vertex: int = 256, n_interior: int = 256, extra=None,
                  h: float = 1e-4, safety: float = 1.5, seed: int = 0) -> tuple:
    """Hbar (K, d, d): safety * max over sample points of |central-difference Hessian| (symmetrised)."""
    rng = np.random.default_rng(seed)
    bx = np.array(_box_axes(grid), float)                    # (d, 2)
    d = bx.shape[0]
    lo, hi = bx[:, 0], bx[:, 1]
    verts = np.where(rng.random((n_vertex, d)) < 0.5, lo, hi)
    inter = lo + (hi - lo) * rng.random((n_interior, d))
    pts = np.vstack([verts, inter] + ([extra] if extra is not None else []))
    n = pts.shape[0]
    E = np.eye(d) * h
    Vp = np.concatenate([pts[:, None, :] + E[None], pts[:, None, :] - E[None]], 1).reshape(-1, d)
    _, G = j_and_grad(prop, plans, Vp, w, w_ret)
    G = G.reshape(n, 2, d, len(plans), d)                    # (n, +/-, perturbed coord l, K, grad coord k)
    Hs = (G[:, 0] - G[:, 1]) / (2 * h)                       # (n, l, K, k)
    Hs = np.abs(Hs).max(0)                                   # (l, K, k)
    Hs = np.transpose(Hs, (1, 2, 0))                         # (K, k, l)
    Hs = np.maximum(Hs, np.transpose(Hs, (0, 2, 1)))
    return safety * Hs, {"n_points": int(n), "h": h, "safety": safety}


def eta_loc(prop, ncl: NLClass, problem, hbar=None, hess_kw=None, chunk: int = 8192, return_parts: bool = False):
    """Raw eta_loc (B,) for every grid point of `ncl` on `problem` (multiply by c_q to normalise)."""
    t0 = time.perf_counter()
    plans = build_plans(prop, problem)
    w, w_ret = problem.utility.w, problem.utility.w_ret
    V = class_vectors(ncl)
    W = half_widths(ncl)                                     # (B, d)
    if hbar is None:
        hbar, hinfo = hessian_bound(prop, plans, ncl.grid, w, w_ret, **(hess_kw or {}))
    else:
        hinfo = {"given": True}
    t1 = time.perf_counter()
    B = V.shape[0]
    eta = np.zeros(B)
    J = np.zeros((B, len(plans)))
    parts = {"grad_term_by_coord": np.zeros(V.shape[1]), "hess_term_max": 0.0, "grad_norm_max": 0.0} \
        if return_parts else None
    for s in range(0, B, chunk):
        Jc, Gc = j_and_grad(prop, plans, V[s:s + chunk], w, w_ret, chunk=chunk)
        Wc = W[s:s + chunk]
        gterm = np.abs(Gc) * Wc[:, None, :]                  # (b, K, d)
        hterm = 0.5 * np.einsum("bd,kde,be->bk", Wc, hbar, Wc)
        tot = gterm.sum(-1) + hterm                          # (b, K)
        eta[s:s + chunk] = tot.max(1)
        J[s:s + chunk] = Jc
        if return_parts:
            kk = tot.argmax(1)
            parts["grad_term_by_coord"] += gterm[np.arange(len(kk)), kk].sum(0)
            parts["hess_term_max"] = max(parts["hess_term_max"], float(hterm.max()))
            parts["grad_norm_max"] = max(parts["grad_norm_max"], float(np.linalg.norm(Gc, axis=-1).max()))
    t2 = time.perf_counter()
    info = {"hess": hinfo, "hess_s": t1 - t0, "grad_s": t2 - t1, "B": int(B)}
    if return_parts:
        parts["grad_term_by_coord"] = (parts["grad_term_by_coord"] / B).tolist()
        # Lipschitz constant of J on the class box (for h*): sampled max gradient norm + Hessian slack over a cell
        wmax = W.max(0)
        parts["L_J"] = parts["grad_norm_max"] + float(max(np.linalg.norm(hk, 2) for hk in hbar) * np.linalg.norm(wmax))
        info["parts"] = parts
    return eta, J, hbar, info


def certify_condition(r_bar_grid: float, eta_alive: np.ndarray, eps: float) -> bool:
    """R_bar_grid(Theta_t) + 2 max_{g in Theta_t} eta_loc(g) <= eps."""
    return bool(r_bar_grid + 2.0 * float(np.max(eta_alive)) <= eps)


# ------------------------------------------------------------------------------------------------ regret-level variant
def hessian_bound_diff(prop, plans, grid, w, w_ret, n_vertex: int = 256, n_interior: int = 256, h: float = 1e-4,
                       safety: float = 1.5, seed: int = 0):
    """Hbar_diff (K, K, d, d) >= sup |Hessian of J(pi_a) - J(pi_b)| (same sampling scheme as hessian_bound)."""
    rng = np.random.default_rng(seed)
    bx = np.array(_box_axes(grid), float)
    d = bx.shape[0]
    lo, hi = bx[:, 0], bx[:, 1]
    pts = np.vstack([np.where(rng.random((n_vertex, d)) < 0.5, lo, hi), lo + (hi - lo) * rng.random((n_interior, d))])
    n = pts.shape[0]
    E = np.eye(d) * h
    Vp = np.concatenate([pts[:, None, :] + E[None], pts[:, None, :] - E[None]], 1).reshape(-1, d)
    _, G = j_and_grad(prop, plans, Vp, w, w_ret)
    G = G.reshape(n, 2, d, len(plans), d)
    Hs = np.transpose((G[:, 0] - G[:, 1]) / (2 * h), (0, 2, 3, 1))     # (n, K, k, l)
    Hs = 0.5 * (Hs + np.transpose(Hs, (0, 1, 3, 2)))
    Hd = np.abs(Hs[:, :, None] - Hs[:, None, :]).max(0)               # (K, K, d, d)
    return safety * Hd, {"n_points": int(n), "h": h, "safety": safety}


def eta_loc_diff(prop, ncl: NLClass, problem, chunk: int = 4096, hess_kw=None):
    """Raw regret-level bound (B,):  eta_diff(g) >= max_{a,b} sup_{theta in C(g)} |D_ab(theta) - D_ab(g)|,
    D_ab = J(pi_a) - J(pi_b). Every regret regret_theta(pi) = max_a D_a,pi(theta) then moves by <= eta_diff(g), so the
    certification condition becomes  R_bar_grid + max_{g in Theta_t} eta_diff(g) <= eps  (factor 1, not 2).
    Common-mode value shifts (e.g. a coordinate that moves every policy's J alike) cancel in D_ab.
    NOT the locked methodology definition; reported as a diagnostic / candidate for r2_prereg_lock."""
    plans = build_plans(prop, problem)
    w, w_ret = problem.utility.w, problem.utility.w_ret
    hd, _ = hessian_bound_diff(prop, plans, ncl.grid, w, w_ret, **(hess_kw or {}))
    V = class_vectors(ncl)
    W = half_widths(ncl)
    B = V.shape[0]
    eta = np.zeros(B)
    J = np.zeros((B, len(plans)))
    for s in range(0, B, chunk):
        Jc, Gc = j_and_grad(prop, plans, V[s:s + chunk], w, w_ret, chunk=chunk)
        Wc = W[s:s + chunk]
        Gd = np.abs(Gc[:, :, None, :] - Gc[:, None, :, :])               # (b, K, K, d)
        gterm = (Gd * Wc[:, None, None, :]).sum(-1)
        hterm = 0.5 * np.einsum("bd,xyde,be->bxy", Wc, hd, Wc)
        eta[s:s + chunk] = (gterm + hterm).reshape(len(Wc), -1).max(1)
        J[s:s + chunk] = Jc
    return eta, J


__all__ = ["eta_loc", "hessian_bound", "j_values", "j_and_grad", "build_plans", "torch_params_from_vectors",
           "certify_condition", "BOX", "eta_loc_diff", "hessian_bound_diff"]
