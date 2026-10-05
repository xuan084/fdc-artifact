"""B3-UI: continuous 12-dim class with the SAME universal-inference threshold as JPC (learner side, optional base).

Isolates the finite-support factor at a fixed threshold (methodology 4.3 / 6.3 "same-threshold control").
Evidence: the UI set on the continuous class box,
    Theta_t^c = {v in box : log q_{1:t} - l_t(v) < log(1/delta)},
with the predictable numerator q of the grid plug-in (any predictable q keeps Ville validity for every v, including
off-grid v*). Using l_t(v) ~ l_t(v_hat) - 1/2 (v - v_hat)^T I_obs (v - v_hat) at the continuous MLE v_hat,
    Theta_t^c ~ {v : ||v - v_hat||_{I_obs} < beta_UI},  beta_UI = sqrt(2 * (log(1/delta) - (log q_{1:t} - l_t(v_hat)))),
and, as in B3, J is linearised at v_hat:  UB(pi') = J_hat(pi') - J_hat(pi_hat) + beta_UI ||g_pi' - g_pi_hat||_{I^-1}.
If log q_{1:t} - l_t(v_hat) >= log(1/delta) the set is empty -> MODEL_CONFLICT. The quadratic / linear
approximations are deliberately favourable to the baseline (ignores remainder terms); B3-UI is not inflated.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..certify.eta_loc import j_and_grad, torch_params_from_vectors
from ..certify.status import Status
from ..models.grid_ladder import _box_axes
from .glm_linearised import certify_linear


def loglik_at(prop, observations, v: torch.Tensor) -> torch.Tensor:
    LT, _ = prop.tables(torch_params_from_vectors(v[None], prop.L, prop.R, prop.nb))
    return prop.loglik(LT, observations).sum()


def continuous_mle(prop, observations, v0: np.ndarray, grid, iters: int = 40, ridge: float = 1e-6):
    """Box-projected damped Newton on the exact log-likelihood. Returns (v_hat, l(v_hat), observed information)."""
    bx = np.array(_box_axes(grid), float)
    lo = torch.as_tensor(bx[:, 0], device=prop.device, dtype=prop.dtype)
    hi = torch.as_tensor(bx[:, 1], device=prop.device, dtype=prop.dtype)
    v = torch.as_tensor(np.asarray(v0, float), device=prop.device, dtype=prop.dtype).clamp(lo, hi)
    f = lambda x: loglik_at(prop, observations, x)  # noqa: E731
    cur = float(f(v))
    for _ in range(iters):
        vg = v.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(f(vg), vg)
        Hm = torch.autograd.functional.hessian(f, v)
        A = -Hm + ridge * torch.eye(len(v), device=v.device, dtype=v.dtype)
        try:
            step = torch.linalg.solve(A, g)
        except RuntimeError:
            step = g
        t, improved = 1.0, False
        while t > 1e-6:
            cand = (v + t * step).clamp(lo, hi)
            val = float(f(cand))
            if val > cur - 1e-12:
                improved = val > cur + 1e-10
                v, cur = cand, max(val, cur)
                break
            t *= 0.5
        if not improved:
            break
    Hm = torch.autograd.functional.hessian(f, v)
    info = (-Hm).cpu().numpy()
    return v.cpu().numpy(), cur, info


def ui_radius(log_num: float, ll_hat: float, delta: float) -> float:
    r2 = 2.0 * (math.log(1.0 / delta) - (log_num - ll_hat))
    return math.sqrt(r2) if r2 > 0 else 0.0


def certify_b3_ui(prop, plans, observations, log_num: float, v0, grid, utility, delta: float, eps: float,
                  ridge: float = 1e-2):
    v_hat, ll_hat, info = continuous_mle(prop, observations, v0, grid)
    beta = ui_radius(log_num, ll_hat, delta)
    if beta <= 0:
        return {"status": Status.MODEL_CONFLICT, "pi": None, "r_bar": float("inf"), "v_hat": v_hat, "beta": 0.0}
    J, G = j_and_grad(prop, plans, v_hat[None], utility.w, utility.w_ret)
    J, G = utility.c_q * J[0], utility.c_q * G[0]
    I = 0.5 * (info + info.T) + ridge * np.eye(len(v_hat))
    w, U = np.linalg.eigh(I)
    Vinv = (U / np.maximum(w, ridge)) @ U.T
    c = certify_linear(J, G, Vinv, beta, eps)
    st = Status.CERTIFIED if c["certified"] else Status.NEED_DATA
    return {"status": st, "pi": c["pi"], "r_bar": c["r_bar"], "v_hat": v_hat, "beta": beta, "ll_hat": ll_hat}
