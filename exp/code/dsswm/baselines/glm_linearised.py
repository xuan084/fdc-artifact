"""B3: structured direct baseline (GLM-BAI / transductive RAGE linearised at theta_hat).

Same per-person per-step observations as JPC, same epsilon stopping rule, but evidence is a Wald/GLM ellipsoid
in the continuous 12-dim parameter vector
    v = (alpha_1..L, gamma_1..L, tauL_1..L, beta_1..R, tauR_1..R, psi, lam)
and values are linearised:  J_theta(pi) ~ J_hat(pi) + g_pi^T (v - v_hat).
UB(pi') = J_hat(pi') - J_hat(pi_hat) + beta * || g_pi' - g_pi_hat ||_{V^{-1}},  V = lam0 I + sum_t F(s_t, a_t; v_hat)
F is the expected Fisher information of one round. beta = sqrt(chi2_{d, 1-delta}) (fixed-n Wald radius: NOT
anytime-valid and ignores linearisation error, i.e. deliberately favourable to B3).
B3g (diagnostic): same ellipsoid intersected with the finite grid, exact J on the grid points inside (no linearisation).
Sampling (both): greedy transductive design - pick the legal action at the current state that minimises the
largest post-update width  max_{d in blocking} ||d||_{(V+F_a)^{-1}}  (B3)  /  the set-based blocking models' Mahalanobis
margins (B3g uses the same width rule on the linearised directions).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2


def theta_vec(p: dict, L: int, R: int) -> np.ndarray:
    return np.concatenate([np.asarray(p["alpha"], float), np.asarray(p["gamma"], float), np.asarray(p["tauL"], float),
                           np.asarray(p["beta"], float), np.asarray(p["tauR"], float),
                           [float(np.asarray(p["psi_left"])[0, 1])], [float(p["lam"])]])


def class_theta_vecs(np_params: dict) -> np.ndarray:
    return np.concatenate([np_params["alpha"], np_params["gamma"], np_params["tauL"], np_params["beta"],
                           np_params["tauR"], np_params["psi_left"][:, 0, 1:2], np.asarray(np_params["lam"])[:, None]], 1)


def _torch_params(v: torch.Tensor, L: int, R: int, nb: int):
    a = v[0:L]; g = v[L:2 * L]; tl = v[2 * L:3 * L]
    b = v[3 * L:3 * L + R]; tr = v[3 * L + R:3 * L + 2 * R]
    psi = v[3 * L + 2 * R]; lam = v[3 * L + 2 * R + 1]
    cols = [torch.zeros(L, dtype=v.dtype), psi.expand(L)] + [torch.zeros(L, dtype=v.dtype)] * (nb - 2)
    psi_left = torch.stack(cols, -1)
    return {"alpha": a[None], "gamma": g[None], "tauL": tl[None], "beta": b[None], "tauR": tr[None],
            "psi_left": psi_left[None], "lam": lam.reshape(1)}


def value_and_grad(prop, plans, v_hat: np.ndarray, utility, L: int, R: int):
    """J (K,) and gradients (K, d) at v_hat, through the exact propagator (autograd, CPU float64)."""
    vt = torch.tensor(v_hat, dtype=torch.float64, requires_grad=True)
    params = _torch_params(vt, L, R, prop.nb)
    LT, EY = prop.tables(params)
    Js, Gs = [], []
    for plan, esum in plans:
        J = prop._run_plan(plan, esum, LT, EY, utility.w, utility.w_ret, utility.c_q)[0]
        (gr,) = torch.autograd.grad(J, vt, retain_graph=True)
        Js.append(float(J.detach())); Gs.append(gr.numpy().copy())
    return np.array(Js), np.stack(Gs)


def fisher_rounds(v: np.ndarray, codec, aspace, code: int, actions, c: float, L: int, R: int) -> np.ndarray:
    """Expected Fisher information (len(actions), d, d) of one round from observable state `code`."""
    d = 3 * L + 2 * R + 2
    loads, eng = codec.decode(int(code))
    P = L + R
    ia, ig, itl = 0, L, 2 * L
    ib, itr, ipsi, ilam = 3 * L, 3 * L + R, 3 * L + 2 * R, 3 * L + 2 * R + 1
    tau = np.concatenate([v[itl:itl + L], v[itr:itr + R]])
    out = np.zeros((len(actions), d, d))

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    for k, a_idx in enumerate(actions):
        a = aspace.actions[a_idx]
        inc = {(i, j): l for i, j, l in a.incentives}
        F = np.zeros((d, d))
        py_of = np.zeros(P)   # P(y_p = 1) for participants in active pairs
        active = np.zeros(P, bool)
        for i, j in a.pairs:
            if eng[i] and eng[L + j]:
                b = inc.get((i, j), 0)
                logit = v[ia + i] + v[ib + j] - v[ig + i] * loads[i] + (v[ipsi] if b == 1 else 0.0)
                p = sig(logit)
                gy = np.zeros(d); gy[ia + i] = 1; gy[ib + j] = 1; gy[ig + i] = -loads[i]
                if b == 1:
                    gy[ipsi] = 1
                F += p * (1 - p) * np.outer(gy, gy)
                py_of[i] = py_of[L + j] = p
                active[i] = active[L + j] = True
        for pp in range(P):
            if not eng[pp]:
                continue
            ge = np.zeros(d)
            ge[(itl + pp) if pp < L else (itr + pp - L)] = 1
            ge[ilam] = -loads[pp]
            q0 = sig(tau[pp] - v[ilam] * loads[pp])
            if active[pp]:
                q1 = sig(tau[pp] + c - v[ilam] * loads[pp])
                w = (1 - py_of[pp]) * q0 * (1 - q0) + py_of[pp] * q1 * (1 - q1)
            else:
                w = q0 * (1 - q0)
            F += w * np.outer(ge, ge)
        out[k] = F
    return out


def wald_beta(d: int, delta: float) -> float:
    return float(np.sqrt(chi2.ppf(1 - delta, d)))


def certify_linear(Jhat: np.ndarray, G: np.ndarray, Vinv: np.ndarray, beta: float, eps: float):
    k_hat = int(np.argmax(Jhat))
    D = G - G[k_hat][None]
    widths = np.sqrt(np.maximum(np.einsum("kd,de,ke->k", D, Vinv, D), 0.0))
    ub = Jhat - Jhat[k_hat] + beta * widths
    ub[k_hat] = 0.0
    r_bar = float(ub.max())
    return {"pi": k_hat, "r_bar": r_bar, "certified": r_bar <= eps, "D": D, "ub": ub}


def choose_design_action(Fa: np.ndarray, V: np.ndarray, D_block: np.ndarray, legal_actions, rng, explore=0.05):
    """Greedy transductive step: minimise max_d ||d||_{(V + F_a)^{-1}} over blocking directions."""
    if rng.random() < explore or len(D_block) == 0:
        return int(rng.choice(legal_actions))
    best, best_val = None, np.inf
    for k, a in enumerate(legal_actions):
        Vi = np.linalg.inv(V + Fa[k])
        val = float(np.max(np.einsum("kd,de,ke->k", D_block, Vi, D_block)))
        if val < best_val - 1e-15:
            best, best_val = a, val
    return int(best)
