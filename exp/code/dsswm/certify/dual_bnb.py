"""Lagrangian weak-dual certifier with participant copies + branch-and-bound on hub individuals.

Primal (per challenger pi'):   U* = max_{theta in Theta_t} Delta(theta),
  Delta(theta) = J_theta(pi') - J_theta(pi_hat) = sum_p Delta_p(theta)   (participant attribution, exact).

Copy relaxation. Give every participant p its own copy theta^(p) in Theta_t and relax the consistency
constraints theta^(p) = theta^(1) on every parameter block b (left/right participant blocks and the global
block) with multipliers nu_{p,b}(.) on the block value, subject to sum_p nu_{p,b} = 0:

  L(nu) = sum_p max_{theta in Theta_t} [ Delta_p(theta) + sum_b nu_{p,b}(theta_b) ]  >=  U*   (weak duality,
  for ANY nu with sum_p nu_{p,b} == 0: put all copies at the primal maximiser).

nu = 0 is the pure participant rectangularisation (sum_p max Delta_p). min_nu L(nu) equals the LP relaxation over
block-marginal-consistent copy distributions; we solve that LP with HiGHS, read the multipliers back, rebuild nu
with the sum-zero constraint holding exactly, and REPORT L(nu) evaluated by direct maximisation. The reported bound
is therefore sound independently of LP solver tolerances.

BnB: fixing the block value of 1-2 hub individuals enforces exact consistency on those blocks; the bound is the
max over leaves of leaf dual bounds (leaves whose cheap rectangular bound is <= the incumbent are pruned with that
bound, which is still a valid upper bound). Incumbents are exact Delta values at feasible theta (lower bounds), used
only for pruning and for the COMPUTE_UNKNOWN rule; never reported as an upper bound.

Learner-side module: no ground-truth access.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .status import Status


def block_digits(radices, B: int) -> np.ndarray:
    """(B, n_blocks) mixed-radix digits of the class index (same order as NLClass)."""
    idx = np.arange(B)
    digits = []
    for r in reversed(radices):
        digits.append(idx % r)
        idx = idx // r
    return np.stack(list(reversed(digits)), 1)


def lagrangian_value(D: np.ndarray, digits: np.ndarray, nu: list) -> tuple[float, np.ndarray]:
    """L(nu) = sum_p max_theta [D_p + sum_b nu[p][b][theta_b]]; returns value and per-copy argmax rows."""
    n, P = D.shape
    tot, arg = 0.0, np.zeros(P, dtype=np.int64)
    for p in range(P):
        v = D[:, p].copy()
        for b in range(digits.shape[1]):
            v += nu[p][b][digits[:, b]]
        k = int(np.argmax(v))
        arg[p] = k
        tot += float(v[k])
    return tot, arg


def _lp_dual(D: np.ndarray, digits: np.ndarray, radices) -> dict:
    """Solve the copy LP; return a sound bound L(nu_hat), LP value, incumbent row."""
    n, P = D.shape
    nb = digits.shape[1]
    if P == 1 or n == 1:
        k = int(np.argmax(D.sum(1)))
        return {"bound": float(D.sum(1)[k]), "lp_value": float(D.sum(1)[k]), "inc_rows": [k], "lp_ok": True,
                "lp_s": 0.0}
    t0 = time.perf_counter()
    rows, cols, vals = [], [], []
    r = 0
    for p in range(P):                         # simplex rows
        rows += [r] * n
        cols += list(range(p * n, (p + 1) * n))
        vals += [1.0] * n
        r += 1
    cons_index = {}                            # (p, b, v) -> row
    for p in range(1, P):
        for b in range(nb):
            for v in np.unique(digits[:, b]):
                sel = np.flatnonzero(digits[:, b] == v)
                rows += [r] * (2 * len(sel))
                cols += list(p * n + sel) + list(sel)
                vals += [1.0] * len(sel) + [-1.0] * len(sel)
                cons_index[(p, b, int(v))] = r
                r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, P * n))
    beq = np.zeros(r)
    beq[:P] = 1.0
    c = -D.T.reshape(-1)                      # x = [q_1 .. q_P]
    res = linprog(c, A_eq=A, b_eq=beq, bounds=(0, None), method="highs")
    lp_s = time.perf_counter() - t0
    if res.status != 0:
        nu0 = [[np.zeros(rd) for rd in radices] for _ in range(P)]
        val, arg = lagrangian_value(D, digits, nu0)
        return {"bound": val, "lp_value": None, "inc_rows": list(arg), "lp_ok": False, "lp_s": lp_s}
    y = res.eqlin.marginals
    best = None
    for sign in (1.0, -1.0):                  # both sign conventions give sum-zero nu -> both sound; keep min
        nu = [[np.zeros(rd) for rd in radices] for _ in range(P)]
        for (p, b, v), row in cons_index.items():
            nu[p][b][v] = sign * y[row]
        for b in range(nb):
            nu[0][b] = -sum(nu[p][b] for p in range(1, P))
        val, arg = lagrangian_value(D, digits, nu)
        if best is None or val < best[0]:
            best = (val, arg)
    # primal support rows as extra incumbents
    x = res.x.reshape(P, n)
    sup_rows = list(np.unique(np.argmax(x, 1)))
    return {"bound": best[0], "lp_value": float(-res.fun), "inc_rows": list(best[1]) + sup_rows, "lp_ok": True,
            "lp_s": lp_s}


def dual_bound(D: np.ndarray, digits: np.ndarray, radices) -> dict:
    out = _lp_dual(D, digits, radices)
    tot = D.sum(1)
    out["incumbent"] = float(max(tot[k] for k in out["inc_rows"]))
    out["rect"] = float(D.max(0).sum())
    return out


def bnb_bound(D: np.ndarray, digits: np.ndarray, radices, hubs, incumbent: float) -> dict:
    """Branch on the block values of hub blocks; returns sound upper bound and stats."""
    tot = D.sum(1)
    keys = digits[:, hubs]
    leaves = {}
    for k, key in enumerate(map(tuple, keys)):
        leaves.setdefault(key, []).append(k)
    inc = incumbent
    ub_leaves, n_lp, n_pruned, lp_s = [], 0, 0, 0.0
    # explore leaves in decreasing rectangular bound (best-first) so the incumbent rises early
    order = sorted(leaves.items(), key=lambda kv: -D[kv[1]].max(0).sum())
    for key, idx in order:
        idx = np.asarray(idx)
        Dl, dl = D[idx], digits[idx]
        rect = float(Dl.max(0).sum())
        if rect <= inc:
            ub_leaves.append(rect)
            n_pruned += 1
            continue
        r = _lp_dual(Dl, dl, radices)
        n_lp += 1
        lp_s += r["lp_s"]
        inc = max(inc, float(max(tot[idx[k]] for k in r["inc_rows"])))
        ub_leaves.append(r["bound"])
    return {"bound": float(max(max(ub_leaves), inc)), "incumbent": inc, "n_leaves": len(leaves), "n_lp": n_lp,
            "n_pruned": n_pruned, "lp_s": lp_s}


def choose_hubs(D: np.ndarray, digits: np.ndarray, P: int, k: int) -> list:
    """Hub individuals = participants with the widest attributed decision-gap range over Theta_t.
    Participant p owns block p (radices order: left..., right..., global)."""
    width = D.max(0) - D.min(0)
    order = [int(p) for p in np.argsort(-width) if len(np.unique(digits[:, p])) > 1]
    return order[:k]


def kappa_participant_set(D: np.ndarray) -> float:
    """Set-based participant cancellation coefficient: sum_p range(Delta_p) / range(Delta) over Theta_t.
    Equals sum_p ||d_p||_{V^-1} / ||d||_{V^-1} exactly when Theta_t is an ellipsoid and Delta is linear."""
    tot = D.sum(1)
    den = float(tot.max() - tot.min())
    if den <= 1e-15:
        return float("nan")
    return float((D.max(0) - D.min(0)).sum() / den)


def certify_dual(C: np.ndarray, mask: np.ndarray, k_hat: int, digits: np.ndarray, radices, eps: float,
                 n_hubs: int = 0, exact: bool = True, bnb_scope: str = "certificate") -> dict:
    """Certificate-level bound R_bar = max_{pi'} U*(pi') with dual (+ optional BnB) per challenger.

    C: (B, K, P) per-participant contributions over the whole class; mask: Theta_t membership.
    Status rule: UB <= eps -> CERTIFIED; incumbent LB > eps -> NEED_DATA; otherwise COMPUTE_UNKNOWN.
    bnb_scope: "certificate" branches only challengers whose dual exceeds the global incumbent (enough for R_bar);
               "challenger" branches every challenger whose dual exceeds its own incumbent (per-challenger tightness).
    """
    t0 = time.perf_counter()
    idx = np.flatnonzero(mask)
    Cm, dg = C[idx], digits[idx]
    K, P = C.shape[1], C.shape[2]
    per = []
    glob_inc = 0.0
    for k in range(K):
        if k == k_hat:
            continue
        D = Cm[:, k, :] - Cm[:, k_hat, :]
        d = dual_bound(D, dg, radices)
        row = {"challenger": k, "dual": d["bound"], "lp_value": d["lp_value"], "rect": d["rect"],
               "incumbent": d["incumbent"], "lp_ok": d["lp_ok"], "lp_s": d["lp_s"],
               "kappa_participant": kappa_participant_set(D)}
        if exact:
            row["exact"] = float(D.sum(1).max())
        glob_inc = max(glob_inc, d["incumbent"])
        per.append((row, D))
    # BnB only on challengers that can still move the certificate
    for row, D in per:
        row["bnb"] = row["dual"]
        row["bnb_hubs"] = []
        ref = glob_inc if bnb_scope == "certificate" else row["incumbent"]
        if n_hubs > 0 and row["dual"] > ref + 1e-12:
            hubs = choose_hubs(D, dg, P, n_hubs)
            if hubs:
                b = bnb_bound(D, dg, radices, hubs, ref)
                row.update({"bnb": min(b["bound"], row["dual"]), "bnb_hubs": hubs, "bnb_n_leaves": b["n_leaves"],
                            "bnb_n_lp": b["n_lp"], "bnb_n_pruned": b["n_pruned"], "bnb_lp_s": b["lp_s"]})
                glob_inc = max(glob_inc, b["incumbent"])
    rows = [r for r, _ in per]
    ub_dual = max([0.0] + [r["dual"] for r in rows])
    ub_bnb = max([0.0] + [r["bnb"] for r in rows])
    ub = ub_bnb if n_hubs > 0 else ub_dual
    lb = max(0.0, glob_inc)
    if ub <= eps:
        status = Status.CERTIFIED
    elif lb > eps:
        status = Status.NEED_DATA
    else:
        status = Status.COMPUTE_UNKNOWN
    return {"status": status, "r_bar_ub": ub, "r_bar_dual": ub_dual, "r_bar_bnb": ub_bnb, "r_bar_lb": lb,
            "per_challenger": rows, "wall_clock_s": time.perf_counter() - t0, "set_size": int(len(idx))}


# ---------------------------------------------------------------- T-Lin (ellipsoid) duals
def lin_mu_dual(d: np.ndarray, theta_hat: np.ndarray, V: np.ndarray, beta: float) -> float:
    """Lagrangian dual of sup <d,theta> s.t. ||theta-theta_hat||_V^2 <= beta, minimised numerically over mu > 0:
    g(mu) = <d,theta_hat> + ||d||^2_{V^-1} / (4 mu) + mu * beta."""
    from scipy.optimize import minimize_scalar
    q = float(d @ np.linalg.solve(V, d))
    base = float(d @ theta_hat)
    if q <= 0:
        return base
    f = lambda lm: base + q / (4 * np.exp(lm)) + np.exp(lm) * beta
    m0 = 0.5 * np.log(q / (4 * beta))
    r = minimize_scalar(f, bracket=(m0 - 3, m0 + 3), method="brent", tol=1e-14)
    return float(r.fun)


def lin_copy_dual(d_parts: np.ndarray, theta_hat: np.ndarray, V: np.ndarray, beta: float) -> dict:
    """Participant-copy dual on an ellipsoid: L(nu) = <d,theta_hat> + sqrt(beta) sum_p ||d_p + nu_p||_{V^-1},
    sum_p nu_p = 0, minimised numerically (BFGS on nu_1..nu_{P-1}). Returns optimised and nu=0 (rectangular) values."""
    from scipy.optimize import minimize
    P, dim = d_parts.shape
    Vi = np.linalg.inv(V)
    Vi = 0.5 * (Vi + Vi.T)
    d = d_parts.sum(0)
    sb = np.sqrt(beta)

    def nrm(x):
        return np.sqrt(max(float(x @ Vi @ x), 0.0))

    def L(z):
        nu = z.reshape(P - 1, dim)
        tot = sum(nrm(d_parts[p] + nu[p]) for p in range(P - 1)) + nrm(d_parts[-1] - nu.sum(0))
        return float(d @ theta_hat + sb * tot)

    def grad(z):
        nu = z.reshape(P - 1, dim)
        last = d_parts[-1] - nu.sum(0)
        nl = nrm(last)
        gl = Vi @ last / nl if nl > 0 else np.zeros(dim)
        g = []
        for p in range(P - 1):
            x = d_parts[p] + nu[p]
            nx = nrm(x)
            gp = Vi @ x / nx if nx > 0 else np.zeros(dim)
            g.append(sb * (gp - gl))
        return np.concatenate(g)

    # start at the analytic optimum's neighbourhood (nu_p = d/P - d_p) perturbed, so the optimiser does real work
    rng = np.random.default_rng(0)
    z0 = np.concatenate([d / P - d_parts[p] for p in range(P - 1)]) + 0.3 * rng.standard_normal((P - 1) * dim) * \
        (np.abs(d).mean() + 1e-12)
    r = minimize(L, z0, jac=grad, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
    zs = np.concatenate([d / P - d_parts[p] for p in range(P - 1)])
    return {"opt_numeric": float(r.fun), "opt_analytic": L(zs), "rect_nu0": L(np.zeros((P - 1) * dim)),
            "closed": float(d @ theta_hat + sb * nrm(d))}
