"""Design-orthogonal leverage (learner side, truth-free) -- round 3 RCA-OL mechanism layer.

Notation (plan/methodology.md s.4). A *cell* x is one Bernoulli factor of the E1-NL likelihood:
  pair cell      (i, j, n_i, b)   y_ij ~ Bern(mu),  logit mu = alpha_i + beta_j - gamma_i n_i + psi_b
  retention cell (p, y_p, n_p)    e_p' ~ Bern(mu),  logit mu = tau_p - lam n_p + c y_p      (c known: offset)
(exactly the factor indexing of exact/nl_propagate.py: obs_factors / tables). Canonical link: logit mu = phi^T theta
+ offset, so the model tangent at any theta is span phi (the columns of `Phi`).

  contrast leverage   h_theta(x) = dJ_theta(pi)/dmu(x) - dJ_theta(pi')/dmu(x)  ( = d^pi(x) dV^pi(x) - (same for pi') )
                      d (expected number of draws of cell x) from a forward pass, dV = h/d; h by reverse-mode
                      autodiff through the exact propagator re-parametrised by the per-cell means mu (float64).
  design              xi_hat(x) = (# factor draws of x in the ledger) / (total # factor draws)
  orthogonal leverage Lambda_perp = || (h/xi)^perp ||_{L2(xi)}, perp = residual of the xi-weighted least-squares
                      projection onto span Phi (pseudo-inverse with spectral floor s^2 > ridge * s_max^2, ridge=1e-8,
                      so the projector is exactly idempotent). Cells with xi = 0 and h != 0 -> NEED_DATA.
                      Optional reference tangent Phi_ref (Lin layer, where the truth class is known to be linear):
                      Lambda_perp = || (I - P_model) P_ref (h/xi) ||_xi  (= 0 exactly when model = ref).
  break-even budget   rho* = (eps - R_bar) / Lambda_perp ;  S2 = 1/rho* = Lambda_perp / (eps - R_bar) (+inf if R_bar>=eps)
  variance side       u = d(J(pi) - J(pi'))/dtheta = Phi^T (h * mu(1-mu)),  I_t = Phi^T diag(n_x mu(1-mu)) Phi,
                      A_k = ||P_k u|| / ||u|| (P_k: top-k eigenvectors of I_t, k=3), w_k = sqrt(u^T I_t^+ u);
                      placebo: u permuted inside parameter blocks (fixed seed).
  remainder bound     Rem = [J(mu*) - J(mu_hat)] - <h, r>. For every r with |r(x)| <= a(x) (a from the cell box
                      theta_hat +- half grid step, interval arithmetic through the logistic link) a backward
                      (per-step telescoping / simulation-lemma) recursion with per-row interval products bounds |Rem|
                      rigorously (rem_bound); a cruder global path-expansion bound is reported as rem_bound_path.

Truth isolation: this module never imports envs / streams.generator / offgrid / lin_oos / mechanism.oracle_check and
never reads truth-only attributes; it consumes only the model class, ledger observations and public Problem fields.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..core.dynamics import next_loads

RIDGE = 1e-8
K_STIFF = 3
PLACEBO_SEED = 20261002


# =============================================================================================== projection
def _wproj(Phi: np.ndarray, w: np.ndarray, v: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """L2(w) projection of v onto span(Phi) (rows = support cells)."""
    if Phi.shape[1] == 0 or len(v) == 0:
        return np.zeros_like(v)
    sw = np.sqrt(w)
    A = sw[:, None] * Phi
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    if S.size == 0 or S.max() <= 0:
        return np.zeros_like(v)
    keep = S ** 2 > ridge * S.max() ** 2
    Uk = U[:, keep]
    return (Uk @ (Uk.T @ (sw * v))) / sw


def orth_leverage(h: np.ndarray, xi: np.ndarray, Phi_model: np.ndarray, Phi_ref: np.ndarray | None = None,
                  ridge: float = RIDGE, tol: float = 1e-12) -> dict:
    """Lambda_perp, ||h/xi||_xi, phi_perp and the NEED_DATA cells for one contrast leverage h on design xi."""
    h = np.asarray(h, float)
    xi = np.asarray(xi, float)
    hmax = float(np.abs(h).max()) if h.size else 0.0
    supp = xi > 0
    need = np.flatnonzero((~supp) & (np.abs(h) > tol * max(hmax, 1e-300)))
    w = xi[supp]
    v = h[supp] / w
    norm_v = float(np.sqrt(np.sum(w * v * v)))
    vv = v if Phi_ref is None else _wproj(Phi_ref[supp], w, v, ridge)
    res = vv - _wproj(Phi_model[supp], w, vv, ridge)
    lam = float(np.sqrt(np.sum(w * res * res)))
    resid = np.zeros_like(h)
    resid[supp] = res
    return {"lambda_perp": lam, "norm_h_over_xi": norm_v, "phi_perp": (lam / norm_v) if norm_v > 0 else 0.0,
            "need_data": [int(x) for x in need], "h_mass_need_data": float(np.abs(h[need]).sum()),
            "n_support": int(supp.sum()), "resid": resid}


def rho_star(eps: float, r_bar: float, lam: float) -> dict:
    """Break-even misspecification budget rho* = (eps - R_bar)/Lambda_perp and the S2 variant 1/rho*."""
    margin = float(eps) - float(r_bar)
    rho = math.inf if lam <= 0 else margin / lam
    s2 = math.inf if margin <= 0 else lam / margin
    return {"rho_star": rho, "S2": s2, "margin": margin}


def stiff_alignment(u: np.ndarray, I: np.ndarray, k: int = K_STIFF, blocks: dict | None = None,
                    seed: int = PLACEBO_SEED) -> dict:
    """A_k = ||P_k u||/||u|| on the top-k stiff eigenspace of I, width w_k = sqrt(u^T I^+ u), block placebo."""
    u = np.asarray(u, float)
    I = 0.5 * (np.asarray(I, float) + np.asarray(I, float).T)
    nu = float(np.linalg.norm(u))
    ev, V = np.linalg.eigh(I)
    top = V[:, -k:] if I.shape[0] >= k else V

    def align(x):
        n = float(np.linalg.norm(x))
        return float(np.linalg.norm(top.T @ x) / n) if n > 0 else 0.0

    Ip = np.linalg.pinv(I, rcond=1e-10, hermitian=True)
    w = float(np.sqrt(max(float(u @ Ip @ u), 0.0)))
    rng = np.random.default_rng(seed)
    up = u.copy()
    for idx in (blocks or {}).values():
        idx = np.asarray(idx, int)
        if len(idx) > 1:
            up[idx] = u[idx][rng.permutation(len(idx))]
    ug = u[np.random.default_rng(seed + 1).permutation(len(u))]
    return {"A_k": align(u) if nu > 0 else 0.0, "w_k": w, "A_k_placebo_block": align(up),
            "A_k_placebo_global": align(ug), "u_norm": nu, "trace_I_pinv": float(np.trace(Ip)),
            "top_eig": [float(x) for x in ev[-k:]]}


# =============================================================================================== E1-NL
class NLLeverage:
    """Mu-parametrised exact propagator + leverage / Lambda_perp / A_k / Rem-bound for one finite NL class.

    prop     : exact.nl_propagate.NLPropagator (its device is used; float64)
    static   : True for the load-free static twin (g = g_ret = 1 reparametrisation: tangent drops gamma, lam)
    theta_grid (B, d) : optional class matrix in this module's theta layout (for grid half-widths / theta_hat)
    """

    def __init__(self, prop, static: bool = False, nb_model: int | None = None, theta_grid: np.ndarray | None = None,
                 eps: float = 0.02):
        self.prop = prop
        self.static = bool(static)
        self.L, self.R, self.P, self.N = prop.L, prop.R, prop.P, prop.N
        self.nb = prop.nb if nb_model is None else int(nb_model)
        self.n_pair = prop.U_pair
        self.n_re = self.P * 2 * self.N
        self.C = self.n_pair + self.n_re
        self.eps = float(eps)
        self.dev, self.dt = prop.device, prop.dtype
        self._build_features()
        self.theta_grid = theta_grid
        self.halfwidth = grid_halfwidths(theta_grid) if theta_grid is not None else None
        self._plans: dict = {}

    # ---------------- parameter layout and features ----------------
    def _build_features(self):
        L, R, P, N, nb = self.L, self.R, self.P, self.N, self.nb
        names, blocks = [], {}

        def add(block, k):
            blocks[block] = list(range(len(names), len(names) + k))
            names.extend(f"{block}{i}" for i in range(k))

        add("alpha", L)
        add("beta", R)
        if not self.static:
            add("gamma", L)
        add("psi", nb - 1)
        add("tauL", L)
        add("tauR", R)
        if not self.static:
            add("lam", 1)
        self.names, self.blocks, self.d = names, blocks, len(names)
        Phi = np.zeros((self.C, self.d))
        off = np.zeros(self.C)
        cell_info = []
        for i in range(L):
            for j in range(R):
                for n in range(N):
                    for b in range(nb):
                        x = ((i * R + j) * N + n) * nb + b
                        Phi[x, blocks["alpha"][i]] = 1.0
                        Phi[x, blocks["beta"][j]] = 1.0
                        if not self.static:
                            Phi[x, blocks["gamma"][i]] = -float(n)
                        if b >= 1:
                            Phi[x, blocks["psi"][b - 1]] = 1.0
        for p in range(P):
            for y in range(2):
                for n in range(N):
                    x = self.n_pair + (p * 2 + y) * N + n
                    tb = blocks["tauL"][p] if p < L else blocks["tauR"][p - L]
                    Phi[x, tb] = 1.0
                    if not self.static:
                        Phi[x, blocks["lam"][0]] = -float(n)
                    off[x] = self.prop.c * y
        for i in range(L):
            for j in range(R):
                for n in range(N):
                    for b in range(nb):
                        cell_info.append(("pair", i, j, n, b))
        for p in range(P):
            for y in range(2):
                for n in range(N):
                    cell_info.append(("ret", p, y, n))
        self.Phi, self.offset, self.cell_info = Phi, off, cell_info

    def theta_vector(self, p: dict) -> np.ndarray:
        """Single-theta parameter dict (class np_params row; static rows already reparametrised) -> theta vector."""
        parts = [np.asarray(p["alpha"], float).reshape(-1), np.asarray(p["beta"], float).reshape(-1)]
        if not self.static:
            parts.append(np.asarray(p["gamma"], float).reshape(-1))
        parts.append(np.asarray(p["psi_left"], float)[0, 1:self.nb].reshape(-1))
        parts += [np.asarray(p["tauL"], float).reshape(-1), np.asarray(p["tauR"], float).reshape(-1)]
        if not self.static:
            parts.append(np.asarray([p["lam"]], float).reshape(-1))
        return np.concatenate(parts)

    def theta_matrix(self, np_params: dict) -> np.ndarray:
        """Whole class (B, d) in this layout (vectorised theta_vector)."""
        B = np.asarray(np_params["alpha"]).shape[0]
        parts = [np_params["alpha"], np_params["beta"]]
        if not self.static:
            parts.append(np_params["gamma"])
        parts.append(np.asarray(np_params["psi_left"])[:, 0, 1:self.nb])
        parts += [np_params["tauL"], np_params["tauR"]]
        if not self.static:
            parts.append(np.asarray(np_params["lam"]).reshape(B, 1))
        return np.concatenate([np.asarray(x, float).reshape(B, -1) for x in parts], 1)

    def logits(self, theta: np.ndarray) -> np.ndarray:
        return self.Phi @ np.asarray(theta, float) + self.offset

    def mu_from_theta(self, theta: np.ndarray) -> np.ndarray:
        z = self.logits(theta)
        return 1.0 / (1.0 + np.exp(-z))

    def mu_box(self, theta: np.ndarray, halfwidth: np.ndarray | None = None):
        """Interval arithmetic: theta in theta_hat +- halfwidth  ->  mu(x) in [lo, hi]; a = max deviation from mu_hat."""
        hw = self.halfwidth if halfwidth is None else np.asarray(halfwidth, float)
        z = self.logits(theta)
        rad = np.abs(self.Phi) @ hw
        sig = lambda t: 1.0 / (1.0 + np.exp(-t))  # noqa: E731
        mu, lo, hi = sig(z), sig(z - rad), sig(z + rad)
        return mu, lo, hi, np.maximum(hi - mu, mu - lo)

    # ---------------- design from the ledger ----------------
    def cell_of_factor(self, idx: int) -> int:
        pr = self.prop
        if idx < pr.U_pair:
            return idx
        if idx < 2 * pr.U_pair:
            return idx - pr.U_pair
        if idx < pr.off_grp:
            return self.n_pair + (idx - pr.off_re) // 2
        raise ValueError("grouped factor index in an observation")

    def cell_counts(self, observations) -> np.ndarray:
        cnt = np.zeros(self.C)
        for o in observations:
            idx, _ = self.prop.obs_factors(o)
            for f in idx:
                cnt[self.cell_of_factor(int(f))] += 1.0
        return cnt

    # ---------------- mu-space tables and propagation ----------------
    def lt_from_mu(self, mu: torch.Tensor, a: torch.Tensor | None = None, s=None, occ: torch.Tensor | None = None):
        """(B, C) means -> LT (B, U+1), EY (B, U_pair+1), the layout of NLPropagator.tables.
        a, s: per-cell interval radius and scalar s (q -> q + s a on BOTH outcomes; used by the Rem bounds).
        occ : per-cell additive log-weight on both outcomes (occupancy counting via d mass / d occ)."""
        L, R, P, N, nb = self.L, self.R, self.P, self.N, self.nb
        mu = mu.reshape(-1, self.C)
        B = mu.shape[0]
        mp = mu[:, :self.n_pair].reshape(B, L, R, N, nb)
        mr = mu[:, self.n_pair:].reshape(B, P, 2, N)
        q1p, q0p, q1r, q0r = mp, 1.0 - mp, mr, 1.0 - mr
        if a is not None:
            a = a.reshape(-1, self.C)
            ap = a[:, :self.n_pair].reshape(-1, L, R, N, nb)
            ar = a[:, self.n_pair:].reshape(-1, P, 2, N)
            q1p, q0p, q1r, q0r = q1p + s * ap, q0p + s * ap, q1r + s * ar, q0r + s * ar
        lpy1, lpy0 = torch.log(q1p), torch.log(q0p)
        lr1, lr0 = torch.log(q1r), torch.log(q0r)
        if occ is not None:
            occ = occ.reshape(-1, self.C)
            op = occ[:, :self.n_pair].reshape(-1, L, R, N, nb)
            orr = occ[:, self.n_pair:].reshape(-1, P, 2, N)
            lpy1, lpy0, lr1, lr0 = lpy1 + op, lpy0 + op, lr1 + orr, lr0 + orr
        lre = torch.stack([lr0, lr1], -1)                                   # (B, P, y, N, e)
        lpy = torch.stack([lpy0, lpy1], -1)                                 # (B, L, R, N, nb, y)
        A = lpy[:, :, :, :, None, :, :, None, None]
        Ei = lre[:, :L].permute(0, 1, 3, 2, 4)[:, :, None, :, None, None, :, :, None]
        Ej = lre[:, L:].permute(0, 1, 3, 2, 4)[:, None, :, None, :, None, :, None, :]
        G = torch.logsumexp(A + Ei + Ej, dim=6)
        z = torch.zeros(B, 1, device=mu.device, dtype=mu.dtype)
        LT = torch.cat([lpy1.reshape(B, -1), lpy0.reshape(B, -1), lre.reshape(B, -1), G.reshape(B, -1), z], 1)
        EY = torch.cat([q1p.reshape(B, -1), z], 1)
        return LT, EY

    def plan(self, problem, k: int):
        key = (problem.pid, id(problem), k)
        p = self._plans.get(key)
        if p is None:
            p = self.prop.build_policy_plan(problem.policies[k], problem.loads0, problem.engaged0, problem.H)
            self._plans[key] = p
        return p

    @staticmethod
    def _run(plan, esum, LT, EY, w, w_ret, c_q):
        """Forward pass -> (value (B,), total mass (B,))."""
        b = LT.shape[0]
        Pt = torch.ones(b, 1, device=LT.device, dtype=LT.dtype)
        val = torch.zeros(b, device=LT.device, dtype=LT.dtype)
        for t, st in enumerate(plan):
            ey = EY[:, st["eyidx"]].sum(-1)
            val = val + float(w[t]) * (Pt * ey).sum(1)
            logp = LT[:, st["fidx"]].sum(-1) + st["const"][None]
            W = Pt[:, st["src"]] * torch.exp(logp)
            Pn = torch.zeros(b, st["n_next"], device=LT.device, dtype=LT.dtype)
            Pn = Pn.index_add(1, st["dst"], W)
            Pt = Pn
        val = val + float(w_ret) * (Pt * esum[None]).sum(1)
        return float(c_q) * val, Pt.sum(1)

    def J_mu(self, mu, problem, ks=None) -> np.ndarray:
        mu_t = torch.as_tensor(np.asarray(mu, float), device=self.dev, dtype=self.dt).reshape(-1, self.C)
        LT, EY = self.lt_from_mu(mu_t)
        u = problem.utility
        ks = range(len(problem.policies)) if ks is None else ks
        out = []
        for k in ks:
            plan, esum = self.plan(problem, k)
            out.append(self._run(plan, esum, LT, EY, u.w, u.w_ret, u.c_q)[0].detach().cpu().numpy())
        return np.stack(out, -1)

    def policy_leverage(self, mu: np.ndarray, problem, k: int) -> dict:
        """J, dJ/dmu (C,) and occupancy d (C,) of policy k at mu (one forward + two reverse passes)."""
        mu_t = torch.as_tensor(np.asarray(mu, float), device=self.dev, dtype=self.dt).reshape(1, self.C)
        mu_t.requires_grad_(True)
        occ = torch.zeros(1, self.C, device=self.dev, dtype=self.dt, requires_grad=True)
        LT, EY = self.lt_from_mu(mu_t, occ=occ)
        plan, esum = self.plan(problem, k)
        u = problem.utility
        val, mass = self._run(plan, esum, LT, EY, u.w, u.w_ret, u.c_q)
        g_mu, = torch.autograd.grad(val.sum(), mu_t, retain_graph=True)
        g_occ, = torch.autograd.grad(mass.sum(), occ)
        return {"J": float(val.item()), "grad": g_mu.reshape(-1).cpu().numpy(),
                "occ": g_occ.reshape(-1).cpu().numpy()}

    def contrast_leverage(self, mu: np.ndarray, problem, k1: int, k2: int) -> dict:
        a, b = self.policy_leverage(mu, problem, k1), self.policy_leverage(mu, problem, k2)
        h = a["grad"] - b["grad"]
        with np.errstate(divide="ignore", invalid="ignore"):
            dV1 = np.where(a["occ"] > 0, a["grad"] / a["occ"], 0.0)
            dV2 = np.where(b["occ"] > 0, b["grad"] / b["occ"], 0.0)
        return {"h": h, "dJ": a["J"] - b["J"], "J1": a["J"], "J2": b["J"], "d1": a["occ"], "d2": b["occ"],
                "dV1": dV1, "dV2": dV2}

    # ---------------- remainder bounds ----------------
    def rem_bound(self, mu: np.ndarray, a: np.ndarray, problem, k: int) -> dict:
        """Rigorous bound on |J(mu*) - J(mu) - <dJ/dmu, mu* - mu>| for all mu* with |mu* - mu| <= a (cellwise).

        Backward recursion over the reachable-state plan (V: model value, E >= |V* - V|, Rbar >= |Rem|):
          V_t = rew_t + sum_rows p V_{t+1}
          E_t = |drew| + sum_rows U1 |V_{t+1} - m| + sum_rows (p + U1) E_{t+1}
          R_t = sum_rows U2 |V_{t+1} - m| + sum_rows U1 E_{t+1} + sum_rows p R_{t+1}
        with the distribution clamps sum p* E <= max E, |sum dp V| <= range(V), sum |dp| <= 2, E <= value range,
        and R_t <= E_t + Lb_t (Lb >= |first-order term|, propagated with |f'(0)|).
        per row (product of interval factors, polynomial in s with non-negative coefficients): p = f(0),
        U1 = f(1) - f(0) >= |dp|, U2 = f(1) - f(0) - f'(0) >= |dp - dp_lin|; m = mid-range of V_{t+1} over the rows of
        a source state (rows of a state sum to one under every mu, so constants can be subtracted)."""
        dev, dt = self.dev, self.dt
        mu_t = torch.as_tensor(np.asarray(mu, float), device=dev, dtype=dt).reshape(1, self.C)
        a_t = torch.as_tensor(np.asarray(a, float), device=dev, dtype=dt).reshape(1, self.C)
        LT0, EY0 = self.lt_from_mu(mu_t)
        LT1, _ = self.lt_from_mu(mu_t, a=a_t, s=1.0)
        s0 = torch.zeros((), device=dev, dtype=dt)
        _, dLT = torch.func.jvp(lambda s: self.lt_from_mu(mu_t, a=a_t, s=s)[0], (s0,), (torch.ones_like(s0),))
        EYa = torch.cat([a_t[:, :self.n_pair], torch.zeros(1, 1, device=dev, dtype=dt)], 1)
        plan, esum = self.plan(problem, k)
        u = problem.utility
        cq = float(u.c_q)
        V = cq * float(u.w_ret) * esum
        E = torch.zeros_like(V)
        Rb = torch.zeros_like(V)
        Lb = torch.zeros_like(V)
        m = min(self.L, self.R)
        vcap = cq * float(u.w_ret) * self.P          # a-priori bound on |V* - V| (values lie in [0, vcap])
        for t in range(len(plan) - 1, -1, -1):
            vcap = vcap + cq * float(u.w[t]) * m
            st = plan[t]
            fidx, src, dst = st["fidx"], st["src"], st["dst"]
            S = st["eyidx"].shape[0]
            lp0 = LT0[0, fidx].sum(-1) + st["const"]
            p0 = torch.exp(lp0)
            p1 = torch.exp(LT1[0, fidx].sum(-1) + st["const"])
            dp = p0 * dLT[0, fidx].sum(-1)
            U1 = torch.clamp(p1 - p0, min=0.0)
            U2 = torch.clamp(p1 - p0 - dp, min=0.0)
            Vn, En, Rn = V[dst], E[dst], Rb[dst]
            vmax = torch.full((S,), -math.inf, device=dev, dtype=dt).scatter_reduce(0, src, Vn, "amax")
            vmin = torch.full((S,), math.inf, device=dev, dtype=dt).scatter_reduce(0, src, Vn, "amin")
            dev_v = torch.abs(Vn - 0.5 * (vmax + vmin)[src])
            rew = cq * float(u.w[t]) * EY0[0, st["eyidx"]].sum(-1)
            rew_a = cq * float(u.w[t]) * EYa[0, st["eyidx"]].sum(-1)
            z = torch.zeros(S, device=dev, dtype=dt)
            rng_v = (vmax - vmin)
            emax = torch.full((S,), 0.0, device=dev, dtype=dt).scatter_reduce(0, src, En, "amax")
            # |sum dp V| <= min(sum U1 |V - m|, TV * range <= range);  sum p* E <= min(sum (p + U1) E, max E)
            b1 = torch.minimum(z.index_add(0, src, U1 * dev_v), rng_v)
            pe = torch.minimum(z.index_add(0, src, (p0 + U1) * En), emax)
            u1e = torch.minimum(z.index_add(0, src, U1 * En), 2.0 * emax)
            V_new = rew + z.index_add(0, src, p0 * Vn)
            E_new = torch.clamp(rew_a + b1 + pe, max=vcap)
            Lb_new = rew_a + z.index_add(0, src, torch.abs(dp) * dev_v) + z.index_add(0, src, p0 * Lb[dst])
            R_new = z.index_add(0, src, U2 * dev_v) + u1e + z.index_add(0, src, p0 * Rn)
            R_new = torch.minimum(R_new, E_new + Lb_new)    # |Rem| <= |e| + |first-order term|
            V, E, Rb, Lb = V_new, E_new, R_new, Lb_new
        assert V.shape[0] == 1
        return {"rem_bar": float(Rb[0]), "dJ_bar": float(E[0]), "J": float(V[0])}

    def rem_bound_path(self, mu: np.ndarray, a: np.ndarray, problem, k: int) -> float:
        """Crude global path-expansion bound F(1) - F(0) - F'(0), F(s) = sum_paths w prod (q + s a) (also rigorous)."""
        dev, dt = self.dev, self.dt
        mu_t = torch.as_tensor(np.asarray(mu, float), device=dev, dtype=dt).reshape(1, self.C)
        a_t = torch.as_tensor(np.asarray(a, float), device=dev, dtype=dt).reshape(1, self.C)
        plan, esum = self.plan(problem, k)
        u = problem.utility

        def F(s):
            LT, EY = self.lt_from_mu(mu_t, a=a_t, s=s)
            return self._run(plan, esum, LT, EY, u.w, u.w_ret, u.c_q)[0].sum()

        s0 = torch.zeros((), device=dev, dtype=dt)
        f0, df = torch.func.jvp(F, (s0,), (torch.ones_like(s0),))
        f1 = F(torch.ones((), device=dev, dtype=dt))
        return float(max(f1 - f0 - df, 0.0))

    # ---------------- variance side ----------------
    def fisher(self, counts: np.ndarray, mu: np.ndarray) -> np.ndarray:
        v = counts * mu * (1.0 - mu)
        return self.Phi.T @ (v[:, None] * self.Phi)

    def theta_gradient(self, h: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return self.Phi.T @ (h * mu * (1.0 - mu))

    # ---------------- learner-side predictor record ----------------
    def predictors(self, problem, observations, theta_hat: np.ndarray, pairs: list, r_bar: float,
                   eps: float | None = None, with_rem: bool = True, with_path: bool = False) -> dict:
        """All Q-family predictors at a certification opportunity (computed BEFORE any scoring).

        observations : the ledger used by the certificate (data strictly before the certification time)
        theta_hat    : learner theta in this layout (LR-set MLE, ties -> lowest index)
        pairs        : [(label, k_pi_hat, k_challenger), ...]
        """
        t0 = time.perf_counter()
        eps = self.eps if eps is None else float(eps)
        counts = self.cell_counts(observations)
        tot = counts.sum()
        xi = counts / tot if tot > 0 else counts
        mu = self.mu_from_theta(theta_hat)
        I = self.fisher(counts, mu)
        box = self.mu_box(theta_hat) if (with_rem and self.halfwidth is not None) else None
        out_pairs, rem_cache = [], {}
        for label, k1, k2 in pairs:
            cl = self.contrast_leverage(mu, problem, k1, k2)
            ol = orth_leverage(cl["h"], xi, self.Phi)
            rs = rho_star(eps, r_bar, ol["lambda_perp"])
            al = stiff_alignment(self.theta_gradient(cl["h"], mu), I, blocks=self.blocks)
            rec = {"label": label, "k1": int(k1), "k2": int(k2), "dJ_hat": cl["dJ"],
                   "lambda_perp": ol["lambda_perp"], "norm_h_over_xi": ol["norm_h_over_xi"], "phi_perp": ol["phi_perp"],
                   "need_data": ol["need_data"], "h_mass_need_data": ol["h_mass_need_data"],
                   "h_l1": float(np.abs(cl["h"]).sum()), **rs,
                   "A_k": al["A_k"], "w_k": al["w_k"], "A_k_placebo_block": al["A_k_placebo_block"],
                   "A_k_placebo_global": al["A_k_placebo_global"], "u_norm": al["u_norm"]}
            if box is not None:
                rr = []
                for k in (k1, k2):
                    if k not in rem_cache:
                        rem_cache[k] = self.rem_bound(box[0], box[3], problem, k)
                        if with_path:
                            rem_cache[k]["rem_bar_path"] = self.rem_bound_path(box[0], box[3], problem, k)
                    rr.append(rem_cache[k])
                rec["rem_bar"] = rr[0]["rem_bar"] + rr[1]["rem_bar"]
                rec["dJ_bar_box"] = rr[0]["dJ_bar"] + rr[1]["dJ_bar"]
                if with_path:
                    rec["rem_bar_path"] = rr[0]["rem_bar_path"] + rr[1]["rem_bar_path"]
            out_pairs.append(rec)
        prim = max(out_pairs, key=lambda r: r["lambda_perp"]) if out_pairs else None
        tr = float(np.trace(np.linalg.pinv(I, rcond=1e-10, hermitian=True)))
        return {"n_obs": len(observations), "last_serial": int(observations[-1].serial) if observations else None,
                "n_factor_draws": float(tot), "pairs": out_pairs,
                "lambda_perp_max": prim["lambda_perp"] if prim else None,
                "S1": prim["lambda_perp"] if prim else None,
                "S2": max(r["S2"] for r in out_pairs) if out_pairs else None,
                "rho_star_min": min(r["rho_star"] for r in out_pairs) if out_pairs else None,
                "A_k_max": max(r["A_k"] for r in out_pairs) if out_pairs else None,
                "trace_I_pinv": tr, "r_bar": float(r_bar), "eps": eps,
                "sec_per_call": time.perf_counter() - t0}


def grid_halfwidths(theta_grid: np.ndarray) -> np.ndarray:
    """Half of the smallest spacing between distinct grid values, per coordinate (0 for constant coordinates)."""
    hw = np.zeros(theta_grid.shape[1])
    for c in range(theta_grid.shape[1]):
        u = np.unique(np.round(theta_grid[:, c], 12))
        hw[c] = 0.5 * float(np.min(np.diff(u))) if len(u) > 1 else 0.0
    return hw


def theta_hat_index(cum: np.ndarray, mask: np.ndarray) -> int:
    """LR-set maximum-likelihood point (ties -> lowest index)."""
    cum = np.asarray(cum, float)
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        idx = np.arange(len(cum))
    return int(idx[int(np.argmax(cum[idx]))])


def select_pairs(J: np.ndarray, mask: np.ndarray, k_theta_hat: int, pi_hat: int) -> list:
    """Pre-registered contrast pairs: (a) pi_hat vs the runner-up under theta_hat, (b) pi_hat vs the minimax challenger
    (best policy of the confidence-set member with the largest regret of pi_hat). (b) falls back to (a) if degenerate."""
    J = np.asarray(J, float)
    K = J.shape[1]
    row = J[k_theta_hat].copy()
    row[pi_hat] = -np.inf
    ka = int(np.argmax(row)) if K > 1 else pi_hat
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        idx = np.arange(J.shape[0])
    reg = J[idx].max(1) - J[idx, pi_hat]
    th = int(idx[int(np.argmax(reg))])
    kb = int(np.argmax(J[th]))
    if kb == pi_hat:
        kb = ka
    return [("theta_hat_runner_up", pi_hat, ka), ("minimax_challenger", pi_hat, kb)]


# =============================================================================================== E1-Lin
class LinLeverage:
    """Closed-form leverage for T-Lin (identity link, mu = phi^T theta, A1: known load dynamics).

    Cells x = (i, j, n_i, b). h(x) = sum_t c_q w_t [1{pi uses x at t} - 1{pi' uses x at t}] (theta-free).
    model: LinClass (static or dynamic tangent); ref: dynamic LinClass (the truth class of E1-Lin is linear in the
    dynamic features, so Lambda_perp = ||(I - P_model) P_ref (h/xi)||_xi; = 0 for the dynamic class, > 0 for
    same-exposure pairs under the static class -- Corollary C')."""

    def __init__(self, model, ref=None, sigma: float = 1.0, nmax: int | None = None):
        self.model, self.ref = model, ref
        self.L, self.R, self.nb = model.L, model.R, model.nb
        self.nmax = model.nmax if nmax is None else nmax
        self.N = self.nmax + 1
        self.C = self.L * self.R * self.N * self.nb
        self.sigma = float(sigma)
        self.cell_info = [(i, j, n, b) for i in range(self.L) for j in range(self.R) for n in range(self.N)
                          for b in range(self.nb)]
        self.Phi = np.stack([model.feature(*c) for c in self.cell_info])
        self.Phi_ref = None if ref is None else np.stack([ref.feature(*c) for c in self.cell_info])
        sl = model.slices()
        self.blocks = {k: list(range(v.start, v.stop)) for k, v in sl.items() if v.stop > v.start}
        if model.static:
            self.blocks.pop("gamma", None)

    def cell(self, i, j, n, b) -> int:
        return ((i * self.R + j) * self.N + n) * self.nb + b

    def occupancy(self, policy, loads0, H, utility, aspace) -> np.ndarray:
        P = self.L + self.R
        loads = np.asarray(loads0, np.int64).copy()
        ones = np.ones(P, dtype=np.int64)
        d = np.zeros(self.C)
        for t in range(H):
            a_idx = policy.act(t, loads, ones)
            a = aspace.actions[a_idx]
            inc = {(i, j): lv for i, j, lv in a.incentives}
            for i, j in a.pairs:
                d[self.cell(i, j, int(loads[i]), inc.get((i, j), 0))] += utility.c_q * utility.w[t]
            loads = next_loads(loads, aspace, a_idx, self.nmax, False)
        return d

    def contrast_leverage(self, problem, k1, k2, aspace) -> np.ndarray:
        return (self.occupancy(problem.policies[k1], problem.loads0, problem.H, problem.utility, aspace)
                - self.occupancy(problem.policies[k2], problem.loads0, problem.H, problem.utility, aspace))

    def cell_counts(self, observations) -> np.ndarray:
        cnt = np.zeros(self.C)
        for o in observations:
            for i, j, b, _ in o.outcomes:
                cnt[self.cell(i, j, int(o.loads[i]), b)] += 1.0
        return cnt

    def predictors(self, problem, observations, theta_hat, pairs, r_bar, eps, aspace) -> dict:
        t0 = time.perf_counter()
        counts = self.cell_counts(observations)
        tot = counts.sum()
        xi = counts / tot if tot > 0 else counts
        I = self.Phi.T @ (counts[:, None] * self.Phi) / self.sigma ** 2
        out = []
        for label, k1, k2 in pairs:
            h = self.contrast_leverage(problem, k1, k2, aspace)
            ol = orth_leverage(h, xi, self.Phi, self.Phi_ref)
            np_ol = orth_leverage(h, xi, self.Phi)          # nonparametric reference (diagnostic)
            rs = rho_star(eps, r_bar, ol["lambda_perp"])
            al = stiff_alignment(self.Phi.T @ h, I, blocks=self.blocks)
            out.append({"label": label, "k1": int(k1), "k2": int(k2),
                        "dJ_hat": float(h @ (self.Phi @ np.asarray(theta_hat, float))),
                        "lambda_perp": ol["lambda_perp"], "lambda_perp_nonparam": np_ol["lambda_perp"],
                        "phi_perp": ol["phi_perp"], "norm_h_over_xi": ol["norm_h_over_xi"],
                        "need_data": ol["need_data"], **rs, "A_k": al["A_k"], "w_k": al["w_k"],
                        "A_k_placebo_block": al["A_k_placebo_block"], "A_k_placebo_global": al["A_k_placebo_global"],
                        "rem_bar": 0.0})
        return {"n_obs": len(observations), "last_serial": int(observations[-1].serial) if observations else None,
                "pairs": out, "lambda_perp_max": max(r["lambda_perp"] for r in out) if out else None,
                "r_bar": float(r_bar), "eps": float(eps), "sec_per_call": time.perf_counter() - t0}
