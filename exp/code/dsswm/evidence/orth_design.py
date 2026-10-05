"""orth-replay design construction (round 3, P3; methodology section 3). Learner side: no truth access.

Goal. For problem k of a stream, the `orth` arm of the reuse switch receives N_k authentic sibling-platform rows whose
design xi_perp
  (i)  lies in the REACHABLE OCCUPANCY POLYTOPE: the convex hull of the per-step occupancy vectors
       occ_pi(s, a) = (1/H) sum_{t<H} P_pi(s_t = s, a_t = a) of legal deterministic policies started in the current
       public state s0, horizon H = H_k, transitions evaluated at the learner's theta_hat (plug-in);
  (ii) matches the Fisher trace of the `vol` replay data at theta_hat:  |tr I(xi_perp) / tr I_vol - 1| <= 0.05
       (both per row; I(xi) = sum_{s,a} xi(s,a) E_theta_hat[ per-round Fisher | s, a ]);
  (iii) is decision-orthogonal:  A(xi; D_k) <= 0.05 for every pre-registered contrast direction u of problem k,
       A(xi; u) = u^T I(xi) u / (||u||^2 tr I(xi))  (share of the design's Fisher information that lies along the
       unit decision gradient u = d(J(pi_hat) - J(pi')) / d theta at theta_hat; in [0, 1]; an isotropic design has
       A = 1/d).
xi_par is the trace-matched design with the LARGEST alignment (same polytope, same trace band).

Solver. Fully-corrective Frank-Wolfe (column generation) on the polytope: the master problem is an LP over the convex
weights of the vertices found so far (trace band and alignment rows with penalised slacks, HiGHS); the linear
minimisation oracle is an EXACT finite-horizon dynamic program over deterministic Markov (time-indexed) policies with
the LP reduced costs as per-(s, a) cost -- every vertex is a legal deterministic policy of the same form as the problem
policies (policy.act(t, loads, engaged)). Deviation from the plan text ("stationary" policies): for a finite horizon
the LMO over stationary policies is not a DP; Markov policies are the standard vertex set of the finite-horizon
occupancy polytope (a superset of the stationary hull). At most `max_iter` (200) oracle calls. When the oracle finds
no column with negative reduced cost, the LP optimum is the optimum over the whole polytope, so a positive slack is a
CERTIFICATE of non-constructibility (under the theta_hat transition model); otherwise the result is "not constructed
within 200 iterations". Either way the problem is `orth_infeasible` (never downgraded to an illegal design).

Data. The designed mixture is executed on a sibling platform handle: each episode samples a vertex policy with the
convex weights, resets the sibling to s0 (n_resets is counted) and plays H steps; rows are collected until N_k.
The realised design (empirical cell counts of the returned rows) is re-scored at theta_hat (realised trace error and
alignment); constructibility is decided on the planned design (realised numbers are reported, not gated).

Truth isolation: this module imports no ground-truth module and reads no truth attribute; it consumes the model class
tables, authentic observations (shadow ledger rows) and public Problem fields only.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linprog

from ..core.dynamics import next_loads
from ..core.provenance import require_authentic

TRACE_TOL = 0.05
A_MAX = 0.05
MAX_ITER = 200
SLACK_PEN = 100.0
RC_TOL = 1e-9


# =============================================================================================== alignment helpers
def alignment(I: np.ndarray, u: np.ndarray) -> float:
    """A(I; u) = u^T I u / (||u||^2 tr I)."""
    u = np.asarray(u, float)
    nu2 = float(u @ u)
    tr = float(np.trace(I))
    if nu2 <= 0 or tr <= 0:
        return float("nan")
    return float(u @ I @ u) / (nu2 * tr)


# =============================================================================================== tabular MDP + LMO
class TabularMDP:
    """Finite-horizon MDP over a (states x actions) table with K successor slots per (s, a).
    nxt (S*A, K) int successor state index, prob (S*A, K) probabilities (rows sum to 1)."""

    def __init__(self, nS: int, nA: int, nxt: np.ndarray, prob: np.ndarray):
        self.nS, self.nA = int(nS), int(nA)
        self.nxt = np.asarray(nxt, np.int64).reshape(nS * nA, -1)
        self.prob = np.asarray(prob, float).reshape(nS * nA, -1)
        assert np.allclose(self.prob.sum(1), 1.0, atol=1e-9), "transition rows must sum to 1"

    def best_policy(self, g: np.ndarray, H: int):
        """argmin over deterministic Markov policies of sum_t E[g(s_t, a_t)] / H  (exact backward DP).
        g: (S*A,) per-(s, a) cost. Returns policy table (H, S) of action indices."""
        S, A = self.nS, self.nA
        g = np.asarray(g, float).reshape(S * A)
        V = np.zeros(S)
        pol = np.zeros((H, S), np.int64)
        for t in range(H - 1, -1, -1):
            Q = (g / H + (self.prob * V[self.nxt]).sum(1)).reshape(S, A)
            pol[t] = np.argmin(Q, 1)
            V = Q[np.arange(S), pol[t]]
        return pol, V

    def occupancy(self, pol: np.ndarray, s0: int) -> np.ndarray:
        """Per-step occupancy (S*A,) of a Markov policy table from state s0 (sums to 1)."""
        S, A = self.nS, self.nA
        H = pol.shape[0]
        d = np.zeros(S)
        d[s0] = 1.0
        occ = np.zeros(S * A)
        for t in range(H):
            sa = np.arange(S) * A + pol[t]
            occ[sa] += d
            w = d[:, None] * self.prob[sa]
            d = np.bincount(self.nxt[sa].reshape(-1), weights=w.reshape(-1), minlength=S)
        return occ / H


# =============================================================================================== column generation
def construct_design(mdp: TabularMDP, s0: int, H: int, t_sa: np.ndarray, c_sa: list, T_target: float,
                     mode: str = "perp", trace_tol: float = TRACE_TOL, a_max: float = A_MAX,
                     max_iter: int = MAX_ITER, inner_trace_tol: float | None = 0.03,
                     inner_a_max: float | None = 0.04) -> dict:
    """Fully-corrective Frank-Wolfe / column generation over the reachable occupancy polytope.

    t_sa      (S*A,) expected per-round Fisher trace contribution of (s, a)
    c_sa      list of (S*A,) arrays: u_p^T F(s, a) u_p / ||u_p||^2 for every contrast direction p
    T_target  per-row Fisher trace of the reference (vol) data
    mode      'perp' : trace band + A_p <= a_max for every p, minimise sum_p A_p   -> xi_perp
              'par'  : trace band, maximise sum_p alignment                        -> xi_par
              'min'  : trace band, minimise sum_p alignment (no A rows)            -> A_min diagnostic
    Two passes: first with the tighter inner band (margin for the realised design), then -- only if the inner pass
    leaves a positive slack -- continued with the pre-registered thresholds. Returns weights, vertex policies, the
    planned design statistics and the convergence certificate."""
    t0 = time.perf_counter()
    T = float(T_target)
    P = len(c_sa)
    if mode not in ("perp", "par", "min"):
        raise ValueError(mode)
    sign = -1.0 if mode == "par" else 1.0
    obj_sa = sign * sum(c_sa) / T                                   # per-(s, a) objective (scaled)
    cols, pols = [], []                                             # column = (occ-weighted) [obj, t/T, c_p/T...]

    def add(pol):
        occ = mdp.occupancy(pol, s0)
        cols.append(np.array([occ @ obj_sa, occ @ t_sa / T] + [occ @ c / T for c in c_sa]))
        pols.append((pol, occ))

    n_oracle = 0
    for g in (t_sa, -t_sa, sum(c_sa), -sum(c_sa)):                  # warm start: extreme vertices
        add(mdp.best_policy(g, H)[0])
        n_oracle += 1

    def solve_lp(ttol, amax):
        X = np.array(cols)                                          # (n, 2 + P)
        n = len(X)
        rows, rhs = [], []
        rows.append(-X[:, 1]); rhs.append(-(1.0 - ttol))            # trace >= (1 - tol) T
        rows.append(X[:, 1]); rhs.append(1.0 + ttol)                # trace <= (1 + tol) T
        if mode == "perp":
            for p in range(P):
                rows.append(X[:, 2 + p] - amax * X[:, 1]); rhs.append(0.0)
        m = len(rows)
        A_ub = np.zeros((m, n + m))
        A_ub[:, :n] = np.array(rows)
        A_ub[:, n:] = -np.eye(m)
        c = np.concatenate([X[:, 0], np.full(m, SLACK_PEN)])
        A_eq = np.concatenate([np.ones(n), np.zeros(m)])[None]
        res = linprog(c, A_ub=A_ub, b_ub=np.array(rhs), A_eq=A_eq, b_eq=[1.0], bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"master LP failed: {res.message}")
        return res, n, m

    def pricing_cost(res, ttol, amax):
        y = res.ineqlin.marginals                                   # d obj / d b_ub  (<= 0)
        y_eq = float(res.eqlin.marginals[0])
        coef = [-t_sa / T, t_sa / T]
        if mode == "perp":
            coef += [(c - amax * t_sa) / T for c in c_sa]
        g = obj_sa - sum(yy * cc for yy, cc in zip(y, coef))
        return g, y_eq

    passes = [(inner_trace_tol, inner_a_max)] if inner_trace_tol is not None else []
    passes.append((trace_tol, a_max))
    hist = []
    res = None
    for ttol, amax in passes:
        converged = False
        while True:
            res, n, m = solve_lp(ttol, amax)
            g, y_eq = pricing_cost(res, ttol, amax)
            if n_oracle >= max_iter:
                break
            pol, V = mdp.best_policy(g, H)
            n_oracle += 1
            rc = float(V[s0]) - y_eq
            hist.append(rc)
            if rc >= -RC_TOL * max(1.0, abs(y_eq)):
                converged = True
                break
            add(pol)
        slack = float(res.x[n:].sum())
        if slack <= 1e-9:
            break
    lam = res.x[:n]
    lam = np.where(lam > 1e-12, lam, 0.0)
    lam = lam / lam.sum()
    occ = sum(w * pols[i][1] for i, w in enumerate(lam) if w > 0)
    tr = float(occ @ t_sa)
    A = [float(occ @ c) / tr if tr > 0 else float("nan") for c in c_sa]
    keep = [i for i, w in enumerate(lam) if w > 0]
    trace_err = tr / T - 1.0
    ok_trace = abs(trace_err) <= trace_tol + 1e-12
    ok_align = all(a <= a_max + 1e-12 for a in A) if mode == "perp" else True
    return {"mode": mode, "weights": [float(lam[i]) for i in keep], "policies": [pols[i][0] for i in keep],
            "occ": occ, "trace_per_row": tr, "trace_err": trace_err, "A": A, "A_max_pairs": max(A) if A else None,
            "feasible": bool(ok_trace and ok_align), "slack": slack, "converged": converged,
            "certified_infeasible": bool(converged and slack > 1e-9), "n_oracle": n_oracle,
            "n_vertices": len(keep), "n_columns": len(cols), "band_used": [ttol, amax],
            "sec": time.perf_counter() - t0}


# =============================================================================================== NL layer
class NLOrthModel:
    """(state, action) tables for the E1-NL class. Structure (theta-free) is built once; `mdp_at(theta)` adds the
    plug-in transition probabilities and the expected per-round cell counts at theta_hat.

    prop : exact.nl_propagate.NLPropagator;  lev : mechanism.leverage.NLLeverage (same factor indexing)."""

    def __init__(self, prop, lev):
        self.prop, self.lev = prop, lev
        cod, asp = prop.codec, prop.aspace
        self.codec, self.aspace = cod, asp
        self.nS, self.nA = cod.size, asp.n
        # every NL action is legal for the replay design (incentive level 1 only, as in the initial data)
        assert max(asp.max_level(a) for a in range(asp.n)) <= 1
        L, P, N = prop.L, prop.P, prop.N
        C = lev.C
        SA = self.nS * self.nA
        K = 2 ** P
        fl, cl, nl = [], [], []
        Fmax = 0
        structs = []
        for s in range(self.nS):
            for a in range(self.nA):
                fidx, const, nxt, _ = prop._struct(s, a)
                structs.append((fidx, const, nxt))
                Fmax = max(Fmax, fidx.shape[1])
        self.FID = np.full((SA, K, Fmax), prop.pad, np.int64)
        self.CONST = np.zeros((SA, K))
        self.NXT = np.zeros((SA, K), np.int64)
        for k, (fidx, const, nxt) in enumerate(structs):
            self.FID[k, :, :fidx.shape[1]] = fidx
            self.CONST[k] = const
            self.NXT[k] = nxt
        # expected cell counts structure: pair slots (cell, ret i y1/y0, ret j y1/y0) and lone retention cells (y=0)
        npair = lev.n_pair
        mP = min(prop.L, prop.R)
        self.PC = np.full((SA, mP), C, np.int64)
        self.RI1 = np.full((SA, mP), C, np.int64)
        self.RI0 = np.full((SA, mP), C, np.int64)
        self.RJ1 = np.full((SA, mP), C, np.int64)
        self.RJ0 = np.full((SA, mP), C, np.int64)
        self.SG = np.full((SA, P), C, np.int64)
        rc = lambda p, y, n: npair + (p * 2 + y) * N + n  # noqa: E731
        for s in range(self.nS):
            loads, eng = cod.decode(s)
            for a in range(self.nA):
                k = s * self.nA + a
                act = asp.actions[a]
                inc = {(i, j): l for i, j, l in act.incentives}
                grouped = np.zeros(P, bool)
                slot = 0
                for i, j in act.pairs:
                    if eng[i] and eng[L + j]:
                        b = inc.get((i, j), 0)
                        self.PC[k, slot] = prop.pair_idx(i, j, loads[i], b)
                        self.RI1[k, slot], self.RI0[k, slot] = rc(i, 1, loads[i]), rc(i, 0, loads[i])
                        self.RJ1[k, slot], self.RJ0[k, slot] = rc(L + j, 1, loads[L + j]), rc(L + j, 0, loads[L + j])
                        grouped[i] = grouped[L + j] = True
                        slot += 1
                for p in range(P):
                    if eng[p] and not grouped[p]:
                        self.SG[k, p] = rc(p, 0, loads[p])
        self.C = C

    def transitions(self, LT_row: np.ndarray) -> np.ndarray:
        lp = LT_row[self.FID].sum(-1) + self.CONST
        return np.exp(lp)

    def expected_cells(self, mu: np.ndarray) -> np.ndarray:
        """(S*A, C) expected per-round factor-draw counts of every cell at mean vector mu."""
        SA = self.NXT.shape[0]
        mu_ext = np.concatenate([mu, [0.0]])
        E = np.zeros((SA, self.C + 1))
        rows = np.arange(SA)[:, None]
        mp = mu_ext[self.PC]
        valid = (self.PC < self.C).astype(float)
        np.add.at(E, (rows, self.PC), valid)
        np.add.at(E, (rows, self.RI1), mp * valid)
        np.add.at(E, (rows, self.RJ1), mp * valid)
        np.add.at(E, (rows, self.RI0), (1 - mp) * valid)
        np.add.at(E, (rows, self.RJ0), (1 - mp) * valid)
        np.add.at(E, (rows, self.SG), (self.SG < self.C).astype(float))
        return E[:, :self.C]

    def mdp_at(self, LT_row: np.ndarray, mu: np.ndarray):
        prob = self.transitions(np.asarray(LT_row, float))
        return TabularMDP(self.nS, self.nA, self.NXT, prob), self.expected_cells(mu)

    def sa_quantities(self, E: np.ndarray, mu: np.ndarray, dirs: list):
        """t(s, a) = E[tr F | s, a]; c_p(s, a) = E[u_p^T F u_p | s, a] / ||u_p||^2 at mu."""
        Phi = self.lev.Phi
        w = mu * (1.0 - mu)
        t_sa = E @ (w * (Phi ** 2).sum(1))
        cs = []
        for u in dirs:
            uh = u / np.linalg.norm(u)
            cs.append(E @ (w * (Phi @ uh) ** 2))
        return t_sa, cs


def run_mixture(handle, s0_loads, s0_eng, policies, weights, H: int, n_rows: int, codec, rng, action_map=None):
    """Execute the designed mixture on a (sibling) platform handle; episodes restart at s0 via reset_to."""
    out = []
    w = np.asarray(weights, float)
    w = w / w.sum()
    n_ep = 0
    while len(out) < n_rows:
        v = int(rng.choice(len(w), p=w))
        pol = policies[v]
        handle.reset_to(np.asarray(s0_loads, np.int64), np.asarray(s0_eng, np.int64))
        n_ep += 1
        for t in range(H):
            if len(out) >= n_rows:
                break
            code = codec.encode(*handle.observable_state())
            a = int(pol[t, code])
            out.append(handle.step(a if action_map is None else int(action_map[a])))
    for o in out:
        require_authentic(o)
    return out, n_ep


def nl_design_for_problem(model: NLOrthModel, lev, LT_row, theta_vec, problem, pairs, s0_code, vol_obs,
                          max_iter: int = MAX_ITER, fallback_order=None) -> dict:
    """xi_perp and xi_par for one NL problem at theta_hat (no data execution). vol_obs: the matched vol rows.
    If every pre-registered pair is behaviourally degenerate (h = 0: the two policies induce identical cell
    occupancies from s0), the contrast pi_hat vs the best-ranked (fallback_order, e.g. J at theta_hat descending)
    policy with a non-degenerate contrast is used instead (label 'fallback_distinct')."""
    t0 = time.perf_counter()
    mu = lev.mu_from_theta(theta_vec)
    dirs, pinfo = [], []
    pairs = list(pairs)
    if fallback_order is not None and pairs:
        pi_hat = pairs[0][1]
        pairs_fb = [("fallback_distinct", pi_hat, int(k)) for k in fallback_order if int(k) != pi_hat]
    else:
        pairs_fb = []
    for label, k1, k2 in pairs + pairs_fb:
        if label == "fallback_distinct" and dirs:
            break
        cl = lev.contrast_leverage(mu, problem, k1, k2)
        u = lev.theta_gradient(cl["h"], mu)
        if np.linalg.norm(u) <= 1e-14:
            pinfo.append({"label": label, "k1": int(k1), "k2": int(k2), "degenerate": True})
            continue
        if any(abs(abs(float(u @ d) / (np.linalg.norm(u) * np.linalg.norm(d))) - 1) < 1e-12 for d in dirs):
            pinfo.append({"label": label, "k1": int(k1), "k2": int(k2), "duplicate_direction": True})
            continue
        dirs.append(u)
        pinfo.append({"label": label, "k1": int(k1), "k2": int(k2), "u_norm": float(np.linalg.norm(u)),
                      "dJ_hat": float(cl["dJ"])})
    t_lev = time.perf_counter() - t0
    I_vol = lev.fisher(lev.cell_counts(vol_obs), mu)
    n_vol = len(vol_obs)
    T_vol = float(np.trace(I_vol)) / n_vol
    A_vol = [alignment(I_vol, u) for u in dirs]
    out = {"n_dirs": len(dirs), "pairs": pinfo, "T_vol_per_row": T_vol, "A_vol": A_vol, "sec_leverage": t_lev}
    if not dirs:
        out.update({"status": "no_decision_direction", "feasible": False})
        return out
    t1 = time.perf_counter()
    mdp, E = model.mdp_at(LT_row, mu)
    t_sa, c_sa = model.sa_quantities(E, mu, dirs)
    out["sec_mdp"] = time.perf_counter() - t1
    H = problem.H
    perp = construct_design(mdp, s0_code, H, t_sa, c_sa, T_vol, mode="perp", max_iter=max_iter)
    par = construct_design(mdp, s0_code, H, t_sa, c_sa, T_vol, mode="par", max_iter=max_iter)
    amin = None if perp["feasible"] else construct_design(mdp, s0_code, H, t_sa, c_sa, T_vol, mode="min",
                                                          max_iter=max_iter, inner_trace_tol=None)
    # polytope range of the trace (diagnostic): min / max per-row trace over the polytope
    tmin = float(mdp.occupancy(mdp.best_policy(t_sa, H)[0], s0_code) @ t_sa)
    tmax = float(mdp.occupancy(mdp.best_policy(-t_sa, H)[0], s0_code) @ t_sa)
    out.update({"perp": perp, "par": par, "min": amin, "trace_range_per_row": [tmin, tmax], "feasible": perp["feasible"],
                "status": "constructed" if perp["feasible"] else
                ("certified_infeasible" if perp["certified_infeasible"] else "not_constructed_within_iter"),
                "fw_sec": time.perf_counter() - t0, "H": int(H)})
    return out


def realized_stats(lev, obs, theta_vec, dirs, T_ref):
    mu = lev.mu_from_theta(theta_vec)
    I = lev.fisher(lev.cell_counts(obs), mu)
    tr = float(np.trace(I)) / max(len(obs), 1)
    return {"trace_per_row": tr, "trace_err": tr / T_ref - 1.0, "A": [alignment(I, u) for u in dirs]}


# =============================================================================================== Lin layer
class LinOrthModel:
    """E1-Lin (identity link, Gaussian, known sigma, known deterministic load dynamics A1). Cells = legal atoms
    (i, j, n_i, b <= maxlev) of baselines.lin_rage.LinGeom; per-round Fisher = sum_pairs x x^T / sigma^2
    (theta-free). States = load codes (no engagement)."""

    def __init__(self, geom, codec, nmax: int, sigma: float):
        self.geom, self.codec = geom, codec
        self.aspace = geom.aspace
        self.legal = np.asarray(geom.legal, np.int64)
        self.nS, self.nA = codec.size, len(self.legal)
        self.sigma = float(sigma)
        SA = self.nS * self.nA
        self.NXT = np.zeros((SA, 1), np.int64)
        self.ATOMS = np.full((SA, geom.ai.shape[1]), -1, np.int64)
        for s in range(self.nS):
            loads, _ = codec.decode(s)
            atoms = geom.action_atoms(loads)
            for ai, a in enumerate(self.legal):
                k = s * self.nA + ai
                nl = next_loads(loads, self.aspace, int(a), nmax, geom.lc.static)
                self.NXT[k, 0] = codec.encode(nl)
                m = geom.am[a]
                self.ATOMS[k, :m.sum()] = atoms[a][m]
        self.mdp = TabularMDP(self.nS, self.nA, self.NXT, np.ones((SA, 1)))

    def sa_quantities(self, dirs):
        F = self.geom.Ftab
        valid = self.ATOMS >= 0
        at = np.where(valid, self.ATOMS, 0)
        tr_atom = (F ** 2).sum(1) / self.sigma ** 2
        t_sa = (tr_atom[at] * valid).sum(1)
        cs = []
        for u in dirs:
            uh = u / np.linalg.norm(u)
            ca = (F @ uh) ** 2 / self.sigma ** 2
            cs.append((ca[at] * valid).sum(1))
        return t_sa, cs

    def fisher_rows(self, obs) -> np.ndarray:
        d = self.geom.lc.d
        I = np.zeros((d, d))
        for o in obs:
            for i, j, b, _ in o.outcomes:
                x = self.geom.lc.feature(i, j, o.loads[i], b)
                I += np.outer(x, x)
        return I / self.sigma ** 2

    def policy_actions(self, pol):
        """Map a policy table over legal-action positions to action indices."""
        return self.legal[pol]


def lin_design_for_problem(model: LinOrthModel, dirs, s0_code, H, vol_obs, max_iter: int = MAX_ITER) -> dict:
    t0 = time.perf_counter()
    I_vol = model.fisher_rows(vol_obs)
    T_vol = float(np.trace(I_vol)) / len(vol_obs)
    t_sa, c_sa = model.sa_quantities(dirs)
    perp = construct_design(model.mdp, s0_code, H, t_sa, c_sa, T_vol, mode="perp", max_iter=max_iter)
    par = construct_design(model.mdp, s0_code, H, t_sa, c_sa, T_vol, mode="par", max_iter=max_iter)
    amin = None if perp["feasible"] else construct_design(model.mdp, s0_code, H, t_sa, c_sa, T_vol, mode="min",
                                                          max_iter=max_iter, inner_trace_tol=None)
    tmin = float(model.mdp.occupancy(model.mdp.best_policy(t_sa, H)[0], s0_code) @ t_sa)
    tmax = float(model.mdp.occupancy(model.mdp.best_policy(-t_sa, H)[0], s0_code) @ t_sa)
    return {"T_vol_per_row": T_vol, "A_vol": [alignment(I_vol, u) for u in dirs], "perp": perp, "par": par,
            "min": amin, "trace_range_per_row": [tmin, tmax], "feasible": perp["feasible"],
            "status": "constructed" if perp["feasible"] else
            ("certified_infeasible" if perp["certified_infeasible"] else "not_constructed_within_iter"),
            "fw_sec": time.perf_counter() - t0, "H": int(H)}


def strip(design: dict) -> dict:
    """JSON-safe summary of a construct_design result (drops policy tables and occupancy vectors)."""
    return {k: v for k, v in design.items() if k not in ("policies", "occ")}
