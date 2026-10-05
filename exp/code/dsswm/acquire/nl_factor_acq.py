"""Factorised one-round observation model for E1-NL acquisition rules (learner side, no truth access).

One platform round (state s, action a) produces an observation that factorises into independent factors:
  * active pair (i, j): joint (y_ij, e_i', e_j')            -> 8 outcomes ("group" factor)
  * engaged participant not in an active pair: e_p' (y=0)    -> 2 outcomes ("single" factor)
  * disengaged participant: e_p' ~ Bern(rho_ret), known      -> 2 outcomes, theta-free ("dis" factor)
Every (s, a) row therefore has <= 4 factors and <= 64 joint outcomes.  Because factors are independent,
KL(P_theta1(.|s,a) || P_theta2(.|s,a)) and the Fisher information are sums over factors, which makes
decision-disagreement (DDA), Fisher-design and IG scores cheap for all (s, a) rows at once.

Tables built here use only the public propagator structure and the model-class log-prob table LT
(NLPropagator.tables over the class) -- never the environment.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..exact.nl_propagate import NLPropagator

NEG = -1e30


class FactorModel:
    def __init__(self, prop: NLPropagator):
        self.prop = prop
        L, R, N, nb, P = prop.L, prop.R, prop.N, prop.nb, prop.P
        self.L, self.R, self.N, self.nb, self.P = L, R, N, nb, P
        self.n_grp = L * R * N * N * nb
        self.off_single = self.n_grp
        self.off_dis = self.n_grp + P * N
        self.pad = self.off_dis + P
        self.n_fac = self.pad + 1
        self.S = prop.codec.size
        self.A = prop.aspace.n
        self.d = 3 * L + 2 * R + 2      # alpha, gamma, tauL, beta, tauR, psi, lam
        self._build_rows()
        self._build_features()

    # ------------------------------------------------------------ indices
    def grp_id(self, i, j, ni, nj, b):
        return (((i * self.R + j) * self.N + ni) * self.N + nj) * self.nb + b

    def single_id(self, p, n):
        return self.off_single + p * self.N + n

    def dis_id(self, p):
        return self.off_dis + p

    def row(self, code, a_idx):
        return int(code) * self.A + int(a_idx)

    def _build_rows(self):
        prop, L, P = self.prop, self.L, self.P
        nrow = self.S * self.A
        rowfac = np.full((nrow, 4), self.pad, dtype=np.int32)
        flat = np.full((nrow, 64, 4), self.pad * 8, dtype=np.int32)
        flat[:, :, 0] = self.pad * 8 + 1          # invalid outcome -> -inf
        nxt = np.zeros((nrow, 64), dtype=np.int64)
        nout = np.zeros(nrow, dtype=np.int32)
        pow2 = 2 ** np.arange(P)
        for code in range(self.S):
            loads, eng = prop.codec.decode(code)
            for a_idx, a in enumerate(prop.aspace.actions):
                r = code * self.A + a_idx
                inc = {(i, j): l for i, j, l in a.incentives}
                facs = []           # (factor id, list of (outcome idx, {p: e'}))
                grouped = np.zeros(P, bool)
                for i, j in a.pairs:
                    if eng[i] and eng[L + j]:
                        b = inc.get((i, j), 0)
                        fid = self.grp_id(i, j, loads[i], loads[L + j], b)
                        outs = [(y * 4 + ei * 2 + ej, {i: ei, L + j: ej}) for y in (0, 1) for ei in (0, 1) for ej in (0, 1)]
                        facs.append((fid, outs))
                        grouped[i] = grouped[L + j] = True
                for p in range(P):
                    if grouped[p]:
                        continue
                    fid = self.single_id(p, loads[p]) if eng[p] else self.dis_id(p)
                    facs.append((fid, [(e, {p: e}) for e in (0, 1)]))
                matched = np.zeros(P, bool)
                for i, j in a.pairs:
                    matched[i] = matched[L + j] = True
                if prop.static:
                    nl = loads.copy()
                else:
                    nl = np.where(matched, np.minimum(loads + 1, prop.nmax), np.maximum(loads - 1, 0))
                lc = int(np.dot(nl, prop.codec._pow_n))
                for k, (fid, _) in enumerate(facs):
                    rowfac[r, k] = fid
                for o, combo in enumerate(itertools.product(*[f[1] for f in facs])):
                    e_next = np.zeros(P, dtype=np.int64)
                    for k, (oi, eb) in enumerate(combo):
                        flat[r, o, k] = facs[k][0] * 8 + oi
                        for p, e in eb.items():
                            e_next[p] = e
                    nxt[r, o] = lc + prop.codec.n_load_codes * int(e_next @ pow2)
                nout[r] = o + 1
        self.ROWFAC, self.FLAT, self.NEXT, self.NOUT = rowfac, flat, nxt, nout

    def _build_features(self):
        """Linear-logit feature vectors used by the closed-form Fisher information."""
        L, R, N, nb, P, d = self.L, self.R, self.N, self.nb, self.P, self.d
        ia = lambda i: i
        ig = lambda i: L + i
        itl = lambda i: 2 * L + i
        ib = lambda j: 3 * L + j
        itr = lambda j: 3 * L + R + j
        ipsi, ilam = 3 * L + 2 * R, 3 * L + 2 * R + 1
        xy = np.zeros((self.n_fac, d))
        xri = np.zeros((self.n_fac, d))
        xrj = np.zeros((self.n_fac, d))
        for i in range(L):
            for j in range(R):
                for ni in range(N):
                    for nj in range(N):
                        for b in range(nb):
                            f = self.grp_id(i, j, ni, nj, b)
                            xy[f, ia(i)] = 1
                            xy[f, ib(j)] = 1
                            xy[f, ig(i)] = -ni
                            xy[f, ipsi] = float(b >= 1)
                            xri[f, itl(i)] = 1
                            xri[f, ilam] = -ni
                            xrj[f, itr(j)] = 1
                            xrj[f, ilam] = -nj
        for p in range(P):
            for n in range(N):
                f = self.single_id(p, n)
                if p < L:
                    xri[f, itl(p)] = 1
                else:
                    xri[f, itr(p - L)] = 1
                xri[f, ilam] = -n
        self.XY, self.XRI, self.XRJ = xy, xri, xrj

    # ------------------------------------------------------------ per-theta factor tables
    def factor_table(self, LT: np.ndarray) -> np.ndarray:
        """LT (B, U+1) class log-prob table -> FT (B, n_fac, 8) factor outcome log-probs (pad = -inf)."""
        prop, L, R, N, nb, P = self.prop, self.L, self.R, self.N, self.nb, self.P
        B = LT.shape[0]
        FT = np.full((B, self.n_fac, 8), NEG, dtype=np.float64)
        for i in range(L):
            for j in range(R):
                for ni in range(N):
                    for nj in range(N):
                        for b in range(nb):
                            f = self.grp_id(i, j, ni, nj, b)
                            for y in (0, 1):
                                for ei in (0, 1):
                                    for ej in (0, 1):
                                        FT[:, f, y * 4 + ei * 2 + ej] = (LT[:, prop.py_idx(y, i, j, ni, b)]
                                                                         + LT[:, prop.re_idx(i, y, ni, ei)]
                                                                         + LT[:, prop.re_idx(L + j, y, nj, ej)])
        for p in range(P):
            for n in range(N):
                f = self.single_id(p, n)
                for e in (0, 1):
                    FT[:, f, e] = LT[:, prop.re_idx(p, 0, n, e)]
            FT[:, self.dis_id(p), 0] = prop.log_1mrho
            FT[:, self.dis_id(p), 1] = prop.log_rho
        FT[:, self.pad, 0] = 0.0
        return FT

    # ------------------------------------------------------------ scores
    @staticmethod
    def factor_kl(ft1: np.ndarray, ft2: np.ndarray) -> np.ndarray:
        """ft1 (n_fac, 8), ft2 (m, n_fac, 8) -> (m, n_fac) KL(P1 || P2) per factor."""
        p1 = np.exp(ft1)
        diff = ft1[None] - ft2
        diff = np.where(p1[None] > 0, diff, 0.0)
        return (p1[None] * diff).sum(-1)

    def row_kl(self, ft1, ft2, rows=None) -> np.ndarray:
        kf = self.factor_kl(ft1, ft2)                      # (m, n_fac)
        rf = self.ROWFAC if rows is None else self.ROWFAC[rows]
        return kf[:, rf].sum(-1)                           # (m, rows)

    def row_logp(self, ft: np.ndarray, rows=None) -> np.ndarray:
        """ft (n_fac, 8) single theta -> (rows, 64) joint outcome log-probs."""
        fl = self.FLAT if rows is None else self.FLAT[rows]
        return ft.reshape(-1)[fl].sum(-1)

    def factor_fisher(self, ft: np.ndarray) -> np.ndarray:
        """Closed-form Fisher information of every factor at one theta: (n_fac, d, d)."""
        p = np.exp(ft)                                     # (n_fac, 8)
        n_fac, d = self.n_fac, self.d
        F = np.zeros((n_fac, d, d))
        g = slice(0, self.n_grp)
        pg = p[g].reshape(-1, 2, 2, 2)                     # y, ei, ej
        py = pg.sum((2, 3))                                # (G, 2)
        q = py[:, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            ri = np.where(py > 0, pg[:, :, 1, :].sum(-1) / py, 0.0)   # P(ei=1 | y)
            rj = np.where(py > 0, pg[:, :, :, 1].sum(-1) / py, 0.0)
        wy = q * (1 - q)
        wi = (py * ri * (1 - ri)).sum(1)
        wj = (py * rj * (1 - rj)).sum(1)
        XY, XRI, XRJ = self.XY[g], self.XRI[g], self.XRJ[g]
        F[g] = (wy[:, None, None] * XY[:, :, None] * XY[:, None, :]
                + wi[:, None, None] * XRI[:, :, None] * XRI[:, None, :]
                + wj[:, None, None] * XRJ[:, :, None] * XRJ[:, None, :])
        s = slice(self.off_single, self.off_dis)
        r = p[s, 1]
        w = r * (1 - r)
        F[s] = w[:, None, None] * self.XRI[s][:, :, None] * self.XRI[s][:, None, :]
        return F

    def row_fisher(self, ff: np.ndarray, rows) -> np.ndarray:
        return ff[self.ROWFAC[rows]].sum(1)


def theta_vector(params_at: dict) -> np.ndarray:
    """NLClass.params_at(k) -> continuous parameter vector in FactorModel feature order."""
    return np.concatenate([params_at["alpha"], params_at["gamma"], params_at["tauL"], params_at["beta"],
                           params_at["tauR"], [params_at["psi_left"][0, 1]], [params_at["lam"]]]).astype(float)


def vector_to_params(v: np.ndarray, L: int, R: int, nb: int) -> dict:
    """Batch of continuous vectors (n, d) -> torch-ready numpy param dict for NLPropagator.tables."""
    v = np.atleast_2d(v)
    n = v.shape[0]
    out = {"alpha": v[:, 0:L], "gamma": v[:, L:2 * L], "tauL": v[:, 2 * L:3 * L], "beta": v[:, 3 * L:3 * L + R],
           "tauR": v[:, 3 * L + R:3 * L + 2 * R], "lam": v[:, 3 * L + 2 * R + 1]}
    psi_left = np.zeros((n, L, nb))
    psi_left[:, :, 1] = v[:, 3 * L + 2 * R][:, None]
    out["psi_left"] = psi_left
    return out


def perstep_jtable(prop: NLPropagator, params: dict, policies, loads0, engaged0, H: int, w, w_ret,
                   chunk: int = 4096) -> np.ndarray:
    """Exact per-step value contributions (B, K, H+1): column t < H = w_t * E[sum_pairs y_t],
    column H = w_ret * E[sum_p e_p(H)].  Unscaled (c_q = 1).  Sum over the last axis == j_table."""
    import torch
    LT, EY = prop.tables(params)
    B = LT.shape[0]
    out = np.zeros((B, len(policies), H + 1))
    for k, pi in enumerate(policies):
        plan, esum = prop.build_policy_plan(pi, loads0, engaged0, H)
        for s in range(0, B, chunk):
            lt, ey = LT[s:s + chunk], EY[s:s + chunk]
            b = lt.shape[0]
            Pt = torch.ones(b, 1, device=lt.device, dtype=lt.dtype)
            cols = []
            for t, st in enumerate(plan):
                eyv = ey[:, st["eyidx"]].sum(-1)
                cols.append(w[t] * (Pt * eyv).sum(1))
                logp = lt[:, st["fidx"]].sum(-1) + st["const"][None]
                W = Pt[:, st["src"]] * torch.exp(logp)
                Pn = torch.zeros(b, st["n_next"], device=lt.device, dtype=lt.dtype)
                Pn.index_add_(1, st["dst"], W)
                Pt = Pn
            cols.append(w_ret * (Pt * esum[None]).sum(1))
            out[s:s + chunk, k, :] = torch.stack(cols, 1).cpu().numpy()
    return out
