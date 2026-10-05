"""Decision-disagreement acquisition (DDA) for E1-NL over a finite class, via exact per-step KL.

The observation of one platform round factorises (see exact.nl_propagate):
  active pair (i,j): y ~ Bern(py[i,j,n_i,b]), then e_i' | y ~ Bern(pe[i,y,n_i]), e_j' | y ~ Bern(pe[L+j,y,n_j])
  engaged, not in an active pair: e_p' ~ Bern(pe[p,0,n_p])
  disengaged: known rho_ret (no information)
so KL(P_a(.|s,x) || P_b(.|s,x)) = S[x] . f(a,b), with S a theta-free 0/1 incidence matrix over (state, action)
and f a vector of elementary KL terms ("pair groups" G[i,j,ni,nj,b] and "lone retention" R0[p,n]).

Learner-side module: uses only the model class parameter grid and public structure.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def class_prob_tables(np_params: dict, c: float, nmax: int, nb: int, g=None):
    """py (B,L,R,N,nb) outcome probabilities and pe (B,P,2,N) retention probabilities for every theta."""
    N = nmax + 1
    gv = np.arange(N, dtype=float) if g is None else np.asarray(g, float)
    a, b, gm = np_params["alpha"], np_params["beta"], np_params["gamma"]
    psi = np_params["psi_left"][..., :nb]
    logit = (a[:, :, None, None, None] + b[:, None, :, None, None] - gm[:, :, None, None, None] * gv[None, None, None, :, None]
             + psi[:, :, None, None, :])
    syn = np_params.get("syn")
    if syn is not None:
        logit = logit + np.asarray(syn, float).reshape(-1, a.shape[1], b.shape[1])[:, :, :, None, None]
    py = _sig(logit)
    tau = np.concatenate([np_params["tauL"], np_params["tauR"]], 1)
    lam = np.asarray(np_params["lam"], float).reshape(-1)
    yv = np.arange(2.0)
    nv = np.arange(N, dtype=float)
    pe = _sig(tau[:, :, None, None] + c * yv[None, None, :, None] - lam[:, None, None, None] * nv[None, None, None, :])
    return py, pe


def single_prob_tables(p: dict, c: float, nmax: int, nb: int):
    """Same as class_prob_tables for one parameter dict (no batch dim) -> batch of 1."""
    d = {k: np.asarray(p[k], float)[None] for k in ("alpha", "beta", "gamma", "tauL", "tauR")}
    psi_left = np.asarray(p["psi_left"], float)
    if psi_left.shape[-1] < nb:
        psi_left = np.concatenate([psi_left, np.zeros(psi_left.shape[:-1] + (nb - psi_left.shape[-1],))], -1)
    d["psi_left"] = psi_left[None]
    d["lam"] = np.asarray([p["lam"]], float)
    if p.get("syn") is not None and np.any(np.asarray(p["syn"]) != 0):
        d["syn"] = np.asarray(p["syn"], float)[None]
    return class_prob_tables(d, c, nmax, nb, g=p.get("g"))


def _klb(p, q, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_features(py_a, pe_a, py_b, pe_b):
    """Elementary KL features of reference a (1,...) against models b (M,...). Returns (nf, M)."""
    k1 = _klb(py_a, py_b)                                   # (M,L,R,N,nb)
    k2 = _klb(pe_a, pe_b)                                   # (M,P,2,N)
    M, L, R, N, nb = k1.shape
    p1 = py_a[0]                                            # (L,R,N,nb)
    # retention terms after an active pair, weighted by the reference outcome probability
    ki = k2[:, :L]                                          # (M,L,2,N)
    kj = k2[:, L:]                                          # (M,R,2,N)
    # G[m,i,j,ni,nj,b]
    wi = (1 - p1)[None, :, :, :, None, :] * ki[:, :, None, 0, :, None, None] \
        + p1[None, :, :, :, None, :] * ki[:, :, None, 1, :, None, None]
    wj = (1 - p1)[None, :, :, :, None, :] * kj[:, None, :, 0, None, :, None] \
        + p1[None, :, :, :, None, :] * kj[:, None, :, 1, None, :, None]
    G = k1[:, :, :, :, None, :] + wi + wj                    # (M,L,R,N,N,nb)
    R0 = k2[:, :, 0, :]                                     # (M,P,N)
    return np.concatenate([G.reshape(M, -1), R0.reshape(M, -1)], 1).T


class IncidenceIndex:
    """theta-free incidence matrix S (n_states * n_actions, nf) for the KL decomposition."""

    def __init__(self, codec, aspace, nmax: int):
        self.codec, self.aspace = codec, aspace
        L, R, P, N, nb = aspace.L, aspace.R, aspace.L + aspace.R, nmax + 1, aspace.nb
        self.L, self.R, self.P, self.N, self.nb = L, R, P, N, nb
        self.nA = aspace.n
        self.nG = L * R * N * N * nb
        self.nf = self.nG + P * N
        S = np.zeros((codec.size * self.nA, self.nf), dtype=np.float32)
        for code in range(codec.size):
            loads, eng = codec.decode(code)
            for a_idx, a in enumerate(aspace.actions):
                row = code * self.nA + a_idx
                inc = {(i, j): l for i, j, l in a.incentives}
                grouped = np.zeros(P, bool)
                for i, j in a.pairs:
                    if eng[i] and eng[L + j]:
                        b = inc.get((i, j), 0)
                        gi = ((((i * R + j) * N + loads[i]) * N + loads[L + j]) * nb + b)
                        S[row, gi] += 1
                        grouped[i] = grouped[L + j] = True
                for p in range(P):
                    if eng[p] and not grouped[p]:
                        S[row, self.nG + p * N + loads[p]] += 1
        self.S = S
        self.max_level = np.array([aspace.max_level(k) for k in range(self.nA)])

    def rows_for_state(self, code: int):
        return self.S[code * self.nA:(code + 1) * self.nA]

    def used_feature_mask(self, legal_actions) -> np.ndarray:
        rows = self.S.reshape(self.codec.size, self.nA, self.nf)[:, legal_actions]
        return rows.reshape(-1, self.nf).max(0) > 0


def next_state_dist(prop, LT_row: np.ndarray, code: int, a_idx: int):
    """Exact distribution of the next observable state under one theta (LT_row: (U+1,) numpy log-table)."""
    fidx, const, nxt, _ = prop._struct(int(code), int(a_idx))
    logp = LT_row[fidx].sum(1) + const
    p = np.exp(logp)
    return nxt, p / p.sum()


def dda_choose(code, ref_py, ref_pe, LT_ref_row, model_py, model_pe, margins, inc: IncidenceIndex, prop,
               legal_actions, rng, two_step: bool = True, explore: float = 0.05):
    """Pick the next probe action from the current state.

    Columns = 1-step probes (cost 1) and 2-step 'steer-then-observe' sequences (cost 2, expectation over the
    reference model's next-state distribution). Rows = blocking models with remaining LR margins.
    max_w min_m sum_c w_c * KL_m(c) / (cost_c * margin_m) (Track-and-Stop style LP), action sampled from w.
    Returns (action, info dict)."""
    legal_actions = np.asarray(legal_actions)
    if rng.random() < explore or len(margins) == 0:
        return int(rng.choice(legal_actions)), {"mode": "explore", "z": None}
    F = kl_features(ref_py, ref_pe, model_py, model_pe)               # (nf, M)
    K1 = inc.rows_for_state(code)[legal_actions] @ F                  # (nA, M)
    cols = [K1]
    first = list(legal_actions)
    if two_step:
        K2 = np.zeros((len(legal_actions), len(legal_actions), F.shape[1]))
        for ia, a in enumerate(legal_actions):
            nxt, p = next_state_dist(prop, LT_ref_row, code, a)
            Rn = inc.S.reshape(inc.codec.size, inc.nA, inc.nf)[nxt][:, legal_actions]   # (16, nA, nf)
            K2[ia] = np.einsum("s,saf,fm->am", p, Rn, F)
        two = (K1[:, None, :] + K2) / 2.0
        cols.append(two.reshape(-1, F.shape[1]))
        first += [a for a in legal_actions for _ in legal_actions]
    C = np.concatenate(cols, 0)                                      # (ncol, M)
    first = np.asarray(first)
    Mrow = C / np.maximum(margins, 1e-6)[None]
    live = Mrow.max(0) > 1e-12
    if not live.any():
        return int(rng.choice(legal_actions)), {"mode": "zero_info", "z": 0.0}
    Mrow = Mrow[:, live]
    ncol, m = Mrow.shape
    # LP: max z s.t. Mrow^T w >= z, sum w = 1, w >= 0   ->   min -z
    c = np.zeros(ncol + 1); c[-1] = -1.0
    A_ub = np.hstack([-Mrow.T, np.ones((m, 1))])
    b_ub = np.zeros(m)
    A_eq = np.zeros((1, ncol + 1)); A_eq[0, :ncol] = 1.0
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * ncol + [(None, None)],
                      method="highs")
        w = np.clip(res.x[:ncol], 0, None)
        z = float(res.x[-1])
    except Exception:  # noqa: BLE001 - LP failure falls back to greedy max-min
        w = np.zeros(ncol); w[int(np.argmax(Mrow.min(1)))] = 1.0
        z = float(Mrow.min(1).max())
    if w.sum() <= 0:
        w = np.zeros(ncol); w[int(np.argmax(Mrow.min(1)))] = 1.0
    w = w / w.sum()
    k = int(rng.choice(ncol, p=w))
    return int(first[k]), {"mode": "dda", "z": z, "two_step": bool(k >= len(legal_actions)), "n_rows": int(m)}
