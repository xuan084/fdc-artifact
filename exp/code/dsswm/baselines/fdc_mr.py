"""FDC-MR: multi-resolution direction union for the FDC certificate (A = 2).  NEW file (fdc_bet exploration).

Observation.  For A = 2 the deviation in a direction (pi', pi) only depends on the set D = {s : pi'(s) != pi(s)} and
the sign pattern sigma_D (sigma_s = +1 if pi'(s) = treatment): with uplift errors e_s = (mu_{s,1} - mu_hat_{s,1}) -
(mu_{s,0} - mu_hat_{s,0}),
        Delta(pi', pi) - Delta_hat(pi', pi) = sum_{s in D} sigma_s w_s e_s  =: Y(D, sigma_D).
So the event controlled for the pair (pi*_q, pi) in Theorem FDC-1 is an event about the SIGNED SEGMENT SUBSET
(D, sigma_D), shared by every problem q and every pair that induces the same subset.  FDC's ledger counts one event per
(q, pi) (sum_q |Pi_{B_q}| = 3386 on CR9 dev); FDC-MR instead puts a data-free weight on every signed subset,

        omega(D, sigma) = rho_{|D|} / (C(S, |D|) 2^{|D|}),   sum_d rho_d = 1,
        beta(D) = ln(K / (delta_main omega(D, sigma))),

and controls E_main = cap_{(D, sigma), k} {Y_k(D, sigma) <= w_k(D, sigma; beta(D), true mu)}; P(E_main^c | A) <=
sum_k sum_{(D, sigma)} delta_main omega / K = delta_main.  On E_main every subset event holds simultaneously, so for a
partition D = T_1 u ... u T_r (sign pattern inherited) Y(D) = sum_j Y(T_j) <= sum_j w(T_j): the certificate uses the
best partition,
        B(D) = min( w(D), min_{T subset D proper, T contains min(D)} B(T) + B(D \\ T) )        (subset DP, 3^S),
optionally with the free per-cell rectangle on E_var as an extra candidate for every block.  Then
        U_q(k) = max_{pi' in Pi_{B_q}} [Delta_hat(pi', pi_hat_q) + B(D(pi', pi_hat_q))]
and the proof of Theorem FDC-1 goes through verbatim (pi' = pi*_q is one challenger, (pi*_q, pi_hat) induces a signed
subset in the union).  The weights rho are data-free (chosen from a pre-declared grid on tuning seeds 900-949).
Cell bounds / variance box / ledger split exactly as ``fdc_bet.FDCBet``.
"""
from __future__ import annotations

import math
from math import comb

import numpy as np

from .fdc_bet import FDCBet, direction_widths
from .frontier_common import jhat_all, pi_hat_indices

__all__ = ["FDCMR", "rho_profile", "RHO_GRID", "subset_dp_tables", "partition_dp", "mr_ledger"]

RHO_GRID = ("uniform", "geo2", "inv_d2", "front3",                      # tuning round 1
            "front2", "front3x", "mixF50", "mixF30", "mixF50x")          # tuning round 2 (declared after round 1)
# mixF*: a fraction rho_F of delta_main goes to FDC's own per-(q, pi) union (family A, whole directions only); the
# rest is spread over signed subsets (family B).  mixF50 / mixF30: family B on |D| <= 3 only (1/3 each);
# mixF50x: family B = front3 profile.
FAMILY_A = {"mixF50": 0.5, "mixF30": 0.3, "mixF50x": 0.5}


def rho_profile(S, name):
    d = np.arange(1, S + 1, dtype=float)
    if name == "uniform":
        r = np.ones(S)
    elif name == "geo2":
        r = 0.5 ** d
    elif name == "inv_d2":
        r = 1.0 / d ** 2
    elif name == "front3":                                      # 3/4 of the mass on |D| <= 3
        r = np.where(d <= 3, 0.25, 0.25 / max(S - 3, 1))
    elif name == "front2":
        r = np.where(d <= 2, 0.35, 0.30 / max(S - 2, 1))
    elif name == "front3x":
        r = np.where(d <= 3, 0.30, 0.10 / max(S - 3, 1))
    elif name in ("mixF50", "mixF30"):
        r = np.where(d <= 3, 1.0 / 3.0, 0.0)
    elif name == "mixF50x":
        r = np.where(d <= 3, 0.25, 0.25 / max(S - 3, 1))
    else:
        raise ValueError(name)
    return r / r.sum()


def mr_ledger(ctx, rho_name, delta_main):
    from .fdc import union_size
    S, K = ctx.S, int(len(ctx.checkpoints))
    rF = FAMILY_A.get(rho_name, 0.0)
    rho = rho_profile(S, rho_name) * (1.0 - rF)
    cnt = np.array([comb(S, d) * 2 ** d for d in range(1, S + 1)], dtype=float)
    with np.errstate(divide="ignore"):
        beta_d = np.where(rho > 0, np.log(K * cnt / (delta_main * np.where(rho > 0, rho, 1.0))), np.inf)
    bound = float(np.sum(np.where(rho > 0, K * cnt * np.exp(-np.where(rho > 0, beta_d, 0.0)), 0.0)))
    out = {"rho": rho.tolist(), "rho_family_A": rF, "n_signed_subsets_by_size": cnt.astype(int).tolist(),
           "beta_by_size": beta_d.tolist(), "bound_main_family_B": bound}
    if rF > 0:
        m = union_size(ctx, "feas")
        out["beta_family_A"] = math.log(m * K / (delta_main * rF))
        out["union_size_family_A"] = m
        bound += m * K * math.exp(-out["beta_family_A"])
    out["bound_main"] = bound
    return out


def subset_dp_tables(S):
    """Per popcount level: arrays (D, T, D^T) over proper submasks T of D that contain the lowest bit of D."""
    levels = {}
    for D in range(1, 1 << S):
        low = D & -D
        T = (D - 1) & D
        while T:
            if T & low:
                levels.setdefault(bin(D).count("1"), []).append((D, T, D ^ T))
            T = (T - 1) & D
    return {L: tuple(np.array(x, dtype=np.int64) for x in zip(*v)) for L, v in levels.items()}


def partition_dp(base, tables, S):
    """B[D] = min(base[D], min over partitions); base indexed by mask (len 2^S), base[0] = 0."""
    B = np.array(base, dtype=float)
    B[0] = 0.0
    for L in range(2, S + 1):
        if L not in tables:
            continue
        Dl, Tl, Rl = tables[L]
        np.minimum.at(B, Dl, B[Tl] + B[Rl])
    return B


class FDCMR(FDCBet):
    """FDC-bet certificate with the multi-resolution signed-subset union and the partition DP."""

    def __init__(self, kind="min", var_box="HG", fpc=True, rect=True, split=(0.045, 0.005), rho="uniform",
                 partition=True, name=None):
        super().__init__(kind=kind, var_box=var_box, fpc=fpc, rect=rect, split=split, name=name)
        if rho not in RHO_GRID:
            raise ValueError(rho)
        self.rho_name = rho
        self.partition = bool(partition)
        if name is None:
            self.name = f"FDC-MR[{kind},{var_box},rho={rho},rect={int(rect)},part={int(partition)}," \
                        f"{split[0]:g}/{split[1]:g}]"

    def setup(self, ctx):
        super().setup(ctx)
        if ctx.P != 2 ** ctx.S:
            raise ValueError("FDC-MR needs the full policy class [2]^S (subset <-> policy bijection)")
        self.mr = mr_ledger(ctx, self.rho_name, self.split[0])
        self.ledger.update({"union": "signed segment subsets, size-graded", **self.mr})
        bits = (ctx.pols * (1 << np.arange(ctx.S))[None, :]).sum(1)
        self._row_of_bits = np.empty(1 << ctx.S, dtype=np.int64)
        self._row_of_bits[bits] = np.arange(ctx.P)
        self._bits = bits
        self._size = np.array([bin(int(m)).count("1") for m in range(1 << ctx.S)])
        self._tables = subset_dp_tables(ctx.S) if self.partition else None
        self._beta_d = np.concatenate([[0.0], np.asarray(self.mr["beta_by_size"])])

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("checkpoints must be visited in increasing order")
        self._last_t = st.t
        lo, hi = self._box(ctx, st)
        mu = st.mu_hat
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        seg = np.arange(ctx.S)
        cache = {}
        out = []
        for q in range(ctx.Q):
            ih = int(ihs[q])
            if ih not in cache:
                h = int(self._bits[ih])
                Dmask = self._bits ^ h                           # (P,) subset of each challenger row
                beta = self._beta_d[self._size[Dmask]]
                fin = np.isfinite(beta)
                wd = np.full(ctx.P, np.inf)
                wd_all = direction_widths(ctx, ih, st.n, st.N, lo, hi, np.where(fin, beta, 1.0), self.kind, self.fpc,
                                          self.grid)
                wd[fin] = wd_all[fin]
                wd[Dmask == 0] = 0.0
                if self.rect:
                    ph = ctx.pols[ih]
                    D = ctx.pols != ph[None, :]
                    rect = (ctx.w[None, :] * np.where(D, hi[seg[None, :], ctx.pols] - lo[seg, ph][None, :], 0.0)).sum(1)
                    wd = np.minimum(wd, rect - (J - J[ih]))       # rect bounds Delta directly: width = rect - Delta_hat
                if self.partition:
                    base = np.empty(1 << ctx.S)
                    base[Dmask] = wd
                    B = partition_dp(base, self._tables, ctx.S)
                    wd = B[Dmask]
                if "beta_family_A" in self.mr:                      # whole direction via FDC's per-(q, pi) family
                    wA = direction_widths(ctx, ih, st.n, st.N, lo, hi, self.mr["beta_family_A"], self.kind, self.fpc,
                                          self.grid)
                    wd = np.minimum(wd, wA)
                cache[ih] = (J - J[ih]) + wd
            Uq = float(np.max(np.where(ctx.feas[q], cache[ih], -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        return out

    def describe(self):
        d = super().describe()
        d.update({"rho": self.rho_name, "partition": self.partition})
        return d
