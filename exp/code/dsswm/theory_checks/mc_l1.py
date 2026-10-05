"""Monte Carlo checks for Lemma L1 (stratified without-replacement functional
Bernstein) and Theorem 1 (QFC certificate) -- task r4_theory_qfc.

Parts
-----
A  L1 coverage: random finite-pool configurations, non-adaptive counts,
   empirical P(Dhat - D >= sqrt(2Vx) + b x / 3) vs e^{-x}, x in {2,3,5}.
   Both tails.  Contrast width: CLT quantile z_{1-e^{-x}} sqrt(V) (not a valid
   bound; used to show the harness has power).
B  Full certificate (Thm 1): (i) Hillstrom S=6 x A=3 visit pools, 15 problems,
   K=20 checkpoints, pool-proportional schedule (random permutation of the
   table), variance UCB via Bernstein inversion; (ii) synthetic S=2 x A=3
   near-tie stress configurations.  Contrast: naive certificate (plug-in
   variance, Gaussian width, x = ln(1/delta), no union).
C  Deterministic checks: Hoeffding convex order on exact MGFs, min_t
   inequality, external reviewer 13/12 counterexample, FCR_old counterexample.
D  Adaptivity demo: fixed-count bound + checkpoint union is NOT valid when
   counts are data-dependent (motivates the amendment to methodology 1.5).

All randomness from numpy Generator seeded from --seed (default 42).
CPU only; <= 4 worker processes; BLAS threads pinned to 1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import json
import math
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

XS = (2.0, 3.0, 5.0)
TASK_ID = "r4_theory_qfc"


# ---------------------------------------------------------------------------
# Core formulas (shared with the certificate implementation)
# ---------------------------------------------------------------------------
def l1_width(V, b, x):
    """Lemma L1 width sqrt(2 V x) + b x / 3."""
    return np.sqrt(2.0 * V * x) + b * x / 3.0


def bernstein_mu_ci(muhat, n, N, x, iters=60):
    """Two-sided WoR Bernstein-inversion CI for a [0,1] binary pool mean.

    Set {mu in [0,1] : |muhat - mu| <= sqrt(2 mu(1-mu) x / n) + x / (3n)}.
    Valid with prob >= 1 - 2 e^{-x} by Lemma L1 applied to a single cell with
    a = +1 and a = -1 (sigma^2 = mu(1-mu) for a binary pool).  The set is an
    interval because mu -> |muhat-mu| - g(mu) is convex on each side.
    n == N  -> point mass at muhat (pool exhausted, mean known exactly).
    n == 0  -> [0, 1].
    Vectorised over arrays of equal shape.
    """
    muhat = np.asarray(muhat, float)
    n = np.asarray(n, float)
    N = np.asarray(N, float)
    nn = np.maximum(n, 1.0)

    def g(mu):
        return np.sqrt(2.0 * mu * (1.0 - mu) * x / nn) + x / (3.0 * nn)

    # upper end: largest mu in [muhat, 1] with mu - muhat <= g(mu)
    lo_u, hi_u = muhat.copy(), np.ones_like(muhat)
    ok1 = (1.0 - muhat) <= g(np.ones_like(muhat))
    for _ in range(iters):
        mid = 0.5 * (lo_u + hi_u)
        inside = (mid - muhat) <= g(mid)
        lo_u = np.where(inside, mid, lo_u)
        hi_u = np.where(inside, hi_u, mid)
    upper = np.where(ok1, 1.0, hi_u)  # hi_u is outside-or-boundary: outward
    # lower end: smallest mu in [0, muhat] with muhat - mu <= g(mu)
    lo_l, hi_l = np.zeros_like(muhat), muhat.copy()
    ok0 = muhat <= g(np.zeros_like(muhat))
    for _ in range(iters):
        mid = 0.5 * (lo_l + hi_l)
        inside = (muhat - mid) <= g(mid)
        hi_l = np.where(inside, mid, hi_l)
        lo_l = np.where(inside, lo_l, mid)
    lower = np.where(ok0, 0.0, lo_l)  # lo_l is outside-or-boundary: outward
    exact = n >= N
    lower = np.where(exact, muhat, lower)
    upper = np.where(exact, muhat, upper)
    empty = n <= 0
    lower = np.where(empty, 0.0, lower)
    upper = np.where(empty, 1.0, upper)
    return lower, upper


def sigma2_ucb(lower, upper):
    """max_{mu in [lower, upper]} mu (1 - mu)."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    nearest = np.clip(0.5, lower, upper)
    return nearest * (1.0 - nearest)


def cp_upper(k, n, alpha=0.05):
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def write_progress(results_dir, epoch, total, metric=None):
    p = Path(results_dir) / f"{TASK_ID}_PROGRESS.json"
    p.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total,
        "step": epoch, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat(),
    }))


# ---------------------------------------------------------------------------
# Part A: L1 coverage on random finite pools
# ---------------------------------------------------------------------------
def make_config(rng, cid):
    kind = "binary" if rng.random() < 0.8 else "nonbinary"
    regime = rng.choice(["tiny", "mid", "large", "deplete"], p=[0.25, 0.3, 0.25, 0.2])
    C = int(rng.choice([1, 1, 2, 3, 6, 12, 18]))
    Nmax = 5000 if kind == "binary" else 2000
    N = np.exp(rng.uniform(np.log(50), np.log(Nmax), size=C)).astype(int)
    pools = []
    for c in range(C):
        mu_t = rng.uniform(0.01, 0.5)
        if kind == "binary":
            ones = max(1, int(round(mu_t * N[c])))
            vals = np.zeros(N[c]); vals[:ones] = 1.0
        else:
            u = rng.random(N[c])
            p1 = mu_t * rng.uniform(0.2, 0.9)  # point mass at 1
            vals = np.where(u < p1, 1.0,
                            np.where(u < p1 + (1 - p1) * rng.uniform(0.3, 0.9), 0.0,
                                     rng.beta(0.5, 2.0, size=N[c])))
        pools.append(vals)
    w = rng.dirichlet(np.ones(C))
    a = w * rng.choice([-1.0, 1.0], size=C)
    if C > 1:
        zero = rng.random(C) < 0.3
        if zero.all():
            zero[rng.integers(C)] = False
        a[zero] = 0.0
    # random non-adaptive schedule: arrivals with Dirichlet cell probabilities,
    # counts capped at pool size (exhaustion), n_c >= 1 where a_c != 0
    p = rng.dirichlet(np.ones(C))
    Ntot = N.sum()
    if regime == "tiny":
        T = int(rng.integers(C, 10 * C + 1))
    elif regime == "mid":
        T = int(np.exp(rng.uniform(np.log(10 * C), np.log(max(10 * C + 1, Ntot // 4)))))
    elif regime == "large":
        T = int(rng.uniform(0.25, 1.0) * Ntot)
    else:
        T = int(rng.uniform(0.6, 1.5) * Ntot)
    n = np.minimum(rng.multinomial(T, p), N)
    if regime == "deplete":
        full = rng.random(C) < 0.5
        n[full] = N[full]
    n = np.where((a != 0) & (n == 0), 1, n)
    return dict(cid=cid, kind=kind, regime=str(regime), C=C, N=N, pools=pools,
                a=a, n=n)


def run_config(args):
    cfg, reps, seed = args
    rng = np.random.default_rng(seed)
    a, n, N, pools = cfg["a"], cfg["n"], cfg["N"], cfg["pools"]
    act = np.nonzero(a)[0]
    mu = np.array([v.mean() for v in pools])
    s2 = np.array([v.var() for v in pools])  # finite-population variance
    D = float((a * mu).sum())
    V = float(sum(a[c] ** 2 * s2[c] / n[c] for c in act))
    b = float(max(abs(a[c]) / n[c] for c in act))
    # exhausted-cell refinement (L1 remark): cells with n_c = N_c are exact
    live = [c for c in act if n[c] < N[c]]
    V_r = float(sum(a[c] ** 2 * s2[c] / n[c] for c in live))
    b_r = float(max([abs(a[c]) / n[c] for c in live], default=0.0))
    dev = np.zeros(reps)
    for c in act:
        if n[c] >= N[c]:
            continue  # exhausted: muhat_c = mu_c exactly
        vals = pools[c]
        if cfg["kind"] == "binary":
            ones = int(vals.sum())
            S = rng.hypergeometric(ones, N[c] - ones, n[c], size=reps).astype(float)
        else:
            S = np.empty(reps)
            chunk = max(1, int(2e7 // N[c]))
            for st in range(0, reps, chunk):
                r = min(chunk, reps - st)
                keys = rng.random((r, N[c]))
                idx = np.argpartition(keys, n[c] - 1, axis=1)[:, : n[c]]
                S[st:st + r] = vals[idx].sum(axis=1)
        dev += a[c] * (S / n[c] - mu[c])
    rows = []
    for x in XS:
        bound = math.exp(-x)
        wdt = l1_width(V_r, b_r, x)  # refined width (<= original width)
        wdt_o = l1_width(V, b, x)
        clt = stats.norm.ppf(1 - bound) * math.sqrt(V)
        up = int((dev >= wdt - 1e-12).sum())
        lo = int((-dev >= wdt - 1e-12).sum())
        cu = int((dev >= clt - 1e-12).sum())
        cl = int((-dev >= clt - 1e-12).sum())
        uo = int((np.abs(dev) >= wdt_o - 1e-12).sum())
        if not live:
            up = lo = cu = cl = 0  # dev == 0 identically
        rows.append(dict(
            config_id=cfg["cid"], kind=cfg["kind"], regime=cfg["regime"], C=cfg["C"],
            n_active=len(act), min_n=int(n[act].min()), max_n=int(n[act].max()),
            min_N=int(N[act].min()), n_exhausted=int((n[act] >= N[act]).sum()),
            min_mu=float(mu[act].min()), V=V, b=b, x=x, e_neg_x=bound, reps=reps,
            width=wdt, width_orig=wdt_o, viol_two_sided_orig=uo, viol_upper=up, viol_lower=lo,
            rate_upper=up / reps, rate_lower=lo / reps,
            ratio_upper=up / reps / bound, ratio_lower=lo / reps / bound,
            ratio_max=max(up, lo) / reps / bound,
            cp95_upper_ratio=cp_upper(max(up, lo), reps) / bound,
            clt_ratio_max=max(cu, cl) / reps / bound,
        ))
    return rows


# ---------------------------------------------------------------------------
# Part B: full certificate (Thm 1)
# ---------------------------------------------------------------------------
def certificate_problem_hillstrom():
    import pandas as pd
    d = pd.read_pickle(str(Path(__import__("os").environ.get("DATA_DIR", "data")) / "hillstrom/tidy.pkl"))
    hs = d["x_history_segment"].to_numpy()
    hs3 = np.where(hs == 0, 0, np.where(hs <= 2, 1, 2))
    seg = hs3 * 2 + d["x_newbie"].to_numpy()
    arm = d["treatment"].to_numpy().astype(int)
    y = d["binary"].to_numpy().astype(int)
    S, A = 6, 3
    w = np.array([(seg == s).mean() for s in range(S)])
    Npool = np.zeros((S, A), int)
    ones = np.zeros((S, A), int)
    for s in range(S):
        for k in range(A):
            m = (seg == s) & (arm == k)
            Npool[s, k] = m.sum()
            ones[s, k] = y[m].sum()
    costs = {"C1": (0, 1, 1), "C2": (0, 1.5, 1), "C3": (0, 2, 1)}
    budgets = (0.2, 0.4, 0.6, 0.8, 1.0)
    problems = [(cn, np.array(cv, float), B) for cn in costs for cv in [costs[cn]] for B in budgets]
    T_total = int(Npool.sum())
    ckpts = np.unique(np.round(np.exp(np.linspace(np.log(1000), np.log(T_total), 20))).astype(int))
    return dict(name="hillstrom_S6", S=S, A=A, w=w, Npool=Npool, ones=ones,
                problems=problems, ckpts=ckpts, schedule="table_permutation")


def certificate_problem_synth(rng, cid):
    S, A = 2, 3
    w = rng.dirichlet(np.ones(S) * 2)
    Npool = np.exp(rng.uniform(np.log(200), np.log(3000), size=(S, A))).astype(int)
    base = rng.uniform(0.05, 0.5, size=S)
    gaps = rng.uniform(0.0, 0.08, size=(S, A))
    mu = np.clip(base[:, None] + gaps, 0.01, 0.99)
    ones = np.maximum(1, np.round(mu * Npool)).astype(int)
    problems = []
    for j in range(5):
        cv = np.concatenate([[0.0], rng.uniform(0.5, 2.0, size=A - 1)])
        B = float(rng.uniform(0.2, 1.5))
        problems.append((f"P{j}", cv, B))
    T_total = int(Npool.sum())
    ckpts = np.unique(np.round(np.exp(np.linspace(np.log(20), np.log(T_total), 20))).astype(int))
    alloc = rng.dirichlet(np.ones(A) * 2, size=S)  # fixed (non-adaptive) arm probs
    return dict(name=f"synth_{cid}", S=S, A=A, w=w, Npool=Npool, ones=ones,
                problems=problems, ckpts=ckpts, schedule="iid_fixed_alloc",
                alloc=alloc)


def _policies(S, A):
    return np.array(list(itertools.product(range(A), repeat=S)), int)  # (P, S)


def run_certificate(args):
    prob, reps, seed, eps_list, delta_main, delta_var, n_cells_union = args
    rng = np.random.default_rng(seed)
    S, A, w = prob["S"], prob["A"], prob["w"]
    Npool, ones = prob["Npool"], prob["ones"]
    ckpts = prob["ckpts"]
    K = 20  # ledger uses the pre-registered K=20 even if rounding merged two
    Pol = _policies(S, A)
    nP = len(Pol)
    mu = ones / Npool
    J = (w[None, :] * mu[np.arange(S)[None, :], Pol]).sum(1)
    L1x = 2 * S * math.log(A) + math.log(K / delta_main)
    xvar = math.log(2 * n_cells_union * K / delta_var)
    naive_x = math.log(1 / (delta_main + delta_var))
    feas, jstar = [], []
    for (_, cv, B) in prob["problems"]:
        f = (w[None, :] * cv[Pol]).sum(1) <= B + 1e-12
        feas.append(f)
        jstar.append(J[f].max())
    feas = np.array(feas)
    jstar = np.array(jstar)
    Q = len(feas)
    flat_cells = (np.arange(S)[None, :] * A + Pol)  # (P, S) flat cell index
    out = {eps: dict(any_false=0, any_false_naive=0, cert_at_end=[], cert_naive_end=[],
                     false_q_naive=0, cert_q_naive=0, false_q=0, cert_q=0)
           for eps in eps_list}
    evar_miss = 0
    pair_cov_miss = 0
    for r in range(reps):
        # --- non-adaptive schedule, generated before any outcome --------------
        if prob["schedule"] == "table_permutation":
            rem = Npool.flatten().copy()
            counts = np.zeros(S * A, int)
            cnt_seq = []
            prev = 0
            for T in ckpts:
                inc = rng.multivariate_hypergeometric(rem, T - prev)
                counts = counts + inc
                rem = rem - inc
                prev = T
                cnt_seq.append(counts.copy())
        else:
            alloc = prob["alloc"]
            pc = (w[:, None] * alloc).flatten()
            draws = np.zeros(S * A, int)
            cnt_seq = []
            prev = 0
            for T in ckpts:
                draws = draws + rng.multinomial(T - prev, pc)
                prev = T
                cnt_seq.append(np.minimum(draws, Npool.flatten()))
        # --- outcomes: WoR prefix sums, sequential hypergeometric -----------
        onesf, Nf = ones.flatten(), Npool.flatten()
        rem1, rem0 = onesf.copy(), Nf - onesf
        Ssum = np.zeros(S * A)
        nprev = np.zeros(S * A, int)
        false_any = {eps: False for eps in eps_list}
        false_any_naive = {eps: False for eps in eps_list}
        evar_bad = False
        pair_bad = False
        for ki, n in enumerate(cnt_seq):
            inc = n - nprev
            got = np.array([rng.hypergeometric(rem1[c], rem0[c], inc[c]) if inc[c] > 0 else 0
                            for c in range(S * A)])
            rem1 -= got
            rem0 -= inc - got
            Ssum += got
            nprev = n.copy()
            nn = np.maximum(n, 1)
            muhat = np.where(n > 0, Ssum / nn, 0.5)
            lo, hi = bernstein_mu_ci(muhat, n, Nf, xvar)
            if ((onesf / Nf) < lo - 1e-12).any() or ((onesf / Nf) > hi + 1e-12).any():
                evar_bad = True
            s2bar = sigma2_ucb(lo, hi)
            live = n < Nf                                       # exhausted cells exact
            vterm = np.where(n > 0, np.where(live, s2bar / nn, 0.0), np.inf)
            vplug = np.where(n > 0, muhat * (1 - muhat) / nn, np.inf)
            binv = np.where(n > 0, np.where(live, 1.0 / nn, 0.0), np.inf)
            Jhat = (w[None, :] * muhat[flat_cells]).sum(1)
            for q in range(Q):
                f = feas[q]
                jh = np.where(f, Jhat, -np.inf)
                ih = int(np.argmax(jh))
                diff = Pol != Pol[ih][None, :]                  # (P, S)
                cpi = flat_cells                                # cells of pi'
                chat = flat_cells[ih][None, :]                  # cells of pihat
                ws2 = (w ** 2)[None, :]
                Vb = np.where(diff, ws2 * (vterm[cpi] + vterm[chat]), 0.0).sum(1)
                Vp = np.where(diff, ws2 * (vplug[cpi] + vplug[chat]), 0.0).sum(1)
                bb = np.where(diff, w[None, :] * np.maximum(binv[cpi], binv[chat]), 0.0).max(1)
                dh = Jhat - Jhat[ih]
                U = np.where(f, dh + np.sqrt(2 * L1x * Vb) + bb * L1x / 3, -np.inf)
                Un = np.where(f, dh + np.sqrt(2 * naive_x * Vp), -np.inf)
                Umax, Unmax = U.max(), Un.max()
                regret = jstar[q] - J[ih]
                # coverage of the pair (pi*_q, pihat_q) by the L1 event
                istar = int(np.argmax(np.where(f, J, -np.inf)))
                if J[istar] - J[ih] > Jhat[istar] - Jhat[ih] + math.sqrt(2 * L1x * Vb[istar]) + bb[istar] * L1x / 3 + 1e-12:
                    pair_bad = True
                for eps in eps_list:
                    o = out[eps]
                    if Umax <= eps:
                        o["cert_q"] += 1
                        if regret > eps:
                            o["false_q"] += 1
                            false_any[eps] = True
                    if Unmax <= eps:
                        o["cert_q_naive"] += 1
                        if regret > eps:
                            o["false_q_naive"] += 1
                            false_any_naive[eps] = True
                    if ki == len(cnt_seq) - 1:
                        o.setdefault("_end", 0)
                        o.setdefault("_endn", 0)
                        o["_end"] += int(Umax <= eps)
                        o["_endn"] += int(Unmax <= eps)
        for eps in eps_list:
            out[eps]["any_false"] += int(false_any[eps])
            out[eps]["any_false_naive"] += int(false_any_naive[eps])
        evar_miss += int(evar_bad)
        pair_cov_miss += int(pair_bad)
    res = []
    for eps in eps_list:
        o = out[eps]
        res.append(dict(
            instance=prob["name"], eps=eps, reps=reps, Q=Q, K=len(ckpts),
            L1x=L1x, xvar=xvar, delta=delta_main + delta_var,
            any_false=o["any_false"], rate_any_false=o["any_false"] / reps,
            cp95_any_false=cp_upper(o["any_false"], reps),
            evar_miss=evar_miss, rate_evar_miss=evar_miss / reps,
            pair_cov_miss=pair_cov_miss,
            mean_cert_per_ckpt_q=o["cert_q"] / (reps * len(ckpts) * Q),
            mean_cert_at_end=o.get("_end", 0) / (reps * Q),
            naive_any_false=o["any_false_naive"], naive_rate_any_false=o["any_false_naive"] / reps,
            naive_mean_cert_per_ckpt_q=o["cert_q_naive"] / (reps * len(ckpts) * Q),
            naive_false_over_cert=(o["false_q_naive"] / o["cert_q_naive"]) if o["cert_q_naive"] else 0.0,
        ))
    return res


# ---------------------------------------------------------------------------
# Part C: deterministic checks
# ---------------------------------------------------------------------------
def deterministic_checks(rng):
    out = {}
    # C1 Hoeffding (1963, Thm 4) convex order on exact MGFs, binary pools
    worst = -np.inf
    n_checked = 0
    for _ in range(300):
        N = int(rng.integers(2, 400))
        K1 = int(rng.integers(0, N + 1))
        n = int(rng.integers(1, N + 1))
        lam = rng.uniform(-3, 3)
        k = np.arange(0, n + 1)
        pw = stats.hypergeom.pmf(k, N, K1, n)
        pr = stats.binom.pmf(k, n, K1 / N)
        from scipy.special import logsumexp
        with np.errstate(divide="ignore"):
            mw = logsumexp(np.log(pw) + lam * k)
            mr = logsumexp(np.log(pr) + lam * k)
        worst = max(worst, mw - mr)
        n_checked += 1
    out["C1_convex_order_max_logMGF_WoR_minus_WR"] = float(worst)
    out["C1_pass"] = bool(worst <= 1e-9)
    out["C1_n_checked"] = n_checked
    # C1b: Bernstein MGF bound for one WR draw: log E e^{l Y} <= v l^2 / (2(1 - l b/3))
    worst2 = -np.inf
    for _ in range(2000):
        mu = rng.uniform(0.001, 0.999)
        nn = int(rng.integers(1, 1000))
        lam = rng.uniform(0, 2.999 * nn)  # need lam b < 3 with b = 1/nn
        b = 1.0 / nn
        y1, y0 = (1 - mu) / nn, -mu / nn
        lhs = np.log(mu * np.exp(lam * y1) + (1 - mu) * np.exp(lam * y0))
        v = mu * (1 - mu) / nn ** 2
        rhs = v * lam ** 2 / (2 * (1 - lam * b / 3))
        worst2 = max(worst2, lhs - rhs)
    out["C1b_bernstein_mgf_max_violation"] = float(worst2)
    out["C1b_pass"] = bool(worst2 <= 1e-12)
    # C2 min_t inequality sqrt(2LV) <= L/t + tV/2 (AM-GM), random
    L = rng.uniform(1e-3, 50, 200000)
    V = rng.uniform(1e-8, 5, 200000)
    t = np.exp(rng.uniform(-8, 8, 200000))
    gap = (L / t + t * V / 2) - np.sqrt(2 * L * V)
    out["C2_min_gap"] = float(gap.min())
    out["C2_pass"] = bool(gap.min() >= -1e-9)
    # C3 external reviewer counterexample: L=1, (dh,V) in {(0,1/2),(-1,2)}
    pts = [(0.0, 0.5), (-1.0, 2.0)]
    u_enum = max(dh + math.sqrt(2 * 1.0 * v) for dh, v in pts)
    ts = np.linspace(0.01, 10, 200001)
    u_dp = min(max(dh + 1.0 / tt + tt * v / 2 for dh, v in pts) for tt in ts[::1000])
    tt = 4.0 / 3.0
    u_dp_exact = max(dh + 1.0 / tt + tt * v / 2 for dh, v in pts)
    out["C3_U_enum"] = u_enum
    out["C3_U_dp_grid"] = float(u_dp)
    out["C3_U_dp_exact_t43"] = u_dp_exact
    out["C3_gap"] = u_dp_exact - u_enum
    out["C3_pass"] = bool(abs(u_enum - 1.0) < 1e-12 and abs(u_dp_exact - 13 / 12) < 1e-12
                          and u_dp >= u_enum - 1e-12)
    # C4 FCR_old counterexample (external reviewer): 5% streams certify 15 all wrong,
    # 95% certify 8 all right
    fwer = 0.05
    fcr_old = 0.05 * 15 / (0.05 * 15 + 0.95 * 8)
    evr = 0.05 * 1.0
    out["C4_FWER"] = fwer
    out["C4_E_V_over_R"] = evr
    out["C4_FCR_old"] = fcr_old
    out["C4_pass"] = bool(abs(fcr_old - 0.0898) < 5e-4 and evr <= 0.05 + 1e-12)
    return out


# ---------------------------------------------------------------------------
# Part D: adaptive-count demo
# ---------------------------------------------------------------------------
def adaptive_demo(rng, reps=2000, N=20000, mu=0.3, M=10000, x=3.0):
    """One cell, one 'checkpoint'.  Adaptive rule: keep drawing one record at a
    time and stop as soon as the fixed-count L1 bound is crossed (or after M).
    Fixed-count validity would require P(cross) <= e^{-x}."""
    ones = int(mu * N)
    pool = np.zeros(N); pool[:ones] = 1.0
    crossed = 0
    nn = np.arange(1, M + 1)
    width = np.sqrt(2 * mu * (1 - mu) * x / nn) + x / (3 * nn)
    for st in range(0, reps, 200):
        r = min(200, reps - st)
        keys = rng.random((r, N))
        idx = np.argpartition(keys, M - 1, axis=1)[:, :M]
        # random order of the first M draws
        perm = rng.permuted(idx, axis=1)
        dev = np.cumsum(pool[perm], axis=1) / nn - mu
        crossed += int((dev >= width).any(axis=1).sum())
    return dict(reps=reps, N=N, mu=mu, M=M, x=x, e_neg_x=math.exp(-x),
                rate_crossed=crossed / reps, ratio=crossed / reps / math.exp(-x))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve()
    ws = here.parents[4]  # .../current
    results_dir = Path(args.results_dir) if args.results_dir else ws / "exp" / "results"
    sub = "pilots" if args.mode == "pilot" else "full"
    out_dir = Path(args.out) if args.out else results_dir / sub / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{TASK_ID}.pid").write_text(str(os.getpid()))

    if args.mode == "pilot":
        n_cfg, reps_a = 200, 10_000
        hill_reps, n_syn, syn_reps = 1000, 40, 500
    else:
        n_cfg, reps_a = 1000, 100_000
        hill_reps, n_syn, syn_reps = 10_000, 200, 2000
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    write_progress(results_dir, 0, 4, {"stage": "A_l1"})

    # Part A
    cfgs = [make_config(rng, i) for i in range(n_cfg)]
    seeds = rng.integers(0, 2**31 - 1, size=n_cfg)
    jobs = [(c, reps_a, int(s)) for c, s in zip(cfgs, seeds)]
    with Pool(args.workers) as pool:
        rowsA = [r for rs in pool.map(run_config, jobs, chunksize=4) for r in rs]
    with open(out_dir / "mc_coverage.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rowsA[0].keys()))
        wr.writeheader(); wr.writerows(rowsA)
    tA = time.time() - t0
    write_progress(results_dir, 1, 4, {"stage": "B_cert", "tA_sec": tA})

    # Part B
    eps_list = [0.0, 0.0025, 0.01]
    hill = certificate_problem_hillstrom()
    syn_rng = np.random.default_rng(args.seed + 1)
    syn = [certificate_problem_synth(syn_rng, i) for i in range(n_syn)]
    cjobs = []
    per = hill_reps // args.workers
    for j in range(args.workers):
        cjobs.append((hill, per, args.seed * 1000 + j, eps_list, 0.04, 0.01, 48))
    for i, p in enumerate(syn):
        cjobs.append((p, syn_reps, args.seed * 7919 + i, [0.0, 0.005, 0.02], 0.04, 0.01, 48))
    with Pool(args.workers) as pool:
        resB = [r for rs in pool.map(run_certificate, cjobs, chunksize=1) for r in rs]
    # merge hillstrom shards
    merged = {}
    for r in resB:
        key = (r["instance"], r["eps"])
        if key not in merged:
            merged[key] = dict(r)
            merged[key]["_w"] = r["reps"]
        else:
            m = merged[key]
            tot = m["_w"] + r["reps"]
            for k in ("any_false", "evar_miss", "pair_cov_miss", "naive_any_false"):
                m[k] += r[k]
            for k in ("mean_cert_per_ckpt_q", "mean_cert_at_end", "naive_mean_cert_per_ckpt_q",
                      "naive_false_over_cert"):
                m[k] = (m[k] * m["_w"] + r[k] * r["reps"]) / tot
            m["_w"] = tot
            m["reps"] = tot
    for m in merged.values():
        m["rate_any_false"] = m["any_false"] / m["reps"]
        m["cp95_any_false"] = cp_upper(m["any_false"], m["reps"])
        m["rate_evar_miss"] = m["evar_miss"] / m["reps"]
        m["naive_rate_any_false"] = m["naive_any_false"] / m["reps"]
        m.pop("_w", None)
    rowsB = list(merged.values())
    with open(out_dir / "cert_coverage.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rowsB[0].keys()))
        wr.writeheader(); wr.writerows(rowsB)
    tB = time.time() - t0 - tA
    write_progress(results_dir, 2, 4, {"stage": "C_det", "tB_sec": tB})

    det = deterministic_checks(np.random.default_rng(args.seed + 2))
    write_progress(results_dir, 3, 4, {"stage": "D_adaptive"})
    adem = adaptive_demo(np.random.default_rng(args.seed + 3))

    # ---- summary -------------------------------------------------------
    ratios = np.array([r["ratio_max"] for r in rowsA])
    by_x = {}
    for x in XS:
        rx = [r for r in rowsA if r["x"] == x]
        worst = max(rx, key=lambda r: r["ratio_max"])
        by_x[str(x)] = dict(
            max_ratio=worst["ratio_max"], worst_config=worst["config_id"],
            worst_kind=worst["kind"], worst_regime=worst["regime"],
            worst_min_n=worst["min_n"], worst_cp95_ratio=worst["cp95_upper_ratio"],
            median_ratio=float(np.median([r["ratio_max"] for r in rx])),
            max_clt_ratio=max(r["clt_ratio_max"] for r in rx),
            n_cfg_clt_over_1=int(sum(r["clt_ratio_max"] > 1.0 for r in rx)),
        )
    hill_rows = [r for r in rowsB if r["instance"].startswith("hillstrom")]
    syn_rows = [r for r in rowsB if r["instance"].startswith("synth")]
    syn_tot = {}
    for eps in sorted({r["eps"] for r in syn_rows}):
        rr = [r for r in syn_rows if r["eps"] == eps]
        reps = sum(r["reps"] for r in rr)
        fa = sum(r["any_false"] for r in rr)
        na = sum(r["naive_any_false"] for r in rr)
        syn_tot[str(eps)] = dict(
            reps=reps, any_false=fa, rate=fa / reps, cp95=cp_upper(fa, reps),
            max_config_rate=max(r["rate_any_false"] for r in rr),
            evar_miss=sum(r["evar_miss"] for r in rr) // 1,
            mean_cert_per_ckpt_q=float(np.mean([r["mean_cert_per_ckpt_q"] for r in rr])),
            mean_cert_at_end=float(np.mean([r["mean_cert_at_end"] for r in rr])),
            naive_any_false=na, naive_rate=na / reps,
            naive_max_config_rate=max(r["naive_rate_any_false"] for r in rr),
            naive_mean_cert_per_ckpt_q=float(np.mean([r["naive_mean_cert_per_ckpt_q"] for r in rr])),
        )
    max_ratio = float(ratios.max())
    cert_max_rate = max([r["rate_any_false"] for r in hill_rows] +
                        [v["max_config_rate"] for v in syn_tot.values()])
    gate_mc = max_ratio <= 1.2
    gate_cert = cert_max_rate <= 0.05
    det_pass = all(v for k, v in det.items() if k.endswith("_pass"))
    summary = dict(
        task_id=TASK_ID, mode=args.mode, seed=args.seed,
        runtime_sec=round(time.time() - t0, 1), concurrent_run=True,
        partA=dict(n_configs=n_cfg, reps_per_config=reps_a, xs=list(XS),
                   max_violation_ratio=max_ratio, by_x=by_x,
                   n_kind=dict(binary=sum(c["kind"] == "binary" for c in cfgs),
                               nonbinary=sum(c["kind"] == "nonbinary" for c in cfgs)),
                   n_regime={k: sum(c["regime"] == k for c in cfgs)
                             for k in ("tiny", "mid", "large", "deplete")}),
        partB=dict(hillstrom=hill_rows, synthetic_totals=syn_tot,
                   n_synth_configs=n_syn, synth_reps=syn_reps,
                   full_certificate_max_violation_rate=cert_max_rate),
        partC=det, partD_adaptive_demo=adem,
        gate=dict(
            mc_max_violation_ratio_le_1p2=gate_mc,
            full_certificate_violation_le_delta=gate_cert,
            deterministic_checks_pass=det_pass,
            proof_checklist_complete=None,  # filled after doc self-check
        ),
        metrics=dict(max_violation_ratio=max_ratio),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_progress(results_dir, 4, 4, {"max_violation_ratio": max_ratio,
                                        "cert_max_rate": cert_max_rate})
    print(json.dumps(dict(max_ratio=max_ratio, gate_mc=gate_mc, cert_max_rate=cert_max_rate,
                          det_pass=det_pass, adem=adem, runtime=summary["runtime_sec"]),
                     default=float))


if __name__ == "__main__":
    main()
