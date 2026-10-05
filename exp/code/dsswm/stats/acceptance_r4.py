"""Unified round-4 acceptance statistics (methodology §2; written into the r4 lock).

Per layer and main method:
* S1 (FWER): k = #streams with >= 1 false certification out of n independent streams; pass iff the one-sided
  exact CP upper bound <= delta (k <= 4 of 200 gives 0.04518).
* S2 (FCR_old): FCR_old = sum n_false / sum n_cert; one-sided 95% instance-cluster bootstrap upper bound
  (B = 1e4, seed 42) <= delta. FWER does not imply S2 (external reviewer counterexample: FWER = 0.05, FCR_old ~ 0.0898).
* E[V / (R v 1)] is reported alongside.
* E3: uncensored fraction, point >= 0.6 AND one-sided CP lower bound >= 0.5.
* E-ms(a): mean per-stream completion, one-sided stream-bootstrap lower bound >= 0.5.
* E2 / E-ms(c): paired mean log ratio, 95% stream-bootstrap CI and one-sided bootstrap p = P*(mean >= margin).
* E-ms(b): paired completion difference, 95% CI and one-sided p = P*(mean <= 0).
* Holm step-down over a family of p values.
* E-ms(d): reproduction of the r3 JPC failure: FWER CP lower bound > delta OR FCR_old cluster lower bound > delta.

All bootstraps resample streams (= instances = clusters) with replacement, use numpy default_rng(seed) and B draws,
and are vectorised over B. Percentile bounds use numpy's default (linear) quantile.
"""
from __future__ import annotations

import math

import numpy as np

from .cp import cp_lower_one_sided, cp_upper_one_sided

B_DEFAULT = 10_000
SEED_DEFAULT = 42
DELTA = 0.05


def _boot_idx(n: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def _boot_means(v: np.ndarray, B: int, seed: int) -> np.ndarray:
    idx = _boot_idx(len(v), B, seed)
    return v[idx].mean(axis=1)


# --------------------------------------------------------------------------------------------------- S1 / S2
def s1_fwer(stream_false_flags, delta: float = DELTA, alpha: float = 0.05) -> dict:
    f = np.asarray(stream_false_flags).astype(bool)
    k, n = int(f.sum()), int(f.size)
    ub = cp_upper_one_sided(k, n, alpha)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "cp_upper": ub,
            "cp_lower": cp_lower_one_sided(k, n, alpha), "pass": bool(ub <= delta)}


def fcr_old(n_false, n_cert) -> float:
    nf = float(np.sum(n_false))
    nc = float(np.sum(n_cert))
    return nf / nc if nc > 0 else 0.0


def s2_fcr_old(n_false, n_cert, delta: float = DELTA, B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
               alpha: float = 0.05) -> dict:
    """Instance-cluster bootstrap of the ratio-of-sums FCR_old; one-sided (1-alpha) upper and lower bounds."""
    nf = np.asarray(n_false, dtype=float)
    nc = np.asarray(n_cert, dtype=float)
    if nf.shape != nc.shape:
        raise ValueError("n_false and n_cert must have the same shape")
    if np.any(nf > nc) or np.any(nf < 0):
        raise ValueError("need 0 <= n_false <= n_cert per stream")
    point = fcr_old(nf, nc)
    idx = _boot_idx(len(nf), B, seed)
    snf = nf[idx].sum(axis=1)
    snc = nc[idx].sum(axis=1)
    boots = np.divide(snf, snc, out=np.zeros_like(snf), where=snc > 0)
    ub = float(np.quantile(boots, 1 - alpha))
    lb = float(np.quantile(boots, alpha))
    return {"fcr_old": point, "upper": ub, "lower": lb, "B": B, "seed": seed,
            "sum_false": float(nf.sum()), "sum_cert": float(nc.sum()), "pass": bool(ub <= delta)}


def e_v_over_r(n_false, n_cert) -> float:
    nf = np.asarray(n_false, dtype=float)
    nc = np.asarray(n_cert, dtype=float)
    return float(np.mean(nf / np.maximum(nc, 1.0)))


def layer_safety(n_false, n_cert, delta: float = DELTA, B: int = B_DEFAULT, seed: int = SEED_DEFAULT) -> dict:
    """S1 and S2 on one layer (per-stream counts of false and total certifications). Safe = S1 AND S2."""
    nf = np.asarray(n_false, dtype=float)
    s1 = s1_fwer(nf > 0, delta)
    s2 = s2_fcr_old(nf, n_cert, delta, B, seed)
    return {"S1": s1, "S2": s2, "E_V_over_R": e_v_over_r(nf, n_cert), "safe": bool(s1["pass"] and s2["pass"])}


def jpc_failure_reproduced(n_false, n_cert, delta: float = DELTA, B: int = B_DEFAULT,
                           seed: int = SEED_DEFAULT) -> dict:
    """E-ms(d): S1 or S2 failure of the r3 JPC shown on the same block (FWER CP lower > delta OR FCR_old lower > delta)."""
    nf = np.asarray(n_false, dtype=float)
    s1 = s1_fwer(nf > 0, delta)
    s2 = s2_fcr_old(nf, n_cert, delta, B, seed)
    fw = s1["cp_lower"] > delta
    fc = s2["lower"] > delta
    return {"fwer_cp_lower": s1["cp_lower"], "fcr_old_lower": s2["lower"], "fwer_fail": bool(fw),
            "fcr_fail": bool(fc), "reproduced": bool(fw or fc)}


# --------------------------------------------------------------------------------------------------- E3 / E-ms(a)
def e3_uncensored(censored_flags, point_thr: float = 0.6, lb_thr: float = 0.5, alpha: float = 0.05) -> dict:
    c = np.asarray(censored_flags).astype(bool)
    n = int(c.size)
    k = int((~c).sum())
    point = k / n if n else float("nan")
    lb = cp_lower_one_sided(k, n, alpha)
    return {"k_uncensored": k, "n": n, "point": point, "cp_lower": lb,
            "pass": bool(point >= point_thr and lb >= lb_thr)}


def completion_lower(rates, thr: float = 0.5, B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
                     alpha: float = 0.05) -> dict:
    """E-ms(a): per-stream completion = correctly certified / 15; one-sided bootstrap lower bound of the mean."""
    r = np.asarray(rates, dtype=float)
    boots = _boot_means(r, B, seed)
    lb = float(np.quantile(boots, alpha))
    return {"mean": float(r.mean()), "lower": lb, "B": B, "seed": seed, "pass": bool(lb >= thr)}


# --------------------------------------------------------------------------------------------------- paired comparisons
def paired_mean_ci(d, B: int = B_DEFAULT, seed: int = SEED_DEFAULT, alpha: float = 0.05,
                   p_side: str = "ge", null: float = 0.0) -> dict:
    """Percentile stream bootstrap of the mean of paired differences d.

    Returns the point mean, the two-sided (1-alpha) CI, and the one-sided bootstrap p value
    p = P*(mean >= null) if p_side == 'ge' (H1: mean < null), or P*(mean <= null) if 'le' (H1: mean > null).
    """
    d = np.asarray(d, dtype=float)
    boots = _boot_means(d, B, seed)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = float(np.mean(boots >= null)) if p_side == "ge" else float(np.mean(boots <= null))
    return {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p_one_sided": p, "null": null,
            "B": B, "seed": seed, "n": int(d.size)}


def paired_log_ratio(n_m, n_x, tau=None, margin: float = math.log(0.8), B: int = B_DEFAULT,
                     seed: int = SEED_DEFAULT, alpha: float = 0.05) -> dict:
    """E2: paired mean log(N_M / N_X) with censored values (None / nan / inf) set to tau.

    Pass iff the 95% CI upper bound < margin (log 0.8). p = P*(mean log ratio >= margin) feeds Holm.
    """
    a = _censor(n_m, tau)
    b = _censor(n_x, tau)
    lr = np.log(a) - np.log(b)
    out = paired_mean_ci(lr, B, seed, alpha, p_side="ge", null=margin)
    out["pass"] = bool(out["ci_hi"] < margin)
    out["margin"] = margin
    return out


def penalized_cost_log_ratio(cost_m, cost_x, done_m, done_x, tau: float, margin: float = math.log(1.15),
                             B: int = B_DEFAULT, seed: int = SEED_DEFAULT, alpha: float = 0.05) -> dict:
    """E-ms(c): paired mean log(cost_M / cost_X). A non-completing method is charged tau; streams where both fail
    enter as the maximal adverse value log(tau / 1) (explicit penalty, never 0). Pass iff CI upper < log 1.15."""
    cm = np.where(np.asarray(done_m, bool), np.asarray(cost_m, float), tau)
    cx = np.where(np.asarray(done_x, bool), np.asarray(cost_x, float), tau)
    lr = np.log(cm) - np.log(cx)
    both_fail = ~np.asarray(done_m, bool) & ~np.asarray(done_x, bool)
    lr = np.where(both_fail, math.log(tau), lr)
    out = paired_mean_ci(lr, B, seed, alpha, p_side="ge", null=margin)
    out["pass"] = bool(out["ci_hi"] < margin)
    out["margin"] = margin
    out["n_both_fail"] = int(both_fail.sum())
    return out


def paired_diff_positive(x_m, x_x, B: int = B_DEFAULT, seed: int = SEED_DEFAULT, alpha: float = 0.05) -> dict:
    """E-ms(b): completion difference (M - X); pass iff the 95% CI lower bound > 0; p = P*(mean <= 0)."""
    d = np.asarray(x_m, float) - np.asarray(x_x, float)
    out = paired_mean_ci(d, B, seed, alpha, p_side="le", null=0.0)
    out["pass"] = bool(out["ci_lo"] > 0)
    return out


def holm(pvals, alpha: float = 0.05) -> dict:
    """Holm step-down. Returns adjusted p values (monotone, capped at 1) and reject flags in input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {"p_adj": adj.tolist(), "reject": (adj <= alpha).tolist(), "alpha": alpha, "m": int(m)}


def _censor(v, tau):
    a = np.array([np.nan if x is None else x for x in np.asarray(v, dtype=object)], dtype=float)
    bad = ~np.isfinite(a)
    if bad.any():
        if tau is None:
            raise ValueError("censored values present but tau is None")
        a[bad] = float(tau)
    if np.any(a <= 0):
        raise ValueError("arrival counts must be positive")
    return a


def reviewer_fcr_counterexample(n_streams: int = 200, n_problems: int = 15, frac_bad: float = 0.05,
                             n_good_cert: int = 8):
    """Per-stream (n_false, n_cert) of the external reviewer counterexample: 5% of streams certify all 15 wrongly, 95% certify
    8 correctly. FWER = E[V/(R v 1)] = 0.05 but FCR_old = 0.75 / 8.35 ~ 0.0898."""
    k = int(round(frac_bad * n_streams))
    nf = np.array([n_problems] * k + [0] * (n_streams - k), dtype=float)
    nc = np.array([n_problems] * k + [n_good_cert] * (n_streams - k), dtype=float)
    return nf, nc
