"""Discrimination statistics for the Q family: AUC / Delta-AUC / PR-AUC with instance-level cluster bootstrap,
logistic and conditional (stratified) logistic regression, likelihood-ratio tests.

All CIs resample CLUSTERS (environment instances), never rows. Scores: larger = predicted more likely positive.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, rankdata


# ----------------------------------------------------------------------------------------------- point estimates
def auc(scores, labels) -> float:
    """Mann-Whitney AUC with mid-ranks (ties count 1/2); nan if a class is empty."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.where(np.isposinf(s), np.finfo(float).max, np.where(np.isneginf(s), -np.finfo(float).max, s))
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(scores, labels) -> float:
    """Average precision (step-wise, ties handled by grouping equal scores); nan without positives."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / npos
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


# ----------------------------------------------------------------------------------------------- cluster bootstrap
def _groups(clusters):
    clusters = np.asarray(clusters)
    keys, inv = np.unique(clusters, return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(len(keys))]


def cluster_bootstrap_stat(stat_fn, clusters, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """stat_fn(row_index_array) -> float. Percentile CI over resampled clusters (nan replicates dropped)."""
    groups = _groups(clusters)
    allidx = np.concatenate(groups) if groups else np.array([], int)
    est = stat_fn(allidx)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        reps[b] = stat_fn(np.concatenate([groups[p] for p in pick]))
    ok = reps[np.isfinite(reps)]
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if len(ok) else (np.nan, np.nan))
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n_boot_ok": int(len(ok)), "n_clusters": len(groups)}


def auc_ci(scores, labels, clusters, B: int = 2000, seed: int = 0) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    out = cluster_bootstrap_stat(lambda ix: auc(s[ix], y[ix]), clusters, B, seed)
    out.update({"n": int(len(y)), "n_pos": int(y.sum())})
    return out


def pr_auc_ci(scores, labels, clusters, B: int = 2000, seed: int = 0) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    out = cluster_bootstrap_stat(lambda ix: pr_auc(s[ix], y[ix]), clusters, B, seed)
    out.update({"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else float("nan")})
    return out


def delta_auc_ci(scores_a, scores_b, labels, clusters, B: int = 2000, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on the same rows, paired cluster bootstrap."""
    a, b, y = np.asarray(scores_a, float), np.asarray(scores_b, float), np.asarray(labels, bool)
    return cluster_bootstrap_stat(lambda ix: auc(a[ix], y[ix]) - auc(b[ix], y[ix]), clusters, B, seed)


# ----------------------------------------------------------------------------------------------- logistic
def _design(X, intercept=True):
    X = np.asarray(X, float)
    X = X.reshape(len(X), -1) if X.size else np.zeros((len(X), 0))
    return np.column_stack([np.ones(len(X)), X]) if intercept else X


def logistic_fit(X, y, intercept: bool = True, ridge: float = 0.0, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Newton-IRLS MLE. Returns beta, se, loglik, converged. (ridge > 0 only to survive separation.)"""
    A = _design(X, intercept)
    y = np.asarray(y, float)
    beta = np.zeros(A.shape[1])
    conv = False
    for _ in range(max_iter):
        eta = A @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        g = A.T @ (y - p) - ridge * beta
        H = A.T @ (A * W[:, None]) + ridge * np.eye(A.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            conv = True
            break
    eta = A @ beta
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    p = 1.0 / (1.0 + np.exp(-eta))
    H = A.T @ (A * (p * (1 - p))[:, None]) + ridge * np.eye(A.shape[1])
    try:
        se = np.sqrt(np.maximum(np.diag(np.linalg.inv(H)), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    return {"beta": beta, "se": se, "loglik": ll, "converged": conv, "n": int(len(y))}


def lr_test(X_base, X_full, y, intercept: bool = True) -> dict:
    """Nested logistic LR test: 2 (ll_full - ll_base) ~ chi2(df)."""
    f0 = logistic_fit(X_base, y, intercept)
    f1 = logistic_fit(X_full, y, intercept)
    df = _design(X_full, intercept).shape[1] - _design(X_base, intercept).shape[1]
    stat = max(2.0 * (f1["loglik"] - f0["loglik"]), 0.0)
    return {"stat": stat, "df": int(df), "p": float(chi2.sf(stat, df)) if df > 0 else float("nan"),
            "ll_base": f0["loglik"], "ll_full": f1["loglik"], "beta_full": f1["beta"].tolist(),
            "se_full": f1["se"].tolist()}


# ----------------------------------------------------------------------------------------------- conditional logistic
def _clogit_negll(beta, strata):
    """strata: list of (X_s (n_s, d), y_s (n_s,)). Exact conditional likelihood (elementary symmetric DP, log-space)."""
    nll, grad = 0.0, np.zeros_like(beta)
    for X, y in strata:
        eta = X @ beta
        m = int(y.sum())
        n = len(y)
        # log e_j over prefixes and gradient via forward DP on (value, d value)
        logE = np.full(m + 1, -np.inf)
        logE[0] = 0.0
        dE = np.zeros((m + 1, len(beta)))          # d e_j / d beta divided by e_j  (weighted mean of features)
        for i in range(n):
            for j in range(min(i + 1, m), 0, -1):
                a, b = logE[j], logE[j - 1] + eta[i]
                if b == -np.inf:
                    continue
                new = np.logaddexp(a, b)
                wa = 0.0 if a == -np.inf else math.exp(a - new)
                wb = math.exp(b - new)
                dE[j] = wa * dE[j] + wb * (dE[j - 1] + X[i])
                logE[j] = new
        nll -= float(eta[y == 1].sum()) - logE[m]
        grad -= X[y == 1].sum(0) - dE[m]
    return nll, grad


def clogit_fit(X, y, strata_ids) -> dict:
    """Conditional logistic regression (strata = matched sets, e.g. instance x stream). Uninformative strata
    (all 0 or all 1) are dropped. No intercept (absorbed by strata)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, int)
    ids = np.asarray(strata_ids)
    strata = []
    for g in np.unique(ids):
        ix = np.flatnonzero(ids == g)
        if 0 < y[ix].sum() < len(ix):
            strata.append((X[ix], y[ix]))
    d = X.shape[1]
    if not strata or d == 0:
        return {"beta": np.zeros(d), "loglik": 0.0, "n_strata": len(strata), "converged": True}
    res = minimize(lambda b: _clogit_negll(b, strata), np.zeros(d), jac=True, method="BFGS",
                   options={"gtol": 1e-9, "maxiter": 500})
    return {"beta": res.x, "loglik": -float(res.fun), "n_strata": len(strata), "converged": bool(res.success)}


def clogit_lr_test(X_base, X_full, y, strata_ids) -> dict:
    y = np.asarray(y, int)
    Xb = np.asarray(X_base, float).reshape(len(y), -1)
    Xf = np.asarray(X_full, float).reshape(len(y), -1)
    f0 = clogit_fit(Xb, y, strata_ids) if Xb.shape[1] else {"loglik": None}
    f1 = clogit_fit(Xf, y, strata_ids)
    if f0["loglik"] is None:   # empty base model: conditional loglik at beta = 0
        ids = np.asarray(strata_ids)
        strata = [(Xf[np.flatnonzero(ids == g)], y[np.flatnonzero(ids == g)]) for g in np.unique(ids)
                  if 0 < y[ids == g].sum() < (ids == g).sum()]
        f0 = {"loglik": -_clogit_negll(np.zeros(Xf.shape[1]), strata)[0]}
    df = Xf.shape[1] - Xb.shape[1]
    stat = max(2.0 * (f1["loglik"] - f0["loglik"]), 0.0)
    return {"stat": stat, "df": int(df), "p": float(chi2.sf(stat, df)), "beta_full": np.asarray(f1["beta"]).tolist(),
            "n_strata": f1["n_strata"]}
