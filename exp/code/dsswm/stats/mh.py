"""Stratified Mantel-Haenszel risk ratio with an instance-level cluster bootstrap (round 3, C1).

RR_MH = sum_s a_s n0_s / N_s  /  sum_s c_s n1_s / N_s
  a_s  events among exposed in stratum s (e.g. false certificates among zero-cost certificates)
  n1_s exposed count, c_s events among unexposed, n0_s unexposed count, N_s = n1_s + n0_s.
Strata with N_s = 0 contribute nothing. Returns inf if the denominator is 0 and the numerator > 0, nan if both 0.
Cluster bootstrap: resample clusters (instances) with replacement, recompute RR_MH on the pooled rows of the drawn
clusters (a cluster drawn twice contributes twice); percentile CI over finite replicates (the number of
non-finite replicates is reported).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def mh_counts(rows, exposure_key, event_key, strata_keys):
    """rows: iterable of dicts -> {stratum: [a, n1, c, n0]}."""
    tab = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        s = tuple(r[k] for k in strata_keys)
        e, y = bool(r[exposure_key]), bool(r[event_key])
        t = tab[s]
        if e:
            t[0] += y
            t[1] += 1
        else:
            t[2] += y
            t[3] += 1
    return dict(tab)


def mh_rr_from_counts(tab) -> float:
    num = den = 0.0
    for a, n1, c, n0 in tab.values():
        N = n1 + n0
        if N == 0:
            continue
        num += a * n0 / N
        den += c * n1 / N
    if den == 0:
        return math.inf if num > 0 else math.nan
    return num / den


def mh_risk_ratio(rows, exposure_key, event_key, strata_keys=()) -> float:
    return mh_rr_from_counts(mh_counts(rows, exposure_key, event_key, strata_keys))


def cluster_bootstrap(rows, cluster_key, stat_fn, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Generic cluster bootstrap of stat_fn(list_of_rows)."""
    by = defaultdict(list)
    for r in rows:
        by[r[cluster_key]].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    point = stat_fn([r for k in keys for r in by[k]])
    reps = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(len(keys), size=len(keys))
        reps[b] = stat_fn([r for i in draw for r in by[keys[i]]])
    fin = reps[np.isfinite(reps)]
    lo, hi = (np.quantile(fin, [alpha / 2, 1 - alpha / 2]).tolist() if len(fin) else (math.nan, math.nan))
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_clusters": len(keys),
            "n_boot": n_boot, "n_nonfinite": int(n_boot - len(fin))}


def stratified_mh_rr(rows, exposure_key, event_key, strata_keys, cluster_key="instance", n_boot=2000, seed=42,
                     alpha=0.05) -> dict:
    out = cluster_bootstrap(rows, cluster_key, lambda rs: mh_risk_ratio(rs, exposure_key, event_key, strata_keys),
                            n_boot=n_boot, seed=seed, alpha=alpha)
    out["strata"] = {"|".join(map(str, k)): v for k, v in mh_counts(rows, exposure_key, event_key, strata_keys).items()}
    return out
