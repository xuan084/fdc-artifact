"""Instance-level (cluster) bootstrap CIs; paired by construction when the statistic uses matched rows."""
from __future__ import annotations

import numpy as np


def cluster_bootstrap(values_by_cluster: dict, stat_fn, B: int = 2000, alpha: float = 0.05, seed: int = 0):
    """values_by_cluster: {cluster_id: array of rows}; stat_fn(list_of_arrays) -> float."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_cluster)
    est = stat_fn([values_by_cluster[k] for k in keys])
    boots = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        boots[b] = stat_fn([values_by_cluster[keys[p]] for p in pick])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


def proportion_ci_by_cluster(flags_by_cluster: dict, B: int = 2000, seed: int = 0):
    return cluster_bootstrap(flags_by_cluster, lambda arrs: float(np.mean(np.concatenate(arrs))), B=B, seed=seed)
