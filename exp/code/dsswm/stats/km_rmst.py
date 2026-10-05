"""Kaplan-Meier survival of 'not yet certified' and restricted mean survival time (RMST)."""
from __future__ import annotations

import numpy as np


def kaplan_meier(times, censored):
    """times: steps to certification (or censoring time); censored: True if never certified within budget.
    Returns (event_times, survival) as step functions S(t) = P(T > t)."""
    t = np.asarray(times, float)
    c = np.asarray(censored, bool)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    uniq = np.unique(t[~c])
    s, surv = 1.0, []
    for u in uniq:
        at_risk = (t >= u).sum()
        d = ((t == u) & ~c).sum()
        s *= 1 - d / at_risk
        surv.append(s)
    return uniq, np.array(surv)


def rmst(times, censored, tau: float) -> float:
    """Integral of the KM curve on [0, tau]."""
    ut, sv = kaplan_meier(times, censored)
    grid_t = np.concatenate([[0.0], ut[ut < tau], [tau]])
    grid_s = np.concatenate([[1.0], sv[ut < tau]])
    return float(np.sum(np.diff(grid_t) * grid_s))
