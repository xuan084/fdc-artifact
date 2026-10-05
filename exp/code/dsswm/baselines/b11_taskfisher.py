"""B11: task-directed Fisher sampling (Wagenmaker et al., arXiv 2306.09210 style) under the joint LR stopping rule.

EQUIVALENCE RECORD (r2_setup_offgrid): B11 is the same estimator as the round-0 B5 variant `joint_TaskDirected`
(run_nl_acquisition_factorial.py, decide_fisher with crit = "T"): the design greedily minimises the largest
task-relevant variance  max_{k != k_hat} (g_k - g_khat)^T M^{-1} (g_k - g_khat) / (gap_k + eps)^2  (gapnorm) over
the candidate element library, per unit cost, where g_k = dJ_theta_hat(pi_k)/dv and M = V + F(element).
This module is that criterion extracted verbatim as a pure function so round-1 runners can reuse it; the
round-0 tuning grid (<= 5 points) and selected configuration are reused, not re-tuned. B11 shares the joint
LR evidence with JPC, so it is a sampling-rule row (HS1), never an HR1 denominator.
"""
from __future__ import annotations

import numpy as np

TUNE_GRID = [{"crit": "T", "gapnorm": True, "look": 1, "explore": 0.0},
             {"crit": "T", "gapnorm": True, "look": 2, "explore": 0.0},
             {"crit": "T", "gapnorm": False, "look": 1, "explore": 0.0},
             {"crit": "T", "gapnorm": False, "look": 2, "explore": 0.0},
             {"crit": "T", "gapnorm": True, "look": 2, "explore": 0.1}]
EQUIVALENT_TO = "run_nl_acquisition_factorial.py::joint_TaskDirected (B5, crit='T')"


def task_directed_gain(Vd: np.ndarray, Fel: np.ndarray, G: np.ndarray, J_hat: np.ndarray, k_hat: int,
                       gapnorm: bool, eps: float) -> np.ndarray:
    """Reduction of the worst task-relevant variance for each candidate element (n_el,). Same formula as B5-T."""
    K = G.shape[0]
    others = [k for k in range(K) if k != k_hat]
    Gd = G[others] - G[k_hat][None]
    if gapnorm:
        gap = J_hat[k_hat] - J_hat[others]
        Gd = Gd / (gap + eps)[:, None]
    f0 = np.einsum("kd,de,ke->k", Gd, np.linalg.inv(Vd), Gd).max()
    Mi = np.linalg.inv(Vd[None] + Fel)
    f1 = np.einsum("kd,nde,ke->nk", Gd, Mi, Gd).max(1)
    return f0 - f1


def choose(Vd, Fel, cost, G, J_hat, k_hat, gapnorm, eps, rng=None, explore=0.0) -> int:
    """Index of the chosen element (argmax gain / cost), with optional uniform exploration."""
    if rng is not None and explore > 0 and rng.random() < explore:
        return int(rng.integers(len(cost)))
    gain = task_directed_gain(Vd, Fel, G, J_hat, k_hat, gapnorm, eps)
    return int(np.argmax(gain / np.asarray(cost, float)))
