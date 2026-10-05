"""Lock-v6 addendum, block B: Lenta replication layer ``LR9`` (CR-style budget frontier on the Lenta SMS RCT).

New file (v5-frozen modules untouched). The class reuses the ``PoolReplayEnv`` truth/learner protocol (pools keyed by
(segment, arm), complete outcome-free schedules from ``make_schedule``, WoR replay, billing) and only changes the
population definition:

* table   : $DATA_DIR/lenta/tidy.pkl (sha256 pinned in LENTA_SHA256), 687,029 rows,
            arms 0 = control (24.9 %), 1 = SMS (75.1 %), A = 2.
* outcome : ``response_att`` (binary purchase indicator).
* segments: S = 9 = age tertile x promo-share level, both cut points computed on the FULL table covariates only
            (labels, never outcomes):
              age  : (-inf, 35], (35, 50], (50, inf)           (full-table 1/3, 2/3 quantiles of x_age = 35, 50)
              promo: x_promo_share_15d == 0 | 0 < p <= m | p > m, m = median of the positive values (full table)
            label = 3 * age_level + promo_level.  This segmentation was never used before (r4's LR8 used age
            quartile x main_format; LR8 full-table cell means exist in r4 artifacts -- disclosed in the addendum).
* split   : NEW split seed 6006 (r4 used 4242), stratified 50/50 by (LR9 segment x arm) with the frozen
            ``pool_replay.split_mask`` rule.  dev half = development (eps*, tuning, pilots); eval half = confirmatory.
* problems: the CR family transposed: kappa = (0, 1), budgets 0.10, 0.15, ..., 0.80 (15 problems), Q = 15.
* checkpoints: K = 20 log-spaced from n_min = 5,000 to tau_R (= rows of the half).

Eval guard (v6 review item 5):
* the eval half is STRUCTURE-ONLY unless the constructor receives ``eval_task_id`` AND
  ``dsswm.stats.prereg_v6.addendum_gate(eval_task_id)`` authorises it (locked v6 addendum, task listed, hashes ok);
  there is no other bypass (no ``allow_eval`` flag).
* loading: pandas reads the whole pickle into memory (it cannot read columns selectively), so outcomes are LOADED
  transiently; this module never caches the outcome column (``pool_replay.load_table``'s shared cache is NOT used),
  keeps only covariates + treatment for structure-only objects, and for the dev half keeps dev-row outcomes only.
  Accurate wording: "eval-half LR9 outcomes are not exposed through any evaluation interface before the lock",
  not "never loaded".
* honest exposure statement: r4 (r4_setup_pool_replay) computed FULL-table LR8 cell means, which include the rows of
  today's LR9 eval half; a new segmentation and split seed do NOT restore an untouched holdout.  Block B is therefore
  a descriptive re-use of an outcome-exposed log.
* the table sha256 is verified by default (check_sha=True).
"""
from __future__ import annotations

import numpy as np

from ..streams import frontier as fr
from .base import TruthAccessError
import pandas as pd

from .pool_replay import DATA_PATHS, PoolReplayEnv, sha256_file, split_mask

__all__ = ["LentaLayerEnv", "lr9_labels", "lr9_problems", "LR9_SPLIT_SEED", "LR9_BUDGETS", "LR9_N_MIN",
           "LENTA_SHA256", "LR9_OUTCOME"]

LENTA_SHA256 = "6d075bbc306df3512f6bc3a795c94686984b4c99da3889c91778d25325565d43"
LR9_SPLIT_SEED = 6006
LR9_BUDGETS = fr.CR_BUDGETS                      # 0.10 ... 0.80, 15 problems
LR9_N_MIN = 5_000
LR9_OUTCOME = "response_att"
AGE_LEVEL_DESC = ("age<=35", "35<age<=50", "age>50")
PROMO_LEVEL_DESC = ("promo=0", "0<promo<=m", "promo>m")


def lr9_cutpoints(df):
    """Covariate-only cut points on the full table: age tertile edges and the positive-promo median."""
    age = df["x_age"].to_numpy().astype(float)
    a1, a2 = (float(v) for v in np.quantile(age, [1.0 / 3.0, 2.0 / 3.0]))
    p = df["x_promo_share_15d"].to_numpy().astype(float)
    m = float(np.median(p[p > 0]))
    return {"age_edges": (a1, a2), "promo_pos_median": m}


def lr9_labels(df, cut=None):
    cut = lr9_cutpoints(df) if cut is None else cut
    age = df["x_age"].to_numpy().astype(float)
    a_lvl = np.searchsorted(np.asarray(cut["age_edges"]), age, side="left").astype(np.int64)   # <=a1 -> 0
    p = df["x_promo_share_15d"].to_numpy().astype(float)
    p_lvl = np.where(p <= 0, 0, np.where(p <= cut["promo_pos_median"], 1, 2)).astype(np.int64)
    lab = 3 * a_lvl + p_lvl
    desc = [f"{AGE_LEVEL_DESC[s // 3]}|{PROMO_LEVEL_DESC[s % 3]}" for s in range(9)]
    return lab, desc, cut


def lr9_problems(outcome=LR9_OUTCOME):
    return [fr.Problem(f"{outcome}|c1|B{b:.2f}", outcome, (0.0, 1.0), b) for b in LR9_BUDGETS]


_COV_CACHE: dict = {}
_COVS = ("x_age", "x_promo_share_15d", "treatment")


def _load_lenta(check_sha, with_outcome):
    """Covariates + treatment (cached, outcome-free); the outcome column is returned only when requested and is
    never cached."""
    if check_sha and sha256_file(DATA_PATHS["lenta"]) != LENTA_SHA256:
        raise RuntimeError("lenta tidy.pkl sha256 differs from the pinned value")
    if not with_outcome and "cov" in _COV_CACHE:
        return _COV_CACHE["cov"], None
    df = pd.read_pickle(DATA_PATHS["lenta"])
    cov = df[list(_COVS)].copy()
    y = df[LR9_OUTCOME].to_numpy().astype(float).copy() if with_outcome else None
    del df
    _COV_CACHE.setdefault("cov", cov)
    return _COV_CACHE["cov"], y


class LentaLayerEnv(PoolReplayEnv):
    """LR9 finite population (one half of the Lenta table). Interface identical to ``PoolReplayEnv``."""

    def __init__(self, half="dev", eval_task_id=None, gate_path=None, check_sha=True):  # noqa: D401
        if half not in ("dev", "eval"):
            raise ValueError("LR9 has only 'dev' and 'eval' halves (no 'full': it would expose eval aggregates)")
        authorised = False
        if half == "eval" and eval_task_id is not None:
            from ..stats.prereg_v6 import addendum_gate
            ok, info = addendum_gate(eval_task_id, path=gate_path)
            if not ok:
                raise TruthAccessError(f"LR9 eval outcomes refused by the v6 addendum gate: {info}")
            if not str(eval_task_id).startswith("v6b_"):
                raise TruthAccessError("LR9 eval outcomes are reserved for block-B eval tasks")
            authorised = True
        need_y = half == "dev" or authorised
        df, y_full = _load_lenta(check_sha, with_outcome=need_y)
        self.layer, self.half, self.split_seed = "LR9", half, LR9_SPLIT_SEED
        self.A = 2
        seg_full, self.seg_desc, self.cut = lr9_labels(df)
        self.S = len(self.seg_desc)
        arm_full = df["treatment"].to_numpy().astype(np.int64)
        dev = split_mask(seg_full, arm_full, LR9_SPLIT_SEED)
        mask = dev if half == "dev" else ~dev
        self.row_index = np.flatnonzero(mask)
        self.seg = seg_full[mask]
        self.arm = arm_full[mask]
        self.structure_only = not need_y
        self._y = {}
        self.clip = {}
        if need_y:
            v = y_full[mask].copy()
            v.setflags(write=False)
            self._y[LR9_OUTCOME] = v
        del y_full
        self.binary_outcomes = (LR9_OUTCOME,)
        self.N = int(mask.sum())
        self.cell = self.seg * self.A + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=self.S * self.A).reshape(self.S, self.A)
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(self.S * self.A)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = LR9_N_MIN
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)

    def _need_outcomes(self):
        if self.structure_only:
            raise TruthAccessError("LR9 eval half is structure-only (no authorised eval_task_id)")

    def true_mu(self, outcome=LR9_OUTCOME):
        self._need_outcomes()
        return super().true_mu(outcome)

    def true_sigma2(self, outcome=LR9_OUTCOME):
        self._need_outcomes()
        return super().true_sigma2(outcome)

    def true_pooled_mu(self, outcome=LR9_OUTCOME):
        self._need_outcomes()
        return super().true_pooled_mu(outcome)

    def outcomes_view(self, outcome=LR9_OUTCOME):
        self._need_outcomes()
        return self._y[outcome]

    def truth_answers(self, problems, eps_grid=None):
        self._need_outcomes()
        return super().truth_answers(problems, eps_grid=eps_grid)
