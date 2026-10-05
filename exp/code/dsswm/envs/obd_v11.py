"""Open Bandit Dataset (ZOZOTOWN, random/all, uniform-random logging) feasibility env (v11 retarget, NEW file; DEV ONLY).

Data discipline.  The salted split lives in $DATA_DIR/open_bandit/ (PROVENANCE.json:
salt 'dsswm-obd-2026-10-05', row_key = CSV row index).  This module reads dev.pkl only; there is no eval option.

Mapping to the paper's setting (group level, segment x arm):
* row = one logged (impression, position) recommendation with a uniformly random item (propensity 1/80 at each
  position); outcome = click.
* segments: covariate-only user-feature cells (frozen value lists, see SEGMENTATIONS), never outcome-informed.
* arms: item groups.  arm a shows a uniformly random item of group a.  A2: control = the 40 items with the lowest
  dev click rate, treatment = the 40 highest (dev-outcome-informed design choice; eval gap shrinks by selection).
  A3: tertiles of dev item click rate (26/27/27 items), arm 0 = lowest (control).
* costs: kappa[s, a] = a * round(G * S * w_s) (G = 8): exposure / traffic cost, heterogeneous through segment size;
  budgets B_q = floor(b_q * sum_s kappa[s, 1]) for b_q = 0.10..0.80 (15 problems) as in seg_v10.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv

OBD_ROOT = __import__("os").environ.get("DATA_DIR", "data") + "/open_bandit/"
UF = ("user_feature_0", "user_feature_1", "user_feature_2", "user_feature_3")
COST_G = 8
_DEV = {}


def obd_dev():
    if "df" not in _DEV:
        _DEV["df"] = pd.read_pickle(OBD_ROOT + "dev.pkl")
    return _DEV["df"]


def item_groups(df, A):
    """Item -> arm map by dev item click-rate rank (ties by item id); arm 0 = lowest group."""
    ctr = df.groupby("item_id").click.mean()
    order = sorted(ctr.index, key=lambda i: (ctr[i], i))
    grp = {}
    for a, chunk in enumerate(np.array_split(np.array(order), A)):
        for i in chunk:
            grp[int(i)] = a
    return grp


def segmentation(df, kind):
    """Covariate-only segment labels + descriptions.  kinds:
    'uf0'      : user_feature_0 levels (tiny level < 1% merged into the largest)
    'uf0x3'    : user_feature_0 x user_feature_3 cells, cells < 2 % merged into their uf0 'other' cell
    'combo8'   : 7 most frequent (uf0..uf3) tuples + other;  'combo16': 15 + other."""
    n = len(df)
    if kind.startswith("combo"):
        K = int(kind[5:])
        key = df[list(UF)].astype(str).agg("|".join, axis=1)
        vc = key.value_counts()
        keep = list(vc.index[:K - 1])
        code = {c: i for i, c in enumerate(keep)}
        lab = key.map(lambda x: code.get(x, K - 1)).to_numpy()
        return lab.astype(np.int64), [f"c{i}" for i in range(K - 1)] + ["other"]
    if kind == "uf0":
        u = df["user_feature_0"].astype(str)
        vc = u.value_counts()
        big = vc.index[0]
        u = u.where(u.map(vc) >= 0.01 * n, big)
        levels = list(u.value_counts().index)
        return u.map({c: i for i, c in enumerate(levels)}).to_numpy().astype(np.int64), [c[:6] for c in levels]
    if kind == "uf0x3":
        u0 = df["user_feature_0"].astype(str)
        k = u0 + "|" + df["user_feature_3"].astype(str)
        vc = k.value_counts()
        k = k.where(k.map(vc) >= 0.02 * n, u0 + "|other")
        vc = k.value_counts()
        # merge any residual tiny cell (< 1 %) into the largest cell
        k = k.where(k.map(vc) >= 0.01 * n, vc.index[0])
        levels = list(k.value_counts().index)
        return k.map({c: i for i, c in enumerate(levels)}).to_numpy().astype(np.int64), [c[:6] + c.split("|")[1][:4]
                                                                                            for c in levels]
    raise ValueError(kind)


def obd_problems(w, A, budgets=fr.CR_BUDGETS, G=COST_G):
    from ..baselines.fdc_dp import SegProblems
    S = len(w)
    k1 = np.rint(G * S * np.asarray(w, float)).astype(np.int64)
    k1 = np.maximum(k1, 1)
    cost = np.stack([a * k1 for a in range(A)], 1)
    tot = int(k1.sum())
    B = np.array([int(math.floor(b * tot + 1e-9)) for b in budgets], dtype=np.int64)
    return SegProblems(cost=cost, budgets=B, qids=[f"OBD|A{A}|B{b:.2f}" for b in budgets], budget_frac=list(budgets))


class OBDEnv(SyntheticPoolEnv):
    """Replay population = the whole OBD dev half (PoolReplayEnv interface; outcome name 'visit')."""

    def __init__(self, seg_kind="uf0x3", A=2, half="dev", n_min=5000, holdout=False):  # noqa: D401
        """holdout=True (shrinkage check): item groups AND segment value lists are built on the even-row_id half of
        dev ('dev-A'); the replay pool is the odd-row_id half ('dev-B'), so the pool never informed the arm design
        (mimics the eval use)."""
        if half != "dev":
            raise PermissionError("obd_v11 is dev-only")
        df = obd_dev()
        if holdout:
            ra = (df["row_id"].to_numpy() % 2) == 0
            grp = item_groups(df[ra], A)
            df = df[~ra].reset_index(drop=True)
        else:
            grp = item_groups(df, A)
        self.seg, self.seg_desc = segmentation(df, seg_kind)
        self.arm = df["item_id"].map(grp).to_numpy().astype(np.int64)
        self.layer, self.half, self.split_seed = f"OBD-{seg_kind}-A{A}", "dev", 0
        self.S, self.A = int(self.seg.max() + 1), int(A)
        y = df["click"].to_numpy(dtype=float).copy()
        y.setflags(write=False)
        self._y = {"visit": y}
        self.clip = {}
        self.binary_outcomes = ("visit",)
        self.N = int(len(self.seg))
        self.row_index = np.arange(self.N)
        self.cell = self.seg * self.A + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=self.S * self.A).reshape(self.S, self.A)
        if (self.pool_sizes <= 0).any():
            raise RuntimeError("empty pool")
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(self.S * self.A)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = int(n_min)
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)
        self.problems = obd_problems(self.w, self.A)
