"""X5 RetailHero layer for lock v9 block B (NEW file; x5_v8.py is locked and untouched).

Same population, segmentation (X9) and outcome as ``x5_v8.X5LayerEnv``.  The dev half is the same dev half.  The eval
half is released ONLY to an eval task id that passes ``dsswm.stats.prereg_v9.addendum_gate`` and names a block-B v9
eval task (prefix ``v9b_full``); the X5 files must match PROVENANCE.json.  Disclosure (lock v9): this eval half was
used once before, by lock-v8 block A (seeds 35000-35199).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from .x5_v8 import GENDERS, ROOT, X5LayerEnv, x9_labels

__all__ = ["X5V9Env"]


class X5V9Env(X5LayerEnv):
    def __init__(self, half="dev", n_min=2000, eval_task_id=None, gate_path=None):  # noqa: D401 - no super().__init__
        if half == "dev":
            df = pd.read_pickle(ROOT / "dev.pkl")
            y = df["outcome"].to_numpy().astype(float)
        elif half == "eval":
            if eval_task_id is None:
                raise PermissionError("X5 eval half (v9): an authorised v9 block-B eval task id is required")
            try:
                from ..stats.prereg_v9 import addendum_gate
            except ImportError as e:
                raise PermissionError(f"X5 eval half (v9): no v9 addendum gate available ({e})") from None
            ok, info = addendum_gate(eval_task_id, path=gate_path)
            if not ok:
                raise PermissionError(f"X5 eval half refused by the v9 addendum gate: {info}")
            if not str(eval_task_id).startswith("v9b_full"):
                raise PermissionError("X5 eval outcomes (v9) are reserved for v9 block-B eval tasks")
            from .data_v8 import x5_provenance_check
            x5_provenance_check()
            df = pd.read_pickle(ROOT / "eval_labels.pkl")
            y = np.load(ROOT / "eval_outcome.npy").astype(float)
            if len(y) != len(df):
                raise RuntimeError("X5 eval outcome / labels length mismatch")
        else:
            raise ValueError(half)
        seg = x9_labels(df)
        arm = df["treatment"].to_numpy().astype(np.int64)
        self.layer, self.half, self.split_seed = "X9", half, None
        self.S, self.A = 9, 2
        self.seg_desc = [f"{g}|age{b}" for g in GENDERS for b in range(3)]
        self.seg, self.arm = seg, arm
        y.setflags(write=False)
        self._y = {"visit": y}
        self.clip = {}
        self.binary_outcomes = ("visit",)
        self.N = int(len(seg))
        self.row_index = np.arange(self.N)
        self.cell = seg * 2 + arm
        self.pool_sizes = np.bincount(self.cell, minlength=18).reshape(9, 2)
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(18)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = int(n_min)
        self.eps_grid = (0.005, 0.0075, 0.01, 0.015)
        self.replan_interval = fr.replan_interval(self.tau_R)
