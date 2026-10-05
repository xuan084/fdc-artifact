"""X5 RetailHero layer (round-8 feasibility; NEW file).  DEV HALF ONLY until a v8 lock exists.

Split: blinded hashed split from ``ingest_v8_fresh.py`` (dev.pkl / eval_labels.pkl / eval_outcome.npy).  The dev half
is always available.  The eval half (lock-v8 block A) is released ONLY to an eval task id that passes
``dsswm.stats.prereg_v8.addendum_gate`` and names a block-A v8 eval task (prefix ``v8a_full``); the three X5 files must
also match the sha256 recorded by the blinded ingest (PROVENANCE.json).  Otherwise PermissionError: no eval outcome can
be read by accident.  The eval loader only builds the finite population (pools, cell means for the truth side); it
computes no statistic beyond what the stream engine and the truth accessors need.

Segmentation X9 (features only, fixed 2026-10-04 before any certification run): gender {U, F, M} x age band
{<38 or invalid (age outside [14, 100]) -> band 0 if < 38, band 1 if invalid or in [38, 53), band 2 if >= 53}.
Outcome 'visit' (public name reused so the CR problem family applies) = X5 target (purchase after the SMS campaign).
Problems: fr.cr_problems (treatment cost 1, budgets 0.10..0.80) -- 15 problems, 512 policies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv

ROOT = (Path(__import__("os").environ.get("DATA_DIR", "data")) / "x5_retailhero")
GENDERS = ("U", "F", "M")


def x9_labels(df):
    g = df["gender"].map({k: i for i, k in enumerate(GENDERS)}).fillna(0).astype(int).to_numpy()
    age = df["age"].to_numpy(dtype=float)
    valid = (age >= 14) & (age <= 100)
    band = np.where(~valid, 1, np.where(age < 38, 0, np.where(age < 53, 1, 2)))
    return (g * 3 + band).astype(np.int64)


class X5LayerEnv(SyntheticPoolEnv):
    def __init__(self, half="dev", n_min=2000, eval_task_id=None, gate_path=None):  # noqa: D401 - no super().__init__
        if half == "dev":
            df = pd.read_pickle(ROOT / "dev.pkl")
            y = df["outcome"].to_numpy().astype(float)
        elif half == "eval":
            if eval_task_id is None:
                raise PermissionError("X5 eval half: an authorised v8 block-A eval task id is required")
            from ..stats.prereg_v8 import addendum_gate
            ok, info = addendum_gate(eval_task_id, path=gate_path)
            if not ok:
                raise PermissionError(f"X5 eval half refused by the v8 addendum gate: {info}")
            if not str(eval_task_id).startswith("v8a_full"):
                raise PermissionError("X5 eval outcomes are reserved for v8 block-A eval tasks")
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
