"""Documented POST-HOC (post-lock, descriptive) eval-half access for block v8-posthoc-D.  NEW file; x5_v8.py,
pool_replay.py and prereg_v8.py are imported, never modified.

The X5 eval loader (x5_v8.X5LayerEnv) releases the eval half only to the locked v8 block-A task ids.  A post-hoc,
descriptive block run AFTER the v8 confirmatory analysis is not a lock task and must not masquerade as one, so this
module provides a separate, logged path:

  1. the v8 addendum must be locked and verify (prereg_v8.load_locked_addendum: hash, code / input / data drift,
     single-commit history), i.e. the confirmatory design is frozen and cannot be influenced by this block;
  2. the confirmatory eval tasks whose streams are re-used must already be SEALED (v8a_full_a for X5, v8c_full for CR9
     eval) -- their results are git-committed before any post-hoc row exists;
  3. the task id must start with 'v8-posthoc-' (never a lock task id);
  4. every access appends one JSON line to exp/results/full/v8_posthoc_D/eval_access_log.jsonl (timestamp, task, layer,
     git HEAD, addendum sha256, reason).

Nothing produced through this path is confirmatory; rows carry 'posthoc': true.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .x5_v8 import GENDERS, ROOT, X5LayerEnv, x9_labels  # noqa: F401

WS = Path(__file__).resolve().parents[4]
LOG = WS / "exp/results/full/v8_posthoc_D/eval_access_log.jsonl"
SEALS = WS / "exp/results/full/v8_seals"
REQUIRED_SEAL = {"X9": "v8a_full_a", "CR9": "v8c_full"}


def _head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(WS), capture_output=True)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def posthoc_gate(task_id: str, layer: str, reason: str, log_path: Path | None = None) -> dict:
    if not str(task_id).startswith("v8-posthoc-"):
        raise PermissionError("post-hoc eval access requires a task id starting with 'v8-posthoc-'")
    from ..stats.prereg_v8 import load_locked_addendum
    lock = load_locked_addendum(None)
    seal = REQUIRED_SEAL[layer]
    if not ((SEALS / f"{seal}.seal.json").exists() and (SEALS / f"{seal}.seal_ref.json").exists()):
        raise PermissionError(f"post-hoc access to {layer} eval refused: confirmatory task {seal} is not sealed")
    rec = {"at": datetime.now().isoformat(), "task_id": task_id, "layer": layer, "half": "eval",
           "git_head": _head(), "v8_addendum_sha256": lock["sha256"], "required_seal": seal,
           "status": "post-hoc descriptive (outside any lock)", "reason": reason}
    p = Path(log_path) if log_path is not None else LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


class X5PosthocEvalEnv(X5LayerEnv):
    """X5 X9 eval half for a post-hoc descriptive task (gate + log above); same population build as X5LayerEnv."""

    def __init__(self, task_id, reason, n_min=2000, log_path=None):  # noqa: D401 - no super().__init__
        self.posthoc_access = posthoc_gate(task_id, "X9", reason, log_path)
        from .data_v8 import x5_provenance_check
        x5_provenance_check()
        df = pd.read_pickle(ROOT / "eval_labels.pkl")
        y = np.load(ROOT / "eval_outcome.npy").astype(float)
        if len(y) != len(df):
            raise RuntimeError("X5 eval outcome / labels length mismatch")
        from ..streams import frontier as fr
        seg = x9_labels(df)
        arm = df["treatment"].to_numpy().astype(np.int64)
        self.layer, self.half, self.split_seed = "X9", "eval", None
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


def cr9_posthoc_eval_env(task_id, reason, log_path=None):
    """CR9 eval half (exposed since r5; streams 33000-33199 used by the sealed v7 block A / v8 block C)."""
    posthoc_gate(task_id, "CR9", reason, log_path)
    from .pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", "eval")
    env.posthoc_access = True
    return env
