"""POST-HOC, DESCRIPTIVE eval-half access for block v10-posthoc-G (critic r5 response).  NEW file; seg_v10_eval.py,
seg_v10.py and prereg_v10.py are imported, never modified.

The v10 eval reader (seg_v10_eval.SegEnvV10Frozen / read_eval_rows) releases an eval half only to a task id that the
locked v10 addendum registers for exactly that layer.  Block v10-posthoc-G runs AFTER the v10 confirmatory analysis,
is not a lock task and must not masquerade as one, so this module provides a separate, logged path (pattern of
posthoc_access_v8):

  1. the v10 addendum must be locked and verify (prereg_v10.load_locked_addendum: hash, code / input / data drift,
     single-commit history), i.e. the confirmatory design is frozen and cannot be influenced by this block;
  2. every confirmatory v10 eval task on the requested table must already be SEALED (seal + seal_ref present):
     X5 -> v10a_full_s16, v10a_full_s32, v10d_full_x5s64; Lenta -> v10b_full_s16, v10b_full_s32, v10b_full_s64;
  3. the task id must start with 'v10-posthoc-G' (never a lock task id);
  4. the declared stream seeds must lie in the fresh post-hoc range 38400-38999 (38000-38399 are the lock-v10 eval
     seeds and are refused);
  5. every read appends one JSON line to exp/results/full/v10_posthoc_G/eval_access_log.jsonl (timestamp, task,
     layer, seeds, git HEAD, addendum sha256, reason, label 'post hoc, descriptive') BEFORE any eval byte is read.

Nothing produced through this path is confirmatory; rows carry 'posthoc': true.  The population built here is the
same as SegEnvV10Frozen(data, S, 'eval') (frozen score model, cut points and problems; whole eval half = replay
population); ``check_digest_against_seal`` compares its env_digest with the sealed lock-v10 eval rows of the layer.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv
from . import seg_v10 as SV
from . import seg_v10_eval as E

WS = Path(__file__).resolve().parents[4]
LOG = WS / "exp/results/full/v10_posthoc_G/eval_access_log.jsonl"
SEALS = WS / "exp/results/full/v10_seals"
REQUIRED_SEALS = {"x5": ("v10a_full_s16", "v10a_full_s32", "v10d_full_x5s64"),
                  "lenta": ("v10b_full_s16", "v10b_full_s32", "v10b_full_s64")}
SEALED_TASK_FOR_LAYER = {("x5", 16): "v10a_full_s16", ("x5", 32): "v10a_full_s32", ("x5", 64): "v10d_full_x5s64",
                         ("lenta", 16): "v10b_full_s16", ("lenta", 32): "v10b_full_s32",
                         ("lenta", 64): "v10b_full_s64"}
POSTHOC_SEEDS = range(38400, 39000)
LOCK_EVAL_SEEDS = range(38000, 38400)
LABEL = "post hoc, descriptive"


def _head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(WS), capture_output=True)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def posthoc_gate(task_id: str, data: str, S: int, seeds, reason: str, log_path: Path | None = None) -> dict:
    if not str(task_id).startswith("v10-posthoc-G"):
        raise PermissionError("post-hoc v10 eval access requires a task id starting with 'v10-posthoc-G'")
    if data not in REQUIRED_SEALS:
        raise PermissionError(f"unknown v10 table {data!r}")
    seeds = [int(s) for s in seeds]
    if not seeds or any(s not in POSTHOC_SEEDS for s in seeds) or any(s in LOCK_EVAL_SEEDS for s in seeds):
        raise PermissionError("post-hoc v10 eval streams must use fresh seeds in 38400-38999 only")
    from ..stats.prereg_v10 import load_locked_addendum
    lock = load_locked_addendum(None)
    for t in REQUIRED_SEALS[data]:
        if not ((SEALS / f"{t}.seal.json").exists() and (SEALS / f"{t}.seal_ref.json").exists()):
            raise PermissionError(f"post-hoc access to {data} eval refused: confirmatory task {t} is not sealed")
    rec = {"at": datetime.now().isoformat(), "task_id": task_id, "layer": E.layer_name(data, S), "half": "eval",
           "seeds": [min(seeds), max(seeds)], "n_seeds": len(seeds), "git_head": _head(),
           "v10_addendum_sha256": lock["sha256"], "required_seals": list(REQUIRED_SEALS[data]),
           "status": "post hoc, descriptive (outside any lock)", "label": LABEL, "reason": reason}
    p = Path(log_path) if log_path is not None else LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
    return {"record": rec, "lock": lock}


class SegEnvV10PosthocEval(SyntheticPoolEnv):
    """Frozen v10 eval-half population for a post-hoc descriptive task (gate + log above, then the private raw
    reader).  Same construction as SegEnvV10Frozen(data, S, 'eval')."""

    def __init__(self, data, S, task_id, seeds, reason, log_path=None):  # noqa: D401 - no super().__init__
        acc = posthoc_gate(task_id, data, S, seeds, reason, log_path)    # gate + log first, then any read
        self.posthoc_access = acc["record"]
        lock = acc["lock"]
        X, arm, y, rows = E._read_eval_rows_raw(data)
        fz = E.load_frozen_seg(None, lock["frozen_segmentation"]["seg_v10_frozen.json"])
        models = E._load_models(fz, data)
        score = E._score(models, X)
        cuts = [float.fromhex(c) for c in fz["data"][data]["S"][str(int(S))]["cuts_hex"]]
        layer = E.layer_name(data, S)
        self.data, self.layer, self.half, self.split_seed = data, layer, "eval", SV.MODEL_SPLIT_SEED
        self.S, self.A = int(S), 2
        self.seg = E.assign_segments(score, cuts)
        self.arm = np.asarray(arm, dtype=np.int64)
        self.seg_desc = [f"q{s}/{S}" for s in range(S)]
        y = np.asarray(y, dtype=float).copy()
        y.setflags(write=False)
        self._y = {"visit": y}
        self.clip = {}
        self.binary_outcomes = ("visit",)
        self.N = int(len(self.seg))
        self.row_index = np.asarray(rows)
        self.cell = self.seg * 2 + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=2 * S).reshape(S, 2)
        if (self.pool_sizes <= 0).any():
            raise RuntimeError(f"{layer} (eval): empty pool under the frozen segmentation")
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(2 * S)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = 5000 if data == "lenta" else 2000
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)
        self.problems = E.frozen_problems(fz, data, S)
        self.score_sha256 = E._sha_arr(score)


def check_digest_against_seal(env, results_root=None):
    """The post-hoc population must be the sealed lock-v10 population of the same layer (env_digest equality)."""
    from .seg_v10 import env_digest
    t = SEALED_TASK_FOR_LAYER[(env.data, int(env.S))]
    root = Path(results_root) if results_root is not None else WS / "exp/results/full"
    with open(root / t / "results.jsonl") as f:
        sealed = json.loads(f.readline())["env_digest"]
    got = env_digest(env)
    if got != sealed:
        raise RuntimeError(f"post-hoc {env.layer} population digest {got} != sealed {t} digest {sealed}")
    return {"sealed_task": t, "env_digest": got}
