"""Lock v12: frozen Open Bandit designs for the two unread campaigns random/women (block A) and random/men (block B),
and GATED eval-half access (NEW file; plan/v12_obd2_plan.md).  obd_v11.py and obd_v11_eval.py are imported, never
modified; every design rule below is the v11 rule applied to the campaign's own dev half.

What is frozen per campaign (DEV half only; ``freeze_design(campaign)`` = ``run_v12.py --campaign C --freeze-design``)
into ``exp/results/v12_gates/obd_v12_<campaign>_frozen.json``:
* segments: the uf0 x uf3 rule (``obd_v11_eval._dev_rule``, which reproduces ``obd_v11.segmentation(df, 'uf0x3')``):
  cells < 2 % of dev rows merged into their user_feature_0 'other' cell, residual cells < 1 % merged into the largest
  cell; S is whatever the rule gives.  Stored as an explicit map; an eval pair not in the map goes to '<uf0>|other' and,
  if its uf0 value was never seen in dev, to the largest dev segment (fallback rows counted);
* arms (A = 2): arm 1 = the top half of the campaign's items by dev click rate (ties by item id;
  ``obd_v11.item_groups``), arm 0 = the rest (women 23 / 23, men 17 / 17); explicit item lists;
* problems: kappa[s, 0] = 0, kappa[s, 1] = max(1, round(8 S w_s)) with the DEV segment shares; budgets
  floor(b_q sum_s kappa[s, 1]), b_q = 0.10 .. 0.80 (15 problems).

Eval access (order fixed; external reviewer v11 lock r2 P2-2).  ``read_eval_rows`` is the only route to eval rows:
  (1) caller arguments (campaign, layer, lock sha256 format);
  (2) the v12 gate ``prereg_v12.addendum_gate`` (locked addendum, canonical hash, inherited v11 -> v5 chain, code /
      input / non-eval data drift, task registered, single-commit history) -- no eval byte is opened;
  (3) the task is registered for exactly this campaign's layer;
  (4) the caller's lock sha256 equals the locked addendum's sha256;
  (5) one access-log line is appended and fsync'ed;
  (6) only then are the campaign's two eval files hashed and checked against the lock and the campaign PROVENANCE
      (``_verify_eval_bytes``, private, not exported; external reviewer v11 lock r2 P2-1);
  (7) eval rows are deserialised (``_read_eval_rows_raw``, private).
Before the lock every eval-file hash is taken from the campaign's hash-bound PROVENANCE.json (no eval file opened).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv
from .data_v6 import file_sha256
from .obd_v11_eval import _dev_rule, _keys, assign_arms, assign_segments, eps_opt_share, frozen_problems

__all__ = ["CAMPAIGNS", "N_ITEMS", "LAYERS", "FROZEN_DIR", "DATA_FILES_V12", "EVAL_FILES", "N_MIN", "COST_G",
           "frozen_path", "access_log", "obd_dev_v12", "current_data_hashes", "data_hashes_v12", "data_drift_v12",
           "data_sha_v12", "freeze_design", "load_frozen", "frozen_problems", "assign_segments", "assign_arms",
           "eps_opt_share", "eval_gate", "read_eval_rows", "OBDEnvV12Frozen"]

from ..stats.prereg import WS_ROOT  # noqa: E402

CAMPAIGNS = ("women", "men")
N_ITEMS = {"women": 46, "men": 34}
LAYERS = {"women": "OBD-WOMEN-UF0X3-A2", "men": "OBD-MEN-UF0X3-A2"}
FROZEN_DIR = WS_ROOT / "exp" / "results" / "v12_gates"
N_MIN = 5000
COST_G = 8
_DS = "$DATA_DIR/open_bandit"
_FILE = {"dev": "dev.pkl", "eval_labels": "eval_labels.pkl", "eval_outcome": "eval_outcome.npy",
         "provenance": "PROVENANCE.json"}
DATA_FILES_V12 = {f"{c}_{k}": f"{_DS}/{c}/{f}" for c in CAMPAIGNS for k, f in _FILE.items()}
EVAL_FILES = tuple(f"{c}_{k}" for c in CAMPAIGNS for k in ("eval_labels", "eval_outcome"))
_DEV_CACHE: dict = {}


def _check_campaign(c):
    if c not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {c!r}")


def frozen_path(c):
    _check_campaign(c)
    return FROZEN_DIR / f"obd_v12_{c}_frozen.json"


def access_log(c):
    _check_campaign(c)
    return WS_ROOT / "exp" / "results" / "full" / "v12_obd" / f"{c}_eval_access_log.jsonl"


def obd_dev_v12(c):
    _check_campaign(c)
    if c not in _DEV_CACHE:
        _DEV_CACHE[c] = pd.read_pickle(DATA_FILES_V12[f"{c}_dev"])
    return _DEV_CACHE[c]


# =============================================================================================== data binding
def _prov(c):
    return json.loads(Path(DATA_FILES_V12[f"{c}_provenance"]).read_text())


def current_data_hashes():
    """name -> sha256 WITHOUT opening an eval file: dev.pkl and PROVENANCE.json of each campaign from their bytes, the
    four eval files from their campaign PROVENANCE.json record (itself hash-bound)."""
    out = {}
    for n, p in DATA_FILES_V12.items():
        c, k = n.split("_", 1)
        out[n] = _prov(c)["files"][_FILE[k]] if n in EVAL_FILES else file_sha256(p)
    return out


def data_hashes_v12():
    cur = current_data_hashes()
    return {n: {"path": p, "sha256": cur[n],
                "source": "campaign PROVENANCE.json record (file not opened before the lock)" if n in EVAL_FILES
                else "file bytes"} for n, p in DATA_FILES_V12.items()}


def data_drift_v12(frozen):
    cur = current_data_hashes()
    return [n for n in DATA_FILES_V12 if cur[n] != ((frozen or {}).get(n) or {}).get("sha256")]


def data_sha_v12(frozen=None):
    """Combined data hash for row binding (no eval byte read); with ``frozen`` every file must keep its hash."""
    cur = current_data_hashes()
    if frozen is not None:
        for n in DATA_FILES_V12:
            want = (frozen.get(n) or {}).get("sha256")
            if want != cur[n]:
                raise RuntimeError(f"raw data {n} sha256 {cur[n][:16]} != frozen {str(want)[:16]}")
    return hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()


def _verify_eval_bytes(c, frozen):
    """PRIVATE (not exported): hash campaign c's two eval files and compare with the lock and the campaign
    PROVENANCE.  Called only by ``read_eval_rows`` after authorisation and after the access-log line."""
    prov = _prov(c)["files"]
    for k in ("eval_labels", "eval_outcome"):
        n = f"{c}_{k}"
        h = file_sha256(DATA_FILES_V12[n])
        if h != ((frozen or {}).get(n) or {}).get("sha256") or h != prov[_FILE[k]]:
            raise RuntimeError(f"OBD {n} bytes differ from the locked / PROVENANCE sha256")


# =============================================================================================== frozen design
def freeze_design(c, out_path=None, overwrite=False):
    """Write campaign c's frozen design json from its DEV half (never touches an eval file)."""
    from .obd_v11 import item_groups, obd_problems, segmentation
    _check_campaign(c)
    out_path = frozen_path(c) if out_path is None else Path(out_path)
    if out_path.exists() and not overwrite:
        raise RuntimeError(f"{out_path} exists: the frozen design is never silently rewritten")
    df = obd_dev_v12(c)
    pair_map, other_map, levels, lab = _dev_rule(df)
    ref, _ = segmentation(df, "uf0x3")
    if not np.array_equal(lab, ref):
        raise RuntimeError("frozen uf0 x uf3 rule does not reproduce obd_v11.segmentation on dev")
    S = len(levels)
    grp = item_groups(df, 2)
    treat = sorted(int(i) for i, a in grp.items() if a == 1)
    ctrl = sorted(int(i) for i, a in grp.items() if a == 0)
    half = N_ITEMS[c] // 2
    if len(treat) != half or len(ctrl) != N_ITEMS[c] - half:
        raise RuntimeError(f"expected {half} / {N_ITEMS[c] - half} items, got {len(treat)} / {len(ctrl)}")
    w = np.bincount(lab, minlength=S) / len(lab)
    sp = obd_problems(w, 2, G=COST_G)
    fz = {"written_by": "obd_v12_eval.freeze_design (dev half only)", "written_at": datetime.now().isoformat(),
          "campaign": c, "layer": LAYERS[c], "S": S, "A": 2, "n_dev": int(len(df)),
          "dev_sha256": file_sha256(DATA_FILES_V12[f"{c}_dev"]),
          "segment_rule": "uf0 x uf3; cells < 2 % of dev rows -> '<uf0>|other'; residual < 1 % -> largest cell; "
                          "segments ordered by dev size (desc) (v11 rule, obd_v11_eval._dev_rule)",
          "segment_levels": levels, "pair_map": dict(sorted(pair_map.items())), "other_map": other_map,
          "fallback_segment": 0, "fallback_rule": "unseen (uf0, uf3) pair -> other_map['<uf0>|other']; unseen uf0 -> "
                                                  "fallback_segment (the largest dev segment)",
          "dev_segment_shares": [float(x) for x in w], "dev_segment_shares_hex": [float(x).hex() for x in w],
          "treat_items": treat, "control_items": ctrl,
          "arm_rule": f"arm 1 = the top half ({half}) of the {N_ITEMS[c]} items by dev click rate (ties by item id; "
                      f"obd_v11.item_groups), arm 0 = the other {N_ITEMS[c] - half}",
          "cost_G": COST_G, "cost": sp.cost.tolist(), "budgets": sp.budgets.tolist(), "qids": list(sp.qids),
          "budget_frac": list(sp.budget_frac), "n_min": N_MIN}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fz, indent=1))
    return fz


def load_frozen(c, path=None, expect_sha256=None):
    p = frozen_path(c) if path is None else Path(path)
    if expect_sha256 is not None and file_sha256(str(p)) != expect_sha256:
        raise RuntimeError("frozen OBD design json differs from the locked sha256")
    fz = json.loads(p.read_text())
    if fz.get("campaign") != c:
        raise RuntimeError(f"frozen design {p} is for campaign {fz.get('campaign')!r}, not {c!r}")
    return fz


# =============================================================================================== eval gate + reader
def eval_gate(c, eval_task_id, layer, gate_path=None):
    """Raise PermissionError unless the v12 gate authorises ``eval_task_id`` for exactly campaign c's layer."""
    if eval_task_id is None:
        raise PermissionError(f"{layer} eval half (v12): an authorised v12 eval task id is required")
    try:
        from ..stats.prereg_v12 import addendum_gate
    except ImportError as e:
        raise PermissionError(f"{layer} eval half (v12): no v12 addendum gate available ({e})") from None
    ok, info = addendum_gate(eval_task_id, path=gate_path)
    if not ok:
        raise PermissionError(f"{layer} eval half refused by the v12 addendum gate: {info}")
    ent = (info.get("eval_tasks") or {}).get(eval_task_id) or {}
    if ent.get("layer") != layer or ent.get("campaign") != c:
        raise PermissionError(f"{layer} eval half: task {eval_task_id!r} is registered for layer {ent.get('layer')!r} "
                              f"/ campaign {ent.get('campaign')!r}")
    return info


def _head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(WS_ROOT), capture_output=True)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def read_eval_rows(c, eval_task_id, layer, lock_sha256, gate_path=None, log_path=None, reason="v12 eval task"):
    """The only route to eval rows; order as in the module docstring.  Any failure before the access-log line refuses
    without opening an eval file; a logging failure refuses too."""
    if c not in CAMPAIGNS:
        raise PermissionError(f"unknown campaign {c!r}")
    if layer != LAYERS[c]:
        raise PermissionError(f"layer {layer!r} is not the v12 {c} layer {LAYERS[c]!r}")
    if not (isinstance(lock_sha256, str) and len(lock_sha256) == 64):
        raise PermissionError("v12 eval read refused: the caller must pass the locked addendum's sha256")
    lock = eval_gate(c, eval_task_id, layer, gate_path)
    if lock.get("sha256") != lock_sha256:
        raise PermissionError("v12 eval read refused: caller's lock sha256 does not match the locked addendum")
    rec = {"at": datetime.now().isoformat(), "task_id": eval_task_id, "campaign": c, "layer": layer, "half": "eval",
           "git_head": _head(), "v12_addendum_sha256": lock_sha256, "reason": reason}
    p = Path(log_path) if log_path is not None else access_log(c)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:                               # logged BEFORE any eval byte is hashed or read
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    _verify_eval_bytes(c, lock.get("data_sha256") or {})
    return _read_eval_rows_raw(c)


def _read_eval_rows_raw(c):
    """PRIVATE ungated (u0, u3, item_id, y, rows); reached only through ``read_eval_rows``."""
    df = pd.read_pickle(DATA_FILES_V12[f"{c}_eval_labels"])
    if "click" in df.columns:
        raise RuntimeError("eval_labels.pkl unexpectedly contains the outcome")
    u0, u3 = _keys(df)
    item = df["item_id"].to_numpy().astype(np.int64)
    y = np.load(DATA_FILES_V12[f"{c}_eval_outcome"]).astype(float)
    if len(y) != len(df):
        raise RuntimeError("OBD eval outcome / labels length mismatch")
    return u0, u3, item, y, df["row_id"].to_numpy().astype(np.int64)


def _read_dev_rows(c):
    df = obd_dev_v12(c)
    u0, u3 = _keys(df)
    return u0, u3, df["item_id"].to_numpy().astype(np.int64), df["click"].to_numpy(dtype=float), \
        df["row_id"].to_numpy().astype(np.int64)


# =============================================================================================== environment
class OBDEnvV12Frozen(SyntheticPoolEnv):
    """Replay population (whole dev or whole eval half of campaign c) under its FROZEN v12 design."""

    def __init__(self, campaign, half="dev", eval_task_id=None, lock_sha256=None, gate_path=None, log_path=None,
                 frozen_path_=None, frozen_sha256=None):  # noqa: D401 - no super().__init__
        _check_campaign(campaign)
        if half not in ("dev", "eval"):
            raise ValueError(half)
        if half == "eval":
            u0, u3, item, y, rows = read_eval_rows(campaign, eval_task_id, LAYERS[campaign], lock_sha256, gate_path,
                                                   log_path)
        else:
            u0, u3, item, y, rows = _read_dev_rows(campaign)
        fz = load_frozen(campaign, frozen_path_, frozen_sha256)
        S, A = int(fz["S"]), int(fz["A"])
        self.campaign = campaign
        self.seg, nfp, nfu = assign_segments(u0, u3, fz)
        self.arm = assign_arms(item, fz)
        self.fallback_rows = {"pair": int(nfp), "uf0": int(nfu)}
        self.layer, self.half, self.split_seed = LAYERS[campaign], half, 0
        self.S, self.A = S, A
        self.seg_desc = [str(x) for x in fz["segment_levels"]]
        y = np.asarray(y, dtype=float).copy()
        y.setflags(write=False)
        self._y = {"visit": y}
        self.clip = {}
        self.binary_outcomes = ("visit",)
        self.N = int(len(self.seg))
        self.row_index = np.asarray(rows)
        self.cell = self.seg * A + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=S * A).reshape(S, A)
        if (self.pool_sizes <= 0).any():
            raise RuntimeError(f"{self.layer} ({half}): empty pool under the frozen design")
        self.pool_rows = [np.flatnonzero(self.cell == k) for k in range(S * A)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = int(fz.get("n_min", N_MIN))
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)
        self.problems = frozen_problems(fz)

    def describe(self):
        return {"campaign": self.campaign, "layer": self.layer, "half": self.half, "S": self.S, "A": self.A,
                "N": self.N, "pool_sizes_min": int(self.pool_sizes.min()), "w": [round(float(x), 5) for x in self.w],
                "cost": self.problems.cost[:, 1].tolist(), "budgets": self.problems.budgets.tolist(),
                "fallback_rows": self.fallback_rows}
