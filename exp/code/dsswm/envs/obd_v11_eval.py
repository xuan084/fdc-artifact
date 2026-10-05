"""Lock v11: frozen Open Bandit design and GATED eval-half access (NEW file; obd_v11.py, seg_v10*.py, fdc_dp.py are
imported, never modified).

What is frozen (from the DEV half only, written once by ``freeze_design`` = ``run_v11.py --freeze-design``) into
``exp/results/v11_gates/obd_v11_frozen.json``:
* segments (S = 9): the uf0 x uf3 rule of ``obd_v11.segmentation(df, 'uf0x3')`` (cells < 2 % of dev rows merged into
  their user_feature_0 'other' cell, residual cells < 1 % merged into the largest cell), stored as an explicit map
  key -> segment, key = '<uf0 value>|<uf3 value>' (dev-observed pairs) or '<uf0 value>|other' (dev-observed uf0
  values).  An eval pair not in the map goes to '<uf0>|other' (the dev merge rule) and, if its uf0 value was never
  seen in dev, to the largest dev segment (``fallback_segment``).  Rows routed by a fallback are counted;
* arms (A = 2): arm 1 = the 40 items with the highest dev click rate (ties by item id; ``obd_v11.item_groups``), arm 0
  = the other 40; stored as explicit item lists;
* problems: kappa[s, 0] = 0, kappa[s, 1] = max(1, round(8 S w_s)) with the DEV segment shares w_s, budgets
  floor(b_q sum_s kappa[s, 1]), b_q = 0.10 .. 0.80 (15 problems) -- identical on dev and eval (policy values use the
  replay half's own segment weights).

Eval access.  ``OBDEnvV11Frozen('eval', eval_task_id=..., lock_sha256=...)`` loads eval rows only through
``read_eval_rows``, which (1) runs ``eval_gate`` (``prereg_v11.addendum_gate``: locked v11 addendum, every hash, task
registered) and checks that the task is registered for exactly this layer, (2) requires the caller's ``lock_sha256``
to equal the locked addendum's sha256, (3) checks the eval files against the lock's data hashes and PROVENANCE.json,
(4) appends one line to the access log BEFORE any eval byte is deserialised, then reads.  Without all of that nothing
of the eval half is read (not even covariates).  The dev half (``half='dev'``) reads dev.pkl only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv
from .data_v6 import file_sha256

__all__ = ["FROZEN_DIR", "FROZEN_JSON", "DATA_FILES_V11", "LAYER", "ACCESS_LOG", "OBDEnvV11Frozen", "freeze_design",
           "load_frozen", "frozen_problems", "assign_segments", "assign_arms", "data_hashes_v11", "data_drift_v11",
           "data_sha_v11", "current_data_hashes", "verify_eval_bytes", "EVAL_FILES", "eps_opt_share", "eval_gate", "read_eval_rows", "S_FROZEN", "A_FROZEN", "N_MIN", "COST_G"]

from ..stats.prereg import WS_ROOT  # noqa: E402

FROZEN_DIR = WS_ROOT / "exp" / "results" / "v11_gates"
FROZEN_JSON = FROZEN_DIR / "obd_v11_frozen.json"
ACCESS_LOG = WS_ROOT / "exp" / "results" / "full" / "v11_obd" / "eval_access_log.jsonl"
LAYER = "OBD-UF0X3-A2-S9"
S_FROZEN, A_FROZEN = 9, 2
N_MIN = 5000
COST_G = 8
UF0, UF3 = "user_feature_0", "user_feature_3"
_DS = "$DATA_DIR/open_bandit"
DATA_FILES_V11 = {
    "obd_dev": f"{_DS}/dev.pkl",
    "obd_eval_labels": f"{_DS}/eval_labels.pkl",
    "obd_eval_outcome": f"{_DS}/eval_outcome.npy",
    "obd_provenance": f"{_DS}/PROVENANCE.json",
}
_PROV_KEY = {"obd_dev": "dev.pkl", "obd_eval_labels": "eval_labels.pkl", "obd_eval_outcome": "eval_outcome.npy"}


# =============================================================================================== data binding
EVAL_FILES = ("obd_eval_labels", "obd_eval_outcome")


def _prov_hashes():
    return json.loads(Path(DATA_FILES_V11["obd_provenance"]).read_text())["files"]


def current_data_hashes():
    """name -> sha256 WITHOUT touching an eval file (external reviewer v11 lock r1 F1/F3): dev.pkl and PROVENANCE.json are hashed
    from their bytes; the two eval files' hashes are the ones PROVENANCE.json recorded at split time (PROVENANCE.json
    is itself hash-bound).  The eval files' actual bytes are verified only by ``verify_eval_bytes``, which runs inside
    ``read_eval_rows`` after full authorisation and after the access-log line is written."""
    prov = _prov_hashes()
    out = {}
    for n, p in DATA_FILES_V11.items():
        out[n] = prov[_PROV_KEY[n]] if n in EVAL_FILES else file_sha256(p)
    return out


def data_hashes_v11():
    cur = current_data_hashes()
    return {n: {"path": p, "sha256": cur[n],
                "source": "PROVENANCE.json record (file not opened before the lock)" if n in EVAL_FILES
                else "file bytes"} for n, p in DATA_FILES_V11.items()}


def data_drift_v11(frozen):
    """Bound files whose current hash (eval files: via PROVENANCE; no eval byte read) differs from ``frozen``."""
    cur = current_data_hashes()
    return [n for n in DATA_FILES_V11 if cur[n] != ((frozen or {}).get(n) or {}).get("sha256")]


def data_sha_v11(frozen=None):
    """Combined data hash (row binding; identical on dev and eval, no eval byte read); with ``frozen`` every file must
    still have its frozen hash."""
    cur = current_data_hashes()
    if frozen is not None:
        for n in DATA_FILES_V11:
            want = (frozen.get(n) or {}).get("sha256")
            if want != cur[n]:
                raise RuntimeError(f"raw data {n} sha256 {cur[n][:16]} != frozen {str(want)[:16]}")
    return hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()


def verify_eval_bytes(frozen):
    """Hash the two eval files from their bytes and compare with the lock (and PROVENANCE).  Private to the gated
    reader: called only after authorisation and logging."""
    prov = _prov_hashes()
    for n in EVAL_FILES:
        h = file_sha256(DATA_FILES_V11[n])
        if h != ((frozen or {}).get(n) or {}).get("sha256") or h != prov[_PROV_KEY[n]]:
            raise RuntimeError(f"OBD {n} bytes differ from the locked / PROVENANCE sha256")


# =============================================================================================== frozen design
def _keys(df):
    return df[UF0].astype(str).to_numpy(dtype=object), df[UF3].astype(str).to_numpy(dtype=object)


def _dev_rule(df):
    """The uf0 x uf3 rule of obd_v11.segmentation, returning the full key -> segment map (pre-merge keys included)."""
    n = len(df)
    u0 = df[UF0].astype(str)
    k0 = u0 + "|" + df[UF3].astype(str)
    vc = k0.value_counts()
    k1 = k0.where(k0.map(vc) >= 0.02 * n, u0 + "|other")
    vc1 = k1.value_counts()
    k2 = k1.where(k1.map(vc1) >= 0.01 * n, vc1.index[0])
    levels = list(k2.value_counts().index)
    code = {c: i for i, c in enumerate(levels)}
    stage1 = dict(zip(k0, k1))                    # pair key -> stage-1 key
    big = vc1.index[0]

    def final(k):
        return code[k] if k in code else code[big]

    pair_map = {k: final(v) for k, v in stage1.items()}
    other_map = {f"{u}|other": final(f"{u}|other") if f"{u}|other" in vc1.index else final(big)
                 for u in sorted(u0.unique())}
    lab = k2.map(code).to_numpy().astype(np.int64)
    if code[big] != 0:
        raise RuntimeError("the largest stage-1 cell is not the largest final segment")
    return pair_map, other_map, levels, lab


def freeze_design(out_path=None, overwrite=False):
    """Write the frozen design json from the DEV half (never touches an eval file)."""
    from .obd_v11 import item_groups, obd_dev, obd_problems, segmentation
    out_path = FROZEN_JSON if out_path is None else Path(out_path)
    if out_path.exists() and not overwrite:
        raise RuntimeError(f"{out_path} exists: the frozen design is never silently rewritten")
    df = obd_dev()
    pair_map, other_map, levels, lab = _dev_rule(df)
    ref, _ = segmentation(df, "uf0x3")
    if not np.array_equal(lab, ref):
        raise RuntimeError("frozen uf0 x uf3 rule does not reproduce obd_v11.segmentation on dev")
    S = len(levels)
    if S != S_FROZEN:
        raise RuntimeError(f"expected S = {S_FROZEN}, got {S}")
    grp = item_groups(df, A_FROZEN)
    treat = sorted(int(i) for i, a in grp.items() if a == 1)
    ctrl = sorted(int(i) for i, a in grp.items() if a == 0)
    if len(treat) != 40 or len(ctrl) != 40:
        raise RuntimeError("expected 40 / 40 items")
    w = np.bincount(lab, minlength=S) / len(lab)
    sp = obd_problems(w, A_FROZEN, G=COST_G)
    fz = {"written_by": "obd_v11_eval.freeze_design (dev half only)", "written_at": datetime.now().isoformat(),
          "layer": LAYER, "S": S, "A": A_FROZEN, "n_dev": int(len(df)), "dev_sha256": file_sha256(DATA_FILES_V11["obd_dev"]),
          "segment_rule": "uf0 x uf3; cells < 2 % of dev rows -> '<uf0>|other'; residual < 1 % -> largest cell; "
                          "segments ordered by dev size (desc)",
          "segment_levels": levels, "pair_map": dict(sorted(pair_map.items())), "other_map": other_map,
          "fallback_segment": 0, "fallback_rule": "unseen (uf0, uf3) pair -> other_map['<uf0>|other']; unseen uf0 -> "
                                                  "fallback_segment (the largest dev segment)",
          "dev_segment_shares": [float(x) for x in w], "dev_segment_shares_hex": [float(x).hex() for x in w],
          "treat_items": treat, "control_items": ctrl,
          "arm_rule": "arm 1 = 40 items with the highest dev click rate (ties by item id), arm 0 = the other 40",
          "cost_G": COST_G, "cost": sp.cost.tolist(), "budgets": sp.budgets.tolist(), "qids": list(sp.qids),
          "budget_frac": list(sp.budget_frac), "n_min": N_MIN}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fz, indent=1))
    return fz


def load_frozen(path=None, expect_sha256=None):
    p = FROZEN_JSON if path is None else Path(path)
    if expect_sha256 is not None and file_sha256(str(p)) != expect_sha256:
        raise RuntimeError("frozen OBD design json differs from the locked sha256")
    return json.loads(p.read_text())


def frozen_problems(fz):
    from ..baselines.fdc_dp import SegProblems
    return SegProblems(cost=np.array(fz["cost"], dtype=np.int64), budgets=np.array(fz["budgets"], dtype=np.int64),
                       qids=list(fz["qids"]), budget_frac=list(fz["budget_frac"]))


def assign_segments(u0, u3, fz):
    """(segments, n_fallback_pair, n_fallback_uf0) under the frozen map."""
    pm, om, fb = fz["pair_map"], fz["other_map"], int(fz["fallback_segment"])
    out = np.empty(len(u0), dtype=np.int64)
    nf_pair = nf_u0 = 0
    for i, (a, b) in enumerate(zip(u0, u3)):
        k = f"{a}|{b}"
        if k in pm:
            out[i] = pm[k]
        elif f"{a}|other" in om:
            out[i] = om[f"{a}|other"]
            nf_pair += 1
        else:
            out[i] = fb
            nf_u0 += 1
    return out, nf_pair, nf_u0


def assign_arms(item_id, fz):
    t, c = set(fz["treat_items"]), set(fz["control_items"])
    item_id = np.asarray(item_id, dtype=np.int64)
    bad = [int(i) for i in np.unique(item_id) if int(i) not in t and int(i) not in c]
    if bad:
        raise RuntimeError(f"items outside the frozen arm lists: {bad[:5]}")
    return np.isin(item_id, sorted(t)).astype(np.int64)


# =============================================================================================== eval gate + reader
def eval_gate(eval_task_id, layer, gate_path=None):
    """Raise PermissionError unless the v11 gate authorises ``eval_task_id`` for exactly this layer."""
    if eval_task_id is None:
        raise PermissionError(f"{layer} eval half (v11): an authorised v11 eval task id is required")
    try:
        from ..stats.prereg_v11 import addendum_gate
    except ImportError as e:
        raise PermissionError(f"{layer} eval half (v11): no v11 addendum gate available ({e})") from None
    ok, info = addendum_gate(eval_task_id, path=gate_path)
    if not ok:
        raise PermissionError(f"{layer} eval half refused by the v11 addendum gate: {info}")
    ent = (info.get("eval_tasks") or {}).get(eval_task_id) or {}
    if ent.get("layer") != layer:
        raise PermissionError(f"{layer} eval half: task {eval_task_id!r} is registered for layer {ent.get('layer')!r}")
    return info


def _head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(WS_ROOT), capture_output=True)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def read_eval_rows(eval_task_id, layer, lock_sha256, gate_path=None, log_path=None, reason="v11 eval task"):
    """Public eval reader.  Order (external reviewer v11 lock r1 F1): caller arguments -> v11 gate (locked addendum, canonical
    hash, inherited chain, code / input / non-eval data drift, task registered, single-commit history; no eval byte
    touched) -> task registered for exactly this layer -> caller's lock sha256 equals the lock -> one access-log line
    appended and fsync'ed -> eval files hashed and checked -> eval rows deserialised.  Any failure before the log
    line refuses without touching an eval file; a logging failure refuses too."""
    if layer != LAYER:
        raise PermissionError(f"layer {layer!r} is not the v11 OBD layer {LAYER!r}")
    if not (isinstance(lock_sha256, str) and len(lock_sha256) == 64):
        raise PermissionError("v11 eval read refused: the caller must pass the locked addendum's sha256")
    lock = eval_gate(eval_task_id, layer, gate_path)
    if lock.get("sha256") != lock_sha256:
        raise PermissionError("v11 eval read refused: caller's lock sha256 does not match the locked addendum")
    rec = {"at": datetime.now().isoformat(), "task_id": eval_task_id, "layer": layer, "half": "eval",
           "git_head": _head(), "v11_addendum_sha256": lock_sha256, "reason": reason}
    p = Path(log_path) if log_path is not None else ACCESS_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:                               # logged BEFORE any eval byte is hashed or read
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    verify_eval_bytes(lock.get("data_sha256") or {})
    return _read_eval_rows_raw()


def _read_eval_rows_raw():
    """Ungated (u0, u3, item_id, y, rows).  Private: reached only through ``read_eval_rows``.  Not exported."""
    df = pd.read_pickle(DATA_FILES_V11["obd_eval_labels"])
    if "click" in df.columns:
        raise RuntimeError("eval_labels.pkl unexpectedly contains the outcome")
    u0, u3 = _keys(df)
    item = df["item_id"].to_numpy().astype(np.int64)
    y = np.load(DATA_FILES_V11["obd_eval_outcome"]).astype(float)
    if len(y) != len(df):
        raise RuntimeError("OBD eval outcome / labels length mismatch")
    return u0, u3, item, y, df["row_id"].to_numpy().astype(np.int64)


def _read_dev_rows():
    from .obd_v11 import obd_dev
    df = obd_dev()
    u0, u3 = _keys(df)
    return u0, u3, df["item_id"].to_numpy().astype(np.int64), df["click"].to_numpy(dtype=float), \
        df["row_id"].to_numpy().astype(np.int64)


# =============================================================================================== environment
class OBDEnvV11Frozen(SyntheticPoolEnv):
    """Replay population (whole dev or whole eval half) under the FROZEN v11 design (PoolReplayEnv interface)."""

    def __init__(self, half="dev", eval_task_id=None, lock_sha256=None, gate_path=None, log_path=None,
                 frozen_path=None, frozen_sha256=None):  # noqa: D401 - no super().__init__
        if half not in ("dev", "eval"):
            raise ValueError(half)
        if half == "eval":
            u0, u3, item, y, rows = read_eval_rows(eval_task_id, LAYER, lock_sha256, gate_path, log_path)
        else:
            u0, u3, item, y, rows = _read_dev_rows()
        fz = load_frozen(frozen_path, frozen_sha256)
        S, A = int(fz["S"]), int(fz["A"])
        self.seg, nfp, nfu = assign_segments(u0, u3, fz)
        self.arm = assign_arms(item, fz)
        self.fallback_rows = {"pair": int(nfp), "uf0": int(nfu)}
        self.layer, self.half, self.split_seed = LAYER, half, 0
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
            raise RuntimeError(f"{LAYER} ({half}): empty pool under the frozen design")
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(S * A)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = int(fz.get("n_min", N_MIN))
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)
        self.problems = frozen_problems(fz)

    def describe(self):
        return {"layer": self.layer, "half": self.half, "S": self.S, "A": self.A, "N": self.N,
                "pool_sizes_min": int(self.pool_sizes.min()), "w": [round(float(x), 5) for x in self.w],
                "cost": self.problems.cost[:, 1].tolist(), "budgets": self.problems.budgets.tolist(),
                "fallback_rows": self.fallback_rows}


def eps_opt_share(env, Js, mu, eps):
    """Mean over problems of the share of FEASIBLE policies that are eps-optimal (enumerates A^S policies; S = 9)."""
    import itertools
    S, A, sp = env.S, env.A, env.problems
    pols = np.array(list(itertools.product(range(A), repeat=S)))
    val = (np.asarray(env.w)[None, :] * mu[np.arange(S)[None, :], pols]).sum(1)
    cst = sp.cost[np.arange(S)[None, :], pols].sum(1)
    J0 = float((np.asarray(env.w) * mu[:, 0]).sum())
    fr_ = [float((val[cst <= sp.budgets[q]] >= Js[q] - eps).mean()) for q in range(sp.Q)]
    return {"mean_share_eps_optimal": float(np.mean(fr_)), "per_problem": [round(x, 4) for x in fr_],
            "n_problems_all_control_eps_optimal": int(sum(J0 >= Js[q] - eps for q in range(sp.Q))),
            "n_policies": int(len(pols))}


_ = math
