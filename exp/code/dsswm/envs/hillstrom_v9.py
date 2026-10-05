"""Hillstrom MineThatData e-mail RCT as a 3-arm real layer for lock v9 block C (NEW file; nothing locked is touched).

Data: $DATA_DIR/hillstrom/tidy.pkl (64,000 customers, sha256 in PROVENANCE.json).
Arms: treatment 0 = No E-Mail, 1 = Mens E-Mail, 2 = Womens E-Mail (A = 3, RCT shares ~1/3 each).

Exposure status (plan/hillstrom_exposure_audit.md): round 4 computed FULL-table cell means of visit, conversion and
spend on the hs3 x newbie (HR6) and hs3 x newbie x category (HR16) segmentations and eval-half truth tables of HR8.
Every row of this table, and every coarsening of those segmentations, is outcome-exposed.  Block C is therefore a
DESCRIPTIVE re-use of an outcome-exposed log (same status as v6 block B, Lenta), whatever split is used below.

Split (v9, new): hashed by source row index with a fixed salt (no outcome, no covariate enters the hash):
    u_i = int(sha256(SALT + ':' + str(i))[:16], 16) / 2**64,   row i in DEV iff u_i < 0.5.
The dev half is always available.  The eval half is released ONLY to an eval task id accepted by
``dsswm.stats.prereg_v9.addendum_gate`` (that module does not exist before the v9 lock is written, so any eval access
before the lock raises PermissionError).  Process note: tidy.pkl is read whole by pandas (the outcome column of eval
rows is in memory transiently while the dev half is built); no statistic of an eval-row outcome is computed here.

Segmentations (features only; cut points from covariates of the FULL table, never from outcomes):
    cat3   purchase-category history: 0 mens-only (x_mens=1, x_womens=0), 1 womens-only (0, 1), 2 both (1, 1)
    CN6    cat3 x newbie                       (S = 6, |Pi| = 729)
    CR6    cat3 x recency (<= 5 | >= 6 months; 5 is the full-table median cut: 31,980 / 32,020 rows)
    CC6    cat3 x channel (Web | Phone or Multichannel)
    CZ6    cat3 x zip (Urban | Suburban or Rural)
None of these four is an r4 layer (r4 used hs3 x newbie, hs3 x newbie x category, hs3 x category).
Problems (hv9_problems): per-arm e-mail cost kappa (no e-mail free) from a cost family, budget B = maximum average cost
per customer in {0.10, 0.15, ..., 0.80}: 15 problems, stop at 12/15 (same budget grid as CR).  Cost families:
    F1 (0, 1, 1)      both e-mails one unit
    F3 (0, 1.5, 1)    the r4 HR cost tier C2 (frontier.HR_COST_TIERS['C2']), inherited, not chosen on v9 data
    F2 (0, 1, 0.5)    womens e-mail half price (no precedent; lowest precedence in the design rule)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv

ROOT = (Path(__import__("os").environ.get("DATA_DIR", "data")) / "hillstrom")
TIDY = ROOT / "tidy.pkl"
TIDY_SHA256 = "3dab9ed72bfdcaf51f6d993975a7ca052ef9355690489ca854217b7f0a68b296"
SALT = "ds-swm/v9/hillstrom/2026-10-04"
ARMS = ("no_email", "mens_email", "womens_email")
OUTCOMES = {"visit": "binary", "conversion": "conversion"}       # public name -> tidy column (binary outcomes only)
SEGMENTATIONS = ("CN6", "CR6", "CC6", "CZ6")
# design B (coarser layers, authors request 2026-10-04): cat2 = mens-only | any womens history (womens-only or
# both); S = 4 layers cat2 x {newbie, recency, channel, zip}; S = 3 layer cat3 alone
SEGMENTATIONS_B = ("CN6", "CR6", "CC6", "CZ6", "DN4", "DR4", "DC4", "DZ4", "C3")
CAT_NAMES = ("mens_only", "womens_only", "both")
RECENCY_CUT = 5                                                  # recency <= 5 -> 0, >= 6 -> 1 (full-table median)
CHANNEL_WEB = 2                                                  # tidy coding 0 Multichannel / 1 Phone / 2 Web
ZIP_URBAN = 2                                                    # tidy coding 0 Rural / 1 Surburban / 2 Urban
HV9_BUDGETS = fr.CR_BUDGETS                                      # 0.10, 0.15, ..., 0.80
COST_FAMILIES = {"F1": (0.0, 1.0, 1.0), "F3": tuple(fr.HR_COST_TIERS["C2"]), "F2": (0.0, 1.0, 0.5)}
FAMILY_PRECEDENCE = ("F1", "F3", "F2")
HV9_KAPPA = COST_FAMILIES["F1"]


def hv9_problems(outcome="visit", family="F1"):
    k = COST_FAMILIES[family]
    return [fr.Problem(f"{outcome}|{family}|B{b:.2f}", outcome, k, b) for b in HV9_BUDGETS]


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hashed_dev_mask(n_rows, salt=SALT):
    """Row i is DEV iff the first 64 bits of sha256(salt:i) / 2^64 < 0.5 (outcome- and covariate-free)."""
    u = np.array([int(hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:16], 16) for i in range(n_rows)],
                 dtype=np.float64) / 2.0 ** 64
    return u < 0.5


def cat3(df):
    m, w = df["x_mens"].to_numpy().astype(int), df["x_womens"].to_numpy().astype(int)
    if np.any((m == 0) & (w == 0)):
        raise ValueError("rows with neither mens nor womens purchase history are not covered by cat3")
    return np.where((m == 1) & (w == 0), 0, np.where((m == 0) & (w == 1), 1, 2)).astype(np.int64)


def _binary_feature(df, key):
    if key == "N":
        return df["x_newbie"].to_numpy().astype(np.int64), ("old", "newbie")
    if key == "R":
        return (df["x_recency"].to_numpy() > RECENCY_CUT).astype(np.int64), ("rec<=5", "rec>=6")
    if key == "C":
        return (df["x_channel"].to_numpy() == CHANNEL_WEB).astype(np.int64), ("phone/multi", "web")
    if key == "Z":
        return (df["x_zip"].to_numpy() == ZIP_URBAN).astype(np.int64), ("non-urban", "urban")
    raise KeyError(key)


def segment_labels(df, seg):
    """(labels, descriptions) for a v9 segmentation; features only."""
    c = cat3(df)
    if seg == "C3":
        return c, list(CAT_NAMES)
    if seg[0] == "C" and seg[-1] == "6":
        b, names = _binary_feature(df, seg[1])
        lab = (c * 2 + b).astype(np.int64)
        return lab, [f"{CAT_NAMES[k // 2]}|{names[k % 2]}" for k in range(6)]
    if seg[0] == "D" and seg[-1] == "4":
        c2 = (c != 0).astype(np.int64)                         # 0 mens-only, 1 any womens history
        b, names = _binary_feature(df, seg[1])
        lab = (c2 * 2 + b).astype(np.int64)
        return lab, [f"{('mens_only', 'any_womens')[k // 2]}|{names[k % 2]}" for k in range(4)]
    raise KeyError(seg)


def _eval_gate(eval_task_id, gate_path=None):
    if eval_task_id is None:
        raise PermissionError("Hillstrom v9 eval half: an authorised v9 block-C eval task id is required")
    try:
        from ..stats.prereg_v9 import addendum_gate          # written with the v9 lock; absent before it
    except ImportError as e:
        raise PermissionError(f"Hillstrom v9 eval half: no v9 addendum gate available ({e})") from None
    ok, info = addendum_gate(eval_task_id, path=gate_path)
    if not ok:
        raise PermissionError(f"Hillstrom v9 eval half refused by the v9 addendum gate: {info}")
    if not str(eval_task_id).startswith("v9c_full"):
        raise PermissionError("Hillstrom v9 eval outcomes are reserved for v9 block-C eval tasks")


class HillstromV9Env(SyntheticPoolEnv):
    """Finite population = one hashed half of the Hillstrom table, pools (segment, arm), binary outcomes."""

    def __init__(self, seg="CN6", half="dev", outcomes=("visit", "conversion"), n_min=1000, eval_task_id=None,
                 gate_path=None, check_sha=True):  # noqa: D401 - no super().__init__
        if half not in ("dev", "eval"):
            raise ValueError(half)
        if half == "eval":
            _eval_gate(eval_task_id, gate_path)
        if check_sha and file_sha256(TIDY) != TIDY_SHA256:
            raise RuntimeError("Hillstrom tidy.pkl sha256 differs from PROVENANCE.json")
        df = pd.read_pickle(TIDY)
        dev = hashed_dev_mask(len(df))
        mask = dev if half == "dev" else ~dev
        lab, desc = segment_labels(df, seg)
        self.layer, self.half, self.split_seed = f"H{seg}", half, SALT
        self.seg_name = seg
        self.S, self.A = len(desc), 3
        self.seg_desc = desc
        self.row_index = np.flatnonzero(mask)
        self.seg = lab[mask]
        self.arm = df["treatment"].to_numpy().astype(np.int64)[mask]
        self._y = {}
        for o in outcomes:
            v = df[OUTCOMES[o]].to_numpy().astype(float)[mask].copy()
            v.setflags(write=False)
            self._y[o] = v
        del df
        self.clip = {}
        self.binary_outcomes = tuple(outcomes)
        self.N = int(mask.sum())
        self.cell = self.seg * self.A + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=self.S * self.A).reshape(self.S, self.A)
        if self.pool_sizes.min() <= 0:
            raise RuntimeError("empty pool")
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(self.S * self.A)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = int(n_min)
        self.eps_grid = (0.005,)
        self.replan_interval = fr.replan_interval(self.tau_R)

    def public_info(self):
        return {"layer": self.layer, "half": self.half, "S": self.S, "A": self.A, "N": self.N,
                "pool_sizes": self.pool_sizes.tolist(), "w": self.w.tolist(), "seg_desc": self.seg_desc,
                "salt": SALT, "tau_R": self.tau_R, "replan_interval": self.replan_interval}


def split_summary():
    """Label-only summary of the hashed split (row counts per half x arm); reads no outcome."""
    df = pd.read_pickle(TIDY)
    dev = hashed_dev_mask(len(df))
    arm = df["treatment"].to_numpy()
    return {"salt": SALT, "dev_rows": int(dev.sum()), "eval_rows": int((~dev).sum()),
            "dev_by_arm": np.bincount(arm[dev], minlength=3).tolist(),
            "eval_by_arm": np.bincount(arm[~dev], minlength=3).tolist(),
            "dev_index_sha256": hashlib.sha256(np.flatnonzero(dev).astype(np.int64).tobytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(split_summary(), indent=1))
