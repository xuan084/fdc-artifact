"""Lock v10: frozen uplift-score segmentations and GATED eval-half access (NEW file; seg_v10 / lenta_v6 / data_v8 are
imported, never modified).

What is frozen (from the DEV halves only, written once by ``freeze_segmentations`` = ``run_v10.py --freeze-seg``):
* the score model of each table: the seg_v10 T-learner (two HistGradientBoostingClassifier, one per arm, fixed
  hyper-parameters, random_state 0) fit on the 30 % model split of the dev half (seed 10010), pickled to
  ``exp/results/v10_gates/score_model_<data>.pkl``; the frozen json records the pickle sha256 and the sha256 of the
  dev replay-pool score vector it produces (a load-time reproducibility check);
* the segmentation of each (table, S): the S - 1 cut points c_1 <= ... <= c_{S-1} = the minimum dev replay-pool score
  of seg_v10's equal-size score-quantile segment j (j = 1..S-1; float.hex in the json).  A row with score x gets
  segment #{j : c_j <= x} (numpy searchsorted side='right': a score equal to a cut point goes to the upper segment);
  the cut points must be strictly increasing;
* the problems of each (table, S): seg_v10.seg_problems on the DEV equal-size weights, i.e. cost kappa[s, 1] = 8 for
  every segment and budgets floor(b_q 8 S) (treat at most floor(b_q S) segments).  They are NOT recomputed from eval
  weights, so the policy classes are identical on dev and eval.

Eval access.  ``SegEnvV10Frozen(data, S, 'eval', eval_task_id=...)`` loads eval rows ONLY when
``dsswm.stats.prereg_v10.addendum_gate(eval_task_id)`` passes (locked v10 addendum, task listed, hashes) AND the
lock's eval-task entry names exactly this layer (``<DATA>-SR<S>``).  Without that, nothing of the eval half is read
(not even covariates).  Halves:
* X5: the eval half of the blinded hashed split of ingest_v8_fresh (eval_labels.pkl covariates + treatment,
  eval_outcome.npy outcome); files checked against PROVENANCE.json.  The replay population is the WHOLE eval half.
* Lenta: the LR9 split (lenta_v6: covariate-only LR9 strata, split seed 6006), eval half = complement of the dev half;
  the replay population is the whole eval half.  pandas reads the whole pickle; eval outcomes are kept only after
  the gate passed.
The dev half (``half='dev'``) applies the same frozen model and cut points to seg_v10's dev replay pool (70 % of the
dev half); it is the population of the v10 dev runner check.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import SyntheticPoolEnv
from . import seg_v10 as SV
from .data_v6 import file_sha256

__all__ = ["FROZEN_DIR", "FROZEN_JSON", "DATA_FILES_V10", "LAYER_DATA_V10", "SegEnvV10Frozen", "freeze_segmentations",
           "load_frozen_seg", "frozen_problems", "assign_segments", "layer_name", "data_hashes_v10",
           "layer_data_sha_v10", "data_drift_v10", "x5_features", "HGB_PARAMS", "eval_gate", "read_eval_rows",
           "lenta_eval_mask"]

from ..stats.prereg import WS_ROOT  # noqa: E402

FROZEN_DIR = WS_ROOT / "exp" / "results" / "v10_gates"
FROZEN_JSON = FROZEN_DIR / "seg_v10_frozen.json"
S_GRID = (16, 32, 64)
HGB_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
                  random_state=0)

_DS = __import__("os").environ.get("DATA_DIR", "data")
DATA_FILES_V10 = {
    "lenta_tidy": f"{_DS}/lenta/tidy.pkl",
    "x5_dev": f"{_DS}/x5_retailhero/dev.pkl",
    "x5_eval_labels": f"{_DS}/x5_retailhero/eval_labels.pkl",
    "x5_eval_outcome": f"{_DS}/x5_retailhero/eval_outcome.npy",
    "x5_provenance": f"{_DS}/x5_retailhero/PROVENANCE.json",
}
LAYER_DATA_V10 = {"lenta": ("lenta_tidy",), "x5": ("x5_dev", "x5_eval_labels", "x5_eval_outcome", "x5_provenance")}


def layer_name(data, S):
    return f"{data.upper()}-SR{int(S)}"


# =============================================================================================== data binding
def data_hashes_v10(names=None):
    names = list(DATA_FILES_V10) if names is None else list(names)
    return {n: {"path": DATA_FILES_V10[n], "sha256": file_sha256(DATA_FILES_V10[n])} for n in names}


def data_drift_v10(frozen):
    return [n for n, d in (frozen or {}).items() if file_sha256(DATA_FILES_V10[n]) != (d or {}).get("sha256")]


def layer_data_sha_v10(data, frozen=None):
    names = LAYER_DATA_V10[data]
    cur = {n: file_sha256(DATA_FILES_V10[n]) for n in names}
    if frozen is not None:
        for n in names:
            want = (frozen.get(n) or {}).get("sha256")
            if want != cur[n]:
                raise RuntimeError(f"raw data {n} sha256 {cur[n][:16]} != frozen {str(want)[:16]}")
    return hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()


# =============================================================================================== features
def x5_features(df):
    """seg_v10's X5 covariates (identical construction to seg_v10._x5_dev; unit-tested on the dev half)."""
    age = df["age"].to_numpy(dtype=float)
    valid = (age >= 14) & (age <= 100)
    g = df["gender"].astype(str)
    t0 = pd.Timestamp("2017-01-01")
    iss = (pd.to_datetime(df["first_issue_date"], errors="coerce") - t0).dt.days.to_numpy(dtype=float)
    red = (pd.to_datetime(df["first_redeem_date"], errors="coerce") - t0).dt.days.to_numpy(dtype=float)
    return np.column_stack([np.where(valid, age, np.nan), (~valid).astype(float), (g == "F").astype(float),
                            (g == "M").astype(float), iss, red, red - iss, np.isnan(red).astype(float)])


def _sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def _dev_replay(data):
    """(X_replay, arm_replay, y_replay, source_rows, model_mask_rows_sha) of seg_v10's dev replay pool."""
    X, arm, y, rows, _ = SV._lenta_dev() if data == "lenta" else SV._x5_dev()
    msk = SV.model_split(arm)
    return X, arm, y, rows, msk


def _score(models, X):
    return models[1].predict_proba(X)[:, 1] - models[0].predict_proba(X)[:, 1]


def assign_segments(score, cuts):
    cuts = np.asarray(cuts, dtype=float)
    if len(cuts) and not np.all(np.diff(cuts) > 0):
        raise RuntimeError("frozen cut points are not strictly increasing")
    return np.searchsorted(cuts, np.asarray(score, dtype=float), side="right").astype(np.int64)


# =============================================================================================== freeze (dev only)
def freeze_segmentations(out_dir=None, overwrite=False):
    """Fit and freeze the score models, cut points and problems from the DEV halves (never touches an eval half)."""
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    out_dir = FROZEN_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jf = out_dir / FROZEN_JSON.name
    if jf.exists() and not overwrite:
        raise RuntimeError(f"{jf} exists: the frozen segmentation is never silently rewritten")
    res = {"written_by": "seg_v10_eval.freeze_segmentations (dev halves only)", "sklearn_version": sklearn.__version__,
           "numpy_version": np.__version__, "hgb_params": HGB_PARAMS, "model_split_seed": SV.MODEL_SPLIT_SEED,
           "model_frac": SV.MODEL_FRAC, "tie_seed": SV.TIE_SEED, "cost_G": SV.COST_G,
           "budget_fracs": list(SV.SR_BUDGETS), "assignment": "segment = #{j : cut_j <= score} (searchsorted right)",
           "data": {}}
    for data in ("x5", "lenta"):
        X, arm, y, rows, msk = _dev_replay(data)
        models = []
        for a in (0, 1):
            sel = msk & (arm == a)
            clf = HistGradientBoostingClassifier(**HGB_PARAMS)
            clf.fit(X[sel], y[sel].astype(int))
            models.append(clf)
        rp = ~msk
        score = _score(models, X[rp])
        ref = SV.fit_scores(data)["score"]
        if not np.array_equal(score, ref):
            raise RuntimeError(f"{data}: frozen model does not reproduce seg_v10.fit_scores on the dev replay pool")
        pk = out_dir / f"score_model_{data}.pkl"
        pk.write_bytes(pickle.dumps(models, protocol=4))
        per_S = {}
        for S in S_GRID:
            lab = SV.seg_labels(score, S)
            cuts = [float(score[lab == j].min()) for j in range(1, S)]
            if not np.all(np.diff(cuts) > 0):
                raise RuntimeError(f"{data} S={S}: tied cut points")
            seg = assign_segments(score, cuts)
            ps = np.bincount(seg * 2 + arm[rp], minlength=2 * S).reshape(S, 2)
            w_eq = np.bincount(lab, minlength=S) / len(lab)
            sp = SV.seg_problems(w_eq)
            per_S[str(S)] = {"cuts_hex": [c.hex() for c in cuts], "layer": layer_name(data, S),
                             "dev_rows_reassigned_vs_seg_v10": int((seg != lab).sum()),
                             "dev_pool_sizes_frozen_rule": ps.tolist(),
                             "cost": sp.cost.tolist(), "budgets": sp.budgets.tolist(), "qids": list(sp.qids),
                             "budget_frac": list(sp.budget_frac)}
        res["data"][data] = {"features": list(SV.LENTA_FEATURES) if data == "lenta" else
                             ["age_valid", "age_invalid", "g_F", "g_M", "issue_day", "redeem_day",
                              "redeem_minus_issue", "no_redeem"],
                             "n_dev": int(len(arm)), "n_model": int(msk.sum()), "n_dev_replay": int(rp.sum()),
                             "model_rows_sha": hashlib.sha256(np.ascontiguousarray(rows[msk]).tobytes()).hexdigest(),
                             "model_pickle": pk.name, "model_pickle_sha256": file_sha256(str(pk)),
                             "dev_replay_score_sha256": _sha_arr(score), "S": per_S}
    jf.write_text(json.dumps(res, indent=1))
    return res


def load_frozen_seg(path=None, expect_sha256=None):
    p = FROZEN_JSON if path is None else path
    if expect_sha256 is not None and file_sha256(str(p)) != expect_sha256:
        raise RuntimeError("frozen segmentation json differs from the locked sha256")
    fz = json.loads(p.read_text())
    return fz


def _eval_rows(data, eval_task_id, layer, gate_path):
    """Gate first, then bind the table to the authorised layer (external reviewer v10 lock r2): ``data`` must be the table of
    ``layer`` (layer = '<DATA>-SR<S>'), so an authorised task cannot be used to read another table."""
    if data not in ("x5", "lenta"):
        raise PermissionError(f"unknown v10 table {data!r}")
    eval_gate(eval_task_id, layer, gate_path)
    if not (isinstance(layer, str) and layer.startswith(data.upper() + "-SR")):
        raise PermissionError(f"table {data!r} does not match the authorised layer {layer!r}")
    return _read_eval_rows_raw(data)


def read_eval_rows(data, eval_task_id, layer, gate_path=None):
    """Public eval reader: ALWAYS runs ``eval_gate`` (locked v10 addendum, registered task, matching table / S)
    before any eval outcome is read (external reviewer v10 lock r1 F1)."""
    return _eval_rows(data, eval_task_id, layer, gate_path)


def _load_models(fz, data, base_dir=None):
    d = fz["data"][data]
    pk = (FROZEN_DIR if base_dir is None else base_dir) / d["model_pickle"]
    if file_sha256(str(pk)) != d["model_pickle_sha256"]:
        raise RuntimeError(f"{data}: score-model pickle differs from the frozen sha256")
    return pickle.loads(pk.read_bytes())


def frozen_problems(fz, data, S):
    from ..baselines.fdc_dp import SegProblems
    e = fz["data"][data]["S"][str(int(S))]
    return SegProblems(cost=np.array(e["cost"], dtype=np.int64), budgets=np.array(e["budgets"], dtype=np.int64),
                       qids=list(e["qids"]), budget_frac=list(e["budget_frac"]))


# =============================================================================================== eval rows (gated)
def eval_gate(eval_task_id, layer, gate_path=None):
    """Raise PermissionError unless the v10 gate authorises ``eval_task_id`` for exactly this layer."""
    if eval_task_id is None:
        raise PermissionError(f"{layer} eval half (v10): an authorised v10 eval task id is required")
    try:
        from ..stats.prereg_v10 import addendum_gate
    except ImportError as e:
        raise PermissionError(f"{layer} eval half (v10): no v10 addendum gate available ({e})") from None
    ok, info = addendum_gate(eval_task_id, path=gate_path)
    if not ok:
        raise PermissionError(f"{layer} eval half refused by the v10 addendum gate: {info}")
    ent = (info.get("eval_tasks") or {}).get(eval_task_id) or {}
    if ent.get("layer") != layer:
        raise PermissionError(f"{layer} eval half: task {eval_task_id!r} is registered for layer {ent.get('layer')!r}")
    return info


def lenta_eval_mask():
    """LR9 split (covariate-only strata x arm, seed 6006): eval = complement of the dev half (labels only)."""
    from .lenta_v6 import lr9_labels
    from .pool_replay import split_mask
    from .lenta_v6 import LR9_SPLIT_SEED, _load_lenta
    cov, _ = _load_lenta(True, with_outcome=False)
    seg9, _, _ = lr9_labels(cov)
    return ~split_mask(seg9, cov["treatment"].to_numpy().astype(np.int64), LR9_SPLIT_SEED)


def _read_eval_rows_raw(data):
    """Ungated eval rows (X, arm, y, source_rows).  Private: reached only through ``_eval_rows`` / ``read_eval_rows``,
    which run ``eval_gate`` first.  Not exported."""
    if data == "x5":
        from .data_v8 import x5_provenance_check
        x5_provenance_check()
        df = pd.read_pickle(DATA_FILES_V10["x5_eval_labels"])
        X = x5_features(df)
        arm = df["treatment"].to_numpy().astype(np.int64)
        y = np.load(DATA_FILES_V10["x5_eval_outcome"]).astype(float)
        if len(y) != len(df):
            raise RuntimeError("X5 eval outcome / labels length mismatch")
        return X, arm, y, np.arange(len(df))
    from .lenta_v6 import LENTA_SHA256, LR9_OUTCOME, LR9_SPLIT_SEED, lr9_labels
    from .pool_replay import split_mask
    if file_sha256(DATA_FILES_V10["lenta_tidy"]) != LENTA_SHA256:
        raise RuntimeError("lenta tidy.pkl sha256 differs from the pinned value")
    df = pd.read_pickle(DATA_FILES_V10["lenta_tidy"])
    seg9, _, _ = lr9_labels(df)
    arm_full = df["treatment"].to_numpy().astype(np.int64)
    ev = ~split_mask(seg9, arm_full, LR9_SPLIT_SEED)
    X = df.loc[ev, list(SV.LENTA_FEATURES)].to_numpy(dtype=float)
    y = df.loc[ev, LR9_OUTCOME].to_numpy(dtype=float)
    rows = np.flatnonzero(ev)
    arm = arm_full[ev]
    del df
    return X, arm, y, rows


# =============================================================================================== environment
class SegEnvV10Frozen(SyntheticPoolEnv):
    """Replay population under the FROZEN v10 score model / cut points / problems (PoolReplayEnv interface)."""

    def __init__(self, data="x5", S=16, half="dev", eval_task_id=None, gate_path=None, frozen_path=None,
                 frozen_sha256=None, frozen_dir=None):  # noqa: D401 - no super().__init__
        if data not in ("lenta", "x5"):
            raise ValueError(data)
        if half not in ("dev", "eval"):
            raise ValueError(half)
        layer = layer_name(data, S)
        if half == "eval":
            X, arm, y, rows = _eval_rows(data, eval_task_id, layer, gate_path)      # gate first, then any read
        fz = load_frozen_seg(frozen_path, frozen_sha256)
        models = _load_models(fz, data, frozen_dir)
        if half == "dev":
            Xd, armd, yd, rowsd, msk = _dev_replay(data)
            rp = ~msk
            X, arm, y, rows = Xd[rp], armd[rp], yd[rp], rowsd[rp]
        score = _score(models, X)
        if half == "dev" and _sha_arr(score) != fz["data"][data]["dev_replay_score_sha256"]:
            raise RuntimeError(f"{data}: frozen score model does not reproduce the frozen dev scores")
        cuts = [float.fromhex(c) for c in fz["data"][data]["S"][str(int(S))]["cuts_hex"]]
        self.data, self.layer, self.half, self.split_seed = data, layer, half, SV.MODEL_SPLIT_SEED
        self.S, self.A = int(S), 2
        self.seg = assign_segments(score, cuts)
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
            raise RuntimeError(f"{layer} ({half}): empty pool under the frozen segmentation")
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(2 * S)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = 5000 if data == "lenta" else 2000
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)
        self.problems = frozen_problems(fz, data, S)
        self.score_sha256 = _sha_arr(score)

    def describe(self):
        return {"layer": self.layer, "half": self.half, "S": self.S, "N": self.N,
                "pool_sizes_min": int(self.pool_sizes.min()), "w_range": [float(self.w.min()), float(self.w.max())],
                "cost": self.problems.cost[:, 1].tolist(), "budgets": self.problems.budgets.tolist()}


_ = math
