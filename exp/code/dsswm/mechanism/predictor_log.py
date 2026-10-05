"""Write-ahead logs for the Q family (methodology s.1 "write separation").

predictors.jsonl  -- written by the LEARNER process at the certification time, BEFORE any truth-based scoring.
                     Each line is flushed and fsync'ed. Records carrying truth-derived fields are refused.
results.jsonl     -- written by the HARNESS after the stream ends (truth scoring).
Analysis joins the two files on KEY = (instance, stream, method, arm, problem) and checks the write-ahead order
(every predictor line was logged before the matching result line).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

KEY = ("instance", "stream", "method", "arm", "problem")
# fields only the harness may know (truth / oracle); a predictor record containing any of them is rejected
TRUTH_FIELDS = frozenset({
    "true_gap", "true_regret", "false_cert", "eta", "eta_dec", "eta_arg", "eta_best", "eta_loc", "mu_flip",
    "theta_star", "theta_index", "truth", "r", "r_norm_xi", "lambda_delta", "rho_ucb", "rho_hat_ucb", "g_star",
    "gap_layer", "layer_true", "J_true", "dJ_star", "rem_exact", "identity_err",
})


def _clean(x):
    """JSON-safe: numpy scalars/arrays -> python, inf/nan -> strings."""
    try:
        import numpy as np
        if isinstance(x, np.generic):
            x = x.item()
        elif isinstance(x, np.ndarray):
            x = x.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _truth_keys(rec, path=""):
    hits = []
    if isinstance(rec, dict):
        for k, v in rec.items():
            if k in TRUTH_FIELDS:
                hits.append(path + str(k))
            hits += _truth_keys(v, path + str(k) + ".")
    elif isinstance(rec, list):
        for i, v in enumerate(rec):
            hits += _truth_keys(v, path + f"[{i}].")
    return hits


class JsonlWriter:
    def __init__(self, path, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._seq = 0

    def _append(self, rec: dict) -> dict:
        line = json.dumps(_clean(rec), sort_keys=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        self._seq += 1
        return rec


def _check_key(rec: dict):
    missing = [k for k in KEY if k not in rec]
    if missing:
        raise KeyError(f"record lacks join keys {missing}")


class PredictorLog(JsonlWriter):
    """Learner-side; call write() at the certification time, before the harness scores anything."""

    def write(self, rec: dict) -> dict:
        _check_key(rec)
        bad = _truth_keys(rec)
        if bad:
            raise ValueError(f"predictor record carries truth-only fields: {bad}")
        rec = {**rec, "logged_at": time.time(), "log_pid": os.getpid(), "phase": "pre_scoring"}
        return self._append(rec)


class ResultLog(JsonlWriter):
    """Harness-side truth scoring, written after the stream ends."""

    def write(self, rec: dict) -> dict:
        _check_key(rec)
        rec = {**rec, "scored_at": time.time(), "phase": "scored"}
        return self._append(rec)


def load_jsonl(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # torn last line after a crash: ignored (resumable)
    return out


def key_of(rec: dict) -> tuple:
    return tuple(rec[k] for k in KEY)


def done_keys(path) -> set:
    return {key_of(r) for r in load_jsonl(path) if all(k in r for k in KEY)}


def join(pred_path, res_path, check_order: bool = True) -> list:
    """Inner join on KEY; with check_order every predictor must precede its result (write-ahead)."""
    preds = {}
    for r in load_jsonl(pred_path):
        preds.setdefault(key_of(r), r)
    out, violations = [], []
    for r in load_jsonl(res_path):
        k = key_of(r)
        if k not in preds:
            continue
        p = preds[k]
        if check_order and not (p["logged_at"] <= r["scored_at"]):
            violations.append(k)
        out.append({"key": list(k), "pred": p, "res": r})
    if violations:
        raise RuntimeError(f"write-ahead violated for {len(violations)} keys, e.g. {violations[:3]}")
    return out
