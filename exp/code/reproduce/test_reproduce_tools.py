"""Unit tests of the ingestion, content-check and reproduction helpers (no data needed).

Run (cwd = exp/code):  python -m pytest reproduce/test_reproduce_tools.py -q
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "reproduce"))

import check_tidy_content as C  # noqa: E402
import ingest_tidy as I  # noqa: E402
import reproduce_frozen as RF  # noqa: E402


def test_content_hash_stable_and_sensitive():
    df = pd.DataFrame({"a": np.arange(5, dtype="int8"), "b": np.linspace(0, 1, 5).astype("float32")})
    h = C.content_sha256(df)
    assert h == C.content_sha256(df.copy())
    df2 = df.copy()
    df2.loc[3, "b"] = np.float32(0.5000001)
    assert C.content_sha256(df2) != h
    assert C.content_sha256(df.astype({"a": "int16"})) != h          # dtype is part of the content
    assert C.content_sha256(df.rename(columns={"a": "x"})) != h      # names too


def test_content_hash_ignores_column_index_storage():
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    with pd.option_context("mode.string_storage", "python"):
        d1 = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    assert C.content_sha256(df) == C.content_sha256(d1)


def test_content_hash_rejects_object_columns():
    with pytest.raises(TypeError):
        C.content_sha256(pd.DataFrame({"s": ["x", "y"]}))


def test_hillstrom_builder_schema(tmp_path):
    raw = tmp_path / "h.csv"
    raw.write_text("recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,visit,conversion,spend\n"
                   "10,2) $100 - $200,142.44,1,0,Surburban,0,Phone,Womens E-Mail,0,0,0\n"
                   "6,3) $200 - $350,329.08,1,1,Rural,1,Web,No E-Mail,1,1,29.99\n")
    df = I.build_hillstrom(raw)
    assert list(df.columns[:4]) == ["treatment", "outcome", "binary", "weight"]
    assert df["treatment"].tolist() == [2, 0] and df["outcome"].tolist() == [0.0, 29.99]
    assert str(df["x_recency"].dtype) == "float32" and str(df["x_zip"].dtype) == "int8"


def test_criteo_builder_dtypes(tmp_path):
    raw = tmp_path / "c.csv.gz"
    cols = I.CRITEO_FLOAT + I.CRITEO_INT
    with gzip.open(raw, "wt") as f:
        f.write(",".join(cols) + "\n" + ",".join(["1.25"] * 12 + ["1", "0", "1", "0"]) + "\n")
    df = I.build_criteo(raw)
    assert list(df.columns) == cols
    assert all(str(df[c].dtype) == "float32" for c in I.CRITEO_FLOAT)
    assert all(str(df[c].dtype) == "int8" for c in I.CRITEO_INT)


def test_pick_seeds_spread_and_registered_only():
    s = list(range(37000, 37200))
    assert RF.pick_seeds(s, 5, None) == [37000, 37050, 37100, 37149, 37199]
    assert len(RF.pick_seeds(s, 10, None)) == 10
    with pytest.raises(SystemExit):
        RF.pick_seeds(s, 1, [12345])


def test_compare_rows_excludes_timing_and_code_hash_only():
    a = {"method": "M", "seed": 1, "eps": 0.1, "N80_pen": 10, "sec": 1.0, "code_sha256": "x", "data_sha256": "d"}
    b = dict(a, sec=2.0, code_sha256="y")
    assert RF.compare_rows(a, b, ("sec",)) == []
    assert [d[0] for d in RF.compare_rows(a, dict(b, N80_pen=11), ("sec",))] == ["N80_pen"]
    assert [d[0] for d in RF.compare_rows(a, dict(b, data_sha256="e"), ("sec",))] == ["data_sha256"]


def test_blocks_cover_headline_tasks():
    tasks = {t for _, ts in RF.BLOCKS.values() for t in ts}
    for t in ("v9a_full_d", "v9b_full_d", "v10a_full_s16", "v10b_full_s64", "v10d_full_x5s64", "v11_obd_full",
              "v12_women_full", "v12_men_full"):
        assert t in tasks


def test_seal_mismatch_fails_task_and_overall():
    assert RF.seal_matches("abc", "abc") and not RF.seal_matches("abc", "abd") and not RF.seal_matches("abc", None)
    assert RF.task_status(True, [], [], [], 4, 4) == "match"
    assert RF.task_status(False, [], [], [], 4, 4) == "SEAL_MISMATCH"         # rows all equal, seal broken
    assert RF.task_status(True, [{"key": 1}], [], [], 4, 4) == "MISMATCH"
    ok = {"task": "a", "status": "match", "sealed_file_matches_seal": True}
    assert RF.overall_match([ok], [])
    assert not RF.overall_match([ok, {"task": "b", "status": "SEAL_MISMATCH", "sealed_file_matches_seal": False}], [])
    # belt and braces: even a task labelled 'match' cannot pass all_match with a failed seal check
    assert not RF.overall_match([dict(ok, sealed_file_matches_seal=False)], [])
    assert not RF.overall_match([ok], ["exp/results/x"])
    assert not RF.overall_match([], [])


def test_v9_frozen_config_mismatch_is_hard_failure():
    class FakeRunner:
        calls = []

        @staticmethod
        def load_frozen(lock=None):
            FakeRunner.calls.append(lock)
            if lock is not None and lock.get("frozen_configs") != {"CR": 1}:
                raise RuntimeError("frozen configs on disk differ from the locked v9 addendum")
            return {"CR": 1}

    assert RF.load_frozen_v9(FakeRunner, {"frozen_configs": {"CR": 1}}) == {"CR": 1}
    FakeRunner.calls.clear()
    with pytest.raises(RF.FrozenConfigMismatch):
        RF.load_frozen_v9(FakeRunner, {"frozen_configs": {"CR": 2}})
    assert FakeRunner.calls == [{"frozen_configs": {"CR": 2}}]               # no fallback reload without the lock


def test_v9_adapter_has_no_silent_fallback():
    import inspect
    src = inspect.getsource(RF.setup_task)
    assert "load_frozen(None)" not in src and "load_frozen_v9(R, lock)" in src
