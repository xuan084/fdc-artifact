"""Lock v4 (round 4): locked_no_eval semantics, v3 reproducibility.

Round 5 bumped LOCK_VERSION to 5; these tests pin version=4 explicitly so the archived v4 rules stay covered."""
import json

import pytest

from dsswm.stats.prereg import (LOCK_VERSION, assert_locked, canonical_hash, check_lock, validate_lock)


def _lock(tmp_path, **kw):
    lock = {"version": 4, "status": "locked", "x": 1, "no_eval_permitted_tasks": ["r4_lr_e1"], **kw}
    lock["sha256"] = canonical_hash(lock)
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    return p


def test_current_version_is_newer_than_4():
    assert LOCK_VERSION >= 5


def test_locked_v4_allows_any_task(tmp_path):
    assert assert_locked(_lock(tmp_path), version=4)["version"] == 4
    assert assert_locked(_lock(tmp_path), version=4, task_id="r4_cr_main_a")["status"] == "locked"


def test_v3_lock_rejected_by_default(tmp_path):
    with pytest.raises(RuntimeError, match="version 3"):
        assert_locked(_lock(tmp_path, version=3), version=4)


def test_locked_no_eval_blocks_confirmatory(tmp_path):
    p = _lock(tmp_path, status="locked_no_eval")
    with pytest.raises(RuntimeError, match="locked_no_eval"):
        assert_locked(p, version=4)
    with pytest.raises(RuntimeError, match="locked_no_eval"):
        assert_locked(p, version=4, task_id="r4_cr_main_a")
    assert assert_locked(p, version=4, task_id="r4_lr_e1")["status"] == "locked_no_eval"


def test_validate_lock_integrity(tmp_path):
    p = _lock(tmp_path, status="locked_no_eval")
    lock = json.loads(p.read_text())
    assert validate_lock(lock, 4)["version"] == 4
    lock["x"] = 2
    with pytest.raises(RuntimeError, match="hash"):
        validate_lock(lock, 4)
    bad = json.loads(_lock(tmp_path, status="provisional").read_text())
    with pytest.raises(RuntimeError, match="status"):
        validate_lock(bad, 4)


def test_hash_tamper_detected_v4(tmp_path):
    p = _lock(tmp_path)
    d = json.loads(p.read_text()); d["x"] = 2
    with pytest.raises(RuntimeError, match="hash"):
        check_lock(d, 4)
