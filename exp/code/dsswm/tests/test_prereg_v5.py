"""Lock v5 (round 5): schema, status <-> gate verdict, code-drift guard, archived v4/v3 reproducibility."""
import hashlib
import json

import pytest

from dsswm.stats.prereg import (ARCHIVE_PATHS, LOCK_PATH, LOCK_VERSION, V5_REQUIRED_KEYS, assert_locked,
                                canonical_hash, check_lock, frozen_code_drift, load_lock, lock_gate, lock_path,
                                validate_lock)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _code(tmp_path):
    root = tmp_path / "code"
    (root / "dsswm" / "baselines").mkdir(parents=True)
    (root / "dsswm" / "tests").mkdir(parents=True)
    (root / "dsswm" / "baselines" / "fdc.py").write_bytes(b"x = 1\n")
    (root / "dsswm" / "tests" / "test_x.py").write_bytes(b"y = 1\n")
    (root / "run_r5_cr_main.py").write_bytes(b"z = 1\n")
    return root, {"dsswm/baselines/fdc.py": _sha(b"x = 1\n"), "dsswm/tests/test_x.py": _sha(b"y = 1\n"),
                  "run_r5_cr_main.py": _sha(b"z = 1\n")}


def _v5(code_sha=None, **kw):
    lock = {k: {} for k in V5_REQUIRED_KEYS}
    lock.update({
        "version": 5, "status": "locked", "candidate_id": "cand_fdc",
        "gate_decision": {"verdict": "GO"},
        "fdc_spec": {"ledger": {"delta_main": 0.045, "delta_var": 0.005, "delta": 0.05}},
        "rigorous_set_R": ["B1", "QFC-pool"], "cr_n_streams": 200, "eval_tasks": {"r5_cr_main_a": {}},
        "no_eval_permitted_tasks": [], "declarations": [], "code_sha256": code_sha or {"dsswm/a.py": "0" * 64},
        "git_commit": "a" * 40,
    })
    lock.update(kw)
    lock["sha256"] = canonical_hash(lock)
    return lock


def _write(tmp_path, lock, name="lock.json"):
    p = tmp_path / name
    p.write_text(json.dumps(lock))
    return p


def test_version_and_paths():
    assert LOCK_VERSION == 5
    assert lock_path(5) == LOCK_PATH
    assert lock_path(4) == ARCHIVE_PATHS[4] and "r4_final" in str(lock_path(4))
    assert "round3_final" in str(lock_path(3))
    with pytest.raises(RuntimeError):
        lock_path(2)


def test_valid_v5_lock_passes():
    assert validate_lock(_v5())["version"] == 5
    assert check_lock(_v5(), task_id="r5_cr_main_a")["status"] == "locked"


@pytest.mark.parametrize("key", ["fdc_spec", "sap", "endpoints", "c4", "seed_manifest", "input_sha256", "git_commit"])
def test_missing_v5_key_rejected(key):
    lock = _v5()
    del lock[key]
    lock["sha256"] = canonical_hash(lock)
    with pytest.raises(RuntimeError, match="missing keys|git_commit"):
        validate_lock(lock)


def test_code_commit_must_be_recorded():
    with pytest.raises(RuntimeError, match="git_commit"):
        validate_lock(_v5(git_commit=None))
    with pytest.raises(RuntimeError, match="git_commit"):
        validate_lock(_v5(git_commit="abc"))


def test_delta_split_checked():
    bad = _v5(fdc_spec={"ledger": {"delta_main": 0.045, "delta_var": 0.01, "delta": 0.05}})
    with pytest.raises(RuntimeError, match="delta split"):
        validate_lock(bad)


def test_status_must_match_gate_verdict():
    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_lock(_v5(gate_decision={"verdict": "NO_GO"}))           # NO_GO but 'locked'
    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_lock(_v5(status="locked_no_eval"))                     # GO but 'locked_no_eval'
    ok = _v5(gate_decision={"verdict": "NO_GO"}, status="locked_no_eval")
    assert validate_lock(ok)["status"] == "locked_no_eval"


def test_non_locked_status_blocks(tmp_path):
    for st in ("draft", "provisional", None):
        p = _write(tmp_path, _v5(status=st))
        with pytest.raises(RuntimeError, match="status"):
            assert_locked(p, version=5, task_id="r5_cr_main_a", verify_code=False)
        ok, why = lock_gate("r5_cr_main_a", path=p, verify_code=False)
        assert not ok and "status" in why


def test_locked_no_eval_blocks_eval_tasks(tmp_path):
    p = _write(tmp_path, _v5(gate_decision={"verdict": "NO_GO"}, status="locked_no_eval"))
    for t in ("r5_cr_main_a", "r5_cr_plugin_b", "r5_cr_factorial", None):
        with pytest.raises(RuntimeError, match="locked_no_eval"):
            assert_locked(p, version=5, task_id=t, verify_code=False)
    ok, why = lock_gate("r5_cr_main_a", path=p, verify_code=False)
    assert ok is False and "locked_no_eval" in why


def test_hash_tamper_detected(tmp_path):
    lock = _v5()
    lock["cr_n_streams"] = 400
    with pytest.raises(RuntimeError, match="hash"):
        check_lock(lock)
    p = _write(tmp_path, lock)
    ok, why = lock_gate("r5_cr_main_a", path=p, verify_code=False)
    assert not ok and "hash" in why


def test_v4_lock_rejected_by_v5_default(tmp_path):
    lock = {"version": 4, "status": "locked", "x": 1}
    lock["sha256"] = canonical_hash(lock)
    with pytest.raises(RuntimeError, match="version 4"):
        assert_locked(_write(tmp_path, lock))


def test_code_drift_guard(tmp_path):
    root, shas = _code(tmp_path)
    p = _write(tmp_path, _v5(code_sha=shas))
    assert assert_locked(p, version=5, task_id="r5_cr_main_a", code_root=root)["version"] == 5
    # test files and runner scripts may change (not frozen method code)
    (root / "dsswm" / "tests" / "test_x.py").write_bytes(b"y = 2\n")
    (root / "run_r5_cr_main.py").write_bytes(b"z = 2\n")
    assert frozen_code_drift(load_lock(p), root) == []
    # frozen method code may not
    (root / "dsswm" / "baselines" / "fdc.py").write_bytes(b"x = 2\n")
    assert frozen_code_drift(load_lock(p), root) == ["dsswm/baselines/fdc.py"]
    with pytest.raises(RuntimeError, match="frozen code changed"):
        assert_locked(p, version=5, task_id="r5_cr_main_a", code_root=root)
    ok, why = lock_gate("r5_cr_main_a", path=p, code_root=root)
    assert not ok and "fdc.py" in why
    (root / "dsswm" / "baselines" / "fdc.py").unlink()
    assert frozen_code_drift(load_lock(p), root) == ["dsswm/baselines/fdc.py (missing)"]


def test_archived_v4_lock_still_valid():
    p = lock_path(4)
    if not p.exists():
        pytest.skip("v4 archive not present")
    lock = load_lock(p)
    assert validate_lock(lock, 4)["version"] == 4
    st = lock["status"]
    if st == "locked":
        assert check_lock(lock, 4, task_id="r4_cr_main_a")
    else:
        with pytest.raises(RuntimeError, match="locked_no_eval"):
            check_lock(lock, 4, task_id="r4_cr_main_a")
    assert assert_locked(version=4, task_id=(lock.get("no_eval_permitted_tasks") or [None])[0]
                         if st == "locked_no_eval" else None)["version"] == 4


def test_archived_v3_lock_still_valid():
    p = lock_path(3)
    if not p.exists():
        pytest.skip("v3 archive not present")
    assert check_lock(load_lock(p), 3)["version"] == 3


def test_live_lock_if_v5():
    if not LOCK_PATH.exists():
        pytest.skip("no live lock")
    lock = load_lock(LOCK_PATH)
    if lock.get("version") != 5 or not lock.get("git_commit"):
        pytest.skip("live lock is not a committed v5 lock yet")
    assert validate_lock(lock, 5)
    assert frozen_code_drift(lock) == []
    if lock["status"] == "locked":
        ok, got = lock_gate("r5_cr_main_a")
        assert ok, got
    else:
        ok, why = lock_gate("r5_cr_main_a")
        assert not ok and "locked_no_eval" in why
