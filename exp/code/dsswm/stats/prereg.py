"""Pre-registration lock helpers (methodology 6.1 step 4; round 5: lock version 5).

Every evaluation task must call `assert_locked(version=5, task_id=...)` at start-up (or `lock_gate(task_id)`, which
returns the refusal reason instead of raising, so that the task can write summary.status='skipped_by_lock'). It
refuses to run unless the lock file of that version has the requested version, a sha256 that matches the canonical
content hash, and a status that permits the task:

* status == 'locked'          -> every evaluation task may run;
* status == 'locked_no_eval'  -> (v4+) the lock is written for the record only (gate verdict != GO); only tasks listed
                                 in lock['no_eval_permitted_tasks'] may run, and only when they pass their own task_id.

Version 5 adds, on top of the v4 rules:
* `validate_lock(lock, 5)` also checks the v5 schema (V5_REQUIRED_KEYS), a 40-hex code-freeze commit, the delta split
  of the FDC ledger (delta_main + delta_var == delta) and that the status agrees with the gate verdict
  (GO <-> 'locked');
* `assert_locked(version=5)` additionally verifies that every frozen non-test module of the dsswm package still has the
  sha256 recorded in lock['code_sha256'] (method code cannot drift after the lock).

Archived locks stay reproducible: `lock_path(version)` maps 3 -> plan/history/round3_final/prereg_lock.json and
4 -> plan/history/r4_final/prereg_lock.json, and `assert_locked(version=4)` / `check_lock(lock, 4)` keep the v4
behaviour unchanged. `validate_lock` checks integrity only and never authorises evaluation by itself.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = WS_ROOT / "plan" / "prereg_lock.json"
LOCK_VERSION = 5
LOCKED_STATUSES = ("locked", "locked_no_eval")
ARCHIVE_PATHS = {
    3: WS_ROOT / "plan" / "history" / "round3_final" / "prereg_lock.json",
    4: WS_ROOT / "plan" / "history" / "r4_final" / "prereg_lock.json",
}
V5_REQUIRED_KEYS = (
    "version", "status", "candidate_id", "gate_decision", "fdc_spec", "rigorous_set_R", "rival_configs",
    "plugin_comparators", "excluded_methods", "endpoints", "sap", "c4", "cr_n_streams", "seed_manifest",
    "input_sha256", "eval_tasks", "no_eval_permitted_tasks", "declarations", "code_sha256", "git_commit", "sha256",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_hash(lock: dict) -> str:
    """sha256 over the canonical JSON of the lock with the `sha256` field removed."""
    body = {k: v for k, v in lock.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def lock_path(version: int = LOCK_VERSION) -> Path:
    """Live lock for the current version, archived copy for older versions."""
    if version == LOCK_VERSION:
        return LOCK_PATH
    if version in ARCHIVE_PATHS:
        return ARCHIVE_PATHS[version]
    raise RuntimeError(f"no lock file known for version {version}")


def load_lock(path: Path | str = LOCK_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _check_version_and_hash(lock: dict, version: int) -> None:
    if lock.get("version") != version:
        raise RuntimeError(f"prereg lock version {lock.get('version')} != {version}: refuse to touch evaluation seeds")
    h = canonical_hash(lock)
    if lock.get("sha256") != h:
        raise RuntimeError(f"prereg lock hash mismatch: stored {lock.get('sha256')} vs computed {h}")


def _validate_v5_schema(lock: dict) -> None:
    missing = [k for k in V5_REQUIRED_KEYS if k not in lock]
    if missing:
        raise RuntimeError(f"prereg lock v5 schema: missing keys {missing}")
    if not isinstance(lock.get("git_commit"), str) or not _HEX40.match(lock["git_commit"]):
        raise RuntimeError(f"prereg lock v5: code-freeze git_commit {lock.get('git_commit')!r} is not a 40-hex hash")
    led = (lock.get("fdc_spec") or {}).get("ledger") or {}
    dm, dv, d = led.get("delta_main"), led.get("delta_var"), led.get("delta")
    if None in (dm, dv, d) or abs(dm + dv - d) > 1e-12:
        raise RuntimeError(f"prereg lock v5: ledger delta split {dm} + {dv} != {d}")
    if not lock.get("rigorous_set_R"):
        raise RuntimeError("prereg lock v5: empty rigorous rival set R")
    verdict = (lock.get("gate_decision") or {}).get("verdict")
    want = "locked" if verdict == "GO" else "locked_no_eval"
    if lock.get("status") != want:
        raise RuntimeError(f"prereg lock v5: status {lock.get('status')!r} inconsistent with gate verdict {verdict!r}")
    if not lock.get("code_sha256"):
        raise RuntimeError("prereg lock v5: empty code_sha256")


def validate_lock(lock: dict, version: int = LOCK_VERSION) -> dict:
    """Integrity check only: version, sha256, a locked status (and for v>=5 the v5 schema)."""
    if lock.get("version") != version:
        raise RuntimeError(f"prereg lock version {lock.get('version')} != {version}: refuse to touch evaluation seeds")
    allowed = LOCKED_STATUSES if version >= 4 else ("locked",)
    if lock.get("status") not in allowed:
        raise RuntimeError(f"prereg lock status={lock.get('status')!r} (need one of {allowed})")
    _check_version_and_hash(lock, version)
    if version >= 5:
        _validate_v5_schema(lock)
    return lock


def check_lock(lock: dict, version: int = LOCK_VERSION, task_id: str | None = None) -> dict:
    """Authorise an evaluation run. v3 behaviour is unchanged (status must be 'locked'); v4 rules unchanged."""
    if lock.get("version") != version:
        raise RuntimeError(f"prereg lock version {lock.get('version')} != {version}: refuse to touch evaluation seeds")
    status = lock.get("status")
    if status == "locked":
        pass
    elif status == "locked_no_eval" and version >= 4:
        permitted = lock.get("no_eval_permitted_tasks") or []
        if task_id is None or task_id not in permitted:
            raise RuntimeError(f"prereg lock status='locked_no_eval': confirmatory evaluation is skipped by the gate "
                               f"(task_id={task_id!r} not in no_eval_permitted_tasks={permitted})")
    else:
        raise RuntimeError(f"prereg lock status={status!r} (need 'locked'): refuse to run evaluation")
    h = canonical_hash(lock)
    if lock.get("sha256") != h:
        raise RuntimeError(f"prereg lock hash mismatch: stored {lock.get('sha256')} vs computed {h}")
    if version >= 5:
        _validate_v5_schema(lock)
    return lock


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen_code_drift(lock: dict, code_root: Path | str = CODE_ROOT) -> list[str]:
    """Frozen dsswm modules (tests excluded) whose current sha256 differs from the lock (or that disappeared)."""
    root = Path(code_root)
    bad = []
    for rel, h in sorted((lock.get("code_sha256") or {}).items()):
        if not rel.startswith("dsswm/") or rel.startswith("dsswm/tests/"):
            continue
        p = root / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
        elif _sha_file(p) != h:
            bad.append(rel)
    return bad


def assert_locked(path: Path | str | None = None, version: int = LOCK_VERSION, task_id: str | None = None,
                  verify_code: bool | None = None, code_root: Path | str = CODE_ROOT) -> dict:
    """Load the lock of `version` (live file for the current version, archive otherwise) and authorise `task_id`.

    For v>=5 the frozen dsswm modules are also compared with lock['code_sha256'] (verify_code defaults to True).
    """
    lock = check_lock(load_lock(lock_path(version) if path is None else path), version, task_id)
    if verify_code is None:
        verify_code = version >= 5
    if verify_code:
        drift = frozen_code_drift(lock, code_root)
        if drift:
            raise RuntimeError(f"prereg lock v{version}: frozen code changed after the lock: {drift[:10]}")
    return lock


def lock_gate(task_id: str, version: int = LOCK_VERSION, path: Path | str | None = None, **kw):
    """(ok, lock_or_reason): non-raising wrapper so evaluation tasks can write summary.status='skipped_by_lock'."""
    try:
        return True, assert_locked(path, version, task_id, **kw)
    except (RuntimeError, OSError, ValueError) as e:
        return False, str(e)
