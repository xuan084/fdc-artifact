"""Lock-v6 ADDENDUM gate (new file; ``stats/prereg.py`` and the v5 lock are untouched).

The addendum is append-only to lock v5: it never edits ``plan/prereg_lock.json`` and cannot change the v5 verdict.
An addendum evaluation task (or the analysis) may run only if ALL of the following hold:

1. the v5 lock still validates (``prereg.check_lock(version=5)``) and no v5-frozen dsswm module drifted;
2. ``plan/prereg_lock_v6_addendum.json`` exists, has every key of ``V6_REQUIRED_KEYS``, ``status == 'locked'``,
   ``addendum_to.version == 5`` and ``addendum_to.sha256`` equal to the live v5 sha256, a 40-hex ``git_commit``, and
   its own canonical sha256 matches (same canonical form as ``prereg.canonical_hash``);
3. ``code_sha256`` is NON-EMPTY and every listed file still has that hash;
4. ``input_sha256`` is non-empty and every listed input (frozen rival configs, pilot summaries, v5 reference results)
   still has that hash -- checked at start-up, on every resume and again by the analysis;
5. ``data_sha256`` (raw Criteo tidy.pkl + csv.gz, Lenta tidy.pkl; ``envs/data_v6.py``) is non-empty and every raw data
   file still has its frozen hash -- checked at start-up, on every resume and by the analysis;
6. the task id is listed in ``eval_tasks``.

The DRAFT file (``plan/prereg_lock_v6_addendum_DRAFT.json``, status 'draft') never authorises anything.
``finalize_addendum`` refuses to overwrite an existing locked addendum (a post-lock fix needs a new version file),
and verifies that the code-freeze commit really contains every bound code file with the bound hash.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from . import prereg

ADDENDUM_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v6_addendum.json"
DRAFT_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v6_addendum_DRAFT.json"
V6_REQUIRED_KEYS = ("version", "status", "addendum_to", "v5_verdict_statement", "blocks", "seed_manifest",
                    "replica_check", "eval_tasks", "frozen_configs", "input_sha256", "code_sha256", "data_sha256",
                    "git_commit")

__all__ = ["ADDENDUM_PATH", "DRAFT_PATH", "V6_REQUIRED_KEYS", "check_addendum", "addendum_gate", "v6_code_drift",
           "v6_input_drift", "finalize_addendum", "load_locked_addendum", "commit_code_mismatch"]


def v6_code_drift(add: dict, code_root: Path | str = prereg.CODE_ROOT) -> list[str]:
    root = Path(code_root)
    bad = []
    for rel, h in sorted((add.get("code_sha256") or {}).items()):
        p = root / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
        elif prereg._sha_file(p) != h:
            bad.append(rel)
    return bad


def v6_input_drift(add: dict, ws_root: Path | str = prereg.WS_ROOT) -> list[str]:
    root = Path(ws_root)
    bad = []
    for rel, h in sorted((add.get("input_sha256") or {}).items()):
        p = root / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
        elif prereg._sha_file(p) != h:
            bad.append(rel)
    return bad


def _schema(add: dict, require_commit: bool) -> None:
    missing = [k for k in V6_REQUIRED_KEYS if k not in add]
    if missing:
        raise RuntimeError(f"v6 addendum schema: missing keys {missing}")
    if not add.get("code_sha256"):
        raise RuntimeError("v6 addendum: empty code_sha256")
    if not add.get("input_sha256"):
        raise RuntimeError("v6 addendum: empty input_sha256")
    from ..envs.data_v6 import DATA_FILES
    dsha = add.get("data_sha256") or {}
    if set(dsha) != set(DATA_FILES):
        raise RuntimeError(f"v6 addendum: data_sha256 must bind exactly {sorted(DATA_FILES)}")
    if not add.get("frozen_configs"):
        raise RuntimeError("v6 addendum: empty frozen_configs")
    if require_commit and not (isinstance(add.get("git_commit"), str) and prereg._HEX40.match(add["git_commit"])):
        raise RuntimeError("v6 addendum: git_commit is not a 40-hex hash")


def check_addendum(add: dict, task_id: str | None = None, v5_lock: dict | None = None,
                   code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT,
                   verify_code: bool = True, verify_inputs: bool = True, verify_data: bool = True) -> dict:
    v5 = prereg.check_lock(prereg.load_lock(prereg.lock_path(5)) if v5_lock is None else v5_lock, 5)
    if verify_code:
        drift = prereg.frozen_code_drift(v5, code_root)
        if drift:
            raise RuntimeError(f"v5-frozen code drifted: {drift[:10]}")
    if add.get("status") != "locked":
        raise RuntimeError(f"v6 addendum status={add.get('status')!r} (need 'locked'): refuse to run evaluation")
    _schema(add, require_commit=True)
    to = add.get("addendum_to") or {}
    if to.get("version") != 5 or to.get("sha256") != v5.get("sha256"):
        raise RuntimeError("v6 addendum does not reference the live v5 lock (version / sha256 mismatch)")
    h = prereg.canonical_hash(add)
    if add.get("sha256") != h:
        raise RuntimeError(f"v6 addendum hash mismatch: stored {add.get('sha256')} vs computed {h}")
    if verify_code:
        drift = v6_code_drift(add, code_root)
        if drift:
            raise RuntimeError(f"v6 code drifted after the addendum lock: {drift[:10]}")
    if verify_inputs:
        drift = v6_input_drift(add, ws_root)
        if drift:
            raise RuntimeError(f"v6 inputs drifted after the addendum lock: {drift[:10]}")
    if verify_data:
        from ..envs.data_v6 import data_drift
        drift = data_drift(add.get("data_sha256"))
        if drift:
            raise RuntimeError(f"raw data drifted after the addendum lock: {drift}")
    if task_id is not None and task_id not in (add.get("eval_tasks") or {}):
        raise RuntimeError(f"task {task_id!r} is not an eval task of the v6 addendum")
    return add


def load_locked_addendum(task_id: str | None = None, path: Path | str | None = None, **kw) -> dict:
    return check_addendum(prereg.load_lock(Path(path) if path is not None else ADDENDUM_PATH), task_id, **kw)


def addendum_gate(task_id: str, path: Path | str | None = None, **kw):
    """(ok, addendum_or_reason) -- non-raising, so a task can write summary.status='skipped_by_lock'."""
    try:
        return True, load_locked_addendum(task_id, path, **kw)
    except (RuntimeError, OSError, ValueError) as e:
        return False, str(e)


def _git(*args, cwd):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def commit_code_mismatch(add: dict, git_commit: str, code_root: Path | str = prereg.CODE_ROOT) -> list[str]:
    """Files of code_sha256 whose content at ``git_commit`` differs from the bound hash (or is absent)."""
    root = Path(code_root)
    _git("cat-file", "-e", f"{git_commit}^{{commit}}", cwd=root)
    bad = []
    for rel, h in sorted((add.get("code_sha256") or {}).items()):
        try:
            blob = _git("show", f"{git_commit}:./{rel}", cwd=root)
        except RuntimeError:
            bad.append(f"{rel} (not in commit)")
            continue
        if hashlib.sha256(blob).hexdigest() != h:
            bad.append(rel)
    return bad


def finalize_addendum(git_commit: str, draft_path: Path | str = DRAFT_PATH, out_path: Path | str = ADDENDUM_PATH,
                      locked_by: str = "authors", now: str | None = None,
                      code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT) -> dict:
    """Main-session helper: turn the reviewed DRAFT into the locked addendum.  Refuses if the output already exists
    (a post-lock fix needs a NEW version, never an overwrite), if code or inputs drifted from the draft, or if the
    commit does not contain the bound code."""
    out_path = Path(out_path)
    if out_path.exists():
        raise RuntimeError(f"{out_path} already exists: a locked addendum is never overwritten (new version needed)")
    draft = prereg.load_lock(draft_path)
    if draft.get("status") != "draft":
        raise RuntimeError("finalize_addendum expects a draft")
    if not (isinstance(git_commit, str) and prereg._HEX40.match(git_commit)):
        raise RuntimeError("git_commit must be a 40-hex hash")
    _schema(draft, require_commit=False)
    drift = v6_code_drift(draft, code_root)
    if drift:
        raise RuntimeError(f"v6 code differs from the draft hashes: {drift}")
    drift = v6_input_drift(draft, ws_root)
    if drift:
        raise RuntimeError(f"input files changed since the draft: {drift}")
    from ..envs.data_v6 import data_drift
    drift = data_drift(draft.get("data_sha256"))
    if drift:
        raise RuntimeError(f"raw data changed since the draft: {drift}")
    bad = commit_code_mismatch(draft, git_commit, code_root)
    if bad:
        raise RuntimeError(f"commit {git_commit} does not contain the bound code: {bad}")
    v5 = prereg.check_lock(prereg.load_lock(prereg.lock_path(5)), 5)
    if (draft.get("addendum_to") or {}).get("sha256") != v5["sha256"]:
        raise RuntimeError("draft does not reference the live v5 lock")
    lock = {k: v for k, v in draft.items() if k not in ("sha256", "sha256_draft")}
    lock.update({"status": "locked", "git_commit": git_commit, "locked_by": locked_by,
                 "locked_at": now or datetime.now().isoformat(), "eval_touched_at_lock": False,
                 "draft_sha256": draft.get("sha256_draft")})
    lock["sha256"] = prereg.canonical_hash(lock)
    out_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    return lock
