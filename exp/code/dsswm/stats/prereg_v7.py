"""Lock-v7 ADDENDUM gate (new file; v5 lock, v6 addendum and their modules are untouched).

The v7 addendum registers the confirmatory test of FDC-BF (a method devised AFTER the v6 eval results, selected on dev
seeds only) on fresh eval seeds.  It is append-only: it never edits ``plan/prereg_lock.json`` or
``plan/prereg_lock_v6_addendum.json`` and cannot change their verdicts.

A v7 evaluation task (or the analysis) may run only if ALL of the following hold:

1. the v5 lock still validates and no v5-frozen dsswm module drifted (``prereg_v6.check_addendum`` does this);
2. the v6 addendum still validates (status locked, own hash, v6 code / inputs / raw data unchanged) -- v7 imports the
   v6 rival implementations (rect_v6, wor_betting_v6, frontier_runner_v6, lenta_v6, data_v6, v6_replica, v6_seal);
3. ``plan/prereg_lock_v7_addendum.json`` exists, has every key of ``V7_REQUIRED_KEYS``, ``status == 'locked'``,
   ``addendum_to`` names version 5 with the live v5 sha256 AND ``v6_addendum`` names the live v6 addendum sha256, a
   40-hex ``git_commit``, and its own canonical sha256 matches;
4. ``code_sha256`` / ``input_sha256`` / ``data_sha256`` are non-empty and every listed file still has its hash;
5. the task id is listed in ``eval_tasks``.

The DRAFT (``plan/prereg_lock_v7_addendum_DRAFT.json``, status 'draft') never authorises anything.
``finalize_addendum`` refuses to overwrite an existing locked v7 addendum and verifies that the code-freeze commit
contains every bound code file with the bound hash.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import prereg, prereg_v6

ADDENDUM_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v7_addendum.json"
DRAFT_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v7_addendum_DRAFT.json"
V7_REQUIRED_KEYS = ("version", "status", "addendum_to", "v6_addendum", "statement", "primary_method", "blocks",
                    "seed_manifest", "replica_check", "eval_tasks", "frozen_configs", "input_sha256", "code_sha256",
                    "data_sha256", "git_commit")

__all__ = ["ADDENDUM_PATH", "DRAFT_PATH", "V7_REQUIRED_KEYS", "check_addendum", "addendum_gate", "finalize_addendum",
           "load_locked_addendum"]


def _schema(add: dict, require_commit: bool) -> None:
    missing = [k for k in V7_REQUIRED_KEYS if k not in add]
    if missing:
        raise RuntimeError(f"v7 addendum schema: missing keys {missing}")
    for k in ("code_sha256", "input_sha256", "frozen_configs", "eval_tasks"):
        if not add.get(k):
            raise RuntimeError(f"v7 addendum: empty {k}")
    from ..envs.data_v6 import DATA_FILES
    if set(add.get("data_sha256") or {}) != set(DATA_FILES):
        raise RuntimeError(f"v7 addendum: data_sha256 must bind exactly {sorted(DATA_FILES)}")
    if require_commit and not (isinstance(add.get("git_commit"), str) and prereg._HEX40.match(add["git_commit"])):
        raise RuntimeError("v7 addendum: git_commit is not a 40-hex hash")


def _live_v6(v5_lock=None, verify_code=True, verify_data=True):
    return prereg_v6.load_locked_addendum(None, v5_lock=v5_lock, verify_code=verify_code, verify_inputs=verify_code,
                                          verify_data=verify_data)


def check_addendum(add: dict, task_id: str | None = None, v5_lock: dict | None = None, v6_add: dict | None = None,
                   code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT,
                   verify_code: bool = True, verify_inputs: bool = True, verify_data: bool = True) -> dict:
    v5 = prereg.check_lock(prereg.load_lock(prereg.lock_path(5)) if v5_lock is None else v5_lock, 5)
    v6 = _live_v6(v5, verify_code, verify_data) if v6_add is None else v6_add
    if add.get("status") != "locked":
        raise RuntimeError(f"v7 addendum status={add.get('status')!r} (need 'locked'): refuse to run evaluation")
    _schema(add, require_commit=True)
    to = add.get("addendum_to") or {}
    if to.get("version") != 5 or to.get("sha256") != v5.get("sha256"):
        raise RuntimeError("v7 addendum does not reference the live v5 lock (version / sha256 mismatch)")
    if (add.get("v6_addendum") or {}).get("sha256") != v6.get("sha256"):
        raise RuntimeError("v7 addendum does not reference the live v6 addendum (sha256 mismatch)")
    h = prereg.canonical_hash(add)
    if add.get("sha256") != h:
        raise RuntimeError(f"v7 addendum hash mismatch: stored {add.get('sha256')} vs computed {h}")
    if verify_code:
        drift = prereg_v6.v6_code_drift(add, code_root)
        if drift:
            raise RuntimeError(f"v7 code drifted after the addendum lock: {drift[:10]}")
    if verify_inputs:
        drift = prereg_v6.v6_input_drift(add, ws_root)
        if drift:
            raise RuntimeError(f"v7 inputs drifted after the addendum lock: {drift[:10]}")
    if verify_data:
        from ..envs.data_v6 import data_drift
        drift = data_drift(add.get("data_sha256"))
        if drift:
            raise RuntimeError(f"raw data drifted after the addendum lock: {drift}")
    if task_id is not None and task_id not in (add.get("eval_tasks") or {}):
        raise RuntimeError(f"task {task_id!r} is not an eval task of the v7 addendum")
    return add


def lock_history_check(path: Path | str) -> str:
    """external reviewer v7 review item 2: the locked addendum must be committed, its file must have been committed EXACTLY ONCE
    (never modified or re-added afterwards) and the working copy must equal that commit's bytes.  Returns the lock
    commit sha."""
    import subprocess
    p = Path(path).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(p.parent), capture_output=True)
    if top.returncode != 0:
        raise RuntimeError("v7 addendum: not inside a git repository")
    root = Path(top.stdout.decode().strip())
    rel = p.relative_to(root)
    log = subprocess.run(["git", "log", "--format=%H", "--", str(rel)], cwd=str(root), capture_output=True)
    commits = log.stdout.decode().split()
    if len(commits) != 1:
        raise RuntimeError(f"v7 addendum: lock file must be committed exactly once (found {len(commits)} commits)")
    blob = subprocess.run(["git", "show", f"{commits[0]}:{rel}"], cwd=str(root), capture_output=True)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        raise RuntimeError("v7 addendum: lock file differs from its original commit")
    return commits[0]


def load_locked_addendum(task_id: str | None = None, path: Path | str | None = None, require_history: bool = True,
                         **kw) -> dict:
    p = Path(path) if path is not None else ADDENDUM_PATH
    add = check_addendum(prereg.load_lock(p), task_id, **kw)
    if require_history:
        lock_history_check(p)
    return add


def addendum_gate(task_id: str, path: Path | str | None = None, **kw):
    """(ok, addendum_or_reason) -- non-raising, so a task can write summary.status='skipped_by_lock'."""
    try:
        return True, load_locked_addendum(task_id, path, **kw)
    except (RuntimeError, OSError, ValueError) as e:
        return False, str(e)


def finalize_addendum(git_commit: str, draft_path: Path | str = DRAFT_PATH, out_path: Path | str = ADDENDUM_PATH,
                      locked_by: str = "authors", now: str | None = None,
                      code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT) -> dict:
    out_path = Path(out_path)
    if out_path.exists():
        raise RuntimeError(f"{out_path} already exists: a locked addendum is never overwritten (new version needed)")
    draft = prereg.load_lock(draft_path)
    if draft.get("status") != "draft":
        raise RuntimeError("finalize_addendum expects a draft")
    if not (isinstance(git_commit, str) and prereg._HEX40.match(git_commit)):
        raise RuntimeError("git_commit must be a 40-hex hash")
    _schema(draft, require_commit=False)
    for name, drift in (("code", prereg_v6.v6_code_drift(draft, code_root)),
                        ("inputs", prereg_v6.v6_input_drift(draft, ws_root))):
        if drift:
            raise RuntimeError(f"v7 {name} differ from the draft hashes: {drift}")
    from ..envs.data_v6 import data_drift
    drift = data_drift(draft.get("data_sha256"))
    if drift:
        raise RuntimeError(f"raw data changed since the draft: {drift}")
    bad = prereg_v6.commit_code_mismatch(draft, git_commit, code_root)
    if bad:
        raise RuntimeError(f"commit {git_commit} does not contain the bound code: {bad}")
    v5 = prereg.check_lock(prereg.load_lock(prereg.lock_path(5)), 5)
    v6 = _live_v6(v5)
    if (draft.get("addendum_to") or {}).get("sha256") != v5["sha256"]:
        raise RuntimeError("draft does not reference the live v5 lock")
    if (draft.get("v6_addendum") or {}).get("sha256") != v6["sha256"]:
        raise RuntimeError("draft does not reference the live v6 addendum")
    lock = {k: v for k, v in draft.items() if k not in ("sha256", "sha256_draft")}
    lock.update({"status": "locked", "git_commit": git_commit, "locked_by": locked_by,
                 "locked_at": now or datetime.now().isoformat(), "eval_touched_at_lock": False,
                 "draft_sha256": draft.get("sha256_draft")})
    lock["sha256"] = prereg.canonical_hash(lock)
    out_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    return lock
