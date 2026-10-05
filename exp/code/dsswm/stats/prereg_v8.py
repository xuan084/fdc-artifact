"""Lock-v8 ADDENDUM gate (new file; the v5 lock, the v6 / v7 addenda and their modules are untouched).

The v8 addendum registers ONE confirmatory test (block A: FDC-BF on the untouched X5 RetailHero eval half, seeds
35000-35199) and two descriptive blocks (B: CR12 scale, C: post-hoc paired rivals on the v7 block-A CR9 streams).  It is
append-only and cannot change any earlier verdict.

A v8 evaluation task (or the analysis) may run only if ALL of the following hold:

1. the v7 addendum still validates (``prereg_v7.load_locked_addendum``: v5 lock, v5-frozen code, v6 addendum, v7
   code / raw data, v7 lock-file history) -- v8 imports v6 / v7 modules unchanged.  v7 INPUT drift is checked
   separately: the set of drifted v7 inputs must equal ``v7_known_input_drift`` of the v8 addendum exactly, each with its
   recorded current sha256 (one known case: the dated erratum appended to ``exp/results/full/v6_summary.md`` in commit
   e740f421 after the v7 lock; disclosed in the v8 addendum).  Any other or further drift refuses;
2. ``plan/prereg_lock_v8_addendum.json`` exists, has every key of ``V8_REQUIRED_KEYS``, ``status == 'locked'``,
   references the live v5 lock, v6 addendum and v7 addendum sha256, has a 40-hex ``git_commit`` and its own canonical
   sha256 matches;
3. ``code_sha256`` / ``input_sha256`` / ``data_sha256`` are non-empty, ``data_sha256`` binds exactly
   ``data_v8.DATA_FILES_V8`` and every listed file still has its hash;
4. the lock file was committed exactly once and equals that commit's bytes (git anchor, as v7);
5. the task id is listed in ``eval_tasks``.

The DRAFT (``plan/prereg_lock_v8_addendum_DRAFT.json``, status 'draft') never authorises anything.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import prereg, prereg_v6, prereg_v7

ADDENDUM_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v8_addendum.json"
DRAFT_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v8_addendum_DRAFT.json"
V8_REQUIRED_KEYS = ("version", "status", "addendum_to", "v6_addendum", "v7_addendum", "v7_known_input_drift", "statement", "primary_method",
                    "blocks", "seed_manifest", "replica_check", "eval_tasks", "frozen_configs", "input_sha256",
                    "code_sha256", "data_sha256", "git_commit")

__all__ = ["ADDENDUM_PATH", "DRAFT_PATH", "V8_REQUIRED_KEYS", "check_addendum", "addendum_gate", "finalize_addendum",
           "load_locked_addendum", "lock_history_check", "v7_input_drift_check"]


def _schema(add: dict, require_commit: bool) -> None:
    missing = [k for k in V8_REQUIRED_KEYS if k not in add]
    if missing:
        raise RuntimeError(f"v8 addendum schema: missing keys {missing}")
    for k in ("code_sha256", "input_sha256", "frozen_configs", "eval_tasks"):
        if not add.get(k):
            raise RuntimeError(f"v8 addendum: empty {k}")
    from ..envs.data_v8 import DATA_FILES_V8
    if set(add.get("data_sha256") or {}) != set(DATA_FILES_V8):
        raise RuntimeError(f"v8 addendum: data_sha256 must bind exactly {sorted(DATA_FILES_V8)}")
    if require_commit and not (isinstance(add.get("git_commit"), str) and prereg._HEX40.match(add["git_commit"])):
        raise RuntimeError("v8 addendum: git_commit is not a 40-hex hash")


def _live_v7(verify_code=True, verify_data=True, known_drift=None):
    v7 = prereg_v7.load_locked_addendum(None, verify_code=verify_code, verify_inputs=False, verify_data=verify_data)
    if verify_code:
        v7_input_drift_check(v7, known_drift or {})
    return v7


def v7_input_drift_check(v7: dict, known: dict, ws_root: Path | str = prereg.WS_ROOT) -> None:
    """The drifted v7 inputs must be exactly the disclosed ones, each at its disclosed current sha256."""
    drift = set(prereg_v6.v6_input_drift(v7, ws_root))
    if drift != set(known):
        raise RuntimeError(f"v7 input drift {sorted(drift)} != disclosed v7_known_input_drift {sorted(known)}")
    for rel, d in known.items():
        if prereg._sha_file(Path(ws_root) / rel) != d.get("current_sha256") or \
                (v7.get("input_sha256") or {}).get(rel) != d.get("locked_sha256"):
            raise RuntimeError(f"v7 input {rel}: hashes differ from the disclosed drift record")


def check_addendum(add: dict, task_id: str | None = None, v7_add: dict | None = None,
                   code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT,
                   verify_code: bool = True, verify_inputs: bool = True, verify_data: bool = True) -> dict:
    v7 = _live_v7(verify_code, verify_data, add.get("v7_known_input_drift")) if v7_add is None else v7_add
    if add.get("status") != "locked":
        raise RuntimeError(f"v8 addendum status={add.get('status')!r} (need 'locked'): refuse to run evaluation")
    _schema(add, require_commit=True)
    if (add.get("addendum_to") or {}).get("version") != 5 or \
            (add.get("addendum_to") or {}).get("sha256") != (v7.get("addendum_to") or {}).get("sha256"):
        raise RuntimeError("v8 addendum does not reference the live v5 lock (version / sha256 mismatch)")
    if (add.get("v6_addendum") or {}).get("sha256") != (v7.get("v6_addendum") or {}).get("sha256"):
        raise RuntimeError("v8 addendum does not reference the live v6 addendum (sha256 mismatch)")
    if (add.get("v7_addendum") or {}).get("sha256") != v7.get("sha256"):
        raise RuntimeError("v8 addendum does not reference the live v7 addendum (sha256 mismatch)")
    h = prereg.canonical_hash(add)
    if add.get("sha256") != h:
        raise RuntimeError(f"v8 addendum hash mismatch: stored {add.get('sha256')} vs computed {h}")
    if verify_code:
        drift = prereg_v6.v6_code_drift(add, code_root)
        if drift:
            raise RuntimeError(f"v8 code drifted after the addendum lock: {drift[:10]}")
    if verify_inputs:
        drift = prereg_v6.v6_input_drift(add, ws_root)
        if drift:
            raise RuntimeError(f"v8 inputs drifted after the addendum lock: {drift[:10]}")
    if verify_data:
        from ..envs.data_v8 import data_drift
        drift = data_drift(add.get("data_sha256"))
        if drift:
            raise RuntimeError(f"raw data drifted after the addendum lock: {drift}")
    if task_id is not None and task_id not in (add.get("eval_tasks") or {}):
        raise RuntimeError(f"task {task_id!r} is not an eval task of the v8 addendum")
    return add


def lock_history_check(path: Path | str) -> str:
    """As prereg_v7: the locked addendum file must have been committed exactly once and equal that commit's bytes."""
    import subprocess
    p = Path(path).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(p.parent), capture_output=True)
    if top.returncode != 0:
        raise RuntimeError("v8 addendum: not inside a git repository")
    root = Path(top.stdout.decode().strip())
    rel = p.relative_to(root)
    log = subprocess.run(["git", "log", "--format=%H", "--", str(rel)], cwd=str(root), capture_output=True)
    commits = log.stdout.decode().split()
    if len(commits) != 1:
        raise RuntimeError(f"v8 addendum: lock file must be committed exactly once (found {len(commits)} commits)")
    blob = subprocess.run(["git", "show", f"{commits[0]}:{rel}"], cwd=str(root), capture_output=True)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        raise RuntimeError("v8 addendum: lock file differs from its original commit")
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
            raise RuntimeError(f"v8 {name} differ from the draft hashes: {drift}")
    from ..envs.data_v8 import data_drift
    drift = data_drift(draft.get("data_sha256"))
    if drift:
        raise RuntimeError(f"raw data changed since the draft: {drift}")
    bad = prereg_v6.commit_code_mismatch(draft, git_commit, code_root)
    if bad:
        raise RuntimeError(f"commit {git_commit} does not contain the bound code: {bad}")
    v7 = _live_v7(known_drift=draft.get("v7_known_input_drift"))
    if (draft.get("v7_addendum") or {}).get("sha256") != v7["sha256"]:
        raise RuntimeError("draft does not reference the live v7 addendum")
    if (draft.get("v6_addendum") or {}).get("sha256") != v7["v6_addendum"]["sha256"]:
        raise RuntimeError("draft does not reference the live v6 addendum")
    if (draft.get("addendum_to") or {}).get("sha256") != v7["addendum_to"]["sha256"]:
        raise RuntimeError("draft does not reference the live v5 lock")
    lock = {k: v for k, v in draft.items() if k not in ("sha256", "sha256_draft")}
    lock.update({"status": "locked", "git_commit": git_commit, "locked_by": locked_by,
                 "locked_at": now or datetime.now().isoformat(), "eval_touched_at_lock": False,
                 "draft_sha256": draft.get("sha256_draft")})
    lock["sha256"] = prereg.canonical_hash(lock)
    out_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    return lock
