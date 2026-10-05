"""Lock-v10 ADDENDUM gate (new file; the v5 lock, the v6 / v7 / v8 / v9 addenda and their modules are untouched).

The v10 addendum registers two confirmatory blocks for FDC-DP over exponentially large segment-policy classes
(A: X5 RetailHero eval half, uplift-score segmentations; B: Lenta LR9 eval half, uplift-score segmentations), each an
IUT over its registered cells, reported separately (no joint claim unless both pass), plus descriptive cells.  It is
append-only and cannot change any earlier verdict.

A v10 evaluation task (or the analysis) may run only if ALL of the following hold:

1. the v9 addendum still validates (``prereg_v9.load_locked_addendum``: through it v8, v7, v6, v5) and the set of
   drifted v9 INPUTS equals ``v9_known_input_drift`` of the v10 addendum exactly (at drafting time: empty);
2. ``plan/prereg_lock_v10_addendum.json`` exists, has every key of ``V10_REQUIRED_KEYS``, ``status == 'locked'``,
   references the live v5 lock and v6 / v7 / v8 / v9 addenda by sha256, has a 40-hex ``git_commit`` and its own
   canonical sha256 matches;
3. ``code_sha256`` / ``input_sha256`` / ``data_sha256`` are non-empty, ``data_sha256`` binds exactly
   ``seg_v10_eval.DATA_FILES_V10`` and every file still has its hash; code and inputs (incl. the frozen score models,
   cut points and problems in exp/results/v10_gates) have not drifted;
4. the lock file was committed exactly once and equals that commit's bytes (git anchor, as v7-v9);
5. the task id is listed in ``eval_tasks``.

The DRAFT (``plan/prereg_lock_v10_addendum_DRAFT.json``, status 'draft') never authorises anything.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import prereg, prereg_v6, prereg_v9

ADDENDUM_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v10_addendum.json"
DRAFT_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v10_addendum_DRAFT.json"
V10_REQUIRED_KEYS = ("version", "status", "addendum_to", "v6_addendum", "v7_addendum", "v8_addendum", "v9_addendum",
                     "v9_known_input_drift", "statement", "primary_method", "blocks", "seed_manifest",
                     "replica_check", "eval_tasks", "frozen_configs", "frozen_segmentation", "input_sha256",
                     "code_sha256", "data_sha256", "git_commit")

__all__ = ["ADDENDUM_PATH", "DRAFT_PATH", "V10_REQUIRED_KEYS", "check_addendum", "addendum_gate",
           "finalize_addendum", "load_locked_addendum", "lock_history_check", "v9_input_drift_check"]


def _schema(add: dict, require_commit: bool) -> None:
    missing = [k for k in V10_REQUIRED_KEYS if k not in add]
    if missing:
        raise RuntimeError(f"v10 addendum schema: missing keys {missing}")
    for k in ("code_sha256", "input_sha256", "frozen_configs", "frozen_segmentation", "eval_tasks"):
        if not add.get(k):
            raise RuntimeError(f"v10 addendum: empty {k}")
    from ..envs.seg_v10_eval import DATA_FILES_V10
    if set(add.get("data_sha256") or {}) != set(DATA_FILES_V10):
        raise RuntimeError(f"v10 addendum: data_sha256 must bind exactly {sorted(DATA_FILES_V10)}")
    if require_commit and not (isinstance(add.get("git_commit"), str) and prereg._HEX40.match(add["git_commit"])):
        raise RuntimeError("v10 addendum: git_commit is not a 40-hex hash")


def v9_input_drift_check(v9: dict, known: dict, ws_root: Path | str = prereg.WS_ROOT) -> None:
    """The drifted v9 inputs must be exactly the disclosed ones, each at its disclosed current sha256."""
    drift = set(prereg_v6.v6_input_drift(v9, ws_root))
    if drift != set(known or {}):
        raise RuntimeError(f"v9 input drift {sorted(drift)} != disclosed v9_known_input_drift {sorted(known or {})}")
    for rel, d in (known or {}).items():
        if prereg._sha_file(Path(ws_root) / rel) != d.get("current_sha256") or \
                (v9.get("input_sha256") or {}).get(rel) != d.get("locked_sha256"):
            raise RuntimeError(f"v9 input {rel}: hashes differ from the disclosed drift record")


def _live_v9(verify_code=True, verify_data=True, known_drift=None):
    v9 = prereg_v9.load_locked_addendum(None, verify_code=verify_code, verify_inputs=False, verify_data=verify_data)
    if verify_code:
        v9_input_drift_check(v9, known_drift or {})
    return v9


def _refs_ok(add: dict, v9: dict) -> None:
    if (add.get("addendum_to") or {}).get("version") != 5 or \
            (add.get("addendum_to") or {}).get("sha256") != (v9.get("addendum_to") or {}).get("sha256"):
        raise RuntimeError("v10 addendum does not reference the live v5 lock (version / sha256 mismatch)")
    for k in ("v6_addendum", "v7_addendum", "v8_addendum"):
        if (add.get(k) or {}).get("sha256") != (v9.get(k) or {}).get("sha256"):
            raise RuntimeError(f"v10 addendum does not reference the live {k} (sha256 mismatch)")
    if (add.get("v9_addendum") or {}).get("sha256") != v9.get("sha256"):
        raise RuntimeError("v10 addendum does not reference the live v9 addendum (sha256 mismatch)")


def check_addendum(add: dict, task_id: str | None = None, v9_add: dict | None = None,
                   code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT,
                   verify_code: bool = True, verify_inputs: bool = True, verify_data: bool = True) -> dict:
    if add.get("status") != "locked":
        raise RuntimeError(f"v10 addendum status={add.get('status')!r} (need 'locked'): refuse to run evaluation")
    v9 = _live_v9(verify_code, verify_data, add.get("v9_known_input_drift")) if v9_add is None else v9_add
    _schema(add, require_commit=True)
    _refs_ok(add, v9)
    h = prereg.canonical_hash(add)
    if add.get("sha256") != h:
        raise RuntimeError(f"v10 addendum hash mismatch: stored {add.get('sha256')} vs computed {h}")
    if verify_code:
        drift = prereg_v6.v6_code_drift(add, code_root)
        if drift:
            raise RuntimeError(f"v10 code drifted after the addendum lock: {drift[:10]}")
    if verify_inputs:
        drift = prereg_v6.v6_input_drift(add, ws_root)
        if drift:
            raise RuntimeError(f"v10 inputs drifted after the addendum lock: {drift[:10]}")
    if verify_data:
        from ..envs.seg_v10_eval import data_drift_v10
        drift = data_drift_v10(add.get("data_sha256"))
        if drift:
            raise RuntimeError(f"raw data drifted after the addendum lock: {drift}")
    if task_id is not None and task_id not in (add.get("eval_tasks") or {}):
        raise RuntimeError(f"task {task_id!r} is not an eval task of the v10 addendum")
    return add


def lock_history_check(path: Path | str) -> str:
    """The locked addendum file must have been committed exactly once and equal that commit (as v7-v9)."""
    import subprocess
    p = Path(path).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(p.parent), capture_output=True)
    if top.returncode != 0:
        raise RuntimeError("v10 addendum: not inside a git repository")
    root = Path(top.stdout.decode().strip())
    rel = p.relative_to(root)
    log = subprocess.run(["git", "log", "--format=%H", "--", str(rel)], cwd=str(root), capture_output=True)
    commits = log.stdout.decode().split()
    if len(commits) != 1:
        raise RuntimeError(f"v10 addendum: lock file must be committed exactly once (found {len(commits)} commits)")
    blob = subprocess.run(["git", "show", f"{commits[0]}:{rel}"], cwd=str(root), capture_output=True)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        raise RuntimeError("v10 addendum: lock file differs from its original commit")
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
            raise RuntimeError(f"v10 {name} differ from the draft hashes: {drift}")
    from ..envs.seg_v10_eval import data_drift_v10
    drift = data_drift_v10(draft.get("data_sha256"))
    if drift:
        raise RuntimeError(f"raw data changed since the draft: {drift}")
    bad = prereg_v6.commit_code_mismatch(draft, git_commit, code_root)
    if bad:
        raise RuntimeError(f"commit {git_commit} does not contain the bound code: {bad}")
    v9 = _live_v9(known_drift=draft.get("v9_known_input_drift"))
    _refs_ok(draft, v9)
    lock = {k: v for k, v in draft.items() if k not in ("sha256", "sha256_draft")}
    lock.update({"status": "locked", "git_commit": git_commit, "locked_by": locked_by,
                 "locked_at": now or datetime.now().isoformat(), "eval_touched_at_lock": False,
                 "draft_sha256": draft.get("sha256_draft")})
    lock["sha256"] = prereg.canonical_hash(lock)
    out_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    return lock


_ = json
