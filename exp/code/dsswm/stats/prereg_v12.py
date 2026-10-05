"""Lock-v12 ADDENDUM gate (NEW file; the v5 lock, the v6-v11 addenda and their modules are untouched).

The v12 addendum registers TWO blocks, each with its own verdict and no pooling (plan/v12_obd2_plan.md):
block A = Open Bandit random/women (primary), block B = random/men; each is confirmatory or descriptive by the frozen,
pre-stated block-status gate.  Append-only: it cannot change any earlier verdict.

A v12 evaluation task (or the analysis) may run only if ALL of the following hold:

1. the v11 addendum still validates (``prereg_v11.load_locked_addendum``; through it v10, v9, v8, v7, v6, v5) and the
   set of drifted v11 INPUTS equals ``v11_known_input_drift`` of the v12 addendum exactly (at drafting time: empty);
2. ``plan/prereg_lock_v12_addendum.json`` exists, has every key of ``V12_REQUIRED_KEYS``, ``status == 'locked'``,
   references the live v5 lock and v6-v11 addenda by sha256, has a 40-hex ``git_commit`` and its own canonical
   sha256 matches;
3. ``code_sha256`` / ``input_sha256`` / ``data_sha256`` / ``frozen_gates`` / ``blocks`` are non-empty;
   ``data_sha256`` binds exactly ``obd_v12_eval.DATA_FILES_V12`` and every file still has its hash (dev.pkl /
   PROVENANCE.json from bytes; the four eval files via their campaign PROVENANCE record -- the gate never opens an
   eval file); code and inputs (incl. every frozen gate file, each also listed in ``frozen_gates``) have not drifted;
   every block has finite ``eps`` and ``thresh`` in (0, 1) and a ``status`` in {confirmatory, descriptive};
4. the lock file was committed exactly once and equals that commit's bytes (git anchor, as v7-v11);
5. the task id is listed in ``eval_tasks``.

The DRAFT (``plan/prereg_lock_v12_addendum_DRAFT.json``, status 'draft') never authorises anything.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from . import prereg, prereg_v6, prereg_v11

ADDENDUM_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v12_addendum.json"
DRAFT_PATH = prereg.WS_ROOT / "plan" / "prereg_lock_v12_addendum_DRAFT.json"
V12_REQUIRED_KEYS = ("version", "status", "addendum_to", "v6_addendum", "v7_addendum", "v8_addendum", "v9_addendum",
                     "v10_addendum", "v11_addendum", "v11_known_input_drift", "statement", "primary_method", "blocks",
                     "seed_manifest", "replica_check", "eval_tasks", "frozen_configs", "frozen_gates",
                     "input_sha256", "code_sha256", "data_sha256", "git_commit")
BLOCK_STATUSES = ("confirmatory", "descriptive")

__all__ = ["ADDENDUM_PATH", "DRAFT_PATH", "V12_REQUIRED_KEYS", "check_addendum", "addendum_gate",
           "finalize_addendum", "load_locked_addendum", "lock_history_check", "v11_input_drift_check"]


def _num01(v, what):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 < v < 1):
        raise RuntimeError(f"v12 addendum: {what} must be a finite number in (0, 1), got {v!r}")


def _schema(add: dict, require_commit: bool) -> None:
    missing = [k for k in V12_REQUIRED_KEYS if k not in add]
    if missing:
        raise RuntimeError(f"v12 addendum schema: missing keys {missing}")
    for k in ("code_sha256", "input_sha256", "frozen_configs", "frozen_gates", "eval_tasks", "blocks"):
        if not add.get(k):
            raise RuntimeError(f"v12 addendum: empty {k}")
    from ..envs.obd_v12_eval import CAMPAIGNS, DATA_FILES_V12, LAYERS
    if set(add["blocks"]) != set(CAMPAIGNS):
        raise RuntimeError(f"v12 addendum: blocks must be exactly {CAMPAIGNS}")
    for c, b in add["blocks"].items():
        _num01(b.get("eps"), f"blocks.{c}.eps")
        _num01(b.get("thresh"), f"blocks.{c}.thresh")
        if b.get("status") not in BLOCK_STATUSES:
            raise RuntimeError(f"v12 addendum: blocks.{c}.status must be one of {BLOCK_STATUSES}")
        if b.get("layer") != LAYERS[c]:
            raise RuntimeError(f"v12 addendum: blocks.{c}.layer must be {LAYERS[c]}")
    for t, e in add["eval_tasks"].items():
        c = e.get("campaign")
        if c not in CAMPAIGNS or e.get("layer") != LAYERS[c]:
            raise RuntimeError(f"v12 addendum: eval task {t} has an invalid campaign / layer")
        if e.get("eps") != add["blocks"][c]["eps"]:
            raise RuntimeError(f"v12 addendum: eval task {t} eps differs from its block")
    if set(add.get("data_sha256") or {}) != set(DATA_FILES_V12):
        raise RuntimeError(f"v12 addendum: data_sha256 must bind exactly {sorted(DATA_FILES_V12)}")
    inp = add.get("input_sha256") or {}
    for rel, h in (add.get("frozen_gates") or {}).items():
        if inp.get(rel) != h:
            raise RuntimeError(f"v12 addendum: frozen gate {rel} is not bound in input_sha256 with the same hash")
    if require_commit and not (isinstance(add.get("git_commit"), str) and prereg._HEX40.match(add["git_commit"])):
        raise RuntimeError("v12 addendum: git_commit is not a 40-hex hash")


def v11_input_drift_check(v11: dict, known: dict, ws_root: Path | str = prereg.WS_ROOT) -> None:
    """The drifted v11 inputs must be exactly the disclosed ones, each at its disclosed current sha256."""
    drift = set(prereg_v6.v6_input_drift(v11, ws_root))
    if drift != set(known or {}):
        raise RuntimeError(f"v11 input drift {sorted(drift)} != disclosed v11_known_input_drift {sorted(known or {})}")
    for rel, d in (known or {}).items():
        if prereg._sha_file(Path(ws_root) / rel) != d.get("current_sha256") or \
                (v11.get("input_sha256") or {}).get(rel) != d.get("locked_sha256"):
            raise RuntimeError(f"v11 input {rel}: hashes differ from the disclosed drift record")


def _live_v11(verify_code=True, verify_data=True, known_drift=None):
    v11 = prereg_v11.load_locked_addendum(None, verify_code=verify_code, verify_inputs=False, verify_data=verify_data)
    if verify_code:
        v11_input_drift_check(v11, known_drift or {})
    return v11


def _refs_ok(add: dict, v11: dict) -> None:
    if (add.get("addendum_to") or {}).get("version") != 5 or \
            (add.get("addendum_to") or {}).get("sha256") != (v11.get("addendum_to") or {}).get("sha256"):
        raise RuntimeError("v12 addendum does not reference the live v5 lock (version / sha256 mismatch)")
    for k in ("v6_addendum", "v7_addendum", "v8_addendum", "v9_addendum", "v10_addendum"):
        if (add.get(k) or {}).get("sha256") != (v11.get(k) or {}).get("sha256"):
            raise RuntimeError(f"v12 addendum does not reference the live {k} (sha256 mismatch)")
    if (add.get("v11_addendum") or {}).get("sha256") != v11.get("sha256"):
        raise RuntimeError("v12 addendum does not reference the live v11 addendum (sha256 mismatch)")


def check_addendum(add: dict, task_id: str | None = None, v11_add: dict | None = None,
                   code_root: Path | str = prereg.CODE_ROOT, ws_root: Path | str = prereg.WS_ROOT,
                   verify_code: bool = True, verify_inputs: bool = True, verify_data: bool = True) -> dict:
    if add.get("status") != "locked":
        raise RuntimeError(f"v12 addendum status={add.get('status')!r} (need 'locked'): refuse to run evaluation")
    _schema(add, require_commit=True)
    if task_id is not None and task_id not in (add.get("eval_tasks") or {}):
        raise RuntimeError(f"task {task_id!r} is not an eval task of the v12 addendum")
    v11 = _live_v11(verify_code, verify_data, add.get("v11_known_input_drift")) if v11_add is None else v11_add
    _refs_ok(add, v11)
    h = prereg.canonical_hash(add)
    if add.get("sha256") != h:
        raise RuntimeError(f"v12 addendum hash mismatch: stored {add.get('sha256')} vs computed {h}")
    if verify_code:
        drift = prereg_v6.v6_code_drift(add, code_root)
        if drift:
            raise RuntimeError(f"v12 code drifted after the addendum lock: {drift[:10]}")
    if verify_inputs:
        drift = prereg_v6.v6_input_drift(add, ws_root)
        if drift:
            raise RuntimeError(f"v12 inputs drifted after the addendum lock: {drift[:10]}")
    if verify_data:
        from ..envs.obd_v12_eval import data_drift_v12
        drift = data_drift_v12(add.get("data_sha256"))      # never opens an eval file
        if drift:
            raise RuntimeError(f"raw data drifted after the addendum lock: {drift}")
    return add


def lock_history_check(path: Path | str) -> str:
    """The locked addendum file must have been committed exactly once and equal that commit (as v7-v11)."""
    import subprocess
    p = Path(path).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(p.parent), capture_output=True)
    if top.returncode != 0:
        raise RuntimeError("v12 addendum: not inside a git repository")
    root = Path(top.stdout.decode().strip())
    rel = p.relative_to(root)
    log = subprocess.run(["git", "log", "--format=%H", "--", str(rel)], cwd=str(root), capture_output=True)
    commits = log.stdout.decode().split()
    if len(commits) != 1:
        raise RuntimeError(f"v12 addendum: lock file must be committed exactly once (found {len(commits)} commits)")
    blob = subprocess.run(["git", "show", f"{commits[0]}:{rel}"], cwd=str(root), capture_output=True)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        raise RuntimeError("v12 addendum: lock file differs from its original commit")
    return commits[0]


def load_locked_addendum(task_id: str | None = None, path: Path | str | None = None, require_history: bool = True,
                         **kw) -> dict:
    p = Path(path) if path is not None else ADDENDUM_PATH
    if p.resolve() == DRAFT_PATH.resolve():
        raise RuntimeError("the v12 DRAFT never authorises anything")
    add = check_addendum(prereg.load_lock(p), task_id, **kw)
    if require_history:
        lock_history_check(p)
    return add


def addendum_gate(task_id: str, path: Path | str | None = None, **kw):
    """(ok, addendum_or_reason) -- non-raising, so a task can write summary.status='skipped_by_lock'."""
    try:
        return True, load_locked_addendum(task_id, path, **kw)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
        return False, str(e)


def finalize_addendum(git_commit: str, draft_path: Path | str = DRAFT_PATH, out_path: Path | str = ADDENDUM_PATH,
                      locked_by: str = "main session", now: str | None = None,
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
            raise RuntimeError(f"v12 {name} differ from the draft hashes: {drift}")
    from ..envs.obd_v12_eval import data_drift_v12
    drift = data_drift_v12(draft.get("data_sha256"))          # no eval file opened
    if drift:
        raise RuntimeError(f"raw data changed since the draft: {drift}")
    bad = prereg_v6.commit_code_mismatch(draft, git_commit, code_root)
    if bad:
        raise RuntimeError(f"commit {git_commit} does not contain the bound code: {bad}")
    v11 = _live_v11(known_drift=draft.get("v11_known_input_drift"))
    _refs_ok(draft, v11)
    lock = {k: v for k, v in draft.items() if k not in ("sha256", "sha256_draft")}
    lock.update({"status": "locked", "git_commit": git_commit, "locked_by": locked_by,
                 "locked_at": now or datetime.now().isoformat(), "eval_touched_at_lock": False,
                 "draft_sha256": draft.get("sha256_draft")})
    lock["sha256"] = prereg.canonical_hash(lock)
    out_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    return lock


_ = json
