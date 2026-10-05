"""Lock-v9 addendum: eval seals (new file).  Identical mechanism to ``v8_seal`` (v6 make_seal / verify_seal reused
unchanged, v7/v8 single-commit history anchor); only the commit messages carry the v9 tag.  Seals live in
``exp/results/full/v9_seals``."""
from __future__ import annotations

import json
from datetime import datetime

from .v6_seal import _git, make_seal, repo_root, seal_paths
from .v8_seal import TRAILER, seal_exists, verify_seal

__all__ = ["make_seal", "write_and_commit_seal", "verify_seal", "seal_exists"]


def write_and_commit_seal(seal, seals_dir, now=None):
    sp, rp = seal_paths(seals_dir, seal["task_id"])
    if sp.exists() or rp.exists():
        raise RuntimeError(f"{seal['task_id']}: a seal already exists; sealed eval tasks are never re-sealed")
    sp.parent.mkdir(parents=True, exist_ok=True)
    seal = dict(seal, sealed_at=now or datetime.now().isoformat())
    sp.write_text(json.dumps(seal, indent=1, sort_keys=True))
    root = repo_root(sp)
    rel = sp.resolve().relative_to(root)
    _git(["add", "--", str(rel)], root)
    _git(["commit", "-q", "-m", f"seal(v9): {seal['task_id']} results {seal['results_content_sha256'][:12]}",
          "-m", TRAILER, "--only", "--", str(rel)], root)
    sha = _git(["rev-parse", "HEAD"], root).stdout.decode().strip()
    rp.write_text(json.dumps({"task_id": seal["task_id"], "seal_commit": sha, "seal_path": str(rel)}, indent=1))
    rrel = rp.resolve().relative_to(root)
    _git(["add", "--", str(rrel)], root)
    _git(["commit", "-q", "-m", f"seal-ref(v9): {seal['task_id']} -> {sha[:12]}", "-m", TRAILER, "--only", "--",
          str(rrel)], root)
    return sha
