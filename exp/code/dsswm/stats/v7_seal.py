"""Lock-v7 addendum: eval seals (new file).  Same mechanism as ``v6_seal`` (whose ``make_seal`` / ``verify_seal`` are
reused unchanged); only the commit messages carry the v7 tag.  Seals live in ``exp/results/full/v7_seals``."""
from __future__ import annotations

import json
from datetime import datetime

from .v6_seal import _git, make_seal, repo_root, seal_paths
from .v6_seal import verify_seal as _verify_seal_v6

__all__ = ["make_seal", "write_and_commit_seal", "verify_seal", "seal_exists"]
TRAILER = "Co-Authored-By: Assistant <noreply@example.org>"


def _commits_touching(root, rel):
    out = _git(["log", "--format=%H", "--", str(rel)], root).stdout.decode().split()
    return out


def verify_seal(seals_dir, task, current_rows, methods, eps, seeds, code_sha256, data_sha256, add_sha):
    """v6 checks + history anchor (external reviewer v7 review item 1): the seal file and its ref sidecar must each have been
    committed EXACTLY ONCE (never modified or re-added), and the ref must name that single seal commit.  A reseal of
    edited rows therefore cannot pass without rewriting git history."""
    res = _verify_seal_v6(seals_dir, task, current_rows, methods, eps, seeds, code_sha256, data_sha256, add_sha)
    sp, rp = seal_paths(seals_dir, task)
    root = repo_root(sp)
    cs = _commits_touching(root, sp.resolve().relative_to(root))
    cr = _commits_touching(root, rp.resolve().relative_to(root))
    if len(cs) != 1 or cs[0] != res["seal_commit"]:
        raise ValueError(f"{task}: seal file history is not a single original commit ({len(cs)} commits)")
    if len(cr) != 1:
        raise ValueError(f"{task}: seal ref history is not a single original commit ({len(cr)} commits)")
    committed_ref = _git(["show", f"{cr[0]}:{rp.resolve().relative_to(root)}"], root).stdout
    if committed_ref != rp.read_bytes():
        raise ValueError(f"{task}: seal ref edited after its commit")
    return res


def seal_exists(seals_dir, task):
    sp, rp = seal_paths(seals_dir, task)
    return sp.exists() or rp.exists()


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
    _git(["commit", "-q", "-m", f"seal(v7): {seal['task_id']} results {seal['results_content_sha256'][:12]}",
          "-m", TRAILER, "--only", "--", str(rel)], root)
    sha = _git(["rev-parse", "HEAD"], root).stdout.decode().strip()
    rp.write_text(json.dumps({"task_id": seal["task_id"], "seal_commit": sha, "seal_path": str(rel)}, indent=1))
    rrel = rp.resolve().relative_to(root)
    _git(["add", "--", str(rrel)], root)
    _git(["commit", "-q", "-m", f"seal-ref(v7): {seal['task_id']} -> {sha[:12]}", "-m", TRAILER, "--only", "--", str(rrel)], root)
    return sha
