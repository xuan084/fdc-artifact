"""Lock-v6 addendum: eval seals (new file).

When an eval task finishes (and again whenever a resume completes it), the runner writes
``exp/results/full/v6_seals/<task>.seal.json`` -- content hash of ALL main result rows (timing fields excluded),
row count, method / eps / seed coverage, code / data / addendum hashes -- and immediately git-commits that file ALONE.
A file cannot contain the hash of the commit that adds it, so the commit sha is recorded in a sidecar
``<task>.seal_ref.json`` (committed right after, alone) and in the task summary.

``verify_seal`` (used by the replica step and by the analysis) refuses unless:
  * the ref names a commit that exists and is an ancestor of (or equal to) HEAD;
  * the seal file's current bytes equal ``git show <commit>:<path>`` (no edit after sealing);
  * the seal's task / methods / eps / seeds / code / data / addendum hashes equal the expected ones and its coverage
    is complete;
  * the content hash of the CURRENT main rows equals the sealed hash (no edit of results after sealing).

Threat model: accidental drift, partial / resumed runs and post-hoc edits of results, reports or seals are detected
against git history; deliberate rewriting of git history is out of scope (as in standard preregistration practice).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .v6_replica import check_matrix, content_hash

__all__ = ["make_seal", "write_and_commit_seal", "verify_seal", "seal_paths", "repo_root"]


def _git(args, cwd, check=True):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r


def repo_root(path):
    return Path(_git(["rev-parse", "--show-toplevel"], Path(path).parent).stdout.decode().strip())


def seal_paths(seals_dir, task):
    d = Path(seals_dir)
    return d / f"{task}.seal.json", d / f"{task}.seal_ref.json"


def make_seal(task, rows, methods, eps, seeds, code_sha256, data_sha256, add_sha):
    m = check_matrix(rows, methods, eps, seeds)
    if not m["complete"]:
        raise ValueError(f"{task}: refuse to seal an incomplete matrix: {m}")
    return {"task_id": task, "results_content_sha256": content_hash(rows), "n_rows": len(rows),
            "methods": list(methods), "eps": [float(e) for e in eps],
            "seeds": [int(seeds[0]), int(seeds[-1]), len(seeds)], "coverage_complete": True,
            "code_sha256": code_sha256, "data_sha256": data_sha256, "addendum_sha256": add_sha}


def write_and_commit_seal(seal, seals_dir, now=None):
    """Write the seal, commit it alone, then write + commit the sidecar ref with the seal commit sha."""
    from datetime import datetime
    sp, rp = seal_paths(seals_dir, seal["task_id"])
    sp.parent.mkdir(parents=True, exist_ok=True)
    seal = dict(seal, sealed_at=now or datetime.now().isoformat())
    sp.write_text(json.dumps(seal, indent=1, sort_keys=True))
    root = repo_root(sp)
    rel = sp.resolve().relative_to(root)
    _git(["add", "--", str(rel)], root)
    _git(["commit", "-q", "-m", f"seal(v6): {seal['task_id']} results {seal['results_content_sha256'][:12]}",
          "--only", "--", str(rel)], root)
    sha = _git(["rev-parse", "HEAD"], root).stdout.decode().strip()
    rp.write_text(json.dumps({"task_id": seal["task_id"], "seal_commit": sha, "seal_path": str(rel)}, indent=1))
    rrel = rp.resolve().relative_to(root)
    _git(["add", "--", str(rrel)], root)
    _git(["commit", "-q", "-m", f"seal-ref(v6): {seal['task_id']} -> {sha[:12]}", "--only", "--", str(rrel)], root)
    return sha


def verify_seal(seals_dir, task, current_rows, methods, eps, seeds, code_sha256, data_sha256, add_sha):
    sp, rp = seal_paths(seals_dir, task)
    if not sp.exists() or not rp.exists():
        raise ValueError(f"{task}: no eval seal")
    ref = json.loads(rp.read_text())
    sha = ref.get("seal_commit") or ""
    root = repo_root(sp)
    rel = sp.resolve().relative_to(root)
    if _git(["cat-file", "-e", f"{sha}^{{commit}}"], root, check=False).returncode != 0:
        raise ValueError(f"{task}: seal commit {sha!r} does not exist")
    if _git(["merge-base", "--is-ancestor", sha, "HEAD"], root, check=False).returncode != 0:
        raise ValueError(f"{task}: seal commit {sha[:12]} is not in HEAD history")
    committed = _git(["show", f"{sha}:{rel}"], root, check=False)
    if committed.returncode != 0 or committed.stdout != sp.read_bytes():
        raise ValueError(f"{task}: seal file differs from its content at commit {sha[:12]} (edited after sealing)")
    seal = json.loads(sp.read_text())
    errs = []
    want = {"task_id": task, "methods": list(methods), "eps": [float(e) for e in eps],
            "seeds": [int(seeds[0]), int(seeds[-1]), len(seeds)], "coverage_complete": True,
            "code_sha256": code_sha256, "data_sha256": data_sha256, "addendum_sha256": add_sha}
    errs += [k for k, v in want.items() if seal.get(k) != v]
    if not check_matrix(current_rows, methods, eps, seeds)["complete"]:
        errs.append("current main rows incomplete")
    if content_hash(current_rows) != seal.get("results_content_sha256"):
        errs.append("current main rows differ from the sealed content hash")
    if errs:
        raise ValueError(f"{task}: seal check failed: {errs}")
    return {"seal": seal, "seal_commit": sha}
