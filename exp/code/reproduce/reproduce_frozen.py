"""Reproduction runner: re-execute the FROZEN eval configurations of the headline lock blocks, outside the lock gates.

WHAT THIS IS
  For chosen eval seeds of a registered eval task it rebuilds the task's replay environment on the evaluation half,
  runs every registered method with the task's own runner code (``run_r5s_v9.py``, ``run_v10.py``, ``run_v11.py``,
  ``run_v12.py``: their ``TASKS``, ``load_frozen`` / ``task_eps_cfg``, ``init_env``, ``make_jobs`` and ``job``,
  imported UNCHANGED, together with every lock-bound library module), and compares each regenerated row field by field
  with the sealed row of the same (method, eps, seed).  Rows are written to a separate output directory; nothing
  under ``exp/results`` is written (checked: the runner snapshots exp/results before and after and reports any change).

WHAT THIS IS NOT
  It does NOT recreate the historical authorisation of the eval runs and is not a lock gate.  The gates
  (``dsswm/stats/prereg_v9..v12.py``) also check git anchors and the original code bytes; they cannot pass in this
  copy.  This runner replaces each gate's ``addendum_gate`` by a REPRODUCTION STAND-IN that only (a) looks the task up
  in the shipped lock file's ``eval_tasks`` and (b) hands that lock to the reader, so the readers' own layer / campaign
  / lock-hash / data-hash checks still run.  The access logs of the lock-v11 / v12 readers are redirected to the output
  directory.  An exact match shows that the shipped code, frozen gate files and data regenerate the sealed rows; it
  says nothing about WHEN those rows were first produced or whether the evaluation halves were read before the locks.
  See ``exp/PROVENANCE_MANIFEST.md``.

COMPARISON
  Fields compared: every field of the row except the timing fields (``dsswm.stats.v6_replica.TIMING_FIELDS``: sec,
  rs_sec_plan, rs_sec_cert, rs_sec_total) and ``code_sha256`` (a hash of the code files' bytes; differs in a scrubbed
  copy).  ``addendum_sha256`` and ``data_sha256`` are compared.  Values are compared after a JSON round trip, exactly
  (no float tolerance).  The sealed file's content hash is also checked against its seal
  (``results_content_sha256``, the hash the seals bind); a seal mismatch (or a missing seal) fails the task
  (status SEAL_MISMATCH) and the overall result.  For lock-v9 tasks the frozen configurations must equal the lock's
  ``frozen_configs``; a mismatch fails the task (no fallback).

USAGE (cwd = exp/code; data under $DATA_DIR as in the README)
  python reproduce/reproduce_frozen.py --list
  python reproduce/reproduce_frozen.py --blocks v10A v11 --n-seeds 5 [--workers 4] [--out ../reproduce_out]
  python reproduce/reproduce_frozen.py --blocks all --n-seeds 3
  python reproduce/reproduce_frozen.py --tasks v9a_full_d --seeds 37000 37001
  python reproduce/reproduce_frozen.py --blocks all --n-seeds 3 --provenance-by-files   # re-made data (README)
Seeds default to --n-seeds seeds evenly spread over the task's 200 registered eval seeds (first and last included).
Only registered eval seeds are accepted.  Exit code 0 iff every compared row matches and every sealed file equals
its seal.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp" / "results"
FULL = RES / "full"

NOTICE = ("REPRODUCTION RUN OUTSIDE THE LOCK GATES. The lock gates are replaced by a stand-in that only looks the task "
          "up in the shipped lock; this run does not recreate, and is no evidence of, the historical authorisation or "
          "chronology of the sealed eval runs.")

# block -> (lock version, eval tasks)
BLOCKS = {
    "v9A": (9, ("v9a_full_d", "v9a_full_k", "v9a_full_x")),
    "v9B": (9, ("v9b_full_d", "v9b_full_k", "v9b_full_x")),
    "v9C": (9, ("v9c_full_b", "v9c_full_a")),
    "v10A": (10, ("v10a_full_s16", "v10a_full_s32")),
    "v10B": (10, ("v10b_full_s16", "v10b_full_s32", "v10b_full_s64")),
    "v10D": (10, ("v10d_full_x5s64",)),
    "v11": (11, ("v11_obd_full",)),
    "v12women": (12, ("v12_women_full",)),
    "v12men": (12, ("v12_men_full",)),
}
TASK_VERSION = {t: v for v, ts in BLOCKS.values() for t in ts}
RUNNER = {9: "run_r5s_v9", 10: "run_v10", 11: "run_v11", 12: "run_v12"}
EXCLUDED_FIELDS = ("code_sha256",)          # plus dsswm.stats.v6_replica.TIMING_FIELDS


def sealed_rows_path(task):
    v = TASK_VERSION[task]
    base = {11: FULL / "v11_obd" / task, 12: FULL / "v12_obd" / task}.get(v, FULL / task)
    for name in ("results.jsonl", "results.jsonl.gz"):
        if (base / name).exists():
            return base / name
    return None


def read_jsonl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def seal_path(task):
    return FULL / f"v{TASK_VERSION[task]}_seals" / f"{task}.seal.json"


# =============================================================================================== gate stand-in
def load_lock(version):
    pr = importlib.import_module(f"dsswm.stats.prereg_v{version}")
    p = Path(pr.ADDENDUM_PATH)
    return json.loads(p.read_text()), p


def install_standins(out_dir):
    """Replace addendum_gate of prereg_v9..v12 by the reproduction stand-in; redirect the v11 / v12 access logs."""
    info = {}
    for v in (9, 10, 11, 12):
        pr = importlib.import_module(f"dsswm.stats.prereg_v{v}")
        lock, p = load_lock(v)

        def standin(task_id, path=None, _lock=lock, _v=v, **kw):
            if task_id not in (_lock.get("eval_tasks") or {}):
                return False, f"reproduction stand-in (v{_v}): {task_id!r} is not a registered eval task"
            return True, _lock

        standin.__doc__ = "REPRODUCTION STAND-IN (not a lock gate): " + NOTICE
        pr.addendum_gate = standin
        info[f"v{v}"] = {"lock_file": str(p.relative_to(WS)), "lock_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "lock_sha256_field": lock.get("sha256")}
    from dsswm.envs import obd_v11_eval, obd_v12_eval
    obd_v11_eval.ACCESS_LOG = out_dir / "v11_reproduction_access_log.jsonl"
    obd_v12_eval.access_log = lambda c: out_dir / f"{c}_reproduction_access_log.jsonl"
    return info


def expand_data_dir_literals():
    """Scrubbed copy only: the lock-bound readers obd_v11_eval.py / obd_v12_eval.py hold the literal string
    "$DATA_DIR/open_bandit" (a scrubbing artefact; the original files name an absolute path).  The files are hash-bound
    and left unchanged; this expands the literal in the in-memory path tables of this process only."""
    from dsswm.envs import obd_v11_eval, obd_v12_eval
    dd, n = os.environ.get("DATA_DIR", "data"), 0
    for tab in (obd_v11_eval.DATA_FILES_V11, obd_v12_eval.DATA_FILES_V12):
        for k, v in list(tab.items()):
            if isinstance(v, str) and v.startswith("$DATA_DIR"):
                tab[k] = dd + v[len("$DATA_DIR"):]
                n += 1
    return n


PROVENANCE_SUBSTITUTED: list = []


def install_provenance_by_files():
    """--provenance-by-files: the Open Bandit PROVENANCE.json records (random/all, women, men) are hash-bound by locks
    v11 / v12 but are not shipped (their free text identifies the authors' tooling), so a re-split cannot reproduce their
    bytes.  With this option a PROVENANCE.json whose bytes differ is accepted IN PLACE OF the bound one only if its
    "files" record lists exactly the bound dev / eval-labels / eval-outcome hashes AND those three files on disk have
    exactly those hashes; the bound hash is then used for the row binding.  Every substitution is reported."""
    from dsswm.envs import obd_v11_eval as E11, obd_v12_eval as E12
    from dsswm.envs.data_v6 import file_sha256 as real
    l11 = load_lock(11)[0]["data_sha256"]
    l12 = load_lock(12)[0]["data_sha256"]
    fk = {"dev": "dev.pkl", "eval_labels": "eval_labels.pkl", "eval_outcome": "eval_outcome.npy"}
    subs = {E11.DATA_FILES_V11["obd_provenance"]: (
        l11["obd_provenance"]["sha256"],
        {f: (l11[f"obd_{k}"]["sha256"], E11.DATA_FILES_V11[f"obd_{k}"]) for k, f in fk.items()})}
    for c in ("women", "men"):
        subs[E12.DATA_FILES_V12[f"{c}_provenance"]] = (
            l12[f"{c}_provenance"]["sha256"],
            {f: (l12[f"{c}_{k}"]["sha256"], E12.DATA_FILES_V12[f"{c}_{k}"]) for k, f in fk.items()})

    def file_sha256_or_bound_provenance(path, *a, **kw):
        got = real(path, *a, **kw)
        ent = subs.get(str(path))
        if ent is None or got == ent[0]:
            return got
        want, files = ent
        try:
            rec = json.loads(Path(path).read_text()).get("files") or {}
        except (OSError, ValueError):
            return got
        if all(rec.get(f) == sha and real(p) == sha for f, (sha, p) in files.items()):
            name = "/".join(Path(path).parts[-3:])
            if name not in PROVENANCE_SUBSTITUTED:
                PROVENANCE_SUBSTITUTED.append(name)
            return want
        return got                                        # no substitution: the lock check then fails loudly

    E11.file_sha256 = file_sha256_or_bound_provenance
    E12.file_sha256 = file_sha256_or_bound_provenance


# =============================================================================================== per-version adapters
class FrozenConfigMismatch(RuntimeError):
    """The frozen configurations on disk differ from the lock (lock v9 adapter)."""


def load_frozen_v9(R, lock):
    """Frozen configs of a lock-v9 task, checked against the lock's ``frozen_configs`` by the runner's own
    ``load_frozen(lock)``.  A mismatch is a hard failure of the task: there is no fallback to the unchecked gate files
    (an earlier version silently reloaded with ``load_frozen(None)`` and only noted it)."""
    try:
        return R.load_frozen(lock)
    except RuntimeError as e:
        raise FrozenConfigMismatch(f"lock v9: load_frozen(lock) refused ({e}); the frozen configurations on disk do "
                                   f"not equal the lock's frozen_configs, so the task fails") from e


def setup_task(task, version, lock):
    """Build the env exactly as the task's runner does (eval half) and return (module, jobs, key_fn, data_sha,
    notes)."""
    R = importlib.import_module(RUNNER[version])
    T = R.TASKS[task]
    if T["half"] != "eval":
        raise SystemExit(f"{task} is not an eval task")
    add_sha = lock["sha256"] if "sha256" in lock else None
    notes = []
    if version == 9:
        frozen = load_frozen_v9(R, lock)
        cfg = frozen.get(R.cfg_key(T["layer"]), {})
        data_sha = R.data_sha_for(T["layer"], lock)
        R.init_env(T, frozen, eval_task_id=task)
        return R, T, (lambda seeds: [j + (None, add_sha, data_sha) for j in R.make_jobs(T, seeds, cfg)]), data_sha, notes
    if version == 10:
        data_sha = R.data_sha_for(T["data"], lock)
        R.init_env(T, lock, eval_task_id=task)            # also checks frozen configs / segmentation against the lock
        return R, T, (lambda seeds: [j + (None, add_sha, data_sha) for j in R.make_jobs(T, seeds)]), data_sha, notes
    eps, cfg = R.task_eps_cfg(task, lock)                  # checks frozen configs / gate files / eps / THRESH vs lock
    data_sha = R.data_sha_for(lock)
    R.init_env(T, eps, cfg, lock, eval_task_id=task)
    return R, T, (lambda seeds: [j + (None, add_sha, data_sha) for j in R.make_jobs(T, eps, seeds)]), data_sha, notes


def row_key(r):
    return (r["method"], json.dumps(r.get("params"), sort_keys=True), round(float(r["eps"]), 12), int(r["seed"]))


def compare_rows(new, old, timing):
    """Field-by-field comparison (JSON round trip, exact).  Returns list of (field, new, old)."""
    a = json.loads(json.dumps(new, default=str))
    b = json.loads(json.dumps(old, default=str))
    skip = set(timing) | set(EXCLUDED_FIELDS)
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k in skip:
            continue
        if k not in a or k not in b or a[k] != b[k]:
            diffs.append((k, a.get(k, "<missing>"), b.get(k, "<missing>")))
    return diffs


def pick_seeds(all_seeds, n, explicit):
    if explicit:
        bad = [s for s in explicit if s not in all_seeds]
        if bad:
            raise SystemExit(f"seeds {bad} are not registered eval seeds of this task")
        return list(explicit)
    n = max(1, min(n, len(all_seeds)))
    if n == 1:
        return [all_seeds[0]]
    idx = sorted({round(i * (len(all_seeds) - 1) / (n - 1)) for i in range(n)})
    return [all_seeds[i] for i in idx]


def snapshot(root, exclude):
    out = {}
    for dp, _, fs in os.walk(root):
        if exclude is not None and Path(dp).resolve().is_relative_to(exclude):
            continue
        for f in fs:
            p = os.path.join(dp, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def run_task(task, seeds_n, explicit, workers, out_dir, timing):
    version = TASK_VERSION[task]
    lock, _ = load_lock(version)
    t0 = time.time()
    R, T, mk, data_sha, notes = setup_task(task, version, lock)
    setup_sec = time.time() - t0
    seeds = pick_seeds(list(T["seeds"]), seeds_n, explicit)
    _, code_sha = R.code_hashes()
    jobs = [j[:-3] + (code_sha,) + j[-2:] for j in mk(seeds)]
    tdir = out_dir / "rows"
    tdir.mkdir(parents=True, exist_ok=True)
    rfile = tdir / f"{task}.jsonl"
    rows, errs = [], []
    t1 = time.time()
    with get_context("fork").Pool(workers) as pool, open(rfile, "w") as f:
        for r in pool.imap_unordered(R.job, jobs, chunksize=1):
            if r.get("error"):
                errs.append({"method": r["method"], "seed": r["seed"], "error": r["error"][-2000:]})
                continue
            r["reproduction_run"] = True
            f.write(json.dumps(r) + "\n")
            rows.append(r)
    run_sec = time.time() - t1
    # sealed reference rows and their seal
    sp = sealed_rows_path(task)
    res = {"task": task, "lock_version": version, "layer": T.get("layer"), "seeds": seeds, "methods": list(T["methods"]),
           "n_jobs": len(jobs), "n_rows": len(rows), "errors": errs, "setup_sec": round(setup_sec, 1),
           "run_sec": round(run_sec, 1), "workers": workers, "data_sha256": data_sha, "notes": notes,
           "rows_file": str(rfile.relative_to(out_dir))}
    if sp is None:
        res.update(status="no_sealed_rows_shipped")
        return res
    sealed = read_jsonl(sp)
    from dsswm.stats.v6_replica import content_hash
    seal = json.loads(seal_path(task).read_text()) if seal_path(task).exists() else {}
    res["sealed_rows_file"] = str(sp.relative_to(WS))
    res["sealed_content_sha256"] = content_hash(sealed)
    res["seal_results_content_sha256"] = seal.get("results_content_sha256")
    res["sealed_file_matches_seal"] = seal_matches(res["sealed_content_sha256"], res["seal_results_content_sha256"])
    by = {row_key(r): r for r in sealed}
    n_match, mism, missing, fields = 0, [], [], set()
    for r in rows:
        k = row_key(r)
        old = by.get(k)
        if old is None:
            missing.append(list(k))
            continue
        r2 = {kk: vv for kk, vv in r.items() if kk != "reproduction_run"}
        d = compare_rows(r2, old, timing)
        fields |= {kk for kk in set(r2) | set(old) if kk not in set(timing) | set(EXCLUDED_FIELDS)}
        if d:
            mism.append({"key": list(k), "fields": [x[0] for x in d],
                         "first": [[x[0], str(x[1])[:200], str(x[2])[:200]] for x in d[:3]]})
        else:
            n_match += 1
    res.update(n_compared=len(rows) - len(missing), n_exact_match=n_match, n_mismatch=len(mism), mismatches=mism[:20],
               not_in_sealed=missing, fields_compared=sorted(fields),
               status=task_status(res["sealed_file_matches_seal"], mism, missing, errs, len(rows), len(jobs)))
    return res


def seal_matches(content_sha, seal_sha):
    """True iff the shipped sealed rows hash to the seal's results_content_sha256 (a missing seal is a failure)."""
    return seal_sha is not None and content_sha == seal_sha


def task_status(seal_ok, mism, missing, errs, n_rows, n_jobs):
    """'match' only if the sealed file equals its seal AND every regenerated row equals its sealed row; a seal
    mismatch is reported as SEAL_MISMATCH (it fails the task and therefore all_match)."""
    if not seal_ok:
        return "SEAL_MISMATCH"
    return "match" if (not mism and not missing and not errs and n_rows == n_jobs) else "MISMATCH"


def overall_match(results, changed):
    """all_match: every task matched (status 'match', which requires sealed file = seal) and nothing under
    exp/results changed during the run."""
    return (bool(results) and all(r.get("status") == "match" and r.get("sealed_file_matches_seal") is True
                                  for r in results) and not changed)


def render_md(rep):
    """REPORT.md from report.json."""
    L = ["# Reproduction runner report", "", f"> {rep['notice']}", "",
         f"Run {rep['started_at'][:19]} to {rep['ended_at'][:19]}; Python {rep['python']}; "
         + ", ".join(f"{k} {v}" for k, v in rep["versions"].items()) + ".", "",
         "Each regenerated row is compared field by field (exact, after a JSON round trip) with the sealed row of the same "
         "(method, eps, seed). Fields excluded: " + ", ".join(f"`{f}`" for f in rep["excluded_fields"]) + ". "
         "`Sealed file = seal` is the content hash of the shipped sealed rows against the seal's "
         "`results_content_sha256`.", "",
         "| Task | Lock | Seeds | Methods | Rows compared | Exact match | Mismatch | Fields compared | Sealed file = seal "
         "| Run (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in rep["tasks"]:
        sd = t["seeds"] or ["-"]
        L.append(f"| {t['task']} | v{t['lock_version']} | {len(t['seeds'])} ({sd[0]}..{sd[-1]}) | {len(t['methods'])} | "
                 f"{t.get('n_compared', 0)} | {t.get('n_exact_match', 0)} | {t.get('n_mismatch', 'n/a')} | "
                 f"{len(t.get('fields_compared', []))} | {t.get('sealed_file_matches_seal')} | {t['run_sec']} |")
    n_rows = sum(t.get("n_compared", 0) for t in rep["tasks"])
    n_ok = sum(t.get("n_exact_match", 0) for t in rep["tasks"])
    L += ["", f"Total: {n_ok} of {n_rows} regenerated rows equal their sealed rows in every compared field; "
          f"files changed under exp/results during the run (outside the runner's output tree): "
          f"{len(rep['exp_results_changed_files'])}. Overall: **{'all match' if rep['all_match'] else 'MISMATCH'}**.", ""]
    if rep.get("provenance_by_files"):
        L += [f"Option --provenance-by-files was set; PROVENANCE.json records accepted by their file hashes in place "
              f"of the bound bytes: {', '.join(rep.get('provenance_substituted') or []) or 'none'}.", ""]
    notes = [f"- {t['task']}: {n}" for t in rep["tasks"] for n in t.get("notes", [])]
    errs = [f"- {t['task']}: {len(t['errors'])} job error(s)" for t in rep["tasks"] if t.get("errors")]
    errs += [f"- {t['task']}: task error {t['error']}" for t in rep["tasks"] if isinstance(t.get("error"), str)]
    mism = [f"- {t['task']}: {m['key']} fields {m['fields']}" for t in rep["tasks"] for m in t.get("mismatches", [])]
    if notes or errs or mism:
        L += ["## Notes, errors and mismatches", ""] + notes + errs + mism + [""]
    L += ["Lock files used by the gate stand-in (the stand-in only looks the task up in `eval_tasks`; the readers' own "
          "layer, lock-hash and data-hash checks still run):", ""]
    for v, s in rep["gate_standins"].items():
        L.append(f"- {v}: `{s['lock_file']}` (file sha256 `{s['lock_file_sha256'][:16]}...`, "
                 f"sha256 field `{str(s['lock_sha256_field'])[:16]}...`)")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--blocks", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(WS / "exp" / "reproduce_out"))
    ap.add_argument("--render", action="store_true", help="only (re)write REPORT.md from <out>/report.json")
    ap.add_argument("--provenance-by-files", action="store_true",
                    help="accept re-made Open Bandit PROVENANCE.json records by their file hashes (see "
                         "install_provenance_by_files); every substitution is reported")
    a = ap.parse_args()
    if a.render:
        o = Path(a.out)
        (o / "REPORT.md").write_text(render_md(json.loads((o / "report.json").read_text())))
        return
    if a.list:
        for b, (v, ts) in BLOCKS.items():
            print(f"{b:9s} lock v{v}: " + ", ".join(f"{t} ({'rows shipped' if sealed_rows_path(t) else 'NO rows'})"
                                                    for t in ts))
        return
    tasks = []
    for b in (list(BLOCKS) if a.blocks == ["all"] else a.blocks):
        tasks += list(BLOCKS[b][1])
    tasks += [t for t in a.tasks if t not in tasks]
    if not tasks:
        raise SystemExit("nothing to do: pass --blocks ... or --tasks ... (or --list)")
    out_dir = Path(a.out).resolve()
    if out_dir.is_relative_to(RES.resolve()) and not out_dir.is_relative_to((RES / "reproduce").resolve()):
        raise SystemExit("the output directory must not be inside exp/results (except exp/results/reproduce/...)")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(NOTICE, flush=True)
    skip = (RES / "reproduce").resolve() if out_dir.is_relative_to(RES.resolve()) else out_dir
    before = snapshot(RES, skip)                      # exp/results/reproduce/* holds runner outputs only
    stand = install_standins(out_dir)
    n_lit = expand_data_dir_literals()
    if a.provenance_by_files:
        install_provenance_by_files()
    from dsswm.stats.v6_replica import TIMING_FIELDS
    started = datetime.now()
    results = []
    for t in tasks:
        print(f"[reproduce] {t} ...", flush=True)
        try:
            r = run_task(t, a.n_seeds, a.seeds, max(1, min(4, a.workers)), out_dir, TIMING_FIELDS)
        except Exception as e:  # noqa: BLE001 - one failing task must not hide the others
            import traceback
            r = {"task": t, "lock_version": TASK_VERSION[t], "status": "ERROR", "seeds": [], "methods": [],
                 "n_jobs": 0, "setup_sec": 0, "run_sec": 0, "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()[-3000:]}
        print(f"[reproduce] {t}: {r['status']} ({r.get('n_exact_match', 0)}/{r['n_jobs']} rows exact; setup "
              f"{r['setup_sec']} s, run {r['run_sec']} s)", flush=True)
        results.append(r)
    after = snapshot(RES, skip)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    import numpy
    import pandas
    import scipy
    import sklearn
    rep = {"notice": NOTICE, "started_at": started.isoformat(), "ended_at": datetime.now().isoformat(),
           "python": platform.python_version(),
           "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
                        "scikit-learn": sklearn.__version__},
           "excluded_fields": sorted(set(TIMING_FIELDS) | set(EXCLUDED_FIELDS)), "gate_standins": stand,
           "exp_results_changed_files": [str(Path(p).relative_to(WS)) for p in changed],
           "data_dir_literals_expanded": n_lit, "provenance_by_files": bool(a.provenance_by_files),
           "provenance_substituted": list(PROVENANCE_SUBSTITUTED),
           "tasks": results,
           "all_match": overall_match(results, changed)}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out_dir / "REPORT.md").write_text(render_md(rep))
    print(json.dumps({"all_match": rep["all_match"], "exp_results_changed": len(changed),
                      "tasks": {r["task"]: r["status"] for r in results}}), flush=True)
    sys.exit(0 if rep["all_match"] else 1)


if __name__ == "__main__":
    main()
