"""Lock-v8 ADDENDUM runner (new file; v5 / v6 / v7 files untouched).

  A  CONFIRMATORY  X5 RetailHero X9 eval half (untouched, blinded hashed split), seeds 35000-35199, eps 0.015 / 0.02 /
     0.03 (decision at 0.02), stop_k = 15 (N80_pen, x12 and N100_pen from one trajectory).
       v8a_full_a: FDC-BF, FDC-MR[front3], FDC, F = {RECT-ck-HG*, RECT-ck-HG-live, HC-WoR*}, G = {PJC-local*,
                   PJC-menu*}, RECT-ck-BF, RECT-ck-BF+box   (all three eps)
       v8a_full_b: plug-ins and literature rivals, descriptive (eps 0.02 only)
  B  DESCRIPTIVE   CR12 eval half (exposed by r5), seeds 35200-35399, eps 0.001, stop 12/15.
  C  DESCRIPTIVE / POST HOC  CR9 eval half, the v7 block-A streams 33000-33199, eps 0.001, stop_k = 15: PJC*, matched
     Bennett rectangles, own-plan rectangles, plug-ins, and a FDC-BF re-run (reproduction check vs the v7 seal).

Usage (cwd = exp/code)
  run_r5s_v8.py --task v8a_pilot | v8a_pilot_b | v8b_pilot | v8c_pilot     # dev runner checks (dev seeds only)
  run_r5s_v8.py --task v8a_full_a | v8a_full_b | v8b_full_a | v8b_full_b | v8c_full   # EVAL (v8 lock required)
  run_r5s_v8.py --task <task> --replica
  run_r5s_v8.py --analyse A|B|C [--dev]
Rows are written ahead to results.jsonl (fsync), resumable (except sealed eval tasks).  CPU only, BLAS threads 1,
<= 4 worker processes per runner process.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp/results"
GATES = RES / "v8_gates"
N_WORKERS = 4
DEV = range(900, 1000)

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from dsswm.stats import v8_analysis as VA  # noqa: E402
from run_r5s_v7 import x_rank  # noqa: E402

X12_RANK = 12
CODE_FILES = ["run_r5s_v8.py", "run_r5s_v7.py",
              "dsswm/baselines/pjc_bf.py", "dsswm/baselines/fdc_bet.py", "dsswm/baselines/fdc_mr.py",
              "dsswm/baselines/plugin_r5.py", "dsswm/baselines/b2_rect_fe.py",
              "dsswm/envs/x5_v8.py", "dsswm/envs/data_v8.py", "dsswm/stats/prereg_v8.py",
              "dsswm/stats/v8_analysis.py", "dsswm/stats/v8_replica.py", "dsswm/stats/v8_seal.py",
              # v6 / v7 modules used unchanged (bound by the v6 / v7 addenda, re-checked through the v7 gate)
              "dsswm/baselines/rect_v6.py", "dsswm/baselines/wor_betting_v6.py",
              "dsswm/streams/frontier_runner_v6.py", "dsswm/envs/data_v6.py", "dsswm/stats/prereg_v6.py",
              "dsswm/stats/prereg_v7.py", "dsswm/stats/v6_analysis.py", "dsswm/stats/v6_replica.py",
              "dsswm/stats/v6_seal.py", "dsswm/stats/v7_seal.py"]
LOCK_ONLY_FILES = ["build_v8_addendum_draft.py", "run_v8_select.py", "run_pjc_v8dev.py", "analyze_pjc_v8dev.py",
                   "ingest_v8_fresh.py", "dsswm/tests/test_pjc_bf.py", "dsswm/tests/test_v8_addendum.py"]


def _rng(a, b):
    return list(range(a, b + 1))


A_EPS = VA.A_EPS
TASKS = {
    # dev runner checks (dev seeds only)
    "v8a_pilot": dict(block="A", layer="X9", half="dev", seeds=_rng(950, 999), eps=A_EPS, stop_k=15,
                      methods=list(VA.A_CORE)),
    "v8a_pilot_b": dict(block="A", layer="X9", half="dev", seeds=_rng(950, 999), eps=(0.02,), stop_k=15,
                        methods=list(VA.A_DESCR)),
    "v8b_pilot": dict(block="B", layer="CR12", half="dev", seeds=_rng(950, 951), eps=(0.001,), stop_k=None,
                      methods=list(VA.B_METHODS)),
    "v8c_pilot": dict(block="C", layer="CR9", half="dev", seeds=_rng(950, 952), eps=(0.001,), stop_k=15,
                      methods=list(VA.C_NEW)),
    # eval tasks (v8 addendum lock required)
    "v8a_full_a": dict(block="A", layer="X9", half="eval", seeds=_rng(35000, 35199), eps=A_EPS, stop_k=15,
                       methods=list(VA.A_CORE)),
    "v8a_full_b": dict(block="A", layer="X9", half="eval", seeds=_rng(35000, 35199), eps=(0.02,), stop_k=15,
                       methods=list(VA.A_DESCR)),
    "v8b_full_a": dict(block="B", layer="CR12", half="eval", seeds=_rng(35200, 35399), eps=(0.001,), stop_k=None,
                       methods=["FDC-BF", "FDC-MR[front3]", "FDC"]),
    "v8b_full_b": dict(block="B", layer="CR12", half="eval", seeds=_rng(35200, 35399), eps=(0.001,), stop_k=None,
                       methods=["RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR", "RECT-ck-BF+box", "PJC-local", "PJC-menu"]),
    "v8c_full": dict(block="C", layer="CR9", half="eval", seeds=_rng(33000, 33199), eps=(0.001,), stop_k=15,
                     methods=list(VA.C_NEW)),
}
BLOCK_TASKS = {"A": ("v8a_full_a", "v8a_full_b"), "B": ("v8b_full_a", "v8b_full_b"), "C": ("v8c_full",)}
DEV_BLOCK_TASKS = {"A": ("v8a_pilot", "v8a_pilot_b"), "B": ("v8b_pilot",), "C": ("v8c_pilot",)}
SEALS = RES / "full" / "v8_seals"
V7_SEALS = RES / "full" / "v7_seals"
_ENV: dict = {}


# =============================================================================================== helpers
def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hashes():
    per = {p: sha_file(CODE / p) for p in CODE_FILES if (CODE / p).exists()}
    return per, hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()


def progress(task, done, total, metric=None):
    (RES / f"{task}_PROGRESS.json").write_text(json.dumps({
        "task_id": task, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(task, status, txt):
    pf = RES / f"{task}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES / f"{task}_DONE").write_text(json.dumps({"task_id": task, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def gate_layer(layer):
    """Layer key in the v8 frozen-config files: X9 (block A) or CR (blocks B and C: CR9-tuned configs)."""
    return "X9" if layer == "X9" else "CR"


def load_configs(layer, lock=None):
    """Frozen configs from exp/results/v8_gates (written by run_v8_select.py from dev data only).  Never a default
    fallback for a tuned method.  With a locked addendum they must equal lock['frozen_configs']."""
    key = gate_layer(layer)
    cfg = json.loads((GATES / f"{key.lower()}_configs.json").read_text())["selected"]
    for m in ("RECT-ck-HG", "HC-WoR", "PJC-local", "PJC-menu"):
        if not cfg.get(m):
            raise RuntimeError(f"no frozen config for tuned rival {m} on {key}")
    if lock is not None:
        if (lock.get("frozen_configs") or {}).get(key) != cfg:
            raise RuntimeError(f"frozen configs on disk differ from the locked v8 addendum ({key})")
    return cfg


def _alloc(kw):
    return None if kw.get("alloc") is None else np.asarray(kw["alloc"], dtype=float)


def make(name, kw):
    kw = dict(kw or {})
    from dsswm.baselines import pjc_bf as pj
    if name == "FDC-BF":
        from dsswm.baselines.fdc_bet import make_variant
        return make_variant("FDC-BF")
    if name == "FDC-MR[front3]":
        from dsswm.baselines.fdc_mr import FDCMR
        return FDCMR(name="FDC-MR[front3]", rho="front3")
    if name == "RECT-ck-HG-live":
        from dsswm.baselines.rect_v6 import RectCkHGLive
        return RectCkHGLive()
    if name in ("RECT-ck-HG", "RECT-ck-HG[ney]"):
        plan = str(kw.get("plan", "0.5"))
        if plan == "0.5" and kw.get("alloc") is None:
            from dsswm.baselines.rect_v6 import RectCkHG
            return RectCkHG()
        if kw.get("alloc") is not None:
            return pj.RectCkHGPlan(alloc=_alloc(kw), name=name)
        return pj.RectCkHGPlan(share=float(plan), name=name)
    if name in ("HC-WoR", "HC-WoR[ney]"):
        plan = str(kw.get("plan", "0.5"))
        sch, c, tf = kw["schedule"], float(kw["c"]), kw.get("target_frac")
        if plan == "0.5" and kw.get("alloc") is None:
            from dsswm.baselines.wor_betting_v6 import HCWoRRect
            return HCWoRRect(sch, c, tf)
        if kw.get("alloc") is not None:
            return pj.HCWoRPlan(sch, c, tf, alloc=_alloc(kw), name=name)
        return pj.HCWoRPlan(sch, c, tf, share=float(plan), name=name)
    if name in ("PJC-local", "PJC-menu"):
        k = dict(kw)
        k.pop("pick", None)
        k["boundaries"] = tuple(k["boundaries"])
        if "menu" in k:
            k["menu"] = tuple(k["menu"])
        return pj.PJCBF(**k, name=name)
    if name in ("RECT-ck-BF", "RECT-ck-BF+box"):
        return pj.RectCkBF(box=name.endswith("+box"))
    return reg.make_method(name, **kw)


def method_kw(name, cfg):
    """Frozen parameters of a method on this layer ({} for parameter-free methods)."""
    if name in ("RECT-ck-HG", "HC-WoR", "PJC-local", "PJC-menu", "RECT-ck-HG[ney]", "HC-WoR[ney]", "B4-bal",
                "Hait-SW", "B2-rect"):
        if name not in cfg:
            raise RuntimeError(f"no frozen config for {name}")
        return dict(cfg[name])
    return {}


def build_env(layer, half, eval_task_id=None):
    if layer == "X9":
        from dsswm.envs.x5_v8 import X5LayerEnv
        return X5LayerEnv(half, eval_task_id=eval_task_id), fr.cr_problems("visit"), "visit"
    from dsswm.envs.pool_replay import PoolReplayEnv
    return PoolReplayEnv(layer, half), fr.cr_problems("visit"), "visit"


def init_env(layer, half, eps_list, eval_task_id=None):
    env, probs, outcome = build_env(layer, half, eval_task_id)
    assert env.half == half
    ctx0 = build_ctx(env, probs, eps_list[0])
    J = true_policy_values(ctx0.pols, env.w, env.true_mu(outcome))
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    ctxs = {e: build_ctx(env, probs, e) for e in eps_list}
    _ENV.clear()
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, ctxs=ctxs)


def schedule_arrival_digest(sch):
    """sha256 (16 hex) of the arrival segment sequence and every pool's record order of a schedule (labels only)."""
    h = hashlib.sha256(np.ascontiguousarray(sch.seg_seq).tobytes())
    for p in sch.pool_perm:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()[:16]


class _CaptureSchedule:
    """external reviewer v8 review item 2: R2b must hash the schedule the stream engine ACTUALLY consumed.  Wraps
    ``frontier_runner.make_schedule`` (the name the frozen runner calls) for the duration of one run and records every
    schedule it returns; v5-v7 files are not modified."""

    def __enter__(self):
        from dsswm.streams import frontier_runner as FR
        self._fr, self._orig, self.seen = FR, FR.make_schedule, []

        def wrapped(*a, **k):
            sch = self._orig(*a, **k)
            self.seen.append(sch)
            return sch

        FR.make_schedule = wrapped
        return self

    def __exit__(self, *exc):
        self._fr.make_schedule = self._orig
        return False

    def digest(self):
        if len(self.seen) != 1:
            raise RuntimeError(f"expected exactly one schedule per run, captured {len(self.seen)}")
        return schedule_arrival_digest(self.seen[0])


def job(a):
    name, kw, seed, eps, stop_k, code_sha, add_sha, data_sha = a
    ctx = _ENV["ctxs"][eps]
    if stop_k is not None:
        ctx = ctx_with_stop(ctx, stop_k)
    try:
        m = make(name, kw)
        t0 = time.perf_counter()
        with _CaptureSchedule() as cap:
            s, rows = run_stream_v6(_ENV["env"], m, seed, ctx.problems, ctx.eps, outcome=_ENV["outcome"], ctx=ctx,
                                    J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        sec = round(time.perf_counter() - t0, 3)
        arr_digest = cap.digest()
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        curve_full = curve + [curve[-1] if curve else 0] * (len(ck) - len(curve))

        def first_at(target):
            for k, c in enumerate(curve):
                if c >= target:
                    return k
            return None

        k12 = first_at(12)
        n80_pen = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        k15 = first_at(15)
        xx = x_rank([r["U"] for r in rows], ck, ctx.eps, X12_RANK)
        out = {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "stop_k": int(ctx.stop_k)}
        out.update({f"rs_{k}": v for k, v in s.items()})
        out.update({"N80_pen": n80_pen, "N80_raw": int(ck[k12]) if k12 is not None else tau,
                    "k80": k12, "completed80": k12 is not None,
                    "n_false_at_k80": int(nfalse[k12]) if k12 is not None else int(s["n_false"]),
                    "x12": float(xx) if xx is not None else float(tau),
                    "N100_pen": (int(ck[k15]) if (k15 is not None and nfalse[k15] == 0) else tau)
                    if stop_k == 15 else None,
                    "k100": k15 if stop_k == 15 else None,
                    "fwer_event": bool(s["fwer_event"]), "n_false": int(s["n_false"]), "cert_k": s["cert_k"],
                    "decided_pi": s["decided_pi"], "billing_ok": bool(s["billing_ok"]),
                    "schedule_digest": s["schedule_digest"], "arrival_digest": arr_digest,
                    "validity": m.validity, "alloc_kind": m.alloc_kind, "n_checkpoints_visited": len(rows),
                    "n_cert_curve": curve_full, "tau_R": tau, "sec": sec, "code_sha256": code_sha,
                    "addendum_sha256": add_sha, "data_sha256": data_sha, "error": None})
        if name in ("PJC-local", "PJC-menu"):
            out.update({"pjc_path": [int(x) for x in m.path], "pjc_beta": float(m.ledger["beta"]),
                        "pjc_ctrl_share_plans": [round(float(np.mean(p[:, 0])), 6) for p in m.plans]})
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def run_jobs(task, out, jobs_all, add_sha=None, data_sha=None):
    rfile = out / "results.jsonl"
    _, code_sha = code_hashes()
    done = set()
    if rfile.exists():
        keep = []
        for l in rfile.read_text().splitlines():
            try:
                x = json.loads(l)
            except ValueError:
                continue
            key = (x["method"], json.dumps(x["params"], sort_keys=True), x["seed"], x["eps"])
            if (x.get("error") is None and x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
                    and x.get("data_sha256") == data_sha and key not in done):
                done.add(key)
                keep.append(l)
        rfile.write_text("".join(k + "\n" for k in keep))
    jobs = [j for j in jobs_all if (j[0], json.dumps(j[1], sort_keys=True), j[2], float(j[3])) not in done]
    total = len(jobs_all)
    n_done = total - len(jobs)
    print(f"[{task}] {total} jobs, {len(jobs)} to run", flush=True)
    errs = []
    if jobs:
        jobs = [j[:5] + (code_sha, add_sha, data_sha) for j in jobs]
        rp = out / "replica_report.json"
        if rp.exists():
            rp.unlink()
        with get_context("fork").Pool(N_WORKERS) as pool, open(rfile, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                if r.get("error"):
                    errs.append(r)
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if n_done % 10 == 0 or n_done == total:
                    progress(task, n_done, total, {"runs_done": n_done, "errors": len(errs)})
    if errs:
        (out / "errors.log").write_text("\n\n".join(f"{e['method']} {e['seed']}\n{e['error']}" for e in errs))
        raise RuntimeError(f"{len(errs)} jobs failed (see {out / 'errors.log'})")
    rows = [x for x in (json.loads(l) for l in rfile.read_text().splitlines())
            if x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
            and x.get("data_sha256") == data_sha]
    return rows, code_sha


def make_jobs(T, seeds, cfg):
    jobs = []
    for m in T["methods"]:
        kw = method_kw(m, cfg)
        for e in T["eps"]:
            for s in seeds:
                jobs.append((m, kw, s, e, T["stop_k"]))
    heavy = ("FDC-MR[front3]", "Peace-fav", "Peace-rect", "HC-WoR", "HC-WoR[ney]", "PJC-local", "PJC-menu", "FDC-BF")
    jobs.sort(key=lambda j: (j[0] not in heavy,))
    return jobs


def read_rows(path, add_sha=None):
    p = Path(path)
    if not p.exists():
        return []
    return [x for x in (json.loads(l) for l in p.read_text().splitlines())
            if x.get("error") is None and (add_sha is None or x.get("addendum_sha256") == add_sha)]


def current_rows(path, add_sha, data_sha):
    _, code_sha = code_hashes()
    return [r for r in read_rows(path, add_sha) if r.get("code_sha256") == code_sha
            and r.get("addendum_sha256") == add_sha and r.get("data_sha256") == data_sha]


def data_sha_for(layer, lock):
    from dsswm.envs.data_v8 import layer_data_sha_v8
    return layer_data_sha_v8(layer, frozen=None if lock is None else lock["data_sha256"])


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v8 import addendum_gate
    ok, info = addendum_gate(task)
    if not ok:
        (out / "summary.json").write_text(json.dumps({"task_id": task, "status": "skipped_by_lock", "reason": info,
                                                      "written_at": t_start.isoformat(), "eval_touched": False},
                                                     indent=1))
        mark_done(task, "skipped_by_lock", info)
        print(f"[{task}] skipped_by_lock: {info}")
        return None
    return info


def _frozen_all(lock):
    if lock is not None:
        return lock["frozen_configs"]
    return {k: json.loads((GATES / f"{k.lower()}_configs.json").read_text())["selected"] for k in ("X9", "CR")}


def replica(task, T, lock, add_sha):
    from dsswm.stats import v8_replica as R
    is_eval = T["half"] == "eval"
    base = RES / ("full" if is_eval else "pilots") / task
    data_sha = data_sha_for(T["layer"], lock)
    main_rows = current_rows(base / "results.jsonl", add_sha, data_sha)
    from dsswm.stats.v6_replica import check_matrix
    m = check_matrix(main_rows, T["methods"], T["eps"], T["seeds"])
    if not m["complete"]:
        raise SystemExit(f"[{task}] replica refused: planned matrix incomplete {m}")
    if is_eval:
        from dsswm.stats.v8_seal import verify_seal
        verify_seal(SEALS, task, main_rows, T["methods"], T["eps"], T["seeds"], code_hashes()[0], data_sha, add_sha)
    rout = base / "replica"
    rout.mkdir(parents=True, exist_ok=True)
    seeds = T["seeds"][:R.R1_N_SEEDS]
    cfg = load_configs(T["layer"], lock)
    init_env(T["layer"], T["half"], tuple(T["eps"]), eval_task_id=task if is_eval else None)
    run_jobs(task + "_replica", rout, make_jobs(T, seeds, cfg), add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    rep_rows = current_rows(rout / "results.jsonl", add_sha, data_sha)
    rep = R.build_report(task, is_eval, main_rows, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha, per_sha,
                         data_sha, frozen_configs=_frozen_all(lock))
    rep["written_at"] = datetime.now().isoformat()
    (base / "replica_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"task": task, "replica": rep["status"]}))


def _v7_block_a_rows():
    """The git-sealed v7 block-A rows (same streams as block C), verified against the v7 seals."""
    from dsswm.stats.v6_replica import check_matrix
    from dsswm.stats.v7_seal import verify_seal
    import run_r5s_v7 as RV7
    out = []
    for t in ("v7a_full_a", "v7a_full_b"):
        T7 = RV7.TASKS[t]
        seal = json.loads((V7_SEALS / f"{t}.seal.json").read_text())
        rows = [r for r in read_rows(RES / "full" / t / "results.jsonl", seal["addendum_sha256"])
                if r.get("code_sha256") is not None and r.get("data_sha256") == seal["data_sha256"]]
        m = check_matrix(rows, T7["methods"], T7["eps"], T7["seeds"])
        if not m["complete"]:
            raise RuntimeError(f"v7 rows for {t} incomplete: {m}")
        verify_seal(V7_SEALS, t, rows, T7["methods"], T7["eps"], T7["seeds"], seal["code_sha256"],
                    seal["data_sha256"], seal["addendum_sha256"])
        out += rows
    return out


REPRO_FIELDS = ("N80_pen", "N100_pen", "x12", "cert_k", "decided_pi", "schedule_digest", "n_cert_curve", "fwer_event",
                "n_false")


def repro_check(v8_rows, v7_rows):
    a = {int(r["seed"]): r for r in v8_rows if r["method"] == "FDC-BF"}
    b = {int(r["seed"]): r for r in v7_rows if r["method"] == "FDC-BF"}
    diff = sorted(s for s in a if s not in b or any(a[s].get(f) != b[s].get(f) for f in REPRO_FIELDS))
    return {"fields": list(REPRO_FIELDS), "n_compared": len(a), "n_identical": len(a) - len(diff),
            "different_seeds": diff[:20], "pass": len(a) == len(b) and not diff}


def analyse(block, dev):
    from dsswm.stats import v8_replica as R
    if dev:
        tasks, add_sha, lock, base = DEV_BLOCK_TASKS[block], None, None, RES / "pilots"
    else:
        from dsswm.stats.prereg_v8 import load_locked_addendum
        lock = load_locked_addendum(None)
        add_sha = lock["sha256"]
        tasks, base = BLOCK_TASKS[block], RES / "full"
    per_sha, _ = code_hashes()
    frozen = _frozen_all(lock)
    rows, reps, sts = [], [], []
    for t in tasks:
        T = TASKS[t]
        data_sha = data_sha_for(T["layer"], lock)
        cur = current_rows(base / t / "results.jsonl", add_sha, data_sha)
        rf = base / t / "replica_report.json"
        rep = json.loads(rf.read_text()) if rf.exists() else None
        rep_rows = current_rows(base / t / "replica" / "results.jsonl", add_sha, data_sha)
        sealed = None
        if not dev:
            from dsswm.stats.v8_seal import verify_seal
            sealed = verify_seal(SEALS, t, cur, T["methods"], T["eps"], T["seeds"], per_sha, data_sha,
                                 add_sha)["seal"]["results_content_sha256"]
        sts.append(R.validate_report(rep, t, not dev, cur, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha,
                                     per_sha, data_sha, sealed_content_sha256=sealed, frozen_configs=frozen))
        rows += cur
        reps.append(rep)
    spec = dict(VA.BLOCK_SPEC[block])
    extra = {}
    if block == "C":
        v8 = [dict(r, method="FDC-BF@v8") if r["method"] == "FDC-BF" else r for r in rows]
        if dev:
            spec.update(methods_by_eps={0.001: tuple("FDC-BF@v8" if m == "FDC-BF" else m for m in VA.C_NEW)},
                        seeds=tuple(TASKS[tasks[0]]["seeds"]))
            ana_rows = v8
        else:
            v7 = _v7_block_a_rows()
            extra["fdc_bf_reproduction_vs_v7_seal"] = repro_check(rows, v7)
            ana_rows = v8 + v7
    else:
        ana_rows = rows
        if dev:
            T0 = TASKS[tasks[0]]
            mbe = {float(e): tuple(m for t in tasks for m in TASKS[t]["methods"] if e in TASKS[t]["eps"])
                   for e in T0["eps"]}
            spec.update(methods_by_eps=mbe, seeds=tuple(T0["seeds"]))
    # block-level arrival identity across the block's tasks (R2b across tasks)
    from dsswm.stats.v8_replica import check_r2b
    mbe = {float(e): tuple(ms) for e, ms in spec["methods_by_eps"].items()}
    cross = {}
    for e, ms in mbe.items():
        rr = [r for r in rows if abs(float(r["eps"]) - e) < 1e-15]
        cross[str(e)] = check_r2b(rr, spec["seeds"], (e,), sorted({r["method"] for r in rr}))
    # external reviewer v8 review item 1 (P0): the cross-task arrival identity is part of the replica status
    cross_ok = all(c["pass"] for c in cross.values())
    replica_status = {"status": "pass" if all(x == "pass" for x in sts) and cross_ok else "fail",
                      "task_statuses": dict(zip(tasks, sts)), "cross_task_R2b_pass": cross_ok}
    res = VA.analyse_block(block, ana_rows, replica_status, spec=spec)
    res.update({"mode": "dev_runner_check" if dev else "eval", "addendum_sha256": add_sha, "tasks": list(tasks),
                "replica_statuses": dict(zip(tasks, sts)), "replica_status_combined": replica_status,
                "arrival_identity_across_tasks": cross,
                "replica_content_sha256": [r["results_content_sha256"] for r in reps],
                "frozen_configs_used": frozen.get("X9" if block == "A" else "CR"),
                "code_sha256": per_sha, "written_at": datetime.now().isoformat(), **extra})
    out = base / f"v8_analysis_{block}{'_dev' if dev else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"block": block, "decision": res["decision"]}, default=str))


def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--replica", action="store_true")
    ap.add_argument("--analyse", choices=("A", "B", "C"))
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 per process; host rule)")
    args = ap.parse_args()
    N_WORKERS = max(1, min(4, args.workers))
    if args.analyse:
        analyse(args.analyse, args.dev)
        return
    task = args.task
    T = TASKS[task]
    is_eval = T["half"] == "eval"
    out = RES / ("full" if is_eval else "pilots") / task
    out.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    lock, add_sha = None, None
    if is_eval:
        lock = gate_or_skip(task, out, t_start)
        if lock is None:
            return
        add_sha = lock["sha256"]
    else:
        assert all(s in DEV for s in T["seeds"]), "dev tasks may only use seeds 900-999"
    if args.replica:
        replica(task, T, lock, add_sha)
        return
    if is_eval:
        from dsswm.stats.v8_seal import seal_exists
        if seal_exists(SEALS, task):
            raise SystemExit(f"[{task}] already sealed: a sealed eval task is never re-run or resumed")
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    eps_list = tuple(T["eps"])
    cfg = load_configs(T["layer"], lock)
    data_sha = data_sha_for(T["layer"], lock)
    init_env(T["layer"], T["half"], eps_list, eval_task_id=task if is_eval else None)
    env = _ENV["env"]
    jobs = make_jobs(T, T["seeds"], cfg)
    rows, code_sha = run_jobs(task, out, jobs, add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    t_end = datetime.now()
    from dsswm.stats.v6_replica import check_r2
    from dsswm.stats.v8_replica import check_r2b, groups_for
    r2 = [check_r2(rows, tuple(x for x in g if x in T["methods"]), T["seeds"], eps_list)
          for g in groups_for(task, _frozen_all(lock)) if len([x for x in g if x in T["methods"]]) >= 2]
    summary = {"task_id": task, "status": "complete", "block": T["block"], "layer": T["layer"], "half": T["half"],
               "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]), "eps": list(eps_list),
               "stop_k": T["stop_k"], "tau_R": int(env.tau_R), "n_rows": len(rows), "expected_rows": len(jobs),
               "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
               "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
               "cpu_sec_sum": round(float(sum(r["sec"] for r in rows)), 1),
               "timing_note": f"{N_WORKERS} worker processes, BLAS=1; concurrent with other v8 tasks (biased up)",
               "code_sha256": per_sha, "code_sha256_combined": code_sha, "addendum_sha256": add_sha,
               "data_sha256": data_sha, "eval_touched": bool(is_eval),
               "configs": {m: method_kw(m, cfg) for m in T["methods"]},
               "R2_design_identity": r2, "R2b_arrival_identity": check_r2b(rows, T["seeds"], eps_list, T["methods"])}
    if is_eval:
        from dsswm.stats.v8_seal import make_seal, write_and_commit_seal
        seal = make_seal(task, rows, T["methods"], eps_list, T["seeds"], per_sha, data_sha, add_sha)
        summary["seal"] = {"results_content_sha256": seal["results_content_sha256"],
                           "seal_commit": write_and_commit_seal(seal, SEALS),
                           "seal_path": str((SEALS / f"{task}.seal.json").relative_to(WS))}
    summary["per_method_cpu_sec"] = {m: round(float(sum(r["sec"] for r in rows if r["method"] == m)), 1)
                                     for m in sorted({r["method"] for r in rows})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pid = RES / f"{task}.pid"
    if pid.exists():
        pid.unlink()
    mark_done(task, "success", f"{len(rows)} rows, wall {summary['wall_min']} min")
    print(json.dumps({k: summary[k] for k in ("task_id", "n_rows", "wall_min", "cpu_sec_sum")}), flush=True)


if __name__ == "__main__":
    main()
