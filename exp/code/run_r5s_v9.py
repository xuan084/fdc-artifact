"""Lock-v9 ADDENDUM runner (new file; v5 / v6 / v7 / v8 files untouched and imported unchanged).

  A  CONFIRMATORY  Criteo CR9 eval half (4th use of this table; fresh seeds 37000-37199), eps 0.001, stop_k = 15.
       v9a_full_d  dense monitoring grid (77 times = frozen K = 20 grid + 3 geometric interpolants per interval), every
                   method at the same times: TU-FDC (primary), HC-WoR*, RECT-ck-BF-TU (primary rivals), PJC-local*
                   (own K = 77 ledger), FDC-BF[K77], RECT-ck-HG* (K = 77 ledger), TU-LOC   (descriptive)
       v9a_full_k  frozen K = 20 grid: FDC-BF, HC-WoR*, RECT-ck-HG*, RECT-ck-BF, RECT-ck-BF+box, PJC-local*,
                   FDC-MR[front3]   (descriptive)
       v9a_full_x  frozen K = 20 grid, heavy descriptive: FDC-HG, FDC-LOC
  B  CONFIRMATORY  X5 RetailHero X9 eval half (2nd use; fresh seeds 37200-37399), eps 0.02: v9b_full_d / _k / _x as A,
     X5-tuned v8 configs; PJC-local* non-inferiority on the dense grid.
  C  DESCRIPTIVE   Hillstrom 3-arm (outcome-exposed): v9c_full_b design B (CZ6 / F2 / visit, eps 0.01 / 0.0125 /
     0.015, seeds 37400-37599); v9c_full_a design A supplement (CR6 / F3 / visit, eps 0.006 / 0.0075 / 0.01, seeds
     37600-37799).  K = 20 grid, trajectory to 15/15, N80 at 12/15.

Usage (cwd = exp/code)
  run_r5s_v9.py --freeze-hv9                                    # write exp/results/v9_gates/hv9_configs.json (dev only)
  run_r5s_v9.py --task v9a_pilot_d | ... | v9c_pilot_a         # dev runner checks (dev halves, dev seeds 950-999)
  run_r5s_v9.py --task v9a_full_d | ... | v9c_full_a           # EVAL (v9 lock required)
  run_r5s_v9.py --task <task> --replica
  run_r5s_v9.py --analyse A|B|C [--dev]
  run_r5s_v9.py --toy-dense                                    # dense-monitoring toy FWER for RECT-ck-BF-TU (dev)
Rows are written ahead to results.jsonl (fsync), resumable (except sealed eval tasks).  CPU only, BLAS threads 1,
<= 4 worker processes per runner process.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
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
GATES8 = RES / "v8_gates"
GATES9 = RES / "v9_gates"
N_WORKERS = 4
DEV = range(900, 1000)
X12_RANK = 12
DENSE_SUB = 4

import run_r5s_v8 as R8  # noqa: E402
from dsswm.baselines.fdc_loc import dense_grid  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from dsswm.stats import v9_analysis as VA  # noqa: E402
from run_r5s_v7 import x_rank  # noqa: E402

CODE_FILES = ["run_r5s_v9.py", "run_r5s_v8.py", "run_r5s_v7.py", "run_hillstrom_v9_dev.py",
              "dsswm/baselines/rect_tu_v9.py", "dsswm/baselines/fdc_loc.py", "dsswm/baselines/fdc_hg.py",
              "dsswm/baselines/fdc_bet.py", "dsswm/baselines/fdc_mr.py", "dsswm/baselines/pjc_bf.py",
              "dsswm/baselines/multiarm_v9.py", "dsswm/baselines/rect_v6.py", "dsswm/baselines/wor_betting_v6.py",
              "dsswm/envs/x5_v9.py", "dsswm/envs/x5_v8.py", "dsswm/envs/hillstrom_v9.py", "dsswm/envs/data_v9.py",
              "dsswm/envs/data_v8.py", "dsswm/envs/data_v6.py", "dsswm/envs/pool_replay.py",
              "dsswm/streams/frontier_runner_v6.py", "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py",
              "dsswm/stats/prereg_v9.py", "dsswm/stats/v9_analysis.py", "dsswm/stats/v9_replica.py",
              "dsswm/stats/v9_seal.py", "dsswm/stats/prereg_v8.py", "dsswm/stats/v8_analysis.py",
              "dsswm/stats/v8_replica.py", "dsswm/stats/v8_seal.py", "dsswm/stats/prereg_v7.py",
              "dsswm/stats/prereg_v6.py", "dsswm/stats/v6_analysis.py", "dsswm/stats/v6_replica.py",
              "dsswm/stats/v6_seal.py", "dsswm/stats/v7_seal.py"]
LOCK_ONLY_FILES = ["build_v9_addendum_draft.py", "dsswm/tests/test_v9_addendum.py", "dsswm/tests/test_v9_candidates.py",
                   "run_v9_candidates_dev.py", "analyze_v9_candidates_dev.py", "dsswm/tests/test_multiarm_v9.py"]


def _rng(a, b):
    return list(range(a, b + 1))


def _conf_task(block, kind, half, seeds):
    sp = VA.CONF_SPEC[block]
    grid, methods = {"d": ("D", VA.DENSE_METHODS), "k": ("K", VA.K20_METHODS), "x": ("K", VA.HEAVY_K20)}[kind]
    return dict(block=block, layer=sp["layer"], half=half, seeds=seeds, eps=(sp["eps"],), stop_k=15, grid=grid,
                methods=list(methods))


def _c_task(full, half, seeds):
    sp = VA.C_SPEC[full]
    return dict(block="C", layer=sp["layer"], half=half, seeds=seeds, eps=tuple(sp["eps"]), stop_k=15, grid="K",
                methods=list(sp["methods"]))


DEV_SEEDS = _rng(950, 999)
TASKS = {}
for _b, _lo in (("a", 37000), ("b", 37200)):
    for _k in ("d", "k", "x"):
        TASKS[f"v9{_b}_pilot_{_k}"] = _conf_task(_b.upper(), _k, "dev", DEV_SEEDS)
        TASKS[f"v9{_b}_full_{_k}"] = _conf_task(_b.upper(), _k, "eval", _rng(_lo, _lo + 199))
for _d, _lo in (("b", 37400), ("a", 37600)):
    TASKS[f"v9c_pilot_{_d}"] = _c_task(f"v9c_full_{_d}", "dev", DEV_SEEDS)
    TASKS[f"v9c_full_{_d}"] = _c_task(f"v9c_full_{_d}", "eval", _rng(_lo, _lo + 199))
BLOCK_TASKS = {"A": ("v9a_full_d", "v9a_full_k", "v9a_full_x"), "B": ("v9b_full_d", "v9b_full_k", "v9b_full_x"),
               "C": ("v9c_full_b", "v9c_full_a")}
DEV_BLOCK_TASKS = {"A": ("v9a_pilot_d", "v9a_pilot_k", "v9a_pilot_x"),
                   "B": ("v9b_pilot_d", "v9b_pilot_k", "v9b_pilot_x"), "C": ("v9c_pilot_b", "v9c_pilot_a")}
SEALS = RES / "full" / "v9_seals"
_ENV: dict = {}
HV9_DESIGN = {"HCZ6": ("selection_b.json", "design_b.json", "v9c_full_b"),
              "HCR6": ("selection.json", "design.json", "v9c_full_a")}


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


def cfg_key(layer):
    return {"CR9": "CR", "X9": "X9"}.get(layer, layer)


def load_frozen(lock=None):
    """Frozen configs: CR / X9 from exp/results/v8_gates (v8 dev tuning, unchanged), HCZ6 / HCR6 from
    exp/results/v9_gates/hv9_configs.json (written by --freeze-hv9 from the Hillstrom dev selection files).  With a
    locked addendum they must equal lock['frozen_configs']."""
    out = {"CR": json.loads((GATES8 / "cr_configs.json").read_text())["selected"],
           "X9": json.loads((GATES8 / "x9_configs.json").read_text())["selected"]}
    hv = GATES9 / "hv9_configs.json"
    if hv.exists():
        out.update(json.loads(hv.read_text())["layers"])
    for k in ("CR", "X9"):
        for m in ("RECT-ck-HG", "HC-WoR", "PJC-local"):
            if not out[k].get(m):
                raise RuntimeError(f"no frozen config for tuned rival {m} on {k}")
    if lock is not None and lock.get("frozen_configs") != out:
        raise RuntimeError("frozen configs on disk differ from the locked v9 addendum")
    return out


def freeze_hv9():
    """Block C configs from the dev selection files + dev-half cell variances (for the descriptive Neyman plan)."""
    from dsswm.envs import hillstrom_v9 as hv
    base = RES / "pilots/hillstrom_v9"
    layers = {}
    for key, (selname, desname, full) in HV9_DESIGN.items():
        sel = json.loads((base / selname).read_text())
        des = json.loads((base / desname).read_text())["selected"]
        env = hv.HillstromV9Env(des["segmentation"], "dev", outcomes=(des["outcome"],))
        mu = env.true_mu(des["outcome"])
        picks = sel["picks"]
        spec = VA.C_SPEC[full]
        for fam in ("RECT-ck-HG", "HC-WoR", "PJC-local"):
            if picks[fam] not in spec["methods"]:
                raise SystemExit(f"{key}: dev pick {picks[fam]} not in the registered method list")
        if tuple(sorted(spec["eps"])) != tuple(sorted(des["eps_report"])) or spec["primary_eps"] != des["eps_dev"]:
            raise SystemExit(f"{key}: registered eps differ from the dev design file")
        layers[key] = {"segmentation": des["segmentation"], "family": des["family"], "kappa": des["kappa"],
                       "outcome": des["outcome"], "eps_dev": des["eps_dev"], "eps_report": des["eps_report"],
                       "picks": picks, "selection_file": f"exp/results/pilots/hillstrom_v9/{selname}",
                       "design_file": f"exp/results/pilots/hillstrom_v9/{desname}",
                       "sigma2_dev": np.round(mu * (1 - mu), 12).tolist(),
                       "sigma2_note": "dev-half cell variances mu(1-mu) (dev outcomes only); used only by the "
                                      "descriptive FDC-BF[ney] plan"}
    GATES9.mkdir(parents=True, exist_ok=True)
    out = {"written_at": datetime.now().isoformat(), "source": "run_r5s_v9.py --freeze-hv9 (dev files only)",
           "layers": layers}
    (GATES9 / "hv9_configs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["picks"] for k, v in layers.items()}))


# =============================================================================================== methods
def make(name, kw, layer):
    if layer in ("HCZ6", "HCR6"):
        import run_hillstrom_v9_dev as HD
        env = _ENV["env"]
        return HD.make(name, env.S, env.A, np.asarray(_ENV["sigma2_dev"], dtype=float))
    ck = _ENV["block_points"]
    if name == "TU-FDC":
        from dsswm.baselines.fdc_loc import FDCTimeUniform
        return FDCTimeUniform(ck, name="TU-FDC", localise=False)
    if name == "TU-LOC":
        from dsswm.baselines.fdc_loc import FDCTimeUniform
        return FDCTimeUniform(ck, name="TU-LOC", localise=True)
    if name == "FDC-BF[K77]":
        from dsswm.baselines.fdc_bet import make_variant
        m = make_variant("FDC-BF")
        m.name = name
        return m
    if name == "RECT-ck-BF-TU":
        from dsswm.baselines.rect_tu_v9 import RectCkBFTU
        return RectCkBFTU(ck, box=True)
    if name == "TU-PJC":
        from dsswm.baselines.rect_tu_v9 import make_tu_pjc
        return make_tu_pjc(ck, **kw)
    if name == "FDC-HG":
        from dsswm.baselines.fdc_hg import FDCHG
        return FDCHG()
    if name == "FDC-LOC":
        from dsswm.baselines.fdc_loc import FDCLoc
        return FDCLoc(name="FDC-LOC")
    return R8.make(name, kw)


def method_kw(name, cfg, layer):
    if layer in ("HCZ6", "HCR6"):
        return {}
    if name == "TU-PJC":
        return dict(cfg["PJC-local"])
    return R8.method_kw(name, cfg)


def build_env(layer, half, eval_task_id=None):
    if layer == "X9":
        from dsswm.envs.x5_v9 import X5V9Env
        return X5V9Env(half, eval_task_id=eval_task_id), fr.cr_problems("visit"), "visit"
    if layer in ("HCZ6", "HCR6"):
        from dsswm.envs import hillstrom_v9 as hv
        hcfg = _ENV["frozen"][layer]
        env = hv.HillstromV9Env(hcfg["segmentation"], half, outcomes=(hcfg["outcome"],), eval_task_id=eval_task_id)
        return env, hv.hv9_problems(hcfg["outcome"], hcfg["family"]), hcfg["outcome"]
    from dsswm.envs.pool_replay import PoolReplayEnv
    return PoolReplayEnv(layer, half), fr.cr_problems("visit"), "visit"


def init_env(T, frozen, eval_task_id=None):
    _ENV.clear()
    _ENV["frozen"] = frozen
    layer, half, eps_list = T["layer"], T["half"], tuple(T["eps"])
    if layer in ("HCZ6", "HCR6"):
        _ENV["sigma2_dev"] = frozen[layer]["sigma2_dev"]
    env, probs, outcome = build_env(layer, half, eval_task_id)
    assert env.half == half
    ctx0 = build_ctx(env, probs, eps_list[0])
    J = true_policy_values(ctx0.pols, env.w, env.true_mu(outcome))
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    ctxs = {e: build_ctx(env, probs, e) for e in eps_list}
    ck = ctx0.checkpoints.copy()
    if T["grid"] == "D":
        g = dense_grid(ck, DENSE_SUB)
        for e in list(ctxs):
            c2 = copy.copy(ctxs[e])
            c2.checkpoints = g
            ctxs[e] = c2
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, ctxs=ctxs, block_points=ck, layer=layer)


def job(a):
    name, kw, seed, eps, stop_k, code_sha, add_sha, data_sha = a
    ctx = _ENV["ctxs"][eps]
    if stop_k is not None:
        ctx = ctx_with_stop(ctx, stop_k)
    try:
        m = make(name, kw, _ENV["layer"])
        t0 = time.perf_counter()
        with R8._CaptureSchedule() as cap:
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
        Us = [r["U"] for r in rows]
        xx = x_rank(Us, ck, ctx.eps, X12_RANK)
        Um = np.minimum.accumulate(np.array(Us, dtype=float), 0) if Us else np.zeros((0, ctx.Q))
        out = {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "stop_k": int(ctx.stop_k)}
        out.update({f"rs_{k}": v for k, v in s.items() if k in ("sec_cert", "sec_plan", "sec_total", "n_plans",
                                                                   "reselected", "skipped", "served", "billed")})
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
                    "K_eval": int(len(ck)), "n_cert_curve": curve_full, "tau_R": tau,
                    "u12_curve": [float(np.sort(Um[k])[X12_RANK - 1]) for k in range(len(rows))],
                    "block_counts": {str(int(r["t"])): r["n_cells"] for r in rows
                                     if int(r["t"]) in set(int(x) for x in _ENV["block_points"])},
                    "sec": sec, "code_sha256": code_sha, "addendum_sha256": add_sha, "data_sha256": data_sha,
                    "error": None})
        if hasattr(m, "n_stale_evals"):
            out["n_stale_evals"] = int(m.n_stale_evals)
        if getattr(m, "beta_trace", None):
            out["beta_trace"] = m.beta_trace
        if name == "PJC-local" and hasattr(m, "ledger") and isinstance(m.ledger, dict) and "beta" in m.ledger:
            out["pjc_beta"] = float(m.ledger["beta"])
        if hasattr(m, "ledger") and isinstance(m.ledger, dict) and "K" in m.ledger:
            out["ledger_K"] = int(m.ledger["K"])
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "params": kw, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def run_jobs(task, out, jobs_all, add_sha=None, data_sha=None):
    rfile = out / "results.jsonl"
    _, code_sha = code_hashes()
    done = set()
    if rfile.exists():
        keep = []
        for line in rfile.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            key = (x["method"], json.dumps(x["params"], sort_keys=True), x["seed"], x["eps"])
            if (x.get("error") is None and x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
                    and x.get("data_sha256") == data_sha and key not in done):
                done.add(key)
                keep.append(line)
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
    rows = [x for x in (json.loads(line) for line in rfile.read_text().splitlines())
            if x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
            and x.get("data_sha256") == data_sha]
    return rows, code_sha


HEAVY = ("FDC-HG", "FDC-LOC", "TU-LOC", "FDC-MR[front3]", "HC-WoR", "PJC-local", "TU-PJC", "TU-FDC", "FDC-BF[K77]")


def make_jobs(T, seeds, cfg):
    jobs = []
    for m in T["methods"]:
        kw = method_kw(m, cfg, T["layer"])
        for e in T["eps"]:
            for s in seeds:
                jobs.append((m, kw, s, e, T["stop_k"]))
    jobs.sort(key=lambda j: (j[0] not in HEAVY,))
    return jobs


def read_rows(path, add_sha=None):
    p = Path(path)
    if not p.exists():
        return []
    return [x for x in (json.loads(line) for line in p.read_text().splitlines())
            if x.get("error") is None and (add_sha is None or x.get("addendum_sha256") == add_sha)]


def current_rows(path, add_sha, data_sha):
    _, code_sha = code_hashes()
    return [r for r in read_rows(path, add_sha) if r.get("code_sha256") == code_sha
            and r.get("addendum_sha256") == add_sha and r.get("data_sha256") == data_sha]


def data_sha_for(layer, lock):
    from dsswm.envs.data_v9 import layer_data_sha_v9
    return layer_data_sha_v9(layer, frozen=None if lock is None else lock["data_sha256"])


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v9 import addendum_gate
    ok, info = addendum_gate(task)
    if not ok:
        (out / "summary.json").write_text(json.dumps({"task_id": task, "status": "skipped_by_lock", "reason": info,
                                                      "written_at": t_start.isoformat(), "eval_touched": False},
                                                     indent=1))
        mark_done(task, "skipped_by_lock", info)
        print(f"[{task}] skipped_by_lock: {info}")
        return None
    return info


def replica(task, T, lock, add_sha):
    from dsswm.stats import v9_replica as R
    from dsswm.stats.v6_replica import check_matrix
    is_eval = T["half"] == "eval"
    base = RES / ("full" if is_eval else "pilots") / task
    data_sha = data_sha_for(T["layer"], lock)
    main_rows = current_rows(base / "results.jsonl", add_sha, data_sha)
    m = check_matrix(main_rows, T["methods"], T["eps"], T["seeds"])
    if not m["complete"]:
        raise SystemExit(f"[{task}] replica refused: planned matrix incomplete {m}")
    if is_eval:
        from dsswm.stats.v9_seal import verify_seal
        verify_seal(SEALS, task, main_rows, T["methods"], T["eps"], T["seeds"], code_hashes()[0], data_sha, add_sha)
    rout = base / "replica"
    rout.mkdir(parents=True, exist_ok=True)
    seeds = T["seeds"][:R.R1_N_SEEDS]
    frozen = load_frozen(lock)
    cfg = frozen.get(cfg_key(T["layer"]), {})
    init_env(T, frozen, eval_task_id=task if is_eval else None)
    run_jobs(task + "_replica", rout, make_jobs(T, seeds, cfg), add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    rep_rows = current_rows(rout / "results.jsonl", add_sha, data_sha)
    rep = R.build_report(task, is_eval, main_rows, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha, per_sha,
                         data_sha, frozen_configs=frozen)
    rep["written_at"] = datetime.now().isoformat()
    (base / "replica_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"task": task, "replica": rep["status"]}))


def _task_rows_and_status(t, dev, add_sha, lock, frozen, per_sha):
    from dsswm.stats import v9_replica as R
    T = TASKS[t]
    base = RES / ("pilots" if dev else "full")
    data_sha = data_sha_for(T["layer"], lock)
    cur = current_rows(base / t / "results.jsonl", add_sha, data_sha)
    rf = base / t / "replica_report.json"
    rep = json.loads(rf.read_text()) if rf.exists() else None
    rep_rows = current_rows(base / t / "replica" / "results.jsonl", add_sha, data_sha)
    sealed = None
    if not dev:
        from dsswm.stats.v9_seal import verify_seal
        sealed = verify_seal(SEALS, t, cur, T["methods"], T["eps"], T["seeds"], per_sha, data_sha,
                             add_sha)["seal"]["results_content_sha256"]
    st = R.validate_report(rep, t, not dev, cur, rep_rows, T["methods"], T["eps"], T["seeds"], add_sha, per_sha,
                           data_sha, sealed_content_sha256=sealed, frozen_configs=frozen)
    return cur, st, rep


def analyse(block, dev):
    from dsswm.stats import v9_replica as R
    from dsswm.stats.v6_replica import check_r2
    if dev:
        tasks, add_sha, lock = DEV_BLOCK_TASKS[block], None, None
        base = RES / "pilots"
    else:
        from dsswm.stats.prereg_v9 import load_locked_addendum
        lock = load_locked_addendum(None)
        add_sha = lock["sha256"]
        tasks, base = BLOCK_TASKS[block], RES / "full"
    per_sha, _ = code_hashes()
    frozen = load_frozen(lock)
    out = {"block": block, "mode": "dev_runner_check" if dev else "eval", "addendum_sha256": add_sha,
           "tasks": list(tasks), "code_sha256": per_sha}
    if block in ("A", "B"):
        sp = dict(VA.CONF_SPEC[block])
        rows, sts, reps = [], {}, []
        for t in tasks:
            cur, st, rep = _task_rows_and_status(t, dev, add_sha, lock, frozen, per_sha)
            rows += VA.suffix_rows(cur, TASKS[t]["grid"])
            sts[t] = st
            reps.append(rep)
        if dev:
            sp["seeds"] = tuple(TASKS[tasks[0]]["seeds"])
            sp["tasks"] = {t: (TASKS[t]["grid"], tuple(TASKS[t]["methods"])) for t in tasks}
        e = float(sp["eps"])
        methods = sorted({r["method"] for r in rows})
        cross_r2b = R.check_r2b(rows, sp["seeds"], (e,), methods)
        half_all = sorted({f"{m}@{TASKS[t]['grid']}" for t in tasks for g in R.groups_for(t, frozen) for m in g
                           if m in TASKS[t]["methods"]})
        cross_r2 = check_r2(rows, tuple(half_all), sp["seeds"], (e,))
        cross_grid = R.check_cross_grid(rows, sp["seeds"], (e,),
                                        tuple(p for p in R.CROSS_PAIRS if p[0] in methods and p[1] in methods))
        ok = all(s == "pass" for s in sts.values()) and cross_r2b["pass"] and cross_r2["pass"] and cross_grid["pass"]
        replica_status = {"status": "pass" if ok else "fail", "task_statuses": sts,
                          "cross_task_R2b": cross_r2b, "cross_task_R2_frozen_design": cross_r2,
                          "cross_grid_R2c": cross_grid}
        res = VA.analyse_confirmatory(block, rows, replica_status, spec=sp)
        res.update(out)
        res.update({"replica_status_combined": replica_status,
                    "replica_content_sha256": [r["results_content_sha256"] for r in reps],
                    "frozen_configs_used": frozen[cfg_key(sp["layer"])], "written_at": datetime.now().isoformat()})
    else:
        res = dict(out, per_task={})
        for t in tasks:
            cur, st, rep = _task_rows_and_status(t, dev, add_sha, lock, frozen, per_sha)
            full = t.replace("pilot", "full")
            sp = dict(VA.C_SPEC[full])
            if dev:
                sp["seeds"] = tuple(TASKS[t]["seeds"])
            res["per_task"][t] = VA.analyse_c(full, cur, {"status": st}, spec=sp)
            res["per_task"][t]["replica_status"] = st
            res["per_task"][t]["replica_content_sha256"] = rep["results_content_sha256"]
        res["frozen_configs_used"] = {k: frozen[k] for k in ("HCZ6", "HCR6")}
        res["written_at"] = datetime.now().isoformat()
    fname = f"v9_analysis_{block}{'_dev' if dev else ''}.json"
    (base / fname).write_text(json.dumps(res, indent=1, default=str))
    dec = res.get("decision") or {t: v["wording"]["wording"] for t, v in res.get("per_task", {}).items()}
    print(json.dumps({"block": block, "decision": dec}, default=str))
    return res


# =============================================================================================== toy (dev)
def toy_dense():
    """Dense-monitoring toy FWER for the new rivals RECT-ck-BF-TU and TU-PJC (r4 near-tie toy, 200 streams), with RECT-ck-BF on
    the K = 20 grid, TU-FDC (dense) and an invalid control: RECT-ck-BF re-run on the dense grid with the K = 20
    ledger and fresh radii at every dense time (no maximal inequality)."""
    from scipy import stats
    out_d = RES / "pilots/v9_toy_dense"
    out_d.mkdir(parents=True, exist_ok=True)
    names = ["RECT-ck-BF-TU", "TU-FDC", "TU-PJC", "RECT-ck-BF+box[K20]", "INVALID-fresh-K20-ledger"]
    with get_context("fork").Pool(N_WORKERS) as pool:
        rows = [r for ch in pool.imap_unordered(_toy_job, [(n, sd) for n in names for sd in range(900, 920)])
                for r in ch]
    with open(out_d / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table = {}
    for n in names:
        rr = [r for r in rows if r["method"] == n]
        ev = sum(r["fwer_event"] for r in rr)
        table[n] = {"n_streams": len(rr), "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                    "cp_upper95": 1.0 if ev >= len(rr) else float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)),
                    "geomean_N80_over_tau": float(np.exp(np.mean(np.log([r["N80_over_tau"] for r in rr])))),
                    "K_eval": rr[0]["K_eval"], "validity": rr[0]["validity"]}
    (out_d / "summary.json").write_text(json.dumps({"written_at": datetime.now().isoformat(),
                                                    "toy": "r4 near-tie toy, pop seeds 900-919 x perm 0-9, eps 0.02",
                                                    "table": table}, indent=1))
    print(json.dumps({k: (v["false_streams"], round(v["geomean_N80_over_tau"], 4)) for k, v in table.items()}))


def _toy_job(a):
    import math
    name, sd = a
    from dsswm.baselines.fdc_loc import FDCTimeUniform
    from dsswm.baselines.pjc_bf import RectCkBF
    from dsswm.baselines.rect_tu_v9 import RectCkBFTU
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    ck = ctx.checkpoints.copy()
    dense = name != "RECT-ck-BF+box[K20]"
    if dense:
        ctx = copy.copy(ctx)
        ctx.checkpoints = dense_grid(ck, DENSE_SUB)
    out = []
    for p in range(10):
        if name == "RECT-ck-BF-TU":
            m = RectCkBFTU(ck, box=True)
        elif name == "TU-FDC":
            m = FDCTimeUniform(ck, name="TU-FDC", localise=False)
        elif name == "TU-PJC":
            from dsswm.baselines.rect_tu_v9 import make_tu_pjc
            m = make_tu_pjc(ck, mode="local", boundaries=(), rule="half", rect=False)
        elif name == "RECT-ck-BF+box[K20]":
            m = RectCkBF(box=True)
        else:
            class _Inv(RectCkBF):
                validity = "none"

                def _ledger(self, c):
                    L = RectCkBF._ledger(self, c)
                    K, SA = len(ck), c.S * c.A
                    L.update(beta_c=math.log(2 * SA * K / 0.045), alpha_side_var=0.005 / (2.0 * SA * K))
                    return L
            m = _Inv(box=True, name="INVALID-fresh-K20-ledger")
        s, _ = run_stream_v6(env, m, 1000 * sd + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        out.append({"method": name, "K_eval": int(len(ctx.checkpoints)), "validity": m.validity,
                    "population_seed": sd, "perm_seed": s["perm_seed"], "fwer_event": bool(s["fwer_event"]),
                    "n_cert": s["n_cert"], "n_false": s["n_false"], "N80_over_tau": s["N80"] / env.tau_R})
    return out


# =============================================================================================== main
def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--replica", action="store_true")
    ap.add_argument("--analyse", choices=("A", "B", "C"))
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze-hv9", action="store_true")
    ap.add_argument("--toy-dense", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 per process; host rule)")
    args = ap.parse_args()
    N_WORKERS = max(1, min(4, args.workers))
    if args.freeze_hv9:
        freeze_hv9()
        return
    if args.toy_dense:
        toy_dense()
        return
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
        from dsswm.stats.v9_seal import seal_exists
        if seal_exists(SEALS, task):
            raise SystemExit(f"[{task}] already sealed: a sealed eval task is never re-run or resumed")
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    frozen = load_frozen(lock)
    cfg = frozen.get(cfg_key(T["layer"]), {})
    data_sha = data_sha_for(T["layer"], lock)
    init_env(T, frozen, eval_task_id=task if is_eval else None)
    env = _ENV["env"]
    jobs = make_jobs(T, T["seeds"], cfg)
    rows, code_sha = run_jobs(task, out, jobs, add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    t_end = datetime.now()
    from dsswm.stats.v6_replica import check_r2
    from dsswm.stats.v9_replica import check_r2b, groups_for
    eps_list = tuple(T["eps"])
    r2 = [check_r2(rows, tuple(x for x in g if x in T["methods"]), T["seeds"], eps_list)
          for g in groups_for(task, frozen) if len([x for x in g if x in T["methods"]]) >= 2]
    summary = {"task_id": task, "status": "complete", "block": T["block"], "layer": T["layer"], "half": T["half"],
               "grid": T["grid"], "K_eval": int(len(next(iter(_ENV["ctxs"].values())).checkpoints)),
               "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]), "eps": list(eps_list),
               "stop_k": T["stop_k"], "tau_R": int(env.tau_R), "n_rows": len(rows), "expected_rows": len(jobs),
               "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
               "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
               "cpu_sec_sum": round(float(sum(r["sec"] for r in rows)), 1),
               "timing_note": f"{N_WORKERS} worker processes, BLAS=1; concurrent with other tasks (biased up)",
               "code_sha256": per_sha, "code_sha256_combined": code_sha, "addendum_sha256": add_sha,
               "data_sha256": data_sha, "eval_touched": bool(is_eval),
               "configs": {m: method_kw(m, cfg, T["layer"]) for m in T["methods"]},
               "R2_design_identity": r2, "R2b_arrival_identity": check_r2b(rows, T["seeds"], eps_list, T["methods"])}
    if is_eval:
        from dsswm.stats.v9_seal import make_seal, write_and_commit_seal
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
