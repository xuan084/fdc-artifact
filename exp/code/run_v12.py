"""Lock-v12 ADDENDUM runner (NEW file; v5-v11 files untouched and imported unchanged).  plan/v12_obd2_plan.md.

Two Open Bandit campaigns, each under its own frozen dev design (dsswm.envs.obd_v12_eval; v11 written rules):
block A = random/women (primary), block B = random/men.  Frozen 50/50 design, delta 0.05, K = 40 checkpoint grid,
streams run to 15/15 or tau, endpoint N80_pen at 12/15.  No pooling.

Steps (cwd = exp/code; C in {women, men})
  run_v12.py --campaign C --freeze-design         # once, dev only: exp/results/v12_gates/obd_v12_C_frozen.json
  run_v12.py --task v12_C_dev_rule_e0 .. _e4       # dev: RECT-BF-DP-TU + 8 HC configs at one eps of the grid
  run_v12.py --campaign C --select                 # HC tuning + eps rule + block-status gate (no FDC row read)
                                                   #   -> v12_C_eps_hc.json, v12_C_block_status.json
  run_v12.py --task v12_C_pilot                    # dev eps cell, all four methods, seeds 950-999
  run_v12.py --task v12_C_pilot --replica          # dev runner check of the replica machinery
  run_v12.py --campaign C --analyse --dev          # -> exp/results/pilots/v12_dev/v12_C_analysis_dev.json
  run_v12.py --campaign C --freeze-thresh          # THRESH rule -> v12_C_thresh.json + MANIFEST.json
  run_v12.py --task v12_C_full                     # EVAL (locked v12 addendum required; gated, logged, sealed)
  run_v12.py --task v12_C_full --replica
  run_v12.py --campaign C --analyse                # -> exp/results/full/v12_obd/v12_C_analysis.json
Rows are written ahead (fsync) and resumable, except sealed eval tasks.  CPU only, BLAS threads 1, <= 4 workers.
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
GATES = RES / "v12_gates"
DEV_OUT = RES / "pilots" / "v12_dev"
EVAL_OUT = RES / "full" / "v12_obd"
SEALS = RES / "full" / "v12_seals"
MANIFEST_JSON = GATES / "MANIFEST.json"
N_WORKERS = 4
STOP_FRAC = 1.0
DELTA = 0.05

from dsswm.envs.obd_v12_eval import CAMPAIGNS, LAYERS  # noqa: E402
from dsswm.stats import v12_analysis as VA  # noqa: E402

K_GRID, N80_K = VA.K_GRID, VA.N80_K
CODE_FILES = ["run_v12.py", "dsswm/envs/obd_v12_eval.py", "dsswm/stats/prereg_v12.py", "dsswm/stats/v12_analysis.py",
              "dsswm/stats/v12_seal.py",
              # v11 modules reused by import (unchanged, lock-v11-bound)
              "dsswm/envs/obd_v11_eval.py", "dsswm/envs/obd_v11.py", "dsswm/stats/prereg_v11.py",
              "dsswm/stats/v11_analysis.py", "dsswm/stats/v11_replica.py", "dsswm/stats/v11_seal.py",
              "dsswm/baselines/rect_hg_dp.py", "dsswm/baselines/fdc_dp.py", "dsswm/baselines/rect_dp.py",
              "dsswm/envs/seg_v10.py", "dsswm/envs/seg_v10_eval.py", "dsswm/stats/prereg_v10.py",
              # inherited dependencies (imported, unchanged)
              "dsswm/baselines/rect_tu_v9.py", "dsswm/baselines/pjc_bf.py", "dsswm/baselines/wor_betting_v6.py",
              "dsswm/baselines/fdc_bet.py", "dsswm/baselines/b4_bal.py", "dsswm/baselines/rect_v6.py",
              "dsswm/baselines/frontier_common.py", "dsswm/streams/frontier_runner.py",
              "dsswm/streams/frontier_runner_v6.py", "dsswm/streams/frontier.py", "dsswm/envs/pool_replay.py",
              "dsswm/envs/lenta_v6.py", "dsswm/envs/data_v6.py", "dsswm/envs/data_v8.py", "dsswm/envs/base.py",
              "dsswm/stats/prereg.py", "dsswm/stats/prereg_v6.py", "dsswm/stats/prereg_v7.py",
              "dsswm/stats/prereg_v8.py", "dsswm/stats/prereg_v9.py", "dsswm/stats/v6_analysis.py",
              "dsswm/stats/v6_replica.py", "dsswm/stats/v8_replica.py", "dsswm/stats/v6_seal.py",
              "dsswm/stats/v8_seal.py", "dsswm/stats/v9_seal.py", "dsswm/baselines/fdc.py", "dsswm/core/actions.py",
              "dsswm/core/state.py", "dsswm/stats/fp_eb.py"]
LOCK_ONLY_FILES = ["build_v12_addendum_draft.py", "split_obd_v12.py", "dsswm/tests/test_v12_addendum.py"]


def _rule_task(c, i):
    return f"v12_{c}_dev_rule_e{i}"


def pilot_task(c):
    return f"v12_{c}_pilot"


def eval_task(c):
    return f"v12_{c}_full"


TASKS = {}
for _c in CAMPAIGNS:
    for _i, _e in enumerate(VA.EPS_GRID):
        TASKS[_rule_task(_c, _i)] = {"role": "eps_rule", "half": "dev", "eps": float(_e),
                                     "methods": [VA.RIVAL] + list(VA.HC_NAMES), "seeds": list(VA.DEV_SEEDS)}
    TASKS[pilot_task(_c)] = {"role": "dev_cell", "half": "dev", "eps": None, "methods": list(VA.METHODS),
                             "seeds": list(VA.DEV_SEEDS)}
    TASKS[eval_task(_c)] = {"role": "eval_block", "half": "eval", "eps": None, "methods": list(VA.METHODS),
                            "seeds": list(VA.EVAL_SEEDS[_c])}
for _t in list(TASKS):
    TASKS[_t]["campaign"] = _t.split("_")[1]
    TASKS[_t]["layer"] = LAYERS[TASKS[_t]["campaign"]]
EVAL_TASKS = [eval_task(c) for c in CAMPAIGNS]
_ENV: dict = {}


def gate_file(c, kind):
    return GATES / {"frozen": f"obd_v12_{c}_frozen.json", "eps_hc": f"v12_{c}_eps_hc.json",
                    "status": f"v12_{c}_block_status.json", "thresh": f"v12_{c}_thresh.json"}[kind]


GATE_FILES = [gate_file(c, k).name for c in CAMPAIGNS for k in ("frozen", "eps_hc", "status", "thresh")]


# =============================================================================================== helpers
def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes():
    per = {p: sha_file(CODE / p) for p in CODE_FILES}
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


def gate_hashes(c=None):
    cs = CAMPAIGNS if c is None else (c,)
    return {f"exp/results/v12_gates/{gate_file(x, k).name}": sha_file(gate_file(x, k))
            for x in cs for k in ("frozen", "eps_hc", "status", "thresh") if gate_file(x, k).exists()}


def eps_hc(c):
    p = gate_file(c, "eps_hc")
    if not p.exists():
        raise SystemExit(f"{p.name} missing: run the rule tasks and --select first")
    return json.loads(p.read_text())


def block_status(c):
    p = gate_file(c, "status")
    if not p.exists():
        raise SystemExit(f"{p.name} missing: run --select first")
    return json.loads(p.read_text())


def frozen_configs(c, with_thresh=True):
    """Everything campaign c's dev cell / eval run needs beyond the design json, read from its gate files."""
    from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
    sel = eps_hc(c)
    eps = float(sel["block_eps"])
    hc = sel["hc_tuned_at_block_eps"]
    st = block_status(c)
    if float(st["eps"]) != eps:
        raise RuntimeError("block-status gate was computed at a different eps")
    cfg = {"campaign": c, "eps": eps, "block_status": st["status"], "delta": DELTA, "K": K_GRID, "n80_k": N80_K,
           "stop_k": "Q = 15 (stop_frac 1.0)",
           "checkpoints": "fr.checkpoints(n_min = 5000, tau_R = replay N, K = 40)",
           "design": "balanced_alloc(S, A, 0.5) for every method",
           "TU-FDC-DP(b)": {"scheme": "b", "node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO,
                            "split": [0.045, 0.005], "block_points": "the K = 40 grid"},
           "RECT-BF-DP-TU": {"box": True, "split": [0.045, 0.005], "block_points": "the K = 40 grid"},
           "HC-WoR-DP[tuned]": {"schedule": "nstar", "c": float(hc["c"]), "tf": float(hc["tf"]),
                                "source": f"{gate_file(c, 'eps_hc').name} (block-G3 rule at the block eps, dev seeds "
                                          f"950-999)"},
           "RECT-HG-DP": {"guarantee": "checkpoint-strength (CP): valid at the K grid points only",
                          "alpha_side": "delta / (2 S A K)", "row_validity_label": VA.VALIDITY_LABEL[VA.RECT_HG]}}
    if with_thresh:
        p = gate_file(c, "thresh")
        if not p.exists():
            raise SystemExit(f"{p.name} missing: run --analyse --dev and --freeze-thresh first")
        th = json.loads(p.read_text())
        if float(th["eps"]) != eps:
            raise RuntimeError("THRESH gate was computed at a different eps")
        cfg["thresh"] = float(th["thresh"])
        cfg["thresh_applicable"] = st["status"] == "confirmatory"
    return cfg


def load_frozen(c, lock=None, with_thresh=True):
    cfg = frozen_configs(c, with_thresh)
    gates = gate_hashes()
    if lock is not None:
        if lock.get("frozen_configs", {}).get(c) != json.loads(json.dumps(cfg)):
            raise RuntimeError(f"{c}: frozen configs differ from the locked v12 addendum")
        if lock.get("frozen_gates") != gates:
            raise RuntimeError("v12 gate files differ from the locked v12 addendum")
        b = lock["blocks"][c]
        if float(b["eps"]) != cfg["eps"] or float(b["thresh"]) != cfg["thresh"] or b["status"] != cfg["block_status"]:
            raise RuntimeError(f"{c}: eps / THRESH / block status differ from the locked v12 addendum")
    return cfg, gates


def make(name, ck, cfg=None):
    from dsswm.baselines.fdc_dp import FDCDPTimeUniform
    from dsswm.baselines.rect_dp import HCWoRDP, RectBFDPTU
    from dsswm.baselines.rect_hg_dp import RectHGDP
    if name == VA.PRIMARY:
        c = cfg[VA.PRIMARY]
        return FDCDPTimeUniform(ck, scheme="b", node_limit=c["node_limit"], grid_ratio=c["grid_ratio"])
    if name == VA.RIVAL:
        return RectBFDPTU(ck, box=True)
    if name == VA.RECT_HG:
        return RectHGDP()
    if name == VA.HC_TUNED:
        c = cfg[VA.HC_TUNED]
        return HCWoRDP("nstar", c["c"], c["tf"], name=VA.HC_TUNED)
    if name in VA.HC_NAMES:
        c, tf = VA.HC_GRID[VA.HC_NAMES.index(name)]
        return HCWoRDP("nstar", c, tf, name=name)
    raise KeyError(name)


def schedule_arrival_digest(sch):
    h = hashlib.sha256(np.ascontiguousarray(sch.seg_seq).tobytes())
    for p in sch.pool_perm:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()[:16]


class _Capture:
    """Records the schedule the stream ACTUALLY consumed (wraps seg_v10.make_schedule for one run)."""

    def __enter__(self):
        from dsswm.envs import seg_v10 as SV
        self._m, self._orig, self.seen = SV, SV.make_schedule, []

        def wrapped(*a, **k):
            sch = self._orig(*a, **k)
            self.seen.append(sch)
            return sch

        SV.make_schedule = wrapped
        return self

    def __exit__(self, *exc):
        self._m.make_schedule = self._orig
        return False

    def digest(self):
        if len(self.seen) != 1:
            raise RuntimeError(f"expected exactly one schedule per run, captured {len(self.seen)}")
        return schedule_arrival_digest(self.seen[0])


def make_env(c, half, lock=None, eval_task_id=None):
    from dsswm.envs.obd_v12_eval import OBDEnvV12Frozen
    fsha = None if lock is None else lock["frozen_gates"][f"exp/results/v12_gates/{gate_file(c, 'frozen').name}"]
    return OBDEnvV12Frozen(c, half, eval_task_id=eval_task_id, lock_sha256=None if lock is None else lock["sha256"],
                           frozen_sha256=fsha)


def init_env(T, eps, cfg, lock=None, eval_task_id=None):
    from dsswm.baselines.fdc_dp import make_seg_ctx
    from dsswm.envs.seg_v10 import env_digest, true_opt
    env = make_env(T["campaign"], T["half"], lock, eval_task_id)
    assert env.half == T["half"] and env.campaign == T["campaign"]
    Js, mu = true_opt(env)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, eps, DELTA, env.checkpoints(K_GRID), env.tau_R,
                       env.replan_interval, stop_frac=STOP_FRAC)
    _ENV.clear()
    _ENV.update(env=env, Js=Js, mu=mu, ctx=ctx, digest=env_digest(env), cfg=cfg)
    return env


def job(a):
    name, seed, eps, code_sha, add_sha, data_sha = a
    from dsswm.envs.seg_v10 import run_stream_seg
    ctx = _ENV["ctx"]
    try:
        m = make(name, ctx.checkpoints.copy(), _ENV["cfg"])
        t0 = time.perf_counter()
        with _Capture() as cap:
            s, rows = run_stream_seg(_ENV["env"], m, seed, ctx, _ENV["Js"], _ENV["mu"], n80_k=N80_K)
        sec = round(time.perf_counter() - t0, 3)
        out = {"method": name, "seed": int(seed), "eps": float(eps)}
        out.update({k: s[k] for k in ("validity", "schedule_digest", "reached_stop", "stop_k", "n80_k", "k_stop",
                                      "k80", "N80_pen", "N80_raw", "n80_lt_tau", "false_by_k80",
                                      "exhaustion_at_k80", "N_stop_pen", "n_cert", "n_false", "fwer_event",
                                      "cert_k", "decided_pi", "billing_ok", "n_cert_curve", "tau_R")})
        if name in VA.VALIDITY_LABEL:                      # RECT-HG-DP: checkpoint strength only (never 'rigorous')
            out["validity_class_attr"] = out["validity"]
            out["validity"] = VA.VALIDITY_LABEL[name]
        out.update({"campaign": _ENV["env"].campaign, "arrival_digest": cap.digest(), "env_digest": _ENV["digest"],
                    "K_eval": int(len(ctx.checkpoints)), "sec": sec,
                    "rs_sec_cert": {"total": s["sec_cert"], "by_k": s["sec_cert_by_k"]},
                    "rs_sec_total": s["sec_total"], "code_sha256": code_sha, "addendum_sha256": add_sha,
                    "data_sha256": data_sha, "error": None})
        if hasattr(m, "n_stale_evals"):
            out["n_stale_evals"] = int(m.n_stale_evals)
        cs = getattr(m, "cert_stats", None)
        if cs:
            out.update({"bnb_nodes": int(sum(x["nodes"] for x in cs)), "bnb_calls": int(sum(x["bnb_calls"] for x in cs)),
                        "node_limit_hits": int(sum(x["node_limit_hits"] for x in cs)),
                        "b_only_certs": int(sum(x["b_certified"] for x in cs)),
                        "a_certs": int(sum(x["a_certified"] for x in cs)),
                        "beta": float(m.ledger["beta"]), "ledger_K": int(m.ledger["K"])})
        if name.startswith("HC-WoR-DP"):
            out["hc_config"] = {"schedule": m.schedule, "c": float(m.c), "target_frac": float(m.target_frac)}
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def _keep(x, code_sha, add_sha, data_sha):
    return (x.get("error") is None and x.get("code_sha256") == code_sha and x.get("addendum_sha256") == add_sha
            and x.get("data_sha256") == data_sha)


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
            key = (x["method"], x["seed"], x["eps"])
            if _keep(x, code_sha, add_sha, data_sha) and key not in done:
                done.add(key)
                keep.append(line)
        rfile.write_text("".join(k + "\n" for k in keep))
    jobs = [j for j in jobs_all if (j[0], j[1], float(j[2])) not in done]
    total = len(jobs_all)
    n_done = total - len(jobs)
    print(f"[{task}] {total} jobs, {len(jobs)} to run", flush=True)
    errs = []
    if jobs:
        jobs = [j[:3] + (code_sha, add_sha, data_sha) for j in jobs]
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
    return current_rows(rfile, add_sha, data_sha), code_sha


def make_jobs(T, eps, seeds):
    jobs = [(m, s, float(eps)) for m in T["methods"] for s in seeds]
    jobs.sort(key=lambda j: not j[0].startswith("HC-WoR-DP"))
    return jobs


def current_rows(path, add_sha, data_sha):
    _, code_sha = code_hashes()
    p = Path(path)
    if not p.exists():
        return []
    return [x for x in (json.loads(line) for line in p.read_text().splitlines()) if _keep(x, code_sha, add_sha, data_sha)]


def data_sha_for(lock):
    """Row binding hash.  Never opens an eval file: eval-file hashes come from the campaign PROVENANCE.json; eval bytes
    are verified inside the gated, logged reader."""
    from dsswm.envs.obd_v12_eval import data_sha_v12
    return data_sha_v12(None if lock is None else lock["data_sha256"])


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v12 import addendum_gate
    ok, info = addendum_gate(task)
    if not ok:
        (out / "summary.json").write_text(json.dumps({"task_id": task, "status": "skipped_by_lock", "reason": info,
                                                      "written_at": t_start.isoformat(), "eval_touched": False},
                                                     indent=1))
        mark_done(task, "skipped_by_lock", info)
        print(f"[{task}] skipped_by_lock: {info}")
        return None
    return info


def task_dir(task):
    return (EVAL_OUT if TASKS[task]["half"] == "eval" else DEV_OUT) / task


def task_eps_cfg(task, lock=None):
    """(eps, cfg): rule tasks use their grid eps and need no gate file; the dev cell and the eval task use the frozen
    block eps / HC config (the eval task also the frozen THRESH and block status, checked against the lock)."""
    T = TASKS[task]
    if T["role"] == "eps_rule":
        from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
        return T["eps"], {VA.PRIMARY: {"node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO}}
    if T["half"] == "eval":
        cfg, _ = load_frozen(T["campaign"], lock, with_thresh=True)
    else:
        cfg = frozen_configs(T["campaign"], with_thresh=False)
    return cfg["eps"], cfg


# =============================================================================================== replica
def replica(task, lock, add_sha):
    from dsswm.stats import v11_replica as R
    from dsswm.stats.v6_replica import check_matrix
    T = TASKS[task]
    is_eval = T["half"] == "eval"
    base = task_dir(task)
    data_sha = data_sha_for(lock)
    eps, cfg = task_eps_cfg(task, lock)
    main_rows = current_rows(base / "results.jsonl", add_sha, data_sha)
    m = check_matrix(main_rows, T["methods"], (eps,), T["seeds"])
    if not m["complete"]:
        raise SystemExit(f"[{task}] replica refused: planned matrix incomplete {m}")
    if is_eval:
        from dsswm.stats.v12_seal import verify_seal
        verify_seal(SEALS, task, main_rows, T["methods"], (eps,), T["seeds"], code_hashes()[0], data_sha, add_sha)
    rout = base / "replica"
    rout.mkdir(parents=True, exist_ok=True)
    seeds = T["seeds"][:R.R1_N_SEEDS]
    init_env(T, eps, cfg, lock, eval_task_id=task if is_eval else None)
    run_jobs(task + "_replica", rout, make_jobs(T, eps, seeds), add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    rep_rows = current_rows(rout / "results.jsonl", add_sha, data_sha)
    rep = R.build_report(task, is_eval, main_rows, rep_rows, T["methods"], (eps,), T["seeds"], add_sha, per_sha,
                         data_sha)
    rep["written_at"] = datetime.now().isoformat()
    (base / "replica_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"task": task, "replica": rep["status"]}))


# =============================================================================================== dev selection / THRESH
def select(c):
    """HC tuning per eps + eps rule from campaign c's rule-task rows only (no FDC-DP row is loaded), then the pre-stated
    block-status gate on the dev truth at the block eps."""
    from dsswm.envs.obd_v12_eval import eps_opt_share
    from dsswm.envs.seg_v10 import true_opt
    pe, ps = gate_file(c, "eps_hc"), gate_file(c, "status")
    if pe.exists() or ps.exists():
        raise SystemExit(f"{pe.name} / {ps.name} exist: the eps / HC / status selection is never silently rewritten")
    data_sha = data_sha_for(None)
    rows = []
    for i in range(len(VA.EPS_GRID)):
        t = _rule_task(c, i)
        rr = current_rows(task_dir(t) / "results.jsonl", None, data_sha)
        if len(rr) != len(TASKS[t]["methods"]) * len(TASKS[t]["seeds"]):
            raise SystemExit(f"{t}: {len(rr)} current rows (incomplete or stale)")
        rows += rr
    hc = VA.select_hc(rows)
    rule = VA.select_eps([r for r in rows if r["method"] == VA.RIVAL or r["method"] in VA.HC_NAMES], hc)
    beps = VA.block_eps(rule)
    out = {"campaign": c, "written_at": datetime.now().isoformat(), "code_sha256": code_hashes()[1],
           "data_sha256": data_sha,
           "rule": "smallest eps of the grid at which the dev-best of {RECT-BF-DP-TU, HC-WoR-DP tuned at that eps} "
                   "(lower dev geomean N80_pen; tie -> RECT-BF-DP-TU) has strict N80_pen < tau on >= 40/50 dev "
                   "streams (seeds 950-999); FDC-DP never read; none -> block eps = 1.5e-3 (descriptive)",
           "hc_rule": "per eps, lowest dev geomean N80_pen over HC_GRID (nstar; c in {0.5, 0.75}; tf in {0.2, 0.4, "
                      "0.6, 0.8}); ties -> grid order",
           "eps_grid": list(VA.EPS_GRID), "hc_by_eps": hc, "eps_rule": rule, "block_eps": beps,
           "hc_tuned_at_block_eps": {k: hc[repr(float(beps))][k] for k in ("name", "c", "tf")}}
    env = make_env(c, "dev")
    Js, mu = true_opt(env)
    st = VA.block_status(rule, eps_opt_share(env, Js, mu, beps))
    st.update({"campaign": c, "written_at": datetime.now().isoformat(), "code_sha256": code_hashes()[1],
               "data_sha256": data_sha, "dev_env": env.describe(),
               "dev_base_ctr": float(np.asarray(env.outcomes_view("visit")).mean())})
    GATES.mkdir(parents=True, exist_ok=True)
    pe.write_text(json.dumps(out, indent=1))
    ps.write_text(json.dumps(st, indent=1))
    print(json.dumps({"campaign": c, "selected_eps": rule["selected_eps"], "block_eps": beps,
                      "rival": rule["rival_method"], "successes": rule["successes"],
                      "hc": out["hc_tuned_at_block_eps"], "status": st["status"], "criteria": st["criteria"],
                      "mean_share": st["mean_share_eps_optimal"],
                      "all_control_eps_opt": st["n_problems_all_control_eps_optimal"]}))


def freeze_thresh(c):
    p = gate_file(c, "thresh")
    if p.exists():
        raise SystemExit(f"{p} exists: THRESH is never silently rewritten")
    src = DEV_OUT / f"v12_{c}_analysis_dev.json"
    a = json.loads(src.read_text())
    _, code_sha = code_hashes()
    if a["code_sha256_combined"] != code_sha:
        raise SystemExit("dev analysis was produced by different code; re-run --analyse --dev")
    if a["replica_status"] != "pass":
        raise SystemExit("dev replica check did not pass")
    comp = a["cell"]["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]
    th = VA.thresh_rule(comp["ub95_one_sided"])
    out = {"campaign": c, "written_at": datetime.now().isoformat(),
           "rule": "THRESH = min(0.90, UB95_dev + 0.10), rounded up to 3 decimals (plan/v12_obd2_plan.md s7)",
           "eps": a["cell"]["eps"], "ub95_dev": comp["ub95_one_sided"], "ratio_dev": comp["geomean_ratio"],
           "n_dev_streams": a["cell"]["n_streams"], "bootstrap": a["cell"]["bootstrap"], "thresh": th,
           "block_status": block_status(c)["status"], "applicable": block_status(c)["status"] == "confirmatory",
           "applicability_note": "THRESH is compared with the eval UB95 only in a confirmatory block; in a "
                                 "descriptive block it is computed by the same rule and recorded, but not applied",
           "source": str(src.relative_to(WS)), "source_sha256": sha_file(src)}
    p.write_text(json.dumps(out, indent=1))
    write_manifest()
    print(json.dumps({"campaign": c, "ub95_dev": out["ub95_dev"], "thresh": th}))


def write_manifest():
    files = {f: sha_file(GATES / f) for f in GATE_FILES if (GATES / f).exists()}
    MANIFEST_JSON.write_text(json.dumps({"written_at": datetime.now().isoformat(), "sha256": files}, indent=1))


# =============================================================================================== analysis
def analyse(c, dev):
    from dsswm.envs.obd_v12_eval import eps_opt_share
    from dsswm.envs.seg_v10 import true_opt
    from dsswm.stats import v11_replica as R
    task = pilot_task(c) if dev else eval_task(c)
    T = TASKS[task]
    lock, add_sha = None, None
    if not dev:
        from dsswm.stats.prereg_v12 import load_locked_addendum
        lock = load_locked_addendum(task)
        add_sha = lock["sha256"]
    eps, cfg = task_eps_cfg(task, lock)
    per_sha, code_sha = code_hashes()
    data_sha = data_sha_for(lock)
    base = task_dir(task)
    cur = current_rows(base / "results.jsonl", add_sha, data_sha)
    rf = base / "replica_report.json"
    rep = json.loads(rf.read_text()) if rf.exists() else None
    rep_rows = current_rows(base / "replica" / "results.jsonl", add_sha, data_sha)
    sealed = None
    if not dev:
        from dsswm.stats.v12_seal import verify_seal
        sealed = verify_seal(SEALS, task, cur, T["methods"], (eps,), T["seeds"], per_sha, data_sha,
                             add_sha)["seal"]["results_content_sha256"]
    st = R.validate_report(rep, task, not dev, cur, rep_rows, T["methods"], (eps,), T["seeds"], add_sha, per_sha,
                           data_sha, sealed_content_sha256=sealed)
    cell = VA.analyse_cell_v12(cur, T["seeds"], eps)
    env = make_env(c, T["half"], lock, None if dev else task)
    Js, mu = true_opt(env)
    base_ctr = float(np.asarray(env.outcomes_view("visit")).mean())
    bst = block_status(c)
    dev_base = float(bst["dev_base_ctr"])
    res = {"task": task, "campaign": c, "mode": "dev" if dev else "eval", "eps": eps, "block_status": bst["status"],
           "addendum_sha256": add_sha, "code_sha256": per_sha, "code_sha256_combined": code_sha,
           "data_sha256": data_sha, "frozen_gates": gate_hashes(), "frozen_configs_used": cfg,
           "replica_status": st, "cell": cell,
           "eps_units": {"absolute_ctr": eps, "dev_base_ctr": dev_base, "relative_to_dev_base": eps / dev_base,
                         "replay_half_base_ctr": base_ctr, "relative_to_replay_base": eps / base_ctr},
           "eps_optimal_share": eps_opt_share(env, Js, mu, eps), "env": env.describe(),
           "Jstar": [float(x) for x in Js], "written_at": datetime.now().isoformat()}
    comp = cell["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]
    if dev:
        res["thresh_if_frozen_now"] = VA.thresh_rule(comp["ub95_one_sided"])
        res["decision_if_applied"] = VA.decide_block(comp, cell["primary_false_streams"], {"status": st},
                                                     res["thresh_if_frozen_now"], bst["status"])
        out = DEV_OUT / f"v12_{c}_analysis_dev.json"
    else:
        b = lock["blocks"][c]
        res["decision"] = VA.decide_block(comp, cell["primary_false_streams"], {"status": st}, float(b["thresh"]),
                                          b["status"])
        out = EVAL_OUT / f"v12_{c}_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"task": task, "eps": eps, "status": bst["status"], "ratio": round(comp["geomean_ratio"], 4),
                      "ub95": round(comp["ub95_one_sided"], 4), "primary_false": cell["primary_false_streams"],
                      "replica": st, "verdict": (res.get("decision") or res.get("decision_if_applied"))["verdict"]}))
    return res


# =============================================================================================== main
def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--campaign", choices=list(CAMPAIGNS))
    ap.add_argument("--replica", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze-design", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--freeze-thresh", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 per process; host rule)")
    args = ap.parse_args()
    N_WORKERS = max(1, min(4, args.workers))
    if args.freeze_design or args.select or args.freeze_thresh or args.analyse:
        if args.campaign is None:
            raise SystemExit("--campaign is required")
    if args.freeze_design:
        from dsswm.envs.obd_v12_eval import freeze_design
        fz = freeze_design(args.campaign)
        print(json.dumps({k: fz[k] for k in ("campaign", "S", "A", "cost", "budgets", "dev_segment_shares")}))
        return
    if args.select:
        select(args.campaign)
        return
    if args.freeze_thresh:
        freeze_thresh(args.campaign)
        return
    if args.analyse:
        analyse(args.campaign, args.dev)
        return
    task = args.task
    T = TASKS[task]
    is_eval = T["half"] == "eval"
    out = task_dir(task)
    out.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    lock, add_sha = None, None
    if is_eval:
        lock = gate_or_skip(task, out, t_start)
        if lock is None:
            return
        add_sha = lock["sha256"]
    else:
        assert all(s in VA.DEV_SEEDS for s in T["seeds"]), "dev tasks may only use seeds 950-999"
    if args.replica:
        replica(task, lock, add_sha)
        return
    if is_eval:
        from dsswm.stats.v12_seal import seal_exists
        if seal_exists(SEALS, task):
            raise SystemExit(f"[{task}] already sealed: a sealed eval task is never re-run or resumed")
    eps, cfg = task_eps_cfg(task, lock)
    (RES / f"{task}.pid").write_text(str(os.getpid()))
    data_sha = data_sha_for(lock)
    env = init_env(T, eps, cfg, lock, eval_task_id=task if is_eval else None)
    jobs = make_jobs(T, eps, T["seeds"])
    rows, code_sha = run_jobs(task, out, jobs, add_sha=add_sha, data_sha=data_sha)
    per_sha, _ = code_hashes()
    t_end = datetime.now()
    from dsswm.stats.v6_replica import check_r2
    from dsswm.stats.v11_replica import check_r2b, groups_for
    summary = {"task_id": task, "status": "complete", "role": T["role"], "campaign": T["campaign"],
               "layer": T["layer"], "half": T["half"], "eps": eps, "K": K_GRID,
               "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]),
               "tau_R": int(env.tau_R), "pool_sizes_min": int(env.pool_sizes.min()), "env": env.describe(),
               "n_rows": len(rows), "expected_rows": len(jobs), "started_at": t_start.isoformat(),
               "ended_at": t_end.isoformat(), "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
               "cpu_sec_sum": round(float(sum(r["sec"] for r in rows)), 1),
               "timing_note": f"{N_WORKERS} worker processes, BLAS=1; concurrent with other tasks (biased up)",
               "code_sha256": per_sha, "code_sha256_combined": code_sha, "addendum_sha256": add_sha,
               "data_sha256": data_sha, "eval_touched": bool(is_eval),
               "R2_design_identity": [check_r2(rows, g, T["seeds"], (eps,)) for g in groups_for(task, T["methods"])],
               "R2b_arrival_identity": check_r2b(rows, T["seeds"], (eps,), T["methods"])}
    if is_eval:
        from dsswm.stats.v12_seal import make_seal, write_and_commit_seal
        seal = make_seal(task, rows, T["methods"], (eps,), T["seeds"], per_sha, data_sha, add_sha)
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
