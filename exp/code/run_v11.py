"""Lock-v11 ADDENDUM runner (new file; v5-v10 files untouched and imported unchanged).  plan/v11_obd_plan.md.

Open Bandit Dataset (random/all) under the frozen dev design of dsswm.envs.obd_v11_eval (S = 9 uf0 x uf3 segments,
A = 2 item groups, exposure costs, 15 budgets), frozen 50/50 design, delta 0.05, K = 40 checkpoint grid, streams run
to 15/15 or tau, endpoint N80_pen at 12/15.

Steps (cwd = exp/code)
  run_v11.py --freeze-design                       # once, dev only: exp/results/v11_gates/obd_v11_frozen.json
  run_v11.py --task v11_dev_rule_e0 .. _e4         # dev: RECT-BF-DP-TU + 8 HC configs at one eps of the grid
  run_v11.py --select                              # HC tuning + eps rule (no FDC row read) -> v11_gates/v11_eps_hc.json
  run_v11.py --task v11_obd_pilot                  # dev eps cell, all four methods, seeds 950-999
  run_v11.py --task v11_obd_pilot --replica        # dev runner check of the replica machinery
  run_v11.py --analyse --dev                       # dev analysis -> exp/results/pilots/v11_dev/v11_analysis_dev.json
  run_v11.py --freeze-thresh                       # THRESH rule -> v11_gates/v11_thresh.json + MANIFEST.json
  run_v11.py --task v11_obd_full                   # EVAL (locked v11 addendum required; gated, logged, sealed)
  run_v11.py --task v11_obd_full --replica
  run_v11.py --analyse                             # eval analysis -> exp/results/full/v11_obd/v11_analysis.json
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
GATES = RES / "v11_gates"
DEV_OUT = RES / "pilots" / "v11_dev"
EVAL_OUT = RES / "full" / "v11_obd"
SEALS = RES / "full" / "v11_seals"
EPS_HC_JSON = GATES / "v11_eps_hc.json"
THRESH_JSON = GATES / "v11_thresh.json"
MANIFEST_JSON = GATES / "MANIFEST.json"
GATE_FILES = ("obd_v11_frozen.json", "v11_eps_hc.json", "v11_thresh.json")
N_WORKERS = 4
STOP_FRAC = 1.0
DELTA = 0.05

from dsswm.stats import v11_analysis as VA  # noqa: E402

K_GRID, N80_K = VA.K_GRID, VA.N80_K
CODE_FILES = ["run_v11.py", "dsswm/envs/obd_v11_eval.py", "dsswm/envs/obd_v11.py", "dsswm/stats/prereg_v11.py",
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
LOCK_ONLY_FILES = ["build_v11_addendum_draft.py", "dsswm/tests/test_v11_addendum.py"]
EVAL_TASK = "v11_obd_full"
PILOT_TASK = "v11_obd_pilot"


def _rule_task(i):
    return f"v11_dev_rule_e{i}"


TASKS = {_rule_task(i): {"role": "eps_rule", "half": "dev", "eps": float(e), "methods": [VA.RIVAL] + list(VA.HC_NAMES),
                         "seeds": list(VA.DEV_SEEDS)} for i, e in enumerate(VA.EPS_GRID)}
TASKS[PILOT_TASK] = {"role": "dev_cell", "half": "dev", "eps": None, "methods": list(VA.METHODS),
                     "seeds": list(VA.DEV_SEEDS)}
TASKS[EVAL_TASK] = {"role": "confirmatory", "half": "eval", "eps": None, "methods": list(VA.METHODS),
                    "seeds": list(VA.EVAL_SEEDS)}
for _t in TASKS.values():
    _t["layer"] = "OBD-UF0X3-A2-S9"
_ENV: dict = {}


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


def gate_hashes():
    return {f"exp/results/v11_gates/{f}": sha_file(GATES / f) for f in GATE_FILES if (GATES / f).exists()}


def eps_hc():
    if not EPS_HC_JSON.exists():
        raise SystemExit("v11_eps_hc.json missing: run the rule tasks and --select first")
    sel = json.loads(EPS_HC_JSON.read_text())
    if sel["eps_rule"]["selected_eps"] is None:
        raise SystemExit("the eps rule selected no eps: no confirmatory cell (descriptive only)")
    return sel


def frozen_configs(with_thresh=True):
    """Everything the dev cell / eval run needs beyond the design json, read from the gate files."""
    from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
    sel = eps_hc()
    eps = float(sel["eps_rule"]["selected_eps"])
    hc = sel["hc_tuned_at_selected_eps"]
    cfg = {"eps": eps, "delta": DELTA, "K": K_GRID, "n80_k": N80_K, "stop_k": "Q = 15 (stop_frac 1.0)",
           "checkpoints": "fr.checkpoints(n_min = 5000, tau_R = replay N, K = 40)",
           "design": "balanced_alloc(S, A, 0.5) for every method",
           "TU-FDC-DP(b)": {"scheme": "b", "node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO,
                            "split": [0.045, 0.005], "block_points": "the K = 40 grid"},
           "RECT-BF-DP-TU": {"box": True, "split": [0.045, 0.005], "block_points": "the K = 40 grid"},
           "HC-WoR-DP[tuned]": {"schedule": "nstar", "c": float(hc["c"]), "tf": float(hc["tf"]),
                                "source": "v11_eps_hc.json (block-G3 rule at the selected eps, dev seeds 950-999)"},
           "RECT-HG-DP": {"guarantee": "checkpoint-strength (CP): valid at the K grid points only",
                          "alpha_side": "delta / (2 S A K)"}}
    if with_thresh:
        if not THRESH_JSON.exists():
            raise SystemExit("v11_thresh.json missing: run --analyse --dev and --freeze-thresh first")
        th = json.loads(THRESH_JSON.read_text())
        if float(th["eps"]) != eps:
            raise RuntimeError("THRESH gate was computed at a different eps")
        cfg["thresh"] = float(th["thresh"])
    return cfg


def load_frozen(lock=None, with_thresh=True):
    cfg = frozen_configs(with_thresh)
    gates = gate_hashes()
    if lock is not None:
        if lock.get("frozen_configs") != json.loads(json.dumps(cfg)):
            raise RuntimeError("frozen configs differ from the locked v11 addendum")
        if lock.get("frozen_gates") != gates:
            raise RuntimeError("v11 gate files differ from the locked v11 addendum")
        if float(lock["eps"]) != cfg["eps"] or float(lock["thresh"]) != cfg["thresh"]:
            raise RuntimeError("eps / THRESH differ from the locked v11 addendum")
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


def make_env(half, lock=None, eval_task_id=None):
    from dsswm.envs.obd_v11_eval import OBDEnvV11Frozen
    fsha = None if lock is None else lock["frozen_gates"]["exp/results/v11_gates/obd_v11_frozen.json"]
    return OBDEnvV11Frozen(half, eval_task_id=eval_task_id, lock_sha256=None if lock is None else lock["sha256"],
                           frozen_sha256=fsha)


def init_env(T, eps, cfg, lock=None, eval_task_id=None):
    from dsswm.baselines.fdc_dp import make_seg_ctx
    from dsswm.envs.seg_v10 import env_digest, true_opt
    env = make_env(T["half"], lock, eval_task_id)
    assert env.half == T["half"]
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
        out.update({"arrival_digest": cap.digest(), "env_digest": _ENV["digest"], "K_eval": int(len(ctx.checkpoints)),
                    "sec": sec, "rs_sec_cert": {"total": s["sec_cert"], "by_k": s["sec_cert_by_k"]},
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
    """Row binding hash.  Never opens an eval file (external reviewer v11 lock r1 F1): eval-file hashes come from PROVENANCE.json;
    the eval bytes are verified inside the gated, logged reader."""
    from dsswm.envs.obd_v11_eval import data_sha_v11
    return data_sha_v11(None if lock is None else lock["data_sha256"])


def gate_or_skip(task, out, t_start):
    from dsswm.stats.prereg_v11 import addendum_gate
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
    """(eps, cfg) of a task: rule tasks use their grid eps and need no gate files; the dev cell and the eval task use
    the frozen eps / HC config (the eval task also the frozen THRESH, checked against the lock)."""
    T = TASKS[task]
    if T["role"] == "eps_rule":
        from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
        return T["eps"], {VA.PRIMARY: {"node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO}}
    if T["half"] == "eval":
        cfg, _ = load_frozen(lock, with_thresh=True)
    else:
        cfg = frozen_configs(with_thresh=False)
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
        from dsswm.stats.v11_seal import verify_seal
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
def select():
    """HC tuning per eps + eps rule from the rule-task rows only (no FDC-DP row is loaded)."""
    if EPS_HC_JSON.exists():
        raise SystemExit(f"{EPS_HC_JSON} exists: the eps / HC selection is never silently rewritten")
    data_sha = data_sha_for(None)
    rows = []
    for i in range(len(VA.EPS_GRID)):
        t = _rule_task(i)
        rr = current_rows(task_dir(t) / "results.jsonl", None, data_sha)
        if len(rr) != len(TASKS[t]["methods"]) * len(TASKS[t]["seeds"]):
            raise SystemExit(f"{t}: {len(rr)} current rows (incomplete or stale)")
        rows += rr
    hc = VA.select_hc(rows)
    rule = VA.select_eps([r for r in rows if r["method"] == VA.RIVAL or r["method"] in VA.HC_NAMES], hc)
    sel_eps = rule["selected_eps"]
    out = {"written_at": datetime.now().isoformat(), "code_sha256": code_hashes()[1], "data_sha256": data_sha,
           "rule": "smallest eps of the grid at which the dev-best of {RECT-BF-DP-TU, HC-WoR-DP tuned at that eps} "
                   "(lower dev geomean N80_pen; tie -> RECT-BF-DP-TU) has strict N80_pen < tau on >= 40/50 dev "
                   "streams (seeds 950-999); FDC-DP never read",
           "hc_rule": "per eps, lowest dev geomean N80_pen over HC_GRID (nstar; c in {0.5, 0.75}; tf in {0.2, 0.4, "
                      "0.6, 0.8}); ties -> grid order",
           "eps_grid": list(VA.EPS_GRID), "hc_by_eps": hc, "eps_rule": rule,
           "hc_tuned_at_selected_eps": None if sel_eps is None else
           {k: hc[repr(float(sel_eps))][k] for k in ("name", "c", "tf")}}
    GATES.mkdir(parents=True, exist_ok=True)
    EPS_HC_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({"selected_eps": sel_eps, "rival": rule["rival_method"], "successes": rule["successes"],
                      "hc": out["hc_tuned_at_selected_eps"]}))


def freeze_thresh():
    if THRESH_JSON.exists():
        raise SystemExit(f"{THRESH_JSON} exists: THRESH is never silently rewritten")
    a = json.loads((DEV_OUT / "v11_analysis_dev.json").read_text())
    _, code_sha = code_hashes()
    if a["code_sha256_combined"] != code_sha:
        raise SystemExit("dev analysis was produced by different code; re-run --analyse --dev")
    if a["replica_status"] != "pass":
        raise SystemExit("dev replica check did not pass")
    c = a["cell"]["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]
    th = VA.thresh_rule(c["ub95_one_sided"])
    out = {"written_at": datetime.now().isoformat(), "rule": "THRESH = min(0.90, UB95_dev + 0.10), rounded up to 3 "
                                                             "decimals (plan/v11_obd_plan.md s6)",
           "eps": a["cell"]["eps"], "ub95_dev": c["ub95_one_sided"], "ratio_dev": c["geomean_ratio"],
           "n_dev_streams": a["cell"]["n_streams"], "bootstrap": a["cell"]["bootstrap"], "thresh": th,
           "source": "exp/results/pilots/v11_dev/v11_analysis_dev.json", "source_sha256":
               sha_file(DEV_OUT / "v11_analysis_dev.json")}
    THRESH_JSON.write_text(json.dumps(out, indent=1))
    write_manifest()
    print(json.dumps({"ub95_dev": out["ub95_dev"], "thresh": th}))


def write_manifest():
    files = {f: sha_file(GATES / f) for f in GATE_FILES if (GATES / f).exists()}
    MANIFEST_JSON.write_text(json.dumps({"written_at": datetime.now().isoformat(), "sha256": files}, indent=1))


# =============================================================================================== analysis
def analyse(dev):
    from dsswm.envs.obd_v11_eval import eps_opt_share
    from dsswm.envs.seg_v10 import true_opt
    from dsswm.stats import v11_replica as R
    task = PILOT_TASK if dev else EVAL_TASK
    T = TASKS[task]
    lock, add_sha = None, None
    if not dev:
        from dsswm.stats.prereg_v11 import load_locked_addendum
        lock = load_locked_addendum(EVAL_TASK)
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
        from dsswm.stats.v11_seal import verify_seal
        sealed = verify_seal(SEALS, task, cur, T["methods"], (eps,), T["seeds"], per_sha, data_sha,
                             add_sha)["seal"]["results_content_sha256"]
    st = R.validate_report(rep, task, not dev, cur, rep_rows, T["methods"], (eps,), T["seeds"], add_sha, per_sha,
                           data_sha, sealed_content_sha256=sealed)
    cell = VA.analyse_cell(cur, T["seeds"], eps)
    env = make_env(T["half"], lock, None if dev else EVAL_TASK)
    Js, mu = true_opt(env)
    base_ctr = float(np.asarray(env.outcomes_view("visit")).mean())
    dev_base = 0.003504972543167271
    res = {"task": task, "mode": "dev" if dev else "eval", "eps": eps, "addendum_sha256": add_sha,
           "code_sha256": per_sha, "code_sha256_combined": code_sha, "data_sha256": data_sha,
           "frozen_gates": gate_hashes(), "frozen_configs_used": cfg, "replica_status": st, "cell": cell,
           "eps_units": {"absolute_ctr": eps, "dev_base_ctr": dev_base, "relative_to_dev_base": eps / dev_base,
                         "replay_half_base_ctr": base_ctr, "relative_to_replay_base": eps / base_ctr},
           "eps_optimal_share": eps_opt_share(env, Js, mu, eps), "env": env.describe(),
           "Jstar": [float(x) for x in Js], "written_at": datetime.now().isoformat()}
    comp = cell["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]
    if dev:
        res["thresh_if_frozen_now"] = VA.thresh_rule(comp["ub95_one_sided"])
        res["decision_if_applied"] = VA.decide(comp, cell["primary_false_streams"], {"status": st},
                                               res["thresh_if_frozen_now"])
        out = DEV_OUT / "v11_analysis_dev.json"
    else:
        res["decision"] = VA.decide(comp, cell["primary_false_streams"], {"status": st}, float(lock["thresh"]))
        out = EVAL_OUT / "v11_analysis.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"task": task, "eps": eps, "ratio": round(comp["geomean_ratio"], 4),
                      "ub95": round(comp["ub95_one_sided"], 4), "primary_false": cell["primary_false_streams"],
                      "replica": st, "verdict": (res.get("decision") or res.get("decision_if_applied"))["verdict"]}))
    return res


# =============================================================================================== main
def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--replica", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze-design", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--freeze-thresh", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 per process; host rule)")
    args = ap.parse_args()
    N_WORKERS = max(1, min(4, args.workers))
    if args.freeze_design:
        from dsswm.envs.obd_v11_eval import freeze_design
        fz = freeze_design()
        print(json.dumps({k: fz[k] for k in ("S", "A", "cost", "budgets")}))
        return
    if args.select:
        select()
        return
    if args.freeze_thresh:
        freeze_thresh()
        return
    if args.analyse:
        analyse(args.dev)
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
        from dsswm.stats.v11_seal import seal_exists
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
    summary = {"task_id": task, "status": "complete", "role": T["role"], "layer": T["layer"], "half": T["half"],
               "eps": eps, "K": K_GRID, "seeds": [T["seeds"][0], T["seeds"][-1]], "n_streams": len(T["seeds"]),
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
        from dsswm.stats.v11_seal import make_seal, write_and_commit_seal
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
