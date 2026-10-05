"""r5_rival_tuning (PILOT == FULL): pre-declared tuning of the rigorous rivals on CR9 dev, seeds 900-949.

Grid (methodology s4.4 + plan/baseline_qualification_r5.md s4 item 1, frozen in dsswm.streams.r5_registry):
  B4-bal  share     in {0.40, 0.45, 0.50}
  Hait-SW p_min x gamma in {0.02, 0.05, 0.10} x {2/3 (Hait), 1 (Neyman)}   (6 points, widened by the qualification)
  B2-rect explore_c in {1.0 (published sqrt(n_s)), 0.5}                     (dsswm/baselines/b2_rect_fe.py)
Setting: CR9 dev half, eps* = 0.001, 15 problems, K = 20, stop at 12/15, delta = 0.05.
Selection: minimise the geometric mean of N80_pen over the 50 streams among configs with 0/50 false streams; ties
-> the first grid item. FDC is not run and no FDC result is read.
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
TASK = "r5_rival_tuning"
RES = WS / "exp/results"
OUT = RES / "pilots" / TASK
GATES = RES / "r5_gates"
SEEDS = list(range(900, 950))
EPS = 0.001
IDENTITY_SEEDS = (900, 901)

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

FILES = ["dsswm/baselines/b4_bal.py", "dsswm/baselines/hait_sw.py", "dsswm/baselines/b2_rect_fe.py",
         "dsswm/baselines/combgame_joint.py", "dsswm/streams/r5_registry.py", "dsswm/streams/frontier_runner.py",
         "dsswm/tests/test_r5_rival_tuning.py", "run_r5_rival_tuning.py"]
_ENV = {}


def sha(p):
    return hashlib.sha256((CODE / p).read_bytes()).hexdigest()[:16]


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def grid():
    out = []
    for kw in reg.TUNING_GRID["B4-bal"]:
        out.append(("B4-bal", kw, f"B4-bal[share={kw['share']:.2f}]"))
    for kw in reg.TUNING_GRID["Hait-SW"]:
        g = "2/3" if abs(kw["gamma"] - 2 / 3) < 1e-12 else f"{kw['gamma']:g}"
        out.append(("Hait-SW", kw, f"Hait-SW[p_min={kw['p_min']:.2f},gamma={g}]"))
    from dsswm.baselines.b2_rect_fe import B2_EXPLORE_GRID
    for c in B2_EXPLORE_GRID:
        lab = "sqrt(n_s)" if c == 1.0 else f"{c:g}sqrt(n_s)"
        out.append(("B2-rect", {"explore_c": c}, f"B2-rect[explore={lab}]"))
    return out


def progress(done, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def init_env():
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", "dev")
    problems = fr.cr_problems("visit")
    ctx = build_ctx(env, problems, EPS)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    _ENV.update(env=env, ctx=ctx, J=J, Js=np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)]))


def job(a):
    base, kw, label, seed = a
    ctx = _ENV["ctx"]
    m = reg.make_method(base, **kw)
    t0 = time.perf_counter()
    s, _ = run_stream(_ENV["env"], m, seed, ctx.problems, EPS, ctx=ctx, J_true=_ENV["J"], J_star=_ENV["Js"],
                      keep_U=False)
    return {"method": base, "config": label, "params": kw, "seed": seed, "validity": m.validity,
            "N80_pen": int(s["N80"]), "N80_over_tau": s["N80"] / ctx.tau_R, "reached_stop": bool(s["reached_stop"]),
            "completed": bool(s["reached_stop"] and not s["fwer_event"]), "censored": bool(s["censored"]),
            "fwer_event": bool(s["fwer_event"]), "n_cert": s["n_cert"], "n_false": s["n_false"],
            "k_stop": s["k_stop"], "cert_k": s["cert_k"], "billing_ok": bool(s["billing_ok"]),
            "schedule_digest": s["schedule_digest"], "sec": round(time.perf_counter() - t0, 3)}


def identity_job(seed):
    from dsswm.baselines.combgame_joint import CombGameJoint
    ctx = _ENV["ctx"]
    out = []
    for m in (CombGameJoint("rect"), reg.make_method("B2-rect", explore_c=1.0)):
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, EPS, ctx=ctx, J_true=_ENV["J"], J_star=_ENV["Js"],
                             keep_U=True)
        out.append((s, rows))
    (s1, r1), (s2, r2) = out
    same = len(r1) == len(r2) and all(x["n_cells"] == y["n_cells"] and x["U"] == y["U"] and x["n_cert"] == y["n_cert"]
                                      for x, y in zip(r1, r2)) and s1["cert_k"] == s2["cert_k"]
    return {"seed": seed, "identical": bool(same), "checkpoints": len(r1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GATES.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = datetime.now()
    (OUT / "start_time.txt").write_text(t_start.isoformat())
    t0 = time.time()
    init_env()
    ctx = _ENV["ctx"]
    sec_env = round(time.time() - t0, 1)
    print(f"env built in {sec_env}s tau_R={ctx.tau_R} K={len(ctx.checkpoints)} stop_k={ctx.stop_k}", flush=True)
    G = grid()
    jobs = [(b, kw, lab, sd) for (b, kw, lab) in G for sd in SEEDS]
    total = len(jobs) + len(IDENTITY_SEEDS)
    progress(0, total)
    rows = []
    rfile = OUT / "results.jsonl"
    rfile.write_text("")
    with get_context("fork").Pool(4) as pool:
        ident = pool.map(identity_job, IDENTITY_SEEDS)
        print("B2-rect explore_c=1.0 vs r4 identity:", ident, flush=True)
        for i, r in enumerate(pool.imap(job, jobs, chunksize=1)):
            rows.append(r)
            with open(rfile, "a") as f:
                f.write(json.dumps(r) + "\n")
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                progress(i + 1 + len(IDENTITY_SEEDS), total, {"runs_done": i + 1})
                print(f"[{i+1}/{len(jobs)}] {r['config']} seed {r['seed']} sec={r['sec']}", flush=True)

    tau = int(ctx.tau_R)
    table = []
    for b, kw, lab in G:
        rr = [x for x in rows if x["config"] == lab]
        logs = np.log([x["N80_pen"] for x in rr])
        fs = sum(x["fwer_event"] for x in rr)
        table.append({"method": b, "config": lab, "params": kw, "streams": len(rr),
                      "geomean_N80_pen": float(np.exp(logs.mean())),
                      "geomean_N80_over_tau": float(np.exp(logs.mean()) / tau),
                      "median_N80_pen": float(np.median([x["N80_pen"] for x in rr])),
                      "mean_log_N80_pen": float(logs.mean()), "sd_log_N80_pen": float(logs.std(ddof=1)),
                      "completed": sum(x["completed"] for x in rr), "censored": sum(x["censored"] for x in rr),
                      "false_streams": int(fs), "fwer_cp_upper": round(cp_upper(fs, len(rr)), 4),
                      "false_certs": int(sum(x["n_false"] for x in rr)),
                      "billing_ok_all": all(x["billing_ok"] for x in rr), "eligible": fs == 0,
                      "sec_per_stream_mean": float(np.mean([x["sec"] for x in rr]))})
    selected = {}
    for b in ("B4-bal", "Hait-SW", "B2-rect"):
        cand = [t for t in table if t["method"] == b]
        elig = [t for t in cand if t["eligible"] and t["billing_ok_all"] and t["streams"] == len(SEEDS)]
        if not elig:
            selected[b] = {"config": None, "params": None, "reason": "no eligible config (all have false streams)"}
            continue
        best = min(elig, key=lambda t: t["geomean_N80_pen"])  # min() keeps the first grid item on ties
        ties = [t["config"] for t in elig if t["geomean_N80_pen"] == best["geomean_N80_pen"]]
        selected[b] = {"config": best["config"], "params": best["params"],
                       "geomean_N80_pen": best["geomean_N80_pen"], "false_streams": best["false_streams"],
                       "tie_with": [c for c in ties if c != best["config"]],
                       "rule": "argmin geomean N80_pen among 0/50-false configs; ties -> first grid item"}
    with open(OUT / "grid_table.csv", "w", newline="") as f:
        keys = ["method", "config", "streams", "geomean_N80_pen", "geomean_N80_over_tau", "median_N80_pen",
                "sd_log_N80_pen", "completed", "censored", "false_streams", "fwer_cp_upper", "false_certs",
                "billing_ok_all", "eligible", "sec_per_stream_mean"]
        w = csv.DictWriter(f, fieldnames=keys + ["selected"])
        w.writeheader()
        for t in table:
            w.writerow({**{k: t[k] for k in keys}, "selected": selected[t["method"]].get("config") == t["config"]})
    hashes = {p: sha(p) for p in FILES}
    frozen = {
        "task_id": TASK, "frozen_at": datetime.now().isoformat(),
        "setting": {"layer": "CR9", "half": "dev", "eps": EPS, "K": len(ctx.checkpoints), "stop_k": int(ctx.stop_k),
                    "delta": ctx.delta, "Q": ctx.Q, "tau_R": tau, "seeds": [SEEDS[0], SEEDS[-1]],
                    "n_streams": len(SEEDS)},
        "selection_rule": "per rival: argmin geometric mean of N80_pen over seeds 900-949 among grid points with 0/50 "
                          "false streams and billing_ok; ties -> first grid item (grid order as listed)",
        "grid": {b: [t["params"] for t in table if t["method"] == b] for b in selected},
        "frozen_configs": {b: selected[b]["params"] for b in selected},
        "selected": selected,
        "untuned_rivals": {"B3-rect": "r4 frozen config", "Peace-rect": "r4 frozen config", "B1": "no free parameter",
                           "B4": "no free parameter", "Molitor-WoR": "no free parameter",
                           "QFC-pool": "no free parameter (r4 ledger)"},
        "grid_table": table,
        "deviations": ["Hait-SW grid has 6 points (p_min x gamma, gamma in {2/3, 1}) instead of the 3 in "
                       "methodology s4.4, per plan/baseline_qualification_r5.md s4 item 1 (more tuning for the "
                       "rival = conservative direction); task_plan pilot sample count (8 configs x 50) is therefore "
                       "11 configs x 50 = 550 runs.",
                       "B2-rect forced-exploration constant implemented as subclass dsswm/baselines/b2_rect_fe.py "
                       "(r4 combgame_joint.py frozen); explore_c=1.0 identity-checked against r4 on CR9 dev seeds "
                       f"{list(IDENTITY_SEEDS)} and on toy populations (unit test)."],
        "fdc_read": False, "eval_seeds_touched": False,
        "b2_identity_check": ident, "code_sha256_16": hashes,
    }
    json.dump(frozen, open(GATES / "rival_configs.json", "w"), indent=1)
    t_end = datetime.now()
    all_cells = all(t["streams"] == len(SEEDS) for t in table)
    bill = all(t["billing_ok_all"] for t in table)
    sel_ok = all(selected[b]["config"] is not None and selected[b]["false_streams"] == 0 for b in selected)
    ident_ok = all(x["identical"] for x in ident)
    passed = all_cells and bill and sel_ok and ident_ok
    summary = {"task_id": TASK, "mode": "pilot (== full: same grid and seeds)", "started_at": t_start.isoformat(),
               "ended_at": t_end.isoformat(), "wall_min": round((t_end - t_start).total_seconds() / 60, 1),
               "sec_env_build": sec_env, "timing_note": "concurrent run (4 workers; other r5 tasks share the host)",
               "runs": len(rows), "configs": len(G), "pass_criteria": {
                   "every_cell_50_streams": all_cells, "billing_ok": bill,
                   "selected_configs_0_of_50_false": sel_ok, "b2_explore_c1_identical_to_r4": ident_ok},
               "go_no_go": "GO" if passed else "NO_GO", "selected": selected, "grid_table": table,
               "rival_configs": str((GATES / "rival_configs.json").relative_to(WS)), "fdc_read": False}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    return summary


def mark_done(status, summary_txt):
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary_txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


if __name__ == "__main__":
    try:
        s = main()
        txt = "; ".join(f"{b}: {v['config']}" for b, v in s["selected"].items())
        mark_done("success" if s["go_no_go"] == "GO" else "failed", f"{s['go_no_go']} | {txt}")
        print(json.dumps({k: s[k] for k in ("go_no_go", "pass_criteria", "selected", "wall_min")}, indent=1))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
