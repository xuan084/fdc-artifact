"""r4_setup_criteo_replay: real Criteo Uplift v2.1 pool-replay layers CR9 (primary) / CR12 (scale) + HR8 (secondary).

pilot: (1) unit tests (test_criteo_replay + test_pool_replay + test_no_truth_import; full suite reported);
(2) 100 CR9 dev schedules (permutation seeds 900-999): outcome-invariance of the schedule digest, no pool exhausted
before tau_R, vectorised 20-checkpoint replay, arrival-engine billing identity vs. the vectorised counts (full length
for FULL_ENGINE_SEEDS, first PREFIX_K checkpoints for the rest), timing of schedule generation + 20-checkpoint counts;
same checks for CR12 dev (prefix engine) and HR8 dev (full engine); adaptive exhaustion probes (CR9 always-control,
HR8 always-arm-1); (3) finite-population truth (J*, pi*, eps-answer sets, enum == DP; CR9 enum == brute force) and
trivial-policy guard on the dev AND eval halves; (4) layer table + comparison with the planner's diagnostic.
Pass: all old+new tests pass AND 100/100 CR9 schedules outcome-invariant AND billing mismatch 0 AND CR9 enum == brute
force AND guard tables written for D and the eval half.
full: rerun tests and write code + data sha256 for the lock.
Usage: run_r4_setup_criteo_replay.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
TASK = "r4_setup_criteo_replay"
PY = sys.executable
CODE_FILES = ["dsswm/envs/pool_replay.py", "dsswm/streams/frontier.py", "dsswm/tests/test_criteo_replay.py",
              "run_r4_setup_criteo_replay.py"]
DEV_SEEDS = list(range(900, 1000))
FULL_ENGINE_SEEDS = set(range(900, 912))      # 12 full-length arrival-engine replays on CR9 (3 per worker)
PREFIX_K = 13                                 # other seeds: engine checked through the first 13 checkpoints (~1.47M)
PLANNER_DIAG = WS / "plan" / "feasibility_r4" / "criteo_visit.jsonl"

from dsswm.envs.pool_replay import (DATA_PATHS, PoolReplayEnv, ReplayStream, make_schedule,  # noqa: E402
                                    occurrence_rank, replay_fixed, sha256_file)
from dsswm.streams import frontier as fr  # noqa: E402

ENVS: dict = {}      # built in the parent before forking (copy-on-write in the workers)


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                       "final_progress": fp,
                                                       "timestamp": datetime.now().isoformat()}))


def run_tests():
    out = {}
    for name, target in [("criteo_replay", "dsswm/tests/test_criteo_replay.py"),
                         ("pool_replay", "dsswm/tests/test_pool_replay.py"),
                         ("no_truth_import", "dsswm/tests/test_no_truth_import.py"),
                         ("full_suite", "dsswm/tests")]:
        t = time.time()
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", target], cwd=HERE,
                           capture_output=True, text=True, timeout=2400)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        out[name] = {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}
        if r.returncode != 0:
            out[name]["failures"] = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))][:30]
    return out


# ---------------------------------------------------------------------------------------- schedule checks
def check_schedules(args):
    layer, seeds, engine_mode = args
    env = ENVS[(layer, "dev")]
    outcome = "visit"
    ck = env.checkpoints()
    sizes = env.pool_sizes.ravel()
    A = env.A
    rows = []
    for seed in seeds:
        t0 = time.time()
        sch = make_schedule(env, seed)
        n, s1, _ = replay_fixed(env, sch, outcome, ck)
        t_sched = time.time() - t0
        sch_perm = make_schedule(env.with_permuted_outcomes(10_000 + seed), seed)
        invariant = sch.digest() == sch_perm.digest()
        del sch_perm
        cell = sch.seg_seq.astype(np.int64) * A + sch.arm_seq.astype(np.int64)
        rank = occurrence_rank(cell)
        exhausted_before_tau = int((rank >= sizes[cell]).sum())
        last_pull = np.zeros(len(sizes), dtype=np.int64)
        np.maximum.at(last_pull, cell, np.arange(1, env.N + 1))
        first_dry = int(last_pull.min())
        # pool-proportional check: realised arm share per segment at checkpoint 1 vs pool share
        share_dev = float(np.max(np.abs(n[0].reshape(env.S, A) / np.maximum(n[0].reshape(env.S, A).sum(1, keepdims=True), 1)
                                        - env.pool_sizes / env.pool_sizes.sum(1, keepdims=True))))
        full_engine = engine_mode == "full" or (engine_mode == "mixed" and seed in FULL_ENGINE_SEEDS)
        k_stop = len(ck) if full_engine else PREFIX_K
        t_stop = int(ck[k_stop - 1])
        t1 = time.time()
        st = ReplayStream(env, sch, outcome)
        cnt = np.zeros(len(sizes), dtype=np.int64)
        tot = np.zeros(len(sizes))
        k, mism = 0, 0
        arrive = st.arrive
        while st.t < t_stop:
            s, a, y = arrive()
            if a is None:
                mism += 1
                continue
            c = s * A + a
            cnt[c] += 1
            tot[c] += y
            if st.t == ck[k]:
                if not (np.array_equal(cnt, n[k]) and np.allclose(tot, s1[k])) or st.billed != ck[k]:
                    mism += 1
                k += 1
        billing_ok = st.billing_ok() and st.billed == t_stop and st.n_skipped == 0 and k == k_stop
        rows.append({"layer": layer, "seed": seed, "digest": sch.digest()[:16], "outcome_invariant": invariant,
                     "exhausted_before_tau": exhausted_before_tau, "first_pool_dry_at": first_dry,
                     "first_pool_dry_frac": first_dry / env.tau_R, "engine_checked_to": t_stop,
                     "engine_full_length": full_engine, "engine_checkpoints_checked": k,
                     "billing_ok": bool(billing_ok), "billing_mismatch": int(mism + (0 if billing_ok else 1)),
                     "n_reselected": st.n_reselected, "final_counts_match_pools": bool(np.array_equal(n[-1], sizes)),
                     "max_arm_share_dev_ck1": share_dev,
                     "sec_schedule_plus_20ck": round(t_sched, 3), "sec_engine": round(time.time() - t1, 2)})
        del sch, cell, rank, st
    return rows


def adaptive_probe(args):
    """Adaptive rule that always asks for one arm -> exhaustion + re-selection; run twice for determinism."""
    layer, seed, arm, frac = args
    env = ENVS[(layer, "dev")]
    stop = int(frac * env.tau_R)
    logs = []
    for _ in range(2):
        st = ReplayStream(env, make_schedule(env, seed, adaptive=True), "visit")
        first_other, h = None, 0
        while st.t < stop:
            s, a, y = st.arrive(lambda s, avail: arm if arm in avail else avail[0])
            if a != arm and first_other is None:
                first_other = st.t
            h = (h * 1_000_003 + (s * 7 + (a if a is not None else 5)) * 2 + int(y or 0)) % (2 ** 61 - 1)
        logs.append((h, st.billing_ok(), st.n_skipped, first_other, st.billed,
                     int((st.remaining[:, arm] == 0).sum())))
    return {"layer": layer, "seed": seed, "always_arm": arm, "arrivals": stop, "deterministic": logs[0] == logs[1],
            "billing_ok": bool(logs[0][1]) and logs[0][4] == stop, "n_skipped": logs[0][2],
            "first_other_arm_at": logs[0][3], "first_other_arm_frac": (logs[0][3] or 0) / env.tau_R,
            "n_segments_arm_exhausted": logs[0][5]}


# ---------------------------------------------------------------------------------------- truth / guard
def brute_force_check(env, problems, outcome="visit"):
    mu = env.true_mu(outcome)
    ans = env.truth_answers(problems)
    pols = list(itertools.product(range(env.A), repeat=env.S))
    bad = 0
    for p, a in zip(problems, ans):
        best, arg, nf = -np.inf, None, 0
        vals = []
        for pi in pols:
            cost = sum(env.w[s] * p.kappa[x] for s, x in enumerate(pi))
            if cost <= p.budget + fr.FEAS_TOL:
                v = sum(env.w[s] * mu[s, x] for s, x in enumerate(pi))
                vals.append(v)
                nf += 1
                if v > best:
                    best, arg = v, pi
        ok = (abs(best - a.J_star) <= 1e-12 and tuple(arg) == a.pi_star and nf == a.n_feasible and
              all(a.eps_set_size[e] == sum(v >= best - e for v in vals) for e in env.eps_grid))
        bad += (not ok)
    return {"n_problems": len(problems), "n_mismatch": bad, "n_policies": len(pols)}


def problems_for(layer, outcome):
    return fr.cr_problems(outcome) if layer.startswith("CR") else fr.hr_problems(outcome)


def truth_and_guard(env, outcome):
    probs = problems_for(env.layer, outcome)
    ans = env.truth_answers(probs)
    mu, pooled = env.true_mu(outcome), env.true_pooled_mu(outcome)
    diag_grid = fr.CR_EPS_DIAG if env.layer.startswith("CR") else fr.EPS_GRID
    truth_rows, guard_rows = [], []
    for p, a in zip(probs, ans):
        truth_rows.append({"layer": env.layer, "half": env.half, "outcome": outcome, "qid": p.qid,
                           "kappa": list(p.kappa), "budget": p.budget, "J_star": a.J_star,
                           "pi_star": "-".join(map(str, a.pi_star)), "cost_star": a.cost_star,
                           "n_feasible": a.n_feasible, "second_gap": a.second_gap, "dp_agrees": a.dp_agrees,
                           "dp_pi_equal": tuple(a.dp_pi) == a.pi_star,
                           **{f"eps_set_{e}": a.eps_set_size[e] for e in env.eps_grid}})
        if env.layer.startswith("CR"):
            triv = {"treat_by_w_desc": fr.trivial_all_arm(env.w, p, 1),
                    "pooled_uplift_sign": fr.trivial_pooled_greedy(env.w, pooled, p)}
        else:
            triv = {"pooled_greedy": fr.trivial_pooled_greedy(env.w, pooled, p),
                    "all_arm1": fr.trivial_all_arm(env.w, p, 1)}
        g = {"layer": env.layer, "half": env.half, "outcome": outcome, "qid": p.qid, "budget": p.budget,
             "J_star": a.J_star}
        for name, pi in triv.items():
            v = fr.policy_values(np.array([pi]), env.w, mu)[0]
            g[f"regret_{name}"] = float(a.J_star - v)
            g[f"pi_{name}"] = "-".join(map(str, pi))
        g["min_trivial_regret"] = min(g[f"regret_{n}"] for n in triv)
        for e in diag_grid:
            g[f"nontrivial_eps_{e}"] = bool(g["min_trivial_regret"] > e)
        guard_rows.append(g)
    nontriv = {str(e): int(sum(r["min_trivial_regret"] > e for r in guard_rows)) for e in diag_grid}
    out = {"layer": env.layer, "half": env.half, "outcome": outcome, "n_problems": len(probs),
           "eps_grid_locked": list(env.eps_grid), "n_nontrivial_by_eps": nontriv,
           "max_min_trivial_regret": float(max(r["min_trivial_regret"] for r in guard_rows)),
           "min_min_trivial_regret": float(min(r["min_trivial_regret"] for r in guard_rows)),
           "dp_agrees_all": all(r["dp_agrees"] for r in truth_rows),
           "guard_ge5_all_locked_eps": all(nontriv[str(e)] >= 5 for e in env.eps_grid)}
    return truth_rows, guard_rows, out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def summarize_sched(rows, env):
    return {"n_schedules": len(rows), "outcome_invariant": sum(r["outcome_invariant"] for r in rows),
            "billing_mismatch": sum(r["billing_mismatch"] for r in rows),
            "exhausted_before_tau": sum(r["exhausted_before_tau"] for r in rows),
            "final_counts_match_pools": sum(r["final_counts_match_pools"] for r in rows),
            "n_full_length_engine": sum(r["engine_full_length"] for r in rows),
            "engine_arrivals_checked_total": int(sum(r["engine_checked_to"] for r in rows)),
            "first_pool_dry_frac_min": min(r["first_pool_dry_frac"] for r in rows),
            "distinct_digests": len({r["digest"] for r in rows}),
            "max_arm_share_dev_ck1": max(r["max_arm_share_dev_ck1"] for r in rows),
            "sec_schedule_plus_20ck_mean": float(np.mean([r["sec_schedule_plus_20ck"] for r in rows])),
            "sec_schedule_plus_20ck_max": float(np.max([r["sec_schedule_plus_20ck"] for r in rows])),
            "sec_engine_full_mean": float(np.mean([r["sec_engine"] for r in rows if r["engine_full_length"]] or [0])),
            "tau_R": env.tau_R}


def planner_comparison(guard):
    """Our frozen-split guard vs the planner's arm-only-stratified diagnostic (expected close, not identical)."""
    out = {}
    if not PLANNER_DIAG.exists():
        return {"available": False}
    for ln in PLANNER_DIAG.read_text().splitlines():
        r = json.loads(ln)
        lay = {9: "CR9", 12: "CR12"}.get(r.get("S"))
        half = {"dev": "dev", "full": "eval"}.get(r.get("half"))
        if lay is None or (lay, half) in out:
            continue
        ours = guard.get(f"{lay}|{half}|visit")
        if ours is None:
            continue
        out[f"{lay}|{half}"] = {"planner_maxtriv": r["maxtriv"], "ours_max_min_trivial_regret":
                                round(ours["max_min_trivial_regret"], 5),
                                "planner_nontriv": r["nontriv"], "ours_nontriv": ours["n_nontrivial_by_eps"],
                                "planner_N": r["N"], "planner_minpool": r["minpool"]}
    out["note"] = ("planner diag split = arm-only stratified (seed 4242), half 'full' there = complement half; ours = "
                   "frozen joint-cell stratified split. Differences are split noise only.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "started_at": datetime.now().isoformat(),
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)"}
    progress(0, 6, "tests")
    summary["tests"] = run_tests()
    summary["code_sha256"] = {f: sha256_file(HERE / f) for f in CODE_FILES}
    summary["data_sha256"] = {"criteo_tidy_pkl": sha256_file(DATA_PATHS["criteo"]),
                              "criteo_csv_gz": sha256_file(DATA_PATHS["criteo"].parent /
                                                           "criteo-research-uplift-v2.1.csv.gz"),
                              "hillstrom_tidy_pkl": sha256_file(DATA_PATHS["hillstrom"])}
    tests_ok = all(v["returncode"] == 0 for v in summary["tests"].values())

    if args.mode == "full":
        summary["pass"] = tests_ok
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mark_done("success" if tests_ok else "failed", f"full: tests_ok={tests_ok}")
        return

    # ---------------- build envs once (parent), workers fork and share them
    progress(1, 6, "envs")
    t = time.time()
    for layer in ("CR9", "CR12", "HR8"):
        for half in ("dev", "eval"):
            ENVS[(layer, half)] = PoolReplayEnv(layer, half)
    summary["sec_build_envs"] = round(time.time() - t, 1)
    cr9d = ENVS[("CR9", "dev")]
    summary["segment_descriptions"] = {l: ENVS[(l, "dev")].seg_desc for l in ("CR9", "CR12", "HR8")}
    from dsswm.envs.pool_replay import load_table
    df = load_table("criteo")
    modes = fr.criteo_modes(df)
    _, _, med = fr.seg_cr12(df, modes)
    summary["criteo_segment_constants"] = {
        "modes_full_table": {k: float(v) for k, v in modes.items()},
        "nonmode_share_full_table": {f: float((df[f].to_numpy() != modes[f]).mean()) for f in modes},
        "duplicates_identical": {d: bool(np.array_equal(df[d].to_numpy() != modes[d], df[b].to_numpy() != modes[b]))
                                 for d, b in fr.CR_DUPLICATES.items()},
        "cr12_nonmode_medians": med,
        "cr12_refines_cr9": False,
        "split_strata": "joint (CR12 x CR9) cell x arm, seed 4242 (CR12 does not refine CR9: f3 not in CR12)"}
    del df

    # ---------------- (2) schedules
    progress(2, 6, "schedules")
    ctx = mp.get_context("fork")
    jobs = [("CR9", DEV_SEEDS[i::args.workers], "mixed") for i in range(args.workers)]
    jobs += [("CR12", DEV_SEEDS[i::args.workers], "prefix") for i in range(args.workers)]
    jobs += [("HR8", DEV_SEEDS[i::args.workers], "full") for i in range(args.workers)]
    probes = [("CR9", 900, 0, 0.20), ("CR9", 901, 0, 0.20), ("HR8", 900, 1, 1.0), ("HR8", 901, 1, 1.0),
              ("HR8", 902, 2, 1.0)]
    sched_rows = []
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(check_schedules, j) for j in jobs]
        pfuts = [ex.submit(adaptive_probe, p) for p in probes]
        for i, f in enumerate(futs):
            sched_rows += f.result()
            progress(2, 6, "schedules", {"jobs_done": i + 1, "jobs": len(futs)})
        probe_rows = [f.result() for f in pfuts]
    sched_rows.sort(key=lambda r: (r["layer"], r["seed"]))
    write_csv(out_dir / "schedule_checks.csv", sched_rows)
    sched = {l: summarize_sched([r for r in sched_rows if r["layer"] == l], ENVS[(l, "dev")])
             for l in ("CR9", "CR12", "HR8")}
    summary["schedules"] = sched
    summary["adaptive_exhaustion_probe"] = probe_rows

    # ---------------- (3) truth + guard on dev and eval halves
    progress(3, 6, "truth")
    truth_rows, guard_rows, guard = [], [], {}
    for layer in ("CR9", "CR12", "HR8"):
        for half in ("dev", "eval"):
            env = ENVS[(layer, half)]
            for outcome in ("visit", "conversion"):
                tr, gr, g = truth_and_guard(env, outcome)
                truth_rows += tr
                guard_rows += gr
                guard[f"{layer}|{half}|{outcome}"] = g
    write_csv(out_dir / "truth_tables.csv", truth_rows)
    write_csv(out_dir / "trivial_guard.csv", guard_rows)
    summary["guard"] = guard
    progress(4, 6, "brute_force")
    summary["cr9_enum_vs_brute_force"] = {h: brute_force_check(ENVS[("CR9", h)], fr.cr_problems())
                                          for h in ("dev", "eval")}
    summary["planner_diag_comparison"] = planner_comparison(guard)

    # cell tables
    cells = []
    for (layer, half), env in ENVS.items():
        for s in range(env.S):
            for a in range(env.A):
                r = {"layer": layer, "half": half, "seg": s, "seg_desc": env.seg_desc[s], "w_s": env.w[s], "arm": a,
                     "n": int(env.pool_sizes[s, a])}
                for o in ("visit", "conversion"):
                    r[f"mean_{o}"] = float(env.true_mu(o)[s, a])
                cells.append(r)
    write_csv(out_dir / "cell_tables.csv", cells)

    # ---------------- (4) layer table (paper visualization)
    layer_table = []
    for (layer, half), env in ENVS.items():
        g = guard[f"{layer}|{half}|visit"]
        layer_table.append({"layer": layer, "half": half, "dataset": "Criteo v2.1" if layer.startswith("CR")
                            else "Hillstrom", "rows": env.N, "S": env.S, "A": env.A, "|Pi|": env.A ** env.S,
                            "min_pool": int(env.pool_sizes.min()), "tau_R": env.tau_R, "n_min": env.n_min,
                            "replan_interval": env.replan_interval,
                            "checkpoints": env.checkpoints().tolist(),
                            "nontrivial_visit_by_eps": g["n_nontrivial_by_eps"],
                            "max_min_trivial_regret": g["max_min_trivial_regret"]})
    summary["layer_table"] = layer_table
    write_csv(out_dir / "layer_table.csv", layer_table)

    # ---------------- samples
    progress(5, 6, "samples")
    ck = cr9d.checkpoints()
    for seed in DEV_SEEDS[:5]:
        sch = make_schedule(cr9d, seed)
        n, s1, _ = replay_fixed(cr9d, sch, "visit", ck)
        (out_dir / "samples" / f"CR9_dev_seed{seed}.json").write_text(json.dumps({
            "seed": seed, "digest": sch.digest(),
            "first_40_arrivals_seg_arm": [[int(s), int(a)] for s, a in zip(sch.seg_seq[:40], sch.arm_seq[:40])],
            "checkpoints": ck.tolist(), "cell_counts_at_checkpoints": n.tolist(),
            "cell_visit_means_at_checkpoints": np.where(n > 0, s1 / np.maximum(n, 1), np.nan).round(5).tolist()}))

    # ---------------- verdict
    bf = summary["cr9_enum_vs_brute_force"]
    crit = {
        "unit_tests_pass": tests_ok,
        "cr9_schedules_outcome_invariant_100": sched["CR9"]["outcome_invariant"] == 100 == sched["CR9"]["n_schedules"],
        "billing_mismatch_0": all(sched[l]["billing_mismatch"] == 0 for l in sched) and
        all(r["billing_ok"] and r["deterministic"] for r in probe_rows),
        "no_pool_exhausted_before_tauR": all(sched[l]["exhausted_before_tau"] == 0 for l in sched),
        "cr9_enum_equals_brute_force": all(v["n_mismatch"] == 0 for v in bf.values()),
        "guard_tables_written_dev_and_eval": (out_dir / "trivial_guard.csv").exists() and
        all(f"CR9|{h}|visit" in guard for h in ("dev", "eval")),
    }
    summary["pass_criteria"] = crit
    summary["pass"] = all(crit.values())
    summary["go_no_go"] = "GO" if summary["pass"] else "NO_GO"
    summary["informative"] = {
        "cr9_guard_ge5_both_halves_all_locked_eps": all(guard[f"CR9|{h}|visit"]["guard_ge5_all_locked_eps"]
                                                         for h in ("dev", "eval")),
        "hr8_guard_ge5_both_halves_all_locked_eps": all(guard[f"HR8|{h}|visit"]["guard_ge5_all_locked_eps"]
                                                         for h in ("dev", "eval")),
    }
    summary["wall_min"] = round((time.time() - t_start) / 60, 2)
    summary["finished_at"] = datetime.now().isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    progress(6, 6, "done", {"pass": summary["pass"]})
    mark_done("success" if summary["pass"] else "failed", f"pilot pass={summary['pass']} criteria={crit}")
    print(json.dumps({"pass": summary["pass"], "criteria": crit, "schedules": sched, "probes": probe_rows,
                      "guard": {k: v["n_nontrivial_by_eps"] for k, v in guard.items()},
                      "brute": bf, "tests": summary["tests"], "planner": summary["planner_diag_comparison"]},
                     indent=1, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
