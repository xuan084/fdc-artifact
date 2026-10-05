"""r4_setup_pool_replay: setup + pilot for the round-4 real layer (Hillstrom S=6 / S=16, Lenta S=8 pool replay).

New modules: dsswm/envs/pool_replay.py (pools, 4242 split, pre-generated schedules, billing/exhaustion engine,
truth handle isolation), dsswm/streams/frontier.py (segmentations, 15 / 9 problem sets, exact enum / DP / MITM truth,
trivial-policy guard, checkpoints), dsswm/tests/test_pool_replay.py.

pilot: (1) unit tests (test_pool_replay + test_no_truth_import; full suite reported); (2) 100 dev schedules
(permutation seeds 900-999) per layer {HR6, HR16, LR8}: outcome-invariance of the schedule digest, no pool exhausted
before tau_R, arrival-engine billing identity and agreement with the vectorised replay at all 20 checkpoints;
(3) finite-population truth (J*, pi*, eps-answer sets) on dev and full halves, trivial-policy regret and number of
non-trivial problems per eps; (4) reproduction of the external reviewer round-3 6/16 (and 5/13) segment counts and oracle proxy widths.
Pass: all unit tests pass AND 100/100 schedules outcome-invariant AND 0 billing mismatch AND Lenta pools not exhausted
before tau_R.
full: rerun the unit tests and write code + data sha256.
Usage: run_r4_setup_pool_replay.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
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
TASK = "r4_setup_pool_replay"
PY = sys.executable
CODE_FILES = ["dsswm/envs/pool_replay.py", "dsswm/streams/frontier.py", "dsswm/tests/test_pool_replay.py",
              "run_r4_setup_pool_replay.py"]
DEV_SEEDS = list(range(900, 1000))
LAYERS = ["HR6", "HR16", "LR8"]
PRIMARY = {"HR6": "visit", "HR16": "visit", "LR8": "response_att"}

from dsswm.envs.pool_replay import (DATA_PATHS, PoolReplayEnv, ReplayStream, make_schedule,  # noqa: E402
                                    occurrence_rank, replay_fixed, sha256_file)
from dsswm.streams import frontier as fr  # noqa: E402


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
    for name, target in [("pool_replay", "dsswm/tests/test_pool_replay.py"),
                         ("no_truth_import", "dsswm/tests/test_no_truth_import.py"),
                         ("full_suite", "dsswm/tests")]:
        t = time.time()
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", target], cwd=HERE,
                           capture_output=True, text=True, timeout=1800)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        out[name] = {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}
        if r.returncode != 0:
            out[name]["failures"] = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))][:30]
    return out


# ---------------------------------------------------------------------------------------- schedule checks
def check_schedules(args):
    layer, seeds = args
    env = PoolReplayEnv(layer, "dev")
    outcome = PRIMARY[layer]
    ck = env.checkpoints()
    sizes = env.pool_sizes.ravel()
    rows = []
    for seed in seeds:
        t0 = time.time()
        sch = make_schedule(env, seed)
        sch_perm = make_schedule(env.with_permuted_outcomes(10_000 + seed), seed)
        invariant = sch.digest() == sch_perm.digest()
        cell = sch.seg_seq.astype(np.int64) * env.A + sch.arm_seq.astype(np.int64)
        rank = occurrence_rank(cell)
        exhausted_before_tau = int((rank >= sizes[cell]).sum())
        # first arrival at which any pool runs dry (= tau_R expected only if some pool's last record is the last arrival)
        last_pull = np.zeros(len(sizes), dtype=np.int64)
        np.maximum.at(last_pull, cell, np.arange(1, env.N + 1))
        first_dry = int(last_pull.min())
        n, s1, _ = replay_fixed(env, sch, outcome, ck)
        # arrival-by-arrival engine with the common billing rules
        st = ReplayStream(env, sch, outcome)
        cnt = np.zeros(len(sizes), dtype=np.int64)
        tot = np.zeros(len(sizes))
        k, mism = 0, 0
        while st.t < st.tau_R:
            s, a, y = st.arrive()
            if a is None:
                mism += 1
                continue
            cnt[s * env.A + a] += 1
            tot[s * env.A + a] += y
            if k < len(ck) and st.t == ck[k]:
                if not (np.array_equal(cnt, n[k]) and np.allclose(tot, s1[k])) or st.billed != ck[k]:
                    mism += 1
                k += 1
        billing_ok = st.billing_ok() and st.billed == env.tau_R and st.n_skipped == 0 and k == len(ck)
        rows.append({"layer": layer, "seed": seed, "digest": sch.digest()[:16], "outcome_invariant": invariant,
                     "exhausted_before_tau": exhausted_before_tau, "first_pool_dry_at": first_dry,
                     "billing_ok": bool(billing_ok), "billing_mismatch": int(mism + (0 if billing_ok else 1)),
                     "n_reselected": st.n_reselected, "final_counts_match_pools": bool(np.array_equal(n[-1], sizes)),
                     "sec": round(time.time() - t0, 3)})
    return rows


def adaptive_exhaustion_probe(layer, seeds):
    """Exercise exhaustion re-selection with an always-arm-1 adaptive rule; check determinism + billing."""
    env = PoolReplayEnv(layer, "dev")
    out = []
    for seed in seeds:
        logs = []
        for _ in range(2):
            st = ReplayStream(env, make_schedule(env, seed, adaptive=True), PRIMARY[layer])
            first_resel = None
            log_hash = 0
            while st.t < st.tau_R:
                s, a, y = st.arrive(lambda s, avail: 1 if 1 in avail else avail[0])
                if a != 1 and first_resel is None:
                    first_resel = st.t
                log_hash = (log_hash * 1_000_003 + (s * 7 + (a if a is not None else 5)) * 2 + int(y or 0)) % (2**61 - 1)
            logs.append((log_hash, st.billing_ok(), st.n_skipped, first_resel, st.billed))
        out.append({"layer": layer, "seed": seed, "deterministic": logs[0] == logs[1], "billing_ok": logs[0][1],
                    "n_skipped": logs[0][2], "first_reselect_at": logs[0][3], "billed": logs[0][4]})
    return out


# ---------------------------------------------------------------------------------------- truth / tables
def cell_rows(env):
    rows = []
    for s in range(env.S):
        for a in range(env.A):
            r = {"layer": env.layer, "half": env.half, "seg": s, "seg_desc": env.seg_desc[s], "w_s": env.w[s], "arm": a,
                 "n": int(env.pool_sizes[s, a])}
            for o in env._y:
                r[f"mean_{o}"] = float(env.true_mu(o)[s, a])
            rows.append(r)
    return rows


def truth_and_trivial(env, outcome, problems):
    ans = env.truth_answers(problems)
    mu, pooled = env.true_mu(outcome), env.true_pooled_mu(outcome)
    rows = []
    for p, a in zip(problems, ans):
        r = {"layer": env.layer, "half": env.half, "outcome": outcome, "qid": p.qid, "kappa": list(p.kappa),
             "budget": p.budget, "J_star": a.J_star, "pi_star": "-".join(map(str, a.pi_star)), "cost_star": a.cost_star,
             "n_feasible": a.n_feasible, "second_gap": a.second_gap, "dp_agrees": a.dp_agrees,
             **{f"eps_set_{e}": a.eps_set_size[e] for e in fr.EPS_GRID}}
        triv = {"pooled_greedy": fr.trivial_pooled_greedy(env.w, pooled, p),
                "all_arm1": fr.trivial_all_arm(env.w, p, 1)}
        for name, pi in triv.items():
            v = fr.policy_values(np.array([pi]), env.w, mu)[0]
            r[f"regret_{name}"] = float(a.J_star - v)
            r[f"pi_{name}"] = "-".join(map(str, pi))
        r["regret_best_trivial"] = min(r["regret_pooled_greedy"], r["regret_all_arm1"])
        rows.append(r)
    # non-trivial = BOTH simple policies have true regret > eps (conservative); per-policy counts reported alongside
    nontriv = {str(e): int(sum(r["regret_best_trivial"] > e for r in rows)) for e in fr.EPS_GRID}
    for name in ("pooled_greedy", "all_arm1"):
        nontriv[f"only_{name}"] = {str(e): int(sum(r[f"regret_{name}"] > e for r in rows)) for e in fr.EPS_GRID}
    nontriv["max_regret_best_trivial"] = float(max(r["regret_best_trivial"] for r in rows))
    return rows, nontriv


def reviewer_proxy_widths():
    """Re-run the synthesiser's oracle-variance proxy (idea/r4_synth_shape_check.py formula) with the dev segment
    definition; external reviewer round-3 table: 6 seg radsum 0.0662 / quad 0.0243, 16 seg 0.1268 / 0.0369 (visit, N=64000)."""
    out = {}
    for layer in ("HR6", "HR16"):
        env = PoolReplayEnv(layer, "full")
        mu = env.true_mu("visit")
        S, A, K, delta, N = env.S, 3, 20, 0.05, 64000
        rs = V = bmax = 0.0
        for s in range(S):
            ws = env.w[s]
            n = N * ws / A
            v = mu[s] * (1 - mu[s])
            L0 = np.log(2 * S * A * K / delta)
            r = np.sqrt(2 * v * L0 / n) + 7 * L0 / (3 * (n - 1))
            idx = np.argsort(-v)[:2]
            rs += ws * r[idx].sum()
            V += ws ** 2 * v[idx].sum() / n
            bmax = max(bmax, ws / (n - 1))
        L1 = 2 * S * np.log(A) + np.log(K / delta)
        out[layer] = {"radsum": round(float(rs), 4), "quad": round(float(np.sqrt(2 * V * L1) + 7 * L1 * bmax / 3), 4)}
    ref = {"HR6": {"radsum": 0.0662, "quad": 0.0243}, "HR16": {"radsum": 0.1268, "quad": 0.0369}}
    ok = all(abs(out[k][m] - ref[k][m]) <= 5e-4 for k in ref for m in ref[k])
    return {"ours": out, "reviewer_round3": ref, "reproduced": ok}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "started_at": datetime.now().isoformat(),
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)"}
    progress(0, 5, "tests")
    summary["tests"] = run_tests()
    summary["code_sha256"] = {f: sha256_file(HERE / f) for f in CODE_FILES}
    summary["data_sha256"] = {k: sha256_file(v) for k, v in DATA_PATHS.items()}
    tests_ok = all(summary["tests"][k]["returncode"] == 0 for k in ("pool_replay", "no_truth_import"))

    if args.mode == "full":
        summary["pass"] = tests_ok
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mark_done("success" if tests_ok else "failed", f"full: tests_ok={tests_ok}")
        return

    # ---------------- (2) schedules
    progress(1, 5, "schedules")
    jobs = [(layer, DEV_SEEDS[i::args.workers]) for layer in LAYERS for i in range(args.workers)]
    sched_rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for rows in ex.map(check_schedules, jobs):
            sched_rows += rows
    sched_rows.sort(key=lambda r: (r["layer"], r["seed"]))
    write_csv(out_dir / "schedule_checks.csv", sched_rows)
    sched = {}
    for layer in LAYERS:
        rr = [r for r in sched_rows if r["layer"] == layer]
        sched[layer] = {"n_schedules": len(rr), "outcome_invariant": sum(r["outcome_invariant"] for r in rr),
                        "billing_mismatch": sum(r["billing_mismatch"] for r in rr),
                        "exhausted_before_tau": sum(r["exhausted_before_tau"] for r in rr),
                        "final_counts_match_pools": sum(r["final_counts_match_pools"] for r in rr),
                        "first_pool_dry_at_min": min(r["first_pool_dry_at"] for r in rr),
                        "first_pool_dry_frac_min": min(r["first_pool_dry_at"] for r in rr) /
                        PoolReplayEnv(layer, "dev").tau_R,
                        "distinct_digests": len({r["digest"] for r in rr}),
                        "sec_per_schedule_mean": float(np.mean([r["sec"] for r in rr]))}
    summary["schedules"] = sched
    progress(2, 5, "adaptive_probe")
    summary["adaptive_exhaustion_probe"] = adaptive_exhaustion_probe("HR6", [900, 901, 902]) + \
        adaptive_exhaustion_probe("LR8", [900])

    # ---------------- (3) truth + trivial guard + cell tables
    progress(3, 5, "truth")
    cells, triv_rows, nontriv = [], [], {}
    envs = {}
    for layer in LAYERS:
        for half in ("dev", "full"):
            env = PoolReplayEnv(layer, half)
            envs[(layer, half)] = env
            cells += cell_rows(env)
            fams = [("visit", fr.hr_problems("visit")), ("conversion", fr.hr_problems("conversion"))] \
                if layer.startswith("HR") else [("response_att", fr.lr_problems("response_att"))]
            for outcome, probs in fams:
                rows, nt = truth_and_trivial(env, outcome, probs)
                triv_rows += rows
                nontriv[f"{layer}|{half}|{outcome}"] = nt
    write_csv(out_dir / "cell_tables.csv", cells)
    write_csv(out_dir / "trivial_policy_regret.csv", triv_rows)
    summary["n_nontrivial_problems_by_eps"] = nontriv
    summary["dp_agrees_all"] = all(r["dp_agrees"] for r in triv_rows)
    summary["layers"] = {f"{l}|{h}": {"N": e.N, "S": e.S, "A": e.A, "tau_R": e.tau_R, "min_pool": int(e.pool_sizes.min()),
                                      "checkpoints": e.checkpoints().tolist(), "clip": e.clip}
                         for (l, h), e in envs.items()}
    summary["lenta_control_share_dev"] = float(envs[("LR8", "dev")].pool_sizes[:, 0].sum() / envs[("LR8", "dev")].N)
    summary["lenta_age_edges"] = list(fr.lenta_age_edges(
        __import__("dsswm.envs.pool_replay", fromlist=["x"]).load_table("lenta").x_age.to_numpy()))

    # ---------------- (4) external reviewer reproduction
    progress(4, 5, "reviewer_repro")
    hr16 = envs[("HR16", "full")]
    summary["reviewer_repro"] = {"n_seg_S6": envs[("HR6", "full")].S, "n_seg_S16": hr16.S,
                              "min_cell_S16_full": int(hr16.pool_sizes.min()), **reviewer_proxy_widths()}

    # ---------------- samples
    env = envs[("HR6", "dev")]
    for seed in DEV_SEEDS[:5]:
        sch = make_schedule(env, seed)
        n, s1, _ = replay_fixed(env, sch, "visit", env.checkpoints())
        (out_dir / "samples" / f"HR6_dev_seed{seed}.json").write_text(json.dumps({
            "seed": seed, "digest": sch.digest(), "first_40_arrivals": [[int(s), int(a)] for s, a in
                                                                        zip(sch.seg_seq[:40], sch.arm_seq[:40])],
            "checkpoints": env.checkpoints().tolist(), "cell_counts_at_checkpoints": n.tolist(),
            "cell_means_at_checkpoints": np.where(n > 0, s1 / np.maximum(n, 1), np.nan).round(5).tolist()}))

    # ---------------- verdict
    crit = {
        "unit_tests_pass": tests_ok,
        "schedules_outcome_invariant_100": all(sched[l]["outcome_invariant"] == 100 for l in LAYERS),
        "billing_mismatch_0": all(sched[l]["billing_mismatch"] == 0 for l in LAYERS) and
        all(r["billing_ok"] and r["deterministic"] for r in summary["adaptive_exhaustion_probe"]),
        "lenta_not_exhausted_before_tauR": sched["LR8"]["exhausted_before_tau"] == 0,
    }
    summary["pass_criteria"] = crit
    summary["pass"] = all(crit.values())
    summary["go_no_go"] = "GO" if summary["pass"] else "NO_GO"
    summary["wall_min"] = round((time.time() - t_start) / 60, 2)
    summary["finished_at"] = datetime.now().isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    progress(5, 5, "done", {"pass": summary["pass"]})
    mark_done("success" if summary["pass"] else "failed", f"pilot pass={summary['pass']} criteria={crit}")
    print(json.dumps({"pass": summary["pass"], "criteria": crit, "schedules": sched,
                      "nontrivial": nontriv, "reviewer": summary["reviewer_repro"], "tests": summary["tests"]}, indent=1,
                     default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
