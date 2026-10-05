"""r5_fdc_impl (pilot): FDCMethod implementation checks on CR9 dev streams (identity / ledger / r4-path freeze).

  A  ledger on the CR9 dev ctx (beta, x_v, sum_q |Pi_{B_q}|, K, S*A) and all 8 factorial cells;
  B  FDC == FDCAblation('half', 'feas', 'r5') on dev streams 900-909 (N80, cert_k, decided_pi, digest, billing);
  C  FDCAblation('pool', 'qstar', 'r4') == live r4 QFCMethod() == stored r4 G1-shape QFC streams on 900-999;
  D  r4 code path frozen: git diff of the r4 modules against the lock-v4 code commit 6297e7f2 is empty;
  E  unit tests (tests/test_r5_fdc.py with FDC_R4_IDENTITY_ALL=1) and the full test suite.
Descriptive only (no rival comparison, no gate): FDC N80 on the same 10 streams is recorded next to QFC.
CPU only, <= 4 workers, BLAS threads = 1. Only CR9 dev half, dev seeds 900-999.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp" / "results"
OUT = RES / "pilots" / "r5_fdc_impl"
TASK = "r5_fdc_impl"
R4_COMMIT = "6297e7f2"
R4_FROZEN = ["certify/quadknap.py", "baselines/frontier_common.py", "streams/frontier_runner.py",
             "envs/pool_replay.py", "theory_checks/mc_l1.py", "streams/frontier.py"]
PY = sys.executable

from dsswm.baselines.fdc import FACTORIAL_CELLS, FDCAblation, FDCMethod  # noqa: E402
from dsswm.baselines.frontier_common import QFCMethod  # noqa: E402
from dsswm.envs.pool_replay import PoolReplayEnv  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

G = {}
FULL_LOG = None


def progress(step, total, phase):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase}, "updated_at": datetime.now().isoformat()}))


def init():
    env = PoolReplayEnv("CR9", "dev")
    eps = float(json.loads((RES / "r4_gates" / "eps.json").read_text())["eps_star"])
    ctx = build_ctx(env, fr.cr_problems("visit"), eps)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    G.update(env=env, ctx=ctx, J=J, Js=Js)


def make(name):
    return {"FDC": FDCMethod, "FDCx-half-feas-r5": lambda: FDCAblation("half", "feas", "r5"),
            "QFC": QFCMethod, "FDCx-pool-qstar-r4": lambda: FDCAblation("pool", "qstar", "r4")}[name]()


def job(a):
    name, seed = a
    try:
        m = make(name)
        s, _ = run_stream(G["env"], m, seed, G["ctx"].problems, G["ctx"].eps, ctx=G["ctx"], J_true=G["J"],
                          J_star=G["Js"], keep_U=False)
        s["tag"] = name
        return s, None
    except Exception:  # noqa: BLE001
        return {"tag": name, "perm_seed": seed}, traceback.format_exc()


KEYS = ("N80", "cert_k", "decided_pi", "schedule_digest", "n_false", "billed", "served", "skipped", "reselected")


def cmp(a, b):
    return all(a[k] == b[k] for k in KEYS if k in a and k in b)


def run(cmd, env=None, timeout=3000):
    t = time.time()
    p = subprocess.run(cmd, cwd=str(CODE), capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr, round(time.time() - t, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t0 = time.time()
    summ = {"task_id": TASK, "mode": "pilot", "seed": 42, "started_at": datetime.now().isoformat(),
            "timing_note": "concurrent run (<=4 workers on a shared 20-core host)",
            "touched": "CR9 dev half, dev seeds 900-999 only; no evaluation seed, no rival comparison"}
    status = "success"
    try:
        progress(0, 5, "ledger")
        init()
        ctx = G["ctx"]
        cells = {}
        for c in FACTORIAL_CELLS:
            a = FDCAblation(*c)
            a.setup(ctx)
            cells["/".join(c)] = {"name": a.name, "validity": a.validity, "L1": a.params["L1"], "x_v": a.params["x_v"],
                                  "union_size": a.params["union_size"], "C_var": a.params["C_var"],
                                  "alloc_kind": a.alloc_kind}
        m = FDCMethod()
        m.setup(ctx)
        led = dict(m.ledger_info)
        summ["ledger"] = {"S": ctx.S, "A": ctx.A, "P": int(ctx.P), "Q": ctx.Q, "eps": ctx.eps, "delta": ctx.delta,
                          "checkpoints": ctx.checkpoints.tolist(), **led,
                          "beta_4dp": round(led["beta"], 4), "x_v_4dp": round(led["x_v"], 4),
                          "matches_spec": bool(round(led["beta"], 4) == 14.2242 and round(led["x_v"], 4) == 11.8776
                                               and led["union_size"] == 3386 and led["C_var"] == 18)}
        summ["factorial_cells"] = cells

        progress(1, 5, "streams")
        jobs = [(n, s) for s in range(900, 910) for n in ("FDC", "FDCx-half-feas-r5")]
        jobs += [(n, s) for s in range(900, 1000) for n in ("QFC", "FDCx-pool-qstar-r4")]
        with Pool(4, initializer=init) as pool:
            res = pool.map(job, jobs, chunksize=2)
        errs = [e for _, e in res if e]
        if errs:
            (OUT / "errors.log").write_text("\n\n".join(errs))
            raise RuntimeError(f"{len(errs)} stream jobs failed")
        by = {(s["tag"], s["perm_seed"]): s for s, _ in res}
        with open(OUT / "streams.jsonl", "w") as f:
            for s, _ in res:
                f.write(json.dumps(s) + "\n")
        stored = {}
        for l in (RES / "pilots" / "r4_g1_shape_a" / "streams.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["method"] == "QFC" and r.get("layer") == "CR9":
                stored[r["perm_seed"]] = r
        fdc_id = [cmp(by[("FDC", s)], by[("FDCx-half-feas-r5", s)]) for s in range(900, 910)]
        r4_live = [cmp(by[("FDCx-pool-qstar-r4", s)], by[("QFC", s)]) for s in range(900, 1000)]
        r4_stored = [cmp(by[("FDCx-pool-qstar-r4", s)], stored[s]) for s in range(900, 1000)]
        r4_n80 = [by[("FDCx-pool-qstar-r4", s)]["N80"] == stored[s]["N80"] for s in range(900, 1000)]
        fdc_n80 = np.array([by[("FDC", s)]["N80"] for s in range(900, 910)], dtype=float)
        qfc_n80 = np.array([by[("QFC", s)]["N80"] for s in range(900, 910)], dtype=float)
        summ["fdc_identity"] = {"streams": "900-909", "n": 10, "identical": int(sum(fdc_id)),
                                "N80_FDC": fdc_n80.astype(int).tolist(),
                                "n_false_FDC": int(sum(by[("FDC", s)]["n_false"] for s in range(900, 910))),
                                "skipped_FDC": int(sum(by[("FDC", s)]["skipped"] for s in range(900, 910))),
                                "reselected_FDC": [by[("FDC", s)]["reselected"] for s in range(900, 910)]}
        summ["r4_identity"] = {"streams": "900-999", "n": 100, "N80_identical_vs_stored": int(sum(r4_n80)),
                               "all_fields_identical_vs_stored": int(sum(r4_stored)),
                               "all_fields_identical_vs_live_QFC": int(sum(r4_live)),
                               "fields": list(KEYS),
                               "stored_source": "exp/results/pilots/r4_g1_shape_a/streams.jsonl"}
        summ["descriptive_only_not_a_gate"] = {
            "note": "FDC vs r4 QFC on dev 900-909, implementation sanity only (T1/T2 are separate tasks)",
            "geomean_N80_FDC": float(np.exp(np.log(fdc_n80).mean())),
            "geomean_N80_QFC": float(np.exp(np.log(qfc_n80).mean())),
            "geomean_ratio_FDC_over_QFC": float(np.exp(np.log(fdc_n80 / qfc_n80).mean())),
            "mean_sec_cert_FDC": float(np.mean([by[("FDC", s)]["sec_cert"] for s in range(900, 910)]))}
        samples = [{k: by[("FDC", s)][k] for k in ("perm_seed", "N80", "k_stop", "cert_k", "decided_pi",
                                                   "n_false", "reselected", "schedule_digest")} for s in (900, 901, 902)]
        samples += [{"tag": "QFC/FDCx-pool-qstar-r4", **{k: by[("QFC", s)][k] for k in ("perm_seed", "N80", "cert_k")},
                     "stored_N80": stored[s]["N80"]} for s in (900, 950, 999)]
        (OUT / "samples.json").write_text(json.dumps(samples, indent=1))

        progress(2, 5, "r4_freeze")
        rc, out, _ = run(["git", "diff", "--stat", R4_COMMIT, "--", *[f"dsswm/{p}" for p in R4_FROZEN]])
        rc2, st, _ = run(["git", "status", "--porcelain", "--", *[f"dsswm/{p}" for p in R4_FROZEN]])
        summ["r4_freeze"] = {"commit": R4_COMMIT, "files": R4_FROZEN, "diff_empty": bool(rc == 0 and not out.strip()),
                             "worktree_clean": bool(rc2 == 0 and not st.strip())}

        progress(3, 5, "unit_tests")
        envv = dict(os.environ, FDC_R4_IDENTITY_ALL="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        rc, out, sec = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "dsswm/tests/test_r5_fdc.py"], env=envv)
        (OUT / "pytest_r5_fdc.log").write_text(out)
        summ["unit_tests"] = {"file": "dsswm/tests/test_r5_fdc.py", "FDC_R4_IDENTITY_ALL": 1, "exit": rc,
                              "tail": out.strip().splitlines()[-1] if out.strip() else "", "sec": sec}
        progress(4, 5, "full_suite")
        if FULL_LOG is not None:        # reuse a completed full-suite run of the same tree (saves ~45 min)
            while "EXIT " not in FULL_LOG.read_text():
                time.sleep(30)
            out = FULL_LOG.read_text()
            rc = int(out.strip().splitlines()[-1].split()[-1])
            tail = [l for l in out.strip().splitlines() if " passed" in l or " failed" in l]
            summ["full_suite"] = {"exit": rc, "tail": tail[-1] if tail else "", "log": str(FULL_LOG.relative_to(WS)),
                                  "note": "full suite run once on the committed tree; test_r5_fdc.py also run above"}
        else:
            rc, out, sec = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "dsswm/tests"],
                               env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
            (OUT / "pytest_full.log").write_text(out)
            summ["full_suite"] = {"exit": rc, "tail": out.strip().splitlines()[-1] if out.strip() else "", "sec": sec}

        ok = {"ledger": summ["ledger"]["matches_spec"], "fdc_identity_10_10": summ["fdc_identity"]["identical"] == 10,
              "r4_identity_100_100": summ["r4_identity"]["N80_identical_vs_stored"] == 100
              and summ["r4_identity"]["all_fields_identical_vs_live_QFC"] == 100,
              "r4_freeze": summ["r4_freeze"]["diff_empty"] and summ["r4_freeze"]["worktree_clean"],
              "unit_tests": summ["unit_tests"]["exit"] == 0, "full_suite": summ["full_suite"]["exit"] == 0,
              "factorial_all_rigorous": all(c["validity"] == "rigorous" for c in cells.values())}
        summ["checks"] = ok
        summ["verdict"] = "GO" if all(ok.values()) else "NO_GO"
    except Exception:  # noqa: BLE001
        status = "failed"
        summ["error"] = traceback.format_exc()
        summ["verdict"] = "NO_GO"
    summ["sec_total"] = round(time.time() - t0, 1)
    summ["finished_at"] = datetime.now().isoformat()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    progress(5, 5, "done")
    print(json.dumps({k: summ.get(k) for k in ("verdict", "checks", "sec_total")}, indent=1))
    return status


if __name__ == "__main__":
    if "--full-suite-log" in sys.argv:
        FULL_LOG = Path(sys.argv[sys.argv.index("--full-suite-log") + 1]).resolve()
    main()
