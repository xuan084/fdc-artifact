"""FDC-HG dev runner (v9 plan s1-2).  DEV SEEDS ONLY (report seeds 950-999; asserted), dev halves only; no eval seed, no
eval half, no locked file is touched.  New file; reuses the frozen v8 runner job (run_r5s_v8.job) and the frozen v8
dev-tuned rival configs (exp/results/v8_gates/{cr,x9}_configs.json) unchanged.

Tasks
  toy   FWER on the r4 near-tie toy (population seeds 900-919 x perm 0-9 = 200 streams, eps 0.02):
        FDC-HG, FDC-BF and the invalid NAIVE-joint power control (run_fdc_bet.toy_job).
  cr9   CR9 dev half, seeds 950-999, eps 0.001, stop_k 15.
  x5    X5 RetailHero X9 dev half, seeds 950-999, eps 0.015 / 0.02 / 0.03, stop_k 15.
Methods: FDC-HG, FDC-BF, RECT-ck-HG*, HC-WoR*, RECT-ck-BF, PJC-local* (frozen v8 configs per layer; no tuning here).
Output: exp/results/pilots/fdc_hg/<task>/results.jsonl (resumable).  4 workers, BLAS threads 1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/fdc_hg"
REPORT_SEEDS = list(range(950, 1000))
N_WORKERS = int(os.environ.get("FDC_HG_WORKERS", "4"))

import run_fdc_bet as rfb  # noqa: E402
import run_r5s_v8 as R8  # noqa: E402

METHODS = ["FDC-HG", "FDC-BF", "RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local"]
TASKS = {"cr9": dict(layer="CR9", eps=(0.001,)), "x5": dict(layer="X9", eps=(0.015, 0.02, 0.03))}
_LAST = {}
_orig_make = R8.make
_orig_rfb_make = rfb.make


def _make(name, kw):
    if name == "FDC-HG":
        from dsswm.baselines.fdc_hg import FDCHG
        m = FDCHG()
    else:
        m = _orig_make(name, kw)
    _LAST["m"] = m
    return m


def _rfb_make(name, layer):
    if name == "FDC-HG":
        from dsswm.baselines.fdc_hg import FDCHG
        return FDCHG()
    return _orig_rfb_make(name, layer)


R8.make = _make
rfb.make = _rfb_make


def job(a):
    out = R8.job(a)
    if out.get("error") is None:
        m = _LAST.get("m")
        nck = max(int(out["n_checkpoints_visited"]), 1)
        out["sec_per_ck"] = out["sec"] / nck
        if getattr(m, "timing", None):
            out["cert_sec_max"] = float(max(m.timing))
            out["cert_sec_mean"] = float(sum(m.timing) / len(m.timing))
    for k in [k for k in out if k.startswith("rs_")]:
        out.pop(k)                                       # keep rows small (stream summary duplicated elsewhere)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["toy", "cr9", "x5"])
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()
    t0 = datetime.now()
    d = OUT / args.task
    d.mkdir(parents=True, exist_ok=True)
    if args.task == "toy":
        from scipy import stats
        names = args.methods.split(",") if args.methods else ["FDC-HG", "FDC-BF", "NAIVE-joint"]
        with get_context("fork").Pool(N_WORKERS) as pool:
            rows = [r for ch in pool.imap_unordered(rfb.toy_job, [(n, sd) for n in names for sd in range(900, 920)])
                    for r in ch]
        with open(d / "results.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        table = {}
        for n in names:
            rr = [r for r in rows if r["method"] == n]
            ev = sum(r["fwer_event"] for r in rr)
            table[n] = {"n_streams": len(rr), "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                        "cp_upper95": 1.0 if ev >= len(rr) else float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)),
                        "geomean_N80_over_tau": rfb.geo([r["N80_over_tau"] for r in rr]),
                        "billing_ok_all": all(r["billing_ok"] for r in rr), "validity": rr[0]["validity"]}
        (d / "summary.json").write_text(json.dumps({"task": "toy", "written_at": datetime.now().isoformat(),
                                                    "toy": "r4 near-tie toy, population seeds 900-919 x perm 0-9, "
                                                           "eps 0.02", "table": table}, indent=1))
        print(json.dumps({k: (v["false_streams"], round(v["geomean_N80_over_tau"], 4)) for k, v in table.items()}))
        return
    T = TASKS[args.task]
    names = args.methods.split(",") if args.methods else METHODS
    cfg = R8.load_configs(T["layer"])
    R8.init_env(T["layer"], "dev", tuple(T["eps"]))
    assert R8._ENV["env"].half == "dev"
    assert all(900 <= s <= 999 for s in REPORT_SEEDS)
    rfile = d / "results.jsonl"
    done = set()
    if rfile.exists():
        for line in rfile.read_text().splitlines():
            x = json.loads(line)
            if x.get("error") is None:
                done.add((x["method"], x["seed"], x["eps"]))
    jobs = [(n, R8.method_kw(n, cfg), s, e, 15, "dev", "dev", "dev")
            for n in names for e in T["eps"] for s in REPORT_SEEDS if (n, s, e) not in done]
    jobs.sort(key=lambda j: j[0] not in ("FDC-HG", "HC-WoR"))
    print(f"[{args.task}] {len(jobs)} jobs", flush=True)
    errs = 0
    with get_context("fork").Pool(N_WORKERS) as pool, open(rfile, "a") as f:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            if r.get("error"):
                errs += 1
                with open(d / "errors.log", "a") as fe:
                    fe.write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"[{args.task}] done, {errs} errors, wall {(datetime.now() - t0).total_seconds() / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
