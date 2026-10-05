"""r5_cr_plugin_{a,b}: confirmatory E-cost evaluation block (FDC + plug-in / asymptotic variants) on CR9 eval streams.

Usage:
  python run_r5_cr_plugin_a.py --task r5_cr_plugin_a --mode full    # eval seeds from lock v5 eval_tasks[task]
  python run_r5_cr_plugin_a.py --task r5_cr_plugin_a --mode pilot   # smoke/timing on DEV seeds 900-909 only

Same protocol and the same block runner as run_r5_cr_main.py (lock gate -> skipped_by_lock on refusal; CR9,
eps* = 0.001, K = 20, stop 12/15; methods / seeds / rival configs from lock v5; write-ahead resumable results.jsonl;
no analysis inside the block). Only the method list (taken from the lock) and the hashed file list differ.
Pilot mode additionally: (i) tests the lock-refusal path on tampered in-memory copies of the lock, (ii) checks that
FDC rows reproduce the r5_cr_main_a pilot rows stream by stream (same code, same dev seeds).
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run: timings biased up).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import run_r5_cr_main as crm  # noqa: E402
from dsswm.stats import prereg  # noqa: E402

RES = crm.RES
crm.FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/combgame_joint.py",
             "dsswm/baselines/rage_frontier.py", "dsswm/baselines/peace_frontier.py", "dsswm/baselines/maq_desc.py",
             "dsswm/baselines/plugin_r5.py", "dsswm/evidence/ext_plugin.py", "dsswm/streams/r5_registry.py",
             "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py", "dsswm/certify/quadknap.py",
             "dsswm/envs/pool_replay.py", "dsswm/stats/prereg.py", "run_r5_cr_main.py", "run_r5_cr_plugin_a.py"]
# FDC-only file subset: its hash must agree with the main block for the FDC reproducibility cross-check
FDC_FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/streams/frontier_runner.py",
             "dsswm/streams/frontier.py", "dsswm/certify/quadknap.py", "dsswm/envs/pool_replay.py"]
PAIR = {"r5_cr_plugin_a": "r5_cr_main_a", "r5_cr_plugin_b": "r5_cr_main_b"}
CMP_KEYS = ["N80_pen", "N80_raw", "completed", "fwer_event", "cert_k", "billing_ok", "schedule_digest", "n_cert_curve"]


def lock_path_test(task):
    """Refusal path on in-memory tampered copies (no file is modified); real lock must pass."""
    real = prereg.load_lock(prereg.lock_path(5))
    cases = {"real_lock": prereg.lock_gate(task, version=5)[0]}

    def gate(lk):
        try:
            prereg.check_lock(lk, 5, task)
            return {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "reason": str(e)[:300]}
    d = copy.deepcopy(real); d["status"] = "draft"; cases["status_draft"] = gate(d)
    d = copy.deepcopy(real); d["status"] = "locked_no_eval"; cases["locked_no_eval"] = gate(d)
    d = copy.deepcopy(real); d["eval_tasks"][task]["seeds"] = "30000-30001"; cases["hash_tamper"] = gate(d)
    d = copy.deepcopy(real); d["version"] = 4; cases["version_mismatch"] = gate(d)
    drift = prereg.frozen_code_drift(real, CODE / "__nonexistent__")
    cases["code_drift_detected_on_wrong_root"] = bool(drift)
    cases["code_drift_real_root"] = prereg.frozen_code_drift(real)
    ok = cases["real_lock"] and all(not cases[k]["ok"] for k in ("status_draft", "locked_no_eval", "hash_tamper",
                                                                     "version_mismatch")) \
        and cases["code_drift_detected_on_wrong_root"] and not cases["code_drift_real_root"]
    return {"all_pass": bool(ok), "cases": cases}


def fdc_repro(task, out):
    """FDC rows of this block vs the paired main block (same dev/eval seeds) -- endpoint fields must agree exactly."""
    other = RES / ("pilots" if "pilots" in str(out) else "full") / PAIR[task] / "results.jsonl"
    if not other.exists():
        return {"status": "paired_file_missing", "path": str(other)}
    load = lambda p: {x["seed"]: x for x in map(json.loads, p.read_text().splitlines())
                      if x.get("method") == "FDC" and x.get("error") is None}
    a, b = load(out / "results.jsonl"), load(other)
    common = sorted(set(a) & set(b))
    diff = [s for s in common if any(a[s].get(k) != b[s].get(k) for k in CMP_KEYS)]
    return {"paired_task": PAIR[task], "n_common": len(common), "n_mismatch": len(diff), "mismatch_seeds": diff[:20],
            "keys": CMP_KEYS, "fdc_files_sha256": {p: crm.sha_file(CODE / p) for p in FDC_FILES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(PAIR))
    ap.add_argument("--mode", required=True, choices=["pilot", "full"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    task = a.task
    out = Path(a.out) if a.out else RES / ("pilots" if a.mode == "pilot" else "full") / task
    wm = True  # PID / PROGRESS / DONE markers for both modes (the dispatched task is monitored either way)
    if wm:
        (RES / f"{task}.pid").write_text(str(os.getpid()))
    ok, lock = prereg.lock_gate(task, version=5)
    if not ok:
        s = crm.skipped_summary(task, out, lock, a.mode)
        if wm:
            crm.mark_done(task, "success", f"skipped_by_lock: {lock}")
        print(json.dumps(s), flush=True)
        return
    try:
        if a.mode == "full":
            s = crm.run_block(task, "full", out, lock)
        else:
            out.mkdir(parents=True, exist_ok=True)
            lt = lock_path_test(task)
            (out / "lock_path_test.json").write_text(json.dumps(lt, indent=1))
            print(f"lock path test all_pass={lt['all_pass']}", flush=True)
            s = crm.run_block(task, "pilot", out, lock, seeds=crm.DEV_PILOT_SEEDS, half="dev")
        rep = fdc_repro(task, out)
        (out / "fdc_repro_check.json").write_text(json.dumps(rep, indent=1))
        txt = (f"{s['status']} rows={s['n_rows']}/{s['expected_rows']} billing_ok={s['billing_ok_all']} "
               f"fdc_repro={rep.get('n_common')}/{rep.get('n_mismatch')}mismatch")
        if wm:
            crm.mark_done(task, "success" if s["status"] == "complete" else "failed", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        if wm:
            crm.mark_done(task, "failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
