"""r3_replicate_hh1: replication of HH1 (honest abstention: NL Type-2 + E1-Lin OOS) on the round-3 eval seeds.

Runs the UNCHANGED round-2 science code (run_hh1_honesty_oos.py); the only change is the seed source (and output
location), passed through its round-3 CLI overrides (--seeds / --noise-seeds / --out-dir / --no-protocol).

Pilot (smoke, dev seeds 734-739, no eval seed touched):
  OOS scan 734-739 (squeeze + newpart, K=10 each), keep <= 30 OOS cases (<= 12 newpart), Lin std Type-2 734-735
  x 5 problems x {probes_legal, probes_illegal}, NL-S Type-2 734-739 x 5 variants; noise 42.
  pass: 0 crashes AND 0 wrong labels.
Full (requires the v3 lock with status 'locked'; block read from the lock's eval_manifest / seed_blocks 'hd2_hh1'):
  eval 10630-10669 for all three blocks, noise 42/123/456, all kept OOS cases.
  pass: wrong-label CP95 upper <= delta AND Lin OOS (non-near-tie) detection >= 0.9; decidable rate reported.

Usage: run_r3_replicate_hh1.py --mode {pilot,full} [--workers 4]
Concurrent run (4 slots share 20 cores + 1 RTX 4090): timings are "concurrent".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "r3_replicate_hh1"
PY = sys.executable
DELTA = 0.05


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                       "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def seed_plan(mode):
    if mode == "pilot":
        return {"seeds": "734-739", "std_seeds": "734-735", "noise": "42", "n_oos": 30, "n_newpart": 12,
                "source": "task_plan pilot.dev_seeds 734-739 (dev block 720-760)"}
    lock = json.loads(LOCK.read_text())
    assert lock.get("version") == 3 and lock.get("status") == "locked", "full mode requires the locked v3 prereg"
    man = lock.get("eval_manifest") or {}
    spec = lock["seed_blocks"]["eval"]["hd2_hh1"]
    if isinstance(man, dict) and isinstance(man.get("hd2_hh1"), str) and "-" in man["hd2_hh1"]:
        spec = man["hd2_hh1"]
    a, b = (int(x) for x in str(spec).split("-"))
    assert (a, b) == (10630, 10669), spec
    return {"seeds": f"{a}-{b}", "std_seeds": f"{a}-{b}", "noise": "42,123,456", "n_oos": 10 ** 6,
            "n_newpart": 10 ** 6, "source": "plan/prereg_lock.json v3 (locked) seed_blocks.eval.hd2_hh1 / eval_manifest",
            "lock_sha256": lock.get("sha256")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    out = RES_ROOT / ("pilots" if pilot else "full") / TASK
    out.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    (out / "start_time.txt").write_text(datetime.now().isoformat())
    T0 = time.time()
    summ = {"task_id": TASK, "mode": args.mode, "seed": 42, "concurrent_run": True,
            "note": "并发运行（4 槽并行，每任务 4 worker）；计时偏高", "eval_seeds_touched": not pilot}
    try:
        sp = seed_plan(args.mode)
        summ["seed_plan"] = sp
        progress(1, 2, "hh1", {"seeds": sp["seeds"]})
        cmd = [PY, "run_hh1_honesty_oos.py", "--mode", args.mode, "--workers", str(args.workers),
               "--seeds", sp["seeds"], "--std-seeds", sp["std_seeds"], "--noise-seeds", sp["noise"],
               "--n-oos", str(sp["n_oos"]), "--n-newpart", str(sp["n_newpart"]),
               "--out-dir", str(out), "--no-protocol"]
        with open(out / "hh1_stdout.log", "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
        summ["hh1_rc"] = rc
        s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        (out / "hh1_summary.json").write_text(json.dumps(s, indent=1, default=float))
        a = (s.get("JPC") or {}).get("all") or {}
        fam = (s.get("JPC") or {}).get("Lin_OOS_family") or {}
        n_err = int((s.get("meta") or {}).get("n_errors") or s.get("n_errors") or 0)
        crashes = int(rc != 0) + n_err
        summ["hh1"] = {"gate_round2": s.get("gate"), "go_no_go_round2_rule": s.get("go_no_go"),
                       "JPC_all": a, "JPC_by_block": {k: v for k, v in (s.get("JPC") or {}).items() if k != "all"},
                       "baselines": s.get("baselines"),
                       "n_true_non_near_tie_oos_triggered": s.get("n_true_non_near_tie_oos_triggered"),
                       "misclassified_or_undecided": s.get("misclassified_or_undecided")}
        det = fam.get("oos_detection_rate_non_near_tie")
        pc = {"zero_crashes": crashes == 0, "zero_wrong_labels" if pilot else "wrong_label_cp_upper_le_delta":
              (a.get("wrong_label") == 0) if pilot else bool(a.get("wrong_label_cp") and a["wrong_label_cp"][1] <= DELTA)}
        if not pilot:
            pc["lin_oos_detection_ge_0.9"] = det is not None and det >= 0.9
        summ["reported"] = {"wrong_label": a.get("wrong_label"), "n": a.get("n"),
                            "wrong_label_cp": a.get("wrong_label_cp"), "lin_oos_detection_non_near_tie": det,
                            "decidable_within_budget": a.get("decidable_within_budget"),
                            "decidable_cp": a.get("decidable_cp")}
        summ["crashes"] = crashes
        summ["pass_criteria"] = pc
        summ["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summ["wall_clock_s"] = time.time() - T0
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        progress(2, 2, "done", {"go_no_go": summ["go_no_go"]})
        mark_done("success" if crashes == 0 else "failed", f"{args.mode}: {summ['go_no_go']} {pc}")
        print(json.dumps({"go_no_go": summ["go_no_go"], "pass_criteria": pc, "reported": summ["reported"]}, indent=1))
    except Exception as e:  # noqa: BLE001
        summ["error"] = repr(e)
        summ["traceback"] = traceback.format_exc()
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
