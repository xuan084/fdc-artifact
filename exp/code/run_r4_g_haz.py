"""r4_g_haz: Gate G-haz (MS2 precondition) + freeze of Guard c and the MisLid-ms inflation multiple (descriptive).

Methodology §3 / F4 (r4 FCC pre-lock revision 2026-10-02):
  r3 JPC (locked r3 version, via the r4 harness dsswm.streams.ms_r4) on
     MS-H dev instances 800-859 (truth-label hazard caliber eta_near/eps >= 1) and MS-S dev instances 860-899 (static
     twin g = 1), ONE stream per instance (noise seed 42), 15 problems, eps = 0.02, delta = 0.05, T_max = 6000.
  Per layer: S1 = FWER one-sided CP lower bound, S2 = FCR_old one-sided instance-cluster bootstrap lower bound
     (B = 1e4, seed 42). Reproduced iff FWER CP lower > delta OR FCR_old lower > delta (stats.acceptance_r4).
  MS-H not reproduced -> ONE retry with the r3 mu_flip primary caliber (ex-ante bin 2), same dev seeds 800-859
     (both attempts recorded).
  Gate fail -> ms2_status = dead_hazard_invalid (MS2 withdrawn; no pivot; CR9 unaffected).
  Guard c: smallest c in {1.5, 2, 3} with pooled (100 instances) FWER one-sided CP upper <= 0.05; evaluated in
     increasing c order, a larger c is only run if every smaller c fails (identical frozen value, less compute).
  MisLid-ms inflation multiple kappa in {1, 2}: same rule (smallest kappa with FWER CP upper <= 0.05). Diagnostic
     subset (downscaled, see summary.downscale): MS-H 800-804 + MS-S 860-864, first MISLID_PROBLEMS problems.

Usage: run_r4_g_haz.py [--workers 4] [--mode pilot]
Concurrent run: <= 4 CPU workers, BLAS threads = 1; GPU (shared 4090) only for placements / class tables.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES = WS / "exp" / "results"
TASK = "r4_g_haz"
PLANNED_MIN = 60
MSH = list(range(800, 860))
MSS = list(range(860, 900))
GUARD_CS = (1.5, 2.0, 3.0)
MISLID = {1.0: "MisLid-ms", 2.0: "MisLid-ms-k2"}
MISLID_SEEDS = {"MS-H": MSH[:5], "MS-S": MSS[:5]}
MISLID_PROBLEMS = 3
SETUP_PLC = RES / "pilots" / "r4_setup_ms_baselines" / "placements"
CODE_FILES = ["dsswm/streams/ms_r4.py", "dsswm/baselines/ms_common.py", "dsswm/baselines/guard.py",
              "dsswm/baselines/mislid_ms.py", "dsswm/baselines/switched_nl.py", "dsswm/evidence/reuse_switch.py",
              "dsswm/stats/acceptance_r4.py", "dsswm/stats/cp.py", "run_r4_g_haz.py", "run_r3_hazard.py"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(step, total, phase, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
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
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def update_gpu_progress(status, start_iso, wall_min, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    with open(WS / "exp" / "gpu_progress.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        other = "failed" if key == "completed" else "completed"
        if TASK in d.get(other, []):
            d[other].remove(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": PLANNED_MIN, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def update_shared_summary(entry, md):
    with open(RES / "pilot_summary.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            pj = RES / "pilot_summary.json"
            d = json.loads(pj.read_text()) if pj.exists() else {"tasks": {}}
            d.setdefault("tasks", {})[TASK] = entry
            tmp = pj.with_name(pj.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))
            os.replace(tmp, pj)
            pm = RES / "pilot_summary.md"
            txt = pm.read_text() if pm.exists() else ""
            head = f"## {TASK} "
            if head in txt:
                s = txt.index(head)
                e = txt.find("\n## ", s + 1)
                txt = txt[:s] + md.strip() + "\n" + (txt[e:] if e >= 0 else "")
            else:
                txt = txt.rstrip() + "\n\n" + md.strip() + "\n"
            pm.write_text(txt)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


# ============================================================================================ worker
def _job(layer, seed, methods, plc, out_dir, tag, n_problems):
    """Run every method on one instance; each (instance, method) stream is saved as its own part (resumable)."""
    import torch
    torch.set_num_threads(1)
    from dsswm.streams import ms_r4 as M
    parts = out_dir / "parts"
    todo = [m for m in methods if not (parts / f"{tag}_{layer}_i{seed}_{m}.json").exists()]
    if not todo:
        return {"tag": tag, "layer": layer, "seed": seed, "skipped": True}
    t0 = time.perf_counter()
    if layer == "MS-H":
        inst = M.build_msh(seed, plc)
    else:
        inst = M.build_mss(seed)
    build = time.perf_counter() - t0
    for m in todo:
        t1 = time.perf_counter()
        try:
            rows, info = M.run_stream(inst, m, n_problems=n_problems,
                                      decision_path=out_dir / "decisions" / f"{tag}_{layer}_i{seed}_{m}.jsonl")
            rec = {"tag": tag, "layer": layer, "seed": seed, "method": m, "rows": rows, "info": info,
                   "build_sec": build, "sec": time.perf_counter() - t1, "error": None}
        except Exception as e:  # noqa: BLE001
            rec = {"tag": tag, "layer": layer, "seed": seed, "method": m, "rows": [], "info": None,
                   "error": repr(e), "tb": traceback.format_exc()[-3000:]}
        fn = parts / f"{tag}_{layer}_i{seed}_{m}.json"
        tmp = fn.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec, default=str))
        os.replace(tmp, fn)
    return {"tag": tag, "layer": layer, "seed": seed, "skipped": False, "sec": time.perf_counter() - t0}


def run_pool(jobs, out_dir, workers, phase, step, total):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if not jobs:
        return
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_job, *j[:4], out_dir, *j[4:]): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
                msg = f"{r.get('sec', 0):.0f}s" if not r.get("skipped") else "cached"
            except Exception as e:  # noqa: BLE001
                msg = f"FATAL {e!r}"
            done += 1
            progress(step, total, phase, {"done": done, "total": len(jobs), "last": f"{j[4]} {j[0]} i{j[1]}"})
            log(f"[{phase}] {j[4]} {j[0]} i{j[1]} {j[2]} {msg} ({done}/{len(jobs)})")


def load_parts(out_dir, tag):
    recs = []
    for fn in sorted((out_dir / "parts").glob(f"{tag}_*.json")):
        recs.append(json.loads(fn.read_text()))
    return recs


def per_stream(recs, method, layers=None):
    out = []
    for r in recs:
        if r["method"] != method or (layers and r["layer"] not in layers):
            continue
        rows = r["rows"]
        out.append({"layer": r["layer"], "seed": r["seed"], "n_problems": len(rows),
                    "n_cert": int(sum(x["status"] == "CERTIFIED" for x in rows)),
                    "n_false": int(sum(x["false_cert"] for x in rows)),
                    "n_correct": int(sum(x["correct_cert"] for x in rows)),
                    "steps": int(sum(x["new_env_steps"] for x in rows)), "error": r.get("error")})
    return sorted(out, key=lambda x: (x["layer"], x["seed"]))


def safety_block(streams):
    from dsswm.stats.acceptance_r4 import jpc_failure_reproduced, layer_safety
    nf = [s["n_false"] for s in streams]
    nc = [s["n_cert"] for s in streams]
    saf = layer_safety(nf, nc)
    rep = jpc_failure_reproduced(nf, nc)
    npb = sum(s["n_problems"] for s in streams)
    return {"n_streams": len(streams), "n_problems": npb, "n_cert": int(sum(nc)), "n_false": int(sum(nf)),
            "false_streams": int(sum(x > 0 for x in nf)), "completion_mean": float(np.mean(
                [s["n_correct"] / max(s["n_problems"], 1) for s in streams])) if streams else None,
            "cert_rate": float(sum(nc) / npb) if npb else None,
            "S1": saf["S1"], "S2": saf["S2"], "E_V_over_R": saf["E_V_over_R"], "safe": saf["safe"],
            "reproduction": rep, "n_errors": int(sum(s["error"] is not None for s in streams))}


def cell_breakdown(recs, method):
    """Descriptive: certifications / false certifications by gap layer and (MS-H) eta_near/eps bins."""
    tab = defaultdict(lambda: {"n": 0, "cert": 0, "false": 0})
    for r in recs:
        if r["method"] != method:
            continue
        for x in r["rows"]:
            keys = [f"{r['layer']}|gap={x['gap_layer']}"]
            if "eta_near_over_eps" in x:
                e = x["eta_near_over_eps"]
                keys.append(f"{r['layer']}|eta/eps={'[1,1.5)' if e < 1.5 else ('[1.5,2.5)' if e < 2.5 else '>=2.5')}")
            if "mu_flip" in x and r["layer"] == "MS-H":
                keys.append(f"{r['layer']}|mu_flip={'<0.031' if x['mu_flip'] < 0.03125 else ('<0.133' if x['mu_flip'] < 0.1328125 else '>=0.133')}")
            for k in keys:
                tab[k]["n"] += 1
                tab[k]["cert"] += x["status"] == "CERTIFIED"
                tab[k]["false"] += bool(x["false_cert"])
    return {k: {**v, "fcr": v["false"] / v["cert"] if v["cert"] else None} for k, v in sorted(tab.items())}


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    for sub in ("parts", "decisions", "placements", "samples"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "seed": 42, "started": start_iso, "candidate_id": "cand_qfc",
               "dev_seeds": {"MS-H": [MSH[0], MSH[-1]], "MS-S": [MSS[0], MSS[-1]]},
               "constants": {"eps": 0.02, "delta": 0.05, "T_max": 6000, "problems_per_stream": 15, "noise_seed": 42,
                             "streams_per_instance": 1, "bootstrap_B": 10000, "bootstrap_seed": 42},
               "timing_note": "并发运行 (<= 4 CPU workers, BLAS threads = 1; GPU only for MS-H placements and MS-S "
                              "static class tables on the shared 4090)"}
    status = "success"
    try:
        from dsswm.streams import ms_r4 as M
        # ------------------------------------------------------------------ 1. placements (GPU, main process)
        progress(0, 6, "placements")
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        edges_mu = lock["frozen_items"]["r1adv_bin_edges"]["mu_flip"]
        pdir = out_dir / "placements"
        P = None
        msh = {}
        for s in MSH:
            fn = pdir / f"msh_i{s}.json"
            src = SETUP_PLC / f"msh_i{s}.json"
            if fn.exists():
                msh[s] = json.loads(fn.read_text())
            elif src.exists():                         # identical deterministic placement from r4_setup_ms_baselines
                shutil.copy(src, fn)
                msh[s] = json.loads(fn.read_text())
            else:
                if P is None:
                    P = M.MSPlacer(edges_mu)
                t0 = time.perf_counter()
                msh[s] = P.place_msh(s)
                fn.write_text(json.dumps(msh[s]))
                P.drop_seed(s)
                log(f"MS-H i{s}: fill={msh[s]['fill']} draw={msh[s]['draw']} proposals={msh[s]['proposals']} "
                    f"({time.perf_counter() - t0:.1f}s)")
        t0 = time.perf_counter()
        quota = M.prepare_quota(MSS)
        summary["mss_quota"] = {"n_quota_fail": int(sum(v["quota_fail"] for v in quota.values())),
                                "sec": time.perf_counter() - t0}
        fills = np.array([msh[s]["fill"] for s in MSH])
        summary["msh_placement"] = {"n_instances": len(MSH), "n_full_15": int((fills >= M.Q).sum()),
                                    "underfilled": [s for s in MSH if msh[s]["fill"] < M.Q],
                                    "gap_layers": dict(Counter(r["gap_layer"] for s in MSH
                                                               for r in msh[s]["per_problem"])),
                                    "eta_near_over_eps_median": float(np.median(
                                        [r["eta_arg_near_exante"] / M.EPS for s in MSH
                                         for r in msh[s]["per_problem"]])),
                                    "proposals_mean": float(np.mean([msh[s]["proposals"] for s in MSH]))}
        log(f"placements ready: MS-H full15={summary['msh_placement']['n_full_15']}/60, "
            f"MS-S quota_fail={summary['mss_quota']['n_quota_fail']}")
        if P is not None:
            del P
        mss_ok = [s for s in MSS if not quota[s]["quota_fail"]]

        # ------------------------------------------------------------------ 2. JPC + Guard c1.5 (+ MisLid diag)
        progress(1, 6, "jpc_guard15_mislid")
        jobs = [("MS-H", s, ["JPC", "Guard_c1.5"], msh[s], "main", None) for s in MSH]
        jobs += [("MS-S", s, ["JPC", "Guard_c1.5"], None, "main", None) for s in mss_ok]
        jobs += [(l, s, list(MISLID.values()), msh.get(s) if l == "MS-H" else None, "mislid", MISLID_PROBLEMS)
                 for l, ss in MISLID_SEEDS.items() for s in ss if l == "MS-H" or s in mss_ok]
        # longest first (MisLid streams run to T_max) for better packing
        jobs.sort(key=lambda j: 0 if j[4] == "mislid" else 1)
        run_pool(jobs, out_dir, a.workers, "jpc_guard15_mislid", 1, 6)
        main_recs = load_parts(out_dir, "main")

        # ------------------------------------------------------------------ 3. gate verdict per layer (JPC)
        gate = {}
        for layer in ("MS-H", "MS-S"):
            gate[layer] = {"attempt_1": safety_block(per_stream(main_recs, "JPC", [layer]))}
            gate[layer]["attempt_1"]["caliber"] = "eta_near/eps>=1" if layer == "MS-H" else "static twin g=1"
            log(f"G-haz {layer} attempt 1: {json.dumps(gate[layer]['attempt_1']['reproduction'])}")

        # ------------------------------------------------------------------ 4. MS-H retry (mu_flip ante-bin 2)
        progress(2, 6, "msh_retry")
        if not gate["MS-H"]["attempt_1"]["reproduction"]["reproduced"]:
            log("MS-H not reproduced -> retry with r3 mu_flip ante-bin 2 caliber (same dev seeds 800-859)")
            P = M.MSPlacer(edges_mu)
            plc2 = {}
            for s in MSH:
                fn = pdir / f"msh_bin2_i{s}.json"
                if fn.exists():
                    plc2[s] = json.loads(fn.read_text())
                else:
                    plc2[s] = P.place_msh(s, caliber="muflip_bin2")
                    fn.write_text(json.dumps(plc2[s]))
                    P.drop_seed(s)
            del P
            run_pool([("MS-H", s, ["JPC"], plc2[s], "retry", None) for s in MSH], out_dir, a.workers,
                     "msh_retry", 2, 6)
            rr = load_parts(out_dir, "retry")
            gate["MS-H"]["attempt_2"] = safety_block(per_stream(rr, "JPC", ["MS-H"]))
            gate["MS-H"]["attempt_2"]["caliber"] = "r3 mu_flip ante-bin 2 (mu_flip >= 0.1328125)"
            f2 = np.array([plc2[s]["fill"] for s in MSH])
            gate["MS-H"]["attempt_2"]["placement"] = {"n_full_15": int((f2 >= M.Q).sum()),
                                                      "gap_layers": dict(Counter(r["gap_layer"] for s in MSH
                                                                                 for r in plc2[s]["per_problem"]))}
            gate["MS-H"]["attempt_2"]["cells"] = cell_breakdown(rr, "JPC")
            log(f"G-haz MS-H attempt 2: {json.dumps(gate['MS-H']['attempt_2']['reproduction'])}")
        for layer in gate:
            att = [gate[layer][k] for k in ("attempt_1", "attempt_2") if k in gate[layer]]
            gate[layer]["reproduced"] = bool(any(x["reproduction"]["reproduced"] for x in att))
            gate[layer]["reproduced_on"] = next((k for k in ("attempt_1", "attempt_2") if k in gate[layer]
                                                 and gate[layer][k]["reproduction"]["reproduced"]), None)
        g_pass = gate["MS-H"]["reproduced"] and gate["MS-S"]["reproduced"]
        summary["g_haz"] = {"per_layer": gate, "pass": bool(g_pass),
                            "ms2_status": "alive_pending_gfac" if g_pass else "dead_hazard_invalid",
                            "rule": "both MS-H and MS-S: JPC FWER one-sided CP lower > delta OR FCR_old cluster-"
                                    "bootstrap lower > delta; MS-H one retry with r3 mu_flip ante-bin 2",
                            "consequence": "fail withdraws only the multi-step secondary claim MS2; no pivot; CR9 "
                                           "main pillar unaffected"}
        summary["jpc_cells_attempt_1"] = cell_breakdown(main_recs, "JPC")

        # ------------------------------------------------------------------ 5. Guard c freeze (increasing c)
        progress(3, 6, "guard_freeze")
        guard = {}
        frozen_c = None
        for c in GUARD_CS:
            m = f"Guard_c{c:g}"
            if c != GUARD_CS[0]:
                run_pool([("MS-H", s, [m], msh[s], "main", None) for s in MSH] +
                         [("MS-S", s, [m], None, "main", None) for s in mss_ok], out_dir, a.workers,
                         f"guard_{m}", 3, 6)
                main_recs = load_parts(out_dir, "main")
            st = per_stream(main_recs, m)
            pooled = safety_block(st)
            guard[m] = {"pooled_100": {k: pooled[k] for k in ("n_streams", "n_cert", "n_false", "false_streams",
                                                               "completion_mean", "cert_rate", "S1", "S2",
                                                               "E_V_over_R")},
                        "per_layer": {l: {k: v for k, v in safety_block([x for x in st if x["layer"] == l]).items()
                                          if k != "reproduction"} for l in ("MS-H", "MS-S")},
                        "fwer_cp_upper_le_0.05": bool(pooled["S1"]["cp_upper"] <= 0.05)}
            log(f"{m}: false_streams={pooled['false_streams']}/{pooled['n_streams']} "
                f"cp_upper={pooled['S1']['cp_upper']:.4f} completion={pooled['completion_mean']:.3f}")
            if guard[m]["fwer_cp_upper_le_0.05"]:
                frozen_c = c
                break
        summary["guard"] = {"evaluated": guard, "frozen_c": frozen_c,
                            "rule": "smallest c in {1.5,2,3} with pooled FWER one-sided CP upper <= 0.05 (100 dev "
                                    "instances); larger c evaluated only if every smaller c fails",
                            "fallback": None if frozen_c is not None else "no c passes -> c = 3 (largest) frozen, "
                                                                           "flagged"}
        if frozen_c is None:
            summary["guard"]["frozen_c"] = GUARD_CS[-1]
            summary["guard"]["no_c_passes"] = True

        # ------------------------------------------------------------------ 6. MisLid-ms kappa freeze (diag subset)
        progress(4, 6, "mislid_freeze")
        mrec = load_parts(out_dir, "mislid")
        mis = {}
        frozen_k = None
        for kappa, m in MISLID.items():
            st = per_stream(mrec, m)
            pooled = safety_block(st)
            rows = [x for r in mrec if r["method"] == m for x in r["rows"]]
            thr_max = [x.get("x_eps_cert_max") for x in rows if x.get("x_eps_cert_max") is not None]
            eta_min = [x.get("x_eta_hat_min") for x in rows if x.get("x_eta_hat_min") is not None]
            mis[m] = {"kappa": kappa, "n_streams": pooled["n_streams"], "n_problems": pooled["n_problems"],
                      "n_cert": pooled["n_cert"], "n_false": pooled["n_false"],
                      "false_streams": pooled["false_streams"], "S1": pooled["S1"],
                      "status_counts": dict(Counter(x["status"] for x in rows)),
                      "eps_cert_max_over_run": float(max(thr_max)) if thr_max else None,
                      "eta_hat_min_over_run": float(min(eta_min)) if eta_min else None,
                      "conv_2H_range": [float(min(x["x_conv_2H"] for x in rows)), float(max(x["x_conv_2H"]
                                                                                          for x in rows))]
                      if rows else None,
                      "fwer_cp_upper_le_0.05": bool(pooled["S1"]["cp_upper"] <= 0.05)}
            if frozen_k is None and mis[m]["fwer_cp_upper_le_0.05"]:
                frozen_k = kappa
        undecidable = frozen_k is None and all(v["n_cert"] == 0 for v in mis.values())
        if undecidable:                    # 0 certifications for every kappa: tie -> smallest kappa (most favourable)
            frozen_k = min(MISLID)
        summary["mislid"] = {"evaluated": mis, "frozen_kappa": frozen_k if frozen_k is not None else max(MISLID),
                             "tie_zero_cert": bool(undecidable),
                             "rule": "smallest kappa in {1,2} with FWER one-sided CP upper <= 0.05 on the diagnostic "
                                     "subset; if the subset cannot decide it (CP upper of 0/10 = 0.259) and every kappa "
                                     "has 0 certifications, the tie goes to the smaller kappa (the setting most "
                                     "favourable to the opponent)",
                             "subset": {k: v for k, v in MISLID_SEEDS.items()}, "problems_per_stream":
                                 MISLID_PROBLEMS}
        summary["downscale"] = [{
            "item": "MisLid-ms kappa sweep", "planned": "100 dev instances x 15 problems x kappa in {1,2}",
            "actual": f"10 dev instances (MS-H 800-804, MS-S 860-864) x first {MISLID_PROBLEMS} problems x kappa in "
                      "{1,2}",
            "reason": "MisLid-ms (valid ucb eta) checks every step and runs to T_max whenever it cannot certify "
                      "(~30 s/problem, ~450 s/stream); the setup smoke showed eps - 2H*eta_hat < 0 throughout. "
                      "200 full streams would need ~25 CPU-h. The frozen value only feeds the descriptive comparator "
                      "(E-ms(b) deleted in the F2 revision); the run records the max certificate threshold per problem "
                      "to show whether certification was ever possible."}]

        # ------------------------------------------------------------------ 7. outputs
        progress(5, 6, "write")
        allrecs = load_parts(out_dir, "main") + load_parts(out_dir, "mislid") + load_parts(out_dir, "retry")
        with open(out_dir / "results.jsonl", "w") as fh:
            for r in allrecs:
                for x in r["rows"]:
                    fh.write(json.dumps({"run_tag": r["tag"], **x}, sort_keys=True, default=str) + "\n")
        errs = [{"tag": r["tag"], "layer": r["layer"], "seed": r["seed"], "method": r["method"], "error": r["error"],
                 "tb": r.get("tb")} for r in allrecs if r.get("error")]
        summary["n_errors"] = len(errs)
        if errs:
            (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
        billing = int(sum((not x["billing_ok"]) or x["env_n_steps"] != x["n_rounds_billed"]
                          for r in allrecs for x in r["rows"]))
        summary["billing_mismatch"] = billing
        summary["write_ahead_ok"] = bool(all(r["info"]["write_ahead_ok"] for r in allrecs if r.get("info")))
        # samples: false certifications of JPC + a few correct ones per layer
        keys = ("run_tag", "layer", "instance", "method", "problem", "pid", "H", "n_policies", "status",
                "certified_policy", "new_env_steps", "true_regret", "false_cert", "true_gap", "gap_layer",
                "eta_near_over_eps", "mu_flip", "x_eps_cert", "x_eps_cert_end", "x_eps_cert_max", "x_eta_hat_end")
        smp = {"jpc_false_certs": [], "jpc_correct_certs": [], "guard_rows": [], "mislid_rows": []}
        for r in allrecs:
            for x in r["rows"]:
                d = {k: x.get(k) for k in keys if k in x or k == "run_tag"}
                d["run_tag"] = r["tag"]
                if r["method"] == "JPC" and x["false_cert"] and len(smp["jpc_false_certs"]) < 20:
                    smp["jpc_false_certs"].append(d)
                elif r["method"] == "JPC" and x["correct_cert"] and len(smp["jpc_correct_certs"]) < 10:
                    smp["jpc_correct_certs"].append(d)
                elif r["method"].startswith("Guard") and len(smp["guard_rows"]) < 10:
                    smp["guard_rows"].append(d)
                elif r["method"].startswith("MisLid") and len(smp["mislid_rows"]) < 10:
                    smp["mislid_rows"].append(d)
        for k, v in smp.items():
            (out_dir / "samples" / f"{k}.json").write_text(json.dumps(v, indent=1, default=str))
        frozen = {"task_id": TASK, "written": datetime.now().isoformat(), "dev_seeds": "800-899",
                  "guard_c": summary["guard"]["frozen_c"], "guard_rule": summary["guard"]["rule"],
                  "guard_evidence": {m: {"false_streams": g["pooled_100"]["false_streams"],
                                         "n_streams": g["pooled_100"]["n_streams"],
                                         "fwer_cp_upper": g["pooled_100"]["S1"]["cp_upper"],
                                         "completion_mean": g["pooled_100"]["completion_mean"]}
                                     for m, g in summary["guard"]["evaluated"].items()},
                  "mislid_ms_inflation_multiple": summary["mislid"]["frozen_kappa"],
                  "mislid_method_name": MISLID[summary["mislid"]["frozen_kappa"]],
                  "mislid_rule": summary["mislid"]["rule"],
                  "mislid_evidence": {m: {k: v[k] for k in ("n_cert", "n_problems", "false_streams",
                                                            "eps_cert_max_over_run", "eta_hat_min_over_run")}
                                      for m, v in summary["mislid"]["evaluated"].items()},
                  "role": "descriptive comparators only (F2 revision); not part of any confirmatory endpoint",
                  "g_haz_pass": summary["g_haz"]["pass"], "ms2_status": summary["g_haz"]["ms2_status"],
                  "code_sha256": code_sha()}
        (RES / "r4_gates").mkdir(exist_ok=True)
        (RES / "r4_gates" / "ms_frozen.json").write_text(json.dumps(frozen, indent=1, default=str))
        timing = defaultdict(list)
        for r in allrecs:
            if r.get("sec"):
                timing[f"{r['tag']}|{r['layer']}|{r['method']}"].append(r["sec"])
        summary["sec_per_stream"] = {k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
                                     for k, v in sorted(timing.items())}
        summary["code_sha256"] = code_sha()
        summary["pass_criteria"] = {"reproduction_verdict_recorded_per_layer": True,
                                    "guard_c_frozen": summary["guard"]["frozen_c"] is not None,
                                    "mislid_hyper_frozen": summary["mislid"]["frozen_kappa"] is not None,
                                    "no_errors": len(errs) == 0, "billing_mismatch_0": billing == 0}
        summary["pass"] = bool(all(summary["pass_criteria"].values()))
        summary["wall_min"] = (time.time() - t_start) / 60
        summary["end"] = datetime.now().isoformat()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        if a.mode == "pilot":                          # gate: full = same
            fd = RES / "full" / TASK
            fd.mkdir(parents=True, exist_ok=True)
            shutil.copy(out_dir / "summary.json", fd / "summary.json")
            shutil.copy(out_dir / "results.jsonl", fd / "results.jsonl")
            (fd / "samples").mkdir(exist_ok=True)
            for f in (out_dir / "samples").glob("*.json"):
                shutil.copy(f, fd / "samples" / f.name)
        status = "success" if summary["pass"] else "failed"
    except Exception as e:  # noqa: BLE001
        status = "failed"
        summary["fatal"] = repr(e)
        summary["tb"] = traceback.format_exc()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    finally:
        wall = (time.time() - t_start) / 60
        mark_done(status, f"{TASK}: g_haz_pass={summary.get('g_haz', {}).get('pass')} "
                          f"guard_c={summary.get('guard', {}).get('frozen_c')} "
                          f"mislid_kappa={summary.get('mislid', {}).get('frozen_kappa')}")
        update_gpu_progress(status, start_iso, wall, {"layers": "MS-H 800-859, MS-S 860-899", "streams": 100,
                                                      "methods": "JPC, Guard(c ascending), MisLid-ms kappa{1,2} diag",
                                                      "T_max": 6000, "workers": a.workers, "gpu": "RTX 4090 shared",
                                                      "cpu_bound": True, "concurrent": True})
        log(f"done status={status} wall={wall:.1f} min")


if __name__ == "__main__":
    main()
