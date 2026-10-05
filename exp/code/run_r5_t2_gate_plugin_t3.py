"""r5_t2_gate_plugin_t3 (PILOT == FULL for this dev gate): E-cost dev estimate + T3 full-horizon FWER audit + K=60 preview.

Setting: CR9 dev half, eps* = 0.001 (r4_gates/eps.json), Q = 15, delta = 0.05, dev seeds 900-999 (100 paired streams).
(A) E-cost (descriptive, no pass line): FDC vs P = {B2-fav, B2-fav-tight, FIX-bal-fav, Peace-fav, Peace-fav-bal, B3-fav,
    B5} at K = 20, stop 12/15. Paired geomean ratio FDC/p of N80_pen (primary, methodology s5) and of N80_raw
    (no error penalty), two-sided 95% paired-bootstrap CI (B = 1e4, seed 42, resampling streams), false streams.
(B) T3 full horizon: FDC and B2-fav-tight with stop_k = Q = 15 (stream runs until 15/15 certified or tau_R / pool end).
    Every certificate's correctness, N100 (15/15 checkpoint; tau_R if never or if any false cert), per-problem
    certification checkpoint. Gate T3: FDC false streams <= 1/100 in both windows (stop 12/15 and full horizon),
    every FDC false stream audited (re-run with U kept).
(C) K = 60 preview: FDC and B2-fav with K = 60 checkpoints (each method recomputes its own time-union budget from
    len(ctx.checkpoints); FDC beta + ln 3). Ratio only; K = 60 numbers are never mixed with K = 20.
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run). No evaluation seed is touched.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
TASK = "r5_t2_gate_plugin_t3"
RES = WS / "exp/results"
OUT = RES / "pilots" / TASK
AUDIT = OUT / "audit"
SEEDS = list(range(900, 1000))
if os.environ.get("T3_SMOKE"):  # pipeline smoke test only: 2 dev seeds, scratch output dir, no markers
    SEEDS = [900, 901]
    TASK = "r5_t2_gate_plugin_t3_SMOKE"
    OUT = Path(os.environ["T3_SMOKE"])
    AUDIT = OUT / "audit"
B_BOOT = 10_000
BOOT_SEED = 42

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

PLUGINS = ["B2-fav", "B2-fav-tight", "FIX-bal-fav", "Peace-fav", "Peace-fav-bal", "B3-fav", "B5"]
STOP_METHODS = ["FDC"] + PLUGINS
FULL_METHODS = ["FDC", "B2-fav-tight"]
K60_METHODS = ["FDC", "B2-fav"]
FILES = ["dsswm/baselines/fdc.py", "dsswm/baselines/frontier_common.py", "dsswm/baselines/combgame_joint.py",
         "dsswm/baselines/plugin_r5.py", "dsswm/baselines/peace_frontier.py", "dsswm/baselines/rage_frontier.py",
         "dsswm/baselines/maq_desc.py", "dsswm/streams/r5_registry.py", "dsswm/streams/frontier_runner.py",
         "dsswm/envs/pool_replay.py", "run_r5_t2_gate_plugin_t3.py"]
_ENV = {}


def sha(p):
    return hashlib.sha256((CODE / p).read_bytes()).hexdigest()[:16]


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def progress(done, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def init_env():
    from dsswm.envs.pool_replay import PoolReplayEnv
    eps = float(json.loads((RES / "r4_gates" / "eps.json").read_text())["eps_star"])
    env = PoolReplayEnv("CR9", "dev")
    probs = fr.cr_problems("visit")
    c20 = build_ctx(env, probs, eps, K=20)
    J = true_policy_values(c20.pols, env.w, env.true_mu("visit"))
    cfull = dataclasses.replace(c20, stop_k=c20.Q)          # full horizon: stop only at 15/15 (else tau_R)
    c60 = build_ctx(env, probs, eps, K=60, pols=c20.pols)
    _ENV.update(env=env, ctx={"stop": c20, "full": cfull, "k60": c60}, J=J,
                Js=np.array([J[c20.feas[q]].max() for q in range(c20.Q)]), eps=eps)


def job(a):
    window, name, seed = a
    ctx = _ENV["ctx"][window]
    J, Js = _ENV["J"], _ENV["Js"]
    try:
        m = reg.make_method(name)
        t0 = time.perf_counter()
        s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=False)
        ck = ctx.checkpoints
        n_stop_raw = int(ck[s["k_stop"]]) if s["reached_stop"] else int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))
        # every certificate with its true gap (lightweight audit, no re-run)
        certs = []
        for q in range(ctx.Q):
            k = s["cert_k"][q]
            if k < 0:
                continue
            pi = s["decided_pi"][q]
            gap = float(Js[q] - J[pi])
            certs.append({"q": q, "cert_k": int(k), "t": int(ck[k]), "decided_pi": int(pi), "gap": gap,
                          "false": bool(gap > ctx.eps + 1e-12)})
        # first checkpoint with >= 12 certified (stop-window prefix of a full-horizon run)
        k12 = next((i for i, c in enumerate(curve) if c >= 12), None)
        false_by_k12 = (sum(1 for c in certs if c["false"] and c["cert_k"] <= k12) if k12 is not None
                        else sum(c["false"] for c in certs))
        return {"window": window, "method": name, "seed": seed, "validity": m.validity, "alloc_kind": m.alloc_kind,
                "stop_k": int(ctx.stop_k), "K": int(len(ck)), "N_pen": int(s["N80"]), "N_raw": n_stop_raw,
                "N_over_tau": s["N80"] / ctx.tau_R, "completed": bool(s["reached_stop"]),
                "fwer_event": bool(s["fwer_event"]), "censored": bool(s["censored"]), "n_cert": s["n_cert"],
                "n_false": s["n_false"], "false_q": s["false_q"], "k_stop": s["k_stop"], "cert_k": s["cert_k"],
                "decided_pi": s["decided_pi"], "certificates": certs, "k12_in_run": k12,
                "false_certs_by_k12": int(false_by_k12),
                "billing": {"billed": s["billed"], "served": s["served"], "skipped": s["skipped"],
                            "reselected": s["reselected"], "loop_arrivals": s["loop_arrivals"], "t_end": s["t_end"]},
                "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"],
                "n_cert_curve": curve, "sec": round(time.perf_counter() - t0, 3), "error": None}
    except Exception:  # noqa: BLE001
        return {"window": window, "method": name, "seed": seed, "error": traceback.format_exc()}


def audit_job(a):
    """Re-run one false stream with U kept; record every certificate, its true gap and U at certification."""
    window, name, seed = a
    ctx, J, Js = _ENV["ctx"][window], _ENV["J"], _ENV["Js"]
    m = reg.make_method(name)
    s, rows = run_stream(_ENV["env"], m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=True)
    certs = []
    for q in range(ctx.Q):
        if s["cert_k"][q] < 0:
            continue
        pi = s["decided_pi"][q]
        gap = float(Js[q] - J[pi])
        kq = s["cert_k"][q]
        certs.append({"q": q, "cert_k": kq, "t": int(ctx.checkpoints[kq]), "decided_pi": pi, "J_star": float(Js[q]),
                      "J_decided": float(J[pi]), "gap": gap, "eps": ctx.eps, "false": bool(gap > ctx.eps + 1e-12),
                      "U_at_cert": rows[kq]["U"][q] if kq < len(rows) else None})
    return {"window": window, "method": name, "seed": seed, "summary": s, "certificates": certs,
            "rows": [{k: r[k] for k in ("k", "t", "n_cert", "n_false", "new", "new_false", "n_cells", "U")}
                     for r in rows]}


def paired_stats(a, b, idx_boot):
    d = np.log(a) - np.log(b)
    boot = d[idx_boot].mean(1)
    return {"geomean_ratio": float(np.exp(d.mean())),
            "ci95_two_sided": [float(np.exp(np.quantile(boot, 0.025))), float(np.exp(np.quantile(boot, 0.975)))],
            "ub95_one_sided": float(np.exp(np.quantile(boot, 0.95))),
            "frac_faster": float((d < 0).mean()), "frac_tied": float((d == 0).mean()),
            "frac_slower": float((d > 0).mean()), "sd_log_ratio": float(d.std(ddof=1)), "n": int(len(d))}


def git_info():
    def g(*a):
        p = subprocess.run(["git", *a], cwd=str(CODE), capture_output=True, text=True)
        return p.stdout.strip()
    return {"head": g("rev-parse", "HEAD"), "dirty_files": g("status", "--porcelain", "--", ".").splitlines()[:50]}


def method_stats(rr, tau):
    N = np.array([x["N_pen"] for x in rr], float)
    fs = sum(x["fwer_event"] for x in rr)
    return {"validity": rr[0]["validity"], "geomean_N_pen": float(np.exp(np.log(N).mean())),
            "geomean_N_pen_over_tau": float(np.exp(np.log(N).mean()) / tau),
            "geomean_N_raw": float(np.exp(np.log([x["N_raw"] for x in rr]).mean())),
            "median_N_pen": float(np.median(N)), "completed": int(sum(x["completed"] for x in rr)),
            "censored": int(sum(x["censored"] for x in rr)), "false_streams": int(fs),
            "fwer_cp_upper": round(cp_upper(fs, len(rr)), 4), "false_certs": int(sum(x["n_false"] for x in rr)),
            "billing_ok_all": all(x["billing_ok"] for x in rr),
            "sec_per_stream_mean": float(np.mean([x["sec"] for x in rr])), "n": len(rr)}


def plot(by, tau, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in STOP_METHODS:
        st = method_stats([by[("stop", m, s)] for s in SEEDS], tau)
        x = st["geomean_N_raw"]
        y = st["false_streams"] / len(SEEDS)
        rig = m == "FDC"
        ax.scatter(x, y, s=70 if rig else 45, marker="*" if rig else "o", color="black" if rig else "tab:red",
                   zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax.axhline(0.05, ls=":", color="grey", lw=0.8)
    ax.text(ax.get_xlim()[0], 0.052, " delta = 0.05", fontsize=7, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("geomean N80_raw (arrivals; stop 12/15, no error penalty)")
    ax.set_ylabel("false-stream rate (100 dev streams)")
    ax.set_title("CR9 dev, eps*=0.001: speed vs guarantee (black star = FDC, rigorous; red = plug-in/asymptotic)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = datetime.now()
    (OUT / "start_time.txt").write_text(t_start.isoformat())
    init_env()
    c20, cfull, c60 = (_ENV["ctx"][w] for w in ("stop", "full", "k60"))
    assert abs(c20.eps - 0.001) < 1e-12 and len(c20.checkpoints) == 20 and c20.stop_k == 12 and c20.Q == 15
    assert cfull.stop_k == 15 and len(c60.checkpoints) == 60 and c60.stop_k == 12
    tau = int(c20.tau_R)
    ck20 = set(int(x) for x in c20.checkpoints)
    ck60 = set(int(x) for x in c60.checkpoints)
    print(f"tau_R={tau} K20 last={int(c20.checkpoints[-1])} K60 last={int(c60.checkpoints[-1])} "
          f"K20 subset of K60={ck20 <= ck60}", flush=True)
    describe = {}
    for m in sorted(set(STOP_METHODS + FULL_METHODS + K60_METHODS)):
        inst = reg.make_method(m)
        inst.setup(c20)
        try:
            describe[m] = json.loads(json.dumps(inst.describe(), default=str)) if hasattr(inst, "describe") else {}
        except Exception as e:  # noqa: BLE001
            describe[m] = {"describe_error": repr(e)}
        describe[m]["registry"] = {"validity_class": reg.spec(m).validity_class, "role": reg.spec(m).role,
                                   "source": reg.spec(m).source, "defaults": reg.spec(m).defaults}
    slow = ["Peace-fav", "Peace-fav-bal", "B3-fav"]
    jobs = [("stop", m, s) for m in slow for s in SEEDS]
    jobs += [("full", m, s) for m in FULL_METHODS for s in SEEDS]
    jobs += [("k60", m, s) for m in K60_METHODS for s in SEEDS]
    jobs += [("stop", m, s) for m in STOP_METHODS if m not in slow for s in SEEDS]
    total = len(jobs)
    progress(0, total)
    rows, errs = [], []
    rfile, ffile = OUT / "results.jsonl", OUT / "fullhorizon.jsonl"
    rfile.write_text("")
    ffile.write_text("")
    with get_context("fork").Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            if r.get("error"):
                errs.append(r)
            else:
                rows.append(r)
                with open(ffile if r["window"] == "full" else rfile, "a") as f:
                    f.write(json.dumps(r) + "\n")
            if (i + 1) % 20 == 0 or i + 1 == total:
                progress(i + 1, total, {"runs_done": i + 1, "errors": len(errs)})
                print(f"[{i+1}/{total}] last={r['window']}:{r['method']} seed={r['seed']} errors={len(errs)}",
                      flush=True)
    if errs:
        (OUT / "errors.log").write_text("\n\n".join(f"{e['window']} {e['method']} {e['seed']}\n{e['error']}"
                                                    for e in errs))
        raise RuntimeError(f"{len(errs)} stream jobs failed (see errors.log)")

    by = {(r["window"], r["method"], r["seed"]): r for r in rows}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(SEEDS), size=(B_BOOT, len(SEEDS)))  # same resamples for every comparison

    # ---------------- (A) E-cost
    per_stop = {m: method_stats([by[("stop", m, s)] for s in SEEDS], tau) for m in STOP_METHODS}
    Npen = {m: np.array([by[("stop", m, s)]["N_pen"] for s in SEEDS], float) for m in STOP_METHODS}
    Nraw = {m: np.array([by[("stop", m, s)]["N_raw"] for s in SEEDS], float) for m in STOP_METHODS}
    ecost = {}
    for p in PLUGINS:
        ecost[p] = {"N80_pen": paired_stats(Npen["FDC"], Npen[p], idx),
                    "N80_raw": paired_stats(Nraw["FDC"], Nraw[p], idx),
                    "plugin_false_streams": per_stop[p]["false_streams"],
                    "plugin_false_certs": per_stop[p]["false_certs"],
                    "plugin_completed": per_stop[p]["completed"], "validity": per_stop[p]["validity"]}
    fastest_raw = min(PLUGINS, key=lambda p: per_stop[p]["geomean_N_raw"])
    fastest_pen = min(PLUGINS, key=lambda p: per_stop[p]["geomean_N_pen"])
    with open(OUT / "paired_ratios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparator", "validity", "endpoint", "geomean_ratio_FDC_over_p", "ci95_lo", "ci95_hi",
                    "frac_FDC_faster", "frac_tied", "frac_FDC_slower", "p_false_streams", "p_false_certs",
                    "p_completed"])
        for p in PLUGINS:
            for ep in ("N80_pen", "N80_raw"):
                e = ecost[p][ep]
                w.writerow([p, ecost[p]["validity"], ep, e["geomean_ratio"], e["ci95_two_sided"][0],
                            e["ci95_two_sided"][1], e["frac_faster"], e["frac_tied"], e["frac_slower"],
                            ecost[p]["plugin_false_streams"], ecost[p]["plugin_false_certs"],
                            ecost[p]["plugin_completed"]])
    with open(OUT / "per_stream_n80.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + [f"{m}_pen" for m in STOP_METHODS] + [f"{m}_raw" for m in STOP_METHODS])
        for i, s in enumerate(SEEDS):
            w.writerow([s] + [int(Npen[m][i]) for m in STOP_METHODS] + [int(Nraw[m][i]) for m in STOP_METHODS])

    # ---------------- (B) full horizon
    per_full = {}
    for m in FULL_METHODS:
        rr = [by[("full", m, s)] for s in SEEDS]
        st = method_stats(rr, tau)
        st = {k.replace("N_", "N100_"): v for k, v in st.items()}
        st["false_streams_by_12of15_prefix"] = int(sum(x["false_certs_by_k12"] > 0 for x in rr))
        st["false_certs_after_12of15"] = int(sum(x["n_false"] - x["false_certs_by_k12"] for x in rr))
        st["per_problem_median_cert_t"] = []
        for q in range(c20.Q):
            ts = [int(cfull.checkpoints[x["cert_k"][q]]) if x["cert_k"][q] >= 0 else tau for x in rr]
            st["per_problem_median_cert_t"].append({"q": q, "median_t": float(np.median(ts)),
                                                    "never_certified": int(sum(x["cert_k"][q] < 0 for x in rr))})
        st["n_cert_hist"] = {str(int(k)): int(v) for k, v in zip(*np.unique([x["n_cert"] for x in rr],
                                                                           return_counts=True))}
        per_full[m] = st
    # prefix consistency: full-horizon run truncated at its 12/15 checkpoint == stop-window run
    pref = {}
    for m in FULL_METHODS:
        if m not in STOP_METHODS:
            continue
        mism = []
        for s in SEEDS:
            a, b = by[("full", m, s)], by[("stop", m, s)]
            kk = b["k_stop"]
            same = (a["k12_in_run"] == kk) and (a["n_cert_curve"][:(kk + 1 if kk is not None else 20)]
                                                 == b["n_cert_curve"][:(kk + 1 if kk is not None else 20)])
            if kk is not None:
                same = same and all((a["cert_k"][q] == b["cert_k"][q] and a["decided_pi"][q] == b["decided_pi"][q])
                                    for q in range(c20.Q) if b["cert_k"][q] >= 0)
            if not same:
                mism.append(s)
        pref[m] = {"streams_compared": len(SEEDS), "mismatch_seeds": mism}
    full_ratio = None
    N100 = {m: np.array([by[("full", m, s)]["N_pen"] for s in SEEDS], float) for m in FULL_METHODS}
    N100r = {m: np.array([by[("full", m, s)]["N_raw"] for s in SEEDS], float) for m in FULL_METHODS}
    full_ratio = {"N100_pen": paired_stats(N100["FDC"], N100["B2-fav-tight"], idx),
                  "N100_raw": paired_stats(N100r["FDC"], N100r["B2-fav-tight"], idx)}

    # ---------------- (C) K = 60 preview
    per_k60 = {m: method_stats([by[("k60", m, s)] for s in SEEDS], tau) for m in K60_METHODS}
    N60p = {m: np.array([by[("k60", m, s)]["N_pen"] for s in SEEDS], float) for m in K60_METHODS}
    N60r = {m: np.array([by[("k60", m, s)]["N_raw"] for s in SEEDS], float) for m in K60_METHODS}
    k60 = {"per_method": per_k60, "FDC_over_B2-fav": {"N80_pen": paired_stats(N60p["FDC"], N60p["B2-fav"], idx),
                                                      "N80_raw": paired_stats(N60r["FDC"], N60r["B2-fav"], idx)},
           "K20_reference_FDC_over_B2-fav": ecost["B2-fav"],
           "FDC_K60_over_K20": paired_stats(N60p["FDC"], Npen["FDC"], idx),
           "B2fav_K60_over_K20_raw": paired_stats(N60r["B2-fav"], Nraw["B2-fav"], idx),
           "note": "K=60 grid; each method recomputes its own time-union budget; report only, never mixed with K=20",
           "K20_checkpoints_subset_of_K60": ck20 <= ck60}

    # ---------------- audits: every false stream (lightweight from certificates); FDC false streams re-run with U
    fdc_false = [(w, s) for w in ("stop", "full", "k60") for s in SEEDS
                 if ("FDC" in (STOP_METHODS if w == "stop" else FULL_METHODS if w == "full" else K60_METHODS))
                 and by[(w, "FDC", s)]["fwer_event"]]
    audits = []
    if fdc_false:
        with get_context("fork").Pool(min(4, len(fdc_false))) as pool:
            audits = pool.map(audit_job, [(w, "FDC", s) for w, s in fdc_false])
        for a in audits:
            (AUDIT / f"FDC_{a['window']}_seed{a['seed']}.json").write_text(json.dumps(a, indent=1, default=str))
    light = [{"window": r["window"], "method": r["method"], "seed": r["seed"],
              "false_certs": [c for c in r["certificates"] if c["false"]]} for r in rows if r["fwer_event"]]
    (AUDIT / "index.json").write_text(json.dumps({
        "fdc_full_audits": [f"FDC_{a['window']}_seed{a['seed']}.json" for a in audits],
        "all_false_streams": light,
        "note": "every stream with a wrong certificate (all methods, all windows); FDC false streams are re-run with "
                "U kept (one file each)"}, indent=1))
    cert_audit = {w: {m: {"certificates": int(sum(len(by[(w, m, s)]["certificates"]) for s in SEEDS)),
                          "wrong": int(sum(sum(c["false"] for c in by[(w, m, s)]["certificates"]) for s in SEEDS)),
                          "max_gap_over_eps": float(max([c["gap"] / c_eps for s in SEEDS
                                                         for c in by[(w, m, s)]["certificates"]] or [0.0]))}
                      for m in ms}
                  for w, ms, c_eps in (("stop", STOP_METHODS, c20.eps), ("full", FULL_METHODS, c20.eps),
                                       ("k60", K60_METHODS, c20.eps))}
    (AUDIT / "certificate_counts.json").write_text(json.dumps(cert_audit, indent=1))

    plot(by, tau, OUT / "speed_vs_guarantee.png")
    samples = [{k: by[(w, m, s)][k] for k in ("window", "method", "seed", "N_pen", "N_raw", "k_stop", "cert_k",
                                               "decided_pi", "n_false", "billing_ok")}
               for s in (SEEDS[0], SEEDS[len(SEEDS) // 2], SEEDS[-1])
               for w, ms in (("stop", STOP_METHODS), ("full", FULL_METHODS), ("k60", K60_METHODS)) for m in ms]
    (OUT / "samples.json").write_text(json.dumps(samples, indent=1))
    configs = {"setting": {"layer": "CR9", "half": "dev", "eps": c20.eps, "Q": c20.Q, "delta": c20.delta,
                           "tau_R": tau, "windows": {"stop": {"K": 20, "stop_k": 12}, "full": {"K": 20, "stop_k": 15},
                                                     "k60": {"K": 60, "stop_k": 12}},
                           "seeds": [SEEDS[0], SEEDS[-1]], "n_streams": len(SEEDS),
                           "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "unit": "stream (paired)"}},
               "methods": describe, "git": git_info(),
               "code_sha256_16": {p: sha(p) for p in FILES if (CODE / p).exists()}}
    (OUT / "configs.json").write_text(json.dumps(configs, indent=1, default=str))

    fdc_stop_false = per_stop["FDC"]["false_streams"]
    fdc_full_false = per_full["FDC"]["false_streams"]
    audited_ok = all(any(a["window"] == w and a["seed"] == s for a in audits) for w, s in fdc_false)
    t3 = fdc_stop_false <= 1 and fdc_full_false <= 1 and audited_ok
    bill = all(r["billing_ok"] for r in rows)
    t_end = datetime.now()
    summary = {
        "task_id": TASK, "mode": "pilot (== full: dev gate, same streams)", "started_at": t_start.isoformat(),
        "ended_at": t_end.isoformat(), "wall_min": round((t_end - t_start).total_seconds() / 60, 1),
        "timing_note": "concurrent run (4 workers; other r5 tasks share the 20-core host)",
        "setting": configs["setting"],
        "E_cost": {"rule": "descriptive, no pass line; paired geomean ratio FDC/p with two-sided 95% bootstrap CI",
                   "per_plugin": ecost, "fastest_plugin_by_geomean_N80_raw": fastest_raw,
                   "fastest_plugin_by_geomean_N80_pen": fastest_pen},
        "gate_T3": {"rule": "FDC false streams <= 1/100 in both windows (stop 12/15; full horizon to 15/15 or tau_R), "
                            "every FDC false stream audited",
                    "fdc_false_streams_stop": fdc_stop_false, "fdc_cp_upper_stop": per_stop["FDC"]["fwer_cp_upper"],
                    "fdc_false_streams_full": fdc_full_false, "fdc_cp_upper_full": per_full["FDC"]["fwer_cp_upper"],
                    "fdc_false_streams_k60": per_k60["FDC"]["false_streams"], "audited": audited_ok, "pass": t3,
                    "audit_dir": os.path.relpath(AUDIT, WS)},
        "full_horizon": {"per_method": per_full, "FDC_over_B2-fav-tight": full_ratio,
                         "prefix_consistency_vs_stop_window": pref},
        "K60_preview": k60, "per_method_stop": per_stop, "certificate_audit": cert_audit,
        "billing_ok_all": bill, "go_no_go": "GO" if (t3 and bill) else "NO_GO", "eval_seeds_touched": False,
        "outputs": ["summary.json", "results.jsonl", "fullhorizon.jsonl", "paired_ratios.csv", "per_stream_n80.csv",
                    "configs.json", "samples.json", "audit/", "speed_vs_guarantee.png"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def mark_done(status, txt):
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
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


if __name__ == "__main__":
    try:
        s = main()
        g = s["gate_T3"]
        txt = (f"{s['go_no_go']} | T3 pass={g['pass']} FDC false stop={g['fdc_false_streams_stop']}/100 "
               f"full={g['fdc_false_streams_full']}/100 | fastest plug-in {s['E_cost']['fastest_plugin_by_geomean_N80_raw']}")
        mark_done("success", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
