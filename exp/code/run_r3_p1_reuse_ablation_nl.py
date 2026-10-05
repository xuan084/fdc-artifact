"""r3_p1_reuse_ablation_nl: P1 NL-R0 reuse causal ablation (dev seeds only; never touches evaluation seeds).

Design: NL-R0 (E1-NL-S G_1, realisable), gap-quota stream generator (tie/near/clear x 5), {JPC, B3} x
{off, vol, ev1, full}, CRN (every arm of one (instance, stream, method) runs on a fresh copy of the same platform).
  pilot : dev 720-727 x stream 0 x first 5 problems (320 runs) -- pipeline check + magnitudes only
          (GO does NOT depend on the direction of the effect).
  full  : dev 720-739 x streams 0,1,2 x 15 problems (7200 runs) -- power study for r3_prereg_lock
          (instance-level log-ratio SD, A2 MDE at n=120, tie/near/clear composition).
Both modes write, at the end of every problem (learner side, before truth scoring):
  predictors.jsonl                 -- Q-family predictors (Lambda_perp, rho*, A_k, Rem bar ...), write-ahead
  samples/ledgers/*.npz            -- ledger snapshot: per-x counts (x = (observable state code, action)) split by
                                      provenance, per-cell factor counts, theta_hat (LR-set MLE), Theta_t mask
  samples/ledgers/*_obs.jsonl.gz   -- payloads of every observation referenced by a snapshot (serial-keyed)
so that P4 can recompute T0 with zero new interaction.  results.jsonl is the harness (truth) scoring.
Usage: run_r3_p1_reuse_ablation_nl.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3"
TASK = "r3_p1_reuse_ablation_nl"
METHODS = ["JPC", "B3"]
ARMS = ["off", "vol", "ev1", "full"]
EPS, DELTA = 0.02, 0.05
TAU = 3000
TMAX = 6000
RESERVE_START = 761            # quota_fail replacement block: outside every allocated dev / eval seed block
MODES = {
    "pilot": {"seeds": list(range(720, 728)), "streams": [0], "n_problems": 5},
    "full": {"seeds": list(range(720, 740)), "streams": [0, 1, 2], "n_problems": 15},
}
CODE_FILES = ["dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py", "dsswm/streams/gap_quota.py",
              "dsswm/streams/r3_harness.py", "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py",
              "run_r3_p1_reuse_ablation_nl.py"]
PROV_CODE = {"initial": 0, "real": 1, "replay_vol": 2, "replay_vol_pad": 3, "replay_orth": 4}


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


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


# ============================================================================================ worker (learner + harness)
_W: dict = {}


def _worker_ctx():
    if "pub" not in _W:
        torch.set_num_threads(1)
        import dsswm.streams.r3_harness as H
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        from dsswm.streams.gap_quota import QuotaBuilder
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        b = QuotaBuilder(MODES["pilot"]["seeds"][0], ncl, None, CACHE)
        prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
        TH = NLLeverage(prop).theta_matrix(ncl.np_params)
        lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        _W.update(ncl=ncl, prop=prop, pub=PublicNL(prop, ncl, [], [], 1.0, 2, EPS, DELTA), builders={},
                  TH=TH, lev=lev, orig_solve=H.solve, ctx=None)
        H.solve = _hooked_solve                     # learner-side snapshot hook (run_cell calls H.solve)
    return _W


def _hooked_solve(pub, method, k, sw, lr, handle, rng, tmax):
    """Learner side: solve, then -- at the certification time, before ANY truth scoring -- log the predictors and the
    ledger snapshot. Uses only public objects (class J tables, the evidence set, the switch accounting)."""
    res = _W["orig_solve"](pub, method, k, sw, lr, handle, rng, tmax)
    C = _W["ctx"]
    if C is None or sw.arm != C["arm"]:          # sibling platform run of the vol arm (arm 'full'): not logged
        return res
    t0 = time.perf_counter()
    try:
        _snapshot(C, pub, method, k, sw, lr, res)
    except Exception as e:  # noqa: BLE001 - a snapshot failure is recorded, the run continues
        C["snap_errors"].append({"arm": C["arm"], "k": k, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
    C["pred_sec"][(C["arm"], k)] = time.perf_counter() - t0
    return res


def _snapshot(C, pub, method, k, sw, lr, res):
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.mechanism.leverage import select_pairs, theta_hat_index
    lev, TH = _W["lev"], _W["TH"]
    q = pub.problems[k]
    mask = lr.mask().numpy().astype(bool)
    cum = lr.inner.cum.cpu().numpy()
    k_hat = theta_hat_index(cum, mask)
    pi_hat = res["pi"]
    pi_src = "certificate"
    if pi_hat is None:
        pi_hat = int(certify_minimax(pub.Reg[k], mask, pub.eps, pub.top_m)["pi"])
        pi_src = "minimax_on_mask"
    r_bar = float(res["extra"].get("r_bar_end", float("nan")))
    obs = list(lr.obs)
    provs = [r.provenance for r in lr.rows]
    pairs = select_pairs(pub.J[k], mask, k_hat, int(pi_hat))
    pred = lev.predictors(q, obs, TH[k_hat], pairs, r_bar if np.isfinite(r_bar) else 0.0, pub.eps, with_rem=True)
    Jh = np.sort(pub.J[k][k_hat])[::-1]
    key = {"instance": C["seed"], "stream": C["stream"], "method": method, "arm": C["arm"], "problem": int(k)}
    C["plog"].write({**key, "pid": q.pid, "status": res["status"], "pi_hat": int(pi_hat), "pi_hat_source": pi_src,
                     "theta_hat": int(k_hat), "lr_mle": int(lr.mle()), "set_size": int(mask.sum()),
                     "new_steps": int(sw.new_steps), "n_rows_in_set": len(obs),
                     "n_replay_in_set": int(lr.n_replay), "prov_counts": lr.provenance_counts(),
                     "r_bar_method": r_bar, "gap_hat_over_eps": float((Jh[0] - Jh[1]) / pub.eps) if len(Jh) > 1
                     else None, **pred})
    # ---- ledger snapshot (per-x counts by provenance, per-cell factor counts, theta_hat, Theta_t mask)
    codec = pub.prop.codec
    xs = defaultdict(lambda: [0, 0])
    real_obs, rep_obs = [], []
    for o, p in zip(obs, provs):
        rep = p.startswith("replay")
        xs[(codec.encode(o.loads, o.engaged), int(o.action))][1 if rep else 0] += 1
        (rep_obs if rep else real_obs).append(o)
        C["obs_store"].setdefault(int(o.serial), o.payload())
    xk = sorted(xs)
    fn = C["ledger_dir"] / f"i{C['seed']}_s{C['stream']}_{method}_{C['arm']}_p{k:02d}.npz"
    tmp = fn.with_name(fn.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, x_code=np.array([a for a, _ in xk], np.int64), x_action=np.array([b for _, b in xk], np.int64),
        x_n_real=np.array([xs[x][0] for x in xk], np.int64), x_n_replay=np.array([xs[x][1] for x in xk], np.int64),
        cell_counts_real=lev.cell_counts(real_obs), cell_counts_replay=lev.cell_counts(rep_obs),
        theta_hat=np.int64(k_hat), theta_hat_vec=TH[k_hat], lr_mle=np.int64(lr.mle()),
        mask_packed=np.packbits(mask), mask_len=np.int64(mask.size), set_size=np.int64(mask.sum()),
        serials=np.array([int(o.serial) for o in obs], np.int64),
        prov=np.array([PROV_CODE[p] for p in provs], np.int8), pi_hat=np.int64(pi_hat),
        n_rounds_billed=np.int64(sw.n_rounds_billed), new_steps=np.int64(sw.new_steps), cutoff=float(lr.cutoff()))
    os.replace(tmp, fn)
    C["n_snap"] += 1


def run_job(seed, stream, method, n_problems, out_dir):
    """All four arms of one (instance, stream, method). Writes its own part files (resumable)."""
    from dsswm.mechanism.predictor_log import PredictorLog
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.r3_harness import run_cell
    W = _worker_ctx()
    tag = f"i{seed}_s{stream}_{method}"
    parts = out_dir / "parts"
    ppath = parts / f"{tag}_predictors.jsonl"
    if ppath.exists():
        ppath.unlink()                                  # an unfinished job is rerun from scratch
    if seed not in W["builders"]:
        W["builders"] = {seed: QuotaBuilder(seed, W["ncl"], None, CACHE, prop_cpu=W["prop"])}
    b = W["builders"][seed]
    sel = b.select("quota")
    order = np.random.default_rng([seed, 45, stream]).permutation(15)
    ov = [o["cos_max"] for o in b.overlap([sel["base"][i] for i in order])][:n_problems]
    ctx = {"seed": seed, "stream": stream, "plog": PredictorLog(ppath), "ledger_dir": out_dir / "samples" / "ledgers",
           "obs_store": {}, "pred_sec": {}, "snap_errors": [], "n_snap": 0, "arm": None}
    rows, samples, errs = [], [], []
    t_job = time.perf_counter()
    for arm in ARMS:
        ctx["arm"] = arm
        W["ctx"] = ctx
        t0 = time.perf_counter()
        try:
            rr = run_cell(b, W["pub"], stream, "quota", method, arm, tmax=TMAX, n_problems=n_problems, overlap=ov,
                          samples=samples)
            for r in rr:
                ps = ctx["pred_sec"].get((arm, r["problem_index"]), 0.0)
                r["predictor_sec"] = ps
                r["wall_clock_s_method"] = r["wall_clock_s"] - ps
                r["cell_sec"] = time.perf_counter() - t0
                r["problem"] = r["problem_index"]
            rows += rr
        except Exception as e:  # noqa: BLE001 - crashes are counted, not hidden
            errs.append({"seed": seed, "stream": stream, "method": method, "arm": arm, "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
        finally:
            W["ctx"] = None
    with gzip.open(ctx["ledger_dir"] / f"{tag}_obs.jsonl.gz", "wt") as fh:
        for s in sorted(ctx["obs_store"]):
            fh.write(ctx["obs_store"][s] + "\n")
    part = {"tag": tag, "rows": rows, "samples": samples, "errors": errs, "snap_errors": ctx["snap_errors"],
            "n_snapshots": ctx["n_snap"], "job_sec": time.perf_counter() - t_job, "worker_pid": os.getpid()}
    tmp = parts / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(part, default=str))
    os.replace(tmp, parts / f"{tag}.json")
    return {"tag": tag, "n_rows": len(rows), "n_err": len(errs), "sec": part["job_sec"]}


# ============================================================================================ main-process stages
def build_tables(seeds):
    """Class J tables of the candidate pools on GPU (cached), quota selection, deterministic replacement."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.gap_quota import QuotaBuilder, fill_instances
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    t0 = time.perf_counter()
    info, ok = {}, {}

    def check(s):
        if s not in ok:
            b = QuotaBuilder(s, ncl, propg, CACHE, prop_cpu=propc)
            sel = b.select("quota")
            ok[s] = not sel["quota_fail"]
            info[s] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"], "layer_counts": sel["layer_counts"],
                       "tables_computed": b.n_table_computed, "table_sec": b.sec_tables}
        return ok[s]

    used, repl = fill_instances(seeds, check, reserve_start=RESERVE_START)
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
            "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": None,
            "note": "GPU only for class J tables of candidate problems (cached); learner loop + predictors on CPU, "
                    "4 workers, OMP=1 each; 并发运行（与其他 3 个任务共享 4090 与 CPU）"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg
    if dev == "cuda":
        torch.cuda.empty_cache()
    return used, repl, info, {"sec": time.perf_counter() - t0, "vram_peak_mb": vram, "device": dev}


def run_jobs(jobs, out_dir, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [j for j in jobs if not (out_dir / "parts" / f"i{j[0]}_s{j[1]}_{j[2]}.json").exists()]
    done = len(jobs) - len(todo)
    progress(done, len(jobs), "runs", {"resumed_jobs": done})
    log = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_job, *j, out_dir): j for j in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"tag": str(futs[f]), "fatal": repr(e)}
            log.append(r)
            done += 1
            progress(done, len(jobs), "runs", {"last": r.get("tag")})
            print("job", done, "/", len(jobs), r, flush=True)
    return log


def collect(jobs, out_dir):
    from dsswm.mechanism.predictor_log import ResultLog, join, load_jsonl
    rows, samples, errs, snap_errs, n_snap = [], [], [], [], 0
    pred_path, res_path = out_dir / "predictors.jsonl", out_dir / "results.jsonl"
    for p in (pred_path, res_path):
        if p.exists():
            p.unlink()
    with open(pred_path, "w") as fp:
        for j in jobs:
            tag = f"i{j[0]}_s{j[1]}_{j[2]}"
            pp = out_dir / "parts" / f"{tag}.json"
            if not pp.exists():
                errs.append({"tag": tag, "error": "job part missing (worker died)"})
                continue
            part = json.loads(pp.read_text())
            rows += part["rows"]
            samples += part["samples"]
            errs += part["errors"]
            snap_errs += part["snap_errors"]
            n_snap += part["n_snapshots"]
            for rec in load_jsonl(out_dir / "parts" / f"{tag}_predictors.jsonl"):
                fp.write(json.dumps(rec, sort_keys=True) + "\n")
    rlog = ResultLog(res_path, fsync=False)
    for r in rows:
        rlog.write(r)
    try:
        joined = join(pred_path, res_path, check_order=True)
        wa = {"n_joined": len(joined), "n_results": len(rows), "write_ahead_ok": True}
    except RuntimeError as e:
        wa = {"n_joined": None, "n_results": len(rows), "write_ahead_ok": False, "error": str(e)}
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:40], indent=1, default=str))
    if errs or snap_errs:
        (out_dir / "errors.json").write_text(json.dumps({"run_errors": errs, "snapshot_errors": snap_errs}, indent=1))
    return rows, errs, snap_errs, n_snap, wa


# ============================================================================================ analysis
def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n == 0:
        return {"n": 0}
    sd = float(v.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    out = {"n": n, "mean": float(v.mean()), "median": float(np.median(v)), "sd": sd,
           "ci95_t_descriptive": None, "ratio_exp_mean": float(math.exp(v.mean()))}
    if se is not None:
        from scipy.stats import t as tdist
        h = float(tdist.ppf(0.975, n - 1)) * se
        out["ci95_t_descriptive"] = [out["mean"] - h, out["mean"] + h]
        out["mde80_n120"] = (1.959964 + 0.841621) * sd / math.sqrt(120)    # two-sided 5 %, 80 % power, n = 120
    return out


def audit(rows):
    """Billing + provenance audit per row."""
    bad = []
    for r in rows:
        pc = r["prov_counts"] or {}
        rep = sum(v for kk, v in pc.items() if kk.startswith("replay"))
        n_k = r["n_rounds_billed"] - 20 - r["new_env_steps"]
        conds = {"billing_ok": r["billing_ok"], "prov_sum_eq_total": sum(pc.values()) == r["n_rounds_total"],
                 "prov_replay_eq_replay_steps": rep == r["replay_steps"],
                 "pad_eq": pc.get("replay_vol_pad", 0) == r["replay_pad_steps"],
                 "env_eq_billed": r["env_n_steps"] == r["n_rounds_billed"]}
        if r["arm"] == "vol":
            conds["vol_replay_eq_Nk"] = r["replay_steps"] == n_k
            conds["vol_real_eq_n0_plus_new"] = pc.get("initial", 0) + pc.get("real", 0) == 20 + r["new_env_steps"]
        elif r["arm"] == "off":
            conds["off_no_replay"] = rep == 0
            conds["off_drops_old"] = pc.get("initial", 0) + pc.get("real", 0) == 20 + r["new_env_steps"]
        else:
            conds["persist_no_replay"] = rep == 0
            conds["persist_sees_all_billed"] = pc.get("initial", 0) + pc.get("real", 0) == r["n_rounds_billed"]
        if r["arm"] == "ev1" and r["status"] == "CERTIFIED":
            conds["ev1_min_one_new_step"] = r["new_env_steps"] >= 1
        f = [kk for kk, v in conds.items() if not v]
        if f:
            bad.append({"instance": r["instance"], "stream": r["stream"], "method": r["method"], "arm": r["arm"],
                        "k": r["problem_index"], "failed": f})
    by_arm = {a: sum(1 for b in bad if b["arm"] == a) for a in ARMS}
    return {"n_rows": len(rows), "n_bad": len(bad), "bad_by_arm": by_arm, "examples": bad[:20],
            "billing_mismatch_by_arm": {a: sum(not r["billing_ok"] for r in rows if r["arm"] == a) for a in ARMS},
            "replay_billed_total": sum(r["replay_steps"] for r in rows if r["arm"] != "vol")}


def analyse(rows, seeds, streams):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["arm"])].append(r)
    by_cell = {}
    for (m, a), rr in sorted(cells.items()):
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        by_cell[f"{m}|{a}"] = {
            "n": len(rr), "n_nontie": len(nt), "status": dict(Counter(r["status"] for r in rr)),
            "S_nontie_tau3000_total": int(sum(r["cost_tau3000"] for r in nt)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "new_steps_median_nontie": float(np.median([r["new_env_steps"] for r in nt])) if nt else None,
            "zero_cost_rate": float(np.mean([r["zero_cost"] for r in rr])),
            "zero_cost_rate_nontie": float(np.mean([r["zero_cost"] for r in nt])) if nt else None,
            "censored_tau3000": int(sum(r["censored_tau3000"] for r in rr)),
            "false_cert": int(sum(r["false_cert"] for r in rr)),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "pad_steps_total": int(sum(r["replay_pad_steps"] for r in rr)),
            "median_wall_s_method": float(np.median([r["wall_clock_s_method"] for r in rr])),
            "median_predictor_sec": float(np.median([r["predictor_sec"] for r in rr]))}
    # instance-level non-tie cost S_i(M, a) = sum_streams sum_{k non-tie} min(T, tau)
    S, n_nt = {}, {}
    for tau in (1500, 3000, 6000):
        for i in seeds:
            for m in METHODS:
                for a in ARMS:
                    sel = [r for r in rows if r["instance"] == i and r["method"] == m and r["arm"] == a
                           and r["gap_layer"] != "tie"]
                    S[(tau, i, m, a)] = sum(r[f"cost_tau{tau}"] for r in sel) if sel else None
                    n_nt[(i, m, a)] = len(sel)

    def lr(tau, i, m, a, b):
        x, y = S.get((tau, i, m, a)), S.get((tau, i, m, b))
        if x is None or y is None:
            return None
        return float(np.log((x + 1) / (y + 1)))

    contrasts = {
        "A1_JPC_full_off": lambda t, i: lr(t, i, "JPC", "full", "off"),
        "B3_full_off": lambda t, i: lr(t, i, "B3", "full", "off"),
        "A2_I_full_off": lambda t, i: (None if lr(t, i, "JPC", "full", "off") is None or
                                       lr(t, i, "B3", "full", "off") is None else
                                       lr(t, i, "JPC", "full", "off") - lr(t, i, "B3", "full", "off")),
        "A3_JPC_ev1_vol": lambda t, i: lr(t, i, "JPC", "ev1", "vol"),
        "JPC_vol_off": lambda t, i: lr(t, i, "JPC", "vol", "off"),
        "JPC_full_vol": lambda t, i: lr(t, i, "JPC", "full", "vol"),
        "JPC_full_ev1": lambda t, i: lr(t, i, "JPC", "full", "ev1"),
        "B3_vol_off": lambda t, i: lr(t, i, "B3", "vol", "off"),
        "B3_ev1_vol": lambda t, i: lr(t, i, "B3", "ev1", "vol"),
        "B3_full_vol": lambda t, i: lr(t, i, "B3", "full", "vol"),
    }
    per_instance = {i: {name: f(TAU, i) for name, f in contrasts.items()} for i in seeds}
    for i in seeds:
        per_instance[i]["n_nontie_problems"] = n_nt[(i, "JPC", "off")]
        per_instance[i]["S"] = {f"{m}|{a}": S[(TAU, i, m, a)] for m in METHODS for a in ARMS}
    stats = {tau: {name: _stats([f(tau, i) for i in seeds]) for name, f in contrasts.items()}
             for tau in (1500, 3000, 6000)}
    # per-layer (near / clear) JPC full/off and I, pooled per instance
    layer = {}
    for lay in ("near", "clear"):
        def lr_l(i, m, a, b, lay=lay):
            sa = [r["cost_tau3000"] for r in rows if r["instance"] == i and r["method"] == m and r["arm"] == a
                  and r["gap_layer"] == lay]
            sb = [r["cost_tau3000"] for r in rows if r["instance"] == i and r["method"] == m and r["arm"] == b
                  and r["gap_layer"] == lay]
            return float(np.log((sum(sa) + 1) / (sum(sb) + 1))) if sa and sb else None
        layer[lay] = {"JPC_full_off": _stats([lr_l(i, "JPC", "full", "off") for i in seeds]),
                      "I_full_off": _stats([None if lr_l(i, "JPC", "full", "off") is None else
                                            lr_l(i, "JPC", "full", "off") - lr_l(i, "B3", "full", "off")
                                            for i in seeds])}
    comp = Counter(r["gap_layer"] for r in rows if r["method"] == "JPC" and r["arm"] == "off")
    tie_rows = [r for r in rows if r["gap_layer"] == "tie"]
    tie = {f"{m}|{a}": {"n": sum(1 for r in tie_rows if r["method"] == m and r["arm"] == a),
                        "status": dict(Counter(r["status"] for r in tie_rows if r["method"] == m and r["arm"] == a)),
                        "new_steps_median": float(np.median([r["new_env_steps"] for r in tie_rows
                                                             if r["method"] == m and r["arm"] == a]))
                        if any(r["method"] == m and r["arm"] == a for r in tie_rows) else None}
           for m in METHODS for a in ARMS}
    sd_I = stats[TAU]["A2_I_full_off"].get("sd")
    power = {"n_instances_dev": len(seeds), "sd_logratio": {k: v.get("sd") for k, v in stats[TAU].items()},
             "A2_mde80_at_n120": stats[TAU]["A2_I_full_off"].get("mde80_n120"),
             "A2_se_at_n120": (sd_I / math.sqrt(120)) if sd_I is not None else None,
             "fallback_cand_r_trigger_mde_gt_0.6": (stats[TAU]["A2_I_full_off"].get("mde80_n120") or 0) > 0.6,
             "note": "pilot: 8 instances x 1 stream x 5 problems -> SD is a magnitude read only, not the lock number"}
    return {"by_cell": by_cell, "per_instance": per_instance, "contrast_stats": stats, "by_layer": layer,
            "tie_near_clear_counts": dict(comp), "tie_layer": tie, "power": power}


def make_plot(per_instance, seeds, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    arms = ["vol", "ev1", "full"]
    w = 0.38
    for j, m in enumerate(METHODS):
        means, ses = [], []
        for a in arms:
            v = []
            for i in seeds:
                Sx, So = per_instance[i]["S"][f"{m}|{a}"], per_instance[i]["S"][f"{m}|off"]
                if Sx is not None and So is not None:
                    v.append(np.log((Sx + 1) / (So + 1)))
            v = np.asarray(v)
            means.append(v.mean() if len(v) else np.nan)
            ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        x = np.arange(len(arms)) + (j - 0.5) * w
        ax.bar(x, means, w, yerr=ses, capsize=3, label=m, color=["#3b6ea8", "#c8793a"][j])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([f"{a} / off" for a in arms])
    ax.set_ylabel("log((S_arm+1)/(S_off+1)), non-tie, tau=3000")
    ax.set_title(f"NL-R0 dev reuse ablation (n={len(seeds)} instances, mean +/- SE)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_logratio_by_arm.png", dpi=150)
    plt.close(fig)


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = MODES[a.mode]
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    for d in (out_dir, out_dir / "parts", out_dir / "samples" / "ledgers"):
        d.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    try:
        progress(0, 1, "jtables")
        seeds, repl, qinfo, tinfo = build_tables(cfg["seeds"])
        print("tables", tinfo, "replacements", repl, flush=True)
        jobs = [(s, st, m, cfg["n_problems"]) for s in seeds for st in cfg["streams"] for m in METHODS]
        jobs.sort(key=lambda j: (j[2] != "B3", j))          # B3 jobs are slower: start them first
        t_runs = time.perf_counter()
        joblog = run_jobs(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        rows, errs, snap_errs, n_snap, wa = collect(jobs, out_dir)
        au = audit(rows)
        an = analyse(rows, seeds, cfg["streams"])
        try:
            make_plot(an["per_instance"], seeds, out_dir)
        except Exception as e:  # noqa: BLE001
            print("plot failed", e, flush=True)
        n_led = len(list((out_dir / "samples" / "ledgers").glob("*.npz")))
        expected = len(seeds) * len(cfg["streams"]) * cfg["n_problems"] * len(METHODS) * len(ARMS)
        I_stats = an["contrast_stats"][TAU]["A2_I_full_off"]
        crit = {"zero_crashes": len(errs) == 0 and len(rows) == expected and not any("fatal" in j for j in joblog),
                "billing_mismatch_zero_all_four_arms": all(v == 0 for v in au["billing_mismatch_by_arm"].values())
                and all(any(r["arm"] == arm for r in rows) for arm in ARMS),
                "replay_never_billed_and_auditable": au["n_bad"] == 0 and au["replay_billed_total"] == 0,
                "interaction_I_computable_nontie": I_stats.get("n", 0) >= 2 and I_stats.get("mean") is not None}
        if a.mode == "full":
            crit["sd_and_mde_written"] = an["power"]["A2_mde80_at_n120"] is not None
        aux = {"ledger_snapshots_complete": n_led == expected and not snap_errs,
               "predictor_write_ahead_ok": wa["write_ahead_ok"], "n_ledger_snapshots": n_led,
               "n_expected_runs": expected, "false_cert_total": int(sum(r["false_cert"] for r in rows))}
        passed = all(crit.values())
        summary = {"task": TASK, "mode": a.mode, "started_at": started, "finished_at": datetime.now().isoformat(),
                   "note": "并发运行（与 r3_setup_lin_stream / r3_p3 / r3_p5 共享 CPU 与 4090）；计时偏高。"
                           "pilot 只验证管线、读量级，GO 不取决于效应方向",
                   "design": {"layer": "NL-R0 (E1-NL-S G_1, realisable)", "seeds": seeds,
                              "seed_replacements": repl, "streams": cfg["streams"], "n_problems": cfg["n_problems"],
                              "methods": METHODS, "arms": ARMS, "eps": EPS, "delta": DELTA, "tmax": TMAX,
                              "tau_primary": TAU, "kind": "quota", "crn": True, "n_runs": len(rows)},
                   "quota": {str(s): v for s, v in qinfo.items()}, "jtables": tinfo, "runs_sec": runs_sec,
                   "audit": au, "write_ahead": wa, "snapshot_errors": len(snap_errs),
                   "n_run_errors": len(errs), **an, "pass_criteria": crit, "aux_checks": aux, "passed": passed,
                   "go_no_go": "GO" if passed else "NO_GO", "code_sha256": code_sha(),
                   "wall_clock_s": time.perf_counter() - t0}
        summary["metrics"] = {
            "stream_steps_by_arm": {k: v["new_steps_total"] for k, v in an["by_cell"].items()},
            "interaction_I": I_stats, "log_ratio_sd": an["power"]["sd_logratio"],
            "zero_cost_rate": {k: v["zero_cost_rate"] for k, v in an["by_cell"].items()},
            "billing_mismatch": au["billing_mismatch_by_arm"], "tie_near_clear_counts": an["tie_near_clear_counts"],
            "A2_mde80_at_n120": an["power"]["A2_mde80_at_n120"]}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(json.dumps(crit), summary["go_no_go"], flush=True)
        progress(1, 1, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success" if passed else "failed", f"{summary['go_no_go']} {crit}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
