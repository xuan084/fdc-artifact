"""r3_prereg_lock (governance): write plan/prereg_lock.json version 3 before any evaluation seed is generated.

pilot: status=provisional.  Thresholds / constants / operationalisations are copied unchanged from the planner draft
(and checked against idea/hypotheses.md: sha256 + numeric-token cross-check); the lock only ADDS numeric estimates:
P1/P2 SD and MDE, P3/P4/P5 gate readouts, provisional frozen items, and a full wall-clock projection for every
evaluation task with the downscale choice that the projection implies.  A small timing-only smoke on dev seeds
734-735 (the block reserved for evaluation-task smoke/timing, methodology section 1) measures the cells that no P1-P5
pilot timed: offset-null, no-tie and low-overlap streams for {JPC,B3} x {off,ev1,full}, and LR-chi2-grid / B8 x
{off,full} over full 15-problem streams.  No effect sizes are read from that smoke.

full: requires the P1-P5 full (dev-freeze) summaries under exp/results/full/; writes the frozen items, verifies the
thresholds again, git-commits code + lock (status=locked, sha256), records the commit hash, and only then writes the
evaluation manifest (eval_manifest + eval_manifest_sha256) and re-hashes.  Refuses to overwrite a locked file.
Evaluation seeds (>= 10000) are never instantiated here.

Usage: run_r3_prereg_lock.py --mode {pilot,full} [--workers 4] [--skip-smoke]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
TASK = "r3_prereg_lock"
RES_ROOT = WS / "exp" / "results"
LOCK = WS / "plan" / "prereg_lock.json"
DRAFT_ARCHIVE = WS / "plan" / "history" / "round3" / "prereg_lock_draft_planner.json"
HYP = WS / "idea" / "hypotheses.md"
TASK_PLAN = WS / "plan" / "task_plan.json"
WORKERS_PER_TASK = 4
Z975, Z80 = 1.959963984540054, 0.8416212335729143
LOG08, LOG06 = abs(math.log(0.8)), abs(math.log(0.6))
SMOKE_SEEDS = [734, 735]
SAFETY_CONCURRENT = 1.2      # pilot timings measured while 3 other tasks shared the 20 cores
SAFETY_UNLOADED = 1.35       # timings measured with no other task running (full stage runs 4 tasks at once)
NO_DOWNSCALE_MAX = 45.0      # projected minutes at or below which the full design is kept
HARD_MAX = 55.0              # pass criterion

CODE_FILES = [
    "dsswm/evidence/reuse_switch.py", "dsswm/evidence/orth_design.py", "dsswm/baselines/switched_nl.py",
    "dsswm/baselines/lin_rage.py", "dsswm/streams/gap_quota.py", "dsswm/streams/lin_stream.py",
    "dsswm/streams/r3_harness.py", "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py",
    "dsswm/mechanism/oracle_check.py", "dsswm/stats/mh.py", "dsswm/stats/auc.py", "dsswm/stats/prereg.py",
    "run_r3_p1_reuse_ablation_nl.py", "run_r3_p2_lin_stream_smoke.py", "run_r3_p3_orth_replay_feasibility.py",
    "run_r3_p4_t0_mechanism_gate.py", "run_r3_p5_hazard_zone_sampler.py", "run_r3_prereg_lock.py",
    # evaluation code frozen by the full lock (methodology control-plane list + every round-3 evaluation script)
    "dsswm/acquire/multi_regime_probe.py", "dsswm/streams/offgrid.py",
    "run_r3_setup_reuse_switch.py", "run_r3_setup_mechanism.py", "run_r3_setup_lin_stream.py",
    "run_r3_nl_main.py", "run_r3_nl_extra.py", "run_r3_lin.py", "run_r3_static.py", "run_r3_hazard.py",
    "run_r3_controls_zero_effect.py", "run_r3_controls_streams.py", "run_r3_heldout_misspec.py",
    "run_r3_x1_sampler_misspec.py", "run_r3_replicate_hd1b_hd2.py", "run_r3_replicate_hh1.py",
    "run_r3_replica_check.py", "run_r3_analysis_aggregate.py"]


# ------------------------------------------------------------------------------------------------ bookkeeping
def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


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
    out = {f: sha_file(HERE / f) for f in CODE_FILES if (HERE / f).exists()}
    for f in sorted((HERE / "replica").glob("**/*.py")):
        out[str(f.relative_to(HERE))] = sha_file(f)
    out["_combined"] = hashlib.sha256("".join(out[f] for f in sorted(out)).encode()).hexdigest()
    return out


def load(p: Path):
    return json.loads(p.read_text())


def rows(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------------------------------------ thresholds
def threshold_check(draft: dict) -> dict:
    """hypotheses.md must be byte-identical to the draft's hash, and every numeric token of every locked threshold
    must occur in hypotheses.md (the draft thresholds are English transcriptions of the Chinese tables)."""
    text = HYP.read_text()
    h = sha_file(HYP)
    norm = text.replace("−", "-").replace("⁻", "^-").replace("ε", "eps").replace("δ", "delta")
    missing = {}
    for k, s in draft["thresholds_verbatim"].items():
        toks = set(re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", s))
        toks -= {"1", "2", "95"} if k not in ("C1", "HD1b") else {"95"}
        miss = []
        for t in sorted(toks):
            cands = {t, t.replace("e-9", "e-9"), t.replace("1e-9", "1e-9")}
            if t == "1e-9":
                cands |= {"1e-9", "1e^-9"}
            if not any(c in norm for c in cands):
                miss.append(t)
        if miss:
            missing[k] = miss
    return {"hypotheses_md_sha256": h, "draft_sha256": draft["hypotheses_md"]["sha256"],
            "hash_match": h == draft["hypotheses_md"]["sha256"],
            "thresholds_sha256": sha_obj(draft["thresholds_verbatim"]),
            "numeric_tokens_missing": missing, "numeric_token_check_pass": not missing}


# ------------------------------------------------------------------------------------------------ power
def mde(sd, n, margin):
    """Smallest true |effect| giving 80% power for 'CI upper < -margin' (two-sided 95% CI): margin + 2.80 SD/sqrt n.
    Reproduces methodology 5.5 (SD 0.8-1.0, n=120 -> 0.45-0.50)."""
    se = sd / math.sqrt(n)
    return {"sd": sd, "n": n, "se": se, "mde80": margin + (Z975 + Z80) * se, "mde80_no_margin": (Z975 + Z80) * se,
            "margin": margin}


def n_for_mde(sd, target, margin):
    d = target - margin
    return None if d <= 0 else int(math.ceil(((Z975 + Z80) * sd / d) ** 2))


def power_block(p1, p2, mode):
    cs = p1["contrast_stats"]["3000"]
    out = {"source": {"NL": f"{mode}/r3_p1_reuse_ablation_nl summary.contrast_stats[tau=3000]",
                      "Lin": f"{mode}/r3_p2_lin_stream_smoke summary.instance_log_ratios"},
           "definition": "MDE80 = margin + (z_.975 + z_.80) * SD / sqrt(n); margin = |log 0.8| for A2/A3, |log 0.6| "
                         "for A1; n = 120 (NL-R0), 48 (E1-Lin). fallback_cand_r iff NL-R0 A2 MDE80 > 0.6."}
    nl = {}
    for key, name, margin in (("A1_JPC_full_off", "A1", LOG06), ("A2_I_full_off", "A2", LOG08),
                              ("A3_JPC_ev1_vol", "A3", LOG08), ("B3_full_off", "B3_full_off", LOG08)):
        s = cs[key]
        nl[name] = {"pilot_mean": s["mean"], "pilot_median": s["median"], "pilot_n": s["n"], **mde(s["sd"], 120, margin),
                    "n_needed_for_mde_0.6": n_for_mde(s["sd"], 0.6, margin)}
    lin = {}
    lr = p2["instance_log_ratios"]
    for key, name, margin in (("JPC_full_off", "A1", LOG06), ("I_full_off_A2", "A2", LOG08),
                              ("RAGE_full_off", "RAGE_full_off", LOG08)):
        s = lr.get(key)
        if s is None:
            continue
        lin[name] = {"pilot_mean": s["mean"], "pilot_n": s["n"], **mde(s["sd"], 48, margin),
                     "p2_reported_mde80_n120_no_margin": s.get("mde80_n120"),
                     "note": "P2 reported MDE at n=120 without the log-0.8 margin; E1-Lin has 48 instances"}
    a2 = nl["A2"]["mde80"]
    out.update({"NL_R0": nl, "E1_Lin": lin,
                "log_ratio_sd_nl": {k: v["sd"] for k, v in nl.items()},
                "log_ratio_sd_lin": {k: v["sd"] for k, v in lin.items()},
                "A2_MDE": {"NL_R0": a2, "E1_Lin": lin.get("A2", {}).get("mde80")},
                "fallback_cand_r": {"rule": "A2 MDE80 at n=120 (NL-R0) > 0.6", "value": a2, "triggered": a2 > 0.6,
                                    "binding": mode == "full"}})
    if mode == "pilot":
        out["caveat"] = ("pilot SD comes from 8 instances x 1 stream x 5 problems (2-4 non-tie problems per instance); "
                         "small S makes log((S+1)/(S'+1)) heavy-tailed. Full uses 20 instances x 3 streams x 15 "
                         "problems; the binding SD/MDE/fallback come from P1 full only.")
    return out


# ------------------------------------------------------------------------------------------------ timing smoke
def timing_smoke(out_dir, workers):
    from joblib import Parallel, delayed
    import run_r3_setup_reuse_switch as S
    jobs = [(s, k, m, ["off", "ev1", "full"], 15, 6000) for s in SMOKE_SEEDS
            for k in ("offset_null", "no_tie", "low_overlap") for m in ("JPC", "B3")]
    jobs += [(s, "quota", m, ["off", "full"], 15, 6000) for s in SMOKE_SEEDS for m in ("B8", "LR-chi2-grid")]
    jobs.sort(key=lambda j: (j[2] != "B3", j[1] != "offset_null", j[2] != "B8"))   # longest first
    t0 = time.perf_counter()
    res = Parallel(n_jobs=workers, backend="loky")(delayed(S.run_job)(*j) for j in jobs)
    sec = time.perf_counter() - t0
    rr = [r for x, _, _ in res for r in x]
    errs = [e for _, _, ee in res for e in ee]
    with open(out_dir / "timing_smoke.jsonl", "w") as f:
        for r in rr:
            f.write(json.dumps({k: r.get(k) for k in ("instance", "stream", "kind", "method", "arm", "problem_index",
                                                       "gap_layer", "status", "new_env_steps", "billing_ok",
                                                       "wall_clock_s", "cell_sec")}, default=str) + "\n")
    cells = defaultdict(list)
    for r in rr:
        cells[(r["kind"], r["method"], r["arm"])].append(r)
    per = {}
    for (k, m, a), v in sorted(cells.items()):
        per[f"{k}|{m}|{a}"] = {"n": len(v), "sec_per_problem_mean": float(np.mean([x["wall_clock_s"] for x in v])),
                               "sec_per_problem_max": float(np.max([x["wall_clock_s"] for x in v])),
                               "steps_mean": float(np.mean([x["new_env_steps"] for x in v])),
                               "need_data": int(sum(x["status"] != "CERTIFIED" for x in v))}
    return {"seeds": SMOKE_SEEDS, "stream": 0, "n_runs": len(rr), "wall_s": sec, "crashes": len(errs),
            "errors_head": errs[:3], "billing_mismatch": int(sum(not r["billing_ok"] for r in rr)),
            "per_cell": per, "note": "timing only (dev 734-735 = eval-task smoke/timing block); no effect sizes read; "
                                     "measured with no other task running -> safety factor 1.35"}


# ------------------------------------------------------------------------------------------------ cost model
def cost_model(p1dir, p2, p3, p4, p5dir, setup_rs, setup_lin, smoke):
    """Per-problem CPU-seconds per (setting, method, arm) with provenance; returns dict key -> (sec, source, safety)."""
    C = {}
    r1 = rows(p1dir / "results.jsonl")
    cell = defaultdict(dict)
    for r in r1:
        cell[(r["method"], r["arm"])][(r["instance"], r["stream"])] = r["cell_sec"]
    npb = 5
    for (m, a), d in cell.items():
        C[("quota", m, a)] = (float(np.mean(list(d.values()))) / npb, "P1 pilot cell_sec / 5 problems", SAFETY_CONCURRENT)
    jpc_orth = float(np.mean([p["job_sec"] for p in p3["parts"]])) / 15
    C[("quota", "JPC", "orth")] = (jpc_orth, "P3 pilot job_sec / 15 (sibling + FW + run)", SAFETY_CONCURRENT)
    C[("quota", "B3", "orth")] = (jpc_orth * C[("quota", "B3", "vol")][0] / C[("quota", "JPC", "vol")][0],
                                  "extrapolated: JPC orth x (B3 vol / JPC vol); not measured", SAFETY_CONCURRENT)
    sm = (smoke or {}).get("per_cell", {})
    rs = rows(RES_ROOT / "pilots" / "r3_setup_reuse_switch" / "results.jsonl")
    rs_cells = defaultdict(list)
    for r in rs:
        rs_cells[(r["kind"], r["method"], r["arm"])].append(r["wall_clock_s"])
    for kind in ("offset_null", "no_tie", "low_overlap", "quota"):
        for m in ("JPC", "B3", "B8", "LR-chi2-grid"):
            for a in ("off", "ev1", "full"):
                key = f"{kind}|{m}|{a}"
                if key in sm and (kind != "quota" or m in ("B8", "LR-chi2-grid")):
                    C[(kind, m, a)] = (sm[key]["sec_per_problem_mean"], "lock timing smoke dev 734-735 x 15 problems",
                                       SAFETY_UNLOADED)
                elif (kind, m, a) not in C and (kind, m, a) in rs_cells:
                    C[(kind, m, a)] = (float(np.mean(rs_cells[(kind, m, a)])), "setup_reuse_switch smoke (4 problems)",
                                       SAFETY_CONCURRENT)
    for m in ("JPC", "B3"):
        if ("offset_null", m, "ev1") not in C and ("offset_null", m, "full") in C:
            C[("offset_null", m, "ev1")] = (C[("offset_null", m, "full")][0], "fallback: offset-null full cost",
                                            SAFETY_CONCURRENT)
    for kind in ("no_tie", "low_overlap"):            # fall back to quota costs if the smoke was skipped
        for m in ("JPC", "B3"):
            for a in ("off", "ev1", "full"):
                C.setdefault((kind, m, a), (C[("quota", m, a)][0], "fallback: quota cost (not measured)",
                                            SAFETY_CONCURRENT))
    for m in ("B8", "LR-chi2-grid"):
        for a in ("off", "full"):
            if ("quota", m, a) not in C:
                C[("quota", m, a)] = (float(np.mean(rs_cells[("quota", m, a)])), "setup smoke (4 problems)",
                                      SAFETY_CONCURRENT)
    # Lin (P2 median cell sec per 15-problem stream cell; setup_lin smoke means for the rest)
    for k, v in p2["budget"]["median_cell_s_by_cell"].items():
        _, m, a = k.split("|")
        C[("lin", m, a)] = (v / 15, "P2 pilot median cell_s / 15", SAFETY_CONCURRENT)
    lsm = setup_lin["smoke"]["by_cell"]
    for k, v in lsm.items():
        lay, m, a = k.split("|")
        key = ("lin" if lay == "E1-Lin" else "lin_static", m, a)
        if key not in C:
            C[key] = (max(v["median_wall_s"], 0.02) * 2.0, "setup_lin smoke median_wall_s x2 (mean/median guard)",
                      SAFETY_CONCURRENT)
    lin_fw = []
    for r in rows(RES_ROOT / "pilots" / "r3_p3_orth_replay_feasibility" / "lin_designs.jsonl"):
        if r.get("fw_sec") is not None:
            lin_fw.append(r["fw_sec"])
    lfw = float(np.mean(lin_fw)) if lin_fw else 0.5
    for m in ("JPC-Lin", "RAGE"):
        C[("lin", m, "orth")] = (lfw + C[("lin", m, "ev1")][0], "P3 Lin fw_sec mean + ev1 run cost", SAFETY_CONCURRENT)
    # static twins (P4 dev runs: JPC full on static g=1, 15 problems per job, incl. predictors)
    st_jobs = [j["sec"] for j in p4["jobs"] if j["tag"].endswith("static_g1")]
    m1_jobs = [j["sec"] for j in p4["jobs"] if j["tag"].endswith("m1_2eps")]
    base = float(np.mean(st_jobs)) / 15
    q = {a: C[("quota", "JPC", a)][0] for a in ("off", "ev1", "full")}
    sps = sum(r["wall_clock_s"] for r in r1 if r["method"] == "JPC") / max(1, sum(r["new_env_steps"] for r in r1
                                                                                 if r["method"] == "JPC"))
    mult = {"off": q["off"] / q["full"], "ev1": q["ev1"] / q["full"], "full": 1.0}
    for a in ("off", "ev1", "full"):
        C[("static", "JPC", a)] = (base * mult[a], f"P4 static_g1 JPC-full job/15 x P1 JPC {a}/full ratio",
                                   SAFETY_CONCURRENT)
    for a, extra in (("ev20", 20), ("ev200", 200), ("probe20", 20)):
        C[("static", "JPC", a)] = (base * mult["ev1"] + extra * sps, f"static ev1 + {extra} forced steps x {sps:.4f} s/step",
                                   SAFETY_CONCURRENT)
    C[("static", "JPC_dyn_ref", "full")] = (q["full"] * (1 + 1 + 3.4) / 3,
                                            "P1 JPC full x mean class-size factor (g=0.5 uses G_ext: x3.4 assumed)",
                                            SAFETY_CONCURRENT)
    m1 = float(np.mean(m1_jobs)) / 15
    C[("misspec", "JPC", "full")] = (m1, "P4 m1_2eps JPC-full job/15", SAFETY_CONCURRENT)
    C[("misspec", "JPC", "ev1")] = (m1 * mult["ev1"], "P4 m1 x P1 ev1/full", SAFETY_CONCURRENT)
    # hazard (P5 runs: per gap layer; R1 uniform reweighted to the 1/3 quota, R1-adv as observed tie-rich mix)
    r5 = rows(p5dir / "results.jsonl")
    for m in ("JPC", "B3"):
        v = [r["wall_clock_s"] for r in r5 if r["method"] == m]
        lay = {l: [r["wall_clock_s"] for r in r5 if r["method"] == m and r["gap_layer"] == l]
               for l in ("tie", "near", "clear")}
        lm = {l: (float(np.mean(x)) if len(x) >= 3 else None) for l, x in lay.items()}
        if lm["near"] is None:
            lm["near"] = lm["tie"]
        if lm["clear"] is None:
            lm["clear"] = lm["near"]
        C[("R1_adv", m, "full")] = (float(np.mean(v)), "P5 pilot mean (tie-rich placement mix)", SAFETY_CONCURRENT)
        C[("R1", m, "full")] = (float(np.mean(list(lm.values()))), "P5 per-layer means reweighted to 1/3 quota",
                                SAFETY_CONCURRENT)
    return C, {"jpc_sec_per_step": sps}


def project(C, spec):
    """spec: list of (setting, method, arm, n_problem_slots) + fixed minutes; returns minutes."""
    cpu = 0.0
    lines = []
    for setting, m, a, n in spec["cells"]:
        sec, src, saf = C[(setting, m, a)]
        cpu_s = sec * n * saf
        cpu += cpu_s
        lines.append({"cell": f"{setting}|{m}|{a}", "problem_slots": n, "sec_per_problem": sec, "safety": saf,
                      "cpu_s": cpu_s, "source": src})
    minutes = cpu / (WORKERS_PER_TASK * 60) + spec.get("fixed_min", 0.0)
    return minutes, lines


def eval_specs():
    """Designs per evaluation task (one chunk = the largest chunk of the task family), and their preset downscales
    (methodology section 6, in the listed order)."""
    nl_slots = 15 * 3 * 15
    main_arms = [("quota", m, a) for m in ("JPC", "B3") for a in ("off", "vol", "orth", "ev1", "full")]

    def nl_main(orth_streams=3):
        cells = [(s, m, a, nl_slots if a != "orth" else 15 * orth_streams * 15) for s, m, a in main_arms]
        return {"cells": cells, "fixed_min": 1.0}

    def nl_extra(b8_off_streams=3):
        n = 40 * 3 * 15
        return {"cells": [("quota", "LR-chi2-grid", "off", n), ("quota", "LR-chi2-grid", "full", n),
                          ("quota", "B8", "off", 40 * b8_off_streams * 15), ("quota", "B8", "full", n)],
                "fixed_min": 1.0}

    def lin(streams12_core=True, streams12_all=False):
        n0 = 16 * 15
        arms = [(m, a) for m in ("JPC-Lin", "RAGE") for a in ("off", "vol", "orth", "ev1", "full")]
        arms += [("XY-static", "off"), ("G-opt", "full"), ("B1eb", "full")]
        cells = [("lin", m, a, n0) for m, a in arms]
        cells += [("lin_static", "JPC-Lin", a, n0) for a in ("off", "ev1", "full")]
        if streams12_all:
            cells += [("lin", m, a, n0 * 2) for m, a in arms]
            cells += [("lin_static", "JPC-Lin", a, n0 * 2) for a in ("off", "ev1", "full")]
        elif streams12_core:
            cells += [("lin", m, a, n0 * 2) for m in ("JPC-Lin", "RAGE") for a in ("off", "full")]
        return {"cells": cells, "fixed_min": 0.5}

    def static():
        n = 12 * 3 * 15 * 3
        cells = [("static", "JPC", a, n) for a in ("off", "ev1", "ev20", "ev200", "full", "probe20")]
        cells += [("static", "JPC_dyn_ref", "full", n)]
        return {"cells": cells, "fixed_min": 1.0}

    def hazard(b3_streams=3):
        n = 32 * 3 * 15
        cells = [("R1", "JPC", "full", n), ("R1_adv", "JPC", "full", n),
                 ("R1", "B3", "full", 32 * b3_streams * 15), ("R1_adv", "B3", "full", 32 * b3_streams * 15)]
        return {"cells": cells, "fixed_min": 3.0}     # + rejection placement on eval seeds (GPU mu_flip MC)

    def zero_effect(n_inst=48, b3_streams=3):
        n = n_inst * 3 * 15
        cells = [("offset_null", "JPC", a, n) for a in ("off", "ev1", "full")]
        cells += [("offset_null", "B3", a, n_inst * b3_streams * 15) for a in ("off", "ev1", "full")]
        return {"cells": cells, "fixed_min": 0.5}

    def streams(n_inst=48, n_streams=3):
        n = n_inst * n_streams * 15
        cells = [(k, m, a, n) for k in ("no_tie", "low_overlap") for m in ("JPC", "B3") for a in ("off", "ev1", "full")]
        return {"cells": cells, "fixed_min": 1.0}

    def heldout():
        n = 48 * 2 * 15 * 3
        return {"cells": [("misspec", "JPC", "full", n), ("misspec", "JPC", "ev1", n),
                          ("misspec", "JPC", "full", 16 * 15), ("misspec", "JPC", "ev1", 16 * 15)],
                "fixed_min": 3.0}     # replica-side J tables for m1r / alias

    def x1():
        return {"cells": [("misspec", "JPC", "full", 24 * 15 * 2 * 3)], "fixed_min": 1.0}

    return {
        "r3_nl_main_[a-h]": {"n_chunks": 8, "design": nl_main, "presets": [("orth_stream0_only", {"orth_streams": 1})]},
        "r3_nl_extra_[a-c]": {"n_chunks": 3, "design": nl_extra, "presets": [("B8_off_stream0_only",
                                                                              {"b8_off_streams": 1})]},
        "r3_lin_[a-c]": {"n_chunks": 3, "design": lin, "presets": [],
                         "note": "plan baseline = stream 0 all arms + streams 1-2 core arms {JPC-Lin,RAGE}x{off,full}"},
        "r3_static_[a-d]": {"n_chunks": 4, "design": static, "presets": [],
                            "contingency": "ev200 arm on stream 0 only (not a methodology preset; decide at full lock)"},
        "r3_hazard_[a-b]": {"n_chunks": 2, "design": hazard, "presets": [],
                            "contingency": "B3 reference on stream 0 only (not a methodology preset)"},
        "r3_controls_zero_effect": {"n_chunks": 1, "design": zero_effect,
                                    "presets": [],
                                    "nonpreset": [("split_2_chunks_24_inst", {"n_inst": 24}),
                                                  ("split_2_chunks_24_inst+B3_stream0", {"n_inst": 24,
                                                                                         "b3_streams": 1}),
                                                  ("split_3_chunks_16_inst", {"n_inst": 16}),
                                                  ("B3_stream0_only", {"b3_streams": 1})]},
        "r3_controls_streams": {"n_chunks": 1, "design": streams, "presets": [],
                                "nonpreset": [("split_2_chunks_24_inst", {"n_inst": 24}),
                                              ("stream0_only", {"n_streams": 1})]},
        "r3_heldout_misspec": {"n_chunks": 1, "design": heldout, "presets": []},
        "r3_x1_sampler_misspec": {"n_chunks": 1, "design": x1, "presets": [],
                                  "note": "Neyman sampler cost assumed = DDA (JPC full on m1)"},
    }


FIXED_ESTIMATES = {
    "r3_replicate_hd1b_hd2": {"projected_min": 7.7 + 35.0,
                              "source": "HD1b round-2 pilot projection 7.7 min (timing_projection) + HD2 reduced config "
                                        "planner estimate 35 min; not re-measured in round 3"},
    "r3_replicate_hh1": {"projected_min": 45.0, "source": "planner estimate from round-2 hh1 pilot; not re-measured"},
    "r3_replica_check": {"projected_min": 25.0, "source": "planner estimate (300 problems, replica J recompute)"},
}


def projections(C):
    out = {}
    for tid, spec in eval_specs().items():
        base_min, base_lines = project(C, spec["design"]())
        chosen, applied, kw = base_min, [], {}
        lines = base_lines
        for name, k in spec["presets"]:
            if chosen <= NO_DOWNSCALE_MAX:
                break
            kw.update(k)
            chosen, lines = project(C, spec["design"](**kw))
            applied.append(name)
        nonpreset = []
        if chosen > HARD_MAX:
            for name, k in spec.get("nonpreset", []):
                m, ln = project(C, spec["design"](**{**kw, **k}))
                nonpreset.append({"option": name, "projected_min": m})
            ok = [o for o in nonpreset if o["projected_min"] <= HARD_MAX]     # first in preference order
            if ok:
                pick = ok[0]
                applied.append(pick["option"] + " [NON-PRESET: requires task_plan edit before the full stage]")
                kk = dict(spec["nonpreset"])[pick["option"]]
                chosen, lines = project(C, spec["design"](**{**kw, **kk}))
        out[tid] = {"n_chunks": spec["n_chunks"], "projected_min_full_design": base_min,
                    "downscale_applied": applied, "projected_min_chosen": chosen,
                    "status": ("ok" if chosen <= NO_DOWNSCALE_MAX else "tight" if chosen <= HARD_MAX else "OVER"),
                    "nonpreset_options_evaluated": nonpreset, "cells": lines,
                    **({"note": spec["note"]} if "note" in spec else {}),
                    **({"contingency": spec["contingency"]} if "contingency" in spec else {})}
    for tid, v in FIXED_ESTIMATES.items():
        out[tid] = {"n_chunks": 1, "projected_min_full_design": v["projected_min"], "downscale_applied": [],
                    "projected_min_chosen": v["projected_min"],
                    "status": "ok" if v["projected_min"] <= NO_DOWNSCALE_MAX else "tight", "source": v["source"],
                    "unmeasured": True}
    return out


# ------------------------------------------------------------------------------------------------ gate readouts
def gate_block(p1, p2, p3, p4, p5, mode):
    t0 = p4["T0"]
    ii = t0["ii_rem_bar_over_eps"]
    iv_med = p4["metrics"]["phi_perp_median"]
    rate = p3["nl"]["constructible_rate_nontrivial"]
    edges = p5["bin_edges_provisional"]
    binding = mode == "full"
    return {
        "billing_gate": {"P1": p1["audit"]["billing_mismatch_by_arm"], "P2": p2["evidence_boundary"]["billing_mismatch"],
                         "P3_executed_orth": p3["jpc_runs"]["billing_mismatch_executed"],
                         "P5": "0 (pass_criteria.billing_mismatch_zero=%s)" % p5["pass_criteria"]["billing_mismatch_zero"]},
        "T0": {"i_identity_err_max": t0["i_identity_err_max"], "i_pass": t0["i_pass"],
               "ii_rem_bar_over_eps_p90_eta_le_2eps": ii["p90_eta_le_2eps"], "ii_n": ii["n_eta_le_2eps"],
               "ii_pass": ii["p90_eta_le_2eps"] <= 0.5,
               "iii_pass": p4["hard_gates"]["T0_iii_counts_theta_hat_retrievable"],
               "iv_phi_perp_median": iv_med, "iv_clause_triggered": iv_med > 0.3},
        "T0_scope": {"value": "eta <= eps" if ii["p90_eta_le_2eps"] > 0.5 else "eta <= 2 eps",
                     "rule": "hypotheses.md T0: (ii) fails -> Thm 3 scope narrows to eta <= eps, Q family tested only "
                             "in that scope", "binding": binding},
        "phi_perp_clause": {"write": "定向补证无成本优势" if iv_med > 0.3 else None, "binding": binding},
        "orth_confirmatory": {"value": rate >= 0.5, "constructible_rate_nl_stream0": rate,
                              "constructible_rate_nl_stream1_supplement": p3["nl_supplement_streams"][
                                  "constructible_rate_nontrivial"],
                              "constructible_rate_lin": p3["lin"]["constructible_rate"],
                              "rule": "A3o descriptive if constructible rate < 0.5 (hypotheses.md A3o)",
                              "binding": binding,
                              "definitions": p3["definitions"],
                              "caveat": "trace matching holds in expectation under theta_hat only (realised mean "
                                        "err %.3f, max %.3f); Markov (not stationary) policy vertices"
                                        % (p3["nl"]["realized_trace_err"]["mean"], p3["nl"]["realized_trace_err"]["max"])},
        "predictor_variant": {"value": p4["variant"]["choice"], "auc_S1_dev": p4["variant"]["auc_S1"],
                              "auc_S2_dev": p4["variant"]["auc_S2"], "variant_tie_rule": "S2",
                              "binding": binding,
                              "note": "pilot preview from 87 events / 8 positives / 6 clusters; binding choice on P4 "
                                      "full (720-739)"},
        "Q1_dev_signal": {"pooled": p4["Q1_exploratory"]["pooled_static_m1"].get("S1"),
                          "note": "exploratory dev readout; trivial gap baseline AUC 0.93 > S1 0.66 / S2 0.72 -- "
                                  "pre-announced risk for Q1; not evidence"},
        "cand_o_gate": p4["cand_o_gate"], "cand_l_GR_gate": {"pass": False, **p4["cand_l_GR_gate"]},
        "r1adv": {"acceptance": p5["acceptance"]["primary_mu_flip_ge_1_over_M"],
                  "directed_construction_triggered": p5["directed_construction_triggered"],
                  "bin_edges": {"mu_flip": edges["mu_flip_tertiles"], "lambda_n0_S1": edges["lambda_n0_S1_tertiles"],
                                "S2_n0": edges["S2_n0_tertiles"], "rule": edges["rule"], "binding": binding},
                  "primary_ex_ante_caliber": {"value": "mu_flip", "secondary": "Lambda_hat_perp(n0)",
                                              "reason": "placement/acceptance rule of the sampler is defined on "
                                                        "mu_flip (P5 design, written before its run); both binnings "
                                                        "are reported in eval", "binding": binding},
                  "nontie_post_bin_structural_gap": "for gap >= eps, eta_arg/eps in {0} U [1,inf): the [0.5,1) column "
                                                    "of the 3x3 table can only be filled by tie problems",
                  "cell_fill": {"decision": "(b) declare under-filled cells (mu_flip low bin x {[0.5,1), [1,inf)}) as "
                                            "descriptive; no threshold change",
                                "projection": "low-bin x [1,inf) needs ~329 instances at 5/bin/instance vs 64 budget",
                                "binding": binding},
                  "two_flip_calibers_disagree": "g_circ == nearest in 0/135 dev runs; report both"},
        "T_max_Lin": {"value": 20000, "tau_Lin": 20000,
                      "readout": p2["tmax_study"], "mechanical_rule_value": p2["lock_recommendation"]["T_max_Lin"],
                      "reason": "certified non-tie max 4798 sits at the 5000 boundary; orth / B1eb / streams 1-2 "
                                "untimed in P2; cost is not a constraint (Lin chunk < 10 min) -> keep methodology "
                                "default 20000", "binding": binding},
        "lin_streams": {"value": {"stream0": "all arms", "streams_1_2": "core {JPC-Lin,RAGE} x {off,full}"},
                        "reason": "P2 timing: 16 instances x 3 streams of core arms costs < 5 min per chunk",
                        "recommendation_for_full_lock": "extending P-family arms {vol, ev1, orth} to streams 1-2 "
                                                        "costs a few minutes per chunk and aligns the Lin A3 endpoint "
                                                        "with the 3-stream S definition (planner decision)",
                        "binding": binding},
        "Lin_A2_direction_dev": {"mean_I": p2["instance_log_ratios"]["I_full_off_A2"]["mean"],
                                 "note": "dev pilot direction opposite to A2 prediction (8/10 instances > 0); "
                                         "pre-announced; not evidence; threshold unchanged"},
    }


# ------------------------------------------------------------------------------------------------ FULL-mode builders
# The full lock re-derives every frozen item from the P1-P5 FULL (dev-freeze, 720-760) summaries and replaces the
# pilot cost-model projection by the per-chunk projections MEASURED by the evaluation-task pilots (dev 734-760, the
# evaluation scripts themselves).  Nothing here changes a threshold: thresholds_verbatim / constants /
# operationalisations / seed_blocks are asserted identical to the planner draft in build().
EVAL_PILOT_FAMILIES = {
    "r3_nl_main_[a-h]": [f"r3_nl_main_{c}" for c in "abcdefgh"],
    "r3_nl_extra_[a-c]": [f"r3_nl_extra_{c}" for c in "abc"],
    "r3_lin_[a-c]": [f"r3_lin_{c}" for c in "abc"],
    "r3_static_[a-d]": [f"r3_static_{c}" for c in "abcd"],
    "r3_hazard_[a-b]": [f"r3_hazard_{c}" for c in "ab"],
    "r3_controls_zero_effect": ["r3_controls_zero_effect"],
    "r3_controls_streams": ["r3_controls_streams"],
    "r3_heldout_misspec": ["r3_heldout_misspec"],
    "r3_x1_sampler_misspec": ["r3_x1_sampler_misspec"],
    "r3_replicate_hd1b_hd2": ["r3_replicate_hd1b_hd2"],
    "r3_replicate_hh1": ["r3_replicate_hh1"],
    "r3_replica_check": ["r3_replica_check"],
}
LIN_STREAMS_FULL = {"stream0": "all arms (18 cells)",
                    "streams_1_2": "all cells (same 18 cells as stream 0; full design, no downscale)"}


def _lin_all_cells_projection(s):
    """r3_lin pilot cells (sec/problem incl. sibling / orth provider) -> one chunk with all 18 cells on 3 streams."""
    tp = s["timing_projection"]
    cpu = sum(c["sec_per_problem"] * 16 * N_PROB_LIN * 3 * c["safety"] for c in tp["cells"])
    ovh = float(tp.get("per_job_overhead_s_mean") or 0.0) * 16 * 6 * 3 * 1.2
    return cpu / (WORKERS_PER_TASK * 60) + ovh / (WORKERS_PER_TASK * 60) + float(tp.get("allowance_min", 1.0))


N_PROB_LIN = 15


def eval_pilot_projections(lock_proj_pilot):
    """Per family: measured full-chunk minutes (max over chunks) for the full design and for the preset/non-preset
    downscale the methodology allows; the choice follows the lock rule (keep <= 45 min; else presets in order; a
    non-preset only if still > 55 min)."""
    out = {}
    for fam, tids in EVAL_PILOT_FAMILIES.items():
        rec = {"n_chunks": len(tids), "source": "evaluation-task pilots (dev seeds, concurrent timings) -- "
                                                 "exp/results/pilots/<task>/summary.json", "per_chunk": {}}
        full_min, ds_min, ds_name = [], [], None
        for t in tids:
            p = RES_ROOT / "pilots" / t / "summary.json"
            if not p.exists():
                rec["per_chunk"][t] = "missing"
                continue
            s = load(p)
            tp = s.get("timing_projection") or s.get("timing") or {}
            if fam == "r3_nl_extra_[a-c]":
                f, d = tp["projected_min_full_design_b8off_3streams"], tp["projected_min_b8off_stream0_only"]
                ds_name = "B8_off_stream0_only"
            elif fam == "r3_controls_zero_effect":
                f = tp["full_design_48inst_3streams"]["projected_min"]
                d = tp["lock_downscale_per_chunk_24inst_B3_stream0"]["projected_min"]
                ds_name = "split_2_chunks_24_inst+B3_stream0"
            elif fam == "r3_controls_streams":
                f, d = tp["adjusted_for_uncached_eval_tables"]["projected_min_full_adjusted"], None
            elif fam == "r3_lin_[a-c]":
                f, d = _lin_all_cells_projection(s), tp["projected_min_full_chunk"]
                ds_name = "lin_streams_1_2_core_only"
            elif fam == "r3_hazard_[a-b]":
                f, d = tp["projected_min_full_chunk"], (tp.get("downscale_option") or {}).get("projected_min")
                ds_name = "B3_reference_stream0_only (contingency, not a methodology preset)"
            elif fam == "r3_x1_sampler_misspec":
                f, d = tp.get("projected_full_min_corrected") or tp["projected_full_min"], None
            elif fam == "r3_heldout_misspec":
                f, d = tp["projected_full_min"], None
            elif fam == "r3_replicate_hd1b_hd2":
                hd1b = s["hd1b"]["timing_projection"]["projected_full_wall_min_at_4_workers"]
                f, d = hd1b + 35.0, None
                rec["note"] = "HD1b measured %.1f min + HD2 reduced config planner estimate 35 min" % hd1b
            elif fam == "r3_replicate_hh1":
                f, d = 25.0, None
                rec["note"] = "pilot estimate '15-25 min on 4 workers'; upper end used"
            elif fam == "r3_replica_check":
                f, d = lock_proj_pilot.get(fam, {}).get("projected_min_chosen", 25.0), None
                rec["note"] = "no timing projection in the pilot summary; planner estimate kept"
            else:
                f, d = tp["projected_min_full_chunk"], None
            rec["per_chunk"][t] = {"full_design_min": f, "downscale_min": d}
            full_min.append(f)
            if d is not None:
                ds_min.append(d)
        fmax = max(full_min) if full_min else None
        dmax = max(ds_min) if ds_min else None
        applied, chosen = [], fmax
        preset = fam in ("r3_nl_extra_[a-c]", "r3_lin_[a-c]")
        if fmax is not None and fmax > NO_DOWNSCALE_MAX and preset and dmax is not None:
            applied, chosen = [ds_name], dmax
        elif fmax is not None and fmax > HARD_MAX and dmax is not None:
            applied, chosen = [ds_name + " (non-preset; task_plan already split into r3_controls_zero_effect_a/_b, "
                                         "24 instances each, B3 on stream 0 only)"], dmax
        rec.update({"projected_min_full_design": fmax, "downscale_applied": applied, "projected_min_chosen": chosen,
                    "status": ("ok" if chosen <= NO_DOWNSCALE_MAX else "tight" if chosen <= HARD_MAX else "OVER"),
                    "lock_cost_model_min_pilot": lock_proj_pilot.get(fam, {}).get("projected_min_chosen")})
        if fam == "r3_hazard_[a-b]":
            rec["contingency"] = ("NOT applied: B3 reference on stream 0 only would project %.1f min; the full design "
                                  "(%.1f min <= 55) is frozen" % (dmax or 0, fmax or 0))
        if fam == "r3_lin_[a-c]":
            rec["note"] = ("full design = all 18 cells on streams 0-2 (frozen_items.lin_streams); measured stream-0 "
                           "cell costs x 3 streams; the core-only preset (%.1f min) is NOT applied" % (dmax or 0))
        out[fam] = rec
    return out


def _underfilled(projection, target=30.0):
    cols = ["[0,0.5)", "[0.5,1)", "[1,inf)"]
    rows_ = ["low", "mid", "high"]
    out = []
    for i, row in enumerate(projection):
        for j, c in enumerate(row):
            if c["expected_cell_n_at_64_instances"] < target:
                out.append({"ante_bin": rows_[i], "post_bin": cols[j],
                            "expected_problems_at_64_instances": round(c["expected_cell_n_at_64_instances"], 1),
                            "instances_needed": c["instances_needed_unique_problems"]})
    return out


def gate_block_full(p1, p2, p3, p4, p5):
    t0 = p4["T0"]
    ii = t0["ii_rem_bar_over_eps"]
    iv_med = p4["metrics"]["phi_perp_median"]
    rate = p3["nl"]["constructible_rate_nontrivial"]
    edges = p5["bin_edges_frozen"]
    q1 = p4["Q1_exploratory"]["pooled_static_m1"]
    a2v = p2["instance_log_ratios"]["I_full_off_A2"]
    proj5 = p5["projection_for_lock"]
    uf_mu, uf_lam = _underfilled(proj5["mu_flip"]), _underfilled(proj5["lambda_n0"])
    tm = p2["tmax_study"]
    return {
        "billing_gate": {"P1": p1["audit"]["billing_mismatch_by_arm"], "P2": p2["evidence_boundary"]["billing_mismatch"],
                         "P3_executed_orth": p3["jpc_runs"]["billing_mismatch_executed"],
                         "P5": "0 (pass_criteria.zero_crashes=%s)" % p5["pass_criteria"]["zero_crashes"]},
        "T0": {"i_identity_err_max": t0["i_identity_err_max"], "i_pass": t0["i_pass"],
               "ii_rem_bar_over_eps_p90_eta_le_2eps": ii["p90_eta_le_2eps"], "ii_n": ii["n_eta_le_2eps"],
               "ii_pass": ii["p90_eta_le_2eps"] <= 0.5,
               "iii_pass": p4["hard_gates"]["T0_iii_counts_theta_hat_retrievable"],
               "iv_phi_perp_median": iv_med, "iv_clause_triggered": iv_med > 0.3,
               "source": "full/r3_p4_t0_mechanism_gate (dev 720-739, %d ledger snapshots)" % t0["n_ledger_snapshots"]},
        "T0_scope": {"value": "eta <= eps" if ii["p90_eta_le_2eps"] > 0.5 else "eta <= 2 eps",
                     "rule": "hypotheses.md T0: (ii) fails -> Thm 3 scope narrows to eta <= eps, Q family tested only "
                             "in that scope", "binding": True},
        "phi_perp_clause": {"write": "定向补证无成本优势" if iv_med > 0.3 else None, "binding": True},
        "orth_confirmatory": {"value": rate >= 0.5, "constructible_rate_nl_stream0": rate,
                              "constructible_rate_nl_stream1_supplement": (p3.get("nl_supplement_streams") or {}).get(
                                  "constructible_rate_nontrivial"),
                              "constructible_rate_lin": p3["lin"]["constructible_rate"],
                              "constructible_rate_lin_p2_executed": p2.get("orth", {}).get("feasible_rate_nontrivial"),
                              "counts_nl": {k: p3["nl"][k] for k in ("n_nontrivial", "n_constructed",
                                                                     "n_certified_infeasible")},
                              "rule": "A3o descriptive if constructible rate < 0.5 (hypotheses.md A3o)",
                              "binding": True, "definitions": p3["definitions"],
                              "orth_infeasible_rule": "ORTH_INFEASIBLE problems are charged the censored cost tau in S; "
                                                      "A3o is computed on constructible problems only (descriptive)",
                              "caveat": "trace matching holds in expectation under theta_hat only (realised mean "
                                        "err %.3f, max %.3f); Markov (not stationary) policy vertices"
                                        % (p3["nl"]["realized_trace_err"]["mean"], p3["nl"]["realized_trace_err"]["max"])},
        "predictor_variant": {"value": p4["variant"]["choice"], "auc_S1_dev": p4["variant"]["auc_S1"],
                              "auc_S2_dev": p4["variant"]["auc_S2"], "variant_tie_rule": "S2", "binding": True,
                              "note": "binding choice on P4 full (dev 720-739): %d events / %d positives / %d clusters"
                                      % (q1["n"], q1["n_pos"], q1["n_clusters"])},
        "Q1_dev_signal": {"pooled": q1.get("S1"), "pooled_S2": q1.get("S2"),
                          "trivial_gap_baseline": q1.get("neg_log_gap_hat"),
                          "note": "exploratory dev readout; trivial -log gap_hat AUC %.2f > S1 %.2f / S2 %.2f -- Q1 "
                                  "expected negative; pre-registered as the likely outcome; not evidence"
                                  % (q1["neg_log_gap_hat"]["auc"], q1["S1"]["auc"], q1["S2"]["auc"])},
        "Q2_placebo": {"rule": "Q2 uses the GLOBAL permutation placebo (A_k_placebo_global)",
                       "reason": "dev A_k AUC %.2f == block placebo %.2f (block placebo is not a null for A_k)"
                                 % (p4["Q2_exploratory"]["pooled"]["A_k"]["auc"],
                                    p4["Q2_exploratory"]["pooled"]["A_k_placebo_block"]["auc"]),
                       "binding": True},
        "cand_o_gate": p4["cand_o_gate"], "cand_l_GR_gate": {"pass": False, **p4["cand_l_GR_gate"]},
        "static_twin_dev_base_rate": {k: {kk: p4["dev_runs"][k][kk] for kk in ("n_runs", "false_cert",
                                                                               "zero_cost_false", "paid_false")}
                                      for k in ("static_g1", "m1_2eps")},
        "r1adv": {"acceptance": p5["acceptance"]["primary_mu_flip_ge_1_over_M"],
                  "directed_construction_triggered": p5["directed_construction_triggered"],
                  "bin_edges": {"mu_flip": edges["mu_flip_tertiles"], "lambda_n0_S1": edges["lambda_n0_S1_tertiles"],
                                "S2_n0": edges["S2_n0_tertiles"], "rule": edges["rule"],
                                "n_accepted": edges["n_accepted"], "binding": True,
                                "source": "full/r3_p5_hazard_zone_sampler bin_edges_frozen (dev 750-760)"},
                  "primary_ex_ante_caliber": {"value": "mu_flip", "secondary": "Lambda_hat_perp(n0)",
                                              "reason": "placement/acceptance rule of the sampler is defined on "
                                                        "mu_flip (P5 design, written before its run); both binnings "
                                                        "are reported in eval", "binding": True},
                  "calibers_spearman_dev": p5["exante_caliber_agreement"]["spearman_mu_flip_vs_lambda_n0"],
                  "nontie_post_bin_structural_gap": "for gap >= eps, eta_arg/eps in {0} U [1,inf): the [0.5,1) column "
                                                    "of the 3x3 table can only be filled by tie problems",
                  "cell_fill": {"decision": "(b) cells projected below 30 problems are declared descriptive "
                                            "(under-filled); no threshold change",
                                "underfilled_mu_flip": uf_mu, "underfilled_lambda_n0": uf_lam,
                                "budget_instances": proj5["assumptions"]["n_instances"], "binding": True},
                  "R1_vs_R1adv_base_distribution": {
                      "value": "R1 = offgrid.sample_r1_truth (R1 sub-box, the rounds-0..2 R1 distribution named in "
                               "methodology 2.4); R1-adv = uniform on the G_1 hull (grid_ladder.BOX) + mu_flip >= 1/M "
                               "rejection (P5 sampler, frozen edges)",
                      "consequence": "the two layers have different base distributions; R1-vs-R1-adv contrasts are "
                                     "descriptive only; the R1-adv 3x3 table uses R1-adv rows only",
                      "binding": True}},
        "T_max_Lin": {"value": 20000, "tau_Lin": 20000, "readout": tm,
                      "mechanical_rule_value": p2["lock_recommendation"]["T_max_Lin"],
                      "reason": "censored share is flat for tau >= 5000 (%.4f / %.4f / %.4f) but the certified "
                                "non-tie maximum (%d, B1eb) exceeds 5000; methodology default 20000 kept (min 10000); "
                                "cost is not a constraint" % (tm["censored_frac_nontie_by_tau"]["5000"],
                                                              tm["censored_frac_nontie_by_tau"]["10000"],
                                                              tm["censored_frac_nontie_by_tau"]["20000"], tm["max"]),
                      "binding": True},
        "lin_streams": {"value": LIN_STREAMS_FULL,
                        "reason": "P2 full: all observed cells x 3 streams projected %.1f min for 120 instances "
                                  "(< 10 min per 16-instance chunk); lock rule keeps the full design when <= 45 min, so "
                                  "the core-only preset is not applied and every Lin endpoint uses the 3-stream S"
                                  % p2["lock_recommendation"]["est_minutes_120inst_3streams_observed_cells_4workers"],
                        "binding": True},
        "Lin_static_control": {
            "value": "option (b): static-model Lambda_hat_perp computed on the E1-Lin dynamic platform's own rows "
                     "(predictor field static_model_on_dynamic_rows / lambda_perp_static_model_max)",
            "reason": "frozen-load Lin-Static platform gives Lambda_hat_perp == 0 identically (eval pilot max "
                      "2.4e-16), so the static-twin control is degenerate there; Lin-Static rows are kept for cost "
                      "comparison and as a diagnostic",
            "dev_smoke": "dev 742 stream 1: static-model Lambda_hat_perp median 0.23 (130 certifications), dynamic "
                         "class 3e-16", "binding": True},
        "Lin_A2_direction_dev": {"mean_I": a2v["mean"], "n_pos": int(sum(v > 0 for v in a2v["values"])),
                                 "n": a2v["n"],
                                 "note": "dev direction opposite to the A2 prediction; pre-announced; not evidence; "
                                         "threshold unchanged; Tier wording for A2(Lin) failing is pre-registered"},
        "NL_A2_dev_preview": {"mean_I": p1["contrast_stats"]["3000"]["A2_I_full_off"]["mean"],
                              "B3_full_off_mean": p1["contrast_stats"]["3000"]["B3_full_off"]["mean"],
                              "note": "B3 gains from reuse about as much as JPC on dev; A2 null is the likely outcome; "
                                      "Tier B wording pre-registered; not evidence"},
    }


ADDENDA = {
    "status": "operational additions (no threshold, constant or operationalisation in the draft is changed)",
    "event_insufficiency": "C1/C2 (static twins) and the Q1 held-out family: if the number of positive events < 5 the "
                           "test is declared 'not evaluable' and counts as neither direction-consistent nor -inconsistent",
    "tie_strata": "tie / near / clear strata (5/5/5 per instance) are reported separately for every method-arm",
    "probe20_rule": "probe20 = ev(20) whose 20 required new steps are forced multi-regime probes "
                    "(acquire.multi_regime_probe.MultiRegimeProbe: candidate actions covering the most participants; "
                    "prefer actions keeping both sides engaged with left load >= 2); billed as new steps",
    "orth_infeasible": "ORTH_INFEASIBLE problems are charged tau (censored) in S; A3o computed on constructible problems "
                       "only and is descriptive (orth_confirmatory = false)",
    "m1r_note": "m1r and m1 are structurally equivalent at L = R = 2; the paper states m1r is an independently "
                "implemented held-out family, not a structurally different one",
    "hd1b_hd2_wrapper": "HD1b / HD2 replication runs only through run_r3_replicate_hd1b_hd2.py (seeds from "
                        "seed_blocks.eval: hd1b 10600-10629, HD2 10630-10669 / 10630-10649); round-2 scripts' "
                        "hard-coded seeds are never used",
    "replacement_reserves": {
        "rule": "quota_fail seeds are replaced deterministically in seed order from the family reserve; every reserve "
                "block is disjoint from every allocated evaluation block and from the other reserves",
        "nl_r0_main_and_controls": "10800 + 25*i, i = chunk a..h (10800-10999); controls_zero_effect / controls_streams "
                                   "reuse the r3_nl_main rule so theta* and problems are shared (CRN)",
        "nl_extra": "reproduces the r3_nl_main replacement order (same instances)",
        "static": "a/b/c/d 10700 / 10725 / 10750 / 10775",
        "lin": "a/b/c 11000 / 11025 / 11050",
        "hazard": "a/b 11100 / 11125 (moved from 10800 / 10825 by this lock: that block belongs to NL-R0)",
        "heldout": "10548-10599",
        "x1": "none (exploratory; quota_fail recorded)",
    },
    "lin_static_independent_quota": "Lin-Static uses its own static-truth 5/5/5 gap quota (same_problems=True kept as a "
                                    "diagnostic only)",
    "lin_b1eb_radius": "Lin B1eb uses the known-variance Gaussian confidence radius (sigma^2 c_q^2 sum w_t^2 m_t)",
    "static_twin_base_rate": "dev static_g1 false_cert 27/300, m1_2eps 13/300 (P4 full): the no-new-interaction check "
                             "reports false-cert rates per gap layer",
}


# ------------------------------------------------------------------------------------------------ main builders
def build(mode, workers, skip_smoke):
    src = "pilots" if mode == "pilot" else "full"
    out_dir = RES_ROOT / src / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = load(LOCK)
    if cur.get("status") == "locked":
        raise RuntimeError("plan/prereg_lock.json is already locked; refusing to overwrite")
    # the planner draft is the immutable source of thresholds; archive it once
    if cur.get("status") == "draft":
        DRAFT_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        if not DRAFT_ARCHIVE.exists():
            DRAFT_ARCHIVE.write_bytes(LOCK.read_bytes())
    draft = load(DRAFT_ARCHIVE)
    if mode == "full" and cur.get("status") == "provisional":
        prov = DRAFT_ARCHIVE.parent / "prereg_lock_provisional_pilot.json"
        if not prov.exists():
            prov.write_bytes(LOCK.read_bytes())
    progress(1, 6, "threshold_check")
    tc = threshold_check(draft)
    if not tc["hash_match"]:
        raise RuntimeError(f"hypotheses.md hash changed: {tc['hypotheses_md_sha256']} vs {tc['draft_sha256']}")

    def S(name):
        return load(RES_ROOT / src / name / "summary.json")

    p1, p2, p3, p4, p5 = (S(n) for n in ("r3_p1_reuse_ablation_nl", "r3_p2_lin_stream_smoke",
                                         "r3_p3_orth_replay_feasibility", "r3_p4_t0_mechanism_gate",
                                         "r3_p5_hazard_zone_sampler"))
    setup_rs = load(RES_ROOT / "pilots" / "r3_setup_reuse_switch" / "summary.json")
    setup_lin = load(RES_ROOT / "pilots" / "r3_setup_lin_stream" / "summary.json")
    go = {n: s.get("go_no_go") for n, s in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5))}
    go["P1"] = go["P1"] or ("GO" if p1["audit"]["n_bad"] == 0 and p1["n_run_errors"] == 0 else "NO_GO")

    progress(2, 6, "timing_smoke")
    smoke = None
    if mode == "full":
        # timing inputs are pilot measurements: re-use the pilot lock's smoke (dev 734-735) instead of re-running it
        prov = DRAFT_ARCHIVE.parent / "prereg_lock_provisional_pilot.json"
        smoke = load(prov).get("timing_projection", {}).get("timing_smoke") if prov.exists() else None
    elif not skip_smoke:
        smoke = timing_smoke(out_dir, workers)
        print("smoke", json.dumps({k: smoke[k] for k in ("n_runs", "wall_s", "crashes", "billing_mismatch")}),
              flush=True)
    progress(3, 6, "power_and_projection")
    power = power_block(p1, p2, mode)
    if mode == "full":
        tP = {n: load(RES_ROOT / "pilots" / n / "summary.json") for n in (
            "r3_p2_lin_stream_smoke", "r3_p3_orth_replay_feasibility", "r3_p4_t0_mechanism_gate")}
        C, cm_meta = cost_model(RES_ROOT / "pilots" / "r3_p1_reuse_ablation_nl", tP["r3_p2_lin_stream_smoke"],
                                tP["r3_p3_orth_replay_feasibility"], tP["r3_p4_t0_mechanism_gate"],
                                RES_ROOT / "pilots" / "r3_p5_hazard_zone_sampler", setup_rs, setup_lin, smoke)
        proj_cost_model = projections(C)
        proj = eval_pilot_projections(proj_cost_model)
        gates = gate_block_full(p1, p2, p3, p4, p5)
    else:
        C, cm_meta = cost_model(RES_ROOT / src / "r3_p1_reuse_ablation_nl", p2, p3, p4,
                                RES_ROOT / src / "r3_p5_hazard_zone_sampler", setup_rs, setup_lin, smoke)
        proj = projections(C)
        proj_cost_model = None
        gates = gate_block(p1, p2, p3, p4, p5, mode)

    lock = copy.deepcopy(draft)
    for k in ("thresholds_verbatim", "constants", "operationalisations", "seed_blocks", "hypotheses_md"):
        assert lock[k] == draft[k]
    lock.update({
        "version": 3, "status": "provisional" if mode == "pilot" else "frozen_pre_commit", "mode": mode,
        "written_by": f"{TASK} ({mode})", "written_at": datetime.now().isoformat(timespec="seconds"),
        "note": ("Round-3 PROVISIONAL lock from dev-seed pilots (720-760). Thresholds/constants/operationalisations "
                 "are the planner draft, unchanged (draft archived at plan/history/round3/). Numeric estimates below "
                 "are non-binding until the full lock (status=locked) re-derives them from P1-P5 full; evaluation "
                 "seeds remain untouched." if mode == "pilot" else
                 "Round-3 lock: frozen items from dev-seed freeze studies (P1-P5 full)."),
        "eval_seeds_touched": False,
        "threshold_check": tc,
        "go_no_go_inputs": go,
        "power": power,
        "gates": gates,
        "frozen_items": {
            "log_ratio_sd_nl": power["log_ratio_sd_nl"], "log_ratio_sd_lin": power["log_ratio_sd_lin"],
            "A2_MDE": power["A2_MDE"], "fallback_cand_r": power["fallback_cand_r"],
            "T_max_Lin": gates["T_max_Lin"]["value"], "lin_streams": gates["lin_streams"]["value"],
            "orth_confirmatory": gates["orth_confirmatory"]["value"], "T0_scope": gates["T0_scope"]["value"],
            "predictor_variant": gates["predictor_variant"]["value"],
            "r1adv_bin_edges": gates["r1adv"]["bin_edges"],
            "r1adv_primary_caliber": gates["r1adv"]["primary_ex_ante_caliber"]["value"],
            "downscale_decisions": {t: v["downscale_applied"] for t, v in proj.items()},
            **({"Lin_static_control": gates["Lin_static_control"]["value"],
                "R1_vs_R1adv_base_distribution": gates["r1adv"]["R1_vs_R1adv_base_distribution"]["value"],
                "r1adv_underfilled_cells_descriptive": gates["r1adv"]["cell_fill"]["underfilled_mu_flip"],
                "Q2_placebo": "global", "phi_perp_clause": gates["phi_perp_clause"]["write"]}
               if mode == "full" else {}),
            "binding": mode == "full"},
        "timing_projection": {
            "rule": (f"projected = sum(sec/problem x problem slots x safety) / ({WORKERS_PER_TASK} workers x 60) + "
                     f"fixed allowance; safety {SAFETY_CONCURRENT} for timings taken under 4-task concurrency, "
                     f"{SAFETY_UNLOADED} for timings taken unloaded. Keep the full design if <= {NO_DOWNSCALE_MAX} "
                     f"min; else apply the methodology-6 presets in order; if still > {HARD_MAX}, evaluate the listed "
                     "non-preset options (task split / stream reduction) and flag them as requiring a task_plan edit."),
            "per_task": proj, "cost_model_meta": cm_meta,
            **({"source": "full lock: per-chunk projections measured by the evaluation-task pilots (dev seeds, "
                          "concurrent); the pilot cost model is kept for reference",
                "pilot_cost_model_per_task": {t: {k: v.get(k) for k in ("projected_min_full_design",
                                                                         "downscale_applied", "projected_min_chosen")}
                                              for t, v in proj_cost_model.items()}} if mode == "full" else {}),
            "timing_smoke": smoke},
        **({"addenda": ADDENDA} if mode == "full" else {}),
        "code_sha256": code_sha(),
        "git_commit": None, "eval_manifest": None, "eval_manifest_sha256": None,
    })
    lock.pop("to_be_filled_by_r3_prereg_lock", None)
    lock["filled_fields"] = draft["to_be_filled_by_r3_prereg_lock"]
    return lock, out_dir, proj, power, gates, smoke, tc, go


def eval_manifest(lock):
    tp = load(TASK_PLAN)
    tasks = [{"task_id": t["id"], "samples": (t.get("full") or {}).get("samples")}
             for t in tp["tasks"] if t["id"].startswith("r3_") and t["type"] != "setup"
             and not re.match(r"r3_p\d", t["id"]) and t["id"] not in (TASK, "r3_analysis_aggregate")]
    return {"seed_blocks": lock["seed_blocks"]["eval"], "streams": lock["constants"]["streams"],
            "tasks": tasks, "downscale_decisions": lock["frozen_items"]["downscale_decisions"],
            "git_commit": lock["git_commit"], "generated_at": datetime.now().isoformat(timespec="seconds")}


def finalize_full(lock):
    """status=locked -> sha256 -> git commit (code + lock) -> record hash -> manifest -> re-hash -> commit."""
    from dsswm.stats.prereg import canonical_hash
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=WS, capture_output=True, text=True).stdout.strip()
    lock["status"] = "locked"
    lock["sha256"] = canonical_hash(lock)
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    paths = [str(LOCK), str(HERE / "dsswm"), str(HERE / "replica"), str(DRAFT_ARCHIVE.parent),
             *[str(HERE / f) for f in CODE_FILES if f.startswith("run_")]]
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feat(r3_prereg_lock): lock v3 (code + frozen items)\n\nCo-Authored-By: Assistant <noreply@example.org>"], cwd=repo, check=True)
    lock["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                        text=True).stdout.strip()
    lock["eval_manifest"] = eval_manifest(lock)
    lock["eval_manifest_sha256"] = sha_obj(lock["eval_manifest"])
    lock["sha256"] = canonical_hash(lock)
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    subprocess.run(["git", "add", "--", str(LOCK)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feat(r3_prereg_lock): eval manifest after lock commit\n\nCo-Authored-By: Assistant <noreply@example.org>"], cwd=repo,
                   check=True)
    return lock


def update_gpu_progress(start_iso, status, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    d = load(p) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
    end = datetime.now()
    start = datetime.fromisoformat(start_iso.replace("Z", "")) if start_iso else end
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    lst = "completed" if status == "success" else "failed"
    if TASK not in d[lst]:
        d[lst].append(TASK)
    d.get("running", {}).pop(TASK, None)
    d.setdefault("timings", {})[TASK] = {"planned_min": 20, "actual_min": max(1, round((end - start).total_seconds()
                                                                                       / 60)),
                                        "start_time": start.isoformat(), "end_time": end.isoformat(),
                                        "config_snapshot": snapshot}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-smoke", action="store_true")
    a = ap.parse_args()
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    st_file = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / f"{TASK}_start_time.txt"
    start_iso = st_file.read_text().strip() if st_file.exists() else datetime.now().isoformat()
    t0 = time.perf_counter()
    try:
        lock, out_dir, proj, power, gates, smoke, tc, go = build(a.mode, a.workers, a.skip_smoke)
        progress(5, 6, "write_lock")
        if a.mode == "pilot":
            from dsswm.stats.prereg import canonical_hash
            lock["sha256"] = canonical_hash(lock)
            LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
        else:
            lock = finalize_full(lock)
        over = [t for t, v in proj.items() if v["projected_min_chosen"] > HARD_MAX]
        nonpreset = [t for t, v in proj.items() if any("NON-PRESET" in x for x in v["downscale_applied"])]
        crit = {"lock_written_status": lock["status"],
                "thresholds_identical_to_hypotheses_md": tc["hash_match"] and tc["numeric_token_check_pass"]
                and lock["thresholds_verbatim"] == load(DRAFT_ARCHIVE)["thresholds_verbatim"],
                "all_eval_tasks_projected_le_55min": not over,
                "tasks_over_55": over, "tasks_needing_nonpreset_downscale": nonpreset}
        passed = (crit["lock_written_status"] in ("provisional", "locked") and
                  crit["thresholds_identical_to_hypotheses_md"] and crit["all_eval_tasks_projected_le_55min"])
        summary = {"task": TASK, "mode": a.mode, "started_at": start_iso,
                   "finished_at": datetime.now().isoformat(), "wall_clock_s": time.perf_counter() - t0,
                   "pass_criteria": crit, "passed": passed, "go_no_go": "GO" if passed else "NO_GO",
                   "lock_path": str(LOCK.relative_to(WS)), "lock_sha256": lock["sha256"],
                   "threshold_check": tc, "go_no_go_inputs": go, "power": power, "gates": gates,
                   "projection_table": {t: {k: v[k] for k in ("n_chunks", "projected_min_full_design",
                                                              "downscale_applied", "projected_min_chosen", "status")}
                                        for t, v in proj.items()},
                   "timing_smoke": None if smoke is None else {k: smoke[k] for k in smoke if k != "per_cell"},
                   "n_readouts_aggregated": None,
                   "note": ("并发运行说明：P1-P5 计时在 4 任务并发下测得（安全系数 1.2）；本任务的计时冒烟在无其他任务时测得"
                            "（安全系数 1.35）。pilot lock 为 provisional，不具约束力。" if a.mode == "pilot" else
                            "full lock：冻结项全部取自 P1-P5 full（开发种子 720-760）；计时外推取自各评价任务 pilot 的实测"
                            "（并发运行，计时偏高）；门槛与 hypotheses.md 逐字一致（sha256 校验）；评价种子未触碰。")}
        n_read = (len(lock["thresholds_verbatim"]) + sum(len(v["cells"]) for v in proj.values() if "cells" in v)
                  + len(FIXED_ESTIMATES) + 25 + (len(smoke["per_cell"]) if smoke else 0)
                  + sum(len(x) for x in (power["NL_R0"], power["E1_Lin"])))
        summary["n_readouts_aggregated"] = n_read
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
        update_gpu_progress(start_iso, "success", {"mode": a.mode, "timing_smoke_runs": None if smoke is None
                                                   else smoke["n_runs"], "smoke_seeds": SMOKE_SEEDS,
                                                   "cpu_workers": a.workers, "gpu_count": 0,
                                                   "note": "CPU only; gpu slot is a scheduling token"})
        progress(6, 6, "done", {"passed": passed})
        mark_done("success", f"lock {lock['status']} sha256={lock['sha256'][:12]}; passed={passed}; over55={over}")
        print(json.dumps(summary["pass_criteria"], ensure_ascii=False), flush=True)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        update_gpu_progress(start_iso, "failed", {"mode": a.mode, "error": repr(e)})
        mark_done("failed", repr(e)[:500])
        raise


if __name__ == "__main__":
    main()
