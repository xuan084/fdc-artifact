"""r5_prereg_lock: write plan/prereg_lock.json version 5 (methodology s2, s4, s5, s6; task_plan r5_prereg_lock).

Inputs (read only): exp/results/r5_gate_decision.json, exp/results/r5_gates/{rival_configs,c4_model}.json,
plan/{decision_log_r5.json, decision_log_r5.md, baseline_qualification_r5.md}, plan/theory/{fdc_theorem.md,
fdc_proof_code_table.md}, the dev-block summaries (T2 / plug-in / T1) for unit prices only. The FDC ledger is recomputed
from the structural ctx (segment weights, pool sizes, problems, checkpoints) of the dev and eval halves -- no outcome,
no permutation seed, no method run. Evaluation seeds (30000+) are not touched.

Steps
1. Archive the v4 lock to plan/history/r4_final/prereg_lock.json (copy, never overwritten) and check that
   check_lock(v4 archive, version=4) still holds (and v3 archive with version=3).
2. Build the v5 lock: FDC spec + ledger numbers, R + rival_configs.json sha256, plug-in comparators, excluded methods,
   endpoints, SAP, C4, cr_n_streams (from the gate decision), seed manifest, input sha256, eval tasks, declarations,
   code sha256. status = 'locked' iff gate verdict == 'GO', else 'locked_no_eval'.
3. Run the full test suite (must pass before the freeze), then two commits: (a) code + lock with git_commit = null,
   (b) lock with git_commit = hash of (a), re-hashed. Validate on disk (validate_lock v5, assert_locked v5 for an eval
   task, frozen-code drift = 0, git tree of (a) has the recorded code hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES = WS / "exp" / "results"
PIL = RES / "pilots"
PLAN = WS / "plan"
LOCK = PLAN / "prereg_lock.json"
V4_ARCHIVE = PLAN / "history" / "r4_final" / "prereg_lock.json"
V3_ARCHIVE = PLAN / "history" / "round3_final" / "prereg_lock.json"
TASK = "r5_prereg_lock"
WORKERS = 4
PY = sys.executable
CO_AUTHOR = "Co-Authored-By: Assistant <noreply@example.org>"

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")


def load(p):
    return json.loads(Path(p).read_text())


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(WS.resolve()))


def progress(step, total, phase, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total,
        "loss": None, "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


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


# ------------------------------------------------------------------------------------------------ structure / ledger
def ledgers():
    """FDC ledger + plug-in beta_tight from the structural ctx of each layer/half (no outcomes, no seeds)."""
    from dsswm.baselines.fdc import FDC_DELTA, FDC_SHARE, LEDGERS, fdc_ledger
    from dsswm.baselines.plugin_r5 import beta_tight
    from dsswm.envs.pool_replay import PoolReplayEnv
    from dsswm.streams import frontier as fr
    from dsswm.streams.frontier_runner import build_ctx

    eps = float(load(RES / "r4_gates" / "eps.json")["eps_star"])
    out = {}
    for layer, half in (("CR9", "dev"), ("CR9", "eval"), ("CR12", "eval")):
        env = PoolReplayEnv(layer, half)
        ctx = build_ctx(env, fr.cr_problems("visit"), eps)
        L, L60 = fdc_ledger(ctx), fdc_ledger(ctx, K=60)
        out[f"{layer}_{half}"] = {
            "S": int(ctx.S), "A": int(ctx.A), "S_A": int(ctx.S * ctx.A), "n_policies": int(ctx.P), "Q": int(ctx.Q),
            "tau_R": int(ctx.tau_R), "stop_k": int(ctx.stop_k), "eps": float(ctx.eps), "delta": float(ctx.delta),
            "K": int(len(ctx.checkpoints)), "checkpoints": [int(x) for x in ctx.checkpoints],
            "checkpoint_ratio": float((ctx.checkpoints[-1] / ctx.checkpoints[0]) ** (1 / (len(ctx.checkpoints) - 1))),
            "per_q_feasible": L["per_q_feasible"], "union_size_sum_q_Pi_Bq": int(L["union_size"]),
            "beta": float(L["beta"]), "x_v": float(L["x_v"]), "C_var": int(L["C_var"]),
            "bound_main": float(L["bound_main"]), "bound_var": float(L["bound_var"]),
            "K60": {"beta": float(L60["beta"]), "x_v": float(L60["x_v"]),
                    "beta_minus_K20": float(L60["beta"] - L["beta"])},
            "beta_tight_plugin": float(beta_tight(ctx)),
            "pool_sizes": env.pool_sizes.tolist(),
        }
    return out, {"FDC_DELTA": FDC_DELTA, "FDC_SHARE": FDC_SHARE, "LEDGERS": {k: list(v) for k, v in LEDGERS.items()}}, eps


# ------------------------------------------------------------------------------------------------ unit prices
def unit_prices():
    s = load(PIL / "r5_t2_gate_strict" / "summary.json")["per_method"]
    p = load(PIL / "r5_t2_gate_plugin_t3" / "summary.json")
    out = {m: v["sec_per_stream_mean"] for m, v in s.items()}
    for m, v in p["per_method_stop"].items():
        out.setdefault(m, v["sec_per_stream_mean"])
    out["FDC_full_horizon"] = p["full_horizon"]["per_method"]["FDC"]["sec_per_stream_mean"]
    bt = p["full_horizon"]["per_method"].get("B2-fav-tight")
    if bt:
        out["B2-fav-tight_full_horizon"] = bt["sec_per_stream_mean"]
    return out


def project(n_streams, methods, prices, scale=1.0):
    secs = [prices.get(m, max(prices.values())) * scale for m in methods]
    return round(n_streams * sum(secs) / WORKERS / 60 * 1.2 + max(secs) / 60 + 2.0, 1)


# ------------------------------------------------------------------------------------------------ lock content
def code_sha():
    files = sorted((HERE / "dsswm").rglob("*.py")) + sorted(HERE.glob("run_r5_*.py"))
    return {str(p.relative_to(HERE)): sha_file(p) for p in files if "__pycache__" not in p.parts}


def leaf_count(x):
    if isinstance(x, dict):
        return sum(leaf_count(v) for v in x.values())
    if isinstance(x, list):
        return sum(leaf_count(v) for v in x) if x else 1
    return 1


def build_lock(gate, rc, c4m, led, consts, eps, prices):
    from dsswm.streams import r5_registry as reg

    go = gate["verdict"] == "GO"
    n_cr = int(gate["cr_n_streams"])
    ext = bool(gate["enable_ext_blocks"])
    R = list(gate["rigorous_set_R"])
    assert R == list(reg.RIGOROUS_SET_R), (R, reg.RIGOROUS_SET_R)
    d9 = led["CR9_dev"]
    e9 = led["CR9_eval"]
    struct_same = all(d9[k] == e9[k] for k in ("union_size_sum_q_Pi_Bq", "beta", "x_v", "per_q_feasible", "S_A", "K"))
    plugins = ["B2-fav-tight", "FIX-bal-fav", "Peace-fav-bal", "B2-fav", "Peace-fav"]
    plugins_desc_only = ["B3-fav", "B5"]
    main_methods = ["FDC"] + R
    plugin_methods = ["FDC", "B2-fav", "B2-fav-tight", "FIX-bal-fav", "Peace-fav", "Peace-fav-bal", "B3-fav", "B5"]

    inputs = [PLAN / "decision_log_r5.json", PLAN / "decision_log_r5.md", PLAN / "theory" / "fdc_theorem.md",
              PLAN / "theory" / "fdc_proof_code_table.md", PLAN / "baseline_qualification_r5.md",
              PLAN / "methodology.md", PLAN / "task_plan.json", RES / "r5_gate_decision.json",
              RES / "r5_gates" / "rival_configs.json", RES / "r5_gates" / "c4_model.json",
              RES / "r4_gates" / "eps.json", V4_ARCHIVE,
              PIL / "r5_t2_gate_strict" / "summary.json", PIL / "r5_t2_gate_plugin_t3" / "summary.json",
              PIL / "r5_t1_reconcile" / "summary.json", PIL / "r5_fdc_spec_theory" / "summary.json",
              PIL / "r5_baseline_qualification" / "summary.json", PIL / "r5_rival_tuning" / "summary.json"]
    input_sha = {rel(p): sha_file(p) for p in inputs if p.exists()}
    missing_inputs = [rel(p) for p in inputs if not p.exists()]

    def task(seeds, methods, n, extra=None, scale=1.0, est=45, price_as=None):
        d = {"seeds": seeds, "methods": methods, "n_streams": n, "layer": "CR9", "half": "eval", "eps": eps, "K": 20,
             "assert": "dsswm.stats.prereg.lock_gate(task_id) / assert_locked(version=5, task_id=...) at start-up; "
                       "on refusal write summary.status='skipped_by_lock' and exit",
             "planned_min": est, "projected_min_from_dev_prices": project(n, price_as or methods, prices, scale)}
        d.update(extra or {})
        return d

    eval_tasks = {
        "r5_cr_main_a": task("30000-30099", main_methods, 100),
        "r5_cr_main_b": task("30100-30199", main_methods, 100),
        "r5_cr_plugin_a": task("30000-30099", plugin_methods, 100),
        "r5_cr_plugin_b": task("30100-30199", plugin_methods, 100),
        "r5_cr_factorial": task("30000-30199", ["FDCAblation x8", "B4-bal"], 200,
                                {"note": "(half,feas,r5) = FDC and (pool,qstar,r4) = QFC-pool must equal the main block "
                                         "stream by stream; c4_model.json predictions written (timestamped) before "
                                         "any eval result is read"}, scale=1.0, est=40,
                                price_as=["QFC-pool"] * 4 + ["FDC"] * 4 + ["B4-bal"]),
        "r5_cr_k60_sens": task("30000-30099", ["FDC", "QFC-pool", "B4-bal", "B2-rect", "Peace-rect", "B2-fav-tight"],
                               100, {"K": 60, "note": "report only; every method recomputes its own time-union budget "
                                                      "at K = 60; never mixed with K = 20 numbers"}, scale=1.5, est=50),
        "r5_cr_fwer_audit": task("30000-30199", ["FDC (full horizon)", "B2-fav-tight (full horizon, descriptive)"], 200,
                                 {"note": "no stop at 12/15: run to tau_R, all 15 certificates, N100"}, est=35,
                                 price_as=["FDC_full_horizon", "B2-fav-tight_full_horizon"]),
        "r5_cr12_scale": task("30000-30049", ["FDC", "B1", "B4", "B4-bal", "Hait-SW"], 50,
                              {"layer": "CR12", "note": "supporting, not a gate, not in C1; ms per checkpoint recorded"},
                              scale=8.0, est=50),
    }
    if ext:
        eval_tasks["r5_cr_main_ext_a"] = task("30200-30299", main_methods, 100)
        eval_tasks["r5_cr_main_ext_b"] = task("30300-30399", main_methods, 100)
    disabled = [] if ext else ["r5_cr_main_ext_a", "r5_cr_main_ext_b"]
    for t in eval_tasks.values():
        t["projection_note"] = "dev 4-slot concurrent sec/stream x streams / 4 workers x 1.2 + max/60 + 2 min (timing biased up)"

    lock = {
        "version": 5,
        "status": "locked" if go else "locked_no_eval",
        "mode": None,
        "written_by": TASK,
        "written_at": datetime.now().isoformat(),
        "candidate_id": "cand_fdc",
        "headline": ("fastest among the listed and qualified rigorous competitors R (methodology s0); NOT 'fastest among "
                     "all methods with guarantees'"),
        "thesis": ("Freezing the read schedule of a real RCT log licenses a direction-level Bernstein certificate, so FDC "
                   "certifies budgeted targeting policies faster than every listed rigorous competitor while keeping "
                   "stream-level FWER <= delta."),
        "note": ("Round-5 lock. The v4 lock is archived unchanged at plan/history/r4_final/prereg_lock.json and stays "
                 "valid under check_lock(version=4). Dev results (seeds 900-999) are exploratory; confirmatory evidence "
                 "only from eval seeds 30000+ run after this lock."),
        "eval_seeds_touched": False,
        "gate_decision": {
            "source": "exp/results/r5_gate_decision.json", "verdict": gate["verdict"],
            "written_at": gate["written_at"], "c4_mode": gate["c4_mode"],
            "gates_pass": {k: v.get("pass", v.get("deliverable")) for k, v in gate["gates"].items()},
            "T2_per_rival_ub95_dev": gate["gates"]["T2"]["per_rival_ub95"],
            "T2_worst_rival_dev": gate["gates"]["T2"]["worst_rival"],
            "T3_dev": {"stop_false": gate["gates"]["T3"]["stop_false"], "full_false": gate["gates"]["T3"]["full_false"]},
            "flags": gate["flags"],
            "decision_log_entry": "DL-10",
        },
        "fdc_spec": {
            "class": "dsswm.baselines.fdc.FDCMethod (no constructor arguments; ledger computed in setup() from ctx)",
            "registry_name": "FDC", "validity_class": "rigorous", "tuning": "none",
            "design": {"share_control": consts["FDC_SHARE"], "share_treatment": 1 - consts["FDC_SHARE"],
                       "kind": "fixed, data-free (alloc_kind='fixed'; b4_bal.balanced_alloc)",
                       "schedule": ("pre-generated by envs.pool_replay.make_schedule from the permutation seed and labels "
                                    "only (segment arrivals, planned arms, re-selection uniforms); independent of outcomes"),
                       "exhaustion_rule": ("an exhausted (segment, arm) pool re-selects with the pre-drawn uniform over the "
                                           "remaining arms (StreamEngine._resel, p renormalised; A = 2 -> all later arrivals "
                                           "of that segment go to the other arm); both arms exhausted -> the arrival is "
                                           "skipped but still billed"),
                       "two_stage_schedule": "deleted (shares 0.40 / 0.45 exploratory record only, DL entries)"},
            "certificate": {
                "code": "certify.quadknap.qfc_certificate_enum + certify.quadknap.make_stats (inherited from "
                        "baselines.frontier_common.QFCMethod.certify, r4 frozen at 6297e7f2)",
                "width": "Lemma L1: sqrt(2 beta V_bar) + b beta / 3 (direction level, per pair (pi', pi_hat))",
                "variance_ucb": "Lemma L2 Bernstein inversion at exponent x_v (theory_checks.mc_l1.bernstein_mu_ci + "
                                "sigma2_ucb); NOT the contrarian per-checkpoint empirical-Bernstein formula",
                "exhausted_cells": "exact: removed from V and b",
                "n0_rule": "if pi_hat or any challenger touches a cell with n = 0 the width is +inf",
                "center": "pi_hat_q = argmax of J_hat over Pi_{B_q}, ties -> lowest enumeration index",
                "statistic": "U_q = max_{pi' in Pi_{B_q}} [Delta_hat(pi', pi_hat_q) + width(pi', pi_hat_q)]; "
                             "all feasible challengers, no data-dependent pruning",
                "certify_rule": "U_q <= eps -> certified; answer frozen at first certification",
                "stopping": "certificates only at the K fixed checkpoints; stream stops at the first checkpoint with "
                            ">= 12 of 15 problems certified",
            },
            "checkpoints": {"K": 20, "rule": "log-spaced from 50,000 to tau_R (fr.checkpoints)",
                            "CR9_dev": d9["checkpoints"], "CR9_eval": e9["checkpoints"],
                            "adjacent_ratio_eval": e9["checkpoint_ratio"]},
            "ledger": {
                "delta": consts["FDC_DELTA"], "delta_main": consts["LEDGERS"]["r5"][0],
                "delta_var": consts["LEDGERS"]["r5"][1], "C_var": "S * A (from ctx)",
                "beta_formula": "beta = ln(sum_q |Pi_{B_q}| * K / delta_main)",
                "x_v_formula": "x_v = ln(2 * S * A * K / delta_var)",
                "CR9_dev": {k: d9[k] for k in ("union_size_sum_q_Pi_Bq", "K", "S_A", "beta", "x_v", "bound_main",
                                               "bound_var", "per_q_feasible")},
                "CR9_eval": {k: e9[k] for k in ("union_size_sum_q_Pi_Bq", "K", "S_A", "beta", "x_v", "bound_main",
                                                "bound_var", "per_q_feasible")},
                "CR12_eval": {k: led["CR12_eval"][k] for k in ("union_size_sum_q_Pi_Bq", "K", "S_A", "beta", "x_v",
                                                               "n_policies")},
                "dev_eval_identical": struct_same,
                "expected_numbers": {"beta": 14.2242, "x_v": 11.8776},
                "eval_task_assert": ("eval tasks must assert FDCMethod().setup(ctx).params L1 / x_v equal "
                                     "fdc_spec.ledger.CR9_eval.beta / x_v to 1e-9"),
                "no_override": ("constructor takes no beta / x_v / share / delta / K; any overridden FDCAblation gets "
                                "validity='none' and an '-explore' suffix (unit-test guarded)"),
                "beta_14p1_results": "exploratory only (union bound 0.05595 > delta)",
                "K60_sensitivity": {"beta_CR9": e9["K60"]["beta"], "x_v_CR9": e9["K60"]["x_v"],
                                    "beta_increment": e9["K60"]["beta_minus_K20"], "expected_increment": math.log(3)},
            },
            "theorem": {"name": "FDC-1", "file": "plan/theory/fdc_theorem.md",
                        "statement": ("conditional on the pre-generated schedule A, P(exists k <= K, q: U_q(k) <= eps and "
                                      "J(pi*_q) - J(pi_hat_q(k)) > eps | A) <= delta_main + delta_var = 0.05; stream FWER "
                                      "<= delta for both the stop-at-12/15 and the full-horizon windows")},
            "removed": ["two-stage read schedule", "Cauchy-Schwarz tight accounting module (A = 2)", "FPC / Serfling"],
        },
        "rigorous_set_R": R,
        "rival_configs": {
            "path": "exp/results/r5_gates/rival_configs.json",
            "sha256": sha_file(RES / "r5_gates" / "rival_configs.json"),
            "frozen_at": rc["frozen_at"], "selection_rule": rc["selection_rule"], "tuning_seeds": "900-949",
            "frozen_configs": rc["frozen_configs"], "untuned_rivals": rc["untuned_rivals"],
            "grid": rc["grid"],
            "registry_specs": {n: {"validity_class": reg.spec(n).validity_class, "source": reg.spec(n).source,
                                   "module": reg.spec(n).module} for n in R},
            "disclosures": [
                "Hait-SW[p_min=0.02, gamma=2/3] and B4-bal[share=0.5] gave identical N80 on 100/100 dev streams: "
                "|R| = 9 listed methods, 8 distinct dev N80 profiles. Both stay in R (IUT over 9 names).",
                "Molitor-WoR was censored on 100/100 dev streams and B3-rect on 33/100: their ratios are driven by rival "
                "censoring and are reported as such, not as speed-ups of the same kind.",
                "QFC-pool is our own r4 method (hardest dev rival).",
            ],
        },
        "plugin_comparators": {
            "role": "descriptive E-cost comparators (validity 'none' / 'asymptotic'); never in R, never a gate",
            "e_cost_set": plugins,
            "descriptive_only": plugins_desc_only,
            "beta_tight": {"formula": "ln(sum_q |Pi_{B_q}| * K / delta)", "CR9_eval": e9["beta_tight_plugin"]},
            "fastest_plugin_rule": ("the fixed method of e_cost_set with the smallest eval-block geometric mean N80_pen "
                                    "(over all 200 streams), never chosen per stream"),
            "wording": "plug-in faster but without guarantee -- stated explicitly in the paper",
        },
        "excluded_methods": {
            "QFC-half": "ablation of FDC (same design/certificate/code path, r4 ledger); C4 factorial only, not in C1",
            "B2-nominal / B3-nominal / Peace-nominal": "nominal: time-uniform threshold proven only i.i.d. with "
                                                       "replacement; never promoted whatever its FWER",
            "Hait-pooled": "excluded: needs i.i.d. with-replacement sampling",
            "Shekhar & Howard 2605.21736": "related work: fixed-n, not anytime-valid; dominated by Molitor-WoR / B4",
            "H8-*": "model-internal validity only; not run in r5 eval",
            "source": "plan/baseline_qualification_r5.md",
        },
        "endpoints": {
            "N80_pen": ("primary. k* = first checkpoint with >= 12/15 certified. N80_pen = t_{k*} if k* exists and every "
                        "certificate issued up to k* is correct; = tau_R if any earlier certificate is false; = tau_R "
                        "if 12/15 is never reached. Reaching 12/15 exactly at t_K = tau_R gives tau_R with "
                        "completed = true"),
            "N80_raw": "stop time without false-certificate penalty (tau_R if never reaches 12/15)",
            "completed": "reached >= 12/15 at some checkpoint <= tau_R",
            "fwer_event": "any false certificate (gap J(pi*_q) - J(pi_decided) > eps + 1e-12) in the window",
            "censored": "(not completed) or fwer_event",
            "N100": "first checkpoint with 15/15 certified, from the full-horizon run (r5_cr_fwer_audit); "
                    "penalised to tau_R analogously",
            "also_reported": ["per-problem certification times", "per-budget completion curves",
                              "lattice-difference distribution and tie fraction"],
        },
        "sap": {
            "pairing_unit": "eval stream (same permutation seed) across methods",
            "contrast": "d_i = log N80_pen(FDC, i) - log N80_pen(r, i); point estimate exp(mean d)",
            "bootstrap": {"type": "paired percentile bootstrap, resample streams with replacement", "B": 10000,
                          "seed": 42},
            "C1": {
                "test": "intersection-union test (IUT): for every r in R the one-sided 95% upper bound of exp(mean d) "
                        "< 1.0; no multiplicity correction needed for the conjunctive claim",
                "per_rival_claims": ("per-rival statements in the paper use Holm-adjusted one-sided bootstrap p-values "
                                     "p_r = (1 + #{b: mean d*_b >= 0}) / (B + 1), step-down at alpha = 0.05 over |R| = "
                                     f"{len(R)}; per-rival unadjusted 95% UB and Bonferroni UB at 1 - 0.05/{len(R)} "
                                     "are reported alongside"),
                "wording_gate": {"faster_by_10pct": "point <= 0.90 AND UB < 1.0 (Holm-rejected)",
                                 "faster": "UB < 1.0 (Holm-rejected) only", "threshold": 0.90},
                "falsified_if": "any r in R has UB >= 1.0",
            },
            "C2": {
                "rule": "one-sided 95% Clopper-Pearson upper bound of the false-stream rate <= delta = 0.05, both "
                        "windows (stop at 12/15; full horizon to tau_R, 15/15)",
                "critical": gate["c2_cp_critical"],
                "n_streams": n_cr,
                "max_false_streams": gate["c2_cp_critical"][str(n_cr)]["max_false_streams"],
                "interpretation": "CP failure = empirical safety acceptance not met, not a falsification of Theorem "
                                  "FDC-1; stream variation is replay randomness on one fixed log",
                "audit": "every false stream audited individually",
            },
            "E_cost": {"set": plugins, "ci": "two-sided 95% paired percentile bootstrap (B = 10^4, seed 42)",
                       "pass_line": None},
            "K60_sensitivity": {"task": "r5_cr_k60_sens", "K": 60, "FDC_beta": e9["K60"]["beta"],
                                "rule": "report only, not a gate, never mixed with K = 20 numbers"},
            "verdict_rule": ("'positive_result_achieved' iff C1 IUT passes AND C2 passes (both windows); otherwise "
                             "'not_achieved'. C4 only decides the mechanism wording; E-cost has no pass line."),
            "no_stream_exclusion": "all eval streams are analysed; no rerun, no exclusion",
            "practical_significance": "rows saved (N80 diff x bytes/row), wall clock, certificate ms/checkpoint, "
                                      "preprocessing cost; timings are 4-slot concurrent (biased up)",
        },
        "c4": {
            "factorial": {"factors": {"D": ["pool", "half"], "U": ["qstar (Q*|Pi| = 7680)", "feas (sum_q|Pi_Bq| = 3386)"],
                                      "L": ["r4 (0.04, 0.01, 48)", "r5 (0.045, 0.005, S*A=18)"]},
                          "cells": ["pool,qstar,r4", "pool,qstar,r5", "pool,feas,r4", "pool,feas,r5",
                                    "half,qstar,r4", "half,qstar,r5", "half,feas,r4", "half,feas,r5"],
                          "identities": {"half,feas,r5": "FDC", "pool,qstar,r4": "QFC-pool"},
                          "class": "dsswm.baselines.fdc.FDCAblation(design, union, ledger)"},
            "decomposition": ("main effects and interactions on mean log N80_pen (effects coding), Shapley over "
                              "{D, U, L} of log N80_pen(FDC) - log N80_pen(QFC-pool cell); interactions reported "
                              "as they are"),
            "design_share_rule": "Shapley_D / total log speedup >= 2/3 (point estimate; bootstrap CI reported)",
            "certificate_shape": "FDC vs B4-bal[share = 0.5]: paired one-sided 95% UB < 1.0",
            "falsified_if": "design share < 2/3 or FDC/B4-bal UB >= 1.0 (C4 withdrawn; C1 unaffected)",
            "prediction_model": {"path": "exp/results/r5_gates/c4_model.json",
                                 "sha256_file": sha_file(RES / "r5_gates" / "c4_model.json"),
                                 "sha256_internal": c4m["sha256"], "frozen_at": c4m["frozen_at"],
                                 "error_limit_abs_log": c4m["error_limit_abs_log"],
                                 "rule": "any cell |log(pred geomean N80 / observed)| > 0.095 -> withdraw 'mechanism is "
                                         "predictable', keep the descriptive decomposition only",
                                 "dev_validation_max_abs_log_err": gate["gates"]["T1"]["c4_max_abs_log_err"],
                                 "c4_mode_at_gate": gate["c4_mode"]},
            "dev_reference": "exp/results/pilots/r5_t1_reconcile/summary.json (dev design share 0.93; exploratory)",
        },
        "power": {k: gate["power"][k] for k in ("rule", "hardest_rival", "theta_log", "sd_log", "power_N200",
                                                "power_N400", "n_rep", "B_boot")},
        "cr_n_streams": n_cr,
        "ext_blocks_enabled": ext,
        "disabled_tasks": disabled,
        "seed_manifest": {
            "dev": "900-999", "rival_tuning": "900-949", "c4_model_calibration": "900-949",
            "c4_model_validation": "950-999",
            "CR9_eval_main": "30000-30199", "CR9_eval_ext": "30200-30399 (" + ("enabled" if ext else "disabled") + ")",
            "CR12_eval": "30000-30049", "bootstrap_seed": 42, "row_split_seed": 4242,
            "post_unblinding_exploratory": ">= 40000",
            "r3_eval_forbidden": "10000-10799",
            "rule": "eval seeds untouched before this lock; every eval task calls lock_gate / assert_locked(version=5)",
        },
        "unit_prices_sec_per_stream_dev": prices,
        "eval_tasks": eval_tasks,
        "confirmatory_tasks": ["r5_cr_main_a", "r5_cr_main_b"] + (["r5_cr_main_ext_a", "r5_cr_main_ext_b"] if ext else []),
        "no_eval_permitted_tasks": [],
        "declarations": [
            "User ruling 2026-10-03 (after the round-4 dev block): headline narrowed to 'fastest among the listed "
            "rigorous competitors'; plug-in / asymptotic methods are descriptive only. Disclosed as a deviation "
            "(decision_log_r5 DL entries), not as an ex-ante design.",
            "Effect-size gate changed from 0.67 to UB < 1.0 plus a 0.90 wording threshold; C3 demoted to the E-cost "
            "estimate.",
            "The r4 eps selection (r4_g_eps_select) read eval-half non-trivial-problem counts (aggregate truth). Eval "
            "permutations and eval-half row-level outcomes were never used for any method or gate comparison. "
            "Claims are 'certified on the logged population'.",
            "C_var changed from 48 to S*A = 18 and the delta split to 0.045 / 0.005 before dev results; legal tightening.",
            "The 50/50 share of QFC-half was chosen after seeing r4 results; FDC locks 0.50 (not the faster dev 0.40).",
            "The eval-half structural ctx (segment weights, pool sizes, feasible classes) was built by this task to "
            "record the ledger numbers; no outcome and no eval permutation seed was read.",
            "Dev T2 ratios are 0.14-0.53 (> 30% improvement; flagged by protocol); r4-consistency 0 mismatches, 0 false "
            "certificates.",
            "Dev timings are 4-slot concurrent and biased up.",
        ],
        "input_sha256": input_sha,
        "missing_inputs": missing_inputs,
        "structural_ctx": {k: {kk: v[kk] for kk in ("S", "A", "S_A", "n_policies", "Q", "tau_R", "stop_k", "eps",
                                                    "delta", "K", "pool_sizes")} for k, v in led.items()},
        "code_sha256": code_sha(),
        "code_freeze_rule": ("git_commit = the commit that contains this code and the lock (with git_commit null); a "
                             "second commit records the hash. assert_locked(version=5) refuses evaluation if any frozen "
                             "non-test dsswm module differs from code_sha256"),
        "git_commit": None,
        "sha256": None,
    }
    return lock


# ------------------------------------------------------------------------------------------------ git
def git(*a, check=True):
    r = subprocess.run(["git", *a], cwd=WS, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)} failed: {r.stderr[-500:]}")
    return r.stdout.strip()


def git_commit_lock(lock):
    from dsswm.stats.prereg import canonical_hash
    real = WS.resolve()
    lock["sha256"] = canonical_hash(lock)
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    paths = [real / "plan" / "prereg_lock.json", real / "plan" / "history" / "r4_final" / "prereg_lock.json",
             real / "exp" / "code" / "dsswm", *sorted((real / "exp" / "code").glob("run_r5_*.py")),
             *[real / p for p in lock["input_sha256"]]]
    extra = real / "exp" / "code" / "r5_decision_log_entries.json"
    if extra.exists():
        paths.append(extra)
    git("add", "--", *map(str, paths))
    git("commit", "-m", f"feat(r5_prereg_lock): lock v5 ({lock['status']}) + code freeze for the r5 eval block\n\n"
                        f"{CO_AUTHOR}")
    c1 = git("rev-parse", "HEAD")
    lock["git_commit"] = c1
    lock["sha256"] = canonical_hash(lock)
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    git("add", "--", str(real / "plan" / "prereg_lock.json"))
    git("commit", "-m", f"feat(r5_prereg_lock): record code-freeze commit {c1[:10]} in lock v5\n\n{CO_AUTHOR}")
    return lock, c1, git("rev-parse", "HEAD")


def tree_matches(commit, code_hashes):
    """Every recorded code file in the frozen commit has the recorded sha256."""
    prefix = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=HERE, capture_output=True,
                            text=True).stdout.strip()
    bad = []
    for f, h in code_hashes.items():
        r = subprocess.run(["git", "show", f"{commit}:{prefix}{f}"], cwd=HERE, capture_output=True)
        if r.returncode != 0 or hashlib.sha256(r.stdout).hexdigest() != h:
            bad.append(f)
    return bad


def update_gpu_progress(start, status, snapshot):
    p = WS / "exp" / "gpu_progress.json"
    d = load(p) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
    end = datetime.now()
    lst = "completed" if status == "success" else "failed"
    other = "failed" if lst == "completed" else "completed"
    d[other] = [t for t in d.setdefault(other, []) if t != TASK]
    if TASK not in d.setdefault(lst, []):
        d[lst].append(TASK)
    d.setdefault("running", {}).pop(TASK, None)
    d.setdefault("timings", {})[TASK] = {"planned_min": 35, "actual_min": max(1, round((end - start).total_seconds() / 60)),
                                         "start_time": start.isoformat(), "end_time": end.isoformat(),
                                         "config_snapshot": snapshot}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    tmp.replace(p)


def run_pytest(out_dir):
    log = out_dir / "pytest_full.log"
    t = time.perf_counter()
    args = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "dsswm/tests"]
    try:
        import xdist  # noqa: F401
        args += ["-n", str(WORKERS)]
    except ImportError:
        pass
    r = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    log.write_text(r.stdout + "\n" + r.stderr + f"\nEXIT {r.returncode}\n")
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
    return r.returncode == 0, tail[0], round(time.perf_counter() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    start = datetime.now()
    t0 = time.perf_counter()
    out_dir = PIL / TASK if a.mode == "pilot" else RES / "full" / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    try:
        from dsswm.stats.prereg import (LOCK_VERSION, assert_locked, check_lock, frozen_code_drift, lock_gate,
                                        validate_lock)
        assert LOCK_VERSION == 5
        progress(0, 6, "archive_v4")
        cur = load(LOCK)
        if cur.get("version") == 4:
            if not V4_ARCHIVE.exists():
                V4_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(LOCK, V4_ARCHIVE)
            assert sha_file(V4_ARCHIVE) == sha_file(LOCK), "v4 archive differs from the live v4 lock"
        elif cur.get("version") == 5:
            print("live lock is already v5: rebuilding it (v4 archive kept)", flush=True)
            if cur.get("git_commit"):
                raise RuntimeError("a committed v5 lock exists; refusing to rewrite it (lock unchanged rule)")
        v4 = load(V4_ARCHIVE)
        checks = {"v4_archive_validate": bool(validate_lock(v4, 4))}
        try:
            check_lock(v4, 4, task_id="r4_cr_main_a")
            checks["v4_confirmatory_blocked"] = False
        except RuntimeError:
            checks["v4_confirmatory_blocked"] = v4["status"] == "locked_no_eval"
        if v4.get("no_eval_permitted_tasks"):
            checks["v4_assert_locked_permitted_task"] = bool(
                assert_locked(version=4, task_id=v4["no_eval_permitted_tasks"][0]))
        checks["v3_archive_check_lock_v3"] = bool(check_lock(load(V3_ARCHIVE), 3)) if V3_ARCHIVE.exists() else None

        progress(1, 6, "inputs")
        gate = load(RES / "r5_gate_decision.json")
        rc = load(RES / "r5_gates" / "rival_configs.json")
        c4m = load(RES / "r5_gates" / "c4_model.json")
        assert gate["eval_seeds_touched"] is False
        progress(2, 6, "ledger")
        led, consts, eps = ledgers()
        assert abs(led["CR9_eval"]["beta"] - 14.2242) < 5e-5 and abs(led["CR9_eval"]["x_v"] - 11.8776) < 5e-5, led
        prices = unit_prices()
        lock = build_lock(gate, rc, c4m, led, consts, eps, prices)
        lock["mode"] = a.mode

        progress(3, 6, "pytest")
        if a.skip_tests:
            tests_ok, tests_tail, tests_sec = None, "skipped", 0.0
        else:
            tests_ok, tests_tail, tests_sec = run_pytest(out_dir)
            print(f"pytest: ok={tests_ok} {tests_tail} ({tests_sec}s)", flush=True)
            if not tests_ok:
                raise RuntimeError(f"full test suite failed before code freeze: {tests_tail}")
        # code hashes must be recomputed after pytest (no file may change while tests ran)
        assert lock["code_sha256"] == build_lock(gate, rc, c4m, led, consts, eps, prices)["code_sha256"]

        progress(4, 6, "git")
        from dsswm.stats.prereg import canonical_hash
        c1 = c2 = None
        if a.no_git:
            lock["sha256"] = canonical_hash(lock)
            LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
        else:
            lock, c1, c2 = git_commit_lock(lock)

        progress(5, 6, "validate")
        on_disk = load(LOCK)
        checks["validate_lock_v5"] = bool(validate_lock(on_disk, 5)) if on_disk.get("git_commit") else None
        ok, msg = lock_gate("r5_cr_main_a")
        checks["assert_locked_v5_eval_task"] = {"ok": ok, "msg": None if ok else str(msg)[:300]}
        checks["eval_permitted_iff_GO"] = ok == (gate["verdict"] == "GO")
        checks["frozen_code_drift"] = frozen_code_drift(on_disk)
        checks["v4_archive_still_valid_after"] = bool(check_lock(load(V4_ARCHIVE), 4, task_id=(
            v4.get("no_eval_permitted_tasks") or [None])[0]) if v4["status"] == "locked_no_eval" else check_lock(
            load(V4_ARCHIVE), 4))
        checks["v4_archive_sha_unchanged"] = sha_file(V4_ARCHIVE) == lock["input_sha256"].get(rel(V4_ARCHIVE))
        if c1:
            checks["code_commit_tree_matches"] = tree_matches(c1, on_disk["code_sha256"]) == []
        n_leaves = leaf_count(on_disk)
        n_hashes = len(on_disk["code_sha256"]) + len(on_disk["input_sha256"]) + 3
        passed = bool(checks["validate_lock_v5"] and checks["eval_permitted_iff_GO"]
                      and not checks["frozen_code_drift"] and checks["v4_archive_still_valid_after"]
                      and checks["v4_archive_validate"] and (tests_ok is not False)
                      and checks.get("code_commit_tree_matches", True) and n_leaves >= 100)
        summary = {
            "task_id": TASK, "mode": a.mode, "candidate_id": "cand_fdc", "started_at": start.isoformat(),
            "finished_at": datetime.now().isoformat(), "wall_s": round(time.perf_counter() - t0, 1),
            "lock_path": "plan/prereg_lock.json", "lock_version": on_disk["version"], "lock_status": on_disk["status"],
            "lock_sha256": on_disk["sha256"], "code_commit": on_disk["git_commit"], "lock_record_commit": c2,
            "v4_archive": rel(V4_ARCHIVE),
            "metrics": {"lock_sha256": on_disk["sha256"], "code_commit": on_disk["git_commit"],
                        "status": on_disk["status"], "lock_leaf_fields": n_leaves, "hashes_recorded": n_hashes,
                        "code_files_hashed": len(on_disk["code_sha256"]), "inputs_hashed": len(on_disk["input_sha256"]),
                        "cr_n_streams": on_disk["cr_n_streams"], "beta_CR9_eval": led["CR9_eval"]["beta"],
                        "x_v_CR9_eval": led["CR9_eval"]["x_v"], "beta_CR12_eval": led["CR12_eval"]["beta"],
                        "eval_tasks": sorted(on_disk["eval_tasks"])},
            "pass_criteria": {
                "validate_lock_v5": checks["validate_lock_v5"],
                "check_lock_v4_still_valid": checks["v4_archive_still_valid_after"],
                "assert_locked_v5_blocks_unless_locked": "unit tests test_prereg_v5 (status/no_eval/tamper/drift) + "
                                                         f"live gate ok={ok} with status={on_disk['status']}",
                "code_commit_recorded": bool(on_disk["git_commit"]),
                "full_test_suite": {"ok": tests_ok, "tail": tests_tail, "sec": tests_sec},
                ">=100_fields": n_leaves >= 100,
            },
            "checks": checks, "pass": passed, "go_no_go": "GO" if passed else "NO_GO",
            "ledger": led, "missing_inputs": on_disk["missing_inputs"],
            "projection_min": {t: v["projected_min_from_dev_prices"] for t, v in on_disk["eval_tasks"].items()},
            "integrity": {"eval_seeds_touched": False, "methods_run": False,
                          "eval_half_read": "structural ctx only (weights, pool sizes, feasible classes)"},
            "timing_note": "单价来自并发运行（4 槽）的开发块实测，计时偏高",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
        update_gpu_progress(start, "success" if passed else "failed",
                            {"mode": a.mode, "cpu_workers": WORKERS, "gpu_count": 0,
                             "note": "CPU only; gpu slot is a scheduling token; pytest -n 4"})
        progress(6, 6, "done", {"passed": passed})
        mark_done("success" if passed else "failed",
                  f"lock v5 {on_disk['status']} sha256={on_disk['sha256'][:12]} code_commit={str(on_disk['git_commit'])[:10]}"
                  f" cr_n_streams={on_disk['cr_n_streams']} pass={passed}")
        print(json.dumps({k: summary[k] for k in ("metrics", "pass_criteria", "checks", "pass")}, ensure_ascii=False,
                         indent=1, default=str))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        update_gpu_progress(start, "failed", {"error": repr(e)[:300]})
        mark_done("failed", repr(e)[:500])
        raise


if __name__ == "__main__":
    main()
