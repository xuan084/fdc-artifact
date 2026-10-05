"""Build plan/prereg_lock_v6_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage: run_from exp/code:  .venv/bin/python3 build_v6_addendum_draft.py
The DRAFT is status 'draft'; locking is done by the authors with dsswm.stats.prereg_v6.finalize_addendum.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.envs.data_v6 import data_hashes  # noqa: E402
from dsswm.stats import prereg  # noqa: E402
from dsswm.stats.v6_analysis import A_FAMILY, B_METHODS, V5_R  # noqa: E402
from dsswm.stats.v6_replica import COMPARISON_GROUPS, R1_N_SEEDS, TIMING_FIELDS  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


CODE_FILES = ["run_r5s_v6.py", "run_r5s_v6_qualification.py", "build_v6_addendum_draft.py",
              "dsswm/baselines/rect_v6.py", "dsswm/baselines/wor_betting_v6.py", "dsswm/streams/frontier_runner_v6.py",
              "dsswm/envs/lenta_v6.py", "dsswm/stats/prereg_v6.py", "dsswm/stats/v6_analysis.py",
              "dsswm/stats/v6_replica.py", "dsswm/envs/data_v6.py", "dsswm/stats/v6_seal.py",
              "dsswm/tests/test_v6_addendum.py"]
INPUTS = ["exp/results/v6_gates/hc_config_cr9.json", "exp/results/v6_gates/rival_configs_lr9.json",
          "exp/results/pilots/v6_baseline_qualification/summary.json", "exp/results/pilots/v6a_tune/summary.json",
          "exp/results/pilots/v6b_tune/summary.json", "exp/results/pilots/v6b_hc_sens/summary.json",
          "exp/results/pilots/v6a_pilot/summary.json", "exp/results/pilots/v6b_pilot/summary.json",
          "exp/results/pilots/v6c_pilot/summary.json", "exp/results/r5_gates/rival_configs.json",
          "plan/prereg_lock.json", "plan/theory/fdc_theorem.md", "plan/baseline_qualification_r5.md",
          "exp/results/full/r5_cr_main_a/results.jsonl", "exp/results/full/r5_cr_main_b/results.jsonl",
          "exp/results/full/r5_cr_fwer_audit/results.jsonl"]


def main():
    v5 = prereg.check_lock(prereg.load_lock(prereg.lock_path(5)), 5)
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    hc = json.loads((WS / "exp/results/v6_gates/hc_config_cr9.json").read_text())["selected"]
    lr9 = json.loads((WS / "exp/results/v6_gates/rival_configs_lr9.json").read_text())["selected"]
    v5cfg = v5["rival_configs"]["frozen_configs"]
    frozen = {"CR9": {**{k: dict(v) for k, v in v5cfg.items()}, "HC-WoR": hc}, "LR9": lr9}
    ap = json.loads((WS / "exp/results/pilots/v6a_pilot/summary.json").read_text())["dev_ratios_exploratory"]
    devA = {k.split("/")[1].split("@")[0]: {"ratio": round(v["geomean_ratio"], 3),
                                            "ci95": [round(x, 3) for x in v["ci95_two_sided"]],
                                            "ub95": round(v["ub95_one_sided"], 3)} for k, v in ap.items()}
    add = {
        "version": "5+v6-addendum", "status": "draft", "written_at": "2026-10-04 (rev. 5 after external reviewer confirm)",
        "written_by": "author (supp_r5)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v5["sha256"],
                        "git_commit": v5["git_commit"],
                        "verdict": "positive_result_achieved (C1 IUT pass, C2 pass both windows)"},
        "v5_verdict_statement": "Append-only addendum. Nothing in this file edits plan/prereg_lock.json, its rival set "
        "R, its Holm family, its endpoints or its verdict. Blocks A, B and C are supplementary; each has its own family "
        "and rule and NONE of their outcomes can change the v5 verdict 'positive_result_achieved'. They can only change "
        "the WORDING of claims (certificate attribution, endpoint scope, external-validity scope), as pre-declared below.",
        "claim_scope": "Every block-A/C conclusion is limited to the REGISTERED rivals and their frozen configs. A 'keep' "
        "supports only the tested evidence threshold (UB* < 0.70), not a '2x at matched guarantee' statement. RECT-ck-HG "
        "is an exact fixed-n test inversion, not the tightest of all valid constructions; differences between HG, HG-live, "
        "HC and Bern change several construction features at once and do not license a quantitative attribution "
        "'mainly due to guarantee strength'.",
        "blocks": {
            "A": {
                "name": "same-strength rectangular rival stress test",
                "role": "confirmatory for the certificate-attribution wording only (registered rivals/configs)",
                "layer": "CR9", "half": "eval", "eps": 0.001, "K": 20, "stop_k": 12, "delta": 0.05, "Q": 15,
                "methods": {
                    "FDC": "dsswm.baselines.fdc.FDCMethod (v5-frozen, unchanged)",
                    "B4-bal": "v5 frozen config share=0.5 (anchor: same design, WSR20 time-uniform rectangle)",
                    "RECT-ck-HG": "rect_v6.RectCkHG: frozen 50/50 design; exact hypergeometric test inversion per "
                    "(cell, checkpoint), alpha_side = delta/(2 S A K) = 0.05/720; all of delta to the means; running "
                    "intersection over checkpoints; valid at the K pre-specified checkpoints. No tuning parameter.",
                    "RECT-ck-HG-live": "rect_v6.RectCkHGLive: as RECT-ck-HG, but delta_k = delta/K per checkpoint is "
                    "split only over the L_k live cells (0 < n_c(k) < N_c; schedule-measurable): alpha_side(k) = "
                    "delta/(2 K L_k). Empty and exhausted intervals cost nothing. Weakly dominates RECT-ck-HG. No tuning.",
                    "RECT-ck-Bern": "rect_v6.RectCkBern: same design/union; per-cell Lemma L1 Bernstein + Lemma L2 "
                    "variance UCB with FDC's split (0.045 / 0.005). Matched-lemma contrast (not the strongest split). "
                    "No tuning parameter.",
                    "HC-WoR": {"module": "wor_betting_v6.HCWoRRect",
                               "construction": "hedged-capital WoR betting CS (Waudby-Smith & Ramdas JRSSB 2024 s6; WoR "
                               "conditional mean as in WSR NeurIPS 2020), theta = 1/2, alpha = delta/(S A) per cell, "
                               "time-uniform (Ville on theta K+ + (1-theta) K-), rectangle certificate, frozen 50/50 design; "
                               "inversion of the two monotone one-sided regions over the full logical interval (rev. 2 fix), "
                               "empty CS -> logical pool interval",
                               "tuning_grid": "HC_GRID (6 points: prpl c in {0.5, 0.75}; nstar c = 0.75, target_frac in "
                               "{0.1, 0.2, 0.35, 0.5})",
                               "tuning_rule": "lock v5 rule on CR9 dev seeds 900-949 (argmin geomean N80_pen among grid "
                               "points with 0/50 false streams and billing_ok; ties -> first grid item); re-run with the "
                               "fixed code", "frozen_config": hc,
                               "config_file": "exp/results/v6_gates/hc_config_cr9.json"}},
                "family_R_A": list(A_FAMILY),
                "validity_justification": {
                    "RECT-ck-HG": "Frozen design: n_c(k) is a function of the outcome-free schedule (fdc_theorem s2); "
                    "given the schedule the ones among the first n_c(k) draws are Hypergeometric(N_c, M_c, n_c(k)) (Fact 0). "
                    "Exact one-sided inversions, union over 2 S A K events = delta; on the event every rectangle bound "
                    "dominates Delta(pi*, pi_hat) for all q, k.",
                    "RECT-ck-HG-live": "Same, with the union taken conditionally on the schedule over the live (cell, k) "
                    "pairs only: sum_k sum_{live c} 2 delta/(2 K L_k) <= delta (equality unless some L_k = 0); empty ([0,1]) and exhausted (exact) "
                    "intervals have zero miscoverage.",
                    "RECT-ck-Bern": "fdc_theorem s3 with the direction set replaced by +-e_c: E_main has 2 S A K events at "
                    "e^{-beta_c}; E_var unchanged; monotonicity in V.",
                    "HC-WoR": "theta K+ + (1-theta) K- is a nonnegative martingale at the true mean (predictable lambda, "
                    "factors >= 1-c > 0); max(theta K+, (1-theta) K-) >= 1/alpha implies the sum does, so Ville gives "
                    "time-uniform coverage; K+ nonincreasing and K- nondecreasing in m, each inverted separately with "
                    "outward bisection; union over S A cells."},
                "qualification": {"procedure": "r5_baseline_qualification criteria (i)-(v); toy FWER 200 streams",
                                  "file": "exp/results/pilots/v6_baseline_qualification/summary.json",
                                  "note": "NAIVE-control false streams show that a broken certificate is DETECTED; its CP "
                                  "upper bound is below 0.05, so they do not show an FWER above 0.05"},
                "seeds": "31000-31199 (200 NEW eval streams)", "n_streams": 200,
                "endpoint": "N80_pen (v5 definition, unchanged)",
                "analysis": "dsswm.stats.v6_analysis.analyse_block('A'): validated complete matrix (registered methods x "
                "200 seeds, unique, finite > 0), paired FDC/r geomean ratios, paired percentile bootstrap B = 10^4 seed 42 "
                "(same resample matrix for all rivals), one-sided 95% UB and two-sided 95% CI (approximate procedure)",
                "decision_rule": {
                    "statistic": "UB* = max over family_R_A of the one-sided 95% UB of FDC/r (conjunctive, no "
                    "within-family multiplicity correction; per-rival wordings do not jointly inherit 95%)",
                    "keep": "UB* < 0.70 -> pre-registered evidence threshold met over the registered rivals; wording keeps "
                    "the certificate-advantage claim limited to these rivals/configs, with 'valid at K pre-specified "
                    "checkpoints' attached; it does not by itself support '2x at matched guarantee'",
                    "downgrade": "0.70 <= UB* < 1.0 -> FDC faster than every registered rival but the pre-registered "
                    "evidence threshold is NOT met; every point estimate and CI is reported as is",
                    "withdraw": "UB* >= 1.0 -> no certificate-superiority claim over the registered rivals; the 2x2 "
                    "'certificate' row is labelled 'vs the time-uniform WSR20 rectangle only'; main-conference viability "
                    "re-assessed by the authors",
                    "replica_fail": "forces 'withdraw'"},
                "dev_pilot_exploratory": {"seeds": "950-999", "ratios_FDC_over_r": devA},
                "also_reported": "FDC/B4-bal on the new seeds, FDC/RECT-ck-Bern, tie fractions (K = 20 lattice), false "
                "streams per method with CP UB, HC empty-CS events"},
            "B": {
                "name": "Lenta LR9 descriptive re-use of an outcome-exposed log",
                "role": "DESCRIPTIVE ONLY: no verdict, no confirmatory claim, not an untouched holdout",
                "dataset": {"path": "$DATA_DIR/lenta/tidy.pkl",
                            "sha256": "6d075bbc306df3512f6bc3a795c94686984b4c99da3889c91778d25325565d43",
                            "rows": 687029, "arms": "0 control 24.9% / 1 SMS 75.1%", "A": 2,
                            "outcome": "response_att (binary; dev-half rate 0.109)",
                            "sha_check": "LentaLayerEnv verifies the sha256 by default"},
                "outcome_exposure": "r4 (r4_setup_pool_replay) computed FULL-table cell means of the LR8 segmentation, "
                "which include every row of the LR9 eval half. A new segmentation and split seed do not restore an "
                "untouched holdout; block B is therefore a re-use of an outcome-exposed log, reported descriptively.",
                "eval_access": "LR9 eval outcomes are not exposed through any evaluation interface before the lock: "
                "LentaLayerEnv returns a structure-only object unless eval_task_id passes prereg_v6.addendum_gate "
                "(no other bypass; tested). pandas reads the whole pickle into memory (outcomes are loaded transiently "
                "for the covariate cache); the outcome column is never cached and never materialised for eval rows.",
                "layer": "LR9: S = 9 = age tertile (full-table cut points 35, 50) x promo level (x_promo_share_15d = 0 | "
                "(0, 0.6242] | > 0.6242), 2^9 = 512 policies; cut points from covariates only",
                "split": "split seed 6006, stratified 50/50 by (LR9 segment x arm); dev 343,516 rows, eval 343,513 rows",
                "problems": "kappa = (0, 1), budgets 0.10..0.80 step 0.05 (15), stop at 12/15, delta 0.05, K = 20 "
                "log-spaced from 5,000 to tau_R",
                "eps_grid_reported_in_full": [0.002, 0.003, 0.004],
                "eps_selection": "none: all three are reported; grid fixed from DEV only. No eps is chosen after eval.",
                "methods": list(B_METHODS),
                "rival_configs": {"rule": "v5 grids + HC_GRID re-tuned on LR9 dev seeds 900-949 at eps 0.003, v5 rule; "
                                  "re-run with the fixed HC code", "selected": lr9,
                                  "file": "exp/results/v6_gates/rival_configs_lr9.json",
                                  "note": "B4-bal / Hait-SW / B2-rect grid points all reach 12/15 only at tau_R on dev: "
                                  "selection is a tie broken by grid order; B4-bal is therefore share 0.4 and is NOT in "
                                  "the 50/50 comparison group of block B"},
                "hc_sensitivity_report_only": "exp/results/pilots/v6b_hc_sens/summary.json (targets 0.5-1.0; grid "
                "unchanged)",
                "seeds": "32000-32199 (200 NEW eval streams)",
                "reported": "per eps: geomean N80_pen/tau_R per method, share at tau_R (terminal pile-up), FDC/r paired "
                "ratios with CIs, per-problem certification times, false streams (CP UB), non-trivial count on eval",
                "feasibility": "DEV: at eps 0.002 (7/15 non-trivial) no method in R (nor FDC) reaches 12/15 before tau_R; "
                "exact-checkpoint and betting rectangles reach it earlier than FDC on part of the streams. At eps "
                "0.003/0.004 all 15 problems are eps-trivial and every R rival piles up at tau_R. A C1-style test would "
                "be decided by terminal pile-up; hence descriptive.",
                "replica_fail": "B tables are published only in the appendix with the flag 'replica_fail' and are not "
                "cited in the main text",
                "hillstrom": "excluded: A = 3 (FDC is specified for A = 2) and its eval-half summaries were read by the r4 "
                "eps guard",
                "no_pooling": "never pooled with Criteo; separate family; no Holm/IUT across datasets"},
            "C": {
                "name": "N100 supplementary continuation of the nine v5 rigorous rivals",
                "role": "supplementary continuation under a rule fixed after part of the endpoints were already "
                "public: N80 of every method and FDC's N100_pen (r5_cr_fwer_audit) on these 200 streams are known. Not "
                "an independent blinded confirmation; it decides only the endpoint-scope wording. New confirmatory "
                "evidence would need unused streams.",
                "layer": "CR9", "half": "eval", "eps": 0.001, "K": 20, "stop_k": 15,
                "seeds": "30000-30199 (v5 eval streams; continuation by design)",
                "methods": ["FDC"] + list(V5_R), "rivals_bound": list(V5_R),
                "configs": "v5 frozen rival_configs (exp/results/r5_gates/rival_configs.json, sha "
                + v5["rival_configs"]["sha256"] + ")",
                "mechanics": "frozen run_stream with ctx.stop_k = 15; trajectory up to the original 12/15 stop is the same "
                "deterministic function of the seed",
                "endpoint": "N100_pen: first checkpoint with 15/15 certified and no false certificate up to it; tau_R "
                "otherwise",
                "analysis": "v6_analysis.analyse_block('C'): validated complete matrix (FDC + 9 rivals x 200 seeds), paired "
                "FDC/r N100 ratios, one-sided 95% UB, B = 10^4 seed 42",
                "decision_rule": "'n100_pass' iff UB < 1 for every one of the nine rivals (IUT) and the replica passed; "
                "else 'narrow_to_N80' (headline endpoint restricted to 12/15)"}},
        "frozen_configs": frozen,
        "seed_manifest": {"A_tuning_dev": "900-949 (CR9 dev)", "A_pilot_dev": "950-999 (CR9 dev)",
                          "A_eval": "31000-31199", "B_tuning_dev": "900-949 (LR9 dev)", "B_pilot_dev": "950-999 (LR9 dev)",
                          "B_eval": "32000-32199", "C_pilot_dev": "950-957 (CR9 dev)",
                          "C_eval": "30000-30199 (v5 eval streams, continuation)", "bootstrap_seed": 42,
                          "lr9_split_seed": 6006,
                          "never_used_check": "results jsonl search for seeds 31000-31199 / 32000-32199: 0 hits "
                          "(2026-10-03); supports 'no saved run', cannot prove no unsaved access"},
        "replica_check": {
            "frozen_before_running": True, "implementation": "dsswm/stats/v6_replica.py + run_r5s_v6.py --replica",
            "R1_rerun": f"independent process re-runs the FIRST {R1_N_SEEDS} seeds of each eval task (A 31000-31009, "
            "B 32000-32009, C 30000-30009) for every method of the task and every eps of the task (B: 0.002, 0.003, "
            "0.004); canonical rows must be identical", "R1_excluded_fields": list(TIMING_FIELDS),
            "R2_comparison_groups": {k: list(v) for k, v in COMPARISON_GROUPS.items() if k.endswith(("_full", "_full_a",
                                                                                                      "_full_b"))},
            "R2_rule": "within every (seed, eps) the schedule_digest is identical across the task's group; adaptive / "
            "pool-design methods and LR9 B4-bal (share 0.4) are in no group",
            "R3_C_mapping": {"v5_k_stop": "top-level rs_k_stop (r5_cr_main_a schema) or nested run_stream.k_stop "
                             "(r5_cr_main_b schema); k_cut = k_stop, or the last index of the v5 curve if v5 never "
                             "reached 12/15",
                             "N80_pen": "v5 N80_pen", "k80": "v5 k_stop (when v5 completed)",
                             "n_cert_curve[:k_cut+1]": "v5 n_cert_curve[:k_cut+1] (v5 curves are padded to K "
                             "entries after the stop; the padded tail is ignored)",
                             "cert_k[q]": "v5 cert_k[q] if >= 0; else v6 cert_k[q] == -1 or > k_cut",
                             "regression_test": "test_v6_addendum.py::test_r3_real_v5_records (real records of both "
                             "schemas incl. Peace-rect seed 30037)",
                             "FDC N100_pen": "r5_cr_fwer_audit N100_pen (method 'FDC')",
                             "v5_files": ["exp/results/full/r5_cr_main_a/results.jsonl",
                                          "exp/results/full/r5_cr_main_b/results.jsonl",
                                          "exp/results/full/r5_cr_fwer_audit/results.jsonl"]},
            "R2_completeness": "R2 also requires every planned (seed, eps) stream for every group member",
            "report_integrity": "run_r5s_v6.py --replica refuses to write a report unless the task's FULL planned matrix "
            "(methods x eps x seeds, each exactly once) exists for the current code / addendum / data; the report binds "
            "task_id, is_eval, addendum_sha256 (non-null for eval), per-file code sha256, data sha256, methods / eps / "
            "seeds / R1 seeds, the required rule set (R1, R2; + R3 for block-C eval tasks) and the sha256 of the "
            "canonical result rows and of the replica rows (v6_replica.content_hash), and for block-C eval the v5 "
            "reference rows. Writing any new row to a task deletes its report. At analysis, v6_replica.validate_report "
            "RECOMPUTES R1-R3 (compute_checks) from the actual replica rows, the current main rows and the frozen v5 "
            "references (both matrices must be complete) and uses ONLY the recomputed status; the stored report is a "
            "cache that must reconcile exactly (checks, status, every binding) or the analysis refuses. Editing a "
            "report's status / pass fields therefore cannot change any verdict (regression test of the external reviewer's "
            "counterexample).",
            "status": "a block's replica status is 'pass' only if every task of the block has a VALID report with status "
            "'pass'; an invalid or missing report makes the analysis refuse to run",
            "on_failure": {"A": "verdict forced to 'withdraw'", "C": "verdict forced to 'narrow_to_N80'",
                           "B": "appendix-only with flag 'replica_fail'"}},
        "eval_tasks": {
            "v6a_full": {"block": "A", "seeds": "31000-31199", "methods": ["FDC", "B4-bal"] + list(A_FAMILY),
                         "est_cpu_h": 0.9, "est_wall_min_4w": 15},
            "v6b_full_a": {"block": "B", "seeds": "32000-32199", "methods": ["Peace-rect"], "eps": [0.002, 0.003, 0.004],
                           "est_cpu_h": 3.3, "est_wall_min_4w": 50},
            "v6b_full_b": {"block": "B", "seeds": "32000-32199", "methods": [m for m in B_METHODS if m != "Peace-rect"],
                           "eps": [0.002, 0.003, 0.004], "est_cpu_h": 1.3, "est_wall_min_4w": 20},
            "v6c_full_a": {"block": "C", "seeds": "30000-30199", "methods": ["Peace-rect"], "est_cpu_h": 2.3,
                           "est_wall_min_4w": 36},
            "v6c_full_b": {"block": "C", "seeds": "30000-30199",
                           "methods": ["FDC"] + [m for m in V5_R if m != "Peace-rect"], "est_cpu_h": 1.3,
                           "est_wall_min_4w": 20},
            "command": "cd exp/code && .venv/bin/python3 run_r5s_v6.py --task <task>; then --task <task> --replica; "
            "then --analyse A|B|C. Every step re-checks the locked addendum (status, sha256, v5 link, code and input "
            "hashes) and writes the addendum sha256 into every result row and summary."},
        "declarations": [
            "Block A was added after the external reviewer's result_debate review identified that v5 rectangle rivals carry a stronger "
            "(time-uniform) guarantee than FDC (K checkpoints). Dev pilots (exploratory) were run before this draft; "
            "the thresholds 0.70 / 1.0 were given by the authors before any v6 run.",
            "Rev. 2 (after external reviewer addendum review): fixed an HC-WoR inversion error that loosened the rival (accepted-"
            "centre assumption); HC tuning, qualification and dev pilots re-run with the fixed code; RECT-ck-HG-live "
            "added; decision functions validate inputs; replica rules implemented; configs, inputs and the lock bound "
            "by hash.",
            "Rev. 3 (after external reviewer re-check): replica reports bound to task / addendum / code / data / result content and "
            "validated at analysis; R3 supports both v5 schemas and ignores the padded curve tail; raw Criteo and Lenta "
            "data hashes frozen and verified at the v6 layer.",
            "Lock note (authors approval 2026-10-04): the dev pilot rows / summaries bound in input_sha256 were "
            "produced before the final runner / replica / seal code hash (rev. 4-5 changed only provenance, replica and "
            "seal code; method code -- baselines, certificates, envs, harness -- is unchanged since rev. 2), so dev "
            "analyses would need a pilot re-run to re-validate; the dev pilots were not regenerated before locking.",
            "Rev. 5 (after external reviewer confirm): git-committed eval seals of all main rows; analysis requires current rows "
            "== sealed hash; explicit threat model.",
            "Rev. 4 (after external reviewer final check): the analysis recomputes R1-R3 from the actual rows; report pass/status "
            "fields are never trusted.",
            "HC-WoR is tuned on dev (6-point grid); RECT-ck-* have no free parameter; FDC is untuned (v5 frozen). The "
            "rectangle interval families follow the external reviewer's suggestions (exact hypergeometric, betting, live-cell spending); "
            "no other family was tried on dev. Stronger valid rivals may exist; conclusions are limited to these.",
            "Block C re-runs the v5 eval streams; N80 of all methods and FDC's N100_pen there are already public.",
            "Block B re-uses an outcome-exposed log (r4 full-table LR8 means) and is descriptive.",
            "Timings are 4-worker concurrent and biased up."],
        "eval_seal": "when an eval task finishes (and whenever a resume completes it) the runner writes "
        "exp/results/full/v6_seals/<task>.seal.json (content sha256 of ALL main result rows, timing fields excluded; "
        "row count; method / eps / seed coverage; code, data and addendum hashes) and immediately git-commits that file "
        "alone; the commit sha goes into a sidecar <task>.seal_ref.json (committed alone right after; a file cannot "
        "contain the hash of the commit that adds it) and into the task summary. The replica step and the analysis "
        "(dsswm/stats/v6_seal.verify_seal) refuse unless the seal commit exists and is in HEAD history, the seal file "
        "equals its content at that commit, its bindings match, and the content hash of the CURRENT main rows equals "
        "the sealed hash; validate_report additionally requires that sealed hash for every eval task. R2 runs on the "
        "full main matrix (all planned seeds x eps), R1 on the first 10 seeds.",
        "threat_model": "The integrity checks cover accidental drift (code, inputs, raw data, configs), partial and "
        "resumed runs, and post-hoc edits to results, replica reports or seals, all detectable against git history "
        "(eval artifacts are committed immediately after completion: the seal at task completion, results, replica "
        "reports and analyses right after each step). Deliberate rewriting of git history is out of scope, as in "
        "standard preregistration practice.",
        "input_sha256": {p: sha(WS / p) for p in INPUTS},
        "data_sha256": data_hashes(),
        "data_binding": "raw data frozen at the v6 layer (dsswm/envs/data_v6.py; v5 pool_replay untouched): Criteo "
        "tidy.pkl (read by the replay env) + its source csv.gz, Lenta tidy.pkl; verified by prereg_v6.check_addendum "
        "(eval start, every resume, analysis) and by the runner before any env is built; the per-layer combined data "
        "hash is stored in every result row / summary / replica report and resumed rows with another data hash are "
        "discarded",
        "code_sha256": {p: sha(CODE / p) for p in CODE_FILES},
        "lock_procedure": "authors: review -> git commit (code + inputs) -> dsswm.stats.prereg_v6.finalize_addendum("
        "git_commit=<that commit>) writes plan/prereg_lock_v6_addendum.json (refuses to overwrite; verifies the commit "
        "contains the bound code) -> commit the lock -> run eval tasks, replicas, analyses. A post-lock fix requires a "
        "new versioned addendum, never an overwrite.",
        "git_commit": None}
    add["sha256_draft"] = prereg.canonical_hash(add)
    (WS / "plan/prereg_lock_v6_addendum_DRAFT.json").write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(add["sha256_draft"], len(add["code_sha256"]), len(add["input_sha256"]))


if __name__ == "__main__":
    main()
