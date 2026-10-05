"""Build plan/prereg_lock_v7_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage (cwd exp/code):  .venv/bin/python3 build_v7_addendum_draft.py
Locking: authors / experimenter commits code + inputs, then dsswm.stats.prereg_v7.finalize_addendum(git_commit).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.baselines.fdc_bet import VARIANTS, _LAMBDA_GRID  # noqa: E402
from dsswm.envs.data_v6 import data_hashes  # noqa: E402
from dsswm.stats import prereg, prereg_v6  # noqa: E402
from dsswm.stats import v7_analysis as VA  # noqa: E402
from dsswm.stats.v6_replica import R1_N_SEEDS, TIMING_FIELDS  # noqa: E402
from dsswm.stats.v7_replica import COMPARISON_GROUPS  # noqa: E402

import run_r5s_v7 as RV  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT
INPUTS = ["plan/prereg_lock.json", "plan/prereg_lock_v6_addendum.json", "plan/theory/fdc_theorem.md",
          "plan/fdc_bet_exploration.md", "exp/results/v6_gates/hc_config_cr9.json",
          "exp/results/v6_gates/rival_configs_lr9.json", "exp/results/r5_gates/rival_configs.json",
          "exp/results/full/v6_summary.md", "exp/results/full/v6_analysis_A.json",
          "exp/results/pilots/fdc_bet/analysis_dev.json", "exp/results/pilots/fdc_bet/analysis_tune.json",
          "exp/results/pilots/fdc_bet/dev/results.jsonl", "exp/results/pilots/fdc_bet/tune/results.jsonl",
          "exp/results/pilots/fdc_bet/lenta/results.jsonl", "exp/results/pilots/fdc_bet/toy/summary.json",
          "exp/results/pilots/v7a_pilot/summary.json", "exp/results/pilots/v7b_pilot/summary.json",
          "exp/results/pilots/v7_analysis_A_dev.json", "exp/results/pilots/v7_analysis_B_dev.json"]
FDC_BET_ORIGIN_COMMIT = "ec56177f4e67a430b304d10c764a121372851a38"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob_sha(commit, rel):
    r = subprocess.run(["git", "show", f"{commit}:./{rel}"], cwd=str(CODE), capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def dev_numbers():
    out = {}
    for b in ("A", "B"):
        p = WS / f"exp/results/pilots/v7_analysis_{b}_dev.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        for e, pe in a["per_eps"].items():
            prim = pe["FDC-BF|N80_pen"]
            out[f"{b}@{e}"] = {m: {"ratio": round(v["geomean_ratio"], 3), "ub95": round(v["ub95_one_sided"], 3)}
                               for m, v in prim.items()}
    return out


def main():
    v5 = prereg.check_lock(prereg.load_lock(prereg.lock_path(5)), 5)
    v6 = prereg_v6.load_locked_addendum(None)
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    cr9 = RV.load_configs("CR9")
    lr9 = RV.load_configs("LR9")
    frozen = {"CR9": cr9, "LR9": lr9}
    fdc_bet_rel = "dsswm/baselines/fdc_bet.py"
    add = {
        "version": "5+v7-addendum", "status": "draft", "written_at": "2026-10-04",
        "written_by": "author (r5 follow-up, FDC-bet)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v5["sha256"],
                        "git_commit": v5["git_commit"]},
        "v6_addendum": {"path": "plan/prereg_lock_v6_addendum.json", "sha256": v6["sha256"],
                        "git_commit": v6["git_commit"],
                        "block_A_verdict": "downgrade (UB* = 0.888 vs HC-WoR; FDC/RECT-ck-HG 0.851 [UB 0.871])"},
        "statement": (
            "Append-only addendum. It edits neither lock v5 nor the v6 addendum and cannot change their verdicts. It "
            "registers ONE confirmatory test (block A) of a NEW primary method, FDC-BF, on fresh eval seeds, plus one "
            "descriptive block (B). Disclosure: FDC-BF was devised on 2026-10-04 AFTER the v6 eval results were known "
            "(v6 block A downgraded FDC vs same-strength exact-checkpoint / betting rectangles). It is a post-hoc "
            "deviation from the v5/v6 primary method and is reported as such. Selection used dev data only: CR9 dev "
            "seeds 900-949 (tuning of FDC-MR weights) and 950-999 (comparison of ~22 certificate variants), Lenta LR9 "
            "dev half seeds 950-999, and a toy FWER check; no v7 eval seed and no LR9 eval outcome was used. The dev "
            "seeds 950-999 that selected FDC-BF are therefore NOT confirmatory. Only block A is confirmatory, and only over "
            "fresh replay randomisation (permutation seeds 33000-33199) CONDITIONAL on the Criteo CR9 eval table, "
            "which v5/v6 already used with other seeds; block B is descriptive. The method's DESIGN was informed by "
            "the published v6 eval results (which rival construction was tight); only the comparison among candidate "
            "variants used dev data. The success threshold is 0.80, i.e. RELAXED relative to v6 block A's 0.70 "
            "'keep' threshold; this is fixed prospectively here and does not reverse the v6 verdict (downgrade)."),
        "primary_method": {
            "name": "FDC-BF",
            "module": "dsswm.baselines.fdc_bet.make_variant('FDC-BF') -> FDCBet(kind='bennett', var_box='HG', "
                      "fpc=True, rect=False, split=(0.045, 0.005))",
            "frozen_hyperparameters": {
                "variant_tuple": list(VARIANTS["FDC-BF"]), "delta": 0.05, "delta_main": 0.045, "delta_var": 0.005,
                "cell_bound": "Bennett with exact finite-population variance n m (1-m) (N-n)/(N-1) phi(|u|/n), "
                              "phi(x) = e^x - 1 - x (Poisson-binomial representation of the hypergeometric law)",
                "variance_box": "exact hypergeometric two-sided test inversion per (cell, checkpoint), alpha_side = "
                                "delta_var / (2 S A K), running intersection over checkpoints (rect_v6.hg_mean_interval)",
                "union": "FDC's (q, pi, k) union: beta = ln(sum_q |Pi_{B_q}| K / delta_main)",
                "width": "w = min over lambda in lambda_0 * G of (beta + sum_c sup_{m in box} Psi_c(lambda a_c; m)) / "
                         "lambda, lambda_0 = sqrt(2 beta / V_bar_pair)",
                "lambda_grid": {"form": "exp(linspace(ln(1/200), ln 10, 161))", "n": int(len(_LAMBDA_GRID)),
                                "min": float(_LAMBDA_GRID[0]), "max": float(_LAMBDA_GRID[-1])},
                "design": "frozen 50/50 (b4_bal.balanced_alloc 0.5), identical schedule digest to FDC",
                "certification": "U_q(k) = max_{pi' in Pi_{B_q}} [Delta_hat(pi', pi_hat_q) + w(pi', pi_hat_q)] <= eps at "
                                 "the K = 20 pre-specified checkpoints, sticky answers, stop at 12/15 (block A: run "
                                 "continued to 15/15 for N100; see block A)"},
            "no_tuning_parameter_fitted": "FDC-BF has no data-fitted parameter; the lambda grid, split and box were "
                                          "fixed by design and are the same as for every other FDC-bet variant",
            "origin_commit": FDC_BET_ORIGIN_COMMIT,
            "origin_blob_sha256": git_blob_sha(FDC_BET_ORIGIN_COMMIT, fdc_bet_rel),
            "current_sha256": sha(CODE / fdc_bet_rel),
            "guarantee": "Theorem FDC-bet-1 (plan/fdc_bet_exploration.md s2.6): FWER <= 0.05 at the K pre-specified "
                         "checkpoints (both observation windows of fdc_theorem s6). Same guarantee strength as FDC, "
                         "RECT-ck-HG and RECT-ck-HG-live; weaker than the time-uniform HC-WoR / WSR rectangles.",
            "selection_disclosure": {
                "date": "2026-10-04",
                "variants_compared_on_dev": ["FDC-re", "FDC-FPC", "FDC-BF-L2", "FDC-BF", "FDC-KLF", "FDC-KLF+R",
                                             "FDC-MR[uniform]", "FDC-MR[geo2]", "FDC-MR[inv_d2]", "FDC-MR[front3]",
                                             "FDC-MR[uniform,nopart]", "FDC-MR[uniform,norect]",
                                             "FDC-MR[uniform,split.025]", "FDC-MR[front2]", "FDC-MR[front3x]",
                                             "FDC-MR[mixF50]", "FDC-MR[mixF30]", "FDC-MR[mixF50x]"],
                "selection_rule": "authors judgement on dev 950-999 + Lenta dev (not a pre-declared rule): most "
                                  "stable across datasets, simplest; MR[front3] slightly better on CR9 dev but ~0.1 "
                                  "worse on Lenta dev",
                "dev_gate": "GO if dev UB95(FDC-BF / strongest same-strength rectangle) < 0.70 (met: 0.677 vs HG, "
                            "0.695 vs HC-WoR)",
                "dev_numbers_selection_run": {"FDC-BF/RECT-ck-HG": [0.649, 0.677], "FDC-BF/HC-WoR": [0.663, 0.695],
                                              "FDC-BF/FDC": [0.808, 0.838], "format": "[geomean ratio, UB95]"},
                "dev_numbers_v7_runner_check": dev_numbers()},
        },
        "secondary_method": {"name": "FDC-MR[front3]", "module": "dsswm.baselines.fdc_mr.FDCMR(rho='front3') "
                             "(kind 'min', HG box, FPC, free rectangle, partition DP, split 0.045/0.005)",
                             "role": "descriptive only (both blocks); rho profile selected on CR9 tuning seeds 900-949"},
        "blocks": {
            "A": {
                "name": "confirmatory: FDC-BF vs same-design same-or-stronger-guarantee rectangles",
                "role": "CONFIRMATORY (the only verdict of this addendum)",
                "layer": "CR9", "half": "eval", "eps": 0.001, "K": 20, "delta": 0.05, "Q": 15,
                "eps_check": "same as v6 block A (eps 0.001)",
                "seeds": "33000-33199 (200 NEW eval streams)", "n_streams": 200,
                "run": "one run per (method, stream) with stop_k = 15; N80_pen (first checkpoint with >= 12/15 "
                       "certified and no false certificate up to it, else tau_R) and x12 are read off the same "
                       "trajectory -- the trajectory up to the 12/15 stop is a deterministic function of the seed and "
                       "identical to a stop-at-12 run (v6 R3 verified this for the v5 rivals; test "
                       "test_continuation_prefix_matches_stop_at_12)",
                "methods": list(VA.A_METHODS),
                "primary_family": list(VA.A_FAMILY),
                "primary_family_note": "RECT-ck-HG / RECT-ck-HG-live have the same K-checkpoint guarantee as "
                                       "FDC-BF; HC-WoR is time-uniform (stronger). RECT-ck-Bern is not in the family "
                                       "(dominated by RECT-ck-HG, matched-lemma contrast).",
                "rival_configs": "v5 frozen configs (CR9) + v6 tuned HC-WoR (exp/results/v6_gates/hc_config_cr9.json); "
                                 "RECT-ck-* and FDC have no parameter. No rival is re-tuned.",
                "primary_endpoint": "N80_pen (v5 definition)",
                "analysis": "dsswm.stats.v7_analysis.analyse_block('A'): validated complete matrix (16 methods x 200 "
                            "seeds), paired FDC-BF/r geomean ratio, paired percentile bootstrap B = 10^4 seed 42 (same "
                            "resample matrix for all rivals and endpoints), one-sided 95% UB, two-sided 95% CI",
                "decision_rule": {
                    "positive_result_achieved": "UB95(FDC-BF / r) < 0.80 for EVERY r in the primary family "
                                                "{RECT-ck-HG, RECT-ck-HG-live, HC-WoR} (IUT, conjunctive: no "
                                                "multiplicity correction) AND FDC-BF has 0/200 false streams (any false "
                                                "certificate up to the end of its run, i.e. up to 15/15 or tau_R; "
                                                "Clopper-Pearson 95% UB reported) AND replica pass",
                    "positive_result_not_achieved": "otherwise; components: faster_below_1 (0.80 <= UB* < 1), "
                                                    "not_faster (UB* >= 1), validity_failure, replica_fail. All "
                                                    "estimates are reported as they are.",
                    "threshold_origin": "0.80 given by the authors before any v7 run (dev UB was 0.677/0.695)"},
                "secondary_descriptive": [
                    "FDC-BF/r on N80_pen for RECT-ck-Bern, FDC (v5), B4-bal and the nine v5 rigorous rivals",
                    "FDC-BF/r on x12 (log-linear interpolated arrival count of the 12th sticky crossing; unpenalised)",
                    "FDC-BF/r on N100_pen (15/15, penalised) for every rival",
                    "FDC-MR[front3]/r on the same endpoints",
                    "per method: geomean N/tau_R, share at tau_R, false streams (CP UB), false streams by 12/15"],
            },
            "B": {
                "name": "Lenta LR9 descriptive re-use of an outcome-exposed log",
                "role": "DESCRIPTIVE ONLY: no verdict; not an untouched holdout (r4 computed full-table LR8 means; v6 "
                        "block B ran on the same LR9 eval half with seeds 32000-32199)",
                "dataset": "as v6 block B (lenta tidy.pkl, LR9 segmentation, split seed 6006), eval half",
                "eval_access": "dsswm.envs.lenta_v7.LentaLayerEnvV7: eval outcomes only for v7b_* tasks that pass "
                               "prereg_v7.addendum_gate",
                "seeds": "34000-34199 (200 NEW streams)", "eps": list(RV.LR9_EPS_GRID),
                "eps_selection": "none: all three reported", "stop": "12/15 (as v6 block B)",
                "methods": list(VA.B_METHODS),
                "rival_configs": "v6 LR9 frozen configs (exp/results/v6_gates/rival_configs_lr9.json)",
                "reported": "per eps: FDC-BF/r and FDC-MR[front3]/r ratios (N80_pen, x12) with CIs, geomean N/tau_R, "
                            "share at tau_R, false streams (CP UB)",
                "replica_fail": "appendix only with flag 'replica_fail'",
                "no_pooling": "never pooled with Criteo"},
        },
        "frozen_configs": frozen,
        "seed_manifest": {
            "fdc_bet_tuning_dev": "900-949 (CR9 dev)", "fdc_bet_selection_dev": "950-999 (CR9 dev, LR9 dev)",
            "v7_runner_check_dev": "950-999 (v7a_pilot, v7b_pilot; reproduce the selection run)",
            "A_eval": "33000-33199", "B_eval": "34000-34199", "bootstrap_seed": 42, "lr9_split_seed": 6006,
            "never_used_check": "2026-10-04: grep of every results json/jsonl under the workspace for seed / "
                                "perm_seed 33000-33199 and 34000-34199: 0 hits; no code references these ranges "
                                "except the v7 runner/analysis; supports 'no saved run', cannot prove no unsaved access",
            "previously_used_eval": "30000-30199 (v5, v6 C), 31000-31199 (v6 A), 32000-32199 (v6 B)"},
        "replica_check": {
            "implementation": "dsswm/stats/v7_replica.py + run_r5s_v7.py --task <t> --replica",
            "R1": f"independent process re-runs the first {R1_N_SEEDS} seeds of each eval task for every method and "
                  "eps; canonical rows (all fields except timing) identical",
            "R1_excluded_fields": list(TIMING_FIELDS),
            "R2": "schedule_digest identical within every (seed, eps) across the registered 50/50 group, full matrix",
            "R2_groups": {k: list(v) for k, v in COMPARISON_GROUPS.items() if k.startswith("v7") and "full" in k},
            "integrity": "as v6: analysis recomputes R1/R2 from actual rows, the stored report must reconcile exactly; "
                         "eval rows must equal the git-committed seal (exp/results/full/v7_seals)",
            "on_failure": {"A": "verdict component 'replica_fail' -> positive_result_not_achieved",
                           "B": "appendix only, flag 'replica_fail'"}},
        "eval_tasks": {t: {"block": RV.TASKS[t]["block"], "seeds": f"{RV.TASKS[t]['seeds'][0]}-"
                           f"{RV.TASKS[t]['seeds'][-1]}", "eps": list(RV.TASKS[t]["eps"]),
                           "stop_k": RV.TASKS[t]["stop_k"], "methods": list(RV.TASKS[t]["methods"])}
                       for t in ("v7a_full_a", "v7a_full_b", "v7b_full_a", "v7b_full_b")},
        "command": "cd exp/code && python run_r5s_v7.py --task <task> [--workers 4]; then --task <task> --replica; "
                   "then --analyse A|B. Every step re-checks the locked v7 addendum (and through it v5 + v6).",
        "eval_seal": "as v6 (v7_seal.write_and_commit_seal: seal committed alone at task completion, sidecar ref "
                     "commit; replica and analysis verify the seal against git history and the current rows)",
        "threat_model": "as v6: accidental drift (code, inputs, raw data, configs), partial / resumed runs and post-hoc "
                        "edits of results, reports or seals are detected against git history; deliberate rewriting of "
                        "git history is out of scope.",
        "declarations": [
            "Post-hoc deviation: FDC-BF replaces FDC as the primary method after the v6 eval block A downgrade. The "
            "paper must state this (date 2026-10-04, reason: v6 block A, dev-only selection among the variants "
            "listed) and must not present FDC-BF as the originally registered method.",
            "Dev seeds 950-999 were used both to select FDC-BF and for the v7 runner check; dev numbers are not "
            "evidence.",
            "Rivals are not re-tuned for v7; HC-WoR keeps its v6 dev-tuned config; no stronger rival family was "
            "searched for after v6. Stronger valid rivals may exist; conclusions are limited to the registered ones.",
            "Block A runs every method to 15/15 (stop_k = 15) so N80, x12 and N100 come from one trajectory; the 12/15 "
            "prefix equals the stop-at-12 run by construction.",
            "Block B re-uses the outcome-exposed LR9 eval half already used by v6 block B (different seeds) and is "
            "descriptive.",
            "Timings: up to 3 runner processes x 4 workers concurrently; biased up."],
        "code_sha256": {p: sha(CODE / p) for p in RV.CODE_FILES + RV.LOCK_ONLY_FILES},
        "input_sha256": {p: sha(WS / p) for p in INPUTS},
        "data_sha256": data_hashes(),
        "lock_procedure": "review (external reviewer <= 2 rounds) -> git commit (code + inputs) -> prereg_v7.finalize_addendum("
                          "git_commit) writes plan/prereg_lock_v7_addendum.json (refuses to overwrite; verifies the "
                          "commit contains the bound code) -> commit the lock -> eval tasks, replicas, analyses.",
        "git_commit": None,
    }
    add["sha256_draft"] = prereg.canonical_hash({k: v for k, v in add.items() if k != "sha256_draft"})
    out = prereg_v7_draft_path()
    out.write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(f"wrote {out} sha256_draft={add['sha256_draft']}")


def prereg_v7_draft_path():
    from dsswm.stats.prereg_v7 import DRAFT_PATH
    return DRAFT_PATH


if __name__ == "__main__":
    main()
