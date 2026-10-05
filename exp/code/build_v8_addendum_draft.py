"""Build plan/prereg_lock_v8_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage (cwd exp/code):  .venv/bin/python3 build_v8_addendum_draft.py
Locking: commit code + inputs (code freeze), then dsswm.stats.prereg_v8.finalize_addendum(git_commit), commit the lock
file alone.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.baselines.fdc_bet import VARIANTS  # noqa: E402
from dsswm.envs.data_v8 import X5_ROOT, data_hashes_v8  # noqa: E402
from dsswm.stats import prereg, prereg_v7  # noqa: E402
from dsswm.stats import v8_analysis as VA  # noqa: E402
from dsswm.stats import v8_replica as VR  # noqa: E402
from dsswm.stats.v6_replica import R1_N_SEEDS, TIMING_FIELDS  # noqa: E402

import run_r5s_v8 as RV  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT
INPUTS = ["plan/prereg_lock.json", "plan/prereg_lock_v6_addendum.json", "plan/prereg_lock_v7_addendum.json",
          "plan/prereg_lock_v8_addendum_DRAFT.md", "plan/prereg_lock_v8_addendum.md",
          "plan/theory/fdc_theorem.md", "plan/fdc_bet_exploration.md",
          "exp/results/v8_gates/x9_configs.json", "exp/results/v8_gates/cr_configs.json",
          "exp/results/v6_gates/hc_config_cr9.json",
          "exp/results/pilots/pjc_v8/selection.json", "exp/results/pilots/pjc_v8/tune/results.jsonl",
          "exp/results/pilots/pjc_v8/x5tune_v8/results.jsonl", "exp/results/pilots/pjc_v8/x5tune/results.jsonl",
          "exp/results/pilots/pjc_v8/x5rep/results.jsonl", "exp/results/pilots/pjc_v8/x5/results.jsonl",
          "exp/results/pilots/pjc_v8/analysis_x5rep.json", "exp/results/pilots/pjc_v8/analysis_dev.json",
          "exp/results/pilots/pjc_v8/toy/summary.json",
          "exp/results/full/v7_summary.md", "exp/results/full/v7_analysis_A.json",
          "exp/results/pilots/v8a_pilot/summary.json", "exp/results/pilots/v8a_pilot_b/summary.json",
          "exp/results/pilots/v8_analysis_A_dev.json"]
V6_SUMMARY_ERRATUM_COMMIT = "e740f421af0aef8eef4daed764a13bda3db7cea8"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev_numbers():
    p = WS / "exp/results/pilots/v8_analysis_A_dev.json"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    out = {}
    for e, pe in a["per_eps"].items():
        prim = pe["FDC-BF|N80_pen"]
        out[e] = {m: {"ratio": round(v["geomean_ratio"], 3), "ub95": round(v["ub95_one_sided"], 3)}
                  for m, v in prim.items()}
    return {"seeds": "X5 dev 950-999 (v8a_pilot / v8a_pilot_b, runner check; frozen picks)", "FDC-BF/r": out,
            "decision_if_applied": a.get("decision")}


def v7_known_drift():
    v7 = prereg.load_lock(prereg_v7.ADDENDUM_PATH)
    from dsswm.stats import prereg_v6
    out = {}
    for rel in prereg_v6.v6_input_drift(v7, WS):
        out[rel] = {"locked_sha256": v7["input_sha256"][rel], "current_sha256": sha(WS / rel),
                    "commit": V6_SUMMARY_ERRATUM_COMMIT if rel.endswith("v6_summary.md") else None,
                    "reason": ("dated erratum line appended to v6_summary.md during the v7-r2 paper revision "
                               "(FDC slower on 2/200 v6 block-A streams vs each rectangle); original sentence kept; "
                               "no number of any registered result changed") if rel.endswith("v6_summary.md")
                    else "UNEXPLAINED"}
    if any(d["reason"] == "UNEXPLAINED" for d in out.values()):
        raise SystemExit(f"unexplained v7 input drift: {out}")
    return out


def main():
    v7 = prereg_v7.load_locked_addendum(None, verify_inputs=False)
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    x9 = json.loads((WS / "exp/results/v8_gates/x9_configs.json").read_text())
    cr = json.loads((WS / "exp/results/v8_gates/cr_configs.json").read_text())
    frozen = {"X9": x9["selected"], "CR": cr["selected"]}
    prov = json.loads((X5_ROOT / "PROVENANCE.json").read_text())
    rect_pick, hc_pick = x9["selected"]["RECT-ck-HG"], x9["selected"]["HC-WoR"]
    add = {
        "version": "5+v8-addendum", "status": "draft", "written_at": "2026-10-04",
        "written_by": "author (round 8; design approved by the authors 2026-10-04)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v7["addendum_to"]["sha256"],
                        "git_commit": v7["addendum_to"].get("git_commit")},
        "v6_addendum": {"path": "plan/prereg_lock_v6_addendum.json", "sha256": v7["v6_addendum"]["sha256"],
                        "git_commit": v7["v6_addendum"].get("git_commit")},
        "v7_addendum": {"path": "plan/prereg_lock_v7_addendum.json", "sha256": v7["sha256"],
                        "git_commit": v7["git_commit"],
                        "block_A_verdict": "positive_result_achieved (CR9: UB* = 0.761 < 0.80 vs RECT-ck-HG / "
                                           "RECT-ck-HG-live / HC-WoR; 0/200 false streams; replica pass)"},
        "v7_known_input_drift": v7_known_drift(),
        "statement": (
            "Append-only addendum. It edits neither lock v5 nor the v6 / v7 addenda and cannot change their verdicts. "
            "It registers ONE confirmatory test (block A) of FDC-BF (frozen since v7, no parameter changed) on the "
            "UNTOUCHED eval half of a NEW real table (X5 RetailHero SMS-campaign RCT, blinded hashed split), against "
            "(F) three rectangles with read plans re-tuned on that table's dev half (superiority, UB95 < 0.60) and "
            "(G) two PJC-BF joint-certificate rivals, one outcome-adaptive (non-inferiority, UB95 < 1.05), plus two descriptive "
            "blocks: B (CR12 scale; outcome-exposed eval half) and C (post-hoc paired rivals on the v7 block-A CR9 "
            "streams). Disclosures: (i) the PJC family was designed AFTER the v7 eval results, in response to the "
            "two v7 reviews; (ii) X5 RetailHero was chosen AFTER v7 (Criteo has no untouched rows; MegaFon is a "
            "generated synthetic table and was rejected as 'real data'); (iii) the thresholds 0.60 and 1.05 and the "
            "decision eps 0.02 were sized from X5 dev numbers (seeds 900-949); (iv) a v7-bound input "
            "(v6_summary.md) received a dated erratum after the v7 lock (v7_known_input_drift); (v) FDC-BF remains a "
            "post-hoc method relative to v5 / v6."),
        "primary_method": {
            "name": "FDC-BF", "module": "dsswm.baselines.fdc_bet.make_variant('FDC-BF')",
            "variant_tuple": list(VARIANTS["FDC-BF"]), "delta": 0.05,
            "current_sha256": sha(CODE / "dsswm/baselines/fdc_bet.py"),
            "v7_bound_sha256": v7["code_sha256"]["dsswm/baselines/fdc_bet.py"],
            "no_tuning": "FDC-BF is not re-tuned on X5 (no data-fitted parameter); frozen 50/50 design",
            "guarantee": "Theorem FDC-bet-1: FWER <= 0.05 at the K = 20 pre-specified checkpoints"},
        "rivals": {
            "F (rectangles, superiority)": {
                "RECT-ck-HG": f"v6 exact-checkpoint HG rectangle with its own frozen read plan, X5 dev pick "
                              f"{rect_pick['pick']!r} from {{0.3, 0.4, 0.5, 0.6, 0.7, ney, ney23}}",
                "RECT-ck-HG-live": "v6, parameter-free (frozen 50/50; as registered in the approved design)",
                "HC-WoR": f"v6 time-uniform hedged-capital WoR rectangle, X5 dev pick {hc_pick['pick']!r} from the "
                          "v6 schedule grid extended by the x5tune target fractions (9 schedules) x the 7 read plans "
                          "(63 configs)"},
            "G (phased adaptive joint certificates, non-inferiority)": {
                "PJC-local": f"PJC-BF local (RAGE-style phase reset) X5 dev pick {frozen['X9']['PJC-local']['pick']!r} "
                             "from 17 configs (16 phase-reset configs + the no-reset member b=() added after the "
                             "external reviewer v8 review, round 1)",
                "PJC-menu": f"PJC-BF-M (menu-path union, all data) X5 dev pick {frozen['X9']['PJC-menu']['pick']!r} "
                            "from 10 configs",
                "validity": "FWER <= 0.05 at the K checkpoints (FDC-bet-1 per phase / per menu path); "
                            "dsswm/tests/test_pjc_bf.py; toy FWER 0/200 (pjc_v8/toy)",
                "phase_activation": "REGISTERED AS IMPLEMENTED: a new phase plan is computed at its boundary "
                                    "checkpoint (from data up to it) and takes effect at the runner's next replan "
                                    "batch (<= replan_interval arrivals later; e.g. X5 eval: checkpoint 3020, replan "
                                    "at 3200). The arms in between follow the previous phase's count-only preference, "
                                    "so the phase-local counts are determined by the phase-start history and the outcome-free future arrival sequence and the certificate stays "
                                    "valid; the dev tuning used the same behaviour.",
                "no_reset_member": "PJC-BF[local,b=,half] = one phase, deterministic 50/50 count tracking, joint "
                                   "Bennett-FPC width: it differs from FDC-BF only in the arm-assignment rule "
                                   "(deterministic deficit tracking vs FDC-BF's iid 50/50 draws). It is selected as "
                                   "PJC-local on both X5 and CR9 dev, so G-local is NOT an adaptive design; G-menu is."},
            "tuning": {"task": "exp/results/pilots/pjc_v8/x5tune_v8", "seeds": "X5 dev 900-949", "eps": 0.02,
                       "rule": x9["rule"], "tables": "exp/results/v8_gates/x9_configs.json",
                       "plans": "Neyman / sd^(2/3) matrices frozen from the X5 dev half cell variances"},
        },
        "blocks": {
            "A": {
                "role": "CONFIRMATORY (the only verdict of this addendum)",
                "table": "X5 RetailHero (HF pytorch-lifestream/retailhero-uplift, uplift_train joined with clients); "
                         "eval half of the blinded split (100,393 rows): eval_labels.pkl + eval_outcome.npy",
                "split": {k: prov[k] for k in ("salt", "dev_frac", "split_rule", "row_key")},
                "x5_provenance_sha256": sha(X5_ROOT / "PROVENANCE.json"),
                "eval_access": "dsswm.envs.x5_v8.X5LayerEnv('eval', eval_task_id): only for v8a_full* tasks passing "
                               "prereg_v8.addendum_gate; file hashes checked against PROVENANCE.json",
                "segmentation": "X9 = gender {U, F, M} x age band {<38, [38,53) or invalid, >=53} (features only)",
                "problems": "fr.cr_problems('visit'): 15 budgets, treatment cost 1, 512 policies",
                "K": 20, "n_min": 2000, "delta": 0.05, "Q": 15, "stop": "stop_k = 15 (N80 at 12/15 from the same "
                                                                       "trajectory)",
                "eps": list(VA.A_EPS), "decision_eps": VA.DECISION_EPS,
                "seeds": "35000-35199 (200 new streams)", "n_streams": 200,
                "tasks": {"v8a_full_a": list(VA.A_CORE) + ["(all three eps)"],
                          "v8a_full_b": list(VA.A_DESCR) + ["(eps 0.02 only, descriptive)"]},
                "F_family": list(VA.F_FAMILY), "G_family": list(VA.G_FAMILY),
                "primary_endpoint": "N80_pen (v5 definition)",
                "analysis": "dsswm.stats.v8_analysis.analyse_block('A'): paired FDC-BF/r geomean ratio, paired "
                            "percentile bootstrap B = 10^4 seed 42 (one resample matrix for every comparator), "
                            "one-sided 95% UB",
                "decision_rule": {
                    "positive_result_achieved": "at eps 0.02: UB95(FDC-BF/r) < 0.60 for every r in F AND "
                                                "UB95(FDC-BF/r) < 1.05 for every r in G AND FDC-BF 0/200 false streams "
                                                "(whole run) AND replica pass for v8a_full_a and v8a_full_b (IUT)",
                    "positive_result_not_achieved": "otherwise; components rectangle_superiority_failed "
                                                    "(faster_below_1 / not_faster), joint_adaptive_rival_not_inferior_"
                                                    "failed, validity_failure, replica_fail",
                    "wording": "a pass supports 'faster than the rectangles' and 'not slower than PJC by more than 5%'; "
                               "it does NOT support 'faster than PJC' (G superiority UB < 1 is descriptive only). "
                               "If only the G component fails the headline must become 'joint aggregation itself "
                               "(incl. its phased adaptive form) is the source of the gain'.",
                    "threshold_origin": "0.60: X5 dev UB vs F 0.350-0.360 (seeds 900-949 / 930-949, eps 0.02) plus "
                                        "a margin for a different 100k-row half; 1.05: worst dev UB vs G 1.042 "
                                        "(eps 0.01, 20 seeds), 0.97-0.98 at eps 0.02"},
                "secondary_descriptive": [
                    "eps 0.015 and 0.03: all ratios", "x12 and N100_pen ratios", "FDC-MR[front3]/r",
                    "RECT-ck-BF / RECT-ck-BF+box (matched Bennett rectangle: same cell inequality, aggregation only)",
                    "plug-ins (no guarantee) B2-fav, B3-fav, Peace-fav, B5, B2-fav-tight, FIX-bal-fav, Peace-fav-bal: "
                    "speed AND false streams", "v5 literature rivals with their v5 CR9 configs (not re-tuned on X5)",
                    "per method geomean N/tau_R, share at tau_R, false streams (CP 95% UB)"],
                "dev_runner_check": dev_numbers(),
            },
            "B": {
                "role": "DESCRIPTIVE ONLY (CR12 eval half already exposed: r5 FDC seeds 30000-30049)",
                "layer": "CR12", "half": "eval", "eps": 0.001, "seeds": "35200-35399", "stop": "12/15",
                "methods": list(VA.B_METHODS), "configs": "CR9-tuned (cr_configs.json); PJC picks from CR9 dev",
                "reported": "ratios, false streams, per-checkpoint certificate time (4,096 policies)"},
            "C": {
                "role": "DESCRIPTIVE, POST HOC, conditional on a table used three times (CR9 eval half)",
                "layer": "CR9", "half": "eval", "eps": 0.001, "seeds": "33000-33199 (the v7 block-A streams)",
                "stop": "stop_k = 15", "methods_new": list(VA.C_NEW),
                "joined_with": "git-sealed v7 block-A rows of the same streams (verified against the v7 seals)",
                "reproduction_check": "FDC-BF re-run vs the sealed v7 FDC-BF rows on " + ", ".join(RV.REPRO_FIELDS),
                "own_plan_note": "CR9 dev own-plan tuning picked 50/50 for RECT-ck-HG and HC-WoR (identical to the v7 "
                                 "rows); the Neyman variants are descriptive"},
        },
        "frozen_configs": frozen,
        "seed_manifest": {
            "x5_dev_tuning": "900-949 (x5tune_v8; earlier feasibility on 900-949 as well)",
            "x5_dev_runner_check": "950-999 (v8a_pilot / v8a_pilot_b; never used before this addendum)",
            "A_eval": "35000-35199", "B_eval": "35200-35399", "C_eval_reused": "33000-33199 (v7 block A)",
            "bootstrap_seed": 42,
            "never_used_check": "2026-10-04: grep of every json/jsonl under the workspace for seed / perm_seed "
                                "35xxx: 0 hits; no code references 35000-35399 except the v8 runner / analysis",
            "previously_used_eval": "30000-30199 (v5, v6 C), 31000-31199 (v6 A), 32000-32199 (v6 B), "
                                    "33000-33199 (v7 A), 34000-34199 (v7 B)"},
        "replica_check": {
            "implementation": "dsswm/stats/v8_replica.py + run_r5s_v8.py --task <t> --replica",
            "R1": f"independent process re-runs the first {R1_N_SEEDS} seeds of each eval task; canonical rows "
                  "identical", "R1_excluded_fields": list(TIMING_FIELDS),
            "R2": "schedule_digest identical within every (seed, eps) across each registered design group",
            "R2_groups": {t: [list(g) for g in VR.groups_for(t, frozen)] for t in RV.TASKS if "full" in t},
            "R2b": "arrival_digest (arrival segments + pool record orders) identical across ALL methods of a task "
                   "within every (seed, eps); also checked across the tasks of a block in the analysis",
            "on_failure": {"A": "component replica_fail -> positive_result_not_achieved",
                           "B": "flag replica_fail", "C": "flag replica_fail"}},
        "eval_tasks": {t: {"block": RV.TASKS[t]["block"], "layer": RV.TASKS[t]["layer"],
                           "seeds": f"{RV.TASKS[t]['seeds'][0]}-{RV.TASKS[t]['seeds'][-1]}",
                           "eps": list(RV.TASKS[t]["eps"]), "stop_k": RV.TASKS[t]["stop_k"],
                           "methods": list(RV.TASKS[t]["methods"])}
                       for t in ("v8a_full_a", "v8a_full_b", "v8b_full_a", "v8b_full_b", "v8c_full")},
        "command": "cd exp/code && python run_r5s_v8.py --task <task> [--workers 4]; --task <task> --replica; "
                   "--analyse A|B|C. Every step re-checks the locked v8 addendum (and through it v7, v6, v5).",
        "eval_seal": "as v7 (v8_seal: seal committed alone at task completion + sidecar ref commit; single-commit "
                     "history anchor)",
        "threat_model": "as v6/v7: accidental drift, partial/resumed runs and post-hoc edits are detected against git "
                        "history; deliberate history rewriting is out of scope. X5 blinding is PROCEDURAL: the raw "
                        "csv (with all outcomes) is kept under x5_retailhero/raw for provenance; ingest_v8_fresh.py "
                        "parsed the full target column and wrote the eval outcomes unreduced to eval_outcome.npy; the "
                        "integrity checks (data_sha256, PROVENANCE) read that file's bytes to hash them. No eval-"
                        "outcome statistic was computed, inspected or used for design, tuning or thresholds; "
                        "X5LayerEnv('eval') is gated by this lock. This cannot be verified retrospectively beyond "
                        "the code and git history. The v5 lock's own input manifest (e.g. plan/task_plan.json) is not "
                        "re-verified by the v6/v7/v8 gate chain (inherited scope); the chain verifies the v5 lock "
                        "file, v5-frozen code and every v6/v7/v8 binding.",
        "declarations": [
            "FDC-BF is unchanged since v7 and remains a post-hoc method relative to v5/v6.",
            "The PJC family was designed after the v7 eval results, prompted by the v7 reviews (P0-3); its tuning "
            "used dev data only (CR9 dev 900-949, X5 dev 900-949).",
            "X5 RetailHero was selected after v7 because Criteo has no untouched rows (criteo_uplift is a synthetic "
            "stand-in; the real 13.98M-row table is fully split into dev/eval halves already used by v5-v7) and "
            "MegaFon is documented as generated synthetic data.",
            "The thresholds 0.60 (F) and 1.05 (G) and the decision eps 0.02 were sized from X5 dev numbers; the "
            "X5 dev runner check (950-999) was run after these were fixed and does not change them.",
            "Rivals F and G were re-tuned on X5 dev (FDC-BF was not). RECT-ck-HG-live is not re-tuned (as in the "
            "approved design). The literature rivals keep their v5 CR9 configs (descriptive only).",
            "Block B and C are descriptive; block C is post hoc and conditional on a table used three times.",
            "v7_known_input_drift: v6_summary.md received a dated erratum after the v7 lock (commit e740f421); the "
            "v8 gate accepts exactly this drift and nothing else.",
            "external reviewer v8 review round 1 (reviews/v8_review.md) changed the design before the lock: (a) the no-reset PJC "
            "member was added to the local grid and won the dev tuning on X5 and CR9; (b) R2b now hashes the "
            "schedule actually consumed by the run, and the cross-task R2b enters the replica status; (c) "
            "PROVENANCE.json and the final lock markdown are bound; (d) the delayed PJC phase activation is "
            "registered as implemented; (e) blinding wording made procedural. The X5 dev runner check (950-999) was "
            "re-run with the final code.",
            "Timings: up to 3 runner processes x 4 workers concurrently; biased up."],
        "code_sha256": {p: sha(CODE / p) for p in RV.CODE_FILES + RV.LOCK_ONLY_FILES},
        "input_sha256": {p: sha(WS / p) for p in INPUTS},
        "data_sha256": data_hashes_v8(),
        "lock_procedure": "external reviewer review (<= 2 rounds) -> code-freeze commit -> prereg_v8.finalize_addendum(git_commit) "
                          "-> commit the lock file alone -> eval tasks, replicas, analyses.",
        "git_commit": None,
    }
    add["sha256_draft"] = prereg.canonical_hash({k: v for k, v in add.items() if k != "sha256_draft"})
    from dsswm.stats.prereg_v8 import DRAFT_PATH
    DRAFT_PATH.write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(f"wrote {DRAFT_PATH} sha256_draft={add['sha256_draft']}")


if __name__ == "__main__":
    main()
