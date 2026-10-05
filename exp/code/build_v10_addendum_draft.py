"""Build plan/prereg_lock_v10_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage (cwd exp/code):  .venv/bin/python3 build_v10_addendum_draft.py
Locking (authors): external reviewer lock review -> rebuild this draft -> commit code + inputs + draft (code freeze) ->
prereg_v10.finalize_addendum(<freeze commit>) -> commit the lock file alone -> eval tasks, replicas, analyses.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.envs.seg_v10_eval import data_hashes_v10, load_frozen_seg  # noqa: E402
from dsswm.stats import prereg, prereg_v6, prereg_v9  # noqa: E402
from dsswm.stats import v10_analysis as VA  # noqa: E402
from dsswm.stats import v10_replica as VR  # noqa: E402
from dsswm.stats.v6_replica import R1_N_SEEDS, TIMING_FIELDS  # noqa: E402

import run_v10 as RV  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT
INPUTS = ["plan/prereg_lock.json", "plan/prereg_lock_v6_addendum.json", "plan/prereg_lock_v7_addendum.json",
          "plan/prereg_lock_v8_addendum.json", "plan/prereg_lock_v9_addendum.json",
          "plan/prereg_lock_v10_addendum_DRAFT.md", "plan/fdc_dp_theory.md", "plan/fdc_dp_dev_report.md",
          "plan/v10_plan.md", "reviews/fdc_dp_review_r1.md", "reviews/v10_lock_review.md",
          "exp/results/v8_gates/x9_configs.json", "exp/results/v8_gates/cr_configs.json",
          "exp/results/v10_gates/seg_v10_frozen.json", "exp/results/v10_gates/score_model_x5.pkl",
          "exp/results/v10_gates/score_model_lenta.pkl",
          "exp/results/pilots/fdc_dp/analysis.json", "exp/results/pilots/fdc_dp_v2/analysis.json",
          "exp/results/pilots/v10_analysis_A_dev.json", "exp/results/pilots/v10_analysis_B_dev.json",
          "exp/results/pilots/v10_analysis_D_dev.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev_numbers():
    out = {}
    for b in ("A", "B", "D"):
        p = WS / f"exp/results/pilots/v10_analysis_{b}_dev.json"
        if not p.exists():
            out[b] = None
            continue
        a = json.loads(p.read_text())
        out[b] = {"seeds": "dev 950-999 (v10 runner check: final code, frozen segmentation, frozen configs)",
                  "cells": {t: {"eps": c["eps"],
                                "ratio_vs_rival": round(c["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["geomean_ratio"],
                                                        3),
                                "ub95_vs_rival": round(c["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["ub95_one_sided"],
                                                       3),
                                "ratio_vs_hc": round(c["comparisons"][f"{VA.PRIMARY}/HC-WoR-DP"]["geomean_ratio"], 3),
                                "primary_false_streams": c["primary_false_streams"],
                                "all_rigorous_false_streams": c["all_rigorous_false_streams_total"],
                                "rival_share_N80_lt_tau": c["methods"][VA.RIVAL]["share_N80_lt_tau"],
                                "exhaustion_at_k80": {m: c["methods"][m]["exhaustion_at_k80_mean"]
                                                      for m in (VA.PRIMARY, VA.RIVAL)}}
                            for t, c in a["cells"].items()},
                  "decision_if_applied": (a.get("decision") or {}).get("verdict"),
                  "replica": a["replica_status_combined"]["status"]}
    p = WS / "exp/results/pilots/fdc_dp_v2/analysis.json"
    if p.exists():
        a = json.loads(p.read_text())
        out["eps_rule_v2"] = {t: {k: v.get(k) for k in ("selected_eps", "rival", "successes", "n", "fallback")}
                              for t, v in a["eps_rule"].items()}
        out["rule_selected_cells_v2"] = a["rule_selected_cells"]
        out["go_gate_v2"] = {k: a[k]["pass"] for k in ("go_criterion_3", "go_criterion_4_v1_primary_eps",
                                                       "go_criterion_4_rule_selected")}
    return out


def v9_known_drift():
    v9 = prereg.load_lock(prereg_v9.ADDENDUM_PATH)
    drift = prereg_v6.v6_input_drift(v9, WS)
    if drift:
        raise SystemExit(f"unexplained v9 input drift: {drift}")
    return {}


def cell_rows():
    out = {}
    for b, cells in VA.CELLS.items():
        for t, c in cells.items():
            out[t] = dict(c, block=b, role="confirmatory")
    for t, c in VA.DESC_CELLS.items():
        out[t] = dict(c, role="descriptive")
    return out


def main():
    v9 = prereg_v9.load_locked_addendum(None)
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    cfg, seg = RV.load_frozen()
    fz = load_frozen_seg()
    v2 = json.loads((WS / "exp/results/pilots/fdc_dp_v2/analysis.json").read_text())
    for b, cells in VA.CELLS.items():                       # the registered cells must be the rule's selections
        for t, c in cells.items():
            key = f"{c['data']}{'_' if c['data'] == 'x5' else ''}{c['S']}"
            r = v2["eps_rule"][key]
            if r.get("selected_eps") != c["eps"] or r.get("rival") != "RECT-BF-DP":
                raise SystemExit(f"{t}: registered eps {c['eps']} != v2 rule selection {r.get('selected_eps')} "
                                 f"(rival {r.get('rival')})")
    for t, c in VA.DESC_CELLS.items():
        key = f"{c['data']}{'_' if c['data'] == 'x5' else ''}{c['S']}"
        if v2["eps_rule"][key].get("selected_eps") is not None:
            raise SystemExit(f"{t}: descriptive cell but the v2 rule selected an eps")
    add = {
        "version": "5+v10-addendum", "status": "draft", "written_at": "2026-10-05",
        "written_by": "author (round 10; design by the authors 2026-10-05; external reviewer FDC-DP r1 fixes)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v9["addendum_to"]["sha256"],
                        "git_commit": v9["addendum_to"].get("git_commit")},
        "v6_addendum": {"path": "plan/prereg_lock_v6_addendum.json", "sha256": v9["v6_addendum"]["sha256"],
                        "git_commit": v9["v6_addendum"].get("git_commit")},
        "v7_addendum": {"path": "plan/prereg_lock_v7_addendum.json", "sha256": v9["v7_addendum"]["sha256"],
                        "git_commit": v9["v7_addendum"].get("git_commit")},
        "v8_addendum": {"path": "plan/prereg_lock_v8_addendum.json", "sha256": v9["v8_addendum"]["sha256"],
                        "git_commit": v9["v8_addendum"].get("git_commit")},
        "v9_addendum": {"path": "plan/prereg_lock_v9_addendum.json", "sha256": v9["sha256"],
                        "git_commit": v9["git_commit"]},
        "v9_known_input_drift": v9_known_drift(),
        "statement": (
            "Append-only addendum. It edits neither lock v5 nor the v6-v9 addenda and cannot change their verdicts. "
            "It registers two confirmatory blocks testing FDC-DP -- the FDC-BF joint certificate computed exactly "
            "(up to an abstaining node cap) over exponentially large segment-policy classes by multiple-choice "
            "knapsack DP and branch-and-bound -- read under its time-uniform guarantee and monitored at the K = 20 "
            "block grid, against the matched, parameter-free time-uniform Bennett rectangle RECT-BF-DP-TU, on "
            "frozen dev-trained uplift-score segmentations: A on the X5 RetailHero eval half (fresh seeds "
            "38000-38199) and B on the Lenta LR9 eval half (fresh seeds 38200-38399). Each block is an IUT over its "
            "cells; the blocks are reported separately and a joint statement needs both. Confirmatory cells and "
            "their eps were chosen on dev by the pre-stated rival-success rule (v2 dev, after the external reviewer FDC-DP r1 "
            "fixes). Inference is conditional on the two reused finite tables (fresh replay randomness only)."),
        "primary_method": {
            "name": VA.PRIMARY,
            "construction": "dsswm.baselines.fdc_dp.FDCDPTimeUniform(block_points = the K = 20 checkpoint grid, "
                            "scheme='b', node_limit 200000, grid ratio 2000^(1/160)): FDC-BF cells (Bennett-FPC, "
                            "exact-HG variance box at delta_var / (2 S A K)), ledger beta = ln(sum_q |Pi_q| K / "
                            "0.045) by exact counting DP, one centre-independent lambda grid per checkpoint, scheme "
                            "(a) then exact branch-and-bound; capped search abstains (not certified). Weighted "
                            "empirical centre (external reviewer r1 B1). Monitoring grid = block grid, so it is numerically "
                            "identical to FDC-DP(b) (unit tests).",
            "guarantee": "Theorem DP-1 (plan/fdc_dp_theory.md; external reviewer FDC-DP r1: valid) + Lemma TU: FWER <= 0.05 per "
                         "stream and problem family, for all t >= t_1, under the frozen outcome-free 50/50 schedule; "
                         "NOT under adaptive sampling; not simultaneous across seeds, tables, S or methods",
            "contribution": "computational: the statistical certificate is FDC-BF's; the new part is exact "
                            "certification over implicit classes of size up to ~1e19 (S = 64)",
            "fdc_dp_sha256": sha(CODE / "dsswm/baselines/fdc_dp.py"),
            "rect_dp_sha256": sha(CODE / "dsswm/baselines/rect_dp.py")},
        "rivals": {
            "primary": {VA.RIVAL: "dsswm.baselines.rect_dp.RectBFDPTU(block_points = K = 20 grid, box=True): the "
                                  "Bennett-FPC rectangle matched to FDC-DP (same cell inequality, exact-HG variance "
                                  "box, (0.045, 0.005) split, 50/50 design, 2 S A K one-sided events), radii frozen "
                                  "at block points (Lemma TU), exact DP over the policy class; parameter-free; "
                                  "identical to RECT-BF-DP on the block grid (unit test)"},
            "descriptive": {"HC-WoR-DP": f"hedged-capital WoR CS rectangle (time-uniform under any predictable "
                                         f"sampling) with the frozen v8 S = 9 configuration transferred UNTUNED at the "
                                         f"50/50 plan: {cfg['HC-WoR-DP']}",
                            "FDC-DP(a)": "single-column scheme (a) (conservative, no branch-and-bound)"},
            "claim_scope": "claims are restricted to the two named rectangle implementations (RECT-BF-DP-TU; "
                           "HC-WoR-DP as an untuned transfer); not 'the best valid rectangle' (external reviewer r1 s4)"},
        "blocks": {
            "A": {"role": "CONFIRMATORY (verdict 1 of 2)",
                  "table": "X5 RetailHero eval half (blinded hashed split of ingest_v8_fresh, 100,393 rows), the "
                           "whole half is the replay population",
                  "table_reuse": "3rd use of this eval half (v8 A 35000-35199, v9 B 37200-37399); fresh seeds; new "
                                 "segmentation = new problem on old rows",
                  "seeds": "38000-38199 (200 new streams)",
                  "cells": {t: dict(c) for t, c in VA.CELLS["A"].items()}},
            "B": {"role": "CONFIRMATORY (verdict 2 of 2)",
                  "table": "Lenta LR9 eval half (lenta_v6 split seed 6006 = complement of the dev half), the whole "
                           "half is the replay population",
                  "table_reuse": "this eval half was used by v6 B (32000-32199) and v7 B (34000-34199), both "
                                 "DESCRIPTIVE, and the table is outcome-exposed (r4 computed full-table LR8 cell means "
                                 "that include these rows; v6 exposure statement). v10 B is the first confirmatory use "
                                 "and the 3rd fresh-stream use; fresh seeds; new segmentation",
                  "seeds": "38200-38399 (200 new streams)",
                  "cells": {t: dict(c) for t, c in VA.CELLS["B"].items()},
                  "exhaustion_qualification": "Lenta is ~25 % control, so the 50/50 design exhausts control pools; "
                                              "certification can be exhaustion-driven. Every cell reports the mean "
                                              "fraction of exhausted cells (all / control / treatment) at each "
                                              "method's N80 checkpoint; a mechanism claim about pre-exhaustion widths "
                                              "is made only where these fractions are 0 for both methods"},
            "D": {"role": "DESCRIPTIVE (no verdict)", "cells": {t: dict(c) for t, c in VA.DESC_CELLS.items()},
                  "reason": "the rule found no eps on the declared grid at which the rectangles succeed (rivals "
                            "censored); reported as descriptive with the block-A seeds; its grid is never enlarged "
                            "after eval"},
            "decision_rule": {
                "statistic": "paired geometric-mean ratio of N80_pen, TU-FDC-DP(b) / RECT-BF-DP-TU, paired percentile "
                             "bootstrap over the 200 streams (B = 10^4, seed 42, one resample matrix for all cells "
                             "and comparators), one-sided UB95",
                "positive_result_achieved": "per block, for EVERY registered cell: UB95 < 0.80 AND TU-FDC-DP(b) "
                                            "0/200 false streams (whole run to 15/15 or tau); AND replica pass for "
                                            "every task of the block (IUT)",
                "positive_result_not_achieved": "otherwise; per-cell components rival_superiority_failed (sub-label "
                                                "faster_below_1 / not_faster), validity_failure; replica_fail",
                "per_cell_reporting": "each cell is also reported on its own",
                "multiplicity": "IUT within each block; the two blocks are reported separately; a joint X5 + Lenta "
                                "statement only if both blocks are positive; no other combination",
                "threshold_origin": "0.80 = go criterion 4 of plan/fdc_dp_theory.md (written before any dev run)"},
            "endpoint": {"N80_pen": "K = 20 checkpoint at which 12/15 problems are first certified, if no false "
                                    "certificate by then, else tau_R", "horizon": "stop_k = 15 (or tau_R); a false "
                                                                                  "stream = any false certificate "
                                                                                  "over the whole run",
                         "dev_deviation": "v1 dev runs (fdc_dp_dev_report v1) stopped at 12/15; v2 dev and v10 run "
                                          "to 15/15 (external reviewer r1 B2)"},
        },
        "eps_selection": {
            "rule": "plan/v10_plan.md (written before the eps >= 0.03 dev runs): smallest eps on the declared grid "
                    "(X5 0.02 / 0.03 / 0.04 / 0.05; Lenta 0.003 / 0.004 / 0.006 / 0.008) at which the dev-best of "
                    "the two named rectangles (lower geometric-mean N80_pen; tie -> RECT-BF-DP) has strict N80_pen "
                    "< tau on >= 40/50 dev streams; none -> descriptive cell; FDC-DP's speed is never used",
            "implementation": "run_fdc_dp_dev.select_eps (external reviewer r1 B2), applied mechanically to the v2 dev results "
                              "(exp/results/pilots/fdc_dp_v2/analysis.json)",
            "v2_result": {t: {k: v.get(k) for k in ("selected_eps", "rival", "successes", "n", "fallback")}
                          for t, v in v2["eps_rule"].items()}},
        "frozen_segmentation": seg,
        "frozen_segmentation_detail": {
            "score_model": "seg_v10 T-learner: two HistGradientBoostingClassifier (one per arm) with "
                           f"{fz['hgb_params']}, fit on the 30 % dev model split (seed {fz['model_split_seed']}); "
                           f"sklearn {fz['sklearn_version']}; pickled; eval scoring uses the pickles (no refit)",
            "features": {d: fz["data"][d]["features"] for d in fz["data"]},
            "segmentation": "cut points = minimum dev replay-pool score of seg_v10's equal-size score-quantile "
                            "segments 1..S-1 (ties broken by seed 10011 on dev); assignment segment = #{j : cut_j <= "
                            "score} (ties at a cut go up); dev rows reassigned vs seg_v10: "
                            + json.dumps({d: {S: e["dev_rows_reassigned_vs_seg_v10"] for S, e in v["S"].items()}
                                          for d, v in fz["data"].items()}),
            "problems": "frozen from dev: kappa[s, 1] = 8 for every segment, budgets floor(b_q 8 S), b_q = "
                        "0.10, 0.15, ..., 0.80 (treat at most floor(b_q S) segments); NOT recomputed from eval "
                        "weights; policy value uses the eval segment weights",
            "rows": "X5: eval_labels.pkl rows in file order (row ids 0..N-1), outcome eval_outcome.npy; Lenta: "
                    "tidy.pkl rows of the LR9 eval half (np.flatnonzero), outcome response_att",
            "checkpoints": "fr.checkpoints(n_min, tau_R, 20) with n_min 2000 (X5) / 5000 (Lenta), tau_R = eval N"},
        "frozen_configs": json.loads(json.dumps(cfg)),
        "seed_manifest": {
            "dev_runner_check": "seeds 950-999 on the dev replay pools (used before by the FDC-DP v1 / v2 dev runs)",
            "A_eval": "38000-38199", "B_eval": "38200-38399", "D_eval": "38000-38199 (block-A seeds)",
            "bootstrap_seed": 42,
            "never_used_check": "see plan/prereg_lock_v10_addendum_DRAFT.md s7",
            "previously_used_eval": "30000-30199, 31000-31199, 32000-32199, 33000-33199, 34000-34199, "
                                    "35000-35199, 35200-35399, 36000-36199, 37000-37799"},
        "replica_check": {
            "implementation": "dsswm/stats/v10_replica.py + run_v10.py --task <t> --replica",
            "R1": f"independent process re-runs the first {R1_N_SEEDS} seeds of each eval task; canonical rows "
                  "identical", "R1_excluded_fields": list(TIMING_FIELDS),
            "R2": "schedule_digest identical across all four methods (frozen 50/50 design) per (seed, eps)",
            "R2_groups": {t: [list(g) for g in VR.groups_for(t)] for t in RV.TASKS if "_full_" in t},
            "R2b": "arrival_digest identical across all methods of a task per (seed, eps)",
            "on_failure": "replica_fail -> positive_result_not_achieved for the block"},
        "eval_tasks": {t: {"block": RV.TASKS[t]["block"], "role": RV.TASKS[t]["role"], "layer": RV.TASKS[t]["layer"],
                           "data": RV.TASKS[t]["data"], "S": RV.TASKS[t]["S"], "eps": RV.TASKS[t]["eps"],
                           "seeds": f"{RV.TASKS[t]['seeds'][0]}-{RV.TASKS[t]['seeds'][-1]}",
                           "methods": list(RV.TASKS[t]["methods"]), "K": RV.K_GRID, "stop_k": 15, "n80_k": RV.N80_K}
                       for t in RV.TASKS if RV.TASKS[t]["half"] == "eval"},
        "command": "cd exp/code && python run_v10.py --task <task> [--workers 4]; --task <task> --replica; "
                   "--analyse A|B|D. Every step re-checks the locked v10 addendum (and through it v9-v5).",
        "eval_seal": "as v8 / v9 (v10_seal: seal committed alone at task completion + sidecar ref commit; "
                     "single-commit history anchor)",
        "threat_model": "as v6-v9: accidental drift, partial / resumed runs and post-hoc edits are detected against "
                        "git history; deliberate history rewriting is out of scope.",
        "declarations": [
            "FDC-DP's statistical certificate is FDC-BF's (Theorem DP-1); the contribution is computational.",
            "eps per cell chosen on dev by the pre-stated rival-success rule (rule written before the eps >= 0.03 "
            "dev runs; applied mechanically to the v2 dev results after the external reviewer r1 fixes).",
            "Table reuse: X5 eval half 3rd use; Lenta eval half 3rd fresh-stream use (v6 B, v7 B descriptive) and "
            "outcome-exposed by r4 full-table cell means; inference is conditional on the reused finite tables.",
            "HC-WoR-DP is an untuned transfer of the v8 S = 9 configuration (descriptive only).",
            "Lenta control-pool exhaustion: certification can be exhaustion-driven; exhaustion fractions at N80 are "
            "reported per cell and method.",
            "Dev deviation: v1 dev streams stopped at 12/15; v2 dev and v10 run to 15/15 (N80 at 12/15).",
            "v1 dev numbers (fdc_dp_dev_report v1) are superseded by v2 (weighted-centre fix B1).",
            "Claims are restricted to the named rectangle implementations.",
            "Inherited validation scope (external reviewer v10 lock r1 F2): v10 adds no input drift; it inherits v8/v9's "
            "documented exceptions (v8 accepts the recorded v7-input erratum in exp/results/full/v6_summary.md; the "
            "cascade does not revalidate v5's input manifest).",
            "Integrity scope (external reviewer v10 lock r1 F3): no eval outcome was USED by any v10 computation before the lock; "
            "_lenta_dev deserialises the whole Lenta pickle and keeps only dev outcomes, the LR9 split uses "
            "full-table covariates, and manifest hashing reads file bytes. Every v10 eval reader, including the "
            "exported read_eval_rows, runs eval_gate first.",
            "v2 reproduces all 4,800 common-stream N80_pen endpoints of v1 (hence ratios, bounds and eps "
            "selections); horizon, runtimes, recorded certificates and branch-only counts (25 vs 27) differ (F4).",
            "Timings: up to 3 runner processes x 4 workers concurrently with other jobs; biased up."],
        "code_sha256": {p: sha(CODE / p) for p in RV.CODE_FILES + RV.LOCK_ONLY_FILES},
        "input_sha256": {p: sha(WS / p) for p in INPUTS},
        "data_sha256": data_hashes_v10(),
        "dev_runner_check": dev_numbers(),
        "lock_procedure": "external reviewer lock review -> rebuild draft -> code-freeze commit -> "
                          "prereg_v10.finalize_addendum(git_commit) -> commit the lock file alone -> eval tasks, "
                          "replicas, analyses.",
        "git_commit": None,
    }
    add["cells_registry"] = cell_rows()
    add["sha256_draft"] = prereg.canonical_hash({k: v for k, v in add.items() if k != "sha256_draft"})
    from dsswm.stats.prereg_v10 import DRAFT_PATH
    DRAFT_PATH.write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(f"wrote {DRAFT_PATH} sha256_draft={add['sha256_draft']}")


if __name__ == "__main__":
    main()
