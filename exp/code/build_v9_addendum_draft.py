"""Build plan/prereg_lock_v9_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage (cwd exp/code):  .venv/bin/python3 build_v9_addendum_draft.py
Locking: commit code + inputs (code freeze), then dsswm.stats.prereg_v9.finalize_addendum(git_commit), commit the lock
file alone.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.baselines.fdc_bet import VARIANTS  # noqa: E402
from dsswm.envs.data_v9 import data_hashes_v9  # noqa: E402
from dsswm.stats import prereg, prereg_v6, prereg_v8  # noqa: E402
from dsswm.stats import v9_analysis as VA  # noqa: E402
from dsswm.stats import v9_replica as VR  # noqa: E402
from dsswm.stats.v6_replica import R1_N_SEEDS, TIMING_FIELDS  # noqa: E402

import run_r5s_v9 as RV  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT
INPUTS = ["plan/prereg_lock.json", "plan/prereg_lock_v6_addendum.json", "plan/prereg_lock_v7_addendum.json",
          "plan/prereg_lock_v8_addendum.json",
          "plan/prereg_lock_v9_addendum_DRAFT.md", "plan/prereg_lock_v9_addendum.md",
          "plan/v9_candidates_theory.md", "plan/v9_candidates_report.md", "plan/fdc_hg_dev_report.md",
          "plan/hillstrom_v9_dev_report.md", "plan/hillstrom_exposure_audit.md",
          "reviews/v9_candidates_review.md", "reviews/fdc_hg_review.md", "reviews/v9_lock_review.md",
          "exp/results/v8_gates/x9_configs.json", "exp/results/v8_gates/cr_configs.json",
          "exp/results/v9_gates/hv9_configs.json",
          "exp/results/pilots/hillstrom_v9/selection_b.json", "exp/results/pilots/hillstrom_v9/design_b.json",
          "exp/results/pilots/hillstrom_v9/selection.json", "exp/results/pilots/hillstrom_v9/design.json",
          "exp/results/pilots/hillstrom_v9/summary_b.json", "exp/results/pilots/hillstrom_v9/summary.json",
          "exp/results/pilots/v9_candidates/analysis_dev.json", "exp/results/pilots/fdc_hg/analysis_dev.json",
          "exp/results/pilots/v9_analysis_A_dev.json", "exp/results/pilots/v9_analysis_B_dev.json",
          "exp/results/pilots/v9_analysis_C_dev.json", "exp/results/pilots/v9_toy_dense/summary.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev_numbers():
    out = {}
    for b in ("A", "B"):
        p = WS / f"exp/results/pilots/v9_analysis_{b}_dev.json"
        if not p.exists():
            out[b] = None
            continue
        a = json.loads(p.read_text())
        prim = a["comparisons"][f"{VA.PRIMARY}|N80_pen"]
        k20 = a["comparisons"]["FDC-BF@K|N80_pen"]
        out[b] = {"seeds": "dev 950-999 (v9 runner check, final code, frozen configs)",
                  "TU-FDC@D/r": {m: {"ratio": round(v["geomean_ratio"], 3), "ub95": round(v["ub95_one_sided"], 3)}
                                 for m, v in prim.items() if m.endswith("@D")},
                  "FDC-BF@K/r": {m: {"ratio": round(v["geomean_ratio"], 3), "ub95": round(v["ub95_one_sided"], 3)}
                                 for m, v in k20.items() if m.endswith("@K")},
                  "false_streams_total_all_rigorous": a.get("all_rigorous_false_streams_total"),
                  "decision_if_applied": a["decision"]["verdict"], "failing": a["decision"]["failing_components"],
                  "replica": a["replica_status_combined"]["status"]}
    p = WS / "exp/results/pilots/v9_analysis_C_dev.json"
    if p.exists():
        a = json.loads(p.read_text())
        out["C"] = {t: {"wording_if_applied": v["wording"]["wording"], "per_rectangle_ub": v["wording"]["per_rectangle_ub"],
                        "replica": v["replica_status"]} for t, v in a["per_task"].items()}
    p = WS / "exp/results/pilots/v9_toy_dense/summary.json"
    if p.exists():
        out["toy_dense"] = {k: {"false_streams": v["false_streams"], "n": v["n_streams"]}
                            for k, v in json.loads(p.read_text())["table"].items()}
    return out


def v8_known_drift():
    v8 = prereg.load_lock(prereg_v8.ADDENDUM_PATH)
    drift = prereg_v6.v6_input_drift(v8, WS)
    if drift:
        raise SystemExit(f"unexplained v8 input drift: {drift}")
    return {}


def main():
    v8 = prereg_v8.load_locked_addendum(None)
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    frozen = RV.load_frozen()
    cr, x9 = frozen["CR"], frozen["X9"]
    add = {
        "version": "5+v9-addendum", "status": "draft", "written_at": "2026-10-04",
        "written_by": "author (round 9; design by the authors 2026-10-04)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v8["addendum_to"]["sha256"],
                        "git_commit": v8["addendum_to"].get("git_commit")},
        "v6_addendum": {"path": "plan/prereg_lock_v6_addendum.json", "sha256": v8["v6_addendum"]["sha256"],
                        "git_commit": v8["v6_addendum"].get("git_commit")},
        "v7_addendum": {"path": "plan/prereg_lock_v7_addendum.json", "sha256": v8["v7_addendum"]["sha256"],
                        "git_commit": v8["v7_addendum"].get("git_commit")},
        "v8_addendum": {"path": "plan/prereg_lock_v8_addendum.json", "sha256": v8["sha256"],
                        "git_commit": v8["git_commit"],
                        "block_A_verdict": "positive_result_achieved (X5: UB*_F 0.368 < 0.60; UB*_G 1.011 < 1.05 "
                                           "(PJC-local*); 0/200 false streams; replica pass)"},
        "v8_known_input_drift": v8_known_drift(),
        "statement": (
            "Append-only addendum. It edits neither lock v5 nor the v6 / v7 / v8 addenda and cannot change their "
            "verdicts. It registers TWO confirmatory blocks that test the FDC-BF procedure (code unchanged since v7) "
            "read under its time-uniform guarantee (TU-FDC, Theorem TU-1) against rivals of MATCHED guarantee "
            "strength (time-uniform: HC-WoR* and the time-uniform matched Bennett rectangle RECT-ck-BF-TU), every "
            "method monitored at the same 77 dense evaluation times: A on the Criteo CR9 eval half (fresh seeds "
            "37000-37199; 4th fresh-stream use of that table) and B on the X5 RetailHero eval half (fresh seeds "
            "37200-37399; 2nd use, after v8), B with non-inferiority components against PJC-local* (K = 77 ledger) and its time-uniform form TU-PJC. Block C is a "
            "pre-registered DESCRIPTIVE re-use of the outcome-exposed Hillstrom 3-arm log (design B, dev-frozen; "
            "design A as supplement). Disclosures: the time-uniform guarantee is a post-hoc theoretical observation "
            "applied to an unchanged procedure; HC-WoR's guarantee also covers adaptive (predictable) sampling, "
            "TU-FDC's needs the frozen outcome-free schedule; thresholds 0.85 (A) and 0.50 (B) were sized from dev "
            "dense-grid numbers (CR9 UB 0.737, X5 UB 0.365 vs HC-WoR*, seeds 950-999)."),
        "primary_method": {
            "name": "TU-FDC (analysis label TU-FDC@D)",
            "construction": "dsswm.baselines.fdc_loc.FDCTimeUniform(block_points = frozen K = 20 checkpoint grid, "
                            "localise=False): FDC-BF's code path (fdc_bet.direction_widths, Bennett-FPC, exact-HG "
                            "variance box, (q, pi, k) union over the K = 20 block points, (delta_main, delta_var) = "
                            "(0.045, 0.005)); at an evaluation time t the widths of the latest block point t_k <= t "
                            "are used with the current Delta_hat; on the block grid numerically identical to FDC-BF "
                            "(unit test test_tu_on_block_grid_equals_fdc_bf)",
            "fdc_bf_variant_tuple": list(VARIANTS["FDC-BF"]), "delta": 0.05,
            "fdc_bet_sha256": sha(CODE / "dsswm/baselines/fdc_bet.py"),
            "fdc_bet_v8_bound_sha256": v8["code_sha256"]["dsswm/baselines/fdc_bet.py"],
            "fdc_loc_sha256": sha(CODE / "dsswm/baselines/fdc_loc.py"),
            "guarantee": "Theorem TU-1 (plan/v9_candidates_theory.md, external reviewer-reviewed 2 rounds): FWER <= 0.05 "
                         "uniformly over all t >= t_1 under the frozen outcome-free schedule (NOT under adaptive "
                         "sampling)",
            "post_hoc": "the time-uniform reading is a post-hoc theoretical observation (v9 candidates, 2026-10-04) "
                        "applied to an unchanged procedure; no parameter was tuned; FDC-BF remains post hoc relative "
                        "to v5 / v6"},
        "rivals": {
            "primary_time_uniform": {
                "HC-WoR*": {"CR9": f"frozen v7/v8 CR9 config {cr['HC-WoR']}",
                            "X5": f"frozen v8 X5-tuned config {x9['HC-WoR']}",
                            "guarantee": "time-uniform under any predictable sampling rule (stronger class)"},
                "RECT-ck-BF-TU": "dsswm.baselines.rect_tu_v9.RectCkBFTU(box=True): the matched Bennett-FPC "
                                 "rectangle (pjc_bf.RectCkBF+box: same cell inequality, exact-HG variance box, delta "
                                 "split, frozen design) with radii frozen at the latest of the K = 20 block points "
                                 "(Lemma TU); parameter-free; identical to RECT-ck-BF+box on the block grid (unit "
                                 "test); dev toy dense FWER in exp/results/pilots/v9_toy_dense"},
            "non_inferiority_B": {
                "PJC-local*": f"frozen v8 X5-tuned config {x9['PJC-local']} (the no-reset member: one phase, "
                              "deterministic 50/50 count tracking, joint Bennett-FPC width), monitored at the same 77 "
                              "dense times with its own checkpoint ledger K = 77 (valid at those times)",
                "TU-PJC": "rect_tu_v9.make_tu_pjc(**frozen PJC-local* config of the layer; X5: PJC-BF[local,b=,half]): Lemma TU applied to the "
                          "no-reset count-only PJC-local*, widths frozen at the latest K = 20 block point; time-uniform "
                          "for t >= t_1 under the deterministic count-tracking schedule (external reviewer v9 r1 P1)"},
            "descriptive": ["dense grid: PJC-local* (K = 77 ledger; block A), FDC-BF[K77] (fresh widths, K' = 77 "
                            "ledger), RECT-ck-HG* (K = 77 ledger), TU-LOC (TU form of FDC-LOC)",
                            "K = 20 grid: FDC-BF, HC-WoR*, RECT-ck-HG*, RECT-ck-BF, RECT-ck-BF+box, PJC-local*, "
                            "FDC-MR[front3], FDC-HG (exact-HG envelope, NO-GO candidate), FDC-LOC (gap-localised "
                            "union, NO-GO candidate)"],
        },
        "blocks": {
            "A": {
                "role": "CONFIRMATORY (verdict 1 of 2)",
                "table": "Criteo CR9 eval half (pool_replay split seed 4242), K = 20 frozen grid -> dense grid of 77 "
                         "evaluation times (fdc_loc.dense_grid(ck, 4))",
                "table_reuse": "fresh seeds only; the CR9 eval half was used by v5 (30000-30199), v6 A (31000-31199), "
                               "v6 C (continuation of the v5 streams), v7 A (33000-33199) and v8 C (post hoc, v7 "
                               "streams): this is the 4th fresh-stream use of the table",
                "problems": "fr.cr_problems('visit'): 15 budgets, 512 policies", "eps": 0.001,
                "seeds": "37000-37199 (200 new streams)", "stop": "stop_k = 15 (N80 at 12/15 from the same trajectory)",
                "tasks": {t: {"grid": g, "methods": list(ms)} for t, (g, ms) in VA.CONF_SPEC["A"]["tasks"].items()},
                "primary": VA.PRIMARY, "rivals": list(VA.CONF_SPEC["A"]["rivals"]),
                "decision_rule": {
                    "positive_result_achieved": "UB95(TU-FDC@D / r, N80_pen) < 0.85 for r in {HC-WoR@D, "
                                                "RECT-ck-BF-TU@D} AND TU-FDC@D 0/200 false streams (whole run) AND "
                                                "replica pass for v9a_full_d / _k / _x incl. cross-task R2b and "
                                                "cross-task frozen-design R2 (IUT)",
                    "positive_result_not_achieved": "otherwise; components rival_superiority_failed (sub-label "
                                                    "faster_below_1 / not_faster), validity_failure, replica_fail",
                    "threshold_origin": "dev dense grid (cr9d, seeds 950-999): TU-FDC/HC-WoR* 0.708, UB 0.737; "
                                        "0.85 leaves ~0.11 for a different stream set (authors design)"}},
            "B": {
                "role": "CONFIRMATORY (verdict 2 of 2)",
                "table": "X5 RetailHero X9 eval half (blinded hashed split, 100,393 rows), dense grid of 77 times",
                "table_reuse": "used once before (v8 block A, seeds 35000-35199); fresh seeds",
                "eval_access": "dsswm.envs.x5_v9.X5V9Env('eval', task): v9b_full* tasks passing prereg_v9.addendum_gate; "
                               "files checked against PROVENANCE.json",
                "problems": "fr.cr_problems('visit')", "eps": 0.02, "seeds": "37200-37399 (200 new streams)",
                "stop": "stop_k = 15",
                "tasks": {t: {"grid": g, "methods": list(ms)} for t, (g, ms) in VA.CONF_SPEC["B"]["tasks"].items()},
                "primary": VA.PRIMARY, "rivals": list(VA.CONF_SPEC["B"]["rivals"]),
                "non_inferiority": list(VA.CONF_SPEC["B"]["ni"]),
                "decision_rule": {
                    "positive_result_achieved": "UB95(TU-FDC@D / r) < 0.50 for r in {HC-WoR@D, RECT-ck-BF-TU@D} AND "
                                                "UB95(TU-FDC@D / g) < 1.05 for g in {PJC-local@D, TU-PJC@D} AND "
                                                "TU-FDC@D 0/200 false streams "
                                                "AND replica pass for v9b_full_d / _k / _x (IUT)",
                    "positive_result_not_achieved": "otherwise; components rival_superiority_failed, "
                                                    "joint_rival_not_inferior_failed, validity_failure, replica_fail",
                    "wording": "a pass supports 'faster than the two registered time-uniform rectangles' and 'not "
                               "slower by more than 5% than the frozen PJC-local configuration (K = 77 ledger) and its "
                               "time-uniform form TU-PJC'; never 'faster than PJC'",
                    "threshold_origin": "dev dense grid (x5d, 950-999): TU-FDC/HC-WoR* 0.356, UB 0.365; 1.05 "
                                        "inherited from v8 G"}},
            "joint": {"rule": "the time-uniform speed headline ('at matched time-uniform guarantee strength, lower "
                              "paired geometric-mean N80_pen than both registered time-uniform rectangle rivals "
                              "(HC-WoR*, RECT-ck-BF-TU), on fresh streams of each reused finite table') requires A AND "
                              "B positive (IUT across tables); each block's verdict is also reported on its own",
                      "scope": "confirmation is conditional on the two reused finite tables (fresh streams of the same "
                               "populations), not an independent replication across populations, and covers only "
                               "the two registered rectangle constructions, not every possible time-uniform rectangle "
                               "(external reviewer v9 r1 P0)"},
            "C": {
                "role": "DESCRIPTIVE ONLY: pre-registered re-use of an outcome-exposed log; no verdict, not in any "
                        "IUT / Holm family, not merged with A / B, never called a holdout or confirmatory",
                "exposure": "plan/hillstrom_exposure_audit.md: r4 computed full-table cell means of visit, conversion "
                            "and spend (HR6 / HR16) and eval-half truth tables (HR8); every row is outcome-exposed",
                "data": "hillstrom/tidy.pkl sha256 3dab9ed72bfdcaf51f6d993975a7ca052ef9355690489ca854217b7f0a68b296; "
                        "hashed split salt 'ds-swm/v9/hillstrom/2026-10-04' (dev 32,119 / eval 31,881 rows)",
                "main": {"task": "v9c_full_b", "design": "B: CZ6 = cat3 x urban (S = 6, 729 policies), visit, "
                                                          "kappa (0, 1, 0.5) = womens e-mail HALF PRICE, budgets "
                                                          "0.10-0.80, 12/15", "eps": [0.01, 0.0125, 0.015],
                         "primary_eps": 0.0125, "seeds": "37400-37599",
                         "methods": list(VA.C_B_METHODS)},
                "supplement": {"task": "v9c_full_a", "design": "A: CR6 / F3 (0, 1.5, 1) / visit (rivals piled up at "
                                                                "tau_R on dev)", "eps": [0.006, 0.0075, 0.01],
                               "primary_eps": 0.0075, "seeds": "37600-37799", "methods": list(VA.C_A_METHODS)},
                "wording_rule": "wording only, no test: UB95(FDC-BF/r) < 0.95 for every tuned rectangle (RECT-ck-HG*, "
                                "HC-WoR*) at the primary eps AND FDC-BF 0 false streams -> 'on the 3-arm Hillstrom log, "
                                "FDC-BF is also faster than the tuned rectangles (descriptive)'; else 'no speed "
                                "advantage on Hillstrom' plus the running-certificate-bound ratio at checkpoint K-2 (conditional on "
                                "both trajectories reaching it with positive finite bounds; eligible / total reported; "
                                "'unavailable' below 10 eligible pairs); PJC-local*: ratio and CI only",
                "disclosures": ["the design depends on an UNSOURCED womens e-mail half price (F2); under F1 / F3 no "
                                "design met rule B on the dev half, so F2 was the only qualifying cost family",
                                "design-rule revision history v1 -> v2 (design A) -> B is disclosed",
                                "tau_R = whole eval half read; checkpoints coarse near tau_R, ratios are quantised",
                                "dev reference (950-999): FDC-BF/RECT-ck-HG* UB 0.858, /HC-WoR* UB 0.848, /PJC 1.011"],
                "configs": "exp/results/v9_gates/hv9_configs.json (written by run_r5s_v9.py --freeze-hv9 from the "
                           "dev selection files; Neyman plan of FDC-BF[ney] from dev-half cell variances)"},
        },
        "frozen_configs": frozen,
        "seed_manifest": {
            "dev_runner_check": "CR9 / X5 / Hillstrom dev halves, seeds 950-999 (dev only; used before by the FDC-HG, "
                                "v9-candidates and Hillstrom dev reports)",
            "A_eval": "37000-37199", "B_eval": "37200-37399", "C_eval": "37400-37599", "C_supplement": "37600-37799",
            "bootstrap_seed": 42,
            "never_used_check": "2026-10-04: grep of every json/jsonl under the workspace for seed / perm_seed 37xxx: "
                                "0 hits; no code references 37000-37799 except the v9 runner / analysis / tests",
            "previously_used_eval": "30000-30199 (v5, v6 C), 31000-31199 (v6 A), 32000-32199 (v6 B), 33000-33199 "
                                    "(v7 A, v8 C), 34000-34199 (v7 B), 35000-35199 (v8 A), 35200-35399 (v8 B), "
                                    "36000-36199 (v8 post-hoc F)"},
        "replica_check": {
            "implementation": "dsswm/stats/v9_replica.py + run_r5s_v9.py --task <t> --replica",
            "R1": f"independent process re-runs the first {R1_N_SEEDS} seeds of each eval task; canonical rows "
                  "identical", "R1_excluded_fields": list(TIMING_FIELDS),
            "R2": "schedule_digest identical within every (seed, eps) across each registered frozen-design group",
            "R2_groups": {t: [list(g) for g in VR.groups_for(t, frozen)] for t in RV.TASKS if "full" in t},
            "R2b": "arrival_digest identical across ALL methods of a task within every (seed, eps)",
            "cross_task": "blocks A / B: R2b across the d / k / x tasks (same streams on both grids) and R2 across "
                          "every frozen-design member of the three tasks; R2c (v9_replica.check_cross_grid): each method's own "
                          "schedule digest and realised block counts identical across the dense and K = 20 grids "
                          "(covers HC-WoR's Neyman schedule and PJC count paths); all three enter the replica status",
            "on_failure": {"A": "replica_fail -> positive_result_not_achieved", "B": "same", "C": "flag replica_fail"}},
        "eval_tasks": {t: {"block": RV.TASKS[t]["block"], "layer": RV.TASKS[t]["layer"], "grid": RV.TASKS[t]["grid"],
                           "seeds": f"{RV.TASKS[t]['seeds'][0]}-{RV.TASKS[t]['seeds'][-1]}",
                           "eps": list(RV.TASKS[t]["eps"]), "stop_k": RV.TASKS[t]["stop_k"],
                           "methods": list(RV.TASKS[t]["methods"])}
                       for t in RV.TASKS if RV.TASKS[t]["half"] == "eval"},
        "command": "cd exp/code && python run_r5s_v9.py --task <task> [--workers 4]; --task <task> --replica; "
                   "--analyse A|B|C. Every step re-checks the locked v9 addendum (and through it v8, v7, v6, v5).",
        "eval_seal": "as v8 (v9_seal: seal committed alone at task completion + sidecar ref commit; single-commit "
                     "history anchor)",
        "threat_model": "as v6-v8: accidental drift, partial / resumed runs and post-hoc edits are detected against git "
                        "history; deliberate history rewriting is out of scope. X5 blinding of v8 is procedural; the "
                        "X5 eval half has since been read once by the v8 block-A runs (35000-35199), and its v8 "
                        "results are public inside this project.",
        "declarations": [
            "Primary procedure = FDC-BF code unchanged since v7; the time-uniform guarantee (Theorem TU-1) is a "
            "post-hoc theoretical observation applied to that unchanged procedure; on the K = 20 grid TU-FDC = FDC-BF.",
            "Residual asymmetry: HC-WoR is time-uniform under any predictable sampling rule; TU-FDC and RECT-ck-BF-TU "
            "only under the frozen outcome-free schedule. Methods use outcome-free, METHOD-SPECIFIC allocation "
            "schedules with shared arrivals and pool orders (X5 HC-WoR* uses its frozen Neyman plan; PJC uses "
            "deterministic count tracking); R2 checks the registered 50/50 group, R2c checks each method's own "
            "schedule digest and realised block counts across the dense and K = 20 grids.",
            "The v9 primary design is the former secondary dense-grid experiment: the confirmatory procedure is "
            "TU-FDC evaluated at 77 monitoring times through a new wrapper, which can change stopping times and "
            "sticky answers relative to K = 20 FDC-BF; the inherited width parameters are unchanged and the two are "
            "numerically identical when evaluated only on the K = 20 grid.",
            "PJC asymmetry: PJC-local@D carries its own K = 77 checkpoint ledger (an extra ln(77/20) ~ 1.348 in beta and "
            "smaller variance-box tail levels) but uses fresh widths, so its net cost is not predetermined; the "
            "same-grid time-uniform form TU-PJC (Lemma TU applied to the no-reset count-only PJC-local*, widths "
            "frozen at the K = 20 block points) is therefore registered as a second non-inferiority rival in B. NI "
            "wording is limited to these two registered PJC formulations; the matched K = 20 comparison is reported "
            "as before.",
            "Inherited validation scope: v9 introduces no additional input drift; it inherits v8's documented "
            "exceptions (v8 accepts the recorded v7-input erratum in exp/results/full/v6_summary.md, and the gate "
            "cascade does not revalidate v5's input manifest).",
            "RECT-ck-BF-TU is a new rival constructed for v9 (time-uniform upgrade of the v8 matched rectangle); it "
            "has no tuning parameter.",
            "Thresholds 0.85 (A) and 0.50 (B) were set by the authors from dev dense-grid numbers (CR9 UB 0.737, "
            "X5 UB 0.365 vs HC-WoR*, seeds 950-999) before the v9 dev runner check; the runner check (first "
            "measurement of TU-FDC vs RECT-ck-BF-TU and PJC-local@D) did not change them.",
            "The CR9 eval half is used for the 4th time with fresh streams; the X5 eval half for the 2nd time.",
            "FDC-HG and FDC-LOC were NO-GO on dev and enter only as descriptive rows.",
            "Block C (Hillstrom) is descriptive only: outcome-exposed log; design depends on an unsourced womens "
            "half price, the only qualifying cost family; design A in the supplement.",
            "Timings: up to 3 runner processes x 4 workers concurrently with other jobs on the host; biased up."],
        "code_sha256": {p: sha(CODE / p) for p in RV.CODE_FILES + RV.LOCK_ONLY_FILES},
        "input_sha256": {p: sha(WS / p) for p in INPUTS},
        "data_sha256": data_hashes_v9(),
        "dev_runner_check": dev_numbers(),
        "lock_procedure": "external reviewer review (<= 2 rounds) -> code-freeze commit -> prereg_v9.finalize_addendum(git_commit) "
                          "-> commit the lock file alone -> eval tasks, replicas, analyses.",
        "git_commit": None,
    }
    add["sha256_draft"] = prereg.canonical_hash({k: v for k, v in add.items() if k != "sha256_draft"})
    from dsswm.stats.prereg_v9 import DRAFT_PATH
    DRAFT_PATH.write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(f"wrote {DRAFT_PATH} sha256_draft={add['sha256_draft']}")


if __name__ == "__main__":
    main()
