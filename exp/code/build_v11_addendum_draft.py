"""Build plan/prereg_lock_v11_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage (cwd exp/code):  .venv/bin/python3 build_v11_addendum_draft.py
Locking (main session): external reviewer lock review -> rebuild this draft -> commit code + inputs + draft (code freeze) ->
prereg_v11.finalize_addendum(<freeze commit>) -> commit the lock file alone -> eval task, replica, analysis.
No eval file is opened by this script: both eval-file hashes are taken from PROVENANCE.json (external reviewer v11 lock r1 F1/F3).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.envs.obd_v11_eval import LAYER, data_hashes_v11, load_frozen  # noqa: E402
from dsswm.stats import prereg, prereg_v6, prereg_v10  # noqa: E402
from dsswm.stats import v11_analysis as VA  # noqa: E402
from dsswm.stats import v11_replica as VR  # noqa: E402
from dsswm.stats.v6_replica import R1_N_SEEDS, TIMING_FIELDS  # noqa: E402

import run_v11 as RV  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT
INPUTS = ["plan/prereg_lock.json", "plan/prereg_lock_v6_addendum.json", "plan/prereg_lock_v7_addendum.json",
          "plan/prereg_lock_v8_addendum.json", "plan/prereg_lock_v9_addendum.json",
          "plan/prereg_lock_v10_addendum.json", "plan/prereg_lock_v11_addendum_DRAFT.md", "plan/v11_obd_plan.md",
          "plan/obd_feasibility.md", "plan/v11_dev_report.md", "plan/fdc_dp_theory.md",
          "exp/results/v11_gates/obd_v11_frozen.json", "exp/results/v11_gates/v11_eps_hc.json",
          "exp/results/v11_gates/v11_thresh.json", "exp/results/v11_gates/MANIFEST.json",
          "exp/results/pilots/v11_dev/v11_analysis_dev.json",
          "exp/results/pilots/v11_dev/v11_obd_pilot/results.jsonl",
          "exp/results/pilots/v11_dev/v11_obd_pilot/replica_report.json",
          "exp/results/pilots/obd_dev/describe.json", "exp/results/pilots/obd_dev/describe_holdout.json"] + \
         [f"exp/results/pilots/v11_dev/{RV._rule_task(i)}/results.jsonl" for i in range(len(VA.EPS_GRID))]
OPTIONAL_INPUTS = ["reviews/v11_lock_review_r1.md", "reviews/v11_lock_review_r2.md", "reviews/v11_lock_review_r3.md"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def v10_known_drift():
    v10 = prereg.load_lock(prereg_v10.ADDENDUM_PATH)
    drift = prereg_v6.v6_input_drift(v10, WS)
    if drift:
        raise SystemExit(f"unexplained v10 input drift: {drift}")
    return {}


def dev_numbers():
    a = json.loads((WS / "exp/results/pilots/v11_dev/v11_analysis_dev.json").read_text())
    c = a["cell"]
    comp = {k: {x: v[x] for x in ("geomean_ratio", "ub95_one_sided", "ci95_two_sided", "frac_faster", "frac_tied",
                                  "frac_slower")} for k, v in c["comparisons"].items()}
    return {"seeds": "dev 950-999 on the whole dev half (final code, frozen design, frozen configs)",
            "eps": c["eps"], "comparisons": comp,
            "methods": {m: {k: v[k] for k in ("false_streams", "geomean_N80_over_tau", "share_N80_lt_tau",
                                              "share_reached_15", "exhaustion_at_k80_mean")}
                        for m, v in c["methods"].items()},
            "eps_units": a["eps_units"], "eps_optimal_share": a["eps_optimal_share"]["mean_share_eps_optimal"],
            "replica": a["replica_status"], "thresh_if_frozen_now": a["thresh_if_frozen_now"]}


def main():
    v10 = prereg_v10.load_locked_addendum(None)
    opt = [p for p in OPTIONAL_INPUTS if (WS / p).exists()]
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    cfg, gates = RV.load_frozen(None, with_thresh=True)
    fz = load_frozen()
    sel = json.loads(RV.EPS_HC_JSON.read_text())
    th = json.loads(RV.THRESH_JSON.read_text())
    man = json.loads(RV.MANIFEST_JSON.read_text())["sha256"]
    for f in RV.GATE_FILES:
        if man.get(f) != sha(RV.GATES / f):
            raise SystemExit(f"MANIFEST.json does not match {f}")
    dev = dev_numbers()
    if abs(VA.thresh_rule(dev["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["ub95_one_sided"]) - th["thresh"]) > 1e-12:
        raise SystemExit("THRESH gate does not follow the rule applied to the dev analysis")
    eps, thresh = cfg["eps"], cfg["thresh"]
    add = {
        "version": "5+v11-addendum", "status": "draft", "written_at": "2026-10-05",
        "written_by": "experimenter (v11 OBD retarget; design plan/v11_obd_plan.md, committed before any v11 "
                      "dev stream)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v10["addendum_to"]["sha256"],
                        "git_commit": v10["addendum_to"].get("git_commit")},
        **{k: {"path": f"plan/prereg_lock_{k[:-9]}_addendum.json", "sha256": v10[k]["sha256"],
               "git_commit": v10[k].get("git_commit")} for k in ("v6_addendum", "v7_addendum", "v8_addendum",
                                                                  "v9_addendum")},
        "v10_addendum": {"path": "plan/prereg_lock_v10_addendum.json", "sha256": v10["sha256"],
                         "git_commit": v10["git_commit"]},
        "v10_known_input_drift": v10_known_drift(),
        "statement": (
            "Append-only addendum. It edits neither lock v5 nor the v6-v10 addenda and cannot change their verdicts. "
            "It registers ONE confirmatory block: TU-FDC-DP(b) -- the FDC-BF joint certificate computed exactly (up "
            "to an abstaining node cap) over the segment-policy class by multiple-choice knapsack DP and "
            "branch-and-bound, read under its time-uniform guarantee and monitored at the K = 40 block grid -- "
            "against the matched, parameter-free time-uniform Bennett rectangle RECT-BF-DP-TU, on the eval half of a "
            "real uniform-random recommendation log that no earlier lock has replayed (Open Bandit Dataset, "
            "random/all; first use; fresh seeds 39000-39199), under a design frozen from the dev half (9 user-"
            "feature segments, 2 item-group arms, exposure-cost knapsack, 15 budgets). eps was chosen on dev by the "
            "pre-stated rival-success rule; THRESH by the pre-stated dev rule. Inference is conditional on the "
            "finite eval table (fresh replay randomness only)."),
        "primary_method": {
            "name": VA.PRIMARY,
            "construction": "dsswm.baselines.fdc_dp.FDCDPTimeUniform(block_points = the K = 40 checkpoint grid, "
                            "scheme='b', node_limit 200000, grid ratio 2000^(1/160)); identical construction to "
                            "lock v10 except K",
            "guarantee": "Theorem DP-1 (plan/fdc_dp_theory.md) + Lemma TU: FWER <= 0.05 per stream and problem "
                         "family, for all t >= t_1, under the frozen outcome-free 50/50 schedule",
            "fdc_dp_sha256": sha(CODE / "dsswm/baselines/fdc_dp.py"),
            "rect_dp_sha256": sha(CODE / "dsswm/baselines/rect_dp.py")},
        "rivals": {
            "governing": {VA.RIVAL: "dsswm.baselines.rect_dp.RectBFDPTU(block_points = K = 40 grid, box=True): "
                                    "Bennett-FPC rectangle matched to FDC-DP, exact worst-gap DP, parameter-free"},
            "descriptive": {
                VA.HC_TUNED: f"hedged-capital WoR CS rectangle (time-uniform), tuned on dev by the block-G3 rule at "
                             f"the selected eps: {cfg[VA.HC_TUNED]}",
                VA.RECT_HG: "exact hypergeometric per-cell rectangle at delta / (2 S A K) with the worst-gap DP; "
                            "CHECKPOINT-STRENGTH (CP) guarantee (valid at the K grid points only); labelled CP"},
            "claim_scope": "claims are restricted to the named implementations; not 'the best valid rectangle'"},
        "block": {
            "role": "CONFIRMATORY (single verdict)",
            "table": "Open Bandit Dataset random/all eval half (688,159 rows; blinded salted row-level split of "
                     "PROVENANCE.json), the whole half is the replay population (tau = eval N)",
            "table_use": "first use of this eval half; no eval-outcome statistic computed before the lock",
            "layer": LAYER, "task": RV.EVAL_TASK, "seeds": "39000-39199 (200 new streams)", "eps": eps,
            "methods": list(VA.METHODS)},
        "design": {
            "segments": fz["segment_rule"] + "; frozen explicit map in exp/results/v11_gates/obd_v11_frozen.json; "
                        + fz["fallback_rule"] + " (fallback row counts reported)",
            "arms": fz["arm_rule"] + "; arm a = show a uniformly random item of group a; frozen item lists",
            "costs": "kappa[s, 0] = 0, kappa[s, 1] = max(1, round(8 S w_s^dev)); budgets floor(b_q sum kappa[s, 1]), "
                     "b_q = 0.10 .. 0.80 (15 problems); frozen from dev (identical classes on dev and eval); policy "
                     "values use the eval segment weights",
            "cost_vector": [c[1] for c in fz["cost"]], "budgets": fz["budgets"],
            "allocation": "frozen outcome-free 50/50 design balanced_alloc(S, A, 0.5) for every method",
            "checkpoints": "fr.checkpoints(5000, tau = eval N, K = 40); K changed from 20 (v7-v10, feasibility) to "
                           "40 so that ratios are not quantised to ~0.77 / 0.60 / 0.46 grid steps",
            "delta": 0.05, "horizon": "every stream runs to 15/15 certified or tau",
            "endpoint": "N80_pen = the K = 40 checkpoint at which 12/15 problems are first certified if no false "
                        "certificate by then, else tau; a false stream = any false certificate over the whole run"},
        "eps": eps,
        "eps_selection": {
            "rule": sel["rule"], "hc_rule": sel["hc_rule"], "grid": sel["eps_grid"],
            "result": {k: sel["eps_rule"][k] for k in ("selected_eps", "rival", "rival_method", "successes", "n")},
            "trace": [{k: t[k] for k in ("eps", "best", "best_method", "qualifies")} |
                      {"per_rect": {m: {x: v[x] for x in ("method", "successes", "geomean_N80_pen")}
                                    for m, v in t["per_rect"].items()}} for t in sel["eps_rule"]["trace"]],
            "implementation": "dsswm.stats.v11_analysis.select_hc / select_eps (select_eps refuses FDC-DP rows), via "
                              "run_v11.py --select on the rule-task rows only",
            "units": {"absolute_ctr": eps, "relative_to_dev_base_ctr": dev["eps_units"]["relative_to_dev_base"],
                      "dev_base_ctr": dev["eps_units"]["dev_base_ctr"],
                      "dev_mean_share_of_feasible_policies_eps_optimal": dev["eps_optimal_share"],
                      "eval_reporting": "eval base CTR, eps / eval base and the eval eps-optimal share are reported "
                                        "descriptively after the eval run"}},
        "thresh": thresh,
        "thresh_rule": {"rule": th["rule"], "ub95_dev": th["ub95_dev"], "ratio_dev": th["ratio_dev"],
                        "value": thresh, "source": th["source"], "rule_written": "plan/v11_obd_plan.md s6 (before any "
                                                                                 "v11 dev stream)"},
        "decision_rule": {
            "statistic": "paired geometric-mean ratio of N80_pen, TU-FDC-DP(b) / RECT-BF-DP-TU, paired percentile "
                         "bootstrap over the 200 streams (B = 10^4, seed 42, one resample matrix for all "
                         "comparators), one-sided UB95; two-sided CI reported",
            "positive_result_achieved": f"UB95 < THRESH = {thresh} AND TU-FDC-DP(b) 0/200 false streams (whole run "
                                        "to 15/15 or tau) AND replica pass",
            "positive_result_not_achieved": "otherwise; components rival_superiority_failed (sub-label "
                                            "faster_below_1 / not_faster), validity_failure, replica_fail",
            "implementation": "dsswm.stats.v11_analysis.decide via run_v11.py --analyse",
            "descriptive": "ratios vs HC-WoR-DP[tuned] and RECT-HG-DP (CP), RECT-BF-DP-TU vs each; per-method false "
                           "streams with Clopper-Pearson UB; exhaustion at N80; share reaching 15/15; eps-optimal "
                           "share on the eval truth; fallback row counts"},
        "frozen_configs": json.loads(json.dumps(cfg)),
        "frozen_gates": gates,
        "seed_manifest": {
            "dev": "950-999 on the whole dev half (rule tasks, dev cell; the feasibility pilot used 950-979 at K = 20)",
            "eval": "39000-39199", "bootstrap_seed": 42,
            "previously_used_eval": "30000-30199, 31000-31199, 32000-32199, 33000-33199, 34000-34199, 35000-35399, "
                                    "36000-36199, 37000-37799, 38000-38999 (all on other tables)"},
        "replica_check": {
            "implementation": "dsswm/stats/v11_replica.py + run_v11.py --task v11_obd_full --replica",
            "R1": f"independent process re-runs the first {R1_N_SEEDS} eval seeds; canonical rows identical",
            "R1_excluded_fields": list(TIMING_FIELDS),
            "R2": "schedule_digest identical across all four methods per seed (frozen 50/50 design)",
            "R2_groups": [list(g) for g in VR.groups_for(RV.EVAL_TASK)],
            "R2b": "arrival_digest identical across all methods per seed",
            "on_failure": "replica_fail -> positive_result_not_achieved"},
        "eval_tasks": {RV.EVAL_TASK: {"role": "confirmatory", "layer": RV.TASKS[RV.EVAL_TASK]["layer"],
                                      "eps": eps, "seeds": "39000-39199", "methods": list(VA.METHODS), "K": VA.K_GRID,
                                      "stop_k": 15, "n80_k": VA.N80_K}},
        "eval_access": {
            "reader": "dsswm.envs.obd_v11_eval.read_eval_rows: prereg_v11 gate (locked addendum, all hashes, "
                      "registered task) -> task registered for exactly layer " + LAYER + " -> caller's lock sha256 "
                      "equals the locked addendum's -> one line appended to exp/results/full/v11_obd/"
                      "eval_access_log.jsonl -> data hashes vs lock and PROVENANCE -> read",
            "pre_lock_reads": (
                "Split preparation (2026-10-05, before v11): the split script read all.csv, wrote the eval files, "
                "hashed them and computed eval LABEL-only statistics recorded in PROVENANCE.json (position / item "
                "shares; no outcome statistic). v11 code before the lock: no eval OUTCOME byte was opened or "
                "hashed. eval_labels.pkl bytes were hashed (never deserialised) by the first v11 dev runs and the "
                "first draft build (commits 8ed70164 / 393dbb15, before external reviewer lock review r1); after r1 (F1/F3) "
                "both eval hashes are taken from the hash-bound PROVENANCE.json, the gate never opens an eval "
                "file, and eval bytes are hashed only inside the authorised, logged reader."),
            "gate_order": "caller layer + sha format -> v11 gate (status, schema, task membership, inherited chain, "
                          "canonical hash, code / input / non-eval data drift, single-commit history) -> registered "
                          "layer -> caller sha equals lock -> access-log line (fsync) -> eval bytes vs lock and "
                          "PROVENANCE -> deserialise"},
        "command": "cd exp/code && python run_v11.py --task v11_obd_full [--workers 4]; --task v11_obd_full "
                   "--replica; --analyse. Every step re-checks the locked v11 addendum (and through it v10-v5).",
        "eval_seal": "v11_seal (as v8-v10: seal committed alone at task completion + sidecar ref commit; "
                     "single-commit history anchor); exp/results/full/v11_seals",
        "threat_model": "as v6-v10: accidental drift, partial / resumed runs and post-hoc edits are detected against "
                        "git history; deliberate history rewriting is out of scope.",
        "declarations": [
            "FDC-DP's statistical certificate is FDC-BF's (Theorem DP-1); the contribution is computational; the "
            "block tests speed against named implementations.",
            "Headline claim restricted to methods with proven finite-sample / anytime-valid guarantees (user ruling "
            "2026-10-03); plug-in methods are not run.",
            "The arm grouping (top-40 vs bottom-40 items by dev CTR) and the uf0 x uf3 segmentation (picked for "
            "visible heterogeneity) used dev outcomes; legitimate because eval is untouched; the eval uplift gap "
            "shrinks by selection. The feasibility holdout (ratio 0.78 vs 0.72 at eps 5e-4, K = 20) froze only the "
            "ITEM GROUPS on the even-row half; segment lists and costs were recomputed on the odd replay half "
            "(external reviewer v11 lock r1 F4), so it is an item-group holdout, not a rehearsal of the full frozen-design "
            "transfer.",
            "Dev block regenerated once: the first v11 dev run (commit 8ed70164) was bound to pre-r1 code; after "
            "the r1 gate fix (no change to any stream computation) the rule tasks, eps / HC selection, dev cell, "
            "dev replica, dev analysis and THRESH were regenerated on the fixed code; the gate files were rewritten "
            "once for that reason only (values compared in plan/v11_dev_report.md).",
            "Row-level split: 3.3 % of dev rows share an exact timestamp with an eval row (multi-position "
            "impressions); user ids are absent, so repeat users can occur in both halves; dev and eval are weakly "
            "dependent.",
            "First use of the OBD eval half; inference conditional on the finite table.",
            "K changed from 20 to 40 (pre-stated in plan/v11_obd_plan.md); costs frozen from dev segment shares "
            "rather than eval label shares (feasibility suggestion), so no eval read was needed before the lock.",
            "HC-WoR-DP is tuned on the same dev streams used by the eps rule (selection-optimistic for HC; HC is "
            "descriptive). RECT-HG-DP has a checkpoint-strength (CP) guarantee only (descriptive).",
            "THRESH = min(0.90, UB95_dev + 0.10) is a dev-calibrated threshold (rule pre-stated before any v11 dev "
            "stream; value computed after the dev run, before the lock).",
            "The multi-arm (A = 3) cell is not registered (control-pool exhaustion on dev; feasibility s4).",
            "random/men, random/women and bts/* stay unread (reserved).",
            "Timings: up to 5 runner processes x 4 workers concurrently; biased up."],
        "code_sha256": {p: sha(CODE / p) for p in RV.CODE_FILES + RV.LOCK_ONLY_FILES},
        "input_sha256": {p: sha(WS / p) for p in INPUTS + opt},
        "data_sha256": data_hashes_v11(),
        "dev_runner_check": dev,
        "lock_procedure": "external reviewer lock review -> rebuild draft -> code-freeze commit -> "
                          "prereg_v11.finalize_addendum(git_commit) -> commit the lock file alone -> eval task, "
                          "replica, analysis.",
        "git_commit": None,
    }
    from dsswm.stats.prereg_v11 import DRAFT_PATH, _schema
    _schema(add, require_commit=False)
    add["sha256_draft"] = prereg.canonical_hash({k: v for k, v in add.items() if k != "sha256_draft"})
    DRAFT_PATH.write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(f"wrote {DRAFT_PATH} sha256_draft={add['sha256_draft']}")


if __name__ == "__main__":
    main()
