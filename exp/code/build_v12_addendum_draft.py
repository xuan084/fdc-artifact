"""Build plan/prereg_lock_v12_addendum_DRAFT.json from the files on disk (hashes computed, never typed).

Usage (cwd exp/code):  .venv/bin/python3 build_v12_addendum_draft.py
Locking (main session): external reviewer lock review -> rebuild this draft -> commit code + inputs + draft (code freeze) ->
prereg_v12.finalize_addendum(<freeze commit>) -> commit the lock file alone -> per campaign: eval task, replica,
analysis.  No eval file is opened by this script: the four eval-file hashes are taken from the campaign
PROVENANCE.json files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsswm.envs.obd_v12_eval import CAMPAIGNS, LAYERS, N_ITEMS, data_hashes_v12, load_frozen  # noqa: E402
from dsswm.stats import prereg, prereg_v6, prereg_v11  # noqa: E402
from dsswm.stats import v11_replica as VR  # noqa: E402
from dsswm.stats import v12_analysis as VA  # noqa: E402
from dsswm.stats.v6_replica import R1_N_SEEDS, TIMING_FIELDS  # noqa: E402

import run_v12 as RV  # noqa: E402

WS, CODE = prereg.WS_ROOT, prereg.CODE_ROOT
BLOCK_LABEL = {"women": "A (primary)", "men": "B"}
INPUTS = ["plan/prereg_lock.json", "plan/prereg_lock_v6_addendum.json", "plan/prereg_lock_v7_addendum.json",
          "plan/prereg_lock_v8_addendum.json", "plan/prereg_lock_v9_addendum.json",
          "plan/prereg_lock_v10_addendum.json", "plan/prereg_lock_v11_addendum.json",
          "plan/prereg_lock_v12_addendum_DRAFT.md", "plan/v12_obd2_plan.md", "plan/v11_obd_plan.md",
          "plan/v12_dev_report.md", "plan/fdc_dp_theory.md", "exp/results/v12_gates/MANIFEST.json"] + \
         [f"exp/results/v12_gates/{f}" for f in RV.GATE_FILES] + \
         [x for c in CAMPAIGNS for x in (
             f"exp/results/pilots/v12_dev/v12_{c}_analysis_dev.json",
             f"exp/results/pilots/v12_dev/{RV.pilot_task(c)}/results.jsonl",
             f"exp/results/pilots/v12_dev/{RV.pilot_task(c)}/replica_report.json",
             f"exp/results/pilots/v12_dev/{RV.pilot_task(c)}/replica/results.jsonl")] + \
         [f"exp/results/pilots/v12_dev/{RV._rule_task(c, i)}/results.jsonl" for c in CAMPAIGNS
          for i in range(len(VA.EPS_GRID))]
OPTIONAL_INPUTS = ["reviews/v12_lock_review_r1.md", "reviews/v12_lock_review_r2.md"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def v11_known_drift():
    v11 = prereg.load_lock(prereg_v11.ADDENDUM_PATH)
    drift = prereg_v6.v6_input_drift(v11, WS)
    if drift:
        raise SystemExit(f"unexplained v11 input drift: {drift}")
    return {}


def dev_numbers(c):
    a = json.loads((WS / f"exp/results/pilots/v12_dev/v12_{c}_analysis_dev.json").read_text())
    cell = a["cell"]
    comp = {k: {x: v[x] for x in ("geomean_ratio", "ub95_one_sided", "ci95_two_sided", "frac_faster", "frac_tied",
                                  "frac_slower")} for k, v in cell["comparisons"].items()}
    dsc = cell["descriptive_v12"]
    return {"seeds": "dev 950-999 on the whole campaign dev half (final code, frozen design, frozen configs)",
            "eps": cell["eps"], "comparisons": comp,
            "methods": {m: {k: v[k] for k in ("validity", "false_streams", "geomean_N80_over_tau", "share_N80_lt_tau",
                                              "share_reached_15", "exhaustion_at_k80_mean")}
                        for m, v in cell["methods"].items()},
            "full_frontier": {k: dsc["full_frontier"][k] for k in ("geomean_ratio", "ub95_one_sided",
                                                                    "rival_reaches_15_only_at_tau")},
            "uncensored_subset": {k: dsc["uncensored_subset"].get(k) for k in ("geomean_ratio", "ub95_one_sided",
                                                                               "n_streams_in_subset")},
            "eps_units": a["eps_units"], "eps_optimal_share": a["eps_optimal_share"]["mean_share_eps_optimal"],
            "replica": a["replica_status"], "thresh_if_frozen_now": a["thresh_if_frozen_now"]}


def _pilot_started(c):
    return json.loads((WS / f"exp/results/pilots/v12_dev/{RV.pilot_task(c)}/summary.json").read_text())["started_at"]


def decision_rule(c, cfg, status):
    stat = ("paired geometric-mean ratio of N80_pen, TU-FDC-DP(b) / RECT-BF-DP-TU, paired percentile bootstrap over "
            "the 200 streams (B = 10^4, seed 42, one resample matrix for all comparators of this block), one-sided "
            "UB95; two-sided CI reported")
    impl = ("dsswm.stats.v12_analysis.decide_block via run_v12.py --campaign " + c + " --analyse (confirmatory: "
            "v11_analysis.decide unchanged; descriptive: v11_analysis.decide is called only to validate the inputs "
            "(UB, false-stream count, replica status, THRESH range); its internal THRESH comparison and verdict are "
            "discarded and have no reporting or decision effect; the record carries verdict = 'descriptive')")
    if status == "confirmatory":
        return {"statistic": stat,
                "positive_result_achieved": f"UB95 < THRESH = {cfg['thresh']} AND TU-FDC-DP(b) 0/200 false streams "
                                            "(whole run to 15/15 or tau) AND this block's replica pass",
                "positive_result_not_achieved": "otherwise; components rival_superiority_failed (sub-label "
                                                "faster_below_1 / not_faster), validity_failure, replica_fail",
                "implementation": impl}
    return {"statistic": stat,
            "verdict": "none: the block is DESCRIPTIVE by the pre-stated block-status gate; the analysis record "
                       "carries verdict = 'descriptive' and no positive / negative wording",
            "thresh": f"computed by the pre-stated rule ({cfg['thresh']}) and recorded, NOT applicable (never "
                      "compared with the eval UB95 for a verdict)",
            "descriptive_report": ["paired geometric-mean ratio TU-FDC-DP(b) / RECT-BF-DP-TU with one-sided UB95 and "
                                   "two-sided CI", "per-method false streams (whole run) with Clopper-Pearson UB",
                                   "eval eps-optimal share and all-control count", "exhaustion at N80",
                                   "full-frontier ratio (rows to 15/15)", "uncensored-subset ratio",
                                   "descriptive ratios vs HC-WoR-DP[tuned] and RECT-HG-DP (CP)",
                                   "replica status (R1, R2, R2b)"],
            "implementation": impl}


def block_entry(c, cfg):
    fz = load_frozen(c)
    sel = json.loads(RV.gate_file(c, "eps_hc").read_text())
    st = json.loads(RV.gate_file(c, "status").read_text())
    th = json.loads(RV.gate_file(c, "thresh").read_text())
    dev = dev_numbers(c)
    if abs(VA.thresh_rule(dev["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["ub95_one_sided"]) - th["thresh"]) > 1e-12:
        raise SystemExit(f"{c}: THRESH gate does not follow the rule applied to the dev analysis")
    if datetime.fromisoformat(st["written_at"]) >= datetime.fromisoformat(_pilot_started(c)):
        raise SystemExit(f"{c}: block status was not frozen before the dev cell (FDC-DP rows) started")
    seeds = VA.EVAL_SEEDS[c]
    return {
        "block": BLOCK_LABEL[c], "campaign": f"random/{c}", "status": st["status"],
        "status_gate": {k: st[k] for k in ("rule", "criteria", "selected_eps", "eps", "n_problems",
                                           "n_problems_all_control_eps_optimal", "mean_share_eps_optimal",
                                           "per_problem_share", "n_policies")} |
        {"written_at": st["written_at"], "dev_cell_started_at": _pilot_started(c)},
        "layer": LAYERS[c], "task": RV.eval_task(c), "seeds": f"{seeds[0]}-{seeds[-1]} (200 new streams)",
        "eps": cfg["eps"], "thresh": cfg["thresh"], "methods": list(VA.METHODS),
        "table": f"Open Bandit Dataset random/{c} eval half (blinded salted row-level split, "
                 f"open_bandit/{c}/PROVENANCE.json); the whole half is the replay population (tau = eval N)",
        "table_use": "first use of this eval half; no eval-outcome statistic computed before the lock",
        "design": {
            "S": fz["S"], "A": 2, "n_items": N_ITEMS[c],
            "segments": fz["segment_rule"] + "; frozen explicit map in " + str(RV.gate_file(c, "frozen").relative_to(WS))
                        + "; " + fz["fallback_rule"] + " (fallback row counts reported)",
            "arms": fz["arm_rule"] + "; arm a = show a uniformly random item of group a; frozen item lists; an eval "
                                     "item outside the lists stops the run",
            "costs": "kappa[s, 0] = 0, kappa[s, 1] = max(1, round(8 S w_s^dev)); budgets floor(b_q sum kappa[s, 1]), "
                     "b_q = 0.10 .. 0.80 (15 problems); frozen from dev; policy values use the eval segment weights",
            "cost_vector": [x[1] for x in fz["cost"]], "budgets": fz["budgets"],
            "dev_segment_shares": [round(x, 5) for x in fz["dev_segment_shares"]]},
        "eps_selection": {
            "rule": sel["rule"], "hc_rule": sel["hc_rule"], "grid": sel["eps_grid"],
            "result": {k: sel["eps_rule"][k] for k in ("selected_eps", "rival", "rival_method", "successes", "n")}
            | {"block_eps": sel["block_eps"], "hc_tuned_at_block_eps": sel["hc_tuned_at_block_eps"]},
            "trace": [{k: t[k] for k in ("eps", "best", "best_method", "qualifies")} |
                      {"per_rect": {m: {x: v[x] for x in ("method", "successes", "geomean_N80_pen")}
                                    for m, v in t["per_rect"].items()}} for t in sel["eps_rule"]["trace"]],
            "units": {"absolute_ctr": cfg["eps"], "dev_base_ctr": dev["eps_units"]["dev_base_ctr"],
                      "relative_to_dev_base_ctr": dev["eps_units"]["relative_to_dev_base"],
                      "dev_mean_share_of_feasible_policies_eps_optimal": dev["eps_optimal_share"]}},
        "thresh_rule": {"rule": th["rule"], "ub95_dev": th["ub95_dev"], "ratio_dev": th["ratio_dev"],
                        "value": th["thresh"], "source": th["source"]},
        "thresh_applicable": st["status"] == "confirmatory",
        "decision_rule": decision_rule(c, cfg, st["status"]),
        "dev_runner_check": dev}


def main():
    v11 = prereg_v11.load_locked_addendum(None)
    opt = [p for p in OPTIONAL_INPUTS if (WS / p).exists()]
    missing = [p for p in INPUTS if not (WS / p).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    man = json.loads(RV.MANIFEST_JSON.read_text())["sha256"]
    for f in RV.GATE_FILES:
        if man.get(f) != sha(RV.GATES / f):
            raise SystemExit(f"MANIFEST.json does not match {f}")
    cfgs, gates = {}, None
    for c in CAMPAIGNS:
        cfgs[c], gates = RV.load_frozen(c, None, with_thresh=True)
    blocks = {c: block_entry(c, cfgs[c]) for c in CAMPAIGNS}
    add = {
        "version": "5+v12-addendum", "status": "draft", "written_at": datetime.now().date().isoformat(),
        "written_by": "experimenter (v12 OBD replication; design plan/v12_obd2_plan.md, committed before any "
                      "split or dev read of either campaign)",
        "addendum_to": {"version": 5, "path": "plan/prereg_lock.json", "sha256": v11["addendum_to"]["sha256"],
                        "git_commit": v11["addendum_to"].get("git_commit")},
        **{k: {"path": f"plan/prereg_lock_{k[:-9]}_addendum.json", "sha256": v11[k]["sha256"],
               "git_commit": v11[k].get("git_commit")} for k in ("v6_addendum", "v7_addendum", "v8_addendum",
                                                                  "v9_addendum", "v10_addendum")},
        "v11_addendum": {"path": "plan/prereg_lock_v11_addendum.json", "sha256": v11["sha256"],
                         "git_commit": v11["git_commit"]},
        "v11_known_input_drift": v11_known_drift(),
        "statement": (
            "Append-only addendum. It edits neither lock v5 nor the v6-v11 addenda and cannot change their verdicts. "
            "It registers TWO blocks of a second replication of lock v11, each with its own block-specific outcome "
            "(a verdict for a confirmatory block, a descriptive report with verdict = 'descriptive' for a descriptive "
            "block) and no pooling: "
            "block A (primary) on the eval half of Open Bandit random/women and block B on the eval half of "
            "random/men, two uniform-random campaigns that no earlier step had read. In each block TU-FDC-DP(b) is "
            "compared with the matched, parameter-free time-uniform rectangle RECT-BF-DP-TU under the IDENTICAL "
            "written v11 protocol, with design, eps, HC tuning and THRESH re-derived on the campaign's own dev half "
            "by the v11 rules, and a pre-stated block-status gate (plan s6) deciding whether the block is "
            "confirmatory or descriptive. OUTCOME OF THE GATE (dev, frozen before the registered, saved dev cells -- the "
            "first saved FDC-DP dev rows; an unsaved pre-rule smoke test and the test suite's one-stream check are "
            "disclosed below and fed no rule): BOTH BLOCKS ARE "
            "DESCRIPTIVE. Gates (i) and (ii) pass for both campaigns, gate (iii) fails: the mean eps-optimal policy "
            "share is " + f"{blocks['women']['status_gate']['mean_share_eps_optimal']:.3f}" + " for women (eps "
            + f"{blocks['women']['eps']:g}" + ") and " + f"{blocks['men']['status_gate']['mean_share_eps_optimal']:.3f}"
            + " for men (eps " + f"{blocks['men']['eps']:g}" + "), both > 0.5. Both blocks are nevertheless locked, "
            "run once on their eval halves, sealed, replicated and analysed, and reported descriptively with no "
            "verdict. Inference is conditional on each finite eval table."),
        "block_status_summary": {c: {"status": blocks[c]["status"], "eps": blocks[c]["eps"],
                                     "criteria": blocks[c]["status_gate"]["criteria"],
                                     "mean_share_eps_optimal": blocks[c]["status_gate"]["mean_share_eps_optimal"],
                                     "n_problems_all_control_eps_optimal":
                                         blocks[c]["status_gate"]["n_problems_all_control_eps_optimal"],
                                     "thresh_computed": blocks[c]["thresh"],
                                     "thresh_applicable": blocks[c]["thresh_applicable"]} for c in CAMPAIGNS},
        "primary_method": {
            "name": VA.PRIMARY,
            "construction": "dsswm.baselines.fdc_dp.FDCDPTimeUniform(block_points = the K = 40 checkpoint grid, "
                            "scheme='b', node_limit 200000, grid ratio 2000^(1/160)); identical to lock v11",
            "guarantee": "Theorem DP-1 (plan/fdc_dp_theory.md) + Lemma TU: FWER <= 0.05 per stream and problem "
                         "family, for all t >= t_1, under the frozen outcome-free 50/50 schedule",
            "fdc_dp_sha256": sha(CODE / "dsswm/baselines/fdc_dp.py"),
            "rect_dp_sha256": sha(CODE / "dsswm/baselines/rect_dp.py")},
        "rivals": {
            "governing": {VA.RIVAL: "dsswm.baselines.rect_dp.RectBFDPTU(block_points = K = 40 grid, box=True): "
                                    "Bennett-FPC rectangle matched to FDC-DP, exact worst-gap DP, parameter-free"},
            "descriptive": {
                VA.HC_TUNED: "hedged-capital WoR CS rectangle (time-uniform), tuned per campaign on dev by the "
                             "block-G3 rule at the block eps (frozen_configs.<campaign>)",
                VA.RECT_HG: "exact hypergeometric per-cell rectangle at delta / (2 S A K) with the worst-gap DP; "
                            "CHECKPOINT-STRENGTH (CP) guarantee only; v12 rows carry validity "
                            "'checkpoint_strength_CP' (never 'rigorous')"},
            "claim_scope": "claims are restricted to the named implementations; not 'the best valid rectangle'"},
        "blocks": blocks,
        "pooling": "none: each block has its own block-specific outcome (confirmatory block: verdict; descriptive "
                   "block: descriptive report, verdict = 'descriptive', never a positive / negative verdict); block B "
                   "cannot rescue block A or vice versa; both outcomes "
                   "are reported whatever they are",
        "frozen_configs": json.loads(json.dumps(cfgs)),
        "frozen_gates": gates,
        "seed_manifest": {
            "dev": "950-999 on each campaign's whole dev half (rule tasks, dev cell)",
            "eval": {c: f"{VA.EVAL_SEEDS[c][0]}-{VA.EVAL_SEEDS[c][-1]}" for c in CAMPAIGNS}, "bootstrap_seed": 42,
            "previously_used_eval": "30000-30199, 31000-31199, 32000-32199, 33000-33199, 34000-34199, 35000-35399, "
                                    "36000-36199, 37000-37799, 38000-38999, 39000-39199 (all on other tables)"},
        "replica_check": {
            "implementation": "dsswm/stats/v11_replica.py (imported unchanged) + run_v12.py --task <task> --replica",
            "R1": f"independent process re-runs the first {R1_N_SEEDS} eval seeds of the block; canonical rows "
                  "identical",
            "R1_excluded_fields": list(TIMING_FIELDS),
            "R2": "schedule_digest identical across all four methods per seed (frozen 50/50 design)",
            "R2_groups": [list(g) for g in VR.groups_for(RV.eval_task("women"))],
            "R2b": "arrival_digest identical across all methods per seed",
            "on_failure": "confirmatory block: replica_fail -> positive_result_not_achieved for that block; "
                          "descriptive block: the replica failure is reported explicitly in the descriptive record "
                          "(replica_status = 'fail'), verdict stays 'descriptive'"},
        "descriptive_prestated": [
            "ratios vs HC-WoR-DP[tuned] and RECT-HG-DP (CP), RECT-BF-DP-TU vs each",
            "per-method false streams with Clopper-Pearson UB; exhaustion at N80; share reaching 15/15",
            "full-frontier ratio (N_stop_pen, rows to 15/15; horizon-sensitive; never the endpoint)",
            "uncensored-subset ratio (streams where neither TU-FDC-DP(b) nor RECT-BF-DP-TU had an exhausted pool at "
            "N80; bootstrap B = 10^4 seed 42 on the subset; subset size reported)",
            "eval base CTR, eps relative to it, eval eps-optimal share and all-control count; fallback row counts"],
        "eval_tasks": {RV.eval_task(c): {"role": f"block {BLOCK_LABEL[c]} ({blocks[c]['status']})", "campaign": c,
                                         "layer": LAYERS[c], "eps": cfgs[c]["eps"], "thresh": cfgs[c]["thresh"],
                                         "block_status": blocks[c]["status"],
                                         "seeds": f"{VA.EVAL_SEEDS[c][0]}-{VA.EVAL_SEEDS[c][-1]}",
                                         "methods": list(VA.METHODS), "K": VA.K_GRID, "stop_k": 15, "n80_k": VA.N80_K}
                       for c in CAMPAIGNS},
        "eval_access": {
            "reader": "dsswm.envs.obd_v12_eval.read_eval_rows(campaign, task, layer, lock_sha256): caller arguments "
                      "-> prereg_v12 gate (no eval byte) -> task registered for exactly this campaign and layer -> "
                      "caller's lock sha256 equals the lock -> one line appended (fsync) to "
                      "exp/results/full/v12_obd/<campaign>_eval_access_log.jsonl -> _verify_eval_bytes (private) "
                      "against the lock and the campaign PROVENANCE -> deserialise",
            "pre_lock_reads": (
                "Split preparation (2026-10-05, exp/code/split_obd_v12.py, run once after the plan commits; the "
                "split rule was committed before any outcome inspection): the splitter loaded the full source "
                "<campaign>.csv (all columns, clicks included) from the zip, partitioned it by the salted rule, "
                "serialised the eval clicks to eval_outcome.npy, wrote eval_labels.pkl, and opened and hashed both "
                "eval files to record their sha256 in the campaign PROVENANCE.json. It computed no eval-outcome "
                "statistic: the only eval quantities recorded are the eval row count and non-outcome timestamp-"
                "overlap counts. The PROVENANCE flag eval_outcomes_touched = false means 'no eval-outcome statistic "
                "computed or inspected'; it does not deny this outcome-independent split processing (the frozen "
                "PROVENANCE file is not rewritten). After the split, no v12 dev, builder, gate or test code path "
                "opened, hashed or deserialised either campaign's eval files: all four eval hashes are taken from "
                "the hash-bound campaign PROVENANCE.json."),
            "gate_order": "caller campaign / layer / sha format -> v12 gate (status, schema, task membership, "
                          "inherited v11 -> v5 chain, canonical hash, code / input / non-eval data drift, "
                          "single-commit history) -> registered campaign + layer -> caller sha equals lock -> "
                          "access-log line (fsync) -> eval bytes vs lock and PROVENANCE -> deserialise"},
        "command": "cd exp/code && for C in women men: python run_v12.py --task v12_$C_full [--workers 4]; "
                   "--task v12_$C_full --replica; --campaign $C --analyse. Every step re-checks the locked v12 "
                   "addendum (and through it v11-v5).",
        "eval_seal": "v12_seal (as v8-v11: seal committed alone at task completion + sidecar ref commit; "
                     "single-commit history anchor); exp/results/full/v12_seals",
        "threat_model": "as v6-v11: accidental drift, partial / resumed runs and post-hoc edits are detected against "
                        "git history; deliberate history rewriting is out of scope.",
        "declarations": [
            "FDC-DP's statistical certificate is FDC-BF's (Theorem DP-1); the contribution is computational; the "
            "blocks test speed against named implementations.",
            "Headline claim restricted to methods with proven finite-sample / anytime-valid guarantees (user ruling "
            "2026-10-03); plug-in methods are not run.",
            "The protocol (uf0 x uf3 segmentation rule, top-half item grouping by dev CTR, traffic-share costs, K = "
            "40, eps grid and rule, THRESH rule) was chosen on random/all dev for v11 and is applied here "
            "mechanically to each campaign's own dev half; the arm grouping uses dev outcomes, so each eval uplift "
            "gap shrinks by selection.",
            "The block-status gate (plan s6) is new in v12, pre-stated before any split; it can only demote a block "
            "to descriptive.",
            "Row-level split within each campaign: dev rows sharing an exact timestamp with the campaign's own eval "
            "half: women 3.27 %, men 3.04 %; user ids are absent, so repeat users can occur in both halves and "
            "across campaigns. Cross-campaign exact-timestamp sharing (non-outcome): women rows vs random/all eval "
            "half 35 (0.004 %), men 29 (0.006 %), women vs men 27 rows; all three campaigns cover 2019-11-24 .. "
            "2019-11-30 (open_bandit/<campaign>/PROVENANCE.json).",
            "First use of both eval halves; inference conditional on each finite table.",
            "Same dev seeds 950-999 as v11 (different tables).",
            "HC-WoR-DP is tuned on the same dev streams used by the eps rule (selection-optimistic for HC; HC is "
            "descriptive). RECT-HG-DP has a checkpoint-strength (CP) guarantee only (descriptive).",
            "THRESH = min(0.90, UB95_dev + 0.10) per campaign is dev-calibrated (rule pre-stated; value computed "
            "after the campaign's dev run, before the lock).",
            "Before the eps rule ran, a smoke test ran one dev stream (seed 950, eps 5e-4) of each of the four "
            "methods (incl. TU-FDC-DP(b)) on each campaign's frozen dev env to check the code path; its rows were not "
            "saved, fed no rule and cannot be independently compared (plan/v12_dev_report.md). This is a disclosed "
            "exception to the plan's wording that the status file is written 'before any FDC-DP dev row exists' "
            "(plan s6): the status files precede the registered, saved dev cells in both runs.",
            "The top-level open_bandit/PROVENANCE.json is not amended (lock v11 binds its sha256); the pointer to "
            "the v12 splits is open_bandit/V12_SPLITS.json.",
            "Dev block regenerated once: run 1 (committed d23b7aaa) was bound to code before 39dace28, which only "
            "relabels descriptive-block outputs (no verdict; THRESH marked not applicable) on coordinator "
            "instruction after the gate result; the rule tasks, eps / HC / status selection, dev cells, dev replica, "
            "dev analysis and THRESH were regenerated on that code. All 4,980 dev rows reproduce run 1 exactly "
            "(N80_pen, cert_k, decided policies, digests, false flags, N_stop_pen, exhaustion); the gate files "
            "differ only in code hash, timestamps, the added applicability fields and, as expected, the THRESH "
            "source_sha256 (the regenerated dev analysis) and the corresponding MANIFEST hashes; the frozen design "
            "files were not rewritten. 'Rewritten once' refers to the eps/HC, block-status and THRESH gates and "
            "MANIFEST (plan/v12_dev_report.md).",
            "Dated observation (2026-10-05, not a design change): the identical protocol gives near-trivial frontiers "
            "on the two smaller campaigns because the rival-success rule pushes eps to 7.5e-4 (women) and 1e-3 "
            "(men), where most feasible policies are eps-optimal (dev: almost every segment has a positive uplift).",
            "The test suite runs one dev stream per method (women dev, seed 950, eps 1.5e-3) as a code-path check; "
            "it feeds no rule.",
            "bts/* stays unread.",
            "Timings: up to 4 runner processes x 4 workers concurrently; biased up."],
        "code_sha256": {p: sha(CODE / p) for p in RV.CODE_FILES + RV.LOCK_ONLY_FILES},
        "input_sha256": {p: sha(WS / p) for p in INPUTS + opt},
        "data_sha256": data_hashes_v12(),
        "lock_procedure": "external reviewer lock review -> rebuild draft -> code-freeze commit -> "
                          "prereg_v12.finalize_addendum(git_commit) -> commit the lock file alone -> per block: eval "
                          "task, replica, analysis.",
        "git_commit": None,
    }
    from dsswm.stats.prereg_v12 import DRAFT_PATH, _schema
    _schema(add, require_commit=False)
    add["sha256_draft"] = prereg.canonical_hash({k: v for k, v in add.items() if k != "sha256_draft"})
    DRAFT_PATH.write_text(json.dumps(add, indent=1, ensure_ascii=False))
    print(f"wrote {DRAFT_PATH} sha256_draft={add['sha256_draft']}")


if __name__ == "__main__":
    main()
