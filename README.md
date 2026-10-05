# Anonymised code and artifact package (double-blind review)

This package holds the code, the locked pre-registrations (base lock v5 and addenda v6 to v12), the external lock
reviews, the sealed result rows and analyses, the frozen gates, and the scripts that recompute every number printed
in the paper and supplement. Identifiers have been removed: author names, e-mails, user names, machine paths, host
names, git author data, and the names of the agent tooling and of the external review tool. No raw or derived
datasets are included; the Data section below says how to obtain each one and gives its SHA-256.

## Layout

    exp/code/dsswm/                     library: certificates (FDC, FDC-BF, FDC-DP, FDC-HG, ...), rectangles and other
                                        baselines, replay environments, statistics, lock gates, seals, tests
    exp/code/dsswm/baselines/fdc_dp.py  FDC-DP (exact joint certificate over exponentially large segment-policy classes)
    exp/code/dsswm/baselines/rect_dp.py RECT-BF-DP(-TU), the matched rectangle rival of lock v10
    exp/code/dsswm/baselines/fdc_bet_guarded.py  guarded public entry point for checkpoint-strength FDC-BF (see
                                        "Checkpoint API usage contract" below); fdc_bet.py itself is lock-bound
    exp/code/dsswm/envs/seg_v10*.py     uplift-score segmentations (dev fit, frozen eval cut points)
    exp/code/dsswm/baselines/fdc_{pw,ls}.py, dsswm/envs/pw_reference.py, run_v11_pw_dev.py
                                        localised union ledgers FDC-PW / FDC-LS (supplement S20; development only,
                                        negative; tests dsswm/tests/test_fdc_{pw,ls}.py; notes plan/v11_plan.md)
    exp/code/run_v11.py, dsswm/envs/obd_v11{,_eval}.py, dsswm/stats/{prereg_v11,v11_analysis,v11_replica,v11_seal}.py
                                        lock v11 (Open Bandit Dataset, never-replayed randomized log): runner, frozen-design
                                        environment and gated evaluation reader, lock gate, analysis, replica, seal;
                                        tests dsswm/tests/test_v11_addendum.py; run_obd_dev_pilot.py: feasibility pilot;
                                        build_v11_addendum_draft.py: lock-draft builder (paper Section 6.6, supplement S21)
    exp/code/run_v12.py, split_obd_v12.py, dsswm/envs/obd_v12_eval.py, dsswm/stats/{prereg_v12,v12_analysis,v12_seal}.py
                                        lock v12 (Open Bandit random/women and random/men, DESCRIPTIVE replications of
                                        lock v11): runner, blinded salted split, campaign-parameterised gated reader, lock
                                        gate, analysis with the pre-stated block-status gate, seal; tests
                                        dsswm/tests/test_v12_addendum.py; build_v12_addendum_draft.py: lock-draft builder
                                        (paper Section 6.6, supplement S23)
    exp/code/dsswm/tests/               unit and regression tests (pytest)
    exp/code/run_r5s_v6.py ... run_v11.py   lock-block runners (v6, v7, v8, v9, v10, v11)
    exp/code/run_fdc_dp_dev.py          FDC-DP development runs and the pre-stated eps rule (lock v10)
    exp/code/run_v7_posthoc_2x2.py, run_v8_posthoc_{D,E,F,F2}.py, run_v8_select.py   post-hoc descriptive blocks
    exp/code/build_v*_addendum_draft.py builders of the lock drafts; ingest_v8_fresh.py: blinded X5 dev/eval split
    exp/code/recompute_v10_from_sealed_rows.py   artifact helper: recompute the v10 analyses from the sealed rows
    exp/code/run_*.py (others)          runners of earlier rounds (r2-r5), kept for completeness
    plan/prereg_lock.json               base lock (v5); plan/prereg_lock_v{6..11}_addendum.{json,md}: locked addenda (v11: json only);
                                        *_DRAFT.*: the drafts that were reviewed before each lock
    plan/*dev_report.md, plan/*theory*.md, plan/theory/   development reports and theory notes bound by the locks
    reviews/                            external reviews of the locks and of the FDC-DP / FDC-HG methods
    exp/results/full/                   eval outputs: v*_summary.md, v*_analysis_*.json, per-task summary.json and
                                        replica_report.json, v*_seals/ (seal + seal_ref per eval task), post-hoc blocks
    exp/results/full/<task>/results.jsonl   sealed rows shipped for the 6 v10 tasks and the 4 tasks the verify chain reads
                                        (v7a_full_b, v8a_full_a, v8a_full_b, v8c_full)
    exp/results/pilots/                 development-half analyses and summaries referenced by the locks
                                        (v11: pilots/v11_dev/ rule and pilot rows, pilots/obd_dev/ feasibility rows)
    exp/results/full/v11_obd/           lock-v11 sealed rows (v11_obd_full/results.jsonl), replica rows and report,
                                        v11_analysis.json, eval_access_log.jsonl; v11_seals/; v11_summary.md
    exp/results/v11_gates/              lock-v11 frozen design, eps / HC selection, THRESH, MANIFEST
    exp/results/full/v12_obd/           lock-v12 sealed rows (v12_{women,men}_full/results.jsonl), replica rows and
                                        reports, v12_{women,men}_analysis.json, {women,men}_eval_access_log.jsonl;
                                        v12_seals/; v12_summary.md
    exp/results/v12_gates/              lock-v12 frozen designs, eps / HC selection, block-status gate, THRESH (computed,
                                        not applicable), MANIFEST; exp/results/pilots/v12_dev/: development rule and cell
                                        rows, replica, development analyses (run logs not shipped)
    exp/results/v{6,8,9,10}_gates/      frozen configurations; v10_gates/ holds the frozen score models
                                        (score_model_{x5,lenta}.pkl, scikit-learn 1.9.1) and seg_v10_frozen.json
    exp/results/r4_gates, r5_gates      earlier frozen gates referenced by the base lock
    writing/scripts/                    verify_r{3,4,4b,5,5b,6,7,7b,8a,8,8b,8c,8d}_numbers.py, gen_r{4,5,5b}_supp_tables.py,
                                        gen_r7_v11_tables.py (supplement S21 tables), gen_r8_v12_tables.py (S23 tables),
                                        gen_r6_blockG_tables.py (block-G tables), make_fig_epscurve.py (Figure 2)
    writing/latex_acm/main.tex, main.pdf, writing/supplement/  paper (revision r8d) and supplement sources that the
                                        verify scripts check; main_pre_r8.tex: the r8a text that verify_r8 compares with;
                                        main_pre_r8b.tex, supplement_pre_r8b.tex, exp/results/full/v12_summary_pre_r8b.md:
                                        the r8 texts that verify_r8b rebuilds the r8b wording from; main_pre_r8c.tex,
                                        supplement_pre_r8c.tex, main_r8b.pdf: the r8b state; main_pre_r8d.tex,
                                        supplement_pre_r8d.tex, main_r8c.pdf: the r8c state (both re-checked by verify_r8d)
    writing/motivation_sources.{md,bib}  sources of the motivation (Sections 1, 2, 6.2 and supplement S16): URL, date,
                                        verbatim quote and verdict for each, all opened and checked on 2026-10-05
    requirements.txt                    pinned versions (Python 3.12.14)

## Setup

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    export DATA_DIR=/path/to/datasets      # only needed to rerun experiments; default ./data

Runners are started from `exp/code`; they resolve the package root as two levels above that directory. CPU only;
BLAS threads are pinned to 1 and each runner uses at most 4 worker processes (`--workers`).

## Data (not included)

Datasets are expected under `$DATA_DIR/<dir>/`. The SHA-256 values are those of our local copies.

| Dataset | Source | Files read and SHA-256 |
|---|---|---|
| Criteo Uplift v2.1 (13,979,592 rows), dir `criteo_uplift_real/` | Hugging Face `criteo/criteo-uplift`, file `criteo-research-uplift-v2.1.csv.gz` (CC-BY-NC-SA 4.0) | `criteo-research-uplift-v2.1.csv.gz` (311,422,618 bytes) `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`; our `tidy.pkl` `600a9a57a0e93a90c659552f482609321d97c58b1c9c90a6a391e05811546cd8` |
| X5 RetailHero (200,039 clients), dir `x5_retailhero/` | Hugging Face `pytorch-lifestream/retailhero-uplift` (mirror of the ods.ai RetailHero data), files `uplift_train.csv.gz`, `clients.csv.gz` | `uplift_train.csv.gz` `23aced68634c605acb93a7ab450aabac1a0ce0c8104e59ffed9941109ddc4ccd`; `clients.csv.gz` `b8985170e03dc65fa532fb6b8ca6dd70ea5c30c35ea7c1019ee6ea2034d04099`; split outputs below |
| Open Bandit Dataset (ZOZOTOWN; 1,374,327 rows of `random/all`), dir `open_bandit/` | ZOZO Research, `https://research.zozo.com/data_release/open_bandit_dataset.zip` (CC BY 4.0; Saito et al., NeurIPS 2021 Datasets and Benchmarks) | zip (412,931,917 bytes) `e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78`; `open_bandit_dataset/random/all/all.csv` is read (lock v11), and `random/women/women.csv` and `random/men/men.csv` (lock v12, split outputs in the paragraph below). Our `random/all` split outputs: `dev.pkl` `ac59e8b9b40ee29f026daf8b1057a455678d60fb4a5201e75ec6acd0157ee748`, `eval_labels.pkl` `b27cd3b93d56e2a14ef6417f2f9d9623b5b125688578efa419aa22e7adaeb8af`, `eval_outcome.npy` `cd0e23a339de3ed0aba6a14af1ab00f468a0c400966e0ad4c5b66dd190436ca8`, `PROVENANCE.json` `656ca6fc973c3b7df775cbe197c9c4f8f657d83982b49cae6cf22be5c6683118` |
| Lenta (687,029 rows), dir `lenta/` | scikit-uplift `fetch_lenta`, i.e. `https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz` (research use) | `lenta_dataset.csv.gz` `b531544f6c072d22f232d91e20ffcaca265acf224e02687502a40e6d56135682`; `tidy.pkl` (62,520,974 bytes) `6d075bbc306df3512f6bc3a795c94686984b4c99da3889c91778d25325565d43` |
| Hillstrom (64,000 rows), dir `hillstrom/` | MineThatData e-mail challenge, `Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` | raw csv `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`; `tidy.pkl` (2,113,247 bytes) `3dab9ed72bfdcaf51f6d993975a7ca052ef9355690489ca854217b7f0a68b296` |

Tidy tables. The Lenta and Hillstrom `tidy.pkl` files are pandas DataFrames with columns `treatment, outcome, binary,
weight` followed by the source covariates prefixed `x_` (Lenta: `response_att, response_amt, x_age, x_children,
x_main_format, x_months_from_register, x_promo_share_15d, x_food_share_1m, x_k_var_cheque_3m,
x_mean_discount_depth_15d`; Hillstrom: `x_recency, x_history, x_mens, x_womens, x_newbie, x_channel, x_zip,
x_history_segment, conversion`). The Criteo `tidy.pkl` is the csv.gz loaded as one DataFrame. The ingest code that
produced these three pickles is not part of this package; an independent re-ingest will in general not reproduce
the pickle bytes, so the pinned-hash checks in `dsswm/envs/{data_v6,lenta_v6,hillstrom_v9}.py` will then report data
drift. The hashes let one confirm that the same source file was used.

X5 split. `exp/code/ingest_v8_fresh.py` builds the blinded X5 split from the two raw files: a row is in the
development half iff `int(sha256(f"{salt}|x5_retailhero|{client_id}")[:8], 16) / 2**32 < 0.5`, with salt
`dsswm-v8-2026-10-04`. Our outputs: `dev.pkl` `3a8a5d8e35599f69461f403c36210dfa497073d9f679d1f6e039d165aa85eae4`,
`eval_labels.pkl` `500fb32edaa522812adb6728df8bdbeb1fa2e6d8db0c08f750b1f1827b223eaa`, `eval_outcome.npy`
`ed0f95f4a4cbbdee8f81c9dc548bfe37e9e8b31beee9045dc7c0f62daa0e322d` (99,646 dev rows, 100,393 eval rows).
Open Bandit split (lock v11). From `random/all/all.csv`, a row is in the development half iff
`int(sha256(f"{salt}|open_bandit|{row_id}")[:8], 16) / 2**32 < 0.5`, salt `dsswm-obd-2026-10-05`, where `row_id` is the
CSV's unnamed first column (the 0-based row index 0..1,374,326). This gives 686,168 development and 688,159 evaluation
rows. `dev.pkl` holds all 90 columns of the development rows (`row_id` renamed from the unnamed index);
`eval_labels.pkl` holds every column except `click` for the evaluation rows in original order; `eval_outcome.npy` holds
their `click` as int8, aligned row by row (made read-only). `PROVENANCE.json` records the download date, zip hash,
split rule and the file hashes above; its hash is bound by the lock, and the lock-v11 reader takes the evaluation-file
hashes from it. Put the four files under `$DATA_DIR/open_bandit/`. Until lock v12, `random/men`, `random/women` and
`bts/*` were never read; `bts/*` is still unread. The split was written before any click was read, and only label statistics of the evaluation half were computed
at split time; 3.3% of development rows share an exact timestamp with an evaluation row (supplement S21, disclosure ii).
Open Bandit splits (lock v12, descriptive). `exp/code/split_obd_v12.py` splits `random/women/women.csv` and
`random/men/men.csv` from the same zip once each: a row is in the development half iff
`int(sha256(f"{salt}|{name}|{row_id}")[:8], 16) / 2**32 < 0.5`, with salt `dsswm-obd-women-2026-10-05` and name
`open_bandit_women`, or salt `dsswm-obd-men-2026-10-05` and name `open_bandit_men`, and `row_id` the CSV's unnamed first
column. This gives 432,009 / 432,576 (women) and 226,819 / 226,130 (men) development / evaluation rows. Outputs per
campaign under `$DATA_DIR/open_bandit/{women,men}/` (same layout as above): women `dev.pkl`
`db0f612ca1d5dd9bd060583d3a5212b5f3a0672ca236387997e7b1cc4f3662cb`, `eval_labels.pkl`
`ab99556b118dc6a2cd28a333caf802b2470e230247e8596d5de04187f78fd8ce`, `eval_outcome.npy`
`19a0f4243eb77f1b8b6154ff080a6cf6df953b49427e70e19bdd4df1e386b2b2`, `PROVENANCE.json`
`98013351a3b05fd0a1f4f4ddf3d30dbb15d9bdeb221bd0137602caa71e5c8953`; men `dev.pkl`
`039ae7524be20d449f594945aeffb8b14364797d693678d63403fdab21b2b22f`, `eval_labels.pkl`
`e76afe2ca3d8a1bc604fd680e8511aa7b2da9059856b89f7b3ec8150876a534b`, `eval_outcome.npy`
`ec3adc8dd277656c387d382fb0c3d1c04d212d24e3a4deb64093aba39e06e3a4`, `PROVENANCE.json`
`aab9400b7e8f609313a84d7ed79401cebfeb8151b07d96faa3a06961084b42a7`. The lock binds these hashes (evaluation hashes
from the provenance records). The `random/all` files and their provenance record were not modified; the pointer to the
new splits is a separate `open_bandit/V12_SPLITS.json`. No Open Bandit data file is shipped here.
The Hillstrom split salt is `ds-swm/v9/hillstrom/2026-10-04` (`dsswm/envs/hillstrom_v9.py`). Both salts are functional
constants and were left unchanged.

## Tests

    cd exp/code
    ../../.venv/bin/python3 -m pytest dsswm/tests -q          # about 7-8 min

`dsswm/tests/conftest.py` turns tests that fail only because a dataset or a helper file is absent into skips
(set `DSSWM_STRICT=1` to disable this). The expected result in this copy is in "Test status" at the end.

## Reproducing the paper tables from the sealed results

Everything below runs without data, without git history and without the lock gates.

1. Paper and supplement numbers (all tables and all printed figures of locks v6 to v10 and the post-hoc blocks).
   From the package root:

       .venv/bin/python3 writing/scripts/verify_r8d_numbers.py

   `verify_r8d` (the paper as submitted, revision r8d) first re-runs `verify_r8b` on the r8b text and `verify_r8c` on the
   r8c text, each in a temporary copy of the package, and requires `0 mismatch(es)` from both. Revision r8c added sourced
   motivation (citations in Sections 1, 2 and 6.2, sourced and dated prices in supplement S16; see
   `writing/motivation_sources.md`) and moved former appendix Tables A2 and A3 unchanged to supplement S24; revision
   r8d applied reviewer wording fixes. Neither changes a registered number: `verify_r8c` checks that the multiset of
   numbers in the paper differs from r8b only by listed price, date, section-reference and layout tokens, and
   `verify_r8d` rebuilds the current texts from the r8c copies by exactly the r8d replacements (byte equality). Both
   repeat the page check (12 pages, body ends on page 8). Expected final line `r8d: 0 mismatch(es)`.

   `verify_r8b` (revision r8b, revision r8b: wording fixes only, no number changes) runs `verify_r8` with its
   whole chain and lists the 8 r8 text checks whose wording was deliberately changed in r8b as SUPERSEDED (exhaustion
   qualified "at N80", branch-and-bound "per lock-v10 cell", the v12 tolerance sentence replacing "(gate 0.5)",
   "outcomes" for "data", C1 "is consistent with", Section 7 "errata to locked files"). It rebuilds `main.tex`,
   `supplement.tex` and `v12_summary.md` from their `*_pre_r8b` copies by exactly the r8b replacements (byte equality),
   checks every new string and the absence of each replaced one, recomputes the facts behind the new wording from the
   sealed lock-v12 rows (tolerance shares 15.5% / 19.9% versus 8.6% in lock v11; no false certificate over the whole run;
   no exhausted pool at N80; which runs reach the full-table horizon at 15 of 15), checks the dated corrections in
   `v12_summary.md`, and repeats the page check. Expected final line `r8b: 0 mismatch(es)`; inside it, `verify_r8`
   ends with `r8: 8 mismatch(es)`, exactly the superseded items.

   `verify_r8` (revision r8) runs `verify_r8a` (revision r8a, which runs `verify_r7b` and the
   whole chain below) and lists the 6 chain text checks whose wording was deliberately shortened in r8 to make room for
   lock v12 as SUPERSEDED, each with its replacement checked and every cut fact checked verbatim in supplement S16, S17
   or S22. It then recomputes lock v12 from the sealed rows `exp/results/full/v12_obd/v12_{women,men}_full/results.jsonl`
   (paired by seed, bootstrap B = 10^4, seed 42): every comparison, the full-frontier and uncensored-subset readings,
   F/T/S shares, false streams, exhaustion, stopping points and checkpoint gains, against the analysis JSONs; checks
   every lock-v12 number of the paper (Section 6.6, Section 7) and of supplement S23 and S2 against the analyses, the
   development analyses, `v12_gates/`, the lock, the plan, the development report, `v12_summary.md` and the two lock
   reviews; checks the S23 tables against `gen_r8_v12_tables.py`; and checks the page budget of `main.pdf` (12 pages,
   body ends on page 8). Expected final line `r8: 0 mismatch(es)`. Inside it, `verify_r8a` ends with
   `r8a: 6 mismatch(es)`: exactly the superseded items. Both lock-v12 blocks are descriptive by the pre-stated gate;
   no script derives a verdict for them.

   `verify_r8a` (revision r8a) runs `verify_r7b` and lists the 8 r7b text checks reworded in r8a (C1 foregrounding,
   Section 8 clause) as SUPERSEDED, each with its replacement checked.

   The older entry point is

       .venv/bin/python3 writing/scripts/verify_r7b_numbers.py --skip-census

   `verify_r7b` (the paper as submitted, revision r7b) runs `verify_r7` with its whole chain, lists the 12 r7 text
   checks whose wording was deliberately corrected in r7b as SUPERSEDED (each with its replacement checked), then
   recomputes from `exp/results/full/v11_obd/v11_obd_full/results.jsonl` (paired by seed, bootstrap B = 10^4, seed 42)
   the lock-v11 headline, the per-stream stopping points and exhaustion fractions (the lock-v10 pre-exhaustion
   condition is not met at block level), the descriptive uncensored-subset (0.821) and full-frontier (0.853) ratios and
   the development ratio, checks the corrected text of the paper, supplement S2/S9/S14/S20-S22 and
   `exp/results/full/v11_summary.md`, and asserts that the removed wording is absent. Expected final line
   `r7b: 0 mismatch(es)`; `[note]` lines say that the dataset provenance record, git history and `main.pdf` are not
   shipped, so those checks are skipped.

   `verify_r7` (revision r7) first runs `verify_r6` with its whole chain, lists the 21 r6
   checks whose wording was deliberately replaced or moved to the supplement in r7 as SUPERSEDED (each with its
   replacement checked: moved paragraphs verbatim in supplement S22, Figure 2 in S19), and then checks every lock-v11
   number printed in the paper and in supplement S21 against `exp/results/full/v11_obd/v11_analysis.json`, the
   development analysis, the v11 gates, the lock declarations and the replica report; the S21 tables against
   `gen_r7_v11_tables.py`; the thesis, abstract and sentence rules; and the Open Bandit bibliography entry. Against
   the r7 text its final line was `r7: 0 mismatch(es)`; against the r7b text it reports exactly the 12 items that
   `verify_r7b` lists as SUPERSEDED; two `[note]` lines say that the dataset provenance record and `main.pdf` are not
   shipped, so the split numbers are checked against the analysis and gates and the page check is skipped.
   Inside `verify_r7`, `verify_r6` itself ends with `r6: 21 mismatch(es)`: exactly the superseded items.

   `verify_r6` (revision r6d) first runs `verify_r5b`, then checks the r6, r6c and r6d text,
   the S20 ratios (recomputed from `exp/results/pilots/fdc_pw/rows*/*.jsonl.gz` against the block-G1 development rows),
   the pricing arithmetic of the priced-access workflow (S16), and the block-G numbers against `exp/results/full/v10_posthoc_G/summary.json` and the shipped block-G rows (gunzipped on the
   fly); against the r6d text its final line was `r6: 0 mismatch(es)`, and against the r7 text it reports exactly the 21 items that `verify_r7` lists as SUPERSEDED. Chain text checks whose wording was deliberately replaced in r6 or
   r6c are listed by name as SUPERSEDED in `verify_r6`, each with its replacement checked; the older scripts are
   therefore not claimed to pass on their own against the current text.
   `verify_r5b` first runs `verify_r5`, which runs the chain `verify_r4b -> verify_r4 -> verify_r3`, and then checks
   the lock-v10 numbers; inside `verify_r6` its text checks are reported as OK or SUPERSEDED, and only the final
   `r7b: 0 mismatch(es)` line of `verify_r7b` is the expected result. Each script recomputes the printed numbers from `exp/results/full/v*_analysis_*.json`, the
   post-hoc `summary.json` files, the development analyses in `exp/results/pilots/`, and (for `verify_r3`) the sealed
   rows of v7a_full_b, v8a_full_a, v8a_full_b and v8c_full. It compares them with `writing/latex_acm/main.tex` and
   `writing/supplement/supplement.tex`, regenerates the supplement tables with `gen_r*_supp_tables.py` and checks
   that they equal the shipped `.tex` tables. Run on their own against the current text, the older scripts report the
   superseded wording as mismatches; this is expected, and `verify_r6` lists each such check with its replacement.
   `--skip-census` skips the near-tie census, which reads the development halves of Criteo and X5. With the data
   present, drop the flag to recompute it as well.

   | Lock / block | Source files | Read by |
   |---|---|---|
   | v12, descriptive (main-text Sec. 6.6 and 7, supplement S23) | `v12_obd/v12_{women,men}_analysis.json`, `v12_obd/v12_{women,men}_full/{results.jsonl,replica_report.json}`, `v12_obd/*_eval_access_log.jsonl`, `pilots/v12_dev/v12_{women,men}_analysis_dev.json`, `v12_gates/*.json`, `plan/prereg_lock_v12_addendum.json`, `plan/v12_{obd2_plan,dev_report}.md`, `reviews/v12_lock_review_r{1,2}.md`, `v12_summary.md` | `verify_r8`, `gen_r8_v12_tables` |
   | v11 (main-text Sec. 6.6, supplement S21) | `v11_obd/v11_analysis.json`, `v11_obd/v11_obd_full/replica_report.json`, `pilots/v11_dev/v11_analysis_dev.json`, `v11_gates/*.json`, `plan/prereg_lock_v11_addendum.json` | `verify_r7`, `verify_r7b` (also `v11_obd_full/results.jsonl`, `v11_summary.md`), `gen_r7_v11_tables` |
   | localised ledgers, development only (supplement S20, Sec. 6.5 pointer) | `pilots/fdc_pw/rows*/*.jsonl.gz`, `pilots/fdc_pw/s20_summary.json` (Hamming / value-gap diagnostics; `gen_r6d_s20.py --diag` needs the development halves) | `verify_r6`, `gen_r6d_s20` |
   | post-hoc block G (main-text Figure 2 and Sec. 6.5, supplement S19) | `v10_posthoc_G/summary.json` (rows in `v10_posthoc_G/rows/*.jsonl.gz`) | `verify_r6`, `gen_r6_blockG_tables`, `make_fig_epscurve` |
   | v10 A, B, D (main-text Table 2, Sec. 6.5, supplement S17 and ledger) | `v10_analysis_{A,B,D}.json`, `pilots/fdc_dp_v2/analysis.json` | `verify_r5b`, `gen_r5b_supp_tables` |
   | v9 A, B, C | `v9_analysis_{A,B,C}.json`, `v9_summary.md`, `pilots/v9_analysis_{A,B}_dev.json` | `verify_r4b`, `verify_r5`, `gen_r4_supp_tables` |
   | post-hoc blocks D, E, F, F2 | `v8_posthoc_{D,E,F,F2}/summary.json` | `verify_r4`, `verify_r5`, `gen_r5_supp_tables` |
   | v8 A, C | `v8_analysis_{A,C}.json`; sealed rows of v8a_full_a/b, v8c_full | `verify_r3`, `verify_r4`, `verify_r4b`, `verify_r5` |
   | v7 A | `v7_analysis_A.json`; sealed rows of v7a_full_b | `verify_r3`, `verify_r4`, `verify_r4b`, `verify_r5` |
   | v6 (erratum) | `v6_summary.md` | `verify_r4` |
   | development records | `pilots/{fdc_hg,v9_candidates,hillstrom_v9,width_prop}/...` | `verify_r4`, `gen_r4_supp_tables` |

   The remaining analyses (`v6_analysis_*`, `v7_analysis_B`, `v8_analysis_B`) are shipped for inspection; the verify scripts
   do not read them.

2. Lock-v10 analyses from the sealed rows (the headline result):

       cd exp/code && ../../.venv/bin/python3 recompute_v10_from_sealed_rows.py

   For each of the six v10 tasks it checks that the content hash of the shipped `results.jsonl` equals
   `results_content_sha256` in the committed seal, then recomputes blocks A, B and D with
   `dsswm.stats.v10_analysis` (the code path of `run_v10.py --analyse`) and compares every numeric field and the
   verdict with the shipped `v10_analysis_*.json`. Expected output: `v10 recompute: 0 mismatch(es)`; in this copy all
   six hashes match, verdicts `positive_result_achieved` (A, B) and `descriptive` (D), max |diff| 0.

## Rerunning a lock block

Each runner has development twins (dev halves, dev seeds 900-999) and eval tasks (eval halves, fresh seeds).
With the data under `$DATA_DIR` (cwd `exp/code`):

    python run_v10.py --task v10a_pilot_s16          # dev twin of v10a_full_s16; also _s32, v10b_pilot_s16/_s32/_s64,
                                                     # v10d_pilot_x5s64
    python run_v10.py --analyse A --dev              # -> exp/results/pilots/v10_analysis_A_dev.json
    python run_fdc_dp_dev.py ...                     # development runs and the eps rule (see the file header)
    python run_r5s_v9.py --task v9a_pilot_d          # lock-v9 dev twins; run_r5s_v8.py / _v7.py / _v6.py likewise

An eval task runs only behind its lock gate:

    python run_v10.py --task v10a_full_s16           # eval rows (fresh seeds 38000-38199), sealed on completion
    python run_v10.py --task v10a_full_s16 --replica # replica check against the replica model
    python run_v10.py --analyse A                    # -> exp/results/full/v10_analysis_A.json
    python run_v11.py --task v11_obd_full            # lock v11 (seeds 39000-39199); then --replica, --analyse
    python run_v12.py --task v12_women_full          # lock v12, descriptive (seeds 39200-39399; v12_men_full:
                                                     # 39400-39599); then --replica, --analyse with --campaign

The gate (`dsswm/stats/prereg_v10.py`, and likewise `prereg_v6..v9`) checks the whole chain v5 -> v10: every lock's
canonical hash, the SHA-256 of every bound code file, input and data file, and a git anchor (the lock file was
committed exactly once and equals that commit's bytes). The seals are checked against git history in the same way.
**In this scrubbed copy the gates cannot pass, so eval tasks write `status: skipped_by_lock` and stop.** Scrubbing
changed the bytes of bound code and lock files, and the original git history is not included. This is the intended
behaviour of the gate, not a defect. To regenerate eval rows bit-for-bit, use the unscrubbed repository with its
history, which will be released at de-anonymisation. The sealed rows shipped here, with the content-hash check
above, are the means to audit the reported eval results in the meantime. The gate logic itself is covered by
unit tests that build their own temporary git repositories (`test_v*_addendum.py`, `test_prereg_v5.py`).

Post-hoc block G (descriptive, outside any lock; needs the data): `cd exp/code && python3 run_v10_posthoc_G.py --list`
lists its cells; run the dev cells with `--cells ...`, then `--select-hc` (writes `hc_tuned.json`), then the eval
cells, then `--analyse`. Rows are keyed by a hash of the library files, so shipped rows (`rows/*.jsonl.gz`, gunzip
first) are reused and only missing jobs are run. `exp/results/full/v10_posthoc_G/SUMMARY.md` records the rules and
one disclosed deviation (a failed first eval launch that logged access but computed no row), and ends with a dated
errata section (r6c: cost range 1-24, branch-only denominator, S = 16 runtime; r6d: the 80% pre-horizon wording
replaces the earlier "uncensored" shorthand).

## Checkpoint API usage contract

`FDCBet.certify` (`dsswm/baselines/fdc_bet.py`) is the evaluated checkpoint-strength FDC-BF. It recomputes variance
boxes and Chernoff widths at every call and checks only that times do not decrease, while its ledger
`beta = ln(M K / delta_main)` and its box level `delta_var / (2 S A K)` count exactly the K predeclared checkpoints
`ctx.checkpoints`. Arbitrary increasing times therefore do not authorise fresh widths: a certificate issued between
checkpoints with fresh widths is an event the K-event ledger never paid for. The file is hash-bound by locks v7-v10,
so it is left unchanged, and new code should use the guarded entry point instead:

    from dsswm.baselines.fdc_bet_guarded import make_guarded_variant
    m = make_guarded_variant("FDC-BF")      # identical to make_variant("FDC-BF") on the checkpoint grid
    m.setup(ctx); m.certify(ctx, state)     # raises OffGridCertifyError if state.t is not a predeclared checkpoint
                                            # or not strictly later than the previous call

For dense (off-grid) monitoring use `dsswm.baselines.fdc_loc.FDCTimeUniform(block_points=checkpoints)` (TU-FDC,
Remark 4.2 of the paper): it refreshes widths only at block points and reuses the last block's widths in between.
The implicit-class certificates `FDCDP` / `RectBFDP` carry the same guard (dense monitoring: `FDCDPTimeUniform`,
`RectBFDPTU`). Tests: `dsswm/tests/test_fdc_bet_guarded.py`. Every evaluated FDC-BF row was produced on the
checkpoint grid or through `FDCTimeUniform`, so the contract changes no result.

## Lock index

`file sha256 (original)` is the SHA-256 of the unscrubbed file bytes. `canonical sha256` is the lock's own
`sha256` field: the hash that every eval row, seal and analysis binds as `addendum_sha256`. It is unchanged by
scrubbing because it is stored inside the file. `file sha256 (this copy)` is the hash of the shipped, scrubbed
file. `git commit` is the lock's recorded code-freeze commit in the original history.

| Lock | Locked at | File sha256 (original) | Canonical sha256 (bound by results) | File sha256 (this copy) | Code-freeze commit |
|---|---|---|---|---|---|
| v5 base `prereg_lock.json` | (round 5) | `e8ecb8e8a4da0c5410e8e137b60c191305f6de378cc6466bd9c36075dee73a95` | `64196b5aeb03097730112fb915efd9639e7b77bfbe2574abed62e14256a67948` | identical (no edit) | `ad9dbbc8` |
| v6 `prereg_lock_v6_addendum.json` | 2026-10-04 01:34 | `eee15b776c3a979d2982ecbed2c11f3d74252876ce867805bdcf89b29269f105` | `9c8dc94479c575989a79faf6aa04e7449270de9c07ee2f3e73615b924f31d497` | `8944a6584f64d367f471a1d2c555e9d45573e7aaa6bfc291f73386b40aa2fc7b` | `10985185` |
| v7 `prereg_lock_v7_addendum.json` | 2026-10-04 06:35 | `1f1a67d1a7edb915effca9b64360372af00c0c41db3371d19a89030d9b2d4b87` | `47e195f0577e70c7d856164176ec7de165717896ed0fb839283e70cc0e1cc68f` | `3f8e8e596c18b54bf168d3ff16d09a998a02e47a0f447e4394fb69470d895705` | `bfd929bd` |
| v7 `prereg_lock_v7_addendum.md` | | `7fb5f80e0bed27abdfec6ffc4c80e89aa8700335ccf938cc064b8421cc5fbc2d` | | identical (no edit) | |
| v8 `prereg_lock_v8_addendum.json` | 2026-10-04 09:56 | `fd8d4e4487f8111f160b469f8b85a3023bf5b6a50f07c19a5fa1f0573c3d019f` | `729bdcbd32f821b842db8c52594388bdf5fbeb4dfa790748c1f80a9852a1cff0` | `73fab4aad45ce8613b74ff85c564e8ee06ecbc35214ccbb6e339074a09e79d77` | `5de13da1` |
| v8 `prereg_lock_v8_addendum.md` | | `6c605b8ea222ae2f0b025573d1f6a0b14aea58f827a0f503719606e5474f12cd` | | `3c27d9d93c99f24aa5bbe6497287f75a5286c4b66e23fab99c401435ba41426e` | |
| v9 `prereg_lock_v9_addendum.json` | 2026-10-05 02:29 | `e7d8772530b336be5451206957ec87579f63430f5f3ab93c1a1bc4fd0ced65e2` | `9540de0c7266e931b8c7945cb7382e7809f47196a8e41cf49267db67541fda79` | `e2d7e8f114eb4b255a8dc7c91e73183e3b0226fd27853eba24592e673e9a9468` | `29be77b7` |
| v9 `prereg_lock_v9_addendum.md` | | `7e920d0e7988b38bba38d0e74c592788f6f952f91dbc3d2184d6fcdd742f2d84` | | `9d7498e70d3b8686cbe2958903dd4a8431c7eb660e88078771469a35d88ddaf6` | |
| v10 `prereg_lock_v10_addendum.json` | 2026-10-05 06:35 | `fc9ff22625d205d8d8ce2aae10639400705b4c5235da9c6f349c653c32e0a173` | `ecaf78e1c5424d996ed775194a6f0f8041e0ce3c5c1b9effc4a99388069d82d8` | `b7b9f16c0f00153009ca42c5aa7865ec14335a26ac26d088bb8e940ec309eaf8` | `2984121d` |
| v11 `prereg_lock_v11_addendum.json` | 2026-10-05 15:12 | `d91e6e99feaa4e9adfa8d9e03a219a388307984e0a0e7b39502503a7dc3b2e52` | `4ee733915bf2ac7759bcf90b2bc945b5dce88d3c3625bd1e2d7531a65e629963` | `1902174c8fb2b73744abaa7355ef6223a00ab826873f69d85dcabb4271a5a014` | `1741ebcf` |
| v12 `prereg_lock_v12_addendum.json` | 2026-10-05 18:34 | `838d8f4e60adc307b8c6af1a60811d2e64c1159059bf074596f0045197dedb53` | `9a99ef28b3c2213853faab65cbf914ce5cfc25fa16f9a950d383336a3075c189` | `aba6ef975dedcefa2a4221f50ca81b02bd23f88ac16e6eed8ea6980024e38f69` | `20a5e314` |
| v10 `prereg_lock_v10_addendum.md` | | `2bfc9c1a076dcaa0e06adc23ce5b3c1ccfbad7c7ccd014795398fde79406e7d1` | | `32c552ba10b28aba5190686d6a0467e1672bf3b4ca5a88b291e69b9ee8207449` | |

Scrubbing edits to the locks and to all other text files: data paths read `$DATA_DIR/...`; workspace paths are
relative to the package root; the agent-tool names, the name of the external review tool (now "external reviewer";
its review files moved from a tool-named folder to `reviews/`), and session / author labels were replaced by
generic words; git commit trailers name a generic assistant. Numbers, hashes, seeds, thresholds and decision rules
were not edited. Some lock and report prose is in Chinese. The sealed `results.jsonl` files and the frozen score-model
pickles are byte-identical to the originals; they contained no identifiers.

## Reproducibility notes

* Seeds: development 900-999 (v10 dev twins 950-999); eval seeds per lock: v6 31000-31199 and 32000-32199 (block C continues
  the v5 streams 30000-30199), v7 33000-33199 and 34000-34199, v8 35000-35399 (block C re-runs 33000-33199),
  v9 37000-37799, v10 38000-38399 (block G 38600-38799), v11 39000-39199 (development 950-999 on the Open Bandit development half), v12 39200-39399 (women) and 39400-39599
  (men) (development 950-999 on each campaign's development half). Bootstrap B = 10^4, seed 42.
* The raw rows of the v6-v9 eval tasks, the replica rows and the run logs are not shipped (about 150 MB). The analyses,
  summaries and seals are shipped, and so are the rows that the verify chain and the v10 recompute read.
* The seal sidecars (`*.seal_ref.json`) name commits of the original history, which is not included.

## Test status

Result in this copy (Python 3.12.14, no data present, outside any git repository): **848 passed, 103 skipped,
7 failed** in 7.5 min. Revision r6c added `test_fdc_bet_guarded.py` (4 tests, all pass); revision r6d added `test_fdc_pw.py` and `test_fdc_ls.py` (12 tests; in this copy the three files give 16 passed in about 3 min); a rerun of the r6c copy
from inside a git checkout gave 855 passed, 103 skipped, 6 failed (the 5 truth-import false positives and
`test_live_lock_if_v5`; the commit-resolving test passes there). The skips are tests whose dataset is absent. All 7 failures are expected:

* `test_no_truth_import.py::test_static_no_truth_imports[stats/prereg_v6|v7|v8|v9|v10.py]` (5): **known false
  positive.** This static check rejects any import whose dotted name contains `.envs` from a `stats/` module. The lock
  gates `prereg_v6..v10` lazily import `envs/data_v6`, `data_v8`, `data_v9` and `envs/seg_v10_eval` only to compare
  data-file SHA-256 values with the lock (`DATA_FILES*`, `data_drift*`). They load no outcomes and no ground truth,
  and the runtime import-closure test of the same file passes. The same 5 tests fail identically in the unscrubbed
  original tree. The test file is hash-bound in the v5 lock and the gate modules in the addenda, so neither was
  edited. See `exp/results/full/truth_import_audit.md`, written for v6-v8; v9 and v10 follow the same pattern.
* Revision r7 added `test_v11_addendum.py` (26 tests; 26 pass in the unscrubbed tree). In this copy it gives 18 passed,
  7 skipped (evaluation data absent) and 1 failed: `test_real_gate_layer_sha_logging_and_order` runs the real v11 gate,
  which re-checks the hashes of the v5-frozen code, and scrubbing changed the bytes of five of those files
  (`dsswm/certify/quadknap.py`, ...). This is the same by-design lock-hash failure as the two tests below.
* Revision r8 added `test_v12_addendum.py` (28 tests; 28 pass in the unscrubbed tree; not included in the totals above).
  In this copy it gives 19 passed, 6 skipped (evaluation data absent) and 3 failed:
  `test_real_gate_refuses_uncommitted_lock_without_eval_access` and `test_real_gate_layer_sha_logging_and_order[women|men]`
  run the real v12 gate, which re-checks the hashes of the v5-frozen code changed by scrubbing. This is the same
  by-design lock-hash failure as the v11 test above.
* `test_prereg_v5.py::test_live_lock_if_v5` and `test_v6_addendum.py::test_finalize_refuses_overwrite_and_bad_commit`
  (2): **lock-hash tests that fail by design in a scrubbed copy.** The first compares the SHA-256 of every code file
  bound by the live v5 lock with the shipped bytes. Scrubbing changed the bytes of the files that contained paths or
  tool names (e.g. `dsswm/certify/quadknap.py`), so it reports drift. The second resolves the package's git `HEAD` and
  checks the bound code against that commit. That needs the original history: it fails with "not a git repository"
  outside a repository, and with a code mismatch inside a fresh one. See "Rerunning a lock block" above.
