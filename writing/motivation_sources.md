# Motivation sources: verified 2026-10-05

Every source below was opened during this task (WebFetch, direct PDF download and text extraction, the arXiv API, or Crossref DOI metadata). Quotes are copied from the fetched text. Access date for all: **2026-10-05**. BibTeX for verified sources not already in `latex_acm/references.bib`: `writing/motivation_sources.bib`.

## Summary: recommended citations

| # | Key | Claim it supports | Verdict |
|---|---|---|---|
| 1a | `awscleanrooms2026pricing` (existing) | Clean-room record matching billed per 1,000 records matched ($0.50) plus $100 one-time fee per collaboration; queries $2/CRPU-hour, 32 CRPU default, 60 s minimum | SUPPORTS (re-verified) |
| 1d | `awsentityresolution2026pricing` (new) | Standalone matching billed per record ($0.25 per 1,000 records processed) | SUPPORTS (same vendor, so not independent) |
| 1c | `snowflake2026dcrcost` (new) | A second vendor bills clean-room analyses by compute time, not per record | PARTIAL: supports "billed per query/compute", not "per record" |
| 1b | `googleadh2026privacychecks` (new) | Clean-room results must be aggregates over a minimum number of users (about 50 under difference checks) | SUPPORTS the "aggregate-only queries with minimum-count thresholds" sentence in S16; it is not about prices |
| 2 | `feldman2021individual` (existing) | A record left out of an analysis incurs no privacy loss under individual accounting, if the choice to leave it out does not depend on its data | SUPPORTS (with that condition, which the outcome-free frozen plan meets) |
| 2 | `ebadi2015personal` (new) | Per-individual privacy budgets; a query that does not touch someone's data costs that person nothing | SUPPORTS |
| 2 | `census2021dsep` (existing) | Total ε = 19.61 for the 2020 redistricting data product | SUPPORTS (re-verified) |
| 2 | `apple2017dp` (existing) → replace with `apple2017dpoverview` (new) | "ε = 4 per donation" | Existing source PARTIAL (says ε = 4 and "per-event"; the word "donation" is absent). New source SUPPORTS verbatim |
| 3 | `awsgroundtruthblog` (existing) | $0.26 per human annotation | PARTIAL: verified, but it is a 2019 worked example whose composition the post does not give |
| 3 | `awssagemaker2026pricing` (new) | Current Ground Truth list price per reviewed object: $0.08 / $0.04 / $0.02 by monthly tier, labour extra | SUPPORTS |
| 4 | `johari2017peeking` (existing) | Always-valid inference deployed at Optimizely | SUPPORTS (Optimizely is named in the abstract) |
| 4 | `johari2022always` (new) | Journal version; "trade off sample size and power" | SUPPORTS |
| 4 | `lindon2026anytime` (new) | Netflix: experiments may be "stopped early" with anytime-valid inference | SUPPORTS |
| 4 | `schultzberg2023sequential` (new) | Spotify; mSPRT "used ... by Optimizely, Uber, Netflix, and Amplitude" | SUPPORTS |
| 5 | `schultzberg2024riskaware` (new) | Platforms bound the risk of the ship decision through an explicit decision rule | PARTIAL: an internal platform practice, not a requirement |
| 5 | `fda2019adaptive` (optional) | Regulators require "the chance of erroneous conclusions" to be controlled | DOES NOT SUPPORT for targeting (clinical trials only; nonbinding) |
| 5 | EU DSA Art. 37 | Annual independent audits of very large platforms | DOES NOT SUPPORT (no statistical error requirement) |
| 6 | `zhao2019uplift` (new) | Uber: uplift model on experiment data to "maximize the net value given the budget" | SUPPORTS (budget-constrained targeting after an experiment) |
| 6 | `ai2022lbcf` (new) | WWW'22: budget-constrained treatment selection evaluated offline on RCT data at several budgets; deployed to hundreds of millions of users | SUPPORTS |
| 6 | `goldenberg2020freelunch` (new) | Booking.com: the treated fraction is adjusted dynamically after deployment under an ROI constraint | SUPPORTS "budget/threshold is changed after analysis" |
| 6 | `diemert2018large` (existing) | "The budget is often set only after the analysis" | DOES NOT SUPPORT as worded: the paper evaluates uplift over every treated fraction (Qini curves), but says nothing about when budgets are set |
| 7 | `li2011unbiased` (new, optional) | Replay of logged randomized data for offline evaluation | SUPPORTS |

**Count:** 24 sources opened and assessed for a claim. 20 are verified and usable (SUPPORTS, or PARTIAL with the stated scope); two of these are the two Apple documents, and MTurk and Spotify Confidence were opened only as side checks. 4 are rejected for the claim they would carry: FDA guidance, EU DSA Art. 37 and Microsoft ExP for "audit requires"; `diemert2018large` for "budget set after analysis". 5 more could not be used: LiveRamp pricing (not public), Ads Data Hub pricing (not public), Databricks pricing page (not fetched; the docs page only links out), Netflix tech blog (HTTP 403), and Google Cloud data labelling (not verified).

---

## Task 1. Per-record or per-query billing in clean rooms

**1a. AWS Clean Rooms pricing.** URL https://aws.amazon.com/clean-rooms/pricing/ (Amazon Web Services; undated page; accessed 2026-10-05).
- Quotes: "You pay for the usage of compute on a price-per-CRPU-hour" basis, US East (N. Virginia) "$2.00 per CRPU-hour". "By default, AWS Clean Rooms allocates 32 CRPUs to each Spark SQL query." "Spark SQL queries run on a per-second basis (with a 60-second minimum charge)."
- Under AWS Entity Resolution on AWS Clean Rooms: rule-based matching "$0.50 per 1,000 records matched". "For rule-based matching, one collaborator is required to pay a one-time $100.00 matching fee per collaboration, and this fee is assigned to any collaborator paying for data matching."
- Verdict: **SUPPORTS**. S16's "$1.07 per minimum-billed 32-CRPU query" checks out: 32 × (60/3600) h × $2 = $1.067.

**1b. Google Ads Data Hub.** URL https://developers.google.com/ads-data-hub/guides/privacy-checks, "Privacy checks in Ads Data Hub" (Google; last updated 2026-09-25 UTC).
- Quotes: "Aggregation checks ensure that every row contains a large enough number of users to protect end-user privacy." "Difference checks require approximately 50 unique users per result row." "Noise injection requires approximately 20 unique users per result row."
- Verdict: **SUPPORTS** the claim that clean-room outputs are aggregate rows above a minimum user count, which is what the certifier's per-cell counts and sums need. No public Ads Data Hub price page was found; a search turned up only third-party summaries, so no price claim can rest on it.

**1c. Snowflake Data Clean Rooms.** URL https://docs.snowflake.com/en/user-guide/cleanrooms/cleanroom-cost, "Snowflake Data Clean Rooms operational costs" (Snowflake; undated).
- Quote: "The cost of executing a workload depends on the time required for the workload to complete within the warehouse specified by the user."
- Verdict: **PARTIAL**. A second vendor bills per analysis by compute time, which is not per record. The per-checkpoint query bill therefore follows checkpoints and compute rather than rows, consistent with S16's remark on AWS queries.
- Databricks: https://docs.databricks.com/aws/en/clean-rooms/ says only "To learn more about Databricks Clean Rooms pricing, see [link]". The pricing page itself was not opened, so it is **not used**.
- LiveRamp/Habu: no public per-record or per-query price was found, so it is **not used**.

**1d. AWS Entity Resolution (standalone).** URL https://aws.amazon.com/entity-resolution/pricing/ (AWS; undated).
- Quotes: "Rule-based and ML-powered matching: $0.25 per 1,000 records processed". "A record is defined as a row of data that can have multiple columns representing input data".
- Verdict: **SUPPORTS** per-record billing of identity matching. It comes from the same vendor, so per-record billing is verified from **one vendor only**. The paper should say "e.g., AWS" rather than imply an industry-wide price.

## Task 2. Per-record and individual privacy accounting

**Feldman & Zrnic.** NeurIPS 2021 (Advances in NeurIPS 34, pp. 28080–28091, per the NeurIPS proceedings BibTeX); arXiv:2008.11193 v4 (2022-01-08). Text read from the arXiv PDF.
- Quotes: "we keep track of a personalized estimate of the privacy loss divergence for each individual in the analyzed dataset ... We do so by applying each analysis only to the points that are estimated to have sufficient leftover privacy budget." "By the current design of Algorithm 3, Xi ∉ St implies no privacy loss for point Xi. This is true because, conditional on a1,...,at−1, Xi being inactive ensures that St would be the same regardless of whether the input to Algorithm 3 is S or S−i."
- Verdict: **SUPPORTS**, with a condition. A record is free only if the decision to leave it out does not depend on its own data beyond the published outputs. FDC-BF's frozen read plan is drawn from segment and arm only, never from outcomes, so it meets the condition. The paper should say so in a few words, e.g. "an unread record, chosen without looking at its outcome, spends no budget".

**Ebadi, Sands & Schneider.** "Differential Privacy: Now it's Getting Personal", POPL '15, Mumbai, DOI 10.1145/2676726.2677005 (pages 69–81 from Crossref/ACM). Text read from the authors' PDF at https://www.cse.chalmers.se/~gersch/popl2015.pdf.
- Quotes: "In PDP each individual has its own personal privacy level. We describe ProPer, a interactive system for implementing PDP which maintains a privacy budget for each individual." "For example, any query about the drinking habits of adults offers 0-differential privacy for Adrian, aged 13". Also: "If ε1 is large, the analyst has blown the budget by analysing the small group, even though that study did not touch the data of the larger part of the population."
- Verdict: **SUPPORTS**. This is the cleanest statement that untouched records spend nothing.

**census2021dsep.** URL https://www.census.gov/.../2021-06-09.html, "Census Bureau Sets Key Parameters to Protect Privacy in 2020 Census Results" (U.S. Census Bureau, 2021-06-09).
- Quote: "The approved DAS production settings reflect a total privacy-loss budget for the redistricting data product ... of ε=19.61, which includes ε=17.14 for the persons file and ε=2.47 for the housing unit data."
- Verdict: **SUPPORTS**. Say "for the redistricting data product".

**apple2017dp (existing).** URL https://machinelearning.apple.com/research/learning-with-privacy-at-scale (Apple, 2017-12-06).
- Quotes: "we define a per-event privacy parameter, ϵ"; "For this use case, we set the parameters for CMS to be m = 1024, k = 65,536, and ϵ = 4".
- Verdict: **PARTIAL**. The word "donation" does not appear on this page.

**apple2017dpoverview (new).** URL https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf, "Differential Privacy Overview" (Apple; undated text, PDF created 2017-11-02).
- Quotes: "The Apple differential privacy implementation incorporates the concept of a per-donation privacy budget (quantified by the parameter epsilon)". "For emoji, Apple uses a privacy budget with epsilon of 4, and submits one donation per day." QuickType uses epsilon 8 and Health types epsilon 2.
- Verdict: **SUPPORTS** "ε = 4 per donation (emoji)". Cite this PDF instead of, or alongside, `apple2017dp`.

## Task 3. Per-item labelling prices

**awsgroundtruthblog.** URL https://aws.amazon.com/blogs/machine-learning/annotate-data-for-less-with-amazon-sagemaker-ground-truth-and-automated-data-labeling/ (Chalupka, Zhdanov, McKinney; 2019-02-06).
- Quote: "Without automatic data labeling, the annotations would have cost $0.26 * 1000, which equals $260." Also: "Instead, you paid $158.08 for 608 human labels".
- Verdict: **PARTIAL**. The figure is real, but it is one 2019 worked example and the post does not break it down. (Arithmetically it equals $0.08 + 5 × $0.036, but the post does not say so; do not claim this.)

**awssagemaker2026pricing (new).** URL https://aws.amazon.com/sagemaker/ai/pricing/, Ground Truth section (AWS; undated). The rendered page fills its prices from AWS's pricing data file at https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/sagemaker/USD/current/groundtruth.json, which I downloaded and read (US East, N. Virginia).
- Page text: "You are charged for the number of dataset objects that are reviewed. A dataset object is defined as an atomic unit of data across all modalities."
- Tiers: "Less than 50,000 objects" $0.08; "50,000 to 1,000,000 objects" $0.04; "Greater than 1,000,000 objects" $0.02 per reviewed object per month.
- Labour: "If you use Amazon Mechanical Turk for labeling, you are charged per object per review instance." The suggested image-classification price per labeler is $0.012.
- Verdict: **SUPPORTS** a current per-object list price for the service fee alone. Note that Amazon Mechanical Turk itself "permanently closed on September 30, 2026" (https://www.mturk.com/pricing, fetched 2026-10-05), so the MTurk labour line is stale. Use only the service fee as a lower bound.
- Google Cloud Data Labeling, Scale AI and Labelbox were **not verified**. No official per-item price was found, so none is used.

## Task 4. Sequential / always-valid inference on experimentation platforms

**johari2017peeking (existing).** KDD '17, pp. 1517–1525, DOI 10.1145/3097983.3097992 (Crossref). Text read from the KDD PDF.
- Quote: "This paper reports on novel statistical methodology, which has been deployed by the commercial A/B testing platform Optimizely to communicate experimental results to their customers." And: "Not only does this make it safe for a user to continuously monitor, but it empowers her to detect true effects more efficiently."
- Verdict: **SUPPORTS**. It is Optimizely.

**johari2022always (new).** Operations Research 70(3):1806–1821, 2022, DOI 10.1287/opre.2021.2135. Text read from the open-access PDF.
- Quotes: "We define always valid p-values and confidence intervals that let users try to take advantage of data as fast as it becomes available, providing valid statistical inference whenever they make their decision." "Figure 3 shows that the benefit from stopping early outweighs the cost of additional slack in always valid decision boundaries at reasonable power levels". Endnote: "The methodology of this paper forms the core of the statistical backend of the Optimizely platform."
- Verdict: **SUPPORTS**.

**lindon2026anytime (new).** JASA, published online 2026-08-27, DOI 10.1080/01621459.2026.2692052 (Crossref title "Anytime-Valid Inference in Linear Models with Applications to Regression-Adjusted Causal Inference"); arXiv:2210.08589 (Netflix authors).
- Quotes from the arXiv abstract: the method "formally allows experiments to be continuously monitored for significance, stopped early, and safeguards against statistical malpractices in data collection." "We demonstrate the practical utility of our approach through simulations and applications to real A/B test data from Netflix."
- Verdict: **SUPPORTS**. The Netflix tech blog post returned HTTP 403 and is not used.

**schultzberg2023sequential (new).** Spotify Engineering, 2023-03-21, Schultzberg & Ankargren.
- Quotes: "A main objective of experimentation is to know early on if end users are negatively affected by the experience being tested." mSPRT was "Popularized, used, and extended by Optimizely, Uber, Netflix, and Amplitude, for example."
- Verdict: **SUPPORTS**.

Also opened: Spotify Confidence docs (https://confidence.spotify.com/docs/experiments/stats/sequential-tests): "Sequential tests make it possible to analyze results during an experiment without jeopardizing the statistical integrity." This is SUPPORTS, but redundant with the above.

## Task 5. Does anything REQUIRE an error guarantee before deploying a targeting policy?

- **FDA, "Adaptive Designs for Clinical Trials of Drugs and Biologics" (Nov 2019).** Read from the PDF at https://www.fda.gov/media/78495/download: "the design, conduct, and analysis of an adaptive clinical trial intended to provide substantial evidence of effectiveness should satisfy four key principles: the chance of erroneous conclusions should be adequately controlled, ..." Every page is headed "Contains Nonbinding Recommendations". **DOES NOT SUPPORT** a requirement for targeting or advertising; it is only a cross-domain analogy.
- **EU Digital Services Act, Art. 37** (https://www.eu-digital-services-act.com/Digital_Services_Act_Article_37.html): "Providers of very large online platforms and of very large online search engines shall be subject, at their own expense and at least once a year, to independent audits ..." There is no statistical error requirement. **DOES NOT SUPPORT**.
- **Microsoft ExP, "Patterns of Trustworthy Experimentation: Post-Experiment Stage"** (Machmouchi & Gupta, 2021-12-24). It describes ship decisions with "an adequately powered A/B test" and says there is no "standard way to evaluate those tradeoffs". It states no hard requirement. **DOES NOT SUPPORT** "require".
- **Schultzberg, Ankargren & Frånberg (arXiv:2402.11609, 2024), Spotify.** Quotes: "This paper introduces the theoretical framework for decision rules guiding the evaluation of experiments at Spotify." "Being unclear about what outcomes lead to a positive product decision means that there is no mechanism for properly controlling the risks of the experiment at the level that matters to the company, namely the decision to ship the feature or not." **PARTIAL**: a platform chooses to bound the error of the ship decision; nothing mandates it.
- **Recommendation:** no source found states that an audit, regulator or platform *requires* a statistical guarantee before a targeting or advertising policy is deployed. Replace "an audit may require an error guarantee" with the weaker wording "a platform that wants to bound the risk of deploying the wrong policy needs an error guarantee", citing `schultzberg2024riskaware` (and optionally `johari2017peeking`).

## Task 6. Industrial budget-constrained targeting after a randomized experiment

- **Zhao & Harinen (Uber), IEEE DSAA 2019, pp. 422–431, DOI 10.1109/DSAA.2019.00057** (Crossref); arXiv:1908.05372 PDF read. Quotes: "While running a randomized experiment helps the team identify the best performing promotion and quantify its impact, applying an uplift model enables the team to personalize the targeting to maximize the net value given the budget." "The algorithms have been implemented in a Python package as a horizontal solution for uplift modeling at Uber." **SUPPORTS** budgeted targeting after an experiment. It does not say when the budget is fixed.
- **Ai et al., LBCF, WWW '22, pp. 2310–2319, DOI 10.1145/3485447.3512103**; arXiv:2201.12585 v2 PDF read. Quotes: the paper addresses "how to select the right amount of incentives (i.e. treatment) to each user under budget constraints". Figure 5 caption: "Offline test results on comparing the proposed algorithm with other baseline methods under different budgets ... on the real-world RCT" data. The abstract says it is "serving over hundreds of millions of users" on "a large-scale video platform". **SUPPORTS** evaluating a budget frontier on a randomized log, and it is a WWW precedent.
- **Goldenberg et al. (Booking.com), RecSys '20, pp. 486–491, DOI 10.1145/3383313.3412215**; arXiv:2008.06293 PDF read. Quotes: the method "dynamically optimizes the incremental treatment outcome subject to the required Return on Investment (ROI) constraints". "Dynamic adjustment of the threshold in group D allowed for exposing the treatment to 47% of the population". "We suggested a dynamic calibration technique, which allows adjusting the model threshold and exposing a different portion of the population to the treatment, by relying on online performance." **SUPPORTS** "the treated fraction is chosen or changed after the analysis".
- **Diemert et al. 2018 (existing cite for "budget is often set only after the analysis")**, PDF read. The paper defines uplift and Qini curves "as a function of the number of customers treated" and notes "It is usual that advertisers keep only a small control population as it costs them in potential revenue." It says nothing about when a budget is set. **DOES NOT SUPPORT** the sentence as written. Keep it as the dataset citation, and move the "budget set after analysis" support to `goldenberg2020freelunch` and `ai2022lbcf`.

## Task 7. Replay on logged data (optional)

- **Li, Chu, Langford & Wang, WSDM '11, pp. 297–306, DOI 10.1145/1935826.1935878**; arXiv:1003.5956. The abstract describes a "replay methodology for contextual bandit algorithm evaluation" that can "provide provably unbiased evaluations" from logged randomized data. **SUPPORTS** citing replay as an established offline protocol. I found no source on replaying *sequential stopping* of an experiment from a log in a different order.

---

## Proposed replacement text (main paper; about 8 added lines in total)

**Intro, line ~66 (first sentence).**
> A platform that has run a randomized targeting experiment must choose which budget-constrained policy to deploy, and if it wants to bound the risk of deploying the wrong one, as platforms do for ship decisions~\citep{schultzberg2024riskaware}, it needs an error guarantee rather than a point ranking. The budget is often fixed or adjusted only after the analysis~\citep{goldenberg2020freelunch,ai2022lbcf}, so the guarantee should hold at every budget on the frontier at once.

…and keep "which binds when outcome access is priced per record (\S\ref{sec:setting})".

**§2, "When outcome reads are the binding cost".**
> The frozen plan needs each row's segment and arm, never its outcome; if outcomes are stored and free, a full scan needs no certificate. Reads dominate when each outcome read is billed, as when a clean room bills record matching per 1,000 records~\citep{awscleanrooms2026pricing}; under individual privacy accounting, where a record left unread by an outcome-independent plan incurs no privacy loss~\citep{ebadi2015personal,feldman2021individual}; and when outcomes are labelled or delayed~\citep{chapelle2014modeling}. FDC-BF is not differentially private. [rest unchanged]

**§6.2, "A priced-access workflow" (first sentence + price sentence).**
> In a clean room that bills record matching per record, e.g. AWS Entity Resolution at \$0.50 per 1,000 records matched~\citep{awscleanrooms2026pricing}, fewer reads mean a smaller bill. [workflow unchanged] … The \$100 rule-based matching fee is charged once per collaboration, so it does not depend on the certifier.

Drop "Ten Criteo-sized campaigns re-certified four times a month save about \$10k a month" from the main paper (see below), or label it "in a hypothetical schedule of ...".

**New: experiment-size motivation (2–3 sentences, §2 or intro paragraph 1).**
> Experimentation platforms already use anytime-valid tests so that an experiment can stop once its evidence suffices~\citep{johari2022always,lindon2026anytime,schultzberg2023sequential}; FDC-BF asks the same question of a finished log. A 50/50 replay is not a prefix of the original logging order, so fewer replay reads correspond only approximately to a smaller or shorter experiment, and $N_{80}$ should be read as outcomes revealed, not users enrolled.

**New: industrial uplift budget case (one sentence).**
> Budgeted targeting from randomized data is routine: Uber fits uplift models to experiment data "to maximize the net value given the budget"~\citep{zhao2019uplift}, and a deployed WWW'22 system compares treatment-selection rules across budgets on an RCT log~\citep{ai2022lbcf}.

## Proposed S16 cost table (every price sourced and dated)

| Scenario (published unit price; source, access date) | Criteo | X5 |
|---|---|---|
| Outcome reads saved per certification (measured) | 505k | 42.7k |
| Share of the rectangle's outcome bill (measured) | 36% | 64% |
| Rule-based record matching, $0.50 per 1,000 records matched (AWS Clean Rooms pricing, accessed 2026-10-05; excludes one-time $100 fee per collaboration) | $253 | $21 |
| Standalone identity matching, $0.25 per 1,000 records processed (AWS Entity Resolution pricing, accessed 2026-10-05) | $126 | $11 |
| Human review, service fee only, $0.04 per reviewed object, 50k–1M/month tier, labour excluded (Amazon SageMaker AI pricing, Ground Truth, accessed 2026-10-05) | $20.2k | $1.7k |
| Records with untouched privacy budget (measured; principle: Ebadi et al. 2015, Feldman & Zrnic 2021) | +7.2% | +42.5% |

Arithmetic: 505,304 × 0.00025 = $126.3 and 42,708 × 0.00025 = $10.7; 505,304 × 0.04 = $20,212 and 42,708 × 0.04 = $1,708. At X5's size alone the <50k tier ($0.08) applies and gives $3.4k; say "at the 50k–1M tier" in the caption. Privacy row: 505,304 / 6,989,799 = 7.23% and 42,708 / 100,393 = 42.5%.

Text items:
- "$2 per CRPU-hour ... about $1.07 for a minimum-billed 32-CRPU query" is **verified**; keep it.
- Census ε = 19.61 (redistricting data product) is **verified**; keep it.
- Apple: change "ε = 4 per donation" to cite `apple2017dpoverview`, or reword to "ε = 4 per event" if it stays on `apple2017dp`.

**Numbers without an adequate source, to delete or relabel:**
1. **"Human labels, $0.26 each" ($131k / $11.1k).** The only source is a 2019 blog worked example with no stated breakdown, and its MTurk labour component now refers to a service that closed on 2026-09-30. Replace it with the sourced service-fee row above, or delete it.
2. **"Ten Criteo-sized campaigns re-certified four times a month ... about $10k a month" (§6.2 and S16).** The campaign count and certification frequency have no source; they are an assumed scenario. Delete it from the main paper, or label it explicitly as a hypothetical schedule in S16.
3. **"No public per-record price exists for purchased conversion data".** This is a claim of absence that I could neither confirm nor refute (no vendor price was found). Soften it to "we found no public per-record list price for conversion data".
4. **"The budget is often set only after the analysis~\citep{diemert2018large}".** The citation does not support the claim; re-cite as proposed above.
5. **"an audit may require an error guarantee".** No source found; reword as proposed above.
