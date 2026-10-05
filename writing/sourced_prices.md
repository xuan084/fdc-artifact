# Sourced price points for the cost table (accessed 2026-10-04)

Reliability key: [V] fetched and quoted from the primary page this session; [S] secondary source; [U] not verified this session.

## 1. Clean-room / privacy-safe join (primary: AWS Clean Rooms pricing, [V])
URL: https://aws.amazon.com/clean-rooms/pricing/
Quoted: Spark SQL "$2.00 per CRPU-hour", default "32 CRPUs" per query, minimum charge 60 seconds; PySpark "$4.00 per CRPU-hour", "10-minute minimum charge"; differential privacy add-on "$2.00 per CRPU-hour" additional.
AWS Entity Resolution (same page, section "AWS Entity Resolution on AWS Clean Rooms"): rule-based matching "$0.50 per 1,000 records matched" plus a "one-time $100.00 matching fee per collaboration" (fee basis rechecked 2026-10-05: per collaboration, not per matching run); provider-based matching "$0.10 per 1,000 records"; data preparation "$0.10 per 1,000 records".

Derived numbers.
- Per-record matching price (directly quoted, no throughput assumption): $0.10 to $0.50 per 1,000 records (provider/rule-based Entity Resolution). The paper's hypothetical $2 per 1,000 is 4x to 20x above these, so it is an upper-end figure, not a sourced one.
- Compute-based join: one Spark SQL query at the default 32 CRPUs for the 60 s minimum costs 32 x $2 / 60 = $1.07 (with the DP add-on $2.13). Per-1,000-row cost depends on rows per query; ASSUMPTION: a query joins and returns N rows. At N = 1,000 rows (a small incremental read batch billed at the minimum) it is $1.07 per 1,000 rows; at N = 1M rows it is $0.001 per 1,000. Reads are billed per query, so the number of checkpoint reads (not rows) drives cost in the small-batch regime. Report both ends or the per-record ER price.
- Caveat: Entity Resolution prices record linkage, not a billed "outcome read"; map as an analogue. LiveRamp, Habu, Snowflake, Google Ads Data Hub: pricing is quote-based/credit-based and not publicly stated per row [U]; not used.

## 2. Human-verified labels
- Mechanical Turk fees [V], https://www.mturk.com/pricing (page notes closure on 2026-09-30): "20% fee on the reward and bonus amount (if any) you pay Workers"; "HITs with 10 or more assignments will be charged an additional 20% fee"; "The minimum fee is $0.01 per assignment or bonus payment." Cost per label = reward x 1.2 (1.4 with 10+ assignments) + effective minimum. With a $0.05 reward and 3 assignments: 3 x (0.05 x 1.2) = $0.18 per verified item; with a $0.02 reward, 1 assignment: $0.024 + min fee rules.
- AWS Ground Truth blog [V], https://aws.amazon.com/blogs/machine-learning/annotate-data-for-less-with-amazon-sagemaker-ground-truth-and-automated-data-labeling/ : "$0.26 * 1000, which equals $260" for 1,000 bounding-box bird annotations without automated labeling; hence $0.26 per human-labelled object in that example (a complex task).
- Ground Truth Mechanical Turk rates [S, secondary, official page not retrievable]: Wring guide (https://wring.co/blog/aws-sagemaker-ground-truth-pricing-guide) lists "Image Classification: $0.012/image", "Text Classification: $0.012/unit", "Bounding Box: $0.036/object", attributing them to the AWS Ground Truth pricing page. Do not cite as primary. A $0.08/object first-tier AWS fee appeared in search snippets only [U].
- Google Cloud data labeling: units defined (text: 50 words x labelers; image: images x labelers) but per-unit dollars not retrievable [U].
- Derived: $0.10 per verified label sits between the $0.012 simple-classification crowd rate and the $0.26 complex-annotation rate, and equals roughly one 3-assignment MTurk item at $0.028 reward x 1.2. Honest framing: "illustrative, within the public $0.012 to $0.26 range".

## 3. Purchased conversion / incrementality measurement
No per-row public price found. Meta/Google conversion lift is free above spend thresholds (search summaries: about USD 30k per test window for Meta; Google via account rep) [S, aggregator blogs, unverified]; brand-lift studies quoted at USD 30k to 100k+ [S]. Not usable as a per-row price; recommend omitting a purchased-data row or describing it qualitatively. Google Ads help pages found: https://support.google.com/google-ads/answer/12003020 (About Conversion Lift) [not fetched for prices].

## 4. Per-record privacy accounting
- Feldman and Zrnic, "Individual Privacy Accounting via a Renyi Filter", arXiv:2008.11193 [V, abstract]: each participant has a personalized budget; standard accounting "is overly conservative, especially for 'typical' data points which incur little privacy loss". Supports the claim that an unread record spends none of its budget; contains no deployment epsilon.
- US Census 2020 [V], https://www.census.gov/.../2021-06-09.html : "ε=19.61, which includes ε=17.14 for the persons file and ε=2.47 for the housing unit data." (total privacy-loss budget for the redistricting product; a per-person budget over all releases, not per query).
- Apple [V partial], https://machinelearning.apple.com/research/learning-with-privacy-at-scale : emoji discovery and Safari resource detection ε = 4 per donation. Per-day totals (health ε=2, emoji 4, QuickType 8 with 2 donations/day) came from search snippets of Apple's Differential Privacy Overview PDF, which I could not parse [U].
- Derived: with a per-person lifetime budget ε_tot and a per-read cost ε_read, a record read k times spends k·ε_read. A sourced anchor: Apple ε = 4 per donation; Census ε_tot = 17.14 per person. Example row: if each outcome read spends ε = 0.1 of a per-person ε_tot = 4 (illustrative split, not sourced), then 40 reads exhaust a record. State the split as an assumption.

## Recommended 3 price points for the table
1. Clean-room: $0.50 per 1,000 records (AWS Entity Resolution rule-based, +$100 base), alternative $0.10 per 1,000 (provider-based); optionally also "$1.07 per query at minimum billing, Spark SQL 32 CRPUs x $2/CRPU-hour x 60 s". Cite awscleanrooms_pricing. Replace the $2 per 1,000.
2. Verified label: $0.10 retained as a mid-range figure but sourced as "between $0.012 (crowd classification; secondary source) and $0.26 (AWS blog complex annotation example); MTurk adds a 20% fee (40% for 10+ assignments)". Cite awsgroundtruth_blog and mturk_pricing. If only primary sources are allowed, use $0.26 (blog) as the high case and MTurk reward x 1.2 as the formula.
3. Privacy: per-record budget anchored on Census ε_tot = 17.14 per person (persons file) or Apple ε = 4 per donation; table row should state "ε_read = ε_tot / reads allowed" under an explicit assumption. Cite census2021dsep, apple2017dp, feldman2020individualarxiv.
Skip a purchased-data row (no public price).
