# Can external reviewer's route to 8 (a prospective scaling study on unused data) be run on OBD? (2026-10-05, dev only)

**external reviewer's route to 8 (r6 and r7 reviews).** A prospectively locked study with the following design:
- S = 16, 32 and 64 segments;
- three development-selected tolerances, held common across S;
- 200 paired streams per setting, on an evaluation population that is unused at its own lock;
- dense monitoring, with schemes (a) and (b);
- full-frontier reporting.

**Candidate populations.** The only unused randomized populations left are the OBD `random/men` (about 453k rows) and `random/women` (about 865k rows) campaigns. Both are still unread. The `random/all` evaluation half is exposed by lock v11.

**Check, on the `random/all` dev half (already open; no new outcome read).** User features yield 401 distinct tuples. Ranked by size:

| S | Rows in the S-th largest tuple | Share of rows in the top S − 1 |
|---|---|---|
| 16 | 5,915 | 70% |
| 32 | 3,629 | 81% |
| 64 | 1,226 | 91% |

**Click counts per cell.** The base CTR is 0.35% and each arm gets half of a segment's rows.
- At S = 64 the smallest segments have about 2 clicks per segment × arm cell per half.
- With equal-size score-quantile segments on the men campaign (about 226k rows per half), it is about 6 clicks per cell at S = 64 and about 25 at S = 16.

**Consequence.** Any tolerance at which a 15-budget frontier could be certified within the table would be of the order of the base rate itself. At such a tolerance most policies are ε-optimal, so the frontier is near-trivial (compare combo16 in `plan/obd_feasibility.md`, where 83% of policies are ε-optimal at ε = 1e-3).

**Verdict: NO-GO for a large-S scaling study on OBD.** The large-class (S = 16–64) evidence stays on the reused X5 and Lenta tables, as the paper already discloses. A second nine-segment replication on `random/men` or `random/women` is feasible. It would add breadth (the critic's optional P1), but not the scaling evidence external reviewer asks for. The score-7 consensus does not depend on either study.
