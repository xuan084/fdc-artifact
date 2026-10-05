# external reviewer review round 2 (final) — lock v9 addendum

threadId 01a10784-9966-7963-99c5-83ae7784a763; read-only sandbox; approval-policy never; 2026-10-05.

**Verdict: lock.** No blocking fixes or new P0/P1 issues found.

| Round-1 item | Status | Evidence |
|---|---|---|
| P0 — universal rectangle claim / population scope | Resolved | headline covers both registered rivals, conditional on the reused tables (addendum §3 joint headline, disclosure 9; builder `joint`) |
| P1 — PJC ledger asymmetry | Resolved | K77 cost and fresh-width trade-off disclosed; B requires NI against both PJC formulations (v9_analysis.py CONF_SPEC B `ni`) |
| P1 — method-specific schedules / cross-grid checks | Resolved | corrected disclosure; R2c checks HC's own digest and PJC's realised counts at common block points, and enters the verdict (v9_replica.check_cross_grid; run_r5s_v9.py:506) |
| P1 — dense monitoring is a design change | Resolved | unchanged width parameters distinguished from the new wrapper / stopping opportunities / primary design |
| P1 — inherited validation exceptions | Resolved | both exceptions carried forward |
| P2 — Hillstrom wording-helper validation | Partially resolved | empty input, invalid UBs and negative counts rejected; exact-set validation is optional (an arbitrary singleton passes without `rectangles`). Normal analysis supplies the registered set, so non-blocking. |
| P2 — conditional K−2 diagnostic | Resolved | selection disclosed; eligible/total and explicit unavailable implemented |

**TU-PJC is valid for the registered configuration.** With no resets the half-plan stays fixed and allocation uses counts only (pjc_bf.py:230, :241). Conditional on arrivals, counts are deterministic and nondecreasing, satisfying Lemma TU. On the block-point variance event the frozen widths dominate the deterministic true-MGF widths; the existing (q, π, k) union covers changing selected policies without another monitoring penalty. The code enforces the no-reset/local/half/no-rectangle configuration, builds a K20 ledger, updates boxes only at block points, and uses frozen counts/boxes with current centres between them (rect_tu_v9.py:129, :155).

Dev results agree with the disclosure, including B's governing TU-PJC UB 1.01555. Recomputed A/B replica checks pass. The invalid toy control's 0/200 is correctly disclosed as uninformative for ledger validity; not a blocker.

Read-only verification: 17 tests passed (file-writing test excluded). No files modified.
