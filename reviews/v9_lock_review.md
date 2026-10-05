# external reviewer review round 1 — lock v9 addendum (DRAFT)

threadId 01a105ce-acc0-7360-96b3-4c4a587e9865; read-only sandbox; approval-policy never. Recovered verbatim from the previous agent's transcript (2026-10-04).

**Overall verdict: fix before lock.** I found no validity defect in RECT-ck-BF-TU under the stated frozen-schedule assumptions. The main issues are claim scope, incomplete disclosure of comparison asymmetries, and narrower integrity checks than the prose suggests.

| Severity | File:line | Finding | Concrete fix |
|---|---|---|---|
| **P0** | [prereg_lock_v9_addendum_DRAFT.md:53](plan/prereg_lock_v9_addendum_DRAFT.md:53); [build_v9_addendum_draft.py:197](exp/code/build_v9_addendum_draft.py:197) | The registered headline says **“every time-uniform rectangle”**, but tests only two specified constructions/configurations. Fresh streams also provide inference conditional on the reused tables, not independent replication across populations. | Register: “Lower paired geometric-mean N80_pen than both registered time-uniform rectangle rivals, on fresh streams of each reused finite table.” Explicitly state the conditional scope of confirmation. |
| **P1** | [prereg_lock_v9_addendum_DRAFT.md:24](plan/prereg_lock_v9_addendum_DRAFT.md:24); [pjc_bf.py:87](exp/code/dsswm/baselines/pjc_bf.py:87) | PJC@K77 is valid, but its ledger pays an additional **ln(77/20) ≈ 1.348** in β and uses smaller variance-box tail levels. Its fresh widths can offset that cost; the net effect is not predetermined. Crucially, the selected no-reset, count-only PJC also qualifies for a Lemma-TU extension. Thus NI against PJC@K77 is not NI against the strongest available PJC formulation. | Disclose this asymmetry in §0 and qualify the NI wording as against **the frozen PJC-local configuration with its K77 ledger**. Prefer adding a predeclared TU-PJC sensitivity comparison; retain the existing matched K20 comparison. |
| **P1** | [prereg_lock_v9_addendum_DRAFT.md:10](plan/prereg_lock_v9_addendum_DRAFT.md:10); [run_r5s_v9.py:496](exp/code/run_r5s_v9.py:496) | “Every compared method … uses that frozen schedule” is inaccurate: X5 HC uses Neyman allocation, and PJC uses deterministic tracking. Cross-task R2 checks the registered 50/50 group only. I reproduced passing task checks, cross-R2 and cross-R2b with different X5 HC schedule digests on D and K. This demonstrates a detection gap, not an observed real-run mismatch. | Say **outcome-free, method-specific allocation schedules**, with shared arrivals/pool orders. Add cross-grid R2 for HC’s own Neyman schedule. For PJC, verify actual count/allocation paths at common times; its adaptive-mode schedule digest does not encode chosen arms. |
| **P1** | [prereg_lock_v9_addendum_DRAFT.md:9](plan/prereg_lock_v9_addendum_DRAFT.md:9) | The theoretical reinterpretation leaves K20 FDC-BF unchanged, but the confirmatory procedure now uses a new wrapper and 77 monitoring times, which can change stopping and sticky answers. “Not a design change” needs this distinction. | Explicitly disclose that the former secondary dense-grid experiment becomes the v9 primary design; inherited width parameters remain unchanged, and numerical identity holds when evaluated only on K20. |
| **P1** | [prereg_lock_v9_addendum_DRAFT.md:5](plan/prereg_lock_v9_addendum_DRAFT.md:5); [prereg_lock_v8_addendum.md:14](plan/prereg_lock_v8_addendum.md:14) | The blanket “every file they bind [is] untouched” omits inherited exceptions: v8 accepts the recorded v7 input erratum, and the cascade does not revalidate v5’s input manifest. V8 explicitly disclosed both. | Carry those disclosures into §0. Say v9 introduces no additional drift and inherits v8’s documented validation scope. |
| **P2** | [v9_analysis.py:184](exp/code/dsswm/stats/v9_analysis.py:184) | `word_block_c({}, 0)` returns `faster_descriptive`; a negative-infinite UB also passes. The normal `analyse_c` path supplies both registered rivals, so this is a helper-validation loophole rather than a demonstrated runner bypass. | Require the exact task-specific rectangle set, finite positive UBs, and a valid nonnegative integer false-stream count. Add rejection tests. |
| **P2** | [v9_analysis.py:211](exp/code/dsswm/stats/v9_analysis.py:211) | The K−2 diagnostic is conditional on both trajectories reaching that checkpoint, having positive finite running-minimum bounds, and retaining at least ten pairs. It is not an unconditional “uncensored” width comparison and can disappear entirely. | Register that selection explicitly; report eligible/total counts and an explicit unavailable result. Rename it a running-certificate-bound ratio, or compute the intended widths on continued diagnostic trajectories. |

RECT-ck-BF-TU’s construction is sound: on the block-point variance event, each frozen radius dominates the deterministic true-MGF Chernoff radius. Lemma TU then supplies simultaneous coverage after that block point. Current-centre intervals, deterministic bounds, the frozen HG box, and running intersections preserve coverage. Empty-at-block cells retain infinite stochastic radii; exhaustion between blocks yields exact deterministic bounds. Updating the HG box only at block points is correct.

The primary rectangle comparison is defensible: identical monitoring opportunities, matched Bennett construction for RECT, and previously frozen HC configurations. PJC@K77 is also a reasonable **specified baseline**, provided the narrower interpretation above is registered. Comparing TU-FDC@D directly against PJC@K20 would introduce unequal monitoring opportunities; a same-grid TU-PJC sensitivity is more informative.

`decide_confirmatory` and `analyse_confirmatory` implement the stated strict thresholds, B’s NI margin, primary method/epsilon, whole-run false-stream condition, and replica requirement correctly. Missing, duplicate, wrong-epsilon and wrong-primary rows are rejected. The runner incorporates all three task statuses and both cross-task checks.

The v9→v8 cascade, X5/Hillstrom gate signatures and prefixes, Hillstrom binding, and inherited seal verification are correctly connected. The live v8 cascade passed, and Hillstrom’s hash matched. Final v9 lock/seal validation remains outstanding.

Read-only verification: **13 v9 addendum tests and 8 candidate tests passed**; the one file-writing test was excluded. No files were changed.

---

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
