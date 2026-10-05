"""Independent Tier-2 replica of E1-Lin and E1-NL-S ground truth, exact J and (full mode) B1.

Written from the text of plan/methodology.md sections 2-3 only. This package must NOT import
`dsswm`; it receives problems as plain data plus an opaque policy callable
`policy(t, loads, engaged) -> (pairs, incentives)` supplied by the harness bridge.
"""
