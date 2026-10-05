"""Lock-v6 addendum: pre-registered analysis functions (new file; frozen with the addendum).

Same paired statistics as the v5 aggregate (``run_r5_analysis_aggregate.paired``): per-stream log ratio
d_i = log N(FDC, i) - log N(r, i); point estimate exp(mean d); paired percentile bootstrap over streams (B = 10^4,
seed 42, the SAME resample matrix for every comparator of a block); one-sided 95% UB = exp(q_0.95 of the bootstrap
means); two-sided 95% CI.  The percentile bootstrap is an approximate procedure; the conjunctive (IUT) rules need no
within-family multiplicity correction, but the per-rival wordings do not jointly inherit a 95% guarantee.

Input validation (v6 review item 2): ``build_matrix`` refuses missing, extra or duplicate (method, eps, seed) rows,
non-finite or non-positive endpoints, and any method set other than the registered one; ``decide_block_a`` /
``decide_block_c`` refuse a per-rival dict that is not exactly the registered family, non-finite bounds, or a replica
status other than 'pass' / 'fail'.  A failed replica forces the conservative verdict.

Decision rules (numbers fixed by the authors before any v6 run):
Block A (registered same-strength rectangular rivals; endpoint N80_pen):
  UB* = max over the registered family of the one-sided 95% UB of FDC/r
    UB* < 0.70          -> 'keep'       the pre-registered evidence threshold for a certificate advantage over the
                                        REGISTERED rivals/configs is met (supports the tested magnitude only)
    0.70 <= UB* < 1.0   -> 'downgrade'  FDC faster than every registered rival but the pre-registered evidence
                                        threshold is not met; report every point estimate and CI as is
    UB* >= 1.0          -> 'withdraw'   no certificate-superiority claim over the registered rivals
    replica 'fail'      -> 'withdraw'   (forced)
Block C (N100 continuation, the nine v5 rivals R; endpoint N100_pen):
  'n100_pass' iff UB_r < 1 for every r in R; else 'narrow_to_N80'; replica 'fail' -> 'narrow_to_N80' (forced).
Block B (Lenta LR9): descriptive only; replica 'fail' -> table published only with the flag 'replica_fail' in the
  appendix and not cited in the main text.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = ["boot_idx", "paired", "decide_block_a", "decide_block_c", "cp_upper", "build_matrix", "analyse_block",
           "BLOCK_SPEC", "B_BOOT", "BOOT_SEED", "A_KEEP", "A_WITHDRAW", "V5_R", "A_FAMILY"]

B_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05
A_KEEP = 0.70
A_WITHDRAW = 1.0
V5_R = ("B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool")
A_FAMILY = ("RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR")
B_METHODS = ("FDC", "B4-bal", "QFC-pool", "B2-rect", "Peace-rect", "Hait-SW", "RECT-ck-HG", "RECT-ck-HG-live",
             "RECT-ck-Bern", "HC-WoR")
BLOCK_SPEC = {
    "A": {"methods": ("FDC", "B4-bal") + A_FAMILY, "family": A_FAMILY, "seeds": tuple(range(31000, 31200)),
          "eps": (0.001,), "key": "N80_pen"},
    "B": {"methods": B_METHODS, "family": None, "seeds": tuple(range(32000, 32200)), "eps": (0.002, 0.003, 0.004),
          "key": "N80_pen"},
    "C": {"methods": ("FDC",) + V5_R, "family": V5_R, "seeds": tuple(range(30000, 30200)), "eps": (0.001,),
          "key": "N100_pen"},
}


def cp_upper(k, n, alpha=ALPHA):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def boot_idx(n, B=B_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def paired(a, b, idx):
    """a = FDC, b = comparator (same streams, same order); ratio a / b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired: arrays must be 1-d, equal length and finite")
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("paired: endpoints must be positive")
    d = np.log(a) - np.log(b)
    boot = d[idx].mean(1)
    return {"geomean_ratio": float(np.exp(d.mean())),
            "ub95_one_sided": float(np.exp(np.quantile(boot, 0.95))),
            "ci95_two_sided": [float(np.exp(np.quantile(boot, 0.025))), float(np.exp(np.quantile(boot, 0.975)))],
            "frac_faster": float((d < 0).mean()), "frac_tied": float((d == 0).mean()),
            "frac_slower": float((d > 0).mean()), "mean_log_ratio": float(d.mean()),
            "sd_log_ratio": float(d.std(ddof=1)) if len(d) > 1 else None, "n": int(len(d))}


def build_matrix(rows, methods, seeds, eps, key):
    """{(method, eps): array over seeds}.  Refuses missing / extra / duplicate rows and invalid endpoints."""
    seeds = list(seeds)
    want = {(m, float(e)) for m in methods for e in eps}
    got = {}
    for r in rows:
        k = (r["method"], float(r["eps"]))
        if k not in want:
            raise ValueError(f"unregistered (method, eps) in rows: {k}")
        if int(r["seed"]) not in set(seeds):
            raise ValueError(f"unregistered seed {r['seed']} for {k}")
        d = got.setdefault(k, {})
        if int(r["seed"]) in d:
            raise ValueError(f"duplicate row {k} seed {r['seed']}")
        v = r.get(key)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"invalid endpoint {key}={v!r} for {k} seed {r['seed']}")
        d[int(r["seed"])] = float(v)
    for k in sorted(want):
        if k not in got or len(got[k]) != len(seeds):
            raise ValueError(f"incomplete: {k} has {len(got.get(k, {}))}/{len(seeds)} seeds")
    return {k: np.array([got[k][s] for s in seeds]) for k in want}


def _replica_status(replica):
    st = (replica or {}).get("status")
    if st not in ("pass", "fail"):
        raise ValueError(f"replica check not completed (status={st!r}); no decision without a finished replica")
    return st


def _family_ub(per_rival, family):
    if family is None or set(per_rival) != set(family):
        raise ValueError(f"per-rival results {sorted(per_rival)} != registered family {sorted(family or [])}")
    ub = {}
    for k, v in per_rival.items():
        u = v.get("ub95_one_sided")
        if u is None or not math.isfinite(u) or u <= 0:
            raise ValueError(f"invalid UB for {k}: {u!r}")
        ub[k] = float(u)
    return ub


def decide_block_a(per_rival: dict, replica: dict, family=A_FAMILY) -> dict:
    ub = _family_ub(per_rival, family)
    st = _replica_status(replica)
    worst = max(ub, key=ub.get)
    u = ub[worst]
    verdict = "keep" if u < A_KEEP else ("downgrade" if u < A_WITHDRAW else "withdraw")
    if st == "fail":
        verdict = "withdraw"
    return {"UB_star": u, "governing_rival": worst, "verdict": verdict, "replica_status": st, "per_rival_ub": ub,
            "thresholds": {"keep": "UB* < 0.70", "downgrade": "0.70 <= UB* < 1.0", "withdraw": "UB* >= 1.0"},
            "scope": "registered rivals and frozen configs only"}


def decide_block_c(per_rival: dict, replica: dict, family=V5_R) -> dict:
    ub = _family_ub(per_rival, family)
    st = _replica_status(replica)
    ok = all(u < 1.0 for u in ub.values()) and st == "pass"
    return {"verdict": "n100_pass" if ok else "narrow_to_N80", "replica_status": st, "per_rival_ub": ub,
            "failing": sorted(k for k, u in ub.items() if u >= 1.0)}


def analyse_block(block, rows, replica, spec=None, B=B_BOOT):
    """Validated analysis of one block. ``spec`` defaults to BLOCK_SPEC[block] (dev tests may pass another)."""
    sp = dict(BLOCK_SPEC[block] if spec is None else spec)
    M = build_matrix(rows, sp["methods"], sp["seeds"], sp["eps"], sp["key"])
    idx = boot_idx(len(sp["seeds"]), B=B)
    out = {"block": block, "key": sp["key"], "n_streams": len(sp["seeds"]), "per_eps": {}}
    for e in sp["eps"]:
        fdc = M[("FDC", float(e))]
        out["per_eps"][str(e)] = {m: paired(fdc, M[(m, float(e))], idx) for m in sp["methods"] if m != "FDC"}
    first = out["per_eps"][str(sp["eps"][0])]
    if block == "A":
        out["decision"] = decide_block_a({m: first[m] for m in sp["family"]}, replica, sp["family"])
    elif block == "C":
        out["decision"] = decide_block_c({m: first[m] for m in sp["family"]}, replica, sp["family"])
    else:
        out["decision"] = {"verdict": "descriptive" if _replica_status(replica) == "pass" else "replica_fail"}
    return out
