"""H4 paired analysis: JPC savings vs baselines at kappa_high vs kappa=0 (same instance seeds, same noise seed).

Usage: analyze_h4_kappa_pair.py --mode {pilot,full}
Reads  exp/results/{pilots,full}/nl_reuse_kappa_{high,zero}/{paired.json,summary.json,results.jsonl}
Writes exp/results/{pilots,full}/nl_reuse_kappa_zero/h4_paired.json and h4_bar.png

Saving statistic per kappa level (log scale, < 0 means JPC needs fewer new env steps):
  S_kappa(b) = log( sum_problems JPC_steps / sum_problems b_steps )   (stream totals, instance = cluster)
Saving difference (H4): D(b) = S_zero(b) - S_high(b); D > 0  <=> savings larger under kappa_high.
Paired instance-level bootstrap: the same resampled instance ids are used for both kappa levels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

BASES = ("B1", "B2", "B3", "B3g", "ORACLE_DDA", "best_B123_per_problem")


def load(d):
    pp = json.loads((d / "paired.json").read_text())["per_problem"]
    sm = json.loads((d / "summary.json").read_text())
    return pp, sm


def per_instance(pp, b):
    out = {}
    for r in pp:
        if b in r and r.get(b) is not None:
            out.setdefault(r["instance"], []).append((r["JPC"], r[b]))
    return {k: np.array(v, float) for k, v in out.items()}


def log_total(arrs):
    a = np.concatenate(arrs)
    return math.log(max(a[:, 0].sum(), 1.0) / max(a[:, 1].sum(), 1.0))


def pp_median(arrs, add=1.0):
    a = np.concatenate(arrs)
    return float(np.median((a[:, 0] + add) / (a[:, 1] + add)))


def nontrivial_median(arrs):
    a = np.concatenate(arrs)
    m = np.maximum(a[:, 0], a[:, 1]) > 0
    if not m.any():
        return float("nan")
    return float(np.median(a[m, 0] / np.maximum(a[m, 1], 1.0)))


def paired_boot(hi, ze, fn, B=4000, seed=7, diff="sub"):
    keys = sorted(set(hi) & set(ze))
    rng = np.random.default_rng(seed)

    def stat(ks):
        h, z = fn([hi[k] for k in ks]), fn([ze[k] for k in ks])
        return z - h
    est = stat(keys)
    boots = np.array([stat([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(B)])
    boots = boots[np.isfinite(boots)]
    lo, up = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(est), "ci95": [float(lo), float(up)], "n_instances": len(keys),
            "frac_boot_gt_0": float(np.mean(boots > 0))}


def single_boot(d, fn, B=4000, seed=7):
    keys = sorted(d)
    rng = np.random.default_rng(seed)
    est = fn([d[k] for k in keys])
    boots = np.array([fn([d[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(B)])
    boots = boots[np.isfinite(boots)]
    lo, up = np.quantile(boots, [0.025, 0.975])
    return float(est), [float(lo), float(up)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    args = ap.parse_args()
    root = WS / "exp" / "results" / ("pilots" if args.mode == "pilot" else "full")
    hp, hs = load(root / "nl_reuse_kappa_high")
    zp, zs = load(root / "nl_reuse_kappa_zero")
    out = {"mode": args.mode, "instances_high": hs["instance_seeds"], "instances_zero": zs["instance_seeds"],
           "noise_seeds": [hs["noise_seeds"], zs["noise_seeds"]],
           "definition": "S_kappa(b)=log(sum JPC / sum b) over the stream (new env steps); D(b)=S_zero-S_high; "
                         "D>0 means JPC's saving is larger under kappa_high. Paired instance bootstrap (B=4000).",
           "per_baseline": {}}
    for b in BASES:
        hi, ze = per_instance(hp, b), per_instance(zp, b)
        if not hi or not ze:
            continue
        sh, ch = single_boot(hi, log_total)
        sz, cz = single_boot(ze, log_total)
        out["per_baseline"][b] = {
            "stream_ratio_high": math.exp(sh), "stream_ratio_high_ci95": [math.exp(x) for x in ch],
            "stream_ratio_zero": math.exp(sz), "stream_ratio_zero_ci95": [math.exp(x) for x in cz],
            "D_log_stream_ratio": paired_boot(hi, ze, log_total),
            "pp_median_ratio_plus1_high": pp_median(list(hi.values())),
            "pp_median_ratio_plus1_zero": pp_median(list(ze.values())),
            "D_pp_median_plus1": paired_boot(hi, ze, pp_median),
            "nontrivial_median_high": nontrivial_median(list(hi.values())),
            "nontrivial_median_zero": nontrivial_median(list(ze.values())),
        }
    # RMST-based ratio vs best baseline (by RMST at each level's own calibrated T_max; and at the common step cap)
    def rmst_block(s):
        m = s["methods"]
        rb = {b: m[b]["rmst_new_env_steps"] for b in ("B1", "B2", "B3") if b in m}
        best = min(rb, key=rb.get)
        cap = s["extended"]["rmst_at_step_cap"]
        capb = {b: cap[b]["rmst"] for b in ("B1", "B2", "B3") if b in cap}
        bestc = min(capb, key=capb.get)
        return {"T_max_common": s["T_max_common"], "best_baseline": best,
                "rmst_ratio_vs_best": m["JPC"]["rmst_new_env_steps"] / max(rb[best], 1e-9),
                "step_cap_tau": cap["tau"], "best_baseline_at_cap": bestc,
                "rmst_ratio_vs_best_at_cap": cap["JPC"]["rmst"] / max(capb[bestc], 1e-9),
                "jpc_completion": m["JPC"]["completion_rate"], "jpc_false_cert": m["JPC"]["n_false_cert"],
                "jpc_fcr_cp_upper": m["JPC"]["fcr_cp_upper"],
                "completion": {k: v["completion_rate"] for k, v in m.items()},
                "false_cert": {k: v["n_false_cert"] for k, v in m.items()},
                "fcr_cp_upper": {k: v["fcr_cp_upper"] for k, v in m.items()},
                "rmst": {k: v["rmst_new_env_steps"] for k, v in m.items()},
                "jpc_zero_step_problems": None}
    out["rmst"] = {"high": rmst_block(hs), "zero": rmst_block(zs)}
    for lvl, pp in (("high", hp), ("zero", zp)):
        out["rmst"][lvl]["jpc_zero_step_problems"] = int(sum(r["JPC"] == 0 for r in pp))
        out["rmst"][lvl]["ties_both_zero_vs_best_B123"] = int(sum(r["JPC"] == 0 and r.get("best_B123_per_problem") == 0
                                                                for r in pp))
    # pre-registered pilot pass criterion
    d_main = out["per_baseline"]["best_B123_per_problem"]
    lit_high = out["per_baseline"]["best_B123_per_problem"]["pp_median_ratio_plus1_high"]
    lit_zero = out["per_baseline"]["best_B123_per_problem"]["pp_median_ratio_plus1_zero"]
    b3 = out["per_baseline"].get("B3", {})
    out["gate"] = {
        "literal_pp_median_zero_closer_to_1_than_high": bool(abs(1 - lit_zero) < abs(1 - lit_high)),
        "literal_values": {"high": lit_high, "zero": lit_zero},
        "stream_total_D_vs_best_B123_gt_0": bool(d_main["D_log_stream_ratio"]["estimate"] > 0),
        "stream_total_D_vs_B3_gt_0": bool(b3.get("D_log_stream_ratio", {}).get("estimate", -1) > 0),
        "jpc_false_cert_zero_eq_0": bool(out["rmst"]["zero"]["jpc_false_cert"] == 0),
    }
    out["H4_supported_pilot"] = bool(out["gate"]["stream_total_D_vs_best_B123_gt_0"] and
                                     out["gate"]["jpc_false_cert_zero_eq_0"])
    out["H4_ci_excludes_0_vs_best_B123"] = bool(d_main["D_log_stream_ratio"]["ci95"][0] > 0)
    (root / "nl_reuse_kappa_zero" / "h4_paired.json").write_text(json.dumps(out, indent=1, default=float))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bs = [b for b in ("B3", "B3g", "B2", "B1", "ORACLE_DDA") if b in out["per_baseline"]]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        w = 0.38
        x = np.arange(len(bs))
        for j, (lvl, col) in enumerate((("high", "#3b6fb6"), ("zero", "#c8763a"))):
            v = [out["per_baseline"][b][f"stream_ratio_{lvl}"] for b in bs]
            ci = np.array([out["per_baseline"][b][f"stream_ratio_{lvl}_ci95"] for b in bs])
            ax.bar(x + (j - 0.5) * w, v, w, color=col, label=f"kappa={lvl}",
                   yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], capsize=3)
        ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"JPC/{b}" for b in bs])
        ax.set_ylabel("stream-total new-step ratio (log)")
        ax.set_title(f"H4 ({args.mode}): JPC saving at kappa_high vs kappa=0 (instance bootstrap 95% CI)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(root / "nl_reuse_kappa_zero" / "h4_bar.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        out["plot_error"] = repr(e)
    print(json.dumps({"gate": out["gate"], "H4_supported_pilot": out["H4_supported_pilot"],
                      "H4_ci_excludes_0_vs_best_B123": out["H4_ci_excludes_0_vs_best_B123"],
                      "D_best": d_main["D_log_stream_ratio"], "D_B3": b3.get("D_log_stream_ratio"),
                      "rmst": {k: {kk: out["rmst"][k][kk] for kk in ("best_baseline", "rmst_ratio_vs_best",
                                                                     "rmst_ratio_vs_best_at_cap", "jpc_completion",
                                                                     "jpc_false_cert", "jpc_zero_step_problems")}
                               for k in ("high", "zero")}}, indent=1, default=float))


if __name__ == "__main__":
    main()
