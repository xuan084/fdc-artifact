"""r5_analysis_aggregate: read-only aggregation of the round-5 blocks under the lock v5 SAP.

  python run_r5_analysis_aggregate.py --mode pilot   # dev data (seeds 900-999), validates every code path
  python run_r5_analysis_aggregate.py --mode full    # eval blocks (seeds 30000+), confirmatory

Reads raw results.jsonl only (no rerun, no stream exclusion). Computes
  C1   per-rival paired geomean ratio FDC/r, one-sided 95% UB (paired percentile bootstrap, B = 1e4, seed 42),
       IUT, Holm-adjusted bootstrap p-values, Bonferroni UB, 0.90 wording gate;
  C2   false streams + one-sided 95% Clopper-Pearson UB in both windows (stop 12/15; full horizon);
  C4   2x2x2 factorial: coded effects, Shapley over {D,U,L}, design share >= 2/3, FDC vs B4-bal UB,
       frozen-prediction |log error| <= 0.095 (else descriptive);
  E-cost (two-sided CI, no pass line), K = 60 sensitivity, CR12 support, practical significance;
  verdict = 'positive_result_achieved' iff C1 IUT passes AND C2 passes (both windows), else 'not_achieved'.
Outputs: summary.json, claims.json, table1.csv/.md, table2.md, fig1-5 (PDF + PNG).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parent.parent  # .../current
RES = WS / "exp" / "results"
TASK = "r5_analysis_aggregate"

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams import r5_registry as reg  # noqa: E402
from dsswm.stats import prereg  # noqa: E402

B_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05
DELTA = 0.05
R_SET = list(reg.RIGOROUS_SET_R)
CELLS = [("pool", "qstar", "r4"), ("pool", "qstar", "r5"), ("pool", "feas", "r4"), ("pool", "feas", "r5"),
         ("half", "qstar", "r4"), ("half", "qstar", "r5"), ("half", "feas", "r4"), ("half", "feas", "r5")]
LEVEL = {"D": ("pool", "half"), "U": ("qstar", "feas"), "L": ("r4", "r5")}
CR_BUDGETS = [round(0.10 + 0.05 * i, 2) for i in range(15)]
# Okabe-Ito colour-blind safe palette
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
LS = ["-", "--", "-.", ":"]


def cell_name(c):
    return f"F[{c[0]},{c[1]},{c[2]}]"


# --------------------------------------------------------------------------------------------- sources
def sources(mode):
    P, F = RES / "pilots", RES / "full"
    if mode == "pilot":
        return {
            "main": [P / "r5_t2_gate_strict/results.jsonl"],
            "main_schema_check": [P / "r5_cr_main_a/results.jsonl", P / "r5_cr_main_b/results.jsonl"],
            "plugin": [P / "r5_t2_gate_plugin_t3/results.jsonl"],
            "plugin_schema_check": [P / "r5_cr_plugin_a/results.jsonl", P / "r5_cr_plugin_b/results.jsonl"],
            # dev C2 full window: the 100-stream T3 full-horizon run (adapter); the 50-stream r5_cr_fwer_audit dev
            # pilot (same schema as the eval block) is cross-checked against it stream by stream
            "fwer_audit": [P / "r5_t2_gate_plugin_t3/fullhorizon.jsonl"], "fwer_audit_adapter": True,
            "fwer_audit_xcheck": [P / "r5_cr_fwer_audit/results.jsonl"],
            "factorial": [P / "r5_t1_reconcile/results.jsonl"],
            "factorial_schema_check": [P / "r5_cr_factorial/results.jsonl"],
            "predictions": P / "r5_cr_factorial/pred_dryrun/predictions.json",
            "k60": [P / "r5_cr_k60_sens/results.jsonl"],
            "cr12": [P / "r5_cr12_scale/results.jsonl"],
            "cr12_summary": P / "r5_cr12_scale/summary.json",
            "replica": P / "r5_replica_check/summary.json",
            "seeds_main": list(range(900, 1000)), "half": "dev",
            "tau_R": 6989793, "tau_R_cr12": 6989793,
            "out": P / TASK,
        }
    return {
        "main": [F / "r5_cr_main_a/results.jsonl", F / "r5_cr_main_b/results.jsonl"],
        "plugin": [F / "r5_cr_plugin_a/results.jsonl", F / "r5_cr_plugin_b/results.jsonl"],
        "fwer_audit": [F / "r5_cr_fwer_audit/results.jsonl"],
        "factorial": [F / "r5_cr_factorial/results.jsonl"],
        "predictions": F / "r5_cr_factorial/predictions.json",
        "k60": [F / "r5_cr_k60_sens/results.jsonl"],
        "cr12": [F / "r5_cr12_scale/results.jsonl"],
        "cr12_summary": F / "r5_cr12_scale/summary.json",
        "replica": F / "r5_replica_check/summary.json",
        "seeds_main": list(range(30000, 30200)), "half": "eval",
        "tau_R": 6989799, "tau_R_cr12": 6989799,
        "out": F / TASK,
    }


# --------------------------------------------------------------------------------------------- loading
T1_CFG = {"F[pool,qstar,r4]": cell_name(CELLS[0]), "F[pool,qstar,r5]": cell_name(CELLS[1]),
          "F[pool,feas,r4]": cell_name(CELLS[2]), "F[pool,feas,r5]": cell_name(CELLS[3]),
          "L4b QFC-half@K20 = FDCx[half,qstar,r4]": cell_name(CELLS[4]), "F[half,qstar,r5]": cell_name(CELLS[5]),
          "F[half,feas,r4]": cell_name(CELLS[6]), "FDC": cell_name(CELLS[7]), "B4-bal[share=0.50]": "B4-bal"}


def norm(r):
    """Normalise the four on-disk row schemas (flat / rs_-prefixed / run_stream-nested / plugin-T3 N_*)."""
    rs = r.get("run_stream") or {}

    def g(k, *alts, default=None):
        for key in (k, *alts):
            if key in r and r[key] is not None:
                return r[key]
            if key in rs and rs[key] is not None:
                return rs[key]
            if "rs_" + key in r and r["rs_" + key] is not None:
                return r["rs_" + key]
        return default
    billing = r.get("billing") or {}
    return {
        "method": r["method"], "seed": int(r["seed"]),
        "N80_pen": float(g("N80_pen", "N_pen")), "N80_raw": float(g("N80_raw", "N_raw", "N80_pen", "N_pen")),
        "completed": bool(g("completed", "reached_stop")), "fwer_event": bool(g("fwer_event")),
        "n_false": int(g("n_false", default=0) or 0), "n_cert": g("n_cert"),
        "cert_k": list(g("cert_k", default=[]) or []), "n_cert_curve": g("n_cert_curve"),
        "k_stop": g("k_stop"), "sec": g("sec", "sec_total"), "sec_cert": g("sec_cert"),
        "validity": g("validity"), "billing_ok": g("billing_ok", default=billing.get("billing_ok", None)),
        "error": r.get("error"),
        "N100_pen": r.get("N100_pen"), "N100_raw": r.get("N100_raw"), "fwer_event_full": r.get("fwer_event_full"),
        "fwer_event_by_12of15": r.get("fwer_event_by_12of15"), "completed_15": r.get("completed_15"), "cert_ms_per_checkpoint": r.get("cert_ms_per_checkpoint"),
    }


def read_rows(paths, filt=None, rename=None):
    out = {}
    dup = 0
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(str(p))
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if filt and not filt(r):
                continue
            if rename is not None:
                key = r.get("config", r.get("method"))
                if key not in rename:
                    continue
                r = dict(r)
                r["method"] = rename[key]
            n = norm(r)
            k = (n["method"], n["seed"])
            if k in out:
                dup += 1
                if out[k]["N80_pen"] != n["N80_pen"]:
                    raise RuntimeError(f"conflicting duplicate row {k} in {paths}")
            out[k] = n
    return out, dup


def t3_fullhorizon_adapter(paths, ck, tau):
    """Dev-only: map r5_t2_gate_plugin_t3/fullhorizon.jsonl (stop_k = 15) onto the r5_cr_fwer_audit fields."""
    out = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            k12 = r.get("k12_in_run")
            n80 = tau if (k12 is None or k12 < 0 or r["false_certs_by_k12"] > 0) else ck[k12]
            out[(r["method"], int(r["seed"]))] = {
                "method": r["method"], "seed": int(r["seed"]), "N100_pen": float(r["N_pen"]), "N100_raw": float(r["N_raw"]),
                "completed_15": bool(r["completed"]), "fwer_event_full": bool(r["fwer_event"]),
                "fwer_event_by_12of15": bool(r["false_certs_by_k12"] > 0), "N80_pen": float(n80),
                "n_false": int(r["n_false"]), "n_cert": r["n_cert"], "fwer_event": bool(r["fwer_event"])}
    return out


def arr(rows, m, seeds, key="N80_pen"):
    miss = [s for s in seeds if (m, s) not in rows]
    if miss:
        raise RuntimeError(f"missing rows for {m}: {len(miss)} seeds (first {miss[:5]}); no exclusion allowed")
    return np.array([rows[(m, s)][key] for s in seeds], dtype=float)


# --------------------------------------------------------------------------------------------- statistics
def cp_upper(k, n, alpha=ALPHA):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def boot_idx(n, B=B_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def paired(a, b, idx, bonf_m=None):
    """a = FDC, b = comparator; ratio a/b. Same idx for every comparator (paired by stream)."""
    d = np.log(a) - np.log(b)
    boot = d[idx].mean(1)
    out = {"geomean_ratio": float(np.exp(d.mean())),
           "ub95_one_sided": float(np.exp(np.quantile(boot, 0.95))),
           "ci95_two_sided": [float(np.exp(np.quantile(boot, 0.025))), float(np.exp(np.quantile(boot, 0.975)))],
           "frac_faster": float((d < 0).mean()), "frac_tied": float((d == 0).mean()),
           "frac_slower": float((d > 0).mean()), "mean_log_ratio": float(d.mean()),
           "sd_log_ratio": float(d.std(ddof=1)) if len(d) > 1 else None, "n": int(len(d)),
           "p_boot_one_sided": float((1 + (boot >= 0).sum()) / (len(boot) + 1))}
    if bonf_m:
        out["ub_bonferroni"] = float(np.exp(np.quantile(boot, 1 - ALPHA / bonf_m)))
    return out


def holm(pvals, alpha=ALPHA):
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    adj, running, rejected, stop = {}, 0.0, {}, False
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
        if not stop and pvals[k] <= alpha / (m - i):
            rejected[k] = True
        else:
            stop = True
            rejected[k] = False
    return adj, rejected


def lattice_index(vals, ck):
    pos = {int(x): i for i, x in enumerate(ck)}
    return np.array([pos.get(int(round(v)), np.nan) for v in vals], dtype=float)


def shapley_stats(g):
    """g: dict cell -> mean log N80. Returns total log speed-up (FDC - QFC-pool, negative = faster), phi, effects."""
    fs = ("D", "U", "L")

    def v(S):
        return g[tuple(LEVEL[f][1 if f in S else 0] for f in fs)]
    base = v(set())
    tot = v(set(fs)) - base
    phi = {}
    for f in fs:
        others = [h for h in fs if h != f]
        s = 0.0
        for r in range(3):
            for sub in __import__("itertools").combinations(others, r):
                w = math.factorial(r) * math.factorial(2 - r) / 6
                s += w * (v(set(sub) | {f}) - v(set(sub)))
        phi[f] = s
    x = {c: [1 if c[i] == LEVEL[f][1] else -1 for i, f in enumerate(fs)] for c in CELLS}
    labels = {(0,): "D", (1,): "U", (2,): "L", (0, 1): "DxU", (0, 2): "DxL", (1, 2): "UxL", (0, 1, 2): "DxUxL"}
    eff = {lab: float(np.mean([g[c] * np.prod([x[c][i] for i in ix]) for c in CELLS]) * 2) for ix, lab in labels.items()}
    return tot, phi, eff


# --------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")


CB7 = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999"]  # Okabe-Ito minus black/yellow


def style(i):
    return dict(color=CB7[i % 7], ls=["-", "--", "-."][(i // 7) % 3])


def fig1(main, seeds, ck, out, tag):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for i, m in enumerate(["FDC"] + R_SET):
        C = np.array([main[(m, s)]["n_cert_curve"] for s in seeds], dtype=float)
        med = np.median(C, 0)
        if m == "FDC":
            ax.fill_between(ck, np.quantile(C, .25, 0), np.quantile(C, .75, 0), step="post", color="black", alpha=.15,
                            lw=0)
            ax.step(ck, med, where="post", color="black", lw=2.4, label="FDC (ours)")
        else:
            st = style(i - 1)
            off = 0.08 * (i - 5)  # visual offset only (identical medians would hide each other); noted in title
            ax.step(ck, med + off, where="post", lw=1.2, label=m, **st)
    ax.axhline(12, ls=":", color="grey", lw=.8)
    ax.text(ck[0], 11.4, "stop: 12/15 certified", fontsize=7, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("log rows read (arrivals), log scale")
    ax.set_ylabel(f"problems certified (median of {len(seeds)} streams)")
    ax.set_title(f"Fig. 1  Certified problems vs arrivals, CR9 {tag} (FDC IQR shaded; rival curves offset by <= 0.32 for visibility)",
                 fontsize=8.5)
    ax.legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    save(fig, out, "fig1_certified_vs_arrivals")
    plt.close(fig)


def fig2(c4, out, tag):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    fs = ["D", "U", "L"]
    lab = {"D": "design\n(pool -> 50/50)", "U": "union\n(Q*|Pi| -> feasible)", "L": "ledger\n(r4 -> r5)"}
    vals = [-c4["shapley_log"][f] for f in fs]
    lo = [-c4["shapley_ci95"][f][1] for f in fs]
    hi = [-c4["shapley_ci95"][f][0] for f in fs]
    ax.bar(range(3), vals, color=[OI[5], OI[1], OI[3]], yerr=[np.array(vals) - lo, np.array(hi) - vals], capsize=3)
    ax.set_xticks(range(3), [lab[f] for f in fs])
    ax.set_ylabel("Shapley component of log speed-up\n(FDC vs QFC-pool, 95% CI)")
    ax.set_title(f"design share = {c4['design_share']:.2f} "
                 f"[{c4['design_share_ci95'][0]:.2f}, {c4['design_share_ci95'][1]:.2f}] (rule >= 2/3)", fontsize=7.5)
    ax = axs[1]
    items = [("FDC / B4-bal\n(certificate shape)", c4["FDC_vs_B4bal"]), ("FDC / QFC-pool\n(total)", c4["FDC_vs_QFCpool"])]
    for i, (nm, p) in enumerate(items):
        ax.errorbar([i], [p["geomean_ratio"]], yerr=[[p["geomean_ratio"] - p["ci95_two_sided"][0]],
                                                     [p["ci95_two_sided"][1] - p["geomean_ratio"]]],
                    fmt="o", color=OI[6] if i == 0 else OI[5], capsize=4)
        ax.text(i + .08, p["geomean_ratio"], f"{p['geomean_ratio']:.3f}\nUB {p['ub95_one_sided']:.3f}", fontsize=7,
                va="center")
    ax.axhline(1.0, ls=":", color="grey", lw=.8)
    ax.set_xlim(-.5, 1.9)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(2), [x[0] for x in items])
    ax.set_ylabel("geomean N80 ratio (95% CI)")
    fig.suptitle(f"Fig. 2  Factorial decomposition and certificate shape, CR9 {tag}", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig2_factorial_shapley")
    plt.close(fig)


def fig3(per_method, out, tag):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    mk = {"rigorous": ("o", True), "none": ("s", False), "asymptotic": ("^", False)}
    order = sorted(per_method, key=lambda m: per_method[m]["geomean_N80_over_tau"])
    for j, m in enumerate(order):
        p = per_method[m]
        v = p["validity_class"]
        marker, filled = mk.get(v, ("D", False))
        col = "black" if m == "FDC" else (OI[5] if v == "rigorous" else OI[6])
        x = p["geomean_N80_over_tau"]
        y = p["false_stream_rate"]
        ax.errorbar([x], [y], yerr=[[0], [p["fwer_cp_upper"] - y]], fmt=marker, ms=7 if m == "FDC" else 5,
                    mfc=col if filled else "white", mec=col, ecolor=col, elinewidth=.6, capsize=2, alpha=.9)
        dy = 0.072 - 0.0042 * (j % 8) if y < 0.03 else y + 0.004
        ax.annotate(m, (x, y), xytext=(x, dy), textcoords="data", fontsize=6.2, color=col, rotation=0, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=.3, color=col, alpha=.6) if y < 0.03 else None)
    ax.axhline(DELTA, ls="--", color="grey", lw=.8)
    ax.text(0.995, DELTA + .0015, "delta = 0.05", fontsize=7, color="grey", ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.003, 0.078)
    ax.set_xscale("log")
    ax.set_xlabel("geomean N80_pen / tau_R (log; left = faster)")
    ax.set_ylabel("false-stream rate (bar to one-sided 95% CP UB)")
    ax.set_title(f"Fig. 3  Speed vs guarantee, CR9 {tag}: filled = proven finite-sample, hollow = plug-in / "
                 "asymptotic", fontsize=7.8)
    fig.tight_layout()
    save(fig, out, "fig3_speed_vs_guarantee")
    plt.close(fig)


def fig4(budget_curves, n100, out, tag):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    for i, (m, cur) in enumerate(budget_curves.items()):
        if m == "FDC":
            ax.plot(CR_BUDGETS, cur, color="black", lw=2.2, marker="o", ms=3, label="FDC (ours)")
        else:
            ax.plot(CR_BUDGETS, cur, lw=1.0, label=m, **style(i - 1))
    ax.set_xlabel("budget B (max treated share)")
    ax.set_ylabel("fraction of streams certified\nby the 12/15 stop")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=5.8, ncol=2, frameon=False)
    ax.set_title("per-budget completion (stop window)", fontsize=8)
    ax = axs[1]
    for i, (m, d) in enumerate(n100.items()):
        x = np.sort(np.asarray(d["N100_pen_over_tau"]))
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="black" if m == "FDC" else OI[6],
                lw=2 if m == "FDC" else 1.2, label=f"{m} (geomean {d['geomean_N100_pen_over_tau']:.3f})")
        x8 = np.sort(np.asarray(d["N80_pen_over_tau"]))
        ax.step(x8, np.arange(1, len(x8) + 1) / len(x8), where="post", color="black" if m == "FDC" else OI[6],
                lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("N / tau_R (log); solid = N100, dotted = N80")
    ax.set_ylabel("ECDF over streams")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax.set_title("full horizon: 15/15 (N100)", fontsize=8)
    fig.suptitle(f"Fig. 4  Per-budget completion and N100, CR9 {tag}", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig4_per_budget_completion")
    plt.close(fig)


def fig5(cr12, k60, k20ref, out, tag):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, block, title in ((axs[0], cr12, "CR12 (|Pi| = 4096), K = 20"), (axs[1], k60, "CR9, K = 60 vs K = 20")):
        names = list(block["paired"].keys())
        for i, m in enumerate(names):
            p = block["paired"][m]
            ax.errorbar([i], [p["geomean_ratio"]], yerr=[[p["geomean_ratio"] - p["ci95_two_sided"][0]],
                                                         [p["ci95_two_sided"][1] - p["geomean_ratio"]]],
                        fmt="o", color=OI[5], capsize=3, label="K = 60" if (i == 0 and ax is axs[1]) else None)
            if ax is axs[1] and m in k20ref:
                q = k20ref[m]
                ax.errorbar([i + .22], [q["geomean_ratio"]], yerr=[[q["geomean_ratio"] - q["ci95_two_sided"][0]],
                                                                   [q["ci95_two_sided"][1] - q["geomean_ratio"]]],
                            fmt="s", mfc="white", color=OI[6], capsize=3, label="K = 20 (main)" if i == 0 else None)
        ax.axhline(1.0, ls=":", color="grey", lw=.8)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_ylabel("geomean N80 ratio FDC / method (95% CI)")
        ax.set_title(f"{title}: n = {block['n_streams']}", fontsize=8)
        if ax is axs[1]:
            ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Fig. 5  Supporting blocks (report only, not gates), {tag}", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig5_cr12_k60")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main analysis
def progress(done, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def analyse(mode):
    src = sources(mode)
    out = src["out"]
    out.mkdir(parents=True, exist_ok=True)
    tag = "dev (seeds 900-999, exploratory)" if mode == "pilot" else "eval (seeds 30000-30199, confirmatory)"
    lock = prereg.load_lock()
    lock_info = {"version": lock["version"], "status": lock["status"], "sha256": lock["sha256"]}
    try:
        prereg.validate_lock(lock, 5)
        lock_info["valid"] = True
    except RuntimeError as e:
        lock_info["valid"] = False
        lock_info["error"] = str(e)
    lock_info["frozen_code_drift"] = prereg.frozen_code_drift(lock)
    if mode == "full" and not lock_info["valid"]:
        raise RuntimeError(f"lock v5 invalid: {lock_info}")
    checks = {}
    inputs = {}
    # ---------------- conditional extension blocks (r5_cr_main_ext_a/b): run only if lock cr_n_streams == 400
    P_, F_ = RES / "pilots", RES / "full"
    ext_expected = "complete" if int(lock.get("cr_n_streams", -1)) == 400 else "skipped_by_lock"
    ext_blocks = {}
    for et in ("r5_cr_main_ext_a", "r5_cr_main_ext_b"):
        if mode == "full":
            sp = F_ / et / "summary.json"
        else:
            sp = P_ / et / "full_mode_dryrun" / "summary.json"
            if not sp.exists():
                sp = P_ / et / "summary.json"
        st, why = "missing", None
        if sp.exists():
            d = json.loads(sp.read_text())
            st = d.get("full_mode_under_current_lock") if mode == "pilot" and "full_mode_under_current_lock" in d \
                else d.get("status")
            why = d.get("reason")
        ok = (st == ext_expected) or (ext_expected == "skipped_by_lock" and st == "missing" and mode == "pilot")
        ext_blocks[et] = {"status": st, "expected": ext_expected, "reason": why, "ok": ok,
                          "summary": str(sp.relative_to(WS)) if sp.exists() else None}
        if mode == "full" and ext_expected == "complete" and st == "complete":
            src["main"].append(F_ / et / "results.jsonl")
    checks["extension_blocks"] = {"cr_n_streams": lock.get("cr_n_streams"), "blocks": ext_blocks,
                                  "ok": all(b["ok"] for b in ext_blocks.values()),
                                  "note": ("ext blocks skipped_by_lock (cr_n_streams != 400); main = main_a + main_b only"
                                           if ext_expected == "skipped_by_lock" else
                                           "ext blocks enabled; their rows are pooled into the main block")}
    tau = src["tau_R"]
    ck = [int(x) for x in fr.checkpoints(50_000, tau, 20)]
    seeds = src["seeds_main"]
    n = len(seeds)
    n_expected = lock["cr_n_streams"] if mode == "full" else 100

    # ---------------- main block (C1, C2-stop window)
    main, dup = read_rows(src["main"])
    inputs["main"] = {str(p.relative_to(WS)): sha(p) for p in src["main"]}
    methods_main = ["FDC"] + R_SET
    N = {m: arr(main, m, seeds) for m in methods_main}
    errors_main = [k for k, r in main.items() if r["error"]]
    # Hait-SW (frozen p_min=0.02, gamma=2/3) collapses to the B4-bal design: flag rather than count as distinct evidence
    hs_seeds = [s for s in seeds if ("Hait-SW", s) in main and ("B4-bal", s) in main]
    hs_bad = [s for s in hs_seeds if main[("Hait-SW", s)]["N80_pen"] != main[("B4-bal", s)]["N80_pen"]]
    hait_identical = bool(hs_seeds) and not hs_bad
    checks["hait_sw_vs_b4bal"] = {
        "n_compared": len(hs_seeds), "n_differ": len(hs_bad), "identical": hait_identical,
        "flag": ("Hait-SW == B4-bal on every stream (N80_pen identical): report as one design, not two independent "
                 "rivals; Holm family size unchanged (lock v5) but the paper must say so") if hait_identical else None}
    checks["main_complete"] = {"n_streams": n, "expected": n_expected, "ok": n == n_expected and not errors_main,
                               "duplicate_rows": dup, "error_rows": len(errors_main)}
    idx = boot_idx(n)
    m_R = len(R_SET)
    c1 = {}
    for r in R_SET:
        c1[r] = paired(N["FDC"], N[r], idx, bonf_m=m_R)
        ki_f, ki_r = lattice_index(N["FDC"], ck), lattice_index(N[r], ck)
        dk = ki_f - ki_r
        c1[r]["lattice_diff_hist"] = {str(int(k)): int(v) for k, v in zip(*np.unique(dk[~np.isnan(dk)],
                                                                                    return_counts=True))}
        c1[r]["lattice_diff_median"] = float(np.nanmedian(dk))
        c1[r]["frac_off_lattice"] = float(np.isnan(dk).mean())
    adj, rej = holm({r: c1[r]["p_boot_one_sided"] for r in R_SET})
    for r in R_SET:
        p = c1[r]
        p["p_holm"] = adj[r]
        p["holm_rejected"] = bool(rej[r])
        p["ub_lt_1"] = p["ub95_one_sided"] < 1.0
        if p["geomean_ratio"] <= 0.90 and p["ub_lt_1"] and p["holm_rejected"]:
            p["wording"] = "faster_by_10pct"
        elif p["ub_lt_1"] and p["holm_rejected"]:
            p["wording"] = "faster"
        else:
            p["wording"] = "not_shown_faster"
    iut_pass = all(c1[r]["ub_lt_1"] for r in R_SET)
    hardest = max(R_SET, key=lambda r: c1[r]["ub95_one_sided"])
    progress(1, 8, {"stage": "C1"})

    # per-method summary (main + plugin)
    def method_summary(rows, m, sds, validity_class):
        rr = [rows[(m, s)] for s in sds]
        Np = np.array([x["N80_pen"] for x in rr])
        fs = int(sum(x["fwer_event"] for x in rr))
        secs = [x["sec"] for x in rr if x["sec"] is not None]
        return {"validity_class": validity_class, "n": len(rr),
                "geomean_N80_pen": float(np.exp(np.log(Np).mean())),
                "geomean_N80_over_tau": float(np.exp(np.log(Np).mean()) / tau),
                "geomean_N80_raw": float(np.exp(np.log([x["N80_raw"] for x in rr]).mean())),
                "median_N80_pen": float(np.median(Np)),
                "completed": int(sum(x["completed"] for x in rr)), "completion_rate": float(np.mean([x["completed"] for x in rr])),
                "censored": int(sum((not x["completed"]) or x["fwer_event"] for x in rr)),
                "n_at_tau_R": int(sum(x["N80_pen"] >= tau for x in rr)),
                "false_streams": fs, "false_stream_rate": fs / len(rr), "fwer_cp_upper": cp_upper(fs, len(rr)),
                "false_certs": int(sum(x["n_false"] for x in rr)),
                "billing_ok_all": all(bool(x["billing_ok"]) for x in rr if x["billing_ok"] is not None),
                "sec_per_stream_mean": float(np.mean(secs)) if secs else None,
                "sec_per_stream_median": float(np.median(secs)) if secs else None}
    per_method = {m: method_summary(main, m, seeds, reg.spec(m).validity_class) for m in methods_main}

    # ---------------- plugin block (E-cost)
    pfilt = (lambda r: r.get("window", "stop") == "stop") if mode == "pilot" else None
    plug, pdup = read_rows(src["plugin"], filt=pfilt)
    inputs["plugin"] = {str(p.relative_to(WS)): sha(p) for p in src["plugin"]}
    ecost_set = list(lock["plugin_comparators"]["e_cost_set"])
    desc_only = list(lock["plugin_comparators"]["descriptive_only"])
    pN = {m: arr(plug, m, seeds) for m in ["FDC"] + ecost_set + desc_only}
    checks["plugin_FDC_equals_main_FDC"] = {
        "n_mismatch": int((pN["FDC"] != N["FDC"]).sum()), "ok": bool((pN["FDC"] == N["FDC"]).all())}
    ecost = {}
    for m in ecost_set + desc_only:
        ecost[m] = paired(pN["FDC"], pN[m], idx)
        ecost[m]["role"] = "e_cost_set" if m in ecost_set else "descriptive_only"
        ecost[m]["N80_raw"] = paired(arr(plug, "FDC", seeds, "N80_raw"), arr(plug, m, seeds, "N80_raw"), idx)
        per_method[m] = method_summary(plug, m, seeds, reg.spec(m).validity_class)
    fastest_plugin = min(ecost_set, key=lambda m: per_method[m]["geomean_N80_pen"])
    progress(2, 8, {"stage": "E-cost"})

    # ---------------- C2 (stop window from main; full horizon from audit)
    fs_stop = per_method["FDC"]["false_streams"]
    if src.get("fwer_audit_adapter"):
        audit = t3_fullhorizon_adapter(src["fwer_audit"], ck, tau)
    else:
        audit, _ = read_rows(src["fwer_audit"])
    inputs["fwer_audit"] = {str(p.relative_to(WS)): sha(p) for p in src["fwer_audit"]}
    a_seeds = sorted({s for (m, s) in audit if m == "FDC"})
    fdc_a = [audit[("FDC", s)] for s in a_seeds]
    fs_full = int(sum(bool(x["fwer_event_full"]) for x in fdc_a))
    crit = lock["sap"]["C2"]["critical"]
    c2 = {"rule": lock["sap"]["C2"]["rule"],
          "stop_window": {"n_streams": n, "false_streams": fs_stop, "cp_upper": cp_upper(fs_stop, n),
                          "pass": cp_upper(fs_stop, n) <= DELTA},
          "full_horizon": {"n_streams": len(a_seeds), "false_streams": fs_full, "cp_upper": cp_upper(fs_full, len(a_seeds)),
                           "pass": cp_upper(fs_full, len(a_seeds)) <= DELTA,
                           "false_streams_by_12of15_prefix": int(sum(bool(x["fwer_event_by_12of15"]) for x in fdc_a)),
                           "false_certs_total": int(sum(x["n_false"] for x in fdc_a))},
          "critical_table_200": crit.get("200"),
          "false_stream_seeds_stop": [s for s in seeds if main[("FDC", s)]["fwer_event"]],
          "false_stream_seeds_full": [s for s in a_seeds if audit[("FDC", s)]["fwer_event_full"]],
          "interpretation": lock["sap"]["C2"]["interpretation"]}
    c2["pass"] = c2["stop_window"]["pass"] and c2["full_horizon"]["pass"]
    # consistency: 12/15 prefix of the full-horizon run == stop-window FDC
    common = [s for s in a_seeds if ("FDC", s) in main]
    mm = [s for s in common if audit[("FDC", s)]["N80_pen"] != main[("FDC", s)]["N80_pen"]]
    checks["audit_prefix_equals_main_FDC"] = {"n_compared": len(common), "mismatch_seeds": mm, "ok": not mm}
    if mode == "full":
        checks["audit_complete"] = {"n": len(a_seeds), "expected": 200, "ok": len(a_seeds) == 200}
    n100 = {}
    for m in sorted({m for (m, s) in audit}, key=lambda x: x != "FDC"):
        rr = [audit[(m, s)] for s in a_seeds if (m, s) in audit]
        v = np.array([x["N100_pen"] for x in rr], dtype=float)
        n100[m] = {"n": len(rr), "geomean_N100_pen": float(np.exp(np.log(v).mean())),
                   "geomean_N100_pen_over_tau": float(np.exp(np.log(v).mean()) / tau),
                   "completed_15_of_15": int(sum(bool(x["completed_15"]) for x in rr)),
                   "false_streams_full": int(sum(bool(x["fwer_event_full"]) for x in rr)),
                   "N100_pen_over_tau": (v / tau).tolist(),
                   "N80_pen_over_tau": (np.array([x["N80_pen"] for x in rr]) / tau).tolist()}
    if "FDC" in n100 and "B2-fav-tight" in n100:
        common_n = [s for s in a_seeds if ("B2-fav-tight", s) in audit]
        a1 = np.array([audit[("FDC", s)]["N100_pen"] for s in common_n], dtype=float)
        a2 = np.array([audit[("B2-fav-tight", s)]["N100_pen"] for s in common_n], dtype=float)
        n100["FDC_over_B2-fav-tight_N100"] = paired(a1, a2, boot_idx(len(common_n)))
    if src.get("fwer_audit_xcheck"):
        xc, _ = read_rows(src["fwer_audit_xcheck"])
        cm = [k for k in xc if k in audit]
        bad = [list(k) for k in cm if xc[k]["N100_pen"] != audit[k]["N100_pen"]
               or bool(xc[k]["fwer_event_full"]) != bool(audit[k]["fwer_event_full"])
               or xc[k]["N80_pen"] != audit[k]["N80_pen"]]
        checks["fwer_audit_schema_path_vs_T3_fullhorizon"] = {"n_compared": len(cm), "mismatch": bad, "ok": not bad}
    progress(3, 8, {"stage": "C2"})

    # ---------------- C4 factorial
    if mode == "pilot":
        fac, _ = read_rows(src["factorial"], rename=T1_CFG, filt=lambda r: r.get("K", 20) == 20)
    else:
        fac, _ = read_rows(src["factorial"])
    inputs["factorial"] = {str(p.relative_to(WS)): sha(p) for p in src["factorial"]}
    f_seeds = sorted({s for (m, s) in fac if m == cell_name(CELLS[7])})
    Lc = {c: np.log(arr(fac, cell_name(c), f_seeds)) for c in CELLS}
    fidx = boot_idx(len(f_seeds))
    g = {c: Lc[c].mean() for c in CELLS}
    tot, phi, eff = shapley_stats(g)
    bs_phi = np.zeros((B_BOOT, 3))
    bs_share = np.zeros(B_BOOT)
    Gb = {c: Lc[c][fidx].mean(1) for c in CELLS}
    for b in range(B_BOOT):
        t_b, p_b, _ = shapley_stats({c: Gb[c][b] for c in CELLS})
        bs_phi[b] = [p_b["D"], p_b["U"], p_b["L"]]
        bs_share[b] = p_b["D"] / t_b if t_b != 0 else np.nan
    share = phi["D"] / tot if tot else float("nan")
    b4b = arr(fac, "B4-bal", f_seeds)
    fdc_f = np.exp(Lc[CELLS[7]])
    qfc_f = np.exp(Lc[CELLS[0]])
    vs_b4 = paired(fdc_f, b4b, fidx)
    vs_qfc = paired(fdc_f, qfc_f, fidx)
    # identities vs main block
    ident = {}
    for c, m in ((CELLS[7], "FDC"), (CELLS[0], "QFC-pool")):
        cs = [s for s in f_seeds if (m, s) in main]
        bad = [s for s in cs if fac[(cell_name(c), s)]["N80_pen"] != main[(m, s)]["N80_pen"]]
        ident[f"{cell_name(c)}=={m}"] = {"n_compared": len(cs), "mismatch_seeds": bad, "ok": not bad}
    csb = [s for s in f_seeds if ("B4-bal", s) in main]
    ident["B4-bal(factorial)==B4-bal(main)"] = {
        "n_compared": len(csb), "ok": all(fac[("B4-bal", s)]["N80_pen"] == main[("B4-bal", s)]["N80_pen"] for s in csb)}
    checks["factorial_identities"] = ident
    # prediction model
    pred = json.loads(Path(src["predictions"]).read_text())
    perr = {}
    for c in CELLS:
        nm = cell_name(c)
        pc = pred["cells"][nm]
        obs = float(np.exp(g[c]))
        perr[nm] = {"pred_geomean": pc["pred_geomean_N80_primary"], "obs_geomean": obs,
                    "abs_log_err": abs(math.log(pc["pred_geomean_N80_primary"] / obs)),
                    "abs_log_err_secondary_M1_transfer": abs(pc["pred_log_secondary_M1_transfer"] - g[c])}
    max_err = max(v["abs_log_err"] for v in perr.values())
    lim = lock["c4"]["prediction_model"]["error_limit_abs_log"]
    pred_ok = max_err <= lim
    # dev validation half (seeds 950-999) reproduces the lock's dev_validation_max_abs_log_err
    if mode == "pilot":
        vs = [s for s in f_seeds if s >= 950]
        errs_v = []
        for c in CELLS:
            o = float(np.exp(np.log(arr(fac, cell_name(c), vs)).mean()))
            errs_v.append(abs(math.log(pred["cells"][cell_name(c)]["pred_geomean_N80_primary"] / o)))
        checks["c4_dev_validation_max_abs_log_err"] = {
            "recomputed": max(errs_v), "lock": lock["c4"]["prediction_model"]["dev_validation_max_abs_log_err"],
            "ok": abs(max(errs_v) - lock["c4"]["prediction_model"]["dev_validation_max_abs_log_err"]) < 1e-9}
    c4 = {"n_streams": len(f_seeds),
          "cells": {cell_name(c): {"cell": list(c), "geomean_N80_pen": float(np.exp(g[c])), "mean_log": float(g[c]),
                                   "false_streams": int(sum(fac[(cell_name(c), s)]["fwer_event"] for s in f_seeds))}
                    for c in CELLS},
          "total_log_speedup_FDC_vs_QFCpool": float(tot), "total_ratio": float(math.exp(tot)),
          "shapley_log": {k: float(v) for k, v in phi.items()},
          "shapley_share": {k: float(v / tot) for k, v in phi.items()},
          "shapley_ci95": {f: [float(np.quantile(bs_phi[:, i], .025)), float(np.quantile(bs_phi[:, i], .975))]
                           for i, f in enumerate("DUL")},
          "design_share": float(share),
          "design_share_ci95": [float(np.nanquantile(bs_share, .025)), float(np.nanquantile(bs_share, .975))],
          "design_share_lower95_one_sided": float(np.nanquantile(bs_share, .05)),
          "design_share_ge_2_3": bool(share >= 2 / 3),
          "effects_coded_log": eff,
          "FDC_vs_B4bal": vs_b4, "FDC_vs_QFCpool": vs_qfc,
          "certificate_shape_ub_lt_1": bool(vs_b4["ub95_one_sided"] < 1.0),
          "prediction": {"per_cell": perr, "max_abs_log_err": max_err, "limit": lim, "within_limit": pred_ok,
                         "predictions_file": str(Path(src["predictions"]).relative_to(WS)),
                         "predictions_written_at": pred.get("written_at"),
                         "mode": "predictive" if pred_ok else "descriptive"},
          "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED}}
    c4["supported"] = bool(c4["design_share_ge_2_3"] and c4["certificate_shape_ub_lt_1"])
    c4["mechanism_wording"] = (
        ("design-driven and predictable" if pred_ok else "design-driven (descriptive decomposition only)")
        if c4["supported"] else "C4 withdrawn (decomposition reported descriptively); C1 unaffected")
    if mode == "pilot" and src.get("factorial_schema_check"):
        fs_rows, _ = read_rows(src["factorial_schema_check"])
        bad = [k for k, v in fs_rows.items() if k in fac and fac[k]["N80_pen"] != v["N80_pen"]]
        checks["factorial_schema_path_vs_t1"] = {"n_compared": sum(1 for k in fs_rows if k in fac),
                                                 "mismatch": [list(k) for k in bad], "ok": not bad}
    progress(4, 8, {"stage": "C4"})

    # ---------------- K60 sensitivity
    k60r, _ = read_rows(src["k60"])
    inputs["k60"] = {str(p.relative_to(WS)): sha(p) for p in src["k60"]}
    k_seeds = sorted({s for (m, s) in k60r if m == "FDC"})
    kidx = boot_idx(len(k_seeds))
    k_methods = sorted({m for (m, s) in k60r if m != "FDC"}, key=lambda m: (m not in R_SET, m))
    k60 = {"n_streams": len(k_seeds), "K": 60, "rule": lock["sap"]["K60_sensitivity"]["rule"],
           "paired": {m: paired(arr(k60r, "FDC", k_seeds), arr(k60r, m, k_seeds), kidx) for m in k_methods},
           "false_streams": {m: int(sum(k60r[(m, s)]["fwer_event"] for s in k_seeds)) for m in ["FDC"] + k_methods}}
    k20ref = {}
    ks_main = [s for s in k_seeds if ("FDC", s) in main]
    if len(ks_main) == len(k_seeds):
        ki = boot_idx(len(ks_main))
        for m in k_methods:
            if m in R_SET:
                k20ref[m] = paired(arr(main, "FDC", ks_main), arr(main, m, ks_main), ki)
            elif ("FDC", ks_main[0]) in plug and (m, ks_main[0]) in plug:
                k20ref[m] = paired(arr(plug, "FDC", ks_main), arr(plug, m, ks_main), ki)
    k60["K20_same_streams"] = k20ref

    # ---------------- CR12
    c12r, _ = read_rows(src["cr12"])
    inputs["cr12"] = {str(p.relative_to(WS)): sha(p) for p in src["cr12"]}
    c_seeds = sorted({s for (m, s) in c12r if m == "FDC"})
    cidx = boot_idx(len(c_seeds))
    c_methods = [m for m in R_SET if (m, c_seeds[0]) in c12r]
    cr12 = {"n_streams": len(c_seeds), "layer": "CR12", "role": "supporting, not a gate, not in C1",
            "paired": {m: paired(arr(c12r, "FDC", c_seeds), arr(c12r, m, c_seeds), cidx) for m in c_methods},
            "per_method": {}}
    for m in ["FDC"] + c_methods:
        rr = [c12r[(m, s)] for s in c_seeds]
        ms = [x for r in rr for x in (r["cert_ms_per_checkpoint"] or [])]
        cr12["per_method"][m] = {"geomean_N80_pen": float(np.exp(np.mean(np.log([r["N80_pen"] for r in rr])))),
                                 "completed": int(sum(r["completed"] for r in rr)),
                                 "false_streams": int(sum(r["fwer_event"] for r in rr)),
                                 "cert_ms_per_checkpoint_mean": float(np.mean(ms)) if ms else None,
                                 "cert_ms_per_checkpoint_median": float(np.median(ms)) if ms else None}
    progress(5, 8, {"stage": "K60/CR12"})

    # ---------------- per-budget completion, per-problem certification times
    budget_curves, per_problem_t = {}, {}
    for m in methods_main:
        CK = np.array([main[(m, s)]["cert_k"] for s in seeds])
        budget_curves[m] = (CK >= 0).mean(0).tolist()
        tt = np.where(CK >= 0, np.array(ck)[np.clip(CK, 0, None)], np.nan)
        per_problem_t[m] = [float(np.nanmedian(tt[:, q])) if np.isfinite(tt[:, q]).any() else None
                            for q in range(tt.shape[1])]
    # ---------------- replica gate
    rep = json.loads(Path(src["replica"]).read_text()) if Path(src["replica"]).exists() else None
    replica = None if rep is None else {
        "status": rep.get("status"), "go_no_go": rep.get("go_no_go"), "n_reruns": rep.get("n_reruns"),
        "n_mismatch": rep.get("n_mismatch"), "n_errors": rep.get("n_errors"), "hash_pass": rep.get("hash_pass"),
        "blocks_aggregate": rep.get("blocks_aggregate"), "file": str(Path(src["replica"]).relative_to(WS))}

    # ---------------- practical significance
    pkl = (Path(__import__("os").environ.get("DATA_DIR", "data")) / "criteo_uplift_real/tidy.pkl")
    gz = pkl.parent / "criteo-research-uplift-v2.1.csv.gz"
    prov = json.loads((pkl.parent / "PROVENANCE.json").read_text())
    nrows_ds = int(prov["n_rows"])
    bpr_mem = pkl.stat().st_size / nrows_ds
    bpr_gz = gz.stat().st_size / nrows_ds
    g_f = per_method["FDC"]["geomean_N80_pen"]
    rows_saved = {}
    for r in R_SET + [fastest_plugin]:
        gr = per_method[r]["geomean_N80_pen"]
        d = arr(main if r in R_SET else plug, r, seeds) - N["FDC"]
        rows_saved[r] = {"geomean_rows_saved": gr - g_f, "median_per_stream_rows_saved": float(np.median(d)),
                         "MB_saved_in_memory": (gr - g_f) * bpr_mem / 1e6, "MB_saved_gz": (gr - g_f) * bpr_gz / 1e6,
                         "frac_of_tau_saved": (gr - g_f) / tau}
    cert_ms = {}
    cm_rows = main
    if mode == "pilot":
        cm_rows, _ = read_rows(src["main_schema_check"])
    for m in methods_main:
        v = [r["sec_cert"] * 1000 / ((r["k_stop"] + 1) if r["k_stop"] is not None and r["k_stop"] >= 0 else 20)
             for (mm_, s), r in cm_rows.items() if mm_ == m and r["sec_cert"] is not None]
        cert_ms[m] = {"ms_per_checkpoint_mean": float(np.mean(v)) if v else None,
                      "ms_per_checkpoint_median": float(np.median(v)) if v else None, "n_streams": len(v)}
    practical = {
        "bytes_per_row": {"in_memory_tidy_pkl": bpr_mem, "csv_gz": bpr_gz, "n_rows_dataset": nrows_ds,
                          "note": "bytes/row = file size / 13,979,592 rows (Criteo Uplift v2.1); 'rows read' = arrivals"},
        "rows_saved_vs": rows_saved,
        "hardest_rival": hardest,
        "wall_clock_sec_per_stream": {m: per_method[m]["sec_per_stream_mean"] for m in per_method},
        "certificate_ms_per_checkpoint": cert_ms,
        "certificate_ms_source": ("pilot: r5_cr_main_a/b dev pilot rows (seeds 900-909), t2 rows carry no sec_cert"
                                  if mode == "pilot" else "main eval blocks"),
        "preprocessing_cost": "FDC ledger/union enumeration happens in setup(); it is included in sec_per_stream "
                              "(not timed separately); CR12 ms/checkpoint in cr12.per_method",
        "timing_note": "all timings from 4-slot concurrent runs on a shared 20-core host: biased up"}
    if mode == "pilot":
        sc, _ = read_rows(src["main_schema_check"])
        bad = [list(k) for k, v in sc.items() if k in main and main[k]["N80_pen"] != v["N80_pen"]]
        checks["main_schema_path_vs_t2"] = {"n_compared": sum(1 for k in sc if k in main), "mismatch": bad,
                                            "ok": not bad}
        ps, _ = read_rows(src["plugin_schema_check"])
        bad = [list(k) for k, v in ps.items() if k in plug and plug[k]["N80_pen"] != v["N80_pen"]]
        checks["plugin_schema_path_vs_t3"] = {"n_compared": sum(1 for k in ps if k in plug), "mismatch": bad,
                                              "ok": not bad}
        # exact reproduction of the T2 dev gate
        t2 = json.loads((RES / "pilots/r5_t2_gate_strict/summary.json").read_text())["gate_T2"]["per_rival"]
        dif = {}
        for r in R_SET:
            dif[r] = max(abs(c1[r]["geomean_ratio"] - t2[r]["geomean_ratio"]),
                         abs(c1[r]["ub95_one_sided"] - t2[r]["ub95_one_sided"]),
                         abs(c1[r]["ci95_two_sided"][0] - t2[r]["ci95_two_sided"][0]),
                         abs(c1[r]["ci95_two_sided"][1] - t2[r]["ci95_two_sided"][1]),
                         abs(c1[r]["frac_faster"] - t2[r]["frac_faster"]))
        checks["reproduces_T2_dev_exactly"] = {"max_abs_diff": max(dif.values()), "per_rival": dif,
                                               "ok": max(dif.values()) == 0.0}
        t3 = json.loads((RES / "pilots/r5_t2_gate_plugin_t3/summary.json").read_text())["E_cost"]["per_plugin"]
        d3 = {m: abs(ecost[m]["geomean_ratio"] - t3[m]["N80_pen"]["geomean_ratio"]) for m in t3 if m in ecost}
        checks["reproduces_T3_ecost_point"] = {"max_abs_diff": max(d3.values()), "ok": max(d3.values()) < 1e-9}
        sd = json.loads((RES / "pilots/r5_t1_reconcile/shapley_dev.json").read_text())
        dd = max(abs(sd["shapley_log"][k] - c4["shapley_log"][k[0].upper()]) for k in ("design", "union", "ledger"))
        checks["reproduces_t1_shapley_point"] = {"max_abs_diff": dd, "ok": dd < 1e-9,
                                                 "note": "t1 used B = 2000 for CIs; here B = 1e4 (SAP), points equal"}
    progress(6, 8, {"stage": "practical"})

    # ---------------- verdict
    verdict = "positive_result_achieved" if (iut_pass and c2["pass"]) else "not_achieved"
    status = "complete"
    if mode == "full":
        if not checks["main_complete"]["ok"]:
            status = "incomplete_main_block"
        if replica and replica.get("blocks_aggregate"):
            status = "blocked_by_replica_check"
    summary = {
        "task_id": TASK, "mode": mode, "status": status, "data": tag,
        "confirmatory": mode == "full",
        "lock": lock_info, "tau_R": tau, "checkpoints": ck, "n_streams_main": n,
        "bootstrap": {"type": "paired percentile, streams resampled, same resamples for all rivals", "B": B_BOOT,
                      "seed": BOOT_SEED},
        "verdict": verdict, "verdict_rule": lock["sap"]["verdict_rule"],
        "C1": {"iut_pass": iut_pass, "hardest_rival_by_ub": hardest, "per_rival": c1,
               "falsified_if": lock["sap"]["C1"]["falsified_if"],
               "holm": {"alpha": ALPHA, "m": m_R, "all_rejected": all(c1[r]["holm_rejected"] for r in R_SET)}},
        "C2": c2, "C4": c4,
        "E_cost": {"set": ecost_set, "descriptive_only": desc_only, "per_plugin": ecost,
                   "fastest_plugin_rule": lock["plugin_comparators"]["fastest_plugin_rule"],
                   "fastest_plugin": fastest_plugin, "FDC_over_fastest_plugin": ecost[fastest_plugin],
                   "wording": lock["plugin_comparators"]["wording"]},
        "K60_sensitivity": k60, "CR12": cr12, "N100": n100,
        "per_method": per_method, "per_budget_completion": {"budgets": CR_BUDGETS, "curves": budget_curves},
        "per_problem_median_cert_time": per_problem_t,
        "practical_significance": practical, "replica_check": replica,
        "checks": checks, "inputs_sha256": inputs,
        "code_sha256": sha(Path(__file__)),
    }
    return summary, out, main, seeds, ck, budget_curves, n100, k20ref


# --------------------------------------------------------------------------------------------- outputs
def fmt_ci(p):
    return f"{p['geomean_ratio']:.3f} [{p['ci95_two_sided'][0]:.3f}, {p['ci95_two_sided'][1]:.3f}]"


def write_tables(S, out):
    import csv
    pm = S["per_method"]
    rows = []
    for m in ["FDC"] + R_SET + S["E_cost"]["set"] + S["E_cost"]["descriptive_only"]:
        p = pm[m]
        cmp_ = S["C1"]["per_rival"].get(m) or S["E_cost"]["per_plugin"].get(m)
        rows.append({"Method": m, "Validity": p["validity_class"],
                     "Group": "ours" if m == "FDC" else ("R (rigorous)" if m in R_SET else "plug-in (descriptive)"),
                     "GeoMean N80": round(p["geomean_N80_pen"]), "GeoMean N80/tau": round(p["geomean_N80_over_tau"], 4),
                     "FDC/Method [95% CI]": "-" if m == "FDC" else fmt_ci(cmp_),
                     "UB95": "-" if m == "FDC" else round(cmp_["ub95_one_sided"], 4),
                     "Holm p": round(cmp_["p_holm"], 5) if m in R_SET else "-",
                     "Wording": cmp_.get("wording", "-") if m in R_SET else "-",
                     "Faster %": "-" if m == "FDC" else round(100 * cmp_["frac_faster"], 1),
                     "Tied %": "-" if m == "FDC" else round(100 * cmp_["frac_tied"], 1),
                     "Completion": f"{p['completed']}/{p['n']}",
                     "At tau_R": p["n_at_tau_R"],
                     "False streams (CP UB)": f"{p['false_streams']} ({p['fwer_cp_upper']:.4f})",
                     "Note": ("identical to B4-bal on every stream" if m == "Hait-SW"
                              and S["checks"].get("hait_sw_vs_b4bal", {}).get("identical") else "")})
    with open(out / "table1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["Method", "Validity", "GeoMean N80", "FDC/Method [95% CI]", "UB95", "Faster %", "Completion",
            "False streams (CP UB)"]
    md = [f"Table 1. CR9 {S['data']}; N80_pen geometric means over {S['n_streams_main']} paired streams; ratio < 1 "
          "means FDC reads fewer rows. Rigorous rivals R carry proven finite-sample guarantees; plug-in rows are "
          "descriptive (faster plug-ins have no guarantee). Completion counts a stream that first reaches 12/15 at the "
          "last checkpoint t_K = tau_R as completed with N80 = tau_R (lock v5); column 'At tau_R' in table1.csv "
          "gives how many streams sit at tau_R.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) + (" (= B4-bal)" if c == "Method" and r["Note"] else "")
                                    for c in cols) + " |")
    if S["checks"].get("hait_sw_vs_b4bal", {}).get("identical"):
        md += ["", "Hait-SW with its frozen dev-tuned config (p_min = 0.02, gamma = 2/3) reduces to the B4-bal design and "
               "gives identical N80 on every stream; it is listed for completeness, not as independent evidence."]
    (out / "table1.md").write_text("\n".join(md) + "\n")
    # Table 2: rigorous-rival qualification
    rc = json.loads((RES / "r5_gates/rival_configs.json").read_text())["frozen_configs"]
    t2 = ["Table 2. Rigorous-rival qualification (lock v5 set R). Every rival's guarantee holds under without-"
          "replacement replay, adaptive counts, checkpoint stopping and one delta over 15 problems; source "
          "plan/baseline_qualification_r5.md.", "",
          "| Method | Source / certificate | Validity | Tuning (dev 900-949) | Frozen config | Dev GeoMean N80/tau | "
          "Dev false streams |", "|---|---|---|---|---|---|---|"]
    tuning = {"B4-bal": "share in {0.40, 0.45, 0.50}", "B2-rect": "explore const in {1.0, 0.5}",
              "Hait-SW": "p_min x gamma (6 configs)"}
    for m in R_SET:
        s = reg.spec(m)
        p = S["per_method"][m]
        t2.append(f"| {m} | {s.source} | {s.validity_class} | {tuning.get(m, 'none (published / r4 constants)')} | "
                  f"{json.dumps(rc.get(m) or {})} | {p['geomean_N80_over_tau']:.3f} | {p['false_streams']}/{p['n']} |")
    t2.append("")
    t2.append("Excluded from R: QFC-half (ablation of FDC, C4 only); B2/B3/Peace-nominal (thresholds proven only with "
              "replacement); Hait-pooled (needs with-replacement sampling); Shekhar & Howard 2605.21736 (fixed-n, "
              "not anytime-valid); H8-* (model-internal validity).")
    (out / "table2.md").write_text("\n".join(t2) + "\n")


def write_claims(S, out):
    c1, c2, c4, e = S["C1"], S["C2"], S["C4"], S["E_cost"]
    rel = str(out.relative_to(WS))
    hard = c1["per_rival"][c1["hardest_rival_by_ub"]]
    claims = {
        "task_id": TASK, "mode": S["mode"], "confirmatory": S["confirmatory"], "verdict": S["verdict"],
        "verdict_rule": S["verdict_rule"], "lock_sha256": S["lock"]["sha256"],
        "flags": [f for f in (S["checks"].get("hait_sw_vs_b4bal", {}).get("flag"),
                              S["checks"].get("extension_blocks", {}).get("note")) if f],
        "claims": [
            {"id": "C1", "statement": "FDC reaches 12/15 certified problems with fewer logged rows than every listed "
                                      "rigorous competitor in R (geometric mean of paired N80_pen ratios < 1).",
             "status": "supported" if c1["iut_pass"] else "falsified",
             "numbers": {r: {"ratio": p["geomean_ratio"], "ci95": p["ci95_two_sided"], "ub95": p["ub95_one_sided"],
                             "ub_bonferroni": p["ub_bonferroni"], "p_holm": p["p_holm"], "wording": p["wording"],
                             "frac_faster": p["frac_faster"]} for r, p in c1["per_rival"].items()},
             "hardest_rival": c1["hardest_rival_by_ub"], "hardest_ub95": hard["ub95_one_sided"],
             "falsification_condition": c1["falsified_if"],
             "evidence": [f"{rel}/table1.csv", f"{rel}/fig1_certified_vs_arrivals.pdf", f"{rel}/summary.json#C1"]},
            {"id": "C2", "statement": "FDC keeps the stream-level false-certificate rate at or below delta = 0.05 "
                                      "(one-sided 95% Clopper-Pearson upper bound) in both windows.",
             "status": "supported" if c2["pass"] else "falsified",
             "numbers": {"stop_window": c2["stop_window"], "full_horizon": c2["full_horizon"]},
             "falsification_condition": "CP upper bound > 0.05 in either window (n = 200: more than 4 false streams)",
             "evidence": [f"{rel}/summary.json#C2", f"{rel}/fig3_speed_vs_guarantee.pdf"]},
            {"id": "C4", "statement": "The speed-up over QFC-pool is design-driven (Shapley design share >= 2/3) and "
                                      "the direction-level certificate beats the rectangle certificate on the same "
                                      "50/50 design (FDC vs B4-bal UB < 1).",
             "status": "supported" if c4["supported"] else "withdrawn",
             "mechanism_mode": c4["prediction"]["mode"], "mechanism_wording": c4["mechanism_wording"],
             "numbers": {"design_share": c4["design_share"], "design_share_ci95": c4["design_share_ci95"],
                         "shapley_log": c4["shapley_log"], "FDC_vs_B4bal": {
                             k: c4["FDC_vs_B4bal"][k] for k in ("geomean_ratio", "ci95_two_sided", "ub95_one_sided")},
                         "prediction_max_abs_log_err": c4["prediction"]["max_abs_log_err"],
                         "prediction_limit": c4["prediction"]["limit"]},
             "falsification_condition": "design share < 2/3 or FDC/B4-bal UB >= 1.0 (C4 withdrawn, C1 unaffected); "
                                        "any cell |log(pred/obs)| > 0.095 -> descriptive wording only",
             "evidence": [f"{rel}/fig2_factorial_shapley.pdf", f"{rel}/summary.json#C4"]},
            {"id": "E-cost", "statement": "Cost of the guarantee: paired FDC / plug-in ratios (descriptive, no pass "
                                          "line); plug-in methods carry no finite-sample guarantee.",
             "status": "descriptive",
             "numbers": {m: {"ratio": p["geomean_ratio"], "ci95": p["ci95_two_sided"], "role": p["role"]}
                         for m, p in e["per_plugin"].items()},
             "fastest_plugin": e["fastest_plugin"],
             "falsification_condition": "none (estimate only)",
             "evidence": [f"{rel}/table1.csv", f"{rel}/fig3_speed_vs_guarantee.pdf"]},
            {"id": "S-K60", "statement": "K = 60 checkpoint-grid sensitivity (report only, never mixed with K = 20).",
             "status": "descriptive",
             "numbers": {m: {"ratio": p["geomean_ratio"], "ci95": p["ci95_two_sided"]}
                         for m, p in S["K60_sensitivity"]["paired"].items()},
             "falsification_condition": "none (report only)", "evidence": [f"{rel}/fig5_cr12_k60.pdf"]},
            {"id": "S-CR12", "statement": "CR12 (|Pi| = 4096) scale support (not a gate, not in C1).",
             "status": "descriptive",
             "numbers": {m: {"ratio": p["geomean_ratio"], "ci95": p["ci95_two_sided"]}
                         for m, p in S["CR12"]["paired"].items()},
             "falsification_condition": "none (supporting)", "evidence": [f"{rel}/fig5_cr12_k60.pdf"]},
            {"id": "P-practical", "statement": "Practical significance: rows (and bytes) not read, wall clock, "
                                               "certificate cost per checkpoint.",
             "status": "descriptive",
             "numbers": {"rows_saved_vs_hardest": S["practical_significance"]["rows_saved_vs"][c1["hardest_rival_by_ub"]],
                         "cert_ms_per_checkpoint_FDC": S["practical_significance"]["certificate_ms_per_checkpoint"]["FDC"]},
             "falsification_condition": "none", "evidence": [f"{rel}/summary.json#practical_significance"]},
        ]}
    (out / "claims.json").write_text(json.dumps(claims, indent=1))
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    a = ap.parse_args()
    t0 = datetime.now()
    pid = RES / f"{TASK}.pid"
    pid.write_text(str(os.getpid()))
    progress(0, 8, {"stage": "start"})
    status, txt = "success", ""
    try:
        S, out, main_rows, seeds, ck, budget_curves, n100, k20ref = analyse(a.mode)
        tag = "dev" if a.mode == "pilot" else "eval"
        fig1(main_rows, seeds, ck, out, tag)
        fig2(S["C4"], out, tag)
        fig3({m: S["per_method"][m] for m in S["per_method"]}, out, tag)
        fig4(budget_curves, {k: v for k, v in n100.items() if isinstance(v, dict) and "N100_pen_over_tau" in v},
             out, tag)
        fig5(S["CR12"], S["K60_sensitivity"], k20ref, out, tag)
        write_tables(S, out)
        write_claims(S, out)
        S["started_at"] = t0.isoformat()
        S["ended_at"] = datetime.now().isoformat()
        S["wall_min"] = round((datetime.now() - t0).total_seconds() / 60, 2)
        figs = ["fig1_certified_vs_arrivals", "fig2_factorial_shapley", "fig3_speed_vs_guarantee",
                "fig4_per_budget_completion", "fig5_cr12_k60"]
        S["outputs"] = {"rendered": {f: all((out / f"{f}.{e}").exists() for e in ("pdf", "png")) for f in figs},
                        "files": sorted(p.name for p in out.iterdir())}
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
        progress(8, 8, {"stage": "done", "verdict": S["verdict"]})
        txt = f"{a.mode}: verdict={S['verdict']} C1_iut={S['C1']['iut_pass']} C2={S['C2']['pass']} " \
              f"C4={S['C4']['supported']} checks_ok={all(v.get('ok', True) for v in S['checks'].values() if isinstance(v, dict) and 'ok' in v)}"
        print(txt, flush=True)
    except Exception:
        status = "failed"
        txt = traceback.format_exc()
        print(txt, flush=True)
    finally:
        if pid.exists():
            pid.unlink()
        prog = {}
        pf = RES / f"{TASK}_PROGRESS.json"
        if pf.exists():
            prog = json.loads(pf.read_text())
        (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": txt[-2000:],
                                                      "final_progress": prog, "mode": a.mode,
                                                      "timestamp": datetime.now().isoformat()}))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
