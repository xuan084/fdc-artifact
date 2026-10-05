"""Column-width figure for paper section 6.5: TU-FDC-DP / RECT-BF-DP-TU against eps (block G1, eval).

r6c: larger markers and a marker-style legend (critic r6 P2-3); same size, same data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path(__file__).resolve().parents[2]
S_ = json.loads((W / "exp/results/full/v10_posthoc_G/summary.json").read_text())["G1"]
OUT = W / "writing/latex_acm/figures/fig5_eps_curve.pdf"
COL = {16: "#2a78d6", 32: "#eb6834", 64: "#1baf7a"}
MK = {16: "o", 32: "s", 64: "^"}

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.38), sharey=True)
for ax, (d, name) in zip(axes, [("x5", "X5"), ("lenta", "Lenta")]):
    for S in (16, 32, 64):
        g = S_[f"G1-eval-{d}-{S}"]["by_eps"]
        es = sorted(g, key=float)
        x = [float(e) for e in es]
        y = [g[e]["ratios"]["FDC/RECT-BF-DP-TU"]["geomean_ratio"] for e in es]
        cens = [g[e]["methods"]["RECT-BF-DP-TU"]["n80_lt_tau"] < 0.8 * S_[f"G1-eval-{d}-{S}"]["n"] for e in es]
        ax.plot(x, y, color=COL[S], lw=1.1, label=f"$S$ = {S}")
        for xi, yi, c in zip(x, y, cens):
            ax.plot(xi, yi, MK[S], ms=4.6, color=COL[S], mfc="white" if c else COL[S], mew=1.0)
        le = [float(e) for e in es if g[e]["lock_eps"]]
        for xi in le:
            yi = y[x.index(xi)]
            ax.plot(xi, yi, "o", ms=9.5, mfc="none", mec="0.15", mew=0.8)
    ax.set_title(name, fontsize=8, pad=2)
    ax.set_xlabel(r"tolerance $\varepsilon$", labelpad=1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(length=2, pad=1.5)
    if d == "lenta":
        ax.set_xticks([0.004, 0.008, 0.012])
axes[0].set_ylabel("FDC-DP / rect. rows", labelpad=1)
axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axes[0].set_ylim(0, 0.85)
h0, l0 = axes[0].get_legend_handles_labels()
axes[1].legend(h0, l0, frameon=False, fontsize=6.5, loc="upper right", handlelength=1.4, borderaxespad=0.1)
from matplotlib.lines import Line2D
hs = [Line2D([], [], ls="", marker="o", ms=4.0, color="0.3", mew=1.0, label="filled: \u226580% before $\\tau_R$"),
      Line2D([], [], ls="", marker="o", ms=4.0, color="0.3", mfc="white", mew=1.0, label="hollow: <80%"),
      Line2D([], [], ls="", marker="o", ms=7.0, mfc="none", mec="0.15", mew=0.8, label="lock $\\varepsilon$")]
axes[0].legend(handles=hs, frameon=False, fontsize=6.0, loc="upper right", handlelength=0.8, handletextpad=0.3,
               borderaxespad=0.1, labelspacing=0.25)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT)
print(OUT)
