"""
Paper-quality version of the Stage 0 figure for the USENIX proposal.

Reads stage0_results.csv (written by stage0_allocation.py) and draws the same
two panels as stage0_gap.png, sized for a two-column USENIX page (figure*,
7 in wide) and saved as vector PDF so it stays sharp in print.

Usage:  python3 plot_paper.py stage0_results.csv figs/stage0.pdf
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
rows = [{k: float(v) if v not in ("", "nan") else float("nan") for k, v in r.items()}
        for r in csv.DictReader(open(src))]

# Times-like serif to match the USENIX body font; stix math to match.
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False})

# Blue/orange carry the uniform-vs-greedy contrast (colour-blind safe); MIP and LP in greys.
style = {"uniform": dict(color="#1f4e9c", ls="-",  marker="o"),
         "greedy":  dict(color="#c2601a", ls="--", marker="s"),
         "mip":     dict(color="#333333", ls="-.", marker="^"),
         "lp":      dict(color="#999999", ls=":",  marker="v")}
label = {"uniform": "uniform cut", "greedy": "greedy merit order", "mip": "MIP optimum", "lp": "LP bound"}

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.25))

# (a) cost vs depth on the fully heterogeneous cluster
sub = [r for r in rows if r["h"] == 1.0]
for k in ("uniform", "greedy", "mip", "lp"):
    ax[0].plot([r["B"] for r in sub], [r[k] for r in sub], ms=3.5, lw=1.2, label=label[k], **style[k])
ax[0].set_xlabel("cluster power cut $B$ (fraction of nominal)")
ax[0].set_ylabel("weighted SLO cost")
ax[0].set_title("(a) heterogeneous cluster ($h=1$)")
ax[0].legend(frameon=False, loc="upper left")

# (b) ratio to optimum vs heterogeneity at B = 0.20 (h = 1 omitted: optimum cost is 0 there)
sub = [r for r in rows if abs(r["B"] - 0.20) < 1e-9 and r["mip"] > 1e-9]
for k in ("uniform", "greedy"):
    ax[1].plot([r["h"] for r in sub], [r[k] / r["mip"] for r in sub], ms=3.5, lw=1.2, label=label[k], **style[k])
ax[1].axhline(1.0, color="#bbbbbb", lw=0.8)
ax[1].set_xlabel("node heterogeneity $h$ (0 = identical nodes)")
ax[1].set_ylabel("cost / optimum")
ax[1].set_title("(b) cut $B=0.20$")
ax[1].legend(frameon=False, loc="upper left")

fig.tight_layout(pad=0.4, w_pad=1.5)
fig.savefig(out)
print("wrote", out)
