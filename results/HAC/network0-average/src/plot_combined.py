"""
plot_combined.py
Rebuild only maps/hac_percolation_combined.pdf from the cached linkage matrices,
with a configurable set of snapshot maps in the bottom row.

Default snapshots: 24 h, 48 h, 77 h (best province-alignment threshold), 5 d.

Usage:
  /Users/nk821/opt/anaconda3/envs/MIR/bin/python network0-average/src/plot_combined.py
  /Users/nk821/opt/anaconda3/envs/MIR/bin/python network0-average/src/plot_combined.py 12 24 77 168
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

matplotlib.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
})

ROOT  = Path(__file__).resolve().parents[1]
PKL   = ROOT / "data/network_paper.pkl"
CACHE = ROOT / "data/hac_linkage_average_cache.pkl"
OUT   = ROOT / "maps/hac_percolation_combined.pdf"

MAP_HOURS = [float(x) for x in sys.argv[1:]] or [24, 48, 77.14, 120]
FIT_RANGE_H = (6, 4000)

_PALETTE = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6",
           "#bfef45","#fabed4","#469990","#dcbeff","#9a6324","#fffac8","#800000",
           "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#c0c0c0","#000000"]
def col(c): return _PALETTE[int(c) % len(_PALETTE)]


def hlabel(h: float) -> str:
    if h < 48:
        return f"{h:g}h"
    d = h / 24
    return f"{d:g}d" if abs(d - round(d)) < 0.05 else f"{h:g}h"


D = pickle.load(open(PKL, "rb"))
nodes_4326, edges_4326 = D["nodes_4326"], D["edges_4326"]
comp_results = pickle.load(open(CACHE, "rb"))

all_h = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))

# ── percolation curve on a log grid (visually identical to the full sweep) ──
grid = np.unique(np.clip(np.round(np.geomspace(3600, all_h.max(), 600)), None, all_h.max()))
ncl = np.zeros(len(grid), int); lrg = np.zeros(len(grid), int)
for i, t in enumerate(grid):
    sizes = []
    for ids, Z in comp_results:
        lbl = fcluster(Z, t=t, criterion="distance")
        _, cnt = np.unique(lbl, return_counts=True)
        sizes.extend(cnt.tolist())
    sizes.sort(reverse=True)
    ncl[i] = len(sizes); lrg[i] = sizes[0]

hx = grid / 3600
lh = np.log10(hx[lrg > 0]); ls = np.log10(lrg[lrg > 0])
fm = (lh >= np.log10(FIT_RANGE_H[0])) & (lh <= np.log10(FIT_RANGE_H[1]))
(alpha, log_a), cov = np.polyfit(lh[fm], ls[fm], 1, cov=True)
a = 10 ** log_a; ci = 1.96 * np.sqrt(cov[0, 0])
resid = ls[fm] - (alpha * lh[fm] + log_a)
r2 = 1 - np.sum(resid**2) / np.sum((ls[fm] - ls[fm].mean())**2)


def assign(t: float) -> dict[int, int]:
    out, off = {}, 0
    for ids, Z in comp_results:
        lbl = fcluster(Z, t=t, criterion="distance").tolist()
        for nid, l in zip(ids, lbl):
            out[nid] = l + off
        off += max(lbl)
    return out


fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, len(MAP_HOURS), height_ratios=[2, 1], hspace=.38, wspace=.08)

axc = fig.add_subplot(gs[0, :]); axc2 = axc.twinx()
axc.plot(hx, ncl, color="#2c3e50", lw=1.5, label="N clusters")
axc2.plot(hx, lrg, color="#2980b9", lw=1.5, ls="--", label="Largest cluster")
axc2.plot(hx, a * hx**alpha, color="#85c1e9", lw=1.2, ls=":",
          label=fr"$S_{{\max}}\propto t^{{{alpha:.2f}\pm{ci:.2f}}}$ ($R^2$={r2:.3f})")
for s in (axc, axc2):
    s.set_xscale("log"); s.set_yscale("log")
for i, h in enumerate(MAP_HOURS):
    axc.axvline(h, color="#e74c3c", lw=1.0, alpha=.35)
    y = .97 if i % 2 == 0 else .88
    axc.text(h * 1.03, y, hlabel(h), color="#c0392b", fontsize=18, va="top",
             transform=axc.get_xaxis_transform())
axc.set_xlabel("Travel-time threshold (hours, Tobler)")
axc.set_ylabel("Number of clusters", color="#2c3e50")
axc2.set_ylabel("Largest cluster size (nodes)", color="#2980b9")
h1, l1 = axc.get_legend_handles_labels(); h2, l2 = axc2.get_legend_handles_labels()
axc.legend(h1 + h2, l1 + l2, loc="center right", fontsize=17)
axc.set_title("(a)  HAC percolation - Roman road network (manuscript graph, Tobler travel time)",
              loc="left")

for k, h in enumerate(MAP_HOURS):
    ax = fig.add_subplot(gs[1, k])
    edges_4326.plot(ax=ax, color="#ccc", lw=.25, alpha=.6, rasterized=True)
    cl = assign(h * 3600)
    nn = nodes_4326.copy()
    nn["c"] = nn.node_id.map(cl).fillna(-1).astype(int).map(col)
    nn.plot(ax=ax, color=nn["c"].tolist(), markersize=.8, alpha=.85, rasterized=True)
    gi = int(np.searchsorted(grid, h * 3600))
    gi = min(gi, len(ncl) - 1)
    ax.set_title(f"({'bcdefg'[k]})  t = {hlabel(h)}  ({ncl[gi]} clusters)",
                 loc="left", fontsize=18)
    ax.set_axis_off()

plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Combined PDF -> {OUT}   snapshots: {[hlabel(h) for h in MAP_HOURS]}")
print(f"Scaling law: S_max ~ t^{alpha:.3f}  (95% CI +/-{ci:.3f}, R^2={r2:.3f})")
