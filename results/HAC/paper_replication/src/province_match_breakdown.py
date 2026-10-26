"""
province_match_breakdown.py
At the peak-alignment travel-time threshold, which Roman provinces does the HAC
clustering recover well, and which does it fragment or merge away?

For threshold t (default = the ARI-peak, ~78 h) and each province p:
  n_nodes    nodes of p with a cluster label
  recall     fraction of p's nodes that fall in p's single dominant cluster
             ( = how intact the province stays)
  precision  fraction of that dominant cluster that is actually province p
             ( = how province-specific the cluster is; low => p is glued to neighbours)
  f1         harmonic mean of the two
  n_frag     number of clusters holding >=10% of p's nodes (1 = clean)
Sorted by f1.

Outputs:
  data/province_match_<t>h.csv
  maps/province_match_<t>h.png    horizontal bar chart of F1, coloured by regime

Usage:
  /Users/nk821/opt/anaconda3/envs/MIR/bin/python paper_replication/src/province_match_breakdown.py [threshold_hours]
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

ROOT  = Path(__file__).resolve().parents[1]
PKL   = ROOT / "data/network_paper.pkl"
CACHE = ROOT / "data/hac_linkage_average_cache.pkl"

T_H = float(sys.argv[1]) if len(sys.argv) > 1 else 78.2
t   = T_H * 3600

D = pickle.load(open(PKL, "rb"))
nt = D["node_table"]
prov = dict(zip(nt.node_id, nt.province_coarse))
comp_results = pickle.load(open(CACHE, "rb"))

# global cluster label per node
lab, off = {}, 0
for ids, Z in comp_results:
    fl = fcluster(Z, t=t, criterion="distance")
    for nid, l in zip(ids, fl):
        lab[nid] = l + off
    off += int(fl.max())

rows = []
for nid, l in lab.items():
    p = prov.get(nid, "Unknown")
    if p != "Unknown":
        rows.append((p, l))
d = pd.DataFrame(rows, columns=["province", "cluster"])
clus_size = d.cluster.value_counts().to_dict()

out = []
for p, g in d.groupby("province"):
    n = len(g)
    vc = g.cluster.value_counts()
    k = vc.index[0]
    overlap = vc.iloc[0]
    recall = overlap / n
    precision = overlap / clus_size[k]
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    n_frag = int((vc / n >= 0.10).sum())
    out.append(dict(province=p, n_nodes=n, recall=round(recall, 3),
                    precision=round(precision, 3), f1=round(f1, 3), n_frag=n_frag))

res = pd.DataFrame(out).sort_values("f1", ascending=False).reset_index(drop=True)
csv = ROOT / f"data/province_match_{T_H:g}h.csv"
res.to_csv(csv, index=False)

print(f"Threshold {T_H:g} h ({T_H/24:.2f} d),  {d.cluster.nunique()} clusters,  "
      f"{len(res)} provinces\n")
with pd.option_context("display.max_rows", None, "display.width", 120):
    print(res.to_string(index=False))

# ── bar chart ──────────────────────────────────────────────────────────────
r = res.sort_values("f1")
def band(f):
    return "#1a9850" if f >= 0.6 else "#fee08b" if f >= 0.35 else "#d73027"
fig, ax = plt.subplots(figsize=(9, max(6, .3 * len(r))))
ax.barh(r.province, r.f1, color=[band(f) for f in r.f1])
ax.set_xlabel("F1  (recall x precision of the province's dominant cluster)")
ax.set_xlim(0, 1)
ax.set_title(f"How well each Roman province is recovered by the HAC clusters "
             f"(t = {T_H:g} h)")
ax.axvline(0.6, color="#888", lw=.7, ls=":"); ax.axvline(0.35, color="#888", lw=.7, ls=":")
plt.tight_layout()
png = ROOT / f"maps/province_match_{T_H:g}h.png"
plt.savefig(png, dpi=150); plt.close()
print(f"\nCSV  -> {csv}\nPlot -> {png}")
