"""
province_alignment_curve.py
How well do the HAC travel-time clusters match the Roman provincial map, as a
function of the travel-time threshold?

For a log-spaced sweep of thresholds it cuts every component's linkage tree,
builds the global clustering, and scores it against the AWMC "Roman Empire AD 200"
provinces (from the cloned repo) with:
  ARI  - Adjusted Rand Index          (chance-corrected pair agreement)
  NMI  - Normalised Mutual Information (arithmetic)
  hom  - homogeneity   (each cluster falls within one province)
  com  - completeness  (each province falls within one cluster)
  V    - their harmonic mean
Scored twice: against the 58 fine units (Italian regiones separate) and against
the coarse set (regiones + Roma merged into "Italia").

Outputs (paper_replication/):
  data/province_alignment.csv           threshold_h, n_clusters, ari/nmi/... (fine + coarse)
  maps/province_alignment_curve.png     ARI + NMI vs threshold, peak marked

Usage:
  /Users/nk821/opt/anaconda3/envs/MIR/bin/python paper_replication/src/province_alignment_curve.py
"""
from __future__ import annotations
import pickle
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
OUT_CSV = ROOT / "data/province_alignment.csv"
OUT_PNG = ROOT / "maps/province_alignment_curve.png"

N_THRESH = 400
SNAPSHOT_HOURS = [6, 12, 24, 48, 168, 288]


# ── fast partition-comparison metrics (contingency table) ───────────────────
def _contingency(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    c = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.float64)
    np.add.at(c, (ai, bi), 1.0)
    return c

def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return -np.sum(p * np.log(p))

def scores(clusters, truth) -> dict:
    c = _contingency(clusters, truth)
    n = c.sum()
    # ARI (rows = clusters, cols = provinces)
    comb2 = lambda x: x * (x - 1) / 2.0
    sc = comb2(c).sum(); sa = comb2(c.sum(1)).sum(); sb = comb2(c.sum(0)).sum()
    exp = sa * sb / comb2(n); mx = 0.5 * (sa + sb)
    ari = 0.0 if mx == exp else (sc - exp) / (mx - exp)
    # mutual information + marginal entropies
    pij = c / n
    pi = pij.sum(1, keepdims=True)          # cluster marginal
    pj = pij.sum(0, keepdims=True)          # province marginal
    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / (pi * pj)[nz]))
    h_clus  = _entropy(c.sum(1))
    h_truth = _entropy(c.sum(0))
    nmi = 0.0 if (h_clus + h_truth) == 0 else 2 * mi / (h_clus + h_truth)
    hom = 1.0 if h_truth == 0 else mi / h_truth   # clusters pure w.r.t. provinces
    com = 1.0 if h_clus  == 0 else mi / h_clus    # provinces intact within clusters
    v   = 0.0 if (hom + com) == 0 else 2 * hom * com / (hom + com)
    return dict(ari=ari, nmi=nmi, hom=hom, com=com, v=v)


# ── load ───────────────────────────────────────────────────────────────────
D = pickle.load(open(PKL, "rb"))
nt = D["node_table"]
prov_fine   = dict(zip(nt.node_id, nt.province))
prov_coarse = dict(zip(nt.node_id, nt.province_coarse))
comp_results = pickle.load(open(CACHE, "rb"))

all_ids = [nid for ids, _ in comp_results for nid in ids]
mask = np.array([prov_fine.get(i, "Unknown") != "Unknown" for i in all_ids])
kept = np.array(all_ids)[mask]
tf = np.array([prov_fine[i]   for i in kept])
tc = np.array([prov_coarse[i] for i in kept])
pos = {nid: k for k, nid in enumerate(kept)}
print(f"{len(kept)} / {len(all_ids)} nodes with a known province")
print(f"  {len(set(tf))} fine units, {len(set(tc))} coarse units")

all_h = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))
thr = np.unique(np.geomspace(3600, all_h.max(), N_THRESH))


def assign(t):
    out = np.full(len(kept), -1, dtype=np.int64)
    off = 0
    for ids, Z in comp_results:
        lbl = fcluster(Z, t=t, criterion="distance")
        for nid, l in zip(ids, lbl):
            p = pos.get(nid)
            if p is not None:
                out[p] = l + off
        off += int(lbl.max())
    return out


rows = []
for t in thr:
    cl = assign(t)
    sf = scores(cl, tf); sc = scores(cl, tc)
    rows.append(dict(threshold_h=t / 3600, n_clusters=len(np.unique(cl)),
                     **{f"{k}_fine": v for k, v in sf.items()},
                     **{f"{k}_coarse": v for k, v in sc.items()}))
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"CSV -> {OUT_CSV}")

pk = df.loc[df.ari_coarse.idxmax()]
print(f"Peak ARI (coarse): {pk.ari_coarse:.3f} at {pk.threshold_h:.1f} h "
      f"({pk.threshold_h/24:.2f} d), {int(pk.n_clusters)} clusters")

# ── plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
x = df.threshold_h
ax.plot(x, df.ari_coarse, color="#c0392b", lw=2.0)
ax.set_xscale("log")
ax.set_xlabel("Travel-time threshold (hours)")
ax.set_ylabel("Adjusted Rand Index")
ax.set_ylim(0, 1)
for h in SNAPSHOT_HOURS:
    ax.axvline(h, color="#ccc", lw=.8, ls="--", alpha=.7)
    ax.text(h * 1.03, .98, f"{h}h" if h < 48 else f"{h//24}d", color="#999",
            fontsize=8, va="top", transform=ax.get_xaxis_transform())
ax.axvline(pk.threshold_h, color="#c0392b", lw=1.0, alpha=.4)
ax.annotate(f"peak ARI {pk.ari_coarse:.2f}  @ {pk.threshold_h:.0f} h ({pk.threshold_h/24:.1f} d)",
            xy=(pk.threshold_h, pk.ari_coarse), xytext=(pk.threshold_h * 1.6, pk.ari_coarse - .10),
            fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=.8))
ax.set_title("HAC travel-time clusters vs. Roman provinces (AWMC AD 200)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"Plot -> {OUT_PNG}")
