"""
hac_percolation.py  (paper-network version)

Hierarchical Agglomerative Clustering on intra-component travel-time distances,
swept across all merge-height thresholds -> percolation structure of the Roman
road network AS BUILT IN THE MANUSCRIPT (Tobler travel time, Itiner-e export,
AWMC AD200 provinces).  This is the paper_replication counterpart of
../../src/hac_percolation.py; it changes only the input graph and the travel-time
model, not the method.

Inputs
  paper_replication/data/network_paper.pkl   (from build_network.py)

Outputs (paper_replication/maps/ and paper_replication/data/)
  maps/hac_percolation_curve.png
  maps/hac_percolation_combined.pdf
  maps/hac_dendrogram_comp{i}.png
  data/hac_clusters.csv           node_id, province, component, cluster_* per snapshot
  data/hac_linkage_average_cache.pkl
  stdout: province-alignment ARI / NMI table + bootstrap CI on the best threshold

Usage (from repo root):
  /Users/nk821/opt/anaconda3/envs/MIR/bin/python paper_replication/src/hac_percolation.py
"""

from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import normalized_mutual_info_score


def adjusted_rand_score(a, b) -> float:
    """Fast ARI via a dense contingency table (sklearn's is dominated by
    type_of_target introspection on large label vectors)."""
    a = np.asarray(a); b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    c = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(c, (ai, bi), 1)
    n = c.sum()
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_c = comb2(c).sum()
    sa = comb2(c.sum(1)).sum(); sb = comb2(c.sum(0)).sum()
    exp = sa * sb / comb2(n)
    mx = 0.5 * (sa + sb)
    return 0.0 if mx == exp else (sum_c - exp) / (mx - exp)

ROOT = Path(__file__).resolve().parents[1]
PKL  = ROOT / "data/network_paper.pkl"
MAPS = ROOT / "maps"
DATA = ROOT / "data"
CACHE = DATA / "hac_linkage_average_cache.pkl"
MAPS.mkdir(exist_ok=True)

LINKAGE_METHOD = "average"     # network travel-time distances -> no Euclidean assumption
MIN_COMP = 3
SNAPSHOT_HOURS = [6, 12, 24, 48, 168, 288]         # 6h .. 12d
FIT_RANGE_H = (6, 4000)

COMP_NAMES = {0: "Continental (Europe-Asia-Africa)", 1: "British Isles", 2: "Cyprus",
              3: "Crete", 4: "Sicily", 5: "Sardinia", 6: "Balearics/other"}


# ── load ─────────────────────────────────────────────────────────────────────
print(f"Loading {PKL}")
with open(PKL, "rb") as f:
    D = pickle.load(f)
G: nx.Graph = D["graph"]
node_prov = dict(zip(D["node_table"].node_id, D["node_table"].province))
nodes_4326 = D["nodes_4326"]
edges_4326 = D["edges_4326"]
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

comps = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)
         if len(c) >= MIN_COMP]
print(f"  {len(comps)} components with >= {MIN_COMP} nodes  "
      f"(sizes: {[len(c) for c in comps]})")


# ── per-component HAC (cached) ───────────────────────────────────────────────
def component_linkage(nodes: set) -> tuple[list[int], np.ndarray]:
    sub = G.subgraph(nodes)
    keys = list(sub.nodes())
    ids  = [G.nodes[k]["node_id"] for k in keys]
    n = len(keys)
    kidx = {k: i for i, k in enumerate(keys)}
    rows, cols, vals = [], [], []
    for u, v, d in sub.edges(data=True):
        rows.append(kidx[u]); cols.append(kidx[v]); vals.append(d["time_s"])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    D_ = dijkstra(A, directed=False)          # C implementation, dense (n, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = linkage(squareform(D_, checks=False), method=LINKAGE_METHOD)
    return ids, Z


if CACHE.exists():
    print(f"Loading linkage cache {CACHE}")
    comp_results = pickle.load(open(CACHE, "rb"))
else:
    comp_results = []
    for i, c in enumerate(comps):
        print(f"  component {i} ({len(c)} nodes) - all-pairs Dijkstra + linkage...", flush=True)
        comp_results.append(component_linkage(c))
        print(f"    max merge height {comp_results[-1][1][:,2].max()/3600:.1f} h")
    pickle.dump(comp_results, open(CACHE, "wb"))
    print(f"Cached -> {CACHE}")


# ── threshold sweep ─────────────────────────────────────────────────────────
all_h = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))
thresholds = np.concatenate([[0.0], all_h])
print(f"\nSweeping {len(thresholds)} thresholds (0 -> {all_h.max()/3600:.0f} h)")

n_clusters = np.zeros(len(thresholds), dtype=int)
largest    = np.zeros(len(thresholds), dtype=int)
for ti, t in enumerate(thresholds):
    sizes = []
    for ids, Z in comp_results:
        if t == 0:
            sizes.extend([1] * len(ids))
        else:
            sizes.extend(Counter(fcluster(Z, t=t, criterion="distance")).values())
    sizes.sort(reverse=True)
    n_clusters[ti] = len(sizes)
    largest[ti] = sizes[0]

# steepest fractional drop in cluster count, restricted to the meaningful regime
# (t > 1 h; below that every merge is a local pair and dominates the raw diff)
_reg = thresholds > 3600
_d = np.diff(np.log(n_clusters[_reg].astype(float))) / np.diff(np.log(thresholds[_reg]))
t_crit = thresholds[_reg][1:][int(np.argmin(_d))]
print(f"Critical threshold (steepest log-log cluster drop): {t_crit/3600:.1f} h")


# ── power-law fit  S_max ~ t^alpha ──────────────────────────────────────────
# Merge heights are far denser at low t, so a raw OLS over every threshold is
# dominated by the 6-50 h region. Fit instead on one representative point per
# log-spaced bin, giving each decade equal weight (standard for power-law OLS).
hrs = thresholds / 3600
m = hrs > 1
hx, ncx, lgx = hrs[m], n_clusters[m], largest[m]
_bins = np.geomspace(FIT_RANGE_H[0], FIT_RANGE_H[1], 60)
_bi = np.clip(np.searchsorted(_bins, hx) - 1, 0, len(_bins) - 2)
_lh_all = np.log10(hx); _ls_all = np.log10(np.maximum(lgx, 1))
lh = np.array([_lh_all[_bi == k].mean() for k in range(len(_bins) - 1) if np.any(_bi == k)])
ls = np.array([_ls_all[_bi == k].mean() for k in range(len(_bins) - 1) if np.any(_bi == k)])
fm = (lh >= np.log10(FIT_RANGE_H[0])) & (lh <= np.log10(FIT_RANGE_H[1]))
(alpha, log_a), cov = np.polyfit(lh[fm], ls[fm], 1, cov=True)
a = 10 ** log_a
ci = 1.96 * np.sqrt(cov[0, 0])
resid = ls[fm] - (alpha * lh[fm] + log_a)
r2 = 1 - np.sum(resid**2) / np.sum((ls[fm] - ls[fm].mean())**2)
print(f"Scaling law: S_max ~ t^{alpha:.3f}  95% CI [{alpha-ci:.3f}, {alpha+ci:.3f}]  R^2={r2:.3f}")


# ── percolation curve ──────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
ax1.plot(hx, ncx, color="#2c3e50", lw=1.5, label="N clusters")
ax2.plot(hx, lgx, color="#2980b9", lw=1.5, ls="--", label="Largest cluster size")
ax2.plot(hx, a * hx**alpha, color="#85c1e9", lw=1.2, ls=":",
         label=fr"Fit $S_{{\max}}\propto t^{{{alpha:.2f}\pm{ci:.2f}}}$ ($R^2$={r2:.3f})")
for s in (ax1, ax2):
    s.set_xscale("log"); s.set_yscale("log")
for h in SNAPSHOT_HOURS:
    ax1.axvline(h, color="#aaa", lw=.8, ls="--", alpha=.7)
    ax1.text(h*1.03, .97, f"{h}h" if h < 48 else f"{h//24}d", color="#888",
             fontsize=8, va="top", transform=ax1.get_xaxis_transform())
ax1.set_xlabel("Travel-time threshold (hours, Tobler)"); ax1.set_ylabel("Number of clusters", color="#2c3e50")
ax2.set_ylabel("Largest cluster size (nodes)", color="#2980b9")
l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc="center right", fontsize=9)
plt.title("HAC percolation - Roman road network (manuscript graph, Tobler travel time)")
plt.tight_layout(); plt.savefig(MAPS / "hac_percolation_curve.png", dpi=150); plt.close()
print(f"Curve -> {MAPS/'hac_percolation_curve.png'}")


# ── dendrograms ────────────────────────────────────────────────────────────
for i, (ids, Z) in enumerate(comp_results):
    n = len(ids)
    Zh = Z.copy(); Zh[:, 2] /= 3600
    trunc = n > 50
    fig, ax = plt.subplots(figsize=(14, 5) if trunc else (max(10, n // 3), 6))
    kw = dict(Z=Zh, ax=ax, color_threshold=0, above_threshold_color="#444",
              link_color_func=lambda _: "#444", no_labels=trunc)
    if trunc:
        kw.update(truncate_mode="lastp", p=min(50, n - 1), show_contracted=True)
    else:
        kw["labels"] = [str(x) for x in ids]
    dendrogram(**kw)
    ax.axhline(t_crit / 3600, color="#e74c3c", lw=1.2, ls="--",
               label=f"critical {t_crit/3600:.1f} h")
    ax.legend(fontsize=9); ax.set_ylabel("Travel time (h)")
    ax.set_title(f"{COMP_NAMES.get(i, f'Component {i}')} - {n} nodes, {LINKAGE_METHOD} linkage")
    plt.tight_layout(); plt.savefig(MAPS / f"hac_dendrogram_comp{i}.png", dpi=150); plt.close()
print(f"Dendrograms -> {MAPS}/hac_dendrogram_comp*.png")


# ── cluster assignments per snapshot ───────────────────────────────────────
def assign(t: float) -> dict[int, int]:
    out, off = {}, 0
    for ids, Z in comp_results:
        lbl = ([1] * len(ids) if t == 0 else
               fcluster(Z, t=t, criterion="distance").tolist())
        for nid, l in zip(ids, lbl):
            out[nid] = l + off
        off += max(lbl) if lbl else 0
    return out

snap = {h: assign(h * 3600) for h in SNAPSHOT_HOURS}

comp_of = {}
for ci, c in enumerate(comps):
    for k in c:
        comp_of[G.nodes[k]["node_id"]] = ci

all_ids = [nid for ids, _ in comp_results for nid in ids]
df = pd.DataFrame({"node_id": all_ids})
df["province"]  = df.node_id.map(node_prov).fillna("Unknown")
df["component"] = df.node_id.map(comp_of)
for h in SNAPSHOT_HOURS:
    df[f"cluster_{h}h" if h < 48 else f"cluster_{h//24}d"] = df.node_id.map(snap[h])
df = df.sort_values("node_id").reset_index(drop=True)
df.to_csv(DATA / "hac_clusters.csv", index=False)
print(f"Clusters CSV -> {DATA/'hac_clusters.csv'}")


# ── combined PDF: curve + 4 static snapshot maps ───────────────────────────
_PALETTE = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6",
           "#bfef45","#fabed4","#469990","#dcbeff","#9a6324","#fffac8","#800000",
           "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#c0c0c0","#000000"]
def col(c): return _PALETTE[int(c) % len(_PALETTE)]

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=.38, wspace=.08)
axc = fig.add_subplot(gs[0, :]); axc2 = axc.twinx()
axc.plot(hx, ncx, color="#2c3e50", lw=1.5, label="N clusters")
axc2.plot(hx, lgx, color="#2980b9", lw=1.5, ls="--", label="Largest cluster")
axc2.plot(hx, a * hx**alpha, color="#85c1e9", lw=1.2, ls=":",
          label=fr"$S_{{\max}}\propto t^{{{alpha:.2f}\pm{ci:.2f}}}$ ($R^2$={r2:.3f})")
for s in (axc, axc2):
    s.set_xscale("log"); s.set_yscale("log")
for h in SNAPSHOT_HOURS:
    axc.axvline(h, color="#aaa", lw=.8, ls="--", alpha=.7)
axc.set_xlabel("Travel-time threshold (hours)"); axc.set_ylabel("N clusters", color="#2c3e50")
axc2.set_ylabel("Largest cluster (nodes)", color="#2980b9")
h1, hl1 = axc.get_legend_handles_labels(); h2, hl2 = axc2.get_legend_handles_labels()
axc.legend(h1 + h2, hl1 + hl2, loc="center right", fontsize=9)
axc.set_title("(a)  HAC percolation - Roman road network (manuscript graph)", loc="left")

for k, h in enumerate([6, 12, 24, 48]):
    ax = fig.add_subplot(gs[1, k])
    edges_4326.plot(ax=ax, color="#ccc", lw=.25, alpha=.6)
    cl = assign(h * 3600)
    nn = nodes_4326.copy()
    nn["c"] = nn.node_id.map(cl).fillna(-1).astype(int).map(col)
    nn.plot(ax=ax, color=nn["c"].tolist(), markersize=.8, alpha=.85)
    idx = min(int(np.searchsorted(thresholds, h * 3600)), len(n_clusters) - 1)
    ax.set_title(f"({'bcde'[k]})  t = {h}h  ({n_clusters[idx]} clusters)", loc="left", fontsize=10)
    ax.set_axis_off()
plt.savefig(MAPS / "hac_percolation_combined.pdf", dpi=150, bbox_inches="tight"); plt.close()
print(f"Combined PDF -> {MAPS/'hac_percolation_combined.pdf'}")


# ── province alignment: snapshot ARI/NMI ───────────────────────────────────
known = df[df.province != "Unknown"]
pl = known.province.values
print(f"\nProvince alignment  ({len(known)} nodes with known province)")
print(f"{'threshold':<10}{'N clust':>9}{'ARI':>9}{'NMI':>9}")
best = (-1, None)
for h in SNAPSHOT_HOURS:
    col_ = f"cluster_{h}h" if h < 48 else f"cluster_{h//24}d"
    ari = adjusted_rand_score(pl, known[col_].values)
    nmi = normalized_mutual_info_score(pl, known[col_].values, average_method="arithmetic")
    lab = f"{h}h" if h < 48 else f"{h//24}d"
    print(f"{lab:<10}{df[col_].nunique():>9}{ari:>9.4f}{nmi:>9.4f}")
    if ari > best[0]:
        best = (ari, lab)
print(f"Best snapshot: {best[1]}  ARI={best[0]:.4f}")


# ── best-threshold search over ALL merge heights + bootstrap CI ─────────────
print("\nExhaustive best-threshold search (ARI vs AWMC provinces) + bootstrap CI...")
kn_ids = known.node_id.tolist()
idx_of = {nid: i for i, nid in enumerate(kn_ids)}
cand = all_h                                  # every merge height
label_mat = np.full((len(cand), len(kn_ids)), -1, dtype=np.int32)
for ids, Z in comp_results:
    cols = [(idx_of[n], k) for k, n in enumerate(ids) if n in idx_of]
    if not cols:
        continue
    loc = np.array([c[1] for c in cols]); gl = np.array([c[0] for c in cols])
    off = int(label_mat.max() + 1)
    for ti, t in enumerate(cand):
        lab = fcluster(Z, t=t, criterion="distance").astype(np.int32)
        label_mat[ti, gl] = lab[loc] + off
prov_arr = known.set_index("node_id").loc[kn_ids, "province"].values
ari_full = np.array([adjusted_rand_score(prov_arr, label_mat[ti]) for ti in range(len(cand))])
bti = int(np.argmax(ari_full))
print(f"  best threshold {cand[bti]/3600:.2f} h ({cand[bti]/86400:.2f} d)  ARI={ari_full[bti]:.4f}")

TOPK = 100
top = np.argsort(ari_full)[-TOPK:]
rng = np.random.default_rng(42)
NBOOT = 500
N = len(kn_ids); bt, ba = np.empty(NBOOT), np.empty(NBOOT)
for b in range(NBOOT):
    s = rng.integers(0, N, N); pb = prov_arr[s]
    ab = np.array([adjusted_rand_score(pb, label_mat[k][s]) for k in top])
    j = int(np.argmax(ab)); bt[b] = cand[top[j]] / 3600; ba[b] = ab[j]
tlo, thi = np.percentile(bt, [2.5, 97.5]); alo, ahi = np.percentile(ba, [2.5, 97.5])
print(f"  95% CI threshold [{tlo:.1f} h, {thi:.1f} h]  ([{tlo/24:.2f} d, {thi/24:.2f} d])")
print(f"  95% CI ARI       [{alo:.4f}, {ahi:.4f}]")
print("\nDone.")
