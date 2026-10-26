"""
apply_hac_to_nulls_v2.py — OPTIMIZED VERSION

Apply itinereX's exact hierarchical agglomerative clustering (HAC) code to KNN and Delaunay
null models. Produces the same outputs: cluster assignments CSV at multiple thresholds.

OPTIMIZATION: Pre-computes all-pairs shortest paths upfront, then applies HAC per-component
using the cached distances. This is much faster than recomputing distances per component,
which is critical for graphs with many isolated components.

Usage (from repo root):
  python scripts/apply_hac_to_nulls_v2.py --model knn
  python scripts/apply_hac_to_nulls_v2.py --model delaunay
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
from collections import Counter

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Apply HAC to spatial null models (optimized)")
parser.add_argument(
    "--model",
    choices=[
        "knn",
        "delaunay",
        "knn_mst",
        "delaunay_mst",
        "province_knn_mst_bridged",
        "province_delaunay_mst_bridged",
    ],
    required=True,
    help="Which null model to analyze",
)
parser.add_argument("--output-dir", default="results/Null-models/data/processed/hac_null_analysis",
                    help="Output directory for results")
args = parser.parse_args()

model_name = args.model

MODEL_PATHS = {
    "knn": Path("results/Null-models/data/processed/global_spatial_null_knn_uniformrange.pkl"),
    "delaunay": Path("results/Null-models/data/processed/global_spatial_null_delaunay_uniformrange.pkl"),
    "knn_mst": Path("results/Null-models/data/processed/global_spatial_null_knn_mst_uniformrange.pkl"),
    "delaunay_mst": Path("results/Null-models/data/processed/global_spatial_null_delaunay_mst_uniformrange.pkl"),
    "province_knn_mst_bridged": Path("results/Null-models/data/processed/province_spatial_null_knn_mst_bridged_uniformrange.pkl"),
    "province_delaunay_mst_bridged": Path("results/Null-models/data/processed/province_spatial_null_delaunay_mst_bridged_uniformrange.pkl"),
}

MODEL_TITLES = {
    "knn": "Global KNN",
    "delaunay": "Global Delaunay",
    "knn_mst": "Global KNN+MST",
    "delaunay_mst": "Global Delaunay+MST",
    "province_knn_mst_bridged": "Province-Bridged KNN+MST",
    "province_delaunay_mst_bridged": "Province-Bridged Delaunay+MST",
}

MODEL_OUTPUT_DIRS = {
    "knn": "null_knn",
    "delaunay": "null_delaunay",
    "knn_mst": "null_knn_mst",
    "delaunay_mst": "null_delaunay_mst",
    # Reuse existing writable HAC directories; filenames remain distinct.
    "province_knn_mst_bridged": "null_knn_mst",
    "province_delaunay_mst_bridged": "null_delaunay_mst",
}

output_dir = Path(args.output_dir) / MODEL_OUTPUT_DIRS[model_name]
output_dir.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
NULL_PICKLE = MODEL_PATHS[model_name]
OUT_CSV = output_dir / f"hac_clusters_{model_name}.csv"
CACHE_PATH = output_dir / f"hac_linkage_average_cache_{model_name}.pkl"
OUT_CURVE = output_dir / f"hac_percolation_curve_{model_name}.png"
OUT_COMBINED = output_dir / f"hac_percolation_combined_{model_name}.png"

LINKAGE_METHOD = "average"
# Focus on broader timescales: 24h, 48h, and lines beyond 10^2 hours.
SNAPSHOT_HOURS = [48, 168, 288, 1200]
SNAPSHOTS = [(h, h * 3600) for h in SNAPSHOT_HOURS]
HIGHLIGHT_HOURS = [h for h in SNAPSHOT_HOURS if h == 48 or h > 100]

print(f"\n{'='*70}")
print(f"HAC Analysis on Spatial Null Model ({model_name.upper()})")
print(f"{'='*70}")

# ── Load null model ───────────────────────────────────────────────────────────
print(f"\nLoading null model from: {NULL_PICKLE}")
with open(NULL_PICKLE, "rb") as f:
    null_data = pickle.load(f)

G_null = null_data["graph"]
meta = null_data.get("meta", {})

print(f"  Nodes: {len(G_null.nodes())}")
print(f"  Edges: {len(G_null.edges())}")
print(f"  Model: {meta.get('model', 'unknown')}")
print(f"  Scope: {meta.get('scope', 'unknown')}")

# Load province map from original itinereX
PROVINCE_CSV = Path("data_cleaning/cleaned_data/G_clean_weighted_edges_with_province.csv")
province_labels_dict = None

if PROVINCE_CSV.exists():
    print(f"\nLoading province assignments...")
    prov_df = pd.read_csv(PROVINCE_CSV)
    within = prov_df[~prov_df["is_crossing"] & ~prov_df["is_outside"] & prov_df["province"].notna()].copy()
    
    node_provs = {}
    for row in within.itertuples(index=False):
        for nid in (int(row.u_id), int(row.v_id)):
            node_provs.setdefault(nid, Counter())[str(row.province)] += 1
    
    province_labels_dict = {nid: c.most_common(1)[0][0] for nid, c in node_provs.items()}
    print(f"  Loaded {len(province_labels_dict)} nodes with province assignments")

# ── Connected components ───────────────────────────────────────────────────────
print("\n Identifying connected components...")
comps = sorted(nx.connected_components(G_null), key=len, reverse=True)
print(f"  {len(comps)} total components")
for i, c in enumerate(comps[:5]):
    print(f"    [{i}]  {len(c)} nodes")
if len(comps) > 5:
    print(f"    ... and {len(comps) - 5} more")

comps = [c for c in comps if len(c) >= 3]
print(f"  Keeping {len(comps)} components with >=3 nodes")

# ── Check cache ───────────────────────────────────────────────────────────────
if CACHE_PATH.exists():
    print(f"\nLoading cached linkage from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        comp_results = pickle.load(f)
    print(f"  Loaded {len(comp_results)} components")
else:
    print(f"\nComputing all-pairs shortest paths on {len(G_null.nodes())} nodes...")
    print("  (This may take 2-5 minutes...)")
    lengths = dict(nx.all_pairs_dijkstra_path_length(G_null, weight="time_s"))
    print(f"  Distance cache complete. Processing {len(comps)} components...")
    
    comp_results = []
    for idx, comp in enumerate(comps):
        n = len(comp)
        
        # Progress every 10% or for first 5
        if (idx + 1) % max(1, len(comps) // 10) == 0 or idx < 5:
            print(f"    [{idx+1:6d}/{len(comps)}]  {n:6d} nodes", flush=True)
        
        node_keys = sorted(list(comp))
        # Null-model graphs are keyed directly by original node IDs.
        # If node_id attr is missing, use the node key itself to avoid collisions.
        node_ids = [int(G_null.nodes[n].get("node_id", n)) for n in node_keys]
        
        # Build distance matrix from pre-computed all-pairs paths
        D = np.zeros((n, n), dtype=np.float64)
        for i, u in enumerate(node_keys):
            for j, v in enumerate(node_keys):
                if i != j:
                    D[i, j] = lengths[u].get(v, np.inf)
        
        # HAC linkage
        condensed = squareform(D, checks=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = linkage(condensed, method=LINKAGE_METHOD)
        
        comp_results.append((node_ids, Z))
    
    print(f"\nSaving linkage cache to {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(comp_results, f)

print(f"  Processing complete. {len(comp_results)} components ready for threshold sweep.")

# ── Threshold sweep ───────────────────────────────────────────────────────────
print(f"\nThreshold sweep...")
all_heights = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))
thresholds = np.concatenate([[0.0], all_heights])

print(f"  Sweeping {len(thresholds)} unique thresholds "
      f"(0 to {all_heights.max()/3600:.1f}h)")

n_clusters_arr = np.zeros(len(thresholds), dtype=int)
largest_arr = np.zeros(len(thresholds), dtype=int)

for ti, t in enumerate(thresholds):
    sizes = []
    for node_ids, Z in comp_results:
        n = len(node_ids)
        if t == 0:
            sizes.extend([1] * n)
        else:
            labels = fcluster(Z, t=t, criterion="distance")
            counts = Counter(labels)
            sizes.extend(counts.values())
    
    sizes_sorted = sorted(sizes, reverse=True)
    n_clusters_arr[ti] = len(sizes_sorted)
    largest_arr[ti] = sizes_sorted[0] if sizes_sorted else 0

print(f"  Done.")

# ── Percolation curve (itinereX style) ───────────────────────────────────────
hours = thresholds / 3600
mask = hours > 1
hours_x = hours[mask]
n_clusters_arr_x = n_clusters_arr[mask]
largest_arr_x = largest_arr[mask]

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.plot(hours_x, n_clusters_arr_x, color="#2c3e50", linewidth=1.5, label="N clusters")
ax2.plot(hours_x, largest_arr_x, color="#2980b9", linewidth=1.5, linestyle="--", label="Largest cluster size")
ax1.set_xscale("log")
ax2.set_xscale("log")
ax1.set_yscale("log")
ax2.set_yscale("log")

ax1.set_ylabel("Number of clusters", color="#2c3e50", fontsize=11)
ax2.set_ylabel("Largest cluster size (nodes)", color="#2980b9", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#2c3e50")
ax2.tick_params(axis="y", labelcolor="#2980b9")
ax1.set_xlabel("Distance threshold (travel time, hours)", fontsize=11)

for h, _ in SNAPSHOTS:
    ax1.axvline(h, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.7)
for h in HIGHLIGHT_HOURS:
    ax1.axvline(h, color="#e74c3c", linewidth=1.0, alpha=0.35)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

plt.title(f"HAC Percolation - {MODEL_TITLES[model_name]} Null Model", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_CURVE, dpi=150)
plt.close()
print(f"  Percolation curve saved -> {OUT_CURVE}")

# ── Cluster assignment at each snapshot ───────────────────────────────────────
def assign_clusters(comp_results, t: float, offset: int = 0) -> dict[int, int]:
    """Assign cluster labels at threshold t (exact itinereX code)."""
    assignment = {}
    cluster_offset = offset
    for node_ids, Z in comp_results:
        n = len(node_ids)
        if t == 0 or n == 1:
            labels = list(range(1, n + 1))
        else:
            labels = fcluster(Z, t=t, criterion="distance").tolist()
        for nid, lbl in zip(node_ids, labels):
            assignment[nid] = lbl + cluster_offset
        cluster_offset += int(max(labels)) if labels else 0
    return assignment


_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#c0c0c0", "#000000",
]


def cluster_color(cluster_id: int) -> str:
    return _PALETTE[cluster_id % len(_PALETTE)]

# ── Snapshot cluster assignments ──────────────────────────────────────────────
print(f"\nSnapshot assignments:")
snapshot_clusters = {}
for h, t_s in SNAPSHOTS:
    snapshot_clusters[h] = assign_clusters(comp_results, t_s)
    idx = min(int(np.searchsorted(thresholds, t_s, side="left")), len(n_clusters_arr) - 1)
    print(f"  {h:3d}h -> {n_clusters_arr[idx]:6d} clusters")


# Combined percolation figure (curve + 4 static cluster maps)
MAP_HOURS = HIGHLIGHT_HOURS

# Precompute edge segments and node coordinates for fast plotting
def _node_id_and_xy(node_key) -> tuple[int, tuple[float, float]]:
    attrs = G_null.nodes[node_key]
    node_id = int(attrs.get("node_id", node_key))
    if "x" in attrs and "y" in attrs:
        return node_id, (float(attrs["x"]), float(attrs["y"]))
    if isinstance(node_key, tuple) and len(node_key) >= 2:
        return node_id, (float(node_key[0]), float(node_key[1]))
    return node_id, (np.nan, np.nan)


node_xy = {}
for n in G_null.nodes():
    node_id, xy = _node_id_and_xy(n)
    node_xy[node_id] = xy

edge_segments = []
for u, v in G_null.edges():
    u_id = int(G_null.nodes[u].get("node_id", u))
    v_id = int(G_null.nodes[v].get("node_id", v))
    if u_id in node_xy and v_id in node_xy:
        x1, y1 = node_xy[u_id]
        x2, y2 = node_xy[v_id]
        if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
            edge_segments.append([(x1, y1), (x2, y2)])


def _snapshot_arrays(cluster_map: dict[int, int]) -> tuple[np.ndarray, np.ndarray, list]:
    xs, ys, colors = [], [], []
    for nid, cid in cluster_map.items():
        if nid in node_xy:
            x, y = node_xy[nid]
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
                colors.append(cluster_color(cid))
    return np.asarray(xs), np.asarray(ys), colors


fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.38, wspace=0.08)

# Top row: percolation curve
ax1c = fig.add_subplot(gs[0, :])
ax2c = ax1c.twinx()

ax1c.plot(hours_x, n_clusters_arr_x, color="#2c3e50", linewidth=1.5, label="N clusters")
ax2c.plot(hours_x, largest_arr_x, color="#2980b9", linewidth=1.5, linestyle="--", label="Largest cluster size")
ax1c.set_xscale("log")
ax2c.set_xscale("log")
ax1c.set_yscale("log")
ax2c.set_yscale("log")
ax1c.set_ylabel("Number of clusters", color="#2c3e50", fontsize=10)
ax2c.set_ylabel("Largest cluster size (nodes)", color="#2980b9", fontsize=10)
ax1c.tick_params(axis="y", labelcolor="#2c3e50")
ax2c.tick_params(axis="y", labelcolor="#2980b9")
ax1c.set_xlabel("Distance threshold (travel time, hours)", fontsize=10)

for h, _ in SNAPSHOTS:
    ax1c.axvline(h, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.7)
for h in MAP_HOURS:
    ax1c.axvline(h, color="#e74c3c", linewidth=1.0, alpha=0.35)

_l1, _lb1 = ax1c.get_legend_handles_labels()
_l2, _lb2 = ax2c.get_legend_handles_labels()
ax1c.legend(_l1 + _l2, _lb1 + _lb2, loc="center right", fontsize=9)
ax1c.set_title(f"(a) HAC Percolation - {MODEL_TITLES[model_name]} Null", fontsize=12, loc="left")

# Bottom row: static cluster maps
subplot_labels = ["(b)", "(c)", "(d)", "(e)"]
for col, (h, sublabel) in enumerate(zip(MAP_HOURS, subplot_labels)):
    ax_map = fig.add_subplot(gs[1, col])

    lc = LineCollection(edge_segments, colors="#cccccc", linewidths=0.25, alpha=0.6)
    ax_map.add_collection(lc)

    clust_map = snapshot_clusters[h]
    xs, ys, cols = _snapshot_arrays(clust_map)
    if len(xs) > 0:
        ax_map.scatter(xs, ys, c=cols, s=1.8, alpha=0.85, linewidths=0)

    label = f"{h}h" if h < 48 else f"{h//24}d"
    idx = min(int(np.searchsorted(thresholds, h * 3600, side="left")), len(n_clusters_arr) - 1)
    ax_map.set_title(f"{sublabel}  t = {label}  ({n_clusters_arr[idx]} clusters)", fontsize=10, loc="left")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_axis_off()

plt.savefig(OUT_COMBINED, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Combined figure saved -> {OUT_COMBINED}")

# ── CSV output ────────────────────────────────────────────────────────────────
print(f"\nGenerating output CSV...")
all_node_ids = list({nid for node_ids, _ in comp_results for nid in node_ids})

df = pd.DataFrame({"node_id": all_node_ids})

if province_labels_dict:
    df["province"] = df["node_id"].map(province_labels_dict).fillna("Unknown")
else:
    df["province"] = "Unknown"

for h, t_s in SNAPSHOTS:
    col = f"cluster_{h}h" if h < 48 else f"cluster_{h // 24}d"
    df[col] = df["node_id"].map(snapshot_clusters[h]).fillna(-1).astype(int)

df = df.sort_values("node_id").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"  CSV saved -> {OUT_CSV}  ({len(df)} nodes)")

# ── Province alignment (ARI/NMI) ───────────────────────────────────────────────
if province_labels_dict and len(province_labels_dict) > 0:
    print(f"\n-- Province alignment --")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    cluster_cols = [
        (h, f"cluster_{h}h" if h < 48 else f"cluster_{h // 24}d")
        for h, _ in SNAPSHOTS
    ]
    
    known = df[df["province"] != "Unknown"].copy()
    if len(known) > 0:
        province_labels = known["province"].values
        
        print(f"{'Threshold':<12} {'N clusters':>12} {'ARI':>8} {'NMI':>8}")
        print("-" * 44)
        
        best_ari = -np.inf
        best_col_label = ""
        
        for h, col in cluster_cols:
            cluster_labels = known[col].values
            ari = adjusted_rand_score(province_labels, cluster_labels)
            nmi = normalized_mutual_info_score(
                province_labels, cluster_labels,
                average_method="arithmetic"
            )
            n_clust = df[col].nunique()
            label = f"{h}h" if h < 48 else f"{h // 24}d"
            
            marker = " < best ARI" if ari > best_ari else ""
            print(f"{label:<12} {n_clust:>12} {ari:>8.4f} {nmi:>8.4f}{marker}")
            
            if ari > best_ari:
                best_ari = ari
                best_col_label = label
        
        print("-" * 44)
        print(f"Best province match: {best_col_label}  (ARI = {best_ari:.4f})")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Summary:")
print(f"  Model: {MODEL_TITLES[model_name]}")
print(f"  Nodes: {len(G_null.nodes())}")
print(f"  Edges: {len(G_null.edges())}")
print(f"  Components (>=3 nodes): {len(comp_results)}")
print(f"  CSV output: {OUT_CSV}")
print(f"  Curve output: {OUT_CURVE}")
print(f"{'='*70}\n")
