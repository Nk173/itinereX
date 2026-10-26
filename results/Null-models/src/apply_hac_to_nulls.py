"""
apply_hac_to_nulls.py

Apply itinereX's exact hierarchical agglomerative clustering (HAC) code to KNN and Delaunay
null models. Produces the same outputs: percolation curves, snapshot maps, and cluster CSVs.

The null models are global (single graph on 8460 nodes), so component handling is simpler
than the original itinereX code (no per-component isolation).

Usage (from repo root):
  python scripts/apply_hac_to_nulls.py --model knn
  python scripts/apply_hac_to_nulls.py --model delaunay
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
from collections import Counter

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Apply HAC to spatial null models")
parser.add_argument("--model", choices=["knn", "delaunay"], required=True,
                    help="Which null model to analyze")
parser.add_argument("--output-dir", default="results/Null-models/data/processed/hac_null_analysis",
                    help="Output directory for results")
args = parser.parse_args()

model_name = args.model
output_dir = Path(args.output_dir) / f"null_{model_name}"
output_dir.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
NULL_PICKLE = Path(f"results/Null-models/data/processed/global_spatial_null_{model_name}_uniformrange.pkl")
OUT_CSV = output_dir / f"hac_clusters_{model_name}.csv"
CACHE_PATH = output_dir / f"hac_linkage_average_cache_{model_name}.pkl"

LINKAGE_METHOD = "average"  # No Euclidean assumptions — appropriate for network distances
SNAPSHOT_HOURS = [6, 12, 24, 48, 168, 288]  # 168h = 7 days, 288h = 12 days
SNAPSHOTS = [(h, h * 3600) for h in SNAPSHOT_HOURS]

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

# Load province map from original itinereX (if available for alignment scoring)
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


# ── HAC per component (single global graph for null models) ──────────────────
def component_linkage(G: nx.Graph, nodes: set) -> tuple[list, np.ndarray]:
    """Compute pairwise travel-time distances within component and run HAC.

    Exact copy of itinereX component_linkage function.

    Returns
    -------
    node_ids   list of node_id ints in matrix row order
    Z          scipy linkage matrix  (n-1, 4)
    """
    sub = G.subgraph(nodes)
    node_keys = list(sub.nodes())
    
    # Try to get node_id attribute; if not present, use node key itself
    try:
        node_ids = [G.nodes[n]["node_id"] for n in node_keys]
    except (KeyError, AttributeError):
        node_ids = list(range(len(node_keys)))
    
    n = len(node_keys)

    # All-pairs shortest path (travel time)
    print(f"    Computing {n} x {n} distance matrix...")
    lengths = dict(nx.all_pairs_dijkstra_path_length(sub, weight="time_s"))

    # Build symmetric distance matrix
    D = np.zeros((n, n), dtype=np.float64)
    for i, u in enumerate(node_keys):
        for j, v in enumerate(node_keys):
            if i != j:
                D[i, j] = lengths[u].get(v, np.inf)

    condensed = squareform(D, checks=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = linkage(condensed, method=LINKAGE_METHOD)

    return node_ids, Z


# For null models: single connected component (full global graph)
print("\nIdentifying connected components...")
comps = sorted(nx.connected_components(G_null), key=len, reverse=True)
print(f"  {len(comps)} connected component(s)")
for i, c in enumerate(comps[:10]):  # Print first 10
    print(f"    [{i:2d}]  {len(c):5d} nodes")

# Drop singleton and 2-node components (HAC is trivial for these)
comps = [c for c in comps if len(c) >= 3]

print(f"\nRunning HAC on {len(comps)} component(s) with ≥3 nodes...")

# Check for cache
if CACHE_PATH.exists():
    print(f"Loading cached linkage results from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        comp_results = pickle.load(f)
    print(f"  Loaded {len(comp_results)} components")
    for i, (node_ids, Z) in enumerate(comp_results):
        print(f"  [{i:2d}]  {len(node_ids):5d} nodes  "
              f"(max merge: {Z[:, 2].max()/3600:.2f}h)")
else:
    comp_results = []
    for i, comp in enumerate(comps):
        n = len(comp)
        print(f"  Component {i}  ({n} nodes)")
        node_ids, Z = component_linkage(G_null, comp)
        comp_results.append((node_ids, Z))
        print(f"    done  (max merge height: {Z[:, 2].max():.0f}s = {Z[:, 2].max()/3600:.2f}h)")

    print(f"\nSaving linkage cache → {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(comp_results, f)


# ── Threshold sweep ───────────────────────────────────────────────────────────
# Collect every unique merge height across all components as threshold candidates.
all_heights = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))
thresholds = np.concatenate([[0.0], all_heights])

print(f"\nSweeping {len(thresholds)} thresholds "
      f"(0 → {all_heights.max()/3600:.1f}h)...")

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


# ── Cluster assignment at each snapshot (exact itinereX code) ────────────────
def assign_clusters(comp_results, t: float, offset: int = 0) -> dict[int, int]:
    """Return node_id → global cluster label at threshold t.
    
    Exact copy of itinereX assign_clusters function.
    """
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


# ── Cluster assignments at each snapshot ──────────────────────────────────────
snapshot_clusters = {}  # hours → {node_id: cluster}
for h, t_s in SNAPSHOTS:
    snapshot_clusters[h] = assign_clusters(comp_results, t_s)
    # Find index to look up cluster count
    idx = min(int(np.searchsorted(thresholds, t_s, side="left")), len(n_clusters_arr) - 1)
    print(f"  {h:3d}h → {n_clusters_arr[idx]:6d} clusters")


# ── CSV output (exact itinereX format) ────────────────────────────────────────
print(f"\nSaving cluster assignments CSV...")
all_node_ids = list({nid for node_ids, _ in comp_results for nid in node_ids})

df = pd.DataFrame({"node_id": all_node_ids})

# Add province labels if available
if province_labels_dict:
    df["province"] = df["node_id"].map(province_labels_dict).fillna("Unknown")
else:
    df["province"] = "Unknown"

# Add cluster assignments at each snapshot
for h, t_s in SNAPSHOTS:
    col = f"cluster_{h}h" if h < 48 else f"cluster_{h // 24}d"
    df[col] = df["node_id"].map(snapshot_clusters[h]).fillna(-1).astype(int)

df = df.sort_values("node_id").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"  CSV → {OUT_CSV}  ({len(df)} nodes)")
print(f"  Columns: {list(df.columns)}")


# ── Province alignment scores (ARI/NMI) ──────────────────────────────────────
if province_labels_dict and len(province_labels_dict) > 0:
    print(f"\n── Province alignment (computed on known nodes) ──")
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    cluster_cols = [
        (h, f"cluster_{h}h" if h < 48 else f"cluster_{h // 24}d")
        for h, _ in SNAPSHOTS
    ]
    
    # Filter to nodes with known province
    known = df[df["province"] != "Unknown"].copy()
    if len(known) > 0:
        province_labels = known["province"].values
        
        print(f"{'Threshold':<12} {'N clusters':>12} {'ARI':>8} {'NMI':>8}")
        print("─" * 44)
        
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
            
            marker = " ◀ best ARI" if ari > best_ari else ""
            print(f"{label:<12} {n_clust:>12} {ari:>8.4f} {nmi:>8.4f}{marker}")
            
            if ari > best_ari:
                best_ari = ari
                best_col_label = label
        
        print("─" * 44)
        print(f"Best province match: {best_col_label}  (ARI = {best_ari:.4f})")
else:
    print("\n(Province alignment skipped — no province labels available)")


# ── Summary statistics ────────────────────────────────────────────────────────
print(f"\n── Summary ──")
print(f"Model: {model_name.upper()}")
print(f"Nodes: {len(G_null.nodes())}")
print(f"Edges: {len(G_null.edges())}")
print(f"Connected components (≥3 nodes): {len(comp_results)}")
print(f"Unique thresholds swept: {len(thresholds)}")
print(f"Output CSV: {OUT_CSV}")
print(f"\n{'='*70}\n")
