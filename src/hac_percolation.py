"""
hac_percolation.py
Hierarchical Agglomerative Clustering on intra-component travel-time distances,
swept across all distance thresholds to identify percolation structure.

HAC is run independently within each connected component (nodes in different
components have no path between them and cannot share a cluster). The linkage
matrix from each component encodes the full merge history, so the threshold
sweep is just repeated cheap cuts — no re-running of HAC.

Outputs (maps/ and data/):
  maps/hac_dendrogram_comp{i}.png  – dendrogram per connected component
  maps/hac_snapshot_low.html       – many clusters (early merges, local structure)
  maps/hac_snapshot_crit.html      – estimated critical threshold
  maps/hac_snapshot_high.html      – few clusters (late merges, regional structure)
  maps/hac_percolation_curve.png
  data/hac_clusters.csv            – node_id, province, component, cluster at each snapshot

Usage (from repo root):
  python src/hac_percolation.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
from collections import Counter

from centrality_utils import load_graph, node_province_map

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_CURVE  = Path("maps/hac_percolation_curve.png")
OUT_CSV       = Path("data/hac_clusters.csv")
CACHE_PATH    = Path("data/hac_linkage_cache.pkl")

# Average linkage makes no Euclidean assumptions — appropriate for network
# travel-time distances. Ward is mathematically valid here too but assumes
# the distance space is approximately Euclidean.
LINKAGE_METHOD = "average"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading graph...")
G, id_to_node, edges_4326, nodes_4326 = load_graph()

print("Loading province assignments...")
prov_map = node_province_map(id_to_node)  # node_id → province

# ── Connected components ──────────────────────────────────────────────────────
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"\n{len(comps)} connected components:")
for i, c in enumerate(comps):
    print(f"  [{i:2d}]  {len(c):5d} nodes")

# Drop singleton and 2-node components — HAC is trivial/degenerate for these
comps = [c for c in comps if len(c) >= 3]


# ── HAC per component ─────────────────────────────────────────────────────────
def component_linkage(G: nx.Graph, nodes: set) -> tuple[list, np.ndarray]:
    """Compute pairwise travel-time distances within a component and run HAC.

    Returns
    -------
    node_ids   list of node_id ints in matrix row order
    Z          scipy linkage matrix  (n-1, 4)
    """
    sub = G.subgraph(nodes)
    node_keys = list(sub.nodes())
    node_ids  = [G.nodes[n]["node_id"] for n in node_keys]
    n = len(node_keys)

    # All-pairs shortest path (travel time)
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


if CACHE_PATH.exists():
    print(f"\nLoading cached linkage results from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        comp_results = pickle.load(f)
    print(f"  Loaded {len(comp_results)} components")
    for i, (node_ids, Z) in enumerate(comp_results):
        print(f"  [{i:2d}]  {len(node_ids):5d} nodes  "
              f"(max merge: {Z[:, 2].max()/3600:.2f}h)")
else:
    print(f"\nNo cache found — computing HAC for {len(comps)} components with ≥3 nodes")
    comp_results: list[tuple[list, np.ndarray]] = []

    for i, comp in enumerate(comps):
        n = len(comp)
        print(f"  Component {i}  ({n} nodes) — computing distances...", flush=True)
        node_ids, Z = component_linkage(G, comp)
        comp_results.append((node_ids, Z))
        print(f"    done  (max merge height: {Z[:, 2].max():.0f}s = {Z[:, 2].max()/3600:.2f}h)")

    print(f"\nSaving linkage cache → {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(comp_results, f)


# ── Dendrograms ───────────────────────────────────────────────────────────────
# Large components (>100 nodes) are truncated to their last p merges so the
# plot stays readable — this shows the high-level structure (how the big
# clusters collapse into each other) without printing thousands of leaf labels.
# Small components show the full tree with node_id labels.

from scipy.cluster.hierarchy import dendrogram

COMP_NAMES = {
    0: "Continental network",
    1: "British Isles",
    2: "Sicily",
    3: "Sardinia",
    4: "Cyprus",
    5: "Crete",
    6: "Other island",
}

def save_dendrogram(
    node_ids: list,
    Z: np.ndarray,
    comp_idx: int,
    out_path: Path,
    crit_threshold: float = None,
) -> None:
    n = len(node_ids)
    title = COMP_NAMES.get(comp_idx, f"Component {comp_idx}")

    # Truncate large components to last p merges for readability
    truncate = n > 50
    p = min(50, n - 1)  # last p merges shown as collapsed triangles

    # Scale Z heights to hours for the axis
    Z_h = Z.copy()
    Z_h[:, 2] = Z_h[:, 2] / 3600

    fig, ax = plt.subplots(figsize=(14, 5) if truncate else (max(10, n // 3), 6))

    dend_kwargs = dict(
        Z=Z_h,
        ax=ax,
        orientation="top",
        color_threshold=0,   # all black — colour by threshold line instead
        above_threshold_color="#444444",
        link_color_func=lambda _: "#444444",
        no_labels=truncate,
    )
    if truncate:
        dend_kwargs.update(truncate_mode="lastp", p=p, show_contracted=True)
    else:
        dend_kwargs["labels"] = [str(nid) for nid in node_ids]

    dendrogram(**dend_kwargs)

    # Mark critical threshold
    if crit_threshold is not None:
        ax.axhline(
            crit_threshold / 3600,
            color="#e74c3c", linewidth=1.2, linestyle="--",
            label=f"Critical threshold ({crit_threshold/3600:.1f}h)",
        )
        ax.legend(fontsize=9, loc="upper left")

    ax.set_ylabel("Travel time (hours)", fontsize=11)
    ax.set_xlabel(
        f"Node clusters  (last {p} merges shown, {n} total nodes)" if truncate
        else "Node ID",
        fontsize=10,
    )
    ax.set_title(f"HAC Dendrogram — {title}  ({n} nodes, {LINKAGE_METHOD} linkage)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Dendrogram → {out_path}")


print("\nSaving dendrograms...")
for i, (node_ids, Z) in enumerate(comp_results):
    out = Path(f"maps/hac_dendrogram_comp{i}.png")
    # Pass crit threshold after it's computed — store Z results first, plot after sweep
    save_dendrogram(node_ids, Z, i, out, crit_threshold=None)
print("  (critical threshold lines will be added after the sweep)")


# ── Threshold sweep ───────────────────────────────────────────────────────────
# Collect every unique merge height across all components as threshold candidates.
# Nothing changes between merges, so this is the minimal exact set of thresholds.
all_heights = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))
# Add a zero sentinel (all singletons)
thresholds = np.concatenate([[0.0], all_heights])

print(f"\nSweeping {len(thresholds)} thresholds "
      f"(0 → {all_heights.max()/3600:.1f}h)...")

n_clusters_arr = np.zeros(len(thresholds), dtype=int)
largest_arr    = np.zeros(len(thresholds), dtype=int)

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
    largest_arr[ti]    = sizes_sorted[0] if sizes_sorted else 0


# ── Critical threshold: steepest drop in number of clusters (curve label only) ─
drop = np.diff(n_clusters_arr)
crit_idx = int(np.argmin(drop)) + 1
t_crit   = thresholds[crit_idx]

# Fixed snapshot thresholds
SNAPSHOT_HOURS = [6, 12, 24, 48, 168]   # 168h = 7 days
SNAPSHOTS = [(h, h * 3600) for h in SNAPSHOT_HOURS]

print(f"\nSnapshot thresholds:")
for h, t_s in SNAPSHOTS:
    # find nearest threshold index
    idx = int(np.searchsorted(thresholds, t_s, side="left"))
    idx = min(idx, len(n_clusters_arr) - 1)
    print(f"  {h:4d}h : {n_clusters_arr[idx]} clusters")


# ── Percolation curve ─────────────────────────────────────────────────────────
# Plot log(hours) directly on a linear axis — gives true uniform log spacing
# without matplotlib's log-scale quirks near zero.
hours    = thresholds / 3600
log_hours = np.log(np.where(hours == 0, 1e-6, hours))

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.plot(log_hours, n_clusters_arr, color="#2c3e50", linewidth=1.5, label="N clusters")
ax2.plot(log_hours, largest_arr,    color="#2980b9", linewidth=1.5, linestyle="--", label="Largest cluster size")

ax1.set_ylabel("Number of clusters", color="#2c3e50", fontsize=11)
ax2.set_ylabel("Largest cluster size (nodes)", color="#2980b9", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#2c3e50")
ax2.tick_params(axis="y", labelcolor="#2980b9")

# Replace raw log values on x-axis with readable hour labels
hour_ticks = [0.1, 0.5, 1, 6, 12, 24, 48, 168, 500, 2000]
ax1.set_xticks([np.log(h) for h in hour_ticks])
ax1.set_xticklabels([f"{h}h" if h < 48 else f"{h//24}d" for h in hour_ticks])
ax1.set_xlabel("Distance threshold (log travel time)", fontsize=11)

for h, t_s in SNAPSHOTS:
    ax1.axvline(np.log(h), color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(np.log(h) + 0.05, ax1.get_ylim()[1] * 0.97,
             f"{h}h" if h < 48 else f"{h//24}d",
             color="#888888", fontsize=8, va="top")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

plt.title("HAC Percolation — Roman Road Network", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_CURVE, dpi=150)
plt.close()
print(f"\nPercolation curve → {OUT_CURVE}")

# Redraw dendrograms now that t_crit is known
print("Saving dendrograms with critical threshold...")
for i, (node_ids, Z) in enumerate(comp_results):
    out = Path(f"maps/hac_dendrogram_comp{i}.png")
    save_dendrogram(node_ids, Z, i, out, crit_threshold=t_crit)


# ── Cluster assignment at each snapshot ──────────────────────────────────────
def assign_clusters(comp_results, t: float, offset: int = 0) -> dict[int, int]:
    """Return node_id → global cluster label at threshold t."""
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

# ── Pleiades region assignment (spatial join) ─────────────────────────────────
print("\nAssigning Pleiades regions to nodes...")
pleiades = gpd.read_file(Path("data/pleiades-regions-magis-pelagios.geojson"))
pleiades = pleiades[["name", "geometry"]].rename(columns={"name": "pleiades_region"})

# Spatial join: each node gets the first matching Pleiades polygon it falls within
nodes_joined = gpd.sjoin(
    nodes_4326[["node_id", "geometry"]],
    pleiades,
    how="left",
    predicate="within",
)
# Drop duplicate matches (node in multiple overlapping polygons) — keep first
nodes_joined = nodes_joined.drop_duplicates(subset="node_id", keep="first")
pleiades_map = dict(zip(nodes_joined["node_id"], nodes_joined["pleiades_region"]))
print(f"  {nodes_joined['pleiades_region'].notna().sum()} / {len(nodes_4326)} nodes matched a Pleiades region")


# ── Cluster assignments at each snapshot ──────────────────────────────────────
snapshot_clusters: dict[int, dict[int, int]] = {}  # hours → {node_id: cluster}
for h, t_s in SNAPSHOTS:
    snapshot_clusters[h] = assign_clusters(comp_results, t_s)


# ── Snapshot maps ─────────────────────────────────────────────────────────────
_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6",
    "#bfef45","#fabed4","#469990","#dcbeff","#9a6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#c0c0c0","#000000",
]

def cluster_color(cluster_id: int) -> str:
    return _PALETTE[cluster_id % len(_PALETTE)]


def save_snapshot_map(
    nodes_4326: gpd.GeoDataFrame,
    edges_4326: gpd.GeoDataFrame,
    cluster_map: dict[int, int],
    hours: int,
    out_path: Path,
) -> None:
    nodes = nodes_4326.copy()
    nodes["cluster"] = nodes["node_id"].map(cluster_map).fillna(-1).astype(int)
    nodes["color"]   = nodes["cluster"].map(cluster_color)
    n_clust = nodes["cluster"].nunique()

    center = nodes.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=5, tiles="CartoDB positron")

    fg_roads = folium.FeatureGroup(name="Road network", show=True)
    folium.GeoJson(
        edges_4326[["geometry"]].__geo_interface__,
        style_function=lambda _: {"color": "#cccccc", "weight": 1, "opacity": 0.5},
    ).add_to(fg_roads)
    fg_roads.add_to(m)

    fg_nodes = folium.FeatureGroup(name=f"Clusters ({hours}h threshold)", show=True)
    for row in nodes.itertuples(index=False):
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=row.color,
            fill=True,
            fill_color=row.color,
            fill_opacity=0.85,
            weight=0,
            tooltip=f"node_id={int(row.node_id)}<br>cluster={int(row.cluster)}",
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    label = f"{hours}h" if hours < 48 else f"{hours//24}d"
    legend = f"""
    <div style="position:fixed;bottom:40px;right:40px;z-index:1000;background:white;
                border:1px solid #ccc;border-radius:6px;padding:12px 16px;
                font-family:Arial,sans-serif;font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15);">
        <b>HAC clusters</b><br>
        Threshold: <b>{label}</b><br>
        N clusters: <b>{n_clust}</b>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(out_path))
    print(f"Map saved → {out_path}")


print("\nSaving snapshot maps...")
for h, t_s in SNAPSHOTS:
    label = f"{h}h" if h < 48 else f"{h // 24}d"
    out = Path(f"maps/hac_snapshot_{label}.html")
    save_snapshot_map(nodes_4326, edges_4326, snapshot_clusters[h], h, out)


# ── CSV ───────────────────────────────────────────────────────────────────────
# Columns: node_id, province, pleiades_region, cluster_Xh / cluster_Xd
# Enables correlation of HAC clusters against both membership classifications.
print("\nSaving CSV...")
all_node_ids = list({nid for node_ids, _ in comp_results for nid in node_ids})

df = pd.DataFrame({"node_id": all_node_ids})
df["province"]       = df["node_id"].map(prov_map).fillna("Unknown")
df["pleiades_region"] = df["node_id"].map(pleiades_map).fillna("Unknown")

for h, t_s in SNAPSHOTS:
    col = f"cluster_{h}h" if h < 48 else f"cluster_{h // 24}d"
    df[col] = df["node_id"].map(snapshot_clusters[h]).fillna(-1).astype(int)

df = df.sort_values("node_id").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"CSV saved → {OUT_CSV}  ({len(df)} nodes)")
print(f"Columns: {list(df.columns)}")


# ── Province alignment scores ─────────────────────────────────────────────────
# Adjusted Rand Index (ARI): 1.0 = perfect match, 0.0 = random, can be negative
# Normalised Mutual Information (NMI): 1.0 = perfect, 0.0 = no shared information
# Both are computed only on nodes with a known province assignment.
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

cluster_cols = [
    (h, f"cluster_{h}h" if h < 48 else f"cluster_{h // 24}d")
    for h, _ in SNAPSHOTS
]

# Filter to nodes with known province
known = df[df["province"] != "Unknown"].copy()
province_labels = known["province"].values

print("\n── Province alignment (" + str(len(known)) + " nodes with known province) ──")
print(f"{'Threshold':<12} {'N clusters':>12} {'ARI':>8} {'NMI':>8}")
print("─" * 44)

best_ari, best_col_label = -np.inf, ""
for h, col in cluster_cols:
    cluster_labels = known[col].values
    ari = adjusted_rand_score(province_labels, cluster_labels)
    nmi = normalized_mutual_info_score(province_labels, cluster_labels, average_method="arithmetic")
    n_clust = df[col].nunique()
    label = f"{h}h" if h < 48 else f"{h // 24}d"
    marker = " ◀ best ARI" if ari > best_ari else ""
    print(f"{label:<12} {n_clust:>12} {ari:>8.4f} {nmi:>8.4f}{marker}")
    if ari > best_ari:
        best_ari = ari
        best_col_label = label

print("─" * 44)
print(f"Best province match: threshold = {best_col_label}  (ARI = {best_ari:.4f})")

print("\nDone.")
