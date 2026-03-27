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

from centrality_utils import load_graph, node_province_map, html_to_pdf

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_CURVE  = Path("maps/hac_percolation_curve.png")
OUT_CSV       = Path("data/hac_clusters.csv")
CACHE_PATH    = Path("data/hac_linkage_average_cache.pkl")

# Average linkage makes no Euclidean assumptions — appropriate for network
# travel-time distances. Ward is mathematically valid here too but assumes
# the distance space is approximately Euclidean.
LINKAGE_METHOD = "average"
# LINKAGE_METHOD = "single"


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
SNAPSHOT_HOURS = [6, 12, 24, 48, 168, 288]   # 168h = 7 days, 288h = 12 days
SNAPSHOTS = [(h, h * 3600) for h in SNAPSHOT_HOURS]

print(f"\nSnapshot thresholds:")
for h, t_s in SNAPSHOTS:
    # find nearest threshold index
    idx = int(np.searchsorted(thresholds, t_s, side="left"))
    idx = min(idx, len(n_clusters_arr) - 1)
    print(f"  {h:4d}h : {n_clusters_arr[idx]} clusters")


# ── Percolation curve ─────────────────────────────────────────────────────────
hours = thresholds / 3600
hours_x = hours[hours>1]
n_clusters_arr_x = n_clusters_arr[hours>1]
largest_arr_x = largest_arr[hours>1]
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.plot(hours_x, n_clusters_arr_x, color="#2c3e50", linewidth=1.5, label="N clusters")
ax2.plot(hours_x, largest_arr_x,    color="#2980b9", linewidth=1.5, linestyle="--", label="Largest cluster size")
ax1.set_xscale("log")
ax2.set_xscale("log")
ax1.set_yscale("log")
ax2.set_yscale("log")

# ── Power-law fit: S_max ~ t^alpha ────────────────────────────────────────────
# Fit over the scaling regime: t = 6h to 4000h.
# np.polyfit(cov=True) returns the covariance matrix of the OLS estimates;
# sqrt of the diagonal gives standard errors → 95% CI = ±1.96 SE.
_lh = np.log10(hours_x[largest_arr_x > 0])
_ls = np.log10(largest_arr_x[largest_arr_x > 0])
_mask = (_lh >= np.log10(6)) & (_lh <= np.log10(4000))
(_alpha, _log_a), _cov = np.polyfit(_lh[_mask], _ls[_mask], 1, cov=True)
_a = 10 ** _log_a
_se_alpha  = np.sqrt(_cov[0, 0])
_ci_alpha  = 1.96 * _se_alpha          # 95% CI half-width
_resid = _ls[_mask] - (_alpha * _lh[_mask] + _log_a)
_r2 = 1 - np.sum(_resid**2) / np.sum((_ls[_mask] - _ls[_mask].mean())**2)
_fit_y = _a * hours_x ** _alpha
ax2.plot(hours_x, _fit_y, color="#85c1e9", linewidth=1.2, linestyle=":",
         label=fr"Fit: $S_{{\max}} \propto t^{{{_alpha:.2f} \pm {_ci_alpha:.2f}}}$  ($R^2={_r2:.3f}$)")
print(f"\nScaling law fit (largest cluster):  S_max ∝ t^{_alpha:.3f}  "
      f"95% CI [{_alpha - _ci_alpha:.3f}, {_alpha + _ci_alpha:.3f}]  "
      f"(a={_a:.2f},  R²={_r2:.4f},  fit range 6–4000h)")

ax1.set_ylabel("Number of clusters", color="#2c3e50", fontsize=11)
ax2.set_ylabel("Largest cluster size (nodes)", color="#2980b9", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#2c3e50")
ax2.tick_params(axis="y", labelcolor="#2980b9")

ax1.set_xlabel("Distance threshold (travel time, hours)", fontsize=11)

for h, t_s in SNAPSHOTS:
    ax1.axvline(h, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(h * 1.03, 0.97,
             f"{h}h" if h < 48 else f"{h//24}d",
             color="#888888", fontsize=8, va="top",
             transform=ax1.get_xaxis_transform())

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


# Barrington Atlas rank → marker radius and display label
_RANK_RADIUS = {1: 9, 2: 7, 3: 5}
_RANK_LABEL  = {1: "Rank 1 (major)", 2: "Rank 2", 3: "Rank 3", 4: "Rank 4/5", 5: "Rank 4/5"}
_CITY_COLOR  = "#2c3e50"


def _parse_rank(raw) -> int:
    """Normalise Barrington Atlas rank to an integer (1–5). Unknown → 5."""
    try:
        return int(str(raw).strip().split()[0])
    except (ValueError, IndexError):
        return 5


def save_snapshot_map(
    nodes_4326: gpd.GeoDataFrame,
    edges_4326: gpd.GeoDataFrame,
    cluster_map: dict[int, int],
    hours: int,
    out_path: Path,
    cities_df: pd.DataFrame | None = None,
    provinces_gdf: gpd.GeoDataFrame | None = None,
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

    # Optional province boundaries overlay
    if provinces_gdf is not None:
        prov_4326 = provinces_gdf.to_crs("EPSG:4326")
        fg_prov = folium.FeatureGroup(name="Province boundaries", show=True)
        folium.GeoJson(
            prov_4326.__geo_interface__,
            style_function=lambda _: {
                "color": "#2c3e50",
                "weight": 1.5,
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(fields=["province"], aliases=["Province:"]),
        ).add_to(fg_prov)
        fg_prov.add_to(m)

    # Optional cities overlay
    if cities_df is not None:
        fg_cities = folium.FeatureGroup(name="Cities (OxREP)", show=False)
        for _, city in cities_df.iterrows():
            rank   = _parse_rank(city["Barrington Atlas Rank"])
            radius = _RANK_RADIUS.get(rank, 4)
            folium.CircleMarker(
                location=[city["Latitude (Y)"], city["Longitude (X)"]],
                radius=radius,
                color=_CITY_COLOR,
                fill=True,
                fill_color=_CITY_COLOR,
                fill_opacity=0.75,
                weight=1,
                tooltip=(
                    f"<b>{city['Ancient Toponym']}</b>"
                    f"<br>Modern: {city['Modern Toponym']}"
                    f"<br>Province: {city['Province']}"
                    f"<br>{_RANK_LABEL.get(rank, 'Unranked')}"
                    f"<br>Dates: {city['Start Date']} – {city['End Date'] if pd.notna(city['End Date']) else 'ongoing'}"
                ),
            ).add_to(fg_cities)
        fg_cities.add_to(m)

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
    html_to_pdf(out_path)


print("\nLoading cities (OxREP)...")
cities_df = pd.read_excel(Path("data/OxREP_data.xlsx"), sheet_name="Cities")
print(f"  {len(cities_df)} cities loaded")

print("Loading province boundaries...")
provinces_gdf = gpd.read_file(Path("provinces/provinces.shp"))
print(f"  {len(provinces_gdf)} provinces loaded")

print("\nSaving snapshot maps...")
for h, t_s in SNAPSHOTS:
    label = f"{h}h" if h < 48 else f"{h // 24}d"
    out = Path(f"maps/hac_snapshot_{label}.html")
    save_snapshot_map(
        nodes_4326, edges_4326, snapshot_clusters[h], h, out,
        cities_df=cities_df if h == 48 else None,
        provinces_gdf=provinces_gdf if h == 288 else None,
    )


# ── Combined PDF: percolation curve + 4 static snapshot maps ──────────────────
print("\nSaving combined PDF figure...")
COMBINED_PDF = Path("maps/hac_percolation_combined.pdf")
_MAP_HOURS = [6, 12, 24, 48]   # 6h, 24h, 10d, 42d

fig = plt.figure(figsize=(20, 11))
gs  = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.38, wspace=0.08)

# ── Top row: percolation curve (full width) ────────────────────────────────
ax1c = fig.add_subplot(gs[0, :])
ax2c = ax1c.twinx()

ax1c.plot(hours_x, n_clusters_arr_x, color="#2c3e50", linewidth=1.5, label="N clusters")
ax2c.plot(hours_x, largest_arr_x,    color="#2980b9", linewidth=1.5, linestyle="--",
          label="Largest cluster size")
ax2c.plot(hours_x, _a * hours_x ** _alpha, color="#85c1e9", linewidth=1.2, linestyle=":",
          label=fr"Fit: $S_{{\max}} \propto t^{{{_alpha:.2f} \pm {_ci_alpha:.2f}}}$  ($R^2={_r2:.3f}$)")

ax1c.set_xscale("log"); ax2c.set_xscale("log")
ax1c.set_yscale("log"); ax2c.set_yscale("log")
ax1c.set_ylabel("Number of clusters",           color="#2c3e50", fontsize=10)
ax2c.set_ylabel("Largest cluster size (nodes)", color="#2980b9", fontsize=10)
ax1c.tick_params(axis="y", labelcolor="#2c3e50")
ax2c.tick_params(axis="y", labelcolor="#2980b9")
ax1c.set_xlabel("Distance threshold (travel time, hours)", fontsize=10)

for h, t_s in SNAPSHOTS:
    ax1c.axvline(h, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1c.text(h * 1.03, 0.97,
              f"{h}h" if h < 48 else f"{h//24}d",
              color="#888888", fontsize=8, va="top",
              transform=ax1c.get_xaxis_transform())

# Mark the 4 map thresholds
for h in _MAP_HOURS:
    ax1c.axvline(h, color="#e74c3c", linewidth=1.0, alpha=0.35)

_l1, _lb1 = ax1c.get_legend_handles_labels()
_l2, _lb2 = ax2c.get_legend_handles_labels()
ax1c.legend(_l1 + _l2, _lb1 + _lb2, loc="center right", fontsize=9)
ax1c.set_title("(a)  HAC Percolation — Roman Road Network", fontsize=12, loc="left")

# ── Bottom row: 4 static snapshot maps ────────────────────────────────────
_subplot_labels = ["(b)", "(c)", "(d)", "(e)"]
for col, (h, sublabel) in enumerate(zip(_MAP_HOURS, _subplot_labels)):
    ax_map = fig.add_subplot(gs[1, col])

    edges_4326.plot(ax=ax_map, color="#cccccc", linewidth=0.25, alpha=0.6)

    _clust_h = assign_clusters(comp_results, h * 3600)
    _nodes = nodes_4326.copy()
    _nodes["color"] = (
        _nodes["node_id"]
        .map(_clust_h)
        .fillna(-1).astype(int)
        .map(cluster_color)
    )
    _nodes.plot(ax=ax_map, color=_nodes["color"].tolist(), markersize=0.8, alpha=0.85)

    _label = f"{h}h" if h < 48 else f"{h//24}d"
    _idx   = min(int(np.searchsorted(thresholds, h * 3600, side="left")),
                 len(n_clusters_arr) - 1)
    ax_map.set_title(f"{sublabel}  t = {_label}  ({n_clusters_arr[_idx]} clusters)",
                     fontsize=10, loc="left")
    ax_map.set_axis_off()

plt.savefig(COMBINED_PDF, dpi=150, bbox_inches="tight")
plt.close()
print(f"Combined PDF → {COMBINED_PDF}")


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