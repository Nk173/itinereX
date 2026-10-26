"""
centrality_utils.py
Shared helpers for centrality analysis scripts.
"""

import pickle
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import folium
from pathlib import Path

PICKLE_PATH  = Path("data_cleaning/cleaned_data/G_clean_weighted_with_ids.pkl")
PROVINCE_CSV = Path("data_cleaning/cleaned_data/G_clean_weighted_edges_with_province.csv")

# Fixed node colour — opacity encodes centrality
NODE_COLOR = "#c0392b"
# Opacity range: least central nodes are faint but still visible
MIN_OPACITY = 0.05
MAX_OPACITY = 1.0


def load_graph() -> tuple[nx.Graph, dict, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load the cleaned weighted graph and associated GeoDataFrames.

    Returns
    -------
    G           NetworkX graph with node_id/edge_id attributes and time_s weights
    id_to_node  dict mapping node_id (int) → (x, y) tuple
    edges_4326  GeoDataFrame of edges in EPSG:4326
    nodes_4326  GeoDataFrame of nodes in EPSG:4326
    """
    data = pickle.load(open(PICKLE_PATH, "rb"))
    id_to_node = data["id_to_node"]

    edge_table = data["edge_table"]
    node_table = data["node_table"]

    G = nx.Graph()
    for row in node_table.itertuples(index=False):
        nkey = id_to_node[int(row.node_id)]
        G.add_node(nkey, node_id=int(row.node_id), degree=int(row.degree))

    for row in edge_table.itertuples(index=False):
        u = id_to_node[int(row.u_id)]
        v = id_to_node[int(row.v_id)]
        G.add_edge(u, v,
                   edge_id=int(row.edge_id),
                   weight=float(row.weight),
                   time_s=float(row.time_s),
                   dist_m=float(row.dist_m))

    return G, id_to_node, data["edges_4326"], data["nodes_4326"]


def node_province_map(id_to_node: dict) -> dict[int, str]:
    """Return a dict mapping node_id → province name.

    Each node is assigned to the most common province among its incident edges.
    Crossing/outside edges are ignored. Falls back to 'Unknown'.
    """
    prov = pd.read_csv(PROVINCE_CSV, usecols=["edge_id", "u_id", "v_id", "province", "is_crossing", "is_outside"])
    within = prov[~prov["is_crossing"] & ~prov["is_outside"] & prov["province"].notna()].copy()

    from collections import Counter
    node_provs: dict[int, Counter] = {}
    for row in within.itertuples(index=False):
        for nid in (int(row.u_id), int(row.v_id)):
            node_provs.setdefault(nid, Counter())[str(row.province)] += 1

    return {nid: c.most_common(1)[0][0] for nid, c in node_provs.items()}


def scores_to_opacity(scores: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """Map scores to opacity values in [MIN_OPACITY, MAX_OPACITY].

    Log-scaling is used by default because centrality distributions are
    heavy-tailed — without it, all but the top few nodes would be invisible.
    """
    s = np.array(scores, dtype=float)
    if log_scale:
        s = np.log1p(s - s.min())
    vmin, vmax = s.min(), s.max()
    if vmax == vmin:
        return np.full(len(s), MAX_OPACITY)
    norm = (s - vmin) / (vmax - vmin)
    return MIN_OPACITY + norm * (MAX_OPACITY - MIN_OPACITY)


def _legend_html(label: str, score_min: float, score_max: float, color: str) -> str:
    """Return an HTML string for a fixed-color opacity legend."""
    # Parse hex color for rgba usage
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return f"""
    <div style="
        position: fixed;
        bottom: 40px; right: 40px;
        z-index: 1000;
        background: white;
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
        min-width: 140px;
    ">
        <b>{label}</b>
        <div style="margin-top:8px; display:flex; align-items:stretch; gap:8px;">
            <div style="
                width: 16px;
                background: linear-gradient(
                    to bottom,
                    rgba({r},{g},{b},{MAX_OPACITY}),
                    rgba({r},{g},{b},{MIN_OPACITY})
                );
                border-radius: 3px;
            "></div>
            <div style="display:flex; flex-direction:column; justify-content:space-between; line-height:1.3;">
                <span>High ({score_max:.2e})</span>
                <span style="margin-top:40px;">Low ({score_min:.2e})</span>
            </div>
        </div>
    </div>
    """


def html_to_pdf(html_path: Path, wait_ms: int = 3000) -> Path:
    """Render a saved folium HTML map to PDF via a headless Chromium browser."""
    from playwright.sync_api import sync_playwright
    pdf_path = html_path.with_suffix(".pdf")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(html_path.resolve().as_uri())
        page.wait_for_timeout(wait_ms)
        page.pdf(path=str(pdf_path), format="A3", landscape=True, print_background=True)
        browser.close()
    print(f"PDF saved  → {pdf_path}")
    return pdf_path


def save_map(
    nodes_4326: gpd.GeoDataFrame,
    edges_4326: gpd.GeoDataFrame,
    score_col: str,
    label: str,
    out_path: Path,
    log_scale: bool = True,
) -> None:
    """Save a Folium map with nodes coloured by opacity and edges as grey background.

    Most central nodes are fully opaque; least central nodes are near-transparent.

    Parameters
    ----------
    nodes_4326  GeoDataFrame with node geometry + score_col column
    edges_4326  GeoDataFrame with edge geometry (background layer)
    score_col   Column name in nodes_4326 containing the centrality score
    label       Human-readable name shown in layer controls, tooltips, and legend
    out_path    Path to write the HTML file
    log_scale   Apply log scaling before normalising (recommended for heavy-tailed scores)
    """
    nodes_4326 = nodes_4326.copy()
    scores = nodes_4326[score_col].values
    opacities = scores_to_opacity(scores, log_scale=log_scale)
    nodes_4326["opacity"] = opacities

    center = nodes_4326.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=5, tiles="CartoDB positron")

    # Background: road network
    fg_roads = folium.FeatureGroup(name="Road network", show=True)
    folium.GeoJson(
        edges_4326[["geometry"]].__geo_interface__,
        style_function=lambda _: {"color": "#bbbbbb", "weight": 1, "opacity": 0.5},
    ).add_to(fg_roads)
    fg_roads.add_to(m)

    # Nodes: opacity encodes centrality score
    fg_nodes = folium.FeatureGroup(name=label, show=True)
    for row in nodes_4326.itertuples(index=False):
        score   = float(getattr(row, score_col))
        opacity = float(row.opacity)
        province = getattr(row, "province", "Unknown")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=NODE_COLOR,
            weight=0,
            fill=True,
            fill_color=NODE_COLOR,
            fill_opacity=opacity,
            tooltip=f"node_id={int(row.node_id)}<br>{label}={score:.2e}<br>province={province}",
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend = _legend_html(
        label=label,
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        color=NODE_COLOR,
    )
    m.get_root().html.add_child(folium.Element(legend))

    m.save(str(out_path))
    print(f"Map saved → {out_path}")
    html_to_pdf(out_path)
