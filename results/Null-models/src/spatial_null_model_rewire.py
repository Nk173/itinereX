"""
Spatial null model for province road subnetworks.

Goal:
- Keep the same nodes for each graph (province-level or global).
- Build spatial null graphs from explicit connection rules (KNN or Delaunay).
- Optionally match original edge count for comparability.

Method:
- Start from each original province graph.
- Repeatedly pick two edges and propose a double-edge swap.
- Reject swaps that create self-loops, duplicate edges, or overly long edges.
- Accept swaps with higher probability when they reduce total spatial cost.

Outputs:
- Null-model graphs pickle.
- Per-graph comparison CSV (original vs null metrics).
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Hashable, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


_REPO_ROOT = Path(__file__).resolve().parents[3]  # itinereX repo root
DEFAULT_ITINEREX_DIR = _REPO_ROOT
DEFAULT_REPO_DIR = _REPO_ROOT / "results" / "Null-models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build spatial null models for province subnetworks."
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="province",
        choices=["province", "global"],
        help="Run null model per province or once on the full empire network",
    )
    parser.add_argument(
        "--itinereX-dir",
        type=Path,
        default=DEFAULT_ITINEREX_DIR,
        help="Path to itinereX repository containing province_subnetworks.pkl",
    )
    parser.add_argument(
        "--input-pkl",
        type=str,
        default="province_subnetworks.pkl",
        help="Input pickle filename inside --itinereX-dir",
    )
    parser.add_argument(
        "--global-input-pkl",
        type=str,
        default="G_clean_weighted_with_ids.pkl",
        help="Input pickle filename for --scope global (expects node_table/edge_table)",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=DEFAULT_REPO_DIR / "data" / "processed" / "province_subnetworks_spatial_null.pkl",
        help="Output pickle path for rewired province subnetworks",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=DEFAULT_REPO_DIR / "data" / "processed" / "spatial_null_model_summary.csv",
        help="Output CSV path for null-model summary",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="knn",
        choices=["knn", "delaunay", "knn_mst", "delaunay_mst"],
        help="Spatial null model type (optionally add euclidean MST backbone)",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=4,
        help="Number of nearest neighbors for --model knn",
    )
    parser.add_argument(
        "--match-edge-count",
        action="store_true",
        help="Force null graphs to have same edge count as original graph",
    )
    parser.add_argument(
        "--connect-provinces",
        type=str,
        default="none",
        choices=["none", "mst_centroid", "mst_minpair"],
        help=(
            "Only for --scope province. Add inter-province bridge edges after per-province null construction. "
            "mst_centroid builds an MST over province centroids and connects nearest node pairs; "
            "mst_minpair builds an MST over minimum inter-province node-pair distances."
        ),
    )
    parser.add_argument(
        "--allow-sea-crossings",
        action="store_true",
        help=(
            "Allow null edges/bridges that are absent in empirical cross-province connectivity. "
            "By default this is OFF, so sea-crossing shortcuts are masked out."
        ),
    )
    parser.add_argument(
        "--max-add-dist-quantile",
        type=float,
        default=0.98,
        help="Distance quantile cap when adding missing edges to match edge count",
    )
    parser.add_argument(
        "--delaunay-max-dist-quantile",
        type=float,
        default=0.95,
        help=(
            "For Delaunay models, drop candidate edges longer than this quantile "
            "of empirical within-graph edge lengths"
        ),
    )
    parser.add_argument(
        "--weight-rule",
        type=str,
        default="uniform_range_times",
        choices=["distance_speed", "rank_preserve_times", "sample_preserve_times", "uniform_range_times"],
        help=(
            "How to assign edge travel-time weights for knn/delaunay models: "
            "distance_speed uses median speed; rank_preserve_times preserves original time_s distribution "
            "by rank; sample_preserve_times samples from original time_s values; "
            "uniform_range_times samples uniformly between regional min/max time_s"
        ),
    )
    parser.add_argument(
        "--swaps-per-edge",
        type=float,
        default=5.0,
        help="Target accepted swaps per edge (e.g., 5.0)",
    )
    parser.add_argument(
        "--max-dist-quantile",
        type=float,
        default=0.95,
        help="Hard cap for new edge length based on original edge-length quantile",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="Distance penalty strength for accepting longer rewired edges",
    )
    parser.add_argument(
        "--max-attempt-factor",
        type=float,
        default=50.0,
        help="Max attempts = factor * target_swaps",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=6,
        help="Skip rewiring for very small graphs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def _node_xy(G: nx.Graph, node: Hashable) -> Tuple[float, float]:
    """Infer node coordinates from node object or node attributes."""
    if isinstance(node, (tuple, list)) and len(node) >= 2:
        x, y = node[0], node[1]
        if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)):
            return float(x), float(y)

    attrs = G.nodes[node]
    if "x" in attrs and "y" in attrs:
        return float(attrs["x"]), float(attrs["y"])
    if "lon" in attrs and "lat" in attrs:
        return float(attrs["lon"]), float(attrs["lat"])

    raise ValueError(f"Could not infer coordinates for node: {node}")


def _euclidean_dist(coord_u: Tuple[float, float], coord_v: Tuple[float, float]) -> float:
    dx = coord_u[0] - coord_v[0]
    dy = coord_u[1] - coord_v[1]
    return float(math.hypot(dx, dy))


def _edge_length(
    G: nx.Graph,
    u: Hashable,
    v: Hashable,
    coords: Dict[Hashable, Tuple[float, float]],
) -> float:
    data = G.get_edge_data(u, v, default={})
    if isinstance(data, dict) and "length_m" in data and pd.notna(data["length_m"]):
        try:
            val = float(data["length_m"])
            if val > 0:
                return val
        except Exception:
            pass
    return _euclidean_dist(coords[u], coords[v])


def _edge_key(u: Hashable, v: Hashable) -> frozenset:
    return frozenset((u, v))


def _median_speed_mps(G: nx.Graph) -> float:
    """Estimate typical travel speed from existing length/time edge attributes."""
    speeds = []
    for _, _, data in G.edges(data=True):
        if not isinstance(data, dict):
            continue
        if "length_m" not in data or "time_s" not in data:
            continue
        try:
            length = float(data["length_m"])
            time_s = float(data["time_s"])
            if length > 0 and time_s > 0:
                speeds.append(length / time_s)
        except Exception:
            continue
    if not speeds:
        return 1.0
    return float(np.median(speeds))


def _edge_attrs_from_length(length_m: float, speed_mps: float) -> Dict[str, float]:
    out = {"length_m": float(length_m)}
    if speed_mps > 0:
        out["time_s"] = float(length_m / speed_mps)
    return out


def _extract_original_times(G: nx.Graph) -> list[float]:
    times = []
    for _, _, data in G.edges(data=True):
        if not isinstance(data, dict) or "time_s" not in data:
            continue
        try:
            t = float(data["time_s"])
            if t > 0 and np.isfinite(t):
                times.append(t)
        except Exception:
            continue
    return times


def _assign_weights_to_null_graph(
    H: nx.Graph,
    G_original: nx.Graph,
    coords: Dict[Hashable, Tuple[float, float]],
    rng: random.Random,
    weight_rule: str,
) -> Dict[str, float]:
    """Assign length_m/time_s to null edges based on selected weight rule."""
    stats = {
        "weight_rule_fallback": 0.0,
        "weight_rule_used": weight_rule,
    }

    if H.number_of_edges() == 0:
        return stats

    # Always set geometric lengths for null edges.
    edge_dist = []
    for u, v in H.edges():
        dist = _euclidean_dist(coords[u], coords[v])
        edge_dist.append((dist, u, v))
        H[u][v]["length_m"] = float(dist)

    if weight_rule == "distance_speed":
        speed_mps = _median_speed_mps(G_original)
        for dist, u, v in edge_dist:
            if speed_mps > 0:
                H[u][v]["time_s"] = float(dist / speed_mps)
        return stats

    orig_times = _extract_original_times(G_original)
    if not orig_times:
        # Fallback to speed-based times if original has no usable times.
        stats["weight_rule_fallback"] = 1.0
        stats["weight_rule_used"] = "distance_speed"
        speed_mps = _median_speed_mps(G_original)
        for dist, u, v in edge_dist:
            if speed_mps > 0:
                H[u][v]["time_s"] = float(dist / speed_mps)
        return stats

    if weight_rule == "rank_preserve_times":
        # Preserve the empirical time_s distribution exactly when counts match.
        edge_dist.sort(key=lambda x: x[0])
        times_sorted = sorted(orig_times)

        if len(times_sorted) != len(edge_dist):
            # If counts differ, use quantile mapping to approximate distribution.
            stats["weight_rule_fallback"] = 1.0
            q = np.linspace(0.0, 1.0, num=len(edge_dist), endpoint=True)
            mapped_times = np.quantile(times_sorted, q)
            for (dist, u, v), t in zip(edge_dist, mapped_times):
                H[u][v]["time_s"] = float(t)
            return stats

        for (dist, u, v), t in zip(edge_dist, times_sorted):
            H[u][v]["time_s"] = float(t)
        return stats

    if weight_rule == "sample_preserve_times":
        # Preserve distribution approximately via resampling with replacement.
        for _, u, v in edge_dist:
            H[u][v]["time_s"] = float(rng.choice(orig_times))
        return stats

    if weight_rule == "uniform_range_times":
        t_min = float(np.min(orig_times))
        t_max = float(np.max(orig_times))
        for _, u, v in edge_dist:
            H[u][v]["time_s"] = float(rng.uniform(t_min, t_max))
        return stats

    raise ValueError(f"Unsupported weight rule: {weight_rule}")


def _update_edge_attrs(
    attrs: Dict[str, Any],
    new_length: float,
    old_length: float,
    scale_time_by_length: bool = False,
) -> Dict[str, Any]:
    out = dict(attrs) if attrs is not None else {}
    out["length_m"] = float(new_length)
    if scale_time_by_length and "time_s" in out:
        try:
            old_time = float(out["time_s"])
            if old_length > 0 and old_time > 0:
                out["time_s"] = old_time * (new_length / old_length)
        except Exception:
            pass
    return out


def _graph_from_tables(node_table: pd.DataFrame, edge_table: pd.DataFrame) -> nx.Graph:
    """Construct full graph from node/edge tables in G_clean_weighted_with_ids.pkl."""
    G = nx.Graph()

    for _, row in node_table.iterrows():
        node_id = int(row["node_id"])
        attrs = {
            "x": float(row["x"]) if pd.notna(row.get("x")) else np.nan,
            "y": float(row["y"]) if pd.notna(row.get("y")) else np.nan,
        }
        G.add_node(node_id, **attrs)

    for _, row in edge_table.iterrows():
        u = int(row["u_id"])
        v = int(row["v_id"])
        attrs: Dict[str, Any] = {}
        if "dist_m" in row and pd.notna(row["dist_m"]):
            attrs["length_m"] = float(row["dist_m"])
        if "time_s" in row and pd.notna(row["time_s"]):
            attrs["time_s"] = float(row["time_s"])
        if "weight" in row and pd.notna(row["weight"]):
            attrs["weight"] = float(row["weight"])
        if "edge_id" in row and pd.notna(row["edge_id"]):
            attrs["edge_id"] = int(row["edge_id"])
        G.add_edge(u, v, **attrs)

    return G


def _resolve_itinerex_path(base_dir: Path, rel_path: str) -> Path:
    """Resolve an itinereX-relative path with fallback to base_dir/data/<filename>."""
    p = base_dir / rel_path
    if p.exists():
        return p
    p2 = base_dir / "data_cleaning" / "cleaned_data" / Path(rel_path).name
    if p2.exists():
        return p2
    return p


def _node_to_province_map(province_graphs: Dict[str, nx.Graph]) -> Dict[Hashable, str]:
    out: Dict[Hashable, str] = {}
    for prov, g in province_graphs.items():
        for n in g.nodes():
            out[n] = prov
    return out


def _empirical_allowed_province_pairs(
    G_empirical: nx.Graph,
    node_to_province: Dict[Hashable, str],
) -> Set[Tuple[str, str]]:
    """Undirected province pairs that exist as cross-province edges in empirical graph.
    
    Args:
        G_empirical: empirical graph (node IDs mapped to provinces via node_to_province dict)
        node_to_province: dict mapping node ID → province name
    
    Returns:
        set of (province_i, province_j) tuples (sorted alphabetically)
    """
    allowed: Set[Tuple[str, str]] = set()
    for u, v in G_empirical.edges():
        pu = node_to_province.get(u)
        pv = node_to_province.get(v)
        if pu is None or pv is None or pu == pv:
            continue
        key = (pu, pv) if pu < pv else (pv, pu)
        allowed.add(key)
    return allowed


def _node_id_to_province_map_from_coords(
    G: nx.Graph, 
    province_subgraphs: Dict[str, nx.Graph]
) -> Dict[Hashable, str]:
    """Map node IDs in G to provinces by matching node coordinates to province subgraph nodes.
    
    Args:
        G: graph with node attributes 'x', 'y'
        province_subgraphs: dict mapping province_name → subgraph (nodes use (x, y) tuples)
    
    Returns:
        dict mapping node ID in G → province name
    """
    # Build reverse map: (x, y) coordinate tuple → province name
    coord_to_province: Dict[Tuple[float, float], str] = {}
    for prov_name, prov_graph in province_subgraphs.items():
        for node_id in prov_graph.nodes():
            if isinstance(node_id, tuple) and len(node_id) == 2:
                coord_to_province[node_id] = prov_name
    
    # Map nodes in G to provinces by coordinate matching
    node_to_prov: Dict[Hashable, str] = {}
    for node_id in G.nodes():
        x = G.nodes[node_id].get('x')
        y = G.nodes[node_id].get('y')
        if x is not None and y is not None:
            coord_key = (x, y)
            if coord_key in coord_to_province:
                node_to_prov[node_id] = coord_to_province[coord_key]
    
    return node_to_prov



def _filter_edges_by_province_mask(
    H: nx.Graph,
    node_to_province: Dict[Hashable, str],
    allowed_pairs: Set[Tuple[str, str]],
) -> int:
    """Remove null edges between province pairs not observed in empirical graph."""
    removed = 0
    for u, v in list(H.edges()):
        pu = node_to_province.get(u)
        pv = node_to_province.get(v)
        if pu is None or pv is None or pu == pv:
            continue
        key = (pu, pv) if pu < pv else (pv, pu)
        if key not in allowed_pairs:
            H.remove_edge(u, v)
            removed += 1
    return removed


def _build_empty_with_nodes(G: nx.Graph) -> nx.Graph:
    """Create a graph with identical node set and node attributes."""
    H = nx.Graph()
    H.add_nodes_from((n, dict(attrs)) for n, attrs in G.nodes(data=True))
    return H


def _edge_list_with_dist(
    coords: Dict[Hashable, Tuple[float, float]],
) -> list[tuple[float, Hashable, Hashable]]:
    nodes = list(coords)
    out = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            out.append((_euclidean_dist(coords[u], coords[v]), u, v))
    out.sort(key=lambda x: x[0])
    return out


def _match_edge_count_by_distance(
    H: nx.Graph,
    coords: Dict[Hashable, Tuple[float, float]],
    target_edges: int,
    max_add_dist_quantile: float,
) -> Dict[str, float]:
    """Adjust edge count to target by removing longest edges or adding shortest non-edges."""
    removed = 0
    added = 0

    if H.number_of_edges() > target_edges:
        edges_with_dist = []
        for u, v in H.edges():
            edges_with_dist.append((_euclidean_dist(coords[u], coords[v]), u, v))
        edges_with_dist.sort(reverse=True, key=lambda x: x[0])
        for _, u, v in edges_with_dist:
            if H.number_of_edges() <= target_edges:
                break
            H.remove_edge(u, v)
            removed += 1

    if H.number_of_edges() < target_edges:
        all_pairs = _edge_list_with_dist(coords)
        dist_values = [d for d, _, _ in all_pairs]
        max_allowed = float(np.quantile(dist_values, max_add_dist_quantile)) if dist_values else float("inf")
        for dist, u, v in all_pairs:
            if H.number_of_edges() >= target_edges:
                break
            if dist > max_allowed:
                break
            if not H.has_edge(u, v):
                H.add_edge(u, v)
                added += 1

    return {
        "removed_for_match": float(removed),
        "added_for_match": float(added),
    }


def _add_mst_backbone(
    H: nx.Graph,
    coords: Dict[Hashable, Tuple[float, float]],
) -> Dict[str, float]:
    """Add euclidean MST edges (via Delaunay candidate graph) to reduce sparsity."""
    nodes = list(H.nodes())
    if len(nodes) < 2:
        return {
            "mst_edges_candidate": 0.0,
            "mst_edges_added": 0.0,
            "components_before_mst": float(nx.number_connected_components(H)) if H.number_of_nodes() else 0.0,
            "components_after_mst": float(nx.number_connected_components(H)) if H.number_of_nodes() else 0.0,
            "mst_fallback_chain": 0.0,
        }

    components_before = float(nx.number_connected_components(H))
    points = np.array([coords[node] for node in nodes], dtype=float)

    mst_fallback_chain = 0.0
    candidate = nx.Graph()
    candidate.add_nodes_from(nodes)

    try:
        tri = Delaunay(points, qhull_options="QJ")
        for simplex in tri.simplices:
            a = nodes[int(simplex[0])]
            b = nodes[int(simplex[1])]
            c = nodes[int(simplex[2])]
            for u, v in ((a, b), (b, c), (a, c)):
                if u == v:
                    continue
                w = _euclidean_dist(coords[u], coords[v])
                candidate.add_edge(u, v, weight=float(w))
    except Exception:
        mst_fallback_chain = 1.0

    if candidate.number_of_edges() == 0:
        # Deterministic fallback for degenerate geometry.
        ordered = sorted(nodes, key=lambda n: (coords[n][0], coords[n][1]))
        for i in range(len(ordered) - 1):
            u = ordered[i]
            v = ordered[i + 1]
            w = _euclidean_dist(coords[u], coords[v])
            candidate.add_edge(u, v, weight=float(w))

    mst_edges = list(nx.minimum_spanning_edges(candidate, algorithm="kruskal", data=False))

    added = 0
    for u, v in mst_edges:
        if not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1

    components_after = float(nx.number_connected_components(H))
    return {
        "mst_edges_candidate": float(len(mst_edges)),
        "mst_edges_added": float(added),
        "components_before_mst": components_before,
        "components_after_mst": components_after,
        "mst_fallback_chain": mst_fallback_chain,
    }


def _province_centroid(
    G: nx.Graph,
    coords: Dict[Hashable, Tuple[float, float]],
) -> Tuple[float, float]:
    nodes = list(G.nodes())
    if not nodes:
        return (float("nan"), float("nan"))
    xs = [coords[n][0] for n in nodes if n in coords]
    ys = [coords[n][1] for n in nodes if n in coords]
    if not xs or not ys:
        return (float("nan"), float("nan"))
    return (float(np.mean(xs)), float(np.mean(ys)))


def _closest_node_pair_between_graphs(
    G1: nx.Graph,
    G2: nx.Graph,
    coords: Dict[Hashable, Tuple[float, float]],
) -> Tuple[Hashable, Hashable, float] | None:
    best = None
    best_dist = float("inf")
    for u in G1.nodes():
        if u not in coords:
            continue
        cu = coords[u]
        for v in G2.nodes():
            if v not in coords:
                continue
            d = _euclidean_dist(cu, coords[v])
            if d < best_dist:
                best_dist = d
                best = (u, v, d)
    return best


def _connect_provinces_with_mst(
    province_graphs: Dict[str, nx.Graph],
    mode: str,
    allowed_pairs: Set[Tuple[str, str]] | None = None,
) -> Tuple[nx.Graph, Dict[str, float]]:
    """Compose province graphs and add MST-based inter-province bridge edges."""
    if not province_graphs:
        return nx.Graph(), {
            "province_bridge_edges_added": 0.0,
            "province_bridge_pairs_considered": 0.0,
            "province_bridge_mode_used": -1.0,
        }

    combined = nx.compose_all(list(province_graphs.values()))
    provinces = sorted([p for p, g in province_graphs.items() if g.number_of_nodes() > 0])
    if len(provinces) < 2 or mode == "none":
        return combined, {
            "province_bridge_edges_added": 0.0,
            "province_bridge_pairs_considered": 0.0,
            "province_bridge_mode_used": 0.0,
        }

    coords_all: Dict[Hashable, Tuple[float, float]] = {}
    for g in province_graphs.values():
        for n in g.nodes():
            if n in coords_all:
                continue
            try:
                coords_all[n] = _node_xy(combined, n)
            except Exception:
                continue

    speed_mps = _median_speed_mps(combined)

    prov_graph = nx.Graph()
    prov_graph.add_nodes_from(provinces)
    pair_to_nodes: Dict[Tuple[str, str], Tuple[Hashable, Hashable, float]] = {}

    centroids = {
        p: _province_centroid(province_graphs[p], coords_all)
        for p in provinces
    }

    pair_count = 0
    for i, p1 in enumerate(provinces):
        for p2 in provinces[i + 1 :]:
            pair_count += 1
            if allowed_pairs is not None:
                kprov = (p1, p2) if p1 < p2 else (p2, p1)
                if kprov not in allowed_pairs:
                    continue
            g1 = province_graphs[p1]
            g2 = province_graphs[p2]
            near = _closest_node_pair_between_graphs(g1, g2, coords_all)
            if near is None:
                continue
            u, v, dmin = near
            pair_to_nodes[(p1, p2)] = (u, v, dmin)

            if mode == "mst_centroid":
                c1 = centroids[p1]
                c2 = centroids[p2]
                if any(np.isnan([c1[0], c1[1], c2[0], c2[1]])):
                    w = dmin
                else:
                    w = _euclidean_dist(c1, c2)
            else:
                w = dmin
            prov_graph.add_edge(p1, p2, weight=float(w))

    if prov_graph.number_of_edges() == 0:
        return combined, {
            "province_bridge_edges_added": 0.0,
            "province_bridge_pairs_considered": float(pair_count),
            "province_bridge_mode_used": 0.0,
        }

    mst_prov = list(nx.minimum_spanning_edges(prov_graph, algorithm="kruskal", data=False))

    added = 0
    for p1, p2 in mst_prov:
        key = (p1, p2) if (p1, p2) in pair_to_nodes else (p2, p1)
        if key not in pair_to_nodes:
            continue
        u, v, d = pair_to_nodes[key]
        if combined.has_edge(u, v):
            continue
        combined.add_edge(u, v, **_edge_attrs_from_length(d, speed_mps))
        added += 1

    mode_code = 1.0 if mode == "mst_centroid" else 2.0
    return combined, {
        "province_bridge_edges_added": float(added),
        "province_bridge_pairs_considered": float(pair_count),
        "province_bridge_mode_used": mode_code,
    }


def build_knn_null(
    G: nx.Graph,
    rng: random.Random,
    k: int,
    match_edge_count: bool,
    max_add_dist_quantile: float,
    weight_rule: str,
    add_mst_backbone: bool,
) -> Tuple[nx.Graph, Dict[str, float]]:
    """Build KNN-based spatial null graph with same nodes as original."""
    H = _build_empty_with_nodes(G)
    nodes = list(H.nodes())
    n = len(nodes)
    if n < 2:
        return H, {
            "generated_edges": 0.0,
            "removed_for_match": 0.0,
            "added_for_match": 0.0,
        }

    coords = {node: _node_xy(G, node) for node in nodes}
    k_eff = max(1, min(k, n - 1))

    edge_set = set()
    for u in nodes:
        dists = []
        for v in nodes:
            if u == v:
                continue
            dists.append((_euclidean_dist(coords[u], coords[v]), v))
        dists.sort(key=lambda x: x[0])
        for dist, v in dists[:k_eff]:
            edge_set.add(_edge_key(u, v))

    for key in edge_set:
        u, v = tuple(key)
        H.add_edge(u, v)

    stats = {
        "generated_edges": float(H.number_of_edges()),
        "removed_for_match": 0.0,
        "added_for_match": 0.0,
        "mst_edges_candidate": 0.0,
        "mst_edges_added": 0.0,
        "components_before_mst": float(nx.number_connected_components(H)),
        "components_after_mst": float(nx.number_connected_components(H)),
        "mst_fallback_chain": 0.0,
    }

    if add_mst_backbone:
        mst_stats = _add_mst_backbone(H, coords)
        stats.update(mst_stats)

    if match_edge_count:
        adjust = _match_edge_count_by_distance(
            H,
            coords,
            target_edges=G.number_of_edges(),
            max_add_dist_quantile=max_add_dist_quantile,
        )
        stats.update(adjust)

    weight_stats = _assign_weights_to_null_graph(
        H,
        G_original=G,
        coords=coords,
        rng=rng,
        weight_rule=weight_rule,
    )
    stats.update(weight_stats)

    return H, stats


def build_delaunay_null(
    G: nx.Graph,
    rng: random.Random,
    match_edge_count: bool,
    max_add_dist_quantile: float,
    delaunay_max_dist_quantile: float,
    weight_rule: str,
    add_mst_backbone: bool,
) -> Tuple[nx.Graph, Dict[str, float]]:
    """Build Delaunay-triangulation spatial null graph with same nodes as original."""
    H = _build_empty_with_nodes(G)
    nodes = list(H.nodes())
    n = len(nodes)
    if n < 3:
        return H, {
            "generated_edges": 0.0,
            "removed_for_match": 0.0,
            "added_for_match": 0.0,
            "delaunay_fallback_knn": 1.0,
            "mst_edges_candidate": 0.0,
            "mst_edges_added": 0.0,
            "components_before_mst": 0.0,
            "components_after_mst": 0.0,
            "mst_fallback_chain": 0.0,
        }

    coords = {node: _node_xy(G, node) for node in nodes}
    points = np.array([coords[node] for node in nodes], dtype=float)

    fallback_knn = 0.0
    edge_set = set()
    try:
        tri = Delaunay(points, qhull_options="QJ")
        for simplex in tri.simplices:
            a = nodes[int(simplex[0])]
            b = nodes[int(simplex[1])]
            c = nodes[int(simplex[2])]
            edge_set.add(_edge_key(a, b))
            edge_set.add(_edge_key(b, c))
            edge_set.add(_edge_key(a, c))
    except Exception:
        # Degenerate point set fallback: KNN graph.
        fallback_knn = 1.0
        return_graph, stats = build_knn_null(
            G,
            rng=rng,
            k=4,
            match_edge_count=match_edge_count,
            max_add_dist_quantile=max_add_dist_quantile,
            weight_rule=weight_rule,
            add_mst_backbone=add_mst_backbone,
        )
        stats["delaunay_fallback_knn"] = fallback_knn
        return return_graph, stats

    for key in edge_set:
        u, v = tuple(key)
        H.add_edge(u, v)

    # Trim overly long Delaunay edges to keep province geometry plausible.
    removed_long = 0
    empirical_lengths = [_edge_length(G, u, v, coords) for u, v in G.edges()]
    if empirical_lengths:
        max_allowed = float(np.quantile(empirical_lengths, delaunay_max_dist_quantile))
        for u, v in list(H.edges()):
            if _euclidean_dist(coords[u], coords[v]) > max_allowed:
                H.remove_edge(u, v)
                removed_long += 1

    stats = {
        "generated_edges": float(H.number_of_edges()),
        "removed_for_match": 0.0,
        "added_for_match": 0.0,
        "delaunay_fallback_knn": fallback_knn,
        "delaunay_removed_long_edges": float(removed_long),
        "mst_edges_candidate": 0.0,
        "mst_edges_added": 0.0,
        "components_before_mst": float(nx.number_connected_components(H)),
        "components_after_mst": float(nx.number_connected_components(H)),
        "mst_fallback_chain": 0.0,
    }

    if add_mst_backbone:
        mst_stats = _add_mst_backbone(H, coords)
        stats.update(mst_stats)

    if match_edge_count:
        adjust = _match_edge_count_by_distance(
            H,
            coords,
            target_edges=G.number_of_edges(),
            max_add_dist_quantile=max_add_dist_quantile,
        )
        stats.update(adjust)

    weight_stats = _assign_weights_to_null_graph(
        H,
        G_original=G,
        coords=coords,
        rng=rng,
        weight_rule=weight_rule,
    )
    stats.update(weight_stats)

    return H, stats


def rewire_spatial_degree_preserving(
    G: nx.Graph,
    rng: random.Random,
    swaps_per_edge: float = 5.0,
    max_dist_quantile: float = 0.95,
    beta: float = 5.0,
    max_attempt_factor: float = 50.0,
    rewire_time_rule: str = "rank_preserve_times",
) -> Tuple[nx.Graph, Dict[str, float]]:
    """
    Spatially constrained double-edge-swap rewiring.

    Preserved exactly:
    - Node set
    - Number of edges
    - Degree sequence

    Approximately preserved:
    - Spatial edge-length distribution (through hard cap + distance-penalized acceptance)
    """
    H = G.copy()

    if H.number_of_edges() < 2:
        return H, {
            "target_swaps": 0,
            "accepted_swaps": 0,
            "attempts": 0,
            "acceptance_rate": 0.0,
            "max_allowed_dist": float("nan"),
        }

    coords = {n: _node_xy(H, n) for n in H.nodes()}
    orig_lengths = [
        _edge_length(H, u, v, coords)
        for u, v in H.edges()
    ]
    max_allowed_dist = float(np.quantile(orig_lengths, max_dist_quantile))

    target_swaps = int(max(1, round(swaps_per_edge * H.number_of_edges())))
    max_attempts = int(max(100, round(max_attempt_factor * target_swaps)))

    accepted = 0
    attempts = 0

    edge_set = {_edge_key(u, v) for u, v in H.edges()}
    speed_mps = _median_speed_mps(G)

    while attempts < max_attempts and accepted < target_swaps:
        attempts += 1

        edges = list(H.edges())
        if len(edges) < 2:
            break

        (a, b), (c, d) = rng.sample(edges, 2)
        if len({a, b, c, d}) < 4:
            continue

        # Two candidate swap pairings.
        pairings = [((a, d), (c, b)), ((a, c), (b, d))]
        rng.shuffle(pairings)

        accepted_this_attempt = False
        for (u1, v1), (u2, v2) in pairings:
            if u1 == v1 or u2 == v2:
                continue

            new_k1 = _edge_key(u1, v1)
            new_k2 = _edge_key(u2, v2)
            old_k1 = _edge_key(a, b)
            old_k2 = _edge_key(c, d)

            # Check for duplicate edges (ignore the two edges being replaced).
            exists1 = new_k1 in edge_set and new_k1 not in {old_k1, old_k2}
            exists2 = new_k2 in edge_set and new_k2 not in {old_k1, old_k2}
            if exists1 or exists2:
                continue

            old_len1 = _edge_length(H, a, b, coords)
            old_len2 = _edge_length(H, c, d, coords)
            new_len1 = _euclidean_dist(coords[u1], coords[v1])
            new_len2 = _euclidean_dist(coords[u2], coords[v2])

            if new_len1 > max_allowed_dist or new_len2 > max_allowed_dist:
                continue

            old_total = old_len1 + old_len2
            new_total = new_len1 + new_len2
            delta_norm = (new_total - old_total) / max(1e-9, max_allowed_dist)

            accept = False
            if delta_norm <= 0:
                accept = True
            else:
                p_accept = math.exp(-beta * delta_norm)
                if rng.random() < p_accept:
                    accept = True

            if not accept:
                continue

            attrs1 = dict(H.get_edge_data(a, b, default={}))
            attrs2 = dict(H.get_edge_data(c, d, default={}))

            H.remove_edge(a, b)
            H.remove_edge(c, d)

            if rewire_time_rule == "scale_by_length":
                new_attrs1 = _update_edge_attrs(attrs1, new_len1, old_len1, scale_time_by_length=True)
                new_attrs2 = _update_edge_attrs(attrs2, new_len2, old_len2, scale_time_by_length=True)
            elif rewire_time_rule == "distance_speed":
                new_attrs1 = _edge_attrs_from_length(new_len1, speed_mps)
                new_attrs2 = _edge_attrs_from_length(new_len2, speed_mps)
            else:
                # Preserve only geometry here; assign time_s globally after rewiring.
                new_attrs1 = _update_edge_attrs(attrs1, new_len1, old_len1, scale_time_by_length=False)
                new_attrs2 = _update_edge_attrs(attrs2, new_len2, old_len2, scale_time_by_length=False)
                new_attrs1.pop("time_s", None)
                new_attrs2.pop("time_s", None)

            H.add_edge(u1, v1, **new_attrs1)
            H.add_edge(u2, v2, **new_attrs2)

            edge_set.remove(old_k1)
            edge_set.remove(old_k2)
            edge_set.add(new_k1)
            edge_set.add(new_k2)

            accepted += 1
            accepted_this_attempt = True
            break

        if not accepted_this_attempt:
            continue

    stats = {
        "target_swaps": target_swaps,
        "accepted_swaps": accepted,
        "attempts": attempts,
        "acceptance_rate": (accepted / attempts) if attempts > 0 else 0.0,
        "max_allowed_dist": max_allowed_dist,
    }

    if rewire_time_rule in {"rank_preserve_times", "sample_preserve_times"}:
        weight_stats = _assign_weights_to_null_graph(
            H,
            G_original=G,
            coords=coords,
            rng=rng,
            weight_rule=rewire_time_rule,
        )
        stats.update(weight_stats)
    else:
        stats.setdefault("weight_rule_fallback", 0.0)
        stats.setdefault("weight_rule_used", rewire_time_rule)

    return H, stats


def graph_metrics(G: nx.Graph) -> Dict[str, float]:
    out: Dict[str, float] = {
        "num_nodes": float(G.number_of_nodes()),
        "num_edges": float(G.number_of_edges()),
        "density": float(nx.density(G)),
        "is_connected": float(nx.is_connected(G)) if G.number_of_nodes() > 0 else 0.0,
        "num_components": float(nx.number_connected_components(G)) if G.number_of_nodes() > 0 else 0.0,
        "avg_clustering": float(nx.average_clustering(G)) if G.number_of_nodes() > 0 else 0.0,
    }

    if G.number_of_edges() > 0:
        lengths = []
        for u, v, data in G.edges(data=True):
            if isinstance(data, dict) and "length_m" in data and pd.notna(data["length_m"]):
                try:
                    lengths.append(float(data["length_m"]))
                except Exception:
                    pass
        out["mean_length_m"] = float(np.mean(lengths)) if lengths else float("nan")
    else:
        out["mean_length_m"] = float("nan")

    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    base_model = "knn" if args.model.startswith("knn") else "delaunay"
    add_mst_backbone = args.model.endswith("_mst")

    rewired: Dict[str, nx.Graph] = {}
    rows = []

    if args.scope == "province":
        input_path = _resolve_itinerex_path(args.itinereX_dir, args.input_pkl)
        print(f"Loading province subnetworks from: {input_path}")

        with open(input_path, "rb") as f:
            raw = pickle.load(f)

        if "province_subgraphs" not in raw:
            raise KeyError("Input pickle does not contain key 'province_subgraphs'.")

        subnetworks: Dict[str, nx.Graph] = raw["province_subgraphs"]
        print(f"Loaded {len(subnetworks)} province graphs")
        items = list(subnetworks.items())
    else:
        input_path = _resolve_itinerex_path(args.itinereX_dir, args.global_input_pkl)
        print(f"Loading full network from: {input_path}")
        with open(input_path, "rb") as f:
            raw = pickle.load(f)
        if "node_table" not in raw or "edge_table" not in raw:
            raise KeyError(
                "Global input pickle must contain 'node_table' and 'edge_table'."
            )
        G_global = _graph_from_tables(raw["node_table"], raw["edge_table"])
        print(
            f"Loaded full graph: {G_global.number_of_nodes()} nodes, "
            f"{G_global.number_of_edges()} edges"
        )
        items = [("__GLOBAL__", G_global)]

    for i, (province, G) in enumerate(items, start=1):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"[{i:>2}/{len(items)}] {province}: {n} nodes, {m} edges")

        if n < args.min_nodes or m < 2:
            rewired[province] = G.copy()
            orig = graph_metrics(G)
            null = graph_metrics(G)
            rows.append(
                {
                    "province": province,
                    "status": "skipped_small_graph",
                    **{f"orig_{k}": v for k, v in orig.items()},
                    **{f"null_{k}": v for k, v in null.items()},
                    "target_swaps": 0,
                    "accepted_swaps": 0,
                    "attempts": 0,
                    "acceptance_rate": 0.0,
                    "max_allowed_dist": float("nan"),
                    "generated_edges": float(G.number_of_edges()),
                    "removed_for_match": 0.0,
                    "added_for_match": 0.0,
                    "delaunay_fallback_knn": 0.0,
                }
            )
            continue

        if base_model == "knn":
            H, stats = build_knn_null(
                G,
                rng=rng,
                k=args.knn_k,
                match_edge_count=args.match_edge_count,
                max_add_dist_quantile=args.max_add_dist_quantile,
                weight_rule=args.weight_rule,
                add_mst_backbone=add_mst_backbone,
            )
            stats.setdefault("target_swaps", 0.0)
            stats.setdefault("accepted_swaps", 0.0)
            stats.setdefault("attempts", 0.0)
            stats.setdefault("acceptance_rate", 0.0)
            stats.setdefault("max_allowed_dist", float("nan"))
            stats.setdefault("delaunay_fallback_knn", 0.0)
        elif base_model == "delaunay":
            H, stats = build_delaunay_null(
                G,
                rng=rng,
                match_edge_count=args.match_edge_count,
                max_add_dist_quantile=args.max_add_dist_quantile,
                delaunay_max_dist_quantile=args.delaunay_max_dist_quantile,
                weight_rule=args.weight_rule,
                add_mst_backbone=add_mst_backbone,
            )
            stats.setdefault("target_swaps", 0.0)
            stats.setdefault("accepted_swaps", 0.0)
            stats.setdefault("attempts", 0.0)
            stats.setdefault("acceptance_rate", 0.0)
            stats.setdefault("max_allowed_dist", float("nan"))
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        if args.scope == "global" and not args.allow_sea_crossings:
            province_path = _resolve_itinerex_path(args.itinereX_dir, args.input_pkl)
            if province_path.exists():
                with open(province_path, "rb") as f:
                    prov_raw = pickle.load(f)
                if "province_subgraphs" in prov_raw:
                    # Build coordinate-based node mapping (works across different node ID schemes)
                    node_to_prov = _node_id_to_province_map_from_coords(G, prov_raw["province_subgraphs"])
                    allowed_pairs = _empirical_allowed_province_pairs(G, node_to_prov)
                    removed_mask = _filter_edges_by_province_mask(H, node_to_prov, allowed_pairs)
                    stats["removed_by_empirical_province_mask"] = float(removed_mask)

        rewired[province] = H

        orig = graph_metrics(G)
        null = graph_metrics(H)

        row = {
            "province": province,
            "status": "ok",
            **{f"orig_{k}": v for k, v in orig.items()},
            **{f"null_{k}": v for k, v in null.items()},
            **stats,
        }
        row["delta_density"] = row["null_density"] - row["orig_density"]
        row["delta_avg_clustering"] = row["null_avg_clustering"] - row["orig_avg_clustering"]
        row["delta_mean_length_m"] = row["null_mean_length_m"] - row["orig_mean_length_m"]
        rows.append(row)

    # Save rewired graphs
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    if args.scope == "province":
        payload_graph_key = "province_subgraphs"
    else:
        payload_graph_key = "graph"

    graph_payload = rewired if args.scope == "province" else rewired["__GLOBAL__"]
    bridge_stats: Dict[str, float] = {
        "province_bridge_edges_added": 0.0,
        "province_bridge_pairs_considered": 0.0,
        "province_bridge_mode_used": 0.0,
    }

    if args.scope == "province" and args.connect_provinces != "none":
        allowed_pairs = None
        if not args.allow_sea_crossings:
            # To get the empirical inter-province connections, we need the full empirical graph
            # and a mapping from its node IDs to province names.
            global_path = _resolve_itinerex_path(args.itinereX_dir, args.global_input_pkl)
            province_path = _resolve_itinerex_path(args.itinereX_dir, args.input_pkl)

            if global_path.exists() and province_path.exists():
                with open(global_path, "rb") as f:
                    global_raw = pickle.load(f)
                with open(province_path, "rb") as f:
                    prov_raw = pickle.load(f)

                if "node_table" in global_raw and "edge_table" in global_raw and "province_subgraphs" in prov_raw:
                    G_emp = _graph_from_tables(global_raw["node_table"], global_raw["edge_table"])
                    
                    # The empirical graph uses integer IDs, but province subgraphs use coordinate tuples.
                    # We build a map from coordinates to province names from the province data.
                    # Then we map the empirical graph's integer nodes to provinces via their coordinates.
                    node_to_prov = _node_id_to_province_map_from_coords(G_emp, prov_raw["province_subgraphs"])
                    
                    allowed_pairs = _empirical_allowed_province_pairs(G_emp, node_to_prov)
                    print(f"Found {len(allowed_pairs)} allowed empirical inter-province pairs.")
                else:
                    print("Warning: Could not load data needed to build empirical province mask.")
            else:
                print("Warning: Could not find global or province data to build empirical province mask.")

        combined_graph, bridge_stats = _connect_provinces_with_mst(
            province_graphs=rewired,
            mode=args.connect_provinces,
            allowed_pairs=allowed_pairs,
        )
        # Keep per-province graphs and also export combined network for downstream analyses.
        graph_payload = rewired
    else:
        combined_graph = rewired["__GLOBAL__"] if args.scope == "global" else None

    out_payload = {
        payload_graph_key: graph_payload,
        "meta": {
            "model": args.model,
            "base_model": base_model,
            "add_mst_backbone": add_mst_backbone,
            "scope": args.scope,
            "connect_provinces": args.connect_provinces,
            "allow_sea_crossings": args.allow_sea_crossings,
            "seed": args.seed,
            "knn_k": args.knn_k,
            "match_edge_count": args.match_edge_count,
            "max_add_dist_quantile": args.max_add_dist_quantile,
            "delaunay_max_dist_quantile": args.delaunay_max_dist_quantile,
            "weight_rule": args.weight_rule,
            "swaps_per_edge": args.swaps_per_edge,
            "max_dist_quantile": args.max_dist_quantile,
            "beta": args.beta,
            "max_attempt_factor": args.max_attempt_factor,
            "min_nodes": args.min_nodes,
            "source_input": str(input_path),
            **bridge_stats,
        },
    }

    if args.scope == "province" and combined_graph is not None:
        out_payload["graph"] = combined_graph

    with open(args.output_pkl, "wb") as f:
        pickle.dump(out_payload, f)

    # Save summary
    summary_df = pd.DataFrame(rows).sort_values("province")
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False)

    print("\nDone.")
    print(f"Null-model graphs saved to: {args.output_pkl}")
    print(f"Summary CSV saved to:    {args.output_summary_csv}")
    print("\nQuick check:")
    print(f"  Mean generated edges: {summary_df['null_num_edges'].mean():.1f}")
    if args.match_edge_count:
        print(f"  Mean edges removed for match: {summary_df['removed_for_match'].mean():.1f}")
        print(f"  Mean edges added for match:   {summary_df['added_for_match'].mean():.1f}")


if __name__ == "__main__":
    main()
