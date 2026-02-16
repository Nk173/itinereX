from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List, Sequence
import time
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.ops import snap as shapely_snap

try:
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:  # pragma: no cover
    DBSCAN = None

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


def _progress(iterable: Iterable, *, enabled: bool = True, **kwargs):
    if not enabled or _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)


def _progress_bar(*, total: int, desc: str, enabled: bool = True):
    if not enabled or _tqdm is None:
        return None
    return _tqdm(total=int(total), desc=str(desc))


def _tree_query_dwithin(tree: STRtree, geom, distance: float):
    """Return STRtree query results for items within distance of geom.

    Uses predicate='dwithin' when available (Shapely 2), otherwise falls back
    to querying with a buffer.
    """
    try:
        return tree.query(geom, predicate="dwithin", distance=float(distance))
    except TypeError:
        return tree.query(geom.buffer(float(distance)))

@dataclass
class CleanNetworkResult:
    roads_src: gpd.GeoDataFrame
    roads_m: gpd.GeoDataFrame
    noded_lines_m: gpd.GeoDataFrame
    segments_m: gpd.GeoDataFrame
    graph: nx.Graph
    diagnostics: dict

def load_roads_geojson(
    path: str,
    assume_crs: str = "EPSG:4326",
    metric_crs: str = "EPSG:3395",
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load roads, ensure CRS, return (src, metric) GeoDataFrames."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(assume_crs, allow_override=True)
    gdf_m = gdf.to_crs(metric_crs)
    return gdf, gdf_m

def prepare_lines(gdf_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode multilines, drop empties, keep LineStrings only."""
    gdf = gdf_m.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.explode(index_parts=True).reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type == "LineString"].copy().reset_index(drop=True)
    return gdf

def diagnose_near_miss_t_junctions(
    lines_m: gpd.GeoDataFrame,
    tol_m: float = 10.0,
    show_progress: bool = True,
    max_endpoints: Optional[int] = None,
) -> Dict[str, int]:
    """Count endpoints that are within tol_m of another line but not intersecting.
    This is the common reason you see a 'T' visually but no graph junction.
    """
    geoms = list(lines_m.geometry)
    tree = STRtree(geoms)

    near_miss = 0
    checked = 0
    print('Checking T-junction near-misses')
    for ls in _progress(geoms, enabled=show_progress):
        c = list(ls.coords)
        if len(c) < 2:
            continue
        for xy in (c[0], c[-1]):
            p = Point(float(xy[0]), float(xy[1]))
            checked += 1
            if max_endpoints is not None and checked > int(max_endpoints):
                return {
                    "endpoints_checked": int(checked - 1),
                    "near_miss_endpoints": int(near_miss),
                    "max_endpoints": int(max_endpoints),
                }

            cand = _tree_query_dwithin(tree, p, float(tol_m))

            # Shapely 2: indices; fallback: geometries
            if len(cand) == 0:
                continue
            if isinstance(cand[0], (int, np.integer)):
                cand_geoms = [geoms[int(i)] for i in cand]
            else:
                cand_geoms = list(cand)

            # Ignore self
            found = False
            for other in cand_geoms:
                if other is ls:
                    continue
                if p.distance(other) <= float(tol_m) and not p.intersects(other):
                    found = True
                    break
            if found:
                near_miss += 1

    return {
        "endpoints_checked": int(checked),
        "near_miss_endpoints": int(near_miss),
    }


def snap_lines_to_themselves(
    lines_m: gpd.GeoDataFrame,
    snap_tol_m: float = 10.0,
    method: str = "endpoints",
    show_progress: bool = True,
) -> gpd.GeoDataFrame:
    """Snap linework within snap_tol_m.

    Methods:
    - 'endpoints' (fast): snap only each line's endpoints onto nearby linework.
      This targets the common T-junction near-miss problem.
    - 'self' (slow): shapely.ops.snap(line, unary_union(all_lines), tol) for each line.
    - 'none': no snapping.
    """
    if snap_tol_m <= 0 or method == "none":
        return lines_m.copy()

    geoms = list(lines_m.geometry)

    if method == "self":
        target = unary_union(geoms)
        print('Snapping lines to themselves (full geometry)')
        snapped = [shapely_snap(g, target, float(snap_tol_m)) for g in _progress(geoms, enabled=show_progress)]
    else:
        # Fast path: snap endpoints only
        tree = STRtree(geoms)
        print('Snapping line endpoints to nearby linework')
        snapped = []
        for ls in _progress(geoms, enabled=show_progress):
            c = list(ls.coords)
            if len(c) < 2:
                snapped.append(ls)
                continue

            new_coords = [tuple(map(float, xy)) for xy in c]
            coord_dim = len(new_coords[0]) if len(new_coords) else 2

            for endpoint_index in (0, -1):
                xy = new_coords[endpoint_index]
                p = Point(float(xy[0]), float(xy[1]))
                cand = _tree_query_dwithin(tree, p, float(snap_tol_m))

                if len(cand) == 0:
                    continue
                if isinstance(cand[0], (int, np.integer)):
                    cand_items = [(int(i), geoms[int(i)]) for i in cand]
                else:
                    # STRtree returned geometries; recover indices by identity when possible
                    cand_items = [(None, g) for g in cand]

                best_pt = None
                best_dist = float("inf")
                for idx, other in cand_items:
                    if other is ls:
                        continue
                    d = float(p.distance(other))
                    if d == 0.0 or d > float(snap_tol_m):
                        continue
                    # Project p onto other line
                    q = other.interpolate(other.project(p))
                    dq = float(p.distance(q))
                    if dq < best_dist:
                        best_dist = dq
                        best_pt = q

                if best_pt is not None:
                    if coord_dim <= 2:
                        new_coords[endpoint_index] = (float(best_pt.x), float(best_pt.y))
                    else:
                        tail = tuple(new_coords[endpoint_index][2:])
                        new_coords[endpoint_index] = (float(best_pt.x), float(best_pt.y), *tail)

            # Avoid invalid/degenerate lines if both endpoints collapsed
            if len(new_coords) >= 2 and new_coords[0] != new_coords[-1]:
                snapped.append(LineString(new_coords))
            else:
                snapped.append(ls)

    out = lines_m.copy()
    out["geometry"] = snapped
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out[out.geometry.geom_type == "LineString"].copy().reset_index(drop=True)
    return out


def node_lines_at_intersections(lines_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Node/split linework at all (exact) intersections using unary_union."""
    u = unary_union(list(lines_m.geometry))

    parts: List[LineString] = []
    if u.geom_type == "LineString":
        parts = [u]
    elif u.geom_type == "MultiLineString":
        parts = list(u.geoms)
    else:
        # geometry collection: grab lines
        print(f"Unary union resulted in {u.geom_type}, extracting LineStrings")
        for gg in _progress(getattr(u, "geoms", [])):
            if gg.geom_type == "LineString":
                parts.append(gg)
            elif gg.geom_type == "MultiLineString":
                parts.extend(list(gg.geoms))

    out = gpd.GeoDataFrame({"geometry": parts}, crs=lines_m.crs)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out[out.geometry.geom_type == "LineString"].copy().reset_index(drop=True)
    return out


def lines_to_segments(lines_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert LineStrings (with vertices) into single-step segments between consecutive coords."""
    segs: List[LineString] = []
    print('Converting lines to segments')
    for ls in _progress(lines_m.geometry):
        c = list(ls.coords)
        if len(c) < 2:
            continue
        for a, b in zip(c[:-1], c[1:]):
            u = (float(a[0]), float(a[1]))
            v = (float(b[0]), float(b[1]))
            if u == v:
                continue
            segs.append(LineString([u, v]))

    gdf = gpd.GeoDataFrame({"geometry": segs}, crs=lines_m.crs)
    gdf["dist_m"] = gdf.geometry.length.astype(float)
    return gdf


def segments_to_graph(
    segments_m: gpd.GeoDataFrame,
    *,
    base_speed_kmh: Optional[float] = None,
    weight_mode: str = "dist",
) -> nx.Graph:
    """Build an undirected graph from segment endpoints.

    If base_speed_kmh is provided and weight_mode='time', edges will be weighted
    by travel time (seconds) and store both dist_m and time_s attributes.
    Otherwise, edges are weighted by distance (meters).
    """
    G = nx.Graph()
    print('Building graph from segments')
    speed_mps = None
    if base_speed_kmh is not None:
        speed_mps = float(base_speed_kmh) * 1000.0 / 3600.0
        if speed_mps <= 0:
            raise ValueError("base_speed_kmh must be > 0")

    use_time = (str(weight_mode).lower() == "time")

    for row in _progress(segments_m.itertuples(index=False)):
        geom = row.geometry
        c = list(geom.coords)
        u = (float(c[0][0]), float(c[0][1]))
        v = (float(c[-1][0]), float(c[-1][1]))
        dist_m = float(getattr(row, "dist_m", geom.length))
        time_s = float(dist_m / speed_mps) if (use_time and speed_mps is not None) else float("nan")
        w = float(time_s) if (use_time and speed_mps is not None) else float(dist_m)
        if G.has_edge(u, v):
            prev_w = float(G[u][v].get("weight", w))
            if w < prev_w:
                G[u][v].update(weight=w, dist_m=dist_m, time_s=time_s, geometry=geom)
        else:
            G.add_edge(u, v, weight=w, dist_m=dist_m, time_s=time_s, geometry=geom)
    return G


def _d2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2


def _oriented_coords(ls: LineString, start_xy: Tuple[float, float], tol_m: float = 0.05) -> List[Tuple[float, float]]:
    """Return coords oriented so the first coordinate matches start_xy (within tol)."""
    c = list(ls.coords)
    a = (float(c[0][0]), float(c[0][1]))
    b = (float(c[-1][0]), float(c[-1][1]))
    tol2 = float(tol_m) ** 2
    if _d2(a, start_xy) <= tol2:
        out = [(float(x), float(y)) for (x, y, *_) in c]
        out[0] = (float(start_xy[0]), float(start_xy[1]))
        return out
    if _d2(b, start_xy) <= tol2:
        cc = list(reversed(c))
        out = [(float(x), float(y)) for (x, y, *_) in cc]
        out[0] = (float(start_xy[0]), float(start_xy[1]))
        return out
    out = [(float(x), float(y)) for (x, y, *_) in c]
    out[0] = (float(start_xy[0]), float(start_xy[1]))
    return out


def simplify_degree2_topology(
    G: nx.Graph,
    max_iter: int = 10,
    *,
    additive_attrs: Sequence[str] = ("weight", "dist_m", "time_s"),
    geometry_attr: str = "geometry",
    geom_tol_m: float = 0.05,
) -> nx.Graph:
    """Remove degree-2 nodes in topology, until stable.

    additive_attrs are summed along collapsed chains when present.
    """
    cur = G
    print('Simplifying degree-2 nodes')
    for _ in _progress(range(int(max_iter))):
        important = {n for n, d in cur.degree() if int(d) != 2}
        H = nx.Graph()
        seen = set()

        for u in important:
            for v in cur.neighbors(u):
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                seen.add((v, u))

                prev, cur_node = u, v
                totals: Dict[str, float] = {}
                d0 = cur[u][v]
                for a in additive_attrs:
                    val = d0.get(a, 0.0)
                    try:
                        totals[a] = float(val)
                    except Exception:
                        totals[a] = 0.0

                coords_accum: Optional[List[Tuple[float, float]]] = None
                g0 = d0.get(geometry_attr)
                if isinstance(g0, LineString):
                    coords_accum = _oriented_coords(g0, (float(u[0]), float(u[1])), tol_m=float(geom_tol_m))
                    coords_accum[0] = (float(u[0]), float(u[1]))
                    coords_accum[-1] = (float(v[0]), float(v[1]))

                while cur_node not in important:
                    nbrs = [n for n in cur.neighbors(cur_node) if n != prev]
                    if len(nbrs) != 1:
                        break
                    nxt = nbrs[0]
                    d1 = cur[cur_node][nxt]
                    for a in additive_attrs:
                        val = d1.get(a, 0.0)
                        try:
                            totals[a] += float(val)
                        except Exception:
                            pass

                    if coords_accum is not None:
                        g1 = d1.get(geometry_attr)
                        if isinstance(g1, LineString):
                            cc = _oriented_coords(g1, (float(cur_node[0]), float(cur_node[1])), tol_m=float(geom_tol_m))
                            if cc:
                                coords_accum.extend(cc[1:])
                    prev, cur_node = cur_node, nxt
                    seen.add((prev, cur_node))
                    seen.add((cur_node, prev))

                if u == cur_node:
                    continue
                if coords_accum is not None and len(coords_accum) >= 2:
                    coords_accum[0] = (float(u[0]), float(u[1]))
                    coords_accum[-1] = (float(cur_node[0]), float(cur_node[1]))
                    try:
                        totals[geometry_attr] = LineString(coords_accum)  # type: ignore[assignment]
                    except Exception:
                        pass
                if H.has_edge(u, cur_node):
                    prev_w = float(H[u][cur_node].get("weight", float("inf")))
                    new_w = float(totals.get("weight", 0.0))
                    if new_w < prev_w:
                        H[u][cur_node].update(**totals)
                else:
                    H.add_edge(u, cur_node, **totals)

        for n in important:
            if n not in H:
                H.add_node(n)

        if H.number_of_nodes() == cur.number_of_nodes() and H.number_of_edges() == cur.number_of_edges():
            return H
        cur = H

    return cur


def merge_close_degree1_nodes(
    G: nx.Graph,
    tol_m: float,
    *,
    min_samples: int = 2,
    prec_m: float = 0.01,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """Merge clusters of degree-1 nodes that are within tol_m of each other.

    This targets the common visual artifact where two nearby dangling endpoints
    look like a degree-2 junction on the map.

    Notes:
    - Only degree-1 nodes are eligible to be merged.
    - Edges are rewired to a cluster centroid node.
    - Parallel edges are collapsed by keeping the minimum weight.
    """
    tol_m = float(tol_m)
    if tol_m <= 0:
        return G, {"enabled": 0, "deg1_before": int(sum(1 for _, d in G.degree() if int(d) == 1))}
    if DBSCAN is None:
        raise RuntimeError("scikit-learn is required for merge_close_degree1_nodes (DBSCAN import failed)")

    deg = dict(G.degree())
    deg1_nodes = [n for n, d in deg.items() if int(d) == 1]
    deg1_before = int(len(deg1_nodes))
    if deg1_before < int(min_samples):
        return G, {"enabled": 1, "deg1_before": deg1_before, "clusters": 0, "merged_nodes": 0}

    coords = np.asarray([(float(n[0]), float(n[1])) for n in deg1_nodes], dtype=float)
    labels = DBSCAN(eps=tol_m, min_samples=int(min_samples)).fit(coords).labels_

    cluster_ids = sorted({int(l) for l in labels.tolist() if int(l) >= 0})
    if not cluster_ids:
        return G, {"enabled": 1, "deg1_before": deg1_before, "clusters": 0, "merged_nodes": 0}

    def _quantize_xy(xy: Tuple[float, float]) -> Tuple[float, float]:
        if prec_m <= 0:
            return (float(xy[0]), float(xy[1]))
        qx = round(float(xy[0]) / float(prec_m)) * float(prec_m)
        qy = round(float(xy[1]) / float(prec_m)) * float(prec_m)
        return (float(qx), float(qy))

    mapping: Dict[Tuple[float, float], Tuple[float, float]] = {}
    merged_nodes = 0

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        if idx.size < int(min_samples):
            continue
        centroid = coords[idx].mean(axis=0)
        new_node = _quantize_xy((float(centroid[0]), float(centroid[1])))
        for i in idx.tolist():
            old = (float(coords[i][0]), float(coords[i][1]))
            mapping[old] = new_node
        merged_nodes += int(idx.size)

    if not mapping:
        return G, {"enabled": 1, "deg1_before": deg1_before, "clusters": 0, "merged_nodes": 0}

    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        uu = (float(u[0]), float(u[1]))
        vv = (float(v[0]), float(v[1]))
        uu2 = mapping.get(uu, uu)
        vv2 = mapping.get(vv, vv)
        if uu2 == vv2:
            continue
        w = float(data.get("weight", 0.0))
        dist_m = float(data.get("dist_m", float("nan")))
        time_s = float(data.get("time_s", float("nan")))
        geom = data.get("geometry")

        geom2 = None
        if isinstance(geom, LineString):
            cc = list(geom.coords)
            a0 = (float(cc[0][0]), float(cc[0][1]))
            b0 = (float(cc[-1][0]), float(cc[-1][1]))
            uu0 = (float(u[0]), float(u[1]))
            vv0 = (float(v[0]), float(v[1]))
            tol2 = float(prec_m if prec_m > 0 else 0.05) ** 2
            if _d2(a0, uu0) <= tol2 and _d2(b0, vv0) <= tol2:
                pass
            elif _d2(a0, vv0) <= tol2 and _d2(b0, uu0) <= tol2:
                cc = list(reversed(cc))
            cc[0] = (float(uu2[0]), float(uu2[1]))
            cc[-1] = (float(vv2[0]), float(vv2[1]))
            if len(cc) >= 2 and cc[0] != cc[-1]:
                geom2 = LineString([(float(x), float(y)) for (x, y, *_) in cc])
        if H.has_edge(uu2, vv2):
            prev_w = float(H[uu2][vv2].get("weight", float("inf")))
            if w < prev_w:
                H[uu2][vv2].update(weight=w, dist_m=dist_m, time_s=time_s, geometry=geom2)
        else:
            H.add_edge(uu2, vv2, weight=w, dist_m=dist_m, time_s=time_s, geometry=geom2)

    # Preserve isolated nodes (rare), including any cluster centroids
    for n in G.nodes():
        nn = (float(n[0]), float(n[1]))
        nn2 = mapping.get(nn, nn)
        if nn2 not in H:
            H.add_node(nn2)

    deg1_after = int(sum(1 for _, d in H.degree() if int(d) == 1))
    return H, {
        "enabled": 1,
        "deg1_before": deg1_before,
        "deg1_after": deg1_after,
        "clusters": int(len(cluster_ids)),
        "merged_nodes": int(merged_nodes),
    }


def build_clean_network_from_geojson(
    path: str,
    assume_crs: str = "EPSG:4326",
    metric_crs: str = "EPSG:3395",
    snap_tol_m: float = 10.0,
    simplify_deg2: bool = False,
    show_progress: bool = True,
    progress_desc: str = "Clean rebuild",
    collect_timings: bool = True,
    diagnose: bool = True,
    diagnose_max_endpoints: Optional[int] = None,
    snap_method: str = "endpoints",
    merge_deg1_tol_m: float = 0.0,
    merge_deg1_min_samples: int = 2,
    merge_deg1_prec_m: float = 0.01,
    base_speed_kmh: Optional[float] = None,
    weight_mode: str = "dist",
) -> CleanNetworkResult:
    timings_s: Dict[str, float] = {}

    step_names = [
        "load",
        "prepare",
        "diagnose_before",
        "snap",
        "diagnose_after",
        "node",
        "segmentize",
        "graph",
    ]
    if float(merge_deg1_tol_m) > 0:
        step_names.append("merge_deg1")
    if simplify_deg2:
        step_names.append("simplify_deg2")

    pbar = _progress_bar(total=len(step_names), desc=progress_desc, enabled=show_progress)
    step_i = 0

    def _step(name: str, fn):
        nonlocal step_i
        step_i += 1
        if pbar is None and show_progress:
            print(f"[{step_i}/{len(step_names)}] {name}...")
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        if collect_timings:
            timings_s[name] = float(t1 - t0)
        if pbar is not None:
            pbar.set_postfix_str(name)
            pbar.update(1)
        return out

    try:
        roads_src, roads_m = _step(
            "load",
            lambda: load_roads_geojson(path, assume_crs=assume_crs, metric_crs=metric_crs),
        )
        lines = _step("prepare", lambda: prepare_lines(roads_m))

        if diagnose:
            diag_before = _step(
                "diagnose_before",
                lambda: diagnose_near_miss_t_junctions(
                    lines,
                    tol_m=snap_tol_m,
                    show_progress=show_progress,
                    max_endpoints=diagnose_max_endpoints,
                ),
            )
        else:
            diag_before = {"skipped": 1}

        lines_snapped = _step(
            "snap",
            lambda: snap_lines_to_themselves(
                lines,
                snap_tol_m=snap_tol_m,
                method=snap_method,
                show_progress=show_progress,
            ),
        )

        if diagnose:
            diag_after = _step(
                "diagnose_after",
                lambda: diagnose_near_miss_t_junctions(
                    lines_snapped,
                    tol_m=snap_tol_m,
                    show_progress=show_progress,
                    max_endpoints=diagnose_max_endpoints,
                ),
            )
        else:
            diag_after = {"skipped": 1}

        noded = _step("node", lambda: node_lines_at_intersections(lines_snapped))
        segments = _step("segmentize", lambda: lines_to_segments(noded))
        G = _step(
            "graph",
            lambda: segments_to_graph(
                segments,
                base_speed_kmh=base_speed_kmh,
                weight_mode=weight_mode,
            ),
        )

        if float(merge_deg1_tol_m) > 0:
            G, merge_info = _step(
                "merge_deg1",
                lambda: merge_close_degree1_nodes(
                    G,
                    float(merge_deg1_tol_m),
                    min_samples=int(merge_deg1_min_samples),
                    prec_m=float(merge_deg1_prec_m),
                ),
            )
        else:
            merge_info = {"enabled": 0}

        if simplify_deg2:
            G = _step("simplify_deg2", lambda: simplify_degree2_topology(G))
    finally:
        if pbar is not None:
            pbar.close()

    diagnostics = {
        "snap_tol_m": float(snap_tol_m),
        "near_miss_before": diag_before,
        "near_miss_after": diag_after,
        "noded_lines": int(len(noded)),
        "segments": int(len(segments)),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "deg2_nodes": int(sum(1 for _, d in G.degree() if int(d) == 2)),
        "deg1_nodes": int(sum(1 for _, d in G.degree() if int(d) == 1)),
        "timings_s": timings_s,
        "snap_method": str(snap_method),
        "diagnose": bool(diagnose),
        "diagnose_max_endpoints": None if diagnose_max_endpoints is None else int(diagnose_max_endpoints),
        "merge_deg1": merge_info,
        "base_speed_kmh": None if base_speed_kmh is None else float(base_speed_kmh),
        "weight_mode": str(weight_mode),
    }

    return CleanNetworkResult(
        roads_src=roads_src,
        roads_m=roads_m,
        noded_lines_m=noded,
        segments_m=segments,
        graph=G,
        diagnostics=diagnostics,
    )
