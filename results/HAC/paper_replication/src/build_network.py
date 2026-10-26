"""
build_network.py
Reconstruct the Roman road network exactly as in Pažout, Brughmans, de Soto,
Mazzamurro et al., "Network Science reveals the structure and lasting impact of
the Roman road system" (Sci. Data resubmission 2025), so that downstream HAC /
percolation analysis runs on the *same* graph the manuscript analyses.

Source data (cloned, untouched):
  paper_replication/roman-road-networks/data/roman_roads/roman_roads.shp
      – the Itiner-e export used by the manuscript. 14,901 LineString segments,
        already split at intersections, in EPSG:3395. Carries the pre-computed
        fields the manuscript uses:  LENGTH_GEO (geodesic length, m) and
        Avg_Slope (mean slope along the segment, %).
  paper_replication/roman-road-networks/data/roman_provinces_simple/roman_provinces.shp
      – AWMC "Roman Empire AD 200" provinces (+ Italian regiones), EPSG:4326,
        corrected by A. Pažout 2023.

Network construction (mirrors network-analysis.Rmd, chunk "build roads networks"):
  * nodes  = LineString endpoints (the manuscript uses sfnetworks::as_sfnetwork,
             directed = FALSE, which nodes only at segment endpoints – the
             segments are already noded at intersections upstream).
  * edge weight (travel time, seconds) via Tobler's hiking function:
        slope = Avg_Slope / 100
        pace  = 0.6 * exp(3.5 * |slope + 0.05|)      [s / m]
        time_s = LENGTH_GEO * pace
    (identical to the Rmd:  pace <- 3/5*exp(7/2*(abs(slope+0.05))) )
  * the two ancient ferry links (Bosporus, Dardanelles) that the manuscript adds
    to join Europe<->Asia are ALREADY PRESENT in this Itiner-e export – verified:
    the giant component already spans lon [-9.4, 41.6], lat [22.7, 52.2] and
    contains both the Byzantium and Chalcedon sides. No links are added here.

Endpoint snapping: coordinates are rounded to 2 dp in EPSG:3395 (1 cm) before
matching. Parallel segments between the same pair of nodes are collapsed to the
single fastest one (min time_s); 2 zero-length segments are dropped. This yields
10,506 nodes / 14,770 edges / 13 components  (manuscript: 10,516 / 14,901 / 13 –
the small deltas are the dropped self-loops and collapsed parallels).

Outputs (paper_replication/data/):
  network_paper.gpkg         – nodes + edges as GeoDataFrames (EPSG:4326)
  network_paper.pkl          – dict: nx.Graph + node/edge tables + id maps
                               (same shape centrality_utils.load_graph expects)

Usage (from repo root):
  /Users/nk821/opt/anaconda3/envs/MIR/bin/python paper_replication/src/build_network.py
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]          # paper_replication/
CLONE     = ROOT / "roman-road-networks"
ROADS_SHP = CLONE / "data/roman_roads/roman_roads.shp"
PROV_SHP  = CLONE / "data/roman_provinces_simple/roman_provinces.shp"

OUT_GPKG  = ROOT / "data/network_paper.gpkg"
OUT_PKL   = ROOT / "data/network_paper.pkl"

SNAP_DP   = 2          # round EPSG:3395 coords to this many decimals (1 cm)


# ── Tobler travel time ───────────────────────────────────────────────────────
def tobler_time_s(length_m: np.ndarray, avg_slope_pct: np.ndarray) -> np.ndarray:
    """Seconds to traverse a segment, Tobler's hiking function (manuscript form)."""
    slope = np.asarray(avg_slope_pct, dtype=float) / 100.0
    pace  = 0.6 * np.exp(3.5 * np.abs(slope + 0.05))     # s / m
    return np.asarray(length_m, dtype=float) * pace


# ── Build ────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Reading roads: {ROADS_SHP}")
    roads = gpd.read_file(ROADS_SHP)
    print(f"  {len(roads)} segments, CRS {roads.crs}")
    roads_4326 = roads.to_crs(4326)

    # LENGTH_GEO field, with fallback to lengthGeo (as the Rmd does)
    length_m = roads["LENGTH_GEO"].where(roads["LENGTH_GEO"] > 0, roads["lengthGeo"])
    slope_pct = roads["Avg_Slope"].fillna(0.0)
    t_s = tobler_time_s(length_m.to_numpy(), slope_pct.to_numpy())

    # ── assemble edges keyed by snapped endpoints ────────────────────────────
    rec = []          # (a_xy, b_xy, time_s, length_m, slope, type, certainty, lo, hi, name, geom4326)
    n_selfloop = 0
    for i, row in enumerate(roads.itertuples(index=False)):
        g = row.geometry
        if g is None or g.is_empty:
            n_selfloop += 1
            continue
        c = list(g.coords)
        a = (round(c[0][0], SNAP_DP), round(c[0][1], SNAP_DP))
        b = (round(c[-1][0], SNAP_DP), round(c[-1][1], SNAP_DP))
        if a == b:
            n_selfloop += 1
            continue
        rec.append((
            a, b, float(t_s[i]), float(length_m.iloc[i]), float(slope_pct.iloc[i]),
            getattr(row, "Type", None), getattr(row, "Segment_s", None),
            getattr(row, "Lower_Date", None), getattr(row, "Upper_Date", None),
            getattr(row, "Name", None), roads_4326.geometry.iloc[i],
        ))
    print(f"  dropped {n_selfloop} empty/zero-length segments")

    # ── stable node ids (sorted by coord) ───────────────────────────────────
    coords = sorted({r[0] for r in rec} | {r[1] for r in rec})
    xy_to_id = {xy: i + 1 for i, xy in enumerate(coords)}
    print(f"  {len(coords)} unique nodes")

    # ── collapse parallel edges: keep fastest ───────────────────────────────
    best: dict[tuple[int, int], tuple] = {}
    for (a, b, ts, lm, sl, ty, ce, lo, hi, nm, geom) in rec:
        u, v = xy_to_id[a], xy_to_id[b]
        key = (u, v) if u < v else (v, u)
        if key not in best or ts < best[key][2]:
            best[key] = (key[0], key[1], ts, lm, sl, ty, ce, lo, hi, nm, geom)
    edges = list(best.values())
    print(f"  {len(edges)} edges after collapsing parallels")

    # ── node -> province (point in polygon, AWMC AD200) ─────────────────────
    print("Assigning provinces to nodes (point-in-polygon)...")
    prov = gpd.read_file(PROV_SHP)[["province", "geometry"]]
    id_to_xy = {i: xy for xy, i in xy_to_id.items()}
    nodes_gdf = gpd.GeoDataFrame(
        {"node_id": list(id_to_xy.keys())},
        geometry=[Point(*id_to_xy[i]) for i in id_to_xy],
        crs="EPSG:3395",
    ).to_crs(4326)
    joined = gpd.sjoin(nodes_gdf, prov, how="left", predicate="within")
    joined = joined.drop_duplicates("node_id", keep="first")
    nid_prov = dict(zip(joined["node_id"], joined["province"]))

    # coarse province: merge Italian regiones + Roma into "Italia"
    def coarse(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "Unknown"
        if "Regio" in p or p == "Roma":
            return "Italia"
        return p
    nid_prov_coarse = {k: coarse(v) for k, v in nid_prov.items()}
    nid_prov = {k: (v if isinstance(v, str) else "Unknown") for k, v in nid_prov.items()}
    n_known = sum(v != "Unknown" for v in nid_prov.values())
    print(f"  {n_known} / {len(nid_prov)} nodes inside a province polygon")

    # ── networkx graph ─────────────────────────────────────────────────────
    G = nx.Graph()
    lon = {i: nodes_gdf.geometry.iloc[k].x for k, i in enumerate(nodes_gdf["node_id"])}
    lat = {i: nodes_gdf.geometry.iloc[k].y for k, i in enumerate(nodes_gdf["node_id"])}
    for i, xy in id_to_xy.items():
        G.add_node(i, node_id=i, x=xy[0], y=xy[1], lon=lon[i], lat=lat[i],
                   province=nid_prov.get(i, "Unknown"),
                   province_coarse=nid_prov_coarse.get(i, "Unknown"))
    for eid, (u, v, ts, lm, sl, ty, ce, lo, hi, nm, geom) in enumerate(edges, start=1):
        G.add_edge(u, v, edge_id=eid, time_s=ts, weight=ts, length_m=lm,
                   slope_pct=sl, road_type=ty, certainty=ce,
                   lower_date=lo, upper_date=hi, name=nm)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(comps)} components")
    for i, c in enumerate(comps[:8]):
        print(f"  [{i}] {len(c):5d} nodes")

    # ── tables ─────────────────────────────────────────────────────────────
    node_table = pd.DataFrame({
        "node_id": list(id_to_xy.keys()),
        "x": [id_to_xy[i][0] for i in id_to_xy],
        "y": [id_to_xy[i][1] for i in id_to_xy],
        "lon": [lon[i] for i in id_to_xy],
        "lat": [lat[i] for i in id_to_xy],
        "province": [nid_prov.get(i, "Unknown") for i in id_to_xy],
        "province_coarse": [nid_prov_coarse.get(i, "Unknown") for i in id_to_xy],
        "degree": [G.degree(i) for i in id_to_xy],
    })
    edge_table = pd.DataFrame([{
        "edge_id": d["edge_id"], "u_id": u, "v_id": v,
        "time_s": d["time_s"], "length_m": d["length_m"], "slope_pct": d["slope_pct"],
        "road_type": d["road_type"], "certainty": d["certainty"],
        "lower_date": d["lower_date"], "upper_date": d["upper_date"], "name": d["name"],
    } for u, v, d in G.edges(data=True)])

    nodes_4326 = gpd.GeoDataFrame(
        node_table, geometry=gpd.points_from_xy(node_table.lon, node_table.lat),
        crs="EPSG:4326")
    edge_geom = {d["edge_id"]: g for (_, _, _, _, _, _, _, _, _, _, g), d
                 in zip(edges, (dd for _, _, dd in G.edges(data=True)))}
    edges_4326 = gpd.GeoDataFrame(
        edge_table, geometry=[edge_geom[e] for e in edge_table.edge_id], crs="EPSG:4326")

    # ── save ───────────────────────────────────────────────────────────────
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    nodes_4326.to_file(OUT_GPKG, layer="nodes", driver="GPKG")
    edges_4326.to_file(OUT_GPKG, layer="edges", driver="GPKG")
    print(f"\nGeoPackage -> {OUT_GPKG}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({
            "graph": G,
            "id_to_node": id_to_xy,                 # node_id -> (x, y) EPSG:3395
            "node_table": node_table,
            "edge_table": edge_table,
            "nodes_4326": nodes_4326,
            "edges_4326": edges_4326,
            "crs_metric": "EPSG:3395",
            "params": {"snap_dp": SNAP_DP, "tobler": "0.6*exp(3.5*|slope+0.05|)"},
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle    -> {OUT_PKL}")


if __name__ == "__main__":
    main()
