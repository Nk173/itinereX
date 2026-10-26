"""
export_geojson.py
Export the cleaned Roman road network to GeoJSON for use in QGIS.

Outputs (written to data/):
  - rrn_edges.geojson  – road segments as LineStrings with original metadata
  - rrn_nodes.geojson  – intersections/terminals as Points

Raw metadata (name, dates, type, certainty, etc.) is joined from the source
NDJSON by finding the raw segment with the greatest geometric overlap for
each cleaned edge. Because the cleaning process splits and merges raw
segments, the join is approximate — most edges will match their source
segment well, but long merged edges may inherit metadata from one of several
contributing segments.

Usage (from repo root):
  python src/export_geojson.py
"""

import json
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import shape

PICKLE_PATH   = Path("data_cleaning/cleaned_data/G_clean_weighted_with_ids.pkl")
PROVINCE_CSV  = Path("data_cleaning/cleaned_data/G_clean_weighted_edges_with_province.csv")
RAW_NDJSON    = Path("data_cleaning/raw/route-segments-all-1764347876852.ndjson")
OUT_EDGES     = Path("data_cleaning/cleaned_data/rrn_edges.geojson")
OUT_NODES     = Path("data_cleaning/cleaned_data/rrn_nodes.geojson")

# Fields to carry over from the raw source data
RAW_FIELDS = [
    "name", "type", "lowerDate", "lowerDateError",
    "upperDate", "upperDateError", "constructionPeriod",
    "segmentCertainty", "description", "itinerary",
    "author", "bibliography", "passability",
]

# ── Load cleaned network ──────────────────────────────────────────────────────
print(f"Loading {PICKLE_PATH}...")
data = pickle.load(open(PICKLE_PATH, "rb"))
edges: gpd.GeoDataFrame = data["edges_4326"].copy()
nodes: gpd.GeoDataFrame = data["nodes_4326"].copy()

print(f"Loading {PROVINCE_CSV}...")
prov = pd.read_csv(PROVINCE_CSV, usecols=["edge_id", "province", "is_crossing", "is_outside"])
edges = edges.merge(prov, on="edge_id", how="left")

# ── Load raw source segments ──────────────────────────────────────────────────
print(f"Loading {RAW_NDJSON}...")
raw_features = []
with open(RAW_NDJSON) as f:
    for line in f:
        feat = json.loads(line)
        props = feat.get("properties", {})
        geom = shape(feat["geometry"])
        row = {field: props.get(field) for field in RAW_FIELDS}
        row["geometry"] = geom
        raw_features.append(row)

raw_gdf = gpd.GeoDataFrame(raw_features, crs="EPSG:4326")
print(f"  {len(raw_gdf)} raw segments loaded")

# ── Spatial join: match each cleaned edge to its best raw segment ─────────────
# Project to metric CRS for accurate length calculations
print("Joining raw metadata to cleaned edges (spatial overlap)...")
METRIC_CRS = "EPSG:3395"
edges_m = edges[["edge_id", "geometry"]].to_crs(METRIC_CRS)
raw_m   = raw_gdf.to_crs(METRIC_CRS).reset_index().rename(columns={"index": "raw_idx"})

# Candidate pairs via spatial index (intersects)
joined = gpd.sjoin(edges_m, raw_m[["raw_idx", "geometry"]], how="left", predicate="intersects")

# For each cleaned edge, keep the raw segment with the longest intersection
def _best_overlap(group, edges_geom, raw_geom):
    eid = group.name
    e_geom = edges_geom[eid]
    best_len = -1.0
    best_raw = None
    for raw_idx in group["raw_idx"].dropna().astype(int):
        try:
            inter = e_geom.intersection(raw_geom[raw_idx])
            if inter.is_empty:
                continue
            length = float(inter.length)
            if length > best_len:
                best_len = length
                best_raw = raw_idx
        except Exception:
            continue
    return best_raw

edges_geom_dict = dict(zip(edges_m["edge_id"], edges_m.geometry))
raw_geom_dict   = dict(zip(raw_m["raw_idx"],   raw_m.geometry))

print("  Computing best-overlap match per edge...")
best_raw_idx = (
    joined.groupby("edge_id")
    .apply(_best_overlap, edges_geom=edges_geom_dict, raw_geom=raw_geom_dict)
    .rename("best_raw_idx")
)

# Pull metadata from the best-matching raw segment
raw_meta = raw_m.set_index("raw_idx")[RAW_FIELDS]
meta_df  = (
    best_raw_idx
    .dropna()
    .astype(int)
    .map(lambda i: raw_meta.loc[i] if i in raw_meta.index else pd.Series({f: None for f in RAW_FIELDS}))
)

if isinstance(meta_df, pd.Series):
    # map() returned a Series of Series — expand to DataFrame
    meta_df = pd.DataFrame(meta_df.tolist(), index=meta_df.index)

meta_df.index.name = "edge_id"
meta_df = meta_df.reset_index()

edges = edges.merge(meta_df, on="edge_id", how="left")

# ── Tidy up ───────────────────────────────────────────────────────────────────
edges["dist_m"] = edges["dist_m"].round(2)
edges["time_s"] = edges["time_s"].round(2)
edges = edges.drop(columns=["weight"])

# ── Export edges ──────────────────────────────────────────────────────────────
print(f"Writing {OUT_EDGES}  ({len(edges)} edges)...")
edges.to_file(OUT_EDGES, driver="GeoJSON")

# ── Export nodes ──────────────────────────────────────────────────────────────
nodes_out = nodes.drop(columns=["x", "y"])
print(f"Writing {OUT_NODES}  ({len(nodes_out)} nodes)...")
nodes_out.to_file(OUT_NODES, driver="GeoJSON")

print("\nDone.")
print(f"  Edges → {OUT_EDGES}")
print(f"  Nodes → {OUT_NODES}")
print("\nFields in edges GeoJSON:")
for col in edges.columns:
    if col != "geometry":
        print(f"  {col}")
