# data_cleaning/

Turns the raw Itiner-e linework into a cleaned, weighted, analyzable graph.

```
raw/           Raw inputs: itinere_roads.* (source linework shapefile/geojson/gpkg),
               route-segments-all-*.ndjson (raw route segment export)
scripts/       itinerex_clean_network.py (cleaning library used by notebooks/itinere-X_network.ipynb)
               export_geojson.py (re-exports the cleaned graph as rrn_edges/nodes.geojson)
cleaned_data/  Cleaned/derived outputs: G_clean_weighted_with_ids.pkl, G_clean_weighted_edges_with_province.csv,
               province_subnetworks.pkl, crossing_edges.geojson, outside_provinces_edges.geojson,
               rrn_edges.geojson, rrn_nodes.geojson
```

Run `notebooks/itinere-X_network.ipynb` top-to-bottom to regenerate everything
in `cleaned_data/` from `raw/`. Downstream analyses (`results/HAC/`,
`results/Null-models/`, `src/centrality_*.py`) read from `cleaned_data/`.
