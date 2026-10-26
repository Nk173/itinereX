# results/Null-models/

Spatial/temporal null models for the Roman road network: alternative graphs
built on the same node set under purely geometric or temporal connection
rules, used as a baseline to judge whether the empirical network's structure
(e.g. its [[HAC]] percolation / province alignment) is more than what
geography alone would produce.

## Provenance

This is the null-model subset of
[github.com/hayaalmuhanna/Roman-Provinces-ECI](https://github.com/hayaalmuhanna/Roman-Provinces-ECI)
(CC BY-SA 4.0 — see `LICENSE` in that repo), adapted to run against this
repo's own cleaned network. The upstream repo also contains an unrelated ECI
(Economic Complexity Index) analysis, which was **not** brought in here —
only the null-model pipeline, the KNN/Delaunay/MST spatial-null scripts, the
HAC-on-nulls scripts, and their already-generated outputs.

Path constants in the scripts below were rewritten from the original
author's local machine paths / the upstream repo's own layout to this repo's
layout (`data_cleaning/cleaned_data/` for the empirical graph,
`results/Null-models/...` for outputs). Logic was not otherwise modified.

## Contents

```
src/
  null_pipeline/                 Package: four null-model families (empire/province × spatial/temporal),
                                  topology generation, weight assignment, HAC + scaling metrics, comparison
  run_null_model_pipeline.py     CLI entry point for null_pipeline
  spatial_null_model_rewire.py   KNN / Delaunay (+MST backbone) spatial nulls, global or per-province,
                                  with empirical-adjacency masking against unrealistic sea crossings
  apply_hac_to_nulls.py          Apply itinereX's HAC method to a KNN/Delaunay null graph
  apply_hac_to_nulls_v2.py       Same, extended to MST-backboned and province-bridged variants
  plot_null_model_maps.py        Render empirical + null-model network maps side by side
docs/
  NULL_MODEL_GENERATION_METHODOLOGY.md   How each spatial null model is constructed (KNN, Delaunay, MST)
  MASKING_IMPLEMENTATION_SUMMARY.md      How sea-crossing edges are masked out of the spatial nulls
data/processed/                  Generated null graphs, percolation/scaling metrics, ARI/NMI matrices,
                                  HAC-on-null outputs, and summary CSVs (already generated; see below)
figures/
  hac_null_analysis/              Percolation curves/combined figures for each null model
  null_model_maps/                Network maps for the empirical graph and each null model
```

## Null-model families

- **empire_spatial** / **empire_temporal** — one null graph over the whole network
- **province_spatial** / **province_temporal** — null graphs built per province, then bridged

Spatial nulls connect nodes by **KNN** (k=4) or **Delaunay triangulation**,
optionally reinforced with a Euclidean **MST backbone** for connectivity;
edge travel times are assigned by `uniform_range_times` (sampled from the
empirical time_s range, not rank- or distance-preserving). Cross-province
edges that are geometrically plausible but empirically impossible (e.g. a
Channel or Gibraltar "shortcut") are masked out using the empirically
observed province-adjacency pairs.

## Running

All commands below assume the repo root as the working directory and that
`data_cleaning/cleaned_data/G_clean_weighted_with_ids.pkl` /
`G_clean_weighted_edges_with_province.csv` already exist (see `data_cleaning/`).

```bash
# Four null-model families, HAC + percolation + scaling metrics
python results/Null-models/src/run_null_model_pipeline.py --n-realizations 5

# Global spatial nulls (KNN / Delaunay, with MST backbone)
python results/Null-models/src/spatial_null_model_rewire.py --scope global --model knn_mst
python results/Null-models/src/spatial_null_model_rewire.py --scope global --model delaunay_mst

# Apply itinereX's HAC method to a generated null graph
python results/Null-models/src/apply_hac_to_nulls_v2.py --model knn_mst

# Render comparison maps
python results/Null-models/src/plot_null_model_maps.py
```

## Key outputs already in `data/processed/`

- `global_spatial_null_{knn,delaunay}_{,mst_}uniformrange.pkl` — the null graphs themselves
- `spatial_null_model_summary*.csv` — per-model network statistics vs. the empirical network
- `hac_null_analysis/null_{knn,delaunay}{,_mst}/` — HAC percolation curves/clusters run on each null
- `null_pipeline/matrices/cluster_{ari,nmi}_*.csv` — cluster alignment between empirical and null runs
- `null_pipeline/scaling/` — decay-model fits comparing empirical vs. null travel-time scaling
