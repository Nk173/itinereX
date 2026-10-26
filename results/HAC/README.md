# results/HAC/

Hierarchical agglomerative clustering (HAC) on intra-component travel-time
distances, swept across all distance thresholds to identify percolation
structure in the Roman road network — and how well the emergent clusters
align with the Roman provinces.

## Contents

```
src/
  hac_percolation.py       HAC per connected component + threshold sweep + percolation
                            curve + dendrograms + snapshot maps + province ARI/NMI
  hac_threshold_search.py  Exhaustive search for the merge height that best aligns HAC
                            clusters with province membership (+ bootstrap CI on ARI)
data/                      hac_clusters.csv, linkage caches
maps/                      Dendrograms, percolation curve/combined figure, snapshot maps
paper_replication/         Self-contained re-run of the same method on the exact network
                            analysed in the manuscript (Pažout et al., Sci. Data). See its
                            own README.md for details — it does not touch anything outside
                            itself.
```

## Running

From the repo root:

```
python results/HAC/src/hac_percolation.py
python results/HAC/src/hac_threshold_search.py
```

These read the cleaned network from `data_cleaning/cleaned_data/` and province
boundaries from `reference/provinces/`, and write into `results/HAC/data/` and
`results/HAC/maps/`.

## Key result

The HAC clusters align with the AWMC "Roman Empire AD 200" provinces best at a
travel-time threshold of ~77 hours (~3.2 days): **ARI ≈ 0.70** (bootstrap 95% CI
[0.68, 0.71]), using the 10,430+ nodes with a known province. See
`paper_replication/README.md` for the same analysis on the manuscript's own
network/travel-time model.
