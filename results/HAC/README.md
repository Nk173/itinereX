# results/HAC/

Hierarchical agglomerative clustering (HAC) on intra-component travel-time
distances, swept across all distance thresholds to identify percolation
structure in the Roman road network — and how well the emergent clusters
align with the Roman provinces.

Two independent runs live here, on two different networks, **under two
different HAC settings** (see below):

```
network0-average/   HAC (scipy linkage, method="average") on the exact network
                     and Tobler-hiking-function travel time used in Pažout et
                     al.'s manuscript. Self-contained (own src/data/maps/README);
                     see its own README.md for details.
network1-single/     HAC (scipy linkage, method="single") on itinereX's own
                     independently cleaned network, constant 2 km/h travel time.
  src/                  hac_percolation.py, hac_threshold_search.py
  data/                 hac_clusters.csv, linkage cache
  maps/                 Dendrograms, percolation curve/combined figure, snapshot maps
```

## Are the settings the same?

**No.** `network0-average` uses scipy's `linkage(..., method="average")`;
`network1-single` uses `linkage(..., method="single")`. This is not
incidental: under single linkage, cutting the dendrogram at travel-time
threshold $t$ reproduces *exactly* the network's own connected components
with every edge of weight $>t$ removed — i.e. HAC under single linkage is
mathematically equivalent to a percolation sweep over the network's edges,
computed once per component instead of independently at every threshold.
Average linkage (`network0-average`) does not have this property; it
clusters by mean pairwise travel time instead. The folder names carry the
setting for exactly this reason — don't compare ARI/NMI numbers between the
two without accounting for it.

(`network1-single/src/hac_percolation.py` previously had its active
`LINKAGE_METHOD` reverted to `"average"` in source while the committed
`hac_clusters.csv` / cache were actually generated with `"single"` — a
leftover from manual testing. This has been fixed so the checked-in script
matches the checked-in data.)

## Running

From the repo root:

```
python results/HAC/network1-single/src/hac_percolation.py
python results/HAC/network1-single/src/hac_threshold_search.py
```

These read the cleaned network from `data_cleaning/cleaned_data/` and
province boundaries from `reference/provinces/`, and write into
`results/HAC/network1-single/data/` and `results/HAC/network1-single/maps/`.
For `network0-average`, see its own README.md.

## Key results

- **network0-average** (validation, manuscript network + Tobler travel time):
  ARI ≈ 0.70 (bootstrap 95% CI [0.68, 0.71]) at ~77 hours (~3.2 days), a
  single sharp peak, using the 10,430+ nodes with a known province.
- **network1-single** (itinereX's own network, constant-speed travel time):
  ARI ≈ 0.43 (95% CI [0.43, 0.45]) at ~25–32 hours, with two comparable
  local maxima rather than one sharp peak, using the 8,411 nodes with a
  known province.

See `paper/main.tex` for the write-up — the manuscript network is used as a
validation/calibration check that the method recovers a known administrative
partition; the primary point of applying HAC to itinereX's own network is
not to reproduce that number but to find emergent spatial units of interest
in a network nobody has clustered before.
