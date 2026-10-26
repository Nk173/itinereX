# paper_replication/

Self-contained re-run of the itinereX HAC / travel-time percolation analysis on
the **exact network analysed in the manuscript**

> Pažout A., Brughmans T., de Soto P., Mazzamurro M., Coto-Sarmiento M., Filet C.,
> Herrera Malatesta E., Nielsen M. L., Ølgaard G. E., Vahlstrup P. B.
> *Network Science reveals the structure and lasting impact of the Roman road system.*
> (Sci. Data, resubmission 2025)

Nothing here touches the top-level `src/`, `data/`, `notebooks/` or `maps/`. This
folder is additive.

## Why

The main repo builds its own cleaned network from the raw Itiner-e linework with a
flat 2 km/h travel-time model. To make the HAC / percolation results directly
comparable to (and citable alongside) the manuscript, this folder rebuilds the
graph the way the manuscript does and re-runs the same method on top.

| | main repo (`../src`) | here (`paper_replication`) |
|---|---|---|
| Network | own cleaning of raw Itiner-e (8,460 n / 12,849 e / 12 comp) | manuscript's Itiner-e export, endpoint-noded (10,506 n / 14,770 e / 13 comp) |
| Travel time | flat 2 km/h on Mercator planar length | **Tobler's hiking function** on `Avg_Slope`, `LENGTH_GEO` (geodesic) |
| Europe↔Asia | none | Bosporus + Dardanelles ferry links (already in the export) |
| Provinces | `provinces/provinces.shp` | AWMC "Roman Empire AD 200" (+ regiones), from the cloned repo |
| Method | HAC per component → threshold sweep → percolation curve / ARI | *identical* |

## Contents

```
roman-road-networks/      git clone of github.com/MatteoMazzamurro/roman-road-networks (untouched)
src/build_network.py      raw shapefile -> network_paper.pkl / .gpkg
src/hac_percolation.py    HAC + threshold sweep + percolation curve + province ARI/NMI + bootstrap CI
data/                     build + analysis outputs (git-ignored large files)
maps/                     figures
```

## Environment

Runs under the `MIR` conda env (geopandas 1.1.3, networkx 3.4.2, scipy, scikit-learn):

```
/Users/nk821/opt/anaconda3/envs/MIR/bin/python paper_replication/src/build_network.py
/Users/nk821/opt/anaconda3/envs/MIR/bin/python paper_replication/src/hac_percolation.py
```

`build_network.py` runs in seconds. `hac_percolation.py` takes ~10–15 min (the
all-pairs travel-time distance matrix for the 9,132-node continental component
plus the bootstrap over merge-height thresholds).

## Travel-time model (matches `network-analysis.Rmd`)

```
slope  = Avg_Slope / 100                       # segment mean slope, fraction
pace   = 0.6 * exp(3.5 * |slope + 0.05|)        # seconds per metre (Tobler)
time_s = LENGTH_GEO * pace                      # LENGTH_GEO = geodesic length, m
```

## Network reconciliation

| | manuscript | this build |
|---|--:|--:|
| segments (edges) | 14,901 | 14,770 |
| nodes | 10,516 | 10,506 |
| connected components | 13 | 13 |
| giant component | Europe–Asia–Africa | 9,132 nodes, lon [−9.4, 41.6], lat [22.7, 52.2] |

Deltas: 2 zero-length segments dropped; ~129 parallel segments between the same
node pair collapsed to the fastest one (HAC uses shortest-path distance, so this
is lossless for the analysis). Endpoint snapping at 1 cm (EPSG:3395).

## Headline results (average linkage, Tobler travel time)

- **Critical threshold** (steepest log–log collapse in cluster count) ≈ **14 h**.
- **Scaling law** `S_max ∝ t^0.96` (95% CI [0.92, 1.00], R² 0.98) over the 6 h–4000 h
  regime, fitted with equal weight per log-decade — the largest travel-time cluster
  grows ~linearly with the tolerated trip length. (An earlier `t^1.15` was an OLS
  artefact of merge-height density at low t; `plot_combined.py`'s log-grid gives 0.93.)
- **Emergent regions vs. Roman provinces** (Adjusted Rand Index / NMI vs. AWMC AD 200,
  10,430 nodes with a known province):

  | snapshot | clusters | ARI | NMI |
  |---|--:|--:|--:|
  | 24 h | 514 | 0.20 | 0.73 |
  | 2 d  | 154 | 0.52 | 0.80 |
  | 7 d  | 22  | 0.49 | 0.78 |

  Exhaustive search over every merge height peaks sharply at **≈ 77 h ≈ 3.2 days**,
  **ARI 0.70** (bootstrap 95% CI on ARI [0.68, 0.71]; the best-threshold estimate is
  a hard peak — every bootstrap resample selects the same 77 h merge height).
- Interpretation: at a ~3-day travel-time budget the road network's spontaneous
  regions align strongly with the imperial provincial map — a quantitative handle
  on the manuscript's argument that road-system planning operated at the provincial
  level, and the starting point for the resilience / targeted-attack analysis the
  manuscript flags but does not perform.

## Figures

- `maps/hac_percolation_curve.png` – percolation curve + power-law fit
- `maps/hac_percolation_combined.pdf` / `.png` – curve (a) + snapshot maps.
  Regenerate the snapshot set without re-running the whole analysis:
  `python src/plot_combined.py 24 48 77.14 120`   (hours; default = 24 48 77 120)
- `maps/hac_dendrogram_comp{0..7}.png` – per-component dendrograms
- `maps/province_alignment_curve.png` – ARI / NMI / homogeneity / completeness of the
  clusters vs. the AWMC AD 200 provinces **at every travel-time threshold**
  (`src/province_alignment_curve.py`, → `data/province_alignment.csv`).
  ARI rises from ~0, peaks at **0.69 at ~78 h (3.3 d), ~66 clusters**, then falls as
  clusters outgrow province scale. Homogeneity (clusters inside one province) and
  completeness (provinces kept whole) cross near the peak — that crossing is the
  travel-time scale at which cluster granularity ≈ province granularity. The fine
  (regiones separate) and coarse (Italia merged) ARI curves are near-identical.

See `hac_run.log` for the full console output.
