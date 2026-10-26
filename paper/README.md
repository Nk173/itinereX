# paper/

Draft manuscript built on the HAC analysis in `results/HAC/`, targeting
a Royal Society journal (Royal Society Open Science), following the
methodological precedent of Arcaute et al. (2016), "Cities and regions in
Britain through hierarchical percolation" (`reference/papers/rsos.150691.pdf`).

```
main.tex     Manuscript source (standard article class as a placeholder
             layout -- swap in the official rsos.cls template and preamble
             for submission; section structure/content do not need to change)
refs.bib     Bibliography
main.bbl     Pre-built bibliography (so the PDF can be rebuilt with a single
             pdflatex pass if bibtex/biber isn't available)
main.pdf     Compiled draft
figures/     fig1 = results/HAC/network0-average/maps/hac_percolation_combined.pdf
             (the calibration run's figure; there is deliberately no fig2 --
             see Status below)
```

## Status

- **Introduction**, **Discussion**, **Authors' contributions**, and
  **Funding** are left as placeholders (`\todo` boxes) — not yet written.
- **Methods** (Data, HAC/percolation) and **Results** are filled in, with
  numbers computed directly from `results/HAC/network1-single/data/hac_clusters.csv`
  and `results/HAC/network0-average/` logs/data.
- Two networks, two HAC settings: `network0-average` (calibration, scipy
  `linkage(..., method="average")`) and `network1-single` (primary, `method="single"`,
  chosen because it is exactly equivalent to a percolation sweep over the
  network's own edges). This is spelled out in Methods §2.2 so the two
  runs' ARI/NMI numbers are never read as directly comparable.
- §3.1 (network0-average) is framed as a **calibration** check that the
  method can recover a known administrative partition — not as the paper's
  main result.
- §3.2 (network1-single) is framed as the **primary, novel** contribution:
  applying the method to itinereX's own network for the first time to find
  candidate spatial units, not as an attempt to replicate §3.1. Its
  province-alignment benchmark (ARI ≈ 0.43) is reported in prose only —
  **no dedicated figure or table** — since a single alignment statistic
  doesn't do justice to the clusters and needs a closer look first.
- Next step (not yet in the manuscript): compare network1-single's
  clusters against the spatial/temporal null models in
  `results/Null-models/` to separate a genuine topological signal from an
  artefact of the constant-speed travel-time model.

## Building

```bash
cd paper
pdflatex main.tex   # main.bbl is pre-built, so one pass is enough to view
# To rebuild main.bbl after editing refs.bib or citations:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
