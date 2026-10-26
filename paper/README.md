# paper/

Draft manuscript built on the [[HAC]] analysis in `results/HAC/`, targeting
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
figures/     fig1 = results/HAC/paper_replication/maps/hac_percolation_combined.pdf
             fig2 = results/HAC/maps/hac_percolation_combined.pdf, rasterized to
                    PNG at 200dpi (the source PDF is ~12MB; the raster copy
                    here is the figure actually used in the manuscript)
```

## Status

- **Introduction**, **Discussion**, **Authors' contributions**, and
  **Funding** are left as placeholders (`\todo` boxes) — not yet written.
- **Methods** (Data, HAC/percolation) and **Results** (validation network
  first, primary itinereX network second) are filled in, with numbers
  computed directly from `results/HAC/data/hac_clusters.csv` and
  `results/HAC/paper_replication/` logs/data.
- The gap between the validation network's alignment (ARI ≈ 0.70) and the
  primary network's (ARI ≈ 0.43) is flagged as a discussion point worth
  developing against the null models in `results/Null-models/`.

## Building

```bash
cd paper
pdflatex main.tex   # main.bbl is pre-built, so one pass is enough to view
# To rebuild main.bbl after editing refs.bib or citations:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
