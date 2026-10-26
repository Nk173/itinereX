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
figures/     fig1 = calibration network's percolation curve + snapshots
             fig2 = calibration network's ARI-vs-threshold curve
             fig3 = calibration network's per-province F1 breakdown
             fig4 = nearest-neighbour spatial null model's percolation curve
             fig5 = Delaunay-triangulation spatial null model's percolation curve
```

Figures 1-3 were regenerated from `results/HAC/network0-average/src/` with
larger matplotlib font sizes (`rcParams` + explicit `fontsize=` bumped
throughout `plot_combined.py`, `province_alignment_curve.py`,
`province_match_breakdown.py`, plus a staggered label fix in
`plot_combined.py` so the "77.14h"/"5d" threshold markers stop colliding at
the larger size) -- the originals had unreadably small axis labels once
shrunk to page width. Figures 4-5 are copied as-is from the null-model
analysis in `results/Null-models/`. `network1-single/src/hac_percolation.py`
got the same font-size treatment in source but wasn't rerun (not currently
used as a figure in the paper; rerunning needs the full `data_cleaning/`
pipeline rather than a cheap cache replay).

## Status

- **Two questions, stated explicitly in the Introduction** (§1): (1) can
  percolation-equivalent HAC identify spatial units that emerge from the
  itinereX road network itself, and (2) are those units significant, or
  just artefacts of the network's own construction. Everything else in the
  Introduction is still a placeholder.
- **§2.1 (Data)** and **§2.2 (HAC/percolation)** are each a tight paragraph
  in the terse, citation-anchored style of Pažout et al. (2026), with no
  repo paths or `network0-average`/`network1-single` labels in the prose —
  those stay in code, figure captions, and Data accessibility.
- **§2.3 (Entropy of cluster sizes)**, new: introduces, in words only, a
  planned analysis following Molinero et al. (2017) — tracking the Shannon
  entropy of the cluster-size distribution across thresholds to locate
  percolation stages as entropy peaks. Not yet carried out on either
  network.
- **§2.4 (Comparison with spatial null models)**, new: describes the two
  spatial null families (nearest-neighbour, Delaunay, both with an MST
  connectivity backbone) and shows their percolation curves (Figs. 4-5) as
  a first look — both are smooth and gradual, unlike the real network's
  curve. The actual threshold-by-threshold comparison against the real
  network is explicitly still in progress.
- **§3.1**, renamed from "Calibration: recovering a known administrative
  partition (network0-average)" to **"Identifying spatial units using
  HAC"**. Now includes all three calibration figures (percolation
  curve/snapshots, ARI-vs-threshold curve, per-province F1 breakdown).
- **§3.2**, renamed from "Towards emergent spatial units in the itinereX
  network (network1-single)" — that title claimed a result this analysis
  doesn't support yet — to **"Applying HAC to the itinereX network"**. Its
  closing paragraph (previously long and winding) is now three sentences
  that point to §2.3/§2.4 instead of re-explaining the null-model
  rationale inline.
- All `\texttt{repo/path}` references and `network0-average`/
  `network1-single` labels removed from Results prose and figure captions
  (they remain in Data accessibility, where they belong).
- **Discussion / Authors' contributions / Funding** are still placeholders
  — not touched this round, so the Discussion `\todo` still mentions
  `network1-single` and a repo path; that's next.

## Building

```bash
cd paper
pdflatex main.tex   # main.bbl is pre-built, so one pass is enough to view
# To rebuild main.bbl after editing refs.bib or citations:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
