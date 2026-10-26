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
figures/     fig1 = network0-average/maps/hac_percolation_combined.pdf
             fig2 = network0-average/maps/province_alignment_curve.png
             fig3 = network0-average/maps/province_match_78.2h.png
             (all three from the calibration run; there is deliberately no
             figure for the primary network1-single result -- see Status)
```

All three figures were regenerated from `results/HAC/network0-average/src/`
with larger matplotlib font sizes (`rcParams` + explicit `fontsize=` bumped
throughout `plot_combined.py`, `province_alignment_curve.py`,
`province_match_breakdown.py`) -- the originals had unreadably small axis
labels/annotations once shrunk to page width. `network1-single/src/hac_percolation.py`
got the same font-size treatment in source but wasn't rerun (it isn't
currently used as a figure in the paper, and rerunning it needs the full
`data_cleaning/` pipeline rather than a cheap cache replay).

## Status

- **Two questions, stated explicitly in the Introduction** (§1): (1) can
  percolation-equivalent HAC identify spatial units that emerge from the
  itinereX road network itself, and (2) are those units significant, or
  just artefacts of the network's own construction — answerable only by
  comparing against the null models in `results/Null-models/`. Everything
  else in the Introduction is still a placeholder; these two questions are
  meant to survive into the full version.
- **Methods** (Data, HAC/percolation) and **Results** are filled in, with
  numbers computed directly from `results/HAC/network1-single/data/hac_clusters.csv`
  and `results/HAC/network0-average/` logs/data.
- §2.1 (Data) is a single paragraph, written in the terse, citation-anchored
  style of Pažout et al. (2026): no repo paths in the prose (those move to
  §2.2's naming sentence and to Data accessibility), more detail on the
  province-boundary source (AWMC AD 200, approximate, clipped to modern
  coastlines, fine 57-unit vs. coarse 47-unit partition), and the network
  described as built from the Itiner-e linework and cleaned following the
  same general procedure as Pažout et al.
- Two networks, two HAC settings: `network0-average` (calibration, scipy
  `linkage(..., method="average")`) and `network1-single` (primary, `method="single"`,
  chosen because it is exactly equivalent to a percolation sweep over the
  network's own edges). This is spelled out in Methods §2.2 so the two
  runs' ARI/NMI numbers are never read as directly comparable.
- §3.1 (network0-average) is framed as a **calibration** check that the
  method can recover a known administrative partition — not as the paper's
  main result. It now includes all three calibration figures: the
  percolation curve/snapshots (Fig. 1), the full ARI-vs-threshold curve
  (Fig. 2), and the per-province F1 breakdown at the peak threshold (Fig. 3).
- §3.2 (network1-single) is framed as the **primary, novel** contribution:
  applying the method to itinereX's own network for the first time to find
  candidate spatial units, not as an attempt to replicate §3.1. Its
  province-alignment benchmark (ARI ≈ 0.43) is reported in prose only —
  **no dedicated figure or table** — since a single alignment statistic
  doesn't do justice to the clusters and needs a closer look first.
- **Discussion / Authors' contributions / Funding** are still placeholders.
- Next step (not yet in the manuscript): compare network1-single's
  clusters against the spatial/temporal null models in
  `results/Null-models/` to separate a genuine topological signal from an
  artefact of the constant-speed travel-time model — i.e. answer question
  (2) above.

## Building

```bash
cd paper
pdflatex main.tex   # main.bbl is pre-built, so one pass is enough to view
# To rebuild main.bbl after editing refs.bib or citations:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
