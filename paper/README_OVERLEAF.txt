FORGE, ICLR 2027 submission source
==================================

Main file:  FORGE_ICLR2027_paper.tex
Compiler:   pdfLaTeX
Bibliography: BibTeX (not biber)

In Overleaf: Menu -> Compiler -> pdfLaTeX, and Main document ->
FORGE_ICLR2027_paper.tex. Compile TWICE. The second pass is what resolves
cross-references and settles longtable column widths.

You do NOT need to run BibTeX. A prebuilt FORGE_ICLR2027_paper.bbl ships with
this archive, so citations resolve from the first pdflatex pass. This is
deliberate: on Overleaf's free plan the compile timed out before BibTeX could
run, which left every citation undefined.

The figures are downsampled to 300 DPI and alpha-flattened for the same reason.
The originals are about 700 DPI with an unused alpha channel, which made a
single pass cost roughly 6.9 s instead of 0.8 s. If you need the full-resolution
figures, take them from paper/figures/ in the main repository.

Layout of this archive
----------------------
The official ICLR style files are at the archive ROOT rather than in a
subdirectory, because Overleaf does not reliably search subdirectories for
.sty and .bst files. Do not move them into a folder.

  iclr2027_conference.sty / .bst   official ICLR 2027 style
  fancyhdr.sty, natbib.sty         vendored dependencies
  math_commands.tex                vendored, not currently \input
  forge.bib                        bibliography, 77 verified entries
  figures/                         only the figures the document includes

GENERATED FILES -- do not hand-edit
-----------------------------------
Three .tex files are script-generated from frozen result artifacts. Editing
them by hand puts a number in the paper that no artifact supports, which is
exactly what the project's evidence ledger exists to prevent.

  generated_supplement.tex          Tables S1-S16
  generated_product_structures.tex  Table S17, the panel product structures
  generated_appendix_tables.tex     per-seed semantic tables, 38-descriptor panel

To change a value in any of them, fix the generator or its artifact in the
main repository and re-render. Prose edits to the surrounding appendix text are
fine and belong in FORGE_ICLR2027_paper.tex.

Placeholders
------------
Red [TODO: ...] and [PENDING ...] markers are deliberate. They mark values
awaiting the full-corpus refit or the prospective campaign, and they render in
red so an unfilled number cannot reach a PDF unnoticed. Do not silently fill
them with plausible values.

Camera-ready
------------
Uncomment \iclrfinalcopy near the top to de-anonymize.
