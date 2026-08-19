FORGE - ICLR 2027 submission source

Main file:    FORGE_ICLR2027_paper.tex
Compiler:     pdfLaTeX
Bibliography: BibTeX

Compile twice. A prebuilt .bbl is included, so bibtex does not need to run.

Style files (iclr2027_conference.sty/.bst, fancyhdr, natbib) sit at the top
level rather than in a subfolder, since Overleaf doesnt search subfolders
for them. Leave them where they are.

These three are written by `forge paper render` and get overwritten on regeneration:
  generated_supplement.tex
  generated_product_structures.tex
  generated_appendix_tables.tex
Edit the surrounding prose in the main .tex instead.

Figures here are the paper's contracted copies. `forge paper verify` checks their
exact hashes; `forge paper bundle` includes only figures referenced by the LaTeX.

Build in an isolated directory with:
  forge paper build

Create and compile-check the deterministic Overleaf bundle with:
  forge paper bundle

Red TODO/PENDING markers are placeholders for the final refit and the
prospective campaign. They are meant to be visible.

Uncomment \iclrfinalcopy for the camera-ready.
