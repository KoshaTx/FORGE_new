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

Why this directory is flat, since it looks untidy and mostly cannot be fixed:
the main .tex, the three generated .tex files and the panel/dossier figures are
pinned by path and sha256 in configs/reproduction/iclr2027.json, so moving any
of them breaks `forge paper verify`; the style files have to sit at the root for
Overleaf. What can be tidied is build output. LaTeX writes .aux/.log/.fls/.out/
.blg/.fdb_latexmk beside the sources; they are gitignored, and `make clean` now
removes them. The .bbl and .pdf are tracked on purpose -- submission wants both.

latex/ is an unused alternate toolchain: pandoc templates and lua filters for an
ICLR and a Nature Biotech layout. Nothing in the repository references it and
pandoc is not a dependency, so it does not participate in `forge paper build`.
It kept byte-identical copies of the five root style files; those were removed
so there is one canonical copy of each. Its iclr2027_conference.bib is unique
and stays. Delete the directory if the pandoc route is abandoned for good.

Build in an isolated directory with:
  forge paper build

Create and compile-check the deterministic Overleaf bundle with:
  forge paper bundle

Red TODO/PENDING markers are placeholders for the final refit and the
prospective campaign. They are meant to be visible.

Uncomment \iclrfinalcopy for the camera-ready.
