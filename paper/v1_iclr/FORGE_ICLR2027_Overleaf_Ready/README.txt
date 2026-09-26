FORGE ICLR 2027: optimized Overleaf project

Upload FORGE_ICLR2027_Overleaf_Ready.zip as a new Overleaf project.
Main document: main.tex
Compiler: pdfLaTeX

The manuscript, tables, algorithms, captions and references remain editable.
All manuscript prose is in main.tex. Figure assets are in figures/.
No shell escape, external programs or automatic figure generation are required.
The official ICLR template files and license notices are unchanged.
Author comments have been removed from manuscript and bibliography sources.

The overview, ROI chart and reaction schemes are precompiled vector PDFs.
Optional editable LaTeX sources for these three figures are in figures/.
To rebuild a figure locally from the project root, use, for example:
  pdflatex -jobname=figures/reported_roi_plot figures/reported_roi_plot.tex
Then compile main.tex. If changing figure dimensions, update its includegraphics
width and height in main.tex to the EXPORTBOX dimensions printed during its build.
ROI values remain in figures/reported_roi_values.csv.

The mouse image is embedded in a PDF at its original 3024 x 4032 resolution.
Every RGB pixel was verified against the supplied image; no resampling or
lossy compression was applied. Other supplied structures are unchanged.

Verified locally with pdfLaTeX, TeX Live 2026: nine main pages, 39 pages total.
Earlier optimization benchmark, before the latest prose update:
median clean-build time over three runs: 4.75 seconds, versus 11.07 seconds
for the inline-figure version on the same machine. Overleaf build times depend
on its servers and account limits; this export was not tested on those servers.
The historical implementation record is mathematical_review.json.
