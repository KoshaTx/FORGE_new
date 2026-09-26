# FORGE, ICLR 2027

`FORGE_ICLR2027_paper.tex` is the canonical manuscript for this directory. It uses the official
ICLR 2027 conference style in anonymous review mode, including the conference bibliography style.
Historical computational measurements are preserved. Mathematical claims have been corrected and
scoped to their assumptions; the author-supplied exploratory in vivo experiment is now included.
`NEW_MODEL_EVIDENCE_PLAN.md` tracks the separate work needed for new-model claims.

## Main-text organization

The compiled main paper occupies nine pages. It retains:

- Ionizable-lipid architecture and the three reaction families, followed by molecular identity,
  reaction programs, exact assembly, component novelty, and L1/L2/L3 evidence.
- The four stages of FORGE, moved from the appendix overview to the start of the main method section.
- Program conditioning, role/core supervision, source-balanced training and the central
  ideal-generation theorem with its assumptions and limits.
- Equal-attempt evaluation budgets and the distinction between sampling and training-seed variation.
- Decoder and architecture ablations, including their reaction-dependent and negative results.
- Structural-realism diagnostics and their limits.

Detailed algorithms, proofs, full ablation tables and seed-level reporting remain in the appendix.
The reaction-family comparison table, program-embedding equation and generated-product examples
are also in the appendix. The main paper retains both central assembly comparison tables and the complete
experimental figure. Route-closure outcome reporting and its two supplementary tables are
omitted from this revision; source result artifacts and conditional route-assessment definitions
remain available. Repeated descriptions were shortened without changing numerical inputs,
formal statements or figure sizes. `iclr_provenance.json` records the moves and preservation checks.
No new experiments were run for this revision. Four author-supplied imaging values were added as
a descriptive experimental result; their provenance and unresolved protocol details are recorded
in `experimental/in_vivo/evidence_record.json`.

The main method follows the workflow: overview, program conditioning, graph representation and flow,
Transformer and training, sampling and verification, then recursive route assessment.

## Appendix organization

The supplement has four appendices:

| Appendix | Purpose |
|---|---|
| A. Method and implementation details | Program and graph encoding, Transformer and objectives, numerical sampling, verification, route assessment, and the inference algorithm. |
| B. Evaluation protocol and additional results | Protocol and metrics, assembly controls, ablations, novelty and realism, route closure, and guidance diagnostics. Seed-level tables accompany their comparisons. |
| C. Theoretical analysis and proofs | Formal graph support, catalogue reachability, discrete flow, sampling-budget results, and route guarantees. |
| D. Reaction schemes and generated examples | Reaction-family comparison, reaction schemes, generated-product and exact-precursor examples, and the generated-structure atlas. The chemical introduction remains in the main preliminaries. |

The former standalone reporting-table appendix is integrated into the results. Catalogue definitions
sit beside their proofs, and the repeated graph-space definition is consolidated into the theory setup.
Historical numerical results and reference labels are retained. The mathematical revision repairs
the formulation, metric definitions and proof scope. See `REVISION_NOTES.md` for substantive
corrections and `mathematical_review.json` for historical implementation correspondence.
The main experimental figure includes the supplied mouse image and three lipid structures.

Two-dimensional molecular drawings use thin gray bonds and dark atom labels. The generated
examples and atlas are vector PDFs rebuilt from the frozen structure receipts with
`figures/chemical_drawings/render.py`; matching SVGs and input/output hashes accompany them.
The supplied experimental drawings are retained. The atlas pairs each generated graph with
its exact L1 building blocks. Precursor-origin colors remain in the algorithm schematic.

## Build

```bash
cd paper/v1_iclr
latexmk -pdf -interaction=nonstopmode -halt-on-error FORGE_ICLR2027_paper.tex
```

The directory is self-contained: figures, generated tables, and both bibliography databases are
local. `iclr_provenance.json` records their original repository paths and SHA-256 hashes. Original input hashes remain recorded alongside any local editorial revisions. Numerical tables
remain unchanged; the schematic notation and bibliography have documented local edits. Refresh numerical inputs from verified
source artifacts; do not hand-edit reported values.

## ICLR template

The unmodified style files and reference template come from the
[official ICLR 2027 bundle](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
The bundle and extracted files are hash-recorded in `iclr_provenance.json`.

The manuscript loads `iclr2027_conference` and `times`; `iclrfinalcopy` is disabled.
The [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) allow nine pages of
main text at initial submission, with references and appendices excluded. The AI-use, ethics and
reproducibility statements are also excluded. In the current build, the Discussion ends on page 8 and the experimental figure appears on page 9; disclosure
statements and references begin on page 10.
The official margins and font sizes are unchanged.

The AI-use statement records the author-declared research, software, analysis and writing assistance.
The ethics statement distinguishes computational evidence from the exploratory mouse measurements;
animal procedures were approved by Children's Hospital of Philadelphia, as confirmed by the author.
The protocol number is omitted at the author's request. The reproducibility statement
links to the appendix account of historical source provenance and archive-access limitations.

## Overleaf

Use `FORGE_ICLR2027_Overleaf_Ready.zip` for upload. Its unpacked project is in
`FORGE_ICLR2027_Overleaf_Ready/`. Select `main.tex` and pdfLaTeX in Overleaf.
The overview, ROI chart and reaction schemes are precompiled vector PDFs, with optional
editable figure sources in `figures/`. The mouse image is embedded in a PDF with every
original RGB pixel preserved. Manuscript prose, equations, tables, captions and references
remain editable. The build requires no shell escape or external figure-generation tools.

Before the latest prose revision, three local clean builds had a median duration of 4.75 seconds, compared with 11.07 seconds
for the inline-figure export. A fresh extraction of the ZIP compiled successfully to nine main
pages and 39 pages overall. The export was tested locally with TeX Live 2026, not on Overleaf's
servers. `overleaf_ready_validation.json` records hashes, timings and preservation checks.

The clean export is in `FORGE_ICLR2027_Overleaf_Clean/`, with one `main.tex`, one
`references.bib`, the unchanged official template files, and the historical implementation record
at the root. Required images and ROI data are in `figures/`. Manuscript and bibliography comments
are removed; official template comments and license notices are retained. Compile `main.tex`
with pdfLaTeX. The validated output has nine main pages and 39 pages overall, matching the
canonical PDF in extracted text. `clean_overleaf_validation.json` records the file hashes
and comparison.

`FORGE_ICLR2027_Overleaf/` and `overleaf_export/` contain matching full source exports.
The older full-source and clean ZIP archives retain their previous contents; use the new
`FORGE_ICLR2027_Overleaf_Ready.zip` for the current manuscript and compilation optimizations.
Historical venue names in citations and source-provenance paths are retained accurately.
The experimental extension remains a working draft until the protocol and replication fields
listed in `REVISION_NOTES.md` are completed.

The latest author-supplied prose is synchronized across the canonical manuscript and all four
export sources. The overview contains minor grammar and duplication corrections. The Ready ZIP
was rebuilt and tested from a fresh extraction; the main paper remains nine pages.
