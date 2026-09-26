# FORGE: manuscript across 22 assembly families

`FORGE_ICLR2027_paper.pdf` presents the paper in final manuscript form, retaining the original
`v1_iclr` title, motivation, Introduction prose, mathematical exposition and proofs. The scientific
scope and evaluations cover all 22 assembly families. Missing values and conclusions appear as
short bracketed placeholders or table dashes; these are missing results, never zero.

- Main source: `FORGE_ICLR2027_paper.tex`.
- Detailed methods: `sections/implementation_22.tex`.
- Results narrative: `sections/results_22.tex`.
- Evaluation and complete result tables: `sections/appendix_22.tex`.
- Reaction schemes: Figure 5 and `sections/reaction_figures_22.tex`.
- Portable Overleaf source: `FORGE_ICLR2027_22Families_Overleaf.zip`, compile `main.tex`.
- Evidence tracking: `RESULTS_LEDGER.json`, `RESULTS_CHECKLIST.md` and `QUALITY_METRICS.json`.

The paper includes explicit structural-quality and realism tables, a full 24-descriptor panel,
ten all-family matrices, and controls, ablations, generalization, L2/L3, computational cost and
structure-atlas results. Quality, TRAIN support, realism, diversity and route closure remain
separate outcomes. Scientific limitations stay in the paper; execution status and drafting
instructions stay in these supporting files.

Figure 5 presents the Ugi three-component, aza-Michael acrylate and reductive-amination
chemistries. Companion schemes cover the other 19 families, giving 22 unique families in total.
Their source records contain 22 representative TRAIN assemblies and 34 registered events;
repeated events are distinguished from the number of arrows displayed. These are reaction
illustrations with exact source replay, not measurements of newly generated model outputs.
Unchanged substituents use arrow-local R labels, with retained atoms, bonds, charges and hidden
attachments checked against the source graphs. The grouping of families into figures follows
`figures/all_family_reactions/render.json` and is not fixed to a particular number of pages.

No final 22-family experimental results have been admitted by this editorial revision. Existing
three-family numerical tables and single-checkpoint TRAIN diagnostics are not used to fill new
result cells. The incoming manuscript and previous editorial versions remain under `archive/`.
Its experimental material is preserved there; the active paper follows the computational scope.
The separate `paper/v1_iclr/` manuscript is unchanged.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error FORGE_ICLR2027_paper.tex`.
Build the portable package with `.venv/bin/python paper/v1_iclr22/package_overleaf.py` from the
repository root. Extract the ZIP into a fresh directory, compile `main.tex`, and run
`verify_placeholders.py --export-pdf <extracted directory>/main.pdf` with the repository Python.
Validation and visual-review receipts record exact source and output hashes.
The validator checks the Methods implementation pins, reaction-record and chemistry-review
receipts, complete family/event accounting, every schematic audit, and equality of the source
and portable-export SVG/PDF files. Rendered-page visual review is a separate check. Reaction
figures do not populate or replace any numerical result placeholder.
