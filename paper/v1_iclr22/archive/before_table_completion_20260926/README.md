# FORGE: manuscript across 22 assembly families

`FORGE_ICLR2027_paper.pdf` presents the paper in final manuscript form, retaining the original
`v1_iclr` title, motivation, mathematical exposition and proofs. Introduction prose is preserved
except for the second contribution bullet, which now distinguishes completed development
diagnostics from unrun trained baselines and generalization. The scientific
scope and evaluations cover all 22 assembly families. Missing values and conclusions appear as
short bracketed placeholders or table dashes; these are missing results, never zero.

- Main source: `FORGE_ICLR2027_paper.tex`.
- Detailed methods: `sections/implementation_22.tex`.
- Results narrative: `sections/results_22.tex`.
- Evaluation and training/route tables: `sections/appendix_22.tex`.
- Structural quality, realism and diversity: `sections/quality_current.tex`.
- Reaction schemes: Figure 5 and `sections/reaction_figures_22.tex`.
- Portable Overleaf source: `FORGE_ICLR2027_22Families_Overleaf.zip`, compile `main.tex`.
- Evidence tracking: `RESULTS_LEDGER.json`, `RESULTS_CHECKLIST.md` and `QUALITY_METRICS.json`.

The paper includes explicit structural-quality and realism tables, a full 24-descriptor panel,
ten all-family matrices, and controls, ablations, generalization, L2/L3, computational cost and
a structure-atlas placeholder. Quality, TRAIN support, realism, diversity and route closure remain
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

Current measured development evidence is populated with its original scope: one-checkpoint
TRAIN-derived selected requests, separate 352-request training/decoder diagnostics, CAL
representation and synthetic-reference sensitivity studies, and computational routes to vendor-listed
leaves. Independent training seeds, final held-out generation, independently qualified empirical
realism comparisons and unrun controls remain explicit gaps. Development estimates are not
experimental synthesis, delivery performance or final independent validation.

The selected cohort contains 1,408 requests: 1,387 exact L1, 1,324 limited exact-plus-design,
616 computationally makeable, 1,189 L2-ready, 119 direct-only and 27 strict dossiers. A vendor
listing is documentary listing evidence, not verified stock or a synthesis-success probability.
The separate 352-request controls and 1,821-reference synthetic CAL panel are not pooled with
this cohort. See `current_summary.json`, `current_quality_evidence.json`,
`current_appendix_evidence.json` and their pinned `current_evidence_manifest.json`.

The pre-update manuscript is preserved under `archive/before_measured_results_20260925/`;
older three-family numerical evidence remains archived and unchanged. The separate `paper/v1_iclr/`
manuscript is unchanged. `RESULTS_LEDGER.json` retains null final-study artifacts while recording
computed development findings and outstanding evidence requirements.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error FORGE_ICLR2027_paper.tex`.
Build the portable package with `.venv/bin/python paper/v1_iclr22/package_overleaf.py` from the
repository root. Extract the ZIP into a fresh directory, compile `main.tex`, and run
`verify_placeholders.py --export-pdf <extracted directory>/main.pdf` with the repository Python.
Before compiling, `verify_placeholders.py --sources-only` checks the documentary evidence and source invariants without claiming PDF validation. Packaging runs this check automatically.
Validation and visual-review receipts record exact source and output hashes.
The validator checks the Methods implementation pins, reaction-record and chemistry-review
receipts, complete family/event accounting, every schematic audit, and equality of the source
and portable-export SVG/PDF files. Rendered-page visual review is a separate check. Reaction
figures do not populate or replace any numerical result placeholder.

The unchanged chemistry illustrations retain their original assessment scope. Four source events
have no registered net-byproduct balance contract and remain explicitly unassessed for that check.
Exact replay and the existing schematic checks are preserved; this document update performs no new
chemistry qualification.

The portable package includes the three `build_*` evidence-rendering scripts as provenance.
Rerunning them requires the original repository result files; compiling the included tables and
figures requires no external results or model execution.
