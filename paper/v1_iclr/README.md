# FORGE, ICLR 2027

`FORGE_ICLR2027_paper.tex` is the canonical manuscript for this directory. It uses the official
ICLR 2027 conference style in anonymous review mode, including the conference bibliography style.
The scientific claims and historical numerical inputs are preserved from the copied manuscript.
`NEW_MODEL_EVIDENCE_PLAN.md` tracks the separate work needed for new-model claims.

## Main-text organization

The main text has been expanded before a later page-budget editing pass. Material previously
confined to the appendix now explains:

- Ionizable-lipid architecture and the three reaction families, followed by molecular identity,
  reaction programs, exact assembly, component novelty, and L1/L2/L3 evidence.
- The four stages of FORGE, moved from the appendix overview to the start of the main method section.
- Layerwise program conditioning, role/core supervision, and source-balanced training.
- Equal-attempt evaluation budgets and the distinction between sampling and training-seed variation.
- Decoder and architecture ablations, including their reaction-dependent and negative results.
- Structural-realism diagnostics and their limits.

Detailed algorithms, proofs, full ablation tables, and seed-level reporting remain in the appendix.
The main-text additions are traced to their existing appendix sources in `iclr_provenance.json`.
No new experiments or numerical estimates were introduced by this revision.

The main method follows the workflow: overview, program conditioning, graph representation and flow,
Transformer and training, sampling and verification, then recursive route assessment.

## Appendix organization

The supplement has four appendices:

| Appendix | Purpose |
|---|---|
| A. Method and implementation details | Program and graph encoding, Transformer and objectives, numerical sampling, verification, route assessment, and the inference algorithm. |
| B. Evaluation protocol and additional results | Protocol and metrics, assembly controls, ablations, novelty and realism, route closure, and guidance diagnostics. Seed-level tables accompany their comparisons. |
| C. Theoretical analysis and proofs | Formal graph support, catalogue reachability, discrete flow, sampling-budget results, and route guarantees. |
| D. Reaction schemes and generated examples | Reaction schemes and the generated-structure atlas; the chemical introduction is in the main preliminaries. |

The former standalone reporting-table appendix is integrated into the results. Catalogue definitions
sit beside their proofs, and the repeated graph-space definition is consolidated into the theory setup.
Numerical results, citations, algorithms and proof content are retained; this is an organizational pass,
not a page-length cut.

## Build

```bash
cd paper/v1_iclr
latexmk -pdf -interaction=nonstopmode -halt-on-error FORGE_ICLR2027_paper.tex
```

The directory is self-contained: figures, generated tables, and both bibliography databases are
local. `iclr_provenance.json` records their original repository paths and SHA-256 hashes. Files
renamed from `gem_` to `iclr_` retain identical contents. Refresh numerical inputs from verified
source artifacts; do not hand-edit reported values.

## ICLR template

The unmodified style files and reference template come from the
[official ICLR 2027 bundle](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
The bundle and extracted files are hash-recorded in `iclr_provenance.json`.

The manuscript loads `iclr2027_conference` and `times`; `iclrfinalcopy` is disabled.
The [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) allow nine pages of
main text at initial submission, with references and appendices excluded. The AI-use and optional
reproducibility statements are also excluded. Do not change the official margins or font sizes.

The AI-use statement records the formatting and editorial assistance used for this conversion. Authors must
complete the broader AI-use disclosure and review description before treating this as a submission.

## Overleaf

`FORGE_ICLR2027_Overleaf.zip` contains a standalone project with `main.tex` as its entry point.
`FORGE_ICLR2027_Overleaf/` and `overleaf_export/` contain the same exported source files.
These exports are derived from the canonical manuscript; refresh them together after source edits.
Historical venue names in cited publications or source-provenance paths are retained accurately.
