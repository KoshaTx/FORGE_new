# FORGE: 22-family ICLR working draft

This directory now contains the **22-family result scaffold**. New-model numerical results are
explicitly marked **pending**, including the abstract and conclusion. This is a working draft,
not a submission-ready paper or a claim that final evaluations have completed.

- **Read:** `FORGE_ICLR2027_paper.pdf`.
- **Edit:** `FORGE_ICLR2027_paper.tex`, `sections/results_22.tex` and `sections/appendix_22.tex`.
- **Track evidence:** `RESULTS_CHECKLIST.md` and `RESULTS_LEDGER.json` map 15 result blocks to
  endpoints, populations, uncertainty, required artifacts and manuscript labels.
- **Overleaf:** upload `FORGE_ICLR2027_22Families_Overleaf.zip`; select `main.tex` and pdfLaTeX.
- **Validation:** `iclr22_validation.json` records the local builds, input/output hashes and checks.

## What the placeholders cover

Ten all-family matrices reserve results for exact L1; cohort/source/split accounting; validity,
connectedness and failures; individual training seeds; structural checks; separate head/ring/support outcomes; distributional realism;
diversity/novelty; generalization; and L2/L3 route dispositions. Additional tables cover training,
matched controls and the original-three-family bridge, decoder/repair ablations, architecture,
theory/implementation correspondence and conditional guidance. Three figure slots reserve the
all-family evidence overview, training/cost curves, and the seeded structure/failure atlas.

Every placeholder is missing evidence, not zero and not an expected positive outcome. The planned
minimum replication is three independent training seeds; actual seed IDs, requests and budgets
must be frozen before final evaluation. All requested outputs count. Separate seed variation,
sampling variation, and dependence introduced by batch-level selection.

Structural alerts, qualified design-condition agreement, TRAIN-feature support, realism,
diversity/novelty and route closure remain separate outcomes. New-model results cannot be filled
from old three-family tables or current TRAIN-derived development diagnostics. Priors used to
repair a molecule cannot independently validate that molecule's realism. Negative and null results
have explicit slots. Guidance remains conditional on the existing authorization and readiness gates.

## Where to find the quality and realism metrics

The main Results section now includes two explicit tables: **Lipid structural-quality metrics**
and **Lipid realism and distribution metrics**, each with raw/final result placeholders. They name
11 structural/support measures and nine distribution measures, and distinguish current TRAIN
reference diagnostics from the independent study still needed. The appendix gives exact definitions,
a 22-family head/ring/support matrix, and every one of the 24 implemented descriptors with reference,
raw, final and normalized Wasserstein comparison columns. `QUALITY_METRICS.json` pins their source
implementations and records the missing final results. Passing the limited design checks does not
establish that a molecule is free of all concerning motifs.

## Historical material and scope

The copied manuscript's computational tables, proofs, sources and structure figures remain in
explicitly labeled historical appendices. Their values were not recomputed or relabeled as new-model
measurements. `archive/three_family_source.tex` and `archive/three_family_source.pdf` preserve the
complete incoming manuscript. Its author-supplied experimental material remains locally available
but is excluded from the active computational 22-family narrative. No new experimental work is
proposed by this scaffold. The separate `paper/v1_iclr/` manuscript was not edited.

Historical training settings and the 194-atom/three-closure support are labeled as historical.
The current 22-family development support of 254 heavy atoms and 12 closure edges, with the
qualified Br extension, is distinguished from the final run contract still to be pinned.

The copied `NEW_MODEL_EVIDENCE_PLAN.md` retains the original dated audit and now links to the
current scaffold and later development evidence. Its old completion-stage measurements are not
a current status report. Inherited export projects and validation records are preserved under
`archive/`; use the newly named 22-family ZIP for the current draft.

## Build

```bash
cd paper/v1_iclr22
latexmk -pdf -interaction=nonstopmode -halt-on-error FORGE_ICLR2027_paper.tex
```

The source uses the existing local ICLR style and bibliography files. New placeholder tables are
editable LaTeX, and no fake numerical plot or generated molecular structure is included. Formatting
and page count are for an evidence-planning draft; final claims and space allocation will be revised
once the verified results are available. Build and inspect the updated PDF after filling cells.
