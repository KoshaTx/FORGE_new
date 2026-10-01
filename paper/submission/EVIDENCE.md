# Paper → implementation and evidence

Numbering refers to the [submitted PDF](FORGE.pdf), not older GEM/ICLR drafts. Historical filenames
retain their original names: a renderer named `gem_table7` need not produce Table 7 in this PDF.
`manifest.json` pins the reviewer assets; `make review-check` checks those bytes. The map below
separates computed results, author-reported experiments and idealized mathematical statements.

## Method map

Paths below are relative to the repository root and describe the **current implementation**.
Appendix A.8 and `paper/v1_iclr/mathematical_review.json` qualify historical correspondence.

| Paper | Code | What to inspect |
|---|---|---|
| §§2.2–2.4; Appendix A.5 | `forge/assembly/`, `forge/model/reaction_program_evaluation.py` | Roles, constitutional identity, reverse decomposition, exact forward replay and ambiguity |
| §§3.1–3.3; Appendix A.1 | `forge/model/synthesis_program_graph.py`, `synthesis_program_layout.py`, `reaction_program_transformer.py` | Whole-graph encoding, role/core masks, program embeddings and morphology |
| §§3.3–3.4; Appendix A.2–A.3 | `forge/flow/rstar.py`, `forge/model/reaction_program_flow.py`, `synthesis_program_training.py` | Reference rates, training objectives and source balancing |
| §3.5; Appendix A.4 | `forge/model/synthesis_program_sampling.py`, `forge/model/_synthesis_sampling/` | Capped numerical updates, fixed-coordinate projection and terminal core-saturation decoding |
| §3.6; Appendix A.6–A.7 | `forge/synthesis/assessment/`, `forge/synthesis/engine/`, `forge/synthesis/evidence/`, `forge/synthesis/terminals/` | Bounded L2/L3 assessment, evidence-qualified admission, unresolved and censored outcomes |
| §4; Appendix B | `experiments/phase1/multireaction/`, `configs/multireaction/`, `paper/forge_paper/` | Study orchestration, fixed budgets, seed aggregation and table generation |

Appendix C's theorem is about an ideal reference-population CTMC with stated assumptions. It is
not a guarantee of the finite trained model, capped sampler or terminal decoder. Exact L1 is a
computational transform check; route closure and experimental execution are separate outcomes.

## Tables

Links point to retained table inputs. These rendered artifacts can be inspected without weights;
recomputing the measurements requires the historical inputs described in [ARTIFACTS.md](ARTIFACTS.md).
All seed means/SDs use training seeds as independent units, not attempts as replicates.

| PDF item | Evidence file in `paper/v1_iclr/` | Producer / upstream contract |
|---|---|---|
| Table 1 — conditioned, null, cyclic | [shared_program_figure_rows.tex](../v1_iclr/generated/shared_program_figure_rows.tex) | `paper/forge_paper/gem_table1.py`; `configs/reproduction/gem_table1_core_saturation_complete_v1.json` |
| Table 2 — common Ugi benchmark | [common_ugi_benchmark_completed_rows.tex](../v1_iclr/generated/common_ugi_benchmark_completed_rows.tex) | `paper/forge_paper/completed_evidence_v1.py`; `configs/reproduction/natbiotech_v1_completed_evidence_v1.json` |
| Table 3 — production comparison | [production_comparison_transposed_rows.tex](../v1_iclr/generated/production_comparison_transposed_rows.tex) | `completed_evidence_v1.py`, `gem_table4.py`; matched production contracts |
| Table 4 — exact seed counts | [production_seed_exact_counts_rows.tex](../v1_iclr/generated/production_seed_exact_counts_rows.tex) | `completed_evidence_v1.py`, `gem_table6.py`; final conditioned evaluations |
| Table 5 — decoder / source ablation | [decoder_source_ablation_rows.tex](../v1_iclr/generated/decoder_source_ablation_rows.tex) | `gem_table5.py`; `configs/reproduction/gem_table5_decoder_source_ablation_v1.json` |
| Table 6 — architecture ablation | [architecture_ablation_rows.tex](../v1_iclr/generated/architecture_ablation_rows.tex) | `gem_table8.py`; `configs/reproduction/gem_table8_architecture_ablations_v1.json` |
| Table 7 — benchmark seed vectors | [common_ugi_seed_rows.tex](../v1_iclr/generated/common_ugi_seed_rows.tex) | `completed_evidence_v1.py`; completed-evidence contract |
| Table 8 — decomposition outcomes | [common_ugi_decomposition_rows.tex](../v1_iclr/generated/common_ugi_decomposition_rows.tex) | `completed_evidence_v1.py`; common assessments |
| Table 9 — finite catalogue comparison | [catalogue_comparison_transposed_rows.tex](../v1_iclr/generated/catalogue_comparison_transposed_rows.tex) | `gem_table9.py`; `configs/reproduction/gem_table9_catalogue_comparison_v1.json` |
| Table 10 — structural realism | [lipid_realism_rows.tex](../v1_iclr/generated/lipid_realism_rows.tex) | `gem_table7.py`; `configs/reproduction/gem_table7_lipid_realism_v1.json` |
| Table 11 — HeLa diagnostic | [completed_evidence_macros.tex](../v1_iclr/generated/completed_evidence_macros.tex), `ForgePotency*` macros | `results/phase1/ugi_high_potency_challenger_adjudication_v1/result.json`; failed advancement criterion retained |
| Table 12 — chemistry definitions | Related [manuscript source](../v1_iclr/FORGE_ICLR2027_paper.tex), appendix reaction-family table | Descriptive reaction definitions, not a new measured result |
| Table 13 — generated atlas | [atlas_rows.tex](../v1_iclr/figures/forge_generated_sample_atlas_core_saturation_v1/atlas_rows.tex), [chemical drawings](../v1_iclr/figures/chemical_drawings/) | `paper/forge_paper/sample_visualization.py`; frozen sample-atlas receipts |

The shared-null row in Table 2 uses its common-benchmark assessment; it must not be substituted
for the differently evaluated shared-null row in Table 1. The three-family catalogue analysis in
Table 9 and common Ugi analysis in Table 2 also use different novelty reference protocols.
Neither their denominators nor their novelty counts should be mixed.

## Figures

| PDF item | Source material | Evidence class |
|---|---|---|
| Figure 1 — method schematic | [schematic_body.tex](../v1_iclr/figures/forge_algorithm_schematic/schematic_body.tex) | Method illustration, not a measured result |
| Figure 2 — imaging and ROI values | [mouse image](../v1_iclr/experimental/in_vivo/mouse_imaging_supplied.png), [ROI CSV](../v1_iclr/experimental/in_vivo/reported_roi_values.csv) | Author-reported measurements; newer PDF layout differs from older TeX |
| Figure 3 — reaction schemes | [schemes_body.tex](../v1_iclr/figures/forge_reaction_schemes/schemes_body.tex) | Registered reaction illustrations |
| Figure 4 — Ugi products and recovered precursors | [chemical_drawings/](../v1_iclr/figures/chemical_drawings/) (`sample_*`), [render.py](../v1_iclr/figures/chemical_drawings/render.py) | Deterministic examples from frozen computational results |

## Experimental record

The [author-supplied record](../v1_iclr/experimental/in_vivo/evidence_record.json) and
[ROI CSV](../v1_iclr/experimental/in_vivo/reported_roi_values.csv) preserve the original inputs.
The four signals are 599,700; 2,036,000; 2,881,000; and 219,500 RLU for FORGE-1, FORGE-2,
FORGE-3 and MC3. `2,881,000 / 219,500 = 13.125…`, reported as 13.1-fold in the paper.
They are not means over biological replicates; no significance test or error bars are supplied.

**Administration route: intramuscular**, as stated in §4.6 of the submitted PDF and confirmed by
the user during reviewer cleanup on 2026-09-30. The dated
[clarification record](experimental_clarifications.json) supersedes the provisional intravenous
description in the older evidence record for this submission. The older record remains unchanged
for traceability; this clarification does not independently verify laboratory execution.

There are remaining author questions before claiming exact experimental reproducibility:

- The abstract/§4.6 use potency language, while the introduction and Appendix B.5 distinguish the
  single reporter values from biological potency and variability. The README reports the signals
  and fold ratio without adding a statistical efficacy claim.
- The retained record still lacks replicate counts, animal/site allocation, exact candidate and
  checkpoint identifiers, purification/identity/purity records and complete formulation/imaging
  details. Three lipid designs are not three biological replicates of one treatment.

These are existing evidence gaps, not reasons to alter results during a repository cleanup.

## Study boundaries

- This submission: shared three-family historical model; final evaluation checkpoint step **9,143**;
  RNG seeds **20260825, 20260826, 20260827** (seed labels 0, 1, 2).
- The separate `paper/v1_iclr22/` work and `compose_lipid_*` training workflows are **not** evidence
  for this paper's three-family claims. Their computational makeability metric is separate.
- `paper/v0/` and its `configs/reproduction/iclr2027.json` are an archived manuscript contract.
- `docs/PLAN.md` preserves an earlier research plan; it is not a description of the submitted results.
- Present-day refactored code is useful for inspection and development. Historical source
  fingerprints, not filename similarity, establish correspondence to the reported runs.
