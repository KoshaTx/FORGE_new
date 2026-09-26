# Evidence needed for the 22-family model

Audit date: 2026-09-24. Status: proposed evaluation plan; no new training, sampling,
sealed-holdout access, or paid jobs launched for this audit.

## Current manuscript scaffold

`RESULTS_CHECKLIST.md` and `RESULTS_LEDGER.json` now map all planned 22-family results to explicit
manuscript placeholders. `sections/results_22.tex` and `sections/appendix_22.tex` are the editable
new-model result slots. All final cells remain pending. The original prose below is a dated audit
of the earlier completion stage, not the latest development status; its historical numbers are not
updated or promoted into new-model final results.

Later development context is recorded in
[structural-quality milestone](../../results/phase1/compose_lipid_quality_milestone_v3/README.md)
and [ongoing repair comparison](../../results/phase1/compose_lipid_structure_repair_v1/).
These TRAIN-derived diagnostics remain separate from independently trained, held-out final evidence.

## Assessment

The original paper supports its principal generation result with three independently
trained seeds, 3,072 attempts per program per seed, and independently trained
conditioned, null, and cyclic controls. The new model has one trained checkpoint and
small TRAIN-derived development evaluations. Its checked completion improves exact
assembly, but does not yet establish equivalent quality, generalization, realism,
or a causal benefit from learned reaction conditioning.

We should first finish the metric contract and investigate the weakest families on
development data, then freeze the model and inference procedure before collecting
the final replicated results. More optimizer updates may help, but the current
pilot-versus-production comparison changes both exposure and batch size and cannot
answer that question causally.

This document inventories the results needed to replace the original model's
evidence. It is not a requirement to rerun every exploratory appendix experiment
before the next development test. Guidance experiments remain claim-dependent and
subject to the repository's existing authorization and gates.

## Sources and current evidence

The manuscript reviewed is [the copied paper](FORGE_ICLR2027_paper.tex), including
its results and appendix. Its generated tables, figures and bibliographies are local snapshots of the historical inputs,
with source paths and hashes recorded in `iclr_provenance.json`. Copying the paper
did not create a separate set of new-model evidence. New tables must eventually use
versioned, pinned new-model inputs, preserving the original results.

Historical numbers below are **paper-reported**: the TeX tables, macro override order,
and reproduction configurations were inspected. This audit does not claim to have
reproduced the historical experiments. Input hashes, table inventory, and availability
checks for historical pins are recorded in
[audit.json](../../results/phase1/compose_lipid_paper_evidence_audit_v1/audit.json).

Current observations come from:

- [Training completion](../../results/phase1/compose_lipid_training_eight_fp32_v2/completion-summary.json):
  2,794 optimizer steps, 402,336 presentations per family, 22 families. The training
  completion file's `generated_quality_evaluated: false` predates subsequent evaluations.
- [Qualified cohort](../../results/phase1/compose_lipid_training_cohort_v1/cohort.json):
  1,192,065 admitted records; 80,401 pending records excluded; `vitamin_b5_multistep`
  has no admitted training support. Record counts are not automatically unique product counts.
- [Fresh completion confirmation](../../results/phase1/compose_lipid_completion_confirmation_v1/result.json):
  64 requests per family, 1,408 total, one fixed checkpoint. Raw exact yield is
  404/1,408 (28.7%); strict is 509/1,408 (36.2%); explicitly completed is
  768/1,408 (54.5%). Completed valid-connected yield is 1,223/1,408 (86.9%).
  The 702 novel exact products are whole-product TRAIN novelty, not paper-equivalent
  role-specific component novelty. Aggregate uniqueness is summed within families;
  it does not establish global cross-family uniqueness.
- [Decode diagnosis](../../results/phase1/compose_lipid_decode_diagnosis_v1/):
  fixed-weight decoder comparisons and a small gold-control denoising probe. These
  are development evidence, not held-out convergence or independent training replication.

The original headline exact yields were Ugi 96.4 ± 2.1%, repeated aza-Michael
72.2 ± 1.6%, and repeated reductive amination 52.3 ± 3.4% across training seeds.
These are useful historical reference points, not directly comparable targets for
the new 22-family macro average. The data, program definitions, request populations,
inference procedure, and training schedule have changed.

## Result-by-result collection matrix

Labels refer to the existing manuscript's LaTeX table identifiers. “Missing” below
means missing from the reviewed new-checkpoint evidence, not an assertion that no
related artifact exists elsewhere in the repository.

| Evidence block / original location | Data points to collect for the new model | Existing evidence and remaining work |
|---|---|---|
| **1. Data and program coverage** — Experimental settings; reaction-scheme figure | Qualified records and unique product constitutions by family, registered program, source study, role and split; unique precursor inventories; evidence tier; admitted/excluded counts and reasons; train/CAL/TEST overlap checks; size, closure, repeat and reaction-step distributions; actual sampling mass and presentations | Cohort, weights and admission pins exist. Assemble a paper-facing census from those pins. Preserve exclusions and the unsupported B5 family. Broad family labels alone do not identify equivalent chemistry. |
| **2. Main generation quality** — `iclr-shared-programs`, `production-comparison`, `production-seed-counts` | Attempt counts; validity and connectedness separately; exact L1 yield; distinct exact yield; candidate and verified decomposition coverage; candidate replay precision; exact ambiguity; abstention, invalidity and source-domain failure counts; fixed-coordinate and support violations. Report every family and registered program, seed, and inference arm | Small development raw/strict/completed results exist. Need frozen held-out evaluation, independent training seeds, complete metric alignment, and per-program/size/repeat strata. Report equal-family macro and pooled counts explicitly; neither may hide a failing family. |
| **3. Learned reaction-conditioning effect** — semantic-controls subsection and production tables | Independently trained conditioned/null/cyclic arms with matched data, exposure, optimizer, checkpoint rule, request budget and verification. Per-seed paired exact-yield differences and ranges; validity, novelty and diversity alongside yield | No matched controls for this checkpoint. Wrong-label inference on the current model is only a diagnostic. Audit information visible through layouts, typed cores, masks, routed heads, source priors and completion. A cyclic label with correct roles/core tests the incremental label contribution, not removal of all reaction information. |
| **4. Decoder, noise and completion contribution** — `decoder-source-ablation` | Paired raw, strict and checked-completion readouts; core-saturation versus topology policy; program-role versus global noise if that claim is retained. Exact gains/losses, altered coordinates, accepted/rejected proposals and reasons, verifier calls, latency, product/component concentration and novelty changes | Fixed-checkpoint diagnosis and fresh completion confirmation exist. Repeat on the final frozen evaluation seeds. Separate inference-only comparisons from retrained source-prior comparisons. Completion's batch admission order and grouping are part of the frozen method. |
| **5. Novelty, diversity and catalogue comparison** — `catalogue-comparison`, component-novelty subsections | Whole-product novelty against TRAIN and separately the full declared reference; role- and program-indexed component novelty; exact component-novel yield per attempt; distinct component-novel yield separately; component frequencies, Shannon effective counts, concentration; distinct-product ECFP4 diversity; finite catalogue inventory and reachable tuple counts; catalogue coverage of each split | Current product novelty, inverse-Simpson counts, and any-TRAIN-component diagnostics are partial. Need the paper's role-specific identity contract and Shannon metric. Rebuild finite-catalogue comparisons using admissible training components only. Tuple counts are not unique-product counts; component novelty is not proof of strong catalogue escape. |
| **6. Generalization** — held-components subsection, `ugi-common-benchmark`, `baseline-seed-results` | Exact products containing designated held components per attempt; unique held identities recovered, role-wise coverage and denominators; component-disjoint performance; source-study/size/repeat strata where supported by frozen splits. Held-reaction-family results remain secondary stress tests | Current layouts are TRAIN-derived. Fresh sampling seeds do not provide held-out generalization. Freeze all model/decoder choices on TRAIN/CAL before one final TEST assessment. Never use TEST identities to select proposals or enforce admission. |
| **7. External and internal baselines** — `ugi-common-benchmark`, `baseline-seed-results`, `baseline-decomposition` | Common split, method-visible inventory, attempt definition, seed set, verifier and budget ledger for finite catalogue, learned inventory selector, matched null/post-hoc, FACT matched/generous and the paper's external comparator set (DeFoG, GenMol/SAFE, RGFN), where their native contracts apply. Record all failed attempts, unsupported programs, training/inference/verification cost, ambiguity and abstentions | Historical baseline rows cannot stand in for comparisons to the new model on a changed benchmark. First establish a common Ugi/three-program bridge. Expand to other qualified programs only when supported. Do not force native methods into incompatible chemistry or present unsupported runs as performance zeros. Retain genuine runtime/no-output failures with their reason and denominator. |
| **8. Lipid realism and distribution fidelity** — `lipid-realism` | Method-blind fingerprint and descriptor precision/coverage against a source-study-held-out lipid reference; grouped classifier two-sample AUC; molecular and component effective counts; descriptor distributions and tails, size, branching, cycles, charge and role morphology. Separate chemically scoped comparisons from broad-corpus distribution comparisons | Descriptor means and pairwise diversity exist. No corresponding new-model source-held-out realism study is established. High novelty/diversity cannot establish realism. Preserve the original negative realism finding when discussing historical evidence. |
| **9. Architecture and joint generation** — `architecture-ablations` | If mechanistic claims are retained: input-only program versus layerwise conditioning; role/core losses; routed adapters/heads; gradient-conflict control; whole-graph versus role-factorized models with matched and generous capacity. Exact yield by family, held-out denoising losses, held-component yield, diversity and residualized cross-role dependence | Historical ablations used a separate shorter training study and must not be mixed with final production rows. Map each ablation to the actual new architecture. Additional repeat-supervision ablation is relevant to observed failures. Stage these after the final method is settled; do not run an uncontrolled Cartesian sweep. |
| **10. Training adequacy and efficiency** — settings plus new scaling evidence | Per-family loss and corruption-time curves; atom/bond/parent/closure/repeat errors; exact yield at predeclared development checkpoints; update appearances per family, presentations, batch size, LR, clipping/gradient statistics; wall time, hardware, memory, checkpoint and evaluation cost. Matched-exposure batch/update comparison, separately from exposure scaling | Numerical correctness, restart and execution timing are established for the pinned FP32 path. Quality is not. The existing low-exposure pilot cannot isolate the effect of fewer updates. Source-control reconstruction errors are diagnostic and are not all chemical errors. |
| **11. Synthesis evidence and route dispositions** — `route-evidence-baselines`, `route-dispositions` | For a frozen method-blind component union and candidate sample: exact L1, L2 evidence and L3 terminal status; complete/unresolved/censored products and components; evidence IDs, source hashes, route depth and call/time budgets; complete-route yield per generation attempt and unresolved-reason counts | New exact-L1 products have not inherited old route dossiers. Assess new identities against pinned evidence. Preserve bounded-library abstentions; do not call reaction-enumerated support route-certified. Ugi can remain the deepest route case without claiming equally deep coverage of all 22 families. |
| **12. Guidance, illustrations and formal claims** — guidance subsection, `hela-property-guidance`, sample figures/atlas, theory appendix | Guidance only if retained as a new-model claim and authorized: matched unguided/post-hoc/in-trajectory budgets, zero-guidance identity, eligible/nonuniform groups, route-ready yield and abstentions. For figures: frozen representative selection, exact products, atom origins, precursors, novelty references, and failure examples. Recheck mathematical assumptions for the actual support and completion mapping | Historical negative guidance results can remain explicitly historical or be omitted from a narrowed narrative. Broad biological optimization and wet-lab work are outside scope. New model figures must come from admitted new outputs. Proof assumptions and decoder/completion semantics need review; historical theorems do not certify an altered implementation automatically. |

## Metric contract to settle before final sampling

1. **Attempt denominator:** every requested sample counts, including malformed output,
   invalid graphs, source-domain failures, abstentions, rejected completion proposals,
   and exceptions. Preserve original, proposal and selected output with one immutable
   attempt ID. Report unique products within each family and globally separately.
2. **Identity:** canonical constitutional stereo-free graphs, with the paper's charge,
   bond, mapping, disconnectedness and tautomer policy. Precursor equality is role-
   and program-step-aware; repeated occurrences retain registered semantics.
3. **Verification:** distinguish candidate-search coverage from coverage after exact
   replay. Report candidate passing traces / checked traces as precision, and
   verified products / attempts as yield. Replay of already accepted exact traces is
   a verifier invariant, not independent evidence of perfect candidate precision.
4. **Novelty:** product novelty, any-component novelty, role-indexed component novelty,
   designated held-component recovery, and full-reference novelty are separate
   endpoints. State attempt versus distinct-product denominators. Ambiguous exact
   products remain in exact yield but do not silently enter a unique-decomposition
   component-novelty endpoint.
5. **Uncertainty:** final seed summaries use mean ± sample SD across independently
   trained models, with paired seed contrasts and observed ranges. Request-level
   intervals may describe a fixed model, but do not replace retraining. Completion's
   population-level admission guards can couple requests; uncertainty for completed
   outputs should respect those batches rather than assume independent Bernoulli draws.
6. **Conditioning and layout budget:** inventory all information supplied to the model
   and decoder, including template/morphology selection. A TRAIN-derived layout may
   carry size, role, repeat or program information. Do not describe an oracle layout
   evaluation as unconditional discovery of that information. Freeze an admissible
   request distribution and use it consistently across comparisons.
7. **Completion fairness:** report the learned model before and after completion. Give
   comparator arms the same eligible completion/checking opportunity when testing a
   learned-conditioning claim, or explicitly factor out that opportunity. Charge all
   checks and proposals to the budget. Freeze the TRAIN membership reference, donor
   choice, candidate cap, request order, admission batch size and concentration rule.

## Historical comparisons that need an explicit bridge

- Map original `ugi_3cr_agile`, repeated BL and repeated LX to qualified new programs
  by reaction ID/version, role binding, occupancy/repetition, core and source domain.
  Similar family names alone are insufficient. Report old and new on a common
  admissible benchmark if claiming improvement or regression.
- The final production, common-Ugi, catalogue and architecture tables are different
  studies. The architecture configuration specifies 1,700 steps, while the original
  final decoder study uses checkpoint 9,143. Do not merge their values as one model.
- The common benchmark reports 862.6 distinct component-novel Ugi products per 1,000,
  while the fixed-catalogue comparison reports 839.2 component-novel products per
  1,000. Reconcile inventory visibility, identity, uniqueness and denominators from
  the source ledgers before selecting a new headline metric; this audit does not
  assume the two endpoints are interchangeable or label the difference an error.
- TeX override order matters: final common-route macros specify 11,021 union
  components and 30 closed components; the final route sample specifies 256 exact
  products, zero complete products, and 4/379 complete components. Earlier values in
  `completed_evidence_macros.tex` are superseded. These are historical bounded-evidence
  results, not assertions about new products or about chemical impossibility.

## Recommended execution order

**A. Finish the evidence specification and development diagnosis.** Build the program
bridge and metric exporter; reuse stored attempts for available metrics. Investigate
maleate and preassembled thiol-yne first, then the other low-yield programs. Classify
source-domain rejection, inverse-search failure, repeated-component disagreement,
core saturation and topology errors separately. Distinguish a checker coverage gap
from a learned-model failure using admitted source controls and round trips.

**B. Resolve the training question with a bounded development comparison.** Compare
smaller family batches/more updates at equal presentations, preserving data weights,
model, decoder and evaluation panel. Keep a separate exposure axis if testing longer
training. Record schedule/LR choices in advance and use TRAIN/CAL only. The original
381 versus 9,143 family-containing updates motivates this test; it does not prove
undertraining. Do not pick a final schedule using repeated TEST evaluations.

**C. Freeze one method and replicate the main study.** If matching the original
sampling scale, use three independent training seeds and 3,072 attempts per family
per seed for each of conditioned, null and cyclic:

- 22 × 3 × 3,072 = **202,752 attempts per trained arm** across seeds.
- Three arms total **608,256 generation attempts**, based on **nine training fits**.
- Raw/strict/completed comparisons can reuse each latent trajectory when the
  implementation permits; verifier and completion costs remain additional.
- Per-program allocation within families must be prespecified; 3,072 per broad
  family does not imply 3,072 for every registered program inside it.
- The current checkpoint is a development model. Decide prospectively whether its
  seed can appear in descriptive reporting. Fresh training seeds give a cleaner
  confirmatory estimate after architecture/decoder choices used this checkpoint.
- These counts describe a proposed design, not an authorization or cost estimate.
  Measure end-to-end evaluation throughput before committing the final budget;
  the observed training runtime does not predict verification/runtime costs.

**D. Collect novelty, generalization, realism and baseline evidence on the frozen
outputs and common contracts.** Much is rescoring rather than new generation. Keep
external baselines scoped to supported programs. Add architecture experiments for
the mechanistic claims actually retained, with matched schedules and capacity.

**E. Update routes, figures and manuscript claims.** Reassess new precursor identities
against the bounded evidence library; report unresolved outcomes. Regenerate tables,
captions and representative samples from hash-pinned ledgers. Retain negative results
and limitations. No prospective experiments, candidate procurement, panel locking,
or new biological optimization are required by this plan.

Long-running paid work, if subsequently executed within authorization, must use
detached jobs, persisted identifiers/input pins, incremental checkpoints/results,
and restartable monitoring. Historical missing outputs are not a prerequisite for
new training; unresolved historical comparisons should instead be labelled as such.

## Fresh development results by family

The following table is generated from the pinned confirmation results. All counts
use 64 TRAIN-derived requests, one checkpoint, and are development observations.
They must not be substituted for the original three-seed held-out results.

| Family | Raw exact | Strict exact | Completed exact | Completed valid-connected |
|---|---:|---:|---:|---:|
| `a3_amine_aldehyde_alkyne` | 13 | 35 | 38 | 47 |
| `acid_epoxide_diester_multistep` | 27 | 32 | 32 | 64 |
| `aema_aza_thiol_addition` | 19 | 20 | 45 | 60 |
| `aldehyde_ugi3` | 41 | 51 | 51 | 64 |
| `aldehyde_ugi4` | 35 | 59 | 59 | 64 |
| `alpha_isocyanoester_dihydroimidazole` | 21 | 29 | 29 | 35 |
| `amine_alkylation` | 0 | 0 | 17 | 64 |
| `amine_epoxide_opening` | 1 | 1 | 46 | 58 |
| `aryl_reductive_amination` | 13 | 18 | 18 | 64 |
| `aza_michael_acrylamide` | 2 | 2 | 40 | 62 |
| `aza_michael_acrylate` | 2 | 4 | 43 | 63 |
| `disulfide_michael` | 0 | 0 | 40 | 51 |
| `epoxide_opening_o_acylation` | 4 | 5 | 38 | 55 |
| `iphos_ring_opening` | 41 | 40 | 42 | 44 |
| `ketone_isocyanide_amide` | 39 | 51 | 51 | 52 |
| `ketone_ugi4` | 8 | 22 | 22 | 35 |
| `maleate_addition` | 0 | 0 | 0 | 50 |
| `o_esterification` | 26 | 29 | 46 | 62 |
| `passerini_3cr` | 57 | 60 | 60 | 64 |
| `preassembled_thiol_yne_tail_amidation` | 1 | 1 | 1 | 55 |
| `reductive_amination` | 8 | 11 | 11 | 57 |
| `thiolactone_aminolysis_michael` | 46 | 39 | 39 | 53 |
