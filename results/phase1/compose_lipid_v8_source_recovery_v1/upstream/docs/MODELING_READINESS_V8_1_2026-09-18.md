# COMPOSE-Lipid v8.1 modelling readiness

Date: 2026-09-18. Status: target measure frozen; exact model-program and path
materialization are next. No GPU training has run.

## Outcome

The 200,000-molecule chemistry corpus was not rebuilt or filtered again. Its
component representation and model-facing evaluation split were corrected
before fitting any training distribution.

- Corpus: 200,000 unique constitutional targets, comprising 11,267 reported
  source anchors and 188,733 exact virtual constructions.
- Complete-component manifest: all 200,000 targets resolved into canonical,
  stereo-free precursor constitutions with family roles and multiplicities;
  45,515 globally unique component constitutions.
- Split: 159,782 train, 19,975 calibration, 19,975 formal test, and 268
  reference-only.
- Target measure: all 159,782 train targets, 23 supported reaction families,
  equal expected family mass, bounded source confidence, and no Michael/BEAE
  family tilt.
- Biology, BEAE outcomes, and GPU use: none.

The authoritative artifacts are:

- `artifacts/corpus_build_v2/post_instruction_component_manifest_v8_1/`
- `artifacts/corpus_build_v2/post_instruction_generator_splits_v8_1/`
- `artifacts/corpus_build_v2/post_instruction_target_measure_v8_1/`
- `audits/post_instruction_generator_splits_v8_1_independent.json`
- `audits/post_instruction_target_measure_v8_1_independent.json`

## Why the old split was superseded

The v8 corpus remains the accepted chemistry release. The old v8 split is no
longer model-facing because its precursor grouping used family metadata IDs
rather than a uniform structural identity. That created two problems:

1. some IDs collapsed many distinct complete reagents, most visibly in the
   Vitamin-B5 family; and
2. the same complete molecular constitution could have different IDs across
   families.

When the old 7,002-row `unseen_precursor_identity` panel was re-audited using
canonical complete-component structures, 3,734 rows had no component structure
that was actually absent from training. This was an evaluation-label defect,
not evidence that the molecules or reaction constructions were wrong.

The corrected v8.1 split uses globally scoped canonical component
constitutions. Its independent audit found:

- zero selected component structures in training;
- zero selected source studies in training across all formal families;
- zero exact-combination semantic failures;
- zero coarse-morphology semantic failures; and
- every formal test row assigned to at least one named panel.

The term **regional morphology** is deliberate. The corresponding signature is
a grouped, coarse regional/whole-molecule descriptor. It is not exact graph
topology and must not be reported as such.

## Frozen test panels

The 19,975 formal test targets cover:

- unseen complete component structure: 9,259 targets;
- unseen exact component combination: 4,676 targets;
- unseen coarse regional morphology: 5,512 targets; and
- globally held source study: 532 targets.

Four targets belong to more than one panel. Exact-combination targets use only
components observed in the training set of the same reaction family. Selected
source PMIDs are absent from training across every formal family.

| family | components | held structures | train | cal | test | component | combination | morphology | study | source train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `a3_amine_aldehyde_alkyne` | 172 | 9 | 6,144 | 768 | 768 | 226 | 312 | 230 | 0 | 640 |
| `acid_epoxide_diester_multistep` | 5,591 | 97 | 8,560 | 1,070 | 1,070 | 467 | 282 | 321 | 0 | 185 |
| `aema_aza_thiol_addition` | 373 | 21 | 4,900 | 613 | 613 | 460 | 0 | 109 | 44 | 815 |
| `aldehyde_ugi3` | 132 | 4 | 5,541 | 692 | 692 | 205 | 280 | 207 | 0 | 861 |
| `aldehyde_ugi4` | 368 | 14 | 8,908 | 1,113 | 1,113 | 407 | 250 | 333 | 127 | 487 |
| `alpha_isocyanoester_dihydroimidazole` | 2,220 | 123 | 8,351 | 1,044 | 1,044 | 648 | 83 | 313 | 0 | 77 |
| `amine_alkylation` | 6,947 | 74 | 8,059 | 1,008 | 1,008 | 302 | 404 | 302 | 0 | 189 |
| `amine_epoxide_opening` | 1,921 | 78 | 6,833 | 854 | 854 | 385 | 100 | 256 | 113 | 367 |
| `aryl_reductive_amination` | 4,020 | 33 | 4,836 | 605 | 605 | 181 | 243 | 181 | 0 | 117 |
| `aza_michael_acrylamide` | 750 | 19 | 5,116 | 640 | 640 | 191 | 257 | 192 | 0 | 130 |
| `aza_michael_acrylate` | 1,093 | 51 | 7,020 | 878 | 878 | 403 | 0 | 252 | 223 | 1,795 |
| `disulfide_michael` | 4,759 | 84 | 7,739 | 968 | 968 | 290 | 388 | 290 | 0 | 19 |
| `epoxide_opening_o_acylation` | 1,315 | 53 | 5,071 | 634 | 634 | 409 | 10 | 190 | 25 | 0 |
| `iphos_ring_opening` | 5,446 | 89 | 7,495 | 937 | 937 | 296 | 360 | 281 | 0 | 397 |
| `ketone_isocyanide_amide` | 2,214 | 123 | 9,495 | 1,187 | 1,187 | 704 | 127 | 356 | 0 | 468 |
| `ketone_ugi4` | 114 | 9 | 5,897 | 737 | 737 | 642 | 0 | 95 | 0 | 50 |
| `maleate_addition` | 1,415 | 43 | 5,807 | 726 | 726 | 332 | 177 | 217 | 0 | 323 |
| `o_esterification` | 1,397 | 32 | 4,873 | 609 | 609 | 270 | 157 | 182 | 0 | 5 |
| `passerini_3cr` | 4,040 | 59 | 7,007 | 876 | 876 | 266 | 348 | 262 | 0 | 0 |
| `preassembled_thiol_yne_tail_amidation` | 1,415 | 45 | 10,620 | 1,327 | 1,327 | 1,189 | 0 | 138 | 0 | 45 |
| `reductive_amination` | 2,579 | 34 | 5,831 | 729 | 729 | 217 | 294 | 218 | 0 | 28 |
| `thiolactone_aminolysis_michael` | 1,927 | 70 | 9,483 | 1,185 | 1,185 | 537 | 293 | 355 | 0 | 68 |
| `vitamin_b5_multistep` | 1,102 | 24 | 6,196 | 775 | 775 | 232 | 311 | 232 | 0 | 12 |

Panel counts are not required to be equal within a family. The split fills each
family's fixed 10% test quota using the strongest available non-overlapping
structural groups. A zero in one panel therefore means another panel used that
family's quota; it does not remove that family from formal evaluation.

## Train-only target measure

The target measure follows decision 0037:

- each of the 23 supported reaction families receives probability mass
  `1 / 23`;
- within a mixed family, a source target can receive at most four times the
  per-target mass of a virtual target;
- targets are uniform within each family/evidence stratum;
- regional morphology remains a joint empirical train-only condition rather
  than independently sampled marginal knobs; and
- no Michael-addition or BEAE-adjacent family receives extra base-model mass.

The realized global source-anchor probability mass is 0.1352664556. This is
below 0.5 because the fourfold per-target cap prevents a few source rows from
dominating much larger virtual strata. The measure contains 15,162 empirical
joint family-core/morphology groups.

All 159,782 train targets retain mass, including 2,297 source targets whose
compatible decompositions still need projection into the exact model-program
format. They may not be silently dropped or replaced. A projection failure is
a readiness blocker to fix, not permission to redraw the training set.

## FORGE-derived inductive biases that remain binding

The corpus and measure do not replace the model architecture. The base process
must retain:

- anonymous, permutation-equivariant region and port embeddings;
- canonical reaction roles rather than source-specific component IDs;
- joint regional context rather than independent head/tail marginals;
- semantic regional organization without component-factorizing the generated
  molecule;
- exact reaction program, scaffold, port, occupancy, valence, charge, and
  hydrogen support as hard constraints;
- branch, length, unsaturation, ester placement, ring, and composition as soft
  family-aware morphology signals rather than universal hard quotas;
- alias aggregation for primitive actions reaching the same successor; and
- exact execution state kept separate from canonical identity/deduplication.

The regional context is fitted on train only. Calibration may select the soft
regional guidance strength and compare the regional model with its matched
ablation. Formal test and BEAE do not select the base model.

## Next gate

1. Project every weighted supervision record into an exact model program.
2. Preserve one frozen program witness per target; do not multiply target mass
   by alternative routes or path length.
3. Materialize exact paths in restartable shards with one CPU per worker and at
   most four local workers.
4. Run CPU replay, candidate-support, alias-aggregation, cached/live parity, and
   regional permutation-invariance checks.
5. Only then launch the bounded regional-context versus matched-ablation health
   run.

The base chemistry process remains broad and outcome-blind. The later pulmonary
controller operates over frozen supported regional programs and scores completed
endpoints; it does not alter this training measure.
