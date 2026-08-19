# FORGE computational results evidence matrix

Date locked: 4 August 2026  
Machine-readable source: `manuscript/COMPUTATIONAL_RESULTS_EVIDENCE_MATRIX.json`

This document inventories the computational evidence currently present in the
local FORGE repository. It is not a substitute for the manuscript evidence
ledger and it contains no prospective synthesis, formulation or biological
outcomes. Every result below is linked to a locally hash-pinned artifact.

## Claim and status vocabulary

Claim classes follow the paper-writing contract:

- **Measured**: direct experimental observation.
- **Computed**: deterministic or statistical result computed from a frozen
  artifact.
- **Reported**: statement or datum preserved from an external source.
- **Inferred**: interpretation supported by the evidence but not directly
  measured as stated.
- **Proposed**: planned analysis or experiment without a completed result.

Evidence status is one of `verified_frozen`, `verified_nonselecting`,
`diagnostic_only`, `trained_pending_evaluation`, `proposed_pending` or
`historical_stale`.

The following language is locked:

- R1 is **reaction-enumerated support**, not route-certified chemistry.
- The model-facing identity is constitutional and stereochemistry-free.
- Exact Ugi forward assembly establishes L1 graph consistency, not synthesis
  success, L2/L3 closure or procurement closure.
- Exact catalog absence establishes structural component novelty only. It is
  assigned missing route knowledge until route planning is performed and is
  never treated as chemical infeasibility by default.
- Graph support is only within the declared atom, bond, charge, size, branching
  and cycle-rank bounds.
- Morphology-level applicability enrichment is authorized. Potency remains a
  terminal-ranking step, and synthesis trajectory guidance is not authorized.
- The present Ugi program is an AGILE-type amine-aldehyde-isocyanide
  three-component reaction program.

## 1. Corpus and split evidence

| ID | Exact result | Status | Primary evidence | Safe manuscript wording |
|---|---|---|---|---|
| `corpus.r0_constitutional_reconciliation` | 15,433 source rows; 15,229 unique constitutional graphs; 204 rows collapsed; 100 B4 mixture rows excluded from single-structure weighting; zero model stereochemical labels | Computed; verified frozen | `results/m0_03/r0_reconciliation.json` (`16d2cff2…ad310`) | After constitutional standardization, the model-facing corpus comprised 15,229 unique lipid graphs derived from 15,433 source rows. |
| `corpus.constitutional_splits` | Source-study 10,591/2,310/2,328; head 10,659/2,285/2,285; linker/scaffold 10,659/2,285/2,285; component family 12,209/1,920/1,100; zero groups crossed folds | Computed; verified frozen | `results/m0_03/constitutional_split_result.json` (`c43bdfaa…bf32`) | Source-study, headgroup, linker/scaffold and component-family partitions were frozen before model fitting, with no group assigned across folds. |
| `corpus.cross_platform_registry_audit` | Recovery: held head 260/2,285 (11.38%); held linker/scaffold 402/2,285 (17.59%); held component family 100/1,100 (9.09%); full source study 0/2,328 | Computed; diagnostic only | `results/m0_04_constitutional/result.json` (`443c8fa2…4853`) | A cross-platform leakage and reaction-registry coverage audit exposed partial cross-group reconstruction and no complete source-study reconstruction. |
| `corpus.agile_measured_labels` | 1,200 nominal measurements; 1,100 curated single-structure graphs; 100 mixture abstentions; zero HeLa or RAW 264.7 label mismatches | Computed; verified frozen | `results/m0_07/agile_label_reconciliation.json` (`44e9f451…13a7`) | We reconciled all 1,200 AGILE measurement rows, retaining 1,100 single-compound constitutional graphs and treating 100 mixture rows as structure-level abstentions. |
| `corpus.agile_virtual_ugi_L1` | 12,276/12,276 exact unique decompositions under the frozen transform; 93 original components = 22 heads + 62 aldehydes + 9 isocyanides | Computed; verified frozen | `results/m0_09/agile_virtual_ugi3_capability.json` (`9143ae81…a7a0`) | The frozen Ugi transform produced one exact component decomposition for each of 12,276 original virtual products. |
| `corpus.phase1_ugi_union` | 15,229 R0 graphs; 464,265 R1 reaction-enumerated products; 12,386 Ugi-L1 products; 95 current components = 24/62/9 | Computed; verified frozen | `results/phase1/data_contract.json` (`0c6dd056…bfef`); current split manifest `4ac82852…975` | The current Ugi-L1 union comprised 12,386 unique constitutional products assembled from 95 components; the separate R1 corpus comprised 464,265 reaction-enumerated products. |
| `corpus.ugi_component_expansion` | 586 candidate rows; 424 L1-admitted components = 264 heads + 107 aldehydes + 53 isocyanides; 329 outside the current 95; only one E2 route-closed | Computed; verified frozen | `results/phase1/ugi_component_expansion/result.json` (`374e3edc…2666`) | The Ugi registry was expanded to 424 L1-compatible structures; only one expanded component was already closed through the stricter E2 route criterion. |
| `corpus.ugi_expanded_enumeration` | 1,486,415 unique products; 12,386 current-union + 1,474,029 expanded; train/calibration/heldout = 483,032/349,013/654,370 | Computed; verified frozen | `results/phase1/ugi_expanded_enumeration/result.json` (`fbff9d65…abf7`); products `545b0372…0962` | Exact atom-mapped forward assembly yielded 1,486,415 unique constitutional products before balanced-corpus selection. |
| `corpus.ugi_balanced_training_set` | 112,386 selected products = all 12,386 current-union + 100,000 expanded; train/calibration/heldout = 66,464/15,800/30,122 | Computed; verified frozen | `results/phase1/ugi_balanced_chemistry_corpus_v2/result.json` (`1b655ce6…63a6`); assignments `9a50c703…5184`; cache `b862b7a5…9c46` | Deterministic source-stratified, family-weighted selection produced the 112,386-product training corpus. |

### Corpus claim boundaries

The M0-04 percentages are not generator-generalization scores. The test asks
whether components outside a held group, passed through a frozen reaction
registry, reconstruct the exact held product. Structural motif transfer may be
substantial even when exact product reconstruction is zero.

The component counts 93, 95 and 424 refer to different universes:

- 93: original 12,276-product virtual AGILE component set;
- 95: current 12,386-product Phase 1 union;
- 424: expanded L1-admitted Phase 1 registry.

Likewise, 1,497,144 is a raw Cartesian upper bound, 1,486,415 is the unique
enumerated product universe and 112,386 is the selected balanced training
corpus.

## 2. Route evidence and architectural consequence

| ID | Exact result | Status | Primary evidence | Safe manuscript wording |
|---|---|---|---|---|
| `routes.source_adjudication` | 1,100 exact measured L1 records; 100 mixture abstentions; 12,276 virtual transform-consistency records; 45 L2 route instances, of which 44 are structure-resolved; one abstention; zero reported experimental negatives | Computed from reported evidence; verified frozen | `results/m0_05_source_adjudication/result.json` (`02156ba2…3159`); evidence ledger `37e88fe7…3272` | Source adjudication retained 44 structure-resolved upstream routes from 45 route instances and no reported experimental route failures. |
| `routes.L2_supervision_decision` | 45 routes; 44 resolved; 76 reaction instances; 11 route families; 15 exact LX_2024 aldehyde transfers; zero upstream head routes; zero reported negatives; zero complete product dossiers | Inferred from computed evidence; verified frozen | `results/m0_09/l2_supervision_decision.json` (`d6bd79b7…9938`) | Dense product/L1 supervision and sparse upstream-route evidence motivated joint product/L1 generation followed by recursive evidence-aware L2/L3 completion. |
| `routes.conservative_leaf_closure_composition` | 23/24 exact-source component programs and 47/47 family-projected programs have every proposed leaf closed by current procurement records; 1,734/12,276 products have terminal-or-exact upstream leaf coverage and 8,432/12,276 have terminal/exact/family-projected coverage | Computed from reported evidence; verified frozen | component programs `523880bc…3edf`; products `2cb7861b…4fdd`; procurement configs `84577b3c…2960`, `9167b0fe…f157` | Current procurement composition gives conservative upstream-leaf coverage to 1,734 products using terminal or exact-source programs and 8,432 when family-projected programs are included. |
| `routes.exact_source_step_forward_verification` | 46/46 exact-source upstream steps uniquely reproduce their expected canonical product: 15 esterifications, 17 primary-alcohol oxidations, 7 amine formylations and 7 formamide dehydrations; 24/24 component programs pass | Computed from reported evidence; verified frozen | `results/phase1/ugi3_exact_source_forward_verification/result.json` (`1af7734a…1f36`); ledger `830dc166…6a38` | Exact canonical products were uniquely reconstructed for all 46 extracted upstream steps spanning all 24 exact-source component programs. |
| `routes.complete_computational_dossiers` | 23 exact-source routed components are L2-forward-verified and L3-closed; one is L3-open; 47 family projections remain non-exact; 1,734/12,276 products have complete exact-source computational L1/L2/L3 dossiers | Computed from reported evidence; verified frozen | `results/phase1/ugi3_complete_computational_dossiers/result.json` (`74d2cd6d…b0ea`); component ledger `73f20af6…fa5a`; product ledger `4cc614c7…dfb` | Under exact-source route evidence and the frozen procurement snapshot, 1,734 of 12,276 products have complete forward-consistent computational L1/L2/L3 dossiers. |
| `routes.production_registry_route_readiness` | 41/424 L1-admitted components are route-complete: 17 accepted terminals and 24 exact-source, uniquely forward-verified, L3-closed components; the role counts are 17/264 heads, 18/107 aldehydes and 6/53 isocyanides | Computed from reported evidence; verified frozen | `results/phase1/ugi3_production_registry_route_readiness/result.json` (`d748eae9…bf4d`); component ledger `722e42dd…c99d`; gap ledger `13cf7302…f0c` | Among 424 structurally admitted production components, 41 are accepted terminals or have exact-source, forward-verified upstream programs closed to frozen terminal evidence. |
| `routes.proposal_augmented_search_v1` | 173 single-step proposals for 30 one-gap components; 27 targets have a graph-consistent known-family hypothesis; 138 proposal rows remain new-family hypotheses | Computed; verified frozen | `results/phase1/graph2edits_semantic_readjudication_v1/result.json` (`5c05307a…a0b0`) | A learned reaction proposer expanded route search, but proposal confidence alone was not treated as evidence or route closure. |
| `routes.matched_synthesis_guidance_null_v1` | Guided 2/56 versus post-hoc 3/56 route-ready unique terminals; only 5/48 checkpoint groups had mixed route utility | Computed; verified frozen | `results/phase1/ugi_synthesis_guidance_failure_audit_v1/result.json` (`bbbf1d23…a9c2`) | Route-aware resampling did not improve the matched endpoint, so complete-product L1/L2/L3 search is retained before panel lock without claiming trajectory-level synthesis guidance. |
| `routes.hybrid_proposer_recovery_v1` | On 36 hidden exact routes, Graph2Edits recovered 22 at top 1 and 29 at top 5; AiZynthFinder recovered 18 at top 1 and 19 at top 5; the top-5 union did not exceed Graph2Edits | Computed; verified frozen | `results/phase1/hybrid_single_step_recovery_audit_v1/result.json` (`f7f47e6a…dcb5`) | The production cascade checks exact evidence first, uses Graph2Edits as the primary single-step proposer and reserves AiZynthFinder for residual multistep search. |
| `routes.aizynthfinder_residual_v1` | AiZynthFinder returned proposals for all 64 Graph2Edits-residual components and all 97 unresolved leaves; public-stock search solved 44 and 29, respectively | Computed proposal diagnostic; verified frozen | `results/phase1/aizynthfinder_checkpoint_residual_v1/result.json` (`4fce0428…97c`) | Public-stock solutions are retained as route hypotheses and do not create procurement evidence, route closure or synthesis-success probability. |
| `routes.proposal_aware_current_evidence_v2` | Four exact current supplier records increased productive route-ready finals from 3 to 6 and created 14 checkpoint utility changes; 4 new components and 7/48 mixed groups still missed the frozen 5-component and 8-group guidance gates | Computed; verified frozen | `results/phase1/ugi3_proposal_aware_checkpoint_contrast_v2/result.json` (`be566690…312`); supplier pack `4f392215…b4d` | Expanded evidence improves post-generation route assessment, but synthesis tilting and lambda tuning remain unpromoted; do not relax a near-missed gate after observing the result. |

The exact-source-only graph transforms reconstruct every expected product for
the 46 extracted upstream steps, but this does not qualify general substrate
scope or encode reagents and conditions. Family-projected programs remain a
separate, non-exact evidence tier even when their leaves are procurement-closed.
The resulting 1,734 complete computational dossiers do not establish that the
exact lipid was synthesized or that synthesis, formulation or biological
testing succeeded.

Across the expanded production registry, structural support is much broader
than route support. One additional cross-platform aldehyde transfer is now
exact-source, forward-verified and L3-closed, giving 41 route-complete
components in total. The remaining 383 components include one exact-source
L3-open record, 87 family projections, 111 records outside the current route
support and 184 unassessed heads. Structural novelty therefore remains distinct
from route closure.

The justified architecture is therefore hierarchical for two independent
reasons:

1. Product-to-L1 supervision is dense, whereas complete upstream-route
   supervision is sparse, role-imbalanced and positive-heavy.
2. Separating broad molecular generation from route completion prevents the
   narrower documentation distribution from defining molecular support.

A learned single-step proposer is now integrated as a hypothesis generator, but
its output does not create evidence. The matched synthesis-guidance controller
was operational and changed sampled identities, yet failed to improve
route-ready yield. The production workflow therefore performs proposal-augmented
complete-product routing after generation and before prospective panel lock.

## 3. Biological oracle evidence

| ID | Exact result | Status | Primary evidence | Safe manuscript wording |
|---|---|---|---|---|
| `oracle.production_matrix` | Selected role-aware D-MPNN 3-seed ensemble; equal-endpoint calibration R2 0.3673, RMSE 1.8584, Spearman 0.5960; post-selection equal-endpoint test R2 0.1830; HeLa mean test R2 0.2534; RAW mean test R2 0.1126; zero guidance domains authorized | Computed; verified frozen | `results/m0_07/oracle_production_result.json` (`268f8f59…961d`); checkpoint `46f13b8a…e61c` | Across prespecified scaffold and held-component schemes, the selected oracle achieved equal-endpoint post-selection R2 = 0.183 and was frozen with universal guidance abstention. |
| `oracle.lantern_random_split_diagnostic` | LANTERN HeLa random test R2 0.8198 (n=110), RMSE 1.4164, Pearson r 0.9063; FORGE selected oracle on identical random assignments R2 0.7845 | Computed; diagnostic only | `results/m0_07/lantern_reproduction_result.json` (`f5a5b31f…a2df`); FORGE metrics `019d43f1…ef34` | The released LANTERN checkpoint reproduced R2 = 0.820 and the selected FORGE oracle reached R2 = 0.785 on the same random assignments with train-only preprocessing. |
| `oracle.distributional_applicability_v3` | Generated pool: 165 interpolative, 215 boundary and 3,595 extrapolative; 153 exact-new products remain interpolative. Heldout interpolative n=2,732, R2=0.463, rho=0.653; extrapolative n=895, R2=-0.726, rho=0.016 | Computed; verified nonselecting | `results/phase1/ugi_distributional_applicability_v3/result.json` (`a3e5c4bd…61f8`); generated ledger `ed12ac72…992f`; heldout ledger `8af8624a…0da2` | Exact component novelty was recorded independently from a four-view chemical applicability bin; 153 generated products with exact-new components remained interpolative under the frozen definition. |
| `oracle.interpolative_conditional_conformal` | Original foldwise gate passes for held head (4 eligible folds) and held aldehyde--isocyanide pair (3); other role domains fail coverage or calibration-support requirements; zero guidance authorized | Computed; diagnostic only | `results/phase1/ugi_interpolative_conformal_v1/result.json` (`a7ad0ddb…e2ef`) | Conditional uncertainty improved support for selected interpolative role shifts, but evidence remained incomplete across precursor roles and biological guidance continued to abstain. |
| `oracle.applicability_morphology_proposal_v1` | Fresh supported-terminal rate 3.87% broad versus 8.64% promoted; 2.23-fold enrichment; every one of 57,190 qualified programs retains non-zero probability | Computed; verified frozen | `results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json` (`51ff8d26…d796`) | Morphology-aware allocation more than doubled access to the frozen chemical-support region without truncating qualified morphology support. |
| `oracle.high_potency_morphology_challenger_v1` | Group-disjoint morphology classifier AUC 0.623, but fresh matched tail-lane yield was 27 unique conservative-high products for applicability allocation versus 23 for nested potency allocation at 38 oracle calls per arm | Computed; verified frozen | `results/phase1/ugi_high_potency_challenger_adjudication_v1/result.json` (`a508b45d…1cc3`) | The single frozen potency challenger did not improve terminal discovery efficiency; potency morphology tilting was closed and terminal conservative ranking retained. |

The LANTERN source pipeline fits feature and target scaling across all 1,100
records before applying its persisted split. Its reproduction therefore
contains test-information leakage and cannot select the FORGE oracle. The
difference is not assigned entirely to leakage because architecture, features
and checkpoint selection also differ. Random-split results do not establish
held-component transfer.

The production oracle predicts in-vitro transfection; it is not an in-vivo
endpoint oracle. The promoted controller uses no potency labels: it reallocates
morphology probability toward terminal chemistry inside the frozen multiview
support region. Within that region, conservative HeLa predictions may rank
completed products. They do not alter molecular trajectories.

Exact role-specific identity remains a provenance field; applicability is a
separate product-and-component chemical-distance field. The applicability
proposal enriches but does not certify supported terminal chemistry, so every
completed molecule is still assessed. The negative potency challenger closes
additional same-data morphology-potency tuning without weakening the promoted
applicability controller.

## 4. Sparse representation evidence

| ID | Exact result | Status | Primary evidence | Safe manuscript wording |
|---|---|---|---|---|
| `representation.broad_sparse_roundtrip` | 15,229/15,229 exact constitutional round-trips; 1,891/1,891 aromatic round-trips; maximum 282 heavy atoms; N≤64 covers 10,702/15,229; N≤96 covers 13,879/15,229 | Computed; verified frozen | `results/m0_06_sparse_full_support/result.json` (`cfd8b686…0469`) | The sparse constitutional encoding reconstructed all 15,229 accepted lipid graphs exactly, including all 1,891 aromatic structures. |
| `representation.v5_canonical_program` | 15,229/15,229 exact round-trips; 30,458/30,458 atom-permutation tests preserve exact serialization and graph identity; all sparse programs valid | Computed; verified frozen | `results/phase1/v5_canonical_representation_audit.json` (`5374eef0…4148`) | A deterministic sparse offspring program gave exact graph round-trips for all structures and identical serializations under tested atom permutations. |
| `representation.ugi_support_bounds` | Bounds: attachment ≤2, children ≤3, component atoms ≤32, cycle rank ≤2, decorations ≤7, junction budget ≤7, total atoms ≤80; cover 424/424 admitted components, 112,386/112,386 balanced products and 1,486,415/1,486,415 expanded products | Computed; verified frozen | `results/phase1/ugi_support_bounds_audit_v2/result.json` (`ac847df3…a29`) | The declared Ugi support bounds cover the complete admitted component registry and both the balanced and expanded exact-forward product sets. |

The V5 root and traversal are deterministic serialization coordinates, not the
model's chemical semantics. Ugi-specific semantics come from precursor origin,
core membership and attachment context. The Ugi support audit also covers
299/307 LNPDB hydrophobic proxy motifs (97.39%; weighted 98.78%), but this is a
capacity result only—not Ugi compatibility, route evidence or oracle coverage.

## 5. Full-morphology generator evidence

### Training run

The unguided product-plus-L1 generator completed 4,750 deterministic float32
steps on an L4 GPU and stopped by calibration patience. The lowest calibration
loss was 2.8894 at step 500. The run used 66,464 training, 15,800 calibration
and 30,122 held-out products. No route or biological guidance entered training
or sampling.

Primary artifact:
`results/phase1/ugi_joint_sparse_balanced_v2_full/result.json`
(`7427757f…2843`).

Training or held-out loss must not be described as molecular validity, novelty,
route closure or synthesis performance.

### Fixed 256-program checkpoint series

| Step | Valid | Unique valid | Exact frozen products | Median nearest Morgan | Median heteroatoms | Median logP |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 255/256 (99.61%) | 254 | 2 | 0.625 | 3 | 11.0417 |
| 500 | 251/256 (98.05%) | 249 | 2 | 0.620 | 3 | 10.8976 |
| 1,000 | 255/256 (99.61%) | 252 | 1 | 0.625 | 4 | 9.7307 |
| 2,000 | 254/256 (99.22%) | 252 | 9 | 0.6731 | 5 | 8.6990 |
| 4,000 | 256/256 (100%) | 255 | 44 | 0.7524 | 5 | 8.3691 |

Reference medians are six heteroatoms and logP 7.8193. The audit shows a
realism-versus-reproduction trade-off: later checkpoints move closer to the
reference descriptors while reproducing more exact frozen products. This gate
is explicitly nonselecting.

Evidence: `results/phase1/ugi_joint_sparse_balanced_v2_checkpoint_series_audit.json`
(`38b756d2…485a`).

### Held-component open-endedness gate

The probe fixes 512 morphology programs. It does **not** supply component
identities or product graphs. A `held_role` value is inherited from the probe
construction only for stratification. For every valid generated product, the
three actual precursor graphs are recovered by deterministic inverse Ugi
decomposition and then compared with the admitted catalogs.

| Metric | Step 1,000 | Step 2,000 |
|---|---:|---:|
| Valid products | 511/512 (99.80%) | 506/512 (98.83%) |
| Unique valid products | 505 | 501 |
| Three components reconstructed | 511/511 | 506/506 |
| Exact forward product reconstruction | 511/511 | 506/506 |
| All three handles pass | 511/511 | 503/506 |
| At least one component outside all 424 admitted components | 429/511 (83.95%) | 404/506 (79.84%) |
| Train-catalog-only component triples | 66 | 85 |
| Exact heldout component products | 13 | 16 |
| Exact calibration component products | 3 | 1 |
| Exact frozen products | 23 | 65 |
| Exact strict-training products | 18 | 52 |
| Outside-catalog heads | 258/511 | 224/506 |
| Outside-catalog aldehydes | 329/511 | 308/506 |
| Outside-catalog isocyanides | 69/511 | 59/506 |
| Median logP | 10.7275 | 9.8223 |
| Median heteroatoms | 4 | 4 |

At step 2,000, three products failed the head-handle policy because their
inverse-reconstructed head contained three distinct amine sites. Both
checkpoints reconstruct every valid product exactly through the frozen Ugi
adapter. Structural component novelty does not imply a complete route,
synthesis success, product-level novelty or biological activity.

Evidence:
`results/phase1/ugi_joint_sparse_balanced_v2_held_component_novelty_gate.json`
(`cd371b1f…d705`). The step-1,000 and step-2,000 checkpoint hashes are
`90f5f0bd…c532` and `e1a0092a…d21a`, respectively.

### Held-out loss is a separate likelihood audit

| Evaluation | Step 1,000 total loss | Step 2,000 total loss |
|---|---:|---:|
| Complete held-out fold (n=30,122) | 3.6623 | 3.4344 |
| Fixed probe (n=512) | 1.8392 | 1.6638 |

Step 2,000 has lower fixed-noise denoising loss. This does not contradict the
generation gates: step 1,000 is more open-ended and slightly more valid,
whereas step 2,000 is more distribution-conforming and reproduces more frozen
products. This likelihood audit was nonselecting; the later matched v3 gate
described in Section 6 selected step 1,000.

### Unconditional program-prior gate

| Metric | Step 1,000 | Step 2,000 |
|---|---:|---:|
| Valid products | 1,009/1,024 (98.54%) | 993/1,024 (96.97%) |
| Unique valid products | 998 | 990 |
| At least one component outside all 424 admitted components | 844/1,009 (83.65%) | 793/993 (79.86%) |
| All three handles pass | 1,007/1,009 | 989/993 |
| Exact forward reconstruction | 1,009/1,009 | 993/993 |
| Exact frozen products | 53 | 100 |
| Exact strict-training products | 48 | 96 |
| Median logP | 10.5125 | 9.4955 |
| Median heteroatoms | 4 | 5 |

This independent 1,024-draw gate corroborates the same checkpoint trade-off.
Its programs are drawn from the training-fold-weighted unconditional program
prior, without supplying component identities or product graphs. It is also
nonselecting.

Evidence:
`results/phase1/ugi_joint_sparse_balanced_v2_unconditional_component_gate.json`
(`b40ef47d…78d7`).

## 6. Frozen production generator and matched size-only ablation

The matched size-only arm completed training, and the final v3 comparison
sampled both architectures at all five shared checkpoints on the same fresh
1,024-program draw, with identical sampling seed, step count and batch size.
The selector used only the 66,464 training and 15,800 calibration products as
its descriptor and exact-reproduction reference. All 30,122 heldout products
were excluded from architecture and checkpoint selection.

| Training fact | Value |
|---|---:|
| Completed steps | 4,750/8,000 planned |
| Stop reason | calibration early stopping |
| Best calibration step | 500 |
| Best calibration loss | 3.9183 |
| Best checkpoint SHA-256 | `b9767d17…332a` |
| Step-1,000 checkpoint SHA-256 | `523a7555…d16` |
| Step-2,000 checkpoint SHA-256 | `b60649e4…356` |

Evidence:
`results/phase1/ugi_joint_sparse_size_only_v1_full/result.json`
(`2627d842…a16c`). All five downloaded snapshot hashes match the hashes in the
completed result.

| Architecture | Step | Frozen-gate result | Failed check(s) |
|---|---:|---|---|
| Full morphology program | 250 | Fail | Heteroatom KS distance |
| Full morphology program | 500 | Fail | Heteroatom KS distance |
| Full morphology program | **1,000** | **Pass** | None |
| Full morphology program | 2,000 | Fail | Exact forward reconstruction |
| Full morphology program | 4,000 | Fail | Exact-product reproduction ceiling |
| Size only | 250 | Fail | Heteroatom KS distance |
| Size only | 500 | Fail | Heteroatom KS distance |
| Size only | 1,000 | Fail | Inverse component reconstruction |
| Size only | 2,000 | Fail | Inverse component reconstruction; handles |
| Size only | 4,000 | Fail | Declared support; exact-product reproduction ceiling; handles |

Only the full-morphology step-1,000 checkpoint passed every prespecified gate,
so it is frozen as the production product-plus-L1 generator. In the matched
fresh sample, 1,007/1,024 draws were valid; all 1,007 valid products yielded
three inverse-reconstructed components, exact Ugi forward reconstruction and
three qualified handles. Of the 1,024 attempts, 847 contained at least one
component outside the 424-component admitted catalog and satisfied all usable
open-endedness checks (82.71%). Thirty-nine of 1,007 valid products exactly
matched the 82,264 selection-visible products (3.87%). Mean normalized
Wasserstein-1 and mean Kolmogorov–Smirnov distances were 0.1584 and 0.2400,
respectively; no valid product violated declared support or the forbidden
substructure policy.

Evidence:
`results/phase1/ugi_architecture_checkpoint_selection_v3.json`
(`fe90fc84…3be5`), policy
`configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v3.json`
(`a10b4fbd…3bb4`), selected sample
`results/phase1/ugi_architecture_selection_v3/full_step1000/result.json`
(`9066554c…12a`) and production manifest
`results/phase1/ugi_product_l1_production_generator_v1.json`
(`c27352c1…e68a`).

This freeze authorizes unguided complete-product generation, deterministic L1
component recovery and exact Ugi forward verification only. It does not
authorize L2/L3 route completion, synthesis or biological guidance, a
calibrated synthesis-success probability, or experimental claims.

### Nonselecting post-selection candidate-eligibility audit

The frozen checkpoint was then audited without reopening selection. Among its
1,007 valid products, 157 contained an alkyne (15.59%; reference
13,397/82,264, 16.29%) and 61 contained an N–N bond (6.06%; reference
4,123/82,264, 5.01%). No valid generated or reference product contained the
frozen aldehyde, peroxide or carbon-centred allene/cumulene queries. Seventy-eight
valid products fell below the selection-reference 1st percentile for
heteroatom fraction (7.75%; reference 813/82,264, 0.99%); none exceeded the
99th-percentile heteroatom count or fraction. These motif hits are review flags,
not automatic chemistry failures.

All 1,007 valid products remained inside declared graph support, passed inverse
component recovery, exact forward reconstruction and all three handle checks;
zero contained a forbidden substructure. The 17 invalid attempts comprised 16
molecule-sanitization failures and one terminal-support failure. The audit used
only the 82,264 training-plus-calibration reference products and skipped all
30,122 heldout rows without molecule evaluation.

Evidence:
`results/phase1/ugi_candidate_eligibility_audit_v2.json`
(`2619322d…7474`) under config
`configs/model/phase1_ugi_candidate_eligibility_audit_v2.json`
(`6b9abb3d…9aaf`). The superseded v1 audit is retained with an invalidation
record because its initial cumulene SMARTS also matched sulfone O=S=O motifs;
the corrected v2 query requires a carbon centre.

### Descriptive post-selection held-family stress test

The frozen step-1,000 generator was evaluated on 512 morphology programs
associated with held component families. The sampler received only coarse
morphology fields, not source component or product identities. Of 512 attempts,
511 products were valid, 505 were unique, and all 511 valid products passed
inverse component recovery, all three handle policies and exact Ugi forward
reconstruction. Four hundred twenty-nine of 512 attempts contained a component
outside the frozen 424-component catalog. Exact whole-product comparison placed
488 products outside the complete 112,386-product corpus, 18 in training and 5
in heldout.

This is a descriptive stress test, not a pristine heldout performance estimate:
the probe had been inspected before the production checkpoint was frozen. It
does not establish recovery of a requested held component, non-Ugi
generalization, L2/L3 route closure, synthesis or biological function.

Evidence:
`results/phase1/ugi_product_l1_postselection_held_component_stress_v1.json`
(`cf2e6f34…991e`) under config
`configs/model/phase1_ugi_product_l1_postselection_held_component_stress_v1.json`
(`07899965…6e4`).

### Exact post-selection structural provenance and static route diagnostic

Exact role plus constitutional identity partitioned 3,021 reconstructed
component occurrences into 1,053 original current-catalog occurrences, 590
admitted non-original known occurrences and 1,378 occurrences absent from the
424-component catalog. The 590 admitted non-original occurrences comprised 16
with the explicit cross-platform-transfer source class (one exact component
identity) and 574 expanded-known occurrences (60 identities). Molecular
similarity did not assign provenance.

At product level, 30 of 1,007 products used only original current-catalog
components, 130 contained a transferred or expanded known component but no
catalog-absent component, and 847 contained at least one catalog-absent
component. An exact identity join to the frozen route-readiness ledger found 3
products whose three components were all in static route-complete tiers, 157
within-catalog products with at least one noncomplete tier and 847 products with
a catalog-absent component. All 1,378 catalog-absent occurrences were assigned
`not_assessed_missing_route_knowledge`, never chemical infeasibility. This was
static evidence composition, not route planning.

Evidence:
`results/phase1/ugi_postselection_provenance_audit_v1/result.json`
(`1db79178…10cb`) under config
`configs/model/phase1_ugi_postselection_provenance_audit_v1.json`
(`ffe6be8e…f9`). Component and product ledgers are hash-pinned as
`4fb0414f…eb6` and `cb401b54…d7d`.

### Nonselecting post-selection tail-chemotype audit

The same 1,007 reconstructed products were audited by precursor role against
the 82,264 selection-visible training-plus-calibration products. Only 19 of
1,007 generated aldehyde components contained an ester-like carbonyl (1.89%),
compared with 40,136/82,264 reference aldehyde occurrences (48.79%). No
generated aldehyde component contained a carbon-carbon double bond, compared
with 27,028/82,264 reference occurrences (32.86%). Thus, exact product validity
and open-ended component generation coexist with substantial depletion of two
selection-visible aldehyde modes.

The audit also found directly adjacent carbon branch atoms in 27/1,007
aldehyde components (2.68%) and 63/1,007 isocyanide components (6.26%), whereas
neither role contained an adjacent branch pair in the 82,264-product
selection-visible reference. These findings make the frozen step-1,000 model
ineligible for prospective candidate lock. They do not reopen the completed
model-selection decision: step 1,000 remains the immutable product-plus-L1
baseline while a separately versioned challenger is evaluated against the same
structural and open-endedness gates.

This is a descriptive, nonselecting structural audit. It excludes heldout
component families and provides no evidence about route closure, synthesis
feasibility, formulation or biological activity.

Evidence:
`results/phase1/ugi_tail_chemotype_audit_v1.json`
(`c781761c…071e`) under config
`configs/model/phase1_ugi_tail_chemotype_audit_v1.json`
(`7953bbfe…c011`).

## 7. Results that remain pending

| Pending result | Evidence status | What must happen before a claim is allowed |
|---|---|---|
| Candidate panel lock | Proposed pending | Apply the frozen applicability proposal, terminal support policy, conservative potency ranking, diversity policy and proposal-augmented route search to matched candidate pools before locking the prospective panel. |
| Generated and prospective route dossiers | Proposed pending | Extend the completed original-library dossiers to generated beyond-catalog candidates and the locked prospective panel, with forward verification, evidence grade, availability and uncertainty for every branch. |
| Prospective route selection | Proposed pending | Run proposal-augmented complete-product routing on an arm-balanced shortlist before panel lock and preserve the disposition of every routed and attempted candidate. Mid-trajectory synthesis guidance has already been tested and was not promoted. |
| Prospective biological validation | Proposed pending | Test an arm-balanced panel selected by the promoted applicability proposal and the identical conservative terminal potency-ranking policy, with measured formulation and biological outcomes. Potency trajectory tilting has already been tested and was not promoted. |
| Prospective experiment | Proposed pending | Lock candidates and report every synthesis attempt through product identity, conversion, isolated yield, purity, precursor failures, LNP formulation and function. |

## 8. Contradictions and stale records

1. **Phase 1 manifest.**
   `results/phase1/product_prelaunch_audit.json` embeds the historical manifest
   SHA `0ffe68b3…2dad`. The current manifest is
   `data/splits/phase1/manifest.json` with SHA `4ac82852…975`, which matches the
   current data contract. Use the latter.

2. **Size-only operational status.**
   The preflight (`3c94c54c…68791`) says awaiting approval and the launch
   receipt (`ca3da5e4…55c7`) says running. Both are valid historical records.
   The completed result (`2627d842…a16c`) supersedes them for training status,
   and the final v3 selector (`fe90fc84…3be5`) supersedes the earlier statement
   that matched generation evaluation was pending.

3. **Full-morphology checkpoint path alias.**
   Some intermediate artifacts cite `..._full_interim/checkpoint_step_2000.pt`.
   The canonical path is now `..._full/checkpoint_step_2000.pt`. Both identify
   the same checkpoint by SHA `e1a0092a…d21a`.

4. **Measured-product semantics.**
   1,200 is the nominal AGILE measurement ledger including 100 mixtures;
   1,100 is the curated single-structure graph set.

5. **Documentation lag.**
   `manuscript/FORGE_EVIDENCE_LEDGER.md` predates the frozen production oracle
   and current generator gates. `docs/DATA_PROVENANCE.md`,
   `docs/M0_06_DEFOG_FEASIBILITY.md` and
   `docs/M0_09_L2_SUPERVISION_INVENTORY.md` preserve earlier milestone counts.
   They should not override the current hash-pinned results.

## 9. Consistency verification

Targeted arithmetic, partition, reconstruction and hash checks pass:

- 15,433 - 204 = 15,229;
- every constitutional split sums to 15,229;
- 264 + 107 + 53 = 424 admitted components;
- 24 + 62 + 9 = 95 current components;
- 95 + 329 = 424;
- 483,032 + 349,013 + 654,370 = 1,486,415;
- 12,386 + 1,485,802 - 11,773 = 1,486,415;
- 66,464 + 15,800 + 30,122 = 112,386;
- 12,386 + 100,000 = 112,386;
- 12,276 + 1,100 - 990 = 12,386;
- every held-component and unconditional novelty partition sums to its valid
  product denominator;
- every valid product in those four gates has three inverse-reconstructed
  components and exact Ugi forward reconstruction;
- the current Phase 1 manifest hash matches the data contract;
- independent route/procurement composition reproduces 23/24 exact-source,
  47/47 family-projected, 1,734/12,276 terminal-or-exact and 8,432/12,276
  terminal/exact/family-projected counts;
- the 424-component readiness partition sums exactly, with 41 route-complete
  components split as 17 heads, 18 aldehydes and 6 isocyanides;
- the completed size-only status and all five snapshot hashes match its result;
- all 16 path/hash identities in the frozen production-generator manifest
  authenticate locally;
- the v3 selection reference contains exactly 66,464 training plus 15,800
  calibration products and excludes all 30,122 heldout products;
- the selected fresh sample reconciles as 1,007 valid plus 17 invalid attempts,
  with 847/1,024 usable open-ended successes and 39/1,007 exact
  selection-visible products;
- the corrected motif audit uses the same 1,007 valid-product denominator and
  82,264-product reference while skipping every heldout row.
- the descriptive held-family stress test reconciles 511 valid plus one invalid
  attempt, with exact inverse, forward and handle checks for all valid products;
- the post-selection provenance counts sum to 3,021 component occurrences and
  1,007 products, and the three static route-diagnostic strata also sum to
  1,007;
- the tail-chemotype audit uses 1,007 generated occurrences per tail-bearing
  role and 82,264 selection-visible occurrences per role; the generated
  aldehyde counts are 19 ester-bearing, zero carbon-carbon-double-bond and 27
  adjacent-branch occurrences, while generated isocyanides contain 63
  adjacent-branch occurrences and both reference roles contain zero adjacent
  branch pairs.

The machine-readable sibling contains complete unshortened SHA-256 values,
exact denominators, safe wording, limitations, pending results and stale-record
resolutions.

---

# Addendum, 6 August 2026: route construction, availability and makeability

The sections above were locked on 4 August 2026 and predate the work recorded
here. Where a routing statement above conflicts with this addendum, this
addendum is later and governs. No earlier artifact was rewritten.

## 10. Route construction over the route-blinded shortlist

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| One identical, arm-blind bounded hybrid L1/L2/L3 cascade was applied to all 256 v2 shortlist products, with a single content-addressed planner cache keyed only by canonical component target and frozen planner context | Computed | `results/phase1/ugi_bounded_hybrid_route_cascade_v1/result.json` sha256 `3066e306e3ab77ded09fbb8d76c9d0aab46540e8fa689063b896e17c8ed0eb20`; status `bounded_hybrid_route_cascade_complete` | Report as the route-blinded measurement; state that arm, cohort, authority tier and potency never entered route search |
| 49 components appear under more than one generation arm and each produced exactly one cache key and one route outcome | Computed | same artifact, `cache` block | Use as the arm-blindness demonstration |
| 256 of 256 products passed independent exact constitutional Ugi L1 forward reconstruction | Computed | same artifact, `summary.candidates` | Report as the L1 gate; never call it synthesis evidence |
| Under exact indexed evidence 19 of 256 products closed, and 121 of 122 unresolved components were never expanded because they lie outside the frozen evidence index | Computed | same artifact, `interpretation.unresolved_never_expanded` | Always report the never-expanded split beside the closure rate; the metric measures index coverage, not synthesizability |
| One candidate was excluded by the declared graph-support contract on atom state (Si, 0, False, 2) and was retained in the denominator rather than dropped | Computed | same artifact, `summary.route_assessment_not_admitted_candidates` | Report as a preserved failure; the gate was not relaxed |
| Neither proposal engine closed any route under the proposal-only contract | Computed | same artifact | Report as designed behaviour, not as engine failure |

## 11. Bounded planner reach and its limits

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| At 60 seconds and 300 iterations per component, applied identically, a bounded multistep planner connected 60 of 120 unresolved components to a frozen public catalogue snapshot with zero execution failures | Computed | `results/phase1/ugi_planner_reach_sweep_v1/result.json` sha256 `666300eb5a5c5af65425a78fb159200feebec4f8039dd63f4dab8d52313cce7c` | Report as reachability under a declared budget; not route evidence and not procurement |
| Cross-engine agreement between Graph2Edits and AiZynthFinder fired on 118 of 120 components (98.3%) and has no discriminative power | Computed null | `results/phase1/ugi_planner_reach_sweep_v1/` with `results/phase1/ugi_bounded_hybrid_route_cascade_v1/graph2edits_proposals.jsonl.gz` | Report as a rejected verification signal; both engines are USPTO-trained and default to the same generic interconversions |
| Forward reconstruction using the engine's own retro template passed 1,183 of 1,186 proposals (99.75%) and is circular for a template-based engine | Computed null | `results/phase1/ugi_engine_template_verification_v1/result.json` sha256 `02e0b55d46acbbb6ecbf73cdc4c0405846af25b0e3a4fefbb1e50581e55d9282` | Report as a rejected verification signal |
| Adjudicating planner proposals against the eleven local forward-resolver transforms verified only 15 of 1,187 proposals, all isocyanides | Computed | `results/phase1/ugi_planner_proposal_verification_v1/result.json` sha256 `9c862608833be22b33dbec76505c27b80dd459cb0ba456e0d8413513736b28f4` | Report as evidence that local transform coverage, not chemistry, limited the earlier adjudication |

## 12. Online availability replaces the hand-curated terminal ledger

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| Availability for 1,553 structures was read from PubChem vendor registrations, of which 477 have listed vendors, with a recorded access time and a declared 30-day expiry | Computed | `results/phase1/ugi_online_procurement_snapshot_v1/result.json` sha256 `d02b7dee82ee5009f60819b7c09533b39b4b6b4d65481f83a22f38f3d524b95c`; accessed 2026-08-06, expires 2026-09-05 | Report as a dated screening snapshot; a vendor listing is not a quote, stock level, lead time or purity specification |
| Local knowledge comprises one qualified L1 transform, four upstream reactions, eleven forward-resolver transforms and a 60-item terminal-material ledger, against roughly 48,000 templates inside the proposal engines | Computed | `data/vendor/qualified_reactions_v1.json`; `configs/route/graph2edits_l2_forward_resolver_v1.json`; `data/source_cache/aizynthfinder_public_v4_4_1/uspto_templates.csv.gz` | Use to justify why availability moved off the local ledger |

## 13. Published routes applied deliberately and forward-verified

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| The AGILE Tail A route, run forward through the repository's qualified esterification and primary-alcohol-oxidation transforms, reproduces 161 of 163 in-domain aldehydes exactly; 142 have both starting materials with listed vendors and every alpha,omega-diol is available | Computed | `results/phase1/ugi_tail_a_disconnection_v1/result.json` sha256 `f4de2516566adffec9c1508e8a927b45da39f353d56711269a30ba05066acfc4` | Report as deliberate application of published chemistry with independent forward verification; forward reproduction does not verify substrate scope |
| The isocyanide route, run forward through the qualified amine-formylation and formamide-dehydration transforms, reproduces 11 of 11 unresolved isocyanides; 7 have a precursor amine with listed vendors | Computed | `results/phase1/ugi_isocyanide_route_v1/result.json` sha256 `eef4326459d059eaca6b5f20f3a53f2036baf0e8722c7bcdd11129a6c088bea8` | As above |
| Generic retrosynthesis models underperform on lipid-scale aliphatic chemistry; domain-qualified transforms close the gap | Inferred | Sections 11 and 13 read together | Report as a methodological finding, supported by the contrast between planner reach and deliberate application |

## 14. Makeability across the applicability-supported population

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| The oracle-scored population is 2,590 unique products resolving to 201 unique components, a combinatorial space of 56,235 role triples | Computed | `results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz`; `results/phase1/ugi_indomain_makeability_v1/result.json` sha256 `d491286d25016f6bd68232d8f25263d080b9abfabd53171f9862f09bd8d8da06` | Report the component economy as a design property of the Ugi adapter |
| 2,517 of 2,590 in-domain products (97.2%) have a complete route to vendor-listed starting materials for every component; 73 are route-only and none is blocked | Computed | `results/phase1/ugi_indomain_makeability_v1/result.json` | Report as the headline routing result; state that it combines qualified transforms, learned proposals, independent forward verification and a dated availability snapshot |
| Of 201 components, 26 are directly purchasable, 151 route to purchasable material and 24 have a route with at least one unconfirmed material | Computed | same artifact, `summary.component_verdicts` | Report by role; amine heads are largely commodity, tail components are constructed |
| Makeability filtering is non-distorting: broad and support arm retention 98.3% and 96.4%, conservative-high potency 41.6% to 41.6%, `qualified_role_holdout` 12.5% to 12.4% | Computed | same artifact with `indomain_makeability_ledger.jsonl.gz` | Report as the bias audit; note that before the published routes were applied the same filter reduced `qualified_role_holdout` to 0.1% |
| 19 aldehydes require fatty acids that are positional isomers of natural chains at non-natural unsaturation positions and are genuinely unavailable | Computed | `results/phase1/ugi_tail_a_disconnection_v1/tail_a_disconnection_ledger.jsonl.gz` | Report as the one residual chemical constraint |

## 15. Branching: makeable but weakly scorable

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| The branch-conditioned lane produced 3,462 realized carbon-branched products from 4,096 attempts, of which 65 were all-view interpolative and 39 unique products were oracle-scored, an in-domain yield near 1.9% | Computed | `results/phase1/ugi_branch_exploration_applicability_v1/result.json` | Report that applicability, not generation or synthesis, limits branched candidates |
| The main 32,768-draw production pool contains no carbon-branched component, because it predates the branch-scheduler correction | Computed | `results/phase1/ugi_indomain_makeability_v1/indomain_makeability_ledger.jsonl.gz` | Report explicitly; branched candidates exist only in the separate lane |
| All four branched aldehydes forward-verify by Tail A from 4-methylnonanoic acid or its homologue plus commodity diols | Computed | `results/phase1/ugi_tail_a_disconnection_v1/` | Report that synthesis is not the branching constraint |
| The highest-scoring in-domain candidate overall is branched (LCB90 8.90, mean 14.20, first of 2,478) but sits in the weakest authority tier, while the five `qualified_role_holdout` branched candidates score LCB90 0.53 to 1.63 and none is conservative-high | Computed | `results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz` | Report the anticorrelation; do not present the top branched score as a validated prediction |

## 16. Candidate proposal

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| A stratified 20-candidate proposal spans seven `qualified_role_holdout`, three branched, seven chemotype-diversity and three deliberately low-ranked controls, over 20 distinct chemotypes, 11 amines, 12 aldehydes and 7 isocyanides, with LCB90 from -6.45 to 8.90 | Computed | `results/phase1/ugi_stratified_panel_proposal_v1/result.json` sha256 `366c090ece8292d3fc9a605f704fe7dc6780aaf1c7151e62574fe649f8b06e11` | Report as a proposal for chemist review; it is not a panel lock and no preregistration exists |
| At approximately eight evaluable compounds the broad-versus-support causal comparison is not powered | Inferred | Panel-size guidance in the lossless handoff and the proposal artifact | State that the prospective study is a feasibility and hit-rate design by explicit decision |

## 17. Claims that remain unavailable

| Claim | Status |
|---|---|
| Any generated route succeeds experimentally | Not available; no synthesis has been attempted |
| Any generated lipid forms an acceptable LNP | Not available |
| Any generated lipid delivers mRNA in vitro or in vivo | Not available |
| Substrate scope of the applied transforms on these specific substrates | Not verified; forward reproduction is not scope verification |
| Current purchasability after 2026-09-05 | Snapshot expired; must be refreshed |
