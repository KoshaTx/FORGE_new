# Decision Log

Append-only. Dated. Every M0 task adds an entry — **including negative results**, which are first-class
deliverables here.

---

## 2026-07-28 — Ugi variant resolved: 3CR, acid-free (M0-01)

**Decision.** The assembly chemistry is **Ugi three-component, acid-free**. Config
`configs/assembly/ugi_variant.yaml` → `variant: ugi_3cr_acid_free`.

**Evidence.**

- Registry `atom_mapped_reaction_smarts`:
  `[NX3;H2,H1:1].[CX3H1:2]=[OX1].[C;-1,+0;X1:3]#[N;+1,+0;X2:4]>>[N:1][CH1:2][C+0:3](=O)[NH1+0:4]`
  — exactly three reactant components, no carboxylic acid.
- Registry `selectivity_policy`: "Acid-free Ugi three-component condensation forming an alpha-amino
  amide."
- Product architecture R¹-NH-CH(R²)-C(=O)-NH-R⁴ is an α-amino amide. Classic Ugi-4CR yields an
  α-**acyl**amino amide.
- AGILE table has exactly `A_smiles`/`B_smiles`/`C_smiles`. Row 0:
  `CN(C)CCN` + `O=C(CCCCCCCC)OCCCCCC=O` + `CCCCCCCCCCCC[N+]#[C-]`
  → `CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(C)C`, which carries a free secondary amine
  (`NCCN(C)C`), not an N-acyl group.
- Qualified by exact reconstruction of 1,200/1,200 rows in the nominal AGILE
  table. M0-07 separately determines which rows define single-compound
  supervision.

**Consequence (load-bearing).** The ester in AGILE lipids is carried by the **aldehyde** component
(B = 6-oxohexyl octanoate), not by a fourth acid component. This is why AGILE's Tail A route exists
(fatty acid + diol → EDC coupling → DMP oxidation): the ester is built into the aldehyde *before* the
Ugi step. **Ester construction is an L2 problem, not an L1 problem** — which is the cleanest single
justification for the L1/L2/L3 decomposition.

**Open flag.** The PI has referred to the lab chemistry as "Ugi-4." If the wet lab genuinely runs
classic Ugi-4CR, that is a different lipid architecture and the 1,200 AGILE labels do **not** supervise
it. That would be a scope change, not a config change. Needs human confirmation; does not block M0.

---

## 2026-07-28 — Corpus anchoring defect measured (informs M0-04)

**Finding.** The 464,265-product enumeration is **not anchored in LNPDB**, contrary to design intent.

| Diagnostic | Value |
|---|---|
| Block-pool provenance | `programmatic_rational` 390/490 (79.6%), `curated_literature` 75, `agile_measured` 25 |
| Block-handle counts | nine handles at exactly 39 each — a `chain_length × n_branches × unsaturation` grid |
| R0 ∩ R1 exact canonical | 1,308 / 15,433 = 8.5% |
| …of which AGILE-1200 | 1,200 |
| **Non-AGILE real lipids recovered** | **108 / 14,233 = 0.76%** |

The enumeration reproduces the one library whose components were seeded into the pool and little else.
R0 entered only as an IPF reweighting target, which matches marginals but cannot conjure structures the
block pool cannot reach.

**Consequences.** (1) R1 renamed to **"reaction-enumerated support"**; "route-certified" is retired.
(2) The R1′ rebuild must harvest blocks from `R0_train` only, split before decomposition, to avoid
circularity. (3) 0.76% is the baseline M0-04 must beat, and must be reproduced as a control.

**Viability.** Qualified-family product motifs are present in real lipids: amide 41.3%, aza-Michael
36.3%, epoxide 6.9%, iphos 3.7%. The `reductive_amination` figure of 99.8% is a **degenerate motif**
(essentially any C–N bond) and is not reportable — it is the worked example motivating the M0-05
precision audit.

---

## 2026-07-28 — Support tiers redefined (M0-02 prerequisite)

**Decision.** Replace I/F/B/N with **E0/E1/E2/E3**. The prior Tier F was logically inconsistent: under
exhaustive enumeration, a product built from pool blocks by qualified transforms is already in R1, so F
is empty but for stereochemical/protonation/filter artifacts.

**E2 (known assembly, generated precursor) is the primary open-endedness claim. E3 (new assembly
family) is exploratory and the paper does not depend on it.**

---

## 2026-07-28 — L2 supervision is too thin for a standalone lipid route model (M0-09)

**Decision.** Use a **hybrid L2 architecture: generic neural proposal plus deterministic,
lipid-specific template search**. Do not build a standalone learned lipid-specific route model from
the currently available corpus.

**Measured inventory.**

- Generic reaction pretraining remains worthwhile: USPTO-MIT contains 479,035 reaction records
  (478,998 RDKit-parseable; 37 invalid), while USPTO-50K provides 50,016 reactions across 10 labeled
  classes for controlled benchmarking.
- AGILE Supplementary Note 1 contributes 48 upstream reaction instances, 48 unique intermediate or
  terminal products, and 25 terminal tail scaffolds. Terminal depths are 2 at depth 1 and 23 at depth 2.
- Seven internal RM protocols contribute seven one-step ester products: four acrylates and three
  propiolates. All seven yield/appearance/LCMS fields are blank.
- Combined lipid-specific coverage is 55 reaction instances and 32 terminal precursor scaffolds:
  ester 23, aldehyde 18, isocyanide 7, acrylate 4, carbonate 0, heterocycle formation 0.
- No explicit failed L2 synthesis was found. Missing RM outcomes are not failures.

**Corpus boundaries.** The AGILE 12,276 virtual candidates, AGILE 1,200 measured products, R0 15,433
real lipids, and R1 464,265 enumerated products remain useful for structural pretraining, oracle
training, block prioritization, or product-prior ablations. They do not provide observed upstream
reaction outcomes and therefore do not count as L2 supervision.

**R1 correction retained.** R1 is not discarded, but it cannot dominate by size or train L2. The R1′
audit must split R0 before extraction, harvest LNPDB/LiON/other components from `R0_train` only, audit
decomposition precision, and beat the 0.76% non-AGILE held-out recovery control without leakage.

**Immediate data priority.** Standardize internal attempt logging, including failures, and curate
carbonate, acrylate, heterocycle, and failed routes from patents/SI. Full counts, definitions, source
hashes, and QA flags are in `docs/M0_09_L2_SUPERVISION_INVENTORY.md` and
`results/m0_09/result.json`.

---

## 2026-07-28: Preserve every LNPDB lipid in a calibrated route-awareness map (M0-09)

**Decision.** Map every unique LNPDB lipid to its source-linked components and zero or more reusable
route records. Preserve unknown routes, source discrepancies, forward verification, procurement, and
execution closure as separate states. The whole-molecule product prior remains unchanged.

**Computed result.** The ledger covers all 12,837 unique canonical lipids represented by 19,797 raw
LNPDB records and all 726 normalized component annotations. AGILE contributes 16 exact two-step route
records: 11 of 12 aldehyde-esters and all 5 isocyanides. The publication-level purchase statement for
20 amines remains a historical vendor claim, not procurement closure.

The hash-pinned AGILE measured file resolves source-label omissions in the raw LNPDB export without
rewriting the raw values. Five component labels have missing or incomplete alkene stereochemistry,
and C2 has a conflicting formamide-like raw string. B5 remains unresolved because the measured
Z dec-2-enoate conflicts with the supplementary information's chain length and stereochemical
evidence.

Across the 1,200 AGILE products, 1,100 have source evidence for all three inputs and 100 contain B5.
Across all LNPDB lipids, 1,177 have source evidence for all annotated components, 2,134 have partial
evidence, 9,426 are unassessed, and 100 contain the explicit discrepancy. These are route-awareness
counts. There are zero forward-verified routes, zero procurement-closed components, and zero
execution-closed lipids.

**Scope.** This is an inventory and retrieval substrate. It does not implement or train the L2 model.
Future route records must be split by publication and component family before learned evaluation.
Full methods, evidence states, and output hashes are in `docs/M0_09_ROUTE_AWARENESS_MAP.md` and
`results/m0_09/route_awareness_result.json`.

---

## 2026-07-28: R0 splits frozen before component extraction (M0-03)

**Decision.** Freeze four independent R0 split schemes now. M0-04 and every later task must load
`data/splits/m0_03/` and may extract components from `R0_train` only. They must not recompute groups or
move structures between folds.

**Computed result.** All 15,433 R0 structures occur exactly once in each scheme, and no connected group
spans folds:

| Scheme | R0_train | R0_cal | R0_heldout | Connected groups |
|---|---:|---:|---:|---:|
| Source study | 10,770 | 2,330 | 2,333 | 24 |
| Headgroup | 10,803 | 2,315 | 2,315 | 2,967 |
| Linker/scaffold | 10,803 | 2,315 | 2,315 | 2,006 |
| Component family | 12,313 | 1,920 | 1,200 | 3 |

Groups reuse `leakage_group_id`, `study_split_groups_json`,
`component_holdout_groups_json`, and `reaction_family_holdout_groups`. Multi-annotated structures are
collapsed into connected components before deterministic whole-group assignment. The linker/scaffold
key also preserves compatibility with the heteroatom-connector grouping used by `splits_v1`.

**Why the inherited fold labels were not copied directly.** Within R0, `held_study` has no validation
fold and `held_reaction_family` assigns all 15,433 structures to training. Under the stricter
source-annotation groups required here, the inherited folds also split 92 head groups, 11
linker/scaffold groups, and one source-study connected group. The M0-03 folds close those leaks.

**Limitations.** Component-family holdout has only three observed signatures, so its 79.8/12.4/7.8%
partition cannot approximate 70/15/15 and remains a secondary stress test. RDKit could not parse 230
LNPDB head annotation occurrences. Their raw strings were retained as grouping tokens, not discarded or
treated as valid molecules.

---

## 2026-07-28: R1-prime improves component holdouts and maps registry boundaries (M0-04)

**Decision.** Retain R0 as the real-lipid anchor. Keep the existing reaction-enumerated corpus as
realism-weighted auxiliary structural data, and keep R1-prime diagnostic until M0-05 reports
decomposition precision. Neither corpus is route-certified, neither supplies L2 supervision, and
neither defines the whole-graph discrete flow's action space.

**Computed result.** The existing non-AGILE control reproduced exactly at 108/14,233 = 0.759%.
Train-derived R1-prime recovered 213/2,135 = 9.98% of non-AGILE headgroup-held-out structures and
435/2,307 = 18.86% of non-AGILE linker/scaffold-held-out structures. It recovered 0/2,333
source-study-held-out structures. The component-family stress test contained 1,200 AGILE structures
and recovered 200/1,200 = 16.67%; it had no non-AGILE denominator.

Median nearest ECFP4 distance among exact misses was 0.514 for source-study, 0.268 for headgroup, 0.259
for linker/scaffold, and 0.000 for component-family holdout. The last value means that at least half of
the exact misses shared a folded ECFP4 bit vector with a reachable molecule; fingerprint identity did
not make them exact molecular matches.

**Corpus scale and boundary.** Exhaustive train-only enumeration produced 2.99 million to 5.74 million
unique products per scheme. Reductive amination and amide coupling contributed 73.7% to 78.8% and
19.5% to 24.7% of reactant combinations, respectively; Ugi-3CR contributed 0.15% to 0.30%. These are
enumeration-composition counts, not reaction-motif hit rates or precision estimates.

**Interpretation correction.** The 0/2,333 source-study result is a cross-platform
reaction-registry coverage and leakage audit, not a generalization score for FORGE. It confirms that
the frozen transforms over training-derived components did not recreate exact products from four
withheld platforms. LM_2019 / LM_3CR (1,080 structures) uses an isocyanide-mediated
amine-ketone-isocyanide 3-CR, JL_2024 (623) uses A3 coupling, and XH_2025 (500) uses amine/thiol
addition to dialkyl maleates. None of those reported final transforms is currently implemented.
SX_2025 (130) still requires a source-route audit.

**Per-platform addendum.** Exact reverse and forward reconstruction found an alternative registered
disconnection for 900 LM_2019 / LM_3CR products, 0 JL_2024 products, 250 XH_2025 products, and 81
SX_2025 products. These graph disconnections are not assigned as reported publication routes. Across
all 2,333 structures, zero products had any exact decomposition whose complete set of role-typed
components existed in the source-study training pool. No held-out component was added to that pool.
The result resolves the exact-recovery zero as a combination of platform-level component withholding
and final-reaction-registry boundaries, while leaving motif transfer as a separate structural and L2
question.

**Consequence.** The result is split-dependent and does not support a single aggregate coverage claim.
Proceed with whole-molecule discrete-flow generation, calibrated applicability-gated AGILE tilting, and
the M0-09 hybrid L2 route design. Evaluate broad cross-platform transfer through structural
whole-lipid benchmarks. Evaluate synthesis grounding through Ugi-specific component holdouts and
prospective E2 candidates. Do not generate products by sampling the audited component cross-product.
Full methods, uncertainty, family counts, source hashes, and limitations are in
`docs/M0_04_R1_PRIME_AUDIT.md`, `results/m0_04/result.json`, and
`results/m0_04/source_platform_registry_audit.json`.

---

## 2026-07-28: LNPDB route-source curation is finite and row-linked (M0-09 addendum)

**Decision.** Use the immutable raw LNPDB export, not canonical-structure-aggregated R0 provenance, as
the source of exact publication-to-component relationships. Retain R0 as the realism and split anchor.

**Computed inventory.** The 19,797 raw LNPDB records reproduce all 12,837 LNPDB canonical structures
in R0 and all 19,797 source record identifiers exactly. They resolve to 42 publication families and one
commercial catalog source. Twenty-six publications carry PMC identifiers; 16 require publisher or
author acquisition. Across roles, the source contains 726 normalized entries: 706 parsed structures
and 20 quarantined invalid source strings.

**QA finding.** LNPDB reports PMID `29786478` for rows whose DOI is
`10.1016/j.bioactmat.2024.05.012`. That PMID is unrelated. The DOI resolves to PMID `38779592`. The
ledger preserves the reported identifier and records the correction explicitly.

**Boundary.** Every normalized component has a row-level source, but no new component is declared
route-complete or purchasable. Route and procurement fields begin unknown. The M0-04 exact-recovery
percentages remain structural diagnostics and will not be presented as synthesis coverage.

**Acceptance rule.** A component closes only through time-stamped procurement evidence or a
forward-verified route to procurement-verified or internally stocked leaves. Whole-lipid execution
requires L1 closure plus recursive L2 and L3 closure for every component.

---

## 2026-07-28: Count upstream component synthesis separately from lipid assembly (M0-09 addendum)

**Decision.** Review LNPDB source chemistry at the subcomponent level and enforce L1 versus L2 in the
saved schema. A general procedure that attaches a purchased head or tail is L1 assembly evidence. It
does not count as a route for making that head or tail.

**First reviewed source.** Li et al., PMID `36997680`, covers 81 normalized LNPDB components: 70
heads, one linker, and ten tails. Its supplementary information contains one exact five-step L2 route
for a shared activated ricinoleic acrylate precursor. It contains zero upstream routes for the ten
alcohol tails and zero upstream routes for the 72 source-reported amine heads. The source instead
treats those building blocks as commercial through group-level historical statements without exact
catalog identifiers.

General Procedure A supplies ten linker-tail subassembly examples with isolated yields of 61% to 84%.
General Procedure B supplies the final aza-Michael assembly. Both remain L1 records. They do not
inflate the L2 count.

**Consequence.** LNPDB may still contain substantial upstream head, tail, and linker chemistry across
its other papers and cited prior art. One source cannot establish sufficiency. Continue source review,
prioritizing explicit component syntheses and validated general-procedure scope. Use broader reaction
data and current procurement evidence to close components whose upstream route is absent from their
LNPDB source.

---

## 2026-07-28: Focus the experimental paper on AGILE-type Ugi 3-CR chemistry

**Decision.** The current FORGE paper will make its primary generative, route-completion, tilting, and
prospective synthesis claims within the AGILE-type amine–aldehyde–isocyanide Ugi 3-CR family.
General lipid chemistry may
support upstream precursor planning, but multi-assembly lipid generation is reserved for a separate
computational extension.

**Current Ugi-3 capability.** AGILE Supplementary Note 1 reports 25 terminal tail routes: 18
aldehydes and seven isocyanides, comprising 48 upstream reaction instances. Twenty-four products
have structure-resolved route records; B5 remains the sole unresolved source-to-measured identity
conflict. The source reports product-specific proton NMR but no isolated yields for these upstream
steps.

The aldehyde routes cover direct fatty aldehydes, four-carbon and six-carbon spacers, ester-bearing
degradable motifs, C8 through C18 hydrophobes, branching, and several unsaturation patterns. The
isocyanide routes are much thinner: mostly straight C11 through C18 chains and one cis-unsaturated
C18 example.

**Consequence.** Prioritize Ugi-compatible isocyanide diversity, then aldehyde spacer and
degradability coverage. Non-Ugi LNPDB chemistry counts only when it supplies an upstream route or
reusable transformation for an amine, aldehyde, isocyanide, or their procurement leaves. It does not
expand the current paper's final-assembly claim.

---

## 2026-07-28: Measure a bounded Ugi-3 precursor space, not universal tail synthesis

**Decision.** Seek complete transformation-family coverage within a declared,
bounded Ugi-3 precursor envelope. Do not seek one literature route for every
individual tail, and do not claim that every chemically imaginable tail is
covered.

**Measurement rule.** Keep source-extracted supervision separate from proposed
reaction-family applications. After the reaction basis and applicability
boundaries are frozen, build a reproducible virtual-space census over versioned
precursor catalogs. Canonicalize and deduplicate products, enforce Ugi-handle
and molecular-support constraints, report filter attrition, and count distinct
amine heads, aldehydes, isocyanides, compatible triples, and whole lipids by
motif and evidence state.

**Consequence.** A broadly applicable two-step family may support many proposed
tails, but only its reported examples count as route labels. A proposed member
becomes route-complete only after deterministic forward checks and recursive L3
closure, and it becomes experimentally supported only after a prospective
outcome. This preserves the open-ended generation claim without reducing FORGE
to building-block enumeration or overstating synthesis coverage.

---

## 2026-07-29: Separate exact upstream routes from reaction-family capability

**Decision.** Do not rank unlike evidence on one ladder. Store transformation
evidence, exact-component observation, route closure, operational availability,
and prospective outcome as orthogonal axes. Keep forward verification as a
separate status. An exact literature route does not imply current procurement
or complete route closure, and cross-assembly observation does not imply direct
Ugi compatibility.

**Computed audit.** The rational pool contains 39 isocyanides. Five have exact
structure-resolved AGILE routes and 34 remain bounded reaction-family
candidates. The exact examples cover four straight saturated members and one
straight monounsaturated member. No branched or diunsaturated candidate has an
exact structure-resolved AGILE upstream route. None of the 39 is route-complete
and none has a prospective outcome.

AGILE contributes 20 exact amine heads, but zero have extracted upstream routes
or current item-level procurement closure. The Miao 2019 cross-assembly source
adds ten exact isocyanides and 12 exact heads, with four heads exactly
overlapping AGILE. These observations broaden component evidence without being
misclassified as direct Ugi-3 validation.

**Chemistry basis.** Independent exact component-route evidence and two general
isocyanide-dehydration studies support a primary-amine to formamide to
isocyanide route family for bounded aliphatic exploration. Reported substrate,
solubility, steric, functional-group, and aromatic limitations remain part of
the applicability boundary. A TFA versus TEA discrepancy in one independent
supplement is preserved as an unresolved QA flag.

**Consequence.** The 175,032 raw product of 39 isocyanides, 51 aldehydes, and
88 amines is an upper bound on precursor triples only. It is not a molecule
count or synthesis-coverage claim. Later route planning and tilted generation
must score whole generated lipids against the separate evidence axes and may
promote a candidate only through explicit forward, procurement, and prospective
checks.

---

## 2026-07-29: Correct Ugi terminology and adopt synthesis-program framing

**Terminology correction.** The active configuration is
`variant: ugi_3cr_agile`. The reaction uses amine, aldehyde, and isocyanide
reactant components and has no carboxylic-acid reactant component. The earlier
shorthand `acid-free` is retired from active project terminology because the
AGILE procedure uses an acidic phosphorus catalyst. Legacy source and registry
text is preserved as source evidence, not repeated as manuscript terminology.

**Framework decision.** FORGE is a synthesis-grounded joint generator of
complete ionizable-lipid graphs and complete synthesis programs. A synthesis
program is a route tree composed of reaction-family-specific transformations;
a lipid does not belong to one unique reaction family. The first deep
prospective instantiation uses AGILE-type Ugi 3-CR final assembly, while its
upstream precursor tree may use esterification, oxidation, formylation,
dehydration, and other reaction families.

**Jointness boundary.** The target is joint at the system and objective level.
Dense product and final-assembly supervision may support joint molecule and L1
modeling. Sparse L2 supervision remains recursive and modular unless the M0-09
census demonstrates enough complete, diverse route trees for tighter shared
modeling. Broad structure-only lipid data train the molecular prior with route
losses masked rather than fabricated.

**Claim boundary.** The architecture can accept additional synthesis programs,
but each requires corresponding mappings, scope evidence, failures, forward
verification, and prospective validation. The current paper does not claim
that FORGE has already learned synthesis across all ionizable-lipid chemistry.

---

## 2026-07-29: Audit every current Ugi aldehyde and amine head

**Decision.** Evaluate every one of the 51 aldehydes and 88 amine heads in the
frozen pool by exact source identity, literal registry handle policy, upstream
route evidence, and operational availability. Do not infer an exact route from
a related substrate, and do not convert a historical purchase statement into
current procurement closure.

**Computed result.** Seventeen AGILE aldehyde routes are structure-resolved.
Eleven targets occur exactly in the current aldehyde pool, six are additional
source products outside that pool, and B5 remains the only unresolved
source-to-measured identity discrepancy. All 51 candidates pass the registry's
aldehyde-handle policy, but none is computationally route-complete because
forward verification and terminal-leaf procurement remain open.

All 88 amine heads contain at least one N-H handle. The registry's literal raw
site-multiplicity filter passes 78 and rejects 10. One rejected structure is
A5, an experimentally measured AGILE head with three N-H matches, although the
registry allows only one or two. This is a policy-semantic conflict, not
evidence of synthetic incompatibility. The literal multiplicity filter is
therefore prohibited as a hard candidate gate until symmetry and
site-equivalence behavior are reconciled and the qualification is rerun.

Twenty-one heads have historical group-level vendor evidence through AGILE or
the Miao cross-assembly source. Zero have an extracted upstream route and zero
have current item-level procurement closure. The artifact emits a complete
88-head procurement queue rather than silently treating common amines as
available.

**Architecture consequence.** This negative result does not change the joint
whole-lipid and synthesis-program plan. It identifies two M0 prerequisites for
joint training and route-aware sampling: reconcile the A5 registry gate and
close or route the campaign-relevant heads with time-stamped evidence.

---

## 2026-07-29: Close current procurement for 19 of 20 AGILE heads

**Decision.** Treat an amine head as currently procurable only when a
time-stamped, US-region primary vendor page resolves the exact campaign
structure and free-base form, item identifier, purity, and an explicit stock or
shipping observation. An exact catalog listing that requires a cart check does
not close availability. Prices are excluded because they are volatile and do
not establish identity or availability.

**Computed result.** All 20 AGILE heads resolve by exact canonical identity to
the frozen pool with zero identity or form discrepancies. Nineteen have current
item-level vendor closure. A9 has an exact MilliporeSigma item, but its current
availability remains unresolved. The snapshot is valid for 30 days and the
builder rejects an expired snapshot.

Across the complete 88-head pool, 69 candidates still require procurement or
upstream-route resolution. No head has an extracted upstream synthesis route.
The A5 item is currently procurable, but its separate measured-library versus
registry multiplicity conflict remains unresolved.

**Consequence.** Procurement closure is recorded on the operational
availability axis only. It does not imply route-family compatibility, forward
verification, prospective synthesis success, or computationally complete
candidate routes. The result strengthens L3 coverage without changing the
joint whole-lipid and synthesis-program architecture or the M0-only execution
boundary.

---

## 2026-07-29: Qualify Ugi amine multiplicity by symmetry-distinct sites

**Decision.** Retain the vendored registry's allowed Ugi amine multiplicities
of one or two, but apply them to symmetry-distinct required-handle matches.
Retain the raw RDKit substructure-match count as a diagnostic. Do not add an
A5 identity exception. Deduplicate forward products, and require explicit
reacting-site selection whenever multiple unique products remain.

**Measured-library qualification.** The deterministic audit covers all 1,200
AGILE records, the complete 20 by 12 by 5 measured component cross-product.
The qualified transform reconstructs all 1,200 expected products and all 1,200
pass the symmetry-distinct amine-site gate. The literal raw-count gate would
reject 60 products, all using A5.

A5 has three raw N-H matches but one symmetry-distinct site. Its three raw
forward outcomes collapse to one unique product in every one of its 60 measured
combinations. A17 similarly has two raw matches and one unique product. A19 and
A20 each have two symmetry-distinct sites and two unique products, so their
reacting atom must remain explicit in a generated or decomposed route.

**Candidate-pool consequence.** The raw diagnostic passes 78 of 88 heads. The
qualified policy passes 85 and rejects three heads with three
symmetry-distinct sites. This changes the compatibility-filtered precursor
upper bound from 155,142 to 169,065 triples. Neither number is a count of
unique products, route-complete candidates, or synthesizable lipids.

**Architecture consequence.** The A5 gate is resolved without changing the
joint whole-lipid and synthesis-program framing. The result strengthens dense
L1 supervision and establishes an atom-site annotation requirement for
multi-product heads. It does not add L2 route supervision or authorize model
training.

---

## 2026-07-29: Reduce the AGILE virtual library to a recursive component queue

**Decision.** Use the 12,276 AGILE virtual structures for exact Ugi L1
decomposition and component prioritization, not as observed L2 route
supervision. Require exact forward reconstruction and preserve the reacting
amine atom. Deduplicate components before upstream route work.

**Computed result.** All 12,276 products have one exact qualified
decomposition and no product has multiple component tuples. The set is the
complete 22 by 62 by 9 Cartesian product and contains 93 unique components.
Exact-identity capability joins cover 18 of 22 heads, 18 of 62 aldehydes, and
7 of 9 isocyanides. Exact extracted source procedures cover 17 aldehydes and
all 7 matched isocyanides. Seventeen heads close as current procurement
terminals. The source procedures remain incomplete until forward verification
and terminal procurement close, so no aldehyde, isocyanide, or virtual product
is called route-complete.

**Consequence.** Recursive route work scales with 93 unique components rather
than 12,276 product rows. The largest immediate coverage gap is the
62-member aldehyde set, including 44 structures outside the current exact
capability join. This result strengthens the L1 side of the
hierarchically joint design but does not add observed L2 reactions, yields, or
failures and does not authorize model training.

**Recursive program result.** All 24 exact component procedures, covering 17
aldehydes and 7 isocyanides, are reproduced exactly at the structure level.
The same bounded transformations assign family-projected programs to another
45 aldehydes and 2 isocyanides. Across 71 structural programs, there are 65
unique proposed intermediates and 33 unique proposed leaves. The five
nonterminal heads remain a procurement-first queue. Family projection does not
establish exact substrate evidence, and no proposed leaf or route is called
closed.

**Terminal-queue result.** Deduplicating the 33 proposed leaves with the five
unresolved heads yields 38 exact structures: 14 fatty acids, 4 diols, 6
primary alcohols, 9 primary amines, and 5 amine heads. Twenty-three carry only
a historical source vendor claim. A time-stamped US review closes all 14 fatty
acids, all six primary alcohols, all four diols, eight exact free-base primary
amines, and three of the five initially unresolved heads, reducing the
unresolved terminal queue to three structures. High-purity oleylamine is
catalogued but lacks an accepted current US stock or shipping observation.
The exact 3-aminoquinuclidine free base and 1,1-dimethylhydrazine require
explicit operational closure or a complete route from an accepted salt.
Dependency incidence is retained to prioritize shared leaves, but it is not
reported as unique product coverage.

---

## 2026-07-29: Use a hierarchically joint model and treat exploration loss as a measured risk

**Decision.** Factor the joint molecule and synthesis-program distribution as
`p(x, b, R) = p_theta(x, b) q_phi(R | x, b)`. Generate the complete lipid graph
and its L1 decomposition jointly, then construct recursive L2 routes and close
L3 terminal materials with specialized modules. Feed the calibrated synthesis
value back into discrete-flow sampling. Generation followed by synthesis
assessment without feedback remains the post-hoc baseline.

**Rationale.** The current data are dense for lipid structures and Ugi
product-component pairs but sparse for exact complete upstream route trees. A
monolithic decoder would not add scientific rigor by itself and could entangle
the broad lipid prior with the narrow route-labeled distribution. That would
favor familiar components, short routes, and dominant reaction families,
weakening the open-endedness claim.

**Required evaluation.** Pre-register a synthesis-guidance-strength sweep and
report route closure, novelty, diversity, and E2 fraction together. Compare the
same product prior under zero guidance, post-hoc filtering, and route-guided
sampling at matched budgets. Guidance must not be selected by route closure
alone.

**Manuscript treatment.** Keep the editor-facing title and opening centered on
"Synthesis-grounded generative design of ionizable lipids." Explain the
hierarchical joint factorization in Figure 1, Results, and Methods. Do not make
"fully joint" or "hierarchically joint" a title-level claim.

---

## 2026-07-29: Transfer hydrophobic motifs across platforms without inheriting routes or labels

**Decision.** Add a versioned cross-platform hydrophobic-motif transfer inventory to M0-09. Use
non-Ugi lipid libraries to identify experimentally realized hydrophobic backbones, branching,
unsaturation, ester placement, and degradable motifs that may be converted into Ugi-compatible
aldehyde or isocyanide components. Do not attempt to reconstruct every source lipid through Ugi
chemistry.

**Required boundary.** Preserve the source attachment atoms and separate the transferable motif from
the platform-specific handle and central core. Every admitted transfer must name a mapped common
precursor, a proposed Ugi component realization, evidence grade, recursive route state, and terminal
status. Ambiguous motif clipping remains structure-only. Source-lipid activity is provenance, not a
label inherited by the converted component or new Ugi product.

**Rationale.** Tail motifs can recur across A3, maleate, Ugi, and other combinatorial platforms even
when their exact product graphs and reactant roles differ. This lets the broad whole-lipid corpus
inform Ugi E2 exploration without misclassifying non-Ugi products as Ugi route examples. The
inventory is proposed transfer space and component prioritization, not observed L2 supervision or a
claim of route closure.

**Source-reaction rule.** LNPDB identifies the source paper and motif, but the
primary article and supplement provide the chemistry evidence. Every transfer
record must link to the actual reported source-component reaction, including
reactants, conditions, yield, purification, outcome evidence, and locator.
The separate reaction proposed to install a Ugi aldehyde or isocyanide handle
requires its own evidence and closure state. A graph transform is not an
experimental reaction record.

**Durable reuse boundary.** FORGE's reusable synthesis-program architecture
consists of the route-tree schema, reaction records, reusable precursor
subroutes, evidence and availability states, forward checks, failure taxonomy,
and synthesis-value interface. A new final-assembly chemistry can reuse these
objects, but it must add its own adapter, reactant roles, scope evidence,
chemistry-specific supervision, and prospective validation. Infrastructure
generality does not imply zero-shot chemical support.

---

## 2026-07-29: Bound biological tilting by available oracle evidence

**Decision.** Keep component-held-out evaluation, calibrated uncertainty, an oracle-guidance-strength
sweep, independent representation checks, and prospective applicability bins. Do not require or
silently impute apparent pKa, particle size, polydispersity, encapsulation efficiency, or formulation
robustness across the structural corpus because those measured labels are not currently available for
every lipid.

**Implementation boundary.** Applicability uses evidence that actually exists: molecular
fingerprints, learned embeddings, component families, structural clusters, and
structure-computable descriptors. The uncertainty estimator and any conservative potency score must
be calibrated under the shifts the generator is expected to create. No particular lower-confidence
formula is assumed valid in advance.

**Prospective treatment.** Formulation-dependent properties remain measured downstream outcomes and
candidate-advancement criteria after synthesis. A transferred motif may have useful structural and
experimental provenance, but neither source-lipid potency nor formulation behavior transfers to the
new Ugi component or product.

---

## 2026-07-29: Authorize targeted tail-source expansion after one exact route closes

**Decision.** Continue cross-platform tail review through a frozen,
value-prioritized source queue. Do not expand into exhaustive LNPDB reaction
mining. Native Ugi closure remains the first priority, followed by
high-information transfer sources.

**Computed result.** The two-platform pilot contains 11 exact source
attachments from JL_2024 and XH_2025. Seven primary alcohols yield novel,
frozen-transform-compatible aldehyde structures, and four secondary alcohols
remain structure-only. One route now closes computationally under the declared
evidence policy. The exact JL_2024 B16 alcohol precursor is converted to
2-decyltetradecanal by a reported pyridinium-chlorochromate oxidation in 86%
isolated yield, and TCI America D5283 provides current US item-level
procurement closure for the exact alcohol.

**Claim boundary.** The closed record establishes an exact upstream
alcohol-to-aldehyde route, accepted terminal availability, and deterministic
compatibility with the frozen Ugi transform. It does not establish prospective
Ugi conversion, final-lipid isolation, formulation, biological activity, or
oracle applicability. The six other aldehyde proposals remain incomplete.

**Next source order.** Audit JC_2023 next because it is native to the
AGILE-type Ugi 3-CR envelope and contains exact branched-aldehyde and
isocyanide routes. Audit LX_2024 next for cross-platform aldehyde transfer.
Continue only while missing encoded route knowledge materially constrains the
declared biologically relevant Ugi space.

---

## 2026-07-29: Add JC_2023 as independent native-Ugi L2 evidence

**Decision.** Treat JC_2023 as an independent source-resolved Ugi evidence
package. Count its exact product preparations as new route observations, but
do not count repeated component structures as new chemical identities or infer
terminal procurement closure.

**Computed result.** The source adds four structure-resolved L2 route instances
and eight L2 reaction instances: one branched ester-aldehyde route, two
saturated isocyanide preparations, and one oleyl-isocyanide preparation. The
oleyl isocyanide has a reported 35% final isolated yield. One final
phenylphosphinic-acid-catalyzed Ugi product is isolated and analytically
characterized at a reported 70.71% yield.

The branched aldehyde labeled B5 in JC_2023 exactly matches AGILE B2 and
LNPDB component `lnpdb-tail1-668dc43d423c6abc`. This independently
corroborates the route but does not resolve AGILE's separate B5 mapping
discrepancy. The raw LNPDB oleyl-tail string does not encode an isocyanide, so
the source-resolved structure is stored as reconciliation evidence without
overwriting the raw record.

**QA boundary.** The source contains inconsistent mass and mmol fields for the
branched aldehyde intermediates and an incompatible printed `6.2 mmol` final
product amount at an 87.57 micromol input scale. FORGE preserves the printed
values, records the inconsistency, and does not silently repair them.

**Consequence.** Native-Ugi supervision is stronger, but it remains
positive-heavy and does not yet close current procurement or head synthesis.
Proceed to LX_2024 for distinct transferable-tail chemistry while retaining
the unresolved-head and terminal-material queues.

---

## 2026-07-29: Admit 15 LX_2024 aldehydes as direct Ugi-role transfers

**Decision.** Add the 15 source-reported LX_2024 aldehyde tails to the
route-grounded transfer registry. Treat their original reductive-amination
lipids and biological outcomes as source-platform evidence only. Do not
transfer those product labels to a new Ugi lipid.

**Computed result.** The source provides ten aromatic diester aldehydes and
five aromatic triester aldehydes, each with an EDC and DMAP preparation family,
an exact source structure, and product-specific proton NMR. All 15 aldehydes
produce exactly one sanitized product under the frozen Ugi transform with the
reference amine and isocyanide.

Fourteen aldehydes match parsed LNPDB components exactly. A2-8 is recovered
from the source despite an invalid trailing underscore in its LNPDB string.
One extra parsed LNPDB triheptanoate is absent from the reported 15-member
source set and remains excluded. The A3-6b structure specifies
tris(2-methylhexanoate), while the quantitative prose names heptanoic acid, so
the conflict is preserved and the reagent line is not treated as an
unqualified exact-substrate procedure.

**Head finding.** The source identifies all 12 amine heads and historical
vendors, but provides neither current catalog-level closure nor upstream head
syntheses. This strengthens identity and historical availability evidence,
not present-day L3 closure.

**Consequence.** The transfer pilot has passed decisively for aldehydes. Further
broad paper mining is no longer automatic. Continue only for a declared gap:
new isocyanide chemistry, unresolved head or terminal closure, negative
substrate-scope evidence, or a route failure repeatedly encountered by
high-value generated candidates.

---

## 2026-07-29: Select hierarchical joint product-L1 generation with hybrid L2 closure

**Decision.** Use a joint model for the complete lipid and exact L1 component
decomposition, followed by a hybrid recursive L2 layer that combines
deterministic atom-mapped chemistry, bounded search, and learned proposal or
ranking. Do not build a monolithic complete-route decoder from the current
corpus.

**Evidence.** The frozen inventory contains 1,200 measured products passing the
qualified Ugi-site policy, 12,276 virtual products with exact unique
decompositions, 44 structure-resolved L2 route instances, 11 L2 route
families, and 24 components with exact source programs. It also contains zero
reported negative outcomes, zero upstream amine-head routes, and zero product
routes closed through every L2 and L3 branch.

**Interpretation.** Product and final-assembly supervision is dense enough for
joint learning. Upstream supervision is useful for reusable route templates
and later proposal or ranking, but is positive-heavy and incomplete. The
hierarchical factorization preserves synthesis-guided sampling without forcing
sparse route trees into a monolithic decoder.

**Stopping policy.** Broad source mining stops after the completed JC_2023 and
LX_2024 audits. Further review is triggered only by repeated missing-knowledge
failures, unresolved panel materials, a declared isocyanide-scope gap, or the
need for negative outcome evidence. Numerical operational-closure thresholds
are frozen in `configs/route/m0_09_l2_supervision_decision.json`.

---

## 2026-07-29: Protect broad support and limit pre-prospective synthesis-value claims

**Decision.** Continue broad-corpus whole-lipid training while applying masked
L1 decomposition losses only to chemically reviewed examples. Do not
fine-tune solely on the 12,276 Ugi products without a frozen broad-corpus
replay or source-balancing policy.

**Rationale.** The hierarchical architecture can still collapse if later
joint training forgets the broad lipid corpus. The narrow route-labeled
distribution must not determine the product model's molecular support.
Monitor broad-corpus validation loss, chemical-space coverage, motif or
topology recall, component entropy, and concentration on common AGILE
components.

**Synthesis-value boundary.** Before prospective outcomes include failures,
FORGE uses an evidence-weighted route-completion value based on closure,
forward consistency, evidence, burden, and uncertainty. Raw route-model
likelihood is prohibited as the synthesis value, and the value is not called a
calibrated probability of experimental synthesis success.

**Coupling requirement.** The route value must change molecular transition
probabilities before candidate lock. Select guidance strength subject to
frozen diversity, broad-coverage, and component-novelty floors. Report route
closure separately for familiar AGILE components, transferred known
components, and genuinely generated components.

---

## 2026-07-29: Freeze the M0-05 blinded decomposition review packet

**Decision.** Do not admit any reaction family to joint L1 training from exact
round-trip reconstruction alone. Require two independent chemists to review a
frozen packet and record each family as admit, restrict, or reject.

**Computed result.** The M0-04 ledger contains 16,424 exact decompositions after
deduplication across the four holdout schemes. The frozen M0-05 packet contains
319 unique blinded cases: 241 observed-corpus decompositions, 15 AGILE virtual
Ugi applicability cases, 15 source-routed LX_2024 aldehyde-transfer cases, and
48 adversarial controls. The observed frame reaches the target of 30 cases per
family or includes every available case when fewer exist. Eleven families have
at least one candidate; `iphos_amine_dioxaphospholane` has zero and remains
explicitly unassessed.

**Specificity design.** The packet contains eight controls each for role swaps,
same-role component replacements, wrong reaction families, non-Ugi products
forced through Ugi, wrong reactive sites, and nonduplicated high-risk
graph-reconstructing decompositions. The last set is a chemical challenge, not
an assumed negative.

**Claim boundary.** All 271 noncontrol cases reconstruct mechanically under
their declared transforms. Chemical precision, specificity, reviewer
agreement, and family disposition remain unresolved until human review. The
current M0-05 result is therefore `awaiting_human_chemist_review`, not
complete.

---

## 2026-07-29: Replace mandatory chemist sign-off with source-evidence adjudication

**Decision.** Human chemist review no longer blocks exact Ugi supervision.
Use a literature-grounded chemistry agent to inspect hash-pinned primary
articles and supplementary schemes, then apply a deterministic evidence gate
for identity, reactant roles, reactive-site qualification and exact forward
reconstruction. Preserve the 319-case blinded packet as an optional external
audit.

**Evidence result, amended by the M0-07 source reconciliation below.** Four
supplements and 27 chemistry pages are hash-verified. The gate admits 1,100
source-reported single-compound AGILE products as exact Ugi L1 supervision,
retains 100 B4 mixture executions as non-single-graph abstentions, and admits
12,276 virtual products only to a transform-consistency objective. It admits
44 exact source-derived L2 route instances and abstains on the unresolved
exact upstream route to the AGILE B5 aldehyde.

**No inflation rule.** Analogue and family precedent cannot become exact route
labels. Cross-platform tail proposals cannot become observed L2 supervision.
Missing evidence cannot become a negative outcome. A source conflict remains
an abstention until resolved by new source evidence or prospective experiment.

**Consequence.** The primary Ugi L1 training gate is now operational without
subjective sign-off. Non-Ugi R0 decompositions remain structure-only unless
they receive independent source qualification under the same policy.

---

## 2026-07-29: Reject dense independent-edge flow for the production prior

**Decision.** Keep whole-lipid discrete flow matching, but do not promote the
tested dense independent-edge parameterization to full product-prior training.
Build a sparse or hierarchical topology flow with factorized endpoint
selection, full pair reachability, and explicit connectivity and valence
constraints.

**Input correction.** The current hash-pinned R0 has median 53 and maximum 282
heavy atoms. Full-R0 coverage is 70.48% at 64 atoms and 91.23% at 96 atoms,
not the previously documented 86.5%, 96.8%, and maximum 138. Ninety-two rows
contain F and 252 contain Si, leaving 15,089 rows within the declared
C/N/O/S/P vocabulary. These exclusions and the 1,251 supported rows above 96
are explicit support facts.

**Dense reference result.** Standard clean-marginal edge cross-entropy is
sparsity-calibrated, with held-out bond-density errors of 0.0123 at N=64 and
0.0031 at N=96. Bonded-edge recall falls from 0.474 to 0.237, however, while
overall edge accuracy remains above 0.95 because absent edges dominate. None
of 96 sampled endpoints is both valid and connected.

**No-edge stress result.** Adding a 0.25 bonded-edge auxiliary term keeps bond
recall near 0.52 and makes every endpoint connected. It also raises endpoint
bond-density error to 0.136 at N=64 and 0.150 at N=96. All endpoints fail
sanitization through valence errors. Reweighting the same dense pair objective
trades disconnected underbonding for invalid overbonding.

**Claim boundary.** This bounded result rejects one parameterization. It does
not reject DeFoG-style discrete flow matching, whole-molecule generation, Ugi
inpainting, or hierarchical molecule and route coupling. A sparse topology
probe must pass before full product-prior training.

---

## 2026-07-29: Select sparse tree-plus-closure flow for the product prior

**Decision.** Use a canonical spanning-tree plus residual-closure
parameterization for later whole-lipid discrete flow implementation. Preserve
full atom-pair reachability, all observed cycle ranks, connectivity by
construction, valence-aware terminal decisions, and explicit repair
accounting. Do not silently cap molecular size at 96 atoms.

**Representation result.** All 15,089 single-fragment R0 structures within the
declared C/N/O/S/P vocabulary round-trip exactly. The maximum observed cycle
rank is 12. The representation eliminates dense N by N by hidden-dimension
edge states, although the reference implementation still materializes N by N
parent-pointer logits.

**Bounded result.** At N=64 and N=96, all 48 sampled endpoints per size were
valid and connected, all valid topologies were novel relative to bounded
training data, and no terminal constraint repairs were required. Training
throughput was 825.51 and 432.02 graphs per second, respectively. Scaling dry
runs completed through N=282 while remaining below the frozen memory ceiling.

**Claim boundary.** This selects a production representation for later
implementation. It does not constitute full product-prior training, prove
converged generative quality, or establish prospective lipid performance.

---

## 2026-07-30: Reconcile AGILE assays and B4/B5 identities before oracle fitting

**Decision.** Treat the published AGILE experiment as 1,200 nominal library
measurements, not 1,200 unqualified single-molecule training examples. All
biological oracle fits must use the deterministic M0-07 reconciled artifact.
Direct fitting from the root AGILE CSV is prohibited.

**Source finding.** The HeLa and RAW response values in FORGE's vendored root
AGILE table match the official article source workbook for all 1,200 labels.
The 235-label problem reported by LANTERN therefore does not describe this
specific root table. The B4/B5 representation defect is present: B4 is a
cis/trans mixture and B5 is a pure-trans compound, while the root table encodes
them as two opposite single stereoisomers.

**Training policy.** Exclude the 100 B4 mixture measurements from
single-graph supervision and preserve them in a dedicated ledger. Retain the
100 B5 measurements with their graph corrected to pure trans. The model-facing
representation remains stereochemistry-free, yielding 1,100 unique
constitutional graphs. Isomeric identity is retained only for chemistry and
provenance.

**Verification.** All 1,200 HeLa and RAW labels match the official workbook.
All 1,100 retained HeLa structure-label pairs match LANTERN. The curated
artifact contains 1,100 unique model graphs, the exclusion ledger contains 100
B4 mixtures, and no unavailable formulation measurements are imputed.

**Downstream consequence.** M0-05 is rebuilt to admit 1,100 exact
single-compound L1 records and retain 100 mixture abstentions. The exact B5
upstream route remains unresolved despite resolution of its product identity.
The AGILE-derived portion of R0 and dependent M0-03, AGILE-specific M0-04, and
M0-06 corpus counts must be rebuilt before product-model training. Non-AGILE
M0-04 conclusions and unrelated M0-09 route evidence remain valid.

---

## 2026-07-30: Freeze leak-aware M0-07 oracle evaluation splits

**Decision.** Freeze the complete oracle evaluation contract before any model
fit. The exact LANTERN random split is a reproduction diagnostic only. Model
selection uses a valid scaffold holdout plus held-component and
held-component-pair evaluations. The 12,276 virtual candidates are unlabeled
and support only applicability and distance analysis.

**Source audit.** The hash-pinned LANTERN artifact named
`Murcko_scaffold.npy` assigns all six Murcko scaffold groups across multiple
partitions. It is not scaffold-disjoint and is audit-only. LANTERN's separate
`scaffold_balanced.npy` artifact assigns four scaffold groups to training, one
to calibration, and one to test with zero leakage. Its limited six-scaffold
support must be reported alongside the component holdouts.

**Frozen contract.** The artifact contains 35,200 row assignments over eight
schemes: exact random and scaffold-balanced partitions plus deterministic
five-fold held-head, held-aldehyde, held-isocyanide, and all three
component-pair schemes. Each held group appears in test exactly once per
scheme and never crosses into training or calibration in that fold. Seed 1729
and every input SHA256 are recorded.

**Selection and calibration.** Random-split performance cannot select the
oracle. Calibration uses split-conformal absolute residuals with 80%, 90%, and
95% nominal coverage. No model is frozen by this split task. Apparent pKa,
particle size, polydispersity, encapsulation efficiency, and formulation
robustness remain unavailable and are not imputed.

---

## 2026-07-30: Retain strict applicability limits after the classical oracle matrix

**Decision.** Do not freeze an AGILE oracle from the classical representation
lane. Continue the molecular-graph, region-aware, and lipid-pretrained lanes
before selection. Treat the exact random split only as a diagnostic.

**Computed result.** The matrix contains 768 fits over three representations,
four regressors, two endpoints, and 32 frozen partitions. The best random
diagnostic reached R² = 0.635 for HeLa and R² = 0.461 for RAW 264.7. Equal
weighting of the seven selection-eligible schemes selected random forest with
Morgan plus RDKit features as the classical-lane leader, with mean R² = 0.215
for HeLa and 0.087 for RAW 264.7.

**Applicability finding.** Entire-component shifts were substantially harder.
The best mean HeLa R² was -0.035 for held heads, -0.009 for held aldehydes, and
0.299 for held isocyanides. The corresponding RAW 264.7 values were 0.114,
-0.009, and 0.018. Component-pair holdouts were easier because individual
components remained represented elsewhere in training.

**Calibration finding.** The classical leaders had mean absolute 90% coverage
gaps of 0.066 for HeLa and 0.045 for RAW 264.7, but mean interval widths were
8.648 and 5.319 mTP units. Near-nominal coverage therefore does not imply
precise extrapolation.

**Candidate-library finding.** All 12,276 unlabeled virtual candidates were
unique under the stereochemistry-free model representation. Maximum Morgan
similarity to the labeled set had median 0.939 and minimum 0.508. These values
measure structural proximity only, not biological confidence.

**Consequence.** The eventual guidance policy must attenuate or abstain for
unseen aldehyde and head families unless a later representation earns stronger
calibrated transfer. Tail-route expansion enlarges synthesis support; it does
not create biological labels. AGILE remains a general-transfection oracle, not
an in-vivo endpoint oracle.

---

## 2026-07-30: Reproduce but exclude the released LANTERN checkpoint

**Decision.** Preserve LANTERN's released AGILE checkpoint as an exact
literature diagnostic. Prohibit it from model selection, calibration, and
oracle freezing.

**Computed result.** The hash-pinned checkpoint, 2,048 circular features, 210
expert descriptors, and exact 880/110/110 random split reproduce HeLa test R²
= 0.820, RMSE = 1.416, and Pearson r = 0.906.

**Leakage finding.** The committed source pipeline fits MinMax scaling to all
1,100 feature vectors and all 1,100 HeLa labels before applying the split.
Test feature and label ranges therefore influence preprocessing. The
reproduction retains this behavior for source fidelity and labels it as
test-information leakage.

**Interpretation.** The difference from FORGE's leakage-free random diagnostic
cannot be assigned entirely to leakage because architecture, source features,
and checkpoint selection also differ. Neither random-split result establishes
component-family transfer. The frozen scaffold and component holdouts remain
selection-relevant.

**Auxiliary-data rule.** Native and related Ugi-3 datasets may enter a
predeclared shared-encoder or transfer-learning comparison, but differently
normalized assay values are not silently concatenated. Keep additional data
only when it improves frozen AGILE component holdouts or declared external
transfer without exposure to held labels.

---

## 2026-07-30: Expand the endpoint comparison and defer the lock

**Decision.** Add intramuscular reporter delivery followed by functional
editing to the endpoint comparison and retain it as the current lowest-risk
option, but defer the endpoint lock while M0 computation continues. Keep the
software endpoint-generic and require an explicit endpoint identifier. Do not
hard-code a recommendation into the molecular generator, route model, or
oracle.

**Rationale.** The earlier vaccine package imported the evidence burden of a
standalone vaccine paper into a synthesis-grounded AI platform paper. Nature
Biotechnology precedents support a leaner progression. LiON validates AI-guided
lipid discovery through reporter mRNA delivery in mouse and ferret lungs.
Bowen Li's earlier combinatorial LNP study progresses from reporter delivery to
Cre and Cas9 reporter editing without disease correction. AGILE directly
reports a HeLa-to-IM reporter correlation for 15 selected lipids. A compact IM
campaign can therefore use reporter delivery and distribution for the selected
panel, then functional editing in one or more leads. Disease correction and
exhaustive mechanism are strengthening experiments, not automatic
requirements.

**Liver boundary.** AGILE does not report HeLa-to-IV-liver correlation, so it
cannot be treated as a liver oracle. Historical work shows that HeLa can be
informative for hepatocellular delivery within some matched lipidoid libraries
and formulations, but primary hepatocytes were more predictive and formulation
altered the relationship. Liver remains viable only after an external
liver-labelled corpus passes an assay, formulation, source, and chemistry
audit, or after a prospective hepatocyte-relevant bridge controls in vivo
advancement.

**Vaccine alternative.** The laboratory reports mature vaccine capability, and
AGILE, JC_2023, and Miao 2019 align more naturally with IM expression and APC
biology. If vaccination is selected, the minimum claim-bearing package requires
IM delivery plus at least one predeclared antigen-specific response. Both
adaptive arms, extended durability, and challenge are recommended only when
they support the chosen claim.

**Boundary.** The selected-panel HeLa-to-IM correlation is not a universal
transfer guarantee. AGILE HeLa and RAW 264.7 measurements remain
general-transfection labels. Unavailable apparent pKa, particle, and
formulation measurements are never imputed.

**Lock condition.** The endpoint remains formally unlocked until the editing
target or antigen, endpoint-specific bridge, in vivo functional assay, animal
capacity, benchmark LNP, formulation protocol, and minimum tolerability
measurements are confirmed.

---

## 2026-07-30: Admit auxiliary 3CR supervision without raw label pooling

**Decision.** Authorize a controlled M0-07 comparison using JC_2023 and LM_2019
as auxiliary supervision. Prohibit direct concatenation of their LNPDB targets
with AGILE. Compare AGILE-only training against source pretraining followed by
AGILE fine-tuning and a shared encoder with study-specific heads.

**Computed source result.** JC_2023 contributes 288 unique native AGILE-type
Ugi products with HeLa labels. Its 16 amine heads all overlap AGILE exactly,
while five of six aldehydes and one of three source-corrected isocyanides are
new relative to the corresponding AGILE roles. Thirty-two corrected complete
products overlap AGILE. LM_2019 contributes 1,080 unique related
isocyanide-mediated 3CR products with 1,080 HeLa, 36 BMDC, and 12 BMDM records
and no exact AGILE product overlap.

**Identity gate.** Primary-source Figure 1 resolves JC C1 as oleyl, C2 as
saturated C18, and C3 as saturated C11. The prior M0-09 review had C2 and C3
swapped and has been corrected. LNPDB also encodes all 18 A3 product graphs
without the tertiary amine present in source A3. The audit preserves those raw
graphs, rebuilds their model identities from the source components only when
the qualified transform produces one unique product, and verifies 288 of 288
corrected model identities by exact forward reconstruction.

**Assay boundary.** The auxiliary endpoint values are z-scored separately
within each study and endpoint. They cannot be interpreted as one shared
quantitative target. Study-specific heads or sequential transfer preserve that
distinction.

**Leakage gate.** Auxiliary filtering is fold-specific and occurs before any
pretraining or joint fitting. Exclude exact AGILE test products for every
split, then exclude held scaffolds, component structures, or component pairs
for their corresponding evaluations. Prediction heads are keyed by study and
endpoint. Available LNPDB auxiliary z-scores use complete study-endpoint
populations, so internal auxiliary test metrics are transductive diagnostics
and cannot select the final oracle.

**Retention gate.** Keep an auxiliary strategy only if it improves the frozen
AGILE component holdouts or a declared external-transfer endpoint without
exposure to held labels. A random-split gain alone is insufficient.

---

## 2026-07-30: Freeze a constitutionally leakage-safe graph-pretraining corpus

**Decision.** Pretrain the M0-07 lipid graph encoder only on the frozen
constitutionally deduplicated R0 ledger. Exclude every exact constitutional
match to any of the 1,100 reconciled AGILE model graphs globally before
pretraining. Do not use the 12,276 virtual candidates or any biological labels
for encoder pretraining.

**Computed result.** R0 contains 15,433 eligible rows and 15,229 unique
constitutional graphs. All 1,100 AGILE constitutions occur in R0 through 1,220
rows. Their removal leaves 14,213 rows, which collapse to 14,129 unique
pretraining graphs. The final exact constitutional and standard-InChI
connectivity overlap with AGILE is zero.

**Leakage finding.** Source provenance is not a safe substitute for molecular
identity. Removing rows tagged `agile_measured1200` would leave 20
LNPDB-derived target constitutions in pretraining. The test suite freezes this
counterexample.

**Support correction.** The upstream training manifest does not describe the
pinned R0 accurately. Actual R0 support includes F and Si, 509 net-charged
records, triple bonds, and graphs up to 282 atoms. Encoder vocabulary and size
support must be derived from the audited data, and graph truncation is
prohibited. Potential and explicitly specified stereochemistry are reported
under separate definitions because the manifest's stereocenter field is
ambiguous.

**Training boundary.** The model-facing pretraining ledger contains only a
hashed graph ID, constitutional SMILES, and graph-size fields. Provenance,
component annotations, study fields, and assays remain in audit ledgers and
cannot enter the tensors. R0 pretraining is whole-graph self-supervision only;
any later region-supervised pretraining would require fold-specific component
filtering.

---

## 2026-07-30: Pass the sparse graph runtime gate without selecting an oracle

**Decision.** Authorize the full supervised graph matrix after the
deterministic tensor cache and resumable fit scheduler are frozen. Runtime does
not select a scientific model.

**Computed result.** From clean commit `232d9fd`, all reconciled AGILE product
and A/B/C role graphs, all 14,129 R0 pretraining graphs, and all 12,276 virtual
applicability candidates tensorized without truncation. On 700 training-only
records, median CPU epoch times were 0.209 s for whole-graph D-MPNN, 0.230 s
for edge-aware GIN, and 0.556 s for Ugi component-role-aware D-MPNN. Maximum
atom and edge permutation deviation was `7.45e-8`. No calibration or test
target was accessed.

**Consequence.** Run random-initialized supervised graph models and label-free
R0 pretraining as independent lanes. Cache tensors once, then schedule
architecture, partition, endpoint, and seed fits independently with resumable
manifests and bounded CPU threads.

---

## 2026-07-30: Condition the Ugi adapter by region without fragment generation

**Decision.** Keep the broad whole-lipid discrete-flow backbone
region-agnostic. In the Ugi synthesis-program adapter, add product-core,
amine-origin, aldehyde-origin, isocyanide-origin, attachment, and
attachment-relative graph-distance context. Use shared whole-graph message
passing with lightweight role-conditioned adapters and region-balanced losses.

**Boundary.** FORGE continues to generate one complete connected lipid graph.
The head and tails are not produced by independent fragment generators and are
not selected as component IDs. All molecular regions evolve simultaneously,
retain cross-region message passing, and are evaluated through exact L1
round-trip consistency.

**Evaluation.** Compare a region-agnostic flow, late role embeddings, and
role-conditioned message passing with attachment-relative positions. A fully
separate head or tail model is an overfitting and enumeration control, not the
default. Any gain must survive held-head, held-tail, component-pair, topology,
and open-ended component-novelty evaluations.

---

## 2026-07-30: Predeclare the final M0-07 oracle freeze

**Timing boundary.** Freeze the scientific adjudication contract after 187 of
576 supervised graph fits and 4 of 384 label-free transfer fits had completed,
before either graph lane had an aggregate result and before any graph model
could be selected.

**Selection.** Compare the classical, random-initialized graph, and label-free
R0 transfer lanes using the same seven eligible scaffold, component, and
component-pair schemes. Select one representation and model across both HeLa
and RAW 264.7 endpoints by equal-endpoint, equal-scheme mean test R². Break
ties using lower RMSE, then lower absolute 90% coverage gap, then a lexical
candidate identifier. Exclude the random diagnostic, released LANTERN
checkpoint, and unlabeled 12,276-product set from selection.

**Guidance gate.** A held-out scheme authorizes guidance only when its mean
test R² is at least 0.10, mean Spearman correlation is at least 0.20, at least
60% of folds have positive R², and the mean absolute 90% conformal-coverage
gap is at most 0.10. These are minimum evidence gates, not claims of strong
prediction. Authorized domains use conservative conformal lower-confidence
guidance rather than raw mean maximization.

**Applicability.** Map novel-head, novel-aldehyde, novel-isocyanide, and
two-role novelty to their corresponding frozen component schemes. Require the
scaffold scheme for every domain and the relevant component-pair schemes for
novel combinations. Any failed required gate abstains. Three simultaneously
unseen component roles, non-Ugi final assemblies, and in vivo endpoint claims
always abstain under M0-07.

**Checkpoint boundary.** This adjudication freezes the winning representation,
model family, and evidence-bounded use policy. A deterministic full-data
production refit and hash-pinned checkpoint remain required before guided
generation.

---

## 2026-07-30: Amend the M0-07 freeze before aggregate graph results

**Timing and reason.** An independent read-only code audit identified that the
initial freeze would select an architecture directly from outer-test metrics.
The audit completed when 301 of 576 supervised fits and 83 of 384 transfer
fits existed. Neither graph lane had an aggregate result, and no partial
scientific performance metric was inspected. This amendment therefore responds
to the evaluation design, not to model outcomes.

**Architecture selection.** Select the one production representation and model
using equal-endpoint, equal-scheme calibration R² only. Break ties using
calibration RMSE, calibration Spearman correlation, and then the lexical
candidate identifier. Outer-test R², RMSE, rank correlation and coverage do
not select the architecture. They remain post-selection evaluation evidence
and inputs to the predeclared deploy-or-abstain gate.

**Nested held-component audit.** Within every outer scheme and fold, select a
candidate using that fold's calibration rows only, then evaluate the selected
candidate on that fold's untouched test rows. Report this fold-local nested
selection audit separately from the fixed production candidate's
cross-validated score. A domain authorizes guidance only when both the fixed
candidate and the nested selection audit pass the original performance and
coverage thresholds. These internal results remain development-corpus
evidence, not a substitute for prospective experimental validation.

**Completeness and provenance.** Require exact lane completion statuses, the
complete predeclared candidate roster, every expected fold and an immutable
hash manifest for every neural fit used to choose the production epoch count.
The production loader authenticates the checkpoint through the frozen
production-result artifact and verifies the selected model, data policy,
chemistry contract and software versions.

**Applicability enforcement.** Infer the guidance domain from canonical amine,
aldehyde and isocyanide identities stored in the checkpoint. Before scoring,
require exact forward reconstruction of the supplied complete product through
the hash-pinned Ugi transform. Three unseen component roles, a product that
fails forward reconstruction, a non-Ugi final assembly and an in vivo endpoint
remain explicit abstentions. The caller cannot authorize a candidate by
supplying a less novel domain label.

**Uncertainty wording.** Foldwise split-conformal intervals remain valid
evaluation objects for their fitted fold models. The maximum held-domain
radius transferred to the full-data production refit is reported as a
conservative empirical held-domain radius, not as a new finite-sample coverage
guarantee for the refitted ensemble.

---

## 2026-07-30: Freeze the M0-07 oracle with universal guidance abstention

**Decision.** Freeze
`supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble` as
the M0-07 production architecture, then abstain from AGILE oracle tilting in
every declared guidance domain. Do not weaken the predeclared fixed-model or
fold-local nested evidence gates.

**Selection evidence.** The complete matrix contains 768 classical fits, 576
supervised-graph fits, and 384 label-free transfer fits. Architecture selection
used calibration metrics only. The selected model reached equal-endpoint
calibration R² = 0.3673, RMSE = 1.8584, and Spearman correlation = 0.5960.
After selection, its equal-endpoint test R² was 0.1830. The label-free R0
transfer candidates did not provide stronger held-component evidence.

**Abstention evidence.** Every guidance domain requires the valid
scaffold-balanced audit. The fold-local calibration-selected model failed that
single scaffold test fold for both HeLa and RAW 264.7, and additional
component-role failures occurred. Consequently, neither known-component novel
combinations nor unseen-head, unseen-aldehyde, unseen-isocyanide, paired-unseen,
three-unseen, non-Ugi, or in-vivo use is authorized for biological tilting.
The selected model remains an auditable predictor and a future update target,
not a validated guidance oracle under the present labels.

**Production artifact.** A deterministic three-seed ensemble for both
endpoints was refit on all 1,100 reconciled single-structure records. The
checkpoint is 5,041,009 bytes with SHA-256
`46f13b8a3cfef1f891e308dfda75e90685d82d1289f0061acdae4be18870e61c`.
`results/m0_07/oracle_production_result.json` authenticates the checkpoint,
freeze result, configuration, selected model, data policy, exact Ugi chemistry
contract, applicability index, and inference software versions.

**Inference boundary.** Candidate domain classification is automatic. It
requires exact reconstruction through the pinned AGILE-type Ugi 3-CR transform
and compares canonical amine, aldehyde, and isocyanide identities with the
checkpoint-owned 1,100-record component index. The caller cannot supply a less
novel domain. An incompatible product returns `non_ugi_final_assembly`, and
all current domain scores return `abstain`.

**Data boundary.** No guided-generation or virtual-candidate label was used.
No apparent pKa, particle size, polydispersity, encapsulation efficiency, or
formulation property was imputed. AGILE remains a general in-vitro
transfection dataset and is not an in-vivo endpoint oracle.

---

## 2026-07-30: Extend sparse product support to fluorinated and siloxane lipids

**Decision.** Expand the later sparse product-prior vocabulary from
C/N/O/S/P to C/N/O/S/P/F/Si. Retain deterministic Kekulization before the
discrete flow and require sanitization to restore the exact constitutional
graph and aromaticity. Keep ring closures available across the complete
molecule rather than restricting them to a head region.

**Evidence.** The 92 fluorine-containing R0 records are coherent fluorinated
aromatic lipidoids, primarily from KW_2014. The 252 silicon-containing records
form a coherent LX_2024_3 siloxane-lipid series. They are not parser artifacts
and should not be discarded merely to simplify the atom vocabulary.

**Representation result.** All 15,433 current R0 structures round-trip exactly
through the sparse tree-plus-closure representation and recover the same
sanitized constitutional graph. This includes all 1,903 aromatic structures.
The bounded full-support runs remained 100% valid and connected at N=64 and
N=96 with no terminal constraint repairs. The N=282 scaling dry run remained
below the frozen memory ceiling.

**Scope boundary.** This establishes representation support only. The current
R0 and split artifacts predate the M0-07 AGILE source correction and remain
quarantined from production-prior training. Rebuilding those inputs is still
required before a full flow fit.

---

## 2026-07-30: Freeze a lipid-native ring gate for the primary Ugi campaign

**Decision.** Preserve broad sparse-graph representation support for observed
lipid ring topologies, but do not equate representability with automatic
candidate-lock support. The primary Ugi campaign automatically admits acyclic
products or products containing exactly one 5- or 6-membered ring in the amine
head. It does not automatically admit macrocycles, fused rings, spiro systems,
bridged systems, or rings in aldehyde- or isocyanide-derived regions.

**Evidence.** The 1,100 reconciled single-structure AGILE records contain 660
cyclic products. All 660 have exactly one 5- or 6-membered heterocycle in the
amine head, while every aldehyde and isocyanide component is acyclic. No
product contains a macrocycle, fused ring, spiro center, or bridgehead. The
pre-reconciliation broad R0 contains 8,127 cyclic records, including 477
macrocycle records, which confirms that broad representation support is wider
than the prospectively validated Ugi topology.

**No fixed ring vocabulary.** The gate constrains topology rather than exact
ring fragments. Novel 5- or 6-membered head heterocycles remain representable
and can enter the campaign when their reactive site, assembly compatibility,
and L2/L3 route closure satisfy the frozen evidence policy.

**Abstention rule.** A candidate outside automatic topology support must
abstain from primary candidate lock until exact source or prospective assembly
evidence, complete L2/L3 closure, and an explicit support amendment exist.
Ring support does not imply biological-oracle applicability, and the M0-07
universal guidance abstention remains in force.

**Artifact.** `results/m0_06_ring_support/result.json` records the hash-pinned
R0 and reconciled AGILE census, component-role localization, topology
definitions, and frozen gate. The broad R0 census must be rerun after the
constitutional corpus rebuild; the corrected AGILE evidence already
authenticates the primary Ugi gate.

---

## 2026-07-30: Freeze LNPDB head transfer as a queue, not a vocabulary

**Decision.** Use LNPDB-derived heads to broaden structural precedent and
prioritize route work beyond the 20 AGILE amines. Do not restrict the
whole-lipid generator to this identity set, and do not admit a head merely
because it occurs in an experimentally studied source lipid.

**Census.** The component ledger contains 408 head entries, of which 396 parse
cleanly. Twenty are exact AGILE identities. Another 259 pass the qualified
Ugi N-H amine policy and primary topology gate: 130 are acyclic, 22 contain
one 5-membered ring, and 107 contain one 6-membered ring. Sixty-two parsed
heads lie outside the primary ring gate, 55 lack a qualified Ugi amine, and 12
source structures remain invalid.

**Generation boundary.** Heads remain generated at atom and bond resolution as part of the
complete lipid graph. A generated head may be absent from both AGILE and
LNPDB. Candidate support depends on a site-defined Ugi amine, exact forward
reconstruction, and route or procurement closure, not membership in the
frozen queue.

**Evidence boundary.** Thirty-six transfer candidates require explicit
reactive-site resolution. None of the 259 has assessed procurement or upstream
route evidence in the current LNPDB component ledger, so the artifact admits
zero candidates automatically. Source-lipid occurrence and biological
performance are provenance only and do not transfer to a new Ugi product.

**Operational rule.** Do not route all 259 heads preemptively. Prioritize heads
that appear in high-value generated candidates or provide a scientifically
important expansion, then attach exact procurement or route evidence. Ring
topologies outside the automatic gate require a separate support amendment.

---

## 2026-07-30: Freeze label-free R0 transfer as a nonselected diagnostic

**Decision.** Preserve the completed frozen-encoder and fine-tuned D-MPNN
transfer comparison as a diagnostic pretraining ablation. Neither transfer
candidate is an authorized guidance oracle or production checkpoint.

**Selection evidence.** The frozen cross-representation policy used
equal-endpoint, equal-scheme calibration R² and did not use outer-test metrics
to choose an architecture. The fine-tuned transfer candidate ranked 12th with
calibration R² = 0.0353, while the frozen-head candidate ranked 14th with
calibration R² = -0.0335. The selected role-aware supervised graph model ranked
first with calibration R² = 0.3673.

**Post-selection evidence.** Equal-endpoint test R² was -0.0437 for fine-tuning
and -0.0788 for the frozen head. These negative values did not determine model
selection. They are post-selection evidence that the current label-free
objective did not improve held-component generalization.

**Guard.** `results/m0_07/oracle_graph_transfer_decision.json` pins the complete
384-fit, 128-ensemble matrix and the cross-representation freeze. Silent
promotion is prohibited. Reopening the lane requires a new hash-pinned
cross-representation freeze with complete held-component evidence and an
explicit applicability-policy decision.

---

## 2026-07-30: Block the FlowER transfer pilot and apply the frozen demotion rule

**Decision.** Do not run or fabricate the M0-10 fine-tuning curve. Demote
FlowER to an orthogonal consistency check. Keep the deterministic atom-mapped
forward verifier load-bearing. This disposition does not block training the
whole-lipid product model or implementing the hybrid L2 route system.

**Asset evidence.** The workspace has neither a hash-pinned FlowER source
checkout nor the pretrained `model.2880000_95.pt` checkpoint. The May 2026
probe memo and scripts are present and hash-pinned, but the scripts acquired
code and archives in a transient Modal image. That probe remains prior
evidence, not the M0-10 experiment.

**Supervision evidence.** Each of the three frozen upstream classes has one
positive and one negative registry qualification example, but zero elementary
bond-electron mechanism trajectories. Atom-mapped transform reconstruction is
not a FlowER trajectory label. The 32-example fine-tuning curve therefore has
a deficit of 32 trajectories per class. The current precursor capability
ledger contains 82 prospective outcome entries, all `not_attempted`, and no
measured conversion outcomes.

**Frozen rule.** Headline status requires both improved held-out forward
consistency after fine-tuning and a prospective conversion Spearman confidence
interval excluding zero. Neither condition is currently evaluable, so the rule
fails mechanically. Literature yields, deterministic transform validity, and
the earlier zero-shot probe may not substitute for either condition.

**Reopening boundary.** Vendor an exact source commit and authenticated
checkpoint, curate at least 32 elementary trajectories per target class, freeze
the reaction-class holdout before fitting, and acquire a locked prospective
conversion ledger. No automatic download or compute run is authorized by the
M0-10 blocker artifact.

---

## 2026-07-30: Freeze an evidence-bounded M0-09 source-paper priority overlay

**Decision.** Preserve `results/m0_09/source_queue.csv` as the immutable
acquisition authority and add `source_priority_queue.csv` as a deterministic
review overlay. This avoids invalidating the source acquisition and chemistry
review hash chain.

**Computed result.** The overlay retains all 43 source rows, including 42
publications and one commercial source. Four completed chemistry reviews remain
as evidence references, while 39 sources receive an active `next_review_rank`.
All sources have explicit biological-provenance, exact-identity novelty,
method-accessibility and exact-component-redundancy states. Two sources have
curated motif-distinctiveness evidence and five have explicit Ugi-conversion
evidence. Forty-one sources therefore retain at least one missing priority
axis.

**Scoring contract.** Six equal-weight axes rank expected evidence value:
curated motif distinctiveness, source-linked biological provenance, exact
component novelty relative to AGILE, source-method accessibility, explicit
Ugi-handle conversion and exact component redundancy. Missing evidence receives
zero credit and remains listed explicitly. Ties prefer more assessed axes, then
the original acquisition rank and source identifier.

**Claim boundary.** The score prioritizes source review. It is not a
biological-potency, synthesis-success, route-closure or procurement score.
LNPDB assay provenance does not transfer a biological label to a component or
generated Ugi product. Exact structural novelty does not establish motif
transfer or Ugi compatibility.

**Artifact.** `results/m0_09/source_priority_result.json` records the frozen
policy, all input hashes, coverage counts and output digest. Run with
`make m0-09-source-priority`.

---

## 2026-07-30: Freeze corrected constitutional R0 splits

**Decision.** Replace the historical 15,433-row split bundle for model-facing
work with immutable splits over the 15,229 reconciled constitutional graphs.
The historical assignments remain audit inputs only.

**Computed result.** Constitutional reconciliation collapses 204 duplicate
rows, excludes 100 B4 mixture measurements from single-graph weight, and
preserves their constitutions through exact B5 support. Held-out denominators
are 2,328 for source study, 2,285 for headgroup, 2,285 for linker/scaffold, and
1,100 for component family. No connected group spans folds under any scheme.

**Consequence.** Every downstream model and corpus analysis must load
`data/splits/m0_03_constitutional/`. Extracting components or learned
preprocessing before applying these splits is prohibited.

---

## 2026-07-30: Freeze corrected constitutional M0-04 coverage

**Decision.** Preserve M0-04 as a reaction-registry coverage and leakage audit.
Do not optimize the registry against the observed held-out results.

**Computed result.** R1-prime exactly recovers 260/2,285 headgroup-held-out
graphs, 402/2,285 linker/scaffold-held-out graphs, 100/1,100 component-family
graphs, and 0/2,328 source-study-held-out graphs. The corresponding non-AGILE
component-based results are 205/2,065 and 341/2,224. Historical and corrected
full-corpus controls reproduce 108/14,233 and 112/14,129, respectively.

**Interpretation.** The source-study zero exposes whole-platform withholding
and reaction-registry boundaries. It is not whole-lipid generator performance
and does not establish that held-out products are unsynthesizable. The modest
exact recovery under component-based holdouts confirms that reaction
enumeration is useful auxiliary support but cannot define molecular support.

**Consequence.** Keep the product generator as a broad whole-graph model.
Expand route knowledge only through versioned adapters or repeated
generator-triggered missing-knowledge failures, followed by new untouched
evaluation.

---

## 2026-07-30: Freeze corrected sparse full-support feasibility

**Decision.** Select the sparse spanning-tree plus residual-closure
representation for Phase 1 whole-lipid product-prior implementation.

**Computed result.** All 15,229 corrected constitutional graphs and all 1,891
aromatic graphs round-trip exactly. The declared C/N/O/S/P/F/Si vocabulary
covers every row, with a maximum of 282 heavy atoms. N=64 and N=96 include
10,702 and 13,879 rows. Bounded probes produced valid, connected, unique
endpoints and passed a dry run at N=282.

**Claim boundary.** This result establishes representation and runtime
feasibility. It is not a trained product prior or evidence of prospective
lipid performance.

---

## 2026-07-30: Close M0 and authorize Phase 1 product plus L1 training

**Decision.** Following review of the corpus, oracle, representation, and
hierarchical architecture, the user explicitly authorized execution of the
broad whole-lipid plus joint Ugi-L1 training plan. M0 remains the immutable
scientific foundation for Phase 1.

**Training boundary.** Pretrain the sparse whole-lipid backbone on corrected
R0 with low-weight, realism-weighted auxiliary reaction-enumerated structures.
Then train Ugi role adapters and L1 heads before joint unfreezing with
source-balanced broad-corpus replay. Fine-tuning solely on the 12,276 Ugi
virtual products is prohibited.

**Frozen initial comparison.** Evaluate broad-to-Ugi batch allocations of
75:25, 50:50, and 25:75 unless a recorded pre-run memory or optimization gate
justifies narrowing the grid. Select using calibration evidence while enforcing
floors on validity, connectedness, broad-corpus coverage, diversity, effective
component count, and component novelty.

**Guidance boundary.** The first product-model training has no AGILE biological
tilt because every current oracle domain abstains. It has no complete
synthesis-value tilt because no product currently closes every L2 and L3
branch. Biological and synthesis controllers remain separately gated later
modules.

---

## 2026-07-30: Clarify AGILE oracle endpoint nomenclature

**Decision.** Preserve the official AGILE mTP measurements as the biological
estimands. `expt_Hela` denotes mTP measured in HeLa cells and `expt_Raw`
denotes mTP measured in the RAW 264.7 macrophage cell line. The word `Raw` in
the latter field name does not denote an unprocessed plate readout.

**Optimization and reporting.** Target standardization is fitted on training
rows only and predictions are returned to the original mTP scale for
calibration and evaluation. Cross-study assay values are not pooled as though
they shared one numerical scale. If biological guidance is later reopened, it
must use the frozen conservative uncertainty policy; direct maximization of an
unqualified predicted mTP mean remains prohibited.

---

## 2026-07-30: Freeze Phase 1 product and Ugi-L1 data contract

**Decision.** Use corrected constitutional R0 under its frozen source-study
split as the primary broad-lipid stream. Allocate 90% of broad examples to R0
and 10% to R1. Sample R1 only through `realism_weight`; raw
reaction-family-count sampling remains prohibited.

**Measured correction.** The 12,276 virtual Ugi products and 1,100
source-adjudicated measured products overlap on 576 exact source-SMILES
identities. The initial source-level union therefore contained 12,800 rows,
with measured source provenance taking priority for exact overlaps. A later
Phase 1 identity audit supersedes this row count for constitution-only model
training while preserving the source records.

**Component holdouts.** The initial source-level union contains 24 amine heads, 65
aldehyde-derived components, and 10 isocyanides. The constitution-only component
counts and assignments are superseded by the 2026-07-31 correction below. Freeze separate SHA-ranked
train/calibration/heldout assignments for each role. These component-held-out
schemes, rather than a product-level random split, evaluate L1 generalization.

**Training boundary.** Preserve the broad-to-Ugi replay comparison at 75:25,
50:50, and 25:75. Phase 1 contains no biological or synthesis-value guidance.
The authoritative artifacts are `data/splits/phase1/manifest.json`,
`data/splits/phase1/ugi_l1_assignments.csv.gz`, and
`results/phase1/data_contract.json`.

---

## 2026-07-31: Correct Ugi Phase 1 identity and freeze orthogonal synthesis semantics

**Identity correction.** The product generator explicitly models constitutional graphs without
stereochemistry. Reapplying that identity to the 12,276 virtual and 1,100 measured Ugi source rows
increases their overlap from 576 source-SMILES matches to 990 constitutional matches. The correct
training union contains 12,386 unique constitutional graphs, not 12,800 rows. There are no
conflicting component mappings among the newly collapsed identities. The constitutional component
universe contains 24 amines, 62 aldehyde-derived components and 9 isocyanides.

**Provenance safeguard.** Do not discard or overwrite the 13,376 source rows. Preserve each row in
`data/splits/phase1/ugi_l1_constitutional_provenance.csv.gz`, while exposing each constitutional
product exactly once to the model. Measured provenance takes identity priority on overlap without
duplicating training probability.

**Semantic gate.** The qualified atom-mapped transform exactly reconstructed and annotated all
12,386 products, reproduced all 1,100 measured semantic ledgers, and assigned one exact five-atom
core per product. Precursor origin and core membership are orthogonal. Core positions map to one
amine atom, one aldehyde-derived atom, two isocyanide-derived atoms and one assembly-introduced
amide oxygen. Every precursor-origin region is connected and has one role-matched core anchor.

**Positional decision.** A canonical core root or removable virtual root is used only for sparse
serialization. Ugi model semantics use exact origin labels, independent core membership and
adapter-defined anchors. Core and own-anchor graph distances condition chemistry only after a valid
topology exists; all-anchor distances and root depth remain ablations.

**Morphology decision.** The core-protected functional support roundtrips all 12,386 full products
and retains every core atom. Component-weighted branching is strongly localized: amine-derived
heads contain zero to three exterior branch nodes, aldehyde-derived components contain at most one
and 90% contain none, and all nine isocyanide tails contain none. Generate precursor-origin branch
budgets from unique-component distributions. Do not allow Cartesian product frequency or a generic
graph decoder to create repeated tail-junction chains unsupported by the training chemistry.

---

## 2026-07-30: Pass the sparse whole-lipid training smoke gates

**Decision.** Advance the sparse spanning-tree plus residual-closure product
backbone from representation feasibility to the first bounded GPU training
lane. Preserve the full 282-heavy-atom support and float32 deterministic
execution for the initial run.

**Computed result.** The real-corpus eight-record overfit gate reduced mean
total loss from 18.5959 to 8.2867 over 3,000 steps, a 55.44% reduction against
the frozen 50% requirement. The 40-step corpus smoke trained across all five
size buckets, processed 212 graphs at 421 graphs per second on CPU, and wrote
a checkpoint whose model-state SHA-256 is
`9c7f3a9c0cc15ec5e9a09f78f3e4d69ec95872d53954d2e96bcebdfe573d6960`.

**Failure found and corrected.** The first smoke attempt failed because its
parent-pointer budget could not admit one 282-atom molecule. The support was
not reduced. The budget was corrected, and a preflight invariant now requires
every batch configuration to admit at least one molecule at the declared
maximum.

**Adversarial correction.** The production denoiser is separate from the M0
feasibility probe. Residual closure edges now participate in deterministic
node message passing, and the current noised closure-bond state conditions the
closure prediction. This prevents ring-bearing heads from being represented
but then partially ignored by the production network.

**Full-corpus preflight.** Every one of the 464,265 R1 products passes the
exact production kekulization and support policy. All are valid and connected,
the maximum heavy-atom count is 140, the maximum residual-closure count is one,
and every R1 atom state is contained in the R0-derived vocabulary. This
hash-pinned audit is now a required input to training.

**Calibration monitoring.** Validation uses fixed, deterministic batches from
`R0_cal` only, with identical corruption noise at each checkpoint.
`R0_heldout` remains untouched. In the 40-step smoke, calibration total loss
decreased from 24.6711 at initialization to 19.4052.

**Recovery policy.** Training writes atomic latest, best-calibration, and final
checkpoints. Each checkpoint pins the config and data manifest and carries the
model, optimizer, sampling RNG, PyTorch RNG, counters, loss history, and
calibration history. A completed-checkpoint resume test restored step 40 with
the identical model-state SHA-256 and unchanged example and validation
histories. Full runs refuse to overwrite existing artifacts unless explicitly
authorized.

**Boundary.** This is an optimization and portability gate, not evidence of
generated-molecule validity or Ugi L1 generalization. Phase 1 remains free of
biological and synthesis-value guidance.

---

## 2026-07-30: Select the future HeLa campaign oracle independently

**Decision.** Preserve the equal-HeLa-and-RAW M0-07 freeze as an immutable
cross-endpoint audit, but do not use an endpoint average for a HeLa-directed
campaign. Rank the complete frozen candidate matrix using HeLa calibration
evidence alone. A later RAW 264.7 campaign requires its own endpoint-specific
freeze.

**Computed result.** The HeLa-only calibration ranking selects
`supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble`,
which is also the original shared winner. Its equal-scheme HeLa calibration
R-squared is 0.5042. Its HeLa outer-test R-squared of 0.2534 is descriptive
post-selection evidence and did not enter the ranking.

**Safeguards.** RAW metrics have zero selection weight. Outer-test metrics and
coverage gaps cannot alter the ranking. The candidate universe, held-component
schemes, and fitted models remain hash-pinned to the M0 artifact. This
endpoint-specific selection does not authorize biological guidance while the
frozen applicability policy abstains in every evaluated domain.

---

## 2026-07-30: Pass the deterministic CUDA preflight

**Decision.** Authorize the Phase 1 sparse whole-lipid training path on an
NVIDIA L4 under the frozen float32, no-mixed-precision and deterministic-kernel
contract. This closes the platform-specific portion of the adversarial
prelaunch gate.

**Computed result.** The production six-layer, 6,050,359-parameter model ran
three forward, backward and optimizer steps on CUDA 13.0 with PyTorch
2.11.0. Two independent seeded runs produced exactly identical loss histories
and final model-state SHA-256 values. An interrupted run checkpointed after
step one, restored model, optimizer, device generator and CUDA RNG state, and
then reproduced both the uninterrupted loss history and final model-state hash
exactly. The stronger checkpoint-resume path used 286,027,776 bytes of peak
allocated GPU memory.

**Representation coverage.** The device test included cyclic saturated and
unsaturated lipids, a branched aromatic example, an ester-bearing example and a
charged head. All three residual closures and all four constitutional graphs
round-tripped exactly.

**Boundary.** This test establishes CUDA kernel compatibility, finite
optimization, deterministic execution and exact checkpoint recovery. It does
not establish full-run convergence, generated-molecule validity, Ugi L1
generalization, biological activity or synthesis-route closure.

**Artifact.** `results/phase1/product_cuda_preflight.json` records the complete
runtime, source hashes, losses, memory measurements and model hashes. Reproduce
with `make phase1-product-cuda-preflight`.

---

## 2026-07-31: Reject V3 endpoint morphology and require a lipid-scale V4 flow

**Decision.** Preserve the V3 checkpoint and diagnostics as a failed
architecture probe. Do not resume the 20,000-step run until a revised model
passes an early whole-lipid morphology gate. Validity and decreasing
cross-entropy are necessary but insufficient acceptance criteria.

**Observed optimization.** The recoverable V3 run reached step 3,000 after
149,126 examples. Its R0 calibration loss improved from 24.9694 at
initialization to 13.7747, with the best value at the saved step. The observed
R0:R1 mixture was 89.60:10.40 against the frozen 90:10 target.

**Observed endpoint failure.** In an uncurated 32-molecule, 32-step sample,
validity, connectedness, uniqueness and exact-train novelty were all 1.0.
However, inferred tail-region branching was 17.78% against 4.07% in R0_train.
The grid contained scattered heteroatom pendants and overly bushy hydrophobic
regions. Doubling reverse-flow resolution to 64 steps did not correct the
failure: tail branching remained 16.62%. Therefore, sampling discretization is
not the main explanation.

**R0 morphology evidence.** The frozen 10,591-molecule R0_train fold is
structurally diverse but organized. Tail-region atoms comprise 66.05% of atoms,
tail-region branching is 4.07%, and 91.60% of tail-region connected components
are path-like under the frozen generic region annotation. Head atoms are
strongly ring-enriched, nitrogen has median root distance 3, and oxygen has
median root distance 8. Carbonyl and two-single-bond bridge environments
account for 65,246 of 70,973 oxygen environments; terminal single-bond oxygen
accounts for 4,895.

**Architectural diagnosis.** V3 uses six local tree-message-passing layers, a
pooled global mean, and separate terminal prediction heads for atoms, regions,
parents and bonds. It has no explicit root-distance state or global region
coherence objective. This is sufficient to match several one-site marginals
while failing to coordinate long hydrophobic arms and localized functional
groups.

**V4 direction.** Preserve whole-molecule atom-level generation and discrete
flow matching. Add a generic lipid morphology level that represents the
connected root-aware topology, region, root distance, junction status, closure
structure and hydrophobic-arm continuity before atom and bond refinement.
Condition full atom and bond generation on this morphology using local
edge-aware processing plus long-range graph attention. Do not introduce named
head or tail fragments, a fixed component vocabulary, or hardcoded tail
templates. Branching, unsaturation, rings and novel heads remain generated
variables.

**Required gate.** V4 must be evaluated against R0 on tail pathness, regional
branching, root-distance profiles, region connectivity, oxygen-environment
localization, ring localization, validity, diversity and novelty. A setting
that improves validity while retaining V3-like pendant or branching failure
does not advance to full training.

**Sampler defect audit.** The shared pointer R-star transition initially
allowed probability to leak onto masked parent targets. This could expose the
denoiser to future-node or padded-node parents during reverse flow. The update
now zeros transition rates outside the variable-specific candidate set and has
a deterministic regression test. Resampling the saved step-3,000 V3 checkpoint
after this correction did not rescue morphology. In the matched 32-molecule
sample, tail branching was 19.46%, tail path-like component fraction was
76.19%, and terminal single-bond oxygen was 31.54%, against R0_train values of
4.07%, 91.60% and 6.90%, respectively. The pointer defect was real, but the
remaining failure confirms that V3 lacks sufficient lipid-scale organization.
The source correction also invalidates the earlier V3 CUDA preflight as an
authorization artifact. Its historical measurements remain intact, but a new
GPU run would be required before making any claim about the corrected V3
source. The explicit invalidation record is
`results/phase1/product_cuda_preflight_v3_invalidation.json`.

**V4 topology representation.** Replace independently generated parent
pointers with the exact offspring sequence of the frozen canonical
breadth-first tree. The per-atom offspring counts are generated by discrete
flow. A dynamic program samples from the model logits conditioned on the valid
breadth-first queue language, then reconstructs the unique parent sequence.
This remains atom-level whole-graph generation and introduces neither named
fragments nor a component vocabulary.

**Bounded R0 gate.** On a deterministic 64-record, at-most-64-atom R0 subset,
the 3,000-step offspring-flow probe produced branching of 12.25% against
11.69%, leaf atoms of 18.41% against 17.21%, and mean maximum root distance of
16.74 against 18.42. A separate conditional region flow on settled trees
reproduced head, interface and tail organization. In the composed
256-morphology endpoint, tail atom share was 60.27% against 60.39%, tail
branching was 2.79% against 2.39%, and path-like tail components were 95.78%
against 96.55%. Both structured stages reported zero terminal repair. This
passes the bounded architecture gate but not the full-corpus product gate;
closures and atom/bond chemistry remain unimplemented in V4.

---

## 2026-07-31: Select a BFS spanning tree with preorder serialization for V5

**Decision.** Use a deterministic polar-root breadth-first spanning tree,
serialize that fixed tree in depth-first preorder, and generate the resulting
offspring word with exact tree-language constraints. Generate the `K` sparse
closure edges directly as endpoint pairs. Do not generate parent pointers or a
closure-stub degree sequence in the primary model. Closure stubs remain an
ablation only.

**Canonicalization gate.** The product graph is first normalized to the
declared stereochemistry-free constitutional identity. All 15,229 accepted R0
graphs reconstructed exactly. The initial BFS audit reproduced the complete
serialized state under 30,458 of 30,458 direct random atom permutations,
including 27,982 permutations of graphs with nontrivial symmetry. A later
three-way audit reproduced every graph and serialized state under one further
permutation of every record for all three representations.

**Measured traversal tradeoff.** Breadth-first serialization had the shortest
normalized closure sequence span, 0.049, but interleaved Ugi component origins:
the mean origin transition fraction was 0.738 and the mean number of runs per
origin label was 8.91. Pure depth-first discovery and serialization improved
those values to 0.088 and 1.275, respectively, but increased normalized closure
span to 0.228. The selected hybrid retained the BFS tree's mean closure tree
distance of 5.212 while reaching origin transition 0.090, 1.30 runs per origin
label and closure span 0.166. Its held-out fixed-bigram cross-entropy was 1.005
nats per joint region and offspring token, close to pure DFS at 1.002 and below
BFS at 1.037.

**Selection-fold correction.** The initial comparison above used
`R0_heldout` for the fixed-bigram evaluation and summarized structural metrics
over all R0 records. That is not an acceptable architecture-selection policy.
The authoritative V2 comparison trains the fixed bigram on `R0_train`, selects
on `R0_cal`, and reserves `R0_heldout` for deterministic exact-reconstruction
and permutation-invariance checks only. Ugi-origin continuity uses exact
semantic products in `R0_train`. The V2 artifact supersedes the numerical
selection values in the preceding historical paragraph.

On the corrected calibration fold, the hybrid's fixed-bigram cross-entropy is
1.077 nats per token versus 1.076 for pure DFS and 1.116 for BFS. Unlike pure
DFS, the hybrid preserves zero root-distance stretch because its spanning tree
is breadth first. Pure DFS has mean per-atom stretch 0.354 and mean maximum
stretch 3.377 atoms. The hybrid also improves mean closure tree distance from
5.077 to 4.859 and normalized closure span from 0.226 to 0.137 relative to pure
DFS, while Ugi-origin transition remains nearly tied at 0.090 versus 0.088.
This supports the hybrid as the lipid-morphology choice rather than selecting
the numerically smallest token cross-entropy in isolation.

**Budget correction.** Head/interface and tail junction budgets are defined on
tree degree. Closure endpoints alter final molecular degree but do not create
new hydrophobic arms, so closure count, direct closure feasibility and coarse
valence capacity constrain them separately. Total atom count is derived from
the three regional counts rather than predicted by a redundant independent
head.

**Sparse-state and closure contract.** The stored representation contains an
offspring word, tree-bond labels and `K` lexicographically ordered closure
pairs. It contains neither a dense adjacency matrix nor stored parent
pointers. This gives `O(N + K)` stored state and structural decisions, not a
claim of end-to-end linear runtime. The cycle-rank choice must be masked to an
initially feasible tree and capacity state. Every admitted closure prefix must
retain an exact residual completion, with probabilities renormalized over only
those feasible next pairs. Bounded `K` limits search depth but does not by
itself make exact search cheap, so the implementation requires brute-force
agreement on small graphs and a worst-case `N = 282, K = 12` benchmark. This
guarantee covers coarse topology and capacity, not later exact bond-order or
aromatic feasibility.

**Why no weighted spanning tree.** Every bridge, including every acyclic tail
chain bond, belongs to every spanning tree. The observed choice affects cyclic
regions, and all audited canonical encodings are exact and permutation stable.
There is no current evidence that a hand-weighted chemical spanning-tree
objective would improve the representation enough to justify another source
of nonuniqueness.

**Boundary.** This freezes the representation contract for the bounded V5
morphology gate. It does not establish trained morphology quality, chemical
validity, biological guidance or synthesis-route closure.

---

## 2026-07-31: Freeze the Ugi-first morphology source and strict product split

**Strict partition.** Add a mutually exclusive `primary_product_fold` to the
12,386-product constitutional Ugi ledger. A product is training only when all
three precursor roles are training components, calibration when none is held
out and at least one is calibration, and held out when any role is held out.
The resulting counts are 4,362 train, 3,350 calibration and 4,674 held out.
The role-specific columns remain for protocol-specific evaluations.

**Core-anchored exterior representation.** Across all 95 unique precursor
components, each precursor-derived exterior has exactly one boundary bond to
its qualified Ugi-core port. Under the selected hybrid serializer all 12,386
products share one origin run order. Represent the Ugi topology as the fixed
five-atom qualified product core plus three jointly denoised atom-resolution
exterior trees and sparse within-origin closures. The generated morphology
program contains per-role exterior counts, junction budgets and cycle ranks;
component identities are provenance and sampler keys only and never model
state. This is discrete graph flow, not autoregressive atom appending.

**Branch source.** Estimate offspring noise sources and morphology-program
mass from unique components rather than Cartesian product rows. Use equal
precursor-role loss mass. Current support admits compact amine branching,
at most one aldehyde-derived exterior branch, and no isocyanide-tail branch.
The architecture retains bounded support, but the initial campaign assigns no
probability to repeated unsupported tail-junction chains.

**Leakage controls.** Topology denoising receives no exact atom state, bond
state or clean target graph distance. Origin layout is derived from a generated
program and the adapter-fixed core, not copied from a source product. Distances
are recomputed only after topology validation for chemistry realization.
Before the broad-pretrained arm is fit, remove exact held Ugi products from
R0/R1 per evaluation protocol. A strict unseen-component claim additionally
requires excluding broad records that contain the held component; otherwise
label the result label-free transfer to a held Ugi role.

**Chemistry boundary.** Separate topology conditions from chemistry targets.
Use explicit aromatic atom and bond states under the production V3 convention;
do not mix Kekulized targets with aromatic ones. Terminal decorations and exact
atom/bond/charge states are loss targets, never topology conditions.

---

## 2026-07-31: Make expanded Ugi training the production default

**Decision.** Train the first production generator on observed and exact
forward-enumerated Ugi products, expanded through additional chemically
qualified amine, aldehyde and isocyanide components. Broad non-Ugi pretraining
is optional and decision-gated rather than an automatic production arm.

**Rationale.** The current paper prospectively validates a Ugi-3 synthesis
program. Ugi products provide exact reaction-core, precursor-origin and L1
forward-consistency supervision. Transferable head, tail, branching,
unsaturation and degradability motifs can be realized as new Ugi-compatible
components without asking the production model to learn incompatible final
assembly cores. Broad pretraining may still help representation learning, but
it may also cause negative transfer and requires additional decontamination.
It is justified only by a measured Arm-A gap, not by framework ideology.

**Claim boundary.** This does not reduce FORGE to a finite library model. The
flow generates complete component subgraphs and product chemistry without
component IDs in neural state. The general framework claim comes from the
shared graph, route and adapter interfaces; the trained and prospectively
validated system remains explicitly Ugi-3-specific.

---

## 2026-07-31: Pass the first generated-topology Ugi composition gate

**Result.** A sparse closure scorer trained on 12 strict-training cyclic heads
recovered all 12 training closure sets, 13 of 14 calibration cyclic-head sets
and two of three novel held cyclic-head sets. The only miss was an unseen
bicyclic head, for which the model produced a different but topologically
feasible two-closure pairing. Cycle-aware terminal support removed
closure-infeasible offspring words without repair.

Composing the 100-step morphology smoke checkpoint, sparse closure scorer and
500-step chemistry smoke checkpoint generated 24 complete structures. Twenty-
two sanitized successfully and all 22 valid structures were unique. Only one
matched any of the 12,386 frozen Ugi products exactly; median nearest-neighbor
Morgan similarity was 0.695. Generated heavy-atom, ring and heteroatom
distributions remained within the broad ranges of the frozen Ugi corpus. The
two failures were aromatic-kekulization errors.

**Boundary.** This is an engineering smoke gate, not a model-performance
claim. The chemistry model used 96 coverage-selected training products and the
terminal decoder is categorical argmax without structural repair. Production
training must improve aromatic consistency, preserve diversity and pass the
frozen held-component and open-endedness evaluations.

---

## 2026-07-31: Decision-gate staged versus joint sparse chemistry

**Question frozen before selection.** Retain the exact core-anchored sparse
tree-plus-closure representation and compare two matched decoders: a staged
topology-then-chemistry model and a chemistry-aware joint sparse flow that
denoises offspring, atom and parent-bond states through one shared backbone.
The tree visualization is neither an acceptance nor a rejection criterion.
Selection requires matched folds, global morphology programs, decoder support
and quantitative chemistry/morphology metrics.

**Initial engineering evidence.** All 421 compact expanded-chemistry records
project exactly into the joint representation. A 500-step joint smoke reduced
fixed-calibration loss from 13.484 to 2.045 at step 400. On the same 24 global
morphology programs as the staged smoke, both arms produced 24 of 24 unique,
valid molecules under the same valence-constrained terminal support.

**Confound discovered.** The compact 421-product chemistry cover is not a
distributional training corpus. Its greedy component-coverage selection chose
an alkyne-bearing filler in every product: 421 of 421 references contain a
triple bond, whereas only 8.15% of all 1,486,415 expanded products do so
unweighted and 9.04% under the frozen family-balance weights. Consequently,
the smoke molecules cannot select the production architecture; apparent bond
artifacts partly reflect exemplar-selection density rather than model design.

**Action.** Preserve the 421 records as an interface and exact-coverage test.
Run the immediate architecture ablation on the existing 12,386-product Ugi
corpus, then build a provenance-stratified, family-weighted subset of the full
expanded enumeration for production. Uniform enumeration-row sampling remains
prohibited, observed/current products remain the realism anchor, and source
activity labels are not inherited by enumerated products.

**Clean-corpus smoke.** A joint model trained for 500 CPU steps on the original
12,386-product corpus selected step 300 by fixed-calibration loss (14.867 to
1.405). On the same 24 global morphology programs used by the earlier staged
smoke, it produced 23 of 24 valid molecules with 23 unique structures and one
exact corpus match. Median nearest-reference Morgan similarity was 0.611,
versus 0.695 in the historical staged smoke. This suggests useful joint
topology-chemistry coupling and greater exploration, but does not select the
architecture: the historical staged smoke used an unconstrained terminal
decoder, the joint smoke used valence-constrained support, and both used tiny
training budgets. Production selection remains gated on equal data, model
capacity, decoder, sampling programs, evaluation seeds and compute.

**Balanced production corpus.** The exact annotation build completed for
24,386 products: all 12,386 current Phase 1 Ugi products plus 12,000
deterministically selected expanded exact-forward products.  The frozen folds
contain 12,464 training, 3,800 calibration and 8,122 held-out products.  The
training sampler assigns equal mass to the current realism anchor and expanded
stratum, then balances precursor families within each source; component and
family identifiers remain outside neural tensors.

The original smoke corpus explains the visually linear samples: only 6.5% of
current products contain a carbon-branched aldehyde component and none contain
a carbon-branched isocyanide component.  In the selected expanded stratum,
30.6% of products contain a carbon-branched aldehyde component and 64.0%
contain a carbon-branched isocyanide component, with at most two carbon branch
points in an admitted component.  Balanced-corpus sampling is therefore the
first meaningful test of controlled tail branching.  Absence of branching in
the original-corpus smoke is not an architecture-selection result.

**Balanced smoke evidence.** A shared hash-pinned cache now materializes the
24,386 exact chemistry records and joint sparse projections once.  Observed
support maxima are 62 exterior atoms in total, 26 per component, three
children, seven junction units, two cycles, two attachments and seven terminal
decorations.  Both arms use those data-derived bounds.  The cache avoids
repeating several minutes of RDKit graph reconstruction for every run.

On a fixed 48-program calibration probe stratified by source and tail-branch
class, the 500-step joint smoke produced 47 valid, unique molecules and the
staged smoke produced 48 valid, unique molecules.  Neither arm produced a tail
branch run longer than two adjacent branch nodes; three of 96 joint tail
sequences and five of 96 staged tail sequences had a run of two.  Median
nearest similarity to all frozen Ugi products was 0.587 for joint and 0.627 for
staged, with zero exact matches in either arm.  Both smokes contain undertrained
chemistry artifacts and are not production-quality checkpoints.

These results retain the joint sparse flow as the leading production design:
its production-width model has 1.08 million parameters, compared with 2.99
million across the separate staged morphology and chemistry models, while its
smoke validity is comparable and exploration is at least as broad.  This is a
provisional engineering choice, not a performance claim.  Final selection
still requires matched production training, quantitative held-out evaluation
and identical program/decoder support.
## 2026-08-01 — Source-balanced matched generator smoke comparison

- The corrected fixed probe contains 48 calibration morphology programs: 24 from
  the current Phase 1 Ugi union and 24 from the expanded exact-forward stratum.
  The expanded half is balanced across linear, aldehyde-branched,
  isocyanide-branched and both-tail-branched programs.
- On these identical programs, both the chemistry-aware joint sparse flow and
  the staged morphology-then-chemistry arm generated 48/48 valid, unique
  molecules with exact global-program support and no terminal tree repair.
- Actual graph adjacency, rather than visual branch counting or preorder run
  length, shows a maximum connected tail-branch component of two nodes in both
  arms.  The prior pathological five-junction tail cascade is absent.
- Across all 24,386 balanced reference products, oxygen–oxygen bonds and
  cumulated carbon double bonds have zero prevalence.  Neither arm generated
  either unsupported motif on the corrected 48-program probe.  Nitrogen–
  nitrogen and triple bonds are not treated as generic artifacts because they
  occur in 10.94% and 12.07% of the frozen reference products, respectively.
- The joint arm remains the leading production design because it couples
  topology and chemistry in one sparse flow and uses substantially fewer
  parameters than the staged pair.  This smoke comparison is not a paper-level
  superiority result: final selection requires matched training and a larger
  held-component evaluation.
- Frozen comparison artifact:
  `results/phase1/ugi_matched_generator_smoke_comparison.json`.

### Larger held-program follow-up

- The same audit was expanded to 256 calibration programs: 128 current-union
  and 128 expanded exact-forward programs, with the expanded half evenly split
  among linear, aldehyde-branched, isocyanide-branched and both-tail-branched
  morphology classes.
- Before the declared chemistry-support mask, the joint arm generated 255/256
  valid molecules but emitted oxygen–oxygen bonds in 5/255 valid products.
  Oxygen–oxygen bonds are absent from all 24,386 frozen reference products and
  are outside the present Ugi lipid support; they are not interpreted as useful
  novelty.
- The valence-constrained terminal decoder now masks oxygen–oxygen adjacency
  while leaving topology unchanged.  Re-evaluation preserved 255/256 validity,
  removed all unsupported O–O motifs and retained 252 unique valid molecules.
- The staged arm generated 253/256 valid, all unique molecules.  Its sampler
  now records isolated terminal-support failures instead of aborting an entire
  evaluation batch.
- Joint validity is modestly higher, whereas the staged smoke checkpoint is
  closer to the reference heteroatom, logP and triple-bond distributions.  The
  joint checkpoint therefore remains viable but not yet production-complete;
  a wider family-balanced training slice is required before final selection.
- Frozen corrected comparison:
  `results/phase1/ugi_matched_generator_smoke_comparison_support_masked_256.json`.
- The detached L4 production run completed 7,500 steps and stopped under the
  frozen calibration-patience rule.  Step 500 remained the selected checkpoint
  (`127e531d...`), and its hash exactly matches the independently downloaded
  interim checkpoint.  Calibration degradation after step 500 therefore
  reflects real overfitting of the narrow 6,000-product expanded training
  slice; it is not a cloud-transfer or checkpoint-selection artifact.

## 2026-08-01 — Audit and correct the Ugi production support ceilings

- A hash-pinned capacity audit separates exact Ugi support from conservative
  capacity checks on source-extracted LNPDB hydrophobic components.  Capacity
  evidence is not interpreted as Ugi compatibility, route closure or
  biological-label transfer.
- All 424 admitted Ugi components fit the configured 32 exterior atoms, three
  children, seven junction units and two cycles per precursor origin.  All
  seven cross-platform motifs already converted into exact Ugi aldehydes also
  fit.  The four deferred secondary-alcohol motifs fail handle-conversion
  policy rather than model capacity.
- The same bounds cover 299/307 (97.39%) unique parsed LNPDB hydrophobic
  components and 98.78% after weighting by their occurrence in LNPDB lipids.
  The eight exceptions comprise seven unusually large multi-arm components
  and one steroid-like component with cycle rank four.  Every parsed component
  fits the child-count and junction ceilings, so ordinary branching is not the
  limiting factor.
- The previous 68-exterior-atom product ceiling was insufficient for two of
  112,386 widened-corpus products.  Across the complete 1,486,415-product
  expanded enumeration, 371 products (0.025%) exceed 68 and the observed
  maximum is 76.  Phase 1 v2 therefore uses a ceiling of 80 exterior atoms,
  which covers the complete frozen enumerated universe with four atoms of
  headroom.  Per-component and cycle ceilings remain unchanged; they are not
  expanded merely to absorb exceptional, currently non-Ugi LNPDB motifs.
- The initial audit remains a recorded negative result under the prior model
  contract.  A v2 audit must pass against the corrected 80-atom training
  contract before GPU training is launched.

### Corrected v2 gate

- The widened cache completed for all 112,386 products with 66,464 training,
  15,800 calibration and 30,122 held-out records.  Its observed maxima are 69
  exterior atoms per product, 26 per component, three children, seven junction
  units, two cycles, two attachments and seven terminal decorations.  The
  792,454,343-byte cache is pinned by SHA-256
  `b862b7a54c0cb325f0a962fea42a78138df1a81c88b9c0e86d56b6448e519c46`.
- The corrected v2 support audit passes.  The 80-atom ceiling covers all
  112,386 widened products and all 1,486,415 products in the complete frozen
  expanded enumeration; the latter has an observed maximum of 76.  No
  per-component branching or cycle ceiling was relaxed.
- A fixed 256-program calibration probe was frozen before v2 training.  It
  contains 128 current-union programs and 128 expanded programs.  The expanded
  half is balanced jointly across the four observed tail-branch classes and
  two novelty strata: 64 transferred-known and 64 bounded structural-expansion
  programs.  The probe hash is
  `61e3210ad3ecb5c507f9faa10b755e96adbb41ff59e48a15c9458036192d266b`.
- The first v2 run retains the 1.08-million-parameter joint sparse architecture
  and equal source-stratum sampling mass.  It changes only the evidence-backed
  support ceiling, widened data and regularization/training contract.  Serial
  checkpoints are frozen at steps 250, 500, 1,000, 2,000, 4,000, 6,000 and
  8,000; model selection remains calibration-driven and sample quality is
  evaluated on the fixed probe rather than inferred from training loss.

### Complete Ugi product-generator factorization

- The production Ugi generator is explicitly
  `p(M) p_theta(x,b | M)`, where `M` contains only per-origin exterior counts,
  junction budgets, cycle ranks and attachment counts.  `M` is sampled from a
  training-fold, source- and family-balanced program prior; precursor roles are
  recombined independently at this coarse level and no component identifier or
  stored molecular fragment enters the neural state.
- The joint sparse flow generates the exact offspring topology, atom states and
  bond states conditional on `M`.  A second morphology denoiser would duplicate
  the offspring channel and is therefore not part of the v2 production path.
  This remains full graph generation: the coarse program fixes neither a
  precursor molecular graph nor a catalog identity.
- The 250-step, reduced-width CPU smoke is an interface test, not a selected
  model.  On the novelty-stratified 256-program probe it produced 255/256 valid
  and 254 unique valid molecules without tree repair.  Validity was 128/128 for
  familiar programs, 64/64 for transferred-known programs and 63/64 for
  bounded structural-expansion programs.  Generated median heteroatom count
  improved from three at step 125 to four at step 250, versus six in the
  reference corpus; this remaining gap is a reason to run the full model, not
  evidence to relax chemistry support.
- The training-fold program prior is now a separate hash-pinned artifact
  (`eaaef78b...`).  It contains 86 amine-role, 19 aldehyde-role and 35
  isocyanide-role coarse states after aggregating the same source-stratified,
  family-raked product weights used for training.  The largest independent
  role combination is 76 exterior atoms and remains inside the 80-atom model
  support.
- A frozen 1,024-program unconditional draw (`b5109d19...`) has a mean of 35.6
  exterior atoms and spans 14–62 in this sample.  It contains 561 linear-tail,
  43 aldehyde-branched, 390 isocyanide-branched and 30 both-tail-branched
  programs.  These programs can be passed directly to the joint graph flow
  after checkpoint selection; they choose no component graph or catalog ID.
- The reduced-width 250-step smoke was also composed with 256 independently
  sampled prior programs.  The complete path produced 254/256 valid, all-unique
  valid molecules with no tree repair, confirming that fixed reference programs
  are not required for decoding.  It is not yet a credible production model:
  10 valid structures exactly matched the frozen corpus, median logP was 10.75
  versus 7.82 in the reference, and median heteroatom count was four versus six.
  These negative quality gaps remain explicit checkpoint-selection metrics.

### Select the sparse ring-closure module on novel-component calibration

- The closure dependency was re-audited against the widened cache before the
  production launch.  Its 87 strict-training cyclic components, 22 novel
  calibration components and 25 novel held-out components exactly match the
  widened 424-component universe; this is not an older or narrower ring
  corpus.  The 134 cyclic components contain 135 closure edges and occur
  primarily in amine-derived heads, with two cyclic isocyanide components and
  no cyclic aldehyde components.
- The previous 300-step smoke artifact retained the latest checkpoint even
  though novel-component calibration NLL was best at step 200.  This is
  recorded as an auxiliary checkpoint-selection defect, not interpreted as a
  failure of the sparse representation.
- Closure training now persists separate best and latest checkpoints and uses
  novel-component calibration mean NLL for selection, with exact-set recovery
  only as a tie-breaker.  Held-out cyclic components are evaluated once after
  the selected checkpoint is restored; they are not repeatedly exposed during
  optimization.
- The deterministic 4,000-step CPU run selected step 3,800.  It recovered
  21/22 novel calibration closure sets and 19/25 novel held-out sets exactly;
  train recovery was 86/87.  The selected checkpoint is pinned by SHA-256
  `a97507ac6a9eeba41d0cc351666cffdd21069d13bc78c0db671ab3a30c710b5d`.
  Exact recovery is a stringent reconstruction statistic, not a claim that a
  different feasible closure in a newly generated component is chemically
  wrong.
- Composing this selected closure scorer with the reduced-width step-250 joint
  smoke produced 256/256 valid molecules, 254 unique valid structures and no
  terminal tree repair on the frozen novelty-stratified probe.  The joint
  chemistry checkpoint remains intentionally undertrained; the composition
  result validates the production interface rather than molecular quality.

### Frozen v2 production preflight

- The complete repository test suite and static diff checks pass after the
  widened corpus, support audit, unconditional program prior, serial-checkpoint
  persistence and novelty-stratified sampling changes.
- The production execution is pinned as
  `ugi_joint_sparse_balanced_v2_1cf363a57343`: one Modal L4, four CPUs,
  24,576 MiB memory, deterministic float32, at most 8,000 steps and a four-hour
  hard timeout.  Evaluations occur every 250 steps, progress is committed at
  every evaluation and calibration-based early stopping cannot begin before
  step 2,000.
- The immutable preflight record is
  `results/phase1/ugi_joint_sparse_balanced_v2_launch_preflight.json` (SHA-256
  `720a12d3281ae50cb29b3a64a58ab865cc291b6a92211f3472ea2a934b64ebe6`).
  All local evidence gates pass, and explicit approval for the billable run was
  recorded on 2026-08-01.
- Modal app `ap-KeQjOOElm1aqlZWqv42L5K` was launched in detached mode.  After
  staging the frozen 756-MiB tensor cache, the L4 task became active and
  committed initial best and latest checkpoints to the persistent training
  volume.  The launch receipt is
  `results/phase1/ugi_joint_sparse_balanced_v2_launch_receipt.json`.  Model
  training and checkpoint evaluation remain incomplete until their artifacts
  are downloaded and hash-verified.

## 2026-08-01 - Complete v2 Arm A training and retain a checkpoint-selection gate

- The detached v2 Ugi joint sparse run completed 4,750 steps and stopped under
  the frozen calibration-patience rule.  Step 500 minimized the source-macro
  calibration loss at 2.8894.  Every downloaded serial checkpoint hash matches
  the hash recorded by the cloud result, so the local evidence is attributable
  to the frozen `ugi_joint_sparse_balanced_v2_1cf363a57343` execution.
- Complete molecules were generated on the same frozen 256-program probe at
  steps 250, 500, 1,000, 2,000 and 4,000.  Validity was 98.0-100% and each
  checkpoint produced at least 249 unique valid molecules.  Median generated
  heteroatom count improved from three at steps 250-500 to five at steps
  2,000-4,000, versus six in the 112,386-product reference corpus.  Median
  logP moved from 11.04 at step 250 to 8.37 at step 4,000, versus 7.82 in the
  reference corpus.
- The same series exposes a nontrivial realism-versus-reproduction tradeoff.
  Exact matches to the frozen Ugi corpus were 2, 2, 1, 9 and 44 across the five
  checkpoints, while median nearest-product Morgan similarity increased from
  0.625 at step 250 to 0.752 at step 4,000.  Later training therefore improves
  selected molecular marginals but increasingly returns to frozen products.
- No production checkpoint is frozen from loss or validity alone.  Steps
  1,000-2,000 remain the current Pareto candidates pending held-component and
  generated-component novelty evaluation.  The calibration-best step 500 and
  the high-validity step 4,000 remain retained evidence, not automatic model
  selections.
- The hash-pinned comparison is
  `results/phase1/ugi_joint_sparse_balanced_v2_checkpoint_series_audit.json`.
  The size-only conditioning arm has completed its CPU smoke and fixed-probe
  evaluation, but its billable full cloud run remains unlaunched pending the
  explicit approval recorded as absent in its immutable preflight.

## 2026-08-01 - Complete the held-component and actual component-novelty gate

- A frozen 512-program probe was drawn only from the strict heldout product
  fold and stratified by source, inherited component-novelty stratum, held
  precursor-role combination and tail-branch class.  Its SHA-256 is
  `6b16f867fb72e92222df25cbd510b3acc586a2dde38a16e7c4afe63c669bc630`.
  The probe supplies morphology programs, not component identities or product
  graphs.
- Generated amine, aldehyde and isocyanide precursor graphs are now recovered
  by the deterministic inverse of the fixed Ugi adapter.  The inverse was
  first gated against every admitted component: 358 reference products cover
  all 264 heads, 107 aldehydes and 53 isocyanides, with 424/424 exact precursor
  reconstructions.  This is exact adapter inversion, not unconstrained
  retrosynthesis.
- On the same 512 heldout programs, step 1,000 generated 511/512 valid products
  and step 2,000 generated 506/512.  Every valid product at both checkpoints
  yielded all three precursor graphs and reconstructed the identical product
  under the frozen forward Ugi transform.  Step 1,000 passed the frozen handle
  policy for all three components in 511/511 products.  Step 2,000 passed in
  503/506; its three failures were generated heads with three distinct amine
  handle sites, above the allowed one-or-two-site policy.
- Actual component identity, rather than the probe's inherited label, shows a
  real novelty-realism tradeoff.  At step 1,000, 429/511 products (83.95%)
  contain at least one precursor graph outside all 424 admitted components,
  compared with 404/506 (79.84%) at step 2,000.  These are structural
  component-novelty counts only; they do not imply that the complete product is
  novel, route-closed or experimentally synthesizable.
- Step 2,000 better fits the complete 30,122-record heldout fold under the same
  deterministic one-draw corruption audit: record-weighted total loss is
  3.4344 versus 3.6623 at step 1,000.  It also moves generated logP toward the
  frozen corpus (median 9.82 versus 10.73; reference 7.82), but both checkpoints
  remain heteroatom-poor (median four versus six).  Step 2,000 exactly matches
  65/506 frozen products on this held-morphology probe, versus 23/511 at step
  1,000, so its realism gain is accompanied by more product reproduction.
- No production checkpoint is frozen.  Step 1,000 retains higher validity,
  handle qualification and component novelty; step 2,000 has lower heldout
  loss and improved descriptor realism.  The matched size-only arm and the
  final preregistered novelty/realism rule remain unresolved inputs to model
  selection.
- The complete nonselecting gate is
  `results/phase1/ugi_joint_sparse_balanced_v2_held_component_novelty_gate.json`
  (SHA-256
  `cd371b1f959a822a650b32daf28e107b972d921f51b37c2c274693e259a2d705`).
  It records checkpoint, cache, registry, reaction, closure and probe hashes,
  validates matched checkpoint inputs/configuration/source marginals, and
  preserves per-held-role loss and per-role component-similarity details.

### Frozen unconditional-program corroboration

- The same two checkpoints were also sampled on the previously frozen 1,024
  draws from the training-fold-weighted unconditional program prior (SHA-256
  `b5109d192f5d60435c00fc4aab2aa0e4b9dba8d1b00f669cb06f591652c9729a`).
  This removes the heldout-product morphology-program bias while continuing to
  supply no component graph or identity.
- The production-like draw corroborates, rather than resolves, the tradeoff.
  Step 1,000 produced 1,009/1,024 valid and 998 unique valid products; step
  2,000 produced 993/1,024 valid and 990 unique.  At least one structurally
  outside-catalog precursor occurred in 844/1,009 (83.65%) and 793/993
  (79.86%), respectively.  Exact frozen-product matches rose from 53 to 100.
- Step 2,000 again improves the chemistry marginals: median heteroatom count is
  five versus four at step 1,000 and six in the reference, while median logP is
  9.50 versus 10.51 and 7.82 in the reference.  Its lower unconditional
  validity arises mainly from 23 terminal-support failures versus one at step
  1,000; aromatic kekulization failures are eight versus fourteen.
- All valid samples at both checkpoints exactly forward-reconstruct from their
  inverse-Ugi precursors.  Frozen handle-policy pass rates are 99.80% at step
  1,000 and 99.60% at step 2,000; all failures are heads with three distinct
  candidate amine sites.  Again, structural component novelty is not route
  closure or a synthesis-success claim.
- The nonselecting unconditional audit is
  `results/phase1/ugi_joint_sparse_balanced_v2_unconditional_component_gate.json`
  (SHA-256
  `b40ef47dc714324fc6c3a449c77c0aa89218eb355070339b694f314808fc78d7`).
  It strengthens the evidence that step 1,000 is the more open-ended checkpoint
  and step 2,000 is the more distribution-conforming checkpoint, but selection
  remains deferred until the matched size-only architecture comparison.

### Size-only external launch remains blocked, not running

- The current immutable size-only preflight is
  `results/phase1/ugi_joint_sparse_size_only_v1_launch_preflight.json`
  (SHA-256
  `3c94c54c58208d7c2d0e29b4d631d6513b2884b013bb71423bfcde6bf1368791`).
  A detached Modal L4 launch was not submitted: no billable task started and no
  corpus or source bundle was uploaded.  Launch requires explicit approval of
  the Modal destination, frozen payload and billable L4 cost class; it must not
  be inferred from generic authorization to continue local model work.

## 2026-08-01 - Freeze architecture/checkpoint selection before opening size-only samples

- The previous section records the state at the time of the immutable preflight.
  It is now superseded operationally: after explicit user approval, Modal app
  `ap-W7FUzwBAjvqIWSJEpnvBjw` launched the size-only run reported by the cloud as
  `ugi_joint_sparse_size_only_v1_39a69b5bf465`.  The run artifacts remain
  quarantined from scientific evaluation until the run-name fingerprint,
  bundled source hash, config hash and every downloaded checkpoint hash are
  reconciled against the launch receipt.  Seeing that files exist on the volume
  is not treated as attribution or as model evidence.
- Before reading any size-only serial-generation output, freeze
  `configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v1.json`
  (SHA-256
  `803a470be205691f0f0af2634cad1a47788991b2688dc565e92a2c1b9fa913fd`).
  No existing machine-readable architecture-selection policy conflicted with
  this addition.  The already inspected held-component artifact remains
  explicitly nonselecting diagnostic evidence and is not represented as an
  untouched final test.
- Compare the full-morphology and size-only arms only on the previously frozen
  1,024-draw training-fold unconditional program prior (`b5109d19...`) at their
  common serial checkpoints.  The primary endpoint is usable open-ended yield:
  the fraction of all attempted draws that are valid, recover all three
  precursor graphs, exactly forward-reconstruct, pass all three frozen handle
  policies and contain at least one reconstructed precursor absent from the
  complete 424-component admitted catalog.  Failed draws remain in the
  denominator.
- Every candidate must first pass 95% validity, exact inverse/forward invariants,
  99% all-handle qualification, zero forbidden or declared-support-violating
  valid outputs, 98% uniqueness among valid products, a 15% ceiling on exact
  frozen-product reproduction and role-specific component-collapse safeguards.
  The deterministic inverse gate remains 424/424 exact (264 heads, 107
  aldehydes and 53 isocyanides).
- Molecular realism is a gate over complete empirical descriptor distributions,
  not selected medians.  For heavy-atom count, molecular weight, logP, rings,
  rotatable bonds and heteroatoms, require normalized Wasserstein-1 no greater
  than 0.60 per descriptor and 0.30 on average, KS distance no greater than 0.70
  per descriptor and 0.35 on average, and at least 55% generated mass inside
  each reference q05-q95 interval.  These intentionally broad floors allow
  open-ended generation but reject severe distribution displacement.
- Use 10,000 paired bootstrap resamples over the same 1,024 program indices
  (seed `20260801`).  Superiority requires both an absolute usable-yield
  advantage of at least two percentage points and a paired 95% percentile
  interval excluding zero.  If the primary difference is not statistically and
  operationally meaningful and both arms pass all floors, prefer size-only
  conditioning for weaker conditioning and parsimony.  Frozen-product
  reproduction and per-role component concentration remain safeguards; heldout
  loss is secondary and nonselecting because it can reward reproduction.
- Thresholds must not be rewritten after size-only outputs are opened.  A
  material revision requires a new versioned policy, disclosure of the observed
  outputs and a newly frozen selection draw.

### V1 selection invalidation and independently frozen V2 correction

- The provisional v1 selector is invalidated before any production-generator
  freeze.  The v1 policy pinned the 12,386-row lineage assignment file while
  its descriptor-policy text named all 112,386 frozen production Ugi products.
  The provisional implementation followed the pinned smaller path, and its
  exact-product membership calculation also compared noncanonical sample
  strings with canonical reference strings.  These are evaluation-contract
  defects, not model findings.
- Preserve the complete provisional artifact at
  `results/phase1/ugi_architecture_checkpoint_selection_v1_invalid_reference_population.json`
  (SHA-256
  `5355f53b2e511f146779d75a587be5f11dea1abc354755d4ae76b35f9a876a54`)
  and its invalidation record at
  `results/phase1/ugi_architecture_checkpoint_selection_v1_invalidation.json`
  (SHA-256
  `4807e638ed21998a9fbc25a439cbf8b56a46c74fb3d76beb6aaa4a7df91843b3`).
  The provisional full-morphology step-2,000 choice may not be used as final
  selection evidence.
- Before producing or inspecting replacement model samples, independently draw
  and freeze 1,024 new programs at seed `20260802` from the unchanged
  training-fold program prior:
  `results/phase1/ugi_program_prior_v3_selection_samples_1024.json` (SHA-256
  `4980793deaf2f32d8d6b6b44751ad8661569704bf09b343f91b94e928bdd92f1`).
  This draw contains no component identity or graph.
- Freeze the corrected versioned policy at
  `configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v2.json`
  (SHA-256
  `1f40c32bd2c44699011a3a5282b33edd10e16992f73216ea1bfe770381c0fc53`).
  It hash-pins the v1 policy and explicitly overrides only the contradictory
  reference path, the independent draw and matched sampling seed.  Every
  numerical hard gate, collapse safeguard, descriptor threshold, uncertainty
  threshold and selection rule remains unchanged from v1.  The corrected
  evaluator and its regression tests are also hash-pinned before replacement
  sampling.
- The v2 selector must fail closed unless the two training results and every
  evaluated checkpoint embed identical path/hash records for assignments,
  semantic products, semantic atoms, atom vocabulary and prepared cache.  It
  must verify those hashes locally and against each cloud mounted-input
  manifest.  Product and precursor strings must be canonicalized before
  reference membership, uniqueness or component-distribution accounting.
- Generate both architectures at all five common checkpoints on the new draw
  with the same sampling seed.  Do not interpret interim output until all ten
  artifacts exist and the fail-closed selector applies the already frozen v2
  contract.

## 2026-08-01 - Separate upstream evidence closure from forward-verified dossiers

- Compose the 93-component structural-program ledger with the frozen current
  terminal-procurement snapshot and all 12,276 exact Ugi product
  decompositions.  Seventeen heads are accepted procurement terminals.  Of 24
  exact-source upstream component programs, 23 have every proposed leaf
  currently procurement-closed; the oleyl-isocyanide program remains open at
  the exact oleylamine leaf.  All 47 family-projected programs also terminate
  in currently procurement-closed leaves, while five heads remain unresolved.
- At product level, 1,734 structures have all three upstream components either
  accepted as terminals or supported by exact-source programs with terminal
  closure.  A further 6,698 have complete upstream leaf closure only after
  admitting one or more family-projected programs, and 3,844 remain incomplete.
  Exact L1 graph reconstruction is retained for every product.
- These counts do **not** establish a complete forward-verified synthesis
  dossier.  The component programs are source-reproduced or structurally
  projected, but transformation-specific deterministic forward verification
  has not yet been executed for every upstream step.  Family projection is not
  exact route evidence, and neither L1 reconstruction nor upstream closure is
  experimental synthesis success.  Accordingly, the number of complete
  forward-verified dossiers remains zero.
- The conservative census is frozen at
  `results/phase1/ugi3_dossier_coverage/result.json` (SHA-256
  `1a4b74b552ecd293966f309cf3d127ee8e8d52b389f5edf94bd84760bda93333`).
  Its component and product ledgers preserve the evidence tier and exact
  remaining gap for every record.  The next route implementation target is
  transformation-specific forward verification, not additional indiscriminate
  LNPDB mining.

## 2026-08-01 - Qualify exact-source-only upstream graph transforms

- Add a separate, hash-pinned exact-source-only registry for the four observed
  upstream graph transformations.  The registry is
  `configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json` (SHA-256
  `a8afb8a6164c4eb149c8d58f991db9a94405a89cf46719a5970df9f2d322bd0e`)
  with one role contract per transform.  The frozen source-route artifact and
  every variant are independently hash-pinned by the gate config.
- Require the registry to declare exact-source scope, match each binding to its
  source transformation and observed pair count, and preserve explicit false
  claims for general enumeration, substrate-scope extrapolation, conditions
  encoding and experimental success.  The graph transforms encode molecular
  change only; reported reagents, solvent, temperature and workup remain route
  evidence metadata.
- All 46 extracted steps now reproduce exactly one unique canonical expected
  product: 15/15 esterifications, 17/17 primary-alcohol oxidations, 7/7 amine
  formylations and 7/7 formamide dehydrations to isocyanides.  Thus all 24
  exact-source component programs pass deterministic step reconstruction.
  Multiproduct expected matches are classified as ambiguous rather than passed,
  and wrong-product, no-product, invalid-input and execution-error outcomes
  remain explicit ledger states.
- The correct claim is limited to deterministic reconstruction of these 46
  source-resolved pairs.  This does not establish general substrate scope,
  encode conditions, prove experimental success or by itself close complete
  product dossiers.  Reaction-family projections remain outside this gate.
- The portable result is frozen at
  `results/phase1/ugi3_exact_source_forward_verification/result.json` (SHA-256
  `1af7734a06d846cbe4d73a5db2013ea0b8a67f5d6b2221736d27bbf620d61f36`)
  and the 46-row ledger at SHA-256
  `830dc166f6eebca25cda460d8ce156b41860c6a1a45ba605a2356caf30eb6a38`.

## 2026-08-01 - Close exact-source computational L1/L2/L3 dossiers

- Compose the 46-step exact-source forward-verification ledger with the original
  93-component program ledger, all 12,276 exact Ugi L1 decompositions, the prior
  leaf-coverage ledgers and the frozen terminal/procurement snapshot.  Fail
  closed on missing, duplicated, ambiguous, mismatched or non-unique L2 steps
  and on every unresolved terminal leaf.
- Seventeen components are accepted terminals.  Twenty-three of 24 exact-source
  routed components have every L2 step uniquely forward-verified and every L3
  leaf closed.  The oleyl-isocyanide program has verified L2 steps but remains
  L3-open at its oleylamine leaf.  Forty-seven family-projected programs are
  L3-closed but remain explicitly non-exact; five heads remain unresolved.
- Exactly 1,734 of 12,276 products satisfy the complete exact-source
  computational-dossier contract: exact qualified Ugi L1 reconstruction plus
  three components that are either accepted terminals or exact-source,
  uniquely L2-forward-verified and L3-closed.  A further 6,698 products have
  family-projected support that is not exact-source, and 3,844 remain
  computationally incomplete.
- A complete computational dossier is not evidence that the exact product was
  synthesized and is not experimental success.  This audit does not establish
  isolation, yield, purity, formulation or biological function.  Procurement
  evidence remains tied to its frozen time-stamped snapshot.
- The final result is
  `results/phase1/ugi3_complete_computational_dossiers/result.json` (SHA-256
  `74d2cd6d6b4ba0175927ce5be689a79822e7c72aa6536425f72893fc2874b0ea`).
  The component and product ledgers have SHA-256
  `73f20af60ba3635a5cb5426d034c860e3596d9309d9749a232e378d39ca5fa5a`
  and `4cc614c70aae7ec1a2442748757eff13e6b808bef5b60b730858da4d0453adfb`,
  respectively.

## 2026-08-01 - Exclude heldout structures from final architecture selection

- Preserve all ten independently sampled v2 checkpoint artifacts, but invalidate
  v2 without selecting a production generator. Its intended descriptor and
  exact-reproduction reference included the 30,122-product component-family
  heldout fold, contradicting the frozen rule that heldout molecular structures
  remain untouched until architecture and checkpoint selection are complete.
  In addition, the hash-pinned v2 selector correctly failed before metrics when
  a metadata-bearing product-reference record reached a validator that accepts
  only a bare path/hash identity. No v2 decision artifact exists.
- The complete failure and sample provenance are recorded in
  `results/phase1/ugi_architecture_checkpoint_selection_v2_invalidation.json`
  (SHA-256
  `2c7d3dcb198661a7ecef6be2cb7becc7826a7d6841f1d4f22d0273ef1defb7fc`).
  Thresholds were not changed in response to generated metrics.
- Before replacement sampling, freeze a third versioned contract at
  `configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v3.json`
  (SHA-256
  `a10b4fbd3f1f4400cc9c4e02ae67a57f9b1a2486e3f28115f57a4bbfb7d73bb4`).
  It retains every numerical gate and the original selection algorithm, but
  restricts descriptor realism and exact selection-visible reproduction to the
  66,464 training and 15,800 calibration products. All 30,122 heldout products
  are excluded from the final v3 selector. Earlier held-component outputs remain
  explicitly nonselecting diagnostics, so later fold-level results are
  descriptive rather than an untouched final test.
- Freeze a new independent 1,024-program training-prior draw at seed
  `20260803` in
  `results/phase1/ugi_program_prior_v4_selection_samples_1024.json` (SHA-256
  `10f468743cfc6231c8546982bfddfd501d87c0bca507c61f043bffed436d3e32`).
  This draw contains morphology programs only, without component identities or
  molecular graphs. The v3 selector and its nine focused regression tests
  were hash-pinned before any v3 checkpoint sampling.

## 2026-08-01 - Audit route readiness across the 424-component production registry

- Extend the evidence census from the original 93-component virtual set to all
  424 L1-admitted production components using constitutional identity within
  each Ugi role. All 93 original components match exactly once. Structural
  admission, handle qualification, provenance and motif similarity remain
  explicitly ineligible for route-evidence promotion.
- Forty-one components are route-complete under the frozen computational
  contract: 17 accepted terminal heads and 24 exact-source routed components
  with unique forward reconstruction and L3 closure. The latter comprise the
  23 closed original exact-source programs plus one cross-platform aldehyde
  transfer whose exact alcohol-to-aldehyde transform was independently
  re-executed against the qualified source-specific registry.
- The remaining registry contains one exact-source L3-open isocyanide, 87
  family-projected components, 111 components explicitly outside current route
  support and 184 components with route evidence not yet assessed. The
  route-complete role counts are 17/264 heads, 18/107 aldehydes and 6/53
  isocyanides. No handle-only record is called route-complete.
- The smallest recurring gaps, ranked only by component count and without
  structure-based chemical inference, are ten isocyanides with no reported
  upstream route, 36 family-projected isocyanides, 38 aldehydes with no reported
  route, 51 family-projected aldehydes, 63 heads outside current route support
  and 184 heads not yet assessed. This ranking defines targeted evidence work;
  it does not authorize broad literature mining or imply product-level closure.
- The result is
  `results/phase1/ugi3_production_registry_route_readiness/result.json`
  (SHA-256
  `d748eae9c9373d2cd4b420c9e17fb21dfa214f355135f766ed3081934f9cbf4d`).
  The component and recurring-gap ledgers have SHA-256
  `722e42dda35e550fa211c95f66098edddee741af838a468a8931c49c3e0bc99d`
  and `13cf73023daed1d718e9f55dc79c63b35cc82b79543345fd28cf03d47de7030c`,
  respectively. Structural component novelty remains distinct from route
  closure, exact-product synthesis and experimental success.

## 2026-08-01 - Freeze the production product-plus-L1 generator under the v3 selector

- Run the final comparison only after freezing
  `configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v3.json`
  (SHA-256
  `a10b4fbd3f1f4400cc9c4e02ae67a57f9b1a2486e3f28115f57a4bbfb7d73bb4`).
  The reference comprises the 66,464 training and 15,800 calibration products
  only. All 30,122 heldout products are excluded from descriptor and exact
  reproduction calculations. Earlier held-component results remain descriptive
  and nonselecting.
- Sample all five common checkpoints from both the full-morphology and size-only
  arms on the same independently frozen 1,024-program draw, eight sampling
  steps, batch size 16 and seed 20260803. Mechanical provenance and common
  training-input identity pass for all ten artifacts.
- Only `full_morphology_program_conditioning:step_1000` passes every frozen
  gate. Steps 250 and 500 fail heteroatom KS; full-morphology step 2,000 fails
  exact forward reconstruction; full-morphology step 4,000 exceeds the exact
  selection-visible reproduction ceiling. The size-only checkpoints fail,
  respectively, heteroatom KS at steps 250 and 500, inverse decomposition at
  step 1,000, inverse decomposition plus handle qualification at step 2,000,
  and declared support, reproduction and handle checks at step 4,000.
- The selected fresh sample contains 1,007 valid products from 1,024 attempts
  and 994 unique valid products. Every valid product yields three reconstructed
  components, exact frozen Ugi forward reconstruction and three qualified
  handles. Eight hundred forty-seven attempts contain at least one component
  outside the 424-component catalog and satisfy the usable open-endedness
  contract (82.7148%). Thirty-nine of 1,007 valid products match the 82,264
  selection-visible products exactly (3.8729%). Mean normalized Wasserstein-1
  is 0.15837 and mean KS is 0.24000. No valid product violates declared graph
  support or the forbidden-substructure policy.
- Freeze the decision at
  `results/phase1/ugi_architecture_checkpoint_selection_v3.json` (SHA-256
  `fe90fc849bbc88c8b004ceebcb4aae95ef6ac8158b8e1a6220792750aac83be5`),
  the selected sample at
  `results/phase1/ugi_architecture_selection_v3/full_step1000/result.json`
  (SHA-256
  `9066554cbf5009f68fafd136fef795e3c81dd71a772223b45ac48049661df12a`)
  and the checkpoint at
  `results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt`
  (SHA-256
  `90f5f0bd3e41e2884f6588d9875db0b7bf34e5c5ea7fdc1f9d8565fba8c0c532`).
  The complete 16-artifact production identity is pinned in
  `results/phase1/ugi_product_l1_production_generator_v1.json` (SHA-256
  `c27352c11a2e210fd76a6e4ae510bbbb89e51c92c28514fd9b4693f82a50e68a`).
- This freeze authorizes unguided complete-product generation, deterministic
  inverse L1 component recovery and exact Ugi forward verification only. It
  does not authorize L2/L3 route completion, synthesis-value or biological
  guidance, a synthesis-success probability, non-Ugi empirical support or any
  experimental claim. Post-selection diagnostics cannot reopen the architecture
  or checkpoint decision.

## 2026-08-01 - Audit candidate eligibility without reopening generator selection

- Audit the selected 1,024-draw sample against only the 82,264 selection-visible
  products. Skip all 30,122 heldout rows without molecule evaluation. The audit
  is candidate eligibility and manual-review evidence only; it cannot alter the
  frozen architecture or checkpoint and evaluates no routes or biological or
  synthesis guidance.
- Correct the initial allene/cumulene query before interpreting the audit. The
  v1 pattern `[!#1]=[!#1]=[!#1]` also matched sulfone O=S=O motifs. Preserve the
  v1 result and its explicit invalidation, then replace the query with the
  carbon-centred `[!#1]=[#6]=[!#1]` and add a sulfone-negative regression test.
- Of 1,007 valid products, 157 contain an alkyne (15.59%; reference
  13,397/82,264, 16.29%) and 61 contain an N–N bond (6.06%; reference
  4,123/82,264, 5.01%). Aldehyde, peroxide and carbon-centred allene/cumulene
  counts are zero in both generated and reference products. These motifs are
  flags for chemical review, not automatic rejection rules.
- Seventy-eight valid products fall strictly below the reference 1st percentile
  for non-carbon-heavy-atom fraction (7.75%; reference 813/82,264, 0.99%). No
  valid product exceeds the 99th-percentile heteroatom count or fraction. This
  identifies a low-heteroatom tail for later eligibility review without
  changing the selected generator.
- All 1,007 valid products pass declared graph support, inverse component
  reconstruction, exact forward reconstruction and all three handle checks;
  zero contain a frozen forbidden substructure. The 17 invalid draws comprise
  16 molecule-sanitization failures and one terminal-support failure.
- Freeze the corrected audit config at
  `configs/model/phase1_ugi_candidate_eligibility_audit_v2.json` (SHA-256
  `6b9abb3d2a2597a1274ef646fcda05aa223f5e6c4808421d7c3e36a3cd049aaf`)
  and result at `results/phase1/ugi_candidate_eligibility_audit_v2.json`
  (SHA-256
  `2619322d5584b117cfcee1e83b79e7f6b08964e73b9261722fc1cadba8457474`).

## 2026-08-01 - Report the held-component program analysis as a descriptive stress test

- Freeze no new checkpoint, threshold or model-selection rule. The component-family
  fold and its first 512-program sample were inspected before the production
  checkpoint was frozen, so the result is explicitly **not** a pristine or
  untouched held-out estimate. It is a post-selection descriptive stress test of
  the already frozen step-1,000 product-plus-L1 generator.
- Revalidate every probe row against the 112,386-row frozen assignment ledger.
  Each source product is in the primary heldout fold and contains at least one
  held component family. The sampler receives only the four coarse morphology
  fields (node counts, junction budgets, cycle ranks and attachment counts), not
  component identities, component graphs or product graphs. Results are
  stratified across all seven observed combinations of held precursor roles.
- From 512 attempted held-family-associated programs, 511 products are valid and
  505 are unique. All 511 valid products yield three inverse-L1 components with
  qualified handles and reconstruct the generated product exactly under the
  frozen Ugi forward transform. Four hundred twenty-nine attempts (83.79%) meet
  the structural open-endedness definition by containing a component outside the
  424-component catalog while passing the same L1 checks. The frozen-product
  identity audit finds 488 products outside the complete 112,386-product corpus,
  18 exact training products and 5 exact heldout products.
- Preserve the single-corruption denoising stress metric over all 30,122 heldout
  records (record-weighted total loss 3.66227) only as a descriptive diagnostic;
  it is neither exact likelihood nor a checkpoint-selection statistic. This
  analysis does not establish recovery of a requested held component, non-Ugi
  generalization, L2/L3 route closure, synthesis success or biological activity,
  and it does not decide whether to launch broad pretraining.
- The threshold-free contract is
  `configs/model/phase1_ugi_product_l1_postselection_held_component_stress_v1.json`
  (SHA-256
  `07899965880df531773ff8587493b6570cd41e1e9ecaea653a757e9048f786e4`).
  The result is
  `results/phase1/ugi_product_l1_postselection_held_component_stress_v1.json`
  (SHA-256
  `cf2e6f34ecb86d9780925295cff3f11bc44c3d5c7b90b5556c0fa9777f991e16`).

## 2026-08-01 - Resolve exact structural provenance in the frozen selected sample

- Classify every reconstructed precursor occurrence in the frozen 1,024-draw
  selected sample by exact Ugi role plus canonical constitutional identity.
  Molecular similarity does not assign or promote provenance. The primary
  mutually exclusive strata are original current Ugi catalog component,
  admitted transferred-or-expanded known component, and graph absent from the
  frozen 424-component catalog.
- Of 3,021 reconstructed component occurrences, 1,053 match an original
  current-catalog component, 590 match an admitted non-original known
  component and 1,378 are absent from the catalog. The corresponding unique
  role-specific identities are 30, 61 and 519. The original current-catalog
  definition uses the registry's explicit `is_current_catalog` flag and
  contains 95 catalog entries; it is not the separate 93-component historical
  dossier subset used by the upstream route-evidence audit.
- The registry provenance supports a narrower known-component substratification:
  16 sampled occurrences, representing one exact component identity, carry
  the explicit `cross_platform_hydrophobic_transfer` source class; the other
  574 admitted non-original occurrences represent 60 expanded-known exact
  identities. No transfer status is inferred from motif or fingerprint
  similarity.
- At product level, 30 of 1,007 reconstructed products use only original
  current-catalog components, 130 contain at least one admitted
  transferred-or-expanded known component but no catalog-absent component,
  and 847 contain at least one catalog-absent component. These are structural
  provenance strata, not route, synthesis or biological claims.
- Compose the frozen 424-component route-readiness ledger only by the same
  exact role-identity join. Three generated product occurrences have all three
  components in static route-complete tiers, 157 remain within the catalog but
  contain at least one noncomplete tier, and 847 contain a catalog-absent
  component. Every one of the 1,378 catalog-absent component occurrences is
  labeled `not_assessed_missing_route_knowledge`, never chemically infeasible.
  This is static evidence composition and performs no route planning.
- Freeze the contract at
  `configs/model/phase1_ugi_postselection_provenance_audit_v1.json` (SHA-256
  `ffe6be8ecce694fecaf8baed8da8034fd67bc255a45a9dcea346a715e67f45f9`)
  and the result at
  `results/phase1/ugi_postselection_provenance_audit_v1/result.json` (SHA-256
  `1db7917885e8d2905eebe2faebdc40a92d8a3458bb5b47ed89ec372a7c7610cb`).
  The component and product ledgers have SHA-256
  `4fb0414f42eb37738ca9e26015084d3ab544fb6f7eea31c61073f14033723eb6`
  and `cb401b548aa88ed7895e2d04c6929bd1b17395557a97cb96b26841f6dd4d0d7d`,
  respectively. The audit is nonselecting and cannot reopen the frozen model.

## 2026-08-01 - Authorize the synthesis-routing and synthesis-guidance milestone

- The user explicitly authorized proceeding from the frozen product-plus-L1
  generator into synthesis routing and synthesis-value guidance.
- Preserve the frozen step-1,000 generator as the reproducible baseline. Any
  branching-motivated retraining is a separately versioned challenger and must
  pass the same validity, L1 exactness, diversity, novelty and distribution
  gates before replacing it.
- Implement in fail-closed order: freeze the structured pre-prospective
  synthesis-value and matched-budget contracts; add typed route assessments,
  caching and restartable-sampler plumbing; pass bitwise zero-guidance
  equivalence; qualify a bounded hybrid L2 planner; close a nontrivial generated
  L1/L2/L3 dossier set; then run the matched route-guided versus post-hoc sweep.
- Do not use raw route likelihood, static catalog membership or component
  novelty as synthesis-success probability. Missing route knowledge remains
  distinct from chemical incompatibility and budget exhaustion.
- Biological tilting remains disabled because every frozen M0-07 guidance
  domain currently abstains.

## 2026-08-01 - Require a tail-branching challenger before candidate lock

- Audit carbon-skeleton branching by exact precursor role in the frozen
  step-1,000 sample against the 82,264 selection-visible training and
  calibration products. A branch atom is a carbon with at least three carbon
  neighbors; an adjacent branch edge joins two such atoms directly.
- Among 1,007 reconstructed products, 27 aldehyde components (2.68%) and 63
  isocyanide components (6.26%) contain adjacent carbon branch atoms. The union
  is 87 products (8.64%). Neither tail-bearing role contains an adjacent branch
  pair in the 82,264-product selection-visible reference.
- Generated tail branch runs have maximum length two, so the audit does not
  support the visual interpretation of five consecutive chemical branch
  junctions. It does establish a local tail-junction-spacing shift that exact
  validity and graph uniqueness did not reveal.
- Preserve the frozen generator as the baseline, but require a separately
  versioned challenger sampling or training policy before prospective candidate
  lock. First test the already available exact offspring decoder with the
  corpus-derived branch-run constraint; retrain only if that challenger fails
  the frozen product/L1, diversity and distribution gates.
- Freeze the audit config at
  `configs/model/phase1_ugi_postselection_branching_audit_v1.json` (SHA-256
  `bb6c5428d231395ec85ed9ca6159f8ace2554ba4e98338c0efe3d009898560b9`)
  and the result at
  `results/phase1/ugi_postselection_branching_audit_v1.json` (SHA-256
  `fd2dc759ec972d88a373e40867a263d0f9f92b3b7b996a530cf18ade3236ee03`).
  This is a nonselecting morphology audit, not a synthesis-feasibility claim.

## 2026-08-01 - Treat the full tail-chemotype audit as a candidate-lock blocker

- Extend the branch-specific finding to the complete frozen tail-chemotype
  audit. Among the 1,007 aldehyde-component occurrences in the selected
  step-1,000 sample, 19 contain an ester-like carbonyl (1.89%) and none
  contains a carbon-carbon double bond. The corresponding selection-visible
  training-plus-calibration counts are 40,136/82,264 (48.79%) and
  27,028/82,264 (32.86%), respectively.
- The same audit reproduces the off-distribution adjacent-branch result: 27
  aldehyde occurrences (2.68%) and 63 isocyanide occurrences (6.26%) contain
  directly adjacent carbon branch atoms, compared with zero occurrences in
  either role among the 82,264 selection-visible reference products.
- These negative chemotype results coexist with the previously frozen
  product-plus-L1 validity and open-endedness results. They do not invalidate
  step 1,000 as the immutable reproducible baseline, but they do mean that the
  baseline is not ready for prospective candidate lock.
- Evaluate a separately versioned challenger against the same
  selection-visible tail-chemotype distributions and the existing validity,
  L1, diversity, novelty and distribution gates. Do not modify or replace the
  frozen baseline artifacts in place.
- The audit is descriptive and nonselecting. It does not inspect heldout
  component families and makes no inference about route closure, synthesis
  feasibility, formulation or biological activity.
- Freeze the audit config at
  `configs/model/phase1_ugi_tail_chemotype_audit_v1.json` (SHA-256
  `7953bbfece04f40711a4478d22f36166feed9761e4484f0b2eeebdedab75c011`)
  and the result at
  `results/phase1/ugi_tail_chemotype_audit_v1.json` (SHA-256
  `c781761c5052d2ef88aa93a43356c7f076c2d99b8a0ff13904feaf4424071e4b`).

## 2026-08-01 - Prioritize repeated generated-component L2 evidence gaps

- Compose the admitted 424-component registry, original 93-component dossier,
  production route-readiness ledger and frozen generated-component provenance
  by exact role plus canonical constitutional identity. The audit uses local,
  hash-pinned evidence only and performs no route planning or literature search.
- Registry route completeness remains 41/424: 17/264 amine heads, 18/107
  aldehyde body-tails and 6/53 isocyanide tails. In the 1,007 generated
  products, 649/3,021 component occurrences are statically route-complete:
  142/1,007 heads, 85/1,007 aldehydes and 422/1,007 isocyanides.
- At product level, 3/1,007 products are already statically route-complete, 116
  have exactly one noncomplete component, 408 have two and 480 have three. The
  audit identifies 593 unique noncomplete exact components, of which 239 recur
  at least twice, across 34 repeated graph-derived architecture chemotypes.
- Rank exact targets first by products whose only current gap is that exact
  component, then by generated-product reach. The two highest-ranked targets,
  `CCCCCCCCCCCCCCCCCC=O` and `CCCCCCCCCCCCCCCCCCCC=O`, each guarantee six
  one-gap product closures under the explicit counterfactual that the exact
  component becomes route-complete. Together they cover 12 such products. The
  top five targets guarantee 26; the top ten guarantee 40 and incidentally
  cover all gaps in 45 products.
- The leading reusable structural group is
  `oxoester_aldehyde_body_tail-b546ad97cfa93fe0`, a length-independent
  graph-derived aldehyde architecture containing 17 noncomplete exact
  components and touching 556 generated products, including 35 one-gap
  products. One admitted component with the same coarse architecture is
  route-complete, but this structural analogy does not transfer route evidence
  to any other exact component.
- Preserve role breadth in follow-up work. The leading isocyanide exact target
  is `[C-]#[N+]CCCCCCCCCCCCC` (68 generated products; two one-gap products),
  and the leading repeated head exact target is `NCCCN1CCCCC1` (39 generated
  products; two one-gap products). These are evidence-retrieval and bounded
  planner priorities, not asserted routes.
- No input record supports chemical incompatibility for any prioritized gap.
  Family-projected, exact-route-unreported, route-not-assessed, catalog-absent
  and exact-route/L3-open states remain separate. A priority rank is not a
  synthesis-feasibility or biological claim.
- Freeze the contract at
  `configs/route/phase1_ugi3_l2_coverage_priority_audit_v1.json` (SHA-256
  `2bd33c5749ecec6a313fe800ad0121e66f05566af0c625b0c06f523ade5b3cac`)
  and the result at
  `results/phase1/ugi3_l2_coverage_priority_audit_v1/result.json` (SHA-256
  `01e9cbb2f52dfb4ebe5b97221183fb1bc0a309d7ed52e4ff4a4f39d93def86ea`).

## 2026-08-01 - Establish the typed recursive synthesis-assessment foundation

- Add a fail-closed L2/L3 assessment contract with distinct terminal outcomes
  for complete closure, chemical incompatibility, outside declared support,
  missing route knowledge, budget exhaustion, invalid input and execution
  failure. Missing knowledge and exhausted computation must never be promoted
  to incompatibility.
- Every assessment carries a typed recursive route tree and an evidence-bearing
  trace. Exact-source expansion requires an exact substrate match, unique
  forward evidence and one forward product. Family-level projections may be
  retained as evidence but cannot close a route.
- Terminal closure additionally requires a current L3 availability assessment.
  Unavailable leaves are outside support; expired or unassessed availability
  remains missing knowledge.
- Freeze separate counters for logical planner calls, physical cache hits and
  misses, route expansions, product candidates, forward-verifier calls and
  elapsed time. Reaching any frozen search limit returns budget exhaustion,
  not a chemical judgment.
- Add a deterministic content-addressed cache keyed by canonical constitutional
  target identity, product context, planner/search/value hashes, L1/upstream
  registry hashes, variant and verifier hashes, L3 snapshot and access policy,
  software versions, identity and stereochemistry policy, and frozen budget
  limits and usage. Cache entries contain the complete structured assessment,
  are checksummed and cannot be silently overwritten by different content.
- This milestone establishes infrastructure only. It does not qualify a
  dynamic or learned route planner, estimate synthesis-success probability,
  promote family precedent to exact evidence, or alter the frozen generator
  and baseline artifacts.

## 2026-08-01 - Qualify the frozen exact-evidence-only Ugi route adapter

- Translate the 93 frozen Ugi component-program records into the typed
  recursive planner contract without performing route search or inference.
  Accepted terminal records and exact-source steps with one uniquely verified
  forward product are the only evidence allowed to close a branch.
- The diagnostic closes 40/93 admitted components: 17 accepted amine
  terminals and 23 exact-source L2 programs with closed L3 leaves. The other
  53 remain missing knowledge: one exact-source program has an unassessed L3
  leaf, 47 have family-projected evidence only and five have provenance but no
  assigned exact upstream program.
- Outcome counts reproduce the prior component dossier classifications exactly
  for all 93 records, including the prior complete-component flag. A first
  pass produces 93 deterministic cache misses and a second identical pass
  produces 93 cache hits with byte-equivalent structured assessments.
- L3 status is time-pinned to the frozen US procurement snapshot accessed
  2026-07-29 and expiring 2026-08-28. Current closure, explicit
  unavailability, expiry and unassessed status remain distinct and fail
  closed through the recursive assessment contract.
- This result qualifies evidence translation and diagnostic cache plumbing
  only. It is not a general dynamic planner, does not estimate synthesis
  success and does not authorize production synthesis guidance.

## 2026-08-01 - Do not promote the step-2,000 branch-spacing checkpoint

- Freeze a one-shot, independent two-checkpoint experiment before product
  sampling. The already selected full-morphology architecture, the new 1,024
  unconditional-program draw, chemistry seed, closure checkpoint, decoded-tree
  branch-spacing limits and all v3 validity, exact-L1, handle, diversity,
  collapse and descriptor gates are identical for steps 1,000 and 2,000.
- Both corrected-sampling candidates pass every unchanged v3 gate, have exact
  component recovery and exact forward reconstruction fractions of 1.0, and
  have zero decoded-tree branch-spacing violations. Terminal admission is
  reported separately without changing the raw-valid denominators: all 1,007
  raw-valid step-1,000 products and all 992 raw-valid step-2,000 products pass
  exact L1 terminal admission.
- The unchanged within-architecture selector chooses step 1,000. Its usable
  open-ended yield is 0.8164 versus 0.7354 for step 2,000; the paired absolute
  difference is 0.0811 with a 95% bootstrap interval of 0.0537 to 0.1094.
  Step 2,000 has better descriptor-distance summaries but is not noninferior on
  the frozen primary endpoint, so realism cannot override the primary result.
- Do not promote the step-2,000 checkpoint and do not alter the immutable v1
  production manifest. The experiment does show that the decoded-tree
  branch-spacing constraint itself is compatible with the selected step-1,000
  checkpoint and every unchanged gate; whether that corrected sampling layer
  can be versioned for candidate lock is adjudicated separately rather than
  inferred by rewriting this checkpoint-promotion rule after seeing outputs.
- Freeze the policy at
  `configs/model/phase1_ugi_branch_spacing_checkpoint_promotion_policy_v1.json`
  (SHA-256
  `3832e1504b01206d3a960da34b95eec649df938821c3bab9d3e5e34e329b5c8e`),
  the preflight at
  `results/phase1/ugi_branch_spacing_checkpoint_promotion_v1/preflight.json`
  (SHA-256
  `cbe1e9130f1b634c22f870946e9ab5afef263b08332124ed6b0bafb7d23c48c2`),
  the unchanged-selector result at
  `results/phase1/ugi_branch_spacing_checkpoint_promotion_v1/selection_result.json`
  (SHA-256
  `107c49defb80f9cd9b2b71431786da5c1e2f1ac77862b6eb5a3e297f98d03ac5`),
  and the fail-closed decision at
  `results/phase1/ugi_branch_spacing_checkpoint_promotion_v1/result.json`
  (SHA-256
  `59dfdc4547c07f17d35f4522972b507ad9a8a819e45cf7035785552446b419fa`).

## 2026-08-01 - Qualify bounded hybrid Ugi route search without promoting weak evidence

- Compose the frozen exact-evidence adapter with bounded exact-identity
  retrieval across all 424 admitted production components. Preserve the typed
  complete, missing-knowledge and outside-support outcomes, recursive L2/L3
  trees, exact forward verification, time-pinned L3 status, deterministic
  budgets and content-addressed cache.
- The exact-only baseline closes 40/424 components. Hybrid qualification adds
  exactly one component, the independently re-executed exact-source oxidation
  route for `jl-2024-b16-2-decyltetradecan-1-ol`, for a total of 41/424. The
  added route consumes one verifier call and closes only because its exact
  substrate route uniquely reconstructs the target and its terminal material
  has current frozen US procurement evidence.
- No general substrate-scope template is admitted. All 87 family projections
  and 189 provenance priorities remain missing knowledge; 106 components
  remain outside declared route support. None of the 53 missing targets in the
  original component universe newly closes. Five legacy readiness records are
  reclassified from outside support to missing knowledge under the newer typed
  exact-source taxonomy, without changing any route-complete flag.
- A first pass produces 424 deterministic cache misses and a second identical
  pass produces 424 cache hits with unchanged structured assessments. Route
  completeness agrees with the frozen readiness ledger for all 424 components.
- Family projection, motif similarity, provenance and handle qualification may
  propose or prioritize evidence work but cannot close a route. This is a
  bounded diagnostic, not a qualified general dynamic planner, not a synthesis
  success probability and not authorization to connect route values to
  production molecular sampling.
- Freeze the contract at
  `configs/route/phase1_ugi3_hybrid_search.json` (SHA-256
  `d13ab4be85ef26820718a673e9ad5ea2bf80f2ec4ebbff269cf41f21dffd690a`),
  the result at `results/phase1/ugi3_hybrid_search/result.json` (SHA-256
  `05905d2a5f7eee83cf5a7028d89bcb3e5374959789ecc7ed3c2a692ed75bcbbf`)
  and the deterministic assessment ledger at
  `results/phase1/ugi3_hybrid_search/assessment_ledger.json.gz` (SHA-256
  `2002a770ad3683aff3274b39de3fe327b1dfe2269eaf5224b1ae41231c184ea2`).

## 2026-08-01 - Admit four high-impact aldehydes using exact evidence only

- Perform a bounded primary-source and current-procurement audit for the five
  predeclared aldehyde priorities from the L2 coverage analysis. Admit exact
  identities only; do not infer closure from homologous series, family
  precedent or a generic alcohol-oxidation template.
- Admit undecanal and octadecanal as current direct-procurement terminals under
  item-level US supplier records. Independently admit exact primary-alcohol
  oxidation programs for octadecanal, eicosanal and docosanal only after local
  source acquisition, SHA-256 pinning, visual inspection of the reported
  procedure, exact reactant/product identity checks, unique frozen-transform
  forward reconstruction and current L3 closure of the starting alcohol.
- Abstain on heneicosanal. The bounded audit found neither current exact-product
  procurement closure nor an exact-substrate, product-characterized procedure
  satisfying the evidence policy. This abstention must not be interpreted as
  chemical incompatibility.
- Exact route-complete registry coverage increases from 41/424 to 45/424. In
  the 1,007 corrected step-1,000 products, complete three-component dossiers
  increase from 3 to 24: five products close through undecanal, six through
  octadecanal, six through eicosanal and four through docosanal. The resulting
  21-product gain is evidence-attributable and does not rely on template
  inflation.
- Do not promote a general primary-alcohol oxidation rule, route-success
  probability or production synthesis-guidance policy from these records. The
  procedures are exact alternatives for named products; all other substrates
  remain governed by their own evidence status.
- Freeze the evidence pack at
  `configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json` (SHA-256
  `533fcc9907ec5cf76054d2172c7e54cc68ee26cdf31911bca0e2f7caf5ff79eb`),
  the audit contract at
  `configs/route/phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json`
  (SHA-256
  `b271127faeb78eebcb00c3c23e5e8e0c810f1aa08ddd3c8587c3bb00df403e42`),
  the result at
  `results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/result.json`
  (SHA-256
  `40fae7dc1b3695a696217cb388a8a43bf9922f6a5604500fb2021552dc98714a`),
  the exact-route verification ledger (SHA-256
  `ea92af40ee0b1c52e7e92150e79e29582e5959bc6cd46c7ee5b04a1fd72fb1b3`)
  and the product-impact ledger (SHA-256
  `764d83c94561faaf99da1e48bbfa6a2f42ea7c60bf5520bb5ec2a8ddd0aff40e`).

## 2026-08-01 - Compose the targeted aldehyde evidence as a typed exact overlay

- Preserve the frozen 41/424 bounded-hybrid result and compose the newly
  admitted aldehyde evidence as a versioned, exact-identity overlay. Direct
  current-procurement records return typed terminals; eicosanal and docosanal
  return exact-source one-step route trees terminating at current frozen L3
  evidence. The independently recorded octadecanal route remains an unselected
  alternative because current direct procurement already closes that target.
- Reassess all 424 admitted registry components. Exactly four targets newly
  close: two through current exact procurement and two through recursive exact
  routes. Complete outcomes increase from 41 to 45; missing knowledge changes
  from 277 to 274 and outside support from 106 to 105. The latter distinction
  records that one newly evidenced identity was previously outside declared
  support rather than silently rewriting it as missing knowledge.
- Delegate heneicosanal and every unrelated identity to the unchanged hybrid
  source. Heneicosanal remains an explicit abstention. No family, analogue,
  motif, provenance or handle-only record can close a branch.
- Rebuild the parent hybrid assessment under the current environment. Its
  424-component assessment ledger is byte-identical to the frozen ledger. The
  sole result-metadata difference is recorded explicitly as the RDKit version
  change from 2025.09.6 to 2026.03.4; every other result field is required to
  remain identical.
- This overlay remains diagnostic. It is neither a qualified dynamic general
  planner, a synthesis-success probability nor authorization for production
  synthesis guidance.
- Freeze the overlay diagnostic contract at
  `configs/route/phase1_ugi3_targeted_exact_overlay_diagnostic_v1.json`
  (SHA-256
  `b073de4a2684f01558089bfd5176df283333630c5e9ac90a60ad3c4f5afeb165`),
  the result at
  `results/phase1/ugi3_targeted_exact_overlay_diagnostic_v1/result.json`
  (SHA-256
  `22d025e5c684e0bdc21b875dc1482fefb86cca3e7109b0b2e27c2971169b037b`)
  and its deterministic 424-component assessment ledger (SHA-256
  `b2fa938a14b2e44259ff1cac7efb027c07543e01fe4d31475d86c0b0f31bc06d`).

## 2026-08-01 - Close the piperidine head and preserve the C13-isocyanide abstention

- Reassess the two highest-impact remaining non-aldehyde identities. Admit
  `NCCCN1CCCCC1` as an exact current procurement terminal using the official
  Sigma-Aldrich US marketplace listing for preferred partner ChemScene LLC,
  product `CIAH987F216E`. The exact identity is resolved by CAS 3529-08-6 and
  InChI key `JMUCXULQKPWSTJ-UHFFFAOYSA-N`; multiple item-level SKUs reported
  five-day shipment. Distinguish this closing record from the separate legacy
  Aldrich 694134 listing, whose cart-dependent availability is not used.
- Time-pin the head listing to the 2026-08-02 assessment and 2026-09-01 expiry.
  The typed overlay overrides only the exact role-qualified constitution and
  delegates every other identity to the unchanged source.
- Abstain on 1-isocyanotridecane. The reviewed AGILE supplement reports exact
  C11, C12, C14, C16, C17 and C18 homologues under the two-step formylation and
  dehydration family, but not the exact C13 product. Neighboring-homologue
  interpolation remains nonclosing evidence and does not become an inferred
  exact route.
- Exact registry coverage increases from 45/424 to 46/424. The head occurs in
  39 of 1,007 corrected generated products and newly completes five dossiers,
  increasing complete product dossiers from 24 to 29. The unresolved C13
  isocyanide occurs in 68 products and is the sole remaining gap for four
  products after head closure; those four remain incomplete.
- This result remains diagnostic and time-dependent. It does not qualify a
  generic isocyanide synthesis template, estimate synthesis success or
  authorize production synthesis guidance.
- Freeze the evidence pack at
  `configs/route/phase1_ugi3_targeted_role_gap_evidence_v1.json` (SHA-256
  `0d6a8c1e8c45101df8329c3c99e15c11d6f70c6ae81ad1ece2ce23be9bae73da`),
  the audit contract at
  `configs/route/phase1_ugi3_targeted_role_gap_evidence_audit_v1.json`
  (SHA-256
  `a4c2f40eed36187e4acae4872394cd97890f7215e90a84d1aabeb5419cde2941`),
  the result at
  `results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/result.json`
  (SHA-256
  `2d4cb30d844b6dfafe7aafb8c2eaae001104cce3b5cd618fdc98586cf0bdfeca`)
  and the deterministic product-impact ledger (SHA-256
  `e32ad68ac7d8ec80589997337da7718bc3a152d2c8a3c902b9d8cb28ba4762c0`).

## 2026-08-02 - Admit two high-leverage commercial head terminals

- Admit isopropylamine (`CC(C)N`) and tert-butylamine (`CC(C)(C)N`) as
  exact-identity current procurement terminals using official Sigma-Aldrich US
  item records. Record identity, item-level availability, assessment time and
  an expiry requiring refresh before candidate lock. The browser-visible
  supplier records could not be archived through direct HTTP retrieval, so the
  limitation remains explicit rather than being hidden.
- Override only the two exact role-qualified identities and delegate every
  other component to the unchanged evidence stack. No generic amine route,
  supplier-family inference or chemistry template is admitted.
- Exact registry coverage increases from 46/424 to 48/424. Complete dossiers
  in the selected 1,007-product generator sample increase from 29 to 38; the
  two heads newly close five and four products, respectively.
- Keep the result diagnostic. Supplier status must be refreshed before any
  candidate lock, and the records do not authorize synthesis guidance.
- Freeze the evidence pack at
  `configs/route/phase1_ugi3_high_leverage_head_terminals_v1.json` (SHA-256
  `b0e9fdaa527839f650939ee4b247604cacd5672d4df5ea01a51236d97c335add`),
  the audit contract at
  `configs/route/phase1_ugi3_high_leverage_head_terminal_audit_v1.json`
  (SHA-256
  `2753c15906699a991ad50edd53771b981f57ee8d392f97c9b771de9f01272cbb`),
  the result at
  `results/phase1/ugi3_high_leverage_head_terminal_audit_v1/result.json`
  (SHA-256
  `8b5b5d7cbb697b8b275159bdf5b5b2ddd8bc318ac3b1a3b12fc47ba3a5c19f4a`)
  and its product-impact ledger (SHA-256
  `9786f877e648cd490e84131bc63a9c1951605af4c7c64d616a6cbc6e609fbc36`).

## 2026-08-02 - Freeze structured nonprobabilistic synthesis values

- Implement typed component and product synthesis-value records over the
  recursive route assessor. Preserve exact blocker taxonomy, forward state,
  weakest evidence support, route depth, terminal-leaf state and explicit
  unknown protection and purification burdens.
- Keep L1 forward consistency at the product level. A product is complete only
  if exact L1 reconstruction holds and every role-specific component route
  closes. Component closure never inherits L1 state.
- Do not create a scalar synthesis score or experimental synthesis-success
  probability. Serialized records contain explicit null fields for both and
  reject non-null values. Cross-axis ordering remains a Pareto relation;
  missing knowledge, outside support and incompatibility are not collapsed
  into a common negative class.
- Compose the frozen evidence stack over all 1,007 selected generated products.
  The 610 unique generated components yield 24 complete, 560 missing-knowledge
  and 26 outside-support values. Of these, 519 generated identities are absent
  from the registry and remain missing knowledge. Thirty-eight products have
  exact L1 plus three closed component branches; 969 remain noncomplete.
- The value layer is ready for diagnostics and controller plumbing tests, but
  sparse identity-concentrated support does not authorize production guidance.
- Freeze the audit contract at
  `configs/route/phase1_ugi3_synthesis_value_audit_v1.json` (SHA-256
  `f99a73ef7a5bdb8f2e5a0312eb467cf4712e0c58e95efc0909dc0db7092df806`),
  the result at
  `results/phase1/ugi3_synthesis_value_audit_v1/result.json` (SHA-256
  `a1137d14b876a3dc53167e28dbfb55b1fb54acd845d3cdffe37d25e6aa7d7f27`),
  the component ledger (SHA-256
  `7e16da6ed54609137917b4fb4c936db211662ec5c2ec92bff464624f4d95e297`)
  and the product ledger (SHA-256
  `324aca317d1a1a6262f05202d42d0c4e8b4f853779e5be3926e5f0bf60624cc5`).

## 2026-08-02 - Keep synthesis coupling blocked pending restartable sampling

- Audit the selected sampler path. No exact component identity exists at an
  arbitrary noisy categorical state; it becomes available only after
  constrained tree decoding, closure placement, valence-constrained chemistry,
  sanitization and component recovery.
- Do not score partial invalid graphs with the exact route assessor and do not
  relabel terminal route filtering as guided generation.
- Authorize only a behavior-preserving refactor into restartable state advance
  and deterministic terminal completion, followed by exact zero-guidance
  equivalence and fake-controller SMC tests. The first defensible exact-planner
  coupling is completion-rollout particle resampling before candidate lock.
- Keep production guidance blocked because only 24/610 unique generated
  components and 38/1,007 products currently close. An exact-only value would
  risk collapsing generation onto documented identities.
- Record the implementation boundary in
  `docs/PHASE1_SYNTHESIS_GUIDANCE_SAMPLER_AUDIT.md`.

## 2026-08-02 - Qualify the restartable zero-guidance sampler refactor

- Expose cloneable initialization, exact-step advance and constrained terminal
  finalization for the selected joint sparse flow. Preserve the existing public
  sampler as a compatibility wrapper and retain the monolithic implementation
  temporarily as a private qualification reference.
- Expose terminal molecule completion behind an explicit closure-generator
  state, including closure placement, valence-constrained chemistry,
  sanitization, component recovery and optional exact L1 verification.
- Audit the selected step-1,000 production checkpoint on 12 fixed morphology
  programs, eight flow steps, seed 20260802 and batch sizes 1, 4 and 12. Every
  generated offspring array, chemistry logit, decoration logit and hidden state
  is bitwise identical to the frozen monolithic schedule in all three batch
  partitions.
- Implement diagnostic Feynman--Kac/SMC controller mechanics with keyed random
  substreams. A fake two-particle value produces the analytic 0.25/0.75
  ancestry law; guidance strength zero returns identity ancestry without
  consuming a resampling seed.
- Treat these results as plumbing qualification only. No synthesis scalar,
  planner rollout, production synthesis guidance or biological guidance is
  enabled.
- Freeze the refactor audit contract at
  `configs/model/phase1_ugi_restartable_sampler_equivalence_v1.json` (SHA-256
  `93163d6841e8ec8723d6e3d68a57c973e5c06879cc4104aa1972a92b0f5d9c96`)
  and the result at
  `results/phase1/ugi_restartable_sampler_equivalence_v1/result.json` (SHA-256
  `dee2de2804e8d01039769376213e5270ad59175444f9cf1bff9c5f82e7630052`).

## 2026-08-02 - Add propylamine as a versioned exact terminal and retain the cyclohexylamine abstention

- Admit propylamine (`CCCN`) as an exact-current procurement terminal using
  the official Sigma-Aldrich US item record for product 240958. The page
  resolved the exact CAS, constitutional SMILES and InChI key and reported SKU
  240958-50ML available to ship that day. Direct HTTP archival failed twice,
  so the locally frozen extraction records that limitation and expires after
  seven days.
- Do not admit cyclohexylamine (`NC1CCCCC1`). Its exact Sigma-Aldrich identity
  was confirmed, but the indexed availability text named a ship date preceding
  the audit snapshot. Treating that stale date as current procurement closure
  would violate the frozen availability rule.
- Exact registry coverage increases from 48/424 to 49/424. Propylamine occurs
  in 22 of 1,007 generated products and closes four additional one-gap
  dossiers, increasing complete product dossiers from 38 to 42. Current
  generated-component closure is 25/610.
- Preserve the frozen synthesis-value v1 artifacts byte-for-byte. Apply this
  one-component delta in a new v2 audit rather than rewriting the earlier
  record. V2 remains typed and nonprobabilistic: scalar values and synthesis
  success probabilities are still null, and unknown protection and
  purification burdens remain unknown.
- Production synthesis guidance remains unauthorized. The support gain is
  useful but still too sparse and identity-concentrated to justify exact-only
  guided generation.
- Freeze the procurement evidence at
  `configs/route/phase1_ugi3_second_wave_head_terminals_v1.json` (SHA-256
  `660682af3068794a3e194ddc529f5bdc9877d49ecf3441cfad3bbdb2ce103358`),
  its audit contract at
  `configs/route/phase1_ugi3_second_wave_head_terminal_audit_v1.json`
  (SHA-256
  `4854b4769ef7e7fe0a9a39e6b8a6c690b9ec1cd1cde248e4d204bee974aa23c2`),
  its result at
  `results/phase1/ugi3_second_wave_head_terminal_audit_v1/result.json`
  (SHA-256
  `9a6763a2bd2154c3d065232cf17e50459a1bb0b8c511e98ff0f62a0aa9627431`)
  and its product-impact ledger (SHA-256
  `08b92b1ba54d975501d6acf0784be3128ceb2d203a2076eb865e799df2a98301`).
- Freeze the versioned synthesis-value contract at
  `configs/route/phase1_ugi3_synthesis_value_audit_v2.json` (SHA-256
  `73a4ceed0cfa45138696ab0789ff44b969ef50c52148e4f6c382014386b064f3`),
  its result at `results/phase1/ugi3_synthesis_value_audit_v2/result.json`
  (SHA-256
  `d498aeec5588f3b3791e67f3122c522f1e47a89418865e3a175c3d6498de111d`),
  its component ledger (SHA-256
  `95e691068a67156551427dfebd1a82368ac7e55dbcdcd7cec98c9e468e01b846`)
  and its product ledger (SHA-256
  `ace4bd653c4c8bcac5d88692d989cc1ce4eac744b9fea3975fb13905239cba1b`).

## 2026-08-02 - Qualify diagnostic fixed-budget terminal-rollout plumbing

- Add batch-row particle extraction with keyed rollout RNG state and ancestry
  application to the restartable sparse-flow state. Reject ancestry that would
  cross a frozen morphology-program group; the productive flow RNG remains
  unchanged by ancestry selection.
- Add a fixed-budget diagnostic rollout executor that accepts only sealed
  terminal candidates. Invalid products and valid products that fail exact L1
  are retained as distinct dispositions and never reach the value evaluator.
- In the frozen two-particle, two-rollout diagnostic, all four completions are
  recorded, while only terminal indices 0 and 3 are evaluated. The executor
  records two logical planner calls and two verifier calls. Repetition is
  deterministic under keyed substreams.
- Retain the exact zero-guidance identity and analytic 0.25/0.75 two-particle
  fake-value law. The audit does not aggregate a real synthesis value, call the
  route planner, authorize synthesis guidance or evaluate biological guidance.
- Freeze the audit contract at
  `configs/model/phase1_ugi_fixed_budget_rollout_plumbing_v1.json` (SHA-256
  `e9e2e3706b9ae7c3aeca0119ec6f88f49c2cd6c450aeb8528a0b20b1ed1644c5`)
  and the result at
  `results/phase1/ugi_fixed_budget_rollout_plumbing_v1/result.json` (SHA-256
  `5aa1684915f0ee7a2520f6ea559c69a321a6d2ed6286214d8b20e2380d115bfe`).

## 2026-08-02 - Admit archived current cyclohexylamine procurement evidence

- Replace the earlier stale-listing abstention only after archiving a different
  primary supplier's current item page. The Chem-Impex archive identifies
  cyclohexylamine by name and CAS 108-91-8, reports at least 99.9% assay, lists
  five concrete pack-size SKUs as `Ships Today` and exposes active product
  variants. The raw HTML is retained locally and hash-pinned.
- Admit only `amine_head / NC1CCCCC1` as an exact current L3 terminal. This is
  procurement closure, not evidence that a particular Ugi product succeeds.
  Candidate lock still requires a refreshed availability check.
- Exact registry closure increases from 49/424 to 50/424. Cyclohexylamine occurs
  in 17 of the 1,007 generated products and closes three prior one-gap dossiers,
  increasing complete generated-product dossiers from 42 to 45.
- Preserve v1 and v2 synthesis-value artifacts byte-for-byte. Version 3 changes
  exactly the cyclohexylamine component: generated-component closure rises from
  25/610 to 26/610 and complete product values rise from 42/1,007 to 45/1,007.
  Scalar values and synthesis-success probabilities remain null.
- The exact literature use of 1-isocyanopentadecane found during the same audit
  is nonclosing: the source reports its use and product characterization but not
  an exact preparation or current procurement record for the isocyanide itself.
- Freeze the supplier evidence pack at
  `configs/route/phase1_ugi3_third_wave_head_terminals_v1.json` (SHA-256
  `ec16ed0ec407354cae3efa58b68db935fa9c74aa4fef7988fd5871964896c6cb`),
  the audit contract at
  `configs/route/phase1_ugi3_third_wave_head_terminal_audit_v1.json` (SHA-256
  `5951ae7d0fca7896e611a48da68450aab63dd64e2fd853ff2e458d9b0ccfcabe`)
  and its result at
  `results/phase1/ugi3_third_wave_head_terminal_audit_v1/result.json` (SHA-256
  `e75d0ee41318640b70b0c7bd34cda409df786dcd6b140417b0ff13a039cada09`).
- Freeze synthesis-value v3 at
  `configs/route/phase1_ugi3_synthesis_value_audit_v3.json` (SHA-256
  `ddac5a9d658d3aac6412baf64c202bb31527104b48887dd4680e3e8ea88fcf6f`),
  with result SHA-256
  `75dceb84bef3f77694fcb3262b89d18ceb975e3d506df26bafda5b2803a5fcec`,
  component-ledger SHA-256
  `1e1b652b99d4560ede34a2ab95dd440eff6d02301955d3df1463441898957699`
  and product-ledger SHA-256
  `cedd42814b82f6e176b52bf9f8588ca9da728b57be5dd38f5fb8230cad18d2fe`.

## 2026-08-02 - Qualify bond-stochastic terminal decoding without changing sparse topology

- Investigate the selected generator's low tail-chemistry diversity with
  matched decoder experiments rather than attributing it to morphology by
  inspection. Preserve the initial fully stochastic decoder diagnostic as a
  negative result: sampling atom, bond and decoration states together recovers
  some rare bond modes but introduces semantic failures and overgenerates
  unsupported oxygen chemistry. Its result remains frozen at
  `results/phase1/ugi_terminal_decoder_challenger_v1/evaluation.json` (SHA-256
  `2dd03e1ac04ff9f225bb2acd16dd5bc618ea54ccb144318c17af14b0d32464ff`).
- Preserve the first bond-only challenger as failed under its absolute
  reference-absence gate; do not relax that gate after seeing its output. That
  experiment nevertheless establishes the causal candidate: the same
  morphology programs, productive-flow seed and decoded topology yield richer
  aldehyde bond-order chemistry when only terminal bond states are sampled.
  Its nonselecting result is
  `results/phase1/ugi_bond_stochastic_decoder_challenger_v1/evaluation.json`
  (SHA-256
  `fe146b6d4dc20fe1be539e9f196d333e69fe518965fd9709bda233641cbf1359`).
- Freeze a new independent confirmation contract before drawing 1,024
  previously uninspected morphology programs. Use program seed 20260809, flow
  seed 20260810 and terminal seed 20260812, with no retries or seed search.
  Compare masked argmax with bond-stochastic terminal decoding at temperature
  1.0 on the step-2,000 checkpoint. Verify that all matched rows have identical
  programs, offspring trees and topology metadata; only the terminal bond
  decoder differs.
- Both confirmation arms produce 987 valid products from 1,024 attempts
  (96.3867%), and every valid product has exact component recovery and exact L1
  forward reconstruction. The bond-stochastic arm is 98.9868% unique among
  valid products and reproduces 11.3475% of the 82,264 selection-visible
  products exactly, below the frozen 15% ceiling.
- Bond-stochastic decoding increases the aldehyde-tail C=C occurrence from
  4.7619% to 17.7305% while retaining ester-like carbonyl occurrence
  (30.2938% versus 30.1925%). Effective aldehyde chemotype count increases from
  89.97 to 117.00, and aldehyde feature-distribution L1 error decreases from
  0.6568 to 0.5119. Reference-absent-feature increases remain at or below
  0.102 percentage points, and all nine frozen confirmation gates pass.
- Conclude narrowly that sampled morphology is not the proximate cause of the
  suppressed tail bond-order modes in this matched experiment. This does not
  establish that the morphology parameterization is globally optimal. Qualify
  step 2,000 with bond-stochastic terminal decoding for a new production pool;
  retain atom and decoration argmax, the sparse topology representation and all
  branch constraints unchanged.
- Preserve the original step-1,000/argmax production manifest byte-for-byte.
  Freeze the versioned successor at
  `results/phase1/ugi_product_l1_production_generator_v2.json` (SHA-256
  `7fd0e763a1460290601d6accf31ecb96b2e9ebea325f2b2cdceb8174316a3059`).
  The independent evaluation is
  `results/phase1/ugi_bond_stochastic_confirmation_v1/evaluation.json`
  (SHA-256
  `f533ddc28c3c9f08b0828d750910e3f546331c068b4bf434984d5bd0ae11b0b9`).
  Confirmation molecules are evaluation-only and cannot be reused for
  prospective candidate lock. Synthesis-value and biological guidance remain
  subject to their existing independent authorization gates.

## 2026-08-02 - Freeze a fresh v2 product-plus-L1 evaluation pool

- Freeze a nonselecting 4,096-program draw before generation using program seed
  20260813, productive-flow seed 20260814 and terminal-bond seed 20260815. Use
  the versioned step-2,000 bond-stochastic production identity, eight flow
  steps, batch size 16 and the unchanged per-role adjacent-branch limits. The
  contract is
  `configs/model/phase1_ugi_product_l1_v2_fresh_pool_v1.json` (SHA-256
  `92544cab4e67c34b1d6ae122122322a7b5fae802347aad223cec0bed7ec7be22`),
  and the program draw has SHA-256
  `46537953d40ecb14704b0dde8c461b9e4c48769206a03417a630b5ae4f633860`.
- Generate 3,975 valid products from 4,096 attempts (97.0459%) and 3,912
  unique valid products (98.4151% of valid). All 3,975 valid products recover
  all three components and pass exact frozen L1 forward reconstruction. The
  frozen sample result is
  `results/phase1/ugi_product_l1_v2_fresh_pool_v1/sample/result.json` (SHA-256
  `859b160a0194197c54743842263021afb6bac9b6a7e11dc391727e576afb17f3`).
- Preserve the first audit invocation as a fail-closed mechanical
  invalidation: its evaluator read terminal-decoder metadata from the result
  root instead of `sampling.terminal_decoder` and produced no numeric result.
  Do not change the sample or scientific thresholds. The invalidation has
  SHA-256
  `a1d5603a3738ecb9b02ad40adf8f7bdb620a5d40939256707700f6c995ad2b40`.
- Under the corrected hash-pinned audit, 398/3,975 valid products (10.0126%)
  reproduce a train- or calibration-visible product exactly. Exclude all
  30,122 held-component rows from this calculation. Of the exact-L1 products,
  317 use only original Ugi components, 506 contain an admitted transferred or
  expanded known component without a catalog-absent component, and 3,152
  contain at least one component absent from the frozen 424-component catalog.
  Structural absence is open-endedness evidence, not route closure.
- The 3,975 exact-L1 aldehyde components comprise 723 unique constitutional
  identities and 447 graph-derived chemotype signatures, with effective
  occurrence-weighted chemotype count 138.82. Aldehyde C=C, ester-like
  carbonyl and carbon-branch occurrence fractions are 16.55%, 30.54% and
  7.22%, respectively; adjacent carbon-branch chains remain absent. The
  isocyanide components comprise 335 unique identities and 110 chemotype
  signatures, with effective chemotype count 28.08.
- Permit the pool to advance only to nonselecting oracle-applicability and
  recursive-route assessment. Do not select prospective candidates or enable
  synthesis or biological guidance. Freeze the corrected audit contract at
  `configs/model/phase1_ugi_product_l1_v2_fresh_pool_audit_v2.json` (SHA-256
  `da147ebe1be7584e62ec035e48b69d7f7f6f27294ab91ee45dc0d3ac0557e07e`)
  and result at
  `results/phase1/ugi_product_l1_v2_fresh_pool_v1/audit.json` (SHA-256
  `48f071e5776d956dc1f1bc99f1244c550f324f2d83097a54fde5a6be721a6f5e`).

## 2026-08-02 - Attribute tail-chemistry biases before proposing corrections

- Audit the frozen 4,096-attempt v2 pool against three distinct references:
  selection-visible product occurrences, unique selection-visible components
  and the admitted 424-component registry. Keep topology-program incidence
  separate from terminal atom and bond chemistry so a morphology change is not
  used to address a chemistry-decoder error, or vice versa.
- Aldehyde positive-junction mass is 6.9346% in the production program prior,
  7.4219% in the frozen program draw and 7.2201% among exact-L1 generated
  aldehydes. The 0.2017-percentage-point draw-to-product gap attributes the low
  branch incidence primarily to the morphology-program prior rather than the
  terminal chemistry decoder.
- Terminal chemistry, not topology, controls the audited unsaturation and
  oxygenated-tail discrepancies. Generated aldehydes contain C=C, C#C,
  ester-like carbonyl and ether motifs at 16.5535%, 22.3648%, 30.5409% and
  5.9371%, respectively. The corresponding selection-visible occurrence
  references are 32.8621%, 13.9873%, 48.7900% and 0%.
- The generated product pool contains alkynes at 22.7925%, compared with
  16.2854% of selection-visible Ugi products and 10.0466% of broad observed
  lipids. Alkynes remain reference-supported and are therefore a calibration
  target, not a forbidden motif. Carbon-centred allene/cumulene motifs occur in
  1.0566% of generated products and in neither reference corpus. Frequency
  alone does not authorize a new hard mask; preserve these as explicit
  eligibility and chemical-review flags pending a frozen selection policy.
- Freeze the attribution contract at
  `configs/model/phase1_ugi_chemistry_bias_attribution_v1.json` (SHA-256
  `6c1ee74f3e65cf6fbc24ac2e1b9664a4085ace42f17f52251e3bf9d2cdbba618`)
  and result at `results/phase1/ugi_chemistry_bias_attribution_v1.json`
  (SHA-256
  `c81bd392dfc3542ec46bbe5361677f24a38e51a88cf1f5e20ab651ffdafa6744`).

## 2026-08-02 - Build a component-family-balanced morphology-prior challenger

- Estimate a second morphology-program distribution by balancing unique
  components within role and component family, then combine it in a fixed
  50/50 mixture with the frozen product/family-raked realism prior. Component
  identities and component graphs are estimator metadata only and never enter
  the sampled morphology program or neural state.
- The mixture increases positive-junction mass from 6.9346% to 13.4673% for
  aldehyde-derived components and from 40.6140% to 63.1641% for
  isocyanide-derived components. It preserves a product-distribution anchor
  while countering occurrence-frequency dominance by a small number of common
  components.
- Freeze one 1,024-program diagnostic draw before chemistry sampling using seed
  20260816 and no retries or seed search. It contains 135 aldehyde-branched
  programs (13.1836%) and 642 isocyanide-branched programs (62.6953%). This is
  morphology-prior evidence only; it does not replace the selected production
  generator.
- Freeze the prior contract at
  `configs/model/phase1_ugi_program_prior_challenger_v1.json` (SHA-256
  `a42fee8f316dfb214f6ff5220e3351f1b833485a387ab08cbf119fec32d6507a`),
  the prior result at `results/phase1/ugi_program_prior_challenger_v1.json`
  (SHA-256
  `2762cc6bee8ea2906af8de6987cdd03e82c06f0ae1e05632ec804d7a9a906569`),
  the draw contract at
  `configs/model/phase1_ugi_program_prior_challenger_draw_v1.json` (SHA-256
  `31e9a532e14d72b21ec747b1d3a4f5e073dc114fb4a7327f680632666caf0994`)
  and the draw at `results/phase1/ugi_program_prior_challenger_draw_v1.json`
  (SHA-256
  `97ffd2da35f215837966b392148c41c965a76f3eccdc6ecdb3cdbedf7be174fc`).

## 2026-08-02 - Reject step 4,000 as a production replacement

- Test the step-4,000 checkpoint under the already qualified bond-stochastic
  terminal decoder before combining it with the morphology challenger. The
  frozen draw yields 989 valid products from 1,024 attempts (96.5820%), with
  100% exact-L1 reconstruction and 976 unique valid products.
- It reproduces 232/989 frozen Ugi products exactly (23.4580%), exceeding the
  preregistered 15% ceiling. Retain it only as a chemistry-learning diagnostic;
  do not replace the step-2,000 production checkpoint.
- Freeze the diagnostic contract at
  `configs/model/phase1_ugi_step4000_bond_stochastic_diagnostic_v1.json`
  (SHA-256
  `1767a887b86f4bdb707b52e8b4a471c21cf1bce16abe9624333c2c3f03120b87`)
  and result at
  `results/phase1/ugi_step4000_bond_stochastic_diagnostic_v1/result.json`
  (SHA-256
  `b2ff21bb2b84e3553857f8b73e5b725e29579e919e48ba4e793d62e95cec2a21`).

## 2026-08-02 - Reject the combined chemistry-plus-morphology challenger

- Freeze a single 1,024-attempt combined diagnostic before sampling: the
  component-family-balanced 50/50 morphology prior, step-4,000 chemistry
  checkpoint, qualified bond-stochastic terminal decoder, flow seed 20260817
  and terminal seed 20260818. No retries or seed search are permitted.
- The challenger yields 995 valid products (97.1680%), all of which recover
  exact L1, and 992 unique valid products (99.6985%). Aldehyde branching reaches
  13.3668% and aldehyde C#C falls to 15.7789%, so the targeted morphology and
  upper-triple-bond checks pass.
- The challenger fails four gates frozen before inspection: exact frozen-product
  reproduction is 17.4874% (maximum 15%); carbon-centred allene/cumulene
  occurrence is 0.8040% (maximum 0.5%); aldehyde C=C is 19.6985% (minimum 20%);
  and aldehyde ester-like carbonyl occurrence is 35.6784% (minimum 40%). Validity,
  exact L1, uniqueness, product-alkyne and aldehyde-branch gates pass.
- Do not promote the challenger and do not relax or tune the failed thresholds
  after observing the result. Retain production generator v2. Use the result to
  motivate frozen downstream chemotype stratification and eligibility review,
  rather than repeatedly tuning terminal logits to the same diagnostic sample.
- Freeze the challenger contract at
  `configs/model/phase1_ugi_combined_chemistry_morphology_challenger_v1.json`
  (SHA-256
  `f41e5fc31eb82e08bf9f31cba523aed1b6046715d6bc8db4a38e19b2628d63ff`),
  its sample at
  `results/phase1/ugi_combined_chemistry_morphology_challenger_v1/result.json`
  (SHA-256
  `42370f97593e7686639b5f7ed953d52ab21639757c087a0591537ccf3e4e1cc5`),
  the evaluation contract at
  `configs/model/phase1_ugi_combined_chemistry_morphology_challenger_evaluation_v1.json`
  (SHA-256
  `9164978ba241dd8f15166f6610003decfdf9d07a87be054ed103a2b492d0b70e`)
  and its evaluation at
  `results/phase1/ugi_combined_chemistry_morphology_challenger_v1/evaluation.json`
  (SHA-256
  `affeae9713fd159dad9aa01458533fb39e6a8fb28b444621c1f214c86f9f074a`).

## 2026-08-02 - Transfer exact route values onto the fresh v2 pool

- Assess all 3,975 valid, exact-L1 products in the frozen fresh v2 pool using
  only exact role plus constitutional-component identities already owned by
  the authenticated hybrid and v3 synthesis-value ledgers. Do not infer a
  route from component similarity, provenance, handle qualification or a
  family projection. Treat every unmatched fresh component as missing route
  knowledge rather than chemical incompatibility.
- The pool contains 1,919 unique generated component identities: 40 are
  route-complete, 1,833 are missing knowledge and 46 are outside declared
  support. Across component occurrences, 3,711 are complete, 7,012 are
  missing knowledge and 1,202 are outside support.
- Exactly 147/3,975 product dossiers close all three L2/L3 branches; 802
  products have exactly two route-complete components, 1,666 have one and
  1,360 have none. The 147 closed products comprise 96 products using only
  original Ugi components and 51 using an admitted transferred or expanded
  known component. None of the 3,152 products containing a component absent
  from the frozen 424-component catalog is route-complete. Structural novelty
  therefore remains distinct from open-ended synthesis grounding.
- Preserve null scalar values and null synthesis-success probabilities. This
  audit does not plan new routes or authorize synthesis guidance. Freeze the
  contract at
  `configs/route/phase1_ugi3_fresh_pool_route_coverage_v1.json` (SHA-256
  `c51be038a819843959d47340b4cf55a2478663e03b2385b0b16be4a81d204d39`),
  the result at
  `results/phase1/ugi3_fresh_pool_route_coverage_v1/result.json` (SHA-256
  `16ea205e816e87a1daa3e8d4c765d7564f74c4fd887f8a7d44c62556705ac1d0`),
  the component ledger at SHA-256
  `2535046846b4f6293273960c46ca9dc873bd3b37acf3849fe7871f162f975c03`
  and the product ledger at SHA-256
  `53c91af25ac37be5de8436865ec610e020e0900b232ae844c4ad63b2cd742d3e`.

## 2026-08-02 - Freeze the fresh-pool one-gap L2 priority frontier

- Rank the unresolved exact component in each of the 802 products that already
  close two of three component branches. This is a deterministic route-mining
  priority audit, not a claim that any proposed route succeeds.
- Of the one-gap products, 723 are blocked by missing route knowledge and 79 by
  an outside-support component. The unresolved roles are 446 aldehydes, 232
  amine heads and 124 isocyanides. There are 328 unique missing-knowledge
  identities; a greedy occurrence-weighted frontier needs 13 identities to
  cover 25% of missing-knowledge one-gap products, 50 for 50%, 148 for 75% and
  256 for 90%.
- The leading exact targets are `[C-]#[N+]CCCCCCCCCCCCC` (32 one-gap
  products), `[C-]#[N+]CCCCCCCCCCCCCCC` (19), heptadecanal (19),
  `C#CCCCCCCCCCCCCCC=O` (17) and 1,1-dimethylhydrazine (16). Send
  missing-knowledge targets
  only to exact-source retrieval or a separately qualified template search;
  send outside-support targets to declared-support review rather than route
  mining.
- Freeze the contract at
  `configs/route/phase1_ugi3_fresh_pool_route_priority_v1.json` (SHA-256
  `e1714df2943d8d45cfdc166de5e11f3f72905d1f9c0ce081ba4aba5c6f08b4e5`),
  the result at
  `results/phase1/ugi3_fresh_pool_route_priority_v1/result.json` (SHA-256
  `ff63a51203d9a91704a3ff282edac6547cb78b46b3deb07e8153ac2bff42ddf5`)
  and the priority ledger at SHA-256
  `fd43ec738becfafe0f2aa282507dd2eb670a3f66af5f361b4e0665106894c5f4`.

## 2026-08-02 - Audit the fresh v2 pool with the frozen HeLa oracle without guidance

- Run the frozen role-aware D-MPNN ensemble on all 3,975 exact-L1 fresh-pool
  products using the HeLa-only campaign endpoint. Recreate the exact recorded
  inference environment (RDKit 2025.09.6, scikit-learn 1.8.0 and PyTorch
  2.11.0) in an isolated environment after the production checkpoint correctly
  rejected the newer project environment. Do not weaken the checkpoint's
  software-version guard.
- The pool contains 12 known-component combinations, 19 unseen-head products,
  354 unseen-aldehyde products, 26 unseen-isocyanide products, 788 unseen-head
  plus aldehyde products, 49 unseen-head plus isocyanide products, 593 unseen-
  aldehyde plus isocyanide products and 2,134 products with all three
  components unseen by the measured oracle corpus.
- Among the 147 route-complete products, 10 use oracle-known components in a
  novel combination; the remaining 137 occupy one- or multi-component unseen
  domains. Among the 802 one-gap products, only two use oracle-known
  components. These counts quantify the mismatch between expanded synthesis
  support and the much narrower biological-label support.
- Every candidate retains `guidance_action=abstain`; no raw ensemble mean is
  used to rank or select molecules. The prediction distributions are
  descriptive diagnostics only and do not reopen biological tilting.
- Freeze the contract at
  `configs/bio/phase1_ugi_fresh_pool_oracle_audit_v1.json` (SHA-256
  `8569f091e02aa76a976e3a07243ff7c82e1dcdd5b7dc78d1a04b973361d49581`),
  the result at
  `results/phase1/ugi_fresh_pool_oracle_audit_v1/result.json` (SHA-256
  `a0762611449ea1f97495794bfada811309e4c912ea34771620b491aaa41e94de`)
  and the candidate ledger at SHA-256
  `7c8425cad54553383a8c7ddbf476f10501e4e6093064ef4ab927723f0805a18e`.

## 2026-08-02 - Audit fresh-v2 tail-axis and joint-chemotype coverage

- Compare the frozen nonselecting 3,975-product fresh v2 pool against only the
  selection-visible train/calibration component catalog, while keeping the
  original 12,276-product corpus descriptive. This audit occurs after production
  generator v2 was frozen and cannot alter checkpoint selection.
- The aldehyde-derived component samples cover every selection-visible state of
  carbon count, carbon-path diameter, branch-atom count, C=C count, C#C count,
  ester-like carbonyl count, ether count, ring count and oxygen count. The
  isocyanide-derived samples cover every audited marginal state except the
  smallest four-carbon and carbon-diameter-one states.
- Marginal support does not imply complete joint-distribution recovery. Against
  the expanded selection-visible catalog, unique chemotype recall is 82.0225%
  for aldehydes and 85.3659% for isocyanides; architecture recall is 100% and
  71.4286%, respectively. Against the original 12,276-product component set,
  unique chemotype recall is 91.9355% for aldehydes and 88.8889% for
  isocyanides.
- Generated frequencies remain asymmetric: aldehyde C=C and ester-like
  carbonyls are underrepresented relative to selection-visible occurrences,
  whereas aldehyde C#C is overrepresented. Therefore support is broad enough to
  proceed with route-grounding and stratified candidate evaluation, but the
  manuscript must not claim uniform or exhaustive continuum coverage.
- Freeze the audit contract at
  `configs/model/phase1_ugi_tail_chemotype_audit_v2.json` (SHA-256
  `de7bf966eb379d51009276c0a2099f74236f4f40cb36294da6bb3e0e65489881`)
  and result at `results/phase1/ugi_tail_chemotype_audit_v2.json` (SHA-256
  `840a9535ea6e16765d98b04bd63d6ca54afb919e61dc6e6a7ace623e2ddc523c`).

## 2026-08-02 - Intersect the 1.49-million-product Ugi expansion with route evidence

- Treat the actual 1,486,415 unique constitutional products from the frozen
  424-component exact forward enumeration as a bounded component-continuum
  census. This is not the complete atom-level model support and does not turn
  forward consistency into experimental synthesis evidence.
- Exactly 1,830 products (0.1231%) contain three components with strict complete
  routes. A further 45,034 products (3.0297%) contain only exact-complete or
  family-projected components, 305 contain an exact route with an open L3
  terminal and no projection, and 861 combine a family projection with that L3
  open state. The evidence-bounded exact/projected envelope including the L3
  open state therefore contains 48,030 products (3.2313%).
- Of the remaining products, 560,763 (37.7259%) contain missing route knowledge
  without a current outside-support branch, and 877,622 (59.0429%) contain at
  least one component outside the current route registry. Neither state is
  chemical impossibility, and family projection is not exact executed L2
  evidence. The census therefore demonstrates a nontrivial route-guidance
  landscape while explicitly refusing to estimate existential synthesizability
  or synthesis-success probability.
- Freeze the contract at
  `configs/route/phase1_ugi3_enumerated_routeability_census_v1.json` (SHA-256
  `6f91dfdeff1a03c727d6bde8a4d0e94f7fcb2a32b5b8d2d617f6b3420441252e`)
  and the result at
  `results/phase1/ugi3_enumerated_routeability_census_v1/result.json` (SHA-256
  `5ccebef57afa3f5bcc35558f353a1d849944d4632a15174ef322fa68b498a0df`).

## 2026-08-02 - Audit actual starting materials beneath the optimistic AGILE-template stress test

- Keep the 1,486,415-product template-saturation result as a structural
  counterfactual only. Materialize all 122 role-summed projected tail leaves and
  all 264 admitted heads, then match them by exact constitution to unexpired US
  item-level evidence. Keep current procurement, historical source use, exact
  L2 execution and prospective Ugi success as separate axes.
- Current evidence closes 25/69 aldehyde-program leaves, 10/53
  isocyanide-program leaves and 31/264 heads. Thus 35/122 tail leaves are
  currently evidenced and 87 remain procurement-unresolved; the latter are
  missing evidence, not proven unavailable.
- Historical exact source use covers 23 aldehyde leaves and seven isocyanide
  leaves. Across both roles, 23 leaves are current and historical, 12 are
  current-only, seven are historical-only and 80 are neither. Visual inspection
  of Miao 2019 Supplementary Note 1 confirms that ethyl isocyanoacetate was used
  in the reported syntheses, not the ethyl-glycinate leaf mechanically projected
  beneath it. Parent-component observation therefore cannot close an upstream
  leaf by association.
- Sixty-three of 107 aldehyde programs and 10/53 isocyanide programs have all
  projected leaves currently closed. Both tail programs are leaf-closed for
  165,600 products (11.1409%); adding a current head reduces this to 18,810
  products (1.2655%). These remain starting-material identity closures, not
  exact-substrate L2 routes. The independently frozen strict exact-route tier is
  only 1,830 products (0.1231%).
- This result preserves the purpose of synthesis guidance: generic AGILE
  templates can be attached everywhere, but real terminal evidence and exact
  route support are far narrower. Guidance must use closure, exact evidence,
  burden and uncertainty; the matched post-hoc arm must receive the identical
  evidence and compute budget.
- Freeze the audit contract at
  `configs/route/phase1_ugi3_precursor_leaf_closure_audit_v1.json` (SHA-256
  `5caa0650f5ac83da441cf4a831d6c5cf0d1c91874591a863a9b0d67125827c94`),
  the result at
  `results/phase1/ugi3_precursor_leaf_closure_audit_v1/result.json` (SHA-256
  `0c0dab513ac034ecc19ac6e97509db51f58b01bad356b6b81bae681daf7cfa64`),
  the 122-row leaf ledger at SHA-256
  `b45effe39d5e2ca8be8a55d76efd5f090992310eaaef2bd3ad5a093b4af23442`
  and the 264-row head ledger at SHA-256
  `b35bb2aa162e0c5f8ea01a7ae782f4541929cf957f7b62ee9736250f0bff73f0`.

## 2026-08-02 - Seal an independent route-registry saturation holdout

- Separate candidate-driven route development from the final coverage test.
  Permit the frozen fresh-v2 pool to direct evidence mining, but prohibit the
  sealed holdout from doing so.
- Draw and hash-seal 4,096 morphology programs with program seed `20260819`.
  Do not generate or inspect their molecular realizations until the expanded
  route registry, exact evidence hashes and scoring policy are frozen. Reserve
  flow seed `20260820` and terminal seed `20260821` for the one-time reveal.
- Admit a new reaction family only when the development data contain at least
  three unique affected components and ten one-gap product occurrences, and
  only after an atom-mapped transform, primary-source execution, forward
  verification, a substrate-scope boundary, a terminal-material policy and an
  explicit uncertainty/failure taxonomy are qualified. An exact locked-
  candidate exception requires separate authorization.
- After registry freeze, report the exact-L1 denominator, route-complete
  product fraction, number of complete component branches, missing-knowledge
  and outside-support counts by role, novelty strata and evidence family. Do
  not tune thresholds or the registry against the revealed molecules.
- Freeze the contract at
  `configs/route/phase1_ugi3_route_saturation_holdout_v1.json` (SHA-256
  `c76b43fb59bd4802f5bf8596c8b2095ff8abfd59448b26e75c1f41264b679a79`),
  the program draw at
  `results/phase1/ugi3_route_saturation_holdout_v1/programs.json` (SHA-256
  `20c29c692a3ce6a73584ae4182db93416550a531f36a9e21c1144d1b8efb6ad8`)
  and the unrevealed-molecule seal at
  `results/phase1/ugi3_route_saturation_holdout_v1/seal.json` (SHA-256
  `551843130c434916691a2e3855e833f572bad007dc3a6d3605566325b3de34a1`).

## 2026-08-02 - Triage fresh-pool route gaps before expanding reaction-family support

- Classify the unresolved exact component in each of the 802 fresh-v2
  one-gap products without promoting family projections to exact routes.
  Preserve procurement evidence, projected leaves, exact-substrate execution
  and declared route support as separate axes.
- The 349 unique gap components comprise 130 head procurement or exact-route
  gaps, 33 projected-family targets whose leaves are all current, 56
  projected-family targets with partial leaf closure, 106 projected-family
  targets with no current projected leaf, 21 outside-support review targets
  and only three components without a mechanical program family.
- Across product occurrences, the three existing projected programs already
  explain 506 gaps: fatty-acid/diol esterification followed by oxidation (202),
  primary-alcohol oxidation (201), and amine formylation followed by formamide
  dehydration (103). Exact-substrate evidence and terminal closure remain
  required; these counts do not validate the projected reactions.
- The only direct family-discovery candidates are three carbonate-linked
  aldehydes affecting four one-gap products. They fail the frozen minimum of
  ten one-gap product occurrences, so broad reaction-family mining is not
  indicated now. Prioritize exact-substrate evidence within the existing
  families, current procurement or exact upstream routes for heads, and
  separate declared-support review.
- Freeze the triage contract at
  `configs/route/phase1_ugi3_route_gap_triage_v1.json` (SHA-256
  `eac9572ca8cf2cd224576a17ecddcb5fc92a73c7e562c41b146e0d4ac3956fe5`),
  the result at
  `results/phase1/ugi3_route_gap_triage_v1/result.json` (SHA-256
  `908004b97a3a90e3caf5e0ea4404b5cc44c9306856639fc5b9400e1aa3541a37`)
  and the 349-row ledger at SHA-256
  `fdb06de006abf1eda6dc0c5fc4d5d58d4f8db2506bd60df809a008cfaff7f216`.

## 2026-08-02 - Qualify one exact three-step C18 aldehyde route without family promotion

- Admit an exact route to octadec-17-ynal only for the three reported
  substrate/product pairs: stearolic-acid reduction to octadec-9-yn-1-ol,
  exact-chain alkyne isomerization to octadec-17-yn-1-ol and oxidation to
  octadec-17-ynal. Every step uniquely reconstructs the declared product, and
  all 300 randomized-SMILES verification trials pass.
- Close the sole structural terminal leaf with the exact MedChemExpress
  HY-W341997 stearolic-acid item observed at 99.90% purity and explicitly in
  stock in 50-mg and 100-mg packages. Preserve the timestamped official-page
  observation separately from the official identity datasheet, expire current
  procurement on 2026-09-02 and exclude the cached Cloudflare block page from
  evidence.
- Freeze the procurement assessment at 2026-08-03T02:30:00Z, enforce strict
  UTC parsing and require `observed <= assessment <= expiry`. The current
  terminal status must be refreshed before candidate lock.
- Keep the qualification exact-pair only. Do not transfer the route to a
  homologue, admit a reaction-family template or interpret graph
  reconstruction as experimental synthesis success. The assessment is
  complete at L2/L3 but does not authorize synthesis-guided sampling, candidate
  selection or sealed-holdout reveal.
- Freeze the contract at
  `configs/route/phase1_ugi3_exact_c18_route_v1.json` (SHA-256
  `84c025d5c8df3bf00bbf7c2150e6039febc1f2c3936a251dcc22b25a9d2fda76`),
  the result at `results/phase1/ugi3_exact_c18_route_v1/result.json`
  (SHA-256
  `eabf54d2983b4dde25ad3a0a084e018098d7a227a5cedb0cfa82b43522bd7475`),
  the assessment at SHA-256
  `d791cb7e70e53588b6a2888404c7e7a100abc00ccd4e41e409f8f57ab1dc4b6b`
  and the exact-step ledger at SHA-256
  `459632d0bf8bb00265997c55c8899fb79c08cc2d1b1508d29eb743192b49143b`.

## 2026-08-02 - Refresh development-pool coverage with the exact C18 route

- Apply the qualified C18 synthesis value to exactly one role-qualified
  component identity in immutable fresh-pool coverage v3. The component occurs
  in 107 exact-L1 products: 55 move from zero to one complete component branch,
  39 move from one to two and 13 move from two to three.
- Fully route-complete products increase from 172 to 185 among 3,975 exact-L1
  products. All 13 newly complete products contain a component absent from the
  frozen component catalog. This is a deterministic development-pool coverage
  update, not a prospective hit rate or synthesis-success probability.
- Preserve scalar synthesis values and success probabilities as null, promote
  no reaction family and keep the independent molecular holdout sealed. Use the
  refreshed ledger to update route-mining priorities only after the remaining
  exact-route audits are frozen.
- Validate the complete fail-closed assessment policy rather than copying it
  opaquely, bind the result to the configuration hash and reject any policy
  mutation that authorizes guidance, family promotion or holdout reveal.
- Own the exact C18 route configuration as a v4 input and reproduce the full
  route audit before accepting its value. Reject contradictory extra scope,
  unowned result hashes and assessment times outside the procurement window;
  pinning only a result and assessment is insufficient evidence ownership.
- Freeze the refresh at
  `configs/route/phase1_ugi3_fresh_pool_route_coverage_v4.json` (SHA-256
  `393788b917523a74cd81c093026bb1bb8bb049d0f36185c6ad5f3c35fb01a69d`),
  the result at
  `results/phase1/ugi3_fresh_pool_route_coverage_v4/result.json` (SHA-256
  `ff6b8582efe8912b3fc458c009f8d4b43eee548b9250bd14e39d059b83358a69`),
  the component ledger at SHA-256
  `2343ecfb56f4e6e3e8423ada26088b9354e0a230be8297c165b35db1d0744550`
  and the product ledger at SHA-256
  `7a65b5f30d7746f1ca0b3ecd4e7cfd54c61d5cf65a81dcee39df729de95fa2c0`.

## 2026-08-02 - Reject the conflicted C16 source and qualify an independent exact route

- Reject WO 2024/073486 A2 Example 2 Step 1 as `source_conflict`: the written
  oct-7-yn-1-ol (C8) plus 1-bromoheptane (C7) reactants contain 15 carbons and
  cannot yield the reported C16 product. Do not repair the record by inferring
  a different alkyl bromide, and do not use the rejected source anywhere in the
  admitted route.
- Admit an independent exact four-step route to hexadec-15-ynal: exact C3+C13
  chain construction, THP deprotection, exact C16 alkyne isomerization and
  product-specific alcohol oxidation. All four transforms uniquely reconstruct
  their declared products and all 400 randomized-SMILES verification trials
  pass.
- Close the two graph-reactant leaves with exact, timestamped Sigma-Aldrich
  300810 and TCI America B0935 item observations. Their current status expires
  on 2026-09-02 and must be refreshed before candidate lock. Reagents and
  solvents recorded only as reaction conditions are not silently converted
  into graph-reactant branches.
- Freeze an explicit assessment time and require strict UTC parsing,
  top-level/item timestamp agreement and
  `observed <= assessment <= expiry`. Reject malformed, future or expired
  observations and any mutation of the 30-day refresh policy.
- Keep all four transformations exact-pair only. No family, analogue or
  homologue promotion follows from this route, and the exact qualification
  alone does not update coverage or authorize guidance or holdout reveal.
- Freeze the exact route at
  `configs/route/phase1_ugi3_exact_c16_route_v1.json` (SHA-256
  `72079699d839967de8e5290195e6d0cfe0b38cc03a0c527f7c71119fcb213527`),
  the result at `results/phase1/ugi3_exact_c16_route_v1/result.json`
  (SHA-256
  `99ab8ff669df0e4eb97564abf6ba9ec6b713cdb59c0265a966f101121b4031f4`),
  the assessment at SHA-256
  `97bdd52114c83388d4c9146e1e2569c6ab786903ca62a6004cc2d0017ee75a10`
  and the route ledger at SHA-256
  `81ce6a63ac5616ed69b5b4ac1feff55993fe59e0be5cc516f055d1458a403e93`.

## 2026-08-02 - Refresh development-pool coverage with the exact C16 route

- Apply the qualified C16 synthesis value to exactly one role-qualified
  component identity in immutable fresh-pool coverage v4. The component occurs
  in 99 exact-L1 products: 41 move from zero to one complete branch, 40 move
  from one to two and 18 move from two to three.
- Fully route-complete products increase from 185 to 203 among the same 3,975
  products. Exactly one component value changes; no component identity metadata
  or product metadata changes, and all expected transitions reproduce without
  discrepancy.
- Preserve null scalar values and null synthesis-success probabilities. Promote
  no homologue or family, keep candidate selection and guidance unauthorized,
  and leave the independent holdout sealed.
- Freeze the refresh at
  `configs/route/phase1_ugi3_fresh_pool_route_coverage_v5.json` (SHA-256
  `38fb52131bcb4c79f122a19a7997da3fe4c8d7e6f6fa229add4b3fbc5c9e35c3`),
  the result at
  `results/phase1/ugi3_fresh_pool_route_coverage_v5/result.json` (SHA-256
  `6f80724272a0e82a6bde6ce6581ce0383a016b452bc132a86a6289b51f34bc95`),
  the component ledger at SHA-256
  `570c3f43857e30d650e7ef6e21d25a5e0755766d41e6b0b91667b3fbe8a4cd63`
  and the product ledger at SHA-256
  `1afc543d50e7b5d8230ae54ec8edae33c73933a8b55434634c21f88b532972b5`.

## 2026-08-02 - Use bounded family evidence as graded guidance, not exact-route identity

- Freeze the final nonselecting priority refresh over v5 at
  `configs/route/phase1_ugi3_fresh_pool_route_priority_v4.json` (SHA-256
  `22e18b44d592b106546222e917f5ff4b80121716b3ab3cde4cfea1f78b91118c`),
  with result SHA-256
  `adae9a7b111e6a715e63b705f11b8e6270c747541f192ec799cc29d7977a0211`
  and 401-row priority-ledger SHA-256
  `55062a7ea94641f5fa5ab0a7735854772653d471615f7c252bc19ce4f5dd694e`.
- The resulting 897 one-gap products contain 377 exact missing-knowledge
  targets and 24 outside-support targets. Existing projected program families
  cover 209 exact components and 515 affected product occurrences; 177 of
  those occurrences already have every projected leaf currently evidenced but
  still lack qualified exact or bounded family scope.
- Freeze this family-aware classification at
  `configs/route/phase1_ugi3_route_gap_triage_v5.json` (SHA-256
  `39ca8b4c649ad7f0622e99a809eb78fe15545ea51112c70e22a95cace1387416`),
  with result SHA-256
  `01530b93b85d4ab889d16e47565b3de61d14ab92acb6f94ea7ddc0daebaae978`
  and ledger SHA-256
  `887d81b6737f5baa81d1566c499e6323931d65565b3da8ec6111439f04aa8ef7`.
- Continue to distinguish exact source-executed routes from family-projected
  programs. The latter already provide broad proposal support and should not be
  discarded merely because they are not exact-substrate observations.
- Calibrate family evidence within explicit applicability domains defined by
  reaction class, reactive handle, functional-group compatibility and bounded
  homologous range. Separate interpolation from extrapolation, preserve
  forward consistency and terminal-material closure for each projected target,
  and attach higher uncertainty than to exact evidence.
- Permit a qualified family projection to contribute a graded synthesis value
  during development and sampling. Do not relabel it as an exact observed route
  or a synthesis-success probability, and require an exact, candidate-specific
  dossier before prospective lock.
- Stop indefinite identity-by-identity mining once the calibrated family lane
  covers repeated homologous gaps and the remaining attractive failures are
  classified as exact-candidate confirmation, missing knowledge or evidenced
  incompatibility. This preserves the matched exact-evidence registry pair and
  sealed holdout while making the production planner chemically scalable.

## 2026-08-02 - Freeze a prereveal paired-registry protocol without revealing molecules

- Define R0/`without_c18` and R1/`with_c18` as immutable registries applied to
  the same independently sealed 4,096-program draw, flow seed and terminal
  seed. R0 need not equal historical coverage v3; all non-C18 evidence,
  including the qualified C16 record, must be canonically identical in both.
- Permit exactly one role-qualified mutation at octadec-17-ynal. Additions,
  deletions, regressions, similarity closure, homologue transfer, family
  promotion, replanning, resampling and post-reveal evidence are prohibited.
- Predeclare the exact-L1 denominator as products passing validity, component
  reconstruction and exact forward reconstruction. The paired estimand is the
  fraction of that fixed denominator moving from noncomplete under R0 to
  complete under R1, with a two-sided exact 95% interval. It is descriptive
  marginal registry coverage, not a synthesis-success probability or a claim
  of universal route saturation.
- Keep identity-bearing ledgers private and hash-pinned; expose only aggregate
  counts. Correctly fail closed while the final registry inputs remain unbound,
  and reject protocol freeze timestamps more than five minutes in the future.
  No holdout molecule has been generated or inspected.
- Freeze the protocol at
  `configs/route/phase1_ugi3_route_registry_pair_protocol_v1.json` (SHA-256
  `b2148b3d1f2f31fc2f486e37e3b6f23d12e2812b260f8fdcf44b0881445bcf1f`)
  and retain the inert builder configuration at
  `configs/route/phase1_ugi3_route_registry_pair_builder_v1.json` (SHA-256
  `cb5a2d6e238225fc23dbcb0279b777fc9d9e446c4380730f16459170952bf003`)
  until explicit final ledger and evidence hashes are supplied.

## 2026-08-02 - Freeze the final prereveal R0/without-C18 parent component ledger

- Construct the parent registry deterministically from the final v5 component
  ledger while restoring exactly one role-qualified octadec-17-ynal record
  from immutable pre-C18 v3. Preserve all 1,918 non-C18 v5 records, including
  the qualified four-step C16 route, without source-record mutation.
- Keep the C18 parent record noncomplete with zero route steps. Add or delete no
  component identity, admit no reaction-family or homologue template, and do
  not materialize the registry pair or access the molecular holdout.
- Freeze 1,919 records with 44 complete, 1,830 missing-knowledge and 45
  outside-support component values. The ledger is directly accepted by the
  downstream parent-record loader while the paired-registry builder remains
  inert and unbound.
- Freeze the parent-ledger config at
  `configs/route/phase1_ugi3_frozen_parent_component_ledger_v1.json` (SHA-256
  `fcd82034b58bfbaadcf346a5aa4a590c5617530a8754920ddd27810bc1cc0e11`),
  the audit result at
  `results/phase1/ugi3_frozen_parent_component_ledger_v1/result.json`
  (SHA-256
  `700bfc782d736e2738bf7948769fa7ccd2e985cb1803bb46595e88543aa323e4`)
  and the frozen component ledger at SHA-256
  `dfbeeb0e94377a1e4da2192c073f932123168514388a209c51a5427c98b21491`.

## 2026-08-02 - Harden and bind the immutable prereveal R0/R1 registry pair

- Before any molecular holdout generation, reject the first pair materialization
  as insufficiently lineage-owned: its manifests named the source component
  ledger, but the independent validator did not prove that both snapshots used
  the identical pinned, frozen ledger. Archive those outputs intact under
  `ugi3_route_registry_pair_v1_superseded_pre_lineage_hardening`; authorize no
  result claim from them.
- Require the exact C18 config, result and assessment to be independently
  hash-pinned. Reproduce the full C18 audit from the config-owned evidence and
  require byte-identical result and assessment artifacts before admitting the
  R1 value. This closes contradictory-scope and repinned-assessment bypasses.
- Require both manifests to own the same source-ledger path, SHA-256, schema and
  frozen status. Require its record count to match both snapshots and require
  the registry-diff source hash to equal that validated ledger hash. Freeze the
  diff schema in the protocol.
- Materialize 1,919-record R0 and R1 snapshots differing at exactly one
  role-qualified C18 record. All 1,918 non-C18 records, including the qualified
  C16 route, are byte-equivalent. No similarity, family or homologue promotion
  is admitted.
- Independently validate the final binding with one authorized delta and an
  absent, uninspected molecular holdout. The binding authorizes the one-time
  blinded runner only after that runner separately enforces no rendering,
  overwrite or retry.
- Freeze the corrected protocol at SHA-256
  `36666eefdc8b0347191165041ea84cc7d9ce977f71442e2e961b79cabb0cd80b`,
  the bound builder config at SHA-256
  `3fd955ed7cff7db682c3d97461e0a2d25c618cba6b151f0ae3399b7311962a9f`,
  and the final binding at SHA-256
  `2f9481ecc2b30335749f57dc250323f64b8b401ec135a4ac565e29aa3e3ebd09`.
  R0/R1 manifest SHA-256 values are respectively
  `bc25cef83f183fed8686fd718319bd53acefb8d9ff5b076c7fb546b1abad2a08`
  and
  `3fbe8a597b185a776167b42ac7080b11813626a5cd78fc198ccf0f503cbd1d89`;
  the exact-one-change diff SHA-256 is
  `09b433bd1647b65263891e735a241c8c6f07d8d106a9c1f086b6fb829daecfb9`.

## 2026-08-02 - Audit graded family evidence without scalarizing or running guidance

- Join the final v5 exact component values to the final v5 one-gap route
  triage without changing either source assessment. Across 1,919 components,
  retain 45 exact-current completions as the highest evidence class and retain
  209 mechanical family projections at lower confidence: 32 with all projected
  leaves current, 57 with partial current leaves and 120 with no current leaf.
- Preserve 1,620 remaining missing-knowledge records and 45 outside-support
  records as distinct typed states. Keep incompatibility, budget exhaustion,
  invalid input and execution error as separate zero-count schema states rather
  than merging them into an undifferentiated failure class.
- On the 401-component one-gap frontier, the three family-projection strata
  account for 177, 92 and 246 product occurrences, respectively; unprojected
  missing knowledge accounts for 287 and outside support for 95. These are
  development-pool evidence counts, not experimental hit rates.
- Do not qualify bounded family scope in this audit. Every family projection
  retains its exact component outcome as `missing_knowledge`, remains below
  exact-current evidence and requires exact scope and a candidate-specific
  dossier before prospective lock.
- Define no scalar synthesis value or synthesis-success probability, run no
  guidance, select no candidate, and access no holdout identity. The categorical
  ordering is an evidence hierarchy only.
- Freeze the audit at
  `configs/route/phase1_ugi3_graded_family_evidence_audit_v1.json` (SHA-256
  `7080422192e98cab422ac6ebf95420ae252138b61bb0a38c646f1af75b855de7`),
  the result at
  `results/phase1/ugi3_graded_family_evidence_audit_v1/result.json` (SHA-256
  `da179092251cda471feabe63a35bffa6d9f752f2b09c5d5d5b87456e55d88dc0`)
  and the 1,919-row evidence ledger at SHA-256
  `26d96faa5194429df85af8477c74395c2cf35db46d143b5d1fac5024d9d827ad`.

## 2026-08-03 - Census constitutional applicability axes without qualifying family scope

- Compute candidate-specific constitutional axes for all 209 family-projected
  components: carbon count, longest carbon-only segment, reactive-center graph
  reach, carbon-carbon unsaturation count and handle-relative position, carbon
  branch-point count and position, radius-two reactive-handle neighborhood,
  projected-leaf currency, role and program family. All 209 records have
  complete axis metadata; alkene stereochemistry is not inferred.
- Freeze a descriptive recurrence rule using four-carbon count bands and a
  minimum of three exact components per otherwise matched structural bucket.
  This yields 25 recurrent interpolation-bucket candidates containing 136
  components and 312 one-gap product occurrences, plus 60 sparse
  extrapolation-review buckets containing 73 components and 203 occurrences.
- The recurrent/sparse component split is 74/18 for fatty-acid/diol
  esterification followed by oxidation, 51/22 for direct primary-alcohol
  oxidation and 11/33 for amine formylation/formamide dehydration. This shows
  that most isocyanide projections remain structurally sparse under the frozen
  axes even though all retain a valid isocyanide handle.
- Treat recurrence only as a queue for bounded scope calibration. It does not
  establish interpolation, forward compatibility or exact reaction evidence;
  sparse buckets are not chemical incompatibilities. Projected-leaf currency
  remains a separate axis and every exact component outcome remains unchanged.
- Qualify no family scope, define no scalar or success probability, run no
  guidance, select no candidate and access no holdout or registry-pair input.
- Freeze the census at
  `configs/route/phase1_ugi3_family_applicability_census_v1.json` (SHA-256
  `f697e3d8db59cbd58904209062aedc3e02b939de0d6954bb21f90d2a182d6894`),
  the result at
  `results/phase1/ugi3_family_applicability_census_v1/result.json` (SHA-256
  `8412f7d9f97724f5f379ab8a9aa03326b324176e7e606e2fad5593cfbf5652d8`)
  and the 209-row census ledger at SHA-256
  `8942f25d012863ae19af24c42149686f54c7a799fc6db9106a6818fa3e1864b0`.

## 2026-08-03 - Qualify synthetic matched-budget orchestration only

- Add a diagnostic two-arm orchestration contract around the existing
  restartable-sampler, fixed-budget rollout and fake-value SMC plumbing. One
  frozen schedule supplies identical morphology-program bytes, integer
  checkpoints, generator/closure checkpoint hashes and productive-generation
  seeds to guided and post-hoc arms. Route-assessment seeds remain keyed and
  arm-specific.
- Reserve route compute before admitting each common schedule unit. Under the
  frozen synthetic budget, both arms admit the same two-unit prefix and censor
  the third unit; no dummy planner or verifier calls are added. Both arms record
  two productive-generation calls, two terminal completions and two final
  candidates.
- Record logical and physical planner/verifier calls independently. In the
  synthetic cache test, each arm makes two logical planner calls but only one
  physical planner call after an arm-internal cache hit. Both arms begin from
  the same frozen cache-snapshot hash but receive distinct deterministic clone
  IDs, so the later post-hoc arm cannot inherit the guided arm's warm cache.
- Seal all admitted post-hoc terminal and generation-trace bytes into one lock
  manifest before the first post-hoc route assessment. At zero guidance, require
  terminal bytes, trace bytes, validity and exact-L1 state to be bitwise equal
  across arms; divergence fails closed.
- This is a synthetic, nonselecting plumbing qualification. The checkpoint and
  cache hashes in this diagnostic identify fake fixtures, not production model
  or cache artifacts. Define no synthesis scalar or success probability, run no
  real route planner or synthesis guidance, use no biological guidance, select
  no candidate and access no sealed holdout.
- Freeze the contract at
  `src/forge/product/ugi_matched_budget_orchestration.py` (SHA-256
  `0643853cc15f92faf88c0c731db8e5691fbc899123386d2f4fd7e49806608cf5`),
  its config at
  `configs/model/phase1_ugi_matched_budget_orchestration_v1.json` (SHA-256
  `51c8804b034d648d7e64ca64066964e271db93dd9e20928e0d7aeb1309d6b5ad`)
  and the result at
  `results/phase1/ugi_matched_budget_orchestration_v1/result.json` (SHA-256
  `0c4b9f82df8fa68a4c81e3a228472cda3d73f2d1c04e5d69ea80a6d9a1ee2122`).

## 2026-08-03 - Execute and validate the one-shot blinded route-saturation holdout

- Execute the hash-bound one-shot runner once, after the durable claim, output
  and attempt paths are proven absent and every pinned dependency passes the
  complete prereveal validation. The claim is durable and the experiment is
  not retryable or resampleable.
- Generate the preregistered 4,096 sealed products without rendering or
  identity-bearing public output. Of these, 3,960 pass the frozen validity,
  component-reconstruction and exact-L1 eligibility contract.
- Apply the immutable paired registries to the identical eligible products.
  R0/without-C18 completes 201 products and R1/with-C18 completes 222. The
  single exact octadec-17-ynal route therefore causes 21 products to move from
  noncomplete to complete: 0.005303 of the fixed eligible denominator, with a
  two-sided exact 95% interval of 0.003286 to 0.008095.
- The target component occurs 139 times, but only 21 occurrences flip complete
  product status because every other required component branch must already be
  complete. This is direct evidence that exact upstream component knowledge
  changes product-level route closure and that L1 decomposition alone is not
  sufficient.
- Interpret the result only as a marginal registry-coverage estimand under the
  frozen generator, candidate distribution and evidence policy. It is not a
  synthesis-success probability, a universal route-saturation estimate, a
  guided-versus-post-hoc comparison or evidence that the remaining products
  are chemically impossible.
- Independently rerun the completed-output validator. It recomputes and verifies
  the sealed private ledgers while returning only the preregistered public
  aggregate; status is `completed_blinded_holdout_valid`.
- Freeze the execution config at SHA-256
  `4ef005f8f33508fa2e336fa379ffdee7e747ec422f6925bb05da2c2c58063f58`,
  the public aggregate at SHA-256
  `929b2fe33cbe34f98121dedcf28c6122f364ff21397e94956a93b689d0e284fe`,
  the execution manifest at SHA-256
  `627e7e8072c1595866c479d724a92d3a9d17d103ead603318a68720790f80738`
  and the final execution seal at SHA-256
  `919f91e4fe7402713c98a6dede7f272d7030d9153c564e59865e07b3ad4be9d7`.

## 2026-08-03 - Qualify immutable planner-cache snapshots and isolated overlays

- Preserve the hash-frozen `FilePlannerCache` implementation byte-for-byte at
  SHA-256
  `72a07031fc7cd623446c99625333d848b79059ada995f4b5c3b7bdfd53b523db`
  and add snapshot and overlay behavior in a separate additive module.
- Bind each immutable base snapshot to one cache context and L3 evidence window
  with a root-independent manifest containing sorted cache keys, canonical
  relative paths, entry-byte hashes and assessment hashes. Reject symlinks,
  unexpected paths, noncanonical bytes, checksum mismatches, key/assessment
  conflicts and context mismatches.
- Require an explicit assessment time satisfying
  `l3_accessed_at_utc <= assessment_at_utc < l3_expires_at_utc`; reject evidence
  before access and fail closed at or after expiry without consulting the wall
  clock.
- Give guided and post-hoc arms distinct writable overlays over the same
  read-only base. Read overlay first and base second, write only to the arm-local
  overlay, require byte-identical initial arm states and reject overlapping cache
  roots or conflicting assessments.
- Record before/after manifests for the base and both overlays, require the base
  manifest to remain unchanged and seal the deterministic audit with distinct
  clone IDs. This qualifies cache ownership and isolation only: define no
  synthesis scalar or success probability, run no guidance, use no biological
  guidance, select no candidate, modify no runner and access no holdout.
- Freeze the additive implementation at
  `src/forge/route/planner_cache_snapshot.py` (SHA-256
  `0f43b4da2b065d97fe5a32cb2d7a76cf7da7be2053a49053a58c52cc6b1feffb`),
  its focused tests at `tests/test_route_planner_cache_snapshot.py` (SHA-256
  `3db141421782f0900399f93449c5a54d3f779b30f4fedddb75e0678c5ca5f1d3`)
  and the provenance result at
  `results/phase1/planner_cache_snapshot_overlay_v1/result.json` (SHA-256
  `1aa4bb9fdbc8960f2590d95602f2b85c57b528d7059af8ea1864396c18b54243`).

## 2026-08-03 - Qualify lazy matched-arm planner-cache binding

- Add a typed, root-independent preflight seal for the two matched planner-cache
  overlays. Before any generation, require their manifests to be byte-identical
  and empty and bind the seal to the immutable base snapshot, route-planner
  context, route seed, planner budgets and explicit L3 assessment time.
- Keep the hash-pinned matched-budget runner unchanged. Bind the guided overlay
  lazily from the runner-supplied clone ID at its first assessment callback;
  before binding the post-hoc overlay, revalidate that its root is untouched and
  bind its distinct runner-supplied clone ID. Do not duplicate the runner's
  private clone-ID or schedule-hash formulas.
- Reject cache-root retargeting, base mutation, overlay mutation between
  preflight and bind, post-hoc-before-guided ordering, cross-arm clone-ID reuse,
  same-arm clone-ID drift and guided writes after the post-hoc phase begins.
  Finalize through `MatchedFilePlannerCacheOverlays` and require the immutable
  base and frozen guided overlay to remain unchanged.
- Qualify this as diagnostic cache ownership and binding only. Execute no actual
  generator or route source, define no synthesis scalar, run no synthesis or
  biological guidance, select no candidate and access no holdout.
- Freeze the additive binding implementation at
  `src/forge/product/ugi_matched_planner_cache_binding.py` (SHA-256
  `c81f80bee6131e394c7e9df223a1303792c92a94bfb5084416cbdda2402af624`),
  its adversarial tests at
  `tests/test_ugi_matched_planner_cache_binding.py` (SHA-256
  `1e5a87fd6606a17006fa6c5cce4684787b5b3739308560abcdc2e40d018faafb`)
  and the provenance result at
  `results/phase1/ugi_matched_planner_cache_binding_v1/result.json` (SHA-256
  `7995d8f833a3cc07b59e630e288bdd00668bd001962e5392b1eaba1cf46bdafc`).

## 2026-08-03 - Qualify the additive zero-guidance rehearsal composer

- Add one programmatic rehearsal entrypoint that composes the hash-pinned
  matched-budget runner, a typed restartable generator/closure adapter, the
  lazy matched-cache binding and the complete terminal-to-route assessor. Keep
  the runner and selected checkpoint artifacts unchanged.
- Permit only `guidance_strength = 0`. Before productive generation, authenticate
  the matched runner and terminal-route source, bind the selected generator and
  closure checkpoint hashes, planner context, immutable cache snapshot and
  route/source qualification, and require an exact uncensored budget for the
  frozen schedule.
- Prove byte identity of productive terminals and generation traces and record a
  separate constitutional chemistry-projection identity receipt for every
  exact-L1 terminal. Retain every typed per-role and product route receipt; do
  not reduce the audit to a scalar score.
- Require guided and post-hoc ledgers, structured route values and final overlay
  manifests to match while preserving distinct runner-owned cache clone IDs.
  Reject any checkpoint, context, cache, qualification, source, budget,
  terminal, trace or structured-route drift.
- Qualify the composer with synthetic dependency-injected generator and route
  sources only. Do not load the production generator, execute the production
  route source, define a synthesis scalar, run nonzero or biological guidance,
  select candidates or access a private holdout. Real production wiring remains
  a separate pending step.
- Freeze the additive implementation at
  `src/forge/product/ugi_zero_guidance_rehearsal.py` (SHA-256
  `dc24bd0deebf53c3303c6880229becf2bbfe08c3e0c30fcb7ca8c21ec270b884`),
  its focused tests at `tests/test_ugi_zero_guidance_rehearsal.py` (SHA-256
  `9a83b462ada973f472b415457ff6bf8b7787930c8879d2209eec310e3eb8d9e6`)
  and the qualification result at
  `results/phase1/ugi_zero_guidance_rehearsal_composer_v1/result.json` (SHA-256
  `682a5b5c770b3ddd1bce6789bcd9f8d1422dea4ca803607966e9e99b8714b4a0`).

## 2026-08-03 - Freeze production zero-guidance inputs and complete-result provenance

- Freeze the production rehearsal at 12 hash-pinned morphology programs, eight
  restartable steps, seed `20260802`, zero guidance and terminal decoder
  `ugi_joint_terminal_completion:argmax:v1`. Authenticate the selected generator
  checkpoint, closure checkpoint, production manifest, program draw and
  restartable-equivalence config and result directly from disk before returning
  any schedule.
- Derive each three-role route reservation from the typed runtime
  `PlannerCacheContext`; do not copy planner or verifier ceilings into the
  schedule. Require the explicit assessment time to lie inside the exact frozen
  L3 accessed-inclusive, expires-exclusive evidence window and reject any
  expansive biology, selection, holdout or nonzero-guidance scope.
- Add an execution-result wrapper that retains every canonical composer-result
  byte plus the complete `ProductionTerminalSupportAudit` for every assessed
  arm/unit. Require exact audit coverage and order, bind each audit to the locked
  terminal and arm-local cache clone, and require guided and post-hoc support
  audits to be identical in every field except the deliberately distinct cache
  clone identity.
- Reject missing, extra or duplicate audits; support, source, graph, role
  registry, planner-context or assessment-time drift; nonisolated cache clones;
  noncanonical or duplicate-key JSON; checksum tampering; and any expansion of
  zero-guidance scope. The additive collector delegates planner construction and
  only retains immutable audits; it does not route or scalarize independently.
- This decision qualifies deterministic production inputs and the final
  result/provenance envelope only. It does not execute the selected generator or
  production route source, does not write a completed composer result, defines
  no synthesis scalar or probability, runs no nonzero or biological guidance,
  selects no candidate and accesses no holdout.
- Freeze the rehearsal config at
  `configs/model/phase1_ugi_production_zero_guidance_rehearsal_v1.json`
  (SHA-256
  `cd52b7e5f896ce391f8fd9a0192b5c2b02bec4b84813fc71c7aafa29c071d554`),
  the config loader at
  `src/forge/product/ugi_production_zero_guidance_config.py` (SHA-256
  `f457f79e790ac58a5f922b70ead5e929275c54637a91037016e4d4fffebacffe`),
  the result wrapper at
  `src/forge/product/ugi_production_zero_guidance_result.py` (SHA-256
  `8383672d4a9b178756853bc5dbe1c9935de6167d894ed081a1446f60de11e108`),
  its focused tests at
  `tests/test_ugi_production_zero_guidance_result.py` (SHA-256
  `2bb01fc0644176c529b61324d9465b4cb7178074814779b1a143a1fbca50d861`)
  and the nonexecution qualification receipt at
  `results/phase1/ugi_production_zero_guidance_execution_contract_v1/result.json`
  (SHA-256
  `8f9e8c6cf374bb0c88944ea589fd8e9e4bf4409a1905dc4b32a8b8ac8f72f2c9`).

## 2026-08-03 - Execute the authenticated production zero-guidance rehearsal

- Execute the frozen 12-program, eight-step selected-generator schedule through
  both matched arms at exactly zero guidance. The run produced 12 terminal
  completions and 12 final candidates per arm, with no censored units and
  bitwise-identical productive terminals and chemistry projections between the
  guided and post-hoc arms.
- Reverify every retained terminal against the exact atom-mapped Ugi transform.
  All 12 distinct terminals reconstructed exactly at L1 and all three component
  handles passed the qualified registry policy. Retain 12 route receipts per arm
  and all 24 complete terminal-support audits.
- Preserve strict compute identity: each arm used 12 productive generation
  calls, 36 logical planner calls, 29 physical planner calls, 17 logical
  verifier calls and 17 physical verifier calls. The immutable base cache
  remained empty and unchanged; guided and post-hoc overlays remained isolated
  under distinct clone identities.
- Observe one route-complete candidate and 11 route-incomplete candidates among
  the 12 unguided terminals. Treat this only as a zero-guidance support result,
  not a calibrated probability or synthesis scalar. Every retained audit keeps
  `scalar_value = null` and `success_probability = null`.
- Seal productive generation with implementation SHA-256
  `8a8dfa39af82acdba2e9e25b31e3f79d218de36f3e5c827b6675aec4847b8739`
  across 34 source files and the exact Python, NumPy, RDKit and Torch versions.
  Seal the separate matched-arm orchestration implementation at SHA-256
  `a7793eb67d1984d5a3ba7a2a626a4a975ce4aee1679caf181ffcb173f4e67e27`.
- Freeze the completed result at
  `results/phase1/ugi_production_zero_guidance_rehearsal_v1/result.json`
  (file SHA-256
  `7bc95bbf57e22d6034ee01bc0b2a2d1568a88cc1295dfd1ac281345e30b1af54`;
  canonical execution-result SHA-256
  `221e77856f2ec8265eb7b817ca09ccfe959cfc993245b7f2219467de6f4bdeea`)
  and its execution manifest at
  `results/phase1/ugi_production_zero_guidance_rehearsal_v1/execution_manifest.json`
  (file SHA-256
  `ab220e641d472187000639d40804bebdbc6ab08b8702b60ce030fde96d0bed79`).
- This execution clears the zero-guidance production gate. It does not itself
  define the synthesis scalar, authorize biological guidance, select a
  candidate, access a private holdout or establish a synthesis-success
  probability. The next decision is the preregistered synthesis-value and
  Pareto/SMC guidance policy.

## 2026-08-03 - Separate FORGE from the broad lipid-generator program

- Treat the broad reaction-program-agnostic lipid generator and FORGE as two
  distinct scientific programs rather than two presentations of the same
  result. The broad program asks whether a lipid-native model can generate
  diverse, realistic ionizable lipids across chemistry platforms. FORGE asks
  whether complete, evidence-qualified synthesis programs can change which
  molecules are generated and make the resulting designs experimentally
  executable.
- Preserve the shared generator as an explicitly disclosed methodological
  dependency where appropriate, but do not reuse the same principal candidate
  cohort, prospective result, headline figure or biological conclusion across
  the two manuscripts. Disclose the related manuscript at submission and cite
  it when public.
- Retain the matched in-trajectory synthesis-guidance versus post-hoc experiment
  as FORGE's intended causal differentiator. The primary computational endpoint
  is the number of distinct, beyond-catalog candidates with exact L1 assembly
  and complete evidence-qualified L2/L3 closure under matched productive
  generation, planner, verifier and final-candidate budgets. Prespecified
  diversity, component-novelty, biological-applicability and distributional
  floors remain constraints, not optional secondary metrics.
- Keep this comparison bounded and off the prospective critical path. Continue
  constructing complete route dossiers and preparing the prospective panel
  while guidance is evaluated. A positive comparison promotes synthesis
  guidance to the main FORGE method. A null or adverse comparison is reported
  honestly and does not delay prospective execution; it removes the causal
  guidance claim and triggers a portfolio decision to integrate the route work
  with the broad-generator paper or reposition it around a genuinely distinct
  route resource/method, rather than forcing an overlapping generator paper.
- A learned retrosynthesis backend remains proposal-only. It may expand L2/L3
  hypotheses during terminal rollouts, but cannot create evidence, declare a
  route complete, or directly assign synthesis value. Independent lineage,
  forward verification, substrate-scope evidence, current terminal-material
  status and route-policy checks remain authoritative.

## 2026-08-03 - Make the L2 proposal lane lipid-aware without a lipid-only whitelist

- Select the official reaction-class-unknown Graph2Edits USPTO-50K release as
  the first learned proposal candidate because its source, checkpoint and exact
  corpus bytes can be pinned. This is a candidate pending runtime, license and
  frozen FORGE benchmark gates; it is not yet an activated production source.
- Keep the learned proposal search chemically broad. Do not restrict its
  support to transformations previously published in LNP papers, because that
  would turn documentation density into a false synthesis boundary.
- Make the hybrid system lipid-aware by searching exact lipid-component
  evidence and admitted programs first, then allowing bounded general
  proposals, and independently grading every hypothesis for Ugi component
  role, handle correctness, forward reconstruction, long-chain and unsaturated
  substrate scope, chemoselectivity, operational compatibility, purification
  burden and current terminal-material closure.
- Keep exact lipid precedent, non-lipid precedent and model-only proposal
  provenance separate. None of these labels alone creates evidence-qualified
  closure or contributes directly to synthesis value.
- Compare strict lipid-precedented, unrestricted learned and hybrid lanes under
  matched budgets on a frozen lipid-component benchmark. Do not fine-tune on
  the current 45-route local inventory: it is positive-only and role-imbalanced.
  Reconsider fine-tuning after a larger component-family-disjoint route corpus
  and experimental negative set exist.
- Freeze the six raw/canonicalized Graph2Edits splits in
  `configs/route/graph2edits_usp50k_training_corpus_manifest_v1.json`
  (SHA-256
  `f3c109121757e9ff2b7f63a29fa88f030b9753d3e786e6f21e6c6e55db8d2720`)
  and the updated inactive acquisition candidate at
  `configs/route/graph2edits_proposal_backend_candidate_v1.json` (SHA-256
  `834d3714c7e7c2ee9b98e09ef07466b76f29038988c655a13e69a93eed4539ff`).
- Correct the L2 verifier boundary before target materialization. The vendored
  Ugi registry is an L1 assembly verifier, not an L2 proposal verifier. L2
  screening now independently tries every in-scope assignment from the
  hash-pinned exact-source, exact-C16 and exact-C18 upstream registries. A
  Graph2Edits reaction-class label or score cannot select, define or validate
  the transform. No admitted verifier, multiple forward products or multiple
  reconstructing assignments are explicit fail-closed censoring states; exact
  source and exact-pair transforms may not be projected beyond their admitted
  scope.
- Freeze, without executing, a 120-target development-only comparison of the
  strict lipid-precedented lane, unrestricted Graph2Edits proposal lane and
  strict-first hybrid under matched proposal, verification, screening, time and
  memory budgets. Only the hybrid may be promoted, and only if every provenance,
  validity, adversarial-control and efficiency gate passes. Materialize all 120
  constitutionally unique, valid connected targets from development-only
  sources with one primary stratum each, isolate 36 exact scoring truths from
  lane inputs, freeze 36 local-source visibility masks and retain 12 valid
  wrong-role chemistry controls. The final inactive binding is
  `configs/route/single_step_proposal_lane_qualification_benchmark_v2.json`
  (SHA-256
  `4bea2147786056bd73d45a40fc2154730d76d0dbbe9ae9ec9206cd5ca1951e7b`).

## 2026-08-03 - Qualify synthesis-value source behavior without rewriting history

- Preserve every historical synthesis-value, exact-route and fresh-pool
  configuration and ledger unchanged. Their fail-closed source pins remain
  valid provenance boundaries rather than defects to bypass.
- Replay synthesis-value v1-v3 and current fresh-pool route coverage v5 through
  temporary configuration copies that change only the declared
  `value_source` SHA-256 from
  `dba5b11067513e4f431f78f3b9a316dfe0d58c5f046210c170c55b9a0e241e3c`
  to
  `6e77063f45012144912883f16a12698eda809446635c94d59d1ab6211993f189`.
- All four component ledgers and all four product ledgers reproduce
  byte-for-byte. All four summaries and provenance-normalized results reproduce
  semantically. Raw results differ only because they truthfully record the
  temporary qualified-config hash and candidate source hash.
- Treat fresh-pool v4 as a separate nested qualification boundary: its builder
  independently re-executes the immutable exact-C18 audit, whose own config
  pins the historical value source. Do not bypass that ownership check. A
  source-qualified exact-C18 vNext is required, together with an exact-C16
  vNext, before a new production lineage is frozen.
- This audit establishes behavior preservation on its admitted replay scope;
  it does not authorize historical repinning, synthesis guidance, candidate
  selection, a synthesis-success probability or sealed-holdout access.
- Freeze the audit config at
  `configs/route/phase1_synthesis_source_supersession_audit_v1.json` (SHA-256
  `d8c5f90b58f7fe61ead7a991e6225e4c8506fbb3249ba7f827f3a2edfa5b33f4`)
  and the result at
  `results/phase1/synthesis_source_supersession_audit_v1/result.json` (SHA-256
  `a07c01b247f1c47e8596112bec99a8c6035cb03eae9e39072b58d443a7cde7bd`).

## 2026-08-03 - Complete the source-qualified v6 prerequisite chain and stop before nonzero guidance

- Replay exact-C18 and exact-C16 under the current synthesis-value source
  without editing their immutable historical configs. Both step ledgers and
  both assessments reproduce byte-for-byte; summaries and normalized results
  reproduce semantically. Freeze the replay result at SHA-256
  `855436b3eab7388a5f5b02578ce5b27105754cbdcb4f48b504d27c4ee101848d`.
- Freeze one source owner over the current source, the four-target compatibility
  audit and both exact-route replays. The qualification result is SHA-256
  `7b72a64e2bc53fbe55aed6751610e9426c735a41553dc95556da123269fd2ad4`.
- Create fresh-pool route-coverage v6 strictly from immutable v5 bytes. No
  component or product value changes; both compressed ledgers are byte-identical
  to v5. Freeze the v6 result at SHA-256
  `339d11287ad1489336c1e949dab001aad05a169fa97dbe48c15c2b9857fec722`.
- Requalify the historical 12-pair beta-zero identity receipt and the synthetic
  matched-budget scheduling contract as prerequisites against v6. This is a
  lineage/contract receipt, not a rerun of production generation or the route
  planner. Freeze it at SHA-256
  `ec9d139db61d1809e09adc0bd2d1027b7c40d302849db4418068b87e410eec64`.
- Stop before nonzero guidance. No production synthesis scalar is frozen, the
  current vNext cumulative-source runtime is not qualified, and L3 must be
  refreshed before a future production execution. Do not select candidates or
  access the sealed holdout while these gates remain open.

## 2026-08-03 - Freeze route-completion utility, current L3 lineage and grouped SMC schedules

- Use a conservative binary route-completion utility, never a synthesis-success
  probability. Exact L1 plus current exact closure of all three component roles
  receives one; substantively assessed nonclosure receives zero; censored,
  nonexact or errored assessment receives no numeric utility. Its controller
  bridge applies zero incremental log-weight to censoring and cannot upgrade
  evidence. A preregistered 5% censor ceiling fails closed.
- Rebuild only the four cumulative inputs whose immutable configs pinned the
  superseded synthesis-value source. Preserve all scientific summaries and
  ledger bytes. The qualified cumulative source identity is
  `0fca91fae36da762eda53695405ec74cf5f9eacd19aee6ca3e4495a6a4d427c6`.
- Freeze the current decision horizon inside all 11 authenticated L3 windows
  and reproduce all 12 locked zero-guidance product values exactly under the
  original per-terminal support-boundary semantics. Do not describe this as a
  new generator execution.
- Replace the ambiguous 64-particle statement with an additive static schedule
  contract: 16 exact morphology programs times four stochastic particles per
  program, for every calibration and evaluation seed. Program bytes and indices
  are frozen; the 128 programs are disjoint across seeds. Resampling groups by
  `(seed, program_index)` and may never cross morphology programs.
- Do not authorize nonzero guidance yet. A real selected-model grouped
  lambda-zero run must still prove bitwise identity through closures, exact L1
  and all three synthesis checkpoints. Static schedule qualification does not
  close that execution gate. Candidate selection, biology and the sealed
  holdout remain out of scope.

## 2026-08-03 - Add an executable, source-neutral proposal-lane v3 contract

- Preserve the frozen v1 proposal benchmark, v2 target/hidden-truth binding and
  current exact-pair L2 resolver byte-for-byte. Add a v3 development contract
  that derives execution readiness from their authenticated receipts, the
  isolated runtime lock and the already-passed two-repeat normalized semantic
  smoke. Remove manual license, ten-repeat device, aggregate `all_gates_passed`
  and separate execution-authorization booleans from scientific readiness.
  Institutional license compliance remains an external owner responsibility.
- Add a source-neutral discovery layer downstream of the unchanged exact
  resolver. It reports exact known route, known-family forward-consistent
  projection, retained new-family hypothesis, ambiguity or rejection. Family
  projections reuse registry-owned transforms and establish graph consistency
  only; every state retains null evidence and success probability, forbids
  route closure and cannot enter synthesis value.
- Treat local novelty as missing independent scope/evidence, never as chemical
  incompatibility by itself. Once any proposal source passes the identical
  independent forward, scope, evidence, operational and L3 contracts, preserve
  the same adjudicated route state and value irrespective of whether the
  hypothesis originated from Graph2Edits, templates or retrieval.
- Complete the additive v3 scorer with primary-stratum metrics, paired
  per-target differences, strict/learned/overlap provenance, invalid and
  duplicate fractions, budget states, latency and memory. Separate scientific
  hard checks from operational preference checks. The adjudicator may recommend
  a later versioned review but always leaves production activation false.
- Freeze the unexecuted v3 contract at
  `configs/route/single_step_proposal_lane_qualification_benchmark_v3.json`
  (SHA-256
  `9b55e4fe1fb3c800ba146a62ff537b49080a132b75abecd50b307548072f2829`).
  Do not execute the 120-target benchmark, modify the production planner,
  select candidates or access a sealed holdout in this implementation step.

## 2026-08-03 - Qualify the real selected-model grouped lambda-zero identity

- Preserve restartable-equivalence v1 as historical provenance and use the
  versioned v2 receipt for the current selected implementation. V2 binds the
  qualified Python/NumPy/RDKit/PyTorch runtime and exactly reproduces the old
  batch-size 1, 4 and 12 terminal hashes.
- Execute one write-once, nonselecting 16-program by 4-particle calibration
  assignment through checkpoints 2, 4 and 6 with identity ancestry only. Pair
  all checkpoint completions, all 64 productive finals and all 64 direct
  selected-callback references bitwise.
- All identity checks pass. Final native validity and exact-L1 are 61/64; the
  three invalid particles remain frozen and were not repaired, retried or
  replaced. This is an execution-identity gate, not a perfect-validity claim.
- Freeze the config at
  `configs/model/phase1_ugi_grouped_zero_guidance_identity_v1.json` (SHA-256
  `328b78ccf82c88699b0d2f883489ffbd59a4794a013bb04eea687d9d89a28ff3`)
  and the result at
  `results/phase1/ugi_grouped_zero_guidance_identity_v1/result.json` (file
  SHA-256
  `9bf9818cbfccf60cf492fb54bdb932eff431df117e9020d68c01548b4b85a07f`;
  logical SHA-256
  `dbdc45b1b98dcfae9f9e8f18daabe284db4a3ce71a02976b1f071526493f86c4`).
- Continue to block nonzero synthesis guidance. A fresh, independently reviewed
  execution receipt must bind the current L3 horizon, cumulative-source
  runtime, route evaluator/cache isolation, selected adapter, grouped schedule
  and this lambda-zero result before explicitly authorizing a bounded nonzero
  run. Biology, candidate selection and sealed-holdout access remain false.

## 2026-08-03 - Separate exact component novelty from oracle applicability

- Preserve exact role-specific component identity as provenance and define
  chemical applicability independently across complete-product, amine,
  aldehyde and isocyanide views. Exact-new does not imply extrapolative, and
  interpolative does not itself authorize guidance.
- Retain v1 and v2 as historical diagnostics. Use v3 as the current audit:
  count Morgan plus robust descriptors, with exact component identities and
  products sharing the held component excluded from calibration references.
  Targets, predictions and errors do not set the applicability thresholds.
- In the 3,975-product fresh pool, classify 165 as interpolative, 215 as
  boundary and 3,595 as extrapolative. Of 3,963 products containing an
  exact-new component, 153 are interpolative. The fixed oracle separates these
  bins strongly on outer-test records (interpolative R2 0.463 and rho 0.653;
  extrapolative R2 -0.726 and rho 0.016).
- Add a nonauthorizing, fold-resolved conditional conformal diagnostic using
  the already-frozen fit predictions. Held-head and
  held-aldehyde--isocyanide-pair schemes pass the numerical gates; held
  aldehydes fail coverage stability, held isocyanides lack three eligible
  folds, and two pair schemes lack enough interpolative calibration records.
- Keep all potency guidance, raw-mean ranking and candidate selection off. The
  conditional audit followed inspection of v3 and cannot retroactively create
  a pristine authorization test. A later bounded pilot requires a separate
  versioned review and explicit role-domain restrictions.

## 2026-08-03 - Authorize one bounded HeLa potency-guidance diagnostic

- At the user's explicit direction, prioritize one HeLa mTP potency-guidance
  diagnostic before the nonzero synthesis-guidance experiment. This is an
  incremental scope change, not authorization for unrestricted biological
  optimization or prospective candidate selection.
- Keep the selected three-seed Ugi component-role-aware D-MPNN and its frozen
  HeLa endpoint unchanged. RAW 264.7 labels, synthesis values, the route planner
  and proposal engines must make zero calls in this diagnostic.
- Permit nonzero biological weight only for valid exact-L1 products for which
  the complete product and all three component views are interpolative under
  the frozen version-3 applicability policy. Exact measured combinations remain
  neutral. Restrict exact-new guidance to the two role-shift schemes supported
  by the conditional diagnostic: amine only, or aldehyde plus isocyanide with a
  known amine. Every other role pattern, boundary or extrapolative structure
  abstains with zero incremental potential.
- Require a newly hash-pinned conservative utility, matched in-trajectory and
  post-hoc arms, exact lambda-zero identity, deterministic ancestry replay and
  complete invalid/abstention accounting before execution. Do not use generated
  candidates or outer-test outcomes to select the utility transform, guidance
  strength or seed.
- Treat a technically correct run with no eligible nonuniform controller group
  as `signal_limited_by_applicability`, not as a negative biological result.
  The diagnostic can support predicted HeLa-potency enrichment only; it is not
  evidence of in-vivo efficacy and cannot lock a prospective panel.

## 2026-08-03 - Keep all generated roles primary and census applicability before guidance

- Retain the primary Ugi path as fully generated amine, aldehyde and
  isocyanide graphs. A known-head clamp is diagnostic-ablation material only;
  it is not the production model and cannot define the main biological result.
- Reuse the frozen unguided pool rather than launching a new or selectively
  retried rollout. All 4,096 scheduled attempts are retained: 3,975 are valid
  exact-L1 terminals, 42 fail molecule sanitization and 79 fail terminal
  support. No oracle, synthesis, proposal or nonzero-guidance call is made.
- Apply the already-frozen version-3 applicability thresholds independently to
  the complete product, amine, aldehyde and isocyanide views. Preserve the
  existing conditional role support: amine only, or aldehyde plus isocyanide.
  Exact measured combinations remain neutral and every unsupported role pattern
  remains abstained.
- Record the all-three-new baseline explicitly: 2,134 terminals contain exact-
  new amine, aldehyde and isocyanide graphs; 7 are interpolative in all four
  views, 64 are boundary and 2,063 are extrapolative. All 2,134 remain
  abstained. This is a nonselecting support census, not permission to guide the
  all-three-new region.
- Across the entire pool, 47 terminals meet a currently supported exact-new
  role pattern and are interpolative in all four views, but they receive no
  potency score here. Twelve exact measured combinations are neutral and 3,916
  terminals abstain. The result supports designing a role-aware trust-region
  controller; it does not authorize its nonzero execution.
- Freeze the policy at
  `configs/bio/phase1_ugi_generated_role_applicability_census_policy_v1.json`
  (SHA-256
  `d6b02e8b8500d5954f717b7e7743674ac1444f4e1613089ba3eaad4df4693fa5`),
  the readiness config at
  `configs/model/phase1_ugi_generated_role_applicability_census_readiness_v1.json`
  (SHA-256
  `07259f182ef622ba5ba795b64c66326c44c026d50d4cad8a5d15236307dd6f84`)
  and the write-once result at
  `results/phase1/ugi_generated_role_applicability_census_v1/result.json`
  (file SHA-256
  `af84d3016cae0fe3296904e31dde8535abb265efa85cb2fddb24ece3bc3a8548`;
  logical SHA-256
  `777d429cec7c1fe9d4dd32bb493ea971c9cf7c9d9a43fee37c033e22a361efde`).

## 2026-08-03 - Freeze a measured-only envelope for bounded biology mode

- Keep open generation unchanged and fully generate the amine, aldehyde and
  isocyanide graphs. Implement the new structural envelope as a separate
  bounded-mode controller; outside that mode its multiplier is exactly one and
  its incremental potential is exactly zero. The fixed-head path remains a
  diagnostic ablation only.
- Derive the envelope before applying it to generated structures, using unique
  measured AGILE graphs separately for the product, amine, aldehyde and
  isocyanide views. Do not read HeLa or RAW labels, pKa, LNP size or formulation
  measurements. Use eleven transparent integer graph features: heavy atoms,
  carbon, nitrogen, oxygen, other heteroatoms, carbon-subgraph diameter, carbon
  branch atoms, adjacent branch edges, carbon-carbon unsaturation, rings and
  combined ester-or-ether count.
- Freeze inclusive bounds as outward-rounded 1.5-IQR Tukey fences using linear
  type-7 quartiles on unique measured graphs. The reference contains 1,100
  products, 20 amines, 11 aldehydes and 5 isocyanides; 640 of 1,100 measured
  rows lie inside all four view envelopes. This reference coverage is reported,
  not silently repaired by looking at generated candidates.
- Apply the frozen envelope once to the existing 3,975 valid exact-L1 terminal
  pool. 477 lie inside all four structural envelopes and 3,498 lie outside.
  Cross-tab the result with the unchanged version-3 applicability and role
  policy: 40 records form a nontrivial currently supported, envelope-contained
  contrast; 12 exact measured combinations remain neutral.
- Preserve all-three-new abstention. Of 2,134 products with exact-new amine,
  aldehyde and isocyanide graphs, 127 lie inside the structural envelope and
  2,007 lie outside, but all 2,134 remain abstained. Envelope membership is
  support evidence, not potency and not authorization to guide this role shift.
- Make zero oracle, synthesis, proposal or generator-trajectory calls. Nonzero
  guidance remains blocked. The next gate is a separately frozen bounded-mode
  lambda-zero identity experiment, and only after review of this census.
- Freeze the derivation config at
  `configs/bio/phase1_ugi_bounded_structural_envelope_derivation_v1.json`
  (SHA-256
  `b0e5e3064aeda0b97d9e8a3fecb9ae00c90d9794fe3e68d240a81de536f73368`),
  the measured-only envelope at
  `results/phase1/ugi_bounded_structural_envelope_v1/envelope.json` (file
  SHA-256
  `2caa3f95f8edb662f56bba881cedf54b398d4530e81bcd51489716f9a4ef7f94`;
  logical SHA-256
  `2ae8eecdf6337bbbe93cc33c6d32ea32f8838c89f6e8bfe775696e0bc6ec07ad`),
  and the census at
  `results/phase1/ugi_bounded_structural_envelope_census_v1/result.json` (file
  SHA-256
  `916e324bd61a1522f10159b49da1761c82f661466e40ef9bf33227f5976a9c79`;
  logical SHA-256
  `681f36827f8ae473654e4f69ca1047a3eac6f8d0632030cd969f47975c19962c`).

## 2026-08-03 - Replace the feature box with a smoothed joint morphology-program prior

- Retain the measured-only eleven-feature envelope and its census as write-once
  historical diagnostics, but supersede them as the primary bounded structural
  controller. The feature-wise Cartesian envelope excluded 460 of the 1,100
  measured AGILE rows and did not preserve their joint morphology distribution.
- Define the structural anchor at the complete-program level: per-role node
  counts, junction budgets, cycle ranks and attachment counts are modeled as one
  12-coordinate Ugi morphology program. The 1,100 measured rows contain 336
  unique complete programs with an inverse-Simpson effective count of 192.06;
  all 1,100 measured rows are included by construction. Component identities do
  not enter this state and the atom and bond graphs remain generated.
- Reject exact-tuple membership as the primary hard bound. Retain it as a strict
  diagnostic ablation and use an occurrence-weighted Laplace kernel over complete
  measured programs as the primary morphology prior. Coordinate scales are the
  occurrence-weighted IQR with an integer floor of one. Freeze the bandwidth and
  local radius at the higher 95th percentile of leave-one-unique-program-out
  nearest-neighbor distance (`0.037037037037037035`). This admits nearby unseen
  complete programs without independently recombining precursor-role marginals.
- The existing 4,096-program schedule was drawn from independent role marginals
  and is diagnostic only: 323 programs are exact anchors, 706 are in the
  calibrated local neighborhood and 3,067 are outside. Among 3,975 valid exact-L1
  terminals, the counts are 312, 687 and 2,976, respectively. The unchanged
  applicability policy provides 3 exact-anchor and 29 local-neighborhood active
  support records, versus 15 outside; this is an audit, not a guidance result.
- Preserve the hard biological boundary in the version-3 multiview chemical
  applicability and role-pattern policy. Morphology-kernel proximity is a
  structural prior, not biological confidence. All 2,134 all-three-new products
  remain abstained, including the 181 in the exact-plus-local morphology
  neighborhood. Open mode remains an exact identity operation.
- Make zero oracle, synthesis, proposal, generator-trajectory or nonzero-guidance
  calls. A new whole-program schedule must be frozen from the exact-plus-local
  joint kernel, followed by lambda-zero identity, before any bounded potency
  pilot. Nonzero guidance remains unauthorized.
- Freeze the audit config at
  `configs/model/phase1_ugi_joint_program_support_audit_v2.json` (SHA-256
  `b06513be3a320b416397e32279a7cbf3150a6a3ddf946bb5006dd737a4d44ac4`),
  the result at
  `results/phase1/ugi_joint_program_support_audit_v2/result.json` (file SHA-256
  `c388495979cd4016e78703d55abbea082df4906f0a299e04f4fc29b8b990416e`;
  logical SHA-256
  `1af4d54043be02c3c7ddb60d503f9a647d3863f766a92d09e595459bc280d8a9`),
  the 336-program ledger (SHA-256
  `4f6972dda92f28a844a06c0212fc7f2a2a94be275fc44142ad4b2836f951a613`)
  and the explicit supersession record (file SHA-256
  `85c5191ec64bff39f2ab96cee4ac68ae8f555c01cffb7bba871f6f656c0b6c29`;
  logical SHA-256
  `bbb8a8ebb71a3278639ecac965d4174552616ec7f4d6e6c02c37d8cd8da95ac8`).

## 2026-08-04 - Separate chemical support from role-policy attrition before guidance

- Treat the one-seed 3.3% active-eligibility rate as the intersection of
  multiple gates, not as evidence that the generator or morphology proposal
  failed. The seed produced 241/256 valid exact-L1 terminals. Twenty-three novel
  terminals passed all four chemical views, while only eight passed the separate
  role-pattern evidence policy.
- In the 3,975-record postvalid broad census, 153 novel products pass all four
  chemical views. The current role policy admits 47 and abstains on 106 (69.3%)
  despite structural interpolation. Exact novelty is provenance, not chemical
  extrapolation; unsupported role shifts still require empirical calibration.
- Identify the whole-product fingerprint radius as the tightest structural
  gate. Do not remove it: it is the only joint-product check. Diagnose and
  validate its relationship to held-family error before changing it.
- Use the fixed worst normalized radius as the primary continuous diagnostic.
  At radius at most one, the six held-component/pair schemes retain 41.4% of OOF
  rows; equal-scheme MAE falls 12.0% and RMSE 10.9%, with exact-label-clustered
  bootstrap intervals excluding zero. This is promising exploratory evidence,
  not independent authorization.
- Do not promote the learned monotone challenger. Although aggregate ranking
  improves, its globally lowest-risk 25% retains only 7.9% of held-aldehyde and
  7.6% of held-head rows while retaining 40.0% in two pair regimes. It fails the
  support-balance safeguard.
- Keep the version-3 policy, oracle, generator and candidate set unchanged.
  Nonzero guidance, dynamic morphology reweighting and partial-state SMC remain
  unauthorized. The next gate is an independently confirmed role-aware
  continuous policy followed by a dynamic terminal census with partial-state
  rollouts.
- Freeze the versioned exploratory config at
  `configs/bio/phase1_ugi_selective_risk_proposal_diagnosis_v1.json` (SHA-256
  `96c3750851c4b96fbecd5015702bcf3784c2706b8a6f6fc74a544f719de8c8b7`),
  the result at
  `results/phase1/ugi_selective_risk_proposal_diagnosis_v1/result.json`
  (file SHA-256
  `4cd54dfc1a5245996a149756162f0d4ad9eb41cb2ee9ab9b28c65464fdd5cedd`)
  and the 6,600-row grouped OOF ledger (SHA-256
  `b0a6f5c2f424c0c54d383016c6999b4c44f8c7a6edbae04269e382fe33743c72`).

## 2026-08-04 - Complete the dynamic frozen-prior terminal census before choosing a controller

- Preserve the selected Ugi generator and sample 1,024 of the 4,096 frozen
  morphology-program rows by deterministic content-hash priority, without
  terminal filtering. Draw two independent states per row, save checkpoints at
  steps 2, 4 and 6 and draw four native continuations per saved state.
- Complete all 32 restartable shards: 24,576/24,576 terminal attempts and 6,144
  partial states are present. Of the terminal attempts, 23,795 (96.82%) are
  valid exact-L1 Ugi products and 16,395 valid product SMILES are unique.
- Independently validate shard artifacts, the exact coordinate lattice,
  deterministic seed namespaces, packed-state hashes, reconstruction of
  omitted deterministic tensors, the ordered ledger and aggregate counts.
- Apply the previously frozen, target-free multiview radius only as a read-only
  terminal label. The census contains 934 `R <= 1` outcomes, including 839 novel
  products and 45 all-three-role-new products. This is further evidence that
  exact component novelty is not synonymous with chemical extrapolation.
- Do not infer a controller from raw in-sample counts. Compare morphology-only
  (`M0`) and saved-partial-state (`M1`) predictors with all cross-validation and
  bootstrap partitions grouped by `program_sha256`; 1,024 sampled rows contain
  941 unique morphology hashes. Four continuations from one state form one
  binomial observation.
- Keep potency, routes, synthesis, proposal engines and candidate selection out
  of this analysis. Dynamic morphology reweighting, delayed SMC and terminal
  screening remain alternatives until the program-disjoint analysis selects the
  simplest controller that provides a material supported-diverse-yield gain.
- Freeze the census logical result as
  `3b440eff098047b8284a49f12d8f190fcd19164f4c9b0e1f086ee4fd3ef3e1e7`
  and the independent validation logical result as
  `a45c44d0404f7ace6f95adefe52ac99ed3d42bc3017dce7df9abed40aae38eb2`.

## 2026-08-04 - Retain morphology as the proposal challenger and reject partial-state SMC

- Score all 24,576 frozen terminal attempts with the previously frozen,
  target-free multiview radius. Define support as a valid exact-L1 product with
  `R <= 1`; do not use potency, routes, synthesis, proposal engines or terminal
  outcomes as model inputs.
- Evaluate the 12-coordinate morphology program (`M0`) and saved categorical
  partial-state summaries (`M1`) with five program-SHA-disjoint outer folds.
  Aggregate the outcomes as `k/24` for one sampled program row and `k/4` for
  one saved partial state. Cluster the 10,000-replicate bootstrap on
  `program_sha256`.
- Confirm that morphology has strong ranking signal: AUROC 0.869, average
  precision 0.164 at 3.80% prevalence and an 8.85% relative Brier reduction.
  A top-quarter allocation raises support to 11.85% and unique-supported-product
  yield from 1.94% to 5.92% per attempt.
- Reject delayed partial-state SMC at this gate. `M1` worsens Brier score by
  0.40% relative to `M0`; its average-precision gain is only 8.1%, its support
  yield gain is 5.6%, and the clustered Brier interval crosses zero.
- Keep plain terminal screening as the current operational controller under the
  strict frozen version-1 gate. Morphology probabilities are under-dispersed,
  and duplicate program rows make the row-level top-quarter allocation cover
  23.8% of unique program hashes. Do not silently relax those gates after
  inspecting the result.
- Treat a support-preserving dynamic morphology proposal on the unused program
  population as the next confirmatory challenger. This is justified by the
  program-disjoint ranking result but is not yet production authorization. It
  does not remove later conservative potency or synthesis tilting; it removes
  only the unsupported early partial-state controller.
- Freeze the controller-analysis result at
  `results/phase1/ugi_dynamic_controller_analysis_v1/result.json` (file SHA-256
  `ad6fad9a79debe09eee1e630c242109dd0aa5f05a8a9c6f9bd732df40ecdae7a`;
  logical SHA-256
  `5b8a0f496d2a30edc5e90fb590646a1936595630d1b6c5522bb8c36e93572740`).

## 2026-08-04 - Confirm morphology-aware allocation on every unused program

- Preserve the frozen support model and score all 3,072 morphology-program rows
  excluded from the controller-development census. Generate one fresh native
  terminal per row without retries, potency, routes, synthesis values or
  proposal-engine calls. All 48 write-once shards complete; 2,979/3,072
  terminals (96.97%) are valid exact-L1 products.
- Evaluate the previously frozen morphology score only after the fresh terminal
  ledger is complete. Define target-free support exactly as before: a valid
  exact-L1 product with multiview structural radius `R <= 1`. The confirmation
  contains 120 supported terminals (3.91%) and 116 unique supported SMILES.
- Correct uncertainty for repeated morphology tuples by resampling whole
  `program_sha256` clusters. The 3,072 rows contain 2,454 unique morphology
  hashes. This correction was frozen before terminal support was scored; it
  replaces the staged row-level bootstrap and does not relax any effect-size
  threshold.
- Confirm out-of-sample morphology ranking: AUROC 0.846 and average precision
  0.155 at 3.91% prevalence. The highest predeclared score quartile contains
  86/768 supported terminals (11.20%) and 83 unique supported SMILES, a 186.7%
  support-rate improvement and 186.2% unique-supported-yield improvement over
  the full unused population. Support rates decrease monotonically across the
  four quartiles (11.20%, 4.43%, 0%, 0%).
- Confirm the actual support-preserving proposal, not only a hard top-quartile
  diagnostic. Its expected supported-terminal rate is 7.65% versus 3.91% under
  the uniform unused-program prior, a 95.9% relative improvement. The
  program-clustered 95% interval for the absolute gain is 2.88--4.63 percentage
  points; the corresponding relative-gain interval is 80.6--112.0%.
- Promote the support-preserving morphology proposal as the sampling allocation
  for the next bounded biological diagnostic. Do not authorize partial-state
  SMC: the prior analysis showed no material incremental partial-state signal.
  This confirmation also does not itself authorize potency or synthesis
  guidance. The next gate remains a separately frozen, matched
  applicability-bounded potency diagnostic against terminal post-hoc ranking.
- Freeze the confirmation census at
  `results/phase1/ugi_morphology_proposal_confirmation_v1/result.json` (file
  SHA-256
  `717c1a024207c3779893742f7f988e9c51ebc9c4722e1335655b2013c1acf20b`;
  logical SHA-256
  `89924c81ae58b9bb65b4c79a21c9fc09c3f9251e547f4a217c651c087b86b810`)
  and the confirmation analysis at
  `results/phase1/ugi_morphology_proposal_confirmation_analysis_v1/result.json`
  (file SHA-256
  `20b4229a18c155bc1433e1df84e5e2b6baf6074cafb4827ed9681571ce852eee`;
  logical SHA-256
  `7f4233290721e516c732b5d3f76e4335110b98b372a33bc747731fafc4e16d16`).

## 2026-08-04 - Freeze exhaustive qualified morphology support separately from evaluation panels

- Correct the support interpretation before the bounded biological diagnostic.
  The 4,096-program fresh pool is a Monte Carlo draw from the role-factorized
  morphology prior; its 1,024 development rows, 3,072 untouched confirmation
  rows and any 128-row paired panel are evaluation schedules, not production
  morphology support.
- Define production support procedurally as the complete Cartesian product of
  the frozen, qualified role-level states: 86 amine-head states, 19
  aldehyde-body/tail states and 35 isocyanide-tail states. This yields 57,190
  distinct whole-program tuples. All qualified role combinations satisfy the
  80-exterior-atom bound; the largest contains 76 exterior atoms.
- This construction exhaustively recombines qualified role-level morphologies,
  including whole-program combinations absent from an observed or sampled
  schedule. It deliberately does not create every integer tuple inside loose
  numerical maxima, because such a construction would introduce unsupported
  tail branching, closure and attachment combinations. Broadening the
  role-level state support remains a separately qualified data/model change.
- Define the broad prior as the product of the three frozen role-state
  probabilities. Apply the independently confirmed morphology score only
  through a 50:50 support-preserving mixture relative to that broad prior. Every
  one of the 57,190 programs retains positive broad and proposal probability;
  the exact importance correction is `p_broad / q`, with a maximum observed
  ratio of 1.99995.
- The broad distribution has an effective support of 5,339.86 programs and the
  scored proposal 2,460.94 programs. This is purposeful computational
  concentration, not support truncation. The proposal does not consume potency,
  routes or synthesis and does not authorize partial-state SMC or potency
  guidance. The next gate is oracle reliability under the morphology-enriched
  supported distribution.
- Freeze the config at
  `configs/model/phase1_ugi_complete_morphology_proposal_v1.json` (SHA-256
  `53083b70393f8e2815d39154daa8d2d714908f93cfc1b830f37051d394c2b580`),
  the result at
  `results/phase1/ugi_complete_morphology_proposal_v1/result.json` (file SHA-256
  `93543b3b563338f24435c595799118bbf416e64a8b7f0998a642b2065ba064d2`;
  logical SHA-256
  `bf8321431558ae4e7122b91c3dc4487b30919d0c83f930edf488118ddf133a66`)
  and the 57,190-program ledger at SHA-256
  `273fdb44bc2df8405ea4835ac32c7305bfc2d59caa8ec51b138464f57bb17675`
  (logical SHA-256
  `468a1bfe692fc2a0036abc6a19530edc81c09c8e49733b481b1ad7808777b6b1`).
  An independent rebuild reproduced both artifacts byte-for-byte.

## 2026-08-04 - Retain morphology enrichment but withhold potency authority for novel heads

- Stress-test the frozen HeLa oracle under the confirmed morphology proposal
  without generating new predictions. Map each curated AGILE product to its
  exact generator morphology and reweight the six structured out-of-fold
  schemes by `q_M / p_broad` inside the previously frozen multiview chemical
  support radius (`R <= 1`). The morphology score was trained without potency
  or oracle error.
- Preserve the generator training-fold boundary. Of 1,100 measured products,
  950 (86.36%) map into the complete 57,190-program production support. The 150
  excluded products occupy 76 morphology programs present only outside the
  generator training-fold prior. Do not add those held-out programs merely to
  improve audit coverage; evaluate the 5,700 structured OOF rows whose
  morphologies the production generator can actually sample.
- Observe favorable aggregate covariate-shift behavior. Relative to the broad
  supported distribution, morphology enrichment reduces equal-scheme MAE by
  7.71% (clustered 95% interval, 4.65--10.82%) and RMSE by 4.70% (1.73--7.97%),
  increases 90% conformal coverage from 83.84% to 85.98%, and increases the
  equal-scheme weighted midrank correlation from 0.626 to 0.649. The minimum
  per-scheme importance-weight ESS fraction is 0.802.
- Do not declare the full potency gate passed. Five structured schemes retain
  positive fit and strong ranking, but the held-head scheme has weighted
  `R^2 = -0.124` despite a weighted midrank correlation of 0.405. This fails
  the predeclared per-scheme absolute-fit gate and specifically withholds
  potency authority for newly generated amine heads.
- Keep the aldehyde--isocyanide held-pair result as a separately defensible
  diagnostic lane (`R^2 = 0.615`, weighted midrank correlation 0.769, 89.08%
  conformal coverage), but do not silently redefine the failed global gate.
  Any narrowed potency experiment must explicitly abstain on the amine-only
  novelty pattern and be frozen as an adaptive, role-restricted diagnostic.
- Freeze the config at
  `configs/bio/phase1_ugi_morphology_enriched_oracle_audit_v1.json` (SHA-256
  `66bf033adb20a1e82c3b33f7476221e03377f75fcc9738b33bd4ba807d703fe1`),
  the result at
  `results/phase1/ugi_morphology_enriched_oracle_audit_v1/result.json` (file
  SHA-256
  `e3a4fea5c3d4cbdf6db0772a04cca51e5af5ac653b44d05a5e774fb3f2c5de8e`;
  logical SHA-256
  `37f99079d7610a9c4c0e25c3952b886a4f2b3724f1d6fd7370727a689e6a42df`)
  and the 5,700-row ledger at SHA-256
  `0205b9dd4aa124c86b0d193ca14d73679b17e2df7870e9f53b7db642877d74d0`
  (logical SHA-256
  `cb30fc7deaf6b70464940ab74c22e8dd5fcdb978da4f2b8c679f75506ff34abb`).
  An independent rebuild reproduced both artifacts byte-for-byte.

## 2026-08-04 - Separate applicability allocation, oracle authority and potency allocation

- Describe `q_support` as an applicability-enriched, support-preserving
  morphology proposal. It raises the probability of completing a molecule
  inside the fixed multiview chemical-support region but never certifies an
  individual terminal. The terminal chemical policy remains authoritative.
- Keep two estimands distinct. Report realized production yield directly under
  `q_support` without correcting away the intended enrichment. Use
  `p_broad / q_support` only when estimating an expectation under the original
  broad prior from proposal samples. Report both quantities when both questions
  are scientifically relevant.
- Interpret the morphology-reweighted structured OOF audit narrowly: the
  proposal does not degrade, and modestly improves, oracle behavior on
  reweighted measured AGILE chemistry. It does not establish calibration on
  newly generated chemistry within the same morphology program; that requires
  prospective generated compounds.
- Treat the familiar-head plus novel aldehyde/isocyanide potency lane as a new,
  adaptive hypothesis motivated by the failed held-head gate. Freeze its design
  before further evaluation and require prospective confirmation. Do not call
  the role restriction a globally predeclared success.
- Define biological authority lexicographically. Potency may be nonneutral only
  for a valid exact-L1 terminal that passes every frozen product/component
  chemical-support view, the authorized role-novelty policy and the frozen
  uncertainty threshold. Outside that set the potency value is exactly neutral;
  predicted potency cannot compensate for failed applicability.
- Test a pre-generation morphology-level potency value before considering a
  production potency proposal. Evaluate both conservative continuous potency
  and top-potency-quartile enrichment with program-group-disjoint validation,
  clustered uncertainty, effective-support and role-diversity safeguards. Do
  not require exact regression calibration if ranking and enrichment are the
  decision target, but do require positive independently estimated signal.
- If qualified, define the potency morphology proposal as a broad-support
  mixture between `p_broad` and a normalized
  `q_support * exp(beta * V_potency)` term. Freeze the mixture strength and beta
  using ESS, diversity and trust-region budgets rather than maximum predicted
  potency. Compare it against both broad-prior and `q_support` post-hoc-ranking
  arms at matched generator, oracle and candidate budgets.
- Keep the main generator open over all three precursor-derived subgraphs. A
  familiar-head clamp is permitted only as a controller diagnostic. In the main
  campaign, novel-head terminals remain generatable and synthesis-eligible but
  receive biological abstention until head-generalization evidence improves.

## 2026-08-04 - Retain morphology enrichment; gate complete-molecule MH as an optional refinement

- Do not replace the independently confirmed `q_support` morphology proposal
  with Metropolis--Hastings. Use `q_support` for first-stage allocation, generate
  complete valid Ugi products, and apply the frozen terminal applicability
  policy. At the observed 7.65% supported-terminal yield, this direct baseline
  is operationally tractable and remains the production reference.
- Distinguish filtering from a sampling claim. Retaining terminals with
  `A(x) = 1` produces the `q_support` distribution conditioned on the frozen
  applicability event. Post-hoc potency ranking or resampling is a separate
  baseline and must not be described as trajectory-level potency guidance.
- Consider clean complete-molecule MH only after the morphology-level potency
  signal passes its independent gate and only as an additional matched arm.
  Initialize multiple chains from diverse applicable terminals produced by
  `q_support`; do not ask chains initialized from the broad prior to discover a
  rare hard-walled region.
- Apply a hard applicability indicator only to the explicitly bounded
  exploitation target. It must not redefine FORGE's global molecular support.
  Preserve a separate broad exploration arm in which novel-head and other
  unauthorized biological lanes remain generatable but receive neutral potency
  utility.
- Do not claim exact MH over terminal molecular graphs unless the required
  density ratio is tractable. Audit whether the frozen generator exposes every
  stochastic transition probability, deterministic mask and canonicalization,
  and whether multiple traces can decode to the same terminal graph. If the
  terminal marginal is unavailable, formulate and label the chain over the
  augmented `(morphology, generation trace, terminal graph)` state.
- Any structured role-subgraph corruption/regeneration kernel must return a
  complete valid Ugi product and expose both forward and reverse proposal
  probabilities. Include occasional broad independence refreshes to improve
  irreducibility and cross-family mixing; correct every proposal in the MH
  ratio. Validity alone is not evidence of reversibility.
- If a soft applicability-risk annealing path is used to initialize or temper
  chains, distinguish its intermediate distributions from the final hard-wall
  target. Potency may activate only after the frozen applicability and
  role-generalization policy authorizes it.
- Require MH to outperform `q_support` plus terminal applicability selection
  and matched post-hoc potency ranking at equal generator calls, oracle calls,
  wall time and final portfolio size. Evaluate unique supported-and-potent
  molecules per compute, integrated autocorrelation/effective sample size,
  between-chain agreement, unique ancestors, component-family mixing, chemical
  diversity and prior coverage. Acceptance rate alone is not a success metric.
- Abandon the MH arm if proposal ratios are not exact, chains are sticky or
  family-trapped, final effective sample size is poor, or it provides no
  matched-compute gain. Such an outcome leaves the primary FORGE contribution
  intact: applicability-enriched generation followed by terminal selection,
  with potency guidance retained only where independently justified.

## 2026-08-04 - Select a stronger morphology challenger without relaxing applicability

- Freeze a 16-setting development sweep over mixture strengths
  `rho = {0.25, 0.50, 0.75, 0.90}` and score powers
  `kappa = {0.50, 1.00, 1.50, 2.00}`. Keep the V3 multiview applicability
  thresholds, valid exact-L1 requirement and all 57,190 qualified programs
  unchanged. Do not consume potency, routes or new generator trajectories.
- Predeclare proposal floors before evaluating the grid: at least 35% of the
  broad inverse-Simpson effective program count, 40% of broad Shannon support,
  importance-weight ESS fraction at least 0.50, no single-program probability
  above 0.005, at least 50% of every broad role-marginal effective count, at
  least 99% of broad exact-L1 validity, and at least 50% of the reference
  proposal's supported molecule and component effective counts.
- Select `rho = 0.50, kappa = 1.50` as the sole challenger. On the reused
  3,072-terminal confirmation ledger it raises counterfactual expected support
  from 7.653% to 8.471% (cluster-bootstrap 95% interval, 6.734--10.249%) while
  passing every frozen floor. Its full-support inverse-Simpson effective count
  is 2,056, Shannon effective count is 6,156, importance ESS fraction is 0.671
  and maximum program probability is 0.00262.
- Reject more aggressive settings despite their larger apparent support yield.
  The maximum-yield setting (`rho = 0.90, kappa = 2.00`) reaches 13.354% only
  counterfactually but fails the effective-support, importance-ESS, amine
  marginal, Shannon-support and exact-L1 floors. Do not trade broad support for
  a prettier acceptance percentage.
- Diagnose remaining abstention under the selected challenger. Approximately
  83.99% of probability fails multiple chemical views, 3.66% fails the product
  view alone, 0.40% fails the aldehyde view alone and 3.48% is invalid or lacks
  exact L1 reconstruction. Product fingerprint and descriptor failures remain
  the largest nonexclusive modes (87.01% and 81.52%), showing that exact
  terminal chemistry—not morphology alone—is now the main bottleneck.
- Catch and correct an input-version mismatch during implementation: a first
  diagnostic reconstruction used V2 applicability thresholds while the frozen
  confirmation labels used V3. Add a hard invariant requiring the reconstructed
  abstention categories to reproduce every stored support label exactly before
  any sweep result can be written.
- Do not promote the challenger from this reused ledger. The 3,072 outcomes are
  no longer untouched after the original proposal confirmation, so using them
  to choose `kappa` makes the result development evidence. Keep the confirmed
  50:50, power-1.0 proposal as the production reference until one new untouched
  program/terminal experiment confirms the frozen challenger.
- Freeze the sweep at
  `configs/bio/phase1_ugi_morphology_proposal_strength_sweep_v1.json` and the
  result at
  `results/phase1/ugi_morphology_proposal_strength_sweep_v1/result.json`.
  MH and partial-state SMC remain unauthorized.

## 2026-08-04 - Promote the power-1.5 applicability proposal after fresh terminal confirmation

- Freeze `rho = 0.50, kappa = 1.50` before drawing a second independent terminal
  outcome for each of the 3,072 development-program rows. Use new particle and
  terminal seeds, the unchanged frozen generator, no retries or repairs, and no
  potency, route, synthesis or nonzero-guidance calls. The program identities
  are reused; only the stochastic terminal outcomes are fresh.
- On the fresh outcomes, observe support rates of 3.874% under the broad prior,
  7.792% under the previously confirmed power-1.0 proposal and 8.642% under the
  power-1.5 challenger. The challenger adds 0.850 percentage points, or 10.91%
  relative, over the current proposal. A paired program-cluster bootstrap gives
  a 95% interval of 0.497--1.235 percentage points (7.14--14.55% relative).
- Preserve validity and useful diversity. Expected exact-L1 validity changes
  from 96.466% under the previous proposal to 96.255% under the challenger.
  Supported-molecule inverse-Simpson effective count changes from 96.08 to
  84.21, retaining 87.64%. All development-time full-support, role-marginal,
  importance-ESS and concentration floors remain satisfied.
- Record that the fresh point rates were inspected before the final paired
  bootstrap configuration was frozen. Treat the confidence interval as a
  transparent paired confirmation diagnostic, not a pristine preregistered
  test. The proposal parameters, applicability thresholds, generator and
  terminal seeds were nevertheless frozen before generating the fresh outcomes.
- Promote the challenger as the production applicability-oriented morphology
  proposal over all 57,190 qualified programs. Every program retains positive
  probability. The promoted proposal has inverse-Simpson effective support
  2,056, Shannon effective support 6,156, importance ESS fraction 0.671 and
  maximum single-program probability 0.00262.
- Do not interpret promotion as terminal applicability certainty: about 91.4%
  of proposal-weighted terminals still abstain. Keep exact terminal filtering
  authoritative. Potency guidance, MH and partial-state SMC remain unauthorized;
  the next gate is a role-restricted morphology-level potency-signal test.
- Freeze the adjudication at
  `results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json`
  and the complete promoted proposal at
  `results/phase1/ugi_promoted_morphology_proposal_v1/result.json` with its
  57,190-row `proposal_ledger.jsonl.gz`.

## 2026-08-04 - Restrict the potency claim to the novelty regime actually tested

- Audit every fold of `held_aldehyde_isocyanide_pair_5fold` before constructing
  a potency proposal. All 1,100 test rows contain aldehyde and isocyanide
  identities that remain present in the corresponding fitting data; only the
  exact aldehyde-isocyanide pair is absent. No fold tests an exact-new aldehyde,
  an exact-new isocyanide or two simultaneously exact-new tail identities.
- Supersede the earlier nonexecuting `amine_only` potency lane because the
  morphology-enriched held-head stress test has negative absolute fit
  (`R2 = -0.124`), despite positive rank correlation. Do not let that lane
  authorize biological guidance.
- Define the first morphology-potency signal cohort as familiar measured heads
  and individually represented aldehyde/isocyanide identities in unseen pairs,
  with all four frozen chemical views inside `R <= 1`. Continuous similarity
  remains necessary but cannot upgrade an unseen-pair validation into an
  exact-new-component claim.
- Keep exact-new heads and tails in the unguided biological exploration lane.
  They remain generatable and synthesis-assessable, but receive neutral potency
  utility until simultaneous role-identity or structural-family holdouts, or
  prospective measurements, support a stronger claim.
- Freeze the read-only audit at
  `results/phase1/ugi_potency_novelty_lane_audit_v1/result.json`.

## 2026-08-04 - Pass the morphology-to-observed-HeLa signal gate

- Fit only low-capacity morphology models to observed `expt_Hela` MTP within
  the authorized 525-product cohort. Do not use oracle predictions as training
  targets and do not advance generator trajectories.
- Select the role-factorized additive ridge model, not the shallow nonlinear
  diagnostic. Under exact-program group-disjoint cross-validation it achieves
  Spearman 0.452 (program-cluster bootstrap 95% interval 0.369--0.527), reduces
  MAE from 2.602 for the constant baseline to 2.262 and raises observed MTP by
  1.636 in the highest-scoring predicted quartile (95% interval 1.127--2.216).
- Require secondary blocking by individual role morphology. The selected model
  remains positive when holding out aldehyde states (Spearman 0.250),
  isocyanide states (0.291) and aldehyde-isocyanide state pairs (0.359).
  Seventeen of nineteen evaluable heads have positive within-head rank
  correlation; the median is 0.213.
- Interpret this as evidence for pre-generation morphology-level potency
  allocation, not atom- or bond-level guidance and not prospective potency.
  Potency tilting remains unpromoted until it beats the same terminal ranking
  policy at matched generator and oracle budgets.
- Freeze the result at
  `results/phase1/ugi_morphology_potency_signal_v1/result.json`.

## 2026-08-04 - Freeze a nested potency proposal for matched diagnosis

- Fit the selected additive model on the full authorized cohort with the modal
  nested-CV ridge penalty and derive a conservative LCB using the exact-program
  out-of-fold absolute-residual 90% conformal radius. Define positive potency
  utility from the upper half of the calibration LCB distribution.
- Give zero potency utility, rather than zero generative probability, to
  morphology programs containing role states absent from the authorized
  biology cohort. All 57,190 qualified morphology programs remain possible.
- Define morphology value as the frozen applicability support score multiplied
  by conservative potency utility. Tilt from the promoted applicability
  proposal, never directly from the broad prior.
- Sweep only the frozen mixture and inverse-temperature grid under predeclared
  diversity, effective-support, importance-ESS, marginal-support and maximum-
  concentration safeguards. Select `gamma = 0.75, beta = 8.0` for the matched
  diagnostic. It increases model-implied supported-potency value by 77.2%,
  while retaining inverse-Simpson effective support 1,876 versus 2,056,
  Shannon effective support 5,832 versus 6,156 and broad-over-proposal
  importance ESS fraction 0.654.
- Only 260 programs have role morphology fully represented in the authorized
  biological cohort; their mass changes from 13.52% to 16.01%. This narrow
  authority is explicit and does not remove or biologically score other
  programs.
- Do not promote potency tilting from this distribution calculation. The next
  gate is the matched three-arm terminal experiment: broad plus ranking,
  applicability proposal plus the same ranking, and nested potency proposal
  plus the same ranking.
- Freeze the sweep at
  `results/phase1/ugi_morphology_potency_proposal_sweep_v1/result.json`.

## 2026-08-04 - Keep synthesis guidance on its independent exact-evidence gate

- Treat the exact route stack as qualified through lambda-zero only. The typed
  planner, isolated cache, exact L1/L2/L3 evaluator, binary route-completion
  utility, restartable controller and selected-v3 lambda-zero identity checks
  are complete, but no nonzero synthesis-guidance outcome exists.
- Run the single reviewed `lambda = 0.25` production calibration seam next.
  Complete its missing config, CLI and tests before execution; do not substitute
  Graph2Edits for this gate.
- Keep Graph2Edits as optional proposal-engine work. Its runtime and checkpoint
  are pinned and deterministic smoke inference passes, but the real three-lane
  benchmark and production activation remain incomplete. Proposals cannot
  create route evidence or enter `V_syn` without exact forward/evidence closure.

## 2026-08-04 - Do not promote morphology potency tilting from the strict confirmatory lane

- Generate 3,072 fresh terminals in each of the broad, promoted-applicability
  and nested-potency morphology arms using common uniforms and common
  particle/terminal seeds. Preserve one attempt per arm and draw, with no
  oracle, route, synthesis, repair or retry during generation. Obtain 2,964,
  2,946 and 2,928 strict exact-L1 terminals, respectively.
- Apply the novelty regime literally tested by the held-pair validation:
  individually familiar amine, aldehyde and isocyanide identities in an unseen
  product combination, with all four chemical views interpolative. Every
  fully familiar generated combination is an exact measured product; every
  novel product contains at least one exact-new component. Consequently, zero
  terminals qualify and no oracle call is made under this strict lane.
- Preserve this as a valid null rather than loosening the applicability rule
  after observing generation. Morphology potency tilting is not promoted; the
  production default remains promoted applicability allocation followed by
  terminal conservative ranking.
- Freeze generation at
  `results/phase1/ugi_matched_morphology_terminal_generation_v1/result.json`
  and the confirmatory adjudication at
  `results/phase1/ugi_morphology_potency_matched_v1/result.json`.

## 2026-08-04 - Rank exact-new components continuously, but keep the result exploratory

- Correct the interpretation that exact component novelty is itself an
  applicability failure. An unseen exact identity can remain chemically
  interpolative. Treat identity novelty as a risk covariate and reporting
  stratum; require exact L1, an unseen product, and interpolative product,
  amine, aldehyde and isocyanide views.
- Use only the two pre-existing pattern-specific calibration scales. The
  familiar-head/two-new-tail pattern supplies 38 matched oracle calls per arm;
  the exact-new-head-only pattern supplies four. Other exact-new patterns lack
  a frozen calibration and abstain.
- In the two-new-tail stratum, the nested-potency arm yields 24 unique
  conservative-high-potency products versus 20 for applicability-only at the
  same 38 oracle calls; mean conservative utility is 0.474 versus 0.391. In
  the small new-head stratum the corresponding counts are three versus two.
  Combined, the point gain is five unique high-utility products, but the paired
  95% interval crosses zero (-0.00065 to 0.00260 per generator call).
- Interpret the direction as useful exploratory ranking evidence, not a passed
  guidance gate. Simultaneous held-component/family validation or prospective
  measurements remain necessary before calibrated absolute-potency claims for
  exact-new components. Do not authorize candidate lock or reverse the strict
  confirmatory null.
- Freeze the result at
  `results/phase1/ugi_continuous_novelty_matched_ranking_v1/result.json`.

## 2026-08-04 - Qualify synthesis guidance mechanically but record a null route endpoint

- Preserve the first lambda-0.25 execution failure as a fail-closed artifact:
  generation and route work completed, but post-run publication rejected a
  tuple/list representation mismatch. Inspect no terminal, ancestry or route
  outcome, add a direct regression test, and authorize exactly one same-seed,
  same-strength operational rerun under a signed amendment and separate cache.
- The amended rerun produces nonidentity guidance: three ancestry changes at
  checkpoints four and six, three of 64 terminal molecular identities change,
  and selected-set overlap is 30/32. Thus the non-post-hoc synthesis mechanism
  is functional.
- The endpoint is null. Guided and post-hoc arms each produce one route-complete
  molecule among 56 unique productive terminals and one among 32 selected
  candidates, under exactly matched generation, planner and verifier budgets.
- Do not promote synthesis tilting or automatically run stronger lambdas. The
  current exact route signal is too sparse for this controller to improve yield
  on the frozen assignment. Retain post-hoc exact route assessment while route
  evidence coverage is expanded.
- Freeze the successful result at
  `results/phase1/ugi_production_synthesis_guidance_seam_v4_retry1/result.json`,
  the amendment at
  `configs/model/phase1_ugi_production_synthesis_guidance_seam_v4_retry_amendment_v1.json`,
  and preserve the original failure receipt unchanged.

## 2026-08-04 - Retain proposal-augmented routing but reject the family-readiness trajectory tilt

- Resolve 27 apparently ambiguous Graph2Edits proposals as semantic duplicates
  of a qualified primary-alcohol oxidation program. Across the 30 one-gap
  diagnostic components, 27 have graph-consistent known-family discovery
  hypotheses; eight proposal rows were already classified as known family and
  138 remain genuinely new-family hypotheses. Proposal discovery remains a
  search receipt, not route evidence.
- Expand terminal route triage with independently adjudicated family evidence.
  In the frozen 124-product panel, 23 broad-arm and 41 applicability-arm
  products are exact or family-all-current route-ready; balanced arm-wise
  comparison can support 23 products per arm. Family-partial and unresolved
  products remain separate and are not called route-closed.
- Repeat the single reviewed `lambda = 0.25` matched guidance test using the
  frozen family-readiness utility while excluding raw proposal scores and
  proposal-only hypotheses. The guided and post-hoc arms each assess 64 final
  particles and 56 canonical representatives. Guidance yields two route-ready
  final products versus three post hoc, changing eight of 64 molecular
  identities; it therefore does not improve the matched endpoint.
- Diagnose sparse, weakly actionable intermediate reward. Only five of 48
  morphology-local checkpoint groups contain mixed route utility and seven
  resample. Because ancestry can change only among four particles sharing one
  morphology, the controller cannot exploit the broader terminal-panel route
  coverage. Proposal-engine occurrences in this run comprise 19 semantic
  known-family rediscoveries and five new-family hypotheses; the latter
  correctly remain evidence-free.
- Do not tune additional lambdas on the same schedule and do not promote
  mid-trajectory synthesis guidance. Retain Graph2Edits and the qualified
  reaction-family registry for complete-product L1/L2/L3 search before the
  prospective panel is locked. This is post hoc to molecular generation but
  not post hoc to experimental selection.
- Freeze the successful matched run at
  `results/phase1/ugi_graded_route_readiness_guidance_v1_retry3/result.json`
  and its adjudication at
  `results/phase1/ugi_synthesis_guidance_failure_audit_v1/result.json`.

## 2026-08-04 - Authorize one decision-aligned potency challenger, without promotion

- Audit whether the failed continuous morphology-potency proposal targeted the
  wrong decision variable. Fit only low-capacity logistic models to the upper
  fold-local quartile of observed HeLa MTP in the same 525-product authorized
  cohort; do not change the applicability region, generator or terminal oracle.
- Select the role-factorized logistic model. Under exact-program grouped
  validation it achieves ROC AUC 0.623 (program-cluster bootstrap 95% interval
  0.565--0.679) and average precision 0.355 at 0.255 prevalence. Its
  highest-probability quartile raises observed HeLa MTP by 1.177 (95% interval
  0.693--1.673).
- Require transfer beyond exact programs. AUC remains 0.601 with aldehyde
  morphology states held out, 0.603 with isocyanide states held out and 0.638
  with aldehyde--isocyanide state pairs held out. Within-head stability also
  passes: 12 of 15 evaluable heads have AUC above 0.5, the median is 0.551 and
  13 of 15 show positive top-quartile enrichment.
- This read-only result does not promote potency tilting. Freeze one
  support-preserving challenger over the same conservative 260 authorized
  morphology programs, leaving all 57,190 programs possible. The selected
  `gamma = 0.75, beta = 8.0` proposal improves its model-implied
  supported-high-potency value by 25.4% while passing every pre-existing
  diversity, concentration and importance-ESS safeguard.
- Authorize exactly one fresh 3,072-draw-per-arm matched terminal comparison
  against the promoted applicability proposal with identical terminal support
  checks and conservative ranking. Promote only on realized unique,
  applicable, high-potency yield at matched compute; otherwise close
  morphology potency tilting and retain terminal ranking.
- Freeze the signal audit at
  `results/phase1/ugi_morphology_high_potency_challenger_v1_retry1/result.json`
  and the pre-generation proposal at
  `results/phase1/ugi_morphology_high_potency_proposal_v1/result.json`.

## 2026-08-04 - Close morphology-level potency tilting after the fresh matched challenger fails

- Complete the single pre-authorized fresh comparison with 3,072 terminal
  attempts in each of the broad, promoted-applicability and nested-potency arms.
  Common uniforms and particle/terminal seeds are shared across arms. No
  terminal scoring, selection, repair, retry, route or synthesis call occurs
  during generation. Exact-L1 validity remains matched: 2,971 broad, 2,955
  applicability and 2,962 nested-potency terminals.
- Apply the same frozen multiview support policy, novelty-pattern calibration,
  deterministic potency-independent oracle subset and matched oracle budgets.
  In the authorized familiar-head/two-new-tail lane, applicability enrichment
  yields 90 eligible terminals before budget matching and the nested-potency
  proposal yields 82. With 38 oracle calls per arm, the corresponding unique
  conservative-high-potency product counts are 27 and 23.
- The nested-minus-applicability difference is -0.00130 unique high-potency
  products per generator call (paired 95% interval, -0.00228 to 0.00098). The
  restricted new-head diagnostic also does not improve (two versus zero unique
  conservative-high-potency products at three oracle calls per arm).
- Conclude that the measured morphology classifier signal was real enough to
  justify one test but did not translate into improved terminal discovery
  efficiency. The failure is not explained by molecular validity and cannot be
  repaired by selecting another proposal from the same biological data.
- Do not promote morphology-level potency tilting, partial-state potency SMC or
  complete-molecule MH. Close additional same-data potency-proposal tuning.
  Retain the independently promoted applicability morphology proposal followed
  by exact terminal support assessment and conservative terminal HeLa ranking.
  This remains a generative applicability controller plus post-generation
  potency selection, not trajectory-level potency optimization.
- Freeze generation at
  `results/phase1/ugi_high_potency_challenger_terminal_generation_v1/result.json`,
  ranking at
  `results/phase1/ugi_high_potency_challenger_continuous_ranking_v1/result.json`
  and final adjudication at
  `results/phase1/ugi_high_potency_challenger_adjudication_v1/result.json`.
# 2026-08-04 - Independent retrosynthesis audit identifies evaluator undercoverage

- Ran a hash-pinned, diagnostic-only AiZynthFinder 4.4.1 lane on 26 unresolved
  conservative-high Ugi components and 14 matched route-ready controls.
- The public policy returned top-10 single-step hypotheses for all 40 targets.
  An eight-second bounded public-stock search solved one of nine unresolved targets
  and none of seven matched controls; the solved unresolved oxoester aldehyde closed
  in one step to two public-stock precursors.
- Decision: current unresolved labels are evaluator/ledger states, not evidence of
  molecular unsynthesizability. Keep post-hoc L1/L2/L3 routing as the production
  procedure. Admit learned outputs only as quarantined hypotheses requiring
  independent verification and evidence adjudication.
- The prior negative synthesis-guidance result remains evaluator-limited. Re-run
  synthesis tilting only after a proposal-augmented evaluator passes a frozen route
  recovery benchmark, and promote it only if it outperforms matched post-hoc routing
  on unique, diverse, route-closed yield per compute.
- Evidence:
  `docs/PHASE1_AIZYNTHFINDER_COMPONENT_ROUTE_DIAGNOSTIC.md` and
  `results/phase1/aizynthfinder_component_route_diagnostic_v1/result.json`.

## 2026-08-05 - Qualify Graph2Edits for source-neutral route adjudication, not production authority

- Freeze a 120-target proposal-recovery benchmark before revealing exact route
  truth. Thirty-six targets carry hidden exact precursor sets; 12 adversarial
  controls remain reserved for the full operational benchmark.
- AiZynthFinder recovers 18/36 exact routes at top 1 and 19/36 at top 5.
  Graph2Edits recovers 22/36 at top 1 and 29/36 at top 5, including 13/18 held
  reaction-family targets. Adding AiZynthFinder does not increase the top-5
  union beyond Graph2Edits' 29/36.
- All seven joint misses are amine formylations already represented in the
  frozen exact-source lipid registry. Retain the hybrid architecture: the
  evidence-backed lipid registry supplies known chemistry, Graph2Edits supplies
  broad source-neutral proposals and AiZynthFinder remains an independent
  diagnostic challenger.
- Authorize Graph2Edits for the next adjudication experiment only. Raw learned
  proposals remain hypotheses and cannot set route closure, synthesis value or
  generation-time guidance. Require exact forward reconstruction, substrate
  scope, operational compatibility, independent evidence and L3 terminal
  closure before changing the production evaluator.
- The existing negative synthesis-guidance result remains frozen. Repeat a
  matched synthesis-tilt challenger only if proposal adjudication increases
  independently supported, nonuniform route value on generated candidates;
  promotion still requires higher unique, diverse, route-closed yield than
  matched post-generation routing.
- Evidence:
  `docs/PHASE1_SINGLE_STEP_RECOVERY_AUDIT.md` and
  `results/phase1/hybrid_single_step_recovery_audit_v1/result.json`.

## 2026-08-05 - Remove reaction-family labels as chemistry gates

- Invalidate the first source-neutral proposal adjudication. It circularly
  required a legacy `program_family` annotation before allowing an
  independently forward-consistent Graph2Edits proposal to infer that program.
  This converted missing knowledge-base coverage into a false chemical
  rejection and must not support production or manuscript claims.
- Under proposal-first adjudication, 109 of 113 targets (96.5%) have at least
  one graph-consistent hypothesis. Fifty-eight targets remain inside the exact
  whole-program source envelope and 13 are also closed to the frozen current
  terminal inventory. Reaction-family names are henceforth evidence-retrieval
  annotations rather than prerequisites for proposal survival.
- Re-execute the proposed chemistry step by step. All 109 coherent programs
  reproduce every expected product: 107 esterifications, 107 primary-alcohol
  oxidations, two amine formylations and two formamide dehydrations. Twenty-six
  programs have all exact starting materials current. Thirteen of these were
  rejected only because the full product lay outside the exact AGILE spacer or
  size envelope.
- Represent a route as a sequence of transformation steps. Preserve the exact
  whole-program source envelope as a higher-confidence evidence tier, but do
  not let its chain-length or spacer-length categories define chemical
  possibility. Unknown-family proposals remain quarantined hypotheses rather
  than disappearing from search.
- Call the 26 outcomes `terminal-closed route hypotheses`, not experimentally
  proven routes or synthesis-success probabilities. Missing current terminal
  evidence remains unresolved knowledge, not chemical incompatibility.
- Invalidate the first stepwise route-value contrast because it replaced rather
  than unioned legacy positive route states. The corrected monotone contrast
  preserves all five legacy positives but still fails the pre-existing rerun
  gate: four of 48 checkpoint groups are mixed and three productive finals are
  route-ready. Do not rerun or tune synthesis tilting on that small old
  checkpoint panel.
- Continue post-generation proposal-first routing on the larger current pools
  while expanding terminal-material coverage. A new guidance experiment is
  authorized only after independently qualified route states create adequate
  nonuniform checkpoint contrast.
- Evidence:
  `results/phase1/ugi3_source_neutral_proposal_adjudication_v2/result.json`,
  `results/phase1/ugi3_stepwise_route_program_adjudication_v1/result.json` and
  `results/phase1/ugi3_stepwise_route_value_contrast_v2/result.json`.

## 2026-08-05 - Freeze one hybrid route stack and retain synthesis assessment before panel lock

- Keep the route architecture singular and ordered: exact evidence-backed
  registry first, Graph2Edits as the primary learned single-step proposer and
  AiZynthFinder only as a residual multistep/public-stock challenger. Every
  learned proposal enters the same exact forward-execution, current-terminal
  and evidence adjudication pipeline; neither model score nor a named reaction
  family establishes route closure.
- On 64 checkpoint components without a coherent Graph2Edits proposal,
  AiZynthFinder returns a one-step proposal for all 64 and a public-ZINC-stock
  solution for 44. On 97 forward-verified programs whose remaining blocker is
  an upstream leaf, it returns proposals for all 97 and public-stock solutions
  for 29. These are proposal-only diagnostics, not current procurement evidence
  or experimentally demonstrated routes.
- A hybrid counterfactual using those public-stock solutions as if they were
  authoritative would raise the frozen checkpoint contrast to 21 of 48 mixed
  groups and 20 unique route-ready productive finals. Interpret this only as an
  upper-bound diagnosis that route-knowledge coverage can suppress contrast;
  do not use it to authorize guidance.
- Independently verify exact current US supplier evidence for four high-impact
  leaves: 9-decen-1-ol, 1-decanol, 9-decyn-1-ol and pentylamine. Their identities
  and formulas are structure-checked, evidence is item-level and time-limited,
  and procurement is not treated as substrate-scope or synthesis-success
  evidence.
- Recompute the proposal-aware checkpoint contrast without changing its frozen
  gates. The current evidence raises unique route-ready productive finals from
  three to six and creates 14 utility transitions from zero to one. It yields
  four newly terminal-closed proposal components and seven of 48 mixed groups
  (14.58%), with at least one mixed group at every checkpoint.
- Do not move the preregistered gate because the result is close. The challenger
  still fails the minimum five new qualified components, eight mixed groups and
  15% mixed-group fraction. Synthesis tilting, lambda tuning, partial-state
  synthesis SMC and a synthesis-guided headline remain unauthorized.
- Use proposal-first L1/L2/L3 routing after molecular generation but before the
  prospective panel is locked. The manuscript identity is synthesis-aware
  whole-lipid generation with prospective validation. The central causal
  comparison remains the promoted applicability proposal versus broad
  generation under identical terminal ranking, route assessment, synthesis,
  formulation and assay budgets.
- Evidence:
  `results/phase1/aizynthfinder_checkpoint_residual_v1/result.json`,
  `results/phase1/ugi3_hybrid_proposal_closure_sensitivity_v1/result.json`,
  `configs/route/phase1_ugi3_hybrid_high_impact_leaf_terminals_v1.json`,
  `results/phase1/ugi3_precursor_leaf_closure_audit_v2/result.json` and
  `results/phase1/ugi3_proposal_aware_checkpoint_contrast_v2/result.json`.

## 2026-08-05 - Freeze the all-fold Ugi production-refit contract

- The selected Ugi architecture will be refit once from random initialization
  on all 112,386 exact-forward Ugi products. Broad non-Ugi pretraining remains
  an optional ablation and is not the initializer for the primary model.
- Preserve the selected run's expected weighted draws per structural record:
  1,000 steps at batch 128 over 66,464 selection-training products maps to
  1,700 fixed steps at batch 128 over 112,386 production products. No
  data-dependent early stopping or post-refit checkpoint selection is allowed.
- All three former development folds become training data. Their loss is an
  explicitly in-sample diagnostic and cannot be described as calibration or
  model-selection evidence. The fixed step-1,700 state is exported through the
  legacy `checkpoint_best.pt` filename only for downstream compatibility.
- A local end-to-end smoke test trained on exactly 112,386 records, confirmed
  equal 0.5/0.5 source mass, used no component identifiers, route guidance or
  oracle guidance, and passed the fixed-final-step and partition tests.
- Evidence:
  `configs/model/phase1_ugi_joint_sparse_balanced_v2_production_refit.json`,
  `results/phase1/ugi_joint_sparse_balanced_v2_production_refit_smoke/result.json`
  and `tests/test_ugi_joint_sparse_training_partition.py`.

## 2026-08-05 - Recalibrate biological applicability from held-component errors

- Supersede the first label-bootstrap diagnostic. Molecule-level metrics still
  use every measured lipid, but folds and confidence intervals must cluster by
  the chemistry whose generalization is claimed: held head, aldehyde,
  isocyanide or the corresponding held component pair. Treating repeated
  products sharing one component as independent would be pseudoreplication.
- Freeze candidate normalized radii before the component-group audit and keep
  the scaffold-balanced split external to threshold fitting. Generated-product
  yield, exact novelty and final-generator samples cannot tune the boundary.
- The component-group-nested audit selects normalized radius 2.0 for ranking
  authority and 1.5 for stricter absolute/LCB authority. Nested retained
  coverage is 68.27% and 59.65%, respectively. Ranking has pooled Spearman
  0.682 and observed top-quartile enrichment 2.44 MTP units; absolute authority
  has pooled R2 0.481, Spearman 0.687 and top-quartile enrichment 2.46 MTP
  units. Scheme-specific component-cluster bootstrap lower bounds remain above
  zero for both rank correlation and top-quartile enrichment.
- Exact-new components are not vetoed. A completed lipid may receive oracle
  authority when its continuous product and role chemistry lies inside the
  validated radius and its novelty regime matches structured evidence.
  Molecules outside the authority radius remain valid exploration candidates.
- These thresholds become eligible for production use only after a fresh census
  from the final all-fold generator. Simultaneously exact-new role claims still
  require matching held-role or prospective evidence.
- Evidence:
  `configs/bio/phase1_ugi_applicability_recalibration_v2.json` and
  `results/phase1/ugi_applicability_recalibration_v2/result.json`.

## 2026-08-05 - Retain the step-1,000 argmax result as a nonselecting checkpoint diagnostic

- Supersede the earlier administrative assumption that the exposure-matched
  step-1,700 endpoint must be exported without a molecular checkpoint audit.
  The user explicitly requested that the all-fold refit use the best justified
  generator checkpoint rather than lower in-sample training loss or a fixed
  exposure analogy alone.
- Freeze a fresh 1,024-program draw from the unchanged source- and
  family-balanced morphology prior before checkpoint sampling. Apply the
  unchanged v3 validity, exact-L1, open-endedness, collapse and
  descriptor-realism gates to steps 100, 500, 1,000, 1,500 and 1,700 under
  matched programs, diffusion settings and random seeds.
- Under masked-argmax decoding, step 1,000 is the only checkpoint passing every
  inherited gate:
  98.44% of attempted draws are valid, 99.80% of valid products are unique,
  precursor reconstruction and exact forward assembly are both 100%, 99.31%
  of reconstructed products pass all three handle policies and usable
  open-ended yield is 90.72%.
- Step 100 is rejected for heteroatom-distribution and aldehyde-component
  collapse; step 500 is rejected for heteroatom-distribution mismatch. Steps
  1,500 and 1,700 improve aggregate descriptor agreement but fall below the
  frozen 99% handle-qualification floor, yield 84.38% and 82.81% usable
  open-ended products, and reproduce 6.02% and 6.76% of the all-fold corpus
  exactly versus 2.78% at step 1,000. Decreasing in-sample training loss is
  therefore not checkpoint-selection evidence.
- Treat this as all-fold production quality control, not held-component
  generalization: all 112,386 products entered refit training. Generalization
  claims remain attached to the development-stage component-disjoint audits.
- Do not use this argmax result to freeze the production checkpoint. The already
  qualified production generator uses branch-run feasibility limits and
  bond-stochastic terminal decoding at temperature 1.0; checkpoint selection
  must match that deployed sampling law. Freeze a new independent 1,024-program
  draw and repeat all five checkpoints under the production decoder without
  changing any scientific threshold or selection rule.
- The first selection attempt stopped on an inherited test-file hash that had
  changed only because a later additive branch-spacing regression test entered
  the shared worktree. The selector implementation retained its frozen v3
  hash; the current ten-test suite passed. A disclosed v2 administrative
  amendment updated only that test-file provenance before selection metrics
  were computed; no threshold, algorithm, sample or checkpoint changed.
- Evidence:
  `configs/model/phase1_ugi_production_refit_checkpoint_selection_policy_v1.json`,
  `configs/model/phase1_ugi_production_refit_checkpoint_selection_policy_v2.json`,
  `results/phase1/ugi_production_refit_checkpoint_audit_probe_1024.json` and
  `results/phase1/ugi_production_refit_checkpoint_selection_v1/result.json`.
  The production-decoder audit is frozen by
  `configs/model/phase1_ugi_production_refit_checkpoint_selection_policy_v4.json`.

## 2026-08-05 - Freeze refit step 1,000 under the production stochastic decoder

- Repeat the all-fold checkpoint audit on a new independent 1,024-program draw
  using the deployed sampling law: eight reverse-flow steps, per-role adjacent
  branch ceilings `[2, 1, 1]`, exact-L1 terminal admission and bond-stochastic
  final bond decoding at temperature 1.0. Program, flow and terminal-decoder
  random streams are independently frozen; no retry or seed search is allowed.
- Select refit step 1,000. It is again the only checkpoint passing every
  inherited validity, exact-L1, open-endedness, collapse and descriptor-realism
  gate. Of 1,024 attempted draws, 994 are valid, 993 valid products are unique,
  938 are usable open-ended successes, component inverse recovery and exact
  forward assembly are both 100%, and 99.30% of reconstructed products pass all
  three handle policies. Only 1.91% of valid products exactly reproduce a refit
  corpus product.
- Steps 100 and 500 fail only the frozen heteroatom KS floor. Steps 1,500 and
  1,700 achieve slightly closer bulk descriptor distributions but fail the 99%
  three-handle floor, yield fewer usable open-ended products and reproduce more
  refit-corpus products. Lower in-sample loss and later exposure therefore do
  not justify their promotion.
- The production-decoder result is byte-deterministic across two complete
  selector evaluations. Freeze the versioned v3 production manifest without
  rewriting the prior v1/v2 manifests. A new restartable production seam and a
  fresh, non-selection candidate census remain required before downstream
  candidate locking.
- Refit the morphology prior on the same complete 112,386-product population
  before production sampling. The development prior contained 86, 19 and 35
  role-level states because it correctly excluded calibration and held-out
  folds during model development; the final all-fold population contains 106,
  39 and 40 states. Retaining the development prior would silently exclude 45
  observed role states and reduce the final generator's legitimate morphology
  support. The deterministic all-fold prior therefore expands complete
  role-state support from 57,190 to 165,360 combinations without adding any
  state absent from the qualified production corpus.
- Evidence:
  `configs/model/phase1_ugi_production_refit_checkpoint_selection_policy_v4.json`,
  `results/phase1/ugi_production_refit_decoder_checkpoint_selection_v1/result.json`
  (SHA-256
  `d6aa9225bf842d6e40bac6b883ab57d7c968e09430e24a0a50adeba4ce542195`)
  `results/phase1/ugi_program_prior_v3_all_fold.json` and
  `results/phase1/ugi_product_l1_production_generator_v3.json`.

## 2026-08-05 - Reopen chemistry realization after the ester/ether correlation audit

- Do not reinterpret the earlier step-1,000 selection as evidence that the
  generator reproduces local lipid chemotypes. Its frozen gates covered
  validity, exact L1 assembly, handles, descriptor realism, uniqueness and
  training-set reproduction, but did not measure ester, ether, branching or
  unsaturation fidelity. This omission is now explicit rather than repaired by
  weakening any prior gate.
- The source-balanced all-fold target contains aldehyde ester-like carbonyls in
  64.57% of weighted products and no aldehyde ether oxygen. Under fully
  stochastic terminal decoding, numerical integration from 8 to 64 steps is
  stable but leaves aldehyde ester near 22--30% and ether near 21--28%.
  Integration error is therefore not the explanation.
- On 4,096 held-out clean support graphs, the step-1,500 checkpoint reconstructs
  98.04% of aldehyde decoration anchor--atom--bond triples. The decoration head
  can read the target when the local scaffold is already correct. On 256 frozen
  free trajectories, however, the native categorical endpoint is only 37.5%
  valid and has 21.88% aldehyde ester; feasibility-constrained terminal
  re-decoding raises validity to 98.05% and ester to 30.28% but does not restore
  the target distribution. The endpoint is already chemically uncoordinated;
  terminal re-decoding is not the root cause.
- The current vector field flows decoration anchor, atom and bond coordinates
  but does not receive those noisy coordinates as input, and scaffold-node
  predictions receive no state from attached decorations. This is a missing
  conditional dependency, not evidence for an ester bonus, ether penalty or
  marginal quota.
- Test one matched architectural correction: bidirectional anchor-local
  decoration conditioning. Noisy decoration states are aggregated into their
  current anchor node, and the anchor-node state conditions the slot outputs.
  Train an explicit legacy baseline and challenger with identical train fold,
  seed, optimizer, source/family-balanced sampling and serial checkpoint grid.
  The pilot may authorize multi-seed confirmation but cannot promote a
  production checkpoint by itself.
- The existing production receipt intentionally fails closed after the source
  change. A selected challenger would require chemistry-aware checkpoint
  selection, independent confirmation, all-fold refitting and a new
  restartable-equivalence receipt before production use.
- Evidence:
  `results/phase1/ugi_balanced_terminal_reference_v1.json`,
  `results/phase1/ugi_decoration_readout_audit_v1.json`,
  `results/phase1/ugi_flow_endpoint_joint_chemistry_audit_v1.json`,
  `configs/model/phase1_ugi_decoration_coupling_legacy_v1.json` and
  `configs/model/phase1_ugi_decoration_coupling_challenger_v1.json`.

## 2026-08-05 - Select 3,000 development steps for the decoration-coupled generator

- Compare the legacy vector field with one matched architectural correction:
  bidirectional anchor-local conditioning between noisy decoration states and
  their current scaffold anchors. Keep corpus, sampling weights, seed,
  optimizer, model width, terminal sampler and program schedule matched.
- Screen serial stochastic checkpoints on a frozen 1,024-program schedule using
  exact-L1 validity, program-balanced local tail chemistry and program-matched
  branch geometry. The challenger materially improves local chemistry over the
  legacy model; retain challenger steps 1,500, 3,000 and 4,000 for independent
  confirmation under a rule frozen before confirmation sampling.
- Run three paired replicates per finalist on a completely different frozen
  1,024-program shard: 9,216 attempted products in total. Every replicate passes
  the 95% validity, 98% unique-valid, exact-L1, component-diversity and adjacent
  tail-branch gates. Mean local-chemistry MAE is 0.04155, 0.03576 and 0.04304
  for steps 1,500, 3,000 and 4,000, respectively. Mean branch-geometry JSD is
  0.11069, 0.09143 and 0.08553 bits; mean validity is 0.98079, 0.96029 and
  0.96387.
- Apply the frozen hierarchy without using loss or visual preference. Steps
  3,000 and 4,000 survive chemistry and branch noninferiority. Their mean
  validity differs by less than the frozen 0.01 tie margin, so select the
  earlier 3,000-step duration. This selects training exposure only; no
  development checkpoint becomes the production generator.
- The first cloud launch failed during remote module import because the runner
  resolved the mounted policy under `/configs` rather than
  `/root/forge_repo/configs`. No sampling function ran and no scientific output
  was created. Correct only the remote repository path, record the failed app
  and original policy hash in the frozen policy, then execute the unchanged
  experiment once.
- Evidence:
  `configs/model/phase1_ugi_decoration_checkpoint_confirmation_policy_v1.json`,
  `results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1/evaluation_v1.json`,
  `results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1/program_matched_tail_chemistry_v1.json`
  and
  `results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1/branch_arm_geometry_v1.json`.

## 2026-08-05 - Freeze the all-fold decoration-coupled production refit

- Refit the selected architecture once from random initialization on all
  112,386 exact-forward Ugi products. Preserve the selected expected weighted
  draws per structural record: 3,000 development steps at batch 128 over 66,464
  products map to 5,072.79 steps over 112,386 products and therefore to a fixed
  5,100-step production run after rounding to the nearest 100.
- Use all former train, calibration and heldout folds for training. All reported
  loss values are in-sample monitoring only; there is no early stopping or
  post-refit checkpoint choice. Preserve the source- and component-family-raked
  0.5/0.5 source target, exclude component identifiers from neural tensors and
  exclude route and biological values from training.
- A ten-step CPU smoke loads exactly 112,386 records, reproduces the 0.5/0.5
  source mass, verifies fixed-final-step selection and retains bidirectional
  anchor-local decoration conditioning. The full L4 run requires explicit cloud
  authorization because it uploads private source and the frozen training cache
  to the user's Modal workspace.
- After explicit authorization, complete the deterministic L4 refit under run
  `ugi_decoration_coupling_production_refit_v1_d4a23a76b796`. It reaches the
  frozen 5,100-step endpoint without early stopping or restart. The final
  in-sample monitoring losses are 0.64108 on weighted training draws and 0.44394
  on the all-fold diagnostic pass; neither value selects a checkpoint. The
  exported fixed-endpoint model tensors are exactly equal between
  `checkpoint_best.pt` and `checkpoint_step_5100.pt` despite container-format
  hash differences.
- The production result SHA-256 is
  `0905dcc05fad7f89f97a9a8f5ff66045e82b47b9aedd422e0ca9efcf7da32985`;
  the downstream-compatible checkpoint hash is
  `3a3fbf2ad1e6a080bbba0476748d10e78de7cd5122e153d8fa7fc67b60823497`.
  Runtime provenance records Python 3.11.12, torch 2.11.0+cu130, CUDA 13.0,
  deterministic algorithms, float32 precision and no mixed precision.
- Evidence:
  `configs/model/phase1_ugi_decoration_coupling_production_refit_v1.json`,
  `scripts/modal_phase1_ugi_decoration_coupling_production_refit_v1.py`,
  `tests/test_ugi_decoration_coupling_production_refit_policy.py` and
  `results/phase1/ugi_decoration_coupling_production_refit_smoke_v1/result.json`,
  plus
  `results/phase1/ugi_decoration_coupling_production_refit_v1/result.json`.

## 2026-08-05 - Reject the 5,100-step refit after its fresh production census

- Freeze four new 1,024-program schedules after development selection and the
  all-fold refit are complete. Use four disjoint program, reverse-flow and
  stochastic-terminal random streams, no retries or repairs, and the exact
  Python source tree used for training. The combined non-selection census
  contains 4,096 attempted products.
- The fixed 5,100-step refit passes exact-L1 reconstruction for every valid
  product, valid-product uniqueness (98.08%) and the zero-adjacent-tail-branch
  requirement. It does not pass the preregistered production gates: 3,862 of
  4,096 products are valid (94.29%; required 95%) and 3,746 of 3,862
  reconstructed products pass all three precursor-handle policies (97.00%;
  required 99%). Do not weaken either threshold or silently filter the failed
  products.
- The validity failure is systematic rather than one bad shard. Shard valid
  counts are 976, 959, 959 and 968. All 234 invalid products occur when the
  sampled head component has cycle rank one; head cycle ranks zero and two are
  100% valid in this census. The failures divide into 157 terminal-support and
  77 molecule-sanitization failures.
- The handle deficit is also localized. The isocyanide handle passes in every
  valid product. Nine aldehyde-derived components contain two aldehyde sites;
  107 amine-derived components contain more than the allowed one or two
  symmetry-distinct primary/secondary amine sites. This is a real synthesis
  adapter mismatch, not an L1 reconstruction error.
- Open-endedness and chemistry coverage remain encouraging but cannot override
  the failed gates: 81.43% of valid products are absent from the 112,386-product
  refit corpus, and the valid set contains 1,378, 839 and 267 unique generated
  amine-, aldehyde- and isocyanide-derived components. A program-matched audit
  against all 112,386 production records gives local-tail-chemistry MAE of
  0.0132--0.0175 across shards. These are descriptive findings, not reasons to
  promote the failed endpoint.
- Treat this census as development evidence. The next permitted diagnostic is
  a common-random-number comparison of the already-exported all-fold step-3,000
  snapshot against the rejected step-5,100 endpoint on these same schedules.
  If step 3,000 passes the frozen molecular gates and preserves chemistry and
  branching, it may become a provisional checkpoint only; promotion still
  requires a completely new, untouched final census. If it fails, correct the
  cycle-one closure/terminal-support mechanism and precursor-handle support
  before another independent confirmation.
- Evidence:
  `configs/model/phase1_ugi_decoration_coupling_production_fresh_census_v1.json`,
  `results/phase1/ugi_decoration_coupling_production_fresh_census_v1/result.json`,
  `results/phase1/ugi_decoration_coupling_production_fresh_census_v1/program_matched_tail_chemistry_dev_reference_v1.json`
  and
  `results/phase1/ugi_decoration_coupling_production_fresh_census_v1/branch_arm_geometry_dev_reference_v1.json`.

## 2026-08-05 - Retain the refit only as a constrained stochastic candidate generator

- Preserve the preceding negative result: the all-fold 5,100-step refit does
  not pass the preregistered 95% raw-validity and 99% all-handle promotion
  thresholds and is not reclassified as an unconstrained production generator.
  Neither threshold is weakened.
- A separate operational question is whether the same frozen stochastic model
  is suitable for candidate generation when exact terminal chemistry is an
  explicit part of the sampling contract. Of 4,096 untouched raw draws, 3,746
  are valid, exact-L1 and qualified at every precursor handle, for a 91.46%
  raw-to-admitted yield. The admitted set is 98.02% unique and 80.86% absent
  from the 112,386-product all-fold refit corpus. No rejected molecule is
  repaired or retried.
- Retain this checkpoint for the narrower, accurately named system: a
  **constrained stochastic graph-flow generator for complete Ugi-compatible
  ionizable lipids**. Exact terminal admission conditions the stochastic model
  on sanitization, exact L1 reconstruction and registry-defined precursor
  handles. It is not evidence that every raw draw is valid or synthesis-ready.
- Supersede the existing argmax-era production candidate pool. Freeze a new
  independent matched broad-versus-applicability schedule, generate raw
  terminals with stochastic atom, bond and decoration decoding, seal all raw
  outcomes, and apply the identical exact admission rule before biological
  classification. The already promoted rho-0.50, score-power-1.50 morphology
  proposal, biological applicability policy and post-generation potency
  ranking remain frozen.
- Route assessment remains downstream of molecular generation and upstream of
  panel lock. It must use identical proposal, verification and evidence rules
  for both molecular arms and cannot retroactively alter the raw candidate
  denominator.
- Evidence:
  `results/phase1/ugi_decoration_coupling_production_fresh_census_v1/result.json`,
  `configs/model/phase1_ugi_production_candidate_schedule_v2.json` and the
  forthcoming versioned stochastic production-candidate ledger.

## 2026-08-05 - Restore full-corpus morphology support for branch exploration

- Preserve the promoted 57,190-program applicability proposal and its completed
  broad-versus-support causal candidate run. Do not extrapolate its confirmed
  morphology score onto role states that were absent from that proposal's
  development support.
- The preliminary branching shortlist exposed a scheduler defect rather than a
  demonstrated generator defect. Its 19-state aldehyde support contained only
  one branched state, `(8 exterior atoms, 2 junction units, cycle rank 0,
  attachment count 1)`. Consequently, 1,493 admitted aldehyde-branched
  terminals from the matched production pool were almost uniformly nine-carbon,
  non-ester components and did not reproduce the measured AGILE branched-tail
  architecture.
- Rebuild the exploration support from the already frozen all-fold program
  prior used by the final generator. Its 106 amine, 39 aldehyde and 40
  isocyanide role states define 165,360 complete programs, a 2.89-fold expansion
  over the development support without inventing any role state absent from the
  112,386-product production-refit corpus.
- The restored aldehyde branch support spans 7--24 exterior atoms and includes
  the exact coarse state `(16, 1, 0, 1)` carried by the measured AGILE branched
  aldehyde. Isocyanide branch support spans 4 and 6--24 exterior atoms. Freeze a
  4,096-draw exploration schedule conditioned on at least one tail-origin
  junction and stratified by the broad-prior branch-class mass. It covers every
  restored branch-size state, includes 210 draws of the AGILE-like aldehyde
  state and differs from the ideal conditional proposal by less than 0.04% in
  importance ratio.
- This schedule is an unguided morphology-exploration lane. It does not change
  biological applicability thresholds, does not carry a potency claim and does
  not establish realized carbon branching. Promotion requires a fresh
  stochastic run followed by exact terminal admission and direct audits of
  carbon branching, carbon count, ester retention, branch adjacency, diversity
  and corpus novelty.
- The new Modal launch remains pending explicit authorization because it would
  upload the pinned private source, checkpoint, corpus metadata and schedule to
  a new four-shard cloud job. No cloud output is claimed before that approval.
- Evidence:
  `configs/model/phase1_ugi_full_corpus_branch_exploration_schedule_v1.json`,
  `results/phase1/ugi_full_corpus_branch_exploration_schedule_v1/result.json`,
  `results/phase1/ugi_full_corpus_branch_exploration_schedule_v1/schedule.json`,
  `src/forge/product/ugi_full_corpus_branch_exploration_schedule.py` and
  `tests/test_ugi_full_corpus_branch_exploration_schedule.py`.
## 2026-08-05 - Complete the full-support branch scheduler, generation and applicability lane

- The previously pending Modal launch completed. The restored all-fold morphology
  support (106 amine, 39 aldehyde and 40 isocyanide role states, 165,360 complete
  programs) was sampled with the frozen 4,096-draw branch-conditioned schedule.
- Generation: 4,096 attempts, 3,833 raw valid and exact-L1 products, 3,678
  chemically admitted (89.795%), 3,652 unique, 3,432 absent from the 112,386-product
  refit corpus. Realized carbon branching: 3,462 branched products, 1,243
  aldehyde-only, 1,703 isocyanide-only, 516 both tail origins, 216 realized linear
  after chemistry, 570 long ester-bearing branched aldehydes and zero adjacent
  carbon branch-point chains. Admission failures were 144 amine-handle, 11
  aldehyde-handle and 263 invalid or nonexact-L1 outcomes.
- Applicability rescore on the branch lane: 65 all-view interpolative rows, 20 exact
  measured neutral controls, 45 new oracle-scored rows representing 39 unique
  products, 42 calibrated rows and 17 conservative-high rows representing 15 unique
  products. All 45 new scored rows were long, ester-bearing aldehyde-origin carbon
  branches with one branch point.
- Interpretation. Branch capability was restored by repairing the morphology support
  and conditionally sampling a declared branch lane. The generator was not retrained,
  no branching reward was added and neither the oracle nor terminal chemistry was
  relaxed. The measured AGILE set contains 100 branch products but only one branched
  aldehyde across 20 heads and five isocyanides, so effective independent branch-family
  support remains one aldehyde. This lane supports prospective branch selection; it
  does not establish broad branched-tail oracle generalization.
- Evidence:
  `results/phase1/ugi_full_corpus_branch_exploration_schedule_v1/`,
  `results/phase1/ugi_full_corpus_branch_exploration_candidates_v1/` and
  `results/phase1/ugi_branch_exploration_applicability_v1/`.

## 2026-08-05 - Freeze the corrected v2 route-blinded production shortlist

- The v2 shortlist supersedes the preliminary branching shortlist. It contains 256
  unique products: 96 broad_prior, 96 support_enriched and 64 branch_exploration,
  in three cohorts of 192 potency_exploitation, 32 branched_oracle_scored_candidate
  and 32 branched_synthesis_exploration_no_potency_claim.
- The 192 potency rows are preserved exactly from the v1 matched linear arms. All 15
  unique conservative-high branch products are retained. The 32 oracle-scored branch
  rows span four authority tiers (16 weak-aldehyde exploratory, 10 simultaneous
  exact-new-tail exploratory, four qualified-role holdout, two multiple-exact-new-role).
  Twenty-nine of those 32 are exact members of the 112,386-product structural refit
  corpus but were never biologically measured; all 32 potency-neutral exploration rows
  are corpus-absent and carry no potency claim.
- Selection used no route outcome (`route_outcomes_used: false`). The shortlist is
  route-blinded, not route-complete, and is not a locked synthesis panel.
- Evidence: `results/phase1/ugi_production_route_shortlist_v2/result.json`
  (`cd6dfb9ae9de1ea3164df74422de42b6b14ea0082e8df34a5de1325fc0a9cbfd`) and its
  shortlist ledger
  (`d7b80c3d456ff56665d1c3c3f1e18a9bb9c7533864dfb03a0a92a6f0ba59bb7f`).

## 2026-08-05 - Bounded hybrid route cascade over the v2 shortlist

- Ran one identical, arm-blind, bounded hybrid L1/L2/L3 route cascade across all
  256 route-blinded shortlist products. Single content-addressed planner cache
  keyed only by canonical component target and frozen planner context; the
  inherited guided/post-hoc cache split was not reused because it belonged to the
  closed synthesis-guidance experiment. Generation arm, cohort, authority tier
  and potency never reached route search: 49 components appear under more than
  one arm and each produced exactly one cache key and one route outcome.
- Denominators: 256 candidates in, 256 out; 158 unique components (48 amine, 91
  aldehyde, 19 isocyanide); 765 role assessments with 609 cache hits over 156
  unique computations. Exact constitutional Ugi L1 forward reconstruction passed
  for 256/256.
- One candidate (`branch_exploration:37`) failed declared graph-support
  re-verification on atom state `(Si, 0, False, 2)`; the declared vocabulary
  derived from 15,229 R0 and 464,265 R1 rows contains silicon only as
  `(Si, 0, False, 0)`. The gate was not relaxed. The candidate is retained in the
  denominator with its exact disposition. Terminal chemical admission at
  generation and declared graph-support re-verification at route admission are
  therefore distinct contracts, and the branch lane produced a product that
  separates them.
- Outcome under exact indexed evidence: 19/256 products closed (7 broad, 5
  support, 7 branch); 36/158 components closed. Balanced fillable per causal arm
  is 5, below the declared 18 minimum and 28 preferred feasibility thresholds.
- Proposal engines closed zero additional routes. Graph2Edits produced 909
  hypotheses over 120 unresolved components with 48 graph-consistent under the
  independent forward resolver, and covered zero of 30 amine heads, reproducing
  the frozen single-step benchmark's amine-formylation gap on a fresh component
  set. AiZynthFinder produced 713 residual hypotheses and solved 25 components to
  public stock. Neither can close a route under the proposal-only contract, and
  neither did.
- **Interpretation, load-bearing.** 119 of 120 unresolved components were never
  expanded: the frozen exact-evidence source is an index, not a retrosynthetic
  search, and a target absent from it returns `missing_knowledge` at depth zero
  having attempted no disconnection. The closure rate therefore measures coverage
  of a hand-curated index reaching a 60-item terminal-material ledger. It is not
  a synthesizability estimate. The rate is directly comparable to the frozen
  1,855-row census at approximately 6.5-6.9% because the metric definition is
  unchanged.
- Evidence: `configs/route/phase1_ugi_bounded_hybrid_route_cascade_v1.json`,
  `results/phase1/ugi_bounded_hybrid_route_cascade_v1/`,
  `src/forge/product/ugi_bounded_hybrid_route_cascade.py` and
  `tests/test_ugi_bounded_hybrid_route_cascade_v1.py`.

## 2026-08-05 - Replace the hand-curated availability ledger with an online procurement snapshot

- Two defects in the cascade were identified and corrected. AiZynthFinder had run
  only on the 72 components Graph2Edits left without a graph-consistent
  hypothesis, so 48 unresolved components were never asked whether a route to
  catalog stock exists; and the inherited 8-second/60-iteration engine-benchmark
  budget was too small for 40-80 heavy-atom lipid precursors. A planner reach
  sweep re-ran all 120 unresolved components at 60 seconds and 300 iterations,
  identical for every component: 60/120 solved to public catalog with zero
  execution failures, of which 24 were among the 48 never previously tested.
- Two candidate verification signals were tested and rejected. Cross-engine
  agreement fired on 118/120 components (98.3%) and has no discriminative power,
  because both engines are USPTO-trained and default to the same generic
  functional-group interconversions. Forward reconstruction using the engine's
  own retro template passed 1,183/1,186 proposals (99.75%) and is circular for a
  template-based engine, since the proposal is generated by applying that
  template. Neither is used as evidence.
- A single-step reading of disconnection class initially suggested the engines
  never build carbon skeletons. That was an artifact of examining single-step
  proposals only. Inside the solved multistep routes, 65 of 102 steps are
  skeleton-changing and include the AGILE Tail A esterification logic.
- **Scope correction.** The local reaction and availability knowledge is one
  qualified L1 transform, four upstream reactions, eleven forward-resolver
  transforms and a 60-item terminal-material ledger, against roughly 48,000
  templates in the engines. Adjudicating novel components against that local set
  measures bookkeeping, not chemistry. Availability is now determined by an
  online PubChem vendor snapshot with a recorded access time and a declared
  30-day expiry. A vendor listing is a screening signal and is not a quote, stock
  level, lead time or purity specification; zero listed vendors is evidence a
  compound must be synthesised, not proof it cannot be obtained.
- Evidence: `configs/route/phase1_ugi_planner_reach_sweep_v1.json`,
  `results/phase1/ugi_planner_reach_sweep_v1/`,
  `results/phase1/ugi_engine_template_verification_v1/`,
  `results/phase1/ugi_planner_proposal_verification_v1/`,
  `scripts/phase1_run_ugi_online_procurement_lookup_v1.py` and
  `results/phase1/ugi_online_procurement_snapshot_v1/`.

## 2026-08-05 - Route-aware decision package over the v2 shortlist

- Assembled one nonselecting dossier for all 256 candidates carrying molecule and
  exact component graphs, generation and morphology receipts, chemical
  descriptors, authority tier with oracle mean/sd/LCB90/conformal-q90, exact L1
  verification, the role-level route tree with proposal provenance, unresolved
  risks, a declared chemotype cluster and a recommended reporting stratum.
- Panel fill against the declared 18-minimum and 28-preferred feasibility
  thresholds, by evidence rung, balanced per causal arm: exact indexed evidence
  5; exact or family-projected 35; exact, family or bounded-planner reach 43.
  The rungs are ordered and are not interchangeable.
- Practical makeability, combining routes with the online procurement snapshot:
  38 candidates are `buy_and_make` (every component purchasable, or reachable by
  a route whose starting materials are all purchasable), 70 are `route_only`
  (routes exist, some starting materials unconfirmed) and 148 are `blocked`.
  Balanced per causal arm is 13 `buy_and_make`, or 33 including `route_only`.
- Component availability by role: 34/48 amine heads, 7/19 isocyanide tails and
  2/91 oxoester aldehydes have listed vendors. Heads are commodity chemicals;
  the tail components are bespoke and must be synthesised, which is precisely the
  division of labour in the AGILE procedure.
- Reporting strata: 11 exact-route candidates with the strongest biological
  authority, three exact-route exploratory, five exact-route with no potency
  claim, 68 family-projected conservative-high, three family-projected with no
  potency claim, 44 planner-reachable triage-only, 121 unresolved and one
  excluded outside declared graph support. 115 declared chemotype clusters, 18
  among exact-route candidates.
- No candidate was selected and no panel was locked. `PREREGISTRATION.md` was not
  created. The procurement snapshot expires 2026-09-05 and must be refreshed
  before any lock.
- Evidence: `configs/route/phase1_ugi_route_aware_decision_package_v1.json` and
  `results/phase1/ugi_route_aware_decision_package_v1/`.

## 2026-08-05 - Open scope items recorded rather than silently deferred

- The v2 shortlist was selected before route or availability information existed,
  which was correct for an unbiased platform measurement but is not the right
  pool for panel selection. The full applicability-supported population is 2,590
  oracle-scored products resolving to 201 unique components, of which roughly 100
  have been routed. Extending coverage to the remaining components would let a
  panel be selected from every candidate the oracle supports, filtered by
  makeability, without disturbing the frozen 256-product measurement. Not run.
- 58 of the 158 shortlist components serve only the 32 potency-neutral
  exploration candidates and therefore contribute nothing to panel selection.
- `make test` reports 67 failures and 14 errors that predate this work. They stem
  from hash-pinned source files modified on 2 August, notably
  `src/forge/value/synthesis.py`, whose digest no longer matches the C16/C18
  route configs frozen the same day. No pin was rewritten to make them pass.
- Manuscript, evidence-matrix and evidence-ledger reconciliation has not started
  and remains required for computational closeout.

## 2026-08-06 - Route the whole applicability-supported population, not the shortlist

- The v2 shortlist was selected on potency, diversity and branch quotas before any
  route or availability information existed. That is correct for an unbiased
  platform measurement and wrong as a pool from which to choose a synthesis
  panel: 93% of it proved unroutable under the frozen index, while candidates
  that missed the potency cut were never routed at all.
- The full oracle-scored population is 2,590 unique products resolving to only
  201 unique components, so covering it costs about a hundred component lookups
  rather than thousands of molecules. Component-level work with roll-up to
  products replaces per-product routing.
- The frozen 256-product cascade is unchanged and remains the route-blinded
  measurement and the matched-arm cohort.
- Evidence: `results/phase1/ugi_indomain_makeability_v1/`.

## 2026-08-06 - Replace the hand-curated availability ledger with an online snapshot

- Local knowledge is one qualified L1 transform, four upstream reactions, eleven
  forward-resolver transforms and a 60-item terminal-material ledger, against
  roughly 48,000 templates inside the proposal engines. Adjudicating novel
  components against that local set measures bookkeeping, not chemistry.
  Availability is now read from PubChem vendor registrations with a recorded
  access time and a declared 30-day expiry.
- A vendor listing is a screening signal. It is not a quote, stock level, lead
  time or purity specification, and zero listed vendors is evidence a compound
  must be synthesised rather than proof it cannot be obtained.
- Snapshot accessed 2026-08-06, expires 2026-09-05. It must be refreshed before
  any purchase or panel lock.
- Evidence: `scripts/phase1_run_ugi_online_procurement_lookup_v1.py`,
  `results/phase1/ugi_online_procurement_snapshot_v1/`.

## 2026-08-06 - Apply published routes deliberately; generic planners miss lipid chemistry

- A bounded planner left most components "blocked", which reflected its search
  budget and its 2020 public-catalog stock definition rather than chemistry. Two
  published routes were therefore applied deliberately and verified by running
  the repository's own qualified transforms forward against the exact target.
- Tail A (`ugi3_upstream_esterification_exact_source_v1` then
  `ugi3_upstream_primary_alcohol_oxidation_exact_source_v1`): 161 of 163
  aldehydes forward-verified, 142 with both starting materials purchasable, and
  every alpha,omega-diol available.
- Isocyanide route (`ugi3_upstream_amine_formylation_exact_source_v1` then
  `ugi3_upstream_formamide_dehydration_exact_source_v1`): 11 of 11 forward-
  verified, 7 with a purchasable precursor amine.
- Makeability across the 2,590 in-domain products became 2,517 complete paths to
  vendor-listed material (97.2%), 73 route-only (2.8%) and zero blocked. No
  threshold was relaxed at any point; every increase came from closing a gap in
  our own coverage.
- **Finding worth reporting.** Generic retrosynthesis models underperform on
  lipid-scale aliphatic chemistry: AiZynthFinder terminates at a frozen public
  catalog and its USPTO template priors do not rank a clean acid-plus-diol
  esterification highly for C16-C20 chains. Domain-qualified transforms close the
  gap. Forward reproduction of a target does not verify substrate scope.
- Two candidate verification signals were tested and rejected. Cross-engine
  agreement fired on 118 of 120 components (98.3%); forward reconstruction using
  the engine's own retro template passed 1,183 of 1,186 proposals (99.75%) and is
  circular for a template-based engine. Neither is used as evidence.
- Evidence: `results/phase1/ugi_tail_a_disconnection_v1/`,
  `results/phase1/ugi_isocyanide_route_v1/`,
  `results/phase1/ugi_planner_reach_sweep_v1/`,
  `results/phase1/ugi_engine_template_verification_v1/`,
  `results/phase1/ugi_planner_proposal_verification_v1/`.

## 2026-08-06 - Makeability filtering audited for bias at every level

- Before the published routes were applied, filtering on makeability distorted
  the population badly: `qualified_role_holdout`, the only lane with genuine
  exact-new-role support, fell from 12.5% to 0.1%; isocyanide diversity fell to
  27% retained; unsaturated isocyanides to zero; and conservative-high potency
  from 41.6% to 36.4%. The distortion was caused by our own missing coverage, not
  by chemistry.
- After both published routes were applied the filter is essentially
  non-distorting: broad and support arm retention 98.3% and 96.4%,
  conservative-high 41.6% to 41.6%, `qualified_role_holdout` 12.5% to 12.4%
  (323 to 311), and every other authority tier within 0.6 points.
- One bias is real and remains: 19 aldehydes require fatty acids that are
  positional isomers of natural chains at non-natural unsaturation positions, and
  those are genuinely not sold.
- Evidence: `results/phase1/ugi_indomain_makeability_v1/result.json`.

## 2026-08-06 - Branching is makeable but barely scorable

- The main 32,768-draw production pool contains no carbon-branched component at
  all, because it predates the branch-scheduler correction. Branched candidates
  exist only in the separate 4,096-draw branch-conditioned lane, which was never
  merged into the rescoring ledger.
- That lane produced 3,678 admitted products and 3,462 realized carbon-branched
  products, but only 65 all-view interpolative rows and 39 unique oracle-scored
  products: an in-domain yield near 1.9%. The binding constraint is applicability,
  not generation and not synthesis. AGILE measured 100 branched products but all
  share one branched aldehyde, so effective branch support is a single aldehyde
  and nearly every generated branched lipid is extrapolative.
- Synthesis is not the limit: all four branched aldehydes forward-verify by Tail A
  from 4-methylnonanoic acid (66 vendors) or its homologue plus commodity diols.
- The highest-scoring in-domain candidate in the entire study is branched
  (LCB90 8.90, mean 14.20), ranking first of 2,478, but it sits in the weakest
  authority tier. The five `qualified_role_holdout` branched candidates score
  LCB90 0.53 to 1.63 and none is conservative-high. Confidence and score are
  anticorrelated within the branched set.
- Widening applicability to admit more branched candidates would be exactly the
  threshold-weakening the contract forbids. More branched in-domain candidates
  require additional identically configured branch tranches, yielding roughly 39
  scored products per 4,096 draws.
- Evidence: `results/phase1/ugi_branch_exploration_applicability_v1/result.json`.

## 2026-08-06 - Stratified 20-candidate proposal for chemist review

- Proposed, not locked. No preregistration was created and no candidate was
  selected irreversibly. The user intends to synthesise approximately eight
  compounds chosen by a chemist from this list.
- Twenty candidates across four strata: seven `qualified_role_holdout`, three
  branched, seven chemotype-diversity and three deliberately low-ranked controls.
  Twenty distinct declared chemotypes, eleven amines, twelve aldehydes, seven
  isocyanides, LCB90 spanning -6.45 to 8.90. Every candidate is makeable.
- Controls are included deliberately: without low-ranked compounds a positive
  prospective result cannot be attributed to ranking rather than to the whole
  population being active.
- At roughly eight evaluable compounds the broad-versus-support causal comparison
  is not powered and is deferred by explicit user decision. This becomes a
  feasibility and hit-rate study rather than the causal experiment.
- The three highest-scoring branched candidates all sit in the weakest authority
  tier. A prospective result driven by them would be evidence of successful
  extrapolation, which the current evidence does not license in advance.
- Evidence: `results/phase1/ugi_stratified_panel_proposal_v1/`.

## 2026-08-06 - Superseded generator denominators corrected in manuscript prose

- The Abstract, generator Results and Methods still reported the 3,975-of-4,096
  development pool and described the final model as stopping after 4,750 updates
  under calibration-loss selection. Both were obsolete and are corrected.
- Generation denominators now come from the two-arm 32,768-draw production run:
  30,180 admitted (92.1%), broad-prior 15,055/16,384 and support-enriched
  15,125/16,384, with 22,843 admitted products (75.7%) absent from the all-fold
  refit corpus. The branch-conditioned 4,096-draw tranche is not pooled in.
- Morphology percentages in the Results described an older, scheduler-limited
  run and were recomputed over the admitted production pool: unsaturated 33.4%,
  ester-containing 70.8%, any branched tail 32.3%, aldehyde-origin branching
  4.9%. The prior 16.6/30.5/7.2 figures are not from this pool.
- The `component_novelty_class` three-way split (317 catalog-only, 506 curated,
  3,152 catalog-absent) does not survive the new run, in which all 30,180
  admitted products are `generated_complete_components`. The paragraph was
  rewritten rather than renumbered.
- Methods now separate development training (4,750 updates, calibration-loss
  rule, step 3,000 selected under matched duration) from the final refit (5,100
  fixed updates on all 112,386 records, no early stopping, no post-refit
  checkpoint selection).
- Route resolution is reported on its own stated denominator: 2,515 of 2,590
  applicability-supported products (97.1%) from 201 components. It is not
  presented as a fraction of the 30,180 admitted pool.
- The superseded 3,975 evidence-ledger row was marked HISTORICAL / DIAGNOSTIC
  ONLY rather than deleted, and two rows were added for the production run and
  route resolution.
- `COMPUTATIONAL_RESULTS_EVIDENCE_MATRIX.md` references to 4,750 steps were left
  unchanged: they describe the development training run, where the figure is
  correct.
- No artifact, config or hash pin was modified. This is a prose correction only.

## 2026-08-06 - Test-suite provenance cleanup: tracked pins, environment split, pin quarantine

- Root cause of the standing red suite: 280 of 305 source files pinned by a
  config were never committed. A SHA-256 pin asserts "this artifact came from
  exactly this code"; when the file is untracked the assertion is unfalsifiable
  and the pinned content is destroyed by the next edit. This is why the
  `synthesis.py` and `ugi_chemistry_flow.py` drift was irreversible rather than
  merely inconvenient. All 280 in-repo files are now tracked (commit b315519),
  with a guard test preventing recurrence.
- Two pins resolve outside this repository
  (`../electron_flow_lipids/modal_apps/flower_{diagnostic,probe}.py`) and cannot
  be tracked here. They are recorded as explicit known exceptions, not skipped
  silently: artifacts depending on them are reproducible only on a machine
  holding that sibling checkout at the pinned revision.
- Correction to an earlier reading: `src/forge/value/synthesis.py` had not
  drifted from its pin. It carries two pinned hashes across configs -- 8 pin the
  current content, 12 pin a superseded one -- and the file matches its current
  pin. The failures are older configs referencing a hash the file legitimately
  moved past, which is a versioning problem, not a corrupted source.
- Byte-exact manifest comparisons conflated the scientific artifact with the
  build environment. The M0-07 oracle-split test failed solely because numpy
  moved 2.4.2 -> 2.5.1 and rdkit 2025.09.6 -> 2026.03.4; regeneration confirmed
  the split assignments are byte-identical. The comparison now holds the data
  strictly and reports environment drift as a visible, named skip.
- 82 tests fail on pinned-input hash mismatches whose pinned revisions are
  unrecoverable. They are quarantined through an explicit registry of exact node
  ids (`tests/unreproducible_pins.py`): 68 as non-strict xfail, and 14 as skips
  because their pin check runs inside a module-scoped fixture, where pytest
  cannot express a setup error as xfail. Matching is by node id only, so a real
  regression in a neighbouring test in the same module still fails. Non-strict
  xfail means a repaired test reports XPASS rather than passing silently, so the
  list cannot rot; a guard test also fails if a quarantined test is renamed or
  deleted.
- Two distinct drifted pins account for all of them: `src/forge/value/synthesis.py`
  (superseded hash referenced by 12 older configs) and the restartable-equivalence
  receipt's generator implementation hash, `8a8dfa39...` against a current tree
  at `dab5e59a...`.
- Final state: 1,314 passed, 68 xfailed, 16 skipped, 0 failed, 0 errors, 0 xpassed.
- Rejected: bulk re-pinning the stale hashes to current values. It would turn
  the suite green immediately and would assert that stored results came from
  code that did not produce them.
- The motivation is signal, not tidiness: with 69 known-benign failures the
  seventieth is invisible. No config, artifact, hash pin or result value was
  modified, and no test was deleted or weakened.

## 2026-08-06 - Isocyanide route pass rescoped to all in-domain isocyanides

- The pass covered only isocyanides whose makeability verdict was `blocked` or
  `route_only`. Isocyanides already judged `make_from_purchasable` were never
  written to the ledger, so consumers reading that artifact saw a resolved
  component as unresolved purely because no row existed for it. Two dossier
  components -- both tridecyl isocyanide, behind 335 in-domain products -- were
  reported UNRESOLVED for this reason alone. This was a bookkeeping gap, not a
  chemistry limit, and is the same residual-only scoping defect already fixed in
  the Tail A pass.
- Now covers every in-domain isocyanide plus the separately scored
  branch-conditioned lane: 15 isocyanides behind 2,635 products, all 15
  forward-verified through formylation then dehydration, 11 with a purchasable
  precursor amine, 2,594 products unlocked where the amine is listed.
- Dossiers rebuilt: 20 candidates, 60 components, 21 purchased, 39 synthesised,
  0 unresolved, 29 distinct materials to buy, at most 4 synthetic steps.
- Data suite rebuilt: 25 of 25 claims resolved.
- No gate was relaxed and no threshold moved. The four unpurchasable precursor
  amines remain unpurchasable and are reported as such.

## 2026-08-06 - Manuscript reframed from delivery perspective and rewritten to venue register

- The draft was reframed for an LNP and mRNA delivery audience after a prose and
  framing audit against Su et al. (Nat Biomed Eng 2026), AGILE (Nat Commun 2024),
  PeptiVerse and PepMLM. The analysis is recorded in
  `docs/MANUSCRIPT_PROSE_ANALYSIS.md` and the section plan in
  `docs/MANUSCRIPT_REFRAME_PLAN.md`.
- Framing: the claim is a de novo generative model for ionizable lipids carried to
  prospective in vivo validation, with synthesis-awareness as the enabler. The
  generative-versus-enumerative distinction is now drawn explicitly, because prior
  work described as "AI-driven generation" ranks enumerated candidates and a
  reviewer will otherwise assume the ground is taken.
- Written: Abstract (145 words, within the 150 limit, unreferenced), a four-
  paragraph Introduction where none previously existed, and Results 1 and 3-6.
  The Discussion was rewritten. Main text is 2,713 words against a 3,000 limit,
  leaving room for the two prospective sections.
- Cut from main text and retained in this log: the SMC and MH sampling work, the
  synthesis-tilting diagnostic, the potency-guided trajectory challenger (23 vs 27
  supported conservative-high products), the graph-edit versus planner blinded
  comparison, and the scheduler and checkpoint history. A negative result now
  earns main-text space only where a reviewer would otherwise assume the wrong
  choice was made.
- The Results section on corpus construction was removed as duplicative of
  Methods; only the route-evidence imbalance was retained in Results, because it
  motivates constructing upstream chemistry rather than learning it.
- Facts corrected against artifacts during the rewrite: the in-domain population
  comprises 23 amines, 163 aldehydes and 15 isocyanides, against the AGILE
  library's 20, 12 and 5. Of the 21 unresolved aldehydes, 19 require
  non-naturally-unsaturated fatty acids, not all 21; the sentence was corrected
  before it entered the draft.
- New references added: Hajj et al. (Small 2019, 306Oi10 branched tail) and
  Sabnis et al. (Mol Ther 2018, Moderna lipid 5).
- No artifact, config or hash pin was modified. Every number in the rewritten
  sections traces to `results/phase1/forge_data_suite_v1/`.

## 2026-08-06 - Manuscript rewritten against MANUSCRIPT_AUTHORITY.md; panel reduced to ten

- `docs/MANUSCRIPT_AUTHORITY.md` is now the authoritative source for framing, claim
  hierarchy, section structure, figure architecture and panel design. It supersedes
  `MANUSCRIPT_REFRAME_PLAN.md`. `MANUSCRIPT_PROSE_ANALYSIS.md` remains in force for
  sentence-level style.
- Priority claim fixed as: the first prospective in vivo validation of ionizable
  lipids designed de novo by a molecular generative model. Not "the first
  AI-designed LNP", which is false given the AI-guided screening literature, and not
  "the first generative ionizable-lipid model", which is false given Ou et al.
- Evidence boundary corrected. The abstract no longer asserts that routed designs
  "can be made"; before prospective synthesis it says only that designs are
  connected to computationally complete routes terminating in purchasable starting
  materials.
- "Open-ended" removed throughout. Generation is now described as de novo within a
  declared molecular support, since atom, bond, size, branching and cycle support
  are bounded.
- The null generation-time route-guidance diagnostic was demoted from a main figure
  to Extended Data Fig. 9 and one main-text sentence.
- Panel rebuilt from 20 to 10: six primary, two de novo frontier selected on
  component novelty against the expanded registry, two matched low-ranked controls.
  Ten synthesis slots cannot power a causal comparison between generation arms, so
  that comparison is left to fresh computational samples.
- Deviation from a literal reading of the authority, made deliberately and recorded
  here: ranking the primary stratum purely by activity lower bound filled it
  entirely with exploratory-tier designs, because chemotype de-duplication let a
  marginally higher-scoring exploratory candidate displace a qualified one sharing
  its chemotype. The primary stratum is therefore drawn preferentially from the
  qualified role-holdout tier. The cost is small, with the best qualified lower
  bound at 6.59 against 6.96 overall, and it preserves the stratum's purpose: a
  prospective hit there tests calibrated ranking rather than extrapolation, which is
  what the frontier stratum is for. Reversible by removing the preference in
  `scripts/phase1_build_ugi_stratified_panel_proposal_v1.py`.
- Panel now spans ten chemotypes from 7 amines, 7 aldehydes and 4 isocyanides, with
  6 qualified role-holdout and 4 exploratory-tier candidates. Dossiers: 30
  components, 10 purchased, 20 prepared, 0 unresolved, 19 catalogue materials, at
  most four steps. Data suite remains 25 of 25 resolved.
- Figure assets for the previous Figures 1 and 2 were archived to
  `manuscript/figures/_archive_2026-08-06/` with a README recording why each was
  superseded. Nothing was deleted and no new artwork was produced.
- Main text is 2,310 words written against a 3,000 limit, with nine result
  placeholders and five figure legends. Abstract is 149 words, unreferenced.

## 2026-08-06 - Prediction-cohort panel locked at 30; Q/N/X protocol v2

- Panel expanded from 10 to 30 under `docs/MANUSCRIPT_AUTHORITY.md` sections 19-22.
  Prediction-evidence cohorts renamed Q/N/X because the earlier Tier A/B/C labels
  collided with the route-evidence labels R1/R2/R3.
- Final allocation, preferred quota achieved without fallback: 12 Q-high, 7 Q-low,
  8 N, 3 X. Benchmarks and vehicle sit outside the 30.
- Cohorts differ along one axis only: whether the aldehyde identity was measured
  (Q), near the measured set (N), or beyond it (X). Amine and isocyanide identities
  are measured in every cohort, so the panel reads as one progression rather than
  four generalization problems.
- The similarity envelope, 0.65 for aldehydes, is derived rather than chosen: it is
  the minimum internal nearest-neighbour Tanimoto among the 11 measured aldehydes.
  A generated aldehyde at or above it is at least as close to the measured set as
  the most isolated measured aldehyde, for which the oracle already issues
  qualified predictions.

### Three defects in protocol v1, corrected in v2

- **Chemistry filter under-specified and applied late.** It was applied only after
  X selection surfaced enols, allenes and hemiacetals. Applied globally in v2 it
  removed 223 enones from the qualified pool of 2,407, roughly 9%, all of which
  would otherwise have been eligible for synthesis.
- **Diversity ceiling inherited from the ten-slot panel.** At 0.70 whole-molecule
  Tanimoto, with only 23 amines and 15 isocyanides in the population, the twelfth
  Q-high candidate fell to LCB90 -3.04, overlapping the control band. Raised to
  0.80 within Q-high, the cohort spans 6.59 to 2.13.
- **Diversity applied globally rather than per cohort.** This forced Q-low controls
  to be structurally distant from Q-high and collapsed the control pool to one
  candidate. Corrected: there is no cross-cohort similarity exclusion, because a
  control that differs only in predicted rank is what makes the comparison
  interpretable. Q-low is now selected by nearest-neighbour matching to Q-high on
  non-oracle covariates.
- Resulting balance, standardized differences: molecular weight +0.07, cLogP -0.20,
  TPSA +0.48, heavy atoms +0.06, rotatable bonds +0.05, route steps 0.00. Score
  bands do not overlap: Q-high floor 2.13 against Q-low ceiling -0.85.

### Statistical correction

My earlier Fisher values and power claim were wrong and are corrected in authority
section 22. Recomputed: 4/6 versus 0/2 gives one-sided p = 0.214, not 0.107; 8/12
versus 0/6 gives p = 0.011, not 0.008. Ex ante power for 12 versus 6 at true rates
67% against 10% is 0.65, not "properly powered". The primary endpoint is the
continuous biological measurement; thresholded hit rate is secondary.

### X cohort reported as it is

X is 3 products from 2 distinct unseen aldehydes: one near-envelope aldehyde in two
component contexts, one more remote aldehyde in one. The fourth slot could not be
filled without weakening the chemistry or route criteria, so it was not filled and
went to Q-low instead. The reason X is thin is itself a result: beyond the envelope
our generated aldehydes are either chemically malformed or leave the oxo-ester
chemotype that the Tail A route covers. The envelope tracks where our synthesis
knowledge ends, not only where the activity model's evidence ends.

### AGILE H9

Recovered from the AGILE Supplementary and verified against the reported ESI-MS
(C36H69N3O3, M+H 592.53). Our oracle abstains on it: domain `unseen_aldehyde`, raw
mean 3.18, no qualified score. Its aldehyde sits at Tanimoto 0.90 from a measured
one and its isocyanide is exactly measured, which is what exposed the identity-based
abstention rule as too blunt for homologs and motivated the N cohort. Structures
recorded in `data/reference/agile_benchmark_lipids.json`.

Independently, the top-ranked N candidate shares H9's aldehyde and isocyanide
exactly, differing only in the head group. FORGE was given no information about H9.

### Artifacts

- `configs/bio/phase1_ugi_prediction_cohorts_v1.json` preserved unchanged.
- `configs/bio/phase1_ugi_prediction_cohorts_v2.json` amendment.
- `results/phase1/ugi_prediction_cohort_panel_v2/` panel ledger, within-cohort
  similarity matrices, panel-specific Tail A, isocyanide and procurement ledgers,
  and `FORGE_PREDICTION_COHORT_PANEL.md`.
- All 30 candidates route-complete, at most 4 synthetic steps, 42 distinct
  catalogue materials. Data suite remains 25 of 25 resolved.
- Oracle inference requires the pinned refit stack (rdkit 2025.09.6,
  scikit-learn 1.8.0, torch 2.11.0); the checkpoint refuses to load otherwise and
  that guard was not bypassed.

## 2026-08-06 - Chemistry filter corrected; prediction-cohort panel v3

Correction to the entry above: the 223 designs removed by the v2 chemistry filter
were **not** malformed. The audit categorised every exclusion and found 223 of 223
were chemically valid alpha,beta-unsaturated esters excluded by an overbroad
reactive-group alert, and 0 malformed structures. The earlier entry listed them
alongside enols and allenes, which conflated two different things.

- The v2 SMARTS `[CX3]=[CX3][CX3]=[OX1]` was intended to flag alpha,beta-unsaturated
  ketones and also matched esters. The motif is present in the measured source
  library: 100 of 1,100 measured AGILE products carry it, with median HeLa 6.27
  against a full-set median of 5.25.
- Corrected to `[CX3]=[CX3][CX3](=[OX1])[#6]`, requiring a second carbon neighbour on
  the carbonyl. Unit-tested, 12 of 12 cases pass, including explicit negative tests on
  the measured 2-decenoate ester aldehyde and its Ugi product.
- This is not a claim that alpha,beta-unsaturated esters are unreactive. The narrower
  conclusion is that a blanket exclusion was unsupported here and contradicted the
  empirically measured design domain.
- Q eligible pool restored 2,184 to 2,407. Q selection rerun deterministically:
  9 of 12 Q-high and 5 of 7 Q-low identities retained. Score separation improved from
  2.13/-0.85 to 3.19/-0.74. Covariate balance improved on every axis; TPSA fell from
  +0.48 to +0.03 as a by-product of the correction rather than by reopening matching.
- Restored designs entering the panel: 3 Q-high, 1 Q-low. They were not inserted; the
  frozen ranking, diversity and matching rules selected them.
- N and X re-audited under the corrected filter and are unchanged.

Also correcting an overbroad statement in the entry above: "the similarity boundary
marks where our synthesis knowledge ends" claims more than the evidence supports. The
supported statement is that within the current enumerated Ugi building-block universe,
route rules and product-quality filters, structurally remote aldehydes were sparsely
represented and frequently coupled to unsupported or undesirable chemotypes.

Artifacts: `configs/bio/phase1_ugi_prediction_cohorts_v3.json`,
`results/phase1/ugi_prediction_cohort_panel_v3/` containing the panel ledger,
`CHEMISTRY_FILTER_V3_AUDIT.md`, `PREPROCUREMENT_SIGNOFF.md`, candidate manifest and
hash, locked blinding key, bill-of-materials stub and similarity matrices. v1 and v2
preserved unchanged.

## 2026-08-06 - Comparator package descoped; pre-procurement closeout

- The proposed six-model, three-regime comparator package was descoped before any
  comparator was trained. It was disproportionate to a prospective discovery paper:
  AGILE compared against four simple models on one scaffold split and proceeded to
  synthesis, and the Nature Biotechnology LiON precedent leads with prospective
  biology rather than model comparison. Recorded in
  `configs/bio/phase1_ugi_comparator_protocol_v1.json` under `descope`.
- Retained: a regularized component-additive ridge, a Morgan ExtraTrees model, a
  nearest-neighbour predictor and a null baseline, evaluated on component-aware
  5-fold cross-validation of the measured 1,100 with held-out aldehyde families.
- Result: component-additive ridge MAE 2.137, Pearson +0.530, Spearman +0.469, ahead
  of Morgan ExtraTrees at 2.455 / +0.472 / +0.451 and nearest neighbour at 2.510 /
  +0.413 / +0.378. Much of the signal in this dataset is factorial component identity.
- On the locked panel, all three baselines separate Q-high from Q-low with mean
  differences +2.58, +3.46 and +3.04. A positive prospective Q-high versus Q-low
  result therefore validates the ranking concept but does not on its own show the
  neural oracle was required. This is stated in the analysis plan rather than left
  for a reviewer.
- The frozen oracle's reported rho of 0.612 came from a different split and is not
  comparable to these figures. No superiority claim is made.
- Comparator predictions for all 30 candidates were frozen after the panel was locked
  and did not influence any candidate identity.
- Written: `docs/PHASE1_UGI_PROSPECTIVE_ANALYSIS_PLAN_V1.md`, and in the panel
  directory `CHEMISTRY_REVIEW_PACKET.md`, `bill_of_materials.csv`,
  `PREPROCUREMENT_COMPUTATIONAL_CLOSEOUT_V3.md`.
- Panel remains NOT cleared for procurement: chemistry signoff and vendor completion
  of the bill of materials are external blockers. Manuscript rewrite from 10 to 30
  candidates is deferred to before submission, not before ordering.

## 2026-08-06 - Computational development frozen; panel final

- The v3 30-candidate panel is final. Computational development for the prospective
  panel is frozen. No further model families, LOCO work, uncertainty calibration or
  benchmarking before synthesis. A candidate identity changes only for a real
  chemistry or procurement impossibility found during human review, never for a score.
- Strategic reset recorded as authority sections 23, 23a, 23b and 23c. The generative
  framework is the contribution; the potency predictor is a conservative allocation
  mechanism for a finite experimental budget. The comparator result, in which a
  component-additive ridge is the strongest simple baseline and all baselines separate
  Q-high from Q-low, supports that framing rather than undermining it.
- Novelty statement sharpened away from "first generative lipid model", which
  LUMI-lab's autonomous loop would contest, to: existing AI-guided lipid discovery
  predicts, ranks or selects from chemically prescribed design spaces, whereas FORGE
  treats molecular proposal and synthetic realization as one problem.
- In vivo package extended one step to editing in the target organ, which lifts the
  claim from delivery demonstrated to delivery producing a functional genomic outcome
  in the intended tissue. Disease-modification, selectivity and durability claims are
  not licensed by that endpoint alone.
- Section 23 proposed a replacement figure architecture; it was withdrawn in 23b and
  section 10 stands, because the two were the same five-figure shape and section 10's
  panel plans are more developed. Recorded as an example of restating rather than
  correcting. Q/N/X sits inside Figure 3 panel (e), not as a headline figure.
- Manuscript updated for thirty candidates, the Q/N/X stratification described in
  plain evidence terms, and the sharpened novelty statement. Stale ten-candidate
  references removed.
- Terminology fixed throughout: the 30 are prospectively selected candidates, not
  hits, until they show experimental activity.

## 2026-08-06 - Branched cohort added; panel v4 at 36

- A structural audit of the locked 30 found zero branched tails on either side. The
  cause was traced and is neither a selection artifact nor a generator deficiency.
- The main 32,768-draw production run carries `maximum_adjacent_branch_runs [2,1,1]`,
  which suppresses aldehyde-side branching: 28.2% of admitted designs are branched
  but **zero** carry an ester-linked branched aldehyde across all 32,768 draws. The
  panel was drawn only from that run.
- A correction to an earlier claim in this log: I first concluded FORGE could not
  generate the motif. That was wrong, and was based on auditing only the main
  production run. The separate branch-conditioned 4,096-draw lane, generated earlier
  under the same frozen model, produced 720 ester-linked branched aldehyde designs,
  45 applicability-qualified, 38 passing the chemistry and physicochemical screens,
  and 14 of 14 checked were route-complete. No retraining was proposed or performed.
- Score comparability verified before use: the branch lane draws conformal quantiles
  from the same set as the main pool (3.91472, 5.309392, 6.214549 against the main
  pool's four; the lane simply never reaches 4.295945), and lcb90 = oracle_mean minus
  conformal_q90 holds exactly in both, 4186/4186 and 42/42. An lcb90 of 8.90 in the
  lane is directly comparable to 6.59 in the main pool.
- Motivation: aldehyde branching is enriched 3.8-fold among the top 50 measured AGILE
  products, 17 of 50 against 9% of the full set. One measured aldehyde carries the
  motif and appears in 100 measured products with median HeLa 5.32 and max 15.08.
- **No eligibility criterion was loosened.** B passed the identical chemistry filter,
  physicochemical envelope, applicability qualification and route-completeness
  requirement. Only the within-cohort diversity metric differs, because a
  whole-product Tanimoto ceiling is self-defeating for a cohort that exists to test
  one motif: at the 0.70 ceiling used for N and X, only 2 of 14 route-complete
  branched candidates survive, purely because they share the branched aldehyde. The
  rule for B is distinct amine head, preferring a distinct isocyanide.
- **No branched candidate is both high-confidence and high-scoring.** The five
  qualified role-holdout branched designs top out at lcb90 1.63, below Q-high's floor
  of 3.19, and none is conservative-high. The high scorers are all exploratory tier.
  B therefore spans both, reported as two subgroups and never pooled:
  - B_exploratory, n=4, lcb90 3.17 to 8.90, generated longer-spacer variants of the
    4-methylnonanoate acyl chain;
  - B_qualified, n=2, lcb90 1.34 to 1.63, both on the exact measured AGILE branched
    aldehyde, which is directly purchasable, serving as an in-domain reference on the
    same acyl motif.
- The locked 30 are unchanged. B is additive. Panel is now 36: 12 Q-high, 7 Q-low,
  8 N, 3 X, 6 B. All 36 route-complete, at most 4 synthetic steps, 45 catalogue
  materials, 15 amines, 19 aldehydes, 10 isocyanides.
- Artifacts: `configs/bio/phase1_ugi_prediction_cohorts_v4.json` (v1, v2, v3
  preserved), `results/phase1/ugi_prediction_cohort_panel_v4/` with refreshed
  manifest and hash, six-block blinding key, and a 45-row bill of materials.

## 2026-08-06 - Backups frozen; manuscript updated to the 36-candidate panel

- Ordered backup lists generated deterministically under the same frozen eligibility,
  chemistry, envelope and diversity rules as the panel: Q-high 3 (lcb90 2.85, 2.58,
  2.45), Q-low 3 (-3.84, -3.63, -3.43), N 3 (8.09, 7.68, 6.57), B 2 (7.05, 1.68).
  No X backup: the eligible X pool is genuinely thin and manufacturing one would
  require weakening X eligibility.
- Activation policy, locked in `BACKUP_LISTS.locked.json`: backups may be used ONLY
  for a documented pre-synthesis chemistry-signoff or procurement impossibility. They
  may not replace candidates that fail synthesis, purification, formulation or
  biological testing once experimental work begins; those failures stay in the
  end-to-end accounting. The panel hash is recorded before and after any permitted
  replacement.
- Manuscript updated from thirty to thirty-six throughout, and the panel paragraph
  rewritten to describe the branched cohort in plain evidence terms: branching is
  enriched among the most active measured lipids (17 of the 50 highest-transfecting
  products against 9% of the library), one measured aldehyde carries the motif, the
  main generation run constrains adjacent branch runs so these designs come from a
  separately conditioned run of the same frozen model, four are generated spacer
  variants outside the prediction domain and two use the measured branched aldehyde
  and remain inside it. The paragraph states explicitly that no branched design is
  both high-confidence and high-scoring.
- Main text is 2,706 words against the 3,000 limit with nine result placeholders
  remaining, all prospective. PDF rebuilds with no errors.
- Panel remains NOT cleared for procurement: chemistry signoff and vendor completion
  of the 45-row bill of materials are the two external blockers.

## 2026-08-06 - Panel structure figure rendered (authorized)

- Figure production was authorized by the user for this item only. All thirty-six
  locked structures rendered to `manuscript/figures/panel_v4/panel_36_structures.png`
  (2400x3780, RDKit, four columns), grouped by prediction-evidence cohort and ranked
  within each, labelled with candidate ID, cohort and score.
- Embedded in the manuscript as Figure 4a with a caption stating explicitly that the
  two score types are not comparable: the calibrated conformal lower bound where the
  activity model issues a qualified estimate (Q and B), and the conservative ensemble
  mean-minus-standard-deviation where it abstains (N and X).
- Manuscript now 15 pages, builds without error.
- No other figure artwork has been produced. The remaining figures in the section 10
  architecture stay unmade pending explicit authorization.

## 2026-08-19 - Reproducible experiment runner and refactor provenance gate

- Added the strict `forge.experiment.v1` DAG contract, content-addressed local execution, keyed
  replicate/stage seeds, atomic stage commits, verified resume, failure receipts, and independent
  downloaded-run verification. Added one generic Modal launcher; GCP remains deferred.
- Added public `forge.corpus` and registry-backed `forge.assembly` seams. The L1 adapter reports exact
  forward consistency only and does not promote reaction support to route certification or synthesis
  success.
- Rebuilt the Phase 1 product/L1 corpus in an isolated run. Both compressed ledgers reproduced the
  frozen bytes exactly: 12,386 constitutional model products and all 13,376 source rows. A second
  independent execution reproduced all four stage artifacts byte-for-byte.
- The first execution exposed an inherited plumbing defect: `phase1_data` called the shared gzip CSV
  writer without its required field order. Supplying the already-frozen `ASSIGNMENT_FIELDS` fixed the
  invocation; the regenerated assignment ledger remained byte-identical.
- Archived 137 exact historical source/config blobs. The active-plus-archive gate now verifies 739
  pins over 337 files with zero drift. Three pre-existing unrecoverable digests remain exact exceptions;
  they are no longer broad exceptions tied to a mutable current file.
- This is an engineering/provenance milestone. It changes no scientific gate, candidate status, sealed
  holdout, biological authorization, or paper claim.

## 2026-08-19 - Register resumable Phase 1 training and diagnostic sampling DAGs

- Registered separate CLI-owned training and sampling experiments. Training verifies the frozen
  112,386-record balanced tensor cache before running the joint sparse-flow and closure trainers;
  the production joint refit remains fixed-final-step and explicitly requests an L4 GPU.
- Checkpoints now carry optimizer state, Python/NumPy/PyTorch random states, trainer-generator state,
  loss/evaluation history, and selection counters. Failed stages retain a fingerprint-bound partial
  workspace and `--resume` continues only the identical run contract.
- Registered restartable contiguous sampling shards over the selected development checkpoint and the
  matched 3,072-program draw. Seeds are derived independently per shard; merging is deterministic;
  no retry, repair, candidate selection, route call, oracle call, or synthesis-value call is allowed.
  Exact Ugi-L1 reconstruction is reported without promotion to L2/L3 closure or synthesis success.
- The bounded CPU smoke run completed 10 joint steps over all 112,386 training records and 300
  closure steps. The four-sample diagnostic returned four valid, terminal-valid, exact-L1 products.
  These are pipeline qualification results, not production training or candidate evidence.
- `scripts/` is now documented as a frozen compatibility/provenance surface. New workflows use the
  CLI; legacy producers remain in place until a registered DAG covers their behavior and every
  historical path/hash reference is recoverable. The stale root migration note was removed after its
  durable repository provenance moved into `DATA_PROVENANCE.md`.
- This engineering change does not launch production GPU training, select candidates, access sealed
  holdouts, alter guidance, or change a scientific claim.

## 2026-08-19 - Extend the provenance gate over `configs/`, which pins its own source

- **The gate had a blind spot.** `verify_artifact_pins.py` scanned `results/` and `docs/provenance/`
  only. A frozen config also pins the code that produced it, as `inputs.source.{path,sha256}`, and
  those pins outnumber the result-declared ones roughly four to one. None of them were ever checked,
  so `make verify-pins` read 739 verified / 0 drift while **216 config-declared pins over 117 files
  had gone stale**. The green gate was accurate about what it looked at and silent about the rest.
- Of the 117 drifted files, **87 had their pinned bytes at `cc947f5`**, this repository's first
  commit — so the drift was introduced by work done here, most of it by the `forge.core` migration
  editing modules that a frozen config pins. The remaining 30 exist in no commit at all.
- **Fixed a defect in the gate itself:** `collect_pins` assumed absolute roots, so the advertised
  `--root` flag crashed on any relative path. That is why nobody had pointed it at `configs/`.
- Ran the archiver over `results/` + `docs/provenance/` + `configs/`. It recovered **312 additional
  historical revisions**; the archive grew 137 -> 449 entries, a strict superset losing nothing.
  Config drift fell 216 -> 92. Combined coverage is now **2,041 verified pins over 667 distinct
  files**, up from 739 over 337. `configs/` is now in the archiver's default roots.
- Added `make verify-pins-code` (all three roots) and `make archive-pins`. The new `--allow-drift N`
  flag is a **burn-down ratchet, not an acceptance mechanism**: it holds the known backlog flat and
  fails the moment drift grows. Negative-tested at N-1 (exit 1), N (exit 0), and on the unratcheted
  main gate (exit 0). Accepted drift still requires a reviewed `known_artifact_drift.json` entry.
- **92 pins over 36 files remain drifted and unrecoverable.** They are listed by
  `make verify-pins-code`. Each needs an individual ruling on whether its bytes are genuinely
  unrecoverable and its artifact frozen; `REFACTOR_BASELINE.md` forbids adding entries to make a
  gate pass, so none were added. This is an open decision for the user, not a chore.
- `make verify-pins` is unchanged at 739 / 0 drift. This is an engineering and provenance change: it
  alters no scientific gate, candidate, sealed holdout, biological authorization, or paper claim.

## 2026-08-19 - Freeze the ICLR reproduction graph and retire the first proven-dead code wave

- Froze `configs/reproduction/iclr2027.json` over the authoritative LaTeX source, twelve numerical
  evidence roots, three generated LaTeX inputs and all ten figures actually included by the paper.
  All 26 direct files verify at their declared SHA-256 values.
- Added `forge paper doctor|verify|reproduce|render|build|bundle`. Artifact replay is named as such:
  the reproduction receipt states `numerical_recomputation_executed: false`. Strict diagnosis walks
  1,190 recursive path/hash identities and currently finds 147 active, 25 archived, two drifted and
  1,016 absent identities. Full numerical recomputation is therefore **not ready** on this checkout;
  the final JSON files are not presented as a substitute for unavailable original corpora, remote
  checkpoints or external-engine outputs.
- Paper compilation and Overleaf packaging no longer depend on stale `manuscript/` shell scripts.
  Both run in clean temporary directories through the CLI. Setting the TeX reproducible-build epoch
  made two independent PDFs byte-identical at
  `04d54a7fca0666c0ac90d921c359e80bd62a41b65ab6662ca69220ec51ed7980`;
  the deterministic 23-file bundle reproduced at
  `569390451730c1562a8ed9dc8163b57db81624515cfbe3cbf27bea45a61c871f`.
- Moved data vendoring, generic Modal dispatch and provenance verify/archive implementations from
  top-level scripts into typed package modules. Removed the two stale paper shell builders, the old
  Nature-draft builder, seventeen superseded experiment-specific Modal launchers and six unused
  figure builders: 30 top-level Python/shell files in total. Three Modal launchers remain as direct
  ICLR producers and one remains because three historical identities are not fully archived.
- Added the reproducible code survey at `provenance/code-retirement/iclr2027.json`. After the first
  removal wave it classifies 24 files as CLI-only, 262 as paper-only, 69 as shared, 14 as blocked by
  historical pins and 287 as further retirement candidates. The latter are an audit queue, not an
  instruction for bulk deletion; unique acquisition, chemistry-adjudication and negative-result logic
  still requires review.
- Result-facing provenance remains 739 verified pins over 337 files with zero drift. Including frozen
  configs now verifies 2,067 pins over 686 files while holding the inherited 92-pin drift backlog
  flat. No known-drift exception was added.
- Re-ran the current bounded CPU pipelines after the retirement wave. Training run
  `a249f0e4dcc084bff48e81794348cf9fdabb88ba48f246c05d8b696477ff306d` completed and verified all
  three stages; independent reproduction met its frozen strict/statistical contracts. Sampling run
  `3c3f005674c588ab32db98b582ddfc2364f296348060ddabfa5543326cb0fc51` completed, verified and
  reproduced both outputs byte-for-byte
  (`a07a9a034ccbde1a6776bed98f48f2b341a7ed3edced24169a7baae6de21edc0` result,
  `82278c058fe11159f4d56dd50aea73ee56f06fe45c07198d692cd4dcd3e88de5` shard tree).
- The supported-core gate passes 66 tests, strict typing and lint. The unscoped historical test suite
  remains red with 202 failing/error cases across 73 test files. Representative blockers include
  absent historical result ledgers and sealed-sample directories plus frozen source hashes affected
  by the concurrent `bio` to `potency` namespace migration. These failures were not skipped, repinned
  or weakened to make this refactor green.
- This is an engineering and reproducibility change. It launches only bounded CPU smoke training and
  sampling; no production training, production sampling, route engine, biological guidance or
  candidate selection ran, and no scientific result or evidence tier changed.

## 2026-08-19 - Complete the `bio` to `potency` extraction and make the full-suite blocker reproducible

- Moved all 49 oracle, applicability, morphology, ranking, and authorized-diagnostic implementations
  from `forge.bio` to `forge.potency`. `forge.bio` now contains only its package initializer, generic
  endpoint interface, frozen endpoint decision, and liver/muscle/vaccine implementations. All 49
  moved modules import successfully.
- Added an executable namespace boundary: every old path must be absent, every declared move target
  must exist under `potency`, runtime source/scripts/tests may not import a removed `forge.bio.*`
  module, and the six-file endpoint-only `bio` surface is exact. The remaining `product -> potency`
  dependencies are recorded as transitional coupling for the later product split, not hidden by this
  move.
- Corrected a provenance mistake made during the move: 22 frozen configs had been edited to name the
  new path. Restored their historical `src/forge/bio/...` source identities and made the relocation
  resolver carry the migration instead. The frozen-code archive grew from 449 to 460 exact content
  blobs; the combined gate improved from 2,067 verified / 92 drift to 2,090 verified / 72 drift. The
  result-facing gate is 742 verified pins over 340 files with zero drift. No drift exception was
  added.
- Added `forge maintenance test-report` and `make test-baseline-report`. A clean-cache run collected
  1,668 nodes and found exactly the 201 reviewed failing/error nodes, with zero new failures, zero
  resolved failures, and zero stale cache entries. The pinned report is
  `results/maintenance/bio_to_potency_migration_v1/test_baseline.json`.
- The full suite is still **blocked, not green**: all 54 files in `docs/missing_test_inputs.txt` remain
  unavailable. The report preserves that negative result instead of skipping tests, weakening a
  scientific gate, substituting data, or changing frozen expected hashes.
- `make verify` authenticates all 30 vendored assets. `make check-core` passes provenance, strict
  typing over 50 supported source files, lint over the migrated namespaces, and 70 supported tests.
  The refreshed code survey classifies 25 CLI, 262 paper, 69 shared, 7 historical-pin-blocked, and 294
  further retirement candidates; those candidates remain a review queue rather than an automatic
  deletion instruction.
- This is an engineering/provenance closeout. It launches no training, sampling, route engine,
  biological guidance, or candidate selection and changes no scientific result, evidence tier,
  sealed holdout, or paper claim.

## 2026-08-19 - Triage the 72 drifted code pins; 56 accepted, 16 held open

- **The drift is inherited, not caused by the restructuring.** Checked against `cc947f5`, this
  repository's first commit: 18 of the 20 drifted files *already* differed from their pinned digest
  there and none matched. The pinned revisions therefore predate this tree. An earlier note in this
  log attributing the config-pin drift largely to the `forge.core` migration was true of the
  *recoverable* pins, which are now archived; it does not describe this residue.
- **Recovery was attempted before any exception was written, and failed exhaustively.** Each of the
  37 distinct missing revisions was searched for by content in every blob of this repository (2,194
  over 31 commits), every blob of the predecessor `forge` repository (2,226 over 245 commits), and
  every copy of the filename on the originating filesystem (153 copies). Zero matches. The
  experiments ran against a working tree carrying uncommitted edits, so the recorded hash describes
  a file state that was never committed anywhere and cannot be reconstructed.
- 29 exceptions covering **56 pins over 13 files** were added to `docs/known_artifact_drift.json`.
  Every one meets both halves of the stated bar: the bytes are unrecoverable, and every declaring
  config carries an explicit `frozen_*` status. `CODE_DRIFT_BACKLOG` drops 72 to 16 and was
  negative-tested at 15, which fails.
- **16 pins are deliberately left open, in two groups.** Two are the unrun
  `phase1_ugi_planner_reach_sweep_v1` pins against `ugi_bounded_hybrid_route_cascade_v1`, whose
  outputs are present but a later vintage than the config froze against; that needs a decision about
  which vintage the sweep runs on, not an exception. The other 14 sit under eight configs whose
  schema has no `status` field at all — `fresh_pool_route_coverage` carries `status` and `task` at
  v1-v3 and drops both from v4 onward — so "the artifact is frozen" is not asserted and the
  acceptance bar cannot be applied. Fixing that schema regression is a prerequisite, not a chore.
- This is a provenance bookkeeping change. It alters no scientific gate, candidate, sealed holdout,
  biological authorization, or paper claim, and `make verify-pins` is unchanged at 742 / 0 drift.

## 2026-08-20 - Adopt a root scientific library and explicit experiment applications

- Removed the `src/` indirection and top-level `scripts/` workflow surface. `forge/` is now only the
  reusable scientific library; `cli/`, `experiments/_runtime/`, named applications under
  `experiments/phase1/`, `paper/`, and `tools/` are separate dependency layers. Catch-all
  `forge.audit`, `forge.bio`, `forge.cli`, `forge.data`, `forge.design`, `forge.experiment`,
  `forge.route`, and `forge.value` namespaces no longer exist.
- Unified training and sampling as separate typed DAGs in the `product_l1` application. The current
  source tree reproduced CPU training run
  `0e01e1d3c5dece81213c33b48f302ac00660528bf8fcc1eead8307c968034b6a` under its declared
  strict/statistical contracts. Sampling run
  `fc4731a2de7b6f93c40b7290a54a247d978f659073e82f3fc4cb6a26ce63f3be` reproduced both outputs
  byte-for-byte. Both bind source digest
  `745f05d4fcf4e810c96211e4a406073231626f0959d5e8938dcae43f5bdfd367` and record their seed
  hierarchies. The L4 production DAG was planned with all 14 uploads verified; no remote job ran.
- Quarantined historical producers and reports under `experiments/archive/`; active experiment code
  is forbidden by test from importing it. The explicit catalog exposes five supported experiment
  specifications. The current retirement survey records 11 CLI, 183 paper, 73 shared, and 285
  retire-candidate files; candidates remain an audit queue, not an automatic deletion instruction.
- Preserved historical path identities through 837 reviewed move records and 478 content-addressed
  code/config revisions. Result provenance verifies with zero drift. Config-inclusive verification
  accounts for 2,117 pins; the same four genuinely unresolved revisions remain visible under the
  four-item burn-down ratchet. No known-drift exception or scientific gate was added.
- A clean-cache full-suite run collected 1,678 nodes. After repairing all 36 path/hash regressions
  introduced by the move, the observation returned to exactly the reviewed 201-node baseline with
  zero new failures. The suite is still blocked, not green, because all 54 documented historical
  artifacts remain absent. `make verify` authenticates all 30 vendored assets, and `make check-core`
  passes architecture, provenance, strict typing, lint, and its supported tests.
- The attributable closeout is
  `results/maintenance/root_package_architecture_v1/result.json`. This was an engineering and
  reproducibility change: no production training or sampling, synthesis-value or biological
  guidance, candidate selection, sealed-holdout access, evidence-tier change, or paper-claim change
  occurred.

## 2026-08-20 - Authorize a bounded computational multi-reaction extension

- The user authorized the source-grounded computational extension in
  `MULTIREACTION_COMPUTATIONAL_PLAN.md`: Ugi remains the deep validated case, BL_2023 repeated
  aza-Michael addition is the full second family, and LX_2024 repeated reductive amination is a
  lighter stress test. This amendment supersedes the earlier statement that every ICLR experiment
  uses only a Ugi adapter, but does not authorize wet-lab work, biological-label transfer,
  arbitrary retrosynthesis, yield claims, or new synthesis-success claims.
- Source review showed that both added libraries require repeated execution of a two-reactant
  transform for polyamines. The shared object is therefore an ordered reaction program, not a
  fixed one-head/one-tail tuple. Each step is registry-backed and every admitted complete program
  must reproduce the source product exactly in forward replay.
- The first deterministic corpus build examined 773 BL_2023 and 851 LX_2024 source rows, preserved
  all 1,624 row identities, and collapsed them to 891 constitutional products. It admitted 610 of
  720 source-reported aza-Michael members and 154 of 180 reductive-amination members, passing the
  frozen 576 and 144 gates. The atlas preserves 127 abstentions, including unresolved reverse
  decompositions, ambiguous decompositions, inconsistent LNPDB product/component assignments, and
  the A3-6b source conflict. Exact source-adjudicated forward precision among admitted programs is
  1.0 by construction; this is not a synthesis-success rate.
- The admitted set contains 1,344 exact program steps and deterministic component-disjoint product
  folds of 328 train, 196 calibration and 240 heldout products. Sampling weights equalize total
  weight per reaction program within each fold; raw family counts are not used as the prior. Exact
  isotope-tagged forward replay assigns accumulator-versus-repeat precursor origin to all 56,588
  atoms across all 764 admitted products without a component identifier entering model state.
- This decision changes the computational data/model interface and paper experiment plan. It does
  not alter any frozen Ugi artifact, candidate, holdout, evidence tier, route dossier, synthesis
  value, or biological result.

## 2026-08-20 — Multi-reaction sparse-flow plumbing passes; scientific comparison remains blocked

- Implemented a role-blocked whole-product sparse representation for the 764 admitted aza-Michael
  and reductive-amination products. It covers the complete observed support: 194 heavy atoms,
  three residual closures and program depth four, with no small-molecule truncation.
- Froze precursor role plus complete program depth as the invariant semantic coordinates. Ordered
  steps remain in the exact execution trace, but identical repeated precursor copies are not given
  fabricated per-atom step identities when graph symmetry makes those identities non-identifiable.
- Added matched program-conditioned, null and program-ID-swapped training arms and a separate
  checkpoint-consuming sampler. The deterministic JSON tensor checkpoint is non-executable and
  reproduced byte-for-byte.
- Diagnostic run `30af60ef1ffd67098406c034e8db58b0835dffad16369324c7b94267ee3bebbc`
  completed and independently reproduced byte-for-byte. The authenticated training checkpoint is
  `ab82baf6f096e73da51ea04105615cf6c7123c9cd02acc132f70dbc8c797aa26`; all three
  four-step arms had finite loss. Sampling gave 3/4 sanitized graphs and 0/4 exact L1 programs.
- Decision: the pipeline gate passes, but the four-step checkpoint supplies no comparative model
  evidence. Zero exact-L1 samples is retained as a negative diagnostic. Do not use this smoke in a
  paper table or claim multi-family generation quality; production multi-seed training remains a
  separate pending experiment.

## 2026-08-20 — Reaction-core representation and strict overfit gate pass; production remains blocked

- The first multi-reaction tensorization encoded precursor origin and program depth but omitted the
  registry reaction-core position. That left the model without the intended local semantic
  coordinate. The corrected v2 atom ledger carries namespaced registry map positions for 6,798 atoms
  and has core annotation in all 764 admitted products. Ugi origin and core state remain orthogonal;
  no component identifier or stored fragment enters the model.
- Added position-aware sparse decoding because canonical node position is required to distinguish
  otherwise symmetric parent-pointer targets. A first bounded run without this feature failed the
  tensor reconstruction gate. A later apparent tensor failure was traced to the evaluator taking an
  argmax over impossible self/future parents; masking to the same admissible parent set used by the
  model corrected the measurement without changing a frozen threshold. Both negative intermediate
  outcomes are retained as engineering evidence and were not converted into weaker gates.
- Frozen overfit run `53b7d98c2c3dfcba76eaa5664966470c1710af4137061458dfe1af1f14b71863`
  passed all five prespecified gates: finite loss, final/initial loss ratio 0.0001424 against a 0.5
  ceiling, exact fixed-noise tensors for 2/2 family-balanced records, 16/16 valid native samples and
  16/16 exact-L1 products. All 24 exact decomposition traces reproduced their products in forward
  replay. The strict reproduction hashes are `9d753862d430fb727658228d85ccc9c55f112ce88a83b1e260c4b66efaf44cc0`
  (training checkpoint), `4cd9f6be4a372f5653da3269c72726b6dd47651c84518a3a6538a50796b50332`
  (training result), `46c2795314bd0aa02b99465f36699d0148c45e94caa6b0f4e5064b3a7cb6e636`
  (samples), `fa5e680eaccb93b328c1bada552a79e12a216fa0cf1b055f9e728157d20ad2b6`
  (sampling result), and `f9fd33c16e89ab3bf93a9b7962ba48c6991ae350f941dc20eab2cc714ed61dbc`
  (qualification result).
- Added role- and family-specific non-selecting metrics and deterministic within-stage restart.
  Restart checkpoints capture model, optimizer, Python/NumPy/torch RNG and explicit generator state;
  a forced interruption at step two produces the same final model hash and published metrics as
  uninterrupted training. Interruption history remains private work state so it cannot change the
  scientific artifact bytes.
  Smoke run `a786b22cdcd4fcd5a70b90c8670b79c331b8736e7167914bbed9f22bbeb43a22`
  reproduced byte-for-byte and retained its negative 0/4 exact-L1 result.
- **Decision:** the reaction-core representation gate passes for the BL/LX repeated-program subset.
  It does not authorize a paper-facing production comparison. The next gate is a single matched
  Ugi+BL+LX representation supporting three distinct Ugi precursor roles, the assembly-introduced
  amide oxygen, aromatic chemistry and the full union size support. Until that closes, a BL/LX-only
  production run would answer the wrong question and will not be launched.

## 2026-08-20 — Shared Ugi+BL+LX whole-product representation gate passes

- Implemented one aromatic-aware program-graph contract for AGILE-type Ugi 3CR, BL_2023 repeated
  aza-Michael assembly and LX_2024 repeated reductive amination. It carries program identity,
  complete program depth, exact precursor-origin role and namespaced reaction-core position. Its
  origin-component blocks are deterministic serialization spans only: no component identifier,
  stored fragment or purchasable-block vocabulary enters model state.
- The fail-closed full census represented and constitutionally reconstructed **113,150/113,150**
  declared products and **4,472,034/4,472,034** atoms: 112,386 Ugi, 610 aza-Michael and 154
  reductive-amination records. It preserved explicit aromatic atoms and bonds, the observed maximum
  of 194 heavy atoms and all three residual closures. No molecule was size-capped or omitted.
- Ugi's five exact core atoms remain adapter-fixed and orthogonal to precursor origin: 561,930 core
  atom states were marked fixed across the Ugi corpus. Auxiliary-family core positions are retained
  as semantic coordinates but are not silently promoted to fixed chemistry invariants.
- CLI run `d40b98c40a28bf300aa84a9cd5b92a13a4dd4524e76997929451d3682cae746a`
  completed successfully and independent strict reproduction was byte-identical. Its result matches
  the checked-in qualification artifact at SHA-256
  `487c7382af3a18099505aa1e01b981b2b9b9c7b191e0cd7a36b79ad5c318bb88`; the representation digest is
  `ef10c20f443ee8a272b916a6da76556cdbaca9e53d58cae75abb5c6069301534`.
- The supported-core gate passes strict typing, lint and all scoped tests. Result provenance now
  verifies 773 pins over 360 files with zero drift; config/code-inclusive verification accounts for
  2,201 pins while preserving the inherited four-item drift ratchet. The unscoped suite collected
  1,702 nodes and returned exactly the reviewed 201 missing-artifact failures, with no new, resolved
  or stale-cache nodes; all 54 documented historical inputs remain absent.
- **Decision:** the shared representation gate is closed. This establishes lossless support, not
  generalization, synthesis success or route certification, and does not authorize production
  training. The next gate is shared-cache/model integration with exact Ugi fixed-core noising and
  decoding, followed by a bounded three-program overfit/equivalence test and Ugi regression control.

## 2026-08-20 - Shared cache, model, and fixed-state sampling integration gate passes

- Built a deterministic model-ready cache from one training-fold product per admitted program. The
  cache assigns equal total mass to AGILE-type Ugi 3CR, BL_2023 repeated aza-Michael assembly, and
  LX_2024 repeated reductive amination. It contains complete sparse product targets and categorical
  program coordinates, but no component identifier, precursor SMILES, fingerprint, fragment token,
  or biological label. This three-record cache is an integration set, not a production corpus.
- The Ugi regression control mapped both serializers back to original product atom indices. The
  shared record preserved all 5 adapter-fixed atom states and all 7 adapter-fixed edge states and
  endpoints exactly. BL and LX retained their reaction-core positions as conditioning coordinates
  but had zero fixed atoms or edges. On those zero-fixed records, shared noising tensors, loss, and
  reported loss components were exactly equal to the generic sparse-flow path.
- A first 768-step run failed the unchanged exact-tensor gate because one of 32 LX parent pointers
  was wrong. Increasing the bounded optimization budget closed that training failure. Native
  sampling then produced 12 of 12 valid and constitutionally exact molecules but initially reported
  only 8 of 12 exact tensors. The sole mismatch was parent-bond slot zero, the masked root sentinel,
  which the decoder had incorrectly overwritten with an argmax despite excluding it from noising,
  loss, and molecular construction. Increasing training to 3,072 steps did not change this pattern.
  The decoder was corrected to preserve all inactive sentinels; no acceptance threshold was relaxed.
- In the final computed run, loss fell from 16.315578 to 0.000008165, a final-to-initial ratio of
  0.0000005004. All three records reconstructed every active tensor exactly at fixed noise. The
  checkpoint-separated sampler produced 12 of 12 valid molecules, 12 of 12 exact tensors, and 12 of
  12 exact source graphs, with four samples from each program and zero fixed-state changes across
  initialization, 32 flow steps, and terminal decoding.
- The attributable artifacts are
  `results/phase1/shared_synthesis_program_training_integration_v1/`: cache
  `bd053e2d79e584e4f119d96aed73810f56d49bded62bf7dc8525d187f4b855bf`, cache result
  `587e44881783798b137b39555732f665513d3315b67c2758a0560b21c690caa8`, training checkpoint
  `feb9d0c27531e183a4108a7dec7f58360c80969eed3a82ade123da46fcd66e5e`, training result
  `289fa4caa5a6a62325bb74d8159fe20425f98e4c9e47c35b6cf16ef7624fe50a`, samples
  `6842bbe03e5b93ff455988e9b16f62fe75d00c92ed12df86d1ec63bff32b13f8`, sampling result
  `280102a21f604a8000df686d7d525e3b3f38390397637568af8f672c0f24f96f`, and final qualification
  `ecfd968181e3729843d4ccac6f3a7d046dc8475ff23e5ead49fbe2193c74cad1`. Registered run
  `b2912ab0065ff187e25afd44bdd1c30a02fcfe66dd0efd158560006b5da02863` verified and reproduced
  every stage artifact byte-for-byte. Produced-artifact receipts use logical paths, so independent
  reruns do not record transient staging directories as data dependencies.
- Vendored-data verification passed for all 30 present assets. Result provenance verifies 794 pins
  over 365 files with no drift. Config/code-inclusive verification accounts for 2,242 pins while
  preserving the inherited four-item drift ratchet; no new drift was introduced.
- The scoped architecture, type, lint and test gate passed. The full-suite baseline audit collected
  1,710 tests and returned exactly the frozen 201 failures caused by 54 documented unavailable
  historical inputs, with no new failures, resolved failures or stale-cache nodes.
- **Decision:** the bounded shared cache/model/noising/loss/decoder integration gate passes. This is
  computed pipeline evidence only. It does not demonstrate held-component generalization,
  synthesis success, route certification, or biological performance, and it does not authorize a
  production multi-reaction training or sampling launch.

## 2026-08-20 - Freeze the matched multi-reaction production design; launch remains blocked

- The deterministic preflight verified the actual model-union partitions before freezing a design:
  Ugi contributes 66,464 training, 15,800 calibration and 30,122 heldout products; BL contributes
  258, 158 and 194; LX contributes 70, 38 and 46. Ugi retains
  `family_balance_weight_raw`; BL and LX retain `source_balanced_weight`. Every program is normalized
  within fold before arm-level mass is assigned, so raw family counts cannot become the prior.
- The proposed comparison contains four architecture- and compute-matched arms: Ugi-only,
  conditioned Ugi+BL+LX, null-conditioned Ugi+BL+LX and a nonselecting cyclic program-ID control.
  The three-family arms assign exactly one-third total mass to each program. Every arm uses the same
  194-atom, three-closure sparse model, shared source marginals, three deterministic seeds, 1,700
  optimizer steps, effective batch 128 and diagnostic checkpoints at steps 100, 500, 1,000, 1,500
  and 1,700. The final step is fixed before training; neither calibration nor heldout results select
  a checkpoint or threshold.
- Family-specific evaluation must report validity, connectedness, exact-L1 decomposition coverage,
  exact-forward replay precision, abstention, ambiguity, diversity, effective component count and
  component/whole-product novelty. Ugi retention uses a paired hierarchical-bootstrap
  noninferiority comparison against the Ugi-only arm. Fixed-state changes and support overflow have
  zero tolerance. Held-reaction-family analysis is secondary, and the degenerate
  reductive-amination substructure hit rate remains forbidden.
- The design preflight passed all 12 gates. Artifact
  `results/phase1/shared_synthesis_program_production_design_v1/result.json` has SHA-256
  `54ca338e3e7706eccf6af4f32eaa80e8782136301446ed268400f3ffccab2ea7`. Registered run
  `85559f2ad7b66e6a1f694d0cc77c59e6b3336203b1a7e5bfd5f36f98f8021f68` verified and reproduced the
  result byte-for-byte. Result provenance verifies 803 pins across 368 files with zero drift;
  config/code-inclusive verification accounts for 2,268 pins under the unchanged four-item drift
  ratchet.
- All scoped architecture, type, lint and design tests pass. A clean full-suite run collected 1,716
  tests and returned exactly the frozen 201 failures attributable to 54 documented unavailable
  historical inputs, with no new failures, resolved failures or stale-cache nodes.
- **Decision:** the production-qualification design is frozen. This is a proposed experiment backed
  by computed corpus/preflight checks, not a trained-model result. It performs zero route or oracle
  calls, selects no candidates and does not authorize production training or sampling. Production
  cache/trainer/evaluator implementation and any GPU launch require explicit authorization after
  review of this design.

## 2026-08-20 - Production pipeline implemented; external L4 launch awaits explicit upload approval

- Implemented the frozen four-arm production pipeline without changing the design. The packed,
  numeric-only cache contains all 113,150 declared records and preserves the 194-heavy-atom and
  three-closure support. Its checked-in cache SHA-256 is
  `db1e71b836b42d817cb658c469973db1722af2f1388292c79201c861ad33db41`. Training is
  restartable at optimizer-step boundaries and publishes an allow-listed, hash-authenticated
  checkpoint archive. The evaluator loads published checkpoints with the restricted PyTorch
  loader, samples every fixed checkpoint without repair or retry, reports coverage and precision
  separately, performs a complete held-component tensor census in the full profile, and writes a
  seeded random plus worst-case molecule audit without selecting a candidate.
- The repinned two-step CPU execution smoke completed all four arms and both evaluation stages under
  run `48da8a4081c6bfdb356653dba16eaac5b14855d9fef20b7fce41f10e4e76f81f`. All 13
  infrastructure gates passed. As expected for two optimizer steps, 0 of 40 sampled graphs were
  valid and the component-disjoint exact-tensor fraction was zero. This negative is retained as a
  plumbing result only and is not model-quality evidence. The authenticated training and evaluation
  receipts have SHA-256 values `50d809761dc04bda7c8a85411e5150036874438750834e4bc994164a99fe26e8`
  and `441f31f46451e9a245f3c5ceb8bd7d16b30aa87f4381d4ccbf4536a152f71c07`.
- Vendored-data verification passed for all 30 assets. The current suite collected 1,724 tests and
  reproduced exactly the reviewed 201 failures caused by the unchanged 54 missing historical
  inputs, with zero new failures, zero resolved failures and zero stale nodes. Result provenance
  verifies 820 pins across 372 files with no drift. Config/code-inclusive verification accounts for
  2,341 pins under the unchanged four-item drift ratchet. All scoped tests, lint and targeted typing
  pass.
- Added a separate one-job L4 preflight that must exercise the exact full model, effective batch 128,
  every arm, deterministic challenger replay and a 32 by 194-atom capacity stress before production
  training. It also authenticates the smoke receipts and no-new-failures baseline. The preflight
  makes zero route, oracle or candidate-selection calls and cannot publish a model candidate.
- **Decision:** no external computation ran. The sandboxed Modal connection failed, and the elevated
  request was denied because it would upload repository source, pinned configurations, the 7.3 MB
  cache and smoke receipts to Modal and incur external L4 cost without an explicit payload-level
  approval. Do not bypass that control. After the user explicitly approves this Modal upload and
  cost, run the bounded L4 preflight first. Launch the three fixed production replicates only if
  every preflight gate passes.

## 2026-08-21 - Make LNPDB the common row source without conflating data and evidence

- Added one typed LNPDB catalogue for stable row identity, study selection, whole-lipid structures,
  normalized targets, and source-reported head/linker/tail slots. The potency and multi-reaction
  builders now consume this same interface instead of independently parsing the 19,797-row CSV.
- Reduced the new potency contract to raw LNPDB plus one config. It contains 2,200 `YX_2024`, 288
  `JC_2023`, and 1,128 `LM_2019` observations. The 3,616 model rows plus 200 configured B4-mixture
  exclusions account for all 3,816 selected source rows. Every target is explicitly
  `lnpdb_within_study_endpoint_zscore`; the typed loader still requires a study/endpoint pair.
  Reaction-program IDs, component payloads, exact-program eligibility, parallel AGILE raw labels,
  and model-identity correction flags were removed from the potency-row schema.
- The multi-reaction corpus retains exactly the same admitted and abstained products and is
  byte-identical after migration. LNPDB supplies rows, products and reported components. The
  reaction registry, paper reviews and supplements remain mandatory evidence for transform
  semantics, source conflicts, and admission; they do not add duplicate observations or labels.
  Exact atom-origin ledgers and balanced caches remain derived artifacts. The virtual Ugi corpus
  remains separately declared because its unmeasured products are absent from LNPDB.
- Checked potency artifacts live under `results/phase1/potency_study_corpus_v2/`. The observation
  SHA-256 is
  `7773f88c64e1793b9b6a5d188b8805d48d0542749554e2aba43af7c157d6886f`; the result SHA-256 is
  `573cd3e8f57bccfc3e7b12263750c77488ce05f8361233726848c9672957d810`. Registered run
  `c23a5cf0683d4930c1f90f365e3da5329acc6de0f2f5500843d85ab9902d8ba8` verified and reproduced
  both artifacts byte-for-byte.
- All 30 vendored assets verified. Result provenance verifies 822 pins with zero drift;
  code/config-inclusive verification accounts for 2,346 pins under the unchanged four-item drift
  ratchet. The scoped type, lint and 127-test gate passed. The full suite collected 1,735 tests and
  reproduced exactly the 201 reviewed failures caused by 54 unavailable historical inputs, with no
  new, resolved, or stale-cache nodes.
- **Decision:** LNPDB is the authoritative row-level source for new potency and LNPDB-backed
  reaction-program work. Evidence and derived training caches remain separate by function, not
  treated as independent datasets. Frozen M0 results and the completed raw-label HeLa diagnostic
  retain their original bytes. Cross-study target pooling remains prohibited, and this consolidation
  authorizes no biological guidance, candidate selection, synthesis call, or potency claim.

## 2026-08-21 - Replace the unrun L4-only preflight with matched accelerator qualification

- The frozen production design contained an L4 execution target inherited from prior deterministic
  CUDA qualification, but no measured L4-versus-A100/H100 cost-to-result comparison supported that
  choice for the larger four-arm workload. Hardware is an execution decision, not part of the
  scientific identity of the matched comparison.
- Replaced the unexecuted L4-only specification with two execution-only experiments sharing one
  hash-pinned contract: exact NVIDIA L4 and exact NVIDIA A100-40GB. Each runs the full 192-hidden,
  four-layer model for every arm at effective batch 128, includes one warm-up and three measured
  optimizer steps, repeats the conditioned challenger exactly, and performs a full-batch stress on
  the 194-heavy-atom boundary. GPU identity, physical memory class, at least ten-percent memory
  headroom, finite optimization, zero fixed-state changes and exact deterministic replay are
  fail-closed gates.
- Each result records per-step timing, examples per second, peak allocated and reserved memory,
  CUDA math-mode flags, physical device identity, and a training-only projection for all four arms
  and three production replicates. Modal GPU prices are frozen as execution metadata at their
  2026-08-21 published rates. The adjudicator requires both verified runs to have identical source,
  config, inputs, model, runtime, design, cache and math mode, then orders eligible targets by
  projected training-only GPU cost and wall time. It cannot select a model or molecule.
- Both Modal request plans validate. Each would upload the active private `forge/` and
  `experiments/` Python source, the 7.3 MB packed cache, frozen configs, smoke receipts, test-baseline
  receipt and lockfile. The attempted L4 launch was rejected by the external-action approval gate
  because this exact private-source upload and billable Modal destination require a separate,
  explicit user confirmation. No external benchmark ran and no cloud cost was incurred.
- All 30 present vendored assets verify. The 35 accelerator, experiment-model, architecture and CLI
  regression tests pass; scoped lint, formatting and typing pass. The full suite reproduces exactly
  the frozen 201-item missing-historical-input failure set with no accelerator-test failure, matching
  the checked-in no-new-failures baseline. Code/config-inclusive provenance verifies 2,354 pins under
  the unchanged four-item historical drift ratchet.
- **Decision:** the prior L4 choice is no longer treated as benchmark-justified. Production remains
  blocked until the user explicitly approves the disclosed Modal payload and costs, both bounded
  accelerator benchmarks pass, the selection receipt is frozen, and the selected exact accelerator
  passes the production preflight. H100 is not included because the deterministic float32 workload
  must first show that A100-40GB fails the cost/time tradeoff; adding another cost class without that
  evidence would not change the current decision.

## 2026-08-21 - Matched accelerator qualification selects L4; final confirmation remains blocked

- After disclosure of the private source/cache payload and a USD 4.10 combined cap, the user
  authorized the matched Modal L4 and A100-40GB benchmark pair. Both completed under executable
  source fingerprint `9bad187578b59c580493fa8d85cbf6662a650650672abfc0bea755373601fbd0`.
  Their run directories were downloaded directly from the committed Modal volume and independently
  passed the self-contained manifest verifier.
- Every fail-closed gate passed on physical `NVIDIA L4` and `NVIDIA A100-SXM4-40GB` devices: all four
  arms were finite, deterministic challenger replay was exact, fixed-state failures and route/oracle
  calls were zero, the full 194-atom support was exercised, the checked-in cache and smoke receipts
  matched, raw family-count sampling was absent, and the frozen test baseline had no new failures.
  Peak reserved memory was 660,602,880 bytes on both devices, far inside both memory envelopes.
- Under the prespecified training-only projection, three L4 replicates require 4,434.30 GPU-seconds
  and USD 0.9844, versus 4,792.79 GPU-seconds and USD 2.7942 on A100-40GB. These estimates exclude
  evaluation, startup, CPU, memory, transfer and storage. The bounded three-step timing result is an
  execution estimate only and provides no model-quality evidence.
- Verified L4 run `bb668b65570d5267aee0bb6bf0d6223c09cb8db7b442a96a9e3cb43b4957a481`
  produced result SHA-256 `4bfdbaad36091373c58aa70513612d4d3a125542abc6e02e43e2d4d2bf1b68d0`.
  Verified A100 run `88e51151d9c18f8896efd3fa4fc78ce2d8177cce19c6b7c2ec135bd4aa5eb368`
  produced result SHA-256 `8e3b19fba66c9163f55200b1705ab073adeb77dee40900b6d8f51a87bdad978a`.
  The selector chose L4 and wrote the pinned receipt with SHA-256
  `5d9d2cf14c4f74f12d6068763502c2f47898ba29cd3a34b6f5fde0d8e76fbfb8`. The receipt
  snapshots both verified benchmark results and manifests beneath its tracked evidence directory;
  it does not depend on ignored local run folders or continued access to the Modal volume.
- The external execution exposed three launcher defects, retained as negative engineering results:
  `uv_sync` received an image path instead of a local project path; the remotely imported entrypoint
  assumed a local file hierarchy; and volume paths were compared before resolving Modal's mount while
  the local client called a remote-only `reload` API. The benchmark computations were not used until
  their committed artifacts were downloaded and verified. The launcher now normalizes local/remote
  roots, resolves the mounted-volume boundary, avoids the illegal reload, and detaches scalar logging
  from autograd.
- All 30 vendored assets verify. The 40 focused accelerator, Modal, experiment-model, architecture and
  CLI tests pass; scoped lint, formatting, typing and diff-integrity checks pass. The full suite
  collected 1,744 tests and reproduced exactly the frozen 201 failures caused by 54 unavailable
  historical inputs, with zero new, resolved or stale failures. Code/config-inclusive provenance
  verifies 2,395 pins under the unchanged four-item historical drift ratchet.
- **Decision:** L4 is the benchmark-supported production accelerator; A100-40GB and H100 are not
  justified for this workload. The checked-in production DAG already requests exact L4. A separate
  confirmation on the active transport-fixed source was rejected because it is a third private upload
  and billable job beyond the approved pair. Do not bypass that control. No production replicate has
  launched; production remains blocked until the user explicitly authorizes and the final selected-L4
  confirmation passes.

## 2026-08-21 - Selected-L4 confirmation passes and closes the accelerator gate

- The user explicitly authorized one additional selected-target L4 preflight, the disclosed private
  payload and an additional USD 1.50 cap. Modal run
  `d97c3dd86b4c2383382a4ed91055135bda0235aaab2db6006d7c1a741c69a2a7` completed on physical
  `NVIDIA L4` under compute-source fingerprint
  `4106ce1d6f204b30be0e897ada92a2d510120658424344bb1a81d16e52bc910d`.
- All benchmark gates passed: every arm remained finite, deterministic replay was exact, fixed-state
  failures and route/oracle calls were zero, the 194-atom maximum-support batch executed, cache and
  smoke pins matched, raw family-count sampling was absent, and no candidate was selected. Peak
  reserved memory was 660,602,880 bytes. The short-run projection was 1,109.42 GPU-seconds and USD
  0.2463 per replicate, or USD 0.7389 GPU-only for three replicates; it excludes evaluation, startup,
  CPU, memory, transfer and storage and is not model-quality evidence.
- The downloaded run independently passed the self-contained verifier. Its result SHA-256 is
  `7df8f04de6357e9436df852df6ad6d2e45965f56eb53a06f2973a2e7b595f56e`; the tracked confirmation
  receipt SHA-256 is `7d26b80c01f34045d5bfa2e052828b0aefb992d68a51930774674176c0920748`.
  Run, stage and result JSON are frozen under
  `results/phase1/shared_synthesis_program_accelerator_selection_v1/selected_l4_preflight/`.
- Automatic local download initially failed after the verified GPU computation because the private
  staging directory's random basename violated the verifier's content-addressed run-name invariant.
  The artifact was recovered directly from the committed Modal volume and verified before use. The
  downloader now stages inside a private parent while preserving the run-id child basename; the new
  regression test passes. This post-execution local transport correction changes no model, data,
  training, gate or remote compute path and did not trigger another GPU run.
- All 30 vendored assets verify. The 41 focused accelerator, Modal, experiment-model, architecture and
  CLI tests pass; scoped formatting, lint, typing and diff-integrity checks pass. Code/config-inclusive
  provenance verifies 2,416 pins under the unchanged four-item historical drift ratchet. The full
  suite collected 1,745 tests and reproduced exactly the frozen 201 failures caused by 54 unavailable
  historical inputs, with zero new, resolved or stale failures.
- **Decision:** the selected exact-L4 accelerator gate is closed. The production DAG is input-ready
  and requests exact L4. This confirmation supplies execution evidence only: it launches no production
  replicate, establishes no model quality, and authorizes no candidate, route, oracle or biological
  selection.

## 2026-08-21 - Freeze paper execution evidence and implement cross-run production adjudication

- Added a deterministic cross-run adjudicator for the three authorized full production replicates.
  It independently verifies each run and its stage artifacts, requires one executable source, one
  experiment specification and one frozen scientific design, enforces replicates 0/1/2 and seeds
  20260825/20260826/20260827, and snapshots the authenticated run, stage, training, evaluation and
  sample evidence into the result package. It rejects any run that performed or failed to exclude
  route, biological-oracle or candidate-selection calls.
- Ugi retention is adjudicated at the fixed final checkpoint using the frozen 10,000-resample paired
  seed rule. Independent training seeds are the resampling unit; generated molecule rows are not
  promoted to independent training replicates. All frozen validity, connectedness, exact-L1,
  forward-precision, diversity, effective-component, component-novelty and held-component margins
  remain fail-closed. A noninferiority failure is preserved as a valid negative result.
- The same aggregate now implements the required catalogue-free versus post-hoc reaction-filtering
  comparison. It compares the shared program-conditioned arm against the compute-matched shared-null
  arm at equal heldout attempt budgets for Ugi, BL and LX, then reports raw validity, exact-L1 yield,
  coverage among valid products and forward-replay precision separately. This analysis uses complete
  generated graphs and exact adapters; component identifiers and fragment tokens remain absent.
- Froze `docs/PAPER_EXPERIMENT_EXECUTION_MATRIX.md` as the paper-facing separation among computed,
  ready, proposed and prospective evidence. The existing nonzero synthesis-guidance and HeLa
  potency-tilting diagnostics remain closed negative results. Held-reaction-family generalization
  remains a secondary, unexecuted stress test rather than a paper gate. The broad ML-baseline suite
  is not promoted to a pre-prospective blocker under the authoritative strategic reset.
- The paper's production-model novelty denominator remains all 112,386 fitted products: 21,960 of
  26,235 distinct admitted products, or 83.7%, are absent. The 89.0% train-fold statistic is not
  substituted for this production-model claim.
- The prospective analysis plan requires
  `results/phase1/ugi_prediction_cohort_panel_v3/candidate_manifest.json`, which is absent from the
  current repository. The available 40-candidate v6 panel is not the locked 30-candidate v3 panel
  and cannot substitute for it. Prospective execution therefore remains fail-closed pending exact
  artifact recovery, current procurement evidence, human chemistry signoff and PI endpoint/design
  decisions. No candidate was regenerated, replaced or selected.
- **Decision:** finish local verification, then rerun the selected L4 preflight on the final source
  snapshot. Launch the three full production replicates only if every preflight gate passes. Do not
  modify executable source between that preflight and the three launches.

## 2026-08-21 - Correct the paper scope to computational only

- The user explicitly corrected the execution plan: the current paper does not include synthesis,
  formulation, in-vitro experiments or in-vivo experiments. Earlier prospective candidate and
  wet-lab plans are historical records, not paper requirements, blockers or next steps.
- The missing 30-candidate v3 manifest and the available historical 40-candidate v6 panel are
  therefore irrelevant to current paper readiness. Do not recover, substitute, relock, procure or
  experimentally advance either panel as part of this paper.
- Biological endpoint selection, assay design, formulation controls, animal design, procurement and
  laboratory capacity are outside current scope. Existing biological and synthesis-guidance
  diagnostics remain bounded computational evidence with their frozen nonclaims.
- **Decision:** paper completion requires the authorized computational production comparison,
  sampling and held-component analyses, catalogue-free versus post-hoc ablation, aggregate
  adjudication and manuscript integration. No wet-lab experiment may be added to the readiness list.

## 2026-08-21 - Add a strong finite-component catalogue baseline

- The user explicitly authorized a direct finite-component-vocabulary comparison for the
  multi-reaction paper experiment. The selected baseline is an oracle catalogue assembler rather
  than a weak learned fragment model: it receives exact train-fold component identities and the
  exact qualified forward transform for Ugi, BL and LX.
- Within each reaction family, it samples components independently from source-weighted role
  marginals. Ugi uses `family_balance_weight_raw`; BL and LX use `source_balanced_weight`. It never
  samples by raw family counts, never accesses calibration or heldout components, makes zero route
  or biological-oracle calls, performs no repair or retry, and emits at most one uniformly selected
  exact forward product per tuple attempt.
- The full arm uses the same 3,072 attempts per family and seeds 20260825, 20260826 and 20260827 as
  final-checkpoint FORGE evaluation. It is a CPU experiment with deterministic samples and pinned
  input ledgers. The aggregate adjudicator accepts the three catalogue receipts only when source,
  design, seeds and replicate identities exactly match the production runs.
- **Decision:** the primary differentiating metric is distinct valid, uniquely decomposed exact-L1
  products containing at least one component constitution absent from the train-fold catalogue per
  1,000 attempts. Report raw validity, exact-L1 yield, unique exact-L1 product yield, whole-product
  novelty, internal diversity and effective component count separately. The catalogue value for
  component escape is zero by construction and is not sufficient by itself; the paper must expose
  the catalogue's expected reaction-validity advantage and FORGE's intended breadth advantage
  without a composite winner score. The Cartesian tuple-space size is an upper bound, not a count
  of unique valid products.
- Catalogue coverage is reported directly for train, calibration and held-component source folds,
  with unique source-component coverage by role. The maximum attainable *component* novelty of the
  baseline is exactly zero. This is kept separate from empirical whole-product novelty because a
  finite catalogue can still make unseen products by recombining familiar components.
- Every descriptive FORGE-minus-catalogue contrast receives a two-sided paired-seed bootstrap
  interval using the frozen 10,000-resample settings. Molecules are not pseudoreplicates.

## 2026-08-21 - Refresh the production no-new-failures receipt before the final source freeze

- A clean-cache full-suite run collected 1,747 nodes and reproduced exactly the reviewed 201
  failures caused by the unchanged 54 missing historical artifacts. It introduced zero new
  failures, resolved zero reviewed failures and left no stale cache nodes.
- Refreshed the rolling maintenance receipt and its two mutable input identities: the current
  repository contract in `AGENTS.md` and the current test-baseline report. This changes no
  scientific gate, historical failure identity, chemistry evidence or model result.
- The four-item code/config drift burn-down remains unchanged and visible. No new known-drift
  exception was added and no scientific gate was relaxed.
- **Decision:** freeze the executable source, rerun the selected-target L4 preflight on that exact
  fingerprint, and make no executable-source edits between the passing preflight and the six matched
  production/catalogue launches.

## 2026-08-21 - Repair CUDA RNG restoration exposed by an interrupted production launch

- The first three attached Modal production clients encountered a local DNS interruption and
  canceled their remote calls. Resuming the resulting partial training stages then failed
  identically in all three replicates: `torch.load(..., map_location="cuda")` had moved serialized
  CUDA RNG-state tensors onto the accelerator, while `torch.cuda.set_rng_state_all` requires CPU
  byte tensors.
- The restart primitive now validates every RNG state as a one-dimensional byte tensor and moves
  RNG bookkeeping tensors back to CPU before calling the PyTorch setters. Focused CPU regression
  tests cover the CUDA-map-location normalization and fail-closed dtype validation. This changes no
  training seed, optimizer state, checkpoint schedule, model architecture or scientific gate.
- A clean-cache full suite collected 1,749 nodes and reproduced exactly the same 201 reviewed
  failures from the same 54 missing historical artifacts, with zero new failures. Readiness v3
  records that observation.
- The repaired executable source fingerprint is
  `6fd98fa6717e149de631c551c19b5b571bef36c6bd4930f51f65be68f3445a86`.
  Its fresh NVIDIA L4 preflight run
  `e4b5bfa58822c97734a972988d92f84a0194699be31b38db72e636033ad042cd`
  passed all 20 fail-closed gates, including exact deterministic replay and the 194-atom
  maximum-support stress.
- **Decision:** preserve the canceled attempts as negative operational evidence, use detached Modal
  clients for production durability, and rerun both catalogue and GPU replicates on the repaired
  source so aggregate adjudication never mixes executable fingerprints.

## 2026-08-23 - Preserve the shared multi-reaction production comparison as a negative result

- The three GPU production replicates and three matched CPU catalogue replicates completed on one
  frozen executable source, specification, design and seed set. The aggregate now handles
  zero-survivor metrics without coercing undefined effective-component counts, component novelty
  or forward precision to zero. Undefined cells are never imputed or complete-case bootstrapped.
- The frozen Ugi non-inferiority decision failed. Shared training failed raw validity,
  connectedness, exact-L1 coverage, effective component count and held-component exact-L1
  coverage; it retained forward precision, diversity and component novelty among survivors. The
  conditioned exact-L1 yield per attempt was 0.03147 for Ugi, 0 for BL and 0.14659 for LX, versus
  0.14410, 0.03451 and 0.12956 under matched post-hoc filtering, respectively.
- The finite catalogue attained much higher exact-L1 yield but zero component escape by
  construction. FORGE produced 31.47 and 33.42 open-ended exact-L1 products per 1,000 attempts for
  Ugi and LX, respectively, but zero for BL. This is a validity--open-endedness tradeoff, not a
  composite win.
- A training/calibration-only diagnosis found zero BL exact-L1 samples at every checkpoint in all
  seeds and a decline in mean shared-model Ugi calibration yield from 0.10156 at step 100 to
  0.02669 at the fixed final step. This localizes failure before heldout evaluation but does not
  distinguish optimization interference from insufficient program-specific capacity.
- **Decision:** the completed primary comparison remains a valid negative result and must not be
  replaced by checkpoint selection or same-data retuning. Any program-specific adapter or routing
  head is a newly frozen follow-up experiment. Manuscript text and figures must not claim shared
  multi-reaction success from this run.

## 2026-08-23 - Implement and audit a reaction-program cross-attentive graph Transformer follow-up

- The user explicitly authorized a new architecture after the frozen MPNN comparison failed. The
  completed negative comparison remains unchanged and is not replaced by this follow-up.
- The new denoiser retains the complete sparse whole-product state, bounded 194-atom/three-closure
  support, component-free tensors and adapter-fixed chemistry. It replaces four local
  message-passing blocks with full graph self-attention carrying parent/child/closure relation
  biases. Every block separately cross-attends to program, depth, precursor-role and reaction-core
  memory tokens.
- Every block also contains three softly program-routed residual adapters. The objective adds
  equal-present-state precursor-role and reaction-core consistency losses to the existing atom,
  pointer and bond flow loss. Training computes one loss per source program, equalizes raw gradient
  norms and applies deterministic PCGrad before combining family gradients.
- Initial run `4c08f2a818ecc1d5a0f29578fc04903e12838bc40d36dc6303d6edb721e0b830`
  passed the first implementation gate, but the subsequent code audit found that atom-aligned
  role/core memory tokens lacked positional keys. That run remains reproducible historical evidence
  but is superseded for architecture qualification because cross-attention could only observe a
  semantic multiset rather than the structural slot associated with each coordinate.
- Positional-memory run `1e6fece41895341faa9dfb2cbed5de7fe7372c0de711fada5644162d28df7a91`
  is a frozen negative result. At the predeclared 1,200 steps it exactly reconstructed the BL and LX
  records, but not Ugi: one Ugi parent pointer remained wrong, giving 0.95556 parent accuracy and
  zero exact-record reconstruction. Total loss still fell from 18.8357 to 3.66166e-4, and all other
  five gates passed. The receipt is
  `results/phase1/reaction_program_transformer_overfit_positional_v1/result.json`.
- **Decision:** do not advance the 1,200-step positional model to accelerator preflight and do not
  weaken the exact gate. Run one separately pinned 2,400-step optimization follow-up with identical
  architecture, data, seed and acceptance criteria. Neither bounded attempt is evidence of BL/Ugi
  generalization, and the failed MPNN production result remains the immutable baseline.

## 2026-08-23 - Remove unstable reaction-family gradient-norm amplification

- Positional-memory follow-up run
  `fd752a10d39dea34340769fecfd3d6682b05757b7abe2213d0510104e1d40a5f` is a frozen negative
  result. Merely extending the original norm-equalized optimizer to 2,400 steps destabilized the
  bounded problem: the final raw family gradient norms diverged to 1,608.27, 0.12355 and 2,524.72,
  and none of the three records reconstructed exactly. The receipt is
  `results/phase1/reaction_program_transformer_overfit_positional_v2/result.json`.
- The failure localized the defect to per-family norm equalization. At step 1,200 the prior run had
  raw norms of 0.00126, 8.00e-9 and 0.85778, so rescaling every family to their mean amplified the
  nearly converged family's numerical gradient by tens of millions. This is not a safe balancing
  rule near an optimum.
- The corrected rule retains exactly stratified family sampling, gives each family equal loss mass,
  and applies deterministic PCGrad conflict projection, but never amplifies gradients to a common
  norm. A zero-gradient family therefore contributes zero instead of numerical noise.
- Corrected run `e299bd07e494e8d2b77651398fbd5b895051dcf21e45f1e9bc790106d2cce3ae`
  passed all six unchanged gates and reproduced both stages byte for byte. Total loss fell from
  18.8357 to 2.83678e-5; Ugi, BL and LX each reached exact tensor reconstruction, all fixed states
  remained exact, and final raw family norms were 0.00179, 0.000344 and 0.000368.
- **Decision:** qualify the corrected architecture and optimizer for a separately frozen accelerator
  preflight only. The receipt is
  `results/phase1/reaction_program_transformer_overfit_v1/result.json`. This bounded memorization
  result is not evidence that held-component or multi-reaction generalization is fixed; production
  evaluation must retain the original split and failed MPNN baseline unchanged.

## 2026-08-23 - Freeze and smoke-test the Transformer four-arm production follow-up

- Frozen design run `8a9886a192652752dc5f8d34b1210254da7748e5e999f7fb29d7bc63d49081c6`
  passed all 13 design and prerequisite gates. It retains the original Ugi-only, shared-conditioned,
  null-program and cyclic-program control arms; component-disjoint folds; fixed-final checkpoint;
  exact-L1 evaluation; three replicate seeds; and 194-atom/three-closure support. The only model
  change is the separately qualified six-layer, eight-head reaction-program graph Transformer with
  three routed experts and stable equal-family-mass PCGrad.
- The first CPU smoke attempt completed training but exposed one architecture-specific sampler
  access to the old MPNN `.backbone`. No evaluation result was published. Both model classes now
  expose the same immutable support attributes, and sampling no longer inspects implementation
  internals.
- Corrected smoke run `f24ba6894a07ad552783ffe98dfcfdfeecadfa8afef5dec25a1f032a4b26c80d`
  passed all eight training and 13 evaluation gates across all four arms. Fixed-state failures,
  support overflows, route calls, oracle calls and candidate selections were all zero. The full
  repository run collected 1,760 tests and reproduced exactly the 201 reviewed failures caused by
  54 absent historical artifacts, with zero new failure nodes and zero stale cache nodes.
- L4 and A100-40GB preflight specifications are now hash-pinned to the Transformer design, smoke
  receipts and current regression receipt. They are execution-only capacity/determinism checks and
  have not been launched in this change.
- **Decision:** the Transformer follow-up is ready for its bounded accelerator preflight, but full
  production remains blocked until one exact GPU class passes. Do not reuse the earlier MPNN
  accelerator receipt, and do not interpret smoke success as held-component generalization.

## 2026-08-23 - Optimize the Transformer training hot path without changing family weighting

- A production-shaped CPU profile localized less than 1% of step time to cache sampling, record
  loading and collation. The dominant cost was the three-family PCGrad backward, followed by dense
  graph attention. The prior PCGrad implementation also materialized many CUDA scalars as Python
  values inside each micro-batch, which would serialize accelerator execution.
- The optimized path compiles stratified family indices and weights once per arm, vectorizes empty-
  safe graph and semantic losses, collects metrics and fixed-state checks once per optimizer step,
  constructs graph relations by indexed assignment instead of dense one-hot tensors, and flattens
  PCGrad projection. A no-conflict fast path computes the exact equal-family gradient mean after one
  pairwise conflict check; the sequential deterministic projection remains unchanged when a
  conflict exists. Raw `reaction_family` counts are never used and the sampler reproduces the prior
  five-batch RNG sequence exactly.
- Reproducible benchmark
  `results/phase1/transformer_training_optimization_v1/result.json` used the 5,289,517-parameter
  production Transformer, batch size 12 and 158 padded atoms. Median forward + loss + PCGrad time
  fell from 0.408209 s to 0.385620 s (1.05858x). Graph-bias construction was 1.69492x faster and the
  PCGrad backward was 1.08293x faster. Graph biases and PCGrad gradients were bit-identical; the
  maximum full-objective loss difference was 1.90735e-6 and passed the frozen `rtol=1e-6`,
  `atol=1e-7` numerical-equivalence policy.
- Fused scaled-dot-product attention was evaluated but not retained: it was neutral in the complete
  local path and slightly slower in an isolated CPU training-backward measurement, while a current
  production-GPU comparison could not be run without separate external upload/cost approval. The
  qualified explicit attention implementation therefore remains in production.
- Current-source overfit run
  `d5cd9d8705e8e02694f3f119ab0ce321e1757f200fffb1d45be6626dcc1264eb` passed all six gates,
  including exact Ugi/BL/LX tensor reconstruction. Four-arm smoke run
  `d3640f379cd993487637ce50525967277e505c9df200c7a331d653300ea8ff61` passed every training
  and evaluation gate with zero fixed-state failures. All 30 vendored assets verified. The full
  suite reproduced the same 201 reviewed failures from absent historical artifacts and introduced
  no Transformer-related failure node.
- **Decision:** retain the measured sampler, loss, graph-bias and PCGrad optimizations. Production
  launch remains blocked on a separately approved current-source L4 preflight; the CPU benchmark is
  not evidence of CUDA throughput or model quality.

## 2026-08-23 - Current-source Transformer L4 preflight passes

- With explicit user approval for the bounded external run, Modal execution
  `2ca97d586bce5c00f3960ba7fca7935b95f0569b974c6a52d00b53b269afd5b9` evaluated source
  `15765d81e863e14e3c71d95720fd42fe222cb62acea7d1d1ec33ec1562390d8a` on an NVIDIA L4
  under deterministic float32. The downloaded run and its single result artifact verified against
  the remote manifests.
- All 23 fail-closed gates passed. The benchmark exercised all four frozen arms, all 194 supported
  atoms and three observed closures, the complete factorized layout schedule and the full cache fold
  counts. Fixed-state failures, route calls, oracle calls and candidate selections were all zero;
  raw-family-count sampling remained absent.
- The peak reserved allocation was 3,772,776,448 bytes of 23,659,151,360 bytes total, leaving 84.1%
  measured memory headroom. The maximum-194-atom stress step completed in 1.43652 s. Median measured
  optimizer-step time was 1.06533 s for the shared conditioned arm, 1.05598 s for the null control,
  1.07385 s for the cyclic-program control and 0.469181 s for Ugi-only. The exact conditioned repeat
  reproduced model-state hash
  `6b48d5c4130a7dc47523fa6bf8bb60f685b8fe5fcd13d3a8a7c7fc6584b83b56`.
- The frozen training-only projection is 6,229.39 s (1.73039 GPU-hours) and $1.38292 per replicate,
  or 18,688.16 s and $4.14877 for all three replicates at the pinned 2026-08-21 L4 rate. Evaluation,
  startup, CPU, memory, transfer and storage are excluded exactly as declared by the config.
- **Decision:** the exact L4 class is qualified for the separately frozen three-replicate Transformer
  production comparison. This preflight establishes capacity, throughput and deterministic replay,
  not model quality. Do not launch the production replicates without a separate explicit execution
  instruction.

## 2026-08-23 - Expand the manuscript baseline and ablation reporting contract

- The user explicitly requested manuscript placeholders for the complete comparison suite rather
  than only the previously implemented production arms. The common Ugi contract now reserves rows
  for the ionizable-lipid DAG generator, SynFlowNet, RGFN, SynCoGen, unconditional DeFoG,
  GenMol/SAFE, the finite catalogue oracle, a learned inventory selector, shared-null post-hoc
  assessment, FACT-matched and FACT-generous.
- The architecture table separately reserves one-at-a-time interventions for layerwise reaction-
  program cross-attention, precursor-role consistency loss, reaction-core consistency loss,
  program-routed adapters and PCGrad. Ugi-only, shared-conditioned, shared-null and cyclic-program
  controls remain in the multi-reaction production table.
- The historical fixed-scaffold model is scientific lineage rather than an independent comparator.
  The failed sparse MPNN is an internal development predecessor rather than a mechanistic ablation.
  Neither appears as a manuscript baseline; their frozen records remain in repository history.
- **Decision:** these rows are explicit TBD reporting obligations only. They do not convert an
  unexecuted comparison into evidence, silently authorize implementation or launch, or make every
  external method a blocker for reporting already frozen negative and positive results.

## 2026-08-23 - Audit and freeze the v1 paper experiment-readiness matrix

- The current-source Transformer four-arm experiment and finite-catalogue oracle both pass their
  registered input doctors. The exact executable fingerprint remains
  `15765d81e863e14e3c71d95720fd42fe222cb62acea7d1d1ec33ec1562390d8a`, matching the
  NVIDIA L4 preflight that passed all 23 gates. No full Transformer replicate exists on this source.
  Existing full catalogue runs have older executable identities and cannot enter the matched
  Transformer aggregate, so three current-source CPU runs are still required.
- The paired production adjudicator is implemented. The common external-baseline importer and
  assessor, conditional cross-role diagnostic, FACT-matched/generous models, learned inventory
  selector, five Transformer mechanism ablations, six published-method adapters, method-neutral
  route-evidence aggregate, and v1 result renderer are not implemented. Manuscript placeholders for
  those methods are therefore not launch-ready results.
- Froze one common Ugi baseline protocol: train fold only; seeds 20260825--27; 3,072 attempts per
  seed; failed and invalid generations retained in the denominator; no repair/retry, route calls,
  biological-oracle calls or candidate selection; and one constitutional exact-L1 assessor that
  reports both decomposition coverage and forward-replay precision.
- Added a machine-readable matrix and `forge paper experiments` diagnosis. It separately reports
  immediate production readiness, setup completeness for all retained manuscript rows and final
  result completeness. Optional leave-one-reaction-family-out remains secondary and never becomes a
  hard gate.
- **Decision:** preserve the passing production source freeze, run the three Transformer and three
  matching catalogue replicates, and aggregate them before changing executable experiment code for
  the remaining baselines and ablations. Every later adapter must receive its own smoke, provenance
  and capacity gate. External rows must be run on the common split or remain TBD; published values
  from other datasets cannot be substituted.

## 2026-08-23 - Implement the retained computational baselines, ablations and v1 renderer

- The user explicitly superseded the previous execution order by instructing implementation of the
  conditional-dependence diagnostic, FACT controls, learned inventory selector, five Transformer
  ablations, six external-baseline interfaces, common synthesis-evidence assessment, v1 result
  renderer and held-reaction-family study. This intentionally invalidates the earlier source
  fingerprint and its L4 preflight; no production result from that receipt will be reused.
- Added a common Ugi attempt ledger that retains invalid and failed native attempts, call totals and
  method-visible components. One registry-backed assessor now computes constitutional validity,
  exact-L1 decomposition coverage, forward-replay precision, ambiguity, held-component yield and
  method-relative open-endedness. The finite catalogue and Transformer evaluators export canonical
  Ugi attempt ledgers, and the learned selector emits and assesses its ledger directly.
- Implemented the source-balanced autoregressive train-inventory selector. It uses only train-fold
  component identities and `family_balance_weight_raw`; out-of-inventory escape is zero by
  construction. Its CPU smoke run passed the exact denominator, zero-call, no-selection and
  coverage/precision gates.
- Implemented one paired Transformer study containing the full model, the five one-at-a-time
  mechanism ablations, FACT-matched and FACT-generous. FACT masks role-to-role messages in both the
  graph layers and program-token encoder; an intervention test verifies that changing other-role
  atom/core states cannot change target-role predictions. The matched arm has exactly the full
  model's parameter count and the generous arm must be strictly larger. The CPU smoke completed all
  eight arms, checkpoints, heldout losses, sampling, molecule reports and zero-call gates.
- Implemented three leave-one-reaction-family-out arms. Their results are marked secondary and
  `hard_gate=false`. The CPU smoke completed for all three held programs without weakening any
  production gate.
- Implemented the within-count-context role-shuffling diagnostic and generated-sample residual
  cross-role fidelity metric. The first deterministic diagnostic execution produced mean AUC
  0.48845, failed the 0.60 detection threshold and had component-family overlap between tuple-group
  folds. Therefore no cross-role-dependence or joint-generation advantage claim is authorized from
  that execution. Rerun it once on the final source; if the negative result persists, remove the
  corresponding claim rather than changing the threshold.
- Pinned upstream commits and legal status for Ou DAG+Chem, SynFlowNet, RGFN, SynCoGen, DeFoG and
  GenMol/SAFE; implemented the common train/split/reaction export, clean-checkout verifier, strict
  native receipt importer and failed-attempt preservation. This is not falsely called six runnable
  native ports: Ou lacks released paper-specific code, SynCoGen lacks a usable license, and the four
  other upstream training ports still require implementation and qualification.
- Added lookup-only common route assessment. It promotes only exact role/constitution evidence and
  treats every index miss as an abstention. The current frozen index is the bounded FORGE-candidate
  cascade, not an exhaustive method-blind union; its coverage cannot support a cross-method
  synthesis-success claim until that broader evidence scope is separately frozen.
- Added `forge paper collect-results-v1` and `forge paper render-results-v1`. The collector extracts
  normalized rows from hash-pinned source results without manual transcription. The renderer uses
  the three training seeds as the independent units, applies the frozen 10,000-resample bootstrap,
  emits LaTeX row fragments and figure CSVs, renders undefined metrics as N/R rather than zero, and
  refuses strict publication output unless every retained method, mechanism and held-family row has
  the exact paired seed set. A partial smoke preview completed successfully and is explicitly marked
  non-publication.
- **Decision:** internal implementations are ready for a final repository verification and fresh
  L4 preflight. Full production remains unlaunched. External empirical rows remain blocked until
  qualified native ports or author/licensing resolution exists, and the common L2/L3 table remains
  descriptive evidence-index coverage rather than synthesis-success comparison.

## 2026-08-23 - Close the baseline and ablation implementation milestone

- Froze executable source fingerprint
  `d1d13f12d3b6cd7276b6d52c78730559df28d8ed97de4f4ca826392d92fdc762` after the
  common assessor, learned selector, Transformer/FACT study, held-family study, external contract and
  result renderer were complete. Current-source smoke run IDs are recorded in
  `results/phase1/paper_experiment_implementation_v1/result.json`; every listed result reproduced
  byte-for-byte.
- Verified all 30 vendored assets, passed active-source lint, passed four mypy groups covering
  17/65/6/10 source files and passed the focused benchmark, external-adapter, renderer, Transformer
  and architecture-boundary tests. The full historical test baseline remains blocked at 201 known
  failures from 54 documented missing historical inputs, with zero new failures.
- The provenance verifier reports seven drift declarations from old frozen receipts: three old
  catalogue aggregates plus the earlier readiness matrix/paper, layout benchmark and maintenance
  test-baseline receipt. These are expected invalidations caused by the authorized implementation,
  not accepted drift. Their historical bytes and expected hashes remain unchanged; fresh
  current-source runs must supersede them before production readiness can be green.
- The final diagnostic measured mean conditional-role discrimination AUC 0.48845 against the frozen
  0.60 threshold and its tuple-group folds were not component-family disjoint. This is a completed
  negative result: no cross-role-dependence or joint-generation advantage claim is authorized, and
  the gate will not be retuned.
- The six external rows have a common split/export/receipt/import/assessment interface, but they are
  not falsely declared runnable native ports. Ou DAG+Chem lacks the paper-specific author code,
  SynCoGen requires licensing resolution, and SynFlowNet, RGFN, DeFoG and GenMol/SAFE require
  qualified native ports. Published values from other datasets remain inadmissible substitutes.
- The common route evaluator remains lookup-only and abstains on missing evidence. Its current index
  covers the bounded FORGE cascade rather than a method-blind union, so cross-method synthesis-success
  ranking is not authorized.
- **Decision:** the authorized internal implementation milestone is complete. Before any full GPU
  production run, rerun the exact NVIDIA L4 preflight on this source. The entire paper experiment
  suite is not yet setup-complete because external native runs and a method-blind union evidence
  scope remain unresolved; strict rendering must stay red until real paired-seed rows exist.

## 2026-08-23 - Admit runnable external baselines and method-blind route evidence

- Audited the exact pinned upstream releases rather than treating a repository URL as a runnable
  paper baseline. Retain three empirical methods under the common Ugi protocol: RGFN at
  `6ce59169f855ed18f34ba4e8279de93bee306e4f`, unconditional DeFoG at
  `365bda9affadd5c2307014a0532ddaa244399441`, and GenMol/SAFE at
  `add09fc83b7255bd09c797e527c0f4b51f5fb7c1`.
- Implemented clean-checkout, fixed-final native runners for all three retained methods. RGFN uses
  the exact registry-derived three-reactant Ugi transform and a constant non-property reward;
  DeFoG recomputes the custom 194-heavy-atom data support and directly decodes raw graphs; GenMol
  trains from scratch with the pinned SAFE tokenizer and strict decoding. Every adapter retains one
  row per requested attempt and makes zero route, biological-oracle or candidate-selection calls.
  Repair, retry and substitution of original-paper numbers are forbidden.
- Exclude Ou DAG+Chem from the empirical table because the released repository omits the
  paper-specific author implementation; exclude SynFlowNet because the released action space
  supports at most two reactants and cannot represent the qualified AGILE three-reactant step;
  exclude SynCoGen because its repository has no usable code license. Retain all three as cited
  methodological context. A sequential pseudo-reaction is not an admissible SynFlowNet port.
- Implemented a method-blind route-evidence union. It freezes the exact role/constitution union from
  all assessed attempt ledgers, emits an opaque public worklist without method identity, seals
  method membership separately, and requires one complete/unresolved/search-censored disposition
  for every union component before per-method closure is computed. Missing evidence remains an
  abstention. This supports a common route-evidence-closure comparison, not a synthesis-success
  probability.
- Tightened the v1 result collector and renderer to require
  `route_scope=method_blind_cross_method_union` for strict common-method rows. The earlier bounded
  FORGE-only route index cannot enter the cross-method table. Removed numerical placeholders for
  the three excluded external methods while retaining the citations and exclusion reasons.
- Verified deterministic smoke request preparation against clean exact checkouts for RGFN, DeFoG
  and GenMol. Full native training has not run. `forge paper experiments --strict` now reports zero
  setup blockers and `setup_complete_for_all_retained_rows=true`; that is readiness, not a result.
- Reproduced the repository test baseline exactly: 1,778 collected nodes, the same 201 known
  failures caused by 54 documented absent historical inputs, and zero new failures. All 30 vendored
  assets verify; focused baseline/union tests, lint and strict typing pass.
- The earlier frozen receipts remain immutable. Eleven unrecoverable intermediate identities were
  never committed or archived; exact reviewed path/digest exceptions now document them, while
  `results/phase1/external_baseline_route_union_readiness_v1/result.json` pins the current source and
  supersedes the old readiness conclusion. The provenance gate reports zero unreviewed drift.
- The fresh NVIDIA L4 run for source
  `085e063a5c87d4fcc0f3a2a343bde6118617355dba7de5ee22f054555267327e` is planned as run
  `46b2e06ea31aca29371a2b64f6289f14306804f2db443c2cf9211f18886cc545` but was not launched because
  external paid-compute approval for this newly changed private source was not granted. No older
  L4 receipt qualifies this source.
- **Decision:** the retained experiment suite is implementation-complete and runnable. Production
  remains fail-closed until the exact-source L4 preflight passes, after which the three paired-seed
  internal studies and the three retained native baseline studies may launch. The method-blind
  evidence union is frozen only after all resulting assessed ledgers exist.

## 2026-08-23 - Qualify the final experiment source on NVIDIA L4

- With explicit user approval, launched the fresh Modal NVIDIA L4 preflight for executable-source
  digest `085e063a5c87d4fcc0f3a2a343bde6118617355dba7de5ee22f054555267327e`.
- Run `9939ea913b54d9462213a7a8488436d1b12a1f7c2b11a3a30d3ea4385c25b7f5` completed and its
  downloaded artifact independently verifies against the remote run and stage manifests. The
  result digest is `9e2618f70db4e7bc099a76b139475434ff0391b4a1793ff0292773ad0f2e636a`.
- All 23 prespecified gates pass, including exact deterministic replay, all-four-arm finite loss,
  fixed-state preservation, full 194-heavy-atom support, factorized-layout closure, at least 10%
  memory headroom, exact frozen training/evaluation budgets, absence of raw-family-count sampling,
  and zero route, oracle and candidate-selection calls.
- The maximum observed reserved GPU memory was 3,772,776,448 bytes on a 23,659,151,360-byte NVIDIA
  L4. The projected four-arm training time is 1.7211 hours per replicate and the projected L4 GPU
  line-item cost is USD 1.3755 per replicate, excluding CPU, memory, evaluation, transfer and
  storage.
- Repinned the paper experiment matrix to this exact source, run and result. The strict readiness
  audit now reports `source_freeze.intact=true`, `immediate_production_ready=true`, zero blockers and
  complete setup for every retained numerical row.
- **Decision:** authorize launch of the three paired-seed Transformer production replicates and
  their matched catalogue, learned-inventory, mechanism/FACT, held-family and retained native
  baseline runs without further executable-source changes. Full paper results remain pending; the
  preflight establishes capacity and determinism, not model quality.

## 2026-08-23 - Cancel L4 production and prepare matched A100/H100 execution

- At the user's explicit instruction, stopped all four active Modal L4 applications associated with
  the production launch: `ap-hf20VA8gq3jwNqZYL4v43E`, `ap-xjevkBQAnL2AB3BAheYoMk`,
  `ap-4vW8MBgU3IRChCYhxQsorf` and the orphaned `ap-9uVDT71vCdVayJmqH3hbz2`. No completed L4
  production result is admitted from those cancelled applications.
- Preserved the scientific design exactly and added two execution alternatives: NVIDIA A100-40GB
  and exact NVIDIA H100. The H100 request uses provider resource `H100!` to prohibit silent H200
  substitution. Both alternatives keep float32 strict determinism, the same four arms, seeds,
  effective batch, 1,700 optimizer steps and evaluation budget.
- Froze a matched A100/H100 benchmark policy that selects the lowest projected training wall time,
  with projected GPU line-item cost as the tie-breaker. The benchmark changes no model-quality
  metric and makes zero route, oracle or candidate-selection calls.
- Corrected the production adjudicator to accept only the current Transformer production IDs and
  to reject mixed experiment identities across replicates; the retired sparse-MPNN ID can no longer
  enter the Transformer aggregate.
- All 30 vendored inputs verify and the focused accelerator, aggregate and Modal-plan tests pass.
  The previous L4 source freeze is intentionally invalidated by these executable changes.
- Paid A100/H100 execution was not started because the external-action approval boundary requires a
  new explicit authorization to upload the current private source and incur both benchmark charges.
- **Decision:** production remains fail-closed. After that explicit approval, run the two short
  benchmarks in parallel, adjudicate the measured wall-time winner, repin the source freeze and
  launch only the selected GPU's three production replicates.

## 2026-08-23 - Qualify exact H100 and reject the substituted A100 allocation

- With explicit user approval, launched the A100-40GB and exact H100 Modal benchmarks in parallel
  on executable-source digest
  `668fb8d431bc48da2b63992c75cc3f58b666d797b57117da0ba5c8d0765906c5`.
- H100 application `ap-9IzUnIIFqkVAZrsgmjTOn4` completed as run
  `9327c755b39b1217e0f838f33dc346a1b26e50b641cf446262f6059fb890b3e1`. Its downloaded run
  independently verifies and result digest
  `0cc81c7324984a00ebb93a23ea606faa239d04a2d277064dfe5e95b4c983080b` passes all 23 gates on an
  `NVIDIA H100 80GB HBM3`.
- The H100 projection is 0.77393 training hours and USD 3.0564 GPU line-item cost per replicate,
  excluding evaluation, startup, CPU, memory, transfer and storage. Peak reserved memory is
  3,772,776,448 bytes out of 85,017,624,576 bytes.
- A100 application `ap-Yvz24qmaxlIyxzQT4bSBCl` requested `A100-40GB`, but the provider allocated
  `NVIDIA A100-SXM4-80GB`. The immutable physical-identity gate therefore failed while the other 22
  gates passed. The raw failed result and failure receipt are preserved under
  `results/phase1/transformer_a100_40gb_accelerator_failure_v1/`; the gate was not relaxed.
- The substituted A100-80GB projected 1.34966 hours per replicate, so the qualified H100 was also
  42.7% faster than the observed A100 allocation. This throughput comparison is execution evidence,
  not a model-quality result.
- Repinned the paper matrix to the exact H100 run and switched its three production commands to the
  predeclared H100 experiment specification. This changes no executable source, model setting,
  seed, data input or evaluation budget.
- **Decision:** exact H100 is the sole qualified production target. The benchmark authorization does
  not itself launch the three full paid production replicates.

## 2026-08-23 - Launch three exact-H100 Transformer production replicates

- With explicit user approval, launched full-profile replicates 0, 1 and 2 concurrently from the
  predeclared `phase1-transformer-synthesis-program-production-h100` specification on exact
  `H100!` resources.
- The Modal application identities are `ap-Cmc93u1lO2LqRk6j4TEJd5` for replicate 0,
  `ap-sxbvd0L4yBZPPBRMjwQj8h` for replicate 1 and `ap-rdlp7Oz879EhLXsFc1pose` for replicate 2.
- All three launch plans are bound to executable-source digest
  `668fb8d431bc48da2b63992c75cc3f58b666d797b57117da0ba5c8d0765906c5`; the H100 source freeze
  was intact immediately before launch. No model, data, seed, precision, arm or evaluation setting
  changed after preflight.
- **Decision:** the applications are running. No production result or scientific conclusion is
  admitted until each downloaded run independently verifies and the three-replicate adjudicator
  completes.

## 2026-08-24 - Freeze the paired reaction-program semantic intervention

- The original production clients disconnected after all three replicates had entered their fourth
  arm. Their fingerprint-bound restart ledgers survived on the persistent experiment volume, and
  detached resume continued from the most recent 25-step restart boundary rather than retraining
  completed work. All three training stages subsequently passed with zero fixed-state failures and
  published 20 authenticated model checkpoints each.
- Replicates 0 and 1 completed their original fixed-checkpoint evaluations and downloaded as verified
  runs. Replicate 2 completed training but its original evaluation stopped when a string emitted by
  RDKit could not be parsed again by the metric layer. The complete training stage was recovered by
  its manifest hashes; this is an evaluation failure, not lost training or a scientific negative.
- The sampler now requires a generated canonical SMILES to survive an independent sanitizing RDKit
  parse before marking the row valid. Invalid rows remain in the attempt ledger and are not repaired
  or retried.
- Replaced the misleading interpretation of the cyclic-ID training arm with a proper inference-time
  intervention. The new diagnostic reuses the three frozen final conditioned checkpoints and pairs
  the exact same Ugi layouts and flow noise under six conditions: factual coordinates, each of the
  two wrong reaction-family IDs, forward and reverse registry-role cycles, and removal of all
  program coordinates. No checkpoint is retrained or selected.
- Local CPU smoke run `d629661719eb8bfac5f06f8891d85d9a768ceb2c03e97a5084fae270e9f79af0`
  completed, independently verified, evaluated all six conditions over all three training seeds,
  and passed every execution and provenance gate. Its two attempts per condition are plumbing only
  and are not interpreted scientifically.
- **Decision:** the full paired semantic intervention is launch-ready on the qualified exact H100.
  Completion will report conditional dependence without forcing a positive effect. A changed output
  is not synthesis evidence, and the null inference condition remains an out-of-distribution
  diagnostic rather than a substitute for the separately trained null arm.

## 2026-08-24 - Qualify the Modal hydration fix and launch the semantic intervention

- The first approved launch exposed an older detached application,
  `ap-lnf0vFKGja8VNx4YDu1lY3`, that had retried a remote-hydration failure every ten minutes since
  2026-08-22. The container imported `experiments._runtime` from Modal's `/root` entrypoint mount
  before the separately copied `/opt/forge-project/forge` package was on `PYTHONPATH`, causing
  `ModuleNotFoundError: No module named 'forge'`. This application had produced no scientific
  result and was stopped rather than allowed to retry again.
- Stopped the first semantic-intervention application `ap-6x7xQB5JORD8E0kFf0d5Ii` before its
  queued H100 task could encounter the same defect. The three uploaded checkpoint archives remain
  immutable inputs; no partial output is admitted from this stopped launch.
- Added `/opt/forge-project` to the Modal image `PYTHONPATH` during image construction and added a
  launcher regression assertion. All 18 focused Modal and semantic-intervention tests pass.
- Ran the real CPU-only Modal installation smoke with the repaired launcher. Run
  `fddca3166cc6ba358f494341fd00a9418e506c990343e0bc0f357e33156bb9ff`, executable-source digest
  `e8cab91e9c6396cb8209e4c219aa1027bd7e8085d011fe6659509c0c87e8b1a7`, completed remotely,
  downloaded and independently verified.
- Relaunched the explicitly approved full exact-H100 semantic intervention as detached application
  `ap-RIW9WEncu8dRRRvgSNAS8l`, request
  `d88ecd957c3b9f8c24904598b39bc471baacee06338949f8d7b33fe9ac6fca1c`, on the same qualified
  source digest. Modal reports one active task; no scientific result is admitted before remote
  completion, download and manifest verification.
- **Decision:** the remote import boundary is now qualified independently of GPU allocation. The
  full semantic intervention is launched on exact `H100!`; its effect remains an outcome, never an
  execution gate.

## 2026-08-24 - Diagnose weak BL/LX generation with component-local layouts and a calibration-only repair study

- Replicates 0 and 1 of the frozen Transformer comparison showed exact-L1 yield near 3--4% for the
  repeated aza-Michael program and 18--19% for the repeated reductive-amination program. Inspection
  found a concrete pre-model defect: the factorized layout prior pooled reaction-core coordinate
  counts globally by role, then scattered them across repeated component copies. A sampled BL
  program could therefore place multiple copies' core coordinates in one precursor block and none
  in another before the Transformer received the program.
- The layout prior now preserves a count-only reaction-core signature for each component occurrence.
  It still stores no component identity, SMILES, fingerprint, atom state, parent pointer or bond
  target. Exhaustive support validation and regression tests require every repeated component to
  retain its own signature.
- Added optional repeat-equivalence and within-component-position coordinates to the reaction-program
  memory, plus a repeat-consistency objective over aligned exterior atom and parent-bond
  distributions. Component-instance indices exist only as loss masks and never enter an embedding,
  so this does not introduce a component vocabulary. The option defaults off, preserving strict
  loading of the completed production checkpoints.
- Added `strict_valence_topology_argmax` as a separate terminal policy. It preserves fixed adapter
  states, realizes the requested tree and every requested closure once under the existing atom/bond
  valence support, and abstains with a recorded reason when exact support is unavailable. It performs
  no fallback, closure omission, repair or retry; the historical unconstrained argmax path remains
  unchanged for frozen evaluations.
- Frozen a nonselecting two-model by two-decoder study: corrected-layout reference versus the
  repeat-aware Transformer, each evaluated with unconstrained and strict decoding. Only BL and LX
  calibration products are read. Layouts and flow-noise seeds are paired across all four cells; no
  heldout product, route service, oracle or candidate-selection call is permitted. The repeat loss
  weight is fixed at 0.25 rather than tuned.
- Current-source local smoke run `f09b2efd7da4174f2b739da0026234356284ca6b3d6869f880a430877aa62f85`
  completed and independently verified. It trained both models, evaluated all four cells for both
  programs, preserved every fixed state, reported coverage and precision, and recorded zero repairs,
  route calls, oracle calls and candidate selections. Its two samples per cell/program and two
  optimizer steps are plumbing only; all scientific metrics are intentionally uninterpreted.
- Current-source CPU benchmark
  `results/phase1/transformer_training_layout_repair_v1/result.json` retained numerical-equivalence
  gates (maximum loss difference `1.90735e-6`, gradient difference zero) and measured a 1.053x median
  optimized-path speedup on the unchanged 5,289,517-parameter reference Transformer. This is not a
  CUDA throughput result.
- **Decision:** the BL/LX failure now has a testable causal repair rather than an unstructured
  architecture change. The three-replicate H100 calibration descriptor is prepared but not launched;
  paid execution requires separate approval and a current exact-H100 preflight. Calibration outcomes
  may diagnose the next freeze but cannot select a production model or support a heldout claim.

## 2026-08-24 - Run seed 0 of the BL/LX repair calibration and preserve Ugi

- Before paid training, expanded the calibration-only evaluator to report Ugi as a nonselecting
  anti-regression anchor in every model/decoder cell. This supersedes the previous statement that
  only BL and LX are reported; no heldout product is read. Local smoke run
  `183cc47338aa6fb7902e84d25796454ab024b87bf17eabbf169a59f59f28ce5d` completed and independently
  verified the four cells, all three families and every fail-closed execution gate.
- With explicit user approval, exact-H100 preflight run
  `2afcef2e92874b5e62ee934d7e8b2a98ff358ae6e35bf1d64738c7dff9278f95` passed all gates and
  independently verified on an NVIDIA H100 80GB HBM3. Peak reserved memory was 3,911,188,480 bytes;
  deterministic replay was exact and fixed-state failures were zero.
- Full calibration replicate 0 ran from the identical executable-source digest
  `7a23b6fd10b139ce6eb096d47fee42d1ab224a2c2a32644d79fa93b74beb9241`. Run
  `9aa0115b3cd8d909db7fe1dbb4463a25cd5cfe0ce1395d55eb599d4dcc611589` trained both 5.3M-parameter
  models for 1,700 optimizer steps, evaluated 512 attempts per family in each of four cells,
  completed, downloaded and independently verified.
- Under strict decoding, repeat-aware conditioning changed exact-L1 yield per attempt from 9.96% to
  9.38% for BL, 17.58% to 44.14% for LX, and 78.91% to 78.91% for Ugi. Corresponding raw-valid
  fractions changed from 99.61% to 97.46%, 22.66% to 50.20%, and 91.21% to 93.75%.
- Retro-decomposition coverage and exact-forward replay precision are reported together. Precision
  was 100% in all six strict-decoder family/model cells. BL coverage remained only 10.00% for the
  reference and 9.62% for the repeat-aware model, so roughly 90% of valid BL attempts had no admitted
  exact-L1 decomposition. LX coverage improved from 77.59% to 87.94%; Ugi remained 86.51% versus
  84.17%.
- Every execution gate passed: calibration-only access, immutable fixed states, matched attempt
  counts, no repairs or retries, zero support overflow, no forbidden reductive-amination motif
  metric, zero route or oracle calls and no candidate selection. The 512 molecules are attempts,
  not independent experimental replicates, and one seed does not establish seed robustness.
- **Decision:** do not spend two additional seeds yet. The repeat-aware intervention is positive for
  LX and preserves Ugi, but it does not address BL. Diagnose BL reaction-core/handle realization on
  the paired attempt ledger before freezing a revised intervention; additional seeds would measure
  robustness of the current systematic BL failure rather than repair it. The hash-pinned summary is
  `results/phase1/bl_lx_repair_calibration_seed0_v1/result.json`.

## 2026-08-24 - Promote the BL aza-Michael junction to an adapter-fixed constraint

- Re-inspected the hash-pinned BL_2023 supplement
  (`46be449ddcef11da686d499015c868c767f4c02714c54e3d5ef5db35c4a42490`). General Procedure B,
  pages 9--10, reports repeated aza-Michael assembly between an amine head and an acrylate tail;
  Supplementary Figure S1, page 13, reports the 72-head by 10-tail library. The qualified registry
  remains the sole executable chemistry source for the atom-mapped transform; no SMARTS or component
  structure was retyped into model code.
- Added a versioned representation intervention that changes exactly one scientific field relative
  to the frozen base representation: all six semantic reaction-core coordinates of
  `bl_2023_repeated_aza_michael` are adapter-fixed. Ugi remains fixed and LX remains learnable. The
  generated head and tail interiors remain variable, and no component ID, SMILES, fingerprint or
  fragment token enters the model.
- The full-corpus census represented all 113,150 declared products and passed every gate. Across all
  610 admitted BL products, all 6,350 semantic core atoms are fixed with zero policy mismatches. The
  qualification receipt is
  `results/phase1/bl_core_constrained_representation_v1/result.json`.
- Built the deterministic 113,150-record packed cache at SHA-256
  `3744e066325f3e203139797ab48eb54725f95ce09b39209905fb656df53c5973`. Frozen fold counts, 194-atom
  support, three-closure support and source-balanced sampling are unchanged. Exhaustive layout
  validation covered nine semantic bundles; 4,096 additional seeded BL draws stayed within the five
  BL training-fold bundles and had zero fixed-state failures.
- Prepared a one-arm calibration that reuses the previous seed-0 repeat-aware Transformer and both
  terminal decoders. Relative to the previous matched strict cell, the only intended intervention is
  the BL semantic reaction-core mask. Ugi and LX remain nonselecting anti-regression diagnostics;
  only calibration products are read. Local smoke run
  `90c5c8e8d267b2c16e41cd75d8c6d23aea6d27467094ae2b35c052ef11987a2e` completed and independently
  verified all execution gates. Its two optimizer steps and two attempts per family are plumbing
  only and have no scientific interpretation.
- All 42 focused representation, cache, layout, Transformer and intervention tests pass; formatting,
  lint and all 30 vendored-data pins pass. The repository-wide `make test` remains red on 203 tests:
  201 exactly match the recorded historical-artifact baseline and two additional failures are in the
  untouched paper-experiment-matrix diagnosis. No failing node is in the changed constraint scope;
  no historical scientific gate or missing artifact was weakened to obtain a green local result.
- **Decision:** the BL reaction-semantic constraint is implementation- and smoke-qualified. Do not
  launch paid training from this decision entry. A fresh exact-H100 preflight and separate explicit
  paid-launch approval are required before the single full seed-0 calibration can run. Exact-L1
  coverage and forward-replay precision must still be reported together; exact replay remains
  transform consistency, not synthesis-success probability or route certification.

## 2026-08-24 - Run the BL-core seed-0 calibration without changing the Ugi train fold

- Added and ran a two-step exact-H100 preflight under current executable-source digest
  `b99e0d351a8ae7f77995a68f9595ac0f83a91b6d843e53d0346139aa2f5f2187`. The first launch exposed a
  descriptor/runtime mismatch before training because the smoke config declared CPU while the stage
  allocated CUDA; it produced no scientific output. Corrected preflight run
  `65c05a70ab277302502c296994de905ecfeadbab9b588185b6af7ca4fc3e9f10` ran on an observed NVIDIA H100
  80GB HBM3, passed every gate and independently verified.
- With explicit user approval, full run
  `639cf90dbcb57ba2ae2f32e255e2b208f01152c1de5b8c3efdae263534bf7140` trained the single 5,328,301-
  parameter BL-core-constrained repeat-aware Transformer for the fixed 1,700 optimizer steps. Modal
  preempted the first container after the persisted step-825 receipt. The same content-addressed run
  resumed from its model, optimizer and RNG snapshot, reached the fixed step-1,700 checkpoint, then
  completed, downloaded and independently verified.
- The controlled data contract was preserved: Ugi used 66,464 train products and 15,800 calibration
  products; none of the 30,122 Ugi heldout products was read. BL/LX used their existing 258/70 train
  and 158/38 calibration products. Program mass remained one third per family; no product identity,
  finite component vocabulary, route service, oracle or candidate selection entered training.
- Under strict decoding, BL exact-L1 yield increased from 48/512 (9.38%) in the matched historical
  repeat-aware cell to 369/512 (72.07%), a +62.70 percentage-point or 7.69x change. BL
  retro-decomposition coverage was 100% among valid products and exact-forward replay precision was
  100%; the strict decoder separately recorded 92 abstentions and performed no repair or retry.
- The non-BL diagnostics reveal a tradeoff rather than a universal improvement. Strict exact-L1 yield
  was essentially unchanged for LX, 226/512 (44.14%) versus 225/512 (43.95%), while Ugi decreased
  from 404/512 (78.91%) to 371/512 (72.46%). Exact-forward replay precision remained 100% for all
  three families. The unconstrained decoder independently moved BL from 4.49% to 23.83%, LX from
  35.74% to 34.77%, and Ugi from 41.21% to 37.11%.
- Every execution gate passed: fixed final checkpoint, calibration-only access, immutable fixed
  states, matched 512-attempt cells, no repairs or retries, coverage plus precision, no forbidden
  reductive-amination motif metric, zero route/oracle calls and no candidate selection. The frozen
  hash-pinned summary is
  `results/phase1/bl_core_constraint_calibration_seed0_v1/result.json`.
- **Decision:** the BL reaction-core constraint worked strongly enough to retain, but the final
  training mixture is not frozen and remaining production seeds are not launched. Next run one
  separately frozen seed-0 ablation that increases Ugi exposure using only the same 66,464-product
  Ugi train fold, with BL/LX data and all other training/evaluation controls matched. The 30,122 Ugi
  heldout products remain untouched. One seed is calibration evidence, not across-seed robustness or
  a final production-model selection.

## 2026-08-24 - Test additional Ugi train-fold exposure without changing BL/LX exposure

- Added an opt-in, explicit per-program microbatch allocation to the stratified production sampler.
  The default path preserves the historical equal-family allocation exactly. The new calibration
  appends ten Ugi rows to each historical 11-Ugi/11-BL/10-LX microbatch, producing
  21-Ugi/11-BL/10-LX batches. It does not remove BL/LX rows, change source marginals, alter equal-family
  PCGrad, introduce component identities or change any product fold.
- Local smoke run `53fa9d6e46be764cea24d09643267e5b7737e2af6dd70aa0a87fe52940e6a974`
  completed and independently verified. It observed the exact frozen row counts, passed the train-
  fold and heldout-access gates, preserved fixed states and evaluated both decoders for all three
  families. Its two steps and two attempts per cell are execution plumbing only.
- Exact-H100 preflight run `c9c579791ee13789d8472e8202f9e39e8ed30d253cd6482355ec224e3dc528a5`
  ran on an observed NVIDIA H100 80GB HBM3, completed, downloaded and independently verified. Every
  exposure, fixed-state, calibration-only, coverage/precision and no-repair gate passed.
- Full run `f2d85f206bab5b877b76ee421991ca4e80b477bf863a0369d8a61c48ff4e4973`
  trained the unchanged 5,328,301-parameter model for the fixed 1,700 steps. It processed 142,800 Ugi
  rows versus the historical 74,800, while BL remained 74,800 and LX remained 68,000. Every Ugi draw
  came with replacement from the same 66,464-product train fold; none of the 30,122 Ugi heldout
  products was read. The final checkpoint, training result and calibration evaluation completed,
  downloaded and independently verified.
- Under strict decoding, exact-L1 yield changed from 369/512 (72.07%) to 373/512 (72.85%) for BL,
  225/512 (43.95%) to 229/512 (44.73%) for LX, and 371/512 (72.46%) to 374/512 (73.05%) for Ugi.
  The Ugi gain was only three products or 0.59 percentage points and remained 5.86 points below the
  pre-BL-constraint 404/512 (78.91%) result. Forward replay precision was 100% for all three strict
  cells. Ugi raw validity increased to 87.11%, but conditional exact-decomposition coverage decreased
  to 83.86%, leaving exact-L1 yield nearly unchanged.
- The unconstrained decoder did not provide corroborating Ugi improvement: Ugi changed from 37.11%
  to 36.33%, BL from 23.83% to 16.80%, and LX from 34.77% to 38.87%. Every execution gate passed, and
  no repair, retry, route call, oracle call or candidate selection occurred. The frozen summary is
  `results/phase1/ugi_train_exposure_calibration_seed0_v1/result.json`.
- Formatting, lint, all 30 vendored-data pins and all 29 sampler/calibration/Modal focused tests pass.
  The repository-wide suite remains red on exactly its recorded 203-node historical and missing-
  artifact baseline; none of those nodes is in the changed sampler or experiment scope.
- **Decision:** retain the BL reaction-core constraint, but do not freeze this larger Ugi-exposure
  schedule or launch remaining production seeds. A 90.9% increase in Ugi row exposure did not
  materially recover Ugi exact-L1 yield, so simple resampling of the same train fold is not the main
  repair. Diagnose cross-family parameter interference at the shared output and routed-adapter
  boundary next. Do not add or reassign heldout products.

## 2026-08-24 - Freeze and run the three-seed final conditioned Transformer

- Froze the standard 32-row microbatch, 128-example effective batch, equal-program-mass
  BL-core-constrained repeat-aware Transformer as the final conditioned arm. The rejected enlarged-
  Ugi schedule is absent. The production descriptor fixes three seeds, 1,700 optimizer steps, a
  strict no-repair terminal decoder, 512 calibration attempts and 3,072 heldout attempts per
  reaction program.
- The earlier positive seed-0 calibration checkpoint was not reused: it authenticates a calibration-
  only design and only 512 attempts. Exact-H100 preflight run
  `48aafaaa0fb82705864bd2e2d055d0fc1ebd075d66c3e7bd9df265b9b753c2f9` passed and independently
  verified before production.
- Under the identical executable source SHA-256
  `a154389d2f8109e2d91dc508c0d9ab06a10bbe61fbfa6411663a931bb3355c39`, production runs
  `647a8c643b841a569a3a8fd9a721b431fdf27da4cd5d2c3667c53e86fa57eeaa`,
  `6f51e43124edca1c241bc466eb896b8e6689a80306bd12653a0e36eaefee0131` and
  `cb1454baeeb6c9a7a3f5f6351ce7ac587480151cfe76b8363af85a0c109ab183` completed on observed NVIDIA
  H100 80GB HBM3 accelerators, downloaded and independently verified.
- Heldout exact-L1 yield per attempt across seeds 20260825/20260826/20260827 was
  72.27%/90.30%/73.47% for Ugi, 76.53%/81.51%/74.35% for BL and
  38.61%/14.78%/40.43% for LX. Means were 78.68%, 77.46% and 31.27%, respectively. Exact forward-
  replay precision was 100% in every family/seed cell. LX therefore remains the most seed-sensitive
  family; no seed is selected or excluded.
- The method-blind common Ugi assessor preserved all 3,072 attempts per seed and produced passing,
  hash-pinned results at `results/phase1/forge_transformer_common_assessment_seed{0,1,2}_v1/`.
  Ugi unique method-visible open-ended exact-L1 yield was 702.15, 843.75 and 707.68 products per
  1,000 attempts. Bounded route-index misses remain abstentions and cannot support a synthesis-
  success comparison.
- **Decision:** these are the final three conditioned-model replicates for the current paper
  comparison. Report the complete across-seed distribution. Do not retrain, select a favorable seed,
  relax the strict decoder or interpret exact L1 replay as synthesis success.

## 2026-08-24 - Qualify matched baseline execution without weakening upload boundaries

- Current-source CPU finite-catalogue replicates
  `6ee5e552b8f622f9dda9c326246b211c32fcb2dc2d09a0a5c7e1d23c994a3cb4`,
  `5a492f06fd1722d7e19a8fc79627f83afe2d7c884a7a7d118a0ac3fcdbccf9fe` and
  `f20cbd3ff05112fd5a95ec01663a8c247acec5aba17e4de321c6c13521a8080f` completed and independently
  verified. Their common Ugi assessments are frozen under
  `results/phase1/finite_catalogue_common_assessment_seed{0,1,2}_v1/`. The catalogue has zero
  method-visible open-ended products by construction; this is the bounded-discovery contrast, not a
  validity defect.
- Split the learned-inventory selector's CPU smoke from its exact-H100 preflight so allocated-device
  equality remains fail-closed. Local run
  `45879045172e2eb7d67f601ee960443e45e2eb01f26c13142ddeed74671e29b3` and exact-H100 preflight
  `75775d201ddb32ac0ba8876857518d5168ba5e34becfe3f25b3c70b27220fa8b` both passed and independently
  verified. The three full learned-inventory replicates have not launched because exporting the
  private common molecular inputs to a new Modal job requires explicit user approval.
- Rebased the five one-factor Transformer ablations and FACT-matched/generous controls onto the final
  repeat-aware BL-core model and strict terminal decoder. Local eight-arm smoke run
  `9c0301e27fa67e15a409b814fc78f0fd9ccb6aed9ac0554e6396800bdc03ce4e` passed all execution and
  parameter-count gates. Its tiny smoke metrics are not scientific results; exact-H100 preflight and
  production remain unlaunched pending the same explicit private-input upload approval.
- The full local conditional-dependence diagnostic
  `e4d137d85897198ffac92fc54d19ff27be3046ae25e1f9398c334ea672d34376` completed and independently
  verified. Mean cross-validated intact-versus-shuffled classifier AUC was 0.4885, below the frozen
  0.60 threshold, and component-family overlap prevented a strict component-disjoint interpretation.
  The diagnostic therefore does not authorize a claim that the Ugi data exhibit detectable cross-
  role dependence; this negative result is retained.
- Prepared all nine full native requests for pinned RGFN, DeFoG and GenMol checkouts at seeds
  20260825--20260827 with 3,072 attempts each, plus one smoke request per method. No external native
  job was launched and no data were transferred to the new Modal baseline volume after the upload
  boundary rejected the request.
- **Decision:** preserve every qualified request and negative result. Continue already-authorized
  internal H100 controls, but do not upload private common inputs, heldout data or external checkouts
  to a new Modal destination without the user's explicit informed approval.

## 2026-08-24 - Adjudicate the final conditioned model and finite-catalogue baseline

- Added a separate final-production adjudicator rather than rewriting the historical negative
  four-arm aggregate. It independently verifies three promoted FORGE runs, three frozen four-arm
  control runs and three finite-catalogue runs, then binds all nine to executable-source digest
  `a154389d2f8109e2d91dc508c0d9ab06a10bbe61fbfa6411663a931bb3355c39`, paired seeds
  20260825--20260827, common molecular inputs and 3,072 heldout attempts per program and seed.
- The new immutable result is
  `results/phase1/final_bl_core_production_adjudication_v1/result.json`. Every input run and copied
  evidence artifact verifies; all zero-call, no-selection, no-repair, fixed-state, support,
  coverage/precision and forbidden-metric gates pass. The aggregate reports seed means, sample
  standard deviations and paired-seed bootstrap intervals without treating molecule rows as
  independent replicates.
- The final model met every prespecified Ugi-retention gate relative to the Ugi-only arm. Ugi
  exact-L1 yield was 78.68% versus 25.42% for Ugi-only, a paired difference of 53.26 percentage
  points (95% paired-seed bootstrap interval 41.24--61.82 points). Final exact-L1 means remained
  78.68% for Ugi, 77.46% for repeated aza-Michael and 31.27% for repeated reductive amination.
- FORGE produced 729.28, 313.91 and 34.83 unique operationally open-ended exact-L1 products per
  1,000 attempts for Ugi, aza-Michael and reductive amination, respectively. The train-only finite
  catalogue produced zero for all three by construction. This is the declared bounded-discovery
  contrast; it is not a synthesis-success comparison.
- The shared-null and cyclic-ID checkpoints use the same seeds, data, molecular support and attempt
  budget, but predate the promoted BL-core constraint. Their aggregate contrasts are retained as
  descriptive context, not presented as one-factor causal effects on the final model. The full
  paired inference-time program-semantic intervention remains the causal test.
- All focused adjudication and architecture tests pass, all 30 vendored inputs verify, and a clean-
  cache collection of 1,812 tests has exactly the recorded 201 historical/missing-artifact failures
  with zero new failures.
- The strict code/config provenance gate exposed five frozen receipts that pinned intermediate,
  uncommitted bytes later superseded by the final mechanism config, paper matrix, historical
  adjudicator, manuscript and refreshed test-baseline report. Each missing digest was searched over
  every version of its path in current-repository Git history, the frozen-code archive, the working
  tree and task-temporary copies; none was recoverable. Five exact expected/current identity pairs
  were therefore added to `docs/known_artifact_drift.json`. The declaring receipts remain immutable,
  and a different future digest still fails closed.
- **Decision:** Task 1 of `docs/PAPER_RESULTS_TASKS.md` is complete. Freeze this aggregate and do not
  retrain, seed-select, relax gates or overwrite the historical negative result. Proceed next to the
  inference-only program-semantic intervention using the promoted final checkpoints.

## 2026-08-24 - Complete the final-checkpoint program-semantic intervention

- Implemented a versioned inference-only intervention over the three promoted
  `bl_core_constrained_repeat_aware` checkpoints. Within every paired comparison, checkpoint,
  morphology layout and flow noise are identical; only the family token, precursor roles or all
  program coordinates change. No model was retrained and there were zero route calls, oracle calls
  or candidate-selection operations.
- Local smoke `a1ce6cbd225c80ac7de593050d71bf646809409715a569722c1bb3ba3a4a6264`
  passed before remote execution. Exact-H100 preflight
  `a80832881ecdc5d8d326b334adb585399ea1911ad9be343941d21114a7038d96`
  and full run `a653dcea7db0ac2e745e4efc9738d1e4bb750cd70fe318ed7192ad33e937f4dd`
  used identical executable source SHA-256
  `c8260b6e691978e06d07e442ca25201f79a8a9ce8357a6147c3ad98c86aaacdd`, completed on observed
  NVIDIA H100 80GB HBM3 accelerators, downloaded and independently verified.
- The full experiment evaluated 3,072 paired attempts for each of six conditions and each of three
  independent training seeds, producing 55,296 attempt rows. Factual Ugi exact-L1 yield was 39.69%.
  Removing all learned program coordinates reduced it to 3.62%, a factual advantage of 36.07
  percentage points (95% paired-seed bootstrap interval 34.28--37.24 points). Substituting the LX
  family token reduced yield to 17.09%, a 22.60-point advantage (14.06--27.47 points).
- Cyclically shifting precursor roles caused smaller but consistent exact-L1 losses: 0.82 points
  (0.29--1.69) for the forward cycle and 4.49 points (1.37--8.27) for the reverse cycle. The BL
  family-token substitution yielded 37.70%, only 2.00 points below factual, and its interval
  (-2.18--5.60) crossed zero. Canonical outputs changed for roughly 47--52% of paired attempts, but
  this is conditional dependence rather than chemical correctness. Exact forward-replay precision
  was 100% among every exact decomposition, and decomposition ambiguity was zero.
- The frozen hash-pinned summary is
  `results/phase1/final_program_semantic_intervention_v1/result.json`; the complete metrics and
  attempt ledger remain in the verified run. The null condition is out of distribution and cannot
  be interpreted as a matched trained baseline.
- **Decision:** Task 2 of `docs/PAPER_RESULTS_TASKS.md` is complete. The final Transformer uses the
  joint reaction-program coordinate system, with strong evidence for program presence and the
  Ugi-versus-LX distinction plus role sensitivity. Do not claim that every family ID is equally
  informative: the Ugi-versus-BL token effect is unresolved across three seeds. Proceed to the
  prespecified eight-arm Transformer mechanism and FACT study without selecting or modifying these
  results.

## 2026-08-25 - Complete the three-seed Transformer mechanism and FACT production runs

- All eight arms completed for independent seeds 20260825--20260827 on exact NVIDIA H100 80GB
  accelerators and executable source SHA-256
  `c8260b6e691978e06d07e442ca25201f79a8a9ce8357a6147c3ad98c86aaacdd`. The independently verified
  run IDs are `5f149b098c61a855dc1bf51b5ba1c3c31162d35fb5e84abeb174bde30a5eb54f`,
  `9f5633821b0234c367d7f1058703ba05034d2971c04ee45a7e9a26af83f311c1` and
  `3715b68bb7faaa8815ff575e07ff4c00f7983327322462ed8c2b463629172e46`.
- At 3,072 held-out attempts per program and seed, the full Transformer produced mean exact-L1
  yields of 79.18% for Ugi, 76.25% for repeated aza-Michael and 31.81% for repeated reductive
  amination. FACT-matched produced 5.20%, 39.37% and 50.99%; FACT-generous produced 29.67%, 68.70%
  and 56.13%, respectively. FORGE exceeded both FACT controls in every seed for Ugi and repeated
  aza-Michael, while both FACT controls exceeded FORGE in every seed for repeated reductive
  amination.
- Removing layerwise program cross-attention reduced mean exact-L1 yield by 5.45 percentage points
  for Ugi and 4.08 points for repeated reductive amination, but increased repeated aza-Michael yield
  by 1.74 points. Other one-factor interventions were also chemistry dependent. The arm without
  gradient-conflict control slightly exceeded the full-model mean for all three programs.
- The 24-row seed ledger is
  `results/phase1/transformer_mechanism_study_v1/seed_rows.jsonl` with SHA-256
  `4a382994c6dbc7bd36b362a18baf1e01da11ee5bf9c02a5a875b4ee18d93f938`. The paper-render result is
  `results/phase1/transformer_mechanism_study_v1/paper/result.json` with SHA-256
  `f3c7e4d332defe03649658467d320a89ade5d1ff129d43f86ac3b62518427c5e`.
- **Decision:** populate the manuscript architecture table and report the reaction-dependent result,
  including the adverse LX contrast and the no-gradient-control result. Do not claim universal
  superiority, detected cross-role dependence, or that every Transformer mechanism independently
  improves exact-L1 yield. Task 3 remains open only for its common-assessment and compute-accounting
  reporting requirements; these completed production runs are frozen and must not be retrained or
  seed-selected.

## 2026-08-25 - Insert every completed result into the v1 manuscript without filling missing methods

- Added a fail-closed completed-evidence renderer driven by
  `configs/reproduction/natbiotech_v1_completed_evidence_v1.json`. It verifies exact path/SHA-256
  pins before generating manuscript macros, table rows, figure summaries, CSV inputs and a result
  receipt. The renderer performs no model training, route search, oracle calls or candidate selection.
- Completed common Ugi assessment for the finite-catalogue oracle, shared-null post-hoc control,
  FACT-matched, FACT-generous and FORGE Transformer at paired seeds 20260825--20260827. These
  molecular and exact-L1 rows use zero route/oracle calls and do not import the bounded FORGE route
  index into a cross-method evidence comparison.
- Replaced supported manuscript placeholders with the final three-program production comparison,
  semantic interventions, Transformer mechanism study, catalogue tradeoff, Ugi novelty audits,
  bounded route cascade, synthesis-guidance failure audit and HeLa potency-guidance failure audit.
  The manuscript now reports the native held-component result as a negative finding: one exact-L1
  recovery in 9,216 attempts (0.11 per 1,000).
- Removed the unsupported role-specific held-component and optional held-reaction-family claims.
  The remaining red TBD cells are limited to the learned inventory selector, RGFN, DeFoG,
  GenMol/SAFE and their method-blind route-evidence union. Historical compute fields absent from
  frozen receipts are marked N/R rather than estimated.
- All 30 vendored inputs verify. Focused renderer, CLI and paper-matrix tests pass. The full
  1,814-node suite retains exactly the 201 documented historical missing-artifact failures with zero
  new failures; the refreshed baseline report is `blocked_known_failures`, not a green full-suite
  claim.
- **Decision:** freeze the inserted values through the completed-evidence config and renderer. Do not
  fill the remaining external or route-comparison rows from published values, native FORGE evidence
  or manual transcription. Complete the learned inventory and native external runs before freezing
  the method-blind union and strict final paper renderer.

## 2026-08-25 - Supersede the GenMol/SAFE empty-output labeling defect without retraining

- The completed native GenMol/SAFE ledgers incorrectly labeled an empty strict SAFE decode as
  `generated` because the adapter tested only `decoded_smiles is not None`. The original native
  receipts and sample files remain immutable. A new explicit supersession changes only the
  contradictory `generated` plus empty-payload state to `invalid`; it creates no molecule, changes
  no molecular payload, and performs no retraining, resampling, repair, route call, oracle call or
  candidate selection.
- The hash-pinned supersessions corrected 962, 925 and 968 labels at seeds
  20260825--20260827 while preserving all 3,072 requested attempts and compute-accounting totals in
  each seed. Future native runs now classify an empty SAFE decode as invalid at the runtime boundary.
- The common assessor passed for all three corrected ledgers. Valid connected yields were 62.73%,
  63.38% and 62.14% (mean 62.75%, sample SD 0.62 percentage points), but only 15, 20 and 17 unique
  valid products were produced. Exact Ugi L1 yield was 0/3,072 in every seed. The result is therefore
  paper-admissible as a negative external baseline with severe mode collapse, not as evidence that
  GenMol/SAFE generates Ugi-compatible lipids under this matched training protocol.
- The final assessment receipts are
  `results/phase1/genmol_safe_common_assessment_seed0_v1/result.json` (SHA-256
  `1702e839253fffe4ae1860f145a7c38124a4b0d06842d520ec7be084e9e3da0e`), seed 1
  `6aabb367c6e2204a5584d882d0ba39d8bcf2a924636a80f2daa4518954a5bb96`, and seed 2
  `ba402185811d38c5285161621a715b4203f67856c509312abbf7e8632016822c`.
- **Decision:** retain the original malformed native receipts as historical evidence and use only
  the explicit superseding ledgers for common assessment. Do not retrain or rerun GenMol/SAFE for
  this label-only defect. Include the resulting zero exact-L1 finding when the external-baseline
  table is next regenerated.

## 2026-08-25 - Add one method-blind whole-lipid structural-realism diagnostic

- Added `forge experiment assess-lipid-realism` as a separate question from exact Ugi-L1 replay:
  whether each requested generation attempt yields a diverse connected graph near the empirical
  structure distribution of source-study-held-out observed constitutional R0 lipids. This is a
  secondary post-hoc diagnostic, not a new confirmatory endpoint and not a replacement for exact
  synthesis-program evaluation.
- The common protocol is `configs/multireaction/common_lipid_realism_v1.json` (SHA-256
  `a671b7453a46b0103feabb536116235302e0559f72d3b002fb0b67f2d936f855`). It selects 4,096 R0-train
  structures by deterministic SHA-256 rank only to freeze robust descriptor scaling, and 2,048
  source-study-held-out structures as the evaluation manifold. Selection is independent of method
  outputs. The declared common support remains C/N/O/P/S and at most 194 heavy atoms; the 398
  excluded R0-train structures are explicitly counted rather than silently discarded.
- The assessment reports Morgan/Tanimoto and interpretable whole-lipid descriptor manifolds
  separately, including both precision and coverage. It also reports continuous nearest-reference
  distances, robustly standardized descriptor Wasserstein distance, grouped classifier two-sample
  AUC, validity/connectedness, uniqueness, Shannon effective molecule count and internal diversity.
  Failed, invalid, duplicate and out-of-support attempts remain in the 3,072-attempt denominator.
  QED is intentionally absent, and the evaluator performs zero route, proposal-engine or biological-
  oracle calls.
- Added a fail-closed seed aggregator. It requires identical config/reference pins, identical
  attempt budgets and the full seed set 20260825--20260827 for every included method; it reports the
  mean and sample standard deviation over training seeds and never treats molecules as independent
  training replicates. Missing metrics, such as a classifier AUC for a collapsed generator, remain
  explicitly not estimable rather than imputed.
- Focused tests, Ruff, Black, MyPy and all 30 vendored-data pins pass. The complete suite returns
  exactly the frozen 201 historical/missing-artifact failures and no new failures. During that
  check, the old M0-07 split writer was moved to the shared deterministic gzip serializer so Python
  3.11 no longer leaks a platform-specific gzip OS byte into a byte-exact frozen artifact.
- The current pre-DeFoG aggregate is
  `results/phase1/common_lipid_realism_v1/pre_defog_aggregate/result.json`. FORGE produced 86.60%
  connected outputs, a 95.68% unique fraction among connected outputs and a mean effective count of
  2,479 molecules. GenMol/SAFE produced 62.75% connected outputs but only a 0.90% unique fraction and
  effective count 3.25; RGFN produced no molecular outputs. The finite-catalogue and learned-
  inventory methods had higher descriptor-manifold precision per attempt (5.46% and 5.66%) than
  FORGE (0.88%), while FORGE's descriptor distance was slightly lower. Fingerprint-manifold
  precision was zero for FORGE and GenMol/SAFE and near zero for the inventory methods, and grouped
  classifier AUC remained near 0.97 where estimable. These absolute values show that no evaluated
  method should be claimed to reproduce the full held-out observed-lipid distribution.
- **Decision:** retain this as a multi-axis structural diagnostic. Do not collapse it into a single
  “reasonable molecule” score, use it to select candidates, or describe it as ionization,
  formulation, delivery, activity or synthesis success. Apply the unchanged evaluator to all three
  DeFoG ledgers when they finish, then regenerate the complete aggregate and paper table.

## 2026-08-25 - Complete the common molecular baseline and structural-realism paper tables

- DeFoG seed 20260827 completed on the previously authorized exact H100 run. Its downloaded samples,
  receipt and checkpoint match the hashes recorded by the remote runtime. The canonical import
  retained all 3,072 attempts and made zero repair, retry, route, oracle or selection calls.
- The frozen common Ugi assessor returned 929 exact-L1 products for seed 20260827, or 302.4 per
  1,000 attempts. Across three independent seeds, unconditional DeFoG produced $280.6\pm26.6$
  exact-L1 products and $15.5\pm2.4$ held-component exact-L1 products per 1,000 attempts. FORGE
  remained higher in exact-L1 yield at $786.8\pm100.8$ per 1,000 attempts, while DeFoG materially
  exceeded FORGE on recovery of specifically reserved held components.
- The final method-blind structural aggregate is
  `results/phase1/common_lipid_realism_v1/final_aggregate/result.json` (SHA-256
  `810d9ea329dce00c57ff19de36da0b3daa9efc73f6c7b5df7ff6bc9745f21d17`). DeFoG produced
  $1064.3\pm129.9$ effective molecules and 100% uniqueness among connected outputs, but only
  $17.3\pm1.4$ attempts per 1,000 entered the held-out descriptor manifold. No method reproduced
  the full observed-lipid distribution; grouped classifier AUC remained 0.961--0.974 where
  estimable and fingerprint-manifold precision was zero or near zero.
- The hash-pinned completed-evidence renderer now includes all three external methods and the learned
  finite-inventory selector, renders non-estimable quantities as N/E, and populates common benchmark,
  seed, decomposition, inventory, compute and structural-realism tables. Exact-L1 rates are also
  inserted into the otherwise pending route-evidence comparison.
- The resulting 25-page manuscript compiles without undefined citations, undefined references,
  overfull boxes or fatal LaTeX errors, and the result-heavy pages passed visual inspection. All 30
  vendored inputs verify. The full suite collected 1,822 tests and reproduced exactly the 201
  documented missing-artifact failures with zero new failures; the baseline status remains
  `blocked_known_failures`, not a green full-suite claim.
- **Decision:** freeze the common molecular and structural tables as complete. Preserve the negative
  RGFN and GenMol/SAFE results, the weak FORGE held-component recovery and the failure of every
  method to reproduce the full held-out structural distribution. Leave cross-method L2/L3 fields
  TBD until the method-blind component union is frozen and adjudicated.

## 2026-08-25 - Complete the method-blind cross-method L2/L3 evidence assessment

- Froze the exact role/constitution union from all 27 retained common-assessment ledgers (nine
  methods, three seeds each) only after those ledgers were complete. The union contains 11,313
  components. Its public worklist contains no method identity; private membership remains in a
  separate sealed ledger. The union result is
  `results/phase1/common_ugi_method_blind_route_union_v1/result.json` (SHA-256
  `978450c4702cc984b3cd382df94fc4ddc956add0d4b717032856de5e434da4d7`).
- Applied the pre-existing exact-source Ugi component dossier library to that public worklist with
  no access to private membership and no planner, proposal-engine, biological-oracle or selection
  calls. Of 11,313 components, 29 closed; 38 family projections were explicitly barred from
  closing, four had incomplete exact evidence and 11,242 had no exact-source record. Every
  non-closed item is an explicit missing-evidence abstention. The evidence result is
  `results/phase1/common_ugi_method_blind_route_evidence_v1/result.json` (SHA-256
  `bf3626b630f3504e829bd424613841a087db89677e80a2e18f0f10e9bdbeec35`).
- Adjudicated every method and seed against that one complete disposition index. FORGE generated
  $786.8\pm100.8$ exact-L1 products per 1,000 attempts but zero verified-upstream, terminal-evidence
  or complete-dossier products; all exact-L1 products abstained under this sparse library. DeFoG,
  shared-null and FACT-matched likewise had zero complete dossiers. The finite catalogue oracle and
  learned inventory selector closed only $0.1\pm0.2$ and $0.7\pm0.3$ dossiers per 1,000 attempts,
  respectively. RGFN and GenMol/SAFE had no exact-L1-eligible products. The adjudication result is
  `results/phase1/common_ugi_method_blind_route_adjudication_v1/result.json` (SHA-256
  `36b474711c5e20e9720e2ae6e62a495b4594ec973bae8fa15db52a7f93f3199f`).
- Removed the redundant DeFoG-plus-AiZynthFinder table row. A proposal-only planner result cannot
  close exact evidence and therefore cannot produce a distinct admissible L2/L3 value without new
  exact forward verification and terminal-material evidence.
- **Decision:** publish this as a negative result about the coverage of the current exact evidence
  library, not as a ranking of intrinsic synthesizability or a synthesis-success probability.
  Preserve family projections as abstentions, preserve all unknown components, and do not promote
  the bounded FORGE cascade into the cross-method comparison.

## 2026-08-25 - Freeze deterministic FORGE-generated sample visualization

- Added the versioned display contract
  `configs/reproduction/natbiotech_v1_forge_sample_figure_v1.json` and the reproducible renderer
  `paper/forge_paper/sample_visualization.py`. The figure reads the hash-pinned seed-0 common Ugi
  assessment and structural-realism ledgers; it does not resample the model or inspect biological,
  planner or procurement outputs.
- The frozen display pool contains 23 constitutionally unique, connected, support-conforming,
  descriptor-manifold, method-visible open-ended products with a unique exact-L1 trace. Three rows
  were selected by ascending SHA-256 rank, yielding attempt indices 2256, 2931 and 1586. This is a
  deterministic display policy, not an experimental-candidate selection rule.
- Each row shows the FORGE-generated constitutional graph, three deterministic post-hoc RDKit
  ETKDGv3 conformers optimized with MMFF94s (UFF only as the declared fallback), and the exact amine,
  aldehyde and isocyanide precursor constitutions returned by the common Ugi verifier. Hydrogens are
  omitted from the 3D rendering for clarity. FORGE does not generate three-dimensional coordinates.
- The final PNG is
  `paper/v1/figures/forge_generated_samples_v1/forge_generated_samples.png` (SHA-256
  `738e3049c2980cc9188c825f45099df1584a3c9556f414a1959b056602dca972`). Its selection receipt is
  `paper/v1/figures/forge_generated_samples_v1/selected_samples.json` (SHA-256
  `5c78e882f7793935a4240c222d457b6bca35e955a3cefe2625b8c42891e65f09`). The attributable result is
  `results/phase1/forge_generated_sample_figure_v1/result.json` (SHA-256
  `a6ffe5834de4108a870e39475477bd134cec521a5c5f28d35fb0a9e40b8f588b`). A repeated render reproduced
  all three hashes exactly.
- **Decision:** include this as a qualitative visualization of open-ended constitutional generation
  and exact L1 decomposition. Do not present descriptor-manifold membership as evidence of activity,
  formulation behavior or synthesis success, and do not present the post-hoc conformers as model
  outputs or ground-truth structures.

## 2026-08-25 - Replace the conformer display with reaction-program coordinates

- Superseded the manuscript's v1 conformer figure because FORGE does not generate three-dimensional
  coordinates and a post-hoc conformer ensemble obscured the model's actual scientific contribution.
  The v1 config, artifact and renderer branch remain reproducible but are no longer included in the
  manuscript.
- The v2 figure shows, for each row, the jointly generated constitutional graph, the same graph
  colored by exact precursor origin with enlarged reaction-core halos, and the exact amine,
  aldehyde and isocyanide L1 building blocks. Origin and core assignments are recovered by replaying
  the hash-pinned atom-mapped qualified Ugi transform; they are not inferred from substructure
  matching and are not a finite component catalogue used by the generator.
- To avoid three visually redundant long-chain examples, v2 uses a frozen deterministic display
  diversity rule. It starts from the lowest constitutional SHA-256 rank and adds the structure with
  maximum minimum ECFP4 distance, breaking ties by SHA-256. Among the unchanged 23 eligible products,
  this selects attempt indices 2256, 1194 and 2699. The rule remains display-only and performs no
  experimental candidate selection.
- The v2 PNG is
  `paper/v1/figures/forge_generated_samples_v2/forge_generated_samples.png` (SHA-256
  `d5eca1219abc5415be71d2d0c27d5c5dd16ec81579b41bd331bcb8282471fd65`). Its selection and exact-origin
  receipt is `paper/v1/figures/forge_generated_samples_v2/selected_samples.json` (SHA-256
  `aa370f57326e6866d2388408ea51db701ca975b15322b8397978ef40dd2129e6`). The attributable v2 result is
  `results/phase1/forge_generated_sample_figure_v2/result.json` (SHA-256
  `14391da8e6a798d6a649b69c7fd5395a5e8136e983d4a7d66c80a8fa66e9247e`).
- **Decision:** use v2 in the paper. Interpret its colors as a visualization of exact reaction-program
  semantics and its L1 precursors as transform-consistent decompositions. Do not interpret either as
  activity, synthesis success, route closure or experimental-candidate status.

## 2026-08-25 - Add the generated-structure appendix atlas

- Added a three-page appendix atlas containing 12 deterministic open-ended, exact-L1 FORGE Ugi
  products. Every row reports a role-colored constitutional graph, one decorated post-hoc 3D view and
  the exact canonical constitutional SMILES. Purple identifies the amine-derived head, teal the
  aldehyde-derived body and tail, orange the isocyanide-derived tail and gray the assembly-introduced
  oxygen. The color assignments come from exact atom-mapped forward replay.
- The atlas applies the same frozen display eligibility as the main sample figure, then takes the
  first 12 identities under SHA-256-seeded ECFP4 MaxMin ordering. The selected attempt indices are
  2256, 1194, 2699, 2919, 329, 1776, 1861, 382, 2078, 2546, 1152 and 1059. This is a display-only
  diversity rule and makes no biological, procurement, route or experimental candidate selection.
- Each 3D view uses one deterministic RDKit ETKDGv3 embedding followed by up to 400 MMFF94s
  iterations. One of 12 structures required the declared deterministic ETKDG random-coordinate
  fallback; all used MMFF94s, and three converged within the iteration budget. These coordinates are
  decorative post-hoc illustrations, not FORGE outputs or predicted bioactive conformations.
- The atlas pages have SHA-256 values
  `a7b1dd55ed3dfc4f9b60b90a56b8ad0bbf267b97341d63edb16839e871c3eb2e`,
  `6aeee7bd361863efc4a92d6f9533910707aa5b9bd4b630347ca35d560785ded5` and
  `9400f342a2f6388080b12ab8a48b9d6525ce5d941db3dcbab441bbfe9eb69a0b`. The exact sample and
  conformer receipt is
  `paper/v1/figures/forge_generated_sample_atlas_v1/atlas_samples.json` (SHA-256
  `36ff7ca3b4a7728676e9963cc16e9c03c40d2fc857e27433928ad00286821534`). The attributable result is
  `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `e7e747d219a4c3237542c9e4e401d570bbd6ad614b2687726daacd2d5650bf9b`).
- **Decision:** include the atlas in the appendix as a qualitative identity and reaction-program
  visualization. Do not use its 3D views as conformer evidence or its deterministic display set as a
  prospective selection panel.

## 2026-08-25 - Adopt white editorial styling for generated-structure figures

- Superseded the tinted panels, rounded cards and colored sample badges in both the main generated-
  structure figure and appendix atlas with a pure-white canvas, open whitespace and thin gray column
  and row rules. Exact precursor-origin colors remain on the molecular depictions and L1 role rules;
  no sample identity, structure, coordinate or scientific annotation changed.
- The revised main PNG has SHA-256
  `1b7be3da094ba43dde9361012a63ee3a694d2c35fef03d308b3c075bf3f875ad`; its unchanged selection and
  exact-origin receipt remains
  `aa370f57326e6866d2388408ea51db701ca975b15322b8397978ef40dd2129e6`. The superseding attributable
  result is `results/phase1/forge_generated_sample_figure_v2/result.json` (SHA-256
  `003d4cd437d8c565e663f7840ed3378f7b9118b00a2f6cc8e93d27062129503a`).
- The revised atlas pages have SHA-256 values
  `55f54ef122fdab04e4c6fadb2e20eda74a8cd1cf34c5e1d4f199060abc201558`,
  `97c3cf9f7512b0b390a5cae0314d5588aa5fa30bd0868159c356b3e372e7815a` and
  `686ab5ea6588f3de4d9a35bf01242c80471646900fe420c980c4f90f2a0751fd`. Its unchanged sample and
  conformer receipt remains
  `36ff7ca3b4a7728676e9963cc16e9c03c40d2fc857e27433928ad00286821534`; the superseding attributable
  result is `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `d7a3861736bfd436ec0214dccad36ff7453861d70eaaff7f05ea2ac63945e681`).
- **Decision:** use the white editorial renderings in the manuscript. This is a display-only change;
  the prior scientific interpretation and nonclaims remain in force.

## 2026-08-25 - Present the generated-structure atlas as one multipage table

- Replaced the three separately numbered atlas figures with one `longtable` containing all 12
  deterministic display rows. The three hash-pinned raster sections are now pagination units within
  a single table, with one caption, one label and repeated embedded column headings.
- No molecular identity, display ordering, semantic annotation, conformer coordinate, SMILES string,
  eligibility rule or artifact hash changed. This is solely a manuscript-organization change.
- **Decision:** cite the appendix atlas as Table 12 rather than Figures 6--8. Preserve the existing
  display-only and post-hoc 3D nonclaims in the unified caption.

## 2026-08-25 - Replace the redundant atlas semantic map with a conventional 2D graph

- Removed the precursor-origin-colored 2D map from the appendix atlas because it repeated the same
  partition already communicated by the decorated 3D view and by the main reaction-program figure.
  The replacement is a conventional RDKit 2D molecular depiction with standard bond and heteroatom
  coloring. The 3D column retains the exact atom-origin colors and legend.
- The revised atlas columns are ID, conventional 2D constitutional graph, decorated post-hoc 3D view
  and canonical SMILES. The page SHA-256 values are
  `d32d949798d70ae4985d046e41255cb14ad00c5d52998e3ae0554165a2562f04`,
  `6d431766acf73e29b8ece6cd1f5ca8c69046c83ee5ac5b4abc60bd96a0f316f3` and
  `c36840b38008bda77aa70ee6a573c050dceb7cca54692a5340025814bdcd0e32`.
  The unchanged sample and conformer receipt remains
  `36ff7ca3b4a7728676e9963cc16e9c03c40d2fc857e27433928ad00286821534`.
  The superseding attributable result is
  `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `9b534600f431f86046413085666a28758b9ab8e3dbe70f971244a25c8ea25433`).
- **Decision:** use the conventional 2D graph in the appendix table. This changes only the display;
  the selected molecules, order, coordinates, evidence and nonclaims remain unchanged.

## 2026-08-25 - Remove the redundant atlas ID column and narrow SMILES

- Removed the display-rank and attempt-ID column from the appendix atlas. Canonical SMILES remains the
  exact row identity, while the hash-pinned receipt retains display ranks and attempt indices for
  reproducibility.
- Reallocated the recovered width to larger conventional 2D and decorated 3D depictions. Narrowed the
  canonical-SMILES column and reduced its wrap width to 26 characters. The revised page SHA-256 values
  are `35e2154dc20c24c6b05c255e269db0c93a3bd839a8ec5103d7ce99cd4dbdf20b`,
  `0994fdad48bd49976b47693d0e29c647054dbad65955294bbedf9e4401f81d86` and
  `ecff742a8fa0c174c419bf209c87a321ed16c84182f64c9a7bf18b89afac09a6`.
  The superseding attributable result is
  `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `5c0504da31031378da9beeae8430a92743950faf391f739f573d8519fcf5bdd9`).
- **Decision:** present only the conventional 2D graph, decorated 3D view and canonical SMILES in the
  visible table. Preserve machine-readable IDs in the receipt rather than spending manuscript width
  on them.

## 2026-08-25 - Place canonical SMILES beneath each 2D atlas structure

- Collapsed the separate canonical-SMILES column into the conventional 2D cell. The visible appendix
  atlas now has two columns: a standard 2D molecular graph with its canonical SMILES directly below,
  and the decorated post-hoc 3D view. This preserves exact row identity while giving both molecular
  depictions more width.
- The revised page SHA-256 values are
  `8fd0ef1bd6e79a1c9008cc3dae687da187dfec6f82a997dc5de7b5ebe20074cf`,
  `9b05e0c095879fb82a6b54b8d503a92d11ec1e16acfa4cf19893d06d4c43f636` and
  `d7c5b8510212d9952194fe6bc3633667c298e908cec211ff34a10f1aa514f6a8`.
  The unchanged sample and conformer receipt remains
  `36ff7ca3b4a7728676e9963cc16e9c03c40d2fc857e27433928ad00286821534`.
  The superseding attributable result is
  `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `beeb31968f7b8091ab921d3fec541939aeb83410ba15184f3bac3f8d0792a453`).
- **Decision:** use the two-column atlas with SMILES embedded beneath the corresponding 2D graph.
  This is a display-only change; molecule selection, ordering, coordinates, evidence and scientific
  nonclaims remain unchanged.

## 2026-08-25 - Typeset the generated-structure atlas as a native LaTeX table

- Replaced the three page-sized raster composites with a native two-column LaTeX `longtable`.
  The renderer now emits one conventional 2D image and one decorated post-hoc 3D image per row, plus
  hash-pinned LaTeX row markup. The manuscript owns the caption, repeated column headings, rules,
  pagination and canonical-SMILES typography, so these elements use the paper's fonts and spacing.
- Removed the raster legend. Its precursor-origin color definitions remain in the paper-native table
  caption. The 12 rows are paginated four per page, and canonical SMILES remains directly beneath
  the corresponding 2D structure.
- The atlas renderer contract advances to
  `forge.paper.forge_generated_sample_atlas.v2`. The generated LaTeX rows have SHA-256
  `67e7fa9959c757b5d73ad1a0fc6ff9ed1bc1d2b8167da4aeb791c922f986827d`; the unchanged sample and
  conformer receipt remains
  `36ff7ca3b4a7728676e9963cc16e9c03c40d2fc857e27433928ad00286821534`. The superseding attributable
  result is `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `d66111653ef8480db8bcf9e34c3d377133a98e935a664caf8b0ce37b39016834`).
- **Decision:** use paper-native table construction and individual molecular assets for the appendix
  atlas. This is a display-only change; molecule selection, order, coordinates, evidence and
  scientific nonclaims remain unchanged.

## 2026-08-25 - Move the generated-structure atlas to the appendix end

- Placed the generated-structure atlas after the theoretical appendix so the multipage visual table
  no longer interrupts technical results.
- Row separation had never worked: a row terminator's optional argument only raises that row's
  depth, which a full-height minipage already exceeds, so the previous `2 mm` was silently
  discarded. Separation now comes from a struck row height that exceeds both panels, which also
  keeps the column divider unbroken and fixes the multipage break at the same place for every row
  instead of following whether a SMILES string happened to wrap. With the forced page breaks after
  rows four and eight removed, the 12 rows paginate 3/5/4 across three filled pages rather than
  4/4/4 across three mostly empty ones.
- Restyled the table: rules are light gray and confined to the head and foot, the two panels are
  separated by a continuous hairline rather than a full-width rule per row, headings are small
  capitals, canonical SMILES moved from `\tiny` ragged-right to centered `\scriptsize` gray
  monospace, and a display-order index column was added. The row artifact now names manuscript
  lengths and styles instead of carrying its own dimensions, rules and page breaks, so the layout
  can be retuned without reissuing a hash-pinned artifact. The atlas `layout.columns` contract gains
  `display_diversity_rank`.
- Fixed two rendering defects the restyle exposed. Decorated views were centered on the atom
  centroid while scaled by the projected extent, so a lipid whose mass sits in its tail had its head
  clipped by the canvas edge; they are now centered on the projected bounding box. Both panels are
  also trimmed to their drawing, since a fixed-aspect canvas letterboxes a molecule and that margin
  became dead space in the appendix row. The decorated-view canvas moves to 1360x900 to suit its
  narrower column, and bond stroke and atom radius are now fractions of one drawn C--C bond so the
  drawing keeps a constant visual weight across canvas aspects.
- The generated LaTeX rows have SHA-256
  `59c33c4f7113e079d658e4d9d48972d02ffaa63367ec51821de964e3354d35d3`; the unchanged sample and
  conformer receipt remains
  `36ff7ca3b4a7728676e9963cc16e9c03c40d2fc857e27433928ad00286821534`. The superseding attributable
  result is `results/phase1/forge_generated_sample_atlas_v1/result.json` (SHA-256
  `dbb6589e8feda489006d4b9bfcc6a6e5e529706c8fe0cb021a1c876d15d17a31`), repinned in
  `configs/reproduction/natbiotech_v1_completed_evidence_v1.json` (SHA-256
  `41a6b2a7ce8a541b1a65839b3c08a86654df2c7cf1a193ed28e40d695b7ddbc5`), after which the
  completed-evidence render was reissued. Macro values did not change.
- **Decision:** keep the atlas as the final appendix section with visibly separated molecular rows.
  This changes only manuscript order, layout and panel framing; molecule selection, order,
  constitutional identity, SMILES, conformer coordinates, evidence receipt and scientific nonclaims
  remain unchanged.

## 2026-08-26 - Add training-derived local chemistry constraints without retraining

- Added a versioned, training-fold-only local-support policy for all three current reaction programs.
  It records program- and precursor-role-conditioned element/bond neighborhoods, three-membered-ring
  compositions and observed component carbon/heteroatom bounds. It stores no component identity,
  fingerprint or complete fragment. The deterministic policy is
  `results/phase1/local_chemistry_support_v1/policy.json` (SHA-256
  `855c64ff810ab3266ba9630fdc243423842d932e79305eb8e8eeb438f66bef34`); its result is
  `results/phase1/local_chemistry_support_v1/result.json` (SHA-256
  `bfa0b07dcf9cdd42abf7b73dacca31725b6e5bd8adf1339c806126c9ed0a5a40`). It was fit
  from 66,464 Ugi, 258 BL and 70 LX training products only.
- Added `strict_local_chemistry_argmax` as a new opt-in terminal decoder while preserving the frozen
  unconstrained and strict-valence decoders. The new path masks unsupported local element/bond
  assignments, masks unsupported three-membered closures before atom realization and enforces
  observed program-role carbon/heteroatom bounds incrementally. It performs one decode with no
  repair, retry, component lookup, route call or oracle call; an infeasible request abstains.
- Added a separate method-blind, program-only assessment that supplements rather than replaces raw
  exact-L1 yield. On the already frozen FORGE attempt ledgers, local-support-qualified exact-L1 yield
  was 52.34%, 68.88% and 54.92% for seeds 0, 1 and 2, respectively, versus raw exact-L1 yields of
  72.27%, 90.30% and 73.47%. Precision among raw exact-L1 products was 72.43%, 76.28% and 74.75%.
  These negative differences are retained: the policy is intentionally stricter than graph validity
  and exact L1 replay, and absence from observed support is an abstention rather than proof of
  chemical impossibility.
- Added an evaluation-only resampling experiment that binds the exact three frozen final checkpoint
  archives to this policy. No weights are updated. A CPU seed-0 smoke run completed all 20 gates with
  no fixed-state failures, repair, retry, route call, oracle call or candidate selection. Its result is
  `runs/phase1-local-chemistry-resampling/e6c1e24d76c4a5cdbe0a88dba7456dfc6e5cf61b03e2b7bf9769df06a3ffbbd7/stages/resampling/artifacts/result.json`
  (SHA-256 `decb50504d9a81314a40845b2fc977095139d4c9192e6a6609323eb584dbd707`).
- **Decision:** retain the frozen raw production results as the primary historical result and use the
  local-support decoder as a transparent constrained-resampling policy for improved displayed and
  downstream samples. Do not call this a finite component vocabulary, chemical-validity oracle,
  synthesis-success probability or route certificate. A full three-seed H100 resampling remains a
  separate paid launch and must not be inferred from the passing smoke run.

## 2026-08-26 - Complete three-seed H100 local-chemistry resampling

- The authorized evaluation-only H100 experiment completed for frozen production replicates 0, 1
  and 2 (seeds 20260825, 20260826 and 20260827). All three downloaded runs independently verify and
  share executable source SHA-256
  `1109b6cecb644ad1cc6c447344189e9757ed1812e9f562721e3aaf9f7d29cee1`, one experiment
  specification, one frozen checkpoint design and local-support policy SHA-256
  `855c64ff810ab3266ba9630fdc243423842d932e79305eb8e8eeb438f66bef34`. No model weights were
  updated.
- At 3,072 held-out attempts per program and seed, constrained exact-L1 yield (mean +/- sample SD)
  was 75.88 +/- 11.25% for Ugi, 59.75 +/- 5.14% for BL and 18.07 +/- 5.56% for LX. Relative to the
  frozen original decoder means, the constrained decoder retained 96.44%, 77.14% and 57.77% of Ugi,
  BL and LX exact-L1 yield, respectively. Strict-constraint abstention rates were 2.68 +/- 2.37% for
  Ugi, 34.44 +/- 8.73% for BL and 54.64 +/- 6.65% for LX. The much larger BL/LX cost is preserved as
  a data-sparse-family limitation rather than hidden by changing the gate.
- A direct graph audit covered all 27,648 final-heldout attempts and all 15,029 valid products. Every
  attempt records the bound policy. Across valid products, the declared liabilities were absent:
  zero O--O bonds, zero N--O bonds and zero three-membered rings. This audit does not imply synthesis
  success, stability, safety, activity or route closure.
- The attributable aggregate is
  `results/phase1/local_chemistry_resampling_adjudication_v1/result.json` (SHA-256
  `ebd5054b58ec27a9a0e513fe8b29ec98f6bbcb551ada54a0d7bd83ee28f0b5f9`). It can be reproduced with
  `forge experiment adjudicate-local-chemistry-resampling` over the three verified run directories.
- **Decision:** qualify the constrained decoder for clean qualitative galleries and explicit
  downstream acceptance, especially for Ugi. Do not supersede the original production metrics: the
  weights are unchanged and strict decoding trades yield for bounded local support. Retain the
  original decoder as the primary production comparison and report the constrained result as a
  transparent sampling-policy sensitivity analysis.

## 2026-08-26 - Freeze role-local ring morphology before atlas replacement

- Extended the training-fold-only local-support policy to closure-defined ring morphology. The v2
  policy records complete element and precursor-role multisets for variable training closures
  relative to the deterministic spanning tree. Fixed adapter cycles cannot authorize a generated
  closure. The policy stores no component identity, fingerprint or complete fragment and does not
  retrain any checkpoint. Its frozen artifact is
  `results/phase1/local_morphology_support_v2/policy.json` (SHA-256
  `50dc654b215c2b1e952910cb5a4b5dc228824ac2a61731ca4c8a20035c680538`); the build result is
  `results/phase1/local_morphology_support_v2/result.json` (SHA-256
  `282cfcb76aa71c090b7981c2b24f38fa4a8c7f108e79fa749642e057089478f6`).
- The train-fold audit found no Ugi four-member ring support. Ugi five-, six- and seven-member rings
  occur in the amine-head role, while the only admitted Ugi isocyanide-tail ring signature is a
  six-member all-carbon ring. BL ring signatures are confined to the amine-head role; LX admits a
  six-member all-carbon aldehyde-tail ring and a six-member C4N2 amine-head ring. These are observed
  local-support statements, not claims of chemical impossibility outside the data.
- The strict decoder now rejects a variable closure before atom realization when its complete
  reaction-family and role multiset is absent from training support. Atom realization subsequently
  requires the observed role-and-element ring signature. Existing v1 policies retain their exact
  historical behavior and do not silently claim full-cycle support.
- A local CPU execution smoke completed and independently verified as run
  `775403a326ae1ff2b7e71da3a879de2d5da8a783a9c47f5ae0751eb1e42882fd`. The full seed-0 diagnostic
  is frozen at 3,072 attempts per program with thresholds fixed before execution: at least 90% exact-
  L1 yield retention per program, at most 0.02 absolute internal-diversity loss, at least 80%
  effective-component-count retention, zero oxygen-containing three- or four-member rings, zero Ugi
  oxygen rings outside the amine-head role among exact-L1 components and zero unsupported role-ring
  signatures among exact-L1 components.
- **Decision:** do not replace the paper atlas from the smoke run. Run the exact-H100 preflight and
  full seed-0 diagnostic only after explicit approval to upload the private source and pinned inputs
  to paid Modal compute. Promote the policy to three-seed resampling and regenerate the atlas only if
  every prespecified seed-0 gate passes; otherwise retain and report the negative result without
  changing thresholds.

## 2026-08-26 - Reject v2 role-local morphology after the frozen seed-0 diagnostic

- The authorized exact-H100 preflight completed and independently verified as run
  `bf33c93dedb19404ac1b6b5484446edf9c4c12b6a8afe5065565d03f41d47240`. Every execution gate
  passed with the fixed checkpoint archive SHA-256
  `08d9ebb04497d9ee78488c100315f1d50eef65bdeae84d9252d61cc4dead77e5`, morphology-policy
  SHA-256 `50dc654b215c2b1e952910cb5a4b5dc228824ac2a61731ca4c8a20035c680538`, no repair or retry,
  and zero training, route, oracle or candidate-selection calls.
- The contingent full seed-0 run then completed all 3,072 held-out attempts per program and
  independently verified as run
  `c42ab83331f77b5b3f0097abcce459a18a95cfc7e7447cb43a806ac90c901d3b`, with executable source
  SHA-256 `adab947055663caa02125c57e3311438efeee906acbcc4db6e7f593b35e0769b`. The independent audit
  initially failed closed because repeated BL roles serialize multiple components as a list. The
  adjudicator was corrected to assess every repeated component separately, with a regression test;
  the generated ledger, thresholds and sampling result were not changed or rerun.
- Versus the frozen v1 seed-0 decoder, v2 exact-L1 yield retention was 91.73% for BL, 71.41% for LX
  and 82.19% for Ugi. Effective-component-count retention was 88.29%, 63.21% and 76.69%,
  respectively. Internal-diversity change passed for BL but decreased by 0.0754 for LX and 0.0210
  for Ugi. The graph audit found zero O--O bonds, zero N--O bonds and zero small oxygen rings for BL
  and LX, but five oxygen-containing three- or four-member rings among valid Ugi products. It also
  found 6, 961 and 63 unsupported exact-L1 role-ring signatures for BL, LX and Ugi, respectively.
- The attributable negative result is
  `results/phase1/local_morphology_seed0_adjudication_v1/result.json` (SHA-256
  `ae426eb2c6ab94455e8c96868cc64e6d0b69bc5d3a1952e4fcf9965eea97bf87`). It preserves the
  prespecified thresholds and records `qualified_for_three_seed_resampling_and_atlas: false`.
- **Decision:** do not promote v2 to three-seed resampling and do not replace the paper atlas from
  this diagnostic. Retain the already qualified v1 constrained decoder and its frozen paper result.
  Treat v2 as a negative seed-0 result: its stronger closure mask did not reliably enforce the final
  product-level morphology contract and imposed unacceptable Ugi/LX yield and diversity costs.

## 2026-08-26 - Complete the Ugi-only Transformer seed-0 training and matched sampling gate

- The exact-H100 two-step preflight completed and independently verified as run
  `c3055d4e00818d8be444d0f50ba3a8bf44ed12647a8d013c0fa07a358a0eb322`. The contingent production
  run then completed exactly 5,100 optimizer steps on an NVIDIA H100 80GB HBM3 and independently
  verified as run `f35b30a5a7d3bd6386f0b4c5f578f5808c9cb070df16c7092e1759dc99680ead`.
  It used the 66,464 train-fold products for fitting, the disjoint 15,800 calibration products only
  for nonselecting diagnostics, and no held-out product for fitting or checkpoint selection.
- The final-step checkpoint is SHA-256
  `745271c7552bd9212c97a04cf004a7f7724cc3acc4b7e062cd6963a8a7239951`. Its final training loss was
  0.2131 and its nonselecting calibration loss was 6.5486. The larger calibration loss is preserved;
  the model was frozen at the prespecified final step rather than selected post hoc.
- The checkpoint-specific original constrained Ugi sampler then returned all 3,072 fixed program
  attempts. The downloaded run independently verifies as
  `4d7d2a67253732f7428e450ef8af4d44e769222c3fbd8aeae74b9900022ac7a9`. Exactly 2,883 products
  (93.85%) were valid, terminal-valid and exact-L1; 189 failures remain in the denominator. Sampling
  made zero repair, retry, route, oracle or candidate-selection calls. One infrastructure
  cancellation was recovered by authenticating completed shard receipts and replaying only the
  incomplete shard from its original fixed seed.
- The attributable summary is `results/phase1/ugi_v0_port_transformer_seed0_v1/result.json`
  (SHA-256 `7ff7aad960e39e1598ca02782683b578e0ebf694c9a5fd9b9edfe7342e534167`).
- **Decision:** admit this checkpoint to the matched v0 evaluation harness. Do not claim superiority,
  lipid realism, diversity, novelty, held-component improvement or seed stability from exact-L1
  yield alone. Run the common assessment against v0 before deciding whether the Transformer
  supersedes the historical model, and retain this as a single-seed Ugi-only result.

## 2026-08-26 - Implement the tree-aware Ugi Transformer development challenger

- Added a tree-aware sparse Transformer that derives parent, child, sibling, ancestor and descendant
  relations from the current noisy offspring state at every forward pass. It never consumes the
  clean target tree. Invalid intermediate forests are handled by a deterministic fail-soft relation
  construction rather than by exposing terminal topology.
- Added one global Ugi-core token, three role-specific core-port tokens, node-to-program routing that
  exposes only the global and matching role token, role-specific residual adapters, attachment- and
  junction-program consistency losses and whole-role source masking. Existing GRU and dense
  Transformer backbones and the restartable constrained sampler retain their original interfaces.
- Froze a development-only seed-0 contract at 3,000 optimizer steps and 384,000 weighted train-fold
  draws, matching qualified v0 development exposure rather than the previous 5,100-step schedule.
  Calibration remains diagnostic and held-out products select nothing. The four-arm ablation ladder
  and molecular promotion gates are fixed in
  `configs/model/phase1_ugi_tree_transformer_ablation_ladder_v1.json`.
- The final fused-attention source completed the local two-step smoke as run
  `80ecb70a02397009d127efb4240fc91d921419c8d0291364285c850c55bc1988`; strict reproduction was
  byte-identical for every declared artifact. A one-thread CPU forward microbenchmark measured 1.21
  ms/batch for the dense program Transformer and 1.49 ms/batch for the tree-aware challenger at
  batch 4, 30 nodes, width 96 and two layers. This is an engineering overhead measurement, not an
  H100 throughput or scientific result. Moving the structural kernel from manual attention to
  scaled-dot-product attention reduced tree-model latency from 1.72 to 1.49 ms/batch in the same
  representative setup.
- `make verify` authenticated all 30 vendored assets. A clean-cache 1,866-node test run reproduced
  189 known historical/missing-artifact failure nodes and introduced zero new failures against the
  frozen baseline. The suite remains blocked on the unchanged 54 files in
  `docs/missing_test_inputs.txt`; none was fabricated or converted into a scientific pass. The
  attributable engineering record is
  `results/phase1/ugi_tree_transformer_engineering_v1/result.json` (SHA-256
  `8ca0a063e9ee192acc4abb6b6e9c4516de673a5e4fd6d5f64d6ee2259d603300`).
- **Decision:** the challenger is implementation-qualified for an H100 preflight, but no paid launch
  is authorized by this entry and no improvement over v0 has been measured. Run the matched
  development ladder and common molecular calibration before promoting a checkpoint or making an
  exact-L1, realism, novelty or held-component claim.

## 2026-08-26 - Freeze matched tree-Transformer execution and calibration contracts

- Added a single exact-H100 descriptor whose smoke replicate exercises the most demanding
  component-masking arm and whose four full replicates map one-to-one to the dense corrected,
  tree-relational, balanced-consistency and full component-masking arms. Every arm uses the same
  train fold, initialization/minibatch seed, 3,000 steps, batch 128, 384,000 weighted draws and
  ten-checkpoint grid. The runtime envelope now states Modal's real 24-hour cap and preserves
  authenticated resume checkpoints.
- Removed `held_component_exact_l1_per_1000` from calibration-based checkpoint selection. It remains
  a non-selecting held-out diagnostic only. Held-out products cannot select an architecture,
  checkpoint, temperature or threshold.
- Froze 512 calibration-fold morphology programs before remote training: 256 from the observed Ugi
  union and 256 from expanded exact-forward support, balanced within source over calibration-role
  classes. Run `9bd9ce0b79753c0c61e5a51de1083389ebce90697a305f4e666a9f3dcf07ea6d`
  reproduced byte-identically; the program artifact SHA-256 is
  `ad51b852116e03ca6ab9ed3e44fefd899795f435337a1f615564ec697aa01054`.
- Added an authenticated, restartable evaluator that samples every prespecified checkpoint on the
  same ordered programs and random streams, retains failures in the denominator and runs the shared
  exact-L1, lipid-realism, local-chemistry and role-morphology assessors. Checkpoint selection is
  deferred to cross-arm/v0 adjudication; candidate selection is absent.
- `make verify` authenticated all 30 vendored assets. The locked `uv` full suite collected 1,871
  nodes and reproduced 189 known missing-artifact failures with zero new failures; 54 historical
  inputs remain absent. `make test` itself is miswired to a Homebrew Python 3.14 environment without
  pytest, so the locked `uv` environment supplied the actual test result.
- The planned exact-H100 preflight did **not** start. The execution layer rejected transfer of the
  private source and pinned molecular inputs to Modal without a new explicit approval. The frozen v0
  step-3000 checkpoint is also absent locally and must be retrieved from its private Modal volume
  before matched v0 calibration resampling. The attributable readiness record is
  `results/phase1/ugi_tree_transformer_execution_readiness_v1/result.json`.
- **Decision:** preserve the implementation and calibration draw exactly. Do not claim any model
  improvement. After explicit transfer/compute approval, run the exact-H100 smoke; launch all four
  full arms only after it verifies, then retrieve and resample v0 on the same calibration draw.

## 2026-08-26 - Complete the exact-H100 tree-Transformer development ladder

- The first remote preflight failed before optimizer step zero because the stage wrote its generated
  effective config inside the trainer output directory, which the trainer correctly requires to be
  empty for a fresh run. The stage now keeps that config beside the trainer directory. Focused runtime
  and tree-Transformer tests passed before rerunning; no scientific gate, dataset or model setting was
  changed.
- The repaired two-step exact-H100 preflight completed and independently verified as run
  `bd92ce73a6c92b390dfdd7e4d558bd2cc423002ae3ef5ba09e229a99b8c5eb16`. It exercised the most
  demanding `full_component_masking` arm on CUDA/float32 with Modal accelerator request `H100!`.
- All four matched 3,000-step development arms then completed and independently verified: dense
  corrected schedule `7a4fa03792fa47547292e9f93e9aef7c5ab291cf24d89122e0d28a5fd104a1bc`,
  tree relations and routing `bb7a9515d57269765a62b400c19cbb3571a9b244d1ccb3e323f6e4595a9dea82`,
  tree plus balanced consistency
  `0e575860ba665b816df6b659db8fcc19da6385681d0128e11655d35726461267`, and full component masking
  `a204c61a5d6c0658d4bb785885d02161cf9f5bf6140e663836d62b846bc6cb53`. Each preserves all ten
  prespecified checkpoint snapshots, the same 66,464 train products, 15,800 calibration products,
  384,000 weighted train draws and zero held-out selection.
- The attributable execution record is
  `results/phase1/ugi_tree_transformer_development_h100_v1/result.json`. The historical v0 volume was
  located in the separately configured `nitya` Modal workspace. After explicit cross-workspace
  authorization, exactly
  `/ugi_decoration_coupling_v1_challenger_ba78c781ba17/checkpoint_step_3000.pt` was downloaded from
  `nitya/forge-phase1-training`. The 14,032,416-byte local checkpoint matches its frozen expected
  SHA-256 `4a9dc01ec1bcc83cd18c257185b18fe2b66b2b19d93a12ed44f696b490bbad50`.
- **Decision:** training completion alone selects no architecture or checkpoint. Do not compare the
  arms using their differently weighted training objectives and do not claim exact-L1, realism,
  novelty or held-component improvement. Run the frozen 512-program molecular calibration for every
  checkpoint and the matched v0 assessment before promotion.

## 2026-08-26 - Qualify the matched tree-Transformer molecular calibration for launch

- Added one paired four-program exact-H100 execution preflight, four independent full Transformer
  calibration descriptors and one full frozen-v0 descriptor. Each full Transformer job evaluates
  all ten prespecified checkpoints on the same 512 ordered calibration programs and random streams;
  v0 uses the same programs, sampler settings and assessors. Every descriptor requests `H100!`,
  CUDA and float32, retains every failed attempt in the denominator and performs no repair, retry,
  route, oracle or candidate-selection call.
- Corrected the shared joint sampler so reverse flow runs on the descriptor-selected CUDA device;
  terminal graph construction and RDKit assessment remain on CPU. Resume receipts are now bound to
  the requested device, preventing a CPU partial result from being admitted to a CUDA run.
- Froze the cross-arm/v0 rule before opening any full calibration result. A Transformer checkpoint
  must improve unique open-ended, whole-product-novel exact-L1 yield by at least 0.02 with a paired
  95% bootstrap interval above zero; it must be noninferior within 0.02 on validity, exact-L1, local
  support, role support, tail support and component novelty, and retain at least 80% of v0 effective
  component count. The method-blind adjudicator reconstructs every metric from authenticated attempt
  ledgers using 10,000 fixed-seed paired program-index bootstrap resamples. If no checkpoint passes,
  no Transformer is promoted and the negative result is complete.
- All six descriptors pass `forge doctor`; all 30 vendored inputs verify. Thirty-five focused tests
  pass. The clean-cache full suite collected 1,877 nodes and reproduced 189 known failures from the
  54 documented missing historical inputs, with zero new failure nodes.
- The attributable readiness result is
  `results/phase1/ugi_tree_transformer_calibration_readiness_v1/result.json`. Its executable source
  SHA-256 is `38bcb7a74de881fcbd968ec3d219c2f43e25a3ed1d54fb8c17c80548715118ae`.
- **Decision:** the calibration is implementation-ready but paid compute is not authorized by this
  entry. After explicit approval, run and independently verify the exact-H100 preflight; only then
  launch the four full Transformer jobs and the full v0 job in parallel and apply the frozen
  adjudicator. Do not claim model improvement before that result exists.

## 2026-08-26 - Promote the tree-relational Transformer from matched molecular calibration

- The first approved exact-H100 preflight failed before scientific work because the Modal upload
  omitted a hash-pinned transitive config input,
  `configs/multireaction/common_ugi_baseline_protocol_v1.json`. The runtime now derives one bounded,
  authenticated upload closure from exact `path`/`sha256` config pins and uses that same closure for
  planning and transfer. Forty-three focused runtime/calibration tests passed; no dataset, model,
  sampler, seed, assessor or scientific gate changed.
- The corrected two-stage preflight completed and independently verified as run
  `087a5fedb957b44acff63d3ee1c51798005c2c872715fbd99b024ff2305ae75c`. All five approved full jobs
  then completed and independently verified with executable source SHA-256
  `a632c678e42132c2913364b6f64b15b9b7785a43683e8662c16dbfdde80d8af9`: dense corrected
  `f37fa5cb2e0607d891f38d1e2f59d5fccb734ab9526b5d747a48f6d0a2e6fa87`, tree relations and routing
  `a225f5c6157df76bfd65e3652f443703e6ef0602e74ab63f9bba1132a8a719fe`, tree plus balanced
  consistency `2b6b3058d6223f68c906d487671d68f6b2eaa7765a02120fa9f0813d668418a8`, full component masking
  `68f11969b3b22dc72e093a899774282574d469c7b945ff7092107ed838d11eff`, and frozen v0 reference
  `7fe0144146697bba82879a57ed646f0bbeb2ab331c029aa908ee4e3071c62336`.
- The frozen 10,000-resample paired adjudicator selected exactly one of 40 Transformer checkpoints:
  `tree_relations_and_routing:step_2700`. On the 512 calibration programs, its primary unique
  open-ended, whole-product-novel exact-L1 yield was 0.8633 versus 0.8301 for v0, an absolute paired
  improvement of 0.0332 (95% bootstrap interval 0.0039 to 0.0410). It retained validity 0.9863,
  exact-L1 0.9785, local support 0.9492, role support 0.4805, tail support 0.5820, component novelty
  0.8922 and 87.2% of v0 effective component count. Every frozen promotion and noninferiority check
  passed. Higher raw-novelty early checkpoints failed at least one prespecified support gate and were
  not selected.
- The attributable adjudication is
  `results/phase1/ugi_tree_transformer_calibration_adjudication_v1/result.json` (SHA-256
  `4dccd7e2b3c704e86757be4542cafa3dd5e78e4af6978e8e3c356ae185e90f02`); the run ledger is
  `results/phase1/ugi_tree_transformer_calibration_execution_v1/result.json` (SHA-256
  `99a8972e97ead7c75a7cacef35f9dac090d179023ee091833b64d9786c796a0a`). Final verification
  authenticated all 30 vendored assets; 45 focused tests passed and seven skipped. The locked full
  suite reproduced exactly the 189 frozen missing-artifact failure nodes with no additions or
  removals. `make test` remains environment-blocked because it invokes Homebrew Python 3.14 without
  pytest; that target executed no test and was not treated as a pass.
- **Decision:** promote the tree-relations-and-routing architecture at checkpoint 2700 to the next
  Ugi-only evaluation stage. This is a calibration-based architecture/checkpoint decision, not a
  held-component or final production result. It does not establish synthesis success, route closure,
  biological activity, procurement, or seed stability.

## 2026-08-26 - Freeze the fresh-seed Ugi tree-relational production evaluation

- Froze the calibrated `tree_relations_and_routing` architecture at step 2700 before production
  training. The production contract uses three fresh independent training seeds (`20260905`,
  `20260906`, `20260907`), exactly 2,700 steps and 345,600 realism-weighted train-fold draws per
  seed. Calibration or held-out rows cannot select a later checkpoint, architecture or threshold.
- Each seed is evaluated once on the same ordered 3,072-program component-family stress draw with
  eight stochastic reverse-flow steps, terminal temperature 1.0 and no repair or retry. Because this
  draw appeared in earlier diagnostics, held-component strata are descriptive rather than a pristine
  final test. Conditioning on held-family morphology does not establish regeneration of the exact
  unseen component identity.
- Froze aggregation by independent training seed, reporting the mean and sample standard deviation
  across three seeds. Molecule attempts are not treated as independent replicates. Prespecified
  validity, exact-L1, open-ended-yield, novelty, local-support, role-support, tail-support and
  effective-component-count gates classify either a qualified production result or a completed
  negative result; they never trigger reselection.
- The descriptor is doctor-ready on exact `H100!` and all 30 vendored inputs verify. Fifty-six
  focused training, sampling, evaluation, aggregation and runtime tests pass; seven tests skip only
  because the legacy 421-record expanded-chemistry fixture is not materialized. The full 1,877-node
  suite reproduces the exact 189-node historical missing-artifact failure set recorded in the
  repository, with zero new failure nodes. Ruff and `git diff --check` pass.
- The attributable readiness record is
  `results/phase1/ugi_tree_relational_production_readiness_v1/result.json` (SHA-256
  `2f8f1451cb40e282a46ba6727dc90d2232df8e3b8023aa8db3c596cc51004b57`). Its executable source
  SHA-256 is `f7941230d416b0c170baabb84079314bc933c0e86fa88b4f594e7b11b9b09c65`.
- **Decision:** paid compute is not authorized by this entry. After explicit private-source-transfer
  and paid-compute approval, run one exact-H100 smoke preflight. Only if it independently verifies,
  launch the three fresh-seed full jobs in parallel and apply the frozen aggregate adjudicator. This
  readiness result is not evidence of final Ugi performance, synthesis success, route closure,
  procurement or biological activity.

## 2026-08-26 - Fail the first production preflight on an inherited split label

- The approved exact-H100 smoke request
  `b0c1902357546a698ecb7635fec7b12dffaab7fcbbce7619939440f0a46a18db` failed closed in the training
  stage before any optimizer step or scientific evaluation. No full production job was launched.
  The derived production config disabled calibration-based selection but inherited the base
  calibration config's `development_split` label, which the existing trainer correctly rejects
  unless calibration early stopping is active.
- Corrected only the derived partition contract to the trainer's existing `fixed_train_only` mode
  with `fixed_final_step` selection. The train fold, architecture, selected duration, seeds, sampler,
  assessors, attempt budgets and scientific gates did not change. The production regression test now
  calls the runtime partition validator directly, so a builder-only test cannot miss this mismatch.
- The failure receipt is
  `results/phase1/ugi_tree_relational_production_preflight_attempt1_v1/result.json` (SHA-256
  `a77d4991abb0c1a9ba805796358cbddb85baacbdcf4be0f77a5d2ec94c05e026`). The corrected readiness
  receipt is `results/phase1/ugi_tree_relational_production_readiness_v2/result.json` (SHA-256
  `86ed4c50fdd97f3574101afc66e48da14e3fd2e0db1ae00d1ef628f16845b6a2`), with executable source
  SHA-256 `94bb0384b14accf8cd8bce866a3b43d3d1707c471abf6e9b64fb6f035479811c`.
- **Decision:** the first paid preflight authorization has been consumed by the failed attempt. Do
  not infer permission for a second paid H100 attempt. Obtain fresh approval for the corrected smoke
  retry and launch the three full fresh-seed jobs only if that retry independently verifies.

## 2026-08-26 - Complete the fresh-seed Ugi tree-relational evaluation as a negative result

- The corrected exact-H100 smoke run
  `9c8579eb1a36a3f2756a0c2823994df6b35aeca52d616361d3fe11088894cc12` completed and independently
  verified. The three authorized full runs then completed without restart and independently verified:
  seed `20260905` run `2735dbfb4276ba21a792808f954a76607158e1ac29df3990422c0b63e7b96c81`,
  seed `20260906` run `0bbd58b8e87a344861f7b6b8c88843c368d429adb2a1be328324e895991bc0d8`,
  and seed `20260907` run `779a97f358b334ddfd30e70006116fe9619d334aec573ebce3437da4d8603975`.
  Every run used executable source SHA-256
  `94bb0384b14accf8cd8bce866a3b43d3d1707c471abf6e9b64fb6f035479811c` and contributed exactly
  3,072 attempts.
- Across independent training seeds, validity was `0.9571 +/- 0.0047`, exact-L1 yield was
  `0.9365 +/- 0.0116`, and local-support-qualified exact-L1 yield was `0.8631 +/- 0.0157` (mean and
  sample standard deviation). These missed the frozen mean validity floor `0.97`, frozen mean
  exact-L1 floor `0.95`, and per-seed local-support floor `0.88`; seed `20260906` also missed the
  per-seed exact-L1 floor `0.93` at `0.9287`. No threshold is relaxed after observing the result.
- The main open-endedness and support-retention checks did pass: unique open-ended exact-L1 yield was
  `0.8239 +/- 0.0113`, component novelty was `0.8916 +/- 0.0228`, role-supported exact-L1 yield was
  `0.5372 +/- 0.0503`, tail-supported exact-L1 yield was `0.6026 +/- 0.0473`, and mean effective
  component count retained `0.9277` of the fixed historical v0 reference. Whole-product novelty to
  train was `0.9269 +/- 0.0131`. The held-component result (`20.51 +/- 2.78` exact-L1 products per
  1,000 attempts) remains descriptive because the stress draw was previously used.
- The attributable aggregate is
  `results/phase1/ugi_tree_relational_production_aggregate_v1/result.json` (SHA-256
  `2bdf30b4ad6d738988cb72d6a2170c63825a7d39513d028580e4bee3c93724a7`). It authenticates every
  run/evaluation input and treats training seed, not molecule row, as the replication unit.
- **Decision:** classify the frozen experiment as `negative_full_ugi_evaluation`. The Transformer
  preserves the intended open-ended generation, novelty, role support, tail support and component
  diversity, but it is not production-qualified because constitutional validity, exact Ugi replay
  and local chemistry support are not yet stable enough across seeds. Do not promote this result by
  changing thresholds or selecting the best seed. Any follow-up must be a separately frozen
  diagnostic or new model/sampler experiment.

## 2026-08-26 - Attribute the Ugi tree-relational production failures before retraining

- Ran the frozen, non-selecting failure-attribution diagnostic over all 9,216 saved production
  attempts. It made no generator, synthesis-planner, proposal-engine or route-oracle calls and did
  not alter any production gate. The attempt-level join was authenticated against the three run
  manifests, evaluation-detail archives, aggregate result and pinned local-chemistry policy.
- The mutually exclusive outcomes were 7,954 exact-L1 locally supported products (`86.3064%`), 677
  exact-L1 products with unsupported local chemistry (`7.3459%`), 280 invalid terminal-support
  outcomes (`3.0382%`), 190 valid native-forward-exact products on which the method-blind retro
  assessor abstained (`2.0616%`), and 115 molecule-sanitization failures (`1.2478%`). Thus the
  largest nonpassing class is unsupported local chemistry, while terminal feasibility accounts for
  280 of 395 invalid attempts (`70.8861%`).
- All valid saved samples reconstruct exactly through their native component traces. The 190 valid
  exact-L1 misses are therefore assessor-coverage failures rather than failures of native forward
  assembly. They concentrate on multi-nitrogen heads and ambiguous native forward outcomes: the
  abstention rate rises from about `0.48%` for one- or two-nitrogen heads to `50.0%` for four
  nitrogens and `82.6%` for five nitrogens; no product with one native forward outcome abstained.
- The 677 locally unsupported exact-L1 products contain 898 unsupported edge occurrences. The
  dominant product-level signature is an N--O single bond (462 products; `68.24%` of this class),
  followed by C=N double (106), F--N single (49), O--S single (47), and N--S single (31). These
  signatures were recomputed from the pinned policy rather than trusted from stored labels.
- Terminal-support failures are associated with the sampled head topology: all 280 occurred when
  the amine-head cycle rank was one. The saved attempts do not preserve the exact terminal decoder
  exception subtype, so distinguishing atom-state, bond-state, aromatic-state and sanitization
  causes requires instrumentation in a new diagnostic run; it cannot be recovered honestly from
  these archives.
- The attributable result is
  `results/phase1/ugi_tree_relational_failure_attribution_v1/result.json` (SHA-256
  `d6e5ef6adf29d7aba98545fbba9cb515e7570e734f6a9bb8dc8af494b181f70a`); its complete attempt ledger
  is `results/phase1/ugi_tree_relational_failure_attribution_v1/attempt_attributions.jsonl.gz`
  (SHA-256 `068af1ed562004a71276f501f74a131b228afa108501e361d74c72b2b328c9ce`). A clean rerun produced
  byte-identical files. All 30 vendored inputs verify, focused tests pass, and the full suite
  reproduces exactly the recorded 189-node historical missing-artifact failure set with zero new
  failures.
- **Decision:** do not launch an unchanged retraining run. First preserve exact terminal decoder
  failure subtypes, then qualify role-local support masks or losses for the dominant unsupported
  bond signatures without introducing a finite component vocabulary, and independently test a
  broader method-blind Ugi retro policy against the native forward traces. Only after those bounded
  interventions are frozen should a new sampler or training experiment run.

## 2026-08-26 - Qualify typed decoding and a bounded role-edge intervention without retraining

- Added typed terminal-decoder failures that retain the failing stage, stable reason code and local
  node or edge context. Molecule-construction failures now separately retain an aromaticity,
  valence, sanitization, topology-contract or unclassified runtime subtype. Neither path repairs a
  molecule, retries a draw or removes a failed attempt from the denominator.
- Refactored the Ugi reverse adapter to expose transform-consistent reverse/forward candidates
  separately from registry-qualified exact-L1 traces. On all 190 valid native-forward-exact
  production attempts where the method-blind exact-L1 assessor abstained, the native component
  trace was recovered by the transform-consistency diagnostic but failed the frozen registry handle
  policy. There were zero admissible exact-L1 promotions. Rejections included 117 three-site, 21
  four-site and 6 five-site amine heads, plus 47 two-aldehyde-site components; some products fail
  more than one role. This is a substrate-scope result, not permission to broaden the registry.
- The training-fold-only v2 role-local policy detects all 677 previously attributed unsupported-edge
  exact-L1 products, contains 27 role-edge and 12 role-cycle signatures from 66,464 training
  records, and stores no component identity. The attributable readiness result is
  `results/phase1/ugi_tree_relational_intervention_readiness_v1/result.json` (SHA-256
  `965117522442807f42a9c6b33ab5a8a35ecfc9997934e675636b9b7916a1c82c`); its retro ledger is
  `retro_attempts.jsonl.gz` (SHA-256
  `f9b51aeea7386db2aa818b9c38f2d10bc7d8e184aa14b7e478f1a303b1e3e206`).
- A first 32-attempt paired CPU smoke applied role edges, complete role-cycle signatures and
  component hard bounds together. It removed unsupported chemistry but reduced locally supported
  exact-L1 products from the paired original 30/32 to 28/32 by rejecting two otherwise locally
  supported cyclic-head products. Preserve this broad-scope smoke as an over-constrained negative
  result; its result SHA-256 is
  `2bf6619e9f07d50173acb769758d9072738b18d23df840f846b83f486830ca3f`.
- Froze a narrower follow-up that masks only exact training-supported role/element/bond signatures,
  the failure axis responsible for the dominant N--O and other exotic bonds. On the same 32
  programs and random streams, it produced 31 valid products, 30 registry-qualified exact-L1
  products and 30 locally supported exact-L1 products. It therefore retained all 30 paired-original
  locally supported products while reducing locally unsupported exact-L1 products from 2 to 0.
  N--O and O--O product fractions were both zero. The remaining terminal failure is now precisely
  recorded as `role_local_bond_state_unavailable` on an amine-head cycle edge. The attributable
  result, including executable-source pins, is
  `results/phase1/ugi_tree_relational_edge_constrained_resampling_smoke_v3/result.json` (SHA-256
  `f08e53bfeecf52c719be409a9672e1874ff7f732d2679e4570f4ccd6924462a2`). Its sampling ledger is
  byte-identical to the clean reproduction (SHA-256
  `3248864107446ecfa913058057a825d10a7a18141a377fa957d2ad1178beed26`). The earlier v1 result is
  superseded because it omitted explicit executable-source pins; v2 is superseded because the final
  renderer-free import boundary and blinded-runner dependency manifest were applied afterward.
- Verification closed with all 30 vendored assets authenticated, 30 focused intervention and import
  tests passing, no Ruff or whitespace errors, and the repository-wide test cache restored to the
  frozen 189 expected failures with zero new or unexpectedly resolved failures.
- **Decision:** keep the completed three-seed production result negative and keep the registry
  exact-L1 gate unchanged. Do not retrain yet. The edge-only frozen-checkpoint sampler has passed a
  paired structural preflight and is ready for one full 3,072-attempt seed-0 diagnostic on exact
  H100 compute. That paid run requires explicit authorization; it is not launched by this entry.

## 2026-08-27 - Preserve the full role-edge constrained resampling as a negative result

- Ran the explicitly authorized 3,072-attempt seed-0 frozen-checkpoint diagnostic on one exact
  `NVIDIA H100 80GB HBM3`. The downloaded run
  `phase1-ugi-tree-relational-edge-constrained-resampling-seed0-h100-v1/7b5a8d26bf75f9579e6bad7be39f57d32e41b9584fcecc56368187057838f8e9`
  passed independent manifest verification. Its source SHA-256 is
  `e02120a4ca762dfe3c21e159d9a08869216ed482fdc85b7fa74ecd939376a6c7`, and its result and diagnostic
  archive SHA-256 values are respectively
  `08728a9158b3efcd7689a396a9b7d85fbdb77df58789fd6a637b3ba35381347d` and
  `07e57aab0d2ff835c33edbe3e70a35d6e4366c9cbcc04b9ff69520d52ea2371f`.
- The constrained sampler produced 2,935 valid products (95.54%), 2,875 exact-L1 products (93.59%)
  and 2,843 local-support-qualified exact-L1 products (92.55%). Relative to the same 3,072 original
  attempts, it retained 98.53% of raw exact-L1 products and increased local-support-qualified
  exact-L1 products from 2,664 to 2,843. All attempts remained in the denominator; there were 112
  typed terminal failures and 25 typed molecule-sanitization failures.
- N--O and O--O product fractions were both zero, but one small three- or four-membered oxygen-ring
  product remained. More importantly, 32 exact-L1 products (1.04% of attempts) still failed the
  independently frozen broader local-chemistry assessment. The zero-unsupported-product gate
  therefore failed. The 32-attempt CPU smoke did not expose this residual at production scale.
- Structural diversity remained high (mean pairwise ECFP4 distance 0.675) and 90.43% of valid
  products were novel to the training set, but the realism C2ST AUC was 0.960, so the generated and
  observed lipid distributions remain readily distinguishable.
- **Decision:** preserve this run as `negative_constrained_sampling_result`. Do not relax the local
  support gate, rerun the same job, or promote the constrained sampler. The result localizes the next
  engineering question to the mismatch between the role-edge generation mask and the broader
  method-blind local-chemistry assessor; it does not justify another blind retraining run.
- Verification authenticated all 30 vendored assets, passed the 22 focused experiment, Modal and
  architecture tests, passed Ruff and whitespace checks, and returned the complete suite to the
  exact frozen 189-node missing-artifact baseline with zero new or unexpectedly resolved failures.

## 2026-08-27 - Freeze source-linked BL/LX reaction-enumerated structural support

- Added a common component-family splitter used by Ugi and the BL/LX expansion. Components are
  clustered and assigned to train, calibration or heldout before any product enumeration. The
  `amine_head` namespace is shared across BL and LX to prevent cross-reaction leakage; BL acrylate
  and LX aldehyde repeat-component families remain reaction-specific.
- Preserved all 764 exact source-executed products and added a distinct
  `admit_transform_consistency` structural-supervision disposition. Computed products can supervise
  whole-graph and exact semantic-coordinate learning, but are never promoted to observed synthesis,
  synthesis success, L2/L3 route closure, procurement or biological evidence.
- Enumerated 24,231 source-linked component pairs with exact registry-backed forward programs. The
  build admitted 22,187 new transform-consistent products and abstained on 1,434 attempts: 1,260
  produced other than one constitutional product, 143 exceeded declared model support, 15 had
  ambiguous atom origins and 16 represented eight products with multiple component factorizations.
  Another 610 computed attempts duplicated source-executed products and were superseded by the
  stronger source evidence.
- The final corpus has 22,951 products and 1,461,956 exact semantic-atom rows. BL has 4,757 train,
  2,350 calibration and 1,551 heldout products; LX has 6,904 train, 3,372 calibration and 4,017
  heldout products. Within every program and fold, source-executed and computed strata each receive
  0.5 total mass, with inverse component-family-size weighting. The model loader tensorized every
  record without truncation at the declared 194-heavy-atom support and normalized each fold's
  weights to one.
- The original source-corpus files and fold assignments remain frozen. The expanded structural-
  family contract changes 443 source-product fold labels in a new versioned split rather than
  rewriting the older exact-component split.
- Run `8836d21ced6526431f779cfe3209455d4fd31574652def2a8ce5bb5ef538fd37`
  passed manifest verification and an independent strict reproduction with byte-identical outputs.
  The attributable result is
  `results/phase1/bl_lx_reaction_enumerated_expansion_v1/result.json` (SHA-256
  `0574e4c322fc011b350f7cc2c8baaefa3bd121de36f0bcf32d3721e8ace47cc6`). The configuration SHA-256
  is `ea62ab4ead3f25adedaa4854555a493432ec290154d35a076a61e996e1818999`.
- **Decision:** use this source-balanced expansion as the versioned BL/LX structural-training input
  for the next model experiment. It closes the data-volume and loader-readiness question, not model
  performance. Do not launch paid training or claim multi-reaction generalization from this corpus
  build alone.

## 2026-08-27 - Freeze heterogeneous BL/LX reaction-enumerated support at demonstrated capacity

- Added deterministic mixed-repeat enumeration so one BL or LX program can use distinct
  source-linked repeat components at different reactive sites. Component structural families are
  assigned before product construction, product folds retain heldout-precedence, and homogeneous
  tuples remain in the earlier evidence layer.
- Preserved the same exact one-product forward-replay, constitutional support, semantic-origin and
  enumerated-set factorization gates. No component identity enters model conditioning, no biological
  label is inherited, and computed rows remain reaction-enumerated support rather than observed
  synthesis, synthesis-success or route-closure evidence.
- A frozen Ugi-raw-count target failed closed. BL/train admitted 35,217 uniquely factorized mixed
  products against 61,707 required additions. The attributable negative result is
  `results/phase1/bl_lx_mixed_repeat_capacity_v1/result.json` (SHA-256
  `b99c483f16326490877a41db36c9a684869d4a30e0e74ab6571ab6831890fcf7`). The factorization gate
  was not relaxed.
- Froze an exposure-balanced replacement with 30,000 train, 8,000 calibration and 8,000 heldout
  products for each of BL and LX. The final 92,000-product corpus preserves 22,951 v1 products and
  adds 69,049 mixed-repeat products from 187,337 fixed attempts. It contains 194,126 exact program
  steps and 5,563,294 exact semantic-atom rows.
- Source-executed, homogeneous computed and mixed computed rows receive 0.5, 0.25 and 0.25 mass
  within every program/fold; inverse component-family-size weighting applies inside each stratum.
  Thus family exposure is controlled without claiming equal raw support to Ugi.
- Run `6dbe714882f4a5df448b661ace4c8e790f6f03a1dba756e4bcbc168ddfdaf6ea` passed independent
  manifest verification and strict reproduction; all nine artifacts were byte-identical. The
  attributable result is `results/phase1/bl_lx_mixed_repeat_expansion_v1/result.json` (SHA-256
  `67c464bd191cec9d0bc3e6bef84079c49316a72f465f18ddcd5b494847c0ad76`) and the production config
  SHA-256 is `2b2d96177351bddcc57fcc61ef0d66e5a55007cf2cb7b1fd86889cbd1882f7a5`.
- Verification authenticated all 30 vendored assets. The clean-cache suite collected 1,919 nodes
  and reproduced 189 known missing-artifact failures with zero new failures. Ruff, focused tests and
  mypy passed. The broader `check-core` wrapper remains blocked before testing by 13 unrelated
  pre-existing provenance drifts in the shared dirty worktree; none is an input to this corpus.
- **Decision:** use the exposure-balanced mixed-repeat corpus as the next BL/LX structural-training
  input. Do not weaken the uniqueness gate or describe computed products as observed chemistry.
  This closes the virtual-library volume question; model performance still requires a new versioned
  training experiment and is not established by corpus construction alone.

## 2026-08-27 - Qualify the mixed BL/LX corpus against model support and close the local training handoff

- The first full shared-representation census of the 92,000-product mixed BL/LX corpus failed
  closed. It identified 2,185 homogeneous computed products outside the frozen Phase 1 model
  support: 2,107 exceeded three graph closures and 78 used atom states absent from the pinned
  vocabulary. The representation gate and vocabulary were not relaxed.
- Added a versioned model-support-qualified rebuild that applies the pinned atom vocabulary and
  three-closure limit before admission, records every exclusion, and deterministically refills the
  existing 30,000 train, 8,000 calibration and 8,000 heldout quotas for each family. All 764
  source-executed products remain present. The final 92,000 rows comprise 20,002 supported
  homogeneous computed and 71,234 mixed computed products in addition to the source rows. The
  result is `results/phase1/bl_lx_mixed_repeat_expansion_v2/result.json` (SHA-256
  `5fa8e4f8bbc79ea4d286b1193262a9cf4bdea56a1ddeecfaf17363c387c5e980`).
- The superseding representation census passed all eight gates for 204,386 Ugi/BL/LX products and
  9,841,801 atoms, with the full 194-heavy-atom and three-closure bounds represented exactly. The
  result is `results/phase1/shared_synthesis_program_mixed_representation_v1/result.json` (SHA-256
  `426666c7f59dc6f5ec6143dc06d23cbfaef6c51af475ef20c53cdc6bbbed8ed7`).
- Derived the expanded-data design from the qualified Transformer contract. All 13 gates pass; the
  four arms, three seeds, matched compute, equal family mass, fixed-final checkpoint policy and Ugi
  retention rules are unchanged. The design remains launch-blocked. Packed all 204,386 products in
  a deterministic numeric-only 13.7 MB cache with exact folds, zero fixed-state failures and no raw
  family-count sampling. The cache SHA-256 is
  `249b92744c22109cd5994ad831555df85575bfb6bd58010dce71517e5e881114`.
- The first optimizer smoke then failed closed because the count-only sampling prior assumed every
  repeated role instance had the same atom count. Mixed products correctly violate that assumption.
  Replaced the scalar size draw with an exchangeable sorted size-multiset draw, retaining counts but
  no component identity, order, SMILES or fragment. The rerun completed two deterministic optimizer
  steps with two examples from each family, finite losses and zero fixed-state failures. Its result
  is `results/phase1/shared_synthesis_program_mixed_training_smoke_v1/result.json` (SHA-256
  `cf17c4d4a16178bbe67c3304dc6e49bfa9b4a0ff912c83ed584bf398f0525843`).
- Strict independent reproductions were byte-identical for the support-qualified corpus (run
  `bde0ca3187cb035344743320a6fdf4daddce525d769b957f0c367e3c96e74992`), representation
  (`d4aac5a22e29cf0ce8652a5901c9464d10f4756cd50bb46dde86b32c99966e1e`), derived design
  (`dbc212f574084626d9feee51e5f173ed91f18c9c7ea10e85ab615dfdc82761ef`), packed cache
  (`d003f48b609dba8ca43de7e71008535234e819ff02b2e9a73960ef363c986431`) and training smoke
  (`55b264409716252067177d30d5f164d897ae84e0e24e25cfdf52fddc3dc38802`). Vendored data
  verification passed 30/30; 58 focused tests, Ruff and mypy passed.
- The repository-wide suite still does not pass because historical result files are absent. It now
  reports the 189 frozen missing-artifact failures plus one unrelated paper-renderer expectation
  mismatch introduced elsewhere in the pre-existing dirty worktree. The mismatch is
  `test_completed_evidence_renderer_uses_only_pinned_nonselecting_results`; this task does not alter
  that renderer or rewrite the frozen failure baseline.
- **Decision:** the expanded shared Transformer pipeline is locally execution-ready. Treat the
  corpus as reaction-enumerated support and the smoke as infrastructure only. Do not claim improved
  BL/LX generation until a new production run is evaluated, and do not upload private inputs or
  launch paid exact-H100 work without explicit authorization.

## 2026-08-27 - Optimize the shared Transformer training hot path without changing its contract

- Profiled the full 5,289,901-parameter Ugi/BL/LX Transformer on a source-balanced batch drawn from
  the frozen 204,386-record packed cache. The measured hot path was attention plus the three-family
  balanced backward; packed-cache materialization and collation took a representative 1.58 ms for
  a production-size 32-record CPU batch and was not the limiting path.
- Replaced materialized attention scores/probabilities with PyTorch scaled-dot-product attention,
  while retaining the same query/key/value/output parameters and additive graph/program masks.
  Removed a per-layer CUDA host synchronization from the shared structural-attention validator; CPU
  callers retain the defensive empty-memory check.
- Rewrote repeat-component consistency to gather only admitted matched positions before evaluating
  class-valued atom and bond differences. On the pinned benchmark batch this reduced evaluated
  class-valued pair elements from 4,604,256 to 4,212 (99.91%) without changing the loss. The
  optimizer update is now launched before detached metrics cross the accelerator boundary, avoiding
  a backward-to-AdamW host synchronization bubble without changing parameters or diagnostics.
- The nine-repeat deterministic CPU benchmark improved the median forward, loss and equal-family
  PCGrad backward time from 0.33830 s to 0.30421 s (1.112x). Losses were identical, maximum absolute
  gradient difference was 4.77e-7 inside the frozen tolerance, graph-bias outputs were bit-identical,
  and standalone PCGrad gradients remained bit-identical. The attributable result is
  `results/phase1/shared_synthesis_program_training_performance_v1/result.json` (SHA-256
  `81e6247e951cda82634511e39a381e7215dad12e793e7b350e3ace8f0061390b`).
- The actual two-step training/evaluation DAG completed and verified as run
  `197d0c26c5e0962be9a8d1cf86dfb72cadf07dfd34354e6ffe83cfdd35b234f7`; an independent strict
  reproduction was byte-identical for all six published artifacts.
- **Decision:** retain these contract-preserving optimizations for subsequent training. This CPU
  result establishes implementation equivalence and a local speedup, not exact-H100 throughput.
  Keep deterministic float32, the full 194-heavy-atom support, the frozen source-balanced mixture
  and per-family PCGrad unchanged. A current exact-GPU preflight is still required before estimating
  production wall time or enabling mixed precision, compilation or alternative batch geometry.

## 2026-08-27 - Retain eager true FP32 after exact-H100 optimization qualification

- Ran two authenticated, verified profiles on an exact NVIDIA H100 80GB HBM3 using the frozen
  5,289,901-parameter mixed Ugi/BL/LX Transformer, source-balanced 32 x 4 PCGrad step and full
  194-heavy-atom support. The eager true-FP32 baseline measured 0.35855 s per optimizer step
  (351.63 examples/s), reserved 1.71 GB at peak and replayed losses and final parameters exactly.
- BF16 eager and compiled execution were slower than true FP32 and changed the three-step parameter
  update by 17.85% in relative L2. FP32 compilation was 1.092x faster but changed the update by
  18.42%; variable padded sequence lengths triggered repeated recompilation. These modes failed
  their prespecified numerical-equivalence gates.
- Increasing the microbatch from 32 x 4 to 64 x 2 or 128 x 1 was 1.349x and 1.320x faster,
  respectively, but changed where per-family PCGrad projection occurs and changed the update by
  about 24.5%. These are different scientific optimizers and were not promoted.
- At the user's explicit request, separately qualified TF32 rather than assuming it was equivalent.
  TF32 eager changed the update by only 0.46% in relative L2, but was 1.2% slower than the matched
  true-FP32 baseline. TF32 plus compilation was only 1.032x faster and inherited the 18.43%
  compile-path update drift. Neither cleared the frozen promotion gates.
- The attributable summary is
  `results/phase1/shared_synthesis_program_mixed_h100_profile_v1/result.json` (SHA-256
  `1d39ab70653998660c29525185bf8bd8d3888357b757a6484c9dc237fb70adeb`); its two verified source
  artifacts have SHA-256 values
  `b06bc0994b0ce277b487dbc76c30756d991de1a50df441a991b1c4ce203b138f` and
  `447b4ea784b366c6dd2c2d54909b0bd568c4d8240345cb0f81ae610de4a055d5`.
- **Decision:** retain deterministic eager true FP32, the 32 x 4 batch geometry, equal-family source
  balancing, PCGrad and full 194-atom support. Do not enable TF32, BF16, compilation or a new batch
  geometry: none produced a contract-preserving speedup. The next optimization target is the
  variable-shape collation/attention schedule or the PCGrad implementation itself, measured without
  changing its mathematical partition.

## 2026-08-27 - Complete recovered seed-0 mixed-family training and support-conditioned evaluation

- The authorized exact-H100 conditioned-arm run was externally interrupted at optimizer step 1,475,
  not by a numerical or model failure. Its deterministic restart checkpoint was intact. Resuming the
  same content-addressed workspace completed all 1,700 prescribed steps, 217,600 examples, five
  fixed checkpoints and zero fixed-state failures. The final model-state SHA-256 is
  `ea9afefa77fe4a473bf5911c0b63e10dfdf652aff44090ad8b009c9f3a324dd0`.
- The first evaluation then failed closed before model sampling because independently factorized BL
  role-size draws could combine into a graph larger than the declared 194-heavy-atom support. This
  was a count-prior support bug, not evidence about the trained model. Replaced rejection-prone
  independent draws with the exact factorized role-size product law conditioned on total size at
  most 194. This performs no clipping, repair, retry, attempt dropping, component lookup or heldout
  access.
- The corrected preflight materialized all 16,896 frozen layouts across 18 evaluation cells. Every
  layout was in support; the observed maxima were 194 atoms for BL, 166 for LX and 63 for Ugi. The
  local and remote preflight results were byte-identical with SHA-256
  `8ec7aa67c2875a6e3f5bde26bca26e0aa57fc20840eb4452ab7451f2047f6677`.
- Evaluation-only run `f06674c61e3019aa5c54dbefb8e44fb1c36bacc7fffda9235e64a713eedc3d5f`
  completed on the recovered checkpoints with zero training, repair, retry, route, oracle or
  candidate-selection calls. At the fixed final checkpoint and 3,072 heldout attempts per family,
  exact-L1 yield was 70.41% for BL, 30.34% for LX and 72.01% for Ugi. Raw validity was 70.48%,
  40.98% and 89.36%, respectively. Exact forward-replay precision was 100% for every family;
  decomposition coverage was 99.91%, 74.03% and 80.58%.
- The full attributable evaluation result has SHA-256
  `c9dff27cfe2e4c010456828d9d93664380727da1d5d9671977b50676fe550217`.
  The compact result is
  `results/phase1/shared_synthesis_program_mixed_primary_seed0_v1/result.json` (SHA-256
  `8c9123decbc53df1d0ee215d1bf26dde7b315a4e3741738e62070ec934673859`).
- **Decision:** seed 0 is a positive execution result for Ugi and BL, with exact-L1 yield near 72%
  and perfect admitted forward replay. LX remains the limiting family at 30.3% exact-L1 yield.
  Do not report across-seed variance, robustness or baseline superiority until matched seeds 1 and
  2 and the prespecified comparisons run under the same support-conditioned layout contract.

## 2026-08-27 - Admit full role-local morphology as a bounded Transformer projection ablation

- Audited the frozen v0-versus-mixed comparison and identified an information asymmetry: v0
  receives per-role exterior size, junction budget, cycle rank and core-attachment count, whereas
  the frozen mixed Transformer receives role layout sizes and only a global closure count. This is
  a confound in any claim about the GRU and Transformer backbones themselves.
- Added four node-aligned, categorical role-local program coordinates to the shared Transformer
  memory. Zero is an explicit unconditioned state and observed nonnegative integers use `value + 1`,
  so a conditioned zero-junction role is distinguishable from missing conditioning. The coordinates
  contain no component identifier, molecular fragment, fingerprint or finite component vocabulary.
- The values are derived deterministically from exact training graphs. The factorized sampling prior
  now draws role morphology conditional on role sizes and conditions the finite product law on the
  existing three-closure support without clipping, repair, retry or attempt dropping. Frozen models
  remain compatible and ignore the new states unless `role_morphology_conditioning=true`.
- Across the 204,386-record mixed cache, observed maxima are 160 exterior atoms, 29 junction units,
  three cycles and eight core attachments; the embeddings retain the already declared 194-atom and
  three-closure architectural support. The full intervention has 5,442,541 parameters versus
  5,328,685 for the frozen repeat-aware reference.
- Local run `e96e19e170922e23e626e6d5f8cd0742c75711c7247984d262531f44393971e2`
  completed the two-step balanced Ugi/BL/LX training, strict checkpoint reload and Ugi sampling DAG.
  All execution gates passed, fixed-state failures were zero and route, oracle, repair, retry and
  candidate-selection calls were zero. The result SHA-256 is
  `a15cc924bc4f9bc26df5efa99f5651f89f7fb47179cf8c92b031b55fe583de4e`.
- **Decision:** run one exact-H100 seed-0 intervention under the unchanged 1,700-step, 32 x 4,
  equal-family PCGrad schedule and compare it on the same ordered 3,072 Ugi programs. Treat this as
  a projection ablation. Do not attribute an improvement to the Transformer backbone alone or
  claim across-seed robustness unless the full-information result first passes and is replicated.

## 2026-08-28 - Retain output-exact Ugi sampling and assessment optimizations

- Profiled one frozen 3,072-attempt assessment before changing the implementation. Wall time was
  47.96 s; the dominant cumulative costs were rebuilding Ugi identity references (16.88 s),
  rebuilding the observed-lipid realism reference (13.31 s) and the frozen C2ST (10.18 s). Exact-L1
  decomposition itself used only 1.34 s. Added content-addressed, in-process caches keyed by the
  resolved input paths, file SHA-256 values and complete frozen policy/role tuple. A second matched
  arm then completed in 19.18 s with every selected metric unchanged.
- Tested three-way threaded assessment as an explicit alternative. It regressed cold and warm wall
  time to 129.26 s and 62.55 s because RDKit and sklearn oversubscribed the available CPU. This
  negative result is retained; assessor execution remains sequential.
- Removed repeated CUDA synchronization from fixed-state audits, kept strict terminal decoding on
  CPU after one batched logits transfer, eliminated per-sample CPU-to-GPU-to-CPU terminal copies,
  restored adapter-fixed states in place under inference mode and reused per-batch time tensors.
  The batch size, RNG stream geometry, 32 denoising steps, strict decoder, support and all scientific
  gates remained fixed. A local 64-program checkpoint comparison reproduced the complete sample and
  receipt rows exactly; the final in-place change reduced its CPU wall time from 17.69 s to 16.11 s.
- Repeated the gated two-checkpoint, 3,072-attempt-per-checkpoint experiment on an exact NVIDIA H100
  80GB HBM3. The post-preflight comparison interval fell from 369 s to 244 s (1.512x, 33.88% less
  wall time). More importantly, the complete comparison result and detail archive were byte-identical
  to the prior run, with SHA-256 values
  `9a3eda0a90adfc81dcd563506b5a7bf1bfd3225f963d1b77257f46eea6762028` and
  `3d327b7f8d589f22a7deef5a79c9de3e3259ba0ad4f663d8e22fd5707611b481`.
- The attributable benchmark is
  `results/phase1/ugi_sampling_evaluation_performance_v1/result.json` (SHA-256
  `95b24439a70be0e144d1166c3bb9d48014075cf5eb07bedddbdad48769094565`). The optimized verified run
  is `44fe4d98fd34af48deed183d718b455c09c1fe0419a0b773caf101cecf61c678`.
- **Decision:** retain the content-addressed reference caches and output-exact sampler changes.
  Reject thread-parallel assessment. Do not change inference batch geometry, precision, sample
  steps, declared molecular support or terminal policy without a separate numerical and
  distributional equivalence qualification.

## 2026-08-28 - Freeze exposure-matched shared-plus-specialist reaction models

- Audited the authenticated full-role-morphology Transformer rather than attributing its Ugi gap to
  architecture alone. Its 1,700-step equal-family schedule exposed the shared model to 68,000 Ugi,
  74,800 BL and 74,800 LX train-fold rows. The qualified v0 development checkpoint saw 384,000 Ugi
  draws (3,000 steps x effective batch 128). The mixed Transformer's Ugi path was therefore short by
  316,000 examples; it did not receive comparable Ugi training exposure.
- Added zero-initialized residual specialists after every shared Transformer block. A newly attached
  specialist is output-identical to the authenticated shared checkpoint before training. Only
  148,992 specialist parameters are trainable (2.66% of the combined model); all 5,442,541 shared
  parameters remain frozen. Specialist checkpoints contain only the delta plus authenticated base,
  cache, seed and exposure receipts.
- Froze exact cumulative exposure at 384,000 rows for each family. Ugi receives 316,000 additional
  source-balanced train-fold draws in 2,469 optimizer steps, ending with microbatches 32/32/32. BL
  and LX each receive 309,200 additional draws in 2,416 steps, ending with 32/32/16. The short final
  update is explicit; no record is added through batch padding and no heldout row is accessed.
- Added an opt-in exact program-topology decoder. It fixes only declared reaction-core assembly
  edges, confines generated interior edges to their precursor component and requires exact
  per-role exterior-size, junction, cycle and core-attachment coordinates. It contains no component
  identity or finite component inventory. Infeasible attempts abstain once; repair and retry remain
  prohibited. Legacy terminal policies are unchanged for historical comparisons.
- The authenticated CPU smoke used the real final shared checkpoint, conserved its six-row test
  exposure through a short final microbatch, produced zero fixed-state failures and passed every
  no-heldout/no-repair/no-route gate. The result is
  `results/phase1/reaction_specialist_ugi_smoke_v1/result.json` (SHA-256
  `2aed7c7730dba055e85463a73796cedd18ca6bbc5d7089ab4107accde95b87c2`); the delta checkpoint SHA-256
  is `882146c85096673eef11677cb6a166e77680c77ba8dbbf9dca8a4e57b6288f7c`.
- Froze Modal group `phase1-reaction-specialists-seed0-h100`: one exact-H100 execution preflight
  gates three independently allocated, parallel family-specialist experiments. The Ugi experiment
  is followed on its accelerator by a paired 3,072-program comparison with v0 using method-blind
  exact-L1, validity, local-chemistry, lipid-realism, diversity and novelty assessment. Ugi example
  exposure is matched; total pretraining compute and model factorization are explicitly not called
  matched.
- `make verify` authenticated all 30 vendored inputs. The 99-test focused model, sampler, training,
  orchestration and assessment suite passed. The repository-wide test cache still contains 190
  failures: 189 are inherited baseline failures dominated by 54 documented missing historical
  artifacts, and one is the already observed completed-evidence renderer mismatch outside this
  implementation. The specialist experiment is locally qualified, but the repository as a whole is
  therefore not represented as globally green.
- **Decision:** use shared pretraining plus exact family specialists as the next bounded test of the
  exposure diagnosis. Do not discard the shared Transformer, unfreeze its backbone, train on raw
  family counts, or claim superiority from the smoke. A fresh paid H100 launch still requires the
  user's explicit authorization; negative or non-inferior outcomes remain first-class results.

### Production-launch correction

- The user authorized one paid exact-H100 preflight followed by three paid exact-H100 specialist
  experiments in parallel. Initial Modal app `ap-yHRYnVNFRWOsOcSW2RtRrf` was stopped before a
  preflight receipt or production result was admitted after inspection showed that the generic DAG
  runner executes every topological stage sequentially inside one accelerator allocation. The
  descriptor's parallel metadata did not change that runtime behavior.
- Replaced that invalid execution shape with a strict Modal group contract: one standalone
  preflight experiment, followed only on exit code zero by three concurrently launched standalone
  Ugi, BL and LX experiment requests. Each member requests its own exact `H100!`; the Ugi request
  alone continues to the paired v0 assessment. Group members pin their complete experiment
  descriptors and the group planner requires one common executable source fingerprint.
- The corrected group dry-run passed with group SHA-256
  `8a1f2707a504fdc55b5419b046431d515de84f40324f1458d7c86fe1d44fe92a` and source SHA-256
  `c306cb61c0f25285fdbe82316ebceb2f227a3c9d2ddfcdd4d3d387c90e01454e`. Its preflight request is
  `e29f71d43060146481ecd94dff856aa82518dbe71ffcb188fd4f7c3711cb2d59`; independently allocated
  Ugi, BL and LX requests are respectively
  `4cf7e072e5ed0a6372be4f9f71378bcbe0ab89f449cdb94a2d6a0e98f0635fce`,
  `6d82b1221d2e8b4d37b1ffa8e56122d3425a0a326e4cb2ad6ef562c85bc564b0` and
  `7260c46a9e9b23cad5938ef89fea1d21b01f8816e9133f10a2bd6f980e6ce66e`.
  All 20 group/runtime/specialist tests and the broader 110-test model, sampler and experiment
  regression set passed; all 30 vendored inputs verified.
- **Decision:** the stopped app is inadmissible and must not be resumed. The corrected group requires
  a fresh paid authorization because the first launch consumed the prior authorization and the
  executable source fingerprint changed during the orchestration repair.

## 2026-08-28 - Preserve completed reaction specialists and recover Ugi assessment without retraining

- The user authorized the corrected exact-H100 group. Preflight run
  `34d287c7022d9b9b602ee379b676c707b23b2c943982f5dbf0c84d7f61554c4e` completed and verified,
  after which Ugi, BL and LX received independent exact-H100 allocations concurrently. Executable
  source SHA-256 at launch was
  `a42ff9bb93cb8bab7fc8d41907a2df8e840fff06e89df78ed9495c832b76b109`.
- BL run `beba255a969b06b7b730c93d8a4cd085e9e1dc973718be6b6306f3e13badb6d2` and LX run
  `cb87b261bc5d0c4cc0b6442579c4bd2e9a69ee7ce816efad34fac1d620f8cfb2` completed and verified.
  Each reached exactly 384,000 cumulative train-fold examples in 2,416 optimizer steps, with zero
  fixed-state failures, heldout access, repairs, retries, route calls or oracle calls. Their delta
  checkpoint SHA-256 values are respectively
  `42977fea0e89312bc4f7f67f9bcd7b89074a4027d28ab06422e8c734388da414` and
  `978541e440fe71a4c97106519d6c5dede9b81008d911c228180d0abab10820d3`.
- Ugi training in remote run
  `aada90e0d50f69fbbf2bc465f38c84e970ee1bd3a8610885ec1fcec071f0d0e9` also completed its exact
  384,000-example exposure contract in 2,469 optimizer steps and passed every training gate. Its
  authenticated delta checkpoint, result and progress SHA-256 values are
  `7f28c776052da19600f770b3f0341abc9514f3029b01d742069c41233a08ae08`,
  `404fa21eff57ae5aae6e8788030414921d7d1cd77c16ed5ae3e75cc5615a711c` and
  `d3a2adfbb53e7dfb297867275ae46e8ab083c0451eeb65c930b27c4497c3c725`. They are preserved under
  `results/phase1/reaction_specialist_ugi_seed0_h100_v1/`.
- The Ugi job failed only after sampling and method-blind assessment, when the decision writer
  requested the frozen per-attempt open-ended exact-L1 metric but the selected-metric projection
  exposed only its per-1,000-attempt form. Added the algebraically identical per-attempt label,
  made decision metrics config-driven and fail-closed, and added a standalone recovery descriptor
  that requires the adjacent specialist checkpoint and training result as explicit hash-pinned
  inputs. The recovery has one evaluation stage and declares zero training calls.
- `make verify` authenticated all 30 vendored assets. The 12-test focused specialist/recovery suite
  passed and the exact-H100 smoke plan resolved all pinned uploads. The repository-wide suite still
  fails on the already documented historical result-artifact gaps and stale frozen-output
  expectations; no missing evidence was fabricated and no gate was relaxed.
- **Decision:** BL, LX and Ugi specialist training are complete and must not be rerun. Run one new
  paid exact-H100 four-program recovery preflight; only if it verifies, run the full 3,072-program
  Ugi-specialist-versus-v0 assessment from the preserved checkpoint. This follow-up consumes a new
  paid authorization because the assessment source and descriptor have changed.

### Recovery-preflight result-labeling correction

- Authorized recovery request
  `f871ba6a20650cd3f07525d0e8ed71d8af4460e13237492273274e10f433ff0b` ran on an exact H100 and
  failed closed before the full assessment was launched. On the four-program smoke draw, the mixed
  specialist produced fewer than two unique valid molecules, so pairwise ECFP4 diversity was
  correctly `null`; the selected-metric projection incorrectly coerced that optional diagnostic
  to `float`.
- Optional small-sample metrics now preserve `null`, and cross-method deltas remain unavailable
  when either method lacks the metric. Metric key sets must still match, and every metric used by
  the frozen decision rule is checked separately and fails closed if unavailable. This changes no
  molecule, sampler, assessor denominator, decision threshold or trained parameter.
- The 13-test focused suite passes. The corrected request is
  `b92febec1644671ffd3724e2435f820d3cd5ba8707a1ae558f55897e3f8e3539` under source SHA-256
  `a3c82712edd2b15304801d6c712b46f32cc81eef380c97ae35b8fab09aba07cf`.
- **Decision:** the failed smoke is an inadmissible engineering preflight, not a scientific result.
  It consumed the prior paid authorization. Do not launch a corrected smoke or the full assessment
  without a fresh explicit authorization.

### Verified recovery and full Ugi-specialist comparison

- The user authorized one corrected exact-H100 recovery preflight and a gated full assessment.
  Four-program preflight run
  `acd519cd4b4ff320f2cb8ff65fb347b1b9ec979d8fd380d0a9d423e627399a13` completed and independently
  verified all nine execution gates. Only then was the full assessment launched.
- Full exact-H100 run `323ed40d62a18fa6de6012e293a828f7d1027a0b2fc6a6b40f798a4726b0b39c`
  completed and independently verified under source SHA-256
  `a3c82712edd2b15304801d6c712b46f32cc81eef380c97ae35b8fab09aba07cf`. The result and detail
  archive SHA-256 values are respectively
  `a7b9512974261b5b00c7f45d8a63005acfb1845d0e7012c4bf7dac975ce478d5` and
  `59216245cc6f7c2211684db85dfacd98a42dfb277cf9556fbb4a220c13e69c7e`.
- On the same ordered 3,072 programs per method, the exposure-matched Ugi specialist achieved
  5.31% validity, 5.27% exact-L1 yield and 3.87% local-support-qualified exact-L1 yield. Frozen v0
  achieved 97.56%, 97.17% and 95.48%, respectively. The primary unique open-ended,
  whole-product-novel exact-L1 rate was 4.95% for the specialist and 75.33% for v0, a -70.38
  percentage-point specialist-minus-v0 difference.
- The specialist had higher component novelty conditional on a successful decomposition (93.83%
  versus 78.32%) and every specialist valid product was novel to training, but these conditional
  values do not compensate for its 5.31% attempt-level validity. No repair, retry, route call,
  oracle call, candidate selection or additional training occurred.
- **Decision:** reject the shared-plus-Ugi-specialist arm under the frozen gate and retain v0 as the
  qualified Ugi reference. Exposure matching through lightweight residual adapters did not close
  the Transformer validity or exact-L1 gap. Do not launch more specialist seeds or BL/LX evaluation
  solely from this result; first attribute the Ugi failure to topology decoding, adapter capacity,
  or incompatibility between frozen shared features and the specialist delta using the saved
  attempt ledger.

## 2026-08-28 - Qualify an explicit topology-coupled Transformer correction

- The strongest loose-decoding full-role-morphology Transformer achieved 81.22% exact-L1 yield on
  the frozen 3,072-program Ugi draw, versus 97.17% for v0. Inspection identified a factorization
  mismatch: v0 learns child counts and constructs a program-feasible forest before decoding
  chemistry, whereas the mixed Transformer predicted every parent and closure independently and
  checked the requested topology only after decoding.
- Added a child-count head with direct exterior-offspring cross-entropy and role-local
  junction-budget consistency. Added one exact Ugi topology conditional that conserves exterior
  size, junction count, cycle rank and core attachments under the pinned 5/6/7-member ring,
  heavy-degree and role-local branch-spacing support. Chemistry is then predicted in a second
  Transformer pass conditioned on that topology. This selects no stored component or fragment and
  does not repair or retry a completed product.
- The exhaustive CPU support preflight constructed exact requested morphology for all 3,072 frozen
  programs: 1,040 zero-cycle, 1,845 one-cycle, 186 two-cycle and one three-cycle row, with zero
  failures or mismatches. The result is
  `results/phase1/ugi_transformer_topology_support_preflight_v2/result.json` (SHA-256
  `b7721f4231f11942634ad5de5421c467924d324fcc50129bbdf7ed80e396e9f2`).
- The authenticated CPU training smoke passed every gate. It updated only 149,764 Ugi adapter plus
  child-count-head parameters while keeping 5,442,541 shared parameters frozen, conserved its exact
  eight-example exposure, produced zero fixed-state failures and reduced its two-step smoke loss
  from 4.0050 to 2.5642. Verified run ID is
  `d3e95ace6c9e8fd8a17bab909a70e69119b9938ce02d4f8321fcb187272f8a13`.
- Froze one fail-closed exact-H100 sequence: execution preflight, exposure-matched seed-0 topology
  specialization to 384,000 cumulative Ugi examples, then one paired 3,072-program comparison with
  v0 using the existing method-blind assessors. The local readiness receipt is
  `results/phase1/ugi_transformer_topology_correction_v2/result.json`.
- `make verify` authenticated all 30 vendored assets and the 53-test focused Transformer,
  topology-decoder, specialist, sampler, experiment and evidence suite passed. The repository-wide
  suite remains non-green with 191 cached failing/error node IDs dominated by missing historical
  result artifacts; the only architecture-boundary failure is a pre-existing untracked legacy
  experiment-group descriptor. No topology-correction test appears in that failure cache, and the
  repository is not represented as globally green.
- **Decision:** the implementation and full frozen program support are qualified locally, but the
  scientific gap is not yet called fixed. A paid H100 launch still requires explicit authorization.
  Claim closure only if the paired production assessment actually matches or exceeds the frozen v0
  reference without weakening validity, local-support, novelty or no-repair gates; preserve a
  negative outcome if it does not.

## 2026-08-28 - Replace frozen topology specialization with matched end-to-end shared retraining

- The user chose to integrate the transferable v0 biases into the shared Ugi/BL/LX Transformer and
  retrain the complete model, rather than continue with a frozen shared backbone plus a Ugi-only
  topology specialist. The new production model has 5,593,463 trainable parameters; no shared
  parameter is frozen and no reaction-specific GRU is attached.
- Added smoothed program/role-conditioned atom and bond noise marginals with program and global
  backoff. Every atom and bond state retains strictly positive support in every cell. The production
  experiment contains a matched global-source control because the program/role source may be
  retained only if it improves held-component generation without component-catalog collapse.
- Added equal-present-role chemistry loss, direct exterior child-count and junction-budget
  objectives, program-routed terminal-node chemistry and closure output heads, and a second
  chemistry prediction pass conditioned on the exact training topology. These mechanisms are
  program-generic and use role, topology and reaction-program coordinates; they do not select a
  predefined head, linker, tail, fragment or component.
- Froze seed 0 at 9,000 optimizer steps, effective batch size 128, equal Ugi/BL/LX family mass and
  checkpoints 100/500/1,700/4,500/9,000. This gives each family 384,000 examples. The paid
  descriptor first runs a three-step exact-H100 preflight and launches the two matched production
  arms only if that stage completes.
- Registered smoke run `c4d2dd73329a89fd82515a114b7fbdfc0a0957e3882b95ee5df14e19381dce7c`
  completed and independently verified. Both arms trained, checkpointed and sampled; all fixed
  states were exact, both source arms had equal parameter counts, all program/role marginals had
  full support, and route, oracle and candidate-selection calls were zero. The hash-pinned readiness
  receipt is `results/phase1/shared_bias_end_to_end_retraining_v1/readiness.json`.
- **Decision:** the end-to-end retraining implementation is locally qualified. Do not call the v0
  gap closed, promote the program/role source, or use either arm for paper claims until the exact-H100
  preflight and fixed step-9,000 evaluation complete. Paid H100 execution remains a separate
  explicitly approved action.

## 2026-08-28 - Preserve the failed shared-bias H100 preflight and correct its scope binding

- The user authorized one paid three-step exact-H100 preflight followed, only on verification, by
  both seed-0 9,000-step end-to-end arms. Modal app
  `ap-92X4FDC8pNaV51h3LNySnF` ran request
  `700d92de4f0efb90e13fa66de64ec04e319b0e2eec25b98578e8f7e1d08c9a9a` under source SHA-256
  `29713623b41cabcaafab39ac19c0f074bb0018c6db9c55c94a53174877945706`.
- The preflight trained both three-step arms but failed closed before completing evaluation because
  the generic evaluator interpreted every `full` device profile as a production-quality evaluation
  and rejected the intentionally bounded preflight sampling budget. This was experiment wiring, not
  a model, memory, CUDA, or scientific failure. The production stage did not launch.
- Corrected the evaluator to bind reduced full-device budgets only to the explicit
  `h100_preflight` execution scope. The `production` scope still rejects any deviation from 32 flow
  steps, 512 calibration samples, 3,072 held-out samples per supported program, or the uncapped
  component-disjoint fold. An explicit regression test confirms that substituting the preflight
  budget into production still fails.
- The 18-test focused evaluator/preflight suite passed, formatting and lint passed, and corrected
  local smoke run `33098fe9cb4c01397c503a079104195e018ab3266a9d3c7a6a13a79a02035011`
  completed both arms with a verified stage-manifest SHA-256 of
  `dfc9e021a35267982ed44df414bb53ed7303e7add28ae0250e92afe1a6f2ffb5`.
- **Decision:** preserve the failed preflight as an inadmissible engineering result and do not launch
  production from it. The corrected retry is frozen as request
  `c96e086fc971c392acd8386d62e5e783aa705be19b7aa3535b7ab81c050088ef` under source SHA-256
  `ed0cb774636583adf5799d70850ea1766d721ba0762cd4b492360ce2008b8007`; because the one-preflight
  authorization was consumed, a new paid retry requires fresh explicit authorization.

## 2026-08-28 - Bound the assessor's classifier thread pool and reuse assessment work

- Built one reproducible CPU assessment benchmark before changing anything: `make
  assessment-benchmark`, i.e. `PYTHONPATH=.:tools python3 -m forge_maintenance
  assessment-benchmark`. It derives three deterministic 3,072-attempt ledgers from the hash-pinned
  Ugi corpus and constitutional R0 - 2,000 corpus products, 800 observed lipids, 64 disconnected
  products, 32 unparseable products, 88 native invalid and 88 native failed attempts, interleaved by
  a seeded digest - runs the common Ugi, frozen route-evidence and whole-lipid realism assessors
  over each, and emits a SHA-256 over every emitted attempt row and every metric. The ledger bytes
  are deterministic and their SHA-256 values are recorded. The local-chemistry and role-morphology
  assessors are outside the benchmark because their pinned policy artifacts are absent from this
  checkout; neither was changed.
- The cold profile attributed a 47.96 s single-ledger run as: rebuilding the Ugi identity reference
  16.0 s, the observed-lipid realism reference 13.0 s, the grouped classifier two-sample test
  11.1 s, whole-lipid descriptor vectors 11.4 s, exact-L1 reverse decomposition with exact forward
  replay 1.4 s, and pairwise ECFP4 diversity 1.2 s computed twice.
- The single largest cost was not chemistry. The frozen C2ST fits a boosted tree on 4,096 rows and
  24 features; on this 16-core host that took 27.87 s at the default OpenMP thread count and 1.51 s
  at one thread, an 18x difference spent in fork/join barriers on a problem far below the size where
  the estimator's parallel regions pay off. Its five fold AUCs are bit-identical at 1, 2, 4, 8 and
  16 threads, so the pool is now bounded through `threadpoolctl` and `C2ST_OPENMP_THREADS`. A
  process-wide `OMP_NUM_THREADS=1` was measured and rejected: it took the pre-change tree from
  80.74 s to 66.19 s but made the realism reference build slower, 9.57 s to 13.27 s.
- Retained five output-exact reuse changes. The Ugi identity reference memoizes component
  canonicalization per role, because the 112,386-row balanced corpus contains only 424 distinct
  component constitutions. The realism reference parses each selected R0 lipid once instead of once
  for its descriptors and again for its fingerprint. Internal diversity reuses the fingerprints the
  realism pass already computed. A single-program ledger reuses its overall evaluation for the
  per-program block rather than recomputing it, including its O(n^2) pairwise diversity, and that
  diversity reuses the molecules the evaluator already parsed. Exact forward replay records one
  verdict per reverse trace instead of replaying the reaction for every duplicate reverse outcome
  and re-parsing the same product each time. The lipid root rule resolves its lexicographic
  tie-break one coordinate at a time, so whole-graph eccentricity and the canonical ranking are
  usually never computed, and its traversals walk a materialized adjacency list.
- Measured by alternating the pristine and changed trees inside each round on one host and taking
  the minimum over three rounds, because the host carried unrelated user load throughout. On three
  3,072-attempt ledgers in the production thread configuration the workload fell from 132.87 s to
  44.95 s (2.96x); the cold first seed from 68.47 s to 28.98 s (2.36x) and the warm steady state
  from 32.08 s to 7.90 s per seed (4.06x). With both arms pinned to one OpenMP thread, which removes
  the classifier effect from both, the same workload fell from 49.16 s to 37.71 s (1.30x) and its
  cold start from 33.08 s to 22.28 s (1.48x). The attributable result is
  `results/maintenance/ugi_assessment_cpu_performance_v1/result.json`.
- Output identity is the acceptance criterion, not a summary comparison. All twelve runs - two
  trees, two thread configurations, three rounds - produced the same combined digest
  `7f9dfcfb4247be8341001c7317c3ca48a72517ffc75a5a5a25bee897379c2b2b` over every common-assessed
  attempt row, the complete common Ugi assessment, every route-assessed row and result, and every
  realism-assessed row and result. `tests/test_assessment_output_identity.py` re-derives each
  rewritten quantity a second, naive way - the pre-change breadth-first traversal, the per-row
  canonicalization, a second evaluation pass, a fresh parse for every fingerprint, and the
  classifier at a different thread count - and requires exact equality.
- Measured but not adopted: a process pool over the two cold reference builds does beat serial with
  bit-identical output, start-up included - 6,144 whole-lipid descriptor vectors 6.249 s serial
  versus 2.822 s on eight workers, and 112,386 identity product parses plus canonicalization 8.081 s
  serial versus 1.009 s on sixteen workers. It is deferred rather than rejected: the production
  assessor runs in the same process as GPU sampling, so any pool must be spawn-only and CUDA-safe,
  which is a separate qualification against a paid run.
- **Decision:** retain the bounded classifier thread pool and the five reuse changes. Do not set a
  process-wide thread limit. No policy field, denominator, threshold, tolerance or frozen selection
  changed, and this wall-time work is not a scientific result. Assessment remains sequential across
  ledgers; process-level parallelism across independent (program, seed) assessments is the largest
  remaining lever and requires its own authorization and qualification.

## 2026-08-28 - Attribute the production evaluation stage and remove its repeated chemistry

- The two preceding wall-time entries were measured on subsets. Nobody had profiled the stage the
  paid run actually executes, `model.shared-synthesis-program-production-evaluation-from-pins.v1`.
  Added `make evaluation-profile`, i.e. `PYTHONPATH=.:tools python3 -m forge_maintenance
  evaluation-profile`, which runs the real entry point against the frozen seed-0 program-role pins
  on CPU. It keeps the frozen five checkpoints, three programs, 32 flow steps, batch 128 and the
  `strict_valence_topology_argmax` decoder, reduces only the per-cell sample counts, and binds that
  reduction to the config's own `h100_preflight` execution scope so a reduced budget can never
  claim production status. It wraps the functions the evaluation already calls and reports
  exclusive wall time per phase; the wrappers are removed afterwards.
- Two structural findings correct a standing assumption. The frozen assessors associated with this
  arm - the common Ugi assessment, the whole-lipid realism assessor, its classifier two-sample test
  and the method-blind route evidence - are not in this stage. It emits the attempt ledgers and
  those assessors run as later stages, so the 2026-08-28 assessment work does not touch this one.
  And there is no repeated per-checkpoint work to hoist: the production cache, the factorized
  layout prior and both reference ledgers are each built exactly once, and unpacking plus rebuilding
  all five checkpoints costs 0.28 s. Per-run overhead outside the evaluation body is also not the
  problem: hashing the 112.6 MB checkpoint archive takes 0.047 s, the repository source fingerprint
  0.055 s and importing torch and RDKit 0.60 s.
- The cold attribution of a 3,072-attempt run was 625.0 s: the model forward 456.9 s (73.1%), exact
  reverse decomposition with forward replay 61.0 s (9.8%), the R-star updates 35.0 s (5.6%), the
  training reference ledgers 26.1 s (4.2%), the Ugi identity references 16.9 s (2.7%) and the
  layout prior 13.7 s (2.2%). Everything else together is under 1.5%, including the O(n^2) pairwise
  ECFP4 diversity. The forward's 73% is an artifact of profiling on CPU: it is the only phase the
  H100 accelerates, so what the paid run actually pays for is the 130.5 s device-independent
  remainder, which runs at exactly the speed measured here.
- Inside that remainder, exact decomposition is one program. Per valid row it cost 105.7 ms for
  `lx_2024_repeated_reductive_amination` against 7.55 ms for `bl_2023_repeated_aza_michael` and
  0.35 ms for `ugi_3cr_agile`, and it is extremely heavy tailed: over 200 LX rows the median was
  7.27 ms, the slowest single row 38.5 s, and the slowest ten rows 94.5% of the total. The reverse
  search rebuilds the same one-step assemblies 3.40x on average within a single decomposition -
  once to confirm a candidate disconnection, then again as the shared prefix of every forward
  replay - so the transform now takes a caller-owned reuse table, created per public call and
  discarded with it. The one-step assembly is a pure function of the accumulator SMILES, the repeat
  SMILES and the outcome bound given the frozen registry transform, and an unparseable input still
  raises on its first, uncached occurrence. Separately, `load_reaction_program_training_references`
  now memoizes canonicalization per distinct source string, because its 66,464 train-fold Ugi rows
  draw on 295 distinct precursor constitutions and its 60,000 multi-reaction rows on 219 terminal
  heads and 54 repeat components. Product SMILES are all distinct and remain a per-row parse.
- Measured by alternating the pristine and changed trees within each round and taking the minimum
  over three rounds, because the host carried unrelated load throughout. Exact decomposition of the
  same 3,072-row ledger fell from 33.63 s to 12.18 s (2.76x), LX alone from 30.95 s to 10.00 s
  (3.10x), and the training reference ledgers from 27.74 s to 13.78 s (2.01x). Projected onto the
  frozen 16,896-attempt budget, including a separate probe that measured the uncapped
  46,122-record component-disjoint reconstruction, this takes the stage's device-independent floor
  from about 368 s to about 205 s. The attributable record is
  `results/maintenance/production_evaluation_cpu_performance_v1/result.json` (SHA-256
  `68e1486f7aed31cff6abd25d5c69f81447bfa96f5eafccd63030ce513c753838`).
- Output identity is the acceptance criterion. All six alternating runs produced the combined
  digest `78bc51f82bac9c0690141e016f5ac2b021ba7d6039efac20f57fccd16e744351` over every reference
  set, every adjudicated row and every evaluated metric. Two complete end-to-end profiled runs, one
  per tree, emitted byte-identical artifacts: `samples.jsonl.gz`
  `c564d21ddf7bd7ed3b226e8b1e3396e57218e6268d473f4111bc92667feb1f40`, `molecule_report.html`
  `ac2440ee80e3389596878b8306f25007c92d616d268869df46301610fe12042f` and the common Ugi attempt
  ledger `ab418ebfd91cc86760467f5b29d80d2c59e5c305853c6ff14e867b93733cc650`, with an identical
  `result.json` apart from its own artifact receipts.
  `tests/test_evaluation_reuse_identity.py` re-derives each rewritten quantity a second, naive way -
  the pre-change one-step assembly with no reuse table, and a fresh canonicalization per occurrence -
  requires exact equality, requires the fixture to actually exercise the reuse, and requires invalid
  input to keep failing.
- Measured and rejected: the deferred process pool. Sampling and assessment are interleaved per
  cell inside the one process that holds the CUDA context, so a fork-based pool is unsafe; a
  spawn-only pool is not, because the workers re-enter the interpreter and import only RDKit and
  the assembly adapters, and it produced identical rows. On 2,246 valid rows it returned 12.06 s serial against 6.58 s on four workers (1.83x) and
  6.14 s on eight (1.96x). Doubling the workers buys almost nothing because the workload is bounded
  by one item, not by throughput: on the changed tree a single LX product still takes 13.70 s, which
  is 53.2% of its whole 600-row ledger, so no pool of any width can finish that ledger faster than
  13.70 s. The production container also reserves four CPUs. That is a small, concurrency-risky gain
  inside a paid CUDA process on top of a change that already removed three times more, so it is not
  adopted.
- **Decision:** retain the per-call forward-layer reuse table and the reference canonicalization
  memo. Do not add a process pool to the evaluation stage. No gate, denominator, threshold, policy
  field or frozen selection changed, and this wall-time work is not a scientific result. Ten times
  end-to-end is not reachable: after this change the stage's device-independent floor is about
  205 s at the frozen budget, of which about 41 s is one-time setup that must parse 126,464 distinct
  reference products and exhaust the fail-closed layout support, about 85 s is a reverse search
  whose cost sits in a handful of pathological products, and about 26 s is the uncapped
  component-disjoint reconstruction. Even a four-worker pool applied everywhere lands near 3x from
  the pre-change baseline, not 10x, and the single worst product bounds it from below.

## 2026-08-28 - Locate the remaining Transformer training cost in padding and PCGrad walk count

- Followed the 2026-08-27 instruction to attack the variable-shape collation/attention schedule or
  the PCGrad implementation without changing its mathematical partition. Deterministic eager true
  FP32, the 32 x 4 geometry, equal-family source balancing and full 194-heavy-atom support are
  unchanged; TF32, BF16, compilation and alternative batch geometry were not re-proposed.
- Added `experiments/phase1/multireaction/cpu_training_step_profile.py`, a reproducible CPU profile
  of the frozen optimizer step. It carries an inline pre-optimization implementation of every hot
  path it changes and runs the two arms back to back on the same micro-batch, because absolute
  seconds on a shared workstation drift by tens of percent within one run and are not comparable.
  The attributable result is
  `results/phase1/shared_synthesis_program_cpu_step_profile_v1/result.json` (SHA-256
  `4097339a4d14204754c1e0b59f826b261477493e899fe8d6fe1f4dd0412d8034`).
- Six execution-only changes landed: the graph-relation bias resolves each mask once and is
  materialized head-major; the absent-key fill is applied once per forward instead of once per
  attention layer; family selection, both chemistry cross entropies, the state-balanced semantic
  cross entropy and the repeat-consistency bond selection each resolve their mask once and gather by
  integer row; the fixed-state check compares in place; and the sequential PCGrad backend flattens
  each family straight into its row of the projection buffer. Losses and accumulated parameter
  gradients are **bit-identical** to the pre-optimization path, asserted by `torch.equal` in
  `tests/test_reaction_program_transformer.py`.
- Their measured effect is a 6.53x reduction in host-synchronizing dispatches per optimizer step,
  940 to 144, at an unchanged kernel-launching dispatch count of about 16,300. The paired CPU wall
  time improved by a median 1.043x, which is at the edge of what this machine can resolve. Host
  synchronization is a device-independent count, not a CPU timing artifact; its accelerator value is
  unmeasured here.
- The dominant remaining cost is padding. At the frozen packing the mean padded width is 115.3
  against a mean of 48.2 true heavy atoms, so 42.8% of node-linear work and 20.8% of attention work
  is on real atoms. The already-implemented padding-aware quantile-bin sampler at 32 bins raises
  those to 73.9% and 54.9% and measured 1.59x on the frozen 32 x 4 geometry. It is not enabled in
  the frozen mixed-arm runtime. It preserves each exact categorical source measure but changes which
  records share a micro-batch, so it is a packing decision, not a bit-identical execution change.
- Measured but not implemented: the three per-family PCGrad backward walks cost 2.851x one walk of
  their sum. The family losses depend on disjoint batch rows, so one walk already carries all three
  activation gradients; three walks buy only the per-family split of the parameter reduction. Taking
  that split from a single walk projects a 1.757x optimizer step. It requires a manual per-family
  parameter reduction for every parameterized operation, and the model forward would have to keep
  the batch axis attributable for the shared-index position embeddings, the token-type constants,
  the closure slots and the two count-logit parameters. It is not bit-exact, because the weight
  reduction would run over the family's rows rather than over all rows with the others at zero.
- Negative results: the batched-VJP PCGrad backend is 1.13x **slower** than the sequential backend
  on CPU, because `vmap` replaces one graph walk with a batched one at three times the work rather
  than exploiting row disjointness; it may still win where launch overhead dominates, which this
  machine cannot show. Making the attention query, key and value contiguous measured as noise and
  was reverted. The macOS CPU `bmm` batch loop and `bernoulli_` dropout-mask cost are host artifacts
  and were not optimized against.
- The four named suites behave as before: `tests/test_reaction_program_transformer.py` passes all 24
  tests including 7 new equivalence tests, and the model-related suite slice fails on exactly the
  same 21 absent historical artifacts with and without these changes.
- **Decision:** retain the six bit-identical execution changes. Do not change the frozen mixed-arm
  packing without an explicit decision: quantile-bin packing is the largest available lever and is
  measured, but it changes micro-batch composition and therefore where per-family PCGrad projection
  is computed, which is the same objection that blocked 64 x 2. Treat the single-walk per-family
  parameter reduction as the next optimization target, gated on its own exact-gradient test and a
  fresh exact-H100 equivalence run. This CPU profile establishes execution equivalence, dispatch and
  padding accounting, and a local speedup only; it is not evidence of H100 throughput or of model
  quality, and it authorizes no paid run.
