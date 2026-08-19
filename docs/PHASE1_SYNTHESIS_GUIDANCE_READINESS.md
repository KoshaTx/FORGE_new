# Phase 1 synthesis-guidance implementation readiness

## Status and scope boundary

This document began as an implementation-readiness map. On 2026-08-01, after
the product-plus-L1 generator and post-selection provenance audits were frozen,
the user explicitly authorized the synthesis-routing and synthesis-guidance
milestone. Implementation must follow the fail-closed order below. Production
tilting remains gated on zero-guidance equivalence, independent L2 planner
qualification, adequate exact support and a nontrivial generated-candidate
L1/L2/L3 dossier set.
Biological guidance and prospective candidate locking remain unauthorized.

The frozen product-model integration target is the selected
`full_morphology_program_conditioning:step_1000` checkpoint
(`90f5f0bd3e41e2884f6588d9875db0b7bf34e5c5ea7fdc1f9d8565fba8c0c532`). The selection result is
`results/phase1/ugi_architecture_checkpoint_selection_v3.json`. The matched closure checkpoint is
`results/phase1/ugi_closure_expanded_full/checkpoint_best.pt`
(`a97507ac6a9eeba41d0cc351666cffdd21069d13bc78c0db671ab3a30c710b5d`). Model selection evaluated
product and L1 behavior only; it did not evaluate L2 routes or synthesis guidance.

For the selected architecture, node counts, junction budgets, cycle ranks, and attachment counts are
supplied by the frozen morphology-program draw. A future guidance experiment should therefore keep
those fields fixed and matched across arms. Guidance may redistribute probability over offspring
arrangements, atom and bond states, decorations, and particle ancestry within the supplied programs.

Biological guidance remains off.

## Reusable primitives already present

The repository now contains conservative evidence, recursive assessment,
cache components and a nonprobabilistic synthesis-value layer, but no qualified
general L2 route proposer or guidance implementation:

- `forge.product.ugi_generated_components.generated_ugi_component_smiles` recovers the three exact
  precursor graphs from generated Ugi product semantics.
- `forge.product.ugi_held_component_gate._forward_reconstructs_product` verifies the exact L1 Ugi
  round trip. This private helper should be promoted to an appropriately named public module before
  reuse rather than imported as a private API.
- `forge.route.ugi3_complete_computational_dossiers.classify_component_program` composes exact-source
  L2 step verification with L3 leaves and fails closed on missing, duplicated, ambiguous, mismatched,
  or unverified steps.
- `forge.route.ugi3_production_registry_route_readiness.classify_component_evidence` assigns the
  strongest explicit evidence tier without promoting structural similarity, handle qualification, or
  provenance into route evidence.
- `forge.route.qualified_forward.load_qualified_forward_reaction` and `unique_forward_products`
  provide deterministic, qualified forward-verification primitives.
- `forge.route.planner` and `forge.route.planner_cache` provide the typed,
  fail-closed recursive assessment contract, budget ledger and
  content-addressed cache. These modules compose evidence supplied by a source;
  they do not infer new chemistry.
- `forge.route.ugi3_hybrid_search` qualifies a bounded 424-component hybrid
  evidence source. `forge.route.ugi3_targeted_exact_overlay` and
  `forge.route.ugi3_targeted_role_gap_overlay` add only authenticated exact
  identities and delegate every other target unchanged. The final
  `forge.route.ugi3_high_leverage_head_terminals` overlay admits two additional
  exact current-procurement terminals without changing any other identity.
- `forge.route.ugi3_exact_c18_route` qualifies one exact, contiguous three-step
  route to octadec-17-ynal, retains step-level conditions provenance and closes
  its exact stearolic-acid terminal independently. Its transforms are
  exact-pair-only and cannot be used as family templates.
- `forge.route.ugi3_exact_c16_route` qualifies an independent exact four-step
  route to hexadec-15-ynal, closes two exact structural terminal leaves and
  preserves a carbon-inconsistent alternative patent step as rejected evidence.
  It likewise admits no family or homologue scope.
- `forge.value.synthesis` implements typed component and product values,
  explicit unknown burden states, role-preserving aggregation and a Pareto
  comparator. It deliberately exposes neither a scalar value nor a synthesis
  success probability.
- `forge.value.ugi3_synthesis_value_audit` composes the frozen evidence layers
  over the selected 1,007-product generator sample. Generated components absent
  from the registry remain `missing_knowledge`; they are never relabeled as
  chemically incompatible.
- `forge.value.ugi3_fresh_pool_route_coverage_v4` applies the qualified C18
  assessment to one role-qualified identity in the 3,975-product development
  pool and recomputes every product value without assigning scalar scores.
- `forge.value.ugi3_fresh_pool_route_coverage_v5` applies the qualified C16
  assessment to one additional role-qualified identity. It changes exactly one
  component value across 99 products and raises complete development products
  from 185 to 203 without family or homologue promotion.
- `forge.route.ugi3_route_registry_pair_contract` and
  `forge.route.ugi3_route_registry_pair_builder` freeze the prereveal paired
  comparison and materialize it only after explicit final hashes are supplied.
  The checked-in builder remains unbound and cannot reveal the molecular
  holdout.
- `forge.route.ugi3_virtual_programs._aldehyde_program` and `_isocyanide_program` demonstrate
  deterministic role-specific family projections. They are private audit helpers, and their outputs
  remain family projections unless exact evidence and qualified forward scope support a stronger tier.
- `forge.product.ugi_joint_sparse_flow.sample_ugi_joint_sparse_terminals` implements the current
  discrete-flow trajectory through `_rstar_step`.
- `forge.product.ugi_joint_end_to_end_sampling.sample_ugi_joint_end_to_end` composes joint-flow output,
  closure placement, valence-constrained decoding, molecule sanitization, and precursor recovery.
- `forge.product.ugi_joint_sparse_sampling` exposes cloneable initialization,
  exact-step advance and constrained terminal finalization. The compatibility
  wrapper is bitwise equivalent to the frozen monolithic schedule on the
  selected checkpoint across three batch partitions.
- `forge.product.ugi_joint_end_to_end_sampling.complete_ugi_joint_terminals`
  exposes closure placement, valence-constrained chemistry, sanitization,
  component recovery and optional L1 verification behind an explicit closure
  RNG state.
- `forge.product.ugi_synthesis_guidance` implements controller mechanics only:
  stable Feynman--Kac ancestry probabilities, keyed random substreams, an
  exact zero-guidance identity bypass and a fixed-budget terminal-rollout
  executor that rejects partial, invalid and nonexact-L1 states before value
  evaluation. It has been tested only with fake values and does not define a
  synthesis objective.
- `forge.product.ugi_joint_sparse_sampling` now also exposes particle extraction
  with keyed rollout RNG state and within-program ancestry application.
  Ancestry that would cross frozen morphology-program groups fails closed.

The current route artifacts are evidence-composition and bounded-search audits,
not a general route proposer. In particular, an empty route set means that a
route was not curated or found under the declared scope; it does not establish
that the component is unsynthesizable.

## Minimal module boundaries

These boundaries define the smallest coherent authorized implementation.

| Proposed module | Contract |
|---|---|
| `forge.route.planner` | **Implemented.** Typed recursive assessment returning a complete assessment and bounded-search trace for one role-qualified component. Exact-evidence lookup remains separate from any future learned or general route proposer. |
| `forge.route.planner_cache` | **Implemented and qualified diagnostically.** Content-addressed storage for route trees, evidence, verifier records, missing-knowledge states, budget exhaustion, and failures. |
| `forge.value.synthesis` | **Implemented as a structured, nonprobabilistic layer.** Typed component-to-product aggregation, explicit L1 state, unknown burden states and a Pareto-safe comparator are tested. No scalar policy is authorized. |
| `forge.value.distill` | Optional noisy-state distillation from a frozen planner cache with component-family-disjoint splits, field-level auxiliary targets, uncertainty, and abstention. |
| `forge.product.ugi_synthesis_guidance` | **Diagnostic controller and terminal-rollout budget mechanics implemented; chemistry coupling remains blocked.** Analytic fake-value probabilities, keyed substreams, exact zero-guidance bypass, terminal-lock admission and explicit budget records are tested. Real generator-completion rollouts, route-cache traces and matched-arm orchestration remain to be implemented. |
| `scripts/phase1_run_matched_synthesis_guidance.py` | A future single, hash-pinned entry point for zero-guidance, planner-guided, distilled-guided, and matched post-hoc arms. |

The current sampler has been refactored without behavioral change into
restartable operations equivalent to:

1. `initialize_ugi_joint_sparse_state`;
2. `advance_ugi_joint_sparse_state`;
3. `finalize_ugi_joint_sparse_terminal`; and
4. `complete_ugi_joint_terminal`, including closure placement, valence-constrained chemistry,
   sanitization, precursor recovery, and exact L1 verification.

The existing `sample_ugi_joint_sparse_terminals` and
`sample_ugi_joint_end_to_end` entry points remain compatibility wrappers. The
frozen-checkpoint audit at
`results/phase1/ugi_restartable_sampler_equivalence_v1/result.json` is bitwise
equal to the monolithic reference for 12 fixed programs, eight flow steps and
batch sizes 1, 4 and 12. This qualifies the refactor only; it does not qualify a
synthesis signal.

## Pre-prospective `V_syn` contract

Before prospective positive and negative outcomes exist, `V_syn` is an
**evidence-weighted route-completion value**, not a calibrated probability of synthesis success. Raw
route-model likelihood is prohibited.

Each route assessment must preserve at least:

- exact L1 forward-reconstruction state;
- route closure state;
- forward-consistency state;
- weakest evidence tier;
- exact-source versus family-projected status;
- route depth;
- protection and deprotection burden, including an explicit unknown state;
- purification burden, including an explicit unknown state;
- L3 state: current closed, unavailable, expired, or unassessed;
- closed, unresolved, and unavailable leaves;
- missing-knowledge class;
- substrate-scope and bounded-search uncertainty; and
- planner budget-exhaustion and execution status.

The product record must retain the three role-specific component assessments and an L1/L2/L3 value
decomposition. Distiller epistemic uncertainty must remain separate from planner/search uncertainty.

The implemented value record intentionally contains null `scalar_value` and
`success_probability` fields. If a future sampler requires a scalar, it must be
derived by a hash-frozen monotonic policy over the structured record. The
policy must guarantee that, with all other fields fixed:

- closing a leaf cannot reduce value;
- exact evidence cannot rank below family evidence;
- unique forward verification cannot reduce value;
- added route burden, uncertainty, unavailable leaves, or missing knowledge cannot improve value;
- a family projection cannot become exact-source closure; and
- budget exhaustion cannot be relabeled as substantive chemical rejection.

The policy must resolve cross-axis trade-offs before guided outputs are inspected. No field or output
may be named `success_probability`, `p_success`, or otherwise presented as the probability that an
experimental synthesis succeeds.

## Planner cache and distillation

A route-cache key should include:

- role plus canonical constitutional component SMILES;
- the product/component tuple when product context affects assessment;
- planner model, checkpoint, search-policy, and value-policy hashes;
- maximum depth, expansions, products, verifier calls, and time budget;
- L1 reaction, upstream reaction-registry, and variant hashes;
- L3 snapshot hash, region, access time, and expiry;
- RDKit and relevant software versions; and
- stereochemistry and identity policies.

Cache exact closure, family projection, missing knowledge, substantive rejection, budget exhaustion,
invalid input, and execution failure as distinct states. An expired L3 snapshot must invalidate
current closure rather than silently reuse an earlier positive result. Report logical planner calls,
physical cache misses, cache hits, verifier calls, and elapsed resources separately.

If authorized later, distillation should use complete terminal molecules assessed under one frozen
planner/cache version. Training examples should include noisy joint sparse states at declared flow
times. The model should predict the structured value fields as auxiliary targets in addition to any
frozen scalar and should abstain outside its measured domain. Component-family-disjoint splits and a
candidate-selection leakage audit are required. Distillation does not upgrade the evidence tier of
its labels.

## Intermediate coupling and matched post-hoc design

Exact closure is not defined for an invalid noisy partial graph. The minimal safe coupling is
therefore planner-rollout Feynman–Kac/SMC rather than attaching exact-closure labels directly to
partial states.

At frozen intermediate checkpoints, complete deterministic-budget rollouts from each particle,
assess the completed molecules, and resample ancestors with an annealed potential such as

```text
P(A_k = i) = softmax_i(
    log w_(i,k-1) + beta_k * Vhat_(i,k) - beta_(k-1) * Vhat_(i,k-1)
)
```

After resampling, continue the unchanged `_rstar_step` transitions. Thus synthesis guidance changes
the probability mass assigned to intermediate molecular states before candidate lock. At zero
guidance strength, the sampler must reproduce the selected baseline exactly. A later distilled arm
may replace terminal planner rollouts with an abstention-aware `V_psi(state, t)` at the same SMC
checkpoints; direct per-channel logit modification is not required for the first safe implementation.

The matched post-hoc arm must use one frozen budget ledger containing:

- identical morphology-program indices and generator/closure checkpoints;
- identical sampling steps, particle count, branching schedule, and terminal-completion budget;
- identical logical planner and verifier budgets;
- matched wall-clock/GPU caps and reported realized resources; and
- identical final candidate count.

The guided arm may consume route values at intermediate checkpoints and resample. The post-hoc arm
must execute the same productive generation and completion budget without route feedback, seal each
terminal candidate and generation trace, then assess and filter routes. Planner evaluation before the
post-hoc terminal lock is a contract violation.

Use deterministic keyed random-number substreams based on arm, program index, particle, checkpoint,
and rollout rather than relying on one sequential stream after the arms diverge. Give each arm an
identically seeded or cloned cache state; do not let the second arm inherit warm-cache advantages from
the first. If an arm exhausts a shared budget, apply the preregistered censoring/stopping rule to both
arms rather than adding dummy calls or silently granting more compute.

Guidance strength must be selected under frozen floors for validity, exact L1 consistency, uniqueness,
broad-distribution coverage, and per-role component novelty/effective diversity. Report route closure
separately for familiar catalog components, transferred known components, and genuinely generated
components so a collapse onto the small known route-complete set remains visible.

## Current frozen value census

The composed audit at
`results/phase1/ugi3_synthesis_value_audit_v3/result.json` freezes the present
boundary:

- 50 of 424 admitted registry components are exact route-complete;
- 273 registry components remain missing knowledge and 101 remain outside
  declared support;
- the selected generator sample contains 610 unique generated components, of
  which 26 are route-complete, 560 are missing knowledge and 24 are outside
  support;
- 519 of the 610 generated identities are absent from the admitted registry and
  therefore remain explicit missing route knowledge;
- 45 of 1,007 generated products have exact L1 and all three component branches
  closed; 962 remain noncomplete;
- protection and purification burden are unknown for all 610 generated
  component identities; and
- zero component or product records contain a scalar value or synthesis-success
  probability.

This census is diagnostic. The 45 closed dossiers demonstrate end-to-end value
composition, but their sparse and identity-concentrated support cannot yet
justify production synthesis pressure.

## Actual precursor-leaf closure beneath the 1.49-million-product stress test

The optimistic AGILE-template stress test assigns a mechanical component
program to every expanded product, but it deliberately does not assume that
the projected terminal leaves are available. The follow-up audit at
`results/phase1/ugi3_precursor_leaf_closure_audit_v1/result.json` resolves that
question using exact constitutional identity, unexpired US item-level supplier
evidence and separately adjudicated historical source use.

- 31 of 264 admitted amine heads have current item-level terminal evidence.
- 25 of 69 aldehyde-program leaves and 10 of 53 isocyanide-program leaves have
  current item-level terminal evidence: 35 of 122 role-summed tail leaves.
- Exact historical source use is recorded for 23 aldehyde-program leaves and
  seven isocyanide-program leaves. Of the 122 leaves, 23 are both historically
  used and currently evidenced, 12 are current-only, seven are historical-only
  and 80 have neither form of evidence in the frozen corpus.
- 63 of 107 aldehyde programs and 10 of 53 isocyanide programs have every
  projected leaf currently closed.
- 165,600 of 1,486,415 products (11.1409%) have both tail programs closed at the
  current starting-material identity level. Only 18,810 (1.2655%) also have a
  currently closed amine head.
- The independently frozen exact-route census remains narrower: 1,830 products
  (0.1231%) have all three component branches supported by strict exact route
  dossiers.

Historical use, procurement and exact L2 execution are orthogonal evidence
axes. For example, visual review of the Miao 2019 supplement confirms execution
with ethyl isocyanoacetate, but does not report the ethyl-glycinate leaf produced
by mechanical inversion of the generic isocyanide program. The projected leaf
therefore remains unclosed. Likewise, historical use of an alcohol in another
lipid platform does not establish current inventory or exact oxidation scope.

The operational consequence is that synthesis guidance remains nonvacuous.
Template attachment alone cannot discriminate the expanded universe, whereas
current leaves, exact-substrate routes, route burden and uncertainty can. The
matched post-hoc baseline must receive the same terminal evidence and planner
budget so the comparison isolates when synthesis information enters sampling.

## Candidate-driven exact-route saturation update

The nonselecting 3,975-product fresh pool now provides the development set for
targeted route-registry expansion. It is separate from the previously selected
1,007-product diagnostic sample and from the independently sealed saturation
holdout.

- Fresh-pool coverage v3 contains 172 products for which exact L1 and all three
  component branches are complete. It retains 849 one-gap products and never
  imputes a scalar synthesis value or experimental success probability.
- Gap triage found that the unresolved frontier is dominated by exact identities
  inside already represented upstream program classes. No new reaction family
  meets the frozen admission threshold of at least three unique development
  components and ten one-gap product occurrences. Candidate-driven mining
  therefore prioritizes exact-substrate evidence rather than indiscriminately
  expanding the family registry.
- One exact three-step route to catalog-absent octadec-17-ynal is now qualified
  through current stearolic-acid procurement. The exact-identity-only v4 refresh
  changes one component value across 107 occurrences and increases complete
  development products from 172 to 185; it promotes no neighboring homologue or
  reaction family.
- A second exact route to hexadec-15-ynal is now qualified through a
  carbon-balanced four-step chain and two current exact terminal materials. The
  internally inconsistent WO 2024/073486 A2 step is retained as a rejected
  `source_conflict` and is not used. The isolated v5 coverage refresh reproduces
  its frozen delta exactly: 99 products change, 18 become complete and the final
  branch-count bins are 1,211/1,664/897/203 for zero/one/two/three complete
  branches.
- The 4,096-program independent holdout has now been executed exactly once by
  the blinded, hash-bound runner. Of 3,960 exact-L1-eligible products, the
  immutable R0/without-C18 registry completes 201 and the R1/with-C18 registry
  completes 222. Thus, one exact C18 component route changes 21 complete-product
  outcomes (0.5303%; exact 95% interval 0.3286--0.8095%) while all other
  evidence, including the qualified C16 route, remains identical.
- The C18 component occurs 139 times but flips only 21 products because every
  other precursor branch must also close. This establishes that upstream route
  evidence is product-level consequential without implying universal route
  saturation or experimental synthesis success.
- Final v5 triage identifies 209 one-gap components already expressible by the
  three existing upstream program families, affecting 515 development-product
  occurrences. For 177 occurrences every projected terminal leaf is currently
  evidenced; the remaining gap is bounded family applicability or exact
  substrate confirmation, not invention of a new reaction family.
- Family evidence will therefore enter production development as an explicit
  lower-confidence tier, separated from exact source execution. A reusable
  family may support proposals and graded guidance only within its declared
  applicability domain; every locked candidate still requires an exact
  candidate-specific route dossier and refreshed terminal status.

This workflow measures when additional route knowledge stops changing coverage
of the model's biologically relevant candidate distribution. It does not claim
that the registry is a complete chemistry universe or that an unresolved route
is chemically impossible.

## L3 decision-horizon gate as of 2026-08-03

All procurement records admitted into the current 45-component exact registry
are valid at the present audit time, but their validity windows are not uniform.
The 203 route-complete development products have the following mutually
exclusive earliest-expiry census:

- five products expire at 2026-08-09T05:16:00Z through the `CCCN` head overlay;
- three expire at 2026-08-09T05:51:40Z through the `NC1CCCCC1` head overlay;
- 160 expire at 2026-08-28T18:23:53Z through the July AGILE-head snapshot; and
- 35 expire at 2026-08-28T20:49:00Z through the July virtual-terminal snapshot.

Every currently route-complete product depends on an isocyanide precursor from
the July virtual-terminal snapshot, so no current product-level route-complete
claim has a decision horizon beyond 2026-08-28T20:49:00Z. A real guidance run
must therefore freeze both an `as_of_utc` and the operational decision horizon.
If any required leaf expires before that horizon, the affected component and
product fail closed unless a new immutable exact-identity procurement overlay
is independently refreshed. Historical snapshots must never be mutated.

Candidate lock requires a new item-level availability check for every terminal
leaf used by every selected route even when the development snapshot has not
yet expired. Expired or missing evidence remains an L3 knowledge state; it must
not be relabeled as chemical incompatibility or outside support.

## Exact blockers as of 2026-08-02

1. The typed recursive assessor, bounded hybrid evidence source, planner cache,
   structured synthesis-value implementation and diagnostic SMC controller
   exist. No qualified general L2 route proposer, scalar value policy,
   distiller or production synthesis-guidance value exists.
2. The upstream qualified-reaction registry is explicitly exact-source-only and forbids general
   enumeration and substrate-scope extrapolation.
3. Current exact registry support remains narrow: 50 of 424 components are
   complete after all targeted exact overlays. Under the typed bounded source,
   273 remain missing knowledge and 101 remain outside declared support.
   Family, analogue, motif, provenance and handle-only evidence remain
   nonclosing.
4. Thirty-one of 264 amine heads have current exact terminal identities after
   all time-pinned procurement overlays are composed; no general upstream
   head-route model exists. Supplier availability must be refreshed before
   candidate lock.
5. The original frozen L3 snapshot contains 36 records, closes 35, and expires
   after 30 days. Targeted supplier overlays are separately time-pinned, with
   the newest head record expiring on 2026-08-09. There is no general availability
   service or refresh workflow.
6. Source routes contain purification prose, but there is no normalized protection/purification
   burden schema or frozen burden policy.
7. There is no calibrated substrate-scope or search-uncertainty estimator.
8. The evidence inventory contains no reported negative synthesis outcomes, so a calibrated
   experimental success probability cannot be trained or claimed.
9. The sampler and terminal completion path are restartable and pass bitwise
   zero-guidance equivalence. The real 12-program production rehearsal now also
   passes terminal lock, exact L1 reverification, isolated immutable-cache
   accounting and matched-arm compute enforcement without censoring. Exact
   route assessment remains terminal-only and is deliberately undefined on a
   noisy categorical state. The rehearsal observed one route-complete and 11
   route-incomplete terminals; no scalar or success probability was defined.
10. The historical product/L1 configuration deliberately disables synthesis
    guidance and remains immutable. The newly authorized milestone requires a
    separate frozen configuration and cannot rewrite the historical contract.
11. Exact evidence currently completes 45 of 1,007 corrected generated-product
    dossiers and only 26 of 610 unique generated components. This is a useful
    diagnostic closure set, but it is concentrated on named exact identities
    and does not justify a smooth or calibrated synthesis value over novel
    generated chemistry.
12. **Completed:** the immutable 1,919-record R0/R1 pair and dedicated one-shot
    runner executed the independent holdout once under pinned model inputs, no
    rendering, no overwrite or retry, immediate output sealing and
    aggregate-only reporting. The completed-output validator independently
    reproduced the public aggregate. The result establishes marginal registry
    sensitivity, not a synthesis-success probability or guidance policy.

An evidence-only lookup adapter can exercise future cache and SMC plumbing, but
it remains a diagnostic smoke test. Using it as the production guidance signal
would strongly favor the 50 known route-complete components and could
manufacture apparent route closure through component collapse.

## Authorized build order

1. Retain the implemented structured synthesis-value records and Pareto
   comparator as the diagnostic contract. Do not derive a scalar policy yet.
2. **Completed:** freeze a zero-guidance refactor policy containing the selected
   checkpoint, fixed program draw, seed, batch partitions and equivalence
   criteria.
3. **Completed:** refactor the selected sampler and terminal completion path,
   and pass bitwise zero-guidance equivalence tests.
4. **Completed:** analytic SMC probabilities, keyed substreams, the exact
   zero-guidance bypass, fixed-budget terminal admission, shared cross-arm
   censoring and post-hoc preassessment locking pass deterministic tests. The
   authenticated production rehearsal executed 12 real terminal completions per
   arm with typed terminal-to-assessor integration, immutable isolated planner
   caches, exact matched compute and complete support-audit retention.
5. Continue targeted exact-support expansion while deciding whether the first
   guidance experiment is an
   exact-evidence diagnostic or requires a separately qualified broader route
   proposer. Do not promote family templates beyond their admitted scope.
6. If authorized and supported by sufficient labels, populate and freeze a
   planner-label cache, then train and evaluate the optional distilled value
   model with component-family-disjoint splits and abstention.
7. Before real guidance, freeze a new experiment policy containing the selected
   checkpoints, any scalar `V_syn` ordering, route/verification budgets, L3
   snapshot policy, guidance checkpoints and strengths, diversity floors,
   matched-arm budgets and failure handling.
8. Run the preregistered guidance-strength sweep and matched post-hoc
   comparison. Stop if any frozen validity, coverage, diversity, novelty,
   provenance or budget gate fails.

## Required tests before any guided experiment

- Zero-guidance output is bitwise identical to the selected sampler for fixed seeds.
- Full-morphology program fields remain identical across arms.
- A synthetic two-particle test produces the expected SMC ancestry probabilities.
- Partial states cannot claim exact route closure.
- Exact L1, component recovery, and qualified forward verification remain invariant.
- Family projections, handle qualification, motif similarity, and provenance cannot promote exact
  evidence.
- Missing, ambiguous, duplicated, mismatched, or unverified route steps fail closed.
- Unavailable or expired L3 evidence invalidates current closure.
- Empty search results, budget exhaustion, substantive rejection, invalid input, and execution errors
  remain distinct.
- Cache keys invalidate on changes to planner, policy, registry, verifier, software, identity policy,
  or L3 snapshot.
- Cache artifacts are deterministic and record complete provenance.
- The `V_syn` comparator passes exhaustive or property-based monotonicity checks for every field.
- Distillation splits are component-family disjoint, retain field-level labels, and test abstention.
- Guided and post-hoc arms consume the frozen productive-call, planner, verifier, and final-candidate
  budgets.
- The post-hoc planner cannot be called before terminal candidate lock.
- Keyed random-number substreams are deterministic and independent of arm-specific call order.
- Guidance-strength selection enforces validity, broad-coverage, uniqueness, and per-role component
  diversity/novelty floors.
- Output artifacts report L1/L2/L3 value decomposition and familiar/transferred/generated strata.
- Biological guidance remains explicitly disabled.
