# Results to populate in the 22-family manuscript

All new result cells are **pending**, not zero and not a performance target. This scaffold does not launch experiments or open holdouts. The final model, inference policy, request distribution, splits and budgets must be frozen before final evaluation. Existing TRAIN-derived development results and historical three-family results are separate evidence populations.

The proposed final reporting unit is an independently trained model seed. Target at least the original study's three independent seeds; fix actual seed IDs, attempt counts and pairing before final evaluation. Report seed values, mean ± sample SD, paired contrasts and ranges. Request/batch-level intervals are supplemental and must respect dependence from population selection. Do not infer training uncertainty from repeated sampling of one checkpoint.

| ID | Evidence block | Required measurements | Denominator / unit | Priority |
|---|---|---|---|---|
| R01 | Cohort and source support | Admitted records, unique products, programs, studies and role components; exclusions, pending families, overlap, weights, size and closure support | family/program/source/split; unique constitutions separate from rows | Required |
| R02 | Training and reproducibility | Training seeds, presentations and updates per family, optimizer, LR, clipping, batch, precision, hardware, runtime, memory, numerical/restart equivalence | each training run and checkpoint; learning curves by family and corruption time | Required |
| R03 | Exact assembly across all families | Valid and connected yields, raw/checked exact L1, distinct exact yield, decomposition coverage, candidate precision, accepted replay invariant, ambiguity and failures | all requests, all22 families, each program and independent training seed | Required |
| R04 | Conditioning and common benchmarks | Conditioned, independently trained shared null and cyclic controls; three-family bridge and qualified native comparator set | matched data/exposure/requests/completion/verification; paired seed differences | Required |
| R05 | Decoder and chemistry repairs | Raw, constrained, completed, head, joint atom/bond motifs, final ring, full pipeline and selection ablations; changes and negative results | paired identical checkpoint/logits/requests; equal-cost and larger-pool contrasts separate | Required |
| R06 | Independent structural assessment | Representation and chemical alerts, context flags, head retention, ring-condition agreement, carbon-domain organization, blind review and source-control disagreement | all requests and applicable/assessed/pass/fail/abstain per check, family and source control | Required |
| R07 | Distributional lipid realism | Descriptor distributions/tails and normalized Wasserstein; fingerprint and descriptor precision/coverage; grouped C2ST AUC | family-matched and broader observed reference; source-held-out groups and valid/unique denominators | Required |
| R08 | Diversity, novelty and catalogue reach | Product/component unique counts, Shannon and inverse-Simpson effective counts, ECFP4 diversity, role/program novelty, distinct component-novel yield, catalogue reachability | all attempts and exact/unambiguous subsets; global cross-family deduplication | Required |
| R09 | Generalization and strata | Designated held-component yield, identity coverage, source/component split performance, size/closure/repeat/depth strata | independent trained seeds; group-level uncertainty; unseen-family stress secondary | Required |
| R10 | L2/L3 and complete dossiers | L2 evidence-qualified branches/steps, L3 identity/availability leaves, complete dossier yield, unresolved/conflict/censored states and E0-E3 tiers | all requests; products, distinct family-role components and leaves separately | Required |
| R11 | Architecture and learning mechanisms | Joint versus factorized matched/generous capacity, conditioning placement, role/core/repeat supervision, adapters and gradient handling | independently retrained matched-exposure controls; per-family losses/yields/cost | Claim-dependent |
| R12 | Training adequacy and efficiency | Matched-exposure batch/update experiment, separately exposure scaling; checkpoint quality, throughput, latency, GPU-hours and calls per success | same admitted corpus/request population; hardware/precision fixed or explicit | Required |
| R13 | Representative structures and failures | All-family seeded product/precursor atlas, head/tail origins, topology, novelty, L1/L2/L3 annotations, before/after repairs and unresolved failures | all22 represented; seeded random rows plus separately identified failure cases | Required |
| R14 | Theory and implementation correspondence | Bounded support, actual loss versus reference loss, masks/readout/repair/selection semantics and sampling dependence | final released implementation and explicit theorem assumptions | Required |
| R15 | Guidance if retained | Unguided/post-hoc/in-trajectory matched contrast, zero-guidance identity, eligibility and abstention; synthesis versus permitted potency pilot separate | only after applicable route/policy/authorization gates; no silent biological extension | Conditional, not launch authorization |

## Required population and claim boundaries

- Preserve every request, including invalidity, no decomposition, source-domain rejection, ambiguity, abstention, timeout and exceptions. Report denominators and exclusions explicitly.
- Show all 22 families, program variants, equal-family macro averages, pooled counts and the worst family. Also deduplicate products globally across families.
- Distinguish decomposition search coverage, candidate-trace replay precision, exact product yield and the accepted-trace replay invariant. The invariant cannot stand in for candidate precision.
- Keep independent chemical alerts, program-condition agreement, TRAIN feature support, distributional realism, diversity/novelty and route completeness separate. No uncalibrated universal quality score or pKa/delivery claim.
- Report per-check applicable, assessed, pass, fail and abstain populations. Head-survival qualification is currently scoped; extending a predicate to other families requires source controls before candidate scoring.
- Ring interpretation must distinguish graph cycle rank, connected ring systems, SSSR and serialization-dependent fundamental cycles. A changed or unknown basis is not silently a pass.
- Preserve product and component distinct identities, Shannon/inverse-Simpson effective counts, novel observations and novel distinct identities. Recompute absolute floors on the actual final population.
- TRAIN motif priors cannot independently certify realism; include source-held-out/method-blind controls and record alert rates on positive controls.
- Label unmatched native comparator families unsupported, not zero. A cyclic reaction label that retains correct roles/core is a limited label-control.
- Documentary L2/L3 closure is required for complete dossier claims. Static catalogue membership, motif similarity, supplier access failure and exact L1 alone do not establish closure or synthesis success.
- Guidance remains conditional on existing authorization and readiness gates. No wet-lab experiment or biological optimization is added as a paper-readiness task.

## Filling a placeholder

For each ID, attach a versioned per-attempt/per-seed result artifact with source/config/checkpoint/policy/split hashes, a runnable analysis command and passed verification receipt. Record computed, reported, inferred and proposed claims separately. Replace a cell only from admitted results; retain negative and null outcomes. `RESULTS_LEDGER.json` is the machine-readable map. The included original tables remain historical until a matched bridge is completed.

## Explicit quality and realism tables

- Main structural-quality table: validity, connectedness, exact-plus-qualified-design yield, chemical alerts, context flags, head retention, ring allocation/size, atom/ring reference support and exact-plus-observed yield.
- Main realism table: fingerprint precision/coverage/recall, distinct supported yield, nearest similarity, internal diversity, per-descriptor distance, and pending independent descriptor/classifier studies.
- Appendix: separate head/ring/support outcomes for all 22 families, exact metric definitions, and all 24 implemented descriptors.
- `QUALITY_METRICS.json` pins definitions to current code. Final cells remain missing; TRAIN diagnostics cannot populate independent final results.
