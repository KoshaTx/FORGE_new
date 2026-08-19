# Phase 1 dynamic frozen-prior terminal census

## Purpose

This census tests the frozen Ugi generator before choosing an applicability
controller. It asks whether terminal structural support is associated with the
sampled morphology program or with a saved partial graph state. It does not
score potency, routes or synthesis and cannot select prospective candidates.

## Frozen design

- Select 1,024 of 4,096 frozen morphology-program rows by a deterministic
  content-hash priority, without inspecting terminal outcomes.
- Draw two independent generator states for each selected row.
- Save each state after transition steps 2, 4 and 6.
- Draw four independent native continuations from every saved state.
- Preserve every terminal attempt, including invalid chemistry.

This gives 6,144 saved partial states and 24,576 terminal attempts. The
serialization records the full morphology program, generator state and
continuation identity, but the controller analysis may not use identifiers,
seeds or terminal information as input features.

## Validated result

The census completed all 32 restartable shards and an independent read-only
validator reconstructed the full coordinate lattice, deterministic seeds,
packed-state hashes, omitted deterministic tensors, ordered terminal ledger and
aggregate counts.

| Quantity | Count |
|---|---:|
| Terminal attempts | 24,576 |
| Saved partial states | 6,144 |
| Valid, exact-L1 products | 23,795 |
| Invalid products | 781 |
| Unique valid product SMILES | 16,395 |

The exact-L1 rate is 96.82%. It is stable across checkpoint steps: 96.81% at
step 2, 96.78% at step 4 and 96.88% at step 6. The 781 failures comprise 532
terminal-support failures and 249 molecule-sanitization failures.

Applying the previously frozen, target-free multiview radius as a read-only
terminal label gives 934 structurally supported outcomes (`R <= 1`), including
839 novel products and 45 products with all three precursor roles exact-new.
Exact component novelty is therefore not equivalent to chemical extrapolation.
This radius is a structural support coordinate, not a calibrated probability of
biological success.

## Analysis contract

The following analysis is additive and program-disjoint:

- `M0` uses only the 12-value morphology program and aggregates 24 outcomes per
  sampled program row.
- `M1` uses only categorical channels saved at one partial checkpoint and
  aggregates its four continuations as one binomial outcome.
- All cross-validation and bootstrap partitions group on `program_sha256`.
  The 1,024 selected rows contain 941 unique morphology hashes.
- Terminal molecules, terminal support, potency, routes, synthesis, seeds,
  hashes and row identifiers are forbidden as input features.

The simplest controller that achieves preregistered, program-disjoint gains in
supported-diverse yield per unit compute will be retained. Morphology proposal,
delayed rollout guidance and plain terminal screening remain competing options;
none is authorized by the census alone.

## Program-disjoint controller analysis

The morphology-only model shows strong out-of-group ranking signal. Across
program-disjoint folds, its AUROC is 0.869 and its average precision is 0.164
against a 0.038 support prevalence. Allocating the top quarter of sampled
program rows increases the supported-terminal rate from 3.80% to 11.85% and
the unique-supported-product yield from 1.94% to 5.92% per terminal attempt.
The Brier improvement over a fold-specific intercept is 0.00324 (95% clustered
interval 0.00208–0.00449).

Saved partial-state categorical features do not provide a material additional
signal. Their Brier score is slightly worse than applying the morphology score
to the same outcomes (0.03349 versus 0.03335), and their top-quarter supported
yield is 12.52%, only 5.6% above the morphology allocation. The clustered Brier
difference interval crosses zero (-0.00074 to 0.00048). Delayed partial-state
SMC is therefore not justified by this evidence.

The strict version-1 production gate retains plain terminal screening as the
current operational controller because the raw morphology probabilities are
under-dispersed and the row-level top-quarter allocation covers 23.8% rather
than 25% of unique program hashes owing to duplicate morphology rows. This does
not erase the ranking result. The next warranted challenger is a
support-preserving dynamic morphology proposal evaluated on unused programs;
it is not an authorization for potency guidance or production deployment.

## Frozen artifacts

- Census config:
  `configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_v1.json`
  (`7500db52feb13f644d9a9585df649ec36b051b1792c9543fb0db5840f1542602`)
- Census result file:
  `results/phase1/ugi_dynamic_frozen_prior_terminal_census_v1/result.json`
  (`af56a3aecbf441414e5dc71881cf242c73524f71f9689bf40ac95fac33654c84`)
- Census logical result:
  `3b440eff098047b8284a49f12d8f190fcd19164f4c9b0e1f086ee4fd3ef3e1e7`
- Validator config:
  `configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_validator_v1.json`
  (`da3fc4b49c4f39e6e578b82aed952c19b247f43ce9a09a675641855c3796e73c`)
- Validation result file:
  `results/phase1/ugi_dynamic_frozen_prior_terminal_census_validation_v1/result.json`
  (`cd6635e34e0fbba39e5aca21baa50b0c7b4817854c09bd33c54ecc108537be7b`)
- Validation logical result:
  `a45c44d0404f7ace6f95adefe52ac99ed3d42bc3017dce7df9abed40aae38eb2`
- Controller-analysis config:
  `configs/bio/phase1_ugi_dynamic_controller_analysis_v1.json`
  (`acdcef9e2be520162e6d8c9c70f3d56f3c96fac57d80dbe24990fcf01e19a7ca`)
- Controller-analysis result file:
  `results/phase1/ugi_dynamic_controller_analysis_v1/result.json`
  (`ad6fad9a79debe09eee1e630c242109dd0aa5f05a8a9c6f9bd732df40ecdae7a`)
- Controller-analysis logical result:
  `5b8a0f496d2a30edc5e90fb590646a1936595630d1b6c5522bb8c36e93572740`
