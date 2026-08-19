# Phase 1 matched synthesis-guidance preregistration

## Status

The experiment is specified but **not authorized to execute**. The restartable
sampler v2 equivalence receipt passes, while the downstream production
zero-guidance v2 rehearsal remains blocked by an authenticated value-source
supersession. No route coverage, guided result, biological improvement or
candidate outcome is reported here.

The machine-readable contract is
`configs/model/phase1_ugi_matched_synthesis_guidance_preregistration_v1.json`.
Its loader and pure adjudicator are
`src/forge/product/ugi_matched_synthesis_experiment.py`.

## Scientific question and arms

The causal question is narrow: under the same whole-product prior, productive
generation budget, terminal-completion budget, route-assessment ceilings and
final-candidate budget, does route information used during sampling yield more
distinct route-complete beyond-catalog candidates than applying the same route
assessment only after generation?

Four arms are frozen:

1. `unguided_product_prior`: the selected product prior without synthesis
   guidance;
2. `post_hoc_route_filtering`: the same prior, with every terminal and trace
   sealed before route assessment or selection;
3. `in_trajectory_smc`: synthesis value may alter ancestry before candidate
   lock at the frozen checkpoints;
4. `finite_library_baseline`: contextual ranking inside the frozen component
   catalog.

Only arm 3 versus arm 2 enters the causal promotion decision. Arm 1 supplies
the anti-collapse reference. Arm 4 is field-facing context and is explicitly
excluded from the generator-compute match; by construction it cannot satisfy a
beyond-catalog endpoint.

## Frozen schedule and compute

Calibration uses seeds `20260821`--`20260823`; evaluation uses independent
seeds `20260824`--`20260828`. There are no retries and no seed search. Each
generator arm uses 64 particles, eight sampling steps, one terminal rollout per
particle at checkpoints 2, 4 and 6, and one final completion per particle.

The derived per-seed generator-arm budget is:

- 1,280 product-transition calls: `64*8` main transitions plus
  `64*((8-2)+(8-4)+(8-6))` rollout transitions;
- 256 terminal completions: 192 checkpoint rollouts plus 64 final completions;
- ceilings of 768 logical/physical planner calls and 3,328 logical/physical
  verifier calls, derived from 256 three-role reservations of 3 planner and 13
  verifier calls;
- exactly 32 final candidates;
- caps of 10,800 wall seconds and 7,200 GPU-device seconds.

Unguided, post-hoc and SMC arms share these ceilings exactly. Realized physical
calls, cache hits, wall time and GPU time remain separately reported because
different terminals can induce different cache behavior. They may not exceed
their caps. Product-transition calls, terminal completions and final-candidate
counts must be exact. The finite-library arm receives the same terminal/route
assessment and final-candidate budget but no product-generator calls.

## Primary endpoint

The primary endpoint is the number among the 32 selected candidates of unique
constitutional products that satisfy every condition below:

- valid complete molecular graph;
- exact, qualified and forward-consistent Ugi L1 assembly;
- all three recovered precursor roles recursively closed through L2/L3;
- current evidence-qualified terminal leaves under the same frozen L3
  decision horizon;
- at least one role component outside the frozen role-specific 424-component
  catalog derived from the 112,386-product production assignments.

Products are deduplicated by canonical constitutional identity before counting.
Handle qualification, family projection, analogue evidence and route-model
likelihood do not close a route. The endpoint is therefore not a retrosynthesis
model score or a synthesis-success probability.

## Anti-collapse and oracle safeguards

Relative to the paired unguided arm, SMC may lose at most 0.02 absolute validity
and 0.02 absolute exact-L1 fraction. It must retain at least 90% of uniqueness,
90% of internal diversity, 90% of broad-distribution coverage, 80% of
beyond-catalog fraction and 80% of each role's effective component count. The
maximum concentration of any component within a role may rise by at most 0.10
absolute.

At least 99% of selected valid L1 candidates must receive a descriptive oracle
applicability assessment. Every oracle action must remain `abstain`, every
guidance score must remain null and biology cannot rank or promote candidates.
Applicability bins are reported descriptively only. This preserves the current
decision that the selected HeLa oracle is not authorized for generative
guidance.

## No-signal gates

Before nonzero guidance, all of the following must pass:

- zero-guidance restartable SMC is bitwise identical to the selected sampler;
- an exactly uniform effective ancestry law returns the identity ancestry and
  consumes no random seed;
- a slightly nonuniform law still resamples;
- equal guidance increments do not erase nonuniform base weights;
- complete morphology-program fields remain identical across matched arms.

Failure of any gate blocks promotion. These are identity and sensitivity tests,
not evidence that the synthesis value is scientifically adequate.

## Calibration and promotion rule

The calibration sweep is `0.25, 0.5, 1.0, 2.0`, with zero retained only as an
identity control. Among strengths satisfying every safeguard, select the
smallest strength attaining the largest positive aggregate primary-endpoint
improvement over post-hoc. If no strength improves on post-hoc, guidance is not
promoted and evaluation is not used to tune a replacement.

On the five sealed evaluation seeds, SMC is promoted only if:

- it strictly beats post-hoc on the primary endpoint for all five paired seeds;
- the aggregate paired difference is positive;
- every no-signal, compute, provenance, route and safeguard gate passes.

A tie or loss on any seed fails promotion. With five strict wins, the exact
one-sided sign probability under a symmetric null is `1/32 = 0.03125`. There is
no retry. A null or adverse result removes the causal synthesis-guidance claim;
it does not rewrite the experiment or erase the route resource.

## Current authenticated blocker

The v2 restartable equivalence configuration and receipt authenticate at
SHA-256 `c34b8b9f...` and `30a52af9...`, respectively, and pass bitwise batch
partition equivalence. The downstream v2 production rehearsal fails before
generation or routing because both historical fresh-pool v2 and v3 configs pin:

```text
src/forge/value/synthesis.py
expected: dba5b11067513e4f431f78f3b9a316dfe0d58c5f046210c170c55b9a0e241e3c
observed: 6e77063f45012144912883f16a12698eda809446635c94d59d1ab6211993f189
```

All other declared v2/v3 inputs reproduce. The current source is a coherent
structured/Pareto implementation and its direct tests pass, but the frozen
v1--v5 route/value configurations all own the earlier source hash. This is a
pending value-source supersession, not permission to edit historical pins.

The legitimate repair path is:

1. retain every historical config and result unchanged;
2. semantically replay the current structured value implementation against the
   frozen raw inputs and ledgers;
3. require byte-identical ledgers when semantics are claimed unchanged, or
   explicitly characterize any delta;
4. write a new versioned source-qualification receipt and new versioned
   route/value configs that pin the current source;
5. rebuild the cumulative source identity and rerun the production
   zero-guidance rehearsal v2;
6. separately freeze a scalar/ordering policy and a current L3 snapshot before
   authorizing nonzero guidance.

No hash check may be relaxed and no old evidence result may be overwritten.

## Remaining execution gates

Even after value-source supersession, nonzero execution remains blocked until:

- the v2 production zero-guidance rehearsal passes end to end;
- a scientifically justified synthesis ordering or scalar is frozen without
  being called a synthesis-success probability;
- current L3 evidence and decision horizons are refreshed and sealed;
- per-arm selection and metric ledgers are implemented and authenticated;
- candidate selection remains blind to the private holdout.

This preregistration fixes the question and failure rule now while preserving a
fail-closed distinction between a runnable controller, a qualified synthesis
value and an empirical guidance result.
