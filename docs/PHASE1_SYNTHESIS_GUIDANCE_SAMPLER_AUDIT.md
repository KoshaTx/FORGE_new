# Phase 1 synthesis-guidance sampler audit

## Decision

The selected Ugi generator can support a future synthesis-guided experiment.
Its trajectory and terminal-completion APIs are now restartable, but no exact
planner rollout, shared budget or real synthesis potential is connected. Do not
wire the exact route assessor to terminal output and call the result guided
generation. That would be post-hoc assessment.

The immediate implementation order is:

1. preserve the frozen generator and terminal-completion behavior (**passed**);
2. expose restartable trajectory operations with exact zero-guidance
   equivalence (**passed**);
3. validate controller probabilities, keyed substreams and zero-guidance bypass
   against deterministic fake values (**passed**);
4. implement diagnostic fixed-budget rollout admission (**passed**), then
   matched guided/post-hoc orchestration and expand qualified synthesis
   support; and
5. only then run a preregistered guidance-strength sweep against a matched
   post-hoc arm.

Biological guidance and production synthesis guidance remain disabled.

## Observed sampler path

`forge.product.ugi_joint_sparse_flow.sample_ugi_joint_sparse_terminals`:

- samples noisy offspring, atom, parent-bond and decoration states from
  role-conditioned marginals;
- advances those categorical states for the declared number of flow steps with
  `_rstar_step`;
- invokes a final model pass at time one;
- constrained-decodes a valid offspring tree only after the flow trajectory;
  and
- returns terminal topology and chemistry logits.

`forge.product.ugi_joint_end_to_end_sampling.sample_ugi_joint_end_to_end` then:

- samples sparse closure placement;
- assembles the fixed Ugi core and generated role-specific exterior topology;
- applies valence-constrained terminal chemistry decoding;
- sanitizes the complete molecular graph;
- recovers the three exact component graphs; and
- optionally verifies exact L1 forward reconstruction.

Consequently, an exact role-qualified component identity and exact route value
do not exist at an arbitrary intermediate `_rstar_step` state. They exist only
after constrained topology decoding, closure placement, valence-constrained
chemistry realization, sanitization and component recovery.

## Missing interfaces

The current sampler now exposes cloneable partial trajectory state, exact-step
advance, constrained terminal finalization, terminal molecule completion,
within-program ancestry and keyed particle extraction. A diagnostic
fixed-budget executor also enforces terminal admission before fake-value
evaluation. It still does not expose:

- an end-to-end real-model completion-rollout driver;
- a frozen scalar aggregation policy for structured route values;
- complete route-cache trace accounting; or
- a shared productive-compute ledger for guided and post-hoc arms.

These are engineering blockers, not evidence that synthesis guidance is
conceptually impossible.

## Safe first coupling

The first defensible exact-planner coupling is terminal-rollout
Feynman--Kac/SMC:

1. advance a particle population to a frozen intermediate checkpoint;
2. clone each partial state;
3. complete a fixed number of rollouts with the unchanged generator and
   terminal decoder;
4. assess only the valid completed products with the structured route-value
   layer;
5. aggregate rollout values under a preregistered rule; and
6. resample particle ancestry before continuing generation.

This changes molecular transition probability before candidate lock. A future
distilled value model may replace expensive terminal rollouts only after it is
trained and tested with component-family-disjoint splits and explicit
abstention.

Direct route scoring of invalid partial graphs is prohibited. Direct per-logit
route modification is not required for the first implementation.

## Coverage constraint

The current composed audit closes 45 of 1,007 products and 26 of 610 unique
generated components. The registry closes 50 of 424 admitted components, while
519 generated identities are outside the registry and are correctly labeled
missing knowledge.

At this coverage, an exact-only controller would mostly reward a small set of
documented identities. Its apparent gain could be component collapse rather
than improved allocation over open-ended chemistry. Production guidance
therefore remains blocked until either exact support materially expands or a
separately qualified broader route proposer/value model is available.

## Refactor gate result

The behavior-preserving operations are now exposed as:

```text
initialize_ugi_joint_sparse_state
advance_ugi_joint_sparse_state
finalize_ugi_joint_sparse_terminal
complete_ugi_joint_terminal
```

The existing public samplers remain compatibility wrappers. The artifact-level
audit uses the selected step-1,000 checkpoint, 12 fixed programs, eight flow
steps, seed 20260802 and batch sizes 1, 4 and 12. Every topology, chemistry and
hidden-state array is bitwise equal to the frozen monolithic schedule. The
result is frozen at
`results/phase1/ugi_restartable_sampler_equivalence_v1/result.json`.

## Controller and rollout-plumbing smoke tests

The initial deterministic fake-value smoke tests now establish:

- exact zero-guidance identity ancestry without consuming a resampling seed;
- the analytically expected 0.25/0.75 ancestry probabilities in a two-particle
  example; and
- deterministic keyed substreams independent of coordinate ordering.

The versioned fixed-budget diagnostic at
`results/phase1/ugi_fixed_budget_rollout_plumbing_v1/result.json` additionally
establishes:

- four requested completion rollouts produce four explicit records;
- only the two sealed, valid and exact-L1 terminals reach the fake evaluator;
- invalid and nonexact-L1 terminals remain distinct dispositions;
- realized logical planner and verifier calls are counted separately; and
- ancestry cannot cross frozen morphology-program groups.

This is still a fake-value plumbing audit. It neither aggregates real
structured route values nor executes the route planner.

Still required are end-to-end tests that no value is evaluated on uncompleted
partial graphs, morphology programs remain identical across arms, productive
generation/completion/planner/verifier budgets match, post-hoc route calls occur
only after terminal lock, and shared censoring applies on budget exhaustion.

Passing this smoke test qualifies plumbing only. It does not qualify a
synthesis signal.

## Authorization boundary

The structured value implementation and this sampler audit authorize the
restartable zero-guidance refactor and fake-controller tests. They do not
authorize:

- a scalar synthesis objective;
- synthesis-success probability claims;
- production route-guided sampling;
- candidate locking;
- family-template promotion; or
- biological tilting.
