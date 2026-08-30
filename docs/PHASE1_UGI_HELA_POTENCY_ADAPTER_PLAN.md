# Phase 1 Ugi HeLa potency-adapter diagnostic

## Objective

Test whether a small, frozen-backbone adapter can steer the final Ugi product distribution toward
higher **predicted** HeLa mTP while preserving verified exact-L1 assembly, validity, and diversity.
This is a computational diagnostic. It does not establish measured activity, experimental synthesis,
formulation performance, or delivery efficacy.

The previous morphology-level potency tilt is a frozen negative result. This milestone tests a
different intervention: the molecular denoiser receives a percentile target at the atom, bond,
pointer, and closure level. No generator weights are retrained unless they belong to the new adapter.

## Condition and model delta

The typed request is

```text
PotencyCondition(
    endpoint_id="expt_Hela",
    target_quantile=0.90,
    policy_id="ugi_hela_mtp_percentile_adapter_v1",
)
```

The target is a percentile rather than a raw assay value. Training labels come from the 1,100
single-compound `YX_2024::HeLa` observations in the LNPDB-only potency corpus. Each component-
disjoint training fold maps its values to a train-only empirical CDF. Cross-study normalized labels
are never pooled.

Every Transformer layer receives a zero-initialized bottleneck residual adapter. The adapter depends
on the requested percentile and the current flow time. The authenticated generator, program encoder,
output heads, and reaction-core decoder remain frozen. A null condition bypasses all adapter
parameters, even after training, and must reproduce the base logits bit for bit.

The adapter is admitted only for `ugi_3cr_agile`, the frozen endpoint and policy identifiers, the
declared percentile support, and flow-time intervals qualified below. A request outside that support
fails; an unqualified time interval contributes an exact zero residual.

## Adapter fit

Each update contains two independently drawn Ugi batches:

- 64 measured products, balanced across train-fold mTP quartiles, train the primary masked
  denoising term conditioned on their train-fold ECDF percentile;
- 64 unlabeled train-fold products, drawn from the frozen source weights, distill the null model into
  the median condition (`q=0.5`) on the same corrupted states.

Only adapter parameters are trainable. The frozen full contract is 500 AdamW updates at learning
rate `3e-4`, weight decay `1e-4`, gradient clipping `1.0`, deterministic float32, and no PCGrad.

## Signal qualification

Cross-fitting uses all five folds of both prespecified component-disjoint schemes:

1. held amine head;
2. held aldehyde/isocyanide pair.

For identical held-out corruptions, the score is the null denoising NLL minus the `q=0.9` denoising
NLL. It is evaluated at early, middle, and late flow times. A deterministic shuffled-training-label
adapter is fitted under the same optimizer and random stream.

A time bin qualifies within a split scheme only if all of the following hold:

- high-versus-low potency AUROC is at least `0.60`, with component-cluster bootstrap lower bound
  above `0.50`;
- Spearman correlation with potency residualized against train-only role morphology is at least
  `0.15`, with bootstrap lower bound above zero;
- real-minus-shuffled AUROC is at least `0.05`, with paired cluster-bootstrap lower bound above
  zero.

Only middle or late bins may guide generation. The production policy uses the intersection of bins
that qualify under both split schemes. If that intersection is empty, the milestone ends as a
negative result and no conditioned checkpoint is promoted.

## Matched generation experiment

Nonzero execution remains separately gated. If authorized after signal qualification, seed 0 first
generates 1,024 attempts per arm with the same Ugi program rows, order, source randomness, 32-step
budget, strict reaction-core decoder, and no repair or retry:

1. null condition;
2. potency condition at `q=0.9`;
3. shuffled-label condition at `q=0.9`;
4. post-hoc ranking of the already generated null samples.

The existing completed-molecule ensemble remains the final assessor. Per unseen-component pattern,
the oracle budget is the minimum applicability-eligible count across arms. Candidates enter that
budget in a deterministic potency-blind hash order. Unsupported products abstain and contribute zero
to the primary per-attempt endpoint.

The primary endpoint is unique, applicability-supported, conservative-high, verified exact-L1
products per 1,000 generation attempts. Promotion requires all of:

- at least 20% relative and 5-per-1,000 absolute improvement over null;
- positive paired-program bootstrap lower bound;
- validity and exact-L1 yield each within 5 percentage points of null;
- at least 90% of null distinctness and 80% of null Shannon effective count;
- zero fixed-state violations and support overflows.

Only after seed 0 passes may the same contract be run over all three frozen base checkpoints with
3,072 attempts per seed and arm. Across three training seeds, results are descriptive means, sample
standard deviations, exact counts, and seed ranges—not bootstrap confidence intervals over training
variation.

## Implementation and authorization state

The code path, config, CPU smoke experiment, adapter overlay format, sampler condition, statistical
gates, and promotion rule are implemented. The full cross-fit was subsequently authorized and run
under `configs/bio/phase1_ugi_hela_potency_adapter_full_v1.json`. It completed all ten
component-disjoint folds but qualified no middle or late flow-time bin because the real-label
adapter did not exceed the shuffled-label control by the frozen AUROC margin. No final adapter was
promoted. Every nonzero guided-generation run remains disabled, and the matched generation
experiment must not run from this negative result.

## Paired ordinal v2 diagnostic

The v1 failure attribution identified two correctable design problems: its real-label and shuffled-
label arms did not begin from the same adapter state, and ordinary endpoint denoising did not force
the network to use the requested percentile. The v2 diagnostic addresses those problems without
changing the frozen generator or running guided generation.

For every optimizer step, the real and shuffled arms now receive the same measured molecules,
retention molecules, flow times, corruptions, initialization, optimizer settings, and update count.
Their only difference is the assignment of measured potency percentiles. Every measured molecule is
evaluated under three counterfactual requests, `q10`, `q50`, and `q90`. The primary loss reconstructs
the molecule under its nearest percentile anchor; an ordinal margin term requires closer anchors to
reconstruct it better than farther anchors. A paired receipt records these invariants for every fold.

Validation retains the five held-head and five held-aldehyde/isocyanide-pair folds. In addition to
the v1 AUROC and morphology-residual gates, each flow-time bin must now show both:

- strict `q10 < q50 < q90` reconstruction preference for low-potency products and the reverse
  preference for high-potency products, with an outer-quartile monotonic fraction of at least `0.30`,
  a real-minus-shuffled gap of at least `0.05`, and a paired cluster-bootstrap lower bound above
  zero;
- nearest-anchor accuracy of at least `0.45`, a real-minus-shuffled gap of at least `0.05`, and a
  paired cluster-bootstrap lower bound above zero.

Only a middle or late flow-time bin that passes every original signal gate and both direct ordinal
gates under both component-disjoint schemes can produce a checkpoint. No checkpoint authorizes
nonzero guidance; that still requires a separate decision after the cross-fit result is reviewed.

The measured data limitation is unchanged. `YX_2024::HeLa` contains 1,100 products but only 20
independent heads and 55 independent aldehyde/isocyanide pairs. It contains no newly measured head
or tail chemotypes beyond that factorial grid. The implementation records this explicitly and does
not fabricate examples or pool noncomparable assays from other studies. Acquiring genuinely new
measured chemotypes would strengthen future transfer tests, but prospective measurement is outside
the present computational paper and is not an execution task here.

The v2 config is
`configs/bio/phase1_ugi_hela_potency_adapter_ordinal_v2.json`. Its full cross-fit and every nonzero
guided-generation run remain disabled pending separate explicit authorization.

## Direct partial-state value diagnostic

The completed v2 result showed that reconstruction under requested percentiles is an unnecessarily
indirect potency objective: its strongest held-pair middle-time arm contained directional signal but
failed the prespecified nearest-anchor accuracy gate. The next bounded diagnostic therefore predicts
the final HeLa mTP value directly rather than inferring it from which percentile reconstructs the
molecule most easily.

The authenticated Transformer remains in evaluation mode with every parameter frozen. On explicit
request only, it exposes its final node representations for a corrupted state. A small value head
pools these representations by global mean, global maximum and declared Ugi precursor role, appends
flow-time features, and predicts a scalar training-fold-standardized mTP value. It receives no
component identifiers, stored precursor graphs or fragment tokens. A Huber regression term is paired
with a within-batch ordinal contrast, so both value calibration and ranking are trained directly.

The complete 1,100-product LNPDB YX HeLa factorial is evaluated under the existing five held-head
and five held aldehyde/isocyanide-pair folds. Real and shuffled heads share exact initialization,
partial-state features, molecule batches and optimizer draws; only the training-label permutation
differs. Evaluation remains separate at early, middle and late flow times and retains the original
AUROC, morphology-residual Spearman, shuffled-control and clustered-bootstrap gates. A heavy-atom-
count prediction head over the same frozen summaries is a structural plumbing control. Only middle
or late bins passing every gate under both split schemes can produce a value-head checkpoint.

The implementation is frozen in
`configs/bio/phase1_ugi_hela_partial_state_value_v1.json`. Its verified local smoke is engineering
evidence only. On 2026-08-30 the user explicitly authorized the full H100 cross-fit. A passing
cross-fit still does not authorize nonzero guided generation.
