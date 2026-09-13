# Head specifications before backbone message passing: bounded development probe

Date: 2026-09-09. Specify this comparison before fitting or inspecting its outcomes.

## Question and comparison

Can the existing four head specifications help a trainable step-9143 backbone make better
reconstruction and legal-placement predictions than matched additional fitting with zero targets?
The coordinates are heavy-atom diameter, carbon-skeleton diameter, nitrogen count and oxygen count.
They are already supplied to the original baseline decoder. No additional chemistry targets,
component identities, biological labels or new support are introduced.

The earlier semantic residual detached backbone hidden states and changed final head atom logits
only. Measured-only full-model tuning supplied no additional semantic fields and changed the source
measure. Neither result establishes the effect of this comparison. Conversely, this experiment
changes both conditioning access and trainable adaptation relative to the old residual; it does not
isolate injection location alone by comparing the two historical experiments.

Load the authenticated original checkpoint. Keep an untouched evaluation baseline and make two
independent copies. Each copy receives an identically zero-initialized affine projection from four
coordinates to the original hidden dimension. Divide the coordinates by the original fixed scales
[10, 10, 4, 4]. Add the projection on valid amine-role nodes immediately before the first backbone
attention block, using a scoped hook on the unmodified production model. The control supplies zeros;
the semantic arm supplies the informative coordinates. Both can learn the shared projection bias.
The control's four input-weight columns receive no data gradients, so equal parameter count does
not mean equal effective feature capacity. Every original model parameter remains trainable.

Initial evaluation predictions must exactly equal the untouched model for both arms on the first
original fitting batch. No hidden-state or program-memory cache may cross an optimizer update.
The injection can affect every output through attention; preservation checks therefore cover the
other roles and topology as well as head atoms. The wrapper must remove its hook even on failure.

## Population, exposure, and training

Reuse the exact semantic-residual-v2 selection, component/product subdivisions, original source
weights, saved corruption seeds, four times [0.2, 0.5, 0.8, 0.95], and exposure permutation. Authenticate
all inputs and the original coverage census before materializing selected records. Every selected
record must pass a metadata-first Ugi TRAIN mask. The 1,024 fitting draws and two sets of 256
evaluation draws retain duplicates and all failures. These evaluation subdivisions exclude products
or components from this new fit only; they are not held out from the original backbone.

The applicability restriction remains explicit: 1,966 of 66,464 Ugi TRAIN products, representing
0.272008221% of the original source-weighted Ugi measure. The other rows remain in the frozen
coverage ledger. This experiment neither expands the decoder nor reduces the declared corpus.

Regenerate the original native corruption once per saved batch from the unchanged checkpoint
program-role marginals, requiring exact clean/noisy/topology-conditioned tensor hashes. Preserve
original evaluation batch shapes. For the fitting permutation, gather stored noisy rows and pad
only masked positions to the widest source batch represented in that optimizer batch. Do not
re-noise, discard records or shrink molecular support. The new mixed batch shape is shared between
arms; it need not be numerically identical to the old feature-extraction geometry.

Train each arm for exactly 64 AdamW updates, batch size 64, covering all 4,096 product-time examples
once. Use learning rate 0.0001, weight decay 0.00005 and gradient clipping 1.0, as in the original
checkpoint training recipe. Reset optimizer state identically. Keep only the final update as the
evaluated checkpoint; preserve progress checkpoints for recovery, not model selection. No seed,
learning-rate, exposure, time or role sweep is permitted by this probe.

Retain every loss term and coefficient from the checkpoint's actual `semantic_objective`, including
equal-present-role chemistry weighting, role/core consistency, repeat consistency, offspring and
junction losses, and the second chemistry pass with target topology. The two passes share native
corrupted chemistry. Use the existing loss implementation explicitly; the generic specialization
runner does not wire all these terms. Source balancing is already in the saved draws and must not
be applied a second time to losses. The backbone stays in evaluation mode with gradients enabled,
so dropout is intentionally disabled in both arms. This is a matched diagnostic departure from the
original stochastic training procedure, not exact reproduction of that procedure.

Run locally in deterministic CPU float32 with two threads. Record actual forward/example/update
counts, first-update timing and subsequent update timings, finite-gradient checks, initial/final
states, and all exposure/input hashes. The maximum is 451 model forwards and 28,864 model examples:
three initial identity forwards, 256 training forwards and 192 final/baseline evaluation forwards,
each of size 64. There are 128 optimizer updates across the two arms. Preserve interrupted work and
failures; no failed or partial metrics can masquerade as an admitted result.

## Evaluation and necessary advancement criteria

Evaluate untouched baseline and the two final fitted arms on both saved evaluation subdivisions,
all four times, and both native-noisy and target-topology inputs. Save all predictions and coordinate
sufficient statistics. Report variable atom and bond correctness/NLL by role, including actually
changed versus unchanged input coordinates. Report variable topology-pointer accuracy with native
noisy topology. Empty structurally impossible field/role strata remain visible with zero counts.

For the original target-topology diagnostic, run the unchanged baseline amine placement selector
on every saved draw/time. Correctness is constitutional whole-product equality after substituting
head atoms on the target graph; all target bonds and non-head atoms remain fixed. Preserve every
candidate, singleton, absent target and invalid candidate. Require the same legal symbol universe
between arms and paired target-presence preservation in both subdivisions.

Necessary advancement requires all of the following:

- Component-disjoint selected correctness over all non-singleton candidate populations strictly
  exceeds both untouched baseline and zero-target control.
- Repeated-component selected correctness over the same population definition does not decrease
  relative to either comparator.
- Corrupted and unchanged variable atom/bond correctness does not decrease for any role in either
  topology-input mode or evaluation subdivision, relative to either comparator.
- Native-noisy variable-pointer accuracy does not decrease in either subdivision relative to either
  comparator. Candidate universes, paired target presence and mechanical identity checks pass.

The primary baseline has only five errors among 852 target-present competing placements. Keep all
68 absent-target cases and 104 singletons in the reported 1,024 component-disjoint attempts. Do not
change the panel to evade this ceiling. Report each time separately even when it disagrees with
the pooled comparison; an attractive time or head-only subresult cannot rescue a failed gate.

These are necessary development checks, not statistical evidence of realism or generalization.
They do not measure newly generated whole-molecule validity, exact L1, diversity or novelty, and a
head-conditioning improvement does not establish improved tails. A failure stops this arm without
automatic generation or enlarged fitting. A pass permits preparing a separately pinned integration
comparison; every requirement of `PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md` still applies, including
full-reference realism, unchanged structural gates, calibrated blind review and fresh confirmation.
The active user goal is not complete until those requirements are demonstrated.

## Evidence and validation

Freeze a JSON configuration with protocol, code, input and prior-result SHA-256 pins before model
work. Archive the executed sources and input receipts. Authenticate inputs again on completion.
Save numerical results, failed attempts, tests and independent review, and record the finding in
the decision log. No production sampler/source, original weights, chemistry gates, data splits,
held-out structure access, routing, biological guidance, remote compute or candidate selection is
changed or introduced. Repository-wide definition of done remains unmet while its preserved full
test suite is failing; a local probe result must not be represented as release readiness.
