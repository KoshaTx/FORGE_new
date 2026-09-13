# Amine semantic residual: bounded development experiment

Date: 2026-09-08. This protocol is specified before residual fitting or outcome inspection.

## Question and comparison

Can the frozen step-9143 predictor use the four amine specifications already supplied to the
baseline decoder to rank its legal atom placements better? The specifications are heavy-atom
diameter, carbon-skeleton diameter, nitrogen count, and oxygen count. No donor, branching, or
additional tail targets are introduced. The failed committed-bond refresh probe remains stopped.

Compare the unchanged backbone with two identically initialized residual heads. Both receive
detached atom-aligned backbone hidden states and flow time. The semantic arm additionally receives
the four coordinates, scaled by fixed constants [10, 10, 4, 4]; the control receives zeros in those
four positions. Each is a 64-unit ReLU MLP with a zero-initialized output projection. The correction
applies only to variable amine atom logits. Backbone parameters and every other output remain
unchanged. A position-independent composition bias would cancel between equal-composition
placements; the nonlinear interaction with hidden states is the mechanism being tested.

The zero-input control matches architecture, initialization, parameter count, examples, noise,
optimizer and updates. Its four zero-input weight columns receive no data gradient. Thus it is
an information ablation, not a claim of identical effective feature capacity. Improvement over
extra fitting alone requires beating both the control and the untouched backbone.

## Population, partition, and exposure

Interpret only Ugi TRAIN structures. First preserve a complete coverage census, including
unsupported rows and their original source-weighted mass. The fitting population is the subset
within the existing decoder's program support with an extractable four-coordinate amine target.
This is a declared decoder-applicability experiment, not broader corpus coverage or permission to
shrink production support. Source weights within each subdivision retain their original ratios.

Partition by canonical amine identity using SHA-256 with a fixed domain and bucket 0 of 5 for
component-disjoint evaluation. Within the other four component buckets, reserve product-hash
bucket 0 of 5 for repeated-component evaluation. Domains, ledgers, and partition masses are saved
before any model evaluation or fitting. No split is retried to improve balance or outcomes.
The backbone has previously seen TRAIN; these are subdivisions for residual fitting, not unseen
backbone data. Report absent components, overlap, and sampled component coverage explicitly.

Draw 1,024 fitting products, 256 component-disjoint evaluation products, and 256 repeated-component
evaluation products with replacement under the corresponding normalized source measure. Preserve
duplicates and every draw. Each product is evaluated at flow times 0.2, 0.5, 0.8, and 0.95 using
native, paired chemistry corruption and exact target topology. Both fitted arms reuse identical
frozen features and noisy states. Extract features in CPU float32 batches of 64 with two threads;
store only the variable amine hidden states/logits plus complete replay metadata and hashes.

Fit each head for exactly 64 AdamW updates, batch size 64, learning rate 0.001, weight decay 0.0001,
gradient clipping 1.0. A fixed permutation covers all 4,096 product-time fitting examples once.
Optimize mean per-product cross-entropy over its variable amine atoms, then average products.
Keep the final checkpoint; do not select intermediate checkpoints, learning rates, seeds, or arms
using evaluation outcomes. Save update losses, exposure order, initial/final states, and pins.

## Evaluation and advancement

Use the original baseline amine selector with its existing support, feasibility checks, candidate
order, and deterministic maximum-score rule. Obtain candidate placements by observing that
implementation rather than reproducing its enumeration. The original target topology and bond
states define a teacher-forced reconstruction diagnostic. A placement is correct when replacing
only its amine states yields the same stereo-free constitutional product graph. Symmetry-equivalent
placements count as correct. Failed completions and absent targets are recorded, never filtered.

Compare candidate symbol-pattern universes across all arms, report any selected atom-state changes,
and require the same universe. Report exact selected correctness, ties and correct-rank intervals,
singleton populations, absent-target populations, coverage, and genuinely corrupted atom accuracy
and negative log likelihood. Stratify by flow time and by the two residual-evaluation populations.

The primary measure is mean exact selected correctness over all non-singleton candidate populations
in the component-disjoint evaluation draws, pooling the four prespecified times. The semantic arm
must strictly exceed both baseline and control on this measure, with no decrease in pooled
genuinely corrupted atom accuracy relative to either. Empty informative populations are insufficient
evidence. Candidate coverage/universes and all unaffected outputs must be preserved. Report the
repeated-component comparison and each time separately even if they disagree with the pooled
result. This small, single-seed development gate is not a statistical or molecular-realism claim.
Target-presence preservation is paired: neither comparison population may lose a correct target
that is present for baseline or control on the same draw/time, even if gains elsewhere offset it.

A failed gate stops this arm and preserves its negative result. A pass permits preparing a
separately pinned sampler integration and matched generation comparison. It does not satisfy the
user's goal. Every requirement in `PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md`, including full-reference
realism, validity, exact L1, diversity, novelty, calibrated visual review, and fresh confirmation,
remains necessary. No gate is weakened by this experiment.

## Execution and evidence

The versioned JSON configuration pins this protocol, census, inputs, checkpoint, source files,
seeds, and constants before fitting. Save failures separately from admitted results. Run only local
CPU work; no remote compute, routing, biological guidance, prospective selection, or molecular
generation is part of this study. Preserve runnable code, tests, numeric results and decision-log
findings, including an unchanged repository-wide test failure baseline where applicable.
