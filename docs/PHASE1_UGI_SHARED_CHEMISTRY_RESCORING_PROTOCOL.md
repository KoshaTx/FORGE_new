# Shared-chemistry topology rescoring: exploratory generation test

Date: 2026-09-08. Defined before successor generation. Status: implementation and verification.

## Hypothesis and selection history

The combined candidate-conditioned completion rule C failed its frozen two-role gate and remains
rejected. Its B control used the existing baseline chemistry forward to score legal completions
under alternative topologies. B improved target-class probability and NLL over the original A
topology score in both roles of that small TRAIN reconstruction probe. This motivates a distinct
generation experiment: feed shared chemistry confidence back into topology selection.

Selecting B for this follow-up is **post hoc and exploratory**. The previous probe is motivation,
not confirmation or proof of realism at terminal flow time 1. Do not alter C's failed decision,
combine its head rule with B's tail rule, tune a coefficient, or use the old probe as independent
confirmation. Prior measured-only and measured-heavy training changes also failed their realism
tests; this experiment changes terminal inference without fitting model weights.

## Frozen comparison and scope

Use exactly the original first 256 requests, order, program conditions, four baseline amine
coordinates, flow seed, 32 steps, batch size 128, float32 CPU arithmetic, and checkpoint step 9143
from the baseline arm of
`configs/multireaction/ugi_donor_slack_tiered_head_group_entropy_terminal_offset_bondweight2_vs_amine_joint_support_cpu_seed0_v1.json`.
The draw is `ugi_joint_substitution_semantic_program_draw_seed0_v6/program_draw.json`, SHA-256
`ad5f6afdbd074ba2a11459ab7a74e276f17f9701dddce8522bf04f7f95133f7a`.
Use two CPU threads and deterministic kernels. No training, remote compute, biological guidance,
routing, repair, retry, support expansion, heldout structure access, or candidate filtering.

Retain the original baseline and the exact saved failed treatment as controls. Reproduce every
baseline sample field and sampler summary with a passive capture, and authenticate its historical
row projection against all 256 saved baseline rows. The treatment must have the same authenticated
requests and runtime settings. These are reused development requests, never fresh confirmation.

The experiment runs outside the production sampler. Preserve its source and historical pins.
Capture the original sampler's native endpoint prediction A and post-topology chemistry prediction
B with read-only hooks. Under this exact baseline there are two time-1 forwards in each of two
128-record batches. Save immutable CPU copies of all model inputs and outputs, batch identities,
layout masks, original final states, and RNG states. Assert the call schedule and conditioning
contract. Hooks must be removed on success or failure, make no model or random calls, and leave
all original outputs, model state and random streams equal to the uncaptured control.
Here the control omits the forward-input capture; both executions retain minimal read-only
endpoint observation to compare actual terminal arrays and sampler-local RNG states.

## Successor rule

For every request, use the same complete original B topology as the common background. Construct
the unchanged original ordered amine and constructive ester candidate sets using only the
declared program, four amine coordinates, fixed core/attachments and registry policies. Do not
compute target graph labels or consult variable target atom/bond identities. Preserve every
candidate serialization, root-ordering constraint and original pre-choice A score. Orient closure
endpoints with the original A logits, including their original float32 addition and tie rule.

For each role separately, install one alternative in that common background while keeping all
native atom/bond slots, other-role topology and conditioning fixed. Use shared B predictions for
the unchanged strict chemistry decoder, with original baseline policy flags and no new topology
draw inside the strict decoder. Require completed pointers to equal the requested graph and apply
the original final molecular reconstruction check. Score successful completions by the sum of
full-vocabulary log-softmax at **all affected-role variable atom, parent-bond and closure-bond
states**. Exclude fixed, root-parent and padded coordinates; assert identical masks across that
role's candidates. Use float64 for probability normalization and summation, retaining float32
model predictions. This is a heuristic completion-confidence score, not a calibrated likelihood,
realism score or synthesis probability.

The new topology score is exactly **A + completion confidence**, with coefficient **1**. Score both
roles against the same original background before either selection. Use a separate CPU Torch
generator seeded **2026090831**, visiting requests in ascending order and amine before aldehyde.
Sample the original ordered candidate softmax, retaining serialization multiplicity. Install both
chosen roles simultaneously; never score the aldehyde against the newly selected head.

If either role's choice law is unavailable, or any candidate in either role has an expected strict
abstention or invalid final molecule, record the complete reason and fall back for the **whole
product** to its already sampled baseline topology and final product. Do not drop a candidate,
assign a failed candidate negative infinity, resample A, draw another background, or consume
rescore RNG on fallback. Retain all attempted counterfactual completions and mark any unevaluated
work explicitly. Unexpected errors and topology mismatches are operational failures.

For successful reselections, recompute chemistry once on the jointly selected complete topology
using the original 128-record batch layout and saved conditioning. Apply the unchanged strict
decoder and molecular validation. A joint completion may fail even when its separate role
alternatives passed: retain that final failure without repair or reselection. Fallback rows retain
their original baseline completion exactly. Count every final forward and any evaluated fallback
row; do not undercount uniform batch work.

Before coefficient 1 runs, coefficient 0 must bypass completion scoring and reselection and
reproduce the captured baseline topology, final products and RNG state exactly. There is no
zero-delta draw from A. This and passive-capture identity qualify the terminal replay experiment;
production integration would require its own sampler equivalence tests if the arm advances.

## Budget and artifacts

Preflight the **complete** two-role candidate budget over all 256 requests before any successor
completion. The cap is **16,384** counterfactual strict completion examples. A metadata-only bound
suggests at most 11,916; the exact authenticated enumeration is authoritative. Exceeding the cap
stops the experiment, without a selected prefix or changed cap. Chunk counterfactual decoding in
groups of 32; retain the original 128-row layout width and conditioning when slicing predictions.
The successor has at most two final 128-row neural forwards. Baseline control and capture each
execute their original complete flow and are counted separately.

Persist configuration and input/source hashes before execution, each completed batch and every
failure, all original/selected states, candidate scores and completion outcomes, baseline and
rescore RNG states, model/example/decoder counts, and exact-L1 assessment. Do not overwrite an
attempt or silently rerun a failed attempt. Numerical findings must remain attributable to the
actual frozen source bytes. Preserve the earlier calibration-layout incident as historical context;
it supplies no data for this experiment.

## Decision and full goal requirements

Use `docs/PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md` unchanged. Require 100% structural validity
and exact Ugi L1 reconstruction, zero final program abstentions/fixed violations, and no loss in
unique exact-L1 products, effective component count, pairwise diversity, any-component novelty,
whole-product novelty or unique novel exact-L1 products relative to the paired baseline. Report
each role's novelty incidence, distinct count and effective count. Keep every requested attempt in
the prescribed denominator. If a final validity/exact-L1 failure makes the 100% requirement
unattainable, stop subsequent successor batches and retain all unattempted requests explicitly.

All six unchanged full measured-TRAIN role-descriptor distances must improve: Wasserstein, energy
and MMD-squared under both attempt and unique-product weighting. The admitted-reference subset
cannot replace the full reference. Report graph-sensitive distances and familiarity/novelty strata
with their post-treatment limitations. Do not promote on confidence, descriptors or fingerprints
alone. Keep the frozen blinded visual criteria, actual reviewer calibration and identity, and a
separate fresh confirmation after any development pass. A pending reviewer is not a pass.

A failed preservation or realism gate rejects this arm; no coefficient or role-specific variant
is selected from this run. A development pass permits the subsequent integration/review/confirmation
work only. This isolated experiment cannot declare the complete realism goal or global Phase 1
definition of done achieved. Preserve focused validation and the repository-wide failure record.
