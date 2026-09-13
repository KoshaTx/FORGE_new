# Committed-atom bond refresh: bounded hypothesis test

Date: 2026-09-08. Status: implementation of the prerequisite probe; no positive result yet.

The objective remains improved lipid realism with preserved validity, exact constitutional Ugi
reconstruction, diversity and novelty, without weakening any existing gate. This experiment
tests whether the frozen model can use atom commitments made by the strict decoder when it
predicts the remaining bond states. It is not an additional chemistry-flow trajectory.

## Mechanism and limitation

The current decoder commits ester constraints and complete atom states, then continues choosing
bonds from predictions made before those atom assignments. The proposed successor would evaluate
the frozen model once after the complete atom loop and component-support checks, immediately
before variable bond selection. It would copy actual committed atom states, selected topology,
adapter-fixed chemistry and forced ester bonds into the existing model input; uncommitted bond
inputs would retain their original values. Only parent-bond and closure-bond logits could change.
No earlier atom or topology choice would be reconsidered.

This differs from the previously failed extra 8/16-step chemistry flows, which redraw chemistry
before strict decoding. It cannot correct already committed atom placement. Supplying correct
training commitments is an oracle sensitivity test; generated commitments can be wrong.
The failed treatment's measured-frequency tail-count law also bypasses neural scores, so the
proposed successor uses the frozen amine baseline, not that failed treatment.

## Prerequisite: TRAIN response probe

Use the authenticated step-9143 checkpoint and unchanged support policies. Select sixteen
source-adjudicated measured TRAIN products from the admitted population before evaluating any
prediction. Deterministic selection must cover every admitted measured precursor component;
otherwise retain the fixed-bound failure. No heldout molecule is interpreted or used for selection.

At each prespecified flow time, 0.5 and 0.9, use the same native deterministic corruption and
exact target topology for all conditions:

1. Stale input: corrupted chemistry with the target topology.
2. Identical-input repeat: the exact same tensors, for numerical reproducibility.
3. Correct commitments: replace atom states and the decoder-defined forced ester bonds with
   their known training values; leave every evaluated variable bond input bit-identical.

Locate oracle ester commitments using the existing decoder routine and check every commitment
against the target. Exclude these bonds and every fixed/padded coordinate from evaluated losses.
Use the declared bond vocabulary. Report genuinely corrupted uncommitted bonds separately from
unchanged bonds, including per-role target negative log likelihood, accuracy counts and exact
bond-vector recovery. Equal role weighting prevents long tails dominating the primary loss.

Advance only if **both flow times** pass all prespecified checks: every role has a nonempty
corrupted-bond stratum; repeated inputs agree within tolerance; role-balanced target loss improves
beyond the repeat tolerance; no role loses corrupted-bond accuracy; and at least one corrupted
bond is corrected. Empty strata mean insufficient evidence and no advancement. Preserve every
negative result. No model fitting or molecular generation occurs in this prerequisite.

## Conditional sampler implementation

Only after the prerequisite passes, implement an explicit optional bond-refresh callback through
the existing sampler functions, with `None` preserving the original execution. Keep model/provider
logic in a separate module. The callback receives copied committed states and masks and may
return only finite, correctly shaped bond logits. Do not use stack mutation, observational hooks,
or a copy of the entire strict decoder to implement the intervention.

Archive any historical source bytes changed by the explicit extension and give the successor
new source pins. Require disabled-callback and identity-provider controls to reproduce every
baseline product, topology, abstention, terminal tensor and RNG endpoint exactly. Count additional
model calls explicitly; do not consume additional random draws. Fixed state, chemistry support,
atom-count limits, branch budgets, decomposition and exact forward reconstruction remain enforced.

## Conditional matched sampling test

Prepare a separate hash-pinned 128-request pilot only if the prerequisite and implementation
controls pass. The comparator is the original amine baseline on the same full batch. Freeze the
request, seeds, provider, call budget and decision before generation. Preserve failed attempts;
no repair, replacement draw, completed-product filtering or candidate selection is permitted.

The paired baseline must retain 128/128 observed validity and exact L1. Apply every additive
preservation requirement in `PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md`, including all-attempt
novelty, unique novel L1 yield, effective component count, uniqueness and pairwise diversity.
Report distinct, effective and novel counts for each precursor role separately.

A preservation pass alone is insufficient. All six frozen train-reference realism distances,
complementary graph diagnostics and calibrated blinded appearance requirements remain necessary.
Any eventual development pass must be followed by a separately frozen confirmation using fresh,
output-blind requests. This document, an oracle-probe pass, or successful sampler plumbing cannot
complete the realism goal or authorize a model-promotion claim.
