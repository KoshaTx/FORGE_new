# Phase 1 HeLa potency-only diagnostic contract

Status: frozen policy, not execution authorization.

This additive contract creates one narrow exception to the earlier blanket
biological-guidance abstention: a standalone, matched-compute HeLa potency
diagnostic may be reviewed for execution under the rules below. It does not
modify, promote or supply evidence for the matched synthesis-guidance
preregistration. Synthesis planning, Graph2Edits/proposal guidance, candidate
selection, prospective locking and in-vivo claims remain outside scope.

## Evaluator precedence

Every scheduled terminal attempt is retained. The evaluator applies the
following order and fails closed:

1. Require a locked, valid, exact-L1 Ugi terminal.
2. Assign exact measured component combinations a neutral value.
3. Require all four frozen version-3 views—product, amine, aldehyde and
   isocyanide—to be interpolative.
4. Permit only exact-new amine-only candidates or exact-new
   aldehyde-plus-isocyanide-pair candidates. All other novelty patterns are
   neutral.
5. Require the separately qualified oracle runtime to agree with generator
   runtime identity and exact forward reconstruction.
6. Compute `LCB90 = ensemble mean - max eligible-fold q90` for the exact
   pattern, then `u = max(0, 2 * Fn(LCB90) - 1)`, where `Fn` is the
   right-continuous empirical CDF of calibration-only LCB90 values for that
   pattern.

Thus every invalid, unsupported, boundary, extrapolative, measured or
at/below-calibration-median candidate has utility exactly zero. Applicability
does not itself earn a bonus, and raw oracle means are never optimized.

## Prespecified experiment

All eight frozen disjoint 16-program × 4-particle schedules are retained:
20260821–20260823 are calibration seeds and 20260824–20260828 are evaluation
seeds. There is no seed search, retry or exclusion. Every seed must pass a
matched λ=0 identity replay before any nonzero run begins; seed 20260821 must
also reproduce the historical selected-v3 hard reference. The nonzero
diagnostic uses guidance strength 0.25 and checkpoint tempering at steps 2, 4
and 6 with β = 0.25, 0.50 and 0.75. Across all eight seeds this yields 512
terminal particles and 1,536 checkpoint completions per arm. Product schedule
and terminal-assessment attempts are matched. Synthesis and proposal calls are
both exactly zero.

## Primary endpoint

For each of the five evaluation seeds, compute the mean terminal utility over
all 64 guided particles minus the mean over all 64 post-hoc particles; invalid
and abstained terminals contribute zero. The primary endpoint is the
unweighted mean of those five paired differences.

A positive diagnostic signal requires all of the following:

- a strictly positive mean paired evaluation-seed delta;
- at least three of five evaluation-seed deltas nonnegative;
- at least one replay-verified nonuniform checkpoint guidance event; and
- no frozen safeguard failure.

Conditional LCB90 values, eligible fractions, all calibration-seed outcomes
and checkpoint diagnostics are secondary. No result from this diagnostic can
select or lock a prospective lipid without a later, separately frozen review.

The executable source of truth is
`configs/bio/phase1_ugi_hela_potency_diagnostic_authorization_v1.json`.
