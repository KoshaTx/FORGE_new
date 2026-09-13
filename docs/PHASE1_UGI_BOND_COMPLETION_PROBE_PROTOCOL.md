# Bond-completion uncertainty: finite one-decision replay

Date: 2026-09-08. Defined before observing bond-choice masks or executing this diagnostic.

The placement probe found no amine greedy-placement error and negligible aldehyde topology-law
impact. It summed only deterministic completions induced by atom placements. Test whether omitted
bond decisions have practically relevant effects under the same frozen predictions and gates.
The active realism objective and every original preservation requirement remain unchanged.

## Fixed contexts and budgets

Use exactly the 286 original topology contexts for requests 0–7, both roles, in the saved order of
`ugi_completion_placement_probe_v1/attempt_0002`. Authenticate that result, its independently
audited artifacts, source bytes, original parent capture, and TRAIN-only layout inputs. Reconstruct
the original records and require full saved-batch layout equality. Restore saved candidate pointers,
original shared-B predictions and original greedy atom placements. Do not enumerate a different
topology universe, add clean target labels, fit a model or select contexts using outcomes.

On CPU with two Torch threads, first capture one original strict completion for every topology.
Require exact identity with the saved complete original terminal arrays, reason, molecule and score.
Count every original allowed bond mask, including singleton and forced masks. Enumerate every
nonselected allowed state at each tested-role variable bond along the original decision path.
Capture and count other-role decisions too, but do not intervene on them.

The cap is **1,024 original strict captures plus 16,384 replay strict calls**. Replay work is the
number of nonselected allowed bond states plus one original-choice identity control per topology.
If this exceeds the cap, preserve the complete census and stop without shrinking the panel,
truncating alternatives, extending the budget or choosing a different seed. If either role has no
alternative bond decision anywhere in these contexts, the intended two-role mechanism cannot
pass: preserve the census and stop before replay. This is a result about the fixed-atom sequential
decoder path, not all legal chemistry or alternative atom placements.

## One decision deviation and identity controls

Capture direct `_argmax_allowed` calls from the original strict decoder's variable-bond loop.
Do not intercept parent-pointer or atom selection, or nested amine-selector calls. Record row,
parent/closure field, slot, endpoints, selected state, actual allowed class IDs, logits, forced
constraints, capacities, used valence and a prefix digest. Bond class IDs retain the decoder's
vocabulary meaning; they are not reinterpreted as numeric bond orders.

For each alternative, clone function globals locally and substitute only that originally allowed
decision. Authenticate the complete original prefix and the nominated call's mask/logits before
the substitution. Let the unmodified decoder recompute every later mask and greedy decision.
Later masks may legitimately change as shared valence is consumed; a single decision deviation
can change several final bonds. Record the suffix and preserve every final failure.

Run one additional control per topology: choose the first tested-role bond and return its original
selected state, including if its mask is singleton. If no tested bond exists, rerun without an
override. Require complete terminal/reason identity. Each helper call performs exactly one strict
decode; no hidden reference rerun, randomness, model construction or neural forward is permitted.

Every successful completion must retain all atom states, pointers, fixed states and other-role
chemistry. A changed atom, pointer or fixed state is an implementation invariant failure. A changed
other-role bond is an explicit isolation failure and makes the role law unavailable. Use original
final molecular validation and exact registry-qualified constitutional Ugi reverse/forward checks.
Retain failures and all attempted branches; never repair, replace, resample or renormalize survivors.

## Scores, practical screen and interpretation

Score completed atoms and bonds over the original identical full-role variable-coordinate masks
using float64 full-vocabulary log probabilities of the saved float32 predictions. Keeping atoms
fixed does not remove their original score contribution. Sum over the finite union of the original
completion and every single-decision-deviation completion. Deduplicate identical serialized final
states within each topology, including identity controls, before log-sum-exp. Preserve all branch
memberships and report constitutional grouping separately without redefining its weighting law.

Report original score, best-minus-original regret, finite mass minus best, and mass minus original;
report original, best and finite-mass topology laws using the same saved A scores and coefficient
one. Preserve all 16 request/role denominators. Any failed original or replay branch makes that
whole request/role law unavailable. Report no-alternative and singleton counts, neighborhood size,
deduplication, changed suffix choices, exact-L1 failures and actual work separately.

Use **1e-10** only to label numerical equality. A practically relevant mechanism requires at least
one complete failure-free request with competing topologies in **each role**, where either the
best-completion law or finite-mass law differs from the original greedy-completion law by total
variation **at least 0.01**. Qualifying requests may differ by role. This prespecified one-percentage-
point probability-mass screen is a necessary reason for further investigation, not a statistical
test, an estimate of molecular change under a particular RNG coupling, or a realism gate.

This finite neighborhood does not cover arbitrary sequences of alternative bond decisions, all
legal assignments, or a graph partition function. Extra mass may reflect assignment count or
serialization multiplicity rather than improved chemistry. Passing this screen neither reverses
the rejected shared-chemistry arm nor establishes diversity, novelty, realism or generalization.
No sampler trial follows automatically; any successor must retain the full original improvement
protocol, visual calibration, fresh confirmation and global validation requirements.

Archive source/config/input hashes, complete census, original/branch receipts, terminal arrays,
scores, failures, work counts and elapsed time in fresh directories. No production sampler/model
edit, topology draw, new flow trajectory, training, remote compute, routing, biological optimization,
heldout structure interpretation, prospective selection or gate change occurs in this diagnostic.
