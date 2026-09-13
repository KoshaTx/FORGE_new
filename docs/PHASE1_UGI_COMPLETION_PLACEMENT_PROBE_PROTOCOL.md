# Alternative supported chemical placements: bounded diagnostic

Date: 2026-09-08. Defined before executing this diagnostic.

The shared-chemistry rescoring arm remains rejected because it lost distinct products and
pairwise diversity. Its matched-randomness control established a conditional score effect,
alongside lower aldehyde novelty. This diagnostic asks whether scoring one greedy chemical
completion misses better full-completion scores or appreciable mass over other placements
already enumerated by the original decoder. It does not establish improved lipid realism.

## Fixed panel and work limits

Use requests **0 through 7**, in their saved order, from the authenticated native baseline
capture of `ugi_shared_chemistry_rescoring_v1/attempt_0001`. Test both amine and aldehyde roles
and every original ordered target-free topology candidate. Keep the original B chemistry
predictions, native chemistry input, other-role topology, four-coordinate amine condition,
TRAIN-only layout prior, registry, support bounds, vocabulary and strict decoder unchanged.
This convenience panel is a mechanism probe, not a representative sample or confirmation set.

Authenticate all source and input pins and require every restored layout tensor to equal the
saved layout before interpretation. No checkpoint loading, neural model construction, forward
call, fitting, topology draw, new flow trajectory or remote execution is permitted. No measured
reference distances, novelty labels or clean chemical targets select placements or topologies.

Count all topology candidates before chemistry work; stop if the panel exceeds **1,024**.
Preflight makes one unchanged strict completion per topology to observe its original selector
candidate pool. These are real decoder calls, not free enumeration. Count every placement in
every observed pool, including the originally selected placement. Stop before counterfactual
replay if this total exceeds **16,384**; preserve the preflight and do not shrink the panel.
The maximum is 1,024 preflight strict calls plus 16,384 replay strict calls, counted separately.
No subsequent panel, role, coefficient, seed, placement truncation or budget expansion is part
of this protocol. Local execution uses CPU, two Torch threads, deterministic arithmetic and
float64 full-vocabulary log-probability calculations on the saved float32 logits.

## Observation and controlled replay

Observe the unmodified amine and ester selector functions with passive instrumentation and
save their ordered candidate pools and selected indices. Preserve selector-stage failures.
For each observed placement, rerun the unchanged strict decoder while replacing only the
tested selector's return value with that existing candidate. Recompute and authenticate the
same original pool on replay. Restore temporary bindings and instrumentation even on error.
The other role continues through its original selector and must retain the same choice.

The originally selected placement must reproduce the complete terminal tensors, reason,
canonical constitutional product and full-role confidence exactly. Every successful replay
must preserve requested pointers, fixed states and all chemistry outside the tested role.
Check final molecular validity and exact constitutional Ugi reverse/forward reconstruction;
retain unsuccessful placements, reasons and work in the attempt ledger. A candidate admitted
by a partial selector is not assumed to be a legal completed molecule. Do not repair, resample
or discard failures to manufacture an eligible topology law.

Score each completed placement using all variable atom and bond coordinates in the tested
role, with identical masks for every competing topology and placement. Exclude fixed and
padded coordinates. Partial motif scores and atom-only scores are recorded for attribution
but cannot substitute for these full-completion log probabilities.

Deduplicate identical serialized terminal chemistry within each fixed topology before
computing log-sum-exp. Require duplicate states to have identical scores. Retain serialization
multiplicity in the ledger, and report constitutional product grouping separately; graph
identity does not define a different weighting law. The resulting sum covers only unique
successful deterministic completions induced by these enumerated placements. It is **not**
the partition function over all legal bond assignments, all atom states, or molecular graphs.

## Diagnostic outputs and decision

For each topology report pool size, failures, unique completed states, constitutional products,
the original full-completion score, best completed score, and log-sum-exp over completed states.
Separate the best-minus-original improvement from log-sum-exp-minus-best residual mass.
For each request and role report whether either correction varies across competing topologies;
constant corrections cannot change a normalized topology law. Report ranges and descriptive
law differences with coefficient one, without drawing products or selecting a successor.

Use a fixed **1e-10** absolute tolerance solely to label numerical equality. A mechanistic
signal requires at least one of the eight requests with two or more competing topologies and a
complete failure-free placement pool with a nonconstant correction in **each** tested role
(the qualifying request may differ by role). This is a necessary reason for further
investigation, not a promotion gate or proof that the corrected score predicts realism.
If either role has no such signal,
stop this mechanism under this protocol. If failures prevent a complete law, label the law
unavailable and retain every topology; do not renormalize over survivors. Any future experiment
requires a separately specified mechanism and retains every original improvement requirement.

Archive source bytes, config, used input hashes, preflight, complete placement ledger, terminal
arrays, actual call counts, elapsed time and failure records in a fresh results directory.
Keep prior negative results and repository-wide test failures intact. Scientific visual review
and fresh confirmation remain pending; no generation arm, model promotion, gate weakening,
heldout structure interpretation, routing, biological optimization or candidate selection occurs.
