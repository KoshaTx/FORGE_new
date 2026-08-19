# Phase 1 matched synthesis-guidance runner

## Status

The runner is implemented and qualified only as a device-agnostic development
orchestration layer. It does not authorize production nonzero guidance,
biological guidance, prospective candidate locking, sealed-holdout access, or a
cloud execution. The prepared plan remains fail-closed until the grouped
lambda-zero execution, production generator/cache-adapter qualification, and a
fresh current-L3 execution review are complete.

## Frozen matched design

Each seed contains 16 distinct frozen morphology programs and four stochastic
particles per program, for 64 particles. Ancestry is computed independently
inside each four-particle program group; cross-program ancestry is rejected.
The grouped schedule qualification freezes eight seeds, 128 distinct programs,
and 512 particle streams. Lane initialization must attest that it consumed the
exact 64-seed manifest. Productive step-8 completion consumes each stored
particle seed deterministically, while checkpoint rollout seeds are keyed
separately. An adapter that ignores the supplied particle streams fails closed.

Synthesis checkpoints occur after steps 2, 4, and 6. Each checkpoint performs
one terminal completion per particle and uses the binary exact-dossier route
completion value. A planner-censored assessment remains null and contributes a
neutral incremental weight; it is never imputed as route failure. Genealogical
potentials follow the selected ancestor into the next checkpoint.

The matched ceiling per seed is 1,280 product transitions, 256 terminal
completions, 768 logical planner calls, 3,328 logical verifier calls, and 32
selected development endpoints. Actual logical and physical calls, cache hits,
generator transitions, wall time, and device time are recorded separately.

## Route-blind productive admission

All productive terminal bytes and generation traces are sealed before any
post-hoc checkpoint-shadow assessment. Invalid or nonexact-L1 terminals are
excluded before identity. Canonical constitutional identity is derived only
from immutable terminal bytes. Productive terminals are then deduplicated
before route assessment, and one representative per identity is chosen with a
keyed route-blind priority. Only that representative is assessed. Duplicates
inherit the representative's result and contribute zero route compute; a
duplicate may never be selected because it happened to receive a better route
outcome.

Endpoint selection ranks exact complete dossiers above observed incomplete
routes above censored assessments, followed by the frozen keyed tie-break. The
post-hoc endpoint subset is sealed before its 192 matched checkpoint-shadow
assessments begin. Those shadows cannot enter the productive pool or influence
generation or selection.

## Cache and receipt requirements

Guided and post-hoc arms share one immutable base snapshot but use distinct,
initially identical writable overlays. Every assessment context carries the
base snapshot, planner context, cache preflight, and authoritative clone ID.
The runner requires final before/after cache audit receipts and rejects base
mutation, overlay aliasing, or mutation of one arm while binding the other.

Every terminal carries an admission record. Every assessed representative has
an assessment receipt; every exact-dossier success also has a route-dossier
receipt. Checkpoint batches retain terminal, trace, value-policy, evaluation,
assessment, dossier, censor, and compute records. The stable run serializer
preserves censoring as JSON null and explicitly leaves synthesis-success
probability undefined.

## Production boundary

The selected generator and terminal evaluator are hash-pinned inputs, but the
production generator/cache adapter remains intentionally unqualified in this
plan. The development runner can exercise typed fakes and adapter tests; it
must not be described as a completed production nonzero-guidance execution.
Only a later, independently reviewed artifact may change that authorization.

## Runtime and historical provenance

The current selected lane is qualified by
`ugi_restartable_sampler_equivalence_v2` in the repository `.venv` runtime
(Python 3.14.2, NumPy 2.5.1, RDKit 2026.03.4, and Torch 2.13.0). The v2 receipt
matches the current selected-restartable-generator and synthesis-guidance source
hashes and reproduces the one-, four-, and twelve-sample partitions bitwise.
System Python has different dependency versions and correctly fails the runtime
identity check; it is not the execution environment.

The historical v1 equivalence receipt and the v1 matched-budget reproduction
pin remain stale after later source changes. They are preserved rather than
repinned. This historical failure does not invalidate current v2, but any reuse
of the old matched-budget artifact requires a versioned successor rather than a
mutation of the frozen v1 record. The new runner and terminal evaluator are not
hash-owned inputs of that historical receipt and did not cause its drift.
