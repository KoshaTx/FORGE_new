# Complete-candidate graph scoring diagnostic

Date: 2026-09-08. Status: protocol before fitting; no realism improvement established.

## Question and comparison

Can nonadditive interactions within a complete candidate head or aldehyde topology improve the
frozen checkpoint's probability of a TRAIN target skeleton within the actual baseline decoder
candidate set? The scientific objective remains the unchanged
`PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md`: better lipid realism while preserving structural
validity, exact Ugi reconstruction, diversity and novelty. This diagnostic does not satisfy it.

The previous structured-topology residual already tested additive offspring and closure scores
with a forest-normalized training loss. Its negative result is preserved. This test instead uses
the actual joint amine candidate set after semantic/local-support filters, and the actual
constructive ester candidate set. It retains candidate order and serialization multiplicity.
A complete candidate's residual depends on adjacency through two message-passing rounds.
The degree-only control retains each candidate's node degrees but suppresses neighbor messages;
it tests local degree preferences without those interactions. Both start with zero scalar output,
identical parameters and identical fitting draws. Inactive neighbor weights in the control mean
this is not an equal-effective-capacity claim.

## Frozen support and labels

Use the original step-9143 checkpoint, registry, morphology policy and baseline four-coordinate
amine semantic target. No failed treatment semantics, exact directional tail targets, new atom
classes, candidate pruning, repairs, retries, or altered size/branch/cycle restrictions are allowed.
Raw neural logits remain unchanged. Candidate enumeration calls the existing baseline helpers;
cached ordering and scores must reproduce them. Closure endpoint scores retain the original
floating-point addition, direct/reverse maximum, and tie convention.

Correct labels compare candidate and target unweighted heavy-atom skeletons with fixed core,
role, and attachment colors. Sum over every candidate in the target graph class; serialization
positions are not chemical identity. Unknown target exterior atom types or bond orders cannot
define a topology label. Retain empty candidate sets, absent targets, singleton graph classes,
all-positive sets, and excluded source mass in the coverage ledger. A topology positive is not
evidence that all subsequent chemistry choices can realize the complete molecule.

## Data and features

Start from the authenticated census of all 66,464 Ugi TRAIN products. Its existing semantic and
program applicability subset is retained, with every excluded row and its original source mass
reported. The subset is a training diagnostic scope, not a changed production support gate.
Mask assignment rows to TRAIN before accessing any structure. Use constitutional canonical
component identities. No sealed component, product, or broad-corpus structures are interpreted.

For each of amine and aldehyde independently, hash
`forge.ugi_candidate_graph_component_partition.v1:2026090824:{role}:{component}` with SHA-256;
integer remainder zero modulo five withholds that component from residual fitting. Fit requires
both components to be nonwithheld. Within that population the analogous product hash with domain
`forge.ugi_candidate_graph_product_partition.v1:2026090824:` assigns bucket zero to repeated-component
evaluation. Fixed source-weighted draws: 1,024 fit, 256 amine-disjoint, 256 aldehyde-disjoint,
256 repeated-component. The two disjoint evaluation populations may overlap; disclose that
dependence. Do not change the split or draw seed after inspecting predictions.

Flow times are 0.2, 0.5, 0.8, 0.95. Extract contextual features once from the original backbone
using native noisy topology, atom, and bond states. Do not substitute clean target parents or
closure endpoints into the feature input. The original coarse program/roles/core masks remain
available. Clean target topology is used only for labels and enumeration support that the baseline
already conditions on. Record clean/noisy/prediction hashes and noise seeds. CPU, float32,
two threads, deterministic algorithms; feature batches of 64.

## Model and fitting budget

Shared two-role scorer: frozen contextual node features, time, role indicator, fixed node colors,
and candidate degree feed a 32-wide ReLU projection. Two rounds combine a self projection with
a mean-neighbor projection and ReLU. Masked mean pooling feeds a zero-initialized scalar linear
output. The control suppresses neighbor messages while retaining the actual candidate degrees.
Neither model uses target graph labels as inputs or modifies the backbone.

For each role case, the loss is logsumexp of all candidate scores minus logsumexp of target-class
scores. The score is the unchanged baseline score plus the learned residual. No-positive and
empty cases contribute zero with explicit unlearnable counts; all-positive cases contribute exact
zero. Average over both roles and all fixed sampled product/time exposures, retaining original
sampling mass rather than renormalizing to easier reachable examples. Also report conditional
metrics on reachable informative cases. No target-dependent candidate deletion is permitted.

Exactly 64 AdamW updates per arm, batch size 64 product/time cases, learning rate 0.001,
weight decay 0.0001, gradient norm cap 1.0. All 4,096 fit exposures occur once in a fixed shuffled
order. Draw seed 2026090825, noise seed 2026090826 plus forward-call index, order seed 2026090827,
initialization seed 2026090828. Save initialization, final states, exposure order, losses and
feature artifacts. Evaluate final checkpoints only; no hyperparameter or checkpoint selection.
Process candidates in bounded chunks and accumulate each case's gradient before advancing;
never truncate candidate support to fit memory.
Use 128-candidate chunks. A detached score pass computes the exact class-loss score derivative;
a second pass recomputes one chunk at a time and accumulates its weighted gradient. Model weights
remain fixed within each 64-exposure optimizer batch. This bounds live activations while retaining
all candidate probabilities and the original loss denominator.

An unplanned calibration-layout inspection occurred during candidate API development before this
experiment. `results/phase1/ugi_candidate_graph_access_incident_v1/result.json` records the exact
scope. That record is excluded from features, labels, fixtures and evaluation. The experiment's
TRAIN-only data path does not establish zero non-TRAIN record access throughout the session.

## Decisions

Before fitting, require exact baseline candidate ordering/score equivalence, graph-label checks,
zero-residual equivalence, and at least one reachable target with competing graph classes in each
role's fit and corresponding component-disjoint evaluation population. Failure is a recorded
applicability finding, not permission to change support or select a more favorable partition.

Report each role, flow time, and evaluation population separately. The primary comparison is
mean target graph-class probability and graph-class NLL on reachable cases with competing graph
classes. A mechanism pass requires higher probability and lower NLL than both original baseline
and degree-only control in **each role's component-disjoint population**, with graph-class argmax
accuracy no lower than baseline in either that population or the repeated-component population.
Retain all-case denominators and absent-target counts alongside these conditional measures.
These are dependent TRAIN reconstruction measurements, not independent held-out model accuracy.

A pass permits a separately pinned sampler integration and matched generation test, including
zero-delta output/probability/RNG equivalence. It does not promote a model. All existing descriptor,
graph, novelty, exact-L1, diversity, calibrated visual-review and fresh-confirmation requirements
remain necessary. Fixed role sizes mean this method can change head shape and ester/spacer
placement, not coarse head:tail size or unsupported tail branching. Preserve failed results.
