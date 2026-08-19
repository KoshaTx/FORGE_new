# Phase 1 V5: degree-budgeted sparse morphology flow

Status: the full-heavy-atom V5 tree is a completed diagnostic prototype. A support-skeleton audit is
now the gate before Ugi-first production training; V5 is not approved for full-corpus training.

The diagnostic V5 model treated every heavy atom as a morphology node. Terminal carbonyl, hydroxyl
and related decorations can therefore appear as tree leaves, and carbonyl centers can appear to be
hydrophobic junctions. The production representation compares the full-heavy graph with a literal
carbon-only negative control and a functional support skeleton. The functional candidate retains all
carbon, nitrogen, phosphorus, ring, charged, connector and protected Ugi-core atoms while moving only
terminal neutral non-carbon decorations into the conditional chemistry-realization target. Exact
atom-level reconstruction remains mandatory. See `docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md`.

## Why the representation is changing

The dense graph objective asked the model to classify mostly absent atom pairs. The independent
parent-pointer model removed the dense edge tensor but still produced poor lipid morphology. The
exact BFS offspring probe restored connected tree structure, yet its local child predictions can
still create rare but severe overbranching.

The R0 training audit supports a stronger sparse formulation. Across 10,591 eligible R0 training
lipids:

- 73.39% of all heavy atoms have molecular degree two;
- 87.45% have degree at most two;
- 84.59% of tail-region atoms have degree two;
- 95.93% of tail-region atoms have degree at most two;
- molecular cycle rank has median one and 90th percentile two.

The production topology model should therefore generate a globally feasible morphology program and
only actual molecular edges. It should not restore a bond-versus-no-bond decision for every atom
pair. The representation audit selected a deterministic breadth-first spanning tree serialized in
depth-first preorder. This separates short-path tree selection from sequence ordering.

The stored state and the number of generated structural decisions scale as `O(N + K)`. This is not
an end-to-end linear-runtime claim. Polar-root preprocessing and neural attention may remain
quadratic, and the V5 sampler must not reuse the dense `tree_pair_ring_sizes` helper. Runtime and
memory through the declared 282-atom limit remain bounded-gate measurements.

## Minimal morphology program

The first gate uses the smallest nonredundant global state:

```text
M = (n_head, n_interface, n_tail, J_head_interface, J_tail, K)
```

where:

- the three `n` values are region counts and derive total heavy-atom count `N`;
- `J_head_interface` and `J_tail` are tree-degree-excess budgets by region;
- `K` is molecular cycle rank and therefore the number of sparse closure edges.

Closure incidence, regional closure counts and graph-theoretic leaf counts are derived after node
states and closure edges are generated. They are not predicted as additional independent variables.
A generic hydrophobic-arm count is not a hard broad-corpus variable until a chemically defensible arm
annotation is available. In the Ugi-conditioned adapter, the two tail-bearing component origins
provide an exact family-specific condition.

For each active atom, the morphology stage generates:

```text
(region, tree_children, coarse_valence_state[, component_origin])
```

The exact atom state may replace the coarse valence state if the bounded gate shows that the existing
14-state atom vocabulary is stable enough to generate before connectivity.

## Exact constraints

Let `c_i` be the number of rooted-tree children for atom `i` in the selected preorder word.

```text
sum_i c_i = N - 1
d_tree_i = 1[i is not root] + c_i
```

The preorder offspring sequence must also satisfy the Lukasiewicz queue language:

```text
Q_j = 1 + sum_{i=0}^j (c_i - 1)
Q_j > 0 for j < N - 1
Q_(N-1) = 0
```

Under the frozen canonical preorder serialization, a valid offspring sequence reconstructs exactly
one parent tree. No parent-pointer logits are required.

Tree leaves and branch budgets apply to `d_tree`. Closure endpoints alter final molecular degree but
do not create a new hydrophobic arm, so they are controlled separately by closure count and valence
capacity:

```text
J_region = sum over atoms in region of max(0, d_tree_i - 2)
```

This makes degree-three and degree-four tail junctions possible, but each consumes an explicit global
budget. Ordinary tail atoms are path-like by default.

## Sparse closures

After decoding the exact tree, generate `K` unordered closure-edge pairs sequentially. For each edge,
score a first endpoint and then a second endpoint conditioned on the first. Apply feasibility masks:

- no self-loop;
- no duplicate edge;
- no existing tree edge;
- sufficient remaining coarse degree and bond-order capacity;
- exact completion remains possible for all remaining closure edges.

Direct pairs avoid requiring a separately generated closure-degree sequence to be realizable on the
complement of the tree. The cycle-rank head must first be masked so the generated tree and coarse
capacity state admit at least `K` closure edges. At every prefix, the sampler must renormalize only
over lexicographically ordered candidate edges for which an exact residual completion exists. An
infeasible global program cannot be rescued by backtracking, and a search cutoff cannot support an
exact guarantee. Observed `K` has median one, 90th percentile two and declared maximum twelve, but
that bounds search depth rather than pairwise branching. Exact-search runtime must therefore pass a
worst-case `N = 282, K = 12` benchmark before practical feasibility is claimed. Closure incidence is
derived and may be used as an auxiliary target. Regional closure localization is learned as a bias
with nonzero support for rare valid cases, not as a finite ring vocabulary. Closure stubs remain an
ablation only.

This feasibility contract applies to coarse topology and capacity. It does not prove that a later
exact atom, bond-order and aromatic assignment exists; that is a separate chemistry gate.

## Chemistry coupling

Topology cannot be generated without a valence envelope. Node states must expose enough chemistry
before tree decoding and closure pairing to rule out impossible heavy degrees. Bond orders are then
generated only for the `N - 1 + K` present edges while tracking remaining endpoint valence.

The morphology gate must compare:

1. exact atom states generated before topology;
2. coarse valence states followed by exact atom refinement.

Whichever route is retained must preserve formal charge, aromatic eligibility, explicit aromatic
hydrogen state, and the full declared atom vocabulary. It may not gain validity by truncating R0.

## Region and Ugi semantics

Broad-corpus head, interface, and tail annotations are generic structural context, not chemical
component identities. They may guide regional degree laws but cannot establish a hard semantic tail
count.

The Ugi L1 adapter adds exact source-adjudicated atom origins:

```text
amine-derived, aldehyde-derived, isocyanide-derived, shared Ugi core
```

The two tail-bearing origins are therefore conditioned directly in Ugi generation without selecting
stored head or tail IDs. All component atoms and bonds remain generated states.

## Canonical representation gate

The completed representation gate established:

1. constitutional canonicalization removes atom-order dependence before tree construction;
2. all 15,229 accepted R0 records reconstruct exactly;
3. all three audited tree codes remain exact under 15,229 direct random atom permutations;
4. the selected tree is the deterministic polar-root breadth-first spanning tree;
5. the selected serialization is depth-first preorder over that fixed tree.

Traversal selection uses `R0_cal`, never `R0_heldout`. Exact reconstruction and permutation
invariance may be audited over all records because they are deterministic representation properties.
Comparative learnability, region continuity and closure-locality measurements are calibration-fold
selection criteria. Ugi-origin continuity is measured only on exact semantic products assigned to
`R0_train`. The selected hybrid retains the BFS discovery tree while placing each subtree in preorder,
which preserves zero root-distance distortion while keeping Ugi component origins substantially more
contiguous than BFS serialization. A pure DFS discovery tree can lengthen tree paths relative to true
molecular shortest paths. The exact calibration-fold values are pinned in
`results/phase1/v5_tree_traversal_comparison.json`.

On `R0_cal`, the hybrid has zero mean root-distance stretch, while pure DFS has mean stretch 0.354
atoms and mean maximum stretch 3.377 atoms. The hybrid also reduces mean closure tree distance from
5.077 to 4.859 and normalized closure span from 0.226 to 0.137 relative to pure DFS. Its fixed-bigram
cross-entropy is 1.077 nats per token, essentially tied with pure DFS at 1.076 and better than BFS at
1.116. On the 1,100 exact `R0_train` Ugi products, origin transition is 0.090 for the hybrid, 0.088
for pure DFS and 0.738 for BFS. These measurements favor the hybrid because it preserves true
head-to-atom distances while giving up almost none of preorder's sequence learnability or component
continuity.

The hybrid does not win every serialization metric. Breadth-first ordering can create longer
contiguous runs of the generic tail label by interleaving atoms at the same depth across different
arms. That value is reported rather than hidden, but it is not interpreted as better within-arm
sequence coherence. The selection is a declared tradeoff among calibration-fold learnability,
subtree and component-origin continuity, and closure locality.

Every bridge, including every acyclic tail-chain bond, is present in every spanning tree, so no
hand-weighted spanning-tree objective is justified by the current evidence.

## Bounded morphology gate

The first V5 model generates no atom identities or bond orders. It must reproduce, on held records and
generated endpoints:

- node-count and region-count distributions;
- tree-degree and final molecular-degree distributions by region;
- tree-junction budgets;
- tree leaves and maximum root distance;
- hydrophobic-arm diagnostics once qualified;
- tail path likeness and branch incidence;
- cycle rank and closure localization;
- random-sample and worst-case visual morphology;
- topology diversity and novelty;
- throughput and memory through 282 heavy atoms.

The closure implementation must additionally agree with brute-force feasibility on small graphs,
avoid greedy traps through exact prefix completion checks, reject globally infeasible programs before
sampling and pass the declared worst-case runtime and memory benchmark.

Means alone cannot pass. Report per-molecule quantiles and the fraction of samples exceeding frozen
R0 tail-branch, degree-four, arm-count, and fragmentation limits.

## Deferred alternative

Contracting maximal degree-two paths into variable-length strands is an exact graph-theoretic
compression and the R0 audit suggests substantial compression. It is not the first production model.
It adds segment ordering and cyclic special cases before the simpler degree-budgeted tree has been
tested. Revisit it only if V5 still fails on long-chain morphology or measured runtime.

## Scope

V5 remains a whole-molecule atom-level discrete flow. It has no finite head, tail, linker, ring, or
building-block vocabulary. Biological guidance, L2 route generation, and synthesis-value guidance
remain outside the authorized Phase 1 morphology gate.
