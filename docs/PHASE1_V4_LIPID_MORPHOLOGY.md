# Phase 1 V4 lipid morphology flow

## Decision

V3 is not accepted as the broad whole-lipid prior. Its optimization and
validity gates passed, but its endpoints did not reproduce the global
organization of R0 lipids. V4 must model lipid-scale morphology explicitly
while retaining complete atom-level graph generation.

## First-principles factorization

Let \(x=(A,B)\) be the complete molecular graph, with atom states \(A\) and
bond states \(B\). Let \(M\) be a generic rooted morphology:

- the heavy-atom count;
- a canonical breadth-first offspring sequence that exactly defines one
  connected rooted spanning tree;
- residual closure endpoints;
- structural region per atom;
- root distance derived from the tree;
- terminal, path and junction status derived from the tree;
- local cycle context derived from the closures.

FORGE V4 factorizes:

\[
p(x,M)=p(N)\,p_{\theta}(T\mid N)\,p_{\rho}(R\mid T)\,
p_{\kappa}(C\mid T,R)\,p_{\psi}(A,B\mid T,R,C).
\]

Here \(T\) is the rooted tree, \(R\) is structural region and \(C\) is the
residual closure state. Every learned stage is a discrete flow. The morphology
stages generate the global organization of one complete connected graph. The
chemistry flow assigns every atom identity and bond order on that graph. The
system does not select or assemble named head, linker or tail fragments.

## Why this is still whole-molecule generation

The molecule is represented by one atom index set and one connected graph from
the start. Morphology variables describe relationships among those atoms; they
are not component IDs. A tail remains a generated sequence of atoms and bonds.
A ring remains a generated set of closure edges and atom states. A novel head
remains a generated subgraph.

The morphology level can support:

- variable head size and topology;
- zero, one or multiple head rings;
- variable interface placement;
- variable number and length of hydrophobic arms;
- controlled branching at learned positions;
- unsaturation and degradable chemistry;
- structures outside every frozen component cross-product.

## R0 evidence that defines the state

The frozen R0_train fold contains 10,591 eligible structures.

- Tail-region atoms comprise 66.05% of atoms.
- Tail-region branching is 4.07%.
- 91.60% of tail-region connected components are path-like.
- Head atoms have a 29.43% ring-atom fraction.
- Tail atoms have a 3.57% ring-atom fraction.
- Nitrogen has median graph distance 3 from the selected polar root.
- Oxygen has median graph distance 8.
- Carbonyl and two-single-bond bridge environments account for 65,246 of
  70,973 oxygen environments.
- Terminal single-bond oxygen accounts for 4,895 oxygen environments.

These values are evaluation targets, not hard generation quotas.

## Neural architecture

### Morphology flow

The canonical breadth-first tree is represented by an offspring sequence
\(c_0,\ldots,c_{N-1}\), where \(c_i\) is the number of tree children assigned
to atom \(i\). A valid sequence satisfies:

\[
\sum_i c_i=N-1,\qquad
1+\sum_{j=0}^{i}(c_j-1)>0\ \text{for }i<N-1,
\]

and the final queue balance is zero. Under the frozen canonical breadth-first
ordering, this sequence maps one-to-one to the parent-pointer tree. It removes
arbitrary parent collisions while retaining every rooted tree in the declared
offspring support.

The topology denoiser receives:

- noisy per-atom offspring states;
- canonical atom position;
- noisy breadth-first queue balance;
- time and node-count context.

It uses:

- global self-attention over every atom slot for long hydrophobic arms;
- position and queue-balance embeddings;
- a structured terminal sampler that draws exactly from the predicted
  offspring logits conditioned on the valid breadth-first tree language.

This terminal step is not a repair. Dynamic programming normalizes the model
distribution over valid tree sequences and samples from that conditional
distribution without changing atom count or inserting fragments.

After the tree settles, a conditional region flow assigns head, interface and
tail state. It uses exact tree messages, global attention, root-distance and
degree embeddings, and a depth-conditioned source distribution. The polar root
is head by definition. Region transition evidence may be used as a declared
conditional prior, with its strength selected on calibration rather than the
held-out fold.

Residual closure endpoints are generated only after the tree and region states
settle. This keeps ring formation separate from tree branching and makes cycle
size and region localization directly auditable.

### Chemistry flow

The chemistry denoiser receives the settled morphology plus noisy:

- atom states;
- spanning-tree bond orders;
- closure bond orders.

It uses local edge-aware processing and global attention to predict complete
atom and bond states. Functional-group coherence is learned from the joint
local environment, not enforced through a named functional-group vocabulary.

### Terminal sampling

V4 settles tree topology, then structural regions, then residual closures,
before refining atom and bond states. It must not independently redraw every
endpoint head from one stale hidden state. Valence and aromatic consistency are
maintained during chemistry refinement and audited as sampler interventions.

## Corpus policy

The broad prior remains anchored to real R0 structures. R1 remains a
low-weight auxiliary reaction-accessible product stream and cannot define the
prior.

Before freezing sampling weights, recover publication or experiment identities
inside the large leakage-connected source supergroup. Do not interpret the
supergroup hash as one literal publication and do not apply inverse-group
weighting blindly.

Compare:

1. row-balanced R0;
2. publication-balanced R0;
3. structural-cluster-balanced R0;
4. a frozen mixture of publication and structural-cluster balance.

Select on R0 calibration morphology and distribution coverage. Do not use the
held-out fold to select weights.

## Acceptance gates

Every early checkpoint reports:

- validity and connectedness;
- uniqueness and exact-train novelty;
- regional atom fractions;
- regional branching;
- tail path-like component fraction;
- tail component length and diameter;
- root-distance distributions for C, N and O;
- oxygen environment distribution;
- ring localization by region;
- unsaturation localization;
- region connectivity and transitions;
- source and structural-cluster coverage;
- terminal constraint interventions.

V4 does not advance to full training if it reaches validity through post-hoc
repair while retaining scattered oxygen pendants, incoherent region islands or
V3-like excess tail branching.

## Ugi L1 boundary

After broad-prior acceptance, the Ugi L1 adapter conditions or inpaints the
reaction-defining product core and jointly predicts the exact amine, aldehyde
and isocyanide decomposition. Broad-corpus replay protects general lipid
support. Recursive L2 and L3 route closure remains a separate hierarchical
module coupled later through a calibrated synthesis value.
