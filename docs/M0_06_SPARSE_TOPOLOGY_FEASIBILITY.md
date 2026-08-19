# M0-06 Sparse Topology Feasibility Probe

Status: **accepted as the product-prior representation for later implementation**

This is a bounded representation-selection probe. It is not a trained product prior and does not
establish prospective lipid performance.

## Representation

The tested representation uses:

- deterministic Kekulization before flat graph flow;
- a canonical spanning tree with one parent pointer and parent bond per non-root atom;
- factorized endpoint and bond variables for residual closure edges;
- empirical node-count and cycle-rank priors;
- terminal connectivity and valence constraints;
- full atom-pair reachability without a fixed component inventory.

It eliminates dense `N x N x hidden_dimension` edge states. The current reference implementation
still materializes `N x N` parent-pointer logits, which is a documented target for later optimization.

## Corpus support

The initial probe covered all 15,089 single-fragment R0 structures within a
C/N/O/S/P vocabulary. A subsequent full-support extension admitted fluorine
and silicon after establishing that the excluded rows were coherent
experimental chemotypes rather than parser noise: 92 fluorinated aromatic
lipidoids and 252 siloxane lipids.

The extended C/N/O/S/P/F/Si vocabulary covers all 15,229 constitutionally
deduplicated current R0 structures.
Every structure round-trips exactly at both the sparse edge-label level and the
sanitized constitutional-graph level. All 1,891 aromatic structures recover
exactly after deterministic Kekulization and sanitization-based aromaticity
restoration. The maximum observed cycle rank remains 12, with no hidden
head-only or single-ring restriction.

## Lipid-native ring policy

Representation support and prospective-campaign support are deliberately
separate. The broad prior retains experimentally observed cyclic structures
without imposing a fixed ring-fragment vocabulary. That representation
capacity does not automatically authorize every ring topology for the primary
Ugi campaign.

The reconciled 1,100-record AGILE set contains 660 cyclic products. Every
cyclic product has exactly one 5- or 6-membered heterocycle inherited from the
amine head. The aldehyde and isocyanide components are acyclic. No AGILE
product contains a macrocycle, fused ring, spiro center, or bridgehead.

The primary Ugi automatic topology gate therefore admits:

- acyclic products;
- products with exactly one 5- or 6-membered ring in the amine head;
- novel ring atom identities and substitution patterns within that topology,
  provided the component receives the required route and evidence closure.

It does not automatically admit macrocycles, fused rings, spiro systems,
bridged systems, or rings in the aldehyde- or isocyanide-derived regions. A
candidate outside the automatic gate must abstain from primary candidate lock
until exact source or prospective assembly evidence, complete L2/L3 closure,
and an explicit support amendment exist. This gate prevents arbitrary ring
exploration without reducing the generator to a fragment vocabulary.

## Fixed probe result

Both bounded runs used 2,048 training rows, 256 calibration rows, 256 held-out rows, 200 optimization
steps and 48 endpoint samples.

| Metric | N=64 | N=96 |
|---|---:|---:|
| Training graphs per second | 825.51 | 432.02 |
| Endpoint validity | 1.000 | 1.000 |
| Endpoint connectedness | 1.000 | 1.000 |
| Valid and connected | 1.000 | 1.000 |
| Topology novelty among valid | 1.000 | 1.000 |
| Terminal constraint repair fraction | 0.000 | 0.000 |
| Atom-count Wasserstein distance | 1.526 | 2.654 |
| Cycle-rank Jensen-Shannon divergence | 0.0116 | 0.0123 |

The N=96 to N=64 training-throughput ratio is 0.523, above the frozen 0.3 floor. Scaling dry runs
completed at N=128, N=192 and the observed maximum N=282. At N=282, the probe processed 766.26
graphs per second in a no-gradient forward pass and remained below the frozen 12 GB RSS ceiling.

The corrected-corpus full-support rerun also passed every frozen gate. Training
throughput was 789.37 graphs per second at N=64 and 431.63 at N=96. All 96 sampled
endpoints across both sizes were valid and connected, and no terminal
constraint repair was required. The N=282 dry run processed 773.69 graphs per
second in the original full-support run and 749.30 graphs per second after the
constitutional corpus correction.

## Decision

Implement the later whole-lipid product prior with sparse or hierarchical discrete topology flow.
Use the spanning-tree and residual-closure factorization as the reference design, while preserving:

- complete atom-and-bond generation within declared support;
- connectivity by construction;
- valence-aware terminal decisions;
- all observed cycle ranks;
- full pair reachability;
- explicit accounting for every constraint repair or omitted closure.

Do not revert to the rejected dense independent-edge objective, silently cap molecules at 96 atoms,
or interpret the perfect endpoint-validity result as converged generative quality. The bounded probe
uses strong terminal chemistry constraints and a deliberately small optimization budget.

## Reproduction

```bash
make m0-06-sparse
make m0-06-sparse-full-support
make m0-06-ring-support
```

The frozen artifacts are `results/m0_06_sparse/result.json` and
`results/m0_06_sparse_full_support/result.json`. The ring-policy artifact is
`results/m0_06_ring_support/result.json`. They record seeds, configuration
hashes, input file hashes, environments, timings, metrics, thresholds and
decisions.
