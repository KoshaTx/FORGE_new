# M0-06 DeFoG Feasibility Gate

## Decision

Do not promote the tested dense edge-matrix parameterization to full
product-prior training. Keep discrete flow matching and whole-lipid
generation, but replace the dense independent edge representation with a
sparse or hierarchical molecular topology process before the production run.

This is not a decision to generate fragments or select building blocks. The
model must still generate the complete lipid graph. The change concerns how
the sparse set of molecular bonds is represented and updated.

## Frozen probe

The gate uses the source-study split frozen in M0-03. It trains small
permutation-equivariant clean-marginal predictors at `N_max=64` and
`N_max=96`. Nodes and upper-triangle edges are corrupted independently using
the DeFoG linear interpolation path. The edge vocabulary includes an explicit
no-bond class. Endpoint sampling uses the minimum R-star continuous-time
Markov chain rate and an empirical training-fold node-count prior.

Each run uses 160 optimization steps. This budget probes representation and
sampling failure modes. It is not a converged product-prior training run.

Two objectives were frozen:

1. `standard_edge_ce` uses the reference clean-marginal node and edge
   cross-entropy objective.
2. `bond_recall_auxiliary` adds a bonded-edge loss with weight 0.25 to test
   whether countering no-bond dominance produces a different failure.

The two objectives use identical data subsets, initialization seeds, sampling
seeds, and architectures at each size.

## Input support audit

The hash-pinned R0 file does not match the size profile previously copied into
the plan.

| Profile | Rows | At most 64 | At most 96 | Maximum |
|---|---:|---:|---:|---:|
| Full current R0 | 15,433 | 10,877, 70.48% | 14,079, 91.23% | 282 |
| Declared C/N/O/S/P support | 15,089 | 10,694, 70.87% | 13,838, 91.71% | 282 |
| Previously documented | 15,433 | 86.5% | 96.8% | 138 |

The current R0 median is 53 heavy atoms. Ninety-two structures contain F and
252 contain Si, so 344 rows are outside the currently declared C/N/O/S/P
element vocabulary. They are reported as support exclusions, not silently
discarded observations.

The current corpus therefore leaves 1,251 declared-vocabulary structures
above 96 heavy atoms. A successful 96-node dense run would not by itself
justify calling the full current R0 size range supported.

## Results

### Reference edge cross-entropy

| Metric | N=64 | N=96 |
|---|---:|---:|
| Training throughput, graphs/s | 249.46 | 114.01 |
| Peak process RSS after run | 0.83 GB | 1.16 GB |
| Held-out node accuracy | 0.865 | 0.881 |
| Held-out bond recall | 0.474 | 0.237 |
| Held-out edge accuracy | 0.953 | 0.965 |
| Held-out bond-density calibration error | 0.0123 | 0.0031 |
| Endpoint bond-density error | 0.0312 | 0.0083 |
| Endpoint connectedness | 0/48 | 0/48 |
| Endpoint RDKit validity | 0/48 | 1/48 |
| Valid and connected | 0/48 | 0/48 |

The dense computation itself fits in memory and its throughput falls by the
expected order as the number of pair slots grows. The failure is statistical
and chemical. At 96 nodes, standard edge cross-entropy retains excellent
overall edge accuracy and sparsity calibration while bonded-edge recall falls
by 0.237. The overall accuracy is dominated by correctly predicting absent
edges. Sampled endpoints are sparse but disconnected.

### Bond-recall auxiliary stress

| Metric | N=64 | N=96 |
|---|---:|---:|
| Held-out bond recall | 0.529 | 0.516 |
| Held-out bond-density calibration error | 0.0801 | 0.1106 |
| Endpoint bond-density error | 0.1363 | 0.1503 |
| Endpoint connectedness | 48/48 | 48/48 |
| Endpoint RDKit validity | 0/48 | 0/48 |

The auxiliary term prevents the 96-node recall collapse and makes every
endpoint connected, but it overpredicts bonds. All 96 generated endpoints fail
sanitization through atom-valence violations. Reweighting the same dense
independent-edge objective therefore trades disconnected underbonding for
invalid overbonding.

Exact full-graph reconstruction is zero under both objectives. This is a
deliberately strict metric over every node and edge, and the bounded run is not
converged. It is reported because the gate requires it, not hidden or softened.

## Architecture consequence

The production candidate is a hierarchical sparse discrete flow:

1. Generate the complete set of heavy atoms and their flat atom states.
2. Generate a connected sparse molecular backbone through categorical
   endpoint choices rather than scoring every absent pair independently.
3. Generate bond orders on active edges.
4. Add a sparse set of ring-closure or residual edges through factorized
   endpoint proposals.
5. Apply explicit valence and connectivity masks to transitions.

The endpoint proposal must retain full pair reachability. A fixed neighbor
list, fixed fragment library, or fixed degree cap would narrow molecular
support and is not acceptable. The Ugi product motif remains a condition or
inpainting constraint for the prospective instantiation, not a supplied
product topology.

This representation remains a whole-graph discrete flow. It does not change
the settled hierarchical molecule and synthesis-program factorization, the
recursive L2/L3 route system, or oracle and synthesis tilting.

## Claim boundary

M0-06 establishes that the tested dense representation should not be scaled
into the full FORGE product prior. It does not establish production generative
quality for the proposed sparse replacement. A bounded sparse architecture
test is required before full product-prior training.

The complete machine-readable record, including hashes, configuration,
size-stratified metrics, marginals, memory, throughput, and decision checks, is
in `results/m0_06/result.json`.
