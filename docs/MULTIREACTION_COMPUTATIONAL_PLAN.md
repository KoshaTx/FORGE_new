# Computational multi-reaction extension

Status: authorized 2026-08-20; implementation and evidence gates are fail-closed.

## Scientific purpose

The extension tests whether reaction programs are a reusable semantic coordinate system for
whole-lipid generation rather than a Ugi-specific encoding. Ugi remains the deep case. BL_2023
aza-Michael addition is the full computational second family; LX_2024 reductive amination is a
lighter third-family stress test. There is no wet-lab or biological-label transfer in this scope.

The permitted conclusion is family-bounded: known reaction structure supplies roles and step
semantics while the model generates molecular identity within the declared graph support. Exact
forward replay establishes transform consistency, not synthesis probability or L2/L3 closure.

## Source-grounded data contract

Inputs are pinned in `configs/multireaction/lnpdb_reaction_programs_v1.json`. The shared typed LNPDB
catalogue is the only row/product/component source:

- LNPDB source rows;
- the vendored qualified-reaction registry;
- the existing source-adjudication document;
- the BL_2023 and LX_2024 supplementary PDFs whose chemistry pages were visually reviewed.

The last three inputs are evidence, not competing datasets. They define transform semantics,
validate source claims, and resolve known defects before admission. They never contribute an
additional measured row or label.

The corpus collapses duplicate measurements by stereo-free constitutional product identity while
preserving every original row in a separate provenance ledger. It admits a product only when:

1. the source resolves one terminal head and reaction-family assignment;
2. recursive reverse execution reaches that exact head within the declared step bound;
3. every repeated component satisfies the registry role policy;
4. every individual step and the complete ordered program replay to the product exactly;
5. source component identity agrees with the reviewed source overlay where one is available; and
6. no recorded source conflict applies.

Abstentions remain in the atlas with a reason. The A3-6b reagent/structure conflict is not erased.
The degenerate reductive-amination substructure hit rate is never computed or reported.

## Program representation

A program is an accumulator plus an ordered sequence of repeated inputs:

```text
terminal head --step 1(+ tail)--> intermediate 1
              --step 2(+ tail)--> intermediate 2
              ...
              --step k(+ tail)--> complete lipid
```

This is required by the primary and polyamine libraries. Treating each final lipid as one
two-reactant event would misrepresent the chemistry. The shared contract is implemented in
`forge/assembly/api.py`; registry-backed execution is in `forge/assembly/program.py`.

## Frozen data gates

| Program | Source library | Minimum admitted | Required reports |
|---|---:|---:|---|
| BL_2023 repeated aza-Michael | 720 | 576 | source-library coverage and source-adjudicated precision |
| LX_2024 repeated reductive amination | 180 | 144 | source-library coverage and source-adjudicated precision |

The component split is deterministic and exact-component-disjoint. A product is held out if any
component is held out, calibration if none is held out and at least one is calibration, and train
otherwise. Training weights give each program equal total weight within a fold; raw family counts
never become the training prior.

## Model and experiment boundary

`ReactionProgramVocabulary` and `ReactionProgramConditioning` expose program, precursor role,
registry-derived reaction-core position and complete program depth without component identifiers or
fragments. Core positions are namespaced by program; an exterior state is explicit. Ordered reaction
execution is preserved in the step ledger. Individual copies of an identical repeated precursor are
not assigned artificial step identities because those identities can be non-identifiable under
product-graph symmetry. `AdapterNodeConditioning` accepts adapter-defined state cardinalities instead
of assuming the Ugi vocabulary. Existing Ugi behavior is preserved by backwards-compatible defaults.

The active corpus DAG is `phase1-multireaction-corpus`. It writes:

- `reaction_program_atlas.csv.gz`;
- `reaction_program_steps.csv.gz`;
- `semantic_atoms.csv.gz`, with exact precursor-role origin for every admitted product atom and
  6,798 registry-derived reaction-core positions across all 764 admitted products;
- `source_provenance.csv.gz`;
- `component_disjoint_splits.csv.gz`;
- `manifest.json` and `result.json`, both with authenticated input identities.

The adapter-conditioned multi-family training smoke is implemented as
`phase1-multireaction-training-smoke`. Its training and sampling stages are separate and connected
only by an authenticated deterministic checkpoint. The smoke consumes the model-ready artifacts
without importing historical producers, covers all 194 heavy atoms and three closures observed in
the admitted corpus, and runs matched program, null and program-ID-swapped arms. The sampler draws
factorized count-only programs; it never selects a component identifier or stored fragment.

The current CPU smoke is a plumbing gate, not a model result. Run
`a786b22cdcd4fcd5a70b90c8670b79c331b8736e7167914bbed9f22bbeb43a22` reproduced every training and
sampling artifact byte-for-byte. All three four-step arms had finite loss; 2/4 samples were valid and
connected, and 0/4 had an exact L1 program. That negative is expected from four optimizer steps and
blocks any scientific comparison or paper-facing sampling from this checkpoint. The non-selecting
evaluator retains separate BL/LX validity, connectedness, diversity, product novelty, exact-L1
coverage, forward-replay precision, ambiguity and role-specific component statistics.

The separate two-record overfit gate freezes its thresholds before execution and tests whether the
representation can memorize one example from each admitted family. Run
`53b7d98c2c3dfcba76eaa5664966470c1710af4137061458dfe1af1f14b71863` passed all five gates and
reproduced every artifact byte-for-byte: the loss ratio was 0.0001424, fixed-noise graph tensors were
exact for 2/2 records, and native sampling produced 16/16 valid, connected and exact-L1 products with
24/24 decomposition traces passing exact forward replay. This is representation qualification, not
generalization: only two unique products were generated, no generated component was novel, and the
LX samples had ambiguous exact decompositions. Long-running arms now checkpoint model, optimizer and
all RNG state in private stage work storage; an automated interrupted-versus-uninterrupted test
requires identical final model bytes and published training metrics.

The shared representation gate is now implemented as
`phase1-shared-synthesis-program-representation`. It serializes connected atom-origin components in
a deterministic parent-before-child order while retaining the complete whole-product graph. The
blocks are derived serialization spans, not component identities or fragment tokens. The contract
supports arbitrary evidenced precursor roles, an explicit assembly-introduced role, namespaced core
positions, explicit aromatic atom/bond states and program depths one through four.

The full census represented and constitutionally reconstructed all 113,150 declared records and all
4,472,034 atoms: 112,386 Ugi, 610 repeated aza-Michael and 154 repeated reductive-amination products.
It preserves the complete union support of 194 heavy atoms and three closures; no product was
truncated. All 561,930 Ugi core atom states are explicitly fixed by the Ugi adapter, while the
auxiliary core positions remain conditioning coordinates rather than being assumed invariant. The
result digest is `ef10c20f443ee8a272b916a6da76556cdbaca9e53d58cae75abb5c6069301534`.
This is a representation qualification, not generalization evidence or production authorization.

The shared cache/model integration gate is implemented as
`phase1-shared-synthesis-program-integration`. Its four stages build a three-record equal-program-
mass training cache, train a shared sparse flow, sample from a separately loaded JSON tensor
checkpoint, and apply a fail-closed qualification. Sampling layouts contain only program identity,
depth, per-node role/core coordinates, counts, and adapter-fixed Ugi states; all variable target atom,
pointer, and bond values are scrubbed before sampling.

The bounded result passed the exact Ugi regression control (5/5 fixed atoms and 7/7 fixed edges),
zero-fixed BL/LX equivalence to the generic sparse-flow noising and loss, 3/3 exact fixed-noise tensor
reconstructions, and 12/12 valid, exact native samples with zero fixed-state changes. The initial
768-step run and an inactive-root-sentinel decoder defect are retained as negative intermediate
results in `DECISION_LOG.md`; no acceptance threshold was relaxed.

The matched production-qualification design is frozen as
`phase1-shared-synthesis-program-production-design`. It verifies the real corpus fold counts and
weight fields, then fixes four architecture- and compute-matched arms: Ugi-only, conditioned
three-family, null-conditioning and cyclic program-ID control. Each arm uses three seeds, 1,700
optimizer steps, effective batch 128, the full 194-atom/three-closure support and fixed-final
checkpoint selection. The three-family arms assign one-third total training mass to each program;
within-program records retain `family_balance_weight_raw` for Ugi and `source_balanced_weight` for
BL/LX. Calibration and heldout products never enter training or select a checkpoint.

All family reports must include both exact-L1 decomposition coverage and exact-forward replay
precision, plus validity, connectedness, abstention, ambiguity, diversity, effective component
count and novelty. The Ugi challenger is subject to frozen paired noninferiority margins and zero
tolerance for fixed-state changes or support overflow. Held-reaction-family analysis remains a
secondary stress test. The degenerate reductive-amination substructure hit rate remains forbidden.

The design receipt is strictly reproducible and explicitly leaves production training, sampling
and candidate selection unauthorized. Implementing the production cache/trainer/evaluator and
launching GPU arms require a new explicit authorization after review of this design. A BL/LX-only
production run would not satisfy the frozen matched comparison and must not be presented as one.

The user subsequently authorized implementation and a fail-closed launch sequence. The production
cache, restartable four-arm trainer, fixed-checkpoint evaluator, held-component reconstruction census
and nonselecting molecule audit are implemented. The versioned CPU execution smoke passes every
infrastructure gate but, at two optimizer steps, produces no valid samples and supplies no model-
quality evidence. The earlier unrun L4-only preflight has been replaced by matched L4 and explicit
A100-40GB accelerator benchmarks. Both authenticate the exact full runtime, checked-in cache, smoke
receipts and no-new-test-failures baseline before exercising every arm, effective batch 128,
deterministic replay and a 32-by-194-atom maximum-support stress. After both pass, a hash-pinned
adjudicator selects the lower projected training-only GPU cost, breaking a tie by wall time; this is
an execution decision, not model-quality evidence. The user approved the disclosed private Modal
payload and a USD 4.10 cap. Matched L4 and A100-40GB runs under source fingerprint `9bad1875...`
passed every gate and were independently downloaded and verified. L4 was selected: peak reserved
memory was 660,602,880 bytes on both devices, while the frozen training-only projection was USD 0.98
for all three L4 replicates versus USD 2.79 on A100-40GB. The selection receipt is frozen at
`results/phase1/shared_synthesis_program_accelerator_selection_v1/result.json`. The user separately
authorized the final selected-L4 confirmation. Run
`d97c3dd86b4c2383382a4ed91055135bda0235aaab2db6006d7c1a741c69a2a7` passed every gate on physical
NVIDIA L4, exercised the 194-atom boundary, reproduced exact deterministic replay and projected USD
0.74 training-only GPU cost for all three replicates. Its verified manifests and result are frozen
under the selection package's `selected_l4_preflight/` directory. This closes the accelerator gate;
no production replicate has launched.

## Required paper experiments

1. Ugi-only versus source-balanced multi-family training at matched architecture and compute.
2. Program-conditioned versus program-label-shuffled/null controls.
3. Exact forward consistency, validity, connectedness, novelty and diversity per family.
4. Exact-component holdout, held program-family stress, and Ugi retention/forgetting.
5. Catalogue-free generation versus matched post-hoc reaction filtering.
6. Whole-graph generation versus a strong train-only finite-component catalogue assembler. The
   catalogue arm receives exact components and exact forward chemistry, samples source-weighted
   role marginals, and uses the same per-family attempt budget.
7. Coverage and precision of decomposition, with abstention and ambiguity shown explicitly.

The primary differentiating catalogue metric is the number of distinct valid products per 1,000
attempts that have one exact L1 decomposition containing at least one component constitution absent
from the train-fold catalogue. Raw validity, exact-L1 yield, unique exact-L1 products, whole-product
novelty, diversity and effective component count remain separate metrics. The finite catalogue is
zero on open-ended component yield by construction and may be stronger on exact-L1 yield; the paper
must show this tradeoff rather than collapse it into a composite winner score. Its Cartesian tuple
count is an upper bound, not a count of unique valid products. Report source-product catalogue
coverage by train/calibration/held-component fold and unique component coverage by role. “Maximum
attainable novelty” means a zero component-escape ceiling here; it must not be confused with
whole-product novelty obtainable by recombining old components, which is measured empirically.
Report each FORGE-minus-catalogue contrast with a two-sided paired-seed bootstrap interval; the
three independent seeds, not generated molecules, are the resampling units.

The paper must report the three families in separate columns. It may not pool shallow third-family
support into the Ugi evidence tier or imply prospective validation outside Ugi.
