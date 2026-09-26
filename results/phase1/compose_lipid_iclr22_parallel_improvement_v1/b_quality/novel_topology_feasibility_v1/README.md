# Existing novel chemistry on source-witnessed component topology

The audit covered **all 128 current requests** (64 A3, 64 ketone Ugi4) and their 238 distinct request-compatible TRAIN witness profiles. It compared existing saved graphs only. The principal negative finding is **0/64 coupled-ketone topology matches**, including **0/62 exact-L1 products** and **0/2 nonexact products**. This is not a search over all TRAIN ketones, a chemical-impossibility result, or a new quality rejection.

These TRAIN-derived request layouts already have compatible TRAIN component shapes by the preceding census. Here, the question is narrower: does the generated component's existing full adjacency correspond to one of those request-compatible witnessed topologies while preserving the source core and origin coordinates? Atom and bond labels are added in separate levels. Arbitrary exterior-node permutations are allowed; exterior serialization order is not treated as chemistry. The fixed source component-block occurrence and global core coordinates remain anchored, including external core ports. The experiment does not claim a complete search over alternative source origin assignments.

| Family / role | Topology match / 64 | Also bond-labelled / 64 | Full atom/bond-labelled / 64 | Novel identity with topology match / 64 |
|---|---:|---:|---:|---:|
| A3 / aldehyde | 22 | 9 | 9 | 12 |
| A3 / alkyne | 24 | 21 | 12 | 11 |
| A3 / amine_head | 12 | 12 | 6 | 6 |
| Ketone Ugi4 / amine_head | 25 | 24 | 19 | 6 |
| Ketone Ugi4 / carboxylic_acid | 28 | 10 | 7 | 20 |
| Ketone Ugi4 / coupled_ketone | 0 | 0 | 0 | 0 |
| Ketone Ugi4 / isocyanide | 29 | 28 | 28 | 1 |

All 64 requests remain in every family-role denominator, including nonexact products with no admitted component identity. Their graph topology can still be compared, but a matched role does not upgrade whole-product exactness. All precursor roles simultaneously match topology in **2/64 A3** and **0/64 ketone Ugi4** requests. At least one role matches in **47/64 A3** and **52/64 ketone Ugi4** requests. A matched component topology establishes neither whole-molecule realism nor successful synthesis.

Existing novel components on witnessed topology occur in **27 exact A3 requests** and **24 exact ketone Ugi4 requests**. Respectively, **3 and 7** of those requests fail the current limited-design checks. The matched role is not necessarily the failed role; these are not identified repair successes or guaranteed repair targets. All original assessments and denominators remain unchanged.

## Morphology, origin and reactive context

The coupled-ketone role's saved/reconstructed tree morphology equals the requested four-coordinate vector in **19/64 requests** (18 exact, one nonexact), and differs in **45/64** (44 exact, one nonexact). Topology correspondence is still **0/19** within the matching subset and **0/45** within the differing subset. Morphology equality is descriptive on the recorded spanning-tree basis, including reconstructed lowest-parent bases. Some coordinates depend on that basis; this audit does not replace previously admitted basis-specific quality checks or introduce a new rejection gate. All **128/128** saved products preserve the fixed assembly core in the recorded coordinates.

The source templates preserve program, role quantities, repeated-event context and core labels/edges. Correspondence fixes component-origin block occurrence and source-core coordinates, while permitting arbitrary exterior-node relabeling within each origin block. External core ports retain neighbor core coordinates, atom states and bond labels. The matched TRAIN profiles were already qualified for source quantity, block shape, requested morphology/ring sizes and scoped retained-head witness. Matching the existing product graph does not rerun the precursor source executor, establish a retained head after a hypothetical edit, or prove the chemistry of a new atom placement.

## What this supports next

The result supports investigating whole-component topology/morphology control in ketone Ugi4. It does **not** establish whether training or decoding caused the difference. Topology novelty alone is not a defect, and no TRAIN match does not mean bad chemistry. The other roles demonstrate that novelty and an existing witnessed topology can coexist, but no generated coupled-ketone example here supplies that positive witness.

A topology-changing repair remains **unassessed for all 128 requests**. Retaining generated atom states and bond-order assignments on different edges would require defining a new graph and checking valence, reactive-site multiplicity, head survival, exact source replay, identity, novelty/diversity, context and routes. Aggregate atom/bond inventory comparison cannot substitute for those checks; the saved inventory-equality field is descriptive, not an eligibility or chemistry-success test. No such candidate was instantiated, replayed or selected. Therefore this audit motivates a separately reviewed construction test but does not qualify a repair for execution or promotion. Literal TRAIN identity copying remains incompatible with the existing 62/62 novel coupled-ketone observation floor under the unchanged exact population.

## Qualification, provenance and cost

`run_v2/schema_preflight.json` parsed every selected record and all 238 relevant source profiles before matching: 19 selected zero-closure graphs, five nonexact products, 73 selected repeated-role-block records and 41 zero-closure witnesses. Nine focused tests passed, covering exterior permutations, the three evidence levels, core/origin/port protection, bounded unknowns, actual A3/ketone schemas, malformed/cross-origin data and empty closure arrays. Formatting and Ruff passed before freezing.

The independent saved-map verifier checked **2,307 witness/level comparisons**, directly verifying all **173 topology**, **113 bond-labelled**, and **81 full-labelled positive maps** against the saved arrays. These are witness-level counts and can exceed the number of matching requests. It independently checked all 128 denominators and matched-witness identities without repeating isomorphism searches. No searches hit their 20,000-visit / 0.15-CPU-second limits; unknown states would have been retained. Negative matches are the frozen matcher's exhaustive outcomes within its declared origin/core correspondence and admitted witness set, not independent claims about every possible scaffold.

The v2 full match scan used **1.455339 CPU seconds**; preparation (including the complete shape preflight) used **2.537249 CPU seconds** and independent verification **1.936048 CPU seconds**. All work remained within the aggregate 300-CPU-second authorization, using one thread. No GPU, network, TEST, model sampling, route search or source-executor calls occurred.

The v1 scan stopped on an audit-adapter error: NumPy inferred floating dtype for an empty closure list used as indices. Its frozen producer, protocol, log and `run_failure.json` remain preserved. This implementation failure produced no complete scientific result. V2 adds explicit integer arrays and the targeted regression; no input molecule or assessment was changed.

Reproduce from `.worktrees/iclr22-improve-b_quality` with that worktree as `PYTHONPATH`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. The versioned producer is `experiments/phase1/multireaction/novel_topology_feasibility_v2.py`, stages `prepare` then `run`; it deliberately refuses to overwrite frozen output. Use a separately reviewed new output version for any rerun. The protocol pins source modules, tests, selected graphs and their exact construction shards, previous census/verification, TRAIN cache, and all source profiles. `run_v2/verify_saved.py` is the separate bounded verification producer. `closeout.json` pins this complete handoff.

No production default, selected cohort, manuscript, route verdict, likelihood or scientific acceptance gate changed. Parent owns the decision-log and campaign report.
