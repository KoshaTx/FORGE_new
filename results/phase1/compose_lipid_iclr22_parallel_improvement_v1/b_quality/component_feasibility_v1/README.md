# Complete-component feasibility census

The saved TRAIN support set is nonempty for every precursor role in all **128 requests**: 64 A3 and 64 ketone Ugi4. This includes all **33 current limited-design failures** (12 A3; 21 ketone Ugi4). This is a necessary structural-compatibility result, not a successful component replacement or an admitted recombined assembly.

The requests use TRAIN-derived layouts, so compatible TRAIN witnesses are partly expected by construction. The census rules out an empty matched support set for these sampled requests. It does not independently establish decoder causation, broad source adequacy, source-disjoint feasibility, or molecular realism.

| Family / source role | Requests with support | Distinct matched TRAIN identities across requests | Current joint identity among matched witnesses |
|---|---:|---:|---:|
| A3 / aldehyde | 64/64 | 9 | 9/64 |
| A3 / alkyne | 64/64 | 39 | 12/64 |
| A3 / amine_head | 64/64 | 73 | 5/64 |
| Ketone Ugi4 / amine_head | 64/64 | 39 | 18/64 |
| Ketone Ugi4 / carboxylic_acid | 64/64 | 17 | 7/64 |
| Ketone Ugi4 / coupled_ketone | 64/64 | 11 | 0/64 |
| Ketone Ugi4 / isocyanide | 64/64 | 4 | 28/64 |

The last column uses all 64 requests, retaining requests without accepted components. A missing identity match does not establish bad chemistry: the comparison also requires the specific witnessed source tree representation, morphology and ordered core correspondence.

## What was counted

The producer scanned all **61,131 admitted TRAIN graphs** (34,104 A3 and 27,027 ketone Ugi4) without altering weights or sampling. It retained 1,039 distinct component/profile witnesses across 23 assembly templates. Matching preserves source role and quantity, origin-block sizes, fixed assembly core, ordered core coordinates, role morphology, actual saved-tree fundamental ring sizes, and the scoped retained-head witness. Ketone head/site definitions come from the pinned registry and adjudication; A3 does not inherit a Ugi head policy. Every request has all match counts and matching witness indices in `run_v2/result.json`.

No component recombination, forward source-executor call, new molecule, model evaluation, TEST access, network request or GPU call was performed. Compatible components were witnessed separately in existing admitted source graphs; simultaneous mixed-component compatibility remains unassessed.

## Practical limit and next step

All **62 accepted ketone Ugi4 coupled-ketone identities are role-TRAIN-novel**. Each request has only one or two matched TRAIN ketone identities (11 distinct across the family). Under the unchanged 62-component-bearing population, copying even one of those identities would reduce the required 62/62 novel-observation floor. Thus literal whole-component TRAIN copying is not a promotable remedy under the current contract. The source witnesses can instead motivate a whole-component topology feasibility test that retains generated chemistry, but the existence of a novel, fully admissible repaired identity has not been demonstrated.

The next recommendation is a saved-data correspondence/feasibility gate before any new proposals, detailed in `next_test_recommendation.json`. A later repair test would require a separately reviewed protocol, complete request denominators, all existing chemistry and diversity/novelty checks, identity-specific route reassessment and an unchanged-original fallback. No such repair or proposal test was started. No molecule is rejected merely because its topology is visually unfamiliar or absent from TRAIN.

## Verification and cost

The independent saved-witness verifier replayed all **1,039 profile witnesses** from **829 distinct TRAIN records**, independently recounted all 128 request matches, reproduced the earlier complete TRAIN family/role/identity counts, and verified **44 ketone head witnesses with zero failures**. `run_v2/verification.json` passed. The focused producer tests passed **13/13**; source formatting and Ruff passed before freezing.

The full census used **16.946188 CPU seconds**; independent witness verification used **2.019642 CPU seconds**, one thread. The first-200-record preflight projected 16.573 CPU seconds. The original startup attempt failed before the TRAIN scan because it expected `attempts` rather than the actual saved `candidates` key. Its protocol, failure and log remain intact at this directory's top level; it is an implementation startup failure, not a scientific negative. The corrected version uses `run_v2/` and has an actual saved-schema regression. All work remained within the authorized aggregate 600-CPU-second ceiling.

## Reproduction and provenance

Run from `.worktrees/iclr22-improve-b_quality` with that worktree as `PYTHONPATH` and one CPU thread. The frozen producer is `experiments/phase1/multireaction/component_feasibility_census_v2.py`; its `freeze` and `run` stages intentionally refuse overwriting existing artifacts. The independent verifier is `run_v2/verify_saved.py`. Use a new versioned output only after review; do not rerun into these completed directories. `run_v2/protocol.json` binds the source code, tests, all source/cache/layout/registry inputs and the current joint population. `run_v2/verification_protocol.json` binds its separate verifier and all verified outputs. `closeout.json` binds this complete handoff.

The census changes no cohort, production source, selection default, manuscript, chemistry gate or evidence contract. The original joint result remains frozen; later route-clock reassessment is separate and uses the same selected identities. Root owns campaign consolidation and the decision-log entry.
