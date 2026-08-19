# FORGE M0 Report

Status: **M0 scientific and engineering closeout complete, subject to the final
repository-wide verification recorded with the closeout commit.**

Date: 2026-07-30

## Executive conclusion

M0 supports implementation of a broad whole-lipid sparse discrete-flow model
with joint Ugi L1 semantics and a separate hybrid recursive L2 route system.
It does not support a monolithic product-to-complete-route decoder.

The selected architecture is:

`p(x, b, R) = p_theta(x, b) q_phi(R | x, b)`

where `x` is the complete lipid graph, `b` is the exact final Ugi
product-component decomposition when available, and `R` is the recursive L2/L3
route program. Dense supervision supports joint product and L1 learning.
Sparse, positive-heavy upstream supervision supports retrieval, deterministic
transforms, bounded search, and later learned ranking rather than a standalone
L2 model trained from local data.

The broad product model is not a reaction enumerator. It generates one
connected whole graph at atom-and-bond resolution. Ugi role labels and
attachment-relative annotations condition shared message passing; they do not
select component identifiers or generate independent fragments.

The first prospective synthesis instantiation is the AGILE-type
amine-aldehyde-isocyanide Ugi three-component reaction. The architecture can
host additional synthesis programs, but each new final-assembly family requires
its own transform, evidence, applicability rules, and prospective validation.

## Gate summary

| Task | Status | M0 conclusion | Authoritative artifact |
|---|---|---|---|
| M0-01 | Complete | Use AGILE-type amine-aldehyde-isocyanide Ugi 3-CR. It has no carboxylic-acid reactant component, but the reported procedure uses an acidic catalyst. | `results/m0_01/result.json` |
| M0-02 | Complete | Thirty vendored assets are hash-pinned. Missing or altered assets fail closed. | `results/m0_02/result.json` |
| M0-03 | Complete | The model-facing R0 contains 15,229 constitutional graphs with immutable leakage-safe splits. | `results/m0_03/r0_reconciliation.json`, `results/m0_03/constitutional_split_result.json` |
| M0-04 | Complete | Train-derived reaction enumeration improves component-based recovery but remains too narrow to define generator support. | `results/m0_04_constitutional/result.json` |
| M0-05 | Complete | Automated source adjudication admits 1,100 exact measured L1 records, 12,276 transform-consistent virtual L1 records, and 44 exact source-derived L2 routes. | `results/m0_05_source_adjudication/result.json` |
| M0-06 | Complete | Sparse spanning-tree plus residual-closure representation is selected for the production product-prior implementation. | `results/m0_06_sparse_full_support/result.json`, `results/m0_06_ring_support/result.json` |
| M0-07 | Complete with abstention | The role-aware D-MPNN is selected, but no declared biological-guidance domain is authorized. | `results/m0_07/result.json` |
| M0-08 | Decision package complete, endpoint unlocked | Intramuscular functional editing is the current lowest-risk option. No endpoint is hard-coded. | `results/m0_08/result.json` |
| M0-09 | Complete for architecture selection | Use hierarchical joint product/L1 generation with hybrid recursive L2 routing. Continue chemistry curation only when evidence gaps materially constrain eligible candidates. | `results/m0_09/l2_supervision_decision.json`, `results/m0_09/source_priority_result.json` |
| M0-10 | Scientifically blocked and demoted | FlowER cannot run the written transfer gate because the source, checkpoint, mechanism trajectories, and prospective conversion outcomes are unavailable. Deterministic verification remains load-bearing. | `results/m0_10/result.json` |

## M0-01 and M0-02: chemistry identity and immutable inputs

The L1 transform has exactly three reactants: an amine head, an aldehyde, and
an isocyanide. The ester in the primary AGILE products originates in the
aldehyde component, so ester construction is an upstream L2 problem. Manuscript
prose must not call the reaction simply acid-free.

`make verify` authenticates 30 vendored inputs. Substitute or fabricated data
are prohibited. Upstream COMPOSE repositories remain immutable, hash-pinned
inputs.

## M0-03: corrected constitutional R0

The historical input contained 15,433 rows. Constitutional reconciliation:

- collapses 204 duplicate rows into 15,229 unique graphs;
- excludes 100 B4 mixture measurements from single-graph training weight;
- retains the corresponding constitutions through exact B5 support;
- records 100 B5 pure-trans provenance corrections;
- preserves source and isomeric provenance outside the model graph;
- uses no stereochemical labels in the primary product-flow representation.

The corrected split bundle contains zero connected-group leakage. Held-out
denominators are:

- source study: 2,328;
- headgroup: 2,285;
- linker/scaffold: 2,285;
- component family: 1,100.

Every later model and corpus task must load these assignments. It must not
recompute groups or extract reusable components before splitting.

## M0-04: reaction-registry coverage, not generator accuracy

R1-prime extracts components from `R0_train` only and enumerates products under
12 frozen transforms without sampling. Exact recovery is:

| Holdout | All held-out recovery | Non-AGILE recovery |
|---|---:|---:|
| Headgroup | 260/2,285, 11.38% | 205/2,065, 9.93% |
| Linker/scaffold | 402/2,285, 17.59% | 341/2,224, 15.33% |
| Component family | 100/1,100, 9.09% | Not defined |
| Source study | 0/2,328, 0% | 0/2,328, 0% |

The historical and current controls reproduce exactly:

- historical isomeric-string control: 108/14,233;
- current constitutional-string control: 112/14,129.

The source-study zero is a leakage and reaction-registry boundary. Entire
published chemistry platforms are withheld, and several use final assemblies
outside the current registry. It does not imply that the lipids are
unsynthesizable, that shared motifs are absent, or that the whole-lipid
generator cannot learn their structures.

M0-04 is frozen as an audit. It should not be optimized after inspecting the
holdouts. Future route knowledge is added through versioned adapters or
generator-triggered missing-knowledge reviews, followed by new untouched
evaluation.

## M0-05: admissible Ugi L1 supervision

The automated source-evidence gate visually and structurally adjudicates
hash-pinned primary sources and supplementary chemistry. It admits:

- 1,100 exact single-compound measured AGILE L1 records;
- 100 measured B4 mixture executions as abstentions from single-graph training;
- 12,276 exact and unique virtual product decompositions as transform-consistency
  supervision, not measured success;
- 44 structure-resolved exact L2 source routes;
- one unresolved L2 case as an abstention.

The 319-case packet remains an optional external audit. An unrecorded human
chemist decision does not block exact source-adjudicated L1 supervision.

## M0-06: product representation

The corrected R0 contains 15,229 parseable graphs, reaches 282 heavy atoms, and
is fully covered by the declared C/N/O/S/P/F/Si vocabulary. N=64 and N=96 cover
70.27% and 91.14% of R0, respectively.

The selected sparse representation:

- round-trips 15,229/15,229 constitutional graphs exactly;
- round-trips 1,891/1,891 aromatic graphs exactly;
- preserves full pair reachability through parent-pointer and closure states;
- produces valid and connected endpoints in bounded N=64 and N=96 probes;
- passes a memory and throughput dry run at N=282;
- avoids dense N by N hidden edge states.

This is a representation feasibility result, not a trained generative model.

The broad prior retains observed ring structures for representation learning.
The primary automatic Ugi candidate gate allows acyclic products or one
five-membered or six-membered amine-head ring. Macrocycles, fused, spiro,
bridged, and tail-ring topologies require exact evidence and an explicit support
amendment before primary candidate lock. Heads remain generated at atom and bond resolution;
this gate is not a ring-fragment vocabulary.

## M0-07: biological oracle matrix

The matrix compares 17 representation-model candidates across HeLa and RAW
264.7 endpoints using a scaffold-balanced split and six held-component or
held-component-pair schemes. Random splitting is diagnostic only.

The selected candidate is:

`supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble`

Its equal-endpoint metrics are:

- calibration R-squared: 0.3673;
- calibration RMSE: 1.8584;
- calibration Spearman correlation: 0.5960;
- post-selection test R-squared: 0.1830.

The source-faithful LANTERN checkpoint reproduces HeLa random-split
R-squared = 0.8198. The selected FORGE oracle reaches 0.7845 on the same random
assignments with train-only preprocessing. LANTERN's released source pipeline
fits feature and target scaling before splitting, so its score contains test
information and remains a reproduction diagnostic rather than a selection
benchmark.

No declared generative guidance domain passes both the fixed-candidate and
fold-local nested gates. The production checkpoint therefore authenticates and
scores candidates but returns `abstain` for biological tilting. This gate must
not be weakened. Additional measured component-family evidence or a newly
frozen oracle evaluation is required to reopen guidance.

The 12,276 virtual structures have no biological labels. They support L1
transform consistency and oracle distance analysis only.

## M0-08: biological endpoint

The biological endpoint is not locked. The AGILE oracle is a general in-vitro
transfection diagnostic, not a liver-editing or vaccine-immunogenicity oracle.

The current lowest-risk recommendation is intramuscular reporter delivery
followed by functional editing. Liver functional editing remains viable only
behind a liver-relevant bridge or external labeled evidence. The PI must lock
the endpoint, bridge assay, capacity, benchmark, formulation, and advancement
rule before Phase 4 candidate selection.

## M0-09: jointness and L2 supervision

Dense exact L1 supervision includes 12,276 product-decomposition pairs over 93
unique components. The reviewed upstream corpus contains:

- 4 completed source packages;
- 45 L2 route instances, 44 structure-resolved;
- 76 reaction instances;
- 11 reaction families;
- 15 exact LX_2024 aldehyde transfers;
- zero upstream amine-head routes;
- zero reported negative synthesis outcomes;
- zero complete product dossiers that close every L2 and L3 branch.

This supports hierarchically joint generation of complete lipid graphs and
exact Ugi L1 decompositions, followed by hybrid recursive L2 routing using
retrieval, deterministic transformations, bounded search, and later learned
proposal or ranking.

It does not support a monolithic complete-route decoder. Operational route
closure has not yet been achieved.

The source-review priority overlay preserves all 43 acquisition records and
ranks 39 pending sources on six evidence-bounded axes. Missing evidence receives
zero credit and explicit reason codes. Broad mining remains stopped.
Additional review is triggered by repeated missing-route-knowledge failures,
candidate-panel closure needs, missing isocyanide scope, or missing negative
evidence.

The raw 175,032 component triples are an upper bound, not a count of unique,
route-complete, or synthesizable products.

## M0-10: FlowER demotion

The written FlowER pilot cannot be executed honestly because the repository
lacks a vendored FlowER checkout, released checkpoint, and the required
elementary mechanism trajectories. The current evidence registry contains only
one positive and one negative family example for each target class, not the
approximately 32 mechanisms per class required by the gate. Prospective
conversion outcomes are also unavailable.

FlowER is demoted to an optional orthogonal consistency checker. Deterministic
atom-mapped transforms and exact forward round-trip checks remain the
load-bearing verifier. This blocker does not prevent product-model or joint-L1
training.

## Authorized Phase 1 training plan

The user authorized execution of the broad product plus joint Ugi-L1 training
plan on 2026-07-30.

The data roles are:

1. corrected R0, 15,229 observed constitutional lipids, as the primary
   whole-lipid realism anchor;
2. the 464,265 reaction-enumerated lipid products as low-weight auxiliary
   structure data sampled through `realism_weight`, never raw family counts;
3. the 12,276 virtual Ugi products as exact L1 and round-trip supervision,
   capped and source-balanced so that 93 familiar components do not define the
   generator;
4. the 1,100 reconciled measured AGILE products as exact L1 examples and
   separate oracle labels;
5. source-adjudicated external Ugi products only when exact component mapping
   and assay provenance are preserved;
6. the 479,035 USPTO-MIT reactions for future L2 reaction pretraining, not
   whole-lipid product training;
7. R1-prime as a coverage diagnostic, not a bulk product-model corpus.

Training proceeds in three stages:

1. broad sparse whole-lipid pretraining;
2. Ugi-adapter and L1-head warm start;
3. joint unfreezing with frozen broad-corpus replay ratios and masked L1 and
   round-trip losses.

The replay comparison must include broad-to-Ugi allocations of 75:25, 50:50,
and 25:75 unless a pre-run memory or optimization gate justifies a narrower
set. Selection uses calibration evidence only.

The first generator run has no AGILE biological tilt and no full synthesis
tilt. These controllers remain disabled until their corresponding evidence
gates pass.

The selected checkpoint must preserve:

- validity and connectedness;
- broad-corpus validation performance;
- size, element, ring, branching, unsaturation, and topology coverage;
- internal diversity and topology novelty;
- effective component count;
- component novelty outside the frozen 93-component universe;
- exact Ugi forward round-trip consistency;
- no collapse toward common AGILE components.

## Claims permitted after M0

Permitted:

- the sparse whole-lipid representation is feasible across the declared R0
  support;
- exact L1 product-component supervision is dense within the Ugi
  instantiation;
- the data support a hierarchical joint product/L1 architecture and hybrid L2
  routing;
- the current biological oracle must abstain from generative guidance;
- FlowER is not required for Phase 1.

Not yet permitted:

- that a production product prior has been trained;
- that generated E2 candidates are route-complete;
- that AGILE predicts liver or vaccine outcomes;
- that any candidate is synthesizable, formulated, or biologically functional;
- that complete route grounding has been demonstrated beyond Ugi;
- that route-aware guidance outperforms post-hoc filtering.

Those are Phase 1 and prospective experimental results, not M0 findings.
