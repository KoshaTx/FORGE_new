# FORGE lossless project handoff for Claude

**State date:** 5 August 2026, America/New_York  
**Repository:** `/Users/rmaganti/Desktop/thesis_projects_ML/forge`  
**Purpose:** preserve the complete scientific and operational state needed to finish the computational
study and Nature Biotechnology manuscript without silently reopening settled questions or mixing
development-era results with the final production system.

This is a handoff, not a new scientific freeze. Hash-pinned result artifacts remain the source of truth.
When prose conflicts with a later result artifact, use the later result and update the prose. Do not
rewrite, delete or overwrite an earlier artifact to make the history appear cleaner.

## 0. How the receiving agent must begin

Before editing or running anything:

1. Read `AGENTS.md` completely.
2. Read `.agents/skills/forge-production-engineering/SKILL.md` before code or pipeline work.
3. Read `.agents/skills/forge-paper-writing/SKILL.md` before manuscript work, if available in the
   receiving environment.
4. Read `docs/PLAN.md` sections 1 through 5 and 13, then
   `docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md`.
5. Read the newest entries in `docs/DECISION_LOG.md`, but note that its last branch entry still says
   the cloud launch was pending. This handoff records the completed launch and its results; add a new
   decision-log entry rather than rewriting the old one.
6. Inspect `git status --short`. The worktree is intentionally very dirty and contains substantial
   user and agent work. Never reset, checkout, clean or overwrite unrelated files.
7. Treat every negative result as a completed scientific result. Do not retune a failed controller on
   the same evaluation schedule.

### Authority and contradiction resolution

Use this order:

1. Explicit current user instructions.
2. `AGENTS.md` scientific-validity constraints.
3. Frozen configs and hash-pinned result artifacts.
4. This handoff's state interpretation.
5. `docs/DECISION_LOG.md` chronology.
6. Working manuscript, README and older planning prose.

The README is stale: it still says only M0 exists and no GPU training has occurred. The working
manuscript is also partly stale: its 4,096-product generator section and its old training/checkpoint
description predate the final decoration-coupled all-fold refit, 32,768-draw production run and branch
support correction. Never use those passages as evidence without reconciling them against the result
artifacts listed below.

## 1. Executive state in one page

FORGE is a synthesis-aware, whole-lipid generative framework for ionizable lipids. The first deeply
developed and prospectively intended instantiation is the AGILE-type amine-aldehyde-isocyanide Ugi
three-component reaction. FORGE is not a component-ID enumerator: the model generates the molecular
graphs of the amine-, aldehyde- and isocyanide-derived regions, together with exact Ugi L1 assembly
semantics. Recursive L2/L3 routing is performed on completed generated components before the
prospective panel is locked.

The final molecular generator is a **constrained stochastic graph-flow generator for complete
Ugi-compatible ionizable lipids**. It uses a sparse, role-aware discrete flow over topology and
chemistry, fully stochastic terminal atom, bond and decoration decoding, and exact terminal chemical
admission. Admission requires a sanitized connected product, exact constitutional Ugi L1 forward
reconstruction and registry-defined handles for all three components. Rejected draws are not repaired
or retried.

The final all-fold model was trained from random initialization for a fixed 5,100 updates on all
112,386 balanced products. It uses bidirectional anchor-local conditioning so noisy decoration states
and their scaffold anchors communicate during the flow. It is not called an unconstrained production
generator because an untouched raw census missed preregistered raw-validity and all-handle gates. It
is retained under the narrower exact-admission contract, which yields approximately 92% admitted
products in the final matched 32,768-draw production run.

Biological scoring uses the correctly reconciled LANTERN-compatible AGILE HeLa dataset: 1,100 unique
single-structure measurements from 1,200 nominal records, with 100 mixture-derived measurements
excluded. The LANTERN HeLa values match all 1,100 retained records exactly. Biological applicability
is not exact component identity. It is a frozen, multiview completed-molecule support policy based on
the product plus its amine, aldehyde and isocyanide components and validated from held-out prediction
errors. Exact-new but continuously interpolative components may be scored.

A support-preserving morphology proposal is promoted. It increases supported-terminal yield from
3.87% to 8.64%, a 2.23-fold enrichment, while retaining nonzero probability for all 57,190 qualified
development-support morphology programs. It enriches but does not certify terminal applicability.
Every completed product is still assessed. A matched morphology-potency challenger failed to improve
the number of unique, supported, conservative-high candidates (23 versus 27), so potency does not
alter molecular generation. Potency is used only for conservative terminal ranking within authorized
applicability and novelty lanes.

Midtrajectory synthesis tilting also failed its matched gate and is not used. Synthesis remains central
through complete-product L1/L2/L3 routing before panel lock. Exact registry evidence is checked first,
Graph2Edits supplies primary source-neutral single-step hypotheses, AiZynthFinder supplies a residual
multistep/public-stock challenger, and all outputs are independently adjudicated. A proposal is not
evidence and route nonclosure is missing knowledge, not proof of impossibility.

A scheduler defect originally suppressed realistic branching. It has been corrected without retraining
the generator or relaxing the oracle. Full all-fold morphology support contains 165,360 programs and
restores realistic aldehyde and isocyanide branch sizes. A fresh 4,096-draw branch-conditioned run
produced 3,462 carbon-branched products, including 570 long, ester-bearing, singly branched aldehydes
and zero adjacent carbon-branch chains. Forty-five new branched rows were oracle-scored, representing
39 unique products; 17 rows, representing 15 unique products, were conservative-high. This is enough
for prospective branch selection but not evidence of broad branched-tail oracle generalization.

The current route-blinded shortlist is version 2: 256 unique products comprising 96 broad-prior linear
potency candidates, 96 support-enriched linear potency candidates, 32 oracle-scored realistic branched
candidates and 32 corpus-absent, potency-neutral realistic branch explorations. It retains all 15
unique conservative-high branched products. It has not yet been run through the final identical route
cascade, and it is not a locked synthesis panel.

### Frozen master-plan ledger

This table is the authoritative continuation of `docs/PLAN.md` and
`docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md` for the work completed since those documents were written.
The receiving agent must still read both source plans; this ledger records which branches survived
their gates.

| Workstream | Current state | Required continuation |
| --- | --- | --- |
| Broad R0 representation | Complete | Retain as representation evidence; do not launch broad pretraining without a new frozen gap |
| Ugi data reconciliation and exact L1 transform | Complete and frozen | Preserve exact origins, core semantics and provenance hashes |
| 1.486-million exact-forward universe | Complete | Call it reaction-enumerated support, not synthesis evidence |
| 112,386-product balanced training corpus | Complete and frozen | No generator retraining is currently indicated |
| Sparse topology/chemistry representation | Complete and frozen | Deterministic tree + sparse closures; root only for serialization |
| Full morphology versus size-only | Full morphology retained | Size-only remains an ablation, not production |
| Stochastic decoder | Fully stochastic atoms/bonds/decorations retained | No return to terminal argmax |
| Decoration-coupled architecture | Selected | Final all-fold refit is 5,100 updates from scratch |
| Unconstrained raw-generator promotion | Failed | Never relax its 95%/99% gates or call it unconstrained |
| Exact terminal-admission generator | Retained | This is the final production generator contract |
| Main production generation | Complete | Use the matched 32,768-draw pool and hashes |
| Applicability policy | Complete and frozen | Score only complete products; preserve multiview risk and authority tiers |
| Morphology applicability proposal | Promoted | `rho=0.50`, power `1.50`; 3.87% to 8.64%; no retuning |
| Partial-state biological SMC | Rejected | Closed unless new independent signal exists |
| Morphology potency tilt | Rejected | Use conservative terminal ranking; no same-data retuning or MH |
| Exact-evidence synthesis tilt | Null | Negative result retained |
| Proposal-augmented synthesis tilt | Rejected | No repeat lambda sweep; proposal engines remain for terminal routing |
| Graph2Edits | Qualified as primary route proposer | Proposal only; verify every transformation independently |
| AiZynthFinder | Qualified as residual route challenger | Use after/alongside Graph2Edits for unresolved multistep/stock closure |
| Full branch support | Corrected and independently sampled | No generator retraining; use branch lane and unchanged gates |
| Branch oracle assessment | Complete | Enough for candidate selection, not broad branch-family generalization |
| Route-blinded shortlist v2 | Complete | 256 unique products; exact next input |
| Final v2 L1/L2/L3 route cascade | **Not run** | Immediate next computational task |
| Route-aware candidate dossier | Pending | Build after route cascade and balance audit |
| Final panel lock/preregistration | Not authorized | Requires explicit user/PI decision |
| Synthesis, formulation, in-vitro and in-vivo work | Prospective | Do not invent outcomes or fill placeholders |
| Manuscript computational reconciliation | Pending | Replace stale model/run/controller prose after route closeout |
| Final figures | Paused by user | Do not produce until explicitly resumed |

The final computational system currently intended for the paper is therefore:

```text
balanced Ugi structures and exact precursor mappings
    -> sparse role-aware stochastic graph flow
    -> exact terminal chemistry and L1 admission
    -> broad or promoted applicability-enriched morphology proposal
    -> complete-molecule multiview applicability assessment
    -> conservative HeLa potency ranking where authorized
    -> route-blinded diverse shortlist
    -> exact evidence + Graph2Edits + AiZynthFinder L1/L2/L3 cascade
    -> route-aware user/PI panel decision
    -> prospective synthesis, LNP formulation and biological validation
```

Potency and synthesis do not modify intermediate molecular transitions in the final promoted system.
That is a measured negative result, not an unfinished implementation.

## 2. Scientific identity and claims

### Working title

Use:

> **Synthesis-aware generative design of ionizable lipids for mRNA delivery**

Do not use **synthesis-guided** unless a future generation-time synthesis controller is prospectively
promoted; the current synthesis-trajectory experiment was negative. The framework can still be
described as synthesis-grounded in explanatory prose because complete routes are constructed and used
before panel lock. The adjective in the title should remain `synthesis-aware` under the current
evidence.

### FORGE name

The repository subtitle is `Route-Grounded Generative Design of Synthetically Executable Ionizable
Lipids`, but a letter-by-letter acronym expansion has not been formally frozen. Use `FORGE` as the
platform name. Do not invent a forced expansion in the manuscript without the user's approval.

### Correct framing hierarchy

```text
FORGE shared synthesis-aware whole-lipid framework
    -> reaction-program adapter
    -> AGILE-type Ugi-3 trained instantiation
    -> prospective synthesis, formulation and mRNA-delivery validation
```

The broad framework claim is architectural. The trained molecular distribution and planned
prospective evidence are Ugi-3-specific. Do not say the current checkpoint has learned every lipid
chemistry. Do not reduce the paper to a component enumerator or call FORGE merely a Ugi library
ranker.

### Strongest current computational claim

FORGE generates complete Ugi-compatible ionizable-lipid graphs under declared sparse support,
recovers their three precursor components exactly, reconstructs the final product through the frozen
Ugi transform, reallocates morphology probability toward a validated biological applicability region,
conservatively ranks supported completed products and constructs auditable recursive route hypotheses
before panel selection.

### Claims that remain prospective

- A generated precursor route succeeds experimentally.
- A final lipid is isolated at acceptable yield and purity.
- A lipid forms an LNP with acceptable size, PDI, encapsulation and apparent pKa.
- A generated lipid improves mRNA delivery in vitro or in vivo.
- The broad-prior versus support-enriched arm improves prospective biological hit yield.

These require locked experimental data. Preserve the placeholders until then.

## 3. Operating constitution: rules and reasoning discipline

The receiving agent must inherit not only commands but the reasoning strategy used to avoid scientific
breaks.

### Non-negotiable repository rules

- Preserve the dirty worktree. Existing changes belong to the user unless proven otherwise.
- Use `apply_patch` for edits. Do not overwrite files through shell redirection or destructive Git.
- Create new versioned artifacts. Never rewrite v1/v2 outputs to make a later result look cleaner.
- Every numeric artifact must pin all inputs by SHA-256, record deterministic seeds and use a stable
  schema.
- Run focused tests while iterating, formatter/linter on touched Python, then `make verify` and the
  relevant test suite before declaring completion. Never claim an unrun check passed.
- Read chemistry constants from vendored registries. Never retype SMARTS, roles or chemistry policy
  from memory.
- Never weaken an applicability, validity, route, diversity or support threshold after seeing a result.
- Never silently filter away failed generations, failed syntheses or failed formulations. Report the
  original denominator and disposition of every record.
- Route hypotheses from Graph2Edits or AiZynthFinder are proposal-only. Independent forward
  consistency, evidence and terminal-material closure determine route readiness.
- A missing route means unresolved under the declared search and evidence system, not unsynthesizable.
- Do not call reaction-enumerated support route-certified.
- Do not claim any graph is expressible. Say any graph within the declared atom, bond, size and sparse
  topology support is representable.
- Do not modify `compose_v4` or `compose_rgm`; FORGE consumes their hash-pinned outputs.

The following less-frequently triggered `AGENTS.md` rules remain fully active and must not disappear
merely because the current resume point is later in Phase 1:

- Never sample the R1 reaction corpus using raw `reaction_family` counts. Ugi dominates those rows;
  use the declared `realism_weight` field.
- Never extract molecular blocks from all of R0. Blocks must come from `R0_train`, after splitting and
  before decomposition; otherwise a recovery result is circular.
- Never report the old `reductive_amination` substructure hit rate. Its nearly universal C--N motif
  made that statistic a degenerate artifact.
- Never silently cap molecule size because a dense edge model fails. The sparse parameterization is
  the declared resolution of full bounded support.
- Never make `held_reaction_family` a hard pass/fail requirement. It is a secondary zero-shot stress
  test, not the paper's core route claim.
- Always report retro-decomposition precision together with coverage.
- Ugi support tiers are `E0/E1/E2/E3`, not the historical `I/F/B/N`; E2, known final assembly with a
  generated precursor, is the primary open-endedness claim and E3 remains exploratory.
- The Phase 1 molecular identity is constitutional and stereo-free. Preserve all 13,376 source rows
  in provenance while giving the 12,386 unique constitutional mappings one model weight each.
- Ugi origin and reaction-core membership are orthogonal, and the assembly-introduced amide oxygen is
  not falsely assigned to a precursor.
- Branch budgets are precursor-origin and component weighted. Never let Cartesian-product frequency
  or an unconstrained generic decoder create repeated tail-junction chains.
- Keep biological endpoint interfaces generic. HeLa MTP is the current in-vitro ranking endpoint; do
  not hard-code it as the universal or in-vivo endpoint.

### Scope discipline inherited from `AGENTS.md`

The repository contract is fail-closed. Current user authorization covers the completed Phase 1
generator/applicability work and the next route-assessment closeout described here. It does not imply
authorization to build later phases, launch a new broad-pretraining arm, replace the Ugi adapter,
create a second prospective chemistry, access a sealed future holdout or write a preregistration/panel
lock. If an apparently necessary action expands that scope, stop and ask.

The optional broad R0 pretraining arm remains a decision-gated ablation, not a missing mandatory
experiment. The final Ugi-from-scratch model did not reveal a frozen gap that justifies launching it
now. Generality comes from the shared interfaces and atom/bond representation, not from claiming that
every chemistry was included in this checkpoint.

### Reasoning strategy that must continue

1. **Separate representation capacity, learned distribution and experimental evidence.** A decoder can
   represent chemistry that the trained model rarely samples; a generated graph can be valid without
   being oracle-supported; an oracle-supported graph can lack a route; a route hypothesis can fail in
   the laboratory.
2. **Separate morphology from chemistry.** Morphology is a coarse generated condition and allocation
   coordinate, not the molecule itself. The flow jointly generates topology and chemistry within the
   condition. Root coordinates serialize the graph but are not chemical semantics.
3. **Separate applicability from potency.** Applicability determines when the potency oracle may
   influence a decision. A high potency prediction can never purchase permission outside the support
   domain.
4. **Separate proposal from evidence.** Learned retrosynthesis broadens search. It does not validate its
   own proposal or establish experimental feasibility.
5. **Prefer decisive cheap audits before new architecture.** The BFS/DFS, morphology/size-only,
   stochastic/argmax, decoration-coupling, potency-tilt and synthesis-tilt questions were resolved by
   matched frozen comparisons, not aesthetics.
6. **Treat negative results as closure.** Potency tilting and midtrajectory synthesis tilting were
   tested and rejected. Do not keep tuning them on the same data because they would make the project
   more narratively exciting.
7. **Use conditional sampling instead of distorting the model when a rare but valid chemotype is
   needed.** The branch problem was solved by restoring legitimate support and sampling a branch lane,
   not by adding an arbitrary branching reward or relaxing chemical gates.
8. **Use the smallest model that answers the biological question.** Avoid architecture theater. The
   paper is about actionable lipid discovery, not proving every possible guidance algorithm.
9. **Ask what a delivery scientist needs to understand.** Main Results should explain capabilities and
   consequences in plain language. Put flow mathematics, masks and calibration mechanics in Methods or
   Supplementary Notes.
10. **Challenge user suggestions respectfully.** The user explicitly does not want automatic agreement.
    Accept an idea only when it improves rigor, biological relevance or execution efficiency.

### User preferences that matter

- Work at a top-tier ML-research standard but avoid elaborate methods without a decisive scientific
  role.
- Always inspect real ionizable lipids and the AGILE structures before judging topology.
- Preserve meaningful branching, unsaturation, ester chemistry and tail-length variation.
- Prefer clear, delivery-field prose over dense computational jargon.
- Do not produce final manuscript figures until the user asks again. When resumed, use the BioRender-
  like pastel visual system in `manuscript/FIGURE_VISUAL_SYSTEM.md`, not software-style vector boxes.
- The working PDF should retain the supplied Homological Flows aesthetic where practical: matching
  font feel and restrained layout, no line numbers. Nature Biotechnology content structure still
  controls the writing.
- In chat, avoid badly rendered LaTeX when a plain-language or code-block formula is clearer.

## 4. Data layers and exact denominators

### Broad structural lipid data

- `R0` contains 15,229 accepted unique connected constitutional ionizable-lipid graphs.
- It is useful for representation audits and an optional broad structural-pretraining ablation.
- Broad pretraining was **not** used to produce the current final Ugi generator. Do not claim that the
  final checkpoint was pretrained on LNPDB or on the full broad-lipid corpus.
- Non-Ugi lipids never receive fabricated amine/aldehyde/isocyanide origins or an imaginary Ugi core.
- The deterministic sparse representation exactly round-tripped all 15,229 accepted graphs and all
  tested atom permutations; see `results/phase1/v5_canonical_representation_audit.json` and
  `results/m0_06_sparse_full_support/result.json`.

### Reconciled AGILE biological data

Primary reconciliation artifact:

- `results/m0_07/agile_label_reconciliation.json`
- Ledger: `results/m0_07/agile_reconciliation_ledger.csv.gz`
- LANTERN curated source: `data/vendor/lantern_agile_curated.csv`

Exact facts:

- 1,200 nominal AGILE experimental records.
- 1,100 curated single-structure records.
- 100 mixture-derived measurements excluded from single-graph supervision.
- 20 amines, 11 curated aldehydes and 5 isocyanides in the 1,100-record set.
- 100 pure-trans graph corrections recorded for the B5 source discrepancy.
- Zero mismatches against official HeLa labels.
- Zero mismatches against official RAW labels.
- Zero mismatches against LANTERN HeLa values across all 1,100 retained structures.

The biological sample size is 1,100 products but the role-level independent chemical diversity is much
smaller. This is why random product splits cannot establish novel-component generalization and why the
oracle uses structured role and cluster audits.

### Ugi molecular union

- AGILE virtual products: 12,276 exact transform-consistent products.
- Curated measured single structures: 1,100.
- Constitutional overlap between virtual and measured: 990.
- Unique model product-component mappings after constitutional reconciliation: 12,386.
- Preserved source rows: 13,376. Duplicate source strings never receive duplicate model weight merely
  because they collapse to one constitutional graph.
- Ugi transform: AGILE-type amine + aldehyde + isocyanide three-component reaction. There is no
  carboxylic-acid reactant component. Do not casually call the reaction acid-free because the reported
  procedure uses an acidic phosphorus catalyst.

### Expanded exact-forward universe

- Admitted components: 424 total.
- 264 amines.
- 107 aldehydes.
- 53 isocyanides.
- Exhaustive exact forward assembly: 1,486,415 unique Ugi products.
- These are reaction-enumerated support, not observed syntheses and not route-certified products.

### Final balanced production corpus

Artifact:

- `results/phase1/ugi_balanced_chemistry_corpus_v2/result.json`
- Assignments SHA-256:
  `9a50c703ed6aa919b251678c04b7a44e93b3233a12fc13b5d6697683bb5b5184`
- Cached tensors:
  `results/phase1/ugi_balanced_training_cache_v2/`

Counts:

- 112,386 products total.
- 12,386 current Phase 1 union products.
- 100,000 deterministically selected expanded exact-forward products.
- Development folds: 66,464 train, 15,800 calibration and 30,122 heldout.
- The final production refit later uses all 112,386 records. At that point former fold names remain
  provenance labels only; monitoring on all records is in-sample.
- Training sampling is source-stratified and role-family-raked to 0.5 current-union mass and 0.5
  expanded-enumeration mass. Uniform Cartesian product-row sampling is prohibited.
- Component identifiers do not enter neural tensors.

### Morphology support

The coarse per-role morphology state contains:

- exterior node count;
- junction budget;
- cycle rank;
- attachment count.

It contains no component identifier and no stored molecular fragment.

There are two support sets that must not be conflated:

1. The validated applicability-proposal development support contains 86 amine states, 19 aldehyde
   states and 35 isocyanide states, for 57,190 complete morphology programs. Its promoted score is
   valid only on this set.
2. The final all-fold generator prior contains 106 amine states, 39 aldehyde states and 40
   isocyanide states, for 165,360 complete programs. This is the legitimate full production-refit
   morphology support and is used for the branch exploration lane. Do not extrapolate the learned
   applicability morphology score to the additional states.

## 5. Representation and architecture

### Sparse graph representation

The graph is encoded as:

```text
deterministic spanning tree + sparse residual closure edges
```

The spanning tree is represented by a valid offspring word. Learned parent pointers and dense all-pair
edge generation are not used. A canonical root and traversal exist to serialize and decode the graph;
the root is not a chemical origin and root depth is not the model's primary semantic coordinate.

In Ugi mode, chemical organization comes from:

- exact precursor origin: amine-derived, aldehyde-derived or isocyanide-derived;
- orthogonal reaction-core membership;
- adapter-defined attachment context;
- role-relative position;
- whole-graph message passing or recurrent context.

The reaction core and precursor origin are orthogonal. An atom can belong to the Ugi product core and
still retain the precursor from which it arose. The assembly-introduced amide oxygen is not falsely
assigned to a precursor.

### What is generated

Within each supplied morphology program, the flow generates:

- offspring topology;
- atom states;
- parent-bond states;
- sparse closure bonds;
- decoration anchors;
- decoration atom states;
- decoration bond states.

It does not select an amine ID, aldehyde ID or isocyanide ID from a catalog. Every terminal product is
inversely decomposed into three component graphs and reassembled with the exact frozen Ugi transform.

### Final neural architecture

Final all-fold refit artifact:

- `results/phase1/ugi_decoration_coupling_production_refit_v1/result.json`
- Result-file SHA-256:
  `0905dcc05fad7f89f97a9a8f5ff66045e82b47b9aedd422e0ca9efcf7da32985`
- Downstream checkpoint:
  `results/phase1/ugi_decoration_coupling_production_refit_v1/checkpoint_best.pt`
- Checkpoint SHA-256:
  `3a3fbf2ad1e6a080bbba0476748d10e78de7cd5122e153d8fa7fc67b60823497`

Model properties:

- hidden dimension 192;
- four bidirectional recurrent layers;
- dropout 0.15;
- four bond classes;
- bidirectional anchor-local decoration-state conditioning;
- maximum three children per node;
- maximum 32 atoms per precursor-derived component;
- maximum 80 total atoms under the current Ugi support;
- maximum cycle rank two per component;
- maximum junction budget seven;
- maximum seven decorations;
- maximum attachment count two;
- empirical source probability floor `1e-5`.

The flow trains all topology and chemistry channels, including terminal decoration chemistry. Sampling
uses eight reverse-star steps and stochastic categorical draws at temperature 1.0. Atoms are not
argmax-decoded in the final production system. Bonds and decorations are also stochastic. Feasibility
masks restrict categorical support; they do not replace the generative flow. Exact terminal admission
is a separate, declared condition on complete samples.

### Why decoration coupling was added

An audit found that the earlier vector field flowed decoration anchor/atom/bond states but did not feed
the noisy decoration states back into the node representation. The model could reconstruct decoration
chemistry on clean support graphs but undergenerated ester chemistry and overgenerated ether-like
states on free trajectories. This was a missing conditional dependency, not evidence for adding an
ester bonus, ether penalty or marginal quota.

The only matched architectural correction aggregated noisy decoration states into their anchor node
and conditioned slot outputs on the anchor node. On an independent three-replicate comparison,
development step 3,000 was selected by the frozen hierarchy: chemistry and branch fidelity first,
validity noninferiority second, then the earlier checkpoint within the tie margin. Do not describe this
as manually forcing ester frequency.

## 6. Generator chronology: which results are obsolete and which are final

This chronology is essential because the working manuscript currently mixes older and newer stages.

### Stage A: initial full-morphology development model

An earlier architecture audit selected full morphology-program conditioning at development step 1,000
over size-only conditioning because only the full model passed every frozen gate. Artifacts:

- `results/phase1/ugi_production_refit_checkpoint_selection_v1/result.json`
- `results/phase1/ugi_production_refit_decoder_checkpoint_selection_v1/result.json`

These are important architecture evidence but are **not** the final checkpoint or final training story.
Do not retain manuscript text saying the final production model is simply step 1,000.

### Stage B: local chemistry audit and decoration-coupled challenger

The ester/ether audit reopened chemistry realization without invalidating the sparse morphology
architecture. The bidirectionally decoration-coupled architecture won an independently frozen
checkpoint-duration comparison. Development duration: 3,000 steps on the 66,464-product train fold.

### Stage C: all-fold production refit

The selected 3,000-step development exposure at batch 128 was mapped to the same expected weighted
draws per record over 112,386 all-fold products, yielding a fixed 5,100-update all-fold refit. It was
trained once from random initialization, without early stopping or post-refit checkpoint selection.

Runtime:

- NVIDIA L4;
- Python 3.11.12;
- torch 2.11.0+cu130;
- CUDA 13.0;
- deterministic algorithms;
- float32;
- no mixed precision.

Final in-sample monitoring losses were approximately 0.64108 on weighted draws and 0.44394 on the
all-fold diagnostic pass. These values did not select a checkpoint and must not be presented as
generalization estimates.

### Stage D: untouched raw census failed unconstrained promotion

Four fresh 1,024-program schedules produced 4,096 attempts:

- 3,862 valid products: 94.29%, below the preregistered 95% raw-validity gate;
- 3,746 of 3,862 reconstructed products passed all three handle policies: 97.00%, below the 99% gate;
- all valid products reconstructed exactly through L1;
- valid-product uniqueness 98.08%;
- all adjacent-tail-branch constraints passed.

The thresholds were not weakened. Raw failures were localized to cycle-one heads and excess/ambiguous
amine or aldehyde handles. This checkpoint is therefore not called an unconstrained production
generator.

### Stage E: constrained stochastic generator retained

The narrower operational contract admits only complete samples passing sanitization, exact L1 and all
three handle policies. On the same untouched 4,096 raw draws:

- 3,746 admitted: 91.46% raw-to-admitted yield;
- 98.02% of admitted products unique;
- 80.86% absent from the 112,386-product refit corpus.

No rejected molecule is repaired or retried. The accurate name is:

> constrained stochastic graph-flow generator for complete Ugi-compatible ionizable lipids

### Stage F: final matched 32,768-draw production run

Artifact:

- `results/phase1/ugi_constrained_stochastic_production_candidates_v2/result.json`
- Result-file SHA-256:
  `b2781e9d50c07dbe2653ded9ed34b735a274739ee6b4a94076edf924955f4726`
- Internal result content hash:
  `f32b552489ba68068719f4181dfd95fa7b0eacc4ad44d04ed6b28a0d0d56c558`
- Ledger:
  `results/phase1/ugi_constrained_stochastic_production_candidates_v2/terminal_ledger.jsonl.gz`
- Ledger SHA-256:
  `913ca35cc221785328ce41f9caebb981029d83114d45ba634b28aa5e6f000a2d`

Design:

- 16,384 broad-prior attempts;
- 16,384 promoted support-enriched attempts;
- 32 total shards of 1,024;
- stochastic atoms, bonds and decorations at temperature 1.0;
- eight flow steps;
- no oracle, potency, route or synthesis calls during generation;
- raw terminals sealed before admission;
- no retry or repair.

Broad-prior arm:

- 15,531 raw valid and exact-L1 products;
- 15,055 terminal-chemically admitted: 91.888% of raw attempts;
- 14,226 unique admitted products;
- 12,024 admitted products absent from the all-fold refit corpus;
- 476 amine-handle failures;
- 853 invalid or nonexact-L1 outcomes.

Support-enriched arm:

- 15,448 raw valid and exact-L1 products;
- 15,125 terminal-chemically admitted: 92.316% of raw attempts;
- 13,248 unique admitted products;
- 10,819 admitted products absent from the all-fold refit corpus;
- 323 amine-handle failures;
- 936 invalid or nonexact-L1 outcomes.

Do not replace these denominators with the older 3,975-of-4,096 development pool in the final
generator Results section. That older pool remains historical/diagnostic evidence only.

## 7. Biological oracle, applicability and exact meaning of a scored candidate

### Frozen oracle

Artifacts:

- `results/m0_07/oracle_production_result.json`
- `results/m0_07/oracle_production_checkpoint.pt`
- Checkpoint SHA-256:
  `46f13b8a3cfef1f891e308dfda75e90685d82d1289f0061acdae4be18870e61c`

Selected model:

```text
supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble
```

The production ranking endpoint is HeLa mRNA transfection potency (`expt_Hela`, abbreviated MTP in
the project). RAW 264.7 is not used to rank the HeLa campaign, and neither endpoint is an in-vivo
oracle. The selected model's equal-scheme HeLa means were:

- structured test R-squared: 0.25336;
- structured test RMSE: 2.59521;
- structured test Spearman rho: 0.61194;
- calibration R-squared: 0.50421;
- calibration RMSE: 2.13779;
- calibration Spearman rho: 0.73108.

The HeLa training-resubstitution R-squared is 0.85030. It must never be used as generalization
evidence. The structured split results, not resubstitution, determine scientific authority.

### What `biologically applicable` means here

It means one narrow, decision-specific statement:

> On structured held-out AGILE experiments representing the candidate's novelty regime, the frozen
> potency model showed enough empirical reliability that its score is allowed to affect candidate
> ranking for this completed molecule.

It does **not** mean biologically active, safe, deliverable in vivo, synthesizable, identical to an
AGILE component or inside independent one-dimensional descriptor ranges.

Applicability is assessed only on complete valid products using four chemical views:

1. whole product;
2. amine-derived component;
3. aldehyde-derived component;
4. isocyanide-derived component.

The final policy combines continuous product/component similarity, role-aware novelty and calibrated
uncertainty. Exact identity is a reporting stratum and covariate, not a universal veto. A never-
measured component may be scored when it is continuously interpolative under a structured regime in
which the oracle actually performed acceptably. Conversely, a familiar component does not guarantee
that an unfamiliar product or interaction is supported.

Do not replace the joint policy with independent descriptor min/max bounds. Marginally plausible
length, branching, unsaturation and molecular weight do not establish joint support.

### Oracle-policy chronology

The original M0 oracle freeze was deliberately conservative and abstained categorically on many
unseen-component domains. Later Phase 1 structured applicability audits replaced exact-identity vetoes
with a continuous, multiview policy and explicit authority tiers. This is not permission for arbitrary
OOD scoring. It is a calibrated refinement based on held-role, held-pair and structural evaluation.

Production rescoring artifact:

- `results/phase1/ugi_production_full_support_rescoring_v3/result.json`
- Internal result hash:
  `ccd63904084739ba5fc448ca007c692f9fda0779b0134e868af50b2ba32feee8`

Final production-pool counts:

| Arm | Admitted | Raw all-view interpolative rows | Oracle-scored rows | Unique scored products | Conservative-high rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| broad prior | 15,055 | 1,750 | 1,433 | 1,106 | 694 |
| support enriched | 15,125 | 3,435 | 2,900 | 1,945 | 1,429 |

There were 2,590 globally unique oracle calls after deduplication. Exact measured products are neutral
controls rather than novel candidates.

### Authority tiers that must remain visible

- `qualified_role_holdout`: strongest current exact-new role lane; currently associated principally
  with isocyanide-only novelty.
- `exploratory_weak_absolute_aldehyde_generalization`: continuously supported aldehyde novelty but
  weak absolute held-aldehyde generalization; descriptive/exploratory rather than a broad claim.
- `exploratory_simultaneous_exact_new_tails`: both tail roles are exact-new. Held-pair evidence does not
  automatically equal two independently held exact-new identities.
- `exploratory_weak_absolute_head_generalization`: held-head absolute prediction failed. New heads may
  be generated, routed, synthesized and tested, but their scores must not be described as validated
  potency predictions.
- `exploratory_multiple_exact_new_roles` and `exploratory_all_roles_exact_new`: exploratory only.
- exact measured products: neutral controls.

Do not let a high predicted potency score override a weak authority tier. Preserve the tier in every
shortlist and dossier record.

## 8. Applicability proposal and biological-guidance decisions

### Promoted morphology applicability proposal

Artifact:

- `results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json`
- Internal result hash:
  `f24d0ffeb1eff8fb89014dd037562dd6a80637a7483bf9c6e7585f5f22e47e08`

Frozen parameters:

- mixture strength `rho = 0.50`;
- score power `1.50`;
- original 57,190-program applicability-development support;
- unchanged terminal applicability boundary;
- unchanged effective-support, importance-weight, entropy and diversity safeguards.

On a fresh untouched program, supported-terminal yield increased from 3.87% under the broad proposal
to 8.64% under the promoted proposal: a 2.23-fold enrichment. Every qualified program retained
nonzero probability. More aggressive settings reached roughly 12--13%, but failed frozen diversity
or effective-support safeguards and were correctly rejected. Do not optimize this controller again
using potency, routing or prospective outcomes.

Correct interpretation:

> Morphology-aware allocation changes where FORGE samples before graph generation and more than
> doubles the probability of reaching terminal chemistry on which the HeLa oracle is authorized,
> without narrowing the declared 57,190-program support.

It is an applicability-enriched proposal, not an in-domain guarantee. The complete product remains the
authority and must be checked after generation. When reporting production yield under this proposal,
do not importance-correct the enrichment away. Use broad-prior/proposal importance weights only when
estimating a broad-prior-normalized quantity.

The score must not be extrapolated to the additional all-fold morphology states in the 165,360-state
generator support. Those additional states are sampled through the full all-fold prior, as in the
branch-exploration lane.

### Partial-state SMC is closed

A richer partial-state controller did not add sufficient predictive signal beyond morphology. Invalid
or incomplete graphs were never directly scored by the biological oracle. Do not build partial-state
SMC unless genuinely new data or a prespecified new representation creates independent evidence.

### Morphology-level potency tilting is closed

Artifacts:

- `results/phase1/ugi_morphology_potency_matched_v1/result.json`
- Internal result hash:
  `8290d7d8eb767b9f93752d88aef803ec89c34017824cc803895e4ee2cf9c71fc`
- `results/phase1/ugi_high_potency_challenger_adjudication_v1/result.json`
- Internal result hash:
  `76b609be645bc7f18a7a1643e118e621730e6ff63aae0755465b76a05b6c563b`

Morphology contained a modest potency-ranking signal (AUC approximately 0.623; 95% interval
approximately 0.565--0.679), but the decisive matched terminal result favored the simpler production
method:

- applicability proposal plus terminal ranking: 27 unique supported conservative-high products;
- nested morphology-potency proposal plus the same terminal ranking: 23.

Therefore potency tilting was not promoted. The branch is closed on the current data. No same-data
retuning, partial-state SMC or complete-molecule MH is authorized by these results. The production
biological method is:

```text
promoted applicability-enriched morphology proposal
    -> complete stochastic graph generation
    -> frozen continuous multiview applicability assessment
    -> conservative terminal HeLa potency ranking within explicit authority tiers
```

This remains genuinely generative: the support proposal changes sampling before graph generation.
Potency itself is post-generation ranking. State that distinction exactly.

## 9. Synthesis grounding, route proposals and the failed synthesis tilt

### Route levels

- **L1:** exact AGILE-type Ugi product decomposition and forward reassembly.
- **L2:** synthesis of each generated amine, aldehyde and isocyanide component or a declared direct
  commercial path.
- **L3:** closure of every route leaf to a terminal material under the declared procurement/evidence
  snapshot.

Routing is post hoc to molecular graph generation but **not** post hoc to final panel lock. The full
route cascade must be applied identically to the arm-balanced, route-blinded shortlist before a
synthesis panel is locked. Route attrition is part of the platform outcome, not an invisible cleanup
step.

### Proposal engines and their precise roles

1. Check exact registry evidence and known qualified transformations.
2. Use Graph2Edits as the primary source-neutral single-step proposal engine.
3. Use AiZynthFinder 4.4.1 as the residual multistep/public-stock challenger for unresolved cases.
4. Independently verify reaction direction, reactive site, precursor/product consistency, substrate
   compatibility, evidence level and terminal-material availability.

Graph2Edits and AiZynthFinder propose hypotheses; neither produces literature evidence. A route found
by an engine is not penalized merely because the hand-built knowledge base did not contain it, but it
must satisfy the same independent adjudication as a registry-origin hypothesis.

Hidden exact single-step recovery benchmark:

- Graph2Edits: 29/36 routes recovered in top five;
- AiZynthFinder: 19/36 in top five;
- their union did not improve over Graph2Edits on this benchmark;
- Graph2Edits recovered oxidation, esterification and formamide dehydration;
- neither recovered amine formylation.

AiZynthFinder remains useful despite the weaker single-step benchmark: on a residual checkpoint it
closed 44/64 Graph2Edits-missed components to public stock and resolved 29/97 previously unresolved
leaves. This is why both engines remain in the exhaustive final route cascade rather than replacing
one with the other.

Likely local environments, which must be rechecked before use:

- `.venv-graph2edits-py311-arm64`
- `.venv-aizynthfinder-py311-arm64`

Never restrict route search only to reactions previously used in lipid papers. Lipid precedent is a
useful evidence and prioritization feature; applicable chemistry remains applicable outside the lipid
literature. Conversely, never accept a transformation solely because a generic model proposes it.

### Synthesis tilting result

Artifact:

- `results/phase1/ugi_synthesis_guidance_failure_audit_v1/result.json`
- Internal result hash:
  `6a7f3ac9e673504e5ef6e9af42b29f18eef95f1ce2987496b68349f410490204`

Midtrajectory synthesis tilting was not promoted. It did not improve matched final route-ready yield,
and intermediate route reward was sparse and weakly actionable. Repeating lambda tuning on the same
schedule is forbidden. The proposal engines remain in the final route-assessment stack; they were not
discarded merely because the generation-time controller failed.

The production method is:

```text
complete generation
    -> terminal biological assessment/ranking where authorized
    -> route-blinded diverse shortlist
    -> exact registry + Graph2Edits + AiZynthFinder L1/L2/L3 assessment
    -> route-aware panel lock
```

The negative tilt does not erase the synthesis-aware claim. It narrows it correctly: complete
molecules and their routes are jointly considered before experimental commitment, but route value did
not alter intermediate graph transitions in the promoted system.

### Old route-readiness results are not the current shortlist result

The corrected proposal-augmented route-readiness v3 analysis was performed on an older 1,855-row
production population. Approximate unique counts there were:

- broad: 34 exact-current and 311 family-all-current-or-better;
- support enriched: 61 exact-current and 539 family-all-current-or-better.

Those values demonstrate that exact evidence is narrower than chemistry-family reach. They must not
be copied onto the current 256-product shortlist. The current shortlist requires a fresh identical
route run.

## 10. Branching correction and what it did—and did not—change

### Diagnosis

The selected neural generator was not proven incapable of realistic branching. The original
production sampler inherited an old development-support morphology pool that omitted legitimate
all-fold branch states. This was a support/scheduler defect. It was repaired by rebuilding the
all-fold morphology prior and conditionally sampling the declared branch lane, without retraining the
model, adding a branching reward, changing the oracle or relaxing terminal chemistry.

### Frozen branch-support schedule

Directory:

- `results/phase1/ugi_full_corpus_branch_exploration_schedule_v1/`

Hashes:

- result file SHA-256:
  `adcca743b41ea6b5ad896299e32e5035c4c91a2c309b106403d8ece3506bd17f`
- schedule SHA-256:
  `37261cb5b18f9113878897f733656f0c149f524c00cf91b3734da27ad86ef65e`
- support ledger SHA-256:
  `1fcebfa0576031f131daf007af709a1f0a8dc594557ab84454b9e20ada2d815c`

Support expanded from the old 57,190 biological-proposal programs to the legitimate 165,360 all-fold
generator programs. Admissible aldehyde branch sizes are 7--24; isocyanide branch sizes include 4 and
6--24. The 4,096 scheduled draws contained:

- 1,284 aldehyde-only branch conditions;
- 2,162 isocyanide-only branch conditions;
- 650 both-tail-origin branch conditions;
- 210 exact coarse AGILE-like `(16, 1, 0, 1)` aldehyde conditions.

The schedule's importance ratio versus the ideal conditional branch distribution was approximately
0.99965--1.00025.

### Fresh branch-conditioned generation

Directory:

- `results/phase1/ugi_full_corpus_branch_exploration_candidates_v1/`

Hashes:

- result file SHA-256:
  `7a57853b307067fb53d237bc7115a11c04003ad24a2d08f3dffffe4d48d7992b`
- internal result hash:
  `c8933564b7da51ebbb31531e7b3e193a52b8a929304eb9e4111d108294a407e1`
- terminal ledger SHA-256:
  `287807bbf614d6b7bca40ace20661f61ae3ba29af64f4299dde23d28cdefd065`

Counts:

- 4,096 attempts;
- 3,833 raw valid and exact-L1 products;
- 3,678 chemically admitted products: 89.795%;
- 3,652 unique admitted products;
- 3,432 admitted products absent from the 112,386-product refit corpus;
- 3,462 realized carbon-branched products;
- 1,243 aldehyde-only branched;
- 1,703 isocyanide-only branched;
- 516 both tail origins branched;
- 216 realized linear after chemistry;
- 570 realistic long ester-bearing branched aldehydes;
- zero products with adjacent carbon branch-point chains.

Admission failures comprised 144 amine-handle failures, 11 aldehyde-handle failures and 263 invalid or
nonexact-L1 outcomes.

### Branch applicability rescore

Directory:

- `results/phase1/ugi_branch_exploration_applicability_v1/`

Hashes:

- result file SHA-256:
  `a5c782e19f42fffa615305b5a42b8e85a55c897e6f3af1a72815e272545f5032`
- internal result hash:
  `5348488efddcc6af1c5f9a29ade43721c906fc3bc9a532438ed3182ef6e17b80`
- rescoring ledger SHA-256:
  `db8e10c2a144ae8c56e1b58188188f01662060d815afd9654afd89f06cf5b9e3`

Results:

- 65 all-view interpolative rows;
- 20 exact measured neutral controls;
- 45 new oracle-scored rows, representing 39 unique products/oracle calls;
- 42 calibrated rows;
- 17 conservative-high rows, representing 15 unique products.

All 45 new scored rows were long, ester-bearing aldehyde-origin carbon branches with one branch point.
The two main generated branch aldehydes were:

```text
CCCCCC(C)CCC(=O)OCCCC=O          # C14 analogue
CCCCCC(C)CCC(=O)OCCCCCCCC=O      # C18 analogue
```

around the measured AGILE C16 aldehyde:

```text
CCCCCC(C)CCC(=O)OCCCCCC=O
```

Among the 570 realistic long ester-branched aldehydes, 45/570 = 7.9% were newly oracle-scored and
65/570 = 11.4% were inside the applicability set when exact measured controls were included. Fifteen
of 39 unique scored products were conservative-high: 38.5%.

The measured AGILE oracle set contains 100/1,100 branch products, but all use one branched aldehyde
across 20 heads and five isocyanides. The effective independent branch-family support is therefore one
aldehyde, not 100. The correct claim is that branch candidates can enter the applicability domain and
that the current run provides a reasonable prospective branch pool. Do not claim broad branched-tail
oracle generalization.

If route attrition leaves too few branch candidates, generate another identically configured,
independently seeded branch-conditioned tranche. Expect roughly 39 unique scored products per 4,096
attempts. Do not retrain the model or loosen the applicability rule merely to increase this count.

## 11. Current route-blinded shortlist v2: exact resume point

This is the immediate source population for the next route stage.

Artifacts:

- builder: `src/forge/product/ugi_production_route_shortlist_v2.py`
- CLI: `scripts/phase1_build_ugi_production_route_shortlist_v2.py`
- config: `configs/model/phase1_ugi_production_route_shortlist_v2.json`
- result: `results/phase1/ugi_production_route_shortlist_v2/result.json`
- compressed ledger: `results/phase1/ugi_production_route_shortlist_v2/shortlist.jsonl.gz`
- test: `tests/test_ugi_production_route_shortlist_v2.py`

Hashes:

- config SHA-256:
  `3a49425112895660be16d36bec089a4c5d24040ebe0e29b65cad762fbf85a930`
- builder source SHA-256:
  `206f05820782ea39354a3f9f7a1cc58c7e003db41d41881b836aea1895eb1549`
- CLI SHA-256:
  `e4558135e81470afefd869e95c9a0f46d6543c1d7669a50a045486d410dbb859`
- focused test SHA-256:
  `1ba7d372223427464517adf80a1aa5097ff1ef5cdf50e6abcbc2eb2c717da325`
- internal result hash:
  `f1b6297bde4683a3338a4c90942ba9e393dfa917d308f8966c82f7fa79c55d8c`
- result-file SHA-256:
  `cd6dfb9ae9de1ea3164df74422de42b6b14ea0082e8df34a5de1325fc0a9cbfd`
- shortlist ledger SHA-256:
  `d7b80c3d456ff56665d1c3c3f1e18a9bb9c7533864dfb03a0a92a6f0ba59bb7f`

The shortlist contains 256 unique products:

- 96 `broad_prior`;
- 96 `support_enriched`;
- 64 `branch_exploration`.

Cohorts:

- 192 `potency_exploitation`, preserved exactly from the v1 matched linear arms;
- 32 `branched_oracle_scored_candidate`;
- 32 `branched_synthesis_exploration_no_potency_claim`.

Branch-specific facts:

- all 15 unique conservative-high branch products were retained;
- the 32 oracle-scored branch rows span authority tiers: 16 weak-aldehyde exploratory, 10
  simultaneous-exact-new-tail exploratory, four qualified-role holdout and two multiple-exact-new-role
  exploratory;
- 29/32 oracle-scored branch rows are exact products in the 112,386 structural refit corpus but were
  not measured biologically;
- three oracle-scored branch rows are absent from the refit corpus;
- all 32 potency-neutral branch-exploration rows are absent from the refit corpus;
- branch cohorts collectively span 26 unique aldehydes, 47 amines and 15 isocyanides.

This is intentional. Exact membership in the structural training corpus does not make a product
biologically measured and does not make prospective testing invalid. The exploitation cohorts ask the
oracle to rank unmeasured structures inside its supported chemical regimes. The separate exploration
cohort establishes corpus-absent molecular and synthesis breadth without an unsupported potency claim.

The shortlist was selected without reading route outcomes. It is route-blinded, not route-complete,
and is **not** a locked synthesis panel.

Focused verification already completed:

- `black` and `ruff` passed on the v2 files;
- `pytest -q tests/test_ugi_production_route_shortlist_v2.py` passed: three tests;
- broader branch-focused tests passed during the branch correction;
- `make verify` passed before the final v2 shortlist edit;
- the full `make test` suite was not rerun after v2.

## 12. Exact next computational sequence

Do not reopen the generator, applicability controller, potency tilt, partial-state SMC or synthesis
tilt. Resume at the route cascade.

### Step 1: preflight without changing scientific state

```bash
cd /Users/rmaganti/Desktop/thesis_projects_ML/forge
git status --short
make verify
pytest -q tests/test_ugi_production_route_shortlist_v2.py
```

Then identify the existing versioned route evaluator/config whose input schema can consume the v2
shortlist. Inspect rather than guess:

```bash
rg -n "production_route_shortlist|proposal_augmented|Graph2Edits|AiZynthFinder" \
  scripts src/forge configs/route configs/model tests
```

Do not silently point a v1 route config at the v2 ledger. Create a new versioned config and result
directory with the v2 ledger SHA pinned.

### Step 2: run one identical, arm-blind proposal-augmented route cascade

For every one of the 256 rows:

1. verify exact L1 decomposition and forward assembly;
2. query exact evidence/current registry status;
3. run Graph2Edits for missing L2 transformations;
4. run AiZynthFinder on the residual unresolved component/leaf set;
5. perform independent forward/role/evidence/stock adjudication;
6. record complete route program, every leaf and every abstention;
7. preserve candidate arm, authority tier, branch flags and potency fields without exposing them to the
   route algorithms.

Use the same search budget, engine versions, stock snapshot and evidence rules for all arms and
cohorts. Cache exact duplicate components so compute is not wasted, but write candidate-level route
receipts so shared-component reuse remains auditable.

### Step 3: route-attrition and balance audit

Report at minimum, by arm and cohort:

- exact L1 pass;
- exact-evidence current route;
- Graph2Edits proposal coverage;
- AiZynthFinder incremental coverage;
- union proposal coverage;
- fully adjudicated L2/L3 route closure;
- unresolved versus rejected routes;
- number and class of unresolved leaves;
- unique amines, aldehydes and isocyanides retained;
- authority tiers retained;
- conservative-high retained;
- branched and linear retained;
- saturated/unsaturated, ester/ether/amide and tail-length distributions;
- corpus-present versus corpus-absent retained;
- matched broad/support arm counts and any imbalance introduced by routing.

Never say a route-failing candidate is impossible. Say it was unresolved or rejected under the frozen
proposal, evidence and stock snapshot, and give the reason.

### Step 4: determine whether a second branch tranche is needed

Only after route attrition is known. If a sensible prospective panel can retain several chemically
distinct route-ready branch candidates, stop. If not, launch another identically configured branch
tranche with a fresh seed and the same admission/applicability policies. Do not change the branch
definition or oracle boundary.

### Step 5: build a route-aware decision package, not a panel lock

Produce a versioned dossier with:

- complete molecule and exact component graphs;
- generation arm and schedule receipt;
- morphology and chemical descriptors;
- applicability views, authority tier and oracle uncertainty;
- conservative potency score where authorized;
- exact L1 mapping;
- L2/L3 route tree, proposal provenance, evidence tier and terminal materials;
- unresolved risks;
- diversity/chemotype cluster;
- recommended experimental stratum.

The user or PI must authorize the final prospective panel. `AGENTS.md` explicitly does not authorize
creating `PREREGISTRATION.md`, so do not create it during this continuation.

### Step 6: verification and decision log

After the route result and audit are finalized:

```bash
UV_CACHE_DIR=.uv-cache uvx ruff check <touched Python paths>
UV_CACHE_DIR=.uv-cache uvx black --check <touched Python paths>
make verify
make test
```

If the full suite is prohibitively slow, run the complete relevant route, shortlist, applicability and
generator tests and explicitly state what was not run. Add a dated `docs/DECISION_LOG.md` entry for:

1. completed full-support branch scheduler/generation/applicability result;
2. v2 route-blinded shortlist;
3. final route cascade and panel-readiness outcome.

Do not edit the older decision-log entry that says branch cloud generation was pending; history should
show the later completion.

## 13. Manuscript state and exact rewrite map

### Current files

- source of prose truth: `manuscript/FORGE_Nature_Biotechnology_working_draft.md`
- LaTeX build source: `manuscript/FORGE_Nature_Biotechnology_working_draft.tex`
- current repository PDF: `manuscript/FORGE_Nature_Biotechnology_working_draft.pdf`
- styled output PDF: `output/pdf/FORGE_manuscript_homological_style.pdf`
- evidence matrix: `manuscript/COMPUTATIONAL_RESULTS_EVIDENCE_MATRIX.md`
- evidence ledger: `manuscript/FORGE_EVIDENCE_LEDGER.md`
- editorial analysis: `manuscript/NATURE_BIOTECH_DRAFTING_ANALYSIS.md`
- editorial guide: `docs/MANUSCRIPT_EDITORIAL_GUIDE.md`
- visual system for later use: `manuscript/FIGURE_VISUAL_SYSTEM.md`

The Markdown draft is about 5,822 words including Methods, legends and references. Recalculate journal
word counts by section after the computational rewrite; do not treat the raw file count as the Nature
Biotechnology main-text count.

### Stale passages that must be replaced before sharing externally

The current Abstract and generator Results still report 3,975/4,096 from an older pool. The final main
production evidence is the two-arm 32,768-draw run:

- broad: 15,055 admitted from 16,384 attempts;
- support: 15,125 admitted from 16,384 attempts.

The branch-conditioned 4,096-draw lane is a separate purpose-built exploration tranche and must not be
silently pooled into the matched main denominator.

The current Methods says the final model stopped after 4,750 updates under calibration-loss selection.
That is obsolete. The final decoration-coupled all-fold refit was fixed at 5,100 updates, trained once
from random initialization on all 112,386 records, with no early stopping or post-refit checkpoint
selection. Development step 3,000 was selected earlier under the frozen matched duration policy.

The current draft under-describes the raw-census failure and exact-admission contract. It must explain
that the untouched refit did not pass the preregistered unconstrained raw gates, those gates were not
relaxed and the model was retained as a constrained stochastic generator with no repair or retry.

The current generator section's branching percentages describe an older, scheduler-limited run. Use
the final matched production pool for general generation and the independent branch-conditioned run
for restored branch capability. Explain the scheduler defect transparently; do not make it sound like
the neural model was retrained to force branches.

The synthesis Results must state that generation-time synthesis tilting was tested but not promoted,
while the proposal-augmented complete-product route workflow remains in production. Do not imply that
Graph2Edits/AiZynthFinder route results for the current v2 shortlist already exist.

The prospective synthesis, formulation, in-vitro and in-vivo sections contain placeholders. Preserve
them as placeholders until data exist. Never write successful prospective outcomes in anticipatory
past tense.

The evidence matrix and evidence ledger predate the final decoration-coupled refit, matched 32,768
run, branch correction and v2 shortlist. Update them alongside the manuscript so every numeric claim
maps to one exact result path and hash.

### Recommended delivery-facing Results structure

Use finding-led headings, not implementation jargon. A strong current structure is:

1. **FORGE designs complete ionizable lipids together with synthesis programs**
2. **Aligned Ugi chemistry links whole-lipid structures to their three precursors**
3. **A sparse generative model learns the chemical organization of Ugi lipids**
4. **FORGE generates new and chemically varied Ugi-compatible lipids**
5. **Applicability-aware sampling concentrates candidates where biological ranking is supported**
6. **Hybrid route search connects generated components to actionable starting materials**
7. **Prospective synthesis and LNP studies test the complete discovery workflow**

The final titles may change for narrative flow, but they should tell an mRNA/LNP reader what was
learned. Terms such as `terminal chemical admission`, `offspring word`, `support-preserving proposal`,
`bidirectional decoration conditioning` and `authority tier` belong in Methods or a short plain-
language explanation, not in the first sentence of a Results section.

### Prose standard

- Open with the delivery bottleneck: a large ionizable-lipid design space, sparse biological labels
  and the need to convert proposed structures into executable chemistry.
- Explain why Ugi-3 was selected as the first rigorous instantiation: aligned product/component maps,
  biological measurements and practical experimental assembly.
- Introduce computation as the solution to a delivery problem, not as a parade of model modules.
- Distinguish component chemistry, whole-product structure, route evidence and biological outcomes in
  ordinary chemical language.
- Translate every metric into a consequence where space allows. Example: 8.64% means roughly one
  supported terminal per 11.6 valid generations rather than one per 25.8 under the broad proposal.
- Use exact counts and intervals; avoid adjectives such as `robust`, `general` or `state of the art`
  without a frozen comparator.
- The main novelty does not require pretending the failed tilts worked. A whole-lipid generator,
  explicit biological abstention, recursive route construction and prospective in-vivo validation
  can form a coherent high-level paper.

### Figures: current instruction is to pause

Do not create or revise final manuscript figures now. The user explicitly paused figure work. When it
resumes:

- use a BioRender-like pastel, biology-first aesthetic;
- use exact RDKit chemical structures and exact frozen plots for scientific content;
- show route planning as an auditable chemistry pathway, not a generic AI flowchart;
- avoid software-dashboard boxes, neon gradients and generic network imagery;
- retain the restrained typography/layout feel of the supplied Homological Flows PDF without copying
  its scientific rhetoric;
- do not add line numbers to the working aesthetic unless the user later requests a submission build.

## 14. Prospective study design: recommended, not yet locked

The causal experiment should preserve two independently generated arms:

```text
Arm A: broad prior
       -> terminal applicability check
       -> conservative terminal ranking
       -> identical route cascade

Arm B: promoted support-enriched proposal
       -> the same applicability check
       -> the same conservative terminal ranking
       -> the same route cascade
```

Match generation budget, oracle calls, route search budget, diversity rules, synthesis slots,
formulation conditions and assay conditions. Do not let the broad arm keep generating until it catches
up in desirable candidates; supported yield per fixed computational budget is part of the causal
estimand.

The principal prospective claim should be framed as improved discovery efficiency, not universal
potency optimization:

> At matched computational and experimental budgets, applicability-enriched sampling from a frozen
> synthesis-aware generator, followed by the same conservative terminal ranking and route-selection
> procedure, increases the yield of synthesizable and biologically active ionizable lipids relative to
> broad generation followed by ordinary ranking.

Candidate-level experimental denominators should be intention-to-synthesize. Route failures discovered
after lock, synthesis failures and formulation-QC failures must not silently disappear. A secondary
per-protocol analysis may use compounds that were successfully synthesized and formulated.

Panel-size guidance discussed with the user, but not frozen:

- preferred causal comparison: approximately 28 evaluable molecules per arm, locking 30--32 if
  attrition is expected;
- minimum serious proof of concept: approximately 18 per arm;
- separate new-head/route-challenge exploration: roughly 4--8, with no broad oracle-generalization
  claim;
- positive and negative controls plus repeated formulation/plate controls;
- if total capacity is below roughly 60 new lipids, prefer a clean one-shot study over consuming the
  panel in an active-learning calibration round;
- if capacity is substantially above 60, a 12--16-candidate calibration round followed by an untouched
  confirmation round may be justified.

The actual panel size must be recalculated from assay variance, expected attrition, cost and the
effect/hit-rate contrast the group is willing to claim. Technical replicate wells do not increase the
number of independent molecules.

The in-vivo subset should use a predeclared advancement rule and retain representation from both
causal arms. A plausible 8--12-generated-lipid subset should cover multiple chemotypes and include
controls. Advancing only support-arm hits would show that leads were found but would not preserve the
arm-wise discovery comparison.

For exact-new component claims, distinguish:

1. one active product containing an exact-new component: candidate-level discovery;
2. activity across several independent new aldehydes/isocyanides and partner contexts: lane-level
   discovery;
3. calibrated ranking across independently held component identities: oracle-generalization claim.

The current new-head cohort can demonstrate generative and synthesis breadth. It cannot establish
general new-head potency prediction.

## 15. Frozen claims and prohibited overclaims

### Safe now, with the cited artifacts

- FORGE uses a sparse stochastic graph flow to generate complete Ugi-compatible products rather than
  selecting product or component IDs from a fixed library.
- Generated products have exact precursor-origin semantics and exact Ugi L1 reconstruction.
- The final all-fold model uses the 112,386-product balanced Ugi corpus and stochastic chemistry
  decoding with exact terminal admission.
- The matched production run produced high admitted yield, uniqueness and many products absent from
  the structural refit corpus.
- Morphology-aware allocation more than doubled terminal biological-applicability yield while
  preserving full declared development-support probability.
- Potency morphology tilting and midtrajectory synthesis tilting failed their promotion gates and were
  not used in the production method.
- Graph2Edits and AiZynthFinder add complementary route hypotheses that are independently adjudicated.
- Full-corpus conditional generation restores realistic carbon branching without retraining or
  relaxing the oracle.
- The current 256-product route-blinded shortlist is diverse across arms, branch cohorts, components
  and biological authority tiers.

### Not safe until the next route run

- a stated percentage of the v2 shortlist is fully L1/L2/L3 route-closed;
- a comparison of route closure between broad, support and branch arms;
- a locked number of synthesis-ready candidates;
- that every conservative-high or branched candidate has a complete experimental route.

### Not safe until prospective experiments

- generated routes work experimentally;
- generated lipids are potent, safe or effective in vivo;
- the support proposal increases prospective hit rate;
- the oracle generalizes broadly to new heads, both exact-new tails or all-role novelty;
- branched-tail potency generalizes across branch families;
- FORGE outperforms specific published discovery systems without a matched comparator.

### Language substitutions

Use:

- `reaction-enumerated support`, not `route-certified library`;
- `route unresolved under the frozen system`, not `unsynthesizable`;
- `proposal engine`, not `evidence engine`;
- `applicability-enriched sampling`, not `in-domain guarantee`;
- `conservative terminal potency ranking`, not `potency-optimized generation`;
- `constrained stochastic graph-flow generator`, not `unconstrained valid generator`;
- `synthesis-aware` in the title under the current negative generation-time synthesis result;
- `Ugi-3-compatible ionizable lipids` for trained support, not `all ionizable lipid chemistry`.

## 16. Artifact map by scientific question

### Data and representation

- `docs/M0_REPORT.md`
- `results/phase1/data_contract.json`
- `results/phase1/v5_canonical_representation_audit.json`
- `results/m0_06_sparse_full_support/result.json`
- `results/m0_07/agile_label_reconciliation.json`
- `results/phase1/ugi_balanced_chemistry_corpus_v2/result.json`
- `results/phase1/ugi_balanced_training_cache_v2/`

### Final generator

- `results/phase1/ugi_decoration_coupling_production_refit_v1/result.json`
- `results/phase1/ugi_decoration_coupling_production_fresh_census_v1/result.json`
- `results/phase1/ugi_constrained_stochastic_production_candidates_v2/result.json`
- `results/phase1/ugi_constrained_stochastic_production_candidates_v2/terminal_ledger.jsonl.gz`

### Oracle and applicability

- `results/m0_07/oracle_production_result.json`
- `results/phase1/ugi_production_full_support_rescoring_v3/result.json`
- `results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json`
- `results/phase1/ugi_morphology_potency_matched_v1/result.json`
- `results/phase1/ugi_high_potency_challenger_adjudication_v1/result.json`

### Synthesis and routing

- `docs/OFFLINE_SINGLE_STEP_PROPOSAL_BACKEND_DECISION.md`
- `docs/PHASE1_SINGLE_STEP_RECOVERY_AUDIT.md`
- `results/phase1/ugi_synthesis_guidance_failure_audit_v1/result.json`
- `results/phase1/ugi_proposal_augmented_route_readiness_v3/result.json`
- `configs/route/graph2edits_runtime_qualification_macos_arm64_py311_v1.json`
- `configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json`

### Branch correction

- `results/phase1/ugi_full_corpus_branch_exploration_schedule_v1/`
- `results/phase1/ugi_full_corpus_branch_exploration_candidates_v1/`
- `results/phase1/ugi_branch_exploration_applicability_v1/`

### Current shortlist and next input

- `results/phase1/ugi_production_route_shortlist_v2/result.json`
- `results/phase1/ugi_production_route_shortlist_v2/shortlist.jsonl.gz`

If any listed path is missing, search for the exact basename with `rg --files` rather than recreating it
from memory. If a JSON summary conflicts with this prose, inspect its `inputs`, `decision`, `nonclaims`
and content hash before deciding whether this handoff is stale.

## 17. Known traps, obsolete results and likely failure modes

- The worktree contains hundreds of modified/untracked files from the active study. Do not interpret
  `git status` volume as disposable scratch work.
- README status is obsolete.
- The working manuscript and evidence matrix are stale in the specific ways listed above.
- The last branch entry in `docs/DECISION_LOG.md` says the launch is pending. The launch completed;
  append a new entry.
- Old 3,975/4,096 candidate results are not final generator production evidence.
- Step-1,000/argmax results are architecture/checkpoint diagnostics, not the final decoder.
- The final model uses stochastic atom, bond and decoration draws. Do not regress to argmax to make
  samples look cleaner.
- Feasibility masks constrain valid categorical support; they are not evidence that outputs are
  deterministic or repaired.
- The broad R0 corpus was not used for the final Ugi checkpoint. Do not insert a broad-pretraining
  claim because it sounds methodologically fashionable.
- The 112,386 structural refit corpus can include products that were never biologically measured.
  Structural-corpus membership and AGILE biological measurement are different variables.
- Product novelty is not component novelty, and exact-new identity is not continuous extrapolation.
- Tree leaves are not chemical tail counts. Ugi has two principal tail-bearing origins, each of which
  may branch internally.
- A root is serialization, not chemical semantics.
- The 57,190 applicability-program support and 165,360 all-fold generator support serve different
  purposes. Do not score the extra all-fold states with an extrapolated morphology-applicability model.
- The 8.64% applicability rate is not a final-candidate acceptance guarantee and should not be inflated
  by weakening the policy.
- A high MH acceptance fraction would not necessarily mean useful diversity. MH is not currently part
  of the plan.
- Route family labels help retrieve and organize chemistry but must never hard-block an independently
  valid transformation.
- Procurement availability is time-dependent. Refresh item-level stock evidence before panel lock;
  do not reuse an expired snapshot as current.
- Do not create a second chemistry adapter merely to make the framework claim sound broad. A light
  interface test is enough until another program has data and experimental purpose.

## 18. Operating cadence for Claude

Claude should continue with the same working style:

1. Lead every substantive work period by stating the concrete outcome being pursued and the frozen
   assumptions, in plain language.
2. Inspect evidence before proposing an architectural or policy change.
3. Parallelize independent read-only audits or bounded tasks only when file ownership is clear; never
   have two workers edit the same artifact.
4. Give concise status updates during long work. Do not leave the user guessing whether a run is
   active, detached, finished or failed.
5. Report real blockers and failed gates immediately. Do not soften them or silently substitute a new
   metric.
6. Distinguish `implemented`, `tested`, `executed`, `promoted`, `frozen`, `route-ready`, `panel-ready`
   and `prospectively validated`. These words are not interchangeable.
7. When the user asks a conceptual question, answer from the actual artifacts first and then explain
   the intuition. Do not reassure beyond the evidence.
8. When the user suggests a method, evaluate it against the current bottleneck. Do not add it merely
   because it is fashionable or associated with a respected lab.
9. Prefer a small decisive matched experiment over repeated ideation. Once a branch has a frozen
   negative result, move forward.
10. Never declare the computational study complete while a required final-route result, decision-log
    entry, verification suite or evidence/manuscript reconciliation is missing.

## 19. Definition of the next real stopping point

The computational side is ready for paper-level closeout when all of the following are true:

1. the v2 shortlist has received one identical, versioned Graph2Edits + AiZynthFinder + evidence-aware
   L1/L2/L3 assessment;
2. route attrition, candidate authority, chemistry diversity, branching, unsaturation and arm balance
   have been audited with exact denominators;
3. a route-aware candidate dossier exists for user/PI review;
4. the user/PI has explicitly authorized any final panel lock;
5. the decision log contains the branch completion, v2 shortlist and route outcome;
6. relevant tests, `make verify` and ideally `make test` pass, with any exception disclosed;
7. the manuscript, evidence matrix and evidence ledger use the final generator/controller/routing
   facts and no longer cite superseded checkpoints as production results.

The project is not waiting for another generator retrain. It is waiting for the final route cascade,
route-aware adjudication and evidence reconciliation.

## 20. Literal first prompt/instruction for the receiving Claude session

Give Claude this repository and this instruction:

> Work from `/Users/rmaganti/Desktop/thesis_projects_ML/forge`. First read `AGENTS.md` completely,
> then `.agents/skills/forge-production-engineering/SKILL.md`, and then
> `docs/CLAUDE_LOSSLESS_HANDOFF_2026-08-05.md` completely. If working on prose, also read
> `.agents/skills/forge-paper-writing/SKILL.md`, `manuscript/NATURE_BIOTECH_DRAFTING_ANALYSIS.md` and
> `docs/MANUSCRIPT_EDITORIAL_GUIDE.md`. Do not rely on README or the current working manuscript for
> computational truth until you reconcile them against the handoff's hash-pinned artifacts. Preserve
> the dirty worktree and all negative results. Resume exactly at Section 12: preflight the v2
> route-blinded shortlist, build a new hash-pinned identical Graph2Edits + AiZynthFinder + evidence-
> aware L1/L2/L3 route assessment, audit attrition and candidate balance, and return a route-aware
> decision package. Do not retrain the generator, retune applicability or potency, revive SMC/MH or
> synthesis tilting, relax any gate, create `PREREGISTRATION.md`, lock a prospective panel, or make
> final figures without explicit user authorization. Use `apply_patch`, version outputs, pin all
> hashes and update the decision log. Tell the user exactly what you inspected, what is running and
> what remains.

If Claude cannot access a listed environment, data file, checkpoint or commercial stock snapshot, it
must identify the missing path and expected hash, preserve the existing state and stop that affected
lane. It must not fabricate substitute data or silently downgrade the workflow.
