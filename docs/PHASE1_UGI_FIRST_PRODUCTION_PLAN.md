# Phase 1 Ugi-first production plan

Status: approved execution order for Phase 1 product plus L1 training. This document refines the
training order without changing the paper's platform framing or authorizing L2 routing and guidance.

## Scientific hierarchy

```text
FORGE shared synthesis-grounded generative framework
    -> synthesis-program adapter
    -> Ugi-3-trained and prospectively validated instantiation
```

FORGE is the framework. The model trained and prospectively evaluated in this study is specialized
for AGILE-type amine-aldehyde-isocyanide Ugi three-component chemistry. Ugi-3 is selected because it
provides exact final-assembly semantics, aligned product-component records, corrected biological
measurements and a practical prospective synthesis protocol. It is not presented as universal
empirical validation of every lipid synthesis chemistry.

The editor-facing working title remains **Synthesis-grounded generative design of ionizable
lipids**. Endpoint-dependent candidates are **Synthesis-grounded generative design of ionizable
lipids for mRNA delivery** and the broader **...for RNA delivery**. Delivery wording is frozen only
after the prospective endpoint and evidence are locked.

## Why Ugi-first is the fastest production path

A production-quality universal lipid generator is not a prerequisite for training a useful Ugi
model. The first bounded production arm therefore trains the Ugi-conditioned generator from scratch
as soon as the support-skeleton and atom-origin gates pass. Broad-lipid pretraining is now an
optional, decision-gated comparison rather than the automatic second production arm.

This order provides a strong reaction-core and component-origin morphology signal immediately. The
preferred diversity expansion is to qualify additional unique amines, aldehydes and isocyanides,
then enumerate their exact Ugi products with explicit provenance and balanced component sampling.
Broad pretraining is run only if the resulting Ugi model still shows a measurable representation,
sample-efficiency or held-component gap that non-Ugi whole-product structures could plausibly close.
Training only on a Cartesian Ugi enumeration is not sufficient until it demonstrates
component-disjoint generalization and generation outside the frozen component catalog.

## Current data census

The hash-pinned Phase 1 contract currently contains:

- 15,229 corrected observed constitutional R0 lipids;
- 12,276 virtual Ugi products with one exact qualified decomposition;
- 1,100 source-adjudicated measured Ugi products with exact semantic atom annotations;
- 576 exact source-SMILES products shared by the virtual and measured sets;
- 990 shared products after applying the generator's declared constitution-only identity;
- 12,386 unique constitutional Ugi product-component pairs in the model union;
- 24 amine heads, 62 aldehyde-body-tail components and 9 isocyanide tails after the same
  constitution-only canonicalization;
- 464,265 reaction-enumerated auxiliary lipid products, used only through low-weight
  `realism_weight` sampling.

The qualified v2 expansion adds a distinct production layer without changing those base-corpus
facts:

- 424 admitted Ugi components: 264 amines, 107 aldehydes and 53 isocyanides;
- 1,486,415 exact-forward products in the complete frozen expanded universe;
- 112,386 products in the source-stratified production slice, comprising all 12,386 current-union
  products and 100,000 deterministically selected expanded products;
- 66,464 training, 15,800 calibration and 30,122 held-out products under the frozen component-family
  partition;
- a hash-pinned 792,454,343-byte joint sparse cache covering every selected product.

All 12,386 model products now have exact per-atom semantics materialized through the qualified
atom-mapped forward transform. The full gate reproduced all 1,100 measured semantic records, found
one exact five-atom reaction core per product, and admitted one connected region for each precursor
origin. The 13,376 original virtual and measured source rows remain recoverable through a separate
provenance ledger; constitutional collapse changes training weight, not source evidence.

## Shared core and Ugi adapter boundary

The shared FORGE core owns:

- canonical whole-molecule graph and support-skeleton encodings;
- atom-level sparse morphology and chemistry generation;
- generic origin-role conditioning;
- exact sparse closures and valence-aware atom and bond realization;
- common route-tree, evidence, availability, uncertainty and outcome schemas;
- future route-value and biological-guidance interfaces;
- open-endedness, diversity and applicability evaluation.

The Ugi adapter owns:

- the qualified Ugi product core and core-root policy;
- amine-, aldehyde- and isocyanide-derived roles;
- exact L1 decomposition and atom-mapped forward assembly;
- permitted cross-origin bonds and component-handle validation;
- Ugi-specific corpus records, compatibility evidence and prospective protocol.

The shared generator must not contain scattered assumptions that every synthesis program has exactly
one amine, aldehyde and isocyanide. A small protocol is sufficient initially:

```python
class SynthesisProgramAdapter(Protocol):
    family_id: str
    origin_roles: tuple[str, ...]

    def core_and_root(self, product): ...
    def decompose_product(self, product): ...
    def forward_assemble(self, components): ...
    def validate_origins(self, product, origins): ...
    def allowed_cross_origin_bonds(self): ...
    def validate_handles(self, components): ...
```

Do not overbuild this abstraction. A synthetic test adapter can test the generic interface; it is
not evidence that a second lipid chemistry has been experimentally supported.

## One generator, factored through a coarse morphology program

The production Ugi generator is:

```text
p(x, b | c) = p(M | c) p_theta(x, b | M, c)
```

`M` contains only per-origin exterior atom counts, junction budgets, cycle ranks and attachment
counts. It is sampled from a training-fold, source- and family-balanced categorical program prior.
The three role programs are recombined independently at this coarse level; no component graph,
component identifier or stored fragment is selected. The joint sparse flow then denoises offspring,
atom and parent-bond states together in one shared role-aware sequence backbone. Exact atoms,
heteroatoms, functional groups, unsaturation, charges and bonds remain generated states.

The Ugi production default uses exact component-origin embeddings, within-origin positions, the
generated program and shared noisy topology/chemistry context. The five-atom reaction core remains
adapter-owned and exact. A canonical root exists only for deterministic sparse serialization; root
depth is not treated as molecular semantics. The model receives no clean graph distances and does
not append atoms or select fragments autoregressively. A separate topology-only network remains a
diagnostic baseline, not a second production decoder that duplicates the offspring channel.

Generic broad-corpus head/interface/tail embeddings remain an ablation because those labels are
heuristic. Hard origin-specific atom marginals are prohibited. A smoothed, full-support
origin-conditioned noise marginal may be tested against global and uniform marginals, but it is
retained only if it improves held-component generation without component-catalog collapse.

For broad lipids, the support skeleton retains real carbon branching, rings, ionizable head atoms
and connector chemistry while avoiding morphology inflation from terminal decorations. For Ugi
lipids, every Ugi core atom is protected and the skeleton is rooted canonically in that core.

## Ugi morphology contract

Every Ugi-conditioned sample contains:

- one exact connected Ugi product core;
- one connected amine-derived head-bearing origin;
- one connected aldehyde-derived tail-bearing origin;
- one connected isocyanide-derived tail-bearing origin.

Precursor origin and Ugi-core membership are orthogonal annotations. Four core atoms retain their
precursor origins and the fifth, the amide oxygen supplied by the reaction template, is labeled
`assembly_introduced`. `Ugi_core` is therefore not a mutually exclusive fourth precursor origin.
Each precursor has one adapter-defined core anchor. The core or a removable virtual super-root may
order the component subtrees, but neither is interpreted as a chemically privileged molecular root.
The frozen serialization is the breadth-first spanning tree written in preorder: on the
component-training partition it matches pure DFS's perfect per-origin run continuity, while the
broad-lipid audit gives it zero root-distance stretch and appreciably shorter residual-closure spans.

The component-weighted support audit also freezes where branching is learned. Across the 24 unique
amine components, the exterior head-bearing support contains zero to three induced branch nodes.
Across the 62 aldehyde-derived components, the exterior support contains at most one branch node and
90% contain none. All 9 isocyanide-derived tails are path-like with zero induced branch nodes. No
current tail component contains a repeated chain of adjacent branch junctions. Initial Ugi
morphology programs therefore draw branch budgets by precursor origin from component-weighted
distributions, not product-row frequency. The decoder may retain bounded architectural support for
future branched components, but the primary campaign does not assign probability to unsupported
tail-branch patterns until transferred or generated components supply qualified supervision.

The v2 expansion supplies that additional supervision. Its exact admitted components reach three
children, seven junction units and two cycles, while the source/family-balanced sampler prevents
their Cartesian product frequency from becoming model probability. The same limits cover every
admitted Ugi component and 97.39% of unique parsed LNPDB hydrophobic-component proxies; the eight
proxy exceptions are recorded as seven unusually large multi-arm structures and one steroid-like
four-ring structure, not silently absorbed by relaxing the production support.

The complete 95-component boundary audit found exactly one exterior attachment bond from each
precursor-derived component to its adapter-defined core port. All 12,386 products share one origin
run order under the selected serialization. Ugi morphology is therefore represented more simply as
the fixed qualified five-atom product core plus three jointly denoised, atom-resolution exterior
trees and their sparse within-origin closures. The exterior programs contain only generated node
counts, junction budgets and cycle ranks; component identities are absent from neural tensors.

Only adapter-permitted cross-origin bonds are allowed. Tree leaves are not interpreted as chemical
tails. The two tail-bearing origins can contain learned internal branching and multiple terminal
hydrophobic chain ends. The discrete flow still generates complete component graphs at atom and
bond resolution rather than selecting a component identifier or appending atoms autoregressively.

## Training arms

### Arm A: Ugi from scratch

Train the shared atom-level backbone with the Ugi adapter using source-balanced Ugi product records.
The 1,100 measured products receive the strongest realism status. Virtual records provide dense L1
and exact forward-transform supervision but are not called experimentally successful combinations.

### Optional Arm B: broad pretraining followed by Ugi specialization

Pretrain the identical backbone on R0 observed lipids with low-weight reaction-enumerated auxiliary
support. Non-Ugi structures receive only whole-lipid structure supervision. Specialize with the Ugi
adapter while retaining broad-corpus replay or another frozen anti-forgetting policy.

Before fitting Arm B, materialize protocol-specific decontamination ledgers. Exact held Ugi product
constitutions must be removed from R0/R1 for every claimed component-disjoint evaluation. A stricter
"unseen component" claim additionally requires removing broad records carrying that exact held
component; otherwise the result is labeled label-free transfer to a held Ugi role, not unseen-
component generalization.

Do not launch Arm B by default. First train and evaluate Arm A after qualified Ugi component
expansion. Launch Arm B only if a frozen decision gate identifies a material gap in held-component
generalization, molecular realism or sample efficiency. If launched, compare the two arms on the
same calibration evidence and use the better checkpoint for the prospective campaign.

### Optional joint source-balanced arm

The 75:25, 50:50 and 25:75 broad-to-Ugi replay comparison is retained only as a preregistered option
if Arm B passes the decision gate. It is not required for the Ugi-first production checkpoint.

## Enumeration and motif transfer

Exact enumeration from qualified amine, aldehyde and isocyanide components uses the frozen
atom-mapped Ugi transform. Those products provide structure and L1 consistency supervision, not
experimental success labels.

An arbitrary hydrophobic motif from LNPDB is not restitched directly onto the Ugi core and called a
valid example. It first requires:

1. an explicit transferable motif and original attachment record;
2. realization as a chemically valid Ugi aldehyde or isocyanide component;
3. handle and substrate-scope qualification;
4. exact forward assembly;
5. provenance and uncertainty labels;
6. upstream route closure before it can support an E2 synthesis claim.

This procedure expands synthesis-grounded support without inheriting the biological activity or
experimental success of the source lipid.

## Sampling policy

Do not sample product rows uniformly from the Cartesian corpus. Balance or cap by:

- provenance class;
- component role and structural family;
- component pairs and exact triples;
- branching, unsaturation and embedded linker-like chemistry;
- product structural clusters.

Enumeration frequency must not become model probability by default. Measured-product duplication
does not create extra training weight unless the policy explicitly assigns an evidence weight.

## Component-disjoint evaluation

Freeze before model fitting:

- held amine families;
- held aldehyde families;
- held isocyanide families;
- held component pairs and exact triples;
- held structural motifs and product clusters;
- transferred-component evaluation records;
- the prospective beyond-catalog set.

Random product splits are secondary. The main question is whether an atom-level generator learns
transferable component chemistry rather than recombining familiar identifiers.

The mutually exclusive primary partition is now materialized: a product is held out if any role is
held out, calibration if no role is held out and at least one is calibration, and training only when
all three roles are training components. It contains 4,362 train, 3,350 calibration and 4,674
held-out products. Pair/triple, structural-cluster, transferred-component and prospective ledgers
remain required before their respective final evaluations; they are not yet described as frozen.

## Chemistry-realization boundary

Morphology conditions and chemistry targets are separate data objects. The topology condition may
contain only offspring words, sparse closure endpoints, traversal, generated origin/core/port
semantics and adapter-fixed core values. Exact atom states, tree/closure bond states and terminal
decorations are loss/evaluation targets only. Graph distances are recomputed from a verified sampled
topology before chemistry denoising.

Chemistry uses the production V3 convention: constitutional identity without stereochemistry, with
aromatic atom and bond states represented explicitly. Ugi support records are re-audited under
`preserve_aromaticity=true`; a Kekulized constitutional round trip alone is not sufficient evidence
of exact chemistry-target agreement.

## Checkpoint-time molecular evaluation

Do not select training duration from loss alone. At prespecified early and serial checkpoints,
generate complete molecules and report:

- exact Ugi core and forward reconstruction;
- component-origin connectedness and permitted boundaries;
- atom and bond validity without structural repair;
- morphology, functional groups, branching, unsaturation and rings;
- component and whole-lipid novelty;
- diversity and effective component count;
- concentration on common components;
- held-component performance and broad-corpus forgetting where applicable.

Render matched random and worst-case molecules at every major checkpoint. Sampling is diagnostic and
does not use heldout data to tune architecture or hyperparameters.

## Open-endedness gate

Report generated products in three non-overlapping classes:

1. products using only frozen familiar components;
2. products using transferred, previously known components;
3. products containing genuinely generated components absent from the frozen catalog.

For every class report exact-triple novelty, nearest component similarity, whole-lipid novelty,
diversity, route status and biological-oracle applicability. The primary E2 claim requires a
meaningful prospective subset with at least one beyond-catalog component and a complete upstream
route. A wide atom decoder alone does not earn the open-endedness claim.

Generated candidates from a locked campaign are never fed back into that campaign's training set.
Only a later, explicitly versioned model iteration may learn from prospective outcomes, with the
earlier evaluation records permanently excluded from its training claims.

## Current implementation gates

1. Select the support skeleton using all R0 and Ugi structures. Require exact reconstruction,
   permutation stability, connectivity and preservation of real rings and junctions.
2. Materialize and validate Ugi atom-origin semantics across every record admitted to the origin
   loss.
3. Freeze component-disjoint splits and hierarchical sampling weights.
4. Overfit a bounded Ugi set and render complete chemistry-realized molecules. **Complete:** the
   generated-topology smoke composition produced 22 valid, 22 unique molecules from 24 attempts;
   the two failures were aromatic-kekulization errors, not topology failures.
5. Expand the unique Ugi component registry through qualified head and tail transfer, freeze all
   new component-family splits before product enumeration, and build the balanced product corpus.
   **Complete:** 424 components, 1,486,415 exact-forward products, a 112,386-product production
   slice and its exact joint sparse cache are frozen.
6. Train the full Ugi-from-scratch arm with checkpoint-time sampling and held-component evaluation.
7. Run the broad-pretrained arm only if the frozen post-Arm-A decision gate justifies it.
8. Select a production checkpoint only if it preserves validity, Ugi forward consistency,
   diversity and component novelty.

L2 route generation, biological tilting and synthesis-value tilting remain outside the currently
authorized Phase 1 implementation. Their interfaces and future evaluation rules are recorded, but
they are not implemented until their evidence gates and authorization pass.

## Future synthesis and biological guidance

Once authorized and evidence-qualified, synthesis must alter molecular transition probabilities
before candidate lock. Raw route-model likelihood is not a synthesis-success probability. The
synthesis value uses recursive closure, evidence, forward consistency, burden, availability,
uncertainty and missing-knowledge status. Its guidance strength is selected by maximizing route
closure subject to frozen floors on diversity, broad-distribution coverage and component novelty.

The matched post-hoc baseline uses the same product prior, planner, verifier, compute and final
candidate budget but applies route assessment only after molecular generation. This is the causal
ablation of coupling, not the paper's title-level identity.

Biological guidance remains applicability-gated and uncertainty-aware. It cannot be activated while
the production domain policy abstains, and it cannot inherit activity from a transferred motif or
replace an endpoint-relevant experimental bridge.

## Claim boundary

Permitted framework language:

> FORGE is a synthesis-grounded whole-lipid generative framework in which synthesis-program
> adapters couple atom-level molecular generation to recursive synthesis programs.

Permitted current-system language after successful training:

> The model developed here is specialized for Ugi-3-compatible ionizable lipids.

Permitted prospective language only after experiments:

> Complete route grounding, synthesis, formulation and biological validation are demonstrated in
> the Ugi-3 design space.

Do not claim that the present checkpoint has learned every ionizable-lipid synthesis chemistry. Do
not describe FORGE as merely a Ugi component enumerator. The credible middle position is a reusable
atom-level framework with one deeply trained and prospectively validated Ugi instantiation.
