# FORGE: Synthesis-Grounded Generative Design of Ionizable Lipids

*(cautious pre-results alternative: "Generative Design of Ionizable Lipids with Complete Synthesis
Routes")*

Target venue: Nature Biotechnology · Repository: `https://github.com/KoshaTx/forge`

> **STATUS: SCIENTIFICALLY APPROVED IN PRINCIPLE — NOT IMPLEMENTATION-FROZEN.**
> This document is the **Phase-0 decision and data-gate revision** requested at review. Do not begin
> GPU model training or large-scale route-model implementation. The authorized scope is Milestone M0
> (§13) only: decisions, audits, feasibility gates, and data inventories that can fail cheaply.

---

## 1. Corrected scientific thesis

**Claim.** FORGE couples generation of complete ionizable-lipid graphs to recursive construction of
complete synthesis programs, so each candidate arrives with an auditable route to experimentally
supported terminal materials. The framework is prospectively instantiated in AGILE-type
amine–aldehyde–isocyanide Ugi three-component chemistry, where candidates will be evaluated through
synthesis, isolation, formulation, and functional testing.

The headline is joint whole-molecule and synthesis-program generation, followed by prospective
execution. A matched-compute comparison against post-hoc route filtering is the primary causal
ablation of synthesis coupling, not the paper's identity. FORGE must **not** reuse pulmonary delivery
as its endpoint because that is the biological focus of COMPOSE-Lipid (Paper 2).

**Portfolio boundary.** A separate broad lipid-generator program studies reaction-program-agnostic
coverage of diverse ionizable-lipid chemistry. FORGE may reuse and disclose a shared generative
backbone, but its independent scientific question is whether complete, evidence-qualified synthesis
programs alter generation and improve experimental actionability. The manuscripts must use distinct
principal candidate cohorts, headline results and biological conclusions. The matched in-trajectory
versus post-hoc comparison is therefore FORGE's intended causal differentiator, subject to a bounded
go/no-go experiment; it must not delay complete prospective route dossiers or wet-lab execution.

**Architecture and evidence scope.** The product prior is broad within its declared atom, bond, charge,
and size support. A complete synthesis program is a route tree composed of reaction-family-specific
steps, rather than a single family label attached to a lipid. The current paper deeply validates one
final-assembly program, the AGILE-type Ugi 3-CR, while its upstream routes may already contain
esterification, oxidation, formylation, dehydration, and other transformations. Additional
final-assembly programs require their own route evidence and prospective validation.

The reusable platform layer comprises the common route-tree schema,
reaction-record format, upstream precursor subroutes, evidence and availability
states, forward checks, failure taxonomy, and synthesis-value interface. A new
library chemistry reuses this layer but must add its own final-assembly
adapter, reactant roles, substrate-scope evidence, chemistry-specific
supervision, and prospective validation. The infrastructure is reusable;
chemical support is not automatic.

The Ugi L1 transform has no separate linker reactant. Linker-like spacers and degradable motifs may
be embedded in any generated Ugi component, so the transformations that construct them remain part
of recursive L2 route closure.

Two claims are deliberately *not* made: that FORGE already has learned synthesis across all
ionizable-lipid chemistries, or that arbitrary graphs are expressible (see §6 bounded support).

---

## 2. Decision: Ugi 3CR vs 4CR — **RESOLVED as AGILE-type 3CR**

Settled from primary data, not assumption:

| Evidence | Finding |
|---|---|
| Registry `atom_mapped_reaction_smarts` | `[NX3;H2,H1:1].[CX3H1:2]=[OX1].[C;-1,+0;X1:3]#[N;+1,+0;X2:4]>>[N:1][CH1:2][C+0:3](=O)[NH1+0:4]` — **exactly three reactants**, no carboxylic-acid reactant component |
| Registry `selectivity_policy` | Legacy text calls the transform "acid-free"; the chemically precise result is a three-component alpha-amino-amide-forming transform without a carboxylic-acid reactant component |
| AGILE procedure | Uses an acidic phosphorus catalyst, so the reaction must not be described simply as acid-free or catalyst-free |
| Product architecture | R¹-NH-CH(R²)-C(=O)-NH-R⁴ — α-amino amide. Classic Ugi-4CR yields an **α-acylamino** amide (N-acylated) |
| AGILE table | Exactly `A_smiles`, `B_smiles`, `C_smiles`. Row 0: `CN(C)CCN` + `O=C(CCCCCCCC)OCCCCCC=O` + `CCCCCCCCCCCC[N+]#[C-]` → `CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(C)C` — free secondary amine `NCCN(C)C`, **not** N-acyl |
| Qualification | Exact reconstruction of all **1,200/1,200 nominal AGILE table rows**; M0-07 separately qualifies single-compound supervision |

**Chemically important consequence.** The ester in AGILE lipids is carried **by the aldehyde
component** (B = 6-oxohexyl octanoate), not by a fourth acid component. This is precisely why AGILE's
Tail A route exists (fatty acid + diol → EDC coupling → DMP oxidation): the ester is built into the
aldehyde *before* the Ugi step. **Ester construction is an L2 problem, not an L1 problem.** This single
fact justifies the whole L1/L2 split.

**Open flag for the PI (not blocking M0):** if the wet lab genuinely runs classic Ugi-4CR, that is a
*different lipid architecture* (α-acylamino amide) and the 1,200 AGILE labels do **not** supervise it.
That would be a scope change, not a config change. Config key `configs/assembly/ugi_variant.yaml`
defaults to `ugi_3cr_agile`.

---

## 3. Corrected support tiers (replaces I/F/B/N)

The prior Tier F was logically inconsistent: if R1 exhaustively enumerates qualified transforms over
the block pool, a product built from those blocks by those transforms is already in R1, so F is empty
except for stereochemical/protonation/filter artifacts. Replaced by:

| Tier | Definition | Role |
|---|---|---|
| **E0 — enumerated** | Exact product present in the frozen R1′ enumeration | Memorization floor |
| **E1 — known assembly, known terminal components** | Assembly family in registry; every component already an accepted Tier-A/B terminal block; product not in the enumerated subset | Meaningful **only if R1′ is sampled/capped rather than exhaustive** — declare which |
| **E2 — known assembly, generated precursor** | Assembly family known; ≥1 component is *not* an accepted terminal block and requires a complete L2 route to Tier A/B leaves | **PRIMARY open-endedness claim** |
| **E3 — new assembly family** | Required final disconnection outside the qualified registry | **Exploratory only.** Not required for the main paper |

**Headline claim:** *FORGE generates E2 products outside the enumerated lipid universe whose
previously unseen components are accompanied by complete, forward-consistent precursor routes.*

E3 is upside. The paper does not depend on it.

---

## 4. Non-circular R1′ anchoring, registry coverage, and leakage audit

The earlier gate was circular: harvesting blocks from **all** of R0 and then measuring R0 recovery
partly guarantees the result. It measured decomposition coverage, not generalization.

### 4.1 Split before extraction

Partition R0 (15,433) into `R0_train / R0_cal / R0_heldout` **before any retro-decomposition**, under
four independent holdout schemes (reuse the existing leak-free groups in `splits_v1/`):

- source-study holdout · headgroup holdout · linker/scaffold holdout · component-family holdout

### 4.2 Pipeline

1. Retro-decompose **`R0_train` only**.
2. Build the component pool from **`R0_train` only**, tagged `provenance: r0_derived` with occurrence
   counts and source study.
3. Construct **R1′ from training-derived components only**.
4. Evaluate on `R0_heldout`.

### 4.3 Metric

```
Recovery_heldout = |R0_heldout ∩ R1'_train-derived| / |R0_heldout|
```

Plus **nearest-reachable distance** (ECFP4 Tanimoto to closest R1′ member) for held-out lipids not
reproduced exactly — a corpus can be useful without exact recovery, and that must be visible.

The source-study analysis is manuscript-facing as a **cross-platform reaction-registry coverage and
leakage audit**, not as source-study generalization of FORGE. Exact reconstruction tests whether
training-derived, role-compatible components passed through the frozen registry recreate a held-out
product. It does not test whether the whole-lipid generator learns shared hydrophobic motifs or whether
a held-out chemistry could be supported after adding its own synthesis adapter.

Baseline to beat: the measured **0.76%** non-AGILE recovery of the current 464,265-product enumeration
(§5).

### 4.4 Source-grounded decomposition adjudication

Running qualified reactions backwards can yield chemically nonsensical disconnections. Coverage and
exact graph round-trip alone are not evidence of experimental chemistry. Admit supervision through a
hash-pinned source-evidence gate that inspects primary articles and supplementary reaction schemes,
records exact procedure locators, verifies structures and reactive sites, and abstains on unresolved
identity or selectivity.

Keep exact source execution, exact series membership, deterministic transform consistency, analogue
precedent, source conflict and missing evidence separate. The source-grounded gate is load-bearing.
The frozen 319-case blinded packet is retained as an optional external audit but does not block exact
Ugi supervision.

---

## 5. Corpus finding (retained) and terminology correction

**Terminology.** The current 464,265-product R1 is renamed from "route-certified" to
**"reaction-enumerated support"** throughout. Nothing is route-certified until L2 and L3 close.

Measured on disk:

| Diagnostic | Result |
|---|---|
| R0 provenance | **Real.** LNPDB 12,837 canonical (of 19,797 rows) + LiON/LNP_ML 9,194 + AGILE-1200; `molecule_enumeration_performed: false` → 15,433 unique measured lipids |
| Block-pool provenance | `programmatic_rational` **390/490 (79.6%)**, `curated_literature` 75, `agile_measured` **25** |
| Block-handle counts | nine handles at **exactly 39 each** — a `chain_length × n_branches × unsaturation` grid, not extraction |
| R0 ∩ R1 exact | 1,308 / 15,433 = **8.5%** |
| …of which AGILE-1200 | **1,200** |
| **Non-AGILE real lipids recovered** | **108 / 14,233 = 0.76%** |

The enumeration reproduces the one library whose blocks were seeded (AGILE) and little else. R0 entered
only as an IPF **reweighting target**, which matches marginals but cannot conjure unreachable structures.

**Viability of the rebuild.** Qualified-family product motifs are present in real lipids:
`amide_coupling` 41.3%, `aza_michael` 36.3%, `epoxide_opening` 6.9%, `iphos` 3.7% of R0.
The 99.8% `reductive_amination` figure is a **degenerate motif** (essentially any C–N bond), is **not
reportable**, and is the motivating example for §4.4.

**Potentially publishable in its own right:** a large combinatorial corpus can appear chemically broad
while having very poor support for the empirical distribution it intends to model.

---

## 6. Product prior: bounded support + feasibility gate

### 6.1 Corrected support statement

Replace *"any graph is expressible"* with:

> **Any graph within the declared bounded atom vocabulary, bond vocabulary, and size support is
> representable.** Vocabulary: C, N, O, S, P (S and P non-optional — disulfide/thioether, iPhos).
> Size support declared per §6.2 outcome.

The defensible contrast with RGFN / SynCoGen / reaction-space GFlowNets stands: there the *action
space itself* is (reaction × building block), so the cross-product is the support by definition.

### 6.2 Phase-1A feasibility gate (before any full-corpus training)

Dense N×N edge flow is not automatically tractable: N=64 gives 4,096 ordered
pairs and N=96 gives 9,216. The current hash-pinned R0 reaches 282 heavy atoms,
not the previously documented 138. Most entries are absent bonds, so the
objective can be dominated by edge-null prediction.

Train small models at **N_max = 64** and **N_max = 96**. The verified full-R0
coverage is 70.48% and 91.23%, respectively. Within the declared C/N/O/S/P
element vocabulary, coverage is 70.87% and 91.71%. Measure:

endpoint validity · connectedness · atom-count distribution fidelity · **bond sparsity calibration** ·
graph reconstruction · memory · throughput · topology novelty · **performance stratified by molecule size**

**Decision rule:** if dense edge flow degrades at 96 atoms, adopt a sparse/hierarchical edge
parameterization that preserves atom-level support. **Do not silently reduce the paper to small
lipids** — the declared size support must be stated in the paper and matched by the panel.

**M0-06 result.** Dense edge cross-entropy is well calibrated for sparsity but
loses bonded-edge recall and generates disconnected endpoints. A bonded-edge
auxiliary restores recall and connectivity but overbonds every endpoint and
produces zero valid molecules. The production prior therefore retains
whole-lipid discrete flow matching but uses a sparse or hierarchical topology
parameterization with full pair reachability, connectivity, and valence-aware
transitions.

**Sparse follow-up.** A canonical spanning-tree plus residual-closure
representation round-trips all 15,089 single-fragment R0 structures within the
declared C/N/O/S/P vocabulary. Bounded N=64 and N=96 probes produced 100% valid,
connected endpoints with no terminal constraint repairs, retained full
topology novelty among 48 samples per size, and passed scaling dry runs through
the observed maximum of 282 heavy atoms. This selects the representation for
later product-prior implementation. It is not a trained product prior. See
`docs/M0_06_DEFOG_FEASIBILITY.md` and
`docs/M0_06_SPARSE_TOPOLOGY_FEASIBILITY.md`.

---

## 7. Biological endpoint — decided before Phase 4, not Phase 7

The software `Endpoint` interface stays generic, but the **paper endpoint, bridge assay, and
candidate-selection rule must be locked before biological guidance**, because the oracle, bridge,
guidance, ranking, formulation, and animal design all depend on it.

**Current lowest-risk option: intramuscular reporter delivery followed by
functional editing; the endpoint remains unlocked.** AGILE directly reports
HeLa-to-IM reporter translation for a selected 15-lipid panel, whereas no
AGILE HeLa-to-IV-liver correlation is available. A compact
platform-validation package can establish IM reporter delivery and distribution
across the in vivo panel, then Cre reporter conversion or reporter-locus or
endogenous editing in one or more leads. Disease correction and exhaustive
mechanism are strengthening experiments, not universal requirements for the
synthesis-grounded platform claim.

**Liver functional editing remains viable only behind an endpoint-specific
gate.** AGILE cannot be treated as a liver oracle. Either a liver-labelled
external corpus must improve non-leaking held-family prediction after assay and
formulation audit, or a prospective primary-hepatocyte or equivalent bridge
must control in vivo advancement.

**IM vaccination remains a credible alternative** given the laboratory's
reported vaccine capability and the alignment of AGILE, JC_2023, and Miao 2019
with IM expression and APC biology. A vaccine claim requires at least one
predeclared antigen-specific immune response, but it does not automatically
require both adaptive arms, long durability, and a challenge study. The
software interface remains generic and no endpoint is hard-coded before the
final lock.

### 7.1 AGILE oracle matrix (predeclared, frozen before guidance)

AGILE supplies 1,200 nominal Ugi-3CR library measurements with A/B/C identities and measured HeLa
and RAW 264.7 transfection. The blocking M0-07 source gate excludes 100 B4 cis/trans mixture
measurements from single-graph supervision and corrects the 100 B5 records to the pure-trans
identity, leaving 1,100 unique single-structure oracle records. Every oracle fit must consume the
reconciled artifact rather than the raw 1,200-row table.
**AGILE is a predictive general-transfection oracle. It is explicitly NOT an in-vivo endpoint oracle.**

| Representation | Models |
|---|---|
| Morgan count fingerprints | ridge, RF, XGBoost, MLP |
| Physicochemical + lipid expert descriptors | ridge, RF, XGBoost, MLP |
| Molecular graph | D-MPNN, GIN/GAT |
| Region-aware head/linker/tail graph | D-MPNN or graph transformer |
| Lipid-pretrained encoder | linear head, MLP, graph transformer |

The matrix includes LANTERN's strong Morgan plus expert-descriptor baselines and a reproduction of
its exact random-split result as a diagnostic. LANTERN's committed file named
`Murcko_scaffold.npy` is not scaffold-disjoint: all six Murcko groups cross
partitions. Preserve it as an audit-only source artifact and never use it for
model selection. The separate scaffold-balanced artifact is scaffold-disjoint
and is the valid structural holdout. Model selection is based on that split
plus deterministic held-head, held-aldehyde, held-isocyanide and component-pair
evaluations. The 12,276 virtual candidates have no wet-lab endpoint labels and
are used only for applicability and distance analysis. **Freeze the best
calibrated model before guided generation.**

The final architecture is selected from calibration metrics only, with equal
weight across endpoints and eligible schemes. Outer-test metrics cannot select
the architecture. In addition, every outer scheme and fold receives a
fold-local nested audit: select a candidate from that fold's calibration rows,
then evaluate it on the untouched test rows. Domain guidance requires both the
fixed candidate and this nested procedure to pass the predeclared gate.
Cross-validated test results are internal development-corpus evidence, while
prospective locked candidates provide the external experimental test.

The completed classical lane shows why the component holdouts are
load-bearing. The best random diagnostic reached R² = 0.635 for HeLa and 0.461
for RAW 264.7, while equal weighting of all seven eligible schemes reduced the
classical-lane leader to 0.215 and 0.087. Entire held aldehydes produced
near-zero best mean R² for both endpoints. These are interim M0 results, not a
frozen oracle. Later representations must improve calibrated component shift
or the guidance policy must attenuate or abstain in those regions.

LANTERN's released HeLa checkpoint reproduces at random-split R² = 0.820 under
its committed pipeline. That pipeline fits a MinMax scaler to all features and
labels before splitting, so the result contains test-information leakage and
is diagnostic only. It cannot select or calibrate the FORGE oracle.

Additional native or related isocyanide-mediated 3-CR datasets are admitted
only through an explicit auxiliary-supervision comparison. Raw assay values
from separate studies are not pooled when their normalization and batch
semantics differ. Compare AGILE-only training with a shared encoder plus
study-specific heads and source-pretraining followed by AGILE fine-tuning.
Retain the auxiliary data only if it improves the frozen AGILE component
holdouts or a declared external-transfer endpoint without using held AGILE
labels.

The completed source audit admits two comparison datasets without pooling
their targets. JC_2023 contributes 288 native-Ugi HeLa products, including
five aldehydes and one source-corrected isocyanide outside AGILE. The
hash-qualified forward transform verifies all corrected JC model identities:
270 retain the source product graph and 18 malformed A3 database products are
rebuilt from the source-defined components under a unique-product gate. LM_2019
contributes 1,080 related-chemistry HeLa products plus 36 BMDC and 12 BMDM
measurements. All auxiliary endpoint values are within-study z-scores. JC_2023
is the primary native-Ugi transfer source. LM_2019 is restricted to
related-chemistry encoder pretraining and external-transfer analysis.

For every frozen AGILE fold, filter auxiliary training records before model
fitting. Exclude exact target products for all splits, held scaffolds for the
scaffold split, held component structures for component splits, and held
component pairs for pair splits. Use a distinct prediction head for every
study-endpoint pair. Internal auxiliary metrics remain transductive diagnostics
because the available LNPDB z-scores were normalized using each complete
study-endpoint population.

For the lipid-pretrained lane, globally remove every R0 constitutional graph
matching any reconciled AGILE target before self-supervision. The frozen gate
removes 1,220 R0 rows spanning all 1,100 target constitutions and
constitutionally deduplicates the remainder to 14,129 graphs. Provenance tags
are not an adequate leakage control because 20 target constitutions occur in
LNPDB-only R0 rows without AGILE provenance. The encoder must derive its
element, charge, bond, and graph-size support from these pinned graphs. The
older training manifest is not authoritative: the actual R0 contains F, Si,
charged structures, and graphs up to 282 atoms. Truncation is prohibited.

The minimum CPU graph matrix is whole-graph D-MPNN, whole-graph edge-aware GIN,
Ugi-component-role-aware D-MPNN, and frozen R0-pretrained encoder heads using
linear and MLP predictors. The role-aware representation encodes the complete
product and the A, B, and C component structures with role embeddings. It does
not use component IDs or invent a linker slot. A train-only runtime profile
must pass before all folds and seeds are scheduled.

The descriptor arm uses only quantities computable consistently from molecular structure unless an
audited measured dataset is available. The current plan does not assume apparent pKa, particle size,
polydispersity, encapsulation efficiency, or formulation robustness labels for every lipid. Do not
impute these formulation-dependent quantities and present them as observations. Applicability must be
estimated from available molecular views, including fingerprints, learned embeddings, component
families, structural clusters, and structure-computable descriptors.

Oracle tilting must remain uncertainty-aware and bounded. Select the uncertainty estimator and any
conservative score only after calibration under component and structural holdouts. Pre-register a
guidance-strength sweep, monitor departure from the labeled applicability domain, and retain
prospective candidates across declared applicability bins. Agreement across independently trained
representations is a robustness check, not a substitute for experimental evidence. Apparent pKa,
particle size, polydispersity, encapsulation efficiency, and formulation robustness enter as
prospective measurements and advancement criteria after synthesis.

Before animal candidate lock, an endpoint-specific bridge is mandatory — hepatocyte expression/editing
for liver; muscle/APC expression plus DC or macrophage assay for vaccine.

---

## 8. The three synthesis layers (organizing principle — retained, Figure 1 core)

| | **L1 — Final assembly** | **L2 — Subcomponent synthesis** | **L3 — Procurement** |
|---|---|---|---|
| What | Ugi-3CR step joining amine + aldehyde + isocyanide | Making heads, tails, linkers, **esters**, isocyanides, heterocycles | Buying terminal leaves |
| Status | **1,200 nominal library executions; 1,100 exact single-compound L1 records; 100 B4 mixture executions excluded from single-graph training; 12,276 virtual products admitted only for transform consistency** | **The open problem** | **Dynamic**, expires |
| Handled by | Deterministic atom-mapped transform + lab protocol | Learned route generation + forward verification + bounded search | Time-stamped snapshot + backup supplier |
| Uncertainty | ≈ constant across candidates | **Dominates candidate variance** | Vendor-driven |

Three consequences: the route model's job is **L2**; a V_syn dominated by L1 is uninformative
(near-constant → no discriminative signal); **route completeness is an L2 statement** — every
non-commercial component must terminate in Tier A/B leaves.

**Report the V_syn variance decomposition.** If L1 terms explain most of the spread, the score is
measuring the wrong thing.

---

## 9. Synthesis value before and after prospective outcomes

Before prospective positive and negative synthesis outcomes exist, FORGE uses an
**evidence-weighted route-completion value**, not a calibrated probability of experimental success.
Raw route-model likelihood is prohibited as the synthesis value because it can reward reaction and
documentation frequency rather than feasibility.

For each route, preserve a structured value record:

```
z_syn(R) = (
  closure state,
  forward-consistency state,
  weakest evidence grade,
  unresolved or unavailable leaves,
  missing-knowledge state,
  route depth,
  protection and purification burden,
  substrate-scope uncertainty
)
```

The pre-prospective scalar used by a sampler, if one is required, must reproduce a frozen monotonic
ranking policy over `z_syn`. Its interpretation is a route-completion value under the declared
evidence policy. It is not `Pr(experimental synthesis succeeds | x)`.

After a sufficiently sized prospective set reports every attempt, including failures, conversion,
isolation, and route deviations, a separate outcome model may be calibrated as:

```
p_success(x) = Pr[correct product is isolated under the declared campaign policy | x]
```

Only that later model may be called a calibrated synthesis-success probability. Headline
computational metrics remain complete-route coverage and prospective correct-product isolation at
matched budgets. Report the route-value feature and L1/L2/L3 variance decomposition so documentation
density cannot hide inside one scalar.

---

## 10. Route model (L2) — corrected gates

RetroSynFlow ([NeurIPS 2025](https://arxiv.org/abs/2506.04439)) is a **single-step** model
(reaction-center → synthon → reactants, FK steering + SMC). Recursive extension into bounded multistep
search is **our extension**, not something the published model guarantees. Stated as such.

**Primary gate (required):** complete bounded route to Tier A/B leaves on held-out products, held-out
precursor scaffolds, and held-out lipid head/tail families, **within known reaction classes**. Metrics:
complete-route rate, exact round-trip, top-k route recall, route depth, uncertainty, precursor
availability.

**Secondary stress test (not a hard gate):** `held_reaction_family` generalization. A model can be
highly useful for lipid synthesis without inventing routes in a family absent from training. The Nature
Biotechnology claim is complete route execution for generated lipids, **not universal zero-shot
retrosynthesis.**

**Hybrid is acceptable and probably preferable:** neural proposal + deterministic template search may
be more reliable than insisting every component is end-to-end learned.

---

## 11. L2 supervision inventory (must precede any L2 implementation)

Before committing to a learned lipid-specific L2 model, **quantify** — do not assume — the corpus:

| Stratum | To quantify |
|---|---|
| USPTO pretraining | reactions, classes, coverage of lipid-relevant chemistry |
| Patent / SI lipid routes | distinct upstream reactions, unique products, unique precursor scaffolds, route depths |
| AGILE tail routes (SI Note 1) | Tail A (acid + diol → EDC → DMP), Tail B (amine → formylation → POCl₃ dehydration), butyl variant |
| Internal RM protocols (RM-006…RM-067) | 7 custom acrylate/propiolate tails, measured conditions |
| **Negative outcomes** | failed syntheses — the highest-value and scarcest stratum |

Report coverage across ester, isocyanide, aldehyde, carbonate, acrylate, and heterocycle formation.
**R1 cannot supply this**: every R1 block is terminal by construction, so R1 contains zero examples of
*making* a block, and its `reactant_ids` labels were produced by the enumerator (circular for L2).

For the current Ugi-3 paper, prioritize reaction families that produce or
diversify amine heads, aldehyde-bearing tails, and isocyanide tails. Broader
lipid chemistry counts toward the primary inventory only when it can feed one
of those precursor roles or its procurement leaves.

Create a versioned **cross-platform hydrophobic-motif transfer registry** for this targeted use.
This is not a database of complete non-Ugi lipid routes. Each record must preserve:

- source lipid, platform, publication, and exact source attachment atoms;
- the transferable hydrophobic subgraph and the platform-specific handle or core that was removed;
- a mapped common precursor, when one is supported;
- a proposed Ugi aldehyde and/or isocyanide realization;
- exact-substrate, close-analogue, or family-precedent evidence;
- recursive route depth, route-closure state, and terminal-material status;
- biological provenance without inheriting the source lipid's activity label.

A motif remains structure-only if it cannot be separated unambiguously from the source core, lacks a
supported reaction-ready attachment point, or does not close within the declared route policy.
Converting a source motif to a new handle changes the molecule, so neither synthesis success nor
biological activity transfers automatically. Prioritize motifs that add branching, unsaturation,
degradability, or other diversity missing from the frozen AGILE components and that can be realized
through short, evidence-supported routes. Accepted entries can seed E2 component proposals, but the
registry itself is proposed transfer space, not observed L2 supervision.

Keep three support axes independent:

1. **G, generative support:** the whole-lipid model can represent or propose the
   structure.
2. **O, oracle applicability:** the biological predictor can score the
   structure under a declared confidence policy.
3. **R, route support:** the synthesis system can close the structure to
   accepted terminal materials.

Tail transfer expands R. It does not improve O and cannot inherit source-lipid
activity labels. A routing miss must be classified as complete, chemically
incompatible, outside declared support, or missing route knowledge. The final
category is an epistemic gap rather than evidence of unsynthesizability.

Use a tail-first transfer policy. Hydrophobic motifs may be converted only when
their source attachment is exact and a separate Ugi-handle route is supported.
An external head may transfer intact only when it is an exact, site-defined
Ugi-compatible amine. Novel heads remain in model support, but they require
independent site qualification and procurement or L2 closure.

LNPDB defines a weighted tail-motif census, not the complete reaction universe.
Prioritize source papers by marginal route-space value: distinctive and
biologically relevant motif classes, novelty relative to AGILE, accessible
component structures and methods, plausible Ugi-handle conversion, and low
redundancy. General chemistry literature, patents, supplier documents, and
close analogues may provide the best route evidence.

For cross-platform transfer, record both the actual source-component reaction
and the separately proposed Ugi-handle reaction. The source record includes
reactants, stoichiometry, conditions, workup, purification, yield, analytical
evidence, and a precise article or supplement locator. A deterministic
structure conversion or successful forward SMARTS application is not an
experimental reaction record.

Stop route mining at operational closure, when missing route knowledge no
longer materially constrains biologically competitive, oracle-supported Ugi-3
candidates within declared support. Measure weighted motif coverage,
candidate-route closure, marginal gain across repeated curation rounds, and
prospective-panel dossier closure. Freeze numerical thresholds before
generator-driven curation and final prospective selection.

Separate two quantities throughout:

1. **Observed supervision:** source-extracted reaction examples, conditions,
   outcomes, and failures.
2. **Reachable virtual space:** unique precursors and whole lipids proposed by
   applying frozen reaction-family rules to versioned input catalogs.

The second quantity must be deduplicated and reported with applicability
filters, attrition, route depth, evidence state, and input hashes. It is not
training data and is not a universal synthesizability claim. The target is
complete transformation-family coverage within a declared bounded Ugi-3
chemical envelope, not one publication route for every possible molecule.

---

## 12. FlowER — small pilot before integration; frozen demotion rule

FlowER is a mass- and electron-conserving **forward mechanism** model, not a retrosynthesis planner.
A prior in-house zero-shot probe (May 2026, `electron_flow_lipids/memos/11_flower_probe_results.md`)
returned 1/6 strict match with aza-Michael failing and Ugi uncovered — a **prior**, not the experiment.

**Pilot scope (M0):** reaction-class holdout + ~32-mechanism/class fine-tuning curve on 2–3 lipid-
relevant classes, calibration. The registry's mechanism-verified atom-mapped positive *and negative*
examples make curation cheap relative to starting cold.

**Frozen demotion rule (set before results):** FlowER stays in the headline mechanism only if
fine-tuning improves held-out forward consistency **and** its score correlates with prospective
conversion (Spearman ρ, CI excluding 0). Otherwise it is reported as an orthogonal consistency check.
**The deterministic atom-mapped verifier is load-bearing either way.**

---

## 13. Milestone M0 — the only authorized scope

Ordered; each item can fail cheaply and independently. No GPU training, no large-scale route model.

| # | Task | Output | Gate |
|---|---|---|---|
| 1 | ~~Ugi 3CR vs 4CR~~ | **DONE: AGILE-type 3CR without a carboxylic-acid reactant component (§2)** | ✅ |
| 2 | Support-tier redefinition | E0–E3 classifier spec + implementation | E1 declared meaningful-or-empty based on whether R1′ is capped |
| 3 | R0 split **before** extraction | 4 holdout schemes from `splits_v1/` | Leak-free verified |
| 4 | Non-circular R1′ audit | `Recovery_heldout` + nearest-reachable distance | Materially exceeds 0.76% baseline |
| 5 | Source evidence adjudication | Hash-pinned source ledger, reaction schemes, exact identities, transform checks and abstentions | Exact versus computed supervision separated; ambiguous claims excluded |
| 6 | DeFoG feasibility gate | N=64 and N=96 small-model runs | Validity, connectedness, sparsity calibration, size-stratified performance |
| 7 | AGILE oracle matrix | 5 representations × models, 4 split types, calibration | Best model frozen before any guidance |
| 8 | Endpoint decision | Three endpoint contracts and evidence package complete; exact program lock deferred | Locked **before Phase 4** |
| 9 | L2 data inventory | Table per §11 | Sufficiency judged before L2 build |
| 10 | FlowER pilot | Holdout + fine-tuning curve, 2–3 classes | Demotion rule applied |

**Return with the full decision package before broad implementation.**

---

## 14. Revised prospective burden (narrowed)

| Stage | Count |
|---|---|
| Primary Ugi synthesis | **24–32** across four matched arms |
| Orthogonal chemistry (aza-Michael / carbonate) | **6–8** — route dossiers, synthesis, formulation, in vitro. **No second animal campaign** |
| In vivo cohort | **4–6**, chosen by pre-registered synthesis/formulation/bridge criteria |
| Round 2 | **Optional**, not a submission prerequisite |

Four matched arms: FORGE route-guided · same product prior + post-hoc filtering · LUCID fixed-topology ·
reaction-space or expert-library control.

### 14.1 Matched-compute coupling experiment (the core computational result)

Six arms, matched on product-generator calls, route-planner calls, verifier calls, wall-clock/GPU
budget, and final candidate count:

1. product prior, no synthesis info · 2. + post-hoc route filtering · 3. + distilled V_syn guidance ·
4. + planner-in-the-loop FK/SMC · 5. reaction-space generator (RGFN) · 6. LUCID historical

**The result must show guidance changes where probability mass is allocated — not merely that more
routes were evaluated.**

---

## 15. Revised five-figure plan

**Fig 1 — Open-ended product–route generation.** LUCID limitation (527 stored topologies, atom/bond
recoloring only); full-graph product prior; **L1/L2/L3 hierarchy**; R0 / R1′ / breadth data; product–route
dossier; E0–E3 tiers.

**Fig 2 — Computational validation and candidate locking.** DeFoG generation + topology novelty;
AGILE representation × model oracle matrix; complete-route prediction; **route-aware guidance vs
post-hoc filtering at matched compute**; E0/E1/E2 open-endedness; FlowER keep-or-demote; locked panel.

**Fig 3 — Prospective synthesis validates route grounding.** *Probably the most important figure.*
All attempted candidates; correct product; conversion; isolated yield; purity; route changes;
**L1/L2/L3 failure attribution**; matched-arm synthesis success.

**Fig 4 — LNP formation and endpoint-specific bridge.** Common formulation; size/PDI/EE/pKa;
formulation failures retained; in vitro potency; endpoint bridge; pre-registered down-selection.

**Fig 5 — Functional in vivo validation.** All selected candidates; editing or vaccine response; dose
response; tolerability; **structurally distinct successful chemotypes**; synthesis success vs biological
function; optional orthogonal-chemistry inset.

Figures 1–2 computational; 3–5 experimental.

---

## 16. Architecture (unchanged in structure, deferred in execution)

FORGE **consumes** the COMPOSE corpus and registry as hash-pinned vendored data. It does **not** fork
`compose_v4` / `compose_rgm` — Paper 3's contribution is synthesis coupling on an established backbone.

```
forge/
├── PREREGISTRATION.md          # frozen before panel lock
├── docs/  PLAN.md  DATA_PROVENANCE.md  FIGURE_MAP.md  DECISION_LOG.md
├── data/vendor/                # hash-pinned (gitignored; manifest tracked)
├── configs/  assembly/ugi_variant.yaml  corpus/  model/  route/  sampling/  panel/
├── src/forge/
│   ├── data/     corpus.py  splits.py  blocks.py  availability.py  breadth.py
│   ├── product/  flow.py  transformer.py  noise.py  inpaint.py  nodecount.py
│   ├── route/    assembly.py(L1)  precursor.py(L2)  search.py  leaves.py(L3)
│   ├── verify/   deterministic.py  flower.py  roundtrip.py  conservation.py
│   ├── value/    completion.py  utility.py  distill.py  uncertainty.py
│   ├── sample/   fk_steering.py  guidance.py  diversity.py
│   ├── dossier/  schema.py  atommap.py  render.py  export.py
│   ├── bio/      endpoint.py  muscle.py  liver.py  vaccine.py  bridge.py  oracle_matrix.py
│   └── eval/     support_tier.py  anchoring.py  coverage.py  funnel.py  calib.py
├── scripts/      one entry point per M0 task and per gate
└── tests/
```

Product prior: DeFoG discrete flow matching
([ICML 2025 oral](https://openreview.net/forum?id=KPRIwWhqAZ),
[code](https://github.com/manuelmlmadeira/DeFoG)) — subject to §6.2. Scaffold **inpainting**, not
templating: LUCID's validated Ugi SMARTS `[NX3][CX3](=[OX1])[CX4]([#6])[NX3]` is reused for
*detection*, never for topology supply. The audited R0 contains potential
tetrahedral stereochemistry in 74.37% of rows and explicitly specified atom or
bond stereochemistry in 31.98%; the older 88.12% manifest value is not
reproducible under a declared definition. The primary graph representation
excludes stereochemistry. Assign it at dossier time from the route's declared
stereochemistry policy and report stereochemical burden as purification risk.

Baselines: LUCID · DeFoG unconditional · GenMol/SAFE · RGFN · SynCoGen
([arXiv 2507.11818](https://arxiv.org/pdf/2507.11818)) · AiZynthFinder post-hoc filtering.
SynLaD ([arXiv 2607.01105](https://arxiv.org/pdf/2607.01105)) cited as inspiration, not replicated.

### 16.1 Hierarchically joint factorization is deliberate

FORGE models a joint molecule and synthesis-program distribution through a chemically structured
factorization:

`p(x, b, R) = p_theta(x, b) q_phi(R | x, b)`,

where `x` is the complete generated lipid graph, `b` is its L1 final-assembly decomposition, and `R`
is the recursive L2/L3 synthesis program. The product and L1 decomposition are generated jointly.
The L2 system then routes each nonterminal component, and the L3 system closes terminal materials.
This hierarchy does not mean that synthesis is checked only after generation. An evidence-weighted
synthesis value derived from recursive route completion changes molecular transition probabilities
before candidate lock. Generation followed by route assessment without feedback is the post-hoc
baseline, regardless of how sophisticated that downstream planner is.

A single monolithic decoder is not the primary architecture. Current supervision is dense for lipid
structures and Ugi product-component pairs but sparse for exact, complete upstream route trees.
Forcing all outputs through one decoder would waste structure-only data and could entangle the broad
lipid distribution with the narrowest route-labeled chemistry. That creates a specific collapse risk:
the model may overproduce familiar components, short routes, and dominant reaction families while
appearing highly synthesizable.

The hierarchical design keeps the whole-lipid prior broad and makes synthesis pressure observable and
tunable. It is therefore the more rigorous first implementation, not a reduced version of joint
generation.

### 16.2 Region-aware Ugi conditioning preserves whole-molecule generation

The broad discrete-flow backbone remains region-agnostic so that structure-only
lipids from different synthesis platforms can train it. The Ugi synthesis
program adds a lightweight conditioning adapter rather than replacing the
backbone with independent head and tail generators.

For the Ugi-conditioned state, preserve the reaction-defining core and annotate
or jointly represent:

- the product core;
- amine-derived, aldehyde-derived, and isocyanide-derived atom origins;
- the attachment atom for each noncore region;
- graph distance from each generated atom to its attachment or the protected
  core.

Use shared whole-graph message passing with small role-conditioned adapters or
gates. Normalize node and bond losses by region so the larger hydrophobic
regions do not overwhelm the smaller nitrogen-rich head. Keep a global graph
state and cross-region message passing at every step. Branching, unsaturation,
rings, heteroatoms, and degradable motifs remain atom and bond states, not
hand-assigned potency labels.

This is still one connected whole-molecule flow. All regions evolve
simultaneously, and the model generates their atoms, bonds, topology, and
cross-region compatibility jointly. Region labels provide synthesis-program
context; they do not select component IDs or enumerate independently generated
fragments. Compare a region-agnostic flow, late role conditioning, and
role-conditioned message passing with attachment-relative positions. A fully
separate head or tail generator is an overfitting and enumeration control, not
the primary architecture.

Broad support must also be protected during joint L1 training. Use a masked multi-source objective:

```
L = L_lipid(x) + m_b lambda_b L_L1(b | x) + m_b lambda_rt L_roundtrip(x, b)
```

`L_lipid` applies to every eligible broad-corpus lipid. The L1 and round-trip losses apply only when
a source-adjudicated exact or explicitly labeled transform-consistency decomposition exists. Joint
training must use a predeclared broad-corpus
replay ratio or simultaneous source-balanced batches. Fine-tuning only on the 12,276 Ugi products is
not allowed. Monitor broad-corpus validation loss, chemical-space coverage, and motif or topology
recall for forgetting.

The coupling experiment must pre-register:

- a synthesis-guidance-strength sweep, including zero guidance;
- route closure, biological score, structural novelty, internal diversity, broad-corpus coverage,
  component novelty, effective component count, and concentration on common AGILE components;
- route closure stratified into familiar AGILE components, transferred known components, and
  genuinely generated components;
- the same product prior with post-hoc route filtering;
- matched product-model, route-planner, verifier, and final-candidate budgets;
- calibration and abstention outside the route model's applicability domain;
- at least one guidance schedule or route-marginalization ablation that tests whether early, strong
  synthesis pressure unnecessarily narrows molecular exploration.

Choose the final guidance strength by maximizing route closure subject to validation-set floors on
diversity, broad-distribution coverage, and component novelty. Freeze those floors before final
prospective generation. A setting that improves closure by collapsing onto familiar AGILE
components fails.

The editor-facing title and opening remain **Synthesis-grounded generative design of ionizable
lipids**. "Hierarchically joint" belongs in Figure 1, Results, and Methods as the technical
factorization. It is supporting rigor, not the title-level claim.

### 16.3 Ugi-first execution order

Phase 1 does not wait for a finished universal lipid generator. After the support-skeleton and exact
atom-origin gates pass, train a bounded Ugi-conditioned whole-lipid model from scratch. Launch the
identical backbone pretrained on broad observed lipids and specialized to Ugi only if the frozen
Ugi-first decision gate identifies a representation, sample-efficiency or held-component gap that
broad structure-only data could plausibly close. Broad pretraining is a decision-gated empirical
ablation, not a prerequisite that blocks the first working generator.

The complete source hierarchy, adapter boundary, component-disjoint splits, checkpoint sampling and
open-endedness gates are frozen in `docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md`. This change affects
execution order, not platform framing: FORGE remains the reusable synthesis-program framework and
Ugi-3 remains the first deeply trained and prospectively validated instantiation.

---

## 17. Risks

0. **Layer conflation** — reporting L1 coverage as "route coverage." Mitigated by L2 completion rate,
   V_syn variance decomposition, layer-attributed failure taxonomy.
1. **R1′ circularity** — resolved by §4 split-before-extraction.
2. **Decomposition looks good but is chemically wrong** — resolved by §4.4 source-evidence
   adjudication and mandatory abstention.
3. **Dense edge flow does not scale** — resolved by §6.2 gate; fallback is sparse parameterization,
   never silent size reduction.
4. **L2 supervision too thin** — §11 inventory runs before implementation; fallback is hybrid
   neural-proposal + deterministic template search.
5. **Guidance = filtering** — §14.1 is designed to be able to fail.
5a. **Guidance collapses exploration** — §16.1 requires a guidance-strength Pareto analysis and E2
    monitoring. Do not select guidance solely by route-closure rate.
6. **FlowER** — frozen demotion rule; deterministic verifier load-bearing.
7. **E2 fraction near zero** — the open-endedness claim is not earned; report it rather than redefine
   the metric.
8. **Ugi panel capacity** — 24–32 must be confirmed against lab throughput before panel lock.

---

## 18. Open items requiring a PI decision

- **Endpoint** — exact selection is deferred while M0 computation continues.
  IM delivery followed by functional editing is currently the lowest-risk
  option. The exact target, bridge assay, capacity, benchmark, and formulation
  require explicit lock before Phase 4. Liver requires a liver-specific data or
  bridge gate. IM vaccination remains viable if the established laboratory
  workflow is operationally preferable.
- **Ugi-4CR flag** — if the lab truly runs 4CR, AGILE labels do not supervise it (§2). Scope change.
- **Ugi panel capacity** at 24–32 candidates.
- **Reaxys/CAS/Pistachio licensing** — determines machine-readable vs hand-curated Tier-C evidence.
