# FORGE Nature Biotechnology drafting analysis

Status: working editorial analysis for a pre-results manuscript. This document
guides emphasis and structure. It is not itself a source of scientific results.

## Editorial objective

The primary reader is an RNA-delivery, lipid nanoparticle (LNP), or biomaterials
scientist. The manuscript must first establish why synthesis limits the
discovery of ionizable lipids, what experimental capability FORGE adds, and
whether the resulting materials can be made and function in LNPs. Machine
learning rigor remains central, but it should enter the narrative when it
answers one of those questions.

The paper identity is:

> FORGE generates complete ionizable-lipid structures together with auditable
> synthesis programs and prospectively tests whether those designs can be made,
> formulated, and used for functional RNA delivery.

The Ugi three-component reaction is the first deeply trained and prospectively
validated synthesis-program instantiation. It is not the title-level identity
or evidence that every lipid chemistry has already been learned.

## Journal constraints

Nature Biotechnology currently describes an Article as having an unreferenced
abstract of at most 150 words, up to 3,000 words of main text excluding Methods,
references and legends, and up to six display items. The introduction is
unheaded. Results and Online Methods may have topical subheadings; Discussion
does not. Initial submission is format-flexible, but the working document will
use double spacing, continuous line numbers, one-inch margins and page numbers
to support review.

Primary journal guidance:

- [Nature Biotechnology content types](https://www.nature.com/nbt/content)
- [Nature Biotechnology writing and language](https://www.nature.com/nbt/submission-guidelines/writing-and-language)
- [Nature Biotechnology initial submission formatting](https://www.nature.com/nbt/submission-guidelines/initial-formatting)

## What the supplied Homological Flows PDF contributes

The supplied PDF is useful as a visual and organizational reference, not as a
Nature Biotechnology rhetorical template. Useful features include a restrained
title treatment, a compact abstract block, clear section hierarchy, figures
placed near their first discussion and captions that define the object being
shown. The manuscript should not inherit its single-spaced ML-conference layout,
numbered main sections, mathematical density or anonymous-review styling.

The FORGE working document will instead combine:

- a clean title and author block;
- a compact abstract;
- an unheaded introduction;
- finding-led Results subheadings;
- visible figure and result placeholders;
- conventional Nature-style references;
- double-spaced body text, continuous line numbers and page numbers.

## Cross-paper editorial analysis

| Paper | Opening move | ML density in editor-facing prose | Experimental evidence emphasized | What transfers to FORGE |
|---|---|---|---|---|
| Witten et al., LiON, Nature Biotechnology (2025) | Pulmonary delivery bottleneck | One accessible architecture sentence | Search scale, mouse delivery, nebulization and ferret lung delivery | Lead with delivery consequence and let the experimental arc carry the paper |
| Zhou et al., MOLEA, Nature Biotechnology (2026) | Tissue selectivity and off-target delivery | One concise multiobjective-model sentence | Tissue-selective expression, genome editing and disease-relevant function | Define the model by what it enables rather than by its optimization machinery |
| Chen et al., PepMLM, Nature Biotechnology (2026) | Need for target-conditioned peptide binders | Technical mechanism compressed in abstract and introduction | Binding, degradation and disease-relevant applications | Move exact loss and sampling mathematics to Methods while retaining stringent held-out tests |
| Zhang et al., PeptiVerse, Nature Communications (2026) | Translational properties missing from existing prediction tools | No displayed mathematical development in the main narrative | Dataset coverage, similarity-aware splits and property-specific model comparisons | Put data limitations before architectural novelty and let each prediction task select its most defensible model |
| Xu et al., AGILE, Nature Communications (2024) | Need to accelerate LNP screening | Predictor described succinctly | A measured Ugi library, virtual ranking, synthesis and delivery assays | Treat AGILE as a strong aligned predictor and data source, not as a weak strawman |
| Xu et al., LUMI-lab, Cell (2026) | Data scarcity and autonomous molecular discovery | Foundation model and active learning described as parts of a closed loop | More than 1,700 synthesized LNPs, ten experimental cycles and in vivo lung editing | Acknowledge that large-scale AI-guided Ugi discovery is already experimentally mature; distinguish FORGE through open-ended product-plus-route generation, not automation scale |
| Maganti et al., synthesis-constrained discrete diffusion, ICLR GEM workshop (2026) | Enumeration limits ionizable-lipid generation | Scaffold-constrained graph diffusion is the central method | Computational generation with a preserved Ugi scaffold | Position as the historical fixed-topology generative baseline; FORGE must add generated topology and recursive upstream route closure |
| Ou et al., synthesis-DAG lipid generator, NeurIPS AIDrugX workshop (2024) | Synthesizability of generated lipids | Synthesis-DAG generation is central | Computational product and pathway generation from accessible blocks | Avoid claiming that FORGE is the first lipid generator with synthesis information; emphasize complete component routing and prospective execution |

## Prose decisions from close reading

The comparison set supports a specific writing discipline rather than a house
style to imitate.

1. **Open with the therapeutic constraint.** LiON opens with ionizable lipids
   as a bottleneck in nonviral mRNA delivery; MOLEA opens with tissue
   selectivity and off-target toxicity; PepMLM opens with the inaccessible
   target space. FORGE should open with the experimental burden of discovering
   ionizable lipids for RNA delivery, not with graph generation or
   retrosynthesis.
2. **Give the model one job per sentence.** Editor-facing prose should define
   FORGE as designing a complete lipid together with its precursor program.
   Sparse flow matching, morphology conditioning, recursive search and
   evidence schemas should not be stacked into the same opening claim.
3. **Prefer experimental verbs to platform adjectives.** Use *generated*,
   *routed*, *synthesized*, *isolated*, *formulated*, *measured* and
   *delivered*. Avoid relying on *advanced*, *powerful*, *comprehensive*,
   *synergistic* or *transformative* to carry a claim.
4. **Move rapidly from method to consequence.** PepMLM devotes one abstract
   sentence to the masking strategy and then reports experimental binding and
   degradation. LiON and MOLEA similarly let delivery, editing and tissue
   results carry the paper. FORGE should let complete route dossiers,
   correct-product isolation, LNP formation and functional delivery carry the
   narrative once the computational method is identified.
5. **Use numbers as evidence, not decoration.** Place search scale, attempted
   candidates, correct products, route-complete products and biological hits
   in sentences that answer a defined question. Do not front-load architecture
   counts that an editor cannot yet interpret.
6. **Make claim boundaries visible in the prose.** PeptiVerse explicitly
   relates model performance to dataset coverage. FORGE should likewise state
   when a result establishes exact graph reconstruction, route evidence or
   current procurement, and when it does not establish experimental synthesis
   success.
7. **Keep the biological vocabulary lipid-native.** The evidence chain is
   precursor preparation, final-lipid identity and purity, LNP formulation,
   measured particle quality, an endpoint-relevant cellular bridge and
   functional RNA delivery. Protein-language concepts are useful for pacing,
   not as analogies for the biology.

The resulting paragraph rhythm is:

> biological need -> current capability -> unresolved experimental gap ->
> FORGE capability -> measured evidence.

This rhythm should organize the abstract, introduction and first sentence of
each Results section. It should not force every section into identical length
or syntax.

### PeptiVerse: what to borrow and what not to borrow

PeptiVerse is especially useful because it refuses to assume that the most
complex downstream architecture is always best. It begins Results with dataset
composition, uses modality-appropriate similarity splits, compares classical
and neural predictors and states when representation or experimental coverage
is more limiting than model capacity. FORGE should apply the same discipline to
the biological oracle matrix, held-component splits, calibrated abstention and
the sparse L2 evidence base.

The peptide subject matter does not transfer. FORGE should not borrow peptide
developability endpoints, sequence language or binding-centric validation. Its
native evidence chain is precursor preparation, correct-product isolation,
conversion, isolated yield, purity, route deviation, LNP formation, an
endpoint-relevant cellular bridge and functional in vivo RNA delivery.

### Bowen Li's work: what to borrow and what not to borrow

Bowen Li's AI-LNP papers are the closest guide to reader expectations. Their
editor-facing prose moves rapidly from the delivery bottleneck to the model and
then to synthesis, formulation and in vivo function. FORGE should use this
pacing. It should not pattern-match their experimental endpoint, chemistry or
scale. FORGE's biological endpoint must follow the final payload, tissue and
administration route selected for this study.

LUMI-lab also raises the competitive bar. It pretrained on more than 28 million
molecules, enumerated a large four-component Ugi lipid-like space and used a
self-driving laboratory to synthesize and screen more than 1,700 LNPs. That
work makes it untenable to frame FORGE merely as AI-guided Ugi lipid discovery.
The manuscript must instead show what LUMI-lab and finite-library active
learning do not establish: generation beyond a frozen component universe,
recursive preparation of every nonterminal component, an evidence-bearing
route dossier and a causal comparison of route-guided sampling with the same
generator followed by post-hoc route assessment.

## Competitive claim boundary

Do not use any of the following claims:

- first AI platform for ionizable-lipid discovery;
- first generative model for ionizable lipids;
- first synthesis-aware lipid generator;
- synthesis across all ionizable-lipid chemistries;
- experimentally validated route grounding outside the Ugi instantiation;
- route-certified for reaction-enumerated products.

The strongest defensible identity is:

> FORGE is a synthesis-grounded whole-lipid generative framework that jointly
> produces a complete product and its final-assembly decomposition, recursively
> routes noncommercial components to supported terminal materials and uses the
> resulting synthesis value to influence molecular sampling. The framework is
> prospectively instantiated here in AGILE-type amine-aldehyde-isocyanide Ugi
> three-component chemistry.

The word "jointly" should be used with the hierarchy made explicit. Dense data
support joint product and final-assembly generation. Sparse upstream data
support recursive, hybrid L2/L3 routing. This is a deliberate factorization,
not post-hoc synthesizability checking.

## Recommended title strategy

Pre-results working title:

> **Synthesis-grounded generative design of ionizable lipids**

Endpoint-dependent candidate:

> **Synthesis-grounded generative design of ionizable lipids for mRNA delivery**

The suffix "for mRNA delivery" is attractive because it tells a biotechnology
reader immediately why the work matters. It should be frozen only after the
prospective payload and functional delivery evidence are fixed. If the final
campaign is broader than mRNA or the functional evidence remains preliminary,
retain the shorter title.

## Main-text architecture

### Abstract, at most 150 words

Use five moves:

1. Ionizable-lipid discovery limits RNA-delivery development.
2. Finite-library prediction restricts search, while unconstrained generation
   can yield designs whose precursors cannot be executed.
3. FORGE generates complete lipids together with auditable recursive synthesis
   programs.
4. State prospective synthesis, formulation and functional results with exact
   values only after data freeze.
5. State the bounded implication for actionable RNA-delivery materials.

The abstract should contain no equations, architecture acronyms, flow-matching
details, evidence-tier terminology or route-planner names.

### Unheaded introduction

Use five compact paragraphs:

1. RNA medicines depend on delivery, and the ionizable lipid is a major
   determinant of LNP behavior.
2. AI and high-throughput chemistry have accelerated lipid selection, including
   large-scale autonomous Ugi discovery, but most systems still rank or explore
   a chemistry-defined candidate universe.
3. Open-ended molecular generation creates the complementary problem of
   experimental executability. A familiar final assembly does not make a novel
   aldehyde, isocyanide or head accessible.
4. Define complete synthesis grounding as final assembly plus recursive routes
   for every noncommercial component, with evidence and procurement separated
   from actual prospective success.
5. Introduce FORGE, the Ugi three-component instantiation, the controls and the
   complete experimental evidence chain.

### Results

Results subheadings should state findings rather than implementation tasks.
The working sequence is:

1. **FORGE generates complete lipid and synthesis-program dossiers.**
2. **A Ugi-aligned corpus supports joint product and final-assembly generation.**
3. **FORGE explores route-complete chemistry beyond the frozen component library.**
4. **Route information changes molecular sampling at matched computational effort.**
5. **Prospective synthesis tests every component and final assembly.**
6. **Generated lipids form LNPs and support functional RNA delivery.**

Stable corpus and audit findings can be written now. Generator performance,
guidance advantages, candidate counts, yields, formulation metrics and biology
remain explicit placeholders until their artifacts or prospective datasets are
frozen.

### Discussion

Open with the experimentally established advance, not a restatement of the
architecture. Then address:

- why complete upstream routing matters beyond preserving a reaction scaffold;
- whether route guidance outperformed post-hoc assessment, with a null result
  narrowing only that mechanistic claim;
- how open-endedness was demonstrated beyond the finite component catalog;
- the distinction between route evidence and experimental success;
- the Ugi-specific evidence boundary and requirements for new adapters;
- limitations of the current in vitro oracle and selected in vivo endpoint.

Do not use subheadings in Discussion.

## Figure logic

### Figure 1: What FORGE produces

Show the complete lipid graph, exact Ugi L1 decomposition, recursive L2 route
forest, terminal-material evidence, forward checks, uncertainty and one final
product-route dossier. Keep the model diagram legible to a chemist.

### Figure 2: Computational validation

Show whole-lipid validity and diversity, held-component generalization,
beyond-catalog generation, route closure and the matched same-generator
route-guided versus post-hoc ablation. Include finite-library and historical
fixed-topology comparisons without making them the figure identity.

### Figure 3: Prospective synthesis

Show every attempted candidate and every failure. Report precursor success,
final product, conversion, isolated yield, purity, route changes and L1/L2/L3
failure attribution across matched candidate arms.

### Figure 4: LNP formation and biological bridge

Show formulation success and only measured particle properties. Then show the
endpoint-relevant cellular bridge used for down-selection.

### Figure 5: Functional in vivo evidence

Show the frozen endpoint, all advanced candidates, dose and comparator logic,
tolerability, and structurally distinct successful chemotypes. Do not infer
tissue targeting from HeLa or RAW 264.7 transfection labels.

## Mathematical density

No displayed equations are needed in the abstract or introduction. One compact
factorization may appear in Results only if it materially clarifies jointness:

`p(x, b, R) = p_theta(x, b) q_phi(R | x, b)`

The text must immediately define the complete lipid `x`, the exact final
assembly decomposition `b` and the recursive upstream program `R`. The
synthesis-guidance law, discrete-flow objective, sparse morphology program and
planner details belong in Online Methods or a Supplementary Note.

## Result-writing rules

- Lead each Results subsection with the finding.
- Give denominator, split and uncertainty with each headline metric.
- Distinguish observed synthesis from exact forward enumeration.
- Call the 1.49-million product universe reaction-enumerated support, not a
  synthesized or route-certified library.
- Treat the same-generator post-hoc comparison as a causal ablation.
- Preserve negative outcomes, abstentions and prospective failures.
- Do not impute particle size, polydispersity, encapsulation or apparent pKa.
- Do not present HeLa or RAW 264.7 predictions as evidence of an in vivo
  endpoint.
- Do not interpret generated novelty as open-ended synthesis until a novel
  component has complete L2/L3 closure and is prospectively attempted.

## Immediate drafting decision

Proceed with a full pre-results manuscript rather than isolated paragraphs. The
document should contain polished abstract, introduction, stable computational
Results, provisional Results paragraphs with conspicuous placeholders,
Discussion, Online Methods, figure legends and references. This gives each new
result an intended location while preventing unfinished experiments from being
written as completed findings.
