# FORGE manuscript editorial guide

## Purpose

This guide fixes the working editorial logic for the Nature Biotechnology
manuscript. It is not a prose template and should not be used to imitate any
one paper. It extracts decisions about emphasis, mathematical density,
biological framing, and the order of evidence from relevant primary articles.

Target title:

> Synthesis-grounded generative design of ionizable lipids

Endpoint-dependent title candidates, to be frozen only after prospective evidence is available:

> Synthesis-grounded generative design of ionizable lipids for mRNA delivery

> Synthesis-grounded generative design of ionizable lipids for RNA delivery

The editor-facing identity is:

> FORGE generates complete ionizable lipids together with auditable synthesis
> programs and prospectively tests whether those designs can be made,
> formulated, and used for functional RNA delivery.

## Journal constraints

Nature Biotechnology states that an Article has:

- up to 3,000 words of main text, excluding the abstract, Methods,
  references, and legends;
- an unreferenced abstract of at most 150 words;
- up to six display items;
- an unheaded introduction, followed by Results, Discussion, and Online
  Methods;
- topical subheadings in Results and Online Methods, but no subheadings in
  Discussion;
- typically up to 50 references.

Initial submissions do not require special formatting and may be submitted as
Word, PDF, or TeX/LaTeX. The working manuscript nevertheless uses double
spacing, continuous line numbers, one-inch margins, and page numbers because
that format is easy to review.

Sources accessed 30 July 2026:

- Nature Biotechnology, [Formatting your initial
  submission](https://www.nature.com/nbt/submission-guidelines/initial-formatting)
- Nature Biotechnology, [Content
  types](https://www.nature.com/nbt/content)
- Nature Biotechnology, [Writing and
  language](https://www.nature.com/nbt/submission-guidelines/writing-and-language)

## What the comparison papers teach

### PepMLM: technically serious, experimentally led

Chen et al., *Nature Biotechnology* (2026),
doi:10.1038/s41587-025-02761-2:

- The abstract states the therapeutic limitation first, introduces the model
  in one sentence, gives one sentence of computational validation, and spends
  the remaining space on experimental binding, degradation, disease targets,
  and broader utility.
- Figure 1 combines architecture and computational evaluation.
- Figures 2 through 4 are experimental and move from binding to cellular
  function and disease-relevant applications.
- The Results explain training and sampling in plain language. The displayed
  loss and pseudo-perplexity equations appear in Methods, not in the abstract
  or opening narrative.
- The prose repeatedly translates a model property into a biological
  capability, such as designing without structural input, instead of treating
  an optimization metric as the endpoint.

FORGE should use the same discipline. The corresponding capability is not
merely a lower route score. It is the ability to deliver a new lipid, its
precursor tree, and an experimentally actionable dossier.

### PeptiVerse: data limitations before architectural novelty

Zhang et al., *Nature Communications* (2026),
doi:10.1038/s41467-026-74167-w:

- The introduction frames prediction as a developability bottleneck.
- The first Results section begins with dataset composition and experimental
  heterogeneity.
- The study compares multiple representations and model classes instead of
  assuming that the deepest model must win.
- The prose explicitly concludes that data coverage and representation can be
  more limiting than downstream model capacity.
- The main text contains no displayed mathematical development.

This supports FORGE's representation by model oracle matrix, held-component
evaluation, calibrated abstention, and willingness to retain a simple oracle
if it is the most defensible model.

### PepTune: a source for technical exposition, not journal pacing

Tang et al., *Proceedings of the 42nd International Conference on Machine
Learning* (2025), "PepTune: De Novo Generation of Therapeutic Peptides with
Multi-Objective-Guided Discrete Diffusion":

- The paper provides detailed discrete-generation and multiobjective-guidance
  mathematics appropriate to an ML venue.
- Its value for FORGE is the precise treatment of guided discrete generation,
  Pareto objectives, and search.
- Its mathematical density should inform Online Methods and Supplementary
  Notes, not the Nature Biotechnology abstract or introduction.

### LiON and Bowen Li's LNP studies: delivery consequence first

Witten et al., *Nature Biotechnology* (2025),
doi:10.1038/s41587-024-02490-y:

- The abstract begins with ionizable lipids as a delivery bottleneck.
- The directed message-passing architecture receives one sentence.
- Search scale, selected lipids, mouse delivery, nebulization, and ferret lung
  delivery carry the result.

Li et al., *Nature Materials* (2024),
doi:10.1038/s41563-024-01867-3:

- The abstract moves from the mRNA-delivery bottleneck to a combinatorial
  library, model selection, a 40,000-lipid virtual screen, 16 synthesized
  candidates, and one experimentally effective lipid.
- The figure sequence moves from chemistry to screening, model training, and
  experimental validation.

Zhou et al., *Nature Biotechnology* (2026),
doi:10.1038/s41587-026-03109-0:

- The abstract opens with tissue selectivity and off-target toxicity, not the
  neural architecture.
- One sentence defines the AI system.
- The rest establishes in vivo selectivity, genome editing, and disease
  modification.
- Figure 1 introduces the system, Figure 2 establishes screening and
  prediction, and Figures 3 through 6 establish biological performance.

For FORGE, the lipid-native evidence chain is:

> generated molecular graph, complete precursor program, precursor
> preparation, final-lipid isolation, formulation, endpoint-relevant bridge,
> and functional in vivo delivery.

Protein binding and degradation assays are not structural analogues for this
chain. They are useful only as examples of how a paper moves quickly from a
model to experimentally meaningful capability.

### What transfers from protein papers, and what must remain lipid-native

Borrow from Chatterjee-lab papers:

- narrative compression around the model;
- explicit separation of training, computational validation, and prospective
  testing;
- strong held-out evaluation and calibrated claims;
- figure pacing that moves quickly from method to experimentally meaningful
  capability;
- exact mathematical detail in Methods rather than editor-facing prose.

Do not transfer protein-specific notions of developability, binding,
degradation, sequence fitness, or therapeutic validation. FORGE must instead
earn its claims through the ionizable-lipid and LNP evidence chain:

- precursor preparation and final-lipid identity;
- conversion, isolated yield, purity, and route deviations;
- formulation success under a predeclared formulation policy;
- measured particle size, polydispersity, encapsulation efficiency, and
  apparent pKa only where those measurements are actually collected;
- an endpoint-relevant cellular bridge;
- organ exposure or expression when relevant to the stated delivery claim;
- a functional in vivo endpoint and tolerability appropriate to the payload,
  route of administration, and target tissue.

The bridge and in vivo experiment must be selected for the intended delivery
claim. HeLa or RAW 264.7 transfection can support general in vitro delivery
guidance within the aligned Ugi domain, but it cannot by itself establish
liver tropism, muscle delivery, vaccination, or in vivo editing. The
manuscript should therefore keep the final biological endpoint provisional
until the payload, administration route, tissue target, and bridge assay are
frozen.

### AGILE: aligned data, distinct task

Xu et al., *Nature Communications* (2024),
doi:10.1038/s41467-024-50619-z:

- AGILE combines a virtual Ugi lipid library, graph pretraining, 1,200 measured
  lipids, prediction, synthesis, formulation, and biological evaluation.
- Its task is prediction and ranking within a constructed candidate library.
- It supplies aligned Ugi chemistry and HeLa and RAW 264.7 measurements, but
  it does not establish an oracle for an arbitrary in vivo endpoint.

FORGE should describe AGILE as a strong aligned predictor and baseline, not as
a deficient generator.

## Locked prose architecture

### Title

Keep the reaction family out of the title:

> Synthesis-grounded generative design of ionizable lipids

Ugi chemistry is the prospective experimental instantiation, not the platform
identity.

### Abstract

Use five moves in at most 150 words:

1. RNA-delivery need and ionizable-lipid discovery bottleneck.
2. Limitation of finite-library prediction and uncertain accessibility of
   open-ended designs.
3. One accessible sentence defining FORGE.
4. Prospective synthesis, formulation, and functional results, with exact
   values inserted only after candidate lock and data freeze.
5. A bounded significance statement.

Do not include equations, model hyperparameters, FlowER, Retro SynFlow,
guidance coefficients, or evidence-tier nomenclature.

### Unheaded introduction

Aim for five compact paragraphs:

1. RNA medicines depend on delivery, and ionizable-lipid structure is a major
   determinant of LNP function.
2. AI and combinatorial chemistry can prioritize candidates, but existing
   systems usually begin from an experimentally or virtually enumerated
   universe.
3. Open-ended generation expands molecular exploration but can propose
   components with uncertain upstream accessibility. Restricting generation
   to purchasable blocks creates the opposite failure.
4. Define complete synthesis grounding in accessible language: final assembly
   plus recursive preparation of every noncommercial head, tail, or
   linker-like motif to supported terminal materials.
5. Introduce FORGE, the Ugi three-component prospective setting, the matched
   causal controls, and the experimental evidence chain.

There should be no displayed equations in the introduction.

### Results

The first two Results sections carry computational rigor without allowing it
to dominate the paper:

1. **FORGE couples whole-lipid generation to complete synthesis programs.**
   Define the complete product and route dossier, hierarchical joint
   architecture, reaction-program representation, and how route value changes
   sampling before candidate lock.
2. **Data-aligned supervision supports joint assembly generation and recursive
   precursor routing.** Establish the broad structure corpus, reviewed Ugi
   product-component pairs, sparse L2 evidence, source provenance, and why the
   hybrid factorization is deliberate.
3. **Biological and synthesis models are evaluated under prospective chemical
   shifts.** Present the oracle matrix, held-component and held-pair results,
   uncertainty, applicability policy, route evaluation, and abstention.
4. **Synthesis guidance expands route-complete design beyond the component
   library.** Compare zero guidance, post-hoc assessment, and coupled
   generation while preserving diversity and component novelty.
5. **Prospective synthesis tests complete FORGE dossiers.**
6. **Generated lipids form LNPs and support endpoint-relevant RNA delivery.**

Sections 5 and 6 may be divided further after the prospective study.

### Mathematical placement

The main Results should explain the factorization and guidance in words and in
Figure 1. If an equation materially improves precision, use at most one short
factorization in the main text.

Place the following in Online Methods or a Supplementary Note:

- the joint product and synthesis-program factorization;
- the discrete flow-matching objective;
- the evidence-weighted synthesis value;
- the sampling guidance law;
- oracle uncertainty and applicability definitions;
- matched-budget accounting.

The paper should contain enough mathematics for exact reproduction, but the
editor should not need an equation to understand why the platform matters.

## Lipid-native biological language

- Separate the ionizable-lipid molecular graph from the formulated LNP.
- Treat apparent pKa, particle size, polydispersity, encapsulation efficiency,
  and formulation robustness as measured downstream properties unless labels
  actually exist for the relevant dataset.
- Do not impute absent formulation measurements to the broad structural
  corpus.
- Describe the AGILE HeLa and RAW 264.7 models as aligned in vitro
  transfection oracles. Do not call them direct predictors of liver, muscle,
  vaccination, editing, or another in vivo endpoint.
- Keep the final endpoint described as functional RNA delivery until the
  bridge assay and animal design are frozen.
- Report every attempted prospective synthesis and separate precursor
  failure, final assembly failure, isolation, purity, formulation, and
  biological failure.
- Avoid treating source-lipid activity as a label for a transferred tail or a
  newly generated Ugi product.

## Chemistry language

- Use "AGILE-type amine-aldehyde-isocyanide Ugi three-component chemistry" on
  first mention.
- Do not call it simply acid-free because the reported procedure uses an
  acidic catalyst.
- Do not introduce an artificial linker reactant. Linker-like and degradable
  motifs may be embedded in generated components and must be closed through
  upstream routing.
- State clearly that whole-lipid generation is not head by tail enumeration.
- Describe complete routes as computationally route-complete under the
  declared evidence policy until prospective execution supports stronger
  language.

## Sentence-level style

- Lead paragraphs with a biological or experimental claim, then explain the
  computational mechanism that enables it.
- Prefer concrete verbs: generate, construct, synthesize, isolate, formulate,
  deliver, edit, and verify.
- Define one term once and reuse it consistently.
- Use "artificial intelligence" sparingly. Name the actual capability.
- Avoid promotional adjectives such as transformative, revolutionary, vast,
  and unprecedented unless the result supplies a quantitative comparison.
- Do not use em dashes.
- Do not report placeholder results as facts.
- Keep the distinction among observed evidence, computed results, planned
  experiments, and interpretation explicit.

## Figure allocation

The working five-figure story is:

1. FORGE architecture and the complete product-route dossier.
2. Computational validity, oracle calibration, open-endedness, and the causal
   guidance ablation.
3. Prospective precursor and final-lipid synthesis, including every attempt.
4. LNP formulation and the endpoint-relevant biological bridge.
5. Functional in vivo validation.

This leaves one optional display item under the journal limit. It should be
used only if a second chemistry adapter, a major mechanistic result, or a
critical experimental comparison earns a main-text figure.
