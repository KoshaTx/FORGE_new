# FORGE v1 manuscript outline

## Scientific thesis

FORGE uses reaction programs as semantic coordinates for de novo whole-lipid generation, allowing
the model to generate new molecular components rather than select them from a finite catalogue,
while retaining exact assembly checks and recursive synthesis dossiers.

The matched post-hoc and finite-catalogue comparisons test that thesis. The AGILE-type
amine–aldehyde–isocyanide Ugi three-component reaction is the deepest computational case, BL_2023
repeated aza-Michael addition is the full second family, and LX_2024 repeated reductive amination is
a lighter third-family stress test. The current paper is computational only. It contains no
prospective synthesis, formulation, in-vitro or in-vivo claims.

## Working title

**Synthesis-grounded generative design of ionizable lipids**

Alternatives:

- **Reaction-program-guided generative design of ionizable lipids**
- **De novo ionizable-lipid generation with recursive synthesis programs**
- **Reaction programs enable catalogue-free generative design of ionizable lipids**

## Paper-level argument

The paper advances through five claims:

1. Existing modular design methods define candidates by selecting components from a finite
   inventory.
2. FORGE generates complete molecular graphs conditioned on reaction-program semantics without
   component identifiers or fragment tokens.
3. These semantics improve reaction-consistent generation relative to null, incorrect-program and
   post-hoc controls.
4. FORGE can leave the training component catalogue while retaining exact final-assembly
   consistency.
5. Recursive synthesis-program construction connects generated components to evidence-qualified
   terminal materials, with explicit abstention when route evidence does not close.

Each claim receives one main figure.

## Result placeholder convention

Unfinished manuscript numbers use evidence-aware placeholders:

- `{{COMPUTED: artifact/key}}` for a result expected from a specific reproducible artifact.
- `{{PENDING: experiment and metric}}` for an experiment that has not completed.
- `{{NEGATIVE-RESULT TEXT}}` for the prespecified fallback when a gate fails.
- `{{REPORTED: primary citation}}` for facts from external primary literature.
- `{{INFERRED: interpretation}}` for a conclusion drawn from computed or reported evidence.

Plans and hypotheses must not appear as completed results.

## Abstract, 145–150 words

Use six sentences:

1. Ionizable-lipid discovery benefits from modular synthesis and machine-learning-guided screening,
   yet these approaches usually search candidate spaces defined before inference.
2. Introduce FORGE as a synthesis-grounded framework that generates complete lipid graphs using
   reaction programs as semantic coordinates without selecting stored components.
3. Explain that FORGE jointly represents precursor roles, reaction-core positions and program depth
   while generating molecular identity at atom and bond resolution.
4. State the multi-reaction result:
   `{{conditioned versus null/incorrect/post-hoc exact-L1 result}}`.
5. State the finite-catalogue result:
   `{{open-ended exact-L1 products per 1,000 attempts, with paired-seed interval}}`.
6. State the route result and limitation: `{{fraction reaching L1/L2/L3 evidence closure}}`; these
   results establish computational transform consistency and route completeness, not experimental
   synthesis success.

Do not include implementation details, engineering preflight results, experimental potency claims
or the null synthesis-guidance diagnostic.

## Introduction

### Paragraph 1: Ionizable-lipid discovery depends on molecular structure

Establish the importance of ionizable lipids for RNA delivery and the sensitivity of delivery
behavior to changes in head groups, linkers and hydrophobic regions. End by explaining why
experimental programs commonly use modular reactions and library screening.

### Paragraph 2: Current computational methods improve selection within predefined spaces

Credit combinatorial chemistry, virtual libraries and machine-learning ranking. Introduce the exact
limitation: these methods can improve which candidate is selected, but the identities of the
available components still determine which lipids can be considered. Do not claim that rational
design is impossible or that every previous study uses a literal commercial catalogue.

### Paragraph 3: De novo generation creates a synthesis-grounding problem

Explain the central dilemma:

- Restricting generation to known heads, linkers and tails preserves reaction validity but bounds
  discovery.
- Generating unconstrained complete structures expands molecular space but can sever the connection
  to known assembly chemistry and precursor accessibility.
- Post-hoc synthetic-accessibility scores do not identify an exact assembly or explain how a novel
  component is reached.

End with the required capability: generate new molecular identity while preserving
reaction-specific semantics and auditable route evidence.

### Paragraph 4: FORGE joins whole-graph generation to recursive synthesis programs

Describe whole-graph discrete generation, reaction programs as semantic coordinates, the absence of
component identifiers and fragment tokens, exact decomposition and forward replay, and recursive
L1/L2/L3 dossiers. Introduce Ugi as the deep case, BL as the second family and LX as the lighter
stress test. Preview comparisons against null, incorrect-program, post-hoc and finite-catalogue
controls. End with a computational-only conclusion.

# Results

## Result 1: FORGE uses reaction programs to organize whole-lipid generation

**Scientific question:** Can reaction knowledge structure generation without turning the model into
a component selector?

Content:

1. Define a reaction program as roles, ordered transformations, reaction-core positions, program
   depth and coarse per-role morphology.
2. Explain what the model receives:
   - program identity;
   - precursor-role labels;
   - core-position coordinates;
   - program depth;
   - per-origin atom counts, attachment counts, branch budgets and cycle ranks;
   - noisy atom, bond and topology states.
3. Explain what the model does not receive:
   - component identifiers;
   - stored head, linker or tail graphs;
   - fragment tokens;
   - clean target atoms or bonds.
4. Describe whole-graph denoising within the declared 194-heavy-atom and three-closure support.
5. Define exact L1 decomposition and forward replay.
6. Introduce recursive L2 component preparation and L3 terminal-material evidence.
7. Define E0–E3, with E2 as the primary open-endedness tier.

**Primary display:** Figure 1.

This section defines the scientific object and does not require a model-performance claim.

## Result 2: A shared generator learns reaction-consistent lipids across three programs

**Scientific question:** Can one model learn reusable reaction-program coordinates without erasing
the deeply supported Ugi behavior?

Experiments:

- Four matched arms:
  1. Ugi-only conditioned.
  2. Shared Ugi/BL/LX conditioned.
  3. Shared null-conditioned.
  4. Shared cyclic incorrect-program control.
- Three independent training seeds.
- Fixed final checkpoint with identical architecture and compute.
- Equal total program mass in shared training.
- Exact component-disjoint calibration and heldout partitions.
- Ugi retention tested by frozen paired non-inferiority margins.

Report separately for Ugi, BL and LX:

- chemical validity;
- connectedness;
- exact-L1 decomposition coverage;
- exact-forward replay precision;
- exact-L1 yield per generation attempt;
- decomposition abstention;
- decomposition ambiguity;
- unique product fraction;
- whole-product novelty;
- component novelty;
- internal diversity;
- effective component count;
- fixed-state violations;
- support overflows.

Primary statistics:

- mean across three independent seeds;
- individual seed points;
- paired-seed bootstrap intervals;
- no molecule-level pseudoreplication.

Working positive conclusion:

> Reaction-program conditioning preserved Ugi performance while extending exact
> reaction-consistent generation to two additional ionizable-lipid programs.

Required negative-result fallback:

> Shared training did not meet the frozen Ugi non-inferiority criterion, although the negative
> result still defines the limits of reusable reaction-program conditioning.

**Primary display:** Figure 2.

## Result 3: Reaction semantics alter generation rather than merely filtering its output

**Scientific question:** Does the reaction program affect the generative distribution, or can the
same result be obtained by filtering complete molecules afterward?

Comparisons:

- conditioned generation versus matched null generation followed by exact post-hoc reaction
  filtering;
- correct program coordinate versus cyclic incorrect-program coordinate;
- identical attempt budgets;
- identical product support and final reaction adapters;
- zero route-planner, biological-oracle or candidate-selection calls.

Primary endpoint:

- exact forward-verified L1 products per generation attempt.

Secondary endpoints:

- valid products per attempt;
- exact-L1 coverage among valid products;
- forward-replay precision among decomposable products;
- uniqueness and diversity of surviving products;
- computational cost per accepted exact-L1 product.

Causal interpretation:

- Conditioned versus post-hoc tests whether semantic coordinates improve where samples are
  generated.
- Correct versus cyclic program tests whether chemical meaning matters beyond adding a categorical
  input.
- Null versus conditioned tests the contribution of reaction semantics within the same
  architecture.

Working positive conclusion:

> Reaction-program coordinates increased the yield of exact reaction-consistent products at matched
> generation budgets.

Required null-result fallback:

> Exact post-hoc filtering matched conditioned generation, indicating that the current reaction
> coordinates did not measurably improve sampling efficiency.

**Primary display:** Figure 3a–c.

## Result 4: Whole-graph generation escapes a finite component vocabulary

**Scientific question:** What does FORGE gain by generating component identity instead of selecting
from known components?

The strong train-only finite-component catalogue assembler receives exact train-fold components,
exact forward chemistry, source-weighted component marginals and the same per-family attempt budget.
The baseline is deliberately advantaged on reaction validity. Its defining limitation is zero
component escape.

Primary metric:

> Unique open-ended exact-L1 products per 1,000 attempts: distinct valid, forward-consistent products
> containing at least one component constitution absent from the train-fold catalogue.

Report separately:

- raw validity;
- exact-L1 yield per attempt;
- unique exact-L1 products per 1,000 attempts;
- open-ended exact-L1 products per 1,000 attempts;
- whole-product novelty;
- component novelty;
- internal diversity;
- effective component count;
- source-product catalogue coverage by fold;
- catalogue component coverage by role;
- finite tuple-space upper bound.

Do not collapse these metrics into a winner score. The catalogue assembler may have higher exact
reaction validity, whereas FORGE should have nonzero component escape and a less bounded discovery
space.

**Primary displays:** Figure 3d–f, Figure 4 and Table 1.

## Result 5: FORGE generalizes beyond observed component combinations

**Scientific question:** Does component novelty reflect genuine graph generation rather than
memorized products or trivial recombination?

Experiments:

- exact held-component evaluation by precursor role;
- held component pairs and exact tuples;
- whole-product novelty relative to training;
- nearest-neighbor similarity to training products and components;
- train/calibration/heldout catalogue reachability;
- component-origin-specific novelty;
- memorization audit;
- held-reaction-family stress test as a secondary analysis only.

For Ugi, report amine-, aldehyde- and isocyanide-held-out results separately. For BL and LX, report
the repeated precursor and accumulator roles defined by their adapters.

Distinguish three novelty levels:

1. novel product assembled from known components;
2. product containing a component absent from training;
3. product containing a component absent from both training and the expanded reference catalogue.

The held-family experiment must not become a hard gate or a universal zero-shot retrosynthesis
claim.

**Primary display:** Figure 4.

## Result 6: Recursive synthesis programs connect generated lipids to evidence-qualified materials

**Scientific question:** Can FORGE turn a generated graph into an auditable computational synthesis
dossier?

Analysis cascade:

1. Generated complete product.
2. Exact L1 decomposition into program-specific components.
3. Exact forward reconstruction of the product.
4. Commercial-status check for every component.
5. Bounded L2 planning for components without accepted terminal status.
6. Exact or evidence-qualified forward checks for L2 steps.
7. L3 closure to dated, supported terminal-material evidence.
8. Explicit abstention for unresolved identity, scope, selectivity or availability.

Report:

- number entering each stage;
- denominator at every stage;
- exact-L1 coverage and precision;
- L2 planner coverage and precision;
- abstention and ambiguity rates;
- complete L1/L2/L3 dossier fraction;
- evidence tier of every admitted step;
- route depth and route burden;
- failure taxonomy;
- E0/E1/E2 counts;
- no promoted E3 claim unless separately qualified.

Include one representative E2 dossier showing the complete generated lipid, L1 reaction, exact
precursor identities, upstream preparation of the generated component, terminal materials, evidence
locator and evidence tier for every edge, and explicit uncertainty and abstention fields.

Required language:

> Computational route closure records a forward-consistent, evidence-qualified synthesis program.
> It does not estimate experimental synthesis success.

**Primary display:** Figure 5.

## Result 7: Guidance diagnostics define where additional optimization is justified

This result remains short in the main text and fully reported in Extended Data.

Include:

- support-aware sampling or the bounded HeLa potency diagnostic only within the frozen applicability
  domain;
- lambda-zero identity;
- eligible versus abstained samples;
- predicted utility shift with uncertainty;
- novelty and diversity preservation;
- terminal efficiency;
- matched in-trajectory versus post-hoc comparison;
- the generation-time synthesis-guidance diagnostic.

If potency guidance produces no eligible nonuniform contrast, call it signal-limited. If synthesis
guidance is null or adverse, report the result and retain complete-product routing after generation.
Do not claim improved biological activity, delivery or experimental potency. These are computational
predictor diagnostics.

**Primary displays:** Extended Data Figures 8–10.

# Discussion

Use five paragraphs:

1. Restate the contribution: FORGE changes the generated object from an isolated molecular graph to
   a graph interpreted through a reaction program and linked to a recursive synthesis dossier.
2. Explain the finite-vocabulary tradeoff: catalogue methods preserve validity but bound component
   identity; whole-graph generation creates component escape while reaction semantics preserve
   auditability.
3. Interpret the multi-reaction result carefully: Ugi supplies the deepest evidence, BL supplies a
   full second computational family, and LX is a lighter stress test. Family support is reusable
   only after each adapter and evidence set is qualified.
4. State the limitations: the study is computational; route completeness does not estimate synthesis
   success; no biological efficacy is claimed; graph support is bounded; L2/L3 coverage can abstain;
   only three reaction programs are supported; held-family performance is secondary; shallow BL/LX
   evidence does not inherit Ugi evidence depth.
5. Close with the broader implication: reaction programs provide a practical coordinate system for
   generative molecular design when the objective is to expand molecular identity without
   disconnecting candidates from chemistry.

# Main display plan

Use five figures and one table, remaining within the six-display target for a Nature Biotechnology
Article.

## Figure 1 | FORGE generates whole lipids within reaction-program coordinates

- a, finite-component selection versus FORGE whole-graph generation;
- b, exact information entering the model;
- c, sparse discrete-flow generation of atoms, bonds and precursor-origin regions;
- d, program-specific L1 decomposition and forward replay;
- e, recursive L2/L3 synthesis dossier;
- f, E0–E3 evidence tiers and explicit abstention.

## Figure 2 | A shared model learns three ionizable-lipid reaction programs

- a, Ugi, BL and LX datasets, roles and split sizes;
- b, four-arm experimental design;
- c, exact-L1 yield by family and arm;
- d, validity and connectedness;
- e, decomposition coverage versus forward-replay precision;
- f, Ugi retention and non-inferiority intervals.

## Figure 3 | Reaction conditioning improves usable yield at matched budgets

- a, conditioned generation versus post-hoc filtering;
- b, correct program versus null and cyclic incorrect-program controls;
- c, accepted exact-L1 products per 1,000 attempts;
- d, whole-product novelty;
- e, component novelty and effective component count;
- f, cost per accepted forward-verified product.

## Figure 4 | Whole-graph generation exceeds the finite component catalogue

- a, exact train-catalogue coverage by family and fold;
- b, validity versus component-escape tradeoff;
- c, open-ended exact-L1 products per 1,000 attempts;
- d, held-component results by role;
- e, diversity and nearest-neighbor distributions;
- f, examples of novel products, familiar-component recombinations and exact-new components.

## Figure 5 | Recursive synthesis programs produce auditable computational dossiers

- a, L1→L2→L3 closure cascade;
- b, route coverage and precision;
- c, evidence tiers of admitted steps;
- d, route-depth and route-burden distributions;
- e, representative E2 synthesis tree;
- f, abstentions and failure taxonomy.

## Table 1 | Matched production comparison across reaction programs

Group columns by Ugi, BL and LX. Include rows for validity, connectedness, exact-L1 yield,
exact-forward precision, unique exact-L1 products per 1,000 attempts, open-ended exact-L1 products
per 1,000 attempts, component novelty, whole-product novelty, internal diversity and effective
component count. Each cell reports the mean across seeds and a paired-seed interval where
applicable.

# Extended Data figures

1. Data provenance, constitutional deduplication and split construction.
2. Reaction-adapter definitions and exact source-adjudication results.
3. Representation qualification, support coverage and fixed-state checks.
4. Full training curves and fixed-checkpoint metrics for every arm and seed.
5. Complete per-family generative metrics, including abstention and ambiguity.
6. Detailed finite-catalogue construction and fold-wise coverage.
7. Held-component, held-pair and held-reaction-family stress tests.
8. Potency-predictor benchmarking, calibration and applicability policy.
9. Potency-guidance or support-aware sampling diagnostic.
10. Matched synthesis-guidance versus post-hoc diagnostic, including null results.

# Supplementary tables

1. Hash-pinned data-source ledger.
2. Reaction-program definitions, roles and evidence sources.
3. Constitutional identity and provenance reconciliation.
4. Complete split memberships and leakage audits.
5. Model architecture, support and training configuration.
6. All per-seed production metrics.
7. All paired bootstrap contrasts and decision gates.
8. Finite-catalogue component counts and reachability.
9. Decomposition coverage, precision, ambiguity and abstention.
10. L2 planner benchmark and false-positive taxonomy.
11. Generated-candidate L1/L2/L3 dossier ledger.
12. Guidance applicability, abstention and uncertainty results.
13. Compute environment, runtimes, seeds and artifact hashes.

# Online Methods outline

1. LNPDB data sources and constitutional identity.
2. Source-grounded reaction-program admission.
3. Ugi, BL and LX reaction adapters.
4. Component-disjoint split construction.
5. Whole-lipid graph representation and declared support.
6. Reaction-program conditioning.
7. Sparse discrete-flow objective.
8. Source-balanced training and fixed checkpointing.
9. Generation and exact program evaluation.
10. Null, cyclic and post-hoc controls.
11. Finite-component catalogue baseline.
12. Novelty, diversity and open-endedness metrics.
13. Exact decomposition and forward replay.
14. Recursive L2 planning and L3 evidence closure.
15. Potency and synthesis-guidance diagnostics.
16. Statistical analysis.
17. Reproducibility, artifact hashing and software environment.

## Governing repository documents

- [`docs/PLAN.md`](../../docs/PLAN.md)
- [`docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md`](../../docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md)
- [`docs/MULTIREACTION_COMPUTATIONAL_PLAN.md`](../../docs/MULTIREACTION_COMPUTATIONAL_PLAN.md)
- [`AGENTS.md`](../../AGENTS.md)
