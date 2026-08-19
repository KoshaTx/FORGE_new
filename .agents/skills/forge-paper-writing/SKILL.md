---
name: forge-paper-writing
description: Draft, revise, or review FORGE scientific manuscript prose, abstracts, results, methods, captions, decision packages, or reviewer responses. Use when paper claims must remain traceable to data and the approved scientific scope. Do not use for implementation-only tasks.
---

# FORGE Paper Writing

Write from the repository evidence after reading `AGENTS.md` and the relevant task/result files. Preserve
the paper's claim boundaries; do not turn plans or hypotheses into results.

## Build an evidence ledger first

For every material claim, classify its support:

- **Measured:** directly observed in wet-lab or source data.
- **Computed:** produced by a versioned, reproducible analysis.
- **Reported:** taken from a cited external primary source.
- **Inferred:** a stated interpretation of measured, computed, or reported evidence.
- **Proposed:** future work, design choice, or untested mechanism.

Use results only when their artifact records input hashes and the relevant verification passed. If evidence
is missing, write the gap explicitly or omit the claim. Never invent a citation, count, yield, uncertainty,
protocol outcome, or causal explanation.

## Preserve scientific validity

- State denominators, split definitions, units, and uncertainty alongside headline numbers.
- Distinguish retrospective computation from prospective wet-lab validation.
- Report negative and null outcomes with the same prominence as positive outcomes.
- Separate predictive association from causal or mechanistic claims.
- Identify model and data applicability limits, especially chemistry class, support tier, molecule size,
  assay, and biological endpoint.
- Treat human review or PI decisions as unresolved until they are recorded.
- Cite primary literature or primary data whenever available. Keep quoted language minimal.

Use the repository's settled terminology exactly:

- `reaction-enumerated support`, never `route-certified`, until L2 and L3 close.
- E0/E1/E2/E3 support tiers.
- FORGE is a synthesis-grounded whole-lipid and synthesis-program framework. Ugi 3-CR is the first
  prospective instantiation, not the architectural boundary or evidence that every chemistry is
  already supported.
- AGILE-type amine–aldehyde–isocyanide Ugi 3-CR, with no carboxylic-acid reactant component. Do not
  shorten this to `acid-free`, because the reported procedure uses an acidic catalyst.
- Any graph *within the declared bounded atom, bond, and size support* is representable.
- AGILE is a predictive general-transfection oracle, not an in-vivo endpoint oracle.

## Match prose to section

- **Results:** lead with the finding, then the quantitative evidence, robustness checks, and limitation.
- **Methods:** provide enough detail to reproduce the analysis; name frozen inputs, splits, parameters,
  software, seeds, and exclusions.
- **Discussion:** interpret without repeating all results; distinguish evidence, inference, and proposal.
- **Captions:** make the figure or table interpretable without the main text; define cohorts, panels,
  statistics, and error bars.
- **Abstract/title:** claim no more than the completed evidence supports. Use the cautious pre-results
  framing until prospective synthesis is complete.
- **Reviewer response:** answer directly, identify the changed evidence or text, and acknowledge valid
  limitations without rhetorical defensiveness.

## Style and final check

Use precise, compact scientific prose. Prefer concrete subjects and verbs, consistent terminology, and
short paragraphs organized around one claim. Avoid promotional adjectives, vague novelty language,
anthropomorphism, and claims of universality.

Do not use em dashes in manuscript prose. Use commas, parentheses, colons, semicolons, or separate
sentences according to the grammatical relationship.

Before delivery, verify every number against its source artifact, every citation against the cited source,
every acronym at first use, and every conclusion against the authorized scope. Flag any sentence that
depends on an unresolved human judgment.
