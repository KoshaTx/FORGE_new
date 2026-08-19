# FORGE figure visual system

Status: working art direction. Scientific structures, labels and quantitative
panels remain editable and are replaced only from frozen artifacts.

## One BioRender-quality visual family, two panel grammars

Every figure should meet the clean, pastel, publication-ready standard of the
user-supplied BioRender-style examples. The previous box-and-arrow vector
previews are retired as art direction. They may remain as layout records, but
they are not acceptable final figures. Computational panels must use the same
soft biological illustration language, whitespace, palette and line discipline
as the experimental panels. Their content can be more geometric because the
scientific objects are molecular graphs, sampling trajectories, route trees and
evidence records, but they must not look like software architecture diagrams.

### Biology and experimental workflow

Use clean, flat, two-dimensional scientific illustration with rounded forms,
ample whitespace and simple directional arrows. This language is appropriate
for LNP formulation, plates, cells, animals, payloads and tissue-level outcomes.
The user-supplied figures set the reference level: BioRender-like clarity,
limited colors and no decorative realism.

### Molecular generation and route reasoning

Use a BioRender-native but more geometric visual language:

- exact RDKit-rendered molecular structures for any molecule presented as data;
- sparse molecular graphs only when explaining the representation;
- clean route trees with reaction nodes, intermediate cards and terminal leaves;
- compact, illustrated evidence cues for exact precedent, availability,
  uncertainty and prospective outcome rather than dashboard-style status
  boxes;
- probability ribbons, particles or ranked candidate bands to explain guidance;
- small, aligned comparison panels with identical inputs and budgets;
- quantitative plots from frozen result tables, not illustrative curves.

Do not use generic brain icons, glowing AI effects, network spaghetti, dense
rounded cards or a black box labeled "AI". Every arrow must correspond to a
defined computation or experiment. Reagent bottles, plates, microfluidic
mixers, LNPs, cells, animals and tissues should be immediately recognizable to
an mRNA-delivery reader. Exact molecules, axes, values and labels are overlaid
from controlled vector sources after the illustrative composition is fixed.

## Color semantics

Use color consistently across all figures:

| Color | Meaning |
|---|---|
| Navy `#173F73` | FORGE model, sampling state and main arrows |
| Gold `#E2A12C` | Ionizable lipid or final product |
| Teal `#4B9B96` | Synthesis route, precursor or route-complete state |
| Purple `#6D4796` | Biological objective, RNA or delivery readout |
| Coral `#D45B65` | Uncertainty, failure, abstention or unsupported state |
| Pale blue `#D8E7F5` | Verification, containers, panel borders and neutral context |
| Charcoal `#292F36` | Text, chemical bonds and axes |

Color should never be the sole carrier of meaning. Pair colors with labels,
line styles or symbols.

## Figure 1: the product-route dossier

Primary visual question: what does FORGE generate that prior lipid-discovery
systems do not?

Recommended panel sequence:

1. A complete generated ionizable-lipid structure, with the Ugi core subtly
   marked and precursor origins outlined without fragment-ID boxes.
2. Exact L1 decomposition into amine, aldehyde and isocyanide components.
3. Recursive L2 routes for every nonterminal component, terminating in accepted
   L3 materials.
4. Forward checks that reconstruct each component and the final product.
5. A compact dossier card containing evidence grade, availability, route burden,
   uncertainty and prospective status.

The molecule and route tree should be visually coequal. A route should not look
like metadata attached to a more important molecular output.

## Figure 2: applicability-aware generation and synthesis assessment

Primary visual question: which information improved candidate allocation, and
where does synthesis enter the retained production workflow?

The final figure should contain four visually distinct findings:

1. the broad prior over all 57,190 qualified morphology programs and the
   support-preserving applicability proposal;
2. the fresh increase in supported-terminal yield from 3.87% to 8.64%, with
   full morphology support retained;
3. the matched null potency challenger, shown as a tested but unpromoted
   controller rather than a production lane;
4. proposal-augmented complete-product routing, the matched null synthesis
   trajectory diagnostic and the retained pre-lock routing funnel.

The matched synthesis diagnostic can retain its shared-generator layout, but
the route-guided lane must carry an explicit `not promoted` status and the
complete-product routing lane must carry the `production workflow` status.
Matched budgets, terminal denominators and the distinction between exact route
closure and family-supported route readiness must remain visible.

Outcome panels should report applicability yield, route readiness, novelty,
diversity and effective component count together. Do not use an illustrative
curve where a result is not frozen.

## Figures 3 to 5: experimental consequence

### Figure 3

Use a candidate-by-stage matrix showing every attempted route. Rows are locked
candidates; columns are precursor preparation, final assembly, correct product,
conversion, isolated yield, purity and route deviation. Failure colors use
coral and remain visible.

### Figure 4

Use flat biological icons for microfluidic formulation and the cellular bridge,
combined with conventional quantitative plots for measured particle and assay
outcomes. Show only properties that were actually collected.

### Figure 5

Use a restrained animal and payload schematic plus quantitative in vivo panels.
The artwork must follow the final endpoint and route of administration rather
than committing early to liver, muscle, lung, vaccination or a disease model.

## Current concept assets

`manuscript/figures/concepts/forge_biology_workflow_concept_v1.png` is a style
and composition study generated from the user-supplied visual references. It is
not a final scientific figure. It contains no frozen data and must not be used
to imply a selected tissue, endpoint or exact chemistry.

`manuscript/figures/concepts/forge_guidance_comparison_concept_v1.png` is a
computational style study, not the final scientific composition. Its depiction
of route-guided generation is retained only to explain the rejected diagnostic.
The final panel must instead foreground applicability enrichment and
complete-product route search before panel locking. The stylized molecules,
locks, route trees and empty plot axes are placeholders only.

The final figures will rebuild labels, exact arrows, molecules, route nodes and
plots as editable vector elements while preserving the pastel illustration and
spatial hierarchy of the concepts. Raster concept art may supply or inspire
only generic biological icons after visual and licensing review. It must never
be the source of a chemical structure, numerical result, axis or scientific
claim.
