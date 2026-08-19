# M0-09 AGILE Virtual Ugi-3 Component Programs

## Purpose

The 12,276-product AGILE virtual library reduces to 93 unique Ugi components.
This audit converts those components into a bounded recursive work queue. It
does not train a route model and does not call a structural projection an
experimental route.

The deterministic audit is:

```bash
make m0-09-agile-virtual-ugi3-component-programs
```

It consumes the hash-pinned component ledger and exact AGILE source-route
registry. Every exact source program must be reproduced at the structure level
before the same reaction-family program can be projected to another component.

## Result

| State | Components |
|---|---:|
| Accepted procurement terminal | 17 |
| Exact source program | 24 |
| Reaction-family projected program | 47 |
| No assigned upstream program | 5 |
| Total | 93 |

The 24 exact programs comprise 17 aldehydes and 7 isocyanides. All 24 source
procedures are reproduced exactly at the level of reactant, intermediate, and
product structures.

The 47 family projections comprise 45 aldehydes and 2 isocyanides. They are
bounded applications of the same structure transformations, not exact
substrate evidence.

## Substrate-scope coverage

The component ledger carries explicit graph-derived annotations for carbon
count, carbon branching, carbon-carbon unsaturation, ring topology, ester
content, aromatic atoms, and heteroatom counts. The artifact reports exact and
family-projected evidence separately within each scope stratum.

The 62 aldehyde-tail components include 41 saturated, 13 monoene, 4 polyene,
and 4 alkynyl structures. Four are carbon-branched and 56 contain the
ester-bearing degradable motif. Exact source programs cover 9 saturated,
6 monoene, 1 polyene, 1 alkynyl, and 2 branched aldehyde-tail components.
The remaining cases retain family-projection status.

The 9 isocyanide tails span carbon counts 12 through 19. Eight are saturated
and one is mono-unsaturated; exact source programs cover 6 of the saturated
tails and the mono-unsaturated tail. The current AGILE virtual isocyanide set
contains no branched member, so branched-isocyanide support cannot be inferred
from this 12,276-product library.

The 22 amine heads include 13 cyclic and 9 acyclic structures over 2 to 11
carbon atoms. Procurement closure and upstream-route requirements remain
reported separately for each head.

## Program families

| Program | Components | Structural form |
|---|---:|---|
| Fatty acid and diol esterification, then alcohol oxidation | 56 | fatty acid + diol → hydroxy ester → aldehyde |
| Primary alcohol oxidation | 6 | primary alcohol → aldehyde |
| Primary amine formylation, then formamide dehydration | 9 | primary amine → formamide → isocyanide |

Across the 71 nonterminal components with a structural program, the projection
contains 65 unique intermediates and 33 unique proposed leaf structures. This
is the next efficiency gain: procurement and upstream-route resolution can be
performed once per unique leaf rather than once per component or product.

Combining those 33 leaves with the 5 unresolved heads yields a 38-structure
terminal-material queue:

| Terminal class | Unique structures |
|---|---:|
| Fatty acid | 14 |
| Diol | 4 |
| Primary alcohol | 6 |
| Primary amine | 9 |
| Unresolved amine head | 5 |

Twenty-three structures carry an exact source publication claim that the
starting material was purchased. This is historical source evidence only.
A separate time-stamped US item review now closes all 14 fatty acids, all six
primary alcohols, all four diols, eight of the nine primary amines, and three
of the five initially unresolved heads under the frozen procurement policy.
Exact vendor identity, purity, item code, and current stock or shipping
observations are recorded. High-purity oleylamine is catalogued, but accepted
current US stock or shipping has not been established. The exact
3-aminoquinuclidine free base and 1,1-dimethylhydrazine require explicit
operational closure or a complete route from an accepted salt. Three
structures remain unresolved.

The queue records dependency counts so shared leaves can be resolved once. For
example, each of the four diols feeds 14 aldehyde components. The incidence
metric is explicitly a sum over component branches, not a count of unique
virtual products.

## Evidence and closure

Exact source program, family projection, route closure, and procurement closure
remain separate fields.

- An exact source program records a procedure for the exact component.
- A family projection records only a structurally consistent application of a
  source-supported reaction family.
- A proposed leaf is not a procurement terminal until current exact-identity
  availability or internal stock is verified.
- A route remains incomplete until every step is forward-verified and every
  leaf is closed under the frozen policy.

Accordingly, all 62 aldehyde branches and all 9 isocyanide branches remain
incomplete. The 17 accepted heads close only their own procurement branches.
Five heads require exact-identity procurement review followed by upstream route
search if procurement fails.

## Artifacts

- `results/m0_09/agile_virtual_ugi3_component_programs.json`
- `results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz`
- `results/m0_09/agile_virtual_ugi3_terminal_queue.json`
- `results/m0_09/agile_virtual_ugi3_terminal_queue.csv.gz`

The ledger records the complete structural program, exact source route
identifiers, proposed intermediates, proposed leaves, evidence state, closure
state, and next action for every unique component.
