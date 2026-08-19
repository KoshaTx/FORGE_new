# M0-09 Ugi-3 Precursor Capability

## Decision

The current FORGE paper prospectively instantiates the framework in the
AGILE-type amine–aldehyde–isocyanide Ugi three-component reaction. This
chemically controlled validation scope is narrower and more testable than
general lipid retrosynthesis:

1. Obtain or synthesize an amine head.
2. Obtain or synthesize an aldehyde-bearing tail.
3. Obtain or synthesize an isocyanide tail.
4. Apply the validated Ugi-3 final assembly.

There is no separate linker reactant at L1. A spacer, ester, carbonate, ether,
amide, or other linker-like motif may be embedded within one of the three
components, and the chemistry that constructs it remains part of L2.

The whole-molecule discrete flow still generates the complete lipid. This
inventory defines whether the generated graph can be decomposed into those
three Ugi-compatible inputs and whether each input has a complete L2 and L3
path.

General lipid chemistry remains useful as a source of upstream transformations
and tail motifs. Multi-assembly lipid generation is reserved for a later,
purely computational extension so that the present paper can make a strong,
well-supported chemistry claim.

## Direct Ugi-3 route evidence

AGILE Supplementary Note 1 provides the current lipid-native route foundation.

| Ugi-3 input | Reported terminal targets | Route families | Reaction instances | Explicit yields | Structure-resolved routes |
|---|---:|---:|---:|---:|---:|
| Amine heads | 20 measured | 0 upstream | 0 | 0 | 0 |
| Aldehyde tails | 18 | 3 | 34 | 0 | 17 |
| Isocyanide tails | 7 | 2 | 14 | 0 | 7 |
| **Total routed tails** | **25** | **5** | **48** | **0** | **24** |

The 18 aldehyde routes comprise:

- 12 ester-bearing aldehydes with a six-carbon spacer, made by EDC and DMAP
  esterification followed by Dess-Martin oxidation;
- two direct fatty aldehydes, palmitaldehyde and olealdehyde, made by
  Dess-Martin oxidation;
- four ester-bearing aldehydes with a four-carbon spacer, made by the same
  esterification and oxidation sequence.

The seven isocyanide routes comprise:

- five measured C12, C14, C16, C18, and Z-C18:1 isocyanides;
- two additional C11 and C17 isocyanides;
- a shared sequence of primary amine formylation followed by phosphorus
  oxychloride dehydration.

All 25 products have product-specific proton NMR in the source. The source does
not report isolated yields for the upstream tail steps. Twenty-four products
have structure-resolved route records. B5 remains the sole unresolved
source-to-measured identity conflict and is not assigned a route.

JC_2023 provides independent native-Ugi corroboration rather than additional
AGILE identities. It reports a product-specific two-step preparation of
6-oxohexyl 4-methylnonanoate with 86% and 61% step yields, two saturated
isocyanide products under the same formylation and dehydration family, and a
product-specific oleyl-isocyanide route with 35% final isolated yield. Its
isolated iso-A11B5C1 example reports a 70.71% final Ugi yield with ESI-MS and
proton NMR.

The JC_2023 aldehyde is source-labeled B5 but is constitutionally identical to
AGILE B2. This is useful independent execution evidence and does not resolve
AGILE's separate B5 identity discrepancy. The reviewed artifact also preserves
the supplement's inconsistent mass and mmol fields and reconciles, without
overwriting, an LNPDB oleyl-isocyanide string that does not encode an
isocyanide.

LX_2024 adds a distinct aldehyde architecture from a non-Ugi final-assembly
platform. Its 15 source-reported aldehydes contain aromatic cores with two or
three degradable fatty-acid esters. All 15 have source structures,
product-specific proton NMR, and a reported EDC and DMAP preparation family.
All 15 also yield exactly one sanitized product under the frozen Ugi transform
with the reference amine and isocyanide.

This is direct component transfer rather than exact final-product transfer. The
aldehydes expand the Ugi-route support set, while LX reductive-amination
products and biological labels remain attached to their original platform.
The source does not establish that any corresponding Ugi lipid has been
isolated.

## Computed precursor audits

The hash-pinned capability artifact evaluates the 39 programmatic rational
isocyanides in `building_block_pool_v1.json` against exact AGILE route targets.
It finds:

| Axis | State | Isocyanides | Interpretation |
|---|---|---:|---|
| Transformation evidence | Exact source route | 5 | Exact AGILE target and structure-resolved route |
| Transformation evidence | Bounded family applicability without an exact route | 34 | Within the declared aliphatic design grid, without an exact reported route |
| Route closure | Computationally complete | 0 | No candidate yet closes every step and terminal leaf |
| Prospective outcome | Beyond not attempted | 0 | No post-freeze experimental outcome exists |

Exact structure-resolved AGILE upstream-route coverage is therefore 5 of 39, or
12.8%. This is not a failure of the candidate space. It is a statement about
the current strength of evidence. The five exact examples cover four of nine
straight saturated members and one of seven straight monounsaturated members.
Exact evidence is absent for the five straight diunsaturated members and all 18
branched members.

AGILE contributes 20 exact amine head structures. A time-stamped US
procurement snapshot closes 19 at the current item level. A9 has an exact
catalog item, but its page requires a cart check and therefore does not close
current availability. Zero heads have an extracted upstream route. The Miao
2019 cross-assembly source contributes 12 exact amine heads and ten exact
isocyanides. Four of its heads exactly overlap AGILE. These records broaden
observed component chemistry, but they do not prove AGILE-type Ugi 3-CR
compatibility or upstream synthesis for those components.

Head closure requires more than a historical statement that the amines were
commercial. Each review must resolve the exact structure, salt and protection
state, Ugi-reactive amine handle, competing nucleophiles, current vendor and
catalog evidence or internal stock, and an upstream route when procurement does
not close. A purchased head may terminate at L3 only after time-stamped
item-level verification under the declared region and purity policy.

The building-block pool also contains 51 aldehydes and 88 amines. Multiplying
39 by 51 by 88 gives 175,032 precursor triples. This is only a raw Cartesian
upper bound. It is not a count of unique products, compatible reactions,
route-complete lipids, or synthesizable molecules.

The candidate-wide aldehyde and head audit is generated by
`make m0-09-ugi3-aldehyde-head-capability`. Its findings are:

| Diagnostic | Count |
|---|---:|
| Aldehyde candidates | 51 |
| Aldehydes with an exact source route | 11 |
| Structure-resolved AGILE aldehyde routes outside the pool | 6 |
| Unresolved aldehyde identity discrepancies | 1 |
| Amine-head candidates | 88 |
| Heads containing at least one registry N-H handle | 88 |
| Heads passing the literal raw-match diagnostic | 78 |
| Heads failing the literal raw-match diagnostic | 10 |
| Heads passing the symmetry-qualified multiplicity policy | 85 |
| Heads failing the symmetry-qualified policy | 3 |
| Nominal AGILE rows passing the qualified mechanical policy | 1,200 of 1,200 |
| Reconciled single-compound L1 records | 1,100 |
| AGILE heads with current item-level procurement closure | 19 of 20 |
| AGILE heads with an exact catalog item but unresolved availability | 1 |
| All pool heads still requiring procurement or route resolution | 69 |
| Heads with an extracted upstream route | 0 |
| Aldehyde or head candidates that are computationally route-complete | 0 |

The 11 exact aldehyde overlaps are the measured AGILE aldehydes other than B5.
The six additional structure-resolved aldehydes broaden the observed
transformation basis, but they do not occur in the current 51-member pool and
are not transferred to another candidate by analogy.

The raw-match audit exposed and the measured-library qualification resolved a
registry-policy ambiguity. The registry allows one or two amine sites. A5 has
three raw N-H matches, but those atoms are symmetry-equivalent and generate
three identical forward outcomes that deduplicate to one product. Applying the
registry values to symmetry-distinct reactive sites admits A5 without an
identity-specific exception and reconstructs all 1,200 nominal AGILE table rows.
M0-07 separately excludes the 100 B4 mixture records from single-graph
supervision.
The literal raw count remains a diagnostic rather than a compatibility gate.

The qualification also preserves genuine site ambiguity. A19 and A20 each
contain two symmetry-distinct amine sites and yield two unique forward products.
Both satisfy the registry's allowed multiplicity, but a generated or
retro-decomposed candidate must identify the reacting atom explicitly. Across
the complete 88-head pool, 85 pass the qualified site policy and three remain
outside the declared one-or-two-site envelope. The resulting 169,065 precursor
triples are still only a compatibility-filtered upper bound, not a
synthesizable or route-complete count.

The exact AGILE head review records constitutional identity, free-base form,
US-region vendor page, item identifier, purity, and a dated stock or shipping
observation. Nineteen heads satisfy the frozen procurement policy. A9 remains
open because an exact catalog listing without a dated availability observation
is insufficient. The 69-head queue contains A9 and every other pool head not
closed by current item-level evidence. This procurement result does not resolve
upstream synthesis, and it does not make any head computationally
route-complete.

## AGILE virtual product and component census

The 12,276-product AGILE Zenodo set contains final-product SMILES and computed
descriptors, but no component or route fields. A hash-pinned inverse audit
recovers exactly one qualified Ugi decomposition for every product and requires
exact forward reconstruction with an explicit reacting amine atom.

The virtual library is the complete 22 by 62 by 9 Cartesian product. Its 12,276
products reduce to 93 unique components: 22 heads, 62 aldehydes, and 9
isocyanides. Exact-identity capability joins cover 18 heads, 18 aldehydes, and
7 isocyanides. Of these, exact source routes cover 17 aldehydes and all 7
matched isocyanides. Seventeen heads are accepted procurement terminals. The
source routes are not yet forward-verified or closed to accepted leaves, so
zero virtual products are called route-complete.

The recursive component audit yields 33 unique proposed route leaves plus 5
unresolved heads. A time-stamped US procurement snapshot closes all 14 fatty
acids, all six primary alcohols, all four shared diols, eight of nine exact
free-base primary amines, and three of the five initially unresolved heads.
Three of the 38 terminal candidates remain unresolved. High-purity oleylamine
is catalogued but lacks an accepted current US stock or shipping observation.
The exact 3-aminoquinuclidine free base and 1,1-dimethylhydrazine require either
explicit operational closure or a complete route from an accepted salt.

This result converts a large product table into a bounded component-routing
queue. It does not convert the virtual products into observed reaction records
or L2 supervision. Full definitions, forward-site profiles, and artifacts are
in `docs/M0_09_AGILE_VIRTUAL_UGI3_CAPABILITY.md`.

## Independent reaction-family evidence

The capability audit preserves evidence from direct replication and general
isocyanide synthesis separately:

- An independent amine–aldehyde–isocyanide Ugi 3-CR study reports C11, C18, and cis-C18:1
  isocyanides. It reports oleyl isocyanide at 35% isolated yield. Its
  supplementary prose and scheme disagree between TFA and TEA for the
  dehydration, so both representations and the conflict remain explicit.
- Waibel and co-workers report p-toluenesulfonyl chloride and pyridine
  dehydration across long-chain aliphatic formamides. Their scope also exposes
  steric, solubility, hydroxy-substrate, and aromatic limitations.
- Brunelli and co-workers report a micellar p-toluenesulfonyl chloride and
  sodium bicarbonate method across primary, secondary, tertiary aliphatic, and
  benzylic examples. Dodecyl isocyanide is reported at 89% yield and 82% at
  10 mmol scale. Reported aromatic failures delimit the mild method.

Together these sources justify a principled primary-amine to formamide to
isocyanide reaction basis. They do not justify automatically labeling every
member of the 39-structure grid as synthesized. Each exact candidate still
needs substrate-level applicability, deterministic forward consistency,
purification feasibility, and recursive procurement closure.

## Demonstrated structural envelope

The aldehyde side is already moderately diverse:

- hydrophobic chains from C8 through C18;
- straight and one branched chain;
- saturated, internal cis and trans alkene, terminal alkene, terminal alkyne,
  and polyunsaturated examples;
- direct fatty aldehydes and ester-bearing degradable aldehydes;
- four-carbon and six-carbon aldehyde spacers.

The isocyanide side is substantially thinner:

- mostly straight saturated C11 through C18 chains;
- one cis-unsaturated C18 example;
- no demonstrated branched isocyanide;
- no demonstrated ester-, carbonate-, amide-, or ether-bearing isocyanide;
- no reported isolated yields.

This asymmetry matters. The next highest-value curation is not another large
final-lipid library. It is route and outcome evidence for structurally varied,
Ugi-compatible isocyanides, followed by additional aldehyde spacer and
degradability patterns.

## Coverage target and space census

The paper does not need one literature route for every individual tail. It
needs a complete and auditable transformation basis for the declared Ugi-3
precursor envelope. A reaction family enters that basis only when its record
specifies:

- the source-located procedure and demonstrated products;
- atom-mapped reactant and product patterns;
- required functional groups, tolerated motifs, and known incompatibilities;
- route depth, conditions, purification, yields, and failures when reported;
- the procurement or recursively routed leaves;
- the boundary between demonstrated examples and proposed scope.

The route layer may apply a validated family to a new substrate within its
declared applicability domain. Such an application is a route proposal, not a
new literature label. The audit stores five orthogonal axes rather than one
evidence rank:

| Axis | Example states | Question answered |
|---|---|---|
| Transformation evidence | no route reported; bounded family applicability; exact source route; independently reported exact route | What supports the proposed chemical step? |
| Component observation | programmatic candidate; AGILE measured component; independent Ugi component; cross-assembly component | Where has the exact structure been observed? |
| Route closure | incomplete; L1 only; unresolved leaves; computationally complete | Does every branch reach an accepted terminal leaf? |
| Operational availability | internal stock; current item-level vendor evidence; historical blanket claim; resolution required; unavailable | Can the exact material be obtained under the declared policy? |
| Prospective outcome | not attempted; failed; detected; isolated; formulated; biologically functional | What happened after the evidence freeze? |

Forward verification remains a separate operational status. No transformation
or observation tag implies route closure, current availability, or prospective
success.

The reachable-space analysis will count a bounded virtual space, not every
tail imaginable in chemistry. Its inputs are versioned catalogs of admissible
fatty acids, diols or amino alcohols, primary amines, and amine heads, together
with the reviewed transformation rules. The report will:

1. enumerate compatible reaction applications within explicit heavy-atom,
   element, charge, and functional-group limits;
2. canonicalize and deduplicate precursor products;
3. reject products outside the Ugi handle and lipid property constraints;
4. count distinct amine heads, aldehydes, isocyanides, compatible precursor
   triples, and unique whole-lipid products;
5. stratify counts by chain length, branching, unsaturation, degradable
   linkage, route family, route depth, and evidence state;
6. report attrition at every filter and avoid multiplying combinations that
   violate reaction compatibility.

This census is descriptive and prospective. It must never be added to the
source-route training count, and its size must never be used as evidence that
all enumerated members are experimentally synthesizable.

## What broader LNPDB chemistry contributes

A non-Ugi LNPDB paper contributes directly to this paper only when it supports
one of the following:

- synthesis of a hydrophobic acid, alcohol, or amine that is an upstream leaf
  for a Ugi-compatible tail;
- installation of branching, unsaturation, or a degradable linkage that can be
  retained in an aldehyde or isocyanide input;
- conversion to a terminal aldehyde or isocyanide under source-supported
  conditions;
- protection, purification, or compatibility evidence needed to execute that
  precursor route.

Final aza-Michael, acrylate, alkyne, carbonate, or other non-Ugi lipid assembly
does not count as Ugi-3 route coverage. It can supply general chemistry
knowledge, but it remains secondary evidence for this paper.

The fixed cross-platform pilot currently covers two independent sources:
XH_2025 asymmetric maleates and JL_2024 ester-bearing A3 alkynes. Eleven exact
alcohol attachments link to two actual, source-reported DCC/DMAP
esterifications with product-specific isolated yields. Seven primary alcohols
give novel, frozen-transform-compatible aldehyde proposals and four secondary
alcohols remain structure-only. One transferred proposal is computationally
route-complete under the declared evidence policy: an exact literature
procedure converts the exact JL_2024 B16 alcohol precursor to
2-decyltetradecanal in 86% isolated yield, and current US item-level evidence
closes the alcohol terminal. The remaining six aldehyde proposals are
incomplete. This is a positive transfer and targeted-expansion signal, not
observed Ugi L2 supervision or prospective Ugi success.

## Role of the large reaction corpus

The 479,035-record USPTO-MIT corpus remains useful for generic reaction
pretraining after cleaning. The current audit found 478,998 RDKit-parseable
records and 37 invalid records. Screening found 8,236 candidate aldehyde
formations but only 40 candidate isocyanide formations.

Those counts are upper-bound motif screens, not validated routes. They support
the following division of labor:

1. Use the broad corpus to learn generic reaction proposals.
2. Use AGILE and source-reviewed lipid chemistry for Ugi-native specialization.
3. Use deterministic templates and exact forward checks for the five observed
   tail route families.
4. Reject or assign high uncertainty to chemistry outside demonstrated
   applicability.
5. Use prospective synthesis outcomes to calibrate the final route score.

The immediate curation priority is therefore transformation completeness, not
paper count. Source review continues until the Ugi-3 capability matrix covers
the planned structural axes or records an explicit gap. The first virtual-space
census follows only after those rules and applicability boundaries are frozen.

## Route-extraction scope

The fact that an LNPDB lipid was experimentally reported is existential evidence
that at least one route was used. It is not, by itself, a machine-readable route
record. The source may purchase a component, cite an earlier synthesis, place
the method in another supplement or patent, omit an intermediate structure, or
report only a group-level procurement statement.

L2 review therefore follows relevant components beyond the immediate LNPDB
paper. For each Ugi-compatible head or tail motif, review must trace the primary
paper, supplementary files, cited antecedents, patents when needed, and current
procurement evidence. Extracted transformations must record substrate scope,
conditions, yields, purification, incompatibilities, reported failures, atom
mapping, and terminal leaves. Coverage is measured across branching,
unsaturation, degradable esters, carbonates, ethers, amides, heterocycles,
aldehyde formation, isocyanide formation, and other motifs present in the
declared component support.

For transferred motifs, preserve the actual source-component reaction and the
separately proposed Ugi-handle reaction as distinct records. The source paper
may establish that an alcohol participated in an esterification, but that does
not establish oxidation of the same alcohol to an aldehyde or successful Ugi
assembly of that aldehyde.

The goal is not one bespoke route per LNPDB structure. It is a compact,
source-grounded transformation basis that closes a large, explicitly bounded
Ugi-compatible component space. A gap remains a gap until an exact route,
bounded family application, current procurement record, or prospective outcome
is recorded on the appropriate axis.

## Paper-level claim boundary

The current evidence supports a bounded empirical claim inside a broader
synthesis-program architecture:

> FORGE couples complete ionizable-lipid graph generation to recursive synthesis
> programs composed of reaction-family-specific transformations. The current
> prospective instantiation uses AGILE-type Ugi 3-CR final assembly and
> source-grounded routes for its amine, aldehyde, and isocyanide precursors.

It does not yet support a claim that any conceivable lipid tail can be made.
Expansion should be measured against the capability matrix above, with special
attention to isocyanide diversity, route yields, failures, and current
procurement closure.

## Reproduction

```bash
make m0-09-ugi3-capability
make m0-09-ugi3-assembly-qualification
make m0-09-ugi3-aldehyde-head-capability
make m0-09-agile-virtual-smiles
make m0-09-agile-virtual-ugi3-capability
```

The primary result is
`results/m0_09/ugi3_precursor_capability.json`. Its inputs are hash-pinned to
the candidate pool, AGILE routes, component source ledger, paper review
artifact, and the reviewed capability configuration. The route artifact contains
16 exact measured-component routes and eight additional structure-resolved
AGILE-identified routes.

The assembly qualification result is
`results/m0_09/ugi3_assembly_qualification.json`. It is hash-pinned to the
measured 1,200-product AGILE table, qualified reaction registry, and active Ugi
variant. It records the complete forward-outcome profile, the aggregate row
qualification digest, and the A5, A17, A19, and A20 site findings.

The candidate-wide result is
`results/m0_09/ugi3_aldehyde_head_capability.json`. It is hash-pinned to the
building-block pool, AGILE route artifact, qualified reaction registry, and
precursor capability artifact, plus the time-stamped AGILE head procurement
snapshot and the assembly qualification artifact. The procurement snapshot
expires after 30 days so stale availability cannot be silently reported as
current.
