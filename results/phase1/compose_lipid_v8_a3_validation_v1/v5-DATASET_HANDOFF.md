# Dataset handoff: COMPOSE-Lipid complete LNPDB corpus v5

## Release identity

- Release: `compose_lipid_lnpdb_complete_2026-09-13_v5`
- Prepared: 2026-09-13
- Rows: 279,687 unique target IDs and constitutions
- Identity: RDKit-canonical, stereo-free, connected constitutional graph
- Families: 25 primary labels
- Fields: 109 molecule columns
- Biology labels: none
- BEAE outcomes used: no
- Final generator split: not frozen

This is the complete chemistry and provenance handoff. It supports corpus
statistics, descriptor analyses, family comparisons, source-versus-virtual
analyses, and preparation of a separately frozen generator workload. It is not a
table of 279,687 experimentally synthesized or active lipids.

## What changed from v4

Version 4 had 252,034 graphs. Version 5 recovers complete precursor identities
from every LNPDB source graph, removes blanket exclusions for orthogonal handles,
and adds 27,653 globally new exact products:

- 13,713 amine-alkylation products, raising the family from 287 to 14,000;
- 13,940 reductive-amination products, raising the family from 60 to 14,000.

The final two-family build evaluated 36,000 scheduled tuples, produced 35,267
exact passing routes, and retained 733 failures. It identified 589 cross-family
route overlaps. One molecular graph receives one row and one sampling mass;
alternate exact routes remain attached as provenance.

The old 3,833 null-to-target rows are removed. Each now has an exact compatible
decomposition and complete precursor identities. The precursor dictionary also
contains the two precursor identities referenced only by retained alternative
decompositions, yielding 5,997 complete foreign-key targets.

## LNPDB construction coverage

LNPDB contributes 19,797 occurrence rows that collapse to 12,675 unique
constitutional identities. Every identity is retained in
`lnpdb_source_ledger.jsonl.gz` and has exact construction coverage:

| Coverage | Identities |
|---|---:|
| Exact bound program | 8,842 |
| Exact complete-precursor compatible decomposition | 3,833 |
| Total | 12,675 |

The 3,833 decomposition targets have 4,184 records because 251 targets retain
more than one compatible alternative. A compatible decomposition means complete
connected precursors, an explicit local reaction-family site, and exact forward
reconstruction of the stereo-free product. It does not assert that a published
study used that route. `site_assignment_unique` and
`all_product_atoms_accounted_for` expose the strength of the saved mapping.

Current model support includes 11,144 LNPDB identities at up to 96 heavy atoms.
The remaining 1,531 are preserved as `MODEL_DEFERRED_SOURCE` due to model size or
atom-state support; deferral does not invalidate a reported molecule.

## Corpus composition

| Family | LNPDB | Other source | Virtual | Total |
|---|---:|---:|---:|---:|
| A3 amine/aldehyde/alkyne | 754 | 10 | 13,236 | 14,000 |
| Acid/epoxide diester multistep | 176 | 80 | 0 | 256 |
| AEMA aza-thiol addition | 1,173 | 0 | 12,827 | 14,000 |
| Aldehyde Ugi-3 | 1,358 | 0 | 12,642 | 14,000 |
| Aldehyde Ugi-4 | 844 | 1 | 13,155 | 14,000 |
| Alpha-isocyanoester dihydroimidazole | 181 | 3 | 13,589 | 13,773 |
| Amine alkylation | 286 | 0 | 13,714 | 14,000 |
| Amine epoxide opening | 661 | 0 | 13,442 | 14,103 |
| Aryl reductive amination | 168 | 0 | 13,832 | 14,000 |
| Aza-Michael acrylamide | 219 | 0 | 13,528 | 13,747 |
| Aza-Michael acrylate | 2,571 | 28 | 11,401 | 14,000 |
| Disulfide Michael | 52 | 0 | 47 | 99 |
| Epoxide opening/O-acylation | 25 | 0 | 13,975 | 14,000 |
| iPhos ring opening | 549 | 0 | 13,218 | 13,767 |
| Ketone/isocyanide/amide | 900 | 0 | 12,836 | 13,736 |
| Ketone Ugi-4 | 72 | 0 | 13,680 | 13,752 |
| Maleate addition | 463 | 0 | 13,537 | 14,000 |
| O-esterification | 144 | 0 | 13,856 | 14,000 |
| Preassembled thiol-yne tail amidation | 112 | 0 | 13,888 | 14,000 |
| Reductive amination | 60 | 0 | 13,940 | 14,000 |
| Reported LNPDB route unassigned | 255 | 0 | 0 | 255 |
| Reported multistep amino lipid | 9 | 0 | 0 | 9 |
| Reported reference lipid | 4 | 0 | 0 | 4 |
| Thiolactone aminolysis/Michael | 91 | 0 | 28,077 | 28,168 |
| Vitamin B5 multistep | 17 | 1 | 0 | 18 |
| **Total** | **11,144** | **123** | **268,420** | **279,687** |

Scalable classes use a coverage floor near 14,000 and a 30,000 ceiling. Small
source-specific families remain small where no defensible broad construction
contract exists. The 28,168-member thiolactone class remains below the ceiling.

## Expansion rules

The census contains 6,015 complete constitutional precursor identities, including
1,087 recovered by exact LNPDB-compatible decomposition and 460 absent from the
earlier census. An extra alcohol, nitrogen, unsaturation, branch, or orthogonal
handle does not itself reject a precursor.

Every selected virtual product must have complete connected precursors; an
explicit required handle and local site; exact forward/reverse graph trace; atom,
hydrogen, charge, reaction-vocabulary, and atom-state accounting; no sealed
structural match; at most 80 heavy atoms; and global constitutional uniqueness.
Extra or competing handles are saved as context flags.

Raw Cartesian multiplicity is not a sampling prior. A deterministic role-spread
window covers precursor values before a full-period tuple permutation. Selection
then covers precursor pairs, size, branching, unsaturation, hydrophobe count, and
regional morphology. This preserves broad lipid chemistry without letting the
largest near-neighbor grid dominate the corpus.

## Construction supervision

`construction_supervision.jsonl.gz` is target-ID aligned with the molecule table:

| Mode | Rows |
|---|---:|
| Prior exact reaction program | 205,000 |
| LNPDB-informed exact extension | 70,854 |
| Exact compatible-decomposition replay | 3,833 |

Direct rows contain program references, payload digests, reaction IDs,
constitutional precursor IDs, and repeated reagent instances. Decomposition rows
contain selected and alternative decompositions, complete precursor instances,
family-compatible programs, replay status, site uniqueness, and atom-provenance
status. The 3,833 decomposition rows still need direct model-program projection
before the entire corpus can be called path-ready.

## Labels and recommended use

The CSV and JSONL molecule shards carry identical rows. `label_dictionary.csv` is
authoritative. Labels cover identity; source names, PMIDs, occurrence IDs, and
references; primary and compatible families; route-claim scope; programs,
alternate routes, decompositions, and precursor instances; formula, mass,
elements, charge, TPSA, and calculated MolLogP; rings, branching, unsaturation,
and degradable motifs; hydrophobe-region structure; and review context.

Hydrophobes are components of the carbon-only subgraph with at least four
carbons. These are reproducible structural descriptors, not manual tail truth or
biological measurements. `organization_proxy_disposition=HOLD` is a review flag,
not removal from the chemistry table.

For analysis, verify the package, concatenate numeric shards, preserve provenance
IDs, document every filter and count, and treat virtual structures and calculated
descriptors as such. Use balanced resampling for family comparisons because raw
corpus proportions are designed coverage rather than prevalence.

For training, build a new grouped train/calibration/test split from this exact
receipt. Group source aliases, close precursor/scaffold relatives, and alternate
routes to one target. Fit sampling weights on train only. Use hard constraints
for validity, supported atom states, size, charge, and construction invariants.
Use regional morphology and broad lipid organization as soft conditioning or
ranking signals so rare valid structures are retained.

Sealed BEAE structures and outcomes must remain outside corpus selection, split
fitting, architecture tuning, and checkpoint selection. They can later serve as a
downstream evaluation after freezing the base generator.

## Methods language

> We assembled 279,687 unique connected lipid constitutional graphs standardized
> as RDKit-canonical stereo-free SMILES. The corpus contained 11,267
> reported-source anchors and 268,420 exact programmatically assembled virtual
> products. A separate ledger retained all 12,675 deduplicated LNPDB identities.
> Every LNPDB identity had exact construction coverage through a bound program or
> a product-derived complete-precursor compatible decomposition. Virtual selection
> balanced reaction family, precursor roles and pairs, size, branching,
> unsaturation, and regional morphology without biological outcomes. Product-
> derived family and decomposition labels were not treated as historical-route
> claims.

## Limitations and integrity

The release has no activity, delivery, toxicity, formulation, or outcome labels;
no measured pKa, logP, solubility, stability, yield, or purchasing guarantee; no
claim that virtual products were synthesized; and no final grouped generator
split.

The verifier checks payload hashes and sizes, gzip readability, 109-field schema,
CSV/JSONL equivalence, global identity uniqueness, family/source totals, all
12,675 source identities, the 8,842/3,833 construction partition, the
11,144/1,531 model-support partition, all 4,184 decompositions and 3,833 target
dispositions, molecule/supervision order, and every precursor foreign key.

```sh
PYTHONPATH=src python3 scripts/verify_partner_lipid_corpus_v5.py \
  datasets/compose_lipid_lnpdb_complete_2026-09-13_v5
```

Export used RDKit 2025.09.6. Integrity verification establishes saved-byte,
schema, identity, and accounting consistency, not independent experimental
validation of every source annotation or virtual product.
