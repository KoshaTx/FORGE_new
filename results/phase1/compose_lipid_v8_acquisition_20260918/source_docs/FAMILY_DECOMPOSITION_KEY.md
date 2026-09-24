# Family decomposition key

Companion to `CORPUS_FOR_EXTERNAL_USE.md`. For each of the 23 reaction
families: the source literature, the bond-forming program, the role
decomposition, the authorized design knobs, and the architecture subfamilies.

The role tables were derived by perturbation, not by reading field names: each
`primary_metadata` field was altered in isolation and the role whose component
identity changed was recorded. They reflect what the enumerator actually does.
Everything else comes from the project's family architecture registry.

## What this does and does not give you

**Does:** per family — the reaction invariant (so you know which bonds were
formed and therefore where to cut), the roles and how many of each, which
metadata field names each component, the port signature of each architecture,
the knobs that were allowed to vary, and the source papers.

**Does not:** resolve component identifiers to structures. The `*_id` values
are opaque and the corpus files carry no id-to-SMILES table. Atom-level
partitioning still needs your own retrosynthetic split or substructure match —
but the reaction invariant plus the role counts tell you exactly what to look
for.

Component identity is globally consistent: the same id always means the same
molecule, including across families. One amine appearing as `amine_head` in one
family and `amine_core` in another is the same compound in a different role.

A role with count > 1 means the *same* component fills every one of those
positions. Coupled arms stay coupled; they are never independent draws. Eight
families vary that count per row — read the named field.

## Families

### `preassembled_thiol_yne_tail_amidation`

**Thiol-yne thioether lipid construction and head amidation** — 465,048 molecules in the universe.

> Double thiol addition to one alkynoic-acid scaffold followed by amidation to one complete amine; the sulfur arms remain one coupled design object.

| role | count | named by |
|---|---|---|
| `alkynoic_linker` | 1 | `linker_code` |
| `amine_head` | 1 | `head_id` |
| `thiol_tail` | 2 | `tail_code` |

Subcomponents as designed:

- complete amidating amine head
- complete alkynoic-acid/linker scaffold
- one thiol identity repeated on both sulfur arms

Authorized knobs:

- amine head scaffold and spacer
- alkynoic-acid-to-yne spacer A/B/C/D
- paired tail carbon count
- paired positional unsaturation
- paired branch partition after one or two carbons
- paired internal ester position
- reviewed crossed tail variants

Axis values present in the data:

- `tail_axis`: `branch_plus_one_unsaturation`, `single_internal_ester_position`, `positional_mono_unsaturation`, `single_branch_partition`, `linear_saturated_carbon_count`

Architecture subfamilies (2):

| id | scaffold invariant | port signature |
|---|---|---|
| `li_two_arm_glycerol_mimetic` | two thioether arms about the thiol-yne junction plus one amide-linked head | one amide-head port; two equivalent sulfur arms |
| `li_same_platform_unsaturated_head_followup` | same coupled core with source-backed unsaturation/head series | same as Li two-arm |

Must not be collapsed:

- independent left/right thiol-arm selection
- heteroatom-rich head-like tails
- changing sulfur placement without a new scaffold

Source literature:

- **Li biomimetic thiol-yne lipid library** — PMID 22902058 — doi:10.1016/j.biomaterials.2012.07.044
  - cationic thioether lipids with two hydrophobic tails of variable length and a glycerol-mimetic linker

### `thiolactone_aminolysis_michael`

**STAAR thiolactone aminolysis and Michael capture** — 334,859 molecules in the universe.

> An amine opens a complete N-acyl thiolactone, exposing a thiol that then adds to one complete acrylate in fixed order.

| role | count | named by |
|---|---|---|
| `acrylate_tail` | 1 | `acrylate_id` |
| `amine_head` | 1 | `head_id` |
| `thiolactone_region` | 1 | `thiolactone_id` |

Subcomponents as designed:

- complete amine Ai
- complete N-acyl thiolactone Bj
- complete acrylate Ck

Authorized knobs:

- clean amine heads with surviving ionizable site
- Bj acid-derived tail length/branch/unsaturation
- Ck alcohol-derived tail length/branch/unsaturation
- controlled Bj x Ck combinations

Axis values present in the data:

- `acrylate_axis`: `reported_acrylate`, `linear_tail_length`, `positional_mono_unsaturation`, `conservative_branch_partition`
- `head_axis`: `reviewed_clean_head_extension`, `reported_head`
- `thiolactone_axis`: `reported_thiolactone`, `positional_mono_unsaturation`, `linear_tail_length`, `conservative_branch_partition`

Architecture subfamilies (2):

| id | scaffold invariant | port signature |
|---|---|---|
| `staar_sequential_aibjck` | polar head + amide/thiolactone linker-tail + thioether/ester acrylate tail | three inequivalent roles; fixed two-stage order |
| `staar_permanent_cationic_reference` | source permanently charged comparison architecture | source-specific charge/port state |

Must not be collapsed:

- stage reversal
- recursive lipid-like Bj/Ck components
- amines that lose their only basic site

Source literature:

- **STAAR AiBjCk library** — PMID 40234552
  - Ai amine head, Bj N-acyl thiolactone linker-tail and Ck acrylate tail

### `ketone_isocyanide_amide`

**Miao acyclic ketone-isocyanide-secondary-amine lipids** — 271,560 molecules in the universe.

> Secondary amine, complete coupled ketone and complete isocyanide form the exact acyclic amide topology without dihydroimidazole closure.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `coupled_ketone` | 1 | `ketone_id` |
| `isocyanide` | 1 | `isocyanide_id` |

Subcomponents as designed:

- complete secondary amine
- complete coupled ketone
- complete isocyanide

Authorized knobs:

- head scaffold/spacer/N-substitution
- coupled ketone arm length/branch/unsaturation/internal ester
- isocyanide homologues only within source branch

Axis values present in the data:

- `ketone_axis`: `paired_single_internal_ester`, `paired_single_branch_partition`, `one_arm_single_internal_ester`, `one_arm_single_branch_partition`, `reported_source_ketone`

Architecture subfamilies (1):

| id | scaffold invariant | port signature |
|---|---|---|
| `miao_acyclic_secondary_amine_amide` | acyclic amide product | secondary amine prevents cyclic closure |

Must not be collapsed:

- collapsing with dihydroimidazole
- splitting ketone arms

Source literature:

- **Miao acyclic sibling branch** — PMID 31570898
  - secondary amine, ketone and isocyanide form the source acyclic amide branch

### `disulfide_michael`

**Disulfide-bridged ester Michael lipids** — 254,389 molecules in the universe.

> Complete SCC/disulfide acceptors add at explicitly selected non-amide N-H sites; the acceptor linkage and repeated arms remain coupled.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `disulfide_acceptor` | `occupancy` (varies) | `arm_id` |

Subcomponents as designed:

- complete amine/head core
- one complete SCC acceptor repeated at declared sites

Authorized knobs:

- head topology within 1/2/3-site occupancy
- distal tail length/branch/unsaturation/internal ester
- acrylate-to-disulfide spacing
- disulfide-to-oxygen spacing

Axis values present in the data:

- `head_axis`: `reported_disulfide_head`

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `single_site_ldil` | one SCC arm | one selected N-H site |
| `multi_site_symmetric_ldil` | multiple identical SCC arms | 2-3 mapped equivalent/declared sites |
| `four_arm_4a3_scc` | one 4A3 head plus four disulfide-bridged linkers/tails | four declared ports |

Must not be collapsed:

- generic disulfide coupling
- independent repeated arms
- adding at amide or tertiary nitrogens

Source literature:

- **LDIL disulfide-linker library** — PMID 37853662 — doi:10.1021/jacs.3c09143
  - 96 linker-degradable lipids; amino headgroups plus repeated SCC disulfide-ester acceptors

### `acid_epoxide_diester_multistep`

**Acid/epoxide staged 1,2-diester lipids** — 232,240 molecules in the universe.

> Complete amine-bearing acid, complete terminal epoxide and complete hydrophobic acid follow the bound staged ring-opening/esterification program.

| role | count | named by |
|---|---|---|
| `amine_acid_head` | 1 | `head_id` |
| `epoxide` | 1 | `epoxide_id` |
| `hydrophobic_acid` | 1 | `hydrophobic_acid_id` |

Subcomponents as designed:

- complete amine-bearing acid
- complete terminal epoxide
- complete hydrophobic acid

Authorized knobs:

- amine-bearing acid homologues
- epoxide body length/branch/unsaturation/internal ester and ring-to-body spacing
- acid body length/branch/unsaturation/internal ester
- mono/diene and controlled crosses

Axis values present in the data:

- `epoxide_axis`: `linear_total_carbon_length`, `ester_plus_positional_monoene`, `branch_plus_positional_monoene`, `positional_diene`, `branch_plus_internal_ester`, `single_internal_ester_position`, `positional_mono_unsaturation`, `single_branch_partition`
- `head_axis`: `dimethylamino_acid_spacer`, `diethylamino_acid_spacer`, `reported_source_head_acid`
- `hydrophobic_acid_axis`: `ester_plus_positional_monoene`, `linear_total_carbon_length`, `branch_plus_positional_monoene`, `positional_diene`, `branch_plus_internal_ester`, `single_internal_ester_position`, `single_branch_partition`, `positional_mono_unsaturation`

Architecture subfamilies (1):

| id | scaffold invariant | port signature |
|---|---|---|
| `acid_epoxide_site_ordered_12_diester` | retained ionizable acid region plus epoxide-derived and acid-derived hydrophobes | three roles with fixed site order |

Must not be collapsed:

- generic amine-epoxide replay
- unbound site order
- head on wrong side of the amine-bearing acid

Source literature:

- **Advanced Healthcare Materials acid-epoxide diester library** — PMID 37990414 — doi:10.1002/adhm.202302691
  - amine-bearing acid, terminal epoxide and hydrophobic acid in a staged construction

### `aldehyde_ugi4`

**Aldehyde Ugi four-component lipids** — 222,029 molecules in the universe.

> One complete amine, aldehyde, acid and isocyanide form the aldehyde-derived Ugi bis-amide connectivity.

| role | count | named by |
|---|---|---|
| `aldehyde` | 1 | _positional in `precursor_ids`_ |
| `amine` | 1 | _positional in `precursor_ids`_ |
| `carboxylic_acid` | 1 | _positional in `precursor_ids`_ |
| `isocyanide` | 1 | _positional in `precursor_ids`_ |

Subcomponents as designed:

- complete amine
- complete aldehyde
- complete monocarboxylic acid
- complete isocyanide

Authorized knobs:

- head scaffold/spacer/N-substitution
- each hydrophobe length/branch/unsaturation
- aldehyde and acid internal-ester spacing
- isocyanide alkyl/ring/ester homologues
- source-parented one-role grids
- closed two-role squares

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `bowen_4cr_ordinary` | ordinary retained head plus two coherent hydrophobic regions around Ugi core | four inequivalent precursor ports |
| `lumi_ordinary_source_parented` | same Ugi core with ordinary component-role organization | four inequivalent precursor ports |
| `lumi_specialized_fixed` | source-specialized aromatic, sulfur, N-N or polycyclic region held fixed | four inequivalent ports; specialized port frozen |

Must not be collapsed:

- role interchange
- changing to or from specialized components in an ordinary lane
- calling a one-role replacement local without a role-specific analogue edge

Source literature:

- **Bowen Li Nature Materials 4CR library** — PMID 38740955 — doi:10.1038/s41563-024-01867-3
  - 584 assayed lipids and a 40,000-member 40x10x10x10 virtual library
- **LUMI released reagent grid**
  - complete amine, aldehyde, acid and isocyanide tables; virtual reagent contexts are not all experimental products

### `alpha_isocyanoester_dihydroimidazole`

**Alpha-isocyanoester dihydroimidazole lipids** — 219,036 molecules in the universe.

> Primary amine, complete coupled ketone and alpha-isocyanoester close to the source dihydroimidazole ring.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `coupled_ketone` | 1 | `ketone_id` |
| `isocyanide` | 1 | `isocyanide_id` |

Subcomponents as designed:

- complete primary amine
- complete coupled ketone
- fixed/reviewed alpha-isocyanoester

Authorized knobs:

- head spacer/ring/hydroxylation
- coupled ketone arm length/branch/unsaturation/internal ester
- isocyanoester substituent only after explicit review

Axis values present in the data:

- `head_axis`: `cyclic_amine_spacer`, `diethylamino_spacer`, `dimethylamino_spacer`
- `ketone_axis`: `paired_single_internal_ester`, `paired_single_branch_partition`, `one_arm_single_internal_ester`, `one_arm_single_branch_partition`, `reported_source_ketone`

Architecture subfamilies (1):

| id | scaffold invariant | port signature |
|---|---|---|
| `miao_cyclic_dihydroimidazole` | dihydroimidazole ring with coupled ketone-derived substituents | three roles; primary-amine ring closure |

Must not be collapsed:

- mixing with acyclic Miao branch
- splitting ketone arms
- secondary amine closure

Source literature:

- **Miao iLY1809/dihydroimidazole library** — PMID 31570898 — doi:10.1038/s41587-019-0247-3
  - amine, coupled ketone and alpha-isocyanoester form cyclic dihydroimidazole products

### `amine_alkylation`

**Amine N-alkylation lipids** — 188,272 molecules in the universe.

> Explicit non-amide amine sites undergo N-alkylation by complete alkyl electrophiles with exact event count and charge state.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `bromoester_arm` | 2 | `arm_id` |

Subcomponents as designed:

- complete amine head
- complete alkyl/bromoester electrophile

Authorized knobs:

- head scaffold/site/occupancy
- tail length
- ester position/orientation
- branch point and size
- positional unsaturation
- reviewed branch x unsaturation x ester crosses

Axis values present in the data:

- `arm_axis`: `positional_mono_unsaturation`, `conservative_branch_partition`
- `head_axis`: `reported_source_head`

Architecture subfamilies (2):

| id | scaffold invariant | port signature |
|---|---|---|
| `ren_bromoester_n_alkylation` | amine head plus complete bromoester hydrophobes | explicit N site; other Ns nonreactive/tertiary |
| `reference_clinical_n_alkylation` | source-specific N-alkylation scaffold | exact site and event count |

Must not be collapsed:

- O-alkylation
- reaction at tertiary N unless quaternary source stratum
- pooling vitamin-B5 multistep products

Source literature:

- **Ren bromoester-tail library** — PMID 39099464
  - amine heads N-alkylated with complete brominated ester tails

### `iphos_ring_opening`

**iPhos dioxaphospholane ring opening** — 184,345 molecules in the universe.

> Explicit amine sites open complete alkylated dioxaphospholanes while conserving phosphate connectivity, charge and source occupancy.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `phosphate_tail` | 1 | `phosphate_id` |

Subcomponents as designed:

- complete amine with any pre-existing hydrophobic substituent
- complete alkylated dioxaphospholane

Authorized knobs:

- amine-side hydrophobe length/branch/unsaturation/internal ester
- P-side hydrophobe length/branch/unsaturation/internal ester
- nitrogen class and exact x occupancy

Axis values present in the data:

- `head_axis`: `reported_amine`, `positional_mono_unsaturation`, `conservative_branch_partition`, `single_internal_ester`, `linear_arm_length`
- `phosphate_axis`: `reported_phosphate_tail`, `positional_mono_unsaturation`, `conservative_branch_partition`, `single_internal_ester`, `linear_arm_length`

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `iphos_single_zwitterion_nA1Pm` | one Pm on one amine; source tail count depends on nA and Pm | one charged attachment port |
| `iphos_multi_zwitterion_nAxPm` | multiple Pm units on one polyamine | multiple mapped charged ports |
| `iphos_pretailed_amine_architectures` | amine already supplies one or more hydrophobic substituents | pre-existing tail plus x Pm-derived regions |

Must not be collapsed:

- collapsing x values
- generic P-containing lipid family
- altering phosphate connectivity

Source literature:

- **iPhos nAxPm library** — PMID 33542471 — doi:10.1038/s41563-020-00886-0
  - 28 amines nA x 13 alkylated dioxaphospholanes Pm; x records P units per amine

### `passerini_3cr`

**Passerini three-component ionizable lipids** — 128,352 molecules in the universe.

> One amine-bearing carboxylic acid, one aldehyde and one isocyanide form the Passerini alpha-acyloxy amide core; the acid component must retain a distinct ionizable head.

| role | count | named by |
|---|---|---|
| `aldehyde_tail` | 1 | `aldehyde_id` |
| `amine_acid_head` | 1 | `head_id` |
| `isocyanide_tail` | 1 | `isocyanide_id` |

Subcomponents as designed:

- complete amine-bearing carboxylic-acid head
- complete aldehyde tail
- complete isocyanide tail

Authorized knobs:

- head scaffold within acid-head grammar
- tail length C10-C18
- branching
- positional unsaturation
- additional ester position
- controlled two-tail asymmetry

Axis values present in the data:

- `aldehyde_axis`: `linear_tail_length`, `branch_plus_positional_monoene`, `ester_plus_positional_monoene`, `positional_diene`, `positional_mono_unsaturation`, `conservative_branch_partition`, `single_internal_ester_position`, `reported_aldehyde_tail`
- `head_axis`: `reported_head_acid`, `cyclic_amine_acid_spacer`, `diethylamino_acid_spacer`, `dimethylamino_acid_spacer`
- `isocyanide_axis`: `linear_tail_length`, `branch_plus_positional_monoene`, `positional_diene`, `positional_mono_unsaturation`, `conservative_branch_partition`, `reported_isocyanide_tail`

Architecture subfamilies (1):

| id | scaffold invariant | port signature |
|---|---|---|
| `passerini_amine_acid_head_dual_tail` | amine-bearing acid head plus aldehyde and isocyanide hydrophobes | three inequivalent roles |

Must not be collapsed:

- old candidates whose only nitrogen becomes an amide
- ordinary carboxylic acid without retained basic head
- Ugi/Passerini collapse

Source literature:

- **P-3CR biodegradable ionizable lipid library**
  - 144 lipids from amine-bearing carboxylic-acid heads, aldehydes and isocyanides; tails include C10-C18, branch, unsaturation and ester motifs

### `amine_epoxide_opening`

**Amine opening of epoxides** — 97,439 molecules in the universe.

> One or more explicitly selected amine sites open complete epoxides with mapped regioisomer and occupancy.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `epoxide_tail` | `occupancy` (varies) | `tail_id` |

Subcomponents as designed:

- complete amine/core
- complete terminal or glycidyl epoxide

Authorized knobs:

- head/site/occupancy
- C6-C18 body length
- branch position/partition
- positional unsaturation
- glycidyl internal ester position
- reviewed crosses

Axis values present in the data:

- `tail_axis`: `positional_diene`, `branch_plus_positional_monoene`, `positional_mono_unsaturation`, `single_internal_ester_position`, `linear_total_carbon_length`

Architecture subfamilies (4):

| id | scaffold invariant | port signature |
|---|---|---|
| `anderson_ordinary_epoxide_lipidoid` | polyamine plus repeated alkyl epoxide arms | source-declared N-site occupancy |
| `paracyclophane_three_arm_epoxide` | preassembled paracyclophane diamide with repeated epoxide-derived arms | three declared ports |
| `han_db_aminoalcohol_intermediate` | amine plus two repeated body epoxides | mapped N sites and two beta-OH outputs |
| `hepes_or_source_specific_epoxide_core` | fixed heterocyclic/polar core with epoxide arms | source-specific sites/charge |

Must not be collapsed:

- all amines in one pool
- site/occupancy inference
- regioisomer collapse

Source literature:

- **Anderson epoxide-derived lipidoid library** — PMID 20080679 — doi:10.1073/pnas.0910603106
  - amine-containing monomers x epoxide-terminated alkyl chains
- **Paracyclophane Ep series** — PMID 39424104
  - paracyclophane core with repeated epoxide-derived arms
- **Han DB aminoalcohol intermediates** — PMID 38409275
  - first stage of the two-stage DB/rocket architecture

### `ketone_ugi4`

**Ketone Ugi four-component lipids** — 91,395 molecules in the universe.

> One complete amine, one indivisible complete ketone, one acid and one isocyanide form the ketone-Ugi product; the amine must retain a distinct ionizable nitrogen.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `carboxylic_acid` | 1 | `acid_code` |
| `coupled_ketone` | 1 | `ketone_code` |
| `isocyanide` | 1 | `isocyanide_code` |

Subcomponents as designed:

- complete amine head
- complete coupled ketone
- complete acid
- complete isocyanide

Authorized knobs:

- amine head/spacer
- paired ketone-arm length/branch/unsaturation/internal ester and carbonyl spacing
- acid interface/fatty/degradable modes
- isocyanide homologues

Architecture subfamilies (4):

| id | scaffold invariant | port signature |
|---|---|---|
| `linear_coupled_ketone_arms` | complete ketone with two coupled acyclic arms | one ketone port encoding two arms |
| `unsaturated_or_branched_coupled_ketone` | complete coupled ketone with paired or explicit asymmetric morphology | one coupled ketone port |
| `cyclic_coupled_ketone` | complete cyclic/polycyclic hydrophobic ketone | one coupled ketone port |
| `interface_acyl_vs_hydrophobic_acyl` | same Ugi core but acid supplies interface acyl or a hydrophobic/degradable region | acid-role mode label required |

Must not be collapsed:

- splitting ketone arms
- cross-role rescue of an amine that loses its only basic site
- treating interface acetate and long fatty acid as local neighbours

Source literature:

- **LiON ketone-4CR library** — PMID 39658727
  - primary amine, complete ketone, carboxylic acid and isocyanide

### `vitamin_b5_multistep`

**Vitamin-B5 site-ordered multistep lipids** — 80,398 molecules in the universe.

> D-pantothenic acid is the fixed stereodefined core; each source series preserves primary/secondary alcohol site order and final head ester/amide identity.

| role | count | named by |
|---|---|---|
| `series_head` | 1 | `head_id` |
| `series_tail_design` | 1 | `series`, `tail_axis` |

Subcomponents as designed:

- fixed vitamin-B5 core
- series-specific complete tail alcohol/acid inputs
- series-specific complete head acid/amine/alcohol

Authorized knobs:

- conservative hydrophobe length/branch/unsaturation after site-order review
- same-series head homologues
- no cross-series Cartesian expansion

Axis values present in the data:

- `head_axis`: `same_head_scaffold_spacer`, `reported_head_alcohol`, `reported_head_amine`, `reported_head_acid`
- `tail_axis`: `reported_tail6`

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `vitamin_b5_i7_two_event` | B5 carboxyl-tail ester then primary-O head ester | two ordered sites |
| `vitamin_b5_i8_three_ester` | tail alcohol at B5 acid plus head acid primary-O plus tail acid secondary-O | three ordered ester sites |
| `vitamin_b5_i9_dual_tail_head_linkage` | two source-ordered B5 O-acyl tails then head ester or amide | three ordered events; head linkage subtype retained |

Must not be collapsed:

- I7/I8/I9 pooling
- erasing retained R stereochemistry
- cross-series precursor exchange

Source literature:

- **KIST I71-I98 vitamin-B5 library** — PMID 39502027 — doi:10.1002/adhm.202403366
  - 17 lipids in three protected-stage source series around D-pantothenic acid

### `a3_amine_aldehyde_alkyne`

**A3 amine-aldehyde-alkyne lipids** — 70,016 molecules in the universe.

> Each A3 event joins an explicitly selected amine site, aldehyde and terminal alkyne; multi-event programs preserve event count and order.

| role | count | named by |
|---|---|---|
| `aldehyde` | 1 | `aldehyde_id` |
| `alkyne` | 1 | `alkyne_id` |
| `amine_head` | 1 | `head_id` |

Subcomponents as designed:

- complete amine
- complete aldehyde
- complete terminal alkyne or prepared coupled alkyne

Authorized knobs:

- head exchange within exact site contract
- aldehyde series
- alkyne length/branch/positional unsaturation/internal ester
- single versus ordered two-event designs

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `lu_single_event_formaldehyde` | one A3 event using formaldehyde | one selected N site; one alkyne |
| `han_symmetric_a3` | symmetric A3 exterior with biodegradable alkyne options | source-declared repeated ports |
| `han_ordered_two_event_asymmetric` | two ordered A3 events using long-chain aldehydes and asymmetric exteriors | two ordered, nonequivalent event ports |

Must not be collapsed:

- half-built starting products
- single/two-event collapse
- unreviewed cross-paper site assignments

Source literature:

- **Lu A3 high-throughput library** — PMID 39497197
  - 28+2 amines x 21 alkynes with formaldehyde; 623 synthesized lipids
- **Han directed chemical evolution A3 libraries 1-5** — PMID 39578640 — doi:10.1038/s41551-024-01267-7
  - symmetric, biodegradable, asymmetric long-chain-aldehyde, tail and head optimization libraries

### `maleate_addition`

**Amine or thiol addition to dialkyl maleates** — 68,463 molecules in the universe.

> One mapped amine or thiol addition consumes the maleate alkene; the two ester arms of each maleate are one complete design object.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `maleate` | 1 | `arm_id` |

Subcomponents as designed:

- complete amine or thiol nucleophile
- complete dialkyl maleate

Authorized knobs:

- head/site count
- paired tail length
- paired branch position/size
- paired positional unsaturation
- controlled symmetric/asymmetric complete maleates

Axis values present in the data:

- `arm_axis`: `positional_mono_unsaturation`, `conservative_branch_partition`, `single_internal_ester`, `linear_arm_length`, `reported_symmetric_maleate`
- `head_axis`: `reported_amine_head`, `reported_thiol_head`

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `amine_symmetric_maleate` | amine addition to symmetric dialkyl maleate | one nucleophile port; coupled equivalent maleate arms |
| `thiol_symmetric_maleate` | thiol addition to symmetric dialkyl maleate | one sulfur port; coupled equivalent arms |
| `asymmetric_maleate_orientations` | complete asymmetric maleate with explicit addition orientation | coupled nonequivalent arms; positional product label |

Must not be collapsed:

- amine/thiol pooling
- splitting maleate arms
- unlabeled asymmetric regioisomers

Source literature:

- **plug-and-play maleate IL library** — PMID 40060499
  - 50 amines/thiols x 10 complete dialkyl maleates = 500 lipids

### `reductive_amination`

**Aliphatic reductive amination lipids** — 61,320 molecules in the universe.

> One selected amine site and complete aliphatic aldehyde form the reduced C-N connection with exact oxygen loss accounting.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `coupled_aldehyde` | 1 | `aldehyde_id` |

Subcomponents as designed:

- complete amine
- complete aliphatic lipid aldehyde

Authorized knobs:

- head scaffold/spacer/site
- aldehyde-tail length/branch/positional unsaturation
- internal ester position and spacing
- controlled asymmetry

Axis values present in the data:

- `tail_axis`: `paired_positional_diene`, `paired_linear_acyl_tail_length`

Architecture subfamilies (1):

| id | scaffold invariant | port signature |
|---|---|---|
| `jiang_nebulized_ester_aldehyde` | amine head plus ester-containing lipid aldehyde | one selected N site and one aldehyde |

Must not be collapsed:

- aromatic aldehydes
- nested whole-lipid aldehydes
- PMID 39658727 LiON bibliographic alias conflict

Source literature:

- **Jiang nebulized pulmonary reductive-amination library** — PMID 37985700 — doi:10.1038/s41565-023-01548-3
  - amines plus ester-containing lipid aldehydes

### `aryl_reductive_amination`

**Aryl-core reductive amination lipids** — 48,090 molecules in the universe.

> Source aryl aldehyde/amine systems form mapped reduced C-N linkages while retaining their complete aryl ester cores and coupled arms.

| role | count | named by |
|---|---|---|
| `amine_head` | 1 | `head_id` |
| `coupled_aldehyde` | 1 | `aldehyde_id` |

Subcomponents as designed:

- complete amine
- complete preassembled aryl aldehyde core

Authorized knobs:

- head series
- coupled arm length
- coupled branch morphology
- coupled positional unsaturation

Axis values present in the data:

- `tail_axis`: `identical_positional_diene`

Architecture subfamilies (2):

| id | scaffold invariant | port signature |
|---|---|---|
| `xue_a2_aryl_core` | A2 aryl substitution/core topology | source-declared aldehyde and coupled arm ports |
| `xue_a3_aryl_core` | A3 aryl substitution/core topology | source-declared aldehyde and coupled arm ports |

Must not be collapsed:

- pooling with aliphatic reductive amination
- independent coupled-arm selection
- changing aryl substitution as a local tail knob

Source literature:

- **Xue aryl-core reductive-amination library** — PMID 38424061
  - source aryl aldehyde cores and amine heads

### `epoxide_opening_o_acylation`

**Epoxide opening followed by O-acylation** — 42,805 molecules in the universe.

> Mapped amine-epoxide opening creates beta-OH sites, then each declared OH is O-acylated with a complete acyl precursor in source order.

| role | count | named by |
|---|---|---|
| `acyl_tail` | 2 | `acyl_tail_id` |
| `amine_head` | 1 | `head_id` |
| `epoxide_tail` | 2 | `epoxide_tail_id` |

Subcomponents as designed:

- complete amine
- one repeated body epoxide
- one repeated O-acyl hydrophobe

Authorized knobs:

- head/site/occupancy
- body-tail C6-C18 length/branch/unsaturation
- acyl-tail C6-C18 length/branch/unsaturation
- selected internal ester variants
- controlled body x acyl crosses

Axis values present in the data:

- `acyl_axis`: `branch_plus_positional_monoene`, `linear_total_carbon_length`, `positional_mono_unsaturation`
- `epoxide_axis`: `branch_plus_positional_monoene`, `linear_total_carbon_length`, `positional_mono_unsaturation`

Architecture subfamilies (1):

| id | scaffold invariant | port signature |
|---|---|---|
| `han_db_two_stage_four_arm_rocket` | two repeated epoxide-derived bodies and two repeated acyl tails around one head | mapped N ports followed by mapped O ports |

Must not be collapsed:

- calling final products simple epoxide lipidoids
- N-acylation
- mixing repeated arms independently

Source literature:

- **Han DB four-arm rocket series** — PMID 38409275
  - amine-epoxide aminoalcohol intermediate followed by O-acylation

### `aza_michael_acrylate`

**Aza-Michael acrylate lipids** — 33,619 molecules in the universe.

> Mapped amine sites add to complete acrylates under a source-subseries-specific occupancy and prepared-tail contract.

| role | count | named by |
|---|---|---|
| `acceptor_tail` | `occupancy` (varies) | `tail_id` |
| `amine_head` | 1 | `head_id` |

Subcomponents as designed:

- complete amine or paper-specific core
- complete acrylate acceptor; may itself be a prepared multiarm object

Authorized knobs:

- O9-O18 and paper-specific length series
- branch/unsaturation/internal ester within each acceptor grammar
- head/site/occupancy within subseries
- controlled crossed variants

Axis values present in the data:

- `tail_axis`: `branch_plus_positional_monoene`, `positional_mono_unsaturation`, `linear_saturated_body_length`

Architecture subfamilies (5):

| id | scaffold invariant | port signature |
|---|---|---|
| `lipidoid_2008_o_series_full_occupancy` | source amine plus repeated alkyl acrylate | mapped N ports; repeated acceptor |
| `paracyclophane_three_arm_acrylate` | preassembled paracyclophane diamide core with three repeated arms | three source-declared ports |
| `ricinoleic_complete_acrylate` | complete ricinoleic-derived acrylate carries its own hydrophobe/interface | paper-specific mapped ports |
| `multiarm_dendron_acrylate` | complete multiarm degradable acrylate tail object | one acceptor object encodes multiple arms |
| `akinc_or_whitehead_ordinary_acrylate` | source amine and paper-specific ordinary acrylate occupancy | explicit N sites |

Must not be collapsed:

- global amine/acrylate Cartesian pooling
- fragmenting preassembled acceptors
- assuming all N-H sites react

Source literature:

- **2008 O-series lipidoids** — PMID 18438401 — doi:10.1038/nbt1402
  - amine number + acrylate O-code + tail count
- **Paracyclophane Acr series** — PMID 39424104
  - preassembled paracyclophane core plus repeated acrylate arms
- **ricinoleic/pulmonary acrylate libraries**
  - paper-specific complete acrylate tails
- **multiarm dendron acrylate library** — PMID 39742515
  - preassembled multiarm degradable acrylates plus amine cores

### `aza_michael_acrylamide`

**Aza-Michael acrylamide lipidoids** — 31,239 molecules in the universe.

> Mapped amine sites add to complete acrylamide acceptors, preserving the beta-amino-amide interface and exact occupancy.

| role | count | named by |
|---|---|---|
| `acceptor_tail` | `occupancy` (varies) | `tail_id` |
| `amine_head` | 1 | `head_id` |

Subcomponents as designed:

- complete amine core
- one complete acrylamide identity repeated at mapped sites

Authorized knobs:

- head topology within occupancy
- N9-N18 tail homologues
- branching
- positional unsaturation
- internal ester where compatible
- crosses after one-axis qualification

Axis values present in the data:

- `tail_axis`: `branch_plus_positional_monoene`, `positional_mono_unsaturation`, `linear_saturated_body_length`

Architecture subfamilies (2):

| id | scaffold invariant | port signature |
|---|---|---|
| `lipidoid_2008_n_series_full_occupancy` | source amine with source maximum or reported acrylamide occupancy | 1-7 mapped N-addition ports; repeated acceptor |
| `partial_occupancy_acrylamide` | same core at explicitly lower occupancy | explicit subset of N ports |

Must not be collapsed:

- pooling acrylate and acrylamide
- unrecorded occupancy
- independent repeated-tail choices

Source literature:

- **2008 N-series lipidoids** — PMID 18438401 — doi:10.1038/nbt1402
  - amine number + acrylamide N-code + tail count; 1-7 additions depending on amine

### `o_esterification`

**Aminoalcohol O-esterification** — 24,948 molecules in the universe.

> Mapped alcohol sites on one complete aminoalcohol are esterified with one complete monocarboxylic-acid identity while the amine topology is retained.

| role | count | named by |
|---|---|---|
| `acid_tail` | 1 | `acid_code` |
| `aminoalcohol_head` | 1 | `head_id` |

Subcomponents as designed:

- complete aminoalcohol head
- one complete acid body repeated at selected OH sites

Authorized knobs:

- head homologues only within fixed OH occupancy
- tail C8-C18
- double-bond count and position
- branch point/size with fixed total carbon
- branch plus unsaturation
- internal ester position and spacing

Axis values present in the data:

- `acid_axis`: `branch_plus_one_unsaturation`, `positional_mono_unsaturation`, `single_branch_partition`, `linear_saturated_carbon_count`

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `h1_single_oh_single_tail` | one esterifiable OH | one O-acyl port |
| `h2_diol_two_repeated_tails` | two esterifiable OH groups | two equivalent O-acyl ports |
| `h3_triol_three_repeated_tails` | three esterifiable OH groups | three equivalent O-acyl ports |

Must not be collapsed:

- H1/H2/H3 as local head analogues
- mixed acids across equivalent OH sites
- lipid-like multiarm acids used as one tail

Source literature:

- **enzyme-catalysed aminoalcohol/fatty-acid lipid library** — PMID 36269150
  - complete aminoalcohol heads esterified with coherent fatty-acid bodies

### `aldehyde_ugi3`

**Aldehyde Ugi three-component lipids** — 18,842 molecules in the universe.

> One complete primary amine, aldehyde and isocyanide form the exact three-component Ugi amide product.

| role | count | named by |
|---|---|---|
| `aldehyde` | 1 | _positional in `precursor_ids`_ |
| `amine` | 1 | _positional in `precursor_ids`_ |
| `isocyanide` | 1 | _positional in `precursor_ids`_ |

Subcomponents as designed:

- complete primary amine head
- complete aldehyde hydrophobe
- complete isocyanide hydrophobe

Authorized knobs:

- head class/spacer
- aldehyde tail length/branch/unsaturation/internal ester
- isocyanide tail length/branch/ring
- controlled tail asymmetry
- one- then two-coordinate grids

Architecture subfamilies (2):

| id | scaffold invariant | port signature |
|---|---|---|
| `agile_biodegradable_ugi3` | amine head plus biodegradable aldehyde tail and isocyanide tail | three inequivalent ports |
| `bowen_pnas_dual_tail_ugi3` | head plus tail A plus tail B with its own codebook | three inequivalent ports |

Must not be collapsed:

- automatic AGILE/PNAS cross-grid
- using a ketone in the aldehyde family

Source literature:

- **AGILE Ugi-3CR library** — PMID 39060305 — doi:10.1038/s41467-024-50619-z
  - 20 experimental heads x 12 biodegradable aldehyde chains x 5 isocyanide chains; 60,000 virtual pretraining library
- **Bowen Li PNAS muscle-selective 3CR library** — PMID 38060560 — doi:10.1073/pnas.2309472120
  - amine head, aldehyde tail and isocyanide tail

### `aema_aza_thiol_addition`

**AEMA aza-addition followed by thiol addition** — 13,865 molecules in the universe.

> A polyamine receives its declared AEMA occupancy, then the remaining methacrylate ports receive one repeated complete thiol identity.

| role | count | named by |
|---|---|---|
| `amine_core` | 1 | `core_id` |
| `thiol_periphery` | 1 | `thiol_id` |

Subcomponents as designed:

- complete polyamine core
- fixed AEMA interface repeated at mapped N-H sites
- one complete thiol repeated over mapped periphery

Authorized knobs:

- core topology within occupancy stratum
- thiol length
- branching only where body length supports it
- positional unsaturation
- internal ester position
- reviewed cross-paper core/periphery combinations

Axis values present in the data:

- `thiol_axis`: `positional_mono_unsaturation`, `single_internal_ester_position`, `reported_source_periphery`, `missing_linear_homologue`

Architecture subfamilies (3):

| id | scaffold invariant | port signature |
|---|---|---|
| `g1_exact_occupancy` | source polyamine with exact AEMA occupancy and repeated thiol periphery | mapped AEMA ports; identical peripheral arms |
| `higher_generation_aema` | higher-generation core and source-declared occupancy | different port count/topology |
| `unsaturated_periphery_aema` | same staged core with source unsaturated thiols | fixed core/occupancy; unsaturated repeated arms |

Must not be collapsed:

- maximum-NH occupancy inference
- independent peripheral arm mixing
- cross-paper combinations without exact occupancy replay

Source literature:

- **Zhou AEMA degradable dendrimer library** — PMID 26729861 — doi:10.1073/pnas.1520756113
  - 42 cores x 36 peripheries via ordered aza-Michael/AEMA then thiol-Michael chemistry
- **AEMA unsaturated follow-up** — PMID 33305471 — doi:10.1002/anie.202013927
  - source unsaturated periphery series

## Variable-multiplicity families

Read the named field per row rather than assuming a fixed count:

- `disulfide_michael` — `occupancy`
- `iphos_ring_opening` — `event_count`
- `amine_epoxide_opening` — `occupancy`
- `a3_amine_aldehyde_alkyne` — `events`
- `maleate_addition` — `occupancy`
- `aza_michael_acrylate` — `occupancy`
- `aza_michael_acrylamide` — `occupancy`
- `o_esterification` — `occupancy`

## The two Ugi families

`aldehyde_ugi3` and `aldehyde_ugi4` carry `precursor_ids` as an ordered list
rather than named fields:

- `aldehyde_ugi3`: `[amine, aldehyde, isocyanide]`
- `aldehyde_ugi4`: `[amine, aldehyde, carboxylic_acid, isocyanide]`

See `CORPUS_FOR_EXTERNAL_USE.md` section 4.1 on `aldehyde_ugi4` provenance
before publishing from that family.

