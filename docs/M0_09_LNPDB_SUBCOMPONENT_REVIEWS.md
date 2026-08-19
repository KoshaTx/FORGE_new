# M0-09 LNPDB Subcomponent Route Reviews

## Current outcome

Four source-level chemistry reviews are now complete: Li et al., Nature
Biotechnology (2023), AGILE, Chen et al., PNAS (2023), and LX_2024. Summed per
source, the reviewed packages cover 170 normalized component records, 45 L2
route instances, 76 L2 reaction instances, and five L1 assembly families.

Of the 45 L2 route instances, 44 prepare an actual Ugi component and one
prepares the shared precursor used by the Li et al. non-Ugi platform.
Forty-four of the 45 are structure-resolved; the unresolved instance is the
AGILE B5 source-to-measured mapping conflict. These are evidence counts, not
route-complete component counts. Procurement and recursive terminal closure
remain separate.

The first review, Li et al., covers 81 normalized LNPDB components: 70 heads,
one linker, and ten tails.

The source does not provide 80 upstream component routes. It provides:

- one exact five-step L2 route for a shared activated ricinoleic acrylate
  precursor;
- one L1 procedure that couples that precursor to ten purchased alcohol tails;
- one L1 aza-Michael procedure that produces the final 720-member lipid
  library;
- zero upstream syntheses of the ten alcohol tails;
- zero upstream syntheses of the 72 source-reported amine heads.

This distinction is material. Counting the two library procedures as
subcomponent synthesis would incorrectly turn assembly coverage into route
coverage.

## Exact L2 route recovered

The supplementary information reports the following route from methyl
(Z)-12-hydroxyoctadec-9-enoate to the activated ricinoleic acrylate precursor.

| Step | Transformation | Key reagents | Conditions | Isolated yield |
|---|---|---|---|---:|
| 1 | Protect the secondary alcohol as TBDMS ether | TBDMSCl, imidazole | DCM, room temperature, overnight, argon | 88% |
| 2 | Reduce the methyl ester to a primary alcohol | LiAlH4 | THF, 0 degrees C to room temperature, overnight, argon | 77% |
| 3 | Convert the primary alcohol to an acrylate | Acryloyl chloride, DIPEA | DCM, 0 degrees C to room temperature, overnight, argon | 71% |
| 4 | Remove the TBDMS group | 1% HCl | Methanol, 0 degrees C to room temperature, overnight | 63% |
| 5 | Activate the secondary alcohol as a 4-nitrophenyl carbonate | 4-Nitrophenyl chloroformate, pyridine | DCM, room temperature, overnight, argon | 81% |

The validated record includes the reactant and product structure for every
step. Consecutive structures are checked for exact continuity, and the first and
last structures are checked against the declared starting material and target.

## Heads and tails

Supplementary Figure S1 reports 72 amine headgroups and ten alcohol tails. The
experimental section states that the amines came from four commercial vendors
and that all other starting materials were commercially sourced. It does not
provide exact vendor and catalog identifiers for individual heads or tails.

The correct evidence states are therefore:

| Component group | Structures in LNPDB | Upstream routes in this source | Procurement evidence |
|---|---:|---:|---|
| Amine heads | 70 normalized from 72 source structures | 0 | Historical blanket vendor claim |
| Alcohol tails | 10 | 0 | Historical blanket vendor claim |
| Shared ricinoleate precursor | One linker-related core | One five-step family | Historical blanket claim for its starting material |

A historical purchase statement is useful provenance, but it is not current
procurement closure. Exact L3 closure still requires a time-stamped item match
or an internally stocked material.

## L1 procedures kept separate

General Procedure A reacts the activated ricinoleate precursor with each of ten
alcohol tails using DIPEA and DMAP in DCM. The ten isolated product yields range
from 61% to 84%. These products are linker-tail subassemblies, not syntheses of
the original alcohol tails.

General Procedure B performs the final amine aza-Michael addition at 90 degrees
C for two to three days. It defines the final lipid assembly family and reports
a 77% yield for one representative product. It is not L2 supervision.

## Source reconciliation findings

Two tail assignments remain explicitly qualified:

- The source specifies Z stereochemistry for the oleyl member, while the LNPDB
  tail string omits that stereochemistry.
- The linoleyl member has a conflict among the source name, source locants, and
  the cumulene-like LNPDB string.

The source figure contains 72 head structures, while LNPDB contains 70
normalized head structures for this PMID. The route record preserves both
counts and does not silently infer why two structures collapsed.

## Native Ugi corroboration from JC_2023

Chen et al., PMID `38060560`, reports a 16 by 6 by 3 Ugi library containing
16 amine heads, six aldehyde tails, and three isocyanide tails. Its supporting
information adds four structure-resolved L2 route instances and one isolated
final Ugi product.

| Component or product | Route | Evidence recovered |
|---|---|---|
| 6-Oxohexyl 4-methylnonanoate | EDC and DMAP esterification, then Dess-Martin oxidation | Product-specific two-step route, 86% and 61% reported step yields, proton NMR |
| 1-Isocyanoundecane | Primary amine formylation, then formamide dehydration | Exact product structure and proton NMR; product-specific quantities and yield not reported |
| 1-Isocyanooctadecane | Primary amine formylation, then formamide dehydration | Exact product structure and proton NMR; product-specific quantities and yield not reported |
| (Z)-1-Isocyanooctadec-9-ene | Oleylamine formylation, then formamide dehydration | Product-specific conditions, 35% final isolated yield, proton NMR |
| iso-A11B5C1 | Phenylphosphinic-acid-catalyzed Ugi three-component reaction | Isolated final product, 70.71% reported yield, ESI-MS and proton NMR |

The branched aldehyde is labeled B5 in JC_2023 but exactly matches AGILE B2
and LNPDB component `lnpdb-tail1-668dc43d423c6abc`. It is therefore an
independent execution of the same component structure, not a new aldehyde
identity. It does not resolve the separate AGILE B5 mapping discrepancy.

Two source-data problems remain explicit:

- The LNPDB tail2 string linked to the oleyl isocyanide encodes an `N=O`
  terminus rather than an isocyanide. The source-resolved
  (Z)-1-isocyanooctadec-9-ene structure is stored as a reconciliation record,
  while the raw LNPDB string is preserved.
- The supplement's reported masses and mmol values for the branched aldehyde
  route are internally inconsistent, and the final product's printed
  `6.2 mmol` amount is incompatible with its 87.57 micromol starting scale.
  Reported yields are retained, but none of these fields is silently corrected.

JC_2023 materially strengthens exact native-Ugi L2 supervision. It does not
close current procurement for the starting amines or make any route
computationally complete under the L2/L3 policy.

## Direct aldehyde transfer from LX_2024

LX_2024 reports 12 amine heads and 15 aldehyde tails in a reductive-amination
library. The aldehydes are already valid Ugi-role components, so no
handle-conversion hypothesis is needed. Ten use a 3,5-dihydroxybenzaldehyde
core bearing two fatty-acid esters; five use a 2,4,6-trihydroxybenzaldehyde
core bearing three fatty-acid esters.

The source provides a one-step EDC and DMAP esterification family, exact
structures, and product-specific proton NMR for all 15 aldehydes. Every source
aldehyde produces exactly one sanitized product under the frozen Ugi transform
with the reference amine and isocyanide. This establishes deterministic Ugi
compatibility, not experimental Ugi conversion.

Source reconciliation found:

- fourteen exact matches to parsed LNPDB tail records;
- one source-resolved A2-8 structure whose LNPDB string has an invalid trailing
  underscore;
- one extra parsed LNPDB triheptanoate that is not part of the reported
  15-member aldehyde set and is therefore excluded;
- a conflict in which the A3-6b structure and label specify
  tris(2-methylhexanoate), while the quantitative main-text example names
  heptanoic acid.

The source identifies historical vendors for all 12 heads, the two
hydroxybenzaldehyde cores, and the fatty acids. It does not supply current
catalog-level availability, and it reports no upstream head syntheses.

## Interpretation

This paper demonstrates why structural diversity and route-label diversity must
be measured separately. The library has many heads and tails, but the only
explicit upstream chemistry in its source package is the shared precursor
route. The result does not show that the heads or tails are impossible to make.
It shows that their syntheses must be recovered from current suppliers, cited
prior art, patents, other LNPDB papers, or a broader reaction corpus.

The model design remains unchanged:

1. A whole-molecule discrete flow proposes the complete lipid graph.
2. LNPDB-derived chemistry identifies lipid-native assemblies, component roles,
   and reusable route families.
3. The route layer searches for complete L2 paths to procurement-supported
   leaves.
4. Calibrated route evidence can guide or tilt generation without restricting
   the generator to a fixed component enumeration.

Four reviewed sources are still not enough to judge L2 sufficiency. The next
review pass is now gap-driven: it must add distinct isocyanide scope, negative
evidence, head closure, or a missing terminal route rather than merely another
large final-lipid library. Repeated examples are retained as independent
evidence but are not counted as new chemical identities.

## Reproduction

```bash
make m0-09-paper-route-reviews
```

The validated artifact is
`results/m0_09/lnpdb_paper_route_reviews.json`. It records input hashes, the
reviewed PDF hash, source pages, structure-resolved route steps, yields,
component assignments, evidence states, and the current aggregate counts.
