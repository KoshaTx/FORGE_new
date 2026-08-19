# M0-09 AGILE Virtual Ugi-3 Capability Census

## Purpose

The AGILE Zenodo candidate table contains 12,276 unique final-lipid structures
and computed descriptors. It does not contain component labels, upstream
routes, yields, failures, or biological outcomes. FORGE therefore uses it for
two bounded purposes:

1. recover programmatic Ugi product-to-component mappings through inverse
   decomposition and exact forward round-trip;
2. identify the unique amine, aldehyde, and isocyanide components that require
   L2 synthesis or L3 procurement resolution.

An inverse decomposition is not treated as an observed synthesis route. A
forward round-trip is not treated as synthesis success.

## Compact source derivative

The published source file is 116,969,761 bytes because it contains 813
descriptor columns in addition to the SMILES column. The deterministic
extraction:

```bash
make m0-09-agile-virtual-smiles
```

verifies the complete source against SHA-256
`0b3c84321062efc7b5b4ef6a6954cc3340f22de61389418ad53e89e1c92719fe`,
parses and canonicalizes every structure, and writes an 85,911-byte compressed
SMILES-only derivative. The derivative retains source-row identity and all
12,276 unique canonical structures. It retains no descriptor columns and adds
no labels.

## Exact L1 census

The capability audit:

```bash
make m0-09-agile-virtual-ugi3-capability
```

uses the measured-library-qualified Ugi role policy. Every product is
retro-decomposed, and every accepted component tuple must reconstruct the
exact original canonical product. The reacting amine atom is recorded
explicitly for every candidate.

| Result | Count |
|---|---:|
| Source products | 12,276 |
| Products with one exact qualified decomposition | 12,276 |
| Products without an exact decomposition | 0 |
| Products with multiple exact decompositions | 0 |
| Unique amine heads | 22 |
| Unique aldehyde components | 62 |
| Unique isocyanide components | 9 |
| Unique components total | 93 |

The recovered set is the complete 22 by 62 by 9 Cartesian product. This is a
structural property of the virtual library, not evidence that every product was
experimentally synthesized.

Forward enumeration also preserves site behavior:

| Raw outcomes | Unique products | Distinct target sites | Products |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 10,602 |
| 2 | 1 | 1 | 558 |
| 2 | 2 | 1 | 558 |
| 3 | 1 | 1 | 558 |

The target product selects one explicit reacting-site identity even when a
head can produce more than one regioisomer.

## Current L2 and L3 join

The 93 unique components were joined by exact canonical identity to the
existing precursor, source-route, and procurement audits. A capability match
means that some audited record exists for the exact structure. Exact
source-route evidence is reported separately and does not imply route closure.

| Role | Unique | Capability matched | Exact source route | Unmatched | L2/L3 closed |
|---|---:|---:|---:|---:|---:|
| Amine head | 22 | 18 | 0 | 4 | 17 |
| Aldehyde component | 62 | 18 | 17 | 44 | 0 |
| Isocyanide component | 9 | 7 | 7 | 2 | 0 |

The 17 closed heads are accepted procurement terminals under the current
time-stamped policy. The 17 exact aldehyde routes and 7 exact isocyanide routes
are extracted source procedures, but their forward verification, terminal
procurement, and execution closure remain unresolved. No aldehyde or
isocyanide is yet computationally route-complete to accepted leaves.
Consequently, zero of the 12,276 products is currently called route-complete.

This is not evidence that the remaining components are chemically
inaccessible. It is a precise work queue. The next route-census pass must
classify and recursively close the 62 aldehydes, 9 isocyanides, and 5
nonterminal or unmatched heads through exact procurement or upstream
preparations. Existing exact procedures should be verified and closed, while
the unmatched structures require bounded reaction-family routing.

## Artifacts

- `data/derived/agile_virtual12k_smiles.csv.gz`
- `data/derived/agile_virtual12k_smiles.manifest.json`
- `results/m0_09/agile_virtual_ugi3_capability.json`
- `results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz`
- `results/m0_09/agile_virtual_ugi3_component_ledger.csv.gz`

The product ledger preserves all 12,276 component mappings and reacting-site
records. The component ledger is the deduplicated recursive L2/L3 work queue.

The next deterministic pass assigns bounded structural programs to that queue.
It reproduces all 24 exact source programs and projects the source-supported
families to another 47 components without claiming exact evidence. This reduces
the nonterminal route work to 65 unique proposed intermediates and 33 unique
proposed leaves. See
`docs/M0_09_AGILE_VIRTUAL_UGI3_COMPONENT_PROGRAMS.md`.
