# M0-07 AGILE Assay and Structure Reconciliation

Status: **blocking source gate complete**

## Why this gate exists

AGILE reports 1,200 nominal Ugi-library measurements for both HeLa and RAW
264.7 cells. LANTERN subsequently reported label and molecular-identity
problems in an AGILE fine-tuning dataset. FORGE therefore reconciles the assay
table before fitting any biological oracle.

The result is narrower than a claim that all 1,200 FORGE values were wrong:

- all 1,200 HeLa values in the FORGE-vendored root AGILE table match the
  official article source workbook;
- all 1,200 RAW values also match the official source workbook;
- the real blocking issue is the molecular representation of B4 and B5;
- B4 is a measured cis/trans mixture and cannot be one graph;
- B5 is the pure-trans compound, but the root table encoded the opposite
  alkene geometry.

LANTERN provides curated HeLa labels and the B4/B5 identity audit. It does not
provide a curated RAW table, so RAW is independently reconciled to the
official AGILE article source data.

## Compatibility with the FORGE representation

The current FORGE product and oracle representations do not encode alkene
stereochemistry. That does not make the issue irrelevant. When stereochemistry
is removed, B4 and B5 have the same constitutional graph but different
measurements because one is a mixture and the other is a pure compound.
Training both as ordinary single-molecule examples would assign conflicting
labels to one model input.

The frozen policy is:

1. preserve all 1,200 nominal measurements in the audit ledger;
2. exclude the 100 B4 mixture observations from single-graph supervision;
3. preserve those 100 observations in a dedicated exclusion ledger;
4. retain the 100 B5 measurements and correct their isomeric provenance to
   the pure-trans graph;
5. expose constitutional canonical SMILES as the model input;
6. retain isomeric SMILES separately for chemistry and provenance.

This policy yields 1,100 unique constitutional graphs and 1,100 unique
single-compound isomeric graphs.

## Frozen result

| Check | Result |
|---|---:|
| Nominal AGILE library measurements | 1,200 |
| Official HeLa label disagreements | 0 |
| Official RAW label disagreements | 0 |
| B4 mixture measurements excluded from single-graph fitting | 100 |
| B5 pure-trans graph corrections | 100 |
| Curated single-structure oracle records | 1,100 |
| LANTERN HeLa cross-validation disagreements | 0 |
| Unique model-facing constitutional graphs | 1,100 |

No apparent pKa, particle size, polydispersity, encapsulation efficiency or
formulation robustness values are imputed.

## Hard downstream contract

Oracle training must consume:

`results/m0_07/agile_oracle_curated.csv.gz`

It must not fit directly from:

`data/vendor/AGILE_smiles_with_value_group.csv`

The raw file remains valid as a ledger of 1,200 nominal library records and
for carefully qualified chemistry audits. It is not a single-graph biological
training table.

The M0-05 source-evidence gate has been rebuilt under the same policy. It now
admits 1,100 exact single-compound L1 records and retains 100 B4 mixture
executions as non-single-graph abstentions. The exact upstream route to B5
remains an L2 abstention because resolving the B5 product identity does not
resolve the supplementary route's chain-length and naming conflict.

Before product-prior training, the AGILE-derived portion of R0 and dependent
M0-03 splits must also be regenerated. Non-AGILE M0-04 conclusions, the
nominal Ugi transform mechanics, and unrelated upstream route evidence are
not invalidated.

## Reproduction

```bash
make vendor-partial
make verify
make m0-07-agile-reconciliation
```

The gate verifies every input hash, parses both official assay matrices,
requires the declared B4/B5 counts, checks uniqueness, cross-validates all
1,100 retained HeLa records against LANTERN, and emits deterministic gzip
artifacts with fixed timestamps.

## Sources

- AGILE article and official source workbook:
  [Nature Communications 15, 6305 (2024)](https://doi.org/10.1038/s41467-024-50619-z)
- LANTERN analysis:
  [Communications AI & Computing (2026)](https://doi.org/10.1038/s44488-026-00007-x)
- LANTERN data pinned at commit
  [`11240f2`](https://github.com/AsalMehradfar/LANTERN/tree/11240f29ef92323649ae60d177b21df77e2d428b)
