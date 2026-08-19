# M0-07 Auxiliary Ugi and 3CR Oracle Supervision

**Status:** Source and compatibility audit complete. A controlled auxiliary-data
comparison is authorized. Raw cross-study target pooling is prohibited.

## Purpose

Additional ionizable-lipid datasets can broaden the chemistry presented to an
oracle encoder, but they do not automatically share an assay scale or target.
This audit separates native Ugi transfer from related-chemistry transfer before
any model is fitted.

The audit consumes hash-pinned LNPDB, reconciled AGILE, source-review, and
configuration inputs. It writes a deterministic row ledger and a summary with
the same input hashes.

## Label contract

The LNPDB endpoint values in scope are within-study, within-endpoint z-scores.
They are not interchangeable quantitative measurements across studies. FORGE
therefore prohibits concatenating them into one regression target.

Permitted comparisons are:

1. AGILE-only training.
2. Source pretraining followed by AGILE fine-tuning.
3. A shared encoder with study-specific prediction heads.
4. A declared external-transfer diagnostic.

Auxiliary data are retained only if they improve frozen AGILE component
holdouts or a declared external transfer without exposure to held labels.

## JC_2023

JC_2023 contributes 288 unique native AGILE-type Ugi products with HeLa labels.
Its component coverage relative to reconciled AGILE is:

| Role | Unique source components | Exact AGILE overlap | New relative to AGILE |
|---|---:|---:|---:|
| amine head | 16 | 16 | 0 |
| aldehyde | 6 | 1 | 5 |
| isocyanide | 3 | 2 | 1 |

Thirty-two complete products overlap AGILE exactly under the declared
stereochemistry-free representation. Primary-source Figure 1 and the
supplement resolve C1 as oleyl, C2 as saturated C18, and C3 as saturated C11.
Only the malformed LNPDB C1 string changes constitutional identity after this
source reconciliation.

The LNPDB product graph is also inconsistent for all 18 A3 combinations. It
omits the tertiary amine shown in source Figure 1. The audit preserves each raw
graph and rebuilds the model identity from the exact source components only
when the hash-qualified Ugi transform produces one unique product. The final
gate retains 270 source-product identities and 18 explicitly corrected A3
identities, with 288 of 288 model identities passing exact forward
reconstruction.

JC_2023 is the primary native-Ugi auxiliary comparison because it expands tail
chemistry and supplies an external study shift. It does not expand exact
amine-head diversity.

## LM_2019

LM_2019 contributes 1,080 unique related isocyanide-mediated 3CR products.
The audited rows comprise 1,080 HeLa, 36 BMDC, and 12 BMDM measurements.
Repeated structures with different endpoints remain separate observations.
No product overlaps reconciled AGILE exactly.

LM_2019 uses amines, ketones, and isocyanide-linker components. It is therefore
eligible for related-chemistry encoder pretraining and external-transfer
analysis, not direct pooling into the AGILE Ugi target. Its component
interpretation remains database-derived until a primary-source chemistry audit
is pinned.

## Leakage controls

All auxiliary filtering occurs separately inside each frozen AGILE fold before
source pretraining or shared-encoder fitting. Every split excludes exact AGILE
test products from the auxiliary training rows. Scaffold evaluation also
excludes the held scaffolds. Component and component-pair evaluation excludes
auxiliary records containing the held component structures or held component
pair.

Prediction heads are keyed by study and endpoint. LM_2019 HeLa, BMDC, and BMDM
labels are never pooled into one head.

Because LNPDB z-scores were computed from each complete study-endpoint
population, internal auxiliary test metrics are diagnostic and transductive.
They cannot select the final oracle. The selection-relevant result remains
performance on properly isolated AGILE folds.

## Reproduction

```bash
make verify
make m0-07-auxiliary-supervision
PYTHONPATH=src python3 -m pytest tests/test_m0_07_oracle_auxiliary.py -q
```

The generated artifacts are:

- `results/m0_07/oracle_auxiliary_supervision.json`
- `results/m0_07/oracle_auxiliary_records.csv.gz`

This audit qualifies data use. It does not establish that auxiliary
supervision improves the oracle. That decision requires the frozen
component-held-out comparison.
