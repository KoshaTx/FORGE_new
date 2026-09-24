# Original-generator task bundle intake

The user supplied this [Drive bundle](https://drive.google.com/drive/folders/1yWMDZ_4J1WNuVEtyw5Yi5gyatL3PbX0C).
Its README was read before acquisition. This intake restores the 63 missing original task/context
files without changing training admission or any frozen split.

- New bundle: `shasum -a 256 -c SHA256SUMS` passes all 68 entries.
- All 63 task/context files match the earlier supplement's expected hashes exactly.
- After restoring only those files, the earlier package passes all 678 checksum entries.
- All 63 restored files are bound by one of the four supplied family receipts.
- Family 3's precursor catalogue already exists in the earlier delivery and resolves every
  referenced precursor and source-parent ID. No additional catalogue is needed for these references.

| Source records | Rows | Structure availability |
| --- | ---: | --- |
| Family 3 Ugi-4 regional dispositions | 221,184 | All IDs resolve through the existing catalogue |
| Family 5 Ugi-3 author contexts | 12,276 | Complete component SMILES included |
| Family 5 Ugi-3 context constructions | 6,336 | Complete component SMILES included |
| Family 18 STAAR tasks | 334,768 | Complete component SMILES in all 30 shards |
| Family 21 acid/epoxide tasks | 231,984 | Complete component SMILES in all 30 shards |

All 60 shard row counts match the source receipts. All rows retain their source
`training_admissible: false`. Task/context row counts are not counts of admitted model examples.
The audit parses JSON metadata and performs exact ID lookups; it parses no molecular graphs,
executes no reactions and imports no upstream atom annotations into FORGE supervision.

All **478 focused tests** and vendor verification pass. The completed full suite reports
**3,088 passed, 131 failed, 17 setup errors and 92 skipped/expected failures**, with exactly the
same failure/error identities as the preceding completed baseline. The production source snapshot
was unchanged throughout validation. The full repository test gate remains failed; this intake
closes the source-transfer gap without claiming Phase 1 or training readiness is complete.

## Artifacts

- `acquisition.json`: source URL and downloader/transfer pins.
- `package_check.json`: both checksum commands, exact restoration list and receipt bindings.
- `record_audit.json`: per-file counts, schema checks and Ugi-4 catalogue resolution.
- `validation_report.json`: repository validation and comparison with the preceding baseline.
- `result.json`: final consolidated intake receipt.

The original bytes live in `data/source_cache/compose_lipid_original_tasks_2026-09-20/`.
The restored copies live at the paths expected under
`data/source_cache/compose_lipid_supplement_2026-09-19/original_generator_tasks/`.
Source data are locally cached and excluded from Git. Earlier missing-file and incomplete-package
receipts remain unchanged as historical evidence.

## Reproduce

From the repository root, using the existing virtual environment:

```bash
.venv/bin/python results/phase1/compose_lipid_original_tasks_intake_v1/acquire.py
.venv/bin/python results/phase1/compose_lipid_original_tasks_intake_v1/verify_restore.py
.venv/bin/python results/phase1/compose_lipid_original_tasks_intake_v1/audit_records.py
.venv/bin/python results/phase1/compose_lipid_original_tasks_intake_v1/validate.py
.venv/bin/python results/phase1/compose_lipid_original_tasks_intake_v1/closeout.py
```

Acquisition requires network access. Restoration is idempotent and refuses to replace different
existing bytes. Scientific gates remain in `docs/COMPOSE_LIPID_V8_PRETRAINING.md`: the transferred
source files enable their dependent checks but do not themselves qualify a training dataset.
