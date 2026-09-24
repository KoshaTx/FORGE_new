# Supplemental COMPOSE intake — 2026-09-19

The principal data are verified. The complete package remains incomplete because 63 files listed
in the supplied checksum manifest are absent from the shared Drive folder. No training is admitted.

| Check | Result |
| --- | ---: |
| Full target-ID joins | 3,182,837 |
| Unique canonical complete precursor structures | 45,805 |
| Collapsed component-instance references resolved | 8,946,808 |
| Inputs after quantity expansion | 10,616,868 |
| Missing/extra targets or component-multiset mismatches | 0 |
| Corrected split rows | 200,000 |
| Designated held component identities / held studies | 905 / 6 |
| Their overlap with corrected provider TRAIN | 0 / 0 |
| Prior exact reconstructions agreeing with source precursor sets | 47,363 / 47,363 |
| Protected component identities, historical plus corrected split | 1,121 |
| Full-universe targets containing a protected component | 1,368,812 |
| Upper bound remaining before other gates | 1,814,025 |
| Manifest checksum matches / missing / mismatches | 615 / 63 / 0 |

`summary_report.json` consolidates findings and pins the supporting evidence. `result.json` records
the full join and pins `joins.sqlite`; `component_exclusions.json` pins the separate exclusion
ledger. SQLite artifacts are local, reproducible and excluded from Git. `package_check.json`
retains the actual checksum command's nonzero exit code. The original source files remain cached
under `data/source_cache/compose_lipid_supplement_2026-09-19/`, with acquisition receipts.

The missing files are all 30 STAAR task shards, all 30 acid–epoxide diester task shards, and three
Ugi context/tuple files. `missing_source_files.sha256` gives their exact paths and expected hashes.
README/PACKAGE receipts, source definitions and original tasks are data; no downloaded producer
code was executed or modified.

Replay the principal audit into a fresh directory:

```bash
.venv/bin/python -m experiments.phase1.multireaction.compose_lipid_supplement \
  --config configs/multireaction/compose_lipid_supplement_intake_v1.json \
  --output results/phase1/compose_lipid_supplement_intake_replay
```

`component_protection.py` builds the separate global exclusion diagnostic from authenticated
historical inputs and the verified construction index. Its counts do not define a final training
split. Existing product, study, combination/morphology and quarantine protections remain binding.
In particular, the corrected provider TRAIN assignment cannot release 35,979 previously protected
FORGE records. Chemistry and size support still require qualification before fitting weights.

The 3,833 product-derived compatible decompositions explicitly disclaim a historical route.
Precursor structure-set agreement and deterministic replay do not establish experimental success.
The audit parses precursor structures only for identity checks and does not fit support or a model.

Validation: 396 focused tests pass; vendor verification passes. The full suite has 3,006 passes,
131 failures, 17 setup errors and 92 skipped/expected failures. Failure/error IDs match the preceding
completed suite exactly. No new failures were introduced. Full logs, JUnit and source snapshots
are retained; compressed copies of the large logs are trackable.
