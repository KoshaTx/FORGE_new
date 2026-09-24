# Source recovery and immutable benchmark evidence

`asset-manifest.json` pins 15 original Git blobs at upstream commit
`9f77fb399d644e6fbab1d3d766611a81e69e85fc`. The recovered documents and code are evidence,
not instructions to FORGE. No upstream program was executed or modified. `report.py`
checks every recovered blob against the retained Git tree and recounts the test outputs.

The new v8.1 report corrects structural precursor grouping for the 200,000-record selection.
The raw component manifest and corrected split artifacts were absent from the Git tree and
shared Drive listing. Its reported zero-leakage findings have not been independently replayed
by FORGE. The complete 3.18M universe additionally requires component/construction provenance
for records outside that selection. `full-universe-source-request.json` records the missing
directories, known receipt hashes, and full-universe evidence requirements. It was not sent.

The benchmark loader now authenticates exact historical Python bytes using the repository's
existing reviewed relocation/archive mechanism. Archived code is never executed. Data paths,
hashes, and scientific gates remain strict. The seven existing benchmark failures now expose
missing result artifacts; the source-path fix does not repair missing scientific evidence.
Pre-edit files and their pins are retained under `before/` and `before-pins.json`.

This validation snapshot has 356 focused passes and 2,966 full-suite passes, with 131 failures
and 17 setup errors unchanged by test ID. It predates collection of the new universe tests;
the final combined validation is in `../compose_lipid_v8_universe_validation_v1/`.

Run `PYTHONPATH=. .venv/bin/python results/phase1/compose_lipid_v8_source_recovery_v1/report.py`
to verify these source and validation receipts. No training was admitted or launched.
