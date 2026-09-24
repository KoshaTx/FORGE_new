# Full-corpus preparation and supplied-recipe reconstruction

This is the current consolidated receipt directory. It supersedes the v1 and v2 validation
attempts. No training was launched or admitted.

The full preparation ledger accounts for all 3,182,837 supplied records. It admits 76,800
records to program preparation under both existing TRAIN assignments and the additional
protections. The independent replay audit finds 26,619 exact supplied-recipe reconstructions
across nine families; 50,181 eligible records still lack that evidence.

The 135 additional exclusions are iPhos records whose exact same-family component/role/quantity
combination occurs in a protected split. Existing splits and upstream data are unchanged.

All 478 focused tests and vendor verification pass. The full suite has 3,088 passed, 131 failed,
17 setup errors and 92 skipped/expected failures, with no new failures or errors versus the
preceding completed baseline. The source snapshot was unchanged during final validation. The
full repository gate remains failed; the Phase 1 definition of done has not been met.

## Receipts

- `result.json`: consolidated closeout, including the outstanding gates and test result.
- `reconstruction_report.json`: family counts, exclusions, remaining program work, missing
  original-file dependencies and input hashes.
- `readiness.jsonl.gz`: one status for every source record, including protected and unresolved
  records. This large local artifact is excluded from Git and hash-pinned in the report.
- `constitutional_deduplication.json`: identity counts within the eligible population and exact
  subset. This does not qualify deduplication or sampling weights for the full universe.
- `validation_report.json`: focused tests, full-suite result, vendor verification and comparison
  against the preceding completed repository run. Existing failures remain failures.
- `reproduction.json`: an independent rerun reproduced the report, duplicate audit and full
  readiness ledger byte-for-byte.

## Reproduction

From the repository root, with the pinned source artifacts present:

```bash
.venv/bin/python -m experiments.phase1.multireaction.compose_lipid_full_preparation \
  --verify results/phase1/compose_lipid_full_preparation_v1/result.json
.venv/bin/python results/phase1/compose_lipid_full_preparation_validation_v3/reproduce.py
.venv/bin/python results/phase1/compose_lipid_full_preparation_validation_v3/validate.py
.venv/bin/python results/phase1/compose_lipid_full_preparation_validation_v3/closeout.py
```

The validator runs the focused suite, `make verify`, and `make test` using the local virtual
environment. It records the production source snapshot before and after testing.

Fresh full replay commands and the preparation policy are documented in
`docs/COMPOSE_LIPID_V8_PRETRAINING.md`. Current role-binding evidence is
`results/phase1/compose_lipid_role_binding_v2/result.json`.

## Preserved corrections

The first Ugi-3 binding verifier reran an older molecular population while authenticating prior
evidence. It is superseded by byte-only prior-evidence authentication followed by replay of the
current eligible population. The prior implementation, affected receipt hash and correction
are preserved in `compose_lipid_full_preparation_validation_v2/scope_correction.json`.

An initial attempt to repeat the report resolved the virtual-environment Python symlink to its
base interpreter and failed to import RDKit. It did not alter the scientific outputs. The
operational failure is preserved as `reproduction_initial_interpreter_error.json`; the corrected
reproduction uses the virtual-environment interpreter explicitly and passes.

Full-universe partition and study protection, the remaining source programs, full-size
representation, final deduplication and balanced weights, and the repository-wide test gate
remain open. The missing 63 original package files constrain their dependent provenance checks;
they were not a blanket requirement for this preparation work.
