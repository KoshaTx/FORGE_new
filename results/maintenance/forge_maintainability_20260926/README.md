# FORGE maintainability verification

This receipt covers study navigation, acquisition I/O consolidation, product/L1 stage extraction,
shared sampler extraction, terminal-state types, and the new study compatibility CI check.
It does not qualify training, admit a dataset, or reproduce the whole historical repository.

## Results

- The catalog retains its 153 ordered entries and 102 registered stage IDs. All 125 explicit
  repository imports inspected from the moved interfaces resolve.
- All 34 sampler and 65 product/L1 adapter function bodies match the preserved versions after
  removing type annotations and identity casts. The sampler facade is now 1,190 lines; the
  product/L1 stage facade is 339 lines.
- The direct sampler comparison passes 20 checks at seeds 0, 17, and 77, including terminal
  tensors, abstention reasons, fixed states, molecular identities, model-call counts, RNG states,
  and trace records. It runs the baseline sampler and current sampler with the same current
  shared dependencies; it is not an old experiment rerun.
- Saved JUnit reports show 35 study compatibility tests, 58 acquisition/I/O tests, and nine
  historical verification/archive tests passing, with no skips. The pre-edit baseline had
  117 passing focused tests. Additional adapter, COMPOSE fixture, and decoder checks were run
  during implementation; the saved reports are scoped receipts, not a full-suite claim.
- All 30 present vendored assets pass hash verification. All 513 original archive entries are
  preserved; 13 exact source versions were added. Across 1,304 inspected source references,
  1,202 now resolve and no previously resolvable reference was broken. The other references
  point to nine already unresolved historical source identities listed in
  `source-preservation.json`.
- Targeted mypy checks pass for 12 files covering new contracts, checkpoint/state helpers,
  acquisition I/O, core I/O, and the stage adapters. The existing sampler's broader mypy check
  reports the same 103 errors before and after; both logs are retained. No ignores or gates were
  added to hide those errors. Further historical-source recovery and model typing are separate work.

`verification.json` records current input hashes and receipt hashes. `historical-recovery.json`
records the three older source versions recovered from exact local snapshots when historical
verification exposed missing archive entries. Training and remote jobs were not launched.

## Reproduce the focused checks

From the repository root, using the existing development environment:

```sh
make test-study-compatibility
make test-one TEST='tests/test_core_io.py tests/test_source_acquisition_io.py tests/test_m0_09_pmc_sources.py tests/test_m0_09_publisher_sources.py'
make test-one TEST='tests/test_compose_lipid_pretraining.py::test_historical_authentication_cannot_silently_execute_changed_source tests/test_combinatorial_generation.py::test_saved_generation_is_verified_from_attempts_and_forward_replay tests/test_combinatorial_generation.py::test_verifier_rejects_forged_result tests/test_core_provenance_archive.py'
make verify
PYTHONPATH=. .venv/bin/python results/maintenance/forge_maintainability_20260926/sampler_equivalence.py \
  --baseline provenance/frozen-code/sha256/34/34703affff33a07d69967172531656b1f611bb7b4220916c6bfc55b2c7c28342 \
  --output /tmp/forge-sampler-equivalence.json
```

The local runs used `UV_CACHE_DIR=/private/tmp/forge_uv_cache` and
`UV_RUN='uv run --offline --no-sync'` for make targets that invoke uv. These settings reuse the
installed environment without downloading dependencies. The new GitHub Actions job has been
added but has not been executed remotely.
