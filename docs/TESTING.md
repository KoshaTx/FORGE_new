# Test workflow

## Pull-request checks and historical evidence

CI runs paper review, study compatibility, evidence regression, and core code checks separately.
Within the core job, typing, linting and tests each run after a successful dependency installation,
even when another check fails. Local equivalents are `make typecheck`, `make lint-core`, and
`make test-core`; their existing selections and assertions are retained.

The evidence regression job compares the candidate checkout with the PR base commit (or the
previous commit on a push). It scans `results/`, `docs/provenance/`, `configs/`, and `experiments/`
using the same verifier for both trees. It fails on any removed declaration, lost previously
available path/hash identity, removed archive identity, or newly unresolved declaration. Recovering
one identity cannot compensate for losing another. All archive blobs are checked, and malformed
JSON fails instead of silently disappearing from the inventory. The known-drift exception ledger
does not exempt an identity from this comparison.

The `historical-evidence-audit` CI artifact lists the remaining unavailable and external inputs.
Its summary explicitly says **INCOMPLETE** when local declarations remain unresolved. These include
maintenance inventories and other studies, not only the submitted paper. A green regression check
means this change introduced no new evidence loss; it does **not** certify historical reproduction.

To compare against a separate, clean checkout of the base revision:

```sh
PYTHONPATH=.:tools python3 -m forge_provenance.regression \
  --base-root /path/to/base-checkout --output /tmp/forge-provenance-regression.json
```

`make verify-pins` and `make check-core` retain their strict historical drift checks. No old hashes
or exception allowances are changed. Paper-specific availability is checked separately by
`make review-check` and `python3 tools/review_submission.py --require-run-artifacts`; see
[`paper/submission/ARTIFACTS.md`](../paper/submission/ARTIFACTS.md) for the missing paper inputs.

The core tests use the public LNPDB CSV at the exact upstream commit and SHA-256 already recorded
in `forge_data.vendor`. CI fetches and verifies it before testing. To fetch that input locally:

```sh
uv run --frozen python -m forge_data.fetch lnpdb_fc7c389.csv
```

The fetcher does not rewrite the vendor manifest, accept changed bytes, or replace a corrupt local
copy. The full vendor command also requires historical workstation inputs; this selected download
does not supply those inputs or the paper checkpoints.

Tests that require unavailable historical receipts, ledgers or publisher supplements declare their
exact paths with `requires_artifacts`. They report **SKIPPED**, with every missing path in the pytest
summary, rather than counting as a reproduction pass. Present files still undergo the original
assertions; corruption is not a reason to skip. Small synthetic unit tests separately exercise
preflight rejection and missing-input behavior and never become scientific evidence. CI uploads the
`core-test-results` JUnit report, including skips. For strict reproduction, fail on missing inputs:

```sh
uv run --frozen pytest --require-external-artifacts -ra tests/test_shared_synthesis_program_production_preflight.py
```

This option covers the explicitly declared external-artifact tests. The older unrecoverable-pin
quarantine remains separately reported, and the strict paper-artifact verifier remains necessary
for a paper reproduction claim.

## Choosing a test scope

Use the smallest relevant check during development. The repository-wide suite includes complete
corpus rebuilds, byte-reproducibility runs, model exercises, and verification of historical
experiment artifacts. Historical reproduction is separate from new training readiness; missing
old experiment outputs do not block the current training pipeline.

## While implementing

`make test-study-compatibility` checks both study interfaces and shared decoding on small
local fixtures. It covers catalog registration, extracted source inventories and fingerprints,
passive RNG traces, fixed closures, and COMPOSE origin/ring/repeat behavior. It performs no
optimizer training or remote submission. This is a compatibility check, not data admission or
historical result reproduction. See [STUDIES.md](STUDIES.md) for the workflow map.

The default is the affected case or owning file, then stop once it passes:

```sh
make test-one TEST=tests/test_compose_lipid_training_measure.py
# Or one behavior:
make test-one TEST=tests/test_compose_lipid_training_measure.py::test_misjoined_exact_evidence_is_rejected
```

`TEST` is mandatory: omitting it fails immediately instead of accidentally running every test.
Use `TEST_PYTEST_ARGS` for additional pytest arguments. There is no automatic whole-repository
Git-diff selector; unrelated existing workspace edits must not inflate this check.

Documentation/status-only changes need no pytest run. For changed configuration or input pins,
validate the affected schema, hashes and contracts. Do not rebuild unchanged datasets or rerun
unrelated historical experiments. Broaden coverage only for a failure, a shared-interface change,
an unresolved concern, or an integration checkpoint.

For a change spanning the all-family preparation pipeline, use this integration check:

```sh
make test-preparation
```

This explicit suite covers evidence indexing, source identities/quantities, atom origins,
constitutional program graphs, qualified and mapped loaders, family replay, and complete-family
weighting. It runs the existing tests with their existing assertions and fixtures. It does not
cover every model, training loop, routing experiment, potency experiment, or historical artifact.
Add the relevant tests when work touches those areas; this suite is not an automatic dependency
selector or a substitute for dataset admission.
It is not the default after every edit; the affected test file is.

For timings of this suite, without launching the full suite:

```sh
make test-preparation PREPARATION_PYTEST_ARGS='--durations=20'
```

## At a current training-readiness checkpoint

```sh
make test-training OUTPUT=results/phase1/<new-validation-directory>
```

This command verifies vendored inputs and runs the seven files declared in `TRAINING_TEST_FILES`
in `forge/corpus/compose_lipid_training_data.py`: qualified cache controls, family-balanced weights,
the training adapter/objective, noise fitting, run/launch checks, restart primitives and Modal
transport mocks. It records the current source/input hashes, exact test selection, JUnit report,
logs and execution status in `forge.compose_lipid_training_validation.v1`. Every selected file must
execute, with no failures, errors or skips. The result is a scoped engineering check; complete
chemistry, frozen holdouts, full molecular-size support and final dataset admission remain separate.
Synthetic optimizer and mocked-transport checks do not claim real training or GPU qualification.

For environments using the existing venv directly:

```sh
PYTHONPATH=.:tools .venv/bin/python -m experiments.phase1.multireaction.compose_lipid_validation --output results/phase1/<new-validation-directory>
```

The user removed the historical full-suite prerequisite on 2026-09-22. `make test` and
`make test-full` still collect every test when full repository reproduction is wanted. Historical
tests and their failures remain intact; they do not control new training admission. A current
training pass is never presented as a repository-wide pass or as proof of old experimental claims.

Do not repeat a finished check simply because another status update is requested. Reuse a completed
receipt only for its exact tested source and input hashes. Changes that affect that check invalidate
the receipt. If a full run is already active, retain its process identifier and output paths and
monitor that run rather than launching a duplicate.

## Further speed work

Use saved JUnit case timings to choose optimizations. Full-data determinism checks intentionally
rebuild twice; replacing the second execution with a cached result would remove what they test.
Small behavior tests may share immutable setup where independence is preserved, but mutations must
remain isolated. Verify identical outcomes and record a before/after measurement before claiming
that an optimization makes the same suite faster.

Do not enable unrestricted process parallelism without checking shared files, global random/thread
state, memory use, and output ownership. Faster test selection and faster execution are separate
claims. Neither changes the chemistry, leakage, full-size representation, provenance, or admission
requirements for training.

## Legacy retirement checks

`make code-survey-supported` inventories both studies without running workflows. Its exit code 2
means dynamic references need review, not that code is disposable. See
[LEGACY_CODE_RETIREMENT.md](LEGACY_CODE_RETIREMENT.md) for classifications and provenance rules.
Run `make test-one TEST='tests/test_maintenance_code_survey.py tests/test_legacy_cleanup_helpers.py'`
for survey/helper changes; add the owning subsystem tests for actual removals. Shared-interface
changes also require `make test-study-compatibility`. Missing historical inputs remain separately
reported and are never silently counted as passing reproduction.
