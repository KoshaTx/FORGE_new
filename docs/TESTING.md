# Test workflow

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
