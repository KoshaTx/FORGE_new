# Legacy cleanup verification

This pass consolidates five duplicate hashing/atomic-write implementations and removes 19 unused
private facade re-exports. It does not delete whole modules or change scientific algorithms.
`retirements.json` records exact replacements, retained error/keyword/permission contracts, and
pre-edit source hashes. Independent legacy sampler/reference implementations remain available.

## Checks

| Receipt | Tests | Failures | Errors | Skips |
|---|---:|---:|---:|---:|
| affected-tests.xml | 48 | 0 | 0 | 0 |
| baseline-tests.xml | 30 | 0 | 0 | 0 |
| core-tests.xml | 105 | 0 | 0 | 0 |
| final-maintenance-tests.xml | 25 | 0 | 0 | 0 |
| interface-tests.xml | 24 | 0 | 0 | 0 |
| isolated-tests.xml | 36 | 0 | 0 | 0 |
| survey-tests.xml | 15 | 0 | 0 | 0 |

The 35 study-compatibility checks also passed in the working tree and the isolated checkout.
The sampler comparison passed all 20 checks at seeds 0, 17, and 77, with identical results and RNG
state. It uses identical current shared dependencies for both implementations, not full historical
experiment reproduction. All 30 vendored assets verify. Changed Python passes Ruff/Black; the
maintenance CLI and survey pass scoped mypy. No repository-wide typecheck or full-suite claim is made.

The wheel was built from HEAD plus only this cleanup's selected files and installed in a new
temporary environment. Project imports came from that wheel. Existing binary dependencies were
reused through a search path; this does not qualify a fresh dependency download. The temporary
uv cache lacked hatchling initially; the build then succeeded using an existing local cached
hatchling. All 153 catalog entries and 102 registered stage IDs resolved without executing stages.

## Provenance

The expanded baseline inventories 12,974 distinct source identities: 9,265 resolved and 3,709 did
not resolve before this cleanup. It scans results, configuration, experiments, provenance docs,
and paper snapshots; it is much broader than the prior nine-identity unresolved subset. No new
exception ledger or weaker acceptance rule was introduced. All previously resolvable identities
still resolve, all 526 prior archive entries are intact, and ten exact pre-edit versions were added.
`source-identities-before.json.gz` records each identity and its declaring artifacts.

The combined survey is a conservative module inventory, not proof that every symbol is used.
Its [compressed report](../../../provenance/code-retirement/supported_studies_v1.json.gz) is a snapshot of its recorded source/document hashes and pin declarations.
Unbounded dynamic/reflection references and unresolved source pins require explicit review.
The historical diameter-helper export was retained after the final result-script search found
a surviving audit consumer. No whole module met the retirement gate in this pass. Frozen producers remain runnable at their
existing paths; no source blob is presented as proof of historical execution equivalence.

## Reproduce

```sh
make code-survey-supported
make test-one TEST='tests/test_maintenance_code_survey.py tests/test_legacy_cleanup_helpers.py'
make test-study-compatibility
PYTHONPATH=.:tools:paper .venv/bin/python results/maintenance/legacy_cleanup_v1/verify.py
make verify
```

The survey exits 2 when it finds unresolved references; do not treat that as permission to delete.
The retained JUnit files include the exact test names. Source surveys never launch experiments.
There were no training, remote, or sealed-holdout execution calls. Unrelated manuscript and baseline
work remains untouched. No commit or push was performed.
