# Remaining working-tree commit audit

This series preserves the outstanding main-checkout changes after `aa2a5f9d`.
`initial.json` records the original paths, hashes, sizes and five disjoint groups.
`result.json` pins the final selection and validation receipts. No experiment was launched.

- Training: 278 passed, zero skipped/failed/error; 30 vendored assets verify.
- Optional molecular-repair helpers: 39 passed, zero skipped/failed/error.
- Ruff and Black: all 18 selected Python files pass.
- Canonical manuscript builds: 39 and 76 pages; saved PDF text exactly reproduced.
- Root-level manuscript ZIP files pass their CRC checks.

Tests used an isolated checkout of the series base with first the training changes,
then the repair changes. Authenticated non-code inputs were copied without modifying
originals. The first missing-fixture/shared-memory failure is retained separately.
`training_fixture_inputs.json` and `quality_fixture_inputs.json` identify supplied inputs.
The full source and command receipts are in `training_validation_v2/`. These receipts
describe a historical temporary checkout; they are not a fresh dataset admission.

Paper snapshots retain existing prose and scientific scope. `paper_saved_receipt_check.json`
records six stale artifact pointers in three superseded 22-family wrappers. Current build
checks are separate from those old records and from scientific evaluation. The archives
and drafts are preserved rather than relabelled as final results.

Nested `.worktrees/`, `:memory:.ses`, interpreter/build caches, and ignored experimental
data/checkpoints stay local. They are not embedded Git repositories in this series.
