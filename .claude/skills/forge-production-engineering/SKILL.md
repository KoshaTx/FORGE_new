---
name: forge-production-engineering
description: Implement, refactor, review, or test production-quality Python in the FORGE repository. Use for code changes, CLIs, data pipelines, tests, reliability work, or release-readiness checks. Do not use for paper-only prose or exploratory analysis that will not change the repository.
---

# FORGE Production Engineering

Apply this workflow after reading the repository-root `AGENTS.md`. That file controls scientific scope;
this skill controls engineering execution. If they conflict, follow `AGENTS.md`.

## Establish the contract

1. Read the applicable M0 task and its acceptance criteria before editing.
2. Identify inputs, outputs, invariants, failure behavior, and the runnable verification command.
3. Stop if the change expands beyond the authorized milestone or weakens a scientific gate.
4. Prefer the smallest complete change that satisfies the contract. Avoid speculative abstractions,
   framework adoption, and unrelated cleanup.

## Implement for reliable use

- Keep domain logic in typed, testable functions; keep CLI parsing and filesystem I/O thin.
- Validate inputs at boundaries and fail with actionable messages that name the bad field or file.
- Make numeric pipelines deterministic. Record seeds, timestamps, parameters, and input SHA-256 hashes.
- Read chemistry definitions from the vendored registry. Never duplicate SMARTS, policies, or settled
  constants in code.
- Use stable schemas for saved artifacts. Sort serialized mappings and collections when order is not
  meaningful.
- Avoid partial outputs: calculate first, then write complete artifacts. Do not leave a plausible-looking
  result after an exception.
- Preserve backward compatibility unless the task explicitly changes a contract. Document intentional
  schema changes.
- Add a dependency only when the standard library and existing dependencies are insufficient; keep it
  scoped to the appropriate optional dependency group.
- For compute-heavy paths, profile representative inputs, batch or vectorize the hot path, keep memory
  bounded, and benchmark the change on the intended CPU or GPU without reducing scientific support.
- Do not hide errors, silently coerce invalid chemistry, or relax thresholds to obtain a passing result.

## Test the behavior

Add tests proportional to the risk:

- A focused happy-path test for the public behavior.
- A regression test for the defect or scientific invariant being protected.
- Boundary and malformed-input tests where failure would otherwise be silent.
- Determinism and provenance assertions for scripts that produce numbers.
- Schema assertions for persisted JSON/CSV artifacts.

Prefer small fixtures and behavior-level assertions. Do not copy production logic into tests. Tests must
not require network access. Mark tests that require vendored data, and skip only when the documented
asset is absent—not when a result is inconvenient.

## Verify in layers

1. Run the narrowest relevant test while iterating.
2. Run formatting or lint checks for touched Python.
3. Run `make verify && make test` before declaring an M0 task complete.
4. Inspect generated artifacts and the final diff; confirm no unrelated user changes were overwritten.

Report what changed, the exact verification performed, and any residual risk or human decision. Never
describe an unrun check as passing.
