# Commit validation

The proposed source is HEAD `f1b7565ddec912ebe9b72e7fe8f0b9b1f3f01d9f` plus the
explicit `selection.json` paths. `result.json` records every selected source/artifact hash,
the separately staged campaign-only decision log, and imported repository code hashes.
Unrelated working-tree changes are excluded.

- Isolated source: 168 passed, no failures, errors or skips. All imported repository code
  resolves inside the temporary checkout and matches HEAD or the selected source.
- Original evidence workspace: 196 passed, no failures, errors or skips.
- The two saved-route files (28 cases) require original-path/source admissions and
  filesystem attestations; they are not claimed portable.
- The first isolated run preserved 12 failures and 19 setup errors when provenance
  checks refused symlinked or absent inputs. The test/code gates were unchanged.
- `explicit_test_inputs.json` pins the 42 non-code inputs materialized for the isolated
  source run. Other historical non-code artifacts were available by links; none of the
  results-backed Python source was supplied by those links. Git is not the full data bundle.

Original workspace command: `workspace_test_command.json` records the complete `make test-one`
argument list. For the isolated run, archive the base source and tracked result helpers,
overlay the explicit selection, supply the pinned non-code bundle as real files where
required, then use the same test list without `test_candidate_routes.py` and
`test_new_clock_routes.py`. Set `PYTHONPATH` to that checkout and its
`experiments/phase1/route_improvement` directory; verify the imported code hashes.
The original strict source/path attestations must not be rewritten to make a relocated
historical route run pass. No scientific producer or paid job was launched.

The source archive and earlier consolidation receipts retain their original hashes and
pre-commit timestamps. The selected decision log includes only this campaign's appended
entries; earlier unrelated uncommitted log text remains in the working tree.
