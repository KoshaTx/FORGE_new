# Full-universe validation

The preparation audit now covers all 3,182,837 original source records across 23 program
families plus the reference categories. This does not admit a training dataset.

- 377 focused tests pass, including 21 new full-universe accounting/protection cases.
- All 30 vendor assets pass `make verify`.
- Complete-universe row-by-row replay passes.
- Touched implementation formatting and lint pass.
- `make test`: 2,987 pass, 131 fail, 17 setup errors, 92 skipped/xfail, 3,227 total cases.
- Failure/error IDs are unchanged from the preceding validation. No test was weakened or
  skipped to qualify this change. The implementation/test snapshot stayed unchanged.

`checks.json` preserves exact commands, return codes and durations. `source-snapshot.json`
pins the code tested. `universe-replay.json` pins the input readiness receipt and replay
command. The full test log and JUnit results are also retained in deterministic gzip form.

Run `PYTHONPATH=. .venv/bin/python results/phase1/compose_lipid_v8_universe_validation_v1/report.py`
to verify pins and recount the completed test results. `run_checks.py` reruns the actual tests;
recounting a report is not a new test execution.

The final training population remains blocked on authenticated complete component/construction
provenance, corrected/global holdout qualification, the remaining family programs, representation
support for the source-declared larger graphs, and repository test-input recovery. No training,
generation, upstream corpus modification, or paid remote computation was performed.
