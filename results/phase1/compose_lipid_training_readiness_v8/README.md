# Full-population training preparation — 2026-09-21

**900,435 eligible records now have exact computed reconstruction evidence. Training remains
unqualified.** This checkpoint adds 742,603 exact reconstructions without changing source recipes,
protected membership, chemistry gates or molecular-size support.

| Check | Result |
| --- | ---: |
| Source records accounted for | 3,182,837 |
| Eligible for program preparation | 1,272,466 |
| Protected by frozen exclusions | 1,910,371 |
| Unresolved nonprotected partition records | 0 |
| Exact computed reconstructions | 900,435 |
| Eligible records still requiring chemistry qualification | 372,031 |
| Families with exact evidence | 17 |
| Families with complete eligible exact replay | 10 |
| Formal families with eligible records | 22 / 23 |
| Eligible product graph representations checked and passing | 1,272,466 |

The thirteen-family batch evaluated 749,119 pending recipes in 1,570.74 seconds with four CPU
workers and added 701,150 exact reconstructions. A separate fixed anionic iPhos program evaluated
98,862 pending recipes in 807.60 seconds with four workers and added 41,453 exact reconstructions.
Both jobs preserve atomic restartable shards and passed serial/parallel numerical-equivalence
preflights. Every completed record was independently reconciled against the eligible partition,
target identity and original complete component IDs, roles and quantities. No program or charge
variant was selected by matching a desired product. Exact replay establishes computed consistency,
not an experimentally executed route or L2/L3 closure.

The product representation audit checked all eligible records in 476.68 seconds. Sparse edges,
constitutional round-trips, source atom counts, bond support and valence capacity all pass. Eligible
products reach 250 heavy atoms and 12 closures; 68,358 exceed 96 atoms. The source universe still
reaches 254 atoms. A finite model smoke test covers the full 254-atom envelope with zero training
steps. This is product-graph qualification: joint synthesis-program coordinates and vocabulary
fitting on the final admitted population remain required.

The 372,031 pending recipes divide into:

- 266,502 in five families without qualified replay: acid/epoxide diester (101,098), amine
  alkylation (92,699), ketone/isocyanide amide (33,693), amine/epoxide (33,507), and AEMA (5,505).
- 105,529 failures of existing programs: iPhos (57,409), thiol-yne terminal constraints (27,286),
  aryl reductive amination (11,428), O-esterification (4,473), A3 (2,520), maleate (2,262), and
  acrylate source-role tuples (151). Failures include uniqueness, source scope, occupancy, inverse
  and element/hydrogen/charge checks. None exhausted the configured search bounds.

The inspected Miao supplement and figures supply family precedent but do not establish the
required independent secondary-amine acyclic product control. The PDF is present; this is a
branch-specific evidence gap, not a missing-file issue. See the source review below.

The recovered frozen B5 precursor exclusion still protects all 80,398 B5 source records. Thus
all-23-family training is incompatible with the existing frozen exclusions. No exception or scope
change is authorized by this checkpoint. The earlier user scope question remains pending.

Validation: fresh vendor verification passes, as do formatting/lint of the new workflow scripts.
All 2,438 production/config/test snapshot paths are unchanged from validation v7, so its recorded
1,117 focused passes and full-suite result are reused rather than rerun. The full suite remains
3,727 passed, 131 failed, 17 setup errors, and 92 skipped/xfail. Missing historical test artifacts
and the repository validation gate remain unresolved. Final balanced weights, joint program
representation and the admitted training artifact are not yet built. No training was launched.

Receipts:

- [Current worklist](all-family-worklist.json)
- [Merged chemistry evidence](combined-replay-audit.json)
- [Exact failed-check inventory](residual-check-audit.json)
- [Full representation reconciliation](representation-audit.json)
- [Miao acyclic source review](miao-acyclic-source-review.json)
- [Validation reuse check](validation-reuse.json)
- [Pin audit](pin-audit.json)
- [B5 holdout conflict](../compose_lipid_training_readiness_v7/b5-frozen-holdout-conflict.json)

The published replay requests bind the source inputs, implementation, worker counts and seed.
Completed runs are frozen. Their read-only pin verification is runnable with:

```bash
.venv/bin/python results/phase1/compose_lipid_training_readiness_v8/audit_pins.py
```

The checkpoint was constructed by `merge_incremental.py`, followed by
`merge_incremental.py iphos`, `build_worklist.py`, `audit_residuals.py` and
`verify_validation_reuse.py`. Reconciliation uses temporary databases and rejects attempts to
overwrite a completed evidence database.
