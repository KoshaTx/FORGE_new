# Training preparation checkpoint — 2026-09-21

The full-corpus partition is qualified. Training admission remains unqualified.

| Check | Current result |
| --- | ---: |
| Source records accounted for | 3,182,837 |
| Eligible for program preparation | 1,272,466 |
| Protected by frozen exclusions | 1,910,371 |
| Unresolved nonprotected records | 0 |
| Exact computed reconstructions on eligible records | 157,832 |
| Eligible records still needing chemistry qualification | 1,114,634 |
| Formal families with eligible records | 22 / 23 |

The new incremental replay evaluated 109,718 Michael-family records in 365.15 seconds with four
CPU workers. It added 109,567 exact reconstructions and retained 151 unsupported role tuples.
All 430 completed shards were independently reconciled to the eligible partition and complete
source recipes. Serial and parallel outputs agree exactly on 27 samples, including the largest
eligible molecule in each replayed family. Shards can be resumed after interruption; completed
published runs remain frozen.

The recovered historical holdout contains the shared vitamin-B5 core. That precursor protects
80,397 B5 records; the remaining B5 record was already protected. **No B5 record can enter training
under the existing frozen exclusions.** All-23-family training and preserving those exclusions
cannot both pass. The user scope decision remains pending; this checkpoint authorizes no exception.

No size filter was applied. Source support reaches 254 heavy atoms; current eligible support
reaches 250, including 68,358 records above 96 atoms. All complete precursor IDs, roles and
quantities are preserved. Computed reconstruction does not imply an experimentally executed route.

Validation: 1,117 focused tests and vendor verification pass. The full suite reports 3,727 passed,
131 failed, 17 setup errors and 92 skipped/xfail, with no new failures or errors versus v6.
The repository-wide gate still fails. No training was launched.

Receipts:

- [Current worklist](all-family-worklist.json)
- [Full partition](../compose_lipid_full_partition_v1/audit-v2/result.json)
- [Current chemistry reconciliation](incremental-replay-audit.json)
- [B5 holdout conflict](b5-frozen-holdout-conflict.json)
- [Completed parallel replay](../compose_lipid_full_replay_v1/michael/result.json)
- [Validation](../compose_lipid_training_goal_validation_v7/validation_report.json)

Next: qualify the remaining eligible recipes with the established source-derived programs,
resolve unsupported source scopes, validate full-size graph/program representation and freeze
balanced training weights. The final training artifact and admission gate remain unbuilt.
