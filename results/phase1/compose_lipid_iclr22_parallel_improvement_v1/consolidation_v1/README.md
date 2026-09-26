# Consolidated research worktrees

The five campaign worktrees are consolidated into the main working tree. This action adds
reviewed tools and records; it does not change selected molecules or promote experimental defaults.

| Stream | Already in main | Added to main | Preserved only as historical source |
|---|---:|---:|---:|
| A: learned contribution | 0 | 18 | 10 |
| B: structural quality | 2 | 8 | 4 |
| C: generalization | 3 | 0 | 0 |
| D: route evidence | 0 | 7 | 26 |
| E: diversity and joint selection | 5 | 6 | 2 |
| Total | 10 | 39 | 42 |

`inventory.json` records the original hashes and main-workspace state. All 91 original additions
are preserved under `archive/`; `plan.json` records each file's disposition. `main_adaptations.patch`
contains the eight location corrections and small lint cleanups. `result.json` records final
source hashes, validation receipts and preservation checks. Existing worktrees remain intact
because historical evidence binds their paths and source bytes. Unrelated worktrees are unchanged.

Validation: 102 focused cases pass, Ruff and Black pass for all 39 additions, and all 29 new
non-test modules import from main with correct repository roots. No experiment was rerun.

```bash
UV_CACHE_DIR=/private/tmp/forge-iclr22-uv-cache make test-one \
  UV_RUN='env PYTHONPATH=.:experiments/phase1/route_improvement uv run --offline --no-sync' \
  TEST='tests/test_saved_pool_attribution.py tests/test_saved_pool_gate_join.py tests/test_matched_context_null.py tests/test_matched_context_execution.py tests/test_component_feasibility_census_v2.py tests/test_novel_topology_feasibility_v2.py tests/test_joint_component_selection_v2.py experiments/phase1/route_improvement/test_candidate_routes.py experiments/phase1/route_improvement/test_residual_debt.py tests/test_new_clock_routes.py'
```

Historical producers use frozen, one-shot output directories. Do not rerun them over existing
results or rewrite their protocols. Source-bound admissions remain attached to their original
validator. Future executions from relocated code require new versioned protocols/admissions.
The pure `joint_component_policy.py` remains the reviewed reusable eligibility API; the copied
joint experiment producer is its historical cohort-specific counterpart, not a replacement.

Native-null scientific inference must use the immutable, already qualified
`results/phase1/compose_lipid_iclr22_table_completion_v1/null_runtime/source` first on `PYTHONPATH`
and invoke the runner by direct file path. Main-checkout unit tests do not qualify main's model
as the trained-null model. The paid retry remains unsubmitted pending explicit approval.

Evidence is shared under the campaign directory:

- [Current all-family metrics](../FAMILY_METRICS.md): exact L1 1,387/1,408; makeability 624/1,408.
- [Component-support census](../b_quality/component_feasibility_v1/README.md).
- [Whole-component topology audit](../b_quality/novel_topology_feasibility_v1/README.md).
- [Residual route gaps](../d_routes/residual_debt_v1/README.md).
- [Read-only training monitor and collector](../a_attribution/null_fullfit_v2/MONITOR_COLLECTION.md).

No commits or worktree deletions were made. The manuscript and default policies are unchanged.
