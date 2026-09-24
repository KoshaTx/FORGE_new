# Complete COMPOSE universe accounting

The receipt accounts for all 3,182,837 original records: 23 reaction-program families and
268 reference-only records in three reference categories. The 200,000 selection is not a
training cap. All 2,982,837 records outside it remain represented in this ledger.

There are 71,932 currently excluded rows, including 1,660 additional exact source-string
aliases of existing exclusions. Another 129,728 inspected rows remain pending global
qualification, and 2,981,177 outside-selection rows remain pending partition before
decomposition. These are conservative preparation dispositions, not final training counts.
All rows have `training_admitted: false`.

The existing precursor audit supplies exact program evidence for 47,363 rows across eight
families, including 27,731 clear of its known exclusions. The remaining families and the
outside-selection population still need qualification. Lack of an exact source-string match
does not establish product or precursor disjointness.

The ledger preserves 8,774 rows in repeated source-string classes. They must not become
duplicate model weight. Source-declared molecular sizes span 14–254 heavy atoms, with
122,975 rows above 96. These metadata have not been promoted to an RDKit representation
qualification. No size-based exclusion was applied.

The 135 MB compressed metadata ledger is intentionally excluded from Git. Its hash is pinned
in `result.json`; rebuild into a fresh directory with:

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_universe \
  --config configs/multireaction/compose_lipid_v8_universe_v1.json \
  --output-dir results/phase1/compose_lipid_v8_universe_readiness_replay
```

Verify the current receipt with:

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_universe \
  --verify results/phase1/compose_lipid_v8_universe_readiness_v1/result.json
```

Verification authenticates the import and precursor audit, then replays every output row and
the complete summary. Molecular strings stay inside SQLite for positive overlap matching;
held-out and unpartitioned product graphs are never parsed or returned to Python.
