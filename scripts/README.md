# Script migration policy

`scripts/` is a shrinking historical evidence-producer surface, not a user interface. Data install,
provenance maintenance, paper build/bundling, current training, sampling, corpus orchestration, and
generic Modal dispatch now live in the installed CLI/package. Their former top-level scripts were
removed after their historical identities were archived.

## Supported entry points

Current experiments run through the installed CLI:

```bash
forge experiment list
forge doctor phase1-training-smoke
forge experiment run phase1-training-smoke --profile smoke
forge experiment run phase1-training-production --profile full --backend modal
forge experiment run phase1-sampling --profile smoke
forge data verify --allow-partial
forge provenance verify --expect-verified 742
forge paper verify
forge paper reproduce
forge paper build
forge maintenance survey --output provenance/code-retirement/iclr2027.json
forge maintenance test-report --output results/maintenance/bio_to_potency_migration_v1/test_baseline.json
```

No top-level script is a supported operational entry point. The remaining scripts are retained only
when the ICLR reproduction contract reaches them, the recursive evidence graph identifies them, or a
historical path/hash identity is not yet safely retired. Publication generators are invoked through
`forge paper render`; their current location is transitional.

## Legacy categories

- `m0_*` and most `phase1_*` files are frozen result producers or audit entry points.
- Only four `modal_phase1_*` files remain: three are direct ICLR numerical producers and one has
  unresolved historical pin identities. The other experiment-specific launchers were removed.
- figure, table, and manuscript builders are publication tooling and remain separate from model
  execution.

The machine-readable classification is `provenance/code-retirement/iclr2027.json`. It distinguishes
paper/CLI reachability, archived pin identities, and retirement candidates. A candidate is not deleted
in bulk merely because static reachability says it can be; unique acquisition, adjudication, or
negative-result logic still receives human review.

Do not add new top-level workflow scripts. Add typed domain functions, register an experiment stage,
and expose the workflow through `forge`. A legacy script can be removed only after all of the
following hold:

1. a registered CLI DAG reproduces its relevant frozen outputs or the script has no surviving output;
2. every result/config/test/Make target reference has been migrated;
3. every historical source pin resolves through `provenance/frozen-code/`;
4. the script contains no unique acquisition, audit, or evidence-adjudication procedure.

Until then, retained legacy producers are frozen in place: fix shared domain functions, not their
command-line shape, and do not use them as templates for new work.
