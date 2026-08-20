# FORGE architecture

## One dependency direction

```text
forge/ scientific library
        ^
        |
experiments/<app>/ scientific workflows ----> experiments/_runtime/ DAG execution
        ^
        |
cli/ command parsing and dispatch
```

`forge/` contains reusable scientific behavior: chemistry, assembly, corpora, flow/model
primitives, potency interfaces, and synthesis routing. It never imports `cli` or `experiments`.

`experiments/_runtime/` contains the generic immutable-DAG runner, resource declarations,
backends, manifests, and deterministic seed handling. It may reuse `forge.core` byte-stable records
and hashing, but it cannot import a scientific stage or application. It also resolves old
path-plus-digest identities through the content-addressed historical archive.

A named experiment application composes both layers. Its `stages.py` adapts a verified run context
to scientific APIs, and its JSON specifications and local configs live beside that code. The CLI
loads the explicit allow-list in `experiments/catalog.py`; JSON cannot name arbitrary Python.

`cli/` is top-level because command parsing is an application concern, not molecular science.
Likewise, paper builders and repository tooling live under `paper/` and `tools/`.

## Repository layout

```text
forge/
  assembly/        registry-backed final assembly
  chemistry/       molecular identity, reactive sites, graph descriptors
  core/            byte-stable records, hashing, I/O, keyed seeds
  corpus/          corpus identities, splits, and source-balanced policies
  flow/            reusable discrete-flow sampling primitives
  model/           graph representations and neural architectures
  potency/         endpoint, oracle, and applicability APIs
  synthesis/       planners, evidence, terminals, assessments, structured values
cli/               the installed `forge` command
experiments/
  _runtime/        stage/DAG/runtime/backend machinery
  installation_smoke/
  phase1/
    corpus/
    product_l1/    one app containing both training and sampling
    hela_potency/
    synthesis_guidance/
  archive/         runnable historical producers and nonselecting reports
paper/             manuscript build and verification
tools/             data, provenance, and repository-maintenance support
```

There is deliberately no `src/` indirection and no top-level `scripts/` workflow surface. There
are also no catch-all `forge.audit`, `forge.bio`, `forge.cli`, `forge.data`, `forge.design`,
`forge.experiment`, `forge.route`, or `forge.value` namespaces. The executable rules are in
`tests/test_architecture_boundaries.py`.

## Scientific versus experimental code

A module belongs in `forge/` when it can be called with typed/in-memory inputs, returns scientific
objects or values, and does not know a run directory, experiment arm, report filename, or CLI.

A module belongs in an experiment application when it selects a frozen cohort/config, coordinates
training or sampling, writes a result ledger, compares arms, or implements a milestone-specific
gate. Post-hoc reports and old direct entry points belong in `experiments/archive/`, even when they
remain runnable for exact reproduction.

Active applications may never import `experiments.archive`. Anything needed by a current workflow
must live in that application or in `forge/`; an executable architecture test enforces this rule.

This boundary also fixes provenance. Run identities hash the complete executable source set:
`forge/`, the runtime, the catalog, and active experiment applications. Archived producers are not
part of a new run identity. Every spec/config/input/output retains its own path and SHA-256.

## Training and sampling

Training and sampling are two workflows in the same `product_l1` application because they share one
model, tensor-cache schema, graph support, and exact-L1 contract. They are separate DAGs:

- `training_smoke.json` and `training_production.json` build resumable checkpoints;
- `sampling.json` consumes already frozen checkpoints and writes restartable sample shards.

Neither DAG performs route, synthesis-value, or biological guidance. Those authorized diagnostics
have their own applications and cannot silently enter the product prior.

## Historical provenance

Refactoring does not rewrite the identity of completed work. Old `{path, sha256}` references resolve
to exact bytes in `provenance/frozen-code/`; current code uses current paths. `forge provenance
verify` checks both without treating a move as permission to change bytes or relax a gate.
