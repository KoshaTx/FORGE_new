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
loads the explicit allow-list in `experiments/catalog.py`. Each Phase 1 application owns its entries
in a local `specifications.py`; the catalog merges them and rejects duplicate identifiers. JSON
cannot name arbitrary Python.

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
    multireaction/ shared reaction-program experiments
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

## Three-family and 22-family studies

See [STUDIES.md](STUDIES.md) for each study's entry points, evidence, and verification commands.

Both studies remain supported. The three-family manuscript is under `paper/v1_iclr/`; the newer
22-family computational manuscript is under `paper/v1_iclr22/`. Their experiments share the
`multireaction` application and reusable `forge/` modules. A file in that application is not
obsolete merely because one study no longer calls it directly. The 22-family work also has
versioned configs and result-producing analysis scripts in the same application.

The application-owned specification lists organize the supported CLI workflows without moving
their JSON files or changing any experiment identifier. The many historical producers under
`experiments/archive/` remain available for exact reproduction. The code-retirement survey uses
the older `configs/reproduction/iclr2027.json` paper contract; its `review_unreached` label is a
provisional reachability result, not a deletion decision for either supported study. Its older
v1 receipt used the misleading name `retire_candidate` and is superseded by the v2 survey contract.

Within `multireaction/`, `foundation_stages.py` owns the original corpus, training,
sampling, and overfit adapters; `reaction_specialization_stages.py` owns the bounded
specialist and preflight variants; `compose_lipid_training.py` owns the newer COMPOSE
training adapter. `stages.py` keeps the remaining shared adapters and re-exports moved
functions at their previous import paths. `specifications.py` lists the application's
CLI experiment IDs. These modules share the same stage registry, with no change to the
registered identifiers.

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

The product/L1 `stages.py` retains corpus adapters and compatibility exports. Its sibling
`training_stages.py`, `sampling_stages.py`, `evaluation_stages.py`,
`specialist_evaluation_stages.py`, and `semantic_evaluation_stages.py` own the corresponding
adapters. `_stage_support.py` holds their shared publication and restart helpers. Scientific
imports remain inside stage functions; importing these adapters does not initialize a model.

The shared sampler keeps its public entry points and flow loop in
`forge/model/synthesis_program_sampling.py`. Its private `_synthesis_sampling` package separates
contracts, checkpoint loading, state operations, constraints, and strict decoding. Both study
generations use these components. Terminal tensor and array records have explicit typed fields;
the topology-only array record deliberately contains only the three topology fields.

`SAMPLING_SOURCE_FILES` lists the facade and every private sampler module for producers that
record individual source pins. The generic runtime already fingerprints the full active source
trees. Moving a helper requires preserving its historical bytes and including its new module in
current source inventories; keeping only the facade's hash would omit executable dependencies.

## Historical provenance

Refactoring does not rewrite the identity of completed work. Old `{path, sha256}` references resolve
to exact bytes in `provenance/frozen-code/`; current code uses current paths. `forge provenance
verify` checks both without treating a move as permission to change bytes or relax a gate.
