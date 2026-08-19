# FORGE architecture

## Design rule

Scientific policy belongs in domain packages. Execution policy belongs in `forge.experiment`.
Domain code must never import the runner, and JSON experiment files may select only registered stage
implementations. They cannot import arbitrary Python.

The target dependency direction is:

```text
core <- chem <- assembly <- corpus <- generate
                         \             \
                          route <- potency
                                  \
                                   audit

all domains <- experiment stage adapters <- CLI / local / Modal
```

`core` owns lossless records, hashing, and byte-stable I/O. `chem` owns structure identity and is the
eventual RDKit boundary. `assembly` owns registry-defined L1 transform interfaces. `corpus` owns data
identity, splits, and sampling policy. `generate` owns model and sampler primitives. `route` owns L2/L3
planning and evidence. `potency` owns oracle fitting, applicability, and authorized ranking. The runner
orchestrates these domains but does not define their chemistry.

## Public boundaries

New callers import from a package API, for example:

```python
from forge.assembly import Ugi3AssemblyAdapter
from forge.corpus import freeze_phase1_data_contract
from forge.generate import rstar_step
```

Cross-package imports of private names are prohibited in migrated packages. The executable rules are
in `tests/test_architecture_boundaries.py`. Three exact transitional edges remain while frozen source
paths are preserved:

- `corpus.phase1 -> product.phase1_data`
- `assembly.ugi3 -> data.r1_prime_audit`
- `assembly.ugi3 -> product.ugi_held_component_gate`

They are compatibility shims, not permissions to add new coupling. Historical implementations remain
available because frozen results pin their bytes. New work uses the target APIs.

The first legacy-domain extraction is complete. `forge.bio` contains only the generic endpoint
interface, the endpoint decision, and liver/muscle/vaccine endpoint implementations. The 49 oracle,
applicability, morphology, ranking, and authorized-diagnostic modules formerly under that namespace
now live in `forge.potency`. Historical config documents deliberately keep the old source path and
resolve it through `docs/artifact_path_moves.json`; current runtime code must not import a removed
`forge.bio.*` module. `product -> potency` dependencies remain transitional until the product package
is split into corpus and generator domains.

## Experiment boundary

An experiment consists of a strict JSON DAG and registered stage functions. Every stage declares:

- a hash-pinned configuration and every external input;
- dependency stages;
- output filenames and schema versions;
- CPU/GPU, precision, worker, memory, and timeout requirements;
- a named deterministic or statistical random stream.

The runner verifies pins before work starts, writes only to a private partial directory, validates
every declared output, writes a manifest, and commits the completed directory atomically. A failed
stage retains fingerprint-bound scratch state for deterministic checkpoint resume; a committed stage
is reusable only when its fingerprint and every artifact byte verify.

Run identities bind the experiment specification, source tree, lockfile/runtime environment, backend,
resource declaration, profile, replicate, and dependency artifact hashes. Deterministic stages must
produce byte-identical artifacts. Statistical stages use keyed streams so paired arms do not drift when
another arm consumes an extra random draw.

## Frozen provenance

Active restructuring cannot invalidate historical results. Exact historical source/config bytes are
stored by digest under `provenance/frozen-code/`; `forge provenance verify` checks the active path
first and then the archive. The three already-unrecoverable digests in
`docs/known_artifact_drift.json` are bound to exact `(path, expected_sha256)` identities. No path-wide or
current-file exception exists.

Regenerate the archive only as a mechanical recovery operation:

```bash
forge provenance archive
make verify-pins
```

The generator refuses any newly unresolved digest. Never add a known-drift entry to make a gate pass.

## Executable entry points

The installed `forge` CLI is the supported interface for data, provenance, paper builds, corpus,
training, sampling, and remote experiment workflows. `scripts/` now contains only retained historical
evidence/publication producers; it is not an operational API. Do not add new top-level workflow
scripts. See `../scripts/README.md` and the machine-readable retirement survey for the removal gate.
