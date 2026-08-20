# Reproducible experiments

## Install

```bash
uv venv
source .venv/bin/activate
uv sync --frozen --extra dev --extra torch
```

Use `--extra modal` only on machines that launch remote work. `uv.lock` is the software identity used
by run manifests.

Data and provenance maintenance are also CLI-owned:

```bash
forge data vendor                 # writes only exact hash-matching assets
forge data verify --allow-partial
forge provenance verify --expect-verified 742
forge provenance verify --code --expect-verified 2090 --allow-drift 72
forge provenance archive
```

The `--allow-drift 72` value is the existing config-pin burn-down ratchet, not permission to add
drift. The result-facing gate remains 742 verified pins with zero drift.

## Inspect before running

```bash
forge experiment list
forge doctor phase1-corpus
forge experiment plan phase1-corpus --profile full --backend local
forge experiment plan phase1-corpus --profile full --backend modal
```

`doctor` reports every missing or changed input independently. It does not fabricate or download data.
The Modal plan is a dry run: it reports exact upload hashes and the resource envelope without creating a
cloud job. Its run id remains pending until the actual remote runtime and accelerator are captured.

## Local execution

```bash
forge experiment run installation-smoke --profile smoke
forge experiment verify installation-smoke --profile smoke

forge experiment run phase1-corpus --profile full
forge experiment verify phase1-corpus --profile full
forge experiment reproduce phase1-corpus --profile full

forge experiment run phase1-training-smoke --profile smoke
forge experiment verify phase1-training-smoke --profile smoke

forge experiment run phase1-sampling --profile smoke
forge experiment verify phase1-sampling --profile smoke
forge experiment reproduce phase1-sampling --profile smoke
```

Outputs live under `runs/<experiment>/<run_id>/`. They are ignored by Git until deliberately promoted
to `results/` through the scientific review process. A repeated command refuses an existing stage;
`--resume` verifies and reuses complete stages. Failed stages retain a fingerprint-bound
`.partial` directory. Training checkpoints, optimizer state, Python/NumPy/PyTorch random states,
loss history, and selection state remain there, so the matching run can continue from its latest
evaluation checkpoint. A partial directory from a different source/config/input fingerprint is
rejected. Scratch checkpoints are removed only after the completed stage commits atomically.

Profiles declare their number of statistical replicates. Select one explicitly with `--replicate N`.
The runner derives an independent keyed root seed and records both the master seed and replicate seed.
Paired arms use the same named stream in the same replicate.

`reproduce` executes twice from empty temporary roots. It fails if any stage declared `strict`
produces different artifact bytes. Statistical stages are reported separately and are evaluated over
their declared replicate matrix rather than being mislabeled byte-deterministic.

## Training and sampling contracts

`phase1-training-smoke` verifies the frozen 112,386-record tensor cache, then exercises the joint
sparse flow and closure trainer on CPU. Its checkpoints are qualification artifacts, not production
models. `phase1-training-production` uses the same stage APIs with the frozen all-fold,
fixed-final-step contract and an explicit L4 resource declaration:

```bash
forge doctor phase1-training-production
forge experiment plan phase1-training-production --profile full --backend modal
forge experiment run phase1-training-production --profile full --backend modal
```

The last command creates a paid remote job and is intentionally never called by tests or smoke
targets. Production checkpoints must be reviewed and hash-pinned before a later sampling spec can
consume them.

`phase1-sampling` is a separate, restartable diagnostic DAG over the already selected development
checkpoint and matched program draw. It uses contiguous shards, independently derived shard seeds,
no retry or repair loop, and exact Ugi-L1 forward reconstruction. It performs zero route, synthesis-
value, or biological guidance calls and does not lock or rank candidates. The strict smoke profile
must reproduce byte-for-byte; the full profile declares three independent replicates.

## Modal execution

```bash
uv sync --frozen --extra dev --extra torch --extra modal
forge experiment run phase1-corpus --profile full --backend modal
```

The single generic launcher uploads only the experiment spec, lockfile, pinned configurations, and
pinned external inputs. It builds the locked image, allocates the maximum declared resource envelope,
runs the same registered stage functions with backend identity `modal`, verifies the run remotely, and
downloads it to the ordinary local `runs/` directory. There is no silent local fallback.

Modal execution uses the persistent `forge-experiment-runs` volume. A content-addressed request
workspace makes retries safe. `--resume` may reuse verified committed stages or continue a matching
fingerprint-bound partial stage.

Verify a downloaded run independently of the local hardware:

```bash
forge experiment verify-run <run_id>
forge experiment status <run_id>
```

This re-hashes every downloaded artifact against the remote manifests. Re-executing the pinned spec is
the stronger reproducibility check.

## What every run records

- source-tree, experiment, config, input, dependency, output, and lockfile hashes;
- Python/package/platform/accelerator identity;
- explicit resources, device, precision, workers, and determinism mode;
- master, replicate, stage, and stage-derived seeds;
- stage metrics, summaries, failures, output rows, byte counts, and schema versions;
- scientific nonclaims carried by the experiment spec.

Never edit a completed run. A changed input, source tree, resource declaration, backend, or environment
creates a new identity; a changed artifact fails verification.

## Paper reproduction

The authoritative paper has a separate strict contract:

```bash
forge paper verify
forge paper reproduce
forge paper doctor --strict
forge paper render
forge paper build
forge paper bundle
```

`paper verify` checks the exact manuscript, twelve numeric evidence roots, three generated LaTeX
files, and ten included figure files. `paper doctor --strict` recursively follows every embedded
`{path, sha256}` record. It reports full recomputation as unavailable when original external paths,
remote checkpoints, or engine outputs are absent; those gaps are not hidden by the presence of final
JSON. `paper render` is explicit because it rewrites frozen publication outputs and then requires
their bytes to match the contract. The chemist-packet sample pages are verify-only because their
historical producer also writes an internal blind key.

`paper reproduce` verifies those exact evidence bytes, then performs two independent clean PDF and
Overleaf-bundle builds and requires byte identity. Its receipt says
`numerical_recomputation_executed: false`; it is artifact replay, not a substitute for the strict
training-and-analysis rerun.

`paper build` compiles in a temporary clean directory and atomically publishes only the PDF under
`build/paper/`. `paper bundle` creates a deterministically ordered/timestamped Overleaf zip containing
only the source, generated includes, styles, bibliography, and figures actually referenced by LaTeX.

The retirement inventory is reproducible too:

```bash
make code-survey
```

It roots reachability at the supported CLI, registered experiments, direct paper producers, every
active source path in the recursive paper evidence graph, and legacy scripts that name those
artifacts. A file is a retirement candidate only when no supported root reaches it and all of its
historical pin identities are already archived. The survey itself deletes nothing.

After a complete clean-cache test run, produce the machine-readable baseline comparison with:

```bash
pytest -q --tb=no --cache-clear
make test-baseline-report
```

The report intersects pytest's `lastfailed` cache with the exact collected-node cache, compares that
set with `tests/baseline_failures.txt`, and pins a normalized observation plus the missing-input
inventory. A `blocked_known_failures` result is evidence that no new regression was introduced; it is
not a green full suite.
