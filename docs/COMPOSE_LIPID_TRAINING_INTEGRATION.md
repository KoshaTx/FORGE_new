# COMPOSE cache-to-trainer integration

The preparation cache now has a family-weighted adapter and an optimizer-step function for the
existing sparse MPNN and reaction-program Transformer. This is intermediate engineering preparation;
the actual corpus remains unadmitted. No production training configuration or admission receipt is
issued by this integration.

The current acceptance report is `results/phase1/compose_lipid_training_validation_v2/result.json`:
**95 tests passed in 15.460 seconds**, and all 30 vendor assets verified. This replaces the old
full-repository prerequisite. Current readiness is
`results/phase1/compose_lipid_training_readiness_v16/readiness.json`; historical-results recovery
is no longer requested for this training task. Remaining chemistry qualification and the B5 split
conflict still prevent final data admission.

The supplied Ren, Love and Zhou supplements added **126,373** exact, fully prepared records.
The verified cache now contains **1,150,603** unique constitutional graphs across 28 source
bindings; **121,863** eligible records still need chemistry qualification. The previous 1,024,230
records were reused without re-encoding. The increment passed 26 focused tests, independent atom
inventory and tensor checks, both CPU model ingestion checks, and the combined reader check.
No real-corpus optimizer update, GPU run or final training admission occurred. Receipts:
`results/phase1/compose_lipid_user_supplements_v1/result.json` and
`results/phase1/compose_lipid_unified_preparation_v5/verification.json`.

## Data and admission

`forge.corpus.compose_lipid_training_data.ComposeLipidTrainingData` takes the repository root and
four path/SHA-256 pins: `population`, `verification`, `measure`, and `admission`. It uses the existing
`QualifiedProgramCache` with an explicit bound on cached shards.

The final admission receipt must have schema `forge.compose_lipid_training_admission.v1`,
`training_admitted: true`, `training_ready: true`, and `inputs` binding the same population,
verification and compiled measure, plus a `validation` pin. The validation must be a passing
`forge.compose_lipid_training_validation.v1` receipt, including vendor verification and a current
source snapshot that covers this adapter, training primitive and the declared training checks.
It requires all seven selected test files to execute without failures, errors or skips. It does
not require the full repository suite or any unrelated historical experiment outputs. Generate the
receipt with `make test-training OUTPUT=results/phase1/<new-validation-directory>`; see
`docs/TESTING.md`. The user explicitly removed the full-suite prerequisite on 2026-09-22.
Preparation and weight-compilation
receipts remain immutable with their original false admission flags. A new final admission is a
separate artifact; this module cannot produce one.

After admission, the adapter authenticates the compiled weight inputs and exact-evidence database,
requires complete exact support for all 23 formal families, and checks every weight against the
preparation row's identity, constitution, family and position. Each graph has probability
`1 / (23 * family_graph_count)`. Source-role namespaces receive no separate probability mass.
Weights are compiled by the existing `compile_training_measure` function, not fitted by the loader.

Only a cumulative probability vector is retained for the whole population. Graph loading remains
bounded by the shard cache. Sampling uses a caller-owned NumPy generator and preserves draw order
and repetitions. Batches retain precursor quantities and the source-aware component coordinates;
global component identifiers stay outside neural tensors. A model atom limit below the population's
maximum is rejected even when the requested batch contains only smaller examples. Closure support
is enforced by the existing collator.

## Training primitive

`forge.model.compose_lipid_training.compose_lipid_training_step` draws a batch from admitted data,
collates it with source coordinates, applies the existing noising and masked-loss functions,
backpropagates, rejects nonfinite gradients, clips the gradient norm and performs one optimizer
update. Architecture, device, atom/bond noise marginals, semantic-loss weights, batch size and
molecular support are explicit arguments. Marginal fitting and scientific hyperparameter selection
remain separate tasks.

Sampling applies the graph measure once. Losses retain their existing masked reductions; the
integration does not apply a second probability weight or substitute program-balanced gradients
for formal-family sampling. Sparse MPNN accepts zero auxiliary semantic weights because it lacks
the Transformer's semantic prediction heads.

`compose_lipid_forward_loss` exposes the same computation without an optimizer update for numerical
diagnostics. Model, optimizer and random-generator states remain caller-owned. `data.identity`
contains the four pins a checkpoint must bind. CPU restart qualification and detached-submission
plumbing are described below. Final mapped cache compilation, a frozen execution configuration and
live GPU qualification remain pending.

## Validation

Run the affected file only:

```sh
make test-one TEST=tests/test_compose_lipid_training_data.py
```

The recorded check passes 15 tests in 6.302 seconds on CPU. Tests use synthetic metadata and graph
fixtures, with a real weight compiler and a synthetic shard reader. They check admission rejection,
current-source validation, rehashed weight/order tampering, changed evidence, every sampling
probability interval, duplicate draws, large-graph support and preserved source quantities. Both
architectures give exactly equal loss and gradients to their existing objective primitives and
perform finite fixture optimizer updates. Caller RNG restoration reproduces sampled indices.

The real v11 readiness receipt is separately rejected before data loading or optimizer construction:
`Final all-family training admission is required`. The real corpus receives zero optimizer updates.
No full-suite rerun, GPU execution or paid compute occurs in this check. This historical receipt
predates the scoped validation contract; final training readiness remains unresolved.

Receipt: `results/phase1/compose_lipid_training_integration_v1/result.json`. Its request pins the
source code, fixture dependencies and environment lock. Existing preparation and validation receipts
remain historical records with their original hashes.

## Restartable execution

`experiments.phase1.multireaction.compose_lipid_run.run_training` owns the training lifecycle. Its
configuration has schema `forge.compose_lipid_training_config.v1` and the following explicit fields:

- `inputs`: the four admission/data pins above and a `noise_marginals` pin.
- `model`: the existing architecture configuration, including complete atom and closure support.
- `runtime`: positive `optimizer_steps`, `checkpoint_interval`, `batch_size`, `cpu_threads` and
  `maximum_cached_shards`; `precision: float32`, `workers: 0`, `deterministic: true`.
- `optimizer`: AdamW `lr`, `weight_decay`, `betas` and `eps`.
- `semantic_weights` and `gradient_clip_norm`: the explicit objective and clipping settings.

Noise marginals use schema `forge.compose_lipid_noise_marginals.v1`, with `inputs` binding the same
`population` and `measure`, and normalized `node` and `bond` arrays. This interface consumes fitted
marginals; it does not fit them or select scientific hyperparameters. The uniform marginals in tests
are synthetic fixtures only. Current training validation must also cover the run module, stage
adapter and shared restart primitives before this runner admits data.

The fitting function is now
`forge.corpus.compose_lipid_noise_marginals.compile_noise_marginals(repo, output, *, population,
verification, measure, admission, maximum_cached_shards=4)`. It consumes the same four input pins
as the data adapter and requires final admission before reading graphs. It streams each graph once
in compiled record order with its existing family-balanced probability, using the bounded shard
cache and float64 CPU accumulators. For each atom or bond state `k`, the fitted probability is
`sum_g p(g) * count_g(k) / sum_g p(g) * token_count(g)`; the denominator covers the corresponding
atom or bond tokens. This is a probability-weighted pooled token distribution. Parent bonds exclude
the root sentinel, closure bonds count once, and constitutional bond states are single/double/triple.
No padding, nonedges, source-role reweighting, smoothing or molecular-size cutoff is applied.

All declared states must occur, and the streamed count, maximum molecule size and graph probability
mass must match the admitted population. The completed artifact binds its inputs and implementation,
reauthenticates mutable input files before publication, and is published atomically without replacing
an existing output. The compiler does not create an admission receipt or allocate a model.

The compiler and affected adapter pass **25 focused tests in 6.882 seconds**. An independent weighted
reference covers unequal family sizes, a 100-atom graph, ring closures and all three bond states;
the existing runner accepts the result, and repeated compilation is byte-identical. Checks also cover
missing records, rehashed probability tampering, absent states, write failure and admission rejection.
The actual v14 readiness receipt is rejected before graph loading, with no fitted real-corpus output.
These checks do not replace final data admission. Receipt:
`results/phase1/compose_lipid_noise_preparation_v1/result.json`.

The initial state, each configured update boundary and the final state are saved atomically.
Each checkpoint contains model and AdamW state, completed-update/example counters, last-update
metrics, Python and NumPy random states, CPU/CUDA random states and the explicit sampling/noising
generators. Identity binds the source digest, configuration digest, input pins, seed and runtime.
Restart refuses changed identity or checkpoint bytes. A directory lock excludes concurrent writers.

`latest.json` points to a hashed checkpoint generation. The previous generation remains available
until the new generation and pointer have been passed to the backend's volume committer; only two
generations are retained. Interrupted serialization or pointer publication leaves the previous
pointer usable. A commit failure propagates. Transport random draws are isolated from training RNG.
The caller must request resume explicitly; a completed resume performs no extra optimizer updates.
The final stage exports `checkpoint.pt` and `result.json`.

## Detached submission

The registered stage is `model.compose_lipid.training.v1`. It uses the shared Modal runtime and
`RunContext.commit_progress`, so checkpoints are committed incrementally to the persistent volume.
An experiment descriptor must declare the training config, its named inputs and the complete
execution-file inventory. `training_files(repo, config_path)` returns the required path/hash pins,
including tensor shards, source-recipe files, vocabulary remaps and validation inputs. Add these
under unique extra `stage.inputs` labels; the launch check rejects omissions or changed bytes.
Declare `checkpoint.pt` with schema `forge.compose_lipid_training_checkpoint.v1` and `result.json`
with schema `forge.compose_lipid_training_run.v1` as the two outputs. Resources must use strict
determinism, float32, zero loader workers and sufficient CPU threads.

After final admission and execution configuration, the entry points are:

```sh
python -m experiments.phase1.multireaction.compose_lipid_training plan SPEC.json --profile full
python -m experiments.phase1.multireaction.compose_lipid_training submit SPEC.json --profile full
forge experiment modal-status runs/_modal_calls/REQUEST_ID.json
forge experiment modal-collect runs/_modal_calls/REQUEST_ID.json
```

`plan` is local and allocates no compute. `submit` records the request before invoking the shared
launcher with `--detach --launch-only`. The returned receipt binds the function-call ID, request ID,
source/spec digests and uploaded input hashes. Status and collection reconnect to that call without
submitting work. A submission attempt is reserved exclusively; a disconnect, nonzero launcher exit
or invalid receipt leaves a failure record and blocks repeated submission. A failed paid run still
requires diagnosis and fresh execution authorization before any retry. This command never retries
or resubmits automatically.

The local qualification passes **32 tests in 7.581 seconds**, including the existing restart and
Modal transport tests. Both architectures produce exactly equal model/optimizer/RNG states and
sampled draws after interruption and resume, with dropout enabled. Tests also exercise failed
checkpoint writes, changed identities, corrupt checkpoint bytes, volume-commit wiring, complete
upload inventories, admission rejection and ambiguous submissions. Black and Ruff pass. The source
snapshot is unchanged during validation.

These are CPU fixture and mocked-transport results. Live Modal execution, CUDA numerical equivalence,
GPU throughput, convergence and real-corpus training have not been qualified by this check. The real
corpus remains rejected with `Final all-family training admission is required`; no paid compute or
real-data optimizer updates occurred. The full suite was not rerun.

Receipt: `results/phase1/compose_lipid_restart_validation_v1/result.json`.
