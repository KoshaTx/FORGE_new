# Paper-model generation quickstart

**Status: wrapper available; end-to-end qualification with the paper checkpoint is pending.**
The paper's step-9,143 checkpoint archives, training records and designs are not distributed yet.
The commands below check for those exact inputs and stop clearly when they are missing. They do
not substitute the earlier Ugi model used by Vessel or the tiny smoke models in this repository.

## Install

From a checkout of this repository, use the locked development environment:

```sh
uv sync --frozen --extra dev --extra torch
uv run --frozen python -m cli.generate --help
```

Linux with Python 3.12 is the CI environment. CPU is the generation command's default; CUDA is an
explicit option. The frozen Torch version may not have a wheel for every platform. Other platforms
are not qualified by the Linux checks. The historical training environment is not recovered by
installing the current lockfile. Paper-model inference time and memory have not yet been measured.

## Check the inputs now

```sh
uv run --frozen python -m cli.generate --check-inputs
```

This mode does not import Torch or deserialize weights. It prints every required file, its expected
and actual SHA-256, and `verified`, `missing` or `mismatch`. Exit code **0** means all listed inputs
match; **2** means something is missing, changed or malformed. Input readiness is not a generation
quality or numerical-reproduction result.

The production cache, program specifications, registries, assignments and splits are already
committed and verified. For seed 0 the remaining files belong in:

```text
results/phase1/shared_bias_parallel_program_role_seed0_v2/
  checkpoints.tar
  training_result.json
  study_design.json
```

Use the exact hashes in [ARTIFACTS.md](ARTIFACTS.md) and [checkpoints.json](checkpoints.json). Download
links will be added when the original bundle is recovered. There is no working download command yet.
Keep the archive intact; the loader reads the authenticated member without extracting files.

Alternatively, keep these three files together elsewhere and specify the archive:

```sh
uv run --frozen python -m cli.generate --check-inputs \
  --checkpoint /path/to/download/checkpoints.tar --replicate 0
```

`--replicate 1` or `--replicate 2` selects the corresponding paper training run and expected hashes.
This differs from `--seed`, which controls the random stream for the new demonstration samples.

## Generate after the bundle is available

```sh
uv run --frozen python -m cli.generate \
  --family ugi --count 4 --seed 42 --device cpu \
  --output build/generation/ugi-seed42
```

Families are `ugi`, `aza-michael` and `reductive-amination`. The wrapper always selects the
`shared_bias_program_role_source` arm, step 9,143, 32 flow steps and
`strict_reaction_core_saturation_argmax` decoding. It reuses the existing checkpoint loader,
training-fold count prior, sampler and exact-L1 verifier. It does not change the historical
evaluation configurations or launch training/cloud jobs.

`--count` is the number of **attempts**, including invalid structures and failed exact-L1 checks.
The example permits 1–256 attempts, with `--batch-size 4` and `--threads 2` by default. It applies
no retry, ranking or top-up. These request limits do not reduce molecular-size support.
For supported hardware, `--device cuda` selects GPU execution; it never silently falls back.
Deterministic algorithms are requested, but equality across devices or software versions is not
asserted. Record batch size as well as the seed when comparing runs.

## Inspect the output

Each run requires a new output directory and writes:

| File | Contents |
|---|---|
| `attempts.jsonl` | Every attempt, full sampler fields, validity and exact-L1 decomposition/forward-replay evidence |
| `molecules.csv` | Every attempt's index, program, canonical SMILES (blank if absent), validity and exact-L1 outcome |
| `summary.json` | Request, counts, derived layout/flow seeds, input/output hashes, source hashes, versions and elapsed time |

Failed attempts remain in the denominator. Outputs are published together after generation and
serialization succeed. Existing output directories are refused. Zero exact-L1 products is a valid
reported outcome, not a reason to rerun until a desired number passes. Exact L1 is a computational
reaction-transform check, not L2/L3 route closure, synthesis success or delivery prediction.

These examples use new sampling streams and smaller budgets than the paper. They do not rerun
the paper's held-out evaluation or establish novelty. See [examples](../../examples/README.md)
for existing illustrations, and [artifact replay](ARTIFACTS.md#data-and-table-regeneration) for
the existing table-rendering commands and their additional inputs.

## Check the wrapper without paper weights

```sh
uv run --frozen pytest -q tests/test_reviewer_generation.py
```

Tests cover missing/corrupt inputs, relocated bundles, request validation, failure handling,
retention of failed attempts, overwrite refusal and provenance. Runtime tests explicitly use the
committed small smoke model and exercise all three reaction programs. That internal test fixture
cannot bypass the public command's paper-model hash checks and is not a paper-generation result.
