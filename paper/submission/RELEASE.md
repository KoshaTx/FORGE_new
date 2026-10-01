# Reviewer code-release checklist

The source and offline paper checks are available now. Generation with the paper's trained model
is still blocked by unavailable weights and companion inputs. A source release can disclose that
limitation; a release advertised as runnable pretrained generation must resolve it first.

## Recover the model bundle

Start with seed 0 for a bounded generation example. Recover all three seeds for the paper's
across-seed comparison. The exact archive paths and hashes are in [ARTIFACTS.md](ARTIFACTS.md).
For each seed, obtain the matching `checkpoints.tar`, `training_result.json` and `study_design.json`;
their expected identities are recorded in [manifest.json](manifest.json). Preserve the original
files and verify their hashes before loading weights.

Weights alone are insufficient for the current sampler. It constructs its vocabulary and
training-fold count prior from the production cache. The existing evaluation entry point also
requires chemistry registries, assignments and splits. Recover the direct inputs listed in
[`shared_bias_parallel_program_role_seed0_core_saturation_v2.json`](../../configs/multireaction/shared_bias_parallel_program_role_seed0_core_saturation_v2.json):

| Input | Purpose |
|---|---|
| `production_cache` | Vocabulary and training-fold count prior used for sampling |
| `production_design` | Frozen arm and training design |
| `program_config` | Reaction-program specifications |
| `qualified_reaction_families`, `qualified_ugi_reactions` | Chemistry registries |
| `ugi_assignments` | Ugi component assignments used by evaluation |
| `multireaction_atlas`, `multireaction_splits` | Multi-reaction records and evaluation splits |

These are the evaluator's direct dependencies, not a claim that every file is essential to a
future standalone inference interface. Additional inputs are needed for the paper's other tables.
The source configuration remains authoritative for paths and hashes.

On September 30, 2026, a read-only search of both configured Modal workspaces found no
`forge-experiment-runs` volume. All 12 accessible FORGE-prefixed volumes in the active workspace
were inventoried: none contained the documented paper paths, a `checkpoints.tar` archive or a
step-9143 filename. Available training weights were older Ugi/pretraining artifacts. The local
Git clone had no history for the three documented production directories. This search does not
establish that the paper artifacts were deleted or that they are absent from other accounts.
The next retrieval input is the original training workspace or an accessible export of that bundle.

## Qualify a generation example

After recovering the bundle:

1. Verify archive and companion hashes, then verify the selected archive member against the
   training record using the existing strict checkpoint loader.
2. Use the paper arm `shared_bias_program_role_source` at step 9,143. Preserve its vocabulary,
   training-fold prior, 32 flow steps and `strict_reaction_core_saturation_argmax` decoder.
3. Provide and run a bounded example from a fresh checkout, with explicit seed, device and sample
   count. Retain every attempt and its validation outcome, plus input hashes and output provenance.
4. Document the measured runtime, memory, installation command and actual output format. Mark the
   small example as a usability check; it does not reproduce the paper's full evaluation.

Do not use the evaluation config's `smoke` profile as a substitute paper-model example: it requests
steps 1 and 2 with two sampling steps. The full profile has production-scale evaluation budgets.
Neither is an already-qualified lightweight step-9143 demo. No such demo has been run in this cleanup.

## Publish the usable bundle

- Publish stable download locations, sizes, SHA-256 hashes and applicable redistribution terms for
  the weights and required data. Do not replace a missing file with a similarly named model.
- Keep locked installation instructions and CI results, including missing-artifact skips, visible.
  Existing CI verifies software contracts; it does not demonstrate paper-checkpoint inference.
- Resolve release licensing: package metadata declares MIT, but this checkout has no standalone
  license file. Code licensing does not itself establish redistribution rights for weights or data.
- Keep [artifact replay instructions](ARTIFACTS.md#data-and-table-regeneration) separate from fresh
  evaluation, and retain the disclosures about missing historical source/runtime information.

The remaining laboratory details in [EVIDENCE.md](EVIDENCE.md#experimental-record) concern the
experimental claims. They are not prerequisites for releasing source or qualifying a computational
generation example, and this checklist does not require new laboratory experiments.
