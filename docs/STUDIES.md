# Supported computational studies

For the reviewer-facing three-family submission, start with
[`paper/submission/README.md`](../paper/submission/README.md). The exact supplied PDF is newer than
the `v1_iclr` source below. The 22-family extension is a separate study, not part of that paper's
reported three-family comparisons.

Both study generations remain supported. They share scientific implementations; the family
count is a study definition, not a rule for deciding whether a source file is obsolete.
`AGENTS.md` controls the current computational scope and execution authorization.

| Where to start | Original three-family study | Newer 22-family COMPOSE study |
|---|---|---|
| Manuscript and evidence navigation | [Manuscript](../paper/v1_iclr/README.md), [computational plan](MULTIREACTION_COMPUTATIONAL_PLAN.md) | [Manuscript and evidence ledger](../paper/v1_iclr22/README.md) |
| Experiment adapters | `experiments/phase1/multireaction/stages.py`, `foundation_stages.py`, `reaction_specialization_stages.py` | `experiments/phase1/multireaction/compose_lipid_training.py`, `compose_lipid_run.py` |
| Config entry points | `configs/multireaction/shared_production_training_v1.json`, `shared_production_evaluation_v1.json` | Versioned `configs/multireaction/compose_lipid_training_recipe_*.json` and `compose_lipid_training_measure_*.json`; choose the version declared by the result being inspected |
| Scientific implementation | `forge/model/synthesis_program_training.py`, `synthesis_program_sampling.py`, and reaction-program modules | `forge/corpus/compose_lipid_training_data.py`, `forge/model/compose_lipid_training.py`, COMPOSE layout/generation modules, and shared reaction-program/decoder code |
| Existing evidence | Versioned `results/phase1/shared_synthesis_program_*` and their input pins | Versioned `results/phase1/compose_lipid_*`, linked by the manuscript evidence ledger |
| Route interpretation | Preserve the metric and qualification checks declared by each historical result | [Computational makeability contract](COMPOSE_LIPID_COMPUTATIONAL_MAKEABILITY.md); strict dossier closure remains a separate metric |

## Find a workflow

`forge experiment list` lists the supported catalog identifiers. The catalog lives in
`experiments/catalog.py`, with application-owned `specifications.py` lists. The three-family
workflow includes `phase1-multireaction-corpus`, `phase1-multireaction-training-smoke`, and
`phase1-multireaction-overfit`; these are execution commands, not documentation checks.
The newer COMPOSE adapter registers `model.compose_lipid.training.v1` and authenticates its
versioned recipe and admission inputs before execution. A shared-program name alone does not
identify a 22-family experiment.

The older `configs/reproduction/iclr2027.json` contract names the manuscript under `paper/v0/`.
It is a historical reproduction contract, not a complete inventory of either study above.
Unreached files in its code survey require manual review across both studies.

## Check a change locally

- `make test-study-compatibility` checks both study interfaces and shared decoding on small
  fixtures, without training or remote submission.
- `make test-one TEST=tests/<owning-test>.py` is the default while editing.
- `make verify` verifies vendored input hashes.
- `make test-training OUTPUT=results/phase1/<new-validation-directory>` is the separate current
  COMPOSE readiness checkpoint described in [TESTING.md](TESTING.md), not a training launch.

Compatibility fixtures are engineering checks. They do not reproduce full study results or admit
new data. Existing evidence keeps its original source/config/input hashes, denominators, and
metric definitions; refactoring never makes an old result a measurement of the new source tree.
