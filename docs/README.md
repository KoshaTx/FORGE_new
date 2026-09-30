# Documentation map

FORGE keeps scientific decisions and negative results because they are part of the evidence chain.
The flat filenames are retained for now: many are linked by frozen results, decision records, and
automation. This index provides the hierarchy without creating path churn during the code refactor.

## Reviewing the submitted paper

Start with the [submitted PDF and reviewer guide](../paper/submission/README.md),
[paper-to-code map](../paper/submission/EVIDENCE.md) and
[data/checkpoint availability](../paper/submission/ARTIFACTS.md).
The research plans below preserve earlier decisions; they are not a substitute for this submission.

## Contributor and historical context

1. `../AGENTS.md` — authorized scope and non-negotiable scientific constraints.
2. `PLAN.md` — scientific plan; sections 1–5 and 13 are required context.
3. `PHASE1_UGI_FIRST_PRODUCTION_PLAN.md` — frozen Phase 1 execution order.
4. `DECISION_LOG.md` — chronological decisions, including negative results.
5. `MULTIREACTION_COMPUTATIONAL_PLAN.md` — bounded ICLR multi-reaction extension.
6. `PAPER_EXPERIMENT_EXECUTION_MATRIX.md` — current computational execution and prospective evidence gates.
7. `PAPER_RESULTS_TASKS.md` — active ordered checklist for completing the computational paper results.

## Current engineering contracts

- `ARCHITECTURE.md` — package boundaries and allowed dependency direction.
- `REPRODUCIBILITY.md` — local and Modal experiment execution.
- `../configs/reproduction/iclr2027.json` — archived **v0** paper/evidence contract; not the current submission.
- `REFACTOR_BASELINE.md` — inherited test and artifact baseline.
- `DATA_PROVENANCE.md` — data, repository migration, and artifact provenance.
- `provenance_check_audit.md` — strictness differences in legacy pin validators.

The JSON/text files `artifact_path_moves.json`, `known_artifact_drift.json`,
`frozen_source_files.txt`, `migration_exclusions.md`, and `missing_test_inputs.txt` are engineering
inputs or generated inventories. They are not general narrative documentation and must not be
deleted as clutter.

`../provenance/code-retirement/iclr2027.json` is the generated code-reachability and retirement
inventory. Its committed v1 snapshot predates the current three-family and 22-family code map.
Regenerate it with `make code-survey`; do not hand-edit its classifications or use an unreached
classification as permission to delete a study's code.

## Scientific milestone records

- `M0_TASKS.md`, `M0_REPORT.md`, and `M0_*.md` preserve M0 acceptance evidence.
- `PHASE1_*.md`, `SINGLE_STEP_*.md`, `INDEPENDENT_L2_FORWARD_RESOLVER.md`,
  `OFFLINE_SINGLE_STEP_PROPOSAL_BACKEND_DECISION.md`, and
  `RETROCHIMERA_PROPOSAL_BACKEND_PREFLIGHT.md` preserve Phase 1 contracts, audits, and negative
  results.
- `provenance/` maps paper claims to exact computational evidence.

These records may be superseded by a newer version, but superseded is not the same as disposable.
Move them to a future archive only after every inbound reference and artifact pin is updated or
served by the historical-byte archive.

## Manuscript working records

`MANUSCRIPT_*.md`, `FORGE_*EXPERIMENT*.md`, `FORGE_IMPLEMENTATION_REFERENCE.md`,
`FORGE_APPENDIX_MIGRATION_MAP_v1.md`, and `FORGE_PENDING_MANUSCRIPT_CHANGES.md` support manuscript
assembly and claim reconciliation. They may be consolidated after the manuscript source no longer
references them; they are not runtime documentation.

## Historical snapshots

`CLAUDE_LOSSLESS_HANDOFF_2026-08-05.md` is a detailed dated handoff. It is retained as historical
provenance, but its resume point and local prohibitions are not current authority. `AGENTS.md`, frozen
configs, and later entries in `DECISION_LOG.md` control current work.

## Removal policy

A document may be deleted only when all four checks pass:

1. it is not named by `AGENTS.md`, `CLAUDE.md`, the README, a config, or another authoritative doc;
2. no result artifact pins its path and bytes, or those bytes are available in `provenance/frozen-code`;
3. it contains no unique scientific decision, negative result, input hash, or claim qualification;
4. its surviving information has an identified authoritative home.

The root migration note met these conditions after its enduring provenance moved into
`DATA_PROVENANCE.md`. No scientific milestone record currently does.
