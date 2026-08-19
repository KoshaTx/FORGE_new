# FORGE Claude bootstrap

This repository was reorganized on 18 August 2026. The paper lives in `paper/`, evidence records live
in `docs/provenance/`, and historical migration provenance is summarized in
`docs/DATA_PROVENANCE.md`. Asset availability has changed since the migration; never rely on a dated
missing-file count. Use `forge doctor <experiment>` or verify the exact pinned path.

Before doing any work in this repository, read these files completely and in this order:

1. `AGENTS.md`
2. `docs/PLAN.md` sections 1--5 and 13
3. `docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md`
4. `docs/DECISION_LOG.md` entries relevant to the task

Three repository skills live in `.claude/skills/` and are discovered automatically, so they load
when the work matches rather than needing to be opened by hand. Invoke one by name when you want it
early. `.agents/skills` is a symlink to the same directory, kept because each skill also ships an
`agents/openai.yaml` for non-Claude tooling.

For code, pipelines, tests or result artifacts:

- `$forge-production-engineering`

For manuscript prose, captions, evidence reconciliation or reviewer-facing text:

- `$forge-paper-writing`
- `docs/provenance/NATURE_BIOTECH_DRAFTING_ANALYSIS.md`
- `docs/MANUSCRIPT_EDITORIAL_GUIDE.md`

For chemistry-source adjudication:

- `$forge-adjudicate-chemistry-evidence`

`AGENTS.md` is the repository's current scientific-validity contract. Frozen configs and hash-pinned
result artifacts remain the source of numeric truth. The dated lossless handoff is a historical
snapshot: use it for provenance, but do not let its old resume point override later authorizations or
decisions.

Never weaken a gate, treat a route proposal as evidence, call unresolved chemistry
unsynthesizable, invent prospective results, or reset/clean a dirty worktree.

Use `apply_patch` for edits, create versioned artifacts, pin input SHA-256 values, record seeds, run
focused checks while iterating and disclose every check not run. Give the user concise status updates
during long work and distinguish implementation, execution, promotion, freeze, route readiness,
panel readiness and prospective validation.
