# FORGE Claude bootstrap

**Read `MIGRATION_NOTES.md` first.** This repository was reorganized on 18 August 2026. The paper
now lives in `paper/`, not `manuscript/`; the evidence records moved to `docs/provenance/`. More
importantly, 82 of the phase-1 result artifacts the ledgers cite are **not present** — they were
gitignored and left on the machine that produced them. Numeric truth is still the hash-pinned
artifact wherever one exists, but for those 82 the surviving record is the provenance documents and
the `\newcommand` macros at the top of the paper. Do not assume an artifact is on disk; check.

Before doing any work in this repository, read these files completely and in this order:

1. `AGENTS.md`
2. `docs/CLAUDE_LOSSLESS_HANDOFF_2026-08-05.md`
3. `docs/PLAN.md` sections 1--5 and 13
4. `docs/PHASE1_UGI_FIRST_PRODUCTION_PLAN.md`

For code, pipelines, tests or result artifacts, also read:

- `.agents/skills/forge-production-engineering/SKILL.md`

For manuscript prose, captions, evidence reconciliation or reviewer-facing text, also read:

- `.agents/skills/forge-paper-writing/SKILL.md`
- `docs/provenance/NATURE_BIOTECH_DRAFTING_ANALYSIS.md`
- `docs/MANUSCRIPT_EDITORIAL_GUIDE.md`

For chemistry-source adjudication, also read:

- `.agents/skills/forge-adjudicate-chemistry-evidence/SKILL.md`

`AGENTS.md` is the repository's scientific-validity contract. The dated lossless handoff is the
authoritative continuation state for work completed after older plans and manuscript prose. Frozen
configs and hash-pinned result artifacts remain the source of numeric truth.

## Current resume point

Resume at Section 12 of the lossless handoff. The final generator, applicability proposal, potency
decision, synthesis-guidance decision and branch-support correction are closed. The current input is
the 256-product route-blinded shortlist v2. The immediate task is one identical, versioned,
proposal-augmented Graph2Edits + AiZynthFinder + evidence-aware L1/L2/L3 route assessment, followed by
route-attrition and candidate-balance auditing and a route-aware decision package.

Do not:

- retrain the generator;
- retune applicability, potency or synthesis guidance on the same evidence;
- revive partial-state SMC or MH;
- weaken any gate;
- treat a route proposal as evidence;
- call unresolved chemistry unsynthesizable;
- create `PREREGISTRATION.md` or lock a prospective panel;
- invent prospective synthesis, formulation or biological results;
- make final manuscript figures without explicit user authorization;
- reset, clean or overwrite the dirty worktree.

Use `apply_patch` for edits, create versioned artifacts, pin input SHA-256 values, record seeds, run
focused checks while iterating and disclose every check not run. Give the user concise status updates
during long work and distinguish implementation, execution, promotion, freeze, route readiness,
panel readiness and prospective validation.
