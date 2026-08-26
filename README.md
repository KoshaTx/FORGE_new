# FORGE

**Route-Grounded Generative Design of Synthetically Executable Ionizable Lipids**

Computational engine for a Nature Biotechnology paper on synthesis-grounded whole-lipid generation.
FORGE couples a bounded whole-molecule generator to recursive synthesis-program construction and
records auditable L1/L2/L3 evidence without treating computational route support as synthesis success.

The unit of generation is a **product–route dossier**, not a SMILES string.

> **Working in this repo? Read [`AGENTS.md`](AGENTS.md) first.** It defines the authorized scope and
> nine hard constraints that an agent optimizing for green tests will otherwise violate.

## Status

M0 is complete. Bounded Phase 1 product/L1 training, synthesis-routing readiness, and the single
versioned HeLa diagnostic described in `AGENTS.md` are authorized. Later phases and unrestricted
biological optimization are not. A bounded computational multi-reaction extension is also
authorized; Ugi remains the deep case and no new biological or wet-lab scope is implied.

- Plan: [`docs/PLAN.md`](docs/PLAN.md)
- Documentation map: [`docs/README.md`](docs/README.md)
- Tasks: [`docs/M0_TASKS.md`](docs/M0_TASKS.md)
- Data: [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- Decisions: [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)
- Multi-reaction extension: [`docs/MULTIREACTION_COMPUTATIONAL_PLAN.md`](docs/MULTIREACTION_COMPUTATIONAL_PLAN.md)

## Quick start

```bash
uv venv && source .venv/bin/activate
uv sync --frozen --extra dev --extra torch
make vendor    # or: make vendor-partial   (skips the 96 MB R1 asset)
make verify
make check-core
forge doctor
forge experiment run installation-smoke --profile smoke

# Qualified model workflows
forge doctor phase1-training-smoke
forge experiment run phase1-training-smoke --profile smoke
forge experiment run phase1-sampling --profile smoke
forge experiment reproduce phase1-sampling --profile smoke
forge experiment run phase1-multireaction-corpus --profile full
forge experiment run phase1-multireaction-training-smoke --profile smoke
forge experiment run phase1-multireaction-overfit --profile smoke
forge experiment reproduce phase1-multireaction-overfit --profile smoke
forge experiment run phase1-shared-synthesis-program-representation --profile full
forge experiment reproduce phase1-shared-synthesis-program-representation --profile full
forge experiment run phase1-shared-synthesis-program-integration --profile smoke
forge experiment reproduce phase1-shared-synthesis-program-integration --profile smoke
forge experiment run phase1-shared-synthesis-program-production-design --profile full
forge experiment reproduce phase1-shared-synthesis-program-production-design --profile full
forge experiment run phase1-finite-component-catalogue-baseline --profile smoke
# After the final source snapshot is frozen, run the matched three-seed CPU baseline:
make phase1-finite-component-catalogue-full

# Paper and provenance
forge paper verify
forge paper reproduce          # exact artifact replay + two clean packaging builds
forge paper doctor --strict   # reports every blocker to a full numerical rerun
forge paper experiments       # audits every v1 experiment/baseline manuscript row
forge paper build
forge provenance verify --expect-verified 822
make code-survey
make test-baseline-report       # summarize the last clean-cache full-suite run
```

`make vendor` copies hash-pinned assets from absolute paths on the originating workstation. If those
paths do not resolve it fails with the missing list and expected hashes. **Do not substitute data.**

Active experiment specifications live beside their applications under `experiments/`. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for package boundaries and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for local and Modal execution.
The production training DAG is planned or launched explicitly with
`forge experiment ... phase1-training-production --profile full`; it is never triggered by the
smoke workflow.

The archived ICLR v0 source is `paper/v0/FORGE_ICLR2027_paper.tex`. Its exact source, twelve
manuscript evidence roots, generated tables, and included figures are frozen in
`configs/reproduction/iclr2027.json`. `forge paper verify` checks artifact replay. The stricter
doctor additionally walks the recursive path/hash graph and reports unavailable upstream corpus,
checkpoint, and external-engine bytes; it never calls artifact replay a full training reproduction.

## The three synthesis layers

The organizing insight. "Synthesizable" is not one property:

| | L1 — Final assembly | L2 — Subcomponent synthesis | L3 — Procurement |
|---|---|---|---|
| What | Ugi-3CR joining amine + aldehyde + isocyanide | Making heads, tails, linkers, **esters**, isocyanides | Buying terminal leaves |
| Status | **Mechanically qualified for 1,200 nominal rows; 1,100 exact single-compound records after source reconciliation** | **The open problem** | **Dynamic**, expires |
| Uncertainty | ≈ constant across candidates | **Dominates candidate variance** | Vendor-driven |

Preserving a Ugi core is insufficient: if the final coupling works but an ester-bearing tail precursor
cannot be made, the lipid is not executable. The ester comes from the *aldehyde* component — so ester
construction is L2, not L1.

## Open-endedness tiers

| Tier | Meaning |
|---|---|
| E0 | Exact product in the frozen enumeration |
| E1 | Known assembly, all components already accepted terminal blocks |
| **E2** | **Known assembly, ≥1 component needs a generated L2 route — the primary claim** |
| E3 | New assembly family — exploratory, the paper does not depend on it |

## Relationship to other projects

FORGE **consumes** the `compose_rgm` corpus and reaction registry as hash-pinned vendored data. It does
not fork or modify them. That boundary is deliberate: FORGE's protagonist is complete route execution,
while COMPOSE-Lipid's is the generative distribution and biological targeting. FORGE does not reuse
pulmonary delivery as its endpoint.

LUCID (ICLR 2026 GEM workshop) is the historical baseline: it sampled one of 527 stored topologies and
diffused only atom/bond identities, and its synthesizability check was a hardcoded component lookup.
FORGE generates topology and produces auditable routes.
