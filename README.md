# FORGE

**Route-Grounded Generative Design of Synthetically Executable Ionizable Lipids**

Computational engine for a Nature Biotechnology paper. The claim: an open-ended molecular generator
coupled to complete synthesis-route generation produces lipids that arrive with **executable synthesis
dossiers** and show higher **prospective** synthesis success than fixed-topology generation or post-hoc
retrosynthesis filtering.

The unit of generation is a **product–route dossier**, not a SMILES string.

> **Working in this repo? Read [`AGENTS.md`](AGENTS.md) first.** It defines the authorized scope and
> nine hard constraints that an agent optimizing for green tests will otherwise violate.

## Status

**Milestone M0 only** — decisions, audits, feasibility gates, data inventories. The scientific plan is
approved in principle but **not implementation-frozen**. No GPU training. No Phases 1–8.

- Plan: [`docs/PLAN.md`](docs/PLAN.md)
- Tasks: [`docs/M0_TASKS.md`](docs/M0_TASKS.md)
- Data: [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- Decisions: [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
make vendor    # or: make vendor-partial   (skips the 96 MB R1 asset)
make verify
make test
```

`make vendor` copies hash-pinned assets from absolute paths on the originating workstation. If those
paths do not resolve it fails with the missing list and expected hashes. **Do not substitute data.**

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
