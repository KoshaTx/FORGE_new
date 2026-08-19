# Phase 1 independent component-route diagnostic

Status: **complete diagnostic; not admitted as route evidence or synthesis value**  
Date: 2026-08-04

## Question

The conservative-high candidate audit left 45 generated products unresolved under
the current evidence-bearing route evaluator. Thirty-nine failed only at the
aldehyde-derived oxoester component and six failed only at one long-chain
isocyanide. The unresolved aldehydes were usually close homologues of accepted
route-ready components. This diagnostic asks whether an independent public
retrosynthesis policy can propose useful disconnections for those components and
whether bounded public-stock search distinguishes them from matched accepted
controls.

## Frozen engine and authority boundary

The diagnostic uses the public AiZynthFinder 4.4.1 USPTO and ring-breaking policies,
the public USPTO filter, and the public ZINC stock. Every local model, template,
stock and configuration file is hash-pinned in
`configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json`.

The engine may generate route hypotheses only. It cannot create an
`EvidenceRecord`, certify route completion, populate synthesis-success probability,
or contribute to synthesis value. A solved public-stock search remains a planner
result until the exact step is independently forward verified and its substrate
scope, conditions, operational compatibility and terminal availability are
adjudicated.

## Matched target set

The source was the frozen conservative-high molecule audit. It contributed:

- 25 unique unresolved aldehyde-derived oxoester components;
- one unresolved long-chain isocyanide;
- 13 unique nearest route-ready aldehyde controls; and
- one nearest route-ready isocyanide control.

All 40 components received the same top-10 single-step expansion policy. A
deterministic size-spanning panel of eight unresolved components per role and all
their unique paired controls entered an eight-second bounded full search. The final
full-search panel contained nine unresolved targets and seven matched controls.

## Result

The single-step policy returned ten hypotheses for every target:

| Cohort | Role | Targets with hypotheses / targets |
| --- | --- | ---: |
| Unresolved | aldehyde-derived oxoester | 25 / 25 |
| Unresolved | isocyanide | 1 / 1 |
| Matched route-ready control | aldehyde-derived oxoester | 13 / 13 |
| Matched route-ready control | isocyanide | 1 / 1 |

The bounded public-stock planner solved one of nine unresolved targets and none of
seven matched controls. The solved unresolved target,
`C=CCCCCCCCCC(=O)OCCCC=O`, was disconnected in one step to public-stock
`O=CCCCBr` and `C=CCCCCCCCCC(=O)O` under the public USPTO policy.

The key finding is not a 1/16 route-closure rate. It is the mismatch between the
current evaluator and an independent planner:

1. the learned proposal lane has no obvious first-step recall failure on this
   matched component set;
2. the public-stock search fails current route-ready controls as well as unresolved
   generated components; and
3. at least one component classified as unresolved by the current evaluator is
   solvable under the independent bounded planner.

Therefore, current unresolved labels primarily measure the coverage of the frozen
evidence and stock ledgers. They cannot be interpreted as molecular
unsynthesizability.

## Decision

Keep post-generation L1/L2/L3 route assessment as the production and prospective
panel-lock procedure. Add the learned proposal engine as a quarantined route-search
lane, then independently verify and adjudicate its proposed steps. Do not promote a
proposal merely because the public planner reaches its stock.

The prior negative synthesis-tilt result remains evaluator-limited. Synthesis
tilting may be rerun only after the proposal-augmented evaluator passes a frozen
route-recovery and matched-control benchmark. It must then beat identical post-hoc
routing on unique, diverse, route-closed candidates per generator, planner and
wall-clock budget. Until then, post-hoc routing is the production method and
synthesis tilting is an unpromoted challenger.

## Reproduction

```bash
PYTHONPATH=src MPLCONFIGDIR=.matplotlib-cache \
  .venv-aizynthfinder-py311-arm64/bin/python \
  scripts/phase1_run_aizynthfinder_component_route_diagnostic_v1.py

PYTHONPATH=src .venv/bin/python \
  scripts/phase1_render_aizynthfinder_component_route_diagnostic_v1.py
```

Primary artifacts:

- `results/phase1/aizynthfinder_component_route_diagnostic_v1/result.json`
- `results/phase1/aizynthfinder_component_route_diagnostic_v1/route_hypotheses.jsonl.gz`
- `output/pdf/FORGE_AiZynthFinder_route_hypothesis_audit.pdf`
