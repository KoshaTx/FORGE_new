# Additive proposal-lane v3 contract

Status: **implemented and executable for the frozen development benchmark; not
executed and not production-activated**  
Frozen contract:
`configs/route/single_step_proposal_lane_qualification_benchmark_v3.json`

## Purpose

Version 3 removes redundant process booleans from scientific readiness without
altering the frozen v1 policy, v2 target binding or exact L2 resolver. Readiness
is derived from authenticated artifacts:

- the frozen v1 runner binding and its frozen 120-target/hidden-truth receipts;
- the existing isolated runtime lock;
- the existing runtime qualification containing two byte-identical normalized
  semantic smoke outputs; and
- the current exact-scope resolver digest.

Institutional license compliance remains an owner responsibility. It is not a
model-performance result and is not represented by a manually toggled
scientific-readiness Boolean. The v3 contract contains no `all_gates_passed`,
ten-repeat device gate or separate benchmark-execution authorization flag.

Loading the contract does not read target contents or hidden truth. Running the
120-target benchmark, changing the production planner and activating a learned
backend remain separate, unperformed actions.

## Source-neutral discovery states

`forge.route.proposal_discovery_status` wraps the unchanged exact resolver and
uses its registry-owned transforms. It defines five proposal-only states:

1. `exact_known_route` — the existing exact-scope resolver uniquely verifies a
   known admitted route;
2. `known_family_forward_consistent_projection` — one registry-owned family
   transform uniquely reconstructs the target outside its exact evidence
   admission;
3. `new_family_hypothesis_retained` — no known-family transform reconstructs the
   hypothesis, which remains available for bounded evidence discovery;
4. `ambiguous` — the budget, product set, execution or transform assignment
   cannot support a unique structural decision; and
5. `rejected` — an independently exact-admitted assignment fails forward
   reconstruction.

The model score and model reaction-class label never select a transform.
Known-family projection establishes graph consistency only. Every discovery
receipt has null evidence and success-probability fields, forbids route closure
and forbids direct entry into `V_syn`.

Local novelty is never itself chemical incompatibility. A new-family hypothesis
that lacks independent scope, evidence or L3 support remains an unqualified
missing-evidence hypothesis. `adjudicate_source_neutral_route` applies the same
forward, scope, evidence, operational and terminal-closure contract to every
proposal source. Once those checks pass, the caller-supplied route state and
route value are preserved identically whether the hypothesis originated from
Graph2Edits, deterministic templates or retrieval.

## Complete v3 scoring

`forge.route.single_step_proposal_benchmark_v3` adds a v3 output ledger and
scorer. It retains and reports:

- overall and primary-stratum top-1/5/10/20 exact known-route recovery;
- exact-known, known-family projected, new-family retained, ambiguous and
  rejected counts;
- graph-consistent and operational-screen yields with explicit denominators;
- strict, learned and overlap provenance counts and cross-lane constitutional
  overlap;
- invalid and canonical-duplicate fractions;
- every budget-ceiling or resource-exhaustion state;
- p50, p95 and maximum latency and observed host/accelerator memory;
- accepted proposals and retained new-family hypotheses; and
- per-target paired hybrid-minus-strict and hybrid-minus-learned differences.

The frozen adjudicator separates scientific hard requirements from operational
preference checks. It can recommend a later versioned promotion review but
always writes `production_activation_authorized=false`. A scientific failure
cannot be retried or retuned. A technical invalidation before hidden-truth
scoring may be rerun only with the failed ledger retained.

## Verification

Focused verification is:

```bash
.venv/bin/pytest -q tests/test_single_step_proposal_benchmark_v3.py
```

The tests use synthetic targets only. They do not execute Graph2Edits on the
frozen 120-target manifest and do not access a sealed holdout.
