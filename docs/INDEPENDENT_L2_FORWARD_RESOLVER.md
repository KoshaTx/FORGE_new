# Independent upstream L2 forward resolver

Status: **implemented and frozen against development exact-scope records; not
benchmarked and not activated**  
Evidence checked: 2026-08-03

## Purpose

Graph2Edits supplies proposal-only reactant hypotheses. It cannot select or
define the reaction used to verify its own output. The independent resolver in
`forge.route.l2_forward_resolver` tests proposals only against hash-pinned,
upstream **L2** transforms whose exact reactant-role assignment and exact target
have already been admitted.

The Ugi final-assembly transform is never an L2 verifier. Neither a model score
nor a model reaction-class label is read during transform resolution.

## Frozen development scope

`configs/route/graph2edits_l2_forward_resolver_v1.json` is generated
deterministically from the development route ledgers. It currently binds:

- 4 exact-source upstream transforms with 49 exact-pair admissions after three
  targeted aldehyde scope additions;
- 4 exact C16 transforms with one admitted pair each; and
- 3 exact C18 transforms with one admitted pair each.

The total is 11 transforms and 56 exact role-ordered reactant/target
admissions. Duplicate evidence for the same transform, assignment and target is
merged into one verifier assignment; its source-record hashes are retained.
No family or analogue projection is admitted.

The normalized manifest is rebuilt with:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/build_graph2edits_l2_forward_resolver_config.py
```

Manifest SHA-256:
`3cb6012f4bbd73d198159ffcf1f765f45fc0cf526b2ed93e0a836fc7d01cdf04`.

## Resolution algorithm

For one typed `SingleStepRetrosynthesisProposal`, the resolver:

1. canonicalizes the target constitution;
2. considers every admitted upstream L2 transform with matching reactant
   arity;
3. enumerates every unique permutation of the proposed reactant multiset over
   that transform's roles;
4. retains only assignments with an exact frozen pair-and-target admission;
5. fails before partial execution when eligible assignments exceed the frozen
   forward-call budget;
6. executes every eligible assignment through the shared qualified-forward
   implementation; and
7. accepts only when exactly one assignment produces exactly one unique product
   and that product is the exact constitutional target.

The stable outcomes are:

| Outcome | Meaning |
| --- | --- |
| `exact_unique` | Exactly one admitted assignment uniquely reconstructed the target. |
| `censor_no_verifier` | No exact-scope L2 transform/assignment was admitted. |
| `reject_forward_mismatch` | Admitted assignments ran, but none reconstructed the target. |
| `censor_ambiguous_forward_products` | At least one admitted assignment produced multiple unique products. |
| `censor_ambiguous_transform_assignment` | Multiple admitted transform/role assignments uniquely reconstructed the target. |
| `censor_budget_exhausted` | The frozen assignment/product budget was insufficient. |
| `censor_execution_error` | A qualified-forward execution failed and was not interpreted chemically. |

The forward-call cap is 50 per proposal and the per-assignment product cap is
64. The assignment count is evaluated before execution so an ordering-dependent
partial result cannot pass.

## Scientific authority boundary

An exact forward reconstruction establishes graph consistency only. The
resolver output always records:

- `substrate_scope_state="unassessed"`;
- `evidence_tier=null`;
- `success_probability=null`;
- `route_closure_authorized=false`; and
- `may_enter_synthesis_value=false`.

Source-record hashes authenticate why an exact assignment was eligible to be
tested; they are not promoted into an evidence tier by the resolver. Evidence,
operational compatibility and L3 terminal closure remain separate systems.

## Verification

Focused tests cover:

- independent role-permutation enumeration;
- invariance to adversarial changes in model score and model class;
- no-verifier censoring without execution;
- mismatch rejection;
- product and transform-assignment ambiguity;
- pre-execution budget exhaustion;
- execution-error censoring;
- rejection of general-enumeration/Ugi-L1 status; and
- deterministic regeneration and one known development exact-pair replay.

Run:

```bash
.venv/bin/pytest -q tests/test_l2_forward_resolver.py
```

The known-pair replay uses an existing development exact-source route. It is
not the frozen 120-target benchmark, does not call Graph2Edits and supports no
performance claim. The resolver manifest retains
`activation_authorized=false`, `sealed_holdouts_accessed=false` and
`frozen_benchmark_executed=false`.
