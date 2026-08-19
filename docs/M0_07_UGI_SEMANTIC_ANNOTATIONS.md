# M0-07 Ugi semantic whole-graph annotations

## Decision

The 1,100 reconciled exact-source AGILE products admit deterministic,
chemistry-native semantic annotation under the frozen
`ugi_3cr_agile` transform.

These annotations supervise one connected whole-product graph. They do not
authorize component catalog IDs in model state, independent fragment
generation, or separately generated graph distances. Region labels are node
attributes, and all distances are derived from the current whole graph.

## Frozen scope

The artifact is limited to products admitted by both the AGILE label
reconciliation and exact source-evidence gate. It pins the curated product
table, source-evidence result and ledger, qualified reaction registry, Ugi
variant, and measured-product assembly qualification by SHA-256.

Each product carries four deterministic ledgers:

1. Product identity, exact evidence, outcome multiplicity, Ugi core, role
   anchors, and row-level annotation digest.
2. Atom origin, core position, reaction map number, source symmetry class,
   and graph-derived distances.
3. Bond topology, core membership, and core-boundary role.
4. Component-to-product source mappings, including deleted atoms and all
   symmetry-related target mappings.

Ordinary atom-map numbers do not survive product serialization. The pipeline
therefore captures RDKit reaction-origin properties before canonical
serialization and writes map semantics into explicit ledger fields.

## Frozen result

- Products: 1,100
- Product atoms: 52,810
- Product bonds: 52,370
- Component atom rows: 52,810
- Ugi core atoms: 5,500, exactly five per product
- Core-boundary bonds: 3,300, exactly three per product
- Assembly-introduced carbonyl oxygens: 1,100
- Deleted aldehyde oxygens: 1,100
- Connected origin regions: 3,300
- Component atom-order stress mismatches: 0

All target-matching outcomes have one semantic symmetry-class signature.
Source-index multiplicity is retained for symmetric amine sites instead of
being coerced to one arbitrary source atom. A5 products retain three source
mappings and A17 products retain two. A19 and A20 each produce two raw
constitutional outcomes, while exact target matching retains the reported
product.

The role anchors are invariant across all products:

- amine head at reaction map 1;
- aldehyde-derived body and tail at reaction map 2;
- isocyanide-derived tail at reaction map 4.

## Model boundary

This artifact supports joint whole-product and L1 assembly learning. It does
not convert FORGE into a component enumerator. The model still generates one
complete atom-and-bond graph. Origin regions, Ugi core positions, and
component mappings are auxiliary chemistry supervision.

L2 and L3 route closure remain recursive and modular. This artifact does not
claim a complete upstream synthesis route, biological potency, apparent pKa,
particle size, formulation behavior, or prospective synthesis success.

## Reproduction

Run:

```bash
PYTHONPATH=src python3 scripts/m0_07_ugi_semantic_annotations.py
```

The command hash-verifies every input, rebuilds all four gzip ledgers with a
fixed timestamp, performs a reversed-component-atom-order stress test for all
1,100 products, and writes
`results/m0_07/ugi_semantic_annotations_result.json`.
