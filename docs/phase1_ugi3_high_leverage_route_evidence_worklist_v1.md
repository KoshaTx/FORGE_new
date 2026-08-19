# Development-only high-leverage Ugi route-evidence worklist

## Outcome

The frozen missing-knowledge ledger contains 802 one-gap products. The 15
highest-count missing components account for 199 products, below the 201-product
25% threshold. Adding the sixteenth component raises potential coverage to
207/802 (25.81%). Because every product in this ledger has exactly one
noncomplete component, these component counts are disjoint potential unlocks;
they are not predicted synthesis successes.

The 16-component prefix contains 13 components assigned to three existing
projected upstream programs and three amine heads in the procurement-or-exact-
route lane. No program is invented for those heads. If “existing-program only”
is interpreted literally to exclude the three heads, 18 projected-program
components are required to cross 25% (204/802); the first 17 reach only 198.
The present worklist preserves the frozen audit's cardinality-minimal
16-component result and reports the heads separately.

## Grouped leverage

| Role | Existing upstream program | Components | Potential unlocks |
|---|---|---:|---:|
| Isocyanide tail | Primary amine formylation, then formamide dehydration | 4 | 86 |
| Aldehyde body/tail | Fatty-acid/diol esterification, then alcohol oxidation | 5 | 46 |
| Aldehyde body/tail | Primary alcohol oxidation | 4 | 39 |
| Amine head | None established; procurement or exact upstream route lane | 3 | 36 |

The program labels above are mechanical projections retained from
`route_gap_triage_v5`. They are not exact routes for the listed substrates.

## Exact worklist

| Rank | Role | Exact component | Occurrences / potential unlocks | Required projected leaves | Current evidence and precise gap | Learned proposal search |
|---:|---|---|---:|---|---|---|
| 1 | Isocyanide | `[C-]#[N+]CCCCCCCCCCCCC` | 40 | `CCCCCCCCCCCCCN` | Leaf current (1/1); exact-substrate execution or bounded-scope qualification missing | No; evidence qualification is the bottleneck |
| 2 | Isocyanide | `[C-]#[N+]CCCCCCCCCCCCCCC` | 26 | `CCCCCCCCCCCCCCCN` | Leaf current (1/1); exact-substrate execution or bounded-scope qualification missing | No; evidence qualification is the bottleneck |
| 3 | Amine head | `CN(C)N` | 18 | None projected | Current exact procurement/stocked-terminal evidence missing; if not procured, exact upstream route missing | Conditional proposal-only search after procurement and exact-route search |
| 4 | Aldehyde | `C#CCCCCCCCCCCCCCCCCCC=O` | 13 | `C#CCCCCCCCCCCCCCCCCCCO` | Leaf unresolved (0/1) and exact oxidation scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |
| 5 | Isocyanide | `[C-]#[N+]CCC(C)CCC` | 12 | `CCCC(C)CCN` | Leaf unresolved (0/1) and exact formylation/dehydration scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |
| 6 | Aldehyde | `CCCCCCCCCCC(=O)OCCCC=O` | 11 | `CCCCCCCCCCC(=O)O`; `OCCCCO` | Leaves current (2/2); exact esterification/oxidation scope missing | No; evidence qualification is the bottleneck |
| 7 | Amine head | `CC1CCCCC1N` | 10 | None projected | Current exact procurement/stocked-terminal evidence missing; if not procured, exact upstream route missing | Conditional proposal-only search after procurement and exact-route search |
| 8 | Aldehyde | `C=CCCCCCCCCC=O` | 10 | `C=CCCCCCCCCCO` | Leaf unresolved (0/1; historical use 1/1) and exact oxidation scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |
| 9 | Aldehyde | `CCCCCCCCCCC(=O)OCCCCCCCC=O` | 10 | `CCCCCCCCCCC(=O)O`; `OCCCCCCCCO` | Leaves current (2/2); exact esterification/oxidation scope missing | No; evidence qualification is the bottleneck |
| 10 | Aldehyde | `CCCCCCCCCCC(=O)OCCC=O` | 9 | `CCCCCCCCCCC(=O)O`; `OCCCO` | Leaves current (2/2); exact esterification/oxidation scope missing | No; evidence qualification is the bottleneck |
| 11 | Amine head | `NN1CCNCC1` | 8 | None projected | Current exact procurement/stocked-terminal evidence missing; if not procured, exact upstream route missing | Conditional proposal-only search after procurement and exact-route search |
| 12 | Isocyanide | `[C-]#[N+]C(CC)CC(C)CC` | 8 | `CCC(C)CC(N)CC` | Leaf unresolved (0/1) and exact formylation/dehydration scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |
| 13 | Aldehyde | `C#CCCCCCCC(=O)OCCCCCCCC=O` | 8 | `C#CCCCCCCC(=O)O`; `OCCCCCCCCO` | One of two leaves unresolved and exact esterification/oxidation scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |
| 14 | Aldehyde | `C#CCCCCCCCCC(=O)OCCC=O` | 8 | `C#CCCCCCCCCC(=O)O`; `OCCCO` | Leaves current (2/2); exact esterification/oxidation scope missing | No; evidence qualification is the bottleneck |
| 15 | Aldehyde | `C=CCCCCCCCC=O` | 8 | `C=CCCCCCCCCO` | Leaf unresolved (0/1; historical use 1/1) and exact oxidation scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |
| 16 | Aldehyde | `CCCCCCCCCCC(=O)CCCC=O` | 8 | `CCCCCCCCCCC(=O)CCCCO` | Leaf unresolved (0/1) and exact oxidation scope missing | Conditional proposal-only search for unresolved leaf; cannot supply evidence |

## Execution order

1. Retrieve or qualify exact-substrate evidence for ranks 1 and 2; their
   required amine leaves are already current and together represent 66
   potential unlocks.
2. Check exact current procurement for the three heads before any route search.
3. For ranks 6, 9, 10 and 14, focus on exact-substrate execution or bounded
   substrate-scope qualification; all projected leaves are already current.
4. Resolve the named leaf evidence for the remaining projected-program records,
   then seek exact-substrate evidence for the parent step.
5. Use learned retrosynthesis only as proposal-only bounded search where marked.
   A proposal cannot supply evidence, close a route, or admit a reaction family.

## Guardrails

This worklist did not access the sealed saturation holdout. It does not assert
that any projected route works, that any material is purchasable or unavailable,
or that resolving a record will produce an experimental success. It does not
change synthesis guidance or prospective candidate selection.

The frozen machine-readable source of truth is
`results/phase1/ugi3_high_leverage_route_evidence_worklist_v1/manifest.json`.
