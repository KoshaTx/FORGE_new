# Primary L2/L3 success criterion for the 22-family study

The user authorized this change on 2026-09-25 after reviewing the distinction
between the older repository's computational makeability result and strict
exact-source/current-item dossier closure. This document and
`configs/route/compose_lipid_computational_makeability_v1.json` define the new
primary outcome. The manuscript remains unchanged.

A request succeeds when its product passes exact L1 and **every required
component is directly vendor-listed or has at least one complete computational
route whose terminal starting materials are all vendor-listed**.

The primary metric is named **computational makeability with vendor-listed
terminals**. It is a computational coverage metric, not an experimental synthesis
success probability or confirmation of stock, purity, price, lead time, or
procedure applicability.

## Accepted component paths

- **Direct listing:** a dated, identity-bound positive supplier/vendor-directory
  observation for the exact terminal constitution. A SKU, assay or explicit stock
  observation is not required for this primary outcome.
- **Planner route:** a complete target-bound multistep path with a resolved leaf
  set, with every leaf positively vendor-listed. A planner's public-catalog
  membership alone does not establish vendor listing. Independent forward replay
  and substrate-scope evidence are separate reporting fields, not hidden primary
  admission requirements.
- **Forward-applied construction:** a recorded chain of published/registry-defined
  transformations reconstructs the exact target, and all terminal inputs are
  vendor-listed. The target can be one of multiple forward products, as in the
  legacy implementation. Uniqueness, selectivity and experimental substrate scope
  are reported separately.
- **Previously strict-qualified evidence:** actual current direct-terminal or
  complete route records can also satisfy these requirements. A bare old success
  boolean, an expired snapshot or membership in a training corpus cannot.

The logic is OR across complete route alternatives and AND across the leaves of
one route, then AND across every required component of the lipid. Leaves from
different incomplete alternatives cannot be combined into a fictitious path.
Missing queries, supplier errors and bounded search failures remain visible.

## Correspondence to the older repository

The reference is `git@github.com:KoshaTx/forge.git` at
`f58c679594c49312cde666de0fe293bf6ba5c54e`, particularly
`scripts/phase1_run_ugi_indomain_makeability_v1.py`. Its successful product states
are `buy_all` and `buy_and_make`; `route_only` and `blocked` do not enter the
success numerator. Its planner path does not require independent forward replay;
the deliberate Tail A/B constructors do reproduce the target forward.

This contract adopts that scientific criterion without copying implementation
defects. It enforces the previously declared 30-day vendor-snapshot validity,
binds leaves to each actual route, considers alternate paths, retains lookup
uncertainty, and keeps every evaluation request. These corrections are explicit
version differences. The old code's union of leaves from several saved routes,
reuse of stale cache records, and silent request deduplication are not adopted.

## Reporting and success

The primary all-request yield includes every request, including nonexact L1
products, unresolved routes, missing observations and execution failures in the
denominator. Report each of the 22 families separately. On the current development
cohort, denominators remain 64 per family and 1,408 overall; component and leaf
counts are additional summaries, not replacements for the product denominator.
Conditional-on-exact-L1 and unique-constitution statistics may be reported only
with their distinct denominators.

Report computational L2 path coverage and L3 vendor-listing coverage separately,
alongside their intersection. Break down successful paths by direct listing,
planner hypothesis and forward-applied construction; also show exact-source
support, independent replay, uniqueness and applicability when assessed.

The secondary **strict dossier closure** retains its original exact-source,
exact-substrate, unique-forward and current item-level procurement requirements.
Its existing 27/1,408 result remains a strict development result. It is neither
overwritten nor automatically presented as a full evaluation of the new metric.
The new full-cohort makeability result is pending assessment, not zero.

This authorization changes which routing outcome determines primary success.
It does not supply a new measured result or a new numerical pass threshold.
The existing goal of at least 90% exact L1 per family, molecular quality and
diversity requirements, matched controls, independent seeds, and held-out
confirmation remain unchanged. Missing strict dossiers no longer determine
failure of the primary computational makeability metric. Production synthesis
guidance retains its separately authorized readiness gates.

## Execution order

1. Freeze the selected cohort and component identities; query exact direct
   vendor listings once per unique identity and retain dated source receipts.
2. Reuse recorded computational paths and construct appropriate published-family
   proposals. Search unresolved identities with matched, recorded budgets.
3. Resolve each complete path's terminal identities and vendor listings; retain
   all alternatives and unsuccessful attempts.
4. Evaluate the versioned primary criterion on every request and report it next
   to the unchanged strict metric. Confirm on independently frozen evaluations
   before admitting final manuscript results.

The earlier engine audit and frozen discovery pilot remain under
`results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1/planner/remote_engine_reuse_v1/`.
