# FORGE manuscript evidence ledger

This ledger applies to the Markdown source and its LaTeX, PDF and Word exports.
It records the support class for each material claim in the working manuscript.
The manuscript may use stronger prospective language only after the
corresponding experimental dataset is frozen.

## External and journal claims

| Claim | Class | Evidence | Draft treatment |
|---|---|---|---|
| Nature Biotechnology Articles allow a 150-word abstract, 3,000 words of main text, up to six display items, an unheaded introduction, Results, Discussion, and Online Methods | Reported | Nature Biotechnology content-type guidance, accessed 30 July 2026 | Used only to structure the draft |
| Initial submission requires no special template and accepts Word, PDF, or TeX/LaTeX | Reported | Nature Biotechnology initial-formatting guidance, accessed 30 July 2026 | The repository supplies a reviewer-friendly Word shell, not an official journal template |
| Ionizable lipids are a major determinant of LNP RNA delivery | Reported | Hou et al., Nat. Rev. Mater. 2021; Akinc et al., Nat. Biotechnol. 2008 | Stated with citations 1 and 2 |
| Recent AI and high-throughput platforms have prioritized ionizable lipids with in vivo delivery activity | Reported | Witten et al., Nat. Biotechnol. 2025; Li et al., Nat. Mater. 2024; Xu et al., Nat. Commun. 2024; Zhou et al., Nat. Biotechnol. 2026; Wang et al., Nat. Commun. 2024 | Stated with citations 3 through 7 |
| Those systems generally search a candidate universe defined by a library or chemistry before prediction | Inferred | Primary methods and abstract-level design of citations 3 through 7 | Written as "in most cases" and not as a criticism |
| AGILE supplies aligned Ugi chemistry and HeLa and RAW 264.7 measurements | Reported | Xu et al., Nat. Commun. 2024 and official source data | Used to define in vitro oracle tasks only |

## Repository-computed claims

| Claim | Class | Evidence artifact | Draft treatment |
|---|---|---|---|
| The reconciled AGILE set contains 1,100 exact single-compound records and excludes 100 mixture-derived measurements from single-graph supervision | Computed | `results/m0_07/agile_label_reconciliation.json`; input hashes and output hashes recorded; status `completed_blocking_source_reconciliation` | Reported with both denominators and exclusion |
| All 12,276 virtual structures have one exact qualified decomposition under the frozen Ugi transform | Computed | `results/m0_09/agile_virtual_ugi3_capability.json`; input hashes recorded | Described as exact under the reviewed transform, not as experimental synthesis evidence |
| The L2 inventory contains 45 route instances, 76 reaction instances, 11 route families and 24 components with exact source programs | Computed | `results/m0_09/l2_supervision_decision.json`; input hashes recorded | Used to justify reusable hybrid L2 chemistry |
| The reviewed L2 set contains zero reported negative outcomes, zero upstream head routes and zero complete product routes across every L2 and L3 branch | Computed negative result | `results/m0_09/l2_supervision_decision.json` | Reported with the same prominence as positive counts |
| Current supervision supports joint product and L1 generation with hybrid recursive L2 closure, not a monolithic complete-route decoder | Inferred decision | `results/m0_09/l2_supervision_decision.json` and `docs/M0_09_L2_SUPERVISION_DECISION.md` | Explained as a data-aligned choice and a protection against narrowing molecular support |
| The leakage-safe graph-pretraining corpus contains 14,129 unique constitutional graphs and has zero constitutional or standard-InChI connectivity overlap with the 1,100 AGILE oracle graphs | Computed | `results/m0_07/oracle_graph_corpus_result.json` | Reported as an encoder-pretraining boundary, not biological supervision |
| Label-free graph pretraining used masked atomic number, formal charge and bond-type recovery without biological labels, component annotations, provenance fields, virtual candidates or graph truncation | Computed implementation | `configs/bio/m0_07_oracle_graph_pretraining.json`; `results/m0_07/oracle_graph_pretraining_full_result.json` | Reported in Online Methods |
| The initial oracle selection rule was committed before aggregate graph results, then amended after a read-only code audit and before any aggregate or partial graph performance was inspected | Preregistered and amended decision | `configs/bio/m0_07_oracle_freeze.json`; `docs/DECISION_LOG.md`; commits `95191b3` and the later audit-hardening commit | Calibration metrics select the architecture; outer tests evaluate it and support predeclared domain gates |
| The production refit uses all 1,100 reconciled structures only after architecture selection, derives candidate domains from exact component identities and Ugi reconstruction, and applies conservative empirical held-domain radii rather than maximizing the raw oracle mean | Preregistered implementation | `configs/bio/m0_07_oracle_production.json`; `src/forge/bio/oracle_production.py` | Do not call the full-data radius a finite-sample conformal guarantee; do not describe the refit as completed until its checkpoint exists |
| HISTORICAL / DIAGNOSTIC ONLY -- superseded as generator production evidence. An earlier development pool produced 3,975 valid exact-L1 products in 4,096 attempts, including 3,152 products with at least one component absent from the 424-component evaluation catalog | Computed, superseded | `results/phase1/ugi3_fresh_pool_route_coverage_v6/result.json` and the frozen v3 generator artifacts listed in `COMPUTATIONAL_RESULTS_EVIDENCE_MATRIX.json` | Do not cite as generator production evidence; the final denominator is the two-arm 32,768-draw run. Source file `src/forge/value/synthesis.py` has since drifted from the hash pinned by this artifact's config, so the artifact is not currently reproducible from the working tree |
| The final two-arm production run admitted 30,180 of 32,768 attempts (92.1%): broad-prior 15,055/16,384 (91.9%, 14,226 unique) and support-enriched 15,125/16,384 (92.3%, 13,248 unique); 22,843 admitted products (75.7%) were absent from the all-fold refit corpus | Computed | `results/phase1/ugi_constrained_stochastic_production_candidates_v2/result.json` | Do not pool the separate branch-conditioned 4,096-draw exploration tranche into this matched denominator |
| Of the 2,590 admitted products inside the oracle applicability region, built from 201 distinct components, 2,515 (97.1%) resolve to commercially listed starting materials by forward-verified published routes; 75 carry a verified route with unlisted terminal materials and none is route-blocked | Computed | `results/phase1/ugi_indomain_makeability_v1/result.json`; `results/phase1/ugi_tail_a_disconnection_v1/result.json`; `results/phase1/ugi_isocyanide_route_v1/result.json`; `results/phase1/ugi_online_procurement_snapshot_v1/result.json` | Forward verification does not establish substrate scope; a vendor listing is a dated screening signal, not a quotation; refresh procurement before purchase |
| Morphology-aware allocation increased fresh supported-terminal yield from 3.87% to 8.64%, a 2.23-fold enrichment, while retaining non-zero probability for all 57,190 qualified programs | Computed | `results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json` | Describe as applicability enrichment, not guaranteed in-domain generation |
| The nested morphology-potency challenger produced 23 unique supported conservative-high-potency products versus 27 under applicability-only allocation at matched budgets | Computed negative result | `results/phase1/ugi_morphology_potency_matched_v1/result.json` | Potency remains terminal conservative ranking; no trajectory-level potency claim |
| Proposal-augmented routing identified 23 route-ready broad-arm products and 41 route-ready applicability-arm products in the frozen 124-product panel | Computed | `results/phase1/ugi_proposal_augmented_route_readiness_v1/result.json` | Route readiness combines exact closure and independently qualified family evidence; it is not synthesis-success probability |
| Family-readiness trajectory guidance produced 2 route-ready products among 56 unique terminals versus 3 after ordinary generation and complete-product assessment | Computed negative result | `results/phase1/ugi_synthesis_guidance_failure_audit_v1/result.json` | Do not promote synthesis trajectory guidance; retain complete-product routing before panel lock |
| Graph2Edits recovered 29/36 hidden exact routes at top 5 versus 19/36 for AiZynthFinder, with no additional top-5 union recovery | Computed | `results/phase1/hybrid_single_step_recovery_audit_v1/result.json` | Use Graph2Edits as the primary learned proposer; do not imply that model recovery is route evidence |
| Residual AiZynthFinder search produced public-stock solutions for 44/64 Graph2Edits-missed components and 29/97 unresolved leaves | Computed proposal diagnostic | `results/phase1/aizynthfinder_checkpoint_residual_v1/result.json` | Use AiZynthFinder only as a residual multistep challenger; independently adjudicate every proposed path |
| Four exact current supplier records raised route-ready productive finals from 3 to 6, but the proposal-aware contrast still failed the frozen synthesis-guidance gate | Computed negative/near-threshold result | `results/phase1/ugi3_proposal_aware_checkpoint_contrast_v2/result.json`; `configs/route/phase1_ugi3_hybrid_high_impact_leaf_terminals_v1.json` | Do not promote or retune synthesis guidance; use the hybrid route cascade on complete products before prospective panel lock |

## Proposed architecture and experiments

| Claim | Class | Evidence or decision | Draft treatment |
|---|---|---|---|
| FORGE generates a complete connected lipid graph and exact final assembly decomposition | Proposed implementation | `docs/PLAN.md`, `docs/DECISION_LOG.md`, M0-09 architecture decision | Written as platform architecture, not as a completed performance result |
| Recursive L2 routing combines deterministic transforms, retrieval, bounded search and learned ranking | Proposed implementation | M0-09 architecture decision | Written as intended implementation |
| Synthesis trajectory guidance was implemented and tested | Computed diagnostic | `results/phase1/ugi_production_synthesis_guidance_seam_v4_retry1/result.json`; `results/phase1/ugi_synthesis_guidance_failure_audit_v1/result.json` | Mechanism was functional but did not improve either strict route closure or proposal-augmented route readiness; it is not used in production |
| The production system uses morphology-aware applicability allocation, terminal conservative potency ranking and proposal-augmented L1/L2/L3 routing before panel lock | Frozen decision | `docs/DECISION_LOG.md`; `COMPUTATIONAL_RESULTS_EVIDENCE_MATRIX.json` | State each stage and its evidence boundary explicitly |
| Ugi 3-CR is the first prospective instantiation, not the platform boundary | Settled scope decision | `AGENTS.md`, `docs/PLAN.md`, `docs/DECISION_LOG.md` | Stated explicitly |
| The prospective campaign will compare coupled generation, post-hoc route assessment and finite-library selection | Proposed experiment | `docs/PLAN.md` | No outcome is claimed |
| Locked candidates will be followed through precursor preparation, final-lipid isolation, formulation and functional RNA delivery | Proposed experiment | `docs/PLAN.md` | Described as the planned evidence chain |

## Unresolved and placeholder claims

| Item | Current status | Manuscript rule |
|---|---|---|
| Winning HeLa oracle and held-component performance | In progress | Keep highlighted placeholder until the full graph and pretrained-transfer matrices are aggregated |
| Winning RAW 264.7 oracle and held-component performance | In progress | Report separately from HeLa and permit abstention |
| Frozen production-oracle checkpoint | In progress | Do not describe the refit as completed until the selected checkpoint hash and result artifact exist |
| Apparent pKa, particle size, PDI and encapsulation efficiency across the structural corpus | Unavailable | Do not impute or claim |
| Final biological endpoint and bridge assay | PI decision unresolved | Use "functional RNA delivery" and an explicit endpoint placeholder |
| Number of generated or locked candidates | Not available | Placeholder only |
| Prospective synthesis success, yield and purity | Not available | Placeholder only; report every attempt after data freeze |
| Formulation success | Not available | Placeholder only |
| In vivo function | Not available | Placeholder only |
| Advantage of route-guided sampling over post-hoc assessment | Computed null | Remove the efficiency placeholder; retain the molecule-route-dossier framework and pre-lock routing workflow |

## Delivery check

- No prospective outcome is presented as completed.
- No absent formulation property is imputed.
- AGILE is not described as an in vivo endpoint oracle.
- Virtual products are not described as observed syntheses.
- Algorithmic route closure is not described as experimental synthesis
  success.
- The manuscript contains no em dashes.

## Addendum, 6 August 2026: routing, availability and makeability

Later than everything above. Where a routing statement earlier in this ledger
conflicts with this addendum, this addendum governs.

| Claim | Class | Evidence | Draft treatment |
|---|---|---|---|
| FORGE decomposes every generated product exactly into its three Ugi components and reconstructs the product through the frozen transform | Computed | `results/phase1/ugi_bounded_hybrid_route_cascade_v1/result.json`, 256 of 256 | Safe to state directly; this is the mechanism that makes component routing tractable |
| FORGE constructs and independently forward-verifies a complete route to purchasable starting materials for 97.2% of applicability-supported candidates | Computed | `results/phase1/ugi_indomain_makeability_v1/result.json`, 2,517 of 2,590 | State as route construction with verified starting materials; describe the method as qualified transforms plus learned proposals plus independent forward verification plus a dated availability snapshot |
| Route construction uses a curated qualified transform library alongside learned proposal engines | Computed | `configs/route/graph2edits_l2_forward_resolver_v1.json`; `results/phase1/ugi_tail_a_disconnection_v1/`; `results/phase1/ugi_isocyanide_route_v1/` | Describe the hybrid method accurately; do not claim discovery of previously unknown routes |
| Generic retrosynthesis planners underperform on lipid-scale aliphatic chemistry relative to domain-qualified transforms | Inferred | Planner reach 60 of 120 versus deliberate application 161 of 163 and 11 of 11 | State as a methodological observation with both numbers shown |
| Availability is read from a dated online vendor snapshot rather than a local list | Computed | `results/phase1/ugi_online_procurement_snapshot_v1/result.json`, accessed 2026-08-06, expires 2026-09-05 | Report the access date and expiry wherever an availability number appears |
| Route closure under the frozen local index is 19 of 256, and 121 of 122 unresolved components were never expanded | Computed | `results/phase1/ugi_bounded_hybrid_route_cascade_v1/result.json` | If reported at all, always report the never-expanded split; never present it as a synthesizability estimate |
| Filtering candidates on makeability does not distort arm balance, potency or authority tier | Computed | `results/phase1/ugi_indomain_makeability_v1/`, bias audit | State that the panel is selectable on science rather than on tooling coverage |
| Branched candidates are makeable but poorly scorable, at an in-domain yield near 1.9% | Computed | `results/phase1/ugi_branch_exploration_applicability_v1/result.json` | State that applicability, not synthesis, limits branched candidates, and that AGILE's branch support is one aldehyde |
| Cross-engine agreement and engine-own-template self-consistency were tested as verification signals and rejected | Computed null | `results/phase1/ugi_engine_template_verification_v1/result.json`; cascade proposal ledgers | Retain as negative methodological results |
| Substrate scope of the applied transforms on these specific substrates | Not available | Forward reproduction is not scope verification; state the gap explicitly |
| Any prospective synthesis, formulation, in vitro or in vivo outcome | Not available | Placeholder only |
