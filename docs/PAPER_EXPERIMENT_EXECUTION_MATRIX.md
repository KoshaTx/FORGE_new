# Paper experiment execution matrix

Status: executable readiness and claim map for the current Nature Biotechnology paper, 2026-08-23.
This is an execution contract, not a preregistration and not a result. `AGENTS.md`, the frozen
scientific plans and later dated decisions remain authoritative if they conflict with this summary.

The machine-readable source of truth is
`configs/reproduction/natbiotech_v1_experiments.json`. Run `forge paper experiments` (or
`make paper-experiment-readiness`) to distinguish completed evidence, experiments that can launch
on the current source, implemented aggregates waiting for runs, and incomplete manuscript rows.
`forge paper experiments --strict` verifies that every retained numerical row has a runnable setup;
it does not claim that the corresponding production runs or paper results exist.

## Evidence boundary

FORGE is evaluated as a synthesis-grounded whole-lipid generative framework in a computational-only
paper. AGILE-type amine-aldehyde-isocyanide Ugi three-component chemistry is the deepest case.
BL_2023 aza-Michael and LX_2024 reductive amination test whether reaction programs provide reusable
computational coordinates; they do not inherit Ugi route-complete or biological evidence. The paper
contains no prospective synthesis, formulation, in-vitro or in-vivo experiments.

Every paper statement belongs to one of five evidence classes:

| Class | Meaning in this project |
| --- | --- |
| Measured | Direct observation in the frozen source data |
| Computed | Versioned analysis with pinned inputs and a verified result |
| Reported | Finding taken from a cited primary source |
| Inferred | Interpretation explicitly linked to measured, computed or reported evidence |
| Proposed | Unexecuted design, expected outcome or unresolved human decision |

No proposed or merely ready experiment may be written as a result. A completed negative experiment
remains computed evidence and is not reopened to obtain a preferred outcome.

## Computational experiments

| Question | Experiment and statistical unit | Current state | Paper role and allowed conclusion |
| --- | --- | --- | --- |
| Can a whole-graph Ugi model generate beyond the finite production corpus? | Existing production Ugi generation and audits; complete generated products are the reporting unit | Computed and frozen | Report 21,960 of 26,235 distinct admitted products (83.7%) absent from all 112,386 products used to fit the production model. The 89.0% train-fold number is not the production-model novelty claim. Novelty alone is not activity or synthesis evidence. |
| Do reaction-program coordinates transfer beyond Ugi without component tokens? | Four matched arms: Ugi-only conditioned, Ugi+BL+LX conditioned, shared null, and cyclic program-ID control; three independent training seeds | Exact-source H100 preflight passed; full runs are launch-ready on H100 | Report each family separately. Exact L1 replay is transform consistency, not synthesis-success probability. The A100-40GB request was rejected after the provider substituted an A100-80GB. The controls are nonselecting and make zero route or biological-oracle calls. |
| Does the shared model retain Ugi performance? | Paired independent-seed noninferiority comparison at the fixed final checkpoint; 10,000 deterministic bootstrap resamples | Aggregate adjudicator implemented; awaits the three full runs | Report point estimates and one-sided bounds for every frozen retention metric. Generated molecules within a seed are not counted as independent training replicates. A failed margin is a valid negative result. |
| Does semantic conditioning outperform generation followed by reaction filtering? | Shared conditioned arm versus the compute-matched shared-null arm, with exact reaction adapters applied to complete null-arm products; equal attempt budgets per family and seed | Implemented in the aggregate adjudicator; awaits the three full runs | Report raw validity, exact-L1 yield per attempt, decomposition coverage among valid products and forward-replay precision separately. This is the catalogue-free versus post-hoc filtering ablation. It does not select candidates. |
| Does vocabulary-free whole-graph generation escape a finite component inventory? | Shared conditioned arm versus a train-only oracle catalogue assembler using exact forward chemistry, source-weighted independent role marginals and the same per-family attempt budget | Implemented as a three-seed CPU experiment and integrated into aggregate adjudication; full runs pending | Primary differentiating metric: distinct valid, uniquely decomposed exact-L1 products with at least one component absent from the train catalogue per 1,000 attempts. Also report source-product catalogue coverage by fold, role-level component coverage, raw validity, exact-L1 yield, unique exact-L1 yield, whole-product novelty, diversity and effective component count separately. The catalogue has a zero component-escape ceiling by construction but may generate novel recombinations and may dominate validity. |
| Does the model generalize to held components? | Full component-disjoint reconstruction census and native heldout sampling for every supported family | Implemented in each production evaluation; awaits checkpoints | Primary robustness evidence. Report coverage and precision. Do not replace it with a random product split. |
| Does the model generalize to an entirely unseen reaction family? | Three leave-one-family-out arms and three paired seeds | Implemented and CPU-smoke qualified; full runs pending | Secondary robustness only. The result is explicitly marked `hard_gate=false` and cannot block or rescue the main claim. |
| Does joint whole-product generation learn cross-role dependence unavailable to independent role generators? | Within-context role-shuffling diagnostic followed by FACT-matched and FACT-generous; three paired seeds under the current manuscript contract | Diagnostic completed negative; three paired FACT-matched and FACT-generous production runs completed and independently verified | FORGE exceeded both FACT controls in every seed for Ugi and repeated aza-Michael but not for repeated reductive amination. Because the diagnostic did not detect dependence and was not family-disjoint, FACT remains an architecture control and cannot support a detected cross-role-dependence claim. |
| Which Transformer mechanisms matter? | Five one-at-a-time arms: input-only program, no role loss, no core loss, no routed adapters and no PCGrad | Three paired production seeds completed and independently verified; architecture rows rendered into the manuscript | Effects were chemistry dependent, and the no-gradient-conflict-control arm slightly exceeded the full model mean in all three programs. Each intervention is reported separately; the sparse-MPNN development failure is not an ablation. |
| Can a learned finite inventory close the gap to the oracle catalogue? | Autoregressive train-inventory selector under the common Ugi split and attempt budget | Implemented, common-assessor integrated and CPU-smoke qualified; three full runs pending | Strong learned finite-vocabulary baseline. Out-of-inventory component novelty is structurally zero, but validity, held-component behavior and recombination novelty remain empirical. |
| How does FORGE compare with published generators? | RGFN, unconditional DeFoG and GenMol/SAFE under the common Ugi protocol | Clean-checkout native runners, deterministic request packages, receipt validation and common attempt assessment are implemented; exact-checkout smoke packages pass and full native runs are pending | Ou DAG+Chem is excluded because no paper-specific author implementation is released; SynFlowNet is excluded because the released action space cannot represent a three-reactant Ugi step; SynCoGen is excluded because no usable code license is present. These methods remain related-work context, not empty empirical rows. Original-paper numbers cannot fill the table. |
| Do all method outputs close under the same recursive evidence policy? | Method-neutral exact-L1 assessment followed by a lookup-only, method-blind union of exact role/constitution evidence | Union freezer, blinded public worklist, private membership ledger and complete adjudicator are implemented; the final union is frozen only after all retained ledgers exist | Every method is evaluated against the same evidence union. Missing or unresolved evidence is an abstention, and the result is route-evidence closure rather than a synthesis-success probability. The bounded FORGE-only index is rejected by the strict paper collector. |
| Does nonzero synthesis guidance improve route-ready yield? | Matched synthesis-guided versus post-hoc diagnostic under the bounded planner contract | Computed negative and closed | The frozen failure audit found no matched route-ready gain. Production uses complete generation followed by proposal-augmented L1/L2/L3 routing. Do not repeat lambda tuning on the same schedule. |
| Does HeLa potency tilting improve the generator? | Single authorized applicability-gated HeLa diagnostic | Computed negative and closed | Potency morphology tilting was not promoted. Retain applicability-enriched proposal followed by conservative terminal ranking. This supports no in-vivo efficacy claim and authorizes no new same-data proposal search. |

The common Ugi baseline contract is frozen in
`configs/multireaction/common_ugi_baseline_protocol_v1.json`: train fold only, seeds 20260825--27,
3,072 attempts per seed, invalid and failed generations retained in the denominator, no repair or
retry, and one common constitutional and exact-L1 assessor. The retained RGFN, DeFoG and GenMol
adapters consume this exact export without candidate selection, repair or route-oracle calls. A
method row must still be executed under this contract or remain TBD and be excluded from empirical
conclusions; it cannot be populated from a published result on a different dataset.

## Production comparison acceptance contract

The four-arm comparison may launch only when all of the following are true on one unchanged source
snapshot:

1. all registered inputs verify and the checked-in missing-input baseline has no new failures;
2. the exact-source H100 benchmark passes every capacity, deterministic-replay, fixed-state,
   support and zero-call gate; the A100-40GB request is retained as a negative execution result
   because the provider substituted an A100-80GB and the physical-identity gate failed;
3. replicates 0, 1 and 2 use seeds 20260825, 20260826 and 20260827, respectively;
4. every arm uses 1,700 optimizer steps, effective batch 128, float32 deterministic execution and
   the fixed final checkpoint, with no early stopping or heldout-based selection;
5. all three runs have identical source, experiment specification, production design and inputs;
6. the aggregate evidence package snapshots and hashes every run manifest, stage manifest, training
   result, evaluation result and sample ledger.

The production result can be positive or negative. A runtime failure may be resumed only from the
matching fingerprint-bound restart state. A scientific gate failure is reported and is never relaxed.

## Explicitly out of scope

The historical 30- and 40-candidate panels, procurement refresh, chemist signoff, synthesis,
formulation, in-vitro assays, animal studies and biological endpoint selection are not experiments or
readiness gates for this paper. Their absence does not block completion. Historical candidate files
remain provenance records only and must not be recovered, revised or substituted as part of this
execution plan.

The computational paper may report route evidence, predicted properties and source-measured labels
only with their declared limitations. Computational route closure does not mean that a lipid was
synthesized. A supplier match does not mean that material was procured. Predicted potency is not a
measured prospective outcome.

## Main-display evidence map

| Display | Evidence required before finalization |
| --- | --- |
| Figure 1, FORGE concept | Whole-graph generation, reaction-program coordinates and recursive evidence states |
| Figure 2, Ugi computational evidence | Frozen Ugi generation, novelty, diversity, exact L1 and route-evidence audits |
| Figure 3, multi-reaction generality | Four-arm Ugi/BL/LX results, semantic controls and per-family metrics |
| Figure 4, causal ablations | Conditioned generation versus post-hoc filtering, finite-catalogue assembly, Ugi retention and held-component evaluation |
| Figure 5 or Extended Data | Closed negative synthesis-guidance and potency-guidance diagnostics; held-family stress only if later frozen and run |

## Exact execution order after the implementation expansion

The earlier executable fingerprint
`15765d81e863e14e3c71d95720fd42fe222cb62acea7d1d1ec33ec1562390d8a` and its passing L4
preflight are historical: the user subsequently authorized implementation of the retained baseline,
ablation and held-family rows. Those source changes intentionally invalidate that preflight.

1. Finish repository verification and freeze the new source fingerprint.
2. Run matched exact A100-40GB and H100 preflights on that fingerprint. Admit the A100 request as a
   negative execution result if the provider substitutes another class; `H100!` is required so the
   provider cannot substitute H200.
3. Launch the three full four-arm Transformer runs, three CPU catalogue runs, three learned-inventory
   runs, three mechanism/FACT runs and three secondary held-family runs with the paired seed set.
4. Prepare and run the retained native RGFN, DeFoG and GenMol requests for all three seeds, then run
   the common Ugi assessor on every canonical attempt ledger. Published numbers are never
   substituted, and excluded methods do not receive numerical placeholders.
5. Freeze the exact component union from all assessed method ledgers, adjudicate its blinded public
   worklist once, and finalize the method-blind route-evidence union. The private membership ledger
   is used only after evidence adjudication to compute per-method closure.
6. Adjudicate the production comparison and collect normalized paper rows with `forge paper
   collect-results-v1 --route-union <result.json>`. `forge paper render-results-v1` then refuses
   strict publication output until every retained method, mechanism and held-family row has all
   three exact seeds.

The paper has no prospective wet-lab execution step. The held-reaction-family result remains
secondary and never becomes a hard gate.

## Implementation closeout receipts

The executable source fingerprint after this implementation is
`d1d13f12d3b6cd7276b6d52c78730559df28d8ed97de4f4ca826392d92fdc762`. The following
current-source smoke receipts were each reproduced with byte-identical result artifacts:

| Experiment | Run ID |
| --- | --- |
| Learned train-inventory selector | `0d676eff79130064330f8685bf2c7a824dc7f45c4f25d88fd72790382e3d3f51` |
| Eight-arm Transformer mechanism/FACT study | `f6d230b580b9d9deac06a31c93a5e370e900ee218c5d3a9273eff2d3ccb57680` |
| Held-reaction-family study | `133501743e4ddfddb5ea451f05cd2b20a61b407e3e8eb28cfefc2ef62520cf2a` |
| Conditional role-dependence diagnostic | `d38479ad93661831204ed2437ef8b479c5b27285c258703f7d591072d5dfc035` |
| Finite-catalogue common-ledger smoke | `e2b9ae1c7ac376dbb8c14dfa8861f0f4230e20c66f3fb764025eea7e014a8df0` |
| Historical six-method common external export | `dc4ff30cb983937f09f191599e8bd5762e5dd8af7bf2549b5f825d0ce8be6502` |

These historical receipts qualified the implementations that existed at that source, not the
manuscript's numerical rows or the subsequently implemented native ports. The final-source
conditional diagnostic remains negative (mean AUC 0.48845 versus the frozen 0.60 threshold), so it
does not authorize the joint-generation dependence claim. Full paired-seed production starts only
after a fresh exact-source accelerator preflight.

The runnable-baseline and method-blind-union implementation before the accelerator change was pinned in
`results/phase1/external_baseline_route_union_readiness_v1/result.json` at executable-source digest
`085e063a5c87d4fcc0f3a2a343bde6118617355dba7de5ee22f054555267327e`. The exact-source NVIDIA L4
preflight passed all 23 gates as run
`9939ea913b54d9462213a7a8488436d1b12a1f7c2b11a3a30d3ea4385c25b7f5`; its downloaded artifacts
independently verify against the remote manifests. The subsequently authorized A100/H100 setup
changes executable source, so that L4 freeze is now historical and production is fail-closed until
the matched new benchmarks run and one exact target is selected. On source
`668fb8d431bc48da2b63992c75cc3f58b666d797b57117da0ba5c8d0765906c5`, exact H100 run
`9327c755b39b1217e0f838f33dc346a1b26e50b641cf446262f6059fb890b3e1` passed all 23 gates and is
the qualified production target. The A100-40GB request was served by an A100-80GB and remains a
first-class rejected substitution rather than a passing comparator.
