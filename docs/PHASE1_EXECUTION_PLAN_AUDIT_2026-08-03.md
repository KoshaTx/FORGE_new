# Phase 1 execution-plan audit

Date: 2026-08-03

## Purpose

This audit separates safeguards that protect a scientific claim from process
that merely creates additional gates. The standard is a minimal, load-bearing
experimental design: prevent leakage and retrospective tuning, preserve exact
chemistry and provenance boundaries, compare methods under a fair budget, and
retain null results. A gate is not retained merely because it is easy to hash
or phrase as a Boolean.

## Paper-level question and critical path

The central computational question is:

> Under the same trained whole-lipid generator, route planner, verifier and
> candidate budget, does synthesis information used during generation produce
> more distinct, route-complete, beyond-catalog Ugi-3 lipids than applying the
> same route assessment after generation?

The shortest valid path to that result is:

1. authenticate the selected product/L1 generator and current exact L2/L3
   source;
2. finish the grouped matched-arm runner and its ledger authentication;
3. prove selected-model grouped zero-guidance identity end to end;
4. freeze the synthesis-guidance calibration rule before guided outputs;
5. run calibration without accessing evaluation seeds;
6. freeze the selected guidance strength or record no selection;
7. run the paired evaluation once and retain positive, null or adverse results;
8. compute route, novelty, diversity, validity and collapse diagnostics from
   authenticated row-level ledgers.

The optional learned proposal lane can run in parallel. It does not block the
strict-evidence experiment and cannot create route evidence or synthesis value.

## Gate adjudication

| Item | Decision | Reason |
| --- | --- | --- |
| Constitutional identity, selected checkpoint and exact Ugi L1 reconstruction | **Keep hard** | These define the generated object and the trained model being evaluated. |
| Component-family-disjoint splits and heldout exclusion from model selection | **Keep hard** | These prevent component recombination from being mislabeled as generative generalization. |
| Broad non-Ugi pretraining before the Ugi model | **Do not require** | It is a decision-gated ablation, not a prerequisite for the paper's Ugi instantiation. |
| Biological guidance while every declared domain abstains | **Keep prohibited** | The current HeLa/RAW oracle does not justify steering generated chemistry or an in-vivo claim. Descriptive applicability remains allowed. |
| Exact current L2/L3 closure for a positive synthesis utility | **Keep hard** | A decomposition, handle or model proposal is not a complete route. |
| Binary pre-prospective route-completion utility with null censoring | **Keep hard for the first experiment** | It is conservative, interpretable and does not pretend to estimate synthesis success. Sparse support and collapse risk must be reported. |
| Current L3 decision horizon | **Keep hard for route-complete labels** | Procurement evidence is time dependent. Refresh it for prospective lock rather than imposing an unnecessary continuous service. |
| Selected-model grouped lambda-zero identity | **Keep hard** | This proves the controller and restartable refactor do not change the base generator when guidance is absent. |
| Sixteen programs times four particles, with ancestry only within a program | **Keep for the primary experiment** | It permits nontrivial ancestry while preserving matched morphology coverage. Treat other group sizes as later ablations. |
| Censored checkpoint assessment gives zero incremental weight but remains null | **Keep hard** | This prevents missing planner knowledge from becoming a false negative route score. |
| Isolated guided/post-hoc cache overlays and measured compute ledgers | **Keep hard** | One arm must not warm the other, and the comparison must measure rather than assert its budget. |
| Lock all productive post-hoc molecules before any shadow assessment | **Keep hard** | This prevents a supposedly post-hoc evaluator from influencing generation or the productive pool. |
| Calibration/evaluation seed separation and no seed search | **Keep hard** | Guidance strength cannot be chosen from the evaluation result. |
| All five evaluation seeds must be strict wins | **Keep as a conservative internal promotion rule, not the sole publication statistic** | Report paired effect sizes and uncertainty. A later seed-count extension must be frozen before evaluation, not triggered by the result. |
| Exact final candidate count and deduplication before ranking | **Keep hard** | This makes the endpoint comparable and prevents duplicates from inflating route closure. |
| Graph2Edits as a prerequisite for synthesis guidance | **Remove** | The strict exact-evidence route system is sufficient for the first causal experiment. |
| Graph2Edits source, checkpoint and inference-policy hashes | **Keep hard if Graph2Edits results are reported** | They identify the external model actually evaluated. |
| Isolated version-locked Graph2Edits runtime and local-only inference | **Keep hard if executed** | These prevent dependency and download drift. |
| Institutional license review encoded as a scientific/code gate | **Remove** | The checked artifact declares MIT. Record provenance and any owner-required legal attestation out of band; software cannot adjudicate institutional policy. |
| Ten byte-identical runs on a separate production device | **Remove** | The locked runtime already produced two identical normalized runs. A small semantic repeat smoke in the benchmark environment is sufficient; cross-device bitwise equality is not required. |
| Frozen proposal targets, hidden truth, budgets and thresholds | **Keep hard if the benchmark is run** | These prevent target leakage and post-result threshold selection. |
| Independent forward verification of learned proposals | **Keep hard** | A proposal engine cannot verify its own disconnection. Exact reconstruction establishes graph consistency only. |
| Exact-pair resolver as evidence authority | **Reject** | Resolver scope and evidence scope are separate. The current exact-pair resolver is intentionally conservative for the strict-evidence experiment; broader family applicability requires its own chemistry qualification and still cannot create evidence. |
| Manual `all_gates_passed` and duplicate execution-authorized Booleans | **Remove from the next executable contract** | Readiness should be derived from authenticated prerequisites. Production activation remains a separate versioned decision. |
| Proposal-model score in synthesis value | **Keep prohibited** | A beam score is neither substrate evidence nor a probability of experimental success. |

## Proposal-engine decision

Graph2Edits remains an optional, proposal-only baseline. Its first benchmark
should answer whether it recovers known exact route hypotheses and adds useful
forward-consistent hypotheses under a matched budget. The current exact-pair
resolver is adequate for a conservative known-route recovery benchmark. It is
not adequate evidence for general novel-family route closure, and the benchmark
must not imply otherwise.

Before executing that benchmark, the scorer must additionally report:

- per-stratum and paired per-target lane results;
- strict-only, learned-only and overlapping proposal provenance;
- invalid, duplicate and budget-exhaustion counts;
- wall time and peak memory; and
- deterministic adjudication of the frozen operational promotion rule.

Graph2Edits is activated in the production planner only through a later
versioned configuration if it adds useful hypotheses without admitting any
adversarial incompatible target. Even after activation, its outputs remain
non-evidentiary until independently supported.

### Discovery-source neutrality

Proposal provenance controls which checks remain outstanding; it must not
create a permanent route-value penalty. A route discovered by Graph2Edits that
subsequently satisfies the same forward-consistency, substrate-scope, evidence,
operational and terminal-closure contract as a route already encoded in FORGE
receives the same qualified route state and the same synthesis utility. Retain
the discovery source for auditing, but do not reward prior documentation or
penalize model-assisted discovery after evidentiary equivalence is established.

Before equivalence is established, the states remain different for a real
reason: a model-only hypothesis has unresolved evidence rather than evidence of
failure. It is labeled `missing_evidence` or `unqualified_hypothesis`, never
`chemically_incompatible` merely because it was absent from the initial
registry.

## What is sufficient for the computational paper package

The computational package is decision-grade when it contains:

1. the frozen and evaluated product/L1 generator;
2. exact L1/L2/L3 route-value provenance and support coverage;
3. a real zero-guidance equivalence receipt for the final grouped runner;
4. the matched synthesis-guided versus post-hoc result, including a retained
   null or adverse result if that is what occurs;
5. validity, uniqueness, diversity, broad-distribution coverage, component
   novelty and collapse analyses;
6. authenticated calibration selection and evaluation ledgers; and
7. manuscript figures and text that distinguish computed route completion
   from prospective synthesis success.

The learned proposal benchmark is useful but optional. Biological guidance is
not required for this computational comparison and remains disabled until a
separate applicability decision supports it. Prospective synthesis,
formulation and biological testing remain necessary for the final
Nature Biotechnology paper.

## Immediate execution order

1. Complete the current nonzero-runner adversarial review and regression tests.
2. Bind the exact frozen seed-to-program schedule and cache-overlay receipts in
   executable code, not caller attestations.
3. Run the selected-model grouped lambda-zero identity test.
4. If it passes, run the strict-evidence calibration sweep.
5. Freeze the selected strength or a no-guidance decision.
6. Run the paired evaluation once.
7. In parallel, revise and execute the optional Graph2Edits benchmark only
   after its scorer implements the frozen scientific decision.
