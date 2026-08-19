# M0-09 L2 supervision decision

## Outcome

The current evidence supports a hierarchically joint FORGE architecture:

1. jointly learn the complete lipid and its exact L1 Ugi component
   decomposition;
2. close nonterminal components through a hybrid recursive L2 system that
   combines deterministic atom-mapped transformations, bounded search, and
   learned proposal or ranking;
3. keep L3 procurement, evidence strength, route closure, and prospective
   outcomes as separate states.

The inventory does not support a monolithic decoder trained to emit a complete
product and every multistep route branch from one latent representation.

This is an evidence-based architecture choice, not a reduction in jointness.
Synthesis value still feeds back into whole-lipid sampling. The factorization
places dense product and L1 supervision in the joint model while reserving
sparse, compositional L2 chemistry for the subsystem best matched to the
available data.

The factorization also protects molecular support. A monolithic decoder could
confuse frequently documented chemistry with the full useful and synthesizable
space, favoring the 93 familiar components, short routes, and dominant
reaction families. The whole-lipid loss must therefore continue to train on
the broad lipid corpus while decomposition losses are masked to reviewed Ugi
examples. Ugi-only fine-tuning without broad-corpus replay is not allowed.

## Evidence at the decision point

| Stratum | Frozen observation |
|---|---:|
| Nominal AGILE rows passing mechanical Ugi-site policy | 1,200 |
| Reconciled exact single-compound L1 records | 1,100 |
| B4 mixture executions excluded from single-graph supervision | 100 |
| AGILE virtual products with one exact Ugi decomposition | 12,276 |
| Unique AGILE virtual components | 93 |
| Chemistry source packages reviewed | 4 |
| L2 route instances | 45 |
| Structure-resolved L2 route instances | 44 |
| L2 reaction instances | 76 |
| L2 route families | 11 |
| Components with exact source programs | 24 |
| Components with family-projected programs | 47 |
| LX_2024 direct aldehyde transfers | 15 |
| Reported negative synthesis outcomes | 0 |
| Upstream amine-head routes | 0 |
| Complete product routes through every L2 and L3 branch | 0 |

The positive observations are sufficient to encode reusable chemistry, seed
route templates, and train later proposal or ranking components. They are not
balanced or complete enough to justify a learned monolithic route-tree
decoder. In particular, the absence of negative outcomes would make a
standalone learned feasibility model poorly calibrated, and the absence of
head routes and completely closed product trees leaves critical supervision
missing.

Before prospective outcomes exist, the synthesis signal is an
evidence-weighted route-completion value. It combines closure, forward
consistency, evidence, burden, and uncertainty under a frozen policy. Raw
route-model likelihood is not an acceptable substitute, and the value must not
be called a calibrated probability of experimental synthesis success.

For coupled generation, the synthesis value must change molecular transition
probabilities before candidate lock. The final guidance strength is selected
subject to frozen floors on diversity, broad-corpus coverage, and component
novelty. Route closure is reported separately for familiar AGILE components,
transferred known components, and genuinely generated components.

## Frozen future stopping thresholds

These thresholds apply only after the generator, oracle applicability policy,
and candidate pool are frozen. They do not claim that operational closure has
already been reached.

| Criterion | Threshold |
|---|---:|
| Weighted high-priority motif route support | at least 90% |
| Eligible candidates with complete routes | at least 90% |
| Eligible candidates with a route decision, including substantive rejection | at least 95% |
| Eligible candidates failing only for missing route knowledge | at most 5% |
| Marginal weighted-coverage gain for saturation | at most 2.5% |
| Consecutive low-gain curation rounds | 2 |
| Locked prospective candidates with complete dossiers | 100% |
| Locked prospective candidates passing forward consistency | 100% |
| Locked prospective candidates with current terminal closure | 100% |
| High-risk precursors with a backup route | 100% |

Candidate-route decision coverage includes complete routes, chemically
substantive incompatibility, and declared out-of-support decisions. It does
not count an unexamined or undocumented candidate as resolved.

## Mining decision

The broad source-paper phase stops here. JC_2023 strengthened native Ugi
aldehyde and isocyanide supervision, and LX_2024 supplied 15 exact
cross-platform aldehyde transfers. Further chemistry review is
targeted and gap-driven only.

Additional mining is triggered by:

- repeated missing-route-knowledge failures among biologically eligible
  generated candidates;
- an unresolved head or terminal material needed for the locked panel;
- a declared need for additional isocyanide substrate-scope evidence;
- negative or failed synthesis observations needed for calibration.

This policy avoids both failure modes: mistaking missing documentation for
chemical impossibility and attempting to encode every reaction reported in
LNPDB before building the model.

## Claim boundary

The decision supports this statement:

> FORGE jointly generates whole lipids and their final assembly
> decomposition, then recursively constructs complete precursor programs using
> a hybrid synthesis layer whose value guides molecular sampling.

It does not support claiming that FORGE has already learned complete synthesis
across all lipid chemistries. Other final-assembly families can reuse the
route-tree representation and upstream chemistry, but each requires its own
adapter, evidence corpus, and validation.

## Reproduction

```bash
make m0-09-l2-supervision-decision
```

The generated artifact is
`results/m0_09/l2_supervision_decision.json`. Every input is schema-checked and
SHA-256 pinned.
