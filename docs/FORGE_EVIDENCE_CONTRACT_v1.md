# FORGE evidence contract v1: falsifiable analyses and stop rules

Recorded 2026-08-15, before any of the numbers below were computed. This document exists because
the failure mode of the last four weeks was not a shortage of analyses. It was analyses whose
decision rule was chosen after the number appeared. Every entry here fixes the question, the
estimand, the unit, the denominator, the decision rule and the stop rule in advance. Where a
number was seen before its rule was written, that is disclosed in the entry itself.

Governing constraint for this phase:

> **Do not redesign the generator unless the origin ablation or the held-family evaluation reveals
> a real failure.** Strengthen the paper through cleaner evaluation, stricter qualification and
> prospective comparison, not through architecture churn.

Three named triggers, and only these three, license generator work:

1. held-family generalization is poor;
2. the origin channel actively harms generation;
3. ambiguous or chemically problematic outputs contaminate the qualified population, not merely
   the broad exploratory pool.

The active-design branch is closed and is not reopened by any result here. Route-guided flow
matching, hierarchical component generation, new acquisition methods, additional flow objectives,
larger backbones, second reaction adapters and post-hoc chemistry regularizers are out of scope.

---

## 0. What "chemist-like" is allowed to mean

Operational, never cognitive. The claim is a property of the framework, not of the neural sampler.

Permitted:

> FORGE learns a data-driven design grammar over the chemically meaningful precursor roles of
> Ugi-3 lipids. Combined with explicit assembly verification, evidence-bounded ranking, route
> resolution and expert review, that grammar supports semirational exploration beyond a
> predefined component library.

Not permitted: that the model has learned reaction mechanism, experimental intuition, causal
delivery rules, or that any raw sample is a molecule a chemist would choose to make.

The operative distinction, to be held in every section:

> The flow is an ideation engine. The complete pipeline is the chemist-aligned design system.

Three measurable properties carry the claim, and nothing else may be offered in its place:
chemistry-native representation (A1, A2, A3), experimental actionability (C, D), prospective
realization (F, G).

## 0b. What "synthesis-grounded" is allowed to mean, by stage

| Stage | Permitted claim |
|---|---|
| now | route-qualified: forward-verified assembly plus a documented component route to purchasable starting material under a dated snapshot |
| after synthesis | synthesized to a prespecified identity and purity threshold, reported with attrition |
| never, without outcomes | "synthesizable" unqualified, "the routes will work", "the components are easy to prepare", "route completeness predicts yield or purity" |

"Synthesizable" enters the title only after experimental route success is known.

---

## A. Origin-channel ablation (priority 1)

The ablation is split into three parts because `origin_states` is consumed in at least four
distinct ways, and deleting one embedding does not remove the others. Parts A1 and A2 are CPU
work that must complete before any GPU time is spent, because they determine whether the training
run is even well posed.

### A1. Information audit: what role information survives removing the origin embedding

**Question.** After deleting `origin_embedding`, is an atom's precursor origin still determined by
the node channels the model continues to receive?

**Channels.** `AdapterNodeConditioning` embeds five node channels: `origin_states`,
`core_position_states`, `port_states`, `distance_to_core`, `distance_to_own_port`.
`use_all_port_distances` is False in production and is never set True anywhere in the repository,
so all-port distances are not a channel and are excluded. Two of the four remaining channels are
defined relative to the atom's own role, so leakage is expected but its magnitude is unknown.

**Estimand.** Accuracy of the Bayes-optimal predictor of `origin_states` from the tuple
(`core_position_states`, `port_states`, `distance_to_core`, `distance_to_own_port`), fitted as an
empirical conditional distribution on the train fold and evaluated on the heldout fold. Reported
alongside normalized conditional entropy H(origin | channels) / H(origin).

**Unit and uncertainty.** Atoms. Atoms are clustered within products, so the interval is a
cluster bootstrap at the product. Both estimands are reported: atom-level accuracy and
product-level exact-match, meaning every atom in the product recovered.

**Prespecified decision rule.**

| heldout atom accuracy | consequence |
|---|---|
| >= 0.95 | deleting the origin embedding alone is not an information ablation. The primary comparison escalates to level B. No arm may be described as "product-only" anywhere in the paper. |
| 0.60 to 0.95 | level A is a partial ablation. The leakage figure is reported in the same sentence as any level-A result. |
| < 0.60 | level A is a genuine information ablation and may be run as the primary arm. |

**Level B**, referenced above, drops the whole role-indexed group `{origin_states, port_states,
distance_to_own_port}`, leaving `core_position_states` and `distance_to_core`. The same estimand is
computed for level B in the same pass, under the same rule.

**Stop rule.** One pass. The two feature sets above are the only two evaluated. No third feature
set is added after seeing the answer.

**Falsifies.** A finding at or above 0.95 falsifies any claim that a level-A ablation measures the
value of precursor-origin information.

### A2. Structural-dependence audit: where else origin enters

**Question.** Which consumption sites of `origin_states` survive each ablation level?

**Output.** An exhaustive `file:line` list, each site classified as input conditioning, training
objective, sampling constraint, or evaluation only. Not a number.

**Prespecified rule.** Any surviving site in the training objective or the sampling constraint must
be named in the paper's description of the ablation. The word "origin-free" may be used only if no
such site remains.

**Stop rule.** One pass over the repository at a pinned commit.

### A3. Ablation training run (GPU)

Runs only after A1 and A2 fix the level. Matched to production on data, folds, seed policy, step
budget, batch size, sampler and decoding budget. The checkpoint is taken at the same fixed step by
the same rule as production. No checkpoint shopping.

**Seeds.** Three per arm. The active-design branch died because a mean-level difference turned out
to be seed variance, and a single-seed ablation cannot distinguish neutral from noisy.

**Primary endpoints, fixed here.** Held-family validity; admission rate; structural fidelity under
matched morphology; exact-forward-reconstruction rate.

**Prespecified decision rule.**

- Ablation clearly better, meaning non-overlapping seed ranges in its favour on validity and
  held-family loss and fidelity: simplify the model or redesign how origin is represented.
- Neutral, meaning overlapping seed ranges: keep origin as a native structured-output interface,
  claim no generative quality gain, and do not restore "reaction-resolved" to the title.
- Ablation clearly worse: origin is an inductive bias and may be claimed as one, with the leakage
  figure from A1 stated alongside.

**Stop rule.** Three seeds per arm, once. A neutral result is a result. No additional seeds are run
in search of a difference.

---

## B. Held-component-family generative evaluation (priority 2)

**Checkpoint.** `results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt`, the
development checkpoint trained on the 66,464-product train fold only. No retraining is required or
permitted for this analysis.

**Question.** Does the generator produce valid, admitted, chemically plausible products built from
component families absent during training, or does it only recombine familiar components?

**Endpoints.** Validity, admission, uniqueness, structural fidelity under matched morphology,
component-level novelty by role.

**Unit and uncertainty.** Both estimands, always: product-level and distinct-component-level.
Cluster bootstrap at the component family. Morphology conditioning is stated as omega(c) for every
comparison; no marginal comparison is reported without its standardized counterpart.

**Prespecified decision rule.** This analysis, not the chemotype audits, carries the fidelity and
generalization claim. If the model cannot produce valid and admitted products for held-out
families under matched morphology, trigger 1 fires and generator work is licensed.

**Stop rule.** One evaluation at a frozen sampling budget, fixed before the run.

---

## C. Forward-outcome census on the supported and prospective populations (priority 3)

**Question.** For every chemically admitted design, is the target the unique enumerated forward
product of its own components, or one of several?

**Why it matters.** Admission verifies that the target *is* produced and that enumeration did not
saturate. It does not verify that the target is produced *alone*. Those are different properties
and the paper has been conflating them.

**Population.** Full census, no sampling. The union of the main and branch rescoring ledgers, at
the unit of distinct product graphs. Forward enumeration costs seconds at this scale, so sampling
would be a choice, not a constraint.

**Strata.** Prediction-supported status; inside or outside the 12,276-product AGILE enumeration;
aldehyde chemotype; frozen-40 membership.

**Closed-world baseline.** The identical statistic computed on the 12,276-member enumerated
library, so that non-uniqueness can be attributed to generated chemistry or to Ugi-3 lipid
chemistry in general.

**Prespecified decision rule.** Any frozen-40 member whose target is not the unique forward outcome
is flagged. It is replaced before synthesis if a same-arm, route-complete substitute exists;
otherwise the competing outcome is documented as a named risk in that candidate's dossier. Trigger
3 fires only if non-uniqueness is common in the prediction-supported or route-complete population,
not if it is confined to broad exploration.

**Disclosure.** A 300-row head-of-ledger timing probe returned 249 of 300 unique before this rule
was written. The rule above was written with that number known. The probe was the first 300 rows in
file order, not a random sample, and is not reported as a result.

**Stop rule.** One census. The enumeration limit `max_forward_outcomes_per_candidate = 100` is
inherited from the frozen capability config and is not tuned.

**Falsifies.** A non-uniqueness rate in the route-complete population comparable to the enumerated
library's own rate falsifies the reading that generated chemistry is distinctively ambiguous.

---

## D. Inside-versus-outside-enumeration actionability funnel (priorities 4 and 5)

**Question.** Does chemical expansion survive evidence and synthesis qualification, or does every
downstream gate collapse the system back to the original library?

**Strata.** Inside or outside the complete 12,276-product AGILE enumeration.

**Stages.** Chemically admitted; unique forward outcome; prediction-supported; inside the
physicochemical envelope; all components resolved; route-complete within the step budget; frozen
40; final 12.

**Both estimands.** Product-level counts and distinct-component-level counts at every stage.

**Uncertainty.** Cluster bootstrap at the aldehyde component. Products sharing an aldehyde are not
independent evidence about routeability, which is the same pseudo-replication that inflated the
composition analysis.

**First-excluding gate.** Reported per stratum. A stratum removed before route assessment has an
unmeasured route completeness, and that must be stated rather than imputed.

**Route detail, stratified.** Preparative step count, number of components requiring preparation,
and unresolved failure modes by role.

**Prespecified decision rule.** Report Pr(route-complete | outside) minus Pr(route-complete |
inside) with a clustered 90% interval. Parity may be claimed only if that interval lies within plus
or minus 10 points. Otherwise the measured trade-off is reported as the result, which remains
publishable and is not retried with a different threshold.

**Stop rule.** One pass. The 10-point margin is fixed here and not revised after seeing the
interval.

---

## E. Pooled-versus-tier conformal decision (priority 6)

Exact comparison, decided once, on the criterion already frozen in the calibration contract. No new
calibration variant is introduced.

---

## F. Freeze the 12 (priority 7)

All-HIGH, blinded, unconstrained brief. Frozen only after C flags no unresolved competing outcome
among the selected candidates, or after each such candidate carries a documented risk statement.

---

## G. Closed-world enumerated comparator cohort (priority 8, capacity permitting)

**Question.** At matched experimental budget, does de novo generation yield useful lipid chemistry
beyond what exhaustive enumeration and ranking of the original library can supply?

**Design.** 12 generated candidates, preferably outside the enumeration; 6 to 12 top unmeasured
enumerated-library candidates selected with the same predictor, the same physicochemical
constraints and the same protocol; measured AGILE reference lipids rerun concurrently; identical
formulation, characterization, transfection and viability assays.

**Endpoints.** Synthesis success; identity and purity; LNP formation; size and PDI; encapsulation;
transfection; viability; best observed activity; hit rate above a threshold frozen in advance;
experimental and preparative burden.

**Prespecified reporting.** Both outcomes are reported in full. A negative result, meaning
out-of-library candidates are experimentally viable but do not outperform closed-world ranking at
this sample size, is a publishable result and is written as one.

---

## H. Prospective synthesis attrition (priority 9)

Per candidate, the full chain is preserved: starting materials obtained, tail components prepared,
final Ugi product obtained, identity confirmed, purity threshold passed. Stratified by route type:
all components purchased; aldehyde preparation required; isocyanide preparation required; both
tails prepared. Reported with failures by stage and actual versus planned step count.

---

## I. Blinded expert actionability assessment (optional, after F)

Source-blinded, morphology-matched mixture of held-out real Ugi lipids, unmeasured
enumerated-library candidates, route-qualified FORGE candidates, and broad generated candidates
that have not undergone full qualification. Fixed rubric, acceptance rate by source, inter-rater
reliability. Called a blinded expert actionability assessment, never a Turing test.

---

## J. What is explicitly not being built

Route-guided flow matching; a hierarchical component generator; a new active-acquisition method;
another flow objective; a larger backbone; a second reaction adapter; a chemistry regularizer
designed after seeing the unusual aldehydes; any additional unconditional sampling run whose
purpose is a larger headline count, a prettier chemotype, or a larger novelty percentage.

More draws are licensed only for: a confirmation arm under a prespecified deployment policy;
replacing candidates that fail unique-outcome or route qualification; a separate exploratory cohort
from unsupported precursor classes; or filling the enumerated comparator.

The scarce object is experimental labels, not generated molecules.

---

# Amendment 1: Stage 2 sampling protocol and the ablation comparison

Recorded 2026-08-16, **before any held-family sampling output exists**. What was known when this
was written: the stage 1 teacher-forced results (contract section B), the origin information audit
(A1), the structural-dependence audit (A2), the forward-outcome census (C) and the actionability
funnel (D). No sample has been drawn under any held-family program.

## B2. Held-family conditional sampling

### Fixed resources

| resource | value |
|---|---|
| joint checkpoint | `results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt` |
| closure checkpoint | `results/phase1/ugi_closure_expanded_full/checkpoint_best.pt` |
| closure fold provenance | trained on strict train-fold cyclic components, calibration for selection, heldout untouched. Verified before use, because a closure model trained on all folds would leak heldout structure into a development-checkpoint evaluation. |
| prepared cache | pinned by the joint checkpoint's own `inputs` hash |

### Program draw, sealed before sampling

Programs are drawn from heldout-fold products and sealed to disk with their SHA-256 before any
sample is generated. Three strata, each recorded separately and none merged:

1. **held-aldehyde-family programs** drawn from the 21,304 heldout products whose aldehyde family
   fold is `heldout`;
2. **train-aldehyde-family programs** drawn from the 7,148 heldout products whose aldehyde family
   fold is `train`;
3. the **matched subset** of stratum 1, restricted to programs that also occur in stratum 2.

Stratum 3 exists because four of the six held aldehyde families share no morphology program with
any train-family product. **Those four families are reported, never dropped.** Any statement about
them is a statement about stratum 1 only, and must say so.

### Metrics, fixed here

Per stratum, and per role where the metric is role-indexed:

- decoded validity, and chemical admission under the qualified forward transform;
- exact forward reconstruction: the decoded components rebuild the decoded product;
- **unique** forward outcome, by the same census procedure and the same enumeration cap of 100;
- exact per-role morphology adherence: realized node count, junction budget, cycle rank and
  attachment count against the requested program, reported as exact-match rate per role, not as a
  mean absolute error that hides systematic bias;
- component novelty by role against the structural registry and against the train fold;
- product novelty against the train fold and against the 12,276-product enumeration, both sides
  stereo-free;
- distinct-component counts as well as product counts, always.

### Decision rule

This analysis carries the fidelity and generalization claim. Trigger 1 fires, and generator work
becomes licensed, only if the model **cannot** produce valid, admitted products under held-family
morphology. Degraded-but-working is not trigger 1; it is a reported result.

### Stop rule

One draw per stratum at a budget fixed before sampling. No resampling with a different budget,
temperature or seed after seeing admission rates.

## B3. Role-local intervention, secondary

Prespecified now so it cannot be tuned later. It is secondary and does not alter the ablation's
primary gate.

For a base program `c = (c_a, c_d, c_i)`, perturb exactly one role's sub-program to `c'`, holding
the other two sub-programs and the sampling randomness fixed under common random numbers. The
perturbation is a fixed change of the target role's node count, applied identically across arms,
and rejected if the result leaves the declared support envelope.

Measured:

- **on-target**: exact-match rate between the realized and requested node count for the target role;
- **off-target**: fraction of non-target roles whose decoded component is byte-identical to the
  base run's, plus the ECFP4 Tanimoto distribution for those that changed;
- **preservation**: validity, admission and unique assembly, before and after.

The claim available if it succeeds is bounded: *role-specific conditioning changes the intended
head or tail morphology with limited off-target change in the remaining roles.* Not that the rest
of the molecule is literally identical.

## A3 amendment: how the ablation is compared

### Absolute loss levels are protocol-dependent and are not comparable across artifacts

Measured while building stage 1: the denoising loss falls monotonically in `t`, from 5.42 at
t = 0.02 to 2.76 at t = 0.98 on a fixed heldout subset. Any evaluation using a different `t`
distribution or a different record set produces a different absolute level.

Consequence, recorded so the project does not end up with two conflicting "held-family loss"
numbers: the existing gate artifact
`results/phase1/ugi_joint_sparse_balanced_v2_held_component_novelty_gate.json` reports totals of
1.5 to 2.3 on a small frozen probe grouped by which roles are heldout. **That is a different
population under a different protocol and must not be compared to the section B numbers.** Only
within-protocol contrasts are meaningful.

### The ablation is paired, and reported against t

Both arms are evaluated on identical products, identical morphology programs, identical `t`
values from the same fixed grid, and identical noise generator seeds. The reported quantity is
the **paired difference**, never two unrelated averages.

The difference is reported **as a function of t across the grid**, not only as a grid average.
The reason is the hypothesis itself: the repaired inverse shows origin is a deterministic function
of any completed graph, so an origin effect can only exist in the partially generated regime. If
explicit origin conditioning helps, it should help most at low `t`, where the state is most
corrupted. A flat Δ(t) is evidence against the mechanism even if the average is nonzero.

### Seeds

Three per arm, six runs total, matched on corpus, folds, architecture apart from the removed
embedding, optimizer settings, step budget and inference seeds. "Clearly better" requires
non-overlapping seed ranges. A neutral result is a result; no additional seeds are run in search
of a difference.

### Escalation

After this experiment the representation branch stops. A fully role-collapsed model is **not**
built automatically. Escalation requires the origin-embedding result to be genuinely ambiguous
about a load-bearing claim, and it is a decision to be taken explicitly, not a default.

---

# Amendment 2: a correction, and two loosened rules

Recorded 2026-08-16, before any ablation training run and before any sampling output exists.

## The correction: A1 measured a module the production generator does not use

`AdapterNodeConditioning`, with its five-state `origin_embedding` and its port, core-position and
distance channels, belongs to `UgiChemistryFlow`. **The production generator is
`UgiJointSparseFlow`. It does not instantiate `AdapterNodeConditioning` and has no
`origin_embedding`.** Its role channel is `role_embedding`, of shape (3, 192), 576 parameters,
0.0533% of the 1,081,388-parameter model.

A1's numbers are correct for what they measured. **Their application to the production model is
withdrawn.** The 0.552 recoverability figure must not be quoted as a property of the production
generator, and the A1 verdict "level A is a genuine information ablation" does not hold for it.

### What replaces it

Verified over 9,000 records spanning all three folds, zero mismatches:

> `role_states` and `within_role_positions` are **exact deterministic functions of the
> conditioning program**. The program declares per-role node counts and the sparse serialization
> lays nodes out in contiguous role blocks in that order, so role is reconstructible with
> certainty from conditioning the model always receives.

Recoverability is therefore **1.0, not 0.552**. Under this contract's own A1 rule, recoverability
at or above 0.95 means deleting the embedding **is not an information ablation**. Deleting
`role_embedding` removes a redundant re-encoding of the conditioning, not the role information.

### Consequence for A3

**The six runs as specified must not launch.** They would test whether an explicit re-encoding of
a deterministic function of the conditioning aids optimization, at 0.053% of the parameters. That
is a runnable experiment and a legitimate small one, but it is not the experiment the contract
described, and it cannot support any claim about whether precursor-role information helps
generation.

A genuine role-information ablation would require changing the state layout so that role is no
longer implied by position. That changes the modelled object rather than removing a channel from
it, and this contract puts it out of scope.

The defensible reading, which is stronger than an ablation result would have been:

> Role resolution in FORGE is carried by the conditioning program and the state layout, not by a
> removable component. It is a property of the representation, not a feature that can be toggled.

Artifact: `results/phase1/forge_production_role_channel_audit_v1/result.json`.

## Loosened rule 1: Δ(t) is secondary, not a gate

The A3 amendment in Amendment 1 made a flat Δ(t) evidence against the embedding. That is too
strong and is **withdrawn before any run**. An embedding can help representation learning
throughout the trajectory, and the surviving role channels can change the expected time
dependence, so absence of the guessed temporal signature does not refute a real effect.

The primary question for any arm comparison stays: does the manipulated conditioning improve
held-out denoising or conditional sampling? Δ(t) is a **prespecified mechanistic secondary
analysis** with three readings, none of which is a gate:

| Δ(t) | reading |
|---|---|
| positive, strongest at low t | helps, and through the partial-state mechanism |
| positive and roughly uniform | helps, but not through that mechanism |
| approximately zero across t | adds little generative value |

## Loosened rule 2: Stage 2 is named for what it does

Stage 2 is renamed **conditional sampling under held-family-derived morphology programs**, short
form **OOD morphology-conditioned sampling**. The model receives morphology as test-time
conditioning, not held-out molecular identities, so the analysis does not show that FORGE
regenerates unseen chemical families, and no wording anywhere may imply that. Stage 1 already
showed why: four of six held aldehyde families occupy programs with no train-family counterpart,
and exact matching reaches 8.2% of held-aldehyde products.

Its question, in full:

> When asked to generate under structural programs associated with held-out chemistry, does the
> model remain valid, chemically admitted, morphology-faithful and stable?

## Added: OOD degradation, and the interaction, prespecified

For each model and each metric M in {admission, morphology exact match, unique assembly, validity}:

```
Δ_OOD = M(held-derived programs) − M(train-derived programs)
```

and, for any two-arm comparison,

```
Δ_arm = Δ_OOD(arm B) − Δ_OOD(arm A)
```

Prespecified now, and **not a sole gate**. It asks whether a manipulated channel makes the
generator more robust when requested morphology moves toward regions associated with unseen
component families.

## Added: parameter fairness, measured rather than assumed

| arm | parameters |
|---|---|
| full | 1,081,388 |
| minus `role_embedding` | 1,080,812 |
| difference | 576, or 0.0533% |

The difference is trivial, so it is reported and **no parameter-matched control arm is built**.


---

# Amendment 3: the one remaining model experiment, frozen

Recorded 2026-08-16, before the run. Everything upstream is closed: no architecture hunt, no
generator redesign, no further ideation. This is the single experiment that remains justified.

## Why it is needed

The corrected Stage 2 arm uses the production architecture but a step-750 checkpoint, against
production's 5,100. The residual gap between 0.190 and 0.655 ester-linked tails therefore
confounds **training budget** with **fold restriction** and cannot be attributed without a run
that removes one of them.

## The run

Train the **exact production configuration** on the **structural training fold only**:

- same decoration-state conditioning (`bidirectional_anchor_local`), serialization, loss,
  vocabularies and source-marginal construction;
- same optimizer settings and batch size;
- **the production step budget**, scaled by the recorded per-record rule, not the development
  early-stopping schedule;
- checkpoint taken by the production fixed-step rule, never by the fidelity panel;
- heldout families untouched; calibration used only as the historical protocol specifies;
- three seeds if affordable. With one seed it is a decisive diagnostic, not a variance-estimated
  superiority claim.

## Endpoints, and admission is not among them

Primary comparison against matched real held-out lipids, the role-marginal null and random
initialization, on the frozen panel:

**Head.** Rate of the wanted architecture (one Ugi handle plus a tertiary ionizable centre); rate
of two or more genuinely Ugi-reactive sites; handle-to-ionizable-centre spacing; cyclic against
acyclic tertiary architecture.

**Aldehyde tail.** Ester prevalence and placement; alternative linker chemotypes; chain length;
branching and branch position; unsaturation.

**Isocyanide tail.** Hydrophobic carbon count; chain architecture; branching; unsaturation;
terminal chemistry.

**Whole lipid.** Head-to-hydrophobic balance; tail asymmetry; cLogP, molecular weight, TPSA,
rotatable bonds; cross-role dependence; grouped two-sample AUC clustered by component family.

The question: **does the trained model move substantially closer to held-out lipid chemistry than
a context-independent role-marginal sampler under the same coarse program?**

## Outcome contract, fixed now

| outcome | claim |
|---|---|
| head architecture retained, tail statistics substantially recovered, beats the null on the panel, separability falls | FORGE learns a context-dependent ionizable-lipid design prior beyond the legality supplied by reaction constraints and role marginals |
| head good, tails still imperfect | FORGE learns some role-specific lipid organization, particularly ionizable-head structure, while tail-linkage fidelity remains limited |
| resembles the marginal null, or misses major head and tail chemistry | demote the learned-prior claim and proceed on the synthesis-grounded system story and prospective evidence |

## Stop rule

On a negative outcome: **no architecture hunt, and no patching the model against the held-out
panel.** The claim is demoted and the campaign continues. This is the last model experiment
before prospective results.

---

# Amendment 4: the claim boundary, frozen permanently

Recorded 2026-08-17. Framing work is closed. No further coupling audits, terminology searches,
origin-map work or representation ideation.

## The formulation

> **Semantic organization without generative factorization.**

> **The experiment organizes the molecular state; FORGE generates the molecule.**

The production model is a unified whole-molecule generator. The architecture permits cross-region
dependence, verified causally: perturbing one role region moves predictions in the others, with no
zero entries, so this is not three independent generators plus a core.

## What may not be claimed, permanently

The structural corpus is approximately Cartesian across precursor identities. Normalized mutual
information between component identities is 0.0234 to 0.0366, and the aldehyde-by-isocyanide
component grid is fully saturated at 2,625 of 2,625 pairs.

Therefore:

- **do not claim** that the structural model learns preferred head-tail combinations;
- **do not claim** biological cross-role compatibility from structural training;
- **do not claim** the corpus exhibits non-factorial pairing;
- **do not treat weak cross-region coupling as a defect to be fixed.** For a near-Cartesian
  structural library and an unconditional structural generator it is the scientifically correct
  result. Functional coupling belongs downstream, where the biological evidence lives.

## What the structural claim is

> FORGE learns the internal molecular organization of experimentally meaningful regions:
> ionizable-head chemistry, tail architecture, branching, unsaturation, linker chemistry and
> related structure, while sparse biological evidence governs functional selection among complete
> lipids.

## Three kinds of knowledge, acquired in three places

| knowledge | source | status |
|---|---|---|
| experimental semantics: what the regions mean | the Ugi experiment | known, not learned |
| structural: what realizations can inhabit them | 112,386-product corpus | learned broadly |
| functional: which complete lipids perform | 1,100 measured lipids | learned sparsely |

The generator is never asked to rediscover the first or to supply the third.

## Contrast to hold against the fragment and scaffold literature

> **Fragments encode recurring molecular substructure. FORGE encodes recurring scientific
> meaning**, and the molecular realization attached to that meaning stays open-ended.

A head-derived region spans many scaffolds; an aldehyde-derived region spans many linker, branch
and tail chemistries. The semantic label does not collapse the region to a vocabulary.

## Checkpoint discipline for the production-equivalent evaluation

The production duration contract itself records
`selected_development_checkpoint_step: 3000` for this architecture, against
`fixed_production_steps: 5100` on the full corpus under the per-record scaling rule. So the
train-fold checkpoint at step 3000 is the production-equivalent budget **as fixed by the frozen
production contract**, not a choice made against any evaluation panel. Using it is not checkpoint
shopping. Step 750 (`checkpoint_best`, calibration early stopping) is a different and shorter
budget, and results from it are labelled as such.

---

# Amendment 5: the product-only baseline parity contract, frozen before either model is trained

Recorded 2026-08-17, with the coverage splits sealed at
`results/phase1/forge_coverage_splits_v1/splits.json` and **no model of either arm trained**.

## The hypothesis

> Experimentally grounded semantic organization can improve whole-molecule generative learning
> without restricting the molecular vocabulary or factorizing generation into independent
> components.

## The one thing that must be matched, and usually is not

The treatment is **semantic organization of the molecular state**, not extra conditioning
information. FORGE receives a per-role program: three node counts, three junction budgets, three
cycle ranks, three attachment counts. A baseline given only a global atom count would be
conditioned on strictly less, and any FORGE advantage would be uninterpretable.

**Therefore the baseline receives the identical twelve-element program vector**, as a flat
conditioning input with no role semantics attached to any node. Same information, unstructured.

| field | FORGE | product-only baseline | matched? |
|---|---|---|---|
| training products | sealed coverage subset | identical subset | yes |
| program vector | 12 elements, per-role | **identical 12 elements, flat** | yes |
| atom and bond vocabularies | corpus vocabularies | identical | yes |
| backbone family, depth, hidden dim | as production | identical | yes |
| parameter count | 1.08M plus decoration heads | matched to within the removed embeddings, reported exactly | reported |
| flow and noising schedule | as production | identical | yes |
| objective form | cross-entropy | identical | yes |
| optimizer, learning rate, weight decay, clipping | as production | identical | yes |
| total updates, batch size | frozen duration rule | identical | yes |
| sampling budget and decoder | as production | identical | yes |
| post-generation chemistry qualification | qualified Ugi transform | identical | yes |
| **node role labels** | present | **absent** | **no: this is the treatment** |
| **within-role positions** | present | **absent** | **no: this is the treatment** |
| **serialization** | contiguous role blocks | **single flat traversal** | **no: this is the treatment** |
| **source distributions** | per role | **pooled** | **no: consequence of the treatment** |
| **loss weighting** | role-balanced | **pooled** | **no: consequence of the treatment** |
| **closure and exterior partition** | per role | **global** | **no: consequence of the treatment** |

The baseline must **not** receive component identities, role labels, role-specific serialization,
or any information derived from the test product. It must also not be made deliberately weak: the
parity table above is committed before training and any deviation is recorded as a deviation.

## Decomposition for the baseline

The baseline generates a product graph with no role partition, so its components are recovered by
the **repaired inverse**, which the earlier audit established returns exactly one valid component
tuple for every product in the corpus and in the enumerated library. The baseline is therefore not
handicapped on decomposition: it gets a correct factorization post hoc, which is precisely the
comparison a reviewer wants.

## Endpoints

Primary: the frozen lipid-native panel, resolved by region. Head architecture including the
one-handle-plus-tertiary-centre class; aldehyde-region linker class, placement, chain, branching
and unsaturation; isocyanide-region architecture; whole-lipid balance. **Admission and validity are
not primary.** Uncertainty grouped by component family, never by product row.

Primary stratum for the coverage curve: **S2 fixed combination test**, 4,033 products withheld
from every coverage level, so the test set is constant while training size varies. S1 is in-sample
and may never be reported as generalization.

## Staged execution and the stop-loss

**Pilot:** FORGE and baseline at 25% and 100% coverage, one preregistered seed each. Four runs.

**Escalation rule:** proceed to the full 10/25/50/100 by three-seed matrix only if, at 25%
coverage, FORGE shows a coherent advantage across chemically load-bearing panel measures without a
meaningful disadvantage at 100%. Coherent means several load-bearing measures moving together, not
one metric selected after the fact.

**If the pilot shows no meaningful advantage, the representation-efficiency branch stops.** The
recorded outcome becomes: semantic organization provides an experimental interface and a structured
representation rather than demonstrated statistical efficiency. That is an acceptable result, it is
written as such, and it does not reopen the generator.

## The curve, and the claim it would license

Lipid-native fidelity against fraction of the combinatorial library observed, for both arms, with
separate panels for head, aldehyde region, isocyanide region and whole lipid. The pattern that
would license a claim is FORGE's relative advantage *growing* as coverage falls:

> Experimental semantic organization converts repeated molecular structure across a combinatorial
> chemical library into reusable generative knowledge, improving statistical efficiency under
> incomplete product coverage.

Not to be claimed before it is seen. If the baseline wins, the semantic inductive-bias claim is
dropped rather than defended, and no architecture redesign is licensed.

---

# Amendment 6: the pilot's endpoint, staged

Recorded 2026-08-17, before either arm is trained. This **narrows** the pilot rather than widening
it, and it is recorded rather than substituted silently, because Amendment 5 named the lipid-native
panel as primary and this changes the instrument for the first stage.

## Why

The flat arm needs two separable things. Training it is moderate: a permutation in the record
projection, pooled source marginals, a pooled objective and the model switch. **Sampling** from it
is not: the constrained sampler locates precursor regions by slicing
`terminal["offspring"][batch, offset : offset + node_count]` from the program's per-role node
counts, and the tree decoder, closure placement and attachment-count selection all assume those
contiguous blocks. A flat arm would need a global tree decoder plus component recovery through the
repaired inverse. That is the largest remaining implementation in the project.

Building it before knowing whether the effect exists inverts the stop-loss.

## Stage 1: held-out likelihood, no sampler

Primary endpoint for the coverage curve becomes **teacher-forced denoising loss on the sealed
strata**, per arm, per coverage level:

- S2 fixed combination test, 4,033 products, constant across coverage levels;
- S3 held component identity;
- S4 held component family.

Protocol inherited from the stage 1 held-family evaluation: fixed nine-point t grid, identical t
values and noise seeds for both arms, batches ordered identically, weighted mean per metric with
undefined metrics skipped and counted.

This is the standard instrument for a data-efficiency curve and it needs no sampler work.

**One limitation carried forward explicitly.** The held-family evaluation established that
teacher-forced loss is not generative competence. It is used here only as a **paired arm-versus-arm
comparison on identical data**, never as an absolute claim about either arm's chemistry. No
lipid-native claim may rest on stage 1.

## Stage 2: the lipid-native panel, only on escalation

If stage 1 shows a coherent advantage for FORGE that grows as coverage falls, then the flat
sampler is built and Amendment 5's lipid-native panel runs as specified. If stage 1 shows no
coherent advantage, **the branch stops at stage 1** and the recorded outcome is that semantic
organization provides an experimental interface rather than demonstrated statistical efficiency.

The generator is not reopened either way.

## What stage 1 cannot conclude

- nothing about admission, validity or unique assembly, which it does not measure;
- nothing about head, linker or tail chemistry, which requires samples;
- no absolute statement about either arm, only the paired difference and how it moves with coverage.

---

# Amendment 7: the stage 2 gate, and the invariant that must hold before training

Recorded 2026-08-17, before either arm is trained.

## The stage 2 gate is a pattern, not a scalar

Stage 2 is not gated on one validation loss being lower. The gate is a coherent pattern across
paired teacher-forced quantities on the identical frozen S2, S3 and S4 examples, with identical t
grid, noise seeds and batch ordering:

- total held-out cross-entropy;
- atom-type prediction;
- parent-bond and closure-bond prediction;
- offspring and decoration heads;
- role-region-resolved loss, computed **for analysis only** from the ground-truth mapping and never
  supplied to the flat arm.

Define

```
Delta(c) = L_flat(c) - L_FORGE(c)
```

The gate is:

| condition | meaning |
|---|---|
| `Delta(0.25) > 0` | semantic organization helps at sparse coverage |
| **`Delta(0.25) > Delta(1.00)`** | **and it helps MORE as coverage falls** |
| both holding across the primary heads, not one | coherence rather than a selected metric |

The second condition is the hypothesis. A uniform advantage at every coverage level is a weaker
result and does not on its own license the inductive-bias claim; it would be reported as a level
effect rather than an efficiency effect.

If the pattern does not appear coherently, **stop at stage 1.** Do not spend 300 lines on a flat
sampler to rescue a representation claim.

## The invariant that must hold before either arm trains

The flat transformation is a permutation of whole preorder subtrees, so it must change the layout
and nothing else. Verified over a large sample of the train fold, not asserted:

| property | requirement |
|---|---|
| molecular identity | `canonical(G_original) == canonical(G_flat)` after inverting the layout |
| atom targets | identical as a multiset, and identical under the permutation |
| bond targets | identical under the permutation |
| closure targets | endpoints remapped consistently, bond states unchanged |
| decoration anchors | remapped consistently, states unchanged |
| program information | the twelve numbers unchanged |
| record count | no record dropped, added or reordered between arms |
| example identity | both arms see the same product ids in the same order |

Anything less, such as checking only that atom and edge counts agree, would permit the treatment to
be **semantic organization plus an altered molecular dataset**, which is not the experiment.

A failure here voids the arm. It is a correctness gate, not a result.

## What this buys

The experimental hierarchy becomes:

```
representation parity  ->  teacher-forced statistical efficiency  ->  only if positive: generative fidelity
```

And the standing limitation is unchanged: teacher-forced loss is a paired arm-versus-arm comparison
on identical data, and no lipid-native generative claim may rest on stage 1.

## Scope note, recorded so it is not mistaken for evidence

A flat free-running sampler is hard to build precisely because the experimental semantics run
through the state construction and the sampler rather than sitting in a cosmetic embedding. That is
an observation about the implementation, **not evidence of scientific value**, and it must never be
offered as such.

---

# Amendment 8: four gates before launch, and what the gates are not

Recorded 2026-08-17. All four must be green before the pilot runs.

| gate | requirement | status |
|---|---|---|
| 1. graph identity | the flat layout is a permutation of whole preorder subtrees and nothing else | **proven**, 20 properties over all 66,464 train-fold records, zero failures |
| 2. role non-recoverability | role must not be reconstructible from layout or model input | proven at the data level, 0 of 66,464 flat layouts match program blocks; the model-input test is specified |
| 3. information parity | both arms receive the identical twelve-element program vector; only its attachment to molecular regions differs | specified |
| 4. capacity parity | parameter counts matched, and no generic modelling capability removed | specified |

## Gate 4 is a fairness gate, not an ablation gate

The distinction matters and is easy to get wrong. `position_embedding` in the structured arm is
`nn.Embedding(maximum_component_atoms, hidden)` indexed by **within-role** position, so it is
experiment-specific semantic position and must go. But removing it outright would leave the flat arm
with **no positional signal at all**, which no ordinary whole-molecule generator would lack. A
reviewer would correctly say the baseline was crippled rather than ablated.

So the flat arm **keeps a global sequence position embedding** sized to `maximum_total_atoms`, and
the index must run 0 to node_count minus 1 across the whole molecule. A per-role counter that resets
at region boundaries would reintroduce role through position and voids the arm.

The rule generalizing this: **remove experiment-specific semantics, never generic modelling
capacity.** Anything the flat arm loses must be role information, not machinery.

## What these gates are not

The invariant checks and the layout statistics are **instrument validation**. They make the
experiment trustworthy. They are not evidence that semantic organization helps, they do not belong
in a results section, and "0 of 66,464 layouts retain role blocks" is a statement about the
apparatus rather than about FORGE.

Recorded because it would be easy to present a clean instrument as a finding.

---

# Amendment 9: role contiguity cannot be removed, and what that costs the experiment

Recorded 2026-08-17, during implementation, before any arm is trained. This is a **feasibility
finding about the experiment**, not a result about FORGE.

## The finding

Role contiguity in this representation is forced, not chosen. Measured:

| observation | value |
|---|---|
| role exteriors form exactly 3 contiguous runs after tree reordering | **1000 of 1000 records** |
| records where all three role sizes differ | 3,660 of 4,000 (0.915) |
| exterior size medians | amine 8, aldehyde 17, isocyanide 14 |
| role recoverability from position plus program, structured arm | **1.000** |
| role recoverability from position plus program, reordered arm | **0.690** against a 0.436 baseline |
| lift from content alone (atom, bond, offspring states) | +0.033 |

Two facts combine. Each role contributes essentially one tree to the exterior forest, and a valid
preorder forest keeps every tree's nodes contiguous. So **any** valid reordering permutes which
block comes first and cannot break contiguity. Content-based ordering was tried and moved
recoverability only from 0.695 to 0.690.

Underneath that: role regions in this chemistry **are** the branches hanging off the Ugi core. Any
connectivity-respecting canonical serialization makes them contiguous. Role-blocked layout is not an
encoding choice, it is what canonical serialization of this molecule class produces.

Position also carries far more role information than chemistry does, +0.254 lift against +0.033, so
the residue is positional rather than chemical inference. The scoping defence that it might be
chemistry does not hold.

## What this costs

The achievable treatment is **unlabelled regions**, not an undifferentiated graph:

| component of semantic organization | removable? |
|---|---|
| role label per node (`role_embedding`) | yes, removed |
| role-indexed program tables | yes, removed |
| role-balanced objective | yes |
| per-role source distributions | yes |
| program-to-position correspondence | partly; 24.5% of records still coincide |
| **region contiguity itself** | **no. chemistry plus serialization validity force it** |

So the experiment can ask *does labelling the regions and organizing training around them help,
given the regions are there anyway*. It cannot ask *does semantic organization beat an
undifferentiated molecular graph*, which is what the paper thesis states. **The achievable claim is
narrower than the thesis it was meant to test.**

## The options, recorded without preference resolved

**A. Run the narrower experiment.** Honest, and the coverage curve still measures whether role
labelling and role-indexed training earn their keep under sparse coverage. Must be described as
"role labelling and role-indexed training" throughout, never as semantic organization against an
undifferentiated graph, and every statement must carry the 0.690 partial-ablation figure.

**B. Drop the efficiency experiment.** The thesis stands as a representation choice, which is what it
is, supported by evidence already in hand: under identical constraints a role-marginal sampler
reaches full chemical admission and never once builds an ionizable head, while the trained arms build
them at 27 to 39 percent.

**C. Change the representation to break contiguity.** Rejected. That is a different model rather than
an ablation of this one, and it confounds the absence of semantics with a worse serialization.

## Standing rule either way

No claim may describe the flat arm as role-free or as an undifferentiated product generator. The
measured recoverability is 0.690 against 1.000 for the structured arm, and that pair travels with
every statement about it.

---

# Amendment 10: the semantic-ablation branch is closed as not identifiable

Recorded 2026-08-17. **Closed. Do not reopen without a naturally fair baseline.**

## What happened, stated precisely

The baseline-validity gate failed. **Stage 1 never ran, so there is no Stage 1 result**, and nothing
here is a negative finding about FORGE.

Sequence of record: the switch was implemented and 9 of 13 specification tests passed. The leakage
gate then measured role recoverability at 0.690 against a 0.436 baseline in the flat arm, where the
structured arm is 1.000. Decomposition showed the residue is positional (+0.254 lift) rather than
chemical (+0.033), and the cause is structural: each role contributes essentially one tree, a valid
preorder forest keeps every tree contiguous, so roles form exactly 3 contiguous runs in 1000 of 1000
records under any reordering. Role regions in this chemistry are the branches off the Ugi core.

**The intended semantic-free-but-otherwise-identical baseline does not exist cleanly under this
representation.** Removing the information completely requires changing the representation enough
that the comparison becomes a different model rather than a surgical ablation, which defeats its
purpose.

The verification machinery did the job it was built for: it stopped an ablation whose interpretation
would have been wrong.

## Consequences for the paper

**"Semantic organization improves sample efficiency" is motivation, not a claim.** It may appear as
the hypothesis that motivated the design. It may not be stated as something FORGE demonstrates, and
no result may be presented as evidence for it.

Struck from the whiteboard as an earned result:

> ~~Scientific structure should make generative learning easier.~~

Retained as motivation only.

**The earned statement:**

> FORGE preserves experimentally meaningful molecular organization without reducing generation to
> finite component identities or independent component models, and contextual learning recovers
> lipid-native molecular chemistry within that organization.

## What this does not undermine

That role remains partly inferable from topology **supports** the semantic framing rather than
weakening it. It says the semantics are not arbitrary tags painted onto a graph; they are tied to how
this molecular family is actually constructed. What cannot be isolated is the counterfactual "same
representation, no semantic organization at all."

Unaffected and load-bearing: experiment-structured whole-molecule generation; the learned
lipid-chemistry result against the role-marginal null, where that null reaches full admission and
never once builds an ionizable head; open structural discovery; S_pred strictly inside S_gen;
evidence-aware deployment; the synthesis and actionability funnel; prospective biology.

## Standing prohibitions

- do not build the flat sampler;
- do not attempt another serialization to manufacture a semantic-free baseline;
- do not describe the flat arm as role-free anywhere, at any time;
- do not present the coverage splits, the invariant checks or the leakage measurements as results.
  They are apparatus, and the apparatus is now retired with the branch.

The sealed coverage splits and the twenty-property invariant remain in the repository as a record of
a well-specified experiment that turned out not to be identifiable. That is the correct thing to
have on file.

---

# Amendment 11: reopened, reformulated, and bounded

Recorded 2026-08-17, before either arm is trained. This **supersedes Amendment 10's closure** with a
different and identifiable question. The flat-sampler branch stays closed.

## What was wrong with the closed version

It required the baseline to make experimental role **unrecoverable**. That demand was unnecessarily
strong and, as measured, unsatisfiable: role regions in this chemistry are the branches off the Ugi
core, so any valid preorder serialization keeps them contiguous.

## The identifiable question

Let `X` be the clean molecular state, `Z_t` the corrupted state at flow time `t`, `C` the generic
design information both arms receive, and `R` the experimentally grounded role map.

```
L_none(t) = H(X | Z_t, C)          optimal denoising without explicit semantics
L_role(t) = H(X | Z_t, C, R)       optimal denoising with them
Delta_sem(t) = L_none(t) - L_role(t)  ->  I(X ; R | Z_t, C)  >= 0
```

**The gate is `I(X;R|Z_t,C) > 0`, not `R` independent of `(Z_t,C)`.** The no-role arm is *allowed*
to infer role from topology, position and size wherever it can. That makes the comparison
conservative: any measured advantage survives despite the baseline having partial access.

This is why the 0.690 recoverability finding does not invalidate the experiment. It makes it harder.

## Three arms, one serialization

All three use the **flat projection already verified graph-identical over all 66,464 train-fold
records**, plus generic global position, the identical twelve-element program, pooled sources, pooled
loss, the same backbone and matched capacity. The only difference is the role channel:

| arm | role channel |
|---|---|
| `none` | absent |
| `flat_with_role` | the true role map `R` |
| `flat_random_role` | a three-valued label of identical dimensionality, permuted independently per example |

The negative control answers "does the model merely like an extra embedding". The result must be
`L_true < L_none ≈ L_random`, or the finding is about capacity rather than scientific alignment.

**No sampler is built. No FORGE redesign. The prospective pipeline is untouched.**

## The corruption-time convention, stated from measurement so prose cannot reverse it

Measured on a fixed heldout subset: total denoising loss **falls monotonically in `t`**, from 5.42 at
`t = 0.02` to 2.76 at `t = 0.98`. So in this implementation **low `t` is heavily corrupted and high
`t` is nearly clean.**

The hypothesis is therefore that `Delta_sem(t)` is **larger at LOW `t`**: as molecular information is
destroyed, the chemistry stops revealing which region is which, while the experimental semantics
remain known. Any prose asserting the reverse is wrong.

Grid: `t ∈ {0.1, 0.25, 0.5, 0.75, 0.9}`, identical values, noise seeds and batch ordering across arms.

## Endpoints

`Delta_sem(t)` per head: atom identity, parent bond, offspring, closure, decorations, and the
aggregate. Errors additionally broken down by true region for analysis only; the region labels are
used to *score* the no-role arm, never supplied to it.

## Coverage axis and the gate

Run at the sealed 25% and 100% coverage first, one preregistered seed each.

```
Delta_sem(c) = L_none(c) - L_role(c)
```

| condition | reading |
|---|---|
| `Delta_sem(0.25) > 0` | semantics carry information beyond the corrupted state |
| `Delta_sem(0.25) > Delta_sem(1.00)` | and that information matters more as assembled-product supervision thins |
| `L_true < L_none ≈ L_random` | the alignment matters, not the parameter |
| coherent across heads | not a selected metric |

Escalate to 10/25/50/100 with three seeds only if all four hold. Otherwise stop and claim nothing.

## The claim ladder, each rung with its own evidence

```
information            Delta_sem(t) > 0 with the random-role control flat
                       -> "experimentally grounded semantics provide predictive information for
                          molecular denoising beyond what is recoverable from the corrupted state"

learning efficiency    Delta_sem(0.25) > Delta_sem(1.00)
                       -> "the value of that information increases as combinatorial supervision
                          becomes sparse"

generative quality     only if a strong conventional whole-product generator also loses on
                       lipid-native fidelity
                       -> "experimental semantic organization is an effective inductive bias for
                          whole-molecule generation"
```

No rung may be claimed on another rung's evidence. Amendment 10's demotion stands until the first
rung is earned: until then, "scientific structure makes generative learning easier" remains
motivation.

## What stays closed

The semantic-free serialization, the flat sampler, and any further attempt to make role
unrecoverable. Those were the wrong question.

---

# Amendment 12: final corrections, then frozen

Recorded 2026-08-17. Amendment 11 stands as corrected below. **This is the last change before the
run.**

## 1. Notation: coverage and flow time are separate

Coverage fraction `alpha` in {0.25, 1.00}; flow time `t` in {0.1, 0.25, 0.5, 0.75, 0.9}.

```
Delta_sem(t; alpha) = L_none(t; alpha) - L_true(t; alpha)
```

Two distinct hypotheses, and they must not be conflated:

| hypothesis | statement |
|---|---|
| semantic information | `Delta_sem(t; alpha) > 0` |
| data efficiency | `mean_t Delta_sem(t; 0.25) > mean_t Delta_sem(t; 1.00)` |

The second is a `t`-averaged quantity over the frozen grid, written `Delta_bar_sem(alpha)`.

## 2. Corruption convention, from the path definition

The forward path is `q_t = t*delta_x + (1-t)*p_0`, so `t = 1` is clean and `t = 0` is the source.
**Low `t` is more corrupted.** This agrees with measurement: total loss 5.42 at `t = 0.02` against
2.76 at `t = 0.98`.

The compelling secondary pattern is `Delta_sem` **larger at lower `t`**: semantics matter more once
local molecular evidence has been destroyed.

## 3. Capacity parity is exact, not approximate

The no-role arm **keeps the whole role-embedding machinery** and feeds a constant null tag to every
node. Both flat arms therefore carry `nn.Embedding(len(ROLE_NAMES) + 1, hidden)` and **identical
parameter counts**; the 576-parameter gap disappears rather than being reported.

Stronger still: the collated batch is **identical across arms**. Only the model's use of
`role_states` differs. That is as surgical as this comparison can be made.

## 4. The misaligned control replaces the relabelling control

Per-example relabelling of intact regions is **rejected**. The program vector carries the ordered
per-role sizes and those sizes are distinct in 91.5% of molecules, so the model could reverse-engineer
which randomly named block is really which. The control would secretly recover true semantics.

The correct control is a **misaligned map**: within each molecule, permute role labels *across nodes*
while preserving the exact count of each label, fixed deterministically per example. Dimensionality
and marginal frequencies are preserved; scientific alignment is destroyed.

Readings, and neither is required in advance:

| pattern | reading |
|---|---|
| `L_true < L_none ≈ L_misaligned` | alignment carries the effect, not the parameter |
| `L_true < L_none < L_misaligned` | also interpretable: wrong semantics actively hurt, right ones help |

**Do not oversell `none ≈ misaligned` as a requirement.**

## 5. What the loss gap is, stated precisely

At the Bayes optimum, for a categorical head `h`:

```
L*_none,h - L*_true,h = I(X_h ; R | Z_t, C)
```

and for the weighted multi-head objective, the optimal gap is the corresponding weighted sum of
conditional mutual informations. **Finite trained networks give an empirical semantic-utility gap,
not an exact mutual information.**

Manuscript wording, to be used verbatim:

> At the Bayes optimum, the reduction in categorical denoising loss equals the conditional
> information supplied by the semantic role map. We therefore measure the practical value of this
> information through the paired denoising-loss gap.

## 6. The pilot is four runs

`none` and `true` at `alpha` in {0.25, 1.00}, one preregistered seed each. The misaligned arm is a
**cheap follow-up only if the primary signal is positive**, which keeps the pilot at four runs and
roughly $0.40.

## Frozen

No sampler. No model redesign. No reopening anything else. Implementation follows; nothing above
changes after it starts.

---

# Amendment 13: the pilot compares step 3000, not checkpoint_best

Recorded 2026-08-17 after the smoke arm and **before the remaining three arms ran or any evaluation
was performed.**

The pilot configs inherit `calibration_early_stopping` from the challenger config they were derived
from, so each run writes a `checkpoint_best.pt` chosen by its own calibration loss. In the smoke arm
that landed at **step 250** while the run completed 3,000.

**The paired comparison uses `checkpoint_step_3000.pt` for every arm.** Both arms are compared at the
same fixed budget. Using `checkpoint_best.pt` would let each arm select its own step by its own
calibration loss, which is per-arm checkpoint shopping and would break the pairing that the whole
design rests on.

This is the same error class as using `checkpoint_best` when the frozen production contract named
step 3000, which happened earlier today. Recording it as a rule rather than relying on remembering.

`checkpoint_best.pt` and `checkpoint_latest.pt` remain on disk as provenance. Neither is used for
`Delta_sem`.

---

# Amendment 14: the mechanistic association, declared before it is computed

Recorded 2026-08-17, **before any association between role differentiation and semantic utility has
been examined.**

## The question

> Do experimentally defined regions whose molecular chemistry differs more strongly receive more
> benefit from the semantic map?

## The measure, fixed now

For each frozen prediction head `h` with a per-node or role-attributable target, the role
differentiation is the **Jensen-Shannon divergence among the three role-conditional empirical target
distributions on the train fold**:

```
D_h = JSD( p_h(. | amine), p_h(. | aldehyde), p_h(. | isocyanide) )
```

with the uniform three-way mixture as the reference and base-2 logarithms, so `D_h` lies in [0, 1].

Head-to-target mapping, fixed here:

| head | target | role attribution |
|---|---|---|
| `atom_ce` | atom state | the node's own role |
| `parent_bond_ce` | parent bond state | the node's own role |
| `offspring_ce` | child count | the node's own role |
| `closure_bond_ce` | closure bond state | the role of the closure's endpoints, which are same-role by construction |
| `decoration_atom_ce` | decoration atom state | the role of the anchor node |
| `decoration_bond_ce` | decoration bond state | the role of the anchor node |

`decoration_anchor_ce` is excluded: its target is a position, not a categorical chemical state, so a
role-conditional distribution over it is not comparable to the others.

**No alternative divergence will be computed.** If the association is weak, that is the result.

## The comparison

`D_h` against `Delta_bar_sem,h` averaged over the frozen t grid and over the replication seeds, and
separately against `Delta_sem,h(t = 0.10)`, the most corrupted grid point.

Reported: Spearman rank correlation, a permutation p-value over head labels, and a labelled scatter.

## The power limitation, stated in advance

There are **six heads**. A rank correlation on six points is very low-powered: only a near-perfect
ordering can reach conventional significance. This analysis is therefore **descriptive and
one-shot**, and will be reported as such whatever it shows. It may not be featured as a primary
result, and a weak or inconsistent association will be recorded rather than re-measured with another
metric.

## Interpretation if positive

> Semantic information is most useful for molecular variables whose distributions genuinely differ
> across the experimentally defined regions.

That sentence is available only if the association is positive on the declared measure at the
declared heads.

---

# Amendment 15: main-text metrics are chosen by scientific coverage, not by effect size

Recorded 2026-08-17, **before the manuscript ledger was assembled and before any descriptor was
ranked by how well it separates the arms.**

## The error this forecloses

An earlier plan was to pick the main-text descriptors as "the ones the panel already discriminates
on". That is post-hoc metric selection. It looks innocuous, it would improve every number in the
table, and a reviewer would be right to call it out.

## The rule

**Main-text metrics are selected by prespecified scientific coverage of the chemically meaningful
categories already defined in the frozen lipid-native panel, never by observed effect magnitude.**

One or two representative measures per category, fixed here:

| category | main-text measure |
|---|---|
| ionizable head | rate of the one-handle-plus-tertiary-centre motif, and the rate of two or more Ugi-reactive sites |
| linker | ester prevalence in the aldehyde-derived region |
| tail architecture | one hydrophobic-architecture measure: branch count in the aldehyde-derived region |
| whole lipid | tail asymmetry, and the head-to-hydrophobic ratio |
| aggregate | grouped two-sample AUC, clustered by component family |

**The complete 38-descriptor panel goes in the appendix**, so the selection above can be checked
against everything measured. That is what makes the main-text table verifiable rather than curated.

## Standing prohibition

No descriptor may be added to, removed from, or swapped within the main text on the basis of what it
shows. If a listed measure is unflattering, it is reported. If an unlisted measure is striking, it
stays in the appendix.

---

# Amendment 16: terminology, settled

Recorded 2026-08-17.

| level | term |
|---|---|
| the method | **chemistry-informed whole-molecule generation** |
| the principle | **semantic organization without generative factorization** |
| what the ablation manipulates | **chemically aligned** semantics, against **absent** and **misaligned** |

"Chemistry-informed" is the framework term because chemistry organizes more than one role map: region
meaning, molecular coordinates, admissible structure, assembly semantics and the experimental
interface. "Chemically aligned" is reserved for the specific channel the controlled experiment
manipulates, so the two levels stay distinct.

Working title: **FORGE: Chemistry-Informed Whole-Molecule Generation for Ionizable-Lipid Discovery.**

Methods wording: *FORGE uses chemistry-grounded semantic organization without factorizing
whole-molecule generation.*

Results wording: *Correctly aligned chemical semantics reduce denoising ambiguity; misaligned
semantics do not.*
