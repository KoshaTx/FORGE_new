# FORGE: pending changes for the manuscript writer

Running ledger of decisions and new results that are **not yet reflected in either manuscript
draft**. Anyone writing the paper should read this before touching
`manuscript/FORGE_ICLR2027_submission.md` or `manuscript/FORGE_Nature_Biotechnology_working_draft.md`.

Companion documents:
- `docs/FORGE_IMPLEMENTATION_REFERENCE.md` — source of truth for every number
- `docs/FORGE_ICLR_EXPERIMENT_CONTRACT.md` — settled framing, contribution order, experiment priorities

Last updated 2026-08-15.

---

## 1. CORRECTION: the novelty denominator is wrong in both drafts

**Both drafts report 89.0% novelty against the 66,464-product train fold. That denominator is
wrong for these designs.**

The production generator was refit on all three folds (`training_folds: [train, calibration,
heldout]`, `production_training_records: 112386`). It had seen all 112,386 corpus products, not
66,464.

| Denominator | Absent | Fraction |
|---|---|---|
| all 112,386 products the model trained on | 21,960 / 26,235 | **83.7%** |
| train fold only, 66,464 | 23,342 / 26,235 | 89.0% |

Affected locations:
- `FORGE_ICLR2027_submission.md:477` and `:1035`
- `FORGE_Nature_Biotechnology_working_draft.md:157` and `:1213`

The supplementary table at both `:1035`/`:1213` locations carries the gloss *"the model never saw
this molecule during fitting"*, which is the specific claim that fails.

**Fix:** either report 83.7% for the production model, or produce a separate evaluation run from
a train-fold-only checkpoint and report 89.0% for that model, keeping the two runs and their
claims strictly separate.

**The 86.3% against the 12,276-product enumerated library is unaffected** — that reference set is
external to training. It is also the figure the central argument rests on, so the damage is
contained.

---

## 2. NEW RESULT: diversity under support-aware tilting

This was the `[PENDING MEASUREMENT]` in the ICLR draft §6.5. It is now measured, at the matched
16,384-draw budget, over admitted designs in each arm.

**The reallocation does concentrate the search, and unevenly.**

| Quantity | broad_prior | support_enriched | ratio |
|---|---|---|---|
| admitted rows | 15,055 | 15,125 | 1.005 |
| distinct products | 14,226 | 13,248 | 0.931 |
| unique fraction | 0.945 | 0.876 | 0.927 |
| **effective amine heads** (exp Shannon) | **553.2** | **223.3** | **0.404** |
| **effective aldehyde tails** | **219.7** | **220.2** | **1.002** |
| effective isocyanide tails | 37.8 | 26.0 | 0.687 |
| effective morphology programs | 5,520.5 | 3,136.9 | 0.568 |
| novelty vs 112,386 corpus | 83.2% | 79.0% | — |
| novelty vs 66,464 train fold | 88.3% | 85.8% | — |
| novelty vs 12,276 enumerated library | 88.9% | 85.0% | — |

**How to report this honestly.** Proposition 4.2 (no architecture receives zero mass) holds, and
is not contradicted. But the finite-sample consequence is real: **effective amine-head diversity
falls by 60%** and effective morphology-program diversity by 43%.

The mechanism is coherent and worth stating. `S_pred` requires all four views to pass, and the
amine head is the role where the measured library is densest — 20 measured heads against 264 in
the corpus registry. Tilting toward support therefore pulls hardest on the head.

**The aldehyde tail is untouched (1.002×).** That matters, because the paper's headline component
result — 11 measured aldehyde identities expanding to 163 — lives entirely in that role and
survives tilting intact. Whole-product novelty drops only about 4 percentage points on every
denominator.

Suggested framing: *tilting buys 2.23× rankable yield at the cost of amine-head diversity, while
leaving aldehyde-tail diversity and whole-product novelty essentially unchanged.* Do **not**
claim the reallocation is diversity-neutral. It is not.

Script: `scripts/phase1_measure_tilting_diversity_v1.py`, reading
`results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz`,
`results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz`,
`results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz`.

---

## 3. NEW RESULT: pooled versus conditional conformal calibration

From `results/phase1/ugi_interpolative_conformal_v1/result.json`, target coverage 0.90, 1,218
evaluated interpolative rows.

| Scheme | Eligible folds | Pooled coverage | Pooled \|gap\| | Mean per-fold \|gap\| | Worst fold | Gate |
|---|---|---|---|---|---|---|
| held_aldehyde_5fold | 4 | 0.832 | 0.068 | 0.159 | **0.536** | fail |
| held_aldehyde_isocyanide_pair_5fold | 3 | 0.984 | 0.084 | 0.083 | 0.976 | pass |
| held_head_5fold | 4 | 0.910 | 0.010 | 0.059 | 0.827 | pass |
| held_head_aldehyde_pair_5fold | 0 | — | — | — | — | fail |
| held_head_isocyanide_pair_5fold | 0 | — | — | — | — | fail |
| held_isocyanide_5fold | 2 | 0.833 | 0.067 | 0.071 | 0.767 | fail |

**Mean pooled |coverage gap| 0.057 versus mean per-fold 0.093. Only 2 of 6 schemes pass their
gate, and 2 schemes have zero eligible folds.** The worst single fold reaches 0.536 coverage
against a 0.90 target with R² = −0.52.

**Reading:** conditional calibration is unstable at these sample sizes, and pooling is better
calibrated on average. This is evidence to **demote** stratification from a headline contribution
to a reported analysis, or to present pooled as primary with stratified as a secondary view.

Two caveats to carry with the number:
- This artifact evaluates conditional calibration by *distribution bin* within held-component
  schemes. That is related to, but not identical with, the authority-tier stratification used in
  production ranking. A direct pooled-vs-tier comparison on the production strata has not been run.
- The artifact's own adjudication records that *"conditional calibration was specified after
  inspection of the v3 outer-test diagnostic and therefore requires a separate versioned
  authorization review."* It is post-hoc specified and must be labelled as such.

---

## 3b. RESOLVED: the frozen inverse was buggy; Ugi products ARE identifiable

This section replaced two earlier, wrong versions. The conclusion is now settled by a repaired
inverse run to completion on both the generated support and the enumerated reference.

### The finding

Repairing the reverse template and enumerating **every** forward-reconstructing precursor tuple:

| Valid tuples per product | Generated (26,235) | Enumerated reference (12,276) |
|---|---|---|
| 0 | **0** | **0** |
| **1** | **26,235 (100%)** | **12,276 (100%)** |
| >1 | **0** | **0** |

Every product, generated and enumerated, has **exactly one** valid factorization, and for every
generated product that unique tuple **is** the one the model emitted — 26,235 of 26,235,
including all 2,263 secondary-amine-head designs that the frozen inverse could not touch.

**Ugi-3 products are identifiable from the product graph.** Not merely "a valid decomposition
exists" — uniquely recoverable.

### What the earlier 90.8% actually measured

A bug. The frozen inverse is built by string-reversing the forward SMARTS, so the amine
*reactant* pattern `[NX3;H2,H1:1]` becomes a *product* template. RDKit cannot honour a
disjunctive hydrogen count when writing an atom (it warns `multiple H count specifications`) and
emits `[NH2]`. Correct for a primary amine; a valence-4 nitrogen for a secondary one:

```
[NH2](CC)CC          <- diethylamine, emitted with 4 bonds, fails sanitisation
[NH2]1CCNC(O)(CC)C1  <- same defect, cyclic secondary amine
```

Hence `invalid_reactant` on 600 of 600 probed secondary-head products. The repair clears the
template-imposed hydrogen count and lets RDKit infer it. Frozen code was not modified;
`scripts/phase1_repaired_inverse_audit_v1.py` reimplements the inversion alongside it.

**Do not report 90.76% as a scientific result anywhere.** It describes a broken implementation.

### The one honest observation that survives

The frozen inverse was **qualified and validated on the 12,276-member enumerated library, in
which all 22 amine heads are primary** — as are all 20 measured heads. It therefore had no
secondary-amine chemistry in its validation set, and it failed silently on 100% of the generated
designs that had it.

That is the defensible, modest argument for carrying the origin map, and we have first-hand
evidence for it because it happened to us:

> Post-hoc inversion is solvable and unique for this reaction, but its correctness is an
> additional implementation assumption, validated on the enumerated library and not guaranteed
> to transfer to the generated support. Carrying the factorization in the sample makes that
> class of silent failure impossible by construction.

Modest. True. Do not inflate it.

### Claims that are now dead

- ~~product graphs are generally ambiguous under Ugi decomposition~~ — zero ambiguous cases
- ~~product-only methods cannot recover secondary-amine precursors~~ — a repaired inverse recovers 100%
- ~~the origin variable is necessary to reach secondary-amine chemistry~~ — false
- ~~90.8% recovery demonstrates an inherent FORGE advantage~~ — it measures our bug
- ~~routing would otherwise need a learned retrosynthesis model~~ — a deterministic inverse suffices

A logical correction to my own earlier wording: admission proves **existence** of a valid tuple,
not **identifiability**. Identifiability is now established separately, by exhaustive enumeration.

### What contribution 2 becomes

Not factorization capability. A structured-output and interface contribution:

> FORGE jointly generates a complete lipid structure and its precursor-origin assignment, making
> the final-assembly factorization a native model output, verified by the forward reaction and
> used consistently for component-level analysis and routing.

with the counterweight nearby:

> We do not claim Ugi products cannot be decomposed after generation; a reaction-specific
> inverse recovers the factorization uniquely when correctly specified.

**How prominent the origin channel should be now depends entirely on the training ablation.**
If explicit `o` improves validity, fidelity or efficiency, it is a modelling contribution. If
generation is unchanged, it is a structured-output design with no claimed quality gain, and it
should be demoted from the contribution list to the method.

### Title consequence

Drop "reaction-resolved" from the title until the ablation justifies it:
**FORGE: Discrete Flow Matching for De Novo Ionizable Lipid Design.**

### Residual

Three `forward_mismatch` rejections across 38,511 products — candidate tuples correctly rejected
for not rebuilding the target. No product was left without a tuple.

Scripts: `scripts/phase1_repaired_inverse_audit_v1.py` (repaired, exhaustive, with
`--reference` regression), `scripts/phase1_posthoc_decomposition_recovery_v1.py`,
`scripts/phase1_posthoc_recovery_by_amine_class_v1.py`,
`scripts/phase1_posthoc_failure_mode_v1.py` (all three now historical, documenting the bug).


---

## 3c. NEGATIVE RESULT: reaction-factorized batch acquisition does not clear its gate

A gated extension, developed after the generator and the 40-candidate panel were already
frozen. The gate was frozen before any result existed
(`configs/bio/phase1_forge_active_design_v1.json`) and it failed. The workstream is stopped.
Library 1 is untouched. Nothing from this belongs in the main text.

### The idea

Use the product-precursor factorization to choose experiments rather than only to explain
molecules. A measurement on one lipid informs every design sharing its head, its aldehyde tail
or its isocyanide tail, so a role-decomposed Bayesian surrogate should be able to pick a batch
that both finds actives and teaches the ranker. Motivated by the diagnosis that generation
(92.1% admission) and routing (97.1% resolved) are not the bottleneck, but the activity model
is: a component-additive ridge matches the production D-MPNN on held aldehyde families.

### The gate, as frozen

Beat **both** `top_lower_bound` and `component_diversity` on held-family Spearman, by at least
0.03, on at least 3 of 5 seeds.

### The result

| seed | reaction_factorized | top_lower_bound | component_diversity | beat both |
|---|---|---|---|---|
| 20260815 | 0.226 | 0.178 | 0.361 | no |
| 20260816 | 0.380 | 0.175 | 0.409 | no |
| 20260817 | 0.410 | −0.025 | 0.392 | no |
| 20260818 | 0.378 | 0.054 | 0.425 | no |
| 20260819 | 0.384 | 0.100 | 0.421 | no |

**0 of 5 seeds. Gate failed.**

Aggregate over five seeds, five rounds, six arms:

| arm | mean held-family Spearman | mean top-decile hits |
|---|---|---|
| uncertainty | 0.405 | 38.8 |
| component_diversity | 0.401 | 36.2 |
| **reaction_factorized** | **0.356** | **47.8** |
| random | 0.342 | 36.2 |
| top_mean | 0.296 | 52.4 |
| top_lower_bound | 0.096 | 43.6 |

### What it means

There is a learning-versus-discovery frontier and the method sits on it rather than escaping
it. It beats score ranking at learning by a wide margin (0.356 against 0.096) and beats every
diversity baseline at hit discovery (47.8 against 36.2). It does not beat plain
component-diversity or uncertainty sampling at learning, and both of those are free.

The honest reading: with 1,100 measurements and strong component-family shift, broad coverage
teaches this surrogate more than posterior-guided acquisition does. A sophisticated acquisition
function cannot manufacture trustworthy epistemic uncertainty when the feature map and
posterior model do not capture the relevant biological variation. That is consistent with, and
reinforces, the reason FORGE does not guide the molecular flow with predicted potency.

**Do not present the frontier as a win.** The gate asked the method to beat both rivals on the
primary metric and it did not. No rescoring, no new scalarization, no second library.

### Three defects found during the audit, two of which would have produced a confident wrong answer

1. **The activity floor was applied only inside the method.** The five baselines never saw it,
   so the method was confined to the top half of the pool by predicted activity while random
   sampled freely. On a learning metric that is a handicap. Fixed by moving the floor to a
   standalone `apply_activity_floor` called once per round in the harness, with every arm
   receiving the identical filtered pool. The artifact now records pool sizes before and after
   the floor so the audit is in the data.

2. **The gate was ruling on NaN.** The config names the criterion in prose
   ("held_family_spearman after the final round"); the evaluator did a dict lookup with that
   string against a key of `held_family_spearman`, so `.get()` returned its default on every
   seed. The verdict was right by accident. The evaluator now resolves the prose to a recorded
   metric and raises if it cannot, and refuses to rule on non-finite values.

3. **The benchmark was not reproducible from its own seeds.** Arm RNGs were seeded with
   `hash(arm)`, and string hashing is salted per process, so the stochastic baselines drew a
   different stream on every run. `component_diversity` read 0.280 on one run and 0.370 on the
   next from identical code and identical frozen seeds. Replaced with `zlib.crc32`.

None of the frozen knobs were touched: `value_weight`, `activity_floor_quantile`, role weights,
prior and noise variance, feature map, target population and gate all stand as written.

### Known limitation of the benchmark itself

The synthesis-cost dimension is untested. The 1,100 measured lipids carry no route dossier, so
every record costs one unit and the budget binds only through batch size. Cost normalisation is
implemented and unit-tested but has not been exercised on real preparation counts.

Artifact: `results/phase1/forge_active_design_benchmark_v1/result.json`. Branch:
`agent/active-design-20260815`.

### Benchmark integrity, closed

- **Two independent runs reproduce byte-identically**: zero differing fields across every seed,
  arm, round and metric, same verdict. Untrue before the `zlib.crc32` fix.
- **Independent verifier** (`scripts/phase1_verify_active_design_benchmark_v1.py`) imports
  neither the benchmark nor its gate evaluator. It replays 150 rounds from the recorded
  selections, recomputes Spearman, cumulative hits and the gate, and checks four properties the
  harness could not check about itself: no duplicate picks, no candidate reselected across
  rounds, batch sizes as recorded, and an identical pool offered to all arms in all 25
  seed-rounds. It agrees on every check.

### The mixture control, and why the branch stops anyway

The dominance gate compared the method against the endpoints of a trade-off, not the line
joining them. `configs/bio/phase1_forge_mixture_frontier_v1.json` was frozen before results, on
five seeds the first study never used, with every parameter of the method unchanged: ten simple
mixtures allocating batch positions between top-mean and either component diversity or
uncertainty, at exploit fractions 0, 0.25, 0.5, 0.75, 1.

**The frozen rule passed on the mean** — no mixture achieved both a higher Spearman and more
hits than `reaction_factorized` at 0.151 / 55.6. **The result is nevertheless noise.**

| check | value |
|---|---|
| seeds on which the method is dominated | **2 of 5** (20260902, 20260904) |
| paired margin vs `mix_unc_0.75` | **+0.014 ± 0.044** |
| paired margin vs `mix_unc_0.25` | **−0.002 ± 0.045** |
| paired margin vs `mix_div_0.5` | **−0.023 ± 0.062** |
| method's own Spearman across seeds | −0.062 to 0.294, sd **0.135** |
| total spread across all ten mixture arms | **0.155** |
| rank on learning / on hits | 5th / 4th of 11 |
| seeds needed to resolve the one positive gap at 80% power | **~65** |

Every standard deviation is roughly three times its mean; the method is behind on two of the
three nearest mixtures; and its single-seed variance is 87% of the entire between-arm spread.
The mean-level non-domination is an averaging artifact, not a frontier position.

**Decision: the acquisition workstream stops.** No confirmatory study, no Library 2, no
constrained reformulation, no main-text space. Chasing +0.014 across 65 seeds would be looking
for a result rather than testing for one. Nothing about the method was changed at any point.

Artifact: `results/phase1/forge_mixture_frontier_v1/result.json`.

### What the branch was worth

It reinforces a design decision the paper already makes. Broad and diverse acquisition beats
aggressive score selection for learning in this regime, which is the same reason FORGE does not
guide the molecular flow with a predictor that a component-additive ridge matches. That belongs
in the discussion as reasoning, not in the main text as evidence.


---

## 3d. RESOLVED + NEW RESULT: the synthesis grounding is role-specific, and measurable

### The aldehyde question, settled from the frozen definitions

Roughly half of generated aldehyde components lack the ester linkage the AGILE Tail A
preparation requires. The question was whether that is intended support or a mismatch between
the declared chemistry and what the generator can emit. It is **intended support**, and the
error is in our terminology, not in the model.

| source | ester+aldehyde | aldehyde only |
|---|---|---|
| qualified reaction, required handle | `[CX3H1]=[OX1]`, no ester term, no forbidden patterns | |
| measured library, 11 tails | 11 | 0 |
| enumerated library, 62 blocks | 56 | **6** |
| **training corpus, 107 components** | **56** | **51** |

The declared family always admitted non-ester aldehydes; the corpus is 47.7% of them. And the
generator reproduces that composition almost exactly:

| | ester-linked share of distinct aldehyde components |
|---|---|
| training corpus (declared family) | 56/107 = **0.523** |
| generated population | 1015/2044 = **0.497** |

A 0.027 difference. **The generator is not drifting outside its declared support.** What must
change is the prose: the role is named `oxoester_aldehyde_body_tail` and the drafts call these
"ester-linked aldehyde tails", but the family is broader. Correct wording: *the aldehyde role
admits both ester-linked and non-ester components; the current preparation registry supplies a
constructive map for the ester-linked subclass.*

### The amine head is not a gap

Earlier framing here called the absence of an upstream preparation map for amine heads a
limitation. It is not. **Direct procurement is the intended realization operator for that
role** in this workflow: buy the ionizable heads, prepare the specialised tails, run the common
Ugi assembly. All 17 distinct heads in the frozen 40-candidate panel have supplier records.

The realization program is deliberately asymmetric:

| role | intended realization | quantity worth measuring |
|---|---|---|
| amine head | direct procurement | purchasable among **fully assessed** heads |
| aldehyde tail | purchase, or acid + α,ω-diol → hydroxy ester → aldehyde | preparation-schema applicability, forward-verified |
| isocyanide tail | purchase, or primary amine → formamide → isocyanide | preparation-schema applicability, forward-verified |
| final lipid | Ugi-3 assembly | exact forward reconstruction |

### Population-level route-schema coverage

Over every unique component in the admitted population, not the prediction-supported slice:

| role | stratum | n | schema applies |
|---|---|---|---|
| **isocyanide** | all | 876 | **1.000** |
| | **outside enumeration** | **868** | **1.000** |
| **aldehyde** | all | 2,044 | 0.491 |
| | inside enumeration | 45 | 0.889 |
| | **outside enumeration** | **1,999** | **0.482** |

**Every generated isocyanide, including all 868 outside the enumerated set, reduces to a
forward-verified primary amine.** Route-schema coverage for that role does not track proximity
to the block set. For the aldehyde role it partly does, and the gap is entirely the ester
subclass question above: 1,029 of 1,041 failures are "no ester linkage", only 12 have an ester
and resist reconstruction.

### The procurement column is not reportable

The snapshot holds 477 components with a vendor against 7,700 audited, so it reaches at most
6.2%. Absence from it means *not found or never searched*, not *unpurchasable*. No
purchasability figure from this artifact is a route-failure rate, and none should reach the
paper. Procurement has three states and this snapshot distinguishes only two.

### Suggested main-text wording

> FORGE's component roles induce constructive preparation maps beyond the final Ugi assembly.
> All 876 distinct generated isocyanides, including 868 absent from the original isocyanide
> set, were reduced to corresponding primary amines through a forward-verified
> formylation–dehydration route. The documented esterification–oxidation route applied to 49.1%
> of generated aldehyde components and 48.2% of those outside the original aldehyde set, with
> nearly all remaining components lacking the ester linkage that preparation requires. Amine
> heads are sourced directly from commercial inventories in this workflow. These maps provide
> role-specific route skeletons rather than guarantees of procurement, yield or experimental
> success.

### Three bugs, each now a regression test

The audit read 0% for both tail roles until three defects were fixed: multi-product reaction
outcomes were flattened so acid/diol pairs were destroyed; the recovered primary amine came
back as `[NH3+]`; and the fix for that stripped the charges off legitimate `[N+]#[C-]`
isocyanides, breaking the forward check. `tests/test_route_grounding_audit.py` covers all three
plus eight known packet routes and the non-ester scope guard.

Artifact: `results/phase1/forge_route_grounding_audit_v1/result.json`.

### Still open

A fully assessed stratified route-completeness study with real procurement search per leaf,
reporting Pr(route-complete | outside enumeration) against Pr(route-complete | inside
enumeration). Secondary to the origin ablation and the wet lab.


---

## 3e. RESULT: structural reach exceeds biological evidence — FROZEN

**Primary population: the 26,235 distinct admitted products of the frozen production run.** The
3,652-product branch-exploration source is a separate sampling arm; unioning it gives 29,850
with 37 products in common and changes no conclusion, and belongs in an appendix note.

### The aldehyde population is five chemotypes, and the corpus has two

Exclusive labels under a precedence fixed before counting; ester wins over any co-occurring
motif. Multi-label motif prevalence is recorded separately in the artifact.

| chemotype | corpus components | generated components | broad arm | enriched arm | combined | prediction-supported |
|---|---|---|---|---|---|---|
| ester_linked | **56** | 1,015 | 9,173 | 8,520 | 17,693 | **2,590** |
| ketone_and_ether | **0** | 54 | 49 | 41 | 90 | 0 |
| ketone_containing | **0** | 295 | 799 | 839 | 1,638 | 0 |
| ether_linked | **0** | 177 | 236 | 179 | 415 | 0 |
| unlinked_chain_aldehyde | **51** | 503 | 3,969 | 2,430 | 6,399 | 0 |

Two things follow that I had wrong.

**"Non-ester" was never one class.** It is 503 plain chain aldehydes, 295 ketone-containing, 177
ether-linked and 54 carrying both. A plain fatty aldehyde and an ether-linked tail are not the
same biology, so measuring one anchors that chemotype and not the bucket. Every claim about
"the non-ester subclass" is withdrawn.

**Three chemotypes are absent from the training corpus entirely.** The corpus contains only
ester-linked (56) and unlinked chain (51) aldehydes. Ketone-containing, ether-linked and
ketone+ether components — **526 components across 2,143 products** — appear in no corpus record.
So "these populations are learned from the structural corpus" holds for unlinked chain
aldehydes and is **false** for the ketone and ether chemotypes. Those are genuinely novel
linkage chemistry the generator constructed, which is a stronger result than the one I claimed,
and it also means my earlier ester-share match was superficial: the two populations agree on one
binary axis while the composition of the remainder differs completely.

### The proposal arm explains part of the shape, not most of it

| | broad prior | support enriched | corpus |
|---|---|---|---|
| ester share of products | **0.645** | **0.709** | 0.510 |
| secondary-amine head share | **0.107** | **0.062** | — |

Support-aware tilting does move mass toward measured precursor chemistry: ester prevalence rises
6.4 points and secondary-head prevalence falls 42% relative. That is a real chemical explanation
for the concentration already measured in the yield–diversity trade-off.

But it is not the main driver. The broad arm alone sits 13.5 points above the corpus on ester
prevalence, so most of the gap is the base generator reusing ester components across more
products, not the proposal. **Use the broad arm for any statement about what the generator
learned; use the combined pool only for the candidate funnel.**

### What is claimed

Measured conditionals on the frozen run, not properties of the support predicate, which
contains no motif test:

```
Pr[X in S_pred | admitted, aldehyde chemotype != ester_linked ] = 0    (n = 8,542)
Pr[X in S_pred | admitted, secondary-amine head             ] = 0    (n = 2,263)
```

The two classes overlap: 1,001 products carry both, 44% of all secondary-head products. Union
9,804, **37.4% of the primary population**. They are two precursor-role manifestations of one
supervision gap, not independent demonstrations.

### Suggested main-text wording

> **Structural reach exceeds biological evidence.** Of 26,235 distinct admitted products, 8,542
> carry an aldehyde component lacking the ester motif common to all 11 measured aldehyde tails,
> and 2,263 carry a secondary-amine head against 20 measured primary heads; 1,001 carry both, so
> 9,804 products, 37.4% of the population, are affected. None enters the frozen
> prediction-supported set. The excluded aldehydes span several chemotypes — plain chain,
> ketone-containing and ether-linked — of which the ketone and ether classes appear in no
> training-corpus record, and should not be interpreted as a single biological class. These
> designs pass molecular generation and final-assembly admission but are withheld from
> biological ranking because the measured data provide no support for their precursor chemistry.

with the constraint alongside:

> This establishes neither the activity nor the route completeness of the excluded populations;
> they are removed before route assessment. Measuring representative members of a given
> chemotype would anchor that chemotype, not the excluded population as a whole.

### Claim withdrawn

Do not write "the generator reproduces the structural-corpus composition." Component-identity
ester share is similar (0.497 against 0.523), but product prevalence differs by 16.5 points, the
pool mixes a base and a deliberately tilted proposal, and three generated chemotypes have no
corpus counterpart at all. The defensible statement is narrower: *both ester-linked and unlinked
chain aldehydes are represented in the structural corpus, and the fraction of distinct aldehyde
identities carrying the ester motif is similar between corpus and generated population.*

Artifacts: `results/phase1/forge_aldehyde_subclass_funnel_v1/result.json` (funnel),
`results/phase1/forge_aldehyde_chemotype_audit_v1/result.json` (chemotypes and arms, with SMARTS,
precedence, exact reconciliation and inspection samples recorded).
Tests: `tests/test_aldehyde_chemotype_audit.py`.

**This analysis is frozen. No third precursor class, no further stratification.**


---

## 3f. RESULT: morphology-standardized composition, and where the new chemotypes came from

Two audits replacing the withdrawn corpus-composition claim. **Neither is a fidelity proof.**
Held-component-family evaluation carries that, and this is one diagnostic inside it.

### Why the withdrawn claim was not a well-posed test

FORGE is conditional. A run produces `q_pi(x) = Σ_c pi(c) p_theta(x|c)`, and the production pool
is two arms with two different `pi`, one deliberately tilted toward chemistry likely to reach
predictive support. Comparing a run's raw marginal against the corpus charges the proposal's
choices to the model.

### Morphology-standardized product composition

Cells are (carbon count / 4 capped at 6, branch points capped at 2, has ring, C–C unsaturation
capped at 2), computed identically on both sides. Weighting `ω(c)` is the **generated cell size**;
floor 200 products per side; **99.0% of generated products** sit in a compared cell.

| | ester share |
|---|---|
| raw generated products | 0.674 |
| raw corpus products | 0.510 |
| raw delta | +0.165 |
| **standardized delta** | **+0.045** |
| **family-clustered 95% interval** | **[−0.236, +0.551]** |

**The interval includes zero and is 0.787 wide.** So the honest statement is:

> Morphology standardization accounts descriptively for roughly three quarters of the raw
> product-level ester difference. The residual is not distinguishable from zero at the
> resolution the corpus supports.

### Why the interval is that wide, and why I mislabelled this

I called the product-weighted version "properly powered." That was the pseudo-replication error.
The corpus has 112,386 products built from **107 distinct aldehydes in 13 families**:

| cell | corpus products | distinct aldehydes | families | products per aldehyde |
|---|---|---|---|---|
| (4,0,0,1) | 13,174 | **9** | 4 | 1,464 |
| (2,0,0,1) | 12,837 | **4** | 3 | 3,209 |
| (2,2,0,0) | 9,174 | **2** | 2 | 4,587 |

Product rows repeat the same aldehydes combinatorially against many heads and isocyanides.
Weighting by products inflates n roughly a thousandfold without adding independent chemical
evidence. The cell I highlighted as "corpus 0.000 across 12,837 products" is **four molecules**.

The resampling unit must be the aldehyde family, and there are 13 of them. Hence the width.

### Both estimands, neither discarded

| estimand | question it answers | standardized delta |
|---|---|---|
| **product-weighted** | sample a random product under common morphology: how often ester? | **+0.045**, CI [−0.236, +0.551] |
| **component-weighted** | sample a distinct aldehyde identity under common morphology: how often ester? | **−0.086** |

I previously discarded the component-weighted result as underpowered and kept the product one.
That was wrong: they answer different questions and the opposite signs are informative. Together
they suggest FORGE may construct a similar or slightly lower fraction of distinct ester
identities while reusing them across more products. Both are reported; neither is a fidelity
verdict.

### Provenance of the out-of-registry chemotypes

For the 526 generated aldehyde components in ketone or ether chemotypes absent from the corpus
aldehyde registry, radius-2 environments against training-fold environments:

| coverage | components |
|---|---|
| all seen (some role) | 63 |
| ≥ 90% | 42 |
| 50–90% | 421 |
| **< 50%** | **0** |

Mean 0.813 against training aldehydes alone, 0.865 across any role. **No component is built from
unseen local chemistry**, which rules out decoder artifacts assembled from motifs never seen.

It establishes nothing stronger. Radius-2 coverage of long aliphatic chains is close to
automatic because most environments are CH2, so the test only has power against exotic
chemistry. Treat it as a floor.

### Leave-one-family-out: the residual rests on a single family

| | |
|---|---|
| point estimate | +0.045 |
| leave-one-family-out range over 13 families | **[−0.222, +0.445]** |
| dropping the most influential family alone | +0.045 → **+0.445** |

One family carries the estimate. With that and the bootstrap interval spanning zero, **no
precise conditional estimate is supportable from this corpus**, and the analysis is frozen as a
descriptive attenuation only.

### Linkage-centred lineage REVERSES the whole-molecule reading

Whole-molecule radius-2 coverage said no component fell below 50% and the mean was 0.865.
Testing the environment around the **defining ether or ketone centre** instead:

| classification of the defining centre | components | share |
|---|---|---|
| seen in the aldehyde role | **0** | 0.000 |
| absent from aldehydes, seen in another role | 117 | 0.222 |
| partly supported, partly unseen | 35 | 0.067 |
| **unseen in any training role** | **374** | **0.711** |

**71% of these components carry linkage chemistry present in no training role.** The
whole-molecule statistic was dominated by CH2 environments and said nothing about the part that
defines the chemotype, exactly as suspected.

So the compositional-recombination reading is **not supported**. These are local extrapolation
at the defining centre. The earlier sentence "no component is built from unseen local chemistry"
is true of whole-molecule averages and misleading about the defining chemistry; it is withdrawn.

What may be said: 117 components (22%) transfer a linkage environment from another precursor
role into the aldehyde role, which is credible cross-role reuse. The remaining 374 require
chemical scrutiny before any claim, and must not be described as learned chemical reasoning.

### Forward-outcome audit: exact is not unique

200 ketone-containing designs sampled from 1,638:

| outcome | designs | share |
|---|---|---|
| unique — target is the only product | 166 | **0.830** |
| target among 2 products | 34 | **0.170** |
| aldehyde handles per component | 1 in all 200 | |

Every component carries exactly one aldehyde handle, so the ketone is not being mistaken for
it. But **17% have a competing forward outcome**: the intended product is admissible, not
unique.

This confirms empirically that the two terms are different and must not be interchanged:

- **exact forward reconstruction** — the target product is produced (what admission verifies)
- **unique forward reconstruction** — no competing product is produced (not verified)

Designs in the 17% should not be described as carrying an unambiguous experimental program.

### What may be said now

> FORGE models lipid structure conditionally on coarse precursor morphology rather than
> reproducing an unconditional corpus histogram. Standardizing to comparable morphology accounts
> for most of the apparent ester-prevalence difference at the product level, though the residual
> cannot be resolved against a corpus containing 107 distinct aldehydes in 13 families. Broad
> sampling nevertheless constructs precursor structures and role-level chemotypes absent from
> the finite component registries, while support-enriched sampling shifts computation toward the
> narrower chemistry the biological dataset can rank.

The clause "built entirely from local chemistry present in training" is **struck** from the
sentence above. It rested on the whole-molecule radius-2 statistic that the linkage-centred test
overturned earlier in this section, where 71.1% of defining ether and ketone centres are unseen in
any training role. Leaving it in place would have contradicted the finding recorded above it.

Artifact: `results/phase1/forge_matched_morphology_fidelity_v1/result.json`.

**The origin ablation inherits this contract: state ω(c), report both estimands, cluster
uncertainty at the component family, and let held-family evaluation carry fidelity.**


---

## 3g. RESULT: the full census, the funnel, and a comparison bug that had to be fixed

Every analysis below was prespecified in `docs/FORGE_EVIDENCE_CONTRACT_v1.md`, committed at
`cd833c3` before any of these numbers existed.

### A comparison bug, found by cross-checking against the frozen panel

The first pass of both new scripts tested enumeration membership by raw canonical SMILES. **3,652
of the 12,276 enumerated AGILE products carry stereochemistry and our pipeline is constitutional
and emits none**, so every stereo-bearing library member was counted as chemistry the library
never contained. The error moved 798 designs from inside to outside and inflated the
out-of-enumeration share from 87.6% to 90.3%.

It surfaced because the funnel reported 38 of 40 frozen candidates outside the enumeration while
the panel's own frozen `absent_from_agile_library` flag says 32. The panel builder was right: it
strips stereochemistry from both sides. Both new scripts now do the same, and the funnel raises a
hard error if its count ever disagrees with the frozen panel flag again.

**The published 86.3% is unaffected.** That figure is the main-ledger population under the correct
stereo-free comparison. For the record, all four combinations:

| population | comparison | outside | share |
|---|---|---|---|
| main ledger, 26,235 | raw | 23,406 | 89.2% |
| **main ledger, 26,235** | **stereo-free** | **22,645** | **86.3%** |
| union of both, 29,850 | raw | 26,947 | 90.3% |
| union of both, 29,850 | stereo-free | 26,149 | 87.6% |

Quote 86.3% only with the main-ledger population attached, and 87.6% only with the union.

### Forward-outcome census: non-uniqueness is a polyamine head, not an exotic tail

Full census, all 29,850 admitted products, no sampling. Enumeration cap 100, inherited and not
tuned; no design reached it.

| stratum | designs | unique | non-unique | unique rate |
|---|---|---|---|---|
| all admitted | 29,850 | 24,750 | 5,100 | 0.829 |
| not prediction-supported | 27,221 | 22,268 | 4,953 | 0.818 |
| **prediction-supported** | 2,629 | 2,482 | 147 | **0.944** |
| inside enumeration | 3,701 | 3,520 | 181 | 0.951 |
| outside enumeration | 26,149 | 21,230 | 4,919 | 0.812 |
| prediction-supported and outside | 1,042 | 967 | 75 | 0.928 |
| **frozen 40** | 40 | 39 | 1 | **0.975** |
| **enumerated AGILE library** | 12,276 | 11,718 | 558 | **0.955** |

The mechanism is exact rather than inferred. **All 5,100 non-unique designs carry two or more
distinct reactive amine sites and exactly one aldehyde and one isocyanide site.** No other
combination occurs anywhere in the census. Non-uniqueness is regiochemistry at a polyamine head.

**This corrects the reading reported two sessions ago.** The 17% figure from the 200-design ketone
sample was the population base rate, not a ketone effect. By chemotype:

| aldehyde chemotype | designs | unique rate |
|---|---|---|
| ether-linked | 490 | 0.886 |
| ketone-containing | 1,817 | 0.853 |
| ketone and ether | 93 | 0.849 |
| ester-linked | 19,563 | 0.830 |
| unlinked chain aldehyde | 7,887 | 0.817 |

Ketone-containing designs are slightly *less* ambiguous than ester-linked ones. The exotic tail
chemotypes are not the source of ambiguity, and must not be described as if they were.

Conditioning on the mechanism, because the raw inside/outside gap could be nothing but head
composition:

| population | share with multi-site head | P(non-unique \| multi-site head) |
|---|---|---|
| inside enumeration | 0.108 | 0.451 |
| outside enumeration | 0.241 | 0.779 |
| prediction-supported | 0.105 | 0.533 |
| enumerated library | 0.136 | 0.333 |

Both channels contribute. Out-of-library designs carry multi-site heads more than twice as often,
**and** are more often ambiguous given one, which means generated heads more often carry two
chemically *inequivalent* nitrogens rather than two symmetric ones. That is a real property of
open-world head generation and belongs in the limitations, with the post-generation uniqueness
gate as its remedy.

**Trigger 3 does not fire.** Ambiguity decays along the funnel: 17.1% of broad exploration, 5.6%
of the prediction-supported pool, 1 of 40 frozen candidates. It is confined to broad exploration,
so the generator is not narrowed.

**Action required at selection of the 12.** C06 has two forward outcomes from a two-site head.
Under the contract it is excluded from the 12, or carries a named competing-outcome risk in its
dossier. Its competing product is recorded in the artifact. The frozen 40 is not modified.

### Open-world actionability funnel

Unit is distinct product graphs over the union of both rescoring sources. Uniqueness is an
annotation per stage, never a filter, because the frozen 40 was not selected under a uniqueness
gate.

| stage | inside | outside | outside share | distinct aldehydes in | out |
|---|---|---|---|---|---|
| chemically admitted | 3,701 | 26,149 | 0.876 | 62 | 2,513 |
| prediction-supported | 1,587 | 1,042 | **0.396** | 41 | 160 |
| absent from train fold | 574 | 1,038 | 0.644 | 38 | 160 |
| inside physicochemical envelope | 541 | 1,008 | 0.651 | 38 | 157 |
| all components resolved | 541 | 931 | 0.632 | 38 | 130 |
| route-complete within steps | 541 | 931 | 0.632 | 38 | 130 |
| frozen 40 | 8 | 32 | 0.800 | | |
| selected 12 | pending | pending | | | |

**The narrowing is evidential, not synthetic.** The outside share falls from 0.876 to 0.396 at a
single gate, predictive support, and then recovers and holds through every synthesis gate. The
2,513 distinct outside aldehydes collapse to 160 at the same step. Nothing downstream of support
removes anything comparable.

The 0.800 at the frozen 40 is **not** a funnel outcome and must not be read as one. The panel
selector sorts `agile_novel` candidates first within each arm, so 32 of 40 reflects a stated
selection preference operating on a population that was 0.632 outside.

### Route completeness, stratified

| stratum | admitted | reached routing | route-complete | conditional | unconditional |
|---|---|---|---|---|---|
| inside | 3,701 | 541 | 541 | **1.000** | 0.146 |
| outside | 26,149 | 1,008 | 931 | **0.924** | 0.036 |

Difference, outside minus inside, conditional on reaching route assessment: **−0.0764**, cluster
bootstrap 90% interval **[−0.1034, −0.0552]** over 162 aldehyde clusters, seed 20260815.

The denominator was fixed before computing: conditional on reaching route assessment, because
designs removed earlier are never routed and their route completeness is unmeasured.

**Verdict under the prespecified ±10 point rule: measured trade-off, not parity.** The call is
narrow and the narrowness is part of the result. The point estimate sits inside the band; only the
interval's lower end crosses it, at −10.34 against a −10.0 boundary. The rule is not revised.

All 77 outside-stratum losses occur at component resolution, none at the step budget.

**Wording that is earned:**

> Designs outside the complete enumerated library reach route completeness at 0.924 against 1.000
> for designs inside it, a difference of 7.6 points whose clustered interval marginally excludes
> the prespecified parity band. Chemical expansion therefore survives synthesis qualification
> nearly intact. It is predictive support, not routing, that returns the actionable population
> toward the measured library.

Artifacts: `results/phase1/forge_forward_outcome_census_v1/result.json`,
`results/phase1/forge_open_world_actionability_funnel_v1/result.json`.

---

## 3h. WITHDRAWN AS STATED, see 3h-corrected below

Contract section A1, run after the rule was committed.

> **This section measured `AdapterNodeConditioning`, which belongs to `UgiChemistryFlow`. The
> production generator is `UgiJointSparseFlow`, which does not instantiate that module and has no
> `origin_embedding`. Every number below is correct for the module it measured and is withdrawn
> as a statement about the production model. Do not quote 0.552, the level-A/level-B equivalence,
> or the consumption-site table as properties of the production generator. See 3h-corrected.**

### What the surviving channels know

`AdapterNodeConditioning` embeds five node channels. `use_all_port_distances` is False in
production and is never set True anywhere in the repository, so all-port distances are excluded.
Fitting the Bayes-optimal predictor on the train fold and evaluating on the heldout fold:

| quantity | value |
|---|---|
| heldout atom accuracy | **0.5519**, clustered 90% interval [0.5509, 0.5528] |
| product-level exact match | 0.0000 |
| H(origin \| channels) / H(origin) | 0.7252, against H(origin) = 1.670 bits |
| accuracy on precursor-interior atoms | **0.5007**, against a 0.446 majority baseline |
| accuracy on role anchors | 1.0000 (by construction) |
| accuracy on assembly-introduced atoms | 1.0000 (by construction) |

Both by-construction leaks were confirmed rather than assumed: `port_states` names the role at all
337,158 anchors, and an absent own-port distance identifies all 112,386 assembly atoms.

**Under the prespecified rule, 0.5519 < 0.60, so level A is a genuine information ablation.**

### Level A and level B are the same ablation

Unanticipated, and it removes an arm. Both levels give identical accuracy to four decimals and the
identical count of 31 realized channel tuples, because `distance_to_own_port` equals
`distance_to_core` for **97.3%** of precursor-interior atoms and differs by exactly 1 for the rest,
while `port_states` and the assembly marker are already implied by `core_position`. The
role-indexed group is informationally redundant. **Only one ablation arm is needed, not two.**

### What deleting the embedding does not remove

Origin is consumed in four places, and only the first disappears at level A:

| site | file | classification | survives level A |
|---|---|---|---|
| `origin_embedding` | `adapter_node_conditioning.py:48,112` | input conditioning | no |
| role-balanced loss over origin masks | `ugi_chemistry_flow.py:645-662` | training objective | **yes** |
| per-role source distribution broadcast | `ugi_chemistry_flow.py:618-633`, `:988` | generative process | **yes** |
| role-partitioned closures and per-role exteriors | `ugi_joint_sparse_flow.py:106-128` | state and sampling constraint | **yes** |

So the arm must be called an **origin-embedding ablation**. Under the contract's rule, "origin-free"
is not available, and "product-only" is not available. The generative process remains role-
partitioned by construction: the model always knows which role it is currently building, because
each role is serialized as its own spanning tree with its own source distribution and its own loss
term. The embedding restates per atom what the process already enforces.

### What this predicts, and why it should be said before the run

The repaired inverse audit established that Ugi products decompose uniquely from the product graph
alone. Origin is therefore a deterministic function of any *completed* graph, and an ablation can
only matter in the *partially generated* regime, where the model must commit to role structure
before the graph is complete. The prior on this arm should be neutrality, and the contract's
neutral branch already fixes the consequence: keep origin as a native structured-output interface,
claim no generative quality gain, and do not restore "reaction-resolved" to the title.

Artifact: `results/phase1/forge_origin_channel_information_audit_v1/result.json`.

---

## 3h-corrected. RESULT: role resolution is not a removable component

Found while computing the parameter counts for the planned ablation arms: `origin_embedding` does
not appear in the production model's parameters at all.

| fact | value |
|---|---|
| production model class | `UgiJointSparseFlow` |
| instantiates `AdapterNodeConditioning` | **no** |
| has an `origin_embedding` | **no** |
| role channel | `role_embedding`, shape (3, 192) |
| role_embedding parameters | 576, **0.0533%** of 1,081,388 |

`AdapterNodeConditioning` lives in `UgiChemistryFlow`, a separate architecture that the blinded
sampling path never loads.

### Role is exactly determined by the conditioning

Verified over 9,000 records spanning all three folds, **zero mismatches**:

> The conditioning program declares per-role node counts, and the sparse serialization lays nodes
> out in contiguous role blocks in that order. So `role_states` and `within_role_positions` are
> exact deterministic functions of conditioning the model always receives.

Recoverability is **1.0**, not the 0.552 measured for the other module. Under the evidence
contract's own rule, recoverability at or above 0.95 means deleting the embedding is **not** an
information ablation. Deleting `role_embedding` would remove a redundant re-encoding of the
conditioning at 0.053% of the parameters, not the role information.

### What may be claimed, and it is stronger than the ablation would have been

> Role resolution in FORGE is carried by the conditioning program and by the state layout, not by
> a removable component. It is a property of the representation rather than a feature that can be
> switched off.

**Do not write** that an ablation shows precursor-role information helps or does not help
generation. No such ablation has been run, and the one that was planned could not have shown it.
A genuine role-information ablation would require changing the state layout so role is no longer
implied by position, which changes the modelled object and is out of scope.

Artifact: `results/phase1/forge_production_role_channel_audit_v1/result.json`.

---

## 3i. RESULT: held-family evaluation, stage 1. Encouraging, and narrower than it looks

Contract section B. The evaluation checkpoint is the development run
(`ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt`), trained on the 66,464-product
train fold with calibration used only for early stopping. **No retraining.** Verified in the
script: every component family of every train-fold product is a train family, so the model saw
zero held families.

**This stage is teacher-forced denoising loss only.** Admission, uniqueness, morphology adherence
and novelty are sampling measurements and have not been run. Stage 1 therefore cannot fire or
clear trigger 1; it is necessary, not sufficient.

### Why the contrast lives inside the heldout fold

No heldout-fold product is built entirely from train families, by construction. An
out-of-sample all-familiar reference does not exist and was not invented. Instead the comparison
is between heldout-fold products whose role-r family is held and heldout-fold products whose
role-r family is train. Both groups are unseen products; only family novelty differs.

Calibration-fold families are reported separately from heldout-fold families rather than lumped
as "held". Calibration steered early stopping, so those families are weakly seen. Lumping them
hid that the single unseen isocyanide family behaves like a train family while a calibration
family is seven times worse.

In-sample train-fold reference, the fit floor: **1.391**.

### Two estimands, and they disagree

| role | train-family | held marginal | Δ marginal | Δ ω-matched | matched coverage of held |
|---|---|---|---|---|---|
| aldehyde | 2.634 | 3.940 | **+1.306** | **+0.088** | 0.082 |
| amine head | 3.458 | 3.436 | −0.022 | +0.033 | 0.262 |
| isocyanide | 1.651 | 2.181 | +0.530 | **+0.988** | 0.225 |

ω-matching is exact: for every morphology program present in both groups, the same number of
products is drawn from each side, so the groups carry identical program multisets and the pooled
difference is standardized by construction. Matched groups are ordered by program, so
corresponding batches share morphology, the same t from a fixed 9-point grid, and the same noise
generator seed. Only the clean states differ.

### The caveat that governs the claim

**Four of the six held aldehyde families share no morphology program with any train-family
product at all.**

| held aldehyde family | products | matchable |
|---|---|---|
| 26c4e1d574b04edd | 4,383 | yes, Δ = +0.242 |
| c9c8746eff3946f4 | 4,180 | yes, Δ = −0.044 |
| 17158463fd33fdcc | 4,298 | no shared program |
| 0d39f75c4c34704e | 3,196 | no shared program |
| 29748bbbe6e353c5 | 2,645 | no shared program |
| d12265eb5121cde4 | 2,602 | no shared program |

So the +0.088 rests on two families and 8.2% of held-family products. A different aldehyde family
generally brings its own chain lengths and branching, so family novelty and morphology novelty
are largely confounded in this corpus, and no estimand separates them outside the overlap.

The amine role is better supported, at 26.2% coverage, but its per-family spread is wide:
**[−1.531, +2.322]** over seven held families, with several held families modelled *better* than
train families and one much worse. A pooled number hides that, which is the leave-one-family-out
lesson again.

The isocyanide role cannot support any general claim: there is exactly one held family.

### What may be said now

> Within the morphology region where held-family and train-family products can be compared at
> all, products built from unseen aldehyde component families cost 0.088 nats of denoising loss
> against an out-of-sample baseline of 2.634, and unseen amine families cost 0.033 against 3.458.
> That region covers two of six held aldehyde families and 8.2% of their products. The marginal
> difference of 1.306 confounds family novelty with morphology, because four of the six held
> aldehyde families occupy programs no train-family product occupies.

**Do not write** that FORGE generalizes to held component families, unqualified. **Do not quote**
the marginal +1.306 as a generalization penalty either; it is mostly morphology.

### One aggregation defect, fixed

A batch in which no record carries a decoration gives an empty-tensor cross-entropy, so
`decoration_atom_ce` returns NaN and poisons the summed `total`. One group hit this, in nine
batches. The aggregate now rebuilds `total` from the defined terms for every batch, so all groups
share one definition, and skipped batches are counted in the artifact. Only the amine
calibration cell was affected; every aldehyde and heldout number was already clean.

Artifact: `results/phase1/forge_held_family_generative_evaluation_v1/result.json`.

**Stage 2, not yet run:** conditional sampling under held-family morphology, measuring admission,
validity, uniqueness, exact per-role morphology adherence, and component and product novelty by
role. Section B is not complete until it is.

---

## 3i-stage2. RESULT: the constraint-only control moves the claim off admission

Contract section B2, amended by Amendment 2. 3,072 programs, 1,024 per stratum, sealed by
SHA-256 before sampling. Development checkpoint (train fold only) with the train-fold-clean
closure model; settings inherited verbatim from the frozen production execution config.

**Naming.** Conditional sampling under held-family-derived precursor-architecture programs. A
program is test-time conditioning, not a molecular identity, so this is not evidence that unseen
chemical families are regenerated.

### Which program fields are enforced, field by field

The earlier blanket statement that "the whole morphology program is enforced" was **wrong** and
is withdrawn. Measured on the admitted Stage 2 output:

| field | enforced? | evidence |
|---|---|---|
| exterior atom count per role | **yes, to equality** | 9,216 of 9,216 role slots exact |
| cycle rank per role | **yes, to equality** | 9,120 of 9,120 exact, over 6 distinct requested vectors |
| junction budget per role | **no, upper bound only** | 8,118 exact (89.0%), 1,002 under, **0 over** |
| core attachment count | not determined | not recoverable from the saved rows |

So junction budget is a genuine adherence measurement: the flow chooses whether to spend the
budget and leaves 11.0% of it unused, while never exceeding it. Atom count and cycle rank are
constraint properties and remain barred from being reported as learned adherence.

### Validity and admission are one predicate

Validity, terminal validity, component-reconstruction validity and admission agree on all 3,072
rows: 3,040 true, 32 false, no disagreement. One quantity, reported once.

### The result, and why admission is not the headline

| stratum | n | admitted | unique \| adm | distinct | absent from train fold | outside enumeration |
|---|---|---|---|---|---|---|
| A train-family-derived | 1,024 | 0.988 | 0.959 | 1,001 | 0.966 | 0.967 |
| B held-family overlapping | 1,024 | 0.992 | 0.953 | 983 | 0.950 | 0.953 |
| C held-family disjoint | 1,024 | 0.988 | 0.963 | 1,012 | 0.995 | 0.996 |

Δ_OOD for C against A: +0.000 admitted, +0.004 unique, +0.011 uniqueness of admitted.

**Per source family**, because stratum C draws 1,024 programs from six families: pooled admission
0.9883, per-family range [0.9769, 1.0000], leave-one-family-out range **[0.9865, 0.9905]**. The
clustering concern was warranted and the result survives it.

### The constraint-only control, and what it overturned

Identical programs, masks, closure model, schedule, sources, admission predicate and seeds; only
the denoiser replaced. The script refuses to run if any parameter tensor matches the trained
checkpoint.

| arm | admission | nearest-train Tanimoto (median) | cLogP (median) | MW (median) |
|---|---|---|---|---|
| train corpus | — | — | **8.41** | 565.5 |
| trained flow | 0.988 | **0.644** | 9.73 | 549.9 |
| marginal null | **1.000** | 0.373 | 6.86 | 606.9 |
| random init | 0.72 | 0.106 | −4.78 | 810.5 |

**A denoiser emitting nothing but the role-specific empirical source marginals reaches higher
admission than the trained flow.** Admission is therefore a property of the constrained sampler
and **may not be offered as evidence of learned competence**, in Stage 2 or anywhere else.

The learned contribution is distributional, and it is large. The trained flow sits at
nearest-train Tanimoto 0.644 and cLogP 9.73 against a corpus median of 8.41; the marginal null
sits at 0.373 and 6.86, and random initialization at 0.106 and −4.78, which is not a lipid at
all. The masks guarantee that a legal Ugi product is produced. Only the trained flow produces one
that looks like the chemistry the corpus is made of.

This also disposes of a number that looked good and was not: the marginal null is **100% novel
against every reference**. Novelty earned by leaving the distribution is worthless, and the
trained flow's novelty is meaningful precisely because it coexists with Tanimoto 0.644.

### What "disjoint" means, measured

Stratum C was cut against the train-family subset of the heldout fold, which is not the set the
model trained on. Audited against train-fold programs directly:

- **15.9%** of stratum C programs were in fact present in the training programs, so disjointness
  never licensed "never seen during fitting";
- median L1 distance to the nearest training program is **1**, one unit in one coordinate;
- **20.5%** are novel combinations of role subprograms all individually seen; **79.5%** carry at
  least one role-level value unseen in training, of which 20.3% carry two and 1.4% carry three.

So this is local extrapolation, typically one unit from a seen program, with role-level novelty
in about four fifths of cases. Not wholesale novelty.

### What may be said

> Under precursor-architecture programs drawn from held-out component-family regions, the
> complete system produced chemically admitted products at the same rate as under in-regime
> programs, with no loss of unique assembly or output diversity, and the result holds across all
> six source families under leave-one-family-out. Admission itself is guaranteed by the
> constrained sampler rather than learned: a marginal-only denoiser attains a higher admission
> rate under identical constraints. What the trained flow supplies is chemistry that resembles
> the corpus, at nearest-neighbour Tanimoto 0.644 and cLogP 9.73 against a corpus median of 8.41,
> where the marginal denoiser reaches 0.373 and 6.86.

**Do not write** that Stage 2 shows learned generalization from the admission rate. **Do not
write** that unseen chemical families are regenerated. **Do write** that the sampler guarantees
legality and the learned flow supplies the molecular distribution.

Artifacts: `results/phase1/forge_held_family_stage2_v1/{program_draw,samples,scores,anatomy,
null_marginal_null,null_random_init,learned_contribution}.json`.

---

## 3i-panel. RESULT: the lipid-native panel finds what every generic metric hid

The generic metrics said Stage 2 was fine: admission 0.988, unique assembly 0.96, uniqueness of
output 0.99, novelty 0.97, molecular weight and cLogP in range, nearest-train Tanimoto 0.644.
A role-resolved structural panel against the **exact heldout lipids the programs were drawn
from** says something else.

### The defining linkage is missing

| population | ester-linked aldehyde tail |
|---|---|
| train fold, what the checkpoint was fit on | **0.530** |
| heldout fold | 0.570 |
| matched real reference for the Stage 2 programs | **0.708** |
| **development checkpoint, step 1000, Stage 2 output** | **0.029** |
| marginal null | 0.000 |
| production checkpoint, step 1700, 32,768-draw run | **0.655** |

The failure is visible in a single pair. Real: `C#CCCCCCCCCC(=O)OCCC=O`. Generated:
`C#CCCCCCCCC(=O)C=CCCC=O`. The model places the carbonyl and omits the ester oxygen, giving a
ketone where the family has an ester.

### It is not an out-of-regime failure

| stratum | real ester rate | trained-flow ester rate |
|---|---|---|
| A train-family-derived | 0.582 | **0.026** |
| B held-family overlapping | 1.000 | 0.048 |
| C held-family disjoint | 0.541 | 0.012 |

Uniform across strata, including in-regime programs. **So Stage 2's relative conclusion survives:
Δ_OOD is still approximately zero, because the deficiency is the same everywhere.** What does not
survive is the absolute reading.

### What is withdrawn

> ~~Under precursor-architecture programs drawn from held-out component-family regions, the
> complete system produced chemically admitted products at the same rate as under in-regime
> programs, with no loss of unique assembly or output diversity.~~

That sentence is true and useless. The products are admitted, unique and diverse, and 97% of them
lack the ester linkage that defines 53% of the training corpus and 71% of the matched reference.
**Do not describe the step-1000 development checkpoint as chemically productive.**

### The rest of the panel, and what it does not support

Grouped two-sample AUC against the matched real reference, cross-validated with grouping by
aldehyde component family:

| arm | grouped AUC | correlation Frobenius distance | cLogP median |
|---|---|---|---|
| heldout real | reference | reference | 7.69 |
| trained flow | **0.970** | 5.62 | 9.67 |
| marginal null | 1.000 | **4.58** | 6.76 |
| random init | 1.000 | 10.69 | −5.24 |

The trained flow is separable from real lipids at AUC 0.970, and on the dependence measure and on
cLogP the **marginal null is closer to the real distribution than the trained flow is**. The
trained flow's only clear advantages over the null are grouped AUC (0.970 against 1.000) and
nearest-train Tanimoto (0.644 against 0.373).

**So the amphiphile-design-grammar claim is not earned.** It cannot be written as though it were.
At this checkpoint the trained flow does not demonstrably learn the coupled head, linker and tail
organization of real ionizable lipids, and the context-independent marginal sampler is not clearly
worse on that panel.

### What this does and does not affect

**Unaffected: every production result.** The production checkpoint generates ester-linked
aldehyde tails at 0.655, above its own training prevalence. The 86.3% novelty, the funnel, the
census, the frozen 40 and the route work all come from that model and stand.

**Affected: Stage 2 only**, which correctly used the development checkpoint because it is the
only one that never saw the held families. That creates a real tension worth stating plainly: the
sole checkpoint valid for a held-family claim appears to be a weak generative model of the
chemistry, so held-family generative quality is currently unmeasurable with a valid model.

**Open, and a decision rather than an analysis.** The development run's own selection recorded
`best_step` 500 by calibration loss and ran to 4,750 steps; step 1000 was named by the production
duration contract, not chosen as the best generative checkpoint. Evaluating other development
checkpoints would be checkpoint shopping under the contract's stop rule, so it is not done
unilaterally.

### The methodological point worth keeping

Admission, validity, uniqueness, novelty, molecular weight, cLogP and nearest-neighbour Tanimoto
were all acceptable while the model missed the family's defining linkage 97% of the time. Generic
generative metrics cannot detect a lipid-native failure. Any future quality claim about this
generator has to be made on role-resolved structural chemistry.

Artifacts: `results/phase1/forge_held_family_stage2_v1/{amphiphile_panel,learned_contribution}.json`.

---

## 3i-parity. CORRECTION: Stage 2 evaluated a superseded architecture

A configuration parity audit, run because the ester result was too large to accept without one,
found that **Stage 2 did not evaluate the production method.**

The 32,768-draw production run used
`results/phase1/ugi_decoration_coupling_production_refit_v1/checkpoint_best.pt`, not the
joint-sparse refit. Against the Stage 2 checkpoint:

| | Stage 2 checkpoint | production generator |
|---|---|---|
| `decoration_state_conditioning` | **null** | **bidirectional_anchor_local** |
| extra parameter tensors | — | 11, including `decoration_atom_embedding`, `decoration_to_node`, `no_anchor_context` |
| training steps | 1,000 | 5,100 |
| terminal decoder mode | bond_stochastic | stochastic |

Same schema, same cache, no shape mismatches on shared keys. The production model is a strict
superset: the Stage 2 checkpoint is the same line **before** decoration-state conditioning was
added. So the negative Stage 2 result was about a superseded architecture trained for a fifth as
long, and is withdrawn as evidence about FORGE.

### The corrected run, with no retraining required

A train-fold-only checkpoint of the production architecture already existed:
`results/phase1/ugi_decoration_coupling_v1/challenger/checkpoint_best.pt`. Training folds
`["train"]`, 66,464 records, calibration for early stopping, heldout untouched, step 750
preselected by the historical `calibration_early_stopping` rule. Not checkpoint shopping: it is
that run's own recorded selection.

Same sealed programs, same closure model, same seeds.

| quantity | real heldout | production arch (train-fold only) | old arch | marginal null |
|---|---|---|---|---|
| ester-linked aldehyde tail | 0.708 | **0.190** | 0.029 | 0.000 |
| head nitrogens (median) | **2.00** | **2.00** | 1.00 | 1.00 |
| head max N spacing (median) | **3.00** | **3.00** | 0.00 | 0.00 |
| cLogP (median) | 7.69 | **8.18** | 9.67 | 6.76 |
| mean abs correlation difference | reference | **0.089** | 0.104 | 0.105 |
| grouped two-sample AUC | reference | **0.958** | 0.970 | 1.000 |
| admission | — | 0.967–0.977 | 0.988 | 1.000 |
| unique assembly of admitted | — | **0.675–0.690** | 0.959 | 0.963 |

**The ionizable head is now right.** Two nitrogens at spacing three, matching the real median
exactly, where both the old architecture and the marginal null produce one nitrogen and no
spacing. On the dependence measure the production architecture is now the closest arm to real,
ahead of the marginal null, which it was not before.

**The tail linkage is still wrong.** 0.190 ester against 0.708 real and 0.530 in its own training
fold. Median alkene and median ester are both 0 against 1 in real lipids.

### A metric that looked good because the model was wrong

Multi-reactive-site amine heads: real 0.181, production architecture 0.352, old architecture
**0.043**. The old architecture's excellent 0.959 unique-assembly rate was excellent because it
almost never generated a genuine polyamine ionizable head. Getting the head right **causes**
unique assembly to fall to 0.68, exactly as the forward-outcome census predicted when it found
that every non-unique assembly traces to a multi-site head. Two independent analyses agreeing on
the mechanism is the strongest internal check the project has produced.

The production architecture overshoots, at 0.352 against 0.181, so it is not calibrated on this
axis either. It is in the right regime; the old one was not.

### Where this leaves the claims

**Withdrawn:** the Stage 2 negative as evidence about FORGE, and the earlier Δ_OOD result, which
was measured on the superseded architecture. The corrected Δ_OOD is +0.010 admission and −0.015
unique assembly for the disjoint stratum, still approximately zero.

**Still not earned:** the amphiphile-design-grammar claim. The leakage-free correct-architecture
checkpoint reproduces the ionizable head but not the tail linkage chemistry, and remains
separable from real lipids at grouped AUC 0.958.

**Unaffected:** every production result, which comes from the decoration-coupling refit.

**The decisive remaining experiment** is a train-fold-only run of the exact production
configuration at the production step budget. The existing train-fold checkpoint is step 750
against production's 5,100, so the residual gap between 0.190 and 0.655 confounds training budget
with fold restriction and cannot be attributed without it.

Artifacts: `results/phase1/forge_held_family_stage2_v1/{samples_production_arch,
scores_production_arch,amphiphile_panel_v2}.json`.

---

## 3i-head. RESULT: the null cannot build an ionizable head at all

Ionizable nitrogens and Ugi-reactive nitrogens are different things and were being conflated. The
qualified reaction's amine template is `[N&X3;H2,H1:1]`, so it matches primary and secondary
nitrogens only and **correctly excludes tertiary amines**. A head bearing one primary handle plus
a tertiary ionizable centre is the architecture lipid chemists want, and it counts as one
reactive site, not two. Verified on representative heads: 1-(2-aminoethyl)piperidine matches
once, 1-(2-aminoethyl)piperazine twice. The C06 exclusion and the non-uniqueness census both
stand.

Classifying every head on that distinction:

| head architecture | real heldout | production arch | old arch | marginal null |
|---|---|---|---|---|
| **1 Ugi handle + tertiary ionizable N** (wanted) | **0.394** | 0.271 | 0.390 | **0.000** |
| 1 handle, no other ionizable N | 0.426 | 0.377 | 0.567 | **1.000** |
| **2 or more Ugi-reactive sites** (competing) | 0.181 | **0.352** | 0.043 | 0.000 |

**The marginal null never produces an ionizable head.** Not once in 3,072 draws: every head is a
plain single-handle amine with no tertiary centre. It reaches 1.000 admission, 1.000 novelty and
perfect distinctness while being incapable of the single structural feature that makes an
ionizable lipid ionizable.

That is the cleanest demonstration the project has that the learned flow supplies chemistry the
constraints and role marginals do not, and it is invisible to every generic metric: admission,
uniqueness, novelty, molecular weight, cLogP and nearest-neighbour Tanimoto all failed to see it.

### It also corrects an overstatement made earlier the same day

"The ionizable head is now right" for the production architecture is **withdrawn**. That rested on
median head nitrogen count and spacing matching real exactly, which is a coarse statistic that
concealed the composition. Neither trained arm is calibrated: the production architecture
overshoots competing sites at 0.352 against 0.181 and undershoots the wanted architecture at
0.271 against 0.394, while the old architecture matches the wanted rate at 0.390 but nearly never
builds a competing head at 0.043 against 0.181.

The defensible statement is narrower and still worth having:

> Both trained arms generate genuine ionizable head architectures at 27 to 39 percent, where a
> role-marginal sampler under identical constraints generates none. Neither is calibrated to the
> held-out composition.

Artifact: the classification is reproducible from
`results/phase1/forge_held_family_stage2_v1/samples_production_arch.json` and the null arms.

---

## 3k. AUDIT: the formulation is joint, but the corpus has no cross-role signal to learn

Two audits run to decide whether "preserve the experimental factorization without factorizing
generation" can be claimed as a learned property or only as a modeling one.

### The architecture is not block-diagonal

Causal probe: shift every atom state in one role region by one class and measure how far
predictions move elsewhere, against how far they move inside the perturbed region.

| perturbed | amine | aldehyde | isocyanide |
|---|---|---|---|
| amine | 1.000 | 0.083 | 0.005 |
| aldehyde | 0.089 | 1.000 | 0.122 |
| isocyanide | 0.007 | 0.065 | 1.000 |

Fractions of the within-region effect. **No entry is zero**, so information does cross region
boundaries and whole-molecule generation is joint rather than three independent generators plus a
core. The aldehyde region is the hub, and amine-to-isocyanide is weakest at 0.005 and 0.007,
consistent with the aldehyde block sitting between them in the serialization.

Cross-region influence is nevertheless small: 0.5% to 12.2% of within-region influence.

### The corpus pairs components near-independently

Normalized mutual information between component identities in the 66,464-product train fold:

| pair | normalized MI | grid saturation |
|---|---|---|
| amine x aldehyde | 0.0251 | 0.900 |
| amine x isocyanide | 0.0366 | 0.948 |
| aldehyde x isocyanide | 0.0234 | **1.000** |

The aldehyde-by-isocyanide grid is **fully saturated**: every possible pair of those components
appears. That is the signature of a Cartesian enumeration, and a Cartesian enumeration is
factorial by construction.

### What this settles

**The weak coupling is correct, not deficient.** The model is weakly coupled across regions
because its training corpus is nearly factorial across regions. Nothing is being missed.

**Claim the joint formulation. Do not claim learned cross-role dependence.** There is essentially
none in this corpus to learn, so any such claim would be unfalsifiable here and false as stated.

**The learned contribution is within-region, and it is strong.** A role-marginal sampler cannot
build an ionizable head at all, 0 of 3,072 draws, while the trained arms build them at 27 to 39
percent. Within-region atom-level dependence is where the flow demonstrably earns its keep.

### Wording this permits

> FORGE preserves the factorization through which the experiment manipulates a lipid while
> generating the complete molecular state jointly rather than sampling its components
> independently. The regions are coupled in the model, though only weakly in this corpus, whose
> components pair near-independently by construction. What the learned flow demonstrably supplies
> is within-region molecular structure: a role-marginal sampler under identical constraints
> reaches full chemical admission yet never once produces an ionizable head.

**Do not write** that FORGE learns dependencies between heads and tails. **Do not write** that the
corpus exhibits non-factorial pairing. Both are contradicted above.

Artifact: `results/phase1/forge_cross_region_coupling_audit_v1/result.json`.

---

## 3L. RESULT: at the production budget, the flow does learn the lipid chemistry

Amendment 3 executed with **no GPU run required**. The frozen production duration contract itself
records `selected_development_checkpoint_step: 3000` against `fixed_production_steps: 5100` under
its per-record scaling rule, and that train-fold checkpoint already existed:
`ugi_decoration_coupling_v1/challenger/checkpoint_step_3000.pt`, `model_config` and state keys
identical to production, training folds `["train"]`, heldout untouched. Step 3000 is fixed by the
production contract rather than chosen against a panel, so this is not checkpoint shopping. The
earlier step-750 result used the calibration-early-stopping checkpoint, a fifth of the budget.

Same sealed programs, same closure model, same seeds, same panel.

### The chemistry the earlier runs were missing

| arm | ester-linked tail | wanted head architecture | competing sites |
|---|---|---|---|
| real heldout reference | 0.708 | 0.394 | 0.181 |
| **production budget, step 3000** | **0.510** | **0.479** | **0.131** |
| step 750 | 0.190 | 0.271 | 0.352 |
| old architecture | 0.029 | 0.390 | 0.043 |
| marginal null | 0.000 | **0.000** | 0.000 |

**The ester rate is 0.510 against 0.530 in the fold the model was fit on.** The model reproduces
its own training prevalence almost exactly. The 0.708 reference is the held-family subset, which
is ester-enriched relative to the train fold, so 0.510 is the right comparison and the apparent
shortfall against 0.708 is a property of the reference, not a failure.

### The panel

| arm | grouped AUC | correlation Frobenius | mean abs correlation difference |
|---|---|---|---|
| production budget, step 3000 | **0.882** | **3.53** | **0.071** |
| step 750 | 0.958 | 4.80 | 0.089 |
| marginal null | 1.000 | 4.58 | 0.105 |

Best on all three, and separability from real lipids falls from 1.000 to 0.882.

Eight of eleven highlighted descriptor medians now match the real median exactly: head nitrogens
2.00, head nitrogen spacing 3.00, aldehyde carbons 15.00, isocyanide carbons 11.00, tail asymmetry
6.00, head-to-tail ratio 0.31, aldehyde branches 0.00, and **degradable linkages 1.00**. Misses:
tertiary amines 1.00 against 0.00, aldehyde alkenes 0.00 against 1.00, cLogP 8.17 against 7.69.

### Admission and assembly

| stratum | admitted | unique of admitted | distinct | outside enumeration |
|---|---|---|---|---|
| A train-family-derived | 0.978 | 0.919 | 987 | 0.851 |
| B held-family overlapping | 0.956 | 0.918 | 943 | 0.819 |
| C held-family disjoint | 0.981 | 0.841 | 1,004 | 0.985 |

Δ_OOD for the disjoint stratum: +0.004 admission, **−0.078 unique assembly**. A real if modest
out-of-regime cost on unique assembly, which the earlier arms could not show because they were not
producing the polyamine heads that create it.

### The frozen outcome contract resolves positive

Head architecture retained, tail statistics substantially recovered, beats the null on every panel
measure, separability falls. Amendment 3's positive branch licenses:

> FORGE learns a context-dependent ionizable-lipid design prior beyond the legality supplied by
> reaction constraints and role-wise marginals. Under identical constraints and the same sealed
> design programs, a role-marginal sampler reaches full chemical admission while producing no
> ionizable head architecture at all and no ester linkage at all; the trained flow reproduces the
> ester prevalence of its training fold, the ionizable-head architecture, and the head-to-tail
> balance of held-out lipids.

**Caveats that travel with it.** Still separable at grouped AUC 0.882, so not distributionally
matched. Tertiary amines overshoot and aldehyde unsaturation undershoots. One seed, so this is a
decisive diagnostic rather than a variance-estimated superiority result, exactly as Amendment 3
specified for the single-seed case.

**Per the stop rule, model development ends here.** No architecture sweep, no targeted correction:
the outcome is positive, not the all-seeds-fail case that would have licensed one.

### What the earlier Stage 2 negative actually was

Fully explained. An obsolete architecture without decoration-state conditioning, at a fifth of the
production budget. Both causes are now measured: architecture moved ester from 0.029 to 0.190, and
budget moved it from 0.190 to 0.510.

Artifacts: `results/phase1/forge_held_family_stage2_v1/{samples_prod_budget,scores_prod_budget,
amphiphile_panel_v3}.json`.

---

## 3m. RESULT: semantics help inside the observed vocabulary and hurt outside it

Contract amendments 11, 12 and 13. Four arms, `flat_no_role` and `flat_true_role` at coverage
alpha in {0.25, 1.00}, seed 20260817, 3,000 steps, all complete with finite losses. Every arm
compared at `checkpoint_step_3000.pt`, never `checkpoint_best.pt`, which lands at each arm's own
calibration step.

Pairing exact: identical architecture, parameter count, initial weights across all 84 tensors,
minibatch sequence, corruption draws, t values, objective, source law, program vector, global
position and budget. The only difference is whether the correct local role map is exposed.

### Step 1: completion

| arm | final training total | wall |
|---|---|---|
| no_role, alpha 0.25 | 0.7959 | 505s |
| no_role, alpha 1.00 | 0.7252 | 400s |
| true_role, alpha 0.25 | 0.5996 | 374s |
| true_role, alpha 1.00 | 0.5151 | 324s |

### Steps 2 and 3: the gap is positive, coherently

On the fixed combination test stratum, 57 of 60 primary cells are positive, 1 within 1e-4 of zero
and 2 marginally negative (closure bond at t = 0.90, magnitude 3e-4). Total gap:

| alpha | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=0.90 |
|---|---|---|---|---|---|
| 0.25 | +0.439 | +0.418 | +0.278 | +0.159 | +0.059 |
| 1.00 | +0.513 | +0.445 | +0.291 | +0.163 | +0.067 |

**Rung 1 earned.**

### Step 4: it grows sharply toward corruption

Under `q_t = t*delta_x + (1-t)*p_0`, low t is more corrupted. The gap is roughly **7.5x larger** at
t = 0.10 than at t = 0.90, at both coverage levels, monotone across all five grid points and every
head. **Earned, and it is the strongest pattern in the pilot.**

### Step 5: the data-efficiency hypothesis FAILS

| head | alpha 0.25 | alpha 1.00 | sparse > full |
|---|---|---|---|
| total | +0.2707 | **+0.2956** | no |
| atom | +0.0266 | +0.0282 | no |
| parent bond | +0.0177 | +0.0189 | no |
| offspring | +0.1362 | +0.1402 | no |
| closure bond | +0.0193 | +0.0173 | yes |
| decoration anchor | +0.0416 | +0.0501 | no |

Five of six heads go the wrong way. **Not earned.** Under the frozen gate, the escalation to
10/25/50/100 with three seeds **does not fire, and the coverage branch stops here.**

### The finding the primary stratum concealed

Δ_sem on the other two preregistered strata, t-averaged totals:

| stratum | alpha 0.25 | alpha 1.00 |
|---|---|---|
| S2 new combination of familiar components | **+0.271** | **+0.296** |
| S3 held component identity | **−0.101** | **−0.090** |
| S4 held component family | **−0.081** | **−0.390** |

**The role map reverses sign under component-level extrapolation.** It helps on products assembled
from component chemistry the model has seen and actively costs on products built from unseen
component identities, most sharply on unseen families at full coverage.

That is coherent rather than anomalous: a role prior is useful when the chemistry occupying a region
is role-typical, and a liability when it is not. The semantics encourage commitment to role-typical
structure, which is wrong for a component the model has never seen.

### What may be claimed

> Correct experimentally grounded semantics provide useful denoising information beyond what the
> same network infers from the corrupted molecule, generic position and the global program, for
> products assembled from component chemistry the model has observed. The advantage grows roughly
> sevenfold toward the corrupted end of the generative path. It does not extend to products built
> from unseen component identities or families, where the semantic map is a net cost.

**Do not claim** that the value of semantics increases under sparse combinatorial supervision. Five
of six heads contradict it.

**Do not claim** semantics help generalization to new chemistry. Measured, they hinder it.

**Caveats.** One seed per arm, so this is a decisive diagnostic rather than a variance-estimated
result, exactly as the amendment specified for the single-seed case; the internal consistency
across 60 cells and the monotone t trend are what carry it. The gap is an empirical
semantic-utility gap between two finite trained networks, not a measured mutual information, and no
sample was drawn so nothing here speaks to generated chemistry.

**Licensed next, and only this:** the misaligned-role control, which separates "alignment matters"
from "an extra embedding helps". Roughly $0.22. Nothing else in the branch reopens.

Artifact: `results/phase1/forge_semantics_pilot_v1/delta_sem.json`.

---

## 3n. RESULT: the semantic effect replicates, and alignment is what carries it

Closure sections 1A and 2. Contract amendments 11 to 14. All arms at `checkpoint_step_3000.pt`.

### Replication, with seeds as the unit of uncertainty

Three independent paired training seeds at alpha 1.00, on the fixed combination test stratum,
2,048 records. Uncertainty is across **paired seeds, n = 3**. The t-by-head cells are not independent
training replicates and are never used as the unit.

| head | mean Δ̄ | SD | per seed | all positive |
|---|---|---|---|---|
| total | **+0.2479** | 0.0579 | +0.296, +0.183, +0.265 | yes |
| atom | +0.0266 | 0.0015 | +0.0282, +0.0261, +0.0254 | yes |
| parent bond | +0.0170 | 0.0021 | +0.0189, +0.0148, +0.0175 | yes |
| offspring | +0.1409 | 0.0028 | +0.1402, +0.1439, +0.1386 | yes |
| closure bond | +0.0171 | 0.0025 | +0.0172, +0.0146, +0.0195 | yes |
| decoration anchor | +0.0439 | 0.0054 | +0.0501, +0.0417, +0.0400 | yes |

**Six of six heads positive in three of three seeds.** The total head carries the widest spread, a
23% coefficient of variation driven by one lower seed; the per-head quantities are much tighter.

### Corruption dependence replicates too

Ratio of the gap at t = 0.10 to t = 0.90, per seed: **7.70x, 9.89x, 11.25x**. Under
`q_t = t*delta_x + (1-t)*p_0` low t is more corrupted, so the semantic map is worth most where
local molecular evidence has been destroyed. Present in every seed.

### The misaligned control: alignment is the whole story

Third arm, labels permuted across nodes with each label's count preserved, deterministic per example.
Absolute t-averaged loss, lower is better:

| head | true | none | misaligned | fraction of the true gain recovered by misalignment |
|---|---|---|---|---|
| total | 0.8138 | 1.1094 | 1.1149 | **−1.9%** |
| atom | 0.1181 | 0.1463 | 0.1459 | +1.2% |
| parent bond | 0.0747 | 0.0935 | 0.0945 | −5.1% |
| offspring | 0.1648 | 0.3050 | 0.3108 | −4.1% |
| closure bond | 0.1235 | 0.1407 | 0.1463 | −32.4% |
| decoration anchor | 0.1273 | 0.1775 | 0.1727 | +9.5% |

Misalignment recovers between **−32% and +9.5%** of the benefit, and is **actively worse than no
semantics on four of six heads**. Four heads land in the ideal frozen pattern
`L_true < L_none <= L_misaligned`; the two that do not have misaligned essentially tied with none,
gaps of 0.0003 and 0.005 against true-versus-none gaps of 0.028 and 0.050.

**Licensed statement, from the frozen hierarchy:**

> Correct scientific alignment matters. The gain is not explained by supplying an additional
> regional embedding: a dimension-matched channel carrying the same label frequencies in a permuted
> arrangement recovers essentially none of the benefit and is often worse than supplying nothing.

### The full licensed sentence for the semantic mechanism

> Correct experimentally grounded semantics reduce molecular denoising ambiguity beyond what the
> same network infers from the corrupted molecule, generic position and the global program. The
> effect replicates across three paired training seeds on all six prediction heads, grows roughly
> eight to eleven fold toward the corrupted end of the generative path, and disappears when the
> semantic channel is misaligned while retaining its dimensionality and label frequencies.

**Prohibited, all measured and contradicted:** that the value increases under sparse combinatorial
supervision; that semantics help generalization to unseen component identities or families; that the
gap is a measured mutual information; that any of this speaks to generated molecular chemistry, since
no sample was drawn.

Artifact: `results/phase1/forge_semantics_pilot_v1/closure.json`.

---

## 3o. RESULT: the role-divergence mechanism hypothesis is NOT supported

Closure section 3, on the measure declared in Amendment 14 before computation. One shot, no
alternative divergence computed.

| head | D_h (JSD, bits) | mean Δ̄_sem | Δ_sem at t=0.10 |
|---|---|---|---|
| decoration bond | **0.6638** | −0.0005 | −0.0046 |
| closure bond | 0.2180 | +0.0171 | +0.0613 |
| decoration atom | 0.2081 | +0.0028 | +0.0014 |
| atom | 0.1510 | +0.0266 | +0.0472 |
| offspring | 0.0953 | **+0.1409** | **+0.2065** |
| parent bond | 0.0818 | +0.0170 | +0.0324 |

Spearman rho **−0.543** (permutation p = 0.297) against Δ̄_sem, and **−0.429** (p = 0.420) at
t = 0.10.

**The hypothesis is not supported, and the trend runs the other way.** The head with the highest role
divergence gets essentially no benefit, and the head with among the lowest gets by far the largest.
Neither correlation is significant, which the amendment predicted in advance for n = 6.

### What is therefore unavailable

> ~~Semantic information is most useful for molecular variables whose distributions genuinely differ
> across the experimentally defined regions.~~

**Struck.** That sentence required a positive association on the declared measure and there is none.

### One post-hoc reading, labelled as such

The offspring head, child counts, benefits most, and topology is exactly what the role blocks
*constrain*: knowing the role tells the model which per-role program budget applies, which is
topological rather than distributional information. Decoration bond states, at the other end, are
already close to determined by local chemistry. So the benefit may track how much role narrows the
structural search rather than how different the role-conditional distributions are.

**This is a post-hoc interpretation, not a finding.** It is recorded because it is the obvious reading
and would otherwise be reinvented later, and it must not be presented as evidence. Testing it would
require a new preregistered measure, which the amendment forbids.

### Effect on the semantic claim

None. Section 3 was a mechanistic addendum. The primary result stands on its own: chemically aligned
semantics reduce denoising ambiguity, replicated across three paired seeds on all six heads, eight to
elevenfold larger under heavy corruption, and absent when the channel is misaligned. That claim never
depended on the divergence association.

Artifact: `results/phase1/forge_role_divergence_association_v1/result.json`.

---

## 3p. SUPERSEDED BY 3q. Retained for the gate numbers only

> **The resolution in this section is withdrawn.** It concluded "keep tiered because that is what the
> frozen panel implements", which is backwards: Library 0 is a computational pilot, nothing has been
> synthesized, and there is no reason to inherit a post-hoc ranking rule whose own artifact declines
> to authorize it. The gate table and the S_pred definition below remain correct and are used by 3q.

## 3p-original. Calibration gate numbers, retained

Closure section 5. **Resolved once, with the ambiguity surfaced rather than papered over.**

### Production already uses tiered, verified rather than assumed

The frozen 40 carry **four distinct `conformal_q90` values in exact 1:1 correspondence with the four
authority tiers**:

| conformal_q90 | candidates | authority tier |
|---|---|---|
| 3.9147 | 8 | qualified_role_holdout |
| 4.2959 | 6 | exploratory_weak_absolute_head_generalization |
| 5.3094 | 24 | exploratory_weak_absolute_aldehyde_generalization |
| 6.2145 | 2 | exploratory_simultaneous_exact_new_tails |

Ordering statistic is `lcb90` for all 40. So the pooled-versus-tiered question is **not open**: the
frozen panel implements tiered, and `lcb90 = oracle_mean − conformal_q90(tier)`.

### The frozen gate, applied

`results/phase1/ugi_interpolative_conformal_v1/result.json` carries a four-part guidance gate:
coverage gap at most 0.100, positive-r2 fold fraction at least 0.60, mean test r2 at least 0.10,
mean test Spearman at least 0.20, with at least 3 eligible folds.

| scheme | folds | cov gap | r2 | rho | pos frac | gate |
|---|---|---|---|---|---|---|
| held_aldehyde | 4 | **0.1590** | 0.2498 | 0.568 | 0.75 | **FAIL** (coverage) |
| held_aldehyde_isocyanide_pair | 3 | 0.0831 | 0.5326 | 0.708 | 1.00 | pass |
| held_head | 4 | 0.0587 | 0.3442 | 0.635 | 0.75 | pass |
| held_head_aldehyde_pair | 0 | — | — | — | — | FAIL (ineligible) |
| held_head_isocyanide_pair | 0 | — | — | — | — | FAIL (ineligible) |
| held_isocyanide | 2 | 0.0712 | 0.4180 | 0.621 | 1.00 | **FAIL** (2 folds < 3) |

**Two of six pass.** The isocyanide scheme fails only on fold eligibility while passing every metric
criterion. The aldehyde scheme fails on coverage, driven by one fold at coverage 0.536 and r2 −0.52.

### The ambiguity, surfaced once

**The tier covering 24 of the 40 frozen candidates corresponds to the scheme that fails the frozen
coverage criterion.** And the artifact is labelled a *diagnostic*: its own adjudication records that
the conditional calibration "was specified after inspection of the v3 outer-test diagnostic and
therefore requires a separate versioned authorization review", with
`biological_guidance_authorized: false`.

So there is **no frozen decision that selects tiered over pooled on evidence.** The panel implements
tiered; the diagnostic does not license it; and the criterion was applied to held-role schemes rather
than to a pooled-versus-tiered contrast. Reading that artifact as "the frozen criterion chose tiered"
would be wrong.

### Explicit resolution

**Keep tiered, and report the gate outcome as a limitation rather than as authorization.** Reasons,
recorded:

1. it is what the frozen panel implements, and changing it would alter the frozen panel, which is out
   of scope while candidate selection is deferred;
2. no third calibration scheme is introduced, per the directive;
3. the gate result is *consistent* with how the tiers are already named. The tier that fails coverage
   is literally called `exploratory_weak_absolute_aldehyde_generalization`. The panel already encodes
   that this tier's authority is weak, which is also why the ordering statistic is the conservative
   `lcb90` rather than the mean.

### The formal object for Methods

> **S_pred** is the set of generated products admitted by the frozen distributional-applicability
> predicate of `configs/bio/phase1_ugi_distributional_applicability_v3.json`: thresholds derived from
> structures and split stage only, never from targets, predictions or errors; calibration component
> identity excluded from its own reference; the candidate's distribution bin taken as the **worst of
> the product view and the three component views**; and both the fingerprint and descriptor views
> required to pass. Exact-identity novelty and distributional applicability are independent by
> construction.

Ranking within S_pred uses `lcb90 = oracle_mean − conformal_q90(tier)` under tier-stratified split
conformal calibration at 90%.

### Wording that is licensed, and what is not

> Candidates are ranked by a tier-stratified 90% conformal lower bound rather than a predicted mean,
> and the applicability predicate is derived from structure and split stage alone. Of the six
> role-holdout calibration schemes evaluated under the frozen gate, two pass; the scheme matching the
> tier that supplies most of the frozen panel fails its coverage criterion, which is why that tier is
> designated exploratory and weak.

**Do not write** that calibration was validated, that the frozen criterion selected tiered
calibration, or that coverage was verified at 90% for the aldehyde tier. Measured, it is 0.84 pooled
and 0.159 mean absolute gap across folds.

Artifacts: `results/phase1/ugi_interpolative_conformal_v1/result.json`,
`configs/bio/phase1_ugi_distributional_applicability_v3.json`,
`results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz`.

---

## 3q. DECISION: support and ranking are decoupled; the tiered penalty is retired

Closure section 5, resolved. **Supersedes 3p.**

### Why the previous resolution was wrong

3p concluded "keep tiered because the frozen panel implements it". That inherits a post-hoc,
unauthorized ranking rule for no reason: Library 0 is a computational pilot, **nothing has been
synthesized**, and the final panel is not frozen. The weak point can be removed rather than explained
around.

### What the tiered penalty actually rests on

The `conformal_q90` of 5.3094 applied to **24 of the 40** frozen candidates is the q90 of
`held_aldehyde_5fold` **fold 1**, computed from **20 interpolative calibration rows** — exactly the
policy minimum of 20. Across all 16 eligible folds the interpolative calibration rows run 20 to 42,
median 27.

So a 20-row fold set the conservative penalty for 60% of Library 0, in a scheme that fails the frozen
coverage gate at 0.159 against a 0.100 maximum, inside an artifact whose own adjudication sets
`biological_guidance_authorized: false`. That is not a validated 90% bound and must never be described
as one.

### The decoupling

| layer | object | status |
|---|---|---|
| **support** | `S_pred`, the frozen distributional-applicability predicate | **unchanged and frozen** |
| **rank** | the frozen predictive ensemble within `S_pred` | replaces tier-penalised `lcb90` |
| **strata** | the four authority tiers | **descriptive only**, no effect on any score |
| **uncertainty** | one pooled residual summary | reported, never used to rank or to redesign |

`S_pred` is untouched, so `S_pred` strictly inside `S_gen` and the broad-versus-enriched result are
unaffected. A constant pooled quantile does not change ordering anyway; its role is communicating
uncertainty.

### The pooled summary, descriptive only

On the existing frozen splits, 1,218 interpolative rows:

| scheme | pooled empirical 90% coverage | absolute gap | RMSE |
|---|---|---|---|
| held_aldehyde | 0.8321 | 0.0679 | 2.527 |
| held_aldehyde_isocyanide_pair | 0.9838 | 0.0838 | 2.004 |
| held_head | 0.9095 | 0.0095 | 2.434 |
| held_isocyanide | 0.8333 | 0.0667 | 2.473 |
| **record-weighted** | **0.8842** | **0.0158** | — |

**A pooled residual calibration achieves 0.884 record-weighted empirical coverage against a 0.90
nominal target.** Reported to characterize predictive uncertainty. It is not used to rank, and this
number was not used to redesign anything.

**The denominator needs stating carefully, because it is easy to get wrong twice.** The figure is a
record-weighted mean over **four scheme-level pooled evaluations totalling 1,218 evaluation rows**
(417 + 309 + 210 + 282). That is **not** 1,218 distinct lipids: the measured set contains 1,100, and
1,218 exceeds it because each scheme is a different holdout partition of the same measured set, so
lipids recur across schemes. The artifact's own `evaluated_interpolative_rows: 1218` is likewise the
sum, not a count of distinct molecules.

Licensed phrasing: *pooled descriptive calibration yielded 88.4% record-weighted empirical coverage
across four role-holdout schemes, over 1,218 scheme-level evaluation rows drawn from the 1,100-lipid
measured set.* **Do not write** "over 1,218 lipids" or treat the rows as independent.

### Library 0

> Library 0 was constructed using a diagnostic tier-dependent conservative ranking procedure and is
> retained as a frozen computational pilot. Subsequent analysis showed that nominal tier-specific
> coverage was not uniformly supported, so the final prospective selection uses the frozen
> evidence-support definition with a common ranking rule.

Nothing is erased. Nothing was synthesized. Nothing is lost.

### Manuscript wording

> We first restrict biological prioritization to a prespecified evidence-supported region `S_pred`.
> Within this region the predictive ensemble provides a comparative ranking; calibration is reported
> separately to characterize predictive uncertainty rather than to extend ranking beyond the supported
> domain.

> Predicted activity never guides molecular transitions and never defines the generator's structural
> support.

**Prohibited:** that tier-specific conformal calibration guarantees 90% coverage; that the calibration
study selected the tiered scheme; that calibration was validated; that `lcb90` is the final ranking
statistic.

Artifacts: `results/phase1/ugi_interpolative_conformal_v1/result.json`,
`configs/bio/phase1_ugi_distributional_applicability_v3.json`.

---

## 3j. DECISION: unique final assembly is an eligibility criterion for the twelve

Recorded in `configs/bio/phase1_ugi_final_twelve_eligibility_v1.json`, before the blinded
chemistry review returned and before any selection outcome was known.

> A candidate is eligible for the final twelve only if its target product is the unique product
> enumerated from its own three components under the qualified Ugi transformation.

The criterion is chemical, references no ranking quantity, applies uniformly to all forty, and
does not modify the frozen panel.

**It costs nothing.** All 34 HIGH-arm candidates already satisfy it. The single affected
candidate, C06, is in the LOW arm, which the all-HIGH decision already excludes. Its head is
1-(2-aminoethyl)piperazine and the competing outcome condenses through the piperazine ring
nitrogen rather than the primary amine.

So the earned sentence is a verified property of the eligible pool, not a constraint imposed on
it:

> Every candidate taken to synthesis has a uniquely specified final Ugi assembly.

---

## 4. DECISION: the prospective panel is all-HIGH, blinded, unconstrained

Supersedes the earlier 34 HIGH / 6 LOW design and the interim 8 HIGH / 4 LOW proposal.

**Resolved:** `configs/bio/phase1_ugi_chemist_selection_blind_v1.json` has
`controls.decision: OPEN` with three options. The decision is **option 1: no low-ranked
controls.** All 12 synthesis slots go to HIGH-arm designs.

**Rationale to record:** the 6 LOW candidates are predicted at means of −0.15 to 0.23, against a
measured AGILE library with p10 = 0.59 and median 5.25. Spending 4 of 12 synthesis slots
confirming that predicted-dead lipids are dead is a poor trade when synthesis is the bottleneck,
and the 1,100-lipid measured library already establishes the low end.

**The blinding protocol is unchanged and already frozen.** The chemist sees 34 shuffled `L`-coded
structures with components, routes, starting materials, supplier counts, step counts and
molecular weight. She does **not** see `oracle_mean`, `oracle_sd`, `lcb90`, `conformal_q90`,
percentile-beaten, authority tier, arm, selection order, candidate ID, or either novelty flag.
Brief: *"pick the twelve of the thirty-four you think will be good"*, no constraints imposed.

**Why the brief stays unconstrained — verified, not assumed.** The risk of an unconstrained pick
is that it clusters in a narrow predicted band, leaving no dynamic range for a rank-versus-outcome
test. Simulated over 20,000 random 12-of-34 draws:

- LCB90 spread available across the 34: 2.05 to 8.89, range **6.85**
- median spread within a random 12-pick: **5.56**
- 95.6% of picks span ≥ 4.0 units; 63.2% span ≥ 5.0 units

So an unconstrained blind pick preserves the range with high probability, and no stratification
of the offering is needed. The config's stated rationale for imposing nothing — that any
constraint we add becomes part of the selection rule and must be described as such — stands.

**Statistical consequence for the paper.** The prespecified test is no longer a high-versus-low
contrast. It should be a **one-sided Spearman correlation between LCB90 and measured HeLa across
the 12 synthesized designs**, declared before synthesis. At n = 12 that reaches p < 0.05 at
ρ ≈ 0.5. Every one of the 34 is predicted above the measured library median, so "all high" is
honest: even the weakest, P32 at predicted mean 6.37, beats the median measured lipid.

**Concurrent anchoring replaces the LOW arm.** Run measured AGILE reference lipids on the same
plates as same-batch controls. This buys what the LOW arm was buying without spending generated
design slots, and it avoids relying on a historical base rate across assay batches.

**This is a pre-experimental protocol amendment** in the same class as v6 amending v5. Record it
as an amendment with reasoning; do not silently edit the frozen config.

---

## 5. Framing decisions

Full detail in `docs/FORGE_ICLR_EXPERIMENT_CONTRACT.md`. In brief:

- **Title:** *FORGE: Generative Design of Ionizable Lipids with Reaction-Resolved Discrete Flows*
- **Thesis:** ionizable-lipid discovery as a generative-modelling problem; generate broadly, rank
  selectively, preserve the chemistry needed to act
- **The building-block contrast is a result, not the thesis.** The 86.3% belongs under "does the
  model generate new chemistry," not in the abstract's opening move
- **Never write** "we use discrete flow matching unchanged." Write "we instantiate discrete flow
  matching over a new reaction-resolved molecular state and develop a constrained generative
  process"
- **Prospective validation is evidence, not algorithmic novelty**, until outcomes exist
- **Contribution 3 upgrade:** present the proposal tilt as the solution to a KL-regularized
  objective, `argmax_q E_q[log(ρ+ε)] − (1/γ)KL(q‖π)`, giving `π'(c) ∝ π(c)(ρ(c)+ε)^γ`. Name it
  **support-preserving proposal tilting**. Now that §2 shows a real diversity cost, `γ` should be
  presented explicitly as the exploitation/diversity dial it is
- **Generality:** claim only fixed-arity, single-step molecular families with explicit precursor
  roles and a verifiable forward transform. All experiments are one Ugi-3 adapter

---

## 6. Terminology ledger — use these exact terms

| Term | Value | Never confuse with |
|---|---|---|
| attempts | 32,768 (2 arms × 16,384) | — |
| admitted rows | 30,180 | distinct products |
| distinct admitted products | 26,235 | admitted rows |
| novelty, production model | 83.7% vs 112,386 corpus | the 89.0% train-fold figure |
| novelty, train-fold denominator | 89.0% vs 66,464 | only valid for a train-fold-only model |
| novelty, enumerated library | 86.3% vs 12,276 | either training denominator |
| component novelty, training registry | 74.4% | component novelty vs measured |
| component novelty, measured 1,100 | 99.04% | component novelty vs registry |
| rankable yield, prior → tilted | 3.9% → 8.6%, 2.23× | the production arms' 8.75% → 17.70% |
| route-complete | 97.1% **of prediction-supported designs** | of all admitted designs |
| measured library | 1,100 lipids, 20/11/5 components | the enumerated library |
| enumerated library | 12,276 products, 22/62/9 blocks | the measured library |

---

## 7. Still open

- **The origin ablation is half done.** The post-hoc recovery half is answered (§3b). The
  training half — does removing `o` degrade generation quality — still needs a matched run.
- **A direct pooled-versus-authority-tier conformal comparison** on the production strata has not
  been run; §3 above is the nearest available evidence.
- **No wet-lab outcome of any kind exists.**
- The 12 have not been selected. The blinded packet is ready.
