# FORGE: complete implementation reference

**Purpose.** A single, self-contained, implementation-grounded description of what FORGE is
and what it did, written so that someone who has never seen the repository can write an
accurate long-form article from it. Every quantitative claim here was recomputed from frozen
artifacts on 2026-08-15 rather than copied from a manuscript, and where a manuscript number
required tracing to its source that trace is given.

**Status of the work.** All computational work is complete and frozen. No chemistry has been
performed. No lipid has been synthesized, no nanoparticle formulated, no cell transfected and
no animal dosed. Any statement in this document about biology is a statement about a
prediction or a plan.

---

## 1. What FORGE is, in one page

FORGE is a generative model for ionizable lipids: the amine-containing lipids that make
lipid nanoparticles able to carry mRNA into cells.

The standard approach in this field is *enumeration*. A chemist picks a reaction and a set of
purchasable building blocks, takes the combinatorial cross-product, and either makes all of it
or trains a model to predict which members are worth making. The chemistry that can be
discovered is fixed before any model runs. The library our measurements come from is exactly
this: 22 amines × 62 aldehydes × 9 isocyanides = 12,276 products.

FORGE removes the block list. It generates the lipid atom by atom and bond by bond. The
technical problem this creates is that a molecule drawn from nowhere has no known synthesis:
you would have to run a retrosynthesis model on it and hope the model is right.

FORGE's answer is to change the coordinates the model works in. A **design program** states
coarse morphology separately for the amine, aldehyde and isocyanide roles, and that program
fixes which part of the molecule belongs to which reagent before any structure is generated.
The role assignment is therefore a property of the conditioning, not a label the network
predicts. So a finished sample already states its own decomposition: cut the graph along the
role boundaries the program declared and you have the three reagents. Reading them off is one
pass over the graph.
Checking them is one forward reaction and one graph comparison.

The paper's thesis is that sentence. For molecules made by a known reaction from a few
reagents, the reagent factorization is part of what the molecule *is*, and a generator should
produce it rather than leave it to be recovered.

Two consequences follow, and they occupy the rest of the work.

Because the reagents are generated rather than chosen from a catalogue, they have never been
measured, so the activity predictor is extrapolating on almost everything the model makes.
FORGE handles this by treating "what the generator can reach" and "what the data can rank" as
two different sets, and by calibrating uncertainty separately depending on *which reagent role
is unseen* — a variable that only exists because the decomposition is explicit.

And because the reagents are generated, nobody stocks them. FORGE resolves each one backwards
to purchasable material using published preparation chemistry, with a forward check at every
step.

The output is a frozen list of 40 candidate lipids, each with a complete route to catalogue
material, ranked by a calibrated lower bound rather than a raw prediction, locked before any
experiment.

---

## 2. The chemistry

### 2.1 The reaction

FORGE is instantiated on the **Ugi three-component reaction** (Dömling & Ugi, *Angew. Chem.
Int. Ed.* 39:3168, 2000), in the variant used to build the AGILE lipid library (Xu et al.,
*Nat. Commun.* 15:6305, 2024).

Three reagents condense in one step:

| Role | What it is | Contributes |
|---|---|---|
| **amine** | a primary or secondary amine bearing the ionizable head group | the head |
| **aldehyde** | an ester-linked aldehyde | one lipid tail plus part of the backbone |
| **isocyanide** | an alkyl isocyanide | the second lipid tail |

They fuse onto a fixed five-atom **α-aminoamide core**. An acidic phosphorus reagent is used
experimentally but is a *catalyst*, not a fourth structural component — it contributes no
atoms and FORGE does not model it. Getting this wrong is a common misreading; the reaction is
three-component.

Configuration: `configs/assembly/ugi_variant.yaml`
(sha256 `5b97e062b115fcc137b4a05d8f72b74e9cc67a584c05c054ef5f983969bf1427`),
registry `data/vendor/qualified_reactions_v1.json`
(sha256 `296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf`).
Forward enumeration is capped at 128 products per reagent triple.

### 2.2 Why this reaction is a good test case

The Ugi lipids are a genuine reaction-defined family. Delivery performance is dominated by
fine structural detail in the *tails* — chain length, branching, degree and position of
unsaturation, ester placement — which lives inside the aldehyde and isocyanide reagents rather
than in the gross topology of the product. A chemist orders reagents, not products. So the
factorization is the operative object, which is exactly the claim the representation makes.

---

## 3. The data

### 3.1 Measured lipids

**1,100 single-compound Ugi lipids with measured transfection**, from AGILE's published data,
cross-validated against the independently curated LANTERN release.

- File: `results/m0_07/agile_oracle_curated.csv.gz`
  (sha256 `9415f1c0f40399f87d3df00c0c83a341a3776ca110e6bc98763fdcbd420a27be`)
- Endpoints: `expt_Hela` (HeLa transfection) and `expt_Raw` (RAW 264.7). **Only HeLa was used
  for candidate ranking; RAW had zero weight.**
- Provenance field on all rows: `hela_label_source = AGILE_official_source_data_cross_validated_by_LANTERN`
- Of 1,200 nominal source records, 1,100 resolved to exact single compounds and were kept;
  100 mixture-derived executions were retained as provenance but excluded from single-graph
  supervision.

Distinct components across those 1,100 lipids: **20 amine heads, 11 aldehyde tails, 5
isocyanide tails.** This is the "measured" reference set and must not be confused with the
enumerated library below.

### 3.2 The enumerated library

AGILE's virtual candidate library is the complete cross-product of **22 amines × 62 aldehydes
× 9 isocyanides = 12,276 products**.

- File: `results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz`
  (sha256 `2cb7861b2d284eb3262f58d9c7ed30c186d04e047cf33fa7356b470a50904fdd`)

This is the reachable set of any method that composes those blocks, which is why it is the
right yardstick for asking whether FORGE left the enumerable space.

### 3.3 Stereochemistry

The source data **does** carry stereochemistry, and FORGE deliberately discards it for
modelling while retaining it for provenance.

- `structure_policy = single_compound_constitutional_graph_with_isomeric_provenance` on all
  1,100 rows.
- `model_smiles` (model-facing): 0 of 1,100 carry a stereo marker.
- `isomeric_smiles` (provenance): 460 of 1,100 carry one.
- Verified: the reconciliation step **added a stereo marker to 0 rows**. It canonicalizes and
  splits mixtures; it never invents geometry.

Geometry in the measured library follows the feedstock, not a convention:

| Component | Formula | Geometry | Occurrences |
|---|---|---|---|
| oleyl isocyanide `[C-]#[N+]CCCCCCCC/C=C\CCCCCCCC` | C19H35N | **cis (Z)** | 220 |
| oleate-derived aldehyde tail | C24H44O3 | cis | 100 |
| linoleate-derived aldehyde tail | C24H42O3 | cis/cis | 100 |
| 2-enoate aldehyde tail | C16H28O3 | **trans (E)** | 100 |
| undecylenate (terminal alkene) | C17H30O3 | none to assign | 100 |

Totals across measured components: **520 cis, 100 trans, 100 unspecified**. Every *internal*
olefin is cis. The single trans is a conjugated 2-enoate, where trans is simply what that acid
is.

**Consequence for the generated designs:** FORGE emits constitutional structures only, so no
generated candidate specifies E or Z. This is a stated modelling boundary, not an oversight —
the activity predictor also never saw stereochemistry, so predictions do not depend on it. In
practice the chemist should make internal olefins **Z**, matching every internal olefin in the
measured library.

### 3.4 The training corpus

The generator is not trained on 1,100 lipids. It is trained on a much larger corpus of
**112,386 Ugi products** assembled by expanded reaction-enumerated support.

Splits are at the level of **component families**, so a held-out product cannot leak back into
training through a reused reagent:

| Fold | Products |
|---|---|
| train | 66,464 |
| calibration | 15,800 |
| heldout | 30,122 |
| **total** | **112,386** |

- Assignments: `results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz`
  (sha256 `9a50c703ed6aa919b251678c04b7a44e93b3233a12fc13b5d6697683bb5b5184`)
- Component registry across the corpus: **264 amines, 107 aldehydes, 53 isocyanides**

Splitting *before* component extraction is the point. Product-level random splits would let a
model see a tail in training and be tested on a different product using the same tail, which
inflates every novelty and generalization number in this field.

---

## 4. The representation

### 4.1 The reaction-resolved state

A molecular graph is `G = (V, E)` with categorical atom labels and bond labels. FORGE
generates

```
p_theta(G | c),    c = (c_amine, c_aldehyde, c_isocyanide)
r_c : serialized position → {amine, aldehyde, isocyanide, core}
```

where `r_c` is the **role map induced by the design program**, and is deterministic: the
program declares per-role atom counts and the serialization lays those roles out in contiguous
blocks in that order. Verified over 9,000 records across all three folds with zero mismatches
(`results/phase1/forge_production_role_channel_audit_v1/result.json`). Every atom and every
bond therefore carries the reagent role it
originated from, or the core.

The **decomposition** `D(x)` is obtained by restricting `G` to each origin class and closing
the induced cut with the valences the adapter specifies. Because `o` is part of the sampled
state, `D` is a total function computable in `O(|V| + |E|)`. No catalogue, no template
library, no learned retrosynthesis model is consulted at any point.

### 4.2 Serialization

The graph is serialized as a deterministic **spanning tree anchored at the core**, plus a
sparse set of residual **closure edges**. Each node carries:

- an offspring count (how many children)
- an atom state
- a parent-bond state
- where applicable, closure-bond and terminal-decoration states

The role region travels with the node, so role information is present at every step of the
generative trajectory rather than only at the end.

Implementation: `src/forge/product/ugi_joint_sparse_flow.py`,
`src/forge/product/ugi_morphology_program.py`,
`src/forge/product/v5_sparse_representation.py`.

### 4.3 Declared support

The state space is bounded explicitly, by counting rather than by lookup:

| Bound | Value |
|---|---|
| exterior atoms per product | ≤ 80 |
| atoms per component | ≤ 32 |
| children per node | ≤ 3 |
| junction budget | ≤ 7 |
| cycles per component | ≤ 2 |
| core attachments | ≤ 2 |
| decorations | ≤ 7 |
| bond classes | 4 |

Structures outside these bounds are *defined to be outside the model's support*, not generated
and then filtered.

The distinction from a catalogue matters and is the crux of the paper's central argument. A
building-block method is constrained by a **vocabulary**: a finite list, whose closure is
enumerable (12,276 products, in our case). FORGE is constrained by a **bound** on size,
branching, cycle rank and attachment count. A bound names nothing, so a molecule can satisfy it
and still be one no one has ever written down.

### 4.4 Conditioning carries no identity

The model receives only coarse per-role morphology: exterior atom count, junction budget,
cycle rank, core-attachment count. It receives **no component identities, no fragment library
and no reaction templates**, at training or at sampling time. This is enforced in config
(`component_ids_enter_neural_tensors: false`) and is what makes "the reagents are generated"
a literal statement rather than a rhetorical one.

---

## 5. The generative model

### 5.1 Method

**Discrete flow matching** over the categorical state. For each categorical variable `u`:

```
q_t(z_u | x_1) = t·δ_{x_1[u]}(z_u) + (1−t)·p_0^{(o(u))}(z_u),    t ∈ [0,1]
```

with role-specific empirical source marginals `p_0^{(k)}` and `t` sampled uniformly on
[0.02, 0.98]. Variables are interpolated independently given `t`; the target is recoverable
only jointly.

The network is an **x-prediction denoiser**: given the corrupted state, the time and the
morphology conditioning, it predicts the clean categorical state of every head. Heads are
offspring topology, atom identity, parent-bond state, closure bonds, and terminal-decoration
anchors/atoms/bonds. The loss is role-balanced cross-entropy, weighted so smaller components
are not dominated by larger ones.

FORGE uses this machinery essentially unchanged. The contribution is the state it acts on.

### 5.2 Architecture

Deliberately small, so that representation and capacity do not get confused:

| | |
|---|---|
| backbone | 4 bidirectional recurrent layers (GRU) |
| hidden width | 192 |
| dropout | 0.15 |
| **trainable parameters** | **1,159,916** |

### 5.3 Training

Config: `configs/model/phase1_ugi_joint_sparse_balanced_v2_production_refit.json`

| | |
|---|---|
| seed | 20260805 |
| device | CUDA |
| optimizer steps | 1,700 (fixed, no early stopping) |
| batch size | 128 |
| learning rate | 1e-4 |
| weight decay | 5e-5 |
| gradient clip | 1.0 |
| checkpoints | 100, 500, 1000, 1500, 1700 |
| training folds | train + calibration + heldout (production refit) |
| checkpoint selection | `fixed_final_step` — **not** data-dependent |

The step count is not tuned. A development run selected step 1000 on 66,464 records; the
production refit preserves expected weighted draws per structural record across 112,386
records, giving 1,690.93 unrounded, rounded to 1,700. Diagnostics on all three folds are
in-sample monitoring only and were explicitly barred from checkpoint selection.

Sampling during training is source-stratified and role-family raked, 50/50 between the
current union and the expanded exact-forward enumeration.

### 5.4 Sampling

| | |
|---|---|
| reverse-star transitions | 8 |
| device | CPU |
| checkpoint used | 6 |
| draws per arm | 16,384 |
| particle seed base | 20261202 |
| terminal seed base | 20261203 |
| qualified morphology programs | 57,190 |
| schedule seed (common uniforms) | 2026080501 |

At every transition, **masks** restrict the move set: valence, duplicate-edge, self-edge,
adapter-boundary and counting masks. Each mask is a per-transition invariant, so an admissible
partial state stays admissible. The consequence is that FORGE **never generates an
out-of-support structure and discards it** — the declared support is a property of the sampler,
not a filter. This matters when reading the rejection rate below: rejections are chemistry
failures, never bookkeeping failures.

### 5.5 The two arms

The production run is a **matched two-arm experiment**, 16,384 draws each, 32,768 total, with
common particle and terminal randomness across arms:

- **`broad_prior`** — the unmodified morphology prior
- **`support_enriched`** — the reallocated proposal (§9)

Scope of that run, from its frozen config: 0 potency-model calls, 0 applicability-model calls,
0 oracle calls, 0 route calls, 0 synthesis calls, no guidance, no candidate selection, no
retries, no repairs.

---

## 6. Admission: what is checked and what is not

A completed sample is **admitted** only if five conditions hold
(`src/forge/bio/ugi_morphology_potency_matched_adjudication.py:93`):

```python
terminal["valid"]                                   # the product is a valid molecule
and terminal["terminal_valid"]                      # the sampled terminal state is well-formed
and terminal["component_reconstruction_valid"]      # the three pieces are valid reagents of their roles
and verification["exact_product_reconstructed"]     # forward Ugi on those three returns exactly this graph
and not verification["maximum_outcomes_saturated"]  # the forward enumeration did not hit its cap
```

Three things to be precise about, because they are easy to blur:

**This is a post-hoc gate.** Samples are generated and then tested. 7.9% fail. Admission is
not automatic and the paper should not imply it is.

**The decomposition itself is not post hoc.** `D(x)` is a read-off for every sample, including
rejected ones. No retrosynthesis model runs at any point. What the gate tests is whether that
read-off decomposition is chemically legitimate and actually reconstitutes the product.

**Ambiguous verification counts as failure.** If the forward enumeration hits its 128-outcome
cap, we cannot be sure the product was not among the outcomes never enumerated, so the sample
is rejected. The certificate therefore has no false positives.

**What admission does NOT assert:** that any reagent can be bought, that any reagent can be
made, that the reaction will work at the bench, or anything about yield or purity. A design can
clear all five conditions while containing an aldehyde that exists nowhere on earth.
Procurement is a separate and much later question (§10).

---

## 7. Generation results

Recomputed directly from
`results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz`
(sha256 `de51ceb6d452a59f87f1210ce15c9700c562c4bf86105ac1446437e73f280240`, 32,768 rows).

### 7.1 Admission

| Arm | Attempts | Admitted | Rate |
|---|---|---|---|
| broad_prior | 16,384 | 15,055 | 91.89% |
| support_enriched | 16,384 | 15,125 | 92.32% |
| **total** | **32,768** | **30,180** | **92.10%** |

**Distinct admitted products: 26,235.** (30,180 admitted *rows* collapse to 26,235 unique
molecules; the paper's "26,235 distinct lipids" is the deduplicated count and the two must not
be interchanged.)

Rejection reasons, broad arm: 853 `invalid_or_nonexact_l1`, 476 `handle_policy_failure:amine_head`.
Support-enriched: 936 and 323.

### 7.2 Novelty

Denominator: 26,235 distinct admitted products. **Which reference set is used changes the
number by a lot, and conflating them is the single most common error in describing this work.**

| Reference set | Absent from it | Fraction |
|---|---|---|
| **everything the production generator trained on** (112,386 products) | 21,960 | **83.7%** |
| **training fold only** (66,464 products) | 23,342 | 89.0% |
| **AGILE's 12,276-member enumerated library**, compared stereo-free both sides | 22,645 | **86.3%** |
| carries ≥1 component absent from the **training corpus registry** (264/107/53) | 19,520 | 74.4% |
| carries ≥1 component absent from the **measured 1,100** (20/11/5) | 25,982 | 99.04% |

> **CORRECTION, 2026-08-15.** Both manuscript drafts report 89.0% as the headline novelty
> figure, measured against the 66,464-product train fold. That denominator is wrong for these
> designs. The production generator was refit on **all three folds** — the config sets
> `training_folds: [train, calibration, heldout]` and `production_training_records: 112386` —
> so the 26,235 designs came from a model that had seen all 112,386 corpus products, not
> 66,464. Against what the model actually trained on, novelty is **83.7%** (21,960 / 26,235),
> recomputed here from `assignments.csv.gz` and the terminal ledger.
>
> The 89.0% figure is only defensible for a model trained on the train fold alone. Two clean
> fixes: report 83.7% for the production model, or generate a separate evaluation run from a
> train-fold-only checkpoint and report 89.0% for that model, keeping the two runs and their
> claims separate throughout.
>
> The **86.3% against the enumerated library is unaffected** and remains correct, because that
> reference set is external to training entirely. It is also the figure the paper's central
> argument actually rests on.

The 86.3% is the load-bearing one for the paper's argument, because the enumerated library is
precisely the reachable set of any block-composition method over those blocks. Those 22,645
molecules could not have been produced by recombining that block set at any budget.

### 7.3 Component expansion

Distinct component identities among designs entering activity evaluation, against the measured
library:

| Role | Measured (1,100 lipids) | Generated | Expansion |
|---|---|---|---|
| amine head | 20 | 23 | 1.15× |
| **aldehyde tail** | **11** | **163** | **14.8×** |
| isocyanide tail | 5 | 15 | 3.0× |

The expansion is overwhelmingly in the aldehyde tail, which is where delivery-determining
chemistry lives.

### 7.4 Structural composition

Over the 26,235 distinct admitted products:

| Feature | Count | Fraction |
|---|---|---|
| unsaturated in either tail | 11,351 | 43.3% |
| ester-containing tail | 18,565 | 70.8% |
| branched in either tail | 8,294 | 31.6% |
| branched isocyanide | 7,340 | 28.0% |
| **branched within the aldehyde-derived tail** | **1,441** | **5.5%** |
| unsaturated aldehyde only | 9,490 | 36.2% |
| unsaturated isocyanide only | 2,885 | 11.0% |

The aldehyde-branching figure is the interesting one: the model must construct a branch point
inside the tail rather than select a pre-branched block.

Source of §7.2–7.4: `scripts/phase1_compute_manuscript_numbers_v1.py` →
`results/phase1/ugi_manuscript_numbers_v1/result.json`
(result sha256 `32e1e00eb3e792562557a7ca584c1d66642cdd5e89a588c685ef14bfc6d433eb`).

---

## 8. The activity predictor

A separate model from the generator, trained only on the 1,100 measured lipids.

| | |
|---|---|
| architecture | role-aware directed message-passing neural network (D-MPNN) ensemble |
| ensemble seeds | 1729, 11729, 21729 |
| endpoints | expt_Hela (used), expt_Raw (zero weight in ranking) |
| stereochemistry | not used |
| graph truncation | not allowed |
| checkpoint | `results/m0_07/oracle_production_checkpoint.pt`, hash required before inference |

**Performance, and it is not good.** Calibration R² = 0.504; descriptive post-selection test
R² = 0.253 (ρ = 0.612). Under component-aware five-fold cross-validation with held-out aldehyde
families (`results/phase1/ugi_comparator_baselines_v1/out_of_fold_metrics.json`, n = 1,100,
seed 1729):

| Model | MAE | RMSE | Pearson | Spearman |
|---|---|---|---|---|
| label-shuffled null | 2.823 | 3.339 | −0.235 | −0.217 |
| **role-aware D-MPNN (production)** | — | — | — | **0.469** |
| component-additive ridge | 2.137 | 2.794 | 0.530 | 0.469 |
| Morgan extra-trees | 2.455 | 3.280 | 0.472 | 0.451 |
| nearest neighbour | 2.510 | 3.285 | 0.413 | 0.378 |

Read that table honestly: a **component-additive ridge regression matches the neural
ensemble** on held-aldehyde-family generalization. This is the correct reason the predictor is
used as a *selective ranking* model rather than an oracle, and it is why the conformal
machinery in §9 exists at all. Anyone writing about this work should not describe the predictor
as strong.

---

## 9. Prediction under generator-induced shift

### 9.1 Two different supports

**98% of admitted designs contain at least one component that has never been measured.** So a
predictor fit on 1,100 lipids is extrapolating on essentially everything FORGE makes, and it
gets worse the better the generator is.

FORGE treats two sets as distinct:

- **generative support** `S_gen` — the declared bounds of §4.3
- **predictive support** `S_pred` — where the measured data can credibly rank

`S_pred` requires a similarity criterion to the measured library to hold **jointly** for the
complete molecule *and* for each of its three components — four views, all must pass. Distances
are a radius-2 count-Morgan fingerprint distance and a robust physicochemical-descriptor
distance.

Thresholds are derived from held-component calibration with the held component and every
product containing it removed from the reference sets first. From
`results/phase1/ugi_distributional_applicability_v3/result.json`:

| Role | fingerprint interpolative / boundary | descriptor interpolative / boundary | calibration records |
|---|---|---|---|
| amine | 0.537 / 0.846 | 0.364 / 1.090 | 80 |
| aldehyde | 0.186 / 0.330 | 0.262 / 0.541 | 44 |
| isocyanide | — / 0.267 | 0.471 / 0.695 | 20 |

Both obvious responses fail, and the paper says so. Restricting *generation* to `S_pred`
rebuilds the ceiling FORGE exists to remove, since `S_pred` is defined by proximity to the
measured library. Scoring everything equally takes extrapolation at face value.

### 9.2 Stratified conformal ranking

The decomposition supplies a stratification variable that a product-only generator does not
have: **which reagent role is unseen.** Split conformal calibration is performed within each
stratum on residuals `|y − ŷ|` over a calibration fold disjoint from training, and candidates
are ranked by

```
LCB90(x) = oracle_mean(x) − q90(stratum(x))
```

Fitted quantiles, recomputed from the ledger:

| Authority tier | Unseen role(s) | q90 | Calibrated rows |
|---|---|---|---|
| `qualified_role_holdout` | isocyanide only | **3.9147** | 643 |
| `exploratory_weak_absolute_head_generalization` | amine head | **4.2959** | **42** |
| `exploratory_weak_absolute_aldehyde_generalization` | aldehyde tail | **5.3094** | 1,858 |
| `exploratory_simultaneous_exact_new_tails` | aldehyde + isocyanide | **6.2145** | 1,643 |

The width increases monotonically with distance from measured chemistry, which is the
behaviour one wants. Two caveats belong with the number every time it is quoted: the
head-generalization stratum rests on **42 records**, so its quantile is far less precisely
estimated than its width suggests; and exchangeability holds *within* a stratum, with
unseen-role structure a coarse proxy for the actual shift.

Three further tiers exist and carry **no** conformal quantile, so `LCB90` is undefined for
them and they are excluded from candidate selection: `exploratory_multiple_exact_new_roles`,
`exploratory_all_roles_exact_new`, `qualified_unseen_combination`.

### 9.3 Potency-free proposal reallocation

Rather than restricting generation, FORGE moves proposal mass toward architectures more likely
to terminate in rankable chemistry:

```
π'(c) ∝ π(c)·(ρ(c) + ε)^γ,    ρ(c) = Pr[x ∈ S_pred | c]
```

`ρ̂` is computed from the frozen support predicate alone. **No measured activity and no
predicted potency enters this equation** — the predictor appears nowhere in it. Because ε > 0
and γ is finite, `π'(c) > 0` whenever `π(c) > 0`: no architecture is removed from the support,
so the reallocation changes only where samples are spent.

**The headline number and exactly where it comes from.** This is the one figure in the paper
most likely to be mis-sourced, so the trace is given in full.

`results/phase1/ugi_morphology_proposal_challenger_confirmation_analysis_v1/result.json`,
on **3,072 fresh, previously unused morphology programs**, with support defined as
`valid_exact_l1_and_fixed_multiview_radius_at_most_one`:

| | |
|---|---|
| prior (broad) expected support rate | **0.038737 → 3.9%** |
| reallocated proposal expected support rate | **0.086422 → 8.6%** |
| absolute improvement | 0.047685 |
| relative improvement | 1.2310, i.e. **2.23× baseline** |
| valid exact-L1 rate, prior vs proposal | 0.9688 vs 0.9626 |

Observed under the prior on that fresh set: 119 supported of 3,072 attempts, 117 unique SMILES.
Support concentrates sharply in the top score quartile (11.46%, 4.04%, 0.00%, 0.00% across
quartiles 1–4).

A **development sweep** (`ugi_morphology_proposal_strength_sweep_v1`) preceded this and
selected the challenger `rho-0.50_power-1.50` from a grid, subject to twelve frozen floors on
program diversity, role marginals, supported-component effective counts and valid-exact-L1
rate. That sweep explicitly did **not** authorize production replacement
(`production_replacement_authorized: false`), which is why a separate fresh confirmation on
untouched programs exists at all.

**Do not confuse this with the production arms.** The 32,768-draw production run gives a
different, larger separation on a different support definition (`oracle_scored`, which also
requires calibration eligibility):

| Arm | oracle-scored / attempts | oracle-scored / admitted |
|---|---|---|
| broad_prior | 1,433 / 16,384 = 8.75% | 9.52% |
| support_enriched | 2,900 / 16,384 = 17.70% | 19.17% |

Both pairs of numbers are real and both are computable from frozen artifacts. They measure
different quantities on different populations. The paper quotes the 3.9 → 8.6 pair because it
is the prespecified fresh confirmation of the mechanism.

Global unique oracle calls across the production run: **2,590**.

---

## 10. Route resolution

Exact decomposition says which reagents a design needs. It says nothing about whether they
exist.

Among the 2,590 designs retained for ranking there are **201 distinct components**, and heads
and tails behave completely differently: nearly every amine head is purchasable, while almost
no generated aldehyde or isocyanide tail is. That is the expected cost of generating reagents
at the atom-and-bond level.

### 10.1 The two preparations

Both are published chemistry applied to generated structures, not invented routes:

- **Aldehyde tails ("AGILE Tail A")**: carboxylic acid + α,ω-diol → esterify → oxidise. 2 steps.
- **Isocyanide tails**: primary amine → formylate → dehydrate. 2 steps.

### 10.2 Route completeness

Fix a dated procurement snapshot `P` and a registry `T` of documented preparations. A component
`m` is **route-complete at depth ≤ L** iff either `m ∈ P`, or there is a preparation
`m ← {m₁…m_r}` in `T` such that running the reaction *forward* on `{m₁…m_r}` reproduces `m`
**uniquely**, and every `m_j` is route-complete at depth ≤ L−1. A product is route-complete iff
its forward reconstruction is exact and all three components are route-complete.

Where the registry is silent, single-step proposals come from **Graph2Edits** (Zhong et al.,
*Nat. Commun.* 14:3009, 2023) and multistep search from **AiZynthFinder** (Genheden et al.,
*J. Cheminform.* 12:70, 2020), both unmodified. Neither is a contribution of the work; what the
framework contributes is the object they run on, since the recursion is over generated
*components* rather than over a product whose disconnection must first be guessed.

Two constraints make this evidence rather than a planner's opinion:

- **Forward verification at every accepted step.** A disconnection is accepted only if running
  the reaction forward reproduces the intended component.
- **A planner score or reaction-family assignment is never sufficient alone.**

Unresolved components are recorded as *unresolved under the current procurement snapshot*,
never as unsynthesizable. The distinction is load-bearing: the dominant observed failure is a
**missing positional isomer in a supplier catalogue**, not a chemical impossibility.

### 10.3 Result

**97.1% of prediction-supported designs resolve to purchasable starting material.**

Note the denominator carefully. This is 97.1% of *prediction-supported* designs, not of all
26,235 admitted ones. Most admitted designs never receive a procurement verdict at all,
because routing is only run on designs that would actually be ranked.

### 10.4 What route completeness does not claim

Nothing about yield. Nothing about purity. Nothing about whether a reaction will succeed in a
flask. And it rests on a dated snapshot, so a discontinued supplier falsifies it without the
molecule changing.

---

## 11. The prospective panel

Config: `configs/bio/phase1_ugi_prospective_panel_v6.json`, frozen 2026-08-06 **before**
selection was run.

### 11.1 Eligible pool

Every design that satisfies all of:

- terminal chemical admission
- oracle-scored with a parseable calibrated `LCB90`
- absent from the 66,464-product train fold
- inside the frozen physicochemical envelope
- route-complete in ≤ 4 synthetic steps

**Realised pool size: 1,367 candidates**, composed of 653 aldehyde-generalization, 540
dual-new-tail, 114 qualified-role-holdout, 24 head-generalization.

### 11.2 Selection rule

- **Ranking statistic:** `LCB90 = oracle_mean − conformal_q90`. Ranking by raw mean instead
  was measured: it changes 9 of 34 selected candidates, gains 0.18 in median predicted potency,
  and drops the `LCB90` floor from 0.95 to 0.09. Rejected.
- **HIGH arm, 34:** the first 12 taken on score alone; the remaining 22 preferring chemistry
  outside the source library where novelty costs no predicted activity.
- **LOW arm, 6:** lowest-ranked designs drawn from the *same* evidence tier, so the comparison
  is not confounded by evidence class.
- **Diversity:** pairwise whole-product Tanimoto ≤ 0.95, and at most 4 candidates per amine
  head. An earlier 0.80 ceiling was measured to cost 1.78 in median predicted potency and pull
  the floor from beating 74% of the measured library down to 27%, while buying no extra
  aldehyde diversity (23 either way). The head cap carries the diversity principle onto the
  axis that actually matters.

The config documents the AGILE precedent it adopts (component identity, not a fingerprint
threshold, as the diversity currency) and what it explicitly declines to adopt (round-robin
across every head group, which reproduced here yields 22 heads but a floor beating only 9% of
the measured library).

### 11.3 The panel

`results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz`
(sha256 `d9b7c0a70c8d8528…`)

| | |
|---|---|
| candidates | 40 (34 HIGH, 6 LOW) |
| HIGH `LCB90` range | 2.047 – 8.895 |
| HIGH predicted mean range | 6.37 – 14.20 (median 9.07) |
| HIGH above the measured 90th percentile | 18 of 34 |
| LOW `LCB90` range | −5.463 – −5.083 |
| distinct amine heads / aldehydes / isocyanides | 17 / 26 / 7 |
| absent from the train fold | 40 of 40 |
| absent from the 12,276 enumerated library | 32 of 40 |
| maximum preparative steps before final Ugi | 4 |

### 11.4 Chemist review, 40 → 12

The 40 are the **locked computational pool**. A chemistry review selects **12** for synthesis
*before any experimental outcome exists*, on documented criteria (predicted activity, route
feasibility, structural diversity across head and tail architectures). The full 40 remain
visible in the paper. The 12 have not yet been chosen at time of writing.

Statistical consequence worth stating plainly: the HIGH-vs-LOW comparison was designed at
34 vs 6. At 12 candidates it becomes roughly 10 vs 2, which cannot support the binomial
comparison the Methods describe. This is an open issue, not a solved one.

### 11.5 A worked example (panel P07, packet code L01)

```
product      CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCCCC)NCCCN(CCCC)CCCC
             MW 734.25, 4 steps, 2 tails to prepare

amine head   CCCCN(CCCC)CCCN                     purchase, 62 suppliers
aldehyde     CCCCCCCCCCC(=O)OCCCCCC=O            synthesise, AGILE Tail A, 2 steps
             ← undecanoic acid (101 suppliers) + hexane-1,6-diol (93 suppliers)
isocyanide   [C-]#[N+]CCCCCCCCC=CCCCCCCC         synthesise, from primary amine, 2 steps
             ← heptadec-9-en-1-amine, C17H35N (3 suppliers)
```

This example also illustrates a real limitation in action. AGILE's unsaturated isocyanide is
oleyl isocyanide, C19H35N, from oleylamine (C18). P07's is the **C18 isocyanide from a C17
amine** — one CH₂ shorter, and not an AGILE building block, which is why it has 3 suppliers
rather than many. Binary Morgan fingerprints are blind to alkyl chain length (30 pairs measured
at Tanimoto 1.000 differing by 14–182 Da), so the predictor could not distinguish the shorter
homologue from oleyl and had no signal steering it toward the standard block. Fourteen other
panel candidates do specify oleylamine and reproduce AGILE's C19 isocyanide exactly.

---

## 12. Every headline number, with provenance

| Claim | Value | Source |
|---|---|---|
| samples attempted | 32,768 (2 arms × 16,384) | terminal_rescoring ledger |
| admission rate | 92.10% | ledger, recomputed |
| distinct admitted products | 26,235 | ledger, recomputed |
| novel vs training fold | 89.0% | manuscript_numbers_v1 |
| outside the 12,276 enumerated library | 86.3% | manuscript_numbers_v1 |
| ≥1 component absent from training registry | 74.4% | manuscript_numbers_v1 |
| ≥1 component never measured | 99.04% | manuscript_numbers_v1 |
| aldehyde role expansion | 11 → 163 (14.8×) | manuscript_numbers_v1 |
| unsaturated / ester / branched tails | 43.3% / 70.8% / 31.6% | manuscript_numbers_v1 |
| branching inside the aldehyde tail | 5.5% | manuscript_numbers_v1 |
| rankable fraction, prior → proposal | 3.9% → 8.6% (2.23×) | challenger_confirmation_analysis_v1, n = 3,072 fresh |
| conformal q90 by tier | 3.915 / 4.296 / 5.309 / 6.215 | ledger, recomputed |
| unique oracle calls | 2,590 | rescoring result |
| distinct components among ranked designs | 201 | rescoring |
| route-complete among prediction-supported | 97.1% | route assessment |
| eligible pool for the panel | 1,367 | panel v6 config |
| panel | 40 (34 high / 6 low) | panel v6 artifact |
| generator parameters | 1,159,916 | production checkpoint |
| corpus | 112,386 (66,464 / 15,800 / 30,122) | corpus v2 assignments |
| predictor calibration / test R² | 0.504 / 0.253 (ρ = 0.612) | oracle production |

---

## 13. What has not been done

State these plainly; they are not hidden in the repository and should not be hidden in an
article.

**No wet-lab work of any kind.** No synthesis, no LNP formulation, no cell assay, no animal
study, no editing experiment. Every biological statement is a prediction or a plan.

**No matched generative baseline over the product graph alone.** The comparison is against the
*closure of a block set*, which does bound any composition method over those blocks, but it is
not a trained discrete flow over the product graph with post-hoc decomposition. Building one is
a second generative formulation rather than an ablation of this one, because role structure is
carried by the design program and the state layout and cannot be removed without changing the
modelled object.

**No origin-channel ablation, and none is available.** Verified 2026-08-16: the production model
`UgiJointSparseFlow` has no origin embedding, and role is an exact deterministic function of the
design program over 9,000 records spanning all three folds. Deleting the model's 576-parameter
`role_embedding` would remove a redundant re-encoding of the conditioning, not role information,
so it cannot test whether precursor-role information helps generation. Artifact:
`results/phase1/forge_production_role_channel_audit_v1/result.json`.

**No pooled-versus-stratified conformal comparison.** Whether stratification earns its
complexity is untested.

**No novelty/diversity measurement under reallocation.** Support preservation is guaranteed by
construction, but distributional drift *within* the support is not excluded and has not been
measured.

**One reaction, one adapter.** The formalism admits any fixed-arity single-step reaction with a
fixed core. All evidence is Ugi three-component chemistry. Multi-step assembly is excluded by
construction.

**An internal joint-versus-staged generator comparison exists but proves nothing.** Both arms
reach exact-program fraction 1.0 with zero terminal tree repairs, but the artifact's own frozen
policy records `smoke_run_supports_paper_level_superiority_claim: false` and
`selection_status: insufficient_for_final_selection_smoke_only`. It must not be cited as
evidence of superiority.

---

## 14. Traps and common misreadings

Collected because each has already caused an error in this project.

**Four different novelty denominators exist.** Train fold (89.0%), enumerated library (86.3%),
training component registry (74.4%), measured 1,100 (99.04%). Always name the denominator.
Quoting a novelty figure without one is meaningless.

**"Measured library" ≠ "enumerated library."** 1,100 measured lipids with 20/11/5 components,
versus a 12,276-product cross-product of 22/62/9 blocks. Different sets, different roles in the
argument.

**Admitted rows ≠ distinct products.** 30,180 rows, 26,235 molecules.

**97.1% is a fraction of prediction-supported designs**, roughly 8.6% of admitted ones, not of
everything generated.

**3.9% → 8.6% comes from a 3,072-program fresh confirmation**, not from the 32,768-draw
production run. The production arms give 8.75% → 17.70% on a different support definition.
Both are correct; they are not interchangeable.

**"Exact decomposition" is not "makeable."** Admission tests that the reagents rebuild the
product. It says nothing about buying or making them.

**The generator is post-hoc filtered, and that is fine.** 7.9% of samples are rejected. What is
*not* post hoc is the decomposition. Claims of the form "nothing is recovered afterwards" must
be scoped to the factorization or they are wrong.

**No generated structure carries stereochemistry.** The model never represented it and the
predictor never saw it. Internal olefins should be made Z by AGILE precedent, but the design
does not commit.

**Fingerprints are blind to chain length.** Binary folded Morgan fingerprints gave 30 pairs at
Tanimoto 1.000 differing by 14–182 Da. Any similarity-based claim inherits this.

**Conformal coverage is within-stratum and one stratum has 42 records.** The monotone widening
of quantiles is a property of the fit, not a theorem.

**The predictor is weak, and a ridge regression matches it.** Conformal calibration makes the
weakness explicit; it does not remove it.

**Chemical-space embeddings were tried and abandoned.** t-SNE shattered into artefactual
islands; UMAP over Jaccard gave silhouette −0.027 (binary Morgan), −0.021 (count Morgan),
−0.034 (descriptors). At equal n = 1,100 the convex hull areas of measured, enumerated and
generated sets are 1484.9 / 1491.2 / 1385.0 — statistically identical — while median
nearest-neighbour distance is 0.026 / 0.037 / 0.086. The correct interpretation is that
generated designs **densify the same envelope 3.3× more finely** rather than extending beyond
it in any 2D projection. The real expansion is at the component level, which is why the paper
reports component counts rather than a UMAP.

---

## 15. Where things live

| What | Path |
|---|---|
| generator (joint sparse flow) | `src/forge/product/ugi_joint_sparse_flow.py` |
| chemistry flow | `src/forge/product/ugi_chemistry_flow.py` |
| serialization / morphology programs | `src/forge/product/ugi_morphology_program.py` |
| admission predicate | `src/forge/bio/ugi_morphology_potency_matched_adjudication.py:93` |
| terminal rescoring | `src/forge/bio/ugi_production_full_support_rescoring_v3.py` |
| applicability | `src/forge/bio/ugi_distributional_applicability_v3.py` |
| conformal | `src/forge/bio/ugi_interpolative_conformal.py` |
| route assessment | `src/forge/product/ugi_terminal_route_assessment.py` |
| AGILE reconciliation | `src/forge/bio/agile_reconciliation.py` |
| panel selection | `scripts/phase1_select_ugi_prospective_panel_v6.py` |
| manuscript numbers | `scripts/phase1_compute_manuscript_numbers_v1.py` |
| production training config | `configs/model/phase1_ugi_joint_sparse_balanced_v2_production_refit.json` |
| panel config | `configs/bio/phase1_ugi_prospective_panel_v6.json` |
| chemist packet | `manuscript/chemist_packet/FORGE_candidate_smiles.csv` |

Every frozen artifact carries an input manifest with SHA-256 values for its config, runner,
source module and tests, so any number here can be traced to the exact code and inputs that
produced it.
