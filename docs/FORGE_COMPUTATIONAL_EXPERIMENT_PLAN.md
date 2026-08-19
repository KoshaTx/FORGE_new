# FORGE Computational Experiment Plan

## 1. Scope decision

FORGE will be presented as a computational methods paper, at the same level of empirical claim as
TD3B. The paper will not require prospective synthesis, formulation, or biological assays.

The central claim is:

> Reaction-derived roles can organize an open-ended molecular state without factorizing generation;
> allowing information to pass among precursor-derived regions enables FORGE to learn dependencies
> that independently generated regions cannot represent.

The decisive missing evidence is therefore a comparison with a **learned component-factorized
generator**. The existing role-marginal null remains a useful diagnostic, but it is not an adequate
test of this claim because it has no learned contextual model.

## 2. Evidence already available

The following results can be retained and reorganized rather than rerun unless an audit identifies a
provenance or implementation problem:

- The aligned/no-role/misaligned semantic ablation shows that an explicit, correct reaction-role map
  improves finite-capacity denoising.
- The role-marginal null shows that the declared support and one-variable marginals alone do not
  reproduce lipid-native organization.
- Production sampling demonstrates generation outside both the structural corpus and the finite
  precursor enumeration.
- Exact component projection and forward reconstruction establish reaction compatibility within the
  declared support.
- Computational route qualification and evidence-aware context reweighting illustrate downstream
  deployment without altering the trained generator.

These results support the surrounding method. They do not establish that joint generation is better
than a learned factorized alternative.

## 3. Required experiment: joint versus learned factorized generation

### 3.1 Question and estimand

The experiment asks:

> After conditioning on the same architectural context, does cross-role communication improve the
> learned distribution over complete products relative to independently generated precursor regions?

The primary held-out estimand is

\[
\Delta_{\mathrm{joint}}
= L_{\mathrm{factorized}}-L_{\mathrm{FORGE}},
\]

where both losses are evaluated on the component-family-held-out products under the same corruption
distribution. Positive values favor joint modeling.

### 3.2 Preliminary diagnostic: does conditional dependence exist?

Before training the factorized model:

1. Project every held-out product to its amine-, aldehyde-, and isocyanide-derived regions.
2. Within each exact architectural context \(C\), independently permute the three roles across
   products.
3. Reassemble the shuffled regions through the declared Ugi transform.
4. Compare intact and shuffled products with a cross-validated classifier and prespecified
   cross-role descriptor statistics.
5. Group all train/test partitions by component family so that repeated components cannot leak across
   folds.

This manipulation preserves the empirical conditional marginal of every role while destroying
dependence among roles. If intact and shuffled products cannot be distinguished, the present corpus
does not provide an empirical reason to prefer joint generation under the chosen context. If they can
be distinguished, the main experiment tests which model recovers that structure.

This diagnostic is evidence that conditional dependence is present; it is not itself evidence that
FORGE learns it.

### 3.3 Model arms

Train the following models:

#### Arm A: FORGE-joint

Use the current shared contextual denoiser over the complete serialized molecular state. Every role
can influence every other role through the recurrent state.

#### Arm B: FACT-matched

Use the same input embeddings, recurrent backbone, output heads, and parameter count as FORGE, but
prevent information from crossing reaction-role boundaries:

- Process each role block independently.
- Reset or mask the recurrent state at each role boundary in both directions.
- Supply every role with the **complete architectural context \(C\)**, not only its local counts.
- Retain the role identity and within-role coordinates.
- Share weights across the three role passes so that the trainable parameter count remains equal to
  FORGE.
- At sampling time, generate the three regions independently conditional on \(C\), place them around
  the fixed Ugi core, and apply the same projection, masks, and admission rule as FORGE.

This implements a learned distribution of the form

\[
q_a(X_a\mid C)q_d(X_d\mid C)q_i(X_i\mid C)
\]

without depriving the baseline of context, reaction semantics, or chemical support.

#### Arm C: FACT-generous

As a robustness control, train three independent role-specific denoisers, each with approximately the
same width and depth as the joint FORGE backbone. This arm intentionally receives more total
parameters and compute than FORGE. It addresses the objection that FACT-matched failed merely because
one matched parameter budget was divided across three roles.

The main comparison is Arm A versus Arm B. Arm C is a deliberately favorable upper-capacity check for
the factorized family.

### 3.4 Experimental parity

Hold the following fixed across the primary arms:

| Quantity | Required contract |
|---|---|
| Development data | Existing 66,464-product training partition |
| Calibration data | Existing 15,800-product partition |
| Test data | Existing 30,122 component-family-held-out products |
| Context | Complete \(C\) supplied to every arm |
| Serialization | Same graph-preserving, role-blocked representation |
| Corruption | Same source marginals, flow-time draws, and corrupted states |
| Objective | Same role-balanced endpoint-prediction loss |
| Support | Same valence, edge, boundary, counting, and Ugi admission rules |
| Optimizer | Same optimizer, schedule, clipping, batch size, and fixed training budget |
| Sampling | Same contexts, proposal, eight transitions, attempts, and admission procedure |
| Randomness | Paired initialization, minibatches, corruption draws, and sampling seeds |
| Replication | At least five paired training seeds |

Use a checkpoint rule fixed before inspecting the factorized results. Either evaluate the terminal
checkpoint under a fixed budget or select by the same calibration objective in every arm. Do not use
different stopping or model-selection criteria for different arms.

### 3.5 Primary evaluation

Evaluate endpoint-prediction loss on the held-component-family partition:

- aggregate loss over the same flow-time distribution used in training;
- loss at the fixed corruption grid already used by the semantic analysis;
- loss for each precursor-derived region;
- loss for each eligible prediction head;
- the paired difference for every seed.

Report the mean paired difference, every seed-level result, and a 95% confidence interval obtained by
resampling held-out component families and paired seeds. The primary comparison must be defined before
free-generation metrics are inspected.

### 3.6 Cross-role intervention

Measure whether FORGE actually uses the other molecular regions. For each target role \(k\):

1. Keep its corrupted state, flow time, and context fixed.
2. Replace the other two corrupted role blocks with blocks from different held-out products sharing
   the same \(C\).
3. Recompute loss on the unchanged target role.

Define

\[
\Delta_{\mathrm{swap},k}
=L_{\mathrm{FORGE},k}(\text{shuffled other roles})
-L_{\mathrm{FORGE},k}(\text{intact other roles}).
\]

A positive value shows that intact cross-role context helps predict the target region. By
construction, the factorized model should be invariant to this intervention; test that invariance as
an implementation check. Report the effect separately for amine, aldehyde, and isocyanide prediction
and across the fixed corruption-time grid.

### 3.7 Free-generation evaluation

For each seed and model, sample the same architectural contexts with the same attempt budget. Evaluate
three groups of outcomes.

#### A. Component-level competence

- Per-role validity and handle qualification.
- Per-role descriptor distributions.
- Per-role nearest-neighbor distributions to the training corpus.
- Existing ionizable-head and tail-chemistry metrics.

The factorized model must learn competent individual role distributions. If its marginals are grossly
worse, an inferior complete-product distribution cannot be attributed specifically to factorization.

#### B. Cross-role organization

Freeze a compact cross-role descriptor set before generation. It should include chemically meaningful
relationships between:

- head nitrogen environment and combined tail hydrophobic size;
- head heteroatom burden and tail carbon count;
- tail branching and head architecture;
- aldehyde- and isocyanide-derived tail lengths;
- tail unsaturation, linker chemistry, and whole-product amphiphilic balance.

Fit the dependence analysis on development data only. Residualize every descriptor against \(C\), form
the cross-role correlation matrix, and compare each model with the held-out real matrix using a fixed
matrix distance such as the Frobenius norm. Bootstrap component families to obtain uncertainty.

Also report the existing grouped real-versus-generated classifier AUC, but do not use it as the only
joint-fidelity metric because it can detect marginal differences unrelated to factorization.

#### C. Generative safeguards

- completion and decoding rate;
- Ugi admission;
- exact component projection and forward reconstruction;
- unique qualified assembly;
- effective diversity;
- novelty against the full structural corpus;
- novelty against the finite enumeration.

These are safeguards against explaining an apparent fidelity improvement through collapse, copying,
or rejection.

### 3.8 Decision rule and permissible claims

The strong joint-generation claim is supported only if:

1. Intact held-out products exhibit detectable conditional cross-role dependence relative to the
   within-context shuffled control.
2. FACT-matched learns competent per-role marginals.
3. FORGE has lower held-family denoising loss than FACT-matched, with the paired confidence interval
   excluding zero.
4. FORGE is more faithful to held-out cross-role organization than FACT-matched.
5. The result is not explained by worse admission, severe diversity loss, or copying.
6. The qualitative conclusion survives the FACT-generous comparison, or any failure against that arm
   is reported explicitly.

Interpret outcomes narrowly:

- **Loss and generated dependence both improve:** joint modeling provides the claimed empirical
  advantage.
- **Loss improves but generated dependence does not:** claim a denoising or representational benefit,
  not improved joint-distribution recovery.
- **Marginals fail in the factorized arm:** strengthen or retrain the baseline before interpreting the
  comparison.
- **No conditional dependence is detected in real data:** remove the empirical joint-dependence claim
  or adopt a context/data definition under which it is meaningfully testable, without choosing that
  definition using generated outcomes.
- **The matched arm loses but the generous arm does not:** report that the benefit is parameter- or
  sharing-efficient rather than an absolute limitation of factorized models.

## 4. Strongly recommended experiment: learned inventory selector

To distinguish open-ended atom-level generation from intelligent selection over known materials,
train a learned inventory baseline over the frozen 424-component registry.

The baseline should:

- condition on the same \(C\);
- select the amine, aldehyde, and isocyanide jointly or autoregressively rather than independently;
- assemble them through the same forward Ugi transform;
- use the same training split and sampling-context distribution;
- receive a documented and reasonable parameter/training budget.

Compare it with FORGE on:

- admission and unique assembly;
- lipid-native molecular metrics;
- supported-candidate yield;
- effective component and product diversity;
- route completeness;
- novelty outside the registry and finite enumeration.

Out-of-inventory novelty is zero for this baseline by definition. The useful question is whether
FORGE converts its additional structural reach into supported and route-qualified designs without an
unacceptable loss of fidelity or actionability.

This baseline is strongly recommended but secondary to the learned factorized comparison.

## 5. Optional analyses

These experiments are not required for the scoped computational paper:

- **Support-mask sampling ablation.** Sample the same trained checkpoint with generic graph-validity
  masks versus the complete reaction-specific mask set. Run this only if the paper claims an empirical
  quality benefit from the masks; the existing structural guarantee does not require an ablation.
- **Second reaction family.** Required only for an empirical cross-reaction generalization claim.
  Otherwise state clearly that the framework is formulated generally but evaluated only on Ugi-3CR.
- **External generic molecular generators.** These solve a different support problem and are less
  informative than the matched factorized and inventory baselines. Do not make comparative superiority
  claims without running them.
- **Additional semantic seeds.** The existing three-seed aligned/no-role/misaligned result can remain.
  Additional seeds would improve precision but are less important than the missing learned baseline.

## 6. Manuscript changes implied by the computational scope

Remove claims or placeholders that make wet-lab outcomes appear necessary:

- Remove the prospective-results placeholder from the abstract.
- Remove the prospective biological panel from the main evidence figure.
- Remove the prospective-results subsection from the main Results section, or convert it to a short
  future-work paragraph in the Discussion.
- Remove the blank prospective reporting table and unexecuted protocol from the appendix unless they
  are retained explicitly as an archival protocol, clearly outside the paper's evidence.
- Replace statements that the paper establishes experimental discovery with statements about
  computational generation, applicability-supported ranking, and route qualification.
- Describe route qualification as computational evidence of actionability, never as successful
  synthesis.
- Keep biological prediction as a deployment case study. It determines where ranking is permitted; it
  is not part of generator training and is not prospective biological validation.

The main Results sequence should become:

1. Correct reaction-role alignment improves finite-capacity denoising.
2. The structural corpus contains conditional cross-role dependence.
3. Joint FORGE recovers that dependence better than learned factorized generation.
4. Open-ended generation extends beyond the finite component inventory.
5. Exact projection, applicability support, context reweighting, and route qualification make the
   expanded space computationally deployable.

## 7. Reproducibility deliverables

For every new run, record:

- configuration and data hashes;
- split and component-family assignments;
- model parameter count and estimated training/sampling compute;
- initialization, training, corruption, sampling, bootstrap, and permutation seeds;
- checkpoint-selection rule and selected checkpoint;
- attempted, decoded, admitted, distinct, and evaluated sample counts;
- per-seed primary and secondary measurements;
- complete descriptor definitions and code version;
- failures and exclusions with their original denominators.

Add the new artifacts to the manuscript's artifact registry. The factorized implementation should
also include tests establishing that changing one role block cannot change another role's prediction
when \(C\) is fixed.

Separately, rederive and ledger-pin the predictor \(R^2\) values currently quoted from development
records. This is provenance repair, not a new scientific experiment.

## 8. Execution order

1. Freeze the cross-role descriptors, statistical endpoints, and checkpoint rule.
2. Run the within-context role-shuffling diagnostic.
3. Implement FACT-matched and its independence tests.
4. Train five paired FORGE/FACT-matched seeds.
5. Run held-family loss and cross-role intervention analyses.
6. Run matched free-generation comparisons.
7. Train and evaluate FACT-generous.
8. Train the learned inventory selector if resources permit.
9. Freeze artifacts and update the registry.
10. Rewrite the paper around the resulting computational evidence chain.

## 9. Minimum submission package

The minimum defensible submission consists of:

- the existing semantic ablation;
- the conditional-dependence diagnostic;
- the matched learned factorized comparison;
- the cross-role intervention;
- the existing open-endedness, projection, deployment, and route-qualification results;
- a rewritten manuscript with no implied wet-lab evidence.

The learned inventory selector and FACT-generous arm would make the submission substantially stronger.
Prospective biological experiments are outside the required scope of this version of the paper.
