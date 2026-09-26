# Continuous activity conditioning: executable study

2026-09-25. Implementation follows `ACTIVITY_CONDITIONED_GENERATION_PROPOSAL.md`, revision 2.
The user authorized execution. This study reuses previously inspected development folds;
it is exploratory and does not open protected biological identities or reservations.

## Comparisons

The frozen 22-family checkpoint initializes every trainable arm. The molecular target is
the measured constitution; its continuous source-normalized response is an input condition.
Within each fitting partition and assay context, affine standardization is fitted once and
shared by the denoiser and program prior. No response bins replace measurements.

Three matched denoisers receive genuine responses, responses permuted within assay context,
or hidden responses. They have identical structural and context capacity, optimizer budgets,
and replay inputs. Activity dropout preserves the assay context. The original generator is
a separate reference. Adapter-only and adapter-plus-final-block capacity are compared on
inner fitting data before the evaluation partitions are scored.

Anonymous reaction-program priors are original, label-hidden measured, continuous conditional,
or within-context shuffled. Hidden and continuous measured priors share exactly the same
fitting-program support. Program-only, denoiser-only, and joint interventions distinguish
activity control from structural adaptation. A hidden measured-prior control prevents
redistribution toward the assay library alone from being counted as activity guidance.

The continuous prior uses Gaussian response kernels with bandwidth candidates 0.5, 1, and 2
in fit-standardized units, and shrinkage candidates 1 and 5 effective designs. Selection uses
inner partitions only, reports unsupported held programs separately, and compares candidates
on common support. No prior receives precursor identifiers or exterior molecular states.

## Replay and numerical preflight

Before replay selection, freeze 16 distinct original TRAIN examples per family, selection seed
2026092701. Exclude all 1,131 biological-task identities, protected identities, and reservations.
Use original within-family probabilities for selection without replacement; retain fewer only
when a family has fewer eligible examples. Training replay samples families uniformly and uses
the original within-family probabilities renormalized over selected examples. This bounded
replay distribution is an explicit approximation to the original training distribution.

Shared structural adapters receive replay gradients. Biological conditioning is absent for
unlabeled replay, rather than represented as low or zero activity. Fixed coordinates, permitted
pointer alphabets, graph-size support, and exact assembly checks retain their meanings.
Ordinary conditional sampling is the initial intervention; no endpoint-logit interpolation is
presented as CTMC rate guidance.

A local CPU preflight measures training and generation cost and verifies initialization equality,
gradient routing, label-hidden invariance, checkpoint reload, and sampling compatibility. The
final run request will pin numerical budgets, decision margins, data, source, and checkpoints
before new evaluation scores are calculated. Preflight inputs are fitting records only.

## Independent evaluation

Use the previously fitted product-fingerprint control from `local_calibration_v1` as an
independent activity evaluator. Its coefficients, similarity cutoff, and calibration radii are
fixed; generated outputs cannot tune them. Product features avoid the disputed source-material
tail annotations. A versioned applicability check additionally requires each immediate assembly
precursor to occur in the evaluator's fitting partition, using exact-replayed annotations.
Original source-material descriptions remain separate. Unsupported outputs have no admitted
activity score and cannot enter post-hoc selection.

Completed-molecule selection compares deterministic applicability-only selection and post-hoc
prediction ranking with identical proposal and scorer budgets. Deduplicate selected constitutions,
record unfilled selection slots, and retain every failed proposal in yield denominators.
Measure exact assembly, distinct eligible yield, diversity, effective molecule count, and
training duplication alongside predicted response. Calibration establishes no guarantee on
arbitrary generated structures. Predicted response is never reported as measured transfection.

On held measured graphs, compare per-coordinate denoising losses under genuine, hidden, and
permuted responses at matched noisy states and times, including the terminal readout. These
are denoising losses, not molecular log likelihoods. Report reaction-program predictive scores
and support separately. Cell and study remain confounded in these libraries; A549 results do
not establish general lung delivery.

## Provenance and outcomes

Store new artifacts in `results/phase1/continuous_activity_guidance_v1/`. Preserve earlier
negative results and source records. Each completed stage records hashes, denominators,
random seeds, elapsed time, and execution status. An interrupted stage has no admitted result.
Training a conditional model alone is not evidence of guidance improvement.

## Exact statistical objectives

For an observed lipid, let `s` identify the full admitted assay context and let `y` be
the continuous response in the source export. The transformation
`z = (y - mean_fit,s) / std_fit,s` changes numerical scale only. It preserves order,
relative distances within the context, and an exact inverse. It does not recover
absolute photon counts or establish a common scale across the two studies.

The modeled factorization is
`p(G,P | c,s,z) = p_phi(P | c,s,z) p_theta(G | P,s,z)`.
The graph model predicts clean categorical states from corrupted states, time,
the reaction program, the assay context, and the continuous response. Its denoising
objective for one molecule is the sum over the six coordinate families of the
mean negative log probability on variable coordinates in that family. Empty
families contribute zero. This is not a molecule log likelihood and is not a
single mean over all coordinates. The minibatch objective is 0.75 times the
measured-molecule mean plus 0.25 times the unlabeled replay mean. Replay has no
biological condition; it is not assigned zero activity. Label dropout retains
the measured assay context and explicitly marks the response as unobserved.

For fitting designs `i` in a context, the measured program baseline is
`p_hidden(P) = sum_i w_i 1[P_i=P] / sum_i w_i`. Define Gaussian weights
`k_i(z) = w_i exp(-(z-z_i)^2/(2 h^2))`. The continuous program distribution is
`p_phi(P | z) = (sum_i k_i(z) 1[P_i=P] + lambda p_hidden(P)) /
(sum_i k_i(z) + lambda)`.
Weights are normalized to total fitting-design count; they are one for the
present unique-design datasets. Bandwidth and shrinkage are selected only on
inner partitions. The shuffled arm permutes responses within the fitting assay
context and uses the same donor map for its prior and graph model. An unsupported
context, target outside the fitting range, or local kernel effective size below
five produces an explicit abstention. Unsupported observed programs retain their
coverage failure; they are not assigned an artificial probability floor.

The primary graph diagnostic averages each molecule's denoising objective over
the prespecified corruptions at times 0.1, 0.5, and 0.9. Terminal-time prediction
is reported separately. Relative improvement is the difference of mean losses
divided by the comparator's mean loss, rather than a mean of per-molecule ratios.
The earlier capacity-selection objective includes terminal time by design.
Original head groups define the primary descriptive paired bootstrap; the three
training seeds remain separately visible. Reused development partitions and few
independent HeLa heads limit inferential claims.

Generation uses the original strict terminal decoder for every arm. Optional
ring-size and exact-morphology readout policies are disabled. The layout ledger
preserves fitting-derived ring metadata for provenance, but this decoder does
not enforce that metadata. Every attempt records the program-fixed coordinates,
decoded structure or failure, and exact assembly result. Predictor intervals are
inherited from the frozen calibration procedure: they are not newly calibrated
for generated molecules or for the additional immediate-precursor support filter.
