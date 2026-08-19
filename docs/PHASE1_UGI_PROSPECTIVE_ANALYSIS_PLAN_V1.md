# Prospective analysis plan, 30-candidate Ugi panel

Frozen 2026-08-06, before any synthesis, formulation or biological measurement.
Panel: `results/phase1/ugi_prediction_cohort_panel_v3/candidate_manifest.json`.

## Primary

**12 Q-high versus 7 Q-low on continuous biological activity.**

- Inferential unit is the **candidate**, not the well, formulation replicate, plate or animal.
- Readout, dose, time point, background correction, normalization, transformation and
  replicate aggregation are taken from the frozen experimental configuration and are
  not chosen after data are seen.
- Test: exact candidate-level rank or permutation comparison. One-sided
  high-greater-than-low is permitted because it is frozen here; the two-sided
  interval and the raw distributions are reported alongside.
- Report effect size and confidence interval, not only a p-value.

Power is honest: at true rates of 67% versus 10%, one-sided alpha 0.05, a 12-versus-6
comparison has power 0.65. This is informative, not guaranteed.

## Secondary

- Thresholded hit rate using the hit threshold fixed before any outcome is inspected,
  compared by exact Fisher test.
- Spearman association between the frozen oracle score and measured activity.
- Synthesis success, formulation-QC success, end-to-end usable-hit yield.
- Ensemble mean versus LCB90.
- Association of the frozen comparator baselines with activity.

## N cohort, reported separately

Secondary prospective test of analog-supported transfer to exactly one unseen aldehyde.
Report **8 product outcomes across 6 distinct unseen aldehydes**, by aldehyde and by
component context, with similarity to the nearest measured aldehyde, continuous
activity and hit status. Eight products are not eight independent
component-generalization events where aldehydes repeat.

## X cohort, descriptive only

**3 products from 2 distinct unseen aldehydes**: one near-envelope aldehyde in two
component contexts, one remote aldehyde in one. No powered test of general
extrapolation is run or implied.

## Attrition

Report separately at each stage: locked, synthesis attempted, correct product
isolated, formulation QC passed, biological assay completed. Failures stay in the
denominator. **No candidate is replaced once experimental work begins.** The primary
analysis runs among candidates meeting the frozen assay and QC definition; a separate
intention-to-test analysis reports end-to-end success. Failed synthesis is not imputed
as zero activity.

## Sensitivity

- Batch-adjusted analysis using the blocked plate and batch structure in
  `BLINDING_KEY.locked.json`.
- TPSA and other descriptor adjustment as **sensitivity analyses only**.
- No covariate selected after outcomes. Q, N and X are never pooled for the primary claim.

## What the baselines already tell us

Frozen before any outcome, on component-aware 5-fold cross-validation of the measured
1,100:

| model | MAE | Pearson | Spearman |
|---|---:|---:|---:|
| null | 2.823 | -0.235 | -0.217 |
| component-additive ridge | **2.137** | **+0.530** | **+0.469** |
| Morgan ExtraTrees | 2.455 | +0.472 | +0.451 |
| nearest neighbour | 2.510 | +0.413 | +0.378 |

Two consequences must be carried into the write-up.

**The component-additive ridge is the strongest simple baseline**, ahead of the
fingerprint model. Much of the signal in this dataset is factorial component identity.

**All three baselines also separate Q-high from Q-low** on the locked panel
(mean differences +2.58, +3.46, +3.04). So a positive prospective Q-high versus Q-low
result validates the **ranking concept**; it does not by itself establish that the
neural oracle specifically was required. That distinction is stated in the paper
rather than left for a reviewer to find.

The frozen oracle's reported rho of 0.612 came from a different split and is **not**
directly comparable to the 0.469 above. No claim of superiority is made on that basis.

## Claims this plan does not support

- the oracle is state of the art;
- uncertainty is formally calibrated;
- N lies inside a statistically validated applicability domain;
- performance generalizes broadly to unseen building blocks;
- three X products demonstrate general extrapolation.
