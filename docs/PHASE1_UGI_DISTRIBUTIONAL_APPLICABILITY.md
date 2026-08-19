# Phase 1 Ugi distributional applicability and conditional uncertainty

Date: 3 August 2026

## Decision

Exact component identity and biological-oracle applicability are independent
properties.

- **Exact-identity provenance** asks whether the same constitutional precursor
  graph or complete measured combination was observed.
- **Distributional applicability** asks whether the complete product and all
  three precursor-role views lie near the labeled AGILE distribution under a
  calibration policy defined without potency values, predictions or errors.

An exact-new component is therefore not automatically out of distribution.
Conversely, an exact-known component combination is not automatically safe for
biological guidance.

## Frozen v3 definition

Version 3 is the current applicability audit. It uses four views:

1. complete Ugi product;
2. amine-derived component;
3. aldehyde-derived component;
4. isocyanide-derived component.

Each view combines a radius-2, 2,048-dimensional count Morgan distance with a
robust descriptor distance. The descriptor vector contains heavy-atom and
carbon counts, heteroatoms, molecular weight, logP, topological polar surface
area, rotatable bonds, rings, formal charge, unsaturation, branching and
ester/amide/ether counts. A candidate receives the worst bin across both
distance types and all four views.

The thresholds model an exact-new component shift. For each held-component
calibration record, the exact component identity is excluded from the
component reference. Products sharing that held component identity are also
excluded from the product reference. Median and 90th-percentile calibration
distances define the interpolative and boundary limits, respectively. Potency
targets, oracle predictions and prediction errors do not set these limits.

The earlier versions are retained:

- v1 showed that ordinary calibration comparisons can collapse component
  distances to zero when the same precursor occurs in another product;
- v2 fixed component identity leakage and replaced binary Morgan fingerprints
  with count fingerprints, but did not align the whole-product reference with
  the same component shift;
- v3 aligns the product and component comparisons. No v1 or v2 artifact was
  rewritten.

## v3 results

Among 3,975 unguided generated products:

| Applicability bin | Count |
|---|---:|
| Interpolative | 165 |
| Boundary | 215 |
| Extrapolative | 3,595 |

Twelve products exactly reproduced measured combinations and were
interpolative. The remaining 3,963 contained at least one exact-new component.
Of those exact-new products, 153 were interpolative, 215 were boundary and
3,595 were extrapolative. This directly demonstrates that exact component
novelty and chemical distribution shift are not synonyms.

The fixed HeLa oracle's outer-test behavior separated strongly by the frozen
bins:

| Bin | n | R2 | Spearman rho | 90% coverage | Absolute coverage gap |
|---|---:|---:|---:|---:|---:|
| Interpolative | 2,732 | 0.463 | 0.653 | 0.832 | 0.068 |
| Boundary | 3,083 | 0.436 | 0.694 | 0.822 | 0.078 |
| Extrapolative | 895 | -0.726 | 0.016 | 0.522 | 0.378 |

These are pooled descriptive results for the already selected oracle. They do
not reopen model selection or authorize guidance.

## Conditional conformal diagnostic

The additive conditional-calibration audit reuses the original three-seed fit
predictions. It does not retrain or reselect the oracle. Within each outer
fold, it:

1. classifies calibration records under the v3 component-shift policy;
2. computes a finite-sample 90% split-conformal absolute-residual radius using
   only interpolative calibration records;
3. applies that radius only to interpolative outer-test records;
4. evaluates the original foldwise R2, Spearman, positive-R2 fraction and
   coverage-gap gates.

At least 20 calibration and 20 test records per fold and at least three
eligible folds per scheme were required.

| Held scheme | Eligible folds | Mean fold R2 | Mean rho | Positive-R2 folds | Mean coverage gap | Gate |
|---|---:|---:|---:|---:|---:|---|
| Head | 4 | 0.344 | 0.635 | 0.75 | 0.059 | Pass |
| Aldehyde | 4 | 0.250 | 0.568 | 0.75 | 0.159 | Fail |
| Isocyanide | 2 | 0.418 | 0.621 | 1.00 | 0.071 | Fail: insufficient folds |
| Head + aldehyde | 0 | -- | -- | -- | -- | Fail: insufficient calibration |
| Head + isocyanide | 0 | -- | -- | -- | -- | Fail: insufficient calibration |
| Aldehyde + isocyanide | 3 | 0.533 | 0.708 | 1.00 | 0.083 | Pass |

The head-only and aldehyde-plus-isocyanide diagnostics pass the numerical
gates. Production guidance nevertheless remains off because this conditional
audit was specified after inspection of the v3 diagnostic and because several
role domains remain unsupported. In particular, the current evidence does not
support unrestricted potency guidance for products containing arbitrary new
components in all three roles.

## Operational consequence

The generator may continue to create exact-new components. Generated products
must carry both fields:

```text
exact_identity_provenance
overall_distribution_bin
```

For now:

- oracle means remain descriptive;
- raw-mean maximization is prohibited;
- extrapolative candidates receive no potency guidance;
- no potency-guided candidate selection is authorized;
- synthesis-route work proceeds independently of this biological gate.

The next decision is a versioned review of whether a small, explicitly scoped
interpolative-domain pilot is scientifically justified. Any such pilot must
declare the supported role patterns, use conditional uncertainty rather than
raw means and retain matched unguided and post-hoc controls.

## Frozen evidence

- v3 config: `configs/bio/phase1_ugi_distributional_applicability_v3.json`
  (`189737881a...d73b`)
- v3 result: `results/phase1/ugi_distributional_applicability_v3/result.json`
  (file `a3e5c4bd44...61f8`; logical `e0173b349d...1da`)
- v3 generated ledger: `generated_applicability.csv.gz`
  (`ed12ac7236...992f`)
- v3 heldout ledger: `heldout_applicability.csv.gz`
  (`8af8624a3c...0da2`)
- conditional-calibration config:
  `configs/bio/phase1_ugi_interpolative_conformal_v1.json`
  (`efe4340b1e...b927`)
- conditional-calibration result:
  `results/phase1/ugi_interpolative_conformal_v1/result.json`
  (file `a7ad0ddb71...e2ef`; logical `4b3abb203d...c0a`)
