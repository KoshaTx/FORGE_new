# Phase 1 Ugi selective-risk and proposal diagnosis

## Decision

The low active-eligibility rate is not one failure mode. It combines:

1. terminal validity;
2. multiview chemical support;
3. exact-role evidence restrictions; and
4. exact measured products that are intentionally neutral.

The current evidence supports a continuous applicability-policy confirmation. It
does **not** establish that morphology sampling failed, replace the frozen
version-3 policy, authorize potency guidance or justify SMC.

## Frozen terminal accounting

| Gate | Broad frozen census | One lambda-zero seed |
|---|---:|---:|
| Scheduled attempts | not available; census is postvalid | 256 |
| Valid exact-L1 | 3,975 prefiltered | 241 (94.1%) |
| Exact measured, neutral | 12 | 11 |
| Chemically supported novel | 153 | 23 |
| Active under the role policy | 47 | 8 |
| Chemically supported but role-policy abstained | 106/153 (69.3%) | 15/23 (65.2%) |

Exact component novelty is therefore not equivalent to chemical extrapolation.
Seven broad-census products and one pilot product generate all three precursor
roles de novo while remaining interpolative in all four frozen structural views.
They still abstain because there is no all-three-new biological calibration.

The broad census cannot estimate generator invalidity because it contains only
valid exact-L1 terminals. The lambda-zero seed can: 241 of 256 attempts were
valid exact-L1 products.

## The tight chemical gate is the whole-product view

Among the 230 novel valid pilot products:

- 23 pass the whole-product fingerprint radius;
- 50 pass the whole-product descriptor radius;
- the same 23 pass both;
- all 230 pass the isocyanide view;
- 157 pass the amine view; and
- 60 pass the aldehyde view.

Omitting the product view would increase chemically supported novel pilot
products from 23 to 36 and active products from 8 to 17. This localizes the
bottleneck, but does not justify dropping the product view: it is the only view
that tests the joint product and component interaction context.

## Fixed continuous radius has held-error signal

The primary diagnostic is target-free:

```text
R(x) = maximum over product, amine, aldehyde and isocyanide
       and fingerprint and descriptor views of
       distance / frozen interpolative radius
```

`R <= 1` is exactly the continuous form of passing all eight structural
distances. Across the six primary held-component and held-pair schemes, it
retains 2,732 of 6,600 OOF rows (41.4%). Equal weighting across the six schemes
gives:

| Metric | All OOF rows | `R <= 1` | Relative reduction |
|---|---:|---:|---:|
| MAE | 1.944 | 1.710 | 12.0% |
| RMSE | 2.647 | 2.358 | 10.9% |

A 2,000-replicate bootstrap resampling exact product labels gave:

- MAE reduction: 8.3–15.9% (95% interval);
- RMSE reduction: 7.5–14.9%;
- retained coverage: 39.3–43.5%; and
- radius–absolute-error Spearman correlation: 0.121–0.187.

Every retained scheme has positive R² above 0.25 and Spearman correlation above
0.49. At least 75% of eligible folds per scheme have positive R². These are
exploratory diagnostics, not independent confirmation, because the policy was
developed after earlier inspection of the same OOF evidence.

The improvement is not uniform. MAE falls 31.4% for held aldehydes and 31.2%
for held heads, is essentially unchanged for held isocyanides, and worsens by
4.3–7.0% in the three held-pair schemes. This is why the radius is useful as a
transparent support coordinate but is not yet a calibrated universal risk
function.

The separate LANTERN scaffold stress set has zero records inside `R <= 1`. It is
therefore reported separately and is not pooled into the six-scheme analysis.

## The learned challenger is not ready

A nested, exact-label-grouped monotone quantile model improves aggregate error
ranking over categorical bins. It nevertheless fails a frozen support-balance
safeguard. Among its globally lowest-risk 25%:

- 7.9% of held-aldehyde rows are retained;
- 7.6% of held-head rows are retained;
- 40.0% of aldehyde–isocyanide rows are retained; and
- 40.0% of head–isocyanide rows are retained.

The model is partly learning which holdout regimes are easy. Aggregate gains do
not justify allowing it to avoid difficult role shifts. It remains a diagnostic
challenger and cannot replace the transparent fixed radius.

## Consequence for the controller

Do not complete the remaining old pilot seeds or implement hierarchical SMC yet.
One observed seed contains only 16 morphology programs, three with an active
terminal. That is enough to show sparse signal, but not enough to attribute the
problem to morphology rather than terminal chemistry.

The next gate is:

1. specify a role-scheme-aware continuous selective-risk policy with per-scheme
   coverage floors and terminal abstention;
2. confirm it independently rather than reusing this exploratory screen as a
   final authorization;
3. run a dynamic frozen-prior terminal census with saved partial states and
   multiple frozen continuations; and
4. compare supported diverse terminals per generator and oracle call for plain
   rejection, dynamic morphology proposals and delayed rollout guidance.

Only if morphology or a partial-state value predicts terminal support should the
corresponding proposal controller be built. Potency remains a separate utility
that may act only after applicability is established.

## Frozen artifacts

- Config:
  `configs/bio/phase1_ugi_selective_risk_proposal_diagnosis_v1.json`
  (`96c3750851c4b96fbecd5015702bcf3784c2706b8a6f6fc74a544f719de8c8b7`)
- Result:
  `results/phase1/ugi_selective_risk_proposal_diagnosis_v1/result.json`
  (`4cd54dfc1a5245996a149756162f0d4ad9eb41cb2ee9ab9b28c65464fdd5cedd`)
- Grouped OOF risk ledger:
  `results/phase1/ugi_selective_risk_proposal_diagnosis_v1/selective_risk_oof.csv.gz`
  (`b0a6f5c2f424c0c54d383016c6999b4c44f8c7a6edbae04269e382fe33743c72`)
