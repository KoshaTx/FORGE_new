# M0-07 Classical AGILE Oracle Matrix

Status: **classical representation lane complete; oracle not frozen**

## Question

Can structure-computable classical models predict the reconciled AGILE HeLa
and RAW 264.7 transfection measurements when complete molecular components,
component pairs, or scaffolds are withheld?

The random split is retained only as a reproduction diagnostic. It cannot
select the oracle.

## Evidence ledger

| Claim | Evidence class | Source |
|---|---|---|
| The matrix contains 768 fits and 163,680 held-out predictions | Computed | `results/m0_07/oracle_classical_result.json` |
| Every fit uses 1,100 reconciled single-structure records | Computed | Curated input hash recorded in the result |
| The matrix evaluates Morgan counts, 217 RDKit descriptors, and their concatenation with ridge, RF, XGBoost, and MLP regressors | Computed | Frozen config and feature manifest |
| Random-split performance exceeds held-component performance | Computed | Frozen per-fit metric ledger |
| LANTERN's published source pipeline fits a scaler before applying its persisted split | Reported from inspected primary source code | LANTERN commit `11240f29ef92323649ae60d177b21df77e2d428b`, `pipeline/preprocess.py` |
| A low-distance virtual candidate is biologically reliable | Not established | No virtual-candidate endpoint labels exist |

## Frozen evaluation

All imputers, feature scalers, target scalers, and estimators are fitted on
training rows only. Calibration rows are first used after model fitting to
construct finite-sample split-conformal intervals. Seed 1729 and 768
context-specific fit seeds are recorded. Parallel tree models use 12-digit
numeric serialization, which is stable under measured parallel aggregation
variation of at most 3.6 × 10^-15.

The three representations are:

1. 1,024-dimensional radius-2 Morgan count fingerprints without chirality;
2. 217 structure-computable RDKit descriptors;
3. their concatenation.

No apparent pKa, particle size, polydispersity, encapsulation efficiency, or
formulation robustness value is used or imputed.

## Main result

The random diagnostic materially overstates performance under component
shift. On LANTERN's exact random assignments, the best tested classical model
was XGBoost with Morgan plus RDKit features:

| Endpoint | Random-split R² | RMSE | Pearson r | 90% coverage |
|---|---:|---:|---:|---:|
| HeLa | 0.635 | 2.015 | 0.800 | 0.918 |
| RAW 264.7 | 0.461 | 1.349 | 0.685 | 0.909 |

Equal weighting of the seven selection-eligible schemes instead identified
random forest with Morgan plus RDKit features as the classical-lane leader:

| Endpoint | Mean R² | Mean RMSE | Mean Pearson r | Mean absolute 90% coverage gap | Mean 90% interval width |
|---|---:|---:|---:|---:|---:|
| HeLa | 0.215 | 2.723 | 0.572 | 0.066 | 8.648 |
| RAW 264.7 | 0.087 | 1.719 | 0.367 | 0.045 | 5.319 |

The endpoint standard deviations in the 1,100-record corpus are 3.286 mTP
units for HeLa and 1.897 mTP units for RAW 264.7. The conformal intervals are
therefore broad, even when empirical coverage is near its nominal level.

## Component-shift result

The best model within each single-component holdout produced:

| Holdout | HeLa best mean R² | RAW best mean R² |
|---|---:|---:|
| Entire amine heads | -0.035 | 0.114 |
| Entire aldehydes | -0.009 | -0.009 |
| Entire isocyanides | 0.299 | 0.018 |

Pair holdouts were easier because the individual components remained present
with other partners. Best mean R² values ranged from 0.353 to 0.493 for HeLa
and from 0.150 to 0.183 for RAW 264.7. These results support two separate
statements:

1. the classical models learn useful within-component and recombination
   signal;
2. they do not yet justify strong potency guidance for entirely new component
   families, especially transferred aldehyde tails.

Some scaled linear and MLP descriptor fits extrapolated catastrophically on
specific held-aldehyde and scaffold partitions. Those negative results remain
in the ledger. They are not used to change the frozen hyperparameters after
test inspection. Tree ensembles were substantially more stable.

## Unlabeled 12,276-candidate set

Maximum binary Morgan Tanimoto similarity to the 1,100 labeled structures had
a median of 0.939, a mean of 0.906, and a minimum of 0.508. Of 12,276
candidates, 7,194 had similarity of at least 0.9 and 108 were below 0.6.
All 12,276 remain unique after conversion to the stereochemistry-free model
representation.

This is only a structural-proximity audit. It is not biological calibration
and does not supply potency labels.

## Decision

Do not freeze an oracle from this lane. Continue the predeclared molecular
graph, region-aware graph, and lipid-pretrained representation lanes.
Separately reproduce LANTERN's source-faithful random diagnostic, with its
full-data scaling behavior labeled as an audit and excluded from model
selection.

Until a later representation earns stronger component-shift performance,
prospective oracle tilting must:

- use calibrated uncertainty and an explicit applicability domain;
- reduce or abstain from guidance for unseen aldehyde and head families;
- retain synthesis-grounded tail transfer as exploration, not as evidence that
  potency is known;
- preserve candidates across declared applicability bins for prospective
  calibration.

AGILE remains a predictive general-transfection oracle. It is not an in-vivo
endpoint oracle.

## Reproduction

```bash
make verify
make m0-07-oracle-classical
```

The command verifies every input SHA256, rebuilds all 768 fits, emits
deterministic gzip ledgers, and records the complete software and feature
manifest.
