# M0-07 LANTERN Released-Checkpoint Reproduction

Status: **reproduced as an audit-only diagnostic; prohibited from model selection**

## Question

What performance does LANTERN's released AGILE multilayer perceptron achieve
when evaluated with its committed features, random split, checkpoint, and
preprocessing implementation?

## Exact source contract

The reproduction is pinned to LANTERN commit
`11240f29ef92323649ae60d177b21df77e2d428b`. It uses:

- 1,100 curated HeLa structure-label records;
- 2,048 circular count features with chirality;
- 210 RDKit expert descriptors;
- the exact 880/110/110 random train, validation, and test assignment;
- the released seven-layer feedforward checkpoint;
- the source pipeline's MinMax scaling to `[-1, 1]`.

All source artifacts are SHA256 verified. The external feature dictionaries
are loaded through an allowlisted NumPy-only unpickler. The PyTorch checkpoint
is loaded with `weights_only=True`.

## Reproduced result

| Split | Rows | R² | RMSE | MAE | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| Train | 880 | 0.963 | 0.629 | 0.480 | 0.982 | 0.975 |
| Validation | 110 | 0.765 | 1.695 | 1.162 | 0.875 | 0.844 |
| Test | 110 | 0.820 | 1.416 | 1.109 | 0.906 | 0.900 |

The released checkpoint therefore reproduces a strong random-split HeLa
diagnostic.

## Why the result cannot select the FORGE oracle

LANTERN's committed `pipeline/preprocess.py` concatenates all molecular
features and all HeLa labels, fits `MinMaxScaler` on the complete 1,100-row
matrix, and only then applies the train, validation, and test split. The test
feature ranges and test label range therefore affect the model inputs and the
inverse transformation used for predictions.

This is test-information leakage. The reproduction preserves it because the
purpose is source fidelity, but the result is not eligible for model
selection, calibration, or oracle freezing.

The gap between this R² of 0.820 and the best leakage-free FORGE random
diagnostic R² of 0.635 cannot be attributed entirely to leakage. The released
model also uses different source features, a deeper network, and a selected
checkpoint. A causal leakage estimate would require retraining the identical
architecture and features with only the scaling scope changed.

More importantly, both values are random-split results. Neither establishes
generalization to an entirely unseen amine head, aldehyde, isocyanide, or
scaffold. The frozen component holdouts remain the selection-relevant tests.

## Decision

Retain the released-checkpoint score as a faithful literature diagnostic.
Never use it to select or calibrate the FORGE oracle. Continue the molecular
graph, region-aware, lipid-pretrained, and auxiliary native-Ugi supervision
lanes under the frozen leakage-free evaluation contract.

## Reproduction

```bash
make verify
make m0-07-lantern-reproduction
```

The command verifies all source hashes and emits deterministic JSON and gzip
prediction artifacts.
