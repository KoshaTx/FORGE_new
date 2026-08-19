# M0-07 sparse graph runtime gate

## Result

The pure-PyTorch graph implementation passed its train-only CPU gate from clean
commit `232d9fd`. Calibration and test labels were not accessed. The profile
used the 700 training records in fold 0 of the held-aldehyde evaluation.

All admitted structures tensorized without truncation:

| Corpus | Records | Graphs | Atoms | Directed edges | Maximum atoms |
|---|---:|---:|---:|---:|---:|
| Reconciled AGILE product plus A/B/C roles | 1,100 | 4,400 | 105,620 | 205,080 | 60 |
| R0 pretraining | 14,129 | 14,129 | 866,591 | 1,726,776 | 282 |
| Virtual applicability library | 12,276 | 12,276 | 561,838 | 1,114,748 | 62 |

The feature support is 23 atom features and 6 bond features, derived from the
frozen graph-corpus audit. Unknown categories fail closed. Padding, generic
unknown buckets, and graph truncation are not used.

## CPU profile

Each architecture used three deterministic repeats, one warm-up epoch, and five
timed epochs:

| Architecture | Parameters | Median epoch | Median records per second |
|---|---:|---:|---:|
| Whole-graph D-MPNN | 72,961 | 0.209 s | 3,343 |
| Whole-graph edge-aware GIN | 237,572 | 0.230 s | 3,039 |
| Ugi component-role-aware D-MPNN | 204,929 | 0.556 s | 1,259 |

The maximum atom and edge permutation deviation across models and repeats was
`7.45e-8`, below the frozen `1e-6` tolerance. All losses and gradients were
finite. Peak process RSS was below 624 MB.

Runtime does not select a scientific model. It establishes that all three
supervised graph lanes can run across the frozen partitions and seeds on CPU.

## Next steps

1. Cache the graph tensors once with input and feature-vocabulary hashes.
2. Schedule each architecture, partition, endpoint, and seed as a resumable fit.
3. Run label-free R0 pretraining independently from supervised random
   initialization.
4. Preserve training-only target scaling and training-only epoch selection.
5. Use calibration rows only for the final ensemble conformal intervals.
6. Select one architecture across both endpoints by the frozen equal-scheme
   rule, or freeze an applicability-based abstention policy if component
   transfer remains weak.
