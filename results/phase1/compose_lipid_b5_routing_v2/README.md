# Exact source-pair routing for B5 I9

The original `one_tail_acid_knob` generation lane includes recipes with two complete source
compound 11 tails. The previous lane-only profile selection assigned these to the I9 hetero
program, whose source control uses compounds 18 and 11. The source SI instead explicitly
documents the 11/11 pair in the I9 homo branch: Figure S3 (PDF page 25), compounds 26/27
(pages 32-33), and I95/I97 (pages 34-35).

The v2 overlay selects the existing homo program only when the original task declares both tails
as reported and their complete input structures exactly match the source 11/11 pair. Original
tasks and metadata are unchanged. Both full component identities, unit quantities, source core
stereochemistry, head scope, site order, net stages and complete forward/inverse checks remain
mandatory. No target graph is used to select a profile. Other tail disagreements remain pending.

Reproduction:

```sh
.venv/bin/python results/phase1/compose_lipid_b5_routing_v2/build_registry.py
.venv/bin/pytest -q tests/test_compose_lipid_b5_routing.py
.venv/bin/python -m experiments.phase1.multireaction.compose_lipid_b5_replay_v2 \
  --config configs/multireaction/compose_lipid_b5_replay_v2.json \
  --output results/phase1/compose_lipid_supplied_b5_v2_reproduction
.venv/bin/python results/phase1/compose_lipid_b5_routing_v2/compare_replays.py
```

The pinned v2 replay and strict merge add eight exact computed reconstructions. The other 4,069
B5 replay rows are byte-equivalent after JSON decoding, including all previous positive and
negative evidence. B5 now has 3,036 exact and 1,041 pending eligible records. The all-family merge
has 54,789 exact computed reconstructions; no training or experimental-execution claim is made.

The source supplement, independent transcriptions, original registry, v1 code and v1 results are
preserved. `replay-comparison.json` pins both replay receipts and identifies every changed row.
