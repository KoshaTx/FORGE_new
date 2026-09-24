# COMPOSE-Lipid complete LNPDB corpus v5

This release contains **279,687 unique connected constitutional lipid graphs**
with source, family, precursor, construction, regional, and calculated chemistry
labels. It contains no biological outcome labels and did not use sealed BEAE
results for selection. Read [DATASET_HANDOFF.md](DATASET_HANDOFF.md) before use.

## Fast facts

| Quantity | Count |
|---|---:|
| Unique molecular graphs | 279,687 |
| Reported-source anchors in the molecule table | 11,267 |
| LNPDB anchors in current 96-slot model support | 11,144 |
| Other reported-source anchors | 123 |
| Enumerated virtual products | 268,420 |
| Complete deduplicated LNPDB source ledger | 12,675 |
| LNPDB occurrence rows represented | 19,797 |
| LNPDB identities deferred from current model support | 1,531 |
| Direct exact reaction programs | 275,854 |
| Exact compatible-decomposition replays | 3,833 |
| Compatible decompositions, including alternatives | 4,184 |
| Constitutional precursor identities | 5,997 |
| Molecule fields | 109 |

All 12,675 LNPDB identities have exact construction coverage: 8,842 use an
exact bound program and 3,833 use at least one exact product-derived compatible
decomposition. Every molecule in the corpus has precursor identities. Compatible
decomposition establishes exact graph replay under a family contract; it does not
claim the authors' historical route.

## Files

- `molecules_000` through `molecules_004`: target-ID-sorted molecule tables in
  CSV and lossless JSONL.
- `label_dictionary.csv`: definitions and storage types for all 109 fields.
- `family_summary.csv`: source/virtual, construction, topology, and motif counts.
- `construction_supervision.jsonl.gz`: one construction disposition per molecule.
- `precursors.jsonl.gz`: all precursor identities referenced by direct, selected,
  and alternative exact constructions.
- `lnpdb_source_ledger.jsonl.gz`: all 12,675 deduplicated LNPDB identities.
- `lnpdb_compatible_decompositions.jsonl.gz`: 4,184 exact decompositions.
- `lnpdb_decomposition_dispositions.jsonl.gz`: one row for each of the 3,833
  targets covered by compatible decomposition.
- `source_corrections.jsonl.gz`: preserved source-correction ledger.
- `manifest.json`, `checksums.sha256`, and `verification.json`: immutable release
  and verification records.

## Load and verify

```python
import glob
import json
import pandas as pd

root = "datasets/compose_lipid_lnpdb_complete_2026-09-13_v5"
lipids = pd.concat(
    [pd.read_csv(p, compression="gzip", low_memory=False)
     for p in sorted(glob.glob(f"{root}/molecules_*.csv.gz"))],
    ignore_index=True,
)
assert len(lipids) == lipids.target_id.nunique() == 279_687
assert lipids.constitutional_smiles.nunique() == 279_687
lipids["precursor_ids"] = lipids["precursor_ids"].map(json.loads)
```

```sh
PYTHONPATH=src python3 scripts/verify_partner_lipid_corpus_v5.py \
  datasets/compose_lipid_lnpdb_complete_2026-09-13_v5
```

`reported_source` means the molecular identity was reported. `enumerated_virtual`
means an exact saved program assembled the graph, without claiming experimental
synthesis, yield, availability, or performance. Family counts are a designed
coverage allocation rather than literature prevalence. Descriptor and regional
morphology fields are calculated structural labels, not biological labels.

The chemistry corpus and partner handoff are complete. Generator preparation
still requires a new grouped split from this exact receipt and direct model-path
projection for 3,833 compatible-decomposition rows.
