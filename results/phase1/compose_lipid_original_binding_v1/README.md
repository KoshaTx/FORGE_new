# Original recipe binding — 2026-09-20

This audit binds the current eligible preparation population to the restored original task,
context and regional-ledger records. It adds recipe provenance, not a new reaction transform
or training admission. Every input and the implementation are SHA-256 pinned in `result.json`.

| Family | Eligible recipes checked | Verified recipes | Evidence limit |
| --- | ---: | ---: | --- |
| Aldehyde Ugi-3 | 955 | 955 | Exact source context and product identity; source maps are not imported |
| Aldehyde Ugi-4 | 3,222 | 3,222 | Regional tuple/product IDs and precursor catalogue; original product-program payload unavailable |
| STAAR | 5,394 | 5,394 | Exact original task inputs and declared stage order; task contains no product graph |
| Acid/epoxide | 5,279 | 5,279 | Exact original task inputs and declared stage order; reaction executor remains unqualified |
| Total | 14,850 | 14,850 | Zero training admissions |

The audit independently authenticates the current full protection ledger. It selects eligible
construction line numbers before decoding construction payloads, then selects exact original
source lines. It checks file and payload hashes, task/tuple IDs, source precursor IDs, roles,
quantities and canonical component structures. A product cannot choose or repair a recipe.
Global component IDs, source roles, quantities, old/corrected TRAIN requirements and prior
holdouts remain unchanged. Ugi-3 role translation consumes the separately qualified, byte-bound
v2 role witnesses without replaying an older population. Only selected context product graphs
are canonicalized; protected or unassigned target graphs never enter this audit.

Reproduce into a fresh directory:

```bash
.venv/bin/python -m experiments.phase1.multireaction.compose_lipid_original_binding \
  --config configs/multireaction/compose_lipid_original_binding_v1.json \
  --output results/phase1/compose_lipid_original_binding_recheck
.venv/bin/python -m pytest -q tests/test_compose_lipid_original_binding.py
```

The complete per-record ledger is local and hash-bound. For reproducibility compare the
`summary` and `artifacts["bindings.jsonl.gz"]["sha256"]` fields of the two receipts; output
paths necessarily differ. Validation and the independent replay comparison are retained in
`../compose_lipid_original_binding_validation_v2/`. The interrupted v1 validation is a development
record, not a completed validation. The chemistry acquisition abstention is retained in
`../compose_lipid_acid_epoxide_source_v1/`.

The existing exact supplied reconstruction count remains 26,619 across nine families. The
other 50,181 eligible records still need chemistry qualification. Full-universe partition,
representation, constitutional deduplication, balanced weights and the full test gate remain
open. No training or upstream COMPOSE modification was performed.

Completed validation: 511 focused tests and all 30 vendor assets pass. Full suite: 3,121 passed,
131 failed, 17 setup errors, 92 skipped/xfail; failure/error identities are unchanged from the
preceding intake baseline. The production source snapshot stayed unchanged. The repeated audit
reproduces the ledger byte-for-byte. The repository-wide definition of done is not met.
