# Data, checkpoints and reproduction

## What runs from a fresh checkout

```bash
make review-check
python3 tools/review_submission.py --json > /tmp/forge-review-report.json
```

Python 3.10+ is sufficient; no installation, network, RDKit, PyTorch, credentials or GPU is needed.
The checker hashes the 121 indexed review assets and recomputes the conditioned Table 1 means and
sample SDs from the nine Table 4 rows. It also inspects 72 explicitly indexed historical input pins.
At the cleanup baseline, **6 are available with matching hashes and 66 are missing**. Its JSON
output lists every path, expected hash and current status. The inventory covers direct references
from seven table contracts plus the production training/design/checkpoint records; it is not a
recursive inventory of every corpus, ledger, engine or environment needed for training.

The default command succeeds when the review assets and arithmetic are intact, even if disclosed
historical inputs are missing. A changed historical file is always an error. To require all indexed
historical files to be present as well:

```bash
python3 tools/review_submission.py --require-run-artifacts
```

This stricter check is expected to fail on a fresh checkout. Even passing it would establish only
availability of the indexed bytes, not a complete numerical reproduction or recovered environment.

## The paper's model

The paper uses the historical **shared three-family conditional model**, final step **9,143**.
The source of the following identities is
[`mathematical_review.json`](../v1_iclr/mathematical_review.json), `runs[*].pins.checkpoint_archive`.
The old `matches: true` fields describe that historical audit; they do not mean the files exist in
this checkout today.

| Seed label / RNG seed | Expected archive, relative to repo root | Expected SHA-256 |
|---|---|---|
| 0 / 20260825 | `results/phase1/shared_bias_parallel_program_role_seed0_v2/checkpoints.tar` | `10d77b9d8e947d40fe2aaa91ecd68448ff068e08339cecd4c47d862e96e39142` |
| 1 / 20260826 | `results/phase1/shared_bias_parallel_program_role_seed1_v2/checkpoints.tar` | `a3e1e15cb6d55e6a0bc8d161a0061762ef6190c22c957fd957493bbe3860d55d` |
| 2 / 20260827 | `results/phase1/shared_bias_parallel_program_role_seed2_v2/checkpoints.tar` | `f500418e7cdfe6aa59b5d1261cb68c7ff715fa1a0e06498334eb727185f74506` |

These archives are **not bundled in Git**, and this branch does not invent a public download URL
or assume a Modal volume is still accessible. Request the exact archives and companion
`training_result.json` / `study_design.json` records from the maintainers. The manifest records their
expected hashes. Do not substitute the Ugi-only 1,000-step application checkpoint, small smoke
archives, oracle checkpoints or newer 22-family weights for the paper's model.

For historical correspondence, the seed-0 evaluation snapshot is commit
`ac9ef87be4c8e5c14c477bc194a333344e621d3e` with source fingerprint
`3e364c01e8a8f792c8fdfef58524ca714562c9cbed4aac225d39d41a8ebad052`.
No exact match was recovered for the seed-1/seed-2 evaluation fingerprint
`8050c5ac500fd64bf6e3252cc537d3fb16f9d6ef8d06044fdb272618d4591e03`.
A complete historical training-source and environment/compute manifest is also unavailable.
These limits are disclosed in Appendix A.8; current code is not a replacement historical receipt.

## Data and table regeneration

The versioned input contracts are linked in [EVIDENCE.md](EVIDENCE.md). Many full evaluations live
under ignored `runs/` paths; a committed contract identifying them is not the same as distributing
their contents. `make vendor` copies hash-pinned assets from the originating workstation, not from
a public data service. See [data provenance](../../docs/DATA_PROVENANCE.md) for origins and boundaries.

After the maintainers provide the exact required inputs, install the locked development environment:

```bash
uv sync --frozen --extra dev --extra torch
```

Then render into a **new scratch directory**, leaving frozen manuscript tables intact:

```bash
uv run forge paper render-gem-table1 \
  --config configs/reproduction/gem_table1_core_saturation_complete_v1.json \
  --output /tmp/forge-table1-review/generated \
  --result /tmp/forge-table1-review/result.json

uv run forge paper render-completed-evidence-v1 \
  --config configs/reproduction/natbiotech_v1_completed_evidence_v1.json \
  --output /tmp/forge-completed-review/generated \
  --result /tmp/forge-completed-review/result.json
```

These are **artifact replay**, not fresh model evaluation. They fail when required pinned inputs
are missing or changed. Historical renderer labels and styling can differ from the edited manuscript;
`paper/v1_iclr/iclr_provenance.json` records local editorial changes. Compare numeric content as well
as output provenance, rather than overwriting the manuscript with newly generated files.

Training and sampling infrastructure is documented in [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md)
and [STUDIES.md](../../docs/STUDIES.md). A full rerun additionally needs original corpora, split/cache
identities, arm-specific checkpoints, evaluator inputs and a recovered runtime. This cleanup does
not download models, run training, spend cloud compute or claim a completed end-to-end rerun.

## Different commands, different contracts

| Command / path | Scope |
|---|---|
| `make review-check` | This supplied PDF, indexed assets and stated arithmetic check |
| `forge paper render-gem-table1` | Historical three-family Table 1 artifact replay, with external inputs |
| `forge paper render-completed-evidence-v1` | Completed-evidence renderer; original historical names retained |
| `forge paper verify`, `reproduce`, `build`, `doctor` | **Archived v0**, via `configs/reproduction/iclr2027.json` |
| `make test-study-compatibility` | Present-day three-family/22-family software interfaces; not paper replication |
| Building `paper/v1_iclr/FORGE_ICLR2027_paper.tex` | Earlier related manuscript, not the exact submitted PDF |
