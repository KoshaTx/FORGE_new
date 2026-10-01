# Data and checkpoints

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

These archives are **not bundled in Git**. A verified download link will be added when the
bundle is available. Each archive needs its companion `training_result.json` and
`study_design.json` in the same directory. [checkpoints.json](checkpoints.json) records the
expected hashes for these nine files; the [quickstart](QUICKSTART.md) explains placement and
input verification. The production cache, registries and other direct generation inputs
are already committed and checked against the sampling configuration.

The earlier Ugi-only application model and the small smoke weights are different models.
Generation with the paper checkpoint and full numerical reproduction remain unverified.
Appendix A.8 describes limitations in the available historical source/environment records.

## Data and table regeneration

The versioned input contracts are linked in [EVIDENCE.md](EVIDENCE.md). Many full evaluations live
under ignored `runs/` paths; a committed contract identifying them is not the same as distributing
their contents. Full `make vendor` requires hash-pinned assets from the originating workstation,
in addition to available public inputs. The selected public LNPDB test input can be fetched with
`PYTHONPATH=.:tools python -m forge_data.fetch lnpdb_fc7c389.csv`; that dataset alone does not supply
the paper's training cache or checkpoints. See [data provenance](../../docs/DATA_PROVENANCE.md)
for origins and boundaries.

After the maintainers provide the exact required inputs, install the locked development environment:

```bash
uv sync --frozen --extra dev --extra torch
```

Then render into a **new scratch directory**, leaving frozen manuscript tables intact:

```bash
uv run --frozen forge paper render-gem-table1 \
  --config configs/reproduction/gem_table1_core_saturation_complete_v1.json \
  --output /tmp/forge-table1-review/generated \
  --result /tmp/forge-table1-review/result.json

uv run --frozen forge paper render-completed-evidence-v1 \
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

The legacy `forge paper verify/reproduce/build/doctor` commands target the archived **v0**
manuscript, not this submission. Building the `paper/v1_iclr/` LaTeX produces an earlier related
revision. Neither operation establishes reproduction of the submitted PDF.
