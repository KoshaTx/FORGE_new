# Existing FORGE examples

These links expose existing artifacts; they are **not outputs of the new generation wrapper**.
Frozen files remain in their original locations so their paper/provenance references stay valid.

## Paper illustrations and results

- [Ugi product and recovered precursors](../paper/v1_iclr/figures/chemical_drawings/atlas_01_2d.svg)
  and [precursor drawing](../paper/v1_iclr/figures/chemical_drawings/atlas_01_precursors.svg).
- [Generated sample atlas](../paper/v1_iclr/figures/forge_generated_sample_atlas_core_saturation_v1/atlas_rows.tex):
  retained structures and precursor strings used by the related manuscript source. See
  [EVIDENCE.md](../paper/submission/EVIDENCE.md) for correspondence to the supplied PDF.
- [Per-seed exact-L1 counts](../paper/v1_iclr/generated/production_seed_exact_counts_rows.tex)
  and [conditioned/null/cyclic comparison](../paper/v1_iclr/generated/shared_program_figure_rows.tex).
  These are retained results, not freshly recomputed model measurements.

## Engineering smoke example

[`shared_synthesis_program_production_smoke_v1/`](../results/phase1/shared_synthesis_program_production_smoke_v1/)
contains a tiny model archive, training/evaluation records, `samples.jsonl.gz` and a
[molecule report](../results/phase1/shared_synthesis_program_production_smoke_v1/molecule_report.html).
This is a two-step training smoke run, **not the paper's step-9,143 model**. It tests software
interfaces and carries no claim about useful molecule quality or paper performance. The wrapper's
runtime tests use this fixture explicitly; the public generation command rejects it as a paper bundle.

For newly generated outputs after the paper bundle is available, follow the
[generation quickstart](../paper/submission/QUICKSTART.md).
