# FORGE paper versions

- `v0/` contains the archived ICLR 2027 LaTeX manuscript and its complete local asset tree.
- `v1/` contains the computational-only Nature Biotechnology manuscript now under development.
- `v1_iclr/` contains the self-contained, anonymous ICLR 2027 manuscript and its Overleaf export;
  see [`v1_iclr/README.md`](v1_iclr/README.md) for build commands and input provenance.
- `forge_paper/` contains repository tooling for verifying and packaging the archived v0 manuscript.

The v0 reproduction contract is `configs/reproduction/iclr2027.json`. The v1 manuscript begins with
`v1/OUTLINE.md`; planned results remain explicit placeholders until their verified artifacts exist.
The separate v1 experiment matrix is `configs/reproduction/natbiotech_v1_experiments.json`; inspect
it with `forge paper experiments`. This command audits experiment readiness and does not claim that
the configured external baselines or ablations have run.
