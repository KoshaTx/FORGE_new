# FORGE paper versions

**Start with [`submission/`](submission/README.md) and its exact supplied PDF.**

| Directory | Role |
|---|---|
| [`submission/`](submission/README.md) | Author-supplied 39-page PDF, paper-to-code map, checkpoint access and generation instructions |
| [`v1_iclr/`](v1_iclr/README.md) | Earlier related three-family manuscript sources, generated tables, figures and provenance |
| `v1_iclr22/` | Separate 22-family study; not the submission's three-family model |
| `v1/` | Historical Nature Biotechnology working draft and shared result renderings |
| `v1_neurips/`, `v1_neuripsgem/` | Earlier venue-specific manuscript revisions |
| `v0/` | Archived manuscript with a separate frozen reproduction contract |
| `forge_paper/` | Paper verification, result rendering and packaging tools; commands have distinct contracts |

The submitted PDF is newer than the checked-in `v1_iclr` LaTeX/Overleaf exports. See
[`submission/README.md`](submission/README.md#manuscript-source-status) before attempting to build it.

`forge paper verify/reproduce/build/doctor` and `configs/reproduction/iclr2027.json` target **v0**.
`forge paper experiments` audits a historical v1 experiment matrix; it does not claim that every
configured experiment has run. Neither command family establishes reproduction of the
submitted paper.

Historical source, table and figure paths remain stable because provenance records pin their bytes.
