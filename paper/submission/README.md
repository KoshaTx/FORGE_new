# FORGE manuscript

[FORGE.pdf](FORGE.pdf) is the unmodified 39-page manuscript, *FORGE: Reaction-Guided
Generative Design of Ionizable Lipids*. It is marked “Under review as a conference paper
at ICLR 2027”; this repository does not assert acceptance.

SHA-256: `96e1a21b3be3d59431ef1f9c256171a8fcba984305a03999f32f1d35dc66fa35`.

- [Generation quickstart](QUICKSTART.md): installation, model inputs and sampling commands.
- [Data and checkpoints](ARTIFACTS.md): required model bundle and availability.
- [Paper → code and results](EVIDENCE.md): methods, tables, figures and known limitations.
- [Examples](../../examples/README.md): existing molecular drawings and separate smoke outputs.

The paper uses the shared three-family model. Generation with its checkpoint remains untested
until the bundle is supplied; existing smoke tests do not validate the paper's numerical results.

## Manuscript source status

The LaTeX and Overleaf exports in `paper/v1_iclr/` are an earlier related revision, not the
source of this exact PDF. They contain supporting tables and figures but will not rebuild
the submitted manuscript exactly. The corresponding editable source has not been supplied.
The separate 22-family study is described in [STUDIES.md](../../docs/STUDIES.md).
