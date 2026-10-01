# FORGE

**Reaction-Guided Generative Design of Ionizable Lipids**

[Paper](paper/submission/FORGE.pdf) · [Reviewer guide](paper/submission/README.md) ·
[Paper → code and results](paper/submission/EVIDENCE.md) ·
[Data and checkpoints](paper/submission/ARTIFACTS.md) ·
[Generation quickstart](paper/submission/QUICKSTART.md) · [Examples](examples/README.md)

FORGE (**Flow-matched, Open-ended, Route-resolved Generation and Exploration**) generates
complete molecular graphs conditioned on an assembly reaction program. The program supplies
precursor roles, reaction-core positions, transformation order and coarse molecular architecture;
it does not supply component identifiers, stored precursor graphs or fragment tokens.
Generated molecules are decomposed into implied precursors and checked by exact forward replay.
Bounded upstream route assessment returns a computational dossier or an explicit unresolved outcome.

This branch documents the supplied **39-page manuscript**, pinned in
[`paper/submission/manifest.json`](paper/submission/manifest.json). Its experiments use the
historical **shared three-family model**: Ugi 3-CR, repeated aza-Michael addition and repeated
reductive amination. The separate 22-family study is indexed under [other studies](docs/STUDIES.md).

## Results in the paper

Verified exact-L1 yield (% of **all 3,072 attempts per program and seed**), mean ± sample standard
deviation across three independent training seeds (Table 1):

| Reaction program | FORGE conditioned | Shared null + post-hoc verification | Cyclic program |
|---|---:|---:|---:|
| Ugi 3-CR | 96.4 ± 2.1 | 34.8 ± 7.1 | 93.8 ± 4.2 |
| Repeated aza-Michael | 72.2 ± 1.6 | 19.2 ± 4.1 | 71.5 ± 2.2 |
| Repeated reductive amination | 52.3 ± 3.4 | 37.0 ± 2.2 | 50.7 ± 4.9 |

In the common Ugi benchmark (Table 2), FORGE yielded **963.9 ± 21.4** exact-L1 products and
**862.6 ± 19.2** distinct component-novel products per 1,000 attempts. Component novelty is
relative to the verifier and the method-visible training catalogue; it does not prove strong
product-level catalogue escape. The paper retains negative and reaction-dependent ablation results.

The paper also reports four 4-hour FLuc imaging values for three synthesized FORGE lipid
formulations and MC3. FORGE-3's reported signal is **13.1×** the MC3 value. These are individual
reported values, without replicate-level uncertainty or significance tests; see the
[experimental record and remaining questions](paper/submission/EVIDENCE.md#experimental-record).
Exact assembly replay alone does not demonstrate synthesis success or delivery performance.

## Start here: no GPU or model downloads

From the repository root with Python 3.10+:

```bash
make review-check
```

This offline check verifies the submitted PDF and indexed evidence files, recomputes the
conditioned Table 1 means/SDs from Table 4's seed counts, and lists unavailable historical inputs.
It **does not rerun training or inference**. The JSON report includes expected paths and hashes:

```bash
python3 tools/review_submission.py --json > /tmp/forge-review-report.json
```

**Full numerical reproduction is not available from this checkout alone.** The historical
production checkpoint archives and some evaluation ledgers are absent. The paper also records
missing historical source/environment information. [Artifact access and exact limitations](paper/submission/ARTIFACTS.md)
are separate from the checks that run locally. Small smoke checkpoints elsewhere in the tree are
not the paper's production model.

## Code and development

```bash
uv sync --frozen --extra dev --extra torch
uv run --frozen forge --help
uv run --frozen make test-one TEST=tests/test_review_submission.py
```

Installing dependencies requires network access. Training, sampling and data preparation have
additional hash-pinned inputs; follow the [artifact guide](paper/submission/ARTIFACTS.md) before
running them. `make vendor` includes paths on the original workstation and cannot populate all
inputs from public downloads alone. The [test guide](docs/TESTING.md) explains how to fetch the
already-pinned public LNPDB input and how unavailable historical tests are reported. The current
lockfile is a development environment, not a recovered historical training environment.

The [generation quickstart](paper/submission/QUICKSTART.md) provides a bounded three-family
sampling command and an input-only check:

```sh
uv run --frozen python -m cli.generate --check-inputs
```

It reports missing paper checkpoints/companion records before loading any model. The wrapper is
tested with fixtures and a separate small smoke model; paper-checkpoint generation remains pending.

| Directory | Purpose |
|---|---|
| [`forge/model/`](forge/model) and [`forge/flow/`](forge/flow) | Graph representation, conditioned Transformer, training and discrete-flow sampling |
| [`forge/assembly/`](forge/assembly) | Reaction adapters, decomposition and exact replay |
| [`forge/synthesis/`](forge/synthesis) | Bounded precursor route assessment and evidence |
| [`experiments/`](experiments) and [`configs/`](configs) | Versioned study workflows and contracts |
| [`paper/submission/`](paper/submission) | Submitted PDF, reviewer guide and evidence manifest |
| [`paper/v1_iclr/`](paper/v1_iclr) | Related earlier manuscript sources and supporting tables/figures |
| [`tests/`](tests) | Scientific and software contract checks; some require external artifacts |
| [`docs/`](docs), [`results/`](results), [`provenance/`](provenance) | Research history, retained results and immutable source records |

Contributors: read [AGENTS.md](AGENTS.md), the [architecture](docs/ARCHITECTURE.md) and the
[study map](docs/STUDIES.md). Historical paths remain stable because evidence records refer to
their exact bytes. The legacy `forge paper verify/reproduce/build` commands target the archived
**v0** paper; they do not validate or rebuild the submitted PDF linked above.

## License

Project code is released under the [MIT License](LICENSE), matching the existing package declaration.
Third-party data, weights and publication assets retain their applicable terms; the code license
does not grant additional redistribution rights for those materials.
