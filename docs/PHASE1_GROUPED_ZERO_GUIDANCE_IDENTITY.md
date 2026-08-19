# Phase 1 grouped lambda-zero identity qualification

## Decision

The real selected Ugi generator now passes the frozen 16-program by
4-particle grouped lambda-zero execution gate. This closes the execution-
identity blocker that remained after the nonexecuting grouped-SMC schedule was
frozen. It does **not** authorize nonzero synthesis guidance, route-based
selection, biological guidance, candidate locking or sealed-holdout access.

## Frozen artifacts

- Config:
  `configs/model/phase1_ugi_grouped_zero_guidance_identity_v1.json`
  (`328b78ccf82c88699b0d2f883489ffbd59a4794a013bb04eea687d9d89a28ff3`).
- Result:
  `results/phase1/ugi_grouped_zero_guidance_identity_v1/result.json`
  (file SHA-256
  `9bf9818cbfccf60cf492fb54bdb932eff431df117e9020d68c01548b4b85a07f`;
  logical result SHA-256
  `dbdc45b1b98dcfae9f9e8f18daabe284db4a3ce71a02976b1f071526493f86c4`).
- Qualified runtime: Python 3.14.2, NumPy 2.5.1, RDKit 2026.03.4 and
  PyTorch 2.13.0 from `.venv/bin/python`.

Historical restartable-equivalence v1 is preserved unchanged. The current
selected lane is owned by the versioned v2 receipt, which binds the current
generator implementation and reproduces the old 12-program batch partitions
at batch sizes 1, 4 and 12 with identical terminal hashes.

## Executed contract

The qualification used only the first frozen calibration assignment
(`20260821`):

- 16 distinct canonical morphology programs;
- four independently seeded particles per program;
- 64 exact frozen particle seeds;
- eight flow steps;
- checkpoint completions at steps 2, 4 and 6;
- identity ancestry only;
- two independently initialized but identically seeded selected-model pools;
- paired productive final completion for all 64 particles; and
- an independent direct selected-callback reference for every final particle.

The run performed no route evaluation, no synthesis-value calculation, no
candidate selection, no biological scoring and no holdout access.

## Result

All 192 checkpoint completion pairs were bitwise identical. Identity ancestry
preserved the complete pool digest at every checkpoint. All 64 productive final
pairs were bitwise identical, and all 64 matched the direct one-particle
selected-generator callback exactly. Terminal completion never mutated the
source trajectory state.

Native validity and exact-L1 counts were retained rather than repaired or
filtered:

| Stage | Terminal present | Valid | Exact L1 |
| --- | ---: | ---: | ---: |
| Checkpoint 2 | 64/64 | 64/64 | 64/64 |
| Checkpoint 4 | 64/64 | 64/64 | 64/64 |
| Checkpoint 6 | 64/64 | 62/64 | 62/64 |
| Productive final | 64/64 | 61/64 | 61/64 |

The three invalid final particles are part of the frozen result. They were not
repaired, retried, removed or replaced. The gate establishes orchestration
identity, not perfect terminal validity.

## Remaining gates before any bounded nonzero synthesis-guidance run

No nonzero run is currently authorized. A later owner must create a new,
canonical and independently reviewed execution-review receipt that binds the
current artifacts and explicitly sets `nonzero_guidance_authorized` to true.
The receipt must establish all of the following at the actual execution time:

1. the conservative binary route-completion utility remains frozen;
2. the authenticated cumulative route source and L3 snapshot remain inside
   their declared decision horizon (currently ending 2026-08-09 05:00 UTC);
3. the cumulative-source runtime, terminal route evaluator and isolated
   planner-cache behavior are qualified together for the selected adapter;
4. the grouped particle amendment and this real lambda-zero execution receipt
   are bound as the current production-zero-guidance evidence; and
5. a versioned nonzero execution config/plan preserves the preregistered
   schedule, compute accounting, censoring policy, cache isolation and
   anti-collapse measurements while leaving biology, candidate selection and
   holdout access false.

If the run occurs outside the current L3 decision horizon, the L3 evidence and
execution review must be refreshed first. The proposal-lane v3 benchmark and
prospective candidate selection remain separate later gates; neither was
silently imported into this identity qualification.

## Manuscript use

This receipt supports the technical statement that the selected restartable
implementation preserves the unguided generator exactly under the grouped SMC
schedule. It does not yet support a result claiming that synthesis guidance
improves route closure or discovery efficiency. Such claims require an
authorized nonzero comparison and its preregistered analysis.
