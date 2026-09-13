# Current-sampler guidance preflight — 2026-09-11

This bounded local check starts the synthesis-guidance restart. It tests whether an identity
controller can observe the current sampler without changing its output, and whether the frozen
strict synthesis-evidence source can authenticate. It does not qualify nonzero guidance or claim
that the realism problem has been solved.

## Frozen comparison

Use the baseline checkpoint, program draw, decoder and random streams pinned by
`configs/multireaction/ugi_sampling_trace_v1.json`. Retain the original full batch of 128 programs,
32 flow steps, two CPU threads and float32 precision. Run this same batch twice: control and
zero-strength observation. Do not fit, retry failed molecules, shrink support or select products.

The observer calls the existing identity ancestry controller at steps 8, 16, 24, 28 and 31, using
explicitly synthetic unequal values. At strength zero it must return identity without using a
resampling seed. It may record model inputs but cannot modify sampler states. The control and
observed run must have identical complete output rows, sampling summaries, terminal states and
sampler random states, with 34 model forwards in each. Both terminal captures must cover all 128
attempts. Compare the baseline with the historical first batch using the trace runner's declared
saved fields. Reassess structural validity, exact Ugi reconstruction and novelty against TRAIN
identities only. Exact row equality preserves all row-derived diversity and novelty metrics for
this batch; it does not establish preservation under future nonzero guidance.

## Evidence and stopping rule

Authenticate the source-qualified cumulative source at the historical assessment time from
`configs/model/phase1_ugi_production_synthesis_guidance_seam_v4.json`. Inventory every pinned
source-builder input, allowing only existing exact historical aliases or archived bytes. Attempt
the real source loader and retain its complete exception chain if it fails. Record missing loader
paths separately from input pins whose exact expected hash is known. File existence alone does
not authenticate evidence. No contemporary stock claim or procurement action is involved.

Missing source evidence blocks the passive route-contrast experiment. It must not be assigned a
zero candidate utility, treated as chemical impossibility or replaced with the broader component
dossier-resolution statistic. A returned loader still requires the separate source, value and
planner qualifications before candidate assessment. This command performs zero candidate route
assessments and cannot enable nonzero guidance. The independent zero-observation check may finish
even when route evidence is unavailable. Failed identity checks terminate the run with a failure
record. Preserve original files and all failed attempts.

## Reproduction

Run with a fresh output directory:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  -m experiments.phase1.synthesis_guidance.current_sampler_preflight \
  --repo-root . \
  --config configs/model/phase1_ugi_current_sampler_guidance_preflight_v1.json \
  --output-dir results/phase1/ugi_current_sampler_guidance_preflight_v1
```

The result records input hashes, the implementation snapshot, raw outputs, endpoint states,
identity decisions, train-only assessments and the route-source failure manifest. This is a
qualification of passive zero-strength observation only. Resumable trajectories, nonzero
ancestry changes, useful synthesis-value contrast and matched guidance benefit remain untested.
