# Refactor baseline — Phase 0

Recorded 18 August 2026, before any restructuring work. Every later phase reports its deltas
against these numbers. The plan is in `melodic-crafting-chipmunk.md` (Claude plan file); this file
is the measured starting state it is checked against.

## Why this exists

The restructuring's one hard constraint is that **no artifact byte may change**. The 250 pinned
files under `results/` are the paper's evidence chain: they are what lets a number in the manuscript
be traced to the bytes it came from. A refactor that silently alters a `result.json` key order, a
float repr or a CSV column order has broken the paper, not just the code. So the baseline and the
gate come before the first line of restructuring.

## Environment

Until this was done the test suite had **never been run in this repository** — there was no `.venv`
and neither `pytest` nor `rdkit` was importable.

```bash
uv venv && uv pip install -e ".[dev,oracle,torch]"
```

Installed: Python 3.14.5, rdkit, numpy, pandas, scikit-learn, torch 2.13.0, torch-geometric 2.8.0,
xgboost 3.4.1, matplotlib 3.11.1, pytest, ruff, mypy 2.3.1.

Note `torch` is not optional for the test suite despite being an optional extra: 67 test modules
fail at *collection* without it, because `product/phase1_flow.py` defines
`DeterministicSparseFlowBlock` and its siblings inside a `try: import torch` block, so the symbols
simply do not exist when torch is absent.

## Artifact pins — the gate that matters

```
verified   508 pins over 256 distinct files
absent    1030 files not on this machine
foreign     40 container-side paths (skipped)
known        2 accepted pre-existing drift
DRIFT        0
```

`make verify-pins` enforces this. Drift is always fatal; absence is reported but tolerated, because
many pinned inputs legitimately live only on the workstation that produced them. The gate was
negative-tested: appending one byte to `results/m0_03/r0_reconciliation.json` makes it exit non-zero,
and restoring the file makes it green again.

### Accepted pre-existing drift

Two files are recorded in `docs/known_artifact_drift.json` as unrecoverable inherited drift. Both are
unchanged across every commit in this repository, so neither was introduced here:

- `data/splits/phase1/manifest.json` — regenerated upstream after three artifacts pinned it; the
  superseded bytes were never committed.
- `scripts/modal_phase1_product_cuda_preflight.py` — a *code* pin. Two artifacts pin two different
  earlier revisions of the same launcher, so no single state of the file could satisfy both.

An entry is matched on both path and current hash, so if either file changes again the gate turns
fatal. **Do not add entries to make a failing gate pass** — an entry is only correct when the pinned
bytes are genuinely unrecoverable and the artifact that pinned them is frozen.

### One regression found and fixed

The pin verifier's first run caught a break introduced by the immediately preceding commit
(`8d68a24`, the vendoring fix): `make vendor` rewrote `data/vendor/MANIFEST.json`, which
`results/m0_02/result.json` pins. The manifest's *content* was unchanged — 30 assets, zero field
differences — but the bytes moved, because the pinned copy was serialized with a key order the
current writer does not reproduce (`lnpdb_*` before `lantern_*`, which `sort_keys=True` reverses).

Fixed by making vendoring idempotent: an entry is reused verbatim when the asset still hashes to what
the manifest records, and a semantically unchanged manifest is not rewritten at all. Re-running
`make vendor-partial` twice now leaves the file byte-identical. `--refresh-provenance` exists for when
updating the source fields is actually intended.

This is worth remembering as the template for the whole project: the content was fine and the bytes
still moved. Only the hash check caught it.

## Vendored inputs

```
21 of 30 assets present and verified   (9 from COMPOSE git history, 12 from upstream URLs)
 9 absent, all optional                (7 bench .docx, 2 USPTO archives)
```

`make vendor-partial` and `make verify-partial` both pass.

## Tests

First full run in this repository's history. **The suite is not green, and that is the inherited
state — not something the refactor caused.** These are the numbers every later phase is compared to.

```
1,170 passed
  188 failed
   21 error
   68 xfailed      (quarantined by tests/unreproducible_pins.py)
   33 skipped
------
1,480 outcomes across 272 test modules
```

78 distinct modules contain a failure or error. The bulk are artifact-shape tests: they open a
`results/*.csv.gz` and assert row counts, SHA-256 pins and column invariants, and they fail when the
artifact they expect is absent or is a different vintage. Concentrated in
`test_ugi_bounded_hybrid_route_cascade_v1.py`, `test_ugi3_route_registry_pair_*.py` and
`test_single_step_benchmark_manifest.py` — all reading inputs that arrived only in the recent
handoff sync, so several are likely stale expectations rather than real defects.

Two consequences for the restructuring:

1. **Treat every one of these as inherited.** Do not fix a pre-existing failure inside a refactor
   commit; fix it separately, so a genuine regression is never hidden among them. The check that
   matters per phase is "no *new* failures", which means diffing the failing set, not the count.
2. **Investigating the 188 is its own task**, worth doing before Phase 1 migration touches these
   modules — a test that already fails cannot tell you whether your refactor broke something.

Reproduce with:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q --tb=no > baseline.txt 2>&1
```

Note the run takes well over an hour, and `-q` truncates its final summary line; counting the
progress characters (`.` `F` `E` `s` `x`) is the reliable way to get totals.

## Types

```
mypy: 897 errors in 168 files (295 source files checked)
```

Configured **non-blocking** in `pyproject.toml`. This number is a ratchet, not a target to fix in one
pass: annotation *coverage* is already near-total (99.4% of returns, 100% of parameters), so these
errors are about shallow modelling, not missing annotations. `forge.core.*` is held to a strict
override from the start, since it is new code with no legacy to grandfather.

## What the numbers should do

| Phase | verify-pins | tests | mypy |
|---|---|---|---|
| any | 508 verified, 0 drift — **unchanged** | no new failures | not increased |

A dropping `verified` count means an input went missing. Any drift at all means a supposedly-frozen
byte moved. A test flipping from skip to pass is a *finding* worth reporting, not noise — a large
number currently skip on ad-hoc `.exists()` guards rather than registered markers.
