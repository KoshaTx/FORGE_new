# Restructure plan

## What is actually wrong

Moving files between folders would not fix this codebase. The problems are internal.

**There is no API — there is reach-in.** 343 scripts import **610 distinct symbols** from the
library across 999 import sites. All eight `__init__.py` files carry the same stub docstring and
export nothing, so every consumer reaches directly at a module, and 21 modules reach at another
module's *private* underscore helpers. A package that exports nothing is a folder.

**The most-used "API" was plumbing.** Of those 999 import sites, the top entries were
`sha256_file` (65), `atomic_write` (18), `canonical_json_bytes` (18), `sha256_payload` (18). That
is 16% of all traffic, and `forge.core` now absorbs it; `forge.chem` takes another 3%. The
remaining **82% are genuine domain calls spread over ~590 distinct symbols** — which is the real
finding. There is no domain surface, just 590 entry points.

**Modules are vertical slices.** 29 exceed 1,000 lines. `product/defog_feasibility.py` (1,536)
holds RDKit tensorization, a PyTorch training loop, sampling, ECE/Jensen-Shannon/Wasserstein
metrics, graph-to-molecule conversion, a peak-RSS probe, a policy decision, and the artifact
writer — and 41 call sites import from it, so all of that is transitively coupled.

**Types describe nothing.** Annotation coverage is 99.4%, but `dict[str, Any]` appears ~2,073
times and 34.9% of annotated returns are dict-shaped. `NewType`, `TypedDict` and `TypeAlias` are
used zero times outside `core`. You cannot compose against a `dict[str, Any]`.

**Seven package cycles.** `product↔route` 35/5, `route↔value` 16/29, `bio↔product` 8/49,
`data↔route` 2/34, plus three smaller. A package you cannot import without importing back is not
a boundary.

## The four moves, in order

### 1. Give every package an API

Before moving a single file. For each package, the de-facto public surface is whatever other
packages already import; make that explicit in `__init__.py` and mark everything else internal.
This is what makes the later moves safe — once a boundary is declared, breaking it is visible.

The 21 private cross-module imports each name a shared component that never got a home. They get
one, in `core`, `chem`, or the owning domain.

### 2. Model the recurring records

The ~2,073 `dict[str, Any]`s are not 2,073 different shapes. The tracked artifacts under `results/`
are serialized instances of a small number of recurring records — a product row, a component, a
route step, an evidence record — and the `fieldnames` constants modules declare for their CSV
writers are literal schema declarations.

Define the dozen that recur as frozen dataclasses in a domain-types module, with `from_row` /
`to_row` so ledgers keep serializing byte-identically. **Twelve well-chosen types beat forty**; a
shape used once stays a dict.

This is what turns dict-passing into composition, and it is the single highest-leverage change for
"structured better".

### 3. Split the vertical slices

Decompose the largest modules along the seams they already have. The test for whether a concern
deserves its own module is not size — it is whether *other modules import it*. A concern used only
by its own `run_*()` is a long function; a concern others import is a shared component with no home.

`defog_feasibility` is the archetype and the pilot.

### 4. Re-layout into domains

Only now, and it becomes mechanical:

```
core/      artifact, io, hashing, types      DONE
chem/      the RDKit boundary                DONE
corpus/    R0/R1, enumeration, selection, splits
generate/  flow model, sampling, checkpoints, gates
route/     planners, proposals, adjudication, dossiers   (absorbs value/)
potency/   oracle, applicability, morphology
audit/     post-hoc audits and the _vN families
```

Start with `potency/`: of the `bio↔product` cycle, **all 8 outbound and all 46 inbound edges
involve the misfiled `ugi_*` modules and none involve real biology**, so extracting the 30 of them
cuts three of the seven cycles in one move and leaves `bio/` an actual biology package.
Then split `product/` into `corpus/` + `generate/`, then merge `value/` into `route/` — 16/29
edges, one domain that was split by chronology rather than design.

## Mechanism

Moving files is cheap now. A pin binds `{path, sha256}`; moving changes the path, not the bytes, and
`verify_artifact_pins.py` resolves through `docs/artifact_path_moves.json`. Proven both ways: a move
holds at 739 verified / 0 drift, and one tampered byte in a moved file still fails.

One thing genuinely stays put: the 37 modules inside the blinded-execution manifest, which checks its
module set by exact equality to prove which code ran during a sealed holdout. They move last, with
the manifest regenerated as a reviewed act.

Per step: `git mv`, append the rename map, rewrite imports, then gate on **739 pins / 0 drift**, all
modules importing, and the failing-test set staying a subset of `tests/baseline_failures.txt`.

## Corrections to earlier drafts

Two things I proposed that the evidence does not support:

**The `_vN` families are not near-duplicates and must not be merged.** The six
`ugi3_fresh_pool_route_coverage` modules share **zero** top-level function names — 0 of 28. They are
successive analyses that build on each other by import, not versions of one implementation.
Parameterizing them into one function would be inventing a shared abstraction that does not exist.
They move to `audit/` as a group and stay separate.

**Performance work is mostly not warranted.** Three of four measured claims did not survive: caching
canonical SMILES is 1.0x, the SQLite per-row insert is 0.9% of per-product cost, and the fingerprint
persistence win has no consumer. RDKit dominates at 188 µs per outcome versus 1.7 µs for storage.
Only one is real — a redundant re-parse in `ugi_expanded_enumeration`, 1.79x on that stage — and it
is a normalization guard, so it needs stronger proof than "no counterexample in 1,080 outcomes".

## What this does not fix

- **200 tests fail on missing inputs.** 74 modules can be restructured but not verified; `verify-pins`
  and import checks are the real gate there.
- **285 duplicated helpers remain** (`_load_json` 122, `_read_csv` 60, `_pin` 50). Core can absorb
  them now, but migrating changes exception *messages*, so it stays its own pass.
- **`_pin` strictness**: 16 of 50 validate provenance more loosely than the strictest. Migrating will
  surface real failures — findings, not regressions. See `docs/provenance_check_audit.md`.
