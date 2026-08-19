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

### 2. Give the domain types a home — most of them already exist

The ~2,073 `dict[str, Any]`s are not 2,073 shapes, and the fix is smaller than it looks: **the
shared surface of the big modules is already types, not logic.** `ugi_terminal_route_assessment` is
1,381 lines of which **956 are dataclasses**, and 5 of its 7 exports are types, imported by 13
modules. `ugi_nonzero_guidance_runner` is 2,579 lines with **722 lines of dataclasses**, 11 of 15
exports types, imported by 15. That is 1,678 lines of domain records that 28 modules already depend
on, sitting inside modules named after the one analysis that happened to define them.

So this step is mostly relocation, not design. What genuinely needs defining is short:

- **`PinnedInput` `{path, sha256}`** — constructed 506 times across **201 of 286 modules**, an order
  of magnitude more pervasive than anything else. `core.hashing.pin_record` already returns it, and
  modelling it is what lets the 50 divergent `_pin` validators converge on one parse.
- **`RoleName` + `RoleMap`** — done, and it caught a defect: the enum modelled the spelling the
  artifacts barely use.
- **The `SemanticAtom`/`Bond`/`ComponentMapping`/`Product` family** — one owning module, explicit
  field tuples, identical across three corpora. The safest pilot for `from_row`/`to_row`.
- **`OracleMetricRow` / `OraclePredictionRow`** — declared 3× each with a clean shared core; the
  prediction split (point vs ensemble) is honest subtyping.
- **`ArtifactRef`** — a *produced* artifact, distinct from a consumed input.

Leave the rest as dicts. Of 71 declared CSV schemas only 6 cluster at all; inventing a hierarchy
over genuinely diverse ledger formats is the speculative abstraction the contract warns against.

`from_row`/`to_row` must round-trip byte-identically — 63 tracked ledgers are hash-pinned.

### 3. Extract the homeless shared components — and leave the giants alone

The test is whether *other modules import it*, and applying it inverts the obvious ordering.
**Seven of the fifteen largest non-frozen modules export nothing at all** — 7,377 lines across
`decomposition_precision_audit` (2,096), `ugi3_aldehyde_head_capability` (1,459),
`hydrophobic_motif_transfer` (1,381), `oracle_graph_transfer` (1,282), `route_awareness` (1,185)
and two more. Each is one script's implementation. Decomposing them is internal tidying with no API
benefit and no blast radius, so they rank **last** despite being the biggest.

What matters is the opposite: small functions that many modules reach into privately. There are
**147 such symbols across 44 modules**. Ranked:

1. **`_rstar_step`** — 60 lines, *"one Euler step using DeFoG's minimum R-star conditional rate"*,
   the core sampling primitive of the flow model. Reached privately by **9 modules**, and it lives
   inside `defog_feasibility` — a *completed M0-06 feasibility probe*. Production sampling depends
   on a finished experiment. `defog_feasibility` is frozen, so this is extract-and-shim: the new
   home becomes canonical and the frozen copy stays as legacy.
2. **The two type vocabularies** above — 1,678 lines, 28 importers, need a home not a redesign.
3. **`_aldehyde_program` / `_isocyanide_program`** in `ugi3_virtual_programs`, 5 importers each — a
   clean role-program seam.

Also formalise the stage contract rather than invent one. 206 of 291 modules have an entry point,
and 181 of them return `(result_document, *artifact_bytes)` positionally — that **is** the artifact
contract, written as a tuple instead of a type. `core.artifact.ArtifactRun` already models it, so
adopting it is a rename of something that exists.

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

The `potency/` extraction is complete: 49 oracle, applicability, morphology, ranking, and authorized
diagnostic modules moved out of `bio/`. The `bio/` package now contains only its package initializer,
generic endpoint interface, frozen endpoint decision, and the liver/muscle/vaccine implementations.
An executable boundary test requires every declared move target, rejects imports of every removed
module, and fixes that six-file package surface. Frozen configs retain their historical `bio` paths;
the provenance move map resolves those identities to the new bytes.

The remaining `product -> potency` references are real transitional coupling, chiefly shared role
constants and evaluation adapters. They are not hidden by the namespace move. Split `product/` into
`corpus/` + `generate/` before tightening that boundary, then merge `value/` into `route/` — one
domain that was split by chronology rather than design.

## Mechanism

Moving files is cheap now. A pin binds `{path, sha256}`; moving changes the path, not the bytes, and
`forge provenance verify` resolves through `docs/artifact_path_moves.json`. Proven both ways: a move
holds at 742 verified / 0 drift, and one tampered byte in a moved file still fails.

One thing genuinely stays put: the 37 modules inside the blinded-execution manifest, which checks its
module set by exact equality to prove which code ran during a sealed holdout. They move last, with
the manifest regenerated as a reviewed act.

Per step: `git mv`, append the rename map, rewrite imports, then gate on **742 pins / 0 drift**, all
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
