# Restructure plan — move fast, keep provenance

## The unlock

I previously treated 170 files as unmovable. That was wrong, and it was slowing everything down.

A pin records `{path, sha256}`. Moving a file changes the path; **it does not change the bytes**.
The only thing that breaks is the verifier's ability to find the file. So the fix is a rename map,
not a freeze:

```
docs/artifact_path_moves.json    old/path.py -> new/path.py
```

`verify_artifact_pins.py` resolves through it. Content is still checked against the recorded digest,
so the evidence chain is fully intact — a moved file whose bytes changed still fails, which is the
property that matters. Verified pin count must stay at 739 throughout.

**What genuinely stays put:** one thing, not 170. `ugi3_route_saturation_blinded_execution` declares
`RUNTIME_DEPENDENCY_MODULES` and checks it by exact set equality to prove which code ran during a
sealed holdout. Renaming a module in that graph changes the declared set. That is 37 modules, and the
answer is to move them last and regenerate the manifest as an explicit, reviewed act — not to freeze
the other 133 because of it.

## Target

```
src/forge/
  core/      artifact, io, hashing, config, types        DONE
  chem/      the RDKit boundary                          DONE
  corpus/    R0/R1, enumeration, selection, splits
  generate/  flow model, sampling, checkpoints, gates
  route/     planners, proposals, adjudication, dossiers
  potency/   oracle, applicability, morphology
  audit/     frozen post-hoc audits and the _vN families
  legacy/    only what the sealed manifest pins in place
```

## Order, and why

**1. `potency/` — extract 30 misfiled modules from `bio/`.** Highest structural payoff per unit of
risk, and the data is unambiguous: of the `bio ↔ product` cycle, **all 8 outbound and all 46 inbound
edges involve the misfiled `ugi_*` modules and none involve real biology**. The same holds for
`bio ↔ value` and `bio ↔ route`. This one move cuts **three of the seven package cycles** and leaves
`bio/` as an actual biology package (22 modules: liver, muscle, vaccine, endpoint, oracles).

**2. `corpus/` and `generate/` — split `product/`.** 121 modules doing two unrelated jobs: building
the training corpus, and running the model. Cuts `product ↔ route` (35/5) and `product ↔ value` (7/2).

**3. `route/` absorbs `value/`.** The `route ↔ value` cycle is 16/29 edges — the densest in the
codebase. They are one domain that was split by chronology, not by design; `value/` is route-value
scoring. Merging removes the cycle by construction rather than by untangling it.

**4. `audit/`** takes the post-hoc audits and the `_vN` families, including
`ugi3_fresh_pool_route_coverage` ×6, which stop polluting the domain packages.

**5. Fold the strays.** `eval/` and `verify/` have one real module each; `dossier/` is empty.

**6. The sealed 37, last**, with the manifest regenerated deliberately.

## Method per step

Each step is one commit, and the loop is fast:

1. `git mv` the modules. Bytes never change, so content pins stay valid.
2. Append to `docs/artifact_path_moves.json`.
3. Rewrite imports across the tree — mechanical, `from forge.bio.ugi_x` → `from forge.potency.ugi_x`.
4. Leave a re-export shim at any old path a **sealed** module imports. Nothing else needs one.
5. Gates: `verify-pins` at 739/0 drift; all modules import; failing-test set stays a subset of
   `tests/baseline_failures.txt`.

Package `__init__.py` files get a real docstring and an explicit export surface — today all eight are
the same stub, which is why no package has an API.

## What this does not fix, and should not pretend to

- **200 tests already fail** on missing inputs, so 74 modules can be restructured but not verified.
  `verify-pins` plus import checks are the real gate there.
- **The `_vN` families stay.** All 25 are referenced and back frozen artifacts. They move to `audit/`;
  they do not get deleted.
- **285 duplicated helpers remain** (`_load_json` 122, `_read_csv` 60, `_pin` 50). The core capability
  to absorb them now exists; migrating changes exception *messages*, so it stays a separate pass.
- **`_pin` strictness**: 16 of 50 validate provenance more loosely than the strictest. Migrating them
  will surface real failures. That is a finding, not a regression — see `docs/provenance_check_audit.md`.
